//! Friendly stable `fit` entry point for the `glmm` crate.
//!
//! Owns all scratch; dispatches on `ModelSpec::estimator`; returns `Fit`.
//! This is the additive stable public surface —
//! it never touches any kernel; only marshals data and scratch into kernel
//! calls, then copies results out.
//!
//! # Calling convention
//!
//! `x` is a design matrix in **row-major f64** layout: element `(i, j)` is at
//! `x[i * p + j]`. `y` is the response vector of length `n`. The kernels use
//! faer column-major f64 internally; conversion is done here.
//!
//! Cluster ids and extra-grouping ids are derived from `ModelSpec` using the
//! same row-layout rules as MCPower's data-gen layer (mirrors the DGP's
//! `cluster_of_row` / `extra_level_of_row` helpers).
//!
//! # Module layout
//!
//! `fit_warm` dispatches on `(family, re.is_some())` into one estimator
//! module per family — `ols`/`lmm` (Gaussian), `glm` (fixed-effects
//! binomial/Poisson/Gamma/NB), `glmm` (mixed non-Gaussian, plus GLMM-NB) —
//! each of which marshals into the matching numerical kernel
//! (`src/ols.rs`/`src/lmm/mod.rs`/`src/glm.rs`/`src/glmm/`). `common` holds
//! helpers shared by 2+ of those modules; `loop_advanced_seam` holds the
//! unstable dev-only surface re-exported through `crate::loop_advanced`.

use crate::consts::{MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q, MAX_PRIMARY_Q};
use crate::{GroupIds, GroupingRelation, ModelSpec, StartValues, WaldSe};

mod common;
mod core;
mod glm;
mod glmm;
mod lmm;
mod loop_advanced_seam;
mod ols;

use common::{
    assert_group_ids, assert_model_shape, detect_aliased, fit_rank_deficient, spec_sized_from_ids,
    theta_width,
};
// The grouping reorder `spec_sized_from_ids` may apply. Always `pub` (it is a
// `build_workspace` parameter and one of `spec_sized_from_ids_pub`'s returns)
// but the `fit` module itself is private, so the stable crate surface is
// unaffected; `crate::loop_advanced` is what actually publishes it.
pub use common::Perm;

/// Result of `fit`. Fixed-effect estimates cover all p predictors; SE and
/// tau2 have the ranges below. Non-target SE slots are NaN.
///
/// `#[non_exhaustive]`: construct one only through [`fit_cold`]/[`fit_warm`],
/// and match on it with a trailing `..`. Everything the fit reports about
/// ITSELF — as opposed to about the data — lives on [`Fit::diagnostics`], which
/// is `#[non_exhaustive]` for the same reason: a seventh diagnostic is then an
/// additive change instead of a major version.
#[non_exhaustive]
pub struct Fit {
    /// Fixed-effect estimates, length p.
    pub beta: Vec<f64>,
    /// Standard errors: `se[j] = sqrt(Var(β̂_j))` for target predictors,
    /// NaN for non-targets. Length p.
    pub se: Vec<f64>,
    /// Fixed-effect covariance `Cov(β̂)` — a full symmetric **p×p** matrix
    /// (`vcov[i][j]`), NOT vech-packed: unlike `varcorr`, which packs one
    /// variable-sized block per grouping, this is a single square block and
    /// every consumer (R's `vcov()`/`confint`, `multcomp::glht`, any hand-built
    /// Wald contrast) indexes it directly. `se` is its diagonal:
    /// `se[j] == vcov[j][j].sqrt()` wherever both are finite.
    ///
    /// Finite exactly where `se` is — the off-diagonal `vcov[i][j]` is finite
    /// iff `se[i]` and `se[j]` both are. So a [`FitOptions::target_indices`]
    /// subset leaves everything outside the target block NaN (there is no
    /// covariance to report for a coefficient whose variance was never
    /// computed), and a non-converged fit is all-NaN.
    ///
    /// Sources, by path: OLS/GLM/LMM invert the same Cholesky factor `se`'s
    /// forward solve already walks; GLMM `WaldSe::Hessian` takes the β block of
    /// the joint (θ,β) Hessian covariance (exact on every shape
    /// `derivative::supports_shape` accepts, finite-difference elsewhere),
    /// and `WaldSe::Rx` the p×p Schur
    /// inverse — both already formed in full and previously discarded down to a
    /// diagonal. Unlike `stddev_se`, this is populated on the Hessian's RX
    /// fallback too (that fallback inverts a full p×p covariance; only a
    /// double failure, where the Hessian AND the fallback both fail, NaN-fills
    /// — as a non-converged fit).
    pub vcov: Vec<Vec<f64>>,
    /// Per-element Cholesky-scaled values `theta[k]^2 * sigma_sq`. These equal
    /// the random-effect variance components only for diagonal/scalar RE
    /// components (q=1 / scalar-extra — the currently reachable case); slope
    /// (q≥2) models are not yet validated through this field. Empty for OLS.
    pub tau2: Vec<f64>,
    /// Estimated dispersion: `φ` for Gamma (Pearson moment estimator), the
    /// estimated shape `θ` for negative-binomial, the residual variance `σ̂²`
    /// for Gaussian — `RSS/(n−p)` for OLS (raw-row df, matching R
    /// `summary.lm`'s `sigma²`; the same `sigma_sq` that scales `se`/`vcov`)
    /// and the REML `pwrss/(n−p)` for LMM (matching lme4 `sigma()²`; oracle:
    /// `validation/goldens/sleepstudy_lmm.json` `sigma`, asserted in
    /// `fit_sleepstudy_slope_varcorr_matches_lme4`) — and `1.0` for
    /// binomial/Poisson (where dispersion is fixed, not estimated). NaN on a
    /// Gaussian fit with no honest endpoint (non-converged OLS, degenerate
    /// LMM).
    pub dispersion: f64,
    /// Everything the fit reports about itself: convergence, singularity, the
    /// aliased-column mask, the θ boundary state, which variance components
    /// were pinned there, and any [`Note`]s the solver raised. Single storage
    /// location — [`Fit::converged`], [`Fit::singular`] and [`Fit::aliased`]
    /// are one-hop forwarders onto it, not copies.
    pub diagnostics: Diagnostics,
    /// RE (co)variance per grouping: one **vech-packed
    /// lower-triangular** covariance block `D̂ = σ̂²·Λ̂Λ̂'` per grouping, in
    /// declaration order (primary, then each extra). σ̂² is the residual scale
    /// for an LMM and the free GLMM scale `pwrss/n` for dispersion families
    /// (Gamma) — exactly the factor lme4's `VarCorr` stddevs carry; it is ≡ 1
    /// for binomial/Poisson/NB, and the same scale `tau2` reports, so the two
    /// accessors agree. Vech order is column-major
    /// lower-triangular (matching the θ vech convention): for a `q×q` block,
    /// `(0,0),(1,0),…,(q-1,0),(1,1),…,(q-1,q-1)`. Validated against lme4
    /// `VarCorr` (`validation/goldens/sleepstudy_lmm.json`; Gamma scale:
    /// `validation/goldens/sim_gamma_glmm.json`). Empty for OLS/GLM
    /// (no random effects). This is the q≥2-valid replacement for `tau2`'s
    /// per-component variances; `tau2` is retained for back-compat.
    pub varcorr: Vec<Vec<f64>>,
    /// SE of each RE standard deviation, laid out like `tau2` (per θ coordinate,
    /// length `n_theta`; primary block then each extra, in declaration order).
    /// Populated ONLY on a converged GLMM `WaldSe::Hessian` fit, from the θ block
    /// of the joint (θ,β) Hessian covariance that `joint_hessian_cov` inverts and
    /// otherwise discards (exact Hessian on every shape
    /// `derivative::supports_shape` accepts, finite-difference elsewhere).
    /// NaN under `WaldSe::Rx`, on the Hessian RX fallback, and
    /// for OLS/LMM (no Hessian machinery). Correct for SCALAR groupings only (the
    /// reachable GLMM case), where the RE stddev equals its θ so the θ-scale SE is
    /// the stddev SE directly; a q≥2 block would need a delta-method Jacobian.
    ///
    /// Scale caveat for dispersion families (Gamma): this SE stays on the
    /// **θ scale** — lme4's θ-Hessian convention — while `varcorr`/`tau2` carry
    /// the σ̂² factor. Deliberately NOT multiplied by σ̂: the joint Hessian does
    /// not carry cov(σ̂, θ̂), so a σ̂-rescaled SE would be a delta-method value
    /// that matches no oracle. For the φ≡1 families the two scales coincide and
    /// there is no split. (The validation suite skips `sd_se` gating for
    /// dispersion families accordingly.)
    pub stddev_se: Vec<f64>,
    /// Objective evaluations consumed by the θ (LMM) / joint [θ|β] (GLMM)
    /// BOBYQA search — GLMM counts both stages. 0 where no derivative-free
    /// optimizer runs (OLS/GLM closed-form or IRLS paths). Deterministic and
    /// clock-independent; the optimizer-grid campaign's primary metric.
    pub n_eval: usize,
    /// Dev-only optimizer evaluation counters — the stage split, the shrink
    /// phase, the PIRLS-iteration histogram and the AGQ node cost. Present
    /// only under the off-by-default `counters` feature, which is NOT
    /// semver-covered; zeros on every route that runs no derivative-free
    /// search (OLS, GLM).
    #[cfg(feature = "counters")]
    pub counters: crate::counters::EvalCounters,
    /// Minimized optimizer criterion at the accepted point. LMM: the profiled
    /// REML deviance as computed by `reml_deviance` — equals lme4's REMLcrit
    /// minus the data-independent constant df·(1 + ln 2π), df = n − p
    /// (validated against the frozen lme4 sleepstudy reference). GLMM: the
    /// marginal Laplace deviance `d(y,ũ) + ‖ũ‖² + log|A|` (see
    /// `GlmmFit::deviance`), which differs from −2·logLik by a data-only
    /// saturated constant. NaN for OLS/GLM and on optimizer/numerical
    /// failure (GLMM non-convergence surfaces as +∞ internally; mapped to
    /// NaN here). An LMM fit that hits `MaxFunReached` still reports the
    /// finite endpoint deviance here with `converged == false` — the plateau
    /// policy: a `MaxFunReached` cap-out reports its finite endpoint with
    /// `converged == false` rather than NaN-filling.
    pub deviance: f64,
    /// The actual log-likelihood at the fitted parameters — `deviance` with its
    /// dropped data-only constants restored, on the `logLik()` scale (R/lme4):
    ///
    /// - **OLS/GLM** — the standard closed forms (R `logLik.lm`/`logLik.glm`,
    ///   `MASS::glm.nb`), including under prior weights.
    /// - **LMM** — the **REML criterion** `−REMLcrit/2` (this path is
    ///   REML-only): `−½(deviance + (n−p)·(1 + ln 2π))`. REML criteria are
    ///   comparable only between models with IDENTICAL fixed effects — an
    ///   AIC/LRT across different fixed parts is meaningless; check [`Fit::reml`]
    ///   before comparing. Matches `lme4::logLik` on a `REML=TRUE` fit.
    /// - **GLMM** — the marginal Laplace/AGQ log-likelihood:
    ///   `−½·deviance + saturated_loglik` (binomial/Poisson/NB, see
    ///   `family::saturated_loglik`), `−½·deviance` for Gamma (lme4's
    ///   `logLik(glmer)` is `−devfun/2` verbatim, `gamma_aic`'s `+2`
    ///   included — see `fit::common::glmm_loglik`). Matches `lme4::logLik`
    ///   on the same fit, including the aggregated-binomial `cbind(s, m−s)`
    ///   form under `weights=`.
    ///
    /// NaN wherever `deviance`'s failure modes apply (non-converged/degenerate
    /// fits); finite on an LMM `MaxFunReached` endpoint, like `deviance`.
    /// `AIC = 2·df − 2·loglik`, `BIC = df·ln(n) − 2·loglik` with [`Fit::df`].
    pub loglik: f64,
    /// Parameters counted for AIC/BIC: retained fixed effects (`p` minus
    /// aliased columns, matching lme4's NA-coefficient handling) + `n_theta`
    /// RE parameters + 1 if the family estimates a dispersion/scale (Gaussian
    /// σ², Gamma φ unless held fixed via [`FitOptions::dispersion`], NB θ).
    /// 0 on degenerate NaN-fill paths.
    pub df: usize,
    /// `true` iff `loglik` is a REML criterion (the Gaussian LMM paths — REML
    /// is this engine's locked LMM objective) rather than an ML log-likelihood.
    /// Model comparisons (AIC/LRT) across fits with different fixed effects are
    /// invalid when this is set — mirror of lme4's REML-fit `anova` warning.
    pub reml: bool,
    /// Fitted means μ̂ per row (length `n`): the conditional means through the
    /// inverse link — `g⁻¹(Xβ̂ + Zb̂)` for mixed fits (lme4 `fitted()`),
    /// `g⁻¹(Xβ̂)` for fixed-only. Empty on non-converged fits. Always on the
    /// original `y` scale, including on the Gaussian LMM paths — see
    /// `fit::common::lmm_fitted` for how the offset is restored there.
    pub fitted: Vec<f64>,
    /// Random-effect conditional modes `b̂ = Λ̂û` on the natural (link) scale —
    /// lme4's `ranef()` values. One block per grouping in declaration order
    /// (primary, then each extra — same order as `varcorr`), each block
    /// level-major: level `l`'s `q` values (intercept, then slopes in
    /// declaration order) at `[l·q .. (l+1)·q]`. `η̂ = Xβ̂ + Zb̂` reproduces the
    /// linear predictor behind `fitted`. Level counts per grouping are in
    /// [`Fit::ranef_levels`]; `q` per grouping is recoverable from `varcorr`
    /// (vech length). For a nested grouping the block spans
    /// `n_parents·n_per_parent` slots (child id = parent·n_per_parent +
    /// within), padded with zero modes for parents with fewer observed
    /// children. Empty on non-converged fits.
    ///
    /// This is the numbers only. To get them LABELLED — which level each row
    /// belongs to, with the padded slots dropped — use
    /// [`crate::formula::label_ranef`], the one place the block layout is
    /// interpreted. Do not re-derive the slicing in a consumer: which layout a
    /// grouping lands in is a data-dependent speed decision, so a consumer that
    /// inferred it would be wrong on a dataset nobody tested.
    ///
    /// The Gaussian LMM paths never form these during the fit — the profiled
    /// REML criterion does not need them — and recover them once at θ̂ by
    /// back-substitution afterwards, which is why they are here without the
    /// per-evaluation cost of forming them.
    pub ranef: Vec<f64>,
    /// Level count per grouping for slicing [`Fit::ranef`], declaration order
    /// (primary, then each extra). `ranef.len() = Σ_g levels[g]·q_g`. Empty
    /// exactly when `ranef` is.
    pub ranef_levels: Vec<usize>,
}

/// Everything a fit reports about itself, reached as `fit.diagnostics`.
///
/// **Coverage is not uniform across routes**, and the per-field docs say where
/// each one is real. The short version: `converged` and `aliased` are filled
/// everywhere; `pinned` is real on every route that has variance components to
/// pin, while `boundary` distinguishes all three states only on the dense LMM
/// and dense GLMM routes; `notes` can only ever be raised by OLS, GLM and dense
/// LMM — dense GLMM forms no factor to measure a pivot on, and the sparse route
/// refuses an ill-conditioned design outright (reporting `converged: false`)
/// rather than fitting it and flagging. An absent note is therefore "not
/// detected", never "checked and clean".
///
/// `#[non_exhaustive]`: match with a trailing `..`. That is the whole point of
/// collecting these here — the six channels below arrived one at a time, each
/// arrival breaking `Fit`, and a seventh now costs nobody a major version.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Diagnostics {
    /// Whether the optimizer reached its convergence criterion. `false` means
    /// `se`/`vcov`/`dispersion` are the NaN-fill described on each of those
    /// fields.
    pub converged: bool,
    /// `true` iff the fit converged onto the θ boundary (≥ 1 diagonal variance
    /// component pinned at 0 — `boundary == AtBoundary`, OR a converged
    /// diagonal stddev negligible next to its fit's largest — see
    /// [`Fit::has_negligible_component`]), the same condition lme4's
    /// `isSingular` reports. `false` for OLS/GLM and for an LMM
    /// `MaxFunReached` cap-out — a capped endpoint is reported as a point,
    /// not accepted onto the boundary, so it never sets this flag even when
    /// its diagonals are near zero.
    ///
    /// Not a pure restatement of `boundary`: the negligible-component check is
    /// a reporting rule applied to the assembled `varcorr` after the fact, so
    /// `singular` can be `true` with `boundary == Interior`.
    pub singular: bool,
    /// Rank-deficiency mask, length `p`: `true` for a fixed-effect
    /// column dropped because it is aliased (linearly dependent) on an earlier
    /// column, mirroring lme4's `NA`-coefficient behavior. The corresponding
    /// `beta`/`se` slots are `NaN` and `converged` stays `true` (the reduced
    /// model fits). All-`false` when the design is full-rank.
    ///
    /// Decided by the alias gate that runs BEFORE dispatch, not by any fitting
    /// route — a column merely hard to identify is fitted and flagged with a
    /// [`Note::IllConditioned`], not aliased.
    pub aliased: Vec<bool>,
    /// Where in the θ parameter space the accepted point sits.
    pub boundary: Boundary,
    /// Which variance components were pinned to the boundary: per grouping, per
    /// dimension, ALIGNED WITH THE `varcorr` BLOCKS, so `pinned[g][i]` pairs
    /// with `stddev_corr(g).0[i]`. That alignment is the point — it is the
    /// layout both wrapper packages already iterate.
    ///
    /// **Empty means nothing was pinned** — every route with variance
    /// components to pin fills this field on every converged fit; the
    /// dense-view mappers read it straight off the fitted mask
    /// (`common::materialize_diagnostics`) and the two sparse routes name
    /// their own collapsed components explicitly (`pinned_flags`), so there is
    /// no route left that knows a component pinned but declines to say which.
    /// A fit with no pinned component leaves this empty rather than allocating
    /// a grid of `false`: the kernels carry the state as a u64 bit mask and
    /// `pinned_flags` short-circuits to `vec![]` when the mask is zero, so a
    /// warm loop over draws that never pin pays no heap block per draw for
    /// saying nothing happened. Any route with no variance components at all
    /// leaves it empty too (OLS, GLM, and the fixed-effect-only negative
    /// binomial). The one structural exception: a design past the 64
    /// components the internal mask holds — outside the crate's currently
    /// validated envelope — cannot be represented here at all.
    pub pinned: Vec<Vec<bool>>,
    /// Variance-component score at each PINNED component, laid out exactly like
    /// [`Diagnostics::pinned`]: `boundary_score[g][i]` pairs with `pinned[g][i]`
    /// and with `stddev_corr(g).0[i]`. The value is `dD/ds` at `s = 0` in the
    /// variance coordinate `s = θ_jj²` — equivalently `½·∂²D/∂θ_jj²` at the
    /// pinned point, but only where the deviance is even in `θ_jj`. That holds
    /// iff Λ's column `j` has no non-zero entry below the diagonal: with one,
    /// `Σ_kj` for `k > j` carries the term `Λ_kj·Λ_jj`, which is linear (not
    /// even) in `θ_jj`, so the shortcut does not apply.
    /// [`crate::lmm::LmmGroupings::diagonal_has_nonzero_below`] is the gate on this.
    ///
    /// **Positive means the boundary is the constrained optimum**: raising the
    /// component off zero would raise the deviance. A non-positive score at a
    /// pinned component means the pin is not justified by the local geometry.
    /// NaN at every component that is not pinned, at every off-diagonal, and
    /// at a pinned diagonal whose column carries a live off-diagonal below it
    /// — so NaN does not mean "not pinned".
    ///
    /// **Empty means no score was measured** — a fit that did not ask for it
    /// ([`FitOptions::boundary_score`], off by default), an interior fit, a
    /// non-converged fit, every route other than the dense LMM and the blocked
    /// GLMM, and the GLMM shapes with no exact Hessian (structured extras,
    /// dense fallback, sparse). Empty is NOT "nothing was pinned"; read
    /// [`Diagnostics::pinned`] for that. Observation only.
    pub boundary_score: Vec<Vec<f64>>,
    /// KKT residual at the accepted θ̂: the ∞-norm of the deviance's θ gradient
    /// projected onto the box the optimizer searched (diagonals `[0, 1e3]`,
    /// off-diagonals `[±1e3]`) — at a lower bound a positive component is
    /// satisfied and contributes nothing, at an upper bound a negative one
    /// does. Zero to working precision means the accepted point satisfies the
    /// first-order conditions, boundary or interior; a large value means the
    /// optimizer stopped somewhere that is not a constrained stationary point.
    ///
    /// **Coordinates:** on the deviance scale (−2·logL), per unit of θ in the
    /// units the caller's design is in — the projection runs in the internal
    /// scaled θ̃ where the box lives and each component is mapped back by its
    /// row scale before the norm. A derivative w.r.t. θ multiplies by that
    /// scale where [`Fit::stddev_se`], an SE in θ, divides by it.
    ///
    /// The optimizer stops on a trust radius, not on a gradient, so a converged
    /// fit leaves a small finite residual rather than an exact zero; what
    /// counts as small is a measured number, pinned by the calibration in
    /// `src/glmm/tests.rs`. **NaN** wherever no exact gradient exists: every
    /// non-GLMM route, the GLMM structured-extras and dense-fallback shapes,
    /// the sparse routes, and any non-converged fit. Observation only — no
    /// fitting decision reads it.
    pub kkt_grad_norm: f64,
    /// Solver observations that are not one of the fixed channels above. Empty
    /// on a clean fit, and empty allocates nothing.
    pub notes: Vec<Note>,
}

/// Where the accepted θ sits in its parameter space.
///
/// Only the dense LMM and dense GLMM routes distinguish all three. On the
/// sparse routes this is back-derived from `singular`, so `NoOptimum` is
/// unreachable there and `Interior` means "not pinned", not "verified
/// interior" — `pinned` on those routes is nonetheless exact. OLS and GLM have
/// no θ and always report `Interior`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Every variance component settled strictly inside its range.
    Interior,
    /// ≥ 1 diagonal variance component accepted at 0.
    AtBoundary,
    /// The optimizer capped out at a finite endpoint — reported as a point, not
    /// accepted onto the boundary. `deviance` still carries that endpoint.
    NoOptimum,
}

/// A solver observation about the fit that has no dedicated field. The enum
/// variant, not any English sentence, is the stable identifier a caller filters
/// on; the wrappers turn each variant into their own warning category (Python)
/// or condition class (R).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Note {
    /// Fitted, but these columns are not separately identified: the design is
    /// computable and the estimates are honest, while the standard errors are
    /// large because the data genuinely cannot separate the columns. Not a
    /// failure and not a dropped column — contrast `aliased`, which is the
    /// genuinely-redundant case.
    ///
    /// `columns` carries the single column the pivot search named. Its
    /// entangled partners are NOT currently identified: the measurement finds
    /// the worst-conditioned column, not the set it is confounded with, and
    /// inventing a partner rule would be guessing. `pivot` is the measured
    /// scale-invariant ratio at that column, so a caller can rank severity
    /// across fits instead of treating the flag as binary.
    IllConditioned {
        /// Design-matrix column indices, into the `x` the caller passed.
        columns: Vec<u32>,
        /// The measured scale-invariant pivot ratio — smaller is worse.
        pivot: f64,
    },
    /// A GLMM PIRLS inner solve ran the full `PIRLS_MAX_ITERS` (50) cap without
    /// satisfying its convergence band — never a failure surfaced any other way
    /// (that is the `(NaN, NaN, NaN, false)` halving-exhaustion/Cholesky-failure
    /// case, which the fit already rejects or NaN-fills through the usual
    /// channels). Observation-only: the cap stays 50 and no fitted number moves.
    /// FD-Hessian SE evals are excluded — only fit-path evals count.
    PirlsExhausted {
        /// How many fit-path BOBYQA objective evals hit the cap over the course
        /// of this fit (0 if `final_eval` alone is what fired).
        evals: u32,
        /// Whether the FINAL re-evaluation at the converged γ̂ itself hit the
        /// cap — the case that matters, since that solve's ũ/W̃ feed the
        /// reported estimates directly rather than being a rejected trial point.
        final_eval: bool,
    },
    /// A grouping factor declares levels that carry no row but still occupy
    /// random-effect columns, because the block is `max(code)+1` wide: a level
    /// between two observed ones is an empty cluster that contributes nothing to
    /// the likelihood and costs width anyway. Its conditional mode is reported,
    /// fully shrunk to zero — lme4's `ranef()` has no counterpart row at all, so
    /// this is a deliberate divergence, and dropping the level (R's
    /// `droplevels`) removes both the row and the wasted width. A level declared
    /// AFTER the last observed one costs nothing and is never named here.
    ///
    /// Raised by the formula frontend, not by a solver: it is the only layer
    /// that sees both the declared levels and the per-row codes. Reached through
    /// [`crate::formula::Lowered::notes`].
    UnusedGroupingLevels {
        /// The grouping factor, as the formula spells it.
        grouping: String,
        /// The declared level labels with a slot but no row.
        levels: Vec<String>,
    },
    /// One grouping's random-effect design columns sit on very different
    /// scales — the ratio of the largest to the smallest column RMS (the
    /// implicit intercept counted at 1.0) exceeds the formula frontend's
    /// `RE_SCALE_SPREAD_WARN` threshold (1e3). Fitting is unaffected (the
    /// kernel scales internally), but the reported random-effect standard
    /// deviations sit on the raw variable's scale, so a large spread makes
    /// them hard to compare by eye. Mirrors lme4's
    /// `lme4:::checkScaleX(tol = 1000)`.
    ///
    /// Raised by the formula frontend, not by a solver, for the same reason
    /// [`Note::UnusedGroupingLevels`] is: the ratio is measured over the
    /// lowered design, and a caller building `x`/`ModelSpec` by hand never
    /// sees it. Reached through [`crate::formula::Lowered::notes`].
    ReDesignScaleSpread {
        /// The grouping factor, as the formula spells it.
        grouping: String,
        /// max column RMS / min column RMS over the grouping's random-effect
        /// design columns (intercept included at 1.0).
        ratio: f64,
    },
    /// `FitOptions::wald_se == WaldSe::Hessian` was requested, but the
    /// finite-difference joint Hessian came back non-positive-definite (or a
    /// perturbed deviance evaluation was non-finite), so the SE pass fell back
    /// to the RX/Schur route instead. `Fit::se`/`vcov` are still filled (from
    /// the fallback), but `Fit::stddev_se` is NaN — no θ-block SE comes out of
    /// a Schur inverse. Never fires under `WaldSe::Rx`, which never attempts
    /// the joint Hessian.
    HessianSeFallback,
}

impl Diagnostics {
    /// The diagnostics a route that reports no θ-boundary detail can honestly
    /// fill: `boundary` back-derived from `singular`, nothing pinned, no notes,
    /// no aliased column. Every direct-`Fit`-building site (the NaN-fill
    /// returns, and the sparse routes, which assemble a `Fit` rather than a
    /// view) goes through here; the four view mappers go through
    /// `common::materialize_diagnostics` instead, which has a carrier to read.
    ///
    /// The two sparse routes overwrite `pinned` on top of this with
    /// [`pinned_flags`] — they do know which components collapsed. `pinned`
    /// stays empty here because the NaN-fill returns share this helper and have
    /// no varcorr to align a grid against.
    pub(crate) fn from_flags(converged: bool, singular: bool, p: usize) -> Self {
        Diagnostics {
            converged,
            singular,
            aliased: vec![false; p],
            boundary: if singular {
                Boundary::AtBoundary
            } else {
                Boundary::Interior
            },
            pinned: vec![],
            boundary_score: vec![],
            kkt_grad_norm: f64::NAN,
            notes: vec![],
        }
    }
}

/// `q` from a vech length: `len == q(q+1)/2` inverted by the quadratic formula.
/// Single source for the varcorr-block width — [`Fit::stddev_corr`] and the
/// `pinned` reshape in `common::materialize_diagnostics` must agree on it, and
/// that agreement is what makes `pinned[g][i]` pair with `stddev_corr(g).0[i]`.
pub(crate) fn vech_q(len: usize) -> usize {
    (((1 + 8 * len) as f64).sqrt() as usize - 1) / 2
}

/// Relative tolerance for [`Fit::has_negligible_component`]: a converged
/// diagonal stddev at or below this fraction of its fit's largest is reported
/// singular even when the optimizer's own stopping point (governed by
/// `PIN_THETA`, an absolute 1e-4 threshold on the internal θ scale — left
/// untouched, this is a reporting-only check) landed just short of an exact
/// pin. Measured on `pois_cross4_g3000p20_bal_nearzero`: glmm converges to
/// stddev 2.5e-4 against a ~0.44-scale sibling component (ratio ~5.7e-4)
/// while lme4's own optimizer pins the same component to exactly 0 and flags
/// `isSingular`; 1e-3 catches this and any tighter true pin with margin.
const SINGULAR_REL_TOL: f64 = 1e-3;

impl Fit {
    /// Whether the optimizer reached its convergence criterion — the most-read
    /// field in the API, forwarded so that moving it behind `diagnostics` costs
    /// its callers a `()` and nothing else. See [`Diagnostics::converged`].
    pub fn converged(&self) -> bool {
        self.diagnostics.converged
    }

    /// Whether the fit is singular (lme4's `isSingular`). Forwarder — see
    /// [`Diagnostics::singular`].
    pub fn singular(&self) -> bool {
        self.diagnostics.singular
    }

    /// The length-`p` rank-deficiency mask. Forwarder — see
    /// [`Diagnostics::aliased`].
    pub fn aliased(&self) -> &[bool] {
        &self.diagnostics.aliased
    }

    /// Reduce `varcorr[group_idx]` — a vech-packed (column-major lower-triangular,
    /// see [`Fit::varcorr`]) `q×q` covariance block — into per-dimension standard
    /// deviations and a full symmetric `q×q` correlation matrix, mirroring lme4's
    /// `VarCorr` stddev/corr split. `q` is recovered from `vech.len() = q(q+1)/2`
    /// via the quadratic formula. Column `c`'s entries start at vech offset
    /// `c*q - c*(c-1)/2` (the same running cursor `varcorr_block` builds — for
    /// `r >= c`, `idx(r,c) = c*q - c*(c-1)/2 + (r-c)`); change together.
    pub fn stddev_corr(&self, group_idx: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let vech = &self.varcorr[group_idx];
        let len = vech.len();
        let q = vech_q(len);
        debug_assert_eq!(
            q * (q + 1) / 2,
            len,
            "varcorr[{group_idx}] is not a valid vech"
        );

        // `c*(c-1)/2` written as `(c*c - c)/2` to avoid a `c - 1` underflow at
        // c=0 in usize arithmetic (c*c >= c holds for all c >= 0).
        let idx = |r: usize, c: usize| -> usize { c * q - (c * c - c) / 2 + (r - c) };
        let stddev: Vec<f64> = (0..q).map(|i| vech[idx(i, i)].sqrt()).collect();

        let mut corr = vec![vec![0.0; q]; q];
        #[allow(clippy::needless_range_loop)]
        for i in 0..q {
            corr[i][i] = 1.0;
        }
        for c in 0..q {
            for r in (c + 1)..q {
                let rho = vech[idx(r, c)] / (stddev[r] * stddev[c]);
                corr[r][c] = rho;
                corr[c][r] = rho;
            }
        }
        (stddev, corr)
    }

    /// Post-hoc singular check beyond the optimizer's exact-pin decision
    /// ([`Diagnostics::boundary`], unchanged by this — see
    /// [`SINGULAR_REL_TOL`]).
    /// Read-only over the already-assembled `varcorr`; changes no returned
    /// estimate. `false` when `varcorr` is empty (non-mixed-model fit, or a
    /// non-converged fit that left it unassembled).
    ///
    /// `re_scales` is [`common::re_scale_grid`] — the internal RE design-column
    /// scale of each reported standard deviation. The comparison runs on the
    /// INTERNAL standard deviations `sd·s` because the reported ones are not
    /// commensurate with each other: an intercept's is in response units and a
    /// slope's is in response per covariate unit, so their raw ratio moves when
    /// the covariate is re-expressed and the verdict would follow it. On the
    /// internal scale every component is "how much variation this term
    /// contributes", and the check means what it says. All-`1.0` scales — every
    /// intercept-only model — multiply exactly and leave this as it was.
    pub(crate) fn has_negligible_component(&self, re_scales: &[Vec<f64>]) -> bool {
        if self.varcorr.is_empty() {
            return false;
        }
        let stddevs: Vec<Vec<f64>> = (0..self.varcorr.len())
            .map(|g| {
                let sd = self.stddev_corr(g).0;
                let sc = re_scales.get(g);
                sd.iter()
                    .enumerate()
                    .map(|(i, &v)| v * sc.and_then(|s| s.get(i)).copied().unwrap_or(1.0))
                    .collect()
            })
            .collect();
        let max_sd = stddevs.iter().flatten().cloned().fold(0.0_f64, f64::max);
        if max_sd <= 0.0 {
            return false;
        }
        stddevs
            .iter()
            .flatten()
            .any(|&s| s <= SINGULAR_REL_TOL * max_sd)
    }
}

/// Options for `fit`. Carries the safe, defaulted method knobs: unlike
/// a warm start these do not silently move the estimate based on a caller's guess.
#[derive(Clone)]
pub struct FitOptions {
    /// Predictor column indices for which SE is computed.
    pub target_indices: Vec<u32>,
    /// Wald-SE denominator (relocated from `ModelSpec`). Default `Hessian`.
    pub wald_se: WaldSe,
    /// Adaptive Gauss–Hermite node count (relocated from `ModelSpec`). Default 1
    /// (= Laplace). Must be odd and in `1..=MAX_NAGQ`; `>1` is honored on a
    /// binomial/Poisson GLMM with a single grouping factor and `q_p ≤ 3` random
    /// effects per group (scalar intercept → `agq::agq_deviance`, vector RE →
    /// `agq::agq_deviance_vec`, a `k^q_p` product grid). `q_p ≥ 4` is refused by
    /// `assert_model_shape` (a temporary cost/oracle boundary); other ineligible
    /// shapes panic there likewise.
    pub nagq: u8,
    /// Gamma dispersion directive (relocated from `Family::Gamma`). `None` =
    /// estimate φ post-fit (Pearson); `Some(v)` = hold φ fixed at `v`. Ignored by
    /// non-Gamma families.
    pub dispersion: Option<f64>,
    /// Per-row prior (case) weights `wᵢ` — lme4's `weights=`. `None` = unit
    /// weights. `wᵢ` scales each row's loglik/deviance contribution (the
    /// Aitken/M&N prior-weight convention — not an inverse-variance analytic
    /// weight). For an aggregated binomial, `y` is the success PROPORTION and
    /// `wᵢ` the trial count: this is exactly lme4's `cbind(s, m−s)` objective,
    /// whose deviance differs from the expanded-Bernoulli one only by a
    /// data-only saturated constant (same argmin — same β/SE/varcomp).
    /// Dispersion/σ̂² always divide by the raw row count `n−p`, never `Σwᵢ−p`.
    ///
    /// Support matrix: every (family, RE structure, solver) combination,
    /// including AGQ (`nagq > 1`) on the binomial/Poisson shapes it covers — the
    /// per-row `dev_resid` sums in `glmm/agq.rs` carry `wᵢ`, and PIRLS folds the
    /// weights into the conditional mode/curvature (aggregated binomial with small
    /// clusters, e.g. `glmer(cbind(s,m−s) ~ …, nAGQ=k)`, is the canonical case).
    ///
    /// - Gaussian fixed-only: WLS via √wᵢ row pre-scaling, `σ̂² = Σwᵢrᵢ²/(n−p)`;
    ///   matches R `lm(weights=)` (`fit_ols_weighted_matches_r_lm`).
    /// - GLM fixed-only, any family (Binomial/Poisson/Gamma/NB): weighted IRLS —
    ///   `wᵢ` multiplies the working weight and deviance; Gamma's Pearson φ is
    ///   `Σwᵢrᵢ²/(n−p)`; the null deviance uses the weighted mean. Matches R
    ///   `glm(weights=)` (`fit_glm_gamma_weighted_matches_r`,
    ///   `fit_glm_binomial_weighted_aggregated_matches_r`,
    ///   `glm_weighted_deviance_null_golden_value`) and `MASS::glm.nb(weights=)`
    ///   (`fit_glm_nb_weighted_matches_mass`, weighted θ profile).
    /// - Sparse (`Solver::Sparse`) non-Gaussian GLMM, every family: `wᵢ` weights
    ///   the sparse PIRLS working weight/deviance/score identically to the
    ///   dense mixed path below; Gamma's profiled dispersion (`gamma_aic`) and
    ///   `vcov(use.hessian=FALSE)` scale (`glmm_sigma_sq`) take `ws.prior_w`,
    ///   its post-fit Pearson φ̂ sums `wᵢrᵢ²` over raw `n−p` df, and NB's
    ///   marginal-θ profile (`nb_profile_loglik`) takes `opts.weights`.
    ///   Validated against `fit_sparse_gamma_glmm_weighted_matches_lme4` (lme4
    ///   `glmer`), the `sparse_weighted_binomial_*` expanded-vs-aggregated
    ///   tests, and the `sparse_weighted_{poisson,gamma,nb}_matches_replicated`
    ///   weighted-vs-replicated-row equivalence tests.
    /// - Dense (`Solver::NoZ`) mixed GLMM, every non-Gaussian family
    ///   (Binomial/Poisson/Gamma/NB): `wᵢ` weights the working weight,
    ///   deviance, and β-gradient score identically to the sparse binomial path
    ///   above; Gamma's profiled dispersion (`family::gamma_aic`) and its
    ///   `vcov(use.hessian=FALSE)` scale (`family::glmm_sigma_sq`) both take
    ///   `Σwᵢ` in place of `n`. Validated against
    ///   `fit_glmm_cbpp_aggregated_matches_lme4`,
    ///   `fit_glmm_poisson_weighted_matches_lme4`, and
    ///   `fit_glmm_gamma_weighted_matches_lme4` (all lme4 `glmer`, dense).
    /// - Dense (`Solver::NoZ`) mixed Gaussian LMM (weighted REML): every
    ///   row of `[X y Z]` is conceptually √wᵢ-scaled before hitting the unit-
    ///   weight suff-stats/deviance kernel (`LmmSuffStats::add_rows_multi`),
    ///   which folds `wᵢ` straight into the Gram accumulators (`c`, `s`,
    ///   `counts` — the last now `Σ z²·wᵢ`, not a row count; `df = n − p` still
    ///   comes from the raw row count). `σ̂² = pwrss_w/(n−p)` (raw df) falls out
    ///   of the weighted Grams automatically; `tau2 = θ²·σ̂²` and SEs inherit.
    ///   The reported deviance carries an extra `−Σlog wᵢ` (the weighted
    ///   Gaussian log-density's `+½Σlog wᵢ` per row, on the −2ℓ scale) atop the
    ///   usual stripped-constant convention, matching lme4's weighted
    ///   `lmer(weights=)` REMLcrit up to that same stripped constant.
    ///   Validated against `fit_lmm_weighted_matches_lme4` (lme4 `lmer`, dense),
    ///   `fit_lmm_constant_weights_invariant`, and
    ///   `fit_lmm_weighted_boundary_matches_wls`.
    /// - Sparse (`Solver::Sparse`) Gaussian LMM: identical √wᵢ scaling threaded
    ///   through the sparse-Z accumulator instead — `for_each_z_entry`'s
    ///   emitted z values and every raw x/y read in `SparseLmmWorkspace::new`'s
    ///   pass 2 each carry one `√wᵢ` factor, so every Gram product (`ztxy`,
    ///   `cxy`, the packed `pk_*` raw-Gram streams) ends up carrying exactly
    ///   `wᵢ`. Deviance/df stay raw (`n − p`); `fit_mle_sparse` adds the same
    ///   `−Σlog wᵢ` constant as `fit_mle` above. Validated against
    ///   `fit_sparse_lmm_weighted_matches_lme4` (lme4 `lmer`, sparse) and
    ///   `sparse_lmm_constant_weights_invariant`.
    pub weights: Option<Vec<f64>>,
    /// Per-row additive offset `oᵢ` on the linear-predictor scale — R's
    /// `offset=`: `η = o + Xβ (+ Zb)`, every family, every solver path. A fixed
    /// known contribution, NOT a parameter: it adds no column to `X` and never
    /// appears in `beta`/`se`/`vcov`/`df`. The canonical use is a Poisson
    /// exposure, `offset = ln(exposure)`. `None` = no offset (byte-identical to
    /// the pre-offset paths). Gaussian identity-link paths (OLS/LMM) implement
    /// it as the exact `y − o` shift before the sufficient-statistics
    /// accumulation; `Fit::fitted` still reports means on the original `y`
    /// scale (μ̂ = o + Xβ̂ + Zb̂ through the link). Oracle: `glm(offset=)` /
    /// `glmer(offset=)`.
    pub offset: Option<Vec<f64>>,
    /// **Experimental.** Opt into the parallel inner-fit kernels (cluster-outer
    /// AGQ, per-`(i,j)` FD-Hessian grid) — live ONLY when built with the
    /// `parallel` Cargo feature on a non-`wasm32` target. The exclusion is
    /// compile-time, not a runtime check: serial and `wasm32` builds run the
    /// original sequential kernels byte-for-byte regardless of this flag, and
    /// never pull `rayon` in.
    /// (Cluster-outer AGQ is deliberately NOT offered serially: its per-cluster
    /// overhead regresses many-tiny-cluster shapes — measured +12–16% on
    /// observation-level REs — so it exists only as the rayon substrate.)
    ///
    /// Default `false`: parallel results are bit-identical to serial (tested),
    /// but the kernels are new and their perf envelope isn't characterized yet,
    /// so a fit uses them only on explicit opt-in. Batch callers driving many
    /// fits at once (e.g. MCPower's `loop_advanced` hot loop) should keep this
    /// `false` even after opting builds into `parallel` — that caller already
    /// owns the outer parallelism, and stacking a second parallel axis inside
    /// every fit adds per-split overhead (and, for AGQ, a real per-fit CSR
    /// preprocessing cost) for no benefit once the outer loop saturates the
    /// cores.
    ///
    /// Wired to the AGQ cluster-outer restructuring (`agq::agq_deviance`'s
    /// `cluster_rows` path): `false` skips the per-fit `ClusterRowIndex` build and
    /// runs the original node-outer loop. Also gates the FD-Hessian grids (dense and
    /// sparse): `false` keeps the serial cell-by-cell loop. See the design doc above
    /// for the full rationale, including why nesting this under an outer `rayon`
    /// batch loop is safe (one shared work-stealing pool) in a way naive OS-thread
    /// or BLAS-thread nesting is not.
    pub parallel_inner: bool,
    /// Measure [`Diagnostics::boundary_score`] on a fit that pins a variance
    /// component. Default `false`. The score is a diagonal of the exact
    /// second-derivative pass at the pinned point (a hyper-dual PIRLS solve for
    /// the GLMM, the hyper-dual REML kernel for the LMM), which nothing else in
    /// the fit needs: on a small pinned GLMM warm refit it costs about as much
    /// as the fit itself (see `glmm::fit_glmm`'s diagnostics block for the
    /// measurement). [`Diagnostics::kkt_grad_norm`] is unaffected — it comes
    /// from the first-derivative pass and is reported wherever it exists.
    /// Ignored on every route that cannot measure the score (the field stays
    /// empty there either way).
    pub boundary_score: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            target_indices: vec![],
            wald_se: WaldSe::Hessian,
            nagq: 1,
            dispersion: None,
            weights: None,
            offset: None,
            parallel_inner: false,
            boundary_score: false,
        }
    }
}

/// Cold single fit — the default real-data entry. Random-effect
/// level ids are supplied per-row via `ids` ([`GroupIds`]); every level count is
/// derived from them, so `ModelSpec` stays structure-only. Exactly
/// `fit_warm(.., None, ..)`.
///
/// Dispatches on `(family, re.is_some())` (`Gaussian`→OLS/LMM,
/// `Binomial`→GLM/GLMM, Poisson/Gamma/NB likewise; `re: None`→fixed-only,
/// `Some`→mixed). Binomial counts are fit as expanded 0/1 Bernoulli rows (the
/// kernel is Bernoulli) — except when `y` is an aggregated success PROPORTION
/// and [`FitOptions::weights`] carries the trial count, which fits directly
/// without the expansion. That aggregated form is supported on every
/// (family, RE, solver) path, including AGQ (`nagq > 1`) on its binomial/Poisson
/// shapes (see `FitOptions::weights`'s support matrix).
///
/// # Panics
///
/// Panics only on engine invariant violations (`x.len() != n*p`, malformed
/// [`GroupIds`], over-envelope model). Numerical failures signal via
/// `Fit { converged: false, .. }` with NaN-filled estimates — including the
/// degenerate-data cases: a rank-deficient design has its aliased columns
/// dropped and the reduced model fit (`Fit::aliased` flags them, their β/se are
/// NaN), and a design whose aliased column is ALSO an RE slope — unfittable,
/// since the slope has no column left to point at — returns all-NaN with
/// `converged: false` rather than faulting.
///
/// # Examples
///
/// Fixed-only OLS (`re: None`): `y = 2x` exactly, so the fitted slope is 2.
///
/// ```rust
/// use glmm::{fit_cold, Family, FitOptions, GroupIds, ModelSpec};
///
/// let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]; // n=3, p=2, row-major [intercept, x]
/// let y = vec![0.0, 2.0, 4.0];
/// let model = ModelSpec { family: Family::Gaussian, re: None };
/// let opts = FitOptions { target_indices: vec![1], ..FitOptions::default() };
/// let fit = fit_cold(&x, &y, 3, 2, &model, &GroupIds::default(), &opts);
/// assert!(fit.converged());
/// assert!((fit.beta[1] - 2.0).abs() < 1e-9);
/// ```
pub fn fit_cold(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    opts: &FitOptions,
) -> Fit {
    fit_warm(x, y, n, p, model, ids, None, opts)
}

/// Warm single fit — identical to [`fit_cold`] but accepts an optional
/// [`StartValues`] warm start. `Some` warm-starts the optimizer;
/// `None` cold-starts (byte-identical to `fit_cold`). One fit, one answer — the
/// start only shortens the path. The start threads `beta`+`theta` into the
/// LMM/GLMM kernels; fixed-only (OLS/GLM) and the NB global-θ search ignore it
/// (see [`crate::StartValues`]).
///
/// # Panics
///
/// As [`fit_cold`], plus: with `start = Some`, `start.beta.len()` outside
/// `{0, p}` or `start.theta.len()` outside `{0, n_theta}` (the model's RE θ
/// width) — a malformed stable input faults at the entry, not deep in a kernel.
/// Either field may be left empty to cold-start that component alone
/// (see [`crate::StartValues`]).
///
/// # Examples
///
/// An intercept-only Gaussian LMM (3 clusters, `n_theta = 1`) warm-started off a
/// perturbed θ:
///
/// ```rust
/// use glmm::{fit_warm, Family, FitOptions, GroupIds, ModelSpec, ReStructure, Sizing, StartValues};
///
/// let x = vec![1.0; 6]; // n=6, p=1 (intercept-only design)
/// let y = vec![1.0, 2.0, 1.5, 1.2, 2.1, 1.4];
/// let model = ModelSpec {
///     family: Family::Gaussian,
///     re: Some(ReStructure {
///         sizing: Sizing::FixedClusters { n_clusters: 3 },
///         slopes: vec![],
///         extra_groupings: vec![],
///     }),
/// };
/// let ids = GroupIds { primary: vec![0, 1, 2, 0, 1, 2], extra: vec![] };
/// let start = StartValues { beta: vec![0.0], theta: vec![1.0] };
/// let fit = fit_warm(&x, &y, 6, 1, &model, &ids, Some(&start), &FitOptions::default());
/// assert_eq!(fit.beta.len(), 1);
/// ```
#[allow(clippy::too_many_arguments)] // marshals (x, y, n, p, spec, ids, start, opts)
pub fn fit_warm(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Fit {
    assert_eq!(
        x.len(),
        n * p,
        "x must have n*p elements in row-major layout"
    );
    assert_eq!(y.len(), n, "y must have n elements");
    assert_model_shape(model, p, opts.nagq);
    // An EMPTY field means "cold-start this component" — a caller that knows only
    // one of the two (an R `start = list(theta = …)`, lme4's shape) cannot compute
    // the other: the cold β seed is a no-RE GLM fit and the cold θ seed is the
    // blind THETA0 shape, both of which live inside the kernels. So the widths
    // accepted are {0, exact}, and each consumption site falls back on empty.
    if let Some(s) = start {
        assert!(
            s.beta.is_empty() || s.beta.len() == p,
            "StartValues.beta must have p elements, or be empty to cold-start β"
        );
        let n_theta = theta_width(model.re.as_ref());
        assert!(
            s.theta.is_empty() || s.theta.len() == n_theta,
            "StartValues.theta must have n_theta elements for this RE structure, \
             or be empty to cold-start θ"
        );
    }
    // Prior weights: shape/finiteness check at the boundary, not deep in a kernel.
    if let Some(w) = &opts.weights {
        assert_eq!(w.len(), n, "FitOptions.weights must have n elements");
        assert!(
            w.iter().all(|&v| v.is_finite() && v > 0.0),
            "FitOptions.weights must be finite and > 0"
        );
    }
    // Offset: shape/finiteness check at the boundary, like weights above.
    if let Some(o) = &opts.offset {
        assert_eq!(o.len(), n, "FitOptions.offset must have n elements");
        assert!(
            o.iter().all(|&v| v.is_finite()),
            "FitOptions.offset must be finite"
        );
    }
    // Rank-deficiency salvage: drop fixed-effect columns aliased on earlier
    // columns and fit the reduced model (lme4 behavior — NA coefficient, still
    // converges). Path-agnostic: preprocesses the fixed design X before the
    // solver dispatch, so it serves OLS/GLM/LMM/GLMM alike. Runs after the shape
    // asserts (full p still valid) and short-circuits into a reduced re-entry.
    // Tests the UNWEIGHTED X'X at `ALIAS_EPS`, i.e. only genuine redundancy —
    // columns indistinguishable in f64, where there is no separate coefficient
    // to estimate. This is the only place a column is ever dropped. A design
    // that is merely ill-conditioned has a unique, computable answer, and the
    // estimator routes fit it in full and record how badly conditioned it was;
    // the sparse-LMM route is the one exception and still refuses below its own
    // pivot floor.
    if n > 0 && p > 0 {
        let aliased = detect_aliased(x, n, p);
        if aliased.iter().any(|&a| a) {
            return fit_rank_deficient(x, y, n, p, model, ids, start, opts, &aliased);
        }
    }
    // Dispatch over the unified fit core: classify + allocate the per-shape
    // workspace, solve on it, map the lean view to the full `Fit`. Fixed-only is
    // structure-only; mixed derives level counts from the ids. The ids are
    // validated HERE, ahead of the core's own re-check, so a bad `GroupIds`
    // faults against the caller's model with the entry's error message rather
    // than the core's shape-pin panic.
    // The sizing step may also reorder the groupings (`Perm`), so `sized`/`ids`/
    // `perm` travel together from here on: the kernel is fed the reordered ids,
    // and `into_fit` maps the grouping-indexed results back to the order
    // `model` declares.
    let (sized, ids, perm) = match model.re.as_ref() {
        None => (
            model.clone(),
            std::borrow::Cow::Borrowed(ids),
            Perm::IDENTITY,
        ),
        Some(re) => {
            assert_group_ids(re, ids, n);
            spec_sized_from_ids(model, ids)
        }
    };
    let mut ws = core::build_workspace(&sized, perm, n, p, opts);
    let view = core::fit_on(&mut ws, x, y, &ids, start, opts);
    view.into_fit(x, y, &ids, n, p, model, opts)
}
/// Which LMM/GLMM solver a design routes to. `NoZ` is the dense
/// no-Z fast path with bounded stack scratch (the `MAX_*` envelope), kept
/// byte-identical. `Sparse` is the heap sparse-Z path that lifts the caps. The
/// boundary is currently hard-coded at the cap edge; pulling in-envelope-but-large
/// designs into `Sparse` for speed is future, measured work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Solver {
    NoZ,
    Sparse,
}

/// Route a design to `NoZ` (in-envelope) or `Sparse` (over-envelope). The
/// RE-envelope caps — extra-grouping count, primary width `q_p`, per-extra
/// width `q_g` — are a scratch ceiling, not a model ceiling: over the caps
/// routes to `Sparse` instead of failing. Fixed-only and nAGQ legality are
/// handled by `assert_model_shape` (which still runs first and still panics on
/// true invariant violations).
///
/// On top of the structural caps, slope-carrying extra groupings always route
/// `Sparse`, and so does a total `Crossed` level count past
/// `MAX_CROSSED_LEVELS` — see the comments at the clauses.
///
/// The level-count clause reads `Crossed { n_clusters }`, so callers must pass
/// a spec with REAL counts (`spec_sized_from_ids`) — the formula frontend's
/// placeholders (`n_clusters: 1`) would never trip it.
pub(crate) fn classify_design(model: &ModelSpec, _nagq: u8) -> Solver {
    let Some(re) = model.re.as_ref() else {
        return Solver::NoZ; // no random effects ⇒ nothing to make sparse
    };
    let q_p = 1 + re.slopes.len();
    let over = re.extra_groupings.len() > MAX_EXTRA_GROUPINGS
        || q_p > MAX_PRIMARY_Q
        || re
            .extra_groupings
            .iter()
            .any(|g| 1 + g.slopes.len() > MAX_EXTRA_Q);
    // Extra-grouping random slopes always route Sparse. Gaussian: measured on a
    // locked-clock benchmark run — Sparse wins 4–13× across q_g ∈ {2,3,4},
    // n_extra ∈ {2,4,6}; intercept-only extras stay NoZ, 2–1500× the other
    // way. Non-Gaussian: the dense NoZ GLMM
    // kernel builds intercept-only extras exclusively (`glmm::build_z` emits
    // no slope columns for extras; `apply_lambda`/`build_packed_m` carry
    // debug_asserts), so Sparse — whose PIRLS applies full q_g×q_g Λ-blocks
    // per extra level — is the only implemented route.
    let slope_extras = re.extra_groupings.iter().any(|g| !g.slopes.is_empty());
    // Many-level crossed extras route Sparse: every crossed level lands in the
    // dense tail (`reml_deviance`'s `t_dim`; the dense GLMM path's Schur
    // complement likewise), which is cubic in the TOTAL crossed level count —
    // hence the sum over factors, not a per-factor cap. See MAX_CROSSED_LEVELS
    // for the measured blowup and the placeholder threshold's rationale.
    let crossed_levels: usize = re
        .extra_groupings
        .iter()
        .map(|g| match g.relation {
            GroupingRelation::Crossed { n_clusters } => n_clusters as usize,
            GroupingRelation::NestedWithin { .. } => 0,
        })
        .sum();
    if over || slope_extras || crossed_levels > crate::consts::MAX_CROSSED_LEVELS {
        Solver::Sparse
    } else {
        Solver::NoZ
    }
}

#[cfg(test)]
pub(crate) fn classify_design_pub(model: &ModelSpec, nagq: u8) -> Solver {
    classify_design(model, nagq)
}

// ---------------------------------------------------------------------------
// Re-exports for consumers outside `fit` (mirrors `src/glmm/mod.rs`'s
// mod+re-export convention): `sparse.rs` reaches these as `crate::fit::X`,
// `spec.rs` reaches `assert_model_shape_pub`, and the `loop_advanced` cargo
// feature's `crate::loop_advanced` re-exports the dev seam from here.
// ---------------------------------------------------------------------------

#[cfg(test)]
pub(crate) use common::assert_model_shape_pub;
#[cfg(any(test, feature = "loop_advanced"))]
pub use common::spec_sized_from_ids_pub;
pub(crate) use common::{
    assemble_ranef_sparse, assemble_varcorr, glmm_loglik, lmm_fitted, lmm_loglik, model_df,
    nan_vcov, pinned_flags, ranef_level_counts, re_scale_grid, vcov_from_chol,
};
// The one diagnostics carrier. Always `pub` (it is `FitView::diagnostics`'s
// return type), re-exported only for the loop tier — the stable path reads it
// through `Fit`, never directly.
#[cfg(feature = "loop_advanced")]
pub use common::FitDiagnostics;
pub(crate) use glm::{golden_max_ln_theta, nb_profile_loglik};
pub(crate) use glmm::glm_warm_start_beta;
#[cfg(test)]
pub(crate) use lmm::fit_mle_noz_pub;
// Unified fit-core surface for the loop tier.
#[cfg(feature = "loop_advanced")]
pub use core::{build_workspace, fit_on, FitView, FitWorkspace};
#[cfg(feature = "loop_advanced")]
pub use loop_advanced_seam::{
    build_lmm_seam_ws, lmm_objective_at, lmm_sweep_fit, lmm_sweep_fit_on, LmmSeamWs,
    LmmSweepOutcome,
};

// `pub(crate)` only so `src/sparse/tests.rs` can reach the shared Tier 1 pin
// bands; the module is `cfg(test)` either way.
#[cfg(test)]
pub(crate) mod common_tests;
#[cfg(test)]
mod glm_tests;
#[cfg(test)]
mod glmm_tests;
#[cfg(test)]
mod lmm_tests;
#[cfg(test)]
mod ols_tests;
