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
//! (`src/ols.rs`/`src/lmm.rs`/`src/glm.rs`/`src/glmm/`). `common` holds
//! helpers shared by 2+ of those modules; `loop_advanced_seam` holds the
//! unstable dev-only surface re-exported through `crate::loop_advanced`.

use crate::consts::{MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q, MAX_PRIMARY_Q};
use crate::{BinomialLink, Family, GroupIds, GroupingRelation, ModelSpec, StartValues, WaldSe};

mod common;
mod glm;
mod glmm;
mod lmm;
mod loop_advanced_seam;
mod ols;

use common::{
    assert_group_ids, assert_model_shape, detect_aliased, fit_rank_deficient, spec_sized_from_ids,
    theta_width,
};
use glm::{fit_glm, fit_glm_nb};
use glmm::{fit_glmm, fit_glmm_nb};
use lmm::fit_mle;
use ols::fit_ols;

/// Result of `fit`. Fixed-effect estimates cover all p predictors; SE and
/// tau2 have the ranges below. Non-target SE slots are NaN.
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
    /// the joint (θ,β) FD-Hessian covariance, and `WaldSe::Rx` the p×p Schur
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
    /// `parity/goldens/sleepstudy_lmm.json` `sigma`, asserted in
    /// `fit_sleepstudy_slope_varcorr_matches_lme4`) — and `1.0` for
    /// binomial/Poisson (where dispersion is fixed, not estimated). NaN on a
    /// Gaussian fit with no honest endpoint (non-converged OLS, degenerate
    /// LMM).
    pub dispersion: f64,
    /// Whether the optimizer reached its convergence criterion. `false` means
    /// `se`/`vcov`/`dispersion` above are the NaN-fill described on each field.
    pub converged: bool,
    /// RE (co)variance per grouping: one **vech-packed
    /// lower-triangular** covariance block `D̂ = σ̂²·Λ̂Λ̂'` per grouping, in
    /// declaration order (primary, then each extra). σ̂² is the residual scale
    /// for an LMM and the free GLMM scale `pwrss/n` for dispersion families
    /// (Gamma) — exactly the factor lme4's `VarCorr` stddevs carry; it is ≡ 1
    /// for binomial/Poisson/NB, and the same scale `tau2` reports, so the two
    /// accessors agree. Vech order is column-major
    /// lower-triangular (matching the θ vech convention): for a `q×q` block,
    /// `(0,0),(1,0),…,(q-1,0),(1,1),…,(q-1,q-1)`. Validated against lme4
    /// `VarCorr` (`parity/goldens/sleepstudy_lmm.json`; Gamma scale:
    /// `parity/goldens/sim_gamma_glmm.json`). Empty for OLS/GLM
    /// (no random effects). This is the q≥2-valid replacement for `tau2`'s
    /// per-component variances; `tau2` is retained for back-compat.
    pub varcorr: Vec<Vec<f64>>,
    /// SE of each RE standard deviation, laid out like `tau2` (per θ coordinate,
    /// length `n_theta`; primary block then each extra, in declaration order).
    /// Populated ONLY on a converged GLMM `WaldSe::Hessian` fit, from the θ block
    /// of the joint (θ,β) FD-Hessian covariance that `fd_hessian_cov` inverts and
    /// otherwise discards. NaN under `WaldSe::Rx`, on the Hessian RX fallback, and
    /// for OLS/LMM (no Hessian machinery). Correct for SCALAR groupings only (the
    /// reachable GLMM case), where the RE stddev equals its θ so the θ-scale SE is
    /// the stddev SE directly; a q≥2 block would need a delta-method Jacobian.
    ///
    /// Scale caveat for dispersion families (Gamma): this SE stays on the
    /// **θ scale** — lme4's θ-Hessian convention — while `varcorr`/`tau2` carry
    /// the σ̂² factor. Deliberately NOT multiplied by σ̂: the joint Hessian does
    /// not carry cov(σ̂, θ̂), so a σ̂-rescaled SE would be a delta-method value
    /// that matches no oracle. For the φ≡1 families the two scales coincide and
    /// there is no split. (The parity harness skips `sd_se` gating for
    /// dispersion families accordingly.)
    pub stddev_se: Vec<f64>,
    /// Rank-deficiency mask, length `p`: `true` for a fixed-effect
    /// column dropped because it is aliased (linearly dependent) on an earlier
    /// column, mirroring lme4's `NA`-coefficient behavior. The corresponding
    /// `beta`/`se` slots are `NaN` and `converged` stays `true` (the reduced
    /// model fits). All-`false` when the design is full-rank.
    pub aliased: Vec<bool>,
    /// Objective evaluations consumed by the θ (LMM) / joint [θ|β] (GLMM)
    /// BOBYQA search — GLMM counts both stages. 0 where no derivative-free
    /// optimizer runs (OLS/GLM closed-form or IRLS paths). Deterministic and
    /// clock-independent; the optimizer-grid campaign's primary metric.
    pub n_eval: usize,
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
    /// `true` iff the fit converged onto the θ boundary (≥ 1 diagonal variance
    /// component pinned at 0 — `boundary_hit == 1` internally, OR a converged
    /// diagonal stddev negligible next to its fit's largest — see
    /// [`Fit::has_negligible_component`]), the same condition lme4's
    /// `isSingular` reports. `false` for OLS/GLM and for an LMM
    /// `MaxFunReached` cap-out — a capped endpoint is reported as a point,
    /// not accepted onto the boundary, so it never sets this flag even when
    /// its diagonals are near zero.
    pub singular: bool,
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
    /// `g⁻¹(Xβ̂)` for fixed-only. Empty on non-converged fits and on the
    /// Gaussian LMM paths (dense and sparse), which fit via sufficient
    /// statistics and never materialize per-row means — an LMM `fitted` needs
    /// the conditional modes first and lands with them.
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
    /// children. Empty on non-converged fits and on the Gaussian LMM paths
    /// (see [`Fit::fitted`]).
    pub ranef: Vec<f64>,
    /// Level count per grouping for slicing [`Fit::ranef`], declaration order
    /// (primary, then each extra). `ranef.len() = Σ_g levels[g]·q_g`. Empty
    /// exactly when `ranef` is.
    pub ranef_levels: Vec<usize>,
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
        let q = (((1 + 8 * len) as f64).sqrt() as usize - 1) / 2;
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
    /// (`boundary_hit`, unchanged by this — see [`SINGULAR_REL_TOL`]).
    /// Read-only over the already-assembled `varcorr`; changes no returned
    /// estimate. `false` when `varcorr` is empty (non-mixed-model fit, or a
    /// non-converged fit that left it unassembled).
    pub(crate) fn has_negligible_component(&self) -> bool {
        if self.varcorr.is_empty() {
            return false;
        }
        let stddevs: Vec<Vec<f64>> = (0..self.varcorr.len())
            .map(|g| self.stddev_corr(g).0)
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
/// `Fit { converged: false, .. }` with NaN-filled estimates.
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
/// assert!(fit.converged);
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
/// As [`fit_cold`], plus: with `start = Some`, `start.beta.len() != p` or
/// `start.theta.len() != n_theta` (the model's RE θ width) — a malformed stable
/// input faults at the entry, not deep in a kernel.
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
    if let Some(s) = start {
        assert_eq!(s.beta.len(), p, "StartValues.beta must have p elements");
        assert_eq!(
            s.theta.len(),
            theta_width(model.re.as_ref()),
            "StartValues.theta must have n_theta elements for this RE structure"
        );
    }
    // Prior weights: shape-check at the boundary, then reject unsupported paths
    // here (not deep in a kernel) via the capability map below.
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
    // Rank-deficiency salvage: drop fixed-effect columns aliased on
    // earlier columns and fit the reduced model (lme4 behavior — NA coefficient,
    // still converges). Path-agnostic: preprocesses the fixed design X before the
    // solver dispatch, so it serves OLS/GLM/LMM/GLMM alike. Runs after the shape
    // asserts (full p still valid) and short-circuits into a reduced re-entry.
    if n > 0 && p > 0 {
        let aliased = detect_aliased(x, n, p);
        if aliased.iter().any(|&a| a) {
            return fit_rank_deficient(x, y, n, p, model, ids, start, opts, &aliased);
        }
    }
    match (&model.family, model.re.as_ref()) {
        // Fixed-only (OLS/GLM/NB-GLM): `ids` and `start` are unused — a warm start
        // is a no-op for fixed-only fits. No GroupIds shape check.
        (Family::Gaussian, None) => fit_ols(x, y, n, p, opts),
        (
            Family::Poisson { .. }
            | Family::Gamma { .. }
            | Family::Binomial {
                link: BinomialLink::Probit | BinomialLink::Logit,
            },
            None,
        ) => fit_glm(model.family, f64::NAN, x, y, n, p, opts),
        (Family::NegativeBinomial { .. }, None) => fit_glm_nb(x, y, n, p, None, opts),
        // Mixed (LMM/GLMM): validate the ids, size the spec from them, dispatch.
        (family, Some(re)) => {
            assert_group_ids(re, ids, n);
            let sized = spec_sized_from_ids(model, ids);
            // Classify the SIZED spec: the level-count clause needs real
            // `n_clusters`, and the frontend passes placeholders (`sized ==
            // model` for callers who already pass real counts).
            match classify_design(&sized, opts.nagq) {
                Solver::NoZ => match family {
                    Family::Gaussian => {
                        fit_mle(x, y, n, p, &sized, &ids.primary, &ids.extra, start, opts)
                    }
                    Family::NegativeBinomial { .. } => {
                        fit_glmm_nb(x, y, n, p, &sized, &ids.primary, &ids.extra, start, opts)
                    }
                    Family::Binomial { .. } | Family::Poisson { .. } | Family::Gamma { .. } => {
                        fit_glmm(
                            x,
                            y,
                            n,
                            p,
                            &sized,
                            &ids.primary,
                            &ids.extra,
                            f64::NAN,
                            start,
                            opts,
                        )
                        .0
                    }
                },
                // Over-envelope designs: the Gaussian sparse-Z REML
                // path, the sparse NB marginal-θ wrapper, and the sparse non-Gaussian
                // PIRLS driver — every wired family fits, no reachable panic.
                Solver::Sparse => match family {
                    Family::Gaussian => crate::sparse::fit_mle_sparse(
                        x,
                        y,
                        n,
                        p,
                        &sized,
                        &ids.primary,
                        &ids.extra,
                        start,
                        opts,
                    ),
                    Family::NegativeBinomial { .. } => crate::sparse::fit_glmm_nb_sparse(
                        x,
                        y,
                        n,
                        p,
                        &sized,
                        &ids.primary,
                        &ids.extra,
                        start,
                        opts,
                    ),
                    Family::Binomial { .. } | Family::Poisson { .. } | Family::Gamma { .. } => {
                        crate::sparse::fit_glmm_sparse(
                            x,
                            y,
                            n,
                            p,
                            &sized,
                            &ids.primary,
                            &ids.extra,
                            f64::NAN,
                            start,
                            opts,
                        )
                        .0
                    }
                },
            }
        }
    }
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
    // Extra-grouping random slopes always route Sparse. Gaussian: measured
    // (d2 Phase-1 crossover sweep, 2026-07-02, locked machine — Sparse wins
    // 4–13× across q_g ∈ {2,3,4}, n_extra ∈ {2,4,6}; intercept-only extras
    // stay NoZ, 2–1500× the other way). Non-Gaussian: the dense NoZ GLMM
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

pub(crate) use common::{
    assemble_ranef_sparse, assemble_varcorr, glmm_loglik, lmm_loglik, model_df, nan_vcov,
    ranef_level_counts, vcov_from_chol,
};
#[cfg(test)]
pub(crate) use common::{assert_model_shape_pub, spec_sized_from_ids_pub};
pub(crate) use glm::{golden_max_ln_theta, nb_profile_loglik};
pub(crate) use glmm::glm_warm_start_beta;
#[cfg(test)]
pub(crate) use lmm::fit_mle_noz_pub;
#[cfg(feature = "loop_advanced")]
pub use loop_advanced_seam::{
    build_lmm_seam_ws, build_lmm_workspace, lmm_objective_at, lmm_sweep_fit, lmm_sweep_fit_on,
    refit_lmm, LmmSeamWs, LmmSweepOutcome,
};

#[cfg(test)]
mod common_tests;
#[cfg(test)]
mod glm_tests;
#[cfg(test)]
mod glmm_tests;
#[cfg(test)]
mod lmm_tests;
#[cfg(test)]
mod ols_tests;
