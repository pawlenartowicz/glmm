//! Friendly stable `fit` entry point for the `glmm` crate.
//!
//! Owns all scratch; dispatches on `(family, re.is_some())`; returns `Fit`.
//! This is the additive stable public surface —
//! it never touches any kernel; only marshals data and scratch into kernel
//! calls, then copies results out. The full route table — every path from
//! [`fit`]/[`fit_with_start`] to its solver — is in `documentation/algorithms.md`;
//! no route there is feature-gated. This file only tracks its own module
//! split.
//!
//! # Calling convention
//!
//! `x` is a design matrix in **row-major f64** layout: element `(i, j)` is at
//! `x[i * p + j]`. `y` is the response vector of length `n`. Both arrive in a
//! [`Data`]. The kernels use faer column-major f64 internally; conversion is
//! done here.
//!
//! Cluster ids and extra-grouping ids come from the caller's [`GroupIds`]; every
//! random-effect level count is derived from them.
//!
//! # Module layout
//!
//! `fit_checked` dispatches on `(family, re.is_some())` into one estimator
//! module per family — `ols`/`lmm` (Gaussian), `glm` (fixed-effects
//! binomial/Poisson/Gamma/NB), `glmm` (mixed non-Gaussian, plus GLMM-NB) —
//! each of which marshals into the matching numerical kernel
//! (`src/ols.rs`/`src/lmm/mod.rs`/`src/glm.rs`/`src/glmm/`). `common` holds
//! helpers shared by 2+ of those modules.

use crate::consts::{MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q, MAX_PRIMARY_Q};
use crate::inference::Objective;
use std::borrow::Cow;

use crate::{Data, Design, Family, GroupIds, GroupingRelation, ModelSpec, StartValues, WaldSe};

mod common;
mod core;
mod glm;
mod glmm;
mod lmm;
mod ols;

use common::fit_rank_deficient;
// `pub(crate)` so `Fit::r2` and `Fit::icc` check their data as the fit entry
// points do.
pub(crate) use common::validate;
// `pub(crate)` so `crate::predict` checks new rows as the fit checks its data.
pub(crate) use common::validate_design;
// `pub(crate)` so the calibration tier decides a design's aliasing with the
// alias gate's own test.
pub(crate) use common::detect_aliased;
// `pub(crate)` so `orchestrate::run_fit` can check a port's warm-start θ width
// before the fit, the way `validate` checks it.
#[cfg(any(test, feature = "orchestrate"))]
pub(crate) use common::theta_width;
// The spec with its level counts, which the RE workspace builders in
// `crate::lmm` and `crate::glmm` read.
pub(crate) use common::SizedSpec;
// The random-effect-only column count, which the LMM accumulator sizes its
// slope buffers by.
pub(crate) use common::re_only_count;
// The sizer, for the tests outside `fit` that build kernel workspaces from
// data-derived ids.
#[cfg(test)]
pub(crate) use common::spec_sized_from_ids;
// The workspace door, for the tests outside `fit` that need a `Fit` only the
// loop tier can produce (a prestandardized design); `loop_advanced` exports
// it below.
#[cfg(all(test, feature = "formula", not(feature = "loop_advanced")))]
pub(crate) use core::{build_workspace, fit_on};

/// Result of `fit`. Fixed-effect estimates cover all p predictors; SE and
/// tau2 have the ranges below. Non-target SE slots are NaN.
///
/// `#[non_exhaustive]`: construct one only through [`fit`]/[`fit_with_start`],
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
    /// iff `se[i]` and `se[j]` both are. So a [`SeFor::Only`]
    /// subset leaves everything outside the target block NaN (there is no
    /// covariance to report for a coefficient whose variance was never
    /// computed), and a non-converged fit is all-NaN — except an LMM or GLMM
    /// fit that hit `MaxFunReached`: that budget-exhausted exit reports its
    /// finite covariance at the capped endpoint here instead. A GLM or GLMM fit
    /// under [`Note::SeUnavailable`] is all-NaN here and in `se` whatever its
    /// `converged`.
    ///
    /// Sources, by path: OLS/GLM/LMM invert the same Cholesky factor `se`'s
    /// forward solve already walks; GLMM `WaldSe::Hessian` takes the β block of
    /// the joint (θ,β) Hessian covariance (exact on every shape
    /// `derivative::supports_shape` accepts, finite-difference elsewhere),
    /// and `WaldSe::Rx` the p×p Schur
    /// inverse — both already formed in full and previously discarded down to a
    /// diagonal. Unlike `stddev_se`, this is populated on the Hessian's RX
    /// fallback too (that fallback inverts a full p×p covariance; only a
    /// double failure, where the Hessian AND the fallback both fail, leaves it
    /// NaN, under [`Note::SeUnavailable`]).
    pub vcov: Vec<Vec<f64>>,
    /// Per-element Cholesky-scaled values `theta[k]^2 * sigma_sq`. These equal
    /// the random-effect variance components only for diagonal/scalar RE
    /// components (q=1 / scalar-extra — the currently reachable case); slope
    /// (q≥2) models are not yet validated through this field. Empty for OLS.
    /// Laid out like `varcorr`: a block without a random intercept reports
    /// its slopes only.
    pub tau2: Vec<f64>,
    /// Estimated dispersion: `φ` for Gamma (the Pearson moment on a GLM,
    /// `summary(glm)`'s; the maximum-likelihood estimate on a GLMM) and
    /// inverse-Gaussian (the Pearson moment — GLM only, no mixed route), the
    /// estimated shape `θ` for negative-binomial, the residual variance `σ̂²`
    /// for Gaussian — `RSS/(n−p)` for OLS (raw-row df, matching R
    /// `summary.lm`'s `sigma²`; the same `sigma_sq` that scales `se`/`vcov`)
    /// and the criterion's `pwrss/ν` for LMM, `pwrss/(n−p)` under REML and
    /// `pwrss/n` under ML (matching lme4 `sigma()²` on the same fit; oracle:
    /// `validation/goldens/sleepstudy_lmm.json` `sigma`, asserted in
    /// `fit_sleepstudy_slope_varcorr_matches_lme4`) — and `1.0` for
    /// binomial/Poisson (where dispersion is fixed by the family, not
    /// estimated). NaN on a fit reporting `converged == false`, except an LMM
    /// or GLMM fit that hit `MaxFunReached`, which reports it at the capped
    /// endpoint, as `vcov` does. This includes the fixed families' `1.0` and a
    /// φ held fixed through [`FitOptions::dispersion`].
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
    /// for an LMM and ≡ 1 on every GLMM family: a Gamma GLMM carries φ in its
    /// PIRLS weights, so θ̂ is already in the response's absolute units, as
    /// glmmTMB's `VarCorr` reports it (lme4's reports `σ·θ̂` instead). It is the
    /// same scale `tau2` reports, so the two accessors agree. Vech order is column-major
    /// lower-triangular (matching the θ vech convention): for a `q×q` block,
    /// `(0,0),(1,0),…,(q-1,0),(1,1),…,(q-1,q-1)`. Validated against lme4
    /// `VarCorr` (`validation/goldens/sleepstudy_lmm.json`) and glmmTMB's on Gamma
    /// (`validation/goldens/sim_gamma_glmm_tmb.json`). Empty for OLS/GLM
    /// (no random effects). This is the q≥2-valid replacement for `tau2`'s
    /// per-component variances; `tau2` is retained for back-compat.
    ///
    /// A block without a random intercept (`(0 + x | g)`,
    /// [`crate::ReTerm::no_intercept`]) reports its slopes only: the kernel
    /// keeps an intercept column there at zero variance, and that column is
    /// dropped from this block and from `tau2`, `stddev_se` and `ranef`. The
    /// covariance between two terms of one grouping (`(x || g)`) is exactly
    /// `0.0`.
    pub varcorr: Vec<Vec<f64>>,
    /// SE of each RE standard deviation, laid out like `tau2` (per θ coordinate,
    /// length `n_theta`, less the slots of a hidden intercept column, see
    /// `varcorr`; primary block then each extra, in declaration order):
    /// the SE of row k's stddev `√D̂_kk` (the stddev `varcorr` reports) sits at
    /// that row's diagonal slot, and the off-diagonal slots of a q≥2 block are
    /// NaN. Populated ONLY on a converged GLMM `WaldSe::Hessian` fit, from the θ block
    /// of the joint (θ,β) Hessian covariance that `joint_hessian_cov` inverts and
    /// otherwise discards (exact Hessian on every shape
    /// `derivative::supports_shape` accepts, finite-difference elsewhere).
    /// NaN under `WaldSe::Rx`, on the Hessian RX fallback, and
    /// for OLS/LMM (no Hessian machinery). The joint Hessian is taken in the
    /// fit's internal coordinates and the SE is the delta method's on them
    /// (the stddev's square is a polynomial in the internal θ). A stddev of
    /// exactly 0 has no derivative and reports NaN, except where it reads a
    /// single θ coordinate (a SCALAR grouping, or the intercept of a block
    /// whose slopes are not centered): there the SE is that coordinate's,
    /// as at every other value.
    ///
    /// Scale caveat for dispersion families (Gamma): this SE stays on the
    /// **θ scale** — lme4's θ-Hessian convention — while `varcorr`/`tau2` carry
    /// the σ̂² factor. Deliberately NOT multiplied by σ̂: the joint Hessian does
    /// not carry cov(σ̂, θ̂), so a σ̂-rescaled SE would be a delta-method value
    /// that matches no oracle. For the φ≡1 families the two scales coincide and
    /// there is no split. (The validation suite skips `sd_se` gating for
    /// dispersion families accordingly.)
    pub stddev_se: Vec<f64>,
    /// Objective value evaluations of the outer search over θ (LMM) or the
    /// joint `[θ|β]` (GLMM): BOBYQA's and BFGS's together, every stage and
    /// fallback included. A BFGS gradient reuses its point's value evaluation
    /// and is not counted here (see [`Diagnostics::n_grad`]). 0 where no outer
    /// search runs (OLS/GLM closed-form or IRLS paths). Deterministic and
    /// clock-independent; the optimizer-grid campaign's primary metric.
    pub n_eval: usize,
    /// Dev-only optimizer evaluation counters — the stage split, the shrink
    /// phase, the PIRLS-iteration histogram and the AGQ node cost. Present
    /// only under the off-by-default `counters` feature, which is NOT
    /// semver-covered; zeros on every route that runs no outer search (OLS,
    /// GLM).
    #[cfg(feature = "counters")]
    pub counters: crate::counters::EvalCounters,
    /// Minimized optimizer criterion at the accepted point. LMM: the profiled
    /// deviance of the fit's criterion ([`Fit::reml`]) as computed by
    /// `reml_deviance`, on the caller's design. A REML fit's equals lme4's
    /// REMLcrit minus the data-independent constant ν·(1 + ln 2π), ν = n − p
    /// (validated against the frozen lme4 sleepstudy reference); an ML fit's
    /// equals lme4's ML `deviance()` minus the same constant with ν = n. GLMM: the
    /// marginal Laplace deviance `d(y,ũ) + ‖ũ‖² + log|A|` (see
    /// `GlmmFit::deviance`), which differs from −2·logLik by a data-only
    /// saturated constant on binomial/Poisson fits and equals −2·logLik
    /// exactly on Gamma and NB fits (see `fit::common::glmm_loglik`). NaN for
    /// OLS/GLM and on optimizer/numerical failure (GLMM non-convergence
    /// surfaces as +∞ internally; mapped to NaN here). An LMM or GLMM fit
    /// that hits `MaxFunReached` still reports the finite endpoint deviance
    /// here with `converged == false` — the plateau policy: a
    /// `MaxFunReached` cap-out reports its finite endpoint with
    /// `converged == false` rather than NaN-filling.
    pub deviance: f64,
    /// The actual log-likelihood at the fitted parameters — `deviance` with its
    /// dropped data-only constants restored, on the `logLik()` scale (R/lme4):
    ///
    /// - **OLS/GLM** — the standard closed forms (R `logLik.lm`/`logLik.glm`,
    ///   `MASS::glm.nb`), including under prior weights, EXCEPT Gamma and
    ///   inverse-Gaussian: both report the precision log-likelihood (row `i`
    ///   has variance `φ·V(μᵢ)/wᵢ`) rather than R's `logLik.glm`/`Gamma()$aic`
    ///   case-weight one (`−½·(D/φ̂ + gamma_dispersion_term)` at the maximised
    ///   φ̂ for Gamma, `family::inv_gaussian_aic`'s precision form for
    ///   inverse-Gaussian). The two agree only at unit weights; a weighted
    ///   Gamma GLM's `loglik` is the maximised precision value, not R's
    ///   `logLik(fg)` (measured: −47.27 vs −118.21 on
    ///   `fit_glm_gamma_weighted_matches_r`'s data). On a fit that reproduces
    ///   the data exactly the likelihood rises without bound as φ → 0, so a
    ///   deviance that rounds to zero gives `+∞`. An OLS fit made with
    ///   [`FitOptions::reml`] `= Some(true)` reports the REML log-likelihood
    ///   instead, R's `logLik(lm, REML = TRUE)`: `−½(d + (n−p)·(1 + ln 2π))`
    ///   with `d = ln|XᵀWX| + (n−p)·ln(RSS_w/(n−p)) − Σ ln wᵢ` on the caller's
    ///   X (`RSS_w` the weighted residual sum of squares). `d` is the REML
    ///   `deviance` an LMM with the same fixed part reports when its variance
    ///   is pinned at zero, so the two are on one scale (`deviance` itself
    ///   stays NaN on OLS).
    /// - **LMM**: the fit's criterion ([`Fit::reml`]). Under REML (the
    ///   default) the **REML criterion** `−REMLcrit/2`,
    ///   `−½(deviance + (n−p)·(1 + ln 2π))`, matching `lme4::logLik` on a
    ///   `REML=TRUE` fit. REML criteria are comparable only between models with
    ///   IDENTICAL fixed effects: an AIC/LRT across different fixed parts is
    ///   meaningless, so check [`Fit::reml`] before comparing. Under ML
    ///   ([`FitOptions::reml`] `= Some(false)`) the ML log-likelihood
    ///   `−½(deviance + n·(1 + ln 2π))`, matching `lme4::logLik` on a
    ///   `REML=FALSE` fit.
    /// - **GLMM** — the marginal Laplace/AGQ log-likelihood:
    ///   `−½·deviance + saturated_loglik` (binomial/Poisson, see
    ///   `family::saturated_loglik`), `−½·deviance` alone for Gamma and NB
    ///   (Gamma: `deviance` carries the whole precision log-density through
    ///   `family::gamma_dispersion_term` at the ML φ̂ (`Fit::dispersion` on a
    ///   GLMM, unlike the GLM's Pearson moment); NB: `deviance` already
    ///   carries the θ̂-dependent saturated term the outer θ search adds — see
    ///   `fit::common::glmm_loglik`). Matches `lme4::logLik` on the same fit on
    ///   the canonical binomial and Poisson links, including the
    ///   aggregated-binomial `cbind(s, m−s)` form under `weights=`. On
    ///   non-canonical links, NB and Gamma it matches glmmTMB's instead: lme4
    ///   builds its Laplace log-determinant from the expected (Fisher) weight
    ///   there, and on Gamma fits at its `pwrss/n` plug-in scale, so its
    ///   logLik is a different function of the parameters.
    ///
    /// NaN wherever `deviance`'s failure modes apply (non-converged/degenerate
    /// fits); finite on an LMM or GLMM `MaxFunReached` endpoint, like
    /// `deviance`.
    /// `AIC = 2·df − 2·loglik`, `BIC = df·ln(n) − 2·loglik` with [`Fit::df`] —
    /// on a GLMM budget-exhausted exit `df` is 0 (see [`Fit::df`]) and this
    /// identity does not apply there, even though `loglik` itself is finite.
    pub loglik: f64,
    /// Parameters counted for AIC/BIC: retained fixed effects (`p` minus
    /// aliased columns, matching lme4's NA-coefficient handling) + the RE θ
    /// entries the fit estimates (not those a grouping's terms hold at zero,
    /// as in `(x || g)`) + 1 if the family estimates a dispersion/scale (Gaussian
    /// σ², Gamma φ unless held fixed via [`FitOptions::dispersion`], NB θ).
    /// 0 on degenerate NaN-fill paths, and also 0 on a GLMM fit that hits
    /// `MaxFunReached` (gated on `converged`, unlike `loglik`/`deviance`) — the
    /// AIC/BIC identity in [`Fit::loglik`] does not apply at that endpoint. An
    /// LMM `MaxFunReached` endpoint reports its usual nonzero `df` instead.
    pub df: usize,
    /// `true` iff `loglik` is a REML criterion rather than an ML
    /// log-likelihood: a Gaussian mixed fit under REML (the default; a fit
    /// made with [`FitOptions::reml`] `= Some(false)` is by ML and reports
    /// `false`), read off the criterion the fit minimized, and an OLS fit made
    /// with [`FitOptions::reml`] `= Some(true)`.
    /// Model comparisons (AIC/LRT) across fits with different fixed effects are
    /// invalid when this is set — mirror of lme4's REML-fit `anova` warning.
    pub reml: bool,
    /// The number of rows the fit was given, `n`, on every route and whatever
    /// the outcome: a non-converged or degenerate fit reports it too (where
    /// `fitted` is empty), and the rank-deficient salvage reports its reduced
    /// fit's, which is the same `n`.
    pub nobs: usize,
    /// The objective the fit maximised, the label a likelihood-ratio
    /// comparison carries: [`Objective::LeastSquares`] on OLS (also on an OLS
    /// fit made with [`FitOptions::reml`] `= Some(true)`, whose `loglik` is
    /// the REML value), [`Objective::MaximumLikelihood`] on a GLM,
    /// [`Objective::Reml`] or [`Objective::MaximumLikelihood`] on a Gaussian
    /// mixed fit from its criterion ([`Fit::reml`]), and on a GLMM
    /// [`Objective::Laplace`] at [`FitOptions::nagq`] `= 1` or
    /// [`Objective::Agq`] with the node count `nagq` above it. Set on every
    /// fit, a non-converged or degenerate one included. Whether `loglik` is a
    /// REML criterion is [`Fit::reml`]'s answer; this field adds the label and
    /// the node count.
    pub objective: Objective,
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
    /// declaration order; slopes only on a block without a random intercept,
    /// see `varcorr`) at `[l·q .. (l+1)·q]`. `η̂ = Xβ̂ + Zb̂` reproduces the
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
    /// The Gaussian LMM paths never form these during the fit (the profiled
    /// criterion, REML or ML, does not need them) and recover them once at θ̂ by
    /// back-substitution afterwards, which is why they are here without the
    /// per-evaluation cost of forming them.
    pub ranef: Vec<f64>,
    /// Level count per grouping for slicing [`Fit::ranef`], declaration order
    /// (primary, then each extra). `ranef.len() = Σ_g levels[g]·q_g`. Empty
    /// exactly when `ranef` is.
    pub ranef_levels: Vec<usize>,
    /// Conditional covariance of each level's random effects given the data,
    /// at the fitted parameters: lme4's `ranef(condVar = TRUE)` (`postVar`),
    /// on the scale of [`Fit::ranef`]. The layout of `ranef` with each level's
    /// `q` values replaced by the vech of its `q × q` block, in the order of
    /// `varcorr` (column-major lower triangle), so a grouping's block holds
    /// `levels[g]·q_g(q_g+1)/2` values; the conditional SDs are the square
    /// roots of the diagonal entries. A block without a random intercept
    /// reports its slopes only, as `ranef` does, and a padded nested slot
    /// holds the prior covariance (it has no rows).
    ///
    /// On a Gaussian LMM the block is `σ̂²·Λ̂(Λ̂ᵀZᵀWZΛ̂ + I)⁻¹Λ̂ᵀ` restricted to
    /// the level (`W` the prior weights); on a GLMM the same with `σ̂² = 1`
    /// and `W` the observed curvature `−∂²ℓᵢ/∂ηᵢ²` at the conditional modes
    /// (the Laplace approximation; on a Gamma GLMM it carries `1/φ̂`). Both
    /// hold β̂ fixed, as lme4's do.
    ///
    /// Formed once when the `Fit` is assembled, from the factors at the
    /// optimum, and only when [`FitOptions::condvar`] asks for it. Empty on a
    /// fit that did not ask, where `ranef` is, and on the layouts where it is not
    /// computed: a Gaussian LMM on the sparse route (an extra grouping with a
    /// random slope, over the dense envelope or more than
    /// `MAX_CROSSED_LEVELS` crossed levels) and a GLMM on the packed layout,
    /// which would need a selected inverse of the whole system.
    pub condvar: Vec<f64>,
    /// Per level `l`, in the level order of [`Fit::condvar`], the row-major
    /// `q × p` block `K_l` of `Λ̂(Λ̂ᵀZᵀWZΛ̂ + I)⁻¹Λ̂ᵀZᵀWX` on the caller's columns, the
    /// same solves as `condvar` (`q` without a hidden intercept, `p` the
    /// caller's width; an aliased column's entries are NaN). With `V` the
    /// fixed-effect covariance, `Cov(b̂_l, β̂) = −K_l·V`. Empty exactly when
    /// `condvar` is.
    pub(crate) condvar_x: Vec<f64>,
    /// The conditional covariance between the random effects of different
    /// groupings, which `condvar`'s per-level blocks leave out, on its scale;
    /// see [`CondvarCross`]. `None` on a fit with one grouping and wherever
    /// `condvar` is empty.
    pub(crate) condvar_cross: Option<CondvarCross>,
    /// Degrees of freedom of each coefficient `β̂_j`, length p: lmerTest's
    /// Satterthwaite df of the contrast `e_j` on a Gaussian LMM, and the
    /// residual df `n − p` (`p` the retained columns) on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated, whatever
    /// [`FitOptions::satterthwaite`] says. Filled on the target columns
    /// ([`FitOptions::se_for`]) of a Gaussian mixed fit whose pass ran
    /// and of a converged OLS or dispersion GLM fit; NaN on non-target and
    /// aliased columns (as `se`), on an OLS or dispersion GLM fit that did not
    /// converge (as `se`), on every other family (binomial, Poisson, negative
    /// binomial, every GLMM, a Gamma or inverse Gaussian GLM with φ held), when
    /// the pass was switched off, and when it failed
    /// ([`Note::SatterthwaiteCurvature`] with `nonfinite` set). Not `df`, which
    /// is the AIC parameter count.
    pub coef_df: Vec<f64>,
    /// What the Satterthwaite pass (or the residual df) left for
    /// [`Fit::contrast_t`] and [`Fit::contrast_f`]: on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated, the residual-only record (the
    /// full `C` and `ν = n − p`); on a Gaussian LMM (REML or ML), the pass's
    /// state, its switched-off record, the failed record of a fit with no
    /// finite endpoint, or the not-run record of a `Fit` the loop tier's
    /// `FitView::into_fit` built; `None` on every other family.
    pub(crate) satterthwaite: Option<crate::lmm::satterthwaite::SatterthwaiteState>,
    /// The design summary of a Gaussian fit whose only random term is one
    /// random intercept `(1 | g)`, which has no prior weights (an offset is
    /// allowed) and at least one fixed-effect column: `Some` on every such
    /// fit, whatever its criterion (REML or ML) and whether or not it
    /// converged, since the arrays depend on the design alone; `None` on
    /// every other fit, a `(1 | g)` fit with no fixed effects included.
    /// Copied out of the dense LMM accumulator where the `Fit` is assembled,
    /// so no fit computes anything extra for it. The arrays and their flat
    /// layout are on [`DesignSummary`].
    pub(crate) design_summary: Option<DesignSummary>,
}

/// The aggregates a `(1 | g)` fit's dense accumulator holds, as three flat
/// arrays. They are on the design the fit ran on: after aliased columns are
/// dropped, so `p` here is the retained column count, and in the fit's
/// orthogonalized basis `X̃ = X·B·U⁻¹` (`B` the column standardization, `U`
/// the fixed-effect factor), which spans the column space of the caller's `X`.
/// `K` is the highest primary level id with rows plus one, and a level below
/// it with no rows holds zeros.
///
/// - `counts`, length `K`: the row count of each level.
/// - `s`, length `p·K`, the `p × K` matrix `X̃ᵀZ` column-major:
///   `s[a·p + j] = Σ X̃ᵢⱼ` over the rows `i` of level `a`.
/// - `c`, length `p·p`, the `p × p` matrix `X̃ᵀX̃` column-major, both triangles
///   filled (it is symmetric): `c[j·p + k] = c[k·p + j] = Σᵢ X̃ᵢⱼ·X̃ᵢₖ`.
///
/// With `P₀` the residual projection of `X`, `ZᵀP₀Z = diag(counts) − sᵀc⁻¹s`.
// Read by the two-fit comparison (`formula::anova::Model::from_fit`), so a
// build without `formula` (or tests) holds the arrays unread.
#[cfg_attr(not(any(test, feature = "formula")), allow(dead_code))]
pub(crate) struct DesignSummary {
    pub(crate) counts: Vec<f64>,
    pub(crate) s: Vec<f64>,
    pub(crate) c: Vec<f64>,
}

/// The conditional covariance between the random effects of different
/// groupings (all of `Λ̂(Λ̂ᵀZᵀWZΛ̂ + I)⁻¹Λ̂ᵀ` that [`Fit::condvar`] does not
/// hold), as flat arrays on `condvar`'s scale and reported columns.
///
/// One grouping is the family grouping: each of its levels is a family,
/// together with its children when the model has a nested grouping (`m`
/// child slots per family, `m = 0` without one). Every other grouping is a
/// crossed, intercept-only one, and their levels form the tail: the
/// groupings in declaration order with the nested one skipped, except that
/// the declared primary stands at the family grouping's place when the two
/// differ, each grouping's levels in order.
///
/// - `family_grouping`: the declaration index of the family grouping; 0
///   unless the fit made a crossed extra its primary.
/// - `family`, per family `f`, the vech (column-major lower triangle, as
///   `varcorr`) of the `w × w` covariance of its `w = q + m` coefficients: the
///   level's `q` reported coefficients, then the nested grouping's slots
///   `f·m .. (f+1)·m`.
/// - `tail`, the vech of the `t × t` covariance of the tail levels.
/// - `coupling_at` (one entry per family plus one), `coupling_cols` and
///   `coupling`: family `f`'s entries `e` in `coupling_at[f]..coupling_at[f+1]`
///   name a tail level `coupling_cols[e]` and hold `w` values
///   `H_f[e] = coupling[e·w..(e+1)·w]`, one per family coefficient. Tail levels
///   no entry names have `H_f[e] = 0`. With `T` the `tail` matrix,
///   `Cov(coefficient a of f, tail level t) = −Σ_e H_f[e][a]·T[cols_e, t]`, and
///   between coefficients of two different families `f ≠ g`
///   `Σ_e Σ_e' H_f[e][a]·T[cols_e, cols_e']·H_g[e'][a']`.
///
/// Memory: `Σ_f w(w+1)/2 + t(t+1)/2` values plus `w` per entry, the entries
/// at most one per family and tail level its training rows share, and `t` at
/// most `MAX_CROSSED_LEVELS`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CondvarCross {
    pub(crate) family_grouping: usize,
    pub(crate) family: Vec<f64>,
    pub(crate) tail: Vec<f64>,
    pub(crate) coupling_at: Vec<u32>,
    pub(crate) coupling_cols: Vec<u32>,
    pub(crate) coupling: Vec<f64>,
}

/// What the kernels' factor walks return, in kernel coordinates: the
/// per-level blocks behind [`Fit::condvar`], the rows behind
/// [`Fit::condvar_x`], and the pieces `common::assemble_condvar_cross` maps to
/// [`Fit::condvar_cross`].
pub(crate) type CondvarBlocks = (Vec<f64>, Vec<f64>, Option<CondvarCross>);

/// Everything a fit reports about itself, reached as `fit.diagnostics`.
///
/// **Coverage is not uniform across routes**, and the per-field docs say where
/// each one is real. The short version: `converged` and `aliased` are filled
/// everywhere; `pinned` is real on every route that has variance components to
/// pin, while `boundary` distinguishes all three states only on the LMM and
/// GLMM routes; `notes` can only ever be raised by OLS, GLM and LMM — the GLMM
/// forms no factor to measure a pivot on. An absent note is therefore "not
/// detected", never "checked and clean".
///
/// `#[non_exhaustive]`: match with a trailing `..`. That is the whole point of
/// collecting these here — the six channels below arrived one at a time, each
/// arrival breaking `Fit`, and a seventh now costs nobody a major version.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Diagnostics {
    /// Whether the optimizer reached its convergence criterion. `false`
    /// usually means `se`/`vcov`/`dispersion`/`tau2`/`varcorr` are the
    /// NaN-fill (or empty, for `varcorr`) described on each of those fields —
    /// the exception is an LMM or GLMM fit that hit `MaxFunReached`: `se`,
    /// `vcov`, `dispersion`, `tau2` and `varcorr` all report finite values at
    /// that budget-exhausted endpoint. `stddev_se`, `fitted` and `ranef` stay
    /// gated on `converged` on both routes; `df` follows that same gate only
    /// on the GLMM route — an LMM `MaxFunReached` endpoint reports its usual
    /// nonzero `df` instead (see [`Fit::df`]).
    pub converged: bool,
    /// `true` iff the fit converged onto the θ boundary (≥ 1 diagonal variance
    /// component pinned at 0 — `boundary == AtBoundary`, OR a converged
    /// diagonal stddev negligible next to its fit's largest — see
    /// `Fit::has_negligible_component`), the same condition lme4's
    /// `isSingular` reports. `false` for OLS/GLM and for an LMM or GLMM
    /// `MaxFunReached` cap-out — a capped endpoint is reported as a point,
    /// not accepted onto the boundary, so the post-hoc negligible-component
    /// check does not run there even though `varcorr` is finite (both routes
    /// gate that check on `converged` for exactly this reason), and it never
    /// sets this flag even when its diagonals are near zero.
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
    /// A pinned θ diagonal marks a rank-deficient Σ̂, not necessarily a zero
    /// variance: component `i` has zero variance *conditional on the components
    /// before it in its block*. It is a zero variance component only when the
    /// rest of its Λ row is zero too. At a correlation boundary (`|ρ| = 1`) the
    /// trailing diagonal pins while that component's own variance (row `i` of
    /// `ΛΛ′`) stays positive, so read variances from `varcorr`, never from this
    /// flag.
    ///
    /// The flag means that because the pin loop is followed by a Σ-preserving
    /// re-factorization of each pinned RE block
    /// (`crate::lmm::canonicalize_pinned_blocks`): the reported Λ is the
    /// canonical lower Cholesky of Σ̂, so a pinned diagonal's whole column is
    /// exactly zero and the variance that used to sit below it has been folded
    /// into a trailing diagonal. Without that, a zero θ diagonal on a `q ≥ 2`
    /// block could sit above a live off-diagonal and flag a direction in which
    /// Σ̂ still varies.
    ///
    /// **Exception: AGQ (`nagq > 1`).** The flags come from the same canonical
    /// form, but the fit reports the search's own Λ, because the vector AGQ
    /// deviance depends on Λ and not only on Σ. So a pinned diagonal of the
    /// fitted θ̂ can sit above a live entry. The flags still describe Σ̂, as
    /// `varcorr` does.
    ///
    /// **Exception: the sparse route.** `canonicalize_pinned_blocks` skips a
    /// block with `q` past `MAX_PRIMARY_Q` (`crate::lmm::canonicalize_pinned_blocks`),
    /// which only the sparse route ever builds, so on that route a pinned
    /// diagonal can sit above a live off-diagonal after all.
    ///
    /// **Empty means nothing was pinned** — every route with variance
    /// components to pin fills this field on every converged fit: each view
    /// mapper reshapes the fitted mask through
    /// `common::materialize_diagnostics`, so there is no route left that knows
    /// a component pinned but declines to say which.
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
    /// Solver observations that are not one of the fixed channels above. Empty
    /// on a clean fit, and empty allocates nothing.
    pub notes: Vec<Note>,
    /// Gradient evaluations the BFGS outer search made, on the LMM and GLMM
    /// routes that search with it, Laplace and AGQ (listed on
    /// [`Note::GradientSearchFallback`]).
    /// [`Fit::n_eval`] does not count them, because a gradient reuses its
    /// point's value evaluation. 0 on every fit that ran no gradient search.
    /// This field and the two below describe the BFGS search even when BOBYQA
    /// finished the fit after it.
    pub n_grad: usize,
    /// Why the BFGS θ search stopped; `None` on every fit that ran no gradient
    /// search.
    pub outer_stop: Option<OuterStop>,
    /// The BFGS decrement `½·gᵀHg` at the search's endpoint (deviance units),
    /// the quasi-Newton estimate of the decrease left. NaN on every fit that
    /// ran no gradient search.
    pub outer_decrement: f64,
}

/// Where the accepted θ sits in its parameter space.
///
/// Every θ-carrying route distinguishes all four: the LMM over either kernel
/// and the GLMM over every layout report the state the search reached, cap-out
/// (`NoOptimum`) included. OLS and GLM have no θ and always report `Interior`,
/// and so does a `Fit` that failed before any search ran (the degenerate `n`/`p`
/// guards), which says so through `converged: false`.
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
    /// ≥ 1 θ coordinate ended on the upper limit of its search box
    /// (`crate::lmm::THETA_HI`, 1000 on the internal scale), on a fit that
    /// would otherwise read `Interior`: the search stopped on that wall, not at
    /// an optimum. Always comes with [`Note::VarianceAtLimit`], which an
    /// `AtBoundary` or `NoOptimum` fit also carries when it reaches the limit.
    /// `converged`, `singular` and the estimates are the search's own.
    ///
    /// Cold path only: the loop tier's `FitDiagnostics` carrier has no code for
    /// it, so a loop-tier read of such a fit says interior.
    UpperLimit,
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
    /// A GLMM PIRLS inner solve ran the full `PIRLS_MAX_ITERS` (200) cap without
    /// satisfying its convergence band — never a failure surfaced any other way
    /// (that is the `(NaN, NaN, NaN, false)` halving-exhaustion/Cholesky-failure
    /// case, which the fit already rejects or NaN-fills through the usual
    /// channels). The cap stays 200. A note where `final_eval` is `false` costs
    /// nothing observable — that inner solve was a rejected trial point, not
    /// what the fit reports; a note where `final_eval` is `true` means the
    /// reported numbers rest on a PIRLS solve that never settled inside the cap.
    /// FD-Hessian SE evals are excluded — only fit-path evals count.
    PirlsExhausted {
        /// How many fit-path BOBYQA objective evals hit the cap over the course
        /// of this fit (0 if `final_eval` alone is what fired).
        evals: u32,
        /// Whether the FINAL re-evaluation at the reported γ̂ itself hit the
        /// cap — the case that matters, since that solve's ũ/W̃ feed the
        /// reported estimates directly rather than being a rejected trial
        /// point. That re-evaluation runs on a budget-exhausted outer exit
        /// too, not only on a converged one.
        final_eval: bool,
    },
    /// The negative-binomial GLM's alternation between β̂ (IRLS at fixed θ) and θ̂
    /// (profile maximisation at fixed μ̂) ran its full `NB_MAX_OUTER` rounds
    /// without meeting its relative-change tolerance on θ. The fit is reported at
    /// the last θ, with β̂ and the SEs refit there, so the numbers are one
    /// model's; that θ may still be short of the profile optimum. `converged`
    /// keeps the last inner IRLS fit's flag. Mixed negative-binomial models
    /// search ln θ inside their outer BOBYQA instead and never raise this.
    NbShapeUnsettled {
        /// Alternation rounds run (the cap, `NB_MAX_OUTER`, in production).
        rounds: u32,
    },
    /// `FitOptions::wald_se == WaldSe::Hessian` was requested, but the
    /// finite-difference joint Hessian came back non-positive-definite (or a
    /// perturbed deviance evaluation was non-finite), so the SE pass fell back
    /// to the RX/Schur route instead. `Fit::se`/`vcov` are still filled (from
    /// the fallback), but `Fit::stddev_se` is NaN — no θ-block SE comes out of
    /// a Schur inverse. Never fires under `WaldSe::Rx`, which never attempts
    /// the joint Hessian.
    HessianSeFallback,
    /// The GLMM's `ExactProfile` route (the exact β-profile outer search,
    /// `src/glmm/mod.rs`) ended not converged, so the fit reran once on the
    /// `PqlThenJoint` route instead, from the same start. The reported
    /// estimates come from that rerun when it converged; when it also failed
    /// to converge, they come from whichever of the two attempts reached the
    /// lower deviance — `converged` stays `false` either way, and `n_eval`
    /// counts both attempts. Never raised for a shape that does not route
    /// `ExactProfile` (`Joint` and `PqlThenJoint` have no fallback of their
    /// own).
    ExactProfileFallback,
    /// The gradient search (BFGS: LMMs whose random slopes are all on one
    /// grouping; binomial, Poisson and Gamma Laplace GLMMs with at least two θ
    /// on the blocked or structured layout; binomial logit and probit and
    /// Poisson log AGQ GLMMs with one scalar random intercept, no other
    /// grouping and under 50 rows per cluster, cold or from a warm θ of at
    /// least 0.05) did not converge, so BOBYQA, the
    /// derivative-free search every other model uses, continued from the best
    /// point it had reached, with its own full evaluation budget. The reported
    /// estimates and `converged` are that search's. `n_eval` counts both.
    GradientSearchFallback,
    /// A GLMM search reached its endpoint, but no fixed-effect covariance could
    /// be formed there: the β information (the Rx Schur complement, and under
    /// `WaldSe::Hessian` the joint Hessian before it) is not positive definite
    /// to rounding. The estimates, the variance components, the deviance and
    /// `converged` are the search's own; `Fit::se`, `Fit::vcov` and
    /// `Fit::stddev_se` are NaN. Reached on a near-duplicate fixed-effect
    /// column the alias gate keeps, and on fits whose working weights span
    /// many decades.
    ///
    /// On a GLM: the fit converged, but the QR of `√W·X` at its last step has
    /// a zero or non-finite diagonal at a column that is not zero under the
    /// weights, so that column cannot be separated from the ones before it.
    /// [`Note::IllConditioned`] comes with it and names the column (`pivot`
    /// 0); that column's `Fit::beta` is NaN, since the fit held it at its
    /// start value. The other estimates, the deviance, `loglik`, `fitted` and
    /// `converged` stand; `Fit::se` and `Fit::vcov` are NaN.
    SeUnavailable,
    /// The response is not within `1e-3` of an integer count — the raw value
    /// for Poisson/negative-binomial, the derived successes `y·wᵢ` for
    /// binomial. The fit still runs (the deviance and score are continuous in
    /// `y`), but the reported numbers describe a distribution that assumes
    /// integer counts. The `1e-3` tolerance matches R's
    /// `binomial()$initialize` non-integer-count check; R's Poisson/NB
    /// warning is unrelated (it comes from `dpois`/`dnbinom` inside
    /// `poisson()$aic`/`negative.binomial()$aic`, at a much tighter
    /// `1e-7·max(1,|y|)`, so R warns on e.g. `y = 2.0005` where glmm stays
    /// silent). Never raised for Gaussian, Gamma or inverse-Gaussian, whose
    /// response is continuous.
    NonIntegerResponse {
        /// Rows whose response (successes, for binomial) misses the nearest
        /// integer by more than the tolerance.
        rows: u32,
    },
    /// A random-effect term's grouping factor has only one observed level, so
    /// no between-group variance is identifiable from it — lme4 errors here
    /// ("grouping factors must have > 1 sampled level"); this crate instead
    /// drops the term and fits the rest of the model. When every random-effect
    /// term is dropped this way, the model fits with `re: None`, the same as a
    /// formula that declared no random effects at all.
    ///
    /// Raised by the formula frontend, not by a solver: the level count is a
    /// property of the lowering, not of any fitted model. Reached through
    /// [`crate::formula::Lowered::notes`].
    SingleLevelGroupingDropped {
        /// The grouping factor, as the formula spells it.
        grouping: String,
    },
    /// A random-effect term has at least as many random effects as the fit
    /// has rows (more than the rows, on a family other than Gaussian), so its
    /// variance parameters probably cannot be estimated from these data. The
    /// count is the term's observed levels times its columns (the intercept,
    /// one per numeric slope, one per non-base level of a factor slope):
    /// `(1 + x | g)` on 10 levels has 20. lme4 stops on such a model by default
    /// (`check.nobs.vs.nRE`: `lmer` when the rows are at most the count,
    /// `glmer` when they are below it, so an observation-level `(1 | obs)`
    /// passes on a GLMM). Each term is checked against all rows on its own;
    /// the counts are not summed across terms. The fit runs as usual.
    ///
    /// Raised by the formula frontend, not by a solver, for the same reason
    /// [`Note::SingleLevelGroupingDropped`] is. A term dropped by
    /// [`Note::SingleLevelGroupingDropped`] is not checked. Reached through
    /// [`crate::formula::Lowered::notes`].
    FewRowsForRandomEffects {
        /// The term's left-hand side with its intercept written out, `"1"` or
        /// `"1 + x"`.
        terms: String,
        /// The grouping factor, as the formula spells it.
        grouping: String,
        /// The term's random effects: observed levels times columns.
        random_effects: u32,
        /// The rows of the fit.
        rows: u32,
    },
    /// A GLMM's deviance is too large for `f64` to resolve its optimum: its
    /// estimated rounding error at the endpoint exceeds 1e-2 deviance units,
    /// which moves θ̂ by about a tenth of its standard error, and the search
    /// sees a noise several times that. β̂, θ̂ and the deviance may sit off the
    /// optimum; `converged` and every estimate are the search's own. Two
    /// sources add up in the estimate: the deviance itself, about 1e13 or more
    /// (a model that misses a large effect at counts in the billions), and,
    /// with more than one grouping factor, `log|A|`, whose rounding grows with
    /// the extra factors' `θ²·Σwᵢ` (crossed or nested intercepts at counts in
    /// the trillions). A model that captures the missing effect removes the
    /// first source.
    PrecisionLimited,
    /// The Satterthwaite pass of a Gaussian LMM ([`Fit::coef_df`]) met a
    /// curvature it cannot take at face value. Raised when the Hessian of the
    /// profiled deviance over the free variance parameters had directions the
    /// pseudo-inverse left out (flat to `1e-8` of its largest eigenvalue, or
    /// negative), as lmerTest warns; when the df were computed at an endpoint
    /// that did not converge (`MaxFunReached`); and when the pass failed: a
    /// deviance or covariance on the pass's stencil was not finite, the
    /// eigendecomposition of that Hessian failed, or the fit has no finite
    /// endpoint at all. A failed pass fails for the whole fit: every `coef_df`
    /// is NaN and the contrasts return
    /// [`crate::inference::contrast::NoDfReason::Failed`].
    SatterthwaiteCurvature {
        /// Directions of the Hessian dropped as flat.
        dropped: u32,
        /// Directions of the Hessian below `−1e-8` of its largest eigenvalue,
        /// not inverted.
        negative: u32,
        /// Whether the endpoint the df were computed at converged.
        converged: bool,
        /// The pass failed: a deviance or covariance on the stencil was not
        /// finite, the Hessian's eigendecomposition failed, or the fit has no
        /// finite endpoint.
        nonfinite: bool,
    },
    /// The fixed-effects IRLS stopped on separation: the deviance settled
    /// while the last step still moved some linear predictor by a large
    /// amount, so at least one coefficient runs off to ±∞ (binomial, Poisson,
    /// negative binomial). `converged` is `false` and `beta` holds the last
    /// step. On a GLM fit that is the fit itself; on a GLMM fit it is the GLM
    /// the search starts from, so the search never ran: the random-effect
    /// outputs, the SEs and the deviance are NaN.
    Separated,
    /// A random-effect parameter ended on the upper limit of the θ search box
    /// (`crate::lmm::THETA_HI`, 1000 on the search's internal scale, which on
    /// a Gamma inverse-link GLMM is θ in the units of the rescaled response
    /// the search ran on, not the reported θ): the estimates sit
    /// on that limit, not at an optimum. On a Gaussian model this usually means
    /// the residual variance is estimated at or near zero (θ is a standard
    /// deviation relative to the residual one); on other families the random
    /// effect has no finite estimate. Raised on any fit with a finite endpoint
    /// (sets [`Boundary::UpperLimit`] when the boundary would otherwise read
    /// `Interior`); `converged` and `singular` are not changed.
    VarianceAtLimit,
    /// A prediction on new rows ([`crate::predict`]) met grouping levels the
    /// fit has no conditional mode for: an id at or past the grouping's
    /// [`Fit::ranef_levels`] count. Those rows take a zero random effect for
    /// this grouping, and nothing from it in their standard error, while the
    /// row's other groupings keep theirs, as lme4's `allow.new.levels = TRUE`
    /// does. One note per grouping with such rows, in declaration order.
    /// Raised only when [`crate::PredictOptions::allow_new_levels`] is set (the
    /// default); without it the same rows are a [`FitError::Ids`]. Never raised
    /// by a fit.
    NewGroupingLevels {
        /// The grouping's declaration index: 0 the primary grouping, `g + 1`
        /// the extra grouping `g`.
        grouping: u32,
        /// The 0-based rows of the new design with an unseen level, ascending.
        rows: Vec<u32>,
    },
}

impl Diagnostics {
    /// The diagnostics a `Fit` built without a solver carrier can honestly fill:
    /// `boundary` back-derived from `singular`, nothing pinned, no notes, no
    /// aliased column. Its callers are the NaN-fill returns — the degenerate
    /// `n`/`p` guards, the NB alternation's seed fit, the unfittable-random-slope
    /// return — which stand in for a search that never ran and all pass
    /// `converged: false`. Every fitting route goes through
    /// `common::materialize_diagnostics` instead, which has a carrier to read.
    ///
    /// `pinned` stays empty here because those callers have no varcorr to align
    /// a grid against; [`pinned_flags`] is what places the bits when there is
    /// one.
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
            notes: vec![],
            n_grad: 0,
            outer_stop: None,
            outer_decrement: f64::NAN,
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
    /// Changes no returned estimate. `false` when `varcorr` is empty
    /// (non-mixed-model fit, or a non-converged fit that left it unassembled).
    ///
    /// The comparison runs on the fit's internal standard deviations
    /// `kernel_sd` (`LmmGroupings::kernel_stddevs`): the row norms of the
    /// kernel's Λ, one block per grouping, primary first. σ cancels in the
    /// ratio, so they stand in for `σ·‖row‖`. The reported standard deviations
    /// are not commensurate with each other: an intercept's is in response units
    /// and a slope's is in response per covariate unit, so their raw ratio moves
    /// when the covariate is re-expressed and the verdict would follow it. In
    /// the kernel's coordinates every component is "how much variation this
    /// term contributes" on the standardized design, and the check means what
    /// it says. On a predictor whose mean is far from zero the intercept's is
    /// therefore taken at the mean, not at x = 0.
    pub(crate) fn has_negligible_component(&self, kernel_sd: &[Vec<f64>]) -> bool {
        if self.varcorr.is_empty() {
            return false;
        }
        let max_sd = kernel_sd.iter().flatten().cloned().fold(0.0_f64, f64::max);
        if max_sd <= 0.0 {
            return false;
        }
        kernel_sd
            .iter()
            .flatten()
            .any(|&s| s <= SINGULAR_REL_TOL * max_sd)
    }
}

/// Options for `fit`. Carries the safe, defaulted method knobs: unlike
/// a warm start these do not silently move the estimate based on a caller's guess.
///
/// `FitOptions` is `#[non_exhaustive]`: outside this crate a struct literal,
/// including `..` update syntax, does not compile.
///
/// ```compile_fail
/// let o = glmm::FitOptions { se_for: glmm::SeFor::Only(vec![0]), ..Default::default() };
/// ```
///
/// Start from [`FitOptions::default()`] (or a clone of existing options) and
/// assign fields instead:
///
/// ```rust
/// let mut o = glmm::FitOptions::default();
/// o.se_for = glmm::SeFor::Only(vec![0]);
/// ```
// Non-exhaustive so a new knob is an additive change, not a break.
#[non_exhaustive]
#[derive(Clone)]
pub struct FitOptions {
    /// Which coefficients get a standard error. Default [`SeFor::All`].
    pub se_for: SeFor,
    /// Wald-SE denominator (relocated from `ModelSpec`). Default `Hessian`.
    pub wald_se: WaldSe,
    /// Adaptive Gauss–Hermite node count (relocated from `ModelSpec`). Default 1
    /// (= Laplace). Must be odd and in `1..=MAX_NAGQ`; `>1` is honored on a
    /// binomial/Poisson/negative-binomial/Gamma GLMM with a single grouping factor and `q_p ≤ 3` random
    /// effects per group (scalar intercept → `agq::agq_deviance`, vector RE →
    /// `agq::agq_deviance_vec`, a `k^q_p` product grid). `q_p ≥ 4` is refused
    /// with [`FitError::Unsupported`] (a temporary cost/oracle boundary), as
    /// are the other ineligible non-Gaussian mixed shapes. `nagq > 1` is
    /// ignored on Gaussian models and on models without random effects: the
    /// fit runs at `nagq = 1` with no note attached, and the ports warn.
    pub nagq: u8,
    /// Gamma/inverse-Gaussian dispersion directive (relocated from
    /// `Family::Gamma`). `None` = estimate φ (the Pearson moment,
    /// `Σwᵢrᵢ²/(n−p)`, on a GLM; maximum likelihood on a GLMM); `Some(v)` =
    /// hold φ fixed at `v`. Ignored by the φ ≡ 1 families.
    pub dispersion: Option<f64>,
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
    /// `cluster_rows` path): `false` runs scalar AGQ's original node-outer loop
    /// with no per-fit `ClusterRowIndex` build (vector AGQ is cluster-outer on
    /// every build, and a serial build gives it the index once per fit). Also
    /// gates the FD-Hessian grids (dense and sparse): `false` keeps the serial
    /// cell-by-cell loop. See the design doc above
    /// for the full rationale, including why nesting this under an outer `rayon`
    /// batch loop is safe (one shared work-stealing pool) in a way naive OS-thread
    /// or BLAS-thread nesting is not.
    pub parallel_inner: bool,
    /// Run the Satterthwaite pass on a Gaussian LMM fit ([`Fit::coef_df`],
    /// [`Fit::contrast_t`], [`Fit::contrast_f`]): one finite-difference pass
    /// at θ̂, over the profiled deviance (or over its analytic gradient where
    /// the fit's per-group gradient is armed), after the fit and without a
    /// refit.
    /// Default `true`. `false` skips it: `coef_df` stays NaN and the contrasts
    /// return [`crate::inference::contrast::NoDfReason::Disabled`]. No other
    /// family reads it, and no other field of the fit depends on it.
    pub satterthwaite: bool,
    /// Form the conditional variances of the random effects ([`Fit::condvar`])
    /// on a mixed model, which a conditional standard error or future bound
    /// from [`crate::predict`] reads. Default `false`: they cost a set of
    /// solves after the fit (on a GLMM, a re-solve of the mode first) that
    /// grows with the level count, up to as much as the fit itself on a
    /// crossed model with many levels, and a caller fitting in a loop rarely
    /// reads them. With `false`, `condvar` is empty and such a prediction
    /// returns [`FitError::Unsupported`]; no other field of the fit depends on
    /// it.
    pub condvar: bool,
    /// The estimator of a Gaussian mixed model: restricted maximum likelihood
    /// (REML) or maximum likelihood (ML). Default `None`, which resolves to
    /// the usual estimator for the model:
    ///
    /// | Model | `None` | `Some(true)` | `Some(false)` |
    /// |---|---|---|---|
    /// | Gaussian, random part | REML | REML | ML |
    /// | Gaussian, no random part (OLS) | ML log-likelihood | REML log-likelihood | ML log-likelihood |
    /// | any other family | ML | ML | ML |
    ///
    /// On OLS the value moves only the reported log-likelihood (see
    /// [`Fit::loglik`]) and [`Fit::reml`]: β̂, the SEs and `σ̂² = RSS/(n − p)`
    /// are the same under all three. It is read when the `Fit` is assembled,
    /// from the options handed to `FitView::into_fit`, and is not frozen on an
    /// OLS `FitWorkspace`, since the solve does not depend on it. On a GLM or
    /// GLMM (Laplace or AGQ) the fit is by maximum likelihood whatever the
    /// value, so `Some(true)` is ignored there, as [`FitOptions::dispersion`]
    /// is on the families with no dispersion. `Some(false)` off the Gaussian
    /// mixed model, and `Some(true)` on a GLM or GLMM, give a `Fit` bitwise
    /// identical to the `None` fit, with no note attached: the ports warn
    /// about an ignored value, the kernel does not. "Random part" means the
    /// model actually fitted: a grouping dropped for having one level leaves
    /// an OLS or GLM fit, and its row applies.
    pub reml: Option<bool>,
}

/// Which fixed-effect coefficients a fit computes standard errors for
/// ([`FitOptions::se_for`]). A coefficient without one has a NaN `se` slot and
/// a NaN row and column in `vcov`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SeFor {
    /// Every coefficient.
    #[default]
    All,
    /// Only these columns of `x` (0-based). `Only(vec![])` computes none,
    /// which skips the work of the covariance; on a GLMM under
    /// [`WaldSe::Hessian`] that is the joint Hessian.
    Only(Vec<u32>),
}

impl SeFor {
    /// The column indices this selects on a `p`-column design.
    pub(crate) fn columns(&self, p: usize) -> Cow<'_, [u32]> {
        match self {
            SeFor::All => Cow::Owned((0..p as u32).collect()),
            SeFor::Only(cols) => Cow::Borrowed(cols),
        }
    }
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            se_for: SeFor::All,
            wald_se: WaldSe::Hessian,
            nagq: 1,
            dispersion: None,
            parallel_inner: false,
            satterthwaite: true,
            condvar: false,
            reml: None,
        }
    }
}

/// A caller input the fit refuses, returned by [`fit`] and [`fit_with_start`]
/// before any fitting. Each variant carries a message naming the field, and
/// the row (0-based) or the row and column where the check failed; its
/// [`Display`](std::fmt::Display) is that message alone.
///
/// A numerical failure is not a `FitError`: it is reported inside the
/// [`Fit`] (`converged`, the notes, NaN estimates).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// The data do not have the shape `n` and `p` say (`x` not `n*p` long, `y`
    /// not `n` long, a non-finite `x`), or a random slope names a column past
    /// `p`.
    Shape(String),
    /// A response outside the family's support: a non-finite `y`, a Poisson
    /// or negative-binomial `y < 0`, a binomial `y` outside `[0, 1]`, a Gamma
    /// or inverse-Gaussian `y <= 0`.
    ResponseOutOfSupport(String),
    /// Prior weights of the wrong length, or not finite and `> 0`.
    Weights(String),
    /// An offset of the wrong length, or not finite.
    Offset(String),
    /// Grouping ids that do not match the model: missing on a model with
    /// random effects, given on a model without them, or of the wrong length
    /// or count.
    Ids(String),
    /// A warm start of the wrong width, with a non-finite entry, or with a
    /// dispersion that is not finite and `> 0`.
    Start(String),
    /// A [`FitOptions`] value no model can use: an even or zero `nagq`, a
    /// held `dispersion` that is not finite and `> 0`, a [`SeFor::Only`]
    /// column past `p`.
    OptionValue(String),
    /// A model the crate does not fit: a mixed inverse-Gaussian model,
    /// `nagq > 1` on a non-Gaussian mixed model outside the quadrature shapes
    /// (one grouping factor, at most 3 random effects per group), or a second
    /// `NestedWithin` grouping.
    Unsupported(String),
    /// A value past a fixed capacity of the kernels: `nagq` above
    /// [`MAX_NAGQ`](crate::consts::MAX_NAGQ).
    Envelope(String),
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (FitError::Shape(m)
        | FitError::ResponseOutOfSupport(m)
        | FitError::Weights(m)
        | FitError::Offset(m)
        | FitError::Ids(m)
        | FitError::Start(m)
        | FitError::OptionValue(m)
        | FitError::Unsupported(m)
        | FitError::Envelope(m)) = self;
        f.write_str(m)
    }
}

impl std::error::Error for FitError {}

/// Fit `spec` to `data` — the default real-data entry, with a cold start.
/// Random-effect level ids are supplied per row through [`Data::ids`]
/// ([`GroupIds`]); every level count is derived from them, so `ModelSpec`
/// stays structure-only. Exactly [`fit_with_start`] with an empty
/// [`StartValues`].
///
/// Dispatches on `(family, re.is_some())` (`Gaussian`→OLS/LMM,
/// `Binomial`→GLM/GLMM, Poisson/Gamma/NB likewise; `re: None`→fixed-only,
/// `Some`→mixed). Binomial counts are fit as expanded 0/1 Bernoulli rows (the
/// kernel is Bernoulli) — except when `y` is an aggregated success PROPORTION
/// and [`Data::weights`] carries the trial count, which fits directly
/// without the expansion. That aggregated form is supported on every
/// (family, RE, solver) path, including AGQ (`nagq > 1`) on its binomial/Poisson/NB/Gamma
/// shapes (see [`Data::weights`]'s support matrix).
///
/// # Errors
///
/// A [`FitError`] when the data, the model or the options are inputs no fit
/// can take (see its variants), checked before anything is fitted. Numerical
/// failures are not errors: they signal via `Fit { converged: false, .. }`
/// with NaN-filled estimates — including the degenerate-data cases: a
/// rank-deficient design has its aliased columns dropped and the reduced
/// model fit (`Fit::aliased` flags them, their β/se are NaN), and a design
/// whose aliased column is ALSO an RE slope — unfittable, since the slope has
/// no column left to point at — returns all-NaN with `converged: false`
/// rather than an error.
///
/// # Examples
///
/// Fixed-only OLS (`re: None`): `y = 2x` exactly, so the fitted slope is 2.
///
/// ```rust
/// use glmm::{Data, Family, FitOptions, ModelSpec};
///
/// let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]; // n=3, p=2, row-major [intercept, x]
/// let y = vec![0.0, 2.0, 4.0];
/// let model = ModelSpec::fixed(Family::Gaussian);
/// let fit = glmm::fit(&Data::new(&x, &y, 3, 2), &model, &FitOptions::default())?;
/// assert!(fit.converged());
/// assert!((fit.beta[1] - 2.0).abs() < 1e-9);
/// # Ok::<(), glmm::FitError>(())
/// ```
///
/// A mistake in the inputs comes back as an error, not a panic:
///
/// ```rust
/// use glmm::{Data, Family, FitError, FitOptions, ModelSpec, PoissonLink};
///
/// let x = vec![1.0; 3];
/// let y = vec![1.0, -1.0, 2.0];
/// let model = ModelSpec::fixed(Family::Poisson { link: PoissonLink::Log });
/// let Err(err) = glmm::fit(&Data::new(&x, &y, 3, 1), &model, &FitOptions::default()) else {
///     panic!("a negative Poisson count is refused");
/// };
/// assert!(matches!(err, FitError::ResponseOutOfSupport(_)));
/// assert_eq!(err.to_string(), "Poisson response must be >= 0 (row 1)");
/// ```
pub fn fit(data: &Data<'_>, spec: &ModelSpec, opts: &FitOptions) -> Result<Fit, FitError> {
    validate(data, spec, None, opts)?;
    Ok(fit_checked(data, spec, None, opts))
}

/// [`fit`] from a warm start. One fit, one answer — the start only shortens
/// the path. The start threads `beta`+`theta` into the LMM/GLMM kernels, and
/// `dispersion` into the negative-binomial shape search (GLM and GLMM) and the
/// Gamma GLMM's `ln φ` coordinate; the other fixed-only (OLS/GLM) fits ignore
/// it (see [`crate::StartValues`]). Either vector may be left empty to
/// cold-start that component alone, and an all-empty start is [`fit`].
///
/// # Errors
///
/// As [`fit`], plus [`FitError::Start`]: `start.beta.len()` outside `{0, p}`,
/// `start.theta.len()` outside `{0, n_theta}` (the model's RE θ width), a
/// non-finite entry in `start.beta` or `start.theta`, or a `start.dispersion`
/// that is not finite and `> 0`.
///
/// # Examples
///
/// An intercept-only Gaussian LMM (3 clusters, `n_theta = 1`) warm-started off a
/// perturbed θ:
///
/// ```rust
/// use glmm::{Data, Family, FitOptions, GroupIds, ModelSpec, Re, StartValues};
///
/// let x = vec![1.0; 6]; // n=6, p=1 (intercept-only design)
/// let y = vec![1.0, 2.0, 1.5, 1.2, 2.1, 1.4];
/// let model = ModelSpec::mixed(Family::Gaussian, Re::intercept());
/// let ids = GroupIds::new(vec![0, 1, 2, 0, 1, 2]);
/// let start = StartValues::new(vec![0.0], vec![1.0]);
/// let data = Data::new(&x, &y, 6, 1).ids(&ids);
/// let fit = glmm::fit_with_start(&data, &model, &start, &FitOptions::default())?;
/// assert_eq!(fit.beta.len(), 1);
/// # Ok::<(), glmm::FitError>(())
/// ```
pub fn fit_with_start(
    data: &Data<'_>,
    spec: &ModelSpec,
    start: &StartValues,
    opts: &FitOptions,
) -> Result<Fit, FitError> {
    validate(data, spec, Some(start), opts)?;
    Ok(fit_checked(data, spec, Some(start), opts))
}

/// The fit of [`fit`] and [`fit_with_start`] on inputs [`validate`] accepted.
/// `pub(crate)` for the callers that refit a model they already know is
/// valid (the rank-deficiency salvage, the per-term likelihood-ratio refits),
/// which skip the check.
pub(crate) fn fit_checked(
    data: &Data<'_>,
    spec: &ModelSpec,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Fit {
    let Design {
        x,
        n,
        p,
        ids,
        weights,
        ..
    } = data.design;
    let y = data.y;
    // Non-integer counts: still fittable (the deviance and score are
    // continuous in y), but the reported numbers describe a distribution that
    // assumes integer counts, so R's `glm` warns instead of refusing —
    // matched here rather than refused. Binomial's count is the derived
    // successes `y·wᵢ`, not `y` itself. Computed before dispatch so it applies
    // whether the design later aliases columns or not; attached to the
    // returned `Fit` below (and, on the aliased path, by the recursive
    // `fit_checked` call inside `fit_rank_deficient`, which sees the same `y`).
    let non_integer_response_rows = count_non_integer_response(y, spec.family, weights);
    // Rank-deficiency salvage: drop fixed-effect columns aliased on earlier
    // columns and fit the reduced model (lme4 behavior — NA coefficient, still
    // converges). Path-agnostic: preprocesses the fixed design X before the
    // solver dispatch, so it serves OLS/GLM/LMM/GLMM alike. Runs after the
    // input check (full p still valid) and short-circuits into a reduced re-entry.
    // Tests the UNWEIGHTED X'X at `ALIAS_EPS`, i.e. only genuine redundancy —
    // columns indistinguishable in f64, where there is no separate coefficient
    // to estimate. This is the only place a column is ever dropped. A design
    // that is merely ill-conditioned has a unique, computable answer, and the
    // estimator routes fit it in full and record how badly conditioned it was.
    if n > 0 && p > 0 {
        let aliased = detect_aliased(x, n, p);
        if aliased.iter().any(|&a| a) {
            return fit_rank_deficient(data, spec, start, opts, &aliased);
        }
    }
    // Dispatch over the unified fit core: size the spec from the ids, classify
    // and allocate the per-shape workspace, solve on it, map the lean view to
    // the full `Fit`. The workspace owns the ids, so the data handed to
    // `fit_on` carries none.
    let no_ids = GroupIds::default();
    let mut ws = core::build_workspace(spec, ids.unwrap_or(&no_ids), n, p, opts);
    let rows = Data {
        design: Design {
            ids: None,
            ..data.design
        },
        y,
    };
    let view = core::fit_on(&mut ws, &rows, start, opts);
    let mut fit = view.into_fit(&rows, spec, opts);
    // After `into_fit`, which is the last read of the kernel's factor for the
    // fit's own outputs: the pass leaves that factor at its last stencil
    // point (on the shifted response), and nothing reads `ws` after it.
    core::attach_satterthwaite(&mut ws, &mut fit, opts);
    if non_integer_response_rows > 0 {
        fit.diagnostics.notes.push(Note::NonIntegerResponse {
            rows: non_integer_response_rows,
        });
    }
    fit
}

/// Rows whose response is not within [`NON_INTEGER_TOL`] of an integer —
/// the raw count for Poisson/negative-binomial, or the derived successes
/// `y·wᵢ` for binomial (unit weights ⇒ Bernoulli, always integer). `0` for
/// Gaussian, Gamma and inverse-Gaussian, whose response is continuous.
/// [`NON_INTEGER_TOL`] matches R's `binomial()$initialize` tolerance; R's
/// Poisson/NB non-integer warning uses a different, tighter check (see
/// [`Note::NonIntegerResponse`]).
fn count_non_integer_response(y: &[f64], family: Family, weights: Option<&[f64]>) -> u32 {
    const NON_INTEGER_TOL: f64 = 1e-3;
    let off = |v: f64| (v - v.round()).abs() > NON_INTEGER_TOL;
    match family {
        Family::Poisson { .. } | Family::NegativeBinomial { .. } => {
            y.iter().filter(|&&v| off(v)).count() as u32
        }
        Family::Binomial { .. } => y
            .iter()
            .enumerate()
            .filter(|&(i, &v)| off(v * weights.map_or(1.0, |w| w[i])))
            .count() as u32,
        Family::Gaussian | Family::Gamma { .. } | Family::InverseGaussian { .. } => 0,
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
/// handled by `check_model_shape` (which runs first and refuses the shapes
/// no kernel fits).
///
/// On top of the structural caps, slope-carrying extra groupings always route
/// `Sparse`, and so does a total `Crossed` level count past
/// `MAX_CROSSED_LEVELS` — see the comments at the clauses.
///
/// The level-count clause reads the crossed counts of `sized`
/// ([`SizedSpec::extra_counts`]).
pub(crate) fn classify_design(sized: &SizedSpec, _nagq: u8) -> Solver {
    let Some(re) = sized.spec.re.as_ref() else {
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
    // blocked and structured kernels build intercept-only extras exclusively
    // (`build_packed_m` carries a debug_assert), so Sparse — whose PIRLS applies
    // full q_g×q_g Λ-blocks per extra level — is the only implemented route.
    let slope_extras = re.extra_groupings.iter().any(|g| !g.slopes.is_empty());
    // Many-level crossed extras route Sparse: every crossed level lands in the
    // dense tail (`reml_deviance`'s `t_dim`; the dense GLMM path's Schur
    // complement likewise), which is cubic in the TOTAL crossed level count —
    // hence the sum over factors, not a per-factor cap. See MAX_CROSSED_LEVELS
    // for the measured blowup and the placeholder threshold's rationale.
    let crossed_levels: usize = re
        .extra_groupings
        .iter()
        .zip(&sized.extra_counts)
        .map(|(g, &levels)| match g.relation {
            GroupingRelation::Crossed => levels,
            GroupingRelation::NestedWithin => 0,
        })
        .sum();
    if over || slope_extras || crossed_levels > crate::consts::MAX_CROSSED_LEVELS {
        Solver::Sparse
    } else {
        Solver::NoZ
    }
}

// ---------------------------------------------------------------------------
// Re-exports for consumers outside `fit` (mirrors `src/glmm/mod.rs`'s
// mod+re-export convention): `sparse.rs` reaches these as `crate::fit::X`,
// and the `loop_advanced` cargo feature's `crate::loop_advanced` re-exports the
// loop tier from here.
// ---------------------------------------------------------------------------

// The one diagnostics carrier. Always `pub` (it is `FitView::diagnostics`'s
// return type), re-exported only for the loop tier — the stable path reads it
// through `Fit`, never directly.
#[cfg(feature = "loop_advanced")]
pub use common::FitDiagnostics;
pub use common::OuterStop;
pub(crate) use glm::{
    nb_count_groups, nb_profile_loglik, nb_saturated_loglik_grouped, NB_THETA_HI, NB_THETA_LO,
};
// Test-only since the GLMM NB coordinate stopped seeding from it (it seeds from
// `fit_glm_nb`'s θ̂ instead): `fit_glm_nb_capped` calls it inside `glm.rs`, so the
// only cross-module readers left are the tests that pin it.
#[cfg(test)]
pub(crate) use glm::nb_theta_moment_seed;
// The GLMM β cold start. `fit/glmm.rs` calls it directly; the re-export serves
// the tests that seed a hand-built workspace the same way a fit does.
#[cfg(test)]
pub(crate) use glmm::glm_warm_start_beta;
// The packed-row cross-check entry point: `src/sparse/tests.rs` fits an
// in-envelope design both ways through it.
#[cfg(test)]
pub(crate) use glmm::fit_glmm_packed;
#[cfg(test)]
pub(crate) use lmm::{fit_mle_noz_pub, fit_mle_sparse_pub};

/// Test-only: [`fit`]'s fit of a full-rank mixed design, returned with the
/// workspace it ran on, so a test can read what the fit left there
/// (`FitWorkspace::lmm_parts`). `data` carries the ids.
#[cfg(test)]
pub(crate) fn fit_cold_keep_workspace(
    data: &Data<'_>,
    model: &ModelSpec,
    opts: &FitOptions,
) -> (Fit, core::FitWorkspace) {
    let Design { n, p, ids, .. } = data.design;
    let ids = ids.expect("a mixed design carries its ids");
    let mut ws = core::build_workspace(model, ids, n, p, opts);
    let rows = Data {
        design: Design {
            ids: None,
            ..data.design
        },
        y: data.y,
    };
    let fit = core::fit_on(&mut ws, &rows, None, opts).into_fit(&rows, model, opts);
    (fit, ws)
}

// Unified fit-core surface for the loop tier.
#[cfg(feature = "loop_advanced")]
pub use core::{
    build_workspace, curvature_hint, fit_on, fit_on_hinted, satterthwaite, FitView, FitWorkspace,
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
#[cfg(test)]
mod standardize_tests;
