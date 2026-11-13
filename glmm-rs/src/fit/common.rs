//! Shared helpers used by 2+ estimator dispatch files (`ols.rs`/`lmm/mod.rs`/
//! `glm.rs`/`glmm/`): θ-width/varcorr bookkeeping, RE-count sizing from
//! `GroupIds`, rank-deficiency salvage, and the two `Fit::se` fill conventions
//! (target-compact vs predictor-indexed). None of this touches a kernel
//! directly — it is `fit`'s own marshalling logic, factored out of `mod.rs`
//! because every estimator's dispatch needs at least one of these.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;

use faer::Mat;

use crate::consts::{
    GAMMA_PHI_HI, GAMMA_PHI_LO, MAX_DATA_ABS, MAX_LOG_OFFSET_ABS, MAX_TRIALS, NB_THETA_HI,
    NB_THETA_LO,
};
use crate::inference::Objective;
use crate::lmm::Criterion;
use crate::spec::{remap_slopes, set_term_slopes};
use crate::standardize::ColumnScaling;

use crate::{
    ColumnId, Data, Design, Family, GroupIds, GroupingRelation, ModelSpec, ReStructure, ReTerm,
    StartValues,
};

use super::{
    Boundary, CondvarCross, Diagnostics, ErrorMessage, Fit, FitError, FitOptions, Note, SeFor,
    SeType, WeightPower,
};

/// Every diagnostic a fitting arm produces, in one carrier. Each of the five
/// routes fills it once (`OlsFitView::diagnostics` and its four siblings), the
/// loop tier reads it off [`super::core::FitView::diagnostics`], and the stable
/// path materializes it into `Fit` inside the `*_view_to_fit` mappers. One
/// carrier, so a new diagnostic costs no new accessor set.
///
/// **Plain `Copy` data, no `Vec`.** The warm loop calls this per draw, so the
/// carrier holds nothing that allocates — a per-draw allocation here would cost
/// every warm-start caller once per fit. The public reshape — `Boundary` for `boundary_hit`,
/// varcorr-aligned flags for `pinned_components`, a note carrying
/// `(pivot_col, pivot)` — is a translation of these fields and nothing more.
///
/// **What is deliberately NOT here.**
/// - `aliased`: no fitting arm decides it. Dropping a redundant column is the
///   pre-dispatch alias gate's job (`detect_aliased` in `fit_checked`, scattered
///   back by `fit_rank_deficient`), which sits ABOVE the view — every arm's
///   own mask is unconditionally all-false, so a field here would always be
///   empty and would suggest a per-route decision that does not exist.
/// - `singular`: it is `boundary_hit == 1` on every route, ORed at
///   materialization with `Fit::has_negligible_component` — which runs on the
///   assembled `Fit` and so is not decided at view level. Both sides of
///   that OR are only ever set on a converged fit: `boundary_hit == 1` latches
///   only when the fit converged (a capped endpoint stays at `boundary_hit ==
///   2`, "no optimum"), and `has_negligible_component` is checked only when
///   materializing a converged `Fit`, so singularity is not assessed at all on
///   a non-converged one.
#[derive(Clone, Copy)]
pub struct FitDiagnostics {
    /// Whether the fit reached its convergence criterion.
    pub converged: bool,
    /// θ boundary: 0 interior, 1 a component pinned to the floor, 2 no optimum.
    /// **Meaningless on OLS/GLM and on the GLM-NB route** (no θ): 0, on GLM-NB
    /// back-derived from its `Fit::singular`.
    pub boundary_hit: u8,
    /// Bitmask of θ components pinned to the boundary, `diagonal_theta` order.
    /// `FitView::diagnostics` reports it in the caller's declaration order, as
    /// `FitView::theta`.
    /// **Meaningless on OLS/GLM and on the GLM-NB route** (no θ): always 0.
    pub pinned_components: u64,
    /// Scale-invariant per-column pivot ratio of the route's own Gram
    /// (`min_pivot_ratio`), and the column attaining it. NaN on
    /// every route and every return that formed no factor to measure. The
    /// GLM-NB route reads it back off its assembled `Fit`'s
    /// [`crate::Note::IllConditioned`], so there it is NaN unless
    /// `ill_conditioned` is set.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
    /// `pivot` fell below the recording route's own detection floor: the design
    /// is computable but its coefficients are barely identified. Each route
    /// compares against its own constant (`ols::PIVOT_MIN` for OLS and GLM,
    /// `lmm::PIVOT_MIN` for the LMM and GLMM) because the routes were calibrated
    /// separately. NaN pivots compare false, so a route that records none never
    /// flags.
    pub ill_conditioned: bool,
    /// GLMM-only: count of fit-path PIRLS solves (BOBYQA objective evals, not
    /// FD-Hessian SE evals) that ran the full `PIRLS_MAX_ITERS` cap without
    /// converging. 0 on every non-GLMM route and on a GLMM fit with no such
    /// eval — see [`crate::Note::PirlsExhausted`].
    pub pirls_exhausted: u32,
    /// GLMM-only: whether the FINAL re-evaluation at the converged gamma-hat
    /// itself exhausted the PIRLS cap — the case where the reported estimates
    /// rest on a truncated solve rather than a rejected trial point. `false`
    /// on every non-GLMM route.
    pub final_pirls_exhausted: bool,
    /// GLMM-only: `WaldSe::Hessian` was requested but the FD joint Hessian
    /// came back non-PD (or a perturbed deviance evaluation was non-finite),
    /// so the SE pass fell back to the RX/Schur route — see
    /// [`crate::Note::HessianSeFallback`]. `false` on every non-GLMM route and
    /// under `WaldSe::Rx`.
    pub hessian_fallback: bool,
    /// GLMM-only: the `ExactProfile` route ended not converged and the fit
    /// reran once on `PqlThenJoint` — see
    /// [`crate::Note::ExactProfileFallback`]. `false` on every non-GLMM route
    /// and on every shape that does not route `ExactProfile`.
    pub exact_profile_fallback: bool,
    /// GLM and GLMM: the fit has an endpoint but no fixed-effect covariance
    /// could be formed there, so its SE outputs are NaN — see
    /// [`crate::Note::SeUnavailable`]. On the GLM it comes with
    /// `ill_conditioned`, `pivot` 0 and `pivot_col` the first column that
    /// broke down. `false` on OLS and LMM routes.
    pub se_unavailable: bool,
    /// GLMM and LMM: the estimated rounding error of the reported deviance is
    /// past what resolves θ̂ — see [`crate::Note::PrecisionLimited`]. `false`
    /// on the OLS and GLM routes.
    pub precision_limited: bool,
    /// GLM and GLMM: the fixed-effects IRLS stopped on separation — see
    /// [`crate::Note::Separated`]. `false` on OLS and LMM routes.
    pub separated: bool,
    /// LMM and GLMM: the gradient search did not converge and BOBYQA
    /// finished from its incumbent — see
    /// [`crate::Note::GradientSearchFallback`]. `false` on every route that
    /// searches with BOBYQA alone.
    pub gradient_search_fallback: bool,
    /// LMM and GLMM gradient routes: the number of gradient evaluations the
    /// BFGS search made (a gradient reuses its point's value evaluation, so
    /// `n_eval` does not count them). 0 on every fit that ran no gradient
    /// search. This field and the two below describe the BFGS search even
    /// when BOBYQA finished the fit after it.
    pub n_grad: u32,
    /// LMM and GLMM gradient routes: why the BFGS search stopped. `None` on
    /// every fit that ran no gradient search.
    pub outer_stop: Option<OuterStop>,
    /// LMM and GLMM gradient routes: the BFGS decrement `½·gᵀHg` at the
    /// search's endpoint (deviance units), the quasi-Newton estimate of the
    /// decrease left; the certificate accepts an endpoint below 1e-8. NaN on
    /// every fit that ran no gradient search.
    pub outer_decrement: f64,
}

/// Why a gradient (BFGS) outer search stopped, as [`Diagnostics::outer_stop`]
/// reports it. The first three are convergence stops; the rest hand the
/// incumbent to BOBYQA.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OuterStop {
    /// The accepted decrease fell below the search's absolute threshold.
    Fdelta,
    /// The projected gradient fell below its tolerance.
    Pgrad,
    /// No trial could be accepted, but every trial's predicted decrease was
    /// below the threshold: the incumbent sits at the deviance's noise floor.
    NoiseFloor,
    /// No trial was accepted within the line search's halvings.
    LineSearch,
    /// The iteration cap was reached.
    MaxIter,
    /// The gradient was unavailable or non-finite at an accepted point.
    GradUnavailable,
    /// The value at the start point was not finite.
    NonFinite,
}

impl From<crate::optim::bfgs::BfgsStop> for OuterStop {
    fn from(s: crate::optim::bfgs::BfgsStop) -> Self {
        use crate::optim::bfgs::BfgsStop as B;
        match s {
            B::Fdelta => OuterStop::Fdelta,
            B::Pgrad => OuterStop::Pgrad,
            B::NoiseFloor => OuterStop::NoiseFloor,
            B::LineSearch => OuterStop::LineSearch,
            B::MaxIter => OuterStop::MaxIter,
            B::GradUnavailable => OuterStop::GradUnavailable,
            B::NonFinite => OuterStop::NonFinite,
        }
    }
}

impl FitDiagnostics {
    /// The no-θ, no-detection carrier: `converged` and nothing else. OLS and GLM
    /// build on this (GLM adds its pivot), and it is what every "this route has
    /// no such state" placeholder above reduces to.
    pub(crate) fn fixed_only(converged: bool) -> Self {
        FitDiagnostics {
            converged,
            boundary_hit: 0,
            pinned_components: 0,
            pivot: f64::NAN,
            pivot_col: 0,
            ill_conditioned: false,
            pirls_exhausted: 0,
            final_pirls_exhausted: false,
            hessian_fallback: false,
            exact_profile_fallback: false,
            se_unavailable: false,
            precision_limited: false,
            separated: false,
            gradient_search_fallback: false,
            n_grad: 0,
            outer_stop: None,
            outer_decrement: f64::NAN,
        }
    }
}

/// Translate a route's carrier into the public [`Diagnostics`]. Cold path only
/// (the loop tier reads the carrier directly), so the two `Vec`s below are
/// affordable; `notes` stays empty on a clean fit.
///
/// `varcorr` is the block layout the caller has already assembled — see
/// [`pinned_flags`] for why it is the only thing that can place the bits.
///
/// `FitView::kernel_diagnostics`'s `Prebuilt` arm reads these notes back into
/// the carrier — change together.
pub(super) fn materialize_diagnostics(
    d: &FitDiagnostics,
    p: usize,
    varcorr: &[Vec<f64>],
) -> Diagnostics {
    let pinned = pinned_flags(d.pinned_components, varcorr);
    let mut notes = vec![];
    if d.ill_conditioned {
        notes.push(Note::IllConditioned {
            columns: vec![d.pivot_col],
            pivot: d.pivot,
        });
    }
    if d.pirls_exhausted > 0 || d.final_pirls_exhausted {
        notes.push(Note::PirlsExhausted {
            evals: d.pirls_exhausted,
            final_eval: d.final_pirls_exhausted,
        });
    }
    if d.hessian_fallback {
        notes.push(Note::HessianSeFallback);
    }
    if d.exact_profile_fallback {
        notes.push(Note::ExactProfileFallback);
    }
    if d.gradient_search_fallback {
        notes.push(Note::GradientSearchFallback);
    }
    if d.se_unavailable {
        notes.push(Note::SeUnavailable);
    }
    if d.precision_limited {
        notes.push(Note::PrecisionLimited);
    }
    if d.separated {
        notes.push(Note::Separated);
    }
    Diagnostics {
        converged: d.converged,
        // `boundary_hit == 1` is the optimizer's own pin decision; the two
        // mixed mappers OR the post-hoc negligible-component check on top of
        // this, which is why `singular` is not simply `boundary == AtBoundary`.
        singular: d.boundary_hit == 1,
        // No fitting arm decides this (see the carrier's doc); the alias gate
        // above dispatch overwrites it in `fit_rank_deficient`.
        aliased: vec![false; p],
        // Exhaustive over the carrier's documented 0/1/2 — deliberately NOT a
        // `_ => NoOptimum` catch-all. A fourth code would then be reported as
        // "the optimizer found no optimum", which is a specific claim about the
        // fit, not a fallback; widening `boundary_hit` must come here and say
        // what the new code means.
        boundary: match d.boundary_hit {
            0 => Boundary::Interior,
            1 => Boundary::AtBoundary,
            2 => Boundary::NoOptimum,
            other => unreachable!("FitDiagnostics::boundary_hit is 0/1/2, got {other}"),
        },
        pinned,
        notes,
        n_grad: d.n_grad as usize,
        outer_stop: d.outer_stop,
        outer_decrement: d.outer_decrement,
    }
}

/// Flag a θ on the upper limit of its search box: some coordinate with
/// `|units·θ| ≥ THETA_HI·(1 − 1e-6)` (`crate::lmm::THETA_HI`, the box both
/// mixed kernels search in) pushes [`Note::VarianceAtLimit`] and turns an
/// `Interior` boundary into [`Boundary::UpperLimit`]; `AtBoundary` and
/// `NoOptimum` stay. `units·θ` is the kernel θ in the units the search ran in,
/// which the box applies to: `units` is 1 except on a Gamma inverse-link GLMM
/// searched on rescaled data (`GlmmWorkspace::search_units`). The two mixed
/// view mappers call this on a fit with a finite endpoint, after
/// [`materialize_diagnostics`].
pub(super) fn note_theta_at_limit(diagnostics: &mut Diagnostics, theta: &[f64], units: f64) {
    let edge = crate::lmm::THETA_HI * (1.0 - 1e-6);
    if theta.iter().any(|t| (units * t).abs() >= edge) {
        diagnostics.notes.push(Note::VarianceAtLimit);
        if diagnostics.boundary == Boundary::Interior {
            diagnostics.boundary = Boundary::UpperLimit;
        }
    }
}

/// Flag a negative-binomial shape on a limit of its search range: θ̂ within
/// 1e-6 of [`NB_THETA_LO`] or [`NB_THETA_HI`] on the log scale
/// (the relative tolerance of [`note_theta_at_limit`]) pushes
/// [`Note::NbShapeAtLimit`] carrying that limit. A NaN θ matches neither.
/// Called by the NB GLM on a converged fit and by the GLMM view mapper on an
/// NB fit with a finite endpoint.
pub(super) fn note_nb_shape_at_limit(diagnostics: &mut Diagnostics, theta: f64) {
    for limit in [NB_THETA_LO, NB_THETA_HI] {
        if (theta.ln() - limit.ln()).abs() <= 1e-6 {
            diagnostics.notes.push(Note::NbShapeAtLimit { limit });
        }
    }
}

/// Reshape a `diagonal_theta`-ordered pin bitmask into [`Diagnostics::pinned`]'s
/// varcorr-aligned flags. Single source for that placement: every route reaches
/// it through [`materialize_diagnostics`], which each of the four view mappers
/// calls.
///
/// `varcorr` is the ONLY thing that maps bit order onto `pinned[g][i]`: the
/// bitmask is keyed to `diagonal_theta` order, which walks the primary factor's
/// `q_p` vech diagonals and then each extra grouping's, in declaration order —
/// the same factors, in the same order, that `varcorr` carries one block each
/// for. Both leave out a hidden column 0 (its diagonal is structural, and
/// `assemble_varcorr` drops the column), so the counts agree there too. `q_g` comes from the block's vech length via [`super::vech_q`], the
/// same inversion [`Fit::stddev_corr`] uses, so `pinned[g][i]` and
/// `stddev_corr(g).0[i]` cannot drift apart.
///
/// One row per `varcorr` block whatever the mask, all `false` at mask 0, so
/// the grid has the layout of every other per-grouping field; no blocks (a
/// fit without random effects) gives no rows.
pub(crate) fn pinned_flags(mask: u64, varcorr: &[Vec<f64>]) -> Vec<Vec<bool>> {
    let mut k = 0usize;
    varcorr
        .iter()
        .map(|vech| {
            (0..super::vech_q(vech.len()))
                .map(|_| {
                    let bit = k < u64::BITS as usize && (mask >> k) & 1 == 1;
                    k += 1;
                    bit
                })
                .collect()
        })
        .collect()
}

/// RE θ length for a mixed model, from topology ALONE (independent of level
/// counts): primary vech `q_p(q_p+1)/2` plus a `vech(Λ_g)` block `q_g(q_g+1)/2`
/// per extra grouping. Equals `LmmGroupings::n_theta()` for the same spec; used to
/// validate a `StartValues.theta` at the stable boundary before the workspace
/// (which is where `n_theta` otherwise first materializes) is built. `None`
/// (fixed-only) → 0.
pub(crate) fn theta_width(re: Option<&ReStructure>) -> usize {
    let Some(re) = re else { return 0 };
    let q_p = 1 + re.slopes.len();
    let mut w = q_p * (q_p + 1) / 2;
    for g in &re.extra_groupings {
        let q_g = 1 + g.slopes.len();
        w += q_g * (q_g + 1) / 2;
    }
    w
}

/// The θ warm start a `StartValues` actually carries, as the kernels take it.
/// An EMPTY `theta` is a per-component cold start (`StartValues`) and must reach
/// them as `None` — the blind-`THETA0` path — not as an empty slice, which the
/// zip-based seeders would silently read as "leave θ at zero".
pub(super) fn warm_theta(start: Option<&StartValues>) -> Option<&[f64]> {
    start.map(|s| s.theta.as_slice()).filter(|t| !t.is_empty())
}

/// A warm start in the caller's units (θ in slot order, each component empty
/// or of its full width) onto the design `cs` standardized, in place:
/// `β̃ = B⁻¹β`, and θ through the random-effect maps
/// (`LmmGroupings::theta_caller_to_design`). An empty component stays empty.
pub(super) fn start_to_design(
    s: &mut StartValues,
    cs: &ColumnScaling,
    g: &crate::lmm::LmmGroupings,
    scratch: &mut [f64],
) {
    if !s.beta.is_empty() {
        cs.beta_to_internal(&mut s.beta);
    }
    if !s.theta.is_empty() {
        g.theta_caller_to_design(&mut s.theta, scratch, cs);
    }
}

/// vech(column-major lower-tri) of `D̂ = scale·M·Mᵀ` for block `b` of `g`, with
/// `M = N·Λ̃` (`LmmGroupings::kernel_rows_to_caller`). `theta_block` is that
/// block's column-major lower-tri `vech(Λ̃)` prefix (as `primary_lambda` reads
/// it), on the fitted design; `N` maps the block's effects to the caller's
/// columns (`b = N·b̃`). It is read off the covariance, with no factorization, so a
/// centered slope can leave no false zero here. Once a slope is centered `M` is
/// not lower triangular, so the sum runs over every `k`. `map` is the fit's
/// column standardization, which `N` is read from.
pub(crate) fn varcorr_block(
    theta_block: &[f64],
    q: usize,
    scale: f64,
    g: &crate::lmm::LmmGroupings,
    b: usize,
    map: &ColumnScaling,
) -> Vec<f64> {
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta_block, q, &mut lam); // Λ̃ lower-tri, row-major
    g.kernel_rows_to_caller(b, q, q, &mut lam, map);
    let mut vech = Vec::with_capacity(q * (q + 1) / 2);
    for c in 0..q {
        for r in c..q {
            let mut d = 0.0;
            for k in 0..q {
                d += lam[r * q + k] * lam[c * q + k];
            }
            vech.push(scale * d);
        }
    }
    vech
}

/// Assemble `Fit::varcorr`: one vech-packed `D̂ = scale·Λ̂Λ̂'` block
/// per grouping, declaration order (primary, then each extra). `scale` is σ̂²
/// (LMM) or 1.0 (GLMM link scale). Path-independent — a function of θ̂ only.
/// A hidden column 0 (a block without a random intercept) is dropped, so the
/// block covers its slopes alone; the entries between two terms stay, as the
/// exact zeros they are.
/// The primary-then-extras vech walk mirrors the `vech_start` layout assigned
/// in `LmmGroupings::from_cluster_spec_ext` (`src/lmm/mod.rs`) — change together.
/// `pub(crate)` so every route recovers varcorr from θ̂ through the same
/// path-independent assembly.
pub(crate) fn assemble_varcorr(
    theta: &[f64],
    groupings: &crate::lmm::LmmGroupings,
    scale: f64,
    map: &ColumnScaling,
) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(1 + groupings.extra_q.len());
    let mut cursor = 0usize;
    for (b, &q) in std::iter::once(&groupings.primary_q)
        .chain(groupings.extra_q.iter())
        .enumerate()
    {
        let vech = q * (q + 1) / 2;
        let mut block = varcorr_block(&theta[cursor..cursor + vech], q, scale, groupings, b, map);
        if groupings.hidden[b] {
            block.drain(..q);
        }
        out.push(block);
        cursor += vech;
    }
    out
}
/// [`Fit::loglik`] for the Gaussian LMM paths: `−½(deviance + ν(1 + ln 2π))`,
/// the stripped constant restored (see `Fit::deviance`'s LMM contract), then
/// onto the `logLik()` scale. `ν` is the criterion's ([`Criterion::nu`]):
/// `n − p` gives the REML criterion `−REMLcrit/2`, `n` the ML log-likelihood.
/// Call AFTER any weighted `−Σlog wᵢ` deviance correction (the correction is
/// part of lme4's criterion on both, so it belongs inside it). NaN in ⇒ NaN
/// out, and `ν = 0` (REML on the degenerate `n ≤ p` short-circuit, which
/// always reports a NaN deviance) gives NaN too.
pub(crate) fn lmm_loglik(deviance: f64, nu: usize) -> f64 {
    if !deviance.is_finite() || nu == 0 {
        return f64::NAN;
    }
    -0.5 * (deviance + nu as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln()))
}

/// [`Fit::loglik`] for the GLMM paths (dense and sparse, Laplace and AGQ):
/// restore the data-only saturated constant the marginal deviance drops.
/// Binomial/Poisson: `−½·deviance + saturated_loglik` (the deviance's data
/// term is the weighted `dev_resid` sum). Gamma and NB: `−½·deviance` with no
/// correction, because their deviance already carries every normalising
/// term. For Gamma, beside the Laplace or quadrature terms, it holds `D/φ̂`
/// plus `family::gamma_dispersion_term` at the ML φ̂, which together are the
/// whole `−2·Σᵢ log f(yᵢ)` with the Gamma density's constants, so
/// `−½·deviance` is the marginal log-likelihood on glmmTMB's scale. The Gamma
/// GLM's `loglik` (`fit::glm::glm_view_to_fit`) is built from the same
/// `gamma_dispersion_term` at its own ML φ̂, so a Gamma GLMM and its GLM sit
/// on one likelihood and compare by likelihood ratio: on
/// `validation/data/simulated/sim_gamma.csv` both agree with glmmTMB's to
/// within 1e-9. lme4's `logLik` of a Gamma `glmer` fit is a different
/// function. For NB, `deviance` is `dev(θ̂) − 2·saturated_loglik(θ̂)` (see
/// `fit_glmm`'s `nb_dev_term`), which already carries the saturated term, so
/// adding it again here would count it twice. NaN in ⇒ NaN out (the
/// non-converged contract).
pub(crate) fn glmm_loglik(
    family: Family,
    nb_theta: f64,
    deviance: f64,
    y: &[f64],
    prior_w: Option<&[f64]>,
) -> f64 {
    if !deviance.is_finite() {
        return f64::NAN;
    }
    match family {
        Family::Gamma { .. } | Family::NegativeBinomial { .. } => -0.5 * deviance,
        _ => -0.5 * deviance + crate::family::saturated_loglik(family, nb_theta, y, prior_w),
    }
}

/// [`Fit::df`] for a converged fit: retained fixed effects + RE θ parameters +
/// 1 if the family estimates a dispersion/scale. `n_theta` counts the free θ
/// only ([`crate::lmm::LmmGroupings::n_free_theta`]): an entry the grouping's
/// terms hold at zero is not estimated. `dispersion_fixed` is whether
/// `FitOptions::dispersion` is `Dispersion::Fixed` (the Gamma and
/// inverse-Gaussian hold-φ-fixed directive; every other family ignores it).
pub(crate) fn model_df(
    family: Family,
    p_retained: usize,
    n_theta: usize,
    dispersion_fixed: bool,
) -> usize {
    let scale = match family {
        Family::Gaussian | Family::NegativeBinomial { .. } => 1,
        Family::Gamma { .. } | Family::InverseGaussian { .. } => usize::from(!dispersion_fixed),
        Family::Binomial { .. } | Family::Poisson { .. } => 0,
    };
    p_retained + n_theta + scale
}

/// Extra grouping `e`'s θ vech offset — nested and crossed factors store it on
/// their own descriptors, keyed by declaration index.
fn extra_vech_start(g: &crate::lmm::LmmGroupings, e: usize) -> usize {
    if let Some(nf) = g.nested {
        if nf.decl == e {
            return nf.vech_start;
        }
    }
    g.crossed
        .iter()
        .find(|cf| cf.decl == e)
        .map(|cf| cf.vech_start)
        .expect("an extra grouping is either nested or crossed")
}

/// [`Fit::ranef_levels`]: level count per grouping, declaration order (primary,
/// then each extra). A nested grouping spans the padded
/// `n_primary·nested_per_parent` child grid (see `Fit::ranef`'s layout doc).
pub(crate) fn ranef_level_counts(g: &crate::lmm::LmmGroupings) -> Vec<usize> {
    let mut counts = Vec::with_capacity(1 + g.extra_q.len());
    counts.push(g.n_primary);
    for e in 0..g.extra_q.len() {
        let is_nested = g.nested.map(|nf| nf.decl) == Some(e);
        counts.push(if is_nested {
            g.n_primary * g.nested_per_parent
        } else {
            g.crossed
                .iter()
                .find(|cf| cf.decl == e)
                .map(|cf| cf.n_levels)
                .expect("an extra grouping is either nested or crossed")
        });
    }
    counts
}

/// Reported `q` per block, in the order of [`ranef_level_counts`]: a hidden
/// column 0 (a block without a random intercept) is not reported.
fn reported_q(g: &crate::lmm::LmmGroupings) -> impl Iterator<Item = usize> + '_ {
    std::iter::once(g.primary_q)
        .chain(g.extra_q.iter().copied())
        .zip(&g.hidden)
        .map(|(q, &hidden)| q - usize::from(hidden))
}

/// [`Fit::varcorr`] with no estimate: one all-NaN vech per block.
pub(crate) fn nan_varcorr(g: &crate::lmm::LmmGroupings) -> Vec<Vec<f64>> {
    reported_q(g)
        .map(|q| vec![f64::NAN; q * (q + 1) / 2])
        .collect()
}

/// `Fit::tau2` or `Fit::stddev_se` with no estimate: `Σ q(q+1)/2` NaN.
pub(crate) fn nan_theta(g: &crate::lmm::LmmGroupings) -> Vec<f64> {
    vec![f64::NAN; reported_q(g).map(|q| q * (q + 1) / 2).sum()]
}

/// [`Fit::ranef`] with no estimate: `Σ levels·q` NaN.
pub(crate) fn nan_ranef(g: &crate::lmm::LmmGroupings) -> Vec<f64> {
    let len = ranef_level_counts(g)
        .iter()
        .zip(reported_q(g))
        .map(|(levels, q)| levels * q)
        .sum();
    vec![f64::NAN; len]
}

/// [`Fit::ranef`] from the DENSE GLMM workspace's spherical modes `u`
/// (blocked/structured layout: primary block level-major `lvl·q_p + c`, then each
/// extra's scalar indicator columns at its absolute `extra_offsets[e]` — this
/// path carries intercept-only extras exclusively, which is why each extra
/// contributes a single scalar θ here; the Λ_p it shares comes from
/// `crate::lmm::primary_lambda`, the per-eval fill both dense layouts run
/// (`glmm::deviance`'s blocked arm and `glmm::workspace::build_packed_m` for
/// the structured one)).
/// `b = Λ̂û` per block: the primary level's `q_p`-vector through the lower-tri
/// `Λ_p`, each extra level through its scalar θ. Output is `Fit::ranef`'s
/// public layout (per grouping, level-major) — for this path the primary block
/// is already level-major, so it maps through 1:1. [`assemble_condvar`] maps
/// Λ̃ and drops the hidden column the same way; change together.
pub(crate) fn assemble_ranef_dense(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    u: &[f64],
    map: &ColumnScaling,
) -> Vec<f64> {
    let q = g.primary_q;
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta, q, &mut lam);
    // Same map, and for the same reason, as `assemble_ranef_sparse`. Only the
    // primary block needs it: this path's extras are intercept-only (asserted
    // below), so their map is the identity.
    g.kernel_rows_to_caller(0, q, q, &mut lam, map);
    let mut out = Vec::with_capacity(g.k_total);
    // A hidden column 0 is not reported (its mode is exactly zero).
    let first = usize::from(g.hidden[0]);
    for lvl in 0..g.n_primary {
        let base = lvl * q;
        for r in first..q {
            out.push((0..q).map(|c| lam[r * q + c] * u[base + c]).sum());
        }
    }
    debug_assert!(!g.extra_slopes_any, "dense GLMM extras are intercept-only");
    let counts = ranef_level_counts(g);
    for (e, &off) in g.extra_offsets.iter().enumerate() {
        let theta_e = theta[extra_vech_start(g, e)];
        for l in 0..counts[e + 1] {
            out.push(theta_e * u[off + l]);
        }
    }
    out
}

/// [`Fit::ranef`] from the SPARSE workspace's spherical modes `u`
/// (`sparse::for_each_z_entry` layout: primary block component-major
/// `d·n_primary + f`, each extra level-major `extra_offsets[e] + l·q_g + c`
/// with a full `q_g×q_g` `Λ_g` block). Output is `Fit::ranef`'s public layout
/// (per grouping, level-major) — the primary block is transposed into it here.
///
/// Also serves the DENSE LMM recovery (`lmm::recover_ranef`): the dense kernel's
/// RE-column elimination order is the same layout, so the two share this rather
/// than each carrying their own walk. What is sparse-specific is the caller, not
/// the layout. [`assemble_condvar`] maps Λ̃ and drops the hidden column the
/// same way; change together.
pub(crate) fn assemble_ranef_sparse(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    u: &[f64],
    map: &ColumnScaling,
) -> Vec<f64> {
    let q = g.primary_q;
    let s = g.n_primary;
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta, q, &mut lam);
    // `Λ̃û` is the conditional mode on the fitted design, and the block's
    // column map `N` takes it to the caller's columns (`b = N·Λ̃û`). Mapping Λ
    // here rather than the output does it once per block instead of once per
    // level — and the whole map must happen HERE, not later: `lmm_fitted`
    // weights `ranef` by the RAW covariate, so it needs a caller-unit `ranef`.
    // With a centered slope `N·Λ̃` is not lower triangular, so each product
    // runs over every column.
    g.kernel_rows_to_caller(0, q, q, &mut lam, map);
    let mut out = Vec::with_capacity(g.k_total);
    // A hidden column 0 is not reported (its mode is exactly zero), here and
    // on the extras below.
    let first = usize::from(g.hidden[0]);
    for f in 0..s {
        for r in first..q {
            out.push((0..q).map(|c| lam[r * q + c] * u[c * s + f]).sum());
        }
    }
    let counts = ranef_level_counts(g);
    for (e, &off) in g.extra_offsets.iter().enumerate() {
        let q_g = g.extra_q[e];
        let mut lam_g = vec![0.0f64; q_g * q_g];
        crate::lmm::primary_lambda(&theta[extra_vech_start(g, e)..], q_g, &mut lam_g);
        g.kernel_rows_to_caller(e + 1, q_g, q_g, &mut lam_g, map);
        let first = usize::from(g.hidden[e + 1]);
        for l in 0..counts[e + 1] {
            let base = off + l * q_g;
            for r in first..q_g {
                out.push((0..q_g).map(|c| lam_g[r * q_g + c] * u[base + c]).sum());
            }
        }
    }
    out
}

/// Per grouping, in declaration (slot) order: `q`, the first reported row (1
/// on a hidden intercept), the level count and `N·Λ̃`, the block's kernel
/// factor mapped to the caller's rows (row-major `q×q`).
fn caller_lambdas(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    map: &ColumnScaling,
) -> Vec<(usize, usize, usize, Vec<f64>)> {
    let counts = ranef_level_counts(g);
    (0..counts.len())
        .map(|b| {
            let (q, start) = if b == 0 {
                (g.primary_q, 0)
            } else {
                (g.extra_q[b - 1], extra_vech_start(g, b - 1))
            };
            let mut lam = vec![0.0f64; q * q];
            crate::lmm::primary_lambda(&theta[start..], q, &mut lam);
            g.kernel_rows_to_caller(b, q, q, &mut lam, map);
            (q, usize::from(g.hidden[b]), counts[b], lam)
        })
        .collect()
}

/// [`Fit::condvar`] from `blocks`, each level's `q×q` block of `A⁻¹` (row-major,
/// the kernel's spherical coordinates, `A = Λ̃ᵀZ̃ᵀWZ̃Λ̃ + I`), level after level
/// in `Fit::ranef`'s order: `scale·(N·Λ̃)·A⁻¹_l·(N·Λ̃)ᵀ` per level, its
/// reported rows and columns packed as `varcorr`'s vech.
///
/// Mirrors `assemble_ranef_sparse` and `assemble_ranef_dense`: the same map
/// of Λ̃ to the caller's rows, the same hidden column; change together.
pub(crate) fn assemble_condvar(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    blocks: &[f64],
    scale: f64,
    map: &ColumnScaling,
) -> Vec<f64> {
    let mut out = Vec::new();
    let mut at = 0usize;
    for (q, first, levels, lam) in caller_lambdas(theta, g, map) {
        let mut ml = vec![0.0f64; q * q];
        for _ in 0..levels {
            let a = &blocks[at..at + q * q];
            at += q * q;
            // ml = (N·Λ̃)·A⁻¹_l, then the product with (N·Λ̃)ᵀ entry by entry.
            for r in 0..q {
                for d in 0..q {
                    ml[r * q + d] = (0..q).map(|c| lam[r * q + c] * a[c * q + d]).sum();
                }
            }
            for s in first..q {
                for r in s..q {
                    let v: f64 = (0..q).map(|d| ml[r * q + d] * lam[s * q + d]).sum();
                    out.push(scale * v);
                }
            }
        }
    }
    out
}

/// [`Fit::condvar_cross`] from the kernel-coordinate pieces of `A⁻¹` the
/// factor walks return in a [`CondvarCross`] of the same shape: per family
/// the row-major `w×w` block `(A⁻¹)_ff` (`w = q_p + m`, the primary level's
/// kernel columns then its children's), the column-major `t×t` tail block `S⁻¹`
/// (`S` the Schur complement of the families, tail columns in the kernel's
/// order), and per family and tail column `b` its rows share the `w` values of
/// `W_f[:, b]`, `W_f = F_f⁻¹B_f`. With `D_f = diag(N·Λ̃_p, λ_c)` the family's
/// kernel factor mapped to the caller's rows (the hidden column dropped) and
/// `λ_b` the tail's: `family` is `scale·D_f(A⁻¹)_ffD_fᵀ`, `tail` is
/// `scale·λ_aλ_b(S⁻¹)_ab`, and `H_f[b] = D_fW_f[:, b]/λ_b`. Since
/// `(A⁻¹)_fb = −(W_fS⁻¹)_b` and `(A⁻¹)_fg = W_fS⁻¹W_gᵀ` for `f ≠ g`, the
/// products [`CondvarCross`] documents are those blocks, scaled. A column with
/// `λ_b = 0` is dropped: its tail row is zero. The family grouping is slot 0;
/// `unpermute_fit` names it in declaration order.
///
/// Mirrors `assemble_condvar`: the same map of Λ̃ to the caller's rows, the
/// same hidden column; change together.
pub(crate) fn assemble_condvar_cross(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    kernel: CondvarCross,
    scale: f64,
    map: &ColumnScaling,
) -> CondvarCross {
    let lams = caller_lambdas(theta, g, map);
    let (q, first, _, lam_p) = &lams[0];
    let (q, first) = (*q, *first);
    let m = g.nested_per_parent;
    let (w, wr) = (q + m, q - first + m);
    // The extras on these layouts are intercept-only: one `λ` each.
    let lam_c = g.nested.map_or(0.0, |nf| lams[nf.decl + 1].3[0]);
    let kf = g.k_family();
    let t = g.k_crossed();
    let mut lam_t = vec![0.0f64; t];
    for cf in &g.crossed {
        let at = g.extra_offsets[cf.decl] - kf;
        lam_t[at..at + cf.n_levels].fill(lams[cf.decl + 1].3[0]);
    }
    // Row `r` of `D_f` (reported) times a kernel `w`-vector.
    let d_row = |r: usize, v: &[f64]| -> f64 {
        if r < q - first {
            (0..q).map(|c| lam_p[(r + first) * q + c] * v[c]).sum()
        } else {
            lam_c * v[r + first]
        }
    };
    let mut out = CondvarCross {
        family_grouping: 0,
        family: Vec::with_capacity(kernel.family.len() / (w * w).max(1) * wr * (wr + 1) / 2),
        tail: Vec::with_capacity(t * (t + 1) / 2),
        coupling_at: Vec::with_capacity(kernel.coupling_at.len()),
        coupling_cols: Vec::with_capacity(kernel.coupling_cols.len()),
        coupling: Vec::with_capacity(kernel.coupling_cols.len() * wr),
        nested_slots: Vec::new(),
    };
    // `ma = D_f·(A⁻¹)_ff`, reported rows × kernel columns, then times `D_fᵀ`.
    let mut ma = vec![0.0f64; wr * w];
    let mut col = vec![0.0f64; w];
    for a in kernel.family.chunks(w * w) {
        for d in 0..w {
            for (k, v) in col.iter_mut().enumerate() {
                *v = a[k * w + d];
            }
            for r in 0..wr {
                ma[r * w + d] = d_row(r, &col);
            }
        }
        for c in 0..wr {
            for r in c..wr {
                out.family.push(scale * d_row(c, &ma[r * w..(r + 1) * w]));
            }
        }
    }
    for b in 0..t {
        for a in b..t {
            out.tail
                .push(scale * lam_t[a] * lam_t[b] * kernel.tail[b * t + a]);
        }
    }
    out.coupling_at.push(0);
    for f in 0..kernel.coupling_at.len().saturating_sub(1) {
        for e in kernel.coupling_at[f] as usize..kernel.coupling_at[f + 1] as usize {
            let b = kernel.coupling_cols[e];
            let lam_b = lam_t[b as usize];
            if lam_b == 0.0 {
                continue;
            }
            let v = &kernel.coupling[e * w..(e + 1) * w];
            out.coupling_cols.push(b);
            out.coupling.extend((0..wr).map(|r| d_row(r, v) / lam_b));
        }
        out.coupling_at.push(out.coupling_cols.len() as u32);
    }
    out
}

/// [`Fit::condvar_x`] from `blocks`, each level's `q×p` block of
/// `A⁻¹Λ̃ᵀZ̃ᵀWx̃` (row-major, on the fitted design's columns `x̃ = X·B`), in
/// the level order of [`assemble_condvar`]: `(N·Λ̃)` on the left, `B⁻¹` on
/// the right, the reported rows only. Mirrors `assemble_condvar`; change
/// together.
pub(crate) fn assemble_condvar_x(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    blocks: &[f64],
    p: usize,
    map: &ColumnScaling,
) -> Vec<f64> {
    let mut out = Vec::new();
    let mut at = 0usize;
    let mut row = vec![0.0f64; p];
    for (q, first, levels, lam) in caller_lambdas(theta, g, map) {
        for _ in 0..levels {
            let a = &blocks[at..at + q * p];
            at += q * p;
            for r in first..q {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = (0..q).map(|c| lam[r * q + c] * a[c * p + j]).sum();
                }
                map.row_to_caller(&mut row);
                out.extend_from_slice(&row);
            }
        }
    }
    out
}

/// [`Fit::fitted`] for the Gaussian LMM paths: `μ̂ = o + Xβ̂ + Zb̂` per row,
/// scattered straight through the ids. **Z is never materialized** — each row
/// reads its own level's block out of `ranef` and weights it by that row's own
/// covariates, which is what keeps the no-Z property the whole LMM design rests
/// on true. A slope index `≥ p` reads `z`, the row-major `n × s`
/// random-effect-only columns.
///
/// `ranef` is [`Fit::ranef`]'s public layout (per grouping, level-major), so this
/// runs after [`assemble_ranef_sparse`]. The identity link makes μ̂ = η̂, and the
/// offset is added back here because the LMM applies it as an exact `y − o` shift
/// BEFORE accumulation and never sees it again — the same caveat `fit_ols_prebuilt`
/// documents, and the same fix.
#[allow(clippy::too_many_arguments)] // marshals (x, n, p, beta, ranef, groupings, ids…)
pub(crate) fn lmm_fitted(
    x: &[f64],
    z: &[f64],
    s: usize,
    n: usize,
    p: usize,
    beta: &[f64],
    ranef: &[f64],
    g: &crate::lmm::LmmGroupings,
    primary_ids: &[u32],
    extra_ids: &[Vec<u32>],
    offset: Option<&[f64]>,
) -> Vec<f64> {
    let counts = ranef_level_counts(g);
    // Start of each grouping's block in `ranef`, declaration order.
    let mut block_start = Vec::with_capacity(counts.len());
    let mut acc = 0usize;
    // Reported entries per level: a hidden column 0 is not in `ranef`.
    let q_of = |e: usize| {
        let q = if e == 0 {
            g.primary_q
        } else {
            g.extra_q[e - 1]
        };
        q - usize::from(g.hidden[e])
    };
    for (e, &levels) in counts.iter().enumerate() {
        block_start.push(acc);
        acc += levels * q_of(e);
    }
    (0..n)
        .map(|i| {
            let row = &x[i * p..(i + 1) * p];
            let z_row = &z[i * s..(i + 1) * s];
            let mut eta = offset.map_or(0.0, |o| o[i]);
            for (j, &b) in beta.iter().enumerate() {
                eta += row[j] * b;
            }
            let mut add_block = |e: usize, level: usize, slope_cols: &[usize]| {
                let q = q_of(e);
                let base = block_start[e] + level * q;
                let first = if g.hidden[e] {
                    0
                } else {
                    eta += ranef[base]; // intercept component, z = 1
                    1
                };
                for (d, &sc) in slope_cols.iter().enumerate() {
                    let v = if sc < p { row[sc] } else { z_row[sc - p] };
                    eta += ranef[base + first + d] * v;
                }
            };
            add_block(0, primary_ids[i] as usize, &g.primary_slope_cols);
            for (e, level_ids) in extra_ids.iter().enumerate() {
                add_block(e + 1, level_ids[i] as usize, &g.extra_slope_cols[e]);
            }
            eta
        })
        .collect()
}

/// Which grouping the kernel holds in each slot, relative to the order the
/// caller declared them in. Slot 0 is the primary grouping, slot `g+1` the g-th
/// extra — the layout of both the `ModelSpec` and the `GroupIds` the kernels
/// see, and of every grouping-indexed result they emit.
///
/// `spec_sized_from_ids` is the only producer. Its size rule exchanges the
/// primary with exactly one extra. Undoing the reorder is the same exchange
/// applied a second time — a transposition is its own inverse — which is why
/// this records the one swapped index rather than a full index map.
///
/// The two groupings it exchanges need not be equally wide, so data holding
/// one entry per grouping (`varcorr`, `ranef_levels`, `pinned`) moves by
/// [`Perm::swap_slots`], and data holding a block per grouping moves by
/// [`Perm::swap_blocks`] with that block's width: `ranef` levels·q, `condvar`
/// levels·q(q+1)/2, `condvar_x` levels·q·p and `tau2`/`stddev_se` the
/// `varcorr` block length, each with the reported q (slopes only on a block
/// without an intercept); θ the kernel's vech of each block, hidden intercept
/// included; the `pinned_components` mask the block's non-structural
/// diagonals.
///
/// When the swap moved a slope grouping first and the old primary turned out
/// nested in it, the kernel holds the old primary as a `NestedWithin` extra
/// on padded `parent·W + position` slots, and `nested_slots` maps each of its
/// declared levels to its slot. [`unpermute_fit`] then reads that grouping's
/// per-level rows off the slots in declared level order, and hands the map to
/// [`CondvarCross::nested_slots`]. θ needs no map: a nested intercept has the
/// one θ entry a crossed one has.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct Perm {
    /// Extra grouping that traded places with the primary. `None` is the
    /// identity: declaration order reached the kernel untouched.
    swapped_extra: Option<usize>,
    /// Per declared level of the old primary, its child slot in the kernel;
    /// empty unless the old primary was nested.
    nested_slots: Vec<u32>,
}

impl Perm {
    /// Declaration order reached the kernel untouched.
    pub(crate) const IDENTITY: Perm = Perm {
        swapped_extra: None,
        nested_slots: Vec::new(),
    };

    /// Whether the kernel's slot order IS the caller's declaration order.
    pub(crate) fn is_identity(&self) -> bool {
        self.swapped_extra.is_none()
    }

    /// Map a vector holding one entry per grouping between slot order and
    /// declaration order — `v[0]` the primary's, `v[g+1]` the g-th extra's. The
    /// same call converts either way (see the type doc).
    pub(crate) fn swap_slots<T>(&self, v: &mut [T]) {
        if let Some(k) = self.swapped_extra {
            v.swap(0, k + 1);
        }
    }

    /// The per-index form of [`Perm::swap_slots`], for a reader that holds one
    /// grouping index instead of a vector (the profile kernel's slot-to-grouping
    /// map). The same call converts either way.
    pub(crate) fn map_index(&self, i: usize) -> usize {
        match self.swapped_extra {
            Some(k) if i == 0 => k + 1,
            Some(k) if i == k + 1 => 0,
            _ => i,
        }
    }

    /// Same map for a concatenation of per-grouping blocks of unequal width, in
    /// place. `widths(i)` is the width of block `i` of `v` in its CURRENT
    /// order, so the call that undoes a move reads the widths in the moved
    /// order. An empty `v` (a fit with no `condvar`) is left alone.
    pub(crate) fn swap_blocks<T>(&self, v: &mut [T], widths: impl Fn(usize) -> usize) {
        let Some(k) = self.swapped_extra else {
            return;
        };
        if v.is_empty() {
            return;
        }
        let a = widths(0);
        let s: usize = (0..=k).map(&widths).sum();
        let b = widths(k + 1);
        let e = s + b;
        debug_assert!(e <= v.len(), "block widths overrun the vector");
        // [0, 1..=k, k+1] → [1..=k, k+1, 0] → [k+1, 1..=k, 0].
        v[..e].rotate_left(a);
        v[..e - a].rotate_right(b);
    }
}

/// Report a fit's grouping-indexed results in the caller's declaration order.
/// The kernels emit them in slot order, which [`spec_sized_from_ids`]'s size
/// rule may have reordered; this is the single site that undoes it for the
/// stable API, so no `Fit` consumer sees slot order. `beta`, `se`, `vcov`,
/// `fitted`, `deviance` and `loglik` are grouping-order-independent and are not
/// touched.
pub(super) fn unpermute_fit(perm: &Perm, fit: &mut Fit) {
    if perm.is_identity() {
        return;
    }
    // Every block move before `ranef_levels` and `varcorr` move: their widths
    // must still describe the slot order the blocks are in.
    let p = fit.beta.len();
    let (levels, varcorr) = (&fit.ranef_levels, &fit.varcorr);
    let q = |g: usize| super::vech_q(varcorr[g].len());
    perm.swap_blocks(&mut fit.ranef, |g| levels[g] * q(g));
    perm.swap_blocks(&mut fit.condvar, |g| levels[g] * q(g) * (q(g) + 1) / 2);
    perm.swap_blocks(&mut fit.condvar_x, |g| levels[g] * q(g) * p);
    perm.swap_blocks(&mut fit.tau2, |g| varcorr[g].len());
    perm.swap_blocks(&mut fit.stddev_se, |g| varcorr[g].len());
    // The kernel's primary, the family grouping, is the swapped extra.
    if let (Some(c), Some(k)) = (fit.condvar_cross.as_mut(), perm.swapped_extra) {
        c.family_grouping = k + 1;
    }
    perm.swap_slots(&mut fit.ranef_levels);
    perm.swap_slots(&mut fit.varcorr);
    perm.swap_slots(&mut fit.diagnostics.pinned);
    // A nested old primary now leads, `ranef_levels[0]` padded slots wide:
    // one row per declared level, read off its slot.
    let slots = &perm.nested_slots;
    if slots.is_empty() {
        return;
    }
    debug_assert_eq!(fit.varcorr[0].len(), 1, "the old primary is intercept-only");
    let padded = fit.ranef_levels[0];
    let gather = |v: &mut Vec<f64>, w: usize| {
        if v.is_empty() {
            return;
        }
        let rows: Vec<f64> = slots
            .iter()
            .flat_map(|&s| v[s as usize * w..(s as usize + 1) * w].to_vec())
            .collect();
        v.splice(..padded * w, rows);
    };
    gather(&mut fit.ranef, 1);
    gather(&mut fit.condvar, 1);
    gather(&mut fit.condvar_x, p);
    fit.ranef_levels[0] = slots.len();
    if let Some(c) = fit.condvar_cross.as_mut() {
        c.nested_slots = slots.clone();
    }
}

/// The size rule picks which grouping the kernel holds as its primary. It
/// fires only when every extra grouping is `Crossed`; otherwise nothing moves.
/// Then:
///
/// 1. **No grouping has slopes:** the grouping with the most levels becomes the
///    primary.
/// 2. **Exactly one grouping has slopes, and it is an extra:** that grouping
///    becomes the primary, provided the reordered spec still routes `NoZ`
///    ([`spec_sized_from_ids`] checks this with `classify_design` and undoes
///    the swap otherwise).
/// 3. **Otherwise** (two or more slope groupings, or the slope grouping is
///    already the primary): nothing moves.
///
/// Why rule 1 pays. The primary is eliminated as block-diagonal families at
/// cost linear in its level count, while every extra lands in one Schur tail
/// whose width is the total crossed level count and whose factorization is
/// cubic in that width. Which grouping the user happened to write first
/// therefore sets the tail width. MixedModels.jl sorts its `ReMat`s by
/// decreasing level count for exactly this reason; Bates, Alday & Kokandakar
/// (2025), arXiv:2505.11674, measure the effect on InstEval with that sort
/// defeated — ~4.5M vs 775K nonzeros in the Cholesky factor. Only the choice of
/// primary matters: the tail's cost does not depend on the order of the extras
/// inside it, so one transposition is all a reorder needs.
///
/// Why rule 2 pays. A slope on an extra grouping routes the sparse solver,
/// which forms no conditional variances; with the slope grouping as primary
/// the same model gets the per-level elimination, conditional variances and
/// the dense route's speed. It holds back when the reordered spec would route
/// `Sparse` anyway (more than `MAX_CROSSED_LEVELS` crossed levels left in the
/// tail, or a slope block wider than `MAX_PRIMARY_Q`): the order then gives no
/// conditional variances either way, and the swap would move the old primary,
/// possibly one with thousands of levels, out of the per-level elimination
/// into the sparse tail, a move that has not been measured.
///
/// Why rule 3 leaves the order alone. `(1 + x | subject) + (1 | item)` stays
/// as written even when `item` has more levels: moving `item` first would put
/// the slope on an extra, route the fit sparse and lose the conditional
/// variances. And with two slope groupings one of them is an extra whichever
/// goes first.
///
/// Why nothing moves with a `NestedWithin` grouping. Hoisting a child out of
/// its parent turns a nesting that rides inside the primary's block-diagonal
/// elimination into two crossed factors sharing one tail, which can be several
/// times WIDER.
///
/// Ties under rule 1 keep the declared primary (and, among extras, the
/// earliest), so a model already in the chosen order is handed to the kernel
/// exactly as declared.
fn size_rule_perm(re: &ReStructure, primary_levels: usize, extra_counts: &[usize]) -> Perm {
    if re
        .extra_groupings
        .iter()
        .any(|g| g.relation != GroupingRelation::Crossed)
    {
        return Perm::IDENTITY;
    }
    let mut slope_extras = re
        .extra_groupings
        .iter()
        .enumerate()
        .filter(|(_, g)| !g.slopes.is_empty());
    match (
        re.slopes.is_empty(),
        slope_extras.next(),
        slope_extras.next(),
    ) {
        // Rule 1.
        (true, None, _) => {
            let mut best = primary_levels;
            let mut swapped_extra = None;
            for (k, &levels) in extra_counts.iter().enumerate() {
                if levels > best {
                    best = levels;
                    swapped_extra = Some(k);
                }
            }
            Perm {
                swapped_extra,
                nested_slots: Vec::new(),
            }
        }
        // Rule 2; its routing condition is checked by the caller.
        (true, Some((k, _)), None) => Perm {
            swapped_extra: Some(k),
            nested_slots: Vec::new(),
        },
        // Rule 3.
        _ => Perm::IDENTITY,
    }
}

/// A [`ModelSpec`] with the level counts its kernels size their random-effect
/// blocks by. The public spec carries topology only; the counts come from the
/// fit's [`GroupIds`] through [`spec_sized_from_ids`], the only producer outside
/// tests.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SizedSpec {
    /// The spec, in kernel slot order (see [`Perm`]).
    pub(crate) spec: ModelSpec,
    /// Primary grouping level count. 0 on a fixed-only spec.
    pub(crate) n_primary: usize,
    /// Per extra grouping, slot order: the level count of a crossed grouping,
    /// the children per parent (the widest parent's count) of a nested one.
    pub(crate) extra_counts: Vec<usize>,
}

/// The [`SizedSpec`] of `model` on `ids` (`n_primary = max(primary)+1`; per
/// crossed grouping `max(extra[g])+1`; per nested grouping the widest parent's
/// distinct child count). Topology tags, family and slope columns are kept.
/// `re: None` gives the spec as-is with no counts.
///
/// It is also where [`size_rule_perm`]'s grouping reorder is applied, which is
/// why the ids come back with the spec rather than being left to the caller:
/// a sized spec and the ids that match it can only be obtained together, so no
/// path can size a spec and then hand the kernel ids in a different order. The
/// swap moves two `Vec` handles, two slope lists and two counts — nothing
/// per-observation — but the ids must be cloned to be swapped, so the clone
/// happens only on the models the rule actually reorders (`Cow::Borrowed`
/// otherwise). A slope-grouping swap whose reordered spec would route
/// `Sparse` is undone before the clone, and the spec, ids and [`Perm`] come
/// back exactly as declared.
///
/// A slope-grouping swap that stays then checks the grouping it displaced for
/// nesting in the new primary, with the formula front end's two guards
/// ([`detect_flat_nesting`]). When every one of its declared levels passes,
/// it becomes a `NestedWithin` extra on padded ids, sized as the slope-first
/// spelling sizes it, so the kernel gets the input it gets when the user
/// writes the slope grouping first; [`Perm`] carries the level-to-slot map
/// back. Otherwise it stays crossed.
pub(crate) fn spec_sized_from_ids<'a>(
    model: &ModelSpec,
    ids: &'a GroupIds,
) -> (SizedSpec, Cow<'a, GroupIds>, Perm) {
    let mut sized = sized_as_declared(model, ids);
    let Some(re) = model.re.as_ref() else {
        return (sized, Cow::Borrowed(ids), Perm::IDENTITY);
    };
    let perm = size_rule_perm(re, sized.n_primary, &sized.extra_counts);
    let Some(k) = perm.swapped_extra else {
        return (sized, Cow::Borrowed(ids), perm);
    };
    // The spec half of the exchange; applying it twice undoes it.
    let swap_spec = |sized: &mut SizedSpec| {
        std::mem::swap(&mut sized.n_primary, &mut sized.extra_counts[k]);
        let sized_re = sized.spec.re.as_mut().expect("re is Some on this path");
        std::mem::swap(
            &mut sized_re.slopes,
            &mut sized_re.extra_groupings[k].slopes,
        );
        std::mem::swap(&mut sized_re.terms, &mut sized_re.extra_groupings[k].terms);
    };
    swap_spec(&mut sized);
    // Rule 2 moved a slope grouping first: keep the swap only when the
    // reordered spec routes the dense kernel (see `size_rule_perm`).
    let moved_slopes = sized.spec.re.as_ref().is_some_and(|r| !r.slopes.is_empty());
    if moved_slopes && super::classify_design(&sized, 1) == super::Solver::Sparse {
        swap_spec(&mut sized);
        return (sized, Cow::Borrowed(ids), Perm::IDENTITY);
    }
    let mut swapped_ids = ids.clone();
    std::mem::swap(&mut swapped_ids.primary, &mut swapped_ids.extra[k]);
    let mut perm = perm;
    if moved_slopes {
        // Labelled by declared id, so a child's position inside its parent
        // is its declared id order.
        let nested = detect_flat_nesting(&swapped_ids.primary, &ids.primary);
        // A declared level with no rows (a gap a matrix-door caller may
        // leave) has no slot to map its results back from.
        let declared = sized.extra_counts[k];
        if let Some((padded, labels)) =
            nested.filter(|(_, l)| l.iter().flatten().count() == declared)
        {
            let mut slots = vec![0u32; declared];
            for (slot, level) in labels.iter().enumerate() {
                if let Some(level) = level {
                    slots[*level as usize] = slot as u32;
                }
            }
            let sized_re = sized.spec.re.as_mut().expect("re is Some on this path");
            sized_re.extra_groupings[k].relation = GroupingRelation::NestedWithin;
            swapped_ids.extra[k] = padded;
            sized.extra_counts[k] =
                nested_width(&swapped_ids.primary, &swapped_ids.extra[k], sized.n_primary);
            perm.nested_slots = slots;
            debug_assert!(check_nested_ids(sized_re, &swapped_ids).is_ok());
            debug_assert!(super::classify_design(&sized, 1) == super::Solver::NoZ);
        }
    }
    (sized, Cow::Owned(swapped_ids), perm)
}

/// The [`SizedSpec`] of `model` on `ids` in declaration order, with no
/// grouping reorder: the counts [`spec_sized_from_ids`] starts from. Its
/// callers outside it are the tests that hand the forced kernel entries a
/// design exactly as written.
pub(crate) fn sized_as_declared(model: &ModelSpec, ids: &GroupIds) -> SizedSpec {
    let Some(re) = model.re.as_ref() else {
        return SizedSpec {
            spec: model.clone(),
            n_primary: 0,
            extra_counts: Vec::new(),
        };
    };
    let level_count = |v: &[u32]| v.iter().copied().max().map(|m| m as usize + 1).unwrap_or(1);
    let n_primary = level_count(&ids.primary);
    let extra_counts: Vec<usize> = re
        .extra_groupings
        .iter()
        .enumerate()
        .map(|(g, gr)| match gr.relation {
            GroupingRelation::Crossed => level_count(&ids.extra[g]),
            GroupingRelation::NestedWithin => {
                // Per-parent distinct-child-id sets, counted directly from the
                // (primary, extra) row pairs — layout-agnostic (works whether
                // or not the frontend's ids are contiguous per-parent blocks),
                // unlike deriving from a single global `max(extra)+1` divided
                // by `n_primary`, which under-sizes any parent with an
                // above-average child count (the true fix for unbalanced
                // nesting: the count must be the TRUE max, not a
                // `children.div_ceil(n_primary)` global average).
                nested_width(&ids.primary, &ids.extra[g], n_primary)
            }
        })
        .collect();
    SizedSpec {
        spec: ModelSpec {
            family: model.family,
            re: Some(re.clone()),
        },
        n_primary,
        extra_counts,
    }
}

/// Lower-tri Gram `G = XᵀX` (from row-major X) → `aliased_columns` mask. `p` is
/// tiny, so the extra `O(N·p²)` reduction for the drop decision is negligible
/// (the reduced fit recomputes its own suff stats).
///
/// Each column enters the Gram divided by the power of two at its largest
/// `|xᵢⱼ|` (`2^⌊log₂ maxᵢ|xᵢⱼ|⌋`). The test is per column and homogeneous of
/// degree two in a column's scale on both sides, and a power-of-two factor
/// is exact, so this changes no verdict on a design whose Gram is finite; it
/// keeps `Σxᵢⱼ²` finite past `|x| ≈ 1.3e154`, where the raw Gram's
/// diagonal overflows to `∞` and `∞ ≤ ε·∞` marks the column aliased.
pub(crate) fn detect_aliased(x: &[f64], n: usize, p: usize) -> Vec<bool> {
    let (gram, _) = scaled_gram(x, n, p);
    crate::ols::aliased_columns(gram.as_ref(), p, crate::ols::ALIAS_EPS)
}

/// The lower triangle of [`detect_aliased`]'s Gram, with the per-column
/// factors `inv` it scaled the columns by: Gram column `a` is `x_a·inv[a]`.
fn scaled_gram(x: &[f64], n: usize, p: usize) -> (Mat<f64>, Vec<f64>) {
    let mut inv = vec![1.0f64; p];
    for (a, r) in inv.iter_mut().enumerate() {
        let m = (0..n).fold(0.0f64, |m, i| m.max(x[i * p + a].abs()));
        if m > 0.0 && m.is_finite() {
            *r = 2f64.powi(-(m.log2().floor() as i32).clamp(-1022, 1023));
        }
    }
    let mut gram = Mat::<f64>::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            let xa = x[i * p + a] * inv[a];
            for b in 0..=a {
                gram[(a, b)] += xa * (x[i * p + b] * inv[b]);
            }
        }
    }
    (gram, inv)
}

/// An orthonormal basis of the null space of `x` (row-major `n × p`), `k × p`
/// row-major with one row per column `aliased` marks, in column order; empty
/// when none is. Read off the factor [`detect_aliased`] decides on, so the
/// basis and the alias verdicts agree: an aliased column `d` of the scaled
/// design is, to the alias tolerance, `Σ c_j` times the retained columns
/// `j ∈ K` before it, with `L_KKᵀc = l_d` (row `d` of the factor over `K`), so
/// `e_d − Σ c_j e_j` maps the scaled design to zero; multiplying entry `a` by
/// `inv[a]` makes it a null vector of `x` itself. Modified Gram–Schmidt then
/// orthonormalizes the vectors in order.
pub(crate) fn null_basis(x: &[f64], n: usize, p: usize, aliased: &[bool]) -> Vec<f64> {
    let (gram, inv) = scaled_gram(x, n, p);
    let (mask, l) = crate::ols::aliased_factor(gram.as_ref(), p, crate::ols::ALIAS_EPS);
    debug_assert_eq!(mask, aliased, "null_basis: the alias verdicts differ");
    let mut basis: Vec<f64> = Vec::new();
    for d in (0..p).filter(|&d| aliased[d]) {
        let kept: Vec<usize> = (0..d).filter(|&j| !aliased[j]).collect();
        let mut v = vec![0.0f64; p];
        v[d] = 1.0;
        // Back-substitution on the upper triangle `L_KKᵀ`, last kept column first.
        for (t, &j) in kept.iter().enumerate().rev() {
            let s: f64 = kept[t + 1..].iter().map(|&k| l[k * p + j] * -v[k]).sum();
            v[j] = -(l[d * p + j] - s) / l[j * p + j];
        }
        for (va, r) in v.iter_mut().zip(&inv) {
            *va *= r;
        }
        for prev in basis.chunks(p) {
            let dot: f64 = prev.iter().zip(&v).map(|(a, b)| a * b).sum();
            for (va, pa) in v.iter_mut().zip(prev) {
                *va -= dot * pa;
            }
        }
        let norm = v.iter().map(|a| a * a).sum::<f64>().sqrt();
        basis.extend(v.iter().map(|a| a / norm));
    }
    basis
}

/// The spec of the reduced fit on the full columns `kept`: every grouping's
/// slopes moved by [`remap_slopes`] and its terms with them
/// ([`set_term_slopes`]), so the reduced spec passes [`check_model_shape`].
/// The second value lists the full `[X | Z_r]` columns the reduced fit reads
/// as random-effect-only columns, slope index `kept.len() + j` reading
/// `sources[j]`: a dropped column that is a random slope, and every existing
/// `Z_r` column, in the order the slope lists first name them.
pub(super) fn remap_spec_slopes(model: &ModelSpec, kept: &[usize]) -> (ModelSpec, Vec<usize>) {
    let mut spec = model.clone();
    let mut sources = Vec::new();
    if let Some(re) = &mut spec.re {
        re.slopes = remap_slopes(&re.slopes, kept, kept.len(), &mut sources);
        set_term_slopes(&mut re.terms, &re.slopes);
        for g in &mut re.extra_groupings {
            g.slopes = remap_slopes(&g.slopes, kept, kept.len(), &mut sources);
            set_term_slopes(&mut g.terms, &g.slopes);
        }
    }
    (spec, sources)
}

/// The objective of a GLMM fit at `nagq` quadrature nodes: the Laplace
/// approximation at 1, adaptive Gauss-Hermite quadrature above it.
pub(super) fn glmm_objective(nagq: u8) -> Objective {
    if nagq == 1 {
        Objective::Laplace
    } else {
        Objective::Agq { nodes: nagq }
    }
}

/// Fit the reduced (aliased-columns-dropped) model and scatter β/se back to full
/// width: retained slots take the reduced fit, aliased slots are NaN,
/// `converged` follows the reduced fit, `tau2`/`varcorr`/`dispersion` pass through
/// (the RE structure is unchanged; a slope on a kept column moves to its reduced
/// index, and a slope on a dropped column reads the same values from a
/// random-effect-only column of the reduced design, [`remap_spec_slopes`]).
///
/// `Fit::aliased` is the UNION of this level's `aliased` and the reduced fit's own
/// mask, not just this level's — see the OR in the scatter loop. That keeps the
/// field's contract (`NaN` in β/se ⇔ the column was dropped) true across a nested
/// salvage, which the recursion below makes reachable.
///
/// **Recursion.** This re-enters `fit_checked`, so termination is a real
/// obligation. It rests on `kept.len() < p`: every entry hands the recursive
/// call a strictly narrower design, bounding the chain at `p` deep. The only
/// caller is the alias gate (`detect_aliased` on X'X at `ALIAS_EPS`), which
/// holds for a stronger reason too — the reduced design is full-rank in X'X, so
/// the gate never fires a second time and the chain is one deep in practice.
/// The `debug_assert` below pins the general requirement anyway, because a
/// second caller passing a mask from a different matrix would not inherit that
/// argument.
///
/// `kept.len() == 0` (every column aliased — reachable on an all-zero column) is
/// not a termination hazard: `fit_checked` skips the gate at `p == 0`.
pub(super) fn fit_rank_deficient(
    data: &Data<'_>,
    model: &ModelSpec,
    start: Option<&StartValues>,
    opts: &FitOptions,
    aliased: &[bool],
) -> Fit {
    let Design { x, n, p, re, s, .. } = data.design;
    let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let pk = kept.len();
    debug_assert!(
        pk < p,
        "fit_rank_deficient: `aliased` must drop at least one column, or the \
         recursive fit_checked re-enters on identical input"
    );
    let mut to_reduced = vec![usize::MAX; p];
    for (r, &orig) in kept.iter().enumerate() {
        to_reduced[orig] = r;
    }
    // Reduced design (drop aliased columns), row-major.
    let mut xr = vec![0.0f64; n * pk];
    for i in 0..n {
        for (r, &orig) in kept.iter().enumerate() {
            xr[i * pk + r] = x[i * p + orig];
        }
    }
    // StartValues.beta is p-wide → reduce it; theta is RE-only, unchanged. An
    // empty β is the cold-start marker, not a p-wide vector — it stays empty
    // rather than being indexed.
    let start_r = start.map(|s| StartValues {
        beta: if s.beta.is_empty() {
            Vec::new()
        } else {
            kept.iter().map(|&o| s.beta[o]).collect()
        },
        theta: s.theta.clone(),
        dispersion: s.dispersion,
    });
    // Targets: drop aliased targets, reindex survivors into the reduced design.
    let targets_r: Vec<u32> = opts
        .se_for
        .columns(p)
        .iter()
        .filter(|&&t| !aliased[t as usize])
        .map(|&t| to_reduced[t as usize] as u32)
        .collect();
    // Per-row weights are unaffected by dropping fixed-effect COLUMNS, so the
    // rest of `opts` carries over unchanged via `..opts.clone()`.
    let opts_r = FitOptions {
        se_for: SeFor::Only(targets_r),
        ..opts.clone()
    };
    let (model_r, sources) = remap_spec_slopes(model, &kept);
    // Row-major `n × sources.len()`, indexed with the caller's full `p` and
    // `s`, as `Design::re_columns` reads them.
    let zr: Vec<f64> = (0..n)
        .flat_map(|i| {
            sources.iter().map(move |&c| match c.checked_sub(p) {
                None => x[i * p + c],
                Some(j) => re.expect("a slope past `p` reads `re_columns`")[i * s + j],
            })
        })
        .collect();
    let mut reduced = Data {
        design: Design {
            x: &xr,
            p: pk,
            ..data.design
        },
        ..*data
    };
    if !sources.is_empty() {
        reduced = reduced.re_columns(&zr, sources.len());
    }
    let fr = super::fit_checked(&reduced, &model_r, start_r.as_ref(), &opts_r);
    // Scatter reduced β/se to full width; aliased slots stay NaN.
    //
    // `aliased_out` starts as THIS level's mask and takes the reduced fit's mask
    // on top, in the same loop that scatters β. The OR keeps the field's
    // contract — NaN in β/se ⇔ the column was dropped — true no matter how deep
    // the chain runs: if the recursive `fit_checked` salvages again, `fr.aliased[r]`
    // is true for a column this level KEPT and `fr.beta[r]` is the NaN that goes
    // with it, and reporting only this level's mask would scatter that NaN into
    // a slot flagged `aliased == false`. The alias gate cannot fire twice
    // (see the recursion note above), so this is insurance rather than a live
    // path, but it is the cheap kind and the contract is worth pinning.
    let mut aliased_out = aliased.to_vec();
    let mut beta = vec![f64::NAN; p];
    let mut se = vec![f64::NAN; p];
    let mut coef_df = vec![f64::NAN; p];
    for (r, &orig) in kept.iter().enumerate() {
        beta[orig] = fr.beta[r];
        se[orig] = fr.se[r];
        coef_df[orig] = fr.coef_df[r];
        aliased_out[orig] |= fr.diagnostics.aliased[r];
    }
    // The Satterthwaite state stays in the reduced column space; its map to
    // the caller's columns goes through `kept` (composed with the reduced
    // fit's own map, should that fit have salvaged again).
    let satterthwaite = fr.satterthwaite.map(|mut st| {
        st.retained = Some(match st.retained.take() {
            Some(inner) => inner.into_iter().map(|r| kept[r]).collect(),
            None => kept.clone(),
        });
        st
    });
    // Every other diagnostic passes through from the reduced fit, but its note
    // column indices are REDUCED-design indices — translate them back through
    // `kept` so a caller can index them into the `x` it passed in.
    let mut diagnostics = fr.diagnostics;
    diagnostics.aliased = aliased_out;
    for note in &mut diagnostics.notes {
        match note {
            Note::IllConditioned { columns, .. }
            | Note::RobustSeUnidentified { columns }
            | Note::RobustDfSmall { columns, .. } => {
                for c in columns.iter_mut() {
                    *c = kept[*c as usize] as u32;
                }
            }
            // Carry no design-column indices — nothing to translate.
            Note::PirlsExhausted { .. }
            | Note::HessianSeFallback
            | Note::NbShapeUnsettled { .. }
            | Note::NbShapeAtLimit { .. }
            | Note::NonIntegerResponse { .. }
            | Note::SingleLevelGroupingDropped { .. }
            | Note::FewRowsForRandomEffects { .. }
            | Note::SingleLevelGrouping { .. }
            | Note::RepeatedGrouping { .. }
            | Note::SlopeConstantWithinLevels { .. }
            | Note::SeparatedByGrouping { .. }
            | Note::OneTrialPerLevel { .. }
            | Note::ExactProfileFallback
            | Note::GradientSearchFallback
            | Note::SeUnavailable
            | Note::PrecisionLimited
            | Note::Separated
            | Note::VarianceAtLimit
            | Note::DispersionAtLimit { .. }
            | Note::SatterthwaiteCurvature { .. }
            | Note::NewGroupingLevels { .. }
            | Note::RowsNotEstimable { .. }
            | Note::WeightPowerAtBound { .. }
            | Note::WeightPowerSearchFailures { .. }
            | Note::WeightPowerOutsideUnit { .. }
            | Note::WeightPowerImplausible { .. }
            | Note::RobustWithRandomSlope
            | Note::PredictSeModelBased => {}
        }
    }
    // Same scatter in two dimensions: an aliased column has no coefficient, so
    // it has no covariance with anything — its whole row and column stay NaN,
    // exactly as its `se` slot does.
    let scatter = |reduced: &[f64]| {
        let mut full = nan_vcov(p);
        for (ri, &oi) in kept.iter().enumerate() {
            for (rj, &oj) in kept.iter().enumerate() {
                full[oi * p + oj] = reduced[ri * pk + rj];
            }
        }
        full
    };
    let vcov = scatter(&fr.vcov);
    let model_vcov = if fr.model_vcov.is_empty() {
        Vec::new()
    } else {
        scatter(&fr.model_vcov)
    };
    // The robust pieces carry an aliased column, which has no coefficient,
    // only as a zero row of their `t`.
    let robust = fr.robust.map(|pc| pc.widen(&kept, p));
    // `condvar_x` rows are `p` wide: the same scatter, an aliased column NaN.
    let pr = kept.len();
    let mut condvar_x = vec![f64::NAN; fr.condvar_x.len() / pr.max(1) * p];
    for (row_r, row) in fr.condvar_x.chunks(pr.max(1)).zip(condvar_x.chunks_mut(p)) {
        for (r, &orig) in kept.iter().enumerate() {
            row[orig] = row_r[r];
        }
    }
    Fit {
        beta,
        se,
        vcov,
        tau2: fr.tau2,
        dispersion: fr.dispersion,
        diagnostics,
        varcorr: fr.varcorr,
        stddev_se: fr.stddev_se,
        n_eval: fr.n_eval,
        #[cfg(feature = "counters")]
        counters: fr.counters,
        deviance: fr.deviance,
        // Scalars/per-row/RE-shaped fields pass through: loglik and df come
        // from the REDUCED fit (aliased columns carry no parameter — lme4's
        // NA-coefficient df), fitted is per-row (the reduced model's means are
        // the model's means), ranef is RE-shaped (unchanged by fixed-column
        // drops).
        loglik: fr.loglik,
        df: fr.df,
        reml: fr.reml,
        weight_power: fr.weight_power,
        weight_power_estimated: fr.weight_power_estimated,
        se_type: fr.se_type,
        model_vcov,
        robust,
        nobs: fr.nobs,
        objective: fr.objective,
        fitted: fr.fitted,
        ranef: fr.ranef,
        ranef_levels: fr.ranef_levels,
        condvar: fr.condvar,
        condvar_x,
        condvar_cross: fr.condvar_cross,
        coef_df,
        satterthwaite,
        // On the reduced design, so its `p` is the retained column count.
        design_summary: fr.design_summary,
        // Filled by `fit_checked`, the salvage's only caller.
        null_basis: Vec::new(),
    }
}
/// The grouping ids of a mixed model checked against it: the primary id
/// vector is length `n`, the extra vectors align 1:1 with the declared extra
/// groupings, and each is length `n`. [`validate`] returns the error;
/// `build_workspace` panics with its message.
pub(super) fn check_group_ids(re: &ReStructure, ids: &GroupIds, n: usize) -> Result<(), FitError> {
    if ids.primary.len() != n {
        return Err(FitError::Ids(
            "GroupIds.primary must have n elements".into(),
        ));
    }
    if ids.extra.len() != re.extra_groupings.len() {
        return Err(FitError::Ids(
            "GroupIds.extra must align 1:1 with re.extra_groupings (declaration order)".into(),
        ));
    }
    if let Some(g) = ids.extra.iter().position(|e| e.len() != n) {
        return Err(FitError::Ids(
            format!("GroupIds.extra[{g}] must have n elements").into(),
        ));
    }
    Ok(())
}

/// Padded-per-parent child ids: given already-dense `parent_ids` (one per row)
/// and per-row child `labels`, lay the children out as CONTIGUOUS PER-PARENT
/// BLOCKS in the same order the parent's own ids use — parent id `p`'s children
/// occupy `[p·W, p·W + k_p)` where `k_p` is that parent's distinct child count
/// and `W = max_p k_p` (the true max — every block gets the same width, since
/// the kernel's `NestedWithin` sizing is a fixed-width rectangle: `src/lmm/kernel.rs`'s
/// `add_rows_multi` computes a child's RE column as
/// `extra_offsets[e] + id·extra_q[e]`, which only lands in the right parent's
/// block under this padded-block layout — a fresh global lexicographic sort of
/// the joined label does not preserve per-parent contiguity for unbalanced
/// nesting). Slots `k_p..W` in a shorter parent's block are never assigned to
/// any row (padding). Assumes each child label occurs under a SINGLE parent —
/// the caller must have verified nesting (explicit `parent:child` syntax, or
/// [`detect_flat_nesting`] for the flat idiom).
///
/// Returns the ids alongside one label per slot of the padded rectangle
/// (`n_parents · W`, slot `p·W + k`): the child label at every assigned slot and
/// `None` at every padded one. The caller joins the parent's own label in when
/// the formula named a parent (`(1|A/B)`); the flat idiom keeps the child label
/// bare, because `re_group_info` names that block for the child alone and the
/// route it takes is data-dependent.
pub(crate) fn nested_padded_ids<L: Ord + Hash + Clone>(
    parent_ids: &[u32],
    labels: &[L],
) -> (Vec<u32>, Vec<Option<L>>) {
    let n_parents = parent_ids
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(1);
    // Distinct child labels observed under each parent id, lexicographic
    // (BTreeSet) — the within-parent dense order; the specific order among
    // children of one parent doesn't matter statistically (an exchangeable RE),
    // only that it is dense over `0..k_p`.
    let mut children_per_parent: Vec<BTreeSet<&L>> = vec![BTreeSet::new(); n_parents];
    for (&p, c) in parent_ids.iter().zip(labels) {
        children_per_parent[p as usize].insert(c);
    }
    let n_per_parent = children_per_parent
        .iter()
        .map(BTreeSet::len)
        .max()
        .unwrap_or(1)
        .max(1);
    let local_index: Vec<HashMap<&L, u32>> = children_per_parent
        .iter()
        .map(|set| {
            set.iter()
                .enumerate()
                .map(|(i, &l)| (l, i as u32))
                .collect()
        })
        .collect();
    let mut slot_labels: Vec<Option<L>> = vec![None; n_parents * n_per_parent];
    for (p, set) in children_per_parent.iter().enumerate() {
        for (k, &child) in set.iter().enumerate() {
            slot_labels[p * n_per_parent + k] = Some(child.clone());
        }
    }
    let ids = parent_ids
        .iter()
        .zip(labels)
        .map(|(&p, c)| p * n_per_parent as u32 + local_index[p as usize][c])
        .collect();
    (ids, slot_labels)
}

/// Max padding-inflation factor for flat-nesting detection: detect nested only
/// while `n_parents · max_children ≤ NESTING_INFLATION_BOUND ·
/// n_distinct_child_levels`. The guard protects the padded rectangle from
/// inflating the dense RE block — not balance per se: near-balanced nesting
/// (children-per-parent ∈ {1,2,3}, inflation ~1.6×) pads a few empty slots and
/// stays far cheaper than routing its tens of thousands of levels crossed
/// (dense tail is cubic in crossed levels), while an observation-level factor
/// like grouseticks' INDEX (each row its own level, inflation ≫ 2) still fails
/// closed to `Crossed` (measured there: nested 44 s vs crossed 0.16 s).
pub(crate) const NESTING_INFLATION_BOUND: usize = 2;

/// Detect nesting of a flat extra grouping (the lme4 idiom
/// `(1|parent)+(1|child)`, no explicit `parent:child` syntax) from the observed
/// id structure, returning its [`nested_padded_ids`] layout iff routing it
/// nested is both correct AND a win. Two guards, both fail closed to `Crossed`:
///
/// 1. **Genuine nesting** — every distinct `child_labels` value must fall under a
///    single `primary_ids` (parent). A label spanning two parents is genuinely
///    crossed; routing it nested would corrupt the padded family-block Cholesky.
/// 2. **Bounded padding inflation** — the padded rectangle
///    `n_parents · max_children` must not exceed [`NESTING_INFLATION_BOUND`] ×
///    the distinct child-level count (see the constant's rationale). Explicit
///    `parent:child` syntax still pads unbounded — there the user asked for the
///    nested layout outright.
///
/// The fit runs the same check on the grouping a slope grouping displaces when
/// [`spec_sized_from_ids`] moves that slope grouping first.
pub(crate) fn detect_flat_nesting<L: Ord + Hash + Clone>(
    primary_ids: &[u32],
    child_labels: &[L],
) -> Option<(Vec<u32>, Vec<Option<L>>)> {
    let n_parents = primary_ids
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(1);
    let mut parent_of: HashMap<&L, u32> = HashMap::new();
    let mut children_per_parent = vec![0usize; n_parents];
    for (&p, c) in primary_ids.iter().zip(child_labels) {
        match parent_of.insert(c, p) {
            Some(prev) if prev != p => return None, // spans two parents → crossed
            Some(_) => {}                           // repeat obs of a seen child
            None => children_per_parent[p as usize] += 1, // first sighting
        }
    }
    let w = children_per_parent.iter().copied().max().unwrap_or(0);
    if n_parents * w > NESTING_INFLATION_BOUND * parent_of.len() {
        return None; // padded rectangle would inflate the dense RE block
    }
    Some(nested_padded_ids(primary_ids, child_labels))
}

/// The widest parent's distinct child count `W` of a nested grouping, the
/// width of each parent's block of child slots.
fn nested_width(primary: &[u32], child: &[u32], n_primary: usize) -> usize {
    let mut per_parent: Vec<std::collections::HashSet<u32>> = vec![Default::default(); n_primary];
    for (&p, &c) in primary.iter().zip(child) {
        per_parent[p as usize].insert(c);
    }
    per_parent.iter().map(|s| s.len()).max().unwrap_or(1).max(1)
}

/// The child ids of a fit's nested groupings: `parent·W + within` (`W` from
/// [`nested_width`]), the slot layout the kernels read, so a child id never
/// lands in another parent's slots or past the block. Fit-only: `predict` and
/// `Simulator` pass a fit's level ids on fewer rows, whose `W` differs.
fn check_nested_ids(re: &ReStructure, ids: &GroupIds) -> Result<(), FitError> {
    let n_primary = ids
        .primary
        .iter()
        .copied()
        .max()
        .map_or(1, |m| m as usize + 1);
    for (g, (gr, child)) in re.extra_groupings.iter().zip(&ids.extra).enumerate() {
        if gr.relation != GroupingRelation::NestedWithin {
            continue;
        }
        let w = nested_width(&ids.primary, child, n_primary);
        let bad = child
            .iter()
            .zip(&ids.primary)
            .position(|(&c, &p)| c as usize / w != p as usize);
        if let Some(i) = bad {
            return Err(FitError::Ids(ErrorMessage::at_row(
                format!(
                    "GroupIds.extra[{g}] is nested: a child id must be parent * {w} + its \
                     position within the parent, got {} under parent {} (row ",
                    child[i], ids.primary[i]
                ),
                i,
                ")",
            )));
        }
    }
    Ok(())
}

/// Every primary and crossed id below `n`: a level count is `max(id) + 1`, so
/// one huge id would size billions of levels and abort on allocation, past any
/// `catch_unwind`. The formula frontend numbers only observed levels, so its
/// ids always pass. Nested child ids are `parent·W + within` and can reach past
/// `n`; [`check_nested_ids`] bounds them. Fit-only, for the reason that one is.
fn check_id_bound(re: &ReStructure, ids: &GroupIds, n: usize) -> Result<(), FitError> {
    let crossed = (re.extra_groupings.iter().zip(&ids.extra).enumerate())
        .filter(|(_, (gr, _))| gr.relation != GroupingRelation::NestedWithin)
        .map(|(g, (_, v))| (format!("extra[{g}]"), v));
    for (name, v) in std::iter::once(("primary".to_string(), &ids.primary)).chain(crossed) {
        if let Some(i) = v.iter().position(|&id| id as usize >= n) {
            return Err(FitError::Ids(ErrorMessage::at_row(
                format!(
                    "GroupIds.{name}: id {} is not below the row count {n} (row ",
                    v[i]
                ),
                i,
                "); number the levels 0, 1, 2, ... with no gaps",
            )));
        }
    }
    Ok(())
}

/// The message `what (row k)` of the first row `k` whose value fails `ok`.
fn bad_row(v: &[f64], ok: impl Fn(f64) -> bool, what: &str) -> Option<ErrorMessage> {
    v.iter()
        .position(|&a| !ok(a))
        .map(|i| ErrorMessage::at_row(format!("{what} (row "), i, ")"))
}

/// The message `what (row k, value v; glmm::limits::limit)` of the first row
/// `k` whose value fails `ok`.
fn out_of_range(
    v: &[f64],
    ok: impl Fn(f64) -> bool,
    what: &str,
    limit: &str,
) -> Option<ErrorMessage> {
    v.iter().position(|&a| !ok(a)).map(|i| {
        let after = format!(", value {:e}; glmm::limits::{limit})", v[i]);
        ErrorMessage::at_row(format!("{what} (row "), i, &after)
    })
}

/// Within [`MAX_DATA_ABS`]: false for NaN and ±inf, so one test covers
/// finiteness and range.
fn in_data_range(v: f64) -> bool {
    v.abs() <= MAX_DATA_ABS
}

/// The input check [`super::fit`] and [`super::fit_with_start`] share: every
/// caller mistake in `data`, `spec`, `start` and `opts` as a [`FitError`],
/// before anything is fitted. What passes here reaches the kernels' own
/// invariants intact, so a panic past this point is a crate bug. Returns the
/// weight power and the standard error `opts` resolve to.
pub(crate) fn validate(
    data: &Data<'_>,
    spec: &ModelSpec,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Result<(WeightPower, SeType), FitError> {
    // Here and not in `validate_design`: a predict of zero new rows is legal.
    // Bind's `check_rows` (glmm-bind/src/fns/mod.rs) and `formula::materialize`
    // repeat this text - change together.
    if data.design.n == 0 {
        return Err(FitError::Shape(
            "Data has no rows (n = 0); a fit needs at least one row".into(),
        ));
    }
    validate_design(&data.design, spec, opts.nagq, true)?;
    if let (Some(re), Some(ids)) = (spec.re.as_ref(), data.design.ids) {
        // First: `check_nested_ids` sizes a buffer by the largest primary id.
        check_id_bound(re, ids, data.design.n)?;
        check_nested_ids(re, ids)?;
    }
    let Design { n, p, .. } = data.design;
    let y = data.y;
    if y.len() != n {
        return Err(FitError::Shape("y must have n elements".into()));
    }
    if let Some(m) = out_of_range(
        y,
        in_data_range,
        &format!("y must be finite with |y| <= {MAX_DATA_ABS:e}"),
        "MAX_DATA_ABS",
    ) {
        return Err(FitError::ResponseOutOfSupport(m));
    }
    // An EMPTY field means "cold-start this component" — a caller that knows only
    // one of the two (an R `start = list(theta = …)`, lme4's shape) cannot compute
    // the other: the cold β seed is a no-RE GLM fit and the cold θ seed is the
    // blind THETA0 shape, both of which live inside the kernels. So the widths
    // accepted are {0, exact}, and each consumption site falls back on empty.
    if let Some(s) = start {
        let start_err = |m: &str| Err(FitError::Start(m.into()));
        if !(s.beta.is_empty() || s.beta.len() == p) {
            return start_err("StartValues.beta must have p elements, or be empty to cold-start β");
        }
        if !s.beta.iter().all(|v| v.is_finite()) {
            return start_err("StartValues.beta must be finite");
        }
        let n_theta = theta_width(spec.re.as_ref());
        if !(s.theta.is_empty() || s.theta.len() == n_theta) {
            return start_err(
                "StartValues.theta must have n_theta elements for this RE structure, \
                 or be empty to cold-start θ",
            );
        }
        if !s.theta.iter().all(|v| v.is_finite()) {
            return start_err("StartValues.theta must be finite");
        }
        if !s.dispersion.is_none_or(|d| d.is_finite() && d > 0.0) {
            return start_err(
                "StartValues.dispersion must be finite and > 0, or None to cold-start it",
            );
        }
        if let (Family::NegativeBinomial { .. }, Some(d)) = (spec.family, s.dispersion) {
            if !(NB_THETA_LO..=NB_THETA_HI).contains(&d) {
                return start_err(&format!(
                    "warm-start dispersion (the negative-binomial shape seed) must be within \
                     [{NB_THETA_LO:e}, {NB_THETA_HI:e}] (value {d:e}; glmm::limits::NB_THETA_LO, \
                     glmm::limits::NB_THETA_HI)"
                ));
            }
        }
    }
    // Response domain: refuse a response outside the family's support before
    // fitting, rather than let the kernel converge on numbers a valid model
    // could never have produced (a Poisson y < 0 reaching the kernel gives
    // `converged = true` with logLik −∞). Binomial's `y` is always the
    // success PROPORTION (`crate::family`'s aggregated convention: row `i` is
    // `wᵢ` trials with `wᵢ·yᵢ` successes, unit weights ⇒ Bernoulli), so
    // `0..=1` covers both the raw 0/1 case and "successes above trials".
    let refused = |ok: fn(f64) -> bool, what: &str| bad_row(y, ok, what);
    let out_of_support = match spec.family {
        Family::Gaussian => None,
        Family::Poisson { .. } => refused(|v| v >= 0.0, "Poisson response must be >= 0"),
        Family::NegativeBinomial { .. } => {
            refused(|v| v >= 0.0, "negative-binomial response must be >= 0")
        }
        Family::Binomial { .. } => refused(
            |v| (0.0..=1.0).contains(&v),
            "binomial response (the success proportion) must be in [0, 1]",
        ),
        Family::Gamma { .. } => refused(|v| v > 0.0, "Gamma response must be > 0"),
        Family::InverseGaussian { .. } => {
            refused(|v| v > 0.0, "inverse-Gaussian response must be > 0")
        }
    };
    if let Some(m) = out_of_support {
        return Err(FitError::ResponseOutOfSupport(m));
    }
    // Dispersion: refuse a value no family could ever use (a held φ is a
    // variance-family scale, so it must be a finite positive number). A
    // family with no φ to hold (Gaussian, Poisson, binomial, NB) just ignores
    // the field, as its doc says — the ports already clear it with an
    // `argument_ignored` warning before it gets here.
    if let Some(d) = opts.dispersion.held() {
        if !(d.is_finite() && d > 0.0) {
            return Err(FitError::OptionValue(
                "FitOptions.dispersion must be finite and > 0".into(),
            ));
        }
        match spec.family {
            Family::InverseGaussian { .. } => {
                let lo = 1.0 / MAX_DATA_ABS;
                if !(lo..=MAX_DATA_ABS).contains(&d) {
                    return Err(FitError::OptionValue(
                        format!(
                            "dispersion on an inverse-Gaussian model must be within \
                             [{lo:e}, {MAX_DATA_ABS:e}] (value {d:e}; glmm::limits::MAX_DATA_ABS)"
                        )
                        .into(),
                    ));
                }
            }
            Family::Gamma { .. } => {
                // The GLM and the GLMM both use a held φ as `d / s` on the
                // internal weight scale, so the box bounds that value; the
                // message gives the range in the caller's units.
                let s = crate::family::weight_scale(data.design.weights, n);
                if !(GAMMA_PHI_LO..=GAMMA_PHI_HI).contains(&(d / s)) {
                    return Err(FitError::OptionValue(
                        format!(
                            "dispersion on a Gamma model must be within [{:e}, {:e}] \
                             (value {d:e}; glmm::limits::GAMMA_PHI_LO, glmm::limits::GAMMA_PHI_HI)",
                            GAMMA_PHI_LO * s,
                            GAMMA_PHI_HI * s
                        )
                        .into(),
                    ));
                }
            }
            _ => {}
        }
    }
    if let SeFor::Only(cols) = &opts.se_for {
        if let Some(&c) = cols.iter().find(|&&c| c as usize >= p) {
            return Err(FitError::OptionValue(
                format!("FitOptions.se_for column {c} out of range (p={p})").into(),
            ));
        }
    }
    let power = resolve_weight_power(spec, data, opts)?;
    let se = resolve_se_type(spec, power, opts.reml != Some(false), opts)?;
    Ok((power, se))
}

/// The part of [`validate`] that reads no response: the design `x` and the
/// random-effect-only columns against `n`, `p` and `s`, the model's shape
/// ([`check_model_shape`] at `nagq`), its slope columns, the grouping ids, the
/// weights and the offset. `crate::predict` runs it on new rows, with
/// `ids_required` false where it reads no random effect.
pub(crate) fn validate_design(
    design: &Design<'_>,
    spec: &ModelSpec,
    nagq: u8,
    ids_required: bool,
) -> Result<(), FitError> {
    let &Design {
        x,
        n,
        p,
        re,
        s,
        ids,
        weights,
        offset,
    } = design;
    if n.checked_mul(p) != Some(x.len()) {
        return Err(FitError::Shape(
            "x must have n*p elements in row-major layout".into(),
        ));
    }
    if let Some(k) = x.iter().position(|&v| !in_data_range(v)) {
        return Err(FitError::Shape(ErrorMessage::at_row(
            format!("x must be finite with |x| <= {MAX_DATA_ABS:e} (row "),
            k / p,
            &format!(
                ", column {}, value {:e}; glmm::limits::MAX_DATA_ABS)",
                k % p,
                x[k]
            ),
        )));
    }
    let z = re.unwrap_or(&[]);
    if n.checked_mul(s) != Some(z.len()) {
        return Err(FitError::Shape(
            "re_columns must have n*s elements in row-major layout".into(),
        ));
    }
    if let Some(k) = z.iter().position(|&v| !in_data_range(v)) {
        return Err(FitError::Shape(ErrorMessage::at_row(
            format!("re_columns must be finite with |re_columns| <= {MAX_DATA_ABS:e} (row "),
            k / s,
            &format!(
                ", column {}, value {:e}; glmm::limits::MAX_DATA_ABS)",
                k % s,
                z[k]
            ),
        )));
    }
    check_model_shape(spec, nagq)?;
    check_slope_columns(spec, p, s)?;
    // Empty ids count as none: the formula door hands back `GroupIds::default()`
    // when the lowering dropped every grouping, and that fit is fixed-only.
    match (spec.re.as_ref(), ids) {
        (Some(re), Some(ids)) => check_group_ids(re, ids, n)?,
        (Some(_), None) if ids_required => {
            return Err(FitError::Ids(
                "a model with random effects needs GroupIds (Data::ids)".into(),
            ))
        }
        (None, Some(ids)) if !(ids.primary.is_empty() && ids.extra.is_empty()) => {
            return Err(FitError::Ids(
                "GroupIds given for a model without random effects".into(),
            ))
        }
        (Some(_), None) | (None, _) => {}
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(FitError::Weights(
                "Data weights must have n elements".into(),
            ));
        }
        let lo = 1.0 / MAX_DATA_ABS;
        if let Some(m) = out_of_range(
            w,
            |v| (lo..=MAX_DATA_ABS).contains(&v),
            &format!("Data weights must be within [{lo:e}, {MAX_DATA_ABS:e}]"),
            "MAX_DATA_ABS",
        ) {
            return Err(FitError::Weights(m));
        }
        if let Family::Binomial { .. } = spec.family {
            if let Some(m) = out_of_range(
                w,
                |v| v <= MAX_TRIALS,
                &format!("binomial weights (trial counts) must be <= {MAX_TRIALS:e}"),
                "MAX_TRIALS",
            ) {
                return Err(FitError::Weights(m));
            }
        }
    }
    if let Some(o) = offset {
        if o.len() != n {
            return Err(FitError::Offset("Data offset must have n elements".into()));
        }
        let (limit, name) = if crate::family::is_log_link(spec.family) {
            (MAX_LOG_OFFSET_ABS, "MAX_LOG_OFFSET_ABS")
        } else {
            (MAX_DATA_ABS, "MAX_DATA_ABS")
        };
        if let Some(m) = out_of_range(
            o,
            |v| v.abs() <= limit,
            &format!("Data offset must be finite with |offset| <= {limit:e} on this link"),
            name,
        ) {
            return Err(FitError::Offset(m));
        }
    }
    Ok(())
}

/// The LMM criterion `opts.reml` resolves to on `model` (the
/// [`FitOptions::reml`] table): `Some` on a Gaussian mixed model, REML for
/// `None` and `Some(true)` and ML for `Some(false)`; `None` off the LMM path,
/// whatever the value, so no value panics and `Some(true)` on a GLM or GLMM is
/// ignored.
pub(crate) fn resolve_criterion(model: &ModelSpec, opts: &FitOptions) -> Option<Criterion> {
    match (model.family, &model.re) {
        (Family::Gaussian, Some(_)) => Some(match opts.reml {
            Some(false) => Criterion::Ml,
            None | Some(true) => Criterion::Reml,
        }),
        _ => None,
    }
}

/// The fewest fitted rows a fitted weight power takes: the smallest sample
/// the method was simulated on.
const MIN_ESTIMATE_ROWS: usize = 20;

/// Weights whose largest over smallest is at most this count as all equal:
/// the power's profile is flat to rounding.
const EQUAL_WEIGHTS: f64 = 1.0 + 1e-12;

/// The largest weight over the smallest.
pub(crate) fn weight_spread(w: &[f64]) -> f64 {
    let (lo, hi) = w.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    hi / lo
}

/// The first row whose weight raised to `v` is not finite and `> 0`.
pub(crate) fn first_bad_power(w: &[f64], v: f64) -> Option<usize> {
    w.iter().position(|wi| {
        let p = wi.powf(v);
        !(p.is_finite() && p > 0.0)
    })
}

/// [`FitOptions::weight_power`] resolved on `spec` and `data`, or the refusal
/// of a named value. `None` is `Estimate` where a named `Estimate` would fit
/// and the fit is by REML (`reml = None` read as REML), else `Fixed(1.0)`.
pub(crate) fn resolve_weight_power(
    spec: &ModelSpec,
    data: &Data<'_>,
    opts: &FitOptions,
) -> Result<WeightPower, FitError> {
    let weights = data.design.weights;
    match opts.weight_power {
        None if opts.reml != Some(false) && estimate_refusal(spec, data).is_none() => {
            Ok(WeightPower::Estimate)
        }
        None => Ok(WeightPower::Fixed(1.0)),
        Some(WeightPower::Fixed(v)) => {
            if !v.is_finite() {
                return Err(FitError::OptionValue(
                    format!("weight_power must be a finite number, got {v}").into(),
                ));
            }
            // Without weights every row has w = 1, so the value changes nothing.
            let Some(w) = weights.filter(|_| v != 1.0) else {
                return Ok(WeightPower::Fixed(v));
            };
            if spec.family != Family::Gaussian {
                return Err(FitError::Unsupported(
                    format!(
                        "weight_power = {v} applies only to Gaussian models: on {} the \
                         weights are the family's prior weights; use weight_power = 1",
                        spec.family.words()
                    )
                    .into(),
                ));
            }
            if let Some(i) = first_bad_power(w, v) {
                return Err(FitError::Weights(ErrorMessage::at_row(
                    format!(
                        "Data weights raised to weight_power = {v} must be finite and > 0 (row "
                    ),
                    i,
                    "); use a weight_power nearer 0",
                )));
            }
            Ok(WeightPower::Fixed(v))
        }
        Some(WeightPower::Estimate) => match estimate_refusal(spec, data) {
            Some(e) => Err(e),
            None => Ok(WeightPower::Estimate),
        },
    }
}

/// Why `spec` and `data` cannot take a fitted weight power, if they cannot.
fn estimate_refusal(spec: &ModelSpec, data: &Data<'_>) -> Option<FitError> {
    let what = r#"weight_power = "estimate""#;
    let instead = "use weight_power = 1";
    if let Err(e) = one_grouping_gaussian(spec, what, instead) {
        return Some(e);
    }
    let Some(w) = data.design.weights else {
        return Some(FitError::Weights(
            format!("{what} needs weights to fit the power on; give weights, or drop weight_power")
                .into(),
        ));
    };
    if weight_spread(w) <= EQUAL_WEIGHTS {
        return Some(FitError::Weights(
            format!("{what} needs weights that are not all equal; {instead}").into(),
        ));
    }
    let n = data.design.n;
    (n < MIN_ESTIMATE_ROWS).then(|| {
        FitError::Unsupported(
            format!(
                "{what} needs at least {MIN_ESTIMATE_ROWS} rows, the smallest sample \
                 it was checked on (got {n}); {instead}"
            )
            .into(),
        )
    })
}

/// [`FitOptions::se_type`] resolved after the weight power `power`, on a fit
/// by REML when `reml`, or the refusal of a named value. `None` is `Robust` on
/// a REML δ fit with no random slope, else `Model`.
pub(crate) fn resolve_se_type(
    spec: &ModelSpec,
    power: WeightPower,
    reml: bool,
    opts: &FitOptions,
) -> Result<SeType, FitError> {
    let delta_fit = power == WeightPower::Estimate;
    match opts.se_type {
        None if delta_fit && reml && !has_random_slope(spec) => Ok(SeType::Robust),
        None | Some(SeType::Model) => Ok(SeType::Model),
        Some(SeType::Robust) => {
            let what = r#"se_type = "robust""#;
            let instead = r#"use se_type = "model""#;
            one_grouping_gaussian(spec, what, instead)?;
            if !reml && (spec.re.is_some() || delta_fit) {
                return Err(FitError::Unsupported(
                    format!(
                        "{what} needs a REML fit: robust standard errors were checked under \
                         REML only; drop reml = false, or {instead}"
                    )
                    .into(),
                ));
            }
            Ok(SeType::Robust)
        }
    }
}

/// Whether the primary grouping has a random slope (any term's column).
pub(crate) fn has_random_slope(spec: &ModelSpec) -> bool {
    spec.re.as_ref().is_some_and(|re| !re.slopes.is_empty())
}

/// The model shape a fitted weight power and a robust standard error take:
/// Gaussian, with at most one grouping factor (any number of terms on it).
fn one_grouping_gaussian(spec: &ModelSpec, what: &str, instead: &str) -> Result<(), FitError> {
    if spec.family != Family::Gaussian {
        return Err(FitError::Unsupported(
            format!(
                "{what} applies only to Gaussian models, not {}; {instead}",
                spec.family.words()
            )
            .into(),
        ));
    }
    if spec
        .re
        .as_ref()
        .is_some_and(|re| !re.extra_groupings.is_empty())
    {
        return Err(FitError::Unsupported(
            format!(
                "{what} needs at most one grouping factor, not crossed or nested groupings; \
                 {instead}"
            )
            .into(),
        ));
    }
    Ok(())
}

/// The model-shape part of [`validate`], which `build_workspace` also runs
/// (and panics on): the node count, the shapes no kernel fits and each
/// grouping's terms ([`check_terms`]). The slope columns are checked against
/// the data in [`check_slope_columns`]. The RE-envelope
/// caps are not checked here: over them a design routes to the sparse path
/// ([`super::classify_design`]).
pub(crate) fn check_model_shape(model: &ModelSpec, nagq: u8) -> Result<(), FitError> {
    // nAGQ: odd, 1..=MAX_NAGQ; >1 only on a single grouping factor (no extras),
    // q_p ≤ 3, binomial/Poisson/negative-binomial/Gamma GLMM — the shapes whose
    // marginal likelihood is a product of independent per-cluster q-D integrals of
    // a `−½·Σ dev_resid` integrand (`glmm::agq`). NB qualifies because θ_NB enters
    // the conditional density only through a per-row saturated term that is
    // constant in u, so it factors out of every cluster integral and the driver
    // adds it back outside (`nb_term`). Gamma qualifies too: φ divides the
    // deviance inside the integral, which the kernels carry through the prior
    // weights `wᵢ/φ`, and the rest of the log-density depends on φ alone and sits
    // outside, as NB's term does. The range is checked on every model, so even
    // a fixed-only spec cannot smuggle a bad nagq through; a Gaussian or
    // fixed-only model has no integral to approximate, and its kernels never
    // read nagq, so a valid nagq > 1 there fits at Laplace's answer.
    // `nagq` controls how the fit is computed, not what model is being fit, so
    // it is sourced from `FitOptions`, not the spec. Mirrors the ports'
    // warn-and-strip boundary (`orchestrate::lower_with`) — change together.
    let nagq_range = || {
        format!(
            "nagq={} must be odd in 1..={}",
            nagq,
            crate::consts::MAX_NAGQ
        )
    };
    if nagq % 2 == 0 {
        return Err(FitError::OptionValue(nagq_range().into()));
    }
    if nagq > crate::consts::MAX_NAGQ {
        return Err(FitError::Envelope(nagq_range().into()));
    }
    let Some(re) = model.re.as_ref() else {
        return Ok(());
    };
    if nagq > 1 && model.family != Family::Gaussian {
        let agq_family = matches!(
            model.family,
            Family::Binomial { .. }
                | Family::Poisson { .. }
                | Family::NegativeBinomial { .. }
                | Family::Gamma { .. }
        );
        // Each refusal names the condition that failed.
        if !agq_family {
            return Err(FitError::Unsupported(format!(
                "nagq>1 applies only to binomial, Poisson, negative-binomial and Gamma GLMMs, not {:?}",
                model.family
            ).into()));
        }
        if !re.extra_groupings.is_empty() {
            return Err(FitError::Unsupported(
                "nagq>1 needs a single grouping factor (no crossed or nested extra groupings)"
                    .into(),
            ));
        }
        // q_p = 1 + #primary slopes. The q_p ≥ 4 refusal is a TEMPORARY cost /
        // oracle-coverage boundary (the k^q product grid and the dimension-generic
        // loop have no q limit), not a code limit — lifting it later is deleting
        // this one check. Locked decision: cap AGQ to q_p≤3 until oracle coverage
        // extends past it, rather than let uncapped q_p through untested.
        let q_p = 1 + re.slopes.len();
        if q_p > 3 {
            return Err(FitError::Unsupported(
                format!(
                    "nagq>1 with q_p={q_p} random effects per group exceeds the temporary \
                 q_p≤3 cap (a cost/oracle-coverage boundary, not a code limit)"
                )
                .into(),
            ));
        }
    }
    // Inverse-Gaussian mixed models are deliberately not built: the GLMM
    // objective needs the profiled `inverse.gaussian()$aic` term, which the
    // kernel does not have. Refused here rather than in `build_workspace`'s
    // dispatch, so no workspace is allocated for a model that cannot be fit.
    if matches!(model.family, Family::InverseGaussian { .. }) {
        return Err(FitError::Unsupported(
            "inverse-Gaussian mixed models are not implemented (fixed-effects GLM only)".into(),
        ));
    }
    check_terms(&re.slopes, &re.terms, "primary grouping")?;
    for (k, g) in re.extra_groupings.iter().enumerate() {
        check_terms(&g.slopes, &g.terms, &format!("extra grouping {k}"))?;
    }
    // The kernel holds ONE nested slot (`LmmGroupings.nested`) and
    // `from_cluster_spec_ext` would silently let a later `NestedWithin` extra
    // overwrite an earlier one (last wins). The formula frontend detects at most
    // one; this guards the explicit `parent:child` route (defense in depth).
    let n_nested = re
        .extra_groupings
        .iter()
        .filter(|g| matches!(g.relation, GroupingRelation::NestedWithin))
        .count();
    if n_nested > 1 {
        return Err(FitError::Unsupported(
            format!("at most one NestedWithin extra grouping is supported (got {n_nested})").into(),
        ));
    }
    Ok(())
}

/// The slope indices of `model` against the data's `p` design columns and `s`
/// random-effect-only columns (`Design::re_columns`): every index below
/// `p + s`, and every one of the `s` columns read by some slope, so the data's
/// `s` is the [`re_only_count`] `build_workspace` sizes the workspace by. A
/// model without random effects reads none.
fn check_slope_columns(model: &ModelSpec, p: usize, s: usize) -> Result<(), FitError> {
    let mut read = vec![false; s];
    if let Some(re) = model.re.as_ref() {
        let extra = re
            .extra_groupings
            .iter()
            .map(|g| ("extra-grouping", &g.slopes));
        for (what, slopes) in std::iter::once(("primary", &re.slopes)).chain(extra) {
            for &c in slopes {
                let c = c as usize;
                if c >= p + s {
                    return Err(FitError::Shape(
                        format!("{what} slope column {c} out of range (p={p}, s={s})").into(),
                    ));
                }
                if c >= p {
                    read[c - p] = true;
                }
            }
        }
    }
    if let Some(j) = read.iter().position(|&r| !r) {
        return Err(FitError::Shape(
            format!(
                "re_columns column {j} (slope index {}) is read by no random slope",
                p + j
            )
            .into(),
        ));
    }
    Ok(())
}

/// The number of random-effect-only columns a model with these slope indices
/// reads over a `p`-wide design: the highest index past the design, plus one,
/// less `p` (0 when every slope is a design column).
pub(crate) fn re_only_count(slopes: impl IntoIterator<Item = usize>, p: usize) -> usize {
    slopes
        .into_iter()
        .map(|c| c + 1)
        .max()
        .map_or(0, |m| m.saturating_sub(p))
}

/// One block's terms against its `slopes` (see [`ReStructure::terms`]): every
/// term non-empty, the intercept in at most one term and that term first, no
/// column in two terms, `slopes` the terms' columns concatenated, which
/// is the order `LmmGroupings` assigns block columns to terms in, and no
/// column twice in `slopes`. Empty `terms` passes the term checks.
/// `grouping` names the block in the error.
fn check_terms(slopes: &[ColumnId], terms: &[ReTerm], grouping: &str) -> Result<(), FitError> {
    let refuse = |why: String| Err(FitError::Shape(format!("{grouping}: {why}").into()));
    for (j, t) in terms.iter().enumerate() {
        if !t.intercept && t.slopes.is_empty() {
            return refuse(format!("term {j} has neither the intercept nor a slope"));
        }
        if t.intercept && j > 0 {
            return refuse(format!(
                "the intercept is in term {j}; it must be in term 0, and in one term only"
            ));
        }
        let earlier = terms[..j].iter().flat_map(|e| &e.slopes);
        if let Some(c) = t.slopes.iter().find(|c| earlier.clone().any(|e| e == *c)) {
            return refuse(format!("slope column {c} is in two terms"));
        }
    }
    if !terms.is_empty() && !slopes.iter().eq(terms.iter().flat_map(|t| &t.slopes)) {
        return refuse(format!(
            "slopes {slopes:?} are not the terms' columns in term order"
        ));
    }
    // A repeated column would make a covariance block over two copies of it.
    if let Some((_, c)) = (slopes.iter().enumerate()).find(|&(j, c)| slopes[..j].contains(c)) {
        return refuse(format!("slope column {c} is repeated"));
    }
    Ok(())
}

/// The random-effect structures these data cannot identify, as notes:
/// [`Note::SingleLevelGrouping`], [`Note::RepeatedGrouping`],
/// [`Note::OneTrialPerLevel`], [`Note::SlopeConstantWithinLevels`] and
/// [`Note::SeparatedByGrouping`] (their docs give the exact conditions),
/// ordered by grouping index, then in that kind order. A level is a distinct
/// id of the grouping's id vector; a nested id already includes its parent.
/// Values are compared with `==`. Reads only the data and the spec, so no
/// estimate depends on it. Empty on a fixed-only spec.
pub(super) fn identification_notes(design: &Design<'_>, y: &[f64], spec: &ModelSpec) -> Vec<Note> {
    let (Some(re), Some(ids), true) = (spec.re.as_ref(), design.ids, design.n > 0) else {
        return Vec::new();
    };
    let Design { x, n, p, s, .. } = *design;
    let blocks: Vec<(&[u32], &[ColumnId], &[ReTerm])> =
        std::iter::once((&ids.primary[..], &re.slopes[..], &re.terms[..]))
            .chain(
                (re.extra_groupings.iter().zip(&ids.extra))
                    .map(|(g, id)| (&id[..], &g.slopes[..], &g.terms[..])),
            )
            .collect();
    // `validate_design` refuses a slope index past `p + s`, so an index at or
    // past `p` has random-effect-only columns to read.
    let value = |c: ColumnId, i: usize| match (c as usize).checked_sub(p) {
        None => x[i * p + c as usize],
        Some(j) => design.re.expect("a slope index past p reads re_columns")[i * s + j],
    };
    let tri = |m: usize| m * (m + 1) / 2;
    let mut notes = Vec::new();
    for (g, &(gid, slopes, terms)) in blocks.iter().enumerate() {
        let grouping = g as u32;
        if gid.iter().all(|&l| l == gid[0]) {
            notes.push(Note::SingleLevelGrouping { grouping });
        }
        let intercept = !crate::lmm::block_hidden(terms);
        for (h, &(hid, h_slopes, h_terms)) in blocks.iter().enumerate().skip(g + 1) {
            let shared = (intercept && !crate::lmm::block_hidden(h_terms))
                || slopes.iter().any(|c| h_slopes.contains(c));
            if shared && same_partition(gid, hid) {
                notes.push(Note::RepeatedGrouping {
                    first: grouping,
                    second: h as u32,
                });
            }
        }
        let width = gid.iter().max().map_or(0, |&m| m as usize + 1);
        // Trials per level: the summed prior weights, which are the trial
        // counts of an aggregated binomial response (1 per row unweighted).
        let mut trials = vec![0.0; width];
        for (i, &l) in gid.iter().enumerate() {
            trials[l as usize] += design.weights.map_or(1.0, |w| w[i]);
        }
        let binomial = matches!(spec.family, Family::Binomial { .. });
        if binomial && trials.iter().all(|&t| t <= 1.0) {
            notes.push(Note::OneTrialPerLevel { grouping });
        }
        // Without a random intercept a slope constant within levels is a
        // random intercept with a per-level scale, which is identified; the
        // separation argument below needs the intercept too.
        if !intercept {
            continue;
        }
        let mut first_row = vec![usize::MAX; width];
        for (i, &l) in gid.iter().enumerate() {
            if first_row[l as usize] == usize::MAX {
                first_row[l as usize] = i;
            }
        }
        let first = |i: usize| first_row[gid[i] as usize];
        let constant: Vec<usize> = (0..slopes.len())
            .filter(|&k| (0..n).all(|i| value(slopes[k], i) == value(slopes[k], first(i))))
            .collect();
        if !constant.is_empty() {
            // The level effects' variance is Σ_t v_t'Σ_t v_t, with v_t the
            // values of term t's intercept (1) and constant slopes, so each
            // level gives one equation in the products v_t,a·v_t,b (a ≤ b), and
            // the covariance entries are not all identified when they
            // outnumber the distinct product vectors. One unstructured block
            // is one term; there the products are as distinct as v itself.
            // Under split terms only the products count: x = ±1 per level in
            // `(1 + x || g)` gives σ0² + σ1² on every level.
            let members: Vec<Vec<Option<usize>>> = if terms.is_empty() {
                vec![std::iter::once(None)
                    .chain(constant.iter().map(|&k| Some(k)))
                    .collect()]
            } else {
                (terms.iter())
                    .map(|t| {
                        (t.intercept.then_some(None).into_iter())
                            .chain(
                                (constant.iter())
                                    .filter(|&&k| t.slopes.contains(&slopes[k]))
                                    .map(|&k| Some(k)),
                            )
                            .collect()
                    })
                    .collect()
            };
            let at = |m: Option<usize>, r: usize| m.map_or(1.0, |k| value(slopes[k], r));
            let mut vectors: Vec<Vec<f64>> = (first_row.iter().filter(|&&r| r != usize::MAX))
                .map(|&r| {
                    (members.iter())
                        .flat_map(|m| {
                            (0..m.len()).flat_map(move |a| {
                                (a..m.len()).map(move |b| at(m[a], r) * at(m[b], r))
                            })
                        })
                        .collect()
                })
                .collect();
            // `validate_design` refuses a non-finite value, so the order is total.
            vectors.sort_by(|a, b| a.partial_cmp(b).expect("finite values"));
            vectors.dedup();
            let parameters: usize = members.iter().map(|m| tri(m.len())).sum();
            if vectors.len() < parameters {
                notes.push(Note::SlopeConstantWithinLevels {
                    grouping,
                    slopes: constant.iter().map(|&k| k as u32).collect(),
                });
            }
        }
        // A level of one Bernoulli trial gains nothing as the variance grows
        // (its marginal success probability is all that the variance moves),
        // so separation needs some level with more than one trial: two rows,
        // or one row of an aggregated response whose weight is its trials.
        // A response constant over all rows is `ConstantResponse`'s case, not
        // separation through this grouping.
        if binomial
            && y.iter().all(|&v| v == 0.0 || v == 1.0)
            && y.iter().any(|&v| v != y[0])
            && (0..n).all(|i| y[i] == y[first(i)])
            && trials.iter().any(|&t| t > 1.0)
        {
            notes.push(Note::SeparatedByGrouping { grouping });
        }
    }
    notes
}

/// Whether two id vectors put the rows into the same groups: a one-to-one
/// map between their ids, consistent over every row.
fn same_partition(a: &[u32], b: &[u32]) -> bool {
    let width = |ids: &[u32]| ids.iter().max().map_or(0, |&m| m as usize + 1);
    let (mut ab, mut ba) = (vec![u32::MAX; width(a)], vec![u32::MAX; width(b)]);
    a.iter().zip(b).all(|(&u, &v)| {
        let (u, v) = (u as usize, v as usize);
        if ab[u] == u32::MAX && ba[v] == u32::MAX {
            (ab[u], ba[v]) = (v as u32, u as u32);
        }
        ab[u] == v as u32 && ba[v] == u as u32
    })
}

/// Fills `dst[0..n, 0..p]` from row-major `x` (n·p, unweighted). Factored out
/// of `to_col_major` so `fit_on` can fill the `n_max`-sized `x_mat` allocated
/// once at `build_workspace`, instead of allocating a fresh `n×p` `Mat` every
/// call. `dst` must be at least `n×p`; rows/columns past `n`/`p` are left
/// untouched. One column at a time: a strided read, a contiguous write.
pub(super) fn fill_col_major(dst: &mut Mat<f64>, x: &[f64], n: usize, p: usize) {
    let x = &x[..n * p];
    for j in 0..p {
        let col = &mut dst.col_as_slice_mut(j)[..n];
        for (d, &v) in col.iter_mut().zip(x.iter().skip(j).step_by(p)) {
            *d = v;
        }
    }
}

/// [`fill_col_major`], then each standardized column in place:
/// `dst[(i, j)] = (x[i·p + j] − m)/s`, with `m` and `s` the statistics of
/// slope index `first + j` (`first` is 0 for `X`, `p` for the
/// random-effect-only columns). No extra copy of `x`; a column left as it is
/// is copied and not touched again.
pub(super) fn fill_col_major_std(
    dst: &mut Mat<f64>,
    x: &[f64],
    n: usize,
    p: usize,
    first: usize,
    cs: &ColumnScaling,
) {
    fill_col_major(dst, x, n, p);
    for j in 0..p {
        cs.standardize_column(first + j, &mut dst.col_as_slice_mut(j)[..n]);
    }
}

/// Row-major `x` → the column map and the column-major `x̃` (`n` rows) of a
/// route that builds its own workspace and fits the design as given
/// (`FitWorkspace::set_prestandardized`): only the random-slope columns of `g`
/// are scaled ([`ColumnScaling::compute`]). For the test baselines.
#[cfg(test)]
pub(super) fn as_given_design(
    x: &[f64],
    n: usize,
    p: usize,
    g: &crate::lmm::LmmGroupings,
    weights: Option<&[f64]>,
) -> (ColumnScaling, Mat<f64>) {
    let slopes = g
        .primary_slope_cols
        .iter()
        .chain(g.extra_slope_cols.iter().flatten());
    let mut cs = ColumnScaling::with_re_slopes(p, slopes.copied(), []);
    cs.compute(x, n, p, weights, false);
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p.max(1));
    fill_col_major_std(&mut x_mat, x, n, p, 0, &cs);
    (cs, x_mat)
}

/// Converts row-major `x` (n·p, unweighted) into a column-major faer matrix.
/// Thin wrapper over [`fill_col_major`] for the test-only paths that build a
/// throwaway `Mat` once per call.
#[cfg(test)]
pub(super) fn to_col_major(x: &[f64], n: usize, p: usize) -> Mat<f64> {
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p.max(1));
    fill_col_major(&mut x_mat, x, n, p);
    x_mat
}

/// Fills `se[target_indices[i]]` from `var_diag[i]` — the target-COMPACT
/// convention (OLS/GLM: `var_diag` is packed `[0..t]` in target order).
/// `se` must already be NaN-filled; entries fail the `>= 0` guard stay NaN
/// (numerically singular target).
pub(super) fn fill_se_compact(var_diag: &[f64], target_indices: &[u32], se: &mut [f64]) {
    for (i, &ti) in target_indices.iter().enumerate() {
        let vd = var_diag[i];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }
}

/// Fills `se[j]` from `var_diag[j]` — the predictor-INDEXED convention
/// (LMM/GLMM: `var_diag` is full-width `[0..p]`, read at each target index).
/// `se` must already be NaN-filled; entries failing the `>= 0` guard stay NaN
/// (numerically singular target).
pub(super) fn fill_se_by_predictor(var_diag: &[f64], target_indices: &[u32], se: &mut [f64]) {
    for &ti in target_indices {
        let vd = var_diag[ti as usize];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }
}

/// An all-NaN `p×p` [`Fit::vcov`] — the shape every path starts from, and the
/// whole of what a non-converged fit reports (mirroring how `se` NaN-fills).
pub(crate) fn nan_vcov(p: usize) -> Vec<f64> {
    vec![f64::NAN; p * p]
}

/// The residual df `nu` of OLS and of a Gamma or inverse Gaussian GLM with φ
/// estimated (`n` less the estimated columns: the retained ones on a
/// rank-deficient design, not those of a QR breakdown), over the `p` columns
/// of `l`: `coef_df` is ν on the target columns, NaN elsewhere, and the record holds the full `C`, [`vcov_from_chol`] over
/// every column whatever the targets. A fit that did not converge keeps the
/// residual record, with ν (NaN: the callers pass it so) and an all-NaN `C`
/// (its contrasts are NaN, not `NoDf`), and `coef_df` stays NaN, as `se`
/// does; a GLM that converged without a `C` keeps `n − p`. The non-convergence is
/// reported through `converged`, since no pass ran that could fail. On a fit
/// with no estimates (every entry of `beta` NaN) ν is NaN too
/// ([`SatterthwaiteState::residual`]).
///
/// [`SatterthwaiteState::residual`]: crate::lmm::satterthwaite::SatterthwaiteState::residual
#[allow(clippy::too_many_arguments)]
pub(crate) fn residual_df(
    l: faer::MatRef<'_, f64>,
    beta: &[f64],
    nu: f64,
    p: usize,
    scale: f64,
    map: &ColumnScaling,
    target_indices: &[u32],
    converged: bool,
) -> (Vec<f64>, crate::lmm::satterthwaite::SatterthwaiteState) {
    use crate::lmm::satterthwaite::SatterthwaiteState;
    let mut coef_df = vec![f64::NAN; p];
    if !converged {
        return (
            coef_df,
            SatterthwaiteState::residual(Mat::full(p, p, f64::NAN), nu, beta),
        );
    }
    let every: Vec<u32> = (0..p as u32).collect();
    let c = vcov_from_chol(l, p, &every, scale, map);
    for &t in target_indices {
        if let Some(d) = coef_df.get_mut(t as usize) {
            *d = nu;
        }
    }
    let c = Mat::from_fn(p, p, |i, j| c[i * p + j]);
    (coef_df, SatterthwaiteState::residual(c, nu, beta))
}

/// Full `p×p` [`Fit::vcov`] from `l`, the lower-triangular Cholesky factor of a
/// precision matrix (`X'X` for OLS, `X'WX` for GLM, `X'V⁻¹X` for LMM) on the
/// fitted design, scaled by `scale` and read through the column map `B`
/// (`β = B·β̃`): `vcov = scale · B (L L')⁻¹ Bᵀ` on the target block. With the
/// identity map this is `scale · (L L')⁻¹`, and below `eⱼ` reads `bⱼ`, row `j`
/// of `B`.
///
/// No new numerics — this is the SE forward solve, kept instead of discarded.
/// `(L L')⁻¹ = L⁻ᵀ L⁻¹`, so entry `(i, j)` is `scale · (uᵢ · uⱼ)` where `uⱼ = L⁻¹eⱼ`
/// is exactly the vector each `se` target already forward-solves for
/// (`var_diag_j = scale·‖uⱼ‖²` is this matrix's diagonal — see
/// `ols::fit_suff_stats_t_sq`). Solving each target's column once and forming
/// the Gram costs the `p` solves `se` already pays plus the dot products.
///
/// Only `targets` rows/columns are solved: outside that block there is no `u`
/// to dot and the result stays NaN, so `vcov` is finite exactly where `se` is
/// (the `targets`-subset carve-out). When `scale` is finite and `> 0`, a
/// diagonal entry that comes out 0 or not finite, and an off-diagonal entry
/// that is not finite, is NaN: its variance left f64's range. `l` must be
/// the converged factor — every caller gates on `converged` first, the same
/// staleness contract [`crate::ols::OlsFitView::factor`] documents.
pub(crate) fn vcov_from_chol(
    l: faer::MatRef<'_, f64>,
    p: usize,
    target_indices: &[u32],
    scale: f64,
    map: &ColumnScaling,
) -> Vec<f64> {
    let mut vcov = nan_vcov(p);
    // L⁻¹bⱼ for each target j: forward-solve L·u = bⱼ (row j of B, eⱼ when the
    // design is fitted as given), reading L[(i,k)] directly (already
    // lower-triangular, no transposed access).
    let mut cols: Vec<(usize, Vec<f64>)> = Vec::with_capacity(target_indices.len());
    for &tj in target_indices {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        let mut u = vec![0.0f64; p];
        for i in 0..p {
            let mut acc = map.rhs(tj, i);
            for k in 0..i {
                acc -= l[(i, k)] * u[k];
            }
            let l_ii = l[(i, i)];
            u[i] = if l_ii == 0.0 { f64::NAN } else { acc / l_ii };
        }
        cols.push((tj, u));
    }
    let scale_ok = scale > 0.0 && scale.is_finite();
    for (a, (i, ui)) in cols.iter().enumerate() {
        for (j, uj) in cols[a..].iter() {
            let mut v = scale * ui.iter().zip(uj).map(|(x, y)| x * y).sum::<f64>();
            // A variance that underflows or overflows is not representable:
            // NaN, not a false 0 or inf. An off-diagonal 0 stays, because
            // uncorrelated columns give a true 0. Mirrors the OLS map back in
            // `ols::fit_suff_stats_t_sq_mapped`.
            if scale_ok && (!v.is_finite() || (i == j && v == 0.0)) {
                v = f64::NAN;
            }
            vcov[*i * p + *j] = v;
            vcov[*j * p + *i] = v;
        }
    }
    vcov
}
