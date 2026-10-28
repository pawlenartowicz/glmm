//! Shared helpers used by 2+ estimator dispatch files (`ols.rs`/`lmm/mod.rs`/
//! `glm.rs`/`glmm/`): θ-width/varcorr bookkeeping, RE-count sizing from
//! `GroupIds`, rank-deficiency salvage, and the two `Fit::se` fill conventions
//! (target-compact vs predictor-indexed). None of this touches a kernel
//! directly — it is `fit`'s own marshalling logic, factored out of `mod.rs`
//! because every estimator's dispatch needs at least one of these.

use std::borrow::Cow;

use faer::Mat;

use crate::inference::Objective;
use crate::lmm::Criterion;
use crate::spec::set_term_slopes;
use crate::standardize::ColumnScaling;

use crate::{
    ColumnId, Data, Design, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure,
    ReTerm, StartValues,
};

use super::{Boundary, CondvarCross, Diagnostics, Fit, FitError, FitOptions, Note, SeFor};

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
    /// **Placeholder on the NB (`Prebuilt`) routes** — they report no boundary
    /// state, so this is back-derived from their `Fit::singular` and cannot
    /// distinguish 2 from 0. **Meaningless on OLS/GLM** (no θ): always 0.
    pub boundary_hit: u8,
    /// Bitmask of θ components pinned to the boundary, `diagonal_theta` order.
    /// **Meaningless on OLS/GLM** (no θ): always 0. **Placeholder on the
    /// `Prebuilt` routes** — reported as 0, which is indistinguishable from
    /// "nothing was pinned". Those routes assemble a `Fit` themselves and fill
    /// `Diagnostics::pinned` on it directly, so the stable surface is complete;
    /// only this loop-tier carrier, which is read back off the assembled `Fit`
    /// and holds no `Vec`, cannot carry the mask.
    pub pinned_components: u64,
    /// Scale-invariant per-column pivot ratio of the route's own Gram
    /// (`min_pivot_ratio`), and the column attaining it. NaN on
    /// every route and every return that formed no factor to measure. The NB
    /// (`Prebuilt`) routes read it back off their assembled `Fit`'s
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
    /// `ill_conditioned`, `pivot` 0 and `pivot_col` the column that broke
    /// down. `false` on OLS and LMM routes.
    pub se_unavailable: bool,
    /// GLMM-only: the estimated rounding error of the reported deviance is
    /// past what resolves θ̂ — see [`crate::Note::PrecisionLimited`]. `false`
    /// on every non-GLMM route.
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
/// affordable — but both stay empty on a clean fit, which keeps a `Fit`
/// materialized from a clean draw cheap.
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
/// Mask 0 ⇒ no grid: on every success path this function is fed the fitting
/// route's real pin mask, carried in from the view mapper through
/// [`materialize_diagnostics`], so mask 0 here means the fit genuinely pinned
/// nothing — never a route declining to say. The short-circuit itself is a memory choice, not a
/// meaning: a warm loop over draws that never pin allocates no grid of
/// `false` per draw for saying so.
pub(crate) fn pinned_flags(mask: u64, varcorr: &[Vec<f64>]) -> Vec<Vec<bool>> {
    if mask == 0 {
        return vec![];
    }
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
pub(super) fn varcorr_block(
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
/// terms hold at zero is not estimated. `dispersion_fixed` is
/// `FitOptions::dispersion.is_some()` (Gamma's hold-φ-fixed directive; every
/// other family ignores it).
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
/// primary with exactly one extra, and only when every grouping is
/// intercept-only, so each grouping then owns exactly one θ coordinate, one
/// `varcorr` block, one `pinned` row and one `ranef` block. Undoing the reorder
/// is therefore the same exchange applied a second time — a transposition is its
/// own inverse — which is why this records the one swapped index rather than a
/// full index map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Perm {
    /// Extra grouping that traded places with the primary. `None` is the
    /// identity: declaration order reached the kernel untouched.
    swapped_extra: Option<usize>,
}

impl Perm {
    /// Declaration order reached the kernel untouched.
    pub(crate) const IDENTITY: Perm = Perm {
        swapped_extra: None,
    };

    /// Whether the kernel's slot order IS the caller's declaration order.
    pub(crate) fn is_identity(&self) -> bool {
        self.swapped_extra.is_none()
    }

    /// Map a vector holding one entry per grouping between slot order and
    /// declaration order — `v[0]` the primary's, `v[g+1]` the g-th extra's. The
    /// same call converts either way (see the type doc). Covers every
    /// per-grouping and per-θ-coordinate result, the two coinciding because the
    /// rule only fires on intercept-only groupings.
    ///
    /// An EMPTY vector is a documented state for several of the fields this
    /// runs over — nothing pinned, a non-converged fit carrying no `varcorr`,
    /// an LMM's absent `stddev_se` — and stays empty. Any other length must
    /// cover every grouping, so a short one indexes out of bounds here instead
    /// of quietly reordering the wrong pair.
    pub(crate) fn swap_slots<T>(&self, v: &mut [T]) {
        if let Some(k) = self.swapped_extra {
            if v.is_empty() {
                return;
            }
            v.swap(0, k + 1);
        }
    }

    /// Same map for a concatenation of variable-width per-grouping blocks —
    /// [`Fit::ranef`], the one grouping-indexed output whose blocks are wider
    /// than one entry. `widths` must be in the SAME order as `v` currently is;
    /// map it with [`Perm::swap_slots`] afterwards, not before. Empty is a real
    /// state here too (a non-converged fit recovers no conditional modes).
    fn swap_blocks(&self, v: &mut Vec<f64>, widths: &[usize]) {
        let Some(k) = self.swapped_extra else {
            return;
        };
        if widths.is_empty() {
            return;
        }
        let mut starts = Vec::with_capacity(widths.len());
        let mut acc = 0usize;
        for &w in widths {
            starts.push(acc);
            acc += w;
        }
        debug_assert_eq!(acc, v.len(), "block widths must cover the whole vector");
        let mut order: Vec<usize> = (0..widths.len()).collect();
        order.swap(0, k + 1);
        let mut out = Vec::with_capacity(v.len());
        for g in order {
            out.extend_from_slice(&v[starts[g]..starts[g] + widths[g]]);
        }
        *v = out;
    }
}

/// Report a fit's grouping-indexed results in the caller's declaration order.
/// The kernels emit them in slot order, which [`spec_sized_from_ids`]'s size
/// rule may have reordered; this is the single site that undoes it for the
/// stable API, so no `Fit` consumer sees slot order. `beta`, `se`, `vcov`,
/// `fitted`, `deviance` and `loglik` are grouping-order-independent and are not
/// touched.
pub(super) fn unpermute_fit(perm: Perm, fit: &mut Fit) {
    if perm.is_identity() {
        return;
    }
    // `ranef` before `ranef_levels`: the block widths must still describe the
    // layout `ranef` is currently in.
    perm.swap_blocks(&mut fit.ranef, &fit.ranef_levels);
    // The rule moves intercept-only groupings alone, so a level holds one
    // `condvar` value and one `p`-wide `condvar_x` row.
    if !fit.condvar.is_empty() {
        let p = fit.condvar_x.len() / fit.condvar.len();
        let rows: Vec<usize> = fit.ranef_levels.iter().map(|&l| l * p).collect();
        perm.swap_blocks(&mut fit.condvar, &fit.ranef_levels);
        perm.swap_blocks(&mut fit.condvar_x, &rows);
    }
    // The kernel's primary, the family grouping, is the swapped extra.
    if let (Some(c), Some(k)) = (fit.condvar_cross.as_mut(), perm.swapped_extra) {
        c.family_grouping = k + 1;
    }
    perm.swap_slots(&mut fit.ranef_levels);
    perm.swap_slots(&mut fit.tau2);
    perm.swap_slots(&mut fit.stddev_se);
    perm.swap_slots(&mut fit.varcorr);
    perm.swap_slots(&mut fit.diagnostics.pinned);
}

/// The size rule: **when every grouping is intercept-only and `Crossed`, the
/// grouping with the most levels becomes the primary; otherwise nothing moves.**
///
/// Why it pays. The primary is eliminated as block-diagonal families at cost
/// linear in its level count, while every extra lands in one Schur tail whose
/// width is the total crossed level count and whose factorization is cubic in
/// that width. Which grouping the user happened to write first therefore sets
/// the tail width. MixedModels.jl sorts its `ReMat`s by decreasing level count
/// for exactly this reason; Bates, Alday & Kokandakar (2025), arXiv:2505.11674,
/// measure the effect on InstEval with that sort defeated — ~4.5M vs 775K
/// nonzeros in the Cholesky factor.
///
/// Why it is this narrow. Plain "most levels first" is wrong as soon as a
/// `NestedWithin` grouping is present: hoisting a child out of its parent turns
/// a nesting that rides inside the primary's block-diagonal elimination into two
/// crossed factors sharing one tail, which can be several times WIDER. And it is
/// wrong as soon as any grouping carries a random slope, because a wide primary
/// routes the dense kernel whose joint optimizer is known to land above lme4's
/// optimum on the sparse-Gamma shape. With every grouping scalar and crossed
/// neither hazard exists — no nesting relation can be invalidated and no slope
/// block can change hands — and "most levels first" and "smallest crossed tail"
/// become the same argmax.
///
/// Ties keep the declared primary (and, among extras, the earliest), so a model
/// already in the chosen order is handed to the kernel exactly as declared.
fn size_rule_perm(re: &ReStructure, primary_levels: usize, extra_counts: &[usize]) -> Perm {
    let eligible = re.slopes.is_empty()
        && re
            .extra_groupings
            .iter()
            .all(|g| g.slopes.is_empty() && g.relation == GroupingRelation::Crossed);
    if !eligible {
        return Perm::IDENTITY;
    }
    let mut best = primary_levels;
    let mut swapped_extra = None;
    for (k, &levels) in extra_counts.iter().enumerate() {
        if levels > best {
            best = levels;
            swapped_extra = Some(k);
        }
    }
    Perm { swapped_extra }
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
/// otherwise).
pub(crate) fn spec_sized_from_ids<'a>(
    model: &ModelSpec,
    ids: &'a GroupIds,
) -> (SizedSpec, Cow<'a, GroupIds>, Perm) {
    let Some(re) = model.re.as_ref() else {
        let sized = SizedSpec {
            spec: model.clone(),
            n_primary: 0,
            extra_counts: Vec::new(),
        };
        return (sized, Cow::Borrowed(ids), Perm::IDENTITY);
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
    let perm = size_rule_perm(re, n_primary, &extra_counts);
    let mut sized = SizedSpec {
        spec: ModelSpec {
            family: model.family,
            re: Some(re.clone()),
        },
        n_primary,
        extra_counts,
    };
    let Some(k) = perm.swapped_extra else {
        return (sized, Cow::Borrowed(ids), perm);
    };
    std::mem::swap(&mut sized.n_primary, &mut sized.extra_counts[k]);
    let sized_re = sized.spec.re.as_mut().expect("just built with re: Some");
    // Both are empty whenever the rule fires (it requires intercept-only
    // groupings); swapped anyway so the spec half of the exchange is complete
    // rather than complete-by-precondition.
    std::mem::swap(
        &mut sized_re.slopes,
        &mut sized_re.extra_groupings[k].slopes,
    );
    std::mem::swap(&mut sized_re.terms, &mut sized_re.extra_groupings[k].terms);
    let mut swapped_ids = ids.clone();
    std::mem::swap(&mut swapped_ids.primary, &mut swapped_ids.extra[k]);
    (sized, Cow::Owned(swapped_ids), perm)
}

/// Lower-tri Gram `G = XᵀX` (from row-major X) → `aliased_columns` mask. `p` is
/// tiny, so the extra `O(N·p²)` reduction for the drop decision is negligible
/// (the reduced fit recomputes its own suff stats).
pub(crate) fn detect_aliased(x: &[f64], n: usize, p: usize) -> Vec<bool> {
    let mut gram = Mat::<f64>::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            let xa = x[i * p + a];
            for b in 0..=a {
                gram[(a, b)] += xa * x[i * p + b];
            }
        }
    }
    crate::ols::aliased_columns(gram.as_ref(), p, crate::ols::ALIAS_EPS)
}

/// Remap a spec's RE slope x-column indices through the kept-columns map
/// (`to_reduced[orig]` = reduced index, or `usize::MAX` if that column was
/// dropped), and each grouping's terms with them, so the reduced spec passes
/// [`check_model_shape`]. An index past `to_reduced` names a
/// random-effect-only column, which follows the `p_reduced` kept columns in
/// the same order (the reduced fit gets the same `Design::re_columns`).
///
/// `None` means a dropped column is ALSO used as an RE slope — a rank-deficient
/// random slope, which the crate does not fit: the reduced design has no column
/// for that slope to point at, so there is no valid reduced spec to build.
/// Dropping the random slope alongside the fixed column would be a different
/// model and needs its own reporting and its own oracle; it is deliberately not
/// done here. Signalling `None` lets the caller return a non-converged `Fit`
/// instead of panicking — a library panic would take the caller's process
/// down, giving an R/Python user an abort instead of an inspectable fit, and
/// costing a loop caller its whole run over one degenerate draw.
fn remap_spec_slopes(
    model: &ModelSpec,
    to_reduced: &[usize],
    p_reduced: usize,
) -> Option<ModelSpec> {
    let Some(re) = model.re.as_ref() else {
        return Some(model.clone());
    };
    // `collect::<Option<Vec<_>>>` short-circuits on the first dropped column, so
    // one `None` anywhere in any slope list fails the whole remap.
    let remap = |cols: &[u32]| -> Option<Vec<u32>> {
        cols.iter()
            .map(|&c| {
                let r = match to_reduced.get(c as usize) {
                    Some(&r) => r,
                    None => c as usize - to_reduced.len() + p_reduced,
                };
                (r != usize::MAX).then_some(r as u32)
            })
            .collect()
    };
    let extra_groupings = re
        .extra_groupings
        .iter()
        .map(|g| {
            let slopes = remap(&g.slopes)?;
            let mut terms = g.terms.clone();
            set_term_slopes(&mut terms, &slopes);
            Some(Grouping {
                relation: g.relation,
                slopes,
                terms,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    let slopes = remap(&re.slopes)?;
    let mut terms = re.terms.clone();
    set_term_slopes(&mut terms, &slopes);
    Some(ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            slopes,
            terms,
            extra_groupings,
        }),
    })
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

/// The "unfittable model" return: an aliased fixed column is also used as an RE
/// slope, so no reduced spec exists (see [`remap_spec_slopes`]). Carries the
/// crate's standard numerical-failure convention — NaN β/se/vcov/dispersion,
/// `converged: false`, no varcorr, `df: 0` — the same shape the solver routes
/// return on their own failures, so a caller that already handles
/// `converged == false` needs no new branch. `n_eval: 0` is honest: no optimizer
/// ran.
///
/// `aliased` is reported as-is rather than blanked. `converged == false` with
/// every β NaN already says "unfittable"; the mask says which columns caused it,
/// which is the only diagnostic the caller can act on, and it is the same field
/// `fit_rank_deficient` fills on the successful path.
fn unfittable_random_slope_fit(
    n: usize,
    p: usize,
    model: &ModelSpec,
    aliased: &[bool],
    opts: &FitOptions,
) -> Fit {
    let criterion = resolve_criterion(model, opts);
    // `theta_width` less column 0 of each block without a random intercept:
    // the fitted routes' `LmmGroupings::drop_hidden` width, from the spec
    // alone, as no groupings exist on this return.
    let block = |q: usize, terms: &[crate::ReTerm]| {
        q * (q + 1) / 2 - q * usize::from(crate::lmm::block_hidden(terms))
    };
    let tau2_width = model.re.as_ref().map_or(0, |re| {
        block(1 + re.slopes.len(), &re.terms)
            + re.extra_groupings
                .iter()
                .map(|g| block(1 + g.slopes.len(), &g.terms))
                .sum::<usize>()
    });
    let mut fit = Fit {
        beta: vec![f64::NAN; p],
        se: vec![f64::NAN; p],
        vcov: nan_vcov(p),
        tau2: vec![f64::NAN; tau2_width],
        dispersion: f64::NAN,
        diagnostics: Diagnostics {
            aliased: aliased.to_vec(),
            ..Diagnostics::from_flags(false, false, p)
        },
        varcorr: vec![],
        stddev_se: vec![],
        n_eval: 0,
        #[cfg(feature = "counters")]
        counters: crate::counters::EvalCounters::new(),
        deviance: f64::NAN,
        loglik: f64::NAN,
        df: 0,
        // The criterion `opts.reml` resolves to, as a fitted Gaussian mixed
        // model reports it; every GLMM route reports `reml: false`.
        reml: criterion == Some(Criterion::Reml),
        nobs: n,
        // The route's objective: the aliased column is a random slope, so
        // the model is mixed (a criterion on Gaussian, the quadrature on
        // every other family).
        objective: match criterion {
            Some(Criterion::Reml) => Objective::Reml,
            Some(Criterion::Ml) => Objective::MaximumLikelihood,
            None => glmm_objective(opts.nagq),
        },
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
        condvar: vec![],
        condvar_x: vec![],
        condvar_cross: None,
        coef_df: vec![f64::NAN; p],
        satterthwaite: None,
        design_summary: None,
    };
    // A Gaussian LMM without an endpoint: the Satterthwaite record
    // `fit_checked`'s step would have stored (it never runs here).
    if matches!(model.family, Family::Gaussian) {
        if opts.satterthwaite {
            super::core::record_failed_pass(&mut fit);
        } else {
            fit.satterthwaite = Some(crate::lmm::satterthwaite::SatterthwaiteState::switched_off());
        }
    }
    fit
}

/// Fit the reduced (aliased-columns-dropped) model and scatter β/se back to full
/// width: retained slots take the reduced fit, aliased slots are NaN,
/// `converged` follows the reduced fit, `tau2`/`varcorr`/`dispersion` pass through
/// (the RE structure is unchanged; only fixed-column indices are remapped).
///
/// `Fit::aliased` is the UNION of this level's `aliased` and the reduced fit's own
/// mask, not just this level's — see the OR in the scatter loop. That keeps the
/// field's contract (`NaN` in β/se ⇔ the column was dropped) true across a nested
/// salvage, which the recursion below makes reachable.
///
/// Returns an all-NaN, non-converged `Fit` without refitting when a dropped
/// column is also an RE slope (see [`remap_spec_slopes`]) — that model is
/// unfittable, not merely rank-deficient.
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
    let Design { x, n, p, .. } = data.design;
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
    // Remap before building the reduced design: a dropped column used as an RE
    // slope makes the model unfittable, and there is no point paying for the
    // O(n·pk) copy to discover that.
    let Some(model_r) = remap_spec_slopes(model, &to_reduced, pk) else {
        return unfittable_random_slope_fit(n, p, model, aliased, opts);
    };
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
    let reduced = Data {
        design: Design {
            x: &xr,
            p: pk,
            ..data.design
        },
        ..*data
    };
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
            Note::IllConditioned { columns, .. } => {
                for c in columns.iter_mut() {
                    *c = kept[*c as usize] as u32;
                }
            }
            // Carry no design-column indices — nothing to translate.
            Note::PirlsExhausted { .. }
            | Note::HessianSeFallback
            | Note::NbShapeUnsettled { .. }
            | Note::NonIntegerResponse { .. }
            | Note::SingleLevelGroupingDropped { .. }
            | Note::FewRowsForRandomEffects { .. }
            | Note::ExactProfileFallback
            | Note::GradientSearchFallback
            | Note::SeUnavailable
            | Note::PrecisionLimited
            | Note::Separated
            | Note::VarianceAtLimit
            | Note::SatterthwaiteCurvature { .. }
            | Note::NewGroupingLevels { .. } => {}
        }
    }
    // Same scatter in two dimensions: an aliased column has no coefficient, so
    // it has no covariance with anything — its whole row and column stay NaN,
    // exactly as its `se` slot does.
    let mut vcov = nan_vcov(p);
    for (ri, &oi) in kept.iter().enumerate() {
        for (rj, &oj) in kept.iter().enumerate() {
            vcov[oi][oj] = fr.vcov[ri][rj];
        }
    }
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
        counters: crate::counters::EvalCounters::new(),
        deviance: fr.deviance,
        // Scalars/per-row/RE-shaped fields pass through: loglik and df come
        // from the REDUCED fit (aliased columns carry no parameter — lme4's
        // NA-coefficient df), fitted is per-row (the reduced model's means are
        // the model's means), ranef is RE-shaped (unchanged by fixed-column
        // drops).
        loglik: fr.loglik,
        df: fr.df,
        reml: fr.reml,
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
    }
}
/// The grouping ids of a mixed model checked against it: the primary id
/// vector is length `n`, the extra vectors align 1:1 with the declared extra
/// groupings, and each is length `n`. [`validate`] returns the error;
/// `build_workspace` panics with its message.
pub(super) fn check_group_ids(re: &ReStructure, ids: &GroupIds, n: usize) -> Result<(), FitError> {
    if ids.primary.len() != n {
        return Err(FitError::Ids(
            "GroupIds.primary must have n elements".to_string(),
        ));
    }
    if ids.extra.len() != re.extra_groupings.len() {
        return Err(FitError::Ids(
            "GroupIds.extra must align 1:1 with re.extra_groupings (declaration order)".to_string(),
        ));
    }
    if let Some(g) = ids.extra.iter().position(|e| e.len() != n) {
        return Err(FitError::Ids(format!(
            "GroupIds.extra[{g}] must have n elements"
        )));
    }
    Ok(())
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
            return Err(FitError::Ids(format!(
                "GroupIds.extra[{g}] is nested: a child id must be parent * {w} + its \
                 position within the parent, got {} under parent {} (row {i})",
                child[i], ids.primary[i]
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
            return Err(FitError::Ids(format!(
                "GroupIds.{name}: id {} is not below the row count {n} (row {i}); number \
                 the levels 0, 1, 2, ... with no gaps",
                v[i]
            )));
        }
    }
    Ok(())
}

/// The first row whose value fails `ok`, as the ` (row k)` suffix of a
/// [`FitError`] message.
fn bad_row(v: &[f64], ok: impl Fn(f64) -> bool) -> Option<String> {
    v.iter()
        .position(|&a| !ok(a))
        .map(|i| format!(" (row {i})"))
}

/// The input check [`super::fit`] and [`super::fit_with_start`] share: every
/// caller mistake in `data`, `spec`, `start` and `opts` as a [`FitError`],
/// before anything is fitted. What passes here reaches the kernels' own
/// invariants intact, so a panic past this point is a crate bug.
pub(crate) fn validate(
    data: &Data<'_>,
    spec: &ModelSpec,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Result<(), FitError> {
    validate_design(&data.design, spec, opts.nagq, true)?;
    if let (Some(re), Some(ids)) = (spec.re.as_ref(), data.design.ids) {
        // First: `check_nested_ids` sizes a buffer by the largest primary id.
        check_id_bound(re, ids, data.design.n)?;
        check_nested_ids(re, ids)?;
    }
    let Design { n, p, .. } = data.design;
    let y = data.y;
    if y.len() != n {
        return Err(FitError::Shape("y must have n elements".to_string()));
    }
    if let Some(row) = bad_row(y, f64::is_finite) {
        return Err(FitError::ResponseOutOfSupport(format!(
            "y must be finite{row}"
        )));
    }
    // An EMPTY field means "cold-start this component" — a caller that knows only
    // one of the two (an R `start = list(theta = …)`, lme4's shape) cannot compute
    // the other: the cold β seed is a no-RE GLM fit and the cold θ seed is the
    // blind THETA0 shape, both of which live inside the kernels. So the widths
    // accepted are {0, exact}, and each consumption site falls back on empty.
    if let Some(s) = start {
        let start_err = |m: &str| Err(FitError::Start(m.to_string()));
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
    }
    // Response domain: refuse a response outside the family's support before
    // fitting, rather than let the kernel converge on numbers a valid model
    // could never have produced (a Poisson y < 0 reaching the kernel gives
    // `converged = true` with logLik −∞). Binomial's `y` is always the
    // success PROPORTION (`crate::family`'s aggregated convention: row `i` is
    // `wᵢ` trials with `wᵢ·yᵢ` successes, unit weights ⇒ Bernoulli), so
    // `0..=1` covers both the raw 0/1 case and "successes above trials".
    let refused =
        |ok: fn(f64) -> bool, what: &str| bad_row(y, ok).map(|row| format!("{what}{row}"));
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
    if let Some(d) = opts.dispersion {
        if !(d.is_finite() && d > 0.0) {
            return Err(FitError::OptionValue(
                "FitOptions.dispersion must be finite and > 0".to_string(),
            ));
        }
    }
    if let SeFor::Only(cols) = &opts.se_for {
        if let Some(&c) = cols.iter().find(|&&c| c as usize >= p) {
            return Err(FitError::OptionValue(format!(
                "FitOptions.se_for column {c} out of range (p={p})"
            )));
        }
    }
    Ok(())
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
    if x.len() != n * p {
        return Err(FitError::Shape(
            "x must have n*p elements in row-major layout".to_string(),
        ));
    }
    if let Some(k) = x.iter().position(|v| !v.is_finite()) {
        return Err(FitError::Shape(format!(
            "x must be finite (row {}, column {})",
            k / p,
            k % p
        )));
    }
    let z = re.unwrap_or(&[]);
    if z.len() != n * s {
        return Err(FitError::Shape(
            "re_columns must have n*s elements in row-major layout".to_string(),
        ));
    }
    if let Some(k) = z.iter().position(|v| !v.is_finite()) {
        return Err(FitError::Shape(format!(
            "re_columns must be finite (row {}, column {})",
            k / s,
            k % s
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
                "a model with random effects needs GroupIds (Data::ids)".to_string(),
            ))
        }
        (None, Some(ids)) if !(ids.primary.is_empty() && ids.extra.is_empty()) => {
            return Err(FitError::Ids(
                "GroupIds given for a model without random effects".to_string(),
            ))
        }
        (Some(_), None) | (None, _) => {}
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(FitError::Weights(
                "Data weights must have n elements".to_string(),
            ));
        }
        if let Some(row) = bad_row(w, |v| v.is_finite() && v > 0.0) {
            return Err(FitError::Weights(format!(
                "Data weights must be finite and > 0{row}"
            )));
        }
    }
    if let Some(o) = offset {
        if o.len() != n {
            return Err(FitError::Offset(
                "Data offset must have n elements".to_string(),
            ));
        }
        if let Some(row) = bad_row(o, f64::is_finite) {
            return Err(FitError::Offset(format!("Data offset must be finite{row}")));
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
        return Err(FitError::OptionValue(nagq_range()));
    }
    if nagq > crate::consts::MAX_NAGQ {
        return Err(FitError::Envelope(nagq_range()));
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
            )));
        }
        if !re.extra_groupings.is_empty() {
            return Err(FitError::Unsupported(
                "nagq>1 needs a single grouping factor (no crossed or nested extra groupings)"
                    .to_string(),
            ));
        }
        // q_p = 1 + #primary slopes. The q_p ≥ 4 refusal is a TEMPORARY cost /
        // oracle-coverage boundary (the k^q product grid and the dimension-generic
        // loop have no q limit), not a code limit — lifting it later is deleting
        // this one check. Locked decision: cap AGQ to q_p≤3 until oracle coverage
        // extends past it, rather than let uncapped q_p through untested.
        let q_p = 1 + re.slopes.len();
        if q_p > 3 {
            return Err(FitError::Unsupported(format!(
                "nagq>1 with q_p={q_p} random effects per group exceeds the temporary \
                 q_p≤3 cap (a cost/oracle-coverage boundary, not a code limit)"
            )));
        }
    }
    // Inverse-Gaussian mixed models are deliberately not built: the GLMM
    // objective needs the profiled `inverse.gaussian()$aic` term, which the
    // kernel does not have. Refused here rather than in `build_workspace`'s
    // dispatch, so no workspace is allocated for a model that cannot be fit.
    if matches!(model.family, Family::InverseGaussian { .. }) {
        return Err(FitError::Unsupported(
            "inverse-Gaussian mixed models are not implemented (fixed-effects GLM only)"
                .to_string(),
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
        return Err(FitError::Unsupported(format!(
            "at most one NestedWithin extra grouping is supported (got {n_nested})"
        )));
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
                    return Err(FitError::Shape(format!(
                        "{what} slope column {c} out of range (p={p}, s={s})"
                    )));
                }
                if c >= p {
                    read[c - p] = true;
                }
            }
        }
    }
    if let Some(j) = read.iter().position(|&r| !r) {
        return Err(FitError::Shape(format!(
            "re_columns column {j} (slope index {}) is read by no random slope",
            p + j
        )));
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
/// column in two terms, and `slopes` the terms' columns concatenated, which
/// is the order `LmmGroupings` assigns block columns to terms in. Empty
/// `terms` passes. `grouping` names the block in the error.
fn check_terms(slopes: &[ColumnId], terms: &[ReTerm], grouping: &str) -> Result<(), FitError> {
    let refuse = |why: String| Err(FitError::Shape(format!("{grouping}: {why}")));
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
    Ok(())
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
pub(crate) fn nan_vcov(p: usize) -> Vec<Vec<f64>> {
    vec![vec![f64::NAN; p]; p]
}

/// The residual df of OLS and of a Gamma or inverse Gaussian GLM with φ
/// estimated (`ν = n − p`, `p` this fit's columns, the retained ones on a
/// rank-deficient design): `coef_df` is ν on the target columns, NaN
/// elsewhere, and the record holds the full `C`, [`vcov_from_chol`] over
/// every column whatever the targets. A fit that did not converge keeps the
/// residual record, with ν and an all-NaN `C` (its contrasts are NaN, not
/// `NoDf`), and `coef_df` stays NaN, as `se` does; the non-convergence is
/// reported through `converged`, since no pass ran that could fail.
pub(crate) fn residual_df(
    l: faer::MatRef<'_, f64>,
    n: usize,
    p: usize,
    scale: f64,
    map: &ColumnScaling,
    target_indices: &[u32],
    converged: bool,
) -> (Vec<f64>, crate::lmm::satterthwaite::SatterthwaiteState) {
    use crate::lmm::satterthwaite::SatterthwaiteState;
    let mut coef_df = vec![f64::NAN; p];
    let nu = n as f64 - p as f64;
    if !converged {
        return (
            coef_df,
            SatterthwaiteState::residual(Mat::full(p, p, f64::NAN), nu),
        );
    }
    let every: Vec<u32> = (0..p as u32).collect();
    let c = vcov_from_chol(l, p, &every, scale, map);
    for &t in target_indices {
        if let Some(d) = coef_df.get_mut(t as usize) {
            *d = nu;
        }
    }
    let c = Mat::from_fn(p, p, |i, j| c[i][j]);
    (coef_df, SatterthwaiteState::residual(c, nu))
}

/// Full `p×p` [`Fit::vcov`] from `l`, the lower-triangular Cholesky factor of a
/// precision matrix (`X'X` for OLS, `X'WX` for GLM, `X'V⁻¹X` for LMM) on the
/// fitted design, scaled by `scale` and read through the column map `B`
/// (`β = B·β̃`): `vcov = scale · B (L L')⁻¹ Bᵀ` on the target block. With the
/// identity map this is `scale · (L L')⁻¹`, and below `eⱼ` reads `bⱼ`, row `j`
/// of `B`.
///
/// No new numerics — this is the SE forward solve, kept instead of discarded.
/// `(L L')⁻¹ = L⁻ᵀ L⁻¹`, so `vcov[i][j] = scale · (uᵢ · uⱼ)` where `uⱼ = L⁻¹eⱼ`
/// is exactly the vector each `se` target already forward-solves for
/// (`var_diag_j = scale·‖uⱼ‖²` is this matrix's diagonal — see
/// `ols::fit_suff_stats_t_sq`). Solving each target's column once and forming
/// the Gram costs the `p` solves `se` already pays plus the dot products.
///
/// Only `targets` rows/columns are solved: outside that block there is no `u`
/// to dot and the result stays NaN, so `vcov` is finite exactly where `se` is
/// (the `targets`-subset carve-out). `l` must be
/// the converged factor — every caller gates on `converged` first, the same
/// staleness contract [`crate::ols::OlsFitView::factor`] documents.
pub(crate) fn vcov_from_chol(
    l: faer::MatRef<'_, f64>,
    p: usize,
    target_indices: &[u32],
    scale: f64,
    map: &ColumnScaling,
) -> Vec<Vec<f64>> {
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
    for (a, (i, ui)) in cols.iter().enumerate() {
        for (j, uj) in cols[a..].iter() {
            let v = scale * ui.iter().zip(uj).map(|(x, y)| x * y).sum::<f64>();
            vcov[*i][*j] = v;
            vcov[*j][*i] = v;
        }
    }
    vcov
}
