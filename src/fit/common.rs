//! Shared helpers used by 2+ estimator dispatch files (`ols.rs`/`lmm/mod.rs`/
//! `glm.rs`/`glmm/`): θ-width/varcorr bookkeeping, RE-count sizing from
//! `GroupIds`, rank-deficiency salvage, and the two `Fit::se` fill conventions
//! (target-compact vs predictor-indexed). None of this touches a kernel
//! directly — it is `fit`'s own marshalling logic, factored out of `mod.rs`
//! because every estimator's dispatch needs at least one of these.

use std::borrow::Cow;

use faer::Mat;

use crate::{
    Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing, StartValues,
};

use super::{Boundary, Diagnostics, Fit, FitOptions, Note};

/// Every diagnostic a fitting arm produces, in one carrier. Each of the five
/// routes fills it once (`OlsFitView::diagnostics` and its four siblings), the
/// loop tier reads it off [`super::core::FitView::diagnostics`], and the stable
/// path materializes it into `Fit` inside the `*_view_to_fit` mappers. One
/// carrier, so a new diagnostic costs no new accessor set.
///
/// **Plain `Copy` data, no `Vec`.** The warm loop calls this per draw and 0.1.3
/// spent a whole batch removing allocations from that path, so the carrier holds
/// nothing that allocates. The public reshape — `Boundary` for `boundary_hit`,
/// varcorr-aligned flags for `pinned_components`, a note carrying
/// `(pivot_col, pivot)` — is a translation of these fields and nothing more.
///
/// **What is deliberately NOT here.**
/// - `aliased`: no fitting arm decides it. Dropping a redundant column is the
///   pre-dispatch alias gate's job (`detect_aliased` in `fit_warm`, scattered
///   back by [`fit_rank_deficient`]), which sits ABOVE the view — every arm's
///   own mask is unconditionally all-false, so a field here would always be
///   empty and would suggest a per-route decision that does not exist.
/// - `singular`: it is `boundary_hit == 1` on every route, ORed at
///   materialization with [`Fit::has_negligible_component`] — which reads the
///   assembled `varcorr` and so cannot be decided at view level.
#[derive(Clone, Copy)]
pub struct FitDiagnostics {
    /// Whether the fit reached its convergence criterion.
    pub converged: bool,
    /// θ boundary: 0 interior, 1 a component pinned to the floor, 2 no optimum.
    /// **Placeholder on the sparse and NB (`Prebuilt`) routes** — they report no
    /// boundary state, so this is back-derived from their `Fit::singular` and
    /// cannot distinguish 2 from 0. **Meaningless on OLS/GLM** (no θ): always 0.
    pub boundary_hit: u8,
    /// Bitmask of θ components pinned to the boundary, `diagonal_theta` order.
    /// **Meaningless on OLS/GLM** (no θ): always 0. **Placeholder on the
    /// `Prebuilt` routes** — reported as 0, which is indistinguishable from
    /// "nothing was pinned". Those routes assemble a `Fit` themselves and the
    /// sparse ones fill `Diagnostics::pinned` on it directly, so the stable
    /// surface is complete; only this loop-tier carrier, which is read back off
    /// the assembled `Fit` and holds no `Vec`, cannot carry the mask.
    pub pinned_components: u64,
    /// Scale-invariant per-column pivot ratio of the route's own Gram
    /// ([`crate::ols::min_pivot_ratio`]), and the column attaining it. NaN on
    /// every route and every return that formed no factor to measure — the
    /// dense-GLMM, sparse and NB routes record none at all.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
    /// `pivot` fell below the recording route's own detection floor: the design
    /// is computable but its coefficients are barely identified. Each route
    /// compares against its own constant (`ols::PIVOT_MIN` for OLS and GLM,
    /// `lmm::PIVOT_MIN` for dense LMM) because the routes were calibrated
    /// separately. NaN pivots compare false, so a route that records none never
    /// flags. The sparse route is NOT a detector — it refuses below its own
    /// floor and reports that as `converged: false`, so it flags nothing here.
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
        boundary_score: vec![],
        kkt_grad_norm: f64::NAN,
        notes,
    }
}

/// Reshape a `diagonal_theta`-ordered pin bitmask into [`Diagnostics::pinned`]'s
/// varcorr-aligned flags. Single source for that placement, shared by the four
/// view mappers (through [`materialize_diagnostics`]) and by the two sparse
/// routes, which assemble a `Fit` directly.
///
/// `varcorr` is the ONLY thing that maps bit order onto `pinned[g][i]`: the
/// bitmask is keyed to `diagonal_theta` order, which walks the primary factor's
/// `q_p` vech diagonals and then each extra grouping's, in declaration order —
/// the same factors, in the same order, that `varcorr` carries one block each
/// for. `q_g` comes from the block's vech length via [`super::vech_q`], the
/// same inversion [`Fit::stddev_corr`] uses, so `pinned[g][i]` and
/// `stddev_corr(g).0[i]` cannot drift apart.
///
/// Mask 0 ⇒ no grid: on every success path this function is fed the fitting
/// route's real pin mask (via [`materialize_diagnostics`] for the four dense
/// view mappers, or directly by the two sparse routes, which overwrite
/// `Diagnostics::from_flags`'s empty placeholder with this call's result), so
/// mask 0 here means the fit genuinely pinned nothing — never a route
/// declining to say. The short-circuit itself is a memory choice, not a
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

/// Reshape a `diagonal_theta`-ordered per-component score into
/// [`Diagnostics::boundary_score`]'s varcorr-aligned layout. **Mirrors
/// [`pinned_flags`] — change together**: the two must place their `[g][i]` on
/// the same component or a reader pairing them silently reads the wrong score.
/// The owning explanation of why `varcorr` is the only thing that can place a
/// bit lives on `pinned_flags`.
///
/// An all-NaN input returns empty, matching `pinned_flags`'s mask-0
/// short-circuit: nothing measured allocates nothing.
pub(crate) fn pinned_scores(scores: &[f64], varcorr: &[Vec<f64>]) -> Vec<Vec<f64>> {
    if scores.iter().all(|s| s.is_nan()) {
        return vec![];
    }
    let mut k = 0usize;
    varcorr
        .iter()
        .map(|vech| {
            (0..super::vech_q(vech.len()))
                .map(|_| {
                    let s = scores.get(k).copied().unwrap_or(f64::NAN);
                    k += 1;
                    s
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
pub(super) fn theta_width(re: Option<&ReStructure>) -> usize {
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

/// vech(column-major lower-tri) of `D̂ = scale·Λ̂Λ̂'` for one `q×q` grouping.
/// `theta_block` is that grouping's column-major lower-tri `vech(Λ)` prefix
/// (as `primary_lambda` reads it). `D[r][c] = Σ_{k≤min(r,c)} Λ[r][k]·Λ[c][k]`.
///
/// `row_scales[r]` is row `r`'s internal RE design-column scale
/// (`LmmGroupings::block_row_scale`). θ̂ arrives in the solver's internal units,
/// where row `r` of Λ carries a factor `s_r`; dividing it out here is the whole
/// back-map, and it is what makes the reported covariance
/// `D[i][j] = D̃[i][j]/(sᵢ·s_j)` as the scaling identity requires. All-`1.0`
/// scales divide exactly, so an unscaled design is untouched.
pub(super) fn varcorr_block(
    theta_block: &[f64],
    q: usize,
    scale: f64,
    row_scales: &[f64],
) -> Vec<f64> {
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta_block, q, &mut lam); // Λ̃ lower-tri, row-major
    back_map_lambda(&mut lam, q, row_scales);
    let mut vech = Vec::with_capacity(q * (q + 1) / 2);
    for c in 0..q {
        for r in c..q {
            let mut d = 0.0;
            for k in 0..=c {
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
/// The primary-then-extras vech walk mirrors the `vech_start` layout assigned
/// in `LmmGroupings::from_cluster_spec_ext` (`src/lmm/mod.rs`) — change together.
/// `pub(crate)` so the sparse-Z path (`sparse::fit_mle_sparse`) recovers varcorr
/// from θ̂ through the same path-independent assembly as the NoZ `fit_mle`.
pub(crate) fn assemble_varcorr(
    theta: &[f64],
    groupings: &crate::lmm::LmmGroupings,
    scale: f64,
) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(1 + groupings.extra_q.len());
    let mut cursor = 0usize;
    for (b, &q) in std::iter::once(&groupings.primary_q)
        .chain(groupings.extra_q.iter())
        .enumerate()
    {
        let vech = q * (q + 1) / 2;
        // θ̂ is in the solver's internal units; the row scales back-map it.
        let row_scales = block_row_scales(groupings, b, q);
        out.push(varcorr_block(
            &theta[cursor..cursor + vech],
            q,
            scale,
            &row_scales,
        ));
        cursor += vech;
    }
    out
}
/// [`Fit::loglik`] for the Gaussian LMM paths: the REML criterion
/// `−½(deviance + (n−p)(1 + ln 2π))` — the stripped constant restored (see
/// `Fit::deviance`'s LMM contract), then onto the `logLik()` scale. Call AFTER
/// any weighted `−Σlog wᵢ` deviance correction (the correction is part of
/// lme4's REMLcrit, so it belongs inside the criterion). NaN in ⇒ NaN out.
pub(crate) fn lmm_loglik(deviance: f64, n: usize, p: usize) -> f64 {
    // The NaN gate also protects the `n − p` residual df from underflowing on
    // the degenerate n ≤ p short-circuit (which always reports a NaN deviance).
    if !deviance.is_finite() || n <= p {
        return f64::NAN;
    }
    -0.5 * (deviance + (n - p) as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln()))
}

/// [`Fit::loglik`] for the GLMM paths (dense and sparse, Laplace and AGQ):
/// restore the data-only saturated constant the marginal deviance drops.
/// Binomial/Poisson/NB: `−½·deviance + saturated_loglik` (the deviance's data
/// term is the weighted `dev_resid` sum). Gamma: `−½·deviance` with NO
/// correction — lme4's `logLik(glmer fit)` is literally `−devfun/2`, and the
/// `+2` its `Gamma()$aic` data term carries stays inside (so glmer's Gamma
/// logLik sits 1 below `Σwᵢ·log f`; R's `logLik.glm` subtracts that 2 back
/// out, glmer does not — pinned against the sim_gamma validation loglik). NaN in
/// ⇒ NaN out (the non-converged contract).
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
        Family::Gamma { .. } => -0.5 * deviance,
        _ => -0.5 * deviance + crate::family::saturated_loglik(family, nb_theta, y, prior_w),
    }
}

/// [`Fit::df`] for a converged fit: retained fixed effects + RE θ parameters +
/// 1 if the family estimates a dispersion/scale. `dispersion_fixed` is
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
/// (`glmm::build_z` layout: primary block level-major `lvl·q_p + c`, then each
/// extra's scalar indicator columns at its absolute `extra_offsets[e]` — this
/// path carries intercept-only extras exclusively, see `apply_lambda`).
/// `b = Λ̂û` per block: the primary level's `q_p`-vector through the lower-tri
/// `Λ_p`, each extra level through its scalar θ. Output is `Fit::ranef`'s
/// public layout (per grouping, level-major) — for this path the primary block
/// is already level-major, so it maps through 1:1.
pub(crate) fn assemble_ranef_dense(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    u: &[f64],
) -> Vec<f64> {
    let q = g.primary_q;
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta, q, &mut lam);
    // Same back-map, and for the same reason, as `assemble_ranef_sparse` — the
    // internal `b̃ = Λ̃û` carries the design column's scale on row r. Only the
    // primary block needs it: this path's extras are intercept-only (asserted
    // below), so their scale is exactly 1.
    back_map_lambda(&mut lam, q, &block_row_scales(g, 0, q));
    let mut out = Vec::with_capacity(g.k_total);
    for lvl in 0..g.n_primary {
        let base = lvl * q;
        for r in 0..q {
            let mut b = 0.0;
            for c in 0..=r {
                b += lam[r * q + c] * u[base + c];
            }
            out.push(b);
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
/// the layout.
/// Divide block `b`'s row-major lower-tri Λ̃ (as `primary_lambda` writes it) by
/// its rows' internal RE design-column scales, turning the solver's internal
/// factor into the user's: `Λ[r][c] = Λ̃[r][c]/s_r`. The one back-map every
/// θ-derived block quantity funnels through. All-`1.0` scales divide exactly.
fn back_map_lambda(lam: &mut [f64], q: usize, row_scales: &[f64]) {
    for r in 1..q {
        let s_r = row_scales[r];
        if s_r == 1.0 {
            continue;
        }
        for c in 0..=r {
            lam[r * q + c] /= s_r;
        }
    }
}

/// Block `b`'s per-row internal RE design-column scales (`b == 0` primary,
/// `b == e+1` extra grouping `e`), length `q` — the argument every back-map takes.
fn block_row_scales(g: &crate::lmm::LmmGroupings, b: usize, q: usize) -> Vec<f64> {
    (0..q).map(|r| g.block_row_scale(b, r)).collect()
}

/// The internal RE design-column scales laid out like [`Fit::varcorr`] — one
/// row-aligned block per grouping, primary first. Multiplying a reported
/// standard deviation by its entry puts it back on the internal scale, which is
/// where the components of one fit are commensurate with each other:
/// user-scale standard deviations for an intercept and for a slope are in
/// different units (response, vs response per covariate unit), so their ratio
/// moves with the covariate's units.
pub(crate) fn re_scale_grid(g: &crate::lmm::LmmGroupings) -> Vec<Vec<f64>> {
    std::iter::once(g.primary_q)
        .chain(g.extra_q.iter().copied())
        .enumerate()
        .map(|(b, q)| block_row_scales(g, b, q))
        .collect()
}

pub(crate) fn assemble_ranef_sparse(
    theta: &[f64],
    g: &crate::lmm::LmmGroupings,
    u: &[f64],
) -> Vec<f64> {
    let q = g.primary_q;
    let s = g.n_primary;
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta, q, &mut lam);
    // `b̃ = Λ̃û` comes out in the solver's internal RE units, where row r carries
    // the design column's scale s_r; the user's conditional mode is `b = b̃/s_r`.
    // Back-mapping Λ here rather than the output does it once per row instead of
    // once per level — and it must happen HERE, not later: `lmm_fitted` weights
    // `ranef` by the RAW covariate, so it needs a user-scale `ranef`.
    back_map_lambda(&mut lam, q, &block_row_scales(g, 0, q));
    let mut out = Vec::with_capacity(g.k_total);
    for f in 0..s {
        for r in 0..q {
            let mut b = 0.0;
            for c in 0..=r {
                b += lam[r * q + c] * u[c * s + f];
            }
            out.push(b);
        }
    }
    let counts = ranef_level_counts(g);
    for (e, &off) in g.extra_offsets.iter().enumerate() {
        let q_g = g.extra_q[e];
        let mut lam_g = vec![0.0f64; q_g * q_g];
        crate::lmm::primary_lambda(&theta[extra_vech_start(g, e)..], q_g, &mut lam_g);
        back_map_lambda(&mut lam_g, q_g, &block_row_scales(g, e + 1, q_g));
        for l in 0..counts[e + 1] {
            let base = off + l * q_g;
            for r in 0..q_g {
                let mut b = 0.0;
                for c in 0..=r {
                    b += lam_g[r * q_g + c] * u[base + c];
                }
                out.push(b);
            }
        }
    }
    out
}

/// [`Fit::fitted`] for the Gaussian LMM paths: `μ̂ = o + Xβ̂ + Zb̂` per row,
/// scattered straight through the ids. **Z is never materialized** — each row
/// reads its own level's block out of `ranef` and weights it by that row's own
/// covariates, which is what keeps the no-Z property the whole LMM design rests
/// on true.
///
/// `ranef` is [`Fit::ranef`]'s public layout (per grouping, level-major), so this
/// runs after [`assemble_ranef_sparse`]. The identity link makes μ̂ = η̂, and the
/// offset is added back here because the LMM applies it as an exact `y − o` shift
/// BEFORE accumulation and never sees it again — the same caveat `fit_ols`
/// documents, and the same fix.
#[allow(clippy::too_many_arguments)] // marshals (x, n, p, beta, ranef, groupings, ids…)
pub(crate) fn lmm_fitted(
    x: &[f64],
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
    let q_of = |e: usize| {
        if e == 0 {
            g.primary_q
        } else {
            g.extra_q[e - 1]
        }
    };
    for (e, &levels) in counts.iter().enumerate() {
        block_start.push(acc);
        acc += levels * q_of(e);
    }
    (0..n)
        .map(|i| {
            let row = &x[i * p..(i + 1) * p];
            let mut eta = offset.map_or(0.0, |o| o[i]);
            for (j, &b) in beta.iter().enumerate() {
                eta += row[j] * b;
            }
            let mut add_block = |e: usize, level: usize, slope_cols: &[usize]| {
                let q = q_of(e);
                let base = block_start[e] + level * q;
                eta += ranef[base]; // intercept component, z = 1
                for (d, &sc) in slope_cols.iter().enumerate() {
                    eta += ranef[base + 1 + d] * row[sc];
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
/// [`spec_sized_from_ids`] is the only producer. Its size rule exchanges the
/// primary with exactly one extra, and only when every grouping is
/// intercept-only, so each grouping then owns exactly one θ coordinate, one
/// `varcorr` block, one `pinned` row and one `ranef` block. Undoing the reorder
/// is therefore the same exchange applied a second time — a transposition is its
/// own inverse — which is why this records the one swapped index rather than a
/// full index map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Perm {
    /// Extra grouping that traded places with the primary. `None` is the
    /// identity: declaration order reached the kernel untouched.
    swapped_extra: Option<usize>,
}

impl Perm {
    /// Declaration order reached the kernel untouched.
    pub const IDENTITY: Perm = Perm {
        swapped_extra: None,
    };

    /// Whether the kernel's slot order IS the caller's declaration order.
    pub fn is_identity(&self) -> bool {
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
    pub fn swap_slots<T>(&self, v: &mut [T]) {
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
fn size_rule_perm(re: &ReStructure, primary_levels: usize, extras: &[Grouping]) -> Perm {
    let eligible = re.slopes.is_empty()
        && extras
            .iter()
            .all(|g| g.slopes.is_empty() && matches!(g.relation, GroupingRelation::Crossed { .. }));
    if !eligible {
        return Perm::IDENTITY;
    }
    let mut best = primary_levels;
    let mut swapped_extra = None;
    for (k, g) in extras.iter().enumerate() {
        let GroupingRelation::Crossed { n_clusters } = g.relation else {
            // `eligible` already established every extra is Crossed.
            continue;
        };
        if n_clusters as usize > best {
            best = n_clusters as usize;
            swapped_extra = Some(k);
        }
    }
    Perm { swapped_extra }
}

/// A `ModelSpec` copy whose RE **counts** are derived from the supplied `GroupIds`
/// (`n_primary = max(primary)+1`; per crossed grouping `n_clusters =
/// max(extra[g])+1`; per nested grouping `n_per_parent = ⌈children/n_primary⌉`,
/// `children = max(extra[g])+1`). Topology tags (`Crossed`/`NestedWithin`),
/// family, and slope columns are preserved. This is the data path's mechanism for
/// "derive every level count from the ids": the sizing-corrected copy
/// feeds the existing workspace builders unchanged, so no builder/kernel signature
/// changes. `re: None` is returned as-is (fixed-only carries no counts).
///
/// It is also where [`size_rule_perm`]'s grouping reorder is applied, which is
/// why the ids come back with the spec rather than being left to the caller:
/// a sized spec and the ids that match it can only be obtained together, so no
/// path can size a spec and then hand the kernel ids in a different order. The
/// swap moves two `Vec` handles and two spec fields — nothing per-observation —
/// but the ids must be cloned to be swapped, so the clone happens only on the
/// models the rule actually reorders (`Cow::Borrowed` otherwise).
pub(super) fn spec_sized_from_ids<'a>(
    model: &ModelSpec,
    ids: &'a GroupIds,
) -> (ModelSpec, Cow<'a, GroupIds>, Perm) {
    let Some(re) = model.re.as_ref() else {
        return (model.clone(), Cow::Borrowed(ids), Perm::IDENTITY);
    };
    let level_count = |v: &[u32]| v.iter().copied().max().map(|m| m as usize + 1).unwrap_or(1);
    let n_primary = level_count(&ids.primary);
    let extra_groupings: Vec<Grouping> = re
        .extra_groupings
        .iter()
        .enumerate()
        .map(|(g, gr)| {
            let relation = match gr.relation {
                GroupingRelation::Crossed { .. } => {
                    let children = level_count(&ids.extra[g]);
                    GroupingRelation::Crossed {
                        n_clusters: children as u32,
                    }
                }
                GroupingRelation::NestedWithin { .. } => {
                    // Per-parent distinct-child-id sets, counted directly from the
                    // (primary, extra) row pairs — layout-agnostic (works whether
                    // or not the frontend's ids are contiguous per-parent blocks),
                    // unlike deriving from a single global `max(extra)+1` divided
                    // by `n_primary`, which under-sizes any parent with an
                    // above-average child count (the true fix for unbalanced
                    // nesting: `n_per_parent` must be the TRUE max, not a
                    // `children.div_ceil(n_primary)` global average).
                    let mut per_parent: Vec<std::collections::HashSet<u32>> =
                        vec![Default::default(); n_primary];
                    for (&p, &c) in ids.primary.iter().zip(&ids.extra[g]) {
                        per_parent[p as usize].insert(c);
                    }
                    let n_per_parent = per_parent.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
                    GroupingRelation::NestedWithin {
                        n_per_parent: n_per_parent as u32,
                    }
                }
            };
            Grouping {
                relation,
                slopes: gr.slopes.clone(),
            }
        })
        .collect();
    let perm = size_rule_perm(re, n_primary, &extra_groupings);
    let mut sized = ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: re.slopes.clone(),
            extra_groupings,
        }),
    };
    let Some(k) = perm.swapped_extra else {
        return (sized, Cow::Borrowed(ids), perm);
    };
    let sized_re = sized.re.as_mut().expect("just built with re: Some");
    let GroupingRelation::Crossed {
        n_clusters: extra_levels,
    } = sized_re.extra_groupings[k].relation
    else {
        unreachable!("the size rule only fires when every extra is Crossed");
    };
    sized_re.sizing = Sizing::FixedClusters {
        n_clusters: extra_levels,
    };
    sized_re.extra_groupings[k].relation = GroupingRelation::Crossed {
        n_clusters: n_primary as u32,
    };
    // Both are empty whenever the rule fires (it requires intercept-only
    // groupings); swapped anyway so the spec half of the exchange is complete
    // rather than complete-by-precondition.
    std::mem::swap(
        &mut sized_re.slopes,
        &mut sized_re.extra_groupings[k].slopes,
    );
    let mut swapped_ids = ids.clone();
    std::mem::swap(&mut swapped_ids.primary, &mut swapped_ids.extra[k]);
    (sized, Cow::Owned(swapped_ids), perm)
}
/// Lower-tri Gram `G = XᵀX` (from row-major X) → `aliased_columns` mask. `p` is
/// tiny, so the extra `O(N·p²)` reduction for the drop decision is negligible
/// (the reduced fit recomputes its own suff stats).
pub(super) fn detect_aliased(x: &[f64], n: usize, p: usize) -> Vec<bool> {
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
/// dropped).
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
fn remap_spec_slopes(model: &ModelSpec, to_reduced: &[usize]) -> Option<ModelSpec> {
    let Some(re) = model.re.as_ref() else {
        return Some(model.clone());
    };
    // `collect::<Option<Vec<_>>>` short-circuits on the first dropped column, so
    // one `None` anywhere in any slope list fails the whole remap.
    let remap = |cols: &[u32]| -> Option<Vec<u32>> {
        cols.iter()
            .map(|&c| {
                let r = to_reduced[c as usize];
                (r != usize::MAX).then_some(r as u32)
            })
            .collect()
    };
    let extra_groupings = re
        .extra_groupings
        .iter()
        .map(|g| {
            Some(Grouping {
                relation: g.relation.clone(),
                slopes: remap(&g.slopes)?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            sizing: re.sizing.clone(),
            slopes: remap(&re.slopes)?,
            extra_groupings,
        }),
    })
}

/// The "unfittable model" return: an aliased fixed column is also used as an RE
/// slope, so no reduced spec exists (see [`remap_spec_slopes`]). Carries the
/// crate's standard numerical-failure convention — NaN β/se/vcov/dispersion,
/// `converged: false`, no varcorr, `df: 0` — the same shape `fit_mle_sparse`
/// returns on its own failures, so a caller that already handles
/// `converged == false` needs no new branch. `n_eval: 0` is honest: no optimizer
/// ran.
///
/// `aliased` is reported as-is rather than blanked. `converged == false` with
/// every β NaN already says "unfittable"; the mask says which columns caused it,
/// which is the only diagnostic the caller can act on, and it is the same field
/// `fit_rank_deficient` fills on the successful path.
fn unfittable_random_slope_fit(p: usize, model: &ModelSpec, aliased: &[bool]) -> Fit {
    Fit {
        beta: vec![f64::NAN; p],
        se: vec![f64::NAN; p],
        vcov: nan_vcov(p),
        tau2: vec![f64::NAN; theta_width(model.re.as_ref())],
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
        // Only the Gaussian mixed (LMM) routes report a REML criterion; every
        // GLMM route reports `reml: false`. Reached with `re: Some` only, so the
        // family alone decides.
        reml: matches!(model.family, Family::Gaussian),
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
    }
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
/// **Recursion.** This re-enters `fit_warm`, so termination is a real
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
/// not a termination hazard: `fit_warm` skips the gate at `p == 0`.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_rank_deficient(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    start: Option<&StartValues>,
    opts: &FitOptions,
    aliased: &[bool],
) -> Fit {
    let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let pk = kept.len();
    debug_assert!(
        pk < p,
        "fit_rank_deficient: `aliased` must drop at least one column, or the \
         recursive fit_warm re-enters on identical input"
    );
    let mut to_reduced = vec![usize::MAX; p];
    for (r, &orig) in kept.iter().enumerate() {
        to_reduced[orig] = r;
    }
    // Remap before building the reduced design: a dropped column used as an RE
    // slope makes the model unfittable, and there is no point paying for the
    // O(n·pk) copy to discover that.
    let Some(model_r) = remap_spec_slopes(model, &to_reduced) else {
        return unfittable_random_slope_fit(p, model, aliased);
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
    });
    // Targets: drop aliased targets, reindex survivors into the reduced design.
    let targets_r: Vec<u32> = opts
        .target_indices
        .iter()
        .filter(|&&t| !aliased[t as usize])
        .map(|&t| to_reduced[t as usize] as u32)
        .collect();
    // Per-row weights are unaffected by dropping fixed-effect COLUMNS, so the
    // rest of `opts` carries over unchanged via `..opts.clone()`.
    let opts_r = FitOptions {
        target_indices: targets_r,
        ..opts.clone()
    };
    let fr = super::fit_warm(&xr, y, n, pk, &model_r, ids, start_r.as_ref(), &opts_r);
    // Scatter reduced β/se to full width; aliased slots stay NaN.
    //
    // `aliased_out` starts as THIS level's mask and takes the reduced fit's mask
    // on top, in the same loop that scatters β. The OR keeps the field's
    // contract — NaN in β/se ⇔ the column was dropped — true no matter how deep
    // the chain runs: if the recursive `fit_warm` salvages again, `fr.aliased[r]`
    // is true for a column this level KEPT and `fr.beta[r]` is the NaN that goes
    // with it, and reporting only this level's mask would scatter that NaN into
    // a slot flagged `aliased == false`. The alias gate cannot fire twice today
    // (see the recursion note above), so this is insurance rather than a live
    // path, but it is the cheap kind and the contract is worth pinning.
    let mut aliased_out = aliased.to_vec();
    let mut beta = vec![f64::NAN; p];
    let mut se = vec![f64::NAN; p];
    for (r, &orig) in kept.iter().enumerate() {
        beta[orig] = fr.beta[r];
        se[orig] = fr.se[r];
        aliased_out[orig] |= fr.diagnostics.aliased[r];
    }
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
            | Note::UnusedGroupingLevels { .. }
            | Note::ReDesignScaleSpread { .. }
            | Note::HessianSeFallback => {}
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
        fitted: fr.fitted,
        ranef: fr.ranef,
        ranef_levels: fr.ranef_levels,
    }
}
/// Engine-invariant shape check for the data-shaped ids (mirrors `fit_grouped`'s
/// former `cluster_ids.len()==n` panic): the primary id vector is length `n`, the
/// extra vectors align 1:1 with the declared extra groupings, and each is length
/// `n`. A malformed `GroupIds` is an engine invariant violation, so this panics
/// (consistent with the `fit` panic convention), not a `converged:false` return.
pub(super) fn assert_group_ids(re: &ReStructure, ids: &GroupIds, n: usize) {
    assert_eq!(
        ids.primary.len(),
        n,
        "GroupIds.primary must have n elements"
    );
    assert_eq!(
        ids.extra.len(),
        re.extra_groupings.len(),
        "GroupIds.extra must align 1:1 with re.extra_groupings (declaration order)"
    );
    for (g, e) in ids.extra.iter().enumerate() {
        assert_eq!(e.len(), n, "GroupIds.extra[{g}] must have n elements");
    }
}

/// Engine invariant checks for the standalone `fit` path: the kernel's stack
/// scratch is sized off `MAX_PRIMARY_Q`/`MAX_EXTRA_Q`, so a
/// `q` over the cap would overflow it, and every slope column must index into the
/// `p`-wide design. A malformed spec is an engine invariant violation (see the
/// `fit` panic convention), so this asserts rather than returning a `Fit`.
/// Fixed-only models (`re: None`) carry no RE caps to check.
pub(super) fn assert_model_shape(model: &ModelSpec, p: usize, nagq: u8) {
    // nAGQ: odd, 1..=MAX_NAGQ; >1 only on a single grouping factor (no extras),
    // q_p ≤ 3, binomial/Poisson GLMM — the shapes whose marginal likelihood is a
    // product of independent per-cluster q-D integrals. Checked before the RE
    // early-return so even fixed-only specs can't smuggle a bad nagq through.
    // Sourced from `FitOptions` (M3.5), not the spec. Mirrors the Python layer's
    // warn-and-strip boundary (`glmm.fit`) — change together.
    assert!(
        (1..=crate::consts::MAX_NAGQ).contains(&nagq) && nagq % 2 == 1,
        "nagq={} must be odd in 1..={}",
        nagq,
        crate::consts::MAX_NAGQ
    );
    if nagq > 1 {
        let re = model
            .re
            .as_ref()
            .expect("nagq>1 requires a mixed model (re: Some)");
        let single_factor = re.extra_groupings.is_empty();
        let agq_family = matches!(
            model.family,
            Family::Binomial { .. } | Family::Poisson { .. }
        );
        assert!(
            single_factor && agq_family,
            "nagq>1 legal only on a single grouping factor, binomial/Poisson GLMM"
        );
        // q_p = 1 + #primary slopes. The q_p ≥ 4 refusal is a TEMPORARY cost /
        // oracle-coverage boundary (the k^q product grid and the dimension-generic
        // loop have no q limit), not a code limit — lifting it later is deleting
        // this one check. Locked decision: cap AGQ to q_p≤3 until oracle coverage
        // extends past it, rather than let uncapped q_p through untested.
        let q_p = 1 + re.slopes.len();
        assert!(
            q_p <= 3,
            "nagq>1 with q_p={q_p} random effects per group exceeds the temporary \
             q_p≤3 cap (a cost/oracle-coverage boundary, not a code limit)"
        );
    }
    let Some(re) = model.re.as_ref() else {
        return;
    };
    // Inverse-Gaussian mixed models are deliberately not built: the GLMM
    // objective needs the profiled `inverse.gaussian()$aic` term, which the
    // kernel does not have. Faulting here rather than in `build_workspace`'s
    // dispatch means no workspace is allocated for a model that cannot be fit,
    // and `fit_warm` runs this same gate at entry. `orchestrate`'s
    // `catch_unwind` turns it into a clean `Err` for the FFI ports.
    assert!(
        !matches!(model.family, Family::InverseGaussian { .. }),
        "inverse-Gaussian mixed models are not implemented (fixed-effects GLM only)"
    );
    // The RE-envelope caps (extra-grouping count, q_p, q_g) are
    // `classify_design`'s routing boundary — over-envelope designs route to
    // the sparse path instead of aborting here. Only the column-bounds
    // validity asserts remain below (they hold regardless of solver).
    for &col in &re.slopes {
        assert!(
            (col as usize) < p,
            "primary slope column {col} out of range (p={p})"
        );
    }
    for g in &re.extra_groupings {
        for &col in &g.slopes {
            assert!(
                (col as usize) < p,
                "extra-grouping slope column {col} out of range (p={p})"
            );
        }
    }
    // The kernel holds ONE nested slot (`LmmGroupings.nested`) and
    // `from_cluster_spec_ext` would silently let a later `NestedWithin` extra
    // overwrite an earlier one (last wins). The formula frontend detects at most
    // one; this guards the explicit `parent:child` route (defense in depth).
    let n_nested = re
        .extra_groupings
        .iter()
        .filter(|g| matches!(g.relation, GroupingRelation::NestedWithin { .. }))
        .count();
    assert!(
        n_nested <= 1,
        "at most one NestedWithin extra grouping is supported (got {n_nested})"
    );
}
/// Test-only re-export of [`assert_model_shape`] so `spec.rs` can exercise the
/// `nagq` shape-check without routing through a full `fit` call.
#[cfg(test)]
pub(crate) fn assert_model_shape_pub(model: &ModelSpec, p: usize, nagq: u8) {
    assert_model_shape(model, p, nagq);
}

/// Crate-internal re-export of [`spec_sized_from_ids`]: the sparse-Z path's
/// equivalence test (`sparse::tests`) sizes a spec from ids exactly as the stable
/// `fit_warm` entry does, and the `loop_advanced` surface hands it to loop-tier
/// consumers (MCPower) so they normalize RE level counts the same validated way
/// before [`build_workspace`] rather than reimplementing the count derivation.
/// Feed all three returned values onward: the ids belong to the sized spec, and
/// the [`Perm`] is what maps the kernel's grouping-indexed results (θ̂ above all)
/// back to the order the caller declared.
#[cfg(any(test, feature = "loop_advanced"))]
pub fn spec_sized_from_ids_pub<'a>(
    model: &ModelSpec,
    ids: &'a GroupIds,
) -> (ModelSpec, Cow<'a, GroupIds>, Perm) {
    spec_sized_from_ids(model, ids)
}
/// Fills `dst[0..n, 0..p]` from row-major `x` (n·p, unweighted). Factored out
/// of `to_col_major` so the `fit_on` hot-path arms (`Ols`/`Glm`/`LmmDense`) can
/// fill an `n_max`-sized buffer allocated once at `build_workspace`, instead of
/// allocating a fresh `n×p` `Mat` every call — the same buffer-reuse pattern
/// `FitKind::GlmmDense`'s `x_mat` already uses. `dst` must be at least `n×p`;
/// rows/columns past `n`/`p` are left untouched.
pub(super) fn fill_col_major(dst: &mut Mat<f64>, x: &[f64], n: usize, p: usize) {
    for i in 0..n {
        for j in 0..p {
            dst[(i, j)] = x[i * p + j];
        }
    }
}

/// Converts row-major `x` (n·p, unweighted) into a column-major faer matrix.
/// Shared by every unweighted site; `fit_ols`'s √wᵢ-scaled loop is a distinct
/// variant and is not routed through this helper. Thin wrapper over
/// [`fill_col_major`] for the call sites that build a throwaway `Mat` once per
/// call regardless (test-only paths, `fit_glm`, `fit_glm_nb`, `fit_glmm_build`)
/// — not on the `fit_on` hot path, so the allocation here is not worth removing.
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

/// Full `p×p` [`Fit::vcov`] from `l`, the lower-triangular Cholesky factor of a
/// precision matrix (`X'X` for OLS, `X'WX` for GLM, `X'V⁻¹X` for LMM), scaled by
/// `scale`: `vcov = scale · (L L')⁻¹`.
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
) -> Vec<Vec<f64>> {
    let mut vcov = nan_vcov(p);
    // Column j of L⁻¹, for each target j: forward-solve L·u = e_j, reading
    // L[(i,k)] directly (already lower-triangular, no transposed access).
    let mut cols: Vec<(usize, Vec<f64>)> = Vec::with_capacity(target_indices.len());
    for &tj in target_indices {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        let mut u = vec![0.0f64; p];
        for i in 0..p {
            let mut acc = if i == tj { 1.0 } else { 0.0 };
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
