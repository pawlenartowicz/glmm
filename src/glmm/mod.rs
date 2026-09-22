//! GLMM kernel: PIRLS + Laplace/AGQ, glmer-faithful nAGQ=1. Covers
//! Binomial/Poisson/Gamma/negative-binomial with random effects; on NB the
//! outer BOBYQA searches `ln θ_NB` as one more trailing coordinate alongside
//! `[θ_RE | β]`, so β, θ_RE and θ_NB come out of one fit — see `fit_glmm`'s
//! `nb_term`/`n_nb` handling. Optimized per
//! `OuterSearch` (`GlmmWorkspace.outer_search`, fixed per shape at construction),
//! mirroring lme4's structure (Bates, Mächler, Bolker, Walker, *JSS* 67(1), 2015,
//! §3) on the two routes that use it: `Joint` runs one BOBYQA over `[θ | β]`
//! directly, β held fixed inside each PIRLS solve. `PqlThenJoint` first runs
//! BOBYQA over θ alone (`n_theta` dims), profiling β out at each candidate θ by
//! solving for the conditional modes ũ **and** β jointly per PIRLS iteration (a
//! small dense p×p β-Schur border on the RE solve) — the PQL-optimal β for that
//! θ — then warm-starts a `Joint` polish from there; the θ-only pass is only an
//! accelerant, so the polish alone gates convergence and the reported (θ̂, β̂)
//! is the Laplace optimum, not the PQL one. `ExactProfile` (the blocked
//! in-scope shapes) instead profiles β out EXACTLY at each candidate θ — the
//! Laplace optimum for that θ, not merely the PQL one — so the θ-only BOBYQA
//! pass IS the search: no joint polish follows, and its status alone gates
//! convergence. The objective is the Laplace deviance d(y,ũ) + ‖ũ‖² + log|L|²;
//! PIRLS gains step-halving to keep the higher-dimensional joint (u, β) step
//! stable.
//!
//! This module holds three PIRLS backends, one per `A`-layout
//! ([`workspace::GlmmLayout`], picked per RE design shape):
//! `pirls_solve_blocked` (a single grouping, `A` block-diagonal),
//! `pirls_solve_blocked_extras` (intercept-only crossed/nested extras, core
//! blocks plus a crossed Schur), and `pirls_solve_packed` (everything else:
//! fixed-width packed `M` rows and a dense `k×k` `A`) — see
//! `se::blocked_schur_fill`/`se::structured_schur_fill`/`se::packed_schur_fill`
//! for their Schur complement fills. None of them materializes `Z` or `Λ_θ`
//! densely. Designs over `classify_design`'s NoZ envelope take the packed-row
//! backend through this same entry point.
//! All scratch for the three backends lives in `GlmmWorkspace`, allocated once
//! per (spec, max_n) shape — the warm path is zero-alloc (Bobyqa::new once).
//!
//! No σ² scale: binomial dispersion is fixed at 1, so D̂ = Λ̂Λ̂′ directly.
//!
//! BOBYQA is Powell, M.J.D. (2009), *The BOBYQA algorithm for bound constrained
//! optimization without derivatives*, Cambridge report DAMTP 2009/NA06.

use crate::WaldSe;
use bobyqa::Status;
use faer::{Mat, MatRef};

use crate::lmm::{PIN_THETA, THETA0, THETA_TRUTH_FLOOR};

use se::{blocked_schur_fill, packed_schur_fill, structured_schur_fill};
pub(crate) use workspace::GlmmLayout;
use workspace::{fill_z_f64, FitData};

/// PIRLS inner-loop caps. `glm.rs`'s plain IRLS caps at `MAX_IRLS_ITERS = 50`
/// (PIRLS *is* that IRLS plus the +I ridge), but PIRLS itself needs a higher
/// cap: the joint `(u, β)` step of the β-profile outer route can oscillate
/// around its fixed point with a decay ratio near 0.84 on some Bernoulli
/// random-slope fits, and 50 iterations is not always enough to reach the exit
/// band. A solve that exits on the band is bit-identical at any cap, so only
/// fits that exhaust the cap can change.
pub const PIRLS_MAX_ITERS: usize = 200;
/// Backtracking cap for PIRLS step-halving, mirroring lme4 `pwrssUpdate`'s
/// 10-halving discipline: when a full Fisher step raises the penalized deviance
/// above the last accepted value, the u-step is halved and re-evaluated up to
/// this many times before the solve is declared failed. Exhausting it surfaces as
/// the module's `(NaN, NaN, NaN, false)` failure — the same terminal state a raw
/// overshoot reaches, but reached deliberately. Shared by all three PIRLS
/// variants (blocked / structured / packed).
///
/// 16, above lme4's 10: the packed-row layout's FD-Hessian central deviance eval
/// cold-seeds `û = 0` on every evaluation (see `se::SPARSE_FD_STEP_REL`), and on
/// a large-θ̂, many-crossed-grouping, large-count design that cold seed needs more
/// halvings to walk back to the mode than a warm-started fit does. Measured floor
/// on the packed large-θ̂ rung is 11 halvings; 16 was chosen for margin above that
/// floor, not tuned to the exact minimum. The blocked and structured solve steps
/// don't hit this regime and converge well inside the cap.
pub const PIRLS_MAX_HALVINGS: usize = 16;
/// The shapes whose stage-1 Profile solve is the EXACT Laplace β-profile and whose
/// outer search is therefore θ-only (`OuterSearch::ExactProfile`): Laplace, a data
/// term that is the plain deviance, and a PIRLS variant carrying the exact border —
/// the blocked path (`pirls_solve_blocked`, either link class), plus the structured
/// crossed/nested path (`pirls_solve_blocked_extras`), which carries an observed-
/// information twin of its factor (core blocks, coupling, Schur) on both canonical
/// and non-canonical links. Gamma's `gamma_aic` objective has a different β score and
/// keeps the joint search; AGQ keeps theirs too. So does the packed-row layout
/// (`GlmmLayout::Packed`, `workspace.rs`), whose PIRLS carries no observed twin and
/// therefore no exact border — `pirls_solve_packed` refuses a `ProfileExact` step
/// outright. Read by the workspace constructor and by the driver's
/// `debug_assert!` — the tests choose the route through `outer_search`, never
/// through this.
pub(crate) fn exact_profile_shape(
    family: crate::spec::Family,
    nagq: u8,
    layout: GlmmLayout,
) -> bool {
    nagq == 1
        && !matches!(family, crate::spec::Family::Gamma { .. })
        && layout != GlmmLayout::Packed
}
/// Adaptive PIRLS exit on |Δ penalized-deviance|, relative to the objective
/// scale (lme4's pwrss discipline): converged when
/// `|Δpen| < PIRLS_TOL_REL · (1 + |pen|)`. The penalized deviance is O(n), so
/// this relative gate is an ABSOLUTE tolerance of `PIRLS_TOL_REL · n` in the
/// deviance scale — fine for the small/medium validation rungs (n ≲ 1500) but not
/// for VerbAgg (n=7584, the largest individual-Bernoulli binomial rung): at
/// a looser 1e-6 tolerance the absolute slack would be ~8e-3 in deviance, one Newton step
/// short of the quadratic-convergence cliff canonical links otherwise reach
/// "for free" (see `PIRLS_TOL_REL_NONCANON`'s doc comment) — BOBYQA's stage-2
/// objective then carried that ~1 iteration of leftover curvature noise into
/// β̂, landing 4e-3 relative off both lme4 and MixedModels.jl (which agree
/// with each other to 8e-5) instead of the ~1e-4 every other binomial/poisson
/// rung achieves. This is a cliff, not a gradient: 1e-7 and 1e-8 still miss
/// VerbAgg's 1e-3 gate at ~4–5e-3 (the Newton step hasn't yet landed inside
/// the quadratic basin), 3e-9 clears it at 5e-4; 1e-9 keeps a ~5× margin.
/// The 1e-9 setting costs 2–4 extra inner iterations over a looser tolerance,
/// but only until each canonical-link solve reaches its own quadratic floor —
/// negligible on every rung's fit time (verified via `validation/compare.R`
/// full-suite pass, no rung regressed).
pub const PIRLS_TOL_REL: f64 = 1e-9;
/// PIRLS exit for NON-canonical links (probit, Gamma-log/inverse, NB-log) — a
/// decade looser than `PIRLS_TOL_REL`, tight enough to stay well clear of the
/// 1e-6 accuracy cliff described below. Canonical links (logit, Poisson-log) are Newton ⇒ quadratic
/// convergence, so PIRLS overshoots `PIRLS_TOL_REL` to ~machine precision on its
/// own and their deviance is already smooth enough for both the outer BOBYQA β
/// and the FD-Hessian SE. Non-canonical links are Fisher-scoring ⇒ only LINEAR
/// convergence, so the deviance is smooth only to the exit tolerance; at the
/// canonical 1e-6 that ~1e-4 floor leaves β ~3e-3 off and the FD second
/// differences (÷ step²) amplify the noise into a 7–41%-wrong `se_hessian`.
/// Tightening removes that, but the accuracy PLATEAUS: a cbpp-probit tolerance
/// sweep (see `fit::tests::fit_glmm_probit_cbpp_matches_lme4`) shows β pinned at
/// ~5e-5 (40× inside its 2e-3 golden limit) flat across 1e-8…1e-10, with a sharp
/// cliff only at 1e-6→1e-7. 1e-8 sits one decade above the cliff: same β margin
/// as 1e-10 (tighter buys no β safety, only iterations), `se_hessian` 34× inside
/// limit, ~1.5× inner iters vs canonical — paid only on non-canonical fits, and
/// only when SEs go through the FD-Hessian (`se_rx` skips it entirely).
pub const PIRLS_TOL_REL_NONCANON: f64 = 1e-8;
/// CEILING on the PIRLS exit tolerance under the FD-Hessian SE evals ONLY
/// (`joint_hessian_cov`, either stencil arm). Never applied on its own —
/// `pirls_tol_fd` takes `min(this, pirls_tol(family))`, so the SE pass is always
/// at least as converged as the fit that produced the point it differences.
/// Applying it as a plain replacement would put canonical links (fitting at
/// `PIRLS_TOL_REL` = 1e-9) on a LOOSER deviance under the stencil than the one
/// the optimizer converged on — an inversion of what an FD-only tolerance is
/// for, and the reason the `min` is not optional.
///
/// Sizing of the 1e-8: the FD second differences divide the deviance by step²
/// (~1e-4), so PIRLS exit noise reaches `se_hessian` amplified 1e4×. At an exit
/// tolerance of 1e-6 that shows up as a dataset-dependent ~0.3% step wobble
/// (cbpp, h=1e-2 vs 1e-3 — the shipped step landed on the accurate side by
/// measurement, not construction). At 1e-8 the cbpp Hessian is step-invariant to
/// ~5–6 sig figs across h ∈ {1e-2, 1e-3, 1e-4} and sits on the
/// tight-tolerance (1e-12) limit — accurate by construction. Deeper buys
/// nothing: the diagnosis sweep was already flat from 1e-10 to 1e-13, which is
/// why this is a ceiling and not a value that tracks `PIRLS_TOL_REL` downward.
///
/// The fit path never pays it — BOBYQA objective evals keep `pirls_tol`; the
/// cost lands only on the ~m² SE evals that already dominate `WaldSe::Hessian`
/// timing (the same ~1.5×-inner-iteration precedent as
/// `PIRLS_TOL_REL_NONCANON`, whose value this matches).
pub const PIRLS_TOL_REL_FD: f64 = 1e-8;
/// PIRLS exit tolerance for `family`: the standard value for canonical (Newton,
/// quadratic) links, the tight value for non-canonical (Fisher-scoring, linear)
/// links (canonical links overshoot to machine precision, non-canonical don't).
pub(crate) fn pirls_tol(family: crate::spec::Family) -> f64 {
    if crate::family::is_canonical(family) {
        PIRLS_TOL_REL
    } else {
        PIRLS_TOL_REL_NONCANON
    }
}
/// PIRLS exit tolerance for the FD-Hessian SE evals: the tighter of the FD
/// ceiling and the family's own fit tolerance, so the stencil can only ever be
/// more converged than the fit, never less. Canonical links take
/// `PIRLS_TOL_REL` (1e-9); non-canonical links fit at 1e-8 and so take the
/// ceiling. `glmm::se::joint_hessian_cov` writes it into `pirls_tol_override` on
/// both of its stencil arms.
pub(crate) fn pirls_tol_fd(family: crate::spec::Family) -> f64 {
    PIRLS_TOL_REL_FD.min(pirls_tol(family))
}
/// Wide finite β box for the joint BOBYQA — the bounds handed to the optimizer
/// for the β block, and the clamp applied to a warm-start β.
///
/// Deliberately NOT tied to the GLM route's divergence cap any more. That cap
/// bounds the linear predictor and refuses a fit; this is a box that keeps a
/// derivative-free optimizer inside a finite region. They are different objects
/// that happen to share the magnitude 30, and aliasing them made a change to one
/// silently a change to the other. Like any absolute bound on β this box is
/// itself unit-dependent; that is a known, separate question.
pub const BETA_BOX: f64 = 30.0;
/// Base FD step for `joint_hessian_cov`'s joint-deviance Hessian. It is applied
/// ASYMMETRICALLY across the joint (θ, β) vector: `h_θ = FD_STEP_BASE` absolute on
/// the θ block, `h_β = FD_STEP_BASE·max(1, |β̂_k|)` relative on the β block.
/// β enters through η = Xβ and wants relative stepping; θ does not — scaling h_θ
/// with the random-effect SD widens the window exactly where the deviance profile
/// in θ flattens, so the central second difference's O(h²) truncation error grows
/// as θ̂². Measured on the corpus 2026-07-30: dropping the θ scaling divides the
/// distance to the h→0 limit by θ̂² to within 6% on all seven scalar-RE rungs whose
/// θ̂ exceeds 1, at nAGQ = 1 and 7 and 11 alike (θ̂ 1.13 → 5.16, a 20× range in
/// θ̂²), on both `se_hessian` and the θ-block SEs. The step
/// construction, and the toenail evidence behind it, is at `se.rs`'s
/// `ws.fd.fd_steps` loop.
///
/// The BASE VALUE 1e-2 is unchanged by that fix and stays pinned by the curated
/// sweep, not just by the fixture: the Hessian is step-invariant over h ∈
/// [1e-4, 1e-1] on the committed fixture — but NOT on every curated rung:
/// h = 1e-3 blows the validation se_hess gates on the noise side (sim_gamma 1e-2,
/// cbpp_probit 2e-3 vs the 1e-3 band) while 1e-2 holds them at ~1e-4. Both of
/// those rungs have θ̂ < 1, so `max(1, |θ̂|)` was already exactly 1 there and the
/// asymmetry above does not disturb the sweep that pinned this number — h_θ on
/// them is bit-for-bit what it was. Independently, the measured noise knee in θ is
/// at h_θ ≈ 2.5e-4, so 1e-2 keeps a 40× margin on the truncation side.
/// The packed-row layout needs the opposite trade and carries its own
/// `se::SPARSE_FD_STEP_REL` (1e-4, landing on the weighted sparse-Z Gamma
/// golden's FD-step plateau — 1e-3 biases se(β₀) high there) — calibrated
/// separately, do not fold the two constants together. The θ-step fix does NOT
/// transfer to it: that constant is calibrated on the noise side, so removing the
/// θ scaling there would push h_θ further into noise, not out of it.
///
/// **The `_BASE` / `_REL` split in the two names is deliberate.** This one is
/// `_BASE` because the asymmetry above makes "relative" false of the θ block;
/// `SPARSE_FD_STEP_REL` keeps its suffix because it really is applied relatively
/// on every coordinate, θ included. If the packed step ever takes the same
/// asymmetry, rename it in the same change, or the suffix starts lying there
/// instead.
pub const FD_STEP_BASE: f64 = 1e-2;

/// Per-fit GLMM result (mirrors `LmmFit`; no σ² — dispersion is fixed at 1).
pub struct GlmmFit {
    /// `true` iff the outer BOBYQA converged AND the pinned-γ̂ re-eval deviance is
    /// finite. `false` on two distinct exits: an outer search that exhausted its
    /// evaluation budget (`Status::MaxFunReached`) still reports its incumbent
    /// β̂, SEs and deviance here, with `boundary_hit = 2` and
    /// `pinned_components` empty; every other non-convergence (BOBYQA failure,
    /// a non-PD Schur, the degenerate-fit guard) NaN-fills every inference
    /// field instead (see `nan_fit`).
    pub converged: bool,
    /// 0 = interior, 1 = ≥1 diagonal θ pinned (converged), 2 = optimizer/Schur
    /// failure or a budget-exhausted exit (non-converged).
    pub boundary_hit: u8,
    /// Bit k set iff diagonal variance component k pinned (order
    /// [intercept, slope_0, …, extra_1, …]). Mirrors `LmmFit.pinned_components`
    /// (u64 mask — see there for the over-envelope component-count rationale).
    pub pinned_components: u64,
    /// Total BOBYQA evaluations across both stages (stage 1 + stage 2; 0 + stage 2
    /// on the single-stage path).
    pub n_eval: usize,
    /// Observation-only evaluation counters for this fit. Gated because
    /// `GlmmFit` is re-exported `pub` under `loop_advanced`: with `counters`
    /// off, that tier's surface must be unchanged.
    #[cfg(feature = "counters")]
    pub counters: crate::counters::EvalCounters,
    /// Estimated random-intercept variance D̂[0][0] (NaN on non-converged).
    pub tau_squared_hat: f64,
    /// Joint Wald-χ² over `target_indices` (NaN when empty / non-converged).
    pub joint_t_sq: f64,
    /// Set iff the fit ran `WaldSe::Hessian` and `joint_hessian_cov` fell back to the
    /// RX/Schur block (non-PD joint Hessian / non-finite perturbed deviance) — its
    /// `NonPdFellBackToRx` status. Always `false` under `WaldSe::Rx`.
    pub hessian_fallback: bool,
    /// Minimized marginal Laplace deviance at the pinned γ̂ (`d(y,ũ)+‖ũ‖²+log|A|`,
    /// or the AGQ deviance when `nagq>1`), reported as `Fit::deviance`.
    /// `f64::INFINITY` on non-convergence. On an NB fit it carries
    /// `−2·saturated_loglik(θ̂)` on top — not the search's own `nb_term`, which
    /// differs from it by the θ̂-independent constant `2·Σwᵢ·lgamma(yᵢ+1)` — so
    /// the reported value is exactly `−2·logLik`.
    pub deviance: f64,
}

mod agq;
mod assembled;
mod derivative;
mod deviance;
mod pirls;
mod se;
mod workspace;

#[cfg(test)]
pub(crate) use deviance::glmm_laplace_deviance;
// Re-exported so `lmm::kernel`'s REML dual entry points can reuse
// these instead of duplicating them — `derivative` itself is private to
// `glmm`, so a sibling module needs the items re-exported one level up.
pub(crate) use derivative::{unpack_hessian, DerivStatus};
// Both exact Hessian engines, reachable from the test module that drives the
// validation corpus's own datasets through them side by side. Production
// reaches them through `se::joint_hessian_cov` alone. Every caller of these
// four is a `#[cfg(feature = "formula")]` test in `fit::glmm_tests`, so the
// re-export carries that gate too — otherwise `--no-default-features` warns
// on an import nothing left standing can use.
#[cfg(all(test, feature = "formula"))]
pub(crate) use assembled::{
    assembly_routes, gradient_f64, gradient_f64_mode_residual, joint_hessian, mu_clamped_rows,
    packed_gradient,
};
// The unsymmetrized column pass, read by the corpus drivers in
// `fit::glmm_tests` and by the both-layouts cross-check in `sparse::tests`,
// which carries no `formula` gate.
#[cfg(test)]
pub(crate) use assembled::joint_hessian_columns;
// The paired-timing driver's forcing switch and success counter — see their
// doc comments in `assembled.rs`. Same formula-only callers as the block
// above.
#[cfg(all(test, feature = "formula"))]
pub(crate) use assembled::{ASSEMBLED_OK_COUNT, FORCE_DECLINE};
// Same formula-only caller as the two blocks above.
#[cfg(all(test, feature = "formula"))]
pub(crate) use derivative::{
    laplace_gradient, laplace_hessian, supports_shape as supports_exact_shape, MAX_DUAL_N,
};
pub use se::joint_hessian_cov;
// The FD stencils and the packed-row step constant: `se.rs` is their only
// production reader, the FD-margin measurement their only cross-module one.
#[cfg(all(test, feature = "formula"))]
pub(crate) use se::{fd_mixed_diff, fd_second_diff, SPARSE_FD_STEP_REL};
pub(crate) use workspace::fill_packed_cols;
pub(crate) use workspace::StructuredSchur;
#[cfg(test)]
pub(crate) use workspace::{glmm_block_chol, glmm_block_solve};
pub use workspace::{GlmmWorkspace, OuterSearch};

use deviance::laplace_deviance;
use pirls::BetaMode;

// `pub(crate)` so `src/scalar.rs`'s tail-methods test can reuse
// `glmm_extras_q1_dataset` instead of duplicating a structured fixture.
#[cfg(test)]
pub(crate) mod tests;

/// Outcome of the joint-Hessian fixed-effect covariance (either arm — the
/// exact hyper-dual one on every shape `derivative::supports_shape` accepts,
/// the FD stencil elsewhere).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdHessianStatus {
    /// The joint-deviance Hessian was PD and every perturbed eval finite; the
    /// returned covariance is `2·(H_dev⁻¹)_ββ` (lme4 `vcov(use.hessian = TRUE)`).
    Ok,
    /// The joint Hessian was non-PD (or a perturbed deviance was non-finite — the
    /// few-cluster failure mode); the returned covariance is the RX/Schur block.
    NonPdFellBackToRx,
}

/// Fit the clustered-logistic GLMM. `fill_packed_cols` must already have run (a
/// no-op off the packed layout, whose buffers are zero-length) for this
/// (X, ids, N). `beta_start` = spec.effect_sizes. Writes β̂/Var/z² into
/// `ws.betas`, `ws.inference.var_diag`, `ws.inference.t_sq`; returns the GlmmFit summary.
///
/// Convention: `wald_se` selects the fixed-effect covariance — `WaldSe::Rx`
/// inverts the expected-information Schur complement directly (assumes β–θ
/// orthogonality; anticonservative for the GLMM); `WaldSe::Hessian` (glmer
/// `use.hessian = TRUE`) sources it from the Hessian of the joint Laplace
/// deviance instead (exact on every shape `derivative::supports_shape`
/// accepts, finite-difference elsewhere), the lme4-matching default. The optimizer
/// runs one of three routes fixed per shape by `ws.outer_search` (see `OuterSearch`):
/// `Joint` searches `[θ | β]` directly; `PqlThenJoint` profiles β out of a θ-only
/// BOBYQA on the PQL objective purely as a warm-start accelerant, then a joint
/// `[θ | β]` BOBYQA polish alone gates convergence; `ExactProfile` profiles β out
/// EXACTLY at each candidate θ, so its θ-only BOBYQA pass alone gates convergence —
/// no joint polish follows. On every route the reported (θ̂, β̂) is the Laplace
/// optimum, not the PQL one.
///
/// Matches `lme4::glmer` (binomial cbpp fixture; see
/// `fit::tests::fit_glmm_cbpp_matches_lme4`).
///
/// Read-back: on `converged == true`, `ws.params[..n_theta + p]` holds the
/// pinned optimum `[θ̂ | β̂]` — the stable convention documented on
/// `GlmmWorkspace::params`. On `ExactProfile` this β̂ itself is never
/// BETA_BOX-projected (see the `n_eval` write-back below); callers may still
/// feed it back as `theta_start`/`beta_start` for a subsequent fit of related
/// data, and that fed-back start passes through the same
/// `THETA_TRUTH_FLOOR`/`BETA_BOX` clamps as any other start, regardless of
/// which route produced it.
///
/// Errors: no `Result`. An outer search that exhausts its evaluation budget
/// (`Status::MaxFunReached`) still reports its incumbent through `GlmmFit`:
/// `converged = false`, `boundary_hit = 2`, `pinned_components` empty, and the
/// incumbent's β̂, SEs and deviance in the usual fields. Every other
/// non-convergence (BOBYQA failure, a non-PD Schur, or the degenerate-fit
/// guard tripping on an all-infeasible BOBYQA simplex) NaN-fills every
/// β̂/SE/deviance field instead (`f64::NAN` or `f64::INFINITY`; see `nan_fit`).
#[allow(clippy::too_many_arguments)]
pub fn fit_glmm(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    beta_start: &[f64],
    n: usize,
    wald_se: WaldSe,
) -> GlmmFit {
    // Backstop for a hand-built workspace: the blocked and structured layouts
    // build intercept-only extras (`build_packed_m` carries the per-eval
    // debug_assert). `classify_design` routes any extra-slope shape to Sparse,
    // so this is unreachable through `fit_on`/`fit_warm` — it catches a caller
    // that constructs `GlmmWorkspace` directly. Mirrors the same debug_assert in
    // `build_packed_m` (`glmm/workspace.rs`) and `assemble_ranef_dense`
    // (`fit/common.rs`) — change together.
    assert!(
        ws.layout == GlmmLayout::Packed || !ws.groupings.extra_slopes_any,
        "GLMM entry: extra_slopes_any on a blocked or structured layout"
    );
    let (k, p, n_theta) = (ws.k, ws.p, ws.n_theta);

    // γ₀ = [θ₀ | β₀].
    match theta_start {
        Some(ts) => {
            // Floor is diagonal-only: a caller's θ is a Cholesky factor, so its
            // diagonals are non-negative and the floor only keeps them off the
            // pin threshold; off-diagonals are signed, so flooring them would
            // silently rewrite a negative correlation start. lme4 passes
            // `start$theta` through verbatim.
            // A caller's θ is in the design's own units; the solver works on the
            // internally scaled θ̃ = s·θ (`LmmGroupings::set_slope_scales`), so the
            // forward map runs before the floor — which then floors the INTERNAL
            // diagonals, the same scale `PIN_THETA` tests.
            // `MAX_THETA` is the dense envelope's vech ceiling, so the stack
            // block covers every in-envelope shape and a warm loop pays no heap
            // block for the forward map. The packed layout also takes designs
            // past that ceiling (an over-envelope primary block, say), and those
            // — off the warm-loop envelope by construction — take the `Vec`.
            let mut stack = [0.0_f64; crate::consts::MAX_THETA];
            let mut heap: Vec<f64>;
            let s: &mut [f64] = if n_theta <= crate::consts::MAX_THETA {
                &mut stack[..n_theta]
            } else {
                heap = vec![0.0; n_theta];
                &mut heap
            };
            ws.groupings.fill_theta_row_scales(s);
            for ((t, &v), &sc) in ws.params[..n_theta].iter_mut().zip(ts).zip(s.iter()) {
                *t = v * sc;
            }
            for &i in ws.groupings.diagonal_theta() {
                ws.params[i] = ws.params[i].max(THETA_TRUTH_FLOOR);
            }
        }
        None => {
            // Blind start: diagonals THETA0, off-diagonals 0 — the structure-only
            // blind θ₀ `GlmmWorkspace::new` builds (workspace.rs). The former
            // all-THETA0 start implied RE correlation +0.707 for every pair, and
            // on negative-correlation data that converges into the τ=0 boundary
            // basin.
            for t in ws.params[..n_theta].iter_mut() {
                *t = 0.0;
            }
            for &i in ws.groupings.diagonal_theta() {
                ws.params[i] = THETA0;
            }
        }
    }
    for (j, &b) in beta_start.iter().enumerate().take(p) {
        ws.params[n_theta + j] = b.clamp(-BETA_BOX, BETA_BOX);
    }

    // NB: the trailing outer-search coordinate starts at ln of the seed the
    // adapter left in `nb_theta`, clamped into the bracket's box.
    let n_nb = usize::from(matches!(ws.family, crate::Family::NegativeBinomial { .. }));
    let m = n_theta + p;
    if n_nb == 1 {
        ws.params[m] = ws.nb_theta.ln().clamp(ws.lower[m], ws.upper[m]);
    }

    // Within-fit warm-start resets per fit — the incumbent u_seed is NEVER carried
    // across fits. Carrying it would make a fit's result depend on which fits ran
    // before it in the workspace, breaking both the guarantee that a given (spec,
    // data, seed) fits the same way regardless of call order and the reset-state
    // assumption `fit_cold` makes about a fresh `GlmmWorkspace`.
    for v in ws.u_seed[..k].iter_mut() {
        *v = 0.0;
    }
    // Observation-only PIRLS-exhaustion counters — reset per fit like u_seed
    // above, so a `loop_advanced` reuse of this workspace never carries a prior
    // draw's count into the next (see `Note::PirlsExhausted`).
    ws.pirls_exhausted = 0;
    ws.final_pirls_exhausted = false;
    ws.counters.reset();
    ws.pattern.coup_mask = None; // CSR validity is per (fit, pinning mask): ids/z may differ across fits
                                 // Cluster-outer AGQ substrate: built once per fit (cluster_ids is fit-fixed),
                                 // ONLY in `parallel` builds — it exists as rayon's work-splitting substrate.
                                 // Serial builds always run the original node-outer loop: even with the
                                 // reweight hoist in agq_deviance, cluster-outer's residual per-cluster
                                 // overhead (loop restart + CSR gather) regresses many-tiny-cluster shapes
                                 // (observation-level REs: +12–16% measured on grouseticks (1|INDEX), 403
                                 // clusters × 1 row) while its serial cache win on large clusters is small
                                 // (−4% on cbpp), so the non-parallel hot path (batch loops) stays
                                 // byte-for-byte unchanged. The grid campaign's parallel pass owns the
                                 // decision of whether a rows-per-cluster dispatch can unlock the win.
    ws.cluster_rows = if cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
        && ws.nagq > 1
        && ws.parallel_inner
    {
        Some(agq::ClusterRowIndex::build(
            cluster_ids,
            ws.groupings.n_primary,
        ))
    } else {
        None
    };

    // Joint BOBYQA over [θ | β]. Borrow-split mirrors `glmm_laplace_deviance`:
    // `solver` held by `minimize`; the closure calls the shared `laplace_deviance`
    // on the disjoint scratch fields (groupings read; m/lam/eta/prob/w/u/a/a_rhs
    // written). The closure's `gamma` is BOBYQA's candidate point, not the bound
    // `params` (which `minimize` owns as its `x`).
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let nagq = ws.nagq;
    // Outer search route for this shape (`OuterSearch`). Read before the
    // destructure moves `ws`'s fields out by mutable reference; the enum is Copy
    // so this is a plain read.
    let route = ws.outer_search;
    debug_assert!(
        route != OuterSearch::ExactProfile || exact_profile_shape(family, nagq, ws.layout),
        "ExactProfile requested on a shape without an exact profile"
    );
    // Which `BetaMode` stage 1 profiles β with, or `None` to skip stage 1 entirely
    // (the `Joint` route). `stage1_mode` alone decides whether stage 1 runs; the
    // route decides what happens AFTER it — see the `route ==` split below.
    let stage1_mode = match route {
        OuterSearch::Joint => None,
        OuterSearch::PqlThenJoint => Some(BetaMode::ProfilePql),
        OuterSearch::ExactProfile => Some(BetaMode::ProfileExact),
    };
    let weighted = ws.weighted;
    let offset = ws.offset.as_deref();
    let GlmmWorkspace {
        solver,
        solver_stage1,
        params,
        params_stage1,
        lower_stage1,
        upper_stage1,
        beta_rhs,
        lower,
        upper,
        groupings,
        cluster_rows,
        layout,
        prior_w,
        u_seed,
        wx,
        pirls,
        packed,
        structured,
        pattern,
        z_buf,
        border,
        beta_prof,
        beta_seed,
        exact_prof,
        p: pf,
        pirls_exhausted,
        counters,
        ..
    } = ws;
    // Joint vector's trailing `ln θ_NB` index.
    let nb_col = n_theta + p;
    // x is fixed for this fit: widen the slope columns to f64 once (blocked AND
    // structured paths — `build_packed_m`'s primary-core reduction reads it the
    // same way `pirls_solve_blocked`'s does), so every BOBYQA eval's per-solve M
    // fill runs MatRef-free. The packed layout skips it: it never reads z_buf
    // (`fill_m_vals` reads `x` directly). `se.rs`'s `joint_hessian_cov` mirrors
    // this same hoist for its own FD deviance evals.
    if *layout != GlmmLayout::Packed {
        fill_z_f64(groupings, x, z_buf, n);
    }
    // Read-only design view every `laplace_deviance` call below shares — see
    // [`FitData`]. Borrows `groupings`/`z_buf`/`prior_w` only for the lifetime
    // of this destructure's borrow of the workspace; the pinned γ̂ re-evaluation
    // builds its own view from a fresh workspace destructure.
    let data = FitData {
        family,
        groupings,
        layout: *layout,
        x,
        y,
        prior_w: &prior_w[..n],
        weighted,
        cluster_ids,
        extra_ids,
        z_buf,
        offset,
        n,
        p: *pf,
    };

    // NB marginal objective: `dev(θ_RE, β, θ_NB) − 2·nb_profile_loglik(y, y, θ_NB, w)`,
    // the `logL_marginal` the GLM θ profile (`fit::optimize_nb_theta`) maximizes,
    // times −2, so the optimum is the same point; the saturated term depends on
    // θ_NB alone. Same weights the
    // deviance carries (`prior_w`, when `weighted`).
    let nb_term = |ln_theta: f64| -> f64 {
        -2.0 * crate::fit::nb_profile_loglik(
            y,
            y,
            ln_theta.exp(),
            weighted.then_some(&prior_w[..n]),
        )
    };

    // STAGE 1 — θ-only BOBYQA profiling β out of PIRLS at each candidate θ. On
    // `PqlThenJoint` this is an accelerant that warm-starts the `Joint` polish below
    // and never gates convergence; on `ExactProfile` this IS the search — its
    // status alone gates convergence (the `route ==` split after this block).
    // Skipped bit-identically when `stage1_mode` is `None` (the `Joint` route; the
    // A/B tests pin this as the single-stage reference) and when `nagq > 1` —
    // Profile deviance is undefined on the AGQ early-return path
    // (`debug_assert!(beta_mode == BetaMode::Fixed || nagq == 1)`), and AGQ fits must bypass stage 1
    // unchanged. A Laplace-pass warm start for AGQ was measured on the 33 diligent
    // AGQ cells (2026-07-14) and reverted: total eval count was a wash (−0.3%),
    // not worth the added code path for that little a gain.
    let mut n_eval_stage1 = 0usize;
    let mut stage1_out = None;
    // Hoisted out of the `if let` below (rather than left local to it) so it is
    // still in scope at the `ExactProfile` status read further down: a search
    // whose incumbent never left `+INFINITY` (every PIRLS call on this θ start
    // diverges) can still exhaust its rho ladder and report `Status::Converged`,
    // so `converged` must also check that a finite objective was ever seen.
    let mut best1 = f64::INFINITY;
    // PRIMA's `moderatef` maps both `NaN` and `+inf` to `FUNCMAX = 1e30` (bobyqa's
    // objective-value contract), so a run where every PIRLS call diverges is a flat
    // *finite* surface to BOBYQA: it shrinks the trust region to `rho_end` and exits
    // `Status::Converged` having learned nothing. `best1.is_finite()` only catches the
    // all-`+INF` case — it turns finite on the very first finite evaluation, so a run
    // with exactly one genuine evaluation (start included) still passes. Count how many
    // evaluations were actually finite and require at least 2: the smallest bar for "the
    // search compared two real points".
    let mut finite_evals1 = 0usize;
    if let Some(mode) = stage1_mode.filter(|_| nagq == 1) {
        params_stage1[..n_theta].copy_from_slice(&params[..n_theta]); // θ₀ unchanged from the caller's params
        if n_nb == 1 {
            params_stage1[n_theta] = params[nb_col]; // ln θ_NB start, set before the search
        }
        beta_prof[..p].copy_from_slice(&params[n_theta..n_theta + p]); // β₀ (GLM warm start)
        beta_seed[..p].copy_from_slice(&beta_prof[..p]);
        // The incumbent state — best value, finite-evaluation count and the
        // û/β̂ snapshots — is passed in rather than captured so the closure
        // borrows only the PIRLS buffers.
        let mut stage1_obj = |theta: &[f64],
                              best: &mut f64,
                              finite: &mut usize,
                              u_seed: &mut [f64],
                              beta_seed: &mut [f64]|
         -> f64 {
            // Re-seed BOTH latent states from the INCUMBENT (best point so far),
            // not the last-evaluated: BOBYQA's final call is not its best.
            // û is point-determined given θ, and β is likewise point-determined
            // (the PQL β̂(θ)); the incumbent seed only shifts the stopping iterate
            // within tol — the same argument that justifies the u_seed warm start.
            pirls.u[..k].copy_from_slice(&u_seed[..k]);
            beta_prof[..p].copy_from_slice(&beta_seed[..p]);
            // Profile mode SWAPS the β buffers vs the Fixed stage-2 call below:
            // `beta = beta_prof` (in/out profiled β), `beta_step_rhs = beta_rhs`
            // (the δβ border scratch). See deviance.rs's buffer-role comment.
            let (nb, coord) = if n_nb == 1 {
                (theta[n_theta].exp(), Some(theta[n_theta]))
            } else {
                (nb_theta, None)
            };
            let dev = laplace_deviance(
                &data,
                nb,
                nagq,
                &theta[..n_theta],
                beta_prof,
                wx,
                pirls,
                structured,
                pattern,
                packed,
                // Stage 1 writes ws.border.schur via the Profile S_β border; the post-fit
                // SE path runs AFTER stage 2 and its blocked/structured/packed
                // schur_fill rebuilds ws.border.schur from scratch, so this transient use
                // is safe (no read survives into inference).
                border,
                beta_rhs,
                exact_prof,
                // `mode` is `ProfilePql` on `PqlThenJoint`, `ProfileExact` on
                // `ExactProfile` — the two routes that reach this block.
                mode,
                // Never the FD-pass tol here — stage-1 objective evals stay at
                // `pirls_tol` (the field is None outside `joint_hessian_cov`).
                None,
                cluster_rows.as_ref(),
                pirls_exhausted,
                counters,
            );
            let obj = match coord {
                Some(c) => dev + nb_term(c),
                None => dev,
            };
            if obj.is_finite() {
                *finite += 1;
            }
            if obj < *best {
                *best = obj;
                // INCUMBENT-gated snapshot — snapshot only on strict
                // improvement, NOT every eval.
                u_seed[..k].copy_from_slice(&pirls.u[..k]);
                beta_seed[..p].copy_from_slice(&beta_prof[..p]);
            }
            counters.record_eval(crate::counters::Stage::One, obj);
            obj
        };
        let out1 = solver_stage1.minimize(
            |theta| {
                stage1_obj(
                    theta,
                    &mut best1,
                    &mut finite_evals1,
                    &mut u_seed[..],
                    &mut beta_seed[..],
                )
            },
            params_stage1,
            lower_stage1,
            upper_stage1,
        );
        // On `PqlThenJoint`, non-convergence does NOT fail the fit — stage 1 is an
        // accelerant, and the joint polish below proceeds from wherever the
        // incumbent landed (worst case = the cold start). On `ExactProfile`
        // this status IS the fit's convergence status — read below.
        n_eval_stage1 = out1.n_eval;
        // Warm-start / final θ̂,β̂: θ̂₁ (BOBYQA leaves the incumbent in `params_stage1`)
        // and β̂₁ (the incumbent snapshot, NOT `beta_prof`'s last-evaluated value).
        // On `ExactProfile` this write IS the reported optimum, not a warm start.
        params[..n_theta].copy_from_slice(&params_stage1[..n_theta]);
        // β̂₁ is not re-clamped to ±BETA_BOX here: the bobyqa crate itself projects
        // an out-of-box start onto the bounds (PRIMA moderatex-style preproc), so
        // a subsequent joint solve's init already lands in-box without a redundant
        // clamp — and `ExactProfile` never re-clamps at all.
        params[n_theta..nb_col].copy_from_slice(&beta_seed[..p]);
        if n_nb == 1 {
            params[nb_col] = params_stage1[n_theta];
        }
        stage1_out = Some(out1);
    }

    // `budget_exhausted` drives the plateau policy mirrored from `fit_lmm`
    // (`src/lmm/mod.rs`): a
    // `Status::MaxFunReached` exit reports its finite incumbent below with
    // `converged == false` and `boundary_hit == 2` rather than NaN-filling.
    let budget_exhausted;
    let (ok, n_eval) = if route == OuterSearch::ExactProfile {
        // ExactProfile: stage 1 IS the search — θ-only BOBYQA on the exact Laplace profile.
        // `params` already holds [θ̂ | β̂] (the incumbent snapshot written above);
        // no joint polish follows, so `ws.u_seed` written by stage 1's own
        // incumbent-gated snapshot is already the final conditional mode.
        let o = stage1_out.as_ref().expect("ExactProfile runs stage 1");
        debug_assert!(o.status != Status::InvalidArgs);
        budget_exhausted = matches!(o.status, Status::MaxFunReached);
        (
            matches!(o.status, Status::Converged | Status::MaxFunReached)
                && best1.is_finite()
                && finite_evals1 >= 2,
            o.n_eval,
        )
    } else {
        let mut best_obj = f64::INFINITY;
        // mirrors the stage-1 read above — change together.
        let mut finite_evals2 = 0usize;
        let out = solver.minimize(
            |gamma| {
                // Within-fit û warm-start: seed PIRLS from the incumbent (best point so
                // far), not from 0. The conditional mode is point-determined, so the seed
                // only shifts the stopping iterate within the PIRLS exit band.
                pirls.u[..k].copy_from_slice(&u_seed[..k]);
                let (nb, coord) = if n_nb == 1 {
                    (gamma[nb_col].exp(), Some(gamma[nb_col]))
                } else {
                    (nb_theta, None)
                };
                let dev = laplace_deviance(
                    &data,
                    nb,
                    nagq,
                    gamma,
                    beta_rhs,
                    wx,
                    pirls,
                    structured,
                    pattern,
                    packed,
                    // The `Joint` polish's objective is β-FIXED; the Profile border
                    // scratch is inert. `beta_prof` is the spare distinct buffer
                    // (`beta_rhs` is `beta` above). Stage 1 above flips this to Profile
                    // (`ProfilePql` or `ProfileExact`, by `stage1_mode`).
                    border,
                    beta_prof,
                    exact_prof,
                    BetaMode::Fixed,
                    // Never the FD-pass tol here — BOBYQA objective evals stay at
                    // `pirls_tol` (the field is None outside `joint_hessian_cov`).
                    None,
                    cluster_rows.as_ref(),
                    pirls_exhausted,
                    counters,
                );
                let obj = match coord {
                    Some(c) => dev + nb_term(c),
                    None => dev,
                };
                if obj.is_finite() {
                    finite_evals2 += 1;
                }
                if obj < best_obj {
                    best_obj = obj;
                    u_seed[..k].copy_from_slice(&pirls.u[..k]);
                }
                counters.record_eval(crate::counters::Stage::Two, obj);
                obj
            },
            params,
            lower,
            upper,
        );

        debug_assert!(out.status != Status::InvalidArgs);
        budget_exhausted = matches!(out.status, Status::MaxFunReached);
        // Reported eval count is stage 1 + stage 2 (0 + stage 2 on the `Joint` route,
        // so byte-identical to a single-stage run). Only stage 2's status feeds `converged`.
        (
            matches!(out.status, Status::Converged | Status::MaxFunReached)
                && best_obj.is_finite()
                && finite_evals2 >= 2,
            n_eval_stage1 + out.n_eval,
        )
    };

    // NB: the incumbent's dispersion becomes the workspace's fixed θ_NB for
    // everything downstream — the pinned re-eval, the SE path — which all read
    // `ws.nb_theta` the way they read it for a fixed-θ fit.
    if n_nb == 1 {
        ws.nb_theta = ws.params[nb_col].exp();
    }
    let nb_theta = ws.nb_theta;
    // The reported deviance's NB correction: `−2·saturated_loglik(θ̂)`, NOT
    // `nb_term(θ̂)`. `nb_term` (used above, inside the search only) is
    // `−2·nb_profile_loglik(y,y,θ,w)`, which differs from `−2·saturated_loglik`
    // by the θ-independent constant `2·Σwᵢ·lgamma(yᵢ+1)` — irrelevant to an
    // argmin, so the search keeps using `nb_term` unchanged, but wrong by that
    // constant if reused here, where the exact value is what makes
    // `deviance = −2·loglik` hold. Cached now, as a plain f64: this reads
    // `prior_w` out of the destructure above, a borrow the pinned re-eval's own
    // reborrow of `ws.prior_w` further down cannot coexist with.
    let nb_dev_term = if n_nb == 1 {
        -2.0 * crate::family::saturated_loglik(
            family,
            nb_theta,
            y,
            weighted.then_some(&prior_w[..n]),
        )
    } else {
        0.0
    };

    // Sign canonicalization first, then the per-component diagonal pin (β never
    // pins) — mirror `fit_lmm`, `src/lmm/mod.rs`; change together. `diag`
    // borrows ws.groupings; the loop mutates the disjoint field ws.params, so
    // no clone is needed. `ok` is true on both a converged fit and a
    // budget-exhausted one, so the pin below applies to any reported
    // endpoint; `pinned`/`pinned_components` are still latched here on both,
    // but only survive into the returned `GlmmFit` on a converged fit — a
    // budget-exhausted exit zeroes `pinned_components` at assembly.
    if ok {
        // The modes flip with their columns (`fix_mode_signs`), before the θ
        // flip clears the signs that reads: `ws.u_seed` seeds the pinned re-eval.
        // The packed layout orders the primary block slope-major; blocked and
        // structured order it level-major.
        let primary_slope_major = ws.layout == GlmmLayout::Packed;
        crate::lmm::fix_mode_signs(
            &ws.groupings,
            &ws.params[..n_theta],
            &mut ws.u_seed,
            primary_slope_major,
        );
        crate::lmm::fix_column_signs(&ws.groupings, &mut ws.params[..n_theta]);
    }
    let diag = ws.groupings.diagonal_theta();
    let mut pinned_components = 0u64;
    let mut pinned = false;
    if ok {
        for (kk, &ti) in diag.iter().enumerate() {
            if ws.params[ti] <= PIN_THETA {
                ws.params[ti] = 0.0;
                pinned = true;
                if kk < u64::BITS as usize {
                    pinned_components |= 1u64 << kk;
                }
            }
        }
        // Σ-preserving canonical Λ, then the pin test again on the new
        // diagonals (mirror `fit_lmm`, `src/lmm/mod.rs` — change together).
        // The flags are rebuilt from scratch, not OR'd: the fold can retire a
        // component that pinned only as a degenerate coordinate. Nothing moves
        // unless a diagonal pinned, so an interior fit stays bit-identical, and
        // the pinned re-eval below sees the canonicalized θ.
        if crate::lmm::canonicalize_pinned_blocks(&ws.groupings, &mut ws.params[..n_theta]) {
            pinned = false;
            pinned_components = 0;
            for (kk, &ti) in diag.iter().enumerate() {
                if ws.params[ti] <= PIN_THETA {
                    ws.params[ti] = 0.0;
                    pinned = true;
                    if kk < u64::BITS as usize {
                        pinned_components |= 1u64 << kk;
                    }
                }
            }
        }
    }

    // Re-evaluate at the (possibly pinned) γ̂ to refresh M, ũ, W̃. ws.params already
    // holds the pinned γ̂, so call the kernel on it directly — no params copy. The
    // refreshed deviance is the reported marginal deviance (`GlmmFit::deviance`);
    // INFINITY until the re-eval runs.
    let mut final_deviance = f64::INFINITY;
    // Separate from `ws.pirls_exhausted`: exactly one PIRLS solve happens below,
    // so this is 0 or 1, folded into the reported bool after the block (the
    // final re-eval is the case where a truncated solve would feed the
    // returned estimates directly — see `Note::PirlsExhausted`).
    let mut final_exhausted_count = 0u32;
    // The pinned re-eval is not a search evaluation: it gets its own throwaway
    // counter for the same reason `final_exhausted_count` is separate above.
    let mut final_counters = crate::counters::EvalCounters::new();
    if ok {
        // Warm-start the pinned re-eval from the incumbent (its modes are the
        // inference iterate); u_seed holds the BOBYQA incumbent after minimize.
        ws.pirls.u[..k].copy_from_slice(&ws.u_seed[..k]);
        let GlmmWorkspace {
            groupings,
            params,
            beta_rhs,
            p,
            layout,
            z_buf,
            prior_w,
            pirls,
            packed,
            structured,
            pattern,
            wx,
            border,
            beta_prof,
            exact_prof,
            cluster_rows,
            offset,
            ..
        } = ws;
        // Shadow the function-level `offset` borrow (taken before the BOBYQA
        // destructure) with this destructure's own, scoped to the pinned re-eval.
        let offset = offset.as_deref();
        // z_buf still holds this fit's slope copy — x is unchanged since fill_z_f64.
        // On the structured path this re-eval re-packs m_core_buf/cross_* at γ̂, which
        // `structured_schur_fill` then reads.
        let data = FitData {
            family,
            groupings,
            layout: *layout,
            x,
            y,
            prior_w: &prior_w[..n],
            weighted,
            cluster_ids,
            extra_ids,
            z_buf,
            offset,
            n,
            p: *p,
        };
        final_deviance = laplace_deviance(
            &data,
            nb_theta,
            nagq,
            &params[..],
            beta_rhs,
            wx,
            pirls,
            structured,
            pattern,
            packed,
            border,
            // Pinned-γ̂ re-eval is β-FIXED (reports the fitted β); Profile border
            // scratch inert. `beta_prof` is the spare distinct buffer.
            beta_prof,
            exact_prof,
            BetaMode::Fixed,
            // Never the FD-pass tol here — the pinned re-eval stays at `pirls_tol`
            // (the field is None outside `joint_hessian_cov`).
            None,
            cluster_rows.as_ref(),
            &mut final_exhausted_count,
            &mut final_counters,
        );
    }
    ws.final_pirls_exhausted = final_exhausted_count > 0;

    // Degenerate-fit guard. BOBYQA can report `Converged` on a fit that never left
    // an infinite-deviance start: PRIMA's `moderatef` maps a `+inf` objective to the
    // finite ceiling FUNCMAX (1e30), so if the whole initial simplex is infeasible
    // the fval array is uniformly huge (a flat surface) and the trust region shrinks
    // to `rho_end` without ever finding a finite point — a `SMALL_TR_RADIUS` exit
    // that maps to `Status::Converged`. The marginal deviance recomputed at γ̂ above
    // is the honest witness: if it is non-finite the optimizer sat on the start and
    // this is not a converged fit. Step-halving in the PIRLS variants
    // (`pirls_solve_*`) recovers the grouseticks 3-crossed β=0 cold start, which
    // would otherwise overshoot into a ~1e30 weight regime and bail — so this
    // guard does not fire on that case. It remains the backstop for the genuinely
    // unrecoverable path: when halving is EXHAUSTED (`PIRLS_MAX_HALVINGS` reached)
    // PIRLS returns `(NaN, …)` and the deviance is non-finite, and BOBYQA can still
    // report `Converged` off an all-infeasible simplex (PRIMA maps `+inf` to the
    // FUNCMAX ceiling). Do NOT remove it.
    let ok = ok && final_deviance.is_finite();

    if !ok {
        return nan_fit(ws, target_indices, n_eval);
    }

    for j in 0..p {
        ws.betas[j] = ws.params[n_theta + j];
    }

    // Var(β̂): `Rx` inverts the β-information (Schur) directly — fast, but the
    // expected-information Schur complement assumes β–θ orthogonality (exact for
    // the Gaussian LMM, anticonservative for the GLMM where IRLS weights couple
    // β,θ). `Hessian` (default) sources Var(β̂) from the Hessian of the joint
    // Laplace deviance (glmer `use.hessian = TRUE`), the lme4 "correct" denom —
    // exact on every shape `derivative::supports_shape` accepts,
    // finite-difference elsewhere.
    let mut hessian_fallback = false;
    // Reset the θ-block SE each fit (the workspace is reused across fits). Only the
    // Hessian arm's `joint_hessian_cov` refills it; the Rx arm leaves it NaN.
    for v in ws.inference.theta_se[..n_theta].iter_mut() {
        *v = f64::NAN;
    }
    let joint_t_sq = match wald_se {
        WaldSe::Rx => {
            // Schur fill. The blocked layout reuses the per-block factors the
            // blocked PIRLS left in ws.pirls.a_blocks; the structured one reuses
            // the core+Schur factors left in ws.structured.{core_blocks,
            // schur_blk, coupling}; the packed one factors ws.packed.a.
            let inf_ok = match ws.layout {
                GlmmLayout::Blocked => blocked_schur_fill(ws, x, cluster_ids, n),
                GlmmLayout::Structured => structured_schur_fill(ws, x, cluster_ids, n),
                GlmmLayout::Packed => packed_schur_fill(ws, x, n),
            };
            if !inf_ok {
                return nan_fit(ws, target_indices, n_eval);
            }
            // Gamma carries lme4's σ̂² on the RX vcov (`vcov(use.hessian=FALSE)` =
            // σ̂²·Schur⁻¹); fixed-scale families σ̂²≡1. Computed off the converged
            // μ̂/û the pinned-γ̂ re-eval left. The Hessian arm is NOT scaled — the
            // dispersion enters its objective via `gamma_aic`, and its unscaled SE
            // is the oracle-settled match to `vcov(use.hessian=TRUE)`.
            let sigma_sq = crate::family::glmm_sigma_sq(
                family,
                &y[..n],
                &ws.pirls.prob[..n],
                &ws.pirls.u[..ws.k],
                ws.weighted.then(|| &ws.prior_w[..n]),
            );
            // Var(β̂)_jj from chol(Schur) forward-solve (mirrors fit_lmm's recovery).
            let sc = match ws.border.schur.as_ref().llt(faer::Side::Lower) {
                Ok(c) => c,
                Err(_) => return nan_fit(ws, target_indices, n_eval),
            };
            let lschur = sc.L();
            nan_fill_vcov(ws, p);
            for &tj in target_indices {
                let tj = tj as usize;
                // Forward-solve into reusable scratch; fwd_solve[i] is written before
                // it is read as fwd_solve[kk] (kk < i), so no per-target zero-fill is
                // needed.
                for i in 0..p {
                    let mut acc = if i == tj { 1.0 } else { 0.0 };
                    for kk in 0..i {
                        acc -= lschur[(i, kk)] * ws.inference.fwd_solve[kk];
                    }
                    ws.inference.fwd_solve[i] = acc / lschur[(i, i)];
                }
                let vd: f64 = ws.inference.fwd_solve[..p]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
                    * sigma_sq;
                ws.inference.var_diag[tj] = vd;
                ws.inference.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
                    ws.betas[tj] * ws.betas[tj] / vd
                } else {
                    f64::NAN
                };
                // Keep this target's column of L⁻¹: Schur⁻¹ = L⁻ᵀL⁻¹, so the
                // pairwise dots below are `vcov`'s off-diagonals — the same
                // arithmetic `vd` takes the norm of, not thrown away.
                for i in 0..p {
                    ws.inference.vcov_cols[(i, tj)] = ws.inference.fwd_solve[i];
                }
            }
            for &ta in target_indices {
                for &tb in target_indices {
                    let (a, b) = (ta as usize, tb as usize);
                    if b > a {
                        continue; // symmetric — fill both from the lower pair
                    }
                    let mut acc = 0.0;
                    for i in 0..p {
                        acc += ws.inference.vcov_cols[(i, a)] * ws.inference.vcov_cols[(i, b)];
                    }
                    ws.inference.vcov[(a, b)] = acc * sigma_sq;
                    ws.inference.vcov[(b, a)] = acc * sigma_sq;
                }
            }
            // Joint Wald-χ² via the lme helper (Schur is the β-information; σ̂²
            // divides W as in the LMM caller — 1 except Gamma).
            if target_indices.is_empty() {
                f64::NAN
            } else {
                crate::lmm::joint_wald_chi_sq(
                    ws.border.schur.as_ref(),
                    &ws.betas,
                    sigma_sq,
                    target_indices,
                    ws.inference.joint_k_inv.as_mut(),
                    ws.inference.joint_sigma_t_chol.as_mut(),
                    &mut ws.inference.joint_rhs,
                )
            }
        }
        WaldSe::Hessian => {
            // Joint-Hessian covariance into a LOCAL p×p Mat — NOT a ws field: the kernel
            // takes `&mut ws`, so `&mut ws.<field>` for out_cov would alias it. The
            // allocation is acceptable on this default path (the zero-alloc gate
            // pins the Rx warm path). The kernel re-evals PIRLS itself and its
            // RX fallback runs schur_fill internally, so skip the schur_fill above.
            let mut cov = Mat::<f64>::zeros(p, p);
            let status = joint_hessian_cov(ws, x, y, cluster_ids, extra_ids, p, n, &mut cov);
            // Double-failure sentinel: the kernel NaN-fills `cov` when BOTH the joint
            // Hessian and the RX fallback fail. Treat as a failed fit.
            if !cov[(0, 0)].is_finite() {
                return nan_fit(ws, target_indices, n_eval);
            }
            hessian_fallback = matches!(status, FdHessianStatus::NonPdFellBackToRx);
            // Marginal var/t² straight off the covariance diagonal.
            for &tj in target_indices {
                let tj = tj as usize;
                let vd = cov[(tj, tj)];
                ws.inference.var_diag[tj] = vd;
                ws.inference.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
                    ws.betas[tj] * ws.betas[tj] / vd
                } else {
                    f64::NAN
                };
            }
            // `cov` IS Cov(β̂) in full — keep the target block rather than only
            // the diagonal just read. The RX fallback (`NonPdFellBackToRx`) fills
            // `cov` through `rx_cov_into`, which forms a complete p×p inverse
            // too, so it carries a real vcov and is NOT NaN-filled here.
            //
            // Mirror the lower triangle instead of copying both cells: `cov`
            // comes from a solve against an identity, which leaves (a,b) and
            // (b,a) differing in the last bits. The matrix is symmetric
            // mathematically, and every other path builds it symmetric by
            // construction, so pick one value per pair rather than hand a
            // consumer an almost-symmetric vcov.
            nan_fill_vcov(ws, p);
            for &ta in target_indices {
                for &tb in target_indices {
                    let (a, b) = (ta as usize, tb as usize);
                    if b > a {
                        continue;
                    }
                    ws.inference.vcov[(a, b)] = cov[(a, b)];
                    ws.inference.vcov[(b, a)] = cov[(a, b)];
                }
            }
            // Joint Wald-χ²: `joint_wald_chi_sq` expects the β-INFORMATION (it inverts
            // and sub-blocks internally), so pass info = cov⁻¹. Write cov⁻¹ into the
            // now-free ws.border.schur (schur_fill was skipped on this arm) and reuse
            // the helper verbatim — same faer LLT-inverse idiom as `rx_cov_into`.
            if target_indices.is_empty() {
                f64::NAN
            } else {
                use faer::linalg::solvers::Solve;
                match cov.as_ref().llt(faer::Side::Lower) {
                    Ok(chol) => {
                        let mut inv = Mat::<f64>::identity(p, p);
                        chol.solve_in_place(inv.as_mut());
                        for a in 0..p {
                            for b in 0..p {
                                ws.border.schur[(a, b)] = inv[(a, b)];
                            }
                        }
                        crate::lmm::joint_wald_chi_sq(
                            ws.border.schur.as_ref(),
                            &ws.betas,
                            1.0,
                            target_indices,
                            ws.inference.joint_k_inv.as_mut(),
                            ws.inference.joint_sigma_t_chol.as_mut(),
                            &mut ws.inference.joint_rhs,
                        )
                    }
                    Err(_) => f64::NAN,
                }
            }
        }
    };

    // τ̂² = D̂[0][0] = (Λ̂Λ̂')[0][0]. No σ² (binomial). For lower-tri Λ_p stored
    // row-major (lam[r*q + c]), row 0 has only the (0,0) entry nonzero, so
    // D̂[0][0] = Σ_c Λ[0,c]² = Λ[0,0]² — the random-INTERCEPT variance.
    crate::lmm::primary_lambda(
        &ws.params[..n_theta],
        ws.groupings.primary_q,
        &mut ws.pirls.lam,
    );
    let q = ws.groupings.primary_q;
    let mut d00 = 0.0;
    for r in 0..q {
        d00 += ws.pirls.lam[r] * ws.pirls.lam[r];
    }
    GlmmFit {
        converged: !budget_exhausted,
        boundary_hit: if budget_exhausted {
            2
        } else {
            u8::from(pinned)
        },
        pinned_components: if budget_exhausted {
            0
        } else {
            pinned_components
        },
        n_eval,
        #[cfg(feature = "counters")]
        counters: ws.counters,
        tau_squared_hat: d00,
        joint_t_sq,
        hessian_fallback,
        deviance: final_deviance + nb_dev_term,
    }
}

/// NaN-fill `ws.inference.vcov` — the workspace is reused across fits, so every SE arm
/// clears it before writing its target block; entries outside that block stay
/// NaN, keeping `vcov` finite exactly where `var_diag` is.
fn nan_fill_vcov(ws: &mut GlmmWorkspace, p: usize) {
    for a in 0..p {
        for b in 0..p {
            ws.inference.vcov[(a, b)] = f64::NAN;
        }
    }
}

/// NaN-fill the inference outputs on a non-converged / Schur-failure fit, mirror
/// `fit_lmm`'s NaN-fill branch (boundary_hit = 2 = optimizer/Schur failure).
fn nan_fit(ws: &mut GlmmWorkspace, targets: &[u32], n_eval: usize) -> GlmmFit {
    for v in ws.betas.iter_mut() {
        *v = f64::NAN;
    }
    let p = ws.betas.len();
    nan_fill_vcov(ws, p);
    for &t in targets {
        ws.inference.var_diag[t as usize] = f64::NAN;
        ws.inference.t_sq[t as usize] = f64::NAN;
    }
    GlmmFit {
        converged: false,
        boundary_hit: 2,
        pinned_components: 0,
        n_eval,
        #[cfg(feature = "counters")]
        counters: ws.counters,
        tau_squared_hat: f64::NAN,
        joint_t_sq: f64::NAN,
        hessian_fallback: false,
        deviance: f64::INFINITY,
    }
}
