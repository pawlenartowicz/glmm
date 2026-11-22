//! GLMM kernel: PIRLS + Laplace/AGQ, glmer-faithful nAGQ=1. Covers
//! Binomial/Poisson/Gamma/negative-binomial with random effects; on NB and
//! Gamma the outer search carries the dispersion (`ln θ_NB`, `ln φ`) as one
//! more trailing coordinate alongside `[θ_RE | β]`, so β, θ_RE and the
//! dispersion come out of one fit — see `Dispersion::term` and `fit_glmm`'s `n_disp`
//! handling. Binomial, Poisson and Gamma Laplace fits with `n_θ ≥ 2` on the
//! blocked or structured layout search with projected BFGS
//! ([`gradient_route_plan`]), Gamma's `ψ = ln φ` included, and so do AGQ
//! fits of binomial logit or probit and Poisson log with one scalar random
//! intercept (the shapes the exact AGQ gradient covers) below 50 rows per
//! cluster, cold or from a warm θ₀ of at least 0.05, each with BOBYQA as the
//! polish and the fallback. Every other fit is optimized per
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
//! fixed-width packed `M` rows and a `k×k` `A`, dense or sparse-factored) — see
//! `se::blocked_schur_fill`/`se::structured_schur_fill`/`se::packed_schur_fill`
//! for their Schur complement fills. None of them materializes `Z` or `Λ_θ`
//! densely. Designs over `classify_design`'s NoZ envelope take the packed-row
//! backend through this same entry point.
//! All scratch for the three backends lives in `GlmmWorkspace`, allocated once
//! per (spec, max_n) shape. On the shapes the warm-path allocation tests cover
//! (`fit_glmm_warm_path_bounded_alloc` and
//! `fit_glmm_structured_warm_path_bounded_alloc`: logit, blocked and
//! structured) the warm path allocates a small fixed count per fit (each
//! `Bobyqa` is built once: the cold solvers with the workspace, the warm
//! stage-1 and the near-singular polish solver on their first use), except
//! that the record of non-finite evaluations grows on a fit that has one. The
//! free-φ Gamma `ExactProfile` route allocates in every fit (its dispersion
//! probe memo and its scaled-box copies); no test bounds the other shapes.
//!
//! No σ² scale: binomial and Poisson dispersion is fixed at 1, so D̂ = Λ̂Λ̂′
//! directly; NB's θ and Gamma's φ are outer-search coordinates.
//!
//! BOBYQA is Powell, M.J.D. (2009), *The BOBYQA algorithm for bound constrained
//! optimization without derivatives*, Cambridge report DAMTP 2009/NA06.

use crate::consts::{GAMMA_PHI_HI, GAMMA_PHI_LO};
use crate::standardize::ColumnScaling;
use crate::WaldSe;
use faer::MatRef;

use crate::lmm::POLISH_NEAR_SINGULAR;

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
/// 10-halving discipline: when a full step raises the penalized deviance
/// above the last accepted value, the u-step is halved and re-evaluated up to
/// this many times before the solve is declared failed. A trial whose raw η
/// passes the link's clamp is instead pulled back, in one evaluation, to the
/// largest power-of-two fraction of the step that is inside it, and that
/// counts as one halving. Exhausting it surfaces as
/// the module's `(NaN, NaN, NaN, false)` failure — the same terminal state a raw
/// overshoot reaches, but reached deliberately. Shared by all three PIRLS
/// variants (blocked / structured / packed).
///
/// 16, above lme4's 10: the packed-row layout's FD-Hessian central deviance eval
/// cold-seeds `û = 0` on every evaluation (see `deviance::laplace_deviance_at`), and on
/// a large-θ̂, many-crossed-grouping, large-count design that cold seed needs more
/// halvings to walk back to the mode than a warm-started fit does. Measured floor
/// on the packed large-θ̂ rung is 11 halvings; 16 was chosen for margin above that
/// floor, not tuned to the exact minimum. The blocked and structured solve steps
/// don't hit this regime and converge well inside the cap. The GLM IRLS
/// (`glm::evaluate_with_halving`) caps its β step-halving here too, with no
/// measurement of its own, so a change moves GLM convergence verdicts as well.
pub const PIRLS_MAX_HALVINGS: usize = 16;
/// Period-2 detector for the PIRLS iteration. An undamped step that
/// overshoots so that the iteration map has an eigenvalue at or below −1
/// settles into a 2-cycle around the mode. The same-point penalized deviance
/// then rises by less than the halving band per step, while the mixed stopping
/// value keeps alternating by more than its band, so the solve would run out
/// `PIRLS_MAX_ITERS`. Fisher scoring does this on a non-canonical link wherever
/// the observed curvature is more than twice the expected one; PIRLS takes the
/// Newton step there, which removes that cause, and the detector stays as a
/// guard for any step that cycles. After `PIRLS_OSC_TRIGGER`
/// consecutive sign flips of the mixed step, each at least `PIRLS_OSC_RATIO`
/// times the size of the one before, every later step of that solve is
/// halved (relaxation ω = ½), which maps an eigenvalue λ of the undamped step
/// to (1 + λ)/2. A solve that converges before the detector fires is
/// bit-identical. The GLM IRLS (`glm::glm_irls_fit`) runs the same detector,
/// with both constants, on its β step.
pub const PIRLS_OSC_RATIO: f64 = 0.8;
/// See [`PIRLS_OSC_RATIO`].
pub const PIRLS_OSC_TRIGGER: usize = 3;
/// The shapes whose stage-1 Profile solve is the EXACT Laplace β-profile and whose
/// outer search is therefore θ-only (`OuterSearch::ExactProfile`): Laplace, a data
/// term that is the plain deviance, and a PIRLS variant carrying the exact border —
/// the blocked path (`pirls_solve_blocked`, either link class), plus the structured
/// crossed/nested path (`pirls_solve_blocked_extras`); both step with the exact
/// curvature, so their factor is the one the border differentiates on either link
/// class. Gamma fits here too: at a fixed φ its objective is the Laplace
/// deviance on the weights `wᵢ/φ` plus a term in φ alone, so the β-profile is the
/// same exact one, and `ln φ` joins θ in the search as NB's `ln θ_NB` does. AGQ
/// keeps the joint search. So does the packed-row layout
/// (`GlmmLayout::Packed`, `workspace.rs`), whose PIRLS carries no exact border — `pirls_solve_packed` refuses a `ProfileExact` step
/// outright. Read by the workspace constructor and by the driver's
/// `debug_assert!` — the tests choose the route through `outer_search`, never
/// through this.
pub(crate) fn exact_profile_shape(nagq: u8, layout: GlmmLayout) -> bool {
    nagq == 1 && layout != GlmmLayout::Packed
}
/// Adaptive PIRLS exit on |Δ penalized-deviance|, relative to the objective
/// scale (lme4's pwrss discipline): converged when
/// `|Δpen| < PIRLS_TOL_REL · (1 + |pen|)`. The penalized deviance is O(n), so
/// this relative gate is an ABSOLUTE tolerance of `PIRLS_TOL_REL · n` in the
/// deviance scale — fine for the small/medium validation rungs (n ≲ 1500) but not
/// for VerbAgg (n=7584, the largest individual-Bernoulli binomial rung): at
/// a looser 1e-6 tolerance the absolute slack would be ~8e-3 in deviance, one Newton step
/// short of the quadratic-convergence cliff canonical links otherwise reach
/// "for free" — BOBYQA's stage-2
/// objective then carried that ~1 iteration of leftover curvature noise into
/// β̂, landing 4e-3 relative off both lme4 and MixedModels.jl (which agree
/// with each other to 8e-5) instead of the ~1e-4 every other binomial/poisson
/// rung achieves. This is a cliff, not a gradient: 1e-7 and 1e-8 still miss
/// VerbAgg's 1e-3 gate at ~4–5e-3 (the Newton step hasn't yet landed inside
/// the quadratic basin), 3e-9 clears it at 5e-4; 1e-9 keeps a ~5× margin.
/// The 1e-9 setting costs 2–4 extra inner iterations over a looser tolerance,
/// but only until each canonical-link solve reaches its own quadratic floor —
/// negligible on every rung's fit time (verified via `validation/grid/compare.R`
/// full-suite pass, no rung regressed).
pub const PIRLS_TOL_REL: f64 = 1e-9;
/// PIRLS exit for NON-canonical links (probit, cloglog, Gamma-log/inverse,
/// NB-log) — a decade looser than `PIRLS_TOL_REL`. PIRLS takes the Newton step
/// on these links too (`pirls::observed_weights_in_place`). The value rests on
/// the accuracy grid, not on a convergence argument: the outer objective is
/// only as smooth in (θ, β) as the inner solve is converged, and with
/// Fisher-scoring steps at this band the grid's wide probit, cloglog, Gamma/log
/// and NB/log cells (6–8 covariance parameters, or 30000 rows with a shallow
/// likelihood) stop 5e-5 to 5e-3 deviance units above glmmTMB's optimum, where
/// BOBYQA stalls on the leftover noise. With the Newton step at this band all
/// of them, the 30000-row Gamma/log cell that stalled at its start, and every
/// other non-canonical mixed cell of the grid land within dev_eps (4e-5) of
/// glmmTMB's deviance; a 1e-10 band measured no better there and cost 0–60%
/// more wall.
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
/// The fit path never pays it — BOBYQA objective evals run at `pirls_tol`, or
/// looser on `ExactProfile` stage 1 while the trust radius is large (the
/// radius-tied tolerance, `RHO_TOL_C`); the cost lands only on the SE passes. It equals `PIRLS_TOL_REL_NONCANON`, so
/// non-canonical links run their SE passes at their fit tolerance and canonical
/// links keep their tighter 1e-9. The exact-Hessian passes' mode re-solves read
/// `pirls_tol_fd` as well (`derivative.rs`, `assembled.rs`).
pub const PIRLS_TOL_REL_FD: f64 = 1e-8;
/// PIRLS exit tolerance for `family`: `PIRLS_TOL_REL` on the canonical links,
/// `PIRLS_TOL_REL_NONCANON` on the others — see those two for why they differ.
/// The GLM IRLS (`glm::evaluate_with_halving`) reads it too, as the band under
/// which a rise that outlasts every halving still ends converged, with no
/// measurement of its own, so a change moves GLM convergence verdicts as well.
pub(crate) fn pirls_tol(family: crate::spec::Family) -> f64 {
    if crate::family::is_canonical(family) {
        PIRLS_TOL_REL
    } else {
        PIRLS_TOL_REL_NONCANON
    }
}
/// PIRLS exit tolerance for the FD-Hessian SE evals: the tighter of the FD
/// ceiling and the family's own fit tolerance, so the stencil can only ever be
/// more converged than the fit, never less: 1e-9 on canonical links, 1e-8
/// otherwise.
/// `glmm::se::joint_hessian_cov` writes it into `pirls_tol_override` on both of
/// its stencil arms.
pub(crate) fn pirls_tol_fd(family: crate::spec::Family) -> f64 {
    PIRLS_TOL_REL_FD.min(pirls_tol(family))
}
/// Stage-1 PIRLS tolerance tied to the trust radius, on the `ExactProfile`
/// search. Each evaluation of that search is itself an inner PIRLS solve, and
/// BOBYQA only needs the objective to an accuracy small next to what its model
/// resolves at the current radius, of order ρ² (inexact derivative-free
/// optimization, Ehrhardt & Roberts 2021, *J. Math. Imaging Vis.*,
/// doi:10.1007/s10851-021-01020-8: an inner accuracy of order Δ²). So an
/// evaluation's PIRLS exit tolerance is `clamp(RHO_TOL_C·ρ², pirls_tol,
/// RHO_TOL_MAX)`, relative as `pirls_tol` is, with ρ BOBYQA's own resolution
/// (`bobyqa::TrustRadius::rho`, handed to the objective by
/// `Bobyqa::minimize_with_radius`), and `pirls_tol` once ρ ≤
/// `RHO_TOL_FULL_FACTOR`·`GLMM_RHO_END`, so the last rungs of the ρ ladder
/// compare full-tolerance values. The search's first evaluation is solved at
/// `pirls_tol` as well: it is the point the warm-start guard solves and hands
/// over. The incumbent is re-solved at `pirls_tol` after the search (the
/// stage-1 block of `fit_glmm`), so β̂ and the deviance never rest on a loose
/// solve.
///
/// Measured 2026-10-02 (x86, instruction counts) on the loop-tier
/// benchmark shapes against the fixed `pirls_tol`: 5 to 22% fewer instructions
/// per exact-route fit from a cold start, 4 to 21% from a warm one, with the
/// evaluation counts unchanged (0.98 to 1.02) and every deviance within
/// 3.6e-10 of the fixed-tolerance fit's. On aarch64, C = 1e-4 saves about half
/// as much, and a cap of 1e-4 instead of 1e-5 ended far warm starts at worse
/// optima (up to 1.6e-2 deviance), so the cap, not C, is the limit.
pub(crate) const RHO_TOL_C: f64 = 1e-2;
/// Upper cap of the radius-tied stage-1 PIRLS tolerance (relative, as
/// `pirls_tol`). See [`RHO_TOL_C`].
pub(crate) const RHO_TOL_MAX: f64 = 1e-5;
/// The radius-tied stage-1 tolerance is `pirls_tol` once ρ is within this
/// factor of `GLMM_RHO_END`. See [`RHO_TOL_C`].
pub(crate) const RHO_TOL_FULL_FACTOR: f64 = 10.0;
/// Largest estimated rounding error of the reported Laplace deviance, in
/// deviance units, at which a fit is reported without
/// `Note::PrecisionLimited`; the LMM's profiled deviance uses the same bound
/// (`LmmResultView::diagnostics`). The estimate is `ε·(|D| + tr_x)` (`fit_glmm`):
/// - `ε·|D|`: the deviance is a sum of non-negative row terms, each correct to
///   about ε relative, so its absolute error is about `ε·|D|`. On a model that
///   misses a large effect at counts near `e³³`, `D` is 1e15 and this is 0.3.
/// - `ε·tr_x`: with more than one grouping factor, intercept columns of two
///   factors both add up to the all-ones column, so `M = ZΛ` has a null
///   direction and `A = I + M'WM` an eigenvalue of exactly 1. The factor finds
///   that pivot as a difference of entries of the size `θ²·Σwᵢ` of the extra
///   factors' columns, so `log|A|` carries an absolute error of about ε times
///   that size. `tr_x` ([`extra_ztwz_trace`]) is the trace of `M'WM` over the
///   extra factors' columns, an upper bound on it.
///
/// A rounding noise of δ deviance units moves θ̂ by about `√δ` of its standard
/// error, because the deviance rises by 1 at one standard error; 1e-2 is a
/// tenth of one. The search sees a noise several times the estimate: on a
/// 12-cluster Poisson fit missing a crossed effect, σ̂ moved 0.13 standard
/// errors at counts `e²⁹` (estimate 5e-3) and 0.8 at `e³¹` (4e-2). The PIRLS
/// exit band (relative `pirls_tol`, 1e6 deviance units at `D` = 1e15) is not
/// the limit: PIRLS converges past it to the rounding floor, and a band of
/// 1e-13 moved the same fits as far.
pub(crate) const DEVIANCE_FLOOR_MAX: f64 = 1e-2;
/// Distance in `ln φ` below which a fitted φ̂ counts as on the box bound and the
/// fit is refused (`fit_glmm`); BOBYQA projects onto the bound, so an incumbent
/// that reached it sits there to round-off.
const GAMMA_PHI_BOUND_TOL: f64 = 1e-6;

/// The edge of the Gamma coordinate's box a fitted `ln φ̂` sits on (within
/// [`GAMMA_PHI_BOUND_TOL`]), as φ: [`GAMMA_PHI_LO`] or [`GAMMA_PHI_HI`]; `None`
/// inside the box. An incumbent on an edge describes the box, not a maximum of
/// the likelihood, and `fit_glmm` refuses the fit.
pub(crate) fn gamma_phi_box_edge(ln_phi: f64) -> Option<f64> {
    if ln_phi <= GAMMA_PHI_LO.ln() + GAMMA_PHI_BOUND_TOL {
        Some(GAMMA_PHI_LO)
    } else if ln_phi >= GAMMA_PHI_HI.ln() - GAMMA_PHI_BOUND_TOL {
        Some(GAMMA_PHI_HI)
    } else {
        None
    }
}
/// Base FD step for `joint_hessian_cov`'s joint-deviance Hessian. It is applied
/// ASYMMETRICALLY across the joint (θ, β) vector: `h_θ = FD_STEP_BASE` absolute on
/// the θ block, `h_β = FD_STEP_BASE·max(1, |β̂_k|)` relative on the β block.
/// β enters through η = Xβ and wants relative stepping; θ does not — scaling h_θ
/// with the random-effect SD widens the window exactly where the deviance profile
/// in θ flattens, so the central second difference's O(h²) truncation error grows
/// as θ̂². Measured on the corpus: absolute θ stepping divides the
/// distance to the h→0 limit by θ̂² to within 6% on all seven scalar-RE rungs whose
/// θ̂ exceeds 1, at nAGQ = 1 and 7 and 11 alike (θ̂ 1.13 → 5.16, a 20× range in
/// θ̂²), on both `se_hessian` and the θ-block SEs. The step
/// construction, and the toenail evidence behind it, is at `se.rs`'s
/// `ws.fd.fd_steps` loop.
///
/// The BASE VALUE 1e-2 is pinned by the curated sweep, not just by the
/// fixture: the Hessian is step-invariant over h ∈
/// [1e-4, 1e-1] on the committed fixture — but NOT on every curated rung:
/// h = 1e-3 blows the validation se_hess gates on the noise side (sim_gamma 1e-2,
/// cbpp_probit 2e-3 vs the 1e-3 band) while 1e-2 holds them at ~1e-4. Both of
/// those rungs have θ̂ < 1, where `max(1, |θ̂|)` is exactly 1, so absolute θ
/// stepping leaves h_θ on them as it is. Independently, the measured noise knee in θ is
/// at h_θ ≈ 2.5e-4, so 1e-2 keeps a 40× margin on the truncation side.
/// The packed-row layout needs the opposite trade and carries its own
/// `se::SPARSE_FD_STEP_REL` (1e-4, landing on the weighted sparse-Z Gamma
/// golden's FD-step plateau — 1e-3 biases se(β₀) high there) — calibrated
/// separately, do not fold the two constants together. Absolute θ stepping does
/// NOT transfer to it: that constant is calibrated on the noise side, so removing the
/// θ scaling there would push h_θ further into noise, not out of it.
///
/// **The `_BASE` / `_REL` split in the two names is deliberate.** This one is
/// `_BASE` because the asymmetry above makes "relative" false of the θ block;
/// `SPARSE_FD_STEP_REL` keeps its suffix because it really is applied relatively
/// on every coordinate, θ included. If the packed step ever takes the same
/// asymmetry, rename it in the same change, or the suffix starts lying there
/// instead.
pub const FD_STEP_BASE: f64 = 1e-2;

/// Per-fit GLMM result (mirrors `LmmFit`; no σ²: NB's θ and Gamma's φ are
/// outer-search coordinates).
pub struct GlmmFit {
    /// `true` iff the outer BOBYQA converged AND the pinned-γ̂ re-eval deviance is
    /// finite. `false` on two distinct exits: an outer search that exhausted its
    /// evaluation budget (`Status::MaxFunReached`) still reports its incumbent
    /// β̂, SEs and deviance here, with `boundary_hit = 2` and
    /// `pinned_components` empty; every other non-convergence (BOBYQA failure,
    /// the degenerate-fit guard) NaN-fills every inference field instead (see
    /// `nan_fit`). A standard-error failure does not touch it: see
    /// `se_unavailable`.
    pub converged: bool,
    /// 0 = interior, 1 = ≥1 diagonal θ pinned (converged), 2 = optimizer
    /// failure or a budget-exhausted exit (non-converged).
    pub boundary_hit: u8,
    /// Bit k set iff diagonal variance component k pinned (order
    /// [intercept, slope_0, …, extra_1, …]). Mirrors `LmmFit.pinned_components`
    /// (u64 mask — see there for the over-envelope component-count rationale).
    pub pinned_components: u64,
    /// Every evaluation of every search and probe that counts (see
    /// `fit_glmm`).
    pub n_eval: usize,
    /// Observation-only evaluation counters for this fit. Gated because
    /// `GlmmFit` is re-exported `pub` under `loop_advanced`: with `counters`
    /// off, that tier's surface must be unchanged.
    #[cfg(feature = "counters")]
    pub counters: crate::counters::EvalCounters,
    /// Joint Wald-χ² over `target_indices` (NaN when empty / non-converged).
    pub joint_t_sq: f64,
    /// Set iff the fit ran `WaldSe::Hessian` and `joint_hessian_cov` fell back to the
    /// RX/Schur block (non-PD joint Hessian / non-finite perturbed deviance) — its
    /// `NonPdFellBackToRx` status. Always `false` under `WaldSe::Rx`.
    pub hessian_fallback: bool,
    /// Set iff this fit's shape routes `ExactProfile`, that route ended not
    /// converged, and the fit reran once on `PqlThenJoint` — see
    /// [`crate::Note::ExactProfileFallback`]. Always `false` on `Joint` and
    /// `PqlThenJoint` shapes, which have no fallback of their own.
    pub exact_profile_fallback: bool,
    /// The gradient search handed its incumbent to the BOBYQA route — see
    /// [`crate::Note::GradientSearchFallback`].
    pub gradient_search_fallback: bool,
    /// Set iff the search reached an endpoint but no fixed-effect covariance
    /// could be formed there: the Rx Schur complement is not positive definite
    /// to rounding (under `WaldSe::Hessian`, after the joint Hessian failed
    /// too). β̂, θ̂, the deviance and `converged` stand as the search left them;
    /// `var_diag`, `t_sq`, `vcov` and `joint_t_sq` are NaN — see
    /// [`crate::Note::SeUnavailable`].
    pub se_unavailable: bool,
    /// Set iff the estimated rounding error of the reported deviance exceeds
    /// `DEVIANCE_FLOOR_MAX`, so the search could not resolve θ̂ to a tenth
    /// of its standard error — see [`crate::Note::PrecisionLimited`].
    /// `converged` and every estimate stand as the search left them.
    pub precision_limited: bool,
    /// Set iff the fixed-effects GLM the search starts from stopped on
    /// separation, so the search never ran — see [`crate::Note::Separated`].
    /// β̂ is that GLM's last step; every other output is NaN.
    pub separated: bool,
    /// Set iff the fit was refused because a free Gamma `ln φ̂` ended on an
    /// edge of the `[GAMMA_PHI_LO, GAMMA_PHI_HI]` box: that edge, on the
    /// internal weights `ŵ`. The fit is then `nan_fit`'s. `None` otherwise;
    /// see [`crate::Note::DispersionAtLimit`].
    pub phi_at_limit: Option<f64>,
    /// Minimized marginal Laplace deviance at the pinned γ̂ (`d(y,ũ)+‖ũ‖²+log|A|`,
    /// or the AGQ deviance when `nagq>1`), reported as `Fit::deviance`.
    /// `f64::INFINITY` on non-convergence. On NB and Gamma it carries the
    /// dispersion term on top (`−2·saturated_loglik(θ̂)` on NB,
    /// `family::gamma_dispersion_term` on Gamma), so the reported value is
    /// exactly `−2·logLik`.
    pub deviance: f64,
    /// Why the fit did not converge: `StopReason::SearchBudget` on the
    /// budget exit, `StopReason::NoEndPoint` from `nan_fit`,
    /// `StopReason::IrlsDiverged` from `separated_fit`. `None` when
    /// `converged`.
    pub(crate) stop: Option<crate::fit::StopReason>,
}

mod agq;
mod agq_fast;
mod agq_gradient;
mod assembled;
mod bobyqa_route;
mod derivative;
mod deviance;
mod grad_route;
mod pin;
mod pirls;
mod se;
mod start;
mod workspace;

#[cfg(test)]
pub(crate) use deviance::glmm_laplace_deviance;
// `derivative` is private to `glmm`; the fit-module tests read the status
// through this re-export.
#[cfg(test)]
pub(crate) use derivative::DerivStatus;
// Both exact Hessian engines, reachable from the test module that drives the
// validation corpus's own datasets through them side by side. Production
// reaches them through `se::joint_hessian_cov` alone. Every caller of these
// four is a `#[cfg(feature = "formula")]` test in `fit::glmm_tests`, so the
// re-export carries that gate too — otherwise `--no-default-features` warns
// on an import nothing left standing can use.
#[cfg(all(test, feature = "formula"))]
pub(crate) use assembled::{assembly_routes, gradient_f64, joint_hessian, packed_gradient};
// The unsymmetrized column pass, read by the corpus drivers in
// `fit::glmm_tests` and by the both-layouts cross-check in `sparse::tests`,
// which carries no `formula` gate.
#[cfg(test)]
pub(crate) use assembled::joint_hessian_columns;
// Same formula-only caller as the blocks above.
#[cfg(all(test, feature = "formula"))]
pub(crate) use derivative::{
    laplace_gradient, laplace_hessian, supports_shape as supports_exact_shape, MAX_DUAL_N,
};
// The AGQ family list, which `fit::common::agq_shape` reads.
pub(crate) use agq::agq_family;
pub(crate) use pirls::{residual_border, residual_rhs};
pub(crate) use se::condvar_blocks;
#[cfg(test)]
pub(crate) use se::joint_hessian_cov;
// The FD stencils and the step rule: `se.rs` is their only production
// reader, the FD-margin measurement their only cross-module one.
#[cfg(all(test, feature = "formula"))]
pub(crate) use se::{fd_mixed_diff, fd_second_diff, fd_step};
pub(crate) use workspace::fill_packed_cols;
pub(crate) use workspace::StructuredSchur;
#[cfg(test)]
pub(crate) use workspace::{glmm_block_chol, glmm_block_solve};
pub use workspace::{GlmmWorkspace, OuterSearch};

use bobyqa_route::joint_recheck_band;
use deviance::EvalScratch;

// `pub(crate)` so `src/scalar.rs`'s tail-methods test can reuse
// `glmm_extras_q1_dataset` instead of duplicating a structured fixture.
#[cfg(test)]
pub(crate) mod tests;

/// Outcome of the joint-Hessian fixed-effect covariance (either arm — exact
/// wherever an exact rung answers (`se::joint_hessian_cov`), finite-difference
/// otherwise).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdHessianStatus {
    /// The joint-deviance Hessian was PD and every perturbed eval finite; the
    /// returned covariance is `2·(H_dev⁻¹)_ββ` (lme4 `vcov(use.hessian = TRUE)`).
    Ok,
    /// The joint Hessian was non-PD (or a perturbed deviance was non-finite — the
    /// few-cluster failure mode); the returned covariance is the RX/Schur block.
    NonPdFellBackToRx,
}

/// The gradient route's rule: which optimizer `fit_glmm`'s outer search runs
/// on a cold and on a warm fit, from the shape alone.
///
/// Projected BFGS over `[θ | β]` on the fixed-β Laplace objective, with the
/// gradient assembled off each value evaluation's own mode state
/// ([`assembled::gradient_at_mode`]), when all of these hold:
///
/// 1. Laplace (`nagq == 1`), binomial (any link), Poisson or Gamma (log and
///    inverse link): no dispersion coordinate, or Gamma's `ln φ`, which the
///    search carries as a trailing coordinate (NB's `ln θ_NB` changes the
///    variance function, and the gradient has no entry for it);
/// 2. the blocked layout, or the structured one within the dual tail bound
///    (the shapes the assembly serves);
/// 3. `n_θ ≥ 2`. One-θ fits stay on `ExactProfile`, which needs 11 to 24
///    evaluations there; 30 of 46 such cells were slower with BFGS.
///
/// Without a caller's hint, that search runs on β whitened by the β
/// information at its start, cold or warm, with every β̃ start entry seeded
/// at that whitening's curvature (`grad_route::gradient_route`,
/// `GradSearch::start_curvature`). Measured 2026-10-20 against the design
/// whitening and the identity start, BFGS search time (geometric mean) on
/// generated cells with 30 levels at 5 to 500 rows each was 0.52 cold and
/// 0.74 from the generating values on Poisson, 0.75 and 0.89 on binomial,
/// 0.91 and 0.95 on Gamma; on the grid's cells with 2+ θ 0.84 to 1.02 cold
/// and 0.93 to 0.99 from the generating values, no endpoint more than 1e-6
/// above the BOBYQA optimum.
///
/// And projected BFGS over `[λ | β]` on the fixed-β AGQ objective, with the
/// exact AGQ gradient off each value evaluation's modes
/// ([`agq_gradient::assemble`]), when:
///
/// 4. `nagq > 1` on a shape [`agq_gradient::covers_shape`] accepts: binomial
///    logit or probit, or Poisson log, with one scalar random intercept and no
///    extra grouping on the blocked layout. Its alternative is the joint
///    BOBYQA search (`OuterSearch::Joint`), not `ExactProfile`, so one θ is
///    enough here. Both start kinds: `fit_glmm` keeps a fit on BOBYQA when
///    its clusters average `GRAD_WARM_SCALAR_ROWS_MAX` (50) rows or more, and
///    a warm fit when a diagonal θ₀ of its start is below
///    `GRAD_WARM_AGQ_THETA_MIN` (0.05).
///
/// Every other shape keeps its `OuterSearch` route on both start kinds. On
/// the Laplace shapes a warm fit runs BFGS too, at any rows per level.
/// Measured 2026-10-20 from the generating values against warm BOBYQA: the
/// grid's 360 warm cells with 2+ θ at 0.45 of its fit time (geometric
/// mean), none above 1.25; `pois_cross6_g3000p100` at 0.70 (locked clock);
/// and on generated cells at 50 to 500 rows per level, summed fit time 0.53
/// (binomial), 0.62 (Poisson) and 0.28 (Gamma) of it.
///
/// Measured against the BOBYQA route, every endpoint at or below the BOBYQA
/// optimum within `dev_eps`. Binomial and Poisson (shadow search against
/// production in one process, 2026-10-06/07): summed search time 0.46 of it
/// on 94 grid cells and 0.22 on the stress set, no stress cell slower. Gamma
/// (locked clock, 2026-10-09): summed fit time 0.29 of it on the 88 routed
/// log-link grid cells and 0.21 on the 14 inverse-link ones, every cell
/// faster. Covered AGQ shapes below 50 rows per level (locked clock,
/// 2026-10-10, whole fits against the joint BOBYQA search on the grid's and
/// the stress set's cells): geometric mean 0.783 of its time over 25 cold
/// rows (0.51 to 1.10) and 0.825 over 38 warm ones from θ₀ ≥ 0.05 (at most
/// 1.23). A cold cell at 500 rows per level was 1.27x, so 50 rows or more
/// keep BOBYQA.
pub(crate) fn gradient_route_plan(
    family: crate::Family,
    nagq: u8,
    layout: GlmmLayout,
    g: &crate::lmm::LmmGroupings,
    n_theta: usize,
) -> crate::optim::OuterOptimizer {
    use crate::optim::OuterOptimizer;
    let laplace = nagq == 1
        && matches!(
            family,
            crate::Family::Binomial { .. }
                | crate::Family::Poisson { .. }
                | crate::Family::Gamma { .. }
        )
        && matches!(layout, GlmmLayout::Blocked | GlmmLayout::Structured)
        && derivative::supports_shape(layout, g)
        && n_theta >= 2;
    if laplace || agq_gradient::covers_shape(family, nagq, layout, g) {
        OuterOptimizer::Bfgs
    } else {
        OuterOptimizer::Bobyqa
    }
}

/// The curvature hint (`loop_advanced::curvature_hint`) after a `fit_glmm` on
/// `ws`: the last gradient search's final BFGS inverse-Hessian estimate over
/// `[θ | β̃]`, and `[θ | β̃ | ln φ]` on a free Gamma dispersion, on a Laplace
/// fit or a covered AGQ one ([`gradient_route_plan`]). β̃ is always the
/// design whitening's (`BetaScale::set`), the coordinates a hinted fit
/// searches in, which depend on the design alone. A search that ran on β̃
/// whitened by the β information (`ws.grad_beta_info`) has its estimate
/// mapped there: with `β̃_set = T·β̃_info` the β̃ block becomes `T·H·Tᵀ`, the
/// cross blocks `T·H` and `H·Tᵀ`, θ and ψ unchanged. `None` when that fit did
/// not search with BFGS, never updated its estimate, or left one (after the
/// map) that is not positive definite. Free: no evaluation runs.
pub(crate) fn glmm_curvature_hint(ws: &GlmmWorkspace) -> Option<crate::optim::CurvatureHint> {
    if !ws.grad_hint_ready {
        return None;
    }
    // The search ran on the free θ, so its estimate's β̃ block starts there.
    let (nt, p) = (ws.groupings.n_free_theta(), ws.p);
    let m = grad_search_dim(ws.family, nt, p, ws.gamma_phi_held.is_some());
    let h = ws.bfgs.inv_hessian();
    if !ws.grad_beta_info {
        return crate::optim::CurvatureHint::from_inverse(m, h, ws.search_units);
    }
    let t = &ws.grad_beta_map[..p * p];
    // M·H with M = diag(I, T, I): the β̃ rows mix.
    let mut a = h.to_vec();
    for r in 0..p {
        for c in 0..m {
            a[(nt + r) * m + c] = (0..p).map(|k| t[r * p + k] * h[(nt + k) * m + c]).sum();
        }
    }
    // (M·H)·Mᵀ: the β̃ columns mix.
    let mut mapped = a.clone();
    for r in 0..m {
        for c in 0..p {
            mapped[r * m + nt + c] = (0..p).map(|k| a[r * m + nt + k] * t[c * p + k]).sum();
        }
    }
    crate::optim::CurvatureHint::from_inverse(m, &mapped, ws.search_units)
}

/// The gradient search's dimension: `[θ | β̃]`, and `[θ | β̃ | ln φ]` on a
/// Gamma fit whose dispersion is not held, with `n_theta` the free θ (the
/// search holds the structural ones at zero). The curvature hint the search
/// leaves (`glmm_curvature_hint`) and the one `fit::core::fit_on_hinted`
/// accepts are this size.
pub(crate) fn grad_search_dim(
    family: crate::Family,
    n_theta: usize,
    p: usize,
    phi_held: bool,
) -> usize {
    let free_phi = matches!(family, crate::Family::Gamma { .. }) && !phi_held;
    n_theta + p + usize::from(free_phi)
}

/// The saddle walk of an `ExactProfile` endpoint (stage 1's saddle check in
/// `bobyqa_route::stage1` explains it): each nonzero `Λ` column of `x` below
/// `POLISH_NEAR_SINGULAR` is doubled while the deviance falls. `lower(x)`
/// evaluates the doubled point and says whether it was kept; a try it rejects
/// is halved back. Every try is one evaluation, counted in the second value;
/// the first is whether a try was kept.
fn saddle_walk(
    g: &crate::lmm::LmmGroupings,
    x: &mut [f64],
    mut lower: impl FnMut(&[f64]) -> bool,
) -> (bool, usize) {
    let (mut moved, mut tries) = (false, 0usize);
    for col in g.lambda_columns() {
        loop {
            let norm_sq: f64 = x[col.clone()].iter().map(|v| v * v).sum();
            if norm_sq == 0.0 || 4.0 * norm_sq > POLISH_NEAR_SINGULAR * POLISH_NEAR_SINGULAR {
                break;
            }
            for v in x[col.clone()].iter_mut() {
                *v *= 2.0;
            }
            tries += 1;
            if lower(x) {
                moved = true;
            } else {
                for v in x[col.clone()].iter_mut() {
                    *v *= 0.5;
                }
                break;
            }
        }
    }
    (moved, tries)
}

/// The dispersion-bound probe (the joint search's check in
/// `bobyqa_route::joint_passes` explains it): `point[slot]` is set to `bound` and `eval` solves the point,
/// every other coordinate held. A value below `best` keeps the point and is
/// returned; otherwise the slot is restored and `None` returned. The bound is
/// Gamma's lower bound of `ln φ` and NB's upper bound of `ln θ_NB`.
///
/// The three sites count the solve differently, each in its own `eval`:
/// - stage 1 solves through `Stage1Objective::eval`: one evaluation in `n_eval`, its
///   PIRLS exhaustion counted, and stage 1's finite count;
/// - the gradient route counts one evaluation in `n_eval` and restores the
///   PIRLS exhaustion count, so its exhaustion is not counted;
/// - the joint search solves with its own counters and leaves `n_eval`
///   unchanged.
///
/// One rule at all three would move the reported `n_eval` and
/// `Note::PirlsExhausted` on fits that reach a probe.
fn dispersion_bound_probe(
    point: &mut [f64],
    slot: usize,
    bound: f64,
    best: f64,
    mut eval: impl FnMut(&[f64]) -> Option<f64>,
) -> Option<f64> {
    let old = point[slot];
    point[slot] = bound;
    match eval(point) {
        Some(f) if f < best => Some(f),
        _ => {
            point[slot] = old;
            None
        }
    }
}

/// The dispersion coordinate of one fit: the (θ_NB, φ) pair an evaluation
/// runs at, and the dispersion term the outer objective adds.
#[derive(Clone, Copy)]
struct Dispersion<'a> {
    is_gamma: bool,
    held_phi: Option<f64>,
    nb_theta: f64,
    /// 1 when the point carries a dispersion coordinate, else 0.
    n_disp: usize,
    /// `Σ ln yᵢ` on Gamma, 0 elsewhere.
    gamma_sum_ln_y: f64,
    nb_groups: &'a [(f64, f64)],
    weighted: bool,
    prior_w: &'a [f64],
    n: usize,
}

impl Dispersion<'_> {
    /// The (θ_NB, φ) pair one evaluation runs at, off the trailing coordinate
    /// `c = point[slot]`, and the coordinate the dispersion term is read at: NB
    /// reads θ_NB = e^c at φ = 1, Gamma reads φ = e^c, and a held Gamma φ reads
    /// neither — the coordinate is inert and the objective is the fixed-φ
    /// Laplace deviance (its dispersion term a constant, restored in the
    /// reported deviance). Without a dispersion coordinate (`n_disp == 0`) it is
    /// `((nb_theta, 1.0), None)`, and `point[slot]` is not read.
    fn at(&self, point: &[f64], slot: usize) -> ((f64, f64), Option<f64>) {
        let Dispersion {
            is_gamma,
            held_phi,
            nb_theta,
            n_disp,
            ..
        } = *self;
        if n_disp == 0 {
            return ((nb_theta, 1.0), None);
        }
        let c = point[slot];
        match (is_gamma, held_phi) {
            (true, Some(v)) => ((nb_theta, v), None),
            (true, None) => ((nb_theta, c.exp()), Some(c)),
            _ => ((c.exp(), 1.0), Some(c)),
        }
    }

    /// The dispersion term at the coordinate `ln_theta` (`ln θ_NB` or `ln φ`):
    /// NB's `−2·nb_profile_loglik(y, y, θ_NB, w)`, Gamma's
    /// `family::gamma_dispersion_term`. See `fit_glmm`, where it is built.
    fn term(&self, ln_theta: f64) -> f64 {
        let Dispersion {
            is_gamma,
            gamma_sum_ln_y,
            nb_groups,
            weighted,
            prior_w,
            n,
            ..
        } = *self;
        if is_gamma {
            return crate::family::gamma_dispersion_term(
                ln_theta,
                weighted.then_some(&prior_w[..n]),
                n,
                gamma_sum_ln_y,
            );
        }
        // `nb_profile_loglik(y, y, θ, w)` off the per-fit count groups.
        -2.0 * crate::fit::nb_saturated_loglik_grouped(nb_groups, ln_theta.exp())
    }
}

/// The read-only per-fit values `fit_glmm`'s search stages share, built once
/// after the workspace is borrowed apart.
#[derive(Clone, Copy)]
struct SearchCtx<'a> {
    data: FitData<'a>,
    disp: Dispersion<'a>,
    groupings: &'a crate::lmm::LmmGroupings,
    layout: GlmmLayout,
    prior_w: &'a [f64],
    x: MatRef<'a, f64>,
    cluster_ids: &'a [u32],
    /// θ entries a grouping's terms hold at zero.
    structural: &'a [usize],
    family: crate::Family,
    nagq: u8,
    nb_theta: f64,
    k: usize,
    p: usize,
    n: usize,
    n_theta: usize,
    n_free: usize,
    n_disp: usize,
    /// The point's trailing `ln θ_NB` / `ln φ` index.
    nb_col: usize,
    total_dim: usize,
    total_free: usize,
    search_div: f64,
    is_gamma: bool,
    held_phi: Option<f64>,
    agq_finish: bool,
    gamma_sum_ln_y: f64,
    ln_phi_plug: f64,
    gamma_phi_start: Option<f64>,
}

/// The workspace fields `fit_glmm`'s search stages write, borrowed apart
/// once. Each stage takes a [`SearchState::reborrow`] and destructures it
/// into the names its body uses.
struct SearchState<'s> {
    solver: &'s mut bobyqa::Bobyqa,
    config: &'s mut bobyqa::Config,
    solver_polish_wide: &'s mut Option<bobyqa::Bobyqa>,
    solver_stage1_polish: &'s mut Option<bobyqa::Bobyqa>,
    config_stage1: &'s mut bobyqa::Config,
    params: &'s mut Vec<f64>,
    params_stage1: &'s mut Vec<f64>,
    lower_stage1: &'s mut Vec<f64>,
    upper_stage1: &'s mut Vec<f64>,
    lower: &'s mut Vec<f64>,
    upper: &'s mut Vec<f64>,
    beta_scale: &'s mut workspace::BetaScale,
    start_point: &'s mut Vec<f64>,
    u_seed: &'s mut Vec<f64>,
    beta_seed: &'s mut Vec<f64>,
    guard_u: &'s mut Vec<f64>,
    guard_beta: &'s mut Vec<f64>,
    pirls_exhausted: &'s mut u32,
    non_finite_points: &'s mut Vec<f64>,
    counters: &'s mut crate::counters::EvalCounters,
    bfgs: &'s mut crate::optim::bfgs::BfgsState,
    grad_cfg: &'s mut crate::optim::bfgs::BfgsConfig,
    grad_report: &'s mut Option<(crate::optim::bfgs::BfgsStop, usize, f64, u8)>,
    grad_hint_ready: &'s mut bool,
    beta_info_scaling: &'s mut bool,
    grad_beta_info: &'s mut bool,
    grad_beta_map: &'s mut Vec<f64>,
    grad_asm: &'s mut Option<assembled::AssemblyBufs<f64>>,
    grad_scratch: &'s mut Vec<f64>,
    gradient_search_fallback: &'s mut bool,
    agq_gd: &'s mut Option<Box<agq_gradient::AgqGd>>,
}

impl SearchState<'_> {
    /// Every field borrowed again for one stage.
    fn reborrow(&mut self) -> SearchState<'_> {
        SearchState {
            solver: &mut *self.solver,
            config: &mut *self.config,
            solver_polish_wide: &mut *self.solver_polish_wide,
            solver_stage1_polish: &mut *self.solver_stage1_polish,
            config_stage1: &mut *self.config_stage1,
            params: &mut *self.params,
            params_stage1: &mut *self.params_stage1,
            lower_stage1: &mut *self.lower_stage1,
            upper_stage1: &mut *self.upper_stage1,
            lower: &mut *self.lower,
            upper: &mut *self.upper,
            beta_scale: &mut *self.beta_scale,
            start_point: &mut *self.start_point,
            u_seed: &mut *self.u_seed,
            beta_seed: &mut *self.beta_seed,
            guard_u: &mut *self.guard_u,
            guard_beta: &mut *self.guard_beta,
            pirls_exhausted: &mut *self.pirls_exhausted,
            non_finite_points: &mut *self.non_finite_points,
            counters: &mut *self.counters,
            bfgs: &mut *self.bfgs,
            grad_cfg: &mut *self.grad_cfg,
            grad_report: &mut *self.grad_report,
            grad_hint_ready: &mut *self.grad_hint_ready,
            beta_info_scaling: &mut *self.beta_info_scaling,
            grad_beta_info: &mut *self.grad_beta_info,
            grad_beta_map: &mut *self.grad_beta_map,
            grad_asm: &mut *self.grad_asm,
            grad_scratch: &mut *self.grad_scratch,
            gradient_search_fallback: &mut *self.gradient_search_fallback,
            agq_gd: &mut *self.agq_gd,
        }
    }
}

/// One outer search's result: whether it ended at a reported point, its
/// evaluations, whether its budget ran out, and its objective there.
#[derive(Clone, Copy)]
struct RouteResult {
    ok: bool,
    n_eval: usize,
    budget_exhausted: bool,
    obj: f64,
}

/// Fit the clustered-logistic GLMM. `fill_packed_cols` must already have run (a
/// no-op off the packed layout, whose buffers are zero-length) for this
/// (X, ids, N). `beta_start` = spec.effect_sizes. Writes β̂/Var/z² into
/// `ws.betas`, `ws.inference.var_diag`, `ws.inference.t_sq`; returns the GlmmFit summary.
///
/// Convention: `wald_se` selects the fixed-effect covariance — `WaldSe::Rx`
/// inverts the Schur complement of the β information at the mode directly
/// (observed where it differs from Fisher; assumes β–θ
/// orthogonality; anticonservative for the GLMM); `WaldSe::Hessian` (glmer
/// `use.hessian = TRUE`) sources it from the Hessian of the joint Laplace
/// deviance instead (exact wherever an exact rung answers
/// (`se::joint_hessian_cov`), finite-difference otherwise), the lme4-matching default. The optimizer
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
/// `GlmmWorkspace::params`. Callers may feed it back as
/// `theta_start`/`beta_start` for a subsequent fit of related data: the fed-back
/// θ passes through the same `THETA_TRUTH_FLOOR` floor as any other start, and
/// β is taken as given.
///
/// Errors: no `Result`. An outer search that exhausts its evaluation budget
/// (`Status::MaxFunReached`) still reports its incumbent through `GlmmFit`:
/// `converged = false`, `boundary_hit = 2`, `pinned_components` empty, and the
/// incumbent's β̂, SEs and deviance in the usual fields, and so does a gradient
/// search whose BOBYQA route ended above the point it was handed. Every other
/// non-convergence
/// (BOBYQA failure, or the degenerate-fit guard tripping on an all-infeasible
/// BOBYQA simplex) NaN-fills every β̂/SE/deviance field instead
/// (`f64::NAN` or `f64::INFINITY`; see `nan_fit`). A covariance that cannot be
/// formed at the endpoint (a non-PD Schur) NaN-fills the SE outputs alone and
/// sets `GlmmFit::se_unavailable`.
///
/// `map` is the column map `x` carries: β̂ and every SE output are reported
/// through it. The θ search uses one trust radius for all coordinates, so it
/// assumes every column a random-effect block reads as a slope is in units of
/// its own spread (standard deviation 1, or root mean square 1 where it is not
/// centered). `fit_on` provides that: `ColumnScaling::compute` scales those
/// columns even for a design fitted as given (`standardize` false), so `map`
/// is the identity only when no random-effect block has a slope. A direct
/// caller that passes an unscaled slope column searches θ in that column's
/// units. `z` is the `n × s` random-effect-only columns a slope index `≥ p`
/// reads (`ColumnScaling::compute_re_only` scales them), `n × 0` when every
/// slope is a column of `x`.
#[allow(clippy::too_many_arguments)]
pub fn fit_glmm(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    z: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    beta_start: &[f64],
    n: usize,
    wald_se: WaldSe,
    map: &ColumnScaling,
) -> GlmmFit {
    // Backstop for a hand-built workspace: the blocked and structured layouts
    // build intercept-only extras (`build_packed_m` carries the per-eval
    // debug_assert). `classify_design` routes any extra-slope shape to Sparse,
    // so this is unreachable through `fit_on`/`fit` — it catches a caller
    // that constructs `GlmmWorkspace` directly. Mirrors the same debug_assert in
    // `build_packed_m` (`glmm/workspace.rs`) and `assemble_ranef_dense`
    // (`fit/common.rs`) — change together.
    assert!(
        ws.layout == GlmmLayout::Packed || !ws.groupings.extra_slopes_any,
        "GLMM entry: extra_slopes_any on a blocked or structured layout"
    );
    let (k, p, n_theta) = (ws.k, ws.p, ws.n_theta);
    // θ entries a grouping's terms hold at zero: every search below runs on
    // the others (`optim::bobyqa_minimize`), its solver sized by `n_free`.
    let structural = ws.groupings.structural_theta().to_vec();
    let n_free = ws.groupings.n_free_theta();

    let start::StartSeed {
        guard_warm,
        is_gamma,
        held_phi,
        n_disp,
        ln_phi_plug,
        search_div,
        grad_hint,
    } = start::seed_start(ws, x, y, cluster_ids, theta_start, beta_start, n);

    // Joint BOBYQA over [θ | β]. Borrow-split mirrors `glmm_laplace_deviance`:
    // `solver` held by `minimize`; the closure calls the shared `laplace_deviance`
    // on the disjoint scratch fields (groupings read; m/lam/eta/prob/w/u/a/a_rhs
    // written). The closure's `gamma` is BOBYQA's candidate point, not the bound
    // `params` (which `minimize` owns as its `x`).
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let nagq = ws.nagq;
    let (route, grad_route, stage1_mode) = start::pick_route(ws, guard_warm, n);
    let weighted = ws.weighted;
    if matches!(family, crate::Family::NegativeBinomial { .. }) {
        let w = weighted.then_some(&ws.prior_w[..n]);
        crate::fit::nb_count_groups(&y[..n], w, &mut ws.nb_groups);
    }
    let offset = ws.offset.as_deref();
    let GlmmWorkspace {
        solver,
        config,
        solver_polish_wide,
        solver_stage1,
        solver_stage1_warm,
        solver_stage1_polish,
        config_stage1,
        params,
        params_stage1,
        lower_stage1,
        upper_stage1,
        beta_rhs,
        lower,
        upper,
        beta_scale,
        start_point,
        groupings,
        cluster_rows,
        layout,
        prior_w,
        prior_w_phi,
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
        guard_u,
        guard_beta,
        exact_prof,
        p: pf,
        pirls_exhausted,
        non_finite_points,
        counters,
        nb_groups,
        bfgs,
        grad_cfg,
        grad_report,
        grad_hint_ready,
        beta_info_scaling,
        grad_beta_info,
        grad_beta_map,
        grad_asm,
        grad_scratch,
        gradient_search_fallback,
        agq_gd,
        ..
    } = ws;
    // Joint vector's trailing `ln θ_NB` / `ln φ` index.
    let nb_col = n_theta + p;
    // x is fixed for this fit: widen the slope columns to f64 once (every
    // layout — `build_packed_m`'s primary-core reduction reads it the same way
    // `pirls_solve_blocked`'s and `fill_m_vals` do), so every BOBYQA eval's
    // per-solve M fill runs MatRef-free. `se.rs`'s `joint_hessian_cov` mirrors
    // this same hoist for its own FD deviance evals.
    fill_z_f64(groupings, x, z, z_buf, n);
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
    let mut sc = EvalScratch {
        wx,
        pirls,
        structured,
        pattern,
        packed,
        border,
        prior_w_phi,
        beta_rhs,
        beta_prof,
        exact_prof,
        cluster_rows: cluster_rows.as_ref(),
    };

    // NB marginal objective: `dev(θ_RE, β, θ_NB) − 2·nb_profile_loglik(y, y, θ_NB, w)`,
    // the `logL_marginal` the GLM θ profile (`fit::optimize_nb_theta`) maximizes,
    // times −2, so the optimum is the same point; the saturated term depends on
    // θ_NB alone. Same weights the
    // deviance carries (`prior_w`, when `weighted`).
    //
    // Gamma: `laplace_deviance` returns `D/φ + ‖u‖² + log|A|` (PIRLS on
    // `prior_wᵢ/φ`, `prior_w` already the precision-normalised `ŵ = w/s` —
    // see `fit::glmm::prep_glmm_design`); `family::gamma_dispersion_term` is
    // the φ-only rest of `−2·Σᵢ log f(yᵢ; ŵᵢ/φ, μᵢ)`, so the sum is the exact
    // Laplace deviance with φ a free parameter searched alongside θ and β.
    let gamma_sum_ln_y = if is_gamma {
        (0..n).map(|i| y[i].ln()).sum::<f64>()
    } else {
        0.0
    };
    let disp = Dispersion {
        is_gamma,
        held_phi,
        nb_theta,
        n_disp,
        gamma_sum_ln_y,
        nb_groups,
        weighted,
        prior_w,
        n,
    };

    // AGQ fits search with no stage 1: the keys below read the order.
    let agq_finish = nagq > 1;
    let total_dim = n_theta + p + n_disp;
    let total_free = n_free + p + n_disp;
    let ctx = SearchCtx {
        data,
        disp,
        groupings,
        layout: *layout,
        prior_w,
        x,
        cluster_ids,
        structural: &structural,
        family,
        nagq,
        nb_theta,
        k,
        p,
        n,
        n_theta,
        n_free,
        n_disp,
        nb_col,
        total_dim,
        total_free,
        search_div,
        is_gamma,
        held_phi,
        agq_finish,
        gamma_sum_ln_y,
        ln_phi_plug,
        gamma_phi_start: ws.gamma_phi_start,
    };
    let mut st = SearchState {
        solver,
        config,
        solver_polish_wide,
        solver_stage1_polish,
        config_stage1,
        params,
        params_stage1,
        lower_stage1,
        upper_stage1,
        lower,
        upper,
        beta_scale,
        start_point,
        u_seed,
        beta_seed,
        guard_u,
        guard_beta,
        pirls_exhausted,
        non_finite_points,
        counters,
        bfgs,
        grad_cfg,
        grad_report,
        grad_hint_ready,
        beta_info_scaling,
        grad_beta_info,
        grad_beta_map,
        grad_asm,
        grad_scratch,
        gradient_search_fallback,
        agq_gd,
    };
    let start::StartOutcome {
        warm_kept,
        guard_cache,
        guard_beta_kept,
        info_border_ready,
    } = start::choose_start(
        &ctx,
        st.reborrow(),
        &mut sc,
        guard_warm,
        grad_route,
        stage1_mode,
    );
    let grad_route::GradOutcome {
        done: grad_done,
        spent: grad_spent,
        fell_back: grad_fell_back,
        handover: grad_incumbent,
    } = if grad_route {
        grad_route::gradient_route(
            &ctx,
            st.reborrow(),
            &mut sc,
            guard_beta_kept,
            info_border_ready,
            &grad_hint,
        )
    } else {
        grad_route::GradOutcome::default()
    };
    let solver_stage1 = start::restart_search(
        &ctx,
        st.reborrow(),
        solver_stage1,
        solver_stage1_warm,
        warm_kept,
        grad_fell_back,
    );
    let RouteResult {
        ok,
        n_eval,
        budget_exhausted,
        obj: inc_obj,
    } = bobyqa_route::bobyqa_route(
        &ctx,
        st.reborrow(),
        &mut sc,
        solver_stage1,
        route,
        stage1_mode,
        warm_kept,
        guard_cache,
        grad_done,
        grad_fell_back,
        &mut ws.exact_profile_fallback,
    );
    // A gradient search that handed over spent real evaluations on this fit.
    let n_eval = n_eval + grad_spent;
    // The extra factors' `M'WM` trace for the deviance's rounding floor
    // (`deviance_rounding`), shared by the hand-over and the post-pin pass.
    // Read at the route's last evaluation: nothing evaluates between here and
    // the pin pass, and a pin that zeroes an extra factor must not drop that
    // factor's term from the floor.
    let tr_x = extra_ztwz_trace(
        &ws.pirls.w[..n],
        &ws.groupings,
        ws.layout,
        &ws.structured,
        &ws.pattern,
        &ws.packed,
    );
    let (ok, budget_exhausted, inc_obj) = report_handover(
        ws,
        grad_incumbent,
        ok,
        budget_exhausted,
        inc_obj,
        tr_x,
        nb_col,
    );

    let nb_dev_term = settle_dispersion(
        ws,
        y,
        n,
        held_phi,
        is_gamma,
        n_disp,
        nb_col,
        gamma_sum_ln_y,
        inc_obj,
    );

    let pin::PinOutcome {
        final_deviance,
        pinned,
        pinned_components,
        budget_exhausted,
        n_eval,
    } = pin::post_pin(
        ws,
        x,
        y,
        cluster_ids,
        extra_ids,
        n,
        &structural,
        total_dim,
        n_disp,
        held_phi,
        nb_dev_term,
        ok,
        inc_obj,
        budget_exhausted,
        n_eval,
        tr_x,
    );

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
    // Gamma: a φ̂ on either edge of the `ln φ` box is not a maximum-likelihood
    // estimate inside the box: the likelihood keeps rising toward that edge, as
    // on data the mean model reproduces exactly, or its maximum lies past it.
    // The fit is refused, and `phi_at_limit` names the edge.
    let phi_at_limit = (ok && is_gamma && held_phi.is_none())
        .then(|| gamma_phi_box_edge(ws.params[nb_col]))
        .flatten();
    let ok = ok && phi_at_limit.is_none();

    if !ok {
        return GlmmFit {
            phi_at_limit,
            ..nan_fit(ws, target_indices, n_eval)
        };
    }
    // Read before the SE pass: its evaluations overwrite the final re-eval's
    // weights and `M` rows that `extra_ztwz_trace` reads.
    let precision_limited = deviance_rounding(
        final_deviance,
        extra_ztwz_trace(
            &ws.pirls.w[..n],
            &ws.groupings,
            ws.layout,
            &ws.structured,
            &ws.pattern,
            &ws.packed,
        ),
    ) > DEVIANCE_FLOOR_MAX;

    for j in 0..p {
        ws.betas[j] = ws.params[n_theta + j];
    }
    // β̂ on the caller's columns (β = B·β̂ on the fitted design); every SE arm
    // below reports Var, vcov, t² and the joint Wald in the same units.
    map.beta_to_caller(&mut ws.betas);
    if let Some(i) = ws.intercept.filter(|_| ws.intercept_shift != 0.0) {
        ws.betas[i] += ws.intercept_shift;
    }

    let (joint_t_sq, hessian_fallback, se_unavailable) = se::wald_cov(
        ws,
        x,
        z,
        y,
        cluster_ids,
        extra_ids,
        target_indices,
        n,
        wald_se,
        map,
    );

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
        joint_t_sq,
        hessian_fallback,
        exact_profile_fallback: ws.exact_profile_fallback,
        gradient_search_fallback: ws.gradient_search_fallback,
        se_unavailable,
        precision_limited,
        separated: false,
        phi_at_limit: None,
        deviance: final_deviance + nb_dev_term,
        stop: budget_exhausted.then_some(crate::fit::StopReason::SearchBudget),
    }
}

/// The hand-over pass after the outer search: a gradient endpoint the BOBYQA
/// route ended above (`grad_incumbent`, its point and modes in the third part
/// of `ws.grad_scratch`) is reported in the route's place, capped. Returns the
/// reported `(ok, budget_exhausted, objective)`. `tr_x` is the extra
/// factors' `M'WM` trace at the route's last evaluation ([`extra_ztwz_trace`]).
#[allow(clippy::too_many_arguments)]
fn report_handover(
    ws: &mut GlmmWorkspace,
    grad_incumbent: Option<(usize, f64)>,
    ok: bool,
    budget_exhausted: bool,
    inc_obj: f64,
    tr_x: f64,
    nb_col: usize,
) -> (bool, bool, f64) {
    let k = ws.k;
    // Reported as a search that ran out: `budget_exhausted` gives `converged =
    // false` and `boundary_hit = 2` at a finite point, as a capped BOBYQA does.
    // The estimates are the gradient search's, so neither fallback note, which
    // say another search made them, is raised.
    // A handed-over gradient endpoint more than `joint_recheck_band` below
    // the point the BOBYQA route reports is reported in its place, capped
    // (a route that returned no point is a failed fit, as on every route):
    // from the `far` plateau the route can restart and end higher than the
    // endpoint it was handed (5 of the 31 capped precise-count refits at
    // e¹⁴, up to 41 units higher, 2026-11-03).
    // The drop must also clear the deviance's own rounding
    // (`deviance_rounding`), read at the route's last evaluation, which on a
    // converged route sits at the incumbent: on a precision-limited fit the two
    // values differ by rounding alone. Poisson counts near `e³³` with a
    // crossed factor, deviance 562: the endpoint was 0.016 (native) and 0.042
    // (wasm simd128) lower against a rounding estimate of 0.34 and a band of
    // 5.6e-7, and the hand-over reported a capped fit on one platform only.
    // The post-pin pass floors its three comparisons by the same estimate
    // (`pin::post_pin`) — change together.
    match grad_incumbent {
        Some((m, f))
            if ok
                && f < inc_obj
                    - joint_recheck_band(inc_obj).max(deviance_rounding(inc_obj, tr_x)) =>
        {
            let (point, modes) = ws.grad_scratch[m + nb_col..].split_at(m);
            ws.params[..m].copy_from_slice(point);
            ws.u_seed[..k].copy_from_slice(modes);
            ws.gradient_search_fallback = false;
            ws.exact_profile_fallback = false;
            (true, true, f)
        }
        _ => (ok, budget_exhausted, inc_obj),
    }
}

/// The search's dispersion written back for everything downstream, and the
/// reported deviance's dispersion term (returned); also sets
/// `ws.best_objective`.
#[allow(clippy::too_many_arguments)]
fn settle_dispersion(
    ws: &mut GlmmWorkspace,
    y: &[f64],
    n: usize,
    held_phi: Option<f64>,
    is_gamma: bool,
    n_disp: usize,
    nb_col: usize,
    gamma_sum_ln_y: f64,
    inc_obj: f64,
) -> f64 {
    let (family, weighted) = (ws.family, ws.weighted);
    // NB / Gamma: the incumbent's dispersion becomes the workspace's fixed θ_NB
    // or φ for everything downstream — the pinned re-eval, the SE path — which
    // all read `ws.nb_theta` / `ws.gamma_phi` the way they read it for a
    // fixed-dispersion fit.
    if let Some(v) = held_phi {
        ws.gamma_phi = v;
        ws.params[nb_col] = v.ln();
    } else if is_gamma {
        ws.gamma_phi = ws.params[nb_col].exp();
    } else if n_disp == 1 {
        ws.nb_theta = ws.params[nb_col].exp();
    }
    let nb_theta = ws.nb_theta;
    let gamma_phi = ws.gamma_phi;
    // The reported deviance's NB correction: `−2·saturated_loglik(θ̂)`, the
    // search's own `Dispersion::term(θ̂)` summed by row rather than by count group, so
    // `deviance = −2·loglik` holds. Computed once here, at the settled
    // dispersion, and returned as a plain f64 for `pin::post_pin` and the
    // reported deviance.
    // Gamma's term is the search's own: `gamma_dispersion_term` carries every
    // constant of `−2·Σ log f` already, so `deviance = −2·loglik` holds with it.
    let prior_w = &ws.prior_w;
    let nb_dev_term = if is_gamma {
        crate::family::gamma_dispersion_term(
            gamma_phi.ln(),
            weighted.then_some(&prior_w[..n]),
            n,
            gamma_sum_ln_y,
        )
    } else if n_disp == 1 {
        -2.0 * crate::family::saturated_loglik(
            family,
            nb_theta,
            y,
            weighted.then_some(&prior_w[..n]),
        )
    } else {
        0.0
    };
    // A held φ leaves the dispersion term out of the search's objective.
    ws.best_objective = if held_phi.is_some() {
        inc_obj + nb_dev_term
    } else {
        inc_obj
    };
    nb_dev_term
}

/// Estimated rounding error, in deviance units, of a Laplace deviance `f`
/// whose extra-factor columns have `M'WM` trace `tr_x` ([`extra_ztwz_trace`]):
/// `ε·(|f| + tr_x)`. [`DEVIANCE_FLOOR_MAX`] derives both terms and sets the
/// `Note::PrecisionLimited` threshold on it.
fn deviance_rounding(f: f64, tr_x: f64) -> f64 {
    f64::EPSILON * (f.abs() + tr_x)
}

/// `Σᵢ wᵢ·‖mᵢˣ‖²`, the trace of `M'WM` over the columns of the extra grouping
/// factors (`mᵢˣ` is row i of `M = ZΛ` without the primary factor's entries),
/// at the point the last Laplace evaluation solved: the weights `w` (length
/// n, `GlmmWorkspace::pirls`), `M` in the layout's packed rows. A pinned
/// factor's entries are 0 there. 0 on the blocked layout, which has no extra
/// factor. Takes the workspace's fields rather than the workspace, so
/// `fit_glmm` can read it while its fields are borrowed apart.
fn extra_ztwz_trace(
    w: &[f64],
    groupings: &crate::lmm::LmmGroupings,
    layout: GlmmLayout,
    structured: &workspace::StructuredScratch<f64>,
    pattern: &workspace::StructuredPattern,
    packed: &workspace::PackedScratch,
) -> f64 {
    let n = w.len();
    let sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    let q_p = groupings.primary_q;
    match layout {
        GlmmLayout::Blocked => 0.0,
        // `m_core_buf` rows are `[primary | nested]`; row `i`'s `n_cross[i]`
        // crossed entries take the first slots of `cross_theta`
        // (`StructuredScratch`).
        GlmmLayout::Structured => {
            let qc = q_p + groupings.nested_per_parent;
            let (st, nc) = (structured, &pattern.n_cross);
            (0..n)
                .map(|i| {
                    let nested = &st.m_core_buf[i * qc + q_p..(i + 1) * qc];
                    let crossed = &st.cross_theta[..nc[i] as usize];
                    w[i] * (sq(nested) + sq(crossed))
                })
                .sum()
        }
        // Each `m_vals` row is the primary's `q_p` entries, then the extras'
        // (`fill_packed_cols`).
        GlmmLayout::Packed => {
            let width = packed.width;
            (0..n)
                .map(|i| w[i] * sq(&packed.m_vals[i * width + q_p..(i + 1) * width]))
                .sum()
        }
    }
}

/// NaN-fill `ws.inference.vcov` on a fit with no covariance: the
/// `se_unavailable` tail of [`fit_glmm`] and [`nan_fit`]. The workspace is
/// reused across fits, so without it such a fit would serve the previous
/// fit's covariance; on a fit with one, either SE arm writes every pair.
fn nan_fill_vcov(ws: &mut GlmmWorkspace, p: usize) {
    for a in 0..p {
        for b in 0..p {
            ws.inference.vcov[(a, b)] = f64::NAN;
        }
    }
}

/// The fit returned without an outer search when the fixed-effects-only GLM on
/// `x` found no finite maximum (`crate::glm::GlmFitView::separated`). Then the
/// GLMM has none either: along the direction the GLM runs off in, every row's
/// conditional likelihood is non-decreasing for any random effects, so the
/// marginal likelihood is too. Reported like the GLM: not converged, `beta`
/// (the GLM's last iterate, on the fitted design) kept and mapped to the
/// caller's columns, every other estimate NaN. Resets the per-fit observation
/// state [`fit_glmm`] resets at its top, which the diagnostics read.
pub(crate) fn separated_fit(
    ws: &mut GlmmWorkspace,
    beta: &[f64],
    targets: &[u32],
    map: &ColumnScaling,
) -> GlmmFit {
    ws.pirls_exhausted = 0;
    ws.final_pirls_exhausted = false;
    ws.exact_profile_fallback = false;
    ws.gradient_search_fallback = false;
    ws.grad_report = None;
    ws.grad_hint_ready = false;
    ws.counters.reset();
    let mut fit = nan_fit(ws, targets, 0);
    fit.separated = true;
    fit.stop = Some(crate::fit::StopReason::IrlsDiverged);
    let (p, n_theta) = (ws.p, ws.n_theta);
    ws.params[..n_theta].fill(f64::NAN);
    ws.params[n_theta..n_theta + p].copy_from_slice(&beta[..p]);
    ws.betas[..p].copy_from_slice(&beta[..p]);
    map.beta_to_caller(&mut ws.betas);
    fit
}

/// NaN-fill the inference outputs on a fit with no endpoint, mirror `fit_lmm`'s
/// NaN-fill branch (boundary_hit = 2 = optimizer failure).
fn nan_fit(ws: &mut GlmmWorkspace, targets: &[u32], n_eval: usize) -> GlmmFit {
    for v in ws.betas.iter_mut() {
        *v = f64::NAN;
    }
    let p = ws.betas.len();
    nan_fill_vcov(ws, p);
    ws.inference.theta_cov.fill(f64::NAN);
    ws.inference.pivot = f64::NAN;
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
        joint_t_sq: f64::NAN,
        hessian_fallback: false,
        // Unlike `hessian_fallback` (decided only past every `nan_fit` call
        // site), the start-search fallback runs before the degenerate-fit
        // guard that can still route here, so a retry that reran and still
        // ended non-finite is read straight off the workspace rather than
        // hardcoded false.
        exact_profile_fallback: ws.exact_profile_fallback,
        gradient_search_fallback: ws.gradient_search_fallback,
        se_unavailable: false,
        precision_limited: false,
        separated: false,
        phi_at_limit: None,
        deviance: f64::INFINITY,
        stop: Some(crate::fit::StopReason::NoEndPoint),
    }
}
