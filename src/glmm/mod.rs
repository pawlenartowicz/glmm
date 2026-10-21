//! GLMM kernel: PIRLS + Laplace/AGQ, glmer-faithful nAGQ=1. Covers
//! Binomial/Poisson/Gamma/negative-binomial with random effects; on NB and
//! Gamma the outer search carries the dispersion (`ln θ_NB`, `ln φ`) as one
//! more trailing coordinate alongside `[θ_RE | β]`, so β, θ_RE and the
//! dispersion come out of one fit — see `fit_glmm`'s `nb_term`/`n_disp`
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
//! per (spec, max_n) shape — the warm path is zero-alloc (each `Bobyqa` is
//! built once: the cold solvers with the workspace, the warm stage-1 and the
//! near-singular polish solver on their first use),
//! except that the record of non-finite evaluations grows on a fit that has one.
//!
//! No σ² scale: binomial dispersion is fixed at 1, so D̂ = Λ̂Λ̂′ directly.
//!
//! BOBYQA is Powell, M.J.D. (2009), *The BOBYQA algorithm for bound constrained
//! optimization without derivatives*, Cambridge report DAMTP 2009/NA06.

use crate::standardize::ColumnScaling;
use crate::WaldSe;
use bobyqa::Status;
use faer::{Mat, MatRef};

use crate::lmm::{
    GLMM_RHO_POLISH, PIN_THETA, POLISH_NEAR_SINGULAR, RHO_BEGIN_WARM, THETA0, THETA_TRUTH_FLOOR,
};

use se::rx_schur_llt;
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
/// `Note::PrecisionLimited`. The estimate is `ε·(|D| + tr_x)` (`fit_glmm`):
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
/// Box of the Gamma `ln φ` outer-search coordinate (`φ` the dispersion, shape
/// `1/φ`). Wide on purpose: φ is a free maximum-likelihood parameter, and the
/// box only keeps a probe away from overflow in `lnΓ(1/φ)` and from weights
/// `prior_wᵢ/φ` past `f64` range.
pub(crate) const GAMMA_PHI_LO: f64 = 1e-6;
pub(crate) const GAMMA_PHI_HI: f64 = 1e6;
/// Distance in `ln φ` below which a fitted φ̂ counts as on the box bound and the
/// fit is refused (`fit_glmm`); BOBYQA projects onto the bound, so an incumbent
/// that reached it sits there to round-off.
const GAMMA_PHI_BOUND_TOL: f64 = 1e-6;

/// Whether a fitted `ln φ̂` sits on a bound of the Gamma coordinate's box
/// (within [`GAMMA_PHI_BOUND_TOL`]): an incumbent there describes the box, not
/// a maximum of the likelihood, and `fit_glmm` refuses the fit.
pub(crate) fn gamma_phi_on_box_bound(ln_phi: f64) -> bool {
    ln_phi <= GAMMA_PHI_LO.ln() + GAMMA_PHI_BOUND_TOL
        || ln_phi >= GAMMA_PHI_HI.ln() - GAMMA_PHI_BOUND_TOL
}
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
    /// Total BOBYQA evaluations across both stages (stage 1 + stage 2; 0 + stage 2
    /// on the single-stage path).
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
    /// Minimized marginal Laplace deviance at the pinned γ̂ (`d(y,ũ)+‖ũ‖²+log|A|`,
    /// or the AGQ deviance when `nagq>1`), reported as `Fit::deviance`.
    /// `f64::INFINITY` on non-convergence. On an NB fit it carries
    /// `−2·saturated_loglik(θ̂)` on top, so the reported value is exactly
    /// `−2·logLik`.
    pub deviance: f64,
}

mod agq;
mod agq_fast;
mod agq_gradient;
mod assembled;
mod derivative;
mod deviance;
mod pirls;
mod se;
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
/// at that whitening's curvature (`fit_glmm`'s gradient-route block,
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
/// the Laplace shapes the warm member is BFGS too, at any rows per level.
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
) -> crate::optim::RoutePlan {
    use crate::optim::{OuterOptimizer, RoutePlan};
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
        RoutePlan {
            cold: OuterOptimizer::Bfgs,
            warm: OuterOptimizer::Bfgs,
        }
    } else {
        RoutePlan::BOBYQA
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
    let m = grad_search_dim(ws.family, ws.n_theta, ws.p, ws.gamma_phi_held.is_some());
    let h = ws.bfgs.inv_hessian();
    if !ws.grad_beta_info {
        return crate::optim::CurvatureHint::from_inverse(m, h, ws.search_units);
    }
    let (nt, p) = (ws.n_theta, ws.p);
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
/// Gamma fit whose dispersion is not held. The curvature hint the search
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

/// Rows per primary level from which an AGQ fit goes back to BOBYQA (see
/// [`gradient_route_plan`]).
const GRAD_WARM_SCALAR_ROWS_MAX: usize = 50;

/// A warm fit whose θ start has a diagonal `|θ|` above this takes the BOBYQA
/// route. From such far starts the joint `[θ | β]` search can end in a worse
/// local minimum than the profiled `ExactProfile` search does: 2 of 600
/// cloglog crossed starts at θ₀ = 20 and 1000 ended 0.27 and 1.09 above the
/// cold optimum, reported converged (`far_warm_start_scan`, 2026-10-07).
/// Warm starts from a truth or a previous estimate sit far below it.
const GRAD_WARM_THETA_MAX: f64 = 10.0;

/// A warm AGQ fit whose θ start has a diagonal below this takes the BOBYQA
/// route. Measured on 48 warm rows (locked clock, 2026-10-10): every start at
/// θ₀ <= 0.02 was slower than warm BOBYQA (1.13x to 1.57x), every start at
/// θ₀ >= 0.05 at most 1.22x (geomean 0.826 over 38 rows), no start between
/// 0.02 and 0.05. A start below it ends near θ = 0, fails the certificate's
/// near-zero test and pays the rung-3 polish, which costs more than the warm
/// BOBYQA it replaces.
const GRAD_WARM_AGQ_THETA_MIN: f64 = 0.05;

/// A gradient-search endpoint with a diagonal `|θ|` at or above this is not
/// trusted and goes to the BOBYQA route (a tenth of `THETA_HI`; a random-effect
/// standard deviation 100 times the link scale's unit is far past any fitted
/// value on the grid, so this fires only on the plateau toward the box).
const GRAD_THETA_FAR: f64 = 0.1 * crate::lmm::THETA_HI;

/// Above this many bytes, the gradient search's assembly scratch (`G_γ`,
/// `m·k`, and the crossed tail's forms, sized by the row capacity) is
/// released after the search instead of kept for the next fit, so it does
/// not add to a large fit's SE-step peak. Kept up to 16 MiB of `G_γ` alone,
/// it raised the peak of a 300k-row binomial q4 fit with Hessian SEs by 5.3%
/// (188 to 198 MB) and a 300k-row crossed one by 3.2% (2026-10-07). Below it a
/// simulation loop reuses it and allocates nothing per fit.
const GRAD_ASM_KEEP_BYTES: usize = 4 << 20;

/// The gradient route's objective: the fixed-β Laplace deviance over
/// `γ̃ = [θ | β̃]`, every evaluation at the family's full PIRLS tolerance from
/// the incumbent's mode, and its gradient off the state the last evaluation
/// left ([`assembled::gradient_at_mode`]). On a covered AGQ shape
/// (`nagq > 1`, [`agq_gradient::covers_shape`]) the value is the fixed-β AGQ
/// deviance at `nagq` nodes, the joint BOBYQA search's own objective, and the
/// gradient is [`agq_gradient::assemble`] off the modes and `η_fix` it left.
/// On a free Gamma dispersion the
/// point is `[θ | β̃ | ψ]`, `ψ = ln φ` (internal units, not whitened), and the
/// value adds the dispersion term `Gₚ(ψ)`, as the BOBYQA objective does.
/// Under a held φ the search runs over `[θ | β̃]` at that φ, and a trailing
/// slot (the joint polish's inert one) is ignored.
///
/// β is searched whitened, `β̃ = Lᵀβ` (`BetaScale`), and the β block of the
/// gradient maps by `L⁻¹`. On a Laplace fit without a caller's hint `L` comes
/// from the β information at the start (`BetaScale::set_info`, see
/// `fit_glmm`'s gradient-route block), elsewhere from the design (the joint
/// BOBYQA search's `BetaScale::set`). In the caller's units
/// the first step's 0.5 cap is not unit-free: with a slope column scaled by
/// 1e-3 a logit fit stopped 5.5e-4 above the optimum, certified, where the
/// whitened search reaches it (test `gradient_route_is_unit_invariant_in_beta`,
/// 2026-10-07).
///
/// The incumbent mode `u_inc` (`ws.u_seed`) moves only when the driver
/// accepts a point, which it signals by asking for that point's gradient, so
/// a rejected trial never seeds the next one. A value whose PIRLS solve ran
/// its iteration cap is a failed trial (`None`), so an accepted point always
/// rests on a converged mode.
struct GradSearch<'a, 'd> {
    data: &'a FitData<'d>,
    nb_theta: f64,
    /// Quadrature nodes: 1 (Laplace) or the AGQ order.
    nagq: u8,
    /// The AGQ value kernel's cluster-outer row index, handed to each value
    /// evaluation as the joint search hands it (`None` at `nagq == 1`).
    cluster_rows: Option<&'a agq::ClusterRowIndex>,
    /// The AGQ gradient's own row index and buffers (`ws.agq_gd`); `Some`
    /// exactly when `nagq > 1`.
    agq_gd: Option<&'a mut agq_gradient::AgqGd>,
    wx: &'a mut Mat<f64>,
    pirls: &'a mut workspace::PirlsScratch<f64>,
    structured: &'a mut workspace::StructuredScratch<f64>,
    pattern: &'a mut workspace::StructuredPattern,
    packed: &'a mut workspace::PackedScratch,
    border: &'a mut workspace::BorderScratch,
    prior_w_phi: &'a mut [f64],
    beta_rhs: &'a mut [f64],
    beta_prof: &'a mut [f64],
    exact_prof: &'a mut pirls::ExactProfileBufs,
    pirls_exhausted: &'a mut u32,
    counters: &'a mut crate::counters::EvalCounters,
    non_finite_points: &'a mut Vec<f64>,
    grad_asm: &'a mut Option<assembled::AssemblyBufs<f64>>,
    u_inc: &'a mut [f64],
    beta_scale: &'a mut workspace::BetaScale,
    /// The last value evaluation's `[θ | β]`, β in the caller's units, for
    /// the gradient's `∂M/∂θ` (the second part of `ws.grad_scratch`, length
    /// `n_θ + p`).
    x_last: &'a mut [f64],
    /// Whether the last value evaluation is a gradient point: finite, on a
    /// converged mode.
    last_ok: bool,
    /// The Gamma dispersion φ (internal units) the last value evaluation ran
    /// PIRLS at: `e^ψ` on a free dispersion, the held φ under a held one, 1
    /// on every other family.
    gamma_phi: f64,
    /// `ln` of `gamma_phi`: the last point's `ψ` itself on a free dispersion,
    /// so the `ψ` gradient entry is taken at the point the value was.
    ln_phi: f64,
    /// Whether the point carries a free Gamma `ψ = ln φ` after `[θ | β̃]`.
    free_phi: bool,
    /// `Σ ln yᵢ`, the dispersion term's data constant (`fit_glmm`'s
    /// `gamma_sum_ln_y`); read on a free Gamma dispersion only.
    sum_ln_y: f64,
    /// Every β̃ entry's start inverse curvature, `1/(2·Σ prior weight)`, when
    /// β̃ is whitened by the β information (`fit_glmm`'s `grad_beta_info`);
    /// `None` keeps the identity's entries.
    beta_inv_curv: Option<f64>,
    k: usize,
    n_theta: usize,
}

impl crate::optim::Objective for GradSearch<'_, '_> {
    fn value(&mut self, x: &[f64], _resolution: f64) -> Option<f64> {
        let k = self.k;
        let d = self.data;
        let q = self.n_theta + d.p;
        let psi = self.free_phi.then(|| x[q]);
        if let Some(c) = psi {
            self.gamma_phi = c.exp();
            self.ln_phi = c;
        }
        self.x_last.copy_from_slice(&x[..q]);
        self.beta_scale.to_caller(&mut self.x_last[self.n_theta..]);
        self.pirls.u[..k].copy_from_slice(&self.u_inc[..k]);
        let exhausted_before = *self.pirls_exhausted;
        let dev = laplace_deviance(
            d,
            self.nb_theta,
            self.gamma_phi,
            self.nagq,
            self.x_last,
            self.beta_rhs,
            self.wx,
            self.pirls,
            self.structured,
            self.pattern,
            self.packed,
            self.border,
            self.prior_w_phi,
            self.beta_prof,
            self.exact_prof,
            BetaMode::Fixed,
            None,
            self.cluster_rows,
            self.pirls_exhausted,
            self.counters,
        );
        // The Gamma arm of `fit_glmm`'s `nb_term` — change together.
        let dev = match psi {
            Some(c) => {
                dev + crate::family::gamma_dispersion_term(
                    c,
                    d.weighted.then_some(d.prior_w),
                    d.n,
                    self.sum_ln_y,
                )
            }
            None => dev,
        };
        self.counters.record_eval(crate::counters::Stage::Two, dev);
        if !dev.is_finite() {
            self.non_finite_points.extend_from_slice(x);
        }
        self.last_ok = dev.is_finite() && *self.pirls_exhausted == exhausted_before;
        self.last_ok.then_some(dev)
    }
}

impl crate::optim::GradObjective for GradSearch<'_, '_> {
    fn gradient(&mut self, g: &mut [f64]) -> bool {
        // The driver asks only right after the value at the point it accepts.
        debug_assert!(self.last_ok, "gradient requested off a failed value");
        if !self.last_ok {
            return false;
        }
        let k = self.k;
        self.u_inc[..k].copy_from_slice(&self.pirls.u[..k]);
        let d = self.data;
        let (n_theta, p, n) = (self.n_theta, d.p, d.n);
        if let Some(gd) = self.agq_gd.as_deref_mut() {
            // AGQ: `[λ | β]` in the caller's units, so `d.x` is the design.
            // A row at the link's clamp leaves the gradient undefined: a
            // silent hand-over, as an undefined Laplace gradient is.
            let ok = agq_gradient::assemble(
                d.family,
                self.nagq,
                d.groupings.n_primary,
                self.pirls,
                d.prior_w,
                d.x,
                d.y,
                &gd.idx,
                self.x_last,
                &mut gd.sc,
                g,
            );
            if !ok || !g.iter().all(|v| v.is_finite()) {
                return false;
            }
            self.beta_scale.to_search_row(&mut g[n_theta..n_theta + p]);
            return true;
        }
        // The weights the value evaluation handed PIRLS. `laplace_deviance`
        // writes `prior_w_phi` only on Gamma at φ ≠ 1, so at φ = 1 it holds an
        // earlier evaluation's weights (mirrors its `pirls_prior_w` — change
        // together).
        let scaled = matches!(d.family, crate::Family::Gamma { .. }) && self.gamma_phi != 1.0;
        let prior_w = if scaled {
            &self.prior_w_phi[..n]
        } else {
            d.prior_w
        };
        if assembled::gradient_at_mode(
            d.groupings,
            d.family,
            self.nb_theta,
            d.x,
            d.y,
            d.cluster_ids,
            d.extra_ids,
            self.x_last,
            d.z_buf,
            prior_w,
            self.pirls,
            self.structured,
            self.pattern,
            self.grad_asm,
            n_theta,
            p,
            n,
            g,
        )
        .is_none()
        {
            return false;
        }
        // A free Gamma dispersion's `ψ = ln φ` slot trails `[θ | β̃]`; it is
        // not whitened.
        if g.len() > n_theta + p {
            g[n_theta + p] = assembled::gamma_ln_phi_gradient(
                self.ln_phi,
                d.weighted.then_some(d.prior_w),
                n,
                self.pirls.pen_dev,
                &self.x_last[..n_theta],
                &g[..n_theta],
            );
        }
        // ∂/∂β̃ = L⁻¹·∂/∂β, the row map (`x̃ = x·L⁻ᵀ`).
        self.beta_scale.to_search_row(&mut g[n_theta..n_theta + p]);
        true
    }

    /// The start inverse Hessian's seeded entries; every other entry, θ's
    /// included (its curvatures are tens), is the identity's.
    ///
    /// β̃, when it is whitened by the β information (`beta_inv_curv`): every
    /// entry `1/(2·Σ prior weight)`, the curvature `set_info` gives each β̃
    /// axis. The driver keeps these entries out of Shanno's rescaling:
    /// rescaled from the first secant pair, β̃ ended fits certified up to
    /// 1.8e-4 above the optimum, the decrement `½·gᵀHg` too small (measured
    /// 2026-10-20).
    ///
    /// ψ, on a free Gamma dispersion: its curvature at fixed
    /// `ξ = θ·e^{−ψ/2}`. From `F = e^{−ψ}·P(ξ, β) + log|A| + Gₚ(ψ)`,
    /// `∂²F/∂ψ² = P_φ + Gₚ''(ψ)` with `P_φ = e^{−ψ}·P` the last value's
    /// `dev + pen` (`pirls.pen_dev`), of order n; the entry is its inverse.
    /// Inverse link only: on the log link the seed costs warm starts more
    /// evaluations than it saves cold ones (measured on the grid's routed
    /// Gamma/log cells), so ψ keeps the identity's entry there.
    fn start_curvature(&self, inv_curv: &mut [f64]) {
        let d = self.data;
        let q = self.n_theta + d.p;
        if let Some(v) = self.beta_inv_curv {
            inv_curv[self.n_theta..q].fill(v);
        }
        let inverse = matches!(
            d.family,
            crate::Family::Gamma {
                link: crate::GammaLink::Inverse
            }
        );
        if !self.free_phi || !inverse {
            return;
        }
        let c = crate::family::gamma_dispersion_term_d2(
            self.ln_phi,
            d.weighted.then_some(d.prior_w),
            d.n,
        ) + self.pirls.pen_dev;
        if c > 0.0 && c.is_finite() {
            inv_curv[q] = 1.0 / c;
        }
    }
}

impl GradSearch<'_, '_> {
    /// The saddle walk of the `ExactProfile` endpoint, on the gradient
    /// search's endpoint (β held at the endpoint's): each nonzero `Λ` column
    /// below `POLISH_NEAR_SINGULAR` is doubled while the deviance falls. Every
    /// try is a value evaluation, counted in the second value; the first is
    /// whether a try was kept (the endpoint is then not certified). `best` is
    /// the endpoint's value and follows a kept try, whose mode becomes the
    /// incumbent's. Same block and column walk as `fit_glmm`'s stage-1 saddle
    /// check and `crate::lmm::fix_column_signs` — change together.
    fn saddle_walk(
        &mut self,
        g: &crate::lmm::LmmGroupings,
        x: &mut [f64],
        best: &mut f64,
    ) -> (bool, usize) {
        use crate::optim::Objective;
        let (mut moved, mut tries) = (false, 0usize);
        let mut start = 0usize;
        for &q in std::iter::once(&g.primary_q).chain(g.extra_q.iter()) {
            for d in 0..q {
                let col = start..start + (q - d);
                start = col.end;
                loop {
                    let norm_sq: f64 = x[col.clone()].iter().map(|v| v * v).sum();
                    if norm_sq == 0.0 || 4.0 * norm_sq > POLISH_NEAR_SINGULAR * POLISH_NEAR_SINGULAR
                    {
                        break;
                    }
                    for v in x[col.clone()].iter_mut() {
                        *v *= 2.0;
                    }
                    tries += 1;
                    match self.value(x, 0.0) {
                        Some(f) if f < *best => {
                            *best = f;
                            moved = true;
                            let k = self.k;
                            self.u_inc[..k].copy_from_slice(&self.pirls.u[..k]);
                        }
                        _ => {
                            for v in x[col.clone()].iter_mut() {
                                *v *= 0.5;
                            }
                            break;
                        }
                    }
                }
            }
        }
        (moved, tries)
    }
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
/// `GlmmWorkspace::params`. Callers may feed it back as
/// `theta_start`/`beta_start` for a subsequent fit of related data: the fed-back
/// θ passes through the same `THETA_TRUTH_FLOOR` floor as any other start, and
/// β is taken as given.
///
/// Errors: no `Result`. An outer search that exhausts its evaluation budget
/// (`Status::MaxFunReached`) still reports its incumbent through `GlmmFit`:
/// `converged = false`, `boundary_hit = 2`, `pinned_components` empty, and the
/// incumbent's β̂, SEs and deviance in the usual fields. Every other
/// non-convergence (BOBYQA failure, or the degenerate-fit guard tripping on an
/// all-infeasible BOBYQA simplex) NaN-fills every β̂/SE/deviance field instead
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
/// units.
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
    map: &ColumnScaling,
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

    // γ₀ = [θ₀ | β₀]. `guard_warm` is set only on a warm start (the `Some` arm
    // below) — the start guard further down then decides whether the search
    // should begin at the caller's θ₀ or at the blind cold start instead. A
    // cold fit skips the guard entirely.
    let mut guard_warm = false;
    // Set when the start guard keeps the caller's θ₀ (stage 1 then searches
    // from it at `RHO_BEGIN_WARM`).
    let mut warm_kept = false;
    match theta_start {
        Some(ts) => {
            // Floor is diagonal-only: a caller's θ is a Cholesky factor, so its
            // diagonals are non-negative and the floor only keeps them off the
            // pin threshold; off-diagonals are signed, so flooring them would
            // silently rewrite a negative correlation start. lme4 passes
            // `start$theta` through verbatim. `ts` is already on the fitted
            // design (`fit::common::start_to_design`), the scale `PIN_THETA`
            // tests.
            for (t, &v) in ws.params[..n_theta].iter_mut().zip(ts) {
                *t = v;
            }
            for &i in ws.groupings.diagonal_theta() {
                ws.params[i] = ws.params[i].max(THETA_TRUTH_FLOOR);
            }
            // The start guard further down compares a warm caller's θ₀ with
            // the blind cold θ this data would start from otherwise, and the
            // search begins from whichever the outer objective is lower at.
            //
            // A start AT the blind θ₀, bit for bit (MCPower passes `[THETA0; n]`
            // when it has neither a truth nor a carried start), gives the guard
            // nothing to compare: its two candidates are then the same point
            // `[θ₀ | β₀]`, since the guard writes the caller's β₀ into both and
            // shares the dispersion coordinate, so it would run one solve twice
            // and the blind copy would win the tie. Either way the search starts
            // at the caller's point, from the same seeds a cold fit uses (û = 0,
            // β₀, cleared curvature memory), so the guard is skipped on every
            // route.
            // The blind θ₀: mirrors the `None` arm and the guard's blind point;
            // change together.
            let blind = |i: usize| {
                if ws.groupings.diagonal_theta().contains(&i) {
                    THETA0
                } else {
                    0.0
                }
            };
            let at_blind = (0..n_theta).all(|i| ws.params[i].to_bits() == blind(i).to_bits());
            guard_warm = !at_blind;
        }
        None => {
            // Blind start: diagonals THETA0, off-diagonals 0 — the structure-only
            // blind θ₀ `GlmmWorkspace::new` builds (workspace.rs). The former
            // all-THETA0 start implied RE correlation +0.707 for every pair, and
            // on negative-correlation data that converges into the τ=0 boundary
            // basin.
            // Mirrors the warm arm's `blind` closure and the start guard's
            // blind point; change together.
            for t in ws.params[..n_theta].iter_mut() {
                *t = 0.0;
            }
            for &i in ws.groupings.diagonal_theta() {
                ws.params[i] = THETA0;
            }
        }
    }
    for (j, &b) in beta_start.iter().enumerate().take(p) {
        ws.params[n_theta + j] = b;
    }

    // NB and Gamma: the trailing outer-search coordinate. NB's starts at ln of
    // the seed the adapter left in `nb_theta`; Gamma's at ln of the plug-in
    // dispersion `D/Σw` of the β start with no random effect, the no-RE GLM's
    // own deviance dispersion (the RE variance is charged to φ there, so the
    // seed sits above φ̂). Both are clamped into the coordinate's box.
    let is_gamma = matches!(ws.family, crate::Family::Gamma { .. });
    // A caller-held Gamma φ (`GlmmWorkspace::gamma_phi_held`): every evaluation
    // runs at it and the trailing coordinate is inert.
    let held_phi = if is_gamma { ws.gamma_phi_held } else { None };
    let n_disp = usize::from(matches!(
        ws.family,
        crate::Family::NegativeBinomial { .. } | crate::Family::Gamma { .. }
    ));
    let m = n_theta + p;
    // The plug-in seed's `ln φ`, kept when a carried φ starts the coordinate
    // instead: the `ln φ` scale gate in stage 1 measures it.
    let mut ln_phi_plug = f64::NAN;
    if is_gamma {
        let fam = ws.family;
        let (mut d, mut sw) = (0.0, 0.0);
        for i in 0..n {
            let mut eta = ws.offset.as_ref().map_or(0.0, |o| o[i]);
            for j in 0..p {
                eta += x[(i, j)] * ws.params[n_theta + j];
            }
            d += ws.prior_w[i] * crate::family::dev_resid_eta(fam, f64::NAN, y[i], eta);
            sw += ws.prior_w[i];
        }
        // Data the fixed effects reproduce exactly give `D = 0`, which rounds to
        // either sign: the seed goes to the box floor then, never to `ln` of a
        // non-positive number (a NaN start that BOBYQA refuses).
        let disp = d / sw;
        let disp = if disp.is_finite() && disp > 0.0 {
            disp
        } else {
            GAMMA_PHI_LO
        };
        ln_phi_plug = disp.ln().clamp(ws.lower[m], ws.upper[m]);
        let disp = held_phi.or(ws.gamma_phi_start).unwrap_or(disp);
        ws.params[m] = disp.ln().clamp(ws.lower[m], ws.upper[m]);
    } else if n_disp == 1 {
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
    ws.exact_profile_fallback = false;
    ws.gradient_search_fallback = false;
    ws.grad_report = None;
    ws.grad_hint_ready = false;
    ws.search_units = 1.0;
    // A caller's hint applies to this call only.
    let grad_hint = ws.grad_hint_in.take();
    ws.exact_prof.curv_memory = false;
    ws.non_finite_points.clear();
    ws.counters.reset();
    ws.pattern.coup_mask = None; // CSR validity is per (fit, pinning mask): ids/z may differ across fits

    // Cluster-outer AGQ substrate: built once per fit (cluster_ids is fit-fixed).
    // Scalar AGQ (`q_p = 1`) reads it ONLY in `parallel` builds, where it is
    // rayon's work-splitting substrate. Serial builds run scalar AGQ's original
    // node-outer loop: even with the reweight hoist in agq_deviance,
    // cluster-outer's residual per-cluster overhead (loop restart + CSR gather)
    // regresses many-tiny-cluster shapes (observation-level REs: +12–16%
    // measured on grouseticks (1|INDEX), 403 clusters × 1 row) while its serial
    // cache win on large clusters is small (−4% on cbpp), so the non-parallel
    // hot path (batch loops) stays byte-for-byte unchanged. The grid campaign's
    // parallel pass owns the decision of whether a rows-per-cluster dispatch can
    // unlock the win. Vector AGQ (`q_p ≥ 2`) is cluster-outer on every build and
    // builds a transient index per evaluation when none is handed in, so a
    // serial build gets it here too, once per fit, rebuilt in place into the
    // previous fit's buffers.
    let par_index = cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
        && ws.nagq > 1
        && ws.parallel_inner;
    let serial_vec_index = !cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
        && ws.nagq > 1
        && ws.groupings.primary_q >= 2
        && ws.groupings.extra_offsets.is_empty();
    ws.cluster_rows = if par_index || serial_vec_index {
        let mut idx = ws.cluster_rows.take().unwrap_or_default();
        idx.rebuild(cluster_ids, ws.groupings.n_primary);
        Some(idx)
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
    // so this is a plain read. `mut`: the `PqlThenJoint` fallback further down
    // rebinds this once, after an `ExactProfile` attempt ends not converged, to
    // rerun the search on the fallback route — the shape's OWN route, read
    // here, is what the fallback gate checks, and rebinding a route that is
    // never `ExactProfile` again self-limits the retry to one.
    let mut route = ws.outer_search;
    // The gradient route (`gradient_route_plan`): BFGS over `[θ | β]` on this
    // fit's start kind, an AGQ fit only below `GRAD_WARM_SCALAR_ROWS_MAX` rows
    // per primary level, and a warm AGQ fit only from a start with every
    // diagonal θ₀ at or above `GRAD_WARM_AGQ_THETA_MIN`. A warm start at the
    // blind θ₀ is a cold fit here, as it is for the start guard.
    let grad_route = {
        use crate::optim::OuterOptimizer;
        let g = &ws.groupings;
        let opt = if guard_warm {
            ws.route_plan.warm
        } else {
            ws.route_plan.cold
        };
        let near_start = g
            .diagonal_theta()
            .iter()
            .all(|&i| ws.params[i].abs() <= GRAD_WARM_THETA_MAX);
        let few_rows = n < GRAD_WARM_SCALAR_ROWS_MAX * g.n_primary.max(1);
        let off_zero = g
            .diagonal_theta()
            .iter()
            .all(|&i| ws.params[i] >= GRAD_WARM_AGQ_THETA_MIN);
        opt == OuterOptimizer::Bfgs
            && (nagq == 1 || few_rows)
            && (!guard_warm || near_start)
            && (nagq == 1 || !guard_warm || off_zero)
    };
    debug_assert!(
        route != OuterSearch::ExactProfile || exact_profile_shape(nagq, ws.layout),
        "ExactProfile requested on a shape without an exact profile"
    );
    // Which `BetaMode` stage 1 profiles β with, or `None` to skip stage 1 entirely
    // (the `Joint` route). `stage1_mode` alone decides whether stage 1 runs; the
    // route decides what happens AFTER it — see the `route ==` split below.
    // `mut`: rebound alongside `route` by the fallback.
    let mut stage1_mode = match route {
        OuterSearch::Joint => None,
        OuterSearch::PqlThenJoint => Some(BetaMode::ProfilePql),
        OuterSearch::ExactProfile => Some(BetaMode::ProfileExact),
    };
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
    fill_z_f64(groupings, x, z_buf, n);
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
    let nb_term = |ln_theta: f64| -> f64 {
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
    };
    // The (θ_NB, φ) pair one evaluation runs at, off the trailing coordinate
    // `c`, and the coordinate the dispersion term is read at: NB reads θ_NB =
    // e^c at φ = 1, Gamma reads φ = e^c, and a held Gamma φ reads neither — the
    // coordinate is inert and the objective is the fixed-φ Laplace deviance
    // (its dispersion term a constant, restored in the reported deviance).
    let disp_at = |c: f64| -> ((f64, f64), Option<f64>) {
        match (is_gamma, held_phi) {
            (true, Some(v)) => ((nb_theta, v), None),
            (true, None) => ((nb_theta, c.exp()), Some(c)),
            _ => ((c.exp(), 1.0), Some(c)),
        }
    };

    // Start guard (warm fits only): compare the outer objective at the
    // caller's θ₀ against the blind cold start, on the objective this fit's
    // own route runs first (the route's Profile mode when stage 1 runs, i.e.
    // `nagq == 1`; `BetaMode::Fixed`, the `Joint` route's own objective,
    // otherwise), and begin the search from whichever is lower. A tie or a
    // non-finite warm value goes to the blind start: a warm start earns no
    // benefit of the doubt over the point a cold fit of the same data would
    // use anyway. Cold fits (`guard_warm` false) skip this block, so their
    // evaluation count and result are unchanged. Each guard solve has its own
    // discarded `pirls_exhausted`/`counters` scratch: the diagnostics see a
    // guard solve only when the search takes it over (below).
    //
    // The hand-over rule. Each candidate is solved the way a fit's first
    // evaluation is: PIRLS from û = 0 and β₀, with the curvature memory
    // (`ExactProfileBufs::curv_memory`) cleared. When stage 1 runs on the
    // guard's own mode and its first call lands on the picked point bit for
    // bit, the winning solve IS the search's first evaluation: the stage-1
    // objective takes its objective, PIRLS-exhaustion count, û and β̂
    // (`guard_u`/`guard_beta`) instead of solving the point again, and the
    // search starts with the curvature memory that solve left. The memory
    // follows the pick, so the rejected candidate leaves no trace. When
    // BOBYQA moves the start (into its box, or onto a bound it lies within
    // the initial radius of; see the stage-1 objective), the first call
    // misses: it solves its own point from û = 0, β₀ and a cleared memory, as
    // a cold fit's first call does. Either way a warm fit sent to the blind
    // start begins as the cold fit does, and with the same β₀ it runs the
    // cold fit's search (measured on the far-warm-start scan, x86_64: all
    // 1009 such fits are the cold fit bit for bit). The cache is consumed
    // once, so the `PqlThenJoint` retry re-solves its start. On a warm fit
    // with a stage 1 the guard costs one Laplace-deviance solve over a cold
    // fit on a hit and two on a miss; on the `Joint` route it costs two; on
    // the gradient route's Laplace fits it costs one PQL solve when the warm
    // start is kept and two when it is not (below); all outside the reported
    // `n_eval`.
    let mut guard_cache: Option<(f64, u32)> = None;
    // The guard left the picked start's profiled β̂ in `guard_beta`.
    let mut guard_beta_kept = false;
    // The guard's last solve was the kept warm start's `ProfilePql` solve, so
    // `border` already holds the gradient route's β information (below).
    let mut info_border_ready = false;
    if guard_warm {
        let disp_coord = if n_disp == 1 { params[nb_col] } else { 0.0 };
        // The gradient route on a Laplace fit compares the starts on the PQL
        // profile, not on its shape's stage-1 mode. It has no stage 1 to hand
        // a solve over to. On 3,370 such warm fits (grid, stress and sweep
        // cells, 2026-10-20) the PQL profile picked the same start as the
        // exact profile on all but 4, whose fitted deviances moved by at most
        // 2e-10; no other fit's deviance moved by more than 1e-6.
        // On the β-information opt-in, the kept warm start's solve doubles as
        // the β-information solve (below). The fixed-β objective at β₀ is not enough: it kept a θ₀ on
        // the box bound that the profile rejects (`far_warm_start_scan`, logit
        // crossed seed 9, θ₀ = THETA_HI). The route then starts from the
        // guard's profiled β̂ at the chosen θ. On AGQ (no stage 1) the guard
        // compares on the fixed-β AGQ objective at β₀ and keeps no
        // `guard_cache`, so the gradient route starts from β₀ at the guard's θ.
        let pql_guard = grad_route && nagq == 1;
        let guard_mode = if pql_guard {
            BetaMode::ProfilePql
        } else {
            stage1_mode.unwrap_or(BetaMode::Fixed)
        };
        let cacheable = guard_mode != BetaMode::Fixed;
        // `laplace_deviance`'s Fixed mode reads β off `params[n_theta..n_theta
        // + p]` itself (see its doc), not off the `beta` argument's incoming
        // value, so the point passed in must carry β at that offset: `combined`
        // is `[θ | β₀]`, θ written per candidate, β₀ fixed across both.
        let combined = &mut start_point[..n_theta + p];
        combined.copy_from_slice(&params[..n_theta + p]);
        // Solves one candidate from a fit's first-evaluation state. Returns
        // its objective, PIRLS-exhaustion count and the curvature memory it
        // left; with `snap` it also keeps its û and β̂ for the hand-over.
        let mut eval_guard = |combined: &mut [f64], snap: bool| -> (f64, u32, bool) {
            exact_prof.curv_memory = false;
            for v in pirls.u[..k].iter_mut() {
                *v = 0.0;
            }
            // Same buffer routing `laplace_deviance`'s doc requires: Fixed
            // reads β off `beta_rhs` (spare `beta_prof`), Profile profiles
            // `beta_prof` from the β0 seed (spare `beta_rhs`), one tuple-typed
            // `if` (mirrors `laplace_deviance_ws`), so the borrow checker sees
            // one reborrow of each buffer, not two competing ones.
            let (beta, beta_step_rhs): (&mut [f64], &mut [f64]) = if guard_mode == BetaMode::Fixed {
                (beta_rhs, beta_prof)
            } else {
                beta_prof[..p].copy_from_slice(&combined[n_theta..]);
                (beta_prof, beta_rhs)
            };
            let ((nb, phi), coord) = if n_disp == 1 {
                disp_at(disp_coord)
            } else {
                ((nb_theta, 1.0), None)
            };
            let mut guard_exhausted = 0u32;
            let mut guard_counters = crate::counters::EvalCounters::new();
            let dev = laplace_deviance(
                &data,
                nb,
                phi,
                nagq,
                combined,
                beta,
                wx,
                pirls,
                structured,
                pattern,
                packed,
                border,
                prior_w_phi,
                beta_step_rhs,
                exact_prof,
                guard_mode,
                None,
                cluster_rows.as_ref(),
                &mut guard_exhausted,
                &mut guard_counters,
            );
            if snap {
                guard_u[..k].copy_from_slice(&pirls.u[..k]);
                guard_beta[..p].copy_from_slice(&beta_prof[..p]);
            }
            let obj = match coord {
                Some(c) => dev + nb_term(c),
                None => dev,
            };
            (obj, guard_exhausted, exact_prof.curv_memory)
        };
        // The blind cold θ: the same point the `None` arm above builds.
        // Mirrors the `None` arm and the warm arm's `blind` closure; change
        // together.
        let set_blind = |theta: &mut [f64]| {
            theta.fill(0.0);
            for &i in groupings.diagonal_theta() {
                theta[i] = THETA0;
            }
        };
        if pql_guard {
            // Blind first, warm last, so a kept warm start leaves its own
            // border in `border`. The search never takes a guard solve over
            // here (no `guard_cache`): a BOBYQA fallback's stage 1 may run
            // the exact profile, and the cache matches on θ alone.
            set_blind(&mut combined[..n_theta]);
            let (cold_obj, _, cold_mem) = eval_guard(combined, true);
            // `params` takes the blind θ, `combined` the caller's θ₀.
            params[..n_theta].swap_with_slice(&mut combined[..n_theta]);
            let (warm_obj, _, warm_mem) = eval_guard(combined, false);
            if !warm_obj.is_finite() || cold_obj <= warm_obj {
                exact_prof.curv_memory = cold_mem;
            } else {
                params[..n_theta].copy_from_slice(&combined[..n_theta]);
                guard_beta[..p].copy_from_slice(&beta_prof[..p]);
                exact_prof.curv_memory = warm_mem;
                warm_kept = true;
                info_border_ready = true;
            }
            guard_beta_kept = true;
        } else {
            let (warm_obj, warm_exh, warm_mem) = eval_guard(combined, cacheable);
            set_blind(&mut combined[..n_theta]);
            let (cold_obj, cold_exh, cold_mem) = eval_guard(combined, false);
            let (obj, exhausted, memory) = if !warm_obj.is_finite() || cold_obj <= warm_obj {
                params[..n_theta].copy_from_slice(&combined[..n_theta]);
                if cacheable {
                    guard_u[..k].copy_from_slice(&pirls.u[..k]);
                    guard_beta[..p].copy_from_slice(&beta_prof[..p]);
                }
                (cold_obj, cold_exh, cold_mem)
            } else {
                warm_kept = true;
                (warm_obj, warm_exh, warm_mem)
            };
            exact_prof.curv_memory = memory;
            if cacheable {
                guard_cache = Some((obj, exhausted));
                guard_beta_kept = true;
            }
        }
    }
    // A kept warm θ₀ runs stage 1 at `RHO_BEGIN_WARM` (see that constant's
    // doc). The joint search keeps its own radius either way: as a polish it
    // starts at stage 1's optimum, and on the `Joint` route a warm start
    // measured no loss at `GLMM_RHO_BEGIN`.
    // The gradient route (`gradient_route_plan`). Its result replaces the
    // BOBYQA search below, whose code runs only as a fallback:
    //
    // - a certified endpoint (`BfgsOutcome::certified`, and the saddle walk
    //   kept no try) is the fit's optimum;
    // - an endpoint that stopped on its convergence tests but is not
    //   certified is polished by the joint BOBYQA search, β whitened
    //   (`BetaScale`): at `GLMM_RHO_POLISH` (`solver`), as `PqlThenJoint`'s
    //   polish runs, on a Laplace shape; at `RHO_BEGIN_WARM`
    //   (`solver_polish_wide`) on AGQ (no stage 1; `solver` is the whole
    //   search and starts at `GLMM_RHO_BEGIN`). The polish's status is the
    //   fit's;
    // - any other stop (the gradient unavailable, the line search or the
    //   iteration cap) hands the incumbent to this shape's BOBYQA route with
    //   its unchanged budget: stage 1 at `RHO_BEGIN_WARM` where the route has
    //   one; where it has no stage 1 (AGQ, Laplace `Joint`) the joint search
    //   restarts at `GLMM_RHO_BEGIN`. A failed line search or the cap raise
    //   `Note::GradientSearchFallback`.
    //
    // On AGQ a fit this route finishes (a certified endpoint or the polish)
    // then goes through the joint search's check and second pass
    // (`JOINT_RECHECK_TOL`), as a joint BOBYQA pass does; a Laplace shape's
    // finish is the fit's result.
    //
    // `converged` is always the status of whichever optimizer finished.
    let mut grad_done: Option<(bool, usize, bool, f64)> = None;
    // AGQ fits search with no stage 1: the keys below read the order.
    let agq_finish = nagq > 1;
    let mut grad_spent = 0usize;
    let mut grad_fell_back = false;
    let total_dim = n_theta + p + n_disp;
    if grad_route {
        use crate::optim::bfgs::{bfgs_minimize, BfgsStop};
        debug_assert!(
            nagq == 1 || agq_gradient::covers_shape(family, nagq, *layout, groupings),
            "gradient route on an AGQ shape the exact AGQ gradient does not cover"
        );
        // A free Gamma dispersion is searched as the trailing `ψ = ln φ`,
        // from the seed above and on the BOBYQA route's box; on the inverse
        // link with its start curvature (`GradSearch::start_curvature`) unless
        // a hint is given. A held one leaves the search on `[θ | β̃]`.
        let free_phi = is_gamma && held_phi.is_none();
        let m = grad_search_dim(family, n_theta, p, held_phi.is_some());
        debug_assert!((n_disp == 0 || is_gamma) && params.len() == total_dim);
        if guard_beta_kept {
            // A warm fit: the guard profiled β at the θ it picked.
            params[n_theta..nb_col].copy_from_slice(&guard_beta[..p]);
        }
        // The search's start (length m), for a hand-over from the plateau
        // below, and the objective's last `[θ | β]` (length `n_θ + p`):
        // workspace scratch, so a refit allocates nothing here.
        grad_scratch.resize(m + nb_col, 0.0);
        let (grad_start, x_last) = grad_scratch.split_at_mut(m);
        grad_start.copy_from_slice(&params[..m]);
        x_last.fill(0.0);
        // β̃'s whitening. `BetaScale::set` whitens by the design and the
        // prior weights, which leaves the deviance's curvature along a β̃
        // axis at about 2·Σ(working weight), except along the coefficients
        // the random effects absorb: on four measured cells (2026-10-20) the
        // intercept's was 30 to 77 where 2·Σw was 5,800 to 81,000. So on the
        // caller's opt-in (`FitWorkspace::set_beta_info_scaling`), on a
        // Laplace fit without a caller's hint, one `ProfilePql` solve at the
        // start fills `border.schur` with the β information net of what the
        // random effects absorb, and `set_info` whitens by it: every β̃ axis
        // then has curvature about 2·Σ(prior weight), the value
        // `GradSearch::start_curvature` seeds. The solve starts from the
        // search's own first modes (`u_seed`) at the start's θ and β, and is
        // outside `n_eval`, as the start guard's solves are. A kept warm
        // start skips it: the guard's last solve was `ProfilePql` at this θ
        // and dispersion, PIRLS from û = 0 and β₀ rather than from `u_seed`
        // and β̂, so it stopped at the same modes and β̂ within PIRLS
        // tolerance and its border is the one this solve would leave. A solve
        // that is not finite, or an information `set_info` refuses, keeps
        // the design whitening and the identity start. A hinted fit keeps
        // them too, since its hint is in the design's coordinates; AGQ keeps
        // them because its seed was not measured on this whitening. It is
        // off by default because it costs cold fits more than it saves: on
        // 144 cold fits of data built unlike the accuracy grid it took 21%
        // more evaluations, against 2% to 6% fewer on warm ones (2026-10-21).
        beta_scale.set(x, &prior_w[..n], n);
        *grad_beta_info = false;
        let mut beta_inv_curv = None;
        if *beta_info_scaling && nagq == 1 && grad_hint.is_none() {
            let (t, lset) = grad_beta_map.split_at_mut(p * p);
            // `L_setᵀ` column by column, for the hint map below.
            for j in 0..p {
                beta_rhs[..p].fill(0.0);
                beta_rhs[j] = 1.0;
                beta_scale.to_search(&mut beta_rhs[..p]);
                for i in 0..p {
                    lset[i * p + j] = beta_rhs[i];
                }
            }
            let ((nb, phi), _) = if n_disp == 1 {
                disp_at(params[nb_col])
            } else {
                ((nb_theta, 1.0), None)
            };
            beta_prof[..p].copy_from_slice(&params[n_theta..nb_col]);
            pirls.u[..k].copy_from_slice(&u_seed[..k]);
            let mut info_exhausted = 0u32;
            let mut info_counters = crate::counters::EvalCounters::new();
            // A kept warm start's guard objective was finite.
            let finite = info_border_ready
                || laplace_deviance(
                    &data,
                    nb,
                    phi,
                    1,
                    &params[..n_theta],
                    beta_prof,
                    wx,
                    pirls,
                    structured,
                    pattern,
                    packed,
                    border,
                    prior_w_phi,
                    beta_rhs,
                    exact_prof,
                    BetaMode::ProfilePql,
                    None,
                    cluster_rows.as_ref(),
                    &mut info_exhausted,
                    &mut info_counters,
                )
                .is_finite();
            let sw: f64 = prior_w[..n].iter().sum();
            if finite && beta_scale.set_info(border.schur.as_mut(), sw) {
                *grad_beta_info = true;
                beta_inv_curv = Some(1.0 / (2.0 * sw));
                // T = L_setᵀ·L_info⁻ᵀ: column j is `L_setᵀ` applied to the
                // caller-units β of the information-whitened unit vector e_j.
                for j in 0..p {
                    beta_rhs[..p].fill(0.0);
                    beta_rhs[j] = 1.0;
                    beta_scale.to_caller(&mut beta_rhs[..p]);
                    for i in 0..p {
                        t[i * p + j] = (0..p).map(|c| lset[i * p + c] * beta_rhs[c]).sum();
                    }
                }
            } else {
                beta_scale.set(x, &prior_w[..n], n);
            }
        }
        beta_scale.to_search(&mut params[n_theta..nb_col]);
        // AGQ: the gradient's row index and buffers, built on the first such
        // fit and kept for the SE step's gradient-difference Hessian; the index
        // is rebuilt per fit, since a reused workspace may see new ids.
        if nagq > 1 && !agq_gd.as_ref().is_some_and(|g| g.matches_shape(p, nagq)) {
            *agq_gd = Some(Box::new(agq_gradient::AgqGd::for_shape(p, nagq)));
        }
        let mut agq_gd = agq_gd.as_deref_mut().filter(|_| nagq > 1);
        if let Some(gd) = agq_gd.as_deref_mut() {
            gd.idx.rebuild(&cluster_ids[..n], groupings.n_primary);
        }
        let mut obj = GradSearch {
            data: &data,
            nb_theta,
            nagq,
            cluster_rows: cluster_rows.as_ref(),
            agq_gd,
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
            pirls_exhausted,
            counters,
            non_finite_points,
            grad_asm,
            u_inc: &mut u_seed[..],
            beta_scale,
            x_last,
            last_ok: false,
            gamma_phi: held_phi.unwrap_or(1.0),
            ln_phi: held_phi.map_or(0.0, f64::ln),
            free_phi,
            sum_ln_y: gamma_sum_ln_y,
            beta_inv_curv,
            k,
            n_theta,
        };
        let bo = bfgs_minimize(
            &mut obj,
            &mut params[..m],
            &lower[..m],
            &upper[..m],
            grad_cfg,
            bfgs,
            grad_hint
                .as_ref()
                .filter(|h| h.dim() == m)
                .map(|h| h.inv_hessian()),
        );
        *grad_hint_ready = bo.n_grad >= 2;
        let mut n_val = bo.n_val;
        let mut best = bo.f;
        let mut moved = false;
        if bo.converged() {
            let (mv, tries) = obj.saddle_walk(groupings, &mut params[..m], &mut best);
            moved = mv;
            n_val += tries;
        }
        // An endpoint with a diagonal θ out at `GRAD_THETA_FAR` sits on the
        // deviance's plateau toward the box, where the gradient is nearly flat
        // and BFGS can stop far from the optimum while every test passes (a
        // logit crossed fit from θ₀ = THETA_HI ended 2.09 above the cold
        // optimum, certified: `far_warm_start_scan`, 2026-10-07). The BOBYQA
        // route reaches it from there, so such an endpoint is handed over.
        let far = groupings
            .diagonal_theta()
            .iter()
            .any(|&i| params[i].abs() >= GRAD_THETA_FAR);
        let rung = if far {
            1
        } else if !moved && bo.certified(&params[..m], groupings.diagonal_theta()) {
            0
        } else if bo.converged() {
            3
        } else {
            1
        };
        *grad_report = Some((bo.stop, bo.n_grad, bo.decrement, rung));
        if rung == 0 {
            grad_done = Some((true, n_val, false, best));
        } else if rung == 3 {
            // Rung 3: the joint polish from the endpoint. Its own points alone
            // are judged by the `NON_FINITE_RADIUS` check. The joint solver is
            // sized `total_dim`, so under a held φ it gets the inert `ln φ`
            // slot too (the objective ignores it): on a shorter vector BOBYQA
            // refuses to start. On AGQ (no stage 1; `solver` is the whole
            // search) `solver` would restart it at `GLMM_RHO_BEGIN`, so the
            // polish takes `solver_polish_wide` there (see the rungs above).
            obj.non_finite_points.clear();
            let mut finite = 0usize;
            let mut pol_best = best;
            let polish: &mut bobyqa::Bobyqa = if agq_finish {
                solver_polish_wide.get_or_insert_with(|| {
                    let mut c = *config;
                    c.rho_begin = RHO_BEGIN_WARM;
                    bobyqa::Bobyqa::new(total_dim, c)
                        .expect("polish BOBYQA config is valid by construction")
                })
            } else {
                &mut *solver
            };
            let out = crate::optim::bobyqa_minimize(
                polish,
                &mut crate::optim::ValueFn(|gamma: &[f64], _rho: f64| {
                    let f = crate::optim::Objective::value(&mut obj, gamma, 0.0)
                        .unwrap_or(f64::INFINITY);
                    if f.is_finite() {
                        finite += 1;
                    }
                    if f < pol_best {
                        pol_best = f;
                        obj.u_inc[..k].copy_from_slice(&obj.pirls.u[..k]);
                    }
                    f
                }),
                &mut params[..total_dim],
                &lower[..total_dim],
                &upper[..total_dim],
            );
            let ok = matches!(out.status, Status::Converged | Status::MaxFunReached)
                && pol_best.is_finite()
                && finite >= 2
                && !(out.status == Status::Converged
                    && any_within(
                        obj.non_finite_points,
                        &params[..total_dim],
                        NON_FINITE_RADIUS,
                    ));
            grad_done = Some((
                ok,
                n_val + out.n_eval,
                matches!(out.status, Status::MaxFunReached),
                pol_best,
            ));
        } else {
            // Rungs 1 and 2: the BOBYQA route from the incumbent (the start
            // itself when its value was not finite).
            obj.non_finite_points.clear();
            grad_spent = n_val;
            grad_fell_back = true;
            // The Note marks a search that failed (its line search or its
            // iteration cap). Where the gradient is not defined (a clamped η
            // row under quasi-separation, a non-PD observed factor), where the
            // start has no finite value, and on the plateau toward the box,
            // the BOBYQA route is the designed handler and the hand-over is
            // silent: a 20-row quasi-separated binomial would otherwise carry
            // a caution the BOBYQA route never raised.
            *gradient_search_fallback =
                !far && matches!(bo.stop, BfgsStop::LineSearch | BfgsStop::MaxIter);
        }
        // The dispersion-bound check of the joint search below (which
        // explains it), on a converged endpoint with a free Gamma φ: one solve
        // at ψ's lower bound, every other coordinate held, kept when lower.
        // Counted as an evaluation, as the saddle walk's tries are; its PIRLS
        // exhaustion is not counted, as the joint check's is not.
        if let Some(done) = grad_done.as_mut().filter(|d| free_phi && d.0 && !d.2) {
            let psi = params[nb_col];
            params[nb_col] = lower[nb_col];
            done.1 += 1;
            let exhausted = *obj.pirls_exhausted;
            match crate::optim::Objective::value(&mut obj, &params[..m], 0.0) {
                Some(f) if f < done.3 => {
                    done.3 = f;
                    obj.u_inc[..k].copy_from_slice(&obj.pirls.u[..k]);
                }
                _ => params[nb_col] = psi,
            }
            *obj.pirls_exhausted = exhausted;
        }
        beta_scale.to_caller(&mut params[n_theta..nb_col]);
        if far {
            // From the plateau the BOBYQA route restarts where the search
            // started, the guard's pick, not at the endpoint out on it.
            params[..m].copy_from_slice(grad_start);
        }
        // The search's `f64` assembly scratch is the size of an SE step's
        // `G_γ` on large designs; past `GRAD_ASM_KEEP_BYTES` it is released
        // before the SE step allocates its own.
        if grad_asm
            .as_ref()
            .is_some_and(|a| a.f64_bytes() > GRAD_ASM_KEEP_BYTES)
        {
            *grad_asm = None;
        }
    }
    // A gradient search that handed over restarts stage 1 near its incumbent
    // (`RHO_BEGIN_WARM`), as a kept warm start does; where the route has no
    // stage 1 (AGQ, Laplace `Joint`) the joint search restarts from it at
    // `GLMM_RHO_BEGIN`.
    let solver_stage1: &mut bobyqa::Bobyqa = if warm_kept || grad_fell_back {
        solver_stage1_warm.get_or_insert_with(|| {
            let mut c = *config_stage1;
            c.rho_begin = RHO_BEGIN_WARM;
            bobyqa::Bobyqa::new(params_stage1.len(), c)
                .expect("warm BOBYQA config is valid by construction")
        })
    } else {
        solver_stage1
    };

    // The point the search below begins from (after the start guard above).
    // The `PqlThenJoint` fallback restarts here rather than from wherever a
    // failed `ExactProfile` attempt left `params`: the scan behind the
    // fallback found `PqlThenJoint` reaches the same optimum from every start
    // it was tried from, so restarting from the ORIGINAL point is exactly as
    // good as any other and never rewards the failed attempt's own excursion.
    start_point[..total_dim].copy_from_slice(&params[..total_dim]);

    // `PqlThenJoint` fallback (`Note::ExactProfileFallback`): an `ExactProfile`
    // attempt that ends not converged reruns once on `PqlThenJoint`, from the
    // same restart point. The loop below runs the search body once (the common
    // case) or twice (one retry — `fallback_used` latches so a second failure
    // never retries again). `attempt1` holds the first attempt's full state
    // (params/û, snapshotted before the restart resets them for the retry) so
    // it can be restored if the retry does no better.
    let mut fallback_used = false;
    #[allow(clippy::type_complexity)]
    let mut attempt1: Option<(Vec<f64>, Vec<f64>, bool, usize, bool, f64)> = None;
    // On AGQ (no stage 1; `solver` is the whole search) a gradient-route
    // finish takes the joint search's first pass: the loop below reads it in
    // place of running BOBYQA, then runs that search's check and second pass
    // on it.
    let mut grad_finish = if agq_finish { grad_done.take() } else { None };
    let (ok, n_eval, budget_exhausted, inc_obj) = if let Some(done) = grad_done {
        done
    } else {
        loop {
            // Reset the search's starting point on every pass: on the first,
            // `params` already holds it and the modes go back to û = 0 (a
            // gradient search's hand-over included); on a retry it lets the
            // fallback restart clean rather than from the failed `ExactProfile`
            // attempt's own incumbent. A gradient-route finish keeps its
            // endpoint and incumbent modes.
            if grad_finish.is_none() {
                params[..total_dim].copy_from_slice(&start_point[..total_dim]);
                for v in u_seed[..k].iter_mut() {
                    *v = 0.0;
                }
            }

            // STAGE 1 — θ-only BOBYQA profiling β out of PIRLS at each candidate θ. On
            // `PqlThenJoint` this is an accelerant that warm-starts the `Joint` polish below
            // and never gates convergence; on `ExactProfile` this IS the search — its
            // status alone gates convergence (the `route ==` split after this block).
            // Skipped bit-identically when `stage1_mode` is `None` (the `Joint` route; the
            // A/B tests pin this as the single-stage reference). A Laplace-pass warm start
            // for AGQ was measured on the 33 diligent AGQ cells (2026-07-14) and reverted:
            // total eval count was a wash (−0.3%), not worth the added code path for that
            // little a gain.
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
            //
            // A finite incumbent can still be an artefact of the +∞ points. A trust-region
            // step that lands on one fails, so BOBYQA shrinks its radius (Powell 2009)
            // until it reaches `rho_end` at the edge of the region where PIRLS fails, and
            // reports `Converged` at a point that is only a minimum of the moderated
            // surface. `Converged` is therefore also refused when a non-finite
            // evaluation lies within `NON_FINITE_RADIUS` of the returned point: the
            // search could not compare that point with its neighbours on that scale.
            // Evaluations the search walked away from (a far warm start, a failed
            // first step) leave the check silent.
            let mut finite_evals1 = 0usize;
            if let Some(mode) = stage1_mode {
                params_stage1[..n_theta].copy_from_slice(&params[..n_theta]); // θ₀ unchanged from the caller's params
                if n_disp == 1 {
                    params_stage1[n_theta] = params[nb_col]; // ln θ_NB / ln φ start, set before the search
                }
                beta_prof[..p].copy_from_slice(&params[n_theta..n_theta + p]); // β₀ (GLM warm start)
                beta_seed[..p].copy_from_slice(&beta_prof[..p]);
                // The incumbent state (best value, finite-evaluation count and the
                // û/β̂ snapshots) is passed in rather than captured so the closure
                // borrows only the PIRLS buffers.
                // On `ExactProfile`, each evaluation's PIRLS tolerance follows the
                // trust radius (`RHO_TOL_C`); `inc_tol` is the tolerance the
                // incumbent was solved at, read by the re-solve after the search.
                // The first evaluation, at the search's start point, is always
                // solved at `pirls_tol`, as the start guard solves that point: a
                // warm fit's handed-over guard solve and a cold fit's own first
                // solve are then the same computation, and a warm fit the guard
                // sends to the blind start still runs the cold fit's search.
                let tol_full = pirls_tol(family);
                let rho_tol = mode == BetaMode::ProfileExact;
                let full_below = RHO_TOL_FULL_FACTOR * crate::lmm::GLMM_RHO_END;
                let mut inc_tol = tol_full;
                let mut first_eval = true;
                let mut stage1_obj = |theta: &[f64],
                                      rho: f64,
                                      best: &mut f64,
                                      finite: &mut usize,
                                      non_finite: &mut Vec<f64>,
                                      u_seed: &mut [f64],
                                      beta_seed: &mut [f64],
                                      cache: &mut Option<(f64, u32)>,
                                      memo: &mut Vec<DispProbe>,
                                      record: bool|
                 -> f64 {
                    // A point the dispersion probes already solved hands back
                    // that solve (objective, û, β̂); its incumbent bookkeeping ran
                    // when it was probed.
                    if let Some(i) = memo.iter().position(|m| {
                        m.0.iter()
                            .zip(theta)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    }) {
                        let (_, obj, u, b) = memo.swap_remove(i);
                        first_eval = false;
                        pirls.u[..k].copy_from_slice(&u);
                        beta_prof[..p].copy_from_slice(&b);
                        return obj;
                    }
                    let tol = if rho_tol && rho > full_below && !first_eval {
                        (RHO_TOL_C * rho * rho).clamp(tol_full, RHO_TOL_MAX)
                    } else {
                        tol_full
                    };
                    first_eval = false;
                    // The start guard's hand-over: the first call, at exactly the
                    // point the guard picked (θ and the dispersion coordinate, bit
                    // for bit), takes the guard's solve of that point instead of
                    // repeating it. BOBYQA clamps a start into its box and moves
                    // one that lies within its initial radius of a bound onto the
                    // bound or one radius inside it, so the first call can miss
                    // the guard's point (a θ₀ above the θ box, an NB `ln θ_NB`
                    // start just under `ln NB_THETA_HI`); that call then
                    // solves its point from a fit's first-evaluation state, the
                    // curvature memory cleared as a cold fit's first solve has it.
                    let cached = match cache.take() {
                        Some(c)
                            if theta[..n_theta]
                                .iter()
                                .zip(&start_point[..n_theta])
                                .all(|(a, b)| a.to_bits() == b.to_bits())
                                && (n_disp == 0
                                    || theta[n_theta].to_bits()
                                        == start_point[nb_col].to_bits()) =>
                        {
                            Some(c)
                        }
                        Some(_) => {
                            exact_prof.curv_memory = false;
                            None
                        }
                        None => None,
                    };
                    let obj = if let Some((cobj, cexh)) = cached {
                        pirls.u[..k].copy_from_slice(&guard_u[..k]);
                        beta_prof[..p].copy_from_slice(&guard_beta[..p]);
                        *pirls_exhausted += cexh;
                        cobj
                    } else {
                        // Re-seed BOTH latent states from the INCUMBENT (best point so far),
                        // not the last-evaluated: BOBYQA's final call is not its best.
                        // û is point-determined given θ, and β is likewise point-determined
                        // (the PQL β̂(θ)); the incumbent seed only shifts the stopping iterate
                        // within tol, the same argument that justifies the u_seed warm start.
                        pirls.u[..k].copy_from_slice(&u_seed[..k]);
                        beta_prof[..p].copy_from_slice(&beta_seed[..p]);
                        // Profile mode SWAPS the β buffers vs the Fixed stage-2 call below:
                        // `beta = beta_prof` (in/out profiled β), `beta_step_rhs = beta_rhs`
                        // (the δβ border scratch). See deviance.rs's buffer-role comment.
                        let ((nb, phi), coord) = if n_disp == 1 {
                            disp_at(theta[n_theta])
                        } else {
                            ((nb_theta, 1.0), None)
                        };
                        let dev = laplace_deviance(
                            &data,
                            nb,
                            phi,
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
                            prior_w_phi,
                            beta_rhs,
                            exact_prof,
                            // `mode` is `ProfilePql` on `PqlThenJoint`, `ProfileExact` on
                            // `ExactProfile`: the two routes that reach this block.
                            mode,
                            // The radius-tied tolerance while ρ is large on
                            // `ExactProfile`, `pirls_tol` (`None`) otherwise; never
                            // the FD-pass tolerance.
                            (tol > tol_full).then_some(tol),
                            cluster_rows.as_ref(),
                            pirls_exhausted,
                            counters,
                        );
                        match coord {
                            Some(c) => dev + nb_term(c),
                            None => dev,
                        }
                    };
                    if obj.is_finite() {
                        *finite += 1;
                    } else {
                        non_finite.extend_from_slice(theta);
                    }
                    if obj < *best {
                        *best = obj;
                        // INCUMBENT-gated snapshot: snapshot only on strict
                        // improvement, NOT every eval.
                        u_seed[..k].copy_from_slice(&pirls.u[..k]);
                        beta_seed[..p].copy_from_slice(&beta_prof[..p]);
                        inc_tol = tol;
                    }
                    counters.record_eval(crate::counters::Stage::One, obj);
                    if record {
                        memo.push((
                            theta.to_vec(),
                            obj,
                            pirls.u[..k].to_vec(),
                            beta_prof[..p].to_vec(),
                        ));
                    }
                    obj
                };
                // The dispersion coordinate's scale. Two probe solves
                // at `ln φ ± h` (h = the start radius) give its curvature `c` and
                // slope `g` at the start; when the conditional optimum of `ln φ`,
                // `x₀ − g/c`, lies within one cold radius of the plug-in seed (the
                // plug-in seed charges the random-effect variance to φ, so this
                // says the dispersion is decoupled from θ), the
                // search runs in `x = S·z` with S = diag(1, …, 1, s), s a power of
                // two near `sqrt(DISP_SCALE_C_REF/c)` in [DISP_SCALE_MIN, 1], fixed for the run.
                // The probes count as evaluations; a later call at a probed point
                // takes its solve from `memo` instead of solving it again.
                let scale_on = mode == BetaMode::ProfileExact && is_gamma && held_phi.is_none();
                let n1 = params_stage1.len();
                let mut memo: Vec<DispProbe> = Vec::new();
                let out1 = if scale_on {
                    let h = if warm_kept {
                        RHO_BEGIN_WARM
                    } else {
                        config_stage1.rho_begin
                    };
                    // The gate asks the cold fit's question on every fit: does the
                    // plug-in seed lie within one cold radius
                    // (`config_stage1.rho_begin`) of the conditional optimum? From
                    // the plug-in seed (a cold fit, or a warm θ without a carried
                    // φ) that is `|g|/c`; from any other `ln φ` (a carried φ, or the
                    // incumbent of a gradient search that handed over) it is the
                    // distance from `ln_phi_plug`. Two other readings failed.
                    // Gating on the search's own radius halved the gate on a warm θ
                    // and rejected a seed the cold fit accepts (`gami_q4` from a 5%
                    // perturbed optimum: 1578 evaluations against 364 cold, 368 on
                    // the cold radius). Gating a carried φ (`StartValues::dispersion`)
                    // on its own distance passes it whenever it is close, which it
                    // is by construction, also where the dispersion is coupled to θ
                    // and scaling slows the search (`gaml_q2s` from a carried φ̂:
                    // 94 evaluations scaled, 44 unscaled, 52 cold). The probes
                    // stay at `±h`, so on a kept warm θ BOBYQA's first design still
                    // meets them. After a gradient search hands over, BOBYQA starts
                    // at the warm radius but the probes keep the cold one: on the
                    // 64 hand-overs of the Gamma gate cells (both links, caps 1 to
                    // 8 iterations) that took 5673 evaluations, against 5781 with
                    // probes at the warm radius.
                    let h_cold = config_stage1.rho_begin;
                    let j = n1 - 1;
                    let x0: Vec<f64> = params_stage1.to_vec();
                    let mut xp = x0.clone();
                    let mut n_probe = 0usize;
                    let mut fv = [f64::NAN; 3];
                    let mut best_pt = x0.clone();
                    let mut best_f = f64::INFINITY;
                    for (i, d) in [0.0, h, -h].into_iter().enumerate() {
                        xp[j] = x0[j] + d;
                        if xp[j] < lower_stage1[j] || xp[j] > upper_stage1[j] {
                            continue;
                        }
                        let f = stage1_obj(
                            &xp,
                            0.0,
                            &mut best1,
                            &mut finite_evals1,
                            non_finite_points,
                            &mut u_seed[..],
                            &mut beta_seed[..],
                            &mut guard_cache,
                            &mut memo,
                            true,
                        );
                        n_probe += 1;
                        fv[i] = f;
                        if f < best_f {
                            best_f = f;
                            best_pt.copy_from_slice(&xp);
                        }
                    }
                    let c = (fv[1] + fv[2] - 2.0 * fv[0]) / (h * h);
                    let g = (fv[1] - fv[2]) / (2.0 * h);
                    let mut s = 1.0f64;
                    let decoupled = if ws.gamma_phi_start.is_some() || grad_fell_back {
                        (ln_phi_plug - (x0[j] - g / c)).abs() <= DISP_SCALE_GATE * h_cold
                    } else {
                        g.abs() <= DISP_SCALE_GATE * h_cold * c
                    };
                    if c.is_finite() && c > 0.0 && g.is_finite() && decoupled {
                        let r = (DISP_SCALE_C_REF / c).sqrt().clamp(DISP_SCALE_MIN, 1.0);
                        s = 2f64.powi(r.log2().round() as i32);
                    }
                    // Scaled: start at the best probe point. Unscaled: start at
                    // the original point so BOBYQA's initial design meets the
                    // probes. Both first calls are memo hits.
                    if s != 1.0 {
                        params_stage1.copy_from_slice(&best_pt);
                        params_stage1[j] /= s;
                    }
                    let mut lo_z = lower_stage1.to_vec();
                    let mut up_z = upper_stage1.to_vec();
                    lo_z[j] /= s;
                    up_z[j] /= s;
                    let mut xb = vec![0.0f64; n1];
                    let mut hits = 0usize;
                    let o = crate::optim::bobyqa_minimize(
                        solver_stage1,
                        &mut crate::optim::ValueFn(|z: &[f64], rho: f64| {
                            xb.copy_from_slice(z);
                            xb[j] = z[j] * s;
                            let before = memo.len();
                            let f = stage1_obj(
                                &xb,
                                rho,
                                &mut best1,
                                &mut finite_evals1,
                                non_finite_points,
                                &mut u_seed[..],
                                &mut beta_seed[..],
                                &mut guard_cache,
                                &mut memo,
                                false,
                            );
                            hits += before - memo.len();
                            f
                        }),
                        params_stage1,
                        &lo_z,
                        &up_z,
                    );
                    params_stage1[j] *= s;
                    // A probe that stayed the incumbent (never beaten by the
                    // search) is the reported point: its solve is the snapshot.
                    if best1.to_bits() == best_f.to_bits() {
                        params_stage1.copy_from_slice(&best_pt);
                    }
                    let mut o = o;
                    o.n_eval = o.n_eval + n_probe - hits;
                    o
                } else {
                    crate::optim::bobyqa_minimize(
                        solver_stage1,
                        &mut crate::optim::ValueFn(|theta: &[f64], rho: f64| {
                            stage1_obj(
                                theta,
                                rho,
                                &mut best1,
                                &mut finite_evals1,
                                non_finite_points,
                                &mut u_seed[..],
                                &mut beta_seed[..],
                                &mut guard_cache,
                                &mut memo,
                                false,
                            )
                        }),
                        params_stage1,
                        lower_stage1,
                        upper_stage1,
                    )
                };
                // Saddle check on the `ExactProfile` endpoint. Σ = ΛΛᵀ is a sum of
                // `l_d·l_dᵀ` over the columns of Λ, so the deviance is even in each
                // column and a zero column is a stationary point; where a small
                // nonzero variance fits better, it is a saddle. Near it the deviance
                // changes as ‖l_d‖⁴ along the column and its slope as ‖l_d‖³, so a
                // search that reaches it stops there: on the Poisson cell
                // `pois_q2sx2_g3000p20_bal_boundary` of the accuracy grid it stopped
                // at 0.15 of the optimal column, 4.0e-6 deviance above the optimum
                // (0.96 of the dip). Its own iterate, coming down the slope, points
                // along the escape direction, so each nonzero column below
                // `POLISH_NEAR_SINGULAR` is doubled while that lowers the deviance
                // (a power of two, so undoing a failed try restores it bit for bit),
                // and a search that moved restarts there at `GLMM_RHO_POLISH`.
                // Every point is solved at `pirls_tol` (ρ = 0 in `stage1_obj`) and
                // counts as an evaluation; nothing is kept that is not lower. A
                // doubled point that is not finite is only a failed try: it is not
                // recorded for the `NON_FINITE_RADIUS` check, which judges the
                // searches' own points.
                let mut out1 = out1;
                if mode == BetaMode::ProfileExact && out1.status == Status::Converged {
                    let mut moved = false;
                    let mut start = 0usize;
                    // Same block and column walk as `crate::lmm::fix_column_signs`
                    // — change together.
                    for &q in std::iter::once(&groupings.primary_q).chain(groupings.extra_q.iter())
                    {
                        for d in 0..q {
                            let col = start..start + (q - d);
                            start = col.end;
                            loop {
                                let norm_sq: f64 =
                                    params_stage1[col.clone()].iter().map(|v| v * v).sum();
                                if norm_sq == 0.0
                                    || 4.0 * norm_sq > POLISH_NEAR_SINGULAR * POLISH_NEAR_SINGULAR
                                {
                                    break;
                                }
                                for v in params_stage1[col.clone()].iter_mut() {
                                    *v *= 2.0;
                                }
                                let prev = best1;
                                stage1_obj(
                                    params_stage1,
                                    0.0,
                                    &mut best1,
                                    &mut finite_evals1,
                                    &mut Vec::new(),
                                    &mut u_seed[..],
                                    &mut beta_seed[..],
                                    &mut guard_cache,
                                    &mut Vec::new(),
                                    false,
                                );
                                out1.n_eval += 1;
                                if best1 < prev {
                                    moved = true;
                                } else {
                                    for v in params_stage1[col.clone()].iter_mut() {
                                        *v *= 0.5;
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if moved {
                        let polish = solver_stage1_polish.get_or_insert_with(|| {
                            let mut c = *config_stage1;
                            c.rho_begin = GLMM_RHO_POLISH;
                            bobyqa::Bobyqa::new(params_stage1.len(), c)
                                .expect("polish BOBYQA config is valid by construction")
                        });
                        let out2 = crate::optim::bobyqa_minimize(
                            polish,
                            &mut crate::optim::ValueFn(|theta: &[f64], rho: f64| {
                                stage1_obj(
                                    theta,
                                    rho,
                                    &mut best1,
                                    &mut finite_evals1,
                                    non_finite_points,
                                    &mut u_seed[..],
                                    &mut beta_seed[..],
                                    &mut guard_cache,
                                    &mut Vec::new(),
                                    false,
                                )
                            }),
                            params_stage1,
                            lower_stage1,
                            upper_stage1,
                        );
                        out1.n_eval += out2.n_eval;
                        out1.status = out2.status;
                    }
                }
                // The dispersion-bound check of the joint search below, on
                // `ExactProfile` with β profiled: Gamma's lower bound of `ln φ`,
                // NB's upper bound of `ln θ_NB`. Solved at `pirls_tol` and counted
                // as an evaluation, as the saddle check's tries are.
                if mode == BetaMode::ProfileExact
                    && n_disp == 1
                    && held_phi.is_none()
                    && out1.status == Status::Converged
                {
                    let ln_theta = params_stage1[n_theta];
                    let prev = best1;
                    params_stage1[n_theta] = if is_gamma {
                        lower_stage1[n_theta]
                    } else {
                        upper_stage1[n_theta]
                    };
                    stage1_obj(
                        params_stage1,
                        0.0,
                        &mut best1,
                        &mut finite_evals1,
                        &mut Vec::new(),
                        &mut u_seed[..],
                        &mut beta_seed[..],
                        &mut guard_cache,
                        &mut Vec::new(),
                        false,
                    );
                    out1.n_eval += 1;
                    if best1 >= prev {
                        params_stage1[n_theta] = ln_theta;
                    }
                }
                // An incumbent solved at a loose tolerance is solved again at
                // `pirls_tol` from its own (û, β̂) before β̂ and the objective are
                // read off it, so the reported fit never rests on a loose solve.
                // Not a search evaluation (`n_eval` unchanged); a re-solve that
                // comes back non-finite keeps the loose one.
                if inc_tol > tol_full && best1.is_finite() {
                    pirls.u[..k].copy_from_slice(&u_seed[..k]);
                    beta_prof[..p].copy_from_slice(&beta_seed[..p]);
                    let ((nb, phi), coord) = if n_disp == 1 {
                        disp_at(params_stage1[n_theta])
                    } else {
                        ((nb_theta, 1.0), None)
                    };
                    let mut resolve_counters = crate::counters::EvalCounters::new();
                    let dev = laplace_deviance(
                        &data,
                        nb,
                        phi,
                        nagq,
                        &params_stage1[..n_theta],
                        beta_prof,
                        wx,
                        pirls,
                        structured,
                        pattern,
                        packed,
                        border,
                        prior_w_phi,
                        beta_rhs,
                        exact_prof,
                        mode,
                        None,
                        cluster_rows.as_ref(),
                        pirls_exhausted,
                        &mut resolve_counters,
                    );
                    let obj = match coord {
                        Some(c) => dev + nb_term(c),
                        None => dev,
                    };
                    if obj.is_finite() {
                        best1 = obj;
                        u_seed[..k].copy_from_slice(&pirls.u[..k]);
                        beta_seed[..p].copy_from_slice(&beta_prof[..p]);
                    }
                }
                // On `PqlThenJoint`, non-convergence does NOT fail the fit — stage 1 is an
                // accelerant, and the joint polish below proceeds from wherever the
                // incumbent landed (worst case = the cold start). On `ExactProfile`
                // this status IS the fit's convergence status — read below.
                n_eval_stage1 = out1.n_eval;
                // Warm-start / final θ̂,β̂: θ̂₁ (BOBYQA leaves the incumbent in `params_stage1`)
                // and β̂₁ (the incumbent snapshot, NOT `beta_prof`'s last-evaluated value).
                // On `ExactProfile` this write IS the reported optimum, not a warm start.
                params[..n_theta].copy_from_slice(&params_stage1[..n_theta]);
                params[n_theta..nb_col].copy_from_slice(&beta_seed[..p]);
                if n_disp == 1 {
                    params[nb_col] = params_stage1[n_theta];
                }
                stage1_out = Some(out1);
            }

            // `budget_exhausted` drives the plateau policy mirrored from `fit_lmm`
            // (`src/lmm/mod.rs`): a
            // `Status::MaxFunReached` exit reports its finite incumbent below with
            // `converged == false` and `boundary_hit == 2` rather than NaN-filling.
            let budget_exhausted;
            let (ok, n_eval, final_obj) = if route == OuterSearch::ExactProfile {
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
                        && finite_evals1 >= 2
                        && !(o.status == Status::Converged
                            && any_within(non_finite_points, params_stage1, NON_FINITE_RADIUS)),
                    o.n_eval,
                    best1,
                )
            } else {
                // The joint search runs once, or twice when the check after it
                // finds a lower point (`JOINT_RECHECK_TOL`); the second pass
                // starts there.
                //
                // An AGQ fit runs one more information-metric pass after a
                // second pass, from the check's re-solved β at the second pass's
                // θ and at `GLMM_RHO_BEGIN`, and keeps the lower of the two.
                // Where working weights span many decades (|f| near 1e10) the
                // deviance's rounding noise defeats BOBYQA's local model at small
                // radii, so a pass ends where its start leads it; the fresh pass
                // recovers most of those ends. Measured 2026-10-11 on 125 decade
                // designs, hand-over fits: failures 31 to 16 on x86_64 and 30 to
                // 17 on aarch64, none worse, about 29 evaluations more per fit.
                // A fit the gradient route finishes through its polish needs it
                // as much: which of the two routes a decade design takes follows
                // the platform's rounding, and on aarch64 two of the fourteen
                // decades test's fits ended their second pass 1.65 and 8.77
                // units high on the polish route. Measured 2026-10-18 on 420
                // decade designs (60 seeds, 8 to 14 decades): fits more than one
                // unit above the exact profile 16 to 3 on aarch64 and 17 to 4 on
                // x86_64, none worse beyond 6e-6 units, no convergence flag
                // flipped, about 40 evaluations more where it runs.
                let mut info_pass = false;
                let mut info_passes = 0usize;
                let max_info_passes = if agq_finish { 2 } else { 1 };
                let mut n_eval_prev = 0usize;
                let mut first_obj: Option<f64> = None;
                let (read, exhausted) = loop {
                    // A gradient-route finish is the first pass's result as is:
                    // its endpoint is already in the caller's β units.
                    let (read, exhausted, pass_n_eval) = if let Some((ok, ne, ex, f)) =
                        grad_finish.take()
                    {
                        ((ok, ne, f), ex, ne)
                    } else {
                        let mut best_obj = f64::INFINITY;
                        // mirrors the stage-1 read above; change together.
                        let mut finite_evals2 = 0usize;
                        // Stage 1 recorded its own points; on this route it never gates
                        // convergence, so only this search's non-finite points are checked.
                        non_finite_points.clear();
                        // The search runs on β̃ (`BetaScale`): the start maps forward here, the
                        // incumbent back once the convergence read below has measured its
                        // distances in the search's own coordinates.
                        if !info_pass {
                            beta_scale.set(x, &prior_w[..n], n);
                        }
                        beta_scale.to_search(&mut params[n_theta..nb_col]);
                        // A polish whose stage-1 optimum has a block near the singular
                        // boundary starts wider (`POLISH_NEAR_SINGULAR`).
                        let near_singular = !info_pass
                            && stage1_out.is_some()
                            && groupings
                                .diagonal_theta()
                                .iter()
                                .any(|&i| params[i].abs() < POLISH_NEAR_SINGULAR);
                        let solver: &mut bobyqa::Bobyqa = if near_singular {
                            solver_polish_wide.get_or_insert_with(|| {
                                let mut c = *config;
                                c.rho_begin = RHO_BEGIN_WARM;
                                bobyqa::Bobyqa::new(total_dim, c)
                                    .expect("polish BOBYQA config is valid by construction")
                            })
                        } else {
                            &mut *solver
                        };
                        let out = crate::optim::bobyqa_minimize(
                            solver,
                            &mut crate::optim::ValueFn(|gamma: &[f64], _rho: f64| {
                                // Within-fit û warm-start: seed PIRLS from the incumbent (best point so
                                // far), not from 0. The conditional mode is point-determined, so the seed
                                // only shifts the stopping iterate within the PIRLS exit band.
                                pirls.u[..k].copy_from_slice(&u_seed[..k]);
                                let ((nb, phi), coord) = if n_disp == 1 {
                                    disp_at(gamma[nb_col])
                                } else {
                                    ((nb_theta, 1.0), None)
                                };
                                let dev = laplace_deviance(
                                    &data,
                                    nb,
                                    phi,
                                    nagq,
                                    beta_scale.caller_point(gamma, n_theta, p),
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
                                    prior_w_phi,
                                    beta_prof,
                                    exact_prof,
                                    BetaMode::Fixed,
                                    // Never the FD-pass tol here: BOBYQA objective evals stay at
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
                                } else {
                                    non_finite_points.extend_from_slice(gamma);
                                }
                                if obj < best_obj {
                                    best_obj = obj;
                                    u_seed[..k].copy_from_slice(&pirls.u[..k]);
                                }
                                counters.record_eval(crate::counters::Stage::Two, obj);
                                obj
                            }),
                            params,
                            lower,
                            upper,
                        );
                        // Dispersion bound check. A likelihood can keep rising toward one
                        // bound of the dispersion coordinate: Gamma's lower bound of `ln φ`
                        // (data the mean model reproduces exactly) and NB's upper bound of
                        // `ln θ_NB` (no overdispersion, the Poisson limit). Toward the other
                        // bound it falls to −∞ on any data with a positive response, so
                        // only these two are checked. Near the Gamma bound the curvature in
                        // β grows as 1/φ, and near the NB bound the slope in `ln θ_NB`
                        // shrinks as 1/θ_NB, so the trust radius can reach `rho_end` while
                        // the coordinate is still walking out, and BOBYQA reports
                        // `Converged` short of the bound. On simulated Poisson data with a
                        // random intercept and slope, 4 NB fits in 80 on the exact-profile
                        // search stopped at θ_NB 4.0e3 to 6.4e3, up to 4.7e-5 deviance
                        // above the bound, and 2 in 40 on this search (an extra crossed
                        // grouping with a slope) at 7.7e3 and 9.0e3. One solve at the bound, every other coordinate held, tells
                        // the two apart: when it is lower, the incumbent moves there (for
                        // Gamma the on-bound refusal below then applies). At an interior
                        // optimum the bound point is worse and nothing moves. Not a search
                        // evaluation: own counters, `n_eval` unchanged. The gradient
                        // route runs the same check on its endpoint; change together.
                        if n_disp == 1 && held_phi.is_none() && out.status == Status::Converged {
                            let ln_phi = params[nb_col];
                            params[nb_col] = if is_gamma {
                                lower[nb_col]
                            } else {
                                upper[nb_col]
                            };
                            pirls.u[..k].copy_from_slice(&u_seed[..k]);
                            let ((nb, phi), _) = disp_at(params[nb_col]);
                            let mut bound_exhausted = 0u32;
                            let mut bound_counters = crate::counters::EvalCounters::new();
                            let dev = laplace_deviance(
                                &data,
                                nb,
                                phi,
                                nagq,
                                beta_scale.caller_point(&params[..total_dim], n_theta, p),
                                beta_rhs,
                                wx,
                                pirls,
                                structured,
                                pattern,
                                packed,
                                border,
                                prior_w_phi,
                                beta_prof,
                                exact_prof,
                                BetaMode::Fixed,
                                None,
                                cluster_rows.as_ref(),
                                &mut bound_exhausted,
                                &mut bound_counters,
                            );
                            let obj = dev + nb_term(params[nb_col]);
                            if obj < best_obj {
                                best_obj = obj;
                                u_seed[..k].copy_from_slice(&pirls.u[..k]);
                            } else {
                                params[nb_col] = ln_phi;
                            }
                        }

                        debug_assert!(out.status != Status::InvalidArgs);
                        let exhausted = matches!(out.status, Status::MaxFunReached);
                        // Reported eval count is stage 1 + stage 2 (0 + stage 2 on the `Joint` route,
                        // so byte-identical to a single-stage run). Only stage 2's status feeds `converged`.
                        let read = (
                            matches!(out.status, Status::Converged | Status::MaxFunReached)
                                && best_obj.is_finite()
                                && finite_evals2 >= 2
                                && !(out.status == Status::Converged
                                    && any_within(non_finite_points, params, NON_FINITE_RADIUS)),
                            n_eval_stage1 + n_eval_prev + out.n_eval,
                            best_obj,
                        );
                        beta_scale.to_caller(&mut params[n_theta..nb_col]);
                        (read, exhausted, out.n_eval)
                    };
                    if (info_pass && info_passes >= max_info_passes) || !read.0 || exhausted {
                        // A second pass that fails keeps the first pass's fit, and a
                        // third that fails or ends higher keeps the second's, with
                        // every pass's evaluations counted.
                        if let Some(obj) =
                            first_obj.filter(|&o| !read.0 || (info_passes >= 2 && o < read.2))
                        {
                            params[..total_dim].copy_from_slice(&start_point[..total_dim]);
                            u_seed[..k].copy_from_slice(&guard_u[..k]);
                            break ((true, read.1, obj), false);
                        }
                        break (read, exhausted);
                    }
                    // The check: one Laplace solve at the incumbent's θ and
                    // dispersion with β re-solved jointly with the modes (PQL),
                    // then, on AGQ, one solve of the search's own objective at
                    // that β, started from the modes the PQL solve left there
                    // (the incumbent's modes belong to its own β: where counts
                    // are huge, a solve started from them at the re-solved β
                    // can run out of step halvings). Neither is a search
                    // evaluation: own counters, `n_eval` unchanged. When it does
                    // not fire, the incumbent's β is restored and the fit is the
                    // one-pass fit bit for bit.
                    let ((nb, phi), coord) = if n_disp == 1 {
                        disp_at(params[nb_col])
                    } else {
                        ((nb_theta, 1.0), None)
                    };
                    let mut check_exhausted = 0u32;
                    let mut check_counters = crate::counters::EvalCounters::new();
                    pirls.u[..k].copy_from_slice(&u_seed[..k]);
                    beta_prof[..p].copy_from_slice(&params[n_theta..nb_col]);
                    let mut check = laplace_deviance(
                        &data,
                        nb,
                        phi,
                        1,
                        &params[..n_theta],
                        beta_prof,
                        wx,
                        pirls,
                        structured,
                        pattern,
                        packed,
                        border,
                        prior_w_phi,
                        beta_rhs,
                        exact_prof,
                        BetaMode::ProfilePql,
                        None,
                        cluster_rows.as_ref(),
                        &mut check_exhausted,
                        &mut check_counters,
                    );
                    let info_ok = check.is_finite()
                        && beta_scale.set_info(border.schur.as_mut(), prior_w[..n].iter().sum());
                    beta_seed[..p].copy_from_slice(&params[n_theta..nb_col]);
                    params[n_theta..nb_col].copy_from_slice(&beta_prof[..p]);
                    if info_ok && nagq > 1 {
                        check = laplace_deviance(
                            &data,
                            nb,
                            phi,
                            nagq,
                            &params[..total_dim],
                            beta_rhs,
                            wx,
                            pirls,
                            structured,
                            pattern,
                            packed,
                            border,
                            prior_w_phi,
                            beta_prof,
                            exact_prof,
                            BetaMode::Fixed,
                            None,
                            cluster_rows.as_ref(),
                            &mut check_exhausted,
                            &mut check_counters,
                        );
                    }
                    if let Some(c) = coord {
                        check += nb_term(c);
                    }
                    // After a second pass the check is reached only on an AGQ
                    // fit (`max_info_passes`), whose third pass runs whatever
                    // the check found.
                    if info_ok && (info_pass || check < read.2 - joint_recheck_band(read.2)) {
                        // The previous pass's incumbent, for the restore above:
                        // `start_point` and `guard_u` are not read again once the
                        // search has begun. Its β is in `beta_seed`; the check has
                        // already moved `params`' β.
                        start_point[..total_dim].copy_from_slice(&params[..total_dim]);
                        start_point[n_theta..nb_col].copy_from_slice(&beta_seed[..p]);
                        guard_u[..k].copy_from_slice(&u_seed[..k]);
                        // The next pass starts at the check's point, so its
                        // incumbent modes are the check's, solved there.
                        u_seed[..k].copy_from_slice(&pirls.u[..k]);
                        first_obj = Some(read.2);
                        info_pass = true;
                        info_passes += 1;
                        n_eval_prev += pass_n_eval;
                        continue;
                    }
                    params[n_theta..nb_col].copy_from_slice(&beta_seed[..p]);
                    break (read, exhausted);
                };
                budget_exhausted = exhausted;
                read
            };

            // `ExactProfile` not converged, on its first (non-retry) pass: rerun
            // once on `PqlThenJoint` from the restart point rather than report
            // this failure — see `Note::ExactProfileFallback`. `route`/
            // `stage1_mode` rebind to `PqlThenJoint` for the retry, so this arm
            // cannot fire twice. "Not converged" is `!ok` (a hard failure) or
            // `budget_exhausted` (a reported incumbent that never met tolerance)
            // — both count, since both are "ExactProfile ended not converged".
            if route == OuterSearch::ExactProfile && (!ok || budget_exhausted) && !fallback_used {
                fallback_used = true;
                attempt1 = Some((
                    params[..total_dim].to_vec(),
                    u_seed[..k].to_vec(),
                    ok,
                    n_eval,
                    budget_exhausted,
                    final_obj,
                ));
                route = OuterSearch::PqlThenJoint;
                stage1_mode = Some(BetaMode::ProfilePql);
                continue;
            }

            // The fallback ran: report its own result when it converged; when it
            // also failed, keep whichever of the two attempts reached the lower
            // objective (still not converged either way) and restore that
            // attempt's params/û if it is the first one. `n_eval` always sums
            // both attempts — both spent real evaluations on this fit.
            if let Some((p1_params, p1_u, p1_ok, p1_n_eval, p1_budget, p1_obj)) = attempt1.take() {
                ws.exact_profile_fallback = true;
                let total_n_eval = p1_n_eval + n_eval;
                if ok && !budget_exhausted {
                    break (true, total_n_eval, false, final_obj);
                } else if p1_obj <= final_obj {
                    params[..p1_params.len()].copy_from_slice(&p1_params);
                    u_seed[..k].copy_from_slice(&p1_u);
                    break (p1_ok, total_n_eval, p1_budget, p1_obj);
                } else {
                    break (ok, total_n_eval, budget_exhausted, final_obj);
                }
            }
            break (ok, n_eval, budget_exhausted, final_obj);
        }
    };
    // A gradient search that handed over spent real evaluations on this fit.
    let n_eval = n_eval + grad_spent;

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
    // search's own `nb_term(θ̂)` summed by row rather than by count group, so
    // `deviance = −2·loglik` holds. Cached now, as a plain f64: this reads
    // `prior_w` out of the destructure above, a borrow the pinned re-eval's own
    // reborrow of `ws.prior_w` further down cannot coexist with.
    // Gamma's term is the search's own: `gamma_dispersion_term` carries every
    // constant of `−2·Σ log f` already, so `deviance = −2·loglik` holds with it.
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
        // The search's θ, kept so the incumbent check after the pinned
        // re-eval can undo the pin. `start_point` is free once the search
        // loop has ended.
        ws.start_point[..n_theta].copy_from_slice(&ws.params[..n_theta]);
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
        //
        // AGQ is the exception. Its product Gauss–Hermite grid sits in the
        // u = Λ⁻¹b coordinates (`agq::agq_deviance_vec`), so two Λ with the same
        // Σ can give different AGQ deviances. Rotating Λ here would report a
        // point the search did not choose, at a deviance it never saw. On the
        // AGQ route (the gate `deviance::laplace_deviance` uses) the fold runs
        // on a copy that only decides the pin flags, and θ, the re-eval below
        // and the SE pass keep the search's own Λ. The dense AGQ envelope has no
        // extras and `q_p ≤ 3`, so `MAX_THETA` covers the copy.
        let agq_route = derivative::agq_eligible(family, nagq, ws.groupings.primary_q)
            && ws.groupings.extra_offsets.is_empty();
        let mut canon_copy = [0.0_f64; crate::consts::MAX_THETA];
        let theta_c: &mut [f64] = if agq_route {
            canon_copy[..n_theta].copy_from_slice(&ws.params[..n_theta]);
            &mut canon_copy[..n_theta]
        } else {
            &mut ws.params[..n_theta]
        };
        if crate::lmm::canonicalize_pinned_blocks(&ws.groupings, theta_c) {
            pinned = false;
            pinned_components = 0;
            for (kk, &ti) in diag.iter().enumerate() {
                if theta_c[ti] <= PIN_THETA {
                    theta_c[ti] = 0.0;
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
    // Runs once, or twice when the incumbent check at its end undoes the pin
    // or the trough polish after it moves the pinned point.
    let mut reeval = ok;
    let mut polished = false;
    // Whether the stationarity probe at the pinned corner has run: it runs
    // once, and the re-eval that follows it must not run it again.
    let mut probed = false;
    let mut n_eval = n_eval;
    while reeval {
        reeval = false;
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
            prior_w_phi,
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
            solver,
            lower,
            upper,
            beta_scale,
            start_point,
            params_stage1,
            guard_beta,
            u_seed,
            pirls_exhausted,
            counters,
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
            gamma_phi,
            nagq,
            &params[..],
            beta_rhs,
            wx,
            pirls,
            structured,
            pattern,
            packed,
            border,
            prior_w_phi,
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
        // Incumbent check. `PIN_THETA` is absolute on θ, but what a θ that
        // small is worth depends on the working weights: under very precise
        // data (Poisson counts near e¹⁶, working weights about 9e6) a θ̂ of
        // 7e-5 is a real variance, and zeroing it with β held raises the
        // deviance by tens of units. A pin that leaves the objective
        // non-finite or more than `PIN_DEVIANCE_SLACK` above the search's own
        // best is undone: θ goes back to the search's point and the fit is
        // reported there, converged and not pinned.
        let pinned_obj = if n_disp == 1 && held_phi.is_none() {
            final_deviance + nb_dev_term
        } else {
            final_deviance
        };
        // False on a NaN objective too. A polished point is not checked
        // again: it is lower than a pinned point that passed.
        let noise = PIN_NOISE_ULPS * f64::EPSILON * inc_obj.abs();
        let within = pinned_obj <= inc_obj + PIN_DEVIANCE_SLACK.max(noise);
        let polish_gain = PIN_POLISH_DROP.max(noise);
        // Whether the trough polish below runs: the pin lowered the objective
        // by more than `PIN_POLISH_DROP`, or the probe of the pinned corner
        // finds the free θ coordinates off their optimum by more than that.
        // The probe runs only when the drop has not already decided, and
        // only on a pin that is kept (`within`; the undo branch is first).
        let wants_polish = pinned
            && !polished
            && within
            && (inc_obj - pinned_obj > polish_gain
                || (!probed && {
                    // Stationarity probe at the pinned corner. The trough polish
                    // was first keyed on the pin's drop alone, but a search
                    // starved by the trough can also stop at its bottom: on the
                    // crossed Poisson fit of
                    // `crossed_zero_variance_reaches_the_single_grouping_optimum`
                    // the same search ends at θ_h = 5e-7 on x86 (the pin drops
                    // the objective by 4.5e-5, the polish runs) and at θ_h = -1e-8
                    // on aarch64 (3e-8, no polish), both with σ_g at 0.77 against
                    // the `(1 | g)` optimum's 0.53, 2.6 deviance units above it
                    // (2026-10-08). Where the search ends inside a trough that
                    // narrow is round-off, so the drop cannot be the trigger.
                    // Instead each free θ coordinate is probed `PIN_PROBE_STEP`
                    // either way on the polish's own objective (β held, the
                    // pinned coordinates 0), and the gain the two probes predict
                    // along it, the larger of the drop to the lower probe and the
                    // parabola's `g²/(2c)` through the three points, says whether
                    // the pinned point is a minimum of the nested model. A
                    // coordinate on the θ box is probed on its inner side only.
                    // Each probe is a full-tolerance evaluation counted in
                    // `n_eval`, from the search's incumbent mode like the
                    // polish's evaluations. `start_point` holds the search's θ
                    // for the undo branch, which is ruled out here, so it is the
                    // probe's scratch.
                    let pp = *p;
                    let tp = n_theta + pp;
                    let diag = groupings.diagonal_theta();
                    let mut gain = 0.0f64;
                    start_point[..tp].copy_from_slice(&params[..tp]);
                    for j in 0..n_theta {
                        if diag.contains(&j) && params[j] == 0.0 {
                            continue;
                        }
                        let mut f = [f64::NAN; 2];
                        for (side, sgn) in [1.0f64, -1.0].into_iter().enumerate() {
                            let t = params[j] + sgn * PIN_PROBE_STEP;
                            if t < lower[j] || t > upper[j] {
                                continue;
                            }
                            start_point[j] = t;
                            pirls.u[..k].copy_from_slice(&u_seed[..k]);
                            f[side] = laplace_deviance(
                                &data,
                                nb_theta,
                                gamma_phi,
                                nagq,
                                &start_point[..tp],
                                beta_rhs,
                                wx,
                                pirls,
                                structured,
                                pattern,
                                packed,
                                border,
                                prior_w_phi,
                                beta_prof,
                                exact_prof,
                                BetaMode::Fixed,
                                None,
                                cluster_rows.as_ref(),
                                pirls_exhausted,
                                counters,
                            );
                            n_eval += 1;
                        }
                        start_point[j] = params[j];
                        // `f64::min` passes over a NaN side.
                        let lower_probe = f[0].min(f[1]);
                        if lower_probe.is_finite() {
                            gain = gain.max(final_deviance - lower_probe);
                        }
                        if f[0].is_finite() && f[1].is_finite() {
                            let h = PIN_PROBE_STEP;
                            let g = (f[0] - f[1]) / (2.0 * h);
                            let c = (f[0] + f[1] - 2.0 * final_deviance) / (h * h);
                            if c > 0.0 {
                                gain = gain.max(g * g / (2.0 * c));
                            }
                        }
                    }
                    probed = true;
                    gain > polish_gain
                }));
        if pinned && !polished && !within {
            params[..n_theta].copy_from_slice(&start_point[..n_theta]);
            pinned = false;
            pinned_components = 0;
            final_exhausted_count = 0;
            reeval = true;
        } else if wants_polish {
            // Trough polish. A variance whose optimum is 0 makes a trough
            // log(1 + θ²·Σw) in its θ; under very precise data (Σw near 1e7)
            // it is so narrow that the search's trust radius shrinks to its
            // end inside it while the other coordinates are still off their
            // optimum (σ_g up to 0.08 short on crossed Poisson fits near e¹⁴,
            // 0.24 short on the fit above). Such an endpoint sits on the
            // trough's wall or at its bottom; the pin lowers the objective
            // in the first case and the probe above sees the rest. One more
            // joint search from the pinned point, with the pinned coordinates
            // held at 0 (the objective reads them as 0, so the trough is
            // gone), moves the rest; its point is kept only when it is lower.
            // The dispersion stays at the search's value (its coordinate is
            // inert here too). Search evaluations: they count in `n_eval`.
            polished = true;
            let pp = *p;
            let tp = n_theta + pp;
            let diag = groupings.diagonal_theta();
            // The pinned point, kept to restore a polish that finds nothing
            // lower; its zeros also name the held coordinates.
            params_stage1[..n_theta].copy_from_slice(&params[..n_theta]);
            guard_beta[..pp].copy_from_slice(&params[n_theta..tp]);
            let disp = if n_disp == 1 { params[tp] } else { 0.0 };
            // β̃ is scaled by the β information at the pinned θ, from one PQL
            // solve there, the metric of the joint search's second pass
            // (`JOINT_RECHECK_TOL`); the design metric when that solve fails.
            // Not a search evaluation: own counters, `n_eval` unchanged.
            let mut info_exhausted = 0u32;
            let mut info_counters = crate::counters::EvalCounters::new();
            pirls.u[..k].copy_from_slice(&u_seed[..k]);
            beta_prof[..pp].copy_from_slice(&params[n_theta..tp]);
            let info = laplace_deviance(
                &data,
                nb_theta,
                gamma_phi,
                1,
                &params[..n_theta],
                beta_prof,
                wx,
                pirls,
                structured,
                pattern,
                packed,
                border,
                prior_w_phi,
                beta_rhs,
                exact_prof,
                BetaMode::ProfilePql,
                None,
                cluster_rows.as_ref(),
                &mut info_exhausted,
                &mut info_counters,
            );
            if !(info.is_finite()
                && beta_scale.set_info(border.schur.as_mut(), prior_w[..n].iter().sum()))
            {
                beta_scale.set(x, &prior_w[..n], n);
            }
            beta_scale.to_search(&mut params[n_theta..tp]);
            let mut best = final_deviance;
            let out = crate::optim::bobyqa_minimize(
                solver,
                &mut crate::optim::ValueFn(|gamma: &[f64], _rho: f64| {
                    start_point[..total_dim].copy_from_slice(gamma);
                    for &ti in diag {
                        if params_stage1[ti] == 0.0 {
                            start_point[ti] = 0.0;
                        }
                    }
                    pirls.u[..k].copy_from_slice(&u_seed[..k]);
                    let dev = laplace_deviance(
                        &data,
                        nb_theta,
                        gamma_phi,
                        nagq,
                        beta_scale.caller_point(&start_point[..total_dim], n_theta, pp),
                        beta_rhs,
                        wx,
                        pirls,
                        structured,
                        pattern,
                        packed,
                        border,
                        prior_w_phi,
                        beta_prof,
                        exact_prof,
                        BetaMode::Fixed,
                        None,
                        cluster_rows.as_ref(),
                        pirls_exhausted,
                        counters,
                    );
                    if dev < best {
                        best = dev;
                        u_seed[..k].copy_from_slice(&pirls.u[..k]);
                    }
                    dev
                }),
                params,
                lower,
                upper,
            );
            n_eval += out.n_eval;
            if best < final_deviance {
                // `params` is the polish's best point, the one `u_seed` holds.
                for &ti in diag {
                    if params_stage1[ti] == 0.0 {
                        params[ti] = 0.0;
                    }
                }
                beta_scale.to_caller(&mut params[n_theta..tp]);
                final_exhausted_count = 0;
                reeval = true;
            } else {
                // Nothing lower: the pinned point is the fit, re-evaluated
                // below so ũ, W̃ and `M` are its own and not the polish's
                // last point's.
                params[..n_theta].copy_from_slice(&params_stage1[..n_theta]);
                params[n_theta..tp].copy_from_slice(&guard_beta[..pp]);
                final_exhausted_count = 0;
                reeval = true;
            }
            if n_disp == 1 {
                params[tp] = disp;
            }
        } else if probed && !polished {
            // The probe found the pinned point stationary; its evaluations
            // left the workspace at the last probe point, so the pinned
            // point is re-evaluated once more (not a search evaluation).
            // `polished` marks that re-eval as the last pass: nothing is
            // checked again at the same point.
            polished = true;
            final_exhausted_count = 0;
            reeval = true;
        }
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
    // Gamma: a φ̂ on either bound of the `ln φ` box is not a maximum-likelihood
    // estimate. The box only keeps probes away from overflow; a search that
    // runs into it is following a likelihood that keeps rising as φ → 0 (data
    // the mean model reproduces exactly) or φ → ∞, and the incumbent there
    // describes the box, not the data.
    let ok = ok && !(is_gamma && held_phi.is_none() && gamma_phi_on_box_bound(ws.params[nb_col]));

    if !ok {
        return nan_fit(ws, target_indices, n_eval);
    }
    // Read before the SE pass: its evaluations overwrite the final re-eval's
    // weights and `M` rows that `extra_ztwz_trace` reads.
    let precision_limited =
        f64::EPSILON * (final_deviance.abs() + extra_ztwz_trace(ws, n)) > DEVIANCE_FLOOR_MAX;

    for j in 0..p {
        ws.betas[j] = ws.params[n_theta + j];
    }
    // β̂ on the caller's columns (β = B·β̂ on the fitted design); every SE arm
    // below reports Var, vcov, t² and the joint Wald in the same units.
    map.beta_to_caller(&mut ws.betas);

    // Var(β̂): `Rx` inverts the β-information (Schur) directly — fast, but the
    // Schur complement of the information at the mode assumes β–θ orthogonality (exact for
    // the Gaussian LMM, anticonservative for the GLMM where IRLS weights couple
    // β,θ). `Hessian` (default) sources Var(β̂) from the Hessian of the joint
    // Laplace deviance (glmer `use.hessian = TRUE`), the lme4 "correct" denom —
    // exact on every shape `derivative::supports_shape` accepts,
    // finite-difference elsewhere.
    let mut hessian_fallback = false;
    // A covariance that cannot be formed at the endpoint takes the SE outputs
    // with it and nothing else: the search's optimum does not depend on the
    // curvature read off it afterwards. Each arm breaks out with NaN, and the
    // outputs are NaN-filled below.
    let mut se_unavailable = false;
    // Reset the θ-block covariance each fit (the workspace is reused across
    // fits). Only the Hessian arm's `joint_hessian_cov` refills it; the Rx arm
    // leaves it NaN. The pivot likewise: each arm records it where it factors
    // the β information, and an arm that forms no factor leaves it NaN.
    ws.inference.theta_cov.fill(f64::NAN);
    ws.inference.pivot = f64::NAN;
    let joint_t_sq = match wald_se {
        WaldSe::Rx => 'rx: {
            // No scale factor on any family: on Gamma the Schur is built from
            // the working weights of the pinned re-eval, which carry `1/φ̂`
            // (PIRLS runs on `prior_wᵢ/φ̂`), so Schur⁻¹ is already `φ̂·RX⁻¹` —
            // the fitted objective's own information, φ̂ its ML value. lme4's
            // `pwrss/n` belongs to lme4's φ-free objective and does not apply.
            // Var(β̂)_jj from chol(Schur) forward-solve (mirrors fit_lmm's recovery).
            let (sc, qr) = match rx_schur_llt(ws, x, cluster_ids, n) {
                Some(f) => f,
                None => {
                    se_unavailable = true;
                    break 'rx f64::NAN;
                }
            };
            let lschur = sc.L();
            nan_fill_vcov(ws, p);
            // Right-hand side row t of the column map B (eₜ when the design is
            // fitted as given), read as U⁻ᵀbₜ against a factor built on X·U⁻¹:
            // Var(βₜ) = ‖L⁻¹bₜ‖², and the Gram of these columns below is
            // B·Schur⁻¹·Bᵀ.
            for &tj in target_indices {
                let tj = tj as usize;
                for i in 0..p {
                    ws.inference.fwd_solve[i] = map.rhs(tj, i);
                }
                if let Some(qr) = &qr {
                    qr.solve_row(&mut ws.inference.fwd_solve[..p]);
                }
                // Forward-solve in place: fwd_solve[i] is overwritten after the
                // rows above it were, and read only as fwd_solve[kk] (kk < i).
                for i in 0..p {
                    let mut acc = ws.inference.fwd_solve[i];
                    for kk in 0..i {
                        acc -= lschur[(i, kk)] * ws.inference.fwd_solve[kk];
                    }
                    ws.inference.fwd_solve[i] = acc / lschur[(i, i)];
                }
                let vd: f64 = ws.inference.fwd_solve[..p]
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>();
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
                    ws.inference.vcov[(a, b)] = acc;
                    ws.inference.vcov[(b, a)] = acc;
                }
            }
            // Joint Wald-χ² from the same columns: `vcov = CᵀC`.
            crate::lmm::joint_wald_chi_sq(
                ws.inference.vcov_cols.as_ref(),
                &ws.betas,
                1.0,
                target_indices,
                ws.inference.joint_qr.as_mut(),
                &mut ws.inference.joint_z,
            )
        }
        WaldSe::Hessian => 'hessian: {
            // Joint-Hessian covariance into a LOCAL p×p Mat — NOT a ws field: the kernel
            // takes `&mut ws`, so `&mut ws.<field>` for out_cov would alias it. The
            // allocation is acceptable on this default path (the zero-alloc gate
            // pins the Rx warm path). The kernel re-evals PIRLS itself and its
            // RX fallback runs schur_fill internally, so skip the schur_fill above.
            let mut cov = Mat::<f64>::zeros(p, p);
            let status = joint_hessian_cov(ws, x, y, cluster_ids, extra_ids, p, n, &mut cov);
            // Double-failure sentinel: the kernel NaN-fills `cov` when BOTH the joint
            // Hessian and the RX fallback fail.
            if !cov[(0, 0)].is_finite() {
                se_unavailable = true;
                break 'hessian f64::NAN;
            }
            // `cov` is Var(β̃) on the fitted design; the caller's is B·cov·Bᵀ.
            map.congruence(cov.as_mut());
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
            // Joint Wald-χ² from a square root of the covariance: `joint_hessian_cov`
            // leaves `cov_root` (`RᵀR` = the fitted design's Cov(β̂)), so the
            // caller's column `t` is `R·bₜ`, `bₜ` row `t` of the map.
            for &tj in target_indices {
                let tj = tj as usize;
                for i in 0..p {
                    let mut acc = 0.0;
                    for j in 0..p {
                        acc += ws.inference.cov_root[(i, j)] * map.rhs(tj, j);
                    }
                    ws.inference.vcov_cols[(i, tj)] = acc;
                }
            }
            crate::lmm::joint_wald_chi_sq(
                ws.inference.vcov_cols.as_ref(),
                &ws.betas,
                1.0,
                target_indices,
                ws.inference.joint_qr.as_mut(),
                &mut ws.inference.joint_z,
            )
        }
    };
    if se_unavailable {
        nan_fill_vcov(ws, p);
        for &t in target_indices {
            ws.inference.var_diag[t as usize] = f64::NAN;
            ws.inference.t_sq[t as usize] = f64::NAN;
        }
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
        joint_t_sq,
        hessian_fallback,
        exact_profile_fallback: ws.exact_profile_fallback,
        gradient_search_fallback: ws.gradient_search_fallback,
        se_unavailable,
        precision_limited,
        separated: false,
        deviance: final_deviance + nb_dev_term,
    }
}

/// Radius of the non-finite-point convergence check in `fit_glmm`, in the
/// search's own coordinates (θ̃, β̃, ln φ / ln θ_NB). It is a constant of its
/// own, not a search's start radius, so the check's radius does not move with
/// `GLMM_RHO_BEGIN` or `GLMM_RHO_POLISH`.
const NON_FINITE_RADIUS: f64 = 0.1;

/// How far above the search's best objective a pinned fit may land and keep
/// its pin, in deviance units (`fit_glmm`'s incumbent check). Measured
/// 2026-10-04: on the 67 pinned GLMM cells of the accuracy grid and on 614
/// simulated pinned fits (Poisson and binomial; crossed, nested and
/// random-slope shapes) a pin costs at most 5.2e-8, while a pin that zeroes a
/// real variance under very precise data costs 1.9 to 48 units. 1e-6 sits
/// between the two, and below the 4e-5 at which the accuracy grid counts a
/// deviance as worse than a reference (`validation/grid/tol.R`, `dev_eps`).
const PIN_DEVIANCE_SLACK: f64 = 1e-6;

/// Relative deviance drop at which the check after the joint `[θ | β̃]`
/// search (`fit_glmm`) sends it into a second pass, capped at
/// [`JOINT_RECHECK_ABS`] deviance units (`joint_recheck_band`). The check re-solves β at
/// the incumbent's θ jointly with the modes (one PQL step, on AGQ followed by
/// one solve of the AGQ objective at that β). At an optimum that β is no
/// better than the incumbent's, since the PQL β is not the Laplace or AGQ
/// one; a lower value means the search stopped short of a minimum in β. That
/// happens when the working weights span decades (`W = μ` on Poisson log,
/// `μ²/φ` on the Gamma inverse link, with cluster means decades apart): β̃
/// is scaled by the design alone (`BetaScale::set`), one β̃ axis is then far
/// steeper than the others, and BOBYQA stops converged in that valley. The
/// second pass starts from the re-solved β on β̃ scaled by the β information
/// there (`BetaScale::set_info`), the curvature the deviance actually has.
///
/// Measured 2026-10-04: six Poisson-log clusters with log-means spread over
/// eight decades, the check finds 1.3e-7 (relative) on the packed layout's
/// search and 8.5e-9 on 7-node AGQ's, and the second pass reaches the
/// exact-profile optimum (to 2e-8) and the AGQ optimum; over twelve decades
/// both had stopped 800 to 1050 units above it, 3e-8 to 4e-8 relative. On the
/// accuracy grid's AGQ and packed cells the largest value was 2e-14 (a σ = 0
/// fit, PQL and Laplace alike); every other was negative.
const JOINT_RECHECK_TOL: f64 = 1e-9;
/// Cap on the check's band, in deviance units: [`JOINT_RECHECK_TOL`] relative,
/// but never more than this. The relative band grows with |f|, and where a few
/// clusters with huge counts carry nearly all of a large deviance it outgrows
/// the misses it is there to catch. Measured 2026-10-16: six Poisson-log
/// clusters of five rows with log-means spread over 13.75 and 14 decades
/// (|f| from 7.4e11 to 2.4e12), seeds 12345 and 1 to 4: the 7-node AGQ search
/// stopped 1090 to 1460 units above the exact-profile optimum, the check found
/// it 679 to 937 units lower, and the band (737 to 2438 units) kept the
/// second pass from running. One unit stays above the deviance's own rounding
/// (`16·ε·|f|`, the gradient search's floor) for |f| up to 2.8e14, and changes
/// nothing where |f| ≤ 1e9.
const JOINT_RECHECK_ABS: f64 = 1.0;

/// The drop below the incumbent at which the check after the joint search
/// sends it into a second pass: `min(JOINT_RECHECK_TOL·max(|f|, 1),
/// JOINT_RECHECK_ABS)`.
fn joint_recheck_band(f: f64) -> f64 {
    (JOINT_RECHECK_TOL * f.abs().max(1.0)).min(JOINT_RECHECK_ABS)
}

/// How far below the search's best objective the pinned point must land to
/// run `fit_glmm`'s trough polish, in deviance units. Measured 2026-10-05 on
/// 614 simulated pinned Laplace fits (Poisson e² to e¹⁶, binomial up to 1e5
/// trials; crossed, nested and random-slope shapes): the polish ran on 190
/// and lowered 184 of them by more than 1e-6, among them 148 of the 167 fits
/// a polish without this threshold lowers by more than 1e-3; on 240 7-node
/// AGQ random-slope fits it ran on 71 and lowered 65. On the pinned cells of
/// the accuracy grid the pin lowers the objective by at most 1.0e-8, so none
/// of them is polished on the drop. The drop is not the only trigger: the
/// probe of the pinned corner (`PIN_PROBE_STEP`) sends an endpoint the drop
/// misses, one that stopped at the trough's bottom, into the same polish,
/// its predicted gain judged against this threshold.
const PIN_POLISH_DROP: f64 = 1e-7;

/// Step of the stationarity probe at the pinned corner (`fit_glmm`, before
/// the trough polish), in the search's θ̃ units. The gain the probe predicts
/// along a free θ coordinate is judged against `PIN_POLISH_DROP`, like the
/// pin's drop, so the step must resolve gains of that size: the parabola's
/// error in the gain grows as the fourth power of the step, and at
/// `GLMM_RHO_POLISH` (0.03) it read 1e-6 on an endpoint the gradient search
/// had certified, sending it into a polish that found nothing (2026-10-08);
/// at 1e-3 it is about 1e-11, with the round-off of the two probes (1e-9 on
/// a deviance of 200) far below `PIN_POLISH_DROP` as well. A starved
/// endpoint is found by extrapolation, not by the step's reach: on the fit
/// named above the parabola predicts a gain of about 1 from a point 0.24
/// off. Well above `GLMM_RHO_END` (3e-6), the resolution of the search's
/// endpoint.
const PIN_PROBE_STEP: f64 = 1e-3;

/// Floor under `PIN_DEVIANCE_SLACK` and `PIN_POLISH_DROP`, in ULPs of the
/// search's best objective. Both thresholds are absolute, but the two
/// objectives they compare each carry a rounding error that grows with |D|.
/// The floor takes over past |D| ≈ 7e6 for the drop and 7e7 for the slack, so
/// the pin check and the trough polish do not act on that noise. Below those
/// both thresholds keep their measured values.
const PIN_NOISE_ULPS: f64 = 64.0;

/// Whether any of `points` (flattened, `x.len()` coordinates each) lies within
/// Euclidean distance `radius` of `x`.
fn any_within(points: &[f64], x: &[f64], radius: f64) -> bool {
    points.chunks_exact(x.len()).any(|pt| {
        pt.iter()
            .zip(x)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            <= radius * radius
    })
}

/// `Σᵢ wᵢ·‖mᵢˣ‖²`, the trace of `M'WM` over the columns of the extra grouping
/// factors (`mᵢˣ` is row i of `M = ZΛ` without the primary factor's entries),
/// at the point the last Laplace evaluation solved: the weights in
/// `ws.pirls.w`, `M` in the layout's packed rows. A pinned factor's entries are
/// 0 there. 0 on the blocked layout, which has no extra factor. Read by
/// `fit_glmm` against [`DEVIANCE_FLOOR_MAX`].
fn extra_ztwz_trace(ws: &GlmmWorkspace, n: usize) -> f64 {
    let w = &ws.pirls.w[..n];
    let sq = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>();
    let q_p = ws.groupings.primary_q;
    match ws.layout {
        GlmmLayout::Blocked => 0.0,
        // `m_core_buf` rows are `[primary | nested]`; row `i`'s `n_cross[i]`
        // crossed entries take the first slots of `cross_theta`
        // (`StructuredScratch`).
        GlmmLayout::Structured => {
            let qc = q_p + ws.groupings.nested_per_parent;
            let (st, nc) = (&ws.structured, &ws.pattern.n_cross);
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
            let width = ws.packed.width;
            (0..n)
                .map(|i| w[i] * sq(&ws.packed.m_vals[i * width + q_p..(i + 1) * width]))
                .sum()
        }
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
        deviance: f64::INFINITY,
    }
}

/// Reference curvature of the dispersion scale. With θ
/// unprobed, `s = sqrt(DISP_SCALE_C_REF / c)` is the scale that gives `ln φ` the
/// curvature `DISP_SCALE_C_REF` assigns θ; 16 and 64 tied on the gami cells (4 and 256
/// took 6% and 35% more evaluations).
const DISP_SCALE_C_REF: f64 = 16.0;
/// Smallest dispersion scale (a ratio bound of 64).
const DISP_SCALE_MIN: f64 = 1.0 / 64.0;
/// The scale applies only when the probe parabola puts the
/// conditional `ln φ` optimum within `DISP_SCALE_GATE` start radii of the seed.
/// Measured at the start: 0.40–0.48 radii on every gami cell, 3.0–9.5 on
/// every gaml cell.
const DISP_SCALE_GATE: f64 = 1.0;

/// One probe solve kept for reuse, `(θ, objective, û, β̂)`.
type DispProbe = (Vec<f64>, f64, Vec<f64>, Vec<f64>);
