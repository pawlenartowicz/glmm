//! GLMM kernel: PIRLS + Laplace/AGQ, glmer-faithful nAGQ=1. Covers
//! Binomial/Poisson/Gamma/negative-binomial with random effects; on NB and
//! Gamma the outer BOBYQA searches the dispersion (`ln θ_NB`, `ln φ`) as one
//! more trailing coordinate alongside `[θ_RE | β]`, so β, θ_RE and the
//! dispersion come out of one fit — see `fit_glmm`'s `nb_term`/`n_disp`
//! handling. Optimized per
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
    /// Set iff the search reached an endpoint but no fixed-effect covariance
    /// could be formed there: the Rx Schur complement is not positive definite
    /// to rounding (under `WaldSe::Hessian`, after the joint Hessian failed
    /// too). β̂, θ̂, the deviance and `converged` stand as the search left them;
    /// `var_diag`, `t_sq`, `vcov` and `joint_t_sq` are NaN — see
    /// [`crate::Note::SeUnavailable`].
    pub se_unavailable: bool,
    /// Set iff the estimated rounding error of the reported deviance exceeds
    /// [`DEVIANCE_FLOOR_MAX`], so the search could not resolve θ̂ to a tenth
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
/// `map` is the column standardization `x` carries: β̂ and every SE output
/// are reported through it (`ColumnScaling::IDENTITY` for a design fitted
/// as given).
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
        let disp = held_phi.unwrap_or(disp);
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
    // fit on a hit and two on a miss; on the `Joint` route it costs two; all
    // outside the reported `n_eval`.
    let mut guard_cache: Option<(f64, u32)> = None;
    if guard_warm {
        let disp_coord = if n_disp == 1 { params[nb_col] } else { 0.0 };
        let guard_mode = stage1_mode.filter(|_| nagq == 1).unwrap_or(BetaMode::Fixed);
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
        let (warm_obj, warm_exh, warm_mem) = eval_guard(combined, cacheable);
        // The blind cold θ: the same point the `None` arm above builds.
        // Mirrors the `None` arm and the warm arm's `blind` closure; change
        // together.
        combined[..n_theta].fill(0.0);
        for &i in groupings.diagonal_theta() {
            combined[i] = THETA0;
        }
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
        }
    }
    // A kept warm θ₀ runs stage 1 at `RHO_BEGIN_WARM` (see that constant's
    // doc). The joint search keeps its own radius either way: as a polish it
    // starts at stage 1's optimum, and on the `Joint` route a warm start
    // measured no loss at `GLMM_RHO_BEGIN`.
    let solver_stage1: &mut bobyqa::Bobyqa = if warm_kept {
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
    let total_dim = n_theta + p + n_disp;
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
    let (ok, n_eval, budget_exhausted, inc_obj) = loop {
        // Reset the search's starting point on every pass — a no-op on the
        // first (nothing has run yet) and what lets the fallback restart clean
        // rather than from the failed `ExactProfile` attempt's own incumbent.
        params[..total_dim].copy_from_slice(&start_point[..total_dim]);
        for v in u_seed[..k].iter_mut() {
            *v = 0.0;
        }

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
        if let Some(mode) = stage1_mode.filter(|_| nagq == 1) {
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
                                  cache: &mut Option<(f64, u32)>|
             -> f64 {
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
                                || theta[n_theta].to_bits() == start_point[nb_col].to_bits()) =>
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
                obj
            };
            let out1 = solver_stage1.minimize_with_radius(
                |theta, radius| {
                    stage1_obj(
                        theta,
                        radius.rho,
                        &mut best1,
                        &mut finite_evals1,
                        non_finite_points,
                        &mut u_seed[..],
                        &mut beta_seed[..],
                        &mut guard_cache,
                    )
                },
                params_stage1,
                lower_stage1,
                upper_stage1,
            );
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
                for &q in std::iter::once(&groupings.primary_q).chain(groupings.extra_q.iter()) {
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
                    let out2 = polish.minimize_with_radius(
                        |theta, radius| {
                            stage1_obj(
                                theta,
                                radius.rho,
                                &mut best1,
                                &mut finite_evals1,
                                non_finite_points,
                                &mut u_seed[..],
                                &mut beta_seed[..],
                                &mut guard_cache,
                            )
                        },
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
            let mut info_pass = false;
            let mut n_eval_prev = 0usize;
            let mut first_obj: Option<f64> = None;
            let (read, exhausted) = loop {
                let mut best_obj = f64::INFINITY;
                // mirrors the stage-1 read above — change together.
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
                let out = solver.minimize(
                    |gamma| {
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
                        } else {
                            non_finite_points.extend_from_slice(gamma);
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
                // evaluation: own counters, `n_eval` unchanged.
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
                if info_pass || !read.0 || exhausted {
                    // A second pass that fails keeps the first pass's fit, with
                    // both passes' evaluations counted.
                    if let (Some(obj), false) = (first_obj, read.0) {
                        params[..total_dim].copy_from_slice(&start_point[..total_dim]);
                        u_seed[..k].copy_from_slice(&guard_u[..k]);
                        break ((true, read.1, obj), false);
                    }
                    break (read, exhausted);
                }
                // The check: one Laplace solve at the incumbent's θ and
                // dispersion with β re-solved jointly with the modes (PQL),
                // then, on AGQ, one solve of the search's own objective at
                // that β. Neither is a search evaluation: own counters,
                // `n_eval` unchanged. When it does not fire, the incumbent's β
                // is restored and the fit is the one-pass fit bit for bit.
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
                    pirls.u[..k].copy_from_slice(&u_seed[..k]);
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
                if info_ok && check < best_obj - JOINT_RECHECK_TOL * best_obj.abs().max(1.0) {
                    // The first pass's incumbent, for the restore above:
                    // `start_point` and `guard_u` are not read again once the
                    // search has begun. Its β is in `beta_seed`; the check has
                    // already moved `params`' β.
                    start_point[..total_dim].copy_from_slice(&params[..total_dim]);
                    start_point[n_theta..nb_col].copy_from_slice(&beta_seed[..p]);
                    guard_u[..k].copy_from_slice(&u_seed[..k]);
                    first_obj = Some(best_obj);
                    info_pass = true;
                    n_eval_prev += out.n_eval;
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
    };

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
        if pinned && !polished && !within {
            params[..n_theta].copy_from_slice(&start_point[..n_theta]);
            pinned = false;
            pinned_components = 0;
            final_exhausted_count = 0;
            reeval = true;
        } else if pinned && !polished && inc_obj - pinned_obj > PIN_POLISH_DROP.max(noise) {
            // Trough polish. A variance whose optimum is 0 makes a trough
            // log(1 + θ²·Σw) in its θ; under very precise data (Σw near 1e7)
            // it is so narrow that the search's trust radius shrinks to its
            // end inside it while the other coordinates are still off their
            // optimum (σ_g up to 0.08 short on crossed Poisson fits near e¹⁴).
            // Such an endpoint sits on the trough's wall, so the pin lowers
            // the objective. One more joint search from the pinned point, with
            // the pinned coordinates held at 0 (the objective reads them as 0,
            // so the trough is gone), moves the rest; its point is kept only
            // when it is lower. The dispersion stays at the search's value
            // (its coordinate is inert here too). Search evaluations: they
            // count in `n_eval`.
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
            let out = solver.minimize(
                |gamma| {
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
                },
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
                params[..n_theta].copy_from_slice(&params_stage1[..n_theta]);
                params[n_theta..tp].copy_from_slice(&guard_beta[..pp]);
            }
            if n_disp == 1 {
                params[tp] = disp;
            }
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
/// search (`fit_glmm`) sends it into a second pass. The check re-solves β at
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

/// How far below the search's best objective the pinned point must land to
/// run `fit_glmm`'s trough polish, in deviance units. Measured 2026-10-05 on
/// 614 simulated pinned Laplace fits (Poisson e² to e¹⁶, binomial up to 1e5
/// trials; crossed, nested and random-slope shapes): the polish ran on 190
/// and lowered 184 of them by more than 1e-6, among them 148 of the 167 fits
/// a polish without this threshold lowers by more than 1e-3; on 240 7-node
/// AGQ random-slope fits it ran on 71 and lowered 65. On the pinned cells of
/// the accuracy grid the pin lowers the objective by at most 1.0e-8, so none
/// of them is polished.
const PIN_POLISH_DROP: f64 = 1e-7;

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
        // `m_core_buf` rows are `[primary | nested]`, `cross_val` rows hold
        // `n_cross[i]` crossed entries (`StructuredScratch`).
        GlmmLayout::Structured => {
            let qc = q_p + ws.groupings.nested_per_parent;
            let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
            let (st, nc) = (&ws.structured, &ws.pattern.n_cross);
            (0..n)
                .map(|i| {
                    let nested = &st.m_core_buf[i * qc + q_p..(i + 1) * qc];
                    let crossed = &st.cross_val[i * g_cap..i * g_cap + nc[i] as usize];
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
        se_unavailable: false,
        precision_limited: false,
        separated: false,
        deviance: f64::INFINITY,
    }
}
