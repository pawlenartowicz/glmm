//! The BOBYQA route of `fit_glmm`'s outer search (`OuterSearch`): stage 1,
//! the joint search and the `PqlThenJoint` fallback.

use bobyqa::Status;

use super::deviance::{laplace_deviance, EvalScratch};
use super::pirls::BetaMode;
use super::workspace::{self, FitData};
use super::{
    dispersion_bound_probe, pirls_tol, saddle_walk, Dispersion, OuterSearch, RouteResult,
    SearchCtx, SearchState, RHO_TOL_C, RHO_TOL_FULL_FACTOR, RHO_TOL_MAX,
};
use crate::lmm::{GLMM_RHO_POLISH, POLISH_NEAR_SINGULAR, RHO_BEGIN_WARM};

/// Radius of the searches' non-finite-point convergence check, in the
/// search's own coordinates (θ̃, β̃, ln φ / ln θ_NB). It is a constant of its
/// own, not a search's start radius, so the check's radius does not move with
/// `GLMM_RHO_BEGIN` or `GLMM_RHO_POLISH`.
pub(super) const NON_FINITE_RADIUS: f64 = 0.1;

/// Relative deviance drop at which the check after the joint `[θ | β̃]`
/// search (`joint_passes`) sends it into a second pass, capped at
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
pub(super) fn joint_recheck_band(f: f64) -> f64 {
    (JOINT_RECHECK_TOL * f.abs().max(1.0)).min(JOINT_RECHECK_ABS)
}

/// [`crate::optim::bobyqa_minimize`] on `obj` divided by `div`
/// (`GlmmWorkspace::search_div`). The returned `f` is in those units; the
/// searches' closures keep their own unscaled incumbents.
pub(super) fn bobyqa_scaled(
    solver: &mut bobyqa::Bobyqa,
    obj: &mut impl crate::optim::Objective,
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    structural: &[usize],
    div: f64,
) -> bobyqa::Outcome {
    let mut obj = crate::optim::Scaled::new(obj, div);
    crate::optim::bobyqa_minimize(solver, &mut obj, x, lower, upper, structural)
}

/// Whether any of `points` (flattened, `x.len()` coordinates each) lies within
/// Euclidean distance `radius` of `x`.
pub(super) fn any_within(points: &[f64], x: &[f64], radius: f64) -> bool {
    points.chunks_exact(x.len()).any(|pt| {
        pt.iter()
            .zip(x)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            <= radius * radius
    })
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

/// The BOBYQA route: stage 1 where the route has one, then the joint search
/// on `PqlThenJoint`, `Joint` and AGQ, with the `PqlThenJoint` fallback of a
/// failed `ExactProfile` search. `grad_done` is the gradient route's finish:
/// on a Laplace shape it is the result as is; on AGQ it is the joint search's
/// first pass.
///
/// `route` and `stage1_mode` are `mut`: the `PqlThenJoint` fallback rebinds
/// `route` once, after an `ExactProfile` attempt ends not converged, to rerun
/// the search on the fallback route — the shape's OWN route, passed in, is
/// what the fallback gate checks, and rebinding a route that is never
/// `ExactProfile` again self-limits the retry to one. `stage1_mode` is
/// rebound alongside.
#[allow(clippy::too_many_arguments)]
pub(super) fn bobyqa_route<'a>(
    ctx: &SearchCtx<'a>,
    mut st: SearchState<'_>,
    sc: &mut EvalScratch<'a>,
    solver_stage1: &mut bobyqa::Bobyqa,
    mut route: OuterSearch,
    mut stage1_mode: Option<BetaMode>,
    warm_kept: bool,
    mut guard_cache: Option<(f64, u32)>,
    mut grad_done: Option<RouteResult>,
    grad_fell_back: bool,
    exact_profile_fallback: &mut bool,
) -> RouteResult {
    let SearchCtx {
        k,
        total_dim,
        agq_finish,
        ..
    } = *ctx;
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
    if let Some(done) = grad_done {
        done
    } else {
        loop {
            let SearchState {
                params,
                start_point,
                u_seed,
                ..
            } = st.reborrow();
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
                let out1 = stage1(
                    ctx,
                    st.reborrow(),
                    sc,
                    solver_stage1,
                    mode,
                    &mut best1,
                    &mut finite_evals1,
                    &mut guard_cache,
                    warm_kept,
                    grad_fell_back,
                );
                // On `PqlThenJoint`, non-convergence does NOT fail the fit — stage 1 is an
                // accelerant, and the joint polish below proceeds from wherever the
                // incumbent landed (worst case = the cold start). On `ExactProfile`
                // this status IS the fit's convergence status — read below.
                n_eval_stage1 = out1.n_eval;
                stage1_out = Some(out1);
            }

            // `budget_exhausted` drives the plateau policy mirrored from `fit_lmm`
            // (`src/lmm/mod.rs`): a
            // `Status::MaxFunReached` exit reports its finite incumbent below with
            // `converged == false` and `boundary_hit == 2` rather than NaN-filling.
            let budget_exhausted;
            let (ok, n_eval, final_obj) = if route == OuterSearch::ExactProfile {
                let SearchState {
                    params_stage1,
                    non_finite_points,
                    ..
                } = st.reborrow();
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
                let (read, exhausted) = joint_passes(
                    ctx,
                    st.reborrow(),
                    sc,
                    &mut grad_finish,
                    n_eval_stage1,
                    &stage1_out,
                );
                budget_exhausted = exhausted;
                read
            };

            let SearchState { params, u_seed, .. } = st.reborrow();
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
                *exact_profile_fallback = true;
                let total_n_eval = p1_n_eval + n_eval;
                if ok && !budget_exhausted {
                    break RouteResult {
                        ok: true,
                        n_eval: total_n_eval,
                        budget_exhausted: false,
                        obj: final_obj,
                    };
                } else if p1_obj <= final_obj {
                    params[..p1_params.len()].copy_from_slice(&p1_params);
                    u_seed[..k].copy_from_slice(&p1_u);
                    break RouteResult {
                        ok: p1_ok,
                        n_eval: total_n_eval,
                        budget_exhausted: p1_budget,
                        obj: p1_obj,
                    };
                } else {
                    break RouteResult {
                        ok,
                        n_eval: total_n_eval,
                        budget_exhausted,
                        obj: final_obj,
                    };
                }
            }
            break RouteResult {
                ok,
                n_eval,
                budget_exhausted,
                obj: final_obj,
            };
        }
    }
}

/// Stage 1's objective at one point `[θ | dispersion]`: the profiled Laplace
/// deviance plus the dispersion term, at the PIRLS tolerance the trust radius
/// sets, taking over the start guard's solve on its first call and a probe's
/// solve from `memo`. Built once per stage-1 run.
struct Stage1Objective<'o, 'd> {
    data: FitData<'d>,
    disp: Dispersion<'d>,
    sc: &'o mut EvalScratch<'d>,
    pirls_exhausted: &'o mut u32,
    counters: &'o mut crate::counters::EvalCounters,
    start_point: &'o [f64],
    guard_u: &'o [f64],
    guard_beta: &'o [f64],
    mode: BetaMode,
    nagq: u8,
    tol_full: f64,
    rho_tol: bool,
    full_below: f64,
    k: usize,
    p: usize,
    n_theta: usize,
    n_disp: usize,
    nb_col: usize,
    first_eval: bool,
    /// The PIRLS tolerance the incumbent was solved at.
    inc_tol: f64,
}

impl Stage1Objective<'_, '_> {
    /// The objective at `theta` under trust radius `rho`. `best`, `finite`,
    /// `non_finite` and the `u_seed`/`beta_seed` snapshots are the search's
    /// incumbent state; `cache` is the start guard's solve; `record` keeps the
    /// solve in `memo`.
    #[allow(clippy::too_many_arguments)]
    fn eval(
        &mut self,
        theta: &[f64],
        rho: f64,
        best: &mut f64,
        finite: &mut usize,
        non_finite: &mut Vec<f64>,
        u_seed: &mut [f64],
        beta_seed: &mut [f64],
        cache: &mut Option<(f64, u32)>,
        memo: &mut Vec<DispProbe>,
        record: bool,
    ) -> f64 {
        let Stage1Objective {
            data,
            disp,
            start_point,
            guard_u,
            guard_beta,
            mode,
            nagq,
            tol_full,
            rho_tol,
            full_below,
            k,
            p,
            n_theta,
            n_disp,
            nb_col,
            ..
        } = *self;
        let (sc, pirls_exhausted, counters) = (
            &mut *self.sc,
            &mut *self.pirls_exhausted,
            &mut *self.counters,
        );
        // A point the dispersion probes already solved hands back
        // that solve (objective, û, β̂); its incumbent bookkeeping ran
        // when it was probed.
        if let Some(i) = memo.iter().position(|m| {
            m.0.iter()
                .zip(theta)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        }) {
            let (_, obj, u, b) = memo.swap_remove(i);
            self.first_eval = false;
            sc.pirls.u[..k].copy_from_slice(&u);
            sc.beta_prof[..p].copy_from_slice(&b);
            return obj;
        }
        let tol = if rho_tol && rho > full_below && !self.first_eval {
            (RHO_TOL_C * rho * rho).clamp(tol_full, RHO_TOL_MAX)
        } else {
            tol_full
        };
        self.first_eval = false;
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
                sc.exact_prof.curv_memory = false;
                None
            }
            None => None,
        };
        let obj = if let Some((cobj, cexh)) = cached {
            sc.pirls.u[..k].copy_from_slice(&guard_u[..k]);
            sc.beta_prof[..p].copy_from_slice(&guard_beta[..p]);
            *pirls_exhausted += cexh;
            cobj
        } else {
            // Re-seed BOTH latent states from the INCUMBENT (best point so far),
            // not the last-evaluated: BOBYQA's final call is not its best.
            // û is point-determined given θ, and β is likewise point-determined
            // (the PQL β̂(θ)); the incumbent seed only shifts the stopping iterate
            // within tol, the same argument that justifies the u_seed warm start.
            sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
            sc.beta_prof[..p].copy_from_slice(&beta_seed[..p]);
            let ((nb, phi), coord) = disp.at(theta, n_theta);
            // Stage 1 writes ws.border.schur via the Profile S_β border; the post-fit
            // SE path runs AFTER stage 2 and its blocked/structured/packed
            // schur_fill rebuilds ws.border.schur from scratch, so this transient use
            // is safe (no read survives into inference).
            let dev = laplace_deviance(
                &data,
                sc,
                nb,
                phi,
                nagq,
                &theta[..n_theta],
                // `mode` is `ProfilePql` on `PqlThenJoint`, `ProfileExact` on
                // `ExactProfile`: the two routes that reach this block.
                mode,
                // The radius-tied tolerance while ρ is large on
                // `ExactProfile`, `pirls_tol` (`None`) otherwise; never
                // the FD-pass tolerance.
                (tol > tol_full).then_some(tol),
                pirls_exhausted,
                counters,
            );
            match coord {
                Some(c) => dev + disp.term(c),
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
            u_seed[..k].copy_from_slice(&sc.pirls.u[..k]);
            beta_seed[..p].copy_from_slice(&sc.beta_prof[..p]);
            self.inc_tol = tol;
        }
        counters.record_eval(crate::counters::Stage::One, obj);
        if record {
            memo.push((
                theta.to_vec(),
                obj,
                sc.pirls.u[..k].to_vec(),
                sc.beta_prof[..p].to_vec(),
            ));
        }
        obj
    }
}

/// Stage 1: the θ-only BOBYQA search with β profiled out by `mode`, its
/// saddle check, dispersion-bound check and loose-incumbent re-solve. Leaves
/// `[θ̂ | β̂ | dispersion]` in `params` and returns the search's outcome.
#[allow(clippy::too_many_arguments)]
fn stage1<'a>(
    ctx: &SearchCtx<'a>,
    st: SearchState<'_>,
    sc: &mut EvalScratch<'a>,
    solver_stage1: &mut bobyqa::Bobyqa,
    mode: BetaMode,
    best1: &mut f64,
    finite_evals1: &mut usize,
    guard_cache: &mut Option<(f64, u32)>,
    warm_kept: bool,
    grad_fell_back: bool,
) -> bobyqa::Outcome {
    let SearchCtx {
        data,
        disp,
        groupings,
        structural,
        family,
        nagq,
        k,
        p,
        n_theta,
        n_free,
        n_disp,
        nb_col,
        search_div,
        is_gamma,
        held_phi,
        ln_phi_plug,
        gamma_phi_start,
        ..
    } = *ctx;
    let SearchState {
        solver_stage1_polish,
        config_stage1,
        params,
        params_stage1,
        lower_stage1,
        upper_stage1,
        start_point,
        u_seed,
        beta_seed,
        guard_u,
        guard_beta,
        pirls_exhausted,
        non_finite_points,
        counters,
        ..
    } = st;
    params_stage1[..n_theta].copy_from_slice(&params[..n_theta]); // θ₀ unchanged from the caller's params
    if n_disp == 1 {
        params_stage1[n_theta] = params[nb_col]; // ln θ_NB / ln φ start, set before the search
    }
    sc.beta_prof[..p].copy_from_slice(&params[n_theta..n_theta + p]); // β₀ (GLM warm start)
    beta_seed[..p].copy_from_slice(&sc.beta_prof[..p]);
    // The incumbent state (best value, finite-evaluation count and the
    // û/β̂ snapshots) is passed in rather than held so the objective
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
    let mut s1 = Stage1Objective {
        data,
        disp,
        sc,
        pirls_exhausted,
        counters,
        start_point,
        guard_u,
        guard_beta,
        mode,
        nagq,
        tol_full,
        rho_tol,
        full_below,
        k,
        p,
        n_theta,
        n_disp,
        nb_col,
        first_eval: true,
        inc_tol: tol_full,
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
            let f = s1.eval(
                &xp,
                0.0,
                best1,
                finite_evals1,
                non_finite_points,
                &mut u_seed[..],
                &mut beta_seed[..],
                guard_cache,
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
        let decoupled = if gamma_phi_start.is_some() || grad_fell_back {
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
        let o = bobyqa_scaled(
            solver_stage1,
            &mut crate::optim::ValueFn(|z: &[f64], rho: f64| {
                xb.copy_from_slice(z);
                xb[j] = z[j] * s;
                let before = memo.len();
                let f = s1.eval(
                    &xb,
                    rho,
                    best1,
                    finite_evals1,
                    non_finite_points,
                    &mut u_seed[..],
                    &mut beta_seed[..],
                    guard_cache,
                    &mut memo,
                    false,
                );
                hits += before - memo.len();
                f
            }),
            params_stage1,
            &lo_z,
            &up_z,
            structural,
            search_div,
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
        bobyqa_scaled(
            solver_stage1,
            &mut crate::optim::ValueFn(|theta: &[f64], rho: f64| {
                s1.eval(
                    theta,
                    rho,
                    best1,
                    finite_evals1,
                    non_finite_points,
                    &mut u_seed[..],
                    &mut beta_seed[..],
                    guard_cache,
                    &mut memo,
                    false,
                )
            }),
            params_stage1,
            lower_stage1,
            upper_stage1,
            structural,
            search_div,
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
    // Every point is solved at `pirls_tol` (ρ = 0 in `Stage1Objective::eval`) and
    // counts as an evaluation; nothing is kept that is not lower. A
    // doubled point that is not finite is only a failed try: it is not
    // recorded for the `NON_FINITE_RADIUS` check, which judges the
    // searches' own points.
    let mut out1 = out1;
    if mode == BetaMode::ProfileExact && out1.status == Status::Converged {
        let (moved, tries) = saddle_walk(groupings, params_stage1, |x| {
            let prev = *best1;
            s1.eval(
                x,
                0.0,
                best1,
                finite_evals1,
                &mut Vec::new(),
                &mut u_seed[..],
                &mut beta_seed[..],
                guard_cache,
                &mut Vec::new(),
                false,
            );
            *best1 < prev
        });
        out1.n_eval += tries;
        if moved {
            let polish = solver_stage1_polish.get_or_insert_with(|| {
                let mut c = *config_stage1;
                c.rho_begin = GLMM_RHO_POLISH;
                bobyqa::Bobyqa::new(n_free + n_disp, c)
                    .expect("polish BOBYQA config is valid by construction")
            });
            let out2 = bobyqa_scaled(
                polish,
                &mut crate::optim::ValueFn(|theta: &[f64], rho: f64| {
                    s1.eval(
                        theta,
                        rho,
                        best1,
                        finite_evals1,
                        non_finite_points,
                        &mut u_seed[..],
                        &mut beta_seed[..],
                        guard_cache,
                        &mut Vec::new(),
                        false,
                    )
                }),
                params_stage1,
                lower_stage1,
                upper_stage1,
                structural,
                search_div,
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
        let bound = if is_gamma {
            lower_stage1[n_theta]
        } else {
            upper_stage1[n_theta]
        };
        // `Stage1Objective::eval` keeps its own incumbent: `best1` after the
        // solve is the probe's value, below the old one when kept.
        dispersion_bound_probe(params_stage1, n_theta, bound, *best1, |x| {
            s1.eval(
                x,
                0.0,
                best1,
                finite_evals1,
                &mut Vec::new(),
                &mut u_seed[..],
                &mut beta_seed[..],
                guard_cache,
                &mut Vec::new(),
                false,
            );
            out1.n_eval += 1;
            Some(*best1)
        });
    }
    let inc_tol = s1.inc_tol;
    // An incumbent solved at a loose tolerance is solved again at
    // `pirls_tol` from its own (û, β̂) before β̂ and the objective are
    // read off it, so the reported fit never rests on a loose solve.
    // Not a search evaluation (`n_eval` unchanged); a re-solve that
    // comes back non-finite keeps the loose one.
    if inc_tol > tol_full && best1.is_finite() {
        sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
        sc.beta_prof[..p].copy_from_slice(&beta_seed[..p]);
        let ((nb, phi), coord) = disp.at(params_stage1, n_theta);
        let mut resolve_counters = crate::counters::EvalCounters::new();
        let dev = laplace_deviance(
            &data,
            sc,
            nb,
            phi,
            nagq,
            &params_stage1[..n_theta],
            mode,
            None,
            pirls_exhausted,
            &mut resolve_counters,
        );
        let obj = match coord {
            Some(c) => dev + disp.term(c),
            None => dev,
        };
        if obj.is_finite() {
            *best1 = obj;
            u_seed[..k].copy_from_slice(&sc.pirls.u[..k]);
            beta_seed[..p].copy_from_slice(&sc.beta_prof[..p]);
        }
    }
    // Warm-start / final θ̂,β̂: θ̂₁ (BOBYQA leaves the incumbent in `params_stage1`)
    // and β̂₁ (the incumbent snapshot, NOT `beta_prof`'s last-evaluated value).
    // On `ExactProfile` this write IS the reported optimum, not a warm start.
    params[..n_theta].copy_from_slice(&params_stage1[..n_theta]);
    params[n_theta..nb_col].copy_from_slice(&beta_seed[..p]);
    if n_disp == 1 {
        params[nb_col] = params_stage1[n_theta];
    }
    out1
}

/// The joint `[θ | β̃]` search: one pass, or more when the check after a
/// pass finds a lower point (`JOINT_RECHECK_TOL`). A gradient-route finish
/// in `grad_finish` is the first pass. Returns `(ok, n_eval, objective)` and
/// whether the budget ran out.
fn joint_passes<'a>(
    ctx: &SearchCtx<'a>,
    st: SearchState<'_>,
    sc: &mut EvalScratch<'a>,
    grad_finish: &mut Option<RouteResult>,
    n_eval_stage1: usize,
    stage1_out: &Option<bobyqa::Outcome>,
) -> ((bool, usize, f64), bool) {
    let SearchCtx {
        data,
        disp,
        groupings,
        prior_w,
        x,
        structural,
        nagq,
        k,
        p,
        n,
        n_theta,
        n_disp,
        nb_col,
        total_dim,
        total_free,
        search_div,
        is_gamma,
        held_phi,
        agq_finish,
        ..
    } = *ctx;
    let SearchState {
        solver,
        config,
        solver_polish_wide,
        params,
        lower,
        upper,
        beta_scale,
        start_point,
        u_seed,
        beta_seed,
        guard_u,
        pirls_exhausted,
        non_finite_points,
        counters,
        ..
    } = st;
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
        let (read, exhausted, pass_n_eval) = if let Some(RouteResult {
            ok,
            n_eval: ne,
            budget_exhausted: ex,
            obj: f,
        }) = grad_finish.take()
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
                    bobyqa::Bobyqa::new(total_free, c)
                        .expect("polish BOBYQA config is valid by construction")
                })
            } else {
                &mut *solver
            };
            let out = bobyqa_scaled(
                solver,
                &mut crate::optim::ValueFn(|gamma: &[f64], _rho: f64| {
                    // Within-fit û warm-start: seed PIRLS from the incumbent (best point so
                    // far), not from 0. The conditional mode is point-determined, so the seed
                    // only shifts the stopping iterate within the PIRLS exit band.
                    sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
                    let ((nb, phi), coord) = disp.at(gamma, nb_col);
                    let dev = laplace_deviance(
                        &data,
                        sc,
                        nb,
                        phi,
                        nagq,
                        beta_scale.caller_point(gamma, n_theta, p),
                        // The `Joint` polish's objective is β-FIXED; the Profile border
                        // scratch is inert. Stage 1 above flips this to Profile
                        // (`ProfilePql` or `ProfileExact`, by `stage1_mode`).
                        BetaMode::Fixed,
                        // Never the FD-pass tol here: BOBYQA objective evals stay at
                        // `pirls_tol` (the field is None outside `joint_hessian_cov`).
                        None,
                        pirls_exhausted,
                        counters,
                    );
                    let obj = match coord {
                        Some(c) => dev + disp.term(c),
                        None => dev,
                    };
                    if obj.is_finite() {
                        finite_evals2 += 1;
                    } else {
                        non_finite_points.extend_from_slice(gamma);
                    }
                    if obj < best_obj {
                        best_obj = obj;
                        u_seed[..k].copy_from_slice(&sc.pirls.u[..k]);
                    }
                    counters.record_eval(crate::counters::Stage::Two, obj);
                    obj
                }),
                params,
                lower,
                upper,
                structural,
                search_div,
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
                let bound = if is_gamma {
                    lower[nb_col]
                } else {
                    upper[nb_col]
                };
                let kept = dispersion_bound_probe(params, nb_col, bound, best_obj, |x| {
                    sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
                    let ((nb, phi), _) = disp.at(x, nb_col);
                    let mut bound_exhausted = 0u32;
                    let mut bound_counters = crate::counters::EvalCounters::new();
                    let dev = laplace_deviance(
                        &data,
                        sc,
                        nb,
                        phi,
                        nagq,
                        beta_scale.caller_point(&x[..total_dim], n_theta, p),
                        BetaMode::Fixed,
                        None,
                        &mut bound_exhausted,
                        &mut bound_counters,
                    );
                    Some(dev + disp.term(x[nb_col]))
                });
                if let Some(obj) = kept {
                    best_obj = obj;
                    u_seed[..k].copy_from_slice(&sc.pirls.u[..k]);
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
            if let Some(obj) = first_obj.filter(|&o| !read.0 || (info_passes >= 2 && o < read.2)) {
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
        let (check, info_ok) = joint_recheck(ctx, sc, params, u_seed, beta_seed, beta_scale);
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
            u_seed[..k].copy_from_slice(&sc.pirls.u[..k]);
            first_obj = Some(read.2);
            info_pass = true;
            info_passes += 1;
            n_eval_prev += pass_n_eval;
            continue;
        }
        params[n_theta..nb_col].copy_from_slice(&beta_seed[..p]);
        break (read, exhausted);
    };
    (read, exhausted)
}

/// The check after a joint pass: one PQL solve at the incumbent's θ and
/// dispersion with β re-solved jointly with the modes, then on AGQ one solve
/// of the search's own objective at that β. Leaves the re-solved β in
/// `params` and the incumbent's in `beta_seed`. Returns the check's objective
/// and whether the β information scaled `beta_scale`.
fn joint_recheck(
    ctx: &SearchCtx,
    sc: &mut EvalScratch<'_>,
    params: &mut [f64],
    u_seed: &mut [f64],
    beta_seed: &mut [f64],
    beta_scale: &mut workspace::BetaScale,
) -> (f64, bool) {
    let SearchCtx {
        data,
        disp,
        prior_w,
        nagq,
        k,
        p,
        n,
        n_theta,
        nb_col,
        total_dim,
        ..
    } = *ctx;
    let ((nb, phi), coord) = disp.at(params, nb_col);
    let mut check_exhausted = 0u32;
    let mut check_counters = crate::counters::EvalCounters::new();
    sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
    sc.beta_prof[..p].copy_from_slice(&params[n_theta..nb_col]);
    let mut check = laplace_deviance(
        &data,
        sc,
        nb,
        phi,
        1,
        &params[..n_theta],
        BetaMode::ProfilePql,
        None,
        &mut check_exhausted,
        &mut check_counters,
    );
    let info_ok = check.is_finite()
        && beta_scale.set_info(sc.border.schur.as_mut(), prior_w[..n].iter().sum());
    beta_seed[..p].copy_from_slice(&params[n_theta..nb_col]);
    params[n_theta..nb_col].copy_from_slice(&sc.beta_prof[..p]);
    if info_ok && nagq > 1 {
        check = laplace_deviance(
            &data,
            sc,
            nb,
            phi,
            nagq,
            &params[..total_dim],
            BetaMode::Fixed,
            None,
            &mut check_exhausted,
            &mut check_counters,
        );
    }
    if let Some(c) = coord {
        check += disp.term(c);
    }
    (check, info_ok)
}
