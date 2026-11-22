//! The start of `fit_glmm`'s outer search: the start point and the per-fit
//! resets (`seed_start`), the route (`pick_route`).

use faer::MatRef;

use super::deviance::{laplace_deviance, EvalScratch};
use super::grad_route::{GRAD_WARM_AGQ_THETA_MIN, GRAD_WARM_SCALAR_ROWS_MAX, GRAD_WARM_THETA_MAX};
use super::pirls::BetaMode;
use super::{exact_profile_shape, GlmmWorkspace, OuterSearch, SearchCtx, SearchState};
use crate::consts::GAMMA_PHI_LO;
use crate::lmm::{RHO_BEGIN_WARM, THETA0, THETA_TRUTH_FLOOR};

/// The per-fit values `seed_start` leaves for the rest of `fit_glmm`.
pub(super) struct StartSeed {
    /// A warm start the start guard compares with the blind one.
    pub(super) guard_warm: bool,
    pub(super) is_gamma: bool,
    /// A caller-held Gamma φ (`GlmmWorkspace::gamma_phi_held`).
    pub(super) held_phi: Option<f64>,
    /// 1 when the search carries a dispersion coordinate, else 0.
    pub(super) n_disp: usize,
    /// The Gamma plug-in seed's `ln φ` (NaN elsewhere).
    pub(super) ln_phi_plug: f64,
    pub(super) search_div: f64,
    /// The caller's curvature hint, taken for this call.
    pub(super) grad_hint: Option<crate::optim::CurvatureHint>,
}

/// The start point `[θ₀ | β₀ | dispersion]` in `ws.params`, the per-fit
/// resets, and the AGQ cluster index.
pub(super) fn seed_start(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    theta_start: Option<&[f64]>,
    beta_start: &[f64],
    n: usize,
) -> StartSeed {
    let (k, p, n_theta) = (ws.k, ws.p, ws.n_theta);
    // γ₀ = [θ₀ | β₀]. `guard_warm` is set only on a warm start (the `Some` arm
    // below) — the start guard (`choose_start`) then decides whether the search
    // should begin at the caller's θ₀ or at the blind cold start instead. A
    // cold fit skips the guard entirely.
    let mut guard_warm = false;
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
            // A start's structural entries are ignored: the search never
            // moves them.
            ws.groupings.zero_structural(&mut ws.params[..n_theta]);
            // The start guard (`choose_start`) compares a warm caller's θ₀ with
            // the blind cold θ this data would start from otherwise, and the
            // search begins from whichever the outer objective is lower at.
            //
            // A start AT the blind θ₀, bit for bit (a caller passes `[THETA0; n]`
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
            // blind θ₀ `GlmmWorkspace::new` builds (workspace.rs). An all-THETA0
            // start would imply RE correlation +0.707 for every pair, and on
            // negative-correlation data it converges into the τ=0 boundary
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
        // The clamp keeps the start inside the box. It bounds the plug-in and a
        // carried φ. A held φ is used as given (`Dispersion::at` and the incumbent
        // write-back in `settle_dispersion`), so for it the clamp only keeps the start of the
        // inert coordinate inside the box.
        ws.params[m] = disp.ln().clamp(ws.lower[m], ws.upper[m]);
    } else if n_disp == 1 {
        ws.params[m] = ws.nb_theta.ln().clamp(ws.lower[m], ws.upper[m]);
    }

    // Within-fit warm-start resets per fit — the incumbent u_seed is NEVER carried
    // across fits. Carrying it would make a fit's result depend on which fits ran
    // before it in the workspace, breaking both the guarantee that a given (spec,
    // data, seed) fits the same way regardless of call order and the reset-state
    // assumption the fit entry makes about a fresh `GlmmWorkspace`.
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
    let mean_w = ws.prior_w[..n].iter().sum::<f64>() / n as f64;
    // `exp2`, not `powi`: below a mean of about 7.9e-309, `2f64.powi(-1024)`
    // forms 2^1024 = inf and returns 0, where `exp2` gives the subnormal.
    ws.search_div = mean_w.log2().round().exp2().min(1.0);
    let search_div = ws.search_div;
    // A caller's hint applies to this call only.
    let grad_hint = ws.grad_hint_in.take();
    ws.exact_prof.curv_memory = false;
    ws.non_finite_points.clear();
    ws.counters.reset();
    ws.pattern.coup_mask = None; // CSR validity is per (fit, pinning mask)

    // Cluster-outer AGQ substrate: built once per fit (cluster_ids is fit-fixed).
    // Scalar AGQ (`q_p = 1`) reads it ONLY in `parallel` builds, where it is
    // rayon's work-splitting substrate. Serial builds run scalar AGQ's
    // node-outer loop: cluster-outer's residual per-cluster overhead (loop
    // restart + CSR gather) regresses many-tiny-cluster shapes
    // (observation-level REs: +12–16% measured on grouseticks (1|INDEX), 403
    // clusters × 1 row) while its serial cache win on large clusters is small
    // (−4% on cbpp), so the non-parallel hot path (batch loops) stays
    // node-outer. Vector AGQ (`q_p ≥ 2`) is cluster-outer on every build and
    // builds a transient index per evaluation when none is handed in, so a
    // serial build gets it here too, built once and kept across fits.
    let par_index = cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
        && ws.nagq > 1
        && ws.parallel_inner;
    let serial_vec_index = !cfg!(all(feature = "parallel", not(target_arch = "wasm32")))
        && ws.nagq > 1
        && ws.groupings.primary_q >= 2
        && ws.groupings.extra_offsets.is_empty();
    ws.cluster_rows = if par_index || serial_vec_index {
        let mut idx = ws.cluster_rows.take().unwrap_or_default();
        idx.ensure(cluster_ids, ws.groupings.n_primary);
        Some(idx)
    } else {
        None
    };
    StartSeed {
        guard_warm,
        is_gamma,
        held_phi,
        n_disp,
        ln_phi_plug,
        search_div,
        grad_hint,
    }
}

/// The outer search's route for this shape (`OuterSearch`), whether the
/// gradient route runs, and the `BetaMode` stage 1 profiles β with.
pub(super) fn pick_route(
    ws: &GlmmWorkspace,
    guard_warm: bool,
    n: usize,
) -> (OuterSearch, bool, Option<BetaMode>) {
    let nagq = ws.nagq;
    // Outer search route for this shape (`OuterSearch`). Read before the
    // destructure moves `ws`'s fields out by mutable reference; the enum is Copy
    // so this is a plain read.
    let route = ws.outer_search;
    // The gradient route (`gradient_route_plan`): BFGS over `[θ | β]`, an
    // AGQ fit only below `GRAD_WARM_SCALAR_ROWS_MAX` rows
    // per primary level, and a warm AGQ fit only from a start with every
    // diagonal θ₀ at or above `GRAD_WARM_AGQ_THETA_MIN`. A warm start at the
    // blind θ₀ is a cold fit here, as it is for the start guard.
    let grad_route = {
        use crate::optim::OuterOptimizer;
        let g = &ws.groupings;
        let opt = ws.route_plan;
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
    let stage1_mode = match route {
        OuterSearch::Joint => None,
        OuterSearch::PqlThenJoint => Some(BetaMode::ProfilePql),
        OuterSearch::ExactProfile => Some(BetaMode::ProfileExact),
    };
    (route, grad_route, stage1_mode)
}

/// What the start guard leaves for the searches.
pub(super) struct StartOutcome {
    /// The guard kept the caller's θ₀ (stage 1 then searches from it at
    /// `RHO_BEGIN_WARM`).
    pub(super) warm_kept: bool,
    /// The picked start's objective and PIRLS-exhaustion count, for stage 1's
    /// first call to take over.
    pub(super) guard_cache: Option<(f64, u32)>,
    /// The guard left the picked start's profiled β̂ in `guard_beta`.
    pub(super) guard_beta_kept: bool,
    /// `border` already holds the gradient route's β information.
    pub(super) info_border_ready: bool,
}

/// The start guard: on a warm fit, the start `ws.params` keeps the caller's
/// θ₀ or takes the blind cold θ₀, whichever the outer objective is lower at.
pub(super) fn choose_start(
    ctx: &SearchCtx,
    st: SearchState<'_>,
    sc: &mut EvalScratch<'_>,
    guard_warm: bool,
    grad_route: bool,
    stage1_mode: Option<BetaMode>,
) -> StartOutcome {
    let SearchCtx {
        data,
        disp,
        groupings,
        nagq,
        k,
        p,
        n_theta,
        nb_col,
        ..
    } = *ctx;
    let SearchState {
        params,
        start_point,
        guard_u,
        guard_beta,
        ..
    } = st;
    // Set when the start guard keeps the caller's θ₀ (stage 1 then searches
    // from it at `RHO_BEGIN_WARM`).
    let mut warm_kept = false;
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
    // `border` already holds the gradient route's β information (`grad_route::gradient_route`).
    let mut info_border_ready = false;
    if guard_warm {
        let guard_disp = disp.at(params, nb_col);
        // The gradient route on a Laplace fit compares the starts on the PQL
        // profile, not on its shape's stage-1 mode. It has no stage 1 to hand
        // a solve over to. On 3,370 such warm fits (grid, stress and sweep
        // cells, 2026-10-20) the PQL profile picked the same start as the
        // exact profile on all but 4, whose fitted deviances moved by at most
        // 2e-10; no other fit's deviance moved by more than 1e-6.
        // On the β-information opt-in, the kept warm start's solve doubles as
        // the β-information solve (in `grad_route::gradient_route`). The fixed-β objective at β₀ is not enough: it kept a θ₀ on
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
            sc.exact_prof.curv_memory = false;
            for v in sc.pirls.u[..k].iter_mut() {
                *v = 0.0;
            }
            // Profile profiles `beta_prof` from the β0 seed.
            if guard_mode != BetaMode::Fixed {
                sc.beta_prof[..p].copy_from_slice(&combined[n_theta..]);
            }
            let ((nb, phi), coord) = guard_disp;
            let mut guard_exhausted = 0u32;
            let mut guard_counters = crate::counters::EvalCounters::new();
            let dev = laplace_deviance(
                &data,
                sc,
                nb,
                phi,
                nagq,
                combined,
                guard_mode,
                None,
                &mut guard_exhausted,
                &mut guard_counters,
            );
            if snap {
                guard_u[..k].copy_from_slice(&sc.pirls.u[..k]);
                guard_beta[..p].copy_from_slice(&sc.beta_prof[..p]);
            }
            let obj = match coord {
                Some(c) => dev + disp.term(c),
                None => dev,
            };
            (obj, guard_exhausted, sc.exact_prof.curv_memory)
        };
        // The blind cold θ: the same point `seed_start`'s `None` arm builds.
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
                sc.exact_prof.curv_memory = cold_mem;
            } else {
                params[..n_theta].copy_from_slice(&combined[..n_theta]);
                guard_beta[..p].copy_from_slice(&sc.beta_prof[..p]);
                sc.exact_prof.curv_memory = warm_mem;
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
                    guard_u[..k].copy_from_slice(&sc.pirls.u[..k]);
                    guard_beta[..p].copy_from_slice(&sc.beta_prof[..p]);
                }
                (cold_obj, cold_exh, cold_mem)
            } else {
                warm_kept = true;
                (warm_obj, warm_exh, warm_mem)
            };
            sc.exact_prof.curv_memory = memory;
            if cacheable {
                guard_cache = Some((obj, exhausted));
                guard_beta_kept = true;
            }
        }
    }
    StartOutcome {
        warm_kept,
        guard_cache,
        guard_beta_kept,
        info_border_ready,
    }
}

/// The stage-1 solver the BOBYQA route runs, and the point it begins from
/// (`start_point`).
pub(super) fn restart_search<'b>(
    ctx: &SearchCtx,
    st: SearchState<'_>,
    solver_stage1: &'b mut bobyqa::Bobyqa,
    solver_stage1_warm: &'b mut Option<bobyqa::Bobyqa>,
    warm_kept: bool,
    grad_fell_back: bool,
) -> &'b mut bobyqa::Bobyqa {
    let SearchCtx {
        n_free,
        n_disp,
        total_dim,
        ..
    } = *ctx;
    let SearchState {
        config_stage1,
        params,
        start_point,
        ..
    } = st;
    // A kept warm θ₀ runs stage 1 at `RHO_BEGIN_WARM` (see that constant's
    // doc). The joint search keeps its own radius either way: as a polish it
    // starts at stage 1's optimum, and on the `Joint` route a warm start
    // measured no loss at `GLMM_RHO_BEGIN`.
    // A gradient search that handed over restarts stage 1 near its incumbent
    // (`RHO_BEGIN_WARM`), as a kept warm start does; where the route has no
    // stage 1 (AGQ, Laplace `Joint`) the joint search restarts from it at
    // `GLMM_RHO_BEGIN`.
    let solver_stage1: &mut bobyqa::Bobyqa = if warm_kept || grad_fell_back {
        solver_stage1_warm.get_or_insert_with(|| {
            let mut c = *config_stage1;
            c.rho_begin = RHO_BEGIN_WARM;
            bobyqa::Bobyqa::new(n_free + n_disp, c)
                .expect("warm BOBYQA config is valid by construction")
        })
    } else {
        solver_stage1
    };

    // The point the BOBYQA search (`bobyqa_route::bobyqa_route`) begins from
    // (after the start guard in `choose_start`).
    // The `PqlThenJoint` fallback restarts here rather than from wherever a
    // failed `ExactProfile` attempt left `params`: the scan behind the
    // fallback found `PqlThenJoint` reaches the same optimum from every start
    // it was tried from, so restarting from the ORIGINAL point is exactly as
    // good as any other and never rewards the failed attempt's own excursion.
    start_point[..total_dim].copy_from_slice(&params[..total_dim]);
    solver_stage1
}
