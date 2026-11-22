//! The post-pin pass of `fit_glmm`: the pin pass at the search's endpoint,
//! the pinned re-evaluation, the incumbent check, the plateau probe, and at a
//! kept pin the stationarity probe and at most one trough polish.

use bobyqa::Status;
use faer::MatRef;

use super::bobyqa_route::bobyqa_scaled;
use super::deviance::{laplace_deviance, EvalScratch};
use super::pirls::BetaMode;
use super::workspace::FitData;
use super::{agq, GlmmLayout, GlmmWorkspace};
use crate::lmm::{PIN_DEVIANCE_SLACK, PIN_NOISE_ULPS, PIN_POLISH_DROP, PIN_PROBE_STEP};

/// The plateau probe in [`post_pin`] tries each diagonal θⱼ at or above this
/// at θⱼ/e and θⱼ·e. Out on the plateau toward the box a search can stop well
/// above the optimum and still meet its own test: from an intercept loading of
/// 20 on precise Poisson fits whose optimum is 0.5, 115 of 300 BOBYQA-route
/// fits ended converged at about 20, 114 to 208 deviance units above it
/// (2026-11-03). 10 is the gradient route's warm-start gate
/// (`grad_route::GRAD_WARM_THETA_MAX`); it is not tuned for the probe.
const PLATEAU_PROBE_THETA: f64 = 10.0;

/// What the post-pin pass leaves for the rest of `fit_glmm`.
pub(super) struct PinOutcome {
    /// The marginal deviance at the reported endpoint; infinite when the
    /// pass did not run.
    pub(super) final_deviance: f64,
    pub(super) pinned: bool,
    pub(super) pinned_components: u64,
    pub(super) budget_exhausted: bool,
    /// The search's evaluations plus the probes' and the trough polish's.
    pub(super) n_eval: usize,
}

/// The post-pin pass on the search's endpoint `ws.params` (on `ok` only).
/// `inc_obj` is the search's objective there, `nb_dev_term` the dispersion
/// term `settle_dispersion` returned, `tr_x` the extra factors' `M'WM` trace
/// at the search's last evaluation (`extra_ztwz_trace`).
#[allow(clippy::too_many_arguments)]
pub(super) fn post_pin(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
    structural: &[usize],
    total_dim: usize,
    n_disp: usize,
    held_phi: Option<f64>,
    nb_dev_term: f64,
    ok: bool,
    inc_obj: f64,
    budget_exhausted: bool,
    n_eval: usize,
    tr_x: f64,
) -> PinOutcome {
    let (family, nagq, weighted) = (ws.family, ws.nagq, ws.weighted);
    let (k, n_theta, search_div) = (ws.k, ws.n_theta, ws.search_div);
    let (nb_theta, gamma_phi) = (ws.nb_theta, ws.gamma_phi);
    // The post-pin pass, in a fixed order (`crate::lmm::pin_endpoint`, shared
    // with `fit_lmm`; β never pins): the pin pass at the search's endpoint,
    // the pinned re-eval, the incumbent check, and at a kept pin the probe of
    // the free coordinates and at most one trough polish, whose endpoint goes
    // through the pin pass, the re-eval and the incumbent check once more. No
    // second polish. `ok` is true on both a converged fit and a
    // budget-exhausted one, so the pin applies to any reported endpoint;
    // `pinned`/`pinned_components` are latched on both, but only survive into
    // the returned `GlmmFit` on a converged fit — a budget-exhausted exit
    // zeroes `pinned_components` at assembly.
    let mut pinned_components = 0u64;
    let mut pinned = false;
    // AGQ keeps the search's Λ. Its product Gauss–Hermite grid sits in the
    // u = Λ⁻¹b coordinates (`agq::agq_deviance_vec`), so two Λ with the same
    // Σ can give different AGQ deviances. Rotating Λ here would report a
    // point the search did not choose, at a deviance it never saw. On the
    // AGQ route (the gate `deviance::laplace_deviance` uses) the canonical
    // fold runs on a copy that only decides the pin flags, and θ, the
    // re-eval below and the SE pass keep the search's own Λ. Elsewhere the
    // pinned re-eval below sees the canonicalized θ. The dense AGQ envelope
    // has no extras and `q_p ≤ 3`, so `MAX_THETA` covers the copy, which is
    // sliced on that route only: a sparse fit's θ can be longer.
    let agq_route = agq::agq_eligible(family, nagq, ws.groupings.primary_q)
        && ws.groupings.extra_offsets.is_empty();
    debug_assert!(!agq_route || n_theta <= crate::consts::MAX_THETA);

    // Re-evaluate at the (possibly pinned) γ̂ to refresh M, ũ, W̃. ws.params already
    // holds the pinned γ̂, so call the kernel on it directly — no params copy. The
    // refreshed deviance is the reported marginal deviance (`GlmmFit::deviance`);
    // INFINITY until the re-eval runs.
    let mut final_deviance = f64::INFINITY;
    // Separate from `ws.pirls_exhausted`: the last re-eval is the one PIRLS
    // solve this reports, so this is 0 or 1 (reset before each re-eval),
    // folded into the reported bool after the block (the final re-eval is the
    // case where a truncated solve would feed the returned estimates directly;
    // see `Note::PirlsExhausted`).
    let mut final_exhausted_count = 0u32;
    // The pinned re-eval is not a search evaluation: it gets its own throwaway
    // counter for the same reason `final_exhausted_count` is separate above.
    let mut final_counters = crate::counters::EvalCounters::new();
    let mut inc_obj = inc_obj;
    let mut budget_exhausted = budget_exhausted;
    // Trough polishes run: 0 or 1.
    let mut searches = 0u32;
    let mut n_eval = n_eval;
    if ok {
        // The packed layout orders the primary block slope-major; blocked and
        // structured order it level-major (`fix_mode_signs`).
        let primary_slope_major = ws.layout == GlmmLayout::Packed;
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
            post_pin,
            ..
        } = ws;
        // Shadow the destructured `offset` field with the `Option<&[f64]>` view
        // `FitData` takes, scoped to the pinned re-eval.
        let offset = offset.as_deref();
        // z_buf still holds this fit's slope copy — x is unchanged since fill_z_f64.
        // On the structured path each re-eval re-packs m_core_buf/cross_* at γ̂,
        // which `structured_schur_fill` then reads.
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
        let diag = groupings.diagonal_theta();
        // The pin pass at an endpoint (the search's, then the polish's). The
        // modes flip with their columns (`fix_mode_signs`), before the θ flip
        // clears the signs that reads: `u_seed` seeds the pinned re-eval. The
        // endpoint's θ is kept in `start_point`, so the incumbent check after
        // the pinned re-eval can undo the pin, with the sign fix the pin pass
        // applies to θ (the same fix on the same values).
        macro_rules! pin_pass {
            () => {{
                crate::lmm::fix_mode_signs(
                    groupings,
                    &params[..n_theta],
                    &mut u_seed[..],
                    primary_slope_major,
                );
                start_point[..n_theta].copy_from_slice(&params[..n_theta]);
                crate::lmm::fix_column_signs(groupings, &mut start_point[..n_theta]);
                groupings.zero_structural(&mut start_point[..n_theta]);
                let mut canon_copy = [0.0_f64; crate::consts::MAX_THETA];
                crate::lmm::pin_endpoint(
                    groupings,
                    &mut params[..n_theta],
                    if agq_route {
                        Some(&mut canon_copy[..n_theta])
                    } else {
                        None
                    },
                )
            }};
        }
        // The pinned re-eval at `params`, warm-started from the incumbent's
        // modes (the inference iterate; `u_seed` holds the BOBYQA incumbent's).
        // β-FIXED (reports the fitted β); Profile border scratch inert.
        // Never the FD-pass tol here:
        // the pinned re-eval stays at `pirls_tol` (the field is None outside
        // `joint_hessian_cov`).
        macro_rules! reeval {
            () => {{
                sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
                final_exhausted_count = 0;
                laplace_deviance(
                    &data,
                    &mut sc,
                    nb_theta,
                    gamma_phi,
                    nagq,
                    &params[..],
                    BetaMode::Fixed,
                    None,
                    &mut final_exhausted_count,
                    &mut final_counters,
                )
            }};
        }
        // Incumbent check. `PIN_THETA` is absolute on θ, but what a θ that
        // small is worth depends on the working weights: under very precise
        // data (Poisson counts near e¹⁶, working weights about 9e6) a θ̂ of
        // 7e-5 is a real variance, and zeroing it with β held raises the
        // deviance by tens of units. A pin that leaves the objective
        // non-finite or more than `PIN_DEVIANCE_SLACK` above the incumbent
        // (the search's best, or the polish's) is undone. False on a NaN
        // objective too. `lmm::fit_lmm` applies the same rule — change
        // together. The noise floor under the slack here, the plateau probe's
        // `thr` and the polish's `polish_gain` is the larger of
        // `PIN_NOISE_ULPS` ULPs of the incumbent and the deviance's rounding
        // estimate `ε·(|inc| + tr_x)`, the hand-over's (`report_handover`):
        // with an extra grouping factor `log|A|` rounds at the size of that
        // factor's columns (`DEVIANCE_FLOOR_MAX`). The LMM's floor has no
        // `tr_x` term.
        let pinned_obj = |dev: f64| {
            if n_disp == 1 && held_phi.is_none() {
                dev + nb_dev_term
            } else {
                dev
            }
        };
        let noise = |inc: f64| {
            (PIN_NOISE_ULPS * f64::EPSILON * inc.abs()).max(super::deviance_rounding(inc, tr_x))
        };
        let kept = |dev: f64, inc: f64| pinned_obj(dev) <= inc + PIN_DEVIANCE_SLACK.max(noise(inc));

        (pinned, pinned_components) = pin_pass!();
        final_deviance = reeval!();
        if pinned && !kept(final_deviance, inc_obj) {
            // θ goes back to the search's endpoint and the fit is reported
            // there, not pinned.
            params[..n_theta].copy_from_slice(&start_point[..n_theta]);
            pinned = false;
            pinned_components = 0;
            final_deviance = reeval!();
        }
        // Plateau probe (`PLATEAU_PROBE_THETA` gives the measurement). Out on
        // the plateau the deviance changes by ratios of the SD, not by the
        // trust radius, so at a certified endpoint each diagonal θⱼ at or
        // above `PLATEAU_PROBE_THETA` is tried at θⱼ/e and θⱼ·e (β held,
        // counted in `n_eval`). A lower point refuses the certificate: the fit moves
        // there and the trough polish below runs from it.
        let mut log_moved = false;
        if !budget_exhausted {
            let mut probed = false;
            let tp = n_theta + *p;
            let thr = PIN_POLISH_DROP.max(noise(inc_obj));
            let mut best = (final_deviance, usize::MAX, 0.0);
            for &j in diag {
                let t0 = params[j];
                if t0 < PLATEAU_PROBE_THETA || structural.binary_search(&j).is_ok() {
                    continue;
                }
                for f in [(-1.0f64).exp(), 1.0f64.exp()] {
                    let t = t0 * f;
                    if t > upper[j] {
                        continue;
                    }
                    start_point[..tp].copy_from_slice(&params[..tp]);
                    start_point[j] = t;
                    sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
                    let dev = laplace_deviance(
                        &data,
                        &mut sc,
                        nb_theta,
                        gamma_phi,
                        nagq,
                        &start_point[..tp],
                        BetaMode::Fixed,
                        None,
                        pirls_exhausted,
                        counters,
                    );
                    n_eval += 1;
                    probed = true;
                    if dev < best.0 - thr {
                        best = (dev, j, t);
                    }
                }
            }
            if best.1 != usize::MAX {
                params[best.1] = best.2;
                final_deviance = reeval!();
                u_seed[..k].copy_from_slice(&sc.pirls.u[..k]);
                inc_obj = pinned_obj(final_deviance);
                log_moved = true;
            } else if probed && !pinned {
                // Nothing lower: the probe left the workspace at its last
                // point, so the endpoint is re-evaluated (not a search
                // evaluation).
                final_deviance = reeval!();
            }
        }
        if pinned || log_moved {
            let pp = *p;
            let tp = n_theta + pp;
            let polish_gain = PIN_POLISH_DROP.max(noise(inc_obj));
            let drop = inc_obj - pinned_obj(final_deviance);
            // Stationarity probe at the pinned corner. The pin's drop
            // alone is not a sufficient trigger: a search starved by the
            // trough can also stop at its bottom: on the
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
            // The parabola's curvature `c` along each coordinate also
            // scales the polish below, so the probe runs before the
            // polish also when the drop alone would send the point there.
            // Each probe is a full-tolerance evaluation counted in
            // `n_eval`, from the search's incumbent mode like the
            // polish's evaluations. `start_point` holds the endpoint's θ
            // for the undo branch, which is ruled out here, so it is the
            // probe's scratch. A structural coordinate is not a free θ:
            // moving it leaves the model.
            let skip = |j: usize| {
                (diag.contains(&j) && params[j] == 0.0) || structural.binary_search(&j).is_ok()
            };
            let theta_curv = &mut post_pin.curv[..];
            let (gain, unprobed, evals) = crate::optim::post_pin::stationarity_gain(
                &params[..tp],
                &mut start_point[..tp],
                &skip,
                &lower[..n_theta],
                &upper[..n_theta],
                final_deviance,
                PIN_PROBE_STEP,
                theta_curv,
                &mut |t: &[f64]| {
                    sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
                    laplace_deviance(
                        &data,
                        &mut sc,
                        nb_theta,
                        gamma_phi,
                        nagq,
                        t,
                        BetaMode::Fixed,
                        None,
                        pirls_exhausted,
                        counters,
                    )
                },
            );
            n_eval += evals as usize;
            // Whether the trough polish runs: the pin lowered the objective by
            // more than `polish_gain` (`PIN_POLISH_DROP` or the rounding floor,
            // whichever is larger), or the probe finds the free θ
            // coordinates off their optimum by more than that, or could not
            // probe one of them (the polish keeps the pinned point when it
            // finds nothing lower, so running it cannot worsen the fit).
            if drop > polish_gain || gain > polish_gain || unprobed > 0 || log_moved {
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
                // gone), moves the rest. The dispersion stays at the search's
                // value (its coordinate is inert here too). Search evaluations:
                // they count in `n_eval`.
                searches = 1;
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
                sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
                sc.beta_prof[..pp].copy_from_slice(&params[n_theta..tp]);
                let info = laplace_deviance(
                    &data,
                    &mut sc,
                    nb_theta,
                    gamma_phi,
                    1,
                    &params[..n_theta],
                    BetaMode::ProfilePql,
                    None,
                    &mut info_exhausted,
                    &mut info_counters,
                );
                let sum_w: f64 = prior_w[..n].iter().sum();
                if !(info.is_finite() && beta_scale.set_info(sc.border.schur.as_mut(), sum_w)) {
                    beta_scale.set(x, &prior_w[..n], n);
                }
                beta_scale.to_search(&mut params[n_theta..tp]);
                // The θ coordinates are scaled too. The β̃ axes have curvature
                // about 2·Σ(prior weight) (`BetaScale::set_info`); a free θ
                // coordinate the probe finds stiffer than that is searched as
                // θⱼ/sⱼ with sⱼ = √(2·Σ(prior weight)/cⱼ), so its curvature
                // matches theirs; every other θ coordinate keeps sⱼ = 1. Under
                // very precise data a loading near 0 has curvature ∝ Σw: on the
                // precise Poisson `(1 + x1 + x2 | g)` fit of
                // `trough_polish_reaches_the_optimum_on_precise_data` (log-mean e¹⁴) the
                // profiled Hessian at the optimum has eigenvalues 336 (the
                // intercept's loading) and 1.4e8 to 4.5e8 (the slopes'
                // loadings), a condition number of 1.3e6 (2026-10-30).
                // BOBYQA's one trust radius then shrinks to its end each time
                // the intercept's loading has moved about 6e-4. Chained
                // unscaled polishes gain 1e-3 or less after the first three,
                // and twelve of them stop 5e-7 (aarch64 macOS) to 9.2e-4
                // (x86_64 Linux) above the optimum. Scaled, the polish
                // reaches it.
                // The box is scaled with the point, and restored after.
                let theta_scale = &mut post_pin.scale[..];
                theta_scale.fill(1.0);
                post_pin.lower.copy_from_slice(&lower[..n_theta]);
                post_pin.upper.copy_from_slice(&upper[..n_theta]);
                let c_ref = 2.0 * sum_w;
                for j in 0..n_theta {
                    // False on a NaN curvature: a skipped or one-sided coordinate.
                    if theta_curv[j] > c_ref {
                        let sj = (c_ref / theta_curv[j]).sqrt();
                        theta_scale[j] = sj;
                        params[j] /= sj;
                        lower[j] /= sj;
                        upper[j] /= sj;
                    }
                }
                let mut best = final_deviance;
                let out = bobyqa_scaled(
                    solver,
                    &mut crate::optim::ValueFn(|gamma: &[f64], _rho: f64| {
                        start_point[..total_dim].copy_from_slice(gamma);
                        for (t, &sj) in start_point[..n_theta].iter_mut().zip(&*theta_scale) {
                            *t *= sj;
                        }
                        for &ti in diag {
                            if params_stage1[ti] == 0.0 {
                                start_point[ti] = 0.0;
                            }
                        }
                        sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
                        let dev = laplace_deviance(
                            &data,
                            &mut sc,
                            nb_theta,
                            gamma_phi,
                            nagq,
                            beta_scale.caller_point(&start_point[..total_dim], n_theta, pp),
                            BetaMode::Fixed,
                            None,
                            pirls_exhausted,
                            counters,
                        );
                        if dev < best {
                            best = dev;
                            u_seed[..k].copy_from_slice(&sc.pirls.u[..k]);
                        }
                        dev
                    }),
                    params,
                    lower,
                    upper,
                    structural,
                    search_div,
                );
                for (t, &sj) in params[..n_theta].iter_mut().zip(&*theta_scale) {
                    *t *= sj;
                }
                lower[..n_theta].copy_from_slice(&post_pin.lower);
                upper[..n_theta].copy_from_slice(&post_pin.upper);
                n_eval += out.n_eval;
                // A polish stopped by its budget reports its best point, not
                // converged: that point is not a minimum.
                if out.status == Status::MaxFunReached {
                    budget_exhausted = true;
                }
                if best < final_deviance {
                    // `params` is the polish's best point, the one `u_seed`
                    // holds, and the polish's best is the new incumbent.
                    for &ti in diag {
                        if params_stage1[ti] == 0.0 {
                            params[ti] = 0.0;
                        }
                    }
                    beta_scale.to_caller(&mut params[n_theta..tp]);
                    if n_disp == 1 {
                        params[tp] = disp;
                    }
                    inc_obj = pinned_obj(best);
                    // The polish's endpoint is pinned from scratch, so a
                    // coordinate it leaves at or below `PIN_THETA` pins too.
                    (pinned, pinned_components) = pin_pass!();
                    final_deviance = reeval!();
                    if pinned && !kept(final_deviance, inc_obj) {
                        // θ goes back to the polish's endpoint after the sign
                        // fix (`start_point`), folded without the pin
                        // (`PivotFloor::Trace`, Σ kept): the polish moved the
                        // entries below a diagonal it held at 0 as free
                        // coordinates, and an entry under an earlier zero
                        // diagonal folds onto a later one. The flags are the
                        // diagonals that fold leaves exactly 0; a new diagonal
                        // in (0, `PIN_THETA`] is not pinned, though the trace
                        // floor still zeroes one below the round-off of Σ. On
                        // the AGQ route θ keeps the polish's Λ and the fold
                        // runs on the copy.
                        params[..n_theta].copy_from_slice(&start_point[..n_theta]);
                        let mut canon_copy = [0.0_f64; crate::consts::MAX_THETA];
                        (pinned, pinned_components) = crate::lmm::fold_endpoint(
                            groupings,
                            &mut params[..n_theta],
                            if agq_route {
                                Some(&mut canon_copy[..n_theta])
                            } else {
                                None
                            },
                            crate::lmm::re_map::PivotFloor::Trace,
                        );
                        final_deviance = reeval!();
                    }
                } else {
                    // Nothing lower: the pinned point is the fit, re-evaluated
                    // so ũ, W̃ and `M` are its own and not the polish's last
                    // point's.
                    params[..n_theta].copy_from_slice(&params_stage1[..n_theta]);
                    params[n_theta..tp].copy_from_slice(&guard_beta[..pp]);
                    if n_disp == 1 {
                        params[tp] = disp;
                    }
                    final_deviance = reeval!();
                }
            } else {
                // The pinned point stands; the probe left the workspace at
                // its last point, so it is re-evaluated once more (not a
                // search evaluation).
                final_deviance = reeval!();
            }
        }
    }
    ws.counters.record_post_pin(searches);
    ws.final_pirls_exhausted = final_exhausted_count > 0;
    PinOutcome {
        final_deviance,
        pinned,
        pinned_components,
        budget_exhausted,
        n_eval,
    }
}
