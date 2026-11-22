//! The gradient route of `fit_glmm`'s outer search (`gradient_route_plan`):
//! its objective `GradSearch` and the search itself.

use bobyqa::Status;

use super::bobyqa_route::{any_within, bobyqa_scaled, NON_FINITE_RADIUS};
use super::deviance::{laplace_deviance, EvalScratch};
use super::pirls::BetaMode;
use super::workspace::FitData;
use super::{
    agq_gradient, assembled, dispersion_bound_probe, grad_search_dim, saddle_walk, workspace,
    RouteResult, SearchCtx, SearchState,
};
use crate::lmm::RHO_BEGIN_WARM;

/// Rows per primary level from which an AGQ fit goes back to BOBYQA (see
/// [`gradient_route_plan`](super::gradient_route_plan)).
pub(super) const GRAD_WARM_SCALAR_ROWS_MAX: usize = 50;

/// A warm fit whose θ start has a diagonal `|θ|` above this takes the BOBYQA
/// route. From such far starts the joint `[θ | β]` search can end in a worse
/// local minimum than the profiled `ExactProfile` search does: 2 of 600
/// cloglog crossed starts at θ₀ = 20 and 1000 ended 0.27 and 1.09 above the
/// cold optimum, reported converged (`far_warm_start_scan`, 2026-10-07).
/// Warm starts from a truth or a previous estimate sit far below it.
pub(super) const GRAD_WARM_THETA_MAX: f64 = 10.0;

/// A warm AGQ fit whose θ start has a diagonal below this takes the BOBYQA
/// route. Measured on 48 warm rows (locked clock, 2026-10-10): every start at
/// θ₀ <= 0.02 was slower than warm BOBYQA (1.13x to 1.57x), every start at
/// θ₀ >= 0.05 at most 1.22x (geomean 0.826 over 38 rows), no start between
/// 0.02 and 0.05. A start below it ends near θ = 0, fails the certificate's
/// near-zero test and pays the rung-3 polish, which costs more than the warm
/// BOBYQA it replaces.
pub(super) const GRAD_WARM_AGQ_THETA_MIN: f64 = 0.05;

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
/// `gradient_route`), elsewhere from the design (the joint
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
    /// The AGQ gradient's own row index and buffers (`ws.agq_gd`); `Some`
    /// exactly when `nagq > 1`.
    agq_gd: Option<&'a mut agq_gradient::AgqGd>,
    /// The fit's evaluation buffers, handed to each value evaluation as the
    /// joint search hands them (`cluster_rows` is `None` at `nagq == 1`).
    sc: &'a mut EvalScratch<'d>,
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
    /// `Σ ln yᵢ`, the dispersion term's data constant (`SearchCtx`'s
    /// `gamma_sum_ln_y`); read on a free Gamma dispersion only.
    sum_ln_y: f64,
    /// Every β̃ entry's start inverse curvature, `1/(2·Σ prior weight)`, when
    /// β̃ is whitened by the β information (`gradient_route`'s `grad_beta_info`);
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
        self.sc.pirls.u[..k].copy_from_slice(&self.u_inc[..k]);
        let exhausted_before = *self.pirls_exhausted;
        let dev = laplace_deviance(
            d,
            self.sc,
            self.nb_theta,
            self.gamma_phi,
            self.nagq,
            self.x_last,
            BetaMode::Fixed,
            None,
            self.pirls_exhausted,
            self.counters,
        );
        // The Gamma arm of `Dispersion::term` — change together.
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
        self.u_inc[..k].copy_from_slice(&self.sc.pirls.u[..k]);
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
                self.sc.pirls,
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
            &self.sc.prior_w_phi[..n]
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
            self.sc.pirls,
            self.sc.structured,
            self.sc.pattern,
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
                self.sc.pirls.pen_dev,
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
        ) + self.sc.pirls.pen_dev;
        if c > 0.0 && c.is_finite() {
            inv_curv[q] = 1.0 / c;
        }
    }
}

/// What the gradient route leaves for the BOBYQA route and the hand-over.
#[derive(Default)]
pub(super) struct GradOutcome {
    /// The route's finish: a certified endpoint or the polish's result;
    /// `None` hands over.
    pub(super) done: Option<RouteResult>,
    /// Evaluations a search that handed over spent.
    pub(super) spent: usize,
    /// The search handed over to the BOBYQA route.
    pub(super) fell_back: bool,
    /// A handed-over incumbent's search dimension m and objective; its
    /// `[θ | β | ψ]` (caller's β) and modes are in the third part of
    /// `ws.grad_scratch`.
    pub(super) handover: Option<(usize, f64)>,
}

/// The gradient route: projected BFGS over `[θ | β̃]` from the start the
/// guard picked, then the saddle walk, the rung decision and, on rung 3,
/// the joint polish.
pub(super) fn gradient_route<'a>(
    ctx: &SearchCtx<'a>,
    st: SearchState<'_>,
    sc: &mut EvalScratch<'a>,
    guard_beta_kept: bool,
    info_border_ready: bool,
    grad_hint: &Option<crate::optim::CurvatureHint>,
) -> GradOutcome {
    let SearchCtx {
        data,
        disp,
        groupings,
        layout,
        prior_w,
        x,
        cluster_ids,
        structural,
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
        u_seed,
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
        ..
    } = st;
    // The gradient route (`gradient_route_plan`). Its result replaces the
    // BOBYQA search (`bobyqa_route::bobyqa_route`), whose code runs only as a fallback:
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
    //   `Note::GradientSearchFallback`. When that route returns a point higher
    //   than the incumbent by more than the band below, the incumbent is the
    //   fit's capped endpoint (`grad_incumbent`).
    //
    // On AGQ a fit this route finishes (a certified endpoint or the polish)
    // then goes through the joint search's check and second pass
    // (`JOINT_RECHECK_TOL`), as a joint BOBYQA pass does; a Laplace shape's
    // finish is the fit's result.
    //
    // `converged` is always the status of whichever optimizer finished.
    let mut grad_done: Option<RouteResult> = None;
    let mut grad_spent = 0usize;
    let mut grad_fell_back = false;
    // A handed-over incumbent's search dimension m and objective; its
    // `[θ | β | ψ]` (caller's β) and modes are in the third part of
    // `ws.grad_scratch`, which `report_handover` reads.
    let mut grad_incumbent: Option<(usize, f64)> = None;
    use crate::optim::bfgs::{bfgs_minimize, BfgsStop};
    debug_assert!(
        nagq == 1 || agq_gradient::covers_shape(family, nagq, layout, groupings),
        "gradient route on an AGQ shape the exact AGQ gradient does not cover"
    );
    // A free Gamma dispersion is searched as the trailing `ψ = ln φ`,
    // from the seed `start::seed_start` sets and on the BOBYQA route's box; on the inverse
    // link with its start curvature (`GradSearch::start_curvature`) unless
    // a hint is given. A held one leaves the search on `[θ | β̃]`.
    let free_phi = is_gamma && held_phi.is_none();
    let m = grad_search_dim(family, n_theta, p, held_phi.is_some());
    let m_search = grad_search_dim(family, n_free, p, held_phi.is_some());
    debug_assert!((n_disp == 0 || is_gamma) && params.len() == total_dim);
    if guard_beta_kept {
        // A warm fit: the guard profiled β at the θ it picked.
        params[n_theta..nb_col].copy_from_slice(&guard_beta[..p]);
    }
    // The search's start (length m), for a hand-over from the plateau
    // below, and the objective's last `[θ | β]` (length `n_θ + p`):
    // workspace scratch, so a refit allocates nothing here.
    grad_scratch.resize(m + nb_col + m + k, 0.0);
    let (grad_start, rest) = grad_scratch.split_at_mut(m);
    let (x_last, incumbent) = rest.split_at_mut(nb_col);
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
            sc.beta_rhs[..p].fill(0.0);
            sc.beta_rhs[j] = 1.0;
            beta_scale.to_search(&mut sc.beta_rhs[..p]);
            for i in 0..p {
                lset[i * p + j] = sc.beta_rhs[i];
            }
        }
        let ((nb, phi), _) = disp.at(params, nb_col);
        sc.beta_prof[..p].copy_from_slice(&params[n_theta..nb_col]);
        sc.pirls.u[..k].copy_from_slice(&u_seed[..k]);
        let mut info_exhausted = 0u32;
        let mut info_counters = crate::counters::EvalCounters::new();
        // A kept warm start's guard objective was finite.
        let finite = info_border_ready
            || laplace_deviance(
                &data,
                sc,
                nb,
                phi,
                1,
                &params[..n_theta],
                BetaMode::ProfilePql,
                None,
                &mut info_exhausted,
                &mut info_counters,
            )
            .is_finite();
        let sw: f64 = prior_w[..n].iter().sum();
        if finite && beta_scale.set_info(sc.border.schur.as_mut(), sw) {
            *grad_beta_info = true;
            beta_inv_curv = Some(1.0 / (2.0 * sw));
            // T = L_setᵀ·L_info⁻ᵀ: column j is `L_setᵀ` applied to the
            // caller-units β of the information-whitened unit vector e_j.
            for j in 0..p {
                sc.beta_rhs[..p].fill(0.0);
                sc.beta_rhs[j] = 1.0;
                beta_scale.to_caller(&mut sc.beta_rhs[..p]);
                for i in 0..p {
                    t[i * p + j] = (0..p).map(|c| lset[i * p + c] * sc.beta_rhs[c]).sum();
                }
            }
        } else {
            beta_scale.set(x, &prior_w[..n], n);
        }
    }
    beta_scale.to_search(&mut params[n_theta..nb_col]);
    // AGQ: the gradient's row index and buffers, built on the first such
    // fit and kept for the SE step's gradient-difference Hessian.
    if nagq > 1 && !agq_gd.as_ref().is_some_and(|g| g.matches_shape(p, nagq)) {
        *agq_gd = Some(Box::new(agq_gradient::AgqGd::for_shape(p, nagq)));
    }
    let mut agq_gd = agq_gd.as_deref_mut().filter(|_| nagq > 1);
    if let Some(gd) = agq_gd.as_deref_mut() {
        gd.idx.ensure(&cluster_ids[..n], groupings.n_primary);
    }
    let mut obj = GradSearch {
        data: &data,
        nb_theta,
        nagq,
        agq_gd,
        sc,
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
        &mut crate::optim::Scaled::new(&mut obj, search_div),
        &mut params[..m],
        &lower[..m],
        &upper[..m],
        grad_cfg,
        bfgs,
        grad_hint
            .as_ref()
            .filter(|h| h.dim() == m_search)
            .map(|h| h.inv_hessian()),
        structural,
    );
    *grad_hint_ready = bo.n_grad >= 2;
    let mut n_val = bo.n_val;
    let mut best = bo.f * search_div;
    let mut moved = false;
    if bo.converged() {
        // β held at the endpoint's. `best` follows a kept try, whose mode
        // becomes the incumbent's; a kept try leaves the endpoint uncertified.
        let (mv, tries) =
            saddle_walk(
                groupings,
                &mut params[..m],
                |x| match crate::optim::Objective::value(&mut obj, x, 0.0) {
                    Some(f) if f < best => {
                        best = f;
                        obj.u_inc[..k].copy_from_slice(&obj.sc.pirls.u[..k]);
                        true
                    }
                    _ => false,
                },
            );
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
    *grad_report = Some((bo.stop, bo.n_grad, bo.decrement * search_div, rung));
    if rung == 0 {
        grad_done = Some(RouteResult {
            ok: true,
            n_eval: n_val,
            budget_exhausted: false,
            obj: best,
        });
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
                bobyqa::Bobyqa::new(total_free, c)
                    .expect("polish BOBYQA config is valid by construction")
            })
        } else {
            &mut *solver
        };
        let out = bobyqa_scaled(
            polish,
            &mut crate::optim::ValueFn(|gamma: &[f64], _rho: f64| {
                let f =
                    crate::optim::Objective::value(&mut obj, gamma, 0.0).unwrap_or(f64::INFINITY);
                if f.is_finite() {
                    finite += 1;
                }
                if f < pol_best {
                    pol_best = f;
                    obj.u_inc[..k].copy_from_slice(&obj.sc.pirls.u[..k]);
                }
                f
            }),
            &mut params[..total_dim],
            &lower[..total_dim],
            &upper[..total_dim],
            structural,
            search_div,
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
        grad_done = Some(RouteResult {
            ok,
            n_eval: n_val + out.n_eval,
            budget_exhausted: matches!(out.status, Status::MaxFunReached),
            obj: pol_best,
        });
    } else {
        // Rung 1: the hand-over to the BOBYQA route from the incumbent
        // (the start itself when its value was not finite).
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
    // The dispersion-bound check of the joint search
    // (`bobyqa_route::joint_passes`, which explains it), on a converged endpoint with a free Gamma φ: one solve
    // at ψ's lower bound, every other coordinate held, kept when lower.
    // Counted as an evaluation, as the saddle walk's tries are; its PIRLS
    // exhaustion is not counted, as the joint check's is not.
    if let Some(done) = grad_done
        .as_mut()
        .filter(|d| free_phi && d.ok && !d.budget_exhausted)
    {
        let kept = dispersion_bound_probe(&mut params[..m], nb_col, lower[nb_col], done.obj, |x| {
            done.n_eval += 1;
            let exhausted = *obj.pirls_exhausted;
            let f = crate::optim::Objective::value(&mut obj, x, 0.0);
            *obj.pirls_exhausted = exhausted;
            f
        });
        if let Some(f) = kept {
            done.obj = f;
            obj.u_inc[..k].copy_from_slice(&obj.sc.pirls.u[..k]);
        }
    }
    beta_scale.to_caller(&mut params[n_theta..nb_col]);
    if grad_fell_back && best.is_finite() {
        incumbent[..m].copy_from_slice(&params[..m]);
        incumbent[m..].copy_from_slice(&u_seed[..k]);
        grad_incumbent = Some((m, best));
    }
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
    GradOutcome {
        done: grad_done,
        spent: grad_spent,
        fell_back: grad_fell_back,
        handover: grad_incumbent,
    }
}
