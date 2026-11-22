//! Block-diagonal PIRLS solve for the no-extras regime (`pirls_solve_blocked`).

use super::control::{Merit, PirlsControl, Trial};
use super::*;

/// The block-diagonal Laplace layout: `A = M'WM + I` is `s` independent
/// `q×q` blocks because each row loads exactly one cluster's `q` columns of
/// `M = ZΛ_p`. Borrowed from the buffers `pirls_solve_blocked` destructures.
pub(crate) struct BlockedFactor<'a, T> {
    /// `s·q²`, cluster `f` at `f·q²`, row-major lower triangle. Left FACTORED
    /// (per-cluster Crout `L`) by `factor_logdet`.
    pub(crate) a_blocks: &'a mut [T],
    /// len `k`, packed `f·q + local`: `M'r`, then `A⁻¹M'r` in place.
    pub(crate) a_rhs: &'a mut [T],
    /// Row-major `n×q`: `mᵢ = Λ_p'·zᵢ`.
    pub(crate) m_buf: &'a [T],
    /// len `n`: the primary cluster each row loads.
    pub(crate) cluster_ids: &'a [u32],
    pub(crate) family: Family,
    pub(crate) nb_theta: f64,
    /// Primary RE width per cluster.
    pub(crate) q: usize,
    /// Number of primary clusters.
    pub(crate) s: usize,
    /// `q·s`, the packed RE length.
    pub(crate) k: usize,
    /// len `s`: each cluster's half `log|A_f|`, written by `factor_logdet`
    /// (the AGQ scale term reads it; `L`'s diagonal has lost its digits where
    /// `A_f` is within ε of `I`).
    pub(crate) half_logdet: &'a mut [T],
    /// `s·q²`, the layout of `a_blocks`: `factor_logdet` copies each block
    /// (`A_f − I`) into it before factoring (`PirlsScratch::a_excess`).
    pub(crate) excess: &'a mut [T],
}

impl<T: Scalar> LaplaceFactor<T> for BlockedFactor<'_, T> {
    fn eta_from_mode(&mut self, u: &[T], eta_fixed: &[T], eta: &mut [T], y: &[f64], n: usize) -> T {
        let (m_buf, cluster_ids, q) = (self.m_buf, self.cluster_ids, self.q);
        let mut yeta = T::ZERO;
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let ubase = cluster_ids[i] as usize * q;
            let mut e = eta_fixed[i];
            for c in 0..q {
                e += m_row[c] * u[ubase + c];
            }
            eta[i] = e;
            yeta += eta[i].scale(y[i]);
        }
        yeta
    }

    fn scatter(
        &mut self,
        w: &[T],
        prob: &[T],
        eta: &[T],
        y: &[f64],
        prior_w: &[f64],
        _weighted: bool,
        n: usize,
        mut dual: Option<&mut DualStep<T>>,
    ) {
        let (m_buf, cluster_ids) = (self.m_buf, self.cluster_ids);
        let (family, nb_theta, q, s, k) = (self.family, self.nb_theta, self.q, self.s, self.k);
        let a_blocks = &mut *self.a_blocks;
        let a_rhs = &mut *self.a_rhs;
        for v in a_blocks[..s * q * q].iter_mut() {
            *v = T::ZERO;
        }
        for v in a_rhs[..k].iter_mut() {
            *v = T::ZERO;
        }
        if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
            for v in d.obs_blocks[..s * q * q].iter_mut() {
                *v = T::ZERO;
            }
        }
        // --- scatter-pass (scalar): wᵢmᵢmᵢ' and ρᵢ·mᵢ into the blocks, ρᵢ the
        // row score (`row_rho`). ---
        let mut probit_rows = crate::family::ProbitRows::new(family, eta, prob);
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let f = cluster_ids[i] as usize;
            let ubase = f * q;
            let ablk = f * q * q;
            let wi = w[i];
            let resid = super::row_rho(
                family,
                nb_theta,
                i,
                y[i],
                prior_w[i],
                eta[i],
                prob[i],
                probit_rows.as_mut(),
            );
            for r in 0..q {
                a_rhs[ubase + r] += m_row[r] * resid;
                let wr = wi * m_row[r];
                for c in 0..=r {
                    a_blocks[ablk + r * q + c] += wr * m_row[c];
                }
            }
            // Observed-weight twin of the Fisher scatter above, into the
            // observed blocks (same lower-triangle layout).
            if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
                let wo = crate::family::observed_weight(
                    family, nb_theta, y[i], prior_w[i], eta[i], prob[i], wi,
                );
                for r in 0..q {
                    let wr = wo * m_row[r];
                    #[allow(clippy::needless_range_loop)]
                    for c in 0..=r {
                        d.obs_blocks[ablk + r * q + c] += wr * m_row[c];
                    }
                }
            }
        }
    }

    fn factor_logdet(&mut self) -> Option<T> {
        let (q, s) = (self.q, self.s);
        let a_blocks = &mut *self.a_blocks;
        let mut logdet = T::ZERO;
        for f in 0..s {
            let ablk = f * q * q;
            self.excess[ablk..ablk + q * q].copy_from_slice(&a_blocks[ablk..ablk + q * q]);
            let h = crate::linalg::block_chol_unit_shift(&mut a_blocks[ablk..ablk + q * q], q)?;
            self.half_logdet[f] = h;
            logdet += h;
        }
        Some(logdet)
    }

    fn solve_in_place(&mut self) {
        let (q, s) = (self.q, self.s);
        let a_blocks = &*self.a_blocks;
        let a_rhs = &mut *self.a_rhs;
        for f in 0..s {
            let ablk = f * q * q;
            glmm_block_solve(
                &a_blocks[ablk..ablk + q * q],
                q,
                &mut a_rhs[f * q..f * q + q],
            );
        }
    }
}

/// Block-diagonal PIRLS for the no-extras regime (`groupings.extra_offsets` empty):
/// `A = M'WM + I` is `s` independent `q_p×q_p` blocks because each row loads exactly
/// one cluster's `q_p` columns. `m_buf` (row-major n×q_p) holds `mᵢ = Λ_p'·zᵢ`
/// (`zᵢ = [1, x[i, slope_cols]]`, slope columns pre-widened per fit into `z_buf`
/// by `fill_z_f64`), filled once per solve since Λ and x are fixed within one.
/// The η-pass forms ηᵢ and the deviance from it; the scatter-pass accumulates
/// `wᵢ·mᵢmᵢ'` into cluster `i`'s block plus `mᵢ·(yᵢ−pᵢ)` into its RHS — keeping
/// the dense path's `O(n·k²)` Gram/RHS collapsed to `O(n·q_p²)`. Then per block: `rhs_f = (A_f−I)·u_f + g_f`, Crout factor (log|A|
/// off the pivots), solve `u_f`. NOT bit-identical to `pirls_solve_packed` (reordered
/// accumulation) but the same estimator. Mirrors `pirls_solve_packed`'s half-step
/// (w/A from the pre-update u, u updated after) so the two agree to FP error.
/// `lam` is Λ_p row-major (`lam[r·q+c]`) from `primary_lambda`. Leaves `a_blocks`
/// FACTORED (per-block L of the final iterate) for `blocked_schur_fill` to reuse,
/// and eta/prob/w/u filled. Returns `(dev, ‖u‖², log|A|, converged)`; a non-PD
/// block ⇒ `(NaN, NaN, NaN, false)`. Iterates from the caller-provided `u` (the
/// warm-start seed); the caller owns resetting it per fit.
///
/// Step-halving: as the step-halving paragraph of `pirls_solve_blocked_extras`'
/// doc describes for each `beta_step`, with `a_blocks` in place of the
/// structured factor. A converged solve ends by
/// re-evaluating η/μ/W at the returned `u` and rebuilding and refactoring the
/// blocks there ([`evaluate_at_mode`]), so `a_blocks`, `log|A|` and `dev` on
/// return describe that iterate, preserving the `blocked_schur_fill` contract
/// above.
///
/// **β mode (`beta_step`):** `Fixed` holds β at the caller's input (β read-only,
/// FD-Hessian / stage-2 contract). `Profile` adds a joint δβ Schur-border step
/// each iteration — run AFTER the whole block sweep, mirroring `blocked_schur_fill`
/// with the live iteration's W and per-block factors — so the returned (ũ, β̂) is
/// jointly PQL-optimal for this θ, β̂ written back through `beta`. A non-PD S_β
/// surfaces as `(NaN, NaN, NaN, false)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pirls_solve_blocked<T: Scalar>(
    family: Family,
    nb_theta: f64,
    g: &crate::lmm::LmmGroupings,
    cluster_ids: &[u32],
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    beta: &mut [T],
    mut beta_step: BetaStep,
    scratch: &mut PirlsScratch<T>,
    z_buf: &[f64],
    // Dual derivative kernels' per-solve controls (see `DualStep`); `None` on
    // every f64 fit-path call.
    mut dual: Option<&mut DualStep<T>>,
    // n×p scratch: the design residual rows `rᵢ = xᵢ − Tᵀmᵢ` of the Profile β
    // border (`residual_border`). This variant has no dense M, so B' = X'WM is
    // filled by cluster-scatter.
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`Data::offset`), added into
    // `eta_fixed` by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
    // Observation-only — mirrors `pirls_solve_packed`'s `counters`, where
    // the contract is stated.
    counters: &mut crate::counters::EvalCounters,
) -> (T, T, T, bool) {
    // The β-Schur border below runs in f64 (its p×p Cholesky is faer's, and
    // reproducing it generically would move f64 bits). A non-f64
    // instantiation must use BetaStep::Fixed — the derivative entry points do.
    // Loud, not silent: dropping
    // derivatives here would be a wrong gradient, not a slow one.
    assert!(
        T::IS_F64 || matches!(beta_step, BetaStep::Fixed),
        "BetaStep::Profile is f64-only"
    );
    let PirlsScratch {
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        m_buf,
        lam,
        a_blocks,
        a_rhs,
        a_half_logdet,
        a_excess,
        x_rows,
        bt_rows,
        ..
    } = scratch;
    let q = g.primary_q;
    let s = g.n_primary;
    let k = q * s;
    let p = beta.len();
    // The `f64` β-Profile solve reads `X` row by row in its hot passes
    // (`B' = X'WM`, the exact `c_β` passes, `η_fixed`), so it takes a
    // row-major copy once per solve (`PirlsScratch::x_rows`): one contiguous
    // row per observation instead of `p` bounds-checked `MatRef` lookups. Every
    // accumulator sums in the same row order as on the `MatRef` path.
    let rows_ok = T::IS_F64
        && dual.is_none()
        && p > 0
        && matches!(beta_step, BetaStep::Profile { .. })
        && x_rows.len() >= n * p
        && bt_rows.len() >= k * p;
    if rows_ok {
        fill_x_rows(x, &mut x_rows[..n * p], n, p);
    }
    let xr_all: &[f64] = if rows_ok { &x_rows[..n * p] } else { &[] };
    let fx = FixedPart {
        x,
        rows: rows_ok.then_some(xr_all),
        n,
        p,
        offset,
    };
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    refresh_fixed(&fx, beta, eta_fixed);
    // M = ZΛ_p (mᵢ = Λ_p'·zᵢ, zᵢ = [1, x[i, slope_cols]] pre-widened into z_buf
    // per fit) is invariant within one solve — Λ and x are fixed; the iteration
    // only mutates u/η/prob/w/blocks. Fill it once per solve. A per-row
    // recompute costs ~57% of fit runtime on the glmm_slope profile, over 90% of
    // that MatRef indexing, bounds checks and 64-byte return-by-value copies
    // rather than FMA: buffering removes the overhead, not the math. Bit-identical: the same `zlam_row` reduction runs once per
    // (i,c) in the same inner order, and both consumers below read the same f64
    // values in the same order.
    for i in 0..n {
        super::zlam_row(&z_buf[i * (q - 1)..], lam, q, &mut m_buf[i * q..i * q + q]);
    }
    // The `A`-layout view over the buffers the assembly, the factor and the
    // `A⁻¹` solve own. `m_buf` is complete above and read-only from here, so
    // the loop below keeps reading it directly alongside the view.
    let mut layout = BlockedFactor {
        a_blocks: &mut a_blocks[..],
        a_rhs: &mut a_rhs[..],
        m_buf: &m_buf[..],
        cluster_ids,
        family,
        nb_theta,
        q,
        s,
        k,
        half_logdet: &mut a_half_logdet[..],
        excess: &mut a_excess[..],
    };
    // First-trial backtrack seeds (u_prev = 0, beta_prev = caller's β) — only
    // the domain-infeasibility trigger can read them; see `pirls_solve_packed`.
    u_prev[..k].fill(T::ZERO);
    if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
        for j in 0..p {
            beta_prev[j] = beta[j].value();
        }
    }
    // `ex.u_acc` is workspace-persistent (survives across `laplace_deviance`
    // calls at different θ) and is only ever written by an ACCEPTED post-sweep
    // iterate — reseed it here so a pre-first-accept halving in exact mode
    // targets this solve's own u_prev seed, not a stale value left by a
    // previous call.
    if let BetaStep::Profile {
        exact: Some(ex), ..
    } = &mut beta_step
    {
        ex.u_acc[..k].fill(0.0);
    }
    let mut converged = false;
    let mut ctl = PirlsControl::new(family, &beta_step, dual.as_deref_mut(), pirls_tol_override);
    let (mut dev, mut pen, mut logdet) = (T::from_f64(f64::NAN), T::from_f64(f64::NAN), T::ZERO);
    for it in 0..PIRLS_MAX_ITERS {
        counters.set_pirls_iters(it + 1);
        // --- trial evaluation at the CURRENT u: η-pass then η→prob/w/dev. On a
        // fresh accept this is the newly-stepped u; after a halving `continue` it
        // is the backtracked u. Either way the recompute IS the trial evaluation.
        // Loop-split: the transcendental runs vectorized over a materialized η[]
        // with no gather/scatter data deps.
        // --- pass 1: η-pass (scalar gather): form ηᵢ and Σ y·η in one pass ---
        let yeta = layout.eta_from_mode(&u[..], &eta_fixed[..], &mut eta[..], y, n);
        // --- pass 2: η[] → prob[]/w[] + deviance, through the shared family
        // kernel (clamps η in place; `infeasible` flags any raw η past the
        // link's clamp bounds, `family::eta_infeasible`; mirrors
        // `pirls_solve_packed`). ---
        let (d, infeasible) = T::family_pass(
            family,
            nb_theta,
            &mut eta[..n],
            &y[..n],
            &prior_w[..n],
            weighted,
            yeta,
            &mut prob[..n],
            &mut w[..n],
            &mut [],
        );
        dev = d;
        // ‖u‖² is at the CURRENT trial u.
        let pen_u = pen_sq(&u[..k]);
        let penalized = dev + pen_u;
        match ctl.trial(penalized.value(), infeasible) {
            Trial::Proceed => {}
            Trial::RefineJump => {
                // Past the η clamp: one jump back inside it, toward `u_prev`.
                jump_back(
                    &mut layout,
                    family,
                    y,
                    &fx,
                    &BetaStep::Fixed,
                    u,
                    u_prev,
                    beta,
                    eta_fixed,
                    prob,
                    eta,
                    w,
                    k,
                );
                continue;
            }
            Trial::RefineHalve => {
                halve_u(&mut u[..k], &u_prev[..k]);
                continue;
            }
            Trial::Jump => {
                // Past the η clamp: one jump to the largest power-of-two
                // fraction of the step that is back inside it
                // (`clamp_jump`), where halving could take more than the
                // cap allows.
                jump_back(
                    &mut layout,
                    family,
                    y,
                    &fx,
                    &beta_step,
                    u,
                    u_prev,
                    beta,
                    eta_fixed,
                    prob,
                    eta,
                    w,
                    k,
                );
                continue;
            }
            Trial::Halve => {
                // Last full step overshot: halve δu toward the halving target
                // and re-enter the top (the recompute above is the trial
                // evaluation of the halved step). Exact mode halves toward the
                // last ACCEPTED u (`u_acc`, set by the post-sweep merit test
                // below) since `u_prev` there holds the just-rejected trial,
                // not an accepted point.
                if let BetaStep::Profile {
                    exact: Some(ex), ..
                } = &beta_step
                {
                    halve_u_to_acc(&mut u[..k], &ex.u_acc[..k]);
                } else {
                    halve_u(&mut u[..k], &u_prev[..k]);
                }
                // Profile mode: the trial point is the JOINT (u,β) step, so the
                // backtrack halves β toward `beta_prev` in lockstep with u, then
                // refreshes η_fixed for the re-evaluation at the top. Mirrors
                // `pirls_solve_packed`'s Profile backtrack.
                if let BetaStep::Profile { beta_prev, .. } = &beta_step {
                    halve_beta(beta, beta_prev);
                    refresh_fixed(&fx, beta, eta_fixed);
                }
                continue;
            }
            Trial::Fail => return fail(),
        }
        // Snapshot the accepted iterate for the next backtrack, and take a
        // fresh full step from it. `u_prev` is unconditional — the β-Schur
        // border needs δu₀ = u_new − u_prev regardless of mode.
        u_prev[..k].copy_from_slice(&u[..k]);
        if !ctl.exact {
            // Profile mode: snapshot the accepted β as the β-halving twin of u_prev.
            if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
                for j in 0..p {
                    beta_prev[j] = beta[j].value();
                }
            }
        }
        // Newton step: on a link whose exact curvature differs from Fisher the
        // step weight is `W_obs` (see `observed_weights_in_place`), so `A` below
        // is the penalized deviance's Hessian in u and the step converges
        // quadratically, as it does on a canonical link. The dual kernels step
        // with `A_obs` through their own twin (`DualStep::observed`) and keep
        // the Fisher `w` it is built from. Mirrors `pirls_solve_blocked_extras`
        // and `pirls_solve_packed` — change together.
        if dual.is_none() {
            observed_weights_in_place(
                family,
                nb_theta,
                y,
                prior_w,
                &eta[..],
                &prob[..],
                &mut w[..],
                n,
            );
        }
        // --- pass 3: the scatter — wᵢmᵢmᵢ' into cluster i's block, rᵢ·mᵢ into
        // its RHS — with the dual kernels' observed twin filled in the same row
        // pass. On the `f64` β-Profile solve the same row pass also forms the
        // border's `B = M'WX` (`fused_profile_rows`), summed in the same row
        // order with the same expressions as the separate pass. ---
        if rows_ok {
            if let BetaStep::Profile { .. } = &mut beta_step {
                fused_profile_rows(
                    family,
                    nb_theta,
                    y,
                    prior_w,
                    &eta[..],
                    &prob[..],
                    &w[..],
                    &m_buf[..],
                    cluster_ids,
                    xr_all,
                    q,
                    s,
                    p,
                    n,
                    &mut layout.a_blocks[..],
                    &mut layout.a_rhs[..],
                    &mut bt_rows[..k * p],
                );
            }
        } else {
            layout.scatter(
                &w[..],
                &prob[..],
                &eta[..],
                y,
                prior_w,
                weighted,
                n,
                dual.as_deref_mut(),
            );
        }
        // --- per block: rhs_f = g_f − u_old_f ; +I ; factor ; solve δu_f ;
        // u_new_f = u_old_f + δu_f.
        // The step is solved as the increment `A·δu = M'ρ − u_old`, not as the
        // equal `A·u_new = (A − I)·u_old + M'ρ`. That form carries
        // `(A − I)·u_old`, of size θ²·Σwᵢ·|u|, through the right-hand side and
        // the solve, and its rounding lands in u_new undamped along any
        // direction where A's eigenvalue is near 1 — a direction M (nearly)
        // annihilates: crossed intercepts, a slope constant within a cluster.
        // In the increment form the same rounding scales with |δu| and dies out
        // as the solve converges. Measured on a Poisson `(1|g) + (1|h)` at
        // counts near e³³ (θ²·Σwᵢ ≈ 1e15): ‖u‖² moved by 3e-3 between iterates,
        // far outside the 1e-9 band, and the solve ran out its halvings; the
        // increment form converges in 7 iterations. Mirrors
        // `pirls_solve_blocked_extras` and `pirls_solve_packed` — change
        // together.
        // `log|A|` comes off the factor that produces this step's u; the exit
        // refresh below replaces it and a_blocks with their values at the
        // returned u (a step may be re-taken after a halving, so every
        // iteration re-factors). The observed twin solves the same right-hand
        // side on its own factor, staged in `obs_rhs` (the `a_rhs` packing);
        // the main block is formed and factored on every route — it is what
        // `log|A|` and the returned factor come from.
        for (r, &uc) in layout.a_rhs[..k].iter_mut().zip(&u[..k]) {
            *r -= uc;
        }
        if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
            d.obs_rhs[..k].copy_from_slice(&layout.a_rhs[..k]);
        }
        logdet = match layout.factor_logdet() {
            Some(ld) => ld,
            None => {
                // An unevaluable trial in exact mode is a REJECTED trial, not
                // a failed solve ([`PirlsControl::factor_failed`]).
                if ctl.factor_failed() {
                    if let BetaStep::Profile {
                        exact: Some(ex),
                        beta_prev,
                        ..
                    } = &beta_step
                    {
                        halve_prev_to_acc(&mut u[..k], &u_prev[..k], &ex.u_acc[..k]);
                        halve_beta(beta, beta_prev);
                        refresh_fixed(&fx, beta, eta_fixed);
                    }
                    continue;
                }
                return fail();
            }
        };
        layout.solve_in_place();
        pen = T::ZERO;
        for f in 0..s {
            let ablk = f * q * q;
            let ubase = f * q;
            // u_new_f = u_old_f + A_f⁻¹ rhs_f, solved above in `a_rhs`.
            for r in 0..q {
                u[ubase + r] += layout.a_rhs[ubase + r];
            }
            // Observed step: +I, factor, solve the staged right-hand side. A
            // non-PD observed block leaves this cluster on its Fisher solve.
            if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
                let DualStep {
                    obs_blocks,
                    obs_rhs,
                    exact,
                    ..
                } = &mut *d;
                let ob = &mut obs_blocks[ablk..ablk + q * q];
                for r in 0..q {
                    ob[r * q + r] += T::ONE;
                }
                if glmm_block_chol(ob, q) {
                    glmm_block_solve(ob, q, &mut obs_rhs[ubase..ubase + q]);
                    for r in 0..q {
                        u[ubase + r] = u_prev[ubase + r] + obs_rhs[ubase + r];
                    }
                } else {
                    *exact = false;
                }
            }
            for r in 0..q {
                pen += u[ubase + r] * u[ubase + r];
            }
        }
        let exact_obj = ctl.exact_obj;
        // Exact mode: assemble c_β off THIS iteration's factors, then judge the
        // trial against the full Laplace merit. Both live here rather than in the
        // β-Schur border below: the assembly reads only the block factors and the
        // live W (nothing the border produces), and the merit needs the `g_u` its
        // first pass forms. The border still runs on the accepted-or-halved u/β,
        // so this must land before it. A rise past the tol band re-halves toward
        // the last accepted (u, β) exactly like the retrospective test above,
        // just on the merit that actually matters; the first trial always
        // accepts (`l_acc = ∞`).
        if let BetaStep::Profile {
            exact: Some(ex),
            beta_prev,
            ..
        } = &mut beta_step
        {
            // Exact-profile correction: c_β = d log|A|/dβ, entering the same
            // Newton RHS with the same ½ as the objective's `2·logdet` term
            // (`logdet = ½ log|A|`). Direct part: Σᵢ w'ᵢ·xᵢⱼ·hᵢ, hᵢ = mᵢ'A⁻¹mᵢ the
            // RE leverage (`block_leverage` on this iteration's factor). û path:
            // β also moves ũ(β), so log|A|(β) picks up gᵤ'·dũ/dβ with
            // gᵤ = ∂log|A|/∂u and dũ/dβ = −Ã⁻¹M'W̃X; folded as ONE adjoint solve
            // v = Ã⁻¹gᵤ (pass B) rather than materializing the k×p `dũ/dβ` —
            // one adjoint solve instead of a k×p buffer for a term only ever
            // left-multiplied by a row. Ã = A and W̃ = W, the factor in `a_blocks`
            // and the step weight: the Hessian of the penalized deviance in u on
            // every link, the step being Newton.
            let gu_dot_du = {
                let ExactProfileBufs {
                    logdet_u,
                    logdet_beta,
                    fac_f64,
                    curv_h,
                    curv_sdot,
                    ..
                } = &mut **ex;
                let logdet_u = &mut logdet_u[..k];
                let logdet_beta = &mut logdet_beta[..p];
                logdet_u.fill(0.0);
                logdet_beta.fill(0.0);
                // One f64 mirror of the s per-cluster factors for the three
                // passes below (see `ExactProfileBufs::fac_f64`). Built from
                // THIS iterate's factors: the block sweep above leaves
                // `a_blocks` factored, and nothing between here and pass B
                // writes it.
                let fac_f64 = &mut fac_f64[..q * q * s];
                for (o, v) in fac_f64.iter_mut().zip(layout.a_blocks[..q * q * s].iter()) {
                    *o = v.value();
                }
                // pass A: hᵢ, w'ᵢ → direct part into c_β, and gᵤ = ∂log|A|/∂u.
                // The `dw/dη` of the curvature `log|A|` is built from
                // `dW_obs/dη` where the objective is `log|A_obs|` (`exact_obj`),
                // the Fisher one otherwise (`row_weight_eta_deriv`). Its per-row
                // family dispatch is hoisted out of the row loop on the two
                // canonical links, with `row_weight_eta_deriv`'s own closed form
                // (logit `w(1−2μ)` off the tail, Poisson `w`), so the same
                // value; `hᵢ` is kept for the border's curvature
                // (`logdet_beta_curvature`).
                {
                    let pa = ExactPassA {
                        fac: &fac_f64[..],
                        m_buf: &m_buf[..],
                        cluster_ids,
                        x,
                        xr_all,
                        q,
                        p,
                        n,
                    };
                    let general = |i: usize| {
                        super::row_weight_eta_deriv(
                            family,
                            nb_theta,
                            exact_obj,
                            y[i],
                            prior_w[i],
                            eta[i].value(),
                            prob[i].value(),
                            w[i].value(),
                        )
                    };
                    match family {
                        Family::Binomial {
                            link: BinomialLink::Logit,
                        } if !exact_obj => pa.run(curv_h, logdet_beta, logdet_u, |i| {
                            let pr = prob[i].value();
                            // The binomial arm of `in_tail` does not read η.
                            if crate::family::in_tail(family, 0.0, pr) {
                                general(i)
                            } else {
                                w[i].value() * (1.0 - 2.0 * pr)
                            }
                        }),
                        Family::Poisson { .. } if !exact_obj => {
                            pa.run(curv_h, logdet_beta, logdet_u, |i| w[i].value())
                        }
                        // Probit `dW_obs/dη` off a φ tile (`family::ProbitRows`).
                        _ if exact_obj => match crate::family::ProbitRows::new(family, eta, prob) {
                            Some(rows) => {
                                let rows = std::cell::RefCell::new(rows);
                                pa.run(curv_h, logdet_beta, logdet_u, |i| {
                                    rows.borrow_mut()
                                        .observed_weight_eta_deriv(i, y[i], prior_w[i])
                                })
                            }
                            None => pa.run(curv_h, logdet_beta, logdet_u, general),
                        },
                        _ => pa.run(curv_h, logdet_beta, logdet_u, general),
                    }
                }
                // Mode-consistency term for the merit below, formed here because
                // pass B overwrites `logdet_u` in place. See the merit's own
                // comment for why it is needed; δu₀ = u_new − u_prev is this
                // iteration's step toward the mode.
                let mut gu_dot_du = 0.0;
                for c in 0..k {
                    gu_dot_du += logdet_u[c] * (u[c] - u_prev[c]).value();
                }
                // pass B: v = Ã⁻¹ gᵤ per cluster.
                for f in 0..s {
                    let ablk = f * q * q;
                    glmm_block_solve(
                        &fac_f64[ablk..ablk + q * q],
                        q,
                        &mut logdet_u[f * q..f * q + q],
                    );
                }
                // pass C: û path, c_β_j −= Σᵢ w̃ᵢ·(mᵢ·v_f)·xᵢⱼ.
                if rows_ok {
                    with_const_p!(p, P => exact_pass_c_rows::<T, P>(
                        &m_buf[..],
                        &w[..],
                        logdet_u,
                        cluster_ids,
                        xr_all,
                        q,
                        p,
                        n,
                        curv_sdot,
                        logdet_beta,
                    ));
                } else {
                    for i in 0..n {
                        let f = cluster_ids[i] as usize;
                        let mut sdot = 0.0;
                        for c in 0..q {
                            sdot += m_buf[i * q + c].value() * logdet_u[f * q + c];
                        }
                        curv_sdot[i] = sdot;
                        let a = w[i].value() * sdot;
                        for j in 0..p {
                            logdet_beta[j] -= a * x[(i, j)];
                        }
                    }
                }
                gu_dot_du
            };
            // The Laplace objective is `dev + ‖u‖² + log|A|` AT the conditional
            // mode ũ(β); at a trial u off the mode it is not, and the two error
            // terms are not the same order. `dev + ‖u‖²` is stationary at ũ, so
            // it errs by O(‖u−ũ‖²) and from above; `log|A(u)|` errs at FIRST
            // order, either sign. Comparing raw sums across iterates therefore
            // compares points at different mode-offsets, and a trial sitting a
            // first-order step off the mode can score BELOW the attainable
            // optimum — a warm start from a neighbouring θ lands exactly there,
            // and taking it as `l_acc` rejects every later (correct) iterate
            // until the iteration cap. Undo that first-order part with the
            // gradient already at hand: g_u'·δu₀ ≈ log|A(ũ)| − log|A(u)|, a
            // correction that vanishes as δu₀ → 0. The u-step is Newton on the
            // same A that log|A| uses, so δu₀ tracks ũ−u to second order and
            // the corrected merit equals the Laplace deviance to second order in
            // δu₀ (the residual is what the accept band below absorbs).
            let l_trial = (dev + pen_u).value() + 2.0 * logdet.value() + gu_dot_du;
            match ctl.merit(l_trial, gu_dot_du, penalized.value()) {
                Merit::Refine => continue,
                Merit::Reject => {
                    halve_prev_to_acc(&mut u[..k], &u_prev[..k], &ex.u_acc[..k]);
                    halve_beta(beta, beta_prev);
                    refresh_fixed(&fx, beta, eta_fixed);
                    continue;
                }
                Merit::Fail => return fail(),
                Merit::Accept => {}
            }
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                ex.u_acc[c] = u_prev[c].value();
            }
            for j in 0..p {
                beta_prev[j] = beta[j].value();
            }
        }
        // --- Profile-mode joint δβ step (β-Schur border), run AFTER the whole
        // block sweep so every per-cluster factor of A is live in `a_blocks` and
        // δu₀ = u_new − u_prev is complete (u holds u_new, u_prev the pre-step
        // iterate). Mirrors `se::blocked_schur_fill` with THIS iteration's W and
        // this iteration's per-block factors: T = A⁻¹B, S_β = C − B'T,
        // δβ = S_β⁻¹·(X'ρ − B'δu₀), then u_joint = u_new − T·δβ (see
        // `se::packed_schur_fill`'s doc comment for the shared β-Schur Newton step
        // this mirrors), with S_β and the right-hand side formed from the design
        // residual rows (`residual_border`). Exact mode steps with
        // S_β + ½·d²log|A|/dβ² on X'ρ − B'δu₀ − ½c_β (`logdet_beta_curvature`,
        // `border_solve`). `m_buf` already holds mᵢ = Λ'zᵢ, so B's scatter reads
        // it directly (cheaper than se.rs's per-row reconstruction). ---
        if let BetaStep::Profile {
            exact,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            schur_llt_mem,
            ..
        } = &mut beta_step
        {
            // B' = X'WM (p×k), blocked: zero, then scatter the q_p coupling columns
            // per row into cluster f's column band. Uses the live `m_buf[i·q+c] = mᵢ`.
            if rows_ok {
                // Accumulated by the fused row pass (`fused_profile_rows`).
                for c in 0..k {
                    let src = &bt_rows[c * p..c * p + p];
                    for r in 0..p {
                        xtwm[(r, c)] = src[r];
                    }
                }
            } else {
                for r in 0..p {
                    for c in 0..k {
                        xtwm[(r, c)] = 0.0;
                    }
                }
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    let wi = w[i].value();
                    for r in 0..p {
                        let xw = x[(i, r)] * wi;
                        for c in 0..q {
                            xtwm[(r, f * q + c)] += xw * m_buf[i * q + c].value();
                        }
                    }
                }
            }
            // T_f = A_f⁻¹ (M'WX)_f per block, reusing this iteration's factor left in
            // `a_blocks`; ainv_mtwx rows f·q.. hold T_f. (M'WX)_f[c, col] = xtwm[(col, f·q+c)].
            // Mirrors `se::blocked_schur_fill`'s `T_f` solve; change together.
            // On the row-major path the
            // solve runs on all p columns at once, in place on `bt_rows`
            // (which holds (M'WX)_f row-major, `[(f·q + c)·p + col]`, and then
            // T row-major for the residual-row kernel): the same operations,
            // in the same order, per column as `glmm_block_solve`.
            if rows_ok && p <= BORDER_PMAX {
                for f in 0..s {
                    let ablk = f * q * q;
                    let l = &layout.a_blocks[ablk..ablk + q * q];
                    let blk = &mut bt_rows[f * q * p..(f + 1) * q * p];
                    with_const_p!(p, P => block_solve_cols_p::<T, P>(l, q, p, blk));
                    for c in 0..q {
                        for col in 0..p {
                            ainv_mtwx[(f * q + c, col)] = blk[c * p + col];
                        }
                    }
                }
            } else {
                for f in 0..s {
                    let ablk = f * q * q;
                    for col in 0..p {
                        let mut rhs = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
                        for c in 0..q {
                            rhs[c] = T::from_f64(xtwm[(col, f * q + c)]);
                        }
                        glmm_block_solve(&layout.a_blocks[ablk..ablk + q * q], q, &mut rhs[..q]);
                        for c in 0..q {
                            ainv_mtwx[(f * q + c, col)] = rhs[c].value();
                        }
                    }
                }
            }
            // Residual rows rᵢ = xᵢ − T_fᵀmᵢ, then S_β and the right-hand side
            // off them. A block-diagonal, so the per-block solves equal the
            // full A⁻¹M'WX and row i couples to its cluster's q columns only.
            // On the row-major path each tile of residual rows goes straight
            // into the border sums (`BorderAcc`) while it is in cache, and
            // `wx` is not written.
            if rows_ok && p <= BORDER_PMAX {
                let mut tile = [0.0f64; BORDER_PMAX * BORDER_TILE];
                let mut wt = [0.0f64; BORDER_TILE];
                let mut rt = [0.0f64; BORDER_TILE];
                let mut acc = BorderAcc::new();
                let scores = super::ProfileScores::new(family, nb_theta, y, prior_w, eta, prob);
                let mut i0 = 0;
                while i0 < n {
                    let len = (n - i0).min(BORDER_TILE);
                    with_const_p!(p, P => residual_tile_p::<T, P>(
                        xr_all, &bt_rows[..k * p], &m_buf[..], cluster_ids, q, p, i0, len,
                        &mut tile,
                    ));
                    for ii in 0..len {
                        let i = i0 + ii;
                        wt[ii] = w[i].value();
                        rt[ii] = scores.at(i);
                    }
                    let mut cols: [&[f64]; BORDER_PMAX] = [&[]; BORDER_PMAX];
                    for (col, c) in cols[..p].iter_mut().enumerate() {
                        *c = &tile[col * BORDER_TILE..col * BORDER_TILE + len];
                    }
                    acc.add_tile(&cols[..p], &wt[..len], &rt[..len], true);
                    i0 += len;
                }
                let mut tc: [&[f64]; BORDER_PMAX] = [&[]; BORDER_PMAX];
                let tm = ainv_mtwx.as_ref().subrows(0, k);
                assert!(border_cols(tm, &mut tc), "faer Mat columns are contiguous");
                acc.add_rows(&tc[..p], k, |_| 1.0, |j| u_prev[j].value(), true);
                acc.write(p, schur, Some(&mut beta_rhs[..p]));
            } else {
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    for col in 0..p {
                        let mut r = x[(i, col)];
                        for c in 0..q {
                            r -= m_buf[i * q + c].value() * ainv_mtwx[(f * q + c, col)];
                        }
                        wx[(i, col)] = r;
                    }
                }
                let scores = super::ProfileScores::new(family, nb_theta, y, prior_w, eta, prob);
                residual_border(
                    |i| w[i].value(),
                    |i| scores.at(i),
                    wx.as_ref().subrows(0, n),
                    ainv_mtwx.as_ref().subrows(0, k),
                    |j| u_prev[j].value(),
                    schur,
                    &mut beta_rhs[..p],
                );
            }
            // Exact mode: the step is Newton on the Laplace profile, so it also
            // carries log|A|'s gradient `c_β` and curvature in β
            // (`logdet_beta_curvature`), with the same ½ as the objective's
            // `2·logdet` term.
            if let Some(ex) = exact.as_deref_mut() {
                for r in 0..p {
                    beta_rhs[r] -= 0.5 * ex.logdet_beta[r];
                }
                if ctl.wants_curvature(it, &mut ex.curv_memory) {
                    logdet_beta_curvature(
                        ex,
                        x,
                        ainv_mtwx.as_ref(),
                        &m_buf[..],
                        cluster_ids,
                        g,
                        None,
                        family,
                        nb_theta,
                        exact_obj,
                        y,
                        prior_w,
                        &eta[..],
                        &prob[..],
                        n,
                        p,
                    );
                } else {
                    ex.logdet_hess.fill(0.0);
                }
            }
            // δβ in place, inside the trust region on the exact border. Non-PD
            // S_β ⇒ the (NaN,…,false) failure surface.
            if !border_solve(
                schur,
                &mut beta_rhs[..p],
                schur_llt_mem,
                exact.as_deref_mut().map(|ex| (ex, &mut ctl.trust)),
            ) {
                return fail();
            }
            apply_border(&fx, ainv_mtwx, beta_rhs, u, beta, eta_fixed, k);
            // `pen` must track the moved u (‖u_joint‖²), so recompute it.
            pen = pen_sq(&u[..k]);
        }
        if ctl.damp {
            ctl.trust.scale(0.5);
            damp_step(&fx, &beta_step, u, u_prev, beta, eta_fixed, k);
            pen = pen_sq(&u[..k]);
        }
        if ctl.stop(it, (dev + pen).value(), logdet.value()) {
            converged = true;
            break;
        }
    }
    // The returned `dev`, `log|A|` and per-block factor at the returned iterate,
    // so the objective's three terms and the factor `blocked_schur_fill`
    // inherits describe one point — see [`evaluate_at_mode`].
    if converged {
        match evaluate_at_mode(
            &mut layout,
            family,
            nb_theta,
            y,
            prior_w,
            weighted,
            &eta_fixed[..],
            &u[..],
            &mut eta[..],
            &mut prob[..],
            &mut w[..],
            n,
        ) {
            Some((d, ld)) => {
                dev = d;
                logdet = ld;
            }
            None => return fail(),
        }
    }
    (dev, pen, logdet, converged)
}

/// Row-major copy of the first `n` rows of `x` (`n×p`) into `out`
/// (`out[i·p + j] = x[i, j]`), column by column off the contiguous columns the
/// fit path's design always has. See `PirlsScratch::x_rows`.
pub(super) fn fill_x_rows(x: MatRef<f64>, out: &mut [f64], n: usize, p: usize) {
    for j in 0..p {
        match x.col(j).try_as_col_major() {
            Some(col) => {
                let col = &col.as_slice()[..n];
                for (i, &v) in col.iter().enumerate() {
                    out[i * p + j] = v;
                }
            }
            None => {
                for i in 0..n {
                    out[i * p + j] = x[(i, j)];
                }
            }
        }
    }
}

/// `refresh_eta_fixed` off the row-major `X` copy: the same `Σ_j x·β` per
/// row, in the same `j` order, with the offset added afterwards.
pub(super) fn refresh_eta_fixed_rows<T: Scalar>(
    xr: &[f64],
    p: usize,
    beta: &[T],
    eta_fixed: &mut [T],
    n: usize,
    offset: Option<&[f64]>,
) {
    with_const_p!(p, P => refresh_rows_p::<T, P>(xr, p, beta, eta_fixed, n));
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += T::from_f64(o[i]);
        }
    }
}

/// Run `$body` with `const $P: usize` equal to the runtime `p` for
/// `p ∈ 1..=8` (the row kernels below keep their `p`-long accumulators in
/// registers then), and with the `DYN` sentinel otherwise.
macro_rules! with_const_p {
    ($p:expr, $P:ident => $body:expr) => {
        match $p {
            1 => {
                const $P: usize = 1;
                $body
            }
            2 => {
                const $P: usize = 2;
                $body
            }
            3 => {
                const $P: usize = 3;
                $body
            }
            4 => {
                const $P: usize = 4;
                $body
            }
            5 => {
                const $P: usize = 5;
                $body
            }
            6 => {
                const $P: usize = 6;
                $body
            }
            7 => {
                const $P: usize = 7;
                $body
            }
            8 => {
                const $P: usize = 8;
                $body
            }
            _ => {
                const $P: usize = DYN;
                $body
            }
        }
    };
}
use with_const_p;

/// `const P` sentinel of [`with_const_p`]: the kernels read the runtime `p`.
const DYN: usize = 0;
/// Largest `p` the fixed-size accumulators hold.
const P_CAP: usize = 8;

/// The `f64` β-Profile solve's row pass: the blocked scatter (`wᵢmᵢmᵢ'` into
/// cluster `i`'s block, `rᵢ·mᵢ` into its right-hand side) and the border's
/// `B = M'WX` into `bt` (`k×p` row-major, `bt[(f·q + c)·p + r]`), in one
/// sweep. The scatter and `B'` accumulators get the same terms, in the same
/// row order, as `BlockedFactor::scatter` and the border's `B'` scatter
/// compute them.
#[allow(clippy::too_many_arguments)]
fn fused_profile_rows<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    eta: &[T],
    prob: &[T],
    w: &[T],
    m_buf: &[T],
    cluster_ids: &[u32],
    xr_all: &[f64],
    q: usize,
    s: usize,
    p: usize,
    n: usize,
    a_blocks: &mut [T],
    a_rhs: &mut [T],
    bt: &mut [f64],
) {
    with_const_p!(p, P => fused_profile_rows_p::<T, P>(
        family, nb_theta, y, prior_w, eta, prob, w, m_buf, cluster_ids, xr_all, q, s, p, n,
        a_blocks, a_rhs, bt,
    ))
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn fused_profile_rows_p<T: Scalar, const P: usize>(
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    eta: &[T],
    prob: &[T],
    w: &[T],
    m_buf: &[T],
    cluster_ids: &[u32],
    xr_all: &[f64],
    q: usize,
    s: usize,
    p_rt: usize,
    n: usize,
    a_blocks: &mut [T],
    a_rhs: &mut [T],
    bt: &mut [f64],
) {
    let p = if P == DYN { p_rt } else { P };
    let k = q * s;
    a_blocks[..s * q * q].fill(T::ZERO);
    a_rhs[..k].fill(T::ZERO);
    bt[..k * p].fill(0.0);
    let mut probit_rows = crate::family::ProbitRows::new(family, eta, prob);
    for i in 0..n {
        let xr = &xr_all[i * p..i * p + p];
        let f = cluster_ids[i] as usize;
        let ubase = f * q;
        let wi = w[i];
        let resid = super::row_rho(
            family,
            nb_theta,
            i,
            y[i],
            prior_w[i],
            eta[i],
            prob[i],
            probit_rows.as_mut(),
        );
        let wf = wi.value();
        if q == 1 {
            let m0 = m_buf[i];
            a_rhs[f] += m0 * resid;
            a_blocks[f] += wi * m0 * m0;
            let m0f = m0.value();
            let bt_f = &mut bt[f * p..f * p + p];
            for r in 0..p {
                bt_f[r] += xr[r] * wf * m0f;
            }
        } else {
            let m_row = &m_buf[i * q..i * q + q];
            let ablk = f * q * q;
            for r in 0..q {
                a_rhs[ubase + r] += m_row[r] * resid;
                let wr = wi * m_row[r];
                for c in 0..=r {
                    a_blocks[ablk + r * q + c] += wr * m_row[c];
                }
            }
            let bt_f = &mut bt[ubase * p..(ubase + q) * p];
            for r in 0..p {
                let xw = xr[r] * wf;
                for c in 0..q {
                    bt_f[c * p + r] += xw * m_row[c].value();
                }
            }
        }
    }
}

/// `glmm_block_solve` on the `q×p` row-major right-hand side `b`
/// (`b[c·p + col]`), all `p` columns at once: per column the same
/// subtractions and divisions in the same order, so the same bits.
#[inline(always)]
fn block_solve_cols_p<T: Scalar, const P: usize>(l: &[T], q: usize, p_rt: usize, b: &mut [f64]) {
    let p = if P == DYN { p_rt } else { P };
    for r in 0..q {
        let (done, rest) = b.split_at_mut(r * p);
        let row = &mut rest[..p];
        for c in 0..r {
            let lrc = l[r * q + c].value();
            let bc = &done[c * p..c * p + p];
            for col in 0..p {
                row[col] -= lrc * bc[col];
            }
        }
        let d = l[r * q + r].value();
        for v in row.iter_mut() {
            *v /= d;
        }
    }
    for r in (0..q).rev() {
        let (head, tail) = b.split_at_mut((r + 1) * p);
        let row = &mut head[r * p..r * p + p];
        for c in (r + 1)..q {
            let lcr = l[c * q + r].value();
            let bc = &tail[(c - r - 1) * p..(c - r) * p];
            for col in 0..p {
                row[col] -= lcr * bc[col];
            }
        }
        let d = l[r * q + r].value();
        for v in row.iter_mut() {
            *v /= d;
        }
    }
}

/// Residual rows `rᵢ = xᵢ − T_fᵀmᵢ` of rows `i0..i0 + len` into the
/// column-major tile `out` (column `col` at `col·BORDER_TILE`), off the
/// row-major `X` copy and `T` row-major in `tr` (`[(f·q + c)·p + col]`). Each
/// entry subtracts the `q` terms in the same order as the `MatRef` loop.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn residual_tile_p<T: Scalar, const P: usize>(
    xr_all: &[f64],
    tr: &[f64],
    m_buf: &[T],
    cluster_ids: &[u32],
    q: usize,
    p_rt: usize,
    i0: usize,
    len: usize,
    out: &mut [f64; BORDER_PMAX * BORDER_TILE],
) {
    let p = if P == DYN { p_rt } else { P };
    for ii in 0..len {
        let i = i0 + ii;
        let f = cluster_ids[i] as usize;
        let mut r = [0.0f64; BORDER_PMAX];
        let r = &mut r[..p];
        r.copy_from_slice(&xr_all[i * p..i * p + p]);
        let tf = &tr[f * q * p..(f + 1) * q * p];
        let m_row = &m_buf[i * q..i * q + q];
        for c in 0..q {
            let mc = m_row[c].value();
            let t_row = &tf[c * p..c * p + p];
            for col in 0..p {
                r[col] -= mc * t_row[col];
            }
        }
        for col in 0..p {
            out[col * BORDER_TILE + ii] = r[col];
        }
    }
}

/// Pass A of the exact block (see the call site): per row, the leverage
/// `hᵢ = mᵢ'A⁻¹mᵢ` off the iterate's factor, `aᵢ = w'ᵢ·hᵢ` into `c_β`
/// (`logdet_beta`) and `g_u` (`logdet_u`), `hᵢ` into `curv_h`. Generic over the
/// row's `dw/dη` so the caller can hoist the family dispatch out of the loop.
struct ExactPassA<'b, T> {
    fac: &'b [f64],
    m_buf: &'b [T],
    cluster_ids: &'b [u32],
    x: MatRef<'b, f64>,
    xr_all: &'b [f64],
    q: usize,
    p: usize,
    n: usize,
}

impl<T: Scalar> ExactPassA<'_, T> {
    #[inline(always)]
    fn run(
        &self,
        curv_h: &mut [f64],
        logdet_beta: &mut [f64],
        logdet_u: &mut [f64],
        wp_row: impl Fn(usize) -> f64,
    ) {
        if self.xr_all.is_empty() {
            self.run_p::<DYN>(curv_h, logdet_beta, logdet_u, wp_row)
        } else {
            with_const_p!(self.p, P => self.run_p::<P>(curv_h, logdet_beta, logdet_u, &wp_row))
        }
    }

    /// The row loop at `const P` (`DYN`: the runtime `p`, and the `MatRef`
    /// when there is no row-major copy). `c_β` accumulates in registers on a
    /// fixed `P`, in the same row order; `q == 1` reads the leverage as
    /// `(m/L₀₀)²`, `block_leverage`'s own arithmetic at `q = 1`.
    #[allow(clippy::needless_range_loop)]
    #[inline(always)]
    fn run_p<const P: usize>(
        &self,
        curv_h: &mut [f64],
        logdet_beta: &mut [f64],
        logdet_u: &mut [f64],
        wp_row: impl Fn(usize) -> f64,
    ) {
        let (q, p) = (self.q, if P == DYN { self.p } else { P });
        let rows = !self.xr_all.is_empty();
        let mut lb = [0.0_f64; P_CAP];
        if P != DYN {
            lb[..P].copy_from_slice(&logdet_beta[..P]);
        }
        let mut mrow = [0.0_f64; crate::consts::MAX_PRIMARY_Q];
        for i in 0..self.n {
            let f = self.cluster_ids[i] as usize;
            let h = if q == 1 {
                mrow[0] = self.m_buf[i].value();
                let t = mrow[0] / self.fac[f];
                t * t
            } else {
                let ablk = f * q * q;
                for c in 0..q {
                    mrow[c] = self.m_buf[i * q + c].value();
                }
                super::block_leverage(&self.fac[ablk..ablk + q * q], q, &mrow[..q])
            };
            let wp = wp_row(i);
            curv_h[i] = h;
            let a = wp * h;
            if P != DYN {
                let xr = &self.xr_all[i * P..i * P + P];
                for j in 0..P {
                    lb[j] += a * xr[j];
                }
            } else if rows {
                let xr = &self.xr_all[i * p..i * p + p];
                for (l, &xv) in logdet_beta[..p].iter_mut().zip(xr) {
                    *l += a * xv;
                }
            } else {
                for j in 0..p {
                    logdet_beta[j] += a * self.x[(i, j)];
                }
            }
            if q == 1 {
                logdet_u[f] += a * mrow[0];
            } else {
                for c in 0..q {
                    logdet_u[f * q + c] += a * mrow[c];
                }
            }
        }
        if P != DYN {
            logdet_beta[..P].copy_from_slice(&lb[..P]);
        }
    }
}

/// Pass C of the exact block: `c_β −= Σᵢ w̃ᵢ·(mᵢ·v_f)·xᵢ` off the row-major
/// `X`, `mᵢ·v_f` into `curv_sdot`; `c_β` accumulated in registers at a fixed
/// `P`, in row order.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn exact_pass_c_rows<T: Scalar, const P: usize>(
    m_buf: &[T],
    w: &[T],
    v: &[f64],
    cluster_ids: &[u32],
    xr_all: &[f64],
    q: usize,
    p_rt: usize,
    n: usize,
    curv_sdot: &mut [f64],
    logdet_beta: &mut [f64],
) {
    let p = if P == DYN { p_rt } else { P };
    let mut lb = [0.0_f64; P_CAP];
    if P != DYN {
        lb[..P].copy_from_slice(&logdet_beta[..P]);
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        // `q == 1` keeps the q-loop's `0.0 + m·v` (it differs from `m·v`
        // only in the sign of a zero), so the sums stay the loop's.
        let sdot = if q == 1 {
            let mut sdot = 0.0;
            sdot += m_buf[i].value() * v[f];
            sdot
        } else {
            let mut sdot = 0.0;
            for c in 0..q {
                sdot += m_buf[i * q + c].value() * v[f * q + c];
            }
            sdot
        };
        curv_sdot[i] = sdot;
        let a = w[i].value() * sdot;
        let xr = &xr_all[i * p..i * p + p];
        if P != DYN {
            for j in 0..P {
                lb[j] -= a * xr[j];
            }
        } else {
            for (l, &xv) in logdet_beta[..p].iter_mut().zip(xr) {
                *l -= a * xv;
            }
        }
    }
    if P != DYN {
        logdet_beta[..P].copy_from_slice(&lb[..P]);
    }
}

/// `refresh_eta_fixed` off the row-major `X` copy at `const P`: the same
/// `Σ_j x·β` per row, in the same `j` order, from zero, offset added after.
#[inline(always)]
fn refresh_rows_p<T: Scalar, const P: usize>(
    xr: &[f64],
    p_rt: usize,
    beta: &[T],
    eta_fixed: &mut [T],
    n: usize,
) {
    let p = if P == DYN { p_rt } else { P };
    let beta = &beta[..p];
    for (e_out, row) in eta_fixed[..n].iter_mut().zip(xr.chunks_exact(p)) {
        let mut e = T::ZERO;
        for j in 0..p {
            e += beta[j].scale(row[j]);
        }
        *e_out = e;
    }
}

/// The structured kernel's border rows: `B = M'WX` (`k×p` row-major, RE column
/// `c` at `bt[c·p..]`: a row's core columns `core_col(f, local)` and crossed
/// columns `k_family + b`), with the same terms in the same row order as the
/// kernel's own `B'` scatter.
#[allow(clippy::too_many_arguments)]
pub(super) fn structured_border_rows<T: Scalar>(
    w: &[T],
    m_core_buf: &[T],
    cross_theta: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    cluster_ids: &[u32],
    xr_all: &[f64],
    q: usize,
    np: usize,
    s: usize,
    e: usize,
    p: usize,
    n: usize,
    bt: &mut [f64],
) {
    let k = (q + np) * s + e;
    bt[..k * p].fill(0.0);
    with_const_p!(p, P => structured_border_rows_p::<T, P>(
        w, m_core_buf, cross_theta, cross_col, n_cross, cluster_ids, xr_all, q, np, s, p, n, bt,
    ))
}

/// [`structured_border_rows`]'s row loop at `const P` (`DYN`: the runtime `p`).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn structured_border_rows_p<T: Scalar, const P: usize>(
    w: &[T],
    m_core_buf: &[T],
    cross_theta: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    cluster_ids: &[u32],
    xr_all: &[f64],
    q: usize,
    np: usize,
    s: usize,
    p_rt: usize,
    n: usize,
    bt: &mut [f64],
) {
    let p = if P == DYN { p_rt } else { P };
    let qc = q + np;
    let prim_width = q * s;
    let k_family = qc * s;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let wi = w[i].value();
        let xr = &xr_all[i * p..i * p + p];
        let cbase = i * g_cap;
        let ncz = n_cross[i] as usize;
        for local in 0..qc {
            let col = if local < q {
                f * q + local
            } else {
                prim_width + f * np + (local - q)
            };
            let mv = m_core_buf[i * qc + local].value();
            for (b, &xv) in bt[col * p..col * p + p].iter_mut().zip(xr) {
                *b += xv * wi * mv;
            }
        }
        for z in 0..ncz {
            let col = k_family + cross_col[cbase + z] as usize;
            let cv = cross_theta[z].value();
            for (b, &xv) in bt[col * p..col * p + p].iter_mut().zip(xr) {
                *b += xv * wi * cv;
            }
        }
    }
}
