//! Block-diagonal PIRLS solve for the no-extras regime (`pirls_solve_blocked`).

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
            yeta += T::from_f64(y[i]) * eta[i];
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
        // --- scatter-pass (scalar): wᵢmᵢmᵢ' and rᵢ·mᵢ into the blocks.
        // The effective residual rᵢ is logit's (yᵢ−pᵢ) or the general W·working_resid. ---
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let f = cluster_ids[i] as usize;
            let ubase = f * q;
            let ablk = f * q * q;
            let wi = w[i];
            let resid = T::from_f64(prior_w[i])
                * match family {
                    Family::Binomial {
                        link: BinomialLink::Logit,
                    } => T::from_f64(y[i]) - prob[i],
                    other => crate::family::row_score(other, nb_theta, y[i], 1.0, eta[i], prob[i]),
                };
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
            for r in 0..q {
                a_blocks[ablk + r * q + r] += T::ONE;
            }
            if !glmm_block_chol(&mut a_blocks[ablk..ablk + q * q], q) {
                return None;
            }
            for r in 0..q {
                logdet += a_blocks[ablk + r * q + r].ln();
            }
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
/// Step-halving: see `pirls_solve_packed`'s doc — identical mechanism, `a_blocks` here
/// plays the role of `pirls_solve_packed`'s `A`. A converged solve ends by
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
    // n × p = W∘X GEMM scratch for the Profile β-Schur border's C = X'WX
    // (mirrors `pirls_solve_packed`'s `wx`; this variant has no dense M so there is no
    // `wm` twin here — B' = X'WM is filled by cluster-scatter instead).
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`FitOptions::offset`), added into
    // `eta_fixed` by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
    // Observation-only — mirrors `pirls_solve_packed`'s `counters`, where
    // the contract is stated.
    counters: &mut crate::counters::EvalCounters,
) -> (T, T, T, bool) {
    // The β-Schur border below runs in f64 (its X'WX GEMM and p×p Cholesky are
    // faer's, and reproducing them generically would move f64 bits). A non-f64
    // instantiation must use BetaStep::Fixed — the derivative entry points do;
    // exact generic β-profiling comes later. Loud, not silent: dropping
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
        ..
    } = scratch;
    let q = g.primary_q;
    let s = g.n_primary;
    let k = q * s;
    let p = beta.len();
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
    // M = ZΛ_p (mᵢ = Λ_p'·zᵢ, zᵢ = [1, x[i, slope_cols]] pre-widened into z_buf
    // per fit) is invariant within one solve — Λ and x are fixed; the iteration
    // only mutates u/η/prob/w/blocks. Fill it once per solve. Measured on the
    // glmm_slope profile (2026-06): the former per-row recompute closure was
    // ~57% of fit runtime, >90% of that MatRef indexing / bounds checks / 64-byte
    // return-by-value copies rather than FMA — buffering removes the overhead,
    // not the math. Bit-identical: the same `Σ_{r≥c} z_r·lam[r·q+c]` reduction
    // runs once per (i,c) in the same inner order, and both consumers below read
    // the same f64 values in the same order.
    for i in 0..n {
        for c in 0..q {
            let mut acc = T::ZERO;
            for r in c..q {
                let zr = if r == 0 {
                    T::ONE
                } else {
                    T::from_f64(z_buf[i * (q - 1) + (r - 1)])
                };
                acc += zr * lam[r * q + c];
            }
            m_buf[i * q + c] = acc;
        }
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
    let mut pen_accepted = f64::INFINITY; // same-point penalized deviance at the last ACCEPTED iterate
    let mut mixed_prev = f64::INFINITY; // the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` from the previous step
    let mut halvings = 0usize;
    let mut converged = false;
    // Period-2 damping state — see `PIRLS_OSC_RATIO`.
    let mut dmix_prev = f64::NAN;
    let mut osc_flips = 0usize;
    let mut damp = false;
    let min_iters = match dual.as_deref_mut() {
        Some(d) => {
            // Optimistic seed: paired with `observed = !canonical` the step
            // this path takes IS the Hessian step, so one call suffices unless
            // the non-PD observed factor further down takes it back. Mirrors
            // `pirls_solve_blocked_extras`'s `exact` contract — change
            // together; the reasoning is written out there.
            d.exact = true;
            d.min_iters
        }
        None => 0,
    };
    let (mut dev, mut pen, mut logdet) = (T::from_f64(f64::NAN), T::from_f64(f64::NAN), T::ZERO);
    let tol = pirls_tol_override.unwrap_or_else(|| super::super::pirls_tol(family));
    // Exact-Laplace β-profile: the accept/halve decision moves from the
    // penalized-deviance band (below) to the FULL Laplace merit `dev + pen_u +
    // 2·log|A|` after the block sweep, since log|A| depends on this iteration's
    // A and is not known until the factor is formed. `l_acc` is that merit at
    // the last ACCEPTED (u, β) — cold start accepts unconditionally.
    let exact = matches!(beta_step, BetaStep::Profile { exact: Some(_), .. });
    let mut l_acc = f64::INFINITY;
    // `|g_u'·δu₀|` at the point `l_acc` was measured at — how far that stored
    // merit may itself be off. One half of the accept band below; see it for why
    // both endpoints of the comparison are charged. Unread while `l_acc = ∞`
    // (the first trial accepts unconditionally).
    let mut l_acc_slack = 0.0_f64;
    // Careful-mode state (see the exact merit test below). `acc_judged` says
    // whether the stored `l_acc` was itself accepted by a comparison (the
    // first trial is accepted unjudged); `refine` marks an iteration that
    // re-steps u alone at the trial's β; `l_ref` is the merit at the previous
    // iterate of that β and `pen_ref` its `dev + ‖u‖²`, the merit that u-only
    // step backtracks on. `trust` is the border step's trust region.
    let mut careful = false;
    let mut acc_judged = false;
    let mut refine = false;
    let mut l_ref = f64::INFINITY;
    let mut pen_ref = f64::INFINITY;
    let mut ref_halvings = 0usize;
    let mut trust = BorderTrust::new();
    'iters: for it in 0..PIRLS_MAX_ITERS {
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
        // Retrospective step-halving (lme4 `pwrssUpdate`, mirrors `pirls_solve_packed`):
        // convergence band checked BEFORE the overshoot test (near the optimum
        // the iteration is not strictly monotone — a step can land ε above
        // `pen_accepted` yet inside the tol band, and that must converge, not burn
        // all 10 halvings against FP noise). ‖u‖² is at the CURRENT trial u.
        let mut pen_u = T::ZERO;
        #[allow(clippy::needless_range_loop)]
        for c in 0..k {
            pen_u += u[c] * u[c];
        }
        let penalized = dev + pen_u;
        // A careful-mode u-only step (β fixed) is judged the way lme4's
        // `pwrssUpdate` judges a u-only PIRLS step: on `dev + ‖u‖²` against its
        // origin, halving toward that origin with β untouched, never toward the
        // accepted (u, β) the β line search below backtracks to.
        if refine
            && (infeasible
                || !penalized.value().is_finite()
                || penalized.value() - pen_ref > tol * (1.0 + penalized.value().abs()))
        {
            if ref_halvings < PIRLS_MAX_HALVINGS {
                ref_halvings += 1;
                for c in 0..k {
                    u[c] = T::from_f64(0.5) * (u[c] + u_prev[c]);
                }
                continue;
            }
            return (
                T::from_f64(f64::NAN),
                T::from_f64(f64::NAN),
                T::from_f64(f64::NAN),
                false,
            );
        }
        // BAND-TOLERANT overshoot test, mirrors `pirls_solve_packed` (see its comments
        // for why a within-band rise is accepted rather than converged-on or
        // halved): only a rise EXCEEDING the tol band backtracks. A
        // domain-infeasible trial halves regardless of the band (see
        // `pirls_solve_packed`'s comment).
        // Exact mode: the retrospective band above is on `dev + pen_u` alone,
        // which is not the profiled objective (missing 2·log|A|) — only a
        // domain-infeasible trial halves here; an overshoot on `dev + pen_u`
        // is judged later, after log|A| is known, against the FULL merit.
        if infeasible
            || !penalized.value().is_finite()
            || (!exact && penalized.value() - pen_accepted > tol * (1.0 + penalized.value().abs()))
        {
            trust.fail(acc_judged);
            if halvings < PIRLS_MAX_HALVINGS {
                // Last full step overshot: halve δu toward the halving target
                // and re-enter the top (the recompute above is the trial
                // evaluation of the halved step). Exact mode halves toward the
                // last ACCEPTED u (`u_acc`, set by the post-sweep merit test
                // below) since `u_prev` there holds the just-rejected trial,
                // not an accepted point.
                halvings += 1;
                if let BetaStep::Profile {
                    exact: Some(ex), ..
                } = &beta_step
                {
                    #[allow(clippy::needless_range_loop)]
                    for c in 0..k {
                        u[c] = T::from_f64(0.5 * (u[c].value() + ex.u_acc[c]));
                    }
                } else {
                    for c in 0..k {
                        u[c] = T::from_f64(0.5) * (u[c] + u_prev[c]);
                    }
                }
                // Profile mode: the trial point is the JOINT (u,β) step, so the
                // backtrack halves β toward `beta_prev` in lockstep with u, then
                // refreshes η_fixed for the re-evaluation at the top. Mirrors
                // `pirls_solve_packed`'s Profile backtrack.
                if let BetaStep::Profile { beta_prev, .. } = &beta_step {
                    for j in 0..p {
                        beta[j] = T::from_f64(0.5 * (beta[j].value() + beta_prev[j]));
                    }
                    refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
                }
                continue;
            }
            return (
                T::from_f64(f64::NAN),
                T::from_f64(f64::NAN),
                T::from_f64(f64::NAN),
                false,
            ); // halvings exhausted
        }
        // Accept this iterate, snapshot it for the next backtrack, and take a
        // fresh full step from it (cold start: pen_accepted = ∞ ⇒ always
        // accepts). `u_prev` is unconditional — the β-Schur border needs
        // δu₀ = u_new − u_prev regardless of mode. Exact mode defers
        // `halvings`/`pen_accepted`/`beta_prev` bookkeeping to the post-sweep
        // merit test below, since this trial has not been judged against the
        // full Laplace objective yet.
        u_prev[..k].copy_from_slice(&u[..k]);
        if !exact {
            halvings = 0;
            pen_accepted = penalized.value();
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
        // pass. ---
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
        // Profile mode: accumulate the β-gradient X'ρ (ρ = effective residual) into
        // `beta_rhs` — the joint system's bottom-block RHS. A dedicated pass off the
        // fresh prob/eta/w (NOT folded into the scatter loop above), so the Fixed
        // path stays byte-identical. Mirrors `pirls_solve_packed`'s Profile X'ρ fold.
        if let BetaStep::Profile { beta_rhs, .. } = &mut beta_step {
            for v in beta_rhs[..p].iter_mut() {
                *v = 0.0;
            }
            match family {
                Family::Binomial {
                    link: BinomialLink::Logit,
                } => {
                    for i in 0..n {
                        let rho = prior_w[i] * (y[i] - prob[i].value());
                        for j in 0..p {
                            beta_rhs[j] += x[(i, j)] * rho;
                        }
                    }
                }
                other => {
                    for i in 0..n {
                        let rho = crate::family::row_score(
                            other,
                            nb_theta,
                            y[i],
                            prior_w[i],
                            eta[i].value(),
                            prob[i].value(),
                        );
                        for j in 0..p {
                            beta_rhs[j] += x[(i, j)] * rho;
                        }
                    }
                }
            }
        }
        // --- per block: rhs_f = (A_f−I)·u_old_f + g_f ; +I ; factor ; solve u_new.
        // `log|A|` comes off the factor that produces this step's u; the exit
        // refresh below replaces it and a_blocks with their values at the
        // returned u (a step may be re-taken after a halving, so every
        // iteration re-factors).
        // Three phases, in this order because of what each one reads. The fold
        // below needs the UNFACTORED blocks and the pre-step `u`, and the
        // observed twin needs `a_rhs`'s bare `g_f` before the fold overwrites
        // it — so the whole fold runs first. Then the factor (`+I`, Crout,
        // `log|A|` off the pivots) and the `A⁻¹` solve, both over the whole
        // block-diagonal `A`. Then, per cluster, the main solution into `u`,
        // the observed twin's own factor and solve over it, and `‖u‖²`.
        for f in 0..s {
            let ablk = f * q * q;
            let ubase = f * q;
            // Observed right-hand side: rhs_obs = (A_obs,f − I)·u_old_f + g_f,
            // staged in `obs_rhs` (the `a_rhs` packing) for the twin factor and
            // solve below. The main block is formed and factored on every
            // route — it is what `log|A|` and the returned factor come from.
            if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
                let DualStep {
                    obs_blocks,
                    obs_rhs,
                    ..
                } = &mut *d;
                let ob = &obs_blocks[ablk..ablk + q * q];
                for r in 0..q {
                    let mut acc = layout.a_rhs[ubase + r];
                    for c in 0..q {
                        let (hi, lo) = if r >= c { (r, c) } else { (c, r) };
                        acc += ob[hi * q + lo] * u[ubase + c];
                    }
                    obs_rhs[ubase + r] = acc;
                }
            }
            // (A_f − I)·u_old_f added to g_f (in a_rhs), using the still-unfactored
            // symmetric lower triangle.
            for r in 0..q {
                let mut acc = layout.a_rhs[ubase + r];
                for c in 0..q {
                    let (hi, lo) = if r >= c { (r, c) } else { (c, r) };
                    acc += layout.a_blocks[ablk + hi * q + lo] * u[ubase + c];
                }
                layout.a_rhs[ubase + r] = acc;
            }
        }
        logdet = match layout.factor_logdet() {
            Some(ld) => ld,
            None => {
                // Mirrors the `structured_factor` failure arm in
                // `pirls_solve_blocked_extras` (`blocked_extras.rs`, the extras
                // twin) — see its comment for why: an unevaluable trial here is
                // a REJECTED trial, not a failed solve. Change together.
                if let BetaStep::Profile {
                    exact: Some(ex),
                    beta_prev,
                    ..
                } = &beta_step
                {
                    trust.fail(acc_judged);
                    if halvings < PIRLS_MAX_HALVINGS {
                        halvings += 1;
                        refine = false;
                        l_ref = f64::INFINITY;
                        for c in 0..k {
                            u[c] = T::from_f64(0.5 * (u_prev[c].value() + ex.u_acc[c]));
                        }
                        for j in 0..p {
                            beta[j] = T::from_f64(0.5 * (beta[j].value() + beta_prev[j]));
                        }
                        refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
                        continue 'iters;
                    }
                }
                return (
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    false,
                );
            }
        };
        layout.solve_in_place();
        pen = T::ZERO;
        for f in 0..s {
            let ablk = f * q * q;
            let ubase = f * q;
            // u_new_f = A_f⁻¹ rhs_f, solved above in `a_rhs`.
            u[ubase..ubase + q].copy_from_slice(&layout.a_rhs[ubase..ubase + q]);
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
                    u[ubase..ubase + q].copy_from_slice(&obs_rhs[ubase..ubase + q]);
                } else {
                    *exact = false;
                }
            }
            for r in 0..q {
                pen += u[ubase + r] * u[ubase + r];
            }
        }
        // Where the exact curvature differs from Fisher, the objective's `log|A|`
        // is `log|A_obs|` (`evaluate_at_mode`), and the Newton step above already
        // built `a_blocks` from `W_obs`: the factor just taken is `A_obs`'s. The
        // exact profile is `f64`-only (`dual` is `None`), so this holds on every
        // exact-mode solve.
        let exact_obj = exact && crate::family::exact_curvature_differs(family);
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
                let mut mrow = [0.0_f64; crate::consts::MAX_PRIMARY_Q];
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    let ablk = f * q * q;
                    for c in 0..q {
                        mrow[c] = m_buf[i * q + c].value();
                    }
                    // The `dw/dη` of the curvature `log|A|` is built from:
                    // `dW_obs/dη` where the objective is `log|A_obs|`
                    // (`exact_obj`), the Fisher one otherwise.
                    let h = block_leverage(&fac_f64[ablk..ablk + q * q], q, &mrow[..q]);
                    // `family::weight_eta_deriv` is the closed form of this same
                    // `dw/dη`, held equal to this `Dual<1>` line by
                    // `weight_eta_deriv_matches_dual1_of_irls_weight`
                    // (`src/family.rs`); changing either alone moves `f64` bits.
                    let wp = if exact_obj {
                        crate::family::observed_weight_eta_deriv(
                            family,
                            nb_theta,
                            y[i],
                            prior_w[i],
                            eta[i].value(),
                            prob[i].value(),
                            0.0,
                        )
                    } else {
                        let e = crate::dual::Dual::<1> {
                            v: eta[i].value(),
                            d: [1.0],
                        };
                        let (_, w_raw, _) =
                            crate::family::irls_weight_and_resid(family, nb_theta, y[i], e);
                        prior_w[i] * w_raw.d[0]
                    };
                    let a = wp * h;
                    for j in 0..p {
                        logdet_beta[j] += a * x[(i, j)];
                    }
                    for c in 0..q {
                        logdet_u[f * q + c] += a * mrow[c];
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
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    let mut sdot = 0.0;
                    for c in 0..q {
                        sdot += m_buf[i * q + c].value() * logdet_u[f * q + c];
                    }
                    let a = w[i].value() * sdot;
                    for j in 0..p {
                        logdet_beta[j] -= a * x[(i, j)];
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
            // Being first-order, that correction leaves its own O(‖δu₀‖²)
            // residual behind, either sign, so a merit is only good to about the
            // size of the correction it carries. The comparison has TWO
            // endpoints and each carries its own — hence both are charged:
            // `gu_dot_du.abs()` for this trial, `l_acc_slack` for the stored
            // `l_acc`. Charging the trial alone would only accidentally cover
            // the failure, since the deficit that deadlocks the loop belongs to
            // the ACCEPTED point: a warm (u, β) is accepted unjudged as the first
            // trial, and if its merit sits under the value the solve can reach,
            // every later (correct) iterate is rejected until the iteration cap
            // turns the whole evaluation into a non-convergence. Measured on the
            // single-intercept NB-log shape: a warm start carried from θ = 0.55674
            // to θ = 0.55692 left `l_acc` 3.2e-5 under the value the cold solve
            // reaches, every iteration went to halving, and the θ-only outer
            // search read the `inf` that came back as a wall — with the seed's own
            // 1.4e-4 correction charged, that deficit is inside the band. Both
            // terms shrink with their δu₀, so at the mode the test is the value
            // band again, and the convergence test below is untouched.
            // A non-finite merit (an overflowed step) is an overshoot: every
            // comparison with NaN is false, so without this test it would be
            // accepted and poison `l_acc` for the rest of the solve.
            // Careful mode. The slack above trusts `|g_u'·δu₀|` to size a merit's
            // error, which holds only while δu₀ is small. Far from the mode it is
            // not: at a large θ with few clusters the Laplace profile is nearly
            // flat in the β direction the random effects nearly span, the border
            // step can move β by several units, and u lands a long Newton step off
            // its mode. The slack then grows with that step and lets real rises
            // through, while a first-order
            // merit can also sit far below the value its β reaches once u has
            // converged (measured on a 4-cluster NB-log fit at θ = 50: 2.0 under
            // it, with a correction of 0.07), so a stored `l_acc` can refuse every
            // later trial. So the first time the slack alone lets through a rise
            // over a stored merit that was itself judged (a rise over the unjudged
            // first trial is the warm-start case the slack was written for) and the
            // trial's own correction is past the band, the solve stops comparing
            // first-order merits for good: it drops `l_acc`, and judges each later
            // trial only once u-only steps at that trial's β (lme4's `pwrssUpdate`
            // with `uOnly`, backtracking on `dev + ‖u‖²` above) have made its merit
            // stationary, against a stored merit that was converged the same way,
            // on the plain tol band. lme4's `pwrssUpdate` and MixedModels.jl's
            // `pirls!` accept a step only on a real decrease of the objective,
            // which they can evaluate exactly at every iterate; here it is exact
            // only where u sits at its mode, and the u-only steps are what buy
            // that.
            let band = tol * (1.0 + l_trial.abs());
            if careful && (l_trial - l_ref).abs() > band {
                refine = true;
                l_ref = l_trial;
                pen_ref = penalized.value();
                ref_halvings = 0;
                continue;
            }
            let allow = if careful {
                band
            } else {
                band + gu_dot_du.abs() + l_acc_slack
            };
            let reject = !l_trial.is_finite() || l_trial - l_acc > allow;
            if !careful && acc_judged && !reject && l_trial - l_acc > band && gu_dot_du.abs() > band
            {
                careful = true;
                l_acc = f64::INFINITY;
                refine = true;
                l_ref = l_trial;
                pen_ref = penalized.value();
                ref_halvings = 0;
                continue;
            }
            refine = false;
            l_ref = f64::INFINITY;
            // The trust region reads the merit change the border step's trial
            // made against its model (`BorderTrust`), under the same allowance
            // for the merits' own error as the accept test.
            trust.judge(l_acc, acc_judged, l_trial, allow);
            if reject {
                if halvings < PIRLS_MAX_HALVINGS {
                    halvings += 1;
                    for c in 0..k {
                        u[c] = T::from_f64(0.5 * (u_prev[c].value() + ex.u_acc[c]));
                    }
                    for j in 0..p {
                        beta[j] = T::from_f64(0.5 * (beta[j].value() + beta_prev[j]));
                    }
                    refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
                    continue;
                }
                return (
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    false,
                );
            }
            halvings = 0;
            acc_judged = l_acc.is_finite();
            l_acc = l_trial;
            l_acc_slack = gu_dot_du.abs();
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
        // this mirrors). Exact mode steps with S_β + ½·d²log|A|/dβ² on
        // X'ρ − B'δu₀ − ½c_β (`logdet_beta_curvature`, `border_solve`). `m_buf`
        // already holds mᵢ = Λ'zᵢ, so B's scatter reads it directly (cheaper than
        // se.rs's per-row reconstruction). ---
        if let BetaStep::Profile {
            exact,
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            schur_llt_mem,
            ..
        } = &mut beta_step
        {
            // C = X'WX (p×p) via the W∘X GEMM scratch `wx` — blocked_schur_fill's
            // xtwx block, same product. GEMM fills the full p×p (the old scalar
            // loop only filled+mirrored the lower triangle); every downstream read
            // below is over the full matrix (`schur[(r,c)]` for `c in 0..p`), so
            // this is exact.
            for c in 0..p {
                for i in 0..n {
                    wx[(i, c)] = w[i].value() * x[(i, c)];
                }
            }
            faer::linalg::matmul::matmul(
                xtwx.as_mut(),
                faer::Accum::Replace,
                x.subrows(0, n).transpose(),
                wx.as_ref().subrows(0, n),
                1.0,
                Par::Seq,
            );
            // B' = X'WM (p×k), blocked: zero, then scatter the q_p coupling columns
            // per row into cluster f's column band. Uses the live `m_buf[i·q+c] = mᵢ`.
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
            // T_f = A_f⁻¹ (M'WX)_f per block, reusing this iteration's factor left in
            // `a_blocks`; ainv_mtwx rows f·q.. hold T_f. (M'WX)_f[c, col] = xtwm[(col, f·q+c)].
            // Mirrors blocked_schur_fill:360-374.
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
            // S_β = C − B'·T (blocked_schur_fill:378-386). A block-diagonal, so the
            // per-block solves equal the full A⁻¹M'WX and the Σ over k is exact.
            for r in 0..p {
                for c in 0..p {
                    let mut sm = xtwx[(r, c)];
                    for j in 0..k {
                        sm -= xtwm[(r, j)] * ainv_mtwx[(j, c)];
                    }
                    schur[(r, c)] = sm;
                }
            }
            // rhs = X'ρ − B'·δu₀ (beta_rhs holds X'ρ; δu₀ = u_new − u_prev).
            for r in 0..p {
                let mut acc = 0.0;
                for c in 0..k {
                    acc += xtwm[(r, c)] * (u[c] - u_prev[c]).value();
                }
                beta_rhs[r] -= acc;
            }
            // Exact mode: the step is Newton on the Laplace profile, so it also
            // carries log|A|'s gradient `c_β` and curvature in β
            // (`logdet_beta_curvature`), with the same ½ as the objective's
            // `2·logdet` term.
            if let Some(ex) = exact.as_deref_mut() {
                for r in 0..p {
                    beta_rhs[r] -= 0.5 * ex.logdet_beta[r];
                }
                logdet_beta_curvature(
                    ex,
                    family,
                    nb_theta,
                    exact_obj,
                    y,
                    prior_w,
                    &eta[..n],
                    x,
                    ainv_mtwx.as_ref(),
                    &m_buf[..],
                    cluster_ids,
                    g,
                    None,
                    n,
                    p,
                );
            }
            // δβ in place, inside the trust region on the exact border. Non-PD
            // S_β ⇒ the (NaN,…,false) failure surface.
            if !border_solve(
                schur,
                &mut beta_rhs[..p],
                schur_llt_mem,
                exact.as_deref_mut().map(|ex| (ex, &mut trust)),
            ) {
                return (
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    false,
                );
            }
            // Apply: β += δβ; u = u_joint = u_new − T·δβ, i.e.
            // u[f·q+c] −= Σ_j T[(f·q+c, j)]·δβ[j].
            for j in 0..p {
                beta[j] += T::from_f64(beta_rhs[j]);
            }
            for c in 0..k {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += ainv_mtwx[(c, j)] * beta_rhs[j];
                }
                u[c] -= T::from_f64(acc);
            }
            // η_fixed depends on β; refresh for the next trial. `pen` must track the
            // moved u (‖u_joint‖²), so recompute it.
            refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
            pen = T::ZERO;
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                pen += u[c] * u[c];
            }
        }
        // Relaxed step once the period-2 detector has fired (`PIRLS_OSC_RATIO`):
        // move half way from the pre-step iterate, β in lockstep with u.
        if damp {
            trust.scale(0.5);
            let half = T::from_f64(0.5);
            for c in 0..k {
                u[c] = u_prev[c] + half * (u[c] - u_prev[c]);
            }
            if let BetaStep::Profile { beta_prev, .. } = &beta_step {
                for j in 0..p {
                    beta[j] = T::from_f64(beta_prev[j] + 0.5 * (beta[j].value() - beta_prev[j]));
                }
                refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
            }
            pen = T::ZERO;
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                pen += u[c] * u[c];
            }
        }
        // The stopping rule, verbatim: the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` band on
        // successive steps — bit-identical iterate path and returned values to the
        // pre-halving loop when no halving fires. The same-point band cannot be a
        // converge trigger: on a solve that cycles it stops at one point of the
        // cycle rather than at the mode, and it broke the AGQ(k=1) ≡ Laplace
        // reduction when it was one.
        // Exact mode: the exit band must track the same merit the accept/halve
        // decision above uses (dev + pen + 2·log|A|), or the loop could settle
        // on a point that is a fixed point of dev+pen alone but still moving in
        // log|A| — the whole reason the merit moved off the PQL band. The merit's
        // mode-consistency term is deliberately absent here: it is proportional to
        // δu₀, which the band already forces to zero, so including it would change
        // no fixed point and only the iterate count.
        let mixed = (dev + pen).value() + if exact { 2.0 * logdet.value() } else { 0.0 };
        if it + 1 >= min_iters && (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
            converged = true;
            break;
        }
        // Period-2 detector — see `PIRLS_OSC_RATIO`. Off on a dual
        // solve: a halved step is not the one-step Hessian step `DualStep::exact`
        // promises.
        let dmix = mixed - mixed_prev;
        if dual.is_none()
            && dmix.is_finite()
            && dmix_prev.is_finite()
            && dmix * dmix_prev < 0.0
            && dmix.abs() > PIRLS_OSC_RATIO * dmix_prev.abs()
        {
            osc_flips += 1;
        } else {
            osc_flips = 0;
        }
        damp |= osc_flips >= PIRLS_OSC_TRIGGER;
        dmix_prev = dmix;
        mixed_prev = mixed;
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
            None => {
                return (
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    false,
                )
            }
        }
    }
    (dev, pen, logdet, converged)
}
