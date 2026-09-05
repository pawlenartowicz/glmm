//! Block-diagonal PIRLS solve for the no-extras regime (`pirls_solve_blocked`).

use super::*;

/// Block-diagonal PIRLS for the no-extras regime (`groupings.extra_offsets` empty):
/// `A = M'WM + I` is `s` independent `q_p×q_p` blocks because each row loads exactly
/// one cluster's `q_p` columns. `m_buf` (row-major n×q_p) holds `mᵢ = Λ_p'·zᵢ`
/// (`zᵢ = [1, x[i, slope_cols]]`, slope columns pre-widened per fit into `z_buf`
/// by `fill_z_f64`), filled once per solve since Λ and x are fixed within one.
/// The η-pass forms ηᵢ and the deviance from it; the scatter-pass accumulates
/// `wᵢ·mᵢmᵢ'` into cluster `i`'s block plus `mᵢ·(yᵢ−pᵢ)` into its RHS — keeping
/// the dense path's `O(n·k²)` Gram/RHS collapsed to `O(n·q_p²)`. Then per block: `rhs_f = (A_f−I)·u_f + g_f`, Crout factor (log|A|
/// off the pivots), solve `u_f`. NOT bit-identical to `pirls_solve` (reordered
/// accumulation) but the same estimator. Mirrors `pirls_solve`'s half-step
/// (w/A from the pre-update u, u updated after) so the two agree to FP error.
/// `lam` is Λ_p row-major (`lam[r·q+c]`) from `primary_lambda`. Leaves `a_blocks`
/// FACTORED (per-block L of the final iterate) for `blocked_schur_fill` to reuse,
/// and eta/prob/w/u filled. Returns `(dev, ‖u‖², log|A|, converged)`; a non-PD
/// block ⇒ `(NaN, NaN, NaN, false)`. Iterates from the caller-provided `u` (the
/// warm-start seed); the caller owns resetting it per fit.
///
/// Step-halving: see `pirls_solve`'s doc — identical mechanism, `a_blocks` here
/// plays the role of `pirls_solve`'s `A`, so `a_blocks`/`log|A|` on return are
/// from the final step's assembly, preserving the `blocked_schur_fill` contract
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
    lam: &[T],
    z_buf: &[f64],
    m_buf: &mut [T],
    eta: &mut [T],
    prob: &mut [T],
    w: &mut [T],
    u: &mut [T],
    u_prev: &mut [T],
    eta_fixed: &mut [T],
    a_blocks: &mut [T],
    a_rhs: &mut [T],
    // Dual derivative kernels' per-solve controls (see `DualStep`); `None` on
    // every f64 fit-path call.
    mut dual: Option<&mut DualStep<T>>,
    // n × p = W∘X GEMM scratch for the Profile β-Schur border's C = X'WX
    // (mirrors `pirls_solve`'s `wx`; this variant has no dense M so there is no
    // `wm` twin here — B' = X'WM is filled by cluster-scatter instead).
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`FitOptions::offset`), added into
    // `eta_fixed` below and by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
    // Observation-only — mirrors `pirls_solve`'s `counters` (dense.rs), where
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
    let q = g.primary_q;
    let s = g.n_primary;
    let k = q * s;
    let p = beta.len();
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    for i in 0..n {
        let mut e = T::ZERO;
        for j in 0..p {
            e += T::from_f64(x[(i, j)]) * beta[j];
        }
        eta_fixed[i] = e;
    }
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += T::from_f64(o[i]);
        }
    }
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
    // First-trial backtrack seeds (u_prev = 0, beta_prev = caller's β) — only
    // the domain-infeasibility trigger can read them; see `pirls_solve`.
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
    let mut mixed_prev = f64::INFINITY; // today's mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` from the previous step
    let mut halvings = 0usize;
    let mut converged = false;
    let min_iters = match dual.as_deref_mut() {
        Some(d) => {
            // Unconditional `true` because this path pairs it with
            // `observed = !canonical`: the step it takes IS the Hessian step.
            // Mirrors `pirls_solve_blocked_extras`'s `exact` contract — change
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
    'iters: for it in 0..PIRLS_MAX_ITERS {
        counters.set_pirls_iters(it + 1);
        // --- trial evaluation at the CURRENT u: η-pass then η→prob/w/dev. On a
        // fresh accept this is the newly-stepped u; after a halving `continue` it
        // is the backtracked u. Either way the recompute IS the trial evaluation.
        // Loop-split: the transcendental runs vectorized over a materialized η[]
        // with no gather/scatter data deps.
        // --- pass 1: η-pass (scalar gather): form ηᵢ, accumulate Σ y·η ---
        let mut yeta = T::ZERO;
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let ubase = cluster_ids[i] as usize * q;
            let mut e = eta_fixed[i];
            for c in 0..q {
                e += m_row[c] * u[ubase + c];
            }
            eta[i] = e;
            yeta += T::from_f64(y[i]) * e;
        }
        // --- pass 2: η[] → prob[]/w[] + deviance, through the shared family
        // kernel (clamps η in place; `infeasible` flags any raw η outside the
        // link's open domain — Gamma-inverse only, mirrors `pirls_solve`). ---
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
        // Retrospective step-halving (lme4 `pwrssUpdate`, mirrors `pirls_solve`):
        // convergence band checked BEFORE the overshoot test (near the optimum
        // Fisher scoring is not strictly monotone — a step can land ε above
        // `pen_accepted` yet inside the tol band, and that must converge, not burn
        // all 10 halvings against FP noise). ‖u‖² is at the CURRENT trial u.
        let mut pen_u = T::ZERO;
        #[allow(clippy::needless_range_loop)]
        for c in 0..k {
            pen_u += u[c] * u[c];
        }
        let penalized = dev + pen_u;
        // BAND-TOLERANT overshoot test, mirrors `pirls_solve` (see its comments
        // for why a within-band rise is accepted rather than converged-on or
        // halved): only a rise EXCEEDING the tol band backtracks. A
        // domain-infeasible trial halves regardless of the band (see
        // `pirls_solve`'s comment).
        // Exact mode: the retrospective band above is on `dev + pen_u` alone,
        // which is not the profiled objective (missing 2·log|A|) — only a
        // domain-infeasible trial halves here; an overshoot on `dev + pen_u`
        // is judged later, after log|A| is known, against the FULL merit.
        if infeasible
            || (!exact && penalized.value() - pen_accepted > tol * (1.0 + penalized.value().abs()))
        {
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
                // `pirls_solve`'s Profile backtrack.
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
        // fresh full Fisher step from it (cold start: pen_accepted = ∞ ⇒ always
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
        // Exact-mode observed twin of `a_blocks` (non-canonical links only —
        // canonical Fisher IS observed, so `obs_blocks` stays unused there).
        if let BetaStep::Profile {
            exact: Some(ex), ..
        } = &mut beta_step
        {
            if !crate::family::is_canonical(family) {
                for v in ex.obs_blocks[..s * q * q].iter_mut() {
                    *v = 0.0;
                }
            }
        }
        // --- pass 3: scatter-pass (scalar): wᵢmᵢmᵢ' and rᵢ·mᵢ into the blocks.
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
                    other => {
                        let dmu = crate::family::mu_eta(other, eta[i]);
                        let v = crate::family::variance(other, nb_theta, prob[i]);
                        dmu * (T::from_f64(y[i]) - prob[i]) / v
                    }
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
            // Exact-mode twin: A_obs = M'W_obs M + I for the û-path adjoint solve
            // (pass B below reads this same non-canonical observed factor).
            if let BetaStep::Profile {
                exact: Some(ex), ..
            } = &mut beta_step
            {
                if !crate::family::is_canonical(family) {
                    let wo = crate::family::observed_weight(
                        family, nb_theta, y[i], prior_w[i], eta[i], prob[i], wi,
                    )
                    .value();
                    for r in 0..q {
                        let wr = wo * m_row[r].value();
                        #[allow(clippy::needless_range_loop)]
                        for c in 0..=r {
                            ex.obs_blocks[ablk + r * q + c] += wr * m_row[c].value();
                        }
                    }
                }
            }
        }
        // Profile mode: accumulate the β-gradient X'ρ (ρ = effective residual) into
        // `beta_rhs` — the joint system's bottom-block RHS. A dedicated pass off the
        // fresh prob/eta/w (NOT folded into the scatter loop above), so the Fixed
        // path stays byte-identical. Mirrors `pirls_solve`'s Profile X'ρ fold.
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
                        let dmu = crate::family::mu_eta(other, eta[i]).value();
                        let v = crate::family::variance(other, nb_theta, prob[i]).value();
                        let rho = prior_w[i] * dmu * (y[i] - prob[i].value()) / v;
                        for j in 0..p {
                            beta_rhs[j] += x[(i, j)] * rho;
                        }
                    }
                }
            }
        }
        // --- per block: rhs_f = (A_f−I)·u_old_f + g_f ; +I ; factor ; solve u_new.
        // logdet is reset-then-accumulated here (off the factor that produces this
        // step's u); on the converged (final) iteration it — and a_blocks, left
        // holding that iterate's per-block L — describe exactly the factor that
        // produced the returned u, preserving the `blocked_schur_fill` contract
        // (a step may be re-taken after a halving, hence the reset). ---
        logdet = T::ZERO;
        pen = T::ZERO;
        for f in 0..s {
            let ablk = f * q * q;
            let ubase = f * q;
            // Observed step first (reads g_f off `a_rhs` before the Fisher rhs
            // overwrites it): rhs_obs = (A_obs,f − I)·u_old_f + g_f, +I, factor,
            // solve into `obs_u`. A non-PD observed block leaves `obs_u` unused
            // and the step below falls back to the Fisher solve for this block.
            // The Fisher block is still formed and factored below on every
            // route — it is what `log|A|` and the returned factor come from.
            let mut obs_u = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
            let mut have_obs = false;
            if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
                let ob = &mut d.obs_blocks[ablk..ablk + q * q];
                for r in 0..q {
                    let mut acc = a_rhs[ubase + r];
                    for c in 0..q {
                        let (hi, lo) = if r >= c { (r, c) } else { (c, r) };
                        acc += ob[hi * q + lo] * u[ubase + c];
                    }
                    obs_u[r] = acc;
                }
                for r in 0..q {
                    ob[r * q + r] += T::ONE;
                }
                if glmm_block_chol(ob, q) {
                    glmm_block_solve(ob, q, &mut obs_u[..q]);
                    have_obs = true;
                } else {
                    d.exact = false;
                }
            }
            // (A_f − I)·u_old_f added to g_f (in a_rhs), using the still-unfactored
            // symmetric lower triangle.
            for r in 0..q {
                let mut acc = a_rhs[ubase + r];
                for c in 0..q {
                    let (hi, lo) = if r >= c { (r, c) } else { (c, r) };
                    acc += a_blocks[ablk + hi * q + lo] * u[ubase + c];
                }
                a_rhs[ubase + r] = acc;
            }
            for r in 0..q {
                a_blocks[ablk + r * q + r] += T::ONE;
            }
            if !glmm_block_chol(&mut a_blocks[ablk..ablk + q * q], q) {
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
                    if halvings < PIRLS_MAX_HALVINGS {
                        halvings += 1;
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
            for r in 0..q {
                logdet += a_blocks[ablk + r * q + r].ln();
            }
            // solve u_new_f = A_f⁻¹ rhs_f in place (rhs lives in a_rhs[ubase..], copy to u).
            u[ubase..ubase + q].copy_from_slice(&a_rhs[ubase..ubase + q]);
            glmm_block_solve(&a_blocks[ablk..ablk + q * q], q, &mut u[ubase..ubase + q]);
            if have_obs {
                u[ubase..ubase + q].copy_from_slice(&obs_u[..q]);
            }
            for r in 0..q {
                pen += u[ubase + r] * u[ubase + r];
            }
        }
        // Exact mode, non-canonical link: factor this iteration's A_obs blocks
        // (scattered above) for the c_β û-path adjoint solve (pass B below).
        // `obs_ok = false` — a non-PD observed block, or a canonical link where
        // `obs_blocks` was never written — routes c_β's û path onto the Fisher
        // factor already left in `a_blocks` instead; the direct part of c_β is
        // unaffected (it only reads `h_i` off whichever factor `a_blocks` holds).
        let mut obs_ok = false;
        if let BetaStep::Profile {
            exact: Some(ex), ..
        } = &mut beta_step
        {
            if !crate::family::is_canonical(family) {
                obs_ok = true;
                for f in 0..s {
                    let ob = &mut ex.obs_blocks[f * q * q..(f + 1) * q * q];
                    for r in 0..q {
                        ob[r * q + r] += 1.0;
                    }
                    if !glmm_block_chol(ob, q) {
                        obs_ok = false;
                        break;
                    }
                }
            }
        }
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
            // P1 exact-profile correction: c_β = d log|A|/dβ, entering the same
            // Newton RHS with the same ½ as the objective's `2·logdet` term
            // (`logdet = ½ log|A|`). Direct part: Σᵢ w'ᵢ·xᵢⱼ·hᵢ, hᵢ = mᵢ'A⁻¹mᵢ the
            // RE leverage (`block_leverage` on this iteration's factor). û path:
            // β also moves ũ(β), so log|A|(β) picks up gᵤ'·dũ/dβ with
            // gᵤ = ∂log|A|/∂u and dũ/dβ = −Ã⁻¹M'W̃X; folded as ONE adjoint solve
            // v = Ã⁻¹gᵤ (pass B) rather than materializing the k×p `dũ/dβ` —
            // one adjoint solve instead of a k×p buffer for a term only ever
            // left-multiplied by a row. On a canonical link Ã = A (Fisher, already in `a_blocks`)
            // and W̃ = W; on a non-canonical link Ã = A_obs (`obs_ok` from the
            // factor step above) and W̃ = W_obs — a per-ITERATION choice (`obs_ok`
            // is one bool for the whole solve), not per-cluster, since a single
            // non-PD observed block already means this trial falls back to Fisher
            // everywhere for consistency with the fixed-point map `dual` uses.
            let gu_dot_du = {
                let ExactProfileBufs {
                    logdet_u,
                    logdet_beta,
                    obs_blocks,
                    fac_f64,
                    ..
                } = &mut **ex;
                let logdet_u = &mut logdet_u[..k];
                let logdet_beta = &mut logdet_beta[..p];
                logdet_u.fill(0.0);
                logdet_beta.fill(0.0);
                let use_obs = obs_ok;
                // One f64 mirror of the s per-cluster factors for the three
                // passes below (see `ExactProfileBufs::fac_f64`). Built from
                // THIS iterate's factors: the block sweep above leaves
                // `a_blocks` factored, and nothing between here and pass B
                // writes it.
                let fac_f64 = &mut fac_f64[..q * q * s];
                for (o, v) in fac_f64.iter_mut().zip(a_blocks[..q * q * s].iter()) {
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
                    let h = block_leverage(&fac_f64[ablk..ablk + q * q], q, &mrow[..q]);
                    let wp = if w[i].value() <= crate::glm::WEIGHT_CLAMP {
                        0.0
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
                // pass B: v = Ã⁻¹ gᵤ per cluster (Ã = A_obs on a non-canonical
                // link when its factor is PD, else the Fisher factor in `a_blocks`).
                for f in 0..s {
                    let ablk = f * q * q;
                    let fac: &[f64] = if use_obs {
                        &obs_blocks[ablk..ablk + q * q]
                    } else {
                        &fac_f64[ablk..ablk + q * q]
                    };
                    glmm_block_solve(fac, q, &mut logdet_u[f * q..f * q + q]);
                }
                // pass C: û path, c_β_j −= Σᵢ w̃ᵢ·(mᵢ·v_f)·xᵢⱼ.
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    let mut sdot = 0.0;
                    for c in 0..q {
                        sdot += m_buf[i * q + c].value() * logdet_u[f * q + c];
                    }
                    let wt = if use_obs {
                        crate::family::observed_weight(
                            family, nb_theta, y[i], prior_w[i], eta[i], prob[i], w[i],
                        )
                        .value()
                    } else {
                        w[i].value()
                    };
                    let a = wt * sdot;
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
            // correction that vanishes as δu₀ → 0. On a canonical link the
            // u-step is Newton on the same A that log|A| uses, so δu₀ tracks
            // ũ−u to second order and the corrected merit equals the Laplace
            // deviance to second order in δu₀ (the residual is what the accept
            // band below absorbs). On a non-canonical link the u-step is
            // Fisher, not Newton, so δu₀ only approximates ũ−u to the
            // Fisher/observed curvature ratio, and the same band absorbs that
            // larger slack too.
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
            // reaches, all 50 iterations went to halving, and the θ-only outer
            // search read the `inf` that came back as a wall — with the seed's own
            // 1.4e-4 correction charged, that deficit is inside the band. Both
            // terms shrink with their δu₀, so at the mode the test is the value
            // band again, and the convergence test below is untouched.
            if l_trial - l_acc > tol * (1.0 + l_trial.abs()) + gu_dot_du.abs() + l_acc_slack {
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
        // `se::dense_schur_fill`'s doc comment for the shared β-Schur Newton step
        // this mirrors). `m_buf` already holds mᵢ = Λ'zᵢ, so B's scatter reads it
        // directly (cheaper than se.rs's per-row reconstruction). ---
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
                    glmm_block_solve(&a_blocks[ablk..ablk + q * q], q, &mut rhs[..q]);
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
            if let Some(ex) = exact.as_deref() {
                for r in 0..p {
                    beta_rhs[r] -= 0.5 * ex.logdet_beta[r];
                }
            }
            // δβ = S_β⁻¹·rhs in place. Non-PD S_β ⇒ the (NaN,…,false) failure surface.
            if cholesky_in_place(
                schur.as_mut(),
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(schur_llt_mem),
                Spec::default(),
            )
            .is_err()
            {
                return (
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    T::from_f64(f64::NAN),
                    false,
                );
            }
            solve_in_place(
                schur.as_ref(),
                MatMut::from_column_major_slice_mut(&mut beta_rhs[..p], p, 1),
                Par::Seq,
                MemStack::new(schur_llt_mem),
            );
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
        // Today's stopping rule, verbatim: the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` band on
        // successive steps — bit-identical iterate path and returned values to the
        // pre-halving loop when no halving fires (see `pirls_solve` for why the
        // same-point band cannot be a converge trigger).
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
        mixed_prev = mixed;
    }
    (dev, pen, logdet, converged)
}
