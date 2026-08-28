//! Dense PIRLS solve (`pirls_solve`).

use super::*;

/// Penalized-IRLS inner solve by Fisher scoring on the penalized likelihood with
/// M = ZΛ the scaled RE design and a
/// +I ridge (the standard nAGQ=1 reparameterization — u ~ N(0, I)). The `beta_step`
/// mode sets what moves: `BetaStep::Fixed` holds β at the caller's input and solves
/// for the conditional modes ũ(β) alone (the objective stays a function of the
/// caller's β — required by the FD-Hessian path and BOBYQA stage 2); `BetaStep::Profile`
/// adds a joint δβ Schur-border step each iteration (§β-Schur math) so the returned
/// (ũ, β̂) is jointly PQL-optimal for this θ, writing β̂ back through `beta`. At each
/// step A = M'WM + I and the IRLS RHS `M'(W·Mu + (y − p))` give the next u via a
/// dense Cholesky solve. Returns (deviance `2·d`, ‖ũ‖², `log|A|` at the converged
/// iterate, converged); a Cholesky failure surfaces as `(NaN, NaN, NaN, false)`.
/// `log|A|` is read off the same converged factor that solved for the final u —
/// the caller need not re-factor A (faer `llt` is deterministic).
/// Mu, A = M'(W∘M) + I, and the RHS all go through faer GEMM (`Par::Seq`) with
/// `wm` as the W∘M scratch — GEMM accumulation order, deliberately NOT the old
/// per-entry i-order dots (the serial FP-add chain was the dense path's latency
/// floor). `eta_fixed`/`mu` are caller-owned
/// length-n scratch. `A` is left holding the FINAL-iterate A for the caller.
/// Iterates from the caller-provided `u` (the warm-start seed); the caller owns
/// resetting it per fit.
///
/// **Step-halving (lme4 `pwrssUpdate`, retrospective form):** each iteration
/// evaluates the trial `u` first; only if the same-point penalized deviance
/// `dev + ‖u‖²` rose above the last accepted value BY MORE than the tol band
/// does it halve `δu = u − u_prev` and re-evaluate, up to `PIRLS_MAX_HALVINGS`
/// times (then `(NaN, NaN, NaN, false)`). A within-band rise is FP noise near
/// the optimum and is accepted — it never burns a halving. `u_prev` is the
/// caller-owned length-k backtrack buffer; in `Profile` mode the joint (u,β) step
/// is backtracked in lockstep, halving β toward `beta_prev` (the `BetaStep::Profile`
/// twin of `u_prev`) alongside u. Convergence is today's rule, verbatim: the mixed
/// `dev(uⱼ) + ‖uⱼ₊₁‖²` band on successive steps, checked AFTER the step — so
/// when no halving fires the iterate path and returned values are bit-identical
/// to the pre-halving loop (`dev`/`w`/`eta`/`prob` at the assembly point,
/// `pen = ‖u_new‖²`, `a`/`logdet` from the factor that produced the returned u).
/// The same-point band must NOT be a converge trigger — see the in-loop comment
/// for the measured one-iteration-early breakage.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pirls_solve(
    family: Family,
    nb_theta: f64,
    k: usize,
    p: usize,
    m: MatRef<f64>,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    beta: &mut [f64],
    mut beta_step: BetaStep,
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    u_prev: &mut [f64],
    eta_fixed: &mut [f64],
    mu: &mut [f64],
    wm: &mut Mat<f64>,
    // n × p = W∘X GEMM scratch for the Profile β-Schur border's C = X'WX
    // (mirrors `wm` above for M; rebuilt fresh from this iteration's `w` each
    // Profile step, so no stale-value hazard across PIRLS iterations).
    wx: &mut Mat<f64>,
    // `a` must survive this call holding the raw symmetric M'WM+I — `dense_schur_fill`
    // (se.rs) re-factors it after a converged Fixed-mode solve (the `blocked_schur_fill`
    // twin's contract, mirrored here). The Cholesky factor+solve therefore runs on the
    // separate `a_chol` copy below, never on `a` in place.
    a: &mut Mat<f64>,
    // Copy-then-factor target for `a`'s Cholesky (ws.a_chol), sized k×k — mirrors
    // `.llt(Side::Lower)`'s internal `copy_from_triangular_lower` + factor, just
    // against a persistent buffer instead of a fresh heap allocation each call.
    a_chol: &mut Mat<f64>,
    a_rhs: &mut [f64],
    // Persistent scratch for `a_chol`'s in-place `cholesky_in_place`, sized once
    // for k×k at workspace construction (ws.a_llt_mem) — avoids the
    // `.llt(Side::Lower)` per-PIRLS-iteration heap allocation on this hot RE-block
    // solve.
    a_llt_mem: &mut MemBuffer,
    // Per-row linear-predictor offset (`FitOptions::offset`), added into
    // `eta_fixed` by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
) -> (f64, f64, f64, bool) {
    use faer::linalg::matmul::triangular::BlockStructure;
    use faer::linalg::matmul::{matmul, triangular};
    use faer::{Accum, MatMut, Par};
    // m arrives max_n × k with the first n rows live; GEMM needs exact dims.
    let m = m.subrows(0, n);
    // η_fixed,ᵢ = Σ_j x[i,j]·β[j]. In Fixed mode β is invariant across iterations
    // so this once-at-entry fill stands for the whole solve; in Profile mode the
    // δβ step re-fills it after every β update (below), hence the shared helper.
    refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
    // Backtrack seeds for the FIRST trial iterate (which has no accepted
    // predecessor): u_prev = 0 so an infeasible first trial halves toward
    // η = eta_fixed (the canonical cold seed), beta_prev = the caller's β.
    // Dead for the overshoot trigger — it cannot fire before an accept (a rise
    // above an infinite `pen_accepted` never tests true) — so only the
    // domain-infeasibility trigger below ever reads these seeds, and the
    // iterate path is untouched wherever it never fires.
    u_prev[..k].fill(0.0);
    if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
        beta_prev[..p].copy_from_slice(&beta[..p]);
    }
    let mut pen_accepted = f64::INFINITY; // same-point penalized deviance at the last ACCEPTED iterate
    let mut mixed_prev = f64::INFINITY; // today's mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` from the previous step
    let mut halvings = 0usize;
    let mut converged = false;
    let mut dev = f64::NAN;
    let mut pen = f64::NAN; // ‖u‖² at the returned (post-step) iterate
    let mut logdet = 0.0;
    let tol = pirls_tol_override.unwrap_or_else(|| super::super::pirls_tol(family));
    for _ in 0..PIRLS_MAX_ITERS {
        // --- trial evaluation at the CURRENT u: (Mu)ᵢ, then η/dev/prob/W. On a
        // fresh accept this is the newly-stepped u; after a halving `continue` it is
        // the backtracked u. Either way the recompute IS the trial evaluation. ---
        matmul(
            MatMut::from_column_major_slice_mut(&mut mu[..n], n, 1),
            Accum::Replace,
            m,
            MatRef::from_column_major_slice(&u[..k], k, 1),
            1.0,
            Par::Seq,
        );
        // η = η_fixed + Mu (raw), then one batched call into the shared family
        // kernel for deviance + p/W. `infeasible` = any row's RAW η outside the
        // link's open domain (`family::eta_infeasible` — Gamma-inverse only;
        // constant-false, hence dead, for every all-ℝ link).
        let mut yeta = 0.0;
        for i in 0..n {
            let e = eta_fixed[i] + mu[i];
            eta[i] = e;
            yeta += y[i] * e;
        }
        let (d, infeasible) = crate::simd_transcendental::family_pass(
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
        // Retrospective step-halving (lme4 `pwrssUpdate`): the convergence band is
        // tested BEFORE the overshoot test because near the optimum Fisher scoring
        // is not strictly monotone — a step can land ε above `pen_accepted` yet
        // inside the tol band, and that must converge (today's behavior), not burn
        // all 10 halvings against FP noise.
        let pen_u: f64 = u[..k].iter().map(|v| v * v).sum();
        let penalized = dev + pen_u;
        // BAND-TOLERANT overshoot test (the convergence band is consulted before
        // any halving): a rise within the tol band is FP noise near the optimum —
        // Fisher scoring is not strictly monotone there — so it is ACCEPTED (never
        // burns a halving) and the mixed rule below terminates, today's behavior.
        // Only a rise EXCEEDING the band is a genuine overshoot worth backtracking.
        // The band must NOT itself be a converge trigger: the same-point objective
        // flattens quadratically while the iterate is still moving (dev and pen
        // trade off along the valley), so it fires one full iteration before
        // today's mixed rule — measured on the intercept fixture (samepoint diff
        // 3.9e-5 inside the 1.07e-4 band while the mixed sequence was still 7.2e-3
        // apart), returning an iterate ~4e-7 coarse in the objective and breaking
        // the AGQ(k=1) ≡ Laplace 1e-12 gate and the non-canonical FD-Hessian SEs.
        //
        // A domain-infeasible trial iterate (`infeasible`, Gamma-inverse only) is
        // a step failure REGARDLESS of the band: accepting it would let the
        // `clamp_eta` boundary projection define the converged answer (see
        // `family::eta_infeasible`). It halves toward the last accepted feasible
        // iterate — or, on the very first trial, toward the u = 0 / caller-β
        // seeds installed at entry. An infeasible η_fixed itself is beyond
        // halving's reach (u = 0 already gives η = η_fixed): halvings exhaust
        // into the honest (NaN, …, false).
        if infeasible || penalized - pen_accepted > tol * (1.0 + penalized.abs()) {
            if halvings < PIRLS_MAX_HALVINGS {
                // Last full step overshot: halve δu = u − u_prev and re-enter the
                // top (the recompute above is the trial evaluation of the halved
                // step).
                halvings += 1;
                for c in 0..k {
                    u[c] = 0.5 * (u[c] + u_prev[c]);
                }
                // Profile mode: the trial point is the JOINT (u,β) step, so the
                // backtrack halves β toward `beta_prev` in lockstep with u, then
                // refreshes η_fixed for the re-evaluation at the top.
                if let BetaStep::Profile { beta_prev, .. } = &beta_step {
                    for j in 0..p {
                        beta[j] = 0.5 * (beta[j] + beta_prev[j]);
                    }
                    refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
                }
                continue;
            }
            return (f64::NAN, f64::NAN, f64::NAN, false); // halvings exhausted
        }
        // Accept this iterate, snapshot it for the next backtrack, and take a
        // fresh full Fisher step from it (cold start: pen_accepted = ∞ ⇒ always
        // accepts).
        halvings = 0;
        pen_accepted = penalized;
        u_prev[..k].copy_from_slice(&u[..k]);
        // Profile mode: snapshot the accepted β as the β-halving twin of u_prev.
        if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
            beta_prev[..p].copy_from_slice(&beta[..p]);
        }
        // WM = W∘M, then A = M′·(WM) + I — lower triangle only (the Cholesky
        // below reads Side::Lower).
        for c in 0..k {
            for i in 0..n {
                wm[(i, c)] = w[i] * m[(i, c)];
            }
        }
        triangular::matmul(
            a.as_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            m.transpose(),
            BlockStructure::Rectangular,
            wm.as_ref().subrows(0, n),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        for r in 0..k {
            a[(r, r)] += 1.0;
        }
        // IRLS RHS M′(W·Mu + W·r): fold the effective residual into mu in place
        // (mu is dead until next iteration's refill), then one GEMV. Logit's
        // W·r = (y−p); the general branch is W·r = wᵢ·(dμ/dη)·(y−μ)/V.
        match family {
            Family::Binomial {
                link: BinomialLink::Logit,
            } if !weighted => {
                for i in 0..n {
                    mu[i] = w[i] * mu[i] + (y[i] - prob[i]);
                }
            }
            other => {
                for i in 0..n {
                    let dmu = crate::family::mu_eta(other, eta[i]);
                    let v = crate::family::variance(other, nb_theta, prob[i]);
                    mu[i] = w[i] * mu[i] + prior_w[i] * dmu * (y[i] - prob[i]) / v;
                }
            }
        }
        // Profile mode: accumulate the β-gradient X'ρ (ρ = effective residual) into
        // `beta_rhs` for this iteration's δβ Schur step. A dedicated pass off the
        // fresh prob/eta/w — NOT folded into the residual loop above — so the Fixed
        // path stays byte-identical (prob/eta are untouched by that fold; only `mu`
        // was overwritten with the IRLS working vector). This is the joint system's
        // bottom-block RHS `X'ρ`.
        if let BetaStep::Profile { beta_rhs, .. } = &mut beta_step {
            for v in beta_rhs[..p].iter_mut() {
                *v = 0.0;
            }
            match family {
                Family::Binomial {
                    link: BinomialLink::Logit,
                } => {
                    for i in 0..n {
                        let rho = prior_w[i] * (y[i] - prob[i]);
                        for j in 0..p {
                            beta_rhs[j] += x[(i, j)] * rho;
                        }
                    }
                }
                other => {
                    for i in 0..n {
                        let dmu = crate::family::mu_eta(other, eta[i]);
                        let v = crate::family::variance(other, nb_theta, prob[i]);
                        let rho = prior_w[i] * dmu * (y[i] - prob[i]) / v;
                        for j in 0..p {
                            beta_rhs[j] += x[(i, j)] * rho;
                        }
                    }
                }
            }
        }
        matmul(
            MatMut::from_column_major_slice_mut(&mut a_rhs[..k], k, 1),
            Accum::Replace,
            m.transpose(),
            MatRef::from_column_major_slice(&mu[..n], n, 1),
            1.0,
            Par::Seq,
        );
        // Copy A's lower triangle into the persistent `a_chol` scratch (mirrors
        // `.llt(Side::Lower)`'s own `copy_from_triangular_lower`), then factor
        // THAT in place — `a` itself must come out of this call unmutated (see
        // the param doc on `a`).
        a_chol.copy_from_triangular_lower(a.as_ref());
        if cholesky_in_place(
            a_chol.as_mut(),
            LltRegularization::default(),
            Par::Seq,
            MemStack::new(a_llt_mem),
            Spec::default(),
        )
        .is_err()
        {
            return (f64::NAN, f64::NAN, f64::NAN, false);
        }
        // log|A| off the factor that produces this step's u_new. On the converged
        // (final) iteration it — and `a`, left holding that A — describe exactly
        // the factor that produced the returned u, preserving today's "log|A| off
        // the final-iterate factor" contract. Reset before accumulating (a step
        // may be re-taken after a halving).
        logdet = 0.0;
        for r in 0..k {
            logdet += a_chol[(r, r)].ln();
        }
        // Solve in place on the caller's a_rhs scratch (1-col view) — no
        // per-iteration alloc; the solve itself is unchanged. u ← u_new.
        let rhs = MatMut::from_column_major_slice_mut(&mut a_rhs[..k], k, 1usize);
        solve_in_place(a_chol.as_ref(), rhs, Par::Seq, MemStack::new(a_llt_mem));
        pen = 0.0;
        for c in 0..k {
            u[c] = a_rhs[c];
            pen += u[c] * u[c];
        }
        // --- Profile-mode joint δβ step (β-Schur border), taken while `ac` (the
        // LLT of A) is still alive. Mirrors `se::dense_schur_fill` with THIS
        // iteration's W and factor: T = A⁻¹B, S_β = C − B'T, δβ = S_β⁻¹·rhs,
        // then u_joint = u_new − T·δβ (see `se::dense_schur_fill`'s doc comment
        // for the shared construction this mirrors). `u` currently holds
        // u_new = u_prev + δu₀, so δu₀ = u − u_prev. ---
        if let BetaStep::Profile {
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            schur_llt_mem,
            ..
        } = &mut beta_step
        {
            // B' = X'WM (p×k) via the live W∘M scratch `wm` (dense_schur_fill uses
            // the same W̃∘M product, per-entry there / GEMM here).
            matmul(
                xtwm.as_mut(),
                Accum::Replace,
                x.subrows(0, n).transpose(),
                wm.as_ref().subrows(0, n),
                1.0,
                Par::Seq,
            );
            // C = X'WX (p×p) via the W∘X GEMM scratch `wx` — mirrors the X'WM GEMM
            // above (dense_schur_fill's per-entry xtwx block, same product here).
            // GEMM fills the full p×p (not just the lower triangle the old scalar
            // loop mirrored); every downstream read below is over the full matrix
            // (`schur[(r,c)] = xtwx[(r,c)] - …` for `c in 0..p`), so this is exact.
            for c in 0..p {
                for i in 0..n {
                    wx[(i, c)] = w[i] * x[(i, c)];
                }
            }
            matmul(
                xtwx.as_mut(),
                Accum::Replace,
                x.subrows(0, n).transpose(),
                wx.as_ref().subrows(0, n),
                1.0,
                Par::Seq,
            );
            // T = A⁻¹B = A⁻¹(M'WX): transpose-gather B' into ainv_mtwx, solve with
            // this iteration's factor (dense_schur_fill:283-288).
            for r in 0..k {
                for c in 0..p {
                    ainv_mtwx[(r, c)] = xtwm[(c, r)];
                }
            }
            solve_in_place(
                a_chol.as_ref(),
                ainv_mtwx.as_mut(),
                Par::Seq,
                MemStack::new(a_llt_mem),
            );
            // S_β = C − B'·T (dense_schur_fill:289-297).
            for r in 0..p {
                for c in 0..p {
                    let mut s = xtwx[(r, c)];
                    for j in 0..k {
                        s -= xtwm[(r, j)] * ainv_mtwx[(j, c)];
                    }
                    schur[(r, c)] = s;
                }
            }
            // rhs = X'ρ − B'·δu₀ (beta_rhs holds X'ρ; δu₀ = u − u_prev).
            for r in 0..p {
                let mut acc = 0.0;
                for c in 0..k {
                    acc += xtwm[(r, c)] * (u[c] - u_prev[c]);
                }
                beta_rhs[r] -= acc;
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
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            solve_in_place(
                schur.as_ref(),
                MatMut::from_column_major_slice_mut(&mut beta_rhs[..p], p, 1),
                Par::Seq,
                MemStack::new(schur_llt_mem),
            );
            // Apply: β += δβ; u = u_joint = u_new − T·δβ.
            for j in 0..p {
                beta[j] += beta_rhs[j];
            }
            for c in 0..k {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += ainv_mtwx[(c, j)] * beta_rhs[j];
                }
                u[c] -= acc;
            }
            // η_fixed depends on β; refresh it for the next trial evaluation. `pen`
            // must track the moved u (‖u_joint‖²), so recompute it.
            refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
            pen = 0.0;
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                pen += u[c] * u[c];
            }
        }
        // Today's stopping rule, verbatim: the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` band on
        // successive steps — when no halving fires the iterate path, trigger
        // point, and returned values are bit-identical to the pre-halving loop
        // (returned u is the post-step Newton-refined iterate; dev/w/eta/prob at
        // the assembly point; `a`/`logdet` from the factor that produced u).
        let mixed = dev + pen;
        if (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
            converged = true;
            break;
        }
        mixed_prev = mixed;
    }
    (dev, pen, logdet, converged)
}
