use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, LltRegularization};
use faer::linalg::cholesky::llt::solve::solve_in_place;
use faer::sparse::linalg::cholesky::LltRef;
use faer::{Conj, Mat, MatMut, MatRef, Par, Side, Spec};

use super::workspace::{
    glmm_block_chol, glmm_block_solve, glmm_block_solve_panel, StructuredSchur,
};
use super::{PIRLS_MAX_HALVINGS, PIRLS_MAX_ITERS};
use crate::sparse::logdet_llt;
use crate::spec::{BinomialLink, Family};

/// β handling for one PIRLS solve. `Fixed` = today's behavior verbatim (β is an
/// immutable input; the FD-Hessian path and BOBYQA stage 2 REQUIRE this so the
/// objective stays a function of the caller's β). `Profile` = PQL/stage-1 mode:
/// a δβ Schur-border update runs each iteration and the converged β is written
/// back through `beta`. No default — every call site chooses explicitly.
pub(crate) enum BetaStep<'a> {
    Fixed,
    Profile {
        xtwx: &'a mut Mat<f64>,      // p×p  C = X'WX          (ws.xtwx)
        xtwm: &'a mut Mat<f64>,      // p×k  B' = X'WM         (ws.xtwm)
        ainv_mtwx: &'a mut Mat<f64>, // k×p  T = A⁻¹B          (ws.ainv_mtwx)
        schur: &'a mut Mat<f64>,     // p×p  S_β               (ws.schur)
        beta_rhs: &'a mut [f64],     // len p: X'ρ, then rhs, then δβ in place (ws.beta_rhs)
        beta_prev: &'a mut [f64], // len p: last-accepted β for the halving backtrack (ws.beta_prev)
        // Persistent scratch for `schur`'s in-place `cholesky_in_place`, sized once
        // for p×p at workspace construction (ws.schur_llt_mem) — avoids the
        // `.llt(Side::Lower)` per-iteration heap allocation on this hot β-Schur step.
        schur_llt_mem: &'a mut MemBuffer,
    },
}

/// Refill `eta_fixed[i] = offset[i] + Σ_j x[i,j]·β[j]` (the fixed-effect linear
/// predictor). Called once at entry of `pirls_solve` and, in `BetaStep::Profile`,
/// after every β update (the accepted δβ step and each β halving) — the trial
/// evaluation at the top of the loop reads `eta_fixed`, so it must track the
/// current β. `offset` is `FitOptions::offset` (`None` ⇒ this function is
/// byte-identical to the pre-offset version).
fn refresh_eta_fixed(
    x: MatRef<f64>,
    beta: &[f64],
    eta_fixed: &mut [f64],
    n: usize,
    p: usize,
    offset: Option<&[f64]>,
) {
    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p {
            e += x[(i, j)] * beta[j];
        }
        eta_fixed[i] = e;
    }
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += o[i];
        }
    }
}

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
    let tol = pirls_tol_override.unwrap_or_else(|| super::pirls_tol(family));
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
pub(crate) fn pirls_solve_blocked(
    family: Family,
    nb_theta: f64,
    g: &crate::lmm::LmmGroupings,
    cluster_ids: &[u32],
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    beta: &mut [f64],
    mut beta_step: BetaStep,
    lam: &[f64],
    z_buf: &[f64],
    m_buf: &mut [f64],
    eta: &mut [f64],
    prob: &mut [f64],
    w: &mut [f64],
    u: &mut [f64],
    u_prev: &mut [f64],
    eta_fixed: &mut [f64],
    a_blocks: &mut [f64],
    a_rhs: &mut [f64],
    // n × p = W∘X GEMM scratch for the Profile β-Schur border's C = X'WX
    // (mirrors `pirls_solve`'s `wx`; this variant has no dense M so there is no
    // `wm` twin here — B' = X'WM is filled by cluster-scatter instead).
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`FitOptions::offset`), added into
    // `eta_fixed` below and by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
) -> (f64, f64, f64, bool) {
    let q = g.primary_q;
    let s = g.n_primary;
    let k = q * s;
    let p = beta.len();
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    for i in 0..n {
        let mut e = 0.0;
        for j in 0..p {
            e += x[(i, j)] * beta[j];
        }
        eta_fixed[i] = e;
    }
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += o[i];
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
            let mut acc = 0.0;
            for r in c..q {
                let zr = if r == 0 {
                    1.0
                } else {
                    z_buf[i * (q - 1) + (r - 1)]
                };
                acc += zr * lam[r * q + c];
            }
            m_buf[i * q + c] = acc;
        }
    }
    // First-trial backtrack seeds (u_prev = 0, beta_prev = caller's β) — only
    // the domain-infeasibility trigger can read them; see `pirls_solve`.
    u_prev[..k].fill(0.0);
    if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
        beta_prev[..p].copy_from_slice(&beta[..p]);
    }
    let mut pen_accepted = f64::INFINITY; // same-point penalized deviance at the last ACCEPTED iterate
    let mut mixed_prev = f64::INFINITY; // today's mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` from the previous step
    let mut halvings = 0usize;
    let mut converged = false;
    let (mut dev, mut pen, mut logdet) = (f64::NAN, f64::NAN, 0.0);
    let tol = pirls_tol_override.unwrap_or_else(|| super::pirls_tol(family));
    for _ in 0..PIRLS_MAX_ITERS {
        // --- trial evaluation at the CURRENT u: η-pass then η→prob/w/dev. On a
        // fresh accept this is the newly-stepped u; after a halving `continue` it
        // is the backtracked u. Either way the recompute IS the trial evaluation.
        // Loop-split: the transcendental runs vectorized over a materialized η[]
        // with no gather/scatter data deps.
        // --- pass 1: η-pass (scalar gather): form ηᵢ, accumulate Σ y·η ---
        let mut yeta = 0.0;
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let ubase = cluster_ids[i] as usize * q;
            let mut e = eta_fixed[i];
            for c in 0..q {
                e += m_row[c] * u[ubase + c];
            }
            eta[i] = e;
            yeta += y[i] * e;
        }
        // --- pass 2: η[] → prob[]/w[] + deviance, through the shared family
        // kernel (clamps η in place; `infeasible` flags any raw η outside the
        // link's open domain — Gamma-inverse only, mirrors `pirls_solve`). ---
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
        // Retrospective step-halving (lme4 `pwrssUpdate`, mirrors `pirls_solve`):
        // convergence band checked BEFORE the overshoot test (near the optimum
        // Fisher scoring is not strictly monotone — a step can land ε above
        // `pen_accepted` yet inside the tol band, and that must converge, not burn
        // all 10 halvings against FP noise). ‖u‖² is at the CURRENT trial u.
        let pen_u: f64 = u[..k].iter().map(|v| v * v).sum();
        let penalized = dev + pen_u;
        // BAND-TOLERANT overshoot test, mirrors `pirls_solve` (see its comments
        // for why a within-band rise is accepted rather than converged-on or
        // halved): only a rise EXCEEDING the tol band backtracks. A
        // domain-infeasible trial halves regardless of the band (see
        // `pirls_solve`'s comment).
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
                // refreshes η_fixed for the re-evaluation at the top. Mirrors
                // `pirls_solve`'s Profile backtrack.
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
        for v in a_blocks[..s * q * q].iter_mut() {
            *v = 0.0;
        }
        for v in a_rhs[..k].iter_mut() {
            *v = 0.0;
        }
        // --- pass 3: scatter-pass (scalar): wᵢmᵢmᵢ' and rᵢ·mᵢ into the blocks.
        // The effective residual rᵢ is logit's (yᵢ−pᵢ) or the general W·working_resid. ---
        for i in 0..n {
            let m_row = &m_buf[i * q..i * q + q];
            let f = cluster_ids[i] as usize;
            let ubase = f * q;
            let ablk = f * q * q;
            let wi = w[i];
            let resid = prior_w[i]
                * match family {
                    Family::Binomial {
                        link: BinomialLink::Logit,
                    } => y[i] - prob[i],
                    other => {
                        let dmu = crate::family::mu_eta(other, eta[i]);
                        let v = crate::family::variance(other, nb_theta, prob[i]);
                        dmu * (y[i] - prob[i]) / v
                    }
                };
            for r in 0..q {
                a_rhs[ubase + r] += m_row[r] * resid;
                let wr = wi * m_row[r];
                for c in 0..=r {
                    a_blocks[ablk + r * q + c] += wr * m_row[c];
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
        // --- per block: rhs_f = (A_f−I)·u_old_f + g_f ; +I ; factor ; solve u_new.
        // logdet is reset-then-accumulated here (off the factor that produces this
        // step's u); on the converged (final) iteration it — and a_blocks, left
        // holding that iterate's per-block L — describe exactly the factor that
        // produced the returned u, preserving the `blocked_schur_fill` contract
        // (a step may be re-taken after a halving, hence the reset). ---
        logdet = 0.0;
        pen = 0.0;
        for f in 0..s {
            let ablk = f * q * q;
            let ubase = f * q;
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
                a_blocks[ablk + r * q + r] += 1.0;
            }
            if !glmm_block_chol(&mut a_blocks[ablk..ablk + q * q], q) {
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            for r in 0..q {
                logdet += a_blocks[ablk + r * q + r].ln();
            }
            // solve u_new_f = A_f⁻¹ rhs_f in place (rhs lives in a_rhs[ubase..], copy to u).
            u[ubase..ubase + q].copy_from_slice(&a_rhs[ubase..ubase + q]);
            glmm_block_solve(&a_blocks[ablk..ablk + q * q], q, &mut u[ubase..ubase + q]);
            for r in 0..q {
                pen += u[ubase + r] * u[ubase + r];
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
                    wx[(i, c)] = w[i] * x[(i, c)];
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
                let wi = w[i];
                for r in 0..p {
                    let xw = x[(i, r)] * wi;
                    for c in 0..q {
                        xtwm[(r, f * q + c)] += xw * m_buf[i * q + c];
                    }
                }
            }
            // T_f = A_f⁻¹ (M'WX)_f per block, reusing this iteration's factor left in
            // `a_blocks`; ainv_mtwx rows f·q.. hold T_f. (M'WX)_f[c, col] = xtwm[(col, f·q+c)].
            // Mirrors blocked_schur_fill:360-374.
            for f in 0..s {
                let ablk = f * q * q;
                for col in 0..p {
                    let mut rhs = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
                    for c in 0..q {
                        rhs[c] = xtwm[(col, f * q + c)];
                    }
                    glmm_block_solve(&a_blocks[ablk..ablk + q * q], q, &mut rhs[..q]);
                    for c in 0..q {
                        ainv_mtwx[(f * q + c, col)] = rhs[c];
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
            // Apply: β += δβ; u = u_joint = u_new − T·δβ, i.e.
            // u[f·q+c] −= Σ_j T[(f·q+c, j)]·δβ[j].
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
            // η_fixed depends on β; refresh for the next trial. `pen` must track the
            // moved u (‖u_joint‖²), so recompute it.
            refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
            pen = 0.0;
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                pen += u[c] * u[c];
            }
        }
        // Today's stopping rule, verbatim: the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` band on
        // successive steps — bit-identical iterate path and returned values to the
        // pre-halving loop when no halving fires (see `pirls_solve` for why the
        // same-point band cannot be a converge trigger).
        let mixed = dev + pen;
        if (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
            converged = true;
            break;
        }
        mixed_prev = mixed;
    }
    (dev, pen, logdet, converged)
}

/// Factor the structured `A = [[D, C], [C', E]]` in place: Crout-factor each
/// core block `core_blocks[f]` (holding `D_f + I` on entry, its lower `L` on
/// return) and build + factor the Schur complement `schur_blk` (holding `E + I`
/// on entry, `S = (E+I) − Σ_f C_f' A_f⁻¹ C_f` then its `L` on return). Returns
/// `Some(log|A|) = Σ_f log|A_f| + log|S|` (Schur determinant identity), or `None`
/// on a non-PD core block / Schur. `coupling[f]` holds `C_f` (q_core×e row-major),
/// unchanged. Shared by `pirls_solve_blocked_extras` (per iteration) and
/// `structured_schur_fill` (reusing the converged factors). `q_core ≤ MAX_PRIMARY_Q`.
/// `coup_cols`/`coup_ptr` is the per-cluster CSR of C_f's nonzero crossed columns
/// (built once per solve by `pirls_solve_blocked_extras`): the Schur build walks
/// only those columns instead of all `e` — every skipped column of `C_f` is
/// exactly 0.0 (no row of cluster `f` touches that crossed level), so skipping it
/// drops only exact-zero contributions. On an observation-level primary (s ≈ n,
/// grouseticks) this collapses the build from `s·e²/2` to the ~G² true nonzeros
/// per cluster. The downdate itself runs panel-wise per cluster (one batched
/// `A_f⁻¹` solve + one triangular `C_f'·Y` matmul through `ss`'s scratch — the
/// LMM sparse-tail kernels A–D port): each `S[a][b]` still receives exactly one
/// subtraction per touching cluster in the same `f` order; only the dot's
/// internal association moved into the matmul — a result-moving reassociation of
/// the sanctioned class (see `SparseTail`'s doc). The panel path serves
/// `qc > 1` (vector primary and/or nested REs). At `qc == 1` the downdate is
/// rank-1 and the scalar column-at-a-time walk is the **production route**
/// (routed via the `qc != 1` filter below; the panel staging is a +4–7%
/// per-eval loss there — 2026-07-14 drift investigation); the same walk is also
/// the `ss = None` arm, which is the panel path's equality oracle at `qc > 1`.
#[allow(clippy::too_many_arguments)]
fn structured_factor(
    g: &crate::lmm::LmmGroupings,
    core_blocks: &mut [f64],
    coupling: &[f64],
    schur_blk: &mut [f64],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    mut ss: Option<&mut StructuredSchur>,
    force_dense: bool,
) -> Option<f64> {
    let qc = g.primary_q + g.nested_per_parent;
    let s = g.n_primary;
    let e = g.k_crossed();
    let mut logdet = 0.0;
    for f in 0..s {
        let cb = f * qc * qc;
        if !glmm_block_chol(&mut core_blocks[cb..cb + qc * qc], qc) {
            return None;
        }
        for r in 0..qc {
            logdet += core_blocks[cb + r * qc + r].ln();
        }
        // S −= C_f' A_f⁻¹ C_f (lower triangle) over cluster f's NONZERO crossed
        // columns: gather them into a compact row-major qc×e_f panel, batch-solve
        // it, one triangular matmul, then subtract dd's lower triangle through
        // the same `cols` map (cols ascending ⇒ local lower = global lower).
        let coup = f * qc * e;
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        let e_f = cols.len();
        if e_f == 0 {
            continue;
        }
        let Some(StructuredSchur {
            c_panel,
            y_panel,
            dd_temp,
            ..
        }) = ss.as_deref_mut().filter(|_| qc != 1)
        else {
            // Production route for qc == 1 (and the ss = None oracle) — see the
            // fn doc. At qc == 1 the downdate is rank-1: the panel path stages the
            // identical FLOPs (gather → dd_temp → second scatter pass) at ~double
            // the memory traffic, a measured +4–7% per-eval loss on the qc=1
            // cross6 GLMM cells (binb 1.758→1.870 s; 2026-07-14 drift
            // investigation). This scalar walk accumulates each rank-1 dot
            // straight into schur_blk — already minimal. `qc == 1` is the only
            // measured boundary; no qc>1 GLMM structured cell exists in the grid.
            // Sizing of the panel buffers mirrors this condition in
            // `StructuredSchur::new` — change together.
            let mut ycol = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
            for &b in cols {
                let b = b as usize;
                for local in 0..qc {
                    ycol[local] = coupling[coup + local * e + b];
                }
                glmm_block_solve(&core_blocks[cb..cb + qc * qc], qc, &mut ycol[..qc]);
                // y = A_f⁻¹ C_f[:,b]; S[a][b] −= Σ_local C_f[local][a]·y[local],
                // lower triangle only (a ≥ b).
                for &a in cols {
                    let a = a as usize;
                    if a < b {
                        continue;
                    }
                    let mut acc = 0.0;
                    for local in 0..qc {
                        acc += coupling[coup + local * e + a] * ycol[local];
                    }
                    schur_blk[a * e + b] -= acc;
                }
            }
            continue;
        };
        let cpan = &mut c_panel[..qc * e_f];
        for local in 0..qc {
            let crow = &coupling[coup + local * e..];
            for (dst, &b) in cpan[local * e_f..(local + 1) * e_f].iter_mut().zip(cols) {
                *dst = crow[b as usize];
            }
        }
        let ypan = &mut y_panel[..qc * e_f];
        ypan.copy_from_slice(cpan);
        glmm_block_solve_panel(&core_blocks[cb..cb + qc * qc], qc, ypan, e_f);
        // dd = C_f'·Y (e_f×e_f lower, col-major). A row-major qc×e_f buffer
        // viewed col-major e_f×qc IS its transpose — no copy for either side.
        let dd = &mut dd_temp[..e_f * e_f];
        let ct = MatRef::from_column_major_slice(cpan, e_f, qc);
        let yv = MatRef::from_column_major_slice(ypan, e_f, qc).transpose();
        faer::linalg::matmul::triangular::matmul(
            MatMut::from_column_major_slice_mut(dd, e_f, e_f),
            faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
            faer::Accum::Replace,
            ct,
            faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            yv,
            faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        for (bj, &b) in cols.iter().enumerate() {
            let b = b as usize;
            for (&a, &d) in cols[bj..].iter().zip(&dd[bj * e_f + bj..(bj + 1) * e_f]) {
                schur_blk[a as usize * e + b] -= d;
            }
        }
    }
    if e > 0 {
        // Route on the SAME condition structured_ainv_solve uses: the both-paths
        // cross-check runs both at one θ, so `force_dense` must reach the dense arm
        // even when `ss` is Some (the production workspace always caches it).
        match ss {
            Some(ss) if !force_dense => {
                // Gather the dense schur_blk lower triangle into S's fixed CSC
                // pattern. schur_blk is row-major e×e (lower tri at [a·e+b], a ≥ b);
                // axx is CSC, so column b's stored rows a are exactly the a ≥ b we
                // read.
                {
                    let (sym, vals) = ss.axx.parts_mut();
                    let col_ptr = sym.col_ptr();
                    let row_idx = sym.row_idx();
                    for b in 0..e {
                        for slot in col_ptr[b]..col_ptr[b + 1] {
                            let a = row_idx[slot];
                            vals[slot] = schur_blk[a * e + b];
                        }
                    }
                }
                // Numeric sparse LLT into the cached buffer; non-PD ⇒ None (⇒ NaN dev).
                let llt = ss
                    .symbolic
                    .factorize_numeric_llt(
                        &mut ss.l_values,
                        ss.axx.as_ref(),
                        Side::Lower,
                        LltRegularization::default(),
                        Par::Seq,
                        MemStack::new(&mut ss.fac_mem),
                        Spec::default(),
                    )
                    .ok()?;
                let _ = llt; // ends the &'out borrow on l_values (LltRef is Copy; NLL)
                             // logdet_llt returns 2·Σ ln L_ii = log|S|; this fn's `logdet` is the
                             // ½·log convention (deviance.rs multiplies by 2). So add HALF.
                             // Non-finite (non-PD diagonal) ⇒ None, matching the dense non-PD
                             // sentinel.
                let log_s = logdet_llt(&ss.symbolic, &ss.l_values);
                if !log_s.is_finite() {
                    return None;
                }
                logdet += 0.5 * log_s;
            }
            _ => {
                // Dense fallback: the old Crout factor (test cross-check / defensive).
                if !glmm_block_chol(&mut schur_blk[..e * e], e) {
                    return None;
                }
                for b in 0..e {
                    logdet += schur_blk[b * e + b].ln();
                }
            }
        }
    }
    Some(logdet)
}

/// Apply `A⁻¹` to a packed RHS in place, using the factors `structured_factor`
/// left in `core_blocks`/`schur_blk` and the coupling `C_f`. `a_rhs` arrives
/// packed `[g_core in (f,local) order | g_e]` (core part `s·q_core` then crossed
/// `e`) and returns `[u_core | u_e]`: `t_f = A_f⁻¹ g_{core,f}`,
/// `u_e = S⁻¹(g_e − Σ_f C_f' t_f)`, `u_{core,f} = t_f − A_f⁻¹(C_f u_e)`. `e=0`
/// (nested only) stops after `t_f`. Both loops walk cluster `f`'s NONZERO crossed
/// columns via `coup_cols[coup_ptr[f]..coup_ptr[f+1]]` instead of the full dense
/// `0..e` range — every skipped column of `C_f` is exactly 0.0, so its dense
/// contribution was an exact-zero add/subtract and skipping it is bit-identical
/// (same argument as `structured_factor`'s doc comment). Shared by the structured
/// PIRLS solve and the inference Schur fill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_ainv_solve(
    g: &crate::lmm::LmmGroupings,
    core_blocks: &[f64],
    coupling: &[f64],
    schur_blk: &[f64],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    ss: Option<&mut StructuredSchur>,
    force_dense: bool,
    a_rhs: &mut [f64],
) {
    use crate::lmm::MAX_PRIMARY_Q;
    let qc = g.primary_q + g.nested_per_parent;
    let s = g.n_primary;
    let e = g.k_crossed();
    let k_family = qc * s;
    // t_f = A_f⁻¹ g_{core,f} (overwrites a_rhs core); g_e −= Σ_f C_f' t_f.
    for f in 0..s {
        let cb = f * qc * qc;
        let gcb = f * qc;
        let coup = f * qc * e;
        glmm_block_solve(
            &core_blocks[cb..cb + qc * qc],
            qc,
            &mut a_rhs[gcb..gcb + qc],
        );
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        for &b in cols {
            let b = b as usize;
            let mut acc = 0.0;
            for local in 0..qc {
                acc += coupling[coup + local * e + b] * a_rhs[gcb + local];
            }
            a_rhs[k_family + b] -= acc;
        }
    }
    if e == 0 {
        return;
    }
    // Route on the SAME condition as structured_factor: force_dense's dense factor
    // left schur_blk holding the dense L, so the solve must reach the dense arm too
    // even when ss is Some (the both-paths cross-check runs both at one θ).
    match ss {
        Some(ss) if !force_dense => {
            // Reconstruct the factor from the cached symbolic + values (no re-factor;
            // faer 0.24.4 LltRef::new — sparse/linalg/cholesky.rs:4443-4449, verified
            // against the vendored source: a Copy wrapper over two refs, NOT a
            // re-factorization — and back-solve the single e-col.
            let llt = LltRef::new(&ss.symbolic, &ss.l_values);
            let rhs = MatMut::from_column_major_slice_mut(&mut a_rhs[k_family..k_family + e], e, 1);
            llt.solve_in_place_with_conj(Conj::No, rhs, Par::Seq, MemStack::new(&mut ss.solve_mem));
        }
        _ => {
            // Dense fallback (test cross-check / e>0 with no cached factor): schur_blk
            // holds the dense L that the dense-factor branch produced.
            glmm_block_solve(&schur_blk[..e * e], e, &mut a_rhs[k_family..k_family + e]);
        }
    }
    // u_{core,f} = t_f − A_f⁻¹(C_f u_e).
    for f in 0..s {
        let cb = f * qc * qc;
        let gcb = f * qc;
        let coup = f * qc * e;
        let mut v = [0.0_f64; MAX_PRIMARY_Q];
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        #[allow(clippy::needless_range_loop)]
        for local in 0..qc {
            let mut acc = 0.0;
            for &b in cols {
                let b = b as usize;
                acc += coupling[coup + local * e + b] * a_rhs[k_family + b];
            }
            v[local] = acc;
        }
        glmm_block_solve(&core_blocks[cb..cb + qc * qc], qc, &mut v[..qc]);
        for local in 0..qc {
            a_rhs[gcb + local] -= v[local];
        }
    }
}

/// Per-cluster crossed-column pattern (CSR over f): the union of cluster f's
/// rows' cross_col entries — exactly the nonzero-column support of C_f.
/// Counting-sort CSR: counts → prefix → fill (coup_ptr doubles as the write
/// cursors) → shift cursors back → per-cluster sort + dedup-compact (a
/// cluster's rows repeat the same crossed level; a duplicate in the list
/// would double-subtract in the Schur build). e = 0 (nested only) degenerates
/// to an all-empty CSR — n_cross is all zero.
///
/// The pattern is a function of the design AND the θ-pinning mask
/// (`build_packed_m` drops θ=0 crossed groupings from `cross_col`/`n_cross`),
/// so it is fit-invariant only while the pinning mask is: the caller
/// (deviance.rs structured branch) caches it keyed on that mask and rebuilds
/// on transitions — not per eval, not blindly per fit.
pub(crate) fn build_coupling_csr(
    cluster_ids: &[u32],
    cross_col: &[u32],
    n_cross: &[u8],
    s: usize,
    n: usize,
    coup_cols: &mut [u32],
    coup_ptr: &mut [u32],
) {
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    for v in coup_ptr[..s + 1].iter_mut() {
        *v = 0;
    }
    for i in 0..n {
        coup_ptr[cluster_ids[i] as usize + 1] += n_cross[i] as u32;
    }
    for f in 0..s {
        coup_ptr[f + 1] += coup_ptr[f];
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let cbase = i * g_cap;
        for z in 0..n_cross[i] as usize {
            coup_cols[coup_ptr[f] as usize] = cross_col[cbase + z];
            coup_ptr[f] += 1;
        }
    }
    for f in (1..=s).rev() {
        coup_ptr[f] = coup_ptr[f - 1];
    }
    coup_ptr[0] = 0;
    {
        let mut write = 0usize;
        let mut start = 0usize;
        for f in 0..s {
            let end = coup_ptr[f + 1] as usize;
            coup_cols[start..end].sort_unstable();
            coup_ptr[f] = write as u32;
            let mut prev = u32::MAX; // crossed indices are < e ≪ u32::MAX
            for idx in start..end {
                let v = coup_cols[idx];
                if v != prev {
                    coup_cols[write] = v;
                    write += 1;
                    prev = v;
                }
            }
            start = end;
        }
        coup_ptr[s] = write as u32;
    }
}

/// Structured PIRLS for the intercept-only crossed/nested regime
/// (`groupings.structured_extras_eligible()`): `A = M'WM + I` is
/// `[[D, C], [C', E]]` where `D` is block-diagonal over the primary clusters
/// (each `q_core×q_core` core block = the primary RE column + its
/// nested-within-primary children) and the only dense coupling `C` is to the thin
/// crossed width `e`. A per-row scatter (each row touches ONE primary cluster's
/// core columns + one crossed level per crossed grouping — `M = ZΛ` sparsity, NOT
/// row contiguity, so this is layout-independent) assembles the core blocks, the
/// coupling `C_f`, the crossed `E`, and the RHS `g = M'(W·Mu + (y−p))`; then a
/// Schur complement on `e` solves it: factor each `A_f = chol(D_f+I)`, form
/// `S = (E+I) − Σ_f C_f' A_f⁻¹ C_f`, back-substitute `u_e = S⁻¹(g_e − Σ_f C_f'
/// A_f⁻¹ g_{core,f})` then `u_{core,f} = A_f⁻¹ g_{core,f} − A_f⁻¹ C_f u_e`, and
/// `log|A| = Σ_f log|A_f| + log|S|` (Schur determinant identity). `e=0` (nested
/// only) skips the Schur entirely. Collapses the dense `O(n·k²)` Gram + `O(k³)`
/// factor to `O(n·q_core²)` scatter + `O(s·q_core³ + e³)` factor. NOT bit-identical
/// to `pirls_solve` (scatter vs GEMM accumulation) but the same estimator.
/// The `M = ZΛ` nonzeros arrive PACKED (`m_core_buf` core slice + `cross_*`/
/// `n_cross` crossed entries the caller filled via `build_packed_m`), never the
/// dense faer `m`. Leaves
/// `core_blocks` (per-cluster L) + `schur_blk` (Schur L) + `coupling` FACTORED for
/// `structured_schur_fill` to reuse, and eta/prob/w/u filled. Returns
/// `(dev, ‖u‖², log|A|, converged)`; a non-PD core block / Schur ⇒
/// `(NaN, NaN, NaN, false)`. Iterates from the caller-provided `u` (warm-start
/// seed); the caller owns resetting it per fit. `a_rhs` (length ≥ k_total) is the
/// RHS scratch, packed `[g_core in (f,local) order | g_e]`.
///
/// Step-halving: see `pirls_solve`'s doc — identical mechanism, scoped here to
/// `u_prev` only (β is FIXED in this variant). A halved retry re-enters the top
/// and re-runs the full structured factor + Schur (`structured_factor` /
/// `structured_ainv_solve`) on the backtracked `u`; `core_blocks` / `schur_blk` /
/// `coupling` and `log|A|` on return are from the final step's assembly,
/// preserving the `structured_schur_fill` contract above (it reads those
/// buffers).
///
/// **β mode (`beta_step`):** `Fixed` holds β at the caller's input (β read-only,
/// FD-Hessian / stage-2 contract). `Profile` adds a joint δβ Schur-border step
/// each iteration — run AFTER the structured factor + `A⁻¹` u-solve, mirroring
/// `se::structured_schur_fill` with the live iteration's W and structured factors:
/// `T = A⁻¹B` via `p` `structured_ainv_solve` calls (one per β column — the design
/// doc's stated per-eval cost rise), `S_β = C − B'T`, `δβ = S_β⁻¹·(X'ρ − B'δu₀)`,
/// then `u_joint = u_new − T·δβ` and `β += δβ` written back through `beta`. A
/// non-PD S_β surfaces as `(NaN, NaN, NaN, false)`. `B`'s scatter reads the packed
/// `m_core_buf`/`cross_*` nonzeros directly (as `structured_schur_fill` does).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pirls_solve_blocked_extras(
    family: Family,
    nb_theta: f64,
    g: &crate::lmm::LmmGroupings,
    cluster_ids: &[u32],
    m_core_buf: &[f64],
    cross_val: &[f64],
    cross_col: &[u32],
    n_cross: &[u8],
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
    core_blocks: &mut [f64],
    coupling: &mut [f64],
    schur_blk: &mut [f64],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    mut structured_schur: Option<&mut StructuredSchur>,
    force_dense: bool,
    a_rhs: &mut [f64],
    // n × p = W∘X GEMM scratch for the Profile β-Schur border's C = X'WX
    // (mirrors `pirls_solve`'s `wx`).
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`FitOptions::offset`), added into
    // `eta_fixed` below and by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
) -> (f64, f64, f64, bool) {
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let q = g.primary_q;
    let np = g.nested_per_parent;
    let qc = q + np; // core-block width; ≤ MAX_PRIMARY_Q by eligibility
    let s = g.n_primary;
    let prim_width = q * s;
    let k_family = qc * s; // = prim_width + s·np
    let e = g.k_crossed(); // crossed width; 0 ⇒ no Schur
    let k = k_family + e; // = g.k_total
    let p = beta.len();
    // The j-th core-block-local column of cluster f maps to RE column:
    // local < q ⇒ primary component (f·q + local); else ⇒ nested child
    // (prim_width + f·np + (local−q)). Single source for the gather/scatter.
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    for i in 0..n {
        let mut ef = 0.0;
        for j in 0..p {
            ef += x[(i, j)] * beta[j];
        }
        eta_fixed[i] = ef;
    }
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += o[i];
        }
    }
    // First-trial backtrack seeds (u_prev = 0, beta_prev = caller's β) — only
    // the domain-infeasibility trigger can read them; see `pirls_solve`.
    u_prev[..k].fill(0.0);
    if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
        beta_prev[..p].copy_from_slice(&beta[..p]);
    }
    let mut pen_accepted = f64::INFINITY; // same-point penalized deviance at the last ACCEPTED iterate
    let mut mixed_prev = f64::INFINITY; // today's mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` from the previous step
    let mut halvings = 0usize;
    let mut converged = false;
    let (mut dev, mut pen, mut logdet) = (f64::NAN, f64::NAN, 0.0);
    let tol = pirls_tol_override.unwrap_or_else(|| super::pirls_tol(family));
    for _ in 0..PIRLS_MAX_ITERS {
        // --- trial evaluation at the CURRENT u (pass 1 + pass 2). On a fresh accept
        // this is the newly-stepped u; after a halving `continue` it is the
        // backtracked u. Either way the recompute IS the trial evaluation. ---
        // --- pass 1: η-pass — ηᵢ = η_fixed,ᵢ + (Mu)ᵢ over the row's nonzeros ---
        // Reads the packed M nonzeros (contiguous q_core core slice + n_cross[i]
        // crossed entries) `build_packed_m` filled — no faer indexing, crossed term
        // O(G) not O(e).
        let mut yeta = 0.0;
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let m_core = &m_core_buf[i * qc..i * qc + qc];
            let mut mui = 0.0;
            for local in 0..qc {
                mui += m_core[local] * u[core_col(f, local)];
            }
            let cbase = i * g_cap;
            for z in 0..n_cross[i] as usize {
                let b = cross_col[cbase + z] as usize;
                mui += cross_val[cbase + z] * u[k_family + b];
            }
            eta[i] = eta_fixed[i] + mui;
            mu[i] = mui; // keep (Mu)ᵢ for the IRLS residual below
            yeta += y[i] * eta[i];
        }
        // --- pass 2: η[] → prob[]/w[] + deviance, through the shared family
        // kernel (clamps η in place; `infeasible` flags any raw η outside the
        // link's open domain — Gamma-inverse only, mirrors `pirls_solve`). ---
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
        // Retrospective step-halving (lme4 `pwrssUpdate`, mirrors `pirls_solve` /
        // `pirls_solve_blocked`): convergence band checked BEFORE the overshoot test
        // (near the optimum Fisher scoring is not strictly monotone — a step can
        // land ε above `pen_accepted` yet inside the tol band, and that must
        // converge, not burn all 10 halvings against FP noise). ‖u‖² is at the
        // CURRENT trial u; `u[..k]` spans the core (RE-column order) + crossed tail.
        let pen_u: f64 = u[..k].iter().map(|v| v * v).sum();
        let penalized = dev + pen_u;
        // BAND-TOLERANT overshoot test, mirrors `pirls_solve` (see its comments for
        // why a within-band rise is accepted rather than converged-on or halved):
        // only a rise EXCEEDING the tol band backtracks. A domain-infeasible trial
        // halves regardless of the band (see `pirls_solve`'s comment).
        if infeasible || penalized - pen_accepted > tol * (1.0 + penalized.abs()) {
            if halvings < PIRLS_MAX_HALVINGS {
                // Last full step overshot: halve δu = u − u_prev and re-enter the top
                // (the recompute above is the trial evaluation of the halved step; a
                // halved retry re-runs structured_factor/ainv_solve, by design).
                halvings += 1;
                for c in 0..k {
                    u[c] = 0.5 * (u[c] + u_prev[c]);
                }
                // Profile mode: the trial point is the JOINT (u,β) step, so the
                // backtrack halves β toward `beta_prev` in lockstep with u, then
                // refreshes η_fixed for the re-evaluation at the top. Mirrors
                // `pirls_solve_blocked`'s Profile backtrack.
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
        // Accept this iterate, snapshot it for the next backtrack, and take a fresh
        // full Fisher step from it (cold start: pen_accepted = ∞ ⇒ always accepts).
        halvings = 0;
        pen_accepted = penalized;
        u_prev[..k].copy_from_slice(&u[..k]);
        // Profile mode: snapshot the accepted β as the β-halving twin of u_prev.
        if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
            beta_prev[..p].copy_from_slice(&beta[..p]);
        }
        // IRLS effective residual rᵢ = wᵢ·(Mu)ᵢ + W·working_resid, so the
        // scattered RHS is M'(W·Mu + W·r) = (A−I)u + M'(W·r). Logit's W·r=(y−p);
        // the general branch's is (dμ/dη)·(y−μ)/V.
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
        // --- pass 3: scatter — wᵢmᵢmᵢ' into D_f/C_f/E (lower tri), rᵢmᵢ into g ---
        for v in core_blocks[..s * qc * qc].iter_mut() {
            *v = 0.0;
        }
        // coupling is only ever WRITTEN by this scatter (structured_factor and
        // structured_ainv_solve treat it as read-only), so zeroing exactly the
        // coup_cols/coup_ptr pattern the scatter can write is sufficient — nothing
        // else ever touches an out-of-pattern coupling entry between passes.
        for f in 0..s {
            let coup = f * qc * e;
            let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
            for &b in cols {
                let b = b as usize;
                for r in 0..qc {
                    coupling[coup + r * e + b] = 0.0;
                }
            }
        }
        // schur_blk stays a dense clear (unlike coupling above): the dense-fallback
        // branch of structured_factor (ss.is_none() or force_dense) Cholesky-factors
        // it IN PLACE, producing fill-in outside the coup_cols×coup_cols pattern — a
        // sparse zero would leave stale L-factor residue from the prior iteration.
        for v in schur_blk[..e * e].iter_mut() {
            *v = 0.0;
        }
        for v in a_rhs[..k].iter_mut() {
            *v = 0.0;
        }
        // Reads the packed core slice + crossed nonzeros directly — no per-row stack
        // copy / rescan; the `m_core`/`cz_*` rebuild the dense path needed is gone.
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let wi = w[i];
            let ri = mu[i]; // effective residual
            let m_core = &m_core_buf[i * qc..i * qc + qc];
            let cbase = i * g_cap;
            let ncz = n_cross[i] as usize;
            let cb = f * qc * qc;
            let gcb = f * qc;
            let coup = f * qc * e;
            for r in 0..qc {
                let mr = m_core[r];
                a_rhs[gcb + r] += mr * ri;
                let wmr = wi * mr;
                for c in 0..=r {
                    core_blocks[cb + r * qc + c] += wmr * m_core[c];
                }
                for z in 0..ncz {
                    coupling[coup + r * e + cross_col[cbase + z] as usize] +=
                        wmr * cross_val[cbase + z];
                }
            }
            for z in 0..ncz {
                let b = cross_col[cbase + z] as usize;
                let vb = cross_val[cbase + z];
                a_rhs[k_family + b] += vb * ri;
                let wvb = wi * vb;
                for z2 in 0..ncz {
                    let b2 = cross_col[cbase + z2] as usize;
                    if b2 <= b {
                        schur_blk[b * e + b2] += wvb * cross_val[cbase + z2];
                    }
                }
            }
        }
        // Profile mode: accumulate the β-gradient X'ρ (ρ = effective residual) into
        // `beta_rhs` — the joint system's bottom-block RHS. A dedicated pass off the
        // fresh prob/eta/w (NOT folded into the scatter loop above, which overwrote
        // `mu` with the IRLS working vector), so the Fixed path stays byte-identical.
        // Mirrors `pirls_solve_blocked`'s Profile X'ρ fold.
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
        // --- +I ridge (per RE column) on the core diagonals and the E diagonal ---
        for f in 0..s {
            let cb = f * qc * qc;
            for r in 0..qc {
                core_blocks[cb + r * qc + r] += 1.0;
            }
        }
        for b in 0..e {
            schur_blk[b * e + b] += 1.0;
        }
        // --- factor (core blocks + Schur), then apply A⁻¹ to the scattered RHS ---
        logdet = match structured_factor(
            g,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            structured_schur.as_deref_mut(),
            force_dense,
        ) {
            Some(ld) => ld,
            None => return (f64::NAN, f64::NAN, f64::NAN, false),
        };
        structured_ainv_solve(
            g,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            structured_schur.as_deref_mut(),
            force_dense,
            a_rhs,
        );
        // a_rhs[gcb..] now holds u_{core,f}; a_rhs[k_family+b] holds u_e. Scatter to
        // u (RE-column order) and accumulate ‖u‖².
        pen = 0.0;
        for f in 0..s {
            let gcb = f * qc;
            for local in 0..qc {
                let val = a_rhs[gcb + local];
                u[core_col(f, local)] = val;
                pen += val * val;
            }
        }
        for b in 0..e {
            let val = a_rhs[k_family + b];
            u[k_family + b] = val;
            pen += val * val;
        }
        // --- Profile-mode joint δβ step (β-Schur border), run AFTER the structured
        // factor + A⁻¹ u-solve so every core-block / Schur factor is live in
        // `core_blocks`/`schur_blk`/`coupling` and δu₀ = u_new − u_prev is complete
        // (u holds u_new, u_prev the pre-step iterate). Mirrors `se::structured_schur_fill`
        // with THIS iteration's W and factors: T = A⁻¹B (p `structured_ainv_solve`
        // calls, the design's stated per-eval cost rise), S_β = C − B'T,
        // δβ = S_β⁻¹·(X'ρ − B'δu₀), then u_joint = u_new − T·δβ (see
        // `se::dense_schur_fill`'s doc comment for the shared β-Schur Newton step
        // this mirrors). `a_rhs` is reused as the per-column A⁻¹ in/out scratch —
        // safe now, its u-solve was scattered to `u` just above (the borrow note). ---
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
            // C = X'WX (p×p) via the W∘X GEMM scratch `wx` — structured_schur_fill's
            // X'W̃X block, same product. GEMM fills the full p×p (the old scalar
            // loop only filled+mirrored the lower triangle); every downstream read
            // below is over the full matrix (`schur[(r,c)]` for `c in 0..p`), so
            // this is exact.
            for c in 0..p {
                for i in 0..n {
                    wx[(i, c)] = w[i] * x[(i, c)];
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
            // B' = X'WM (p×k): zero, then scatter each row's core + crossed columns
            // from the packed `m_core_buf`/`cross_*` nonzeros (structured_schur_fill:436-456).
            for r in 0..p {
                for c in 0..k {
                    xtwm[(r, c)] = 0.0;
                }
            }
            for i in 0..n {
                let f = cluster_ids[i] as usize;
                let wi = w[i];
                let cbase = i * g_cap;
                let ncz = n_cross[i] as usize;
                for r in 0..p {
                    let xw = x[(i, r)] * wi;
                    for local in 0..qc {
                        xtwm[(r, core_col(f, local))] += xw * m_core_buf[i * qc + local];
                    }
                    for z in 0..ncz {
                        let b = cross_col[cbase + z] as usize;
                        xtwm[(r, k_family + b)] += xw * cross_val[cbase + z];
                    }
                }
            }
            // T = A⁻¹B: ainv_mtwx[:, c] = A⁻¹(M'WX)[:, c], one β column at a time via
            // `structured_ainv_solve` reusing THIS iteration's core-block+Schur factors
            // (structured_schur_fill:457-486). `a_rhs` packs/unpacks the RHS/solution in
            // the (f,local)|crossed layout structured_ainv_solve expects.
            for c in 0..p {
                for f in 0..s {
                    for local in 0..qc {
                        a_rhs[f * qc + local] = xtwm[(c, core_col(f, local))];
                    }
                }
                for b in 0..e {
                    a_rhs[k_family + b] = xtwm[(c, k_family + b)];
                }
                structured_ainv_solve(
                    g,
                    core_blocks,
                    coupling,
                    schur_blk,
                    coup_cols,
                    coup_ptr,
                    structured_schur.as_deref_mut(),
                    force_dense,
                    a_rhs,
                );
                for f in 0..s {
                    for local in 0..qc {
                        ainv_mtwx[(core_col(f, local), c)] = a_rhs[f * qc + local];
                    }
                }
                for b in 0..e {
                    ainv_mtwx[(k_family + b, c)] = a_rhs[k_family + b];
                }
            }
            // S_β = C − B'·T (structured_schur_fill:487-497). Every RE column belongs
            // to a core block or the crossed tail and is populated, so Σ_j over k is exact.
            for r in 0..p {
                for c in 0..p {
                    let mut sm = xtwx[(r, c)];
                    for j in 0..k {
                        sm -= xtwm[(r, j)] * ainv_mtwx[(j, c)];
                    }
                    schur[(r, c)] = sm;
                }
            }
            // rhs = X'ρ − B'·δu₀ (beta_rhs holds X'ρ; δu₀ = u_new − u_prev, both in
            // the u RE-column|crossed layout).
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
            // Apply: β += δβ; u = u_joint = u_new − T·δβ, i.e.
            // u[col] −= Σ_j ainv_mtwx[(col, j)]·δβ[j] over the u layout.
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
            // η_fixed depends on β; refresh for the next trial. `pen` must track the
            // moved u (‖u_joint‖²), so recompute it.
            refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
            pen = 0.0;
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                pen += u[c] * u[c];
            }
        }
        // Today's stopping rule, verbatim: the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` band on
        // successive steps — bit-identical iterate path and returned values to the
        // pre-halving loop when no halving fires (see `pirls_solve` for why the
        // same-point band above cannot itself be a converge trigger).
        let mixed = dev + pen;
        if (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
            converged = true;
            break;
        }
        mixed_prev = mixed;
    }
    (dev, pen, logdet, converged)
}
