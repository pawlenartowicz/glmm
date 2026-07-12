use faer::{Mat, MatRef};

use super::deviance::laplace_deviance_at;
use super::pirls::structured_ainv_solve;
use super::workspace::{fill_z_f64, glmm_block_solve, GlmmWorkspace};
use super::{FdHessianStatus, FD_STEP_REL};

/// Evaluate the joint Laplace deviance at `fd_saved + Σ deltaₖ·e_{coordₖ}`,
/// reusing `ws.fd_saved` (distinct field from `ws.params`, so the disjoint
/// field borrows are legal). `coords`/`deltas` are ≤ 2 long (a diagonal or a
/// mixed partial). Leaves `ws.params` perturbed — callers restore from
/// `ws.fd_saved` between the directional evals via this same write.
fn fd_eval(
    ws: &mut GlmmWorkspace,
    coords: &[usize],
    deltas: &[f64],
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    let m = ws.fd_saved.len();
    ws.params[..m].copy_from_slice(&ws.fd_saved[..m]);
    for (&c, &d) in coords.iter().zip(deltas) {
        ws.params[c] += d;
    }
    // β-FIXED by construction: `laplace_deviance_at` hardcodes `profile_beta =
    // false`. The FD-Hessian differentiates a function of the
    // caller's β, so β must NOT move under these directional evals — a Profile step
    // here would make each `f(γ)` depend on the profiled β̂(γ) and corrupt the
    // second differences.
    laplace_deviance_at(ws, x, y, cluster_ids, n)
}

/// Central second difference of coordinate `k` at step `s`: `(f(+s) − 2·f0 +
/// f(−s))/s²`. Returns the raw value — non-finite if either directional eval
/// diverges — so the caller decides fallback (serial: on the first bad cell;
/// parallel grid: after the whole grid). Extracted from the former `second_diff!`
/// macro so a per-thread worker workspace can call it.
#[allow(clippy::too_many_arguments)]
fn second_diff(
    ws: &mut GlmmWorkspace,
    k: usize,
    s: f64,
    f0: f64,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    let fp = fd_eval(ws, &[k], &[s], x, y, cluster_ids, n);
    let fm = fd_eval(ws, &[k], &[-s], x, y, cluster_ids, n);
    (fp - 2.0 * f0 + fm) / (s * s)
}

/// Symmetric 4-point mixed partial of `(i, j)` at steps `(si, sj)`:
/// `(f(+si,+sj) − f(+si,−sj) − f(−si,+sj) + f(−si,−sj))/(4·si·sj)`. Returns the raw
/// value (non-finite if any of the four evals diverges); fallback is the caller's,
/// as for `second_diff`.
#[allow(clippy::too_many_arguments)]
fn mixed_diff(
    ws: &mut GlmmWorkspace,
    i: usize,
    j: usize,
    si: f64,
    sj: f64,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
) -> f64 {
    let fpp = fd_eval(ws, &[i, j], &[si, sj], x, y, cluster_ids, n);
    let fpm = fd_eval(ws, &[i, j], &[si, -sj], x, y, cluster_ids, n);
    let fmp = fd_eval(ws, &[i, j], &[-si, sj], x, y, cluster_ids, n);
    let fmm = fd_eval(ws, &[i, j], &[-si, -sj], x, y, cluster_ids, n);
    (fpp - fpm - fmp + fmm) / (4.0 * si * sj)
}

/// Fill `out_cov` (p×p) with the RX/Schur fixed-effect covariance `inv(ws.schur)`
/// — `ws.schur` is the β-INFORMATION matrix, so the inverse is the covariance
/// directly (NO factor of 2; that factor only applies to the deviance Hessian,
/// where info = H_dev/2). Reuses `fit_glmm`'s inference-block Schur-fill dispatch
/// (`blocked`/`structured`/`dense`), so it requires `ws.{w, lam, a_blocks, …}` to
/// hold the factors a converged PIRLS at the current `ws.params` left behind.
/// Returns false on a non-PD Schur. Shared by the `fd_hessian_cov` fallback and
/// (later) the Rx production path.
pub(crate) fn rx_cov_into(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> bool {
    use faer::linalg::solvers::Solve;
    let inf_ok = if ws.groupings.extra_offsets.is_empty() {
        blocked_schur_fill(ws, x, cluster_ids, n)
    } else if ws.groupings.structured_extras_eligible() {
        structured_schur_fill(ws, x, cluster_ids, n)
    } else {
        dense_schur_fill(ws, x, n)
    };
    if !inf_ok {
        return false;
    }
    let chol = match ws.schur.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let mut inv = Mat::<f64>::identity(p, p);
    chol.solve_in_place(inv.as_mut());
    for a in 0..p {
        for b in 0..p {
            out_cov[(a, b)] = inv[(a, b)];
        }
    }
    true
}

/// Finite-difference Hessian of the Laplace deviance over the joint (θ,β) at the
/// converged point in `ws.params`; inverts and writes the p×p fixed-effect
/// covariance block into `out_cov`. On a non-PD joint Hessian (or a non-finite
/// perturbed deviance — the few-cluster failure mode) writes the RX/Schur
/// covariance instead and returns the fallback status. Restores `ws.params` on
/// return. Matches glmer `vcov(use.hessian = TRUE)` (factor of 2: deviance =
/// −2logL, so observed info = H_dev/2 and cov = info⁻¹ = 2·H_dev⁻¹).
///
/// FD scheme (tuned against `tests/fixtures/glmm_hessian_vcov.json`, the n=96 /
/// 12-cluster `y ~ x1 + (1|grp)` glmer fit): single-step central second
/// differences at `h_k = FD_STEP_REL·max(1, |γ̂_k|)`. No Richardson extrapolation
/// (dropped 2026-07-04 — the deviance is step-invariant to ~7 sig figs across
/// h ∈ [1e-4, 1e-1] on this fixture, so a second-order correction bought no
/// measurable accuracy). Every eval here runs PIRLS at the tight `PIRLS_TOL_REL_FD` (via
/// `ws.pirls_tol_override`, set on entry / reset on every exit): at the
/// canonical fit tol (1e-6) the FD was NOT step-invariant on cbpp (~0.3%
/// wobble, h=1e-2 vs 1e-3 — the shipped step happened to land on the accurate
/// side); at `PIRLS_TOL_REL_FD` it is step-invariant to ~5–6 sig figs across
/// h ∈ [1e-4, 1e-2] and sits on the tight-tol (1e-12) limit, so the accuracy is
/// by construction, not by luck. The fit path never sees the tight tol.
/// One accepted divergence: on the RX fallback below, the central re-eval that
/// repopulates the Schur factors also runs at the tight tol, so fallback RX
/// numbers differ from the plain `WaldSe::Rx` arm's at tolerance level
/// (slightly MORE converged — harmless).
///
/// Match vs lme4 `vcov(use.hessian=TRUE)`: ~3.4e-7 worst per-entry gap on the
/// committed fixture and ≤2e-5 rel `se_hessian` on every parity rung — but ONLY
/// against an lme4 run at tightened `tolPwrss` (the fixture and the frozen
/// parity references are generated at 1e-13; each records it). At lme4's
/// DEFAULT `tolPwrss = 1e-7` a ~1% gap opens, and it is LME4'S, not ours
/// (measured 2026-07-04):
/// glmer assembles the `log|A|` (ldL2) term of its Laplace deviance from
/// `pp$Xwts` — working weights ONE PIRLS ITERATION BEHIND the converged mode —
/// so its default devfun sits a smooth ~5.6e-4 above the true Laplace deviance
/// (cbpp) and carries ~1% spurious θ/θβ curvature; its shipped vcov and
/// numDeriv agree with each other because both differentiate that same lagged
/// function. Our `log|A|` uses fully-converged weights — for canonical links
/// W = μ(1−μ) at û is the exact η-Hessian, i.e. the textbook Laplace term.
/// Supporting facts: H_ββ is exact (`rx_cov_into` matches lme4
/// `vcov(use.hessian=FALSE)` to ~3.6e-6 method-matched), and the true θ↔β
/// correction RAISES cbpp SEs above RX — as ours does; lme4's default-tol value
/// lowers them, a sign artifact of the lagged weights.
///
/// `m = ws.params.len() = n_theta + p`; the β block is rows/cols `n_theta..m`.
/// Precondition: `ws` is at a CONVERGED fit and `ws.z_buf`-eligible scratch is
/// valid for (x, ids, n) (the deviance evals re-solve PIRLS).
pub fn fd_hessian_cov(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> FdHessianStatus {
    use faer::linalg::solvers::Solve;
    let m = ws.params.len();
    let n_theta = ws.n_theta;

    // Every deviance eval below (f0, the f± stencil, and the fallback's central
    // re-eval) converges PIRLS at the FD-only tight tol — see the doc comment.
    // Reset on every exit alongside `warm_seed_active`.
    ws.pirls_tol_override = Some(super::PIRLS_TOL_REL_FD);

    // Snapshot γ̂ and the per-coordinate FD step; fill z_buf once (blocked path).
    ws.fd_saved[..m].copy_from_slice(&ws.params[..m]);
    for k in 0..m {
        ws.fd_steps[k] = FD_STEP_REL * ws.fd_saved[k].abs().max(1.0);
    }
    if ws.groupings.extra_offsets.is_empty() {
        let GlmmWorkspace {
            groupings, z_buf, ..
        } = &mut *ws;
        fill_z_f64(groupings, x, z_buf, n);
    }

    // Restore γ̂ and take the RX/Schur fallback: re-eval the central deviance to
    // repopulate W̃/Λ̂/block factors at γ̂, then invert the β information.
    macro_rules! fallback {
        () => {{
            let _ = fd_eval(ws, &[], &[], x, y, cluster_ids, n);
            let ok = rx_cov_into(ws, x, cluster_ids, p, n, out_cov);
            debug_assert!(ok, "RX fallback Schur must be PD at a converged fit");
            // The fallback reports an RX vcov, so it carries Gamma's σ̂² like the
            // production Rx arm (fixed-scale families: ×1). The fd_eval above
            // restored the converged μ̂/û at γ̂.
            if ok {
                let sigma_sq = crate::family::glmm_sigma_sq(
                    ws.family,
                    &y[..n],
                    &ws.prob[..n],
                    &ws.u[..ws.k],
                    ws.weighted.then(|| &ws.prior_w[..n]),
                );
                if sigma_sq != 1.0 {
                    for a in 0..p {
                        for b in 0..p {
                            out_cov[(a, b)] *= sigma_sq;
                        }
                    }
                }
            }
            // Double failure (joint Hessian AND RX Schur both non-PD): rx_cov_into
            // leaves out_cov UNTOUCHED on `false`, so in release it would keep stale
            // data while we still report NonPdFellBackToRx. NaN-fill so the caller
            // (the caller routes this to nan_fit) can detect it via is_nan().
            if !ok {
                for a in 0..p {
                    for b in 0..p {
                        out_cov[(a, b)] = f64::NAN;
                    }
                }
            }
            // No joint Hessian on the RX fallback ⇒ no θ-block SE to report.
            for k in 0..n_theta {
                ws.theta_se[k] = f64::NAN;
            }
            ws.params[..m].copy_from_slice(&ws.fd_saved[..m]);
            ws.warm_seed_active = false; // never leak the FD seed into a later fit / BOBYQA
            ws.pirls_tol_override = None; // nor the FD tight tol
            return FdHessianStatus::NonPdFellBackToRx;
        }};
    }

    let f0 = fd_eval(ws, &[], &[], x, y, cluster_ids, n);
    if !f0.is_finite() {
        fallback!();
    }

    // f0 ran cold (warm_seed_active still false), so ws.u now holds the fitted
    // mode û(γ̂). Snapshot it as the fixed shared FD seed and switch every
    // subsequent perturbed eval to warm-start from it (§2/§5: identical seed for
    // all f± keeps the second differences order-independent; the diagonal mixes a
    // cold f0 with warm f±, but u_seed IS f0's own converged mode so a warm f0
    // would return the same deviance to tol — no systematic offset to amplify).
    let kk = ws.k.max(1);
    ws.u_seed[..kk].copy_from_slice(&ws.u[..kk]);
    ws.warm_seed_active = true;

    // Build the symmetric m×m Hessian into ws.hess_scratch (upper, then mirror).
    // Each grid cell is a pure function of the frozen FD seed (fd_saved, fd_steps,
    // u_seed) — see `fd_hessian_cov`'s doc comment — so per-thread worker
    // workspaces compute bit-identical values in any order. Diagonal cells use a
    // single-step central second difference (no Richardson — the fixture is
    // step-invariant to ~7 sig figs across h ∈ [1e-4, 1e-1], see the doc comment).
    let use_par = cfg!(all(feature = "parallel", not(target_arch = "wasm32"))) && ws.parallel_inner;
    if use_par {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use super::workspace::fd_worker_ws;
            use rayon::prelude::*;
            let cells: Vec<(usize, usize)> =
                (0..m).flat_map(|i| (i..m).map(move |j| (i, j))).collect();
            let ws_ro: &GlmmWorkspace = ws; // shared read-only view for map_init
            let results: Vec<(usize, usize, f64)> = cells
                .par_iter()
                .map_init(
                    || fd_worker_ws(ws_ro, n),
                    |wws, &(i, j)| {
                        let h = if i == j {
                            second_diff(wws, i, ws_ro.fd_steps[i], f0, x, y, cluster_ids, n)
                        } else {
                            mixed_diff(
                                wws,
                                i,
                                j,
                                ws_ro.fd_steps[i],
                                ws_ro.fd_steps[j],
                                x,
                                y,
                                cluster_ids,
                                n,
                            )
                        };
                        (i, j, h)
                    },
                )
                .collect();
            // The serial arm fallback!()s on the FIRST non-finite eval; here the
            // whole grid ran first, then we check — same destination (RX fallback),
            // extra work only on the already-failing path. `results` is collected
            // before this point, ending map_init's immutable borrow of `ws` so the
            // fallback (which needs `&mut ws`) is legal.
            if results.iter().any(|&(_, _, h)| !h.is_finite()) {
                fallback!();
            }
            for (i, j, h) in results {
                ws.hess_scratch[(i, j)] = h;
                ws.hess_scratch[(j, i)] = h;
            }
        }
    } else {
        for i in 0..m {
            let hi = ws.fd_steps[i];
            let hii = second_diff(ws, i, hi, f0, x, y, cluster_ids, n);
            if !hii.is_finite() {
                fallback!();
            }
            ws.hess_scratch[(i, i)] = hii;
            for j in (i + 1)..m {
                let hj = ws.fd_steps[j];
                let hij = mixed_diff(ws, i, j, hi, hj, x, y, cluster_ids, n);
                if !hij.is_finite() {
                    fallback!();
                }
                ws.hess_scratch[(i, j)] = hij;
                ws.hess_scratch[(j, i)] = hij;
            }
        }
    }

    // Invert the joint Hessian; non-PD ⇒ RX fallback. cov = 2·(H⁻¹)_ββ.
    let chol = match ws.hess_scratch.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => fallback!(),
    };
    let mut inv = Mat::<f64>::identity(m, m);
    chol.solve_in_place(inv.as_mut());
    for a in 0..p {
        for b in 0..p {
            out_cov[(a, b)] = 2.0 * inv[(n_theta + a, n_theta + b)];
        }
    }
    // θ-block SE: the same cov = 2·H_dev⁻¹, restricted to the θ diagonal. The β
    // block above discards it; expose it here (the whole point of paying for the
    // joint Hessian). For a scalar grouping the RE stddev equals θ, so this is the
    // stddev's SE directly. A rounding-negative diagonal (never seen at a PD point,
    // but the LLT solve can leave a tiny negative) clamps to 0 before the sqrt.
    for k in 0..n_theta {
        ws.theta_se[k] = (2.0 * inv[(k, k)]).max(0.0).sqrt();
    }

    // Restore the converged PIRLS state at γ̂ (W̃/û/μ̂/factors): the stencil leaves
    // the LAST perturbed eval's state in ws, and the dense caller reads ws.prob/
    // ws.u AFTER this returns (Gamma's σ̂² for tau2/varcorr, the Pearson φ̂,
    // mu_hat) — off the perturbed state Gamma's σ̂² was ~2e-3 high (rung-23
    // stddev gate). Same central re-eval the RX fallback uses; the empty
    // perturbation evaluates exactly at γ̂ (fd_saved).
    let _ = fd_eval(ws, &[], &[], x, y, cluster_ids, n);

    ws.params[..m].copy_from_slice(&ws.fd_saved[..m]);
    ws.warm_seed_active = false; // never leak the FD seed into a later fit / BOBYQA
    ws.pirls_tol_override = None; // nor the FD tight tol
    FdHessianStatus::Ok
}

/// Dense Schur fill (crossed/nested path): X'W̃X, X'W̃M, A⁻¹M'W̃X via the `k×k`
/// `ws.a` LLT, and `ws.schur = X'W̃X − X'W̃M·A⁻¹M'W̃X`. Reads `ws.{a, m, w, x via
/// arg}`. Returns false on a non-PD `ws.a`. Unchanged from the pre-Phase-2 inline
/// inference — moved verbatim so the crossed path is byte-for-byte identical.
///
/// PIRLS's `BetaStep::Profile` β-Schur border step (pirls.rs) reuses this exact
/// C = X'WX, B' = X'WM, T = A⁻¹B, S_β = C − B'T construction each iteration (with
/// that iteration's own W and factor), then additionally solves
/// δβ = S_β⁻¹·(X'ρ − B'δu₀) and folds `u_joint = u_new − T·δβ` back into the
/// conditional-mode iterate — the joint (u, β) Newton step within one PIRLS solve.
pub(crate) fn dense_schur_fill(ws: &mut GlmmWorkspace, x: MatRef<f64>, n: usize) -> bool {
    use faer::linalg::solvers::Solve;
    let (k, p) = (ws.k, ws.p);
    for r in 0..p {
        for c in 0..=r {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, r)] * ws.w[i] * x[(i, c)];
            }
            ws.xtwx[(r, c)] = s;
            ws.xtwx[(c, r)] = s;
        }
    }
    for r in 0..p {
        for c in 0..k {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, r)] * ws.w[i] * ws.m[(i, c)];
            }
            ws.xtwm[(r, c)] = s;
        }
    }
    let ac = match ws.a.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return false,
    };
    for r in 0..k {
        for c in 0..p {
            ws.ainv_mtwx[(r, c)] = ws.xtwm[(c, r)];
        }
    }
    ac.solve_in_place(ws.ainv_mtwx.as_mut());
    for r in 0..p {
        for c in 0..p {
            let mut s = ws.xtwx[(r, c)];
            for j in 0..k {
                s -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)];
            }
            ws.schur[(r, c)] = s;
        }
    }
    true
}

/// Blocked Schur fill (no-extras path): reconstruct mᵢ = Λ_p'·zᵢ per row to build
/// X'W̃X (p×p, dense) and the per-cluster coupling X'W̃M (into `ws.xtwm` columns
/// `f·q_p..`), then solve `A_f T_f = (M'W̃X)_f` per block by REUSING the factored
/// `ws.a_blocks` the converged blocked PIRLS left behind (W̃ in `ws.w`, Λ̂ in
/// `ws.lam`), and `ws.schur = X'W̃X − Σ_f (X'W̃M)_f·T_f`. Only the trailing `p×p`
/// Schur LLT (done by the common code after this) stays dense. Returns false if a
/// stored block is not usable (defensive — the PIRLS already proved them PD).
pub(crate) fn blocked_schur_fill(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    n: usize,
) -> bool {
    let (p, q, s) = (ws.p, ws.groupings.primary_q, ws.groupings.n_primary);
    let k = q * s;
    // X'W̃X (p×p).
    for r in 0..p {
        for c in 0..=r {
            let mut sm = 0.0;
            for i in 0..n {
                sm += x[(i, r)] * ws.w[i] * x[(i, c)];
            }
            ws.xtwx[(r, c)] = sm;
            ws.xtwx[(c, r)] = sm;
        }
    }
    // X'W̃M, blocked: zero then scatter the q_p coupling columns per row.
    for r in 0..p {
        for c in 0..k {
            ws.xtwm[(r, c)] = 0.0;
        }
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let mut m_row = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
        #[allow(clippy::needless_range_loop)]
        for c in 0..q {
            let mut acc = 0.0;
            for rr in c..q {
                let zr = if rr == 0 {
                    1.0
                } else {
                    x[(i, ws.groupings.primary_slope_cols[rr - 1])]
                };
                acc += zr * ws.lam[rr * q + c];
            }
            m_row[c] = acc;
        }
        let wi = ws.w[i];
        for r in 0..p {
            let xw = x[(i, r)] * wi;
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                ws.xtwm[(r, f * q + c)] += xw * m_row[c];
            }
        }
    }
    // T_f = A_f⁻¹ (M'W̃X)_f, per block, reusing the stored factor; ainv_mtwx rows
    // f·q_p.. hold T_f. (M'W̃X)_f[c, col] = (X'W̃M)_f[col, c] = ws.xtwm[(col, f·q+c)].
    for f in 0..s {
        let ablk = f * q * q;
        for col in 0..p {
            let mut rhs = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                rhs[c] = ws.xtwm[(col, f * q + c)];
            }
            glmm_block_solve(&ws.a_blocks[ablk..ablk + q * q], q, &mut rhs[..q]);
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                ws.ainv_mtwx[(f * q + c, col)] = rhs[c];
            }
        }
    }
    // Schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X). Exact: A is block-diagonal, so the per-block
    // solves above equal the full A⁻¹M'W̃X; the Σ_j over k is a full sum (every
    // column j belongs to one cluster and is populated — there are no zero columns).
    for r in 0..p {
        for c in 0..p {
            let mut sm = ws.xtwx[(r, c)];
            for j in 0..k {
                sm -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)];
            }
            ws.schur[(r, c)] = sm;
        }
    }
    true
}

/// Structured Schur fill (intercept-only crossed/nested path — `classify_design`
/// routes slope-carrying extras to `Solver::Sparse` for every family): builds `X'W̃X`
/// (p×p) and `X'W̃M` (p×k, by per-row scatter into each row's core + crossed
/// columns), then applies `A⁻¹` to each of the `p` columns of `M'W̃X` by REUSING
/// the core-block + Schur factors the converged structured PIRLS left in
/// `ws.{core_blocks, schur_blk, coupling}` (via `structured_ainv_solve`), and
/// `ws.schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X)`. Mirrors `blocked_schur_fill`; the only
/// difference is the `A⁻¹` apply uses the block+Schur back-substitution instead of
/// per-block solves alone. Returns false on nothing (the factors were already
/// proven PD by the PIRLS) — kept `-> bool` to match the dispatch arms.
pub(crate) fn structured_schur_fill(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    n: usize,
) -> bool {
    let p = ws.p;
    let g = &ws.groupings;
    let (q, np, s) = (g.primary_q, g.nested_per_parent, g.n_primary);
    let qc = q + np;
    let e = g.k_crossed();
    let prim_width = q * s;
    let k_family = qc * s;
    let k = ws.k;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };
    // X'W̃X (p×p).
    for r in 0..p {
        for c in 0..=r {
            let mut sm = 0.0;
            for i in 0..n {
                sm += x[(i, r)] * ws.w[i] * x[(i, c)];
            }
            ws.xtwx[(r, c)] = sm;
            ws.xtwx[(c, r)] = sm;
        }
    }
    // X'W̃M: zero then scatter each row's core + crossed columns. Reads the PACKED
    // M nonzeros (`m_core_buf` core slice + `cross_*`/`n_cross` crossed entries) the
    // converged re-eval's `build_packed_m` left behind — the dense `ws.m` is no
    // longer maintained on the structured path.
    for r in 0..p {
        for c in 0..k {
            ws.xtwm[(r, c)] = 0.0;
        }
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let wi = ws.w[i];
        let cbase = i * g_cap;
        let ncz = ws.n_cross[i] as usize;
        for r in 0..p {
            let xw = x[(i, r)] * wi;
            for local in 0..qc {
                ws.xtwm[(r, core_col(f, local))] += xw * ws.m_core_buf[i * qc + local];
            }
            for z in 0..ncz {
                let b = ws.cross_col[cbase + z] as usize;
                ws.xtwm[(r, k_family + b)] += xw * ws.cross_val[cbase + z];
            }
        }
    }
    // ainv_mtwx[:, c] = A⁻¹ (M'W̃X)[:, c], one fixed-effect column at a time.
    for c in 0..p {
        for f in 0..s {
            for local in 0..qc {
                ws.a_rhs[f * qc + local] = ws.xtwm[(c, core_col(f, local))];
            }
        }
        for b in 0..e {
            ws.a_rhs[k_family + b] = ws.xtwm[(c, k_family + b)];
        }
        structured_ainv_solve(
            &ws.groupings,
            &ws.core_blocks,
            &ws.coupling,
            &ws.schur_blk,
            &ws.coup_cols,
            &ws.coup_ptr,
            ws.structured_schur.as_mut(),
            ws.force_dense_schur,
            &mut ws.a_rhs,
        );
        for f in 0..s {
            for local in 0..qc {
                ws.ainv_mtwx[(core_col(f, local), c)] = ws.a_rhs[f * qc + local];
            }
        }
        for b in 0..e {
            ws.ainv_mtwx[(k_family + b, c)] = ws.a_rhs[k_family + b];
        }
    }
    // Schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X). Every RE column belongs to a core block or
    // the crossed tail and is populated, so the Σ_j over k is a full sum.
    for r in 0..p {
        for c in 0..p {
            let mut sm = ws.xtwx[(r, c)];
            for j in 0..k {
                sm -= ws.xtwm[(r, j)] * ws.ainv_mtwx[(j, c)];
            }
            ws.schur[(r, c)] = sm;
        }
    }
    true
}
