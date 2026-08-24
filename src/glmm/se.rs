use faer::{Mat, MatRef};

use super::deviance::laplace_deviance_at;
use super::pirls::structured_ainv_solve;
use super::workspace::{fill_z_f64, glmm_block_solve, GlmmWorkspace};
use super::{FdHessianStatus, FD_STEP_BASE};

/// Evaluate the joint Laplace deviance at `fd_saved + Σ deltaₖ·e_{coordₖ}`,
/// reusing `ws.fd_saved` (distinct field from `ws.params`, so the disjoint
/// field borrows are legal). `coords`/`deltas` are ≤ 2 long (a diagonal or a
/// mixed partial). Leaves `ws.params` perturbed — callers restore from
/// `ws.fd_saved` between the directional evals via this same write.
#[allow(clippy::too_many_arguments)]
fn fd_eval(
    ws: &mut GlmmWorkspace,
    coords: &[usize],
    deltas: &[f64],
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
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
    laplace_deviance_at(ws, x, y, cluster_ids, extra_ids, n)
}

/// Central second difference of coordinate `k` at step `s`: `(f(+s) − 2·f0 +
/// f(−s))/s²`, where `eval(coords, deltas)` evaluates the deviance at the base
/// point perturbed by `Σ deltaₖ·e_{coordₖ}`. Returns the raw value — non-finite
/// if either directional eval diverges — so the caller decides fallback. The
/// step is a PARAMETER: the dense (`FD_STEP_BASE`) and sparse (`SPARSE_FD_STEP_REL`)
/// paths pass their own deliberately-divergent constants; this helper never sees one.
pub(crate) fn fd_second_diff(
    eval: &mut impl FnMut(&[usize], &[f64]) -> f64,
    k: usize,
    s: f64,
    f0: f64,
) -> f64 {
    let fp = eval(&[k], &[s]);
    let fm = eval(&[k], &[-s]);
    (fp - 2.0 * f0 + fm) / (s * s)
}

/// Symmetric 4-point mixed partial of `(i, j)` at steps `(si, sj)`:
/// `(f(+si,+sj) − f(+si,−sj) − f(−si,+sj) + f(−si,−sj))/(4·si·sj)`. Returns the raw
/// value (non-finite if any of the four evals diverges); fallback is the caller's,
/// as for `fd_second_diff`. Same eval-closure / step-as-parameter contract.
pub(crate) fn fd_mixed_diff(
    eval: &mut impl FnMut(&[usize], &[f64]) -> f64,
    i: usize,
    j: usize,
    si: f64,
    sj: f64,
) -> f64 {
    let fpp = eval(&[i, j], &[si, sj]);
    let fpm = eval(&[i, j], &[si, -sj]);
    let fmp = eval(&[i, j], &[-si, sj]);
    let fmm = eval(&[i, j], &[-si, -sj]);
    (fpp - fpm - fmp + fmm) / (4.0 * si * sj)
}

/// Dense-path adapter: builds the `fd_eval` closure over `ws`/design and applies
/// the shared `fd_second_diff` stencil. Keeps the per-thread worker workspace and
/// β-fixed deviance wiring local to this side.
#[allow(clippy::too_many_arguments)]
fn second_diff(
    ws: &mut GlmmWorkspace,
    k: usize,
    s: f64,
    f0: f64,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) -> f64 {
    let mut eval = |coords: &[usize], deltas: &[f64]| {
        fd_eval(ws, coords, deltas, x, y, cluster_ids, extra_ids, n)
    };
    fd_second_diff(&mut eval, k, s, f0)
}

/// Dense-path adapter for the 4-point mixed partial; see `second_diff`.
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
    extra_ids: &[Vec<u32>],
    n: usize,
) -> f64 {
    let mut eval = |coords: &[usize], deltas: &[f64]| {
        fd_eval(ws, coords, deltas, x, y, cluster_ids, extra_ids, n)
    };
    fd_mixed_diff(&mut eval, i, j, si, sj)
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
/// differences at `h_θ = FD_STEP_BASE` (ABSOLUTE on the θ block) and
/// `h_β = FD_STEP_BASE·max(1, |β̂_k|)` — see the step-construction comment in the
/// body for why θ does not take the relative form. No Richardson extrapolation
/// (dropped 2026-07-04 — the deviance is step-invariant to ~7 sig figs across
/// h ∈ [1e-4, 1e-1] on this fixture, so a second-order correction bought no
/// measurable accuracy). Every eval here runs PIRLS at `pirls_tol_fd(family)` —
/// the tighter of the `PIRLS_TOL_REL_FD` ceiling and the family's own fit
/// tolerance — via `ws.pirls_tol_override`, set on entry and reset on every
/// exit, so the stencil never differences a deviance looser than the one the
/// optimizer converged on. At an exit tolerance of 1e-6 the FD was NOT
/// step-invariant on cbpp (~0.3% wobble, h=1e-2 vs 1e-3 — the shipped step
/// happened to land on the accurate side); at 1e-8 or tighter it is
/// step-invariant to ~5–6 sig figs across h ∈ [1e-4, 1e-2] and sits on the
/// tight-tol (1e-12) limit, so the accuracy is by construction, not by luck.
/// The fit path never sees the override.
/// One accepted divergence: on the RX fallback below, the central re-eval that
/// repopulates the Schur factors also runs at the FD-pass tol, so fallback RX
/// numbers differ from the plain `WaldSe::Rx` arm's at tolerance level
/// (never LESS converged — harmless).
///
/// Match vs lme4 `vcov(use.hessian=TRUE)`: ~3.4e-7 worst per-entry gap on the
/// committed fixture; worst rel `se_hessian` ≤6e-4 across validation rungs
/// (rung 28 `sim_poisson_offset`), all within `tol.R` bands,
/// including the two large-θ̂ rungs (`sim_binomial_bigsd` θ̂ = 4.51, 7.4e-6;
/// `sim_poisson_bigsd` θ̂ = 2.97, 1.4e-5).
///
/// Both engines' numbers carry FD error of their own, and ours is the larger.
/// Re-measured 2026-08-24 over the whole GLMM corpus by
/// `fd_hessian_margin_corpus_measurement` (`src/sparse/fd_margin.rs`), which
/// rebuilds the joint Hessian at δ, δ/2 and δ/4 and extrapolates the β SEs to
/// h = 0 on the central difference's own O(δ²), S = (4·SE(δ/2) − SE(δ))/3.
/// Where that sequence really is truncation-dominated — every dense canonical
/// rung, gaps shrinking ~4× per halving and the two pairs' extrapolations
/// agreeing to ≤3e-6 — the shipped step sits within 7.7e-5 of the limit
/// (`sim_poisson_slope1`); the sparse large-θ̂ rung `sim_sparse_binomial_bigsd`
/// is truncation-dominated as well, at 1.4e-4 with its two extrapolations 20%
/// apart. Elsewhere there is NO h→0 limit to sit near: on the three
/// non-canonical rungs (`cbpp_probit`, `sim_gamma`, `sim_probit_large`) and on
/// the other four sparse canonical ones the SE gaps GROW 1.8–12× per halving,
/// which is FD noise rather than truncation — f64 rounding of an O(10²–10³)
/// deviance divided by δ², a floor no PIRLS tolerance reaches and only a larger
/// δ avoids — and there the two extrapolated limits disagree by as much as the
/// distance they are estimating. What holds corpus-wide is the weaker
/// step-invariance statement: halving δ from the shipped step moves no β
/// standard error by more than 1.4e-4 relative (`cbpp_probit`), and the shipped
/// δ is on the good side of the knee, since halving again moves them more.
/// So the figures above are measured agreement, not a bound on either engine's
/// FD error: the tightest of them are SMALLER than our own step uncertainty, so
/// none of them can be read as either side's accuracy.
/// lme4's `vcov(use.hessian=TRUE)` is itself
/// `lme4:::deriv12` at an ABSOLUTE δ = 1e-4; measured on these references
/// 2026-07-30, that carries 5–9e-6 of its own FD error, δ = 1e-4 sitting past
/// lme4's own noise knee in β, and two runs of its
/// stencil differing by 4.5e-7…1.8e-6.
/// Tightening this number would pin one lme4 cannot reproduce — but ONLY
/// against an lme4 run at tightened `tolPwrss` (the fixture and the frozen
/// validation references are generated at 1e-13; each records it). At lme4's
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
#[allow(clippy::too_many_arguments)]
pub fn fd_hessian_cov(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> FdHessianStatus {
    use faer::linalg::solvers::Solve;
    let m = ws.params.len();
    let n_theta = ws.n_theta;

    // Every deviance eval below (f0, the f± stencil, and the fallback's central
    // re-eval) converges PIRLS at the FD-pass tol — see the doc comment.
    // Reset on every exit alongside `warm_seed_active`.
    ws.pirls_tol_override = Some(super::pirls_tol_fd(ws.family));

    // Snapshot γ̂ and the per-coordinate FD step; fill z_buf once (blocked AND
    // structured paths — `build_packed_m`'s primary-core reduction reads it the
    // same way `pirls_solve_blocked`'s does; the dense-fallback path skips it,
    // matching `fit_glmm`'s hoist).
    ws.fd_saved[..m].copy_from_slice(&ws.params[..m]);
    // Step construction per `FD_STEP_BASE` (θ absolute, β relative). On toenail
    // (θ̂ = 4.708), a θ-relative step of h_θ = 0.047 puts se(β₀) 4.9e-4 above the
    // converged value; at h_θ = 1e-2 it is 2.2e-5, against a noise floor near
    // h = 2.5e-3.
    for k in 0..m {
        ws.fd_steps[k] = if k < n_theta {
            FD_STEP_BASE
        } else {
            FD_STEP_BASE * ws.fd_saved[k].abs().max(1.0)
        };
    }
    if ws.groupings.extra_offsets.is_empty() || ws.groupings.structured_extras_eligible() {
        let GlmmWorkspace {
            groupings, z_buf, ..
        } = &mut *ws;
        fill_z_f64(groupings, x, z_buf, n);
    }

    // Put û(γ̂) back the way it was found. `u_seed` holds the entry mode (see the
    // seeding block below), and every eval here overwrites `ws.u` with its own
    // perturbed mode, so without this the workspace exits carrying an FD leftover
    // where the fit's mode used to be — and since the seed is now READ from
    // `ws.u`, a second `fd_hessian_cov` on the same workspace would anchor on that
    // leftover and return a different (still valid, but different) covariance.
    // `fd_hessian_parallel_bit_identical_to_serial` calls it exactly twice and is
    // what holds this line. Same contract as the `ws.params`/`fd_saved` restore it
    // sits next to: leave the workspace as you found it.
    macro_rules! restore_fd_mode {
        () => {{
            let kk = ws.k.max(1);
            ws.u[..kk].copy_from_slice(&ws.u_seed[..kk]);
        }};
    }

    // Restore γ̂ and take the RX/Schur fallback: re-eval the central deviance to
    // repopulate W̃/Λ̂/block factors at γ̂, then invert the β information.
    macro_rules! fallback {
        () => {{
            let _ = fd_eval(ws, &[], &[], x, y, cluster_ids, extra_ids, n);
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
            restore_fd_mode!();
            ws.warm_seed_active = false; // never leak the FD seed into a later fit / BOBYQA
            ws.pirls_tol_override = None; // nor the FD-pass tol
            return FdHessianStatus::NonPdFellBackToRx;
        }};
    }

    // Anchor the whole FD grid on the fit's OWN converged mode û(γ̂), which the
    // pinned-γ̂ re-eval in `fit_glmm_ws` just left in `ws.u`. Every eval below — f0
    // included — warm-starts from this one fixed seed (`ws.u_seed`, set once here,
    // read by every `laplace_deviance_at` call via `warm_seed_active`), never from
    // the previous eval's own mode. That makes each f(γ) a function of γ alone: the
    // second differences that build the Hessian only stay valid if every f± sees
    // the same seed, because a *chained* seed (eval k warm-started from eval k−1's
    // mode) would make f(γ) depend on evaluation order and corrupt them. f0 shares
    // the seed for the same reason and so reproduces the deviance the optimizer
    // actually reached.
    //
    // This used to run f0 cold (u = 0) and take ITS mode as the seed, on the
    // assumption that a cold solve at γ̂ re-finds û(γ̂). That assumption holds
    // wherever the PIRLS mode problem has one basin, and fails where it has more
    // than one: on a Gamma fit with the INVERSE link the cold solve lands in a
    // different basin than the fit did, and the whole Hessian is then built around
    // a point that is not the fit's optimum. Measured on `sim_gamma`
    // (`y ~ 1 + x + grp + (1 | cluster)`, Gamma-inverse): the fit reaches deviance
    // 936.7683 and a cold f0 at the same γ̂ returns 1034.5678, ~98 above it. The
    // deviance seen along each coordinate then jumps between the two branches, so
    // the second differences measure the branch gap rather than curvature — every
    // diagonal entry came out around −9.8e5, the joint Hessian was indefinite, and
    // the RX fallback formed its Schur at the same wrong mode and was indefinite
    // too. The fit was reported as failed for want of a standard error.
    //
    // Seeding from û(γ̂) makes f0 equal the fit's deviance exactly there, and moves
    // the log-link sibling's f0 from 1.2e-6 to 1.0e-7 off its own `Fit::deviance` —
    // this is the self-consistency the FD needed on every link, not a Gamma patch.
    let kk = ws.k.max(1);
    ws.u_seed[..kk].copy_from_slice(&ws.u[..kk]);
    ws.warm_seed_active = true;

    let f0 = fd_eval(ws, &[], &[], x, y, cluster_ids, extra_ids, n);
    if !f0.is_finite() {
        fallback!();
    }

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
                            second_diff(
                                wws,
                                i,
                                ws_ro.fd_steps[i],
                                f0,
                                x,
                                y,
                                cluster_ids,
                                extra_ids,
                                n,
                            )
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
                                extra_ids,
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
            let hii = second_diff(ws, i, hi, f0, x, y, cluster_ids, extra_ids, n);
            if !hii.is_finite() {
                fallback!();
            }
            ws.hess_scratch[(i, i)] = hii;
            for j in (i + 1)..m {
                let hj = ws.fd_steps[j];
                let hij = mixed_diff(ws, i, j, hi, hj, x, y, cluster_ids, extra_ids, n);
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
    let _ = fd_eval(ws, &[], &[], x, y, cluster_ids, extra_ids, n);

    ws.params[..m].copy_from_slice(&ws.fd_saved[..m]);
    // That re-eval lands on û(γ̂) again but at the FD-pass tol, which is never
    // looser than the fit's, so it is at least as converged as the mode this call
    // was handed. Take the handed one back verbatim, so a second call on the same
    // workspace seeds from the same vector and returns the same bits. The ≲1e-12
    // it leaves between `ws.u` and the `ws.prob`/W̃ the re-eval just wrote is far
    // below anything read off them.
    restore_fd_mode!();
    ws.warm_seed_active = false; // never leak the FD seed into a later fit / BOBYQA
    ws.pirls_tol_override = None; // nor the FD-pass tol
    FdHessianStatus::Ok
}

/// C = X'W̃X (p×p), full matrix, via the W∘X GEMM scratch `ws.wx` (rebuilt fresh
/// from `ws.w` each call). Shared by `dense_schur_fill`/`blocked_schur_fill`/
/// `structured_schur_fill`: all three read the identical `ws.w`/`x` pair for this
/// block (they differ only in how they build `X'W̃M`), so one GEMM fill serves
/// all three rather than three copies of the same scalar triple loop. Mirrors the
/// `pirls.rs` `BetaStep::Profile` xtwx GEMM this construction is shared with.
fn xtwx_fill(ws: &mut GlmmWorkspace, x: MatRef<f64>, n: usize) {
    let p = ws.p;
    for c in 0..p {
        for i in 0..n {
            ws.wx[(i, c)] = ws.w[i] * x[(i, c)];
        }
    }
    faer::linalg::matmul::matmul(
        ws.xtwx.as_mut(),
        faer::Accum::Replace,
        x.subrows(0, n).transpose(),
        ws.wx.as_ref().subrows(0, n),
        1.0,
        faer::Par::Seq,
    );
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
    xtwx_fill(ws, x, n);
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
    xtwx_fill(ws, x, n);
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
                // The one Z read on this path that does not come from `ws.z_buf`,
                // so it applies the RE column's internal scale itself — mirrors
                // `workspace::fill_z_f64` / `build_z`, change together. Indexed by
                // the Λ ROW `rr`, not the column `c`.
                let zr = if rr == 0 {
                    1.0
                } else {
                    x[(i, ws.groupings.primary_slope_cols[rr - 1])]
                        / ws.groupings.primary_slope_scales[rr - 1]
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
    xtwx_fill(ws, x, n);
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
