use faer::{Mat, MatRef};

use super::derivative::DerivStatus;
use super::deviance::laplace_deviance_at;
use super::pirls::{schur_from_residuals, structured_ainv_solve};
use super::workspace::{fill_z_f64, glmm_block_solve, GlmmLayout, GlmmWorkspace};
use super::{FdHessianStatus, FD_STEP_BASE};

/// Evaluate the joint Laplace deviance at `fd_saved + Σ deltaₖ·e_{coordₖ}`,
/// reusing `ws.fd.fd_saved` (distinct field from `ws.params`, so the disjoint
/// field borrows are legal). `coords`/`deltas` are ≤ 2 long (a diagonal or a
/// mixed partial). Leaves `ws.params` perturbed — callers restore from
/// `ws.fd.fd_saved` between the directional evals via this same write.
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
    let m = ws.fd.fd_saved.len();
    ws.params[..m].copy_from_slice(&ws.fd.fd_saved[..m]);
    for (&c, &d) in coords.iter().zip(deltas) {
        ws.params[c] += d;
    }
    // β-FIXED by construction: `laplace_deviance_at` hardcodes `beta_mode =
    // BetaMode::Fixed`. The FD-Hessian differentiates a function of the
    // caller's β, so β must NOT move under these directional evals — a Profile step
    // here would make each `f(γ)` depend on the profiled β̂(γ) and corrupt the
    // second differences.
    // The FD-Hessian evaluations run their own tight PIRLS tolerance and are
    // not part of the search; they get a throwaway counter so nothing they do
    // reaches the reported counts (mirrors the exhaustion counter of
    // `laplace_deviance`, which skips a solve whose override is at or below
    // `pirls_tol`, as the FD tolerance always is).
    let mut fd_counters = crate::counters::EvalCounters::new();
    laplace_deviance_at(ws, x, y, cluster_ids, extra_ids, n, &mut fd_counters)
}

/// Central second difference of coordinate `k` at step `s`: `(f(+s) − 2·f0 +
/// f(−s))/s²`, where `eval(coords, deltas)` evaluates the deviance at the base
/// point perturbed by `Σ deltaₖ·e_{coordₖ}`. Returns the raw value — non-finite
/// if either directional eval diverges — so the caller decides fallback. The
/// step is a PARAMETER: the blocked/structured (`FD_STEP_BASE`) and packed-row
/// (`SPARSE_FD_STEP_REL`) arms pass their own deliberately-divergent constants;
/// this helper never sees one.
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

/// Relative FD step for the packed-row layout's FD-Hessian FALLBACK,
/// [`packed_fd_hessian_cov`] — the rung that answers where the assembled pass
/// declines a packed fit, not that layout's own default. Deliberately NOT the
/// `FD_STEP_BASE` (1e-2) the blocked and structured layouts' stencil takes:
/// the two sit on opposite sides of the truncation-vs-noise trade. On the weighted sparse-Z Gamma golden (`sim_sparse_gamma`,
/// weight-perturbed cell), the flat intercept direction's FD-step plateau (true
/// curvature, found by scanning h) sits at h ∈ [5e-5, 2e-4]; h = 1e-3 falls
/// outside that plateau and biases se(β₀) high, while h = 1e-4 lands on the
/// plateau for both the weighted cell (0.18915 vs lme4 0.18909) and its
/// unweighted sibling (0.18919, unchanged from the 1e-3 step — that golden was
/// already inside the plateau). h = 1e-4 also sits inside the other layouts'
/// already-validated [1e-4, 1e-2] band. Those are the mirror image: at h = 1e-3
/// their FD noise blows the curated se_hess gates (sim_gamma 1e-2, cbpp_probit
/// 2e-3 vs the 1e-3 band) while h = 1e-2 holds them at ~1e-4 — so
/// `FD_STEP_BASE` stays 1e-2 and this one must not be folded back into it.
///
/// Unlike `FD_STEP_BASE`, θ coordinates here stay RELATIVE rather than taking an
/// absolute step: this constant is calibrated on the NOISE side, so dropping the
/// `max(1, |θ̂|)` scaling would shrink h_θ from ~4.7e-4 toward 1e-4 on a large-SD
/// model and push it further into noise, not out of it. This arm needs its own
/// step calibration in the large-θ̂ regime, measured separately.
pub(crate) const SPARSE_FD_STEP_REL: f64 = 1e-4;

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

/// One Hessian entry `(i, j)` of an FD grid: diagonal → `fd_second_diff`,
/// off-diagonal → `fd_mixed_diff`, through the `fd_eval` adapters. Shared by
/// both arms of `packed_fd_hessian_cov` and the rayon arm of
/// `joint_hessian_cov`.
#[allow(clippy::too_many_arguments)]
fn fd_hess_entry(
    ws: &mut GlmmWorkspace,
    i: usize,
    j: usize,
    si: f64,
    sj: f64,
    f0: f64,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) -> f64 {
    if i == j {
        second_diff(ws, i, si, f0, x, y, cluster_ids, extra_ids, n)
    } else {
        mixed_diff(ws, i, j, si, sj, x, y, cluster_ids, extra_ids, n)
    }
}

/// Joint (θ,β) FD Hessian of the Laplace deviance on the PACKED-ROW layout,
/// built into `ws.inference.hess_scratch` (symmetric, `m×m`). Same scheme as the
/// other layouts' stencil — single-step central differences, no Richardson —
/// with its own step rule: `h_k = SPARSE_FD_STEP_REL·max(1, |γ̂_k|)` RELATIVE on
/// every coordinate, θ included (see the constant). Returns false on a
/// non-finite eval, which routes the caller to the RX fallback.
///
/// This is the packed-row layout's rung 3, not its default: the assembled
/// pass answers a packed fit, and this stencil is what a fit that pass
/// declines gets instead — a refused fit keeps a finite-difference Hessian
/// standard error rather than dropping to Rx.
///
/// Every eval here cold-seeds û = 0 inside the packed arm of `laplace_deviance`,
/// so each grid cell is a pure function of `(γ̂, steps, design)` and the
/// per-thread workspaces reproduce the serial values bit-for-bit.
///
/// Tolerance contract: the CALLER sets `ws.fd.pirls_tol_override` around this
/// call and resets it on every exit.
#[allow(clippy::too_many_arguments)]
fn packed_fd_hessian_cov(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    m: usize,
    n: usize,
) -> bool {
    for k in 0..m {
        ws.fd.fd_steps[k] = SPARSE_FD_STEP_REL * ws.fd.fd_saved[k].abs().max(1.0);
    }
    let f0 = fd_eval(ws, &[], &[], x, y, cluster_ids, extra_ids, n);
    if !f0.is_finite() {
        return false;
    }
    let use_par = cfg!(all(feature = "parallel", not(target_arch = "wasm32"))) && ws.parallel_inner;
    if use_par {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use super::workspace::fd_worker_ws;
            use rayon::prelude::*;
            let cells: Vec<(usize, usize)> =
                (0..m).flat_map(|i| (i..m).map(move |j| (i, j))).collect();
            let ws_ro: &GlmmWorkspace = ws;
            let results: Vec<(usize, usize, f64)> = cells
                .par_iter()
                .map_init(
                    || fd_worker_ws(ws_ro, n),
                    |wws, &(i, j)| {
                        let h = fd_hess_entry(
                            wws,
                            i,
                            j,
                            ws_ro.fd.fd_steps[i],
                            ws_ro.fd.fd_steps[j],
                            f0,
                            x,
                            y,
                            cluster_ids,
                            extra_ids,
                            n,
                        );
                        (i, j, h)
                    },
                )
                .collect();
            // The serial arm bails on the FIRST non-finite eval; here the whole
            // grid ran first, then we check — same destination (RX fallback),
            // extra work only on the already-failing path.
            if results.iter().any(|&(_, _, h)| !h.is_finite()) {
                return false;
            }
            for (i, j, h) in results {
                ws.inference.hess_scratch[(i, j)] = h;
                ws.inference.hess_scratch[(j, i)] = h;
            }
        }
    } else {
        for i in 0..m {
            let hi = ws.fd.fd_steps[i];
            for j in i..m {
                let hj = ws.fd.fd_steps[j];
                let h = fd_hess_entry(ws, i, j, hi, hj, f0, x, y, cluster_ids, extra_ids, n);
                if !h.is_finite() {
                    return false;
                }
                ws.inference.hess_scratch[(i, j)] = h;
                ws.inference.hess_scratch[(j, i)] = h;
            }
        }
    }
    true
}

/// The layout's Schur fill: the blocked layout reuses the per-block factors
/// the blocked PIRLS left in `ws.pirls.a_blocks`; the structured one reuses the
/// core+Schur factors left in `ws.structured.{core_blocks, schur_blk,
/// coupling}`; the packed one factors `ws.packed.a`. `xd` is the fixed-effect
/// design the Schur is built on, `x` the design the random-effect slopes are
/// read from; they differ only on [`rx_schur_llt`]'s rebuild.
fn schur_fill(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    xd: MatRef<f64>,
    cluster_ids: &[u32],
    n: usize,
) -> bool {
    match ws.layout {
        GlmmLayout::Blocked => blocked_schur_fill(ws, x, xd, cluster_ids, n),
        GlmmLayout::Structured => structured_schur_fill(ws, xd, cluster_ids, n),
        GlmmLayout::Packed => packed_schur_fill(ws, xd, n),
    }
}

/// The Rx β information `S = X'W̃X − X'W̃M·A⁻¹M'W̃X` at the current mode, left in
/// `ws.border.schur`, and a Cholesky factor to read `S⁻¹` from. `None` when `A`
/// or `S` is not positive definite. Requires `ws.pirls.{w, lam, a_blocks, …}`
/// to hold what a converged PIRLS at the current `ws.params` left behind.
///
/// `S` is a Gram in `X` and squares its condition number, which costs digits
/// on a nearly collinear design (measured: Var(β̂) 9e-5 relative off a
/// high-precision Schur at cond(X) ≈ 2e6, 2e-3 at 7e6). Where the direct
/// factor has a pivot ratio below `GRAM_PIVOT_MIN` (`lmm::DesignQr`, whose
/// comment says why that bound), or does not exist, `S` is rebuilt on
/// `X̃ = X·U⁻¹`, `U` the `DesignQr` of the prior-weighted design, as the LMM
/// kernels fit; `X̃`'s Schur keeps the digits (3e-16 on the same designs). Then
/// the factor returned is `S̃`'s, the `DesignQr` comes with it, and since
/// `S = UᵀS̃U` a right-hand side `b` is solved as `U⁻ᵀb` against it.
///
/// Also records `S`'s pivot ratio in `ws.inference.{pivot, pivot_col}`, on the
/// fitted design's own columns: read off the direct factor where that is the
/// one returned (the ratio its routing test already computed), and off
/// `Uᵀ·L̃`, the rebuilt factor mapped back, where the rebuild ran.
pub(crate) fn rx_schur_llt(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    n: usize,
) -> Option<(crate::linalg::SeqLlt, Option<crate::lmm::DesignQr>)> {
    let p = ws.p;
    if !schur_fill(ws, x, x, cluster_ids, n) {
        return None;
    }
    let direct = crate::linalg::SeqLlt::new(ws.border.schur.as_ref());
    let direct_pivot = direct
        .as_ref()
        .map(|c| crate::ols::min_pivot_ratio(c.L(), p));
    if let Some((pivot, col)) = direct_pivot {
        if pivot >= crate::lmm::GRAM_PIVOT_MIN {
            (ws.inference.pivot, ws.inference.pivot_col) = (pivot, col as u32);
            return direct.map(|c| (c, None));
        }
    }
    let mut qr = crate::lmm::DesignQr::identity(p);
    let mut row = vec![0.0; p];
    let prior_w = &ws.prior_w;
    if !qr.set_from_rows(x.subrows(0, n), |i| prior_w[i].sqrt(), &mut row) {
        if let Some((pivot, col)) = direct_pivot {
            (ws.inference.pivot, ws.inference.pivot_col) = (pivot, col as u32);
        }
        return direct.map(|c| (c, None));
    }
    let mut xt = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for (j, v) in row.iter_mut().enumerate() {
            *v = x[(i, j)];
        }
        qr.solve_row(&mut row);
        for (j, &v) in row.iter().enumerate() {
            xt[(i, j)] = v;
        }
    }
    if !schur_fill(ws, x, xt.as_ref(), cluster_ids, n) {
        return None;
    }
    let chol = crate::linalg::SeqLlt::new(ws.border.schur.as_ref())?;
    let mut l = chol.L().to_owned();
    qr.map_factor(l.as_mut());
    let (pivot, col) = crate::ols::min_pivot_ratio(l.as_ref(), p);
    (ws.inference.pivot, ws.inference.pivot_col) = (pivot, col as u32);
    Some((chol, Some(qr)))
}

/// Fill `out_cov` (p×p) with the RX/Schur fixed-effect covariance `S⁻¹` of
/// [`rx_schur_llt`] — `S` is the β-INFORMATION matrix, so the inverse is the
/// covariance directly (NO factor of 2; that factor only applies to the
/// deviance Hessian, where info = H_dev/2). Returns false on a non-PD Schur.
/// The `joint_hessian_cov` fallback's covariance; also leaves its square root
/// in `ws.inference.cov_root` for the fallback's joint Wald.
pub(crate) fn rx_cov_into(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> bool {
    let (chol, qr) = match rx_schur_llt(ws, x, cluster_ids, n) {
        Some(f) => f,
        None => return false,
    };
    let mut inv = Mat::<f64>::identity(p, p);
    chol.solve_in_place(inv.as_mut());
    // S⁻¹ = U⁻¹·S̃⁻¹·U⁻ᵀ: `map_beta` applies U⁻¹ to each column, then to each
    // row of the product.
    if let Some(qr) = &qr {
        let mut v = vec![0.0; p];
        for j in 0..p {
            qr.map_beta(inv.col_as_slice_mut(j));
        }
        for a in 0..p {
            for (b, vb) in v.iter_mut().enumerate() {
                *vb = inv[(a, b)];
            }
            qr.map_beta(&mut v);
            for (b, &vb) in v.iter().enumerate() {
                inv[(a, b)] = vb;
            }
        }
    }
    for a in 0..p {
        for b in 0..p {
            out_cov[(a, b)] = inv[(a, b)];
        }
    }
    // `cov_root` (`RᵀR = S⁻¹`) off the factor itself, `R = L⁻¹·U⁻ᵀ` (no `U` on
    // the direct factor): the `Rx` arm's own columns. A Cholesky of the `S⁻¹`
    // just formed would square cond(X) again (measured 1e-5 relative on the
    // joint Wald at cond(X) ≈ 1e6).
    let mut e = vec![0.0; p];
    let mut col = vec![0.0; p];
    for j in 0..p {
        e.fill(0.0);
        e[j] = 1.0;
        if let Some(qr) = &qr {
            qr.solve_row(&mut e);
        }
        if crate::ols::triangular_solve_norm_sq(chol.L(), |i| e[i], &mut col, p, false).is_nan() {
            col.fill(f64::NAN);
        }
        for (i, &c) in col.iter().enumerate() {
            ws.inference.cov_root[(i, j)] = c;
        }
    }
    true
}

/// Joint (θ,β) Hessian of the Laplace deviance at the converged point in
/// `ws.params`; inverts and writes the p×p fixed-effect covariance block into
/// `out_cov`. On a non-PD joint Hessian (or a non-finite perturbed deviance —
/// the few-cluster failure mode) writes the RX/Schur covariance instead and
/// returns the fallback status. Restores `ws.params` on return, and leaves a
/// square root of the covariance in `ws.inference.cov_root`. Matches glmer
/// `vcov(use.hessian = TRUE)` (factor of 2: deviance = −2logL, so observed
/// info = H_dev/2 and cov = info⁻¹ = 2·H_dev⁻¹).
///
/// Three rungs, and the body's own routing comment says what sends a cell
/// down each. On every shape `assembled::assembly_routes` takes — the blocked
/// path (no extra groupings), the structured extras path and the packed-row
/// layout, at every `m`, an AGQ-routed shape excepted — the joint Hessian is
/// EXACT, from the assembled pass's first-order lanes: no step, no stencil,
/// no per-cell PIRLS re-solve. Where that pass declines, a shape with a dual
/// kernel (`derivative::supports_shape`) takes the exact hyper-dual pass
/// instead, up to that pass's own lane cap `MAX_DUAL_N`; the AGQ envelope is
/// always that pass's, differentiating the AGQ deviance where the fit used
/// AGQ. The FD stencil is what is left below both: `packed_fd_hessian_cov`
/// at its own step on the packed-row layout, the `FD_STEP_BASE` grid in this
/// function on the other two. Everything below about `FD_STEP_BASE`, step-invariance, the corpus
/// margin measurement and the comparison against lme4's own `deriv12`
/// describes that grid.
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
/// tolerance — via `ws.fd.pirls_tol_override`, set on entry and reset on every
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
/// (rung 28 `sim_poisson_offset`), all within `grid/tol.R` bands,
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
/// function. Our PIRLS exit carries no such lag: every kernel re-evaluates
/// η/μ/W at the iterate it returns and rebuilds and refactors `A` there, so
/// `dev`, `‖u‖²`, `log|A|` and the factor the fills below read all describe
/// that one point. With W at û, W = μ(1−μ) is the exact η-Hessian for
/// canonical links, i.e. the textbook Laplace term.
/// Supporting facts: H_ββ is exact (`rx_cov_into` matches lme4
/// `vcov(use.hessian=FALSE)` to ~3.6e-6 method-matched), and the true θ↔β
/// correction RAISES cbpp SEs above RX — as ours does; lme4's default-tol value
/// lowers them, a sign artifact of the lagged weights.
///
/// `m = n_theta + p`; the β block is rows/cols `n_theta..m`. On NB `ws.params`
/// carries one more trailing entry (`ln θ_NB`), which this (θ, β) FD grid never
/// perturbs — its row and column are appended separately, by central
/// differences of the full objective (the `DispRow::Nb` arm below), so the
/// reported β SE carries θ_NB's uncertainty; unlike the GLM route's
/// `fit_glm_nb`, which conditions on θ̂.
/// Precondition: `ws` is at a CONVERGED fit and `ws.z_buf`-eligible scratch is
/// valid for (x, ids, n) (the deviance evals re-solve PIRLS).
#[allow(clippy::too_many_arguments)]
pub fn joint_hessian_cov(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> FdHessianStatus {
    // The β block of the joint Hessian is a Gram in X (its leading term is
    // `2·X'W̃X`), so inverting it squares cond(X): measured 5e-4 relative on
    // Var(β̂) at cond(X) ≈ 1e6, every rung and family alike, and a forced FD
    // stencil off by its whole value, its β steps being badly scaled along a
    // near-dependency. Where `X'W̃X` has a pivot ratio below `GRAM_PIVOT_MIN`
    // (`lmm::DesignQr`, whose comment says why that bound), every rung runs in
    // the joint search's own β coordinates instead (`BetaScale`, `β̃ = Lᵀβ`
    // on `X̃ = XL⁻ᵀ`, whose weighted Gram is the identity), and
    // `Cov(β̂) = L⁻ᵀ·Cov(β̃)·L⁻¹` maps back. The θ block of the inverse does not
    // depend on how β is parametrized. Random-effect slopes are read from the
    // raw `x` (`z_buf`, the Rx fallback).
    let n_theta = ws.n_theta;
    xtwx_fill(ws, x, n);
    let well_conditioned = crate::linalg::SeqLlt::new(ws.border.xtwx.as_ref())
        .is_some_and(|c| crate::ols::min_pivot_ratio(c.L(), p).0 >= crate::lmm::GRAM_PIVOT_MIN);
    let mut x_search = None;
    let mut beta = Vec::new();
    if !well_conditioned {
        ws.beta_scale.set(x, &ws.prior_w[..n], n);
        if ws.beta_scale.is_active() {
            let mut xt = Mat::<f64>::zeros(n, p);
            let mut row = vec![0.0; p];
            for i in 0..n {
                for (j, v) in row.iter_mut().enumerate() {
                    *v = x[(i, j)];
                }
                ws.beta_scale.to_search_row(&mut row);
                for (j, &v) in row.iter().enumerate() {
                    xt[(i, j)] = v;
                }
            }
            beta.extend_from_slice(&ws.params[n_theta..n_theta + p]);
            ws.beta_scale
                .to_search(&mut ws.params[n_theta..n_theta + p]);
            x_search = Some(xt);
        }
    }
    let xd = x_search.as_ref().map_or(x, |m| m.as_ref());
    let status = joint_hessian_cov_fam(
        ws,
        x,
        xd,
        x_search.is_some(),
        y,
        cluster_ids,
        extra_ids,
        p,
        n,
        out_cov,
    );
    // The Rx fallback's covariance is already the raw design's.
    let mapped = x_search.is_some() && status == FdHessianStatus::Ok;
    // `cov_root` (`RᵀR = Cov(β̂)`, what the joint Wald reads) from the Cholesky
    // factor `L` of the covariance in the coordinates it was computed in, where
    // it is well conditioned: `R = Lᵀ`, or `R = Lᵀ·Mᵀ` where the map back is
    // `Cov(β̂) = M·Cov(β̃)·Mᵀ` (`M` = `BetaScale::to_caller`), so row `r` of `R`
    // is `M` applied to column `r` of `L`. The Rx fallback's `rx_cov_into`
    // left its own.
    if status == FdHessianStatus::Ok {
        match crate::linalg::SeqLlt::new(out_cov.as_ref()) {
            Some(c) => {
                let mut v = vec![0.0; p];
                for r in 0..p {
                    for (i, vi) in v.iter_mut().enumerate() {
                        *vi = c.L()[(i, r)];
                    }
                    if mapped {
                        ws.beta_scale.to_caller(&mut v);
                    }
                    for (j, &vj) in v.iter().enumerate() {
                        ws.inference.cov_root[(r, j)] = vj;
                    }
                }
            }
            None => ws.inference.cov_root.fill(f64::NAN),
        }
    }
    if x_search.is_some() {
        ws.params[n_theta..n_theta + p].copy_from_slice(&beta);
        if mapped {
            let mut v = vec![0.0; p];
            for j in 0..p {
                ws.beta_scale.to_caller(out_cov.col_as_slice_mut(j));
            }
            for a in 0..p {
                for (b, vb) in v.iter_mut().enumerate() {
                    *vb = out_cov[(a, b)];
                }
                ws.beta_scale.to_caller(&mut v);
                for (b, &vb) in v.iter().enumerate() {
                    out_cov[(a, b)] = vb;
                }
            }
        }
    }
    status
}

/// [`joint_hessian_cov`]'s family split, on the fixed-effect design `xd` the
/// rungs differentiate; `x` is the design the random-effect slopes and the Rx
/// fallback read. `beta_scaled`: `xd` is `ws.beta_scale`'s search design.
#[allow(clippy::too_many_arguments)]
fn joint_hessian_cov_fam(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    xd: MatRef<f64>,
    beta_scaled: bool,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
) -> FdHessianStatus {
    // Mixed Gamma: at the fitted φ̂ the objective in (θ, β) is exactly a GLMM on
    // the prior weights `prior_wᵢ/φ̂` with the bare deviance as its data term,
    // so for the duration of this call the workspace IS that GLMM
    // (`GlmmWorkspace::at_fixed_dispersion`). Every rung below then
    // differentiates the fitted objective at fixed φ̂ with no Gamma case of its
    // own, and `joint_hessian_cov_at` appends the `ln φ` row itself.
    //
    // Negative binomial: θ_NB is a coordinate of the outer search too, so its
    // row is appended as well (`DispRow::Nb`), and Cov(β̂) carries its
    // uncertainty, as glmmTMB's does.
    match ws.family {
        // A held φ (`GlmmWorkspace::gamma_phi_held`) is not a coordinate of the
        // fit, so the joint Hessian is `(θ, β)`'s at that φ, with no `ln φ` row.
        crate::spec::Family::Gamma { .. } => {
            // Compute the `ln φ` diagonal's `Gₚ''` term from `ws.prior_w` (the
            // internal `ŵ`) BEFORE `at_fixed_dispersion` swaps it to the
            // per-row shapes `ŵᵢ/φ̂_int` below; `None` on an unweighted fit
            // takes `_d2`'s bit-identical unweighted arithmetic rather than
            // reading (possibly stale) all-ones weights.
            let disp = ws.gamma_phi_held.is_none().then(|| {
                let weights = ws.weighted.then_some(&ws.prior_w[..n]);
                DispRow::Gamma {
                    d2: crate::family::gamma_dispersion_term_d2(ws.gamma_phi.ln(), weights, n),
                }
            });
            ws.at_fixed_dispersion(n, |ws| {
                joint_hessian_cov_at(
                    ws,
                    x,
                    xd,
                    beta_scaled,
                    y,
                    cluster_ids,
                    extra_ids,
                    p,
                    n,
                    out_cov,
                    disp,
                )
            })
        }
        crate::spec::Family::NegativeBinomial { .. } => joint_hessian_cov_at(
            ws,
            x,
            xd,
            beta_scaled,
            y,
            cluster_ids,
            extra_ids,
            p,
            n,
            out_cov,
            Some(DispRow::Nb),
        ),
        _ => joint_hessian_cov_at(
            ws,
            x,
            xd,
            beta_scaled,
            y,
            cluster_ids,
            extra_ids,
            p,
            n,
            out_cov,
            None,
        ),
    }
}

/// The dispersion coordinate `joint_hessian_cov_at` appends to the joint
/// Hessian, when the family's outer search carries one.
#[derive(Clone, Copy)]
enum DispRow {
    /// Mixed Gamma, `ln φ`: the row from the fixed-φ̂ scaling identity, in the
    /// `(ξ, β, t)` coordinates. `d2` is `family::gamma_dispersion_term_d2` at
    /// the fitted `ln φ̂_int` and weights, computed by the caller before
    /// `GlmmWorkspace::at_fixed_dispersion` swaps `prior_w`.
    Gamma { d2: f64 },
    /// Negative binomial, `ln θ_NB`: the row by central differences in
    /// `ln θ_NB`, the diagonal of the full objective and the mixed entries of
    /// the exact gradient (see `joint_hessian_cov_at`), in the absolute
    /// `(θ, β, ln θ_NB)` coordinates.
    Nb,
}

/// [`joint_hessian_cov`]'s body. `disp` names the dispersion coordinate the
/// joint Hessian gains before it is inverted: [`DispRow::Gamma`] with the
/// workspace already holding the fixed-φ̂ GLMM (see the wrapper), or
/// [`DispRow::Nb`]. `None` on every family whose dispersion is fixed. `xd` is
/// the fixed-effect design every evaluation reads, `x` the design `z_buf` and
/// the Rx fallback read, and `beta_scaled` says `xd` is `ws.beta_scale`'s
/// search design (see [`joint_hessian_cov_fam`]).
#[allow(clippy::too_many_arguments)]
fn joint_hessian_cov_at(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    xd: MatRef<f64>,
    beta_scaled: bool,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    out_cov: &mut Mat<f64>,
    disp: Option<DispRow>,
) -> FdHessianStatus {
    let m = ws.n_theta + p;
    let n_theta = ws.n_theta;

    // Snapshot γ̂; fill z_buf once from the raw `x` (every layout, matching
    // `fit_glmm`'s hoist), so the random-effect slopes never come from `xd`.
    // Both live OUTSIDE the exact/FD branch below: `fallback!()` restores
    // `ws.params` from `fd_saved` and `rx_cov_into` reads `z_buf` on both arms.
    ws.fd.fd_saved[..m].copy_from_slice(&ws.params[..m]);
    fill_z_f64(&ws.groupings, x, &mut ws.z_buf, n);

    // Put û(γ̂) back the way it was found. `u_seed` holds the entry mode (see the
    // seeding block below), and every eval here overwrites `ws.pirls.u` with its own
    // perturbed mode, so without this the workspace would exit carrying an FD
    // leftover in place of the fit's mode — and since the seed is read from
    // `ws.pirls.u`, a second `joint_hessian_cov` on the same workspace would anchor on that
    // leftover and return a different (still valid, but different) covariance.
    // `fd_hessian_parallel_bit_identical_to_serial` calls it exactly twice and is
    // what holds this line. Same contract as the `ws.params`/`fd_saved` restore it
    // sits next to: leave the workspace as you found it.
    macro_rules! restore_fd_mode {
        () => {{
            let kk = ws.k.max(1);
            ws.pirls.u[..kk].copy_from_slice(&ws.u_seed[..kk]);
            ws.pirls.prob[..n].copy_from_slice(&ws.inference.fd_saved_prob[..n]);
        }};
    }

    // Restore γ̂ and take the RX/Schur fallback: re-eval the central deviance to
    // repopulate W̃/Λ̂/block factors at γ̂, then invert the β information.
    macro_rules! fallback {
        () => {{
            let _ = fd_eval(ws, &[], &[], xd, y, cluster_ids, extra_ids, n);
            // The fallback reports an RX vcov exactly as the production Rx arm
            // does: on Gamma the weights in place here are `prior_wᵢ/φ̂`, so the
            // Schur inverse is already `φ̂·RX⁻¹` and takes no further factor.
            let ok = rx_cov_into(ws, x, cluster_ids, p, n, out_cov);
            // Double failure (joint Hessian AND RX Schur both non-PD; small
            // Gamma-inverse fits with a large random-effect sd reach it):
            // rx_cov_into leaves out_cov UNTOUCHED on `false`, so it would keep
            // stale data while we still report NonPdFellBackToRx. NaN-fill so the
            // caller, which reports the fit without SEs, can detect it via is_nan().
            if !ok {
                for a in 0..p {
                    for b in 0..p {
                        out_cov[(a, b)] = f64::NAN;
                    }
                }
                ws.inference.cov_root.fill(f64::NAN);
            }
            // No joint Hessian on the RX fallback ⇒ no θ-block SE to report.
            ws.inference.theta_cov.fill(f64::NAN);
            ws.params[..m].copy_from_slice(&ws.fd.fd_saved[..m]);
            restore_fd_mode!();
            ws.fd.warm_seed_active = false; // never leak the FD seed into a later fit / BOBYQA
            ws.fd.pirls_tol_override = None; // nor the FD-pass tol
            return FdHessianStatus::NonPdFellBackToRx;
        }};
    }

    // Anchor the whole FD grid on the fit's OWN converged mode û(γ̂), which the
    // pinned-γ̂ re-eval in `fit_glmm_ws` just left in `ws.pirls.u`. Every eval below — f0
    // included — warm-starts from this one fixed seed (`ws.u_seed`, set once here,
    // read by every `laplace_deviance_at` call via `warm_seed_active`), never from
    // the previous eval's own mode. The packed-row layout is the exception: its arm
    // of `laplace_deviance` replaces the seed with û = 0 on every stencil eval, for
    // the reason measured there. That makes each f(γ) a function of γ alone: the
    // second differences that build the Hessian only stay valid if every f± sees
    // the same seed, because a *chained* seed (eval k warm-started from eval k−1's
    // mode) would make f(γ) depend on evaluation order and corrupt them. f0 shares
    // the seed for the same reason and so reproduces the deviance the optimizer
    // actually reached.
    //
    // The PIRLS mode problem is not guaranteed to have one basin: on a Gamma fit
    // with the INVERSE link, a cold solve (u = 0) at γ̂ can land in a different
    // basin than the fit's own mode, so seeding f0 from a cold start instead of
    // from `ws.u_seed` would build the whole Hessian around a point that is not
    // the fit's optimum. Measured on `sim_gamma`
    // (`y ~ 1 + x + grp + (1 | cluster)`, Gamma-inverse): the fit reaches deviance
    // 936.7683 while a cold f0 at the same γ̂ returns 1034.5678, ~98 above it. The
    // deviance seen along each coordinate would then jump between the two
    // branches, so the second differences would measure the branch gap rather
    // than curvature — every diagonal entry comes out around −9.8e5, the joint
    // Hessian is indefinite, and the RX fallback forms its Schur at the same
    // wrong mode and is indefinite too, so the fit reports as failed for want of
    // a standard error.
    //
    // Seeding from û(γ̂) makes f0 equal the fit's deviance exactly there, and
    // keeps the log-link sibling's f0 within 1.0e-7 of its own `Fit::deviance` —
    // this is the self-consistency the FD needs on every link, not a Gamma patch.
    let kk = ws.k.max(1);
    ws.u_seed[..kk].copy_from_slice(&ws.pirls.u[..kk]);
    ws.inference.fd_saved_prob[..n].copy_from_slice(&ws.pirls.prob[..n]);
    ws.fd.warm_seed_active = true;

    // An exact joint Hessian on every layout: the blocked path, which is also
    // the whole AGQ envelope since the AGQ gate in `deviance.rs` requires
    // `extra_offsets.is_empty()`, the structured extras path, and the
    // packed-row path. The arms below differentiate the final evaluation at
    // the converged mode, so none of them pays a stencil, a step, a per-cell
    // PIRLS re-solve or an FD-pass tolerance.
    //
    // Three rungs, in order, and what sends a cell down to the next:
    //
    //   1. the assembled pass (`assembled::joint_hessian`), first-order lanes
    //      over an explicit `F`/`G` adjoint. It declines on an AGQ-routed
    //      shape; wherever the observed factor `A_obs` its adjoint equation
    //      needs is not positive definite — either inside the kernel (the
    //      returned lanes would then be a Fisher approximation with no
    //      detector) or in its own build; on a row on the link's own η bound,
    //      where the score has stopped moving to first order
    //      (`assembled::eta_clamped_rows`); and on a packed-row shape over its
    //      own memory guard
    //      (`assembled::PACKED_ASSEMBLY_MAX_BYTES`). It chunks, so no `m`
    //      refuses it.
    //   2. the hyper-dual pass (`derivative::laplace_hessian`), a packed
    //      second-order pass. It takes the AGQ envelope, and it takes what
    //      rung 1 declined on a shape with a dual kernel. It refuses
    //      `m > MAX_DUAL_N`, which is its lane cap, and it refuses the
    //      packed-row layout outright — there is no dual twin of that
    //      layout's PIRLS kernel.
    //   3. the FD stencil, for whatever rung 2 refused: the packed-row
    //      layout's own step constant through `packed_fd_hessian_cov`, or the
    //      other layouts' `FD_STEP_BASE` grid below it. It stays because a
    //      refused fit must keep a finite-difference Hessian standard error
    //      rather than drop to Rx.
    //
    // `Unsupported` is a ROUTING answer at both exact rungs, not an error:
    // it falls THROUGH. Only `NotConverged` — a real failure at the accepted
    // point — routes to RX.
    //
    // The shape question has TWO owners, each answering its own half, and
    // neither is inlined here: `assembled::assembly_routes` says whether the
    // assembled engine runs, and `derivative::supports_shape` says whether a
    // dual kernel exists for the layout.
    let mut used_exact = false;
    if (super::assembled::assembly_routes(ws, n)
        || super::derivative::supports_shape(ws.layout, &ws.groupings))
        && !ws.fd.force_fd_hessian
    {
        // `std::mem::replace` (faer's `Mat` implements no `Default`, so
        // `mem::take` does not compile; the swapped-in `Mat::zeros(0, 0)`
        // allocates nothing) avoids aliasing: both Hessian entry points take
        // `&mut ws`, so `&mut ws.inference.hess_scratch` cannot be passed alongside it.
        // Replace-and-put-back leaves the field valid on every path, including
        // the `fallback!()` early return, because the put-back precedes the check.
        let mut hess = std::mem::replace(&mut ws.inference.hess_scratch, Mat::zeros(0, 0));
        let mut g = std::mem::take(&mut ws.inference.grad_scratch);
        let st = super::assembled::joint_hessian(
            ws,
            xd,
            y,
            cluster_ids,
            extra_ids,
            p,
            n,
            &mut g,
            &mut hess,
        );
        let st = match st {
            DerivStatus::Ok(v) => DerivStatus::Ok(v),
            DerivStatus::NotConverged => DerivStatus::NotConverged,
            // Rung 2 on its own terms: the assembled pass declined this cell.
            DerivStatus::Unsupported => super::derivative::laplace_hessian(
                ws,
                xd,
                y,
                cluster_ids,
                extra_ids,
                p,
                n,
                &mut g,
                &mut hess,
            ),
        };
        ws.inference.grad_scratch = g;
        ws.inference.hess_scratch = hess;
        match st {
            DerivStatus::Ok(_) => used_exact = true,
            DerivStatus::Unsupported => {} // fall through to the FD arm
            DerivStatus::NotConverged => fallback!(),
        }
    }
    if ws.layout == GlmmLayout::Packed && !used_exact {
        // The packed-row stencil, rung 3 of the ladder above: same tolerance
        // contract as the arm below, its own relative step on every
        // coordinate (`SPARSE_FD_STEP_REL`).
        ws.fd.pirls_tol_override = Some(super::pirls_tol_fd(ws.family));
        if !packed_fd_hessian_cov(ws, xd, y, cluster_ids, extra_ids, m, n) {
            fallback!();
        }
    } else if !used_exact {
        // Rung 3 on the blocked and structured layouts. Two entry conditions
        // reach it, and they are the whole list:
        //
        //   (a) `ws.fd.force_fd_hessian`, which skips the exact branch
        //       outright — the A/B arm the crate's own FD-vs-exact
        //       comparisons run, never set on a fitting path.
        //   (b) a cell BOTH exact rungs declined: the assembled pass on a
        //       clamped mode state, a non-positive-definite observed factor
        //       or an AGQ-routed shape, and the hyper-dual pass on
        //       `m > MAX_DUAL_N`.
        //
        // A third — a layout with no exact engine at all — is empty as the
        // dispatch stands. The only layout `derivative::supports_shape`
        // refuses is the packed-row one, which has the assembled engine and,
        // under it, its own stencil above; and `supports_shape`'s other
        // clause, a structured crossed tail wider than `DUAL_TAIL_MAX`,
        // cannot fire, that cap being the crossed-level limit
        // `fit::classify_design` already routes past into the packed-row
        // layout.
        //
        // Every deviance eval below (f0, the f± stencil, and the fallback's central
        // re-eval) converges PIRLS at the FD-pass tol — see the doc comment.
        // Reset on every exit alongside `warm_seed_active`.
        ws.fd.pirls_tol_override = Some(super::pirls_tol_fd(ws.family));
        // Step construction per `FD_STEP_BASE` (θ absolute, β relative). On toenail
        // (θ̂ = 4.708), a θ-relative step of h_θ = 0.047 puts se(β₀) 4.9e-4 above the
        // converged value; at h_θ = 1e-2 it is 2.2e-5, against a noise floor near
        // h = 2.5e-3.
        for k in 0..m {
            ws.fd.fd_steps[k] = if k < n_theta {
                FD_STEP_BASE
            } else {
                FD_STEP_BASE * ws.fd.fd_saved[k].abs().max(1.0)
            };
        }

        let f0 = fd_eval(ws, &[], &[], xd, y, cluster_ids, extra_ids, n);
        if !f0.is_finite() {
            fallback!();
        }

        // Build the symmetric m×m Hessian into ws.inference.hess_scratch (upper, then mirror).
        // Each grid cell is a pure function of the frozen FD seed (fd_saved, fd_steps,
        // u_seed) — see `joint_hessian_cov`'s doc comment — so per-thread worker
        // workspaces compute bit-identical values in any order. Diagonal cells use a
        // single-step central second difference (no Richardson — the fixture is
        // step-invariant to ~7 sig figs across h ∈ [1e-4, 1e-1], see the doc comment).
        let use_par =
            cfg!(all(feature = "parallel", not(target_arch = "wasm32"))) && ws.parallel_inner;
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
                            let h = fd_hess_entry(
                                wws,
                                i,
                                j,
                                ws_ro.fd.fd_steps[i],
                                ws_ro.fd.fd_steps[j],
                                f0,
                                xd,
                                y,
                                cluster_ids,
                                extra_ids,
                                n,
                            );
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
                    ws.inference.hess_scratch[(i, j)] = h;
                    ws.inference.hess_scratch[(j, i)] = h;
                }
            }
        } else {
            for i in 0..m {
                let hi = ws.fd.fd_steps[i];
                let hii = second_diff(ws, i, hi, f0, xd, y, cluster_ids, extra_ids, n);
                if !hii.is_finite() {
                    fallback!();
                }
                ws.inference.hess_scratch[(i, i)] = hii;
                for j in (i + 1)..m {
                    let hj = ws.fd.fd_steps[j];
                    let hij = mixed_diff(ws, i, j, hi, hj, xd, y, cluster_ids, extra_ids, n);
                    if !hij.is_finite() {
                        fallback!();
                    }
                    ws.inference.hess_scratch[(i, j)] = hij;
                    ws.inference.hess_scratch[(j, i)] = hij;
                }
            }
        }
    }

    // Mixed Gamma: the dispersion is a coordinate of the fitted objective too,
    // so the joint Hessian gains a trailing row for `t = ln φ − ln φ̂`. PIRLS on
    // `prior_wᵢ/(φ̂·eᵗ)` with `u = v·e^{−t/2}` is the fixed-φ̂ problem at
    // `θ·e^{−t/2}` (Λ is linear in θ), so in the coordinates `(ξ, β, t)` with
    // `θ = ξ·e^{t/2}` the objective separates as
    //
    //   F(ξ, β, t) = e^{−t}·P̂(ξ, β) + L̂(ξ, β) + G(ln φ̂ + t)
    //
    // `P̂ = D(û) + ‖û‖²` and `L̂ = log|A|` both at the weights in place here
    // (`prior_wᵢ/φ̂`), `G` = `family::gamma_dispersion_term`. At `t = 0`, `ξ = θ`
    // and the `(ξ, β)` block is the fixed-φ̂ Hessian every rung above built;
    // the new entries are `∂²F/∂γ∂t = −∇P̂` and `∂²F/∂t² = P̂ + G''`. The β
    // block of the inverse does not depend on how the non-β coordinates are
    // parametrized at a stationary point, so it is Cov(β̂) with φ estimated.
    // `∇P̂` is the partial of the penalized deviance at the mode (`û` is its own
    // minimizer), taken by central differences over the same fixed-seed
    // re-solves the stencil uses: it enters only this rank-one cross term, where
    // a step of 1e-4 is far inside every tolerance the SEs are gated at.
    let disp_hess = match disp {
        None => None,
        Some(DispRow::Gamma { d2 }) => {
            ws.fd.pirls_tol_override = Some(super::pirls_tol_fd(ws.family));
            let _ = fd_eval(ws, &[], &[], xd, y, cluster_ids, extra_ids, n);
            let pen_dev0 = penalized_deviance_at_mode(ws, y, n);
            let mut h = Mat::<f64>::zeros(m + 1, m + 1);
            for a in 0..m {
                for b in 0..m {
                    h[(a, b)] = ws.inference.hess_scratch[(a, b)];
                }
            }
            for k in 0..m {
                let step = GAMMA_PEN_GRAD_STEP * ws.fd.fd_saved[k].abs().max(1.0);
                let _ = fd_eval(ws, &[k], &[step], xd, y, cluster_ids, extra_ids, n);
                let up = penalized_deviance_at_mode(ws, y, n);
                let _ = fd_eval(ws, &[k], &[-step], xd, y, cluster_ids, extra_ids, n);
                let dn = penalized_deviance_at_mode(ws, y, n);
                let g = (up - dn) / (2.0 * step);
                if !g.is_finite() {
                    fallback!();
                }
                h[(m, k)] = -g;
                h[(k, m)] = -g;
            }
            h[(m, m)] = pen_dev0 + d2;
            Some(h)
        }
        Some(DispRow::Nb) => {
            // Negative binomial: θ_NB enters the variance function, so there is
            // no scaling identity to separate it. In the absolute
            // `(θ, β, ln θ_NB)` coordinates the fit searches, with
            // `F(γ, t) = D*(γ; θ_NB = eᵗ) − 2·nb_profile_loglik(y, y, eᵗ)` the
            // full objective (the fit's Laplace or AGQ deviance plus the
            // search's own saturated term):
            //
            //   ∂²F/∂t²   = (F(γ̂, t̂+h) − 2F(γ̂, t̂) + F(γ̂, t̂−h))/h²
            //   ∂²F/∂γ∂t  = (∇_γD*(γ̂; t̂+h) − ∇_γD*(γ̂; t̂−h))/(2h)
            //
            // The second line is a central difference in `t` of the EXACT
            // gradient `∇_γD* = F_γ − adj'·G_γ` (Skaug & Fournier 2006; the
            // assembled pass's value part, `assembled::gradient_f64`): the
            // saturated term does not depend on γ, so it drops out. One
            // difference per coordinate pair instead of the four-point
            // stencil of `F` itself: the mixed row costs two mode re-solves
            // (seeded at the `t̂ ± h` modes the diagonal's own evaluations
            // leave behind) and two gradient passes, against `4m` full
            // re-solves, and the gradient's own error enters divided by `h`
            // rather than by `h²`. Falls back to the stencil wherever the
            // assembled gradient declines.
            ws.fd.pirls_tol_override = Some(super::pirls_tol_fd(ws.family));
            let th0 = ws.nb_theta;
            let lt0 = th0.ln();
            let ht = NB_DISP_FD_STEP * lt0.abs().max(1.0);
            let f00 = nb_full_objective(ws, &[], &[], lt0, xd, y, cluster_ids, extra_ids, n);
            let mut gp = vec![0.0; m];
            let mut gm = vec![0.0; m];
            let fp = nb_full_objective(ws, &[], &[], lt0 + ht, xd, y, cluster_ids, extra_ids, n);
            let okp =
                super::assembled::gradient_f64(ws, xd, y, cluster_ids, extra_ids, p, n, &mut gp)
                    .is_some();
            let fm = nb_full_objective(ws, &[], &[], lt0 - ht, xd, y, cluster_ids, extra_ids, n);
            let okm = okp
                && super::assembled::gradient_f64(ws, xd, y, cluster_ids, extra_ids, p, n, &mut gm)
                    .is_some();
            let mut h = Mat::<f64>::zeros(m + 1, m + 1);
            for a in 0..m {
                for b in 0..m {
                    h[(a, b)] = ws.inference.hess_scratch[(a, b)];
                }
            }
            h[(m, m)] = (fp - 2.0 * f00 + fm) / (ht * ht);
            if okm {
                for k in 0..m {
                    let v = (gp[k] - gm[k]) / (2.0 * ht);
                    h[(m, k)] = v;
                    h[(k, m)] = v;
                }
            } else {
                for k in 0..m {
                    let hk = NB_DISP_FD_STEP * ws.fd.fd_saved[k].abs().max(1.0);
                    let mut corner = |dk: f64, dt: f64| {
                        nb_full_objective(
                            ws,
                            &[k],
                            &[dk],
                            lt0 + dt,
                            xd,
                            y,
                            cluster_ids,
                            extra_ids,
                            n,
                        )
                    };
                    let v = (corner(hk, ht) - corner(hk, -ht) - corner(-hk, ht) + corner(-hk, -ht))
                        / (4.0 * hk * ht);
                    h[(m, k)] = v;
                    h[(k, m)] = v;
                }
            }
            ws.nb_theta = th0;
            if (0..=m).any(|k| !h[(m, k)].is_finite()) {
                fallback!();
            }
            Some(h)
        }
    };

    // Invert the joint Hessian; non-PD ⇒ RX fallback. cov = 2·(H⁻¹)_ββ.
    let md = if disp_hess.is_some() { m + 1 } else { m };
    let chol = match crate::linalg::SeqLlt::new(
        disp_hess
            .as_ref()
            .map_or(ws.inference.hess_scratch.as_ref(), |h| h.as_ref()),
    ) {
        Some(c) => c,
        None => fallback!(),
    };
    // Pivot ratio of the β information this covariance comes from. With θ
    // ordered first, rows and columns `n_theta..m` of the factor are the factor
    // of the β block with θ profiled out (any dispersion row comes after them,
    // so it is held fixed). On `ws.beta_scale`'s search design that block is
    // `F` with the fitted design's information `L·FFᵀ·Lᵀ`, so the ratio is read
    // off `L·F` (`BetaScale::map_factor`), on the fitted design's own columns.
    let beta_factor = chol.L().submatrix(n_theta, n_theta, p, p);
    let (pivot, col) = if beta_scaled {
        let mut l = beta_factor.to_owned();
        ws.beta_scale.map_factor(l.as_mut());
        crate::ols::min_pivot_ratio(l.as_ref(), p)
    } else {
        crate::ols::min_pivot_ratio(beta_factor, p)
    };
    (ws.inference.pivot, ws.inference.pivot_col) = (pivot, col as u32);
    let mut inv = Mat::<f64>::identity(md, md);
    chol.solve_in_place(inv.as_mut());
    for a in 0..p {
        for b in 0..p {
            out_cov[(a, b)] = 2.0 * inv[(n_theta + a, n_theta + b)];
        }
    }
    // θ block: the same cov = 2·H_dev⁻¹. The β block above discards it; expose
    // it here (the whole point of paying for the joint Hessian). For a scalar
    // grouping the RE stddev equals θ, so the diagonal's square root is the
    // stddev's SE directly. On mixed Gamma the inverse is in `(ξ, β, t)`, and
    // `θ = ξ·e^{t/2}` maps it back by the delta method: `dθ = dξ + ½·θ̂·dt`, so
    // `Cov(θₖ, θₗ) = C_kl + ½θₗ·C_km + ½θₖ·C_lm + ¼θₖθₗ·C_mm` with `C = 2·H⁻¹`.
    // The NB row is in absolute coordinates already.
    for k in 0..n_theta {
        for l in 0..n_theta {
            ws.inference.theta_cov[(k, l)] = if matches!(disp, Some(DispRow::Gamma { .. })) {
                let (tk, tl) = (ws.fd.fd_saved[k], ws.fd.fd_saved[l]);
                2.0 * (inv[(k, l)]
                    + 0.5 * tl * inv[(k, m)]
                    + 0.5 * tk * inv[(l, m)]
                    + 0.25 * tk * tl * inv[(m, m)])
            } else {
                2.0 * inv[(k, l)]
            };
        }
    }

    // Restore the converged PIRLS state at γ̂ (W̃/û/μ̂/factors): the stencil leaves
    // the LAST perturbed eval's state in ws, and the caller reads ws.pirls.prob/
    // ws.pirls.u AFTER this returns, on every layout (mu_hat and the reported
    // fit) — off the perturbed state rung 23's stddev once came out ~2e-3 high. Same central re-eval the RX fallback uses; the empty
    // perturbation evaluates exactly at γ̂ (fd_saved).
    let _ = fd_eval(ws, &[], &[], xd, y, cluster_ids, extra_ids, n);

    ws.params[..m].copy_from_slice(&ws.fd.fd_saved[..m]);
    // That re-eval lands on û(γ̂) again but at the FD-pass tol, which is never
    // looser than the fit's, so it is at least as converged as the mode this call
    // was handed. Take the handed one back verbatim, so a second call on the same
    // workspace seeds from the same vector and returns the same bits. `ws.pirls.prob`
    // is put back beside `ws.pirls.u` (`fd_saved_prob`, snapshotted above) so the pair
    // the caller reads comes from one solve, not this re-eval's separate one.
    restore_fd_mode!();
    ws.fd.warm_seed_active = false; // never leak the FD seed into a later fit / BOBYQA
    ws.fd.pirls_tol_override = None; // nor the FD-pass tol
    FdHessianStatus::Ok
}

/// Central-difference step of the NB `ln θ_NB` row in `joint_hessian_cov_at`,
/// relative with a unit floor on every coordinate: the row at this step (the
/// three-point diagonal, the mixed entries as a central difference of the
/// exact gradient or, where that declines, the four-point mixed difference)
/// sits inside the SE gate's 1e-3 against a brute-force Hessian of the full
/// objective (`nb_hessian_se_matches_fd_of_the_full_objective`).
const NB_DISP_FD_STEP: f64 = 2e-3;

/// The NB fit's full objective at `fd_saved + Σ deltaₖ·e_{coordₖ}` and
/// `θ_NB = e^{ln_theta}`: the (Laplace or AGQ) deviance plus the search's own
/// `−2·nb_profile_loglik(y, y, θ_NB, w)`, the term `glmm::fit_glmm` adds per
/// evaluation, read off the count groups `fit_glmm` filled for this `y`
/// (`ws.nb_groups`). Leaves `ws.nb_theta` at the probed value; the caller
/// restores it.
#[allow(clippy::too_many_arguments)]
fn nb_full_objective(
    ws: &mut GlmmWorkspace,
    coords: &[usize],
    deltas: &[f64],
    ln_theta: f64,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) -> f64 {
    debug_assert!(
        n == 0 || !ws.nb_groups.is_empty(),
        "NB count groups not filled"
    );
    ws.nb_theta = ln_theta.exp();
    let dev = fd_eval(ws, coords, deltas, x, y, cluster_ids, extra_ids, n);
    dev - 2.0 * crate::fit::nb_saturated_loglik_grouped(&ws.nb_groups, ws.nb_theta)
}

/// Central-difference step of `∇P̂` in `joint_hessian_cov_at`'s mixed-Gamma
/// dispersion row, relative with a unit floor (`h_k = step·max(1, |γ̂_k|)`).
const GAMMA_PEN_GRAD_STEP: f64 = 1e-4;

/// `P̂ = Σᵢ wᵢ·dᵢ(yᵢ, μ̂ᵢ) + ‖û‖²` at the mode the last `fd_eval` left in the
/// workspace, `wᵢ` the prior weights in place (`prior_wᵢ/φ̂` inside
/// `joint_hessian_cov`'s Gamma arm).
fn penalized_deviance_at_mode(ws: &GlmmWorkspace, y: &[f64], n: usize) -> f64 {
    let d: f64 = (0..n)
        .map(|i| {
            ws.prior_w[i]
                * crate::family::dev_resid_at(
                    ws.family,
                    ws.nb_theta,
                    y[i],
                    ws.pirls.eta[i],
                    ws.pirls.prob[i],
                )
        })
        .sum();
    let kk = ws.k.max(1);
    d + ws.pirls.u[..kk].iter().map(|v| v * v).sum::<f64>()
}

/// C = X'W̃X (p×p), full matrix, via the W∘X GEMM scratch `ws.wx` (rebuilt fresh
/// from `ws.pirls.w` each call). Mirrors the `BetaStep::Profile` xtwx GEMM the
/// PIRLS kernels run with this same construction.
fn xtwx_fill(ws: &mut GlmmWorkspace, x: MatRef<f64>, n: usize) {
    let p = ws.p;
    for c in 0..p {
        for i in 0..n {
            ws.wx[(i, c)] = ws.pirls.w[i] * x[(i, c)];
        }
    }
    faer::linalg::matmul::matmul(
        ws.border.xtwx.as_mut(),
        faer::Accum::Replace,
        x.subrows(0, n).transpose(),
        ws.wx.as_ref().subrows(0, n),
        1.0,
        faer::Par::Seq,
    );
}

/// Packed-row Schur fill: `X'W̃M`, `A⁻¹M'W̃X` via the `k×k` `ws.packed.a`
/// LLT, and `ws.border.schur = X'W̃X − X'W̃M·A⁻¹M'W̃X` (formed by
/// [`schur_from_residuals`]) — the Rx (closed-form Schur) fixed-effect
/// information at the converged state. Reads `ws.packed.a`,
/// the packed `M` rows (`ws.packed.m_cols`/`m_vals`), `ws.pirls.w`, and `x` via
/// arg. Returns false on a non-PD `ws.packed.a`.
///
/// PIRLS's `BetaStep::Profile` β-Schur border step builds the same
/// B' = X'WM, T = A⁻¹B and S_β each iteration (with that iteration's own W and
/// factor, through `pirls::residual_border`), then additionally solves
/// δβ = S_β⁻¹·(X'ρ − B'δu₀) and folds `u_joint = u_new − T·δβ` back into the
/// conditional-mode iterate — the joint (u, β) Newton step within one PIRLS solve.
fn packed_schur_fill(ws: &mut GlmmWorkspace, x: MatRef<f64>, n: usize) -> bool {
    let (k, p, width) = (ws.k, ws.p, ws.packed.width);
    // X'W̃M (p×k) by per-row scatter over the packed nonzeros.
    for r in 0..p {
        for c in 0..k {
            ws.border.xtwm[(r, c)] = 0.0;
        }
    }
    for i in 0..n {
        let wi = ws.pirls.w[i];
        let base = i * width;
        for r in 0..p {
            let xw = x[(i, r)] * wi;
            for t in base..base + width {
                ws.border.xtwm[(r, ws.packed.m_cols[t] as usize)] += xw * ws.packed.m_vals[t];
            }
        }
    }
    let ac = match crate::linalg::SeqLlt::new(ws.packed.a.as_ref()) {
        Some(c) => c,
        None => return false,
    };
    for r in 0..k {
        for c in 0..p {
            ws.border.ainv_mtwx[(r, c)] = ws.border.xtwm[(c, r)];
        }
    }
    ac.solve_in_place(ws.border.ainv_mtwx.as_mut());
    for i in 0..n {
        let base = i * width;
        for c in 0..p {
            let mut r = x[(i, c)];
            for t in base..base + width {
                r -= ws.packed.m_vals[t] * ws.border.ainv_mtwx[(ws.packed.m_cols[t] as usize, c)];
            }
            ws.wx[(i, c)] = r;
        }
    }
    schur_from_residuals(
        |i| ws.pirls.w[i],
        ws.wx.as_ref().subrows(0, n),
        ws.border.ainv_mtwx.as_ref().subrows(0, k),
        &mut ws.border.schur,
    );
    true
}

/// Blocked Schur fill (no-extras path): reconstruct mᵢ = Λ_p'·zᵢ per row to build
/// the per-cluster coupling X'W̃M (into `ws.border.xtwm` columns
/// `f·q_p..`), then solve `A_f T_f = (M'W̃X)_f` per block by REUSING the factored
/// `ws.pirls.a_blocks` the converged blocked PIRLS left behind (W̃ in `ws.pirls.w`, Λ̂ in
/// `ws.pirls.lam`), and `ws.border.schur = X'W̃X − Σ_f (X'W̃M)_f·T_f` (formed by
/// [`schur_from_residuals`]). Only the trailing `p×p` Schur LLT (done by the
/// common code after this) stays dense. Returns false if a stored block is not
/// usable (defensive — the PIRLS already proved them PD).
fn blocked_schur_fill(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    xd: MatRef<f64>,
    cluster_ids: &[u32],
    n: usize,
) -> bool {
    let (p, q, s) = (ws.p, ws.groupings.primary_q, ws.groupings.n_primary);
    let k = q * s;
    // X'W̃M, blocked: zero then scatter the q_p coupling columns per row.
    for r in 0..p {
        for c in 0..k {
            ws.border.xtwm[(r, c)] = 0.0;
        }
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let m_row = blocked_m_row(ws, x, i);
        let wi = ws.pirls.w[i];
        for r in 0..p {
            let xw = xd[(i, r)] * wi;
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                ws.border.xtwm[(r, f * q + c)] += xw * m_row[c];
            }
        }
    }
    // T_f = A_f⁻¹ (M'W̃X)_f, per block, reusing the stored factor; ainv_mtwx rows
    // f·q_p.. hold T_f. (M'W̃X)_f[c, col] = (X'W̃M)_f[col, c] = ws.border.xtwm[(col, f·q+c)].
    for f in 0..s {
        let ablk = f * q * q;
        for col in 0..p {
            let mut rhs = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                rhs[c] = ws.border.xtwm[(col, f * q + c)];
            }
            glmm_block_solve(&ws.pirls.a_blocks[ablk..ablk + q * q], q, &mut rhs[..q]);
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                ws.border.ainv_mtwx[(f * q + c, col)] = rhs[c];
            }
        }
    }
    // Residual rows rᵢ = xᵢ − T_fᵀmᵢ. Exact: A is block-diagonal, so the per-block
    // solves above equal the full A⁻¹M'W̃X, and row i couples to its cluster's
    // q columns only.
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let m_row = blocked_m_row(ws, x, i);
        for col in 0..p {
            let mut r = xd[(i, col)];
            #[allow(clippy::needless_range_loop)]
            for c in 0..q {
                r -= m_row[c] * ws.border.ainv_mtwx[(f * q + c, col)];
            }
            ws.wx[(i, col)] = r;
        }
    }
    schur_from_residuals(
        |i| ws.pirls.w[i],
        ws.wx.as_ref().subrows(0, n),
        ws.border.ainv_mtwx.as_ref().subrows(0, k),
        &mut ws.border.schur,
    );
    true
}

/// Row `i` of `M = ZΛ` on the blocked layout: its cluster's `q_p` entries.
fn blocked_m_row(ws: &GlmmWorkspace, x: MatRef<f64>, i: usize) -> [f64; crate::lmm::MAX_PRIMARY_Q] {
    let q = ws.groupings.primary_q;
    let mut m_row = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
    #[allow(clippy::needless_range_loop)]
    for c in 0..q {
        let mut acc = 0.0;
        for rr in c..q {
            // The one Z read on this path that does not come from `ws.z_buf`
            // (`workspace::fill_z_f64`, change together). Indexed by the Λ
            // ROW `rr`, not the column `c`.
            let zr = if rr == 0 {
                1.0
            } else {
                x[(i, ws.groupings.primary_slope_cols[rr - 1])]
            };
            acc += zr * ws.pirls.lam[rr * q + c];
        }
        m_row[c] = acc;
    }
    m_row
}

/// Structured Schur fill (intercept-only crossed/nested path — `classify_design`
/// routes slope-carrying extras to `Solver::Sparse` for every family): builds
/// `X'W̃M` (p×k, by per-row scatter into each row's core + crossed
/// columns), then applies `A⁻¹` to each of the `p` columns of `M'W̃X` by REUSING
/// the core-block + Schur factors the converged structured PIRLS left in
/// `ws.structured.{core_blocks, schur_blk, coupling}` (via `structured_ainv_solve`), and
/// `ws.border.schur = X'W̃X − X'W̃M·(A⁻¹M'W̃X)` (formed by [`schur_from_residuals`]).
/// Mirrors `blocked_schur_fill`; the only difference is the `A⁻¹` apply uses
/// the block+Schur back-substitution instead of per-block solves alone. Returns false on nothing (the factors were already
/// proven PD by the PIRLS) — kept `-> bool` to match the dispatch arms.
fn structured_schur_fill(
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
    // X'W̃M: zero then scatter each row's core + crossed columns. Reads the PACKED
    // M nonzeros (`m_core_buf` core slice + `cross_*`/`n_cross` crossed entries) the
    // converged re-eval's `build_packed_m` left behind — a dense `M` is no
    // longer maintained on the structured path.
    //
    // This is the one `f64` reader of the `cross_col`/`n_cross`/`cross_val`
    // triple that does not evaluate the deviance itself, so it depends on its
    // callers having done so — `joint_hessian_cov`'s `fallback!()` central
    // `fd_eval`, or `fit_glmm_ws`'s pinned re-eval. It must, because the
    // pattern half of the triple is shared with the dual kernel while
    // `cross_val` is dual-private (`derivative::run_gradient`'s `pattern`
    // parameter doc comment owns the invariant): at a θ̂ with a pinned crossed
    // grouping a dual call leaves `cross_col`/`n_cross` WIDER than the last
    // `f64` `cross_val`. Reading it un-refreshed does not panic — it picks up
    // a stale θ from an earlier optimizer trial and returns a plausible wrong
    // SE.
    for r in 0..p {
        for c in 0..k {
            ws.border.xtwm[(r, c)] = 0.0;
        }
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let wi = ws.pirls.w[i];
        let cbase = i * g_cap;
        let ncz = ws.pattern.n_cross[i] as usize;
        for r in 0..p {
            let xw = x[(i, r)] * wi;
            for local in 0..qc {
                ws.border.xtwm[(r, core_col(f, local))] +=
                    xw * ws.structured.m_core_buf[i * qc + local];
            }
            for z in 0..ncz {
                let b = ws.pattern.cross_col[cbase + z] as usize;
                ws.border.xtwm[(r, k_family + b)] += xw * ws.structured.cross_val[cbase + z];
            }
        }
    }
    // ainv_mtwx[:, c] = A⁻¹ (M'W̃X)[:, c], one fixed-effect column at a time.
    for c in 0..p {
        for f in 0..s {
            for local in 0..qc {
                ws.pirls.a_rhs[f * qc + local] = ws.border.xtwm[(c, core_col(f, local))];
            }
        }
        for b in 0..e {
            ws.pirls.a_rhs[k_family + b] = ws.border.xtwm[(c, k_family + b)];
        }
        structured_ainv_solve(
            &ws.groupings,
            &ws.structured.core_blocks,
            &ws.structured.coupling,
            &ws.structured.schur_blk,
            &ws.pattern.coup_cols,
            &ws.pattern.coup_ptr,
            // force_dense → ss fold: see the structured_ainv_solve calls in
            // pirls_solve_blocked_extras (pirls/blocked_extras.rs).
            if ws.pattern.force_dense_schur {
                None
            } else {
                ws.pattern.structured_schur.as_mut()
            },
            &mut ws.pirls.a_rhs,
        );
        for f in 0..s {
            for local in 0..qc {
                ws.border.ainv_mtwx[(core_col(f, local), c)] = ws.pirls.a_rhs[f * qc + local];
            }
        }
        for b in 0..e {
            ws.border.ainv_mtwx[(k_family + b, c)] = ws.pirls.a_rhs[k_family + b];
        }
    }
    // Residual rows rᵢ = xᵢ − Tᵀmᵢ over the same packed nonzeros the scatter
    // above read.
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let cbase = i * g_cap;
        let ncz = ws.pattern.n_cross[i] as usize;
        for c in 0..p {
            let mut r = x[(i, c)];
            for local in 0..qc {
                r -= ws.structured.m_core_buf[i * qc + local]
                    * ws.border.ainv_mtwx[(core_col(f, local), c)];
            }
            for z in 0..ncz {
                let b = ws.pattern.cross_col[cbase + z] as usize;
                r -= ws.structured.cross_val[cbase + z] * ws.border.ainv_mtwx[(k_family + b, c)];
            }
            ws.wx[(i, c)] = r;
        }
    }
    schur_from_residuals(
        |i| ws.pirls.w[i],
        ws.wx.as_ref().subrows(0, n),
        ws.border.ainv_mtwx.as_ref().subrows(0, k),
        &mut ws.border.schur,
    );
    true
}
