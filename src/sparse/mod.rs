//! Sparse-Z LMM solver. A second Gaussian-mixed path that factors
//! `Λ'Z'ZΛ + I` with a two-level blocked Cholesky specialized to this
//! path's structural class — primary(+nested) FAMILIES are block-diagonal
//! (levels never co-occur across families) and ALL crossed-extra columns form
//! one tail (dense for `e ≤ TAIL_SPARSE_MIN`, fill-reducing sparse above) —
//! lifting the `MAX_*` caps that bound the dense no-Z tail.
//! Mirrors `lmm::reml_deviance` (`lmm.rs:1396`) one level down; validated
//! against it by the both-paths cross-check (`sparse` tests + `glmm/tests.rs`).
//!
//! Split into this LMM half (`mod.rs`) and the GLMM half (`glmm.rs`); test
//! code lives in `tests.rs`. `mod.rs` re-exports the GLMM half's externally
//! consumed items so `crate::sparse::X` paths are unaffected by the split.
//!
//! On designs outside the dense-solver envelope, these modules return a
//! NaN-filled `Fit { converged: false, ... }` instead of panicking — tested by
//! `fit_over_envelope_non_gaussian_never_panics`. The dense/sparse routing
//! decision is made by `fit::classify_design` (see `fit/mod.rs:609`).
//
// `SymbolicCholesky` at module level serves `logdet_llt`, shared by the GLMM
// sparse-Schur PIRLS path (`glmm/pirls.rs`) and the LMM sparse-tail branch
// (`SparseTail`); the small-e LMM eval loop stays faer-sparse-free (blocked
// kernel with the dense tail).
use crate::lmm::LmmGroupings;
use bobyqa::Status;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{
    cholesky_in_place, cholesky_in_place_scratch, LltRegularization,
};
use faer::mat::AsMatMut;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholesky,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, Mat, MatRef, Par, Side, Spec};

mod glmm;
// FD-Hessian noise-margin measurement (`#[ignore]`d, not a gate). Lives under
// `sparse` rather than beside the dense FD code because it drives BOTH paths and
// the sparse deviance evaluator (`glmm::sparse_glmm_deviance`) is private to
// this module tree; the dense side's evaluator is `pub(crate)` and reachable
// from anywhere.
#[cfg(all(test, feature = "formula"))]
mod fd_margin;
#[cfg(test)]
mod tests;

pub(crate) use glmm::{fit_glmm_nb_sparse, fit_glmm_sparse};

/// Refuse floor for the sparse-LMM rank guard, on the scale-invariant
/// per-column pivot ratio of the augmented Schur factor's fixed block
/// ([`crate::ols::min_pivot_ratio`]). Calibrated 2026-07-31 against the same
/// 1-ULP perturbation sweep as the dense route, and deliberately ~600× looser:
/// this path obeys the same `betaRel ≈ 1e-15 / pivot` law with a ~500× worse
/// constant and a hard noise floor at 5e-7, measured 2.7e-4 at pivot 9.7e-11.
/// Sharing the dense `1e-12` here would accept fits whose β̂ has no digits left.
const PIVOT_MIN: f64 = 6e-10;
#[cfg(test)]
use glmm::{sparse_glmm_deviance, SparseGlmmWorkspace};

/// Sparse-Z LMM end-to-end fit: BOBYQA over θ with the
/// sparse profiled-REML deviance, then β̂/σ̂²/SE/varcorr recovered once at θ̂. A
/// superset of the dense NoZ `fit_mle` — on an in-envelope design it reproduces
/// that fit to machine precision (`fit_mle_sparse_matches_noz_in_envelope`).
///
/// Mirrors `fit_lmm` (`lmm.rs:1859`) onto the sparse workspace: the θ seed/bounds
/// and BOBYQA schedule come from `crate::lmm::sparse_lmm_seed` (byte-identical to
/// the NoZ path), and the recovery reads the augmented Schur factor `L`
/// (`sparse_schur_factor`) exactly as `fit_lmm` reads its `fit.factor`. `aliased`
/// is all-false (rank-deficiency is salvaged upstream in `fit_warm` before routing).
/// On failure (non-convergence, rank-deficiency, or numeric failure), returns a
/// NaN-filled `Fit { converged: false, ... }` constructed inline (lines 173-194).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_mle_sparse(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &crate::ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    start: Option<&crate::StartValues>,
    opts: &crate::FitOptions,
) -> crate::Fit {
    let re = model
        .re
        .as_ref()
        .expect("fit_mle_sparse requires a mixed model (re: Some)");
    // Slope x-column indices (primary + per extra grouping), exactly as `fit_mle`
    // derives them for the NoZ workspace.
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();
    let mut g = LmmGroupings::from_cluster_spec_ext(model, n, &slope_cols, &extra_slope_cols);

    // Row-major f64 `x` viewed column-agnostically as an n×p faer MatRef (Z/Gram
    // builders index it as `x[(i, j)]`).
    let xm = MatRef::from_row_major_slice(x, n, p);
    // Before any Z entry is emitted: `for_each_z_entry` divides every slope value
    // by its RE column's internal scale, so the scales must be current for THIS
    // design first (mirrors `accumulate_lmm_rows` on the dense route).
    g.set_slope_scales(xm, opts.weights.as_deref());
    let g = g;
    // WLS-style √wᵢ pre-scaling, same convention as `fit_mle`'s dense path
    // (`add_rows_multi`'s `weights` arg): computed once here, threaded through
    // every z-emission and raw x/y read in `SparseLmmWorkspace::new`.
    let sqrt_w: Option<Vec<f64>> = opts
        .weights
        .as_ref()
        .map(|w| w.iter().map(|v| v.sqrt()).collect());
    // Identity-link offset as the exact y-shift before Gram accumulation —
    // mirrors `fit_mle` (dense); change together.
    let y_shifted: Vec<f64>;
    let y_eff: &[f64] = match &opts.offset {
        Some(o) => {
            y_shifted = y.iter().zip(o).map(|(&yi, &oi)| yi - oi).collect();
            &y_shifted
        }
        None => y,
    };
    let mut ws = SparseLmmWorkspace::new(
        &g,
        xm,
        cluster_ids,
        extra_ids,
        y_eff,
        n,
        p,
        sqrt_w.as_deref(),
    );

    // θ seed + per-component boxes + solver — topology-only, byte-identical to the
    // NoZ path (the superset property depends on this).
    let (mut solver, mut theta, lower, upper) = crate::lmm::sparse_lmm_seed(&g);
    // Cold start = blind seed (diagonals THETA0, off-diagonals 0 — mirror
    // `fit_lmm`'s cold arm, see the basin rationale there); a warm start
    // clamps only its diagonal coordinates to the truth floor, off-diagonals
    // verbatim (mirror `fit_lmm` `lmm.rs:1888-1900`).
    match start {
        Some(s) => {
            debug_assert_eq!(s.theta.len(), theta.len());
            // Forward map into the solver's internal RE scale before the floor —
            // mirror `fit_lmm`'s warm arm; change together.
            let sc = g.theta_row_scales();
            for ((t, &v), &f) in theta.iter_mut().zip(&s.theta).zip(sc.iter()) {
                *t = v * f;
            }
            for &i in g.diagonal_theta() {
                theta[i] = theta[i].max(crate::lmm::THETA_TRUTH_FLOOR);
            }
        }
        None => {
            for t in theta.iter_mut() {
                *t = 0.0;
            }
            for &i in g.diagonal_theta() {
                theta[i] = crate::lmm::THETA0;
            }
        }
    }
    let out = solver.minimize(
        |xs| sparse_reml_deviance(xs, &mut ws),
        &mut theta,
        &lower,
        &upper,
    );
    debug_assert!(out.status != Status::InvalidArgs);
    // The plateau policy, mirrored from `fit_lmm` (`lmm.rs`): a `MaxFunReached`
    // cap-out reports its finite endpoint with `converged == false` rather than
    // NaN-filling — it runs the same pin + rank-guard + recovery as `Converged`.
    // `ModelDegenerate` has no endpoint worth reporting and NaN-fills below.
    let converged_status = matches!(out.status, Status::Converged);
    let has_endpoint = matches!(out.status, Status::Converged | Status::MaxFunReached);

    // Per-component deterministic pin: every DIAGONAL variance component ≤ PIN_THETA
    // collapses to exactly 0 so tau2/varcorr reflect the boundary (mirror `fit_lmm`
    // `lmm.rs:1917-1927`). Applied to any reported endpoint, but the mask (⇒
    // `singular`) only latches when the fit actually converged — a capped
    // endpoint is reported as a point, not accepted onto the boundary. The bit
    // index is the position in `diagonal_theta()` order, not the θ index: that
    // is the order `fit::common::pinned_flags` reshapes against the varcorr
    // blocks (mirror `fit_lmm` and `glmm::fit_glmm`, which build the same mask).
    // This route is the one that takes the widest designs, so the shift is
    // guarded — past 64 components `pinned` still latches and the extra
    // components go unnamed, rather than the shift overflowing.
    let mut pinned = false;
    let mut pinned_components = 0u64;
    if has_endpoint {
        for (kk, &ti) in g.diagonal_theta().iter().enumerate() {
            if theta[ti] <= crate::lmm::PIN_THETA {
                theta[ti] = 0.0;
                if converged_status {
                    pinned = true;
                    if kk < u64::BITS as usize {
                        pinned_components |= 1u64 << kk;
                    }
                }
            }
        }
    }

    // Final eval at θ̂ (post-pin) → augmented Schur factor L in `ws.factor`;
    // rank-guard the p×p fixed block (mirror `fit_lmm` `lmm.rs:1932-1940`).
    let factor_ok = has_endpoint && sparse_schur_factor(&theta, &mut ws).is_some();
    let degenerate = if factor_ok {
        crate::ols::min_pivot_ratio(ws.factor.as_ref(), p).0 < PIVOT_MIN
    } else {
        true
    };
    let converged = converged_status && !degenerate;
    let has_recovery = has_endpoint && !degenerate;

    if !has_recovery {
        return crate::Fit {
            beta: vec![f64::NAN; p],
            se: vec![f64::NAN; p],
            vcov: crate::fit::nan_vcov(p),
            tau2: theta.iter().map(|_| f64::NAN).collect(),
            dispersion: f64::NAN,
            diagnostics: crate::Diagnostics::from_flags(false, false, p),
            varcorr: vec![],
            stddev_se: vec![],
            n_eval: out.n_eval,
            deviance: f64::NAN,
            loglik: f64::NAN,
            df: 0,
            reml: true,
            fitted: vec![],
            ranef: vec![],
            ranef_levels: vec![],
        };
    }

    // Accepted objective at θ̂ post-pin: no `dev` local survives from the BOBYQA
    // loop here (unlike `fit_lmm`'s `dev`), so re-evaluate at the pinned θ — a
    // second Schur factor, but only once per fit (not the hot loop). This is
    // also the evaluation the conditional-mode recovery rides on: arming it here
    // means the per-family factors it needs are kept for this call alone.
    ws.arm_recovery();
    let dev = sparse_reml_deviance(&theta, &mut ws);

    let l = &ws.factor;
    let sigma_sq = {
        let lyy = l[(p, p)];
        lyy * lyy / ((n - p) as f64)
    };

    // β̂: backward solve L_XXᵀ β̂ = l_yX, l_yX[j] = L[(p, j)] (mirror `lmm.rs:1961-1967`).
    let mut beta = vec![0.0f64; p];
    for j in (0..p).rev() {
        let mut acc = l[(p, j)];
        for k in (j + 1)..p {
            acc -= l[(k, j)] * beta[k];
        }
        beta[j] = acc / l[(j, j)];
    }

    // Var(β̂_j) = σ̂²·‖L_XX⁻¹e_j‖² per target; SE = √Var (mirror `lmm.rs:1972-1993`
    // + `fit_mle` `fit.rs:708-714`). Non-target slots stay NaN.
    let mut se = vec![f64::NAN; p];
    let mut u = vec![0.0f64; p];
    for &tj in &opts.target_indices {
        let tj = tj as usize;
        for v in u.iter_mut() {
            *v = 0.0;
        }
        for i in 0..p {
            let b_i = if i == tj { 1.0 } else { 0.0 };
            let mut acc = b_i;
            for k in 0..i {
                acc -= l[(i, k)] * u[k];
            }
            u[i] = acc / l[(i, i)];
        }
        let norm_sq: f64 = u.iter().map(|v| v * v).sum();
        let vd = sigma_sq * norm_sq;
        if vd.is_finite() && vd >= 0.0 {
            se[tj] = vd.sqrt();
        }
    }

    // Var(β̂) = σ̂²·(L_XX L_XX')⁻¹ over the same target block, off the same
    // factor the per-target solve above walks — `se` is its diagonal.
    let vcov = crate::fit::vcov_from_chol(l.as_ref(), p, &opts.target_indices, sigma_sq);

    // tau2[k] = θ̂[k]²·σ̂²; varcorr = vech(σ̂²·Λ̂Λ̂') per grouping — the path-independent
    // assembly shared with `fit_mle` (`fit.rs`).
    // θ̂ is in the solver's internal RE units; the Λ-row scales divide it back into
    // the design's own units before it is squared (mirror `lmm_view_to_fit`).
    let theta_scales = g.theta_row_scales();
    let tau2: Vec<f64> = theta
        .iter()
        .zip(theta_scales.iter())
        .map(|(&t, &s)| (t / s) * (t / s) * sigma_sq)
        .collect();
    let varcorr = crate::fit::assemble_varcorr(&theta, &g, sigma_sq);

    // Same −Σlog wᵢ deviance-constant convention as `fit_mle` (`fit.rs`,
    // Task 5): the weighted Gaussian log-density's +½Σlog wᵢ per row, on the
    // −2ℓ scale, added post-optimization (θ-independent — argmin unchanged).
    let dev = match &opts.weights {
        Some(w) => dev - w.iter().map(|v| v.ln()).sum::<f64>(),
        None => dev,
    };

    // Which components pinned, in the layout the wrappers iterate. Built here
    // rather than inside `from_flags` because that helper also serves the
    // NaN-fill returns, which have no varcorr to place bits against.
    let pinned_grid = crate::fit::pinned_flags(pinned_components, &varcorr);

    // Conditional modes and the per-row means they unlock, off the factors the
    // evaluation above kept. Gated on `converged` like the dense path: a mode at
    // a non-converged θ̂ is not a BLUP of anything.
    let (fitted, ranef, ranef_levels) = match converged.then(|| sparse_recover_u(&ws, &beta)) {
        Some(Some(u)) => {
            let ranef = crate::fit::assemble_ranef_sparse(&theta, &g, &u);
            let fitted = crate::fit::lmm_fitted(
                x,
                n,
                p,
                &beta,
                &ranef,
                &g,
                cluster_ids,
                extra_ids,
                opts.offset.as_deref(),
            );
            (fitted, ranef, crate::fit::ranef_level_counts(&g))
        }
        _ => (vec![], vec![], vec![]),
    };

    let mut fit = crate::Fit {
        beta,
        se,
        vcov,
        tau2,
        dispersion: sigma_sq,
        // This route records no pivot (it REFUSES below `PIVOT_MIN` rather than
        // flagging), so `notes` stays empty and `boundary` is back-derived from
        // `pinned` — see `Diagnostics::from_flags`. `pinned` itself IS real
        // here: the pin loop above knows exactly which components collapsed.
        diagnostics: crate::Diagnostics {
            pinned: pinned_grid,
            ..crate::Diagnostics::from_flags(converged, pinned, p)
        },
        varcorr,
        stddev_se: vec![],
        n_eval: out.n_eval,
        deviance: dev,
        // REML criterion off the weight-corrected deviance (mirrors `fit_mle`'s
        // loglik).
        loglik: crate::fit::lmm_loglik(dev, n, p),
        df: p + theta.len() + 1,
        reml: true,
        fitted,
        ranef,
        ranef_levels,
    };
    fit.diagnostics.singular =
        fit.diagnostics.singular || fit.has_negligible_component(&crate::fit::re_scale_grid(&g));
    fit
}

/// `log det(A) = 2·Σ_j log L[j,j]` for the LLT factor, reading the diagonal
/// straight out of `l_values` per symbolic arm (faer 0.24 exposes no diagonal
/// accessor on `LltRef`): simplicial via the CSC `col_ptr()`/`row_idx()`
/// layout, supernodal via each supernode's dense column-major panel, whose top
/// `ncols×ncols` block is L's triangular diagonal block. Handles whichever arm
/// the construction sites' `SupernodalThreshold::AUTO` picked
/// (`SparseLmmWorkspace::new`, `StructuredSchur::new`/`clone_scratch` —
/// `glmm/workspace.rs`). Returns `+INFINITY` if any diagonal is
/// non-positive/non-finite (mirrors the dense evaluators' non-PD sentinel,
/// `lmm.rs:1530–1561`).
pub(crate) fn logdet_llt(symbolic: &SymbolicCholesky<usize>, l_values: &[f64]) -> f64 {
    use faer::sparse::linalg::cholesky::{supernodal::SupernodalLltRef, SymbolicCholeskyRaw};
    // `ljj <= 0.0` handles negative and zero; `!is_finite()` covers NaN and ±Inf
    // (`NaN.is_finite()` = false). Shared non-PD sentinel for both arms.
    let mut acc = 0.0f64;
    let mut push = |ljj: f64| -> bool {
        if ljj <= 0.0 || !ljj.is_finite() {
            return false;
        }
        acc += ljj.ln();
        true
    };
    match symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(simp) => {
            let col_ptr = simp.col_ptr();
            let row_idx = simp.row_idx();
            let n = col_ptr.len() - 1;
            for j in 0..n {
                let mut ljj = f64::NAN;
                for k in col_ptr[j]..col_ptr[j + 1] {
                    if row_idx[k] == j {
                        ljj = l_values[k];
                        break;
                    }
                }
                if !push(ljj) {
                    return f64::INFINITY;
                }
            }
        }
        SymbolicCholeskyRaw::Supernodal(sup) => {
            let llt = SupernodalLltRef::new(sup, l_values);
            for si in 0..sup.n_supernodes() {
                let panel = llt.supernode(si).val();
                for j in 0..panel.ncols() {
                    if !push(panel[(j, j)]) {
                        return f64::INFINITY;
                    }
                }
            }
        }
    }
    2.0 * acc
}

// ---------------------------------------------------------------------------
// SparseLmmWorkspace — one symbolic Z'Z Cholesky factor + reusable buffers
// ---------------------------------------------------------------------------

/// Dense-tail cutover: crossed tails `e ≤ TAIL_SPARSE_MIN` keep the tuned dense
/// L21/syrk/LLT branch, larger tails take the fill-reducing sparse factor. A
/// sparse factorization loses to the dense tail at small e (assembly
/// indirection, no BLAS-3); the existing sparse corpus tops out at e = 56 (the
/// rung-7 hot path is e = 32), so 128 keeps every golden-pinned cell on the
/// profile-tuned dense code, and the boundary region is µs-scale either way —
/// the exact value is uncritical until a locked-clock crossover sweep (which
/// must vary k_family as well as e: the dense branch's dominant small-e cost is
/// the O(e²·k_family) syrk downdate, not e³/3).
pub(crate) const TAIL_SPARSE_MIN: usize = 128;

/// Family-downdate dense-route trigger: take `FamDowndate::Dense` when one
/// gather over the CSC pattern plus the per-eval `e²` zero costs less than
/// `DD_DENSE_BETA ×` the scatter's per-eval entry count `Σ_f e_f(e_f+1)/2`.
/// Both sides are θ-independent, so the comparison is made once at construction.
///
/// 0.5 — i.e. route dense once the scatter applies more than twice as many
/// entries as the gather reads. Set by the 2026-08-24 locked-clock bracket over
/// the seven sparse-tail cells available (six `cross8` grid cells plus InstEval
/// in the ordered orientation), measuring per-eval cost on both routes with the
/// route forced. Writing `R = Σ_f e_f(e_f+1)/2 / (nnz + e²)`, dense/scatter
/// per-eval time came out 1.055 at R = 0.55, 1.045 at R = 0.92, 0.960 at
/// R = 4.99, 0.825 at R = 12.2, 0.907 at R = 50.1, 0.753 at R = 93.7, 0.752 at
/// R = 122.3. So the crossover lies in (0.92, 4.99) — no corpus shape lands
/// inside that interval — and the threshold `R > 1/β = 2` is placed at its
/// geometric middle, ~2.2× clear of the nearest measured loss and ~2.5× clear of
/// the nearest measured win. Both misroutings near the boundary cost ≤5%.
const DD_DENSE_BETA: f64 = 0.5;

/// Hard cap on the dense accumulator: `e² · 8 B ≤ 256 MB` (e ≲ 5793). Above it
/// the buffer is the wrong trade whatever the entry counts say, and the
/// allocation itself is the objection.
const DD_DENSE_MAX_BYTES: usize = 256 << 20;

// Test-only override: force the sparse-tail branch for small-e fixtures so the
// dense↔sparse equality tests exercise the sparse factor at their existing
// tolerances. Thread-local (each #[test] runs on its own thread), read once in
// `SparseLmmWorkspace::new` — the branch is a construction-time decision.
#[cfg(test)]
thread_local! {
    pub(crate) static FORCE_SPARSE_TAIL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

// Test-only override for the family-downdate route (`FamDowndate`), mirroring
// `FORCE_SPARSE_TAIL` above: `Some(true)` forces the dense accumulator (still
// subject to the memory cap), `Some(false)` the scatter, `None` leaves
// `DD_DENSE_BETA` in charge. Exists so one fixture can be fit both ways and the
// two answers compared; the production rule keeps no fallback of its own, so
// neither arm is dead code kept as an oracle.
#[cfg(test)]
thread_local! {
    pub(crate) static FORCE_DD_ROUTE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Sparse-tail state (branch `e > TAIL_SPARSE_MIN`): the fill-reducing (AMD)
/// sparse Cholesky of the crossed Schur complement `S22 = A22 + I −
/// L21·L21ᵀ`, mirroring `StructuredSchur` (`glmm/workspace.rs`) — symbolic
/// analyzed once per fit off the θ-independent pattern, numeric-refactored per
/// eval. The pattern is the union of the family cliques (two crossed levels
/// couple only if they co-occur within one primary family — the downdate term),
/// the row-co-occurrence `a22_pairs` (a subset whenever every row carries a
/// primary level, unioned for safety), and the full e-diagonal (an unobserved
/// crossed level — reachable, `n_levels` comes from the spec unclamped — sits
/// in no clique but still needs its `+I` slot). AMD reorders the elimination —
/// a sanctioned reassociation of the same Cholesky (same argument as the
/// blocked kernel's doc), so non-PD surfaces are reassociation-equivalent, not
/// bit-identical, to the dense tail's.
///
/// All scatter targets are resolved to CSC slots HERE at construction (an
/// incomplete pattern fails loudly at setup, not as a silently dropped value
/// per eval — the balanced-zero structural bug's failure mode).
pub(crate) struct SparseTail {
    /// Symbolic Cholesky of the S22 pattern (AMD ordering; simplicial or
    /// supernodal by faer's `SupernodalThreshold::AUTO` flops-per-nnz
    /// heuristic — `logdet_llt` handles both arms).
    pub(crate) symbolic: SymbolicCholesky<usize>,
    /// S22's value container in the fixed CSC pattern (lower tri + full
    /// diagonal). Values overwritten per eval; pattern never changes.
    pub(crate) axx: SparseColMat<usize, f64>,
    /// L-factor value buffer, length `symbolic.len_val()`. Overwritten per refactor.
    pub(crate) l_values: Vec<f64>,
    /// Numeric-factor scratch, sized once from `factorize_numeric_llt_scratch`.
    pub(crate) fac_mem: MemBuffer,
    /// Solve scratch for the m-column `S22⁻¹·B̃₂` full solve.
    pub(crate) solve_mem: MemBuffer,
    /// CSC slot of each diagonal entry (t, t), length e — the `+I` targets.
    pub(crate) diag_slots: Vec<u32>,
    /// CSC slot of every `pk_a22` fold emission with `ti ≥ tj`, in the kernel's
    /// exact replay order (pair-major, `jl` outer, `il` inner) — consumed
    /// sequentially by the assembly (mirrors the generation in `new` — change
    /// together).
    pub(crate) a22_slots: Vec<u32>,
    /// How each primary family's `S22 −= L21_f·L21_fᵀ` reaches `axx` — routed
    /// once here off the θ-independent pattern, like every other decision in
    /// this struct.
    pub(crate) fam_dd: FamDowndate,
    /// Per-family compact L21 panel, col-major `e_f`-row columns
    /// (`panel[c·e_f + local]`), sized `fam_w · max_f e_f` once.
    pub(crate) panel: Vec<f64>,
    /// Scratch for the per-family S22 syrk downdate `panel·panelᵀ` (kernel B):
    /// col-major `e_f×e_f`, sized to `max_f e_f²` once, overwritten
    /// (`Accum::Replace`) then applied through `fam_dd`. Empty on the arm that
    /// needs no staging (`FamDowndate::Dense` at `fam_w == 1`, which folds the
    /// rank-1 update straight into the accumulator).
    pub(crate) dd_temp: Vec<f64>,
    /// Scratch for the per-family B̃₂ downdate `panel·U1_sub` (kernel C):
    /// col-major `e_f×m`, sized to `max_f e_f · m` once, overwritten
    /// (`Accum::Replace`) then block-scatter-subtracted into `u2`.
    pub(crate) b2_temp: Vec<f64>,
    /// `X = S22⁻¹·B̃₂` (`e × m`): B̃₂ copied in, full-solved in place per eval
    /// (`LltRef` exposes no arm-agnostic forward-only solve; the deviance
    /// needs only `UᵀU = B̃₂ᵀS22⁻¹B̃₂`, which is permutation-invariant).
    pub(crate) x2: Mat<f64>,
}

/// How a primary family's `S22 −= L21_f·L21_fᵀ` reaches the CSC values.
///
/// Both arms apply the same `Σ_f e_f(e_f+1)/2` lower entries per eval; they
/// differ in what one entry costs. `Scatter` pre-resolves every entry to its own
/// CSC slot, so the per-eval index stream is as long as the value stream and a
/// write can land anywhere in the `nnz(S22)`-wide, AMD-permuted `vals`. `Dense`
/// carries one row index per family-local row instead — `Σ_f e_f` words — and
/// accumulates into a dense lower `e×e` buffer whose active column is
/// contiguous, paying one gather over the CSC pattern per eval for it.
///
/// An enum rather than two zero-sized field sets: the arms' state is disjoint
/// and must never both be allocated — on the widest crossed cells `Scatter`'s
/// slot list is hundreds of MB that `Dense` replaces with the `e×e` buffer — and
/// the generation ↔ replay checklist then reads once per arm instead of covering
/// two possible worlds.
pub(crate) enum FamDowndate {
    /// Per family, CSC slot of every lower pair (a, b ≤ a) of its local crossed
    /// scalar columns, in the downdate's replay order (column-major exclusive:
    /// `b` outer, `a ≥ b` inner — stride-1 over the col-major `dd_temp`);
    /// ragged via `off` (length n_primary+1).
    Scatter { slots: Vec<u32>, off: Vec<usize> },
    /// `rows[off[f] + local]` is the global tail row of family `f`'s local
    /// crossed row `local`. Strictly ascending within a family (`fam_crossed` is
    /// sorted and crossed blocks are laid out level-ascending), so local lower
    /// ⇒ global lower and no re-orientation is needed. Monotone but NOT
    /// contiguous: a crossed factor gets one `LamBlock` per level, so at q = 1 —
    /// every `(1|g)` design — each block is a single tail row. `acc` is the
    /// lower-triangular column-major `e×e` accumulator, zeroed per eval with the
    /// `A22 + I` assembly and gathered into `vals` after the family loop.
    Dense {
        rows: Vec<u32>,
        off: Vec<usize>,
        acc: Vec<f64>,
    },
}

/// Per-fit workspace for the sparse-Z LMM path. Everything θ-independent —
/// `Z'Z`, `Z'[X y]`, `[X y]'[X y]`, the Λ block map — is built once in `new`;
/// every per-eval buffer of the blocked kernel (`sparse_schur_factor`) is
/// allocated here to its correct size and overwritten in place each θ eval,
/// so the optimizer loop is allocation-free.
///
/// Column order follows `build_sparse_z` (slope-major primary, level-major
/// extras), which matches `LmmSuffStats`'s convention — the comparison target.
pub(crate) struct SparseLmmWorkspace {
    pub(crate) g: LmmGroupings,
    /// `Z'[X y]`, `k_total × m` (`m = p + 1`) ROW-major flat (RE row `j`'s m
    /// values contiguous at `j·m` — the B-row fills read whole rows). Fixed
    /// for the design.
    pub(crate) ztxy: Vec<f64>,
    /// `[X y]'[X y]`, `m × m` (full symmetric; only the lower triangle is read
    /// by `sparse_schur_factor`). Direct `W'W` product, cap-free.
    pub(crate) cxy: Mat<f64>,
    /// Packed raw-Gram streams (θ-independent): every `Z'Z` sub-block the
    /// blocked kernel's per-eval assembly folds through Λ, copied once into
    /// flat row-major `q_r×q_c` blocks in EXACTLY the kernel's consumption
    /// order (offsets mirror `sparse_schur_factor` — change together), so the
    /// hot assembly reads sequential memory instead of strided bounds-checked
    /// dense-Gram indexing (~20% of the rung-7 profile before packing).
    /// `pk_fam`: per family `(primary,primary) | per child (child,primary),
    /// (child,child)`; `pk_a21`: per family, per col-block (primary then
    /// children), that family's CO-OCCURRING crossed blocks in order;
    /// `pk_a22`: the co-occurring lower crossed block pairs (`a22_pairs`).
    /// Non-co-occurring blocks are structurally zero and neither packed nor
    /// folded — the zero-fill of the target supplies them. Block-granular
    /// "any entry ≠ 0" == structural co-occurrence: the intercept×intercept
    /// entry is the shared-row COUNT, which balanced-covariate cancellation
    /// (the balanced-zero regression test's lesson) can never zero.
    pub(crate) pk_fam: Vec<f64>,
    pub(crate) pk_a21: Vec<f64>,
    pub(crate) pk_a22: Vec<f64>,
    /// Per family, the `lam_blocks` indices of its co-occurring crossed blocks
    /// (`a21_blk[a21_off[f]..a21_off[f+1]]`), plus each family's start into
    /// `pk_a21` (`pk_a21_off`, length n_primary+1) — the streams are ragged.
    pub(crate) a21_blk: Vec<u32>,
    pub(crate) a21_off: Vec<usize>,
    pub(crate) pk_a21_off: Vec<usize>,
    /// Co-occurring lower crossed block pairs `[row, col]` (`lam_blocks`
    /// indices), parallel to `pk_a22`'s blocks.
    pub(crate) a22_pairs: Vec<[u32; 2]>,
    /// Block-diagonal Λ structure, fixed for the design: one entry per
    /// (grouping, level) block, `[primary 0..n_prim | nested children |
    /// crossed]` in RE-column order. Lets the per-θ assembly run block-wise
    /// instead of materializing the dense `k×k` Λ and paying two `k³` GEMMs
    /// per eval.
    pub(crate) lam_blocks: Vec<LamBlock>,
    /// Concatenated per-grouping `q×q` small-Λ factors (row-major lower-tri as
    /// `primary_lambda` writes them; the upper triangle is never read). Levels
    /// of one grouping share a slice. Refilled once per θ eval by
    /// `fill_lambda_small`.
    pub(crate) lam_small: Vec<f64>,
    /// Family width `w = q_p + n_per·q_n`: one primary level's RE columns plus
    /// its nested children's. Families never share rows, so `A`'s leading
    /// `k_family` block is block-diagonal with `n_primary` blocks of this width.
    pub(crate) fam_w: usize,
    /// Row-major `w×w` scratch: the current family's `A_f = Λ_f'G_fΛ_f + I`,
    /// Crout-factored in place to `L_f` (lower; upper never written or read).
    pub(crate) fam_a: Vec<f64>,
    /// `L21 = A21·L11⁻ᵀ` (`e×k_family` column-major, `e = k_crossed`; columns
    /// family-major, `f·w + local`) — the crossed-tail coupling after family
    /// elimination. A plain col-major `Vec` (the `lmm.rs` `fit.bt` pattern) so
    /// the per-family tri-solve runs as contiguous-column axpys and the syrk
    /// downdate views it through `MatRef::from_column_major_slice`. DENSE-TAIL
    /// branch only — zero-length when `tail` is Some (the sparse branch keeps
    /// per-family compact panels instead; sizing this unconditionally would
    /// silently keep the ~GB-scale dense allocation on huge-e designs).
    pub(crate) l21: Vec<f64>,
    /// `U1 = L11⁻¹·B1`, the family rows of `U = L⁻¹Λ'Z'[X y]`
    /// (`k_family × m` col-major `Vec`, columns contiguous for the UᵀU dots).
    pub(crate) u1: Vec<f64>,
    /// Tail scratch (`e×e`): `S22 = A22 − L21·L21ᵀ` assembled lower-tri, then
    /// LLT-factored in place to `L22` (lower; upper stale, never read).
    /// DENSE-TAIL branch only — 0×0 when `tail` is Some (see `l21`).
    pub(crate) s22: Mat<f64>,
    /// Tail rows of `U`, `e × m` col-major `Vec`. Dense branch:
    /// `U2 = L22⁻¹(B2 − L21·U1)` (the L22 forward solve runs column-oriented on
    /// contiguous slices). Sparse branch: holds `B̃₂ = B2 − L21·U1` instead (no
    /// explicit U₂ exists there — `tail.x2` carries `S22⁻¹B̃₂` and the UᵀU dots
    /// pair the two).
    pub(crate) u2: Vec<f64>,
    /// Fill-reducing sparse factor of the crossed tail; `Some` iff
    /// `e > TAIL_SPARSE_MIN` (decided once here — e is θ-independent) with
    /// branch-conditional buffer sizing on `l21`/`s22`/`tail_llt_mem`.
    pub(crate) tail: Option<SparseTail>,
    /// The augmented Schur factor `L` (dense `m×m` lower Cholesky of
    /// `S = C_xy − UᵀU`), overwritten per eval; `fit_mle_sparse` reads
    /// β̂/σ̂²/SE off it at θ̂ exactly as `fit_lmm` reads `fit.factor`.
    pub(crate) factor: Mat<f64>,
    /// Scratch for the tail's dense `cholesky_in_place` (θ-independent size).
    pub(crate) tail_llt_mem: MemBuffer,
    /// Scratch for the augmented `m×m` `cholesky_in_place` (the `S = C_xy − UᵀU`
    /// factor; kernel A replaced the hand Crout with the same call the dense
    /// tail uses).
    pub(crate) factor_llt_mem: MemBuffer,
    pub(crate) m: usize,
    pub(crate) p: usize,
    /// Row count N — REML df is `N − p` (mirrors `reml_deviance` `lmm.rs:1813`).
    pub(crate) n: usize,
    /// `Some` only for the one post-fit evaluation that feeds
    /// [`sparse_recover_u`]; see [`SparseRecovery`].
    pub(crate) rec: Option<SparseRecovery>,
}

/// Per-family factor state that [`sparse_schur_factor`] otherwise throws away,
/// kept for ONE evaluation so [`sparse_recover_u`] can back-substitute the
/// conditional modes. Off by default (`SparseLmmWorkspace::rec` is `None`), so
/// the θ-search loop is untouched: the fit turns it on for the final evaluation
/// at θ̂ and nowhere else.
///
/// Only two things are missing after an ordinary evaluation. `L_f` is
/// overwritten family by family in the shared `fam_a` scratch, so all but the
/// last are gone. `L21` survives on the DENSE-tail branch (the global `ws.l21`)
/// but not on the sparse-tail one, which consumes each family's compact panel
/// inside its own iteration — hence `panel`, which is a ragged copy of exactly
/// those panels and stays empty on the dense branch.
pub(crate) struct SparseRecovery {
    /// `n_primary · w²` row-major Crout factors, family-major.
    fam_l: Vec<f64>,
    /// Sparse-tail branch only: each family's `e_f × w` col-major L21 panel,
    /// concatenated; family `f` starts at `panel_off[f]` and is `e_f · w` long.
    panel: Vec<f64>,
    /// Length `n_primary + 1`, prefix sums over the panels above.
    panel_off: Vec<usize>,
}

/// One (grouping, level) block of the block-diagonal Λ: local component `d`
/// lives at RE column `start + d·stride` (primary blocks are slope-major with
/// stride `n_primary`; nested/crossed blocks contiguous, stride 1), with values
/// at `lam_small[lam_off + r·q + c]` (r ≥ c). Layout mirrors the deleted dense
/// `build_block_lambda` walk / `reml_deviance_blocked`'s Λ walk (`lmm.rs:1115`).
pub(crate) struct LamBlock {
    start: usize,
    stride: usize,
    q: usize,
    lam_off: usize,
}

impl SparseLmmWorkspace {
    /// Accumulate the θ-independent Grams (`Z'Z`, `Z'[X y]`, `[X y]'[X y]`) in
    /// one pass over the rows and size every blocked-kernel buffer — all
    /// allocations happen here; the eval loop allocates nothing.
    /// `sqrt_w`: `Some(√wᵢ)` per row (prior/case weights, `FitOptions::weights`
    /// square-rooted once by the caller) — threaded into both the z-emission
    /// (`for_each_z_entry`) and pass 2's raw x/y reads below, so every packed
    /// Gram ends up carrying exactly `wᵢ` per row (see `for_each_z_entry`'s
    /// doc). `None` is the unit-weight fast path (no per-row multiply).
    #[allow(clippy::too_many_arguments)] // marshals (g, x, cluster_ids, extra_ids, y, n, p, sqrt_w)
    pub(crate) fn new(
        g: &LmmGroupings,
        x: MatRef<f64>,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        y: &[f64],
        n: usize,
        p: usize,
        sqrt_w: Option<&[f64]>,
    ) -> Self {
        let m = p + 1;

        // Λ block map (fixed for the design): every RE column belongs to exactly
        // one (grouping, level) block. Column layout per `build_sparse_z`:
        // primary slope-major (level f's component d at `d·n_prim + f`), nested
        // children then crossed levels contiguous at their offsets.
        let q_p = g.primary_q;
        let n_prim = g.n_primary;
        let mut lam_blocks: Vec<LamBlock> = Vec::new();
        let mut lam_off = 0usize;
        for f in 0..n_prim {
            lam_blocks.push(LamBlock {
                start: f,
                stride: n_prim,
                q: q_p,
                lam_off,
            });
        }
        lam_off += q_p * q_p;
        if let Some(nf) = g.nested {
            let q_n = nf.q;
            let np = g.nested_per_parent;
            let prim_width = q_p * n_prim;
            for f in 0..n_prim {
                for c in 0..np {
                    let ic = prim_width + (f * np + c) * q_n;
                    lam_blocks.push(LamBlock {
                        start: ic,
                        stride: 1,
                        q: q_n,
                        lam_off,
                    });
                }
            }
            lam_off += q_n * q_n;
        }
        for cf in &g.crossed {
            let off = g.extra_offsets[cf.decl];
            for c in 0..cf.n_levels {
                let ic = off + c * cf.q;
                lam_blocks.push(LamBlock {
                    start: ic,
                    stride: 1,
                    q: cf.q,
                    lam_off,
                });
            }
            lam_off += cf.q * cf.q;
        }
        let lam_small = vec![0.0f64; lam_off.max(1)];

        // Blocked-kernel buffers, sized once from the design's structure.
        let q_nested = g.nested.map(|nf| nf.q).unwrap_or(0);
        let np = g.nested_per_parent;
        let fam_w = q_p + np * q_nested;
        let kf = g.k_family();
        let e = g.k_crossed();
        // Dense-tail cutover, decided once per fit (e is θ-independent) with
        // branch-conditional buffer sizing: the dense branch allocates
        // s22/l21/tail_llt_mem and no sparse state; the sparse branch allocates
        // the pattern/CSC/symbolic/scratches and zero-sized dense buffers
        // (per-eval branching over unconditionally sized buffers would silently
        // keep the ~GB-scale dense allocations on huge-e designs).
        #[cfg(test)]
        let force_sparse = FORCE_SPARSE_TAIL.with(|c| c.get());
        #[cfg(not(test))]
        let force_sparse = false;
        let sparse_tail = e > 0 && (e > TAIL_SPARSE_MIN || force_sparse);
        let tail_llt_mem = MemBuffer::new(cholesky_in_place_scratch::<f64>(
            if sparse_tail { 0 } else { e },
            Par::Seq,
            Spec::default(),
        ));

        // ---- Pass 1 (pattern): family cliques + row-co-occurring crossed
        // block pairs. Replaces the dense k×k `Z'Z` scaffold (2.65 GB transient
        // on the nest3 grid cell — the largest single slice of its ~4.3 GB
        // peak) and the O(e²) all-pairs `blk_nonzero` probe (~74 M probes
        // there) with two O(n) row sweeps: this one discovers structure, the
        // second scatters values straight into the packed streams.
        // Block-granular co-occurrence IS row-sharing (the old probe's
        // intercept×intercept entry is the shared-row COUNT, which balanced
        // covariates can never cancel — the balanced-zero lesson), so
        // a21_blk/a22_pairs come out identical to the dense scan's and the
        // packed streams are bit-identical.
        //
        // Row → block indices: primary level f is block f; a nested child's
        // global id gc (dense over parents, = f·np + c) is block n_prim + gc;
        // crossed factor cf's level l is block cf_base[cf] + l. Mirrors
        // `for_each_z_entry`'s column layout — change together.
        let q_n = q_nested;
        let cb0 = n_prim + n_prim * np; // first crossed block in lam_blocks
        let mut cf_base = Vec::with_capacity(g.crossed.len());
        {
            let mut b = cb0;
            for cf in &g.crossed {
                cf_base.push(b);
                b += cf.n_levels;
            }
        }
        // Fixed per-declaration segment offsets into the per-row entry list
        // (every row carries q_p primary entries, then each extra's q_g, in
        // `for_each_z_entry`'s emission order).
        let n_extras = extra_ids.len();
        let mut seg_off = vec![0usize; n_extras];
        {
            let mut o = q_p;
            for (ei, so) in seg_off.iter_mut().enumerate() {
                *so = o;
                o += g.extra_q[ei];
            }
        }
        let nested_decl = g.nested.map(|nf| nf.decl);
        let mut decl_base = vec![usize::MAX; n_extras];
        for (ci, cf) in g.crossed.iter().enumerate() {
            decl_base[cf.decl] = cf_base[ci];
        }

        let mut fam_crossed: Vec<Vec<u32>> = vec![Vec::new(); n_prim];
        let mut pair_seen = std::collections::HashSet::<(u32, u32)>::new();
        let mut a22_pairs: Vec<[u32; 2]> = Vec::new();
        let mut row_cb: Vec<u32> = Vec::with_capacity(g.crossed.len());
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            row_cb.clear();
            for cf in &g.crossed {
                row_cb.push((decl_base[cf.decl] + extra_ids[cf.decl][i] as usize) as u32);
            }
            for (ai, &bi) in row_cb.iter().enumerate() {
                fam_crossed[f].push(bi);
                for &bj in &row_cb[..=ai] {
                    let key = if bi >= bj { (bi, bj) } else { (bj, bi) };
                    if pair_seen.insert(key) {
                        a22_pairs.push([key.0, key.1]);
                    }
                }
            }
        }
        drop(pair_seen);
        for v in fam_crossed.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }
        // The retired dense scan's iteration order (col block asc, then row
        // block asc) — the kernel's `cur` replay over pk_a22 depends on it.
        a22_pairs.sort_unstable_by_key(|pr| (pr[1], pr[0]));

        // Packed-stream allocation + scatter offsets, sized from the pattern.
        let fam_len = q_p * q_p + np * (q_n * q_p + q_n * q_n);
        let mut pk_fam = vec![0.0f64; n_prim * fam_len];
        let mut a21_blk: Vec<u32> = Vec::new();
        let mut a21_off = Vec::with_capacity(n_prim + 1);
        let mut pk_a21_off = Vec::with_capacity(n_prim + 1);
        // Start of each block's merged slab in pk_a21 (parallel to a21_blk):
        // q rows × (q_p primary + np·q_n child) cols = q·fam_w values.
        let mut a21_slab_off: Vec<usize> = Vec::new();
        let mut pk_a21_len = 0usize;
        for list in &fam_crossed {
            a21_off.push(a21_blk.len());
            pk_a21_off.push(pk_a21_len);
            for &bi in list {
                a21_blk.push(bi);
                a21_slab_off.push(pk_a21_len);
                pk_a21_len += lam_blocks[bi as usize].q * fam_w;
            }
        }
        a21_off.push(a21_blk.len());
        pk_a21_off.push(pk_a21_len);
        let mut pk_a21 = vec![0.0f64; pk_a21_len];
        let mut a22_off_map = std::collections::HashMap::<(u32, u32), usize>::new();
        let mut pk_a22_len = 0usize;
        for pr in &a22_pairs {
            a22_off_map.insert((pr[0], pr[1]), pk_a22_len);
            pk_a22_len += lam_blocks[pr[0] as usize].q * lam_blocks[pr[1] as usize].q;
        }
        let mut pk_a22 = vec![0.0f64; pk_a22_len];

        // ---- Pass 2 (values): Z'[X y], [X y]'[X y], and every packed raw-Gram
        // block, scattered per row over Z's ≤ q_p + Σq_e nonzeros — O(n·(Σq)²)
        // total, same accumulation the dense Gram + `pack` copy produced,
        // block-pair-keyed instead of k×k-staged. NOT the shared
        // `add_rows_multi` accumulator: that path packs per-row level ids into
        // a fixed `[usize; 1 + MAX_EXTRA_GROUPINGS]` stack array (`lmm.rs:542`)
        // and would index out of bounds for over-envelope-by-count designs —
        // the sparse route must stay cap-free.
        let mut ztxy = vec![0.0f64; g.k_total * m];
        let mut cxy = Mat::<f64>::zeros(m, m);
        let mut row: Vec<(usize, f64)> =
            Vec::with_capacity(g.primary_q + g.extra_q.iter().sum::<usize>());
        for i in 0..n {
            // One √wᵢ per row factor: `row`'s z entries already carry it
            // (`for_each_z_entry`), so the raw x/y reads below must carry a
            // matching one — every ztxy/cxy product then carries wᵢ exactly
            // once (see `for_each_z_entry`'s doc and `SparseLmmWorkspace::new`'s
            // top-level doc).
            let sw = sqrt_w.map_or(1.0, |w| w[i]);
            row.clear();
            for_each_z_entry(g, x, cluster_ids, extra_ids, i, sqrt_w, |col, v| {
                row.push((col, v))
            });
            for &(ca, va) in &row {
                for j in 0..p {
                    ztxy[ca * m + j] += va * (sw * x[(i, j)]);
                }
                ztxy[ca * m + p] += va * (sw * y[i]);
            }
            for a in 0..m {
                let wa = if a < p { sw * x[(i, a)] } else { sw * y[i] };
                for b in 0..m {
                    let wb = if b < p { sw * x[(i, b)] } else { sw * y[i] };
                    cxy[(a, b)] += wa * wb;
                }
            }
            let f = cluster_ids[i] as usize;
            let vp = &row[..q_p];
            let fam = &mut pk_fam[f * fam_len..(f + 1) * fam_len];
            for a in 0..q_p {
                for b in 0..q_p {
                    fam[a * q_p + b] += vp[a].1 * vp[b].1;
                }
            }
            // (child, primary) + (child, child) family sub-blocks; sibling
            // child–child blocks never co-occur (one child per row), matching
            // the kernel's structural-zero fill.
            let child = nested_decl.map(|d| {
                let gc = extra_ids[d][i] as usize;
                debug_assert!(
                    gc >= f * np && gc < (f + 1) * np,
                    "nested ids are parent-padded"
                );
                (d, gc - f * np)
            });
            if let Some((d, c)) = child {
                let vc = &row[seg_off[d]..seg_off[d] + q_n];
                let b2 = q_p * q_p + c * (q_n * q_p + q_n * q_n);
                for a in 0..q_n {
                    for b in 0..q_p {
                        fam[b2 + a * q_p + b] += vc[a].1 * vp[b].1;
                    }
                }
                let b3 = b2 + q_n * q_p;
                for a in 0..q_n {
                    for b in 0..q_n {
                        fam[b3 + a * q_n + b] += vc[a].1 * vc[b].1;
                    }
                }
            }
            let fam_list = &fam_crossed[f];
            for (ci, cf) in g.crossed.iter().enumerate() {
                let bi = (cf_base[ci] + extra_ids[cf.decl][i] as usize) as u32;
                let q = cf.q;
                let vx = &row[seg_off[cf.decl]..seg_off[cf.decl] + q];
                // A21 merged slab: block bi's rows × primary then child cols.
                let idx = fam_list
                    .binary_search(&bi)
                    .expect("row's crossed block is in its family clique");
                let slab0 = a21_slab_off[a21_off[f] + idx];
                for a in 0..q {
                    for b in 0..q_p {
                        pk_a21[slab0 + a * q_p + b] += vx[a].1 * vp[b].1;
                    }
                }
                if let Some((d, c)) = child {
                    let vc = &row[seg_off[d]..seg_off[d] + q_n];
                    let base = slab0 + q * q_p + c * q * q_n;
                    for a in 0..q {
                        for b in 0..q_n {
                            pk_a21[base + a * q_n + b] += vx[a].1 * vc[b].1;
                        }
                    }
                }
                // A22 diagonal block (bi, bi), full q×q — the fold reads both
                // triangles of the raw block, as the dense Gram stored them.
                let off = a22_off_map[&(bi, bi)];
                for a in 0..q {
                    for b in 0..q {
                        pk_a22[off + a * q + b] += vx[a].1 * vx[b].1;
                    }
                }
                // Cross-factor A22 blocks (row block = higher index; same-factor
                // level pairs never share a row, so no off-diagonal same-factor
                // blocks arise — exactly the dense scan's outcome).
                for (cj, cf2) in g.crossed.iter().enumerate().take(ci) {
                    let bj = (cf_base[cj] + extra_ids[cf2.decl][i] as usize) as u32;
                    let vx2 = &row[seg_off[cf2.decl]..seg_off[cf2.decl] + cf2.q];
                    let (bh, vh, qh, bl, vl, ql) = if bi >= bj {
                        (bi, vx, q, bj, vx2, cf2.q)
                    } else {
                        (bj, vx2, cf2.q, bi, vx, q)
                    };
                    let off = a22_off_map[&(bh, bl)];
                    for a in 0..qh {
                        for b in 0..ql {
                            pk_a22[off + a * ql + b] += vh[a].1 * vl[b].1;
                        }
                    }
                }
            }
        }
        drop(a22_off_map);

        let tail = if sparse_tail {
            Some(build_sparse_tail(
                &lam_blocks,
                &fam_crossed,
                &a22_pairs,
                kf,
                e,
                m,
                fam_w,
            ))
        } else {
            None
        };

        Self {
            g: g.clone(),
            ztxy,
            cxy,
            pk_fam,
            pk_a21,
            pk_a22,
            a21_blk,
            a21_off,
            pk_a21_off,
            a22_pairs,
            lam_blocks,
            lam_small,
            fam_w,
            fam_a: vec![0.0f64; fam_w * fam_w],
            l21: if sparse_tail {
                Vec::new()
            } else {
                vec![0.0f64; e * kf]
            },
            u1: vec![0.0f64; kf * m],
            s22: if sparse_tail {
                Mat::zeros(0, 0)
            } else {
                Mat::zeros(e, e)
            },
            u2: vec![0.0f64; e * m],
            tail,
            factor: Mat::zeros(m, m),
            tail_llt_mem,
            factor_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                m,
                Par::Seq,
                Spec::default(),
            )),
            m,
            p,
            n,
            rec: None,
        }
    }

    /// Arm [`SparseRecovery`] for the NEXT evaluation. Sized here, once, from
    /// the shapes the workspace already fixes; the panels are ragged, so the
    /// offsets are computed from each family's co-occurring crossed blocks the
    /// same way `sparse_schur_factor` sums `e_f`.
    fn arm_recovery(&mut self) {
        let w = self.fam_w;
        let n_prim = self.g.n_primary;
        let sparse_tail = self.tail.is_some();
        let mut panel_off = Vec::with_capacity(n_prim + 1);
        let mut acc = 0usize;
        for f in 0..n_prim {
            panel_off.push(acc);
            if sparse_tail {
                let e_f: usize = self.a21_blk[self.a21_off[f]..self.a21_off[f + 1]]
                    .iter()
                    .map(|&bi| self.lam_blocks[bi as usize].q)
                    .sum();
                acc += e_f * w;
            }
        }
        panel_off.push(acc);
        self.rec = Some(SparseRecovery {
            fam_l: vec![0.0f64; n_prim * w * w],
            panel: vec![0.0f64; acc],
            panel_off,
        });
    }
}

/// Build the θ-independent sparse-tail state (see `SparseTail`): the scalar S22
/// pattern (family cliques ∪ `a22_pairs`, plus the full diagonal), the AMD
/// symbolic factor (simplicial-or-supernodal by faer's AUTO heuristic), and
/// every scatter target pre-resolved to its
/// CSC slot — a missing entry panics HERE at setup, never as a silently dropped
/// value per eval.
fn build_sparse_tail(
    lam_blocks: &[LamBlock],
    fam_crossed: &[Vec<u32>],
    a22_pairs: &[[u32; 2]],
    kf: usize,
    e: usize,
    m: usize,
    fam_w: usize,
) -> SparseTail {
    // Coupling block pairs (row block index ≥ col block index): every pair of
    // crossed blocks co-occurring within one family (the −L21·L21ᵀ downdate
    // couples exactly these, including each block with itself), unioned with
    // the row-co-occurrence `a22_pairs` for safety (a strict subset whenever
    // every row carries a primary level, as the Z layout guarantees).
    let mut blk_pairs: Vec<(u32, u32)> = Vec::new();
    for list in fam_crossed {
        for (ii, &bi) in list.iter().enumerate() {
            for &bj in &list[..=ii] {
                blk_pairs.push((bi, bj)); // list ascending ⇒ bi ≥ bj
            }
        }
    }
    for pr in a22_pairs {
        blk_pairs.push((pr[0], pr[1]));
    }
    // Scalar pattern triplets: full diagonal first (an unobserved crossed level
    // — reachable, `n_levels` comes from the spec unclamped by the ids — sits
    // in no clique, yet its `+I` needs a slot; the dense branch survives this
    // by writing into a full matrix), then each block pair's scalar entries.
    let mut seen = std::collections::HashSet::<(usize, usize)>::new();
    let mut trips = Vec::<Triplet<usize, usize, f64>>::new();
    for t in 0..e {
        trips.push(Triplet::new(t, t, 0.0));
        seen.insert((t, t));
    }
    for &(bi, bj) in &blk_pairs {
        let (ri, rj) = (&lam_blocks[bi as usize], &lam_blocks[bj as usize]);
        let (ti0, tj0) = (ri.start - kf, rj.start - kf);
        for il in 0..ri.q {
            for jl in 0..rj.q {
                let (a, b) = (ti0 + il, tj0 + jl);
                let key = if a >= b { (a, b) } else { (b, a) };
                if seen.insert(key) {
                    trips.push(Triplet::new(key.0, key.1, 0.0));
                }
            }
        }
    }
    let axx = SparseColMat::<usize, f64>::try_new_from_triplets(e, e, &trips)
        .expect("S22 pattern triplets well-formed");
    let symbolic = factorize_symbolic_cholesky(
        axx.symbolic(),
        Side::Lower,
        Default::default(), // AMD fill-reducing ordering
        CholeskySymbolicParams {
            // AUTO routes per pattern: saturated-fill tails (high per-group
            // counts, small e) go supernodal/BLAS-3, huge-q/low-fill tails
            // stay simplicial. Both arms are read back by `logdet_llt`.
            supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
            ..Default::default()
        },
    )
    .expect("S22 symbolic factorization");
    // CSC slot of every stored (row ≥ col) pattern entry.
    let mut slot = std::collections::HashMap::<(usize, usize), u32>::new();
    {
        let sym = axx.symbolic();
        let col_ptr = sym.col_ptr();
        let row_idx = sym.row_idx();
        for j in 0..e {
            for (k, &ri) in row_idx
                .iter()
                .enumerate()
                .take(col_ptr[j + 1])
                .skip(col_ptr[j])
            {
                slot.insert((ri, j), k as u32);
            }
        }
    }
    let slot_of = |t: (usize, usize)| -> u32 {
        *slot
            .get(&t)
            .unwrap_or_else(|| panic!("S22 pattern missing entry ({}, {})", t.0, t.1))
    };
    let diag_slots: Vec<u32> = (0..e).map(|t| slot_of((t, t))).collect();
    // pk_a22 fold replay order (pair-major, jl outer, il inner, lower-only) —
    // mirrors the kernel's assembly loop, consumed sequentially there.
    let mut a22_slots = Vec::new();
    for pr in a22_pairs {
        let (ri, rj) = (&lam_blocks[pr[0] as usize], &lam_blocks[pr[1] as usize]);
        let (ti0, tj0) = (ri.start - kf, rj.start - kf);
        for jl in 0..rj.q {
            for il in 0..ri.q {
                if ti0 + il >= tj0 + jl {
                    a22_slots.push(slot_of((ti0 + il, tj0 + jl)));
                }
            }
        }
    }
    // Family-downdate route (`FamDowndate`): both arms cost the same
    // `Σ_f e_f(e_f+1)/2` entry applications per eval, so the comparison is over
    // what the dense arm adds (one gather over the stored pattern — `nnz(S22)`,
    // not `e(e+1)/2`, since the pattern is the co-occurrence union — plus the
    // `e²` zero) against what it removes.
    let mut max_ef = 0usize;
    let mut sum_pairs = 0.0f64;
    for list in fam_crossed {
        let ef: usize = list.iter().map(|&bi| lam_blocks[bi as usize].q).sum();
        max_ef = max_ef.max(ef);
        sum_pairs += (ef as f64) * (ef as f64 + 1.0) * 0.5;
    }
    let nnz = axx.symbolic().row_idx().len() as f64;
    let cap_ok = e.saturating_mul(e).saturating_mul(8) <= DD_DENSE_MAX_BYTES;
    let dense_route = cap_ok && nnz + (e as f64) * (e as f64) < DD_DENSE_BETA * sum_pairs;
    #[cfg(test)]
    let dense_route = match FORCE_DD_ROUTE.with(|c| c.get()) {
        Some(want) => want && cap_ok,
        None => dense_route,
    };
    let mut off = Vec::with_capacity(fam_crossed.len() + 1);
    off.push(0);
    let fam_dd = if dense_route {
        // Local → global tail row per family-local crossed row, ascending
        // within a family. This is the same walk the scatter's `cols` does
        // below; keeping its result IS the dense arm's whole index state.
        let mut rows: Vec<u32> = Vec::new();
        for list in fam_crossed {
            for &bi in list {
                let b = &lam_blocks[bi as usize];
                let t0 = b.start - kf;
                rows.extend((t0..t0 + b.q).map(|t| t as u32));
            }
            off.push(rows.len());
        }
        FamDowndate::Dense {
            rows,
            off,
            acc: vec![0.0f64; e * e],
        }
    } else {
        // Per-family downdate slots over the family's local crossed scalar
        // columns (ascending, so local a ≥ b ⇒ global lower) — the kernel's
        // replay order (COLUMN-major exclusive: `b` outer, `a ≥ b` inner, so
        // the replay reads the col-major `dd_temp` stride-1); mirrors the panel
        // scatter — change together. Within a family every (a, b) pair is a
        // distinct CSC slot, so replay order only permutes which slot is hit
        // when — the values are bit-identical either way.
        let mut slots = Vec::new();
        let mut cols: Vec<usize> = Vec::new();
        for list in fam_crossed {
            cols.clear();
            for &bi in list {
                let b = &lam_blocks[bi as usize];
                let t0 = b.start - kf;
                cols.extend(t0..t0 + b.q);
            }
            for bl in 0..cols.len() {
                for a in bl..cols.len() {
                    slots.push(slot_of((cols[a], cols[bl])));
                }
            }
            off.push(slots.len());
        }
        FamDowndate::Scatter { slots, off }
    };
    let l_values = vec![0.0f64; symbolic.len_val()];
    let fac_mem =
        MemBuffer::new(symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default()));
    let solve_mem = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(m, Par::Seq));
    SparseTail {
        symbolic,
        axx,
        l_values,
        fac_mem,
        solve_mem,
        diag_slots,
        a22_slots,
        fam_dd,
        panel: vec![0.0f64; fam_w * max_ef],
        // The dense arm at fam_w == 1 folds the rank-1 update straight into
        // `acc` and never stages — mirrors the `w == 1` branch in
        // `sparse_schur_factor`, change together.
        dd_temp: vec![
            0.0f64;
            if dense_route && fam_w == 1 {
                0
            } else {
                max_ef * max_ef
            }
        ],
        b2_temp: vec![0.0f64; max_ef * m],
        x2: Mat::zeros(e, m),
    }
}

/// Visit row `i`'s nonzero `Z` entries as `(RE column, value)` pairs — the
/// single owner of the per-row scatter in the sparse path's RE-column order
/// `[primary | nested children | crossed]`, matching `LmmSuffStats`'s column
/// indexing (the comparison target). NB: this deliberately diverges from the
/// dense GLMM `build_z` (`glmm/workspace.rs`), whose primary block is
/// level-major (`lvl·q_p + c`) — that is a separate, self-consistent fit path
/// and is NOT the layout to mirror here. Intercept RE columns carry `1.0`;
/// slope RE columns the covariate value from `x`.
/// Primary block is slope-major: intercept for level f at column f, slope k at
/// column `(k+1)·n_primary + f` — the `d·n_primary + f` convention owned by
/// `from_cluster_spec_ext:246` and read by `add_rows_multi`/`primary_gram`.
/// Extra-grouping blocks are level-major: `extra_offsets[e] + level·q_g + c`.
/// For a nested extra, `extra_ids[e][i]` is the global child level (dense over
/// all parents), so the same formula addresses it directly. Shared by the
/// workspace's per-row Gram accumulation and (in tests) `build_sparse_z`.
/// `sqrt_w`: `Some(√wᵢ)` scales every emitted z entry by row `i`'s prior-weight
/// square root (the intercept literal `1.0` becomes `√wᵢ` itself); `None` is
/// unit weight. One `√wᵢ` per entry is the row-weighting invariant pass 2
/// depends on — its own raw x/y reads carry a matching `√wᵢ` factor (see
/// `SparseLmmWorkspace::new`), so every product of two z entries, or of a z
/// entry with an x/y read, ends up carrying exactly `wᵢ` (mirrors
/// `add_rows_multi`'s `zw`-per-side scheme, `lmm.rs:696-733`).
#[inline]
fn for_each_z_entry(
    g: &LmmGroupings,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    i: usize,
    sqrt_w: Option<&[f64]>,
    mut emit: impl FnMut(usize, f64),
) {
    let sw = sqrt_w.map_or(1.0, |w| w[i]);
    let f = cluster_ids[i] as usize;
    // Slope-major primary layout — mirrors add_rows_multi's scatter (change together).
    // Every slope value is read AS A RANDOM-EFFECT column and so takes the RE
    // column's internal scale (`LmmGroupings::set_slope_scales`); the same x column
    // keeps its raw value where the fixed-effect design reads it (`Z'X`, `[X y]'[X y]`).
    // The intercept subcolumn's scale is exactly 1 by construction.
    emit(f, sw);
    for (k, &col) in g.primary_slope_cols.iter().enumerate() {
        emit(
            (k + 1) * g.n_primary + f,
            sw * (x[(i, col)] / g.primary_slope_scales[k]),
        );
    }
    // Extra groupings: intercept at off, slope c at off+1+c.
    for (e, ids_e) in extra_ids.iter().enumerate() {
        let q_g = g.extra_q[e];
        let off = g.extra_offsets[e] + ids_e[i] as usize * q_g;
        emit(off, sw);
        for (c, &col) in g.extra_slope_cols[e].iter().enumerate() {
            emit(off + 1 + c, sw * (x[(i, col)] / g.extra_slope_scales[e][c]));
        }
    }
}

/// Explicit sparse design `Z` (`n × k_total`) in `for_each_z_entry`'s column
/// layout. Test-only since the blocked kernel: production accumulates the
/// Grams per row without materializing Z; the layout tests densify this to
/// pin the column convention.
#[cfg(test)]
pub(crate) fn build_sparse_z(
    g: &LmmGroupings,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) -> SparseColMat<usize, f64> {
    let mut trips: Vec<Triplet<usize, usize, f64>> =
        Vec::with_capacity(n * (g.primary_q + extra_ids.len()));
    for i in 0..n {
        for_each_z_entry(g, x, cluster_ids, extra_ids, i, None, |col, v| {
            trips.push(Triplet::new(i, col, v));
        });
    }
    SparseColMat::try_new_from_triplets(n, g.k_total, &trips).expect("Z triplets well-formed")
}

/// Refill `lam_small` with the per-grouping `q×q` Λ factors at θ, once per
/// eval — the only Λ materialization on this path (the dense `k×k` Λ is never
/// built). Offsets mirror the `lam_blocks` construction in `new` — change
/// together (`[primary | nested | crossed]`, each block `q×q` row-major
/// lower-tri). θ is sliced per grouping: primary from the vech prefix, each
/// extra from its `vech_start`. Shared by the Gaussian blocked kernel
/// (`sparse_schur_factor`) and the non-Gaussian sparse PIRLS
/// (`SparseGlmmWorkspace`, whose `lam_off_decl` maps declarations into the
/// same layout).
pub(super) fn fill_lambda_small(theta: &[f64], g: &LmmGroupings, lam_small: &mut [f64]) {
    let q_p = g.primary_q;
    let mut off = 0usize;
    crate::lmm::primary_lambda(theta, q_p, &mut lam_small[off..off + q_p * q_p]);
    off += q_p * q_p;
    if let Some(nf) = g.nested {
        let q_n = nf.q;
        crate::lmm::primary_lambda(
            &theta[nf.vech_start..],
            q_n,
            &mut lam_small[off..off + q_n * q_n],
        );
        off += q_n * q_n;
    }
    for cf in &g.crossed {
        let q = cf.q;
        crate::lmm::primary_lambda(&theta[cf.vech_start..], q, &mut lam_small[off..off + q * q]);
        off += q * q;
    }
}

/// Column `jl` of one (row-block, col-block) pair of `A = Λ'(Z'Z)Λ` (no `+I`)
/// from a PACKED row-major `q_r×q_c` raw-Gram block: the sandwich reduces to
/// `B_r'·G_blk·B_c` — a double loop over the two blocks' components (Λ
/// lower-tri, so only `a ≥ il`, `b ≥ jl` contribute), emitted per row through
/// `sink(il, value)`. Packed-stream-driven on purpose: entry-wise dense-Gram
/// assembly (per-entry `blk_of_col` lookup + integer division + strided
/// bounds-checked reads) was 49% of the rung-7 profile; the packed replay is
/// sequential. `lo_r`/`lo_c` are the two blocks' `lam_small` offsets. Requires
/// `fill_lambda_small` to have run at this θ (the shared caller
/// `sparse_schur_factor` does).
#[inline]
#[allow(clippy::too_many_arguments)]
fn fold_packed_col(
    gblk: &[f64],
    q_r: usize,
    q_c: usize,
    lam: &[f64],
    lo_r: usize,
    lo_c: usize,
    jl: usize,
    mut sink: impl FnMut(usize, f64),
) {
    // A_blk[il,jl] = Σ_{a≥il} Λ_r[a,il] · Σ_{b≥jl} G_blk[a,b] · Λ_c[b,jl].
    for il in 0..q_r {
        let mut acc = 0.0;
        for a in il..q_r {
            let la = lam[lo_r + a * q_r + il];
            let mut inner = 0.0;
            for b in jl..q_c {
                inner += gblk[a * q_c + b] * lam[lo_c + b * q_c + jl];
            }
            acc += la * inner;
        }
        sink(il, acc);
    }
}

/// Profiled-REML deviance on the sparse path. Numerically equal to
/// `lmm::reml_deviance` at the same θ (same argmin, same dropped additive
/// constants — the both-paths cross-check is the guarantee). Steps:
///   1. blocked Cholesky of A = Λ'Z'ZΛ + I → log|L_ZZ|² (`sparse_schur_factor`);
///   2. forward-solve U = L⁻¹B, B = Λ'Z'[X y];
///   3. S = C_xy − UᵀU (dense m×m, C_xy = ws.cxy = [X y]'[X y]), Crout → L_XX;
///   4. σ̂² = L_XX[p,p]² / (N−p); deviance = log|L_ZZ|² + log|L_XX|² + (N−p)·ln σ̂².
///
/// Normalization transcribed from `reml_deviance` (`lmm.rs:1801-1820`): REML
/// df = N−p, `σ̂² = L[p,p]²/df`, `log|L_XX|² = 2Σ_{j<p} ln L[j,j]`; drops the same
/// additive constants (2π, …). Returns INFINITY on any non-PD factor / non-finite
/// σ̂² (mirrors the dense diagonal guards, `lmm.rs:1804/1815`).
pub(crate) fn sparse_reml_deviance(theta: &[f64], ws: &mut SparseLmmWorkspace) -> f64 {
    let p = ws.p;
    let log_lzz_sq = match sparse_schur_factor(theta, ws) {
        Some(v) => v,
        None => return f64::INFINITY, // non-PD blocked or Schur factor at this θ
    };

    // Deviance off the dense augmented factor's diagonal (mirror reml_deviance:1801-1820).
    let l = &ws.factor;
    let mut log_lxx_sq = 0.0_f64;
    for j in 0..p {
        let ljj = l[(j, j)];
        if !(ljj.is_finite() && ljj > 0.0) {
            return f64::INFINITY;
        }
        log_lxx_sq += ljj.ln();
    }
    log_lxx_sq *= 2.0;

    let lyy = l[(p, p)];
    let df = (ws.n - p) as f64;
    let sigma_sq = lyy * lyy / df;
    if !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        return f64::INFINITY;
    }
    log_lzz_sq + log_lxx_sq + df * sigma_sq.ln()
}

/// Two-level blocked Cholesky of `A = Λ'Z'ZΛ + I` plus the augmented Schur
/// factor at θ — specialized to the class this path serves by construction:
/// primary(+nested) FAMILIES are block-diagonal (levels never share rows across
/// families) and ALL crossed-extra columns form one dense `e×e` tail
/// (`e = k_crossed`). MixedModels.jl's two-level `BlockedSparse` layout, as a
/// lean compiled kernel:
///
///   L11: per-family `w×w` Crout (`w = q_p + n_per·q_n`), pivots → log|L_ZZ|²;
///   L21 = A21·L11⁻ᵀ: per-family `e×w` tri-solves;
///   S22 = A22 − L21·L21ᵀ: one faer triangular syrk downdate, then a dense
///         LLT in place → L22 (tail pivots complete log|L_ZZ|²);
///   U = L⁻¹B forward-solve ONLY (B = Λ'Z'[X y]) — the deviance and recovery
///         need BᵀA⁻¹B = UᵀU, never A⁻¹B, so there is no back-solve;
///   S = C_xy − UᵀU, Crout → the augmented `m×m` factor `L` in `ws.factor`
///         (lower; upper zeroed).
///
/// Per-eval cost O(Σ_f w³) + O(e²·k_family) + O(e³/3): primary levels scale
/// linearly; extras pay the same e³ as MixedModels — on the DENSE-tail branch.
/// Above `TAIL_SPARSE_MIN` (`ws.tail` = Some) the tail steps are replaced by
/// their fill-reducing forms: per-family compact L21 panels, an S22 scatter
/// into a fixed CSC pattern, and faer's AMD sparse LLT (simplicial-or-
/// supernodal by faer's AUTO heuristic) — the same
/// analyze-once/factorize-per-eval split as CHOLMOD in lme4, ~linear in e when
/// the family cliques are small (see `SparseTail`). The recovery seam is
/// unchanged: L22 never escapes an eval — it feeds only the tail pivots of
/// log|L_ZZ|² (log-det is permutation-invariant) and UᵀU = B̃₂ᵀS22⁻¹B̃₂ (also
/// permutation-invariant), so no back-permutation exists anywhere.
///
/// All buffers are workspace-resident — the eval loop allocates nothing (the
/// sparse numeric LLT runs off a pre-sized `MemStack`). Numerically this is
/// the same Cholesky as the generic sparse factorization it replaced up to the
/// elimination order (structure-driven here vs AMD) and GEMM accumulation
/// order — a sanctioned reassociation (same argument as `lmm.rs:1796-1802`),
/// gated by the both-paths deviance-equality tests and the validation suite.
///
/// Returns `log|L_ZZ|²`; `None` on any non-PD pivot (family Crout, tail LLT —
/// `LltRegularization::default()` is a verified no-op, delta = 0 — or Schur
/// Crout), the same INFINITY surface as the dense evaluators (`lmm.rs:1795`).
/// Shared by `sparse_reml_deviance` (diagonal → deviance) and `fit_mle_sparse`
/// (the full factor → β̂/σ̂²/SE recovery). By Cholesky uniqueness `ws.factor`
/// equals the dense path's augmented `fit.factor` at the same θ (both factor
/// the identical reduced augmented Gram X'V⁻¹[X y]): `L[(j,j)]` (j<p) is the
/// L_XX diagonal, `L[(p,j)]` the y-row `l_yX[j]`, and `L[(p,p)]²/(N−p)` the
/// profiled σ̂² — the recovery reads them exactly as `fit_lmm` reads
/// `fit.factor` (`lmm.rs:1959-1993`).
/// Schur L21 Phase B: tri-solve each of the `w` family columns against `L_fᵀ`
/// (stored col-major in `fam_a`, `w×w`) as contiguous-column axpys — columns
/// before `fb+c` are final when read (the lmm.rs:1779 `fit.bt` split_at_mut
/// pattern). `buf` is exactly the family's `w` columns, each `col_len` long
/// (`e` for the dense global `l21` slice, `e_f` for the compact panel), so the
/// dense and sparse-panel arms share this verbatim. One reciprocal replaces `e`
/// scalar divides (≤1-ulp difference vs division, inside every parity band).
fn schur_phase_b(buf: &mut [f64], col_len: usize, w: usize, fam_a: &[f64]) {
    for c in 0..w {
        let (done, rest) = buf.split_at_mut(c * col_len);
        let col_c = &mut rest[..col_len];
        for k in 0..c {
            let l_ck = fam_a[c * w + k];
            let col_k = &done[k * col_len..(k + 1) * col_len];
            for (x, &yv) in col_c.iter_mut().zip(col_k) {
                *x -= l_ck * yv;
            }
        }
        let inv_cc = 1.0 / fam_a[c * w + c];
        for v in col_c.iter_mut() {
            *v *= inv_cc;
        }
    }
}

fn sparse_schur_factor(theta: &[f64], ws: &mut SparseLmmWorkspace) -> Option<f64> {
    // Per-grouping small-Λ factors, filled once and shared by every A-block
    // assembly and the B rows below.
    fill_lambda_small(theta, &ws.g, &mut ws.lam_small);
    let SparseLmmWorkspace {
        g,
        ztxy,
        cxy,
        pk_fam,
        pk_a21,
        pk_a22,
        a21_blk,
        a21_off,
        pk_a21_off,
        a22_pairs,
        lam_blocks,
        lam_small,
        fam_w,
        fam_a,
        l21,
        u1,
        s22,
        u2,
        tail,
        factor,
        tail_llt_mem,
        factor_llt_mem,
        m,
        rec,
        ..
    } = ws;
    let m = *m;
    let w = *fam_w;
    let n_prim = g.n_primary;
    let q_p = g.primary_q;
    let np = g.nested_per_parent;
    let q_n = g.nested.map(|nf| nf.q).unwrap_or(0);
    let kf = g.k_family();
    let e = g.k_crossed();

    // Per-family stride of the packed family stream (mirror `new`'s packing —
    // change together); the A21 streams are ragged, addressed via pk_a21_off.
    let fam_len = q_p * q_p + np * (q_n * q_p + q_n * q_n);
    // First crossed block in `lam_blocks` (after n_prim primary + n_prim·np
    // child blocks); a crossed block's tail row is `start − kf` (stride 1).
    let cb0 = n_prim + n_prim * np;

    if e > 0 {
        // B2 tail rows: fill per crossed block (stride 1), same Λ'-fold as the
        // U1 rows below. Runs BEFORE the family loop so the sparse branch can
        // downdate `B̃₂ = B2 − L21·U1` per family off its compact panel; the
        // dense branch downdates after the loop with one GEMM as before (the
        // fill only reads lam_small/ztxy, so its position is irrelevant to it).
        for br in &lam_blocks[cb0..] {
            let (q, t0) = (br.q, br.start - kf);
            for il in 0..q {
                for c in 0..m {
                    u2[c * e + t0 + il] = 0.0;
                }
                for a in il..q {
                    let la = lam_small[br.lam_off + a * q + il];
                    let zr = (br.start + a) * m;
                    for c in 0..m {
                        u2[c * e + t0 + il] += la * ztxy[zr + c];
                    }
                }
            }
        }
        // Sparse branch: assemble A22 + I into the CSC values now; the family
        // loop scatters each −L21_f·L21_fᵀ downdate on top. Same fold as the
        // dense arm's, sunk through the precomputed slot replay (`a22_slots`,
        // sequential — mirrors the generation in `build_sparse_tail`, change
        // together); every target exists by construction of the
        // diagonal-seeded pattern, verified loudly at setup.
        if let Some(tail) = tail.as_mut() {
            let SparseTail {
                axx,
                a22_slots,
                diag_slots,
                fam_dd,
                ..
            } = tail;
            // Both per-eval accumulators reset together: `vals` takes A22 + I
            // here, `acc` (dense route only) collects the family downdates and
            // is gathered into `vals` after the family loop.
            if let FamDowndate::Dense { acc, .. } = fam_dd {
                acc.fill(0.0);
            }
            let (_, vals) = axx.parts_mut();
            vals.fill(0.0);
            let mut cur = 0usize;
            let mut si = 0usize;
            for pr in a22_pairs.iter() {
                let bri = &lam_blocks[pr[0] as usize];
                let bcj = &lam_blocks[pr[1] as usize];
                let (ti0, tj0) = (bri.start - kf, bcj.start - kf);
                let blk = &pk_a22[cur..cur + bri.q * bcj.q];
                cur += bri.q * bcj.q;
                for jl in 0..bcj.q {
                    fold_packed_col(
                        blk,
                        bri.q,
                        bcj.q,
                        lam_small,
                        bri.lam_off,
                        bcj.lam_off,
                        jl,
                        |il, v| {
                            if ti0 + il >= tj0 + jl {
                                vals[a22_slots[si] as usize] += v;
                                si += 1;
                            }
                        },
                    );
                }
            }
            debug_assert_eq!(si, a22_slots.len(), "a22 slot replay exhausted");
            for &s in diag_slots.iter() {
                vals[s as usize] += 1.0;
            }
        }
    }

    let mut log_lzz_half = 0.0_f64;
    for f in 0..n_prim {
        // A_f = Λ_f'G_fΛ_f + I (row-major lower), assembled block-pair-wise
        // off the packed family stream — (primary, primary), each
        // (child, primary) coupling, each child's own diagonal block; sibling
        // child–child blocks are structurally zero (children never share rows
        // — mirror lmm.rs:1631) and zero-filled. Then Crout in place → L_f
        // (the `reml_deviance` family-Crout pattern, lmm.rs:1668-1687).
        let fam_gram = &pk_fam[f * fam_len..(f + 1) * fam_len];
        let pb = &lam_blocks[f];
        let lo_p = pb.lam_off;
        for jl in 0..q_p {
            fold_packed_col(
                &fam_gram[..q_p * q_p],
                q_p,
                q_p,
                lam_small,
                lo_p,
                lo_p,
                jl,
                |il, v| {
                    if il >= jl {
                        fam_a[il * w + jl] = if il == jl { v + 1.0 } else { v };
                    }
                },
            );
        }
        for c in 0..np {
            let lo_n = lam_blocks[n_prim + f * np + c].lam_off;
            let b2 = q_p * q_p + c * (q_n * q_p + q_n * q_n);
            let rc = q_p + c * q_n;
            for jl in 0..q_p {
                fold_packed_col(
                    &fam_gram[b2..b2 + q_n * q_p],
                    q_n,
                    q_p,
                    lam_small,
                    lo_n,
                    lo_p,
                    jl,
                    |il, v| {
                        fam_a[(rc + il) * w + jl] = v;
                    },
                );
            }
            for c2 in 0..c {
                let rc2 = q_p + c2 * q_n;
                for jl in 0..q_n {
                    for il in 0..q_n {
                        fam_a[(rc + il) * w + (rc2 + jl)] = 0.0;
                    }
                }
            }
            let b3 = b2 + q_n * q_p;
            for jl in 0..q_n {
                fold_packed_col(
                    &fam_gram[b3..b3 + q_n * q_n],
                    q_n,
                    q_n,
                    lam_small,
                    lo_n,
                    lo_n,
                    jl,
                    |il, v| {
                        if il >= jl {
                            fam_a[(rc + il) * w + (rc + jl)] = if il == jl { v + 1.0 } else { v };
                        }
                    },
                );
            }
        }
        for j in 0..w {
            let mut d = fam_a[j * w + j];
            for k in 0..j {
                let v = fam_a[j * w + k];
                d -= v * v;
            }
            if !(d.is_finite() && d > 0.0) {
                return None;
            }
            let l = d.sqrt();
            fam_a[j * w + j] = l;
            log_lzz_half += l.ln();
            for i in (j + 1)..w {
                let mut v = fam_a[i * w + j];
                for k in 0..j {
                    v -= fam_a[i * w + k] * fam_a[j * w + k];
                }
                fam_a[i * w + j] = v / l;
            }
        }
        // The factor this iteration is about to overwrite for the next family —
        // kept only when the post-fit recovery asked for it (see `SparseRecovery`).
        if let Some(r) = rec.as_mut() {
            r.fam_l[f * w * w..(f + 1) * w * w].copy_from_slice(&fam_a[..w * w]);
        }
        // U1 family rows: B1_f = (Λ'Z'[X y])[family rows] — row r is
        // Σ_{a≥il} Λ[a,il]·ztxy_row(col_a) over its owning block (mirrors
        // `reml_deviance_blocked`'s P_zx = ΛᵀZᵀ[Xy], lmm.rs:1291-1303), read
        // off the row-major ztxy — then forward-solved by L_f (col-major
        // kf-stride Vec; the row ops span m ≤ p+1 columns).
        let fb = f * w;
        for r in 0..w {
            let (blkr, il) = if r < q_p {
                (pb, r)
            } else {
                let rr = r - q_p;
                (&lam_blocks[n_prim + f * np + rr / q_n], rr % q_n)
            };
            for c in 0..m {
                u1[c * kf + fb + r] = 0.0;
            }
            for a in il..blkr.q {
                let la = lam_small[blkr.lam_off + a * blkr.q + il];
                let zr = (blkr.start + a * blkr.stride) * m;
                for c in 0..m {
                    u1[c * kf + fb + r] += la * ztxy[zr + c];
                }
            }
        }
        for r in 0..w {
            for k in 0..r {
                let lrk = fam_a[r * w + k];
                for c in 0..m {
                    u1[c * kf + fb + r] -= lrk * u1[c * kf + fb + k];
                }
            }
            let lrr = fam_a[r * w + r];
            for c in 0..m {
                u1[c * kf + fb + r] /= lrr;
            }
        }
        // L21 family block, two phases. Phase A: raw A21 for ALL w family
        // columns in ONE pass over the family's co-occurring blocks (each
        // block's merged slab carries the pb columns then each child's — one
        // block-loop instead of w; per component row a, the scalar
        // h = G_row·λ_c stays in a register and fans out to a's lower Λ_x
        // column). Non-co-occurring blocks stay at the zero-fill; each raw
        // entry belongs to exactly one block and keeps its (a, b) accumulation
        // order, so this equals the per-column form bit for bit.
        let fam_blks = &a21_blk[a21_off[f]..a21_off[f + 1]];
        let a21_fam = &pk_a21[pk_a21_off[f]..pk_a21_off[f + 1]];
        // Phase A A21-fold, shared by both tail arms. For every co-occurring
        // block the h = G_row·λ_c scalar stays in a register and fans out to the
        // block's lower Λ_x column; the h-accumulation and block-column dispatch
        // are identical across arms — only the destination index differs. Dense
        // arm: scatter into the global e-row `l21` at column `fb+c0` (stride `e`),
        // block rows at `t0 = start-kf`. Panel arm: pack into the compact
        // `e_f`-row panel at column `c0` (stride `e_f`), block rows at the running
        // `loc0`. `panel_layout` picks the row base once per block (not per entry).
        // Non-co-occurring blocks stay at the caller's zero-fill; each raw entry
        // belongs to one block and keeps its (a, b) accumulation order, so this
        // equals the per-column form bit for bit.
        let fold_a21 = |out: &mut [f64], stride: usize, col_off: usize, panel_layout: bool| {
            let mut cur = 0usize;
            let mut loc0 = 0usize;
            for &bi in fam_blks {
                let br = &lam_blocks[bi as usize];
                let (q, lo_x) = (br.q, br.lam_off);
                let row_base = if panel_layout { loc0 } else { br.start - kf };
                for a in 0..q {
                    // col-block 0 = primary, 1..=np the children (slab layout
                    // mirrors `new`'s merged packing — change together).
                    for cb in 0..=np {
                        let (q_c, lo_c, c0, roff) = if cb == 0 {
                            (q_p, lo_p, 0, cur + a * q_p)
                        } else {
                            let ch = cb - 1;
                            (
                                q_n,
                                lam_blocks[n_prim + f * np + ch].lam_off,
                                q_p + ch * q_n,
                                cur + q * q_p + ch * q * q_n + a * q_n,
                            )
                        };
                        for jl in 0..q_c {
                            let mut h = 0.0;
                            for b in jl..q_c {
                                h += a21_fam[roff + b] * lam_small[lo_c + b * q_c + jl];
                            }
                            let colbase = (col_off + c0 + jl) * stride;
                            for il in 0..=a {
                                out[colbase + row_base + il] += lam_small[lo_x + a * q + il] * h;
                            }
                        }
                    }
                }
                cur += q * w;
                loc0 += q;
            }
        };
        match tail.as_mut() {
            None => {
                l21[fb * e..(fb + w) * e].fill(0.0);
                fold_a21(&mut l21[..], e, fb, false);
                // Phase B tri-solve on the family's w columns of the global l21.
                schur_phase_b(&mut l21[fb * e..(fb + w) * e], e, w, fam_a);
            }
            Some(SparseTail {
                axx,
                fam_dd,
                panel,
                dd_temp,
                b2_temp,
                ..
            }) => {
                // Sparse tail: the family's L21 block never joins a global
                // e-row matrix — it lives on a compact `e_f × w` col-major
                // panel (`e_f` = Σ q over the family's co-occurring crossed
                // blocks; block bi's rows sit at loc0..loc0+q instead of
                // t0..t0+q) and is consumed within this iteration: S22
                // downdate scatter + B̃₂ downdate.
                let e_f: usize = fam_blks.iter().map(|&bi| lam_blocks[bi as usize].q).sum();
                let panel = &mut panel[..e_f * w];
                panel.fill(0.0);
                fold_a21(panel, e_f, 0, true);
                schur_phase_b(panel, e_f, w, fam_a);
                // Consumed within this iteration, so the recovery copies it out
                // here or never sees it (the dense branch keeps the global `l21`
                // and needs no copy).
                if let Some(r) = rec.as_mut() {
                    r.panel[r.panel_off[f]..r.panel_off[f + 1]].copy_from_slice(panel);
                }
                // S22 −= L21_f·L21_fᵀ, by whichever route `fam_dd` chose. Both
                // arms compute the same lower `panel·panelᵀ` — a RESULT-MOVING
                // reassociation of the per-entry dots, sanctioned as the dense
                // tail's triangular `S22 −= L21·L21ᵀ` matmul below, just before
                // its `Dense tail LLT`; the small contraction width `w` — 1–2
                // for random-intercept crossed factors — means the win is on the
                // `e_f` output dimension. They differ in where the entries land.
                //
                // At `w == 1` the triangular matmul is a rank-1 lower outer
                // product, and a direct hand loop runs at ~half faer's time —
                // faer pays general BLAS-3 setup for a single multiply per
                // entry. Every crossN cell takes that path (`fam_w` is a global
                // scalar; all `(1|g)` random intercepts). It is bit-identical to
                // the faer arm at `w == 1` (one product per lower entry, no
                // summation to reassociate), so parity numbers do not move; the
                // `w ≥ 2` arm keeps the tuned BLAS-3 path for slope-carrying
                // designs.
                match fam_dd {
                    // Scatter arm: stage into `dd_temp`, then replay through the
                    // per-family slot list. The column-major exclusive order (`b`
                    // outer, `a ≥ b` inner) mirrors the slot generation in
                    // `build_sparse_tail` — change together — and reads dd
                    // stride-1 (col b's lower entries are contiguous at
                    // `b·e_f + b..b·e_f + e_f`); the per-column `split_at` gives
                    // the zip exact lengths, killing the bounds checks.
                    FamDowndate::Scatter { slots, off } => {
                        let dd = &mut dd_temp[..e_f * e_f];
                        if w == 1 {
                            let p = &panel[..e_f];
                            for b in 0..e_f {
                                let pb = p[b];
                                let col = &mut dd[b * e_f..b * e_f + e_f];
                                for a in b..e_f {
                                    col[a] = p[a] * pb;
                                }
                            }
                        } else {
                            let panel_ref =
                                MatRef::from_column_major_slice(&panel[..e_f * w], e_f, w);
                            faer::linalg::matmul::triangular::matmul(
                                faer::MatMut::from_column_major_slice_mut(dd, e_f, e_f),
                                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                                faer::Accum::Replace,
                                panel_ref,
                                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                                panel_ref.transpose(),
                                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                                1.0,
                                Par::Seq,
                            );
                        }
                        let (_, vals) = axx.parts_mut();
                        let mut rest = &slots[off[f]..off[f + 1]];
                        for b in 0..e_f {
                            let (col, tail_slots) = rest.split_at(e_f - b);
                            rest = tail_slots;
                            for (&s, &d) in col.iter().zip(&dd[b * e_f + b..b * e_f + e_f]) {
                                vals[s as usize] -= d;
                            }
                        }
                        debug_assert!(rest.is_empty(), "family downdate slot replay exhausted");
                    }
                    // Dense arm: `acc` column `rmap[b]` is contiguous and stays
                    // resident for the whole column sweep, and the only index
                    // stream is `rmap` — `e_f` words, against the scatter's
                    // `e_f(e_f+1)/2`. At `w == 1` the rank-1 update goes straight
                    // in, so `dd_temp` is neither written nor read (its buffer is
                    // sized to zero in `build_sparse_tail` — change together).
                    // `acc` collects `+Σ_f L21_f·L21_fᵀ`; the sign is applied
                    // once by the gather after the family loop.
                    FamDowndate::Dense { rows, off, acc } => {
                        let rmap = &rows[off[f]..off[f + 1]];
                        debug_assert_eq!(rmap.len(), e_f, "dense row map width");
                        if w == 1 {
                            let p = &panel[..e_f];
                            for b in 0..e_f {
                                let pb = p[b];
                                let col = &mut acc[rmap[b] as usize * e..];
                                for a in b..e_f {
                                    col[rmap[a] as usize] += p[a] * pb;
                                }
                            }
                        } else {
                            let dd = &mut dd_temp[..e_f * e_f];
                            let panel_ref =
                                MatRef::from_column_major_slice(&panel[..e_f * w], e_f, w);
                            faer::linalg::matmul::triangular::matmul(
                                faer::MatMut::from_column_major_slice_mut(dd, e_f, e_f),
                                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                                faer::Accum::Replace,
                                panel_ref,
                                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                                panel_ref.transpose(),
                                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                                1.0,
                                Par::Seq,
                            );
                            for b in 0..e_f {
                                let col = &mut acc[rmap[b] as usize * e..];
                                for (&r, &d) in
                                    rmap[b..].iter().zip(&dd[b * e_f + b..b * e_f + e_f])
                                {
                                    col[r as usize] += d;
                                }
                            }
                        }
                    }
                }
                // B̃₂ −= L21_f·U1_f (this family's rows of the B2 downdate). The
                // per-block scalar gemm is now one `panel·U1_sub` into `b2_temp`
                // (Accum::Replace; RESULT-MOVING reassociation, sanctioned as
                // the dense tail's `B̃₂ = B2 − L21·U1` matmul below its
                // LLT). U1_sub is rows
                // fb..fb+w of U1 taken as a strided `w×m` view (row-stride 1,
                // col-stride kf, base `&u1[fb..]`) — no copy — so `u1` feeds the
                // gemm vectorized; the family layout guarantees fb+w ≤ kf, which
                // is exactly the strided-view fits-assert. Then the per-block
                // scatter-subtract preserves the existing target row mapping.
                let panel_ref = MatRef::from_column_major_slice(&panel[..e_f * w], e_f, w);
                let u1_sub = MatRef::from_column_major_slice_with_stride(&u1[fb..], w, m, kf);
                faer::linalg::matmul::matmul(
                    faer::MatMut::from_column_major_slice_mut(&mut b2_temp[..e_f * m], e_f, m),
                    faer::Accum::Replace,
                    panel_ref,
                    u1_sub,
                    1.0,
                    Par::Seq,
                );
                let b2 = &b2_temp[..e_f * m];
                let mut loc0 = 0usize;
                for &bi in fam_blks {
                    let br = &lam_blocks[bi as usize];
                    let t0 = br.start - kf;
                    for a in 0..br.q {
                        for cm in 0..m {
                            // b2 col-major e_f×m: entry (loc0+a, cm) at loc0+a + cm·e_f.
                            u2[cm * e + t0 + a] -= b2[loc0 + a + cm * e_f];
                        }
                    }
                    loc0 += br.q;
                }
            }
        }
    }

    if e > 0 {
        match tail.as_mut() {
            None => {
                // S22 = A22 − L21·L21ᵀ (lower): zero-fill, fold only the co-occurring
                // crossed block pairs (same-factor level pairs never co-occur, so a
                // single crossed factor's A22 is block-DIAGONAL — most pairs skip),
                // add +I, then ONE triangular syrk downdate through faer's blocked
                // FMA kernels (same call shape as lmm.rs:1811-1821; RESULT-MOVING
                // reassociation vs entry-wise subtraction, sanctioned as there).
                for tj in 0..e {
                    for ti in tj..e {
                        s22[(ti, tj)] = 0.0;
                    }
                }
                let mut cur = 0usize;
                for pr in a22_pairs.iter() {
                    let bri = &lam_blocks[pr[0] as usize];
                    let bcj = &lam_blocks[pr[1] as usize];
                    let (ti0, tj0) = (bri.start - kf, bcj.start - kf);
                    let blk = &pk_a22[cur..cur + bri.q * bcj.q];
                    cur += bri.q * bcj.q;
                    for jl in 0..bcj.q {
                        fold_packed_col(
                            blk,
                            bri.q,
                            bcj.q,
                            lam_small,
                            bri.lam_off,
                            bcj.lam_off,
                            jl,
                            |il, v| {
                                let (ti, tj) = (ti0 + il, tj0 + jl);
                                if ti >= tj {
                                    s22[(ti, tj)] = v;
                                }
                            },
                        );
                    }
                }
                for t in 0..e {
                    s22[(t, t)] += 1.0;
                }
                let l21_ref = MatRef::from_column_major_slice(&l21[..e * kf], e, kf);
                faer::linalg::matmul::triangular::matmul(
                    s22.as_mat_mut(),
                    faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                    faer::Accum::Add,
                    l21_ref,
                    faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                    l21_ref.transpose(),
                    faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                    -1.0,
                    Par::Seq,
                );
                // Dense tail LLT in place → L22 (lower; the stale upper is never read).
                cholesky_in_place(
                    s22.as_mat_mut(),
                    LltRegularization::default(),
                    Par::Seq,
                    MemStack::new(tail_llt_mem),
                    Spec::default(),
                )
                .ok()?; // non-positive tail pivot at this θ
                for t in 0..e {
                    let ltt = s22[(t, t)];
                    if !(ltt.is_finite() && ltt > 0.0) {
                        return None;
                    }
                    log_lzz_half += ltt.ln();
                }
                // U2 tail rows: B̃₂ = B2 − L21·U1 (B2 pre-filled above),
                // forward-solved by L22 column-oriented — per RHS column,
                // subtract L22's contiguous column k scaled by the just-final
                // x[k] (unit stride on both sides).
                faer::linalg::matmul::matmul(
                    faer::MatMut::from_column_major_slice_mut(&mut u2[..e * m], e, m),
                    faer::Accum::Add,
                    l21_ref,
                    MatRef::from_column_major_slice(&u1[..kf * m], kf, m),
                    -1.0,
                    Par::Seq,
                );
                // k-outer / c-inner: the column view `s22.col(k)` depends only on
                // k, so hoist it out of the RHS-column loop (mirrors the `l21_ref`
                // hoist above). The per-column forward solve stays k-ascending, so
                // swapping the independent c loop outward is bit-identical.
                for k in 0..e {
                    let s22k = s22.col(k).try_as_col_major().unwrap().as_slice();
                    for c in 0..m {
                        let col = &mut u2[c * e..(c + 1) * e];
                        let xk = col[k] / s22k[k];
                        col[k] = xk;
                        for (x, &s) in col[k + 1..].iter_mut().zip(&s22k[k + 1..]) {
                            *x -= s * xk;
                        }
                    }
                }
            }
            Some(tail) => {
                // Dense downdate route: the family loop accumulated
                // Σ_f L21_f·L21_fᵀ in `acc` instead of applying it per family,
                // so subtract it here in one sequential pass over the stored
                // pattern. This is the reassociation the route buys: the
                // scatter applies `((s − c₁) − c₂) − c₃` family by family, this
                // applies `s − (c₁ + c₂ + c₃)`. Only stored entries are read —
                // `acc`'s strictly-upper half is never written and never
                // gathered, and the pattern's lower-only storage keeps it that
                // way.
                if let SparseTail {
                    axx,
                    fam_dd: FamDowndate::Dense { acc, .. },
                    ..
                } = &mut *tail
                {
                    let (sym, vals) = axx.parts_mut();
                    let col_ptr = sym.col_ptr();
                    let row_idx = sym.row_idx();
                    for j in 0..e {
                        let base = j * e;
                        for k in col_ptr[j]..col_ptr[j + 1] {
                            vals[k] -= acc[base + row_idx[k]];
                        }
                    }
                }
                // Sparse tail: the CSC values already hold S22 = A22 + I −
                // L21·L21ᵀ (assembly before the family loop, downdates inside
                // it); numeric-refactor on the stored symbolic. A reordered
                // (AMD) elimination is a reassociation of the same Cholesky —
                // non-PD at this θ surfaces as Err here or a non-finite
                // logdet below, the same INFINITY surface as the dense arm
                // *by design* (reassociation-equivalent, not bit-identical: a
                // borderline pivot can flip).
                let llt = tail
                    .symbolic
                    .factorize_numeric_llt(
                        &mut tail.l_values,
                        tail.axx.as_ref(),
                        Side::Lower,
                        LltRegularization::default(),
                        Par::Seq,
                        MemStack::new(&mut tail.fac_mem),
                        Spec::default(),
                    )
                    .ok()?;
                // No arm-agnostic forward-only solve exists on `LltRef`, and
                // none is needed: the deviance wants only UᵀU = B̃₂ᵀS22⁻¹B̃₂,
                // which is permutation-invariant — copy B̃₂ (u2) and full-solve
                // X = S22⁻¹·B̃₂ in place (permutation handled internally); the
                // UᵀU dots below pair u2 with x2.
                for c in 0..m {
                    for t in 0..e {
                        tail.x2[(t, c)] = u2[c * e + t];
                    }
                }
                llt.solve_in_place_with_conj(
                    Conj::No,
                    tail.x2.as_mat_mut(),
                    Par::Seq,
                    MemStack::new(&mut tail.solve_mem),
                );
                let _ = llt; // ends the &'out borrow on l_values (LltRef is Copy; NLL)
                             // logdet_llt returns 2·Σ ln L_ii (+INFINITY on a non-PD
                             // diagonal); log_lzz_half is the ½ convention — add HALF.
                let log_s22 = logdet_llt(&tail.symbolic, &tail.l_values);
                if !log_s22.is_finite() {
                    return None;
                }
                log_lzz_half += 0.5 * log_s22;
            }
        }
    }

    // S = C_xy − UᵀU (lower), Cholesky in place → the augmented factor L.
    // The UᵀU assembly is now two triangular syrk downdates — a RESULT-MOVING
    // reassociation of the per-entry U1ᵀU1 / U2ᵀU2 dots, sanctioned exactly as
    // the dense tail's triangular `S22 −= L21·L21ᵀ` matmul above (`s22`
    // None-arm, just before its `Dense tail LLT`) — and
    // `cholesky_in_place` replaces the hand Crout with the same call the dense
    // tail uses for L22. Every write touches only the lower triangle, so the
    // once-zeroed upper stays zero and the recovery's lower-only reads see a
    // proper lower-triangular factor without a re-zero.
    for c in 0..m {
        for r in 0..m {
            factor[(r, c)] = 0.0;
        }
    }
    for c in 0..m {
        for r in c..m {
            factor[(r, c)] = cxy[(r, c)];
        }
    }
    let u1_ref = MatRef::from_column_major_slice(&u1[..kf * m], kf, m);
    faer::linalg::matmul::triangular::matmul(
        factor.as_mat_mut(),
        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        faer::Accum::Add,
        u1_ref.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        u1_ref,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        -1.0,
        Par::Seq,
    );
    // Tail rows of UᵀU: dense arm subtracts U2ᵀU2; sparse arm subtracts
    // B̃₂ᵀ·X2 (u2 holds B̃₂, tail.x2 holds S22⁻¹B̃₂ ⇒ B̃₂ᵀS22⁻¹B̃₂, symmetric, so
    // the triangular call writes exactly the lower entries the scalar dot did).
    let u2_ref = MatRef::from_column_major_slice(&u2[..e * m], e, m);
    let rhs2 = match tail.as_ref() {
        None => u2_ref,
        Some(tail) => tail.x2.as_ref(),
    };
    faer::linalg::matmul::triangular::matmul(
        factor.as_mat_mut(),
        faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
        faer::Accum::Add,
        u2_ref.transpose(),
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        rhs2,
        faer::linalg::matmul::triangular::BlockStructure::Rectangular,
        -1.0,
        Par::Seq,
    );
    // .ok()? preserves the hand Crout's non-PD exit: faer returns Err on any
    // pivot failing `pivot > 0` — non-positive AND non-finite both trip it.
    cholesky_in_place(
        factor.as_mat_mut(),
        LltRegularization::default(),
        Par::Seq,
        MemStack::new(factor_llt_mem),
        Spec::default(),
    )
    .ok()?;
    Some(2.0 * log_lzz_half)
}

/// Spherical conditional modes `û` at θ̂ on the sparse-Z LMM path, over the full
/// RE-column set in `build_sparse_z` order — the layout
/// `fit::common::assemble_ranef_sparse` reads.
///
/// Same identity the dense path recovers (`lmm::recover_ranef`), routed through
/// the block factors this path actually keeps. With `c = (−β̂; 1)` so that
/// `B·c = Λ′Z′(y − Xβ̂)`, the penalized normal equations `A u = B·c` split as
///
/// ```text
/// û₂ = S₂₂⁻¹ (B̃₂·c),        û₁ = L₁₁⁻ᵀ (U₁·c − L₂₁ᵀ û₂)
/// ```
///
/// and every factor on the right survives the final evaluation: `U₁` is
/// `ws.u1`, `L₁₁`'s per-family blocks and (sparse-tail) `L₂₁`'s panels are what
/// [`SparseRecovery`] held back, and `S₂₂⁻¹B̃₂` is `tail.x2` outright on the
/// sparse-tail branch — where `û₂` is therefore a plain matrix–vector product,
/// no solve at all. On the dense-tail branch `ws.u2` holds `U₂ = L₂₂⁻¹B̃₂`, so
/// `û₂ = L₂₂⁻ᵀ(U₂·c)` is one back-substitution against the in-place `ws.s22`.
///
/// Caller contract: the last thing run on `ws` was an evaluation at θ̂ with
/// `rec` armed. Returns `None` if it was not armed, or on a non-positive pivot.
fn sparse_recover_u(ws: &SparseLmmWorkspace, beta: &[f64]) -> Option<Vec<f64>> {
    let rec = ws.rec.as_ref()?;
    let g = &ws.g;
    let p = ws.p;
    let w = ws.fam_w;
    let kf = g.k_family();
    let e = g.k_crossed();
    let n_prim = g.n_primary;
    let q_p = g.primary_q;
    let np = g.nested_per_parent;
    let q_n = g.nested.map(|nf| nf.q).unwrap_or(0);
    // c = (−β̂; 1): contracting any `[X y]`-columned block against it turns it
    // into that block's residual column.
    let dot_c = |row: &dyn Fn(usize) -> f64| -> f64 {
        let mut acc = row(p);
        for (j, &b) in beta.iter().enumerate().take(p) {
            acc -= row(j) * b;
        }
        acc
    };

    // --- crossed block ---
    let mut u2 = vec![0.0f64; e];
    if e > 0 {
        match ws.tail.as_ref() {
            Some(tail) => {
                for (t, slot) in u2.iter_mut().enumerate() {
                    *slot = dot_c(&|c| tail.x2[(t, c)]);
                }
            }
            None => {
                for (t, slot) in u2.iter_mut().enumerate() {
                    *slot = dot_c(&|c| ws.u2[c * e + t]);
                }
                for t in (0..e).rev() {
                    let mut acc = u2[t];
                    for (i, &solved) in u2.iter().enumerate().skip(t + 1) {
                        acc -= ws.s22[(i, t)] * solved;
                    }
                    let ltt = ws.s22[(t, t)];
                    if !(ltt.is_finite() && ltt > 0.0) {
                        return None;
                    }
                    u2[t] = acc / ltt;
                }
            }
        }
    }

    // --- family blocks: L_fᵀ û₁_f = (U₁·c)_f − (L₂₁ᵀ û₂)_f ---
    let mut u = vec![0.0f64; g.k_total];
    for (t, &v) in u2.iter().enumerate() {
        u[kf + t] = v;
    }
    let mut rhs = vec![0.0f64; w];
    for f in 0..n_prim {
        let fb = f * w;
        for (r, slot) in rhs.iter_mut().enumerate() {
            *slot = dot_c(&|c| ws.u1[c * kf + fb + r]);
        }
        if e > 0 {
            match ws.tail.as_ref() {
                // Panel rows are family-local: block `bi`'s `q` rows sit at the
                // running `loc0`, and its crossed rows at `start − k_family`.
                Some(_) => {
                    let panel = &rec.panel[rec.panel_off[f]..rec.panel_off[f + 1]];
                    let fam_blks = &ws.a21_blk[ws.a21_off[f]..ws.a21_off[f + 1]];
                    let e_f = panel.len().checked_div(w).unwrap_or(0);
                    let mut loc0 = 0usize;
                    for &bi in fam_blks {
                        let br = &ws.lam_blocks[bi as usize];
                        let t0 = br.start - kf;
                        for a in 0..br.q {
                            let ut = u2[t0 + a];
                            if ut != 0.0 {
                                for (r, slot) in rhs.iter_mut().enumerate() {
                                    *slot -= panel[r * e_f + loc0 + a] * ut;
                                }
                            }
                        }
                        loc0 += br.q;
                    }
                }
                None => {
                    for (r, slot) in rhs.iter_mut().enumerate() {
                        let col = &ws.l21[(fb + r) * e..(fb + r + 1) * e];
                        for (t, &ut) in u2.iter().enumerate() {
                            *slot -= col[t] * ut;
                        }
                    }
                }
            }
        }
        let l = &rec.fam_l[fb * w..(fb + w) * w];
        for r in (0..w).rev() {
            let mut acc = rhs[r];
            for (i, &solved) in rhs.iter().enumerate().skip(r + 1) {
                acc -= l[i * w + r] * solved;
            }
            let lrr = l[r * w + r];
            if !(lrr.is_finite() && lrr > 0.0) {
                return None;
            }
            rhs[r] = acc / lrr;
        }
        // Family row r → RE column, the same map the U1 fill above walks:
        // primary component r at `r·n_primary + f`, child `ch` component `il`
        // at its own contiguous block start.
        for (r, &v) in rhs.iter().enumerate() {
            let re_col = if r < q_p {
                r * n_prim + f
            } else {
                let rr = r - q_p;
                let br = &ws.lam_blocks[n_prim + f * np + rr / q_n];
                br.start + (rr % q_n) * br.stride
            };
            u[re_col] = v;
        }
    }
    Some(u)
}

/// TEST ONLY: the deterministic LCG the LMM tests use for reproducible designs
/// (copied from `lmm.rs`'s test `lcg`). Yields a value in `(-1, 1)`.
#[cfg(test)]
pub(crate) fn test_lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
}
