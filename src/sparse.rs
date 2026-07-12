//! Sparse-Z LMM solver. A second Gaussian-mixed path that factors
//! `Λ'Z'ZΛ + I` with a two-level blocked Cholesky specialized to this
//! path's structural class — primary(+nested) FAMILIES are block-diagonal
//! (levels never co-occur across families) and ALL crossed-extra columns form
//! one tail (dense for `e ≤ TAIL_SPARSE_MIN`, fill-reducing sparse above) —
//! lifting the `MAX_*` caps that bound the dense no-Z tail.
//! Mirrors `lmm::reml_deviance` (`lmm.rs:1396`) one level down; validated
//! against it by the both-paths cross-check (`sparse` tests + `glmm/tests.rs`).
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
use faer::linalg::cholesky::llt::solve::solve_in_place;
use faer::mat::AsMatMut;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholesky,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Conj, Mat, MatRef, Par, Side, Spec};

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
    let g = LmmGroupings::from_cluster_spec_ext(model, n, &slope_cols, &extra_slope_cols);

    // Row-major f64 `x` viewed column-agnostically as an n×p faer MatRef (Z/Gram
    // builders index it as `x[(i, j)]`).
    let xm = MatRef::from_row_major_slice(x, n, p);
    // WLS-style √wᵢ pre-scaling, same convention as `fit_mle`'s dense path
    // (`add_rows_multi`'s `weights` arg): computed once here, threaded through
    // every z-emission and raw x/y read in `SparseLmmWorkspace::new`.
    let sqrt_w: Option<Vec<f64>> = opts
        .weights
        .as_ref()
        .map(|w| w.iter().map(|v| v.sqrt()).collect());
    let mut ws =
        SparseLmmWorkspace::new(&g, xm, cluster_ids, extra_ids, y, n, p, sqrt_w.as_deref());

    // θ seed + per-component boxes + solver — topology-only, byte-identical to the
    // NoZ path (the superset property depends on this).
    let (mut solver, mut theta, lower, upper) = crate::lmm::sparse_lmm_seed(&g);
    // Cold start = blind seed (diagonals THETA0, off-diagonals 0 — mirror
    // `fit_lmm`'s cold arm, see the basin rationale there); a warm start
    // clamps to the truth floor (mirror `fit_lmm` `lmm.rs:1888-1900`).
    match start {
        Some(s) => {
            debug_assert_eq!(s.theta.len(), theta.len());
            for (t, &v) in theta.iter_mut().zip(&s.theta) {
                *t = v.max(crate::lmm::THETA_TRUTH_FLOOR);
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
    let ok = matches!(out.status, Status::Converged);

    // Per-component deterministic pin: every DIAGONAL variance component ≤ PIN_THETA
    // collapses to exactly 0 so tau2/varcorr reflect the boundary (mirror `fit_lmm`
    // `lmm.rs:1917-1927`). `Fit` carries no `pinned_components` mask (unlike the
    // internal `LmmFit`), so `singular` only needs the any-pinned bit, not the mask.
    let mut pinned = false;
    if ok {
        for &ti in g.diagonal_theta() {
            if theta[ti] <= crate::lmm::PIN_THETA {
                theta[ti] = 0.0;
                pinned = true;
            }
        }
    }

    // Final eval at θ̂ (post-pin) → augmented Schur factor L in `ws.factor`;
    // rank-guard the p×p fixed block (mirror `fit_lmm` `lmm.rs:1932-1940`).
    let factor_ok = ok && sparse_schur_factor(&theta, &mut ws).is_some();
    let degenerate = if factor_ok {
        crate::ols::chol_rank_deficient(ws.factor.as_ref(), p, crate::lmm::EPS_RANK)
    } else {
        true
    };
    let converged = ok && !degenerate;

    if !converged {
        return crate::Fit {
            beta: vec![f64::NAN; p],
            se: vec![f64::NAN; p],
            tau2: theta.iter().map(|_| f64::NAN).collect(),
            dispersion: 1.0,
            converged: false,
            varcorr: vec![],
            stddev_se: vec![],
            aliased: vec![false; p],
            n_eval: 0,
            deviance: f64::NAN,
            singular: false,
        };
    }

    // Accepted objective at θ̂ post-pin: no `dev` local survives from the BOBYQA
    // loop here (unlike `fit_lmm`'s `dev`), so re-evaluate at the pinned θ — a
    // second Schur factor, but only once per fit (not the hot loop).
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

    // tau2[k] = θ̂[k]²·σ̂²; varcorr = vech(σ̂²·Λ̂Λ̂') per grouping — the path-independent
    // assembly shared with `fit_mle` (`fit.rs`).
    let tau2: Vec<f64> = theta.iter().map(|&t| t * t * sigma_sq).collect();
    let varcorr = crate::fit::assemble_varcorr(&theta, &g, sigma_sq);

    // Same −Σlog wᵢ deviance-constant convention as `fit_mle` (`fit.rs`,
    // Task 5): the weighted Gaussian log-density's +½Σlog wᵢ per row, on the
    // −2ℓ scale, added post-optimization (θ-independent — argmin unchanged).
    let dev = match &opts.weights {
        Some(w) => dev - w.iter().map(|v| v.ln()).sum::<f64>(),
        None => dev,
    };

    crate::Fit {
        beta,
        se,
        tau2,
        dispersion: 1.0,
        converged: true,
        varcorr,
        stddev_se: vec![],
        aliased: vec![false; p],
        n_eval: out.n_eval,
        deviance: dev,
        singular: pinned,
    }
}

/// `log det(A) = 2·Σ_j log L[j,j]` for the LLT factor, reading the diagonal from
/// the simplicial CSC `L_values` (faer 0.24 exposes no diagonal accessor on
/// `LltRef`). Requires the symbolic factor to be simplicial (the GLMM
/// sparse-Schur workspace — since the blocked LMM kernel, the sole production
/// caller — pins `SupernodalThreshold::FORCE_SIMPLICIAL`,
/// `glmm/workspace.rs:353`), so `symbolic.raw()` is the
/// `Simplicial` arm and its `col_ptr()`/`row_idx()` give the CSC layout of
/// `l_values`. Returns `+INFINITY` if any diagonal is non-positive/non-finite
/// (mirrors the dense evaluators' non-PD sentinel, `lmm.rs:1795`).
pub(crate) fn logdet_llt(symbolic: &SymbolicCholesky<usize>, l_values: &[f64]) -> f64 {
    use faer::sparse::linalg::cholesky::SymbolicCholeskyRaw;
    let simp = match symbolic.raw() {
        SymbolicCholeskyRaw::Simplicial(s) => s,
        SymbolicCholeskyRaw::Supernodal(_) => {
            unreachable!("logdet_llt requires FORCE_SIMPLICIAL symbolic factor")
        }
    };
    let col_ptr = simp.col_ptr();
    let row_idx = simp.row_idx();
    let n = col_ptr.len() - 1;
    let mut acc = 0.0f64;
    for j in 0..n {
        let mut ljj = f64::NAN;
        for k in col_ptr[j]..col_ptr[j + 1] {
            if row_idx[k] == j {
                ljj = l_values[k];
                break;
            }
        }
        // `ljj <= 0.0` handles negative and zero; `!is_finite()` covers NaN and ±Inf.
        // NaN is also caught here because `NaN.is_finite()` = false.
        if ljj <= 0.0 || !ljj.is_finite() {
            return f64::INFINITY;
        }
        acc += ljj.ln();
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

// Test-only override: force the sparse-tail branch for small-e fixtures so the
// dense↔sparse equality tests exercise the sparse factor at their existing
// tolerances. Thread-local (each #[test] runs on its own thread), read once in
// `SparseLmmWorkspace::new` — the branch is a construction-time decision.
#[cfg(test)]
thread_local! {
    pub(crate) static FORCE_SPARSE_TAIL: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

/// Sparse-tail state (branch `e > TAIL_SPARSE_MIN`): the fill-reducing (AMD)
/// simplicial Cholesky of the crossed Schur complement `S22 = A22 + I −
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
    /// Symbolic Cholesky of the S22 pattern (FORCE_SIMPLICIAL — required by
    /// `logdet_llt`, and right for the ~clique-wide columns; AMD ordering).
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
    /// Per family, CSC slot of every lower pair (a, b ≤ a) of its local crossed
    /// scalar columns, in the downdate's replay order (`a` outer, `b` inner);
    /// ragged via `fam_dd_off` (length n_primary+1).
    pub(crate) fam_dd_slots: Vec<u32>,
    pub(crate) fam_dd_off: Vec<usize>,
    /// Per-family compact L21 panel, col-major `e_f`-row columns
    /// (`panel[c·e_f + local]`), sized `fam_w · max_f e_f` once.
    pub(crate) panel: Vec<f64>,
    /// Scratch for the per-family S22 syrk downdate `panel·panelᵀ` (kernel B):
    /// col-major `e_f×e_f`, sized to `max_f e_f²` once, overwritten
    /// (`Accum::Replace`) then scattered through `fam_dd_slots`.
    pub(crate) dd_temp: Vec<f64>,
    /// Scratch for the per-family B̃₂ downdate `panel·U1_sub` (kernel C):
    /// col-major `e_f×m`, sized to `max_f e_f · m` once, overwritten
    /// (`Accum::Replace`) then block-scatter-subtracted into `u2`.
    pub(crate) b2_temp: Vec<f64>,
    /// `X = S22⁻¹·B̃₂` (`e × m`): B̃₂ copied in, full-solved in place per eval
    /// (the simplicial factor has no forward-only solve; the deviance needs
    /// only `UᵀU = B̃₂ᵀS22⁻¹B̃₂`, which is permutation-invariant).
    pub(crate) x2: Mat<f64>,
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
        }
    }
}

/// Build the θ-independent sparse-tail state (see `SparseTail`): the scalar S22
/// pattern (family cliques ∪ `a22_pairs`, plus the full diagonal), the
/// AMD/simplicial symbolic factor, and every scatter target pre-resolved to its
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
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
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
    // Per-family downdate slots over the family's local crossed scalar columns
    // (ascending, so local a ≥ b ⇒ global lower) — the kernel's replay order
    // (`a` outer, `b ≤ a` inner); mirrors the panel scatter — change together.
    let mut fam_dd_slots = Vec::new();
    let mut fam_dd_off = Vec::with_capacity(fam_crossed.len() + 1);
    fam_dd_off.push(0);
    let mut cols: Vec<usize> = Vec::new();
    let mut max_ef = 0usize;
    for list in fam_crossed {
        cols.clear();
        for &bi in list {
            let b = &lam_blocks[bi as usize];
            let t0 = b.start - kf;
            cols.extend(t0..t0 + b.q);
        }
        max_ef = max_ef.max(cols.len());
        for a in 0..cols.len() {
            for bl in 0..=a {
                fam_dd_slots.push(slot_of((cols[a], cols[bl])));
            }
        }
        fam_dd_off.push(fam_dd_slots.len());
    }
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
        fam_dd_slots,
        fam_dd_off,
        panel: vec![0.0f64; fam_w * max_ef],
        dd_temp: vec![0.0f64; max_ef * max_ef],
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
    emit(f, sw);
    for (k, &col) in g.primary_slope_cols.iter().enumerate() {
        emit((k + 1) * g.n_primary + f, sw * x[(i, col)]);
    }
    // Extra groupings: intercept at off, slope c at off+1+c.
    for (e, ids_e) in extra_ids.iter().enumerate() {
        let q_g = g.extra_q[e];
        let off = g.extra_offsets[e] + ids_e[i] as usize * q_g;
        emit(off, sw);
        for (c, &col) in g.extra_slope_cols[e].iter().enumerate() {
            emit(off + 1 + c, sw * x[(i, col)]);
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
fn fill_lambda_small(theta: &[f64], g: &LmmGroupings, lam_small: &mut [f64]) {
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
/// into a fixed CSC pattern, and faer's AMD/simplicial sparse LLT — the same
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
/// gated by the both-paths deviance-equality tests and the parity harness.
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
                ..
            } = tail;
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
        match tail.as_mut() {
            None => {
                l21[fb * e..(fb + w) * e].fill(0.0);
                let mut cur = 0usize;
                for &bi in fam_blks {
                    let br = &lam_blocks[bi as usize];
                    let (q, t0, lo_x) = (br.q, br.start - kf, br.lam_off);
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
                                let colbase = (fb + c0 + jl) * e;
                                for il in 0..=a {
                                    l21[colbase + t0 + il] += lam_small[lo_x + a * q + il] * h;
                                }
                            }
                        }
                    }
                    cur += q * w;
                }
                // Phase B: tri-solve each column against L_fᵀ as contiguous-column
                // axpys (the lmm.rs:1779 `fit.bt` split_at_mut pattern; columns
                // before fb+c are final when read).
                for c in 0..w {
                    let (done, rest) = l21.split_at_mut((fb + c) * e);
                    let col_c = &mut rest[..e];
                    for k in 0..c {
                        let l_ck = fam_a[c * w + k];
                        let col_k = &done[(fb + k) * e..(fb + k + 1) * e];
                        for (x, &y) in col_c.iter_mut().zip(col_k) {
                            *x -= l_ck * y;
                        }
                    }
                    // One reciprocal instead of e scalar divides (divsd is ~15 cycles
                    // unpipelined); the ≤1-ulp difference vs division is inside every
                    // parity band.
                    let inv_cc = 1.0 / fam_a[c * w + c];
                    for v in col_c.iter_mut() {
                        *v *= inv_cc;
                    }
                }
            }
            Some(SparseTail {
                axx,
                fam_dd_slots,
                fam_dd_off,
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
                // downdate scatter + B̃₂ downdate. Phase A/B mirror the dense
                // arm on the panel — change together.
                let e_f: usize = fam_blks.iter().map(|&bi| lam_blocks[bi as usize].q).sum();
                let panel = &mut panel[..e_f * w];
                panel.fill(0.0);
                let mut cur = 0usize;
                let mut loc0 = 0usize;
                for &bi in fam_blks {
                    let br = &lam_blocks[bi as usize];
                    let (q, lo_x) = (br.q, br.lam_off);
                    for a in 0..q {
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
                                let colbase = (c0 + jl) * e_f;
                                for il in 0..=a {
                                    panel[colbase + loc0 + il] += lam_small[lo_x + a * q + il] * h;
                                }
                            }
                        }
                    }
                    cur += q * w;
                    loc0 += q;
                }
                // Phase B on the e_f-row columns.
                for c in 0..w {
                    let (done, rest) = panel.split_at_mut(c * e_f);
                    let col_c = &mut rest[..e_f];
                    for k in 0..c {
                        let l_ck = fam_a[c * w + k];
                        let col_k = &done[k * e_f..(k + 1) * e_f];
                        for (x, &y) in col_c.iter_mut().zip(col_k) {
                            *x -= l_ck * y;
                        }
                    }
                    let inv_cc = 1.0 / fam_a[c * w + c];
                    for v in col_c.iter_mut() {
                        *v *= inv_cc;
                    }
                }
                // S22 −= L21_f·L21_fᵀ. The fused scalar syrk is now one
                // triangular `panel·panelᵀ` into `dd_temp` (a RESULT-MOVING
                // reassociation of the per-entry dots, sanctioned as the dense
                // arm's syrk at `1608–1610`; the small contraction width `w` —
                // 1–2 for random-intercept crossed factors — means the win is
                // on the `e_f` output dimension), then the exact existing
                // scatter walk reads the accumulated lower entry from the temp.
                // The `a` outer / `b ≤ a` inner order mirrors the slot
                // generation in `build_sparse_tail` — change together.
                let dd = &mut dd_temp[..e_f * e_f];
                let panel_ref = MatRef::from_column_major_slice(&panel[..e_f * w], e_f, w);
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
                let (_, vals) = axx.parts_mut();
                let slots = &fam_dd_slots[fam_dd_off[f]..fam_dd_off[f + 1]];
                let mut si = 0usize;
                for a in 0..e_f {
                    for b in 0..=a {
                        // dd is col-major e_f×e_f: entry (a,b) at a + b·e_f.
                        vals[slots[si] as usize] -= dd[a + b * e_f];
                        si += 1;
                    }
                }
                debug_assert_eq!(si, slots.len(), "family downdate slot replay exhausted");
                // B̃₂ −= L21_f·U1_f (this family's rows of the B2 downdate). The
                // per-block scalar gemm is now one `panel·U1_sub` into `b2_temp`
                // (Accum::Replace; RESULT-MOVING reassociation, sanctioned as
                // the dense arm's B2 matmul at `1676–1683`). U1_sub is rows
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
                for c in 0..m {
                    let col = &mut u2[c * e..(c + 1) * e];
                    for k in 0..e {
                        let s22k = s22.col(k).try_as_col_major().unwrap().as_slice();
                        let xk = col[k] / s22k[k];
                        col[k] = xk;
                        for (x, &s) in col[k + 1..].iter_mut().zip(&s22k[k + 1..]) {
                            *x -= s * xk;
                        }
                    }
                }
            }
            Some(tail) => {
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
                // No forward-only solve exists on the simplicial factor, and
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
    // the dense-tail S22 syrk above (`s22` None-arm, `1608–1610`) — and
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

// ---------------------------------------------------------------------------
// Sparse-Z non-Gaussian GLMM
// ---------------------------------------------------------------------------
//
// The over-envelope non-Gaussian close: a sparse PIRLS driver composing the
// sparse-Z half (per-row Z scatter / block-diagonal Λ, cap-free heap sizing —
// this module) with the family half (per-family IRLS weights/working residual
// and the joint θ+β Laplace deviance — `family.rs` / the dense `glmm` kernel).
// Per BOBYQA eval the packed M = ZΛ row values are refilled at θ and PIRLS
// iterates the conditional modes: each inner step re-weights the k×k system
// A = M'WM + I from the ≤(q_p + Σq_g) nonzeros per row and re-solves through a
// dense heap LLT, whose log-det at the converged mode feeds the Laplace
// deviance. The dense k×k factor (not the Gaussian blocked kernel) is
// deliberate: the Gaussian path's packed Λ'Z'ZΛ streams are θ-independent and
// packed once, but W changes every inner iteration, so the packing would be
// rebuilt per step anyway — the O(n·nnz²) weighted Gram accumulation dominates
// and stays sparse; only the k×k factor is dense, and k (RE columns) is
// moderate for every over-envelope shape this path serves. Perf retuning is
// explicitly out of scope (YAGNI).
//
// The outer optimizer is the single joint [θ | β] BOBYQA — the dense kernel's
// `two_stage = false` shape, which the A/B gate keeps converging to the same
// Laplace optimum. The θ-only PQL stage 1 is an accelerant only and is not
// replicated here.

/// Per-fit workspace for the sparse non-Gaussian GLMM path. Everything is
/// heap-sized off the design (cap-free); the packed M rows have a FIXED width
/// `q_p + Σ q_g` (every row loads exactly one level of every grouping), so the
/// per-row scatter is two flat arrays, no CSR offsets.
pub(crate) struct SparseGlmmWorkspace {
    pub(crate) g: LmmGroupings,
    /// `lam_small` offsets per extra DECLARATION (parallel to `g.extra_offsets`);
    /// the primary block is at 0. Maps `fill_lambda_small`'s
    /// `[primary | nested | crossed]` layout back to declaration order.
    lam_off_decl: Vec<usize>,
    /// Concatenated per-grouping `q×q` Λ factors (row-major lower-tri), refilled
    /// once per θ eval by `fill_lambda_small`.
    lam_small: Vec<f64>,
    /// Packed M = ZΛ nonzeros, fixed row width: row `i`'s entries at
    /// `[i·width, (i+1)·width)`. Columns (`m_cols`, design-fixed, filled once)
    /// follow `for_each_z_entry`'s layout — slope-major primary
    /// (component c at `c·n_primary + f`), level-major extras
    /// (`extra_offsets[e] + level·q_g + c`) — change together. Values
    /// (`m_vals`) are the Λ-folded z entries, refilled per θ eval.
    width: usize,
    m_cols: Vec<u32>,
    m_vals: Vec<f64>,
    // PIRLS state, length n / k — mirrors the dense `GlmmWorkspace` fields of
    // the same names (`pirls_solve` is the reference implementation).
    eta_fixed: Vec<f64>,
    eta: Vec<f64>,
    prob: Vec<f64>,
    w: Vec<f64>,
    mu: Vec<f64>,
    /// Per-row prior weights `wᵢ` (`FitOptions::weights`; all-1 when absent —
    /// zero behavioral change). Enter as `W̃ᵢ ← wᵢ·W̃ᵢ` on the working weight,
    /// `wᵢ·devᵢ` on the deviance, and `wᵢ·ρᵢ` on the score — ρ here is the
    /// PRODUCT W̃·r_working (not R's bare working residual, which prior weights
    /// leave untouched), so it carries the weight. Everything downstream
    /// (A/RHS scatter, β border, Rx Schur) reads `w`/ρ and inherits it.
    /// Every family is wired (Task 7): Gamma's profiled dispersion
    /// (`family::gamma_aic`, called with `Some(&ws.prior_w)`) and its
    /// `vcov(use.hessian=FALSE)` scale (`family::glmm_sigma_sq`) both take
    /// `Σwᵢ`/`wᵢ` in place of `n`/1; its post-fit Pearson φ̂ moment
    /// (`fit_glmm_sparse`'s `dispersion` arm) sums `wᵢrᵢ²` over the raw `n−p`
    /// df. NB's marginal-θ profile (`fit_glmm_nb_sparse`) passes
    /// `opts.weights` straight into `nb_profile_loglik`.
    prior_w: Vec<f64>,
    u: Vec<f64>,
    u_prev: Vec<f64>,
    /// `A = M'WM + I` (k×k, full symmetric — the per-row scatter writes both
    /// triangles); left holding the FINAL iterate's A after a converged PIRLS,
    /// which the Rx Schur fill re-factors (the `dense_schur_fill` contract).
    /// `pirls` must therefore never factor THIS field in place — see `a_chol`.
    a: Mat<f64>,
    /// Copy-then-factor target for `a`'s Cholesky (k×k): `pirls` copies `a`'s
    /// lower triangle in here (mirroring `.llt(Side::Lower)`'s internal
    /// `copy_from_triangular_lower`) and factors THIS buffer in place, leaving
    /// `a` itself untouched for `sparse_glmm_schur` to re-read.
    a_chol: Mat<f64>,
    /// Scratch for `a_chol`'s in-place `cholesky_in_place` (k×k) — avoids the
    /// per-PIRLS-iteration `.llt(Side::Lower)` allocation on the `pirls` hot loop.
    a_llt_mem: MemBuffer,
    a_rhs: Vec<f64>,
    /// PIRLS β state, length p: `beta` is the current β (input for a Fixed
    /// solve, in/out for a Profile solve — the sparse twin of the dense
    /// `BetaStep` split); `beta_prev` its step-halving backtrack twin;
    /// `beta_rhs` the Profile δβ RHS/solution scratch. The Profile border
    /// matrices (`xtwx`/`xtwm`/`ainv_mtwx`/`schur`) mirror `BetaStep::Profile`'s.
    beta: Vec<f64>,
    beta_prev: Vec<f64>,
    beta_rhs: Vec<f64>,
    xtwx: Mat<f64>,
    /// `WX = diag(w)·X` (n×p) scratch for the Profile-mode `xtwx = Xᵀ(WX)`
    /// weighted gemm — refilled each PIRLS iteration (W changes) before the
    /// matmul.
    wx: Mat<f64>,
    xtwm: Mat<f64>,
    ainv_mtwx: Mat<f64>,
    schur: Mat<f64>,
    /// Scratch for `schur`'s in-place `cholesky_in_place` (p×p) — avoids the
    /// per-PIRLS-iteration `.llt(Side::Lower)` allocation on the Profile-mode
    /// β-Schur border step.
    schur_llt_mem: MemBuffer,
    k: usize,
    p: usize,
    /// PIRLS exit-tol override read by `pirls` — the sparse twin of the dense
    /// `GlmmWorkspace::pirls_tol_override`. `Some(PIRLS_TOL_REL_FD)` only around
    /// the `WaldSe::Hessian` FD evals (and the RX-fallback central re-eval);
    /// `None` on the fit path, which therefore stays bit-identical.
    pirls_tol_override: Option<f64>,
}

impl SparseGlmmWorkspace {
    pub(crate) fn new(
        g: &LmmGroupings,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        n: usize,
        p: usize,
    ) -> Self {
        let q_p = g.primary_q;
        // lam_small layout mirrors `fill_lambda_small` — primary, nested, crossed.
        let mut lam_len = q_p * q_p;
        let mut lam_off_decl = vec![0usize; g.extra_offsets.len()];
        if let Some(nf) = g.nested {
            lam_off_decl[nf.decl] = lam_len;
            lam_len += nf.q * nf.q;
        }
        for cf in &g.crossed {
            lam_off_decl[cf.decl] = lam_len;
            lam_len += cf.q * cf.q;
        }
        let width = q_p + g.extra_q.iter().sum::<usize>();
        // m_cols is design-fixed: fill once from the ids (values are θ-dependent
        // and filled per eval by `fill_m_vals`).
        let mut m_cols = vec![0u32; n * width];
        for i in 0..n {
            let mut t = i * width;
            let f = cluster_ids[i] as usize;
            for c in 0..q_p {
                m_cols[t] = (c * g.n_primary + f) as u32;
                t += 1;
            }
            for (e, ids_e) in extra_ids.iter().enumerate() {
                let q_g = g.extra_q[e];
                let base = g.extra_offsets[e] + ids_e[i] as usize * q_g;
                for c in 0..q_g {
                    m_cols[t] = (base + c) as u32;
                    t += 1;
                }
            }
        }
        let k = g.k_total;
        SparseGlmmWorkspace {
            g: g.clone(),
            lam_off_decl,
            lam_small: vec![0.0; lam_len.max(1)],
            width,
            m_cols,
            m_vals: vec![0.0; n * width],
            eta_fixed: vec![0.0; n.max(1)],
            eta: vec![0.0; n.max(1)],
            prob: vec![0.0; n.max(1)],
            w: vec![0.0; n.max(1)],
            mu: vec![0.0; n.max(1)],
            prior_w: vec![1.0; n.max(1)],
            u: vec![0.0; k.max(1)],
            u_prev: vec![0.0; k.max(1)],
            a: Mat::zeros(k.max(1), k.max(1)),
            a_chol: Mat::zeros(k.max(1), k.max(1)),
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                k.max(1),
                Par::Seq,
                Spec::default(),
            )),
            a_rhs: vec![0.0; k.max(1)],
            beta: vec![0.0; p.max(1)],
            beta_prev: vec![0.0; p.max(1)],
            beta_rhs: vec![0.0; p.max(1)],
            xtwx: Mat::zeros(p.max(1), p.max(1)),
            wx: Mat::zeros(n.max(1), p.max(1)),
            xtwm: Mat::zeros(p.max(1), k.max(1)),
            ainv_mtwx: Mat::zeros(k.max(1), p.max(1)),
            schur: Mat::zeros(p.max(1), p.max(1)),
            schur_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                p.max(1),
                Par::Seq,
                Spec::default(),
            )),
            k,
            p,
            pirls_tol_override: None,
        }
    }

    /// Fresh per-thread clone for one FD-Hessian worker: independently-sized, shares
    /// no mutable state with `self`. The design-fixed fields (`g`, `lam_off_decl`,
    /// `width`, `m_cols`, `prior_w`) and the tol override are carried over; the
    /// scratch (`lam_small`/`m_vals`/PIRLS buffers) is cloned only for its SIZE —
    /// every eval refills Λ and M and cold-seeds û = 0 (`sparse_glmm_deviance` with
    /// `pirls_tol_override == Some`), so each grid cell is a pure function of
    /// `(gamma_hat, steps, design)` and reproduces the serial value bit-for-bit. The
    /// two `MemBuffer`s can't be cloned (not `Clone`); they are re-sized from `k`/`p`
    /// exactly as `new` does.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    fn clone_worker(&self) -> SparseGlmmWorkspace {
        SparseGlmmWorkspace {
            g: self.g.clone(),
            lam_off_decl: self.lam_off_decl.clone(),
            lam_small: self.lam_small.clone(),
            width: self.width,
            m_cols: self.m_cols.clone(),
            m_vals: self.m_vals.clone(),
            eta_fixed: self.eta_fixed.clone(),
            eta: self.eta.clone(),
            prob: self.prob.clone(),
            w: self.w.clone(),
            mu: self.mu.clone(),
            prior_w: self.prior_w.clone(),
            u: self.u.clone(),
            u_prev: self.u_prev.clone(),
            a: self.a.clone(),
            a_chol: self.a_chol.clone(),
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                self.k.max(1),
                Par::Seq,
                Spec::default(),
            )),
            a_rhs: self.a_rhs.clone(),
            beta: self.beta.clone(),
            beta_prev: self.beta_prev.clone(),
            beta_rhs: self.beta_rhs.clone(),
            xtwx: self.xtwx.clone(),
            wx: self.wx.clone(),
            xtwm: self.xtwm.clone(),
            ainv_mtwx: self.ainv_mtwx.clone(),
            schur: self.schur.clone(),
            schur_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                self.p.max(1),
                Par::Seq,
                Spec::default(),
            )),
            k: self.k,
            p: self.p,
            pirls_tol_override: self.pirls_tol_override,
        }
    }

    /// Refill the packed M values at the current Λ (`lam_small` must be filled
    /// for this θ): entry c of a row's block is the lower-tri fold
    /// `Σ_{r≥c} z[r]·Λ[r,c]` with `z = [1, x[slope cols…]]` — the same sandwich
    /// `apply_lambda` writes densely on the in-envelope GLMM path.
    fn fill_m_vals(&mut self, x: MatRef<f64>, n: usize) {
        let g = &self.g;
        let q_p = g.primary_q;
        for i in 0..n {
            let mut t = i * self.width;
            for c in 0..q_p {
                let mut acc = 0.0;
                for r in c..q_p {
                    let z = if r == 0 {
                        1.0
                    } else {
                        x[(i, g.primary_slope_cols[r - 1])]
                    };
                    acc += z * self.lam_small[r * q_p + c];
                }
                self.m_vals[t] = acc;
                t += 1;
            }
            for e in 0..g.extra_offsets.len() {
                let q_g = g.extra_q[e];
                let lo = self.lam_off_decl[e];
                for c in 0..q_g {
                    let mut acc = 0.0;
                    for r in c..q_g {
                        let z = if r == 0 {
                            1.0
                        } else {
                            x[(i, g.extra_slope_cols[e][r - 1])]
                        };
                        acc += z * self.lam_small[lo + r * q_g + c];
                    }
                    self.m_vals[t] = acc;
                    t += 1;
                }
            }
        }
    }

    /// Refill `eta_fixed[i] = Σ_j x[i,j]·β[j]` from `self.beta` — the sparse
    /// twin of `pirls::refresh_eta_fixed`. Called at PIRLS entry and, in
    /// Profile mode, after every β update (δβ step and each β halving).
    fn refresh_eta_fixed(&mut self, x: MatRef<f64>, n: usize) {
        for i in 0..n {
            let mut e = 0.0;
            for (j, &b) in self.beta[..self.p].iter().enumerate() {
                e += x[(i, j)] * b;
            }
            self.eta_fixed[i] = e;
        }
    }

    /// Penalized-IRLS inner solve on the packed sparse M rows — the sparse twin
    /// of `glmm::pirls_solve`, with the SAME two β modes: `profile = false`
    /// holds `self.beta` fixed (the FD-Hessian / joint stage-2 contract);
    /// `profile = true` adds the joint δβ Schur-border step each iteration
    /// (PQL — β̂(θ) written back through `self.beta`), backtracked in lockstep
    /// with u. Same discipline verbatim: trial evaluation at the current u,
    /// band-tolerant retrospective step-halving (lme4 `pwrssUpdate`), the mixed
    /// `dev(uⱼ) + ‖uⱼ₊₁‖²` convergence rule, `log|A|` off the factor that
    /// produced the returned u. Every family takes the general Fisher-scoring
    /// branch through `family.rs` (no fused-SIMD logit shortcut here — for
    /// canonical links the general weight/residual reduce to the same
    /// quantities, and this path has no byte-identity gate to a prior
    /// implementation). Returns `(dev, ‖ũ‖², log|A|, converged)`; a non-PD
    /// A/S_β or exhausted halvings surface as `(NaN, NaN, NaN, false)`.
    /// Iterates from whatever `self.u` holds on entry — `pirls` itself never
    /// decides reset vs. warm-start; that call is `sparse_glmm_deviance`'s
    /// (its caller), which cold-seeds `self.u = 0` for FD-Hessian/tight-tol
    /// evals and otherwise leaves the previous eval's converged `u` in place
    /// as a warm start.
    fn pirls(
        &mut self,
        family: crate::Family,
        nb_theta: f64,
        x: MatRef<f64>,
        y: &[f64],
        n: usize,
        profile: bool,
    ) -> (f64, f64, f64, bool) {
        let (k, p, width) = (self.k, self.p, self.width);
        self.refresh_eta_fixed(x, n);
        let tol = self
            .pirls_tol_override
            .unwrap_or_else(|| crate::glmm::pirls_tol(family));
        let mut pen_accepted = f64::INFINITY;
        let mut mixed_prev = f64::INFINITY;
        let mut halvings = 0usize;
        let mut converged = false;
        let mut dev = f64::NAN;
        let mut pen = f64::NAN;
        let mut logdet = 0.0;
        for _ in 0..crate::glmm::PIRLS_MAX_ITERS {
            // Trial evaluation at the current u: (Mu)ᵢ, then η/μ/W/deviance.
            dev = 0.0;
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let base = i * width;
                let mut acc = 0.0;
                for t in base..base + width {
                    acc += self.m_vals[t] * self.u[self.m_cols[t] as usize];
                }
                self.mu[i] = acc;
                let e = crate::family::clamp_eta(family, self.eta_fixed[i] + acc);
                self.eta[i] = e;
                // Canonical-link shortcut (Poisson-log) lives inside this call — see
                // `irls_weight_and_resid`'s doc comment.
                let (mui, wi, _) = crate::family::irls_weight_and_resid(family, nb_theta, y[i], e);
                self.prob[i] = mui;
                self.w[i] = (self.prior_w[i] * wi).max(crate::glm::WEIGHT_CLAMP);
                dev += self.prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], mui);
            }
            // Band-tolerant retrospective step-halving (mirror `pirls_solve` —
            // see its in-loop comment for why the band must not converge). In
            // Profile mode the trial point is the JOINT (u, β) step, so β halves
            // toward `beta_prev` in lockstep with u.
            let pen_u: f64 = self.u[..k].iter().map(|v| v * v).sum();
            let penalized = dev + pen_u;
            if penalized - pen_accepted > tol * (1.0 + penalized.abs()) {
                if halvings < crate::glmm::PIRLS_MAX_HALVINGS {
                    halvings += 1;
                    for c in 0..k {
                        self.u[c] = 0.5 * (self.u[c] + self.u_prev[c]);
                    }
                    if profile {
                        for j in 0..p {
                            self.beta[j] = 0.5 * (self.beta[j] + self.beta_prev[j]);
                        }
                        self.refresh_eta_fixed(x, n);
                    }
                    continue;
                }
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            halvings = 0;
            pen_accepted = penalized;
            self.u_prev[..k].copy_from_slice(&self.u[..k]);
            if profile {
                self.beta_prev[..p].copy_from_slice(&self.beta[..p]);
            }
            // A = M'WM + I and rhs = M'(W·Mu + W·r), accumulated from each row's
            // ≤width nonzeros (full-symmetric A — both triangles written, the
            // `SparseLmmWorkspace` Z'Z convention). Profile additionally
            // accumulates the β-gradient X'ρ (ρ = the effective residual) into
            // `beta_rhs` — the joint system's bottom-block RHS.
            for c in 0..k {
                for r in 0..k {
                    self.a[(r, c)] = 0.0;
                }
                self.a_rhs[c] = 0.0;
            }
            if profile {
                for v in self.beta_rhs[..p].iter_mut() {
                    *v = 0.0;
                }
            }
            for i in 0..n {
                let wi = self.w[i];
                let dmu = crate::family::mu_eta(family, self.eta[i]);
                let v = crate::family::variance(family, nb_theta, self.prob[i]);
                let rho = self.prior_w[i] * dmu * (y[i] - self.prob[i]) / v;
                let q_i = wi * self.mu[i] + rho;
                let base = i * width;
                for ta in base..base + width {
                    let ca = self.m_cols[ta] as usize;
                    let va = self.m_vals[ta];
                    let wva = wi * va;
                    for tb in base..base + width {
                        let cb = self.m_cols[tb] as usize;
                        let vb = self.m_vals[tb];
                        self.a[(ca, cb)] += wva * vb;
                    }
                    self.a_rhs[ca] += va * q_i;
                }
                if profile {
                    for j in 0..p {
                        self.beta_rhs[j] += x[(i, j)] * rho;
                    }
                }
            }
            for r in 0..k {
                self.a[(r, r)] += 1.0;
            }
            // Copy A's lower triangle into the persistent `a_chol` scratch (mirrors
            // `.llt(Side::Lower)`'s own `copy_from_triangular_lower`), then factor
            // THAT in place — `self.a` must come out of this call unmutated (see
            // its field doc; `sparse_glmm_schur` re-reads it post-fit).
            self.a_chol.copy_from_triangular_lower(self.a.as_ref());
            if cholesky_in_place(
                self.a_chol.as_mut(),
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut self.a_llt_mem),
                Spec::default(),
            )
            .is_err()
            {
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            logdet = 0.0;
            for r in 0..k {
                logdet += self.a_chol[(r, r)].ln();
            }
            solve_in_place(
                self.a_chol.as_ref(),
                faer::MatMut::from_column_major_slice_mut(&mut self.a_rhs[..k], k, 1),
                Par::Seq,
                MemStack::new(&mut self.a_llt_mem),
            );
            pen = 0.0;
            for c in 0..k {
                self.u[c] = self.a_rhs[c];
                pen += self.u[c] * self.u[c];
            }
            // Profile-mode joint δβ step (β-Schur border), taken while `ac` is
            // alive — mirrors `pirls_solve`'s Profile block: T = A⁻¹B,
            // S_β = C − B'T, δβ = S_β⁻¹(X'ρ − B'·δu₀), then β += δβ and
            // u ← u_new − T·δβ.
            if profile {
                // B' = X'WM (p×k) via the packed rows; C = X'WX (p×p).
                for r in 0..p {
                    for c in 0..k {
                        self.xtwm[(r, c)] = 0.0;
                    }
                }
                for i in 0..n {
                    let wi = self.w[i];
                    let base = i * width;
                    for r in 0..p {
                        let xw = x[(i, r)] * wi;
                        for t in base..base + width {
                            self.xtwm[(r, self.m_cols[t] as usize)] += xw * self.m_vals[t];
                        }
                    }
                }
                // C = X'WX = Xᵀ diag(w) X via one weighted gemm, replacing the
                // O(p²·n) per-pair loop. Recomputed each PIRLS iteration because
                // W changes with the working weights — same per-iteration
                // invariant as the X'WM assembly just above. WX = diag(w)·X is
                // formed into `wx`, then xtwx = Xᵀ·WX (full p×p, kept
                // full-symmetric as the downstream border reads it).
                for r in 0..p {
                    for i in 0..n {
                        self.wx[(i, r)] = self.w[i] * x[(i, r)];
                    }
                }
                faer::linalg::matmul::matmul(
                    self.xtwx.as_mut(),
                    faer::Accum::Replace,
                    x.transpose(),
                    self.wx.as_ref(),
                    1.0,
                    Par::Seq,
                );
                for r in 0..k {
                    for c in 0..p {
                        self.ainv_mtwx[(r, c)] = self.xtwm[(c, r)];
                    }
                }
                solve_in_place(
                    self.a_chol.as_ref(),
                    self.ainv_mtwx.as_mut(),
                    Par::Seq,
                    MemStack::new(&mut self.a_llt_mem),
                );
                for r in 0..p {
                    for c in 0..p {
                        let mut s = self.xtwx[(r, c)];
                        for j in 0..k {
                            s -= self.xtwm[(r, j)] * self.ainv_mtwx[(j, c)];
                        }
                        self.schur[(r, c)] = s;
                    }
                }
                // rhs = X'ρ − B'·δu₀ (δu₀ = u − u_prev).
                for r in 0..p {
                    let mut acc = 0.0;
                    for c in 0..k {
                        acc += self.xtwm[(r, c)] * (self.u[c] - self.u_prev[c]);
                    }
                    self.beta_rhs[r] -= acc;
                }
                if cholesky_in_place(
                    self.schur.as_mut(),
                    LltRegularization::default(),
                    Par::Seq,
                    MemStack::new(&mut self.schur_llt_mem),
                    Spec::default(),
                )
                .is_err()
                {
                    return (f64::NAN, f64::NAN, f64::NAN, false);
                }
                solve_in_place(
                    self.schur.as_ref(),
                    faer::MatMut::from_column_major_slice_mut(&mut self.beta_rhs[..p], p, 1),
                    Par::Seq,
                    MemStack::new(&mut self.schur_llt_mem),
                );
                for j in 0..p {
                    self.beta[j] += self.beta_rhs[j];
                }
                for c in 0..k {
                    let mut acc = 0.0;
                    for j in 0..p {
                        acc += self.ainv_mtwx[(c, j)] * self.beta_rhs[j];
                    }
                    self.u[c] -= acc;
                }
                self.refresh_eta_fixed(x, n);
                pen = 0.0;
                for c in 0..k {
                    pen += self.u[c] * self.u[c];
                }
            }
            let mixed = dev + pen;
            if (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
                converged = true;
                break;
            }
            mixed_prev = mixed;
        }
        (dev, pen, logdet, converged)
    }
}

/// Joint Laplace deviance at `params = [θ | β]` on the sparse path — the sparse
/// twin of `glmm::laplace_deviance`: refill Λ and the packed M values at θ,
/// seed û (fit-path evals, `pirls_tol_override == None`, warm-start from
/// whatever `ws.u` holds on entry — the previous call's converged mode, fewer
/// PIRLS iterations to reconverge; FD-Hessian/tight-tol evals cold-seed û = 0,
/// order-free as those evals require a seed independent of evaluation order),
/// run the sparse PIRLS, and return
/// `data + ‖ũ‖² + log|A|²` with Gamma's `aic` substitution
/// (`family::gamma_aic`) exactly as the dense objective does. β mode mirrors
/// the dense call sites: `profile_beta = false` copies `params[n_theta..]`
/// into `ws.beta` and holds it fixed (the stage-2 / FD-Hessian contract);
/// `profile_beta = true` reads only the θ prefix (`params` may be a θ-only
/// slice) and lets the PQL δβ step drive `ws.beta` from the CALLER's
/// pre-seeded value (the stage-1 objective — seed `ws.beta` to a fixed β₀
/// before each eval so the objective stays a function of θ alone).
/// Non-convergence / Cholesky failure ⇒ `f64::INFINITY`.
#[allow(clippy::too_many_arguments)]
fn sparse_glmm_deviance(
    family: crate::Family,
    nb_theta: f64,
    params: &[f64],
    ws: &mut SparseGlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    n: usize,
    profile_beta: bool,
) -> f64 {
    let n_theta = ws.g.n_theta();
    let p = ws.p;
    fill_lambda_small(&params[..n_theta], &ws.g, &mut ws.lam_small);
    ws.fill_m_vals(x, n);
    if !profile_beta {
        ws.beta[..p].copy_from_slice(&params[n_theta..n_theta + p]);
    }
    // Fit-path evals (pirls_tol_override == None) carry the previous call's
    // converged û forward as PIRLS's starting point — fewer iterations to
    // reconverge, same fixed point (seed-independence: PIRLS converges to the
    // same conditional mode from any start, only iteration count differs; see
    // the dense analogue `warm_start_objective_is_seed_independent`,
    // src/glmm/tests.rs). FD-Hessian/tight-tol evals (Some(...)) still cold-seed
    // û = 0, preserving the order-free property `sparse_fd_hessian_cov` relies on.
    if ws.pirls_tol_override.is_some() {
        for v in ws.u.iter_mut() {
            *v = 0.0;
        }
    }
    let (dev, pen, logdet, conv) = ws.pirls(family, nb_theta, x, y, n, profile_beta);
    if !conv || !dev.is_finite() {
        return f64::INFINITY;
    }
    let data_term = if matches!(family, crate::Family::Gamma { .. }) {
        crate::family::gamma_aic(y, &ws.prob[..n], dev, n, Some(&ws.prior_w[..n]))
    } else {
        dev
    };
    data_term + pen + 2.0 * logdet
}

/// Rx (closed-form Schur) fixed-effect information at the converged state —
/// the sparse twin of `glmm::se::dense_schur_fill`: `S_β = X'W̃X − X'W̃M·A⁻¹M'W̃X`
/// from the final PIRLS iterate's W̃ (`ws.w`), packed M rows, and A (`ws.a`).
/// Returns `None` on a non-PD A. Local allocations are fine — this is a
/// once-per-fit cold path, not the optimizer loop.
fn sparse_glmm_schur(ws: &mut SparseGlmmWorkspace, x: MatRef<f64>, n: usize) -> Option<Mat<f64>> {
    use faer::linalg::solvers::Solve;
    let (k, p, width) = (ws.k, ws.p, ws.width);
    let mut xtwx = Mat::<f64>::zeros(p, p);
    for r in 0..p {
        for c in 0..=r {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, r)] * ws.w[i] * x[(i, c)];
            }
            xtwx[(r, c)] = s;
            xtwx[(c, r)] = s;
        }
    }
    // X'W̃M (p×k) by per-row scatter over the packed nonzeros.
    let mut xtwm = Mat::<f64>::zeros(p, k);
    for i in 0..n {
        let wi = ws.w[i];
        let base = i * width;
        for r in 0..p {
            let xw = x[(i, r)] * wi;
            for t in base..base + width {
                xtwm[(r, ws.m_cols[t] as usize)] += xw * ws.m_vals[t];
            }
        }
    }
    let ac = match ws.a.as_ref().llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return None,
    };
    let mut ainv_mtwx = Mat::<f64>::zeros(k, p);
    for r in 0..k {
        for c in 0..p {
            ainv_mtwx[(r, c)] = xtwm[(c, r)];
        }
    }
    ac.solve_in_place(ainv_mtwx.as_mut());
    let mut schur = Mat::<f64>::zeros(p, p);
    for r in 0..p {
        for c in 0..p {
            let mut s = xtwx[(r, c)];
            for j in 0..k {
                s -= xtwm[(r, j)] * ainv_mtwx[(j, c)];
            }
            schur[(r, c)] = s;
        }
    }
    Some(schur)
}

/// Relative FD step for the SPARSE joint-deviance Hessian — deliberately NOT
/// the dense `glmm::FD_STEP_REL` (1e-2): the two paths sit on opposite sides
/// of the truncation-vs-noise trade. On the over-width sparse Gamma golden
/// (`sim_sparse_gamma`, 21-dim joint), h = 1e-2 leaves ~3.3e-2 truncation
/// error on se(β₀) (0.1956 vs 0.1892 from a Richardson Hessian at the SAME
/// point); h = 1e-3 lands within ~1e-3 of Richardson and is step-invariant
/// down to 5e-4 (identical SEs — the single-step-FD method floor, tol.R's
/// `stddev_se_rel` note). The DENSE path is the mirror image: at h = 1e-3 its
/// FD noise blows the curated se_hess gates (sim_gamma 1e-2, cbpp_probit 2e-3
/// vs the 1e-3 band) while h = 1e-2 holds them at ~1e-4 — so the dense
/// constant stays 1e-2 and this one must not be folded back into it.
const SPARSE_FD_STEP_REL: f64 = 1e-3;

/// FD-Hessian joint (θ,β) covariance on the sparse path — mirrors
/// `glmm::fd_hessian_cov`'s scheme exactly (single-step central differences, no
/// Richardson extrapolation, step `h_k = SPARSE_FD_STEP_REL·max(1, |γ̂_k|)`
/// (sparse-calibrated, see the constant above),
/// `cov = 2·(H_dev⁻¹)_ββ`, θ SE from the θ diagonal) minus the warm-seed
/// machinery: every eval here cold-seeds û = 0 inside `sparse_glmm_deviance`,
/// which is a constant seed and therefore order-free by the same argument.
/// Returns `None` on a non-finite perturbed deviance or non-PD joint Hessian —
/// the caller falls back to the Rx Schur (the `NonPdFellBackToRx` shape).
/// Tolerance contract: the CALLER sets `ws.pirls_tol_override =
/// Some(PIRLS_TOL_REL_FD)` around this call (and its fallback re-eval) and
/// resets it after — set/reset can't live here because the `?` early returns
/// would skip the reset. Same rationale as the dense `fd_hessian_cov`: at the
/// canonical fit tol the FD is not step-invariant; at the tight tol it is, by
/// construction.
#[allow(clippy::too_many_arguments)]
fn sparse_fd_hessian_cov(
    family: crate::Family,
    nb_theta: f64,
    gamma_hat: &[f64],
    ws: &mut SparseGlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    n: usize,
    parallel_inner: bool,
) -> Option<(Mat<f64>, Vec<f64>)> {
    use faer::linalg::solvers::Solve;
    let m = gamma_hat.len();
    let n_theta = ws.g.n_theta();
    let p = ws.p;
    let f0 = sparse_glmm_deviance(family, nb_theta, gamma_hat, ws, x, y, n, false);
    if !f0.is_finite() {
        return None;
    }
    let steps: Vec<f64> = gamma_hat
        .iter()
        .map(|&g| SPARSE_FD_STEP_REL * g.abs().max(1.0))
        .collect();
    // Each grid cell cold-seeds û = 0 (constant seed), so every eval is a pure
    // function of (gamma_hat, steps, design) — no frozen-seed discipline is needed
    // at all, and per-thread workspaces reproduce the serial values bitwise.
    let mut hess = Mat::<f64>::zeros(m, m);
    let use_par = cfg!(all(feature = "parallel", not(target_arch = "wasm32"))) && parallel_inner;
    if use_par {
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            let cells: Vec<(usize, usize)> =
                (0..m).flat_map(|i| (i..m).map(move |j| (i, j))).collect();
            let ws_ro: &SparseGlmmWorkspace = ws;
            let steps = &steps;
            let results: Vec<(usize, usize, Option<f64>)> = cells
                .par_iter()
                .map_init(
                    || (ws_ro.clone_worker(), gamma_hat.to_vec()),
                    |(wws, pt), &(i, j)| {
                        let mut ev = |coords: &[usize], deltas: &[f64]| -> f64 {
                            pt.copy_from_slice(gamma_hat);
                            for (&c, &d) in coords.iter().zip(deltas) {
                                pt[c] += d;
                            }
                            sparse_glmm_deviance(family, nb_theta, pt, wws, x, y, n, false)
                        };
                        let h = if i == j {
                            let s = steps[i];
                            let fp = ev(&[i], &[s]);
                            let fm = ev(&[i], &[-s]);
                            (fp.is_finite() && fm.is_finite())
                                .then(|| (fp - 2.0 * f0 + fm) / (s * s))
                        } else {
                            let (si, sj) = (steps[i], steps[j]);
                            let fpp = ev(&[i, j], &[si, sj]);
                            let fpm = ev(&[i, j], &[si, -sj]);
                            let fmp = ev(&[i, j], &[-si, sj]);
                            let fmm = ev(&[i, j], &[-si, -sj]);
                            (fpp.is_finite()
                                && fpm.is_finite()
                                && fmp.is_finite()
                                && fmm.is_finite())
                            .then(|| (fpp - fpm - fmp + fmm) / (4.0 * si * sj))
                        };
                        (i, j, h)
                    },
                )
                .collect();
            // Serial arm returns None on the FIRST non-finite eval; here the whole
            // grid ran, then we check — same destination (Rx fallback), extra work
            // only on the already-failing path.
            if results.iter().any(|(_, _, h)| h.is_none()) {
                return None;
            }
            for (i, j, h) in results {
                let h = h.expect("checked all-Some above");
                hess[(i, j)] = h;
                hess[(j, i)] = h;
            }
        }
    } else {
        let mut pt = gamma_hat.to_vec();
        let mut eval = |pt: &mut Vec<f64>, coords: &[usize], deltas: &[f64]| -> f64 {
            pt.copy_from_slice(gamma_hat);
            for (&c, &d) in coords.iter().zip(deltas) {
                pt[c] += d;
            }
            sparse_glmm_deviance(family, nb_theta, pt, ws, x, y, n, false)
        };
        for i in 0..m {
            let hi = steps[i];
            // Diagonal: single-step central second difference (no Richardson — see
            // the doc comment above `fd_hessian_cov`).
            let mut second = |s: f64| -> Option<f64> {
                let fp = eval(&mut pt, &[i], &[s]);
                let fm = eval(&mut pt, &[i], &[-s]);
                if !(fp.is_finite() && fm.is_finite()) {
                    return None;
                }
                Some((fp - 2.0 * f0 + fm) / (s * s))
            };
            hess[(i, i)] = second(hi)?;
            for j in (i + 1)..m {
                let hj = steps[j];
                let mut mixed = |si: f64, sj: f64| -> Option<f64> {
                    let fpp = eval(&mut pt, &[i, j], &[si, sj]);
                    let fpm = eval(&mut pt, &[i, j], &[si, -sj]);
                    let fmp = eval(&mut pt, &[i, j], &[-si, sj]);
                    let fmm = eval(&mut pt, &[i, j], &[-si, -sj]);
                    if !(fpp.is_finite() && fpm.is_finite() && fmp.is_finite() && fmm.is_finite()) {
                        return None;
                    }
                    Some((fpp - fpm - fmp + fmm) / (4.0 * si * sj))
                };
                let hij = mixed(hi, hj)?;
                hess[(i, j)] = hij;
                hess[(j, i)] = hij;
            }
        }
    }
    let chol = hess.as_ref().llt(faer::Side::Lower).ok()?;
    let mut inv = Mat::<f64>::identity(m, m);
    chol.solve_in_place(inv.as_mut());
    let mut cov = Mat::<f64>::zeros(p, p);
    for a in 0..p {
        for b in 0..p {
            cov[(a, b)] = 2.0 * inv[(n_theta + a, n_theta + b)];
        }
    }
    let theta_se: Vec<f64> = (0..n_theta)
        .map(|kk| (2.0 * inv[(kk, kk)]).max(0.0).sqrt())
        .collect();
    Some((cov, theta_se))
}

/// The non-converged NaN `Fit` for the sparse GLMM path (mirrors the dense
/// adapters' NaN-fill shape; `dispersion` NaN, `tau2` NaN per θ coordinate).
fn sparse_glmm_nan_fit(p: usize, n_theta: usize) -> crate::Fit {
    crate::Fit {
        beta: vec![f64::NAN; p],
        se: vec![f64::NAN; p],
        tau2: vec![f64::NAN; n_theta],
        dispersion: f64::NAN,
        converged: false,
        varcorr: vec![],
        stddev_se: vec![],
        aliased: vec![false; p],
        n_eval: 0,
        deviance: f64::NAN,
        singular: false,
    }
}

/// Sparse-Z non-Gaussian GLMM end-to-end fit: the
/// over-envelope sibling of the dense `fit::fit_glmm` adapter, serving
/// Binomial / Poisson / Gamma (and, via `fit_glmm_nb_sparse`, NB) designs that
/// exceed the NoZ envelope. Single joint [θ | β] BOBYQA over the sparse Laplace
/// deviance (the dense kernel's `two_stage = false` shape), θ/β seeding and
/// ρ schedule mirroring `GlmmWorkspace::for_cluster_spec` + `glmm::fit_glmm`
/// (blind THETA0 θ₀ or a warm start floored at `THETA_TRUTH_FLOOR`; β from the
/// no-RE GLM warm start or the caller's `start`), diagonal-θ pin at
/// `PIN_THETA`, and a pinned-γ̂ re-eval whose finite deviance is the
/// convergence witness (the same degenerate-fit guard as the dense kernel).
///
/// SE: both `WaldSe` arms, exactly as the dense path emits them — `Hessian`
/// (default) via the joint FD-Hessian (`sparse_fd_hessian_cov`), falling back
/// to the Rx Schur on a non-PD Hessian; `Rx` via the closed-form Schur
/// conditional on θ̂ (`sparse_glmm_schur`). `tau2`/`dispersion`/`varcorr`
/// mirror `fit::fit_glmm`'s mapping (Gamma's pwrss/n τ² scale and Pearson
/// dispersion included). Returns the mapped `Fit` plus the minimized marginal
/// Laplace deviance (the NB marginal-θ objective kernel); non-NB callers take
/// `.0`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glmm_sparse(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &crate::ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    nb_theta: f64,
    start: Option<&crate::StartValues>,
    opts: &crate::FitOptions,
) -> (crate::Fit, f64) {
    let re = model
        .re
        .as_ref()
        .expect("fit_glmm_sparse requires a mixed model (re: Some)");
    let family = model.family;
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();
    let g = LmmGroupings::from_cluster_spec_ext(model, n, &slope_cols, &extra_slope_cols);
    let n_theta = g.n_theta();
    if n == 0 || p == 0 {
        return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
    }
    let xm = MatRef::from_row_major_slice(x, n, p);
    let mut ws = SparseGlmmWorkspace::new(&g, cluster_ids, extra_ids, n, p);
    if let Some(w) = &opts.weights {
        ws.prior_w[..n].copy_from_slice(w);
    }

    // Joint [θ | β] parameter vector, seeds, and boxes. The θ cold start is the
    // structure-aware blind seed from `blind_theta_and_bounds` — diagonal vech
    // entries at THETA0, OFF-DIAGONAL entries at 0 — the same shape the LMM
    // cold starts adopted in the 2026-07-11 basin fix: with a wide vech block (the
    // over-width q_g=5 shape) all-ones off-diagonals give a badly mis-scaled Λ
    // (D diagonals up to q·THETA0²) and the joint BOBYQA stalls in that basin —
    // measured on sim_sparse_gamma, where the all-THETA0 start converged ~240
    // deviance units above the lme4 optimum with θ̂ ≈ θ₀. A warm start is floored
    // at THETA_TRUTH_FLOOR on every coordinate (mirror `glmm::fit_glmm`). The β
    // portion mirrors `fit::fit_glmm`: caller start verbatim, else the no-RE GLM
    // warm start; clamped into the ±BETA_BOX box.
    let (theta0, mut lower, mut upper) = g.blind_theta_and_bounds();
    let mut params = vec![0.0f64; n_theta + p];
    match start {
        Some(s) => {
            for (t, &v) in params[..n_theta].iter_mut().zip(&s.theta) {
                *t = v.max(crate::lmm::THETA_TRUTH_FLOOR);
            }
        }
        None => params[..n_theta].copy_from_slice(&theta0),
    }
    let beta_start = match start {
        Some(s) => s.beta.clone(),
        None => crate::fit::glm_warm_start_beta(family, nb_theta, xm, y, n, p),
    };
    for (slot, &b) in params[n_theta..].iter_mut().zip(&beta_start) {
        *slot = b.clamp(-crate::glmm::BETA_BOX, crate::glmm::BETA_BOX);
    }
    lower.extend(std::iter::repeat_n(-crate::glmm::BETA_BOX, p));
    upper.extend(std::iter::repeat_n(crate::glmm::BETA_BOX, p));

    // ρ schedule: mirror `GlmmWorkspace::for_cluster_spec` — ρ_begin ≤ RHO_BEGIN
    // and ≤ 0.1·min diagonal θ₀ (= 0.1·THETA0 on the blind start), ρ_end the
    // GLMM-calibrated GLMM_RHO_END, PRIMA-default npt for the joint dimension.
    let rho_begin = (0.1 * crate::lmm::THETA0).min(crate::lmm::RHO_BEGIN);

    // STAGE 1 — θ-only BOBYQA on the PQL objective (β profiled inside PIRLS),
    // mirroring the dense two-stage optimizer: an accelerant that warm-starts
    // the joint stage 2 and never gates convergence. Not optional garnish here:
    // on the over-width gamma golden the single-stage joint solve (dim
    // n_theta + p = 21) stalled ~0.24 deviance units short of the optimum along
    // the weakly-identified intercept↔RE direction; profiling β collapses that
    // valley and stage 2 polishes from the PQL point to the Laplace optimum.
    // Each eval re-seeds `ws.beta` from the same fixed β₀, so the stage-1
    // objective is a deterministic function of θ alone (the order-free
    // requirement the dense stage 1 meets through its incumbent snapshots).
    let n_eval_stage1;
    {
        let npt1 = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        // MIRRORS `config_stage1` in `GlmmWorkspace::from_groupings` — both
        // feed through the shared `apply_campaign_overrides` tail.
        let mut config1 = bobyqa::Config {
            rho_begin,
            rho_end: crate::lmm::GLMM_RHO_END,
            npt: npt1,
            ..bobyqa::Config::new(n_theta)
        };
        crate::lmm::apply_campaign_overrides(&mut config1, n_theta);
        let mut solver1 = bobyqa::Bobyqa::new(n_theta, config1)
            .expect("BOBYQA config constants are valid by construction");
        let beta0: Vec<f64> = params[n_theta..].to_vec();
        let mut theta1: Vec<f64> = params[..n_theta].to_vec();
        let out1 = solver1.minimize(
            |theta| {
                ws.beta[..p].copy_from_slice(&beta0);
                sparse_glmm_deviance(family, nb_theta, theta, &mut ws, xm, y, n, true)
            },
            &mut theta1,
            &lower[..n_theta],
            &upper[..n_theta],
        );
        n_eval_stage1 = out1.n_eval;
        // Warm-start stage 2 at (θ̂₁, β̂(θ̂₁)): one more Profile eval at the
        // incumbent θ̂₁ leaves the profiled β in ws.beta. A non-finite eval
        // (never seen at an incumbent) just keeps the stage-1-independent seed.
        ws.beta[..p].copy_from_slice(&beta0);
        let d1 = sparse_glmm_deviance(family, nb_theta, &theta1, &mut ws, xm, y, n, true);
        if d1.is_finite() {
            params[..n_theta].copy_from_slice(&theta1);
            for (slot, &b) in params[n_theta..].iter_mut().zip(&ws.beta[..p]) {
                *slot = b.clamp(-crate::glmm::BETA_BOX, crate::glmm::BETA_BOX);
            }
        }
    }

    // STAGE 2 — joint [θ | β] polish on the true Laplace objective (β-Fixed
    // per eval), the dense kernel's stage-2 shape. Only this stage's status
    // feeds `converged`. MIRRORS the joint config in
    // `GlmmWorkspace::from_groupings` — both feed through the shared
    // `apply_campaign_overrides` tail.
    let mut config = bobyqa::Config {
        rho_begin,
        rho_end: crate::lmm::GLMM_RHO_END,
        ..bobyqa::Config::new(n_theta + p)
    };
    crate::lmm::apply_campaign_overrides(&mut config, n_theta + p);
    let mut solver = bobyqa::Bobyqa::new(n_theta + p, config)
        .expect("BOBYQA config constants are valid by construction");
    let out = solver.minimize(
        |gamma| sparse_glmm_deviance(family, nb_theta, gamma, &mut ws, xm, y, n, false),
        &mut params,
        &lower,
        &upper,
    );
    debug_assert!(out.status != Status::InvalidArgs);
    let mut ok = matches!(out.status, Status::Converged);

    // Diagonal-θ pin (mirror `glmm::fit_glmm`; β never pins).
    let mut pinned = false;
    if ok {
        for &ti in g.diagonal_theta() {
            if params[ti] <= crate::lmm::PIN_THETA {
                params[ti] = 0.0;
                pinned = true;
            }
        }
    }
    let n_eval = n_eval_stage1 + out.n_eval;
    // Pinned-γ̂ re-eval: refreshes W̃/û/μ̂/A for the inference reads below, and its
    // finite deviance is the degenerate-fit witness (dense kernel's guard).
    let mut final_deviance = f64::INFINITY;
    if ok {
        final_deviance = sparse_glmm_deviance(family, nb_theta, &params, &mut ws, xm, y, n, false);
        ok = final_deviance.is_finite();
    }
    if !ok {
        return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
    }

    let beta: Vec<f64> = params[n_theta..].to_vec();


    // tau2 / dispersion / varcorr off the converged state, BEFORE the FD-Hessian
    // perturbs the workspace (mirrors `fit::fit_glmm`'s mapping, including
    // Gamma's pwrss/n τ² scale and Pearson dispersion).
    let sigma_sq = crate::family::glmm_sigma_sq(
        family,
        &y[..n],
        &ws.prob[..n],
        &ws.u[..ws.k],
        Some(&ws.prior_w[..n]),
    );
    let tau2: Vec<f64> = params[..n_theta]
        .iter()
        .map(|&t| t * t * sigma_sq)
        .collect();
    let dispersion = match family {
        // Weighted Pearson moment φ̂ = Σwᵢrᵢ²/(n−p) — raw n−p df, not Σwᵢ−p
        // (mirrors `fit::fit_glmm`'s Gamma arm, fit.rs:1695-1708).
        crate::Family::Gamma { .. } => match opts.dispersion {
            Some(v) => v,
            None => {
                let mut s = 0.0;
                for (i, (&yi, &mu)) in y[..n].iter().zip(ws.prob[..n].iter()).enumerate() {
                    let r = (yi - mu) / crate::family::variance(family, nb_theta, mu).sqrt();
                    s += ws.prior_w[i] * r * r;
                }
                s / (n - p) as f64
            }
        },
        _ => 1.0,
    };
    // σ̂²-scaled like tau2 above (lme4 VarCorr; σ̂² ≡ 1 for φ≡1 families —
    // mirrors `fit::fit_glmm`'s varcorr, change together).
    let varcorr = crate::fit::assemble_varcorr(&params[..n_theta], &g, sigma_sq);

    // SE per WaldSe arm. The Rx Schur reads the converged W̃/A the re-eval left;
    // the FD-Hessian perturbs the workspace, so its Rx FALLBACK re-evals at γ̂
    // first to restore that state.
    let mut se = vec![f64::NAN; p];
    let mut stddev_se = vec![f64::NAN; n_theta];
    let cov_from_schur = |schur: Mat<f64>, se: &mut [f64]| -> bool {
        // Var(β̂)_jj = σ̂²·‖L⁻¹e_j‖² per target from chol(S_β) (mirror the dense Rx
        // arm, including Gamma's σ̂² on the RX vcov — lme4 `vcov(use.hessian=FALSE)`).
        let sc = match schur.as_ref().llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let lschur = sc.L();
        let mut fwd = vec![0.0f64; p];
        for &tj in &opts.target_indices {
            let tj = tj as usize;
            for i in 0..p {
                let mut acc = if i == tj { 1.0 } else { 0.0 };
                for kk in 0..i {
                    acc -= lschur[(i, kk)] * fwd[kk];
                }
                fwd[i] = acc / lschur[(i, i)];
            }
            let vd: f64 = fwd[..p].iter().map(|v| v * v).sum::<f64>() * sigma_sq;
            if vd.is_finite() && vd >= 0.0 {
                se[tj] = vd.sqrt();
            }
        }
        true
    };
    match opts.wald_se {
        crate::WaldSe::Rx => {
            let schur = match sparse_glmm_schur(&mut ws, xm, n) {
                Some(s) => s,
                None => return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY),
            };
            if !cov_from_schur(schur, &mut se) {
                return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
            }
        }
        crate::WaldSe::Hessian => {
            // FD evals (and the fallback's central re-eval below) converge PIRLS at
            // the FD-only tight tol; reset right after the match so the returned-fit
            // workspace never leaks it (see sparse_fd_hessian_cov's contract).
            ws.pirls_tol_override = Some(crate::glmm::PIRLS_TOL_REL_FD);
            match sparse_fd_hessian_cov(
                family,
                nb_theta,
                &params,
                &mut ws,
                xm,
                y,
                n,
                opts.parallel_inner,
            ) {
                Some((cov, tse)) => {
                    for &tj in &opts.target_indices {
                        let tj = tj as usize;
                        let vd = cov[(tj, tj)];
                        if vd.is_finite() && vd >= 0.0 {
                            se[tj] = vd.sqrt();
                        }
                    }
                    stddev_se.copy_from_slice(&tse);
                }
                None => {
                    // RX fallback (the dense `NonPdFellBackToRx` shape): restore the
                    // converged workspace state the FD loop perturbed, then Schur. The
                    // re-eval runs cold-seeded/tight-tol (see `sparse_glmm_deviance`'s
                    // doc comment), so it is not guaranteed to land back at a finite
                    // deviance for a near-degenerate design — mirrors the dense
                    // `fallback!()` macro (`glmm/se.rs`), which also discards this
                    // return value and gates correctness on the Schur PD check below (a
                    // double failure there already routes to `sparse_glmm_nan_fit`).
                    let _ =
                        sparse_glmm_deviance(family, nb_theta, &params, &mut ws, xm, y, n, false);
                    let schur = match sparse_glmm_schur(&mut ws, xm, n) {
                        Some(s) => s,
                        None => return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY),
                    };
                    if !cov_from_schur(schur, &mut se) {
                        return (sparse_glmm_nan_fit(p, n_theta), f64::INFINITY);
                    }
                    // No joint Hessian ⇒ no θ-block SE (stays NaN), as the dense
                    // fallback reports.
                }
            }
            ws.pirls_tol_override = None; // never leak the FD tight tol past the SE step
        }
    }

    (
        crate::Fit {
            beta,
            se,
            tau2,
            dispersion,
            converged: true,
            varcorr,
            stddev_se,
            aliased: vec![false; p],
            n_eval,
            deviance: final_deviance,
            singular: pinned,
        },
        final_deviance,
    )
}

/// Sparse-Z negative-binomial GLMM: the over-envelope sibling of
/// `fit::fit_glmm_nb`, same **marginal-θ** profile (`lme4::glmer.nb`) — for
/// each candidate θ the inner `fit_glmm_sparse` re-fits the full GLMM at that
/// fixed θ and its minimized marginal Laplace deviance feeds
/// `logL_marginal(θ) = −½·D(θ) + nb_profile_loglik(y, y, θ, weights)`, maximized
/// over `ln θ` by the shared golden-section bracket (mirrors `fit::fit_glmm_nb`,
/// fit.rs:1806). The spec is θ-free (the NB shape is threaded explicitly per
/// candidate); a warm `start` is irrelevant to the global bracket search,
/// exactly as on the dense path. `dispersion = θ̂`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glmm_nb_sparse(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &crate::ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    _start: Option<&crate::StartValues>,
    opts: &crate::FitOptions,
) -> crate::Fit {
    let nb_spec = crate::ModelSpec {
        family: crate::Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: model.re.clone(),
    };
    let theta = crate::fit::golden_max_ln_theta(|t| {
        let th = t.exp();
        let (_fit, dev) =
            fit_glmm_sparse(x, y, n, p, &nb_spec, cluster_ids, extra_ids, th, None, opts);
        -0.5 * dev + crate::fit::nb_profile_loglik(y, y, th, opts.weights.as_deref())
    });
    let mut fit_result = fit_glmm_sparse(
        x,
        y,
        n,
        p,
        &nb_spec,
        cluster_ids,
        extra_ids,
        theta,
        None,
        opts,
    )
    .0;
    fit_result.dispersion = theta;
    fit_result
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

#[cfg(test)]
mod tests {
    use super::*;
    // faer sparse Cholesky call sequence — verified against faer 0.24.4 source.
    use faer::sparse::linalg::cholesky::{factorize_symbolic_cholesky, CholeskySymbolicParams};
    use faer::sparse::linalg::SupernodalThreshold;
    use faer::sparse::{SparseColMat, Triplet};
    // Real path for LltRegularization (the sparse cholesky module only uses it privately).
    use faer::dyn_stack::{MemBuffer, MemStack};
    use faer::linalg::cholesky::llt::factor::LltRegularization;
    use faer::{Conj, Mat, Par, Side, Spec};
    // Solve trait: blanket impl for SolveCore<T>; must be in scope to call .solve().
    use faer::linalg::solvers::Solve;
    // AsMatMut: gives as_mat_mut() → MatMut<'_, T>; as_mut() gives &mut Mat, wrong type.
    use faer::mat::AsMatMut;

    use crate::{Family, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing};

    /// One extra grouping's per-row level ids, packed as the `extra_ids` shape
    /// (`&[Vec<u32>]`) `fit_mle_sparse` expects.
    fn cr_as_extra(cr: &[u32]) -> Vec<Vec<u32>> {
        vec![cr.to_vec()]
    }

    /// Run `f` with the sparse-tail branch forced on (workspaces built inside
    /// take the fill-reducing factor regardless of `TAIL_SPARSE_MIN`). Restores
    /// the flag before returning; each #[test] has its own thread, so a
    /// panicked test cannot leak the flag into another.
    fn with_forced_sparse_tail<T>(f: impl FnOnce() -> T) -> T {
        super::FORCE_SPARSE_TAIL.with(|c| c.set(true));
        let out = f();
        super::FORCE_SPARSE_TAIL.with(|c| c.set(false));
        out
    }

    /// fit_mle_sparse on an in-envelope crossed LMM matches the NoZ fit_mle on
    /// β, varcomp, and SE — the superset property. This is the
    /// unit-level seed of the both-paths cross-check harness.
    #[test]
    fn fit_mle_sparse_matches_noz_in_envelope() {
        use faer::Mat;
        let n = 24;
        let p = 2;
        let mut xflat = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        let mut cl = vec![0u32; n];
        let mut cr = vec![0u32; n];
        let mut st = 5u64;
        for i in 0..n {
            let cov = super::test_lcg(&mut st);
            xflat[i * p] = 1.0;
            xflat[i * p + 1] = cov;
            cl[i] = (i % 4) as u32;
            cr[i] = (i % 3) as u32;
            y[i] = 1.0 + 0.5 * cov + super::test_lcg(&mut st);
        }
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 3 },
                    slopes: vec![],
                }],
            }),
        };
        let ids = crate::GroupIds {
            primary: cl.clone(),
            extra: vec![cr.clone()],
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };

        let noz = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts); // in-envelope ⇒ NoZ
                                                                          // Force the sparse path directly (bypassing classify_design's NoZ route).
        let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);
        let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        let sp = super::fit_mle_sparse(
            &xflat,
            &y,
            n,
            p,
            &sized,
            &cl,
            &cr_as_extra(&cr),
            None,
            &opts,
        );

        assert!(sp.converged && noz.converged);
        for j in 0..p {
            assert!(
                (sp.beta[j] - noz.beta[j]).abs() < 1e-6,
                "β{j} sparse {} vs noz {}",
                sp.beta[j],
                noz.beta[j]
            );
            assert!(
                (sp.se[j] - noz.se[j]).abs() < 1e-6,
                "se{j} sparse {} vs noz {}",
                sp.se[j],
                noz.se[j]
            );
        }
        assert_eq!(sp.varcorr.len(), noz.varcorr.len());
        for (a, b) in sp
            .varcorr
            .iter()
            .flatten()
            .zip(noz.varcorr.iter().flatten())
        {
            assert!((a - b).abs() < 1e-6, "varcorr {a} vs {b}");
        }
        let _ = (x, model);
    }

    /// Route invariance for detected flat nesting: when every child level falls
    /// under a single parent, the nested (padded per-parent ids, `NestedWithin`)
    /// and crossed (flat global ids, `Crossed`) parameterizations are the SAME
    /// statistical model, so REML deviance and β must agree across routes. This
    /// is the correctness lever behind the frontend's inflation-bound detection
    /// (`detect_flat_nesting`): whichever way a near-balanced factor is routed,
    /// the answer cannot change. Near-balanced shape: children-per-parent
    /// {3,2,3} over 3 parents (8 child levels, padded dim 9). Run for both
    /// tail branches (see `sparse_deviance_equals_dense_crossed`).
    #[test]
    fn nested_route_matches_forced_crossed_sparse() {
        run_nested_route_matches_forced_crossed_sparse();
    }
    #[test]
    fn nested_route_matches_forced_crossed_sparse_sparse_tail() {
        with_forced_sparse_tail(run_nested_route_matches_forced_crossed_sparse);
    }
    fn run_nested_route_matches_forced_crossed_sparse() {
        // 8 children × 6 obs each = 48 rows. Parent of child c: 0,0,0,1,1,2,2,2.
        let parent_of_child: [u32; 8] = [0, 0, 0, 1, 1, 2, 2, 2];
        // Padded per-parent layout (W = 3): child c → parent·3 + local index.
        let padded_of_child: [u32; 8] = [0, 1, 2, 3, 4, 6, 7, 8];
        let n = 48;
        let p = 2;
        let mut xflat = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        let mut cl = vec![0u32; n];
        let mut flat = vec![0u32; n];
        let mut padded = vec![0u32; n];
        let mut st = 11u64;
        let parent_eff = [0.8, -0.3, 0.1];
        let child_eff: Vec<f64> = (0..8).map(|_| 0.5 * super::test_lcg(&mut st)).collect();
        for i in 0..n {
            let c = (i % 8) as u32;
            let cov = super::test_lcg(&mut st);
            xflat[i * p] = 1.0;
            xflat[i * p + 1] = cov;
            cl[i] = parent_of_child[c as usize];
            flat[i] = c;
            padded[i] = padded_of_child[c as usize];
            y[i] = 1.0
                + 0.5 * cov
                + parent_eff[cl[i] as usize]
                + child_eff[c as usize]
                + 0.3 * super::test_lcg(&mut st);
        }
        let spec = |relation: GroupingRelation| ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 3 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation,
                    slopes: vec![],
                }],
            }),
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        // Detection-nested route: fit_cold routes it NoZ (nested levels don't
        // count toward the crossed cap) and eliminates per family.
        let nested_model = spec(GroupingRelation::NestedWithin { n_per_parent: 1 });
        let nested_ids = crate::GroupIds {
            primary: cl.clone(),
            extra: vec![padded.clone()],
        };
        let nf = crate::fit_cold(&xflat, &y, n, p, &nested_model, &nested_ids, &opts);
        // Forced-crossed Sparse route on the SAME data, flat ids.
        let crossed_model = spec(GroupingRelation::Crossed { n_clusters: 1 });
        let crossed_ids = crate::GroupIds {
            primary: cl.clone(),
            extra: vec![flat.clone()],
        };
        let sized = crate::fit::spec_sized_from_ids_pub(&crossed_model, &crossed_ids);
        let sp = super::fit_mle_sparse(
            &xflat,
            &y,
            n,
            p,
            &sized,
            &cl,
            &cr_as_extra(&flat),
            None,
            &opts,
        );
        assert!(nf.converged && sp.converged);
        assert!(
            (nf.deviance - sp.deviance).abs() < 1e-6 * sp.deviance.abs().max(1.0),
            "deviance nested {} vs forced-crossed sparse {}",
            nf.deviance,
            sp.deviance
        );
        for j in 0..p {
            assert!(
                (nf.beta[j] - sp.beta[j]).abs() < 1e-5,
                "β{j} nested {} vs crossed {}",
                nf.beta[j],
                sp.beta[j]
            );
        }
    }

    /// A >32-variance-component design (over-envelope-by-count ⇒ sparse) fits
    /// through the sparse-Z path without the two grouping-cap panics the shared
    /// NoZ structures would hit: `add_rows_multi`'s fixed `[usize; 1+MAX_EXTRA_
    /// GROUPINGS]` gid array (dropped from `SparseLmmWorkspace::new`)
    /// and `from_cluster_spec_ext`'s `n_extras <= MAX_EXTRA_GROUPINGS` guard
    /// (removed). 33 components (1 primary + 32 crossed) also exceeds
    /// the old 32-bit `pinned_components` ceiling (u64 now).
    #[test]
    fn sparse_over_32_components_no_overflow() {
        const N_EXTRA: usize = 32; // > MAX_EXTRA_GROUPINGS=6 ⇒ Sparse; +1 primary ⇒ 33 comps
        let n = 60;
        let p = 1;
        let xflat = vec![1.0f64; n * p]; // intercept-only fixed block
        let mut y = vec![0.0f64; n];
        let mut st = 11u64;
        // Each extra grouping g has (2 + g % 2) levels — modest, well-populated.
        let extra: Vec<Vec<u32>> = (0..N_EXTRA)
            .map(|g| {
                let levels = 2 + (g % 2) as u32;
                (0..n).map(|i| (i as u32) % levels).collect()
            })
            .collect();
        let primary: Vec<u32> = (0..n).map(|i| (i % 4) as u32).collect();
        for yi in y.iter_mut() {
            *yi = 1.0 + 0.5 * super::test_lcg(&mut st);
        }

        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![],
                extra_groupings: (0..N_EXTRA)
                    .map(|g| Grouping {
                        relation: GroupingRelation::Crossed {
                            n_clusters: 2 + (g % 2) as u32,
                        },
                        slopes: vec![],
                    })
                    .collect(),
            }),
        };
        // Over-envelope-by-count ⇒ classify_design routes to Sparse.
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ));

        let ids = crate::GroupIds { primary, extra };
        let opts = crate::FitOptions {
            target_indices: vec![0],
            ..crate::FitOptions::default()
        };
        let f = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);

        // The bar is: no panic/overflow through the sparse path, and a finite fit.
        assert!(f.converged, "33-component sparse fit converged");
        assert!(f.beta[0].is_finite(), "β finite");
        assert!(f.se[0].is_finite(), "se finite");
    }

    /// Spike: prove the faer 0.24 sparse-LLT call sequence + logdet-off-the-CSC
    /// convention on a hand-checked 3×3 SPD matrix. Locks the API the whole
    /// sparse path is built on. det(A)=18 ⇒ logdet=ln 18.
    #[test]
    fn sparse_llt_spike_logdet_and_solve() {
        let n = 3usize;
        // Lower triangle of A = [[4,1,0],[1,3,1],[0,1,2]].
        let tri = [
            Triplet::new(0usize, 0usize, 4.0f64),
            Triplet::new(1, 0, 1.0),
            Triplet::new(1, 1, 3.0),
            Triplet::new(2, 1, 1.0),
            Triplet::new(2, 2, 2.0),
        ];
        let a = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &tri).unwrap();

        let params = CholeskySymbolicParams {
            supernodal_flop_ratio_threshold: SupernodalThreshold::FORCE_SIMPLICIAL,
            ..Default::default()
        };
        let symbolic = factorize_symbolic_cholesky(
            a.symbolic(),
            Side::Lower,
            Default::default(), // fill-reducing ordering (AMD-family default)
            params,
        )
        .expect("symbolic factorization");

        let mut l_values = vec![0.0f64; symbolic.len_val()];
        let fac_req = symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default());
        let mut fac_mem = MemBuffer::new(fac_req);
        let llt = symbolic
            .factorize_numeric_llt(
                &mut l_values,
                a.as_ref(),
                Side::Lower,
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(&mut fac_mem),
                Spec::default(),
            )
            .expect("numeric LLT (A is SPD)");

        // Solve A x = b first: llt holds &'out [T] into l_values; must end that
        // borrow before taking &l_values for logdet_llt. `let _ = llt` is the
        // last use of llt (LltRef is Copy) so NLL ends the borrow on the next line.
        let mut rhs = Mat::<f64>::from_fn(3, 1, |i, _| [1.0, 2.0, 3.0][i]);
        let solve_req = symbolic.solve_in_place_scratch::<f64>(1, Par::Seq);
        let mut solve_mem = MemBuffer::new(solve_req);
        llt.solve_in_place_with_conj(
            Conj::No,
            rhs.as_mat_mut(),
            Par::Seq,
            MemStack::new(&mut solve_mem),
        );
        let _ = llt; // ends &'out borrow on l_values

        // Verify logdet: det([[4,1,0],[1,3,1],[0,1,2]]) = 18 ⇒ log det = ln 18.
        let logdet = logdet_llt(&symbolic, &l_values);
        assert!(
            (logdet - 18.0f64.ln()).abs() < 1e-10,
            "logdet {logdet} vs ln 18"
        );

        // Verify solve against the dense LLT of the same matrix.
        let dense = Mat::<f64>::from_fn(3, 3, |i, j| {
            [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]][i][j]
        });
        let bref = Mat::<f64>::from_fn(3, 1, |i, _| [1.0, 2.0, 3.0][i]);
        let x_dense = dense.llt(Side::Lower).unwrap().solve(bref.as_ref());
        for i in 0..3 {
            assert!(
                (rhs[(i, 0)] - x_dense[(i, 0)]).abs() < 1e-10,
                "x[{i}] {} vs dense",
                rhs[(i, 0)]
            );
        }
    }

    /// Blocked-kernel logdet oracle: `sparse_schur_factor`'s log|L_ZZ|² at an
    /// identity-Λ θ (all scalar components 1.0) matches the dense
    /// `logdet(Z'Z + I)` — the θ=identity-Λ case, where A = Z'Z + I exactly.
    #[test]
    fn blocked_logdet_matches_dense_ztz_plus_i() {
        use faer::Mat;
        let n = 4;
        let p = 1;
        let x = Mat::<f64>::from_fn(n, p, |_, _| 1.0);
        let cluster_ids = [0u32, 0, 1, 1];
        let extra_ids = vec![vec![0u32, 1, 0, 1]];
        let y = [1.0f64, 2.0, 3.0, 4.0];
        let model = crate::ModelSpec {
            family: crate::Family::Gaussian,
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 2 },
                    slopes: vec![],
                }],
            }),
        };
        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);
        let z = super::build_sparse_z(&g, x.as_ref(), &cluster_ids, &extra_ids, n);

        let mut ws = super::SparseLmmWorkspace::new(
            &g,
            x.as_ref(),
            &cluster_ids,
            &extra_ids,
            &y,
            n,
            p,
            None,
        );
        // θ = [1, 1] (primary scalar, crossed scalar) ⇒ Λ = I ⇒ A = Z'Z + I.
        let ld = super::sparse_schur_factor(&[1.0, 1.0], &mut ws).expect("Z'Z + I is SPD");

        // Dense reference: logdet(Z'Z + I).
        let zd = z.to_dense();
        let mut ztz = zd.transpose() * &zd;
        for d in 0..g.k_total {
            ztz[(d, d)] += 1.0;
        }
        let dense_ld = {
            let l = ztz.llt(faer::Side::Lower).unwrap();
            let ld_mat = l.L();
            let mut s = 0.0;
            for d in 0..g.k_total {
                s += ld_mat[(d, d)].ln();
            }
            2.0 * s
        };
        assert!(
            (ld - dense_ld).abs() < 1e-9,
            "blocked logdet {ld} vs dense {dense_ld}"
        );
    }

    /// build_sparse_z lays columns out in no-Z RE-column order
    /// `[primary | crossed]` with a 1 per row in its primary level and its
    /// crossed level (scalar intercepts). Checked against the dense pattern.
    #[test]
    fn sparse_z_matches_dense_crossed_intercept() {
        use faer::Mat;
        let n = 4;
        let p = 1;
        // Intercept-only X (unused for intercept RE columns, but the signature takes it).
        let x = Mat::<f64>::from_fn(n, p, |_, _| 1.0);
        let cluster_ids = [0u32, 0, 1, 1]; // 2 primary levels
        let extra_ids = vec![vec![0u32, 1, 0, 1]]; // 1 crossed grouping, 2 levels

        let model = crate::ModelSpec {
            family: crate::Family::Gaussian,
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 2 },
                    slopes: vec![],
                }],
            }),
        };
        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);
        let z = super::build_sparse_z(&g, x.as_ref(), &cluster_ids, &extra_ids, n);

        assert_eq!(z.nrows(), n);
        assert_eq!(z.ncols(), g.k_total);
        // Densify and compare to the expected [primary(2) | crossed(2)] pattern.
        let dense = z.to_dense();
        let expect = [
            [1.0, 0.0, 1.0, 0.0], // row0: primary lvl0, crossed lvl0
            [1.0, 0.0, 0.0, 1.0], // row1: primary lvl0, crossed lvl1
            [0.0, 1.0, 1.0, 0.0], // row2: primary lvl1, crossed lvl0
            [0.0, 1.0, 0.0, 1.0], // row3: primary lvl1, crossed lvl1
        ];
        for i in 0..n {
            for j in 0..g.k_total {
                assert!(
                    (dense[(i, j)] - expect[i][j]).abs() < 1e-12,
                    "Z[{i},{j}] {}",
                    dense[(i, j)]
                );
            }
        }
    }

    /// sparse_reml_deviance equals lmm::reml_deviance at the same θ on an
    /// in-envelope design — the free cross-check at the deviance-value level.
    /// If these disagree, exactly one path is wrong. Run for BOTH tail
    /// branches: bare (dense tail, e=3 ≤ TAIL_SPARSE_MIN) and forced-sparse
    /// (the fill-reducing factor at the same tolerance — AMD reordering is a
    /// sanctioned reassociation).
    #[test]
    fn sparse_deviance_equals_dense_crossed() {
        run_sparse_deviance_equals_dense_crossed();
    }
    #[test]
    fn sparse_deviance_equals_dense_crossed_sparse_tail() {
        with_forced_sparse_tail(run_sparse_deviance_equals_dense_crossed);
    }
    fn run_sparse_deviance_equals_dense_crossed() {
        use crate::{Family, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing};
        use faer::Mat;
        // Small crossed LMM: primary (2 levels) + 1 crossed grouping (3 levels),
        // scalar intercepts, p=2 fixed (intercept + 1 covariate).
        let n = 12;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        let mut y = vec![0.0f64; n];
        let mut cl = vec![0u32; n];
        let mut cr = vec![0u32; n];
        let mut st = 11u64;
        for i in 0..n {
            let cov = super::test_lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = cov;
            cl[i] = (i % 2) as u32;
            cr[i] = (i % 3) as u32;
            y[i] = 1.0 + 0.5 * cov + super::test_lcg(&mut st);
        }
        let extra_ids = vec![cr.clone()];
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 3 },
                    slopes: vec![],
                }],
            }),
        };
        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);

        // Dense reference workspace.
        let mut suff = crate::lmm::LmmSuffStats::with_groupings(p, g.clone());
        suff.add_rows_multi(x.as_ref(), &y, &cl, &extra_ids, None);
        let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);

        let mut ws =
            super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None);

        // θ has 2 scalar components here (primary intercept var, crossed var).
        for theta in [[0.5f64, 0.7], [1.0, 0.2], [0.1, 1.3]] {
            let dense = crate::lmm::reml_deviance(&theta, &suff, &mut fit);
            let sparse = super::sparse_reml_deviance(&theta, &mut ws);
            assert!(
                (dense - sparse).abs() < 1e-8 * (1.0 + dense.abs()),
                "θ={theta:?}: dense {dense} vs sparse {sparse}"
            );
        }
    }

    /// Regression for the structural-pattern seeding fix: a PRIMARY RANDOM SLOPE
    /// (q_p=2) design whose slope covariate is balanced ±1 within each primary
    /// cluster, so `Σx = 0` exactly over every cluster's rows → the intercept×slope
    /// cross-Gram entry `Z'Z[(n_prim+f, f)]` is EXACTLY 0.0. Under the old numeric
    /// seeding (`v != 0.0`) that off-diagonal within-block slot was never reserved,
    /// so a non-diagonal Λ's fill there (Λ'GΛ has a nonzero at that slot) was
    /// silently dropped → wrong A → wrong deviance with no error. With structural
    /// seeding (`|Z|ᵀ|Z| > 0.0`) the slot exists and the deviance matches the dense
    /// oracle. θ is chosen with all three vech components nonzero so Λ is genuinely
    /// non-diagonal. Random-continuous data can't hit the exact zero — it must be
    /// constructed. Companion to `sparse_deviance_equals_dense_crossed` (scalar Λ).
    #[test]
    fn sparse_deviance_equals_dense_primary_slope_balanced_zero() {
        use crate::{Family, ModelSpec, ReStructure, Sizing};
        use faer::Mat;
        // 3 primary clusters × 4 rows; within each, 2 rows x=+1 and 2 rows x=-1.
        let n = 12;
        let p = 2; // intercept + the ±1 slope covariate as fixed effects
        let n_clusters = 3u32;
        let mut x = Mat::<f64>::zeros(n, p);
        let mut y = vec![0.0f64; n];
        let mut cl = vec![0u32; n];
        let mut st = 7u64;
        for i in 0..n {
            let slope_cov = if i % 4 < 2 { 1.0 } else { -1.0 }; // Σ over each cluster = 0
            x[(i, 0)] = 1.0;
            x[(i, 1)] = slope_cov;
            cl[i] = (i / 4) as u32;
            y[i] = 0.5 + 0.3 * slope_cov + super::test_lcg(&mut st);
        }
        let extra_ids: Vec<Vec<u32>> = vec![]; // no extra groupings
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters },
                slopes: vec![1], // primary random slope on x column 1
                extra_groupings: vec![],
            }),
        };
        // Primary slope x-col = 1; no extra groupings.
        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[1], &[]);
        assert_eq!(g.primary_q, 2, "design must be a q_p=2 primary slope");

        // Confirm the target cross-Gram entry is EXACTLY 0.0 (would be dropped
        // under old numeric seeding): intercept col f vs slope col n_prim+f.
        let z = super::build_sparse_z(&g, x.as_ref(), &cl, &extra_ids, n);
        let ztz = z.to_dense().transpose() * &z.to_dense();
        let n_prim = g.n_primary;
        for f in 0..n_prim {
            assert_eq!(
                ztz[(n_prim + f, f)],
                0.0,
                "cross-Gram (slope,intercept) at cluster {f} must be exactly 0"
            );
        }

        // Dense reference workspace (the oracle).
        let mut suff = crate::lmm::LmmSuffStats::with_groupings(p, g.clone());
        suff.add_rows_multi(x.as_ref(), &y, &cl, &extra_ids, None);
        let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);

        let mut ws =
            super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None);

        // θ = vech(Λ) with all three components nonzero → Λ genuinely non-diagonal.
        for theta in [[0.8f64, 0.3, 0.6], [1.0, 0.5, 0.4], [0.2, 0.7, 0.9]] {
            let dense = crate::lmm::reml_deviance(&theta, &suff, &mut fit);
            let sparse = super::sparse_reml_deviance(&theta, &mut ws);
            assert!(
                (dense - sparse).abs() < 1e-8 * (1.0 + dense.abs()),
                "θ={theta:?}: dense {dense} vs sparse {sparse}"
            );
        }
    }

    /// Sparse-tail S22 pattern, pinned on a hand-built two-family fixture with
    /// known cliques: 2 primary families over a 5-level crossed factor
    /// (declared `n_clusters: 5`), family 0 touching levels {0,1}, family 1
    /// {1,2}; levels 3–4 UNOBSERVED (spec count > observed ids — reachable,
    /// `n_levels` is unclamped by the ids). Expected scalar pattern (lower):
    /// the full diagonal plus exactly the within-clique couplings (1,0) and
    /// (2,1) — nothing couples 0↔2 (different families), and the unobserved
    /// columns are diagonal-only (their `+I` slot must still exist). Also pins
    /// the numerics: forced-sparse deviance equals the dense-tail branch's at
    /// several θ, and the full fit matches, unobserved levels included.
    #[test]
    fn sparse_tail_pattern_two_family_cliques_unobserved_level() {
        use faer::Mat;
        let n = 12;
        let p = 1;
        let x = Mat::<f64>::from_fn(n, p, |_, _| 1.0);
        let xflat = vec![1.0f64; n * p];
        let mut y = vec![0.0f64; n];
        let mut st = 23u64;
        for v in y.iter_mut() {
            *v = 1.0 + super::test_lcg(&mut st);
        }
        // 3 replicates of each (family, crossed) incidence pair.
        let cl: Vec<u32> = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let cr: Vec<u32> = vec![0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2];
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 5 },
                    slopes: vec![],
                }],
            }),
        };
        let extra_ids = cr_as_extra(&cr);
        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[vec![]]);
        assert_eq!(g.k_crossed(), 5, "unobserved levels counted from the spec");

        let mut ws_sparse = with_forced_sparse_tail(|| {
            super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None)
        });
        let tail = ws_sparse.tail.as_ref().expect("forced sparse tail");
        let sym = tail.axx.symbolic();
        assert_eq!(
            sym.col_ptr(),
            &[0usize, 2, 4, 5, 6, 7],
            "clique-exact col_ptr"
        );
        assert_eq!(
            sym.row_idx(),
            &[0usize, 1, 1, 2, 2, 3, 4],
            "clique-exact row_idx"
        );

        // Deviance equality vs the dense-tail branch at several θ …
        let mut ws_dense =
            super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p, None);
        assert!(ws_dense.tail.is_none(), "e=5 stays on the dense tail bare");
        for theta in [[0.5f64, 0.7], [1.0, 0.2], [0.1, 1.3]] {
            let d = super::sparse_reml_deviance(&theta, &mut ws_dense);
            let s = super::sparse_reml_deviance(&theta, &mut ws_sparse);
            assert!(
                (d - s).abs() < 1e-8 * (1.0 + d.abs()),
                "θ={theta:?}: dense-tail {d} vs sparse-tail {s}"
            );
        }
        // … and full-fit equality (deviance/β/se), unobserved levels included.
        let opts = crate::FitOptions {
            target_indices: vec![0],
            ..crate::FitOptions::default()
        };
        let fd = super::fit_mle_sparse(&xflat, &y, n, p, &model, &cl, &extra_ids, None, &opts);
        let fs = with_forced_sparse_tail(|| {
            super::fit_mle_sparse(&xflat, &y, n, p, &model, &cl, &extra_ids, None, &opts)
        });
        assert!(fd.converged && fs.converged);
        assert!(
            (fd.deviance - fs.deviance).abs() < 1e-8 * (1.0 + fd.deviance.abs()),
            "deviance dense-tail {} vs sparse-tail {}",
            fd.deviance,
            fs.deviance
        );
        assert!((fd.beta[0] - fs.beta[0]).abs() < 1e-6);
        assert!((fd.se[0] - fs.se[0]).abs() < 1e-6);
    }

    /// One natural fixture over the cutover (e = 150 > TAIL_SPARSE_MIN): the
    /// un-overridden workspace takes the sparse tail, and the fit matches the
    /// NoZ oracle (150 crossed levels stay under MAX_CROSSED_LEVELS, so
    /// fit_cold routes dense NoZ) — the superset property of
    /// `fit_mle_sparse_matches_noz_in_envelope`, across the branch boundary.
    #[test]
    fn sparse_tail_natural_over_cutover_matches_noz() {
        use faer::Mat;
        let n = 600;
        let p = 2;
        let e_levels = 150u32;
        assert!((e_levels as usize) > super::TAIL_SPARSE_MIN);
        let mut xflat = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        let mut cl = vec![0u32; n];
        let mut cr = vec![0u32; n];
        let mut st = 31u64;
        for i in 0..n {
            let cov = super::test_lcg(&mut st);
            xflat[i * p] = 1.0;
            xflat[i * p + 1] = cov;
            cl[i] = (i % 4) as u32;
            cr[i] = (i as u32) % e_levels;
            y[i] = 1.0 + 0.5 * cov + super::test_lcg(&mut st);
        }
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: e_levels,
                    },
                    slopes: vec![],
                }],
            }),
        };
        let ids = crate::GroupIds {
            primary: cl.clone(),
            extra: vec![cr.clone()],
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        let noz = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);
        let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        // Branch sanity: this design builds a sparse tail without any override.
        {
            let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&sized, n, &[], &[vec![]]);
            let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);
            let ws = super::SparseLmmWorkspace::new(
                &g,
                x.as_ref(),
                &cl,
                &cr_as_extra(&cr),
                &y,
                n,
                p,
                None,
            );
            assert!(ws.tail.is_some(), "e=150 routes sparse naturally");
        }
        let sp = super::fit_mle_sparse(
            &xflat,
            &y,
            n,
            p,
            &sized,
            &cl,
            &cr_as_extra(&cr),
            None,
            &opts,
        );
        assert!(noz.converged && sp.converged);
        assert!(
            (sp.deviance - noz.deviance).abs() < 1e-6 * (1.0 + noz.deviance.abs()),
            "deviance sparse {} vs noz {}",
            sp.deviance,
            noz.deviance
        );
        for j in 0..p {
            assert!(
                (sp.beta[j] - noz.beta[j]).abs() < 1e-6,
                "β{j} sparse {} vs noz {}",
                sp.beta[j],
                noz.beta[j]
            );
            assert!(
                (sp.se[j] - noz.se[j]).abs() < 1e-6,
                "se{j} sparse {} vs noz {}",
                sp.se[j],
                noz.se[j]
            );
        }
        for (a, b) in sp
            .varcorr
            .iter()
            .flatten()
            .zip(noz.varcorr.iter().flatten())
        {
            assert!((a - b).abs() < 1e-6, "varcorr {a} vs {b}");
        }
    }

    /// Deterministic builder for one RE-topology case in the cross-check table.
    /// Returns `(xflat, y, n, p, model, ids, opts)`. All designs are
    /// in-envelope (q_p ≤ 8, extras ≤ 6, q_g ≤ 4) so `fit_cold` routes to NoZ
    /// and `fit_mle_sparse` is a valid superset. `test_lcg` seeds are chosen
    /// unique per case so the designs are independent deterministic instances.
    fn build_case(
        label: &str,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
        ModelSpec,
        crate::GroupIds,
        crate::FitOptions,
    ) {
        match label {
            "scalar_intercept_primary" => {
                // (1 | g): intercept-only primary RE, no extras. q_p=1, n_theta=1.
                let n = 24;
                let p = 2;
                let mut xflat = vec![0.0f64; n * p];
                let mut y = vec![0.0f64; n];
                let mut pid = vec![0u32; n];
                let mut st = 13u64;
                for i in 0..n {
                    let cov = super::test_lcg(&mut st);
                    xflat[i * p] = 1.0;
                    xflat[i * p + 1] = cov;
                    pid[i] = (i % 4) as u32;
                    y[i] = 1.0 + 0.5 * cov + super::test_lcg(&mut st);
                }
                let model = ModelSpec {
                    family: Family::Gaussian,
                    re: Some(ReStructure {
                        sizing: Sizing::FixedClusters { n_clusters: 4 },
                        slopes: vec![],
                        extra_groupings: vec![],
                    }),
                };
                let ids = crate::GroupIds {
                    primary: pid,
                    extra: vec![],
                };
                let opts = crate::FitOptions {
                    target_indices: vec![0, 1],
                    ..crate::FitOptions::default()
                };
                (xflat, y, n, p, model, ids, opts)
            }
            "primary_random_slope_q2" => {
                // (1 + x | g): q_p=2 primary with random slope on col 1. The key
                // q_p>1 runtime gate — exercises the non-diagonal primary Λ block
                // in both the dense (reml_deviance_blocked) and sparse paths.
                let n = 24;
                let p = 2;
                let mut xflat = vec![0.0f64; n * p];
                let mut y = vec![0.0f64; n];
                let mut pid = vec![0u32; n];
                let mut st = 17u64;
                for i in 0..n {
                    let cov = super::test_lcg(&mut st);
                    xflat[i * p] = 1.0;
                    xflat[i * p + 1] = cov;
                    pid[i] = (i % 4) as u32;
                    y[i] = 0.5 + 0.3 * cov + super::test_lcg(&mut st);
                }
                let model = ModelSpec {
                    family: Family::Gaussian,
                    re: Some(ReStructure {
                        sizing: Sizing::FixedClusters { n_clusters: 4 },
                        slopes: vec![1], // random slope on col 1; q_p = 2
                        extra_groupings: vec![],
                    }),
                };
                let ids = crate::GroupIds {
                    primary: pid,
                    extra: vec![],
                };
                let opts = crate::FitOptions {
                    target_indices: vec![0, 1],
                    ..crate::FitOptions::default()
                };
                (xflat, y, n, p, model, ids, opts)
            }
            "crossed_two_intercepts" => {
                // (1 | g1) + (1 | g2): primary (3 levels) + one crossed extra (4 levels).
                // Periods 3 and 4 are coprime so every (primary, crossed) cell is
                // populated in n=24 rows (lcm(3,4)=12 → 2 full cycles). n_theta=2.
                let n = 24;
                let p = 2;
                let mut xflat = vec![0.0f64; n * p];
                let mut y = vec![0.0f64; n];
                let mut pid = vec![0u32; n];
                let mut eid = vec![0u32; n];
                let mut st = 23u64;
                for i in 0..n {
                    let cov = super::test_lcg(&mut st);
                    xflat[i * p] = 1.0;
                    xflat[i * p + 1] = cov;
                    pid[i] = (i % 3) as u32;
                    eid[i] = (i % 4) as u32;
                    y[i] = 1.0 + 0.4 * cov + super::test_lcg(&mut st);
                }
                let model = ModelSpec {
                    family: Family::Gaussian,
                    re: Some(ReStructure {
                        sizing: Sizing::FixedClusters { n_clusters: 3 },
                        slopes: vec![],
                        extra_groupings: vec![Grouping {
                            relation: GroupingRelation::Crossed { n_clusters: 4 },
                            slopes: vec![],
                        }],
                    }),
                };
                let ids = crate::GroupIds {
                    primary: pid,
                    extra: vec![eid],
                };
                let opts = crate::FitOptions {
                    target_indices: vec![0, 1],
                    ..crate::FitOptions::default()
                };
                (xflat, y, n, p, model, ids, opts)
            }
            "nested_intercept" => {
                // (1 | g1/g2): primary (4 levels) + nested extra (2 children per
                // parent → 8 global children). Global nested id formula mirrors
                // `ids.rs::extra_level_of_row` for FixedClusters + NestedWithin:
                //   global_id = pid * n_per_parent + (i / n_primary) % n_per_parent
                // giving contiguous [0,2,4,6,1,3,5,7,...] coverage over n=24 rows.
                // k_total = 4 + 4*2 = 12; n_theta = 2 (primary + nested child).
                let n = 24;
                let p = 2;
                let mut xflat = vec![0.0f64; n * p];
                let mut y = vec![0.0f64; n];
                let mut pid = vec![0u32; n];
                let mut cid = vec![0u32; n];
                let mut st = 31u64;
                for i in 0..n {
                    let cov = super::test_lcg(&mut st);
                    xflat[i * p] = 1.0;
                    xflat[i * p + 1] = cov;
                    pid[i] = (i % 4) as u32;
                    // Global nested id: parent·n_per_parent + within_child.
                    cid[i] = pid[i] * 2 + ((i / 4) % 2) as u32;
                    y[i] = 0.8 + 0.3 * cov + super::test_lcg(&mut st);
                }
                let model = ModelSpec {
                    family: Family::Gaussian,
                    re: Some(ReStructure {
                        sizing: Sizing::FixedClusters { n_clusters: 4 },
                        slopes: vec![],
                        extra_groupings: vec![Grouping {
                            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                            slopes: vec![],
                        }],
                    }),
                };
                let ids = crate::GroupIds {
                    primary: pid,
                    extra: vec![cid],
                };
                let opts = crate::FitOptions {
                    target_indices: vec![0, 1],
                    ..crate::FitOptions::default()
                };
                (xflat, y, n, p, model, ids, opts)
            }
            "primary_slope_plus_crossed" => {
                // (1 + x | g1) + (1 | g2): q_p=2 primary + crossed extra (3 levels).
                // Exercises the non-diagonal Λ_p alongside a crossed tail in both paths.
                // k_total = 4*2 + 3 = 11; n_theta = 3 + 1 = 4.
                let n = 24;
                let p = 2;
                let mut xflat = vec![0.0f64; n * p];
                let mut y = vec![0.0f64; n];
                let mut pid = vec![0u32; n];
                let mut eid = vec![0u32; n];
                let mut st = 41u64;
                for i in 0..n {
                    let cov = super::test_lcg(&mut st);
                    xflat[i * p] = 1.0;
                    xflat[i * p + 1] = cov;
                    pid[i] = (i % 4) as u32;
                    eid[i] = (i % 3) as u32;
                    y[i] = 0.6 + 0.4 * cov + super::test_lcg(&mut st);
                }
                let model = ModelSpec {
                    family: Family::Gaussian,
                    re: Some(ReStructure {
                        sizing: Sizing::FixedClusters { n_clusters: 4 },
                        slopes: vec![1], // random slope on col 1; q_p = 2
                        extra_groupings: vec![Grouping {
                            relation: GroupingRelation::Crossed { n_clusters: 3 },
                            slopes: vec![],
                        }],
                    }),
                };
                let ids = crate::GroupIds {
                    primary: pid,
                    extra: vec![eid],
                };
                let opts = crate::FitOptions {
                    target_indices: vec![0, 1],
                    ..crate::FitOptions::default()
                };
                (xflat, y, n, p, model, ids, opts)
            }
            other => panic!("unknown cross-check label: {other}"),
        }
    }

    /// Parses `parity/data_simulated/sim_wide_crossed.csv` into the over-cap
    /// `y ~ 1 + x + (1|g1) + (1|c1)+...+(1|c7)` design — 7 crossed intercept
    /// extras exceed `MAX_EXTRA_GROUPINGS=6`, so `fit_cold` routes to the
    /// sparse-Z path. Shared by the lme4 golden gate and the warm-start A/B
    /// test below. Returns `(x row-major, y, n, p, model, ids)`.
    fn wide_crossed_design() -> (
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
        crate::ModelSpec,
        crate::GroupIds,
    ) {
        let csv = include_str!("../parity/data_simulated/sim_wide_crossed.csv");
        let mut y = Vec::<f64>::new();
        let mut xcol = Vec::<f64>::new();
        let mut g1_raw = Vec::<String>::new();
        let mut c1_raw = Vec::<String>::new();
        let mut c2_raw = Vec::<String>::new();
        let mut c3_raw = Vec::<String>::new();
        let mut c4_raw = Vec::<String>::new();
        let mut c5_raw = Vec::<String>::new();
        let mut c6_raw = Vec::<String>::new();
        let mut c7_raw = Vec::<String>::new();
        // Columns: y, x, g1, c1, c2, c3, c4, c5, c6, c7 (indices 0..9).
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            xcol.push(f[1].parse().unwrap());
            g1_raw.push(f[2].to_string());
            c1_raw.push(f[3].to_string());
            c2_raw.push(f[4].to_string());
            c3_raw.push(f[5].to_string());
            c4_raw.push(f[6].to_string());
            c5_raw.push(f[7].to_string());
            c6_raw.push(f[8].to_string());
            c7_raw.push(f[9].to_string());
        }
        let n = y.len();
        let p = 2;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            x[i * p + 1] = xcol[i];
        }

        // Map string factor labels to dense 0-based ids (first-seen order).
        // Inner fn mirrors `dense_str` from fit.rs test module — same pattern.
        fn dense_str(raw: &[String]) -> (Vec<u32>, usize) {
            use std::collections::HashMap;
            let mut map: HashMap<String, u32> = HashMap::new();
            let mut next = 0u32;
            let ids: Vec<u32> = raw
                .iter()
                .map(|r| {
                    *map.entry(r.clone()).or_insert_with(|| {
                        let v = next;
                        next += 1;
                        v
                    })
                })
                .collect();
            (ids, next as usize)
        }
        let (g1, _) = dense_str(&g1_raw);
        let (c1, _) = dense_str(&c1_raw);
        let (c2, _) = dense_str(&c2_raw);
        let (c3, _) = dense_str(&c3_raw);
        let (c4, _) = dense_str(&c4_raw);
        let (c5, _) = dense_str(&c5_raw);
        let (c6, _) = dense_str(&c6_raw);
        let (c7, _) = dense_str(&c7_raw);

        // n_clusters: 1 placeholders — fit_cold derives true sizes from ids via
        // spec_sized_from_ids. Topology and family are preserved.
        let model = crate::ModelSpec {
            family: crate::Family::Gaussian,
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters { n_clusters: 1 },
                slopes: vec![],
                extra_groupings: vec![
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    }, // c1
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    }, // c2
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    }, // c3
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    }, // c4
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    }, // c5
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    }, // c6
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    }, // c7
                ],
            }),
        };
        let ids = crate::GroupIds {
            primary: g1,
            extra: vec![c1, c2, c3, c4, c5, c6, c7],
        };
        (x, y, n, p, model, ids)
    }

    /// OVER-CAP lme4 REML golden on the `wide_crossed_design` above. Gated
    /// against the frozen lme4 1.1.38 REML golden
    /// (`parity/goldens/sim_wide_crossed_lmm.json`). The oracle is sacred.
    /// Tolerances: β/SE 2e-2 relative, varcomp stddev 3e-2 relative (phase-1 band).
    #[test]
    fn fit_wide_crossed_sparse_matches_lme4() {
        // Serde structs mirror the fit.rs VcBlock/VcEst/VcGolden pattern; `se`
        // added here for the SE gate (LMM golden has `estimates.se` directly).
        // serde ignores unread JSON fields (e.g. `group`, `corr`) by default.
        #[derive(serde::Deserialize)]
        struct VcBlock {
            stddev: Vec<f64>,
        }
        #[derive(serde::Deserialize)]
        struct VcEst {
            beta: Vec<f64>,
            se: Vec<f64>,
            varcomp: Vec<VcBlock>,
        }
        #[derive(serde::Deserialize)]
        struct VcGolden {
            estimates: VcEst,
        }

        let raw = include_str!("../parity/goldens/sim_wide_crossed_lmm.json");
        let gold: VcGolden = serde_json::from_str(raw).expect("golden JSON parses");

        let (x, y, n, p, model, ids) = wide_crossed_design();
        // 7 extras > MAX_EXTRA_GROUPINGS=6 → over-envelope-by-count ⇒ Sparse.
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse,
        ));
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

        assert!(f.converged, "sparse wide-crossed fit must converge");
        // β/SE: 2e-2 relative (phase-1 band). Intercept (j=0) and slope (j=1).
        for j in 0..p {
            let rb = gold.estimates.beta[j];
            let rs = gold.estimates.se[j];
            assert!(
                (f.beta[j] - rb).abs() / rb.abs().max(1e-6) < 2e-2,
                "β[{j}] glmm={} lme4={rb}",
                f.beta[j],
            );
            assert!(
                (f.se[j] - rs).abs() / rs.abs().max(1e-6) < 2e-2,
                "se[{j}] glmm={} lme4={rs}",
                f.se[j],
            );
        }
        // Varcomp stddev: 3e-2 relative (phase-1 band). 8 scalar blocks (q=1):
        // varcorr[0]=g1, [1]=c1, ..., [7]=c7 — primary first, extras in formula order.
        assert_eq!(f.varcorr.len(), 8, "8 scalar varcomp blocks (g1 + c1..c7)");
        for k in 0..8 {
            let ref_sd = gold.estimates.varcomp[k].stddev[0];
            let got_sd = f.varcorr[k][0].sqrt();
            assert!(
                (got_sd - ref_sd).abs() / ref_sd.max(1e-6) < 3e-2,
                "varcomp[{k}] stddev glmm={got_sd:.6} lme4={ref_sd:.6}",
            );
        }
    }

    /// Warm-start A/B on the sparse-routed wide-crossed LMM: a warm fit from
    /// the frozen lme4 θ̂ ("from the truth" — `Fit` doesn't expose θ̂, and
    /// Gaussian tau2/varcorr are both σ²-scaled so θ can't be recovered from
    /// the cold fit; θ̂_k = stddev_k/σ̂ from the golden) and one from a
    /// perturbed θ must land on the cold optimum — β, SE, varcomp stddevs —
    /// and must never degrade convergence status. The sparse sibling of the
    /// fit.rs `fit_warm_*_matches_cold_optimum` pair; β start is irrelevant on
    /// the LMM path (β is solved exactly given θ).
    #[test]
    fn fit_warm_sparse_wide_crossed_matches_cold_optimum() {
        let (x, y, n, p, model, ids) = wide_crossed_design();
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse,
        ));
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        let cold = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(cold.converged, "cold sparse wide-crossed fit must converge");

        // Frozen lme4 golden (sim_wide_crossed_lmm.json): per-grouping stddev
        // in glmm declaration order [g1, c1..c7], and σ̂; θ̂_k = stddev_k/σ̂
        // (scalar blocks).
        const REF_SD: [f64; 8] = [
            0.95374359126349,
            1.00779577183308,
            0.560396926386321,
            0.624586780829176,
            0.596163063210671,
            0.675316308597726,
            0.609496256365909,
            0.437601153218947,
        ];
        const REF_SIGMA: f64 = 0.619378289188346;
        let truth: Vec<f64> = REF_SD.iter().map(|sd| sd / REF_SIGMA).collect();
        let starts = [
            (
                "truth",
                crate::StartValues {
                    beta: cold.beta.clone(),
                    theta: truth,
                },
            ),
            // θ̂ spans ≈ [0.7, 1.6]; 3.0 everywhere is well off every coordinate.
            (
                "perturbed",
                crate::StartValues {
                    beta: vec![0.0; p],
                    theta: vec![3.0; 8],
                },
            ),
        ];
        for (label, start) in &starts {
            let warm = crate::fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
            assert!(warm.converged, "{label}: warm must not degrade convergence");
            for j in 0..p {
                let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
                assert!(
                    rel < 1e-3,
                    "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                    warm.beta[j],
                    cold.beta[j]
                );
                let rel = (warm.se[j] - cold.se[j]).abs() / cold.se[j];
                assert!(
                    rel < 1e-3,
                    "{label}: se[{j}] warm {} vs cold {} (rel {rel})",
                    warm.se[j],
                    cold.se[j]
                );
            }
            // 8 scalar varcomp blocks [g1, c1..c7].
            for k in 0..8 {
                let (w, c) = (warm.varcorr[k][0].sqrt(), cold.varcorr[k][0].sqrt());
                let rel = (w - c).abs() / c;
                assert!(
                    rel < 1e-3,
                    "{label}: varcomp[{k}] stddev warm {w} vs cold {c} (rel {rel})"
                );
            }
        }
    }

    /// Cross-check: force Sparse on in-envelope designs and diff every
    /// output against NoZ. A mismatch is a bug in exactly one path (NoZ is the
    /// oracle). Spans the five RE-topology axes: scalar-intercept, primary slope
    /// (q_p=2 runtime gate), crossed, nested, slope+crossed. Run for both tail
    /// branches — the forced-sparse pass covers the non-diagonal-Λ crossed
    /// tail (`primary_slope_plus_crossed`) through the fill-reducing factor;
    /// the e=0 topologies are unaffected by the flag (no tail exists).
    #[test]
    fn sparse_vs_noz_cross_check_table() {
        run_sparse_vs_noz_cross_check_table();
    }
    #[test]
    fn sparse_vs_noz_cross_check_table_sparse_tail() {
        with_forced_sparse_tail(run_sparse_vs_noz_cross_check_table);
    }
    fn run_sparse_vs_noz_cross_check_table() {
        let cases: &[&str] = &[
            "scalar_intercept_primary",   // (1 | g)
            "primary_random_slope_q2",    // (1 + x | g), q_p=2 runtime gate
            "crossed_two_intercepts",     // (1 | g1) + (1 | g2)
            "nested_intercept",           // (1 | g1/g2)
            "primary_slope_plus_crossed", // (1 + x | g1) + (1 | g2)
        ];
        for label in cases {
            let (xflat, y, n, p, model, ids, opts) = build_case(label);
            let noz = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);
            let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
            let sp = super::fit_mle_sparse(
                &xflat,
                &y,
                n,
                p,
                &sized,
                &ids.primary,
                &ids.extra,
                None,
                &opts,
            );
            assert!(
                noz.converged && sp.converged,
                "{label}: both paths must converge"
            );
            for j in 0..p {
                assert!(
                    (sp.beta[j] - noz.beta[j]).abs() < 1e-6,
                    "{label} β[{j}]: sparse={} noz={}",
                    sp.beta[j],
                    noz.beta[j]
                );
                assert!(
                    (sp.se[j] - noz.se[j]).abs() < 1e-6,
                    "{label} se[{j}]: sparse={} noz={}",
                    sp.se[j],
                    noz.se[j]
                );
            }
            assert_eq!(
                sp.varcorr.len(),
                noz.varcorr.len(),
                "{label}: varcorr block count"
            );
            for (a, b) in sp
                .varcorr
                .iter()
                .flatten()
                .zip(noz.varcorr.iter().flatten())
            {
                assert!((a - b).abs() < 1e-6, "{label} varcorr: sparse={a} noz={b}");
            }
        }
    }

    // ── NoZ↔Sparse crossover grid ──────────────────────────────
    // Generalizes `build_case`/`sparse_vs_noz_cross_check_table` from 5 hand-
    // built topologies to a programmatic sweep of the whole NoZ overlap
    // envelope, shared by the accuracy gate (`noz_sparse_grid_agrees`) and the
    // timed sweep (`noz_sparse_crossover_timed`).

    /// One cell of the crossover grid. All structural cells sit inside the NoZ
    /// overlap envelope (`q_p ≤ MAX_PRIMARY_Q`, `n_extra ≤ MAX_EXTRA_GROUPINGS`,
    /// `q_g ≤ MAX_EXTRA_Q`) so both kernels are valid on every cell; each side
    /// is forced directly (`fit_mle_noz_pub` / `fit_mle_sparse`), bypassing
    /// `classify_design` — whose q_g performance boundary these sweeps set.
    /// Widths include the intercept: `q_p = 1 + primary slopes`,
    /// `q_g = 1 + per-extra slopes`.
    #[derive(Clone, Copy)]
    struct GridCell {
        /// Rows. Structural cells carry the timing size
        /// (`TIMING_ROWS_PER_RE_COL · re_cols`); the accuracy gate overrides it
        /// with the smaller `ACCURACY_ROWS_PER_RE_COL` sizing. The timed
        /// N-control cells carry their swept value as-is.
        n: usize,
        n_primary: usize,
        q_p: usize,
        n_extra: usize,
        q_g: usize,
    }

    /// Estimability safety factor `k` in `n ≥ k · total_re_cols`.
    /// 4 rows per RE column keeps every cell non-singular (the existing
    /// hand-built cases run at ~2.2) while keeping the sweeps cheap. Timing
    /// uses the same sizing: the structural cells measure shape, not N — the
    /// N-control slice verifies N cancels in the NoZ/Sparse ratio.
    const ACCURACY_ROWS_PER_RE_COL: usize = 4;
    const TIMING_ROWS_PER_RE_COL: usize = 4;

    /// Level count of extra grouping `g`. Distinct per grouping (5, 6, 7, …)
    /// so no two crossed factors are identical or mutually confounded.
    fn extra_levels(g: usize) -> usize {
        5 + g
    }

    /// Total random-effect columns of a cell — the estimability driver.
    fn re_cols(c: &GridCell) -> usize {
        c.n_primary * c.q_p
            + (0..c.n_extra)
                .map(|g| extra_levels(g) * c.q_g)
                .sum::<usize>()
    }

    /// Cells too slow for the default `cargo test` gate, run only under the
    /// `loop_advanced` feature. Cut from the 2026-07-01 release
    /// calibration timings: these eight cells cost 27–242s each in release
    /// (wide-θ BOBYQA — q_g=4 puts 10 vech entries per extra grouping — or
    /// big dense primary patches); the remaining 13 cells total ~4s release.
    /// The default subset still spans every axis endpoint except q_g=4, which
    /// stays covered by this gate under `loop_advanced` and by the over-width
    /// lme4 golden (`fit_wide_slopes_sparse_matches_lme4`, q_g=5) vs Sparse.
    fn is_heavy_cell(c: &GridCell) -> bool {
        c.q_g >= 4 || (c.q_p >= 4 && c.n_primary >= 200) || (c.q_p >= 6 && c.n_primary >= 50)
    }

    /// The structural grid: a 2D `q_p × n_primary` patch
    /// (`n_extra = 0`) catching the interaction pure OAT would miss, plus a
    /// crossed slice (`n_extra × q_g` at `q_p=2, n_primary=50`). 21 cells; the
    /// N-control slice lives in the timed test only (it adds no structure).
    fn crossover_structures() -> Vec<GridCell> {
        let mut cells = Vec::new();
        for &q_p in &[1usize, 2, 4, 6, 8] {
            for &n_primary in &[10usize, 50, 200] {
                cells.push(GridCell {
                    n: 0,
                    n_primary,
                    q_p,
                    n_extra: 0,
                    q_g: 1,
                });
            }
        }
        for &q_g in &[1usize, 4] {
            for &n_extra in &[2usize, 4, 6] {
                cells.push(GridCell {
                    n: 0,
                    n_primary: 50,
                    q_p: 2,
                    n_extra,
                    q_g,
                });
            }
        }
        for c in cells.iter_mut() {
            c.n = TIMING_ROWS_PER_RE_COL * re_cols(c);
        }
        cells
    }

    /// Parametric generalization of `build_case`: one deterministic synthetic
    /// design per cell. With `w = max(q_p, q_g)` the fixed design is
    /// `[1, cov₁…cov_{w−1}]` (`p = w`), so slope indices `1..q_p` / `1..q_g`
    /// always reference existing columns. True per-level effects (primary
    /// intercept+slopes, extras likewise, amplitude 0.5·LCG) are injected so
    /// fitted variance components sit off the θ = 0 boundary. Covariates,
    /// effects, and noise all come from `test_lcg(seed)` — unique seed per
    /// cell, no wall clock, no `rand`.
    fn build_grid_case(
        cell: &GridCell,
        seed: u64,
    ) -> (
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
        ModelSpec,
        crate::GroupIds,
        crate::FitOptions,
    ) {
        let GridCell {
            n,
            n_primary,
            q_p,
            n_extra,
            q_g,
        } = *cell;
        let p = q_p.max(q_g);
        let mut st = seed;

        let pid: Vec<u32> = (0..n).map(|i| (i % n_primary) as u32).collect();
        // Extra ids stride by g+1 so each grouping's level pattern differs from
        // the others' and from the primary's `i % n_primary`; integer division
        // still visits every level for n ≫ (g+1)·levels.
        let extra: Vec<Vec<u32>> = (0..n_extra)
            .map(|g| {
                (0..n)
                    .map(|i| ((i / (g + 1)) % extra_levels(g)) as u32)
                    .collect()
            })
            .collect();

        let prim_eff: Vec<f64> = (0..n_primary * q_p)
            .map(|_| 0.5 * super::test_lcg(&mut st))
            .collect();
        let extra_eff: Vec<Vec<f64>> = (0..n_extra)
            .map(|g| {
                (0..extra_levels(g) * q_g)
                    .map(|_| 0.5 * super::test_lcg(&mut st))
                    .collect()
            })
            .collect();

        let mut xflat = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            xflat[i * p] = 1.0;
            for j in 1..p {
                xflat[i * p + j] = super::test_lcg(&mut st);
            }
            let mut mu = 1.0;
            for j in 1..p {
                mu += 0.5 * xflat[i * p + j];
            }
            let c = pid[i] as usize;
            mu += prim_eff[c * q_p];
            for k in 1..q_p {
                mu += prim_eff[c * q_p + k] * xflat[i * p + k];
            }
            for g in 0..n_extra {
                let l = extra[g][i] as usize;
                mu += extra_eff[g][l * q_g];
                for k in 1..q_g {
                    mu += extra_eff[g][l * q_g + k] * xflat[i * p + k];
                }
            }
            y[i] = mu + super::test_lcg(&mut st);
        }

        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_primary as u32,
                },
                slopes: (1..q_p as u32).collect(),
                extra_groupings: (0..n_extra)
                    .map(|g| Grouping {
                        relation: GroupingRelation::Crossed {
                            n_clusters: extra_levels(g) as u32,
                        },
                        slopes: (1..q_g as u32).collect(),
                    })
                    .collect(),
            }),
        };
        let ids = crate::GroupIds {
            primary: pid,
            extra,
        };
        let opts = crate::FitOptions {
            target_indices: (0..p as u32).collect(),
            ..crate::FitOptions::default()
        };
        (xflat, y, n, p, model, ids, opts)
    }

    /// Accuracy gate: NoZ = Sparse across the whole overlap envelope.
    /// A disagreement is a bug in exactly one path, never a tuning knob (NoZ is
    /// the oracle for the overlap; Sparse is separately anchored to lme4 by the
    /// over-cap goldens). Relative bound `|Δ| ≤ TOL·(1 + |ref|)` because the
    /// two paths are not identical arithmetic: the deviances agree only to
    /// rel ~1e-8 (`sparse_deviance_matches_dense_lmm`) and each side is its own
    /// BOBYQA minimization, so θ* can legitimately differ slightly. TOL may be
    /// loosened later ONLY with a documented numerical reason.
    ///
    /// The default gate runs the 15 cheap cells (13 grid + the two appended
    /// q_g ∈ {2,3} boundary cells); the 8 heavy ones (`is_heavy_cell`) run
    /// under `--features loop_advanced`, where the whole envelope is swept.
    #[test]
    fn noz_sparse_grid_agrees() {
        // Frozen after the 2026-07-01 full-grid calibration run (release):
        // observed max rel |Δ| = 2.37e-5, at the q_g=4 crossed cells whose
        // θ-space is 23–63-dimensional. That exceeds the 1e-6 starting bound,
        // so the worst cell was investigated before loosening:
        // `crossover_worst_cell_deviance_parity` shows dense and sparse
        // deviance agree there to rel ~1e-15 at arbitrary θ, so the gap is
        // purely BOBYQA termination scatter between two independent
        // high-dimensional minimizations — not a path bug. 1e-4 gives ~4×
        // margin over the observed max. Loosen further ONLY with a comparable
        // documented numerical reason.
        const TOL: f64 = 1e-4;
        let mut max_rel = 0f64;
        let mut worst = String::new();
        let mut cells = crossover_structures();
        // q_g ∈ {2,3} coverage at the routing boundary: these widths now route
        // to Sparse (`classify_design`'s slope-extra clause), so NoZ=Sparse
        // parity must hold here too. Appended (indices 21–22) so the original
        // 21 cells keep their seed-bound indices; n_extra=2 keeps both cheap
        // enough (~4 s release combined) for the default gate.
        for &q_g in &[2usize, 3] {
            cells.push(GridCell {
                n: 0,
                n_primary: 50,
                q_p: 2,
                n_extra: 2,
                q_g,
            });
        }
        let mut skipped = 0usize;
        let mut checked = 0usize;
        for (idx, c) in cells.iter().enumerate() {
            if is_heavy_cell(c) && !cfg!(feature = "loop_advanced") {
                skipped += 1;
                continue;
            }
            checked += 1;
            let cell = GridCell {
                n: ACCURACY_ROWS_PER_RE_COL * re_cols(c),
                ..*c
            };
            let (xflat, y, n, p, model, ids, opts) =
                build_grid_case(&cell, 0x5eed_0000 + idx as u64);
            let t0 = std::time::Instant::now();
            let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
            let noz = crate::fit::fit_mle_noz_pub(
                &xflat,
                &y,
                n,
                p,
                &sized,
                &ids.primary,
                &ids.extra,
                None,
                &opts,
            );
            let sp = super::fit_mle_sparse(
                &xflat,
                &y,
                n,
                p,
                &sized,
                &ids.primary,
                &ids.extra,
                None,
                &opts,
            );
            eprintln!(
                "cell {idx}: n={n} n_primary={} q_p={} n_extra={} q_g={} — {:.1}s",
                cell.n_primary,
                cell.q_p,
                cell.n_extra,
                cell.q_g,
                t0.elapsed().as_secs_f64()
            );
            let tag = format!(
                "cell {idx} (n={n}, n_primary={}, q_p={}, n_extra={}, q_g={})",
                cell.n_primary, cell.q_p, cell.n_extra, cell.q_g
            );
            assert!(
                noz.converged && sp.converged,
                "{tag}: both paths must converge"
            );
            let mut check = |a: f64, b: f64, what: String| {
                let rel = (a - b).abs() / (1.0 + b.abs());
                if rel > max_rel {
                    max_rel = rel;
                    worst = format!("{tag} {what}");
                }
                assert!(rel <= TOL, "{tag} {what}: sparse={a} noz={b} rel={rel:.3e}");
            };
            for j in 0..p {
                check(sp.beta[j], noz.beta[j], format!("β[{j}]"));
                check(sp.se[j], noz.se[j], format!("se[{j}]"));
            }
            assert_eq!(
                sp.varcorr.len(),
                noz.varcorr.len(),
                "{tag}: varcorr block count"
            );
            for (bi, (sb, nb)) in sp.varcorr.iter().zip(noz.varcorr.iter()).enumerate() {
                assert_eq!(sb.len(), nb.len(), "{tag}: varcorr[{bi}] len");
                for (ei, (a, b)) in sb.iter().zip(nb.iter()).enumerate() {
                    check(*a, *b, format!("varcorr[{bi}][{ei}]"));
                }
            }
        }
        // Report the real margin on success, not just "under the bar".
        eprintln!(
            "noz_sparse_grid_agrees: {checked} cells checked ({skipped} heavy cells \
             need --features loop_advanced), max rel |Δ| = {max_rel:.3e} at {worst}"
        );
    }

    /// Deviance-level parity on the grid's worst-disagreeing cell (q_p=2,
    /// n_extra=2, q_g=4 — the max-|Δ| cell of the 2026-07-01 calibration run).
    /// Dense and sparse deviance agree here to rel ~1e-15 at arbitrary θ, which
    /// is the evidence behind `noz_sparse_grid_agrees`' frozen 1e-4 tolerance:
    /// the fit-level gap on this cell (2.4e-5) is BOBYQA termination scatter,
    /// not a path bug. Cheap (8 deviance evals, no optimization) — stays in the
    /// default gate as the standing witness for that calibration argument.
    #[test]
    fn crossover_worst_cell_deviance_parity() {
        use faer::Mat;
        let c = GridCell {
            n: 0,
            n_primary: 50,
            q_p: 2,
            n_extra: 2,
            q_g: 4,
        };
        let cell = GridCell {
            n: ACCURACY_ROWS_PER_RE_COL * re_cols(&c),
            ..c
        };
        let (xflat, y, n, p, model, ids, _opts) = build_grid_case(&cell, 0x5eed_0000 + 18);
        let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);
        let prim_slopes: Vec<usize> = (1..cell.q_p).collect();
        let extra_slopes: Vec<Vec<usize>> =
            (0..cell.n_extra).map(|_| (1..cell.q_g).collect()).collect();
        let g =
            crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &prim_slopes, &extra_slopes);
        let mut suff = crate::lmm::LmmSuffStats::with_groupings(p, g.clone());
        suff.add_rows_multi(x.as_ref(), &y, &ids.primary, &ids.extra, None);
        let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);
        let mut ws = super::SparseLmmWorkspace::new(
            &g,
            x.as_ref(),
            &ids.primary,
            &ids.extra,
            &y,
            n,
            p,
            None,
        );
        let n_theta = 3 + 2 * 10; // vech(2x2) + 2·vech(4x4)
        let mut st = 99u64;
        let mut max_rel = 0.0f64;
        for t in 0..8 {
            let theta: Vec<f64> = (0..n_theta)
                .map(|_| 0.3 + 0.6 * super::test_lcg(&mut st))
                .collect();
            let dense = crate::lmm::reml_deviance(&theta, &suff, &mut fit);
            let sparse = super::sparse_reml_deviance(&theta, &mut ws);
            let rel = (dense - sparse).abs() / (1.0 + dense.abs());
            eprintln!("θ set {t}: dense={dense:.12e} sparse={sparse:.12e} rel={rel:.3e}");
            max_rel = max_rel.max(rel);
        }
        eprintln!("worst-cell deviance parity: max rel = {max_rel:.3e}");
        assert!(
            max_rel < 1e-8,
            "deviance functions disagree — real path bug"
        );
    }

    /// Min elapsed µs over an adaptive rep count: one probe call (discarded —
    /// it is the cold pass: cache + frequency ramp) sets
    /// `reps ≈ target_loop_s / t_probe`, clamped to [1, 30]; the reported min
    /// is over the following warm calls. Min because timing noise is one-sided
    /// — interference only ever slows a run. Floor 1: it only engages on fits
    /// slower than the loop budget (seconds+), where interference is
    /// proportionally negligible and more reps would cost minutes per cell.
    fn min_time_us<F: FnMut()>(target_loop_s: f64, mut f: F) -> f64 {
        let t0 = std::time::Instant::now();
        f();
        let probe_s = t0.elapsed().as_secs_f64();
        let reps = ((target_loop_s / probe_s.max(1e-9)) as usize).clamp(1, 30);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            f();
            let dt = t0.elapsed().as_secs_f64() * 1e6;
            if dt < best {
                best = dt;
            }
        }
        best
    }

    /// Machine-state guard: read (never write) the pstate/governor sysfs
    /// and report LOCKED/UNLOCKED. A run whose header does not say LOCKED is
    /// noise — do not record it.
    fn machine_lock_header() {
        let read = |path: &str| {
            std::fs::read_to_string(path)
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| "<unreadable>".into())
        };
        let no_turbo = read("/sys/devices/system/cpu/intel_pstate/no_turbo");
        let gov = read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
        let state = if no_turbo == "1" && gov == "performance" {
            "LOCKED"
        } else {
            "UNLOCKED"
        };
        println!("machine: no_turbo={no_turbo} cpu0_governor={gov} -> {state}");
    }

    /// The two cells whose single fits run ~100 s (43/63-dim BOBYQA at
    /// q_g=4, n_extra ≥ 4): ~15 min of sweep on their own, so they live in
    /// `noz_sparse_crossover_heavy_timed` and the main sweep stays ~5 min.
    fn is_ultra_heavy_cell(c: &GridCell) -> bool {
        c.q_g >= 4 && c.n_extra >= 4
    }

    /// Shared driver for the timed sweeps: machine header, then one table row
    /// per cell — min-of-N µs per path (adaptive N, probe pass discarded,
    /// design built outside the timed region).
    fn run_timed_sweep(cells: &[GridCell]) {
        // Rep budget per (cell, path): reps ≈ TARGET_LOOP_S / t_probe,
        // clamped to [1, 30].
        const TARGET_LOOP_S: f64 = 2.0;
        machine_lock_header();
        println!(
            "{:>6} {:>9} {:>4} {:>7} {:>4} {:>12} {:>12} {:>7}  winner",
            "N", "n_prim", "q_p", "n_extra", "q_g", "t_noz_us", "t_sparse_us", "ratio"
        );
        for (idx, cell) in cells.iter().enumerate() {
            let (xflat, y, n, p, model, ids, opts) =
                build_grid_case(cell, 0x71ED_0000 + idx as u64);
            let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
            let t_noz = min_time_us(TARGET_LOOP_S, || {
                std::hint::black_box(crate::fit::fit_mle_noz_pub(
                    &xflat,
                    &y,
                    n,
                    p,
                    &sized,
                    &ids.primary,
                    &ids.extra,
                    None,
                    &opts,
                ));
            });
            let t_sparse = min_time_us(TARGET_LOOP_S, || {
                std::hint::black_box(super::fit_mle_sparse(
                    &xflat,
                    &y,
                    n,
                    p,
                    &sized,
                    &ids.primary,
                    &ids.extra,
                    None,
                    &opts,
                ));
            });
            let ratio = t_sparse / t_noz;
            let winner = if t_noz <= t_sparse { "NoZ" } else { "Sparse" };
            println!(
                "{:>6} {:>9} {:>4} {:>7} {:>4} {:>12.1} {:>12.1} {:>7.2}  {}",
                n, cell.n_primary, cell.q_p, cell.n_extra, cell.q_g, t_noz, t_sparse, ratio, winner
            );
        }
    }

    /// Timed crossover sweep, main slice (~5 min): all structural cells
    /// except the two ultra-heavy ones, plus the N-control slice.
    /// `#[ignore]`d — run explicitly, only
    /// after the machine is locked (user's call), pinned to one P-core:
    ///
    /// ```sh
    /// taskset -c 0 cargo test --release noz_sparse_crossover_timed -- \
    ///     --ignored --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "timed sweep — run pinned on a user-locked machine (see doc-comment)"]
    fn noz_sparse_crossover_timed() {
        let mut cells: Vec<GridCell> = crossover_structures()
            .into_iter()
            .filter(|c| !is_ultra_heavy_cell(c))
            .collect();
        // N control at the baseline structure (q_p=2, n_primary=50, n_extra=0):
        // both paths pay the same shared suff-stat accumulation, so N should
        // cancel in the ratio — this slice verifies that.
        for &n in &[500usize, 2000, 8000] {
            cells.push(GridCell {
                n,
                n_primary: 50,
                q_p: 2,
                n_extra: 0,
                q_g: 1,
            });
        }
        run_timed_sweep(&cells);
    }

    /// Follow-up sweep tightening the q_g crossover locus: the main grid swept
    /// `q_g ∈ {1, 4}` and found NoZ↔Sparse flips between them, so this measures
    /// the two skipped widths at the same crossed slice (`q_p=2, n_primary=50`,
    /// `n_extra ∈ {2,4,6}`). Own cell list — `crossover_structures()` stays
    /// untouched because its cell indices are seed-bound and cited by the
    /// calibration comments. Same invocation as the main sweep:
    ///
    /// ```sh
    /// taskset -c 0 cargo test --release noz_sparse_crossover_qg23_timed -- \
    ///     --ignored --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "timed sweep (q_g ∈ {2,3}) — run pinned on a user-locked machine"]
    fn noz_sparse_crossover_qg23_timed() {
        let mut cells = Vec::new();
        for &q_g in &[2usize, 3] {
            for &n_extra in &[2usize, 4, 6] {
                cells.push(GridCell {
                    n: 0,
                    n_primary: 50,
                    q_p: 2,
                    n_extra,
                    q_g,
                });
            }
        }
        for c in cells.iter_mut() {
            c.n = TIMING_ROWS_PER_RE_COL * re_cols(c);
        }
        run_timed_sweep(&cells);
    }

    /// Timed crossover sweep, ultra-heavy slice (~15 min: two cells whose
    /// single fits run ~100 s). Named so neither sweep's filter substring-matches
    /// the other. Same invocation as the main sweep, run it when the q_g=4
    /// crossed corner matters:
    ///
    /// ```sh
    /// taskset -c 0 cargo test --release noz_sparse_crossover_heavy_timed -- \
    ///     --ignored --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "timed sweep (ultra-heavy cells) — run pinned on a user-locked machine"]
    fn noz_sparse_crossover_heavy_timed() {
        let cells: Vec<GridCell> = crossover_structures()
            .into_iter()
            .filter(is_ultra_heavy_cell)
            .collect();
        run_timed_sweep(&cells);
    }

    /// OVER-WIDTH lme4 REML golden: `y ~ 1 + x1+x2+x3+x4 + (1|gp) + (1 + x1+x2+x3+x4 | ge)`.
    /// `ge` carries q_g = 5 (intercept + 4 slopes) > `MAX_EXTRA_Q = 4`, so the design is
    /// over-envelope by a single grouping's WIDTH — the one over-cap axis with NO dense
    /// (NoZ) twin, since NoZ physically cannot run q_g>4. The sparse-vs-NoZ cross-check
    /// (`sparse_vs_noz_cross_check_table`) therefore never reaches q_g>1, so this golden
    /// is the *sole* oracle for the extras LEVEL-MAJOR multi-slope Z column layout at
    /// width 5: a wrong layout corrupts the whole fit, so matching lme4 on β + SE +
    /// per-term varcomp (the true SDs are distinct, so a column permutation shifts the
    /// diagonal variances and fails) validates the layout end-to-end. The oracle is
    /// sacred — tolerances are the phase-1 band (β/SE 2e-2, varcomp stddev 3e-2 rel).
    #[test]
    fn fit_wide_slopes_sparse_matches_lme4() {
        #[derive(serde::Deserialize)]
        struct VcBlock {
            group: String,
            stddev: Vec<f64>,
        }
        #[derive(serde::Deserialize)]
        struct VcEst {
            beta: Vec<f64>,
            se: Vec<f64>,
            varcomp: Vec<VcBlock>,
        }
        #[derive(serde::Deserialize)]
        struct VcGolden {
            estimates: VcEst,
        }

        let raw = include_str!("../parity/goldens/sim_wide_slopes_lmm.json");
        let gold: VcGolden = serde_json::from_str(raw).expect("golden JSON parses");

        let csv = include_str!("../parity/data_simulated/sim_wide_slopes.csv");
        // Columns: y, x1, x2, x3, x4, gp, ge (indices 0..7).
        let mut y = Vec::<f64>::new();
        let mut xc: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
        let mut gp_raw = Vec::<String>::new();
        let mut ge_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            for k in 0..4 {
                xc[k].push(f[1 + k].parse().unwrap());
            }
            gp_raw.push(f[5].to_string());
            ge_raw.push(f[6].to_string());
        }
        let n = y.len();
        let p = 5; // intercept + x1..x4
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            for k in 0..4 {
                x[i * p + 1 + k] = xc[k][i];
            }
        }

        // Map string factor labels to dense 0-based ids (first-seen order); mirrors
        // `dense_str` in the wide-crossed golden test above.
        fn dense_str(raw: &[String]) -> Vec<u32> {
            use std::collections::HashMap;
            let mut map: HashMap<String, u32> = HashMap::new();
            let mut next = 0u32;
            raw.iter()
                .map(|r| {
                    *map.entry(r.clone()).or_insert_with(|| {
                        let v = next;
                        next += 1;
                        v
                    })
                })
                .collect()
        }
        let gp = dense_str(&gp_raw);
        let ge = dense_str(&ge_raw);

        // primary gp intercept-only (q_p=1); extra ge with slopes on x1..x4 (q_g=5).
        // n_clusters: 1 placeholders — fit_cold derives true sizes from ids.
        let model = crate::ModelSpec {
            family: crate::Family::Gaussian,
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters { n_clusters: 1 },
                slopes: vec![],
                extra_groupings: vec![crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q=4 ⇒ over-WIDTH
                }],
            }),
        };
        // q_g=5 over the NoZ envelope WIDTH ⇒ Sparse (over-width, no NoZ twin).
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse,
        ));
        let ids = crate::GroupIds {
            primary: gp,
            extra: vec![ge],
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1, 2, 3, 4],
            ..crate::FitOptions::default()
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

        assert!(f.converged, "sparse over-width fit must converge");
        // β/SE: 2e-2 relative. Golden coef order (Intercept,x1,x2,x3,x4) == x col order.
        for j in 0..p {
            let rb = gold.estimates.beta[j];
            let rs = gold.estimates.se[j];
            assert!(
                (f.beta[j] - rb).abs() / rb.abs().max(1e-6) < 2e-2,
                "β[{j}] glmm={} lme4={rb}",
                f.beta[j],
            );
            assert!(
                (f.se[j] - rs).abs() / rs.abs().max(1e-6) < 2e-2,
                "se[{j}] glmm={} lme4={rs}",
                f.se[j],
            );
        }

        // Varcomp: glmm order is declaration order [gp(primary,q=1), ge(extra,q=5)];
        // lme4's VarCorr order is descending level count [ge(40), gp(20)] — so map by
        // group NAME, not index. glmm.varcorr[k] is the column-major lower-tri vech of
        // D̂=σ̂²Λ̂Λ̂'; per-term variances sit at the diagonal offsets below.
        assert_eq!(f.varcorr.len(), 2, "two varcomp blocks (gp + ge)");
        let gold_of = |name: &str| {
            gold.estimates
                .varcomp
                .iter()
                .find(|b| b.group == name)
                .expect("golden block")
        };
        // gp: scalar q=1 block.
        let gp_sd = f.varcorr[0][0].sqrt();
        let gp_ref = gold_of("gp").stddev[0];
        assert!(
            (gp_sd - gp_ref).abs() / gp_ref.max(1e-6) < 3e-2,
            "gp stddev glmm={gp_sd:.6} lme4={gp_ref:.6}",
        );
        // ge: q=5 block. Diagonal of D in column-major lower-tri vech (q=5) is at
        // offsets 0,5,9,12,14 for terms (Intercept,x1,x2,x3,x4) — glmm's Λ term order
        // (intercept then slopes [1,2,3,4]) matches the golden's `terms` order.
        const GE_DIAG: [usize; 5] = [0, 5, 9, 12, 14];
        let ge_ref = gold_of("ge");
        for (t, &off) in GE_DIAG.iter().enumerate() {
            let got = f.varcorr[1][off].sqrt();
            let rf = ge_ref.stddev[t];
            assert!(
                (got - rf).abs() / rf.max(1e-6) < 3e-2,
                "ge stddev[{t}] glmm={got:.6} lme4={rf:.6}",
            );
        }
    }

    // ── Sparse non-Gaussian goldens (gamma over-width, NB over-count) ─

    /// Shared serde shape for the two sparse GLMM goldens (fit_m3_goldens.R's
    /// glmm schema). serde ignores unread fields (loglik, corr, …).
    #[derive(serde::Deserialize)]
    struct SgVcBlock {
        group: String,
        stddev: Vec<f64>,
    }
    #[derive(serde::Deserialize)]
    struct SgEst {
        beta: Vec<f64>,
        se_hessian: Vec<f64>,
        se_rx: Vec<f64>,
        varcomp: Vec<SgVcBlock>,
        #[serde(default)]
        dispersion: Option<f64>,
        #[serde(default)]
        theta: Option<f64>,
    }
    #[derive(serde::Deserialize)]
    struct SgGolden {
        estimates: SgEst,
    }

    /// Map string factor labels to dense 0-based ids (first-seen order) —
    /// the `dense_str` pattern shared by the over-cap golden tests above.
    fn dense_ids(raw: &[String]) -> Vec<u32> {
        use std::collections::HashMap;
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut next = 0u32;
        raw.iter()
            .map(|r| {
                *map.entry(r.clone()).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                })
            })
            .collect()
    }

    /// OVER-WIDTH gamma GLMM golden: `y ~ 1 + x1..x4 + (1|gp) + (1 + x1..x4 | ge)`,
    /// gamma/log — `ge` carries q_g = 5 > MAX_EXTRA_Q, so the design routes to the
    /// sparse non-Gaussian PIRLS (`fit_glmm_sparse`); no dense twin exists.
    /// Gated against frozen `glmer(Gamma("log"))` (`parity/goldens/sim_sparse_gamma.json`).
    /// The oracle is sacred.
    ///
    /// Both SE arms are lme4-gated: **Hessian** (glmm's default) against
    /// `se_hessian` (the like-for-like pairing the sim_gamma_glmm golden
    /// settled), and **Rx** against `se_rx` — glmm's Gamma Rx carries lme4's
    /// σ̂² = pwrss/n like `vcov(use.hessian=FALSE)` (`family::glmm_sigma_sq`;
    /// unscaled, the two differ by exactly σ̂ on this dataset).
    #[test]
    fn fit_sparse_gamma_glmm_matches_lme4() {
        // Heavy (~8 min release: n=1200, 21-dim joint BOBYQA + FD-Hessian SE) —
        // gated like the heavy crossover cells; the default suite keeps the NB
        // golden (43 s) + the over-envelope gamma convergence smoke.
        if !cfg!(feature = "loop_advanced") {
            eprintln!(
                "fit_sparse_gamma_glmm_matches_lme4: heavy golden skipped — \
                 run with --features loop_advanced"
            );
            return;
        }
        let raw = include_str!("../parity/goldens/sim_sparse_gamma.json");
        let gold: SgGolden = serde_json::from_str(raw).expect("golden JSON parses");

        let csv = include_str!("../parity/data_simulated/sim_sparse_gamma.csv");
        // Columns: y, x1..x4, gp, ge (indices 0..6).
        let mut y = Vec::<f64>::new();
        let mut xc: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
        let (mut gp_raw, mut ge_raw) = (Vec::<String>::new(), Vec::<String>::new());
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            for k in 0..4 {
                xc[k].push(f[1 + k].parse().unwrap());
            }
            gp_raw.push(f[5].to_string());
            ge_raw.push(f[6].to_string());
        }
        let n = y.len();
        let p = 5;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            for k in 0..4 {
                x[i * p + 1 + k] = xc[k][i];
            }
        }
        let model = crate::ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q ⇒ Sparse
                }],
            }),
        };
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ));
        let ids = crate::GroupIds {
            primary: dense_ids(&gp_raw),
            extra: vec![dense_ids(&ge_raw)],
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1, 2, 3, 4],
            ..crate::FitOptions::default() // default WaldSe::Hessian (see doc)
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(f.converged, "sparse gamma GLMM must converge");

        // β: 2e-2 relative (the over-cap phase-1 band the wide-slopes golden
        // uses); se_hessian at the FD-Hessian floor 3e-2 (compare.R's
        // se_hessian_rel).
        for j in 0..p {
            let rb = gold.estimates.beta[j];
            let rs = gold.estimates.se_hessian[j];
            assert!(
                (f.beta[j] - rb).abs() / rb.abs().max(1e-6) < 2e-2,
                "β[{j}] glmm={} lme4={rb}",
                f.beta[j]
            );
            assert!(
                (f.se[j] - rs).abs() / rs.abs().max(1e-6) < 3e-2,
                "se[{j}] glmm={} lme4={rs}",
                f.se[j]
            );
        }
        // Dispersion: post-fit Pearson moment, same estimator as the golden's
        // hand-computed Σpearson²/(n−p) (the sim_gamma_glmm precedent).
        let rd = gold
            .estimates
            .dispersion
            .expect("gamma golden carries dispersion");
        assert!(
            (f.dispersion - rd).abs() / rd < 3e-2,
            "φ̂ glmm={} lme4={rd}",
            f.dispersion
        );
        // Varcomp via stddev_corr — varcorr is σ̂²-scaled like tau2 (B1 fix),
        // directly lme4's Gamma VarCorr stddev scale. glmm order
        // [gp (primary), ge (extra)]; lme4's VarCorr order is descending level
        // count [ge(40), gp(20)] — map by group NAME.
        let gold_of = |name: &str| {
            gold.estimates
                .varcomp
                .iter()
                .find(|b| b.group == name)
                .expect("golden block")
        };
        let (gp_sds, _) = f.stddev_corr(0);
        let gp_ref = gold_of("gp").stddev[0];
        assert!(
            (gp_sds[0] - gp_ref).abs() / gp_ref.max(1e-6) < 3e-2,
            "gp stddev glmm={:.6} lme4={gp_ref:.6}",
            gp_sds[0]
        );
        let (ge_sds, _) = f.stddev_corr(1);
        let ge_ref = gold_of("ge");
        for (t, &got) in ge_sds.iter().enumerate() {
            let rf = ge_ref.stddev[t];
            assert!(
                (got - rf).abs() / rf.max(1e-6) < 5e-2,
                "ge stddev[{t}] glmm={got:.6} lme4={rf:.6}"
            );
        }

        // Rx arm vs the golden's `se_rx` (σ̂²-scaled, see doc). A second full fit —
        // cheap relative to the Hessian arm's FD sweep on this 21-dim design.
        let f_rx = crate::fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &crate::FitOptions {
                target_indices: vec![0, 1, 2, 3, 4],
                wald_se: crate::WaldSe::Rx,
                ..crate::FitOptions::default()
            },
        );
        assert!(f_rx.converged, "sparse gamma GLMM (Rx) must converge");
        for j in 0..p {
            let rs = gold.estimates.se_rx[j];
            assert!(
                (f_rx.se[j] - rs).abs() / rs.abs().max(1e-6) < 3e-2,
                "rx se[{j}] glmm={} lme4={rs}",
                f_rx.se[j]
            );
        }
    }

    /// OVER-COUNT NB GLMM golden: `y ~ 1 + x + (1|g1) + (1|c1) + … + (1|c7)`,
    /// negbin/log — 7 crossed extras > MAX_EXTRA_GROUPINGS route to the sparse NB
    /// marginal-θ wrapper (`fit_glmm_nb_sparse`). Gated against frozen
    /// `lme4::glmer.nb` (`parity/goldens/sim_sparse_nb.json`). The oracle is
    /// sacred. (glmer.nb printed interim-fit convergence warnings while
    /// generating the reference — from its internal θ-candidate refits — but the
    /// FINAL model carries no convergence messages and its estimates sit near the
    /// simulation truth, so the golden stands.) Rx SE, as the gamma golden.
    #[test]
    fn fit_sparse_nb_glmm_matches_lme4() {
        let raw = include_str!("../parity/goldens/sim_sparse_nb.json");
        let gold: SgGolden = serde_json::from_str(raw).expect("golden JSON parses");

        let csv = include_str!("../parity/data_simulated/sim_sparse_nb.csv");
        // Columns: y, x, g1, c1..c7 (indices 0..9).
        let mut y = Vec::<f64>::new();
        let mut xcol = Vec::<f64>::new();
        let mut fac: Vec<Vec<String>> = vec![Vec::new(); 8]; // g1, c1..c7
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            xcol.push(f[1].parse().unwrap());
            for k in 0..8 {
                fac[k].push(f[2 + k].to_string());
            }
        }
        let n = y.len();
        let p = 2;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            x[i * p + 1] = xcol[i];
        }
        let model = crate::ModelSpec {
            family: Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
                slopes: vec![],
                extra_groupings: (0..7)
                    .map(|_| Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    })
                    .collect(),
            }),
        };
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ));
        let ids = crate::GroupIds {
            primary: dense_ids(&fac[0]),
            extra: fac[1..].iter().map(|f| dense_ids(f)).collect(),
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            wald_se: crate::WaldSe::Rx,
            ..crate::FitOptions::default()
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(f.converged, "sparse NB GLMM must converge");

        // θ̂ (NB dispersion): 5e-2 relative — the sim_nb_glmm golden's band (two
        // independent marginal-θ golden-section searches over re-fit surfaces).
        let rt = gold.estimates.theta.expect("NB golden carries theta");
        assert!(
            (f.dispersion - rt).abs() / rt < 5e-2,
            "θ̂ glmm={} lme4={rt}",
            f.dispersion
        );
        for j in 0..p {
            let rb = gold.estimates.beta[j];
            let rs = gold.estimates.se_rx[j];
            assert!(
                (f.beta[j] - rb).abs() / rb.abs().max(1e-6) < 2e-2,
                "β[{j}] glmm={} lme4={rb}",
                f.beta[j]
            );
            assert!(
                (f.se[j] - rs).abs() / rs.abs().max(1e-6) < 2e-2,
                "se[{j}] glmm={} lme4={rs}",
                f.se[j]
            );
        }
        // 8 scalar varcomp blocks, glmm order [g1 | c1..c7] == lme4's VarCorr
        // order (g1 has the most levels; ties keep formula order — the
        // sim_wide_crossed precedent).
        assert_eq!(f.varcorr.len(), 8, "8 scalar varcomp blocks");
        for k in 0..8 {
            let ref_sd = gold.estimates.varcomp[k].stddev[0];
            let got_sd = f.varcorr[k][0].sqrt();
            assert!(
                (got_sd - ref_sd).abs() / ref_sd.max(1e-6) < 3e-2,
                "varcomp[{k}] stddev glmm={got_sd:.6} lme4={ref_sd:.6}"
            );
        }
    }

    /// Weighted twin of `fit_sparse_gamma_glmm_matches_lme4` (Task 7): same
    /// over-width design and data. Uses `wᵢ = 1 + 0.2·((i mod 3) − 1)`
    /// (0-based row index, cycling 0.8/1.0/1.2), NOT the integer `1 + (i mod
    /// 3)` scheme the Gamma/NB replication tests use: on THIS wide design
    /// (q_g = 5 slope-block extra, 21-dim joint BOBYQA), integer weights up
    /// to 3× drove lme4's `vcov(use.hessian=TRUE)` to implausible SE ~250×
    /// tighter than the unweighted golden's (0.0007 vs 0.19, same effect
    /// sizes, only a mild dispersion shift) — a numerically unstable Hessian
    /// on this over-parameterized shape, not a real precision gain (verified
    /// interactively; `isSingular` false, no convergence messages, yet the
    /// Hessian SE is not credible). The gentler weights keep glmer's Hessian
    /// well-conditioned (SE lands back at the unweighted golden's scale) while
    /// still exercising the same weighted code path. Closes the sparse Gamma
    /// weighting gap — profiled dispersion (`gamma_aic`) and the post-fit
    /// Pearson φ̂ both take `ws.prior_w`. Gated behind `loop_advanced` like its
    /// unweighted sibling (same 21-dim joint BOBYQA + FD-Hessian SE cost).
    /// Generated with (R 4.5.3, lme4 1.1-38):
    /// ```r
    ///   d$w <- 1 + 0.2 * (((seq_len(nrow(d)) - 1) %% 3) - 1)
    ///   f <- glmer(y ~ 1 + x1 + x2 + x3 + x4 + (1|gp) + (1 + x1 + x2 + x3 + x4 | ge),
    ///              family = Gamma("log"), weights = d$w, data = d)
    /// ```
    /// se_hessian/dispersion at the unweighted golden's bands (3e-2). β at
    /// 4e-2, not the unweighted golden's 2e-2: `x1..x4` land within 1% (the
    /// weighting math is exact there), but `(Intercept)` — the design's
    /// least-identified coefficient, t ≈ 1.2, SE ≈ 80% of the point estimate
    /// — drifts ~3.4% between glmm's and lme4's independent 21-dim BOBYQA
    /// paths to the same shallow optimum; se_hessian on that same coefficient
    /// still lands within 0.1%, confirming the curvature (hence the
    /// weighting) is correct and this is optimizer-path scatter on a
    /// poorly-determined direction, not a weighting bug.
    #[test]
    fn fit_sparse_gamma_glmm_weighted_matches_lme4() {
        if !cfg!(feature = "loop_advanced") {
            eprintln!(
                "fit_sparse_gamma_glmm_weighted_matches_lme4: heavy golden skipped — \
                 run with --features loop_advanced"
            );
            return;
        }
        const REF_BETA: [f64; 5] = [
            0.233369872688657,
            0.511759152360149,
            -0.345961162194708,
            0.236273530986550,
            -0.228445413694595,
        ];
        const REF_SE_HESSIAN: [f64; 5] = [
            0.1890907576028805,
            0.0921338121875385,
            0.0846377387177761,
            0.0585608862317671,
            0.0475892885636111,
        ];
        // Pearson moment Σwᵢrᵢ²/(n−p) (`residuals(f, type="pearson")`), NOT
        // `sigma(f)^2` (pwrss/n on the link scale) — the two are different
        // quantities (see `fit_glmm_sparse`'s `dispersion` arm doc) and only
        // the Pearson form matches `glmm`'s `Fit::dispersion` field.
        const REF_DISPERSION: f64 = 0.411217227312831;

        let csv = include_str!("../parity/data_simulated/sim_sparse_gamma.csv");
        // Columns: y, x1..x4, gp, ge (indices 0..6).
        let mut y = Vec::<f64>::new();
        let mut xc: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
        let (mut gp_raw, mut ge_raw) = (Vec::<String>::new(), Vec::<String>::new());
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            for k in 0..4 {
                xc[k].push(f[1 + k].parse().unwrap());
            }
            gp_raw.push(f[5].to_string());
            ge_raw.push(f[6].to_string());
        }
        let n = y.len();
        let p = 5;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            for k in 0..4 {
                x[i * p + 1 + k] = xc[k][i];
            }
        }
        let weights: Vec<f64> = (0..n).map(|i| 1.0 + 0.2 * ((i % 3) as f64 - 1.0)).collect();
        let model = crate::ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![1, 2, 3, 4],
                }],
            }),
        };
        let ids = crate::GroupIds {
            primary: dense_ids(&gp_raw),
            extra: vec![dense_ids(&ge_raw)],
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1, 2, 3, 4],
            weights: Some(weights),
            ..crate::FitOptions::default() // default WaldSe::Hessian
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(f.converged, "weighted sparse gamma GLMM must converge");
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs().max(1e-6) < 4e-2,
                "β[{j}] glmm={} lme4={}",
                f.beta[j],
                REF_BETA[j]
            );
            assert!(
                (f.se[j] - REF_SE_HESSIAN[j]).abs() / REF_SE_HESSIAN[j].max(1e-6) < 3e-2,
                "se[{j}] glmm={} lme4={}",
                f.se[j],
                REF_SE_HESSIAN[j]
            );
        }
        assert!(
            (f.dispersion - REF_DISPERSION).abs() / REF_DISPERSION < 3e-2,
            "φ̂ glmm={} lme4={REF_DISPERSION}",
            f.dispersion
        );
    }

    /// Weighted sparse NB has NO lme4 golden (Task 7): the over-count
    /// `sim_sparse_nb` design (7 crossed extras, several small clusters) is
    /// already fragile for `glmer.nb`'s marginal-θ profile unweighted (see
    /// `fit_sparse_nb_glmm_matches_lme4`'s doc — interim-refit convergence
    /// warnings during golden generation). Verified interactively at THREE
    /// weight magnitudes and none give a trustworthy weighted oracle: integer
    /// `wᵢ = 1 + (i mod 3)` (1/2/3) collapses `isSingular` (several variance
    /// components hit exactly 0); `wᵢ = 1 + 0.2·((i mod 3) − 1)` (0.8/1.0/1.2)
    /// converges (`isSingular` false) but θ̂ jumps 17% off the unweighted
    /// value; `wᵢ = 1 + 0.05·((i mod 3) − 1)` (0.95/1.0/1.05, i.e. a mere ±5%
    /// perturbation) STILL prints "Model failed to converge" and moves θ̂ 4%.
    /// glmm's θ golden-section search, by contrast, stays within <1% of its
    /// unweighted value under the SAME ±5%/±20% perturbations — evidence
    /// glmm's path is the more numerically stable one here, not that it is
    /// ignoring the weights. Rather than pick a weight scheme until lme4
    /// happens to land somewhere assertable (`p`-hacking the oracle — the
    /// oracle is sacred, so it is not tuned to pass), this design is
    /// covered instead by the mathematically exact replication-equivalence
    /// test below, which needs no external oracle.
    ///
    /// Integer prior weights = row replication (NB): `w = 2` on `n` unique
    /// rows fits identically to the same rows each duplicated once — Σwᵢ·devᵢ
    /// over unique rows equals Σdevᵢ over duplicated rows, so the two
    /// marginal-θ profiles (`fit_glmm_nb_sparse`'s golden-section search over
    /// `−½D(θ) + nb_profile_loglik(y, y, θ, weights)`) share an argmax. Full
    /// β/SE/θ equality (NB's dispersion IS θ̂ itself, driven by the SAME
    /// weighted profile on both sides — unlike Gamma's Pearson φ̂, nothing
    /// here depends on the raw row count). Tolerances mirror the
    /// dense-vs-sparse cross-check (sparse.rs:5735-5752): β 2e-3 rel, SE 2e-2
    /// rel, θ 2e-2 rel.
    #[test]
    fn sparse_weighted_nb_matches_replicated() {
        let family = Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        };
        let ((xw, yw, w, nw, idsw), (xd, yd, nd, idsd), p, model) =
            build_sparse_weighted_replication_case(family, 613);
        let opts_w = crate::FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..crate::FitOptions::default()
        };
        let opts_d = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        let fw = crate::fit_cold(&xw, &yw, nw, p, &model, &idsw, &opts_w);
        let fd = crate::fit_cold(&xd, &yd, nd, p, &model, &idsd, &opts_d);
        assert!(fw.converged && fd.converged, "both fits must converge");
        for j in 0..p {
            assert!(
                (fw.beta[j] - fd.beta[j]).abs() < 2e-3 * (1.0 + fd.beta[j].abs()),
                "β[{j}] weighted={} replicated={}",
                fw.beta[j],
                fd.beta[j]
            );
            assert!(
                (fw.se[j] - fd.se[j]).abs() < 2e-2 * (1.0 + fd.se[j].abs()),
                "se[{j}] weighted={} replicated={}",
                fw.se[j],
                fd.se[j]
            );
        }
        assert!(
            (fw.dispersion - fd.dispersion).abs() < 2e-2 * (1.0 + fd.dispersion.abs()),
            "θ̂: weighted={} replicated={}",
            fw.dispersion,
            fd.dispersion
        );
    }

    /// Rung-18 shape: binomial logit, primary (1 + x | g1) + crossed extra
    /// (1 + x | g2), prior weights = size — the first non-Gaussian design with
    /// a slope-carrying extra grouping. IN-envelope (q_g = 2 ≤ MAX_EXTRA_Q):
    /// reaches Sparse purely through classify_design's slope-extras clause.
    /// Gated against frozen glmer (parity/goldens/sim_binomial_slope_crossed.json,
    /// tolPwrss = 1e-13). The oracle is sacred.
    #[test]
    fn fit_sparse_binomial_slope_crossed_matches_lme4() {
        let raw = include_str!("../parity/goldens/sim_binomial_slope_crossed.json");
        let gold: SgGolden = serde_json::from_str(raw).expect("golden JSON parses");

        let csv = include_str!("../parity/data_simulated/sim_binomial_slope_crossed.csv");
        // Columns: incidence, size, x, g1, g2 (indices 0..4). Aggregated
        // binomial: y = incidence/size (proportion), prior weights = size —
        // mirrors parity/oracle/fit.rs's weighted rung-18 lowering.
        let mut y = Vec::<f64>::new();
        let mut size_col = Vec::<f64>::new();
        let mut xcol = Vec::<f64>::new();
        let (mut g1_raw, mut g2_raw) = (Vec::<String>::new(), Vec::<String>::new());
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let incidence: f64 = f[0].parse().unwrap();
            let size: f64 = f[1].parse().unwrap();
            y.push(incidence / size);
            size_col.push(size);
            xcol.push(f[2].parse().unwrap());
            g1_raw.push(f[3].to_string());
            g2_raw.push(f[4].to_string());
        }
        let n = y.len();
        let p = 2;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            x[i * p + 1] = xcol[i];
        }
        let model = crate::ModelSpec {
            family: Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // sized from ids
                slopes: vec![1],                                 // (1 + x | g1)
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![1], // (1 + x | g2)
                }],
            }),
        };
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ));
        let ids = crate::GroupIds {
            primary: dense_ids(&g1_raw),
            extra: vec![dense_ids(&g2_raw)],
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            weights: Some(size_col.clone()),
            ..crate::FitOptions::default() // default WaldSe::Hessian
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(
            f.converged,
            "sparse binomial slope-crossed GLMM must converge"
        );

        // β at 2e-2 relative, se_hessian at the FD-Hessian floor 3e-2 — the
        // sparse-golden bands (gamma/NB precedent).
        for j in 0..p {
            let rb = gold.estimates.beta[j];
            let rs = gold.estimates.se_hessian[j];
            assert!(
                (f.beta[j] - rb).abs() / rb.abs().max(1e-6) < 2e-2,
                "β[{j}] glmm={} lme4={rb}",
                f.beta[j]
            );
            assert!(
                (f.se[j] - rs).abs() / rs.abs().max(1e-6) < 3e-2,
                "se[{j}] glmm={} lme4={rs}",
                f.se[j]
            );
        }
        // Varcomp by group NAME (glmm order is declaration order [g1, g2] but
        // never assume the golden's): two q=2 blocks; varcorr[k] is the
        // column-major lower-tri vech of D̂ on the link scale (binomial: no σ̂
        // scaling), diag at offsets 0, 2 for terms (Intercept, x).
        assert_eq!(f.varcorr.len(), 2, "two q=2 varcomp blocks (g1 + g2)");
        let gold_of = |name: &str| {
            gold.estimates
                .varcomp
                .iter()
                .find(|b| b.group == name)
                .expect("golden block")
        };
        const DIAG_Q2: [usize; 2] = [0, 2];
        for (k, name) in ["g1", "g2"].iter().enumerate() {
            let ref_block = gold_of(name);
            for (t, &off) in DIAG_Q2.iter().enumerate() {
                let got = f.varcorr[k][off].sqrt();
                let rf = ref_block.stddev[t];
                assert!(
                    (got - rf).abs() / rf.max(1e-6) < 3e-2,
                    "{name} stddev[{t}] glmm={got:.6} lme4={rf:.6}"
                );
            }
        }
    }

    // ── Sparse non-Gaussian GLMM cross-checks ────────────────────

    /// Deterministic in-envelope GLMM design shared by the both-paths
    /// cross-checks: 4 primary clusters + one crossed extra (3 levels), p = 2
    /// (intercept + covariate), family-appropriate y generated from a linear
    /// predictor with genuine per-level RE effects. In-envelope on every axis,
    /// so `fit_cold` routes it to the dense NoZ GLMM kernel — the oracle.
    fn build_glmm_case(
        family: Family,
        seed: u64,
    ) -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, crate::GroupIds) {
        let n = 96;
        let p = 2;
        let mut st = seed;
        let n_primary = 4usize;
        let n_extra_levels = 3usize;
        let u_c: Vec<f64> = (0..n_primary)
            .map(|_| 0.6 * super::test_lcg(&mut st))
            .collect();
        let v_e: Vec<f64> = (0..n_extra_levels)
            .map(|_| 0.4 * super::test_lcg(&mut st))
            .collect();
        let mut x = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut eid = vec![0u32; n];
        for i in 0..n {
            let cov = super::test_lcg(&mut st);
            x[i * p] = 1.0;
            x[i * p + 1] = cov;
            pid[i] = (i % n_primary) as u32;
            eid[i] = (i % n_extra_levels) as u32;
            let eta = 0.4 + 0.6 * cov + u_c[pid[i] as usize] + v_e[eid[i] as usize];
            y[i] = match family {
                Family::Binomial { .. } => {
                    let pr = 1.0 / (1.0 + (-eta).exp());
                    let uni = 0.5 * (super::test_lcg(&mut st) + 1.0); // (0, 1)
                    if uni < pr {
                        1.0
                    } else {
                        0.0
                    }
                }
                Family::Poisson { .. } | Family::NegativeBinomial { .. } => {
                    // Count-like data around exp(η) with one-sided jitter — the test
                    // compares two fitters on the SAME data, so exact Poisson/NB
                    // sampling is unnecessary.
                    let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
                    (eta.exp() * jit).round().max(0.0)
                }
                Family::Gamma { .. } => {
                    let jit = 1.0 + 0.3 * super::test_lcg(&mut st);
                    (eta.exp() * jit).max(0.05)
                }
                Family::Gaussian => unreachable!("non-Gaussian cases only"),
            };
        }
        let model = ModelSpec {
            family,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_primary as u32,
                },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: n_extra_levels as u32,
                    },
                    slopes: vec![],
                }],
            }),
        };
        let ids = crate::GroupIds {
            primary: pid,
            extra: vec![eid],
        };
        (x, y, n, p, model, ids)
    }

    /// The sparse parallel FD-Hessian grid must reproduce the serial one BITWISE.
    /// Drives a full `fit_glmm_sparse` (WaldSe::Hessian) twice — `parallel_inner`
    /// off then on — on a crossed binomial design routed through the sparse path,
    /// and asserts the returned marginal deviance, `se`, and `stddev_se` are
    /// bit-identical. `parallel_inner` gates ONLY `sparse_fd_hessian_cov` here (the
    /// sparse fit has no AGQ), so the BOBYQA optimum is shared and any difference
    /// isolates to the parallel grid. Every eval cold-seeds û = 0, so per-thread
    /// worker workspaces (`clone_worker`) are exact — a mismatch is a field-liveness
    /// bug, not noise.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    #[test]
    fn sparse_fd_hessian_parallel_bit_identical_to_serial() {
        let (xflat, y, n, p, model, ids) = build_glmm_case(
            Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            202,
        );
        let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        let run = |parallel_inner: bool| {
            let opts = crate::FitOptions {
                target_indices: vec![0, 1],
                wald_se: crate::WaldSe::Hessian,
                parallel_inner,
                ..crate::FitOptions::default()
            };
            super::fit_glmm_sparse(
                &xflat,
                &y,
                n,
                p,
                &sized,
                &ids.primary,
                &ids.extra,
                f64::NAN,
                None,
                &opts,
            )
        };
        let (fit_s, dev_s) = run(false);
        let (fit_p, dev_p) = run(true);
        assert!(
            fit_s.converged && fit_p.converged,
            "both fits must converge"
        );
        assert_eq!(
            dev_s.to_bits(),
            dev_p.to_bits(),
            "marginal deviance not bit-identical: {dev_s} vs {dev_p}"
        );
        assert_eq!(fit_s.se.len(), fit_p.se.len());
        for (j, (&a, &b)) in fit_s.se.iter().zip(fit_p.se.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "se[{j}] not bit-identical: {a} vs {b}"
            );
        }
        assert_eq!(fit_s.stddev_se.len(), fit_p.stddev_se.len());
        for (k, (&a, &b)) in fit_s
            .stddev_se
            .iter()
            .zip(fit_p.stddev_se.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "stddev_se[{k}] not bit-identical: {a} vs {b}"
            );
        }
    }

    /// Deviance-value cross-check (the internal correctness anchor):
    /// `sparse_glmm_deviance` equals the dense `glmm_laplace_deviance`
    /// at the same (θ, β) on in-envelope designs both can evaluate. The two
    /// PIRLS drivers share the discipline but not the arithmetic order (and the
    /// dense logit path is fused-SIMD where the sparse one takes the general
    /// family branch), so the bound is relative, not bitwise. A disagreement is
    /// a bug in exactly one path.
    #[test]
    fn sparse_glmm_deviance_matches_dense() {
        use faer::Mat;
        for (family, seed) in [
            (
                Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                101u64,
            ),
            (
                Family::Poisson {
                    link: crate::PoissonLink::Log,
                },
                103,
            ),
            (
                Family::Gamma {
                    link: crate::GammaLink::Log,
                },
                107,
            ),
        ] {
            let (xflat, y, n, p, model, ids) = build_glmm_case(family, seed);
            let x = Mat::<f64>::from_fn(n, p, |i, j| xflat[i * p + j]);

            // Dense workspace, mirroring the fit.rs adapter's construction.
            let mut dws = crate::glmm::GlmmWorkspace::for_cluster_spec(p, &model, n, &[], 1);
            crate::glmm::build_z(&mut dws, x.as_ref(), &ids.primary, &ids.extra, n);
            dws.structured_schur = if dws.groupings.structured_extras_eligible() {
                crate::glmm::StructuredSchur::new(&dws.groupings, &ids.primary, &ids.extra, n)
            } else {
                None
            };

            let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n, &[], &[]);
            let mut sws = super::SparseGlmmWorkspace::new(&g, &ids.primary, &ids.extra, n, p);

            // (θ, β) probe points: [θ_primary, θ_extra, β0, β1].
            for params in [
                [0.5f64, 0.7, 0.3, 0.5],
                [1.0, 0.2, -0.2, 0.8],
                [0.15, 1.1, 0.6, -0.4],
            ] {
                let dense = crate::glmm::glmm_laplace_deviance(
                    &params,
                    &mut dws,
                    x.as_ref(),
                    &y,
                    &ids.primary,
                    n,
                );
                let sparse = super::sparse_glmm_deviance(
                    family,
                    f64::NAN,
                    &params,
                    &mut sws,
                    x.as_ref(),
                    &y,
                    n,
                    false,
                );
                assert!(
                    (dense - sparse).abs() < 1e-6 * (1.0 + dense.abs()),
                    "{family:?} params={params:?}: dense {dense} vs sparse {sparse}"
                );
            }
        }
    }

    /// Fit-level both-paths cross-check (the acceptance criterion for the sparse
    /// non-Gaussian path): force the
    /// sparse non-Gaussian solver on in-envelope designs and diff β/SE/τ²
    /// against the dense NoZ GLMM fit reached through `fit_cold`. The two sides
    /// are independent BOBYQA minimizations of (numerically) the same Laplace
    /// deviance — dense two-stage vs sparse single joint stage — so the bound
    /// is optimizer-scatter-sized, not machine precision (the deviance-level
    /// test above is the tight anchor). Covers both `WaldSe` arms.
    #[test]
    fn sparse_glmm_fit_matches_dense_in_envelope() {
        for (family, seed) in [
            (
                Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                201u64,
            ),
            (
                Family::Poisson {
                    link: crate::PoissonLink::Log,
                },
                203,
            ),
            (
                Family::Gamma {
                    link: crate::GammaLink::Log,
                },
                207,
            ),
            (
                Family::NegativeBinomial {
                    link: crate::NegBinomialLink::Log,
                },
                211,
            ),
        ] {
            let (xflat, y, n, p, model, ids) = build_glmm_case(family, seed);
            for wald_se in [crate::WaldSe::Hessian, crate::WaldSe::Rx] {
                let opts = crate::FitOptions {
                    target_indices: vec![0, 1],
                    wald_se,
                    ..crate::FitOptions::default()
                };
                let dense = crate::fit_cold(&xflat, &y, n, p, &model, &ids, &opts);
                let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
                let sp = if matches!(family, Family::NegativeBinomial { .. }) {
                    super::fit_glmm_nb_sparse(
                        &xflat,
                        &y,
                        n,
                        p,
                        &sized,
                        &ids.primary,
                        &ids.extra,
                        None,
                        &opts,
                    )
                } else {
                    super::fit_glmm_sparse(
                        &xflat,
                        &y,
                        n,
                        p,
                        &sized,
                        &ids.primary,
                        &ids.extra,
                        f64::NAN,
                        None,
                        &opts,
                    )
                    .0
                };
                let tag = format!("{family:?}/{wald_se:?}");
                assert!(
                    dense.converged && sp.converged,
                    "{tag}: both paths must converge"
                );
                for j in 0..p {
                    assert!(
                        (sp.beta[j] - dense.beta[j]).abs() < 2e-3 * (1.0 + dense.beta[j].abs()),
                        "{tag} β[{j}]: sparse={} dense={}",
                        sp.beta[j],
                        dense.beta[j]
                    );
                    assert!(
                        (sp.se[j] - dense.se[j]).abs() < 2e-2 * (1.0 + dense.se[j].abs()),
                        "{tag} se[{j}]: sparse={} dense={}",
                        sp.se[j],
                        dense.se[j]
                    );
                }
                assert_eq!(sp.tau2.len(), dense.tau2.len(), "{tag}: tau2 length");
                for (a, b) in sp.tau2.iter().zip(dense.tau2.iter()) {
                    assert!(
                        (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                        "{tag} tau2: sparse={a} dense={b}"
                    );
                }
                assert!(
                    (sp.dispersion - dense.dispersion).abs()
                        < 2e-2 * (1.0 + dense.dispersion.abs()),
                    "{tag} dispersion: sparse={} dense={}",
                    sp.dispersion,
                    dense.dispersion
                );
            }
        }
    }

    /// Aggregated-binomial / expanded-Bernoulli twin datasets on an over-count
    /// design (7 crossed extras ⇒ `Solver::Sparse`): each aggregated row i
    /// carries mᵢ ∈ 2..=5 trials with sᵢ successes; its expanded twin holds mᵢ
    /// one-trial 0/1 rows with the SAME covariate and level ids, so both
    /// describe identical Bernoulli data. Returns
    /// `(aggregated (x, y=s/m, weights=m, n, ids), expanded (x, y, n, ids), p,
    /// model, sat)` where `sat = 2Σᵢ[sᵢ ln(sᵢ/mᵢ) + (mᵢ−sᵢ) ln((mᵢ−sᵢ)/mᵢ)]`
    /// (0·ln0 = 0) is the data-only saturated term by which the aggregated
    /// weighted deviance falls below the expanded one — same argmin.
    #[allow(clippy::type_complexity)]
    fn build_binomial_weighted_pair() -> (
        (Vec<f64>, Vec<f64>, Vec<f64>, usize, crate::GroupIds),
        (Vec<f64>, Vec<f64>, usize, crate::GroupIds),
        usize,
        ModelSpec,
        f64,
    ) {
        let n_agg = 72;
        let p = 2;
        let mut st = 401u64;
        let n_primary = 6usize;
        let extra_levels = [3usize, 4, 3, 5, 3, 4, 3];
        let u_c: Vec<f64> = (0..n_primary)
            .map(|_| 0.5 * super::test_lcg(&mut st))
            .collect();
        let v_e: Vec<Vec<f64>> = extra_levels
            .iter()
            .map(|&l| (0..l).map(|_| 0.3 * super::test_lcg(&mut st)).collect())
            .collect();
        let pid_a: Vec<u32> = (0..n_agg).map(|i| (i % n_primary) as u32).collect();
        let extra_a: Vec<Vec<u32>> = extra_levels
            .iter()
            .enumerate()
            .map(|(g, &l)| (0..n_agg).map(|i| ((i / (g + 1)) % l) as u32).collect())
            .collect();
        let mut xa = vec![0.0f64; n_agg * p];
        let mut ya = vec![0.0f64; n_agg];
        let mut wa = vec![0.0f64; n_agg];
        let (mut xe, mut ye, mut pid_e) = (Vec::new(), Vec::new(), Vec::new());
        let mut extra_e: Vec<Vec<u32>> = vec![Vec::new(); extra_levels.len()];
        let mut sat = 0.0f64;
        for i in 0..n_agg {
            let cov = super::test_lcg(&mut st);
            xa[i * p] = 1.0;
            xa[i * p + 1] = cov;
            let mut e = 0.3 + 0.5 * cov + u_c[pid_a[i] as usize];
            for (g, ids_g) in extra_a.iter().enumerate() {
                e += v_e[g][ids_g[i] as usize];
            }
            let pr = 1.0 / (1.0 + (-e).exp());
            let m = 2 + (i % 4);
            let mut s = 0usize;
            for _ in 0..m {
                let uni = 0.5 * (super::test_lcg(&mut st) + 1.0);
                let yk = if uni < pr { 1.0 } else { 0.0 };
                s += yk as usize;
                ye.push(yk);
                xe.push(1.0);
                xe.push(cov);
                pid_e.push(pid_a[i]);
                for (g, col) in extra_e.iter_mut().enumerate() {
                    col.push(extra_a[g][i]);
                }
            }
            ya[i] = s as f64 / m as f64;
            wa[i] = m as f64;
            let (mf, sf) = (m as f64, s as f64);
            if s > 0 {
                sat += 2.0 * sf * (sf / mf).ln();
            }
            if s < m {
                sat += 2.0 * (mf - sf) * ((mf - sf) / mf).ln();
            }
        }
        let n_exp = ye.len();
        let model = ModelSpec {
            family: Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_primary as u32,
                },
                slopes: vec![],
                extra_groupings: extra_levels
                    .iter()
                    .map(|&l| Grouping {
                        relation: GroupingRelation::Crossed {
                            n_clusters: l as u32,
                        },
                        slopes: vec![],
                    })
                    .collect(),
            }),
        };
        let ids_a = crate::GroupIds {
            primary: pid_a,
            extra: extra_a,
        };
        let ids_e = crate::GroupIds {
            primary: pid_e,
            extra: extra_e,
        };
        (
            (xa, ya, wa, n_agg, ids_a),
            (xe, ye, n_exp, ids_e),
            p,
            model,
            sat,
        )
    }

    /// Non-Gaussian sparse-classified design (7 crossed extras — same RE
    /// shape as `build_binomial_weighted_pair`, `> MAX_EXTRA_GROUPINGS` so
    /// `classify_design` routes `Sparse` regardless of family) for the
    /// weighted/replicated equivalence tests below: integer weight `w = 2` on
    /// `n_unique` rows must fit identically to the same `n_unique` rows each
    /// duplicated once (weights unset) — Σwᵢ·devᵢ over the unique rows equals
    /// Σdevᵢ over the doubled rows, so the two objectives share an argmin.
    /// Returns `((x, y, weights=2, n, ids), (x2, y2, n2=2n, ids2))`.
    #[allow(clippy::type_complexity)]
    fn build_sparse_weighted_replication_case(
        family: Family,
        seed: u64,
    ) -> (
        (Vec<f64>, Vec<f64>, Vec<f64>, usize, crate::GroupIds),
        (Vec<f64>, Vec<f64>, usize, crate::GroupIds),
        usize,
        ModelSpec,
    ) {
        let n = 60;
        let p = 2;
        let mut st = seed;
        let n_primary = 6usize;
        let extra_levels = [3usize, 4, 3, 5, 3, 4, 3];
        let u_c: Vec<f64> = (0..n_primary)
            .map(|_| 0.3 * super::test_lcg(&mut st))
            .collect();
        let v_e: Vec<Vec<f64>> = extra_levels
            .iter()
            .map(|&l| (0..l).map(|_| 0.2 * super::test_lcg(&mut st)).collect())
            .collect();
        let pid: Vec<u32> = (0..n).map(|i| (i % n_primary) as u32).collect();
        let extra: Vec<Vec<u32>> = extra_levels
            .iter()
            .enumerate()
            .map(|(g, &l)| (0..n).map(|i| ((i / (g + 1)) % l) as u32).collect())
            .collect();
        let mut x = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let cov = 0.3 * super::test_lcg(&mut st);
            x[i * p] = 1.0;
            x[i * p + 1] = cov;
            let mut eta = 0.3 + 0.4 * cov + u_c[pid[i] as usize];
            for (g, ids_g) in extra.iter().enumerate() {
                eta += v_e[g][ids_g[i] as usize];
            }
            y[i] = match family {
                Family::Poisson { .. } => {
                    let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
                    (eta.exp() * jit).round().max(0.0)
                }
                Family::Gamma { .. } => {
                    let jit = 1.0 + 0.3 * super::test_lcg(&mut st);
                    (eta.exp() * jit).max(0.05)
                }
                Family::NegativeBinomial { .. } => {
                    // Count-like data around exp(η), one-sided jitter — same
                    // rationale as `build_glmm_case`'s NB arm: the test
                    // compares two fitters on the SAME data, so exact NB
                    // sampling is unnecessary.
                    let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
                    (eta.exp() * jit).round().max(0.0)
                }
                _ => unreachable!("Poisson/Gamma/NB cases only"),
            };
        }
        let model = ModelSpec {
            family,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_primary as u32,
                },
                slopes: vec![],
                extra_groupings: extra_levels
                    .iter()
                    .map(|&l| Grouping {
                        relation: GroupingRelation::Crossed {
                            n_clusters: l as u32,
                        },
                        slopes: vec![],
                    })
                    .collect(),
            }),
        };
        let ids = crate::GroupIds {
            primary: pid.clone(),
            extra: extra.clone(),
        };
        // Row-doubled twin: literal concatenation of every per-row vector with
        // itself, weights unset (all-1) — same data, same likelihood as the
        // unique rows under w=2.
        let mut x2 = x.clone();
        x2.extend_from_slice(&x);
        let mut y2 = y.clone();
        y2.extend_from_slice(&y);
        let mut pid2 = pid.clone();
        pid2.extend_from_slice(&pid);
        let extra2: Vec<Vec<u32>> = extra
            .iter()
            .map(|col| {
                let mut c2 = col.clone();
                c2.extend_from_slice(col);
                c2
            })
            .collect();
        let ids2 = crate::GroupIds {
            primary: pid2,
            extra: extra2,
        };
        let weights = vec![2.0; n];
        ((x, y, weights, n, ids), (x2, y2, 2 * n, ids2), p, model)
    }

    /// Integer prior weights = row replication (Poisson): `w = 2` on `n`
    /// unique rows fits identically to the same rows each duplicated once —
    /// Σwᵢ·devᵢ over unique rows equals Σdevᵢ over duplicated rows, so the
    /// two sparse PIRLS objectives share an argmin. Full β/SE/τ² equality
    /// (Poisson has no estimated dispersion, so nothing else differs between
    /// the two row counts). Tolerances mirror the dense-vs-sparse cross-check
    /// (sparse_glmm_matches_dense_glmm, sparse.rs:5735-5752): β 2e-3 rel, SE
    /// 2e-2 rel, τ² 2e-2 rel — two independent BOBYQA runs of same-argmin
    /// objectives, so the bound is optimizer-scatter-sized.
    #[test]
    fn sparse_weighted_poisson_matches_replicated() {
        let family = Family::Poisson {
            link: crate::PoissonLink::Log,
        };
        let ((xw, yw, w, nw, idsw), (xd, yd, nd, idsd), p, model) =
            build_sparse_weighted_replication_case(family, 601);
        let opts_w = crate::FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..crate::FitOptions::default()
        };
        let opts_d = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        let fw = crate::fit_cold(&xw, &yw, nw, p, &model, &idsw, &opts_w);
        let fd = crate::fit_cold(&xd, &yd, nd, p, &model, &idsd, &opts_d);
        assert!(fw.converged && fd.converged, "both fits must converge");
        for j in 0..p {
            assert!(
                (fw.beta[j] - fd.beta[j]).abs() < 2e-3 * (1.0 + fd.beta[j].abs()),
                "β[{j}] weighted={} replicated={}",
                fw.beta[j],
                fd.beta[j]
            );
            assert!(
                (fw.se[j] - fd.se[j]).abs() < 2e-2 * (1.0 + fd.se[j].abs()),
                "se[{j}] weighted={} replicated={}",
                fw.se[j],
                fd.se[j]
            );
        }
        assert_eq!(fw.tau2.len(), fd.tau2.len());
        for (a, b) in fw.tau2.iter().zip(fd.tau2.iter()) {
            assert!(
                (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                "τ²: weighted={a} replicated={b}"
            );
        }
    }

    /// Gamma twin of `sparse_weighted_poisson_matches_replicated`. Asserts
    /// β/τ² only, NOT SE/dispersion: Gamma's Pearson φ̂ divides by raw `n−p`
    /// df (mirroring `glm(weights=)`/`glmer(weights=)`), and `n` differs
    /// between the weighted (n rows) and replicated (2n rows) encodings, so
    /// φ̂ — and every SE that scales with it — is NOT expected to match
    /// between the two, even though the likelihood/argmin is identical.
    #[test]
    fn sparse_weighted_gamma_matches_replicated() {
        let family = Family::Gamma {
            link: crate::GammaLink::Log,
        };
        let ((xw, yw, w, nw, idsw), (xd, yd, nd, idsd), p, model) =
            build_sparse_weighted_replication_case(family, 607);
        let opts_w = crate::FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..crate::FitOptions::default()
        };
        let opts_d = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        let fw = crate::fit_cold(&xw, &yw, nw, p, &model, &idsw, &opts_w);
        let fd = crate::fit_cold(&xd, &yd, nd, p, &model, &idsd, &opts_d);
        assert!(fw.converged && fd.converged, "both fits must converge");
        for j in 0..p {
            assert!(
                (fw.beta[j] - fd.beta[j]).abs() < 2e-3 * (1.0 + fd.beta[j].abs()),
                "β[{j}] weighted={} replicated={}",
                fw.beta[j],
                fd.beta[j]
            );
        }
        assert_eq!(fw.tau2.len(), fd.tau2.len());
        for (a, b) in fw.tau2.iter().zip(fd.tau2.iter()) {
            assert!(
                (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                "τ²: weighted={a} replicated={b}"
            );
        }
    }

    /// Prior-weight deviance anchor: at the SAME (θ, β) probe points, the
    /// aggregated fit with `prior_w = m` and the expanded Bernoulli fit produce
    /// deviances differing by exactly the data-only saturated constant (the
    /// penalty and log|A| terms coincide — aggregated W̃ᵢ = mᵢ·W̃ scatters the
    /// identical M'W̃M). Tight bound: same arithmetic up to summation order.
    #[test]
    fn sparse_weighted_binomial_deviance_matches_expanded() {
        use faer::Mat;
        let ((xa, ya, wa, n_a, ids_a), (xe, ye, n_e, ids_e), p, model, sat) =
            build_binomial_weighted_pair();
        let family = Family::Binomial {
            link: crate::BinomialLink::Logit,
        };
        let g_a = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n_a, &[], &[]);
        let mut ws_a = super::SparseGlmmWorkspace::new(&g_a, &ids_a.primary, &ids_a.extra, n_a, p);
        ws_a.prior_w[..n_a].copy_from_slice(&wa);
        let g_e = crate::lmm::LmmGroupings::from_cluster_spec_ext(&model, n_e, &[], &[]);
        let mut ws_e = super::SparseGlmmWorkspace::new(&g_e, &ids_e.primary, &ids_e.extra, n_e, p);
        let xam = Mat::<f64>::from_fn(n_a, p, |i, j| xa[i * p + j]);
        let xem = Mat::<f64>::from_fn(n_e, p, |i, j| xe[i * p + j]);
        // (θ×8, β×2) probe points spanning small/moderate RE scales.
        for params in [
            [0.4f64, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.3, 0.5],
            [0.8, 0.2, 0.5, 0.3, 0.6, 0.2, 0.4, 0.7, -0.2, 0.8],
        ] {
            let da = super::sparse_glmm_deviance(
                family,
                f64::NAN,
                &params,
                &mut ws_a,
                xam.as_ref(),
                &ya,
                n_a,
                false,
            );
            let de = super::sparse_glmm_deviance(
                family,
                f64::NAN,
                &params,
                &mut ws_e,
                xem.as_ref(),
                &ye,
                n_e,
                false,
            );
            assert!(
                da.is_finite() && de.is_finite(),
                "params={params:?}: both finite"
            );
            assert!(
                ((da - de) - sat).abs() < 1e-8 * (1.0 + de.abs()),
                "params={params:?}: agg {da} vs exp {de}, sat {sat}"
            );
        }
    }

    /// Prior-weight fit-level check through the stable surface: `fit_cold` on
    /// the aggregated rows with `FitOptions::weights = Some(m)` matches the
    /// expanded Bernoulli fit on β/SE/τ² for both `WaldSe` arms. Two
    /// independent BOBYQA runs of same-argmin objectives, so the bound is
    /// optimizer-scatter-sized (the deviance test above is the tight anchor).
    #[test]
    fn sparse_weighted_binomial_fit_matches_expanded() {
        let ((xa, ya, wa, n_a, ids_a), (xe, ye, n_e, ids_e), p, model, _sat) =
            build_binomial_weighted_pair();
        for wald_se in [crate::WaldSe::Hessian, crate::WaldSe::Rx] {
            let opts_e = crate::FitOptions {
                target_indices: vec![0, 1],
                wald_se,
                ..crate::FitOptions::default()
            };
            let fe = crate::fit_cold(&xe, &ye, n_e, p, &model, &ids_e, &opts_e);
            let opts_a = crate::FitOptions {
                target_indices: vec![0, 1],
                wald_se,
                weights: Some(wa.clone()),
                ..crate::FitOptions::default()
            };
            let fa = crate::fit_cold(&xa, &ya, n_a, p, &model, &ids_a, &opts_a);
            let tag = format!("{wald_se:?}");
            assert!(
                fe.converged && fa.converged,
                "{tag}: both fits must converge"
            );
            for j in 0..p {
                assert!(
                    (fa.beta[j] - fe.beta[j]).abs() < 2e-3 * (1.0 + fe.beta[j].abs()),
                    "{tag} β[{j}]: agg={} exp={}",
                    fa.beta[j],
                    fe.beta[j]
                );
                assert!(
                    (fa.se[j] - fe.se[j]).abs() < 2e-2 * (1.0 + fe.se[j].abs()),
                    "{tag} se[{j}]: agg={} exp={}",
                    fa.se[j],
                    fe.se[j]
                );
            }
            assert_eq!(fa.tau2.len(), fe.tau2.len(), "{tag}: tau2 length");
            for (a, b) in fa.tau2.iter().zip(fe.tau2.iter()) {
                assert!(
                    (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                    "{tag} tau2: agg={a} exp={b}"
                );
            }
        }
    }

    /// Over-envelope non-Gaussian smoke: genuinely over-cap designs with real
    /// signal CONVERGE through the routed `fit_cold` path (an upgrade over the
    /// prior anti-panic floor, which merely returned
    /// non-converged NaN). Over-count binomial/Poisson (7 crossed extras) and
    /// over-width gamma (one q_g = 5 slope-block extra) — the two shapes the
    /// parity rungs use. External truth is the parity datasets; this is
    /// the in-crate convergence gate.
    #[test]
    fn sparse_glmm_over_envelope_converges() {
        // Over-count: y ~ 1 + x + (1|g1) + (1|c1) + … + (1|c7).
        let n = 210;
        let p = 2;
        let mut st = 301u64;
        let n_primary = 6usize;
        let u_c: Vec<f64> = (0..n_primary)
            .map(|_| 0.5 * super::test_lcg(&mut st))
            .collect();
        let extra_levels = [3usize, 4, 3, 5, 3, 4, 3];
        let v_e: Vec<Vec<f64>> = extra_levels
            .iter()
            .map(|&l| (0..l).map(|_| 0.3 * super::test_lcg(&mut st)).collect())
            .collect();
        let mut x = vec![0.0f64; n * p];
        let mut eta = vec![0.0f64; n];
        let pid: Vec<u32> = (0..n).map(|i| (i % n_primary) as u32).collect();
        let extra: Vec<Vec<u32>> = extra_levels
            .iter()
            .enumerate()
            .map(|(g, &l)| (0..n).map(|i| ((i / (g + 1)) % l) as u32).collect())
            .collect();
        for i in 0..n {
            let cov = super::test_lcg(&mut st);
            x[i * p] = 1.0;
            x[i * p + 1] = cov;
            let mut e = 0.3 + 0.5 * cov + u_c[pid[i] as usize];
            for (g, ids_g) in extra.iter().enumerate() {
                e += v_e[g][ids_g[i] as usize];
            }
            eta[i] = e;
        }
        let mk_model = |family: Family| ModelSpec {
            family,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_primary as u32,
                },
                slopes: vec![],
                extra_groupings: extra_levels
                    .iter()
                    .map(|&l| Grouping {
                        relation: GroupingRelation::Crossed {
                            n_clusters: l as u32,
                        },
                        slopes: vec![],
                    })
                    .collect(),
            }),
        };
        let ids = crate::GroupIds {
            primary: pid,
            extra,
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        // Binomial.
        let yb: Vec<f64> = eta
            .iter()
            .map(|&e| {
                let pr = 1.0 / (1.0 + (-e).exp());
                let uni = 0.5 * (super::test_lcg(&mut st) + 1.0);
                if uni < pr {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let model = mk_model(Family::Binomial {
            link: crate::BinomialLink::Logit,
        });
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ));
        let f = crate::fit_cold(&x, &yb, n, p, &model, &ids, &opts);
        assert!(f.converged, "over-count binomial converges");
        assert!(f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()));
        // Poisson.
        let yp: Vec<f64> = eta
            .iter()
            .map(|&e| {
                let jit = 1.0 + 0.4 * super::test_lcg(&mut st);
                (e.exp() * jit).round().max(0.0)
            })
            .collect();
        let model = mk_model(Family::Poisson {
            link: crate::PoissonLink::Log,
        });
        let f = crate::fit_cold(&x, &yp, n, p, &model, &ids, &opts);
        assert!(f.converged, "over-count poisson converges");
        assert!(f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()));

        // Over-width: y ~ 1 + x1..x4 + (1|gp) + (1 + x1..x4 | ge), gamma/log.
        let n = 240;
        let p = 5;
        let n_gp = 8usize;
        let n_ge = 6usize;
        let q_g = 5usize;
        let u_gp: Vec<f64> = (0..n_gp).map(|_| 0.4 * super::test_lcg(&mut st)).collect();
        let v_ge: Vec<f64> = (0..n_ge * q_g)
            .map(|_| 0.25 * super::test_lcg(&mut st))
            .collect();
        let mut x = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        let gp: Vec<u32> = (0..n).map(|i| (i % n_gp) as u32).collect();
        let ge: Vec<u32> = (0..n).map(|i| ((i / 2) % n_ge) as u32).collect();
        for i in 0..n {
            x[i * p] = 1.0;
            for j in 1..p {
                x[i * p + j] = super::test_lcg(&mut st);
            }
            let l = ge[i] as usize;
            let mut e = 0.5 + u_gp[gp[i] as usize] + v_ge[l * q_g];
            for j in 1..p {
                e += (0.4 + v_ge[l * q_g + j]) * x[i * p + j];
            }
            let jit = 1.0 + 0.3 * super::test_lcg(&mut st);
            y[i] = (e.exp() * jit).max(0.05);
        }
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_gp as u32,
                },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: n_ge as u32,
                    },
                    slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q ⇒ over-width
                }],
            }),
        };
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse
        ));
        let ids = crate::GroupIds {
            primary: gp,
            extra: vec![ge],
        };
        let opts = crate::FitOptions {
            target_indices: vec![0, 1, 2, 3, 4],
            ..crate::FitOptions::default()
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(f.converged, "over-width gamma converges");
        assert!(f.beta.iter().all(|b| b.is_finite()) && f.se.iter().all(|s| s.is_finite()));
    }

    /// Task 6: weighted sparse Gaussian LMM. `parity/manifest.json` has no
    /// sparse-Gaussian rung (only sparse binomial/poisson/gamma/nb), so this
    /// fixture is generated directly in R rather than pulled from a committed
    /// parity dataset, and pinned against a frozen lme4 golden (not routed
    /// through the parity oracle harness). Design: primary `(1|g1)` (20
    /// levels) crossed with an extra grouping that carries a random SLOPE
    /// `(1+x2|g2)` (15 levels) — `classify_design`'s `slope_extras` clause
    /// routes any slope-carrying extra grouping Sparse regardless of size,
    /// so this over-envelope classification is a structural property of the
    /// design, not a size accident. Weights `w_i = 1 + (i mod 3)` (0-based
    /// CSV row order), same convention as the dense Task 5 golden. Generated
    /// with (R 4.5.3, lme4 1.1-38):
    /// ```r
    /// library(lme4)
    /// set.seed(2026)
    /// n <- 400; n_g1 <- 20; n_g2 <- 15
    /// g1 <- rep(0:(n_g1 - 1), each = n / n_g1)
    /// g2 <- (seq_len(n) - 1) %% n_g2
    /// x1 <- rnorm(n); x2 <- rnorm(n)
    /// b1 <- rnorm(n_g1, sd = 1.5)
    /// b2_int <- rnorm(n_g2, sd = 1.0); b2_slope <- rnorm(n_g2, sd = 0.8)
    /// y <- 2 + 0.7 * x1 + 1.2 * x2 + b1[g1 + 1] + b2_int[g2 + 1] +
    ///      b2_slope[g2 + 1] * x2 + rnorm(n, sd = 1.0)
    /// w <- 1 + (seq_len(n) - 1) %% 3
    /// d <- data.frame(y, x1, x2, g1, g2)
    /// f <- lmer(y ~ x1 + x2 + (1 | g1) + (1 + x2 | g2), data = d,
    ///           weights = w, REML = TRUE)
    /// print(summary(f)$coefficients, digits = 15)
    /// print(as.data.frame(VarCorr(f)), digits = 15)
    /// print(sigma(f), digits = 15); print(REMLcrit(f), digits = 15)
    /// ```
    #[test]
    fn fit_sparse_lmm_weighted_matches_lme4() {
        const REF_B0: f64 = 2.266552572193687;
        const REF_B1: f64 = 0.638481148260981;
        const REF_B2: f64 = 0.962298890765967;
        const REF_SE0: f64 = 0.5077453454635866;
        const REF_SE1: f64 = 0.0513758552443655;
        const REF_SE2: f64 = 0.2352975102494342;
        const REF_SD_G1: f64 = 1.91959909459237; // g1 (Intercept) sd
        const REF_SD_G2_INT: f64 = 1.02899506510311; // g2 (Intercept) sd
        const REF_SD_G2_SLOPE: f64 = 0.88391047396846; // g2 x2 sd
        const REF_CORR_G2: f64 = -0.28713485636041; // g2 (Intercept)~x2 corr
        const REF_REMLCRIT: f64 = 1331.89208648957;

        // y,x1,x2,g1,g2 — printed by the R generator above, row order preserved.
        const CSV: &str = "\
2.26569271686515,0.520589072918523,1.33689257455037,0,0
-1.5186057959394,-1.07969076235228,0.552710403069674,0,1
-2.9811587236883,0.139238115019273,-1.05652641739161,0,2
-0.0601500700332526,-0.0847487849485765,-0.190182994643526,0,3
-4.81432513086854,-0.666639615284596,-0.909224176086877,0,4
-4.43453898627614,-2.51608903200946,-2.00091550631878,0,5
0.589552441301372,-0.735146797456677,1.90897086539068,0,6
1.92594155455609,-1.02012226313509,1.56898020541778,0,7
-0.309667203228029,0.113554441297307,0.140301225311454,0,8
-4.018794602209,-0.473790981840095,-1.42055946460262,0,9
-0.85804375253163,-0.408214704337928,0.513066799512409,0,10
2.19246042100517,-0.730433278593614,-0.987461198891432,0,11
0.0396200192993124,-0.221436599406174,0.597223916449456,0,12
-0.102030816413376,-0.225816524428951,0.867860400604364,0,13
-2.10851820990764,-2.5468814461283,0.189257763959423,0,14
3.49866965048435,1.34700149929703,-0.0693064069135194,0,0
-1.83351937516147,0.616408145849969,1.07625933935365,0,1
0.389550433600885,0.217564338307888,1.07089132376137,0,2
1.06757985063778,-0.804718830400263,0.421633377427644,0,3
-0.691576583717045,0.68974677762582,-0.12454978063033,0,4
2.38099780779407,-0.32867201409024,0.937192065289507,1,5
1.35114821662376,-0.16468157692584,0.200619126393445,1,6
1.7463285772661,-1.3920288797713,1.44817884292342,1,7
0.774296241405956,1.46582476120139,-1.28218256537373,1,8
3.2331770597448,0.0482068082254438,-0.084554210547583,1,9
3.63842951064809,1.90808383464199,1.10161395316682,1,10
3.80992041934262,1.73094452723732,-0.690699575305395,1,11
-0.422150051089063,0.0581458372592962,-1.72541538917727,1,12
2.95748859539586,0.645328161775171,0.565885692360889,1,13
-0.298134815037274,1.7256289865063,-0.596158417224681,1,14
3.87325259548931,-0.528966917456246,-0.489345024258458,1,0
1.92778170805841,0.166392025612551,0.768893360822469,1,1
3.20449145774661,-0.254723574763946,2.11461189033211,1,2
0.694748017765141,0.332782359437909,0.0447809835460714,1,3
2.9048257296396,0.182432812827635,0.947241069815817,1,4
0.459408074564175,1.164593749518,-2.1245087083504,1,5
1.01977812148409,0.593492850727868,0.0904283264577256,1,6
-3.17149145518441,-0.891779729217369,-1.73914879260394,1,7
-0.507176140710402,0.577253310357403,-1.7022883303277,1,8
5.25987772555658,-0.824905581058043,1.40829639529108,1,9
1.59371483138849,-1.15725385762344,0.69816811272235,2,10
3.93299040680764,0.777998774006214,0.5760854391738,2,11
-0.882950614081925,-1.20422178597307,-0.534336060471033,2,12
-1.20422221508216,0.30671410214768,-1.28198635078153,2,13
1.57549041393497,-0.833658540642056,0.429614218440963,2,14
5.79938991056964,1.41814116350375,-1.12943767906222,2,0
-1.58622346009505,0.711786117850718,2.41976367943378,2,1
-1.9262839558748,-0.402497722862373,0.0782761044809724,2,2
1.99934979881041,0.799362218714509,0.316225234142966,2,3
-4.39468388516595,0.426484435556948,-1.5794002086685,2,4
0.627053096634319,-1.16989866993562,0.518905620032067,2,5
0.827790744616347,-0.206200160293019,1.13539238442515,2,6
-0.787038368825956,-0.930610309243612,0.312404463905503,2,7
1.14738778062059,0.449397565791737,0.05554413008899,2,8
4.42713095224106,-0.644806454562313,1.56545282333373,2,9
1.81780019603085,-0.231422689810958,0.718830766147177,2,10
2.27148869810909,-1.23636801144537,0.634325711079806,2,11
1.25020193047794,-0.960955298223015,1.06590668950288,2,12
-1.17150955810353,0.133956281569561,0.341888554248612,2,13
0.352688193209334,-0.999052722969284,3.14336714054499,2,14
3.40686437894807,-0.141470690813213,0.349950832519039,3,0
-0.471635823978737,0.167329899648813,1.07333545251137,3,1
1.00554687919145,-0.198788612671681,0.916437872494606,3,2
3.50810764792225,-0.291207188092097,0.71765690362959,3,3
0.658857424220627,-1.73439703424929,1.05548502974692,3,4
-0.0904474330216343,-0.272728331732604,0.350592199470931,3,5
-5.09523064290751,-1.79992948169682,-1.93603847989057,3,6
-1.02846795703,1.15274097680304,-0.800741941084278,3,7
0.586483899298131,-1.003319485592,2.34293958213089,3,8
5.68295765735492,0.148210044292539,1.92087838946788,3,9
-0.121531734602508,0.519496680749176,-0.109538639946886,3,10
-0.826802938504754,0.00543629447576128,-2.51282078633865,3,11
0.226562421442393,1.34702083465394,-0.759623633062425,3,12
-1.58569819123598,-0.847033417295996,-1.50546286281683,3,13
-0.779971461417941,0.443398017315772,-1.67824570835886,3,14
0.294002162100748,-0.977149468323314,0.950098626350105,3,0
1.9322911155916,2.12449113551361,0.124953593318981,3,1
2.27730963144657,0.687698960561541,1.43839197208062,3,2
0.472084772409255,-0.343368220180396,-1.32535121599528,3,3
-0.800181337311174,0.785169472673946,-1.00393459317338,3,4
1.71145531996263,-1.15557746077052,0.642208565811121,4,5
5.994760052792,1.5089163463927,1.15655108036238,4,6
2.76846211305985,-1.15549616025736,-0.173573480005486,4,7
-0.190503450643527,-1.55681181020668,-0.964569098629924,4,8
7.36586891209075,-0.0552479252235566,1.85379901629885,4,9
5.46590601134132,0.849193162702715,1.68436511079776,4,10
6.23279004273887,-0.0110967225036592,1.05731214066179,4,11
3.24623705278299,-0.760313831780719,0.814561626549095,4,12
2.35336230811426,1.17579925167624,-1.19459693410643,4,13
4.33623961351255,2.41444970903476,-2.02885658613829,4,14
1.62715060166342,-1.95851276822613,2.12408653976063,4,0
2.50896885107831,1.4735898934298,0.299219062813124,4,1
0.54061021870137,-0.47488396149873,-0.824643051149339,4,2
5.46747558802284,0.981170983785874,1.52434567788945,4,3
-0.80736289054826,-1.82435521953656,-0.614820405730783,4,4
3.1214776049804,-0.260610817352268,-0.45976066348376,4,5
2.39631445252684,-0.95884531257366,0.443640987583446,4,6
0.0746993657381853,-0.490295710066525,-1.39598147444982,4,7
1.35084625343769,-1.07729671360066,-0.502477861628173,4,8
2.4791449476834,0.369425910544446,-1.06059878223795,4,9
3.98134052679875,1.21626648113935,1.17460715763047,5,10
2.95118815646248,-0.493843235614103,0.330371459656535,5,11
0.702861116552784,-0.227784720977188,-0.288433220566114,5,12
3.24685893560117,-1.11164888704727,1.32156210264278,5,13
1.29530044172321,0.995833089409624,-0.0454740273538609,5,14
4.14453044490995,0.561828890946087,-0.214793901616877,5,0
-1.45398705653092,-1.56427855651588,0.13480451241815,5,1
1.31399034047831,1.48107701910531,-0.644997500085118,5,2
1.43520607046579,-1.16920515011223,0.0570719155291017,5,3
3.67137424233706,-1.05201353746636,0.931184671545328,5,4
-1.0977512841226,-1.47184552482902,0.203663452276264,5,5
1.2509943559665,-1.19343692076419,1.62579612344971,5,6
-1.05167159879544,-0.347979424691937,-0.593355958720706,5,7
1.21793715924739,0.532822543149418,-0.308569082055948,5,8
3.67785038705233,-0.710988575850843,1.33644832852931,5,9
-0.825819371934104,-0.198989137632404,-1.20646409452483,5,10
4.05892226705497,-0.281373323518464,-1.40888180770124,5,11
2.22897955694354,1.17092627520498,0.644424623754144,5,12
2.39599317340592,2.28873318592,-0.408867428984605,5,13
-1.6352668953481,-1.06885661449908,-0.979236369824986,5,14
9.38708524659219,1.90620418856242,-0.927231216419259,6,0
3.43041227648816,2.13179771847887,0.79270411853512,6,1
2.01283540814053,0.231456363015352,-0.68909903597529,6,2
5.53768423353945,0.896945737827196,-0.0416516404215562,6,3
0.639810178440599,-1.73879271235152,-0.785618777019648,6,4
5.07122818829766,0.468847940135532,0.4344554586641,6,5
3.62873601312113,-0.544147673197739,-0.665566901846607,6,6
2.59485468781172,-0.165414153808447,0.449936130069846,6,7
2.70892697929705,0.552166562572813,-0.548040463452155,6,8
8.39262686140589,1.0333302116035,1.85693691208123,6,9
5.16427669880862,-0.0461788047835456,0.40207812111575,6,10
5.32354880498233,2.63870731259579,-1.86460318556058,6,11
4.97392739403681,0.589005749225018,1.37493052047263,6,12
3.23165006697256,-0.202377334229212,-0.0746035055403732,6,13
2.43005548122974,0.441560138002382,0.22779389971164,6,14
2.91105803005144,-0.100257702082834,1.38865619323333,6,0
2.95785821542996,-1.09340399199111,0.405822127350406,6,1
5.37220266451377,0.50324733858265,2.56136759868205,6,2
5.30300377418895,0.949240091117635,-0.526334569793747,6,3
2.7779627219107,0.382802056022312,0.352407393770692,6,4
2.82445985644845,0.371578210898053,0.475161141291331,7,5
3.44655441614692,0.157216401651425,0.791905108509305,7,6
3.08578795988607,0.847939889975375,0.831271029700769,7,7
1.26774946058887,-1.28339265650518,0.548697929846243,7,8
3.41203124650436,1.1582786080896,-1.24227498092802,7,9
2.4643441629754,-0.909106213165988,-0.413370331843489,7,10
4.1998551273158,0.334990705168581,0.0710386853613732,7,11
2.65106770610979,0.775854081916407,-0.494271818071687,7,12
3.30351154166187,0.237140152276008,0.322143267993693,7,13
2.27839323409679,-1.5598701345872,-0.796429134375704,7,14
5.47131261424797,0.0243267511888821,0.907116281386659,7,0
3.2402748889524,0.334635189158287,0.207539150347981,7,1
4.43736779623431,0.989410999147139,0.547842405004734,7,2
3.28381655339623,0.500375113541597,-0.22169528429994,7,3
1.46555566233963,0.576907865455451,-0.681797182178005,7,4
4.92305361813861,1.14269094011911,1.45985706973908,7,5
2.1163686791436,-0.707447814422978,-0.862507887588664,7,6
0.00643484611127254,-0.618559640981128,-0.800383855157766,7,7
1.04755524045436,-0.0083141961424557,-0.68514196580317,7,8
7.1062184037943,0.338919733952689,1.82132263410678,7,9
3.57533325210901,1.40540629884683,-0.104744220596189,8,10
5.64779425600001,-0.865269852362635,1.11147452679476,8,11
-0.891668553283255,-0.873953956620092,-1.02752209427414,8,12
0.356549447167564,0.7743182008249,-0.424123583125695,8,13
0.82592984745751,-0.401890241517252,-0.159149418177784,8,14
2.3005213649307,-0.215137830535894,0.658170841972663,8,0
2.41398792663423,0.605256130514011,-0.303464556482093,8,1
1.84386315529868,-0.614567683647475,0.696562801882666,8,2
2.18969036284595,-0.724422931424412,0.112772261493095,8,3
-0.0380423847239506,-0.256942637181777,0.180256561398902,8,4
4.69730769943493,-0.392072736349559,2.1212044307174,8,5
0.903192622928767,-0.686701961744657,-0.67724515768195,8,6
2.58575362837787,0.875840340130818,0.975279636884236,8,7
2.64125275066828,-0.481057986042492,1.09733861872641,8,8
5.84795221822321,0.0683575908680962,1.66738580380783,8,9
2.62193240171985,0.137024892815848,0.324119853477316,8,10
3.8563345261667,-1.85876851887058,0.250784954668287,8,11
3.43108823810242,0.34627657136052,0.153309143680586,8,12
2.27348663149062,1.50163381994693,0.715229305996022,8,13
4.32934407502284,-0.0137381203294002,0.610541619161322,8,14
5.90186898868146,-0.77524519751103,-0.281816891217829,9,0
5.01253698699677,1.3111581356762,0.885998134265093,9,1
5.31144205413505,0.260524556554842,0.594126927122837,9,2
6.19481710543531,1.01401700762575,0.700118281593645,9,3
11.9559811275874,0.215594208474653,2.81491279503349,9,4
5.2303358565521,0.613404626091988,0.277078099153463,9,5
5.15424953941641,1.44598848106203,-0.513741244477688,9,6
7.79144654070019,0.658085900745181,0.422214152789072,9,7
4.24658462605188,0.375234496491491,-0.265564340886777,9,8
5.08617950518273,-0.674688275938179,-0.770354222150151,9,9
1.94542169274344,0.580480573296274,-0.376500645612233,9,10
6.64787716742674,-0.414229332226612,0.110222930784873,9,11
6.80681545552938,2.18050398539242,0.972561427059444,9,12
3.31209546770424,0.342857643650257,0.356994016994474,9,13
2.32671707034836,-0.798024661907156,-0.824167316178698,9,14
7.21935671354296,1.64103616625365,-0.155630678991287,9,0
3.54361245488635,-0.460005111651252,-1.52971660161298,9,1
2.50288083851835,-1.58647098860618,-0.202227510837209,9,2
4.09482283397433,-1.04491567407179,-1.31151722791159,9,3
0.647744748214764,0.0416536658857592,-1.39007893871245,9,4
3.89151529857333,-0.43202637747734,0.962138141645872,10,5
1.13509005446349,0.458591147686799,-0.17813477901026,10,6
0.631617869951369,-0.229351986059321,0.00328960117506923,10,7
0.0684576631481358,-1.85696220984083,0.511615165517015,10,8
2.84919245138515,-0.289674172981153,-0.456060688239042,10,9
0.777461810452513,1.76714123786919,-0.21090411916793,10,10
2.40670685517177,-0.442682897300816,-0.572554868573356,10,11
0.380815972203836,-0.588864402001567,-0.353336222838735,10,12
-1.45935706620346,-0.120588258352901,-1.0496560241288,10,13
1.12096438102176,1.65306128186181,0.7883855209797,10,14
3.48842714061664,-0.871354568629743,0.619231580144126,10,0
2.56978568807385,0.780668270857782,-0.785768492164587,10,1
1.58309604180605,-0.613877869318662,1.78160405193807,10,2
2.12500448831244,-0.327591311306049,-0.249928557425748,10,3
-0.625176013058818,-0.355216015901553,-0.301097396588297,10,4
-0.988401769236405,0.78093407436684,-1.49933210991998,10,5
2.76616843242921,0.608670171785426,0.0342053478098827,10,6
-0.981172655493283,1.07617901747348,-0.911047222176761,10,7
1.93077229615646,1.06555765231322,0.351385587315867,10,8
5.2258968883385,2.16164641282926,0.195414362215204,10,9
5.01422594382777,0.0704564820522911,0.859629926713806,11,10
1.39621745569964,-2.53522521959317,-0.674155138589608,11,11
1.55940430075499,-0.541334144723391,-1.66258565149713,11,12
4.18126886782737,-0.775939573541146,0.390045780841936,11,13
2.19201867024286,-0.295955418239763,0.441365682409681,11,14
3.35019324418225,1.15137290542338,1.28396592434211,11,0
1.17910922211284,-0.568887867531516,-0.163185801701405,11,1
1.37144283995466,-0.805786340025032,0.499569043121688,11,2
4.68883410330573,0.492593861841638,0.460144067064487,11,3
2.49861156696724,0.700758337332243,-0.476282040022087,11,4
0.297125233019732,-0.243125291550084,-1.55664239038557,11,5
1.65586952413087,0.412481176922476,-0.689022539113182,11,6
2.20939174396122,0.498165309358799,0.148407389086632,11,7
3.60768937977397,0.367679114623033,0.135429490904212,11,8
0.473503070559156,-0.5706028691897,-1.37234184047492,11,9
2.65739735938482,0.738816517714368,-0.0931036773678838,11,10
1.09549858311319,-2.26829132566495,-1.11820793439463,11,11
4.34079101292021,0.614041040052466,0.744827357549737,11,12
2.94154761353848,0.40363580599992,0.0776825859578852,11,13
2.5789199169556,2.05157932669931,-0.499928792474106,11,14
7.15562877125623,0.263244987215003,0.611743478942702,12,0
7.52942712893248,0.110421189446662,-0.869034775516044,12,1
7.53658870103454,1.1957636212874,1.16615799034312,12,2
7.69875663943353,0.733309513633103,-0.769731638856476,12,3
3.53349570030228,-0.00451169154164245,-1.80724436689568,12,4
7.75599098760304,-0.0212154995950339,-0.917465740651156,12,5
8.29492652392404,0.138501732981574,-0.356456664570957,12,6
5.77118392614591,-1.07012706995328,-0.176553706785467,12,7
6.51894024206579,1.23606206697686,-0.799311276346493,12,8
8.37182081239118,-0.539810094735419,-0.392796425823479,12,9
8.06156180082909,-0.296228914656068,0.273725082324397,12,10
10.9751035432975,0.688218803349637,0.258141982501913,12,11
5.49644579149136,-0.399007303727868,0.110412063687764,12,12
4.76059001011066,-0.103484852849487,-0.915161122522455,12,13
6.89678540395981,1.52953404059542,0.212936649900559,12,14
7.84927463172794,-0.634636458448834,1.40333534124712,12,0
5.64298472650673,-0.226854589097728,-0.808434808833702,12,1
10.5775268684518,1.5010840927328,1.91953356242643,12,2
6.97098329237163,-0.342463482378634,0.865668884307897,12,3
3.36156799508795,-0.553727556011483,-1.2639077211064,12,4
0.378320492608427,0.257626870698959,-0.956625877294363,13,5
2.02271739165599,0.613121758652457,-0.303370061193043,13,6
2.50087308274843,-0.652287366777574,-0.00220360490056386,13,7
0.331516429772245,-0.138281029086895,-0.913071354975655,13,8
2.1759599046682,-0.123638001869199,-0.649643176443167,13,9
0.935698838684268,-0.964705216383371,0.376311077588093,13,10
4.76027228500782,0.0190431840153162,0.2779963826487,13,11
3.08274517152035,1.69169235319906,-0.754720890372454,13,12
0.362265554013126,0.885809071474169,-0.473084717719469,13,13
-0.209892881545623,-0.729930145482526,-0.91395042710506,13,14
5.29010757986385,0.351884683098013,-2.60329970942661,13,0
2.86222384533804,-0.247330893260195,-0.593467498200933,13,1
0.541305855262702,-0.163591831723322,-0.638979894573173,13,2
4.58747754254356,0.872360560208246,-0.105916956315576,13,3
-1.68077623742475,0.605938213883317,-0.980291154529057,13,4
3.34057025830932,0.927240812612819,0.476235459637523,13,5
2.15594421679729,0.458044436651209,-1.41239128440147,13,6
4.35417214217403,-0.801114024214896,1.63100315238839,13,7
0.929647833888533,-0.497123152995539,-1.12798588227457,13,8
4.14507827036828,-1.21296781485439,0.438451089522611,13,9
7.91086286973373,0.619712599356425,1.90065023437915,14,10
7.07793429487854,1.3596632579562,-0.700760424199841,14,11
3.04880350657769,-1.6210647627138,-0.3717393835228,14,12
3.38479824697546,0.0795594993555375,-0.678503248030561,14,13
2.47498174919331,0.512593137306753,0.101451284461551,14,14
5.67923212120527,-0.21910193815088,0.553279834051488,14,0
5.90169535140175,-0.82593869467054,-1.52222099582696,14,1
4.56353918278271,-1.01357824411184,-1.02154930182434,14,2
4.78511905042227,1.28206624602831,-1.39068213766217,14,3
4.26360932642697,-0.212144499179618,0.433339795920382,14,4
4.20876733341715,1.18487723662641,0.369698592138803,14,5
4.21124382466685,-0.650612436772287,-0.709493645627637,14,6
2.58569634407,1.28736727005853,-1.01894940268142,14,7
2.44114444578842,0.538933215663317,-1.64074625333368,14,8
6.97860927472029,0.0255405538039062,0.682710512141664,14,9
3.07432238994928,1.1639475327246,-0.751876713541874,14,10
8.43803748147673,-0.367625732117171,0.716236911745028,14,11
6.67354287901515,-0.363369569501376,1.99500140843909,14,12
5.85937193154085,-0.00675336474544712,1.34613637848412,14,13
4.17435054827609,-0.100472498864641,-0.826907119029068,14,14
1.09005031614737,-0.14859207836112,1.13846516751087,15,0
-2.83865032099968,-0.182952755436934,-0.670743488970344,15,1
-0.938325171293181,-0.85814783765375,0.28830533745626,15,2
1.94979856599394,0.958159182178004,0.163840249645937,15,3
-0.00341580540936931,-0.839076941832303,0.543705814431928,15,4
-1.26253096019206,-0.759724279311683,0.0089607826965823,15,5
-0.341783234965533,-0.866514819536214,0.29143972911296,15,6
1.48725828702367,1.65902566868139,0.632100222998882,15,7
-2.85197544813362,0.556942277210519,-2.12229472459214,15,8
3.31048872245074,0.601386129662215,-0.273116168728087,15,9
-2.83253555344797,-1.96978682110424,0.379269650648216,15,10
1.11096383170432,-0.380107748973682,0.207411582705335,15,11
1.54743343131236,1.15777354406233,0.130075482647751,15,12
0.416094923017226,1.38963465416294,1.7290152298527,15,13
-1.83423671122836,-2.01150842477619,-0.131777383691539,15,14
-0.108456295708925,-2.33630850037639,0.0901800389952223,15,0
-1.84243945652219,-1.36845977011138,-0.122655944893569,15,1
-1.19521946775632,-1.37309426035361,2.34798597248153,15,2
0.105926440769032,0.737388550085491,-0.62236219518714,15,3
2.34386701060351,0.454709341750177,1.09608012991909,15,4
5.1602502143626,1.77668132830334,1.13795870017689,16,5
3.44693311443338,0.194985588228136,-0.740714885875294,16,6
4.32823108923858,-0.344612622139577,1.23768243980547,16,7
3.52061151400087,1.61267663514805,0.589619597581652,16,8
5.8794048234845,0.775709363612466,1.00741836369821,16,9
0.15566753860005,0.70427912561433,-1.75300650648146,16,10
5.67209827449965,1.07215421601982,-0.773431753710342,16,11
2.52040974711728,-0.214130769685907,-0.386735295761461,16,12
3.41652904947139,0.363631063912172,0.188347531556599,16,13
4.04574049825908,1.08356378657411,-0.0619437868449765,16,14
3.11120645180015,-1.95227615716347,-0.148388226122151,16,0
1.98863932173461,0.0155101786160552,0.599105401299345,16,1
4.12865903208638,-0.87098092003781,0.913847832792144,16,2
3.36060928899094,-0.900934214910906,1.6666000600483,16,3
3.38226224255814,-0.776112067634224,0.603637972396393,16,4
1.50304186062726,0.361430751154015,-1.16201140949401,16,5
3.62454742547555,-0.0173493122643544,-1.37223198751365,16,6
5.51416981373375,-1.11819535979872,1.33404139102141,16,7
2.44199126032025,0.58669137484252,-0.393542558040908,16,8
5.49169300907432,0.486195009924034,0.353585138470633,16,9
-0.951818442783229,0.473894455796676,0.400055481683396,17,10
4.36670800464802,0.361427452891757,0.666740074189319,17,11
2.08786949709306,1.21768364371144,-0.420241256035585,17,12
2.59778611712379,-0.673116960131551,-0.328174632770276,17,13
0.670310546207943,-0.598848241837351,0.437923476479789,17,14
5.51612719379865,0.735873146511805,-0.784044326886472,17,0
1.56464658091896,0.60173965147488,-0.226837232126531,17,1
-0.516310093164733,1.10889193968727,-0.582210315071111,17,2
1.64550816362994,-0.0113905983860746,-1.098110347489,17,3
1.93506142881849,0.605035960687119,0.162352442701946,17,4
3.15528261004126,1.12352815317424,0.450619475819499,17,5
1.77978930877787,-0.705362053710043,0.0982995234973026,17,6
2.6668810620111,1.76247207167883,0.334055561690702,17,7
-0.210390495250704,-0.730020695494859,-0.0903722507191762,17,8
2.48061769643047,-1.63415762678176,-0.175437620600107,17,9
-3.05062592569791,-0.123993779923598,-1.57072135945584,17,10
5.74588350008261,0.204139470306424,1.21159302733537,17,11
2.11431080840813,-0.68323921942861,1.45793829178585,17,12
-1.53936155366522,-1.01636227170084,0.865075477914802,17,13
0.122737096929622,-0.671434165633787,0.589875474734274,17,14
4.02754464819057,1.12186281570044,0.0828203415601401,18,0
-3.89155121488193,-2.10066370498434,-0.403684784749441,18,1
-0.641225919377077,-2.09087185958204,0.193777916274814,18,2
0.968259515344114,-1.29110001012018,-0.553142679203068,18,3
2.67946939293739,1.07199151952253,0.70346433981999,18,4
1.09273451810304,0.894870152083501,0.397668729642592,18,5
-1.14510877567253,-0.842632298901112,0.0100055504835981,18,6
-2.74291672627674,0.0462974922741544,-0.255537690151586,18,7
-0.114157474872505,0.623368148702539,0.301799212469126,18,8
2.46194292180836,0.485091951868738,0.723584081288875,18,9
4.2638499717217,0.531944473401682,1.4151701539861,18,10
0.597612760000028,0.19742268076406,-1.37337950574091,18,11
-1.39896865485694,0.326566784462888,-0.599548070268292,18,12
1.14275226967094,-1.44023199151019,1.71015618024855,18,13
1.26818139246462,2.11470907987023,-0.163937529099855,18,14
-0.357599222211188,-0.184672313220123,1.60460144055646,18,0
0.595219788318556,0.492104327529081,-0.798674754667866,18,1
-1.09841049200863,-1.36132698556342,1.18037424884885,18,2
-0.690708526314822,-0.365217557903885,-0.859589831327811,18,3
-3.46994601445431,0.7561265592095,-0.86974967242439,18,4
1.44537791179307,0.146502808516189,0.295584565295028,19,5
0.320847946955593,0.261947248506149,-0.274837917019254,19,6
-1.67925440798133,-1.30615904372434,-1.11238074519984,19,7
1.00912714915343,0.492164032443828,-1.10611190648605,19,8
2.67352955812131,-0.511104000679042,0.308028588983761,19,9
-0.479845890329478,-1.66057454138599,-1.08568792211512,19,10
5.44120705815668,0.622376370866293,1.01683942966976,19,11
2.88984062323628,0.0157310112822735,1.29398257664835,19,12
-1.27063462328962,-0.733887125964384,-1.20692916422951,19,13
0.502941640703045,0.27521113796498,-1.74997090694174,19,14
4.65159670267425,-1.8587385910095,-2.19522755244751,19,0
1.61625280047006,2.07889036683578,0.780089945806557,19,1
-1.47686362308648,-2.36088032720629,-1.08546476672529,19,2
2.16827566267286,0.327135397295895,1.01612857543453,19,3
-1.37175699513399,-0.227522726278505,-0.21040661077906,19,4
-0.156059773478137,-1.18059531411805,-1.33866666877921,19,5
1.88241314709922,0.330421531559684,-0.876139679397935,19,6
1.50851097286275,-0.406934780043714,0.0471728881584331,19,7
3.69461149161872,1.52522324018498,2.01695237231075,19,8
2.6661495091834,-1.10318386829999,0.372699001110427,19,9
";

        let mut y = Vec::<f64>::new();
        let mut x1 = Vec::<f64>::new();
        let mut x2 = Vec::<f64>::new();
        let mut g1 = Vec::<u32>::new();
        let mut g2 = Vec::<u32>::new();
        for line in CSV.lines().filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').collect();
            y.push(f[0].parse().unwrap());
            x1.push(f[1].parse().unwrap());
            x2.push(f[2].parse().unwrap());
            g1.push(f[3].parse().unwrap());
            g2.push(f[4].parse().unwrap());
        }
        let n = y.len();
        assert_eq!(n, 400);
        let p = 3;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            x[i * p + 1] = x1[i];
            x[i * p + 2] = x2[i];
        }
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();

        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 }, // placeholder likewise
                    slopes: vec![2], // random slope on x2 (col 2) ⇒ slope_extras ⇒ Sparse
                }],
            }),
        };
        let ids = crate::GroupIds {
            primary: g1,
            extra: vec![g2],
        };
        let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        assert!(
            matches!(
                crate::fit::classify_design_pub(&sized, 1),
                crate::fit::Solver::Sparse
            ),
            "slope-carrying extra grouping must route Sparse"
        );

        let opts = crate::FitOptions {
            target_indices: vec![0, 1, 2],
            weights: Some(w.clone()),
            ..crate::FitOptions::default()
        };
        let f = crate::fit_cold(&x, &y, n, p, &model, &ids, &opts);

        assert!(f.converged, "weighted sparse LMM must converge");
        assert!(
            (f.beta[0] - REF_B0).abs() / REF_B0.abs() < 1e-5,
            "β0 {} vs {REF_B0}",
            f.beta[0]
        );
        assert!(
            (f.beta[1] - REF_B1).abs() / REF_B1.abs() < 1e-5,
            "β1 {} vs {REF_B1}",
            f.beta[1]
        );
        assert!(
            (f.beta[2] - REF_B2).abs() / REF_B2.abs() < 1e-5,
            "β2 {} vs {REF_B2}",
            f.beta[2]
        );
        assert!(
            (f.se[0] - REF_SE0).abs() / REF_SE0 < 1e-3,
            "se0 {} vs {REF_SE0}",
            f.se[0]
        );
        assert!(
            (f.se[1] - REF_SE1).abs() / REF_SE1 < 1e-3,
            "se1 {} vs {REF_SE1}",
            f.se[1]
        );
        assert!(
            (f.se[2] - REF_SE2).abs() / REF_SE2 < 1e-3,
            "se2 {} vs {REF_SE2}",
            f.se[2]
        );

        // varcorr[0] = primary g1 (scalar vech), varcorr[1] = extra g2 (2×2
        // vech) — `assemble_varcorr` orders primary-then-extras (fit.rs:528).
        assert_eq!(
            f.varcorr.len(),
            2,
            "two grouping blocks: g1 (scalar) + g2 (2×2)"
        );
        let vc = &f.varcorr[1];
        assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
        let sd_int = vc[0].sqrt();
        let sd_slope = vc[2].sqrt();
        let corr = vc[1] / (sd_int * sd_slope);
        assert!(
            (sd_int - REF_SD_G2_INT).abs() / REF_SD_G2_INT < 1e-3,
            "g2 intercept sd {sd_int} vs {REF_SD_G2_INT}"
        );
        assert!(
            (sd_slope - REF_SD_G2_SLOPE).abs() / REF_SD_G2_SLOPE < 1e-3,
            "g2 slope sd {sd_slope} vs {REF_SD_G2_SLOPE}"
        );
        assert!(
            (corr - REF_CORR_G2).abs() < 0.05,
            "g2 corr {corr} vs {REF_CORR_G2}"
        );

        // g1 is a scalar primary RE: tau2[0] = θ0²·σ̂² is its variance directly.
        let sd_g1 = f.tau2[0].sqrt();
        assert!(
            (sd_g1 - REF_SD_G1).abs() / REF_SD_G1 < 1e-3,
            "g1 sd {sd_g1} vs {REF_SD_G1}"
        );

        // Fit.deviance vs REMLcrit(f) − (n−p)·(1+ln 2π) — same −Σlog wᵢ
        // constant convention pinned by the dense Task 5 golden
        // (`fit_lmm_weighted_matches_lme4`, fit.rs).
        let df = (n - p) as f64;
        let expected = REF_REMLCRIT - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
        assert!(
            (f.deviance - expected).abs() < 1e-3,
            "deviance {} vs lme4-derived {expected}",
            f.deviance
        );
    }

    /// Task 6: constant weights (w ≡ 2) on a sparse-classified design (extra
    /// grouping slope ⇒ `slope_extras` ⇒ `Solver::Sparse`) must reproduce the
    /// unweighted fit's β, SE, and tau2 — same θ̃ = √c·θ argument as the dense
    /// twin `fit_lmm_constant_weights_invariant` (fit.rs), now exercised
    /// through the sparse blocked-Cholesky kernel instead of the dense
    /// suff-stats accumulator.
    #[test]
    fn sparse_lmm_constant_weights_invariant() {
        let n_g1 = 8usize;
        let n_g2 = 6usize;
        let per = 10usize;
        let n = n_g1 * per;
        let mut st = 29u64;
        let mut x = vec![0.0f64; n * 2];
        let mut y = vec![0.0f64; n];
        let mut g1 = vec![0u32; n];
        let mut g2 = vec![0u32; n];
        for i in 0..n {
            g1[i] = (i % n_g1) as u32;
            g2[i] = (i % n_g2) as u32;
            let x1 = super::test_lcg(&mut st);
            x[i * 2] = 1.0;
            x[i * 2 + 1] = x1;
            let re1 = 0.4 * ((g1[i] as f64) - (n_g1 as f64) / 2.0);
            let re2 = 0.3 * ((g2[i] as f64) - (n_g2 as f64) / 2.0);
            y[i] = 0.5 + 0.4 * x1 + re1 + re2 + 0.2 * super::test_lcg(&mut st);
        }
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![1], // slope on x ⇒ slope_extras ⇒ Sparse
                }],
            }),
        };
        let ids = crate::GroupIds {
            primary: g1,
            extra: vec![g2],
        };
        let sized = crate::fit::spec_sized_from_ids_pub(&model, &ids);
        assert!(matches!(
            crate::fit::classify_design_pub(&sized, 1),
            crate::fit::Solver::Sparse
        ));

        let base_opts = crate::FitOptions {
            target_indices: vec![0, 1],
            ..crate::FitOptions::default()
        };
        let unweighted = crate::fit_cold(&x, &y, n, 2, &model, &ids, &base_opts);
        let weighted = crate::fit_cold(
            &x,
            &y,
            n,
            2,
            &model,
            &ids,
            &crate::FitOptions {
                weights: Some(vec![2.0; n]),
                ..base_opts
            },
        );
        assert!(unweighted.converged && weighted.converged);
        for j in 0..2 {
            assert!(
                (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
                "β[{j}] unweighted {} vs w≡2 {}",
                unweighted.beta[j],
                weighted.beta[j]
            );
            assert!(
                (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-6,
                "se[{j}] unweighted {} vs w≡2 {}",
                unweighted.se[j],
                weighted.se[j]
            );
        }
        assert_eq!(unweighted.tau2.len(), weighted.tau2.len());
        for k in 0..unweighted.tau2.len() {
            // BOBYQA rho_end floor, same rationale as the dense twin — 2
            // independently-converged sparse fits, not a shared trajectory
            // (measured ~1.5e-6 relative on this crossed-slope fixture, vs
            // ~2e-8 on the dense scalar-only fixture — the sparse blocked
            // kernel's extra θ-to-Λ indirection costs a bit more precision).
            // A boundary-pinned component (θ collapsed to exactly 0 in both
            // fits) needs an absolute floor: relative error is undefined at 0.
            let denom = unweighted.tau2[k].abs().max(1e-8);
            assert!(
                (unweighted.tau2[k] - weighted.tau2[k]).abs() / denom < 1e-5,
                "tau2[{k}] unweighted {} vs w≡2 {}",
                unweighted.tau2[k],
                weighted.tau2[k]
            );
        }
    }
}
