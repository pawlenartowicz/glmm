//! Sparse-Z LMM solver. A second Gaussian-mixed path that factors
//! `Λ'Z'ZΛ + I` with a two-level blocked Cholesky specialized to this
//! path's structural class — primary(+nested) FAMILIES are block-diagonal
//! (levels never co-occur across families) and ALL crossed-extra columns form
//! one dense tail — lifting the `MAX_*` caps that bound the dense no-Z tail.
//! Mirrors `lmm::reml_deviance` (`lmm.rs:1396`) one level down; validated
//! against it by the both-paths cross-check (`sparse` tests + `glmm/tests.rs`).
//
// `SymbolicCholesky` stays at module level for `logdet_llt`, which the GLMM
// sparse-Schur PIRLS path (`glmm/pirls.rs`) still drives with its own faer
// symbolic factor; the LMM eval loop itself is faer-sparse-free (blocked kernel).
use crate::lmm::LmmGroupings;
use bobyqa::Status;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{
    cholesky_in_place, cholesky_in_place_scratch, LltRegularization,
};
use faer::linalg::cholesky::llt::solve::solve_in_place;
use faer::mat::AsMatMut;
use faer::sparse::linalg::cholesky::SymbolicCholesky;
#[cfg(test)]
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, MatRef, Par, Spec};

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
    let mut ws = SparseLmmWorkspace::new(&g, xm, cluster_ids, extra_ids, y, n, p);

    // θ seed + per-component boxes + solver — topology-only, byte-identical to the
    // NoZ path (the superset property depends on this).
    let (mut solver, mut theta, lower, upper) = crate::lmm::sparse_lmm_seed(&g);
    // Cold start = blind THETA0 per component; a warm start clamps to the truth
    // floor (mirror `fit_lmm` `lmm.rs:1888-1900`).
    match start {
        Some(s) => {
            debug_assert_eq!(s.theta.len(), theta.len());
            for (t, &v) in theta.iter_mut().zip(&s.theta) {
                *t = v.max(crate::lmm::THETA_TRUTH_FLOOR);
            }
        }
        None => {
            for t in theta.iter_mut() {
                *t = crate::lmm::THETA0;
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
    // `lmm.rs:1917-1927`). `Fit` carries no `pinned_components` field (unlike the
    // internal `LmmFit`), so only the θ zeroing — the part that moves the output — is
    // kept; the u32 mask has no home here.
    if ok {
        for &ti in g.diagonal_theta() {
            if theta[ti] <= crate::lmm::PIN_THETA {
                theta[ti] = 0.0;
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
        };
    }

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

    crate::Fit {
        beta,
        se,
        tau2,
        dispersion: 1.0,
        converged: true,
        varcorr,
        stddev_se: vec![],
        aliased: vec![false; p],
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
    /// downdate views it through `MatRef::from_column_major_slice`.
    pub(crate) l21: Vec<f64>,
    /// `U1 = L11⁻¹·B1`, the family rows of `U = L⁻¹Λ'Z'[X y]`
    /// (`k_family × m` col-major `Vec`, columns contiguous for the UᵀU dots).
    pub(crate) u1: Vec<f64>,
    /// Tail scratch (`e×e`): `S22 = A22 − L21·L21ᵀ` assembled lower-tri, then
    /// LLT-factored in place to `L22` (lower; upper stale, never read).
    pub(crate) s22: Mat<f64>,
    /// `U2 = L22⁻¹(B2 − L21·U1)`, the tail rows of `U` (`e × m` col-major
    /// `Vec`; the L22 forward solve runs column-oriented on contiguous slices).
    pub(crate) u2: Vec<f64>,
    /// The augmented Schur factor `L` (dense `m×m` lower Cholesky of
    /// `S = C_xy − UᵀU`), overwritten per eval; `fit_mle_sparse` reads
    /// β̂/σ̂²/SE off it at θ̂ exactly as `fit_lmm` reads `fit.factor`.
    pub(crate) factor: Mat<f64>,
    /// Scratch for the tail's dense `cholesky_in_place` (θ-independent size).
    pub(crate) tail_llt_mem: MemBuffer,
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
    pub(crate) fn new(
        g: &LmmGroupings,
        x: MatRef<f64>,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        y: &[f64],
        n: usize,
        p: usize,
    ) -> Self {
        let m = p + 1;

        // Z'Z (k×k), Z'[X y] (k×m), [X y]'[X y] (m×m) by direct per-row
        // accumulation over Z's ≤ q_p + Σq_e nonzeros per row — O(n·(Σq)²)
        // total, replacing the dense n×k densify + three n×k gemms (~14 M
        // flops on the crossed rungs; the blocked kernel leaves nothing else
        // for a dense Z to feed). NOT the shared `add_rows_multi` accumulator:
        // that path packs per-row level ids into a fixed
        // `[usize; 1 + MAX_EXTRA_GROUPINGS]` stack array (`lmm.rs:542`) and
        // would index out of bounds for over-envelope-by-count designs — the
        // sparse route must stay cap-free. Both Grams are stored full
        // symmetric: `lam_gram_entry` reads Z'Z at arbitrary (ga, gb).
        let mut ztz_dense = Mat::<f64>::zeros(g.k_total, g.k_total);
        let mut ztxy = vec![0.0f64; g.k_total * m];
        let mut cxy = Mat::<f64>::zeros(m, m);
        let mut row: Vec<(usize, f64)> =
            Vec::with_capacity(g.primary_q + g.extra_q.iter().sum::<usize>());
        for i in 0..n {
            row.clear();
            for_each_z_entry(g, x, cluster_ids, extra_ids, i, |col, v| row.push((col, v)));
            for &(ca, va) in &row {
                for &(cb, vb) in &row {
                    ztz_dense[(ca, cb)] += va * vb;
                }
                for j in 0..p {
                    ztxy[ca * m + j] += va * x[(i, j)];
                }
                ztxy[ca * m + p] += va * y[i];
            }
            for a in 0..m {
                let wa = if a < p { x[(i, a)] } else { y[i] };
                for b in 0..m {
                    let wb = if b < p { x[(i, b)] } else { y[i] };
                    cxy[(a, b)] += wa * wb;
                }
            }
        }

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
        let tail_llt_mem = MemBuffer::new(cholesky_in_place_scratch::<f64>(
            e,
            Par::Seq,
            Spec::default(),
        ));

        // Packed raw-Gram streams — every G sub-block the kernel folds, in its
        // exact consumption order (see the field doc; mirrors
        // `sparse_schur_factor` — change together). `ztz_dense` itself is
        // setup-local: after packing, nothing reads the k×k Gram again.
        let cb0 = n_prim + n_prim * np; // first crossed block in lam_blocks
        let pack = |out: &mut Vec<f64>, br: &LamBlock, bc: &LamBlock| {
            for a in 0..br.q {
                let ga = br.start + a * br.stride;
                for b in 0..bc.q {
                    out.push(ztz_dense[(ga, bc.start + b * bc.stride)]);
                }
            }
        };
        // Structural co-occurrence at block granularity: any entry of the raw
        // G block ≠ 0. Sound because the intercept×intercept entry is the
        // shared-row COUNT (> 0 whenever rows are shared); only slope entries
        // can cancel to exact 0.0 (the balanced-zero lesson), never the count.
        let blk_nonzero = |br: &LamBlock, bc: &LamBlock| -> bool {
            for a in 0..br.q {
                let ga = br.start + a * br.stride;
                for b in 0..bc.q {
                    if ztz_dense[(ga, bc.start + b * bc.stride)] != 0.0 {
                        return true;
                    }
                }
            }
            false
        };
        let mut pk_fam = Vec::new();
        let mut pk_a21 = Vec::new();
        let mut a21_blk: Vec<u32> = Vec::new();
        let mut a21_off = Vec::with_capacity(n_prim + 1);
        let mut pk_a21_off = Vec::with_capacity(n_prim + 1);
        for f in 0..n_prim {
            a21_off.push(a21_blk.len());
            pk_a21_off.push(pk_a21.len());
            let pb = &lam_blocks[f];
            pack(&mut pk_fam, pb, pb);
            for c in 0..np {
                let cblk = &lam_blocks[n_prim + f * np + c];
                pack(&mut pk_fam, cblk, pb);
                pack(&mut pk_fam, cblk, cblk);
            }
            // A crossed level co-occurs with a child iff it co-occurs with the
            // parent (child rows ⊆ the primary level's rows), so the primary
            // block decides membership for the whole family.
            for (bi, br) in lam_blocks.iter().enumerate().skip(cb0) {
                if blk_nonzero(br, pb) {
                    a21_blk.push(bi as u32);
                }
            }
            // Merged per-block slabs: block bi's rows carry the pb columns then
            // each child's, so the kernel fills ALL w raw A21 columns in one
            // pass per block (G-row loads amortize across output columns).
            for &bi in &a21_blk[a21_off[f]..] {
                let br = &lam_blocks[bi as usize];
                pack(&mut pk_a21, br, pb);
                for c in 0..np {
                    pack(&mut pk_a21, br, &lam_blocks[n_prim + f * np + c]);
                }
            }
        }
        a21_off.push(a21_blk.len());
        pk_a21_off.push(pk_a21.len());
        let mut pk_a22 = Vec::new();
        let mut a22_pairs: Vec<[u32; 2]> = Vec::new();
        for (bj, bcj) in lam_blocks.iter().enumerate().skip(cb0) {
            for (bi, bri) in lam_blocks.iter().enumerate().skip(bj) {
                if blk_nonzero(bri, bcj) {
                    a22_pairs.push([bi as u32, bj as u32]);
                    pack(&mut pk_a22, bri, bcj);
                }
            }
        }

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
            l21: vec![0.0f64; e * kf],
            u1: vec![0.0f64; kf * m],
            s22: Mat::zeros(e, e),
            u2: vec![0.0f64; e * m],
            factor: Mat::zeros(m, m),
            tail_llt_mem,
            m,
            p,
            n,
        }
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
#[inline]
fn for_each_z_entry(
    g: &LmmGroupings,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    i: usize,
    mut emit: impl FnMut(usize, f64),
) {
    let f = cluster_ids[i] as usize;
    // Slope-major primary layout — mirrors add_rows_multi's scatter (change together).
    emit(f, 1.0);
    for (k, &col) in g.primary_slope_cols.iter().enumerate() {
        emit((k + 1) * g.n_primary + f, x[(i, col)]);
    }
    // Extra groupings: intercept at off, slope c at off+1+c.
    for (e, ids_e) in extra_ids.iter().enumerate() {
        let q_g = g.extra_q[e];
        let off = g.extra_offsets[e] + ids_e[i] as usize * q_g;
        emit(off, 1.0);
        for (c, &col) in g.extra_slope_cols[e].iter().enumerate() {
            emit(off + 1 + c, x[(i, col)]);
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
        for_each_z_entry(g, x, cluster_ids, extra_ids, i, |col, v| {
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
/// linearly; extras pay the same e³ as MixedModels. All buffers are
/// workspace-resident — the eval loop allocates nothing. Numerically this is
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
        factor,
        tail_llt_mem,
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

    if e > 0 {
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
        // U2 tail rows: B2 − L21·U1, forward-solved by L22 column-oriented —
        // per RHS column, subtract L22's contiguous column k scaled by the
        // just-final x[k] (unit stride on both sides). B2 rows fill per
        // crossed block (stride 1), same Λ'-fold as the U1 rows above.
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

    // S = C_xy − UᵀU (lower; m is small — hand loops), Crout in place → the
    // augmented factor L. Upper zeroed once so the recovery's reads see a
    // proper lower-triangular factor.
    for c in 0..m {
        for r in 0..m {
            factor[(r, c)] = 0.0;
        }
    }
    for c in 0..m {
        for r in c..m {
            let mut acc = cxy[(r, c)];
            let (u1r, u1c) = (&u1[r * kf..(r + 1) * kf], &u1[c * kf..(c + 1) * kf]);
            acc -= u1r.iter().zip(u1c).map(|(a, b)| a * b).sum::<f64>();
            let (u2r, u2c) = (&u2[r * e..(r + 1) * e], &u2[c * e..(c + 1) * e]);
            acc -= u2r.iter().zip(u2c).map(|(a, b)| a * b).sum::<f64>();
            factor[(r, c)] = acc;
        }
    }
    for j in 0..m {
        let mut d = factor[(j, j)];
        for k in 0..j {
            let v = factor[(j, k)];
            d -= v * v;
        }
        if !(d.is_finite() && d > 0.0) {
            return None;
        }
        let l = d.sqrt();
        factor[(j, j)] = l;
        for i in (j + 1)..m {
            let mut v = factor[(i, j)];
            for k in 0..j {
                v -= factor[(i, k)] * factor[(j, k)];
            }
            factor[(i, j)] = v / l;
        }
    }
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
    /// Binomial-only for now: `gamma_aic`'s profiled dispersion and the NB
    /// marginal-θ profile assume unit weights, and `fit_warm` rejects those
    /// combinations at the boundary.
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
                let (mui, wi, _) =
                    crate::family::irls_weight_and_resid(family, nb_theta, y[i], e);
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
                for r in 0..p {
                    for c in 0..=r {
                        let mut s = 0.0;
                        for i in 0..n {
                            s += x[(i, r)] * self.w[i] * x[(i, c)];
                        }
                        self.xtwx[(r, c)] = s;
                        self.xtwx[(c, r)] = s;
                    }
                }
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
        crate::family::gamma_aic(y, &ws.prob[..n], dev, n)
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

/// FD-Hessian joint (θ,β) covariance on the sparse path — mirrors
/// `glmm::fd_hessian_cov`'s scheme exactly (single-step central differences, no
/// Richardson extrapolation, step `h_k = FD_STEP_REL·max(1, |γ̂_k|)`,
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
) -> Option<(Mat<f64>, Vec<f64>)> {
    use faer::linalg::solvers::Solve;
    let m = gamma_hat.len();
    let n_theta = ws.g.n_theta();
    let p = ws.p;
    let mut pt = gamma_hat.to_vec();
    let mut eval = |pt: &mut Vec<f64>, coords: &[usize], deltas: &[f64]| -> f64 {
        pt.copy_from_slice(gamma_hat);
        for (&c, &d) in coords.iter().zip(deltas) {
            pt[c] += d;
        }
        sparse_glmm_deviance(family, nb_theta, pt, ws, x, y, n, false)
    };
    let f0 = eval(&mut pt, &[], &[]);
    if !f0.is_finite() {
        return None;
    }
    let steps: Vec<f64> = gamma_hat
        .iter()
        .map(|&g| crate::glmm::FD_STEP_REL * g.abs().max(1.0))
        .collect();
    let mut hess = Mat::<f64>::zeros(m, m);
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
    // entries at THETA0, OFF-DIAGONAL entries at 0 — NOT the all-THETA0 loop the
    // Gaussian `fit_mle_sparse` cold start uses: with a wide vech block (the
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
    {
        let npt1 = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        let config1 = bobyqa::Config {
            rho_begin,
            rho_end: crate::lmm::GLMM_RHO_END,
            npt: npt1,
            ..bobyqa::Config::new(n_theta)
        };
        let mut solver1 = bobyqa::Bobyqa::new(n_theta, config1)
            .expect("BOBYQA config constants are valid by construction");
        let beta0: Vec<f64> = params[n_theta..].to_vec();
        let mut theta1: Vec<f64> = params[..n_theta].to_vec();
        let _ = solver1.minimize(
            |theta| {
                ws.beta[..p].copy_from_slice(&beta0);
                sparse_glmm_deviance(family, nb_theta, theta, &mut ws, xm, y, n, true)
            },
            &mut theta1,
            &lower[..n_theta],
            &upper[..n_theta],
        );
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
    // feeds `converged`.
    let config = bobyqa::Config {
        rho_begin,
        rho_end: crate::lmm::GLMM_RHO_END,
        ..bobyqa::Config::new(n_theta + p)
    };
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
    if ok {
        for &ti in g.diagonal_theta() {
            if params[ti] <= crate::lmm::PIN_THETA {
                params[ti] = 0.0;
            }
        }
    }
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
    let sigma_sq = crate::family::glmm_sigma_sq(family, &y[..n], &ws.prob[..n], &ws.u[..ws.k]);
    let tau2: Vec<f64> = params[..n_theta]
        .iter()
        .map(|&t| t * t * sigma_sq)
        .collect();
    let dispersion = match family {
        crate::Family::Gamma { .. } => match opts.dispersion {
            Some(v) => v,
            None => {
                let mut s = 0.0;
                for (&yi, &mu) in y[..n].iter().zip(ws.prob[..n].iter()) {
                    let r = (yi - mu) / crate::family::variance(family, nb_theta, mu).sqrt();
                    s += r * r;
                }
                s / (n - p) as f64
            }
        },
        _ => 1.0,
    };
    let varcorr = crate::fit::assemble_varcorr(&params[..n_theta], &g, 1.0);

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
            match sparse_fd_hessian_cov(family, nb_theta, &params, &mut ws, xm, y, n) {
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
        },
        final_deviance,
    )
}

/// Sparse-Z negative-binomial GLMM: the over-envelope sibling of
/// `fit::fit_glmm_nb`, same **marginal-θ** profile (`lme4::glmer.nb`) — for
/// each candidate θ the inner `fit_glmm_sparse` re-fits the full GLMM at that
/// fixed θ and its minimized marginal Laplace deviance feeds
/// `logL_marginal(θ) = −½·D(θ) + nb_profile_loglik(y, y, θ)`, maximized over
/// `ln θ` by the shared golden-section bracket. The spec is θ-free (the NB
/// shape is threaded explicitly per candidate); a warm `start` is irrelevant to
/// the global bracket search, exactly as on the dense path. `dispersion = θ̂`.
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
        -0.5 * dev + crate::fit::nb_profile_loglik(y, y, th)
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

        let mut ws =
            super::SparseLmmWorkspace::new(&g, x.as_ref(), &cluster_ids, &extra_ids, &y, n, p);
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
    /// If these disagree, exactly one path is wrong.
    #[test]
    fn sparse_deviance_equals_dense_crossed() {
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
        suff.add_rows_multi(x.as_ref(), &y, &cl, &extra_ids);
        let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);

        let mut ws = super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p);

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
        suff.add_rows_multi(x.as_ref(), &y, &cl, &extra_ids);
        let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);

        let mut ws = super::SparseLmmWorkspace::new(&g, x.as_ref(), &cl, &extra_ids, &y, n, p);

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

    /// OVER-CAP lme4 REML golden: `y ~ 1 + x + (1|g1) + (1|c1)+...+(1|c7)` —
    /// 7 crossed intercept extras exceed `MAX_EXTRA_GROUPINGS=6`, so `fit_cold`
    /// routes to the sparse-Z path. Gated against the frozen lme4 1.1.38 REML
    /// golden (`parity/goldens/sim_wide_crossed_lmm.json`). The oracle is sacred.
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

        let csv = include_str!("../parity/data/sim_wide_crossed.csv");
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
        // 7 extras > MAX_EXTRA_GROUPINGS=6 → over-envelope-by-count ⇒ Sparse.
        assert!(matches!(
            crate::fit::classify_design_pub(&model, 1),
            crate::fit::Solver::Sparse,
        ));
        let ids = crate::GroupIds {
            primary: g1,
            extra: vec![c1, c2, c3, c4, c5, c6, c7],
        };
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

    /// Cross-check: force Sparse on in-envelope designs and diff every
    /// output against NoZ. A mismatch is a bug in exactly one path (NoZ is the
    /// oracle). Spans the five RE-topology axes: scalar-intercept, primary slope
    /// (q_p=2 runtime gate), crossed, nested, slope+crossed.
    #[test]
    fn sparse_vs_noz_cross_check_table() {
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
        suff.add_rows_multi(x.as_ref(), &y, &ids.primary, &ids.extra);
        let mut fit = crate::lmm::LmmFitScratch::with_groupings(p, &g);
        let mut ws =
            super::SparseLmmWorkspace::new(&g, x.as_ref(), &ids.primary, &ids.extra, &y, n, p);
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

        let csv = include_str!("../parity/data/sim_wide_slopes.csv");
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

        let csv = include_str!("../parity/data/sim_sparse_gamma.csv");
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
        // Varcomp: glmm order [gp (primary), ge (extra)]; lme4's VarCorr order is
        // descending level count [ge(40), gp(20)] — map by group NAME. glmm's
        // varcorr[k] is the vech of D̂ = Λ̂Λ̂' on the LINK scale, but lme4's Gamma
        // VarCorr stddevs carry σ̂ (as its tau2 does), so compare σ̂-scaled:
        // σ̂² = tau2[0]/varcorr[0][0] (both are the gp scalar's θ², one σ²-scaled
        // and one not). The q=5 ge diagonal sits at column-major lower-tri
        // offsets 0,5,9,12,14 (the wide-slopes golden convention).
        let sigma_sq = f.tau2[0] / f.varcorr[0][0];
        let gold_of = |name: &str| {
            gold.estimates
                .varcomp
                .iter()
                .find(|b| b.group == name)
                .expect("golden block")
        };
        let gp_sd = f.tau2[0].sqrt();
        let gp_ref = gold_of("gp").stddev[0];
        assert!(
            (gp_sd - gp_ref).abs() / gp_ref.max(1e-6) < 3e-2,
            "gp stddev glmm={gp_sd:.6} lme4={gp_ref:.6}"
        );
        const GE_DIAG: [usize; 5] = [0, 5, 9, 12, 14];
        let ge_ref = gold_of("ge");
        for (t, &off) in GE_DIAG.iter().enumerate() {
            let got = (f.varcorr[1][off] * sigma_sq).sqrt();
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

        let csv = include_str!("../parity/data/sim_sparse_nb.csv");
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

        let csv = include_str!("../parity/data/sim_binomial_slope_crossed.csv");
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
}
