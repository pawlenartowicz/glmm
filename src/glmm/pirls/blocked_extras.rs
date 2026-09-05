//! Structured block+Schur PIRLS solve for the intercept-only crossed/nested-extras regime (`pirls_solve_blocked_extras`, `structured_factor`, `structured_ainv_solve`).

use super::*;

/// The three crossed-tail Schur kernels the structured solve needs from its
/// scalar, split off `Scalar` so the kernel-wide trait in `src/scalar.rs`
/// stays free of any GLMM-workspace type: `StructuredSchur` (the cached sparse
/// LLT) is `f64`-only, and only this module's `structured_factor` /
/// `structured_ainv_solve` route through these. The default bodies are the
/// dense generic arms every non-`f64` scalar takes; `f64` overrides each with
/// the production route.
pub(crate) trait TailKernel: Scalar {
    /// One cluster's Schur downdate `S[cols][cols] −= C_f' A_f⁻¹ C_f`, lower
    /// triangle. `core_l` is cluster `f`'s `qc×qc` Crout factor, `coup` its
    /// `qc×e` row-major coupling block, `cols` its ascending nonzero crossed
    /// columns, `schur` the dense row-major `e×e` accumulator (lower at
    /// `[a·e + b]`, `a ≥ b`).
    ///
    /// The default body is the rank-1 scalar column walk. `f64` overrides it
    /// with the panel route (`glmm_block_solve_panel` + one triangular faer
    /// matmul) at `qc > 1`, which is the production route there and moves the
    /// dot's internal association; the measurement behind that split lives on
    /// this method's `f64` override (moved there with the code).
    #[allow(clippy::too_many_arguments)]
    fn tail_downdate(
        core_l: &[Self],
        qc: usize,
        coup: &[Self],
        e: usize,
        cols: &[u32],
        ss: Option<&mut StructuredSchur>,
        schur: &mut [Self],
    ) {
        let _ = ss;
        tail_downdate_generic(core_l, qc, coup, e, cols, schur)
    }

    /// Factor the crossed-tail Schur complement in `schur` (dense row-major
    /// `e×e`, lower triangle) and return `½·log|S|` — the half-log convention
    /// `structured_factor` accumulates and `deviance.rs` doubles. `None` on a
    /// non-PD tail.
    ///
    /// The default body is the crate-own dense Crout factor, in place, leaving
    /// `schur` holding `L` for `tail_solve`. `f64` overrides it with the cached
    /// sparse LLT when `ss` is `Some`, which is what the production path runs
    /// and what the bit-identity dump pins; `ss = None` selects the dense arm on
    /// both.
    fn tail_factor(schur: &mut [Self], e: usize, ss: Option<&mut StructuredSchur>) -> Option<Self> {
        let _ = ss;
        tail_factor_generic(schur, e)
    }

    /// Solve `S x = rhs` in place for the single `e`-long tail column, using
    /// whichever factor `tail_factor` produced. Default: forward/back
    /// substitution on `schur`'s dense `L`. `f64` override: the cached sparse
    /// back-solve when `ss` is `Some`, the dense substitution otherwise.
    fn tail_solve(schur: &[Self], e: usize, ss: Option<&mut StructuredSchur>, rhs: &mut [Self]) {
        let _ = ss;
        tail_solve_generic(schur, e, rhs)
    }
}

impl<const N: usize> TailKernel for crate::dual::Dual<N> {}
impl<const N: usize, const H: usize> TailKernel for crate::dual::HyperDual<N, H> {}

impl TailKernel for f64 {
    fn tail_downdate(
        core_l: &[f64],
        qc: usize,
        coup: &[f64],
        e: usize,
        cols: &[u32],
        ss: Option<&mut StructuredSchur>,
        schur: &mut [f64],
    ) {
        use crate::glmm::workspace::glmm_block_solve_panel;
        let e_f = cols.len();
        if cols.is_empty() {
            return;
        }
        // A single call (one cluster per invocation, unlike `structured_factor`'s
        // per-cluster loop) never reuses `ss` afterward, so this consumes it
        // directly rather than reborrowing via `as_deref_mut`.
        let Some(StructuredSchur {
            c_panel,
            y_panel,
            dd_temp,
            ..
        }) = ss.filter(|_| qc != 1)
        else {
            // Production route for qc == 1 (and the ss = None oracle). At
            // qc == 1 the downdate is rank-1: the panel path stages the
            // identical FLOPs (gather → dd_temp → second scatter pass) at
            // ~double the memory traffic, a measured +4–7% per-eval loss on
            // the qc=1 cross6 GLMM cells (binb 1.758→1.870 s; 2026-07-14
            // drift investigation). This scalar walk accumulates each rank-1
            // dot straight into `schur` — already minimal. `qc == 1` is the
            // only measured boundary; no qc>1 GLMM structured cell exists in
            // the grid. Sizing of the panel buffers mirrors this condition in
            // `StructuredSchur::new` — change together.
            let mut ycol = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
            for &b in cols {
                let b = b as usize;
                for local in 0..qc {
                    ycol[local] = coup[local * e + b];
                }
                glmm_block_solve(&core_l[..qc * qc], qc, &mut ycol[..qc]);
                for &a in cols {
                    let a = a as usize;
                    if a < b {
                        continue;
                    }
                    let mut acc = 0.0;
                    for local in 0..qc {
                        acc += coup[local * e + a] * ycol[local];
                    }
                    schur[a * e + b] -= acc;
                }
            }
            return;
        };
        let cpan = &mut c_panel[..qc * e_f];
        for local in 0..qc {
            let crow = &coup[local * e..];
            for (dst, &b) in cpan[local * e_f..(local + 1) * e_f].iter_mut().zip(cols) {
                *dst = crow[b as usize];
            }
        }
        let ypan = &mut y_panel[..qc * e_f];
        ypan.copy_from_slice(cpan);
        glmm_block_solve_panel(&core_l[..qc * qc], qc, ypan, e_f);
        // dd = C_f'·Y (e_f×e_f lower, col-major). A row-major qc×e_f buffer
        // viewed col-major e_f×qc IS its transpose — no copy for either side.
        let dd = &mut dd_temp[..e_f * e_f];
        let ct = faer::MatRef::from_column_major_slice(cpan, e_f, qc);
        let yv = faer::MatRef::from_column_major_slice(ypan, e_f, qc).transpose();
        faer::linalg::matmul::triangular::matmul(
            faer::MatMut::from_column_major_slice_mut(dd, e_f, e_f),
            faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
            faer::Accum::Replace,
            ct,
            faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            yv,
            faer::linalg::matmul::triangular::BlockStructure::Rectangular,
            1.0,
            faer::Par::Seq,
        );
        for (bj, &b) in cols.iter().enumerate() {
            let b = b as usize;
            for (&a, &d) in cols[bj..].iter().zip(&dd[bj * e_f + bj..(bj + 1) * e_f]) {
                schur[a as usize * e + b] -= d;
            }
        }
    }

    fn tail_factor(schur: &mut [f64], e: usize, ss: Option<&mut StructuredSchur>) -> Option<f64> {
        use crate::sparse::logdet_llt;
        match ss {
            Some(ss) => {
                // Gather the dense schur lower triangle into S's fixed CSC
                // pattern. schur is row-major e×e (lower tri at [a·e+b], a ≥ b);
                // axx is CSC, so column b's stored rows a are exactly the a ≥ b we
                // read.
                {
                    let (sym, vals) = ss.axx.parts_mut();
                    let col_ptr = sym.col_ptr();
                    let row_idx = sym.row_idx();
                    for b in 0..e {
                        for slot in col_ptr[b]..col_ptr[b + 1] {
                            let a = row_idx[slot];
                            vals[slot] = schur[a * e + b];
                        }
                    }
                }
                // Numeric sparse LLT into the cached buffer; non-PD ⇒ None (⇒ NaN dev).
                let llt = ss
                    .symbolic
                    .factorize_numeric_llt(
                        &mut ss.l_values,
                        ss.axx.as_ref(),
                        faer::Side::Lower,
                        faer::linalg::cholesky::llt::factor::LltRegularization::default(),
                        faer::Par::Seq,
                        faer::dyn_stack::MemStack::new(&mut ss.fac_mem),
                        faer::Spec::default(),
                    )
                    .ok()?;
                let _ = llt; // ends the &'out borrow on l_values (LltRef is Copy; NLL)
                             // logdet_llt returns 2·Σ ln L_ii = log|S|; this method's return is
                             // the ½·log convention (deviance.rs multiplies by 2). So return HALF.
                             // Non-finite (non-PD diagonal) ⇒ None, matching the dense non-PD
                             // sentinel.
                let log_s = logdet_llt(&ss.symbolic, &ss.l_values);
                if !log_s.is_finite() {
                    return None;
                }
                Some(0.5 * log_s)
            }
            None => {
                // Dense fallback: the old Crout factor (test cross-check / defensive).
                if !glmm_block_chol(&mut schur[..e * e], e) {
                    return None;
                }
                let mut logdet = 0.0;
                for b in 0..e {
                    logdet += schur[b * e + b].ln();
                }
                Some(logdet)
            }
        }
    }

    fn tail_solve(schur: &[f64], e: usize, ss: Option<&mut StructuredSchur>, rhs: &mut [f64]) {
        match ss {
            Some(ss) => {
                // Reconstruct the factor from the cached symbolic + values (no re-factor;
                // faer 0.24.4 LltRef::new — sparse/linalg/cholesky.rs:4443-4449, verified
                // against the vendored source: a Copy wrapper over two refs, NOT a
                // re-factorization — and back-solve the single e-col.
                let llt = faer::sparse::linalg::cholesky::LltRef::new(&ss.symbolic, &ss.l_values);
                let rhs = faer::MatMut::from_column_major_slice_mut(&mut rhs[..e], e, 1);
                llt.solve_in_place_with_conj(
                    faer::Conj::No,
                    rhs,
                    faer::Par::Seq,
                    faer::dyn_stack::MemStack::new(&mut ss.solve_mem),
                );
            }
            None => {
                // Dense fallback (test cross-check / e>0 with no cached factor): schur
                // holds the dense L that the dense-factor branch produced.
                glmm_block_solve(&schur[..e * e], e, &mut rhs[..e]);
            }
        }
    }
}

/// The default body of [`TailKernel::tail_downdate`]: the rank-1 scalar column
/// walk, run regardless of `qc` — a non-`f64` scalar never reaches the
/// panel-vs-scalar split (`StructuredSchur` is `f64`-only), so this is the
/// only arm any non-`f64` `T` ever takes. Bit pattern matches `f64`'s own
/// `qc == 1` arm exactly; the two diverge only past `qc > 1`, where `f64`
/// switches to the panel route and this stays on the scalar walk.
pub(crate) fn tail_downdate_generic<T: Scalar>(
    core_l: &[T],
    qc: usize,
    coup: &[T],
    e: usize,
    cols: &[u32],
    schur: &mut [T],
) {
    let mut ycol = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    for &b in cols {
        let b = b as usize;
        for local in 0..qc {
            ycol[local] = coup[local * e + b];
        }
        glmm_block_solve(&core_l[..qc * qc], qc, &mut ycol[..qc]);
        for &a in cols {
            let a = a as usize;
            if a < b {
                continue;
            }
            let mut acc = T::ZERO;
            for local in 0..qc {
                acc += coup[local * e + a] * ycol[local];
            }
            schur[a * e + b] -= acc;
        }
    }
}

/// The default body of [`TailKernel::tail_factor`]: the crate-own dense Crout
/// factor (`glmm_block_chol`) plus the Σ ln L_bb half-log-determinant fold —
/// the same arm `f64`'s override runs when `ss` is `None`.
pub(crate) fn tail_factor_generic<T: Scalar>(schur: &mut [T], e: usize) -> Option<T> {
    if !glmm_block_chol(&mut schur[..e * e], e) {
        return None;
    }
    let mut logdet = T::ZERO;
    for b in 0..e {
        logdet += schur[b * e + b].ln();
    }
    Some(logdet)
}

/// The default body of [`TailKernel::tail_solve`]: forward/back substitution
/// through `glmm_block_solve` on the dense `L` that `tail_factor` left in
/// `schur` — the same arm `f64`'s override runs when `ss` is `None`.
pub(crate) fn tail_solve_generic<T: Scalar>(schur: &[T], e: usize, rhs: &mut [T]) {
    glmm_block_solve(&schur[..e * e], e, &mut rhs[..e]);
}

/// Factor the structured `A = [[D, C], [C', E]]` in place: Crout-factor each
/// core block `core_blocks[f]` (holding `D_f + I` on entry, its lower `L` on
/// return) and build + factor the Schur complement `schur_blk` (holding `E + I`
/// on entry, `S = (E+I) − Σ_f C_f' A_f⁻¹ C_f` then its `L` on return). Returns
/// `Some(log|A|) = Σ_f log|A_f| + log|S|` (Schur determinant identity), or `None`
/// on a non-PD core block / Schur. The Schur factor itself routes through
/// `TailKernel::tail_factor`: the cached sparse LLT when `ss` is `Some` at `T = f64`
/// (the production route), the crate-own dense Crout — `tail_factor`'s default
/// body — at any non-`f64` `T` or when `ss` is `None`. `coupling[f]` holds `C_f`
/// (q_core×e row-major), unchanged. Shared by `pirls_solve_blocked_extras` (per iteration) and
/// `structured_schur_fill` (reusing the converged factors). `q_core ≤ MAX_PRIMARY_Q`.
/// `coup_cols`/`coup_ptr` is the per-cluster CSR of C_f's nonzero crossed columns
/// (built once per solve by `pirls_solve_blocked_extras`): the Schur build walks
/// only those columns instead of all `e` — every skipped column of `C_f` is
/// exactly 0.0 (no row of cluster `f` touches that crossed level), so skipping it
/// drops only exact-zero contributions. A θ-pinned crossed grouping that a dual
/// `T` retains (`build_packed_m`'s pin skip is `f64`-only) is exactly-zero for
/// the other reason — its θ is 0 — so it costs a column here and moves no value. On an observation-level primary (s ≈ n,
/// grouseticks) this collapses the build from `s·e²/2` to the ~G² true nonzeros
/// per cluster. The downdate itself runs panel-wise per cluster (one batched
/// `A_f⁻¹` solve + one triangular `C_f'·Y` matmul through `ss`'s scratch — the
/// LMM sparse-tail kernels A–D port): each `S[a][b]` still receives exactly one
/// subtraction per touching cluster in the same `f` order; only the dot's
/// internal association moved into the matmul — a result-moving reassociation of
/// the sanctioned class (see `SparseTail`'s doc). The panel path serves
/// `qc > 1` (vector primary and/or nested REs). At `qc == 1` the downdate is
/// rank-1 and the scalar column-at-a-time walk is the **production route**
/// (routed via the `qc != 1` filter inside `TailKernel::tail_downdate`, whose `f64`
/// override documents the panel-vs-scalar measurement behind the split); the
/// same walk is also the `ss = None` arm, which is the panel path's equality
/// oracle at `qc > 1`.
#[allow(clippy::too_many_arguments)]
fn structured_factor<T: TailKernel>(
    g: &crate::lmm::LmmGroupings,
    core_blocks: &mut [T],
    coupling: &[T],
    schur_blk: &mut [T],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    mut ss: Option<&mut StructuredSchur>,
) -> Option<T> {
    let qc = g.primary_q + g.nested_per_parent;
    let s = g.n_primary;
    let e = g.k_crossed();
    let mut logdet = T::ZERO;
    for f in 0..s {
        let cb = f * qc * qc;
        if !glmm_block_chol(&mut core_blocks[cb..cb + qc * qc], qc) {
            return None;
        }
        for r in 0..qc {
            logdet += core_blocks[cb + r * qc + r].ln();
        }
        // S −= C_f' A_f⁻¹ C_f (lower triangle) over cluster f's NONZERO crossed
        // columns, via TailKernel::tail_downdate (panel-vs-scalar routing documented
        // there and on structured_factor's own doc comment above).
        let coup = f * qc * e;
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        let e_f = cols.len();
        if e_f == 0 {
            continue;
        }
        T::tail_downdate(
            &core_blocks[cb..cb + qc * qc],
            qc,
            &coupling[coup..],
            e,
            cols,
            ss.as_deref_mut(),
            schur_blk,
        );
    }
    // Schur-determinant identity: log|A| = Σ_f log|A_f| + log|S|. S is factored
    // by TailKernel::tail_factor: the cached sparse LLT when `ss` is `Some` at
    // `T = f64` (production), the crate-own dense Crout — `tail_factor`'s
    // default body — otherwise. No `if e > 0` guard needed: `StructuredSchur::new`
    // returns `None` at `e == 0`, so the sparse arm never sees an empty tail, and
    // the dense arm's `glmm_block_chol` at dimension 0 (`linalg::block_chol`'s
    // `for j in 0..0` loop) over the `.max(1)`-sized `schur_blk` is an
    // indexing-free no-op returning `true` — a NaN-free, logdet-preserving skip.
    logdet += T::tail_factor(schur_blk, e, ss)?;
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
/// (same argument as `structured_factor`'s doc comment). That argument runs the
/// other way too, and is why `build_packed_m` can keep a θ-pinned crossed column
/// at a dual `T`: the retained column's value is exactly 0.0, so carrying it
/// changes no number here and only gives the seeded lane something to
/// differentiate. The tail solve itself
/// routes through `TailKernel::tail_solve`: the cached sparse back-solve when `ss`
/// is `Some` at `T = f64` (production), the dense substitution — `tail_solve`'s
/// default body — otherwise. Shared by the structured PIRLS solve and the
/// inference Schur fill.
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_ainv_solve<T: TailKernel>(
    g: &crate::lmm::LmmGroupings,
    core_blocks: &[T],
    coupling: &[T],
    schur_blk: &[T],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    ss: Option<&mut StructuredSchur>,
    a_rhs: &mut [T],
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
            let mut acc = T::ZERO;
            for local in 0..qc {
                acc += coupling[coup + local * e + b] * a_rhs[gcb + local];
            }
            a_rhs[k_family + b] -= acc;
        }
    }
    if e == 0 {
        return;
    }
    T::tail_solve(schur_blk, e, ss, &mut a_rhs[k_family..k_family + e]);
    // u_{core,f} = t_f − A_f⁻¹(C_f u_e).
    for f in 0..s {
        let cb = f * qc * qc;
        let gcb = f * qc;
        let coup = f * qc * e;
        let mut v = [T::ZERO; MAX_PRIMARY_Q];
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        #[allow(clippy::needless_range_loop)]
        for local in 0..qc {
            let mut acc = T::ZERO;
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
/// Step-halving differs by `beta_step`. `Fixed` and `Profile { exact: None }`
/// (the PQL border) share `pirls_solve`'s retrospective `dev + pen_u` band,
/// halving `u` back toward `u_prev`. `Profile { exact: Some(_) }` (P1) instead
/// halves on the exact Laplace merit (`dev + pen_u + 2·logdet` plus its
/// mode-consistency correction — see the assembly below) and, at the top of
/// the loop, on `infeasible` alone; it also treats a non-PD structured factor
/// on the trial `u` as a rejected step rather than a hard failure, backtracking
/// toward `(u_acc, beta_prev)` the same way — mirrors `pirls_solve_blocked`,
/// change together. Every mode's halved retry re-enters the top and re-runs
/// the full structured factor + Schur (`structured_factor` /
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
pub(crate) fn pirls_solve_blocked_extras<T: TailKernel>(
    family: Family,
    nb_theta: f64,
    g: &crate::lmm::LmmGroupings,
    cluster_ids: &[u32],
    m_core_buf: &[T],
    cross_val: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    beta: &mut [T],
    mut beta_step: BetaStep,
    eta: &mut [T],
    prob: &mut [T],
    w: &mut [T],
    u: &mut [T],
    u_prev: &mut [T],
    eta_fixed: &mut [T],
    mu: &mut [T],
    core_blocks: &mut [T],
    coupling: &mut [T],
    schur_blk: &mut [T],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    mut structured_schur: Option<&mut StructuredSchur>,
    force_dense: bool,
    a_rhs: &mut [T],
    // Dual derivative kernels' per-solve controls (see `DualStep`); `None` on
    // every f64 fit-path call.
    dual: Option<&mut DualStep<T>>,
    // n × p = W∘X GEMM scratch for the Profile β-Schur border's C = X'WX
    // (mirrors `pirls_solve`'s `wx`).
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
    // instantiation must use BetaStep::Fixed — the derivative entry points do.
    // Loud, not silent: dropping derivatives here would be a wrong gradient.
    assert!(
        T::IS_F64 || matches!(beta_step, BetaStep::Fixed),
        "BetaStep::Profile is f64-only on the extras path"
    );
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
        let mut ef = T::ZERO;
        for j in 0..p {
            ef += T::from_f64(x[(i, j)]) * beta[j];
        }
        eta_fixed[i] = ef;
    }
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += T::from_f64(o[i]);
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
    // `ex.u_acc` is workspace-persistent (it survives across `laplace_deviance`
    // calls at different θ) and is only ever written by an ACCEPTED post-sweep
    // iterate — reseed it here so a pre-first-accept halving in exact mode
    // targets this solve's own `u_prev` seed, not a stale value left by a
    // previous call. Mirrors `pirls_solve_blocked` — change together.
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
    // `exact` mirrors `blocked.rs`'s contract — change together. It is
    // `is_canonical` here where `blocked.rs` writes an unconditional `true`: that
    // caller pairs `exact = true` with `observed = !canonical`, so the step it
    // takes IS the Hessian step. This path always takes the Fisher step — an
    // observed step would need observed twins of `core_blocks`, `coupling` and
    // `schur_blk`, all built from W — so on a non-canonical link one kernel call
    // is not exact, and claiming otherwise would stop the caller's refinement
    // loop on lanes that have not settled. `d.observed` and `d.obs_blocks` are
    // read nowhere on this path.
    let min_iters = match dual {
        Some(d) => {
            d.exact = crate::family::is_canonical(family);
            d.min_iters
        }
        None => 0,
    };
    let (mut dev, mut pen, mut logdet) = (T::from_f64(f64::NAN), T::from_f64(f64::NAN), T::ZERO);
    let tol = pirls_tol_override.unwrap_or_else(|| super::super::pirls_tol(family));
    // Exact-Laplace β-profile: the accept/halve decision moves from the
    // penalized-deviance band (below) to the FULL Laplace merit `dev + pen_u +
    // 2·log|A| + g_u'·δu₀` after the structured factor, since log|A| depends on
    // this iteration's A and is not known until the factor is formed. `l_acc` is
    // that merit at the last ACCEPTED (u, β) — a cold start accepts
    // unconditionally. Mirrors `pirls_solve_blocked` — change together.
    let exact = matches!(beta_step, BetaStep::Profile { exact: Some(_), .. });
    let mut l_acc = f64::INFINITY;
    // `|g_u'·δu₀|` at the point `l_acc` was measured at — the accepted endpoint's
    // half of the accept band. Mirrors `pirls_solve_blocked`.
    let mut l_acc_slack = 0.0_f64;
    for it in 0..PIRLS_MAX_ITERS {
        counters.set_pirls_iters(it + 1);
        // --- trial evaluation at the CURRENT u (pass 1 + pass 2). On a fresh accept
        // this is the newly-stepped u; after a halving `continue` it is the
        // backtracked u. Either way the recompute IS the trial evaluation. ---
        // --- pass 1: η-pass — ηᵢ = η_fixed,ᵢ + (Mu)ᵢ over the row's nonzeros ---
        // Reads the packed M nonzeros (contiguous q_core core slice + n_cross[i]
        // crossed entries) `build_packed_m` filled — no faer indexing, crossed term
        // O(G) not O(e).
        let mut yeta = T::ZERO;
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let m_core = &m_core_buf[i * qc..i * qc + qc];
            let mut mui = T::ZERO;
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
            yeta += T::from_f64(y[i]) * eta[i];
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
        // Retrospective step-halving (lme4 `pwrssUpdate`, mirrors `pirls_solve` /
        // `pirls_solve_blocked`): convergence band checked BEFORE the overshoot test
        // (near the optimum Fisher scoring is not strictly monotone — a step can
        // land ε above `pen_accepted` yet inside the tol band, and that must
        // converge, not burn all 10 halvings against FP noise). ‖u‖² is at the
        // CURRENT trial u; `u[..k]` spans the core (RE-column order) + crossed tail.
        let mut pen_u = T::ZERO;
        #[allow(clippy::needless_range_loop)]
        for c in 0..k {
            pen_u += u[c] * u[c];
        }
        let penalized = dev + pen_u;
        // BAND-TOLERANT overshoot test, mirrors `pirls_solve` (see its comments for
        // why a within-band rise is accepted rather than converged-on or halved):
        // only a rise EXCEEDING the tol band backtracks. A domain-infeasible trial
        // halves regardless of the band (see `pirls_solve`'s comment).
        // Exact mode: the retrospective band above is on `dev + pen_u` alone,
        // which is not the profiled objective (missing 2·log|A|) — only a
        // domain-infeasible trial halves here; an overshoot on `dev + pen_u` is
        // judged later, after log|A| is known, against the FULL merit.
        if infeasible
            || (!exact && penalized.value() - pen_accepted > tol * (1.0 + penalized.value().abs()))
        {
            if halvings < PIRLS_MAX_HALVINGS {
                // Last full step overshot: halve δu = u − u_prev and re-enter the top
                // (the recompute above is the trial evaluation of the halved step; a
                // halved retry re-runs structured_factor/ainv_solve, by design).
                // Exact mode halves toward the last ACCEPTED u (`u_acc`, set by the
                // post-sweep merit test below) since `u_prev` there holds the
                // just-rejected trial, not an accepted point.
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
                // `pirls_solve_blocked`'s Profile backtrack.
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
        // Accept this iterate, snapshot it for the next backtrack, and take a fresh
        // full Fisher step from it (cold start: pen_accepted = ∞ ⇒ always accepts).
        // `u_prev` is unconditional — the β-Schur border needs δu₀ = u_new − u_prev
        // regardless of mode. Exact mode defers `halvings`/`pen_accepted`/`beta_prev`
        // bookkeeping to the post-sweep merit test below, since this trial has not
        // been judged against the full Laplace objective yet.
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
        // IRLS effective residual rᵢ = wᵢ·(Mu)ᵢ + W·working_resid, so the
        // scattered RHS is M'(W·Mu + W·r) = (A−I)u + M'(W·r). Logit's W·r=(y−p);
        // the general branch's is (dμ/dη)·(y−μ)/V.
        match family {
            Family::Binomial {
                link: BinomialLink::Logit,
            } if !weighted => {
                for i in 0..n {
                    mu[i] = w[i] * mu[i] + (T::from_f64(y[i]) - prob[i]);
                }
            }
            other => {
                for i in 0..n {
                    let dmu = crate::family::mu_eta(other, eta[i]);
                    let v = crate::family::variance(other, nb_theta, prob[i]);
                    mu[i] = w[i] * mu[i]
                        + T::from_f64(prior_w[i]) * dmu * (T::from_f64(y[i]) - prob[i]) / v;
                }
            }
        }
        // --- pass 3: scatter — wᵢmᵢmᵢ' into D_f/C_f/E (lower tri), rᵢmᵢ into g ---
        for v in core_blocks[..s * qc * qc].iter_mut() {
            *v = T::ZERO;
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
                    coupling[coup + r * e + b] = T::ZERO;
                }
            }
        }
        // schur_blk stays a dense clear (unlike coupling above): the dense-fallback
        // branch of structured_factor (TailKernel::tail_factor's `ss = None` arm, which
        // is what `force_dense` now routes to) Cholesky-factors it IN PLACE,
        // producing fill-in outside the coup_cols×coup_cols pattern — a sparse zero
        // would leave stale L-factor residue from the prior iteration.
        for v in schur_blk[..e * e].iter_mut() {
            *v = T::ZERO;
        }
        for v in a_rhs[..k].iter_mut() {
            *v = T::ZERO;
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
        // --- +I ridge (per RE column) on the core diagonals and the E diagonal ---
        for f in 0..s {
            let cb = f * qc * qc;
            for r in 0..qc {
                core_blocks[cb + r * qc + r] += T::ONE;
            }
        }
        for b in 0..e {
            schur_blk[b * e + b] += T::ONE;
        }
        // --- factor (core blocks + Schur), then apply A⁻¹ to the scattered RHS ---
        logdet = match structured_factor(
            g,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            // force_dense folds into ss: this is the arm today's
            // `Some(ss) if !force_dense` match took, so it is bit-identical.
            if force_dense {
                None
            } else {
                structured_schur.as_deref_mut()
            },
        ) {
            Some(ld) => ld,
            None => {
                // Mirrors the block-Cholesky failure arm in
                // `pirls_solve_blocked` (`blocked.rs`, the no-extras twin) —
                // change together. Exact mode has no retrospective `dev + pen_u`
                // band (the merit test below owns the accept decision), so
                // nothing stands between a wildly overshooting joint (u, β) step
                // and this factor. On
                // grouseticks the cold-start Newton step lands at dev ≈ 3e26,
                // where the crossed-tail Schur loses positive-definiteness to
                // cancellation and the solve would end here — while the same step
                // under the PQL band is simply halved away. An unevaluable trial
                // is a REJECTED trial, not a failed solve: backtrack toward the
                // last accepted (u, β) exactly as the merit rejection below does,
                // and surface the failure only once the halvings are spent.
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
                        continue;
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
        structured_ainv_solve(
            g,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            // same force_dense → ss fold as the structured_factor call above.
            if force_dense {
                None
            } else {
                structured_schur.as_deref_mut()
            },
            a_rhs,
        );
        // a_rhs[gcb..] now holds u_{core,f}; a_rhs[k_family+b] holds u_e. Scatter to
        // u (RE-column order) and accumulate ‖u‖².
        pen = T::ZERO;
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
        // Exact mode: assemble c_β off THIS iteration's structured factors, then
        // judge the trial against the full Laplace merit. Both live here rather
        // than in the β-Schur border below: the assembly reads only the factors and
        // the live W (nothing the border produces), and the merit needs the `g_u`
        // its first pass forms. The border still runs on the accepted-or-halved
        // u/β, so this must land before it. A rise past the tol band re-halves
        // toward the last accepted (u, β) exactly like the retrospective test
        // above, just on the merit that actually matters; the first trial always
        // accepts (`l_acc = ∞`). Mirrors `pirls_solve_blocked`'s exact block —
        // change together.
        if let BetaStep::Profile {
            exact: Some(ex),
            beta_prev,
            ..
        } = &mut beta_step
        {
            // Canonical links only, and `exact_profile_shape` is what keeps a
            // non-canonical structured shape off this route: the û path would need
            // Ã = A_obs, i.e. an observed twin of the WHOLE structured factor (core
            // blocks, coupling AND Schur), where the blocked path needs only a
            // second set of per-cluster blocks. On a canonical link the Fisher A
            // already IS the exact ∂²/∂u², so the factor left by
            // `structured_factor` is the one every pass below wants.
            debug_assert!(crate::family::is_canonical(family));
            let gu_dot_du = {
                let ExactProfileBufs {
                    logdet_u,
                    logdet_beta,
                    tail_inv,
                    tail_r,
                    fac_f64,
                    ..
                } = &mut **ex;
                let logdet_u = &mut logdet_u[..k];
                let logdet_beta = &mut logdet_beta[..p];
                logdet_u.fill(0.0);
                logdet_beta.fill(0.0);
                // One f64 mirror of the s per-cluster factors for pass A below
                // (see `ExactProfileBufs::fac_f64`). Mirrors
                // `pirls_solve_blocked`'s own fill — change together. Built from
                // THIS iterate's factors: `structured_factor` above leaves
                // `core_blocks` factored, and nothing between here and pass A
                // writes it.
                let fac_f64 = &mut fac_f64[..qc * qc * s];
                for (o, v) in fac_f64.iter_mut().zip(core_blocks[..qc * qc * s].iter()) {
                    *o = v.value();
                }
                // S⁻¹ column by column, through the same tail solve the u-step
                // uses, so it inherits whichever factor `structured_factor` left
                // (cached sparse LLT, or the dense L in `schur_blk`). `a_rhs`'s
                // crossed tail is the T-typed staging slot — free now, its u-solve
                // was scattered to `u` just above. Column-major:
                // `tail_inv[b·e + a] = (S⁻¹)_{a,b}`.
                for b in 0..e {
                    for slot in a_rhs[k_family..k_family + e].iter_mut() {
                        *slot = T::ZERO;
                    }
                    a_rhs[k_family + b] = T::ONE;
                    T::tail_solve(
                        schur_blk,
                        e,
                        if force_dense {
                            None
                        } else {
                            structured_schur.as_deref_mut()
                        },
                        &mut a_rhs[k_family..k_family + e],
                    );
                    for a in 0..e {
                        tail_inv[b * e + a] = a_rhs[k_family + a].value();
                    }
                }
                // pass A: hᵢ, w'ᵢ → direct part of c_β, and gᵤ = ∂log|A|/∂u.
                // c_β = d log|A|/dβ enters the same Newton RHS with the same ½ as
                // the objective's `2·logdet` term. Direct part: Σᵢ w'ᵢ·xᵢⱼ·hᵢ with
                // hᵢ = mᵢ'A⁻¹mᵢ the RE leverage. û path: β also moves ũ(β), so
                // log|A|(β) picks up gᵤ'·dũ/dβ with dũ/dβ = −A⁻¹M'WX; folded as ONE
                // adjoint solve v = A⁻¹gᵤ (pass B) rather than materializing the
                // k×p `dũ/dβ`.
                let mut mc = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
                let mut yc = [0.0_f64; crate::lmm::MAX_PRIMARY_Q];
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    let cb = f * qc * qc;
                    let cbase = i * g_cap;
                    let ncz = n_cross[i] as usize;
                    for local in 0..qc {
                        mc[local] = m_core_buf[i * qc + local].value();
                    }
                    let fac = &fac_f64[cb..cb + qc * qc];
                    // Block inverse of A = [[D, C], [C', E]] applied to one row:
                    // mᵢ'A⁻¹mᵢ = ‖L_f⁻¹m_c‖² + rᵢ'S⁻¹rᵢ with rᵢ = C_f'(A_f⁻¹m_c) − m_x.
                    // `rᵢ` is supported on CLUSTER f's coupling columns — a strict
                    // superset of row i's own crossed columns as soon as f has rows
                    // at more than one crossed level — so the walk is over
                    // `coup_cols[f]`; restricting it to the row's columns would drop
                    // real off-column terms of the quadratic form.
                    let mut h = block_leverage(fac, qc, &mc[..qc]);
                    let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
                    if !cols.is_empty() {
                        let coup = f * qc * e;
                        yc[..qc].copy_from_slice(&mc[..qc]);
                        glmm_block_solve(fac, qc, &mut yc[..qc]);
                        // `tail_r` needs no clearing: every column of `cols` is
                        // written here before it is read below, and no other column
                        // is touched.
                        for &b in cols {
                            let b = b as usize;
                            let mut acc = 0.0;
                            for local in 0..qc {
                                acc += coupling[coup + local * e + b].value() * yc[local];
                            }
                            tail_r[b] = acc;
                        }
                        for z in 0..ncz {
                            tail_r[cross_col[cbase + z] as usize] -= cross_val[cbase + z].value();
                        }
                        for &b in cols {
                            let b = b as usize;
                            let col = &tail_inv[b * e..b * e + e];
                            let mut acc = 0.0;
                            for &az in cols {
                                let az = az as usize;
                                acc += tail_r[az] * col[az];
                            }
                            h += acc * tail_r[b];
                        }
                    }
                    let wp = if w[i].value() <= crate::glm::WEIGHT_CLAMP {
                        0.0
                    } else {
                        let et = crate::dual::Dual::<1> {
                            v: eta[i].value(),
                            d: [1.0],
                        };
                        let (_, w_raw, _) =
                            crate::family::irls_weight_and_resid(family, nb_theta, y[i], et);
                        prior_w[i] * w_raw.d[0]
                    };
                    let a = wp * h;
                    for j in 0..p {
                        logdet_beta[j] += a * x[(i, j)];
                    }
                    for local in 0..qc {
                        logdet_u[f * qc + local] += a * mc[local];
                    }
                    for z in 0..ncz {
                        let b = cross_col[cbase + z] as usize;
                        logdet_u[k_family + b] += a * cross_val[cbase + z].value();
                    }
                }
                // Mode-consistency term for the merit below, formed here because
                // pass B overwrites `logdet_u` in place. See the merit's own
                // comment for why it is needed; δu₀ = u_new − u_prev is this
                // iteration's step toward the mode. `g_u` sits in the `a_rhs`
                // packing while u/u_prev are in RE-column order, so the core part
                // maps through `core_col` (the two coincide only at np == 0).
                let mut gu_dot_du = 0.0;
                for f in 0..s {
                    for local in 0..qc {
                        let c = core_col(f, local);
                        gu_dot_du += logdet_u[f * qc + local] * (u[c] - u_prev[c]).value();
                    }
                }
                for b in 0..e {
                    gu_dot_du +=
                        logdet_u[k_family + b] * (u[k_family + b] - u_prev[k_family + b]).value();
                }
                // pass B: v = A⁻¹gᵤ through this iteration's structured factors
                // (Fisher A — canonical links only reach here). `a_rhs` stages it in
                // the generic T the solve is written for; at T = f64 the two staging
                // loops are an identity copy of values the solve would have moved
                // anyway (the same trick the border below uses for its p columns).
                for c in 0..k {
                    a_rhs[c] = T::from_f64(logdet_u[c]);
                }
                structured_ainv_solve(
                    g,
                    core_blocks,
                    coupling,
                    schur_blk,
                    coup_cols,
                    coup_ptr,
                    // same force_dense → ss fold as the structured_factor call above.
                    if force_dense {
                        None
                    } else {
                        structured_schur.as_deref_mut()
                    },
                    a_rhs,
                );
                for c in 0..k {
                    logdet_u[c] = a_rhs[c].value();
                }
                // pass C: û path, c_β_j −= Σᵢ wᵢ·(mᵢ·v)·xᵢⱼ.
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    let cbase = i * g_cap;
                    let mut sdot = 0.0;
                    for local in 0..qc {
                        sdot += m_core_buf[i * qc + local].value() * logdet_u[f * qc + local];
                    }
                    for z in 0..n_cross[i] as usize {
                        let b = cross_col[cbase + z] as usize;
                        sdot += cross_val[cbase + z].value() * logdet_u[k_family + b];
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
            // terms are not the same order. `dev + ‖u‖²` is stationary at ũ, so it
            // errs by O(‖u−ũ‖²) and from above; `log|A(u)|` errs at FIRST order,
            // either sign. Comparing raw sums across iterates therefore compares
            // points at different mode-offsets, and a trial sitting a first-order
            // step off the mode can score BELOW the attainable optimum — a warm
            // start from a neighbouring θ lands exactly there, and taking it as
            // `l_acc` rejects every later (correct) iterate until the iteration
            // cap. Undo that first-order part with the gradient already at hand:
            // g_u'·δu₀ ≈ log|A(ũ)| − log|A(u)|, a correction that vanishes as δu₀ → 0.
            let l_trial = (dev + pen_u).value() + 2.0 * logdet.value() + gu_dot_du;
            // Accept band charges the mode-consistency correction at BOTH
            // endpoints — reasoning in `pirls_solve_blocked`'s merit test,
            // change together.
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
            // C = X'WX (p×p) via the W∘X GEMM scratch `wx` — structured_schur_fill's
            // X'W̃X block, same product. GEMM fills the full p×p (the old scalar
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
            // B' = X'WM (p×k): zero, then scatter each row's core + crossed columns
            // from the packed `m_core_buf`/`cross_*` nonzeros (structured_schur_fill:436-456).
            for r in 0..p {
                for c in 0..k {
                    xtwm[(r, c)] = 0.0;
                }
            }
            for i in 0..n {
                let f = cluster_ids[i] as usize;
                let wi = w[i].value();
                let cbase = i * g_cap;
                let ncz = n_cross[i] as usize;
                for r in 0..p {
                    let xw = x[(i, r)] * wi;
                    for local in 0..qc {
                        xtwm[(r, core_col(f, local))] += xw * m_core_buf[i * qc + local].value();
                    }
                    for z in 0..ncz {
                        let b = cross_col[cbase + z] as usize;
                        xtwm[(r, k_family + b)] += xw * cross_val[cbase + z].value();
                    }
                }
            }
            // T = A⁻¹B: ainv_mtwx[:, c] = A⁻¹(M'WX)[:, c], one β column at a time via
            // `structured_ainv_solve` reusing THIS iteration's core-block+Schur factors
            // (structured_schur_fill:457-486). `a_rhs` packs/unpacks the RHS/solution in
            // the (f,local)|crossed layout structured_ainv_solve expects.
            // `a_rhs` is the generic solve's buffer while the border's data is
            // f64: the staging loops convert element-wise, which at `T = f64` is
            // an identity copy of values this loop already had to copy.
            for c in 0..p {
                for f in 0..s {
                    for local in 0..qc {
                        a_rhs[f * qc + local] = T::from_f64(xtwm[(c, core_col(f, local))]);
                    }
                }
                for b in 0..e {
                    a_rhs[k_family + b] = T::from_f64(xtwm[(c, k_family + b)]);
                }
                structured_ainv_solve(
                    g,
                    core_blocks,
                    coupling,
                    schur_blk,
                    coup_cols,
                    coup_ptr,
                    // same force_dense → ss fold as the structured_factor call above.
                    if force_dense {
                        None
                    } else {
                        structured_schur.as_deref_mut()
                    },
                    a_rhs,
                );
                for f in 0..s {
                    for local in 0..qc {
                        ainv_mtwx[(core_col(f, local), c)] = a_rhs[f * qc + local].value();
                    }
                }
                for b in 0..e {
                    ainv_mtwx[(k_family + b, c)] = a_rhs[k_family + b].value();
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
            // u[col] −= Σ_j ainv_mtwx[(col, j)]·δβ[j] over the u layout.
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
        // same-point band above cannot itself be a converge trigger).
        // Exact mode: the exit band must track the same merit the accept/halve
        // decision above uses (dev + pen + 2·log|A|), or the loop could settle on a
        // point that is a fixed point of dev+pen alone but still moving in log|A|.
        // The merit's mode-consistency term is deliberately absent here: it is
        // proportional to δu₀, which the band already forces to zero, so including
        // it would change no fixed point and only the iterate count.
        let mixed = (dev + pen).value() + if exact { 2.0 * logdet.value() } else { 0.0 };
        if it + 1 >= min_iters && (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
            converged = true;
            break;
        }
        mixed_prev = mixed;
    }
    (dev, pen, logdet, converged)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glmm::workspace::GlmmWorkspace;

    /// The `f64` override IS the extras path's tail code — moved, not
    /// copied — so there is no second copy left to `==` it against; the
    /// bit-identity dump is the instrument for that identity. What this test can
    /// check is the other arm: the generic default body against the `f64`
    /// override on the same `S`. They are NOT expected to match to bits: the
    /// sparse arm factors under an AMD ordering and the dense default in the
    /// natural one, which is a reassociation of the same Cholesky. The assert
    /// bands them and records the measured gap; the two are the same matrix, not
    /// the same rounding.
    #[test]
    fn tail_default_body_is_the_sparse_arms_oracle() {
        let (_x, y, ids, extra_ids, spec) = crate::glmm::tests::glmm_extras_q1_dataset(0, 6);
        let n = y.len();
        let ws = GlmmWorkspace::for_cluster_spec(2, &spec, n, &[], 1);
        let g = &ws.groupings;
        let e = g.k_crossed();
        let mut ss =
            StructuredSchur::new(g, &ids, &extra_ids, n).expect("6 crossed levels ⇒ e > 0 ⇒ Some");

        // A deterministic, diagonally-dominant SPD matrix built ONLY on the real
        // crossed-incidence pattern `ss` carries for this fixture (the same AMD
        // symbolic factor the production path builds) — off-pattern entries stay
        // 0.0 so the dense arm (reads the full e×e block) and the sparse arm
        // (reads only the pattern) factor the identical matrix.
        let mut s_raw = vec![0.0f64; e * e];
        for a in 0..e {
            s_raw[a * e + a] = 10.0 + 4.0 * a as f64;
        }
        {
            let sym = ss.axx.symbolic();
            let col_ptr = sym.col_ptr();
            let row_idx = sym.row_idx();
            for b in 0..e {
                for &a in &row_idx[col_ptr[b]..col_ptr[b + 1]] {
                    if a != b {
                        s_raw[a * e + b] = 0.1;
                    }
                }
            }
        }

        let mut s_dense = s_raw.clone();
        let mut s_sparse = s_raw.clone();

        let ld_dense = tail_factor_generic::<f64>(&mut s_dense, e).expect("dense PD");
        let ld_sparse =
            <f64 as TailKernel>::tail_factor(&mut s_sparse, e, Some(&mut ss)).expect("sparse PD");
        // Measured on this fixture (e=6, 8 primary clusters, near-complete crossed
        // coupling): dense (natural order) vs sparse (AMD order) Cholesky of the
        // same S agree in the half-log-determinant to within 1e-9 — the two
        // orderings are a reassociation of the same sum, not independent
        // computations.
        assert!(
            (ld_dense - ld_sparse).abs() < 1e-9,
            "half-logdet drifted: dense {ld_dense} sparse {ld_sparse}"
        );

        let mut rhs_dense = vec![1.0f64; e];
        let mut rhs_sparse = vec![1.0f64; e];
        tail_solve_generic::<f64>(&s_dense, e, &mut rhs_dense);
        <f64 as TailKernel>::tail_solve(&s_sparse, e, Some(&mut ss), &mut rhs_sparse);
        for i in 0..e {
            assert!(
                (rhs_dense[i] - rhs_sparse[i]).abs() < 1e-9,
                "tail_solve[{i}] drifted: dense {} sparse {}",
                rhs_dense[i],
                rhs_sparse[i]
            );
        }
    }
}
