//! Structured block+Schur PIRLS solve for the intercept-only crossed/nested-extras regime (`pirls_solve_blocked_extras`, `structured_factor`, `structured_ainv_solve`).

use super::control::{Merit, PirlsControl, Trial};
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
    /// triangle. `core_l` is cluster `f`'s `qc×qc` Crout factor, `cols` its
    /// ascending nonzero crossed columns, `coup` its compact `qc×|cols|`
    /// row-major coupling block (`C_f[local][cols[j]]` at `local·|cols| + j`),
    /// `schur` the dense row-major `e×e` accumulator (lower at `[a·e + b]`,
    /// `a ≥ b`).
    ///
    /// The default body is the rank-1 scalar column walk. `f64` overrides it
    /// with the panel route (`glmm_block_solve_panel` + one triangular faer
    /// matmul) at `qc > 1`, which is the production route there and moves the
    /// dot's internal association; the measurement behind that split lives on
    /// this method's `f64` override.
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

    /// Factor the crossed-tail Schur complement `S = I + schur` (`schur`
    /// dense row-major `e×e`, lower triangle, without the identity) and return
    /// `½·log|S|` — the half-log convention `structured_factor` accumulates
    /// and `deviance.rs` doubles. `None` on a non-PD tail.
    ///
    /// The default body is the crate-own dense Crout factor with the identity
    /// kept apart (`linalg::block_chol_unit_shift`), in place, leaving
    /// `schur` holding `L` for `tail_solve`. `f64` overrides it with the
    /// cached sparse LLT when `ss` is `Some`, which is what the production
    /// path runs: it adds the identity into `schur` first, so `schur` holds
    /// `S` for `tail_inverse`, and its `log|S|` is faer's `Σ ln L_ii`; `ss =
    /// None` selects the dense arm on both.
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

    /// `S⁻¹` into `out` (`e×e` column-major, `out[b·e + a] = (S⁻¹)_{a,b}`)
    /// after `tail_factor`, for the exact-profile leverage and curvature.
    /// `stage` is an `e`-long scratch column. Entries between two tail
    /// components ([`tail_components`](crate::glmm::workspace::tail_components))
    /// are exactly zero and are left as they are; the caller keeps them zero.
    ///
    /// The default body is `e` unit solves through `tail_solve`'s dense arm.
    /// `f64` overrides it with the per-component dense inverse when `ss` is
    /// `Some`.
    fn tail_inverse(
        schur: &[Self],
        e: usize,
        ss: Option<&mut StructuredSchur>,
        stage: &mut [Self],
        out: &mut [f64],
    ) {
        let _ = ss;
        tail_inverse_generic(schur, e, stage, out)
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
            y_panel, dd_temp, ..
        }) = ss.filter(|_| qc != 1)
        else {
            // Production route for qc == 1 (and the ss = None oracle). At
            // qc == 1 the downdate is rank-1: the panel path stages the
            // identical FLOPs (gather → dd_temp → second scatter pass) at
            // ~double the memory traffic, a measured +4–7% per-eval loss on
            // the qc=1 cross6 GLMM cells (binb 1.758→1.870 s; measured 2026-07-14).
            // This scalar walk accumulates each rank-1
            // dot straight into `schur` — already minimal. `qc == 1` is the
            // only measured boundary; no qc>1 GLMM structured cell exists in
            // the grid. Sizing of the panel buffers mirrors this condition in
            // `StructuredSchur::new` — change together.
            if qc == 1 {
                // Rank-1 at qc == 1: S[a,b] −= c_a·y_b with y_b = (c_b/l)/l,
                // the two divisions `glmm_block_solve` performs on a 1×1
                // factor. Each entry receives exactly one term per call, so
                // walking row `a` outer (one contiguous row of `schur`) and
                // `b ≤ a` inner (cols ascending ⇒ jb ≤ ja) changes no sum:
                // bit-identical to `tail_downdate_generic`'s column walk. The
                // `0.0 +` keeps that walk's `acc = 0.0; acc += …` sign-of-zero
                // behaviour.
                // `y` is staged in fixed stack chunks so wide clusters need
                // no heap.
                const CHUNK: usize = 256;
                let l = core_l[0];
                let coup = &coup[..e_f];
                let mut ybuf = [0.0_f64; CHUNK];
                for c0 in (0..e_f).step_by(CHUNK) {
                    let c1 = (c0 + CHUNK).min(e_f);
                    let yb = &mut ybuf[..c1 - c0];
                    for (y, &c) in yb.iter_mut().zip(&coup[c0..c1]) {
                        *y = c / l / l;
                    }
                    for (ja, (&a, &ca)) in cols.iter().zip(coup).enumerate().skip(c0) {
                        let a = a as usize;
                        let row = &mut schur[a * e..a * e + a + 1];
                        let hi = (ja + 1).min(c1);
                        for (&b, &y) in cols[c0..hi].iter().zip(&yb[..hi - c0]) {
                            row[b as usize] -= 0.0 + ca * y;
                        }
                    }
                }
                return;
            }
            return tail_downdate_generic(core_l, qc, coup, e, cols, schur);
        };
        let cpan = &coup[..qc * e_f];
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
                for b in 0..e {
                    schur[b * e + b] += 1.0;
                }
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
            // Dense fallback: the crate-own Crout factor (test cross-check / defensive).
            None => tail_factor_generic(schur, e),
        }
    }

    fn tail_inverse(
        schur: &[f64],
        e: usize,
        ss: Option<&mut StructuredSchur>,
        stage: &mut [f64],
        out: &mut [f64],
    ) {
        use faer::linalg::cholesky::llt;
        // Widest component inverted by the crate's own Crout on stack arrays;
        // wider ones go to faer's blocked dense factor and inverse. Measured
        // 2026-09-29 on grouseticks (63 components of 1-8 columns): 33 µs per
        // PIRLS iteration through faer, 10 µs this way.
        const SMALL: usize = 8;
        let Some(ss) = ss else {
            return tail_inverse_generic(schur, e, stage, out);
        };
        // Per component rather than `e` unit solves through the sparse factor:
        // `S` is block-diagonal over the components, so each block of `S⁻¹` is
        // the inverse of the matching block of `S`, and every solve through
        // the whole factor spent most of its work on other components' exact
        // zeros (nested tails) or ran scalar column walks over a dense factor
        // (crossed tails, one component). `schur` still holds the unfactored
        // `S` here: the sparse `tail_factor` arm only gathers from it. The
        // blocks are factored anew, so `S⁻¹` moves at round-off against the
        // unit solves (≤ 5e-13 relative on the tail cells measured).
        let factored = 'blocks: {
            let StructuredSchur {
                comp_ptr,
                comp_cols,
                inv_blk,
                inv_out,
                inv_mem,
                ..
            } = &mut *ss;
            for c in 0..comp_ptr.len() - 1 {
                let cols = &comp_cols[comp_ptr[c] as usize..comp_ptr[c + 1] as usize];
                let m = cols.len();
                if m <= SMALL {
                    // Row-major lower triangle, the layout `block_chol` factors.
                    let mut blk = [0.0f64; SMALL * SMALL];
                    for (i, &a) in cols.iter().enumerate() {
                        for (j, &b) in cols.iter().enumerate().take(i + 1) {
                            blk[i * m + j] = schur[a as usize * e + b as usize];
                        }
                    }
                    if !glmm_block_chol(&mut blk[..m * m], m) {
                        break 'blocks false;
                    }
                    for (j, &b) in cols.iter().enumerate() {
                        let mut col = [0.0f64; SMALL];
                        col[j] = 1.0;
                        glmm_block_solve(&blk[..m * m], m, &mut col[..m]);
                        for (i, &a) in cols.iter().enumerate() {
                            out[b as usize * e + a as usize] = col[i];
                        }
                    }
                    continue;
                }
                let blk = &mut inv_blk[..m * m];
                for (j, &b) in cols.iter().enumerate() {
                    for (i, &a) in cols.iter().enumerate().skip(j) {
                        blk[j * m + i] = schur[a as usize * e + b as usize];
                    }
                }
                if llt::factor::cholesky_in_place(
                    faer::MatMut::from_column_major_slice_mut(blk, m, m),
                    Default::default(),
                    faer::Par::Seq,
                    faer::dyn_stack::MemStack::new(inv_mem),
                    Default::default(),
                )
                .is_err()
                {
                    break 'blocks false;
                }
                // `inverse` fills the lower triangle only; mirrored on scatter.
                let inv = &mut inv_out[..m * m];
                llt::inverse::inverse(
                    faer::MatMut::from_column_major_slice_mut(inv, m, m),
                    faer::MatRef::from_column_major_slice(&inv_blk[..m * m], m, m),
                    faer::Par::Seq,
                    faer::dyn_stack::MemStack::new(inv_mem),
                );
                for (j, &b) in cols.iter().enumerate() {
                    for (i, &a) in cols.iter().enumerate().skip(j) {
                        let v = inv[j * m + i];
                        out[b as usize * e + a as usize] = v;
                        out[a as usize * e + b as usize] = v;
                    }
                }
            }
            true
        };
        if !factored {
            tail_inverse_sparse(ss, e, stage, out);
        }
    }

    fn tail_solve(schur: &[f64], e: usize, ss: Option<&mut StructuredSchur>, rhs: &mut [f64]) {
        match ss {
            Some(ss) => {
                // Reconstruct the factor from the cached symbolic and values (no
                // re-factor: faer's `LltRef::new` is a Copy wrapper over two refs,
                // NOT a re-factorization) and back-solve the single e-col.
                let llt = faer::sparse::linalg::cholesky::LltRef::new(&ss.symbolic, &ss.l_values);
                let rhs = faer::MatMut::from_column_major_slice_mut(&mut rhs[..e], e, 1);
                llt.solve_in_place_with_conj(
                    faer::Conj::No,
                    rhs,
                    faer::Par::Seq,
                    faer::dyn_stack::MemStack::new(&mut ss.solve_mem),
                );
            }
            // Dense fallback (test cross-check / e>0 with no cached factor): schur
            // holds the dense L that the dense-factor branch produced.
            None => tail_solve_generic(schur, e, rhs),
        }
    }
}

/// The default body of [`TailKernel::tail_inverse`]: `e` unit solves through
/// the dense `L` that `tail_factor`'s dense arm left in `schur`.
pub(crate) fn tail_inverse_generic<T: Scalar>(
    schur: &[T],
    e: usize,
    stage: &mut [T],
    out: &mut [f64],
) {
    for b in 0..e {
        stage[..e].fill(T::ZERO);
        stage[b] = T::ONE;
        tail_solve_generic(schur, e, &mut stage[..e]);
        for a in 0..e {
            out[b * e + a] = stage[a].value();
        }
    }
}

/// `e` unit solves through the cached sparse factor: the fallback of `f64`'s
/// [`TailKernel::tail_inverse`] when a component block fails to factor, which
/// `S` being positive definite (its sparse factor succeeded) rules out short
/// of round-off at the edge of definiteness.
fn tail_inverse_sparse(ss: &mut StructuredSchur, e: usize, stage: &mut [f64], out: &mut [f64]) {
    for b in 0..e {
        stage[..e].fill(0.0);
        stage[b] = 1.0;
        <f64 as TailKernel>::tail_solve(&[], e, Some(&mut *ss), &mut stage[..e]);
        out[b * e..b * e + e].copy_from_slice(&stage[..e]);
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
    let e_f = cols.len();
    let mut ycol = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    for (jb, &b) in cols.iter().enumerate() {
        let b = b as usize;
        for local in 0..qc {
            ycol[local] = coup[local * e_f + jb];
        }
        glmm_block_solve(&core_l[..qc * qc], qc, &mut ycol[..qc]);
        for (ja, &a) in cols.iter().enumerate() {
            let a = a as usize;
            if a < b {
                continue;
            }
            let mut acc = T::ZERO;
            for local in 0..qc {
                acc += coup[local * e_f + ja] * ycol[local];
            }
            schur[a * e + b] -= acc;
        }
    }
}

/// The default body of [`TailKernel::tail_factor`]: the crate-own dense Crout
/// factor of `I + schur` with the identity kept apart, returning its half
/// log-determinant (`linalg::block_chol_unit_shift`); the same arm `f64`'s
/// override runs when `ss` is `None`.
pub(crate) fn tail_factor_generic<T: Scalar>(schur: &mut [T], e: usize) -> Option<T> {
    crate::linalg::block_chol_unit_shift(&mut schur[..e * e], e)
}

/// The default body of [`TailKernel::tail_solve`]: forward/back substitution
/// through `glmm_block_solve` on the dense `L` that `tail_factor` left in
/// `schur` — the same arm `f64`'s override runs when `ss` is `None`.
pub(crate) fn tail_solve_generic<T: Scalar>(schur: &[T], e: usize, rhs: &mut [T]) {
    glmm_block_solve(&schur[..e * e], e, &mut rhs[..e]);
}

/// `h + Σ_{b∈cols} tail_r[b] · Σ_{a∈cols} tail_r[a] · tail_inv[b·e + a]`,
/// accumulated onto `h` one `b` at a time: the `rᵢ'S⁻¹rᵢ` half of a row's RE
/// leverage, walked over the whole cluster's coupling columns. The equality
/// oracle for the row-local three-term form
/// `yᵢ'G_f yᵢ − 2·yᵢ'(H_f tᵢ) + tᵢ'S⁻¹tᵢ` that `pirls_solve_blocked_extras`
/// ships: the row site runs both under `cfg(test)` and asserts they agree to
/// `1e-12` relative to the magnitude of the three terms summed — the same
/// shape as `TailKernel::tail_downdate`'s `ss = None` equality arm, but
/// needing no runtime switch, because under `cfg(test)` this walk simply runs
/// beside the shipped one. The two differ only in the association of one
/// quadratic form, so the band is round-off, not tolerance.
#[cfg(test)]
fn tail_quadratic_form(h: f64, tail_inv: &[f64], e: usize, cols: &[u32], tail_r: &[f64]) -> f64 {
    let mut h = h;
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
    h
}

/// Factor the structured `A = [[D, C], [C', E]]` in place: Crout-factor each
/// core block `core_blocks[f]` (holding `D_f` on entry, the identity added at
/// each pivot by `block_chol_unit_shift`; its lower `L` of `D_f + I` on return)
/// and build + factor the Schur complement `schur_blk` (holding `E` on entry,
/// `S = (E+I) − Σ_f C_f' A_f⁻¹ C_f` less its identity, then its `L` on return;
/// both tail arms add the identity when they factor). Returns
/// `Some(log|A|) = Σ_f log|A_f| + log|S|` (Schur determinant identity), or `None`
/// on a non-PD core block / Schur. The Schur factor itself routes through
/// `TailKernel::tail_factor`: the cached sparse LLT when `ss` is `Some` at `T = f64`
/// (the production route), the crate-own dense Crout — `tail_factor`'s default
/// body — at any non-`f64` `T` or when `ss` is `None`. `coupling` holds `C_f`
/// in the compact per-cluster layout (`StructuredScratch::coupling`),
/// unchanged. Shared by `pirls_solve_blocked_extras` (per iteration) and
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
/// `A_f⁻¹` solve + one triangular `C_f'·Y` matmul through `ss`'s scratch, as in
/// the LMM sparse tail): each `S[a][b]` receives exactly one subtraction per
/// touching cluster in the same `f` order; only the dot's internal association
/// differs from the scalar walk, a reassociation that moves results by
/// round-off (see `SparseTail`'s doc). The panel path serves
/// `qc > 1` (vector primary and/or nested REs). At `qc == 1` the downdate is
/// rank-1 and the scalar column-at-a-time walk is the **production route**
/// (routed via the `qc != 1` filter inside `TailKernel::tail_downdate`, whose `f64`
/// override documents the panel-vs-scalar measurement behind the split); the
/// same walk is also the `ss = None` arm, which is the panel path's equality
/// oracle at `qc > 1`.
pub(crate) fn structured_factor<T: TailKernel>(
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
        logdet += crate::linalg::block_chol_unit_shift(&mut core_blocks[cb..cb + qc * qc], qc)?;
        // S −= C_f' A_f⁻¹ C_f (lower triangle) over cluster f's NONZERO crossed
        // columns, via TailKernel::tail_downdate (panel-vs-scalar routing documented
        // there and on structured_factor's own doc comment above).
        let coup = qc * coup_ptr[f] as usize;
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        let e_f = cols.len();
        if e_f == 0 {
            continue;
        }
        T::tail_downdate(
            &core_blocks[cb..cb + qc * qc],
            qc,
            &coupling[coup..coup + qc * e_f],
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
    // the dense arm's `linalg::block_chol_unit_shift` at dimension 0 (its
    // `for j in 0..0` loop) over the `.max(1)`-sized `schur_blk` is an
    // indexing-free no-op returning `Some(0)` — a NaN-free, logdet-preserving skip.
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
        let coup = qc * coup_ptr[f] as usize;
        glmm_block_solve(
            &core_blocks[cb..cb + qc * qc],
            qc,
            &mut a_rhs[gcb..gcb + qc],
        );
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        let nc = cols.len();
        for (j, &b) in cols.iter().enumerate() {
            let b = b as usize;
            let mut acc = T::ZERO;
            for local in 0..qc {
                acc += coupling[coup + local * nc + j] * a_rhs[gcb + local];
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
        let coup = qc * coup_ptr[f] as usize;
        let mut v = [T::ZERO; MAX_PRIMARY_Q];
        let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
        let nc = cols.len();
        #[allow(clippy::needless_range_loop)]
        for local in 0..qc {
            let mut acc = T::ZERO;
            for (j, &b) in cols.iter().enumerate() {
                let b = b as usize;
                acc += coupling[coup + local * nc + j] * a_rhs[k_family + b];
            }
            v[local] = acc;
        }
        glmm_block_solve(&core_blocks[cb..cb + qc * qc], qc, &mut v[..qc]);
        for local in 0..qc {
            a_rhs[gcb + local] -= v[local];
        }
    }
}

/// The structured-extras Laplace layout: `A = [[D, C], [C', E]] + I`, with `D`
/// block-diagonal over the primary clusters (each `q_core×q_core` core block is
/// the primary RE column plus its nested-within-primary children) and `C` the
/// only dense coupling, to the thin crossed width `e`. `M = ZΛ`'s nonzeros
/// arrive packed — a contiguous `q_core` core slice per row plus `n_cross[i]`
/// crossed entries. Borrowed from the buffers `pirls_solve_blocked_extras`
/// destructures.
pub(crate) struct StructuredFactor<'a, T> {
    /// `s·q_core²`, cluster `f` at `f·q_core²`, row-major lower triangle. Left
    /// FACTORED (per-cluster Crout `L`) by `factor_logdet`.
    pub(crate) core_blocks: &'a mut [T],
    /// Cluster `f`'s `C_f` on its coupling columns, `q_core×|cols_f|`
    /// row-major, at `q_core·coup_ptr[f]` (`StructuredScratch::coupling`).
    pub(crate) coupling: &'a mut [T],
    /// `e×e` row-major lower triangle: the crossed `E`, then the Schur
    /// complement `S = (E+I) − Σ_f C_f'A_f⁻¹C_f`, then its factor.
    pub(crate) schur_blk: &'a mut [T],
    /// len `k_total`, packed `[g_core in (f,local) order | g_e]`: `M'r`, then
    /// `A⁻¹M'r` in place.
    pub(crate) a_rhs: &'a mut [T],
    /// len `n`: the row scores `ρᵢ` the u-step's right-hand side scatters,
    /// written by `scatter`'s fold.
    pub(crate) mu: &'a mut [T],
    /// Row-major `n×q_core`: `M`'s core slice per row.
    pub(crate) m_core_buf: &'a [T],
    /// `M`'s crossed values, one per slot (`StructuredScratch::cross_theta`):
    /// slot `z` of every row holds the same value.
    pub(crate) cross_theta: &'a [T],
    /// Crossed column of each row's live slots, `n×MAX_EXTRA_GROUPINGS`,
    /// `n_cross[i]` live per row.
    pub(crate) cross_col: &'a [u32],
    /// Live crossed entries per row.
    pub(crate) n_cross: &'a [u8],
    /// Per-cluster CSR of `C_f`'s nonzero crossed columns.
    pub(crate) coup_cols: &'a [u32],
    pub(crate) coup_ptr: &'a [u32],
    /// Each crossed entry's position in its cluster's columns, same indexing
    /// as `cross_col` (`StructuredPattern::coup_pos`).
    pub(crate) coup_pos: &'a [u32],
    /// The cached sparse tail factor; `None` on nested-only shapes (`e = 0`),
    /// and under the test-only `force_dense_schur`, which routes the tail
    /// through the dense Crout arm instead.
    pub(crate) structured_schur: Option<&'a mut StructuredSchur>,
    pub(crate) g: &'a crate::lmm::LmmGroupings,
    /// len `n`: the primary cluster each row loads.
    pub(crate) cluster_ids: &'a [u32],
    pub(crate) family: Family,
    pub(crate) nb_theta: f64,
}

impl<T: TailKernel> LaplaceFactor<T> for StructuredFactor<'_, T> {
    fn eta_from_mode(&mut self, u: &[T], eta_fixed: &[T], eta: &mut [T], y: &[f64], n: usize) -> T {
        let (m_core_buf, cross_theta, cross_col, n_cross) = (
            self.m_core_buf,
            self.cross_theta,
            self.cross_col,
            self.n_cross,
        );
        let (g, cluster_ids) = (self.g, self.cluster_ids);
        let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
        let q = g.primary_q;
        let np = g.nested_per_parent;
        let qc = q + np;
        let s = g.n_primary;
        let prim_width = q * s;
        let k_family = qc * s;
        let mut yeta = T::ZERO;
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let m_core = &m_core_buf[i * qc..i * qc + qc];
            let mut mui = T::ZERO;
            #[allow(clippy::needless_range_loop)]
            for local in 0..qc {
                let col = if local < q {
                    f * q + local
                } else {
                    prim_width + f * np + (local - q)
                };
                mui += m_core[local] * u[col];
            }
            let cbase = i * g_cap;
            for z in 0..n_cross[i] as usize {
                let b = cross_col[cbase + z] as usize;
                mui += cross_theta[z] * u[k_family + b];
            }
            eta[i] = eta_fixed[i] + mui;
            yeta += eta[i].scale(y[i]);
        }
        yeta
    }

    fn scatter(
        &mut self,
        w: &[T],
        prob: &[T],
        eta: &[T],
        y: &[f64],
        prior_w: &[f64],
        _weighted: bool,
        n: usize,
        mut dual: Option<&mut DualStep<T>>,
    ) {
        let StructuredFactor {
            core_blocks,
            coupling,
            schur_blk,
            a_rhs,
            mu,
            m_core_buf,
            cross_theta,
            cross_col,
            n_cross,
            coup_ptr,
            coup_pos,
            g,
            cluster_ids,
            family,
            nb_theta,
            ..
        } = self;
        let (family, nb_theta) = (*family, *nb_theta);
        let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
        let qc = g.primary_q + g.nested_per_parent;
        let s = g.n_primary;
        let e = g.k_crossed();
        let k_family = qc * s;
        let k = k_family + e;
        // Row score ρᵢ (`row_rho`), so the scattered RHS is M'ρ, the u-step's
        // increment right-hand side before its `− u` (see `pirls_solve_blocked`).
        let mut probit_rows = crate::family::ProbitRows::new(family, eta, prob);
        for i in 0..n {
            mu[i] = super::row_rho(
                family,
                nb_theta,
                i,
                y[i],
                prior_w[i],
                eta[i],
                prob[i],
                probit_rows.as_mut(),
            );
        }
        // --- scatter — wᵢmᵢmᵢ' into D_f/C_f/E (lower tri), rᵢmᵢ into g ---
        for v in core_blocks[..s * qc * qc].iter_mut() {
            *v = T::ZERO;
        }
        // The compact coupling holds the pattern's entries only, all of which
        // the scatter can write.
        let coup_end = qc * coup_ptr[s] as usize;
        for v in coupling[..coup_end].iter_mut() {
            *v = T::ZERO;
        }
        // schur_blk stays a dense clear (unlike coupling above): the dense-fallback
        // branch of structured_factor (TailKernel::tail_factor's `ss = None` arm)
        // Cholesky-factors it IN PLACE, producing fill-in outside the
        // coup_cols×coup_cols pattern — a sparse zero would leave stale L-factor
        // residue from the prior iteration.
        for v in schur_blk[..e * e].iter_mut() {
            *v = T::ZERO;
        }
        for v in a_rhs[..k].iter_mut() {
            *v = T::ZERO;
        }
        // Twin zeroing, mirroring the Fisher zeroing above buffer-for-buffer:
        // `obs_core_blocks`/`obs_coupling` clear in full, and `obs_schur_blk`
        // stays a dense clear for the same fill-in reason `schur_blk` is
        // above. The twin's right-hand side is the Fisher one, copied in by the
        // solve.
        if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
            for v in d.obs_core_blocks[..s * qc * qc].iter_mut() {
                *v = T::ZERO;
            }
            for v in d.obs_coupling[..coup_end].iter_mut() {
                *v = T::ZERO;
            }
            for v in d.obs_schur_blk[..e * e].iter_mut() {
                *v = T::ZERO;
            }
        }
        // Reads the packed core slice + crossed nonzeros directly — no per-row stack
        // copy / rescan.
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let wi = w[i];
            let ri = mu[i]; // effective residual
            let m_core = &m_core_buf[i * qc..i * qc + qc];
            let cbase = i * g_cap;
            let ncz = n_cross[i] as usize;
            let cb = f * qc * qc;
            let gcb = f * qc;
            let coup = qc * coup_ptr[f] as usize;
            let nc = (coup_ptr[f + 1] - coup_ptr[f]) as usize;
            for r in 0..qc {
                let mr = m_core[r];
                a_rhs[gcb + r] += mr * ri;
                let wmr = wi * mr;
                for c in 0..=r {
                    core_blocks[cb + r * qc + c] += wmr * m_core[c];
                }
                for z in 0..ncz {
                    coupling[coup + r * nc + coup_pos[cbase + z] as usize] += wmr * cross_theta[z];
                }
            }
            for z in 0..ncz {
                let b = cross_col[cbase + z] as usize;
                let vb = cross_theta[z];
                a_rhs[k_family + b] += vb * ri;
                let wvb = wi * vb;
                for z2 in 0..ncz {
                    let b2 = cross_col[cbase + z2] as usize;
                    if b2 <= b {
                        schur_blk[b * e + b2] += wvb * cross_theta[z2];
                    }
                }
            }
            // Twin scatter, mirroring the Fisher matrix scatter above
            // buffer-for-buffer, off the observed weight.
            if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
                let wo = crate::family::observed_weight(
                    family, nb_theta, y[i], prior_w[i], eta[i], prob[i], wi,
                );
                for r in 0..qc {
                    let mr = m_core[r];
                    let wmr = wo * mr;
                    #[allow(clippy::needless_range_loop)]
                    for c in 0..=r {
                        d.obs_core_blocks[cb + r * qc + c] += wmr * m_core[c];
                    }
                    for z in 0..ncz {
                        d.obs_coupling[coup + r * nc + coup_pos[cbase + z] as usize] +=
                            wmr * cross_theta[z];
                    }
                }
                for z in 0..ncz {
                    let b = cross_col[cbase + z] as usize;
                    let vb = cross_theta[z];
                    let wvb = wo * vb;
                    for z2 in 0..ncz {
                        let b2 = cross_col[cbase + z2] as usize;
                        if b2 <= b {
                            d.obs_schur_blk[b * e + b2] += wvb * cross_theta[z2];
                        }
                    }
                }
            }
        }
    }

    fn factor_logdet(&mut self) -> Option<T> {
        let StructuredFactor {
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            structured_schur,
            g,
            ..
        } = self;
        // The +I ridge is added at each pivot (`structured_factor`).
        structured_factor(
            g,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            structured_schur.as_deref_mut(),
        )
    }

    fn solve_in_place(&mut self) {
        let StructuredFactor {
            core_blocks,
            coupling,
            schur_blk,
            a_rhs,
            coup_cols,
            coup_ptr,
            structured_schur,
            g,
            ..
        } = self;
        structured_ainv_solve(
            g,
            core_blocks,
            coupling,
            schur_blk,
            coup_cols,
            coup_ptr,
            structured_schur.as_deref_mut(),
            a_rhs,
        );
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
/// coupling `C_f`, the crossed `E`, and the RHS `g = M'ρ − u` of the increment
/// step (see `pirls_solve_blocked`); then a
/// Schur complement on `e` solves it: factor each `A_f = chol(D_f+I)`, form
/// `S = (E+I) − Σ_f C_f' A_f⁻¹ C_f`, back-substitute `δu_e = S⁻¹(g_e − Σ_f C_f'
/// A_f⁻¹ g_{core,f})` then `δu_{core,f} = A_f⁻¹ g_{core,f} − A_f⁻¹ C_f δu_e`,
/// `u += δu`, and
/// `log|A| = Σ_f log|A_f| + log|S|` (Schur determinant identity). `e=0` (nested
/// only) skips the Schur entirely. Collapses the dense `O(n·k²)` Gram + `O(k³)`
/// factor to `O(n·q_core²)` scatter + `O(s·q_core³ + e³)` factor. NOT bit-identical
/// to `pirls_solve_packed` (scatter vs GEMM accumulation) but the same estimator.
/// The `M = ZΛ` nonzeros arrive PACKED (`m_core_buf` core slice + `cross_*`/
/// `n_cross` crossed entries the caller filled via `build_packed_m`), never the
/// dense faer `m`. A converged solve ends by re-evaluating η/μ/W at the
/// returned `u` and rebuilding and refactoring the structured `A` there
/// ([`evaluate_at_mode`]), so it leaves
/// `core_blocks` (per-cluster L) + `schur_blk` (Schur L) + `coupling` FACTORED at
/// that iterate for `structured_schur_fill` to reuse, and eta/prob/w/u filled. Returns
/// `(dev, ‖u‖², log|A|, converged)`; a non-PD core block / Schur ⇒
/// `(NaN, NaN, NaN, false)`. Iterates from the caller-provided `u` (warm-start
/// seed); the caller owns resetting it per fit. `a_rhs` (length ≥ k_total) is the
/// RHS scratch, packed `[g_core in (f,local) order | g_e]`.
///
/// Step-halving differs by `beta_step`. `Fixed` and `Profile { exact: None }`
/// (the PQL border) share `pirls_solve_packed`'s retrospective `dev + pen_u` band,
/// halving `u` back toward `u_prev`. `Profile { exact: Some(_) }` instead
/// halves on the exact Laplace merit (`dev + pen_u + 2·logdet` plus its
/// mode-consistency correction — see the assembly below) and, at the top of
/// the loop, on `infeasible` alone; it also treats a non-PD structured factor
/// on the trial `u` as a rejected step rather than a hard failure, backtracking
/// toward `(u_acc, beta_prev)` the same way ([`PirlsControl`]). Every mode's halved retry re-enters the top and re-runs
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
/// `T = A⁻¹B` via `p` `structured_ainv_solve` calls (one per β column: p solves
/// per iteration), `S_β = C − B'T`, `δβ = S_β⁻¹·(X'ρ − B'δu₀)`,
/// then `u_joint = u_new − T·δβ` and `β += δβ` written back through `beta`; the
/// exact profile adds log|A|'s gradient and curvature in β to that step
/// (`logdet_beta_curvature`). A non-PD S_β surfaces as `(NaN, NaN, NaN, false)`. `B`'s scatter reads the packed
/// `m_core_buf`/`cross_*` nonzeros directly (as `structured_schur_fill` does).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pirls_solve_blocked_extras<T: TailKernel>(
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
    scratch: &mut PirlsScratch<T>,
    structured: &mut StructuredScratch<T>,
    pattern: &mut StructuredPattern,
    // Dual derivative kernels' per-solve controls (see `DualStep`); `None` on
    // every f64 fit-path call.
    mut dual: Option<&mut DualStep<T>>,
    // n×p scratch: the design residual rows `rᵢ = xᵢ − Tᵀmᵢ` of the Profile β
    // border (`residual_border`).
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`Data::offset`), added into
    // `eta_fixed` by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
    // Observation-only — mirrors `pirls_solve_packed`'s `counters`, where
    // the contract is stated.
    counters: &mut crate::counters::EvalCounters,
) -> (T, T, T, bool) {
    // The β-Schur border below runs in f64 (its p×p Cholesky is faer's, and
    // reproducing it generically would move f64 bits). A non-f64
    // instantiation must use BetaStep::Fixed — the derivative entry points do.
    // Loud, not silent: dropping derivatives here would be a wrong gradient.
    assert!(
        T::IS_F64 || matches!(beta_step, BetaStep::Fixed),
        "BetaStep::Profile is f64-only on the extras path"
    );
    let PirlsScratch {
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        a_rhs,
        x_rows,
        bt_rows,
        ..
    } = scratch;
    let StructuredScratch {
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_theta,
    } = structured;
    let force_dense = pattern.force_dense_schur();
    let StructuredPattern {
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_pos,
        structured_schur,
        ..
    } = pattern;
    let structured_schur = structured_schur.as_mut().filter(|_| !force_dense);
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
    // The `A`-layout view over the buffers the assembly, the factor and the
    // `A⁻¹` solve own. The packed `M` nonzeros and the coupling CSR are
    // read-only within one solve, so the loop below keeps reading them directly
    // alongside the view.
    let mut layout = StructuredFactor {
        core_blocks: &mut core_blocks[..],
        coupling: &mut coupling[..],
        schur_blk: &mut schur_blk[..],
        a_rhs: &mut a_rhs[..],
        mu: &mut mu[..],
        m_core_buf: &m_core_buf[..],
        cross_theta: &cross_theta[..],
        cross_col: &cross_col[..],
        n_cross: &n_cross[..],
        coup_cols: &coup_cols[..],
        coup_ptr: &coup_ptr[..],
        coup_pos: &coup_pos[..],
        structured_schur,
        g,
        cluster_ids,
        family,
        nb_theta,
    };
    // Row-major `X` for the `f64` β-Profile row passes, as in
    // `pirls_solve_blocked` (`PirlsScratch::x_rows`): same values, and every
    // accumulator in the same row order as the `MatRef` path.
    let rows_ok = T::IS_F64
        && dual.is_none()
        && p > 0
        && matches!(beta_step, BetaStep::Profile { .. })
        && x_rows.len() >= n * p
        && bt_rows.len() >= k * p;
    if rows_ok {
        super::blocked::fill_x_rows(x, &mut x_rows[..n * p], n, p);
    }
    let xr_all: &[f64] = if rows_ok { &x_rows[..n * p] } else { &[] };
    let fx = FixedPart {
        x,
        rows: rows_ok.then_some(xr_all),
        n,
        p,
        offset,
    };
    // η_fixed,ᵢ = Σ_j x·β, hoisted out of the iteration (β fixed within the solve).
    refresh_fixed(&fx, beta, eta_fixed);
    // First-trial backtrack seeds (u_prev = 0, beta_prev = caller's β) — only
    // the domain-infeasibility trigger can read them; see `pirls_solve_packed`.
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
    let mut converged = false;
    let mut ctl = PirlsControl::new(family, &beta_step, dual.as_deref_mut(), pirls_tol_override);
    let (mut dev, mut pen, mut logdet) = (T::from_f64(f64::NAN), T::from_f64(f64::NAN), T::ZERO);
    for it in 0..PIRLS_MAX_ITERS {
        counters.set_pirls_iters(it + 1);
        // --- trial evaluation at the CURRENT u (pass 1 + pass 2). On a fresh accept
        // this is the newly-stepped u; after a halving `continue` it is the
        // backtracked u. Either way the recompute IS the trial evaluation. ---
        // --- pass 1: η-pass — ηᵢ = η_fixed,ᵢ + (Mu)ᵢ over the row's nonzeros ---
        // Reads the packed M nonzeros (contiguous q_core core slice + n_cross[i]
        // crossed entries) `build_packed_m` filled — no faer indexing, crossed term
        // O(G) not O(e). Σ y·η comes off the same pass.
        let yeta = layout.eta_from_mode(&u[..], &eta_fixed[..], &mut eta[..], y, n);
        // --- pass 2: η[] → prob[]/w[] + deviance, through the shared family
        // kernel (clamps η in place; `infeasible` flags any raw η past the
        // link's clamp bounds, `family::eta_infeasible`; mirrors
        // `pirls_solve_packed`). ---
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
        // ‖u‖² is at the CURRENT trial u; `u[..k]` spans the core (RE-column
        // order) + crossed tail.
        let pen_u = pen_sq(&u[..k]);
        let penalized = dev + pen_u;
        match ctl.trial(penalized.value(), infeasible) {
            Trial::Proceed => {}
            Trial::RefineJump => {
                // Past the η clamp: one jump back inside it, toward `u_prev`.
                jump_back(
                    &mut layout,
                    family,
                    y,
                    &fx,
                    &BetaStep::Fixed,
                    u,
                    u_prev,
                    beta,
                    eta_fixed,
                    prob,
                    eta,
                    w,
                    k,
                );
                continue;
            }
            Trial::RefineHalve => {
                halve_u(&mut u[..k], &u_prev[..k]);
                continue;
            }
            Trial::Jump => {
                // Past the η clamp: one jump to the largest power-of-two
                // fraction of the step that is back inside it (`clamp_jump`).
                jump_back(
                    &mut layout,
                    family,
                    y,
                    &fx,
                    &beta_step,
                    u,
                    u_prev,
                    beta,
                    eta_fixed,
                    prob,
                    eta,
                    w,
                    k,
                );
                continue;
            }
            Trial::Halve => {
                // Last full step overshot: halve δu = u − u_prev and re-enter the top
                // (the recompute above is the trial evaluation of the halved step; a
                // halved retry re-runs structured_factor/ainv_solve, by design).
                // Exact mode halves toward the last ACCEPTED u (`u_acc`, set by the
                // post-sweep merit test below) since `u_prev` there holds the
                // just-rejected trial, not an accepted point.
                if let BetaStep::Profile {
                    exact: Some(ex), ..
                } = &beta_step
                {
                    halve_u_to_acc(&mut u[..k], &ex.u_acc[..k]);
                } else {
                    halve_u(&mut u[..k], &u_prev[..k]);
                }
                // Profile mode: the trial point is the JOINT (u,β) step, so the
                // backtrack halves β toward `beta_prev` in lockstep with u, then
                // refreshes η_fixed for the re-evaluation at the top.
                if let BetaStep::Profile { beta_prev, .. } = &beta_step {
                    halve_beta(beta, beta_prev);
                    refresh_fixed(&fx, beta, eta_fixed);
                }
                continue;
            }
            Trial::Fail => return fail(),
        }
        // Snapshot the accepted iterate for the next backtrack, and take a fresh
        // full step from it. `u_prev` is unconditional — the β-Schur border needs
        // δu₀ = u_new − u_prev regardless of mode.
        u_prev[..k].copy_from_slice(&u[..k]);
        if !ctl.exact {
            // Profile mode: snapshot the accepted β as the β-halving twin of u_prev.
            if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
                for j in 0..p {
                    beta_prev[j] = beta[j].value();
                }
            }
        }
        // Newton step on a link whose exact curvature differs from Fisher —
        // reasoning in `pirls_solve_blocked`, change together.
        if dual.is_none() {
            observed_weights_in_place(
                family,
                nb_theta,
                y,
                prior_w,
                &eta[..],
                &prob[..],
                &mut w[..],
                n,
            );
        }
        // --- pass 3: the scatter — wᵢmᵢmᵢ' into D_f/C_f/E (lower tri), rᵢmᵢ
        // into g — with the dual kernels' observed twin filled in the same row
        // pass. ---
        layout.scatter(
            &w[..],
            &prob[..],
            &eta[..],
            y,
            prior_w,
            weighted,
            n,
            dual.as_deref_mut(),
        );
        // The increment right-hand side M'ρ − u (see `pirls_solve_blocked`,
        // change together), in the `a_rhs` packing; the observed twin solves
        // the same one on its own factor.
        for f in 0..s {
            for local in 0..qc {
                layout.a_rhs[f * qc + local] -= u[core_col(f, local)];
            }
        }
        for b in 0..e {
            layout.a_rhs[k_family + b] -= u[k_family + b];
        }
        if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
            d.obs_rhs[..k].copy_from_slice(&layout.a_rhs[..k]);
        }
        // --- +I, factor (core blocks + Schur), then apply A⁻¹ to the scattered RHS ---
        logdet = match layout.factor_logdet() {
            Some(ld) => ld,
            None => {
                // An unevaluable trial in exact mode is a REJECTED trial, not
                // a failed solve ([`PirlsControl::factor_failed`]).
                if ctl.factor_failed() {
                    if let BetaStep::Profile {
                        exact: Some(ex),
                        beta_prev,
                        ..
                    } = &beta_step
                    {
                        halve_prev_to_acc(&mut u[..k], &u_prev[..k], &ex.u_acc[..k]);
                        halve_beta(beta, beta_prev);
                        refresh_fixed(&fx, beta, eta_fixed);
                    }
                    continue;
                }
                return fail();
            }
        };
        layout.solve_in_place();
        // Observed-information step: the same three objects from `W_obs`, factored
        // and solved by the same two functions on their own buffers, so the Fisher
        // factor `log|A|`, the returned factor and the β border read is untouched.
        // `ss = None`: `dual` is Some only at T ≠ f64, where `TailKernel`'s default
        // bodies run whatever is passed. A non-PD twin drops the WHOLE iteration to
        // Fisher — the crossed Schur couples every cluster, so there is no
        // per-cluster fallback. Mirrors `blocked.rs` — change together.
        let mut have_obs = false;
        if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
            if structured_factor(
                g,
                &mut d.obs_core_blocks,
                &d.obs_coupling,
                &mut d.obs_schur_blk,
                coup_cols,
                coup_ptr,
                None,
            )
            .is_some()
            {
                structured_ainv_solve(
                    g,
                    &d.obs_core_blocks,
                    &d.obs_coupling,
                    &d.obs_schur_blk,
                    coup_cols,
                    coup_ptr,
                    None,
                    &mut d.obs_rhs,
                );
                have_obs = true;
            } else {
                d.exact = false;
            }
        }
        // `a_rhs[gcb..]` holds `δu_{core,f}`; `a_rhs[k_family+b]` holds `δu_e` —
        // the same packing `d.obs_rhs` uses. Add to `u` (RE-column order) and
        // accumulate `‖u‖²`, off the observed solution when the twin factor was
        // PD (`have_obs`) and off the Fisher one otherwise — identical arithmetic
        // and evaluation order either way, only the source slice differs; `log|A|`
        // and the returned factor stay Fisher regardless, as on the blocked path.
        let src: &[T] = if have_obs {
            &dual
                .as_deref()
                .expect("have_obs is set true only under Some(dual) with dual.observed")
                .obs_rhs
        } else {
            &layout.a_rhs[..]
        };
        pen = T::ZERO;
        for f in 0..s {
            let gcb = f * qc;
            for local in 0..qc {
                let val = u[core_col(f, local)] + src[gcb + local];
                u[core_col(f, local)] = val;
                pen += val * val;
            }
        }
        for b in 0..e {
            let val = u[k_family + b] + src[k_family + b];
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
        let exact_obj = ctl.exact_obj;
        if let BetaStep::Profile {
            exact: Some(ex),
            beta_prev,
            ..
        } = &mut beta_step
        {
            // `Ã = A` and `W̃ = W`: the step weight is the exact curvature on
            // every link (`observed_weights_in_place` above), so the factor just
            // taken, the leverage hᵢ and the `S⁻¹` columns below are the
            // objective's own quantities.
            let gu_dot_du = {
                let StructuredFactor {
                    core_blocks,
                    coupling,
                    schur_blk,
                    a_rhs,
                    structured_schur,
                    ..
                } = &mut layout;
                let ExactProfileBufs {
                    logdet_u,
                    logdet_beta,
                    tail_inv,
                    #[cfg(test)]
                    tail_r,
                    tail_g,
                    tail_h,
                    fac_f64,
                    curv_h,
                    curv_sdot,
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
                // S⁻¹ off the factor `structured_factor` left (cached sparse LLT,
                // or the dense L in `schur_blk`), through `TailKernel::tail_inverse`.
                // `a_rhs`'s crossed tail is the T-typed staging slot — free now,
                // its u-solve was scattered to `u` just above. Column-major:
                // `tail_inv[b·e + a] = (S⁻¹)_{a,b}`.
                T::tail_inverse(
                    schur_blk,
                    e,
                    structured_schur.as_deref_mut(),
                    &mut a_rhs[k_family..k_family + e],
                    tail_inv,
                );
                // The coupling `C_f` of the factor the leverage is read off.
                let cpl = |idx: usize| -> f64 { coupling[idx].value() };
                // The two per-cluster forms pass A's row loop reads in place of
                // a walk over cluster `f`'s coupling columns: `H_f = C_f S⁻¹`
                // into `tail_h` (`q_core × |cols_f|`, compact, see
                // `ExactProfileBufs::tail_h`) and `G_f = C_f S⁻¹ C_f'` into
                // `tail_g` (`q_core × q_core`, full). Index conventions are the
                // kernel's own: `coupling` and `tail_h` share the compact layout,
                // `coupling[q_core·coup_ptr[f] + local·|cols_f| + j] =
                // C_f[local][cols_f[j]]` and likewise `H_f` in `tail_h`, and
                // `tail_inv[b·e + a] = (S⁻¹)_{a,b}`. Both are per-ITERATION objects,
                // not per-fit: `S` is a function of `W`, so `tail_inv` above is
                // rebuilt every exact-mode iteration and these follow it.
                // Bitwise symmetry of `S⁻¹`: the per-component dense inverse
                // mirrors one triangle, the unit-solve arms do not. `e²`
                // compares against the `Σ_f |cols_f|²` build below.
                let tail_sym = (0..e).all(|a| {
                    (0..a).all(|b| tail_inv[a * e + b].to_bits() == tail_inv[b * e + a].to_bits())
                });
                for f in 0..s {
                    let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
                    if cols.is_empty() {
                        continue;
                    }
                    let (hb, nc) = (qc * coup_ptr[f] as usize, cols.len());
                    // `H_f[l][j] = Σ_ja C_f[l][ja]·(S⁻¹)_{cols[ja],cols[j]}`,
                    // `ja` outer so the `nc` outputs of a row are independent
                    // accumulators instead of one serial chain each. Every
                    // output still sums the same products from 0.0 in
                    // ascending `ja`: bit-identical to the `j`-outer dot.
                    // When `S⁻¹` is bitwise symmetric (`tail_sym`), the factor
                    // is read as `(S⁻¹)_{cols[j],a}` from column `a` instead,
                    // the same bits from one contiguous `e`-long column rather
                    // than a stride-`e` walk.
                    for l in 0..qc {
                        let crow = &coupling[hb + l * nc..hb + (l + 1) * nc];
                        let hrow = &mut tail_h[hb + l * nc..hb + (l + 1) * nc];
                        hrow.fill(0.0);
                        for (&a, c) in cols.iter().zip(crow) {
                            let (a, c) = (a as usize, c.value());
                            if tail_sym {
                                let col = &tail_inv[a * e..a * e + e];
                                for (h, &b) in hrow.iter_mut().zip(cols) {
                                    *h += c * col[b as usize];
                                }
                            } else {
                                for (h, &b) in hrow.iter_mut().zip(cols) {
                                    *h += c * tail_inv[b as usize * e + a];
                                }
                            }
                        }
                    }
                    let gb = f * qc * qc;
                    for l1 in 0..qc {
                        for l2 in 0..qc {
                            let mut acc = 0.0;
                            for j in 0..nc {
                                acc += tail_h[hb + l1 * nc + j] * cpl(hb + l2 * nc + j);
                            }
                            tail_g[gb + l1 * qc + l2] = acc;
                        }
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
                // Probit `dW_obs/dη` off a φ tile (`family::ProbitRows`).
                let mut probit_rows = if exact_obj {
                    crate::family::ProbitRows::new(family, eta, prob)
                } else {
                    None
                };
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
                    // mᵢ'A⁻¹mᵢ = ‖L_f⁻¹m_c‖² + rᵢ'S⁻¹rᵢ with rᵢ = C_f'yᵢ − tᵢ,
                    // yᵢ = A_f⁻¹m_c and tᵢ the row's own sparse crossed values.
                    // Expanded on the cluster's two precomputed forms, the tail
                    // term is yᵢ'G_f yᵢ − 2·yᵢ'(H_f tᵢ) + tᵢ'S⁻¹tᵢ, so the row
                    // pays q_core² + q_core·ncz + ncz² instead of a walk over
                    // every coupling column of cluster f.
                    //
                    // Reading H_f only on `cols_f` is safe because
                    // support(tᵢ) ⊆ cols_{f(i)}: `build_coupling_csr` builds
                    // `cols_f` by accumulating `cross_col[i·g_cap + z]`,
                    // z < n_cross[i], over EVERY row i of cluster f, from the
                    // same two arrays and the same θ-pin mask this loop reads.
                    // The invariant is load-bearing rather than cosmetic —
                    // `tail_h` stores H_f on `cols_f` only, and `S⁻¹` is dense,
                    // so H_f's entries outside `cols_f` are in general nonzero,
                    // unlike the Schur build's skipped columns, which are
                    // exactly zero.
                    let h_core = block_leverage(fac, qc, &mc[..qc]);
                    let mut h = h_core;
                    let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
                    if !cols.is_empty() {
                        yc[..qc].copy_from_slice(&mc[..qc]);
                        glmm_block_solve(fac, qc, &mut yc[..qc]);
                        let gb = f * qc * qc;
                        let hb = qc * coup_ptr[f] as usize;
                        let (t_gg, t_ght, t_tst) = row_tail_terms(
                            &yc[..qc],
                            qc,
                            &tail_g[gb..gb + qc * qc],
                            &tail_h[hb..hb + qc * cols.len()],
                            cols.len(),
                            e,
                            &cross_col[cbase..cbase + ncz],
                            &coup_pos[cbase..cbase + ncz],
                            &cross_theta[..ncz],
                            &tail_inv[..],
                        );
                        let mut acc = t_gg;
                        acc -= 2.0 * t_ght;
                        acc += t_tst;
                        h += acc;
                        #[cfg(test)]
                        {
                            let nc = cols.len();
                            for (j, &b) in cols.iter().enumerate() {
                                let b = b as usize;
                                let mut acc = 0.0;
                                for (local, &yl) in yc.iter().enumerate().take(qc) {
                                    acc += cpl(hb + local * nc + j) * yl;
                                }
                                tail_r[b] = acc;
                            }
                            for z in 0..ncz {
                                tail_r[cross_col[cbase + z] as usize] -= cross_theta[z].value();
                            }
                            let h_ref =
                                tail_quadratic_form(h_core, &tail_inv[..], e, cols, &tail_r[..]);
                            // Scale of the terms the reassociation sums, not of
                            // their sum: on a row where `C_f'yᵢ` and `tᵢ` nearly
                            // cancel, `yᵢ'G_f yᵢ` and `tᵢ'S⁻¹tᵢ` are each far
                            // larger than `rᵢ'S⁻¹rᵢ`, and the round-off is
                            // proportional to them.
                            let scale = h_core + t_gg.abs() + 2.0 * t_ght.abs() + t_tst.abs();
                            assert!(
                                (h - h_ref).abs() <= 1e-12 * scale.max(1.0),
                                "structured row leverage: reassociated {h} vs cluster walk {h_ref}"
                            );
                        }
                    }
                    // `hᵢ` is kept for the border's curvature
                    // (`logdet_beta_curvature`).
                    let wp = match probit_rows.as_mut() {
                        Some(rows) => rows.observed_weight_eta_deriv(i, y[i], prior_w[i]),
                        None => super::row_weight_eta_deriv(
                            family,
                            nb_theta,
                            exact_obj,
                            y[i],
                            prior_w[i],
                            eta[i].value(),
                            prob[i].value(),
                            w[i].value(),
                        ),
                    };
                    curv_h[i] = h;
                    let a = wp * h;
                    for j in 0..p {
                        logdet_beta[j] += a * if rows_ok {
                            xr_all[i * p + j]
                        } else {
                            x[(i, j)]
                        };
                    }
                    for local in 0..qc {
                        logdet_u[f * qc + local] += a * mc[local];
                    }
                    for z in 0..ncz {
                        let b = cross_col[cbase + z] as usize;
                        logdet_u[k_family + b] += a * cross_theta[z].value();
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
                // pass B: v = Ã⁻¹gᵤ through THIS iteration's factors. `a_rhs`
                // stages it in the generic T the solve is written for; at T = f64
                // the two loops are an identity copy of values the solve would
                // have moved anyway (the same trick the border below uses for its
                // p columns).
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
                    structured_schur.as_deref_mut(),
                    a_rhs,
                );
                for c in 0..k {
                    logdet_u[c] = a_rhs[c].value();
                }
                // pass C: û path, c_β_j −= Σᵢ w̃ᵢ·(mᵢ·v)·xᵢⱼ.
                for i in 0..n {
                    let f = cluster_ids[i] as usize;
                    let cbase = i * g_cap;
                    let mut sdot = 0.0;
                    for local in 0..qc {
                        sdot += m_core_buf[i * qc + local].value() * logdet_u[f * qc + local];
                    }
                    for z in 0..n_cross[i] as usize {
                        let b = cross_col[cbase + z] as usize;
                        sdot += cross_theta[z].value() * logdet_u[k_family + b];
                    }
                    curv_sdot[i] = sdot;
                    let a = w[i].value() * sdot;
                    for j in 0..p {
                        logdet_beta[j] -= a * if rows_ok {
                            xr_all[i * p + j]
                        } else {
                            x[(i, j)]
                        };
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
            match ctl.merit(l_trial, gu_dot_du, penalized.value()) {
                Merit::Refine => continue,
                Merit::Reject => {
                    halve_prev_to_acc(&mut u[..k], &u_prev[..k], &ex.u_acc[..k]);
                    halve_beta(beta, beta_prev);
                    refresh_fixed(&fx, beta, eta_fixed);
                    continue;
                }
                Merit::Fail => return fail(),
                Merit::Accept => {}
            }
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
        // calls per iteration, one per β column), S_β = C − B'T,
        // δβ = S_β⁻¹·(X'ρ − B'δu₀), then u_joint = u_new − T·δβ (see
        // `se::packed_schur_fill`'s doc comment for the shared β-Schur Newton step
        // this mirrors), S_β and the right-hand side formed from the design
        // residual rows (`residual_border`); exact mode as in
        // `pirls_solve_blocked`. `a_rhs` is reused
        // as the per-column A⁻¹ in/out scratch — safe now, its u-solve was
        // scattered to `u` just above (the borrow note). ---
        if let BetaStep::Profile {
            exact,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            schur_llt_mem,
            ..
        } = &mut beta_step
        {
            let StructuredFactor {
                core_blocks,
                coupling,
                schur_blk,
                a_rhs,
                structured_schur,
                ..
            } = &mut layout;
            if rows_ok {
                // `B'` (same terms, same row order as the scatter below), off
                // the row-major `X` (`super::blocked::structured_border_rows`).
                super::blocked::structured_border_rows(
                    &w[..],
                    &m_core_buf[..],
                    &cross_theta[..],
                    &cross_col[..],
                    &n_cross[..],
                    cluster_ids,
                    xr_all,
                    q,
                    np,
                    s,
                    e,
                    p,
                    n,
                    &mut bt_rows[..k * p],
                );
                for c in 0..k {
                    let src = &bt_rows[c * p..c * p + p];
                    for r in 0..p {
                        xtwm[(r, c)] = src[r];
                    }
                }
            } else {
                // B' = X'WM (p×k): zero, then scatter each row's core + crossed columns
                // from the packed `m_core_buf`/`cross_*` nonzeros. Mirrors
                // `se::structured_schur_fill`'s `X'W̃M` scatter; change together.
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
                            xtwm[(r, core_col(f, local))] +=
                                xw * m_core_buf[i * qc + local].value();
                        }
                        for z in 0..ncz {
                            let b = cross_col[cbase + z] as usize;
                            xtwm[(r, k_family + b)] += xw * cross_theta[z].value();
                        }
                    }
                }
            }
            // T = A⁻¹B: ainv_mtwx[:, c] = A⁻¹(M'WX)[:, c], one β column at a time via
            // `structured_ainv_solve` reusing THIS iteration's core-block+Schur factors.
            // Mirrors `se::structured_schur_fill`'s `A⁻¹(M'W̃X)` solve; change
            // together. `a_rhs` packs/unpacks the RHS/solution in
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
                    structured_schur.as_deref_mut(),
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
            // Residual rows rᵢ = xᵢ − Tᵀmᵢ into `wx`, over the row's core and
            // crossed nonzeros (T in the u RE-column|crossed layout), then S_β
            // and the right-hand side off them.
            for i in 0..n {
                let f = cluster_ids[i] as usize;
                let cbase = i * g_cap;
                let ncz = n_cross[i] as usize;
                for col in 0..p {
                    let mut r = x[(i, col)];
                    for local in 0..qc {
                        r -= m_core_buf[i * qc + local].value()
                            * ainv_mtwx[(core_col(f, local), col)];
                    }
                    for z in 0..ncz {
                        r -= cross_theta[z].value()
                            * ainv_mtwx[(k_family + cross_col[cbase + z] as usize, col)];
                    }
                    wx[(i, col)] = r;
                }
            }
            let scores = super::ProfileScores::new(family, nb_theta, y, prior_w, eta, prob);
            residual_border(
                |i| w[i].value(),
                |i| scores.at(i),
                wx.as_ref().subrows(0, n),
                ainv_mtwx.as_ref().subrows(0, k),
                |j| u_prev[j].value(),
                schur,
                &mut beta_rhs[..p],
            );
            // Exact mode: log|A|'s gradient and curvature in β, as in
            // `pirls_solve_blocked` (change together).
            if let Some(ex) = exact.as_deref_mut() {
                for r in 0..p {
                    beta_rhs[r] -= 0.5 * ex.logdet_beta[r];
                }
                if ctl.wants_curvature(it, &mut ex.curv_memory) {
                    logdet_beta_curvature(
                        ex,
                        x,
                        ainv_mtwx.as_ref(),
                        &m_core_buf[..],
                        cluster_ids,
                        g,
                        Some(&CrossedTail {
                            cross_col: &cross_col[..],
                            cross_theta: &cross_theta[..],
                            n_cross: &n_cross[..],
                            e,
                        }),
                        family,
                        nb_theta,
                        exact_obj,
                        y,
                        prior_w,
                        &eta[..],
                        &prob[..],
                        n,
                        p,
                    );
                } else {
                    ex.logdet_hess.fill(0.0);
                }
            }
            // δβ in place, inside the trust region on the exact border. Non-PD
            // S_β ⇒ the (NaN,…,false) failure surface.
            if !border_solve(
                schur,
                &mut beta_rhs[..p],
                schur_llt_mem,
                exact.as_deref_mut().map(|ex| (ex, &mut ctl.trust)),
            ) {
                return fail();
            }
            apply_border(&fx, ainv_mtwx, beta_rhs, u, beta, eta_fixed, k);
            // `pen` must track the moved u (‖u_joint‖²), so recompute it.
            pen = pen_sq(&u[..k]);
        }
        if ctl.damp {
            ctl.trust.scale(0.5);
            damp_step(&fx, &beta_step, u, u_prev, beta, eta_fixed, k);
            pen = pen_sq(&u[..k]);
        }
        if ctl.stop(it, (dev + pen).value(), logdet.value()) {
            converged = true;
            break;
        }
    }
    // The returned `dev`, `log|A|` and core/Schur factors at the returned
    // iterate, so the objective's three terms and the factors
    // `structured_schur_fill` inherits describe one point — see
    // [`evaluate_at_mode`].
    if converged {
        match evaluate_at_mode(
            &mut layout,
            family,
            nb_theta,
            y,
            prior_w,
            weighted,
            &eta_fixed[..],
            &u[..],
            &mut eta[..],
            &mut prob[..],
            &mut w[..],
            n,
        ) {
            Some((d, ld)) => {
                dev = d;
                logdet = ld;
            }
            None => return fail(),
        }
    }
    (dev, pen, logdet, converged)
}

// ===================== Streamed SE pass: streamed structured dual solve =====================

/// Streamed SE pass: block-sized per-row dual buffers for the streamed
/// structured SE pass. Every per-row value (`η`, `p`, `w`, `ρ`, the core `M`
/// slice) is formed for one contiguous row block, consumed, and dropped; the
/// cluster-level and pair-level accumulators stay whole.
pub(crate) struct StreamRows<T> {
    pub(crate) block: usize,
    pub(crate) eta: Vec<T>,
    pub(crate) prob: Vec<T>,
    pub(crate) w: Vec<T>,
    pub(crate) mu: Vec<T>,
    pub(crate) mc: Vec<T>,
    pub(crate) lam: Vec<T>,
}

impl<T: Scalar> StreamRows<T> {
    pub(crate) fn new(block: usize, qc: usize, q: usize) -> Self {
        Self {
            block,
            eta: vec![T::ZERO; block],
            prob: vec![T::ZERO; block],
            w: vec![T::ZERO; block],
            mu: vec![T::ZERO; block],
            mc: vec![T::ZERO; block * qc],
            lam: vec![T::ZERO; q * q],
        }
    }
}

/// `build_packed_m`'s core slice for one row: both fold the primary block
/// through `zlam_row`, and the nested arm below mirrors `build_packed_m`'s —
/// change together.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn m_core_row<T: Scalar>(
    g: &crate::lmm::LmmGroupings,
    lam: &[T],
    theta_nested: T,
    nested_decl: Option<usize>,
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    f: usize,
    i: usize,
    out: &mut [T],
) {
    let q = g.primary_q;
    let np = g.nested_per_parent;
    super::zlam_row(&z_buf[i * (q - 1)..], lam, q, out);
    if np > 0 {
        let global_id = extra_ids[nested_decl.expect("np > 0")][i] as usize;
        let local_id = global_id - f * np;
        for j in 0..np {
            out[q + j] = if j == local_id {
                theta_nested
            } else {
                T::ZERO * theta_nested
            };
        }
    }
}

/// Rows `[i0, i1)` at the iterate `u`: `η` ([`stream_eta`]), the family
/// pass, and on `observed_conv` the `evaluate_at_mode` conversion of `w` to
/// `W_obs`. Returns the block's deviance and the infeasible flag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stream_fill<T: TailKernel>(
    sr: &mut StreamRows<T>,
    g: &crate::lmm::LmmGroupings,
    theta_nested: T,
    nested_decl: Option<usize>,
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    cluster_ids: &[u32],
    cross_theta: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    x: MatRef<f64>,
    beta: &[T],
    offset: Option<&[f64]>,
    u: &[T],
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    i0: usize,
    i1: usize,
    observed_conv: bool,
) -> (T, bool) {
    let nb = i1 - i0;
    let StreamRows {
        eta,
        prob,
        w,
        mc,
        lam,
        ..
    } = sr;
    let yeta = stream_eta(
        mc,
        lam,
        eta,
        g,
        theta_nested,
        nested_decl,
        z_buf,
        extra_ids,
        cluster_ids,
        cross_theta,
        cross_col,
        n_cross,
        x,
        beta,
        offset,
        u,
        y,
        i0,
        i1,
    );
    let (d, infeasible) = T::family_pass(
        family,
        nb_theta,
        &mut eta[..nb],
        &y[i0..i1],
        &prior_w[i0..i1],
        weighted,
        yeta,
        &mut prob[..nb],
        &mut w[..nb],
        &mut [],
    );
    if observed_conv {
        observed_weights_in_place(
            family,
            nb_theta,
            &y[i0..i1],
            &prior_w[i0..i1],
            &eta[..nb],
            &prob[..nb],
            &mut w[..nb],
            nb,
        );
    }
    (d, infeasible)
}

/// The raw `η` of rows `[i0, i1)` at the iterate `u` into `eta` (the
/// `refresh_eta_fixed` + `eta_from_mode` arithmetic), with each row's core
/// `M` slice into `mc`. Returns `Σ yᵢηᵢ` over the block.
#[allow(clippy::too_many_arguments)]
fn stream_eta<T: Scalar>(
    mc: &mut [T],
    lam: &[T],
    eta: &mut [T],
    g: &crate::lmm::LmmGroupings,
    theta_nested: T,
    nested_decl: Option<usize>,
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    cluster_ids: &[u32],
    cross_theta: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    x: MatRef<f64>,
    beta: &[T],
    offset: Option<&[f64]>,
    u: &[T],
    y: &[f64],
    i0: usize,
    i1: usize,
) -> T {
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let q = g.primary_q;
    let np = g.nested_per_parent;
    let qc = q + np;
    let s = g.n_primary;
    let prim_width = q * s;
    let k_family = qc * s;
    let p = beta.len();
    let nb = i1 - i0;
    // The fixed part first, into `eta`, `j` outer as in `refresh_eta_fixed`.
    eta[..nb].fill(T::ZERO);
    for j in 0..p {
        for t in 0..nb {
            eta[t] += beta[j].scale(x[(i0 + t, j)]);
        }
    }
    let mut yeta = T::ZERO;
    for t in 0..nb {
        let i = i0 + t;
        let f = cluster_ids[i] as usize;
        let mut ef = eta[t];
        if let Some(o) = offset {
            ef += T::from_f64(o[i]);
        }
        let mrow = &mut mc[t * qc..t * qc + qc];
        m_core_row(
            g,
            lam,
            theta_nested,
            nested_decl,
            z_buf,
            extra_ids,
            f,
            i,
            mrow,
        );
        let mut mui = T::ZERO;
        #[allow(clippy::needless_range_loop)]
        for local in 0..qc {
            let col = if local < q {
                f * q + local
            } else {
                prim_width + f * np + (local - q)
            };
            mui += mrow[local] * u[col];
        }
        let cbase = i * g_cap;
        for z in 0..n_cross[i] as usize {
            let b = cross_col[cbase + z] as usize;
            mui += cross_theta[z] * u[k_family + b];
        }
        eta[t] = ef + mui;
        yeta += eta[t].scale(y[i]);
    }
    yeta
}

/// `StructuredFactor::scatter`'s accumulator clear, split off so the streamed
/// sweep clears once and scatters block by block.
#[allow(clippy::too_many_arguments)]
fn stream_zero<T: Scalar>(
    qc: usize,
    s: usize,
    e: usize,
    k: usize,
    coup_ptr: &[u32],
    core_blocks: &mut [T],
    coupling: &mut [T],
    schur_blk: &mut [T],
    a_rhs: &mut [T],
    dual: Option<&mut DualStep<T>>,
) {
    for v in core_blocks[..s * qc * qc].iter_mut() {
        *v = T::ZERO;
    }
    let coup_end = qc * coup_ptr[s] as usize;
    for v in coupling[..coup_end].iter_mut() {
        *v = T::ZERO;
    }
    for v in schur_blk[..e * e].iter_mut() {
        *v = T::ZERO;
    }
    for v in a_rhs[..k].iter_mut() {
        *v = T::ZERO;
    }
    if let Some(d) = dual.filter(|d| d.observed) {
        for v in d.obs_core_blocks[..s * qc * qc].iter_mut() {
            *v = T::ZERO;
        }
        for v in d.obs_coupling[..coup_end].iter_mut() {
            *v = T::ZERO;
        }
        for v in d.obs_schur_blk[..e * e].iter_mut() {
            *v = T::ZERO;
        }
    }
}

/// `StructuredFactor::scatter`'s row loop for rows `[i0, i1)`, reading the
/// block's own per-row values; the same arithmetic in the same row order.
#[allow(clippy::too_many_arguments)]
fn stream_scatter<T: TailKernel>(
    sr: &mut StreamRows<T>,
    g: &crate::lmm::LmmGroupings,
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    cluster_ids: &[u32],
    cross_theta: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    coup_ptr: &[u32],
    coup_pos: &[u32],
    core_blocks: &mut [T],
    coupling: &mut [T],
    schur_blk: &mut [T],
    a_rhs: &mut [T],
    mut dual: Option<&mut DualStep<T>>,
    i0: usize,
    i1: usize,
) {
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let qc = g.primary_q + g.nested_per_parent;
    let s = g.n_primary;
    let e = g.k_crossed();
    let k_family = qc * s;
    let nb = i1 - i0;
    let StreamRows {
        eta,
        prob,
        w,
        mu,
        mc,
        ..
    } = sr;
    let mut probit_rows = crate::family::ProbitRows::new(family, &eta[..nb], &prob[..nb]);
    for t in 0..nb {
        let i = i0 + t;
        mu[t] = super::row_rho(
            family,
            nb_theta,
            t,
            y[i],
            prior_w[i],
            eta[t],
            prob[t],
            probit_rows.as_mut(),
        );
    }
    for t in 0..nb {
        let i = i0 + t;
        let f = cluster_ids[i] as usize;
        let wi = w[t];
        let ri = mu[t];
        let m_core = &mc[t * qc..t * qc + qc];
        let cbase = i * g_cap;
        let ncz = n_cross[i] as usize;
        let cb = f * qc * qc;
        let gcb = f * qc;
        let coup = qc * coup_ptr[f] as usize;
        let nc = (coup_ptr[f + 1] - coup_ptr[f]) as usize;
        for r in 0..qc {
            let mr = m_core[r];
            a_rhs[gcb + r] += mr * ri;
            let wmr = wi * mr;
            for c in 0..=r {
                core_blocks[cb + r * qc + c] += wmr * m_core[c];
            }
            for z in 0..ncz {
                coupling[coup + r * nc + coup_pos[cbase + z] as usize] += wmr * cross_theta[z];
            }
        }
        for z in 0..ncz {
            let b = cross_col[cbase + z] as usize;
            let vb = cross_theta[z];
            a_rhs[k_family + b] += vb * ri;
            let wvb = wi * vb;
            for z2 in 0..ncz {
                let b2 = cross_col[cbase + z2] as usize;
                if b2 <= b {
                    schur_blk[b * e + b2] += wvb * cross_theta[z2];
                }
            }
        }
        if let Some(d) = dual.as_deref_mut().filter(|d| d.observed) {
            let wo = crate::family::observed_weight(
                family, nb_theta, y[i], prior_w[i], eta[t], prob[t], wi,
            );
            for r in 0..qc {
                let mr = m_core[r];
                let wmr = wo * mr;
                #[allow(clippy::needless_range_loop)]
                for c in 0..=r {
                    d.obs_core_blocks[cb + r * qc + c] += wmr * m_core[c];
                }
                for z in 0..ncz {
                    d.obs_coupling[coup + r * nc + coup_pos[cbase + z] as usize] +=
                        wmr * cross_theta[z];
                }
            }
            for z in 0..ncz {
                let b = cross_col[cbase + z] as usize;
                let vb = cross_theta[z];
                let wvb = wo * vb;
                for z2 in 0..ncz {
                    let b2 = cross_col[cbase + z2] as usize;
                    if b2 <= b {
                        d.obs_schur_blk[b * e + b2] += wvb * cross_theta[z2];
                    }
                }
            }
        }
    }
}

/// Streamed SE pass: [`pirls_solve_blocked_extras`] at `BetaStep::Fixed` with a
/// dual step (the assembled-Hessian SE call), with every per-row value
/// streamed in row blocks of `sr.block` rows: one fused sweep per iteration
/// forms `η`, the family pass and the scatter block by block. The trial's
/// accept test runs after the sweep (the scatter of a rejected trial is
/// discarded and redone on the halved `u`). Returns
/// `(objective, dev, converged)`: the objective as `structured_laplace_deviance`
/// returns it, and the raw PIRLS deviance `d(y,ũ)` alone.
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_stream_deviance<T: TailKernel>(
    family: Family,
    nb_theta: f64,
    g: &crate::lmm::LmmGroupings,
    params: &[T],
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    cluster_ids: &[u32],
    scratch: &mut PirlsScratch<T>,
    structured: &mut StructuredScratch<T>,
    pattern: &mut StructuredPattern,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    beta: &[T],
    dual: &mut DualStep<T>,
    offset: Option<&[f64]>,
    tol: f64,
    n: usize,
    sr: &mut StreamRows<T>,
) -> (T, T, bool) {
    let nan = || T::from_f64(f64::NAN);
    crate::glmm::workspace::build_packed_m(
        g,
        params,
        z_buf,
        extra_ids,
        &mut scratch.lam,
        cluster_ids,
        &mut structured.m_core_buf,
        &mut structured.cross_theta,
        &mut pattern.cross_col,
        &mut pattern.n_cross,
        n,
    );
    // A dual call keys the CSR cache on the full pattern (mask 0), as
    // `structured_laplace_deviance` does.
    if pattern.coup_mask != Some(0) {
        build_coupling_csr(
            cluster_ids,
            &pattern.cross_col,
            &pattern.n_cross,
            g.n_primary,
            n,
            &mut pattern.coup_cols,
            &mut pattern.coup_ptr,
            &mut pattern.coup_pos,
        );
        pattern.coup_mask = Some(0);
    }
    let q = g.primary_q;
    sr.lam[..q * q].copy_from_slice(&scratch.lam[..q * q]);
    let base_theta = q * (q + 1) / 2;
    let theta_nested = g.nested.map(|nf| params[nf.vech_start]).unwrap_or(T::ZERO);
    let nested_decl = g.nested.map(|nf| nf.vech_start - base_theta);

    let PirlsScratch {
        u, u_prev, a_rhs, ..
    } = scratch;
    let StructuredScratch {
        core_blocks,
        coupling,
        schur_blk,
        cross_theta,
        ..
    } = structured;
    let force_dense = pattern.force_dense_schur();
    let StructuredPattern {
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_pos,
        structured_schur,
        ..
    } = pattern;
    let np = g.nested_per_parent;
    let qc = q + np;
    let s = g.n_primary;
    let prim_width = q * s;
    let k_family = qc * s;
    let e = g.k_crossed();
    let k = k_family + e;
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };
    let block = sr.block;
    // One fused sweep at the current `u`: clear, then per block fill + scatter.
    // `observed_conv` + `twin` select the evaluate-at-mode variant (W_obs, no
    // twin) or the in-loop one (Fisher W, twin scattered).
    macro_rules! sweep {
        ($observed_conv:expr, $twin:expr) => {{
            stream_zero(
                qc,
                s,
                e,
                k,
                &coup_ptr[..],
                &mut core_blocks[..],
                &mut coupling[..],
                &mut schur_blk[..],
                &mut a_rhs[..],
                if $twin { Some(&mut *dual) } else { None },
            );
            let mut dev = T::ZERO;
            let mut infeasible = false;
            let mut i0 = 0usize;
            while i0 < n {
                let i1 = (i0 + block).min(n);
                let (d, inf) = stream_fill(
                    sr,
                    g,
                    theta_nested,
                    nested_decl,
                    z_buf,
                    extra_ids,
                    cluster_ids,
                    &cross_theta[..],
                    &cross_col[..],
                    &n_cross[..],
                    x,
                    beta,
                    offset,
                    &u[..],
                    family,
                    nb_theta,
                    y,
                    prior_w,
                    weighted,
                    i0,
                    i1,
                    $observed_conv,
                );
                dev += d;
                infeasible |= inf;
                stream_scatter(
                    sr,
                    g,
                    family,
                    nb_theta,
                    y,
                    prior_w,
                    cluster_ids,
                    &cross_theta[..],
                    &cross_col[..],
                    &n_cross[..],
                    &coup_ptr[..],
                    &coup_pos[..],
                    &mut core_blocks[..],
                    &mut coupling[..],
                    &mut schur_blk[..],
                    &mut a_rhs[..],
                    if $twin { Some(&mut *dual) } else { None },
                    i0,
                    i1,
                );
                i0 = i1;
            }
            (dev, infeasible)
        }};
    }
    macro_rules! layout {
        () => {
            StructuredFactor {
                core_blocks: &mut core_blocks[..],
                coupling: &mut coupling[..],
                schur_blk: &mut schur_blk[..],
                a_rhs: &mut a_rhs[..],
                mu: &mut [],
                m_core_buf: &[],
                cross_theta: &cross_theta[..],
                cross_col: &cross_col[..],
                n_cross: &n_cross[..],
                coup_cols: &coup_cols[..],
                coup_ptr: &coup_ptr[..],
                coup_pos: &coup_pos[..],
                structured_schur: structured_schur.as_mut().filter(|_| !force_dense),
                g,
                cluster_ids,
                family,
                nb_theta,
            }
        };
    }

    u_prev[..k].fill(T::ZERO);
    let mut pen_accepted = f64::INFINITY;
    let mut mixed_prev = f64::INFINITY;
    let mut halvings = 0usize;
    let mut converged = false;
    dual.exact = true;
    let min_iters = dual.min_iters;
    let mut pen = nan();
    for it in 0..PIRLS_MAX_ITERS {
        let (dev, infeasible) = sweep!(false, true);
        let mut pen_u = T::ZERO;
        #[allow(clippy::needless_range_loop)]
        for c in 0..k {
            pen_u += u[c] * u[c];
        }
        let penalized = dev + pen_u;
        if infeasible
            || !penalized.value().is_finite()
            || penalized.value() - pen_accepted > tol * (1.0 + penalized.value().abs())
        {
            if halvings < PIRLS_MAX_HALVINGS {
                halvings += 1;
                if infeasible {
                    // Past the η clamp: one jump back inside it, as in
                    // `jump_back` (change together). The
                    // smallest fraction over the row blocks is the one over
                    // all rows.
                    let mut s_min = f64::INFINITY;
                    let mut i0 = 0usize;
                    while i0 < n {
                        let i1 = (i0 + block).min(n);
                        let StreamRows {
                            eta, prob, mc, lam, ..
                        } = &mut *sr;
                        for (at, raw) in [(&u[..], &mut eta[..]), (&u_prev[..], &mut prob[..])] {
                            stream_eta(
                                mc,
                                lam,
                                raw,
                                g,
                                theta_nested,
                                nested_decl,
                                z_buf,
                                extra_ids,
                                cluster_ids,
                                &cross_theta[..],
                                &cross_col[..],
                                &n_cross[..],
                                x,
                                beta,
                                offset,
                                at,
                                y,
                                i0,
                                i1,
                            );
                        }
                        let nb = i1 - i0;
                        s_min =
                            s_min.min(clamp_fraction(family, &y[i0..i1], &prob[..nb], &eta[..nb]));
                        i0 = i1;
                    }
                    let f = T::from_f64(clamp_jump(s_min));
                    for c in 0..k {
                        u[c] = u_prev[c] + f * (u[c] - u_prev[c]);
                    }
                    continue;
                }
                for c in 0..k {
                    u[c] = T::from_f64(0.5) * (u[c] + u_prev[c]);
                }
                continue;
            }
            return (nan(), nan(), false);
        }
        u_prev[..k].copy_from_slice(&u[..k]);
        halvings = 0;
        pen_accepted = penalized.value();
        for f in 0..s {
            for local in 0..qc {
                a_rhs[f * qc + local] -= u[core_col(f, local)];
            }
        }
        for b in 0..e {
            a_rhs[k_family + b] -= u[k_family + b];
        }
        if dual.observed {
            dual.obs_rhs[..k].copy_from_slice(&a_rhs[..k]);
        }
        {
            let mut layout = layout!();
            if layout.factor_logdet().is_none() {
                return (nan(), nan(), false);
            }
            layout.solve_in_place();
        }
        let mut have_obs = false;
        if dual.observed {
            if structured_factor(
                g,
                &mut dual.obs_core_blocks,
                &dual.obs_coupling,
                &mut dual.obs_schur_blk,
                coup_cols,
                coup_ptr,
                None,
            )
            .is_some()
            {
                structured_ainv_solve(
                    g,
                    &dual.obs_core_blocks,
                    &dual.obs_coupling,
                    &dual.obs_schur_blk,
                    coup_cols,
                    coup_ptr,
                    None,
                    &mut dual.obs_rhs,
                );
                have_obs = true;
            } else {
                dual.exact = false;
            }
        }
        let src: &[T] = if have_obs { &dual.obs_rhs } else { &a_rhs[..] };
        pen = T::ZERO;
        for f in 0..s {
            let gcb = f * qc;
            for local in 0..qc {
                let val = u[core_col(f, local)] + src[gcb + local];
                u[core_col(f, local)] = val;
                pen += val * val;
            }
        }
        for b in 0..e {
            let val = u[k_family + b] + src[k_family + b];
            u[k_family + b] = val;
            pen += val * val;
        }
        let mixed = (dev + pen).value();
        if it + 1 >= min_iters && (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
            converged = true;
            break;
        }
        mixed_prev = mixed;
    }
    if !converged {
        return (nan(), nan(), false);
    }
    // `evaluate_at_mode`, streamed: W_obs, no twin, then the factor.
    let (dev, infeasible) = sweep!(true, false);
    if infeasible {
        return (nan(), nan(), false);
    }
    let logdet = {
        let mut layout = layout!();
        match layout.factor_logdet() {
            Some(ld) => ld,
            None => return (nan(), nan(), false),
        }
    };
    if !dev.value().is_finite() {
        return (T::from_f64(f64::INFINITY), dev, true);
    }
    (dev + pen + T::from_f64(2.0) * logdet, dev, true)
}

#[cfg(test)]
#[path = "blocked_extras_tests.rs"]
mod tests;
