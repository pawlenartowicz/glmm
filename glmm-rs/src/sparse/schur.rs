//! The sparse-Z LMM path's per-θ blocked Cholesky and augmented Schur factor
//! ([`sparse_schur_factor`]), split along its phases. Every phase adds its
//! pivots straight into the one running `log_lzz_half`, in the order the
//! single-function form summed them.
use super::{
    fill_lambda_small, fold_packed_col, logdet_llt, FamDowndate, LamBlock, SparseLmmWorkspace,
    SparseRecovery, SparseTail,
};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, LltRegularization};
use faer::mat::AsMatMut;
use faer::sparse::SparseColMat;
use faer::{Conj, Mat, MatRef, Par, Side, Spec};

/// The blocked kernel's dimensions, fixed for the fit (see `sparse_schur_factor`).
#[derive(Clone, Copy)]
struct SchurDims {
    m: usize,
    w: usize,
    n_prim: usize,
    q_p: usize,
    np: usize,
    q_n: usize,
    kf: usize,
    e: usize,
    fam_len: usize,
    cb0: usize,
}

/// The buffers every phase only reads: the θ-independent Grams and packed
/// streams of `SparseLmmWorkspace`, plus this θ's small Λ factors.
struct SchurInputs<'a> {
    ztxy: &'a [f64],
    cxy: &'a Mat<f64>,
    pk_fam: &'a [f64],
    pk_a21: &'a [f64],
    pk_a22: &'a [f64],
    a21_blk: &'a [u32],
    a21_off: &'a [usize],
    pk_a21_off: &'a [usize],
    a22_pairs: &'a [[u32; 2]],
    lam_blocks: &'a [LamBlock],
    lam_small: &'a [f64],
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
///   S = C_xy − UᵀU, Cholesky (`cholesky_in_place`) → the augmented `m×m`
///         factor `L` in `ws.factor`
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
/// the same Cholesky as a generic sparse factorization, up to the
/// elimination order (structure-driven here vs AMD) and GEMM accumulation
/// order — a sanctioned reassociation (same argument as `reml_deviance`,
/// `src/lmm/kernel.rs`),
/// gated by the both-paths deviance-equality tests and the validation suite.
///
/// Returns `log|L_ZZ|²`; `None` on any non-PD pivot (family Crout, tail LLT —
/// `LltRegularization::default()` is a verified no-op, delta = 0 — or Schur
/// Cholesky), the same INFINITY surface as the dense evaluators (`src/lmm/kernel.rs`).
/// Shared by `sparse_reml_deviance` (diagonal → deviance) and `lmm::fit_lmm`
/// (the full factor → β̂/σ̂²/SE recovery). By Cholesky uniqueness `ws.factor`
/// equals the dense path's augmented `fit.factor` at the same θ (both factor
/// the identical reduced augmented Gram X'V⁻¹[X y]): `L[(j,j)]` (j<p) is the
/// L_XX diagonal, `L[(p,j)]` the y-row `l_yX[j]`, and `L[(p,p)]²/(N−p)` the
/// profiled σ̂² — the recovery reads them exactly as `fit_lmm` (`src/lmm/mod.rs`)
/// reads `fit.factor`.
pub(super) fn sparse_schur_factor(theta: &[f64], ws: &mut SparseLmmWorkspace) -> Option<f64> {
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
    let dims = SchurDims {
        m,
        w,
        n_prim,
        q_p,
        np,
        q_n,
        kf,
        e,
        fam_len,
        cb0,
    };
    let inp = SchurInputs {
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
    };

    if e > 0 {
        fill_tail_b2(dims, &inp, u2);
        // Sparse branch: assemble A22 + I into the CSC values now; the family
        // loop scatters each −L21_f·L21_fᵀ downdate on top. Same fold as the
        // dense arm's, sunk through the precomputed slot replay (`a22_slots`,
        // sequential — mirrors the generation in `build_sparse_tail`, change
        // together); every target exists by construction of the
        // diagonal-seeded pattern, verified loudly at setup.
        if let Some(tail) = tail.as_mut() {
            assemble_tail_a22(dims, &inp, tail);
        }
    }

    let mut log_lzz_half = 0.0_f64;
    for f in 0..n_prim {
        factor_family(dims, &inp, f, fam_a, &mut log_lzz_half)?;
        // The factor this iteration is about to overwrite for the next family —
        // kept only when the post-fit recovery asked for it (see `SparseRecovery`).
        if let Some(r) = rec.as_mut() {
            r.fam_l[f * w * w..(f + 1) * w * w].copy_from_slice(&fam_a[..w * w]);
        }
        let fb = f * w;
        family_u1(dims, &inp, f, fb, fam_a, u1);
        // L21 family block, two phases. Phase A: raw A21 for ALL w family
        // columns in ONE pass over the family's co-occurring blocks (each
        // block's merged slab carries the pb columns then each child's — one
        // block-loop instead of w; per component row a, the scalar
        // h = G_row·λ_c stays in a register and fans out to a's lower Λ_x
        // column). Non-co-occurring blocks stay at the zero-fill; each raw
        // entry belongs to exactly one block and keeps its (a, b) accumulation
        // order, so this equals the per-column form bit for bit.
        match tail.as_mut() {
            None => {
                l21[fb * e..(fb + w) * e].fill(0.0);
                fold_a21(dims, &inp, f, &mut l21[..], e, fb, false);
                // Phase B tri-solve on the family's w columns of the global l21.
                schur_phase_b(&mut l21[fb * e..(fb + w) * e], e, w, fam_a);
            }
            Some(tail) => family_sparse_tail(dims, &inp, f, fb, fam_a, u1, u2, tail, rec),
        }
    }

    if e > 0 {
        match tail.as_mut() {
            None => dense_tail_factor(
                dims,
                &inp,
                l21,
                u1,
                s22,
                u2,
                tail_llt_mem,
                &mut log_lzz_half,
            )?,
            Some(tail) => sparse_tail_factor(dims, tail, u2, &mut log_lzz_half)?,
        }
    }

    augmented_factor(dims, &inp, u1, u2, tail, factor, factor_llt_mem)?;
    Some(2.0 * log_lzz_half)
}

/// The B2 tail rows of `U2`, one crossed block at a time.
fn fill_tail_b2(dims: SchurDims, inp: &SchurInputs, u2: &mut [f64]) {
    let SchurDims { m, kf, e, cb0, .. } = dims;
    let SchurInputs {
        ztxy,
        lam_blocks,
        lam_small,
        ..
    } = *inp;
    // B2 tail rows: fill per crossed block (stride 1), same Λ'-fold as the
    // U1 rows below. Runs BEFORE the family loop so the sparse branch can
    // downdate `B̃₂ = B2 − L21·U1` per family off its compact panel; the
    // dense branch downdates after the loop with one GEMM (the
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
}

/// `A22 + I` into the sparse tail's CSC values, before the family loop.
fn assemble_tail_a22(dims: SchurDims, inp: &SchurInputs, tail: &mut SparseTail) {
    let SchurDims { kf, .. } = dims;
    let SchurInputs {
        pk_a22,
        a22_pairs,
        lam_blocks,
        lam_small,
        ..
    } = *inp;
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

/// Family `f`'s block `A_f` and its Crout, in place in `fam_a`; its pivots go
/// into `log_lzz_half`. `None` on a non-positive pivot.
fn factor_family(
    dims: SchurDims,
    inp: &SchurInputs,
    f: usize,
    fam_a: &mut [f64],
    log_lzz_half: &mut f64,
) -> Option<()> {
    let SchurDims {
        n_prim,
        q_p,
        np,
        q_n,
        w,
        fam_len,
        ..
    } = dims;
    let SchurInputs {
        pk_fam,
        lam_blocks,
        lam_small,
        ..
    } = *inp;
    // A_f = Λ_f'G_fΛ_f + I (row-major lower), assembled block-pair-wise
    // off the packed family stream — (primary, primary), each
    // (child, primary) coupling, each child's own diagonal block; sibling
    // child–child blocks are structurally zero (children never share rows
    // — mirror `src/lmm/kernel.rs`) and zero-filled. Then Crout in place → L_f
    // (`linalg::block_chol`).
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
    if !crate::linalg::block_chol(&mut fam_a[..w * w], w) {
        return None;
    }
    for j in 0..w {
        *log_lzz_half += fam_a[j * w + j].ln();
    }
    Some(())
}

/// Family `f`'s `U1` rows and their forward solve by `L_f`.
fn family_u1(
    dims: SchurDims,
    inp: &SchurInputs,
    f: usize,
    fb: usize,
    fam_a: &[f64],
    u1: &mut [f64],
) {
    let SchurDims {
        m,
        w,
        n_prim,
        q_p,
        np,
        q_n,
        kf,
        ..
    } = dims;
    let SchurInputs {
        ztxy,
        lam_blocks,
        lam_small,
        ..
    } = *inp;
    let pb = &lam_blocks[f];
    // U1 family rows: B1_f = (Λ'Z'[X y])[family rows] — row r is
    // Σ_{a≥il} Λ[a,il]·ztxy_row(col_a) over its owning block (the
    // P_zx = ΛᵀZᵀ[Xy] block), read off the row-major ztxy — then forward-solved by L_f (col-major
    // kf-stride Vec; the row ops span m ≤ p+1 columns).
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
}

/// Phase A A21-fold, shared by both tail arms. For every co-occurring
/// block the h = G_row·λ_c scalar stays in a register and fans out to the
/// block's lower Λ_x column; the h-accumulation and block-column dispatch
/// are identical across arms — only the destination index differs. Dense
/// arm: scatter into the global e-row `l21` at column `fb+c0` (stride `e`),
/// block rows at `t0 = start-kf`. Panel arm: pack into the compact
/// `e_f`-row panel at column `c0` (stride `e_f`), block rows at the running
/// `loc0`. `panel_layout` picks the row base once per block (not per entry).
/// Non-co-occurring blocks stay at the caller's zero-fill; each raw entry
/// belongs to one block and keeps its (a, b) accumulation order, so this
/// equals the per-column form bit for bit.
fn fold_a21(
    dims: SchurDims,
    inp: &SchurInputs,
    f: usize,
    out: &mut [f64],
    stride: usize,
    col_off: usize,
    panel_layout: bool,
) {
    let SchurDims {
        w,
        n_prim,
        q_p,
        np,
        q_n,
        kf,
        ..
    } = dims;
    let SchurInputs {
        pk_a21,
        a21_blk,
        a21_off,
        pk_a21_off,
        lam_blocks,
        lam_small,
        ..
    } = *inp;
    let lo_p = lam_blocks[f].lam_off;
    let fam_blks = &a21_blk[a21_off[f]..a21_off[f + 1]];
    let a21_fam = &pk_a21[pk_a21_off[f]..pk_a21_off[f + 1]];
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
}

/// Schur L21 Phase B: tri-solve each of the `w` family columns against `L_fᵀ`
/// (stored col-major in `fam_a`, `w×w`) as contiguous-column axpys — columns
/// before `fb+c` are final when read (the `src/lmm/kernel.rs` `fit.bt` split_at_mut
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

/// Family `f`'s L21 block on the sparse-tail branch: its compact panel, then
/// the `S22` and `B̃₂` downdates it feeds within this iteration.
#[allow(clippy::too_many_arguments)]
fn family_sparse_tail(
    dims: SchurDims,
    inp: &SchurInputs,
    f: usize,
    fb: usize,
    fam_a: &[f64],
    u1: &[f64],
    u2: &mut [f64],
    tail: &mut SparseTail,
    rec: &mut Option<SparseRecovery>,
) {
    let SchurDims { w, .. } = dims;
    let SchurInputs {
        a21_blk,
        a21_off,
        lam_blocks,
        ..
    } = *inp;
    let SparseTail {
        axx,
        fam_dd,
        panel,
        dd_temp,
        b2_temp,
        ..
    } = tail;
    let fam_blks = &a21_blk[a21_off[f]..a21_off[f + 1]];
    // Sparse tail: the family's L21 block never joins a global
    // e-row matrix — it lives on a compact `e_f × w` col-major
    // panel (`e_f` = Σ q over the family's co-occurring crossed
    // blocks; block bi's rows sit at loc0..loc0+q instead of
    // t0..t0+q) and is consumed within this iteration: S22
    // downdate scatter + B̃₂ downdate.
    let e_f: usize = fam_blks.iter().map(|&bi| lam_blocks[bi as usize].q).sum();
    let panel = &mut panel[..e_f * w];
    panel.fill(0.0);
    fold_a21(dims, inp, f, panel, e_f, 0, true);
    schur_phase_b(panel, e_f, w, fam_a);
    // Consumed within this iteration, so the recovery copies it out
    // here or never sees it (the dense branch keeps the global `l21`
    // and needs no copy).
    if let Some(r) = rec.as_mut() {
        r.panel[r.panel_off[f]..r.panel_off[f + 1]].copy_from_slice(panel);
    }
    family_downdate(dims, f, e_f, fam_dd, axx, panel, dd_temp);
    b2_downdate(dims, inp, fam_blks, fb, e_f, panel, u1, b2_temp, u2);
}

fn family_downdate(
    dims: SchurDims,
    f: usize,
    e_f: usize,
    fam_dd: &mut FamDowndate,
    axx: &mut SparseColMat<usize, f64>,
    panel: &[f64],
    dd_temp: &mut [f64],
) {
    let SchurDims { w, e, .. } = dims;
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
                for b in 0..e_f {
                    let col = &mut acc[rmap[b] as usize * e..];
                    for (&r, &d) in rmap[b..].iter().zip(&dd[b * e_f + b..b * e_f + e_f]) {
                        col[r as usize] += d;
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn b2_downdate(
    dims: SchurDims,
    inp: &SchurInputs,
    fam_blks: &[u32],
    fb: usize,
    e_f: usize,
    panel: &[f64],
    u1: &[f64],
    b2_temp: &mut [f64],
    u2: &mut [f64],
) {
    let SchurDims { m, w, kf, e, .. } = dims;
    let SchurInputs { lam_blocks, .. } = *inp;
    // B̃₂ −= L21_f·U1_f. This family's rows of the B2 downdate are one
    // `panel·U1_sub` into `b2_temp` (Accum::Replace; a reassociation of
    // the per-block products, as the dense tail's `B̃₂ = B2 − L21·U1`
    // matmul). U1_sub is rows fb..fb+w of U1 taken as a strided `w×m` view (row-stride 1,
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

/// The dense-tail branch after the family loop: `S22`, its LLT (pivots into
/// `log_lzz_half`) and the `U2` rows. `None` on a non-positive tail pivot.
#[allow(clippy::too_many_arguments)]
fn dense_tail_factor(
    dims: SchurDims,
    inp: &SchurInputs,
    l21: &[f64],
    u1: &[f64],
    s22: &mut Mat<f64>,
    u2: &mut [f64],
    tail_llt_mem: &mut MemBuffer,
    log_lzz_half: &mut f64,
) -> Option<()> {
    let SchurDims { m, kf, e, .. } = dims;
    let SchurInputs {
        pk_a22,
        a22_pairs,
        lam_blocks,
        lam_small,
        ..
    } = *inp;
    // S22 = A22 − L21·L21ᵀ (lower): zero-fill, fold only the co-occurring
    // crossed block pairs (same-factor level pairs never co-occur, so a
    // single crossed factor's A22 is block-DIAGONAL — most pairs skip),
    // add +I, then ONE triangular syrk downdate through faer's blocked
    // FMA kernels (same call shape as `src/lmm/kernel.rs`; RESULT-MOVING
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
        *log_lzz_half += ltt.ln();
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
    Some(())
}

/// The sparse-tail branch after the family loop: the numeric refactor of the
/// assembled `S22`, `X2 = S22⁻¹B̃₂`, and `log|S22|` into `log_lzz_half`. `None`
/// on a non-PD tail.
fn sparse_tail_factor(
    dims: SchurDims,
    tail: &mut SparseTail,
    u2: &[f64],
    log_lzz_half: &mut f64,
) -> Option<()> {
    let SchurDims { m, e, .. } = dims;
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

    // logdet_llt returns 2·Σ ln L_ii (+INFINITY on a non-PD diagonal);
    // log_lzz_half is the ½ convention — add HALF.
    let log_s22 = logdet_llt(&tail.symbolic, &tail.l_values);
    if !log_s22.is_finite() {
        return None;
    }
    *log_lzz_half += 0.5 * log_s22;
    Some(())
}

/// The augmented `m×m` factor `L` of `S = C_xy − UᵀU` into `factor`. `None` on
/// a non-PD pivot.
fn augmented_factor(
    dims: SchurDims,
    inp: &SchurInputs,
    u1: &[f64],
    u2: &[f64],
    tail: &Option<SparseTail>,
    factor: &mut Mat<f64>,
    factor_llt_mem: &mut MemBuffer,
) -> Option<()> {
    let SchurDims { m, kf, e, .. } = dims;
    let SchurInputs { cxy, .. } = *inp;
    // S = C_xy − UᵀU (lower), Cholesky in place → the augmented factor L.
    // The UᵀU assembly is two triangular syrk downdates (a reassociation of
    // the per-entry U1ᵀU1 / U2ᵀU2 dots, as the dense tail's triangular
    // `S22 −= L21·L21ᵀ` matmul in the `s22` None-arm, just before its
    // `Dense tail LLT`), and `cholesky_in_place` is the same call the dense
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
    // .ok()? is the non-PD exit: faer returns Err on any
    // pivot failing `pivot > 0` — non-positive AND non-finite both trip it.
    cholesky_in_place(
        factor.as_mat_mut(),
        LltRegularization::default(),
        Par::Seq,
        MemStack::new(factor_llt_mem),
        Spec::default(),
    )
    .ok()?;
    Some(())
}
