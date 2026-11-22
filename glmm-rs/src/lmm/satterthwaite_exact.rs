//! The exact `H_p` and `∂C_p/∂θ_k` of the Satterthwaite pass, on the dense
//! kernel (the family loop, nested children, the balanced collapse, the
//! per-group regression shapes, the QR kernel and the crossed tail) and on the
//! sparse kernel, up to [`EXACT_TAIL_MAX`] crossed columns.
//!
//! **Closed forms.** With `V = I + ZΛΛᵀZᵀ` in the kernel's units (rows scaled
//! by `√ŵ`, the orthogonalized `X̃`, the random-effect column scaling), `P =
//! V⁻¹ − V⁻¹X̃ΦX̃ᵀV⁻¹`, `Φ = (X̃ᵀV⁻¹X̃)⁻¹ = (L_X L_Xᵀ)⁻¹`, `r² = yᵀPy` and
//! `V_k = ZD_kZᵀ`, `D_k = ∂(ΛΛᵀ)/∂θ_k = E_kΛᵀ + ΛE_kᵀ` (`E_k` the unit entry
//! `(r, c)` of θ_k's block on every level), `D_kl = E_kE_lᵀ + E_lE_kᵀ`, the
//! profiled deviance `ln|V| + ln|X̃ᵀV⁻¹X̃| + ν·ln r²` (REML; ML drops the middle
//! term) has, from `∂ln|V| = tr(V⁻¹∂V)` and `∂V⁻¹ = −V⁻¹(∂V)V⁻¹` (Harville
//! 1977; Searle, Casella & McCulloch 1992, ch. 6; Wolfinger, Tobias & Sall
//! 1994), with `a_k = yᵀPV_kPy`,
//!
//! ```text
//! H_kl = tr(P V_kl) − tr(P V_k P V_l)
//!        + (ν/r²)·(2·yᵀPV_kPV_lPy − yᵀPV_klPy) − (ν/r⁴)·a_k·a_l
//! ∂C_p/∂θ_k = −(a_k/ν)·Φ + σ̂²·ΦX̃ᵀV⁻¹V_kV⁻¹X̃Φ,      C_p = σ̂²·Φ
//! ```
//!
//! and under ML `V⁻¹` takes the place of `P` in the two trace terms. `V` is
//! quadratic in θ, so `V_kl ≠ 0`.
//!
//! **Q-space.** No `n × n` matrix is formed. With `S = ZᵀV⁻¹Z`, `F =
//! ZᵀV⁻¹X̃`, `w = ZᵀPy = ZᵀV⁻¹(y − X̃β̂)`, `N = F·L_X⁻ᵀ`, `S_P = ZᵀPZ = S −
//! NNᵀ` and `T = S_P` (REML) or `S` (ML),
//!
//! ```text
//! H_kl = tr(T D_kl) − tr(T D_k T D_l) + (ν/r²)·(2·wᵀD_kS_PD_lw − wᵀD_klw) − (ν/r⁴)·a_k·a_l
//! ∂C̃_p/∂θ_k = W̃ᵀ·[σ̂²·NᵀD_kN − (a_k/ν)·I]·W̃,    W̃ = L_X⁻¹,  a_k = wᵀD_kw
//! tr(S_P D_k S_P D_l) = tr(S D_k S D_l) − 2·tr(NᵀD_kSD_lN) + tr(NᵀD_kN·NᵀD_lN)
//! tr(S_P D_kl)        = tr(S D_kl) − tr(NᵀD_klN)
//! wᵀD_kS_PD_lw        = wᵀD_kSD_lw − (NᵀD_kw)ᵀ(NᵀD_lw)
//! ```
//!
//! Per level `i`, `D_k` is the rank-2 `e_{a_i}λ_{b_i}ᵀ + λ_{b_i}e_{a_i}ᵀ`
//! (`a_i`, `b_i` the level's columns for rows `r` and `c` of the block, `λ_b`
//! column `b` of `Λ`). So the terms with a matrix `S` between two `D`s are sums
//! over pairs of levels `(i, j)`, `i` of θ_k's grouping and `j` of θ_l's, that
//! `S` couples:
//!
//! ```text
//! tr(S D_k S D_l)   = 2·Σ [ S_{a_i a′_j}·(ΛᵀSΛ)_{b_i b′_j} + (SΛ)_{a_i b′_j}·(SΛ)_{a′_j b_i} ]
//! tr(NᵀD_kSD_lN)    = Σ [ (SΛ)_{a′_j b_i}·v′_j·u_i + (ΛᵀSΛ)_{b_i b′_j}·u′_j·u_i
//!                         + S_{a_i a′_j}·v′_j·v_i + (SΛ)_{a_i b′_j}·u′_j·v_i ]
//!                     u_i = row a_i of N, v_i = row b_i of ΛᵀN
//! wᵀD_kSD_lw        = the same with u_i = w_{a_i}, v_i = (Λᵀw)_{b_i}
//! ```
//!
//! and the rest are sums over single levels: `tr(S D_kl) = 2·δ_{cc′}·Σ
//! S_{a_i a′_i}`, `a_k = 2·Σ w_{a_i}(Λᵀw)_{b_i}`, `NᵀD_kN = Σ [u_i v_iᵀ +
//! v_i u_iᵀ]`, `NᵀD_kw = Σ [u_i (Λᵀw)_{b_i} + v_i w_{a_i}]`.
//!
//! **Families.** With no crossed factor `S` is block-diagonal by family (a
//! primary level and its nested children), so the pairs are the pairs of one
//! family. Each layout supplies, per family, `S_ff = Z_fᵀV_f⁻¹Z_f` and `T_f =
//! Z_fᵀV_f⁻¹[X̃ r]`, with the residual column `r = y − X̃β̂` contracted before
//! any product with θ (`[X̃ y]·[−β̂; 1]`), from the factors its own deviance
//! reads, so the derivative has that path's precision. By Woodbury (Henderson
//! & Searle 1981) with `G_f = Z_fᵀZ_f`, `C_f = Z_fᵀ[X̃ r]` and `A_f = I +
//! Λ_fᵀG_fΛ_f = L_fL_fᵀ`:
//!
//! - Gram statistics (the dense and the sparse kernel): `S_ff = G_f − P_fᵀP_f`,
//!   `T_f = C_f − P_fᵀB_f`, `P_f = L_f⁻¹Λ_fᵀG_f`, `B_f = L_f⁻¹Λ_fᵀC_f`, `L_f`
//!   rebuilt at θ (on the dense kernel as its family loop builds it, also on
//!   the collapse and per-group shapes, as the conditional-mode recovery does).
//! - QR triangles (`QrKernel::family_inputs`): `S_ff = YᵀY`, `T_f =
//!   Yᵀ·R̃⁻ᵀR_Z·`, `Y = R̃⁻ᵀR_ZZ`, with no subtraction.
//!
//! **Crossed tails.** Order the random-effect columns families first, then the
//! `e` crossed columns `x`, as both kernels eliminate them. With `V_f = I +
//! Z_FΛ_FΛ_FᵀZ_Fᵀ` (families only), `S⁰ = ZᵀV_f⁻¹Z`, `U = S⁰[:, x]·Λ_x`
//! (`Q × e`) and `T_c = I + Λ_xᵀS⁰_xxΛ_x` (the crossed tail the kernels factor),
//! Woodbury gives `S = S⁰ − W`, `W = U·T_c⁻¹·Uᵀ`, and `ZᵀV⁻¹[X̃ r] =
//! ZᵀV_f⁻¹[X̃ r] − U·T_c⁻¹·Λ_xᵀZ_xᵀV_f⁻¹[X̃ r]`. On Gram statistics `S⁰_fx =
//! G_fx − P_fᵀQ_f`, `Q_f = L_f⁻¹Λ_fᵀG_fx`, and `S⁰_xx = G_xx − Σ_f Q_fᵀQ_f`
//! (in general full), so `U_f = S⁰_fxΛ_x` lives on the crossed columns family
//! `f` touches. The blocks of `S` a family reads are formed whole: `S_ff =
//! S⁰_ff − U_fT_c⁻¹U_fᵀ`, `S_fx = S⁰_fx − U_f·T_c⁻¹U_xᵀ` (every crossed column)
//! and `S_xx = S⁰_xx − U_xT_c⁻¹U_xᵀ`, so the pairs of one family, of a family
//! with a crossed level and of two crossed levels read the true `S`. Two
//! different families couple only through `S_ff′ = −W_ff′`, every pair of
//! them: those pairs are the sum over all pairs of families less the same
//! family's, the first through `X_k = U_FᵀD_kU_F` (`e × e`, family
//! coordinates), `Y_k = U_FᵀD_kN_F` and `y_k = U_FᵀD_kw_F`:
//!
//! ```text
//! Σ_{f,f′} τ(W)   = tr(T_c⁻¹X_k·T_c⁻¹X_l)        (tr(S D_k S D_l), quadratic in S)
//! Σ_{f,f′} σ_N(W) = tr(Y_kᵀ·T_c⁻¹·Y_l)           (tr(NᵀD_kSD_lN), linear in S)
//! Σ_{f,f′} σ_w(W) = y_kᵀ·T_c⁻¹·y_l               (wᵀD_kSD_lw)
//! ```
//!
//! so the families' pairs add `Σ_f [τ(S_ff) − τ(W_ff)] + tr(T_c⁻¹X_kT_c⁻¹X_l)`
//! and `Σ_f [σ(S_ff) + σ(W_ff)] − tr(Y_kᵀT_c⁻¹Y_l)` (and the same for `w`).
//! Splitting the crossed pairs the same way (`S⁰` pairs less the `W` pairs)
//! would cancel as `T_c²`, which is large for a crossed factor with many rows
//! per level; forming `S_xx` and `S_fx` whole cancels once. The remainder
//! pairs every two crossed columns through `T_c⁻¹`, which cannot be read off
//! a sparse factor's pattern: that is the reason for the width limit.
//!
//! Harville, D. A. (1977). Maximum likelihood approaches to variance
//! component estimation and to related problems. *JASA* 72(358), 320–338.
//! Henderson, H. V. & Searle, S. R. (1981). On deriving the inverse of a sum
//! of matrices. *SIAM Review* 23(1), 53–60.
//! Searle, S. R., Casella, G. & McCulloch, C. E. (1992). *Variance
//! Components*. Wiley.
//! Wolfinger, R., Tobias, R. & Sall, J. (1994). Computing Gaussian
//! likelihoods and their derivatives for general linear mixed models. *SIAM
//! Journal on Scientific Computing* 15(6), 1294–1310.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::{Mat, MatMut, MatRef, Par};

use super::super::{
    assemble_fam_a, beta_backward, forward_solve, primary_gram, primary_lambda, Criterion,
    LmmFitScratch, LmmGroupings, LmmKernel, LmmSuffStats,
};
use crate::sparse::SparseLmmWorkspace;

/// The widest crossed tail, in crossed random-effect columns (levels, on
/// intercept-only crossed factors), that takes the exact pass; a wider one
/// takes the value route. The exact pass holds `6 + n` dense `e × e` buffers
/// (`n` the free coordinates) while the fit's own tail buffers are alive, 2 MB
/// each at 500, and costs `O(n·e³)`. Measured on crossed designs of 210 to
/// 3000 columns, its instruction count was 0.14 to 1.8 times the value
/// route's; the width bounds memory. The same width as the GLMM's exact
/// Hessian, `glmm::derivative::DUAL_TAIL_MAX`.
pub(crate) const EXACT_TAIL_MAX: usize = 500;

/// Whether the exact pass covers the kernel: a crossed tail of at most
/// [`EXACT_TAIL_MAX`] columns.
pub(super) fn covers(kernel: &LmmKernel) -> bool {
    kernel.groupings().k_crossed() <= EXACT_TAIL_MAX
}

/// The sums over levels and level pairs, per free coordinate or pair `(k, l)`,
/// `k ≤ l`, at a stride of `n_θ`.
#[derive(Default)]
pub(super) struct Sums {
    /// `tr(S D_k S D_l)`, `tr(NᵀD_kSD_lN)`, `wᵀD_kSD_lw`, `tr(S D_kl)`,
    /// `tr(NᵀD_klN)`, `wᵀD_klw`.
    tr_ss: Vec<f64>,
    tr_nsn: Vec<f64>,
    w_sw: Vec<f64>,
    tr_kl: Vec<f64>,
    n_kl_n: Vec<f64>,
    w_kl_w: Vec<f64>,
    /// `a_k`, `NᵀD_kN` (p × p each, row-major) and `NᵀD_kw` (p each).
    pub(super) a: Vec<f64>,
    pub(super) ndn: Vec<f64>,
    ndw: Vec<f64>,
}

/// The buffers of [`exact_terms`], row-major. Grown on the first call to the
/// shapes it meets; later calls on the same workspace reuse them.
#[derive(Default)]
pub(super) struct ExactScratch {
    pub(super) sums: Sums,
    /// One `p × p` product of the caller's `∂C/∂θ_k`.
    pub(super) prod: Vec<f64>,
    /// Per free coordinate its `(block, column)`; its levels in a family
    /// (`fpos[fpos_start[k]..fpos_start[k + 1]]`, family columns `(a_i, b_i)`)
    /// and among the crossed columns (`xpos`, crossed columns); and for a
    /// crossed coordinate its `(crossed factor, r, c)`.
    key: Vec<(usize, usize)>,
    fpos: Vec<(usize, usize)>,
    fpos_start: Vec<usize>,
    xpos: Vec<(usize, usize)>,
    xpos_start: Vec<usize>,
    xcoord: Vec<Option<(usize, usize, usize)>>,
    /// Λ of a family: its diagonal blocks `(start, q, offset into flam)` and
    /// dense (`w × w`). Λ of the crossed columns: blocks into `xlam`; and of
    /// the crossed columns one family touches, in their order in `xi`.
    flam: Vec<f64>,
    fblk: Vec<(usize, usize, usize)>,
    lam: Vec<f64>,
    xlam: Vec<f64>,
    xlam_off: Vec<usize>,
    xblk: Vec<(usize, usize, usize)>,
    lblk: Vec<(usize, usize, usize)>,
    /// The crossed columns family `f` touches (`e_f`, ascending).
    xi: Vec<usize>,
    /// Per family (`w` its width, `m = p + 1`, `e_f`): `G_f`, `C_f`, `G_fx`,
    /// `L_f`, `P_f`, `B_f`, `Q_f`, the QR builder's `[Y | Z]`, and the inputs
    /// `S⁰_ff` (then `S_ff`), `T_f` (then `Z_fᵀV⁻¹[X̃ r]`) and `S⁰_fx`.
    g: Vec<f64>,
    cf: Vec<f64>,
    gfx: Vec<f64>,
    fa: Vec<f64>,
    pf: Vec<f64>,
    bf: Vec<f64>,
    qf: Vec<f64>,
    yz: Vec<f64>,
    s: Vec<f64>,
    t: Vec<f64>,
    sfx: Vec<f64>,
    /// `S_ffΛ_f`, `Λ_fᵀS_ffΛ_f`, `W_ff`, `W_ffΛ_f`, `Λ_fᵀW_ffΛ_f` (w × w).
    sl: Vec<f64>,
    lsl: Vec<f64>,
    wff: Vec<f64>,
    wsl: Vec<f64>,
    wlsl: Vec<f64>,
    /// `U_f = S⁰_fxΛ_x`, `Λ_fᵀU_f` and a product (w × e_f, on the touched
    /// columns); `T_c⁻¹` on the touched columns (e_f × e_f); `S_fx`,
    /// `S_fxΛ_x`, `Λ_fᵀS_fx`, `Λ_fᵀS_fxΛ_x` (w × e, every crossed column).
    uf: Vec<f64>,
    lsl_x: Vec<f64>,
    tmp: Vec<f64>,
    tinvf: Vec<f64>,
    sfxf: Vec<f64>,
    sfxf_l: Vec<f64>,
    l_sfxf: Vec<f64>,
    l_sfxf_l: Vec<f64>,
    /// `N_f`, `Λ_fᵀN_f` (w × p), `w_f`, `Λ_fᵀw_f` (w).
    nm: Vec<f64>,
    ln: Vec<f64>,
    wv: Vec<f64>,
    lw: Vec<f64>,
    /// `L_X` (p × p, column-major), `β̃`, `[−β̃; 1]`, one row of `N`.
    lx: Vec<f64>,
    beta: Vec<f64>,
    ev: Vec<f64>,
    u: Vec<f64>,
    /// The crossed tail (e × e): `S⁰_xx` then `S_xx`, `T_c⁻¹`, `U_x =
    /// S⁰_xxΛ_x`, `T_c⁻¹U_xᵀ`, `S_xxΛ_x`, `Λ_xᵀS_xxΛ_x`.
    s0x: Vec<f64>,
    tinv: Vec<f64>,
    ux: Vec<f64>,
    ut: Vec<f64>,
    sxl: Vec<f64>,
    sxlx: Vec<f64>,
    /// `Z_xᵀV_f⁻¹[X̃ r]` then `Z_xᵀV⁻¹[X̃ r]`, `T_c⁻¹Λ_xᵀZ_xᵀV_f⁻¹[X̃ r]`, and
    /// `Λ_xᵀ` of the first (e × m); `N_x`, `Λ_xᵀN_x` (e × p), `w_x`, `Λ_xᵀw_x`.
    rx: Vec<f64>,
    rc: Vec<f64>,
    lr: Vec<f64>,
    nx: Vec<f64>,
    lnx: Vec<f64>,
    wx: Vec<f64>,
    lwx: Vec<f64>,
    /// Per free coordinate `X_k` then `T_c⁻¹X_k` (e × e), `Y_k`, `T_c⁻¹Y_k`
    /// (e × p), `y_k`, `T_c⁻¹y_k` (e).
    xk: Vec<f64>,
    yk: Vec<f64>,
    tyk: Vec<f64>,
    yvk: Vec<f64>,
    tyvk: Vec<f64>,
}

/// `v` at length at least `n`, zero-filled where it grows; its first `n`
/// entries are the caller's to overwrite.
fn grow(v: &mut Vec<f64>, n: usize) {
    if v.len() < n {
        v.resize(n, 0.0);
    }
}

impl ExactScratch {
    /// `NᵀD_kN` of free coordinate `k`, `p × p` row-major.
    pub(super) fn ndn(&self, k: usize, p: usize) -> &[f64] {
        &self.sums.ndn[k * p * p..(k + 1) * p * p]
    }
}

/// Block, row and column `(b, r, c)` of θ index `k`: block `b` (the primary,
/// then each extra in declaration order) holds `vech(Λ_b)` column-major.
fn coordinate(g: &LmmGroupings, k: usize) -> (usize, usize, usize) {
    let mut b = 0;
    while g.vech_start(b + 1) <= k {
        b += 1;
    }
    let q = g.block_q(b);
    let (mut t, mut c) = (k - g.vech_start(b), 0);
    while t >= q - c {
        t -= q - c;
        c += 1;
    }
    (b, c + t, c)
}

/// A block-diagonal Λ: blocks `(start, q, offset)`, entry `(r, c)`, `r ≥ c`,
/// of a block at `vals[offset + r·q + c]`. Every column belongs to a block.
#[derive(Clone, Copy)]
struct Lam<'a> {
    blocks: &'a [(usize, usize, usize)],
    vals: &'a [f64],
}

impl Lam<'_> {
    /// `out = S·Λ`, `S` `rows × w` row-major.
    fn right(&self, s: &[f64], rows: usize, w: usize, out: &mut [f64]) {
        for i in 0..rows {
            for &(st, q, off) in self.blocks {
                for j in 0..q {
                    out[i * w + st + j] = (j..q)
                        .map(|k| s[i * w + st + k] * self.vals[off + k * q + j])
                        .sum();
                }
            }
        }
    }

    /// `out = Λᵀ·S`, `S` `w × cols` row-major.
    fn left_t(&self, s: &[f64], cols: usize, out: &mut [f64]) {
        for &(st, q, off) in self.blocks {
            for r in 0..q {
                for c in 0..cols {
                    out[(st + r) * cols + c] = (r..q)
                        .map(|k| self.vals[off + k * q + r] * s[(st + k) * cols + c])
                        .sum();
                }
            }
        }
    }
}

/// One block pair `(A, B)`'s products of a symmetric `S`, row-major `w_A ×
/// w_B`: `S_AB`, `S_ABΛ_B`, `Λ_AᵀS_AB` (unused on one block, where it is
/// `(S_AAΛ_A)ᵀ`) and `Λ_AᵀS_ABΛ_B`.
#[derive(Clone, Copy)]
struct Prods<'a> {
    s: &'a [f64],
    sl: &'a [f64],
    ls: &'a [f64],
    lsl: &'a [f64],
}

/// One block's rows of `N` and `ΛᵀN` (w × p), and its `w` and `Λᵀw`.
#[derive(Clone, Copy)]
struct Vecs<'a> {
    n: &'a [f64],
    ln: &'a [f64],
    wv: &'a [f64],
    lw: &'a [f64],
}

/// Each free coordinate's levels on one block: `pos[start[k]..start[k + 1]]`.
#[derive(Clone, Copy)]
struct Levels<'a> {
    pos: &'a [(usize, usize)],
    start: &'a [usize],
}

impl Levels<'_> {
    fn of(&self, k: usize) -> &[(usize, usize)] {
        &self.pos[self.start[k]..self.start[k + 1]]
    }
}

/// `a_i · b_j` over `p` columns.
fn dot(a: &[f64], i: usize, b: &[f64], j: usize, p: usize) -> f64 {
    (0..p).map(|t| a[i * p + t] * b[j * p + t]).sum()
}

/// Adds the level-pair sums of `tr(S D_k S D_l)`, `tr(NᵀD_kSD_lN)` and
/// `wᵀD_kSD_lw` over the pairs of block `A`'s levels `la` and block `B`'s `lb`,
/// `s` the products of `S` between them, for every coordinate pair (`same`:
/// one block, the pairs `k ≤ l`). With `w` (one family's block of `W`):
/// `tr(S D_k S D_l)` takes the pair sum of `W` with a minus sign and the
/// other two with a plus sign (see [`family_terms`]).
#[allow(clippy::too_many_arguments)] // two blocks' products, vectors and levels
fn add_pairs(
    sums: &mut Sums,
    same: bool,
    wa: usize,
    wb: usize,
    p: usize,
    s: Prods,
    w: Option<Prods>,
    va: Vecs,
    vb: Vecs,
    la: Levels,
    lb: Levels,
    n: usize,
    st: usize,
) {
    // One matrix's terms of the pair (i, j): S_{a_i a′_j}·(ΛᵀSΛ)_{b_i b′_j} +
    // (SΛ)_{a_i b′_j}·(SΛ)_{a′_j b_i}, and the N and w sums. (SΛ)_{a′_j b_i}
    // is (SΛ)[a′_j, b_i] on one block and (ΛᵀS)[b_i, a′_j] across two.
    let terms = |m: &Prods, ai: usize, bi: usize, aj: usize, bj: usize| {
        let (saa, m3bb, m2ab) = (m.s[ai * wb + aj], m.lsl[bi * wb + bj], m.sl[ai * wb + bj]);
        let m2ba = if same {
            m.sl[aj * wa + bi]
        } else {
            m.ls[bi * wb + aj]
        };
        (
            saa * m3bb + m2ab * m2ba,
            m2ba * dot(va.n, ai, vb.ln, bj, p)
                + m3bb * dot(va.n, ai, vb.n, aj, p)
                + saa * dot(va.ln, bi, vb.ln, bj, p)
                + m2ab * dot(vb.n, aj, va.ln, bi, p),
            m2ba * va.wv[ai] * vb.lw[bj]
                + m3bb * vb.wv[aj] * va.wv[ai]
                + saa * vb.lw[bj] * va.lw[bi]
                + m2ab * vb.wv[aj] * va.lw[bi],
        )
    };
    for ka in 0..n {
        let lka = la.of(ka);
        if lka.is_empty() {
            continue;
        }
        for kb in if same { ka } else { 0 }..n {
            let lkb = lb.of(kb);
            if lkb.is_empty() {
                continue;
            }
            let (mut tss, mut tnsn, mut wsw) = (0.0, 0.0, 0.0);
            let (mut tw, mut nw, mut ww) = (0.0, 0.0, 0.0);
            for &(ai, bi) in lka {
                for &(aj, bj) in lkb {
                    let (t, u, v) = terms(&s, ai, bi, aj, bj);
                    tss += t;
                    tnsn += u;
                    wsw += v;
                    if let Some(w) = &w {
                        let (t, u, v) = terms(w, ai, bi, aj, bj);
                        tw += t;
                        nw += u;
                        ww += v;
                    }
                }
            }
            let at = ka.min(kb) * st + ka.max(kb);
            if w.is_some() {
                sums.tr_ss[at] += 2.0 * (tss - tw);
                sums.tr_nsn[at] += tnsn + nw;
                sums.w_sw[at] += wsw + ww;
            } else {
                sums.tr_ss[at] += 2.0 * tss;
                sums.tr_nsn[at] += tnsn;
                sums.w_sw[at] += wsw;
            }
        }
    }
}

/// Adds the single-level sums of one block (`w` wide, `s` its `S`): `a_k`,
/// `NᵀD_kN`, `NᵀD_kw`, and per pair of coordinates in one block and column,
/// `tr(S D_kl)`, `tr(NᵀD_klN)` and `wᵀD_klw`.
#[allow(clippy::too_many_arguments)] // one block's matrices, vectors and levels
fn add_levels(
    sums: &mut Sums,
    w: usize,
    p: usize,
    s: &[f64],
    v: Vecs,
    l: Levels,
    key: &[(usize, usize)],
    n: usize,
    st: usize,
) {
    for ka in 0..n {
        let la = l.of(ka);
        for &(ai, bi) in la {
            sums.a[ka] += 2.0 * v.wv[ai] * v.lw[bi];
            let ndn = &mut sums.ndn[ka * p * p..(ka + 1) * p * p];
            for u in 0..p {
                for c in 0..p {
                    ndn[u * p + c] +=
                        v.n[ai * p + u] * v.ln[bi * p + c] + v.ln[bi * p + u] * v.n[ai * p + c];
                }
                sums.ndw[ka * p + u] += v.n[ai * p + u] * v.lw[bi] + v.ln[bi * p + u] * v.wv[ai];
            }
        }
        if la.is_empty() {
            continue;
        }
        // D_kl ≠ 0 only for θ_k, θ_l in one block and one column, level by
        // level.
        for kb in ka..n {
            if key[ka] != key[kb] {
                continue;
            }
            let at = ka * st + kb;
            for (&(ai, _), &(aj, _)) in la.iter().zip(l.of(kb)) {
                sums.tr_kl[at] += 2.0 * s[ai * w + aj];
                sums.n_kl_n[at] += 2.0 * dot(v.n, ai, v.n, aj, p);
                sums.w_kl_w[at] += 2.0 * v.wv[ai] * v.wv[aj];
            }
        }
    }
}

/// `out = a·b` (row-major `r × k` times `k × c`) at `Par::Seq`; `bt`: `b` is
/// given transposed (`c × k`).
fn gemm(out: &mut [f64], a: &[f64], b: &[f64], r: usize, k: usize, c: usize, bt: bool) {
    let am = MatRef::from_row_major_slice(&a[..r * k], r, k);
    let bm = if bt {
        MatRef::from_row_major_slice(&b[..c * k], c, k).transpose()
    } else {
        MatRef::from_row_major_slice(&b[..k * c], k, c)
    };
    faer::linalg::matmul::matmul(
        MatMut::from_row_major_slice_mut(&mut out[..r * c], r, c),
        faer::Accum::Replace,
        am,
        bm,
        1.0,
        Par::Seq,
    );
}

/// `H_p` over the free θ̃ coordinates `free` at `theta` into `hess`
/// (`n × n`, exactly symmetric), with `a_k` and `NᵀD_kN` left in `x` for the
/// caller's `∂C/∂θ_k`. Reads the factor the caller's deviance evaluation at
/// `theta` left (`L_X`, `β̃`, `r`, and on the sparse kernel this θ's Λ);
/// `nu` is the divisor of σ̂². `false` when a family factor or the crossed
/// tail is not positive definite (every entry of `hess` is then NaN). Only on
/// a kernel the pass [`covers`].
pub(super) fn exact_terms(
    kernel: &mut LmmKernel,
    theta: &[f64],
    free: &[usize],
    nu: f64,
    x: &mut ExactScratch,
    hess: &mut Mat<f64>,
) -> bool {
    let n = free.len();
    hess.resize_with(n, n, |_, _| f64::NAN);
    if !walk(kernel, theta, free, x) {
        for j in 0..n {
            for i in 0..n {
                hess[(i, j)] = f64::NAN;
            }
        }
        return false;
    }
    let f = kernel.factor();
    let p = f.nrows() - 1;
    let r = f[(p, p)];
    let r2 = r * r;
    let reml = kernel.criterion() == Criterion::Reml;
    let st = theta.len();
    let sm = &x.sums;
    for a in 0..n {
        for b in a..n {
            let at = a * st + b;
            let (na, nb) = (x.ndn(a, p), x.ndn(b, p));
            let (trt, trtt) = if reml {
                // tr(NᵀD_kN·NᵀD_lN), both symmetric.
                let tt: f64 = na.iter().zip(nb).map(|(u, v)| u * v).sum();
                (
                    sm.tr_kl[at] - sm.n_kl_n[at],
                    sm.tr_ss[at] - 2.0 * sm.tr_nsn[at] + tt,
                )
            } else {
                (sm.tr_kl[at], sm.tr_ss[at])
            };
            let ndw_dot: f64 = (0..p).map(|i| sm.ndw[a * p + i] * sm.ndw[b * p + i]).sum();
            let w_sp_w = sm.w_sw[at] - ndw_dot;
            let h = trt - trtt + nu / r2 * (2.0 * w_sp_w - sm.w_kl_w[at])
                - nu / (r2 * r2) * sm.a[a] * sm.a[b];
            hess[(a, b)] = h;
            hess[(b, a)] = h;
        }
    }
    true
}

/// Each free coordinate's block and levels, and the family and crossed Λ at
/// `theta`, into `x`.
fn setup(g: &LmmGroupings, theta: &[f64], free: &[usize], x: &mut ExactScratch) {
    let (q, np) = (g.primary_q, g.nested_per_parent);
    let q_n = g.nested.map_or(0, |nf| nf.q);
    let kf = g.k_family();
    x.key.clear();
    x.fpos.clear();
    x.fpos_start.clear();
    x.xpos.clear();
    x.xpos_start.clear();
    x.xcoord.clear();
    for &k in free {
        let (b, r, c) = coordinate(g, k);
        x.key.push((b, c));
        x.fpos_start.push(x.fpos.len());
        x.xpos_start.push(x.xpos.len());
        if b == 0 {
            x.fpos.push((r, c));
            x.xcoord.push(None);
        } else if g.nested.is_some_and(|nf| nf.decl + 1 == b) {
            // One level per nested child, at family columns q + j·q_n.
            x.fpos
                .extend((0..np).map(|j| (q + j * q_n + r, q + j * q_n + c)));
            x.xcoord.push(None);
        } else {
            let ci = g
                .crossed
                .iter()
                .position(|cf| cf.decl + 1 == b)
                .expect("an extra block is nested or crossed");
            let cf = &g.crossed[ci];
            let base = g.extra_offsets[cf.decl] - kf;
            x.xpos
                .extend((0..cf.n_levels).map(|l| (base + l * cf.q + r, base + l * cf.q + c)));
            x.xcoord.push(Some((ci, r, c)));
        }
    }
    x.fpos_start.push(x.fpos.len());
    x.xpos_start.push(x.xpos.len());

    // Λ_f = blockdiag(Λ_p, Λ_n per child): blocks into flam, and dense.
    let w = q + np * q_n;
    x.flam.clear();
    x.flam.resize(q * q + q_n * q_n, 0.0);
    primary_lambda(theta, q, &mut x.flam[..q * q]);
    if let Some(nf) = g.nested {
        primary_lambda(&theta[nf.vech_start..], q_n, &mut x.flam[q * q..]);
    }
    x.fblk.clear();
    x.fblk.push((0, q, 0));
    x.fblk.extend((0..np).map(|j| (q + j * q_n, q_n, q * q)));
    grow(&mut x.lam, w * w);
    x.lam[..w * w].fill(0.0);
    for &(st, bq, off) in &x.fblk {
        for r in 0..bq {
            for c in 0..=r {
                x.lam[(st + r) * w + st + c] = x.flam[off + r * bq + c];
            }
        }
    }
    // Λ_x: one block per crossed level, each factor's q × q shared.
    x.xlam.clear();
    x.xlam_off.clear();
    x.xblk.clear();
    for cf in &g.crossed {
        let off = x.xlam.len();
        x.xlam_off.push(off);
        x.xlam.resize(off + cf.q * cf.q, 0.0);
        primary_lambda(&theta[cf.vech_start..], cf.q, &mut x.xlam[off..]);
        let base = g.extra_offsets[cf.decl] - kf;
        x.xblk
            .extend((0..cf.n_levels).map(|l| (base + l * cf.q, cf.q, off)));
    }
}

/// `L_X`, `β̃` and `[−β̃; 1]` off the augmented factor `f` of the evaluation at
/// θ, into `x`.
fn read_factor(f: MatRef<'_, f64>, x: &mut ExactScratch) {
    let m = f.nrows();
    let p = m - 1;
    grow(&mut x.lx, p * p);
    grow(&mut x.beta, p);
    grow(&mut x.ev, m);
    grow(&mut x.u, p);
    grow(&mut x.prod, p * p);
    for j in 0..p {
        for i in 0..p {
            x.lx[j * p + i] = if i >= j { f[(i, j)] } else { 0.0 };
        }
    }
    beta_backward(f, p, &mut x.beta[..p]);
    for j in 0..p {
        x.ev[j] = -x.beta[j];
    }
    x.ev[p] = 1.0;
}

/// Zeroes the sums for `n_θ` coordinates and `p` columns.
fn reset_sums(sm: &mut Sums, n_theta: usize, p: usize) {
    let nn2 = n_theta * n_theta;
    for v in [
        &mut sm.tr_ss,
        &mut sm.tr_nsn,
        &mut sm.w_sw,
        &mut sm.tr_kl,
        &mut sm.n_kl_n,
        &mut sm.w_kl_w,
    ] {
        grow(v, nn2);
        v[..nn2].fill(0.0);
    }
    grow(&mut sm.a, n_theta);
    grow(&mut sm.ndn, n_theta * p * p);
    grow(&mut sm.ndw, n_theta * p);
    sm.a[..n_theta].fill(0.0);
    sm.ndn[..n_theta * p * p].fill(0.0);
    sm.ndw[..n_theta * p].fill(0.0);
}

/// Where one family's Gram inputs come from.
enum Src<'a> {
    Dense {
        suff: &'a LmmSuffStats,
        prim_gram: &'a mut [f64],
        prim_lam: &'a [f64],
        th_p: f64,
        th_n: f64,
    },
    Sparse(&'a SparseLmmWorkspace),
}

/// The walk over the families and the crossed tail: each layout's per-family
/// inputs, then [`family_terms`], then the crossed tail's own terms. `false`
/// on a family factor or a crossed tail that is not positive definite.
fn walk(kernel: &mut LmmKernel, theta: &[f64], free: &[usize], x: &mut ExactScratch) -> bool {
    let n_theta = theta.len();
    match kernel {
        LmmKernel::Dense { suff, fit } => {
            let g = &suff.groupings;
            read_factor(fit.factor.as_ref(), x);
            setup(g, theta, free, x);
            reset_sums(&mut x.sums, n_theta, suff.m - 1);
            debug_assert!(
                g.nested.is_none_or(|nf| nf.q == 1),
                "a dense nested block is an intercept"
            );
            let LmmFitScratch {
                prim_gram,
                prim_lam,
                qr,
                ..
            } = &mut **fit;
            let (prim_gram, prim_lam) = (&mut prim_gram[..], &mut prim_lam[..]);
            let q = g.primary_q;
            if q > 1 {
                primary_lambda(theta, q, prim_lam);
            }
            let th_p = theta[0];
            let th_n = g.nested.map_or(0.0, |nf| theta[nf.vech_start]);
            if let Some(k) = qr.as_deref_mut().filter(|k| k.on) {
                // The QR kernel covers no crossed factor.
                let (w, m) = (q + g.nested_per_parent, suff.m);
                for v in [&mut x.yz, &mut x.s, &mut x.t] {
                    grow(v, w * (w + m));
                }
                for f in 0..g.n_primary {
                    if !k.family_inputs(
                        f, q, prim_lam, th_p, th_n, &x.ev, &mut x.yz, &mut x.s, &mut x.t,
                    ) {
                        return false;
                    }
                    family_terms(x, w, m - 1, 0, free.len(), n_theta);
                }
                return true;
            }
            let mut src = Src::Dense {
                suff,
                prim_gram,
                prim_lam: &*prim_lam,
                th_p,
                th_n,
            };
            gram_walk(&mut src, g, suff.m, free.len(), n_theta, x)
        }
        LmmKernel::Sparse { ws, .. } => {
            let ws = LmmKernel::sparse_ws(ws);
            read_factor(ws.factor.as_ref(), x);
            setup(&ws.g, theta, free, x);
            reset_sums(&mut x.sums, n_theta, ws.p);
            gram_walk(&mut Src::Sparse(ws), &ws.g, ws.m, free.len(), n_theta, x)
        }
    }
}

/// The walk on Gram statistics: with a crossed tail, a first pass over the
/// families for `S⁰_xx` and `Z_xᵀV_f⁻¹[X̃ r]` and the crossed tail's own
/// quantities ([`crossed_prep`]); then per family its inputs and
/// [`family_terms`]; then [`crossed_terms`].
fn gram_walk(
    src: &mut Src,
    g: &LmmGroupings,
    m: usize,
    n: usize,
    st: usize,
    x: &mut ExactScratch,
) -> bool {
    let p = m - 1;
    let e = g.k_crossed();
    let w = g.primary_q + g.nested_per_parent * g.nested.map_or(0, |nf| nf.q);
    if e > 0 {
        grow(&mut x.s0x, e * e);
        grow(&mut x.rx, e * m);
        src.crossed_grams(g, m, x);
        for f in 0..g.n_primary {
            if !src.family(g, f, m, x) {
                return false;
            }
            let ef = x.xi.len();
            family_s0(x, w, m, ef);
            // S⁰_xx −= Q_fᵀQ_f, Z_xᵀV_f⁻¹[X̃ r] −= Q_fᵀB_f, on the touched columns.
            for a in 0..ef {
                let ga = x.xi[a];
                for b in 0..ef {
                    let v: f64 = (0..w).map(|r| x.qf[r * ef + a] * x.qf[r * ef + b]).sum();
                    x.s0x[ga * e + x.xi[b]] -= v;
                }
                for j in 0..m {
                    let v: f64 = (0..w).map(|r| x.qf[r * ef + a] * x.bf[r * m + j]).sum();
                    x.rx[ga * m + j] -= v;
                }
            }
        }
        if !crossed_prep(x, e, m) {
            return false;
        }
        for (v, len) in [
            (&mut x.xk, n * e * e),
            (&mut x.yk, n * e * p),
            (&mut x.yvk, n * e),
        ] {
            grow(v, len);
            v[..len].fill(0.0);
        }
    }
    for f in 0..g.n_primary {
        if !src.family(g, f, m, x) {
            return false;
        }
        family_s0(x, w, m, x.xi.len());
        family_terms(x, w, p, e, n, st);
    }
    if e > 0 {
        crossed_terms(x, e, p, n, st);
    }
    true
}

impl Src<'_> {
    /// Family `f`'s `G_f` (`x.g`), `C_f` (`x.cf`, the residual column
    /// contracted), its touched crossed columns (`x.xi`, `x.lblk`) with
    /// `G_fx` (`x.gfx`), and `L_f` (`x.fa`). `false` when `A_f` is not
    /// positive definite.
    fn family(&mut self, g: &LmmGroupings, f: usize, m: usize, x: &mut ExactScratch) -> bool {
        let (q, np, n_prim) = (g.primary_q, g.nested_per_parent, g.n_primary);
        let q_n = g.nested.map_or(0, |nf| nf.q);
        let w = q + np * q_n;
        let p = m - 1;
        grow(&mut x.g, w * w);
        grow(&mut x.cf, w * m);
        grow(&mut x.fa, w * w);
        x.xi.clear();
        x.lblk.clear();
        match self {
            Src::Dense {
                suff,
                prim_gram,
                prim_lam,
                th_p,
                th_n,
            } => {
                let slope = q > 1;
                assemble_fam_a(
                    &mut x.fa, prim_gram, prim_lam, suff, f, w, *th_p, *th_n, slope,
                );
                if !crate::linalg::block_chol(&mut x.fa[..w * w], w) {
                    return false;
                }
                // G_f: the primary block, each nested child's count and its
                // cross-Gram with the primary columns (intercept: the child's
                // count; slope d: its Σ x_{slope d}); children never co-occur.
                // RE columns as `from_cluster_spec_ext` lays them out: primary
                // component d at `d·n_primary + f`, child c at
                // `q·n_primary + f·np + c`.
                x.g[..w * w].fill(0.0);
                if slope {
                    primary_gram(suff, g, f, q, prim_gram);
                    for r in 0..q {
                        x.g[r * w..r * w + q].copy_from_slice(&prim_gram[r * q..(r + 1) * q]);
                    }
                } else {
                    x.g[0] = suff.counts[f];
                }
                let col = |r: usize| {
                    if r < q {
                        r * n_prim + f
                    } else {
                        q * n_prim + f * np + (r - q)
                    }
                };
                for c in 0..np {
                    let gc = col(q + c);
                    let n_c = suff.counts[gc];
                    x.g[(q + c) * w + q + c] = n_c;
                    for e in 0..q {
                        let v = if e == 0 {
                            n_c
                        } else {
                            suff.s_raw[(g.primary_slope_cols[e - 1], gc)]
                        };
                        x.g[(q + c) * w + e] = v;
                        x.g[e * w + q + c] = v;
                    }
                }
                for r in 0..w {
                    let sc = suff.s.col(col(r)).try_as_col_major().unwrap().as_slice();
                    let row = &mut x.cf[r * m..(r + 1) * m];
                    row[..p].copy_from_slice(&sc[..p]);
                    row[p] = (0..m).map(|j| sc[j] * x.ev[j]).sum();
                }
                // Touched crossed columns: those the family's intercept column
                // co-occurs with (a child's rows lie inside its parent), each
                // one intercept level. G_fx rows as `assemble_fam_b` reads them.
                let kx = g.k_crossed();
                if kx > 0 {
                    for b in 0..kx {
                        if suff.zx[(f, b)] != 0.0 {
                            x.xi.push(b);
                        }
                    }
                    let ef = x.xi.len();
                    grow(&mut x.gfx, w * ef);
                    for (t, &b) in x.xi.iter().enumerate() {
                        for r in 0..w {
                            x.gfx[r * ef + t] = if r == 0 {
                                suff.zx[(f, b)]
                            } else if r < q {
                                suff.zx_slope[(r * n_prim + f, b)]
                            } else {
                                suff.zx[(col(r), b)]
                            };
                        }
                        let off = x.xblk[b].2;
                        x.lblk.push((t, 1, off));
                    }
                }
            }
            Src::Sparse(ws) => {
                let fam_len = q * q + np * (q_n * q + q_n * q_n);
                let fam = &ws.pk_fam[f * fam_len..(f + 1) * fam_len];
                // G_f from the packed family stream: (primary, primary), then
                // per child (child, primary) and (child, child); children never
                // co-occur.
                x.g[..w * w].fill(0.0);
                for a in 0..q {
                    for b in 0..q {
                        x.g[a * w + b] = fam[a * q + b];
                    }
                }
                for c in 0..np {
                    let (b2, rc) = (q * q + c * (q_n * q + q_n * q_n), q + c * q_n);
                    let b3 = b2 + q_n * q;
                    for a in 0..q_n {
                        for b in 0..q {
                            let v = fam[b2 + a * q + b];
                            x.g[(rc + a) * w + b] = v;
                            x.g[b * w + rc + a] = v;
                        }
                        for b in 0..q_n {
                            x.g[(rc + a) * w + rc + b] = fam[b3 + a * q_n + b];
                        }
                    }
                }
                // C_f from Z'[X̃ y]: primary component d at RE column
                // d·n_primary + f, child c component d at
                // q·n_primary + (f·np + c)·q_n + d (`build_sparse_z`).
                for r in 0..w {
                    let col = if r < q {
                        r * n_prim + f
                    } else {
                        q * n_prim + (f * np) * q_n + (r - q)
                    };
                    let zr = &ws.ztxy[col * m..(col + 1) * m];
                    let row = &mut x.cf[r * m..(r + 1) * m];
                    row[..p].copy_from_slice(&zr[..p]);
                    row[p] = (0..m).map(|j| zr[j] * x.ev[j]).sum();
                }
                // A_f = I + Λ_fᵀG_fΛ_f, then its Crout.
                let lam = Lam {
                    blocks: &x.fblk,
                    vals: &x.flam,
                };
                grow(&mut x.tmp, w * w);
                lam.right(&x.g, w, w, &mut x.tmp);
                lam.left_t(&x.tmp, w, &mut x.fa);
                for j in 0..w {
                    x.fa[j * w + j] += 1.0;
                }
                if !crate::linalg::block_chol(&mut x.fa[..w * w], w) {
                    return false;
                }
                // Touched crossed blocks and G_fx from the family's merged A21
                // slabs: crossed component a's row, primary columns then each
                // child's.
                if g.k_crossed() > 0 {
                    let kf = g.k_family();
                    let blocks = &ws.a21_blk[ws.a21_off[f]..ws.a21_off[f + 1]];
                    for &bi in blocks {
                        let bi = bi as usize;
                        let ci = ws.cf_base.partition_point(|&b0| b0 <= bi) - 1;
                        let cf = &g.crossed[ci];
                        let start = g.extra_offsets[cf.decl] - kf + (bi - ws.cf_base[ci]) * cf.q;
                        x.lblk.push((x.xi.len(), cf.q, x.xlam_off[ci]));
                        x.xi.extend(start..start + cf.q);
                    }
                    let ef = x.xi.len();
                    grow(&mut x.gfx, w * ef);
                    for (idx, &(xoff, bq, _)) in x.lblk.iter().enumerate() {
                        let slab0 = ws.a21_slab_off[ws.a21_off[f] + idx];
                        for a in 0..bq {
                            for b in 0..q {
                                x.gfx[b * ef + xoff + a] = ws.pk_a21[slab0 + a * q + b];
                            }
                            for c in 0..np {
                                let base = slab0 + bq * q + c * bq * q_n;
                                for b in 0..q_n {
                                    x.gfx[(q + c * q_n + b) * ef + xoff + a] =
                                        ws.pk_a21[base + a * q_n + b];
                                }
                            }
                        }
                    }
                }
            }
        }
        true
    }

    /// `G_xx = Z_xᵀZ_x` into `x.s0x` and `C_x = Z_xᵀ[X̃ r]` (the residual
    /// column contracted) into `x.rx`.
    fn crossed_grams(&self, g: &LmmGroupings, m: usize, x: &mut ExactScratch) {
        let e = g.k_crossed();
        let kf = g.k_family();
        let p = m - 1;
        x.s0x[..e * e].fill(0.0);
        match self {
            Src::Dense { suff, .. } => {
                // Diagonal: the level counts; off it the cross-factor
                // co-occurrences (same-factor entries are structurally zero).
                for b in 0..e {
                    x.s0x[b * e + b] = suff.counts[kf + b];
                    for a in 0..b {
                        let v = suff.zx[(kf + a, b)];
                        x.s0x[a * e + b] = v;
                        x.s0x[b * e + a] = v;
                    }
                    let sc = suff.s.col(kf + b).try_as_col_major().unwrap().as_slice();
                    let row = &mut x.rx[b * m..(b + 1) * m];
                    row[..p].copy_from_slice(&sc[..p]);
                    row[p] = (0..m).map(|j| sc[j] * x.ev[j]).sum();
                }
            }
            Src::Sparse(ws) => {
                let pos = |bi: usize| {
                    let ci = ws.cf_base.partition_point(|&b0| b0 <= bi) - 1;
                    let cf = &g.crossed[ci];
                    (
                        g.extra_offsets[cf.decl] - kf + (bi - ws.cf_base[ci]) * cf.q,
                        cf.q,
                    )
                };
                for (pr, &off) in ws.a22_pairs.iter().zip(&ws.a22_off) {
                    let ((ph, qh), (pl, ql)) = (pos(pr[0] as usize), pos(pr[1] as usize));
                    for a in 0..qh {
                        for b in 0..ql {
                            let v = ws.pk_a22[off + a * ql + b];
                            x.s0x[(ph + a) * e + pl + b] = v;
                            x.s0x[(pl + b) * e + ph + a] = v;
                        }
                    }
                }
                for t in 0..e {
                    let zr = &ws.ztxy[(kf + t) * m..(kf + t + 1) * m];
                    let row = &mut x.rx[t * m..(t + 1) * m];
                    row[..p].copy_from_slice(&zr[..p]);
                    row[p] = (0..m).map(|j| zr[j] * x.ev[j]).sum();
                }
            }
        }
    }
}

/// Family inputs on Gram statistics: `P_f = L_f⁻¹Λ_fᵀG_f`, `B_f =
/// L_f⁻¹Λ_fᵀC_f`, `Q_f = L_f⁻¹Λ_fᵀG_fx`, then `S⁰_ff = G_f − P_fᵀP_f` (`x.s`),
/// `T_f = C_f − P_fᵀB_f` (`x.t`) and `S⁰_fx = G_fx − P_fᵀQ_f` (`x.sfx`).
fn family_s0(x: &mut ExactScratch, w: usize, m: usize, ef: usize) {
    for (v, len) in [
        (&mut x.pf, w * w),
        (&mut x.bf, w * m),
        (&mut x.qf, w * ef),
        (&mut x.s, w * w),
        (&mut x.t, w * m),
        (&mut x.sfx, w * ef),
    ] {
        grow(v, len);
    }
    // Forward-solved row by row against L_f (Λ_f lower, so (Λᵀ)_rk = Λ_kr for
    // k ≥ r).
    let (lam, fa) = (&x.lam, &x.fa);
    for (out, src, width) in [
        (&mut x.pf, &x.g, w),
        (&mut x.bf, &x.cf, m),
        (&mut x.qf, &x.gfx, ef),
    ] {
        for r in 0..w {
            for j in 0..width {
                let mut acc: f64 = (r..w).map(|k| lam[k * w + r] * src[k * width + j]).sum();
                for k in 0..r {
                    acc -= fa[r * w + k] * out[k * width + j];
                }
                out[r * width + j] = acc / fa[r * w + r];
            }
        }
    }
    for a in 0..w {
        for b in 0..w {
            let pp: f64 = (0..w).map(|k| x.pf[k * w + a] * x.pf[k * w + b]).sum();
            x.s[a * w + b] = x.g[a * w + b] - pp;
        }
        for j in 0..m {
            let pb: f64 = (0..w).map(|k| x.pf[k * w + a] * x.bf[k * m + j]).sum();
            x.t[a * m + j] = x.cf[a * m + j] - pb;
        }
        for j in 0..ef {
            let pq: f64 = (0..w).map(|k| x.pf[k * w + a] * x.qf[k * ef + j]).sum();
            x.sfx[a * ef + j] = x.gfx[a * ef + j] - pq;
        }
    }
}

/// The crossed tail's quantities once `x.s0x` holds `S⁰_xx` and `x.rx`
/// `Z_xᵀV_f⁻¹[X̃ r]`: `U_x`, `T_c⁻¹`, `T_c⁻¹U_xᵀ`, then `S_xx = S⁰_xx −
/// U_xT_c⁻¹U_xᵀ` in place of `S⁰_xx` with its Λ products, and `Z_xᵀV⁻¹[X̃ r]`
/// with its `N_x` and `w_x`. `false` when `T_c` is not positive definite.
fn crossed_prep(x: &mut ExactScratch, e: usize, m: usize) -> bool {
    let p = m - 1;
    for v in [&mut x.tinv, &mut x.ux, &mut x.ut, &mut x.sxl, &mut x.sxlx] {
        grow(v, e * e);
    }
    for v in [&mut x.rc, &mut x.lr] {
        grow(v, e * m);
    }
    grow(&mut x.nx, e * p);
    grow(&mut x.lnx, e * p);
    grow(&mut x.wx, e);
    grow(&mut x.lwx, e);
    let lam = Lam {
        blocks: &x.xblk,
        vals: &x.xlam,
    };
    lam.right(&x.s0x, e, e, &mut x.ux);
    lam.left_t(&x.ux, e, &mut x.sxlx);
    // T_c = I + Λ_xᵀS⁰_xxΛ_x, its lower factor (column-major) in `sxl`, and
    // T_c⁻¹ = L⁻ᵀL⁻¹ (faer's `llt::inverse`, lower triangle) mirrored.
    for i in 0..e {
        for j in 0..e {
            x.tinv[i * e + j] = x.sxlx[i * e + j] + if i == j { 1.0 } else { 0.0 };
        }
    }
    if !crate::linalg::chol_lower(&x.tinv, e, &mut x.sxl) {
        return false;
    }
    {
        use faer::linalg::cholesky::llt::inverse::{inverse, inverse_scratch};
        let mut mem = MemBuffer::new(inverse_scratch::<f64>(e, Par::Seq));
        inverse(
            MatMut::from_column_major_slice_mut(&mut x.tinv[..e * e], e, e),
            MatRef::from_column_major_slice(&x.sxl[..e * e], e, e),
            Par::Seq,
            MemStack::new(&mut mem),
        );
    }
    for j in 0..e {
        for i in j + 1..e {
            x.tinv[i * e + j] = x.tinv[j * e + i];
        }
    }
    // T_c⁻¹U_xᵀ, then S_xx = S⁰_xx − U_x·T_c⁻¹U_xᵀ (through `sxl`).
    gemm(&mut x.ut, &x.tinv, &x.ux, e, e, e, true);
    gemm(&mut x.sxl, &x.ux, &x.ut, e, e, e, false);
    for (s0, w) in x.s0x[..e * e].iter_mut().zip(&x.sxl[..e * e]) {
        *s0 -= w;
    }
    lam.right(&x.s0x, e, e, &mut x.sxl);
    lam.left_t(&x.sxl, e, &mut x.sxlx);
    // ZᵀV⁻¹[X̃ r] on the crossed rows: R⁰_x − U_x·T_c⁻¹Λ_xᵀR⁰_x.
    lam.left_t(&x.rx, m, &mut x.lr);
    gemm(&mut x.rc, &x.tinv, &x.lr, e, e, m, false);
    gemm(&mut x.lr, &x.ux, &x.rc, e, e, m, false);
    for (r, c) in x.rx[..e * m].iter_mut().zip(&x.lr[..e * m]) {
        *r -= c;
    }
    for t in 0..e {
        x.u[..p].copy_from_slice(&x.rx[t * m..t * m + p]);
        forward_solve(
            MatRef::from_column_major_slice(&x.lx[..p * p], p, p),
            p,
            &mut x.u[..p],
        );
        x.nx[t * p..(t + 1) * p].copy_from_slice(&x.u[..p]);
        x.wx[t] = x.rx[t * m + p];
    }
    lam.left_t(&x.nx, p, &mut x.lnx);
    lam.left_t(&x.wx, 1, &mut x.lwx);
    true
}

/// Adds one family's terms (inputs `x.s = S⁰_ff`, `x.t`, and with a crossed
/// tail of `e` columns `x.sfx = S⁰_fx` on the touched columns `x.xi`; `n` free
/// coordinates, sums at the stride `st`).
///
/// With a crossed tail the family's true blocks are `S_ff = S⁰_ff − W_ff` and
/// `S_fx = S⁰_fx − W_fx`, `W_ff = U_fT_c⁻¹U_fᵀ`, `W_fx = U_fT_c⁻¹U_xᵀ`, `U_f =
/// S⁰_fxΛ_x` (on the touched columns), and two families couple only through
/// `S_ff′ = −W_ff′`. The family's own pairs and its pairs with the crossed
/// levels read the true blocks; the pairs of two different families are the
/// pair sum over every two families of `W` (the low-rank remainder in
/// [`crossed_terms`], from `X_k = U_FᵀD_kU_F`, `Y_k`, `y_k`) less the same
/// family's: `τ` is quadratic in `S`, so `Σ_{f≠f′} τ(−W) = Σ_all τ(W) − Σ_f
/// τ(W_ff)`, and the N and w sums are linear, so they take `+Σ_f σ(W_ff) −
/// Σ_all σ(W)`.
fn family_terms(x: &mut ExactScratch, w: usize, p: usize, e: usize, n: usize, st: usize) {
    let m = p + 1;
    let ef = if e > 0 { x.xi.len() } else { 0 };
    let crossed = e > 0;
    for (v, len) in [
        (&mut x.nm, w * p),
        (&mut x.ln, w * p),
        (&mut x.wv, w),
        (&mut x.lw, w),
        (&mut x.sl, w * w),
        (&mut x.lsl, w * w),
    ] {
        grow(v, len);
    }
    let flam = Lam {
        blocks: &x.fblk,
        vals: &x.flam,
    };
    if crossed {
        for v in [&mut x.uf, &mut x.lsl_x] {
            grow(v, w * ef);
        }
        for v in [&mut x.sfxf, &mut x.sfxf_l, &mut x.l_sfxf, &mut x.l_sfxf_l] {
            grow(v, w * e);
        }
        grow(&mut x.tmp, w * ef.max(w));
        grow(&mut x.tinvf, ef * ef);
        for v in [&mut x.wff, &mut x.wsl, &mut x.wlsl] {
            grow(v, w * w);
        }
        let llam = Lam {
            blocks: &x.lblk,
            vals: &x.xlam,
        };
        // U_f = S⁰_fxΛ_x, then T_f −= U_f·(T_c⁻¹Λ_xᵀZ_xᵀV_f⁻¹[X̃ r]) on the
        // touched rows.
        llam.right(&x.sfx, w, ef, &mut x.uf);
        for r in 0..w {
            for j in 0..m {
                let v: f64 = (0..ef)
                    .map(|a| x.uf[r * ef + a] * x.rc[x.xi[a] * m + j])
                    .sum();
                x.t[r * m + j] -= v;
            }
        }
        // W_ff = U_f·T_c⁻¹·U_fᵀ (U_f lives on the touched columns), then
        // S_ff = S⁰_ff − W_ff in place.
        for a in 0..ef {
            for b in 0..ef {
                x.tinvf[a * ef + b] = x.tinv[x.xi[a] * e + x.xi[b]];
            }
        }
        for r in 0..w {
            for b in 0..ef {
                x.tmp[r * ef + b] = (0..ef)
                    .map(|a| x.uf[r * ef + a] * x.tinvf[a * ef + b])
                    .sum();
            }
        }
        for r in 0..w {
            for c in 0..w {
                x.wff[r * w + c] = (0..ef).map(|a| x.tmp[r * ef + a] * x.uf[c * ef + a]).sum();
            }
        }
        for (s, wf) in x.s[..w * w].iter_mut().zip(&x.wff[..w * w]) {
            *s -= wf;
        }
        // S_fx = S⁰_fx − U_f·T_c⁻¹U_xᵀ over every crossed column.
        for r in 0..w {
            let row = &mut x.sfxf[r * e..(r + 1) * e];
            row.fill(0.0);
            for (a, &ga) in x.xi.iter().enumerate() {
                row[ga] = x.sfx[r * ef + a];
            }
            for (a, &ga) in x.xi.iter().enumerate() {
                let u = x.uf[r * ef + a];
                for (v, t) in row.iter_mut().zip(&x.ut[ga * e..(ga + 1) * e]) {
                    *v -= u * t;
                }
            }
        }
        let xlam = Lam {
            blocks: &x.xblk,
            vals: &x.xlam,
        };
        xlam.right(&x.sfxf, w, e, &mut x.sfxf_l);
        flam.left_t(&x.sfxf, e, &mut x.l_sfxf);
        flam.left_t(&x.sfxf_l, e, &mut x.l_sfxf_l);
        flam.left_t(&x.uf, ef, &mut x.lsl_x);
        flam.right(&x.wff, w, w, &mut x.wsl);
        flam.left_t(&x.wsl, w, &mut x.wlsl);
    }
    // N_f = F_f·L_X⁻ᵀ row by row, and w_f, the contracted column of T_f.
    for r in 0..w {
        x.u[..p].copy_from_slice(&x.t[r * m..r * m + p]);
        forward_solve(
            MatRef::from_column_major_slice(&x.lx[..p * p], p, p),
            p,
            &mut x.u[..p],
        );
        x.nm[r * p..(r + 1) * p].copy_from_slice(&x.u[..p]);
        x.wv[r] = x.t[r * m + p];
    }
    flam.right(&x.s, w, w, &mut x.sl);
    flam.left_t(&x.sl, w, &mut x.lsl);
    flam.left_t(&x.nm, p, &mut x.ln);
    flam.left_t(&x.wv, 1, &mut x.lw);
    let fv = Vecs {
        n: &x.nm,
        ln: &x.ln,
        wv: &x.wv,
        lw: &x.lw,
    };
    let fl = Levels {
        pos: &x.fpos,
        start: &x.fpos_start,
    };
    let sp = Prods {
        s: &x.s,
        sl: &x.sl,
        ls: &[],
        lsl: &x.lsl,
    };
    let wp = crossed.then_some(Prods {
        s: &x.wff,
        sl: &x.wsl,
        ls: &[],
        lsl: &x.wlsl,
    });
    add_levels(&mut x.sums, w, p, &x.s, fv, fl, &x.key, n, st);
    add_pairs(&mut x.sums, true, w, w, p, sp, wp, fv, fv, fl, fl, n, st);
    if !crossed {
        return;
    }
    // The family's levels against every crossed level.
    let xv = Vecs {
        n: &x.nx,
        ln: &x.lnx,
        wv: &x.wx,
        lw: &x.lwx,
    };
    let xl = Levels {
        pos: &x.xpos,
        start: &x.xpos_start,
    };
    add_pairs(
        &mut x.sums,
        false,
        w,
        e,
        p,
        Prods {
            s: &x.sfxf,
            sl: &x.sfxf_l,
            ls: &x.l_sfxf,
            lsl: &x.l_sfxf_l,
        },
        None,
        fv,
        xv,
        fl,
        xl,
        n,
        st,
    );
    // X_k += U_fᵀD_kU_f, Y_k += U_fᵀD_kN_f, y_k += U_fᵀD_kw_f on the touched
    // columns: per level u_i = row a_i of U_f, v_i = row b_i of Λ_fᵀU_f.
    for k in 0..n {
        for &(ai, bi) in fl.of(k) {
            let (u, v) = (
                &x.uf[ai * ef..(ai + 1) * ef],
                &x.lsl_x[bi * ef..(bi + 1) * ef],
            );
            let xk = &mut x.xk[k * e * e..(k + 1) * e * e];
            for a in 0..ef {
                let ga = x.xi[a];
                for b in 0..ef {
                    xk[ga * e + x.xi[b]] += u[a] * v[b] + v[a] * u[b];
                }
                for t in 0..p {
                    x.yk[(k * e + ga) * p + t] += u[a] * x.ln[bi * p + t] + v[a] * x.nm[ai * p + t];
                }
                x.yvk[k * e + ga] += u[a] * x.lw[bi] + v[a] * x.wv[ai];
            }
        }
    }
}

/// The crossed tail's own terms after the families: its single levels and its
/// level pairs on `S_xx`, then the remainder of the pairs of two different
/// families, `tr(T_c⁻¹X_k·T_c⁻¹X_l)`, `−tr(Y_kᵀT_c⁻¹Y_l)` and
/// `−y_kᵀT_c⁻¹y_l` (`X_k`, `Y_k`, `y_k` are zero on a crossed coordinate).
fn crossed_terms(x: &mut ExactScratch, e: usize, p: usize, n: usize, st: usize) {
    let xv = Vecs {
        n: &x.nx,
        ln: &x.lnx,
        wv: &x.wx,
        lw: &x.lwx,
    };
    let xl = Levels {
        pos: &x.xpos,
        start: &x.xpos_start,
    };
    add_levels(&mut x.sums, e, p, &x.s0x, xv, xl, &x.key, n, st);
    add_pairs(
        &mut x.sums,
        true,
        e,
        e,
        p,
        Prods {
            s: &x.s0x,
            sl: &x.sxl,
            ls: &[],
            lsl: &x.sxlx,
        },
        None,
        xv,
        xv,
        xl,
        xl,
        n,
        st,
    );
    // T_c⁻¹X_k (through `sxl`, free now), T_c⁻¹Y_k, T_c⁻¹y_k.
    grow(&mut x.tyk, n * e * p);
    grow(&mut x.tyvk, n * e);
    let family = |k: usize| x.xcoord[k].is_none();
    for k in (0..n).filter(|&k| family(k)) {
        let xk = &mut x.xk[k * e * e..(k + 1) * e * e];
        gemm(&mut x.sxl, &x.tinv, xk, e, e, e, false);
        xk.copy_from_slice(&x.sxl[..e * e]);
        gemm(
            &mut x.tyk[k * e * p..(k + 1) * e * p],
            &x.tinv,
            &x.yk[k * e * p..(k + 1) * e * p],
            e,
            e,
            p,
            false,
        );
        gemm(
            &mut x.tyvk[k * e..(k + 1) * e],
            &x.tinv,
            &x.yvk[k * e..(k + 1) * e],
            e,
            e,
            1,
            false,
        );
    }
    for ka in (0..n).filter(|&k| family(k)) {
        for kb in (ka..n).filter(|&k| family(k)) {
            let (za, zb) = (
                &x.xk[ka * e * e..(ka + 1) * e * e],
                &x.xk[kb * e * e..(kb + 1) * e * e],
            );
            let mut tz = 0.0;
            for a in 0..e {
                for b in 0..e {
                    tz += za[a * e + b] * zb[b * e + a];
                }
            }
            let ty: f64 = x.yk[ka * e * p..(ka + 1) * e * p]
                .iter()
                .zip(&x.tyk[kb * e * p..(kb + 1) * e * p])
                .map(|(a, b)| a * b)
                .sum();
            let tyv: f64 = x.yvk[ka * e..(ka + 1) * e]
                .iter()
                .zip(&x.tyvk[kb * e..(kb + 1) * e])
                .map(|(a, b)| a * b)
                .sum();
            let at = ka * st + kb;
            x.sums.tr_ss[at] += tz;
            x.sums.tr_nsn[at] -= ty;
            x.sums.w_sw[at] -= tyv;
        }
    }
}
