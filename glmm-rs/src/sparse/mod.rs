//! Sparse-Z LMM solver. A second Gaussian-mixed path that factors
//! `Λ'Z'ZΛ + I` with a two-level blocked Cholesky specialized to this
//! path's structural class — primary(+nested) FAMILIES are block-diagonal
//! (levels never co-occur across families) and ALL crossed-extra columns form
//! one tail (dense for `e ≤ TAIL_SPARSE_MIN`, fill-reducing sparse above) —
//! lifting the `MAX_*` caps that bound the dense no-Z tail.
//! Mirrors `lmm::reml_deviance` (`src/lmm/kernel.rs`) one level down; validated
//! against it by the both-paths cross-check (`sparse` tests + `glmm/tests.rs`).
//!
//! The objective and its workspace only: `lmm::fit_lmm` drives them as one of
//! its two kernels, and `fit::lmm::accumulate_lmm_rows` is what builds
//! `SparseLmmWorkspace` per call. Test code lives in `tests.rs`. Which designs
//! come here is decided by `fit::classify_design`.
//
// `SymbolicCholesky` at module level serves `logdet_llt`, shared by the GLMM
// structured crossed-Schur path (`glmm/pirls/blocked_extras.rs`) and the LMM
// sparse-tail branch (`SparseTail`); the small-e LMM eval loop stays
// faer-sparse-free (blocked kernel with the dense tail).
use crate::lmm::{slope_value, Criterion, LmmGroupings};
use faer::dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholesky,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, MatRef, Par, Side, Spec};

mod schur;
use schur::sparse_schur_factor;

// FD-Hessian noise-margin measurement (`#[ignore]`d, not a gate). Lives under
// `sparse` because it drives the sparse-Z LMM fixtures alongside the GLMM ones.
#[cfg(all(test, feature = "formula"))]
mod fd_margin;
#[cfg(test)]
pub(crate) mod tests;

/// `log det(A) = 2·Σ_j log L[j,j]` for the LLT factor, reading the diagonal
/// straight out of `l_values` per symbolic arm (faer 0.24 exposes no diagonal
/// accessor on `LltRef`): simplicial via the CSC `col_ptr()`/`row_idx()`
/// layout, supernodal via each supernode's dense column-major panel, whose top
/// `ncols×ncols` block is L's triangular diagonal block. Handles whichever arm
/// the construction sites' `SupernodalThreshold::AUTO` picked
/// (`SparseLmmWorkspace::new`, and `symbolic_llt` in `glmm/workspace.rs`). Returns `+INFINITY` if any diagonal is
/// non-positive/non-finite (mirrors the dense evaluators' non-PD sentinel,
/// `src/lmm/kernel.rs`'s `reml_deviance`).
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
/// per eval — the failure `sparse_deviance_equals_dense_primary_slope_balanced_zero`
/// guards).
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
    /// Scratch for the per-family S22 syrk downdate `panel·panelᵀ`:
    /// col-major `e_f×e_f`, sized to `max_f e_f²` once, overwritten
    /// (`Accum::Replace`) then applied through `fam_dd`. Empty on the arm that
    /// needs no staging (`FamDowndate::Dense` at `fam_w == 1`, which folds the
    /// rank-1 update straight into the accumulator).
    pub(crate) dd_temp: Vec<f64>,
    /// Scratch for the per-family B̃₂ downdate `panel·U1_sub`:
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
    /// dense-Gram indexing (that indexing costs ~20% of the rung-7 profile).
    /// `pk_fam`: per family `(primary,primary) | per child (child,primary),
    /// (child,child)`; `pk_a21`: per family, per col-block (primary then
    /// children), that family's CO-OCCURRING crossed blocks in order;
    /// `pk_a22`: the co-occurring lower crossed block pairs (`a22_pairs`).
    /// Non-co-occurring blocks are structurally zero and neither packed nor
    /// folded — the zero-fill of the target supplies them. Block-granular
    /// "any entry ≠ 0" == structural co-occurrence: the intercept×intercept
    /// entry is the shared-row COUNT, which balanced-covariate cancellation
    /// (see `sparse_deviance_equals_dense_primary_slope_balanced_zero`) can
    /// never zero.
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
    /// elimination. A plain col-major `Vec` (the `src/lmm/kernel.rs` `fit.bt` pattern) so
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
    /// `S = C_xy − UᵀU`), overwritten per eval; `lmm::fit_lmm` reads β̂/SE off
    /// it at θ̂ exactly as it reads the dense kernel's `fit.factor`.
    pub(crate) factor: Mat<f64>,
    /// σ̂² at the last evaluation's θ, written by [`sparse_reml_deviance`] where
    /// it profiles the residual variance out; NaN whenever that evaluation
    /// returned the non-PD sentinel.
    pub(crate) sigma_sq: f64,
    /// Spherical conditional modes `û` at θ̂ in `build_sparse_z` column order,
    /// written by [`sparse_recover_u`]; empty when no recovery succeeded for
    /// the current fit.
    pub(crate) ranef_u: Vec<f64>,
    /// Scratch for the tail's dense `cholesky_in_place` (θ-independent size).
    pub(crate) tail_llt_mem: MemBuffer,
    /// Scratch for the augmented `m×m` `cholesky_in_place` (the `S = C_xy − UᵀU`
    /// factor, by the same call the dense tail uses).
    pub(crate) factor_llt_mem: MemBuffer,
    pub(crate) m: usize,
    pub(crate) p: usize,
    /// Row count N: `ν` is `N − p` under REML and `N` under ML (`Criterion::nu`,
    /// as in `reml_deviance`, `src/lmm/kernel.rs`).
    pub(crate) n: usize,
    /// `Some` only for the one post-fit evaluation that feeds
    /// [`sparse_recover_u`]; see [`SparseRecovery`].
    pub(crate) rec: Option<SparseRecovery>,
    /// The fixed-effect factor `U` `ztxy`/`cxy`'s `x` part is accumulated
    /// through (`x·U⁻¹`), the same one the dense kernel uses
    /// (`crate::lmm::DesignQr`).
    pub(crate) design_qr: crate::lmm::DesignQr,
    /// Pass-2 offsets, fixed with the pattern: each extra declaration's start
    /// in a row's z-entry list (`seg_off`), each crossed factor's first block
    /// in `lam_blocks` (`cf_base`), each `a21_blk` block's slab in `pk_a21`
    /// (`a21_slab_off`) and each `a22_pairs` pair's block in `pk_a22`
    /// (`a22_off`).
    pub(crate) seg_off: Vec<usize>,
    pub(crate) cf_base: Vec<usize>,
    pub(crate) a21_slab_off: Vec<usize>,
    pub(crate) a22_off: Vec<usize>,
    /// Pass-2 row scratch: one row of `x·U⁻¹` (`p`) and one row's z entries.
    pub(crate) xt: Vec<f64>,
    pub(crate) row: Vec<(usize, f64)>,
    /// The criterion the deviance evaluates, fixed at construction.
    pub(crate) criterion: Criterion,
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
/// at `lam_small[lam_off + r·q + c]` (r ≥ c).
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
    /// `sqrt_w`: `Some(√wᵢ)` per row (prior/precision weights, `Data::weights`
    /// square-rooted once by the caller) — threaded into both the z-emission
    /// (`for_each_z_entry`) and pass 2's raw x/y reads below, so every packed
    /// Gram ends up carrying exactly `wᵢ` per row (see `for_each_z_entry`'s
    /// doc). `None` is the unit-weight fast path (no per-row multiply).
    /// `criterion`: the one the deviance evaluates, frozen with the workspace.
    /// `z`: the `n × s` random-effect-only columns a slope index `≥ p` reads.
    #[allow(clippy::too_many_arguments)] // marshals (g, x, z, cluster_ids, extra_ids, y, n, p, sqrt_w, criterion)
    pub(crate) fn new(
        g: &LmmGroupings,
        x: MatRef<f64>,
        z: MatRef<f64>,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        y: &[f64],
        n: usize,
        p: usize,
        sqrt_w: Option<&[f64]>,
        criterion: Criterion,
    ) -> Result<Self, crate::FitError> {
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
        let force_sparse = tests::FORCE_SPARSE_TAIL.with(|c| c.get());
        #[cfg(not(test))]
        let force_sparse = false;
        let sparse_tail = e > 0 && (e > TAIL_SPARSE_MIN || force_sparse);
        let tail_llt_mem = MemBuffer::new(cholesky_in_place_scratch::<f64>(
            if sparse_tail { 0 } else { e },
            Par::Seq,
            Spec::default(),
        ));

        // ---- Pass 1 (pattern): family cliques + row-co-occurring crossed
        // block pairs, by two O(n) row sweeps rather than a dense k×k `Z'Z`
        // (2.65 GB on the nest3 grid cell) or an O(e²) all-pairs probe: this
        // one discovers structure, the second scatters values straight into
        // the packed streams. Block-granular co-occurrence IS row-sharing (an
        // intercept×intercept entry is the shared-row COUNT, which balanced
        // covariates can never cancel), so a21_blk/a22_pairs equal a dense
        // scan's and the packed streams are bit-identical.
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
        // Iteration order is col block asc, then row block asc — the kernel's
        // `cur` replay over pk_a22 depends on it.
        a22_pairs.sort_unstable_by_key(|pr| (pr[1], pr[0]));

        // Packed-stream allocation + scatter offsets, sized from the pattern.
        let fam_len = q_p * q_p + np * (q_n * q_p + q_n * q_n);
        let pk_fam = vec![0.0f64; n_prim * fam_len];
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
        let pk_a21 = vec![0.0f64; pk_a21_len];
        // Start of each pair's block in pk_a22 (parallel to a22_pairs).
        let mut a22_off: Vec<usize> = Vec::with_capacity(a22_pairs.len());
        let mut pk_a22_len = 0usize;
        for pr in &a22_pairs {
            a22_off.push(pk_a22_len);
            pk_a22_len += lam_blocks[pr[0] as usize].q * lam_blocks[pr[1] as usize].q;
        }
        let pk_a22 = vec![0.0f64; pk_a22_len];

        let tail = if sparse_tail {
            Some(build_sparse_tail(
                &lam_blocks,
                &fam_crossed,
                &a22_pairs,
                kf,
                e,
                m,
                fam_w,
            )?)
        } else {
            None
        };

        let mut ws = Self {
            g: g.clone(),
            ztxy: vec![0.0f64; g.k_total * m],
            cxy: Mat::zeros(m, m),
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
            sigma_sq: f64::NAN,
            ranef_u: Vec::new(),
            m,
            p,
            n,
            rec: None,
            design_qr: crate::lmm::DesignQr::identity(p),
            seg_off,
            cf_base,
            a21_slab_off,
            a22_off,
            xt: vec![0.0f64; p],
            row: Vec::with_capacity(q_p + g.extra_q.iter().sum::<usize>()),
            criterion,
        };
        ws.fill_values(x, z, cluster_ids, extra_ids, y, sqrt_w);
        Ok(ws)
    }

    /// Pass 2 of [`Self::new`], into the buffers the pattern sized: `Z'[X y]`,
    /// `[X y]'[X y]`, the fixed-effect factor `U` and every packed raw-Gram
    /// block, from the rows. Every value buffer is cleared first and the
    /// per-fit state is reset, so a workspace refilled for a draw on the same
    /// rows holds exactly what a new one would, with no allocation.
    pub(crate) fn fill_values(
        &mut self,
        x: MatRef<f64>,
        z: MatRef<f64>,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        y: &[f64],
        sqrt_w: Option<&[f64]>,
    ) {
        let SparseLmmWorkspace {
            g,
            ztxy,
            cxy,
            pk_fam,
            pk_a21,
            pk_a22,
            a21_blk,
            a21_off,
            a22_pairs,
            design_qr,
            seg_off,
            cf_base,
            a21_slab_off,
            a22_off,
            xt,
            row,
            sigma_sq,
            ranef_u,
            rec,
            m,
            p,
            n,
            ..
        } = self;
        let (m, p, n) = (*m, *p, *n);
        let q_p = g.primary_q;
        let np = g.nested_per_parent;
        let q_n = g.nested.map(|nf| nf.q).unwrap_or(0);
        let nested_decl = g.nested.map(|nf| nf.decl);
        let fam_len = q_p * q_p + np * (q_n * q_p + q_n * q_n);
        // `a22_pairs` is sorted by (col block, row block), unique.
        let a22_at = |bh: u32, bl: u32| -> usize {
            let at = a22_pairs
                .binary_search_by_key(&(bl, bh), |pr| (pr[1], pr[0]))
                .expect("row's crossed block pair is in the A22 pattern");
            a22_off[at]
        };
        // ---- Pass 2 (values): Z'[X y], [X y]'[X y], and every packed raw-Gram
        // block, scattered per row over Z's ≤ q_p + Σq_e nonzeros — O(n·(Σq)²)
        // total, same accumulation the dense Gram + `pack` copy produced,
        // block-pair-keyed instead of k×k-staged. NOT the shared
        // `add_rows_multi` accumulator: that path packs per-row level ids into
        // a fixed `[usize; 1 + MAX_EXTRA_GROUPINGS]` stack array (`src/lmm/kernel.rs`)
        // and would index out of bounds for over-envelope-by-count designs —
        // the sparse route must stay cap-free.
        ztxy.fill(0.0);
        cxy.fill(0.0);
        pk_fam.fill(0.0);
        pk_a21.fill(0.0);
        pk_a22.fill(0.0);
        // The fixed-effect design enters every Gram below as `x·U⁻¹`; the
        // z entries (slopes included) stay on the raw `x`.
        design_qr.set_from_rows(x.subrows(0, n), |i| sqrt_w.map_or(1.0, |w| w[i]), xt);
        for i in 0..n {
            // One √wᵢ per row factor: `row`'s z entries already carry it
            // (`for_each_z_entry`), so the raw x/y reads below must carry a
            // matching one — every ztxy/cxy product then carries wᵢ exactly
            // once (see `for_each_z_entry`'s doc and `SparseLmmWorkspace::new`'s
            // top-level doc).
            let sw = sqrt_w.map_or(1.0, |w| w[i]);
            row.clear();
            for_each_z_entry(g, x, z, p, cluster_ids, extra_ids, i, sqrt_w, |col, v| {
                row.push((col, v))
            });
            for (j, xj) in xt.iter_mut().enumerate() {
                *xj = x[(i, j)];
            }
            design_qr.solve_row(xt);
            for &(ca, va) in row.iter() {
                for j in 0..p {
                    ztxy[ca * m + j] += va * (sw * xt[j]);
                }
                ztxy[ca * m + p] += va * (sw * y[i]);
            }
            // Lower triangle only, mirrored after the row loop.
            let wy = sw * y[i];
            for b in 0..p {
                let wb = sw * xt[b];
                let col = cxy.col_as_slice_mut(b);
                for a in b..p {
                    col[a] += (sw * xt[a]) * wb;
                }
                col[p] += wy * wb;
            }
            cxy[(p, p)] += wy * wy;
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
            let fam_list = &a21_blk[a21_off[f]..a21_off[f + 1]];
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
                let off = a22_at(bi, bi);
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
                    let off = a22_at(bh, bl);
                    for a in 0..qh {
                        for b in 0..ql {
                            pk_a22[off + a * ql + b] += vh[a].1 * vl[b].1;
                        }
                    }
                }
            }
        }
        for b in 0..m {
            for a in b + 1..m {
                cxy[(b, a)] = cxy[(a, b)];
            }
        }
        *sigma_sq = f64::NAN;
        ranef_u.clear();
        *rec = None;
    }

    /// Arm [`SparseRecovery`] for the NEXT evaluation. Sized here, once, from
    /// the shapes the workspace already fixes; the panels are ragged, so the
    /// offsets are computed from each family's co-occurring crossed blocks the
    /// same way `sparse_schur_factor` sums `e_f`.
    pub(crate) fn arm_recovery(&mut self) {
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
/// value per eval. A pattern the symbolic factorization cannot hold (an index
/// overflow or an allocation failure, both driven by the crossed level count)
/// is [`pattern_too_large`].
fn build_sparse_tail(
    lam_blocks: &[LamBlock],
    fam_crossed: &[Vec<u32>],
    a22_pairs: &[[u32; 2]],
    kf: usize,
    e: usize,
    m: usize,
    fam_w: usize,
) -> Result<SparseTail, crate::FitError> {
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
    .map_err(|err| pattern_too_large(e, err))?;
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
    let dense_route = match tests::FORCE_DD_ROUTE.with(|c| c.get()) {
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
    let (l_values, fac_mem, solve_mem) = numeric_buffers(&symbolic, e, m)?;
    Ok(SparseTail {
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
    })
}

/// The refusal of a random-effect pattern faer's symbolic sparse Cholesky
/// cannot factor (`err`: an index overflow or an allocation failure), with
/// `cols` the random-effect columns it factors (the crossed levels times
/// their widths on the LMM tail and the GLMM's structured layout), as
/// [`crate::FitError::Envelope`]. Shared by the LMM tail and the GLMM's
/// packed and structured factors.
pub(crate) fn pattern_too_large(cols: usize, err: faer::sparse::FaerError) -> crate::FitError {
    crate::FitError::Envelope(
        format!(
            "the random-effect pattern ({cols} random-effect columns) is too large for the \
             sparse factorization ({err})"
        )
        .into(),
    )
}

/// The numeric factor's values and the factor and solve scratch of `symbolic`
/// for `nrhs` right-hand sides. A supernodal factor's size is known only after
/// the symbolic step, so a refused allocation here is [`pattern_too_large`]
/// too.
pub(crate) fn numeric_buffers(
    symbolic: &SymbolicCholesky<usize>,
    cols: usize,
    nrhs: usize,
) -> Result<(Vec<f64>, MemBuffer, MemBuffer), crate::FitError> {
    let too_large = |err: faer::sparse::FaerError| pattern_too_large(cols, err);
    let mut l_values = Vec::new();
    l_values
        .try_reserve_exact(symbolic.len_val())
        .map_err(|e| too_large(e.into()))?;
    l_values.resize(symbolic.len_val(), 0.0f64);
    let fac_mem = MemBuffer::try_new(
        symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default()),
    )
    .map_err(|e| too_large(e.into()))?;
    let solve_mem = MemBuffer::try_new(symbolic.solve_in_place_scratch::<f64>(nrhs, Par::Seq))
        .map_err(|e| too_large(e.into()))?;
    Ok((l_values, fac_mem, solve_mem))
}

/// Visit row `i`'s nonzero `Z` entries as `(RE column, value)` pairs — the
/// single owner of the per-row scatter in the sparse path's RE-column order
/// `[primary | nested children | crossed]`, matching `LmmSuffStats`'s column
/// indexing (the comparison target). NB: this deliberately diverges from the
/// blocked GLMM RE-column layout (`glmm/workspace.rs`), whose primary block is
/// level-major (`lvl·q_p + c`) — that is a separate, self-consistent fit path
/// and is NOT the layout to mirror here. Intercept RE columns carry `1.0`;
/// slope RE columns the covariate value from `x`.
/// Primary block is slope-major: intercept for level f at column f, slope k at
/// column `(k+1)·n_primary + f` — the `d·n_primary + f` convention owned by
/// `LmmGroupings::from_cluster_spec_ext` and read by
/// `add_rows_multi`/`primary_gram`.
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
/// `add_rows_multi`'s `zw`-per-side scheme, `src/lmm/kernel.rs`). A slope
/// value comes from `x` (`n × p`) or the random-effect-only columns `z`
/// ([`crate::lmm::slope_value`]).
#[inline]
#[allow(clippy::too_many_arguments)] // marshals (g, x, z, p, ids, i, sqrt_w, emit)
fn for_each_z_entry(
    g: &LmmGroupings,
    x: MatRef<f64>,
    z: MatRef<f64>,
    p: usize,
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
        emit((k + 1) * g.n_primary + f, sw * slope_value(x, z, p, i, col));
    }
    // Extra groupings: intercept at off, slope c at off+1+c.
    for (e, ids_e) in extra_ids.iter().enumerate() {
        let q_g = g.extra_q[e];
        let off = g.extra_offsets[e] + ids_e[i] as usize * q_g;
        emit(off, sw);
        for (c, &col) in g.extra_slope_cols[e].iter().enumerate() {
            emit(off + 1 + c, sw * slope_value(x, z, p, i, col));
        }
    }
}

/// Refill `lam_small` with the per-grouping `q×q` Λ factors at θ, once per
/// eval — the only Λ materialization on this path (the dense `k×k` Λ is never
/// built). Offsets mirror the `lam_blocks` construction in `new` — change
/// together (`[primary | nested | crossed]`, each block `q×q` row-major
/// lower-tri). θ is sliced per grouping: primary from the vech prefix, each
/// extra from its `vech_start`. Shared by the Gaussian blocked kernel
/// (`sparse_schur_factor`) and the GLMM packed-row layout
/// (`glmm::PackedScratch`, whose `lam_off_decl` maps declarations into the
/// same layout).
pub(crate) fn fill_lambda_small(theta: &[f64], g: &LmmGroupings, lam_small: &mut [f64]) {
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
/// bounds-checked reads) costs 49% of the rung-7 profile; the packed replay
/// is sequential. `lo_r`/`lo_c` are the two blocks' `lam_small` offsets. Requires
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

/// Profiled deviance on the sparse path under `ws.criterion`. Numerically equal
/// to `lmm::reml_deviance` at the same θ and criterion (same argmin, same dropped
/// additive constants; the both-paths cross-check is the guarantee). Steps:
///   1. blocked Cholesky of A = Λ'Z'ZΛ + I → log|L_ZZ|² (`sparse_schur_factor`);
///   2. forward-solve U = L⁻¹B, B = Λ'Z'[X y];
///   3. S = C_xy − UᵀU (dense m×m, C_xy = ws.cxy = [X y]'[X y]), Cholesky
///      (`cholesky_in_place`) → L_XX;
///   4. REML: σ̂² = L_XX[p,p]² / (N−p), deviance = log|L_ZZ|² + log|L_XX|² + (N−p)·ln σ̂²;
///      ML: σ̂² = L_XX[p,p]² / N, deviance = log|L_ZZ|² + N·ln σ̂².
///
/// Normalization transcribed from `reml_deviance` (`src/lmm/kernel.rs`): df = N−p
/// (REML) or N (ML), `σ̂² = L[p,p]²/df`, `log|L_XX|² = 2Σ_{j<p} ln L[j,j]` (its
/// pivots checked on both criteria); drops the same additive constants (2π, …).
/// Returns INFINITY on any non-PD factor / non-finite σ̂² (mirrors the dense
/// diagonal guards, same function).
pub(crate) fn sparse_reml_deviance(theta: &[f64], ws: &mut SparseLmmWorkspace) -> f64 {
    let p = ws.p;
    ws.sigma_sq = f64::NAN;
    // Same shape guard as the dense twin (`reml_deviance`, `src/lmm/kernel.rs`):
    // at `n <= p` the REML df is not positive, so neither criterion fits there,
    // and a width-0 fixed block has no `L_XX` to read the deviance off.
    if ws.n <= p || p == 0 {
        return f64::INFINITY;
    }
    let log_lzz_sq = match sparse_schur_factor(theta, ws) {
        Some(v) => v,
        None => return f64::INFINITY, // non-PD blocked or Schur factor at this θ
    };

    // Deviance off the dense augmented factor's diagonal (mirror
    // `kernel::reml_deviance`).
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
    let criterion = ws.criterion;
    let df = criterion.nu(ws.n, p) as f64;
    let r_sq = lyy * lyy;
    let sigma_sq = r_sq / df;
    let r_sq_floor = crate::lmm::exact_fit_floor(ws.g.k_total, p + 1, ws.cxy[(p, p)]);
    if !(sigma_sq.is_finite() && sigma_sq > 0.0 && r_sq > r_sq_floor) {
        return f64::INFINITY;
    }
    ws.sigma_sq = sigma_sq;
    match criterion {
        Criterion::Reml => log_lzz_sq + log_lxx_sq + df * sigma_sq.ln(),
        Criterion::Ml => log_lzz_sq + df * sigma_sq.ln(),
    }
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
pub(crate) fn sparse_recover_u(ws: &SparseLmmWorkspace, beta: &[f64]) -> Option<Vec<f64>> {
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
                if (0..e).any(|t| !(ws.s22[(t, t)].is_finite() && ws.s22[(t, t)] > 0.0)) {
                    return None;
                }
                crate::lmm::back_solve(ws.s22.as_ref(), e, &mut u2);
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
