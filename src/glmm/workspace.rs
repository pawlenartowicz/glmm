use bobyqa::{Bobyqa, Config};
use faer::dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholesky,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, MatRef, Par, Side, Spec};

use crate::lmm::{LmmGroupings, GLMM_RHO_END, RHO_BEGIN, THETA_TRUTH_FLOOR};
use crate::scalar::Scalar;

use super::BETA_BOX;

/// Outer search over the variance components, fixed per shape at construction.
/// `Joint`: one `[θ | β]` BOBYQA on the Laplace deviance, β held fixed inside PIRLS —
/// the A/B reference every other route is checked against. `PqlThenJoint`: a θ-only
/// BOBYQA on the PQL β-profile as a warm start, then `Joint` from there; the warm start
/// never gates convergence. `ExactProfile`: a θ-only BOBYQA on the exact Laplace
/// β-profile — the search itself; its status alone decides `converged`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OuterSearch {
    Joint,
    PqlThenJoint,
    ExactProfile,
}

/// Which `A`-layout a design takes. `Packed` is every design `fit::classify_design`
/// sends to `Solver::Sparse` (over the dense envelope, slopes on an extra grouping,
/// more than `MAX_CROSSED_LEVELS` crossed levels) plus every extras design whose
/// core block is too wide for the structured route.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum GlmmLayout {
    Blocked,
    Structured,
    Packed,
}

impl GlmmLayout {
    pub(crate) fn for_design(model: &crate::ModelSpec, g: &LmmGroupings) -> Self {
        match crate::fit::classify_design(model, 1) {
            crate::fit::Solver::Sparse => GlmmLayout::Packed,
            crate::fit::Solver::NoZ if g.extra_offsets.is_empty() => GlmmLayout::Blocked,
            crate::fit::Solver::NoZ if g.structured_extras_eligible() => GlmmLayout::Structured,
            crate::fit::Solver::NoZ => GlmmLayout::Packed,
        }
    }
}

/// Row- and RE-sized PIRLS scratch every route writes: the linear predictor,
/// mean and working weight per row, the packed `M = ZΛ` rows of the blocked
/// path, the RE mode and its backtrack twin, the per-cluster Fisher blocks,
/// the RHS and the AGQ per-cluster scratch. Lengths, with `rows` the row
/// capacity, `k` the RE dimension, `q_p` the primary RE width, `s` the primary
/// cluster count: `eta`/`prob`/`w`/`eta_fixed`/`mu` `rows`, `m_buf` `rows·q_p`,
/// `lam` `q_p²`, `u` `k.max(1)`, `u_prev` `k.max(1)`, `a_rhs` `k.max(1)`,
/// `a_blocks` `(s·q_p²).max(1)`, `agq_scratch` `agq_len(s, q_p, nagq)`. `T = f64` in the workspace;
/// `Dual`/`HyperDual` in the derivative passes' twin.
pub(crate) struct PirlsScratch<T: Scalar> {
    /// PIRLS linear predictor η, length max_n.
    pub eta: Vec<T>,
    /// PIRLS fitted mean μ, length max_n.
    pub prob: Vec<T>,
    /// PIRLS working weights W, length max_n.
    pub w: Vec<T>,
    /// Σ_j x·β, hoisted out of the PIRLS iteration (β fixed within a solve)
    pub eta_fixed: Vec<T>,
    /// (Mu)ᵢ per row, filled by the layout's own row pass. The packed layout
    /// leaves it at (Mu)ᵢ and reads it there; the structured layout overwrites
    /// it in place with the IRLS residual `W·Mu + (y−prob)` before the RHS
    /// scatter. The blocked layout does not use it.
    pub mu: Vec<T>,
    /// n×q_p row-major mᵢ = Λ_p'·zᵢ (blocked path) — filled once per PIRLS solve
    pub m_buf: Vec<T>,
    /// q_p × q_p primary Λ_p scratch (row-major)
    pub lam: Vec<T>,
    /// Current RE-mode iterate û, length k.
    pub u: Vec<T>,
    /// previous accepted PIRLS iterate, step-halving backtrack buffer (len k.max(1))
    pub u_prev: Vec<T>,
    /// length k
    pub a_rhs: Vec<T>,
    /// s · q_p² packed per-cluster q_p×q_p blocks (no-extras path; Σ wᵢmᵢmᵢ'+I then Crout L)
    pub a_blocks: Vec<T>,
    /// AGQ per-cluster scratch; unused on the Laplace path. Shape-dependent:
    /// scalar (`q_p==1`) is `4·n_primary` (center loglik | node u_cj | per-node
    /// loglik | running log-sum); vector (`q_p∈2..=3`) is `2·n_primary + k^q·(q+1)`
    /// (center loglik | running log-sum | product-grid node table). See the
    /// sizing at construction.
    pub agq_scratch: Vec<T>,
}

impl<T: Scalar> PirlsScratch<T> {
    /// Every length from the shape terms `GlmmWorkspace::from_groupings` and
    /// the derivative scratch both derive: `rows` the row capacity, `k` the RE
    /// dimension, `q_p` the primary width, `s` the primary cluster count.
    pub(crate) fn for_shape(rows: usize, k: usize, q_p: usize, s: usize, nagq: u8) -> Self {
        Self {
            eta: vec![T::ZERO; rows],
            prob: vec![T::ZERO; rows],
            w: vec![T::ZERO; rows],
            eta_fixed: vec![T::ZERO; rows],
            mu: vec![T::ZERO; rows],
            m_buf: vec![T::ZERO; rows * q_p],
            lam: vec![T::ZERO; q_p * q_p],
            u: vec![T::ZERO; k.max(1)],
            u_prev: vec![T::ZERO; k.max(1)],
            a_rhs: vec![T::ZERO; k.max(1)],
            a_blocks: vec![T::ZERO; (q_p * q_p * s).max(1)],
            agq_scratch: vec![T::ZERO; super::derivative::agq_len(s, q_p, nagq)],
        }
    }
}

/// θ-dependent values of the structured crossed/nested route, left FACTORED
/// after a converged solve for the Schur fill: core blocks `(q_core²·s).max(1)`,
/// coupling `(q_core·s·e).max(1)`, Schur `(e²).max(1)`, packed core `M`
/// `(rows·q_core).max(1)`, crossed values `(rows·G_cap).max(1)`. Same two
/// instantiations as `PirlsScratch`.
pub(crate) struct StructuredScratch<T: Scalar> {
    /// s · q_core² packed per-cluster core blocks (D_f+I then Crout L)
    pub core_blocks: Vec<T>,
    /// s · q_core · e core↔crossed coupling C_f (row-major per cluster)
    pub coupling: Vec<T>,
    /// e × e Schur S = (E+I) − Σ_f C_f'A_f⁻¹C_f (row-major; Crout L in place)
    pub schur_blk: Vec<T>,
    // Packed M = ZΛ nonzeros for the STRUCTURED path — filled once per deviance
    // eval by `build_packed_m`, then read by the structured PIRLS passes and
    // `structured_schur_fill`. `q_core = primary_q + nested_per_parent`, `G_cap =
    // MAX_EXTRA_GROUPINGS`. Sized once at construction — no per-solve alloc.
    /// max_n · q_core row-major; [i·q_core+local] = M[(i, core_col(f,local))]
    pub m_core_buf: Vec<T>,
    /// max_n · G_cap row-major; nonzero M value (z·θ) per crossed grouping
    pub cross_val: Vec<T>,
}

impl<T: Scalar> StructuredScratch<T> {
    /// `q_core = primary_q + nested_per_parent`, `e = k_crossed()`,
    /// `G_cap = MAX_EXTRA_GROUPINGS`; every length `.max(1)` so the
    /// no-extras path holds the minimum and the first call allocates nothing.
    pub(crate) fn for_shape(rows: usize, s: usize, q_core: usize, e: usize) -> Self {
        Self {
            core_blocks: vec![T::ZERO; (q_core * q_core * s).max(1)],
            coupling: vec![T::ZERO; (q_core * s * e).max(1)],
            schur_blk: vec![T::ZERO; (e * e).max(1)],
            m_core_buf: vec![T::ZERO; (rows * q_core).max(1)],
            cross_val: vec![T::ZERO; (rows * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
        }
    }
}

/// θ-independent index pattern of the structured route — which crossed column
/// each row touches and the per-cluster CSR of coupling columns — plus the
/// cached sparse Schur factor and the test-only dense-Schur switch. `f64`
/// workspace only: the dual passes read this same pattern rather than carrying
/// a twin.
pub(crate) struct StructuredPattern {
    /// max_n · G_cap row-major; its crossed-block-local index b (0..e)
    pub cross_col: Vec<u32>,
    /// max_n; #crossed nonzeros for row i (≤ G ≤ MAX_EXTRA_GROUPINGS < 256)
    pub n_cross: Vec<u8>,
    // Per-cluster CSR of C_f's nonzero crossed columns (cluster f's slice is
    // coup_cols[coup_ptr[f]..coup_ptr[f+1]], sorted + deduped). Rebuilt on
    // pinning-mask transitions (see `coup_mask`) from cross_col/n_cross;
    // structured_factor's Schur build walks it instead of all e columns.
    /// ≤ max_n · G_cap entries before dedup
    pub coup_cols: Vec<u32>,
    /// n_primary + 1 offsets
    pub coup_ptr: Vec<u32>,
    /// θ-pinning mask (bit g = crossed grouping g has θ == 0.0) the current
    /// coup_cols/coup_ptr CSR was built for; `None` ⇒ not built this fit. The
    /// CSR pattern depends on the design AND this mask (build_packed_m drops
    /// pinned groupings), so the structured deviance rebuilds only on mask
    /// transitions. Reset to None at each fit_glmm start (mirrors u_seed).
    pub coup_mask: Option<u32>,
    /// Cached sparse factor of the crossed Schur `S`. `Some` only on the
    /// structured crossed path with `e > 0`; built per fit by `StructuredSchur::new`
    /// per fit by `StructuredSchur::new`. `None` ⇒ the packed/blocked/e=0 paths,
    /// which never touch it.
    pub(crate) structured_schur: Option<StructuredSchur>,
    /// Test-only: force the dense `glmm_block_chol` Schur factor instead of the
    /// cached sparse one, so the both-paths cross-check runs both at one θ. Always
    /// `false` in production (the sparse factor is the only path). The objective
    /// and the pinned-γ̂ re-evaluation both read this same pattern, so the
    /// cross-check compares whole fits, not a single deviance evaluation.
    pub(crate) force_dense_schur: bool,
}

/// Packed-row layout scratch (`GlmmLayout::Packed`): the `M = ZΛ` nonzeros in
/// fixed-width rows, the per-grouping `Λ` factors they fold, and the dense
/// `k×k` `A = M'WM + I` with its factor target.
///
/// Every row loads exactly one level of every grouping, so the row width is
/// FIXED at `width = q_p + Σ q_g` and no CSR offsets are needed: row `i`'s
/// nonzeros are `[i·width, (i+1)·width)`, `m_cols[i·width + t]` is the RE
/// column row `i`'s `t`-th nonzero touches and `m_vals[i·width + t]` its value
/// `(ZΛ)_{i,col}`. Zero-length on the blocked and structured layouts, which
/// never read it — a missed read must panic on the bounds check, not silently
/// return a zero.
pub(crate) struct PackedScratch {
    /// `lam_small` offsets per extra DECLARATION (parallel to
    /// `LmmGroupings::extra_offsets`); the primary block is at 0. Maps
    /// `fill_lambda_small`'s `[primary | nested | crossed]` layout back to
    /// declaration order.
    pub lam_off_decl: Vec<usize>,
    /// Concatenated per-grouping `q×q` Λ factors (row-major lower-tri),
    /// refilled once per θ eval by `fill_lambda_small`.
    pub lam_small: Vec<f64>,
    /// Nonzeros per packed row, `q_p + Σ q_g`.
    pub width: usize,
    /// len `max_n·width`. Design-fixed (filled by [`fill_packed_cols`]); column
    /// order is slope-major primary (component `c` at `c·n_primary + f`) and
    /// level-major extras (`extra_offsets[e] + level·q_g + c`). Initialized to
    /// `u32::MAX`, so a solve that runs before the fill panics on the bounds
    /// check instead of scattering every row's mass into column 0.
    pub m_cols: Vec<u32>,
    /// len `max_n·width`. The Λ-folded `z` entries, refilled per θ eval by
    /// `fill_m_vals`.
    pub m_vals: Vec<f64>,
    /// k × k `A = M'WM + I`, full symmetric (the per-row scatter writes both
    /// triangles). Left holding the FINAL iterate's raw `A` after a converged
    /// solve, which `packed_schur_fill` (se.rs) re-factors — so the PIRLS
    /// Cholesky must run on `a_chol`, never on this field in place.
    pub a: Mat<f64>,
    /// Copy-then-factor target for `a`'s Cholesky (k×k): the solve copies `a`'s
    /// lower triangle in here (mirroring `.llt(Side::Lower)`'s internal copy)
    /// and runs `cholesky_in_place` on THIS buffer.
    pub a_chol: Mat<f64>,
    /// Scratch for `a_chol`'s in-place `cholesky_in_place` (k×k, θ-independent
    /// size) — avoids a per-PIRLS-iteration `.llt(Side::Lower)` allocation.
    pub a_llt_mem: MemBuffer,
}

impl PackedScratch {
    /// Packed-row buffers for a design with `rows` row capacity and `k` RE
    /// columns. `m_cols` is design-fixed but level-id dependent, so it is
    /// allocated here and filled by [`fill_packed_cols`].
    pub(crate) fn for_shape(g: &LmmGroupings, rows: usize, k: usize) -> Self {
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
        PackedScratch {
            lam_off_decl,
            lam_small: vec![0.0; lam_len.max(1)],
            width,
            m_cols: vec![u32::MAX; rows * width],
            m_vals: vec![0.0; rows * width],
            a: Mat::zeros(k.max(1), k.max(1)),
            a_chol: Mat::zeros(k.max(1), k.max(1)),
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                k.max(1),
                Par::Seq,
                Spec::default(),
            )),
        }
    }

    /// Zero-length buffers for the blocked and structured layouts, which never
    /// read them.
    pub(crate) fn unused() -> Self {
        PackedScratch {
            lam_off_decl: vec![],
            lam_small: vec![],
            width: 0,
            m_cols: vec![],
            m_vals: vec![],
            a: Mat::zeros(0, 0),
            a_chol: Mat::zeros(0, 0),
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                1,
                Par::Seq,
                Spec::default(),
            )),
        }
    }
}

/// β-border scratch shared by the Profile-mode PIRLS step and the three
/// Schur fillers in `se.rs`: `X'WX` (p×p), `X'WM` (p×k), `A⁻¹M'WX` (k×p), the
/// Schur complement (p×p) with its factor memory, and the β backtrack buffer.
pub(crate) struct BorderScratch {
    /// p × p
    pub xtwx: Mat<f64>,
    /// p × k
    pub xtwm: Mat<f64>,
    /// k × p  = A⁻¹ M'WX
    pub ainv_mtwx: Mat<f64>,
    /// p × p  X'WX − X'WM A⁻¹ M'WX
    pub schur: Mat<f64>,
    /// Scratch for `schur`'s in-place `cholesky_in_place` (p×p) — avoids the
    /// per-PIRLS-iteration `.llt(Side::Lower)` allocation on the `BetaStep::Profile`
    /// β-Schur border step (packed/blocked/structured PIRLS variants alike).
    pub schur_llt_mem: MemBuffer,
    /// len p: Profile-mode β backtrack buffer (last-accepted β; the halving twin of `u_prev`). Untouched in Fixed mode.
    pub beta_prev: Vec<f64>,
}

/// The finite-difference pass's seed state, set by `joint_hessian_cov` before
/// its grid and restored on every exit; every field is `m`-sized or scalar,
/// so a worker workspace takes it by `clone`. `m = n_theta + p`, the `[θ | β]`
/// block the SE grid covers (never the NB slot).
#[derive(Clone)]
pub(crate) struct FdState {
    /// length m; converged γ̂ snapshot restored each return
    pub fd_saved: Vec<f64>,
    /// length m; per-coordinate FD step h_k
    pub fd_steps: Vec<f64>,
    /// When true, `laplace_deviance_at` seeds PIRLS from `u_seed` (the fitted mode
    /// û(γ̂)) instead of û = 0. Set ONLY by `joint_hessian_cov`, for **every** one of
    /// its evals including the central f0, and reset on every `joint_hessian_cov` exit
    /// so non-FD callers keep their cold, order-free û = 0 start. Same fixed-seed
    /// FD-derivative invariant as `joint_hessian_cov` in se.rs — see there for the
    /// derivation and for why f0 is inside the warm set too.
    pub warm_seed_active: bool,
    /// PIRLS exit-tol override read by `laplace_deviance_at` and forwarded to every
    /// PIRLS variant. `Some(pirls_tol_fd(family))` ONLY while `joint_hessian_cov`
    /// runs (set on entry, reset on every exit — the `warm_seed_active` discipline),
    /// so the FD second differences see a deviance converged at least as far as the
    /// fit's own exit. `None` everywhere else: the fit/BOBYQA path never pays the
    /// extra inner iterations and stays bit-identical.
    pub pirls_tol_override: Option<f64>,
    /// Force the FD stencil on every layout, skipping both exact Hessian
    /// rungs: the blocked and structured shapes land on
    /// `se::joint_hessian_cov`'s own grid, the packed-row layout on
    /// `se::packed_fd_hessian_cov`. Test-and-A/B only: nothing on a fitting
    /// path sets it, and it is how the crate's FD-vs-exact comparisons reach
    /// the stencil — `exact_hessian_matches_fd_on_fixture` on the dense side,
    /// `packed_assembled_se_matches_the_packed_stencil` and
    /// `sparse::fd_margin`'s corpus measurement on the packed one.
    /// Mirrors `force_dense_schur`.
    pub(crate) force_fd_hessian: bool,
}

/// Post-search inference outputs and their scratch: the joint Hessian and
/// gradient, the μ̂ snapshot `joint_hessian_cov` restores, Cov(β̂) and its
/// column scratch, the per-target SE vectors, and the joint-Wald matrices.
pub(crate) struct InferenceScratch {
    /// m × m joint-deviance Hessian
    pub hess_scratch: Mat<f64>,
    /// Scratch for the joint gradient, length `m`. Sized once.
    pub(crate) grad_scratch: Vec<f64>,
    /// μ̂ as `joint_hessian_cov` was handed it, restored on every exit beside
    /// `ws.pirls.u`. The tail re-eval at the end of that function re-solves PIRLS
    /// at γ̂ and writes a fresh `ws.pirls.prob`, while `ws.pirls.u` is put back
    /// verbatim — so without this the workspace exits carrying a (`u`, `prob`) pair from two
    /// different converged solves, and Gamma's σ̂² (`family::glmm_sigma_sq`,
    /// read in `fit/glmm.rs`) is built from the mismatch. Length max_n.
    /// Adding the restore (2026-09-05) moved the bit-identity dump's `theta`
    /// on the dense Gamma rung (`sim_gamma`, rung 23) from 0.23550934106996388
    /// to 0.23550934100844045 — a 2.6e-10 relative shift, the Hessian arm now
    /// reporting the same σ̂² as the Rx arm; every other record stayed
    /// byte-identical.
    pub(crate) fd_saved_prob: Vec<f64>,
    /// p×p Cov(β̂) — `var_diag` is its diagonal, and both are filled together at
    /// the same target indices (NaN elsewhere). Workspace-owned, not returned on
    /// `GlmmFit`, so filling it costs no per-fit allocation and the `Rx` warm
    /// path keeps its zero-alloc gate. Sourced from the full matrix each SE arm
    /// already forms: `Rx` from the Schur forward-solve columns, `Hessian` from
    /// `joint_hessian_cov`'s β block. Mapped to `Fit::vcov` by `fit/glmm.rs`.
    pub vcov: Mat<f64>,
    /// p×p scratch holding column `j` of `L⁻¹` at each target `j` — the `Rx`
    /// arm's per-target forward solves, kept so their pairwise dots can fill
    /// `vcov`'s off-diagonals instead of only `‖·‖²` on its diagonal.
    pub vcov_cols: Mat<f64>,
    /// length p
    pub var_diag: Vec<f64>,
    /// length p
    pub t_sq: Vec<f64>,
    // SE of each θ coordinate = sqrt of the θ-block diagonal of the joint (θ,β)
    // Hessian covariance (length n_theta). Filled ONLY on the `WaldSe::Hessian`
    // GLMM path from the θ block `joint_hessian_cov` already inverts (it otherwise
    // discards it); NaN under `WaldSe::Rx`, on the Hessian RX fallback, and on a
    // non-converged fit. For a SCALAR grouping (q=1, dispersion≡1) the RE stddev
    // equals its θ, so this is that stddev's SE directly (identity delta map); the
    // only reachable GLMM groupings are scalar (intercept-only).
    /// length n_theta
    pub theta_se: Vec<f64>,
    /// length p; Var(β̂)_jj forward-solve scratch (per-target)
    pub fwd_solve: Vec<f64>,
    // joint Wald scratch (reuse lmm::joint_wald_chi_sq):
    /// Inverse of the joint Wald K matrix, p×p (see `lmm::joint_wald_chi_sq`).
    pub joint_k_inv: Mat<f64>,
    /// Cholesky factor of the joint Wald Σ_t, p×p.
    pub joint_sigma_t_chol: Mat<f64>,
    /// Joint Wald right-hand side, length p.
    pub joint_rhs: Vec<f64>,
}

/// The per-fit design every GLMM kernel reads and none writes: the outcome
/// family, the RE topology, the fixed design and response, prior weights
/// (already `[..n]`), the RE level ids, the widened slope columns and the
/// offset. Borrowed from the workspace and the caller's data for the duration
/// of one fit; `nb_theta` is not here because the NB search changes it per
/// evaluation.
#[derive(Clone, Copy)]
pub(crate) struct FitData<'a> {
    pub family: crate::Family,
    pub groupings: &'a LmmGroupings,
    /// Which `A`-layout this design takes — see [`GlmmLayout`]. The one place
    /// the routing decision is read; the kernels never re-derive it.
    pub layout: GlmmLayout,
    pub x: MatRef<'a, f64>,
    pub y: &'a [f64],
    pub prior_w: &'a [f64],
    pub weighted: bool,
    pub cluster_ids: &'a [u32],
    /// Per-row extra-grouping level ids — read only on the structured route
    /// (`build_packed_m`'s nested-indicator and crossed-level reconstruction);
    /// unread on the blocked and packed routes.
    pub extra_ids: &'a [Vec<u32>],
    pub z_buf: &'a [f64],
    /// Per-row linear-predictor offset (`FitOptions::offset`), forwarded to
    /// every PIRLS/AGQ variant's `eta_fixed` fill. `None` ⇒ no offset.
    pub offset: Option<&'a [f64]>,
    pub n: usize,
    pub p: usize,
}

/// All GLMM solver scratch — allocated once per (spec, max_n) shape.
pub struct GlmmWorkspace {
    /// reused RE structure (estimator-agnostic)
    pub groupings: LmmGroupings,
    /// Outcome family/link selecting the PIRLS IRLS math — the arm
    /// `simd_transcendental::family_pass` dispatches to each iteration.
    pub family: crate::Family,
    /// NB dispersion fixed for this fit's PIRLS/AGQ variance/deviance — read only
    /// when `family` is `NegativeBinomial`. Defaulted to `f64::NAN` at
    /// construction; `fit::run_glmm_on` sets it per fit — NB passes its start θ₀,
    /// every other family leaves it NaN — and `fit_glmm` leaves θ̂_NB in it on
    /// exit.
    pub nb_theta: f64,
    /// adaptive GH node count; 1 = Laplace. >1 only fires on the single-grouping-factor
    /// binomial/Poisson AGQ paths — scalar intercept (`agq::agq_deviance`) or vector RE
    /// with `q_p ∈ 2..=3` (`agq::agq_deviance_vec`); ignored otherwise.
    pub nagq: u8,
    /// FitOptions::parallel_inner, copied per fit by the fit.rs adapter (the
    /// nb_theta pattern). Runtime gate for the parallel kernels in `parallel`
    /// builds: when false — or in any serial build — the per-fit
    /// ClusterRowIndex is never built, agq_deviance runs the original
    /// node-outer loop, and the FD-Hessian grids stay serial; a batch caller
    /// pays nothing.
    pub parallel_inner: bool,
    /// Per-cluster row CSR for the cluster-outer AGQ loop — rayon's
    /// work-splitting substrate. Built once per fit in fit_glmm (cluster_ids
    /// is fit-fixed) iff the `parallel` feature is compiled in AND
    /// `nagq > 1 && parallel_inner`; None otherwise. Serial builds are always
    /// node-outer: cluster-outer serially regresses many-tiny-cluster shapes
    /// (see the build site in fit_glmm).
    pub(crate) cluster_rows: Option<super::agq::ClusterRowIndex>,
    /// total RE columns (groupings.k_total)
    pub k: usize,
    /// fixed-effect predictors
    pub p: usize,
    /// count of variance-component (θ) parameters (groupings.n_theta())
    pub n_theta: usize,
    /// Which `A`-layout this shape takes — see [`GlmmLayout`]. Fixed at
    /// construction; the deviance router, the Schur-fill dispatch and the
    /// SE arms all read it instead of re-deriving the predicate.
    pub(crate) layout: GlmmLayout,
    /// Packed-row layout scratch — see [`PackedScratch`].
    pub(crate) packed: PackedScratch,
    /// Joint (θ,β) BOBYQA solver, dimension `n_theta + p` (+1 on NB: the trailing
    /// `ln θ_NB` coordinate).
    pub solver: Bobyqa, // sized n_theta + p
    /// Joint solver's live iterate: `[θ (n_theta) | β (p)]`. STABLE READ-BACK
    /// CONVENTION: after `fit_glmm` returns with `converged == true`, this
    /// holds the pinned optimum — `params[..n_theta]` is θ̂ (boundary
    /// components zeroed) and `params[n_theta..n_theta + p]` is β̂ (on `Joint`
    /// and `PqlThenJoint` a joint `[θ | β]` solve writes this suffix directly;
    /// on `ExactProfile` the θ-only stage-1 incumbent snapshot is copied in
    /// instead — see `OuterSearch` — and `betas` is copied from this suffix
    /// either way) — so a caller may read it back as the warm start for a
    /// subsequent fit of related data. On a non-converged fit the content is an
    /// arbitrary iterate — do not read it. On NB the vector has one more trailing
    /// entry, `ln θ_NB`, whose exponential `fit_glmm` writes back into `nb_theta`
    /// — the `[..n_theta + p]` read-back contract is unchanged.
    pub params: Vec<f64>, // [θ | β]
    /// Joint solver box lower bounds, length `n_theta + p` (+1 on NB).
    pub lower: Vec<f64>,
    /// Joint solver box upper bounds, length `n_theta + p` (+1 on NB).
    pub upper: Vec<f64>,
    /// θ-only BOBYQA solver for the θ-only outer search shared by
    /// `PqlThenJoint` (a warm-start accelerant) and `ExactProfile` (the search
    /// itself — see `OuterSearch`): sized `n_theta` (+1 on NB), configured with
    /// the same `rho_begin`/`GLMM_RHO_END` schedule as `solver` and the LMM
    /// mid-model `npt` rule (`ceil(1.5·n_theta) + 1` at
    /// `n_theta ≥ 3`, else `2·n_theta + 1`) — not
    /// the joint solver's `npt`, which differs. See `fit_glmm`.
    pub solver_stage1: Bobyqa,
    /// θ-only candidate/incumbent buffer for stage 1, length `n_theta` (+1 on
    /// NB); seeded from `params`'s θ prefix at construction.
    pub params_stage1: Vec<f64>,
    /// Stage-1 box, length `n_theta + n_nb`: `lower[..n_theta]` / `upper[..n_theta]`
    /// plus the `ln θ_NB` bound on NB — see `params_stage1`.
    pub lower_stage1: Vec<f64>,
    pub upper_stage1: Vec<f64>,
    /// Outer search route for this shape — see `OuterSearch`.
    pub outer_search: OuterSearch,
    /// PIRLS scratch (sized max_n / k) — see [`PirlsScratch`].
    pub(crate) pirls: PirlsScratch<f64>,
    /// n×(q_p−1) row-major f64 copy of x[:, slope_cols] — filled once per fit
    pub z_buf: Vec<f64>,
    /// Per-row prior weights `wᵢ` (`FitOptions::weights`; all-1 when absent —
    /// zero behavioral change). `wᵢ·W̃ᵢ` on the working weight,
    /// `wᵢ·devᵢ` on the deviance, `wᵢ·ρᵢ` on the score; everything downstream
    /// (A/RHS scatter, β border, Schur, FD Hessian) reads `w`/ρ and inherits it.
    pub(crate) prior_w: Vec<f64>,
    /// True iff `prior_w` was filled from `FitOptions::weights`. Selects between
    /// the two logit arms of `simd_transcendental::family_pass`: the fused
    /// `Σ log1pexp` deviance identity holds only for unweighted Bernoulli rows.
    pub(crate) weighted: bool,
    /// within-fit û warm-start incumbent; RESET to 0 each fit_glmm — never carried across fits
    pub u_seed: Vec<f64>,
    /// max_n × p = W∘X scratch for the X'WX GEMM (rebuilt per PIRLS iteration,
    /// all three pirls variants and the three se.rs schur-fill twins)
    pub wx: Mat<f64>,
    /// Structured crossed/nested route scratch — see [`StructuredScratch`].
    pub(crate) structured: StructuredScratch<f64>,
    /// Structured route's θ-independent index pattern — see [`StructuredPattern`].
    pub(crate) pattern: StructuredPattern,
    /// β-border scratch shared by the Profile-mode PIRLS step and `se.rs` —
    /// see [`BorderScratch`].
    pub(crate) border: BorderScratch,
    /// length p (copied from params[n_theta..])
    pub betas: Vec<f64>,
    // β-profiling (`BetaStep`) scratch — see `pirls::BetaStep`. All length p.
    /// len p: Profile-mode δβ RHS/solution scratch; also the Fixed-mode β-input transient (deviance.rs copies params[n_theta..] here — NOT `betas`, which is the reported output)
    pub beta_rhs: Vec<f64>,
    /// len p: stage-1 profiled-β in/out buffer
    pub beta_prof: Vec<f64>,
    /// len p: stage-1 incumbent β snapshot (mirrors u_seed)
    pub beta_seed: Vec<f64>,
    /// Exact-profile scratch (`pirls::ExactProfileBufs`), sized once here.
    pub(crate) exact_prof: super::pirls::ExactProfileBufs,
    /// Post-search inference outputs and their scratch — see [`InferenceScratch`].
    pub(crate) inference: InferenceScratch,
    /// The finite-difference pass's seed state — see [`FdState`].
    pub(crate) fd: FdState,
    /// Per-row linear-predictor offset (`FitOptions::offset`), read by every
    /// `eta_fixed` refresh (`pirls::refresh_eta_fixed`). `None` ⇒ no offset,
    /// byte-identical to the pre-offset code.
    pub(crate) offset: Option<Vec<f64>>,
    /// Count of fit-path (`pirls_tol_override.is_none()`) PIRLS solves that ran
    /// the full `PIRLS_MAX_ITERS` cap without converging — observation-only,
    /// bookkeeping read back by `FitDiagnostics`/`Note::PirlsExhausted`, never by
    /// any numeric path. Reset to 0 at the top of every `fit_glmm` (mirrors
    /// `u_seed`) so a `loop_advanced` reuse never carries a prior draw's count.
    pub(crate) pirls_exhausted: u32,
    /// Whether the FINAL re-evaluation at the pinned γ̂ itself exhausted the
    /// PIRLS cap. Reset to `false` at the top of every `fit_glmm`.
    pub(crate) final_pirls_exhausted: bool,
    /// Observation-only optimizer counters for the fit in progress — the stage
    /// split, the shrink phase, the PIRLS histogram and the AGQ node cost.
    /// Never read by any numeric path. Reset to `new()` at the top of every
    /// `fit_glmm` (mirrors `pirls_exhausted`) so a `loop_advanced` reuse never
    /// carries a prior draw's counts.
    pub(crate) counters: crate::counters::EvalCounters,
    /// Dual-arithmetic twins of the θ-dependent PIRLS buffers at the FIRST-
    /// derivative order (`Dual<N>`), allocated on the first gradient request for
    /// this workspace and reused thereafter. `None` on every `f64`-only fit, so
    /// a caller that never asks for a gradient pays no memory.
    pub(crate) dual_scratch: Option<Box<super::derivative::GlmmDualScratch>>,
    /// The same at the SECOND-derivative order (`HyperDual<N, H>`), in its own
    /// slot. A shared slot is what a caller that takes both a gradient and a
    /// Hessian would rebuild — one buffer list into the other and back on
    /// every warm refit; the `HyperDual<8,36>` list is 45 `f64` per element.
    /// The separate slot avoids that rebuild.
    pub(crate) hyper_scratch: Option<Box<super::derivative::GlmmDualScratch>>,
    /// `f64` scratch of the packed-row assembled derivative engine — its
    /// assembly buffers, the mode snapshot its solve is taken around, and
    /// `û`'s first-order response `U`. Its own slot rather than a field of
    /// the dual scratch because the dual kernel does not support the
    /// packed-row layout at all, so a packed fit never builds one; `None` on
    /// every layout but `GlmmLayout::Packed`, and on a packed fit until its
    /// first derivative request.
    pub(crate) packed_asm: Option<Box<super::assembled::PackedGradientBufs>>,
}

impl GlmmWorkspace {
    /// Build the GLMM workspace for a Glm+cluster spec. `slope_cols` are the
    /// x_full indices of the primary slopes (`spec.cluster_slope_design_cols`).
    /// Test-only: production callers go through [`Self::for_cluster_spec_ext`],
    /// since a design whose extra groupings carry slopes needs their columns.
    #[cfg(test)]
    pub(crate) fn for_cluster_spec(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
        nagq: u8,
    ) -> Self {
        Self::for_cluster_spec_ext(p, cluster, max_n, slope_cols, &[], nagq)
    }

    /// The same for a design whose EXTRA groupings carry slopes:
    /// `extra_slope_cols[e]` holds the x_full indices of extra grouping `e`'s
    /// slopes, in declaration order. Only the packed-row layout applies a full
    /// `q_g×q_g` Λ block per extra level, so any other layout must be handed
    /// `&[]` here (`from_groupings` asserts it).
    pub(crate) fn for_cluster_spec_ext(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
        extra_slope_cols: &[Vec<usize>],
        nagq: u8,
    ) -> Self {
        let groupings =
            LmmGroupings::from_cluster_spec_ext(cluster, max_n, slope_cols, extra_slope_cols);
        let layout = GlmmLayout::for_design(cluster, &groupings);
        Self::from_groupings(groupings, cluster.family, p, max_n, nagq, layout)
    }

    /// Test-only: the same workspace on the packed-row layout whatever
    /// [`GlmmLayout::for_design`] would pick, so a test can drive the packed
    /// kernel as the oracle for the blocked and structured ones. The caller
    /// still owns [`fill_packed_cols`], exactly as on a production packed fit.
    #[cfg(test)]
    pub(crate) fn for_cluster_spec_packed(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
        nagq: u8,
    ) -> Self {
        let groupings = LmmGroupings::from_cluster_spec(cluster, max_n, slope_cols);
        Self::from_groupings(
            groupings,
            cluster.family,
            p,
            max_n,
            nagq,
            GlmmLayout::Packed,
        )
    }

    /// Build the workspace from an already-constructed `LmmGroupings` (+ family).
    /// The extracted tail of `for_cluster_spec` — everything past the groupings
    /// build depends only on `(groupings, family, p, max_n, nagq)`, never on the
    /// `ModelSpec` or `slope_cols` themselves. Split out so a per-thread FD-Hessian
    /// worker can reconstruct a fresh, identically-sized workspace from a live one's
    /// cloned groupings without re-threading the spec (`fd_worker_ws`).
    pub(crate) fn from_groupings(
        groupings: LmmGroupings,
        family: crate::Family,
        p: usize,
        max_n: usize,
        nagq: u8,
        layout: GlmmLayout,
    ) -> Self {
        // The blocked and structured kernels build intercept-only extra groupings
        // exclusively, so a slope-carrying extra would fit a REDUCED model and
        // report it as a normal success. Only the packed layout applies full
        // q_g×q_g Λ blocks per extra level. Checked here, once per workspace
        // build, rather than in `build_packed_m`, whose per-eval `debug_assert`
        // stays debug-only because it sits in the hot loop.
        assert!(
            layout == GlmmLayout::Packed || !groupings.extra_slopes_any,
            "blocked and structured GLMM kernels cannot fit a slope-carrying extra grouping"
        );
        // The packed layout is Laplace-only: the AGQ kernels factorize the
        // marginal likelihood over independent per-cluster integrals, which the
        // shapes this layout serves (an oversized core, crossed tails, slopes on
        // an extra grouping) do not have. Pinned before `outer_search` below, so
        // the route decision sees the nAGQ this workspace will actually run.
        let nagq = if layout == GlmmLayout::Packed {
            1
        } else {
            nagq
        };
        let k = groupings.k_total;
        let n_theta = groupings.n_theta();
        // The NB dispersion is one trailing coordinate of the outer search, on
        // `ln θ_NB` boxed to the GLM bracket's range. Every other family has no
        // such slot: `n_nb = 0` leaves every dimension and bound below as it was.
        let n_nb = usize::from(matches!(family, crate::Family::NegativeBinomial { .. }));
        let q = groupings.primary_q;
        let n_primary = groupings.n_primary;
        // Structured-path block sizes: core width q_core = q_p + nested children,
        // crossed width e. Buffers stay 1-sized minima when the shape has no
        // extras (the no-extras blocked path never touches them), and on the
        // packed layout, which reads neither the structured scratch nor the
        // exact profile and carries the widest crossed tail of any layout —
        // mirrors `derivative::DenseTwinShape`, which sizes the dual twins the
        // same way.
        let packed_layout = layout == GlmmLayout::Packed;
        let q_core = if packed_layout {
            0
        } else {
            q + groupings.nested_per_parent
        };
        let e_crossed = if packed_layout {
            0
        } else {
            groupings.k_crossed()
        };
        // Whether this family's PIRLS takes the observed-information step, and
        // so whether the exact profile's observed twins carry storage at all.
        let observed = !packed_layout && !crate::family::is_canonical(family);

        // Bounds: θ part from blind_theta_and_bounds; β part = [−BETA_BOX, BETA_BOX].
        let (theta0, mut lower, mut upper) = groupings.blind_theta_and_bounds();
        let mut params = theta0;
        params.extend(std::iter::repeat_n(0.0, p)); // β cold default; overwritten at fit
        lower.extend(std::iter::repeat_n(-BETA_BOX, p));
        upper.extend(std::iter::repeat_n(BETA_BOX, p));
        if n_nb == 1 {
            params.push(0.0); // ln θ_NB start; written by `fit_glmm` from `nb_theta`
            lower.push(crate::fit::NB_THETA_LO.ln());
            upper.push(crate::fit::NB_THETA_HI.ln());
        }

        // ρ_begin ≤ RHO_BEGIN and ≤ 0.1·min diagonal θ₀ (mirror for_cluster_spec_ext)
        // so the cold blind start is not projected onto a bound. The start is the
        // structure-only blind θ₀, so each diagonal entry is THETA0.
        let blind_theta = vec![crate::lmm::THETA0; n_theta];
        let min_diag = groupings
            .diagonal_theta()
            .iter()
            .map(|&i| blind_theta[i].max(THETA_TRUTH_FLOOR))
            .fold(f64::INFINITY, f64::min);
        // Hoisted so the joint solver below and `solver_stage1` (the θ-only stage-1
        // solver) share the exact same computed θ-portion rho_begin — a pure
        // extraction, not a new derivation.
        let rho_begin = (0.1 * min_diag).min(RHO_BEGIN);
        // Feeds through the shared `apply_campaign_overrides` tail.
        let mut config = Config::new(n_theta + p + n_nb);
        config.rho_begin = rho_begin;
        config.rho_end = GLMM_RHO_END;
        crate::lmm::apply_campaign_overrides(&mut config, n_theta + p + n_nb);
        // Stage-1 θ-only BOBYQA config: same rho_begin/rho_end schedule as the
        // joint solver above, but `npt` mirrors the LMM's mid-model rule
        // (`LmmWorkspace::for_cluster_spec_ext`, `src/lmm/mod.rs`), NOT the
        // joint solver's — the two are sized for different-dimension searches
        // and this crate's precedent for a
        // θ-only search is the LMM one. The dimension fed into that shared
        // rule is `n_stage1` — θ, plus the `ln θ_NB` coordinate on NB. Both
        // configs feed through the shared `apply_campaign_overrides` tail.
        let n_stage1 = n_theta + n_nb;
        let npt_stage1 = if n_stage1 >= 3 {
            (3 * n_stage1).div_ceil(2) + 1
        } else {
            2 * n_stage1 + 1
        };
        let mut config_stage1 = Config::new(n_stage1);
        config_stage1.rho_begin = rho_begin;
        config_stage1.rho_end = GLMM_RHO_END;
        config_stage1.npt = npt_stage1;
        crate::lmm::apply_campaign_overrides(&mut config_stage1, n_stage1);
        // θ-only incumbent buffer and its box: the θ prefix of the joint start,
        // plus the ln θ_NB slot on NB (the joint vector's LAST entry, so the
        // stage-1 box is no longer a prefix of the joint box there).
        let mut params_stage1 = params[..n_theta].to_vec();
        let mut lower_stage1 = lower[..n_theta].to_vec();
        let mut upper_stage1 = upper[..n_theta].to_vec();
        if n_nb == 1 {
            let m = n_theta + p;
            params_stage1.push(params[m]);
            lower_stage1.push(lower[m]);
            upper_stage1.push(upper[m]);
        }

        // Route per shape — see `OuterSearch`. Computed here, before `groupings`
        // moves into the struct literal below.
        //
        // The `n_theta <= 2 && p <= 4` skip (to `Joint`, only where the PQL route
        // is still the one taken) is dimension-gated: for small (n_theta, p)
        // shapes stage 1 roughly doubles the PIRLS-solve count (see the field
        // doc) while BOBYQA reaches the same stage-2 optimum blind almost as
        // fast, so it isn't worth its own cost — BUT for wider joint dims the
        // un-warm-started single-stage search can cost much MORE than stage 1
        // saves (a first cut at this threshold, gated only on named datasets
        // rather than a corpus sweep, missed this and shipped a regression — see
        // below).
        //
        // Threshold below is from a locked-machine (`bench-l`, `taskset -c 1`)
        // timing sweep of every rung that reaches this constructor (non-Gaussian
        // dense NoZ; `validation/` datasets with a sparse/LMM path are unaffected by
        // this field and were confirmed identical across both sweep arms) — the
        // full corpus, not a hand-picked dataset list: an earlier hand-picked list
        // missed one loser (Arabidopsis, see below). Protocol: per arm (forced skip
        // vs forced keep), two independent
        // `validation_fit` invocations, each itself the median of 9 timed samples
        // after a discarded warmup (see validation/engines/glmm.rs); invocations agreed
        // within ~2% everywhere, and the table shows the keep-arm/skip-arm medians
        // (Poisson rows re-measured after the dense-PIRLS weight-loop revert that
        // restored the pre-helper per-row math; the logit rows — cbpp, VerbAgg —
        // run the fused-SIMD arm that revert never touched):
        //
        //   dataset             (n_theta,p)  skip-vs-keep fit_median   verdict
        //   cbpp                (1,4)        0.0036 vs 0.0041s (-14%)  SKIP wins
        //   sim_poisson_nested  (2,2)        0.0085 vs 0.0099s (-14%)  SKIP wins
        //   grouseticks         (3,4)        0.4053 vs 0.2168s (+87%)  KEEP wins
        //   Arabidopsis         (2,6)        0.0772 vs 0.0276s (+180%) KEEP wins
        //   VerbAgg             (2,7)        2.1575 vs 1.0082s (+114%) KEEP wins
        //   sim_crossed_at_cap  (7,2)        0.1736 vs 0.1263s (+37%)  KEEP wins
        //
        // Arabidopsis (n_theta=2, p=6) would be a false positive under a looser
        // `n_theta <= 2 && p <= 6` bound: it matches the two true winners on
        // n_theta only, without checking p against a real measurement — it is
        // in fact the single biggest regression in the corpus (skip is 2.8x
        // SLOWER), because the un-warm-started 8-dim joint BOBYQA search costs
        // far more than the skipped stage-1 pass saves.
        // (Correctness is NOT at risk either way — beta/SE/varcomp agree to
        // ~1e-6 relative between skip and keep on Arabidopsis; this is purely a
        // performance threshold, re-derive it if the corpus changes.)
        //
        // The two true winners both have p ≤ 4; every loser (including
        // Arabidopsis) has p ≥ 6 — a wide, data-supported margin — so the
        // threshold is `p ≤ 4`. `n_theta ≤ 2` holds too (grouseticks at
        // n_theta=3 is the nearest loser on that axis and is never
        // miscategorized).
        //
        // With the `ExactProfile` route in place, none of the rows above reach
        // this branch: every dataset in the table is an nAGQ=1 non-Gamma shape
        // and routes `ExactProfile` first. What still reaches `n_theta <= 2 && p <= 4`
        // is Gamma, non-canonical structured extras, and packed-layout extras — a
        // population this sweep never measured. The threshold stands because nothing
        // has re-measured it, not because these numbers still cover it.
        let outer_search = if super::exact_profile_shape(family, nagq, layout) {
            OuterSearch::ExactProfile
        } else if nagq > 1 || (n_theta <= 2 && p <= 4) {
            OuterSearch::Joint
        } else {
            OuterSearch::PqlThenJoint
        };

        // Sized only on the layout that reads it — see [`PackedScratch`].
        let packed = if layout == GlmmLayout::Packed {
            PackedScratch::for_shape(&groupings, max_n, k)
        } else {
            PackedScratch::unused()
        };

        GlmmWorkspace {
            groupings,
            family,
            nb_theta: f64::NAN,
            nagq,
            parallel_inner: false,
            cluster_rows: None,
            k,
            p,
            n_theta,
            layout,
            packed,
            solver: Bobyqa::new(n_theta + p + n_nb, config)
                .expect("BOBYQA config constants are valid by construction"),
            params,
            lower,
            upper,
            solver_stage1: Bobyqa::new(n_stage1, config_stage1)
                .expect("BOBYQA config constants are valid by construction"),
            params_stage1,
            lower_stage1,
            upper_stage1,
            outer_search,
            pirls: PirlsScratch::for_shape(max_n, k, q, n_primary, nagq),
            z_buf: vec![0.0; max_n * (q - 1)],
            prior_w: vec![1.0; max_n],
            weighted: false,
            u_seed: vec![0.0; k.max(1)],
            wx: Mat::zeros(max_n, p),
            structured: StructuredScratch::for_shape(max_n, n_primary, q_core, e_crossed),
            pattern: StructuredPattern {
                cross_col: vec![0u32; (max_n * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
                n_cross: vec![0u8; max_n.max(1)],
                coup_cols: vec![0u32; (max_n * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
                coup_ptr: vec![0u32; n_primary + 1],
                coup_mask: None,
                structured_schur: None,
                force_dense_schur: false,
            },
            border: BorderScratch {
                xtwx: Mat::zeros(p, p),
                xtwm: Mat::zeros(p, k.max(1)),
                ainv_mtwx: Mat::zeros(k.max(1), p),
                schur: Mat::zeros(p, p),
                schur_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                    p,
                    Par::Seq,
                    Spec::default(),
                )),
                beta_prev: vec![0.0; p],
            },
            betas: vec![0.0; p],
            beta_rhs: vec![0.0; p],
            beta_prof: vec![0.0; p],
            beta_seed: vec![0.0; p],
            exact_prof: super::pirls::ExactProfileBufs {
                // `k` is `groupings.k_total`, so these span the structured
                // path's `k_family + e` as well as the blocked path's `q·s`.
                logdet_u: vec![0.0; k.max(1)],
                logdet_beta: vec![0.0; p],
                // Twins of the four buffers above, sized only where the exact
                // profile reads them — the `observed` flag above, which the
                // `DualStep` twins mirror through `derivative::DenseTwinShape`.
                obs_blocks: vec![0.0; super::pirls::obs_len(observed, (q * q * n_primary).max(1))],
                obs_core_blocks: vec![
                    0.0;
                    super::pirls::obs_len(
                        observed,
                        (q_core * q_core * n_primary).max(1)
                    )
                ],
                obs_coupling: vec![
                    0.0;
                    super::pirls::obs_len(
                        observed,
                        (q_core * n_primary * e_crossed).max(1)
                    )
                ],
                obs_schur_blk: vec![
                    0.0;
                    super::pirls::obs_len(observed, (e_crossed * e_crossed).max(1))
                ],
                obs_schur: None,
                u_acc: vec![0.0; k.max(1)],
                tail_inv: vec![0.0; (e_crossed * e_crossed).max(1)],
                tail_r: vec![0.0; e_crossed.max(1)],
                tail_g: vec![0.0; (q_core * q_core * n_primary).max(1)],
                tail_h: vec![0.0; (q_core * n_primary * e_crossed).max(1)],
                fac_f64: vec![0.0; (q_core * q_core * n_primary).max(1)],
            },
            inference: InferenceScratch {
                hess_scratch: Mat::zeros((n_theta + p).max(1), (n_theta + p).max(1)),
                grad_scratch: vec![0.0; n_theta + p],
                fd_saved_prob: vec![0.0; max_n],
                vcov: Mat::zeros(p, p),
                vcov_cols: Mat::zeros(p, p),
                var_diag: vec![0.0; p],
                t_sq: vec![0.0; p],
                theta_se: vec![f64::NAN; n_theta],
                fwd_solve: vec![0.0; p],
                joint_k_inv: Mat::zeros(p, p),
                joint_sigma_t_chol: Mat::zeros(p, p),
                joint_rhs: vec![0.0; p],
            },
            fd: FdState {
                fd_saved: vec![0.0; n_theta + p],
                fd_steps: vec![0.0; n_theta + p],
                warm_seed_active: false,
                pirls_tol_override: None,
                force_fd_hessian: false,
            },
            offset: None,
            pirls_exhausted: 0,
            final_pirls_exhausted: false,
            counters: crate::counters::EvalCounters::new(),
            dual_scratch: None,
            hyper_scratch: None,
            packed_asm: None,
        }
    }
}

/// Fresh workspace for one FD-Hessian worker thread: an independently-sized clone
/// of `src` carrying everything an FD deviance eval reads or mutates, so a rayon
/// grid thread never aliases the live workspace. Bit-identity rests on `fd_eval`
/// restoring `params` from `fd_saved` and seeding û from the frozen `u_seed` every
/// eval — each grid cell is a pure function of `(fd_saved, fd_steps, u_seed,
/// design)`, identical whichever workspace computes it.
///
/// Construction reuses `from_groupings` (fresh scratch, correctly sized) on a
/// CLONE of `src`'s groupings, then copies the load-bearing state the fresh
/// constructor zeroes: the built design (`z`, `z_buf`), the structured crossed
/// factor (rebuilt as its own per-thread scratch — see `StructuredSchur::
/// clone_scratch`), and the `FdState` seed `joint_hessian_cov` set before the grid.
/// A missed field is a silent aliasing bug; the knob-on-vs-off bit-identity test
/// in `glmm/tests.rs` is the enforcement.
///
/// `coup_mask` is deliberately left `None` (the fresh constructor's value): the
/// worker rebuilds its own coupling CSR on the first structured eval, matching the
/// serial path's per-fit rebuild. `nb_theta`/`force_dense_schur` are copied because
/// the deviance reads them; `cluster_rows` stays `None` — on the AGQ path the
/// node-outer fallback it triggers is bit-identical to the cluster-outer loop.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub(crate) fn fd_worker_ws(src: &GlmmWorkspace, n: usize) -> GlmmWorkspace {
    let mut w = GlmmWorkspace::from_groupings(
        src.groupings.clone(),
        src.family,
        src.p,
        n,
        src.nagq,
        src.layout,
    );
    // Built design (`fill_packed_cols` / `fill_z_f64` output — constant across the
    // FD grid). src may be sized for max_n >= n (reusable-workspace surface); only
    // the first n rows are live, and both fills are row-major, so a prefix slice is
    // correct.
    let cols = w.packed.m_cols.len();
    w.packed.m_cols.copy_from_slice(&src.packed.m_cols[..cols]);
    let len = w.z_buf.len();
    w.z_buf.copy_from_slice(&src.z_buf[..len]);
    w.prior_w[..n].copy_from_slice(&src.prior_w[..n]);
    w.weighted = src.weighted;
    // Crossed-Schur factor: fresh per-thread scratch over the same symbolic pattern.
    w.pattern.structured_schur = src
        .pattern
        .structured_schur
        .as_ref()
        .map(|ss| ss.clone_scratch());
    w.pattern.force_dense_schur = src.pattern.force_dense_schur;
    // Mirrors the Fisher factor above so the two cannot drift, but FD workers
    // evaluate `BetaMode::Fixed` and never read the exact-profile twin.
    w.exact_prof.obs_schur = src
        .exact_prof
        .obs_schur
        .as_ref()
        .map(|ss| ss.clone_scratch());
    w.nb_theta = src.nb_theta;
    // FD seed state (joint_hessian_cov sets these before the grid).
    w.params.copy_from_slice(&src.params);
    w.fd = src.fd.clone();
    w.u_seed.copy_from_slice(&src.u_seed);
    w.offset = src.offset.clone();
    w
}

// Kernel — written as borrow-split FREE fns so the BOBYQA closure in fit_glmm can call
// them on destructured workspace fields without re-borrowing the whole workspace.

/// In-place lower Crout Cholesky of a `q×q` block stored row-major in `blk`
/// (lower triangle read; on return the lower triangle holds L). Returns false on
/// a non-positive pivot — the module's failure surface. q ≤ MAX_PRIMARY_Q (8).
/// Thin wrapper over the shared kernel in `crate::linalg::block_chol`.
pub(crate) fn glmm_block_chol<T: crate::scalar::Scalar>(blk: &mut [T], q: usize) -> bool {
    crate::linalg::block_chol(blk, q)
}

/// Solve `L Lᵀ x = b` in place (`b` overwritten with `x`) for the `q×q` lower
/// factor `l` produced by `glmm_block_chol` (row-major, diagonal = L pivots).
/// Forward `L y = b` then back `Lᵀ x = y`.
pub(crate) fn glmm_block_solve<T: crate::scalar::Scalar>(l: &[T], q: usize, b: &mut [T]) {
    for r in 0..q {
        let mut v = b[r];
        for c in 0..r {
            v -= l[r * q + c] * b[c];
        }
        b[r] = v / l[r * q + r];
    }
    for r in (0..q).rev() {
        let mut v = b[r];
        for c in (r + 1)..q {
            v -= l[c * q + r] * b[c];
        }
        b[r] = v / l[r * q + r];
    }
}

/// Panel variant of `glmm_block_solve`: solve `L Lᵀ X = B` in place for a
/// row-major `q×nc` RHS panel (`panel[r·nc..(r+1)·nc]` = row r) against the same
/// row-major factor. Identical substitution with the column loop hoisted inside:
/// each factor entry is read once per row op and the inner loop runs over the
/// contiguous row slice (vectorizable axpy) instead of re-walking the factor
/// once per RHS column.
pub(crate) fn glmm_block_solve_panel<T: crate::scalar::Scalar>(
    l: &[T],
    q: usize,
    panel: &mut [T],
    nc: usize,
) {
    for r in 0..q {
        let (done, rest) = panel.split_at_mut(r * nc);
        let row_r = &mut rest[..nc];
        for c in 0..r {
            let lrc = l[r * q + c];
            for (x, &y) in row_r.iter_mut().zip(&done[c * nc..(c + 1) * nc]) {
                *x -= lrc * y;
            }
        }
        let d = l[r * q + r];
        for x in row_r.iter_mut() {
            *x /= d;
        }
    }
    for r in (0..q).rev() {
        let (head, rest) = panel.split_at_mut((r + 1) * nc);
        let row_r = &mut head[r * nc..];
        for c in (r + 1)..q {
            let lcr = l[c * q + r];
            for (x, &y) in row_r.iter_mut().zip(&rest[(c - r - 1) * nc..(c - r) * nc]) {
                *x -= lcr * y;
            }
        }
        let d = l[r * q + r];
        for x in row_r.iter_mut() {
            *x /= d;
        }
    }
}

/// Cached sparse factor of the `e`-wide crossed Schur complement `S`.
/// `S`'s sparsity pattern is fixed by the crossed incidence (θ-independent), so the
/// symbolic factor is built ONCE per fit here and numeric-refactored every PIRLS
/// iteration / BOBYQA eval into `l_values`. Mirrors `SparseLmmWorkspace`
/// (`sparse.rs`), one factor narrower — only the crossed tail, not the whole system.
/// `None` for nested-only shapes (`e = 0`, no Schur).
// `pub` (fields stay `pub(crate)`) so `Scalar`'s `tail_*` methods — a `pub`
// trait — can name this type in their signatures without tripping the
// default-on `private_interfaces` lint; `#[doc(hidden)]` plus `mod glmm`
// staying private keeps it reachable-but-unnameable outside the crate, so
// nothing joins the public surface.
#[doc(hidden)]
pub struct StructuredSchur {
    /// Symbolic Cholesky of `S`'s pattern (AMD; simplicial or supernodal by
    /// faer's AUTO heuristic — `logdet_llt` handles both). Reused every refactor.
    pub(crate) symbolic: SymbolicCholesky<usize>,
    /// `S`'s value container in the fixed CSC pattern (lower tri + full diagonal).
    /// Values overwritten per PIRLS iteration by a gather from the dense `schur_blk`;
    /// pattern never changes. `parts_mut()` gives (symbolic, values) for the gather.
    pub(crate) axx: SparseColMat<usize, f64>,
    /// L-factor value buffer, length `symbolic.len_val()`. Overwritten per refactor.
    pub(crate) l_values: Vec<f64>,
    /// Numeric-factor scratch, sized once from `factorize_numeric_llt_scratch`.
    pub(crate) fac_mem: MemBuffer,
    /// Solve scratch, sized once from `solve_in_place_scratch(1, …)` — the Schur
    /// back-solve (PIRLS and each SE column) is always a single RHS column.
    pub(crate) solve_mem: MemBuffer,
    /// Downdate panels for `structured_factor`'s per-cluster `S −= C_f'A_f⁻¹C_f`
    /// (the LMM sparse-tail kernels A–D port): `c_panel` the gathered nonzero
    /// coupling columns (row-major `qc×e_f`), `y_panel` its `A_f⁻¹`-solved copy,
    /// `dd_temp` the `C_f'·Y` product (col-major `e_f×e_f`, lower). Sized once
    /// to `max_f e_f` off the FULL θ-independent incidence (`cols_of` in `new` —
    /// a superset of every θ-masked `coup_cols` CSR the fit visits), overwritten
    /// per cluster. Used only at `qc > 1`: at `qc == 1` `TailKernel::tail_downdate`'s
    /// `f64` override routes to the scalar walk instead and `new` sizes these to
    /// 0 (change together — that override's doc carries the qc=1 panel-vs-scalar
    /// measurement behind the split). It stays for qc>1, the only case that has
    /// real qc×e_f batched work to amortize the staging.
    pub(crate) c_panel: Vec<f64>,
    pub(crate) y_panel: Vec<f64>,
    pub(crate) dd_temp: Vec<f64>,
}

impl StructuredSchur {
    pub(crate) fn new(
        g: &LmmGroupings,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        n: usize,
    ) -> Option<StructuredSchur> {
        let e = g.k_crossed();
        if e == 0 {
            return None;
        }
        let s = g.n_primary;
        let k_family = (g.primary_q + g.nested_per_parent) * s;
        // Per-cluster set of crossed block-local columns each cluster touches — the
        // FULL incidence over all crossed groupings (NOT θ-filtered; the pattern must
        // be a superset for every θ the optimizer visits).
        let mut cols_of: Vec<Vec<u32>> = vec![Vec::new(); s];
        for cf in g.crossed.iter() {
            let off = g.extra_offsets[cf.decl];
            let ids = &extra_ids[cf.decl];
            for i in 0..n {
                let f = cluster_ids[i] as usize;
                let b = off + ids[i] as usize * cf.q - k_family;
                cols_of[f].push(b as u32);
            }
        }
        for v in cols_of.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }
        // Pattern triplets: full diagonal + Σ_f (coup_cols[f] × coup_cols[f]) lower tri.
        // Dedup by a visited-set of (a,b) so try_new_from_triplets sees each once.
        let mut seen = std::collections::HashSet::<(usize, usize)>::new();
        let mut trips = Vec::<Triplet<usize, usize, f64>>::new();
        for b in 0..e {
            trips.push(Triplet::new(b, b, 0.0));
            seen.insert((b, b));
        }
        for cols in &cols_of {
            for &bb in cols {
                for &aa in cols {
                    let (a, b) = (aa as usize, bb as usize);
                    if a >= b && seen.insert((a, b)) {
                        trips.push(Triplet::new(a, b, 0.0));
                    }
                }
            }
        }
        let axx = SparseColMat::<usize, f64>::try_new_from_triplets(e, e, &trips)
            .expect("Schur pattern triplets well-formed");
        let symbolic = factorize_symbolic_cholesky(
            axx.symbolic(),
            Side::Lower,
            Default::default(), // AMD fill-reducing ordering
            CholeskySymbolicParams {
                // AUTO: simplicial or supernodal per pattern; `logdet_llt`
                // handles both arms. Mirrors `clone_scratch` — change together.
                supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
                ..Default::default()
            },
        )
        .expect("Schur symbolic factorization");
        let l_values = vec![0.0f64; symbolic.len_val()];
        let fac_mem = MemBuffer::new(
            symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default()),
        );
        let solve_mem = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
        let qc = g.primary_q + g.nested_per_parent;
        let max_ef = cols_of.iter().map(|v| v.len()).max().unwrap_or(0);
        // At qc == 1 `TailKernel::tail_downdate`'s `f64` override routes the downdate
        // to its scalar arm (the panel staging is a +4–7% per-eval loss there;
        // that override's doc carries the measurement), so the panel buffers are
        // never touched — size them to 0. Mirrors the `qc != 1` filter in that
        // override — change together: widening the route without resizing
        // slices zero-length buffers and panics. `clone_scratch` follows
        // automatically (it mirrors these lengths).
        let panel_ef = if qc == 1 { 0 } else { max_ef };
        Some(StructuredSchur {
            symbolic,
            axx,
            l_values,
            fac_mem,
            solve_mem,
            c_panel: vec![0.0f64; qc * panel_ef],
            y_panel: vec![0.0f64; qc * panel_ef],
            dd_temp: vec![0.0f64; panel_ef * panel_ef],
        })
    }

    /// Per-thread clone for an FD-Hessian worker: shares nothing mutable with
    /// `self`. The symbolic pattern (`axx`) is copied and RE-factorized here rather
    /// than cloned — `SymbolicCholesky` is not `Clone`, but `factorize_symbolic_
    /// cholesky` on the same pattern with the same (deterministic AMD) ordering
    /// reproduces it bit-for-bit, so the per-eval numeric refactor lands on the
    /// identical elimination tree as `self`'s. `l_values`/scratch are fresh (their
    /// contents are overwritten every refactor). Mirrors `new`'s tail exactly.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pub(crate) fn clone_scratch(&self) -> StructuredSchur {
        let axx = self.axx.clone();
        let symbolic = factorize_symbolic_cholesky(
            axx.symbolic(),
            Side::Lower,
            Default::default(), // AMD fill-reducing ordering (deterministic)
            CholeskySymbolicParams {
                // Mirrors `new` — change together (same pattern + same params
                // ⇒ same supernodal/simplicial decision on every worker).
                supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
                ..Default::default()
            },
        )
        .expect("Schur symbolic factorization");
        let l_values = vec![0.0f64; symbolic.len_val()];
        let fac_mem = MemBuffer::new(
            symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default()),
        );
        let solve_mem = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
        StructuredSchur {
            symbolic,
            axx,
            l_values,
            fac_mem,
            solve_mem,
            c_panel: vec![0.0f64; self.c_panel.len()],
            y_panel: vec![0.0f64; self.y_panel.len()],
            dd_temp: vec![0.0f64; self.dd_temp.len()],
        }
    }
}

/// Fill the packed rows' RE column indices from the level ids, once per
/// `(design, ids)` — the column half of `M`'s packed rows, which is
/// θ-independent (`fill_m_vals` refills the values per eval).
///
/// Row `i` touches exactly one level of every grouping, so its `width` slots are
/// the primary block's `q_p` components at `c·n_primary + f` (component-major),
/// then each extra grouping's `q_g` components at
/// `extra_offsets[e] + level·q_g + c` (level-major). `extra_offsets` is ABSOLUTE
/// — it already includes the primary block width.
///
/// No-op on the blocked and structured layouts, whose packed buffers are
/// zero-length: they reconstruct `mᵢ` per row from the ids instead.
pub(crate) fn fill_packed_cols(
    ws: &mut GlmmWorkspace,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) {
    let width = ws.packed.width;
    if width == 0 {
        return;
    }
    let g = &ws.groupings;
    let q_p = g.primary_q;
    for i in 0..n {
        let mut t = i * width;
        let f = cluster_ids[i] as usize;
        for c in 0..q_p {
            ws.packed.m_cols[t] = (c * g.n_primary + f) as u32;
            t += 1;
        }
        for (e, ids_e) in extra_ids.iter().enumerate() {
            let q_g = g.extra_q[e];
            let base = g.extra_offsets[e] + ids_e[i] as usize * q_g;
            for c in 0..q_g {
                ws.packed.m_cols[t] = (base + c) as u32;
                t += 1;
            }
        }
    }
}

/// Per-fit hoist of the primary-slope Z columns: `z_buf[i·(q−1)+d] =
/// x[i, slope_cols[d]] / s_d`. θ/β change per BOBYQA eval but `x` and the scales
/// are fixed per fit, so this lifts the MatRef load and the scale division out of
/// the per-solve M fill — the fill becomes a pure contiguous-f64 product. `s_d`
/// is the RE column's internal scale (`LmmGroupings::set_slope_scales`); mirrored
/// by the Rx M row in `se::blocked_schur_fill` — change together. No-op at q_p = 1 (no slope columns).
pub(crate) fn fill_z_f64(g: &LmmGroupings, x: MatRef<f64>, z_buf: &mut [f64], n: usize) {
    let q = g.primary_q;
    for i in 0..n {
        for d in 0..q - 1 {
            z_buf[i * (q - 1) + d] = x[(i, g.primary_slope_cols[d])] / g.primary_slope_scales[d];
        }
    }
}

/// Pack the STRUCTURED-path nonzeros of `M = ZΛ` into the workspace's packed
/// buffers, once per deviance eval — only the `q_core` core + ≤`G` crossed
/// nonzeros each row reads, never a dense `n×k` `M`.
/// `m_core_buf[i·q_core+local]` = the `Λ`-scaled core value
/// `M[(i, core_col(f,local))]` for row `i`'s primary cluster (primary `local<q`:
/// `Σ_{r≥local} z_r·lam[r·q+local]`, the same reduction the blocked-path fill
/// runs; nested `local≥q`: the nested indicator scaled by its
/// θ). For each crossed grouping with `θ≠0`, the row's single active level
/// contributes one nonzero: `cross_val = z·θ`, `cross_col = b` (the crossed
/// block-local index, `0..e`), with `n_cross[i]` the count (`≤ G`). A θ-pinned
/// (θ=0) grouping is skipped — its `z·θ` is 0, so it has no nonzero.
/// Every RE-design value is reconstructed straight from the ids and `z_buf`:
/// the primary core from `z_buf` (the pre-widened slope buffer `fill_z_f64`
/// fills), the nested indicator and the crossed level from `extra_ids`
/// directly — no scan needed for either. Reads `z_buf`, `extra_ids`,
/// `cluster_ids` (only for the nested global→local id conversion — see below),
/// `lam` (filled here via `primary_lambda`), and `params`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_packed_m<T: crate::scalar::Scalar>(
    g: &LmmGroupings,
    params: &[T],
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    lam: &mut [T],
    cluster_ids: &[u32],
    m_core_buf: &mut [T],
    cross_val: &mut [T],
    cross_col: &mut [u32],
    n_cross: &mut [u8],
    n: usize,
) {
    let q = g.primary_q;
    let np = g.nested_per_parent;
    let qc = q + np;
    let k_family = qc * g.n_primary;
    let base_theta = q * (q + 1) / 2;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    // Intercept-only extras on the GLMM structured path (`classify_design`
    // routes extra-slopes shapes to Sparse for every family).
    debug_assert!(!g.extra_slopes_any);
    crate::lmm::primary_lambda(&params[..g.n_theta()], q, lam);
    let theta_nested = g.nested.map(|nf| params[nf.vech_start]).unwrap_or(T::ZERO);
    // Declaration index into `extra_offsets`/`extra_ids` for the nested factor;
    // `None` when there is no nested grouping (`np == 0`, the loop below never
    // reads it then).
    let nested_decl = g.nested.map(|nf| nf.vech_start - base_theta);
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        // Core primary block: the identical `Σ_{r≥c} z_r·lam[r·q+c]` reduction
        // that `glmm/pirls/blocked.rs`'s `pirls_solve_blocked` per-solve M fill runs
        // (whose own comment records it as bit-identical to the z-sourced form) — z_r is
        // 1.0 at r==0 (the RE intercept column) or the pre-widened slope
        // value `z_buf[i·(q−1)+(r−1)]` otherwise.
        for c in 0..q {
            let mut acc = T::ZERO;
            for r in c..q {
                let zr = if r == 0 {
                    T::ONE
                } else {
                    T::from_f64(z_buf[i * (q - 1) + (r - 1)])
                };
                acc += zr * lam[r * q + c];
            }
            m_core_buf[i * qc + c] = acc;
        }
        // Core nested children of parent f: `extra_ids` stores the nested id
        // GLOBAL (dense over all parents — see `GroupIds`'s doc), while the
        // packed core slots are LOCAL to this row's own parent block, so the
        // global id needs `f`'s own `f·np` prefix subtracted back off before
        // it can be compared against the local `j`. The RE design carries a 1.0
        // indicator at the row's own (global) nested id and 0.0 at every other
        // slot, so the packed value is θ_nested at that one local `j` and
        // `0.0 · theta_nested` everywhere else — kept as the multiply, not a
        // bare `0.0` literal, so a transient negative θ (an FD-Hessian
        // perturbation step, or an off-optimum BOBYQA trial — θ is only
        // guaranteed ≥ 0 at the pinned post-fit point, not at every eval)
        // still lands on the same `-0.0` the z-sourced form would have.
        if np > 0 {
            let global_id = extra_ids[nested_decl.expect("np > 0 ⇒ g.nested is Some")][i] as usize;
            // Before the ids-based rewrite, a malformed id panicked on the z
            // bounds check in every build profile; these two asserts keep at
            // least the debug-profile tripwire now that `z` is never read here.
            debug_assert!(
                global_id >= f * np,
                "row {i}: nested global id {global_id} underflows parent {f}'s block (f*np={})",
                f * np
            );
            let local_id = global_id - f * np;
            debug_assert!(
                local_id < np,
                "row {i}: nested local id {local_id} out of range (np={np})"
            );
            for j in 0..np {
                m_core_buf[i * qc + q + j] = if j == local_id {
                    theta_nested
                } else {
                    T::ZERO * theta_nested
                };
            }
        }
        // Crossed: one nonzero per crossed grouping (its single active level), θ-pinned
        // groupings skipped. The RE design carries that level's one 1.0 at
        // column `off + extra_ids[e][i]`, so the active column is a direct
        // index — no scan.
        let mut cnt = 0usize;
        for cf in &g.crossed {
            let theta = params[cf.vech_start];
            // Pin skip, `f64` only (mirrors deviance.rs's pin mask — change
            // together). Dropping a θ=0 crossed grouping narrows the CSR this
            // fills, and that is all it buys: the `e×e` Schur clear and the
            // Crout factorisation run at full design width either way
            // (`pirls/blocked_extras.rs`), so the skip saves scatter and
            // downdate work, never a factorisation.
            //
            // At a dual `T` the column is KEPT, so a lane seeded on that θ has
            // a column to differentiate instead of an all-zero Hessian row.
            // The value part is unchanged by keeping it: the retained column's
            // value is exactly 0.0, so every contribution it makes is an
            // exact-zero add — the same argument `structured_ainv_solve`'s doc
            // comment makes for skipping it.
            //
            // Changing this site WITHOUT deviance.rs's mask is a silent wrong
            // answer, not a no-op: this would write the wide
            // `cross_col`/`n_cross` while the CSR cache key stayed narrow,
            // `build_coupling_csr` would not rerun, and the newly retained
            // column would be scattered into `coupling` but never read back —
            // `structured_factor` and `structured_ainv_solve` walk `coup_cols`
            // only.
            if T::IS_F64 && theta.value() == 0.0 {
                continue;
            }
            // q_g==1 here (see debug_assert above), so vech_start − base_theta is
            // this factor's declaration index into extra_offsets/extra_ids.
            let e = cf.vech_start - base_theta;
            // Same tripwire as the nested branch above: before the ids-based
            // rewrite a malformed id panicked on the z bounds check in every
            // build profile; this keeps at least the debug-profile check.
            debug_assert!(
                (extra_ids[e][i] as usize) < cf.n_levels,
                "row {i}: crossed id {} out of range (n_levels={})",
                extra_ids[e][i],
                cf.n_levels
            );
            let off = g.extra_offsets[e];
            let col = off + extra_ids[e][i] as usize;
            cross_col[i * g_cap + cnt] = (col - k_family) as u32;
            cross_val[i * g_cap + cnt] = theta; // z·θ where z ≡ 1.0 exactly
            cnt += 1;
        }
        n_cross[i] = cnt as u8;
    }
}

/// `∂M/∂θ_a` for one row, in `build_packed_m`'s own packing. `Λ` is linear in
/// θ, so this is a selection, not a derivative of anything: the primary core
/// slot `c` takes `z_r` exactly when θ_a is the vech slot `(r, c)`; the nested
/// core slot takes the row's nested indicator when θ_a is the nested θ; a
/// crossed entry takes its `z` when θ_a is that grouping's θ. Every other slot
/// is zero. Mirrors `build_packed_m` — change together.
///
/// Returns `f64` whatever scalar the caller assembles at: the selection is a
/// constant in every coordinate, so a dual-typed caller lifts it with
/// `from_f64` and zero lanes. Reading it off the θ_a-lane of
/// `build_packed_m`'s own dual output is a different object — that lane is
/// `Σ_a ∂M/∂θ_a·dθ_a`, which coincides with this only for a single seeded
/// coordinate.
///
/// **`skip_pinned` must equal the caller's own scalar predicate `T::IS_F64`.**
/// `build_packed_m`'s crossed loop drops a θ-pinned (θ == 0) grouping at `f64`
/// and KEEPS it at a dual `T`, so the packed row's width and column order
/// differ between the two arms. Passing the matching flag makes the walk over
/// `g.crossed` here reproduce that arm's walk exactly, so `cross_val_d[z]`
/// pairs with the packed `cross_col[i*g_cap + z]` / `cross_val[i*g_cap + z]`
/// for every `z < n_cross[i]`. Passing the wrong one both leaves the tail
/// slots of `cross_val_d` unwritten — stale from the previous coordinate — and
/// pairs every surviving grouping with another grouping's column.
/// Serves both the blocked path (call with an `LmmGroupings` that has
/// no nested/crossed extras — `qc` reduces to `q` and the crossed loop is
/// empty, exactly `pirls_solve_blocked`'s own `m_buf` fill) and the
/// structured path, since the primary-core selection rule is the identical
/// `Σ_{r≥c} z_r·lam[r·q+c]` reduction on both.
#[allow(clippy::too_many_arguments)]
pub(crate) fn packed_m_theta_deriv(
    g: &LmmGroupings,
    a: usize,
    params: &[f64],
    // `T::IS_F64` of the scalar the caller assembles at — see the doc above.
    skip_pinned: bool,
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    cluster_ids: &[u32],
    i: usize,
    m_core_d: &mut [f64],
    cross_val_d: &mut [f64],
) {
    let q = g.primary_q;
    let np = g.nested_per_parent;
    let qc = q + np;
    let base_theta = q * (q + 1) / 2;
    // Intercept-only extras on the GLMM structured path (`classify_design`
    // routes any extra-slopes shape to Sparse for every family) — mirrors
    // `build_packed_m`'s own assert.
    debug_assert!(!g.extra_slopes_any);
    m_core_d[..qc].fill(0.0);
    if a < base_theta {
        // Invert the column-major vech enumeration `primary_lambda` writes
        // (`c` outer, `r` inner from `r == c`) to find the one `(r, c)` slot
        // θ_a scales.
        let mut t = 0;
        #[allow(clippy::needless_range_loop)]
        'vech: for c in 0..q {
            for r in c..q {
                if t == a {
                    let z_r = if r == 0 {
                        1.0
                    } else {
                        z_buf[i * (q - 1) + (r - 1)]
                    };
                    m_core_d[c] = z_r;
                    break 'vech;
                }
                t += 1;
            }
        }
    } else if let Some(nf) = g.nested.filter(|nf| nf.vech_start == a) {
        let f = cluster_ids[i] as usize;
        let nested_decl = nf.vech_start - base_theta;
        let global_id = extra_ids[nested_decl][i] as usize;
        let local_id = global_id - f * np;
        m_core_d[q + local_id] = 1.0;
    }
    // The same two-arm pin rule as `build_packed_m`'s crossed loop — change
    // together. At `f64` a pinned (θ=0) crossed grouping gets no
    // `cross_col`/`cross_val` slot there, so it must not consume one of
    // `cross_val_d`'s `z` positions here; at a dual `T` the column is kept, so
    // it must. Either way every grouping's derivative lands on the slot
    // holding the packed value it pairs with.
    let mut cnt = 0usize;
    for cf in &g.crossed {
        if skip_pinned && params[cf.vech_start] == 0.0 {
            continue;
        }
        cross_val_d[cnt] = if cf.vech_start == a { 1.0 } else { 0.0 };
        cnt += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::Dual;

    #[test]
    fn structured_schur_new_builds_symbolic_for_grouseticks() {
        // Shared grouseticks 3-crossed fixture (INDEX primary, [BROOD, LOCATION]
        // crossed); this test needs only the groupings + ids, not X/y.
        let (model, ids, _x, _y, n, _p) = crate::glmm::tests::grouseticks_3crossed_fixture();
        let g = LmmGroupings::from_cluster_spec(&model, n, &[]);
        let cluster_ids = ids.primary;
        let extra_ids = ids.extra;
        let ss = StructuredSchur::new(&g, &cluster_ids, &extra_ids, n).expect("e = 181 > 0 ⇒ Some");
        assert_eq!(ss.axx.ncols(), g.k_crossed());
        assert_eq!(ss.axx.ncols(), 181);
        // Symbolic factor allocated a non-empty L; diagonal is fully present.
        assert!(
            ss.symbolic.len_val() >= ss.axx.ncols(),
            "at least the e diagonal entries"
        );
        assert_eq!(ss.l_values.len(), ss.symbolic.len_val());
        // Fill-in is far below dense: dense would be e·(e+1)/2 = 16471 lower entries.
        assert!(
            ss.symbolic.len_val() < 16471,
            "sparse factor must have less fill than the dense lower triangle"
        );
    }

    /// `packed_m_theta_deriv`'s primary-vech selection — including the
    /// off-diagonal (intercept-slope covariance) slot, which a diagonal-only
    /// check would miss — held equal to a central difference of
    /// `build_packed_m` itself (exact to round-off: `Λ` is linear in θ). No
    /// extras on this shape, so `qc == q` and the crossed output is empty —
    /// the blocked-path packing this same selection rule serves.
    #[test]
    fn packed_m_theta_deriv_matches_central_diff_primary_vech() {
        let n = 16;
        let n_prim = 4;
        let spec = crate::ModelSpec {
            family: crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters {
                    n_clusters: n_prim as u32,
                },
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        };
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut ids = vec![0u32; n];
        for i in 0..n {
            ids[i] = (i % n_prim) as u32;
            x[(i, 0)] = 1.0;
            x[(i, 1)] = 0.3 + 0.11 * i as f64;
        }
        let g = LmmGroupings::from_cluster_spec(&spec, n, &[1]);
        let n_theta = g.n_theta();
        let q = g.primary_q;
        let qc = q; // no extras on this shape
        let mut z_buf = vec![0.0; n * (q - 1)];
        fill_z_f64(&g, x.as_ref(), &mut z_buf, n);
        let extra_ids: Vec<Vec<u32>> = vec![];
        let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
        let mut lam = vec![0.0; q * q];
        let mut m_core = vec![0.0; n * qc];
        let mut cross_val = vec![0.0; n * g_cap];
        let mut cross_col = vec![0u32; n * g_cap];
        let mut n_cross = vec![0u8; n];
        let mut params: Vec<f64> = (0..n_theta).map(|k| 0.4 + 0.1 * k as f64).collect();
        let eps = 1e-6;

        for a in 0..n_theta {
            let base = params[a];
            params[a] = base + eps;
            build_packed_m(
                &g,
                &params,
                &z_buf,
                &extra_ids,
                &mut lam,
                &ids,
                &mut m_core,
                &mut cross_val,
                &mut cross_col,
                &mut n_cross,
                n,
            );
            let m_plus = m_core.clone();
            params[a] = base - eps;
            build_packed_m(
                &g,
                &params,
                &z_buf,
                &extra_ids,
                &mut lam,
                &ids,
                &mut m_core,
                &mut cross_val,
                &mut cross_col,
                &mut n_cross,
                n,
            );
            let m_minus = m_core.clone();
            params[a] = base;

            let mut m_core_d = vec![0.0; qc];
            let mut cross_val_d: Vec<f64> = vec![];
            for i in [0usize, 1, n - 1] {
                packed_m_theta_deriv(
                    &g,
                    a,
                    &params,
                    true,
                    &z_buf,
                    &extra_ids,
                    &ids,
                    i,
                    &mut m_core_d,
                    &mut cross_val_d,
                );
                for c in 0..qc {
                    let want = (m_plus[i * qc + c] - m_minus[i * qc + c]) / (2.0 * eps);
                    assert!(
                        (m_core_d[c] - want).abs() < 1e-8,
                        "a={a} i={i} c={c}: deriv {} vs central diff {want}",
                        m_core_d[c]
                    );
                }
            }
        }
    }

    /// `packed_m_theta_deriv`'s nested and crossed selections, on
    /// `glmm_extras_q1_dataset`'s nested2+crossed3 shape (`q_core = 3`, one
    /// crossed grouping) — the twin of the primary-vech test above.
    #[test]
    fn packed_m_theta_deriv_matches_central_diff_extras() {
        let (x, y, ids, extra_ids, spec) = crate::glmm::tests::glmm_extras_q1_dataset(2, 3);
        let n = y.len();
        let g = LmmGroupings::from_cluster_spec(&spec, n, &[]);
        let n_theta = g.n_theta();
        let q = g.primary_q;
        let qc = q + g.nested_per_parent;
        let z_buf: Vec<f64> = vec![0.0; n * q.saturating_sub(1)];
        let _ = x; // only ids/extra_ids drive this shape's packing (q_p = 1)
        let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
        let mut lam = vec![0.0; q * q];
        let mut m_core = vec![0.0; n * qc];
        let mut cross_val = vec![0.0; n * g_cap];
        let mut cross_col = vec![0u32; n * g_cap];
        let mut n_cross = vec![0u8; n];
        let mut params: Vec<f64> = (0..n_theta).map(|k| 0.3 + 0.05 * k as f64).collect();
        let eps = 1e-6;

        for a in 0..n_theta {
            let base = params[a];
            params[a] = base + eps;
            build_packed_m(
                &g,
                &params,
                &z_buf,
                &extra_ids,
                &mut lam,
                &ids,
                &mut m_core,
                &mut cross_val,
                &mut cross_col,
                &mut n_cross,
                n,
            );
            let m_plus = m_core.clone();
            let cv_plus = cross_val.clone();
            params[a] = base - eps;
            build_packed_m(
                &g,
                &params,
                &z_buf,
                &extra_ids,
                &mut lam,
                &ids,
                &mut m_core,
                &mut cross_val,
                &mut cross_col,
                &mut n_cross,
                n,
            );
            let m_minus = m_core.clone();
            let cv_minus = cross_val.clone();
            params[a] = base;

            let mut m_core_d = vec![0.0; qc];
            let mut cross_val_d = vec![0.0; g.crossed.len()];
            for i in [0usize, 1, n - 1] {
                packed_m_theta_deriv(
                    &g,
                    a,
                    &params,
                    true,
                    &z_buf,
                    &extra_ids,
                    &ids,
                    i,
                    &mut m_core_d,
                    &mut cross_val_d,
                );
                for c in 0..qc {
                    let want = (m_plus[i * qc + c] - m_minus[i * qc + c]) / (2.0 * eps);
                    assert!(
                        (m_core_d[c] - want).abs() < 1e-8,
                        "a={a} i={i} c={c}: deriv {} vs central diff {want}",
                        m_core_d[c]
                    );
                }
                assert_eq!(
                    n_cross[i] as usize,
                    g.crossed.len(),
                    "no pinning on this fixture"
                );
                for z in 0..g.crossed.len() {
                    let want = (cv_plus[i * g_cap + z] - cv_minus[i * g_cap + z]) / (2.0 * eps);
                    assert!(
                        (cross_val_d[z] - want).abs() < 1e-8,
                        "a={a} i={i} z={z}: deriv {} vs central diff {want}",
                        cross_val_d[z]
                    );
                }
            }
        }
    }

    /// `packed_m_theta_deriv`'s crossed pin rule matches `build_packed_m` on
    /// BOTH arms, on a shape with two crossed groupings where the first is
    /// pinned (θ=0) and the second is active. At `f64` (`skip_pinned = true`)
    /// the packed row carries exactly one crossed entry — the active
    /// grouping's — and `cross_val_d[0]` must pair with THAT entry, not a slot
    /// reserved for the pinned one. At `Dual<1>` (`skip_pinned = false`) the
    /// packed row carries both columns in declaration order, and each
    /// grouping's derivative must land on its own slot.
    #[test]
    fn packed_m_theta_deriv_pairs_with_pinned_crossed_packing() {
        let n_prim = 4;
        let n = 24;
        let n_c1 = 3;
        let n_c2 = 4;
        let mut x = Mat::<f64>::zeros(n, 1);
        let mut ids = vec![0u32; n];
        let mut c1 = vec![0u32; n];
        let mut c2 = vec![0u32; n];
        for i in 0..n {
            ids[i] = (i % n_prim) as u32;
            x[(i, 0)] = 1.0;
            c1[i] = (i % n_c1) as u32;
            c2[i] = ((i / n_c1) % n_c2) as u32;
        }
        let spec = crate::ModelSpec {
            family: crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters {
                    n_clusters: n_prim as u32,
                },
                slopes: vec![],
                extra_groupings: vec![
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed {
                            n_clusters: n_c1 as u32,
                        },
                        slopes: vec![],
                    },
                    crate::Grouping {
                        relation: crate::GroupingRelation::Crossed {
                            n_clusters: n_c2 as u32,
                        },
                        slopes: vec![],
                    },
                ],
            }),
        };
        let extra_ids = vec![c1, c2];
        let _ = x;
        let g = LmmGroupings::from_cluster_spec(&spec, n, &[]);
        let n_theta = g.n_theta();
        assert_eq!(n_theta, 3, "primary + two crossed scalars");
        let q = g.primary_q;
        let qc = q; // no nested
        let z_buf: Vec<f64> = vec![];
        let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
        let mut lam = vec![0.0; q * q];
        let mut m_core = vec![0.0; n * qc];
        let mut cross_val = vec![0.0; n * g_cap];
        let mut cross_col = vec![0u32; n * g_cap];
        let mut n_cross = vec![0u8; n];
        let cross1_theta = g.crossed[0].vech_start;
        let cross2_theta = g.crossed[1].vech_start;
        let mut params = vec![0.0; n_theta];
        params[0] = 0.5; // primary
        params[cross1_theta] = 0.0; // pinned
        params[cross2_theta] = 0.7; // active
        let eps = 1e-6;

        build_packed_m(
            &g,
            &params,
            &z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core,
            &mut cross_val,
            &mut cross_col,
            &mut n_cross,
            n,
        );
        let n_cross_base = n_cross.clone();
        for &i in &[0usize, 1, n - 1] {
            assert_eq!(
                n_cross_base[i] as usize, 1,
                "cross1 pinned ⇒ only cross2's column is packed"
            );
        }

        // Perturb only the active grouping's θ — cross1 stays exactly pinned
        // at 0.0 across the step, so the packed width (n_cross) does not
        // change and a plain central difference is valid.
        params[cross2_theta] = 0.7 + eps;
        build_packed_m(
            &g,
            &params,
            &z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core,
            &mut cross_val,
            &mut cross_col,
            &mut n_cross,
            n,
        );
        let cv_plus = cross_val.clone();
        params[cross2_theta] = 0.7 - eps;
        build_packed_m(
            &g,
            &params,
            &z_buf,
            &extra_ids,
            &mut lam,
            &ids,
            &mut m_core,
            &mut cross_val,
            &mut cross_col,
            &mut n_cross,
            n,
        );
        let cv_minus = cross_val.clone();
        params[cross2_theta] = 0.7;

        let mut m_core_d = vec![0.0; qc];
        for &i in &[0usize, 1, n - 1] {
            assert_eq!(n_cross[i] as usize, 1);
            let want = (cv_plus[i * g_cap] - cv_minus[i * g_cap]) / (2.0 * eps);
            let mut active_d = vec![0.0; 1];
            packed_m_theta_deriv(
                &g,
                cross2_theta,
                &params,
                true,
                &z_buf,
                &extra_ids,
                &ids,
                i,
                &mut m_core_d,
                &mut active_d,
            );
            assert!(
                (active_d[0] - want).abs() < 1e-8,
                "i={i}: deriv {} vs central diff {want}",
                active_d[0]
            );
            // The pinned grouping's θ never claims a slot: differentiating
            // wrt an unrelated (primary) θ must leave the one active
            // position at zero, not shift the pinned grouping into it.
            let mut zero_d = vec![0.0; 1];
            packed_m_theta_deriv(
                &g,
                0,
                &params,
                true,
                &z_buf,
                &extra_ids,
                &ids,
                i,
                &mut m_core_d,
                &mut zero_d,
            );
            assert_eq!(
                zero_d[0], 0.0,
                "i={i}: unrelated θ must not claim the active slot"
            );
        }

        // --- the dual arm of the same rule: `build_packed_m` at `Dual<1>`
        // KEEPS the pinned grouping's column (value exactly 0.0), so the
        // packed row is two wide and `skip_pinned = false` must walk both
        // groupings in declaration order. ---
        let params_d: Vec<Dual<1>> = params.iter().map(|&v| Dual::<1> { v, d: [0.0] }).collect();
        let mut lam_d = vec![Dual::<1> { v: 0.0, d: [0.0] }; q * q];
        let mut m_core_dual = vec![Dual::<1> { v: 0.0, d: [0.0] }; n * qc];
        let mut cross_val_dual = vec![Dual::<1> { v: 0.0, d: [0.0] }; n * g_cap];
        let mut cross_col_dual = vec![0u32; n * g_cap];
        let mut n_cross_dual = vec![0u8; n];
        build_packed_m(
            &g,
            &params_d,
            &z_buf,
            &extra_ids,
            &mut lam_d,
            &ids,
            &mut m_core_dual,
            &mut cross_val_dual,
            &mut cross_col_dual,
            &mut n_cross_dual,
            n,
        );
        let k_family = qc * g.n_primary;
        let mut cvd = vec![0.0; 2];
        for &i in &[0usize, 1, n - 1] {
            assert_eq!(
                n_cross_dual[i] as usize, 2,
                "i={i}: the dual packer keeps the pinned grouping's column"
            );
            for (decl, ids_of) in [(0usize, &extra_ids[0]), (1usize, &extra_ids[1])] {
                let want_col = (g.extra_offsets[decl] + ids_of[i] as usize - k_family) as u32;
                assert_eq!(
                    cross_col_dual[i * g_cap + decl],
                    want_col,
                    "i={i} slot {decl}: declaration order"
                );
            }
            assert_eq!(cross_val_dual[i * g_cap].v, 0.0, "i={i}: pinned θ is 0");
            assert_eq!(cross_val_dual[i * g_cap + 1].v, 0.7, "i={i}: active θ");

            packed_m_theta_deriv(
                &g,
                cross1_theta,
                &params,
                false,
                &z_buf,
                &extra_ids,
                &ids,
                i,
                &mut m_core_d,
                &mut cvd,
            );
            assert_eq!(
                (cvd[0], cvd[1]),
                (1.0, 0.0),
                "i={i}: the pinned grouping owns slot 0 on the dual arm"
            );
            packed_m_theta_deriv(
                &g,
                cross2_theta,
                &params,
                false,
                &z_buf,
                &extra_ids,
                &ids,
                i,
                &mut m_core_d,
                &mut cvd,
            );
            assert_eq!(
                (cvd[0], cvd[1]),
                (0.0, 1.0),
                "i={i}: the active grouping keeps slot 1 on the dual arm"
            );
        }
    }
}
