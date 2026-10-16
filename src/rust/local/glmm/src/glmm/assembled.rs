//! Assembles the Laplace gradient `D*_γ` analytically, as `f64` row passes
//! over the quantities the exact β-profile's own passes already form, in place
//! of a second-order (`HyperDual`) pass that differentiates the objective
//! twice.
//!
//! **The objective is an explicit `F`/`G` pair, never `F_u = 0`.** With
//! `γ = (θ, β)`, `u` the RE vector, `η_i = offset_i + x_i'β + m_i(θ)'u`:
//!
//! ```text
//!   D(γ,u)   = Σ_i prior_w_i · dev_resid(y_i, μ(η_i))        the raw deviance
//!   A(γ,u)   = M'WM + I,  W = diag(prior_w_i · w(η_i))       the curvature: the Fisher
//!                                                             weight, or W_obs where the
//!                                                             exact curvature differs
//!   F(γ,u)   = D + ‖u‖² + log|A|                             the Laplace objective
//!   G(γ,u)   = D_u + 2u                                      what PIRLS solves: G = 0
//!   D*(γ)    = F(γ, û(γ))
//!   D*_γ     = F_γ − adj'·G_γ,   adj = G_u⁻ᵀ F_u             (one adjoint solve)
//! ```
//!
//! `G = 0` is what PIRLS delivers on every family, so this form needs no
//! special case. Gamma is differentiated at its fitted dispersion φ̂, where the
//! objective is exactly this one on the prior weights `prior_wᵢ/φ̂`
//! (`se::joint_hessian_cov` puts those weights in place); the dispersion's own
//! row of the joint Hessian is appended there.
//!
//! Every row's deviance, score and weights are smooth functions of η, a tail
//! row's included (`family::in_tail`), so the kernel's score is the deviance's
//! exact slope on every row. One shape still breaks `G = 0` and is refused: a
//! row on the link's own η bound, where the score has stopped moving to first
//! order relative to what this engine assumes — spelled out at the census in
//! [`joint_hessian_columns`].
//!
//! **Evaluation point.** Every PIRLS variant leaves `eta`, `prob`, `w`, the
//! factor, `dev` and `log|A|` at the returned iterate `u`, which is also where
//! the returned `pen = ‖u‖²` is read, so the Laplace objective is a
//! single-point function and this module differentiates it there: every
//! η-dependent quantity (the residual, the Fisher and observed weights, the
//! per-row leverage, the factor) is read at `u`, and `u_prev` names no
//! evaluation point. Reading any of them one iterate back answers a nearby but
//! different question and cannot reach a tight gradient tolerance on a cell
//! where the last PIRLS step is not already at round-off.
//!
//! Skaug, H.J. and Fournier, D.A. (2006), "Automatic approximation of the
//! marginal likelihood in non-Gaussian hierarchical models," *Computational
//! Statistics & Data Analysis*, 51(2), 699–709.
//!
//! Kristensen, K., Nielsen, A., Berg, C.W., Skaug, H., and Bell, B.M. (2016),
//! "TMB: Automatic Differentiation and Laplace Approximation," *Journal of
//! Statistical Software*, 70(5).
//!
//! Giles, M.B. (2008), "Collected Matrix Derivative Results for Forward and
//! Reverse Mode Algorithmic Differentiation," in *Advances in Automatic
//! Differentiation*, Springer.
//!
//! Magnus, J.R. and Neudecker, H. (2019), *Matrix Differential Calculus with
//! Applications in Statistics and Econometrics*, 3rd ed., Wiley.

use faer::{Mat, MatRef};

use crate::lmm::LmmGroupings;
use crate::scalar::Scalar;
use crate::spec::{BinomialLink, Family};

use super::derivative::{
    agq_eligible, supports_shape, DerivStatus, GlmmDualBufs, GlmmDualScratch, NLanes, Seed,
    MAX_DUAL_REFINEMENTS,
};
use super::deviance::{blocked_laplace_deviance, structured_laplace_deviance};
use super::pirls::{
    block_forward_solve, fill_m_vals, obs_len, packed_m_vals_theta_deriv, pirls_solve_packed,
    stream_fill, structured_ainv_solve, structured_factor, structured_stream_deviance, BetaStep,
    StreamRows, TailKernel,
};
use super::workspace::{
    glmm_block_chol, glmm_block_solve, glmm_block_solve_sub, packed_m_theta_deriv, tail_components,
    GlmmLayout, GlmmWorkspace, StructuredPattern, StructuredSchur,
};
// Named only in `gradient_f64`, which destructures `ws.pirls`/`ws.structured`
// field-by-field.
use super::workspace::{PirlsScratch, StructuredScratch};

/// Everything the assembly writes that is longer than a stack array, sized
/// once per model shape. Sitting on the dual scratch rather than allocated per
/// call, because the assembled Hessian runs one assembly per chunk on the warm
/// path and a per-call `Vec` there is a per-fit allocation.
///
/// `G_γ` is kept whole (`m·k`) rather than one `k`-length column reused per
/// coordinate: the row passes scatter into every coordinate's column as they
/// walk the rows, so a single column would force the loop nest inside out —
/// row-outer becomes coordinate-outer — which re-associates the `f64`
/// instantiation's sums. The footprint is the price of leaving those sums
/// alone.
///
/// Lengths, with `k = q_core·s + e` the packed RE dimension:
/// `tail_inv` `(e²).max(1)`, `tail_col` `e.max(1)`, the tail-component
/// scratch `comp_parent`/`comp_cols` `e` and `comp_ptr` `e + 1`,
/// `d_gamma`/`l_gamma` `m`, `d_u`/`l_u`/`adj` `k.max(1)`, `g_gamma`
/// `(m·k).max(1)`, the crossed-tail forms `tail_h`
/// ([`coupling_len`](super::workspace::coupling_len)) and
/// `tail_g` `(s·q_core²).max(1)`, and the observed factor `(s·q_core²).max(1)` /
/// `coupling_len` / `(e²).max(1)` through `pirls::obs_len`: sized on a
/// non-canonical link, empty on a canonical one. Every reader branches on the
/// static `is_canonical(family)`, never on row state, and the canonical arm
/// solves against the Fisher factor the mode solve already built, so a
/// canonical fit never reads them. (`DualStep`'s own observed twin, sized the
/// same way, serves the dual kernel's inexact step on a clamped canonical fit,
/// which the settle loop in `run_assembled_hessian` resolves; it is not this
/// buffer.)
pub(crate) struct AssemblyBufs<T: Scalar> {
    pub(super) tail_inv: Vec<T>,
    pub(super) tail_col: Vec<T>,
    pub(super) comp_parent: Vec<u32>,
    pub(super) comp_ptr: Vec<u32>,
    pub(super) comp_cols: Vec<u32>,
    pub(super) d_gamma: Vec<T>,
    pub(super) l_gamma: Vec<T>,
    pub(super) d_u: Vec<T>,
    pub(super) l_u: Vec<T>,
    pub(super) g_gamma: Vec<T>,
    pub(super) adj: Vec<T>,
    /// `H_f = C_f S⁻¹` on cluster `f`'s coupling columns only, in `coupling`'s
    /// compact layout: `H_f[l][j]` at `q_core·coup_ptr[f] + l·|cols_f| + j`,
    /// `j` the column's position in `cols_f = coup_cols[coup_ptr[f]..coup_ptr[f + 1]]`;
    /// and `s·q_core²`, `G_f = C_f S⁻¹ C_f'` per cluster: the per-cluster forms
    /// the row passes read the crossed-tail bilinear forms off (see `assemble`).
    pub(super) tail_h: Vec<T>,
    pub(super) tail_g: Vec<T>,
    pub(super) obs_core: Vec<T>,
    pub(super) obs_coup: Vec<T>,
    pub(super) obs_schur: Vec<T>,
    pub(super) packed: PackedAsmBufs<T>,
}

/// What the packed-row route needs on top of [`AssemblyBufs`]'s row passes:
/// the state the assembly is evaluated at, and the dense `k×k` matrices that
/// stand in for the other two routes' per-cluster factors.
///
/// `A` is dense here because the layout's own `A = M'WM + I` is — every row
/// loads one level of every grouping, so there is no cluster block to solve
/// alone — and `A⁻¹` is therefore formed ONCE per assembly rather than applied
/// per row: `k³` for the inverse against `width²` per row per bilinear pair,
/// where the blocked and structured routes pay `q²/2` per row per vector
/// instead (`block_forward_solve`, and [`assemble`]'s per-cluster tail forms).
/// At `k ≤ 356` and `width ≤ 8`
/// across the corpus that trade is the cheap side by orders of magnitude.
///
/// The evaluation state (`m_vals`, `eta`, `prob`, `w`, `u`) is held here on
/// BOTH instantiations, not only the dual one: at `T = f64` the mode solve
/// leaves it in the workspace and it is copied in, so the two arms hand the
/// assembly the same slices and there is no second code path.
///
/// Lengths, with `width` the packed row width and `rows` the row capacity:
/// `a`/`a_inv` `k²` each, `obs` `k²` on a non-canonical link and empty on a
/// canonical one (every reader is a non-canonical arm), `col` `k.max(1)`, `m_vals` `rows·width`, `eta`/`prob`/`w` `rows`, `u`
/// `k.max(1)`, `m_deriv` `width`. Every one of them is empty on a non-packed
/// layout, which never reads them.
pub(crate) struct PackedAsmBufs<T: Scalar> {
    /// `k×k` row-major `A = M'WM + I`, factored in place by
    /// `workspace::glmm_block_chol`.
    pub(super) a: Vec<T>,
    /// `k×k` row-major `A⁻¹`, from `k` unit-vector solves against `a`.
    pub(super) a_inv: Vec<T>,
    /// `k×k` row-major `A_obs = M'W_obs M + I`, factored in place. Read on a
    /// non-canonical link, where it is the adjoint equation's `G_u/2`.
    pub(super) obs: Vec<T>,
    /// `k` scratch for the unit-vector solves that build `a_inv`.
    pub(super) col: Vec<T>,
    /// `rows·width` packed `M` values at the assembly's evaluation point.
    pub(super) m_vals: Vec<T>,
    pub(super) eta: Vec<T>,
    pub(super) prob: Vec<T>,
    pub(super) w: Vec<T>,
    /// `k` conditional mode the assembly differentiates at.
    pub(super) u: Vec<T>,
    /// `width` per-row `∂m/∂θ_a`. `f64` whatever `T` is: `Λ` is linear in θ,
    /// so the selection is a constant in every coordinate.
    pub(super) m_deriv: Vec<f64>,
}

impl<T: Scalar> PackedAsmBufs<T> {
    fn empty() -> PackedAsmBufs<T> {
        PackedAsmBufs {
            a: Vec::new(),
            a_inv: Vec::new(),
            obs: Vec::new(),
            col: Vec::new(),
            m_vals: Vec::new(),
            eta: Vec::new(),
            prob: Vec::new(),
            w: Vec::new(),
            u: Vec::new(),
            m_deriv: Vec::new(),
        }
    }

    /// Sized for a packed shape; `width == 0` means a non-packed layout and
    /// yields [`PackedAsmBufs::empty`]. `obs` is sized only when `observed`
    /// (`!is_canonical`): only a non-canonical fit builds `A_obs` in it.
    fn for_shape(k: usize, rows: usize, width: usize, observed: bool) -> PackedAsmBufs<T> {
        if width == 0 {
            return PackedAsmBufs::empty();
        }
        PackedAsmBufs {
            a: vec![T::ZERO; k * k],
            a_inv: vec![T::ZERO; k * k],
            obs: vec![T::ZERO; obs_len(observed, k * k)],
            col: vec![T::ZERO; k.max(1)],
            m_vals: vec![T::ZERO; rows * width],
            eta: vec![T::ZERO; rows],
            prob: vec![T::ZERO; rows],
            w: vec![T::ZERO; rows],
            u: vec![T::ZERO; k.max(1)],
            m_deriv: vec![0.0; width],
        }
    }
}

impl<T: Scalar> AssemblyBufs<T> {
    /// No buffers at all, for a scratch variant the assembly never runs on:
    /// only the `Dual` rungs reach [`joint_hessian_columns`], while the
    /// `HyperDual` rungs are the packed second-order pass's and would carry
    /// `m·k` second-order numbers — megabytes on a wide `k` — for nothing.
    /// `bufs_match_shape` checks the two cases apart, so an empty set can
    /// never be mistaken for a sized one.
    pub(crate) fn empty() -> AssemblyBufs<T> {
        AssemblyBufs {
            tail_inv: Vec::new(),
            tail_col: Vec::new(),
            comp_parent: Vec::new(),
            comp_ptr: Vec::new(),
            comp_cols: Vec::new(),
            d_gamma: Vec::new(),
            l_gamma: Vec::new(),
            d_u: Vec::new(),
            l_u: Vec::new(),
            g_gamma: Vec::new(),
            adj: Vec::new(),
            tail_h: Vec::new(),
            tail_g: Vec::new(),
            obs_core: Vec::new(),
            obs_coup: Vec::new(),
            obs_schur: Vec::new(),
            packed: PackedAsmBufs::empty(),
        }
    }

    /// Every length in one place, from the same shape terms the rest of the
    /// dual scratch is sized from. Nothing here is allocated lazily: an
    /// assembly on a shape whose crossed tail is empty still holds the
    /// `.max(1)` minimum, so the first call on any shape allocates nothing.
    /// The observed factor (`obs_core`/`obs_coup`/`obs_schur`,
    /// `packed.obs`) is sized only when `observed` (`!is_canonical(family)`):
    /// every reader branches on that static flag, so a canonical fit never
    /// reads it (`binb_q4` at `Dual<12>`: 83 B/row saved).
    ///
    /// `packed_width` is the packed-row width on `GlmmLayout::Packed` and 0 on
    /// every other layout, which allocates none of [`PackedAsmBufs`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_shape(
        m: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_core: usize,
        e: usize,
        observed: bool,
        packed_width: usize,
    ) -> AssemblyBufs<T> {
        Self::for_shape_coup(m, k, rows, s, q_core, e, observed, packed_width, None)
    }

    /// [`Self::for_shape`] with the compact coupling length of `tail_h` and
    /// `obs_coup` given (`Some`), as the dual scratch sizes it from the
    /// design's coupling pattern; `None` takes the
    /// [`coupling_len`](super::workspace::coupling_len) bound.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_shape_coup(
        m: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_core: usize,
        e: usize,
        observed: bool,
        packed_width: usize,
        coup_len: Option<usize>,
    ) -> AssemblyBufs<T> {
        let kk = k.max(1);
        // The crossed-tail reducer and the blocked/structured observed twin
        // belong to the other two routes; the packed route builds its own
        // dense `A_obs` in `PackedAsmBufs` and has no per-cluster factor to
        // reduce against, so on a packed shape these sit at their minimum
        // rather than at `e` (200 on the widest packed rung, which would be
        // megabytes of unread `Dual` elements). Mirrors the same reduction
        // `derivative::DenseTwinShape` applies to the dual PIRLS twins —
        // change together.
        let packed = PackedAsmBufs::for_shape(k, rows, packed_width, observed);
        let (s, q_core, e) = if packed_width > 0 {
            (0, 1, 0)
        } else {
            (s, q_core, e)
        };
        let ee = e.max(1);
        let coup_len = match coup_len {
            Some(len) if packed_width == 0 => len,
            _ => super::workspace::coupling_len(rows, s, q_core, e),
        };
        AssemblyBufs {
            tail_inv: vec![T::ZERO; (e * e).max(1)],
            tail_col: vec![T::ZERO; ee],
            comp_parent: vec![0; e],
            comp_ptr: vec![0; e + 1],
            comp_cols: vec![0; e],
            d_gamma: vec![T::ZERO; m],
            l_gamma: vec![T::ZERO; m],
            d_u: vec![T::ZERO; kk],
            l_u: vec![T::ZERO; kk],
            g_gamma: vec![T::ZERO; (m * k).max(1)],
            adj: vec![T::ZERO; kk],
            tail_h: vec![T::ZERO; coup_len],
            tail_g: vec![T::ZERO; (s * q_core * q_core).max(1)],
            obs_core: vec![T::ZERO; obs_len(observed, (s * q_core * q_core).max(1))],
            obs_coup: vec![T::ZERO; obs_len(observed, coup_len)],
            obs_schur: vec![T::ZERO; obs_len(observed, (e * e).max(1))],
            packed,
        }
    }

    /// Bytes held at `T = f64` by the buffers that scale with the design
    /// (`G_γ` and the crossed tail's), for the gradient search's keep-or-release
    /// decision after a fit.
    pub(crate) fn f64_bytes(&self) -> usize {
        8 * (self.g_gamma.len()
            + self.tail_inv.len()
            + self.tail_h.len()
            + self.tail_g.len()
            + self.obs_core.len()
            + self.obs_coup.len()
            + self.obs_schur.len())
    }

    /// The packed state's `rows·width`, 0 on every non-packed shape — the one
    /// length in [`PackedAsmBufs`] nothing else pins, so it is what
    /// `derivative::bufs_match_shape` checks the sub-struct by.
    pub(crate) fn packed_len(&self) -> usize {
        self.packed.m_vals.len()
    }

    /// The packed observed twin's `k²`, 0 on a canonical link and on every
    /// non-packed shape — the sub-struct's other unpinned length, checked
    /// beside [`AssemblyBufs::packed_len`].
    pub(crate) fn packed_obs_len(&self) -> usize {
        self.packed.obs.len()
    }
}

/// The `f64` PIRLS mode solve every assembled derivative request starts from:
/// `BetaStep::Fixed` (the objective stays a function of the caller's β), at
/// `tol`, with THROWAWAY counters — a solve run for a derivative request never
/// reaches `ws.counters`, so the `pirls_hist`-sum == `n_eval` invariant keeps
/// holding. Mirrors `laplace_gradient`'s own mode solve.
///
/// Leaves the mode in `ws.pirls.u` and the η-dependent state (`eta`, `prob`, `w`,
/// `mu`, the block factors, `beta_rhs`) at the solve's values, not the
/// caller's. Returns false when the solve did not converge or the objective is
/// not finite; the caller restores `ws.pirls.u` either way.
#[allow(clippy::too_many_arguments)]
fn mode_solve_f64(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    tol: f64,
) -> bool {
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    // Fixed-mode β transient, as `laplace_deviance` copies it.
    ws.beta_rhs[..p].copy_from_slice(&ws.params[n_theta..m]);

    let GlmmWorkspace {
        groupings,
        params: prm,
        beta_rhs,
        z_buf,
        prior_w,
        pirls,
        structured,
        pattern,
        wx,
        offset: offset_field,
        ..
    } = ws;
    let offset = offset_field.as_deref();
    let g = &*groupings;
    let extras = !g.extra_offsets.is_empty();
    let mut mode_counters = crate::counters::EvalCounters::new();
    if extras {
        // Keep the mode solve off the cached sparse factor — see
        // `laplace_gradient`'s structured mode solve (`derivative.rs`) for
        // why, and why `pattern.structured_schur`/`force_dense_schur` are
        // taken out and restored around this one f64 call.
        let saved_schur = pattern.structured_schur.take();
        let saved_force_dense = pattern.force_dense_schur;
        pattern.force_dense_schur = false;
        let (dev, _pen_dev, conv, _raw_finite) = structured_laplace_deviance::<f64>(
            family,
            nb_theta,
            g,
            &prm[..m],
            z_buf,
            extra_ids,
            cluster_ids,
            pirls,
            structured,
            pattern,
            x,
            y,
            &prior_w[..n],
            weighted,
            beta_rhs,
            BetaStep::Fixed,
            None,
            wx,
            offset,
            Some(tol),
            n,
            &mut mode_counters,
        );
        pattern.structured_schur = saved_schur;
        pattern.force_dense_schur = saved_force_dense;
        conv && dev.is_finite()
    } else {
        let (dev, _pen_dev, conv, _raw_finite) = blocked_laplace_deviance::<f64>(
            family,
            nb_theta,
            g,
            &prm[..m],
            beta_rhs,
            pirls,
            z_buf,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            None,
            wx,
            BetaStep::Fixed,
            offset,
            Some(tol),
            p,
            n,
            &mut mode_counters,
        );
        conv && dev.is_finite()
    }
}

/// Working-set ceiling of the packed-row route, in bytes: above it the layout
/// keeps its finite-difference stencil, which allocates one worker workspace
/// per thread and no `k×k` dual matrix at all.
///
/// The footprint is a function of the shape alone, so this is arithmetic and
/// not a measurement. Writing `d = (1 + MAX_DUAL_N)·8` for one `Dual` element,
/// `k` the RE dimension, `n` the rows, `width` the packed row width and
/// `m = n_θ + p`, the engine holds
///
/// ```text
///   set          3·k²  +  n·width  +  7n  +  5k  +  m·k  +  2m
///                A, A⁻¹, A_obs   packed M   η,μ,W,   û, D_u,   G_γ   D_γ,
///                                           ρ,w',    ℓ_u, adj,        ℓ_γ
///                                           w_obs,h  col
///   peak bytes   = (d + 8)·set  +  8·(m·k + m + width + 2k)
/// ```
///
/// — one `Dual` set and one `f64` set of the same shape terms, plus the `f64`
/// half's own `U` (`m·k`, `û`'s first-order response), gradient (`m`),
/// `∂m/∂θ` row scratch (`width`) and two mode snapshots (`2k`).
/// [`packed_peak_bytes`] is that expression, and it is now an upper bound in
/// two terms: `A_obs` is sized only on a non-canonical link (`k²` less on a
/// canonical one), and the assembly no longer stores `ρ`, `w'`, `w_obs`, `h`
/// per row (its row passes are fused, so `4n` less). Both stay in the
/// expression on purpose: a smaller figure would let more packed shapes pass
/// this ceiling and so change which SE path they take, a routing change left
/// to its own study. What it leaves out
/// is the crossed-tail reducer and the blocked/structured observed twin,
/// which a packed shape sizes at their `.max(1)` minimum — six elements, not
/// a term.
///
/// Corpus-wide the largest is `sim_sparse_binomial_bigsd` (`k = 356`,
/// `n = 3600`, `width = 8`, `m = 11`, canonical) at 47.02 MiB, then
/// `sim_sparse_gamma` (`k = 220`, `n = 1200`, `width = 6`, non-canonical,
/// already at the `3·k²` figure) at 17.83 MiB; every other packed shape in
/// the corpus is under 2.3 MiB. 256 MiB clears the worst of those by ~5.4×,
/// which at the dominant `k²` term is `k ≈ 880` at that row count on every
/// link, canonical or not, and still refuses a
/// design whose dense `k×k` at `Dual<MAX_DUAL_N>` alone would run to a
/// gigabyte.
pub(crate) const PACKED_ASSEMBLY_MAX_BYTES: usize = 256 << 20;

/// The peak-bytes expression [`PACKED_ASSEMBLY_MAX_BYTES`] documents, for one
/// packed shape.
pub(crate) fn packed_peak_bytes(k: usize, n: usize, width: usize, m: usize) -> usize {
    let dual = (1 + super::derivative::MAX_DUAL_N) * 8;
    let per_set = 3 * k * k + n * width + 7 * n + 5 * k + m * k + 2 * m;
    (dual + 8) * per_set + 8 * (m * k + m + width + 2 * k)
}

/// Routing gate shared by every entry point here: false on a shape with no
/// exact derivative, on an AGQ-routed shape (outside this assembly's scope),
/// and on a packed-row shape whose working set is over
/// [`PACKED_ASSEMBLY_MAX_BYTES`].
///
/// One of the TWO owners of the layout question. This one answers "does the
/// assembled engine run here", and it is what `se::joint_hessian_cov`'s exact
/// branch reads alongside `derivative::supports_shape`. That other owner
/// answers a different question — "is there a dual twin of this layout's
/// PIRLS kernel" — and stays
/// false on `GlmmLayout::Packed`, which is why `laplace_gradient` and
/// `laplace_hessian` keep refusing a packed workspace while this engine takes
/// it. Do not collapse the two.
/// `n` is the CALLER's row count, not the workspace's row capacity: a
/// workspace may be sized for a larger `max_n` and handed a shorter fit, and
/// the guard has to answer the same way for one model however it is reached.
pub(crate) fn assembly_routes(ws: &GlmmWorkspace, n: usize) -> bool {
    let extras = !ws.groupings.extra_offsets.is_empty();
    let layout_ok = if ws.layout == GlmmLayout::Packed {
        packed_peak_bytes(ws.k, n, ws.packed.width, ws.n_theta + ws.p) <= PACKED_ASSEMBLY_MAX_BYTES
    } else {
        supports_shape(ws.layout, &ws.groupings)
    };
    layout_ok && (extras || !agq_eligible(ws.family, ws.nagq, ws.groupings.primary_q))
}

/// Rows where η sits on one of `family::clamp_eta`'s bounds at the mode state
/// held in `eta`, counted over the `n` fitted rows. On such a row the clamped
/// η is a constant, so μ, the score and the weight stop depending on `u` and
/// γ there, which the assembly's per-row derivatives do not model. Every entry
/// point here refuses a fit with a non-zero count.
pub(crate) fn eta_clamped_rows(family: Family, eta: &[f64]) -> usize {
    let (eta_lo, eta_hi) = crate::family::clamp_eta_bounds(family);
    eta.iter().filter(|&&e| e <= eta_lo || e >= eta_hi).count()
}

/// The Laplace gradient `dD*/dγ` assembled in `f64`, `γ = [θ | β]`, written
/// into `grad[..n_θ + p]`.
///
/// The `f64` reference instantiation of the assembly, and the only one that
/// owns its scratch: the shape-sized [`AssemblyBufs`] every other caller reads
/// off the dual scratch has no `f64` variant there.
///
/// Two callers: the NB dispersion row of the joint Hessian
/// (`se::joint_hessian_cov_at`, `DispRow::Nb`), which differences this exact
/// gradient in `ln θ_NB`, and the tests, which check the assembly at
/// `T = f64` against the dual-lane gradient (`laplace_gradient`) as an
/// independent way of differentiating the same objective once. The rest of
/// production reaches this assembly through [`joint_hessian`].
///
/// Solves the conditional mode once, then runs three row passes over the
/// buffers the solve left behind (no second solve, no dual lanes):
///
/// ```text
///   pass 1   the θ-free row quantities: ρ, w', w_obs, h = m'A⁻¹m,
///            and the β halves of D_γ, ℓ_γ, G_γ plus all of D_u, ℓ_u
///   pass 2   the θ halves, one ∂m/∂θ_a per row per coordinate
///   pass 3   adj = G_u⁻¹F_u (one solve against the observed factor),
///            then D*_γ = F_γ − adj'G_γ
/// ```
///
/// `None` — the caller falls back — on a shape with no exact derivative, on an
/// AGQ-routed shape, on a mode solve that does not converge, and on a
/// non-positive-definite observed factor, where the adjoint equation `G_u·adj
/// = F_u` has no Cholesky. A non-PD observed factor is never silently replaced
/// by the Fisher one: that would make the answer approximate with no detector.
///
/// The workspace comes back as found: `u` is snapshotted before the mode solve
/// and restored after, and the solve's PIRLS counters are thrown away so the
/// `pirls_hist`-sum == `n_eval` invariant keeps holding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_f64(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    // Per-row extra-grouping level ids, the same slice `laplace_deviance`
    // takes — read only when `groupings.extra_offsets` is non-empty.
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> Option<()> {
    gradient_f64_with_dev(ws, x, y, cluster_ids, extra_ids, p, n, grad, None)
}

/// [`gradient_f64`], also writing `D_γ` into `dev_grad[..n_θ + p]` when it is
/// given: the partial of the data deviance `D(η(γ, û))` at the fixed mode,
/// which pass 1 and 2 form on the way to `D*_γ`. The penalty `‖u‖²` does not
/// depend on `γ` and `û` minimizes the penalized deviance `P = D + ‖u‖²`, so
/// `D_γ` is the total derivative `∇P̂` of the penalized deviance at the mode
/// (envelope theorem). The mixed-Gamma `ln φ` row of the joint Hessian reads it
/// (`se::joint_hessian_cov_at`, `DispRow::Gamma`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_f64_with_dev(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
    dev_grad: Option<&mut [f64]>,
) -> Option<()> {
    // At the fitted Gamma dispersion, differentiate the fixed-φ GLMM the fit
    // minimized (`GlmmWorkspace::at_fixed_dispersion`); inside, `gamma_phi` is 1
    // and this guard does not fire again.
    if ws.dispersion_scaled() {
        return ws.at_fixed_dispersion(n, |ws| {
            gradient_f64_with_dev(ws, x, y, cluster_ids, extra_ids, p, n, grad, dev_grad)
        });
    }
    // `assembly_routes` also admits the packed layout, whose engine is
    // `packed_gradient`; `assemble` below reads blocked/structured buffers.
    if !assembly_routes(ws, n) || ws.layout == GlmmLayout::Packed {
        return None;
    }
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let k = ws.k;
    let kk = k.max(1);
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let extras = !ws.groupings.extra_offsets.is_empty();
    // Never the fit's own exit tolerance: the caller's override if it set one,
    // the derivative tolerance otherwise. Mirrors `laplace_gradient`.
    let tol = ws
        .fd
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));
    let saved_u: Vec<f64> = ws.pirls.u[..kk].to_vec();
    if !mode_solve_f64(ws, x, y, cluster_ids, extra_ids, p, n, tol) {
        ws.pirls.u[..kk].copy_from_slice(&saved_u);
        return None;
    }
    // The clamped mode state `joint_hessian_columns` refuses, refused here for
    // the same reason and stated there.
    if eta_clamped_rows(family, &ws.pirls.eta[..n]) > 0 {
        ws.pirls.u[..kk].copy_from_slice(&saved_u);
        return None;
    }

    let GlmmWorkspace {
        groupings,
        params: prm,
        z_buf,
        prior_w,
        pirls:
            PirlsScratch {
                m_buf,
                eta,
                prob,
                w,
                u,
                a_blocks,
                ..
            },
        structured:
            StructuredScratch {
                core_blocks,
                coupling,
                schur_blk,
                m_core_buf,
                cross_theta,
            },
        pattern:
            StructuredPattern {
                cross_col,
                n_cross,
                coup_cols,
                coup_ptr,
                coup_pos,
                ..
            },
        ..
    } = ws;
    let g = &*groupings;
    let mut asm = AssemblyBufs::<f64>::for_shape(
        m,
        k,
        n,
        g.n_primary,
        g.primary_q + g.nested_per_parent,
        g.k_crossed(),
        !crate::family::is_canonical(family),
        0,
    );

    let out = assemble(
        g,
        family,
        nb_theta,
        x,
        y,
        cluster_ids,
        extra_ids,
        &prm[..m],
        z_buf,
        &prior_w[..n],
        eta,
        prob,
        w,
        u,
        if extras { m_core_buf } else { m_buf },
        if extras { core_blocks } else { a_blocks },
        coupling,
        schur_blk,
        cross_theta,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_pos,
        n_theta,
        p,
        n,
        &mut asm,
        grad,
        None,
        false,
        None,
    );
    if let (Some(()), Some(dg)) = (out, dev_grad) {
        dg[..m].copy_from_slice(&asm.d_gamma[..m]);
    }
    u[..kk].copy_from_slice(&saved_u); // restore — leave ws.pirls.u as found
    out
}

/// The Laplace gradient `dD*/dγ`, `γ = [θ | β]`, at the mode state the LAST
/// value evaluation left, with no mode solve: the gradient search's half of
/// "one mode solve per point". Written into `grad[..n_θ + p]`.
///
/// `laplace_deviance` with `BetaMode::Fixed` at full PIRLS tolerance at the
/// point `params` leaves `pirls.{eta, prob, w, u}`, the per-cluster factors
/// (`a_blocks`; `core_blocks` and `coupling` on the structured layout) and the
/// packed `M` rows at the returned iterate û, which is exactly what
/// [`assemble`] reads. The caller vouches that nothing moved them since: the
/// GLMM gradient objective (`glmm::GradSearch`) calls this only right after the
/// value evaluation at the point it accepts. Where [`gradient_f64`] re-solves
/// the mode on the dense tail factor, this reads the value pass's own: on the
/// structured layout with the cached sparse Schur factor in use, `schur_blk`
/// still holds the unfactored `S` (the sparse arm only gathers from it), so
/// `S⁻¹` is inverted off that factor (`TailKernel::tail_inverse`, per tail
/// component) and the canonical adjoint solve runs through it.
///
/// Measured against [`gradient_f64`] at the same points (94 grid cells,
/// 2026-10-07): at most 6e-13 absolute and 2.6e-10 relative, and a gradient
/// costs 1.12 value evaluations (median) where the re-solve one cost 1.83.
///
/// `None` where [`gradient_f64`] refuses: a clamped η row, a non-PD observed
/// factor (non-canonical links). The caller has checked the layout (blocked,
/// or structured within the dual tail bound, never packed). On Gamma the
/// caller passes as `prior_w` the weights `wᵢ/φ` the value evaluation handed
/// PIRLS, so the result is the gradient at that fixed φ; the `ln φ` entry is
/// [`gamma_ln_phi_gradient`]'s. `asm` is the per-fit scratch, sized on the
/// first call from the workspace's row capacity and reused while the shape
/// matches, so a search allocates nothing per gradient.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_at_mode(
    g: &LmmGroupings,
    family: Family,
    nb_theta: f64,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    params: &[f64],
    z_buf: &[f64],
    prior_w: &[f64],
    pirls: &PirlsScratch<f64>,
    structured: &StructuredScratch<f64>,
    pattern: &mut StructuredPattern,
    asm: &mut Option<AssemblyBufs<f64>>,
    n_theta: usize,
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> Option<()> {
    if eta_clamped_rows(family, &pirls.eta[..n]) > 0 {
        return None;
    }
    let m = n_theta + p;
    let extras = !g.extra_offsets.is_empty();
    let e = g.k_crossed();
    let q_core = g.primary_q + g.nested_per_parent;
    let k = q_core * g.n_primary + e;
    let observed = !crate::family::is_canonical(family);
    // Row capacity, not this fit's `n`: the sizing is then the same on every
    // fit of the workspace and a loop never re-allocates.
    let rows = pirls.eta.len();
    let coup = super::workspace::coupling_len(rows, g.n_primary, q_core, e);
    let fits = |a: &AssemblyBufs<f64>| {
        a.d_gamma.len() == m
            && a.g_gamma.len() == (m * k).max(1)
            && a.adj.len() == k.max(1)
            && a.tail_inv.len() == (e * e).max(1)
            && a.tail_h.len() == coup
            && a.tail_g.len() == (g.n_primary * q_core * q_core).max(1)
            && a.obs_core.len() == obs_len(observed, (g.n_primary * q_core * q_core).max(1))
    };
    if !asm.as_ref().is_some_and(fits) {
        *asm = Some(AssemblyBufs::<f64>::for_shape(
            m,
            k,
            rows,
            g.n_primary,
            q_core,
            e,
            observed,
            0,
        ));
    }
    let asm = asm.as_mut().expect("sized above");
    let PirlsScratch {
        m_buf,
        eta,
        prob,
        w,
        u,
        a_blocks,
        ..
    } = pirls;
    let StructuredScratch {
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_theta,
    } = structured;
    let StructuredPattern {
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_pos,
        structured_schur,
        force_dense_schur,
        ..
    } = pattern;
    // The factor the value pass used: the sparse one exactly when the
    // structured solve took it (`Some(ss) if !force_dense`, blocked_extras.rs).
    let mut ss = if extras && e > 0 && !*force_dense_schur {
        structured_schur.as_mut()
    } else {
        None
    };
    let tail_ready = if let Some(ss) = ss.as_deref_mut() {
        <f64 as TailKernel>::tail_inverse(
            schur_blk,
            e,
            Some(ss),
            &mut asm.tail_col[..e],
            &mut asm.tail_inv,
        );
        true
    } else {
        false
    };
    assemble(
        g,
        family,
        nb_theta,
        x,
        y,
        cluster_ids,
        extra_ids,
        &params[..m],
        z_buf,
        &prior_w[..n],
        eta,
        prob,
        w,
        u,
        if extras { m_core_buf } else { m_buf },
        if extras { core_blocks } else { a_blocks },
        coupling,
        schur_blk,
        cross_theta,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_pos,
        n_theta,
        p,
        n,
        asm,
        grad,
        None,
        tail_ready,
        ss,
    )
}

/// [`gradient_at_mode`] on a whole workspace, at `ws.params`: the tests'
/// entry. `None` also on a shape the assembly does not route (packed, AGQ,
/// past the dual tail bound). On a dispersion-scaled workspace it reads the
/// weights `prior_wᵢ/φ` the value evaluation left in `prior_w_phi`, and
/// `prior_w` itself at φ = 1, as `glmm::GradSearch` does.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_at_mode_ws(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> Option<()> {
    if !assembly_routes(ws, n) || ws.layout == GlmmLayout::Packed {
        return None;
    }
    let scaled = ws.dispersion_scaled();
    let GlmmWorkspace {
        groupings,
        family,
        nb_theta,
        n_theta,
        params,
        z_buf,
        prior_w,
        prior_w_phi,
        pirls,
        structured,
        pattern,
        grad_asm,
        ..
    } = ws;
    let m = *n_theta + p;
    gradient_at_mode(
        groupings,
        *family,
        *nb_theta,
        x,
        y,
        cluster_ids,
        extra_ids,
        &params[..m],
        z_buf,
        if scaled { prior_w_phi } else { prior_w },
        pirls,
        structured,
        pattern,
        grad_asm,
        *n_theta,
        p,
        n,
        grad,
    )
}

/// The `ψ = ln φ` entry of the Gamma Laplace gradient at fixed `(θ, β)`:
///
/// ```text
///   ∂F/∂ψ = Gₚ'(ψ) − P_φ − ½·θ·∇_θF
/// ```
///
/// with `F = D/φ + ‖û‖² + log|A| + Gₚ(ψ)` and `P_φ = D/φ + ‖û‖²` (`pen_dev`,
/// the value evaluation's `pirls.pen_dev`). In `ξ = θ·e^{−ψ/2}` the objective
/// is `e^{−ψ}·P(ξ, β) + log|A|(ξ, β) + Gₚ(ψ)`, `P` and `log|A|` on the φ = 1
/// weights, so `∂F/∂ψ` at fixed ξ is `Gₚ'(ψ) − P_φ`. Holding θ instead moves
/// ξ as `∂ξ/∂ψ = −ξ/2`, and `ξ·∇_ξF = θ·∇_θF` because Λ is linear in θ; so
/// `grad_theta` is the θ block of [`gradient_at_mode`] at the same φ, in the
/// units of `theta`, and any linear θ parametrization gives the same dot
/// product.
///
/// `weights` and `n` are as `family::gamma_dispersion_term` takes them.
pub(crate) fn gamma_ln_phi_gradient(
    ln_phi: f64,
    weights: Option<&[f64]>,
    n: usize,
    pen_dev: f64,
    theta: &[f64],
    grad_theta: &[f64],
) -> f64 {
    let theta_dot: f64 = theta.iter().zip(grad_theta).map(|(t, g)| t * g).sum();
    crate::family::gamma_dispersion_term_d1(ln_phi, weights, n) - pen_dev - 0.5 * theta_dot
}

/// The joint Laplace Hessian `d²D*/dγdγ'` at the parameters in `ws.params`,
/// written into `hess[..m, ..m]`, and the gradient the same passes produce,
/// written into `grad[..m]`.
///
/// [`joint_hessian_columns`] does the work; this wrapper adds the
/// symmetrization. Every column block is exact, but a pair `H_ij`, `H_ji`
/// comes out of two different chunks and so agrees only to round-off, unlike a
/// packed second-order pass whose triangle is symmetric by construction. The
/// returned matrix is `(H + H')/2`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_hessian(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let m = ws.n_theta + p;
    let st = joint_hessian_columns(ws, x, y, cluster_ids, extra_ids, p, n, grad, hess);
    if matches!(st, DerivStatus::Ok(_)) {
        for i in 0..m {
            for j in 0..i {
                let v = 0.5 * (hess[(i, j)] + hess[(j, i)]);
                hess[(i, j)] = v;
                hess[(j, i)] = v;
            }
        }
    }
    st
}

/// The Hessian columns as the chunks produce them, unsymmetrized — the pass
/// [`joint_hessian`] wraps.
///
/// One `Dual<N>` kernel call per chunk of at most `N` seeded coordinates, then
/// one assembly on the buffers that call left behind: the assembled gradient's
/// value part is `D*_γ` and its lane `j` is `∂²D*/∂γ∂γ_{base+j}`, so a chunk
/// fills `width` whole columns at once. An `m` above the lane ladder's top
/// rung chunks, which a packed second-order pass cannot do — a cross block
/// there needs both of its coordinates' first-order lanes live in one call.
///
/// `Unsupported` — the caller falls back — on a shape with no exact
/// derivative, on an AGQ-routed shape, and on a non-positive-definite observed
/// factor in the assembly's own adjoint solve. A kernel call that reports
/// `!exact` is not itself a refusal: `run_assembled_hessian`'s settle loop
/// re-enters the kernel from the returned `u` until the assembled columns
/// stop moving. `NotConverged` is a real failure at the accepted point, and
/// also what the settle loop returns if its cap is reached without the
/// columns settling.
///
/// The workspace comes back as found: `u` is snapshotted into the dual
/// scratch's own mode buffers before the `f64` mode solve and restored after,
/// and that solve's PIRLS counters are thrown away so the `pirls_hist`-sum ==
/// `n_eval` invariant keeps holding. `eta`, `prob`, `w`, `mu`, `beta_rhs` and
/// the block factors are left at the mode solve's values, exactly as
/// `laplace_gradient` leaves them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn joint_hessian_columns(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    // At the fitted Gamma dispersion, differentiate the fixed-φ GLMM the fit
    // minimized (`GlmmWorkspace::at_fixed_dispersion`); inside, `gamma_phi` is 1
    // and this guard does not fire again.
    if ws.dispersion_scaled() {
        return ws.at_fixed_dispersion(n, |ws| {
            joint_hessian_columns(ws, x, y, cluster_ids, extra_ids, p, n, grad, hess)
        });
    }
    if !assembly_routes(ws, n) {
        return DerivStatus::Unsupported;
    }
    // The packed-row layout's own driver: same `F`/`G` pair and the same
    // chunked first-order lanes, over a dense `k×k` `A` instead of the
    // per-cluster factors the body below reads.
    if ws.layout == GlmmLayout::Packed {
        return packed_joint_hessian_columns(ws, x, y, cluster_ids, p, n, grad, hess);
    }
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let k = ws.k;
    let kk = k.max(1);
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    // Never the fit's own exit tolerance: the caller's override if it set one,
    // the derivative tolerance otherwise. Mirrors `laplace_gradient`.
    let tol = ws
        .fd
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));
    // The FIRST-derivative ladder: the assembly reads first-order lanes only,
    // so a Hessian request here resolves to a `Dual` rung and chunks like a
    // gradient does, on the rung `pick_assembled`'s measured table gives.
    let nl = NLanes::pick_assembled(m);
    let (s, q_p, q_core, e, nagq) = (
        ws.groupings.n_primary,
        ws.groupings.primary_q,
        ws.groupings.primary_q + ws.groupings.nested_per_parent,
        ws.groupings.k_crossed(),
        ws.nagq,
    );
    // Reuse policy, and the scratch itself, are `laplace_gradient`'s: a
    // request at the same `(order, N)` and shape reuses the stored scratch, a
    // different one reallocates once.
    // Same twin sizing/pinning `laplace_gradient` uses.
    let observed = !crate::family::is_canonical(family);
    let extras = !ws.groupings.extra_offsets.is_empty();
    // Above one block of rows the per-row dual values never exist whole. On
    // the blocked layout the pass runs block by block of whole clusters (the
    // row-blocked pass); on the structured layout, whose crossed Schur couples
    // every cluster, the solve and the assembly stream their row values block
    // by block while the cluster- and pair-level accumulators stay whole (the
    // streamed pass). Below one block, the whole-data pass runs unchanged.
    let block_rows = ws.fd.se_row_block;
    let coup_pairs = super::derivative::dual_coupling_pairs(&ws.pattern, s);
    let row_blocked = ws.layout == GlmmLayout::Blocked && !extras && n > block_rows;
    let streamed = ws.layout == GlmmLayout::Structured && extras && n > block_rows;
    let need_build = ws.dual_scratch.as_deref().is_none_or(|sc| {
        sc.lanes() != nl
            || !if row_blocked {
                sc.matches_shape_row_blocked(m, p, k, n, s, q_p, nagq, observed)
            } else if streamed {
                sc.matches_shape_streamed(
                    m, p, k, n, s, q_p, q_core, e, nagq, observed, coup_pairs, block_rows,
                )
            } else {
                sc.matches_shape(
                    m, p, k, n, s, q_p, q_core, e, extras, nagq, observed, 0, coup_pairs,
                )
            }
    });
    if need_build {
        // Drop the old scratch before sizing the new one, so the peak is not
        // old plus new.
        ws.dual_scratch = None;
        ws.dual_scratch = Some(Box::new(if row_blocked {
            GlmmDualScratch::for_shape_row_blocked(
                nl,
                m,
                p,
                k,
                n,
                s,
                q_p,
                nagq,
                observed,
                cluster_ids,
                block_rows,
            )
        } else if streamed {
            GlmmDualScratch::for_shape_streamed(
                nl,
                m,
                p,
                k,
                n,
                s,
                q_p,
                q_core,
                e,
                nagq,
                observed,
                cluster_ids,
                coup_pairs,
                block_rows,
            )
        } else {
            GlmmDualScratch::for_shape(
                nl,
                m,
                p,
                k,
                n,
                s,
                q_p,
                q_core,
                e,
                extras,
                nagq,
                observed,
                // The packed layout left through its own driver above.
                0,
                cluster_ids,
                coup_pairs,
            )
        }));
    }

    // Moved out of the workspace for the duration: the `f64` mode solve below
    // borrows `ws` whole, and the mode snapshot it reads and writes lives on
    // this scratch. Put back on every return path.
    let mut scratch = ws
        .dual_scratch
        .take()
        .expect("just built or confirmed present above");
    scratch.mode_bufs_mut().saved_u[..kk].copy_from_slice(&ws.pirls.u[..kk]);
    let mode_ok = mode_solve_f64(ws, x, y, cluster_ids, extra_ids, p, n, tol);
    {
        let mode = scratch.mode_bufs_mut();
        if !mode_ok {
            ws.pirls.u[..kk].copy_from_slice(&mode.saved_u[..kk]);
            ws.dual_scratch = Some(scratch);
            return DerivStatus::NotConverged;
        }
        mode.u_mode[..k].copy_from_slice(&ws.pirls.u[..k]);
        ws.pirls.u[..kk].copy_from_slice(&mode.saved_u[..kk]); // restore — leave ws.pirls.u as found
    }

    // The census of the mode state, once, before any chunk. PIRLS iterates
    // `u ← A⁻¹[(A − I)u + M'ρ]` with `ρ` the kernel's own row score, so its
    // fixed point is `u = M'ρ`, the mode equation `G = D_u + 2u = 0` the
    // adjoint below differentiates. One thing breaks that on a mode state
    // PIRLS can actually reach: a row on the link's η bound has a score that
    // has stopped moving to first order relative to what this engine assumes.
    // A tail row (`family::in_tail`) does not: its score, weights and their
    // η-derivatives are the η forms of one smooth deviance.
    if eta_clamped_rows(family, &ws.pirls.eta[..n]) > 0 {
        ws.dual_scratch = Some(scratch);
        return DerivStatus::Unsupported;
    }

    let GlmmWorkspace {
        groupings,
        params: prm,
        z_buf,
        prior_w,
        pattern,
        wx,
        offset: offset_field,
        ..
    } = ws;
    let offset = offset_field.as_deref();
    let g = &*groupings;

    // One arm body, five instantiations — the same local-macro shape
    // `laplace_gradient` uses, so the argument list is written once.
    macro_rules! hess_arm {
        ($bufs:expr, $idx:expr, $mode:expr) => {
            if row_blocked {
                run_assembled_hessian_blocked(
                    $bufs,
                    $idx,
                    block_rows,
                    family,
                    nb_theta,
                    g,
                    x,
                    y,
                    &prior_w[..n],
                    weighted,
                    cluster_ids,
                    z_buf,
                    pattern,
                    offset,
                    &prm[..m],
                    &$mode.u_mode[..k],
                    n_theta,
                    p,
                    tol,
                    grad,
                    hess,
                )
            } else {
                run_assembled_hessian(
                    $bufs,
                    family,
                    nb_theta,
                    g,
                    x,
                    y,
                    &prior_w[..n],
                    weighted,
                    cluster_ids,
                    z_buf,
                    extra_ids,
                    pattern,
                    offset,
                    wx,
                    &prm[..m],
                    &$mode.u_mode[..k],
                    n_theta,
                    p,
                    n,
                    tol,
                    grad,
                    hess,
                )
            }
        };
    }
    let st = match &mut *scratch {
        GlmmDualScratch::D4(bufs, idx, mode) => hess_arm!(bufs, idx, mode),
        GlmmDualScratch::D5(bufs, idx, mode) => hess_arm!(bufs, idx, mode),
        GlmmDualScratch::D6(bufs, idx, mode) => hess_arm!(bufs, idx, mode),
        GlmmDualScratch::D8(bufs, idx, mode) => hess_arm!(bufs, idx, mode),
        GlmmDualScratch::D12(bufs, idx, mode) => hess_arm!(bufs, idx, mode),
        // Unreachable: this slot is written only from `pick(m, false)`, so it
        // only ever holds a `Dual` variant. The arm exists because the match
        // is exhaustive over the one shared enum; decline rather than panic.
        GlmmDualScratch::H4(..)
        | GlmmDualScratch::H5(..)
        | GlmmDualScratch::H6(..)
        | GlmmDualScratch::H8(..)
        | GlmmDualScratch::H12(..) => DerivStatus::Unsupported,
    };
    ws.dual_scratch = Some(scratch);
    st
}

/// Rows per block of the row-blocked assembled Hessian on the blocked layout
/// ([`run_assembled_hessian_blocked`]). A block is the run of consecutive
/// primary clusters that first reaches this many rows, so a block holds at
/// least this many rows (the last one may hold fewer) and never splits a
/// cluster. A fit with `n` at or below it takes the whole-data pass
/// ([`run_assembled_hessian`]) unchanged.
///
/// Measured 2026-10-07 (pawel, clock-locked P-core, and an Apple M4): every
/// size tried from 2048 to 65536 rows is faster than the whole-data pass on
/// the binomial and Gamma blocked series from 30k to 3M rows, and 4096 is the
/// fastest or within noise of it on each. The gain is cache residency and the
/// first-touch page faults the block-sized scratch no longer takes, not less
/// work: user instructions move by under 2%. Evidence: pawel
/// `~/.cache/claude-scratch/mmjl-rethink/b3/` and `.../a-impl/`.
///
/// A pinned constant, the same on every target, so a fit splits its rows the
/// same way, and its Hessian moves by the same round-off, on every host and
/// under WASM. `FdState::se_row_block` overrides it for tests only.
pub(crate) const SE_ROW_BLOCK: usize = 4096;

/// Block-sized scratch of [`run_assembled_hessian_blocked`]: one block's dual
/// PIRLS twin, assembly buffers and observed-step twin, its gathered `x`,
/// `y`, prior weights, `z`, offset and block-local cluster ids, and the block
/// list. Lives on the dual scratch (`GlmmDualBufs::row_blocks`), built once by
/// [`GlmmDualScratch::for_shape_row_blocked`], so a repeat call at the same
/// shape and cluster sizes allocates nothing. A call whose largest block
/// outgrows the stored capacity (new cluster ids with larger clusters) grows
/// it once, in [`RowBlockBufs::ensure`].
pub(crate) struct RowBlockBufs<T: Scalar> {
    /// `(first, end)` primary-cluster range of each block, rebuilt per call.
    blocks: Vec<(usize, usize)>,
    /// Row and cluster capacity the buffers below are sized for.
    cap_rows: usize,
    cap_s: usize,
    q: usize,
    p: usize,
    m: usize,
    nagq: u8,
    observed: bool,
    pirls: PirlsScratch<T>,
    asm: AssemblyBufs<T>,
    dual: super::pirls::DualStep<T>,
    /// The block's own `Σ_r ∂/∂γ` of the assembled gradient, at `T`.
    grad: Vec<T>,
    x: Mat<f64>,
    wx: Mat<f64>,
    y: Vec<f64>,
    prior_w: Vec<f64>,
    z: Vec<f64>,
    /// Empty until a call carries an offset.
    offset: Vec<f64>,
    cluster_ids: Vec<u32>,
    /// The fit's groupings with `n_primary`/`k_total` set per block. Cloned
    /// from the workspace's on the first call after a (re)build: every other
    /// field is fixed for the shape this scratch was built for, and a shape
    /// change rebuilds the scratch.
    groupings: Option<LmmGroupings>,
}

impl<T: Scalar> RowBlockBufs<T> {
    /// Sized for the blocks `idx` gives at `block_rows`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn for_shape(
        idx: &super::agq::ClusterRowIndex,
        s: usize,
        block_rows: usize,
        m: usize,
        p: usize,
        q: usize,
        nagq: u8,
        observed: bool,
    ) -> RowBlockBufs<T> {
        let mut blocks = Vec::with_capacity(idx.rows().len() / block_rows.max(1) + 1);
        let (max_rows, max_s) = plan_row_blocks(idx, s, block_rows, &mut blocks);
        let mut b = RowBlockBufs {
            blocks,
            cap_rows: 0,
            cap_s: 0,
            q,
            p,
            m,
            nagq,
            observed,
            pirls: PirlsScratch::<T>::for_shape(0, 0, q, 0, nagq),
            asm: AssemblyBufs::<T>::for_shape(m, 0, 0, 0, q, 0, observed, 0),
            dual: block_dual_step(0, 0, q, observed),
            grad: vec![T::ZERO; m],
            x: Mat::zeros(0, p),
            wx: Mat::zeros(0, p),
            y: Vec::new(),
            prior_w: Vec::new(),
            z: Vec::new(),
            offset: Vec::new(),
            cluster_ids: Vec::new(),
            groupings: None,
        };
        b.ensure(max_rows, max_s);
        b
    }

    /// Grow every block-sized buffer to hold `rows` rows and `s` clusters, if
    /// it does not already. A no-op (no allocation) on a repeat call.
    fn ensure(&mut self, rows: usize, s: usize) {
        if rows <= self.cap_rows && s <= self.cap_s {
            return;
        }
        let rows = rows.max(self.cap_rows);
        let s = s.max(self.cap_s);
        let (q, p, k) = (self.q, self.p, s * self.q);
        self.pirls = PirlsScratch::<T>::for_shape(rows, k, q, s, self.nagq);
        self.asm = AssemblyBufs::<T>::for_shape(self.m, k, rows, s, q, 0, self.observed, 0);
        self.dual = block_dual_step(rows, s, q, self.observed);
        self.x = Mat::zeros(rows, p);
        self.wx = Mat::zeros(rows, p);
        self.y = vec![0.0; rows];
        self.prior_w = vec![0.0; rows];
        self.z = vec![0.0; rows * q.saturating_sub(1)];
        if !self.offset.is_empty() {
            self.offset = vec![0.0; rows];
        }
        self.cluster_ids = vec![0; rows];
        self.cap_rows = rows;
        self.cap_s = s;
    }
}

/// The observed-information twin of one block, sized as
/// `GlmmDualScratch::for_shape` sizes the blocked layout's (`q_core = q`,
/// no crossed extras).
fn block_dual_step<T: Scalar>(
    rows: usize,
    s: usize,
    q: usize,
    observed: bool,
) -> super::pirls::DualStep<T> {
    super::pirls::DualStep {
        observed: false,
        obs_blocks: vec![T::ZERO; obs_len(observed, s * q * q)],
        obs_core_blocks: vec![T::ZERO; obs_len(observed, (q * q * s).max(1))],
        obs_coupling: vec![
            T::ZERO;
            obs_len(observed, super::workspace::coupling_len(rows, s, q, 0))
        ],
        obs_schur_blk: vec![T::ZERO; obs_len(observed, 1)],
        obs_rhs: vec![T::ZERO; obs_len(observed, s * q)],
        min_iters: 0,
        exact: false,
    }
}

/// Splits the `s` primary clusters of `idx` into blocks of consecutive
/// clusters, each closed once it holds at least `block_rows` rows, into
/// `blocks` (cleared first). Returns the largest block's row and cluster
/// counts.
fn plan_row_blocks(
    idx: &super::agq::ClusterRowIndex,
    s: usize,
    block_rows: usize,
    blocks: &mut Vec<(usize, usize)>,
) -> (usize, usize) {
    blocks.clear();
    let (mut c0, mut acc, mut max_rows, mut max_s) = (0usize, 0usize, 0usize, 0usize);
    for c in 0..s {
        acc += idx.cluster_range(c).len();
        if acc >= block_rows || c + 1 == s {
            blocks.push((c0, c + 1));
            max_rows = max_rows.max(acc);
            max_s = max_s.max(c + 1 - c0);
            c0 = c + 1;
            acc = 0;
        }
    }
    (max_rows, max_s)
}

/// The chunked seed-call-assemble body [`joint_hessian_columns`]'s per-`N`
/// match arms hand a typed buffer set to. A call that comes back `!exact`
/// means the step matrix that call took was not the Jacobian of the map PIRLS
/// walks — a non-PD observed factor inside the kernel
/// (`pirls::DualStep::exact`'s own doc comment) — so the assembled columns
/// from that call are not yet the answer.
/// The settle loop below re-enters the kernel from the same `u`, which it
/// carries forward by the kernel's own mutation of `bufs.pirls.u` rather than
/// reseeding, until the columns it writes into `hess` for this chunk stop
/// moving by more than the band `run_gradient`/`run_hessian` use, or until
/// `MAX_DUAL_REFINEMENTS` calls are spent, which is `DerivStatus::NotConverged`.
///
/// `min_iters = 3` on a chunk's first call, the same floor the packed
/// second-order pass takes. The exit test the kernel runs without it inspects
/// the objective VALUE only, and the value sits at the mode from the first
/// step, while `u`'s lanes are the fixed point of a contraction that test
/// never looks at. Three steps is what makes the first-order lanes exact and
/// then reads the objective at an iterate that carries them — and it is also
/// what puts this pass on the same iterate of the same solve as the packed
/// pass and as `laplace_gradient`, so all three differentiate the same
/// function at the same returned `u`. A later call in the settle loop takes
/// `min_iters = 0` instead, the same floor `run_hessian`'s own fallback loop
/// uses. `GlmmDualScratch::exit_mode_step` reports the distance each pass
/// exits at.
///
/// Mirrors the chunked seeding in the derivative kernel's `run_gradient` —
/// change together. The invariant behind re-seeding `u` from the `f64` mode on
/// every pass is stated in full there: the kernel moves `u`, value and lanes,
/// in place, so a pass that did not re-seed would differentiate at the
/// previous pass's moved point.
#[allow(clippy::too_many_arguments)]
fn run_assembled_hessian<T: Seed>(
    bufs: &mut GlmmDualBufs<T>,
    family: Family,
    nb_theta: f64,
    g: &LmmGroupings,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    // The `f64` workspace's shared structured pattern — see `run_gradient`'s
    // doc comment (`derivative.rs`) on the same parameter.
    pattern: &mut StructuredPattern,
    offset: Option<&[f64]>,
    wx: &mut Mat<f64>,
    ws_params: &[f64],
    u_mode: &[f64],
    n_theta: usize,
    p: usize,
    n: usize,
    tol: f64,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let m = n_theta + p;
    let k = u_mode.len();
    let extras = !g.extra_offsets.is_empty();
    let canonical = crate::family::is_canonical(family);
    let lanes = T::LANES;
    let n_chunks = m.div_ceil(lanes.max(1));
    // The streamed pass: a scratch built by
    // `GlmmDualScratch::for_shape_streamed` (structured layout above one block
    // of rows) carries the block buffers and no `n`-length dual row buffers.
    let streamed = bufs.stream_rows.is_some();
    let base_theta = g.primary_q * (g.primary_q + 1) / 2;
    let nested_decl = g.nested.map(|nf| nf.vech_start - base_theta);
    // The objective value is the same in every chunk — only the seeded lanes
    // move — so the last chunk's value is the function's answer.
    let mut last_value = f64::NAN;
    for chunk in 0..n_chunks {
        let base = chunk * lanes;
        let width = lanes.min(m - base);
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            bufs.params[j] = if j >= base && j < base + width {
                T::unit(ws_params[j], j - base)
            } else {
                T::from_f64(ws_params[j])
            };
        }
        for i in 0..p {
            bufs.beta[i] = bufs.params[n_theta + i];
        }
        #[allow(clippy::needless_range_loop)]
        for c in 0..k {
            bufs.pirls.u[c] = T::from_f64(u_mode[c]);
        }
        // The observed-information step: exact lanes in one kernel call on a
        // non-canonical link, where the Fisher `A` is only an approximation to
        // `½h_uu`. Canonical links pass `observed = false` — their `A` already
        // is that Hessian.
        bufs.dual.observed = !canonical;

        // Settles this chunk's columns. A first call that comes back `exact`
        // is accepted outright: one kernel call, one assembly. Otherwise
        // `bufs.pirls.u` carries forward from one call to the next (the
        // kernel's own mutation, never reseeded here) and the loop compares
        // this pass's columns against `hess[(r, base + j)]`, which holds the
        // previous pass's, so the comparison needs no scratch of its own.
        let mut have_prev = false;
        let mut pass_value = None;
        for pass in 0..MAX_DUAL_REFINEMENTS {
            bufs.dual.min_iters = if pass == 0 { 3 } else { 0 };
            // Throwaway: a derivative evaluation is not a fit-path PIRLS
            // solve, so it must not reach `ws.counters`.
            let mut counters = crate::counters::EvalCounters::new();
            let (value, conv) = if streamed {
                let (obj, _dev, conv) = structured_stream_deviance::<T>(
                    family,
                    nb_theta,
                    g,
                    &bufs.params[..m],
                    z_buf,
                    extra_ids,
                    cluster_ids,
                    &mut bufs.pirls,
                    &mut bufs.structured,
                    pattern,
                    x,
                    y,
                    prior_w,
                    weighted,
                    &bufs.beta[..p],
                    &mut bufs.dual,
                    offset,
                    tol,
                    n,
                    bufs.stream_rows.as_deref_mut().expect("streamed"),
                );
                (obj.value(), conv)
            } else if extras {
                let (obj, _pen_dev, conv, _raw_finite) = structured_laplace_deviance::<T>(
                    family,
                    nb_theta,
                    g,
                    &bufs.params[..m],
                    z_buf,
                    extra_ids,
                    cluster_ids,
                    &mut bufs.pirls,
                    &mut bufs.structured,
                    // `pattern` is the `f64` workspace's own pattern (shared, see
                    // `run_gradient`'s doc comment in `derivative.rs`); its
                    // `structured_schur` is harmless at a dual `T` — `TailKernel`'s
                    // default tail methods ignore it regardless.
                    pattern,
                    x,
                    y,
                    prior_w,
                    weighted,
                    &mut bufs.beta[..p],
                    BetaStep::Fixed,
                    Some(&mut bufs.dual),
                    wx,
                    offset,
                    Some(tol),
                    n,
                    &mut counters,
                );
                (obj.value(), conv)
            } else {
                let (obj, _pen_dev, conv, _raw_finite) = blocked_laplace_deviance::<T>(
                    family,
                    nb_theta,
                    g,
                    &bufs.params[..m],
                    &mut bufs.beta[..p],
                    &mut bufs.pirls,
                    z_buf,
                    x,
                    y,
                    prior_w,
                    weighted,
                    cluster_ids,
                    Some(&mut bufs.dual),
                    wx,
                    BetaStep::Fixed,
                    offset,
                    Some(tol),
                    p,
                    n,
                    &mut counters,
                );
                (obj.value(), conv)
            };
            if !conv || !value.is_finite() {
                return DerivStatus::NotConverged;
            }
            let assembled = assemble(
                g,
                family,
                nb_theta,
                x,
                y,
                cluster_ids,
                extra_ids,
                ws_params,
                z_buf,
                prior_w,
                &bufs.pirls.eta,
                &bufs.pirls.prob,
                &bufs.pirls.w,
                &bufs.pirls.u,
                if extras {
                    &bufs.structured.m_core_buf
                } else {
                    &bufs.pirls.m_buf
                },
                if extras {
                    &bufs.structured.core_blocks
                } else {
                    &bufs.pirls.a_blocks
                },
                &bufs.structured.coupling,
                &bufs.structured.schur_blk,
                &bufs.structured.cross_theta,
                &pattern.cross_col,
                &pattern.n_cross,
                &pattern.coup_cols,
                &pattern.coup_ptr,
                &pattern.coup_pos,
                n_theta,
                p,
                n,
                &mut bufs.asm,
                &mut bufs.grad_t,
                match bufs.stream_rows.as_deref_mut() {
                    Some(rows) => Some(StreamCtx {
                        rows,
                        beta: &bufs.beta[..p],
                        theta_nested: g
                            .nested
                            .map(|nf| bufs.params[nf.vech_start])
                            .unwrap_or(T::ZERO),
                        nested_decl,
                        offset,
                        weighted,
                        i0: 0,
                        next: 0,
                    }),
                    None => None,
                },
                false,
                None,
            );
            if assembled.is_none() {
                return DerivStatus::Unsupported;
            }
            // Same band shape as `run_gradient`'s and `run_hessian`'s own
            // settle checks (`derivative.rs`): relative, with an absolute
            // floor so a near-zero lane still settles.
            let settled = (pass == 0 && bufs.dual.exact)
                || (have_prev
                    && (0..m).all(|r| {
                        let d = bufs.grad_t[r].dslice();
                        (0..width).all(|j| {
                            (d[j] - hess[(r, base + j)]).abs() < 1e-10 * (1.0 + d[j].abs())
                        })
                    }));
            for r in 0..m {
                let d = bufs.grad_t[r].dslice();
                for j in 0..width {
                    hess[(r, base + j)] = d[j];
                }
            }
            if settled {
                pass_value = Some(value);
                break;
            }
            have_prev = true;
        }
        let value = match pass_value {
            Some(v) => v,
            // The cap was spent without the columns settling.
            None => return DerivStatus::NotConverged,
        };
        if chunk == 0 {
            for (r, out) in grad.iter_mut().enumerate().take(m) {
                *out = bufs.grad_t[r].value();
            }
        } else {
            debug_assert!(
                (0..m).all(|r| {
                    let v = bufs.grad_t[r].value();
                    (v - grad[r]).abs() <= 1e-12 * (1.0 + grad[r].abs())
                }),
                "every chunk differentiates the same objective at the same mode, \
                 so the value part of the assembled gradient must not move between chunks"
            );
        }
        last_value = value;
    }
    DerivStatus::Ok(last_value)
}

/// [`run_assembled_hessian`] on the blocked layout (no extra groupings), one
/// block of consecutive primary clusters ([`SE_ROW_BLOCK`] rows) at a time.
///
/// With β fixed, the blocked layout's `A`, its mode equation `G = 0` and the
/// adjoint `G_u⁻ᵀF_u` are all block-diagonal over primary clusters, and the
/// objective `D + ‖u‖² + log|A|` is a sum over clusters. So a block of whole
/// clusters runs the dual PIRLS solve and the assembly on its own rows and RE
/// columns alone, and the assembled gradient (value and seeded lanes) is the
/// sum of the blocks'. Nothing is recomputed: each block's per-row and
/// per-cluster dual values live only while that block runs, in the
/// block-sized [`RowBlockBufs`]. The only state shared across blocks is the
/// `k`-length mode `u`, which the settle loop carries forward as the
/// whole-data pass does (a block's slice of it is copied in before its solve
/// and back after).
///
/// Same seeding, settle loop and band as [`run_assembled_hessian`]; change
/// together. The results differ from the whole-data pass at round-off only:
/// each block's PIRLS exit test reads its own block's value, and the gradient
/// is summed per block (measured at most 2.6e-13 relative on the covariance at
/// 3M rows).
#[allow(clippy::too_many_arguments)]
fn run_assembled_hessian_blocked<T: Seed>(
    bufs: &mut GlmmDualBufs<T>,
    idx: &mut super::agq::ClusterRowIndex,
    block_rows: usize,
    family: Family,
    nb_theta: f64,
    g: &LmmGroupings,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    z_buf: &[f64],
    pattern: &StructuredPattern,
    offset: Option<&[f64]>,
    ws_params: &[f64],
    u_mode: &[f64],
    n_theta: usize,
    p: usize,
    tol: f64,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let Some(mut rb) = bufs.row_blocks.take() else {
        // Unreachable: the caller routes here only on a scratch built by
        // `for_shape_row_blocked`. Decline rather than panic.
        return DerivStatus::Unsupported;
    };
    let st = blocked_chunks(
        bufs,
        &mut rb,
        idx,
        block_rows,
        family,
        nb_theta,
        g,
        x,
        y,
        prior_w,
        weighted,
        cluster_ids,
        z_buf,
        pattern,
        offset,
        ws_params,
        u_mode,
        n_theta,
        p,
        tol,
        grad,
        hess,
    );
    bufs.row_blocks = Some(rb);
    st
}

/// The body of [`run_assembled_hessian_blocked`], with the block scratch moved
/// out of `bufs` so both can be borrowed at once.
#[allow(clippy::too_many_arguments)]
fn blocked_chunks<T: Seed>(
    bufs: &mut GlmmDualBufs<T>,
    rb: &mut RowBlockBufs<T>,
    idx: &mut super::agq::ClusterRowIndex,
    block_rows: usize,
    family: Family,
    nb_theta: f64,
    g: &LmmGroupings,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    cluster_ids: &[u32],
    z_buf: &[f64],
    pattern: &StructuredPattern,
    offset: Option<&[f64]>,
    ws_params: &[f64],
    u_mode: &[f64],
    n_theta: usize,
    p: usize,
    tol: f64,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let m = n_theta + p;
    let k = u_mode.len();
    let q = g.primary_q;
    let s = g.n_primary;
    let zw = q - 1;
    let canonical = crate::family::is_canonical(family);
    let lanes = T::LANES;
    let n_chunks = m.div_ceil(lanes.max(1));

    // The index the scratch was built with belongs to the cluster ids of that
    // build; a reused workspace can be called with new ones, so rebuild it in
    // place (no allocation at an unchanged `n` and `s`).
    idx.rebuild(cluster_ids, s);
    let (max_rows, max_s) = plan_row_blocks(idx, s, block_rows, &mut rb.blocks);
    rb.ensure(max_rows, max_s);
    if offset.is_some() && rb.offset.len() < rb.cap_rows {
        rb.offset = vec![0.0; rb.cap_rows];
    }
    let gb = rb.groupings.get_or_insert_with(|| g.clone());
    let all_rows = idx.rows();

    let mut last_value = f64::NAN;
    for chunk in 0..n_chunks {
        let base = chunk * lanes;
        let width = lanes.min(m - base);
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            bufs.params[j] = if j >= base && j < base + width {
                T::unit(ws_params[j], j - base)
            } else {
                T::from_f64(ws_params[j])
            };
        }
        for i in 0..p {
            bufs.beta[i] = bufs.params[n_theta + i];
        }
        #[allow(clippy::needless_range_loop)]
        for c in 0..k {
            bufs.pirls.u[c] = T::from_f64(u_mode[c]);
        }
        rb.dual.observed = !canonical;

        let mut have_prev = false;
        let mut pass_value = None;
        for pass in 0..MAX_DUAL_REFINEMENTS {
            let mut value = 0.0f64;
            let mut all_exact = true;
            for v in bufs.grad_t[..m].iter_mut() {
                *v = T::ZERO;
            }
            for bi in 0..rb.blocks.len() {
                let (b0, b1) = rb.blocks[bi];
                let sb = b1 - b0;
                let kb = sb * q;
                let r0 = idx.cluster_range(b0).start;
                let r1 = idx.cluster_range(b1 - 1).end;
                let rows = &all_rows[r0..r1];
                let nb = rows.len();
                for (t, &ri) in rows.iter().enumerate() {
                    let i = ri as usize;
                    for j in 0..p {
                        rb.x[(t, j)] = x[(i, j)];
                    }
                    rb.y[t] = y[i];
                    rb.prior_w[t] = prior_w[i];
                    rb.z[t * zw..(t + 1) * zw].copy_from_slice(&z_buf[i * zw..(i + 1) * zw]);
                    if let Some(o) = offset {
                        rb.offset[t] = o[i];
                    }
                    rb.cluster_ids[t] = cluster_ids[i] - b0 as u32;
                }
                gb.n_primary = sb;
                gb.k_total = kb;
                rb.pirls.u[..kb].copy_from_slice(&bufs.pirls.u[b0 * q..b1 * q]);
                rb.dual.min_iters = if pass == 0 { 3 } else { 0 };
                // Throwaway: a derivative evaluation is not a fit-path PIRLS
                // solve, so it must not reach `ws.counters`.
                let mut counters = crate::counters::EvalCounters::new();
                let off_b = if offset.is_some() {
                    Some(&rb.offset[..nb])
                } else {
                    None
                };
                let (obj, _pen_dev, conv, _raw_finite) = blocked_laplace_deviance::<T>(
                    family,
                    nb_theta,
                    gb,
                    &bufs.params[..m],
                    &mut bufs.beta[..p],
                    &mut rb.pirls,
                    &rb.z[..nb * zw],
                    rb.x.as_ref().subrows(0, nb),
                    &rb.y[..nb],
                    &rb.prior_w[..nb],
                    weighted,
                    &rb.cluster_ids[..nb],
                    Some(&mut rb.dual),
                    &mut rb.wx,
                    BetaStep::Fixed,
                    off_b,
                    Some(tol),
                    p,
                    nb,
                    &mut counters,
                );
                let v = obj.value();
                if !conv || !v.is_finite() {
                    return DerivStatus::NotConverged;
                }
                value += v;
                all_exact &= rb.dual.exact;
                bufs.pirls.u[b0 * q..b1 * q].copy_from_slice(&rb.pirls.u[..kb]);
                let assembled = assemble(
                    gb,
                    family,
                    nb_theta,
                    rb.x.as_ref().subrows(0, nb),
                    &rb.y[..nb],
                    &rb.cluster_ids[..nb],
                    &[],
                    ws_params,
                    &rb.z[..nb * zw],
                    &rb.prior_w[..nb],
                    &rb.pirls.eta,
                    &rb.pirls.prob,
                    &rb.pirls.w,
                    &rb.pirls.u,
                    &rb.pirls.m_buf,
                    &rb.pirls.a_blocks,
                    &bufs.structured.coupling,
                    &bufs.structured.schur_blk,
                    &bufs.structured.cross_theta,
                    &pattern.cross_col,
                    &pattern.n_cross,
                    &pattern.coup_cols,
                    &pattern.coup_ptr,
                    &pattern.coup_pos,
                    n_theta,
                    p,
                    nb,
                    &mut rb.asm,
                    &mut rb.grad,
                    None,
                    false,
                    None,
                );
                if assembled.is_none() {
                    return DerivStatus::Unsupported;
                }
                for r in 0..m {
                    bufs.grad_t[r] += rb.grad[r];
                }
            }
            // Same band as `run_assembled_hessian`'s settle check.
            let settled = (pass == 0 && all_exact)
                || (have_prev
                    && (0..m).all(|r| {
                        let d = bufs.grad_t[r].dslice();
                        (0..width).all(|j| {
                            (d[j] - hess[(r, base + j)]).abs() < 1e-10 * (1.0 + d[j].abs())
                        })
                    }));
            for r in 0..m {
                let d = bufs.grad_t[r].dslice();
                for j in 0..width {
                    hess[(r, base + j)] = d[j];
                }
            }
            if settled {
                pass_value = Some(value);
                break;
            }
            have_prev = true;
        }
        let Some(value) = pass_value else {
            return DerivStatus::NotConverged;
        };
        if chunk == 0 {
            for (r, out) in grad.iter_mut().enumerate().take(m) {
                *out = bufs.grad_t[r].value();
            }
        }
        last_value = value;
    }
    DerivStatus::Ok(last_value)
}

/// What [`assemble`]'s row loop needs on the streamed structured pass
/// ([`GlmmDualScratch::for_shape_streamed`]): there the dual PIRLS keeps no
/// `n`-length row buffers, so the loop re-forms each row block's `η`, `p`,
/// `W_obs` and core `M` slice at the returned iterate (`pirls::stream_fill`,
/// the same arithmetic as the kernel's own row pass) instead of reading them.
/// `i0` is the first row of the block in `rows`, `next` the row at which the
/// next block is formed.
pub(crate) struct StreamCtx<'a, T> {
    pub(crate) rows: &'a mut StreamRows<T>,
    pub(crate) beta: &'a [T],
    pub(crate) theta_nested: T,
    pub(crate) nested_decl: Option<usize>,
    pub(crate) offset: Option<&'a [f64]>,
    pub(crate) weighted: bool,
    pub(crate) i0: usize,
    pub(crate) next: usize,
}

/// The three row passes, on the buffers the mode solve left: `core_m` the
/// packed core `M` (`m_core_buf` with extras, `m_buf` without — `q_core == q`
/// there, so the stride is the same), `core_fac` its per-cluster Crout factor,
/// and `coupling`/`schur_blk`/`cross_*`/`coup_*` the crossed tail, unread at
/// `e == 0`. Split out of [`gradient_f64`] so the assembly reads plain slices
/// rather than a destructured workspace.
///
/// Generic over the scalar: at `T = f64` this is the gradient itself, at
/// `T = Dual<N>` its lanes are the joint Hessian's seeded columns. Every
/// third derivative the second order needs arrives as a lane of a first-order
/// quantity — `observed_weight`, `weight_eta_deriv` — so
/// there is no third-derivative table on this path.
///
/// `∂M/∂θ_a` is the one quantity lifted from `f64` rather than read off the
/// lanes: `Λ` is linear in θ, so it is a constant selection with zero lanes in
/// every coordinate, whereas the lanes of the packed `M` carry
/// `Σ_a ∂M/∂θ_a·dθ_a` — the same object only for the one seeded coordinate.
#[allow(clippy::too_many_arguments)]
fn assemble<T: TailKernel>(
    g: &LmmGroupings,
    family: Family,
    nb_theta: f64,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    // θ and β as plain `f64`, read only by `packed_m_theta_deriv` — see the
    // `∂M/∂θ_a` note above.
    params: &[f64],
    z_buf: &[f64],
    prior_w: &[f64],
    eta: &[T],
    prob: &[T],
    w: &[T],
    // The returned iterate: the point the kernel's η-dependent state, its
    // factor and the `‖u‖²` penalty all sit at, so every read here goes
    // through it.
    u: &[T],
    core_m: &[T],
    core_fac: &[T],
    coupling: &[T],
    schur_blk: &[T],
    cross_theta: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    coup_pos: &[u32],
    n_theta: usize,
    p: usize,
    n: usize,
    asm: &mut AssemblyBufs<T>,
    grad: &mut [T],
    mut stream: Option<StreamCtx<'_, T>>,
    // Set by [`gradient_at_mode`] only: `S⁻¹` is already in `asm.tail_inv`,
    // inverted off the value evaluation's sparse Schur factor, and `ss` is
    // that factor, for the canonical adjoint solve. Every other caller
    // passes `false, None` and runs exactly as before.
    tail_ready: bool,
    ss: Option<&mut StructuredSchur>,
) -> Option<()> {
    let q = g.primary_q;
    let s = g.n_primary;
    let np = g.nested_per_parent;
    let qc = q + np;
    let e = g.k_crossed();
    let k_family = qc * s;
    let k = k_family + e;
    let prim_width = q * s;
    let m = n_theta + p;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    let extras = !g.extra_offsets.is_empty();
    // Core-block-local column `local` of cluster `f` as an RE column — the
    // order `u` is stored in. The solver packing used for everything else here
    // (`[f·q_core + local | k_family + b]`) is `structured_ainv_solve`'s, and
    // the two coincide only at `np == 0`.
    let core_col = |f: usize, local: usize| -> usize {
        if local < q {
            f * q + local
        } else {
            prim_width + f * np + (local - q)
        }
    };

    // Every buffer below is the caller's shape-sized scratch, destructured
    // into disjoint field borrows so the row passes can hold several at once.
    let AssemblyBufs {
        tail_inv,
        tail_col,
        comp_parent,
        comp_ptr,
        comp_cols,
        d_gamma,
        l_gamma,
        d_u,
        l_u,
        // `G_γ` row-major, `g_gamma[a·k + c] = ∂G_c/∂γ_a`.
        g_gamma,
        adj,
        tail_h,
        tail_g,
        obs_core,
        obs_coup,
        obs_schur,
        ..
    } = asm;
    // The accumulators are `+=`d row by row, so they start at zero; `adj` is
    // assigned outright and needs no clear.
    d_gamma[..m].fill(T::ZERO);
    l_gamma[..m].fill(T::ZERO);
    d_u[..k].fill(T::ZERO);
    l_u[..k].fill(T::ZERO);
    g_gamma[..m * k].fill(T::ZERO);

    // `S⁻¹` column by column, through the same dense tail substitution the
    // mode solve's own factor was left by; column-major, `tail_inv[b·e+a] =
    // (S⁻¹)_{a,b}`, the layout the per-row reducer reads. `S` and its dense
    // `L` are block-diagonal over the tail's components, so each unit solve
    // runs on its own component only (`glmm_block_solve_sub`: the full
    // substitution minus terms that are exact zeros, so the same numbers).
    // The reducer reads `tail_inv` only on one cluster's coupling columns,
    // which lie in one component, so the entries between components are
    // never written and never read. On nested tails this is Σ e_c³ instead of
    // e³ (grouseticks: 20 ms to 0.3 ms per assembly, measured 2026-09-29).
    if e > 0 && !tail_ready {
        let nc = tail_components(coup_cols, coup_ptr, e, comp_parent, comp_ptr, comp_cols);
        for c in 0..nc {
            let cols = &comp_cols[comp_ptr[c] as usize..comp_ptr[c + 1] as usize];
            let m = cols.len();
            for (j, &b) in cols.iter().enumerate() {
                tail_col[..m].fill(T::ZERO);
                tail_col[j] = T::ONE;
                glmm_block_solve_sub(schur_blk, e, cols, &mut tail_col[..m]);
                for (i, &a) in cols.iter().enumerate() {
                    tail_inv[b as usize * e + a as usize] = tail_col[i];
                }
            }
        }
    }

    // The structured-extras system is `A = [[D, C], [C', E+I]]`: `D` block
    // diagonal over the clusters (cluster `f`'s core block `D_f`, identity
    // included), `C_f` cluster `f`'s coupling to the crossed tail, `E+I` the
    // tail's own block. `S = (E+I) − Σ_f C_f'D_f⁻¹C_f` is the tail's Schur
    // complement (`schur_blk`), handed in pre-inverted (`tail_inv`,
    // column-major, `tail_inv[b·e + a] = (S⁻¹)_{a,b}`). By the block-inverse
    // identity (Harville, D.A. (1997), *Matrix Algebra From a Statistician's
    // Perspective*, Springer, ch. 8), a bilinear form over two rows `i`, `j` of
    // the same cluster `f` is
    //
    //   mᵢ'A⁻¹mⱼ = yᵢ'm_{c,j} + rᵢ'S⁻¹rⱼ,
    //   yᵢ = D_f⁻¹m_{c,i},   rᵢ = C_f'yᵢ − tᵢ,
    //
    // `tᵢ` the row's own crossed entries. Expanding the tail term with
    // `H_f = C_f S⁻¹` (`q_core × e`) and `G_f = H_f C_f' = C_f S⁻¹ C_f'`
    // (`q_core × q_core`), `S⁻¹` symmetric:
    //
    //   rᵢ'S⁻¹rⱼ = yᵢ'G_f yⱼ − yᵢ'(H_f tⱼ) − yⱼ'(H_f tᵢ) + tᵢ'S⁻¹tⱼ.
    //
    // `H_f` and `G_f` are formed once per cluster per assembly (`tail_h` on
    // cluster `f`'s coupling columns only, compact — see `AssemblyBufs`;
    // `tail_g` per cluster), so a row pays `q_core² + 2·q_core·ncz + ncz²`
    // instead of a walk over every coupling column of its cluster (`|cols_f|²`).
    // Every sum runs over cluster `f`'s coupling columns (`cols`) only. The
    // forms reassociate the per-row sums over those columns, so the two agree
    // to round-off.
    if e > 0 {
        for f in 0..s {
            let cols = &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize];
            if cols.is_empty() {
                continue;
            }
            let (hb, nc) = (qc * coup_ptr[f] as usize, cols.len());
            for l in 0..qc {
                for (j, &b) in cols.iter().enumerate() {
                    let col = &tail_inv[b as usize * e..b as usize * e + e];
                    let mut acc = T::ZERO;
                    for (ja, &a) in cols.iter().enumerate() {
                        acc += coupling[hb + l * nc + ja] * col[a as usize];
                    }
                    tail_h[hb + l * nc + j] = acc;
                }
            }
            let gb = f * qc * qc;
            for l1 in 0..qc {
                for l2 in 0..qc {
                    let mut acc = T::ZERO;
                    for j in 0..nc {
                        acc += tail_h[hb + l1 * nc + j] * coupling[hb + l2 * nc + j];
                    }
                    tail_g[gb + l1 * qc + l2] = acc;
                }
            }
        }
    }
    // `rᵢ'S⁻¹rⱼ` off the forms above: `yi`/`yj` the two rows' core solves,
    // `ti`/`tj` their crossed values on the shared columns `tc` (one row's own
    // crossed entries: both bilinear partners of a row here are that row and a
    // θ-derivative of it, which share its crossed columns). Only the columns
    // in `cols` (cluster `f`'s coupling columns, sorted) enter: a column
    // outside them (a θ-pinned crossed grouping the `f64` pattern dropped and
    // a dual packer kept) is skipped.
    let tail_form =
        |f: usize, cols: &[u32], yi: &[T], yj: &[T], tc: &[u32], ti: &[T], tj: &[T]| -> T {
            let (hb, nc) = (qc * coup_ptr[f] as usize, cols.len());
            let gb = f * qc * qc;
            let mut t_gg = T::ZERO;
            for l1 in 0..qc {
                let mut inner = T::ZERO;
                for l2 in 0..qc {
                    inner += tail_g[gb + l1 * qc + l2] * yj[l2];
                }
                t_gg += yi[l1] * inner;
            }
            let mut t_cross = T::ZERO;
            for (z, &b) in tc.iter().enumerate() {
                let Ok(j) = cols.binary_search(&b) else {
                    continue;
                };
                let mut hi = T::ZERO;
                let mut hj = T::ZERO;
                for l in 0..qc {
                    hi += yi[l] * tail_h[hb + l * nc + j];
                    hj += yj[l] * tail_h[hb + l * nc + j];
                }
                t_cross += hi * tj[z] + hj * ti[z];
            }
            let mut t_tt = T::ZERO;
            for (z1, &b1) in tc.iter().enumerate() {
                if cols.binary_search(&b1).is_err() {
                    continue;
                }
                let col = &tail_inv[b1 as usize * e..b1 as usize * e + e];
                let mut inner = T::ZERO;
                for (z2, &b2) in tc.iter().enumerate() {
                    if cols.binary_search(&b2).is_err() {
                        continue;
                    }
                    inner += tj[z2] * col[b2 as usize];
                }
                t_tt += ti[z1] * inner;
            }
            t_gg - t_cross + t_tt
        };

    let canonical = crate::family::is_canonical(family);
    let exact_obj = crate::family::exact_curvature_differs(family);
    let mut t = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut yb = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut mad_f64 = [0.0f64; crate::lmm::MAX_PRIMARY_Q];
    let mut cvd_f64 = [0.0f64; crate::lmm::MAX_EXTRA_GROUPINGS];
    let mut mad = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut cvd = [T::ZERO; crate::lmm::MAX_EXTRA_GROUPINGS];
    let mut ta = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut ya = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    // `A_obs = M'W_obs M + I`, accumulated in the same row loop on a
    // non-canonical link (pass 3 below says why it is built here at all).
    if !canonical {
        obs_core.fill(T::ZERO);
        obs_coup.fill(T::ZERO);
        obs_schur.fill(T::ZERO);
    }

    // --- passes 1 and 2 and the `A_obs` scatter, fused into one walk over
    // the rows. Each row's `ρᵢ`, `w'ᵢ`, `w_obs,ᵢ`, `hᵢ` and core solve (`t` or
    // `yb`) are formed once and dropped: pass 2 reads only its own row's
    // pass-1 values, and the three write disjoint accumulators (pass 1 the
    // β rows `n_θ + j` of `D_γ`/`ℓ_γ`/`G_γ` and all of `D_u`/`ℓ_u`, pass 2
    // the θ rows `a < n_θ`, the scatter only `obs_*`), each still in
    // ascending row order. So every sum is the same sequence of operations
    // as three separate passes, bit for bit, without four `rows`-length
    // buffers (416 B/row at `Dual<12>`) or pass 2's repeat of the core solve.
    //
    // Pass 1, per row:
    //   ρᵢ = prior_wᵢ·(dμ/dη)(yᵢ−μᵢ)/V (`family::row_score`),
    //   ∂(prior_wᵢ·devᵢ)/∂ηᵢ = −2ρᵢ, ∂²/∂η² = 2·w_obs,ᵢ, hᵢ = mᵢ'A⁻¹mᵢ
    //   D_βj  += (∂devᵢ/∂ηᵢ)·xᵢⱼ            D_uc  += (∂devᵢ/∂ηᵢ)·M_ic
    //   ℓ_βj  += w'ᵢ·xᵢⱼ·hᵢ                ℓ_uc  += w'ᵢ·M_ic·hᵢ
    //   (G_βj)_c += 2·w_obs,ᵢ·xᵢⱼ·M_ic     (∂M/∂β = 0, so this is all of G_β)
    //
    // Pass 2, per row × per θ coordinate. `Λ` is linear in θ, so
    // `m_{a,i} = ∂mᵢ/∂θ_a` is a selection; `∂ηᵢ/∂θ_a = m_{a,i}'u`:
    //   D_θa  += −2ρᵢ·(m_{a,i}'u)
    //   ℓ_θa  += 2·wᵢ·(mᵢ'A⁻¹m_{a,i}) + w'ᵢ·(m_{a,i}'u)·hᵢ
    //   (G_θa)_c += 2·w_obs,ᵢ·(m_{a,i}'u)·M_ic + (−2ρᵢ)·(m_{a,i})_c
    // The cross-leverage `mᵢ'A⁻¹m_{a,i}` is the one object pass 1 does not
    // already form; it is one dot against the row reduction of `mᵢ`.
    // ---
    for i in 0..n {
        if let Some(sc) = stream.as_mut() {
            if i == sc.next {
                let i1 = (i + sc.rows.block).min(n);
                let _ = stream_fill(
                    sc.rows,
                    g,
                    sc.theta_nested,
                    sc.nested_decl,
                    z_buf,
                    extra_ids,
                    cluster_ids,
                    cross_theta,
                    cross_col,
                    n_cross,
                    x,
                    sc.beta,
                    sc.offset,
                    u,
                    family,
                    nb_theta,
                    y,
                    prior_w,
                    sc.weighted,
                    i,
                    i1,
                    true,
                );
                sc.i0 = i;
                sc.next = i1;
            }
        }
        let (eta_i, prob_i, w_i, mc): (T, T, T, &[T]) = match stream.as_ref() {
            Some(sc) => {
                let t = i - sc.i0;
                (
                    sc.rows.eta[t],
                    sc.rows.prob[t],
                    sc.rows.w[t],
                    &sc.rows.mc[t * qc..t * qc + qc],
                )
            }
            None => (eta[i], prob[i], w[i], &core_m[i * qc..i * qc + qc]),
        };
        let f = cluster_ids[i] as usize;
        let cb = f * qc * qc;
        let fac = &core_fac[cb..cb + qc * qc];
        let cbase = i * g_cap;
        let ncz = if extras { n_cross[i] as usize } else { 0 };
        let cols: &[u32] = if e > 0 {
            &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize]
        } else {
            &[]
        };
        // Pass 1. The core solve (`t` at `e == 0`, `yb` otherwise) stays
        // live for pass 2's cross-leverage below.
        let h = if e == 0 {
            block_forward_solve(fac, qc, mc, &mut t[..qc]);
            let mut acc = T::ZERO;
            for v in t.iter().take(qc) {
                acc += *v * *v;
            }
            acc
        } else {
            yb[..qc].copy_from_slice(&mc[..qc]);
            glmm_block_solve(fac, qc, &mut yb[..qc]);
            // Core and tail folded apart, then one add.
            let mut core_acc = T::ZERO;
            for c in 0..qc {
                core_acc += yb[c] * mc[c];
            }
            let tail_acc = if cols.is_empty() {
                T::ZERO
            } else {
                let tc = &cross_col[cbase..cbase + ncz];
                let tv = &cross_theta[..ncz];
                tail_form(f, cols, &yb[..qc], &yb[..qc], tc, tv, tv)
            };
            core_acc + tail_acc
        };
        let r = match family {
            Family::Binomial {
                link: BinomialLink::Logit,
            } => T::from_f64(prior_w[i]) * (T::from_f64(y[i]) - prob_i),
            other => crate::family::row_score(other, nb_theta, y[i], prior_w[i], eta_i, prob_i),
        };
        // The kernel's stored `w` is the weight the objective's `A = M'WM + I`
        // is built from (`pirls::evaluate_at_mode`): the raw Fisher weight
        // `prior_w·(dμ/dη)²/V` on a link whose exact curvature is Fisher, the
        // observed `W_obs` where the two differ. `w_eta` is that same weight's
        // `dw/dη`. `w_obs` is the mode equation's Jacobian: `w` itself on an
        // exact-curvature link rather than formed a second time.
        let w_eta = if exact_obj {
            crate::family::observed_weight_eta_deriv(
                family,
                nb_theta,
                y[i],
                prior_w[i],
                eta_i,
                prob_i,
                T::ZERO,
            )
        } else {
            crate::family::weight_eta_deriv(family, nb_theta, eta_i, prob_i, w_i)
        };
        let w_obs = if exact_obj {
            w_i
        } else {
            crate::family::observed_weight(family, nb_theta, y[i], prior_w[i], eta_i, prob_i, w_i)
        };
        let dev_eta = T::from_f64(-2.0) * r;
        let a = w_eta * h;
        let two_wo = T::from_f64(2.0) * w_obs;
        for j in 0..p {
            let xij = T::from_f64(x[(i, j)]);
            d_gamma[n_theta + j] += dev_eta * xij;
            l_gamma[n_theta + j] += a * xij;
        }
        for (local, &mcl) in mc.iter().enumerate().take(qc) {
            let c = f * qc + local;
            d_u[c] += dev_eta * mcl;
            l_u[c] += a * mcl;
            let gb = two_wo * mcl;
            for j in 0..p {
                g_gamma[(n_theta + j) * k + c] += gb * T::from_f64(x[(i, j)]);
            }
        }
        for z in 0..ncz {
            let c = k_family + cross_col[cbase + z] as usize;
            let v = cross_theta[z];
            d_u[c] += dev_eta * v;
            l_u[c] += a * v;
            let gb = two_wo * v;
            for j in 0..p {
                g_gamma[(n_theta + j) * k + c] += gb * T::from_f64(x[(i, j)]);
            }
        }

        // Pass 2.
        let dr = T::from_f64(-2.0) * r;
        for a in 0..n_theta {
            packed_m_theta_deriv(
                g,
                a,
                params,
                // The pin rule must match the arm the packed `M` was built on:
                // the `f64` packer drops a θ-pinned crossed grouping, the dual
                // packer keeps it.
                T::IS_F64,
                z_buf,
                extra_ids,
                cluster_ids,
                i,
                &mut mad_f64[..qc],
                &mut cvd_f64,
            );
            for local in 0..qc {
                mad[local] = T::from_f64(mad_f64[local]);
            }
            for z in 0..ncz {
                cvd[z] = T::from_f64(cvd_f64[z]);
            }
            // `∂η_i/∂θ_a = m_{a,i}'u`, read at the iterate `η`, `w` and the
            // factor were built at — mixing the two iterates here would put the
            // θ-direction slope of `η` a step away from everything it
            // multiplies.
            let mut deta = T::ZERO;
            for local in 0..qc {
                deta += mad[local] * u[core_col(f, local)];
            }
            for z in 0..ncz {
                deta += cvd[z] * u[k_family + cross_col[cbase + z] as usize];
            }
            let cross = if e == 0 {
                block_forward_solve(fac, qc, &mad[..qc], &mut ta[..qc]);
                let mut acc = T::ZERO;
                for c in 0..qc {
                    acc += t[c] * ta[c];
                }
                acc
            } else {
                ya[..qc].copy_from_slice(&mad[..qc]);
                glmm_block_solve(fac, qc, &mut ya[..qc]);
                // Same two-fold association as pass 1's own leverage.
                let mut core_acc = T::ZERO;
                for c in 0..qc {
                    core_acc += yb[c] * mad[c];
                }
                let tail_acc = if cols.is_empty() {
                    T::ZERO
                } else {
                    let tc = &cross_col[cbase..cbase + ncz];
                    tail_form(
                        f,
                        cols,
                        &yb[..qc],
                        &ya[..qc],
                        tc,
                        &cross_theta[..ncz],
                        &cvd[..ncz],
                    )
                };
                core_acc + tail_acc
            };
            d_gamma[a] += dr * deta;
            l_gamma[a] += T::from_f64(2.0) * w_i * cross + w_eta * deta * h;
            let tw = T::from_f64(2.0) * w_obs * deta;
            for local in 0..qc {
                g_gamma[a * k + f * qc + local] += tw * mc[local] + dr * mad[local];
            }
            for z in 0..ncz {
                let c = k_family + cross_col[cbase + z] as usize;
                g_gamma[a * k + c] += tw * cross_theta[z] + dr * cvd[z];
            }
        }

        // The `A_obs` scatter, non-canonical links only.
        if !canonical {
            // Same `e > 0` guard as the passes above: the CSR is only built
            // on a structured shape.
            let (coup, nc) = if e > 0 {
                let (lo, hi) = (coup_ptr[f] as usize, coup_ptr[f + 1] as usize);
                (qc * lo, hi - lo)
            } else {
                (0, 0)
            };
            let wo = w_obs;
            for r in 0..qc {
                let wmr = wo * mc[r];
                for c in 0..=r {
                    obs_core[cb + r * qc + c] += wmr * mc[c];
                }
                for z in 0..ncz {
                    obs_coup[coup + r * nc + coup_pos[cbase + z] as usize] += wmr * cross_theta[z];
                }
            }
            for z in 0..ncz {
                let b = cross_col[cbase + z] as usize;
                let wvb = wo * cross_theta[z];
                for z2 in 0..ncz {
                    let b2 = cross_col[cbase + z2] as usize;
                    if b2 <= b {
                        obs_schur[b * e + b2] += wvb * cross_theta[z2];
                    }
                }
            }
        }
    }

    // --- pass 3.
    //   F_γ = D_γ + ℓ_γ           F_u = D_u + 2u + ℓ_u
    //   G_u = 2·A_obs             adj = G_u⁻¹F_u
    //   D*_γ = F_γ − adj'·G_γ
    // ---
    for f in 0..s {
        for local in 0..qc {
            let c = f * qc + local;
            adj[c] = d_u[c] + T::from_f64(2.0) * u[core_col(f, local)] + l_u[c];
        }
    }
    for b in 0..e {
        let c = k_family + b;
        adj[c] = d_u[c] + T::from_f64(2.0) * u[c] + l_u[c];
    }
    // `A_obs = M'W_obs M + I`. On a canonical link `A_obs` IS the Fisher `A`
    // the mode solve already factored; otherwise its rows were scattered in
    // the loop above and it is completed and factored here, and a non-PD
    // factor is the refusal. Built here rather than taken from the dual
    // kernel's own observed twin because a `BetaStep::Fixed` `f64` call
    // leaves no twin at all, so this arm has to exist for the `f64`
    // instantiation regardless — one body serves both.
    if canonical {
        if e == 0 {
            solve_core_blocks(core_fac, qc, s, adj);
        } else {
            // `ss` is `Some` exactly when the value pass factored `S` sparsely
            // (`schur_blk` then still holds the unfactored `S`).
            structured_ainv_solve(
                g, core_fac, coupling, schur_blk, coup_cols, coup_ptr, ss, adj,
            );
        }
    } else {
        for f in 0..s {
            for r in 0..qc {
                obs_core[f * qc * qc + r * qc + r] += T::ONE;
            }
        }
        for b in 0..e {
            obs_schur[b * e + b] += T::ONE;
        }
        if e == 0 {
            for f in 0..s {
                let cb = f * qc * qc;
                if !glmm_block_chol(&mut obs_core[cb..cb + qc * qc], qc) {
                    return None;
                }
            }
            solve_core_blocks(obs_core, qc, s, adj);
        } else {
            structured_factor::<T>(g, obs_core, obs_coup, obs_schur, coup_cols, coup_ptr, None)?;
            structured_ainv_solve(
                g, obs_core, obs_coup, obs_schur, coup_cols, coup_ptr, None, adj,
            );
        }
    }
    // `G_u = 2·A_obs`, so the solve above is off by the factor of two.
    for v in adj.iter_mut() {
        *v *= T::from_f64(0.5);
    }
    for a in 0..m {
        let mut acc = d_gamma[a] + l_gamma[a];
        for c in 0..k {
            acc -= adj[c] * g_gamma[a * k + c];
        }
        grad[a] = acc;
    }
    Some(())
}

/// `rhs_f ← A_f⁻¹ rhs_f` per cluster against the block-diagonal core factor —
/// the whole of `A⁻¹` at `e == 0`, and the first half of
/// `structured_ainv_solve`'s own core sweep otherwise.
fn solve_core_blocks<T: Scalar>(fac: &[T], qc: usize, s: usize, rhs: &mut [T]) {
    for f in 0..s {
        let cb = f * qc * qc;
        glmm_block_solve(&fac[cb..cb + qc * qc], qc, &mut rhs[f * qc..f * qc + qc]);
    }
}

// ---------------------------------------------------------------------------
// The packed-row route
// ---------------------------------------------------------------------------

/// The `f64` PIRLS mode solve the packed-row derivative requests start from,
/// the packed twin of [`mode_solve_f64`]: `BetaStep::Fixed` at `tol`, with
/// THROWAWAY counters, over the same `fill_lambda_small` → `fill_m_vals` →
/// `pirls_solve_packed` sequence the packed arm of `laplace_deviance` runs.
///
/// Warm from whatever `ws.pirls.u` holds, NOT the cold `u = 0` that arm takes
/// under a tolerance override: this is a derivative request at the fit's own
/// γ̂, so the fit's mode is the best seed there is, and the objective it
/// converges to is the same fixed point either way.
///
/// Returns the Laplace objective — `Φ(D) + ‖û‖² + log|A|`, with Gamma's `aic`
/// substitution, exactly as that arm assembles it — or `None` when the solve
/// did not converge or the objective is not finite. Leaves `ws.pirls.u` at the
/// mode and the η-dependent state, `ws.packed.m_vals`, `ws.packed.a` and
/// `ws.packed.a_chol` at the solve's values; the caller restores `u`.
fn packed_mode_solve_f64(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    p: usize,
    n: usize,
    tol: f64,
) -> Option<f64> {
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let (family, nb_theta, weighted, k) = (ws.family, ws.nb_theta, ws.weighted, ws.k);
    // Fixed-mode β transient, as `laplace_deviance` copies it.
    ws.beta_rhs[..p].copy_from_slice(&ws.params[n_theta..m]);

    let GlmmWorkspace {
        groupings,
        params: prm,
        beta_rhs,
        packed,
        pirls,
        prior_w,
        wx,
        z_buf,
        offset: offset_field,
        ..
    } = ws;
    let offset = offset_field.as_deref();
    crate::sparse::fill_lambda_small(&prm[..n_theta], groupings, &mut packed.lam_small);
    fill_m_vals(packed, groupings, z_buf, n);
    let mut mode_counters = crate::counters::EvalCounters::new();
    let (dev, pen, logdet, conv) = pirls_solve_packed(
        family,
        nb_theta,
        k,
        p,
        x,
        y,
        &prior_w[..n],
        weighted,
        beta_rhs,
        BetaStep::Fixed,
        pirls,
        packed,
        wx,
        offset,
        Some(tol),
        n,
        &mut mode_counters,
    );
    if !conv || !dev.is_finite() {
        return None;
    }
    let obj = dev + pen + 2.0 * logdet;
    obj.is_finite().then_some(obj)
}

/// The `f64` half of the packed engine, sized once per shape and kept on the
/// workspace: the assembly's own scratch at `T = f64`, the mode snapshot the
/// solve is taken around, and `û`'s first-order response `U`.
///
/// `U` exists only here. On the blocked and structured routes the dual PIRLS
/// kernel delivers `dû/dγ` in `u`'s lanes, which is what makes the assembled
/// gradient's lanes exact; the packed kernel is `f64`-only, so those lanes are
/// solved for instead — `U = −G_u⁻¹G_γ`, `m` right-hand sides against the
/// factor the `f64` assembly has already built (Skaug & Fournier 2006's
/// implicit-function step, the same one the adjoint takes in the other
/// direction).
pub(crate) struct PackedGradientBufs {
    asm: AssemblyBufs<f64>,
    /// `m·k` row-major: `u_lanes[a·k + c] = ∂û_c/∂γ_a`.
    u_lanes: Vec<f64>,
    /// `m`: the assembled gradient at `T = f64`.
    grad: Vec<f64>,
    /// `k.max(1)`: `ws.pirls.u` as the caller left it, restored on every exit.
    saved_u: Vec<f64>,
    /// `k`: the converged mode the assembly and the dual chunks both sit at.
    u_mode: Vec<f64>,
}

impl PackedGradientBufs {
    fn for_shape(m: usize, k: usize, rows: usize, width: usize, observed: bool) -> Self {
        PackedGradientBufs {
            // `s`/`q_core`/`e` are the other routes' shape terms and are
            // unread here, so they go in at their minimum.
            asm: AssemblyBufs::<f64>::for_shape(m, k, rows, 0, 1, 0, observed, width),
            u_lanes: vec![0.0; (m * k).max(1)],
            grad: vec![0.0; m],
            saved_u: vec![0.0; k.max(1)],
            u_mode: vec![0.0; k.max(1)],
        }
    }

    /// Shape half of the reuse policy, mirroring
    /// `derivative::GlmmDualScratch::matches_shape`: a request at the same
    /// shape reuses, a different one reallocates once.
    fn matches_shape(&self, m: usize, k: usize, rows: usize, width: usize, observed: bool) -> bool {
        self.grad.len() == m
            && self.u_lanes.len() == (m * k).max(1)
            && self.saved_u.len() == k.max(1)
            && self.asm.g_gamma.len() == (m * k).max(1)
            && self.asm.packed.m_vals.len() == rows * width
            && self.asm.packed.a.len() == k * k
            // Sized by `observed` (`PackedAsmBufs::for_shape`): comparing
            // against a flat `k²` would fail on every canonical call and
            // rebuild this scratch each time.
            && self.asm.packed.obs.len() == obs_len(observed, k * k)
    }
}

/// The workspace's packed `f64` scratch, built on the first request for this
/// shape and reused after. Taken OUT of the workspace: every entry point below
/// hands `&mut ws` to the mode solve while holding these buffers.
fn take_packed_bufs(ws: &mut GlmmWorkspace, p: usize, n: usize) -> Box<PackedGradientBufs> {
    let m = ws.n_theta + p;
    let (k, width) = (ws.k, ws.packed.width);
    let observed = !crate::family::is_canonical(ws.family);
    let need_build = ws
        .packed_asm
        .as_deref()
        .is_none_or(|b| !b.matches_shape(m, k, n, width, observed));
    if need_build {
        ws.packed_asm = Some(Box::new(PackedGradientBufs::for_shape(
            m, k, n, width, observed,
        )));
    }
    ws.packed_asm
        .take()
        .expect("just built or confirmed present above")
}

/// The Laplace gradient `dD*/dγ` on the packed-row layout, assembled in `f64`
/// and written into `grad[..n_θ + p]` — the pass that supplies `û`'s lanes to
/// the chunked Hessian, and the `f64` reference the packed Hessian gate
/// compares its columns against.
///
/// Same statuses, and the same meanings, as the other routes'
/// [`joint_hessian_columns`]: `Unsupported` on a mode state where the η clamp
/// binds — the reason is stated there — or on a non-positive-definite
/// observed factor, `NotConverged` on a mode solve that fails.
///
/// The workspace comes back as found in `u`; `eta`, `prob`, `w`, `mu`,
/// `beta_rhs`, `ws.packed.m_vals` and the packed factor are left at the mode
/// solve's values, exactly as `laplace_gradient` leaves the blocked ones.
#[cfg(all(test, feature = "formula"))]
pub(crate) fn packed_gradient(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> DerivStatus {
    // At the fitted Gamma dispersion, differentiate the fixed-φ GLMM the fit
    // minimized (`GlmmWorkspace::at_fixed_dispersion`); inside, `gamma_phi` is 1
    // and this guard does not fire again.
    if ws.dispersion_scaled() {
        return ws.at_fixed_dispersion(n, |ws| packed_gradient(ws, x, y, p, n, grad));
    }
    if !assembly_routes(ws, n) || ws.layout != GlmmLayout::Packed {
        return DerivStatus::Unsupported;
    }
    let m = ws.n_theta + p;
    let mut bufs = take_packed_bufs(ws, p, n);
    let st = packed_gradient_into(ws, x, y, p, n, &mut bufs);
    if matches!(st, DerivStatus::Ok(_)) {
        grad[..m].copy_from_slice(&bufs.grad[..m]);
    }
    ws.packed_asm = Some(bufs);
    st
}

/// Shared body of [`packed_gradient`] and the chunked Hessian's `f64` half:
/// mode solve, clamp census, one assembly at `T = f64`, then `û`'s lanes.
/// Leaves the gradient in `bufs.grad`, the mode in `bufs.u_mode` and
/// `U = −G_u⁻¹G_γ` in `bufs.u_lanes`.
fn packed_gradient_into(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    p: usize,
    n: usize,
    bufs: &mut PackedGradientBufs,
) -> DerivStatus {
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let k = ws.k;
    let kk = k.max(1);
    let width = ws.packed.width;
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let canonical = crate::family::is_canonical(family);
    // Never the fit's own exit tolerance: the caller's override if it set one,
    // the derivative tolerance otherwise. Mirrors `laplace_gradient`.
    let tol = ws
        .fd
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));
    bufs.saved_u[..kk].copy_from_slice(&ws.pirls.u[..kk]);
    let Some(objective) = packed_mode_solve_f64(ws, x, y, p, n, tol) else {
        ws.pirls.u[..kk].copy_from_slice(&bufs.saved_u[..kk]);
        return DerivStatus::NotConverged;
    };
    // The same census, for the same reason, as `joint_hessian_columns` —
    // stated there.
    if eta_clamped_rows(family, &ws.pirls.eta[..n]) > 0 {
        ws.pirls.u[..kk].copy_from_slice(&bufs.saved_u[..kk]);
        return DerivStatus::Unsupported;
    }
    bufs.u_mode[..k].copy_from_slice(&ws.pirls.u[..k]);
    // The evaluation state, copied in so both instantiations of the assembly
    // read it from the same place: at `T = f64` the mode solve leaves it in
    // the workspace, at `T = Dual<N>` the chunk driver rebuilds it in place.
    {
        let pk = &mut bufs.asm.packed;
        pk.m_vals[..n * width].copy_from_slice(&ws.packed.m_vals[..n * width]);
        pk.eta[..n].copy_from_slice(&ws.pirls.eta[..n]);
        pk.prob[..n].copy_from_slice(&ws.pirls.prob[..n]);
        pk.w[..n].copy_from_slice(&ws.pirls.w[..n]);
        pk.u[..k].copy_from_slice(&ws.pirls.u[..k]);
    }
    let out = packed_assemble(
        &ws.groupings,
        family,
        nb_theta,
        x,
        &ws.z_buf,
        y,
        &ws.prior_w[..n],
        &ws.packed.m_cols[..n * width],
        width,
        k,
        n_theta,
        p,
        n,
        &mut bufs.asm,
        &mut bufs.grad,
    );
    ws.pirls.u[..kk].copy_from_slice(&bufs.saved_u[..kk]); // restore — leave ws.pirls.u as found
    if out.is_none() {
        return DerivStatus::Unsupported;
    }
    // `û(γ)` solves `G(γ, û) = 0`, so `dû/dγ = −G_u⁻¹G_γ` — `m` right-hand
    // sides against the SAME factor the assembly's own pass 3 just used for
    // its adjoint (`A` on a canonical link, `A_obs` otherwise), and
    // `G_u = 2·A_obs` puts the ½ in front.
    let fac_len = k * k;
    for a in 0..m {
        let col = &mut bufs.u_lanes[a * k..a * k + k];
        col.copy_from_slice(&bufs.asm.g_gamma[a * k..a * k + k]);
        let fac = if canonical {
            &bufs.asm.packed.a[..fac_len]
        } else {
            &bufs.asm.packed.obs[..fac_len]
        };
        glmm_block_solve(fac, k, col);
        for v in col.iter_mut() {
            *v *= -0.5;
        }
    }
    DerivStatus::Ok(objective)
}

/// The three row passes of [`assemble`] on the packed-row layout, over the
/// evaluation state in `asm.packed` — the same `F`/`G` pair and the same
/// adjoint identity, differing only in how `A⁻¹` is applied.
///
/// Mirrors the blocked and structured reducers (`block_forward_solve`, and
/// [`assemble`]'s per-cluster tail forms) on the shared block-inverse identity;
/// change together. Where those reduce each row against a per-cluster factor, this
/// layout's `A` is dense `k×k` (every row loads one level of every grouping,
/// so there is no cluster block to solve alone), so `A⁻¹` is formed once, off
/// the factor, and every bilinear form `m_i'A⁻¹m_j` is a `width²` double loop
/// over the two rows' own nonzeros.
///
/// Generic over the scalar exactly as [`assemble`] is: at `T = f64` this is
/// the gradient, at `T = Dual<N>` its lanes are the joint Hessian's seeded
/// columns. `∂M/∂θ_a` is the one quantity lifted from `f64` rather than read
/// off the lanes, for the reason stated on [`assemble`].
///
/// `None` on a non-positive-definite `A` or `A_obs`: the adjoint equation
/// `G_u·adj = F_u` has no Cholesky then, and the Fisher factor is never
/// silently put in the observed one's place.
#[allow(clippy::too_many_arguments)]
fn packed_assemble<T: Scalar>(
    g: &LmmGroupings,
    family: Family,
    nb_theta: f64,
    x: MatRef<f64>,
    // The random-slope values `fill_m_vals` read (`workspace::fill_z_f64`).
    z_buf: &[f64],
    y: &[f64],
    prior_w: &[f64],
    // Design-fixed RE column of each packed nonzero, row `i` at `i·width`.
    m_cols: &[u32],
    width: usize,
    k: usize,
    n_theta: usize,
    p: usize,
    n: usize,
    asm: &mut AssemblyBufs<T>,
    grad: &mut [T],
) -> Option<()> {
    let m = n_theta + p;
    let canonical = crate::family::is_canonical(family);
    let exact_obj = crate::family::exact_curvature_differs(family);
    let AssemblyBufs {
        d_gamma,
        l_gamma,
        d_u,
        l_u,
        // `G_γ` row-major, `g_gamma[a·k + c] = ∂G_c/∂γ_a`, as on the other
        // routes; `c` is the RE column directly, since `m_cols` already is.
        g_gamma,
        adj,
        packed:
            PackedAsmBufs {
                a,
                a_inv,
                obs,
                col,
                m_vals,
                eta,
                prob,
                w,
                u,
                m_deriv,
            },
        ..
    } = asm;
    d_gamma[..m].fill(T::ZERO);
    l_gamma[..m].fill(T::ZERO);
    d_u[..k].fill(T::ZERO);
    l_u[..k].fill(T::ZERO);
    g_gamma[..m * k].fill(T::ZERO);

    // `A = M'WM + I`, dense and row-major, by the same per-row scatter over
    // the `width` nonzeros `pirls::PackedFactor::scatter` runs — change
    // together. Factored in place, then inverted by `k` unit-vector solves:
    // `k³` once against `width²` per row per bilinear pair is the cheap side
    // at every packed shape in reach (see [`PackedAsmBufs`]).
    a[..k * k].fill(T::ZERO);
    #[allow(clippy::needless_range_loop)]
    for i in 0..n {
        let base = i * width;
        let wi = w[i];
        for ta in base..base + width {
            let ca = m_cols[ta] as usize;
            let wva = wi * m_vals[ta];
            for tb in base..base + width {
                a[ca * k + m_cols[tb] as usize] += wva * m_vals[tb];
            }
        }
    }
    for r in 0..k {
        a[r * k + r] += T::ONE;
    }
    if !glmm_block_chol(&mut a[..k * k], k) {
        return None;
    }
    for c in 0..k {
        col[..k].fill(T::ZERO);
        col[c] = T::ONE;
        glmm_block_solve(&a[..k * k], k, &mut col[..k]);
        for r in 0..k {
            a_inv[r * k + c] = col[r];
        }
    }

    // `A_obs = M'W_obs M + I` on a non-canonical link, scattered row by row
    // in the loop below and completed after it.
    if !canonical {
        obs[..k * k].fill(T::ZERO);
    }

    // --- passes 1 and 2 and the `A_obs` scatter in one walk over the rows,
    // as in `assemble` (which states why the fusion is bit-identical); the
    // per-row leverage is the only line that differs, and only in how `A⁻¹`
    // is reached. ---
    for i in 0..n {
        let base = i * width;
        let mut h = T::ZERO;
        for ta in base..base + width {
            let ca = m_cols[ta] as usize;
            let mut inner = T::ZERO;
            for tb in base..base + width {
                inner += a_inv[ca * k + m_cols[tb] as usize] * m_vals[tb];
            }
            h += m_vals[ta] * inner;
        }
        let r = match family {
            Family::Binomial {
                link: BinomialLink::Logit,
            } => T::from_f64(prior_w[i]) * (T::from_f64(y[i]) - prob[i]),
            other => crate::family::row_score(other, nb_theta, y[i], prior_w[i], eta[i], prob[i]),
        };
        // Mirrors `assemble` — change together, including the stored-weight rule
        // on a link whose exact curvature differs from Fisher (`exact_obj`).
        let w_eta = if exact_obj {
            crate::family::observed_weight_eta_deriv(
                family,
                nb_theta,
                y[i],
                prior_w[i],
                eta[i],
                prob[i],
                T::ZERO,
            )
        } else {
            crate::family::weight_eta_deriv(family, nb_theta, eta[i], prob[i], w[i])
        };
        let w_obs = if exact_obj {
            w[i]
        } else {
            crate::family::observed_weight(
                family, nb_theta, y[i], prior_w[i], eta[i], prob[i], w[i],
            )
        };
        let dev_eta = T::from_f64(-2.0) * r;
        let av = w_eta * h;
        let two_wo = T::from_f64(2.0) * w_obs;
        for j in 0..p {
            let xij = T::from_f64(x[(i, j)]);
            d_gamma[n_theta + j] += dev_eta * xij;
            l_gamma[n_theta + j] += av * xij;
        }
        for t in base..base + width {
            let c = m_cols[t] as usize;
            let mv = m_vals[t];
            d_u[c] += dev_eta * mv;
            l_u[c] += av * mv;
            let gb = two_wo * mv;
            for j in 0..p {
                g_gamma[(n_theta + j) * k + c] += gb * T::from_f64(x[(i, j)]);
            }
        }

        // Pass 2, per θ coordinate, as `assemble`'s.
        let dr = T::from_f64(-2.0) * r;
        for a_idx in 0..n_theta {
            packed_m_vals_theta_deriv(g, a_idx, z_buf, i, m_deriv);
            // `∂η_i/∂θ_a = m_{a,i}'u`, read at the iterate `η`, `w` and the
            // factor were built at.
            let mut deta = T::ZERO;
            for t in 0..width {
                deta += T::from_f64(m_deriv[t]) * u[m_cols[base + t] as usize];
            }
            // The cross-leverage `m_i'A⁻¹m_{a,i}` — the one object the fit's
            // own row passes do not already form.
            let mut cross = T::ZERO;
            for ta in base..base + width {
                let ca = m_cols[ta] as usize;
                let mut inner = T::ZERO;
                for tb in 0..width {
                    inner += a_inv[ca * k + m_cols[base + tb] as usize] * T::from_f64(m_deriv[tb]);
                }
                cross += m_vals[ta] * inner;
            }
            d_gamma[a_idx] += dr * deta;
            l_gamma[a_idx] += T::from_f64(2.0) * w[i] * cross + w_eta * deta * h;
            let tw = T::from_f64(2.0) * w_obs * deta;
            for t in 0..width {
                let c = m_cols[base + t] as usize;
                g_gamma[a_idx * k + c] += tw * m_vals[base + t] + dr * T::from_f64(m_deriv[t]);
            }
        }

        // The `A_obs` scatter, non-canonical links only.
        if !canonical {
            for ta in base..base + width {
                let ca = m_cols[ta] as usize;
                let wva = w_obs * m_vals[ta];
                for tb in base..base + width {
                    obs[ca * k + m_cols[tb] as usize] += wva * m_vals[tb];
                }
            }
        }
    }

    // --- pass 3, as `assemble`'s pass 3: `F_γ = D_γ + ℓ_γ`,
    // `F_u = D_u + 2u + ℓ_u`, `adj = (2·A_obs)⁻¹F_u`,
    // `D*_γ = F_γ − adj'·G_γ`. ---
    for c in 0..k {
        adj[c] = d_u[c] + T::from_f64(2.0) * u[c] + l_u[c];
    }
    // `A_obs` IS the Fisher `A` only on a canonical link; otherwise its rows
    // were scattered above and it is completed and factored here — mirrors
    // `assemble`'s pass 3.
    if canonical {
        glmm_block_solve(&a[..k * k], k, &mut adj[..k]);
    } else {
        for r in 0..k {
            obs[r * k + r] += T::ONE;
        }
        if !glmm_block_chol(&mut obs[..k * k], k) {
            return None;
        }
        glmm_block_solve(&obs[..k * k], k, &mut adj[..k]);
    }
    // `G_u = 2·A_obs`, so the solve above is off by the factor of two.
    for v in adj[..k].iter_mut() {
        *v *= T::from_f64(0.5);
    }
    for a_idx in 0..m {
        let mut acc = d_gamma[a_idx] + l_gamma[a_idx];
        for c in 0..k {
            acc -= adj[c] * g_gamma[a_idx * k + c];
        }
        grad[a_idx] = acc;
    }
    Some(())
}

/// The packed-row layout's chunked Hessian driver, the twin of
/// [`run_assembled_hessian`]'s loop: one `f64` pass for the mode, the value
/// gradient and `û`'s response `U`, then `⌈m/N⌉` passes that seed `N`
/// coordinates each and read the assembled gradient's lanes as those
/// coordinates' Hessian columns.
///
/// No dual PIRLS kernel runs here, and none exists: the packed kernel is
/// `f64`-only. Everything the assembly reads at `T` is rebuilt directly —
/// `u = û + lanes·U` (exact, because `U` is the implicit-function derivative
/// of the mode), `M = M(θ̂) + lanes·∂M/∂θ` (exact, because `Λ` is linear in
/// θ), then η, and μ/W through the same `Scalar::family_pass` the packed
/// kernel's own exit refresh calls. So the lanes are exact in one pass and
/// there is no refinement loop and no `min_iters` floor to set.
#[allow(clippy::too_many_arguments)]
fn packed_joint_hessian_columns(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    p: usize,
    n: usize,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let k = ws.k;
    let width = ws.packed.width;
    let nagq = ws.nagq;
    let observed = !crate::family::is_canonical(ws.family);
    // The FIRST-derivative ladder, as on the other routes: the assembly reads
    // first-order lanes only, so a Hessian request resolves to a `Dual` rung
    // (`pick_assembled`'s table) and chunks like a gradient does.
    let nl = NLanes::pick_assembled(m);
    // The shape terms go in as they are; `for_shape` is what knows that a
    // non-zero packed width means the blocked and structured twins they size
    // are unread here, and `matches_shape` reads them the same way.
    let (s, q_p, q_core, e) = (
        ws.groupings.n_primary,
        ws.groupings.primary_q,
        ws.groupings.primary_q + ws.groupings.nested_per_parent,
        ws.groupings.k_crossed(),
    );

    let mut f64_bufs = take_packed_bufs(ws, p, n);
    let st = packed_gradient_into(ws, x, y, p, n, &mut f64_bufs);
    if !matches!(st, DerivStatus::Ok(_)) {
        ws.packed_asm = Some(f64_bufs);
        return st;
    }
    let DerivStatus::Ok(objective) = st else {
        unreachable!("checked immediately above")
    };

    // Reuse policy, and the scratch itself, are `laplace_gradient`'s. The
    // packed width is what tells `for_shape` to size the packed state and
    // leave the blocked/structured twins at their minimum.
    let need_build = ws.dual_scratch.as_deref().is_none_or(|sc| {
        sc.lanes() != nl
            || !sc.matches_shape(
                m, p, k, n, s, q_p, q_core, e, false, nagq, observed, width, None,
            )
    });
    if need_build {
        ws.dual_scratch = Some(Box::new(GlmmDualScratch::for_shape(
            nl,
            m,
            p,
            k,
            n,
            s,
            q_p,
            q_core,
            e,
            // The packed layout sizes the structured twin at its minimum
            // whatever its extras (`DenseTwinShape::of`).
            false,
            nagq,
            observed,
            width,
            cluster_ids,
            None,
        )));
    }
    let mut scratch = ws
        .dual_scratch
        .take()
        .expect("just built or confirmed present above");
    let ws_ro: &GlmmWorkspace = ws;
    macro_rules! chunk_arm {
        ($bufs:expr) => {
            packed_hessian_chunks($bufs, ws_ro, x, y, p, n, &f64_bufs, objective, grad, hess)
        };
    }
    let st = match &mut *scratch {
        GlmmDualScratch::D4(bufs, ..) => chunk_arm!(bufs),
        GlmmDualScratch::D5(bufs, ..) => chunk_arm!(bufs),
        GlmmDualScratch::D6(bufs, ..) => chunk_arm!(bufs),
        GlmmDualScratch::D8(bufs, ..) => chunk_arm!(bufs),
        GlmmDualScratch::D12(bufs, ..) => chunk_arm!(bufs),
        // Unreachable: this slot is written only from `pick(m, false)`, so it
        // only ever holds a `Dual` variant. The arm exists because the match
        // is exhaustive over the one shared enum; decline rather than panic.
        GlmmDualScratch::H4(..)
        | GlmmDualScratch::H5(..)
        | GlmmDualScratch::H6(..)
        | GlmmDualScratch::H8(..)
        | GlmmDualScratch::H12(..) => DerivStatus::Unsupported,
    };
    ws.dual_scratch = Some(scratch);
    ws.packed_asm = Some(f64_bufs);
    st
}

/// The chunk loop [`packed_joint_hessian_columns`]'s per-`N` match arms hand a
/// typed buffer set to. Mirrors the chunked seeding in the derivative kernel's
/// `run_gradient` — change together; the invariant behind re-seeding `u` from
/// the `f64` mode on every pass is stated in full there.
#[allow(clippy::too_many_arguments)]
fn packed_hessian_chunks<T: Seed>(
    bufs: &mut GlmmDualBufs<T>,
    ws: &GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    p: usize,
    n: usize,
    f64_bufs: &PackedGradientBufs,
    objective: f64,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let k = ws.k;
    let width = ws.packed.width;
    let family = ws.family;
    let nb_theta = ws.nb_theta;
    let weighted = ws.weighted;
    let g = &ws.groupings;
    let m_cols = &ws.packed.m_cols[..n * width];
    let m_vals_f64 = &ws.packed.m_vals[..n * width];
    let prior_w = &ws.prior_w[..n];
    let offset = ws.offset.as_deref();
    let prm = &ws.params[..m];
    let u_mode = &f64_bufs.u_mode[..k];
    let u_lanes = &f64_bufs.u_lanes[..m * k];
    let lanes = T::LANES;
    let n_chunks = m.div_ceil(lanes.max(1));
    for chunk in 0..n_chunks {
        let base = chunk * lanes;
        let cw = lanes.min(m - base);
        #[allow(clippy::needless_range_loop)]
        for j in 0..m {
            bufs.params[j] = if j >= base && j < base + cw {
                T::unit(prm[j], j - base)
            } else {
                T::from_f64(prm[j])
            };
        }
        {
            let GlmmDualBufs { asm, params, .. } = &mut *bufs;
            let PackedAsmBufs {
                m_vals,
                eta,
                prob,
                w,
                u,
                m_deriv,
                ..
            } = &mut asm.packed;
            #[allow(clippy::needless_range_loop)]
            for c in 0..k {
                let mut v = T::from_f64(u_mode[c]);
                for j in 0..cw {
                    v += T::unit(0.0, j) * T::from_f64(u_lanes[(base + j) * k + c]);
                }
                u[c] = v;
            }
            for (t, &v) in m_vals_f64.iter().enumerate() {
                m_vals[t] = T::from_f64(v);
            }
            for j in 0..cw {
                let a_idx = base + j;
                if a_idx >= n_theta {
                    continue;
                }
                for i in 0..n {
                    packed_m_vals_theta_deriv(g, a_idx, &ws.z_buf, i, m_deriv);
                    for t in 0..width {
                        m_vals[i * width + t] += T::unit(0.0, j) * T::from_f64(m_deriv[t]);
                    }
                }
            }
            // `Σ yᵢηᵢ` off the RAW η, before the family pass clamps it in
            // place — the fused-identity deviance branch inside the pass
            // consumes it, exactly as `pirls::evaluate_at_mode` feeds it.
            let mut yeta = T::ZERO;
            for i in 0..n {
                let mut e = T::ZERO;
                for j in 0..p {
                    e += T::from_f64(x[(i, j)]) * params[n_theta + j];
                }
                if let Some(o) = offset {
                    e += T::from_f64(o[i]);
                }
                let b = i * width;
                for t in b..b + width {
                    e += m_vals[t] * u[m_cols[t] as usize];
                }
                eta[i] = e;
                yeta += T::from_f64(y[i]) * e;
            }
            // The infeasibility flag is discarded here and nowhere else: the
            // value part of this `eta` is the `f64` mode solve's own, which
            // the clamp census upstream already cleared, and `eta_infeasible`
            // branches on the value alone — so no lane can be zeroed by a
            // condition the census did not see.
            let _ = T::family_pass(
                family,
                nb_theta,
                &mut eta[..n],
                &y[..n],
                prior_w,
                weighted,
                yeta,
                &mut prob[..n],
                &mut w[..n],
                &mut [],
            );
            // The objective's curvature weight, as the mode solve's exit
            // refresh stores it (`pirls::evaluate_at_mode`).
            super::pirls::observed_weights_in_place(family, nb_theta, y, prior_w, eta, prob, w, n);
        }
        let assembled = packed_assemble(
            g,
            family,
            nb_theta,
            x,
            &ws.z_buf,
            y,
            prior_w,
            m_cols,
            width,
            k,
            n_theta,
            p,
            n,
            &mut bufs.asm,
            &mut bufs.grad_t,
        );
        if assembled.is_none() {
            return DerivStatus::Unsupported;
        }
        for r in 0..m {
            let d = bufs.grad_t[r].dslice();
            for j in 0..cw {
                hess[(r, base + j)] = d[j];
            }
        }
        if chunk == 0 {
            for (r, out) in grad.iter_mut().enumerate().take(m) {
                *out = bufs.grad_t[r].value();
            }
        } else {
            debug_assert!(
                (0..m).all(|r| {
                    let v = bufs.grad_t[r].value();
                    (v - grad[r]).abs() <= 1e-12 * (1.0 + grad[r].abs())
                }),
                "every chunk differentiates the same objective at the same mode, \
                 so the value part of the assembled gradient must not move between chunks"
            );
        }
    }
    DerivStatus::Ok(objective)
}
