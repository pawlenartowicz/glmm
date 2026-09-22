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
//!   Φ(D)     = D, except Gamma, whose objective substitutes the `aic`
//!   A(γ,u)   = M'WM + I,  W = diag(prior_w_i · w(η_i))       the Fisher PIRLS matrix
//!   F(γ,u)   = Φ(D) + ‖u‖² + log|A|                          the Laplace objective
//!   G(γ,u)   = D_u + 2u                                      what PIRLS solves: G = 0
//!   D*(γ)    = F(γ, û(γ))
//!   D*_γ     = F_γ − adj'·G_γ,   adj = G_u⁻ᵀ F_u             (one adjoint solve)
//! ```
//!
//! `G = 0` is what PIRLS delivers on every family, including Gamma, so this
//! form needs no special case: Gamma's only difference is `F_u ≠ 0` (the `aic`
//! substitution's `Φ' ≠ 1`), while `G` keeps its usual shape everywhere.
//!
//! On a row where μ sits on one of `family::clamp_mu`'s bounds, `D`'s
//! dependence on η stops while the kernel's score keeps forming from the
//! unclamped `dμ/dη`; this module reads that row's deviance slope as 0 and its
//! observed weight and `dw/dη` off `family::clamped_observed_weight`/
//! `clamped_weight_eta_deriv` rather than the general closed forms, so `G = 0`
//! still holds there. Two shapes still break it and are refused: a row on the
//! link's own η bound, where the score has stopped moving to first order
//! relative to what this engine assumes, and a μ-clamped row on the weighted
//! logit link, whose kernel writes a different score than the general form
//! here does — the mechanism for both is spelled out at the census in
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
    block_forward_solve, fill_m_vals, packed_m_vals_theta_deriv, pirls_solve_packed,
    structured_ainv_solve, structured_factor, BetaStep, TailKernel,
};
use super::workspace::{
    glmm_block_chol, glmm_block_solve, packed_m_theta_deriv, GlmmLayout, GlmmWorkspace,
    StructuredPattern,
};
// Named only in this file's `#[cfg(test)]` instruments (`gradient_f64_impl` and
// its callers), which destructure `ws.pirls`/`ws.structured` field-by-field.
#[cfg(test)]
use super::workspace::{PirlsScratch, StructuredScratch};

/// Test-only forcing switch: when set, [`joint_hessian_columns`] declines
/// every call regardless of shape, so `joint_hessian_cov` falls through to
/// the hyper-dual pass — compiles out of the shipped crate.
#[cfg(test)]
pub(crate) static FORCE_DECLINE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Test-only counter: incremented once per `Ok` return from
/// [`joint_hessian`], so a caller can confirm the assembled arm actually ran
/// rather than declining — compiles out of the shipped crate.
#[cfg(test)]
pub(crate) static ASSEMBLED_OK_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

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
/// `tail_inv` `(e²).max(1)`, `tail_col` `e.max(1)`, the four row vectors
/// `rows`, `d_gamma`/`l_gamma` `m`, `d_u`/`l_u`/`adj` `k.max(1)`, `g_gamma`
/// `(m·k).max(1)`, the four crossed-tail reducer temporaries `e.max(1)`, and
/// the observed factor `(s·q_core²).max(1)` / `(q_core·s·e).max(1)` /
/// `(e²).max(1)` on every shape, canonical or not — the same three shapes `DualStep`'s own
/// twin carries, which stays sized to zero on canonical links through
/// `pirls::obs_len`: that twin serves the kernel's own inexact step on a
/// clamped canonical fit, which the settle loop in `run_assembled_hessian`
/// resolves, not this buffer.
pub(crate) struct AssemblyBufs<T: Scalar> {
    pub(super) tail_inv: Vec<T>,
    pub(super) tail_col: Vec<T>,
    pub(super) rho: Vec<T>,
    pub(super) w_eta: Vec<T>,
    pub(super) w_obs: Vec<T>,
    pub(super) lev: Vec<T>,
    pub(super) d_gamma: Vec<T>,
    pub(super) l_gamma: Vec<T>,
    pub(super) d_u: Vec<T>,
    pub(super) l_u: Vec<T>,
    pub(super) g_gamma: Vec<T>,
    pub(super) adj: Vec<T>,
    pub(super) rb: Vec<T>,
    pub(super) sb: Vec<T>,
    pub(super) ra: Vec<T>,
    pub(super) sa: Vec<T>,
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
/// instead (see [`structured_row_reduce`]). At `k ≤ 356` and `width ≤ 8`
/// across the corpus that trade is the cheap side by orders of magnitude.
///
/// The evaluation state (`m_vals`, `eta`, `prob`, `w`, `u`) is held here on
/// BOTH instantiations, not only the dual one: at `T = f64` the mode solve
/// leaves it in the workspace and it is copied in, so the two arms hand the
/// assembly the same slices and there is no second code path.
///
/// Lengths, with `width` the packed row width and `rows` the row capacity:
/// `a`/`a_inv`/`obs` `k²` each — `obs` is sized on every shape, canonical or
/// not, because whether a fit's μ pins a row is not known at sizing time
/// — `col` `k.max(1)`, `m_vals` `rows·width`, `eta`/`prob`/`w` `rows`, `u`
/// `k.max(1)`, `m_deriv` `width`. Every one of them is empty on a non-packed
/// layout, which never reads them.
pub(crate) struct PackedAsmBufs<T: Scalar> {
    /// `k×k` row-major `A = M'WM + I`, factored in place by
    /// `workspace::glmm_block_chol`.
    pub(super) a: Vec<T>,
    /// `k×k` row-major `A⁻¹`, from `k` unit-vector solves against `a`.
    pub(super) a_inv: Vec<T>,
    /// `k×k` row-major `A_obs = M'W_obs M + I`, factored in place. Read on a
    /// non-canonical link, or a canonical one with a μ-clamped row, where it
    /// is the adjoint equation's `G_u/2`.
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
    /// yields [`PackedAsmBufs::empty`]. `obs` is sized on every shape,
    /// canonical or not: a canonical fit with a μ-clamped row still
    /// needs `A_obs` built and factored, and whether any row is pinned is not
    /// known this early.
    fn for_shape(k: usize, rows: usize, width: usize) -> PackedAsmBufs<T> {
        if width == 0 {
            return PackedAsmBufs::empty();
        }
        PackedAsmBufs {
            a: vec![T::ZERO; k * k],
            a_inv: vec![T::ZERO; k * k],
            obs: vec![T::ZERO; k * k],
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
            rho: Vec::new(),
            w_eta: Vec::new(),
            w_obs: Vec::new(),
            lev: Vec::new(),
            d_gamma: Vec::new(),
            l_gamma: Vec::new(),
            d_u: Vec::new(),
            l_u: Vec::new(),
            g_gamma: Vec::new(),
            adj: Vec::new(),
            rb: Vec::new(),
            sb: Vec::new(),
            ra: Vec::new(),
            sa: Vec::new(),
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
    /// `packed.obs`) is sized on every shape, canonical or not: whether
    /// a fit's μ pins a row, which needs `A_obs` even on a canonical link, is
    /// not known this early.
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
        packed_width: usize,
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
        let packed = PackedAsmBufs::for_shape(k, rows, packed_width);
        let (s, q_core, e) = if packed_width > 0 {
            (0, 1, 0)
        } else {
            (s, q_core, e)
        };
        let ee = e.max(1);
        AssemblyBufs {
            tail_inv: vec![T::ZERO; (e * e).max(1)],
            tail_col: vec![T::ZERO; ee],
            rho: vec![T::ZERO; rows],
            w_eta: vec![T::ZERO; rows],
            w_obs: vec![T::ZERO; rows],
            lev: vec![T::ZERO; rows],
            d_gamma: vec![T::ZERO; m],
            l_gamma: vec![T::ZERO; m],
            d_u: vec![T::ZERO; kk],
            l_u: vec![T::ZERO; kk],
            g_gamma: vec![T::ZERO; (m * k).max(1)],
            adj: vec![T::ZERO; kk],
            rb: vec![T::ZERO; ee],
            sb: vec![T::ZERO; ee],
            ra: vec![T::ZERO; ee],
            sa: vec![T::ZERO; ee],
            obs_core: vec![T::ZERO; (s * q_core * q_core).max(1)],
            obs_coup: vec![T::ZERO; (q_core * s * e).max(1)],
            obs_schur: vec![T::ZERO; (e * e).max(1)],
            packed,
        }
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

/// Applies the structured-extras cluster factor `A_f⁻¹` to one row's
/// `m_i = (m_{c,i}, m_{x,i})` through the block-inverse identity for
/// `A_f = [[D_f, C_f], [C_f', S]]` (Harville, D.A. (1997), *Matrix Algebra
/// From a Statistician's Perspective*, Springer, ch. 8):
///
/// ```text
///   A_f⁻¹ m_i = ( D_f⁻¹m_{c,i} − D_f⁻¹C_f·S⁻¹r_i ,  S⁻¹r_i ),
///   r_i = C_f'(D_f⁻¹m_{c,i}) − m_{x,i}
/// ```
///
/// so any bilinear form `m_i'A_f⁻¹m_j` over rows of the SAME cluster reduces
/// to two dots once each row has run through here once: `m_{c,i}'D_f⁻¹m_{c,j}
/// + r_i'S⁻¹r_j = y_i·m_{c,j} + sr_i·r_j`, using either row's own `y`/`sr` and
/// the other row's raw `m_c`/`r`. This function produces one row's
/// `(y_i, r_i, S⁻¹r_i)`: `y_i = D_f⁻¹m_{c,i}` (the full solve `glmm_block_solve`
/// runs against the cluster's factored core `fac`), and `r_i`/`S⁻¹r_i`
/// (`r`/`sr` out params) written only at `cols`, cluster `f`'s crossed
/// coupling columns (a strict superset of row `i`'s own crossed columns as
/// soon as `f` spans more than one crossed level).
///
/// Every entry of `r`/`sr` outside `cols` is left untouched and must not be
/// read. `cross_col`/`cross_val` are row `i`'s own nonzero crossed entries:
/// the packed builder's first `n_cross[i]` columns for this row, not the
/// `g_cap`-wide per-row block it packs them into.
///
/// `S` is supplied pre-inverted as `tail_inv` (column-major, `tail_inv[b·e+a]
/// = (S⁻¹)_{a,b}`) — the caller runs `e` `TailKernel::tail_solve` calls once
/// per cluster factor, not once per row, exactly as `blocked_extras.rs`'s pass
/// A does. Mirrors the blocked reducer (`block_forward_solve`) — change
/// together on the shared block-inverse identity.
#[allow(clippy::too_many_arguments)]
pub(crate) fn structured_row_reduce<T: Scalar>(
    fac: &[T],
    qc: usize,
    m_c: &[T],
    coupling: &[T],
    e: usize,
    cols: &[u32],
    cross_col: &[u32],
    cross_val: &[T],
    tail_inv: &[T],
    y: &mut [T],
    r: &mut [T],
    sr: &mut [T],
) {
    y[..qc].copy_from_slice(&m_c[..qc]);
    glmm_block_solve(fac, qc, &mut y[..qc]);
    if cols.is_empty() {
        return;
    }
    for &b in cols {
        let b = b as usize;
        let mut acc = T::ZERO;
        for local in 0..qc {
            acc += coupling[local * e + b] * y[local];
        }
        r[b] = acc;
    }
    for (&col, &val) in cross_col.iter().zip(cross_val) {
        r[col as usize] -= val;
    }
    for &b in cols {
        let b = b as usize;
        let col = &tail_inv[b * e..b * e + e];
        let mut acc = T::ZERO;
        for &az in cols {
            let az = az as usize;
            acc += r[az] * col[az];
        }
        sr[b] = acc;
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
        let (dev, conv, _raw_finite) = structured_laplace_deviance::<f64>(
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
        let (dev, conv, _raw_finite) = blocked_laplace_deviance::<f64>(
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
/// [`packed_peak_bytes`] is that expression. `A_obs` is sized on every shape,
/// canonical or not (a canonical fit with a μ-clamped row needs it too,
/// and whether a row will be pinned is not known at sizing time), so the
/// `set` term is a flat `3·k²` rather than `(2 + obs)·k²`. What it leaves out
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

/// Rows where μ sits on one of `family::clamp_mu`'s bounds at the mode state
/// held in `prob`, counted over the `n` fitted rows — `family::pinned_mu_bounds`
/// is zero on any row for unweighted Bernoulli logit, whatever `prob` holds:
/// that route's family pass is the fused `log1pexp` identity and calls
/// `family::clamp_mu` on no row, so a saturated row there is an ordinary row
/// whose deviance slope is the exact `−2(y − σ(η))` the assembly already
/// writes.
///
/// A non-zero count refuses a fit only on the weighted binomial logit link
/// ([`logit_clamp_refused`], this count's only reader); on every other link a
/// clamped row is handled in place. The reason is written at the census in
/// [`joint_hessian_columns`].
pub(crate) fn mu_clamped_rows(family: Family, weighted: bool, prob: &[f64]) -> usize {
    let (mu_lo, mu_hi) = crate::family::pinned_mu_bounds(family, weighted);
    prob.iter()
        .filter(|&&mu| mu <= mu_lo || mu >= mu_hi)
        .count()
}

/// Rows where η sits on one of `family::clamp_eta`'s bounds at the mode state
/// held in `eta`, counted over the `n` fitted rows — the η twin of
/// [`mu_clamped_rows`]. On such a row the clamped η is a constant, so μ, the
/// score and the weight stop depending on `u` and γ there, which the
/// assembly's per-row derivatives do not model. Every entry point here
/// refuses a fit with a non-zero count.
pub(crate) fn eta_clamped_rows(family: Family, eta: &[f64]) -> usize {
    let (eta_lo, eta_hi) = crate::family::clamp_eta_bounds(family);
    eta.iter().filter(|&&e| e <= eta_lo || e >= eta_hi).count()
}

/// True on a μ-clamped row on the binomial logit link, whatever the
/// weighting. `assemble` and `packed_assemble` write the logit score as
/// `prior_w·(y − μ)` on both routes, while the structured kernel writes the
/// general form when weighted and the packed kernel always does; on a
/// clamped row those are different numbers, so the assembly's mode equation
/// is not the one PIRLS solved there and the adjoint would be wrong. Zero for
/// unweighted logit: [`mu_clamped_rows`] already returns `0` for that route,
/// so this refuses exactly the weighted-logit case.
pub(crate) fn logit_clamp_refused(family: Family, weighted: bool, prob: &[f64]) -> bool {
    mu_clamped_rows(family, weighted, prob) > 0
        && matches!(
            family,
            Family::Binomial {
                link: BinomialLink::Logit
            }
        )
}

/// Shared body of [`gradient_f64`] and [`gradient_f64_mode_residual`]: solves
/// the conditional mode once, then runs three row passes over the buffers the
/// solve left behind (no second solve, no dual lanes):
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
///
/// `mode_residual`, when `Some`, is filled with the exact mode residual
/// `‖G_u‖`, `G_u = D_u + 2u` (this module's own `G(γ,u)`), read off the row
/// pass's own unfloored `D_u` before `u` is restored — the quantity every
/// formula here assumes is zero. `D_u`'s core block is packed `[f·qc + local]`
/// (the row pass's own per-cluster order); `u` is in RE-column order,
/// `core_col(f, local)`, and the two coincide only at `np == 0`.
/// Mirrors `blocked_extras.rs`'s `gu_dot_du` comment on the same split —
/// change together. The crossed tail needs no remapping: both sides use
/// `k_family + b`. Both halves are read at the returned `u`: `D_u` is formed
/// from `rho`, which pass 1 reads off `eta`/`prob` — the η-dependent state the
/// mode solve leaves at `u`, per this module's own evaluation-point rule — so
/// pairing it with `2u` reads one residual at one iterate and reports the mode
/// equation.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn gradient_f64_impl(
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
    mode_residual: Option<&mut f64>,
) -> Option<()> {
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
    let weighted = ws.weighted;
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
    // the same reasons and stated there.
    if logit_clamp_refused(family, weighted, &ws.pirls.prob[..n])
        || eta_clamped_rows(family, &ws.pirls.eta[..n]) > 0
    {
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
                cross_val,
            },
        pattern:
            StructuredPattern {
                cross_col,
                n_cross,
                coup_cols,
                coup_ptr,
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
        0,
    );

    let out = assemble(
        g,
        family,
        weighted,
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
        cross_val,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        n_theta,
        p,
        n,
        &mut asm,
        grad,
    );
    if let (Some(residual), true) = (mode_residual, out.is_some()) {
        let q = g.primary_q;
        let s = g.n_primary;
        let np = g.nested_per_parent;
        let qc = q + np;
        let prim_width = q * s;
        let k_family = qc * s;
        let core_col = |f: usize, local: usize| -> usize {
            if local < q {
                f * q + local
            } else {
                prim_width + f * np + (local - q)
            }
        };
        let mut worst = 0.0f64;
        for f in 0..s {
            for local in 0..qc {
                let u_idx = core_col(f, local);
                worst = worst.max((asm.d_u[f * qc + local] + 2.0 * u[u_idx]).abs());
            }
        }
        for (&du, &up) in asm.d_u[k_family..k].iter().zip(&u[k_family..k]) {
            worst = worst.max((du + 2.0 * up).abs());
        }
        *residual = worst;
    }
    u[..kk].copy_from_slice(&saved_u); // restore — leave ws.pirls.u as found
    out
}

/// The Laplace gradient `dD*/dγ` assembled in `f64`, `γ = [θ | β]`, written
/// into `grad[..n_θ + p]`.
///
/// The `f64` reference instantiation of the assembly, and the only one that
/// owns its scratch: the shape-sized [`AssemblyBufs`] every other caller reads
/// off the dual scratch has no `f64` variant there.
///
/// A test instrument, not a fit-path entry: it checks the assembly at
/// `T = f64` against the dual-lane gradient (`laplace_gradient`) as an
/// independent way of differentiating the same objective once. Production
/// reaches this assembly only through [`joint_hessian`].
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_f64(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> Option<()> {
    gradient_f64_impl(ws, x, y, cluster_ids, extra_ids, p, n, grad, None)
}

/// Test-only twin of [`gradient_f64`] that also returns the exact mode
/// residual `‖G_u‖` [`gradient_f64_impl`]'s doc comment describes. `None` on
/// every case `gradient_f64` returns `None` on.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gradient_f64_mode_residual(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> Option<f64> {
    let mut residual = 0.0f64;
    let out = gradient_f64_impl(
        ws,
        x,
        y,
        cluster_ids,
        extra_ids,
        p,
        n,
        grad,
        Some(&mut residual),
    );
    out.map(|_| residual)
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
        #[cfg(test)]
        ASSEMBLED_OK_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    #[cfg(test)]
    if FORCE_DECLINE.load(std::sync::atomic::Ordering::Relaxed) {
        return DerivStatus::Unsupported;
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
    // gradient does.
    let nl = NLanes::pick(m, false).expect("the gradient ladder covers every m");
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
    let need_build = ws.dual_scratch.as_deref().is_none_or(|sc| {
        sc.lanes() != nl || !sc.matches_shape(m, p, k, n, s, q_p, q_core, e, nagq, observed, 0)
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
            nagq,
            observed,
            // The packed layout left through its own driver above.
            0,
            cluster_ids,
        )));
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
    // adjoint below differentiates. Two things break that on a mode state
    // PIRLS can actually reach: a row on the link's η bound has a score that
    // has stopped moving to first order relative to what this engine assumes,
    // and a μ-clamped row on the weighted logit link has the kernel writing a
    // different score than the general form `assemble` writes there
    // (`logit_clamp_refused`, unweighted logit exempt — its fused kernel
    // applies no μ clamp). A μ-clamped row on any other link keeps `ρ`'s own
    // expression, unchanged, at the pinned μ; only its deviance slope (0
    // there, μ being a constant) and its observed weight and `dw/dη`
    // (`family::clamped_observed_weight`/`clamped_weight_eta_deriv`, in place
    // of the general closed forms) read differently.
    if logit_clamp_refused(family, weighted, &ws.pirls.prob[..n])
        || eta_clamped_rows(family, &ws.pirls.eta[..n]) > 0
    {
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
        ($bufs:expr, $mode:expr) => {
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
        };
    }
    let st = match &mut *scratch {
        GlmmDualScratch::D4(bufs, _, mode) => hess_arm!(bufs, mode),
        GlmmDualScratch::D5(bufs, _, mode) => hess_arm!(bufs, mode),
        GlmmDualScratch::D6(bufs, _, mode) => hess_arm!(bufs, mode),
        GlmmDualScratch::D8(bufs, _, mode) => hess_arm!(bufs, mode),
        GlmmDualScratch::D12(bufs, _, mode) => hess_arm!(bufs, mode),
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

/// The chunked seed-call-assemble body [`joint_hessian_columns`]'s per-`N`
/// match arms hand a typed buffer set to. A call that comes back `!exact`
/// means the step matrix that call took was not the Jacobian of the map PIRLS
/// walks — a row on the kernel's μ clamp, or a non-PD observed factor inside
/// the kernel (`pirls::DualStep::exact`'s own doc comment has the two
/// causes) — so the assembled columns from that call are not yet the answer.
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
            let (value, conv) = if extras {
                let (obj, conv, _raw_finite) = structured_laplace_deviance::<T>(
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
                let (obj, conv, _raw_finite) = blocked_laplace_deviance::<T>(
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
                weighted,
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
                &bufs.structured.cross_val,
                &pattern.cross_col,
                &pattern.n_cross,
                &pattern.coup_cols,
                &pattern.coup_ptr,
                n_theta,
                p,
                n,
                &mut bufs.asm,
                &mut bufs.grad_t,
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
/// quantity — `observed_weight`, `weight_eta_deriv`, `gamma_phi_prime` — so
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
    weighted: bool,
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
    cross_val: &[T],
    cross_col: &[u32],
    n_cross: &[u8],
    coup_cols: &[u32],
    coup_ptr: &[u32],
    n_theta: usize,
    p: usize,
    n: usize,
    asm: &mut AssemblyBufs<T>,
    grad: &mut [T],
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
        rho,
        w_eta,
        w_obs,
        lev,
        d_gamma,
        l_gamma,
        d_u,
        l_u,
        // `G_γ` row-major, `g_gamma[a·k + c] = ∂G_c/∂γ_a`.
        g_gamma,
        adj,
        rb,
        sb,
        ra,
        sa,
        obs_core,
        obs_coup,
        obs_schur,
        ..
    } = asm;
    // The accumulators are `+=`d row by row, so they start at zero; `rho`,
    // `w_eta`, `w_obs`, `lev` and `adj` are assigned outright and need no
    // clear.
    d_gamma[..m].fill(T::ZERO);
    l_gamma[..m].fill(T::ZERO);
    d_u[..k].fill(T::ZERO);
    l_u[..k].fill(T::ZERO);
    g_gamma[..m * k].fill(T::ZERO);

    // `S⁻¹` column by column, through the same dense tail substitution the
    // mode solve's own factor was left by; column-major, `tail_inv[b·e+a] =
    // (S⁻¹)_{a,b}`, the layout the per-row reducer reads.
    if e > 0 {
        for b in 0..e {
            tail_col[..e].fill(T::ZERO);
            tail_col[b] = T::ONE;
            T::tail_solve(schur_blk, e, None, &mut tail_col[..e]);
            tail_inv[b * e..b * e + e].copy_from_slice(&tail_col[..e]);
        }
    }

    let canonical = crate::family::is_canonical(family);
    let (mu_lo, mu_hi) = crate::family::pinned_mu_bounds(family, weighted);
    let mut pinned_rows = 0usize;
    let mut t = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut yb = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];

    // --- pass 1, per row:
    //   ρᵢ = prior_wᵢ·(dμ/dη)(yᵢ−μᵢ)/V,  ∂(prior_wᵢ·devᵢ)/∂ηᵢ = −2ρᵢ on an
    //   unpinned row and 0 on a row whose μ sits on a `family::clamp_mu`
    //   bound (μ is a constant there); ∂²/∂η² = 2·w_obs,ᵢ, hᵢ = mᵢ'A⁻¹mᵢ
    //   D_βj  += (∂devᵢ/∂ηᵢ)·xᵢⱼ            D_uc  += (∂devᵢ/∂ηᵢ)·M_ic
    //   ℓ_βj  += w'ᵢ·xᵢⱼ·hᵢ                ℓ_uc  += w'ᵢ·M_ic·hᵢ
    //   (G_βj)_c += 2·w_obs,ᵢ·xᵢⱼ·M_ic     (∂M/∂β = 0, so this is all of G_β)
    // ---
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let cb = f * qc * qc;
        let fac = &core_fac[cb..cb + qc * qc];
        let mc = &core_m[i * qc..i * qc + qc];
        let cbase = i * g_cap;
        let ncz = if extras { n_cross[i] as usize } else { 0 };
        let cols: &[u32] = if e > 0 {
            &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize]
        } else {
            &[]
        };
        let h = if e == 0 {
            block_forward_solve(fac, qc, mc, &mut t[..qc]);
            let mut acc = T::ZERO;
            for v in t.iter().take(qc) {
                acc += *v * *v;
            }
            acc
        } else {
            structured_row_reduce(
                fac,
                qc,
                mc,
                &coupling[f * qc * e..],
                e,
                cols,
                &cross_col[cbase..cbase + ncz],
                &cross_val[cbase..cbase + ncz],
                tail_inv,
                &mut yb[..qc],
                rb,
                sb,
            );
            // Two separate folds, then one add — the association the `f64`
            // instantiation has to keep bit for bit.
            let mut core_acc = T::ZERO;
            for c in 0..qc {
                core_acc += yb[c] * mc[c];
            }
            let mut tail_acc = T::ZERO;
            for &b in cols {
                tail_acc += sb[b as usize] * rb[b as usize];
            }
            core_acc + tail_acc
        };
        let r = match family {
            Family::Binomial {
                link: BinomialLink::Logit,
            } => T::from_f64(prior_w[i]) * (T::from_f64(y[i]) - prob[i]),
            other => {
                let dmu = crate::family::mu_eta(other, eta[i]);
                let v = crate::family::variance(other, nb_theta, prob[i]);
                T::from_f64(prior_w[i]) * dmu * (T::from_f64(y[i]) - prob[i]) / v
            }
        };
        rho[i] = r;
        let pinned = prob[i].value() <= mu_lo || prob[i].value() >= mu_hi;
        pinned_rows += usize::from(pinned);
        lev[i] = h;
        // On a pinned row μ is the constant `prob[i]`, so only `dμ/dη` still
        // moves with η — the clamped closed forms read that directly instead
        // of the general ones, which assume μ is still a function of η.
        w_eta[i] = if pinned {
            crate::family::clamped_weight_eta_deriv(family, nb_theta, prior_w[i], eta[i], prob[i])
        } else {
            crate::family::weight_eta_deriv(family, nb_theta, eta[i], prob[i], w[i])
        };
        // The kernel's stored `w` is the raw Fisher weight
        // `prior_w·(dμ/dη)²/V`, the same weight `A = M'WM + I` is built from,
        // and `w_obs` is formed from it.
        w_obs[i] = if pinned {
            crate::family::clamped_observed_weight(
                family, nb_theta, y[i], prior_w[i], eta[i], prob[i],
            )
        } else {
            crate::family::observed_weight(
                family, nb_theta, y[i], prior_w[i], eta[i], prob[i], w[i],
            )
        };
        let dr = T::from_f64(-2.0) * r;
        // `dr` unsplit is still the score `G` differentiates — pass 2's
        // θ columns of `G_γ` keep it. `D_γ`/`D_u` take the deviance's own
        // slope, which is 0 on a pinned row.
        let dev_eta = if pinned { T::ZERO } else { dr };
        let a = w_eta[i] * h;
        let two_wo = T::from_f64(2.0) * w_obs[i];
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
            let v = cross_val[cbase + z];
            d_u[c] += dev_eta * v;
            l_u[c] += a * v;
            let gb = two_wo * v;
            for j in 0..p {
                g_gamma[(n_theta + j) * k + c] += gb * T::from_f64(x[(i, j)]);
            }
        }
    }

    // --- pass 2, per row × per θ coordinate. `Λ` is linear in θ, so
    // `m_{a,i} = ∂mᵢ/∂θ_a` is a selection; `∂ηᵢ/∂θ_a = m_{a,i}'u`:
    //   D_θa  += −2ρᵢ·(m_{a,i}'u)
    //   ℓ_θa  += 2·wᵢ·(mᵢ'A⁻¹m_{a,i}) + w'ᵢ·(m_{a,i}'u)·hᵢ
    //   (G_θa)_c += 2·w_obs,ᵢ·(m_{a,i}'u)·M_ic + (−2ρᵢ)·(m_{a,i})_c
    // The cross-leverage `mᵢ'A⁻¹m_{a,i}` is the one object today's row passes
    // do not already form; it is one dot against the row reduction of `mᵢ`.
    // ---
    let mut mad_f64 = [0.0f64; crate::lmm::MAX_PRIMARY_Q];
    let mut cvd_f64 = [0.0f64; crate::lmm::MAX_EXTRA_GROUPINGS];
    let mut mad = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut cvd = [T::ZERO; crate::lmm::MAX_EXTRA_GROUPINGS];
    let mut ta = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut ya = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let cb = f * qc * qc;
        let fac = &core_fac[cb..cb + qc * qc];
        let mc = &core_m[i * qc..i * qc + qc];
        let cbase = i * g_cap;
        let ncz = if extras { n_cross[i] as usize } else { 0 };
        let cols: &[u32] = if e > 0 {
            &coup_cols[coup_ptr[f] as usize..coup_ptr[f + 1] as usize]
        } else {
            &[]
        };
        if e == 0 {
            block_forward_solve(fac, qc, mc, &mut t[..qc]);
        } else {
            structured_row_reduce(
                fac,
                qc,
                mc,
                &coupling[f * qc * e..],
                e,
                cols,
                &cross_col[cbase..cbase + ncz],
                &cross_val[cbase..cbase + ncz],
                tail_inv,
                &mut yb[..qc],
                rb,
                sb,
            );
        }
        let dr = T::from_f64(-2.0) * rho[i];
        // `dev_eta` is `D_γ`'s slope, 0 on a pinned row; `G_γ`'s θ columns
        // below keep the unsplit `dr`, mirroring pass 1's split.
        let pinned = prob[i].value() <= mu_lo || prob[i].value() >= mu_hi;
        let dev_eta = if pinned { T::ZERO } else { dr };
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
                structured_row_reduce(
                    fac,
                    qc,
                    &mad[..qc],
                    &coupling[f * qc * e..],
                    e,
                    cols,
                    &cross_col[cbase..cbase + ncz],
                    &cvd[..ncz],
                    tail_inv,
                    &mut ya[..qc],
                    ra,
                    sa,
                );
                // Same two-fold association as pass 1's own leverage.
                let mut core_acc = T::ZERO;
                for c in 0..qc {
                    core_acc += yb[c] * mad[c];
                }
                let mut tail_acc = T::ZERO;
                for &b in cols {
                    tail_acc += sb[b as usize] * ra[b as usize];
                }
                core_acc + tail_acc
            };
            d_gamma[a] += dev_eta * deta;
            l_gamma[a] += T::from_f64(2.0) * w[i] * cross + w_eta[i] * deta * lev[i];
            let tw = T::from_f64(2.0) * w_obs[i] * deta;
            for local in 0..qc {
                g_gamma[a * k + f * qc + local] += tw * mc[local] + dr * mad[local];
            }
            for z in 0..ncz {
                let c = k_family + cross_col[cbase + z] as usize;
                g_gamma[a * k + c] += tw * cross_val[cbase + z] + dr * cvd[z];
            }
        }
    }

    // --- pass 3. `Φ = D` on every family but Gamma, whose objective
    // substitutes the profiled-dispersion `aic`; `Φ' = ∂aic/∂D` in closed form
    // there, so the substitution is one multiply rather than a second path.
    //   F_γ = Φ'·D_γ + ℓ_γ        F_u = Φ'·D_u + 2u + ℓ_u
    //   G_u = 2·A_obs             adj = G_u⁻¹F_u
    //   D*_γ = F_γ − adj'·G_γ
    // ---
    let phi = if matches!(family, Family::Gamma { .. }) {
        let mut dev = T::ZERO;
        for i in 0..n {
            dev +=
                T::from_f64(prior_w[i]) * crate::family::dev_resid(family, nb_theta, y[i], prob[i]);
        }
        crate::family::gamma_phi_prime(dev, n, Some(prior_w))
    } else {
        T::ONE
    };
    for f in 0..s {
        for local in 0..qc {
            let c = f * qc + local;
            adj[c] = phi * d_u[c] + T::from_f64(2.0) * u[core_col(f, local)] + l_u[c];
        }
    }
    for b in 0..e {
        let c = k_family + b;
        adj[c] = phi * d_u[c] + T::from_f64(2.0) * u[c] + l_u[c];
    }
    // `A_obs = M'W_obs M + I`. On a canonical link with every row's μ off its
    // `family::clamp_mu` bound, `A_obs` IS the Fisher `A` the mode solve
    // already factored; a pinned row makes the canonical link's own observed
    // weight something other than `V(μ)` there (`clamped_observed_weight`),
    // so on such a row the factor is built and factored here regardless of
    // canonicity, and a non-PD factor is the refusal. Built here rather than
    // taken from the dual kernel's own observed twin because a
    // `BetaStep::Fixed` `f64` call leaves no twin at all, so this arm has to
    // exist for the `f64` instantiation regardless — one body serves both.
    if canonical && pinned_rows == 0 {
        if e == 0 {
            solve_core_blocks(core_fac, qc, s, adj);
        } else {
            structured_ainv_solve(
                g, core_fac, coupling, schur_blk, coup_cols, coup_ptr, None, adj,
            );
        }
    } else {
        obs_core.fill(T::ZERO);
        obs_coup.fill(T::ZERO);
        obs_schur.fill(T::ZERO);
        for i in 0..n {
            let f = cluster_ids[i] as usize;
            let cb = f * qc * qc;
            let coup = f * qc * e;
            let mc = &core_m[i * qc..i * qc + qc];
            let cbase = i * g_cap;
            let ncz = if extras { n_cross[i] as usize } else { 0 };
            let wo = w_obs[i];
            for r in 0..qc {
                let wmr = wo * mc[r];
                for c in 0..=r {
                    obs_core[cb + r * qc + c] += wmr * mc[c];
                }
                for z in 0..ncz {
                    obs_coup[coup + r * e + cross_col[cbase + z] as usize] +=
                        wmr * cross_val[cbase + z];
                }
            }
            for z in 0..ncz {
                let b = cross_col[cbase + z] as usize;
                let wvb = wo * cross_val[cbase + z];
                for z2 in 0..ncz {
                    let b2 = cross_col[cbase + z2] as usize;
                    if b2 <= b {
                        obs_schur[b * e + b2] += wvb * cross_val[cbase + z2];
                    }
                }
            }
        }
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
        let mut acc = phi * d_gamma[a] + l_gamma[a];
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
        offset: offset_field,
        ..
    } = ws;
    let offset = offset_field.as_deref();
    crate::sparse::fill_lambda_small(&prm[..n_theta], groupings, &mut packed.lam_small);
    fill_m_vals(packed, groupings, x, n);
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
    // The same `aic`-for-deviance substitution the packed deviance arm makes,
    // so this engine differentiates the objective that arm evaluates.
    let data_term = if matches!(family, Family::Gamma { .. }) {
        crate::family::gamma_aic(y, &pirls.prob, dev, n, Some(&prior_w[..n]))
    } else {
        dev
    };
    let obj = data_term + pen + 2.0 * logdet;
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
    fn for_shape(m: usize, k: usize, rows: usize, width: usize) -> Self {
        PackedGradientBufs {
            // `s`/`q_core`/`e` are the other routes' shape terms and are
            // unread here, so they go in at their minimum.
            asm: AssemblyBufs::<f64>::for_shape(m, k, rows, 0, 1, 0, width),
            u_lanes: vec![0.0; (m * k).max(1)],
            grad: vec![0.0; m],
            saved_u: vec![0.0; k.max(1)],
            u_mode: vec![0.0; k.max(1)],
        }
    }

    /// Shape half of the reuse policy, mirroring
    /// `derivative::GlmmDualScratch::matches_shape`: a request at the same
    /// shape reuses, a different one reallocates once.
    fn matches_shape(&self, m: usize, k: usize, rows: usize, width: usize) -> bool {
        self.grad.len() == m
            && self.u_lanes.len() == (m * k).max(1)
            && self.saved_u.len() == k.max(1)
            && self.asm.g_gamma.len() == (m * k).max(1)
            && self.asm.packed.m_vals.len() == rows * width
            && self.asm.packed.a.len() == k * k
            && self.asm.packed.obs.len() == k * k
    }
}

/// The workspace's packed `f64` scratch, built on the first request for this
/// shape and reused after. Taken OUT of the workspace: every entry point below
/// hands `&mut ws` to the mode solve while holding these buffers.
fn take_packed_bufs(ws: &mut GlmmWorkspace, p: usize, n: usize) -> Box<PackedGradientBufs> {
    let m = ws.n_theta + p;
    let (k, width) = (ws.k, ws.packed.width);
    let need_build = ws
        .packed_asm
        .as_deref()
        .is_none_or(|b| !b.matches_shape(m, k, n, width));
    if need_build {
        ws.packed_asm = Some(Box::new(PackedGradientBufs::for_shape(m, k, n, width)));
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
/// [`joint_hessian_columns`]: `Unsupported` on a mode state where the μ clamp
/// binds — the reason is stated there — or on a non-positive-definite
/// observed factor, `NotConverged` on a mode solve that fails.
///
/// The workspace comes back as found in `u`; `eta`, `prob`, `w`, `mu`,
/// `beta_rhs`, `ws.packed.m_vals` and the packed factor are left at the mode
/// solve's values, exactly as `laplace_gradient` leaves the blocked ones.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn packed_gradient(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    p: usize,
    n: usize,
    grad: &mut [f64],
) -> DerivStatus {
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
    // The same census, for the same reasons, as `joint_hessian_columns` —
    // stated there.
    if logit_clamp_refused(family, ws.weighted, &ws.pirls.prob[..n])
        || eta_clamped_rows(family, &ws.pirls.eta[..n]) > 0
    {
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
        ws.weighted,
        nb_theta,
        x,
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
    let Some(pinned) = out else {
        return DerivStatus::Unsupported;
    };
    // `û(γ)` solves `G(γ, û) = 0`, so `dû/dγ = −G_u⁻¹G_γ` — `m` right-hand
    // sides against the SAME factor the assembly's own pass 3 just used for
    // its adjoint (`A` on a canonical link with no pinned row, `A_obs`
    // otherwise — `pinned` says which), and `G_u = 2·A_obs` puts the ½ in
    // front.
    let fac_len = k * k;
    for a in 0..m {
        let col = &mut bufs.u_lanes[a * k..a * k + k];
        col.copy_from_slice(&bufs.asm.g_gamma[a * k..a * k + k]);
        let fac = if canonical && !pinned {
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
/// Mirrors the blocked and structured reducers (`block_forward_solve`,
/// [`structured_row_reduce`]) on the shared block-inverse identity — change
/// together. Where those reduce each row against a per-cluster factor, this
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
/// silently put in the observed one's place. `Some(pinned)` otherwise, with
/// `pinned` true when this call's pass 3 built and factored `A_obs` because
/// some row's μ sat on a `family::pinned_mu_bounds` bound — the caller needs
/// it to pick the same factor `A`/`A_obs` for `dû/dγ`.
#[allow(clippy::too_many_arguments)]
fn packed_assemble<T: Scalar>(
    g: &LmmGroupings,
    family: Family,
    weighted: bool,
    nb_theta: f64,
    x: MatRef<f64>,
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
) -> Option<bool> {
    let m = n_theta + p;
    let canonical = crate::family::is_canonical(family);
    let (mu_lo, mu_hi) = crate::family::pinned_mu_bounds(family, weighted);
    let mut pinned_rows = 0usize;
    let AssemblyBufs {
        rho,
        w_eta,
        w_obs,
        lev,
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

    // --- pass 1, per row. Identical statement to `assemble`'s pass 1; the
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
            other => {
                let dmu = crate::family::mu_eta(other, eta[i]);
                let v = crate::family::variance(other, nb_theta, prob[i]);
                T::from_f64(prior_w[i]) * dmu * (T::from_f64(y[i]) - prob[i]) / v
            }
        };
        rho[i] = r;
        let pinned = prob[i].value() <= mu_lo || prob[i].value() >= mu_hi;
        pinned_rows += usize::from(pinned);
        lev[i] = h;
        // Mirrors `assemble` — change together.
        w_eta[i] = if pinned {
            crate::family::clamped_weight_eta_deriv(family, nb_theta, prior_w[i], eta[i], prob[i])
        } else {
            crate::family::weight_eta_deriv(family, nb_theta, eta[i], prob[i], w[i])
        };
        w_obs[i] = if pinned {
            crate::family::clamped_observed_weight(
                family, nb_theta, y[i], prior_w[i], eta[i], prob[i],
            )
        } else {
            crate::family::observed_weight(
                family, nb_theta, y[i], prior_w[i], eta[i], prob[i], w[i],
            )
        };
        let dr = T::from_f64(-2.0) * r;
        let dev_eta = if pinned { T::ZERO } else { dr };
        let av = w_eta[i] * h;
        let two_wo = T::from_f64(2.0) * w_obs[i];
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
    }

    // --- pass 2, per row × per θ coordinate, as `assemble`'s pass 2. ---
    for i in 0..n {
        let base = i * width;
        let dr = T::from_f64(-2.0) * rho[i];
        let pinned = prob[i].value() <= mu_lo || prob[i].value() >= mu_hi;
        let dev_eta = if pinned { T::ZERO } else { dr };
        for a_idx in 0..n_theta {
            packed_m_vals_theta_deriv(g, a_idx, x, i, m_deriv);
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
            d_gamma[a_idx] += dev_eta * deta;
            l_gamma[a_idx] += T::from_f64(2.0) * w[i] * cross + w_eta[i] * deta * lev[i];
            let tw = T::from_f64(2.0) * w_obs[i] * deta;
            for t in 0..width {
                let c = m_cols[base + t] as usize;
                g_gamma[a_idx * k + c] += tw * m_vals[base + t] + dr * T::from_f64(m_deriv[t]);
            }
        }
    }

    // --- pass 3, as `assemble`'s pass 3: `F_γ = Φ'·D_γ + ℓ_γ`,
    // `F_u = Φ'·D_u + 2u + ℓ_u`, `adj = (2·A_obs)⁻¹F_u`,
    // `D*_γ = F_γ − adj'·G_γ`. ---
    let phi = if matches!(family, Family::Gamma { .. }) {
        let mut dev = T::ZERO;
        for i in 0..n {
            dev +=
                T::from_f64(prior_w[i]) * crate::family::dev_resid(family, nb_theta, y[i], prob[i]);
        }
        crate::family::gamma_phi_prime(dev, n, Some(prior_w))
    } else {
        T::ONE
    };
    for c in 0..k {
        adj[c] = phi * d_u[c] + T::from_f64(2.0) * u[c] + l_u[c];
    }
    // `A_obs` IS the Fisher `A` only on a canonical link with every row's μ
    // off its `clamp_mu` bound; a pinned row builds and factors it here
    // regardless of canonicity — mirrors `assemble`'s pass 3.
    if canonical && pinned_rows == 0 {
        glmm_block_solve(&a[..k * k], k, &mut adj[..k]);
    } else {
        obs[..k * k].fill(T::ZERO);
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let base = i * width;
            let wo = w_obs[i];
            for ta in base..base + width {
                let ca = m_cols[ta] as usize;
                let wva = wo * m_vals[ta];
                for tb in base..base + width {
                    obs[ca * k + m_cols[tb] as usize] += wva * m_vals[tb];
                }
            }
        }
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
        let mut acc = phi * d_gamma[a_idx] + l_gamma[a_idx];
        for c in 0..k {
            acc -= adj[c] * g_gamma[a_idx * k + c];
        }
        grad[a_idx] = acc;
    }
    Some(pinned_rows > 0)
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
    // and chunks like a gradient does.
    let nl = NLanes::pick(m, false).expect("the gradient ladder covers every m");
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
        sc.lanes() != nl || !sc.matches_shape(m, p, k, n, s, q_p, q_core, e, nagq, observed, width)
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
            nagq,
            observed,
            width,
            cluster_ids,
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
                    packed_m_vals_theta_deriv(g, a_idx, x, i, m_deriv);
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
        }
        let assembled = packed_assemble(
            g,
            family,
            weighted,
            nb_theta,
            x,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glmm::pirls::{block_leverage, TailKernel};
    use crate::glmm::workspace::glmm_block_chol;

    /// `structured_row_reduce`'s `(y_i, r_i, S⁻¹r_i)` must reproduce the
    /// block-inverse leverage `blocked_extras.rs`'s pass A computes directly
    /// (`h_i = m_i'A_f⁻¹m_i`), on a hand-built two-cluster structured factor
    /// (`qc = 2`, `e = 2`, dense Schur arm — no `StructuredSchur` needed):
    /// `y_i·m_{c,i} + sr_i·r_i` over `cols` must equal `h_i`.
    #[test]
    fn structured_row_reduce_matches_block_inverse_leverage() {
        let qc = 2;
        let e = 2;

        // Cluster 0's SPD core D_0 = B0 B0' + I, factored in place.
        let b0 = [1.3, 0.0, 0.4, 0.9];
        let mut fac0 = [0.0; 4];
        for r in 0..qc {
            for c in 0..qc {
                for k in 0..qc {
                    fac0[r * qc + c] += b0[r * qc + k] * b0[c * qc + k];
                }
            }
            fac0[r * qc + r] += 1.0;
        }
        assert!(glmm_block_chol(&mut fac0, qc));

        let b1 = [0.8, 0.0, -0.5, 1.1];
        let mut fac1 = [0.0; 4];
        for r in 0..qc {
            for c in 0..qc {
                for k in 0..qc {
                    fac1[r * qc + c] += b1[r * qc + k] * b1[c * qc + k];
                }
            }
            fac1[r * qc + r] += 1.0;
        }
        assert!(glmm_block_chol(&mut fac1, qc));

        // Coupling blocks, qc×e row-major.
        let coup0 = [0.3, -0.2, 0.1, 0.4];
        let coup1 = [-0.1, 0.5, 0.2, -0.3];

        // Schur S = (E + I) − Σ_f C_f'D_f⁻¹C_f, dense arm (cols = 0..e, ss = None).
        let mut schur = [0.0; 4];
        schur[0] = 2.0; // (E+I)_{00}
        schur[e + 1] = 2.0; // (E+I)_{11}
        let cols: [u32; 2] = [0, 1];
        <f64 as TailKernel>::tail_downdate(&fac0, qc, &coup0, e, &cols, None, &mut schur);
        <f64 as TailKernel>::tail_downdate(&fac1, qc, &coup1, e, &cols, None, &mut schur);
        let mut schur_fac = schur;
        assert!(<f64 as TailKernel>::tail_factor(&mut schur_fac, e, None).is_some());

        // S⁻¹ columns, column-major: tail_inv[b·e+a] = (S⁻¹)_{a,b}.
        let mut tail_inv = [0.0; 4];
        for b in 0..e {
            let mut rhs = [0.0; 2];
            rhs[b] = 1.0;
            <f64 as TailKernel>::tail_solve(&schur_fac, e, None, &mut rhs);
            for a in 0..e {
                tail_inv[b * e + a] = rhs[a];
            }
        }

        // One test row per cluster.
        #[allow(clippy::type_complexity)]
        let rows: [(&[f64], &[f64], [f64; 2], [u32; 1], [f64; 1]); 2] = [
            (&fac0, &coup0, [0.5, -0.7], [1], [0.6]),
            (&fac1, &coup1, [-0.3, 0.9], [0], [-0.4]),
        ];

        for (fac, coup, m_c, cross_col, cross_val) in rows {
            // Direct block-inverse leverage, mirroring blocked_extras.rs pass A.
            let mut y_direct = m_c;
            glmm_block_solve(fac, qc, &mut y_direct);
            let mut h = block_leverage(fac, qc, &m_c);
            let mut r_direct = [0.0; 2];
            for &b in &cols {
                let b = b as usize;
                let mut acc = 0.0;
                for local in 0..qc {
                    acc += coup[local * e + b] * y_direct[local];
                }
                r_direct[b] = acc;
            }
            for (&col, val) in cross_col.iter().zip(cross_val) {
                r_direct[col as usize] -= val;
            }
            for &b in &cols {
                let b = b as usize;
                let col = &tail_inv[b * e..b * e + e];
                let mut acc = 0.0;
                for &az in &cols {
                    acc += r_direct[az as usize] * col[az as usize];
                }
                h += acc * r_direct[b];
            }

            // The reducer under test.
            let mut y = [0.0; 2];
            let mut r = [0.0; 2];
            let mut sr = [0.0; 2];
            structured_row_reduce(
                fac, qc, &m_c, coup, e, &cols, &cross_col, &cross_val, &tail_inv, &mut y, &mut r,
                &mut sr,
            );

            let got: f64 = (0..qc).map(|c| y[c] * m_c[c]).sum::<f64>()
                + cols
                    .iter()
                    .map(|&b| sr[b as usize] * r[b as usize])
                    .sum::<f64>();
            assert!(
                (got - h).abs() < 1e-12,
                "reducer {got} vs pass-A leverage {h}"
            );
        }
    }
}
