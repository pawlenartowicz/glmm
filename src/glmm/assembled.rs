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
//! It delivers `G = 0` only while none of the kernel's clamps binds, though:
//! the weighted least squares PIRLS iterates uses the weight floored at
//! `glm::WEIGHT_CLAMP`, so on a floored row its fixed point solves a different
//! equation than the one differentiated here, and a μ held at
//! `family::clamp_mu`'s bound freezes the deviance's own η-dependence the same
//! way. Every entry point below censuses the mode state and refuses such a fit
//! — see [`clamped_row_counts`].
//!
//! **Evaluation point.** Every PIRLS variant leaves `dev` and `log|A|` built at
//! the assembly point `u_prev`, while the returned `pen = ‖u‖²` is read at the
//! next iterate `u` — the Laplace objective mixes iterates by construction,
//! and this module differentiates that mixed function, not a fictional
//! single-point one. Every η-dependent quantity (the residual, the Fisher and
//! observed weights, the per-row leverage, the factor) is read at `u_prev`;
//! only the `‖u‖²` penalty term is read at `u`. Reading everything at one
//! iterate or the other answers a nearby but different question and cannot
//! reach a tight gradient tolerance on a cell where the last PIRLS step is not
//! already at round-off.
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
};
use super::deviance::{blocked_laplace_deviance, structured_laplace_deviance};
use super::pirls::{
    block_forward_solve, structured_ainv_solve, structured_factor, BetaStep, TailKernel,
};
use super::workspace::{glmm_block_chol, glmm_block_solve, packed_m_theta_deriv, GlmmWorkspace};

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
/// `(e²).max(1)` — the same three shapes `DualStep`'s own twin carries.
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
        }
    }

    /// Every length in one place, from the same shape terms the rest of the
    /// dual scratch is sized from. Nothing here is allocated lazily: an
    /// assembly on a shape whose crossed tail is empty still holds the
    /// `.max(1)` minimum, so the first call on any shape allocates nothing.
    pub(crate) fn for_shape(
        m: usize,
        k: usize,
        rows: usize,
        s: usize,
        q_core: usize,
        e: usize,
    ) -> AssemblyBufs<T> {
        let kk = k.max(1);
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
        }
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
/// A does. Mirrors the blocked reducer (`block_forward_solve`) and the
/// sparse-Z reducer in `src/sparse/glmm.rs` — change together on the shared
/// block-inverse identity.
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
/// Leaves the mode in `ws.u` and the η-dependent state (`eta`, `prob`, `w`,
/// `mu`, the block factors, `beta_rhs`) at the solve's values, not the
/// caller's. Returns false when the solve did not converge or the objective is
/// not finite; the caller restores `ws.u` either way.
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
        lam,
        z_buf,
        m_buf,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        a_blocks,
        a_rhs,
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_val,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_mask,
        wx,
        offset: offset_field,
        ..
    } = ws;
    let offset = offset_field.as_deref();
    let g = &*groupings;
    let extras = !g.extra_offsets.is_empty();
    let mut mode_counters = crate::counters::EvalCounters::new();
    if extras {
        let (dev, conv, _raw_finite) = structured_laplace_deviance::<f64>(
            family,
            nb_theta,
            g,
            &prm[..m],
            z_buf,
            extra_ids,
            lam,
            cluster_ids,
            m_core_buf,
            cross_val,
            cross_col,
            n_cross,
            coup_cols,
            coup_ptr,
            coup_mask,
            x,
            y,
            &prior_w[..n],
            weighted,
            beta_rhs,
            BetaStep::Fixed,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            mu,
            core_blocks,
            coupling,
            schur_blk,
            None,
            false,
            a_rhs,
            None,
            wx,
            offset,
            Some(tol),
            n,
            &mut mode_counters,
        );
        conv && dev.is_finite()
    } else {
        let (dev, conv, _raw_finite) = blocked_laplace_deviance::<f64>(
            family,
            nb_theta,
            g,
            &prm[..m],
            beta_rhs,
            lam,
            z_buf,
            m_buf,
            x,
            y,
            &prior_w[..n],
            weighted,
            cluster_ids,
            eta,
            prob,
            w,
            u,
            u_prev,
            eta_fixed,
            a_blocks,
            a_rhs,
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

/// Routing gate shared by every entry point here: `None` on a shape with no
/// exact derivative and on an AGQ-routed shape, which is outside this
/// assembly's scope.
fn assembly_routes(ws: &GlmmWorkspace) -> bool {
    let extras = !ws.groupings.extra_offsets.is_empty();
    supports_shape(&ws.groupings)
        && (extras || !agq_eligible(ws.family, ws.nagq, ws.groupings.primary_q))
}

/// Rows where the kernel's own clamps bind at the mode state held in `w` and
/// `prob`: `(Fisher weight on `glm::WEIGHT_CLAMP`, μ on one of
/// `family::clamp_mu`'s bounds)`, counted over the `n` fitted rows.
///
/// Both entry points here refuse a fit with either count non-zero; the reason
/// is written at the census in [`joint_hessian_columns`].
pub(crate) fn clamped_row_counts(family: Family, w: &[f64], prob: &[f64]) -> (usize, usize) {
    let (mu_lo, mu_hi) = crate::family::clamp_mu_bounds(family);
    let mut w_clamped = 0usize;
    let mut mu_clamped = 0usize;
    for (&wi, &mu) in w.iter().zip(prob) {
        if wi <= crate::glm::WEIGHT_CLAMP {
            w_clamped += 1;
        }
        if mu <= mu_lo || mu >= mu_hi {
            mu_clamped += 1;
        }
    }
    (w_clamped, mu_clamped)
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
/// (the row pass's own per-cluster order); `u`/`u_prev` are in RE-column
/// order, `core_col(f, local)`, and the two coincide only at `np == 0`.
/// Mirrors `blocked_extras.rs`'s `gu_dot_du` comment on the same split —
/// change together. The crossed tail needs no remapping: both sides use
/// `k_family + b`. The matching point for `G = D_u + 2u` is `u_prev`, not the
/// returned `u`: `D_u` is formed from `rho`, which pass 1 reads off
/// `eta`/`prob` — the η-dependent state the mode solve leaves at `u_prev`, per
/// this module's own evaluation-point rule — so pairing it with `2u` would
/// read a residual at two different iterates and report the PIRLS step, not
/// the mode equation.
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
    if !assembly_routes(ws) {
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
        .pirls_tol_override
        .unwrap_or_else(|| super::pirls_tol_fd(family));
    let saved_u: Vec<f64> = ws.u[..kk].to_vec();
    if !mode_solve_f64(ws, x, y, cluster_ids, extra_ids, p, n, tol) {
        ws.u[..kk].copy_from_slice(&saved_u);
        return None;
    }
    // The clamped mode state `joint_hessian_columns` refuses, refused here for
    // the same reason and stated there.
    let (w_clamped, mu_clamped) = clamped_row_counts(family, &ws.w[..n], &ws.prob[..n]);
    if w_clamped > 0 || mu_clamped > 0 {
        ws.u[..kk].copy_from_slice(&saved_u);
        return None;
    }

    let GlmmWorkspace {
        groupings,
        params: prm,
        z_buf,
        m_buf,
        prior_w,
        eta,
        prob,
        w,
        u,
        u_prev,
        a_blocks,
        core_blocks,
        coupling,
        schur_blk,
        m_core_buf,
        cross_val,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
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
        u_prev,
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
                worst = worst.max((asm.d_u[f * qc + local] + 2.0 * u_prev[u_idx]).abs());
            }
        }
        for (&du, &up) in asm.d_u[k_family..k].iter().zip(&u_prev[k_family..k]) {
            worst = worst.max((du + 2.0 * up).abs());
        }
        *residual = worst;
    }
    u[..kk].copy_from_slice(&saved_u); // restore — leave ws.u as found
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
/// derivative, on an AGQ-routed shape, on a non-positive-definite observed
/// factor in the assembly's own adjoint solve, and on a kernel call that
/// reports `!exact`, where a non-PD observed factor inside the kernel makes
/// `û`'s returned lanes an approximation with no detector. The Fisher factor
/// is never silently put in the observed one's place. `NotConverged` is a real
/// failure at the accepted point.
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
    if !assembly_routes(ws) {
        return DerivStatus::Unsupported;
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
        sc.lanes() != nl || !sc.matches_shape(m, p, k, n, s, q_p, q_core, e, nagq, observed)
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
    scratch.mode_bufs_mut().saved_u[..kk].copy_from_slice(&ws.u[..kk]);
    let mode_ok = mode_solve_f64(ws, x, y, cluster_ids, extra_ids, p, n, tol);
    {
        let mode = scratch.mode_bufs_mut();
        if !mode_ok {
            ws.u[..kk].copy_from_slice(&mode.saved_u[..kk]);
            ws.dual_scratch = Some(scratch);
            return DerivStatus::NotConverged;
        }
        mode.u_mode[..k].copy_from_slice(&ws.u[..k]);
        ws.u[..kk].copy_from_slice(&mode.saved_u[..kk]); // restore — leave ws.u as found
    }

    // The census of the mode state, once, before any chunk. The adjoint here
    // differentiates the mode equation `G = D_u + 2u = 0`, and the floored
    // weighted least squares PIRLS actually iterates does not satisfy it on a
    // row whose Fisher weight sits on `glm::WEIGHT_CLAMP`; a μ held at its own
    // clamp breaks the same premise from the deviance side. This engine is
    // therefore exact only on a clean census, and a fit with either clamp
    // binding goes to the pass that differentiates the map PIRLS has.
    let (w_clamped, mu_clamped) = clamped_row_counts(family, &ws.w[..n], &ws.prob[..n]);
    if w_clamped > 0 || mu_clamped > 0 {
        ws.dual_scratch = Some(scratch);
        return DerivStatus::Unsupported;
    }

    let GlmmWorkspace {
        groupings,
        params: prm,
        z_buf,
        prior_w,
        cross_col,
        n_cross,
        coup_cols,
        coup_ptr,
        coup_mask,
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
                cross_col,
                n_cross,
                coup_cols,
                coup_ptr,
                coup_mask,
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
/// match arms hand a typed buffer set to. One kernel call per chunk and no
/// refinement loop: a call that comes back `!exact` is a refusal, because the
/// assembly's adjoint equation needs the observed factor the kernel would then
/// not have used.
///
/// `min_iters = 3`, the same floor the packed second-order pass takes. The
/// exit test the kernel runs without it inspects the objective VALUE only, and
/// the value sits at the mode from the first step, while `u`'s lanes are the
/// fixed point of a contraction that test never looks at. Three steps is what
/// makes the first-order lanes exact and then reads the objective at an
/// iterate that carries them — and it is also what puts this pass on the same
/// iterate of the same solve as the packed pass and as `laplace_gradient`, so
/// all three differentiate the same mix of `u_prev` (where `dev` and `log|A|`
/// are built) and `u` (where the `‖u‖²` penalty is read). It costs one extra
/// PIRLS step per chunk. `GlmmDualScratch::exit_mode_step` reports the
/// distance each pass exits at.
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
    cross_col: &mut [u32],
    n_cross: &mut [u8],
    coup_cols: &mut [u32],
    coup_ptr: &mut [u32],
    coup_mask: &mut Option<u32>,
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
            bufs.u[c] = T::from_f64(u_mode[c]);
        }
        // The observed-information step: exact lanes in one kernel call on a
        // non-canonical link, where the Fisher `A` is only an approximation to
        // `½h_uu`. Canonical links pass `observed = false` — their `A` already
        // is that Hessian.
        bufs.dual.observed = !canonical;
        bufs.dual.min_iters = 3;
        // Throwaway: a derivative evaluation is not a fit-path PIRLS solve, so
        // it must not reach `ws.counters`.
        let mut counters = crate::counters::EvalCounters::new();
        let (value, conv) = if extras {
            let (obj, conv, _raw_finite) = structured_laplace_deviance::<T>(
                family,
                nb_theta,
                g,
                &bufs.params[..m],
                z_buf,
                extra_ids,
                &mut bufs.lam,
                cluster_ids,
                &mut bufs.m_core_buf,
                &mut bufs.cross_val,
                cross_col,
                n_cross,
                coup_cols,
                coup_ptr,
                coup_mask,
                x,
                y,
                prior_w,
                weighted,
                &mut bufs.beta[..p],
                BetaStep::Fixed,
                &mut bufs.eta,
                &mut bufs.prob,
                &mut bufs.w,
                &mut bufs.u,
                &mut bufs.u_prev,
                &mut bufs.eta_fixed,
                &mut bufs.mu,
                &mut bufs.core_blocks,
                &mut bufs.coupling,
                &mut bufs.schur_blk,
                // No sparse Schur at a dual `T`: the cached LLT is `f64`-only,
                // so the tail takes `tail_factor`/`tail_solve`'s dense default
                // body.
                None,
                false,
                &mut bufs.a_rhs,
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
                &mut bufs.lam,
                z_buf,
                &mut bufs.m_buf,
                x,
                y,
                prior_w,
                weighted,
                cluster_ids,
                &mut bufs.eta,
                &mut bufs.prob,
                &mut bufs.w,
                &mut bufs.u,
                &mut bufs.u_prev,
                &mut bufs.eta_fixed,
                &mut bufs.a_blocks,
                &mut bufs.a_rhs,
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
        if !bufs.dual.exact {
            return DerivStatus::Unsupported;
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
            &bufs.eta,
            &bufs.prob,
            &bufs.w,
            &bufs.u,
            &bufs.u_prev,
            if extras {
                &bufs.m_core_buf
            } else {
                &bufs.m_buf
            },
            if extras {
                &bufs.core_blocks
            } else {
                &bufs.a_blocks
            },
            &bufs.coupling,
            &bufs.schur_blk,
            &bufs.cross_val,
            cross_col,
            n_cross,
            coup_cols,
            coup_ptr,
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
            for j in 0..width {
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
    u: &[T],
    // The iterate the kernel's η-dependent state was built at (`u` itself is
    // the step it takes from there). Every η-direction read goes through this
    // one; only the `‖u‖²` penalty term reads `u`.
    u_prev: &[T],
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
    let mut t = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];
    let mut yb = [T::ZERO; crate::lmm::MAX_PRIMARY_Q];

    // --- pass 1, per row:
    //   ρᵢ = prior_wᵢ·(dμ/dη)(yᵢ−μᵢ)/V,  ∂(prior_wᵢ·devᵢ)/∂ηᵢ = −2ρᵢ,
    //   ∂²/∂η² = 2·w_obs,ᵢ,  hᵢ = mᵢ'A⁻¹mᵢ
    //   D_βj  += −2ρᵢ·xᵢⱼ                  D_uc  += −2ρᵢ·M_ic
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
        lev[i] = h;
        w_eta[i] = crate::family::weight_eta_deriv(family, nb_theta, eta[i], prob[i], w[i]);
        // WHICH ROWS SEE THE WEIGHT FLOOR, AND WHY THE TWO USES SPLIT.
        //
        // The kernel stores `w` already floored at `glm::WEIGHT_CLAMP`
        // (`Scalar::eval_rows`), and `A = M'WM + I` is built from that floored
        // `w`. So every quantity that is a derivative OF `log|A|` — the
        // per-row leverage `h`, `ℓ_β`, `ℓ_u`, `ℓ_θ`, and `w'`'s zero-fill on a
        // floored row — is right with the floored `w`: the floor is part of the
        // matrix whose determinant the objective takes, and `max(w, c)` is flat
        // where it binds, so the objective really does stop depending on such a
        // row through that term.
        //
        // `w_obs` is not that. It is `½·∂²dev/∂η²`, and the deviance never sees
        // the floor, so the observed weight is formed from the UNFLOORED Fisher
        // weight, recomputed here from the same `eta` the kernel left. That
        // matters because this route uses `A_obs = M'W_obs M + I` as an
        // explicit `G_u` in the adjoint solve `G_u·adj = F_u`, where a wrong
        // `A_obs` lands straight in the answer — unlike an iteration matrix,
        // whose fixed point is the same whatever matrix drove it there.
        //
        // Only on a NON-canonical link: there the assembly builds `A_obs` from
        // this same `w_obs`, so `G_u` and `G_γ` see one weight. On a canonical
        // link `A_obs` IS the Fisher `A` the mode solve already factored and
        // this route reuses that factor, so `G_γ` must keep the floored `w`
        // the factor was built from or the pair disagrees.
        let w_for_obs = if canonical {
            w[i]
        } else {
            let (_, w_raw, _) =
                crate::family::irls_weight_and_resid(family, nb_theta, y[i], eta[i]);
            T::from_f64(prior_w[i]) * w_raw
        };
        w_obs[i] = crate::family::observed_weight(
            family, nb_theta, y[i], prior_w[i], eta[i], prob[i], w_for_obs,
        );
        let dr = T::from_f64(-2.0) * r;
        let a = w_eta[i] * h;
        let two_wo = T::from_f64(2.0) * w_obs[i];
        for j in 0..p {
            let xij = T::from_f64(x[(i, j)]);
            d_gamma[n_theta + j] += dr * xij;
            l_gamma[n_theta + j] += a * xij;
        }
        for (local, &mcl) in mc.iter().enumerate().take(qc) {
            let c = f * qc + local;
            d_u[c] += dr * mcl;
            l_u[c] += a * mcl;
            let gb = two_wo * mcl;
            for j in 0..p {
                g_gamma[(n_theta + j) * k + c] += gb * T::from_f64(x[(i, j)]);
            }
        }
        for z in 0..ncz {
            let c = k_family + cross_col[cbase + z] as usize;
            let v = cross_val[cbase + z];
            d_u[c] += dr * v;
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
                deta += mad[local] * u_prev[core_col(f, local)];
            }
            for z in 0..ncz {
                deta += cvd[z] * u_prev[k_family + cross_col[cbase + z] as usize];
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
            d_gamma[a] += dr * deta;
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
    // `A_obs = M'W_obs M + I`. On a canonical link it IS the Fisher `A` the
    // mode solve already factored; otherwise it is built and factored here,
    // and a non-PD factor is the refusal. Built here rather than taken from
    // the dual kernel's own observed twin because a `BetaStep::Fixed` `f64`
    // call leaves no twin at all, so this arm has to exist for the `f64`
    // instantiation regardless — one body serves both.
    if canonical {
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
