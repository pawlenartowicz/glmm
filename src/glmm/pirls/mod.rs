//! PIRLS module root: `BetaStep`, `refresh_eta_fixed`, `build_coupling_csr`, and the re-exports that keep `crate::glmm::pirls::*` paths stable across the `dense`/`blocked`/`blocked_extras` solve variants.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, LltRegularization};
use faer::linalg::cholesky::llt::solve::solve_in_place;
use faer::{Mat, MatMut, MatRef, Par, Spec};

use super::workspace::{glmm_block_chol, glmm_block_solve, StructuredSchur};
use super::{PIRLS_MAX_HALVINGS, PIRLS_MAX_ITERS};
use crate::scalar::Scalar;
use crate::spec::{BinomialLink, Family};

mod blocked;
mod blocked_extras;
mod dense;

pub(crate) use blocked::pirls_solve_blocked;
pub(crate) use blocked_extras::{
    pirls_solve_blocked_extras, structured_ainv_solve, structured_factor, TailKernel,
};
pub(crate) use dense::pirls_solve;

/// What one `laplace_deviance` call does with β. `Fixed` = β is the caller's input
/// (every SE, derivative and joint-BOBYQA eval). `ProfilePql` = the PQL border, stage 1.
/// `ProfileExact` = the exact-profile border with the log|A| correction and the Laplace
/// merit — the objective is then the exact Laplace β-profile at θ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BetaMode {
    Fixed,
    ProfilePql,
    ProfileExact,
}

/// β handling for one PIRLS solve. `Fixed` = the behavior verbatim (β is an
/// immutable input; the FD-Hessian path and BOBYQA stage 2 REQUIRE this so the
/// objective stays a function of the caller's β). `Profile` = PQL/stage-1 mode:
/// a δβ Schur-border update runs each iteration and the converged β is written
/// back through `beta`. No default — every call site chooses explicitly.
pub(crate) enum BetaStep<'a> {
    Fixed,
    Profile {
        // `Some` = exact Laplace profile (`BetaMode::ProfileExact`); `None` = the PQL
        // border, verbatim. Set only by `laplace_deviance`.
        exact: Option<&'a mut ExactProfileBufs>,
        xtwx: &'a mut Mat<f64>,      // p×p  C = X'WX          (ws.xtwx)
        xtwm: &'a mut Mat<f64>,      // p×k  B' = X'WM         (ws.xtwm)
        ainv_mtwx: &'a mut Mat<f64>, // k×p  T = A⁻¹B          (ws.ainv_mtwx)
        schur: &'a mut Mat<f64>,     // p×p  S_β               (ws.schur)
        beta_rhs: &'a mut [f64],     // len p: X'ρ, then rhs, then δβ in place (ws.beta_rhs)
        beta_prev: &'a mut [f64], // len p: last-accepted β for the halving backtrack (ws.beta_prev)
        // Persistent scratch for `schur`'s in-place `cholesky_in_place`, sized once
        // for p×p at workspace construction (ws.schur_llt_mem) — avoids the
        // `.llt(Side::Lower)` per-iteration heap allocation on this hot β-Schur step.
        schur_llt_mem: &'a mut MemBuffer,
    },
}

/// Length to allocate for an observed-information twin buffer: `len` where the
/// fit takes the observed step, nothing where it does not. Every read and write
/// of a twin — in [`DualStep`] and in [`ExactProfileBufs`] alike — sits behind
/// `dual.observed` or `!family::is_canonical(family)`, the same condition
/// `observed` carries here, so a canonical link never touches one. On a large
/// crossed shape the twins are the biggest buffers in the scratch
/// (`obs_coupling` is `q_core·s·e` elements, 45 `f64` each at
/// `HyperDual<8, 36>`), and sizing them off the fit keeps that allocation and
/// its first-touch page faults off every canonical fit. `obs_schur`, the
/// `StructuredSchur` twin, is gated on the same condition at its own build
/// sites.
pub(crate) fn obs_len(observed: bool, len: usize) -> usize {
    if observed {
        len
    } else {
        0
    }
}

/// Per-solve controls the dual-scalar derivative kernels (`derivative.rs`'s
/// `run_gradient`/`run_hessian`) hand `pirls_solve_blocked` and
/// `pirls_solve_blocked_extras`; `None` on every `f64` fit-path call, which is
/// then byte-identical to the pre-existing Fisher-only solve.
///
/// **Observed-information step (`observed`).** Each PIRLS step solves
/// `u_new = A_obs⁻¹((A_obs − I)u + g)` with `A_obs = M'W_obs M + I`, `W_obs`
/// the observed (Newton) weight `family::observed_weight`, while `log|A|`,
/// the returned factor and the convergence test stay on the Fisher `A` — the
/// objective and its fixed point `ũ` are unchanged, only the path the iterate
/// takes to it. Why: at the mode the lane fixed-point map
/// `du ← (I − A⁻¹H_uu)·du + b` contracts by `‖I − A⁻¹H_uu‖`, which is 0 for
/// `A_obs = H_uu` (lanes exact in one step, as on a canonical link) but only
/// 0.2–0.5 for the Fisher `A` on a non-canonical link, where the refinement
/// loop needed 6–10 kernel calls per gradient (measured 2026-09-02,
/// cbpp_probit / sim_gamma / sim_probit_large). Canonical links pass
/// `observed = false`: their Fisher `A` already IS `½h_uu`.
///
/// The blocked kernel packs the twin as `s` independent `q_p×q_p` blocks
/// (`obs_blocks`); the structured-extras kernel packs it as the same
/// core-block + crossed-tail Schur split the Fisher factor uses
/// (`obs_core_blocks`/`obs_coupling`/`obs_schur_blk`), because a crossed-tail
/// column couples every cluster and there is no per-cluster block to solve
/// alone. Both packings solve through the same `A_obs⁻¹` step. Every twin
/// below is allocated through [`obs_len`], so a canonical link carries none of
/// them at all.
///
/// **Step floor (`min_iters`).** The mixed-deviance exit fires after two
/// steps once the `f64` value sits at the mode, but the second-order lanes
/// need two steps to become exact and the objective must then be read at
/// that `u` — three steps in one solve, where the value test alone would
/// stop at two and force a second kernel call (two more steps) just to read
/// it. `0` leaves the exit rule untouched.
pub(crate) struct DualStep<T> {
    /// Take the observed-information step (non-canonical links).
    pub(crate) observed: bool,
    /// `s·q_p²` scratch for the observed blocked-path blocks, same layout as
    /// `a_blocks`; untouched when `observed` is false, and untouched on the
    /// structured-extras path, which packs its twin into the four buffers
    /// below instead.
    pub(crate) obs_blocks: Vec<T>,
    /// `(q_core² · s).max(1)` twin of the structured kernel's `core_blocks`,
    /// same per-cluster lower-triangle layout, scattered from `W_obs`.
    /// Untouched when `observed` is false and on the blocked path.
    pub(crate) obs_core_blocks: Vec<T>,
    /// `(q_core · s · e).max(1)` twin of `coupling`, `C_obs[f·q_core·e + local·e + b]`.
    /// Shares the `coup_cols`/`coup_ptr` CSR pattern with the Fisher coupling:
    /// the pattern is a function of the design and the θ-pin mask, not of `W`.
    pub(crate) obs_coupling: Vec<T>,
    /// `(e²).max(1)` twin of `schur_blk`, lower triangle.
    pub(crate) obs_schur_blk: Vec<T>,
    /// len `k_total`. The observed right-hand side in the `a_rhs` packing
    /// (`[f·q_core + local | k_family + b]`), then `u_obs = A_obs⁻¹ rhs` in place.
    pub(crate) obs_rhs: Vec<T>,
    /// len `n`. Per-row observed IRLS residual `w_obs,i·(Mu)ᵢ + W·working_residᵢ`.
    /// The structured kernel folds `(A − I)u` into its residual before the
    /// scatter, so the observed right-hand side cannot be recovered from
    /// `a_rhs` and needs its own residual, formed while `mu` still holds `(Mu)ᵢ`.
    pub(crate) obs_resid: Vec<T>,
    /// Do not exit before this many steps have run.
    pub(crate) min_iters: usize,
    /// Written by the solve: true iff every step of every block was an
    /// exact-Hessian step, so the caller may read the lanes after this one
    /// call. Two things take it back, and either is enough:
    ///
    /// - A row on one of the kernel's clamps, on any link, canonical
    ///   included — the step matrix is then not the Jacobian of the map the
    ///   iteration actually walks. See [`clamped_row_present`], which is the
    ///   test, for the derivation.
    /// - A non-PD observed factor on a non-canonical link (the observed weight
    ///   can go negative on an outlying row), where that step fell back to its
    ///   Fisher factor. On the structured-extras path a non-PD twin downgrades
    ///   the WHOLE iteration, not one block: the crossed Schur couples every
    ///   cluster, so there is no per-cluster fallback to take.
    ///
    /// `false` means the lanes have only contracted toward the answer, and the
    /// caller's refinement loop (`derivative.rs`'s `run_gradient` /
    /// `run_hessian`) re-enters until they stop moving.
    pub(crate) exact: bool,
}

/// Exact-profile scratch for the exact Laplace β-profile inside `pirls_solve_blocked`'s and
/// `pirls_solve_blocked_extras`'s Profile mode. `f64` throughout — the β border
/// is `f64`-only. Sized once per workspace, so the warm path allocates nothing.
pub(crate) struct ExactProfileBufs {
    /// len `k_total`. Holds `g_u = ∂log|A|/∂u` after the row pass, then
    /// `v = Ã⁻¹ g_u` in place after the block solve. On the structured-extras
    /// path both live in the `a_rhs` packing (`[f·q_core + local | k_family + b]`),
    /// NOT the RE-column order `u` uses.
    pub(crate) logdet_u: Vec<f64>,
    /// len p. `c_β = d log|A|/dβ` (direct part, then minus the û path).
    pub(crate) logdet_beta: Vec<f64>,
    /// `s·q²`, `a_blocks` layout. `A_obs = M'W_obs M + I` per cluster; only
    /// written on a non-canonical link (blocked path only).
    pub(crate) obs_blocks: Vec<f64>,
    /// `(q_core² · s).max(1)` twin of the structured kernel's `core_blocks`,
    /// same per-cluster lower-triangle layout, scattered from `W_obs`. Written
    /// only in exact mode on a non-canonical link, on the structured-extras
    /// path (`obs_blocks` above is the blocked-path twin).
    pub(crate) obs_core_blocks: Vec<f64>,
    /// `(q_core · s · e).max(1)` twin of the structured kernel's `coupling`,
    /// `C_obs[f·q_core·e + local·e + b]`. Shares the `coup_cols`/`coup_ptr`
    /// CSR pattern with the Fisher coupling: the pattern is a function of the
    /// design and the θ-pin mask, not of `W`.
    pub(crate) obs_coupling: Vec<f64>,
    /// `(e²).max(1)` twin of the structured kernel's `schur_blk`, lower
    /// triangle.
    pub(crate) obs_schur_blk: Vec<f64>,
    /// Cached sparse factor of the OBSERVED crossed Schur — a second
    /// `StructuredSchur` on the same symbolic pattern as the Fisher one, so the
    /// twin factor and the twin solve take the production sparse arm without
    /// overwriting the `axx` / `l_values` that the `tail_inv` columns, the β
    /// border and `se::structured_schur_fill` read off the Fisher factor.
    /// `None` on nested-only shapes (`e = 0`) and on the blocked path.
    pub(crate) obs_schur: Option<StructuredSchur>,
    /// len `k_total`. Last ACCEPTED `u` (RE-column order, as `u` itself) — the
    /// halving target once the accept decision moves after the block sweep
    /// (`u_prev` then holds the trial).
    pub(crate) u_acc: Vec<f64>,
    /// `e×e` column-major `S⁻¹` (`tail_inv[b·e + a] = (S⁻¹)_{a,b}`), the dense
    /// inverse of the structured path's crossed-tail Schur complement. Rebuilt
    /// every exact-mode structured iteration by `e` `TailKernel::tail_solve`
    /// calls on unit vectors; length 1 (unread) when `e == 0` and on the
    /// blocked path.
    pub(crate) tail_inv: Vec<f64>,
    /// len `e`. Per-row crossed residual `r_i = C_f'(A_f⁻¹ m_c) − m_x`, indexed
    /// by crossed column. Only cluster `f`'s coupling columns are ever written
    /// or read in one row, and each is written before it is read, so it needs no
    /// clearing between rows. A stack array cannot serve: `e` is a data
    /// dimension (181 on grouseticks), not a compile-time cap.
    pub(crate) tail_r: Vec<f64>,
    /// f64 mirror of THIS iterate's per-cluster factor, filled once per exact
    /// block and read by every pass below it: `s·q²` in `a_blocks` layout on the
    /// blocked path, `s·q_core²` in `core_blocks` layout on the structured one
    /// (sized for the wider, `q_core ≥ q`). Those buffers are generic `&[T]` (the
    /// derivative kernels instantiate `T = Dual<N>`) while exact mode is f64-only,
    /// and `block_leverage`/`glmm_block_solve` need a plain `&[f64]`; a transmute
    /// is not allowed. Filled per CLUSTER, not per row: `cluster_ids` is the
    /// caller's row order and is not contiguous by cluster, so a per-row copy
    /// cannot be hoisted by a last-seen-cluster check.
    pub(crate) fac_f64: Vec<f64>,
}

/// Does any of these rows sit on a clamp — Fisher weight on
/// `glm::WEIGHT_CLAMP`, or μ on one of `family::clamp_mu`'s bounds? The test
/// [`DualStep::exact`] is taken back by; same two conditions the assembled SE
/// engine refuses a fit on (`glmm::assembled::clamped_row_counts`), sharing
/// `family::clamp_mu_bounds` with it so the two can never drift apart.
///
/// Why a clamped row costs the one-step claim. Writing the PIRLS step as
/// `u ← u + A_obs⁻¹(g(u) − u)` with `g(u) = M'r(u)`, the lane fixed-point map
/// contracts by `‖I − A_obs⁻¹(I − ∂g/∂u)‖`, which is zero — lanes exact after
/// one step — exactly when `A_obs = I − ∂g/∂u = M'W_obs M + I` with the true
/// `W_obs = −∂r/∂η`. A clamp breaks that equality on the row it binds: the
/// floor pins `w` to a constant while `family::observed_weight` (and, on a
/// canonical link, the Fisher `W` that IS the step matrix) still reports the
/// unfloored curvature, and a pinned μ freezes the deviance's η-dependence the
/// same way. The contraction is then nonzero and the caller's refinement loop
/// has to run.
///
/// Branches on `.value()` only, so every lane count decides identically.
pub(crate) fn clamped_row_present<T: Scalar>(family: Family, w: &[T], prob: &[T]) -> bool {
    let (mu_lo, mu_hi) = crate::family::clamp_mu_bounds(family);
    w.iter().zip(prob).any(|(wi, mu)| {
        wi.value() <= crate::glm::WEIGHT_CLAMP || mu.value() <= mu_lo || mu.value() >= mu_hi
    })
}

/// `h = ‖L⁻¹ m‖²` for one row: the forward half of `glmm_block_solve` on the
/// row-major lower factor `l` (q×q), `m` the row's `q` RE-design entries.
pub(crate) fn block_leverage(l: &[f64], q: usize, m: &[f64]) -> f64 {
    let mut t = [0.0_f64; crate::consts::MAX_PRIMARY_Q];
    for r in 0..q {
        let mut v = m[r];
        for c in 0..r {
            v -= l[r * q + c] * t[c];
        }
        t[r] = v / l[r * q + r];
    }
    t[..q].iter().map(|x| x * x).sum()
}

/// Forward half of `glmm_block_solve` on the row-major lower factor `l` (q×q):
/// `t = L⁻¹m`, written into `t`. `block_leverage` is `‖t‖²` at `f64`; a
/// bilinear form `mᵢ'A⁻¹mⱼ` over the same block is `tᵢ·tⱼ`, which is why the
/// assembled gradient needs the vector and not only its norm.
pub(crate) fn block_forward_solve<T: Scalar>(l: &[T], q: usize, m: &[T], t: &mut [T]) {
    for r in 0..q {
        let mut v = m[r];
        for c in 0..r {
            v -= l[r * q + c] * t[c];
        }
        t[r] = v / l[r * q + r];
    }
}

/// Refill `eta_fixed[i] = offset[i] + Σ_j x[i,j]·β[j]` (the fixed-effect linear
/// predictor). Called once at entry of `pirls_solve` and, in `BetaStep::Profile`,
/// after every β update (the accepted δβ step and each β halving) — the trial
/// evaluation at the top of the loop reads `eta_fixed`, so it must track the
/// current β. `offset` is `FitOptions::offset` (`None` ⇒ this function is
/// byte-identical to the pre-offset version).
fn refresh_eta_fixed<T: crate::scalar::Scalar>(
    x: MatRef<f64>,
    beta: &[T],
    eta_fixed: &mut [T],
    n: usize,
    p: usize,
    offset: Option<&[f64]>,
) {
    for i in 0..n {
        let mut e = T::ZERO;
        for j in 0..p {
            e += T::from_f64(x[(i, j)]) * beta[j];
        }
        eta_fixed[i] = e;
    }
    if let Some(o) = offset {
        for i in 0..n {
            eta_fixed[i] += T::from_f64(o[i]);
        }
    }
}

/// Per-cluster crossed-column pattern (CSR over f): the union of cluster f's
/// rows' cross_col entries — exactly the nonzero-column support of C_f.
/// Counting-sort CSR: counts → prefix → fill (coup_ptr doubles as the write
/// cursors) → shift cursors back → per-cluster sort + dedup-compact (a
/// cluster's rows repeat the same crossed level; a duplicate in the list
/// would double-subtract in the Schur build). e = 0 (nested only) degenerates
/// to an all-empty CSR — n_cross is all zero.
///
/// The pattern is a function of the design AND the θ-pinning mask
/// (`build_packed_m` drops θ=0 crossed groupings from `cross_col`/`n_cross`,
/// but only at `T = f64`; a dual `T` keeps them, so the dual pattern can be
/// wider), so it is fit-invariant only while the pinning mask is: the caller
/// (deviance.rs structured branch) caches it keyed on that mask — which is
/// `f64`-only for the same reason — and rebuilds on transitions, not per
/// eval and not blindly per fit.
pub(crate) fn build_coupling_csr(
    cluster_ids: &[u32],
    cross_col: &[u32],
    n_cross: &[u8],
    s: usize,
    n: usize,
    coup_cols: &mut [u32],
    coup_ptr: &mut [u32],
) {
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    for v in coup_ptr[..s + 1].iter_mut() {
        *v = 0;
    }
    for i in 0..n {
        coup_ptr[cluster_ids[i] as usize + 1] += n_cross[i] as u32;
    }
    for f in 0..s {
        coup_ptr[f + 1] += coup_ptr[f];
    }
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        let cbase = i * g_cap;
        for z in 0..n_cross[i] as usize {
            coup_cols[coup_ptr[f] as usize] = cross_col[cbase + z];
            coup_ptr[f] += 1;
        }
    }
    for f in (1..=s).rev() {
        coup_ptr[f] = coup_ptr[f - 1];
    }
    coup_ptr[0] = 0;
    {
        let mut write = 0usize;
        let mut start = 0usize;
        for f in 0..s {
            let end = coup_ptr[f + 1] as usize;
            coup_cols[start..end].sort_unstable();
            coup_ptr[f] = write as u32;
            let mut prev = u32::MAX; // crossed indices are < e ≪ u32::MAX
            for idx in start..end {
                let v = coup_cols[idx];
                if v != prev {
                    coup_cols[write] = v;
                    write += 1;
                    prev = v;
                }
            }
            start = end;
        }
        coup_ptr[s] = write as u32;
    }
}
