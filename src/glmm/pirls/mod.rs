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
pub(crate) use blocked_extras::{pirls_solve_blocked_extras, structured_ainv_solve, TailKernel};
pub(crate) use dense::pirls_solve;

/// What one `laplace_deviance` call does with β. `Fixed` = β is the caller's input
/// (every SE, derivative and joint-BOBYQA eval). `ProfilePql` = the PQL border, today's
/// stage 1. `ProfileExact` = the P1 border with the log|A| correction and the Laplace
/// merit — the objective is then the exact Laplace β-profile at θ.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BetaMode {
    Fixed,
    ProfilePql,
    ProfileExact,
}

/// β handling for one PIRLS solve. `Fixed` = today's behavior verbatim (β is an
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

/// Per-solve controls the dual-scalar derivative kernels (`derivative.rs`'s
/// `run_gradient`/`run_hessian`) hand `pirls_solve_blocked`; `None` on every
/// `f64` fit-path call, which is then byte-identical to the pre-existing
/// Fisher-only solve.
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
/// **Step floor (`min_iters`).** The mixed-deviance exit fires after two
/// steps once the `f64` value sits at the mode, but the second-order lanes
/// need two steps to become exact and the objective must then be read at
/// that `u` — three steps in one solve, where the value test alone would
/// stop at two and force a second kernel call (two more steps) just to read
/// it. `0` leaves the exit rule untouched.
pub(crate) struct DualStep<T> {
    /// Take the observed-information step (non-canonical links).
    pub(crate) observed: bool,
    /// `s·q_p²` scratch for the observed blocks, same layout as `a_blocks`;
    /// untouched when `observed` is false.
    pub(crate) obs_blocks: Vec<T>,
    /// Do not exit before this many steps have run.
    pub(crate) min_iters: usize,
    /// Written by the solve: true iff every step of every block was an
    /// exact-Hessian step — always on a canonical link, and on a
    /// non-canonical one unless some observed block was not PD (the observed
    /// weight can go negative on an outlying row), in which case that step
    /// fell back to its Fisher block and the caller's refinement loop runs as
    /// before.
    pub(crate) exact: bool,
}

/// P1 scratch for the exact Laplace β-profile inside `pirls_solve_blocked`'s and
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
