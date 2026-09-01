//! PIRLS module root: `BetaStep`, `refresh_eta_fixed`, `build_coupling_csr`, and the re-exports that keep `crate::glmm::pirls::*` paths stable across the `dense`/`blocked`/`blocked_extras` solve variants.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, LltRegularization};
use faer::linalg::cholesky::llt::solve::solve_in_place;
use faer::sparse::linalg::cholesky::LltRef;
use faer::{Conj, Mat, MatMut, MatRef, Par, Side, Spec};

use super::workspace::{
    glmm_block_chol, glmm_block_solve, glmm_block_solve_panel, StructuredSchur,
};
use super::{PIRLS_MAX_HALVINGS, PIRLS_MAX_ITERS};
use crate::scalar::Scalar;
use crate::sparse::logdet_llt;
use crate::spec::{BinomialLink, Family};

mod blocked;
mod blocked_extras;
mod dense;

pub(crate) use blocked::pirls_solve_blocked;
pub(crate) use blocked_extras::{pirls_solve_blocked_extras, structured_ainv_solve};
pub(crate) use dense::pirls_solve;

/// β handling for one PIRLS solve. `Fixed` = today's behavior verbatim (β is an
/// immutable input; the FD-Hessian path and BOBYQA stage 2 REQUIRE this so the
/// objective stays a function of the caller's β). `Profile` = PQL/stage-1 mode:
/// a δβ Schur-border update runs each iteration and the converged β is written
/// back through `beta`. No default — every call site chooses explicitly.
pub(crate) enum BetaStep<'a> {
    Fixed,
    Profile {
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
/// (`build_packed_m` drops θ=0 crossed groupings from `cross_col`/`n_cross`),
/// so it is fit-invariant only while the pinning mask is: the caller
/// (deviance.rs structured branch) caches it keyed on that mask and rebuilds
/// on transitions — not per eval, not blindly per fit.
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
