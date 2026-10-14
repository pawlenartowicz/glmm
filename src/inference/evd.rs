//! Symmetric eigendecomposition and the PSD pseudo-inverse built on it.
//!
//! Both go through faer's low-level `linalg::evd::self_adjoint_evd` at
//! `Par::Seq`, with the scratch in a `MemBuffer` owned by the call. The `Mat`
//! convenience entry points (`Mat::self_adjoint_eigen`,
//! `Mat::self_adjoint_eigenvalues`) are not used: like `Mat::llt` (see
//! `linalg::SeqLlt`) they read `faer::get_global_parallelism()` on every call,
//! which on a native build is rayon's global pool whether or not this crate's
//! `parallel` feature is on. At `Par::Seq` the result is the same with and
//! without `parallel`, and two calls on one machine give the same bits.

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::{Col, Mat, MatMut, MatRef, Par};

/// Why [`sym_eigen`] or [`psd_pseudo_inverse`] returned no decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EvdError {
    /// The input has a different number of rows and columns.
    NotSquare,
    /// The input has a NaN or infinite entry.
    NonFinite,
    /// faer's QR iteration on the tridiagonal form did not converge.
    NoConvergence,
}

/// Eigenvalues and, when asked for, the eigenvectors of a symmetric matrix.
#[derive(Debug)]
pub(crate) struct SymEigen {
    /// The eigenvalues in ascending (nondecreasing) order, as faer returns
    /// them. Callers that need another order sort a copy.
    pub values: Vec<f64>,
    /// The unit eigenvectors as columns, column `k` belonging to `values[k]`;
    /// `None` when `sym_eigen` was called with `vectors = false`.
    pub vectors: Option<Mat<f64>>,
}

/// The pseudo-inverse of a symmetric matrix and how its spectrum was split.
/// `rank + dropped + negative` is the dimension.
#[derive(Debug)]
pub(crate) struct PseudoInverse {
    /// `Σ vᵢ vᵢᵀ / λᵢ` over the kept directions; zero when none is kept.
    pub inv: Mat<f64>,
    /// Kept directions: `λ > rel_tol · λ_max`.
    // Read only by the Wald χ², whose callers are in the formula frontend.
    #[cfg_attr(not(feature = "formula"), allow(dead_code))]
    pub rank: usize,
    /// Directions with `|λ| ≤ rel_tol · λ_max`, round-off negatives included.
    pub dropped: usize,
    /// Directions with `λ < −rel_tol · λ_max` (not kept either).
    pub negative: usize,
    /// The largest eigenvalue; `0.0` for an empty matrix.
    // Contract state that only tests read, by design.
    #[cfg_attr(not(test), allow(dead_code))]
    pub lambda_max: f64,
}

/// Eigenvalues (ascending) and optionally eigenvectors of the symmetric
/// matrix `a`. faer reads only the lower triangle; the caller passes a
/// symmetric matrix (checked to 1e-12 relative in debug builds).
///
/// Errors: [`EvdError::NotSquare`], [`EvdError::NonFinite`] for a NaN or
/// infinite entry, [`EvdError::NoConvergence`] from faer. Does not panic on a
/// symmetric input; an asymmetric one fails the symmetry check in debug builds.
pub(crate) fn sym_eigen(a: MatRef<'_, f64>, vectors: bool) -> Result<SymEigen, EvdError> {
    check_input(a)?;
    let n = a.nrows();
    let mut u = vectors.then(|| Mat::<f64>::zeros(n, n));
    let values = decompose(a, u.as_mut().map(|u| u.as_mut()))?;
    Ok(SymEigen { values, vectors: u })
}

/// The pseudo-inverse of the symmetric matrix `a` above a threshold, through
/// its eigendecomposition `A = Σ λᵢ vᵢ vᵢᵀ`:
///
/// `A_τ⁺ = Σ_{λᵢ > τ} vᵢ vᵢᵀ / λᵢ`, with `τ = rel_tol · λ_max`,
///
/// the Moore-Penrose inverse of the truncation `A_τ = Σ_{λᵢ > τ} λᵢ vᵢ vᵢᵀ`.
/// It equals `A⁺` only for a PSD `A` with no eigenvalue in `(0, τ]`: a
/// negative direction is left out, not inverted.
///
/// The spectrum splits three ways. `λ > τ` is kept (`rank`). `λ < −τ` is a
/// genuinely negative direction (`negative`): the matrix is not PSD, and a
/// caller may report it. Everything in between, `|λ| ≤ τ`, is `dropped`: a
/// null direction of a PSD matrix, including the tiny negatives round-off
/// gives such a direction, which must not be mistaken for a negative one. An
/// eigenvalue exactly at `±τ` is dropped. When `λ_max ≤ 0` nothing is kept:
/// `rank` is 0, `inv` is zero, `negative` counts every `λ < 0` and `dropped`
/// the rest.
///
/// `rel_tol` must be finite and `>= 0`, a caller precondition checked by a
/// `debug_assert!`: a negative one would keep negative directions and miscount
/// the split, a NaN one would keep nothing.
///
/// Errors and the symmetry check as for [`sym_eigen`].
pub(crate) fn psd_pseudo_inverse(
    a: MatRef<'_, f64>,
    rel_tol: f64,
) -> Result<PseudoInverse, EvdError> {
    debug_assert!(
        rel_tol.is_finite() && rel_tol >= 0.0,
        "psd_pseudo_inverse: rel_tol must be finite and >= 0, got {rel_tol}"
    );
    check_input(a)?;
    let n = a.nrows();
    let mut v = Mat::<f64>::zeros(n, n);
    let values = decompose(a, Some(v.as_mut()))?;
    let lambda_max = values.last().copied().unwrap_or(0.0);

    let mut inv = Mat::<f64>::zeros(n, n);
    if lambda_max <= 0.0 {
        let negative = values.iter().filter(|&&l| l < 0.0).count();
        return Ok(PseudoInverse {
            inv,
            rank: 0,
            dropped: n - negative,
            negative,
            lambda_max,
        });
    }

    let tau = rel_tol * lambda_max;
    let rank = values.iter().filter(|&&l| l > tau).count();
    let negative = values.iter().filter(|&&l| l < -tau).count();
    // Ascending order: the kept eigenvalues are the last `rank`.
    let kept = n - rank..n;
    for j in 0..n {
        for i in 0..n {
            let mut acc = 0.0;
            for k in kept.clone() {
                acc += v[(i, k)] * v[(j, k)] / values[k];
            }
            inv[(i, j)] = acc;
        }
    }
    Ok(PseudoInverse {
        inv,
        rank,
        dropped: n - rank - negative,
        negative,
        lambda_max,
    })
}

fn check_input(a: MatRef<'_, f64>) -> Result<(), EvdError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(EvdError::NotSquare);
    }
    let mut scale = 0.0f64;
    for j in 0..n {
        for i in 0..n {
            let x = a[(i, j)];
            if !x.is_finite() {
                return Err(EvdError::NonFinite);
            }
            scale = scale.max(x.abs());
        }
    }
    debug_assert!(
        (0..n).all(|j| (0..j).all(|i| (a[(i, j)] - a[(j, i)]).abs() <= 1e-12 * scale)),
        "evd: input is not symmetric to 1e-12 relative"
    );
    Ok(())
}

/// faer's self-adjoint EVD at `Par::Seq`, writing the eigenvectors into `u`
/// when given (`n × n`). Input already checked.
fn decompose(a: MatRef<'_, f64>, u: Option<MatMut<'_, f64>>) -> Result<Vec<f64>, EvdError> {
    let n = a.nrows();
    // `self_adjoint_evd_scratch` with eigenvectors computes `n - 1`, which
    // underflows at `n = 0`; an empty matrix has nothing to decompose.
    if n == 0 {
        return Ok(Vec::new());
    }
    let compute = if u.is_some() {
        ComputeEigenvectors::Yes
    } else {
        ComputeEigenvectors::No
    };
    let mut buf = MemBuffer::new(self_adjoint_evd_scratch::<f64>(
        n,
        compute,
        Par::Seq,
        Default::default(),
    ));
    let mut s = Col::<f64>::zeros(n);
    self_adjoint_evd(
        a,
        s.as_mut().as_diagonal_mut(),
        u,
        Par::Seq,
        MemStack::new(&mut buf),
        Default::default(),
    )
    .map_err(|e| match e {
        faer::linalg::evd::EvdError::NoConvergence => EvdError::NoConvergence,
    })?;
    Ok(s.iter().copied().collect())
}

#[cfg(test)]
#[path = "evd_tests.rs"]
mod tests;
