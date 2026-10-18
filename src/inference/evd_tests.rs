//! `sym_eigen` and `psd_pseudo_inverse` against closed forms: the Toeplitz
//! spectrum, a Householder conjugate with a known eigenbasis, and the four
//! Moore-Penrose conditions. No second solver is used as a reference.

use super::{psd_pseudo_inverse, sym_eigen, EvdError};
use faer::{Mat, MatRef};

fn diag(d: &[f64]) -> Mat<f64> {
    Mat::from_fn(d.len(), d.len(), |i, j| if i == j { d[i] } else { 0.0 })
}

/// The Householder reflector `I − 2 w wᵀ / (wᵀ w)`: symmetric and orthogonal.
fn householder(w: &[f64]) -> Mat<f64> {
    let ww: f64 = w.iter().map(|x| x * x).sum();
    Mat::from_fn(w.len(), w.len(), |i, j| {
        let id = if i == j { 1.0 } else { 0.0 };
        id - 2.0 * w[i] * w[j] / ww
    })
}

/// `tridiag(−1, 2, −1)` of size `n`.
fn toeplitz(n: usize) -> Mat<f64> {
    Mat::from_fn(n, n, |i, j| match i.abs_diff(j) {
        0 => 2.0,
        1 => -1.0,
        _ => 0.0,
    })
}

fn max_abs(a: MatRef<'_, f64>) -> f64 {
    let mut m = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max(a[(i, j)].abs());
        }
    }
    m
}

fn max_abs_diff(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> f64 {
    assert_eq!((a.nrows(), a.ncols()), (b.nrows(), b.ncols()));
    let mut m = 0.0f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            m = m.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    m
}

#[test]
fn toeplitz_spectrum_matches_closed_form() {
    for n in [1usize, 2, 7, 50] {
        let e = sym_eigen(toeplitz(n).as_ref(), false).unwrap();
        assert!(e.vectors.is_none());
        assert_eq!(e.values.len(), n);
        for (idx, &got) in e.values.iter().enumerate() {
            let k = (idx + 1) as f64;
            let want = 2.0 - 2.0 * (k * std::f64::consts::PI / (n as f64 + 1.0)).cos();
            assert!(
                (got - want).abs() <= 1e-13,
                "n = {n}, k = {k}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn householder_conjugate_gives_its_basis_and_reconstructs() {
    // A = Q D Qᵀ with Q symmetric orthogonal, so the eigenvector of d[k] is
    // column k of Q, up to sign.
    let q = householder(&[1.0, -2.0, 0.5, 3.0, 1.5]);
    let d = [2.0, -3.0, 7.0, 0.5, 1.0];
    let a = &q * diag(&d) * q.transpose();
    let e = sym_eigen(a.as_ref(), true).unwrap();
    let v = e.vectors.as_ref().unwrap();

    let mut order: Vec<usize> = (0..d.len()).collect();
    order.sort_by(|&i, &j| d[i].total_cmp(&d[j]));
    for (k, &col) in order.iter().enumerate() {
        assert!((e.values[k] - d[col]).abs() <= 1e-12, "{:?}", e.values);
        let dot: f64 = (0..d.len()).map(|i| v[(i, k)] * q[(i, col)]).sum();
        let sign = dot.signum();
        for i in 0..d.len() {
            assert!(
                (v[(i, k)] - sign * q[(i, col)]).abs() <= 1e-12,
                "vector {k}, entry {i}"
            );
        }
    }

    let rebuilt = v * diag(&e.values) * v.transpose();
    assert!(max_abs_diff(rebuilt.as_ref(), a.as_ref()) <= 1e-12 * max_abs(a.as_ref()));
}

#[test]
fn rank_two_psd_satisfies_moore_penrose() {
    // A = 3 u₁u₁ᵀ + u₂u₂ᵀ with u₁, u₂ orthonormal: rank 2, two zero directions.
    let q = householder(&[1.0, 2.0, 3.0, 4.0]);
    let a = Mat::from_fn(4, 4, |i, j| {
        3.0 * q[(i, 0)] * q[(j, 0)] + q[(i, 1)] * q[(j, 1)]
    });
    let p = psd_pseudo_inverse(a.as_ref(), 1e-8).unwrap();
    assert_eq!((p.rank, p.dropped, p.negative), (2, 2, 0));
    assert!((p.lambda_max - 3.0).abs() <= 1e-12);

    let x = &p.inv;
    let ax = &a * x;
    let xa = x * &a;
    assert!(max_abs_diff((&ax * &a).as_ref(), a.as_ref()) <= 1e-12);
    assert!(max_abs_diff((&xa * x).as_ref(), x.as_ref()) <= 1e-12);
    assert!(max_abs_diff(ax.transpose(), ax.as_ref()) <= 1e-12);
    assert!(max_abs_diff(xa.transpose(), xa.as_ref()) <= 1e-12);
}

#[test]
fn clear_negative_eigenvalue_is_counted_negative() {
    let p = psd_pseudo_inverse(diag(&[4.0, 1.0, -2.0]).as_ref(), 1e-8).unwrap();
    assert_eq!((p.rank, p.dropped, p.negative), (2, 0, 1));
    // The negative direction is not inverted: no −1/2 in `inv`.
    assert!(max_abs_diff(p.inv.as_ref(), diag(&[0.25, 1.0, 0.0]).as_ref()) <= 1e-15);
    assert_eq!(p.inv[(2, 2)], 0.0);
}

#[test]
fn round_off_negative_is_counted_dropped() {
    let p = psd_pseudo_inverse(diag(&[2.0, 1.0, -1e-17 * 2.0]).as_ref(), 1e-8).unwrap();
    assert_eq!((p.rank, p.dropped, p.negative), (2, 1, 0));
    // The dropped direction is not inverted: no −5e16 in `inv`.
    assert!(max_abs_diff(p.inv.as_ref(), diag(&[0.5, 1.0, 0.0]).as_ref()) <= 1e-15);
    assert_eq!(p.inv[(2, 2)], 0.0);
}

#[test]
fn eigenvalue_exactly_at_threshold_is_dropped() {
    // rel_tol · λ_max = 0.25 · 4 = 1 exactly: both +1 and −1 sit on the
    // boundary and neither is kept nor counted negative.
    let p = psd_pseudo_inverse(diag(&[4.0, 1.0, -1.0]).as_ref(), 0.25).unwrap();
    assert_eq!((p.rank, p.dropped, p.negative), (1, 2, 0));
    assert!(max_abs_diff(p.inv.as_ref(), diag(&[0.25, 0.0, 0.0]).as_ref()) <= 1e-15);
    assert_eq!((p.inv[(1, 1)], p.inv[(2, 2)]), (0.0, 0.0));
}

#[test]
fn nonpositive_lambda_max_gives_rank_zero_and_zero_inverse() {
    let p = psd_pseudo_inverse(diag(&[-1.0, -3.0, 0.0]).as_ref(), 1e-8).unwrap();
    assert_eq!((p.rank, p.dropped, p.negative), (0, 1, 2));
    assert_eq!(p.lambda_max, 0.0);
    assert_eq!((p.inv.nrows(), p.inv.ncols()), (3, 3));
    assert_eq!(max_abs(p.inv.as_ref()), 0.0);
}

#[test]
fn empty_matrix_is_ok_and_empty() {
    let a = Mat::<f64>::zeros(0, 0);
    let e = sym_eigen(a.as_ref(), true).unwrap();
    assert!(e.values.is_empty());
    let v = e.vectors.unwrap();
    assert_eq!((v.nrows(), v.ncols()), (0, 0));

    let p = psd_pseudo_inverse(a.as_ref(), 1e-8).unwrap();
    assert_eq!((p.inv.nrows(), p.inv.ncols()), (0, 0));
    assert_eq!((p.rank, p.dropped, p.negative), (0, 0, 0));
    assert_eq!(p.lambda_max, 0.0);
}

#[test]
fn one_by_one() {
    let a = Mat::from_fn(1, 1, |_, _| 5.0);
    let e = sym_eigen(a.as_ref(), true).unwrap();
    assert_eq!(e.values, vec![5.0]);
    assert_eq!(e.vectors.unwrap()[(0, 0)].abs(), 1.0);

    let p = psd_pseudo_inverse(a.as_ref(), 1e-8).unwrap();
    assert_eq!((p.rank, p.dropped, p.negative), (1, 0, 0));
    assert_eq!(p.lambda_max, 5.0);
    assert!((p.inv[(0, 0)] - 0.2).abs() <= 1e-15);
}

#[test]
fn non_square_is_an_error() {
    let a = Mat::<f64>::zeros(2, 3);
    assert_eq!(
        sym_eigen(a.as_ref(), false).unwrap_err(),
        EvdError::NotSquare
    );
    assert_eq!(
        psd_pseudo_inverse(a.as_ref(), 1e-8).unwrap_err(),
        EvdError::NotSquare
    );
}

#[test]
fn nan_is_an_error() {
    let mut a = toeplitz(3);
    a[(1, 1)] = f64::NAN;
    assert_eq!(
        sym_eigen(a.as_ref(), true).unwrap_err(),
        EvdError::NonFinite
    );
    assert_eq!(
        psd_pseudo_inverse(a.as_ref(), 1e-8).unwrap_err(),
        EvdError::NonFinite
    );
}

#[test]
fn two_calls_are_bit_identical() {
    let q = householder(&[0.3, -1.0, 2.0, 0.7, -0.4, 1.1, 0.9]);
    let a = &q * diag(&[5.0, 1e-3, 0.0, 2.5, 4.0, 1e-12, 3.0]) * q.transpose();
    let bits = |m: MatRef<'_, f64>| -> Vec<u64> {
        (0..m.ncols())
            .flat_map(|j| (0..m.nrows()).map(move |i| m[(i, j)].to_bits()))
            .collect()
    };

    let e1 = sym_eigen(a.as_ref(), true).unwrap();
    let e2 = sym_eigen(a.as_ref(), true).unwrap();
    let vbits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(vbits(&e1.values), vbits(&e2.values));
    assert_eq!(
        bits(e1.vectors.as_ref().unwrap().as_ref()),
        bits(e2.vectors.as_ref().unwrap().as_ref())
    );

    let p1 = psd_pseudo_inverse(a.as_ref(), 1e-8).unwrap();
    let p2 = psd_pseudo_inverse(a.as_ref(), 1e-8).unwrap();
    assert_eq!(bits(p1.inv.as_ref()), bits(p2.inv.as_ref()));
    assert_eq!(
        (p1.rank, p1.dropped, p1.negative, p1.lambda_max.to_bits()),
        (p2.rank, p2.dropped, p2.negative, p2.lambda_max.to_bits())
    );
}
