//! The contest layer: lmerTest's `contest1D`, `contestMD` and
//! `get_Fstat_ddf` (lmerTest 3.2.1), ported line for line.
//!
//! Every function takes lmerTest's own inputs, [`ContestInputs`]: the
//! fixed-effect estimates `β`, their covariance `C`, the asymptotic covariance
//! `A` of the variance parameters, and the Jacobian `Jac` of `C` with respect
//! to those parameters, one `p × p` matrix per parameter. The Satterthwaite
//! df of a contrast `l` is `2v² / (gᵀAg)`, with `v = lᵀCl` and
//! `g_k = lᵀ Jac_k l` (Satterthwaite, 1946; Kuznetsova, Brockhoff and
//! Christensen, 2017).

use super::evd::{sym_eigen, EvdError};
use super::tails::{f_upper_p, t_two_sided_p};
use faer::{Mat, MatRef};

/// lmerTest's Satterthwaite inputs: `β` (`p`), `C` (`p × p`, lmerTest's
/// `vcov(model)`), `A` (`k × k`, `vcov_varpar`) and `Jac` (`k` matrices of
/// `p × p`, `Jac_list`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ContestInputs<'a> {
    /// The fixed-effect estimates.
    pub beta: &'a [f64],
    /// Their covariance `C`.
    pub vcov_beta: MatRef<'a, f64>,
    /// The covariance `A` of the variance parameters.
    pub vcov_varpar: MatRef<'a, f64>,
    /// `∂C/∂φ_k`, one matrix per variance parameter `φ_k`.
    pub jac: &'a [Mat<f64>],
}

/// One contrast's t test (`contest1D`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Contest1D {
    /// `lᵀβ`.
    pub estimate: f64,
    /// `√(lᵀCl)`.
    pub std_error: f64,
    /// The Satterthwaite df, `2v² / (gᵀAg)`.
    pub df: f64,
    /// `lᵀβ / √(lᵀCl)`.
    pub t_value: f64,
    /// Two-sided p of `t_value` on `df`.
    pub p_value: f64,
}

/// A contrast matrix's F test (`contestMD`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ContestMD {
    /// The F statistic.
    pub f_value: f64,
    /// The rank `q` of `L C Lᵀ` under lmerTest's rule; `0` when nothing is
    /// retained (F, DenDF and p are then NaN).
    pub num_df: usize,
    /// The denominator df, [`fstat_ddf`] of [`ContestMD::nu`].
    pub den_df: f64,
    /// Upper-tail p of `f_value` on `(num_df, den_df)`.
    pub p_value: f64,
    /// The per-direction df `ν_m`, one per retained eigen-direction, largest
    /// eigenvalue first; on the one-direction route the contest1D df.
    pub nu: Vec<f64>,
    /// The rows of `L`. `num_df < rows` is a rank-deficient `L`, which the
    /// caller reports.
    pub rows: usize,
}

/// `contest1D`: the t test of `lᵀβ = 0`. `v = lᵀCl`, `g_k = lᵀ Jac_k l`,
/// `df = 2v² / (gᵀAg)`, `t = lᵀβ / √v`. A zero `v` gives df 0 (NaN when
/// `gᵀAg = 0` too), t `±∞` (NaN when `lᵀβ = 0`), and NaN p.
pub(crate) fn contest_1d(inp: &ContestInputs<'_>, l: &[f64]) -> Contest1D {
    debug_assert_eq!(
        l.len(),
        inp.beta.len(),
        "contest_1d: l and beta lengths differ"
    );
    let estimate: f64 = l.iter().zip(inp.beta).map(|(a, b)| a * b).sum();
    let var_con = qform(l, inp.vcov_beta);
    let grad: Vec<f64> = inp.jac.iter().map(|j| qform(l, j.as_ref())).collect();
    let df = 2.0 * var_con * var_con / qform(&grad, inp.vcov_varpar);
    let std_error = var_con.sqrt();
    let t_value = estimate / std_error;
    Contest1D {
        estimate,
        std_error,
        df,
        t_value,
        p_value: t_two_sided_p(t_value, df),
    }
}

/// `contestMD`: the F test of `Lβ = 0`, `L` an `r × p` matrix. One row is
/// [`contest_1d`] with `F = t²`. Otherwise `L C Lᵀ = P D Pᵀ` (eigenvalues
/// descending); the `q` directions with `d_m > max(√ε·d₁, 0)` are kept and
/// `PtL = (PᵀL)[1:q]`. `q = 1` is [`contest_1d`] on that row; otherwise
/// `F = Σ_m (PtL_m β)² / d_m / q`, `ν_m = 2d_m² / (g_mᵀAg_m)` with
/// `g_{m,k} = PtL_mᵀ Jac_k PtL_m`, and the DenDF is [`fstat_ddf`]`(ν, 1e-8)`.
///
/// Errors: the eigendecomposition's, for a non-finite `L C Lᵀ`.
pub(crate) fn contest_md(
    inp: &ContestInputs<'_>,
    l: MatRef<'_, f64>,
) -> Result<ContestMD, EvdError> {
    let rows = l.nrows();
    let p = inp.beta.len();
    debug_assert_eq!(l.ncols(), p, "contest_md: L has the wrong column count");
    let row = |i: usize| -> Vec<f64> { (0..p).map(|j| l[(i, j)]).collect() };
    if rows == 1 {
        return Ok(from_1d(contest_1d(inp, &row(0)), 1));
    }

    // L C Lᵀ, symmetrized: the two triangles round differently. Plain loops,
    // not faer's product operator, which reads the global parallelism.
    let l_rows: Vec<Vec<f64>> = (0..rows).map(row).collect();
    let lcl = Mat::from_fn(rows, rows, |i, j| {
        bilinear(&l_rows[i], inp.vcov_beta, &l_rows[j])
    });
    let vl = Mat::from_fn(rows, rows, |i, j| 0.5 * (lcl[(i, j)] + lcl[(j, i)]));
    let eig = sym_eigen(vl.as_ref(), true)?;
    let vectors = eig.vectors.expect("sym_eigen was asked for vectors");
    // lmerTest's order: eigenvalues descending (faer returns them ascending).
    let order: Vec<usize> = (0..rows).rev().collect();
    let d1 = eig.values[order[0]];
    let tol = (f64::EPSILON.sqrt() * d1).max(0.0);
    let kept: Vec<usize> = order.into_iter().filter(|&k| eig.values[k] > tol).collect();
    let q = kept.len();
    if q == 0 {
        return Ok(ContestMD {
            f_value: f64::NAN,
            num_df: 0,
            den_df: f64::NAN,
            p_value: f64::NAN,
            nu: Vec::new(),
            rows,
        });
    }
    // Row m of PtL is eigenvector kept[m] times L.
    let ptl: Vec<Vec<f64>> = kept
        .iter()
        .map(|&k| {
            (0..p)
                .map(|j| (0..rows).map(|i| vectors[(i, k)] * l[(i, j)]).sum())
                .collect()
        })
        .collect();
    if q == 1 {
        return Ok(from_1d(contest_1d(inp, &ptl[0]), rows));
    }

    let mut t2 = 0.0;
    let mut nu = Vec::with_capacity(q);
    for (m, &k) in kept.iter().enumerate() {
        let d = eig.values[k];
        let est: f64 = ptl[m].iter().zip(inp.beta).map(|(a, b)| a * b).sum();
        t2 += est * est / d;
        let grad: Vec<f64> = inp.jac.iter().map(|j| qform(&ptl[m], j.as_ref())).collect();
        nu.push(2.0 * d * d / qform(&grad, inp.vcov_varpar));
    }
    let f_value = t2 / q as f64;
    let den_df = fstat_ddf(&nu, 1e-8);
    Ok(ContestMD {
        f_value,
        num_df: q,
        den_df,
        p_value: f_upper_p(f_value, q as f64, den_df),
        nu,
        rows,
    })
}

/// `get_Fstat_ddf`: the denominator df of an F built from `q = nu.len()`
/// squared t statistics with df `nu`, in lmerTest's case order: one ν gives
/// ν; every `|ν_{m+1} − ν_m| < tol` gives `mean(ν)` (checked before the
/// next rule, so equal ν ≤ 2 give their mean); any `ν ≤ 2` gives 2;
/// otherwise `E = Σ ν/(ν − 2)` and `2E / (E − q)`. An empty `nu` gives NaN.
pub(crate) fn fstat_ddf(nu: &[f64], tol: f64) -> f64 {
    match nu {
        [] => f64::NAN,
        [one] => *one,
        _ if nu.windows(2).all(|w| (w[1] - w[0]).abs() < tol) => {
            nu.iter().sum::<f64>() / nu.len() as f64
        }
        _ if nu.iter().any(|&v| v <= 2.0) => 2.0,
        _ => {
            let e: f64 = nu.iter().map(|&v| v / (v - 2.0)).sum();
            2.0 * e / (e - nu.len() as f64)
        }
    }
}

/// The contestMD row of a one-direction test: `F = t²`, NumDF 1, the t
/// test's df and p.
fn from_1d(t: Contest1D, rows: usize) -> ContestMD {
    ContestMD {
        f_value: t.t_value * t.t_value,
        num_df: 1,
        den_df: t.df,
        p_value: t.p_value,
        nu: vec![t.df],
        rows,
    }
}

/// `xᵀ M x` as lmerTest's `qform`, `Σ_i x_i (M x)_i`.
fn qform(x: &[f64], m: MatRef<'_, f64>) -> f64 {
    bilinear(x, m, x)
}

/// `xᵀ M y`, as `Σ_i x_i (M y)_i`.
fn bilinear(x: &[f64], m: MatRef<'_, f64>, y: &[f64]) -> f64 {
    debug_assert_eq!((m.nrows(), m.ncols()), (x.len(), y.len()));
    (0..x.len())
        .map(|i| x[i] * (0..y.len()).map(|j| m[(i, j)] * y[j]).sum::<f64>())
        .sum()
}

#[cfg(test)]
#[path = "contest_tests.rs"]
mod tests;
