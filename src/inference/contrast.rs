//! t and F tests of linear contrasts of the fixed effects, with lmerTest's
//! Satterthwaite degrees of freedom: [`Fit::contrast_t`] and
//! [`Fit::contrast_f`]. OLS and the Gamma and inverse Gaussian GLMs with φ
//! estimated take the residual df `n − p` instead.
//!
//! Both read only what the fit's Satterthwaite pass (or residual df) left on
//! the `Fit` (no data, no refit), and both take `l` over all `p` caller columns, aliased
//! ones included. A contrast with a non-zero entry on an aliased column is
//! not tested: its fields are NaN and it carries
//! [`ContrastNote::AliasedColumn`] (estimability is not checked). Where no df
//! is available the methods return [`ContrastError::NoDf`], never a z test.

use std::fmt;

use faer::Mat;

use super::contest::{contest_1d, contest_md, fstat_ddf, ContestInputs};
use super::tails::{f_upper_p, t_two_sided_p};
use crate::lmm::satterthwaite::SatterthwaiteState;
use crate::Fit;

/// The t test of `lᵀβ = 0` ([`Fit::contrast_t`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastT {
    /// `lᵀβ̂`.
    pub estimate: f64,
    /// `√(lᵀCl)`, `C = Var(β̂)`.
    pub se: f64,
    /// `estimate / se`.
    pub t: f64,
    /// The df. On a Gaussian LMM the Satterthwaite df, lmerTest's
    /// `2v²/(gᵀAg)` (`contest1D`); for a unit vector `e_j` on a target column
    /// it equals [`Fit::coef_df`]`[j]` to rounding (the latter is the pass's
    /// own form, capped at ν). On OLS and on a Gamma or inverse Gaussian GLM
    /// with φ estimated the residual df `ν = n − p`, also on such a fit that
    /// did not converge, whose `coef_df` is NaN (as are this test's estimate,
    /// se, t and p).
    pub df: f64,
    /// Two-sided p of `t` on `df`.
    pub p_value: f64,
    /// What the test could not take at face value.
    pub notes: Vec<ContrastNote>,
}

/// The F test of `Lβ = 0` ([`Fit::contrast_f`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastF {
    /// The F statistic.
    pub f: f64,
    /// The numerator df: the rank of `L·C·Lᵀ` under lmerTest's rule.
    pub num_df: f64,
    /// The Satterthwaite denominator df (`ν = n − p` on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated). NaN, with every other field, on
    /// such a fit that did not converge, where [`ContrastT::df`] still reports
    /// ν: the eigendecomposition of its non-finite `L·C·Lᵀ` fails.
    pub den_df: f64,
    /// Upper-tail p of `f` on `(num_df, den_df)`.
    pub p_value: f64,
    /// What the test could not take at face value.
    pub notes: Vec<ContrastNote>,
}

/// An observation about one contrast. Match the variant, never a message.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContrastNote {
    /// `l` (or a row of `L`) has a non-zero entry on this aliased column, so
    /// the contrast was not tested and its fields are NaN.
    AliasedColumn {
        /// The caller column, 0-based.
        column: usize,
    },
    /// `L·C·Lᵀ` has rank `rank < rows`: the F test is on the `rank`
    /// directions lmerTest keeps.
    RankDeficientL {
        /// The rank kept (the numerator df).
        rank: usize,
        /// The rows of `L`.
        rows: usize,
    },
}

/// Why a contrast has no Satterthwaite df.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoDfReason {
    /// The fit has no df: neither a Gaussian linear mixed model nor OLS nor
    /// a Gamma or inverse Gaussian GLM with φ estimated (a held φ keeps z).
    /// Also the loop tier's `satterthwaite` on any workspace but a Gaussian
    /// LMM's, OLS included.
    NotGaussianMixed,
    /// The pass was switched off (`FitOptions::satterthwaite = false`).
    Disabled,
    /// A Gaussian LMM `Fit` that carries no pass, such as one built by the
    /// loop tier's `FitView::into_fit`.
    StateMissing,
    /// The pass failed: a deviance or covariance on its stencil was not
    /// finite, the eigendecomposition of its Hessian failed, or the fit has no
    /// finite endpoint to run it at (`Note::SatterthwaiteCurvature` with
    /// `nonfinite` set).
    Failed,
}

impl fmt::Display for NoDfReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            NoDfReason::NotGaussianMixed => {
                "the fit's family has no df (binomial, Poisson, negative binomial, a non-Gaussian GLMM, or a Gamma or inverse Gaussian GLM with φ held); the loop tier's satterthwaite serves Gaussian LMM workspaces only"
            }
            NoDfReason::Disabled => {
                "the Satterthwaite pass was switched off (FitOptions::satterthwaite = false)"
            }
            NoDfReason::StateMissing => {
                "the fit carries no Satterthwaite record (a Gaussian LMM fit assembled by the loop tier's FitView::into_fit, which runs no pass)"
            }
            NoDfReason::Failed => {
                "the Satterthwaite pass failed: the fit has no finite endpoint, or a deviance or covariance on its stencil was not finite"
            }
        })
    }
}

/// Why [`Fit::contrast_t`] or [`Fit::contrast_f`] returned no test.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContrastError {
    /// No Satterthwaite df is available on this fit.
    NoDf(NoDfReason),
}

impl fmt::Display for ContrastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContrastError::NoDf(reason) => write!(f, "no Satterthwaite df: {reason}"),
        }
    }
}

impl std::error::Error for ContrastError {}

impl Fit {
    /// The t test of `lᵀβ = 0` with lmerTest's Satterthwaite df (its
    /// `contest1D`), `l` over all `p` caller columns; on OLS and the
    /// dispersion GLMs with φ estimated, the residual df `n − p`.
    ///
    /// # Errors
    ///
    /// [`ContrastError::NoDf`] when the fit has no Satterthwaite df.
    ///
    /// # Panics
    ///
    /// When `l.len()` is not `p` or `l` is not finite.
    pub fn contrast_t(&self, l: &[f64]) -> Result<ContrastT, ContrastError> {
        let p = self.beta.len();
        assert_eq!(l.len(), p, "contrast_t: l must have p elements");
        assert!(
            l.iter().all(|v| v.is_finite()),
            "contrast_t: l must be finite"
        );
        let st = self.df_state()?;
        let notes = aliased_notes(st, l, p);
        if !notes.is_empty() {
            return Ok(ContrastT {
                estimate: f64::NAN,
                se: f64::NAN,
                t: f64::NAN,
                df: f64::NAN,
                p_value: f64::NAN,
                notes,
            });
        }
        let inputs = Inputs::new(self, st);
        let r = contest_1d(&inputs.contest(), &inputs.restrict(l));
        // A residual-only record has no variance parameters: its df is ν.
        let (df, p_value) = if st.residual_only() {
            (st.nu, t_two_sided_p(r.t_value, st.nu))
        } else {
            (r.df, r.p_value)
        };
        Ok(ContrastT {
            estimate: r.estimate,
            se: r.std_error,
            t: r.t_value,
            df,
            p_value,
            notes,
        })
    }

    /// The F test of `Lβ = 0` with lmerTest's Satterthwaite denominator df
    /// (its `contestMD`), `l` the `rows × p` matrix `L`, row-major over all
    /// `p` caller columns. One row gives `F = t²` of [`Fit::contrast_t`]. On
    /// OLS and the dispersion GLMs with φ estimated the denominator df is
    /// `n − p`, the classical F.
    ///
    /// # Errors
    ///
    /// [`ContrastError::NoDf`] when the fit has no Satterthwaite df.
    ///
    /// # Panics
    ///
    /// When `rows` is 0, `l.len()` is not `rows·p`, or `l` is not finite.
    pub fn contrast_f(&self, l: &[f64], rows: usize) -> Result<ContrastF, ContrastError> {
        let p = self.beta.len();
        assert!(rows > 0, "contrast_f: L must have at least one row");
        assert_eq!(l.len(), rows * p, "contrast_f: l must have rows·p elements");
        assert!(
            l.iter().all(|v| v.is_finite()),
            "contrast_f: l must be finite"
        );
        let st = self.df_state()?;
        let nan = |notes| ContrastF {
            f: f64::NAN,
            num_df: f64::NAN,
            den_df: f64::NAN,
            p_value: f64::NAN,
            notes,
        };
        let notes = aliased_notes(st, l, p);
        if !notes.is_empty() {
            return Ok(nan(notes));
        }
        let inputs = Inputs::new(self, st);
        let reduced: Vec<Vec<f64>> = l.chunks(p).map(|row| inputs.restrict(row)).collect();
        let lr = Mat::from_fn(rows, inputs.beta.len(), |i, j| reduced[i][j]);
        // The eigendecomposition of `L·C·Lᵀ` fails on non-convergence or on a
        // non-finite `C` (the residual record of an OLS or dispersion GLM fit
        // that did not converge); the test is then not computed, so `den_df`
        // is NaN there while `contrast_t`'s df is ν.
        let Ok(md) = contest_md(&inputs.contest(), lr.as_ref()) else {
            return Ok(nan(Vec::new()));
        };
        // A residual-only record: every direction's df is ν, so the mean
        // rule gives ν exactly.
        let (den_df, p_value) = if st.residual_only() {
            let den = fstat_ddf(&vec![st.nu; md.num_df], 1e-8);
            (den, f_upper_p(md.f_value, md.num_df as f64, den))
        } else {
            (md.den_df, md.p_value)
        };
        let mut notes = Vec::new();
        if md.num_df < rows {
            notes.push(ContrastNote::RankDeficientL {
                rank: md.num_df,
                rows,
            });
        }
        Ok(ContrastF {
            f: md.f_value,
            num_df: md.num_df as f64,
            den_df,
            p_value,
            notes,
        })
    }

    /// The pass's state (or the residual-only record of OLS and the
    /// dispersion GLMs) when it carries df, or why there are none. `reml`
    /// marks the Gaussian LMM routes, the only ones that run the pass.
    pub(crate) fn df_state(&self) -> Result<&SatterthwaiteState, ContrastError> {
        let reason = match &self.satterthwaite {
            Some(st) if st.disabled => NoDfReason::Disabled,
            Some(st) if st.failed => NoDfReason::Failed,
            Some(st) => return Ok(st),
            None if self.reml => NoDfReason::StateMissing,
            None => NoDfReason::NotGaussianMixed,
        };
        Err(ContrastError::NoDf(reason))
    }
}

/// [`ContrastNote::AliasedColumn`] for every column outside the state's
/// retained columns where a row of `l` (row-major, `p` wide) is non-zero,
/// ascending.
fn aliased_notes(st: &SatterthwaiteState, l: &[f64], p: usize) -> Vec<ContrastNote> {
    let Some(kept) = st.retained.as_deref() else {
        return Vec::new();
    };
    (0..p)
        .filter(|j| !kept.contains(j) && l.chunks(p).any(|row| row[*j] != 0.0))
        .map(|column| ContrastNote::AliasedColumn { column })
        .collect()
}

/// lmerTest's `(β, C, A, Jac)` over the state's columns: `A = A_eff` and
/// `Jac = [∂C/∂θ_k …, C/τ̂]`, the τ entry formed here. A residual-only record
/// has neither (`A` 0 × 0, no `Jac`): the contest layer then gives the
/// estimate, se and statistic, and the df is ν.
struct Inputs<'a> {
    beta: Vec<f64>,
    st: &'a SatterthwaiteState,
    jac: Vec<Mat<f64>>,
}

impl<'a> Inputs<'a> {
    fn new(fit: &Fit, st: &'a SatterthwaiteState) -> Self {
        let beta = match st.retained.as_deref() {
            Some(kept) => kept.iter().map(|&j| fit.beta[j]).collect(),
            None => fit.beta.clone(),
        };
        let c = &st.c;
        let mut jac = st.jac_theta.clone();
        if !st.residual_only() {
            jac.push(Mat::from_fn(c.nrows(), c.ncols(), |a, b| {
                c[(a, b)] / st.tau
            }));
        }
        Inputs { beta, st, jac }
    }

    fn contest(&self) -> ContestInputs<'_> {
        ContestInputs {
            beta: &self.beta,
            vcov_beta: self.st.c.as_ref(),
            vcov_varpar: self.st.a_eff.as_ref(),
            jac: &self.jac,
        }
    }

    /// A row over the caller's `p` columns, restricted to the state's.
    fn restrict(&self, row: &[f64]) -> Vec<f64> {
        match self.st.retained.as_deref() {
            Some(kept) => kept.iter().map(|&j| row[j]).collect(),
            None => row.to_vec(),
        }
    }
}

#[cfg(test)]
#[path = "contrast_tests.rs"]
mod tests;
