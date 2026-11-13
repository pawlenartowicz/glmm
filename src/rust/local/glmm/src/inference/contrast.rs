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
//!
//! On a robust fit ([`SeType::Robust`]) a one-row test reads the fit's CR2 or
//! HC2 pieces instead, whatever `satterthwaite` says: the
//! robust SE `√(lᵀVl)` and the Pustejovsky–Tipton df of `l`. A test of more
//! rows stays model-based, with [`ContrastNote::ModelBasedJointTest`].

use std::fmt;

use faer::Mat;

use super::contest::{contest_1d, contest_md, fstat_ddf, ContestInputs};
use super::cr2::{Cr2Pieces, DF_SMALL};
use super::tails::{f_upper_p, t_two_sided_p};
use crate::lmm::satterthwaite::SatterthwaiteState;
use crate::{Fit, SeType};

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
    /// with φ estimated the residual df `ν = n − p`; NaN on such a fit that
    /// did not converge, whose `coef_df` is NaN too (as are this test's
    /// estimate, se, t and p). On a robust fit the Pustejovsky–Tipton df of `l`, which
    /// for `e_j` is [`Fit::coef_df`]`[j]`.
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
    /// The numerator df: the rank of `L·C·Lᵀ` under lmerTest's rule; 0 for an
    /// all-zero `L`.
    pub num_df: f64,
    /// The Satterthwaite denominator df (`ν = n − p` on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated). NaN, with every other field, on
    /// such a fit that did not converge, as [`ContrastT::df`] is. NaN for an
    /// all-zero `L`, which has no direction to take a df on.
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
    /// `L·C·Lᵀ` has rank `rank < rows`: the test is on the `rank` directions
    /// lmerTest keeps. Rank 0 (an all-zero `L`) tests nothing: the statistic,
    /// denominator df and p-value are NaN on the F test.
    RankDeficientL {
        /// The rank kept (the numerator df).
        rank: usize,
        /// The rows of `L`.
        rows: usize,
    },
    /// On a robust fit, one cluster alone identifies the contrast: no other
    /// cluster carries information on it, so its robust SE, t, df and p are
    /// NaN (as for [`crate::Note::RobustSeUnidentified`]).
    RobustUnidentified,
    /// On a robust fit, the contrast's robust df ([`ContrastT::df`]) are
    /// below 4, where the robust test's Type I error has no assurance (as for
    /// [`crate::Note::RobustDfSmall`]).
    RobustDfSmall,
    /// On a robust fit, a test of more than one row: there is no evidence
    /// for a robust joint test, so it is the model-based F.
    ModelBasedJointTest,
}

/// Why a contrast has no Satterthwaite df. The `Display` text names no Rust
/// option or type, so a port shows it as given.
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
    /// A Gaussian LMM `Fit` no pass ran on, such as one built by the loop
    /// tier's `FitView::into_fit`, REML or ML.
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
                "this kind of fit has no denominator df (a binomial, Poisson or negative binomial fit, a non-Gaussian GLMM, or a Gamma or inverse Gaussian GLM with φ held)"
            }
            NoDfReason::Disabled => {
                "Satterthwaite degrees of freedom were not computed for this fit"
            }
            NoDfReason::StateMissing => {
                "the fit carries no Satterthwaite record (a Gaussian LMM fit assembled without the Satterthwaite pass)"
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
            ContrastError::NoDf(reason) => write!(f, "{reason}"),
        }
    }
}

impl std::error::Error for ContrastError {}

impl Fit {
    /// The t test of `lᵀβ = 0` with lmerTest's Satterthwaite df (its
    /// `contest1D`), `l` over all `p` caller columns; on OLS and the
    /// dispersion GLMs with φ estimated, the residual df `n − p`. On a robust
    /// fit the robust SE and df, with [`ContrastNote::RobustUnidentified`] or
    /// [`ContrastNote::RobustDfSmall`]; a contrast touching a column outside
    /// [`crate::FitOptions::se_for`] is NaN there, with no note.
    ///
    /// # Errors
    ///
    /// [`ContrastError::NoDf`] when a model fit has no Satterthwaite df.
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
        if let Some(r) = self.robust_inputs() {
            return Ok(robust_t_of(&self.beta, &r, l));
        }
        Ok(contrast_t_of(&self.beta, self.df_state()?, l))
    }

    /// The F test of `Lβ = 0` with lmerTest's Satterthwaite denominator df
    /// (its `contestMD`), `l` the `rows × p` matrix `L`, row-major over all
    /// `p` caller columns. One row gives `F = t²` of [`Fit::contrast_t`]. On
    /// OLS and the dispersion GLMs with φ estimated the denominator df is
    /// `n − p`, the classical F. On a robust fit one row is the robust
    /// [`Fit::contrast_t`]'s t² on its df, and more rows the model-based F
    /// with [`ContrastNote::ModelBasedJointTest`]. An all-zero `L` gets
    /// `num_df` 0, NaN elsewhere, and [`ContrastNote::RankDeficientL`] with
    /// rank 0 on every route.
    ///
    /// # Errors
    ///
    /// [`ContrastError::NoDf`] when the fit has no Satterthwaite df (on a
    /// robust fit, only for more than one row).
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
        if let Some(r) = self.robust_inputs() {
            let model = self
                .df_state()
                .map_err(|ContrastError::NoDf(reason)| reason);
            return robust_f_of(&self.beta, &r, model, l, rows);
        }
        Ok(contrast_f_of(&self.beta, self.df_state()?, l, rows))
    }

    /// What a robust fit's tests read; `None` on a model fit.
    pub(crate) fn robust_inputs(&self) -> Option<RobustInputs<'_>> {
        (self.se_type == SeType::Robust)
            .then(|| RobustInputs::new(self.robust.as_ref(), &self.vcov, self.aliased()))
    }

    /// The pass's state (or the residual-only record of OLS and the
    /// dispersion GLMs) when it carries df, or why there are none. Every
    /// Gaussian LMM `Fit` carries a record (its state, or a switched-off,
    /// failed or not-run one) on either criterion; no record means a family
    /// with no df.
    pub(crate) fn df_state(&self) -> Result<&SatterthwaiteState, ContrastError> {
        let reason = match &self.satterthwaite {
            Some(st) if st.is_not_run() => NoDfReason::StateMissing,
            Some(st) if st.is_disabled() => NoDfReason::Disabled,
            Some(st) if st.is_failed() => NoDfReason::Failed,
            Some(st) => return Ok(st),
            None => NoDfReason::NotGaussianMixed,
        };
        Err(ContrastError::NoDf(reason))
    }
}

/// The test of [`Fit::contrast_t`] on `beta` (all `p` caller columns) and the
/// df record `st`, the only two things it reads. The caller has checked `l`:
/// `p` finite entries. `l` is tested as [`unit_rows`] scales it, so `lᵀCl`
/// stays in range at any scale of `l`; the estimate and se are scaled back.
pub(crate) fn contrast_t_of(beta: &[f64], st: &SatterthwaiteState, l: &[f64]) -> ContrastT {
    let p = beta.len();
    let notes = aliased_notes(st, l, p);
    if !notes.is_empty() {
        return ContrastT {
            estimate: f64::NAN,
            se: f64::NAN,
            t: f64::NAN,
            df: f64::NAN,
            p_value: f64::NAN,
            notes,
        };
    }
    let inputs = Inputs::new(beta, st);
    let (l, s) = unit_rows(l, p);
    let r = contest_1d(&inputs.contest(), &inputs.restrict(&l));
    // A residual-only record has no variance parameters: its df is ν.
    let (df, p_value) = if st.residual_only() {
        (st.nu, t_two_sided_p(r.t_value, st.nu))
    } else {
        (r.df, r.p_value)
    };
    ContrastT {
        estimate: r.estimate / s[0],
        se: r.std_error / s[0],
        t: r.t_value,
        df,
        p_value,
        notes,
    }
}

/// The test of [`Fit::contrast_f`] on `beta` (all `p` caller columns) and the
/// df record `st`, the only two things it reads. The caller has checked `l`:
/// `rows > 0` rows of `p` finite entries. Each row is scaled by
/// [`unit_rows`] before `L·C·Lᵀ` is formed, which leaves F and its df
/// unchanged and keeps `L·C·Lᵀ` in range at any scale of a row.
pub(crate) fn contrast_f_of(
    beta: &[f64],
    st: &SatterthwaiteState,
    l: &[f64],
    rows: usize,
) -> ContrastF {
    let p = beta.len();
    let nan = |notes| ContrastF {
        f: f64::NAN,
        num_df: f64::NAN,
        den_df: f64::NAN,
        p_value: f64::NAN,
        notes,
    };
    let notes = aliased_notes(st, l, p);
    if !notes.is_empty() {
        return nan(notes);
    }
    // lmerTest's one-row shortcut skips the rank rule an all-zero `L` needs.
    if l.iter().all(|&v| v == 0.0) {
        return zero_l(rows);
    }
    let inputs = Inputs::new(beta, st);
    let (l, _) = unit_rows(l, p);
    let reduced: Vec<Vec<f64>> = l.chunks(p).map(|row| inputs.restrict(row)).collect();
    let lr = Mat::from_fn(rows, inputs.beta.len(), |i, j| reduced[i][j]);
    // The eigendecomposition of `L·C·Lᵀ` fails on non-convergence or on a
    // non-finite `C` (the residual record of an OLS or dispersion GLM fit
    // that did not converge); the test is then not computed, so `den_df`
    // is NaN there.
    let Ok(md) = contest_md(&inputs.contest(), lr.as_ref()) else {
        return nan(Vec::new());
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
    ContrastF {
        f: md.f_value,
        num_df: md.num_df as f64,
        den_df,
        p_value,
        notes,
    }
}

/// The test of an all-zero `L` of `rows` rows: `L·C·Lᵀ` has rank 0, so there
/// is nothing to test.
fn zero_l(rows: usize) -> ContrastF {
    ContrastF {
        f: f64::NAN,
        num_df: 0.0,
        den_df: f64::NAN,
        p_value: f64::NAN,
        notes: vec![ContrastNote::RankDeficientL { rank: 0, rows }],
    }
}

/// Each `p`-wide row of `l` times a power of 2 that takes its largest |entry|
/// into `[1, 2)`, and each row's factor; a zero row stays zero, with factor 1.
/// The factor is exact, and the F, t and Wald χ² statistics and their df do
/// not change when a row is scaled, so this only keeps `L·C·Lᵀ` from
/// overflowing or underflowing when `L` is very large or very small.
// shortcut: a row whose own entries are ~1e150 apart can still overflow or
// underflow; scale the columns too if such a contrast is ever needed.
pub(crate) fn unit_rows(l: &[f64], p: usize) -> (Vec<f64>, Vec<f64>) {
    let mut l = l.to_vec();
    let factors = l
        .chunks_mut(p)
        .map(|row| {
            let max = row.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            if max == 0.0 {
                return 1.0;
            }
            // Clamped so the factor itself stays a normal double.
            let k = (max.log2().floor() as i32).clamp(-1020, 1020);
            let f = 2f64.powi(-k);
            row.iter_mut().for_each(|v| *v *= f);
            f
        })
        .collect();
    (l, factors)
}

/// [`ContrastNote::AliasedColumn`] for every column outside the state's
/// retained columns where a row of `l` (row-major, `p` wide) is non-zero,
/// ascending.
fn aliased_notes(st: &SatterthwaiteState, l: &[f64], p: usize) -> Vec<ContrastNote> {
    let Some(kept) = st.retained.as_deref() else {
        return Vec::new();
    };
    notes_on(|j| !kept.contains(&j), l, p)
}

/// [`ContrastNote::AliasedColumn`] for every column `aliased` where a row of
/// `l` is non-zero, ascending.
fn notes_on(aliased: impl Fn(usize) -> bool, l: &[f64], p: usize) -> Vec<ContrastNote> {
    (0..p)
        .filter(|&j| aliased(j) && l.chunks(p).any(|row| row[j] != 0.0))
        .map(|column| ContrastNote::AliasedColumn { column })
        .collect()
}

/// What a test of one contrast reads: the fit's model df record (or why it
/// has none), or a robust fit's pieces.
// Its reader, `pairwise`, is in the formula frontend.
#[cfg_attr(not(feature = "formula"), allow(dead_code))]
pub(crate) enum DfSource<'a> {
    Model(Result<&'a SatterthwaiteState, NoDfReason>),
    Robust(&'a RobustInputs<'a>),
}

/// What a robust fit's one-row tests read.
pub(crate) struct RobustInputs<'a> {
    /// `None` on a fit with no endpoint, whose robust tests are NaN.
    pieces: Option<&'a Cr2Pieces>,
    /// The robust covariance, `p × p` row-major: NaN outside the target
    /// columns and on an unidentified or aliased one.
    vcov: &'a [f64],
    aliased: &'a [bool],
}

impl<'a> RobustInputs<'a> {
    pub(crate) fn new(pieces: Option<&'a Cr2Pieces>, vcov: &'a [f64], aliased: &'a [bool]) -> Self {
        RobustInputs {
            pieces,
            vcov,
            aliased,
        }
    }
}

/// The robust test of [`Fit::contrast_t`] on `beta` and `r`. The caller has
/// checked `l`: `p` finite entries. The se, df and identification check read
/// `l` as [`unit_rows`] scales it, so `lᵀVl` stays in range at any scale of
/// `l`; the se is scaled back.
pub(crate) fn robust_t_of(beta: &[f64], r: &RobustInputs<'_>, l: &[f64]) -> ContrastT {
    let p = beta.len();
    let untested = |estimate, notes| ContrastT {
        estimate,
        se: f64::NAN,
        t: f64::NAN,
        df: f64::NAN,
        p_value: f64::NAN,
        notes,
    };
    let notes = notes_on(|j| r.aliased[j], l, p);
    if !notes.is_empty() {
        return untested(f64::NAN, notes);
    }
    let on: Vec<usize> = (0..p).filter(|&j| l[j] != 0.0).collect();
    let estimate = on.iter().map(|&j| l[j] * beta[j]).sum();
    let Some(pc) = r.pieces else {
        return untested(estimate, Vec::new());
    };
    let (ls, s) = unit_rows(l, p);
    if pc.identified_by_one(&ls) {
        return untested(estimate, vec![ContrastNote::RobustUnidentified]);
    }
    // `vcov` is NaN outside the targets and on a column one cluster alone
    // identifies; only the first leaves `l` untested.
    let outside = on.iter().any(|&j| {
        let mut e = vec![0.0; p];
        e[j] = 1.0;
        r.vcov[j * p + j].is_nan() && !pc.identified_by_one(&e)
    });
    let se = pc.var(&ls).sqrt() / s[0];
    if outside || se.is_nan() {
        return untested(estimate, Vec::new());
    }
    let df = pc.df(&ls);
    let t = estimate / se;
    ContrastT {
        estimate,
        se,
        t,
        df,
        p_value: t_two_sided_p(t, df),
        notes: if df < DF_SMALL {
            vec![ContrastNote::RobustDfSmall]
        } else {
            Vec::new()
        },
    }
}

/// The test of [`Fit::contrast_f`] on a robust fit: one row the robust t² on
/// its df, more the model-based F on `model`, the fit's df record, with
/// [`ContrastNote::ModelBasedJointTest`] when it ran. The caller has checked
/// `l`: `rows > 0` rows of `p` finite entries.
pub(crate) fn robust_f_of(
    beta: &[f64],
    r: &RobustInputs<'_>,
    model: Result<&SatterthwaiteState, NoDfReason>,
    l: &[f64],
    rows: usize,
) -> Result<ContrastF, ContrastError> {
    if rows == 1 {
        if l.iter().all(|&v| v == 0.0) {
            return Ok(zero_l(rows));
        }
        let t = robust_t_of(beta, r, l);
        return Ok(ContrastF {
            f: t.t * t.t,
            num_df: if t.t.is_nan() { f64::NAN } else { 1.0 },
            den_df: t.df,
            p_value: t.p_value,
            notes: t.notes,
        });
    }
    let mut cf = contrast_f_of(beta, model.map_err(ContrastError::NoDf)?, l, rows);
    if !cf.f.is_nan() {
        cf.notes.push(ContrastNote::ModelBasedJointTest);
    }
    Ok(cf)
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
    fn new(beta: &[f64], st: &'a SatterthwaiteState) -> Self {
        let beta = match st.retained.as_deref() {
            Some(kept) => kept.iter().map(|&j| beta[j]).collect(),
            None => beta.to_vec(),
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
