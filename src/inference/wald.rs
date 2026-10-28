//! Wald tests of a linear hypothesis `Lβ = 0` on the fixed effects, as a
//! [`LinearTest`]: the Wald χ² from `β̂` and its covariance, and the F test of
//! [`Fit::contrast_f`](crate::Fit::contrast_f) in the same shape.
//!
//! The Wald χ² is `W = (Lβ̂)ᵀ (L V Lᵀ)⁺ (Lβ̂)` on `r = rank(L V Lᵀ)` df, `V`
//! the covariance of `β̂`. The pseudo-inverse and the rank come from the
//! eigendecomposition of `L V Lᵀ`, keeping the eigenvalues above `1e-8` of the
//! largest, so a set with redundant rows is tested on the directions it
//! spans. Under an invertible change of basis `L → ML` both `W` and `r` are
//! unchanged.

use faer::Mat;

use super::contrast::{ContrastF, ContrastNote, NoDfReason};
use super::evd::psd_pseudo_inverse;
use super::reference::{DenDf, Reference};
use super::tails::chisq_upper_p;

/// The eigenvalue threshold of `L V Lᵀ`, relative to its largest eigenvalue:
/// directions at or below it are not tested.
const RANK_REL_TOL: f64 = 1e-8;

/// A test of a linear hypothesis `Lβ = 0` on the fixed effects.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearTest {
    /// The F statistic, the Wald χ² or the likelihood-ratio χ², as `kind`
    /// says.
    pub statistic: f64,
    /// Which statistic `statistic` is.
    pub kind: LinearKind,
    /// The numerator df: the number of directions of `L` tested. NaN when
    /// the hypothesis was not tested.
    pub num_df: f64,
    /// What `statistic` is referred to: [`Reference::F`] for an F test,
    /// [`Reference::ChiSquared`] for a Wald or likelihood-ratio χ². On a
    /// Wald χ² that was not run (`num_df` NaN) or has rank 0,
    /// `ChiSquared { df: 0 }` is a placeholder and the p-value is NaN: read
    /// the df from `num_df` and the reason from `notes`.
    pub reference: Reference,
    /// `P(X ≥ statistic)` for `X` drawn from `reference`.
    pub p_value: f64,
    /// What the test could not take at face value.
    pub notes: Vec<LinearNote>,
}

/// The statistic of a [`LinearTest`].
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearKind {
    /// An F statistic.
    F,
    /// A Wald χ².
    ChiSquared,
    /// A likelihood-ratio χ², `2(ℓ̂ − ℓ̂₀)`, of the fit against its refit
    /// under `Lβ = 0`.
    LikelihoodRatio,
}

/// An observation about one linear test. Match the variant, never a message.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearNote {
    /// A row of `L` has a non-zero entry on this aliased column, so the
    /// hypothesis was not tested and the statistic, `num_df` and p-value are
    /// NaN.
    AliasedColumn {
        /// The column, 0-based over all `p` columns.
        column: usize,
    },
    /// `L V Lᵀ` has rank `rank < rows`: the test is on the `rank` directions
    /// it spans.
    RankDeficientL {
        /// The rank kept (the numerator df).
        rank: usize,
        /// The rows of `L`.
        rows: usize,
    },
    /// The F test has no denominator df on this fit
    /// ([`ContrastError::NoDf`](super::contrast::ContrastError::NoDf)), so the
    /// hypothesis was not tested and the statistic, both df and the p-value
    /// are NaN.
    NoDf {
        /// Why the fit has no df.
        reason: NoDfReason,
    },
    /// A model refitted for a likelihood-ratio row (the model under
    /// `Lβ = 0`, or on Type II the larger model too) did not converge, or its
    /// log-likelihood is not finite, so the statistic and p-value are NaN.
    ReducedNotConverged,
    /// The likelihood-ratio statistic `2(ℓ₁ − ℓ₀)` came out below zero by
    /// more than round-off: the model under `Lβ = 0` ended above the larger
    /// model's fit, so that fit did not reach its own optimum. The statistic
    /// is set to 0 (p-value 1).
    NegativeStatistic,
}

/// The Wald χ² of `Lβ = 0`: `W = (Lβ̂)ᵀ (L V Lᵀ)⁺ (Lβ̂)` on
/// `r = rank(L V Lᵀ)` df, with `beta` the `p` estimates, `vcov` their `p × p`
/// covariance, `aliased` the fit's aliased-column mask and `l` the `rows × p`
/// matrix `L`, row-major over all `p` columns, aliased ones included.
///
/// Aliased columns are left out of `β̂`, `V` and `L` (their entries of `beta`
/// and `vcov` are not read). A non-zero entry of `L` on one gives a NaN test
/// with [`LinearNote::AliasedColumn`] for each such column, ascending, and
/// `reference` `ChiSquared { df: 0 }`. A rank below `rows` takes
/// [`LinearNote::RankDeficientL`]. A non-finite entry of `V` among the
/// columns tested (a fit that did not converge, or a target subset) gives a
/// NaN statistic, `num_df` and p-value, `reference` `ChiSquared { df: 0 }`
/// and no note. Rank 0 (`L V Lᵀ` with no positive eigenvalue, as when `L` is
/// zero on every column tested) gives statistic `0.0`, `num_df` `0.0`,
/// `reference` `ChiSquared { df: 0 }`, a NaN p-value and
/// [`LinearNote::RankDeficientL`] with `rank` 0.
///
/// # Panics
///
/// When `rows` is 0, `beta`, `aliased` or a row of `vcov` does not have `p`
/// entries, `vcov` does not have `p` rows, `l.len()` is not `rows·p`, or `l`
/// is not finite.
// Its callers (the per-term tests and the two-fit comparison) are in the
// formula frontend, so a build without `formula` compiles it unused, unless
// `loop_advanced` is on, whose `wald_chisq_pub` calls it (the allow is then
// merely unneeded).
#[cfg_attr(not(feature = "formula"), allow(dead_code))]
pub(crate) fn wald_chisq(
    beta: &[f64],
    vcov: &[Vec<f64>],
    aliased: &[bool],
    l: &[f64],
    rows: usize,
) -> LinearTest {
    let p = beta.len();
    assert!(rows > 0, "wald_chisq: L must have at least one row");
    assert_eq!(aliased.len(), p, "wald_chisq: aliased must have p entries");
    assert!(
        vcov.len() == p && vcov.iter().all(|row| row.len() == p),
        "wald_chisq: vcov must be p × p"
    );
    assert_eq!(l.len(), rows * p, "wald_chisq: l must have rows·p elements");
    assert!(
        l.iter().all(|v| v.is_finite()),
        "wald_chisq: l must be finite"
    );
    let untested = |notes| LinearTest {
        statistic: f64::NAN,
        kind: LinearKind::ChiSquared,
        num_df: f64::NAN,
        reference: Reference::ChiSquared { df: 0 },
        p_value: f64::NAN,
        notes,
    };
    let notes: Vec<LinearNote> = (0..p)
        .filter(|&j| aliased[j] && l.chunks(p).any(|row| row[j] != 0.0))
        .map(|column| LinearNote::AliasedColumn { column })
        .collect();
    if !notes.is_empty() {
        return untested(notes);
    }
    let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let row = |i: usize| &l[i * p..(i + 1) * p];
    let lb: Vec<f64> = (0..rows)
        .map(|i| kept.iter().map(|&j| row(i)[j] * beta[j]).sum())
        .collect();
    // `L V Lᵀ` over the kept columns, the lower triangle mirrored so the
    // matrix is exactly symmetric.
    let lv: Vec<Vec<f64>> = (0..rows)
        .map(|i| {
            kept.iter()
                .map(|&b| kept.iter().map(|&a| row(i)[a] * vcov[a][b]).sum())
                .collect()
        })
        .collect();
    let mut m = Mat::<f64>::zeros(rows, rows);
    for i in 0..rows {
        for j in 0..=i {
            let v: f64 = kept
                .iter()
                .enumerate()
                .map(|(k, &b)| lv[i][k] * row(j)[b])
                .sum();
            m[(i, j)] = v;
            m[(j, i)] = v;
        }
    }
    let Ok(pinv) = psd_pseudo_inverse(m.as_ref(), RANK_REL_TOL) else {
        return untested(Vec::new());
    };
    let rank = pinv.rank;
    let mut statistic = 0.0;
    for i in 0..rows {
        for j in 0..rows {
            statistic += lb[i] * pinv.inv[(i, j)] * lb[j];
        }
    }
    let mut notes = Vec::new();
    if rank < rows {
        notes.push(LinearNote::RankDeficientL { rank, rows });
    }
    LinearTest {
        statistic,
        kind: LinearKind::ChiSquared,
        num_df: rank as f64,
        reference: Reference::ChiSquared { df: rank },
        p_value: chisq_upper_p(statistic, rank as f64),
        notes,
    }
}

/// [`wald_chisq`] for the loop tier (`crate::loop_advanced`, no semver
/// promise), with its contract and panics. A hypothesis with no direction
/// tested has a NaN p-value: a row with a non-zero entry on an aliased column
/// is not tested ([`LinearNote::AliasedColumn`]), and an `l` zero on every
/// column tested has 0 df (`num_df` 0, [`LinearNote::RankDeficientL`] with
/// `rank` 0), whose χ² on 0 df gives no p-value.
#[cfg(feature = "loop_advanced")]
#[doc(hidden)]
pub fn wald_chisq_pub(
    beta: &[f64],
    vcov: &[Vec<f64>],
    aliased: &[bool],
    l: &[f64],
    rows: usize,
) -> LinearTest {
    wald_chisq(beta, vcov, aliased, l, rows)
}

impl LinearTest {
    /// The F test of [`Fit::contrast_f`](crate::Fit::contrast_f) as a
    /// [`LinearTest`]: its F, df and p-value copied, referred to
    /// [`Reference::F`] labelled `den_df_kind`, which the caller keys on the
    /// presence of random effects: [`DenDf::Satterthwaite`] for a fit with
    /// them (`contrast_f` tests only the Gaussian linear mixed model among
    /// mixed fits), [`DenDf::Residual`] otherwise (OLS and the Gamma and
    /// inverse Gaussian GLMs with φ estimated). Its notes are carried over
    /// variant by variant.
    // Its caller, the per-term tests, is in the formula frontend.
    #[cfg_attr(not(feature = "formula"), allow(dead_code))]
    pub(crate) fn from_contrast_f(cf: ContrastF, den_df_kind: DenDf) -> LinearTest {
        let notes = cf
            .notes
            .iter()
            .map(|note| match *note {
                ContrastNote::AliasedColumn { column } => LinearNote::AliasedColumn { column },
                ContrastNote::RankDeficientL { rank, rows } => {
                    LinearNote::RankDeficientL { rank, rows }
                }
            })
            .collect();
        LinearTest {
            statistic: cf.f,
            kind: LinearKind::F,
            num_df: cf.num_df,
            reference: Reference::F {
                num_df: cf.num_df,
                den_df: cf.den_df,
                den_df_kind,
            },
            p_value: cf.p_value,
            notes,
        }
    }
}

#[cfg(test)]
#[path = "wald_tests.rs"]
mod tests;
