//! Likelihood-ratio tests: the statistic `T = 2(ℓ₁ − ℓ₀)` of a larger model
//! (log-likelihood `ℓ₁`) against a smaller one nested in it (`ℓ₀`), floored
//! at 0, and its p-value `P(X ≥ T)` for a χ² or a 50:50 mixture reference.
//!
//! **Why a mixture.** When the larger model adds a random-effect dimension,
//! the null puts that variance on the boundary of its space (a variance
//! cannot go below 0). The χ² approximation assumes an interior null and is
//! then conservative. Dropping one dimension from a correlated block of size
//! `d` (one variance and `d − 1` covariances) has the asymptotic null
//! `½χ²_{d−1} + ½χ²_d`: half the time the estimate sits on the boundary and
//! the variance contributes nothing (Self and Liang 1987; Stram and Lee
//! 1994). Fixed columns that differ too are interior parameters and add to
//! both components' df.
//!
//! **The χ²₀ convention.** At `d = 1` the lower component is χ²₀, a point
//! mass at zero: `P(χ²₀ ≥ T)` is 1 at `T ≤ 0` and 0 at `T > 0`. That is why
//! `T = 0` gives p = 1 while the smallest positive `T` gives p near ½.

// The likelihood-ratio arithmetic's only caller is the two-fit comparison of
// the formula frontend (`formula::anova::compare`), so a build without
// `formula` compiles it unused.
#![cfg_attr(not(feature = "formula"), allow(dead_code))]

use super::reference::{Objective, Reference};
use super::tails::chisq_upper_p;

/// A likelihood-ratio test of two nested models.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct LrTest {
    /// `T = 2(ℓ₁ − ℓ₀)` floored at 0; NaN when either log-likelihood is
    /// NaN or `−∞`.
    pub statistic: f64,
    /// `Δnpar`, the larger model's parameter count minus the smaller's.
    pub df: usize,
    /// What `statistic` is referred to.
    pub reference: Reference,
    /// `P(X ≥ statistic)` for `X` drawn from `reference`.
    pub p_value: f64,
    /// What the two fits maximised.
    pub objective: Objective,
    /// What the test could not take at face value.
    pub notes: Vec<LrNote>,
}

/// An observation about one likelihood-ratio test. Match the variant, never
/// a message.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LrNote {
    /// The larger model's fit ended below the smaller model's optimum, which
    /// is a feasible point of the larger model, so it did not reach its own
    /// optimum: `2(ℓ₁ − ℓ₀) = raw`, below zero by more than round-off. The
    /// statistic is 0.
    NegativeStatistic {
        /// The raw value `2(ℓ₁ − ℓ₀)`.
        raw: f64,
    },
    /// The reference is the 50:50 mixture (the random parts differ by one
    /// dimension), which is conservative: the true p-value is usually
    /// smaller. It is derived for one variance component at zero with every
    /// other parameter away from its boundary; `approximate` when the larger
    /// model has other groupings, or the smaller fit has a variance at
    /// exactly zero, where it holds only approximately.
    BoundaryMixture {
        /// Whether the mixture's derivation holds only approximately.
        approximate: bool,
    },
    /// The random parts differ by more than one dimension: the χ²(Δnpar)
    /// reference ignores the boundary and is conservative.
    BoundaryIgnored,
    /// The pair has the exact test's shape (two REML fits, the smaller one
    /// OLS, one random dimension apart), but the larger fit keeps no design
    /// summary: it is not one unweighted Gaussian `(1 | g)` (a fit given
    /// weights keeps none, even all equal to 1). The mixture is used.
    NoDesignSummary,
    /// The exact test refused the larger fit's design summary, for
    /// `reason`. The mixture is used. A grouping with more than 100 levels
    /// with rows is refused by the comparison (`TooManyLevels`), never by
    /// the core's `rlrt_design`; the core refuses the rest.
    ExactRlrtRefused {
        /// The refusal's reason, one plain sentence.
        reason: &'static str,
    },
    /// The two mixed fits use different quadrature node counts (Laplace is
    /// 1), so the test mixes two approximation errors.
    NodeCounts {
        /// The first side's node count.
        m0: u8,
        /// The second side's node count.
        m1: u8,
    },
    /// Two dispersion GLMs whose Wald F cannot be formed (the smaller fit's
    /// coefficient names, aliased columns dropped, are not among the larger
    /// fit's, no column is left to test, or a fit's arrays disagree in
    /// length), so the likelihood-ratio test is used, its χ² reference
    /// asymptotic.
    NoWaldF,
    /// A log-likelihood is not finite (a fit that failed or collapsed), so
    /// the statistic and the p-value are NaN. It replaces the notes on the
    /// p-value (the floor and the boundary notes), which say nothing about a
    /// p-value that was not computed; the notes that state a fact about the
    /// pair (node counts, no design summary, the exact test's refusal, no
    /// Wald F) stay before it.
    NonFiniteLoglik,
}

/// The round-off allowed below zero on `2(ℓ₁ − ℓ₀)` before the floor at 0
/// takes a note, in deviance units: twice the deviance tolerance within
/// which two fits of one model count as the same optimum (`4e-5`, ten times
/// the largest benign difference measured between optimisers), once for
/// each of the two fits.
pub(crate) const NEGATIVE_LR_ROUNDOFF: f64 = 8e-5;

/// The statistic `T = 2(ℓ₁ − ℓ₀)` from the smaller model's log-likelihood
/// `loglik_smaller` (ℓ₀) and the larger model's `loglik_larger` (ℓ₁), and the
/// note it takes.
///
/// A raw value below zero is floored at `0.0` (as lme4's `anova` does); below
/// `−NEGATIVE_LR_ROUNDOFF` the floor carries [`LrNote::NegativeStatistic`]
/// with the raw value. A NaN log-likelihood, or a `−∞` one on either side (a
/// fit that collapsed), gives NaN with no note, never the floor.
pub(crate) fn lr_statistic(loglik_smaller: f64, loglik_larger: f64) -> (f64, Option<LrNote>) {
    if loglik_larger == f64::NEG_INFINITY || loglik_smaller == f64::NEG_INFINITY {
        return (f64::NAN, None);
    }
    let raw = 2.0 * (loglik_larger - loglik_smaller);
    // `f64::max` would turn a NaN into 0.0.
    if raw > 0.0 || raw.is_nan() {
        return (raw, None);
    }
    let note = (raw < -NEGATIVE_LR_ROUNDOFF).then_some(LrNote::NegativeStatistic { raw });
    (0.0, note)
}

/// `P(X ≥ statistic)` for `X` drawn from `reference`: the χ² upper tail, or
/// the mixture's ([`chibar_upper_p`]). NaN for a NaN statistic, and for the
/// references whose p-value is not a function of the statistic alone (the
/// exact RLRT, whose p-value comes with its simulated null) or that no
/// likelihood-ratio test uses (F).
pub(crate) fn lr_p_value(statistic: f64, reference: &Reference) -> f64 {
    match *reference {
        Reference::ChiSquared { df } => chisq_upper_p(statistic, df as f64),
        Reference::ChiBarSquared { df_low } => chibar_upper_p(statistic, df_low),
        Reference::ExactRlrt { .. } | Reference::F { .. } => f64::NAN,
    }
}

/// Upper tail of the 50:50 mixture of χ²_k and χ²_{k+1} (`k = df_low`),
/// `½ Q(k, T) + ½ Q(k + 1, T)` with `Q` the χ² upper tail, and at `k = 0`
/// the χ²₀ convention of the module doc: `Q(0, T)` is 1 at `T ≤ 0` and 0 at
/// `T > 0`, a closed form, since a χ² on 0 df has no density to evaluate.
/// NaN for a NaN `t`.
pub(crate) fn chibar_upper_p(t: f64, df_low: usize) -> f64 {
    if t.is_nan() {
        return f64::NAN;
    }
    let q_low = match df_low {
        0 if t > 0.0 => 0.0,
        0 => 1.0,
        k => chisq_upper_p(t, k as f64),
    };
    0.5 * q_low + 0.5 * chisq_upper_p(t, (df_low + 1) as f64)
}

/// [`lr_p_value`] for the loop tier (`crate::loop_advanced`, no semver
/// promise): `P(X ≥ statistic)` for `X` drawn from `reference`, NaN for a NaN
/// statistic, for the exact RLRT and F references, and for
/// `ChiSquared { df: 0 }` (a χ² on 0 df has no upper tail to evaluate; the
/// caller builds the `Reference` itself, so equal parameter counts reach it).
#[cfg(feature = "loop_advanced")]
#[doc(hidden)]
pub fn lr_p_value_pub(statistic: f64, reference: &Reference) -> f64 {
    lr_p_value(statistic, reference)
}

#[cfg(test)]
#[path = "lr_tests.rs"]
mod lr_tests;
