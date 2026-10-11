//! p-values and quantiles of the reference distributions behind the
//! hypothesis tests: the two-sided Student t p-value, the F and χ² upper
//! tails, and the t and normal quantiles.
//!
//! Every function returns NaN, never an error or a panic, for a NaN argument
//! or a df that is NaN, infinite or `≤ 0`; the quantile wrappers also for a
//! probability outside `[0, 1]`. They match R's `pt`, `pf`, `pchisq` (with
//! `lower.tail = FALSE`), `qt` and `qnorm`.
//!
//! Upper tails are evaluated directly (`ContinuousCdf::sf`), never as
//! `1 − CDF`: the CDF is a number near 1, so `1 − CDF` keeps only its absolute
//! precision and rounds any tail below about 1e-16 to 0. The χ²₁ upper tail at
//! 107.2, for example, is about 4e-25, a p-value a likelihood-ratio test can
//! legitimately report.

#[cfg(any(test, feature = "orchestrate"))]
use commonstats::dist::Normal;
use commonstats::dist::{ChiSquared, ContinuousCdf, FisherF, StudentT};

/// A df the distribution constructors accept: finite and `> 0` (NaN fails).
/// Checked first so a bad df gives NaN rather than a `StatError`.
fn valid_df(df: f64) -> bool {
    df.is_finite() && df > 0.0
}

/// Two-sided p-value of a t statistic on `df` degrees of freedom,
/// `p = 2·P(T_df > |t|)` (R: `2 * pt(-abs(t), df)`). `t = 0` gives 1 (capped,
/// so rounding never pushes it above 1) and `t = ±∞` gives 0. `df` may be
/// fractional (Satterthwaite).
pub(crate) fn t_two_sided_p(t: f64, df: f64) -> f64 {
    if t.is_nan() || !valid_df(df) {
        return f64::NAN;
    }
    StudentT::new(df).map_or(f64::NAN, |d| (2.0 * d.sf(t.abs())).min(1.0))
}

/// Upper-tail p-value `P(F > f)` of an F statistic on `(num_df, den_df)`
/// degrees of freedom (R: `pf(f, num_df, den_df, lower.tail = FALSE)`).
/// `f ≤ 0` (including `−∞`) gives 1 and `f = +∞` gives 0.
pub(crate) fn f_upper_p(f: f64, num_df: f64, den_df: f64) -> f64 {
    if f.is_nan() || !valid_df(num_df) || !valid_df(den_df) {
        return f64::NAN;
    }
    FisherF::new(num_df, den_df).map_or(f64::NAN, |d| d.sf(f))
}

/// Upper-tail p-value `P(χ²_df > x)` (R: `pchisq(x, df, lower.tail = FALSE)`).
/// `x ≤ 0` (including `−∞`) gives 1 and `x = +∞` gives 0.
#[cfg_attr(not(test), allow(dead_code))] // removed by P31-T2
pub(crate) fn chisq_upper_p(x: f64, df: f64) -> f64 {
    if x.is_nan() || !valid_df(df) {
        return f64::NAN;
    }
    ChiSquared::new(df).map_or(f64::NAN, |d| d.sf(x))
}

/// Quantile of Student's t on `df` degrees of freedom (R: `qt(p, df)`);
/// `df` may be fractional. `p = 0` and `p = 1` give `−∞` and `+∞`.
#[cfg(any(test, feature = "orchestrate"))]
pub(crate) fn t_quantile(p: f64, df: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || !valid_df(df) {
        return f64::NAN;
    }
    StudentT::new(df)
        .and_then(|d| d.quantile(p))
        .unwrap_or(f64::NAN)
}

/// Quantile of the standard normal (R: `qnorm(p)`). `p = 0` and `p = 1` give
/// `−∞` and `+∞`.
#[cfg(any(test, feature = "orchestrate"))]
pub(crate) fn normal_quantile(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    Normal::new(0.0, 1.0)
        .and_then(|d| d.quantile(p))
        .unwrap_or(f64::NAN)
}

#[cfg(test)]
#[path = "tails_tests.rs"]
mod tails_tests;
