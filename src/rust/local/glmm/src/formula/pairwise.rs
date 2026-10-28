//! Pairwise comparisons between the levels of one factor, emmeans'
//! `pairs(emmeans(..))`: [`pairwise`].
//!
//! The reference grid crosses the factor's levels, the `by` levels or chosen
//! values, and every level of the other factors of the fixed part; every
//! other numeric variable sits at its mean over the fit rows
//! ([`super::Lowered::numeric_means`]). Each grid row is lowered as the fit's
//! own rows were, and a cell's design row is the plain mean of its grid rows,
//! so the other factors are averaged with equal weights. A comparison is
//! `l'β̂` for `l` the difference of two cells' rows, with SE `√(l'Vl)`, tested
//! on the fit's df: Satterthwaite's for that `l` on a Gaussian linear mixed
//! model, `n − p` on OLS and the dispersion GLMs, and a z test on every other
//! family. The p-values are adjusted within each `by` family of comparisons.

use super::materialize::{new_fixed_design, Column, Lowered, Model, Table};
use super::parse::parse;
use super::Error;
use crate::inference::contrast::{contrast_t_of, ContrastError, ContrastNote, NoDfReason};
use crate::inference::tails::{
    chisq_upper_p, normal_quantile, t_quantile, tukey_quantile, tukey_upper_p,
};
use crate::inference::LinearNote;
use crate::lmm::satterthwaite::SatterthwaiteState;
use crate::{
    BinomialLink, Family, FitError, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink,
};

/// How the p-values of one family of comparisons (every pair within one
/// `by` level) are adjusted. NaN p-values stay NaN and are not counted in the
/// family, as in R's `p.adjust`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Adjust {
    /// Holm's step-down method (`p.adjust(method = "holm")`). emmeans'
    /// default for pairwise comparisons is Tukey instead.
    #[default]
    Holm,
    /// Each p-value times the number of comparisons, capped at 1.
    Bonferroni,
    /// Tukey's method: `P(Q_{k,df} > √2·|t|)`, `Q` the studentized range of
    /// `k` levels on the row's df (infinite on a z row), `k` the levels of the
    /// row's `by` family with an estimable mean.
    Tukey,
    /// No adjustment: `p_adjusted` is `p_value`.
    None,
}

/// What the comparisons are made within: each level of a factor of the
/// fixed part, or each chosen value of a numeric variable of the fixed part.
#[derive(Clone, Copy, Debug)]
pub enum By<'a> {
    /// The levels of this factor.
    Levels(&'a str),
    /// These values of this numeric variable.
    Values(&'a str, &'a [f64]),
}

/// The response-scale form of a comparison on a log or logit link.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RatioKind {
    /// The odds ratio, on a logit link.
    OddsRatio,
    /// The ratio of means (a rate ratio on a count family), on a log link.
    Ratio,
}

/// The pairwise comparisons of [`pairwise`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Pairwise {
    /// The factor whose levels are compared.
    pub factor: String,
    /// The variable the comparisons are made within, if any.
    pub by: Option<String>,
    /// The adjustment of `p_adjusted`.
    pub adjust: Adjust,
    /// The response-scale form of each row's `ratio`, `None` on a link that
    /// is neither log nor logit.
    pub ratio: Option<RatioKind>,
    /// One row per pair of levels `i < j` in the factor's level order,
    /// within each `by` level in its order.
    pub rows: Vec<PairwiseRow>,
    /// The factor's number of levels: each `by` family holds
    /// `levels·(levels − 1)/2` rows.
    pub(crate) levels: usize,
}

/// One comparison of two levels.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseRow {
    /// `"a - b"`: the earlier level first, in the factor's level order.
    pub contrast: String,
    /// The `by` level, or the chosen value as Rust prints it, when there is
    /// a `by`.
    pub by_level: Option<String>,
    /// `l'β̂`, on the link scale.
    pub estimate: f64,
    /// `√(l'Vl)`, `V` the covariance of `β̂`.
    pub se: f64,
    /// The df of the t test; `f64::INFINITY` on a z row.
    pub df: f64,
    /// `estimate / se`: t, or z when `df` is infinite.
    pub statistic: f64,
    /// The two-sided p-value, unadjusted.
    pub p_value: f64,
    /// The p-value adjusted within the row's `by` family.
    pub p_adjusted: f64,
    /// `exp(estimate)`, NaN when [`Pairwise::ratio`] is `None`.
    pub ratio: f64,
    /// `ratio · se` (the delta method), NaN when [`Pairwise::ratio`] is
    /// `None`.
    pub ratio_se: f64,
    /// [`LinearNote::AliasedColumn`] for a comparison that reads an aliased
    /// column (an empty cell), whose fields are then NaN;
    /// [`LinearNote::NoDf`] on a Gaussian linear mixed model without
    /// Satterthwaite df, whose `df`, `statistic` and p-values are then NaN.
    pub notes: Vec<LinearNote>,
}

/// The confidence intervals of a [`Pairwise`]'s comparisons, from
/// [`Pairwise::confint`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseIntervals {
    /// The confidence level, in `(0, 1)`.
    pub level: f64,
    /// The adjustment the intervals use: the result's, except
    /// [`Adjust::Bonferroni`] on an [`Adjust::Holm`] result, since Holm has
    /// no simultaneous interval.
    pub adjust: Adjust,
    /// One interval per row of [`Pairwise::rows`], in its order.
    pub rows: Vec<PairwiseInterval>,
}

/// The confidence interval of one comparison.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairwiseInterval {
    /// The lower bound of the estimate, on the link scale.
    pub lower: f64,
    /// The upper bound of the estimate, on the link scale.
    pub upper: f64,
    /// `exp(lower)`, the lower bound of the ratio; NaN when
    /// [`Pairwise::ratio`] is `None`.
    pub ratio_lower: f64,
    /// `exp(upper)`, the upper bound of the ratio; NaN when
    /// [`Pairwise::ratio`] is `None`.
    pub ratio_upper: f64,
}

impl Pairwise {
    /// Two-sided confidence intervals of the comparisons at `level`, from
    /// each row's estimate, SE and df, simultaneous within each `by` family
    /// as the result's adjustment is (emmeans' `confint` of `pairs()`). The
    /// half-width is the SE times: with [`Adjust::None`], the t quantile at
    /// `1 − α/2` on the row's df, the normal one on a z row (`α = 1 −
    /// level`); with [`Adjust::Bonferroni`], the same at `α/m`, `m` the rows
    /// of the family with a p-value; with [`Adjust::Tukey`], the
    /// studentized-range quantile at `α` of the family's `k` levels with an
    /// estimable mean on the row's df, over `√2`. Holm has no simultaneous interval, so a Holm
    /// result gets the Bonferroni intervals, and
    /// [`PairwiseIntervals::adjust`] says so. On a log or logit link the
    /// ratio's bounds are `exp` of the bounds. A row with NaN fields (an
    /// aliased cell, a fit without df) gets a NaN interval.
    ///
    /// No refit: one Tukey quantile costs 20–80 ms, and a row whose df
    /// repeats the previous row's reuses it, so an OLS or z result pays one
    /// and a Satterthwaite result one per row.
    ///
    /// # Errors
    /// [`FitError::OptionValue`] when `level` is not strictly between 0 and
    /// 1.
    pub fn confint(&self, level: f64) -> Result<PairwiseIntervals, FitError> {
        if !(level > 0.0 && level < 1.0) {
            return Err(FitError::OptionValue(format!(
                "level must be a number strictly between 0 and 1, got {level}"
            )));
        }
        let alpha = 1.0 - level;
        let adjust = match self.adjust {
            Adjust::Holm => Adjust::Bonferroni,
            a => a,
        };
        let k = self.levels;
        let family = (k * k.saturating_sub(1) / 2).max(1);
        let mut tukey: Option<(f64, f64, f64)> = None;
        let mut rows = Vec::with_capacity(self.rows.len());
        for chunk in self.rows.chunks(family) {
            let m = chunk.iter().filter(|r| !r.p_value.is_nan()).count() as f64;
            let k_tested = tukey_k(chunk.iter().map(|r| r.p_value), k);
            for row in chunk {
                let crit = match adjust {
                    Adjust::Tukey if k_tested < 2.0 => f64::NAN,
                    Adjust::Tukey => match tukey {
                        Some((kt, df, q)) if kt == k_tested && df == row.df => q,
                        _ => {
                            let q =
                                tukey_quantile(alpha, k_tested, row.df) / std::f64::consts::SQRT_2;
                            tukey = Some((k_tested, row.df, q));
                            q
                        }
                    },
                    Adjust::Bonferroni => two_sided(alpha / m, row.df),
                    _ => two_sided(alpha, row.df),
                };
                let (lower, upper) = (row.estimate - crit * row.se, row.estimate + crit * row.se);
                let (ratio_lower, ratio_upper) = match self.ratio {
                    Some(_) => (lower.exp(), upper.exp()),
                    None => (f64::NAN, f64::NAN),
                };
                rows.push(PairwiseInterval {
                    lower,
                    upper,
                    ratio_lower,
                    ratio_upper,
                });
            }
        }
        Ok(PairwiseIntervals {
            level,
            adjust,
            rows,
        })
    }
}

/// The two-sided critical value at `alpha`: the t quantile at `1 − alpha/2`
/// on `df`, or the normal one when `df` is infinite.
fn two_sided(alpha: f64, df: f64) -> f64 {
    let p = 1.0 - alpha / 2.0;
    if df == f64::INFINITY {
        normal_quantile(p)
    } else {
        t_quantile(p, df)
    }
}

/// The labels of the cells of a reference grid. Cell `b·k + i` is level `i`
/// of the factor's `k` levels within `by` label `b`.
pub(crate) struct CellLabels {
    /// The factor's levels, in the fit's order.
    pub(crate) levels: Vec<String>,
    /// The `by` levels or chosen values, `None` without a `by`.
    pub(crate) by: Option<Vec<String>>,
}

const FACTOR: &str = "a factor of the fixed part";
const VALUES: &str = "a numeric variable with chosen values";

/// The pairwise comparisons of the levels of `factor` in `model`, within each
/// level or value of `by`, with p-values adjusted by `adjust` (Holm by
/// default; emmeans' default is Tukey).
///
/// # Errors
/// [`Error::UnknownColumn`] when `factor` or the `by` variable is not a
/// variable of the fixed part; [`Error::WrongColumnKind`] for a numeric
/// `factor`, a numeric [`By::Levels`] or a factor [`By::Values`];
/// [`Error::Fit`] with [`FitError::OptionValue`] when `by` names `factor`;
/// and the lowering errors of a grid row ([`Error::TransformNotFinite`] for a
/// transform that is not finite at the mean).
pub fn pairwise(
    model: &Model,
    factor: &str,
    by: Option<By<'_>>,
    adjust: Adjust,
) -> Result<Pairwise, Error> {
    let fit = model.fit();
    let df = fit.df_state().map_err(|ContrastError::NoDf(reason)| reason);
    pairwise_of(
        model.formula(),
        model.lowered(),
        &fit.beta,
        &fit.vcov,
        fit.aliased(),
        df,
        factor,
        by,
        adjust,
    )
}

/// [`pairwise`] on a fit's numbers: `formula` lowered as `lowered`, and the
/// fit's `beta`, `vcov`, `aliased` and df record ([`pairwise_rows`]' `df`).
/// The ports call it with the numbers they stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pairwise_of(
    formula: &str,
    lowered: &Lowered,
    beta: &[f64],
    vcov: &[Vec<f64>],
    aliased: &[bool],
    df: Result<&SatterthwaiteState, NoDfReason>,
    factor: &str,
    by: Option<By<'_>>,
    adjust: Adjust,
) -> Result<Pairwise, Error> {
    let (grid, cell_rows, labels) = reference_grid(
        &fixed_factor_levels(lowered),
        &lowered.numeric_means,
        factor,
        by,
    )?;
    // The fixed design alone: the grid holds no grouping column, though a
    // fixed factor may also name a grouping (`(1 | Block/Variety)`).
    let ast = parse(formula)?;
    let (_, design) = new_fixed_design(&ast, lowered, &grid)?;
    let p = design.p;
    let cells: Vec<Vec<f64>> = cell_rows
        .iter()
        .map(|members| {
            let mut mean = vec![0.0; p];
            for &r in members {
                for (m, v) in mean.iter_mut().zip(&design.x[r * p..(r + 1) * p]) {
                    *m += v;
                }
            }
            mean.iter_mut().for_each(|m| *m /= members.len() as f64);
            mean
        })
        .collect();
    let ratio = ratio_kind(lowered.spec().family);
    Ok(Pairwise {
        factor: factor.to_string(),
        by: by.map(|b| match b {
            By::Levels(name) | By::Values(name, _) => name.to_string(),
        }),
        adjust,
        ratio,
        rows: pairwise_rows(beta, vcov, aliased, df, &cells, &labels, ratio, adjust),
        levels: labels.levels.len(),
    })
}

/// The factors of the fixed part of `lowered`, with their levels in the
/// fit's order: [`reference_grid`]'s `levels`.
pub(crate) fn fixed_factor_levels(lowered: &Lowered) -> Vec<(String, Vec<String>)> {
    lowered
        .factor_levels
        .iter()
        .filter(|(name, _)| {
            lowered.terms.iter().any(|t| {
                t.vars
                    .iter()
                    .zip(&t.numeric)
                    .any(|(v, &numeric)| !numeric && v == name)
            })
        })
        .cloned()
        .collect()
}

/// The response-scale form of a comparison on `family`'s link.
fn ratio_kind(family: Family) -> Option<RatioKind> {
    match family {
        Family::Binomial {
            link: BinomialLink::Logit,
        } => Some(RatioKind::OddsRatio),
        Family::Poisson {
            link: PoissonLink::Log,
        }
        | Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        }
        | Family::Gamma {
            link: GammaLink::Log,
        }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => Some(RatioKind::Ratio),
        _ => None,
    }
}

/// The reference grid of `factor` within `by`, as a table of the fixed
/// part's variables: `levels` the factors of the fixed part with their
/// levels, `means` [`super::Lowered::numeric_means`]. The factor varies
/// fastest, then the `by` variable, then the other factors in `levels`'
/// order; every numeric variable but a [`By::Values`] one is at its mean.
/// Also returns the grid rows of each cell and the cells' labels
/// ([`CellLabels`]).
///
/// # Errors
/// The refusals of [`pairwise`] on `factor` and `by`.
pub(crate) fn reference_grid(
    levels: &[(String, Vec<String>)],
    means: &[(String, f64)],
    factor: &str,
    by: Option<By<'_>>,
) -> Result<(Table, Vec<Vec<usize>>, CellLabels), Error> {
    let levels_of = |name: &str| levels.iter().find(|(n, _)| n == name).map(|(_, l)| l);
    let is_numeric = |name: &str| means.iter().any(|(n, _)| n == name);
    let refuse = |name: &str, expected: &'static str| {
        Err(if is_numeric(name) || levels_of(name).is_some() {
            Error::WrongColumnKind {
                name: name.to_string(),
                expected,
            }
        } else {
            Error::UnknownColumn(name.to_string())
        })
    };
    let Some(factor_levels) = levels_of(factor) else {
        return refuse(factor, FACTOR);
    };
    let (by_name, by_labels, by_values) = match by {
        None => (None, None, None),
        Some(By::Levels(name)) => match levels_of(name) {
            Some(l) => (Some(name), Some(l.clone()), None),
            None => return refuse(name, FACTOR),
        },
        Some(By::Values(name, values)) if is_numeric(name) => (
            Some(name),
            Some(values.iter().map(|v| v.to_string()).collect()),
            Some(values),
        ),
        Some(By::Values(name, _)) => return refuse(name, VALUES),
    };
    if by_name == Some(factor) {
        return Err(Error::Fit(FitError::OptionValue(format!(
            "by names {factor:?}, the factor compared; it must be another variable"
        ))));
    }
    let k = factor_levels.len();
    let nb = by_labels.as_ref().map_or(1, Vec::len);
    let others: Vec<&(String, Vec<String>)> = levels
        .iter()
        .filter(|(n, _)| n != factor && Some(n.as_str()) != by_name)
        .collect();
    let n = k * nb * others.iter().map(|(_, l)| l.len()).product::<usize>();
    let factor_column = |levels: &[String], code: &dyn Fn(usize) -> usize| Column::Factor {
        levels: levels.to_vec(),
        codes: (0..n).map(|r| code(r) as u32).collect(),
    };
    let mut columns = vec![(factor.to_string(), factor_column(factor_levels, &|r| r % k))];
    if let (Some(name), None) = (by_name, by_values) {
        let l = levels_of(name).expect("checked above");
        columns.push((name.to_string(), factor_column(l, &|r| (r / k) % nb)));
    }
    let mut stride = k * nb;
    for (name, l) in others {
        let (s, len) = (stride, l.len());
        columns.push((name.clone(), factor_column(l, &|r| (r / s) % len)));
        stride *= len;
    }
    for (name, mean) in means {
        let column = match by_values {
            Some(values) if Some(name.as_str()) == by_name => {
                (0..n).map(|r| values[(r / k) % nb]).collect()
            }
            _ => vec![*mean; n],
        };
        columns.push((name.clone(), Column::Numeric(column)));
    }
    let cells = (0..k * nb)
        .map(|c| (c..n).step_by(k * nb).collect())
        .collect();
    let labels = CellLabels {
        levels: factor_levels.clone(),
        by: by_labels,
    };
    Ok((Table { columns, n }, cells, labels))
}

/// The comparisons of every pair of cells `i < j` within each `by` family,
/// `cells` the cells' design rows over all `p` columns in [`CellLabels`]'
/// order, and their p-values adjusted per family. `df` is the fit's df record
/// ([`crate::Fit`]'s `df_state`), or why it has none:
/// [`NoDfReason::NotGaussianMixed`] takes the z test from `beta` and `vcov`,
/// any other reason gives rows with NaN df, statistic and p-values and
/// [`LinearNote::NoDf`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn pairwise_rows(
    beta: &[f64],
    vcov: &[Vec<f64>],
    aliased: &[bool],
    df: Result<&SatterthwaiteState, NoDfReason>,
    cells: &[Vec<f64>],
    labels: &CellLabels,
    ratio: Option<RatioKind>,
    adjust: Adjust,
) -> Vec<PairwiseRow> {
    let k = labels.levels.len();
    let families: Vec<Option<&String>> = match &labels.by {
        None => vec![None],
        Some(by) => by.iter().map(Some).collect(),
    };
    let mut out = Vec::new();
    for (b, by_level) in families.into_iter().enumerate() {
        let start = out.len();
        for i in 0..k {
            for j in i + 1..k {
                let l: Vec<f64> = cells[b * k + i]
                    .iter()
                    .zip(&cells[b * k + j])
                    .map(|(u, v)| u - v)
                    .collect();
                let row = match df {
                    Ok(st) => {
                        let t = contrast_t_of(beta, st, &l);
                        Test {
                            estimate: t.estimate,
                            se: t.se,
                            df: t.df,
                            statistic: t.t,
                            p_value: t.p_value,
                            notes: t
                                .notes
                                .iter()
                                .filter_map(|n| match *n {
                                    ContrastNote::AliasedColumn { column } => {
                                        Some(LinearNote::AliasedColumn { column })
                                    }
                                    ContrastNote::RankDeficientL { .. } => None,
                                })
                                .collect(),
                        }
                    }
                    Err(reason) => {
                        let mut z = z_test(beta, vcov, aliased, &l);
                        if reason != NoDfReason::NotGaussianMixed {
                            z.df = f64::NAN;
                            z.statistic = f64::NAN;
                            z.p_value = f64::NAN;
                            z.notes.push(LinearNote::NoDf { reason });
                        }
                        z
                    }
                };
                let (r, r_se) = match ratio {
                    Some(_) => {
                        let r = row.estimate.exp();
                        (r, r * row.se)
                    }
                    None => (f64::NAN, f64::NAN),
                };
                out.push(PairwiseRow {
                    contrast: format!("{} - {}", labels.levels[i], labels.levels[j]),
                    by_level: by_level.cloned(),
                    estimate: row.estimate,
                    se: row.se,
                    df: row.df,
                    statistic: row.statistic,
                    p_value: row.p_value,
                    p_adjusted: f64::NAN,
                    ratio: r,
                    ratio_se: r_se,
                    notes: row.notes,
                });
            }
        }
        if adjust == Adjust::Tukey {
            let k_tested = tukey_k(out[start..].iter().map(|r| r.p_value), k);
            for row in &mut out[start..] {
                let q = std::f64::consts::SQRT_2 * row.statistic.abs();
                row.p_adjusted = if k_tested >= 2.0 {
                    tukey_upper_p(q, k_tested, row.df)
                } else {
                    f64::NAN
                };
            }
        } else {
            let p: Vec<f64> = out[start..].iter().map(|r| r.p_value).collect();
            for (row, q) in out[start..].iter_mut().zip(adjust_p(&p, adjust)) {
                row.p_adjusted = q;
            }
        }
    }
    out
}

/// Tukey's `k` of one `by` family: the levels with an estimable mean, those
/// in at least one tested comparison (a p-value that is not NaN), as emmeans'
/// "tukey method for varying family sizes" counts them. `p_values` are the
/// family's rows `i < j` over `k` levels, in order.
fn tukey_k(p_values: impl Iterator<Item = f64>, k: usize) -> f64 {
    let mut tested = vec![false; k];
    let pairs = (0..k).flat_map(|i| (i + 1..k).map(move |j| (i, j)));
    for ((i, j), p) in pairs.zip(p_values) {
        if !p.is_nan() {
            tested[i] = true;
            tested[j] = true;
        }
    }
    tested.iter().filter(|&&t| t).count() as f64
}

/// One comparison's test before its labels and ratio.
struct Test {
    estimate: f64,
    se: f64,
    df: f64,
    statistic: f64,
    p_value: f64,
    notes: Vec<LinearNote>,
}

/// The z test of `l'β = 0` from `beta` and `vcov` over the columns not
/// `aliased`; the two-sided p is the χ²₁ tail of `z²`. A non-zero entry of
/// `l` on an aliased column gives a NaN test with
/// [`LinearNote::AliasedColumn`] for each such column.
fn z_test(beta: &[f64], vcov: &[Vec<f64>], aliased: &[bool], l: &[f64]) -> Test {
    let notes: Vec<LinearNote> = (0..l.len())
        .filter(|&j| aliased[j] && l[j] != 0.0)
        .map(|column| LinearNote::AliasedColumn { column })
        .collect();
    if !notes.is_empty() {
        return Test {
            estimate: f64::NAN,
            se: f64::NAN,
            df: f64::NAN,
            statistic: f64::NAN,
            p_value: f64::NAN,
            notes,
        };
    }
    let kept: Vec<usize> = (0..l.len()).filter(|&j| !aliased[j]).collect();
    let estimate: f64 = kept.iter().map(|&j| l[j] * beta[j]).sum();
    let var: f64 = kept
        .iter()
        .map(|&a| l[a] * kept.iter().map(|&b| vcov[a][b] * l[b]).sum::<f64>())
        .sum();
    let se = var.sqrt();
    let z = estimate / se;
    Test {
        estimate,
        se,
        df: f64::INFINITY,
        statistic: z,
        p_value: chisq_upper_p(z * z, 1.0),
        notes,
    }
}

/// `p` adjusted by `adjust` as one family, R's `p.adjust`: NaN entries stay
/// NaN and the family size `m` counts the others.
fn adjust_p(p: &[f64], adjust: Adjust) -> Vec<f64> {
    let tested: Vec<usize> = (0..p.len()).filter(|&i| !p[i].is_nan()).collect();
    let m = tested.len() as f64;
    let mut out = p.to_vec();
    match adjust {
        Adjust::None => {}
        Adjust::Bonferroni => {
            for &i in &tested {
                out[i] = (m * p[i]).min(1.0);
            }
        }
        Adjust::Holm => {
            // The i-th smallest p times `m − i + 1` (1-based), kept
            // non-decreasing in that order; a stable sort orders ties as R's
            // `order` does.
            let mut order = tested;
            order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
            let mut running = 0.0_f64;
            for (rank, &i) in order.iter().enumerate() {
                running = running.max((m - rank as f64) * p[i]);
                out[i] = running.min(1.0);
            }
        }
        Adjust::Tukey => unreachable!("Tukey reads each row's df and statistic, in pairwise_rows"),
    }
    out
}

#[cfg(test)]
#[path = "pairwise_tests.rs"]
mod tests;
