//! Comparing two fitted models: the formula nesting check, the checks two
//! fits must pass before a comparison, the refusals that name the fix, and
//! the test each pair gets.
//!
//! [`compare`] takes two [`Model`]s, plain numbers a port's fit object keeps,
//! never a [`Fit`] or a [`Lowered`], so a comparison of two stored fits needs
//! no refit and no data. Its checks run in this order, each a refusal
//! ([`CompareError`]):
//!
//! 1. both fits converged;
//! 2. the same data: the same row count, a bitwise-equal `y`, equal weights
//!    and offsets, the same family and link, and the same held dispersion;
//! 3. one formula nested in the other ([`nesting`]);
//! 4. the estimator: two OLS fits pass whatever their `reml`; otherwise one
//!    REML fit and one ML fit refuse, and so do two REML fits with different
//!    fixed parts;
//! 5. something to test: the two formulas differ in their fixed or random
//!    part, and the two fits in their parameter count.
//!
//! The two sides can be given in either order: the answer names the larger
//! one. Then the case:
//!
//! - two OLS fits: the exact F of `anova.lm` from the two residual sums of
//!   squares, on `(p₁ − p₀, n − p₁)`;
//! - two dispersion GLMs (Gamma, inverse Gaussian, φ estimated): the Wald F
//!   of the larger fit's columns the smaller one lacks, on `(r, n − p₁)`;
//! - every other pair: the likelihood-ratio test, `T = max(0, 2(ℓ₁ − ℓ₀))`,
//!   labelled with the larger model's objective. Its reference follows the
//!   boundary rule: random parts that are the same give χ²(Δnpar); random
//!   parts one dimension apart give the 50:50 mixture of χ²_{Δnpar−1} and
//!   χ²_{Δnpar}, or the exact restricted likelihood-ratio test when both
//!   fits are REML fits, the smaller is OLS and the larger keeps the design
//!   summary of one unweighted `(1 | g)`; random parts more than one
//!   dimension apart give χ²(Δnpar), which ignores the boundary.
//!
//! **A dropped component at zero.** When the random parts alone differ and
//! every variance the smaller model drops is exactly `0.0` on the diagonal
//! of the larger fit's `varcorr`, the larger fit's optimum lies in the
//! smaller model's space, and `T` is exactly 0 (p = 1 on every reference)
//! whatever the round-off between the two log-likelihoods. A NaN or `−∞`
//! log-likelihood on either side gives NaN instead, as on every
//! likelihood-ratio pair, with one note naming the cause.
//!
//! **The per-term tests of one fit.** [`term_tests`] tests each fixed term of
//! one fit from that fit alone, with no refit: Type III by default, Type II
//! on request, an F row or a Wald χ² row by family. Its table has no
//! random-effect rows: random effects are tested by comparing two models.
//!
//! The public items of this module are outside the crate's semver promise
//! until the 1.0 surface review: their names and signatures may change in any
//! release before then.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::hypotheses::{
    check_marginal, missing_cells_note, type2_vcov, type2_xtx, type3, TermHypothesis,
};
use super::parse::{parse, ParsedFormula, RandomEffect};
use super::{fixed_part_relation, FixedPartRelation, FixedTerm, Lowered, ParseError};
use crate::fit::vech_q;
use crate::inference::contrast::ContrastError;
use crate::inference::lr::{lr_p_value, lr_statistic};
use crate::inference::rlrt::{exact_rlrt, RlrtConfig};
use crate::inference::tails::f_upper_p;
use crate::inference::wald::wald_chisq;
use crate::inference::{
    DenDf, LinearKind, LinearNote, LinearTest, LrNote, LrTest, Objective, Reference,
};
use crate::{
    BinomialLink, Family, Fit, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink,
};

/// One of the two sides of a comparison, as given: `First` is `m0`, the first
/// formula or fit, and `Second` is `m1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The first side (`m0`).
    First,
    /// The second side (`m1`).
    Second,
}

impl Side {
    fn label(self) -> &'static str {
        match self {
            Side::First => "m0",
            Side::Second => "m1",
        }
    }
}

/// Which parts of two nested formulas differ.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Differs {
    /// The fixed part only; the random parts are the same.
    Fixed,
    /// The random part only; the fixed parts are the same.
    Random,
    /// Both parts.
    Both,
}

/// How two formulas relate when one is nested in the other ([`nesting`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nesting {
    /// The same fixed part and the same random part.
    Same,
    /// One formula is nested in the other.
    Nested {
        /// The side whose formula is the larger.
        larger: Side,
        /// Which parts differ.
        differs: Differs,
    },
}

/// Whether one formula is nested in the other, and which way, read from the
/// two formula strings alone: no data, no column names, no tolerance.
///
/// One formula is nested in the other when the response is the same, every
/// fixed term of the smaller one is in the larger one, and every grouping of
/// the smaller one is in the larger one with the same or more slopes. Terms,
/// groupings and slopes are compared as sets of variable names, so no order
/// matters: not the order of the terms, of the variables inside an
/// interaction (`a:b` is `b:a`), of the groupings, or of the slopes inside a
/// grouping, and `(1 | a/b)` holds the groupings `a` and `a:b`. The fixed
/// intercept is a term ([`ParsedFormula::has_intercept`]): `y ~ 0 + x` is
/// nested in `y ~ x`, and `y ~ 1` against `y ~ 0 + f` is refused although the
/// column spaces are nested when `f` is a factor. The offset is not read.
///
/// # Errors
/// [`CompareError::Formula`] when a formula does not parse,
/// [`CompareError::DifferentResponse`] when the responses differ, and
/// [`CompareError::NotNested`] when neither formula is nested in the other.
pub fn nesting(f0: &str, f1: &str) -> Result<Nesting, CompareError> {
    let parsed =
        |f: &str, side: Side| parse(f).map_err(|error| CompareError::Formula { side, error });
    let (p0, p1) = (parsed(f0, Side::First)?, parsed(f1, Side::Second)?);
    if p0.dependent != p1.dependent {
        return Err(CompareError::DifferentResponse {
            m0: p0.dependent,
            m1: p1.dependent,
        });
    }
    let (fixed_01, fixed_10) = match fixed_part_relation(&p0, &p1) {
        FixedPartRelation::Same => (true, true),
        FixedPartRelation::FirstInSecond => (true, false),
        FixedPartRelation::SecondInFirst => (false, true),
        FixedPartRelation::Neither => (false, false),
    };
    let (r0, r1) = (random_part(&p0), random_part(&p1));
    let (random_01, random_10) = (random_inside(&r0, &r1), random_inside(&r1, &r0));
    let differs = match (fixed_01 && fixed_10, random_01 && random_10) {
        (true, true) => return Ok(Nesting::Same),
        (true, false) => Differs::Random,
        (false, true) => Differs::Fixed,
        (false, false) => Differs::Both,
    };
    let larger = if fixed_01 && random_01 {
        Side::Second
    } else if fixed_10 && random_10 {
        Side::First
    } else {
        return Err(CompareError::NotNested {
            m0: f0.to_string(),
            m1: f1.to_string(),
        });
    };
    Ok(Nesting::Nested { larger, differs })
}

/// The variable names of a grouping or slope spelling (`"a:b"` is `{a, b}`).
fn vars(name: &str) -> BTreeSet<&str> {
    name.split(':').collect()
}

/// Each grouping (its variable names) with its slopes (each a set of
/// variable names). Every grouping carries an intercept.
fn random_part(f: &ParsedFormula) -> BTreeMap<BTreeSet<&str>, BTreeSet<BTreeSet<&str>>> {
    f.random_effects
        .iter()
        .map(|re| match re {
            RandomEffect::Intercept { group, .. } => (vars(group), BTreeSet::new()),
            RandomEffect::Slope {
                group,
                vars: slopes,
            } => (vars(group), slopes.iter().map(|s| vars(s)).collect()),
        })
        .collect()
}

/// Whether every grouping of `small` is in `large` with the same or more
/// slopes.
fn random_inside(
    small: &BTreeMap<BTreeSet<&str>, BTreeSet<BTreeSet<&str>>>,
    large: &BTreeMap<BTreeSet<&str>, BTreeSet<BTreeSet<&str>>>,
) -> bool {
    small
        .iter()
        .all(|(group, slopes)| large.get(group).is_some_and(|l| slopes.is_subset(l)))
}

/// What a comparison reads of one fitted model: plain numbers, as a port's fit
/// object keeps them, so two stored fits are compared with no refit.
///
/// [`Model::from_fit`] builds one from a [`Fit`] and the [`Lowered`] model it
/// was fitted on; a port builds one from the fields it stored.
#[derive(Clone, Debug)]
pub struct Model {
    /// The formula the model was fitted with.
    pub formula: String,
    /// The family and its link.
    pub family: Family,
    /// [`Fit::loglik`].
    pub loglik: f64,
    /// [`Fit::df`], the parameter count.
    pub df: usize,
    /// [`Fit::nobs`].
    pub nobs: usize,
    /// [`Fit::reml`]: whether `loglik` is a REML criterion.
    pub reml: bool,
    /// [`Fit::objective`].
    pub objective: Objective,
    /// [`Fit::converged`].
    pub converged: bool,
    /// [`Fit::varcorr`].
    pub varcorr: Vec<Vec<f64>>,
    /// Each `varcorr` block's grouping name and term names, in the same
    /// order ([`Lowered::re_groups`]); empty when there is no random part.
    /// The zero-component rule reads it to match blocks and terms by name;
    /// left empty beside a random part, it switches the rule off (the raw
    /// statistic is used), so fill it from the fit's `re_groups`.
    pub re_groups: Vec<(String, Vec<String>)>,
    /// The response as fitted ([`Lowered::y`]).
    pub y: Vec<f64>,
    /// The prior weights, `None` when unweighted.
    pub weights: Option<Vec<f64>>,
    /// The offset, `None` when there is none.
    pub offset: Option<Vec<f64>>,
    /// The dispersion held fixed ([`crate::FitOptions::dispersion`]), `None`
    /// when it is estimated or the family has none.
    pub dispersion_held: Option<f64>,
    /// The design summary `(counts, s, c)` of a Gaussian fit whose only random
    /// term is one unweighted `(1 | g)`, flat and column-major; `None` on
    /// every other fit.
    pub design_summary: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    /// The weighted residual sum of squares `Σwᵢ(yᵢ − μ̂ᵢ)²` of a converged
    /// OLS fit; NaN on every other fit.
    pub rss: f64,
    /// The coefficient names, one per fixed-effect column.
    pub coef_names: Vec<String>,
    /// The coefficients β̂ ([`Fit::beta`]).
    pub coef: Vec<f64>,
    /// The coefficients' covariance ([`Fit::vcov`]).
    pub vcov: Vec<Vec<f64>>,
    /// The aliased-column mask ([`Fit::aliased`]).
    pub aliased: Vec<bool>,
}

impl Model {
    /// The model of `fit`, fitted on `lowered` from `formula` with
    /// `lowered.opts` (the weights, the offset and the held dispersion are
    /// read from there).
    pub fn from_fit(formula: &str, lowered: &Lowered, fit: &Fit) -> Model {
        let opts = &lowered.opts;
        let ols = lowered.model.re.is_none() && lowered.model.family == Family::Gaussian;
        let rss = if ols && fit.converged() && fit.fitted.len() == lowered.y.len() {
            let w = |i: usize| opts.weights.as_ref().map_or(1.0, |w| w[i]);
            (0..lowered.y.len())
                .map(|i| w(i) * (lowered.y[i] - fit.fitted[i]).powi(2))
                .sum()
        } else {
            f64::NAN
        };
        Model {
            formula: formula.to_string(),
            family: lowered.model.family,
            loglik: fit.loglik,
            df: fit.df,
            nobs: fit.nobs,
            reml: fit.reml,
            objective: fit.objective,
            converged: fit.converged(),
            varcorr: fit.varcorr.clone(),
            re_groups: lowered
                .re_groups
                .iter()
                .map(|g| (g.name.clone(), g.terms.clone()))
                .collect(),
            y: lowered.y.clone(),
            weights: opts.weights.clone(),
            offset: opts.offset.clone(),
            dispersion_held: opts.dispersion,
            design_summary: fit
                .design_summary
                .as_ref()
                .map(|d| (d.counts.clone(), d.s.clone(), d.c.clone())),
            rss,
            coef_names: lowered.col_names.clone(),
            coef: fit.beta.clone(),
            vcov: fit.vcov.clone(),
            aliased: fit.aliased().to_vec(),
        }
    }

    /// An OLS fit: Gaussian with no random part, whatever its `reml`.
    fn is_ols(&self) -> bool {
        self.objective == Objective::LeastSquares
    }

    /// A Gamma or inverse Gaussian GLM whose dispersion is estimated. With φ
    /// held, the pair is an ordinary likelihood-ratio pair.
    fn is_dispersion_glm(&self) -> bool {
        matches!(
            self.family,
            Family::Gamma { .. } | Family::InverseGaussian { .. }
        ) && self.objective == Objective::MaximumLikelihood
            && self.dispersion_held.is_none()
    }

    /// The number of random-effect dimensions: the width of every `varcorr`
    /// block, summed.
    fn random_dims(&self) -> usize {
        self.varcorr.iter().map(|b| vech_q(b.len())).sum()
    }
}

/// The diagonal of a vech-packed covariance block ([`Fit::varcorr`]'s
/// layout): the variances.
fn diagonal(block: &[f64]) -> impl Iterator<Item = f64> + '_ {
    let q = vech_q(block.len());
    (0..q).map(move |i| block[i * q - (i * i - i) / 2])
}

/// The options of a comparison: the draws and the seed of the exact
/// restricted likelihood-ratio test. `nsim` is not checked here; callers
/// take `orchestrate::compare_options`, which bounds it at
/// `orchestrate::MAX_NSIM`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompareOptions {
    /// The number of draws to make; none are made when the statistic is 0
    /// or NaN.
    pub nsim: usize,
    /// The seed of the simulation.
    pub seed: u64,
}

/// What a comparison reports of one side, as given.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelInfo {
    /// The side's formula.
    pub formula_label: String,
    /// The parameter count ([`Model::df`]).
    pub npar: usize,
    /// The row count.
    pub nobs: usize,
    /// The log-likelihood, or the REML criterion on a REML fit.
    pub loglik: f64,
    /// `−2ℓ + 2·npar` on an ML fit; `None` on a REML fit.
    pub aic: Option<f64>,
    /// `−2ℓ + ln(nobs)·npar` on an ML fit; `None` on a REML fit.
    pub bic: Option<f64>,
    /// What the fit maximised.
    pub objective: Objective,
    /// The coefficient names, one per fixed-effect column ([`Model::coef_names`]).
    pub coef_names: Vec<String>,
}

impl ModelInfo {
    fn of(m: &Model) -> ModelInfo {
        let ml = |penalty: f64| (!m.reml).then(|| -2.0 * m.loglik + penalty * m.df as f64);
        ModelInfo {
            formula_label: m.formula.clone(),
            npar: m.df,
            nobs: m.nobs,
            loglik: m.loglik,
            aic: ml(2.0),
            bic: ml((m.nobs as f64).ln()),
            objective: m.objective,
            coef_names: m.coef_names.clone(),
        }
    }
}

/// The test of a [`Comparison`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonTest {
    /// A likelihood-ratio test.
    Lr(LrTest),
    /// The F test of two OLS fits, or of two dispersion GLMs.
    Linear(LinearTest),
}

/// Two models compared ([`compare`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Comparison {
    /// The first side as given.
    pub m0: ModelInfo,
    /// The second side as given.
    pub m1: ModelInfo,
    /// The side whose model is the larger; a table lists the other first.
    pub larger: Side,
    /// Which parts of the two formulas differ.
    pub differs: Differs,
    /// The test.
    pub test: ComparisonTest,
}

impl Comparison {
    /// The test's notes as plain sentences, in order, for a port to print
    /// under the table. A note on a fit names its side (`m0` or `m1`), and a
    /// note on a column of the larger fit names its coefficient.
    pub fn note_lines(&self) -> Vec<String> {
        let names = match self.larger {
            Side::First => &self.m0.coef_names,
            Side::Second => &self.m1.coef_names,
        };
        match &self.test {
            ComparisonTest::Lr(t) => t
                .notes
                .iter()
                .flat_map(|n| lr_note_lines(*n, self))
                .collect(),
            ComparisonTest::Linear(t) => t
                .notes
                .iter()
                .map(|n| linear_note_line(*n, names))
                .collect(),
        }
    }
}

fn lr_note_lines(note: LrNote, c: &Comparison) -> Vec<String> {
    let side = c.larger.label();
    let line = match note {
        LrNote::NegativeStatistic { raw } => format!(
            "{side}, the larger model, did not reach its optimum: its log-likelihood is below the \
             smaller model's, 2(l1 - l0) = {raw:e}, by more than round-off. The statistic is set \
             to 0; refit {side}."
        ),
        LrNote::BoundaryMixture { approximate } => {
            let mut lines = vec!["Conservative: the true p-value is usually smaller.".to_string()];
            if approximate {
                lines.push(
                    "Derived for one variance component at zero with every other parameter \
                     away from its boundary; with other groupings or components at zero it is \
                     approximate."
                        .to_string(),
                );
            }
            return lines;
        }
        LrNote::BoundaryIgnored => "The random parts differ by more than one dimension: the \
             chi-squared reference ignores the boundary; conservative."
            .to_string(),
        LrNote::NoDesignSummary => "The exact restricted likelihood-ratio test needs the larger \
             model to be a Gaussian REML fit whose only random term is one unweighted (1 | g), \
             which keeps a summary of its design; this fit keeps none (a fit given weights keeps \
             none, even weights all equal to 1), so the 50:50 mixture is used."
            .to_string(),
        LrNote::ExactRlrtRefused { reason } => format!(
            "The exact restricted likelihood-ratio test cannot be computed: {reason}. The 50:50 \
             mixture is used."
        ),
        LrNote::NodeCounts { m0, m1 } => format!(
            "The two fits use different quadrature node counts, {m0} and {m1}; the \
             likelihood-ratio test mixes two approximation errors."
        ),
        LrNote::NoWaldF => "The Wald F cannot be formed from the two fits' coefficients: it \
             needs the smaller fit's coefficients, aliased columns dropped, among the larger \
             fit's, at least one more left to test, and each fit's names, coefficients, \
             covariance and aliased flags of one length. The likelihood-ratio test is used, with \
             an asymptotic chi-squared reference."
            .to_string(),
        LrNote::NonFiniteLoglik => {
            let bad: Vec<&str> = [(&c.m0, "m0"), (&c.m1, "m1")]
                .into_iter()
                .filter(|(m, _)| !m.loglik.is_finite())
                .map(|(_, label)| label)
                .collect();
            format!(
                "The log-likelihood of {} is not finite (a fit that failed or collapsed), so \
                 the statistic and its p-value are NaN.",
                bad.join(" and ")
            )
        }
    };
    vec![line]
}

/// A [`LinearNote`] as a plain sentence; `names` are the fit's coefficient
/// names, by which a note on a column names it (never by its index, which a
/// port counts from 0 or 1). A column past the end of `names` (a hand-built
/// [`Comparison`]) gets a sentence that names no coefficient.
pub(crate) fn linear_note_line(note: LinearNote, names: &[String]) -> String {
    match note {
        LinearNote::AliasedColumn { column } => match names.get(column) {
            Some(name) => {
                format!("The coefficient {name} is aliased, so the hypothesis was not tested.")
            }
            None => "A tested coefficient is aliased, so the hypothesis was not tested.".into(),
        },
        LinearNote::RankDeficientL { rank, rows } => format!(
            "The {rows} tested coefficients span {rank} directions of their covariance; the F \
             is on those {rank}."
        ),
        LinearNote::NoDf { reason } => {
            format!("No denominator df ({reason}), so the hypothesis was not tested.")
        }
    }
}

/// Why two models cannot be compared. Each message says what is wrong and
/// what to do in plain words; `m0` and `m1` are the two sides as given. The
/// messages name no call syntax: a port that names its own comparison
/// function or REML argument renders that from the variant's fields.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum CompareError {
    /// A side did not converge (`true` for each side that did not).
    NotConverged {
        /// Whether `m0` did not converge.
        m0: bool,
        /// Whether `m1` did not converge.
        m1: bool,
    },
    /// The two fits have different row counts.
    DifferentRows {
        /// `m0`'s row count.
        m0: usize,
        /// `m1`'s row count.
        m1: usize,
    },
    /// The two responses differ in at least one entry (bitwise).
    DifferentY,
    /// The two fits have different prior weights (no weights are unit weights).
    DifferentWeights,
    /// The two fits have different offsets (no offset is a zero offset).
    DifferentOffset,
    /// The two fits have different families or links.
    DifferentFamily {
        /// `m0`'s family.
        m0: Family,
        /// `m1`'s family.
        m1: Family,
    },
    /// The two fits hold different dispersions (`None`: estimated, or no
    /// dispersion).
    DifferentDispersionHeld {
        /// `m0`'s held dispersion.
        m0: Option<f64>,
        /// `m1`'s held dispersion.
        m1: Option<f64>,
    },
    /// A formula does not parse.
    Formula {
        /// The side whose formula does not parse.
        side: Side,
        /// The parser's error.
        error: ParseError,
    },
    /// The two formulas have different responses.
    DifferentResponse {
        /// `m0`'s response.
        m0: String,
        /// `m1`'s response.
        m1: String,
    },
    /// Neither formula is nested in the other.
    NotNested {
        /// `m0`'s formula.
        m0: String,
        /// `m1`'s formula.
        m1: String,
    },
    /// One REML fit and one ML fit, not two OLS fits.
    MixedEstimators {
        /// The side fitted by REML.
        reml: Side,
    },
    /// Two REML fits with different fixed parts, not two OLS fits.
    RemlDifferentFixed,
    /// The two formulas have the same fixed and random parts
    /// ([`Nesting::Same`]), so there is nothing to test.
    SameParts,
    /// The two formulas are nested but the two fits have the same parameter
    /// count (`m0 == m1`), so there is nothing to test.
    NothingToTest {
        /// `m0`'s parameter count.
        m0: usize,
        /// `m1`'s parameter count.
        m1: usize,
    },
}

impl fmt::Display for CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareError::NotConverged { m0, m1 } => {
                let who = match (m0, m1) {
                    (true, true) => "m0 and m1 did not converge, so their log-likelihoods are",
                    (true, false) => "m0 did not converge, so its log-likelihood is",
                    _ => "m1 did not converge, so its log-likelihood is",
                };
                write!(
                    f,
                    "{who} not at a maximum and cannot be compared. Refit until the fit \
                     converges (a simpler random part often helps), then compare again."
                )
            }
            CompareError::DifferentRows { m0, m1 } => write!(
                f,
                "m0 has {m0} rows and m1 has {m1}; a comparison needs both fits on the same \
                 rows. Fit both models to the same rows, dropping the rows with a missing \
                 value in any variable of either formula first."
            ),
            CompareError::DifferentY => write!(
                f,
                "m0 and m1 have different responses y (the same row count, different values); \
                 a comparison needs both fits on the same data. Fit both models to the same \
                 data."
            ),
            CompareError::DifferentWeights => write!(
                f,
                "m0 and m1 were fitted with different weights; a comparison needs the same \
                 weights in both. Refit both models with the same weights."
            ),
            CompareError::DifferentOffset => write!(
                f,
                "m0 and m1 were fitted with different offsets; a comparison needs the same \
                 offset in both. Refit both models with the same offset."
            ),
            CompareError::DifferentFamily { m0, m1 } => write!(
                f,
                "m0 was fitted as {} and m1 as {}; a comparison needs the same family and \
                 link in both. Refit both models with the same family and link.",
                family_words(*m0),
                family_words(*m1)
            ),
            CompareError::DifferentDispersionHeld { m0, m1 } => {
                let held = |d: &Option<f64>| match d {
                    Some(v) => format!("holds the dispersion at {v}"),
                    None => "does not hold the dispersion".to_string(),
                };
                write!(
                    f,
                    "m0 {} and m1 {}; a comparison needs the same held dispersion in both. \
                     Refit both with the same dispersion argument.",
                    held(m0),
                    held(m1)
                )
            }
            CompareError::Formula { side, error } => {
                write!(f, "{}'s formula does not parse: {error}", side.label())
            }
            CompareError::DifferentResponse { m0, m1 } => write!(
                f,
                "the two formulas have different responses ({m0} and {m1}), so neither model \
                 is nested in the other. Compare two models of the same response."
            ),
            CompareError::NotNested { m0, m1 } => write!(
                f,
                "neither formula is nested in the other ({m0} against {m1}): each has a fixed \
                 term, a grouping or a random slope the other lacks. A likelihood-ratio test \
                 needs the smaller model's terms, groupings and slopes inside the larger \
                 model's; compare two nested models, or compare these two by AIC and BIC."
            ),
            CompareError::MixedEstimators { reml } => {
                let (m0, m1) = match reml {
                    Side::First => ("REML", "ML"),
                    Side::Second => ("ML", "REML"),
                };
                write!(
                    f,
                    "m0 was fitted by {m0} and m1 by {m1}; a comparison needs both fitted the \
                     same way. Refit both models by maximum likelihood (ML), or both by REML \
                     when their fixed parts are the same."
                )
            }
            CompareError::RemlDifferentFixed => write!(
                f,
                "m0 and m1 are REML fits with different fixed effects, and REML fits with \
                 different fixed effects cannot be compared by likelihood ratio. Refit both \
                 models by maximum likelihood (ML)."
            ),
            CompareError::SameParts => write!(
                f,
                "m0 and m1 have the same fixed and random parts, so there is nothing to test. \
                 Compare two models that differ by at least one parameter."
            ),
            CompareError::NothingToTest { m0, .. } => write!(
                f,
                "m0 and m1 have the same number of parameters ({m0}), so there is nothing to \
                 test: the larger formula adds no parameter to the smaller one. Compare two \
                 models that differ by at least one parameter."
            ),
        }
    }
}

impl std::error::Error for CompareError {}

/// The family and its link in plain words, as in "poisson with log link".
fn family_words(family: Family) -> String {
    let (name, link) = match family {
        Family::Gaussian => ("gaussian", "identity"),
        Family::Binomial { link } => (
            "binomial",
            match link {
                BinomialLink::Logit => "logit",
                BinomialLink::Probit => "probit",
                BinomialLink::Cloglog => "cloglog",
            },
        ),
        Family::Poisson {
            link: PoissonLink::Log,
        } => ("poisson", "log"),
        Family::Gamma { link } => (
            "gamma",
            match link {
                GammaLink::Log => "log",
                GammaLink::Inverse => "inverse",
            },
        ),
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        } => ("negative binomial", "log"),
        Family::InverseGaussian { link } => (
            "inverse gaussian",
            match link {
                InverseGaussianLink::Log => "log",
                InverseGaussianLink::InverseSquared => "inverse squared",
            },
        ),
    };
    format!("{name} with {link} link")
}

/// Compares two fitted models, given in either order: runs the checks, then
/// the test of the pair's case (see the module header for both). The exact
/// restricted likelihood-ratio test draws `opts.nsim` times from `opts.seed`,
/// serially, so one call gives the same p-value in every build.
///
/// # Errors
/// The first check that fails, as a [`CompareError`].
pub fn compare(a: &Model, b: &Model, opts: &CompareOptions) -> Result<Comparison, CompareError> {
    // 1. Both converged.
    if !(a.converged && b.converged) {
        return Err(CompareError::NotConverged {
            m0: !a.converged,
            m1: !b.converged,
        });
    }
    // 2. The same data.
    if a.nobs != b.nobs {
        return Err(CompareError::DifferentRows {
            m0: a.nobs,
            m1: b.nobs,
        });
    }
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    if bits(&a.y) != bits(&b.y) {
        return Err(CompareError::DifferentY);
    }
    if !same_or_default(&a.weights, &b.weights, a.nobs, 1.0) {
        return Err(CompareError::DifferentWeights);
    }
    if !same_or_default(&a.offset, &b.offset, a.nobs, 0.0) {
        return Err(CompareError::DifferentOffset);
    }
    if a.family != b.family {
        return Err(CompareError::DifferentFamily {
            m0: a.family,
            m1: b.family,
        });
    }
    if a.dispersion_held.map(f64::to_bits) != b.dispersion_held.map(f64::to_bits) {
        return Err(CompareError::DifferentDispersionHeld {
            m0: a.dispersion_held,
            m1: b.dispersion_held,
        });
    }
    // 3. Nested, and which way.
    let nested = nesting(&a.formula, &b.formula)?;
    // 4. The estimator. Two OLS fits pass whatever their `reml`.
    if !(a.is_ols() && b.is_ols()) {
        match (a.reml, b.reml) {
            (true, false) => {
                return Err(CompareError::MixedEstimators { reml: Side::First });
            }
            (false, true) => {
                return Err(CompareError::MixedEstimators { reml: Side::Second });
            }
            (true, true)
                if matches!(
                    nested,
                    Nesting::Nested {
                        differs: Differs::Fixed | Differs::Both,
                        ..
                    }
                ) =>
            {
                return Err(CompareError::RemlDifferentFixed);
            }
            _ => {}
        }
    }
    // 5. Something to test.
    let Nesting::Nested { larger, differs } = nested else {
        return Err(CompareError::SameParts);
    };
    if a.df == b.df {
        return Err(CompareError::NothingToTest { m0: a.df, m1: b.df });
    }
    Ok(Comparison {
        m0: ModelInfo::of(a),
        m1: ModelInfo::of(b),
        larger,
        differs,
        test: case(a, b, larger, differs, opts),
    })
}

/// The test of a pair that passed the checks.
fn case(
    a: &Model,
    b: &Model,
    larger: Side,
    differs: Differs,
    opts: &CompareOptions,
) -> ComparisonTest {
    let (small, large) = match larger {
        Side::First => (b, a),
        Side::Second => (a, b),
    };
    if a.is_ols() && b.is_ols() {
        return ComparisonTest::Linear(ols_f(small, large));
    }
    let mut notes = Vec::new();
    if a.is_dispersion_glm() && b.is_dispersion_glm() {
        match wald_f(small, large) {
            Some(test) => return ComparisonTest::Linear(test),
            None => notes.push(LrNote::NoWaldF),
        }
    }
    let nodes = |o: Objective| match o {
        Objective::Laplace => Some(1),
        Objective::Agq { nodes } => Some(nodes),
        _ => None,
    };
    if let (Some(k0), Some(k1)) = (nodes(a.objective), nodes(b.objective)) {
        if k0 != k1 {
            notes.push(LrNote::NodeCounts { m0: k0, m1: k1 });
        }
    }
    let mut test = lr_test(small, large, differs, opts, notes);
    // A NaN statistic comes from a log-likelihood that is not finite: one
    // note names that, in place of the notes on a p-value it does not have.
    // The notes that state a fact about the pair stay.
    if test.statistic.is_nan() {
        test.notes.retain(|note| match note {
            LrNote::NegativeStatistic { .. }
            | LrNote::BoundaryMixture { .. }
            | LrNote::BoundaryIgnored
            | LrNote::NonFiniteLoglik => false,
            LrNote::NoDesignSummary
            | LrNote::ExactRlrtRefused { .. }
            | LrNote::NodeCounts { .. }
            | LrNote::NoWaldF => true,
        });
        test.notes.push(LrNote::NonFiniteLoglik);
    }
    ComparisonTest::Lr(test)
}

/// The likelihood-ratio test of `large` against `small`, labelled with the
/// larger model's objective, its reference by the boundary rule.
fn lr_test(
    small: &Model,
    large: &Model,
    differs: Differs,
    opts: &CompareOptions,
    mut notes: Vec<LrNote>,
) -> LrTest {
    let df = large.df.saturating_sub(small.df);
    // The zero-component rule needs both log-likelihoods above `−∞` (false
    // for NaN): a fit that failed or collapsed gives NaN, never T = 0.
    let usable = |l: f64| l > f64::NEG_INFINITY;
    let zero = differs == Differs::Random && dropped_at_zero(small, large);
    let (statistic, floor) = if zero && usable(small.loglik) && usable(large.loglik) {
        (0.0, None)
    } else {
        lr_statistic(small.loglik, large.loglik)
    };
    notes.extend(floor);
    let reference = match large.random_dims().saturating_sub(small.random_dims()) {
        0 => Reference::ChiSquared { df },
        1 => {
            if small.is_ols() && small.reml && large.reml && differs == Differs::Random {
                match &large.design_summary {
                    Some((counts, s, c)) => {
                        let cfg = RlrtConfig {
                            nsim: opts.nsim,
                            seed: opts.seed,
                            parallel_inner: false,
                        };
                        let pinned = large.varcorr.first().and_then(|b| b.first()) == Some(&0.0);
                        match exact_rlrt(counts, s, c, large.loglik, small.loglik, pinned, &cfg) {
                            Ok(r) => {
                                return LrTest {
                                    statistic: r.statistic,
                                    df,
                                    reference: Reference::ExactRlrt {
                                        nsim: r.nsim,
                                        seed: r.seed,
                                    },
                                    p_value: r.p_value,
                                    objective: large.objective,
                                    notes,
                                };
                            }
                            Err(refusal) => notes.push(LrNote::ExactRlrtRefused {
                                reason: refusal.reason(),
                            }),
                        }
                    }
                    None => notes.push(LrNote::NoDesignSummary),
                }
            }
            let approximate = large.varcorr.len() > 1
                || small.varcorr.iter().any(|b| diagonal(b).any(|v| v == 0.0));
            notes.push(LrNote::BoundaryMixture { approximate });
            Reference::ChiBarSquared {
                df_low: df.saturating_sub(1),
            }
        }
        _ => {
            notes.push(LrNote::BoundaryIgnored);
            Reference::ChiSquared { df }
        }
    };
    LrTest {
        statistic,
        df,
        reference,
        p_value: lr_p_value(statistic, &reference),
        objective: large.objective,
        notes,
    }
}

/// Whether the larger model's fit puts every variance the smaller model
/// drops at exactly `0.0` (and drops at least one). Blocks and terms are
/// matched by name ([`Model::re_groups`]), never by position: a term of a
/// larger block is dropped when no block of the smaller fit with the same
/// grouping has it. Names are compared as sets of variable names (`a:b` is
/// `b:a`). A grouping lowering dropped (one level) has a block on neither
/// side.
fn dropped_at_zero(small: &Model, large: &Model) -> bool {
    if large.re_groups.len() != large.varcorr.len() {
        return false;
    }
    let mut dropped_any = false;
    for ((group, terms), block) in large.re_groups.iter().zip(&large.varcorr) {
        if vech_q(block.len()) != terms.len() {
            return false;
        }
        let kept: BTreeSet<BTreeSet<&str>> = small
            .re_groups
            .iter()
            .filter(|(g, _)| vars(g) == vars(group))
            .flat_map(|(_, t)| t.iter().map(|t| vars(t)))
            .collect();
        for (term, variance) in terms.iter().zip(diagonal(block)) {
            if kept.contains(&vars(term)) {
                continue;
            }
            if variance != 0.0 {
                return false;
            }
            dropped_any = true;
        }
    }
    dropped_any
}

/// `anova.lm`'s F of two OLS fits from their residual sums of squares:
/// `((RSS₀ − RSS₁)/(p₁ − p₀)) / (RSS₁/(n − p₁))` on `(p₁ − p₀, n − p₁)`, with
/// `p = Model::df − 1` the retained columns and `n` the row count.
fn ols_f(small: &Model, large: &Model) -> LinearTest {
    let (p0, p1) = (small.df as f64 - 1.0, large.df as f64 - 1.0);
    let (num_df, den_df) = (p1 - p0, large.nobs as f64 - p1);
    let f = ((small.rss - large.rss) / num_df) / (large.rss / den_df);
    LinearTest {
        statistic: f,
        kind: LinearKind::F,
        num_df,
        reference: Reference::F {
            num_df,
            den_df,
            den_df_kind: DenDf::Residual,
        },
        p_value: f_upper_p(f, num_df, den_df),
        notes: Vec::new(),
    }
}

/// The Wald F of two dispersion GLMs, all from the larger fit: the tested
/// columns S are its non-aliased columns whose names the smaller fit's
/// non-aliased columns lack, `F = β̂_Sᵀ V_SS⁻¹ β̂_S / r` on `(r, n − p₁)`.
/// `None` when the smaller fit's names, aliased columns dropped, are not
/// among the larger fit's, when no column is left to test, or when a side's
/// arrays disagree in length.
fn wald_f(small: &Model, large: &Model) -> Option<LinearTest> {
    let p = large.coef.len();
    if large.coef_names.len() != p
        || large.aliased.len() != p
        || large.vcov.len() != p
        || large.vcov.iter().any(|row| row.len() != p)
        || small.coef_names.len() != small.aliased.len()
    {
        return None;
    }
    let kept = |m: &'_ Model| -> BTreeSet<String> {
        m.coef_names
            .iter()
            .zip(&m.aliased)
            .filter(|(_, &aliased)| !aliased)
            .map(|(name, _)| name.clone())
            .collect()
    };
    let (small_names, large_names) = (kept(small), kept(large));
    if !small_names.is_subset(&large_names) {
        return None;
    }
    let tested: Vec<usize> = (0..p)
        .filter(|&j| !large.aliased[j] && !small_names.contains(&large.coef_names[j]))
        .collect();
    if tested.is_empty() {
        return None;
    }
    let rows = tested.len();
    let mut l = vec![0.0; rows * p];
    for (i, &j) in tested.iter().enumerate() {
        l[i * p + j] = 1.0;
    }
    let wald = wald_chisq(&large.coef, &large.vcov, &large.aliased, &l, rows);
    let num_df = wald.num_df;
    let den_df = large.nobs as f64 - (large.df as f64 - 1.0);
    let f = wald.statistic / num_df;
    Some(LinearTest {
        statistic: f,
        kind: LinearKind::F,
        num_df,
        reference: Reference::F {
            num_df,
            den_df,
            den_df_kind: DenDf::Residual,
        },
        p_value: f_upper_p(f, num_df, den_df),
        notes: wald.notes,
    })
}

/// Whether two optional per-row vectors are equal, `None` read as `n` copies
/// of `default`.
fn same_or_default(a: &Option<Vec<f64>>, b: &Option<Vec<f64>>, n: usize, default: f64) -> bool {
    let fill = vec![default; n];
    a.as_deref().unwrap_or(&fill) == b.as_deref().unwrap_or(&fill)
}

// ── The per-term tests of one fit ───────────────────────────────────────────

/// Which hypothesis each row of a per-term table tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SsType {
    /// Type II: each term given every term that does not contain it. On the
    /// Gaussian LMM, lmerTest's construction from `XᵀX`; on every other fit,
    /// OLS included, car's, from the fit's covariance of β̂ (the two agree on
    /// OLS without weights, not with prior weights).
    Type2,
    /// Type III, the default: lmerTest's construction, which does not depend
    /// on the contrast coding.
    #[default]
    Type3,
}

/// The options of [`term_tests`]. Start from `TermTestOptions::default()` and
/// set the fields.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermTestOptions {
    /// Which hypotheses the rows test; Type III by default.
    pub ss_type: SsType,
}

/// One fixed term's row of a [`TermTable`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct FixedRow {
    /// The term's label, as R writes it (`"g:h"`).
    pub label: String,
    /// The term's test: an F test, or a Wald χ² ([`term_tests`] says which
    /// fits get which).
    pub test: LinearTest,
    /// lmerTest's `Sum Sq`, `F·σ̂²·num_df` with `σ̂²` the fit's
    /// [`Fit::dispersion`], on an F row; `None` on a Wald χ² row.
    pub sum_sq: Option<f64>,
    /// lmerTest's `Mean Sq`, `F·σ̂²`, on an F row; `None` on a Wald χ² row.
    pub mean_sq: Option<f64>,
}

/// The per-term tests of one fit ([`term_tests`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct TermTable {
    /// Which hypotheses the rows test.
    pub ss_type: SsType,
    /// One row per fixed term, in R's term order; no intercept row and no
    /// random-effect rows.
    pub fixed: Vec<FixedRow>,
    /// The table's notes as plain sentences: lmerTest's missing-cell note on
    /// the Type III table of a design with an empty cell.
    pub notes: Vec<String>,
    /// The fit's coefficient names, one per fixed-effect column, by which a
    /// row note on a column ([`LinearNote::AliasedColumn`]) names it.
    pub coef_names: Vec<String>,
}

/// Why a fit gets no per-term table.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermTestError {
    /// An interaction of the formula lacks one of its marginal terms, so the
    /// rows would test hypotheses that depend on the coding. The message
    /// names each missing term.
    NotMarginal(String),
}

impl fmt::Display for TermTestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermTestError::NotMarginal(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for TermTestError {}

/// The per-term tests of `fit`, the fit of `lowered`: one row per fixed term
/// of [`Lowered::terms`], in that order (R's), each the test of the term's
/// hypothesis set of `opts.ss_type`, read off the one fit with no refit.
///
/// Which fits get which row is read from the fit (the residual or
/// Satterthwaite record it keeps for [`Fit::contrast_f`]), never from
/// `lowered.opts`:
///
/// - OLS, the Gamma and inverse Gaussian GLMs with φ estimated, and the
///   Gaussian LMM (REML or ML) get the F of [`Fit::contrast_f`]: on the
///   residual df `n − p` without a random part ([`DenDf::Residual`]), on
///   Satterthwaite's df with one ([`DenDf::Satterthwaite`]). Where the fit
///   has no df (the pass switched off, missing or failed) the row is NaN
///   with [`LinearNote::NoDf`] and the reason, never a z test.
/// - Every other fit (binomial, Poisson and negative binomial, a dispersion
///   GLM with φ held, every GLMM) gets the Wald χ² of `L β = 0` on
///   [`Fit::vcov`].
///
/// A term with no hypothesis row (its columns aliased) gets a NaN row with
/// [`LinearNote::AliasedColumn`] for each of its aliased columns, in its
/// place in the term order (lmerTest prints such a row last), and a
/// hypothesis set with a non-finite entry (from a covariance that is not
/// finite) a NaN row with no note.
///
/// # Errors
/// [`TermTestError::NotMarginal`] when an interaction of the formula lacks
/// one of its marginal terms (`y ~ x + x:g` lacks `g`).
pub fn term_tests(
    lowered: &Lowered,
    fit: &Fit,
    opts: &TermTestOptions,
) -> Result<TermTable, TermTestError> {
    check_marginal(lowered).map_err(TermTestError::NotMarginal)?;
    let mixed = lowered.model.re.is_some();
    let gaussian = lowered.model.family == Family::Gaussian;
    // F rows on exactly the fits that keep a record for `contrast_f`.
    let f_rows = fit.satterthwaite.is_some();
    let den_df_kind = if mixed {
        DenDf::Satterthwaite
    } else {
        DenDf::Residual
    };
    let retained: Vec<usize> = (0..lowered.p).filter(|&j| !fit.aliased()[j]).collect();
    let hypotheses = match opts.ss_type {
        SsType::Type3 => type3(lowered, &retained),
        SsType::Type2 if gaussian && mixed => type2_xtx(lowered, &retained),
        SsType::Type2 => type2_vcov(lowered, &retained, &fit.vcov),
    };
    let fixed = hypotheses
        .into_iter()
        .zip(&lowered.terms)
        .map(|(h, term)| {
            let test = term_test(fit, &h, term, f_rows, den_df_kind);
            let ms = f_rows.then_some(test.statistic * fit.dispersion);
            FixedRow {
                label: h.term,
                sum_sq: ms.map(|ms| ms * test.num_df),
                mean_sq: ms,
                test,
            }
        })
        .collect();
    let notes = match opts.ss_type {
        SsType::Type3 => missing_cells_note(lowered).into_iter().collect(),
        SsType::Type2 => Vec::new(),
    };
    Ok(TermTable {
        ss_type: opts.ss_type,
        fixed,
        notes,
        coef_names: lowered.col_names.clone(),
    })
}

/// One term's test: the F of [`Fit::contrast_f`] on an F row, labelled
/// `den_df_kind`, else the Wald χ², with the NaN rows of [`term_tests`].
fn term_test(
    fit: &Fit,
    h: &TermHypothesis,
    term: &FixedTerm,
    f_rows: bool,
    den_df_kind: DenDf,
) -> LinearTest {
    let untested = |notes| LinearTest {
        statistic: f64::NAN,
        kind: if f_rows {
            LinearKind::F
        } else {
            LinearKind::ChiSquared
        },
        num_df: f64::NAN,
        reference: if f_rows {
            Reference::F {
                num_df: f64::NAN,
                den_df: f64::NAN,
                den_df_kind,
            }
        } else {
            Reference::ChiSquared { df: 0 }
        },
        p_value: f64::NAN,
        notes,
    };
    let (rows, p) = (h.l.nrows(), h.l.ncols());
    if rows == 0 {
        let aliased = term.columns.clone().filter(|&j| fit.aliased()[j]);
        return untested(
            aliased
                .map(|column| LinearNote::AliasedColumn { column })
                .collect(),
        );
    }
    let l: Vec<f64> = (0..rows)
        .flat_map(|i| (0..p).map(move |j| h.l[(i, j)]))
        .collect();
    if l.iter().any(|v| !v.is_finite()) {
        return untested(Vec::new());
    }
    if !f_rows {
        return wald_chisq(&fit.beta, &fit.vcov, fit.aliased(), &l, rows);
    }
    match fit.contrast_f(&l, rows) {
        Ok(cf) => LinearTest::from_contrast_f(cf, den_df_kind),
        Err(ContrastError::NoDf(reason)) => untested(vec![LinearNote::NoDf { reason }]),
    }
}
