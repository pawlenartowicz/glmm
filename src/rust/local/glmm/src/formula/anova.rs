//! Comparing two fitted models: the formula nesting check, the checks two
//! fits must pass before a comparison, the refusals that name the fix, and
//! the test each pair gets.
//!
//! [`compare`] takes two fitted [`formula::Model`](super::Model)s. It reads
//! off each the plain numbers a port's fit object keeps, never a [`Fit`] or a
//! [`Lowered`], into an `anova::Model`, and compares those with
//! `compare_stored`, so a comparison of two stored fits needs no refit and no
//! data. The ports call `compare_stored` directly; it and `anova::Model` are
//! public with the `orchestrate` feature. The checks run in this order, each a
//! refusal ([`CompareError`]):
//!
//! 1. both fits converged, and each side's `re_groups` lists one grouping per
//!    `varcorr` block, each with one term per block dimension;
//! 2. the same data: the same row count, a bitwise-equal `y`, equal weights
//!    and, on weights that vary, the same power on them (both estimated, or
//!    the same fixed power), equal offsets, the same family and link, and the
//!    same held dispersion;
//! 3. one formula nested in the other ([`nesting`]), and two fits that
//!    estimated their weight power not differing in their random part alone;
//! 4. the estimator: two OLS fits pass whatever their `reml` (a fit that
//!    estimated its weight power is not OLS here); otherwise one REML fit and
//!    one ML fit refuse, and so do two REML fits with different fixed parts;
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
//!   summary of one unweighted `(1 | g)`, and a grouping with at most 100
//!   levels with rows; random parts more than one dimension apart give
//!   χ²(Δnpar), which ignores the boundary.
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
//! one fit: Type III by default, Type II on request. An F row or a Wald χ²
//! row, by family, is read off the fit alone, with no refit; on a fit whose
//! rows are χ² tests, [`TermTest::Lrt`] asks for likelihood-ratio rows
//! instead, each refitting the model under its term's hypothesis, and the
//! default [`TermTest::Auto`] gives them to a Poisson model with random
//! effects. Its table has no random-effect rows: random effects are tested
//! by comparing two models.
//!
//! [`compare`], [`term_tests`], [`nesting`] and their options, results and
//! errors are covered by the crate's semver promise. `anova::Model` and
//! `compare_stored`, what a port's stored fit is compared with, are public
//! only with the `orchestrate` feature, outside it.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use faer::Mat;

use super::hypotheses::{
    check_marginal, missing_cells_note, type2_lr_sets, type2_vcov, type2_xtx, type3,
    type3_null_basis, type3_unrounded, untouched_columns, ReducedDesignError, TermHypothesis,
};
use super::parse::{parse, ParsedFormula, RandomEffect};
use super::{fixed_part_relation, FixedPartRelation, FixedTerm, Lowered, ParseError};
use crate::fit::vech_q;
use crate::inference::contrast::ContrastError;
use crate::inference::lr::{lr_p_value, lr_statistic};
use crate::inference::rlrt::{exact_rlrt, RlrtConfig, RlrtRefusal, RLRT_MAX_LEVELS};
use crate::inference::tails::f_upper_p;
use crate::inference::wald::wald_chisq;
use crate::inference::{
    DenDf, LinearKind, LinearNote, LinearTest, LrNote, LrTest, Objective, Reference,
};
use crate::spec::{remap_slopes, set_term_slopes};
use crate::{ColumnId, Data, Design, Family, Fit, FitOptions, SeFor, StartValues, WaldSe};

mod stored;
#[cfg(not(feature = "orchestrate"))]
pub(crate) use stored::{compare_stored, Model};
#[cfg(feature = "orchestrate")]
pub use stored::{compare_stored, Model};

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
            Side::First => "the first model",
            Side::Second => "the second model",
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
/// the smaller one is in the larger one, each of its random-effect terms
/// inside one term of the larger one's grouping: the same or more slopes, and
/// the intercept only where the larger term has it. So `(x || g)` is nested
/// in `(1 + x | g)`. Terms, groupings and slopes are compared as sets of
/// variable names, so no order matters: not the order of the terms, of the
/// variables inside an interaction (`a:b` is `b:a`), of the groupings, or of
/// the slopes inside a term, and `(1 | a/b)` holds the groupings `a` and
/// `a:b`. A `zerocorr` term counts as one term per variable, and only a term
/// of one variable written the same way, or the intercept alone, lies inside
/// one of those, since the variable may be a factor whose columns `zerocorr`
/// keeps apart. The fixed intercept is a term, present unless the formula
/// removes it with `0 +` or `- 1`: `y ~ 0 + x` is nested in `y ~ x`, and
/// `y ~ 1` against `y ~ 0 + f` is refused although the column spaces are
/// nested when `f` is a factor. The offset is not read.
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

/// One random-effect term as [`nesting`] compares it: whether it holds the
/// intercept, its slopes (each a set of variable names), and whether it is one
/// variable of a `zerocorr` term.
struct RandomTerm<'a> {
    intercept: bool,
    slopes: BTreeSet<BTreeSet<&'a str>>,
    diagonal: bool,
}

impl RandomTerm<'_> {
    /// Whether this term's model lies inside `large`'s: its columns among
    /// `large`'s, and, against a `zerocorr` variable, nothing but that
    /// variable spelled the same way or the intercept alone.
    fn inside(&self, large: &RandomTerm<'_>) -> bool {
        (!self.intercept || large.intercept)
            && self.slopes.is_subset(&large.slopes)
            && (!large.diagonal || self.diagonal || self.slopes.is_empty())
    }
}

/// Each grouping (its variable names) with its random-effect terms.
fn random_part(f: &ParsedFormula) -> BTreeMap<BTreeSet<&str>, Vec<RandomTerm<'_>>> {
    let mut part: BTreeMap<BTreeSet<&str>, Vec<RandomTerm<'_>>> = BTreeMap::new();
    fn slopes(vs: &[String]) -> BTreeSet<BTreeSet<&str>> {
        vs.iter().map(|s| vars(s)).collect()
    }
    for re in &f.random_effects {
        let (group, terms) = match re {
            RandomEffect::Intercept { group, .. } => (
                group,
                vec![RandomTerm {
                    intercept: true,
                    slopes: BTreeSet::new(),
                    diagonal: false,
                }],
            ),
            RandomEffect::Slope { group, vars: vs }
            | RandomEffect::NoIntercept { group, vars: vs } => (
                group,
                vec![RandomTerm {
                    intercept: matches!(re, RandomEffect::Slope { .. }),
                    slopes: slopes(vs),
                    diagonal: false,
                }],
            ),
            RandomEffect::ZeroCorr {
                group,
                vars: vs,
                intercept,
            } => {
                let intercept_term = intercept.then(|| RandomTerm {
                    intercept: true,
                    slopes: BTreeSet::new(),
                    diagonal: false,
                });
                let per_var = vs.iter().map(|v| RandomTerm {
                    intercept: false,
                    slopes: BTreeSet::from([vars(v)]),
                    diagonal: true,
                });
                (group, intercept_term.into_iter().chain(per_var).collect())
            }
        };
        part.entry(vars(group)).or_default().extend(terms);
    }
    part
}

/// Whether every grouping of `small` is in `large`, each of its terms inside
/// one term of `large`'s grouping.
fn random_inside(
    small: &BTreeMap<BTreeSet<&str>, Vec<RandomTerm<'_>>>,
    large: &BTreeMap<BTreeSet<&str>, Vec<RandomTerm<'_>>>,
) -> bool {
    small.iter().all(|(group, terms)| {
        large
            .get(group)
            .is_some_and(|l| terms.iter().all(|t| l.iter().any(|lt| t.inside(lt))))
    })
}

/// The diagonal of a vech-packed covariance block (the stored layout behind
/// [`Fit::varcorr()`]): the variances.
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
    /// The parameter count ([`Fit::df`]).
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
    /// The coefficient names, one per fixed-effect column
    /// ([`Lowered::col_names`]).
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
    /// under the table. A note on a fit names it as the smaller or the larger
    /// model, and a note on a column of the larger fit names its coefficient.
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
    let line = match note {
        LrNote::NegativeStatistic { raw } => format!(
            "The larger model did not reach its optimum: its log-likelihood is below the smaller \
             model's by more than round-off, 2(l1 - l0) = {raw:e}. The statistic is set to 0."
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
        LrNote::NoDesignSummary => "The exact restricted likelihood-ratio test needs a summary \
             of the larger model's design. A fit keeps one only when it is a Gaussian REML fit \
             with at least one fixed-effect column and one unweighted (1 | g) as its only random \
             term (weights all equal to 1 count as weights). This fit keeps none, so the 50:50 \
             mixture is used."
            .to_string(),
        LrNote::ExactRlrtRefused { reason } => format!(
            "The exact restricted likelihood-ratio test was not computed: {reason}. The 50:50 \
             mixture is used."
        ),
        LrNote::NodeCounts { m0, m1 } => format!(
            "The two fits use different quadrature node counts, {m0} and {m1}; the \
             likelihood-ratio test mixes two approximation errors."
        ),
        LrNote::EstimatedWeightPower => "Both models estimated the power on the weights. When \
             the noise is not a power of the weights, this test rejects a true null somewhat \
             more often than its level: about 6 to 9% at the 5% level in simulations."
            .to_string(),
        LrNote::NoWaldF => "The Wald F cannot be formed from the two fits' coefficients: it \
             needs the smaller fit's coefficients, aliased columns dropped, among the larger \
             fit's, at least one more left to test, and each fit's names, coefficients, \
             covariance and aliased flags of one length. The likelihood-ratio test is used, with \
             an asymptotic chi-squared reference."
            .to_string(),
        LrNote::NonFiniteLoglik => {
            let (smaller, larger) = match c.larger {
                Side::First => (&c.m1, &c.m0),
                Side::Second => (&c.m0, &c.m1),
            };
            let who = match (smaller.loglik.is_finite(), larger.loglik.is_finite()) {
                (false, false) => "both models",
                (false, true) => "the smaller model",
                _ => "the larger model",
            };
            format!(
                "The log-likelihood of {who} is not finite (a fit that failed or collapsed), so \
                 the statistic and its p-value are NaN."
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
        LinearNote::RankDeficientL { rank: 0, rows } => format!(
            "The {rows} tested coefficients span no direction of their covariance, so there is \
             nothing to test."
        ),
        LinearNote::RankDeficientL { rank, rows } => format!(
            "The {rows} tested coefficients span {rank} directions of their covariance; the test \
             is on those {rank}."
        ),
        LinearNote::NoDf { reason } => {
            format!("No denominator df ({reason}), so the hypothesis was not tested.")
        }
        LinearNote::ReducedNotConverged => "A model refitted for this term's likelihood-ratio \
             test did not converge, or its log-likelihood is not finite, so the statistic and its \
             p-value are NaN."
            .into(),
        LinearNote::NegativeStatistic => "The model refitted under the term's hypothesis \
             reached a higher log-likelihood than the larger model of this row, by more than \
             round-off: the larger model did not reach its optimum. The statistic is set to 0."
            .into(),
        LinearNote::RobustUnidentified => "One group alone carries the information on this \
             hypothesis, so its robust standard error and df are not computed (NaN). More groups \
             are needed; the model-based standard error assumes the groups share one variance."
            .into(),
        LinearNote::RobustDfSmall => "The robust degrees of freedom are below 4, where the \
             robust test gives no assurance on its error rate. If the predictor is constant \
             within groups, the model-based standard error is the better route."
            .into(),
        LinearNote::ModelBasedJointTest => "A test of more than one df uses the model-based \
             standard errors, not the robust ones."
            .into(),
        LinearNote::LargePoissonCounts => "Counts this large make the likelihood sharp: the \
             model refitted for this term may have stopped in a local minimum, which overstates \
             the statistic. To check, refit that model from a large random-intercept standard \
             deviation."
            .into(),
    }
}

/// The table note of a likelihood-ratio table on a fit that did not
/// converge.
pub(super) fn lrt_not_converged_note() -> String {
    "No model was refitted, so every likelihood-ratio statistic and p-value is NaN.".into()
}

/// The table note of a likelihood-ratio row whose reduced design could not
/// be built: the null basis of the term's hypothesis failed its check.
pub(super) fn lrt_no_basis_note(term: &str) -> String {
    format!(
        "The model under the hypothesis of {term} could not be built (the hypothesis is too \
         ill-conditioned), so its statistic and its p-value are NaN."
    )
}

/// Why two models cannot be compared. Each message says what is wrong and
/// what to do in plain words; the first model and the second model are the
/// two sides as given. The messages name no call syntax but `weight_power`,
/// spelled as the fit's refusals spell it: a port that names its own
/// comparison function or REML argument renders that from the variant's
/// fields.
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
    /// A side's `re_groups` does not describe its random part: it does not
    /// list one grouping per `varcorr` block, each with one term per
    /// dimension of its block.
    ReGroups {
        /// The side whose groupings do not match its variance blocks.
        side: Side,
    },
    /// The two fits put different powers on the weights, or both estimated
    /// theirs and differ only in their random part.
    WeightPowerMismatch(WeightPowerMismatch),
}

/// How the weight powers of two fits ([`crate::Fit::weight_power`]) keep them
/// from a comparison ([`CompareError::WeightPowerMismatch`]). The powers are
/// read only when the weights vary: without weights, or with all weights
/// equal, a power changes nothing.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WeightPowerMismatch {
    /// One fit estimated its power ([`crate::WeightPower::Estimate`]) and the
    /// other held it fixed: two variance models.
    EstimatedVsFixed,
    /// Two different fixed powers.
    DifferentFixed {
        /// `m0`'s power.
        m0: f64,
        /// `m1`'s power.
        m1: f64,
    },
    /// Two fits that each estimated their power, with the same fixed part:
    /// the test of a random part (the exact restricted likelihood-ratio test
    /// among them) needs working weights the two sides share.
    DeltaRandomPartOnly,
}

impl fmt::Display for CompareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompareError::NotConverged { m0, m1 } => {
                let who = match (m0, m1) {
                    (true, true) => "neither model converged, so their log-likelihoods are",
                    (true, false) => "the first model did not converge, so its log-likelihood is",
                    _ => "the second model did not converge, so its log-likelihood is",
                };
                write!(
                    f,
                    "{who} not at a maximum and cannot be compared. Change the model until it \
                     converges (a simpler random part often helps), then compare again."
                )
            }
            CompareError::DifferentRows { m0, m1 } => write!(
                f,
                "the first model has {m0} rows and the second has {m1}; a comparison needs \
                 both fits on the same rows. Fit both models to the same rows, dropping the \
                 rows with a missing value in any variable of either formula first."
            ),
            CompareError::DifferentY => write!(
                f,
                "the two models have different responses y (the same row count, different \
                 values); a comparison needs both fits on the same data. Fit both models to the \
                 same data."
            ),
            CompareError::DifferentWeights => write!(
                f,
                "the two models were fitted with different weights; a comparison needs the \
                 same weights in both. Fit both models with the same weights."
            ),
            CompareError::DifferentOffset => write!(
                f,
                "the two models were fitted with different offsets; a comparison needs the \
                 same offset in both. Fit both models with the same offset."
            ),
            CompareError::DifferentFamily { m0, m1 } => write!(
                f,
                "the first model was fitted as {} and the second as {}; a comparison needs \
                 the same family and link in both. Fit both models with the same family and \
                 link.",
                m0.words(),
                m1.words()
            ),
            CompareError::DifferentDispersionHeld { m0, m1 } => {
                let held = |d: &Option<f64>| match d {
                    Some(v) => format!("holds the dispersion at {v}"),
                    None => "does not hold the dispersion".to_string(),
                };
                write!(
                    f,
                    "the first model {} and the second {}; a comparison needs the same held \
                     dispersion in both. Fit both models with the same held dispersion.",
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
                    "the first model was fitted by {m0} and the second by {m1}; a comparison \
                     needs both fitted the same way. Fit both models by maximum likelihood \
                     (ML), or both by REML when their fixed parts are the same."
                )
            }
            CompareError::RemlDifferentFixed => write!(
                f,
                "the two models are REML fits with different fixed effects; the REML \
                 likelihood depends on the fixed effects, so a comparison needs ML fits. Fit \
                 both models by maximum likelihood (ML)."
            ),
            CompareError::SameParts => write!(
                f,
                "the two models have the same fixed and random parts, so there is nothing to \
                 test. Compare two models that differ by at least one parameter."
            ),
            CompareError::NothingToTest { m0, .. } => write!(
                f,
                "the two models have the same number of parameters ({m0}), so there is \
                 nothing to test: the larger formula adds no parameter to the smaller one. \
                 Compare two models that differ by at least one parameter."
            ),
            CompareError::ReGroups { side } => write!(
                f,
                "{}'s random-effect groupings do not match its variance blocks: a comparison \
                 needs one grouping per block, with one term per dimension of the block. Copy \
                 them from the lowered model the fit was made on, then compare again.",
                side.label()
            ),
            CompareError::WeightPowerMismatch(WeightPowerMismatch::EstimatedVsFixed) => write!(
                f,
                "one model estimated its weight power and the other used a fixed power, so \
                 the two have different variance models; a comparison needs the same weight \
                 power in both. Fit both models with the same weight_power."
            ),
            CompareError::WeightPowerMismatch(WeightPowerMismatch::DifferentFixed { m0, m1 }) => {
                write!(
                    f,
                    "the first model was fitted with weight_power = {m0} and the second with \
                     weight_power = {m1}, so the two have different variance models; a \
                     comparison needs the same weight power in both. Fit both models with the \
                     same weight_power."
                )
            }
            CompareError::WeightPowerMismatch(WeightPowerMismatch::DeltaRandomPartOnly) => write!(
                f,
                "both models estimated their weight power and differ only in their random \
                 part; a test of the random part needs the same working weights in both, \
                 which two estimated powers do not give. If the weights are precisions you \
                 trust, fit both models with weight_power = 1 and compare again. Weights taken \
                 as given when they are not precisions gave tests that rejected a true null \
                 15.5% of the time at n = 100 (nominal 5%) in the simulations behind the \
                 weight power."
            ),
        }
    }
}

impl std::error::Error for CompareError {}

/// Compares two fitted models, given in either order: `compare_stored` on
/// the numbers of each.
///
/// # Errors
/// The first check that fails, as a [`CompareError`].
pub fn compare(
    a: &super::Model,
    b: &super::Model,
    opts: &CompareOptions,
) -> Result<Comparison, CompareError> {
    let stored = |m: &super::Model| Model::from_fit(m.formula(), m.lowered(), &m.opts, m.fit());
    compare_stored(&stored(a), &stored(b), opts)
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
    if let (Some(k0), Some(k1)) = (quadrature_nodes(a.objective), quadrature_nodes(b.objective)) {
        if k0 != k1 {
            notes.push(LrNote::NodeCounts { m0: k0, m1: k1 });
        }
    }
    if a.weight_power_estimated && b.weight_power_estimated {
        notes.push(LrNote::EstimatedWeightPower);
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
            | LrNote::EstimatedWeightPower
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
                        // The level cap is the comparison's, so the exact
                        // test's functions stay callable at any `K`.
                        let with_rows = counts.iter().filter(|&&n| n > 0.0).count();
                        let exact = if with_rows > RLRT_MAX_LEVELS {
                            Err(RlrtRefusal::TooManyLevels)
                        } else {
                            let cfg = RlrtConfig {
                                nsim: opts.nsim,
                                seed: opts.seed,
                                parallel_inner: false,
                            };
                            let pinned =
                                large.varcorr.first().and_then(|b| b.first()) == Some(&0.0);
                            exact_rlrt(counts, s, c, large.loglik, small.loglik, pinned, &cfg)
                        };
                        match exact {
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
    // Defensive: `compare` has already refused groupings that do not match
    // the variance blocks (`CompareError::ReGroups`), here and below.
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

/// Which test the rows of a per-term table run on a fit whose rows are χ²
/// tests: binomial, Poisson and negative binomial fits, a dispersion GLM
/// with φ held, every GLMM. A fit whose rows are F tests (OLS, the Gaussian
/// LMM, a Gamma or inverse Gaussian GLM with φ estimated) has only its F
/// rows: it takes `Auto` and refuses the other two
/// ([`TermTestError::FTable`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TermTest {
    /// The default: the F rows on a fit that has them, the likelihood-ratio
    /// χ² on a Poisson model with random effects (whatever its number of
    /// clusters, groupings or term df), and the Wald χ² on every other
    /// fit. [`TermTable::choice`] says which ran and why ([`ChoiceReason`]).
    #[default]
    Auto,
    /// The Wald χ² of each term's hypothesis, on the fit's covariance of β̂.
    Wald,
    /// The likelihood-ratio χ² of each term's hypothesis, from a refit of
    /// the model under it.
    Lrt,
}

/// The options of [`term_tests`]. Start from `TermTestOptions::default()` and
/// set the fields.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TermTestOptions {
    /// Which hypotheses the rows test; Type III by default.
    pub ss_type: SsType,
    /// Which test the rows run on a fit whose rows are χ² tests;
    /// [`TermTest::Auto`] by default.
    pub test: TermTest,
}

/// The test a per-term table ran, and why ([`TermTable::choice`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TestChoice {
    /// The rows' test: [`LinearKind::F`], the Wald χ²
    /// ([`LinearKind::ChiSquared`]) or the likelihood-ratio χ²
    /// ([`LinearKind::LikelihoodRatio`]).
    pub test: LinearKind,
    /// Why the table runs `test`.
    pub reason: ChoiceReason,
}

/// Why a per-term table runs its test ([`TestChoice`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChoiceReason {
    /// The options asked for it ([`TermTest::Wald`] or [`TermTest::Lrt`]).
    Requested,
    /// [`TermTest::Auto`] on a fit whose rows are F tests: the F is the only
    /// test its table has.
    FTest,
    /// [`TermTest::Auto`] on a Poisson model with random effects: the
    /// likelihood-ratio χ². In the Monte Carlo calibration of the per-term
    /// tests (both tests on the same simulated datasets), the Wald χ² of a
    /// Poisson GLMM was never closer to the nominal level than the
    /// likelihood-ratio χ², and rejected a between-cluster term too often at
    /// few clusters.
    PoissonMixed,
    /// [`TermTest::Auto`] on a binomial model with random effects (Bernoulli
    /// or with trials): the Wald χ². The same calibration found neither test
    /// closer to the nominal level across binomial GLMM designs.
    BinomialMixed,
    /// [`TermTest::Auto`] on a negative binomial model with random effects:
    /// the Wald χ².
    NegativeBinomialMixed,
    /// [`TermTest::Auto`] on a model with random effects of any other family
    /// whose rows are χ² tests (a Gamma GLMM): the Wald χ².
    OtherMixed,
    /// [`TermTest::Auto`] on a model with no random effects whose rows are χ²
    /// tests: the Wald χ².
    NoRandomEffects,
}

/// One fixed term's row of a [`TermTable`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct FixedRow {
    /// The term's label, as R writes it (`"g:h"`).
    pub label: String,
    /// The term's test: an F test, a Wald χ² or a likelihood-ratio χ²
    /// ([`term_tests`] says which fits get which).
    pub test: LinearTest,
    /// lmerTest's `Sum Sq`, `F·σ̂²·num_df` with `σ̂²` the fit's
    /// [`Fit::dispersion`], on an F row; `None` on a χ² row.
    pub sum_sq: Option<f64>,
    /// lmerTest's `Mean Sq`, `F·σ̂²`, on an F row; `None` on a χ² row.
    pub mean_sq: Option<f64>,
}

impl FixedRow {
    /// The row's partial η², `F·df1 / (F·df1 + df2)` from its own F and
    /// df, as `effectsize::F_to_eta2` computes it from an `anova()` table.
    /// On a Gaussian LMM `df2` is Satterthwaite's, so the value is
    /// approximate. NaN on a χ² or likelihood-ratio row and on an F row that
    /// was not tested (its F and df NaN).
    pub fn eta2_partial(&self) -> f64 {
        match self.test.reference {
            Reference::F { num_df, den_df, .. } => {
                let f_df1 = self.test.statistic * num_df;
                f_df1 / (f_df1 + den_df)
            }
            _ => f64::NAN,
        }
    }
}

/// A [`FixedRow`] of `test`, with `Sum Sq` and `Mean Sq` from `dispersion` on
/// an F row.
pub(crate) fn fixed_row(label: String, test: LinearTest, dispersion: f64) -> FixedRow {
    let ms = (test.kind == LinearKind::F).then_some(test.statistic * dispersion);
    FixedRow {
        label,
        sum_sq: ms.map(|ms| ms * test.num_df),
        mean_sq: ms,
        test,
    }
}

/// The per-term tests of one fit ([`term_tests`]).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct TermTable {
    /// Which hypotheses the rows test.
    pub ss_type: SsType,
    /// The test the rows run and why ([`TermTestOptions::test`]).
    pub choice: TestChoice,
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
    /// [`TermTest::Wald`] or [`TermTest::Lrt`] asked of a fit whose rows are
    /// F tests, whose table has only its F rows. The message names the F
    /// test.
    FTable {
        /// The test asked for.
        asked: TermTest,
        /// Where the F rows' denominator df come from.
        den_df: DenDf,
    },
    /// Likelihood-ratio rows asked of a fit made at another quadrature node
    /// count than the options' `nagq`, the count the rows' refits use, so
    /// each row would compare two approximations of the likelihood.
    RefitNodeCount {
        /// The fit's node count (1, the Laplace approximation).
        fit: u8,
        /// The node count of the refits, the options' `nagq`.
        refits: u8,
    },
}

impl fmt::Display for TermTestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TermTestError::NotMarginal(message) => f.write_str(message),
            TermTestError::FTable { asked, den_df } => {
                let default = ", so leave the test at its default";
                let (asked, advice) = match asked {
                    TermTest::Wald => ("Wald chi-squared", default),
                    TermTest::Lrt => ("likelihood-ratio", default),
                    TermTest::Auto => ("chi-squared", ""),
                };
                let fit = match den_df {
                    DenDf::Satterthwaite | DenDf::PustejovskyTipton => {
                        "a Gaussian mixed model are F tests"
                    }
                    DenDf::Residual | DenDf::BellMcCaffrey => {
                        "this fit are F tests on the residual df"
                    }
                };
                write!(
                    f,
                    "per-term tests on {fit}; {asked} rows are not available{advice}."
                )
            }
            TermTestError::RefitNodeCount { fit, refits } => write!(
                f,
                "the fit's quadrature node count is {fit} and the refits of its \
                 likelihood-ratio rows would use {refits} (a count of 1 is the Laplace \
                 approximation), so each row would compare two approximations of the \
                 likelihood. Fit the model with a node count of {refits}, or ask for the Wald \
                 test."
            ),
        }
    }
}

impl std::error::Error for TermTestError {}

/// The quadrature node count of a GLMM objective (1, the Laplace
/// approximation); `None` for every other objective.
fn quadrature_nodes(objective: Objective) -> Option<u8> {
    match objective {
        Objective::Laplace => Some(1),
        Objective::Agq { nodes } => Some(nodes),
        _ => None,
    }
}

/// The test [`TermTest::Auto`] runs on `fit`, the fit of `lowered`, and why:
/// one choice for the whole table, from whether the fit's rows are F tests,
/// its family and whether it has random effects, never from its size.
pub(crate) fn auto_choice(lowered: &Lowered, fit: &Fit) -> TestChoice {
    // F rows on exactly the fits that keep a record for `contrast_f`, as in
    // `term_tests_cached`.
    let (test, reason) = if fit.satterthwaite.is_some() {
        (LinearKind::F, ChoiceReason::FTest)
    } else if lowered.model.re.is_none() {
        (LinearKind::ChiSquared, ChoiceReason::NoRandomEffects)
    } else {
        match lowered.model.family {
            Family::Poisson { .. } => (LinearKind::LikelihoodRatio, ChoiceReason::PoissonMixed),
            Family::Binomial { .. } => (LinearKind::ChiSquared, ChoiceReason::BinomialMixed),
            Family::NegativeBinomial { .. } => {
                (LinearKind::ChiSquared, ChoiceReason::NegativeBinomialMixed)
            }
            _ => (LinearKind::ChiSquared, ChoiceReason::OtherMixed),
        }
    };
    TestChoice { test, reason }
}

/// The per-term tests of `model`'s fit: one row per fixed term of
/// [`Lowered::terms`], in that order (R's), each the test of the term's hypothesis set of
/// `opts.ss_type`, read off the one fit (a likelihood-ratio row also refits
/// the model under its hypothesis).
///
/// Which fits get which row is read from the fit (the residual or
/// Satterthwaite record it keeps for [`Fit::contrast_f`]), never from the
/// options:
///
/// - OLS, the Gamma and inverse Gaussian GLMs with φ estimated, and the
///   Gaussian LMM (REML or ML) get the F of [`Fit::contrast_f`]: on the
///   residual df `n − p` without a random part ([`DenDf::Residual`]), on
///   Satterthwaite's df with one ([`DenDf::Satterthwaite`]). Where the fit
///   has no df (the pass switched off, missing or failed) the row is NaN
///   with [`LinearNote::NoDf`] and the reason, never a z test.
/// - Every other fit (binomial, Poisson and negative binomial, a dispersion
///   GLM with φ held, every GLMM) gets the Wald χ² of `L β = 0` on
///   [`Fit::vcov`] under [`TermTest::Wald`], and the likelihood-ratio χ²
///   under [`TermTest::Lrt`]. Under [`TermTest::Auto`] (the default) a
///   Poisson model with random effects gets the likelihood-ratio χ² and
///   every other fit the Wald χ² ([`ChoiceReason`]).
///
/// [`TermTable::choice`] records the test run and why. A fit with F rows
/// takes only [`TermTest::Auto`].
///
/// A likelihood-ratio row refits the model under the term's hypothesis, as
/// [`crate::fit_with_start`] does, from the fit's β̂, θ̂ and estimated
/// dispersion, by the options the model was fitted with (node count, held
/// dispersion; a negative binomial θ is re-estimated) and its lowering's
/// weights and offset. The Wald and F rows read the fit alone. Type III fits the reduced design
/// `X·N`, `N` an orthonormal basis of the null space of the term's `L` (the
/// hypothesis its Wald row tests); Type II, car's `Anova(type = 2,
/// test.statistic = "LR")`, fits two column subsets of the design, the model
/// of every term that does not contain the term with and without it, each
/// distinct subset once. The statistic `2(ℓ₁ − ℓ₀)`, floored at 0, is
/// referred to χ² on the term's rank; a raw value below zero by more than
/// round-off takes [`LinearNote::NegativeStatistic`]. A refit that does not
/// converge gives a NaN row with [`LinearNote::ReducedNotConverged`]; on a
/// fit that did not converge every row is NaN, nothing is refitted, and a
/// table note says why. The reduced model keeps every random slope: a slope
/// on a column the reduced design leaves out or mixes reads that column's
/// values as a random-effect-only column ([`crate::Data::re_columns`]).
///
/// On very large Poisson counts such a refit can stop in the wrong one of two
/// basins. The model has to carry the dropped column's mean on its random
/// effects, either on that column's own random slope or, with a large
/// random-intercept standard deviation, on the intercept's random effect, and
/// both are local minima. A refit that stops in the higher one reports a
/// converged fit above the reduced model's optimum, and the row's statistic
/// is too large. On one design, `y ~ 1 + x1 + x2 + (1 + x1 + x2 | g)` with 20
/// groups of 10 rows and log-link counts at a log mean of 6, 10, 14 and 18,
/// the reduced refits that converged more than 0.1 deviance units above the
/// lowest point found from several starts were 2 of 600, 6 of 600, 28 of 600
/// and 31 of 80. The likelihood-ratio rows of a Poisson mixed model whose
/// fitted counts average above e⁹ (about 8100) carry
/// [`LinearNote::LargePoissonCounts`]. That threshold lies between the sets
/// at log mean 6 and 10 and was not measured: smaller counts are not free of
/// such refits, only rarer.
///
/// A term with no hypothesis row (its columns aliased) gets a NaN row with
/// [`LinearNote::AliasedColumn`] for each of its aliased columns, in its
/// place in the term order (lmerTest prints such a row last), and a
/// hypothesis set with a non-finite entry (from a covariance that is not
/// finite) a NaN row with no note.
///
/// # Errors
/// [`TermTestError::NotMarginal`] when an interaction of the formula lacks
/// one of its marginal terms (`y ~ x + x:g` lacks `g`),
/// [`TermTestError::FTable`] when `opts.test` is [`TermTest::Wald`] or
/// [`TermTest::Lrt`] on a fit with F rows, and
/// [`TermTestError::RefitNodeCount`] when the rows are likelihood-ratio rows
/// ([`TermTest::Lrt`], or [`TermTest::Auto`] on a Poisson model with random
/// effects) and the fit's quadrature node count is not that of the options
/// the model was fitted with.
pub fn term_tests(
    model: &super::Model,
    opts: &TermTestOptions,
) -> Result<TermTable, TermTestError> {
    term_tests_cached(
        model.lowered(),
        &model.opts,
        model.fit(),
        opts,
        &mut RefitCache::default(),
    )
}

/// Refits of column subsets of one fit's design, each keyed by its subset
/// (caller columns of [`Lowered::x`], ascending), so the likelihood-ratio
/// tables of one fit fit each subset once between them
/// ([`term_tests_cached`]). A cache belongs to the `(lowered, fit)` pair it
/// was filled for, with its options.
#[derive(Default)]
pub(crate) struct RefitCache(Vec<(Vec<usize>, Fit)>);

/// [`term_tests`] on `fit`, the fit of `lowered` made with `fit_opts`, with
/// the likelihood-ratio rows' refits of column subsets kept in `cache`. The
/// likelihood-ratio rows need `fit` made with `fit_opts` on
/// `lowered.data()`: a fit made at another node count is refused
/// ([`TermTestError::RefitNodeCount`]), and weights, an offset or a held
/// dispersion cannot be read back from a [`Fit`], so a caller that fits with
/// other ones gets rows that compare two different likelihoods.
///
/// A caller that builds both types' likelihood-ratio tables of one fit
/// passes one cache to both, and a Type III hypothesis that drops columns
/// reads the Type II design with the same columns, built and fitted the same
/// way. The rows are those of [`term_tests`] bit for bit.
pub(crate) fn term_tests_cached(
    lowered: &Lowered,
    fit_opts: &FitOptions,
    fit: &Fit,
    opts: &TermTestOptions,
    cache: &mut RefitCache,
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
    let choice = match (opts.test, f_rows) {
        (TermTest::Auto, _) => auto_choice(lowered, fit),
        (asked, true) => {
            return Err(TermTestError::FTable {
                asked,
                den_df: den_df_kind,
            })
        }
        (TermTest::Wald, false) => TestChoice {
            test: LinearKind::ChiSquared,
            reason: ChoiceReason::Requested,
        },
        (TermTest::Lrt, false) => TestChoice {
            test: LinearKind::LikelihoodRatio,
            reason: ChoiceReason::Requested,
        },
    };
    // The likelihood-ratio rows refit by `fit_opts`: a fit made at another
    // node count would be compared with refits of another approximation.
    if choice.test == LinearKind::LikelihoodRatio {
        if let Some(nodes) = quadrature_nodes(fit.objective) {
            if nodes != fit_opts.nagq {
                return Err(TermTestError::RefitNodeCount {
                    fit: nodes,
                    refits: fit_opts.nagq,
                });
            }
        }
    }
    let retained: Vec<usize> = (0..lowered.p).filter(|&j| !fit.aliased()[j]).collect();
    let hypotheses = match opts.ss_type {
        SsType::Type3 => type3(lowered, &retained),
        SsType::Type2 if gaussian && mixed => type2_xtx(lowered, &retained),
        // The model covariance, which on a robust fit is not `vcov`.
        SsType::Type2 => type2_vcov(lowered, &retained, &fit.model_vcov_rows()),
    };
    let mut notes: Vec<String> = match opts.ss_type {
        SsType::Type3 => missing_cells_note(lowered).into_iter().collect(),
        SsType::Type2 => Vec::new(),
    };
    let tests: Vec<LinearTest> = if choice.test == LinearKind::LikelihoodRatio {
        let (mut tests, lrt_notes) = lrt_rows(
            lowered,
            fit_opts,
            fit,
            &retained,
            opts.ss_type,
            &hypotheses,
            cache,
        );
        notes.extend(lrt_notes);
        if large_poisson_counts(lowered, fit) {
            for t in tests.iter_mut().filter(|t| t.statistic.is_finite()) {
                t.notes.push(LinearNote::LargePoissonCounts);
            }
        }
        tests
    } else {
        hypotheses
            .iter()
            .zip(&lowered.terms)
            .map(|(h, term)| term_test(fit, h, term, f_rows, den_df_kind))
            .collect()
    };
    let fixed = hypotheses
        .into_iter()
        .zip(tests)
        .map(|(h, test)| fixed_row(h.term, test, fit.dispersion))
        .collect();
    Ok(TermTable {
        ss_type: opts.ss_type,
        choice,
        fixed,
        notes,
        coef_names: lowered.col_names.clone(),
    })
}

/// The log of the mean fitted count above which a Poisson mixed model's
/// likelihood-ratio rows carry [`LinearNote::LargePoissonCounts`]: e⁹, about
/// 8100, between log mean 6 (2 of 600 reduced refits in the wrong basin)
/// and 10 (6 of 600), the rates [`term_tests`] gives. Not itself measured.
pub(super) const LARGE_POISSON_LOG_MEAN: f64 = 9.0;

/// Whether `fit` is a Poisson mixed model, any link, whose fitted counts
/// average above `exp(LARGE_POISSON_LOG_MEAN)`. False on a fit with no
/// fitted values (one that did not converge).
fn large_poisson_counts(lowered: &Lowered, fit: &Fit) -> bool {
    let mu = fit.fitted();
    lowered.model.re.is_some()
        && matches!(lowered.model.family, Family::Poisson { .. })
        && !mu.is_empty()
        && (mu.iter().sum::<f64>() / mu.len() as f64).ln() > LARGE_POISSON_LOG_MEAN
}

/// The likelihood-ratio rows of [`term_tests`], one per hypothesis of
/// `hypotheses` (the Wald rows' sets of `ss_type`, on `retained`), and the
/// table notes they bring.
///
/// Type III refits the model on `X·N`, `N` the null basis of the term's
/// hypothesis ([`type3_null_basis`], from [`type3`]'s rounded `L` and
/// [`type3_unrounded`]'s), or, when `N` is all unit columns, on the column
/// subset it keeps; Type II fits the column subsets of [`type2_lr_sets`],
/// and reads the column sets only, never the fit's covariance. Each distinct
/// subset is fitted once per `cache` ([`fit_subset`]), the retained design
/// itself being `fit`.
/// A term with no column left to test gets the NaN row of the Wald table with
/// its aliased-column notes; a Type III hypothesis with a non-finite entry, a
/// NaN row; a basis that fails its check, a NaN row and the table note
/// [`lrt_no_basis_note`]. Every refit keeps the model's random slopes
/// ([`remap_all`]). A fit that did not converge gets NaN rows and the table
/// note [`lrt_not_converged_note`], and nothing is refitted.
fn lrt_rows(
    lowered: &Lowered,
    fit_opts: &FitOptions,
    fit: &Fit,
    retained: &[usize],
    ss_type: SsType,
    hypotheses: &[TermHypothesis],
    cache: &mut RefitCache,
) -> (Vec<LinearTest>, Vec<String>) {
    let wald = |t: usize| {
        term_test(
            fit,
            &hypotheses[t],
            &lowered.terms[t],
            false,
            DenDf::Residual,
        )
    };
    // A term with no column left to test: the Wald table's NaN row (its
    // hypothesis has no row, so no covariance is read), with its notes.
    let aliased = |t: usize| {
        let mut row = wald(t);
        row.kind = LinearKind::LikelihoodRatio;
        row
    };
    if !fit.converged() {
        let rows = (0..hypotheses.len())
            .map(|t| {
                if hypotheses[t].l.nrows() == 0 {
                    aliased(t)
                } else {
                    nan_lrt_row()
                }
            })
            .collect();
        return (rows, vec![lrt_not_converged_note()]);
    }
    match ss_type {
        SsType::Type3 => {
            let slopes = slope_lists(lowered);
            let exact = type3_unrounded(lowered, retained);
            let restrict =
                |m: &Mat<f64>| Mat::from_fn(m.nrows(), retained.len(), |i, a| m[(i, retained[a])]);
            let mut notes = Vec::new();
            let rows = (0..hypotheses.len())
                .map(|t| {
                    let h = &hypotheses[t];
                    if h.l.nrows() == 0 {
                        return aliased(t);
                    }
                    if h.l.col_iter().any(|c| c.iter().any(|v| !v.is_finite())) {
                        return nan_lrt_row();
                    }
                    let l = restrict(&h.l);
                    let basis = match type3_null_basis(l.as_ref(), restrict(&exact[t].l).as_ref()) {
                        Ok(basis) => basis,
                        Err(ReducedDesignError::NoNullBasis) => {
                            notes.push(lrt_no_basis_note(&h.term));
                            return nan_lrt_row();
                        }
                    };
                    let k = basis.ncols();
                    // The caller columns of `N`'s leading unit columns.
                    let kept: Vec<usize> = untouched_columns(l.as_ref())
                        .iter()
                        .map(|&a| retained[a])
                        .collect();
                    if k == kept.len() {
                        // `N` is the unit columns of `kept`: a column drop,
                        // fitted as the Type II subset of the same columns.
                        fit_subset(lowered, fit_opts, fit, retained, &kept, cache);
                        return lrt_test(
                            fit,
                            fitted(&kept, retained, fit, cache),
                            retained.len() - k,
                        );
                    }
                    let moved = remap_all(&slopes, &kept, k);
                    let reduced = refit_on_basis(lowered, fit_opts, fit, retained, &basis, &moved);
                    lrt_test(fit, &reduced, retained.len() - k)
                })
                .collect();
            (rows, notes)
        }
        SsType::Type2 => {
            let mut rows = Vec::with_capacity(hypotheses.len());
            for (t, (full, reduced)) in type2_lr_sets(lowered, retained).iter().enumerate() {
                if full.len() == reduced.len() {
                    rows.push(aliased(t));
                    continue;
                }
                fit_subset(lowered, fit_opts, fit, retained, full, cache);
                fit_subset(lowered, fit_opts, fit, retained, reduced, cache);
                rows.push(lrt_test(
                    fitted(full, retained, fit, cache),
                    fitted(reduced, retained, fit, cache),
                    full.len() - reduced.len(),
                ));
            }
            (rows, Vec::new())
        }
    }
}

/// Fits the column subset `set` of [`Lowered::x`] (caller columns,
/// ascending) into `cache`, unless it is there already or is the retained
/// design, whose fit is `fit`: the design of those columns as they are,
/// warm-started from their `β̂`, its random slopes moved by [`remap_all`].
fn fit_subset(
    lowered: &Lowered,
    fit_opts: &FitOptions,
    fit: &Fit,
    retained: &[usize],
    set: &[usize],
    cache: &mut RefitCache,
) {
    if set == retained || cache.0.iter().any(|(s, _)| s == set) {
        return;
    }
    let (n, p) = (lowered.n, lowered.p);
    let x: Vec<f64> = (0..n)
        .flat_map(|i| set.iter().map(move |&c| lowered.x[i * p + c]))
        .collect();
    let beta = set.iter().map(|&c| fit.beta[c]).collect();
    let moved = remap_all(&slope_lists(lowered), set, set.len());
    let refitted = refit(lowered, fit_opts, fit, &x, set.len(), &moved, beta);
    cache.0.push((set.to_vec(), refitted));
}

/// The fit of the column subset `set`: `fit` itself when `set` is the
/// retained design, else its refit in `cache` ([`fit_subset`]).
fn fitted<'a>(set: &[usize], retained: &[usize], fit: &'a Fit, cache: &'a RefitCache) -> &'a Fit {
    if set == retained {
        return fit;
    }
    &cache
        .0
        .iter()
        .find(|(s, _)| s.as_slice() == set)
        .expect("every subset is fitted before its row")
        .1
}

/// A likelihood-ratio row with no statistic: NaN, with no note.
fn nan_lrt_row() -> LinearTest {
    LinearTest {
        statistic: f64::NAN,
        kind: LinearKind::LikelihoodRatio,
        num_df: f64::NAN,
        reference: Reference::ChiSquared { df: 0 },
        p_value: f64::NAN,
        notes: Vec::new(),
    }
}

/// The likelihood-ratio row of a term from the fit of the larger model and
/// that of the smaller (the model under the term's hypothesis), on `df`, the
/// term's rank: `2(ℓ_larger − ℓ_smaller)` floored at 0 ([`lr_statistic`])
/// against χ²(`df`), a raw value below zero by more than round-off with
/// [`LinearNote::NegativeStatistic`]. A fit that did not converge, or whose
/// log-likelihood is not finite, gives NaN with
/// [`LinearNote::ReducedNotConverged`]: the caller passes a converged fit of
/// the user's model, so that fit is a refit.
pub(super) fn lrt_test(larger: &Fit, smaller: &Fit, df: usize) -> LinearTest {
    let reference = Reference::ChiSquared { df };
    let usable = |f: &Fit| f.converged() && f.loglik.is_finite();
    let (statistic, notes) = if usable(larger) && usable(smaller) {
        let (statistic, floor) = lr_statistic(smaller.loglik, larger.loglik);
        let note = floor.is_some().then_some(LinearNote::NegativeStatistic);
        (statistic, note.into_iter().collect())
    } else {
        (f64::NAN, vec![LinearNote::ReducedNotConverged])
    };
    LinearTest {
        statistic,
        kind: LinearKind::LikelihoodRatio,
        num_df: df as f64,
        reference,
        p_value: lr_p_value(statistic, &reference),
        notes,
    }
}

/// The refit of a Type III likelihood-ratio row on `X·N`: the columns
/// `retained` of `lowered`'s design times `basis` (`retained.len() × k`,
/// [`type3_null_basis`]), its random slopes moved over the `k` columns and
/// the random-effect-only columns they read ([`remap_all`]), warm-started
/// from `Nᵀβ̂` ([`refit`]).
pub(super) fn refit_on_basis(
    lowered: &Lowered,
    fit_opts: &FitOptions,
    fit: &Fit,
    retained: &[usize],
    basis: &Mat<f64>,
    moved: &MovedSlopes,
) -> Fit {
    let (n, p, k) = (lowered.n, lowered.p, basis.ncols());
    let combine = |v: &[f64], j: usize| -> f64 {
        retained
            .iter()
            .enumerate()
            .map(|(a, &c)| v[c] * basis[(a, j)])
            .sum()
    };
    let x: Vec<f64> = (0..n)
        .flat_map(|i| {
            let row = &lowered.x[i * p..(i + 1) * p];
            (0..k).map(move |j| combine(row, j))
        })
        .collect();
    let beta = (0..k).map(|j| combine(&fit.beta, j)).collect();
    refit(lowered, fit_opts, fit, &x, k, moved, beta)
}

/// A refit of a likelihood-ratio row: `lowered`'s model on the design `x`
/// (`n × k`, row-major), its random slopes on the columns of `x` and the
/// random-effect-only columns of `moved` ([`remap_all`]), by `fit_opts`, the
/// options [`term_tests`] requires `fit` to be made with (node count, held
/// dispersion), and
/// `lowered`'s weights and offset, warm-started from `beta`, the fit's θ̂
/// ([`theta_start`]) and its estimated dispersion ([`dispersion_start`]),
/// which the refit estimates again. Its standard errors are not computed.
/// The inputs are the checked fit's with fewer columns, so the refit skips
/// the entry's input check.
fn refit(
    lowered: &Lowered,
    fit_opts: &FitOptions,
    fit: &Fit,
    x: &[f64],
    k: usize,
    (slopes, sources): &MovedSlopes,
    beta: Vec<f64>,
) -> Fit {
    let mut model = lowered.model.clone();
    if let Some(re) = &mut model.re {
        let mut lists = slopes.iter().cloned();
        re.slopes = lists.next().expect("one slope list per grouping");
        set_term_slopes(&mut re.terms, &re.slopes);
        for g in &mut re.extra_groupings {
            g.slopes = lists.next().expect("one slope list per grouping");
            set_term_slopes(&mut g.terms, &g.slopes);
        }
    }
    // Only the log-likelihood is read: no SE targets and the Rx covariance,
    // so a GLMM refit skips the joint Hessian (the optimum does not depend on
    // the curvature read after it).
    let mut opts = fit_opts.clone();
    opts.se_for = SeFor::Only(vec![]);
    opts.wald_se = WaldSe::Rx;
    let start = StartValues {
        beta,
        theta: theta_start(fit),
        dispersion: dispersion_start(lowered, fit_opts, fit),
    };
    let (n, p, re_p) = (lowered.n, lowered.p, lowered.re_p);
    let z: Vec<f64> = (0..n)
        .flat_map(|i| {
            sources.iter().map(move |&c| match c.checked_sub(p) {
                None => lowered.x[i * p + c],
                Some(j) => lowered.re_columns[i * re_p + j],
            })
        })
        .collect();
    let full = lowered.data();
    let data = Data {
        design: Design {
            x,
            p: k,
            ..full.design
        },
        ..full
    }
    .re_columns(&z, sources.len());
    crate::fit::fit_checked(&data, &model, Some(&start), &opts)
}

/// Every grouping's random-slope columns in `lowered`'s model: the primary
/// grouping's, then each extra grouping's; none without random effects.
fn slope_lists(lowered: &Lowered) -> Vec<Vec<ColumnId>> {
    lowered
        .model
        .re
        .iter()
        .flat_map(|re| {
            std::iter::once(re.slopes.clone())
                .chain(re.extra_groupings.iter().map(|g| g.slopes.clone()))
        })
        .collect()
}

/// A refit's random slopes ([`slope_lists`]'s layout) and the
/// random-effect-only columns they read past the refit's `k` design columns,
/// each as its column of the full model's `[X | Z_r]` ([`remap_slopes`]).
type MovedSlopes = (Vec<Vec<ColumnId>>, Vec<usize>);

/// [`remap_slopes`] on each list of `slopes` for a reduced design of `k`
/// columns led by `kept`, the random-effect-only columns shared between the
/// lists.
fn remap_all(slopes: &[Vec<ColumnId>], kept: &[usize], k: usize) -> MovedSlopes {
    let mut sources = Vec::new();
    let moved = slopes
        .iter()
        .map(|s| remap_slopes(s, kept, k, &mut sources))
        .collect();
    (moved, sources)
}

/// `fit`'s θ̂ as a warm start ([`StartValues::theta`]): each `varcorr` block
/// `D̂` (`Λ̂Λ̂ᵀ` on a GLMM, whose σ² is 1) as its lower Cholesky factor,
/// vech-packed column-major, a pivot at or below `q·ε·tr D̂` (a variance at
/// zero) leaving its column zero. Empty, a cold start, without random effects
/// or with a block that is not finite.
pub(super) fn theta_start(fit: &Fit) -> Vec<f64> {
    let mut theta = Vec::new();
    for block in &fit.varcorr {
        if block.iter().any(|v| !v.is_finite()) {
            return Vec::new();
        }
        let q = vech_q(block.len());
        // Column-major vech: column `c` holds rows `c..q`.
        let at = |r: usize, c: usize| block[c * q - c * c.saturating_sub(1) / 2 + r - c];
        let floor = q as f64 * f64::EPSILON * diagonal(block).sum::<f64>();
        let mut l = vec![0.0; q * q];
        for c in 0..q {
            let pivot = at(c, c) - (0..c).map(|k| l[c * q + k] * l[c * q + k]).sum::<f64>();
            if pivot <= floor {
                continue;
            }
            let d = pivot.sqrt();
            l[c * q + c] = d;
            for r in c + 1..q {
                let dot: f64 = (0..c).map(|k| l[r * q + k] * l[c * q + k]).sum();
                l[r * q + c] = (at(r, c) - dot) / d;
            }
        }
        for c in 0..q {
            theta.extend((c..q).map(|r| l[r * q + c]));
        }
    }
    theta
}

/// `fit`'s estimated dispersion as a warm start
/// ([`StartValues::dispersion`]): the negative binomial θ, or the Gamma or
/// inverse Gaussian φ when the fit estimated it. `None`, a cold start, when
/// the family has none, φ is held, or the value is not finite and positive.
fn dispersion_start(lowered: &Lowered, fit_opts: &FitOptions, fit: &Fit) -> Option<f64> {
    let estimated = fit_opts.dispersion.held().is_none()
        && matches!(
            lowered.model.family,
            Family::NegativeBinomial { .. } | Family::Gamma { .. } | Family::InverseGaussian { .. }
        );
    (estimated && fit.dispersion.is_finite() && fit.dispersion > 0.0).then_some(fit.dispersion)
}

/// The NaN row of a test that did not run, carrying `notes`: an F row
/// labelled `den_df_kind` when `f_rows`, else a χ² row.
pub(crate) fn untested_test(
    f_rows: bool,
    den_df_kind: DenDf,
    notes: Vec<LinearNote>,
) -> LinearTest {
    LinearTest {
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
    }
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
    let (rows, p) = (h.l.nrows(), h.l.ncols());
    if rows == 0 {
        let aliased = term.columns.clone().filter(|&j| fit.aliased()[j]);
        return untested_test(
            f_rows,
            den_df_kind,
            aliased
                .map(|column| LinearNote::AliasedColumn { column })
                .collect(),
        );
    }
    let l: Vec<f64> = (0..rows)
        .flat_map(|i| (0..p).map(move |j| h.l[(i, j)]))
        .collect();
    if l.iter().any(|v| !v.is_finite()) {
        return untested_test(f_rows, den_df_kind, Vec::new());
    }
    if !f_rows {
        return wald_chisq(&fit.beta, &fit.vcov_rows(), fit.aliased(), &l, rows);
    }
    let den_df_kind = if fit.robust_inputs().is_some() {
        den_df_kind.on_robust_fit(rows)
    } else {
        den_df_kind
    };
    match fit.contrast_f(&l, rows) {
        Ok(cf) => LinearTest::from_contrast_f(cf, den_df_kind),
        Err(ContrastError::NoDf(reason)) => {
            untested_test(f_rows, den_df_kind, vec![LinearNote::NoDf { reason }])
        }
    }
}
