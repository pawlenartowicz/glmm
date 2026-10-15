//! Comparing two fitted models: the formula nesting check, the checks two
//! fits must pass before a comparison, the refusals that name the fix, and
//! the test each pair gets.
//!
//! [`compare`] takes two [`Model`]s, plain numbers a port's fit object keeps,
//! never a [`Fit`] or a [`Lowered`], so a comparison of two stored fits needs
//! no refit and no data. Its checks run in this order, each a refusal
//! ([`CompareError`]):
//!
//! 1. both fits converged, and each side's [`Model::re_groups`] lists one
//!    grouping per `varcorr` block, each with one term per block dimension;
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
//! The public items of this module are outside the crate's semver promise
//! until the 1.0 surface review: their names and signatures may change in any
//! release before then.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use faer::Mat;

use super::hypotheses::{
    check_marginal, missing_cells_note, remap_slopes, type2_lr_sets, type2_vcov, type2_xtx, type3,
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
use crate::{
    fit_warm, BinomialLink, ColumnId, Family, Fit, GammaLink, InverseGaussianLink, NegBinomialLink,
    PoissonLink, StartValues, WaldSe,
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
    /// The zero-component rule reads it to match blocks and terms by name.
    /// [`compare`] refuses a side whose `re_groups` does not list one
    /// grouping per `varcorr` block, each with one term per block dimension
    /// ([`CompareError::ReGroups`]), so fill it from the fit's `re_groups`.
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
        LinearNote::ReducedNotConverged => "The model refitted for this term's \
             likelihood-ratio test did not converge, so the statistic and its p-value are NaN."
            .into(),
        LinearNote::RandomSlopeColumn => "Likelihood-ratio test not available: the term's \
             column carries a random slope, so the row is the Wald chi-squared test."
            .into(),
        LinearNote::NegativeStatistic => "The model refitted under the term's hypothesis \
             reached a higher log-likelihood than the larger model, by more than round-off, so \
             the statistic is set to 0: the larger model's fit did not reach its optimum; refit \
             it."
        .into(),
    }
}

/// The table note of a likelihood-ratio table on a fit that did not
/// converge.
pub(super) fn lrt_not_converged_note() -> String {
    "The fit did not converge, so no model was refitted and the likelihood-ratio statistics \
     are NaN."
        .into()
}

/// The table note of a likelihood-ratio row whose reduced design could not
/// be built: the null basis of the term's hypothesis failed its check.
pub(super) fn lrt_no_basis_note(term: &str) -> String {
    format!(
        "The model under the hypothesis of {term} could not be built (the hypothesis is too \
         ill-conditioned), so its likelihood-ratio statistic is NaN."
    )
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
    /// A side's [`Model::re_groups`] does not describe its random part: it
    /// does not list one grouping per [`Model::varcorr`] block, each with one
    /// term per dimension of its block.
    ReGroups {
        /// The side whose groupings do not match its variance blocks.
        side: Side,
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
            CompareError::ReGroups { side } => write!(
                f,
                "{}'s random-effect groupings do not describe its random part: a comparison \
                 needs one grouping per variance block, each listing one term per dimension of \
                 its block, as the fit's lowering gives them. Fill them from the lowered model \
                 the fit was made on, then compare again.",
                side.label()
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
    // Each side's groupings describe its variance blocks. Checked after
    // convergence: a fit that did not converge may have no blocks (an empty
    // `varcorr`) beside its groupings.
    for (side, m) in [(Side::First, a), (Side::Second, b)] {
        let described = m.re_groups.len() == m.varcorr.len()
            && m.re_groups
                .iter()
                .zip(&m.varcorr)
                .all(|((_, terms), block)| {
                    let t = terms.len();
                    t * (t + 1) / 2 == block.len()
                });
        if !described {
            return Err(CompareError::ReGroups { side });
        }
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
    if let (Some(k0), Some(k1)) = (quadrature_nodes(a.objective), quadrature_nodes(b.objective)) {
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
    /// ([`LinearKind::LikelihoodRatio`]). On a likelihood-ratio table, a row
    /// whose term carries a random slope is the Wald χ² with
    /// [`LinearNote::RandomSlopeColumn`].
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
    /// count than `lowered.opts.nagq`, the count the rows' refits use, so
    /// each row would compare two approximations of the likelihood.
    RefitNodeCount {
        /// The fit's node count (1, the Laplace approximation).
        fit: u8,
        /// The node count of the refits, `lowered.opts.nagq`.
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
                let den_df = match den_df {
                    DenDf::Satterthwaite => "Satterthwaite's",
                    DenDf::Residual => "the residual",
                };
                write!(
                    f,
                    "per-term tests on this fit are F tests with {den_df} denominator df; \
                     {asked} rows are not available{advice}"
                )
            }
            TermTestError::RefitNodeCount { fit, refits } => write!(
                f,
                "the fit used {fit} quadrature nodes and the refits of its likelihood-ratio \
                 rows would use {refits} (1 is the Laplace approximation), so each row would \
                 compare two approximations of the likelihood; fit the model with {refits} \
                 nodes, or ask for the Wald test"
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

/// The per-term tests of `fit`, the fit of `lowered` made with
/// `lowered.opts`: one row per fixed term of [`Lowered::terms`], in that
/// order (R's), each the test of the term's hypothesis set of
/// `opts.ss_type`, read off the one fit (a likelihood-ratio row also refits
/// the model under its hypothesis).
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
///   [`Fit::vcov`] under [`TermTest::Wald`], and the likelihood-ratio χ²
///   under [`TermTest::Lrt`]. Under [`TermTest::Auto`] (the default) a
///   Poisson model with random effects gets the likelihood-ratio χ² and
///   every other fit the Wald χ² ([`ChoiceReason`]).
///
/// [`TermTable::choice`] records the test run and why. A fit with F rows
/// takes only [`TermTest::Auto`].
///
/// A likelihood-ratio row refits the model under the term's hypothesis with
/// [`fit_warm`], from the fit's β̂, θ̂ and estimated dispersion, by
/// `lowered.opts` (node count, weights, offset, held dispersion; a negative
/// binomial θ is re-estimated). These rows therefore need `fit` made with
/// `lowered.opts`: a fit made at another node count is refused
/// ([`TermTestError::RefitNodeCount`]), and weights, an offset or a held
/// dispersion cannot be read back from a [`Fit`], so a caller that fits with
/// other ones gets rows that compare two different likelihoods. The Wald and
/// F rows read the fit alone. Type III fits the reduced design
/// `X·N`, `N` an orthonormal basis of the null space of the term's `L` (the
/// hypothesis its Wald row tests); Type II, car's `Anova(type = 2,
/// test.statistic = "LR")`, fits two column subsets of the design, the model
/// of every term that does not contain the term with and without it, each
/// distinct subset once. The statistic `2(ℓ₁ − ℓ₀)`, floored at 0, is
/// referred to χ² on the term's rank; a raw value below zero by more than
/// round-off takes [`LinearNote::NegativeStatistic`]. A refit that does not
/// converge gives a NaN row with [`LinearNote::ReducedNotConverged`]; a term
/// whose reduced design would leave out or mix a column carrying a random
/// slope gets its Wald row with [`LinearNote::RandomSlopeColumn`]; on a fit
/// that did not converge every row is NaN, nothing is refitted, and a table
/// note says why.
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
/// effects) and the fit's quadrature node count is not `lowered.opts.nagq`.
pub fn term_tests(
    lowered: &Lowered,
    fit: &Fit,
    opts: &TermTestOptions,
) -> Result<TermTable, TermTestError> {
    term_tests_cached(lowered, fit, opts, &mut RefitCache::default())
}

/// Refits of column subsets of one fit's design, each keyed by its subset
/// (caller columns of [`Lowered::x`], ascending), so the likelihood-ratio
/// tables of one fit fit each subset once between them
/// ([`term_tests_cached`]). A cache belongs to the `(lowered, fit)` pair it
/// was filled for.
#[derive(Default)]
pub(crate) struct RefitCache(Vec<(Vec<usize>, Fit)>);

/// [`term_tests`], with the likelihood-ratio rows' refits of column subsets
/// kept in `cache`: a caller that builds both types' likelihood-ratio tables
/// of one fit passes one cache to both, and a Type III hypothesis that drops
/// columns reads the Type II design with the same columns, built and fitted
/// the same way. The rows are those of [`term_tests`] bit for bit.
pub(crate) fn term_tests_cached(
    lowered: &Lowered,
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
    // The likelihood-ratio rows refit by `lowered.opts`: a fit made at another
    // node count would be compared with refits of another approximation.
    if choice.test == LinearKind::LikelihoodRatio {
        if let Some(nodes) = quadrature_nodes(fit.objective) {
            if nodes != lowered.opts.nagq {
                return Err(TermTestError::RefitNodeCount {
                    fit: nodes,
                    refits: lowered.opts.nagq,
                });
            }
        }
    }
    let retained: Vec<usize> = (0..lowered.p).filter(|&j| !fit.aliased()[j]).collect();
    let hypotheses = match opts.ss_type {
        SsType::Type3 => type3(lowered, &retained),
        SsType::Type2 if gaussian && mixed => type2_xtx(lowered, &retained),
        SsType::Type2 => type2_vcov(lowered, &retained, &fit.vcov),
    };
    let mut notes: Vec<String> = match opts.ss_type {
        SsType::Type3 => missing_cells_note(lowered).into_iter().collect(),
        SsType::Type2 => Vec::new(),
    };
    let tests: Vec<LinearTest> = if choice.test == LinearKind::LikelihoodRatio {
        let (tests, lrt_notes) =
            lrt_rows(lowered, fit, &retained, opts.ss_type, &hypotheses, cache);
        notes.extend(lrt_notes);
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
        .map(|(h, test)| {
            let ms = f_rows.then_some(test.statistic * fit.dispersion);
            FixedRow {
                label: h.term,
                sum_sq: ms.map(|ms| ms * test.num_df),
                mean_sq: ms,
                test,
            }
        })
        .collect();
    Ok(TermTable {
        ss_type: opts.ss_type,
        choice,
        fixed,
        notes,
        coef_names: lowered.col_names.clone(),
    })
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
/// NaN row; a term whose reduced design would leave out or mix a random-slope
/// column, its Wald row with [`LinearNote::RandomSlopeColumn`]; a basis that
/// fails its check, a NaN row and the table note [`lrt_no_basis_note`]. A fit
/// that did not converge gets NaN rows and the table note
/// [`lrt_not_converged_note`], and nothing is refitted.
fn lrt_rows(
    lowered: &Lowered,
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
    let slope_wald = |t: usize| {
        let mut row = wald(t);
        row.notes.push(LinearNote::RandomSlopeColumn);
        row
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
    let slopes = slope_lists(lowered);
    match ss_type {
        SsType::Type3 => {
            let exact = type3_unrounded(lowered, retained);
            // The slope columns as positions in `retained`: an aliased one
            // stays in no reduced design.
            let on_retained = remap_all(&slopes, retained);
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
                    let Some(retained_slopes) = &on_retained else {
                        return slope_wald(t);
                    };
                    let flat: Vec<usize> = retained_slopes
                        .iter()
                        .flatten()
                        .map(|&c| c as usize)
                        .collect();
                    let l = restrict(&h.l);
                    let basis =
                        match type3_null_basis(l.as_ref(), restrict(&exact[t].l).as_ref(), &flat) {
                            Ok(basis) => basis,
                            Err(ReducedDesignError::SlopeTouched) => return slope_wald(t),
                            Err(ReducedDesignError::NoNullBasis) => {
                                notes.push(lrt_no_basis_note(&h.term));
                                return nan_lrt_row();
                            }
                        };
                    let k = basis.ncols();
                    let kept = untouched_columns(l.as_ref());
                    if k == kept.len() {
                        // `N` is the unit columns of `kept`: a column drop,
                        // fitted as the Type II subset of the same columns.
                        let set: Vec<usize> = kept.iter().map(|&a| retained[a]).collect();
                        let set_slopes = remap_all(&slopes, &set)
                            .expect("the basis keeps every slope column as a unit column");
                        fit_subset(lowered, fit, retained, &set, &set_slopes, cache);
                        return lrt_test(
                            fit,
                            fitted(&set, retained, fit, cache),
                            retained.len() - k,
                        );
                    }
                    let moved = remap_all(retained_slopes, &kept)
                        .expect("the basis keeps every slope column as a unit column");
                    let reduced = refit_on_basis(lowered, fit, retained, &basis, &moved);
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
                let (Some(on_full), Some(on_reduced)) =
                    (remap_all(&slopes, full), remap_all(&slopes, reduced))
                else {
                    rows.push(slope_wald(t));
                    continue;
                };
                fit_subset(lowered, fit, retained, full, &on_full, cache);
                fit_subset(lowered, fit, retained, reduced, &on_reduced, cache);
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
/// warm-started from their `β̂`, its random slopes on `set_slopes` (positions
/// in `set`).
fn fit_subset(
    lowered: &Lowered,
    fit: &Fit,
    retained: &[usize],
    set: &[usize],
    set_slopes: &[Vec<ColumnId>],
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
    let refitted = refit(lowered, fit, &x, set.len(), set_slopes, beta);
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
/// [`type3_null_basis`]), its random slopes on `slopes` (positions among the
/// `k` columns), warm-started from `Nᵀβ̂` ([`refit`]).
pub(super) fn refit_on_basis(
    lowered: &Lowered,
    fit: &Fit,
    retained: &[usize],
    basis: &Mat<f64>,
    slopes: &[Vec<ColumnId>],
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
    refit(lowered, fit, &x, k, slopes, beta)
}

/// A refit of a likelihood-ratio row: `lowered`'s model on the design `x`
/// (`n × k`, row-major), its random slopes on the columns `slopes`
/// ([`slope_lists`]'s layout), by `lowered.opts`, the options
/// [`term_tests`] requires `fit` to be made with (node count, weights,
/// offset, held dispersion), warm-started from `beta`, the fit's θ̂
/// ([`theta_start`]) and its estimated dispersion ([`dispersion_start`]),
/// which the refit estimates again. Its standard errors are not computed.
fn refit(
    lowered: &Lowered,
    fit: &Fit,
    x: &[f64],
    k: usize,
    slopes: &[Vec<ColumnId>],
    beta: Vec<f64>,
) -> Fit {
    let mut model = lowered.model.clone();
    if let Some(re) = &mut model.re {
        let mut lists = slopes.iter().cloned();
        re.slopes = lists.next().expect("one slope list per grouping");
        for g in &mut re.extra_groupings {
            g.slopes = lists.next().expect("one slope list per grouping");
        }
    }
    // Only the log-likelihood is read: no SE targets and the Rx covariance,
    // so a GLMM refit skips the joint Hessian (the optimum does not depend on
    // the curvature read after it).
    let mut opts = lowered.opts.clone();
    opts.target_indices = Vec::new();
    opts.wald_se = WaldSe::Rx;
    let start = StartValues {
        beta,
        theta: theta_start(fit),
        dispersion: dispersion_start(lowered, fit),
    };
    fit_warm(
        x,
        &lowered.y,
        lowered.n,
        k,
        &model,
        &lowered.ids,
        Some(&start),
        &opts,
    )
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

/// [`remap_slopes`] on each list of `slopes`: `None` when a slope column is
/// not in `kept`.
fn remap_all(slopes: &[Vec<ColumnId>], kept: &[usize]) -> Option<Vec<Vec<ColumnId>>> {
    slopes.iter().map(|s| remap_slopes(s, kept).ok()).collect()
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
fn dispersion_start(lowered: &Lowered, fit: &Fit) -> Option<f64> {
    let estimated = lowered.opts.dispersion.is_none()
        && matches!(
            lowered.model.family,
            Family::NegativeBinomial { .. } | Family::Gamma { .. } | Family::InverseGaussian { .. }
        );
    (estimated && fit.dispersion.is_finite() && fit.dispersion > 0.0).then_some(fit.dispersion)
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
