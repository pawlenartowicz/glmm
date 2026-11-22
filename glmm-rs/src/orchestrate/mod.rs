//! Fit orchestration that `glmm-bind` calls for both FFI ports (`glmm-python`, `glmm-r`):
//! formula + data -> [`crate::formula::lower`] -> option overrides
//! -> [`crate::fit`] / [`crate::fit_with_start`], whose [`crate::FitError`]
//! becomes a normal `Err` with its message. A panic (a crate bug) is not
//! caught here: the port layer above catches it and reports it as an internal
//! error. Plain-Rust types only — no `pyo3`/`extendr`
//! types — so everything here runs under plain `cargo test`, and the two
//! ports flatten the same fit the same way from one definition instead of
//! two mirrored copies. The entry points that lower a port's data
//! ([`run_fit`], [`run_compare`], [`run_model_matrix`], [`predict`],
//! [`simulate`]) return a [`PortError`], which keeps the data row an error
//! names apart from its text, so each port names the row in its caller's own
//! numbering. [`run_fit`] fits one formula; [`run_compare`] fits
//! two on the same rows and compares them; [`run_model_matrix`] returns a
//! formula's fixed-effects design without fitting; [`term_labels`] reads a
//! formula's fixed-term labels; [`contrast_test`] tests `Lβ = 0` on what a
//! port stored of one fit, and [`pairwise`] compares a factor's levels on it;
//! [`profile_intervals`] computes a fit's profile-likelihood intervals on its
//! training columns, and [`profile_names`] names their rows.
//!
//! Behind the default-off `orchestrate` cargo feature. Like `loop_advanced`
//! it carries NO semver guarantees: its shape follows the ports' needs.

mod design;
mod fit;
mod predict;
mod stored;

pub use design::{
    model_matrix_columns, run_model_matrix, same_fixed_part, term_labels, FormulaColumns,
};
pub use fit::{
    compare_options, run_compare, run_fit, term_columns, CompareResult, DesignSummary, DfMethod,
    FitResult, NoteInfo, RanefBlock, TermColumns, TermTableError, DEFAULT_NSIM, DEFAULT_SEED,
    MAX_NSIM,
};
pub use predict::{predict, predict_columns, simulate, PredictResult};
pub use stored::{
    confint, contrast_test, effect_sizes, pairwise, pairwise_columns, pairwise_confint,
    profile_intervals, profile_names, CondvarCrossRecord, PairwiseColumns, ProfileResult,
    RobustRecord, SatterthwaiteRecord, StoredParts,
};

/// The largest `q_p` adaptive quadrature takes, which the ports' `agq_fallback`
/// warning names.
pub use crate::consts::MAX_AGQ_Q;
use crate::fit::splice_row;
/// The SDs and correlations of a vech-packed covariance block, as
/// [`crate::Fit::stddev_corr`] gives them.
pub use crate::fit::vech_stddev_corr;
/// Why a fit did not converge, as [`FitResult::stop`] carries it.
pub use crate::fit::StopReason;
use crate::formula::anova::{ChoiceReason, TestChoice};
use crate::formula::{Adjust, RatioKind};
use crate::inference::{LinearKind, Objective};
/// The per-entry θ box of the LMM and GLMM searches, which the ports'
/// `variance_at_limit` warning names.
pub use crate::lmm::THETA_HI;
use crate::{
    BinomialLink, Boundary, ErrorMessage, Family, FitError, GammaLink, InverseGaussianLink,
    NegBinomialLink, OuterStop, PoissonLink,
};

/// The error of an entry point that lowers or fits a port's data: the
/// message, and the 0-based row of the data the port passed that it names,
/// if any. The text counts rows and design columns from 0 over the rows
/// passed, as [`crate::FitError`]'s does; [`PortError::message_shown`] names
/// the row and the design column in the port's own numbering, so the port
/// never parses the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PortError {
    /// The message, as [`crate::FitError`] or [`crate::formula::Error`]
    /// display it. Read it, do not edit it: [`PortError::message_shown`]
    /// splices the row and the column at byte ranges of this text.
    pub message: String,
    /// The 0-based row of the data the port passed, `None` when the error
    /// names no row.
    pub row: Option<usize>,
    /// The byte range of the row's digits in `message`.
    digits: Option<(usize, usize)>,
    /// The 0-based design column `message` names and the byte range of its
    /// digits.
    column: Option<(usize, (usize, usize))>,
    /// Whether the error refuses a value the caller gave a fit option or a
    /// warm start ([`FitError::OptionValue`], [`FitError::Start`]), not the
    /// model or the data.
    pub argument: bool,
}

impl PortError {
    /// The message with `row` in place of the row's number and the 0-based
    /// column plus `column_base` in place of the column's. `row` is, for a
    /// port, the position in its caller's data of the row it passed as
    /// [`PortError::row`]. A number the message does not name is skipped.
    pub fn message_shown(&self, row: usize, column_base: usize) -> String {
        // The column's digits come after the row's, so the column goes in
        // first: a row with more digits would move the column's range.
        let at = |text: &str, (start, end): (usize, usize)| {
            text.is_char_boundary(start) && text.is_char_boundary(end)
        };
        let mut text = self.message.clone();
        if let Some((column, range)) = self.column.filter(|&(_, r)| at(&text, r)) {
            text = splice_row(&text, range, column + column_base);
        }
        if let Some(range) = self.digits.filter(|&r| at(&text, r)) {
            text = splice_row(&text, range, row);
        }
        text
    }

    /// A refusal of a value the caller gave an argument.
    fn argument(message: String) -> Self {
        PortError {
            argument: true,
            ..message.into()
        }
    }
}

impl From<&ErrorMessage> for PortError {
    fn from(m: &ErrorMessage) -> Self {
        PortError {
            message: m.to_string(),
            row: m.row(),
            digits: m.digits(),
            column: m.column_at(),
            argument: false,
        }
    }
}

impl From<String> for PortError {
    fn from(message: String) -> Self {
        PortError {
            message,
            row: None,
            digits: None,
            column: None,
            argument: false,
        }
    }
}

impl From<FitError> for PortError {
    fn from(e: FitError) -> Self {
        PortError {
            argument: matches!(e, FitError::OptionValue(_) | FitError::Start(_)),
            ..e.message().into()
        }
    }
}

impl From<crate::formula::Error> for PortError {
    fn from(e: crate::formula::Error) -> Self {
        (&e.message()).into()
    }
}

impl From<PortError> for String {
    fn from(e: PortError) -> Self {
        e.message
    }
}

/// The message, so `e.contains(..)` reads it as [`crate::ErrorMessage`] is read.
impl std::ops::Deref for PortError {
    type Target = str;
    fn deref(&self) -> &str {
        &self.message
    }
}

impl PartialEq<str> for PortError {
    fn eq(&self, other: &str) -> bool {
        self.message == other
    }
}

impl PartialEq<&str> for PortError {
    fn eq(&self, other: &&str) -> bool {
        self.message == *other
    }
}

impl PartialEq<String> for PortError {
    fn eq(&self, other: &String) -> bool {
        self.message == *other
    }
}

impl std::fmt::Display for PortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// Maps the ports' `family`/`link` strings to [`Family`], for `glmm-bind`,
/// which calls it on the strings its family table has checked and
/// normalized, and on a stored fit's. Every family and link in that table
/// maps to a variant, so an `Err` here comes only from a stored fit whose
/// strings are not the table's.
pub fn family_from_str(family: &str, link: &str) -> Result<Family, String> {
    match family {
        "gaussian" => match link {
            "identity" => Ok(Family::Gaussian),
            other => Err(format!("unsupported gaussian link {other:?}")),
        },
        "binomial" => match link {
            "logit" => Ok(Family::Binomial {
                link: BinomialLink::Logit,
            }),
            "probit" => Ok(Family::Binomial {
                link: BinomialLink::Probit,
            }),
            "cloglog" => Ok(Family::Binomial {
                link: BinomialLink::Cloglog,
            }),
            other => Err(format!("unsupported binomial link {other:?}")),
        },
        "poisson" => match link {
            "log" => Ok(Family::Poisson {
                link: PoissonLink::Log,
            }),
            other => Err(format!("unsupported poisson link {other:?}")),
        },
        "gamma" => match link {
            "log" => Ok(Family::Gamma {
                link: GammaLink::Log,
            }),
            "inverse" => Ok(Family::Gamma {
                link: GammaLink::Inverse,
            }),
            other => Err(format!("unsupported gamma link {other:?}")),
        },
        "negativebinomial" => match link {
            "log" => Ok(Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            }),
            other => Err(format!("unsupported negativebinomial link {other:?}")),
        },
        "inversegaussian" => match link {
            "log" => Ok(Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            }),
            "inverse_squared" => Ok(Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            }),
            other => Err(format!("unsupported inversegaussian link {other:?}")),
        },
        other => Err(format!("unknown family {other:?}")),
    }
}

/// `Σᵢ wᵢ·dᵢ`, `dᵢ` the unit deviance of `family` (a port family name) at
/// `y[i]` and `mu[i]` and `wᵢ` the row's weight (1 without `weights`): the
/// deviance of a GLM at `mu`, R's `sum(residuals(fit, "deviance")^2)`, the
/// weighted residual sum of squares on Gaussian. The link does not enter.
/// `nb_theta` is the negative-binomial shape; the other families ignore it.
///
/// # Errors
/// An unknown family name, or `y`, `mu` and `weights` of different lengths.
pub fn unit_deviance_sum(
    family: &str,
    nb_theta: f64,
    y: &[f64],
    mu: &[f64],
    weights: Option<&[f64]>,
) -> Result<f64, String> {
    // Any link of the family: the unit deviance reads only y and mu.
    let link = match family {
        "gaussian" => "identity",
        "binomial" => "logit",
        _ => "log",
    };
    let fam = family_from_str(family, link)?;
    if y.len() != mu.len() || weights.is_some_and(|w| w.len() != mu.len()) {
        return Err(format!(
            "y, fitted and weights must have the same length, got {}, {} and {}",
            y.len(),
            mu.len(),
            weights.map_or(mu.len(), <[f64]>::len)
        ));
    }
    Ok(y.iter()
        .zip(mu)
        .enumerate()
        .map(|(i, (&y, &mu))| {
            weights.map_or(1.0, |w| w[i]) * crate::family::dev_resid(fam, nb_theta, y, mu)
        })
        .sum())
}

/// [`Boundary`] as the string the ports publish. Exhaustive for the same
/// reason `note_infos` is: a new variant must pick its string here.
fn boundary_name(boundary: Boundary) -> &'static str {
    match boundary {
        Boundary::Interior => "interior",
        Boundary::AtBoundary => "at_boundary",
        Boundary::NoOptimum => "no_optimum",
        Boundary::UpperLimit => "upper_limit",
    }
}

/// [`OuterStop`] as the string the ports publish, exhaustive like
/// `boundary_name`.
fn outer_stop_name(stop: OuterStop) -> &'static str {
    match stop {
        OuterStop::Fdelta => "fdelta",
        OuterStop::Pgrad => "pgrad",
        OuterStop::NoiseFloor => "noise_floor",
        OuterStop::LineSearch => "line_search",
        OuterStop::MaxIter => "max_iter",
        OuterStop::GradUnavailable => "grad_unavailable",
        OuterStop::NonFinite => "non_finite",
    }
}

/// A row's or a table's test as [`TermColumns::kind`] names it.
fn kind_tag(kind: LinearKind) -> &'static str {
    match kind {
        LinearKind::F => "F",
        LinearKind::ChiSquared => "chisq",
        LinearKind::LikelihoodRatio => "lrt",
    }
}

/// A [`TestChoice`] as the two tags `glmm-bind` stores for both ports: the test, named as
/// in [`TermColumns::kind`], and the reason, as in [`TermColumns::reason`].
pub fn choice_tags(choice: &TestChoice) -> (&'static str, &'static str) {
    let reason = match choice.reason {
        ChoiceReason::Requested => "requested",
        ChoiceReason::FTest => "f_test",
        ChoiceReason::PoissonMixed => "poisson_mixed",
        ChoiceReason::BinomialMixed => "binomial_mixed",
        ChoiceReason::NegativeBinomialMixed => "negative_binomial_mixed",
        ChoiceReason::OtherMixed => "other_mixed",
        ChoiceReason::NoRandomEffects => "no_random_effects",
    };
    (kind_tag(choice.test), reason)
}

/// An [`Objective`] as `(label, node count)` for `glmm-bind`'s records: the label is
/// one of `"least_squares"`, `"maximum_likelihood"`, `"reml"`, `"laplace"`
/// and `"agq"`; the node count is 1 on `"laplace"`, the quadrature's on
/// `"agq"` and 0 elsewhere.
pub fn objective_parts(objective: Objective) -> (&'static str, u8) {
    match objective {
        Objective::LeastSquares => ("least_squares", 0),
        Objective::MaximumLikelihood => ("maximum_likelihood", 0),
        Objective::Reml => ("reml", 0),
        Objective::Laplace => ("laplace", 1),
        Objective::Agq { nodes } => ("agq", nodes),
    }
}

/// The inverse of [`objective_parts`], for a port that rebuilds a fit's
/// [`Objective`] from the two stored fields.
///
/// # Errors
///
/// When `(label, nodes)` is not a pair [`objective_parts`] gives: an unknown
/// label, a node count other than 0 on `"least_squares"`,
/// `"maximum_likelihood"` and `"reml"` or other than 1 on `"laplace"`, or
/// fewer than 2 nodes on `"agq"`.
pub fn objective_from_parts(label: &str, nodes: u8) -> Result<Objective, String> {
    match (label, nodes) {
        ("least_squares", 0) => Ok(Objective::LeastSquares),
        ("maximum_likelihood", 0) => Ok(Objective::MaximumLikelihood),
        ("reml", 0) => Ok(Objective::Reml),
        ("laplace", 1) => Ok(Objective::Laplace),
        ("agq", 2..) => Ok(Objective::Agq { nodes }),
        _ => Err(format!(
            "no fit stores the objective {label:?} with {nodes} quadrature {}",
            if nodes == 1 { "node" } else { "nodes" }
        )),
    }
}

/// An [`Adjust`] as the ports name it: `"holm"`, `"bonferroni"`, `"tukey"` or
/// `"none"`.
pub fn adjust_tag(adjust: Adjust) -> &'static str {
    match adjust {
        Adjust::Holm => "holm",
        Adjust::Bonferroni => "bonferroni",
        Adjust::Tukey => "tukey",
        Adjust::None => "none",
    }
}

/// A [`RatioKind`] as the ports name it: `"odds_ratio"` or `"ratio"`.
fn ratio_tag(ratio: RatioKind) -> &'static str {
    match ratio {
        RatioKind::OddsRatio => "odds_ratio",
        RatioKind::Ratio => "ratio",
    }
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "glmm kernel panicked with a non-string payload".to_string()
    }
}

#[cfg(test)]
mod design_tests;
#[cfg(test)]
mod fit_tests;
#[cfg(test)]
mod predict_tests;
#[cfg(test)]
mod stored_tests;
#[cfg(test)]
mod tests;
