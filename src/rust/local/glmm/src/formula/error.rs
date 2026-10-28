//! The crate's error surface. `ParseError`'s four variants and their Display
//! strings are a fixed contract — the parse test suite matches error
//! substrings against them, so both change together with `parse.rs`; `Error`
//! wraps it and adds the materialize-stage (data-dependent) failures. No
//! `thiserror` dep — Display is hand-written so the crate's only third-party
//! dependency stays `regex`.

use std::fmt;

/// Parse-stage (data-free) failures. Variants and Display strings are a fixed
/// contract (change together with `parse.rs`, which produces them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The formula (or its RHS) is empty.
    EmptyFormula,
    /// A token on the RHS is not a valid identifier, or a term is malformed
    /// (see [`parse`](super::parse())'s Errors list).
    Syntax {
        /// 0-based position of the offending token in the formula string.
        pos: usize,
        /// Description of what's wrong with the token.
        msg: String,
    },
    /// A `-` term removal outside parentheses (unsupported).
    TermRemovalUnsupported,
    /// A random-effect term with no column: nothing but `+` signs after its
    /// `1+`, e.g. `(1++|g)`, or the intercept removed with nothing left, e.g.
    /// `(0|g)`.
    EmptySlopeTerm {
        /// The grouping factor of the empty slope term.
        group: String,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::EmptyFormula => write!(f, "formula is empty"),
            ParseError::Syntax { pos, msg } => {
                write!(f, "formula syntax error at position {pos}: {msg}")
            }
            ParseError::TermRemovalUnsupported => {
                write!(f, "term removal with '-' is not supported")
            }
            ParseError::EmptySlopeTerm { group } => {
                write!(f, "empty slope term for group {group}")
            }
        }
    }
}

/// Everything `parse`, `materialize`/`lower` or [`fit`](super::fit()) can
/// reject.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A data-free parse failure (see [`ParseError`]).
    Parse(ParseError),
    /// A name referenced by the formula is not a column of the data table.
    UnknownColumn(String),
    /// The response column exists but is not numeric.
    ResponseNotNumeric(String),
    /// A column referenced as numeric/factor has the wrong kind for its use.
    WrongColumnKind {
        /// The column name whose kind doesn't match its use.
        name: String,
        /// The kind the column was expected to be, phrased to complete
        /// "column … is not …": `"numeric"` for a column read as numbers
        /// (a transform argument, an offset, a `cbind()` count), `"a factor
        /// or a column of whole numbers (grouping variable)"` for a grouping
        /// column, `"a factor, as in the fit"` for a new table's column that
        /// was a factor in the fit ([`super::predict`]), and `"a factor of
        /// the fixed part"` or `"a numeric variable with chosen values"` for
        /// the factor or `by` variable of [`super::pairwise()`].
        expected: &'static str,
    },
    /// A factor in the fixed part of the formula (character, or logical with a
    /// single observed value) has fewer than 2 levels, so no treatment
    /// contrast can be built for it. R's `contrasts<-` refuses the same thing;
    /// a one-level grouping factor in the random part is not this — it is
    /// dropped with a warning instead ([`crate::Note::SingleLevelGroupingDropped`]).
    SingleLevelFactor {
        /// The factor column name.
        name: String,
    },
    /// After factor coding, one random-effect column (the intercept included)
    /// is in two terms on one grouping, e.g. `(1 | g) + (1 + x | g)`. A
    /// factor's term holds the intercept in the crate's treatment coding, so
    /// `(1 + x | g) + (0 + f | g)` is refused too.
    RandomColumnRepeated {
        /// The grouping factor.
        group: String,
        /// The repeated column: `"(Intercept)"`, a slope column, or a
        /// factor's dummy column.
        column: String,
    },
    /// [`super::label_ranef`] was handed a `Fit` and a `re_groups` that do not
    /// describe the same model. The payload says which count disagreed.
    RanefShapeMismatch(String),
    /// The formula removes the intercept and names no fixed-effect term, so the
    /// design would have zero columns.
    EmptyDesign,
    /// A transform term evaluated to a non-finite value (e.g. `log(x)` on a
    /// non-positive `x`, which R would silently drop via `na.omit`), or the
    /// response, a design column or an `offset()` term holds one (NaN
    /// included: missing rows are dropped before lowering). Each is an error
    /// naming the row.
    TransformNotFinite {
        /// The term as spelled, e.g. `log(x)`, the response's name, or a
        /// design column's name (`x`, `x:z`).
        term: String,
        /// 0-based row of the first non-finite value.
        row: usize,
    },
    /// A `cbind(successes, failures)` response with a non-binomial family.
    CbindNeedsBinomial,
    /// A `cbind()` row whose trial count `successes + failures` is not a
    /// positive finite number — the kernel's prior weights must be `> 0`.
    ZeroTrials {
        /// 0-based row.
        row: usize,
    },
    /// A `cbind()` row with a negative success or failure count. Only the sum
    /// is checked by [`Error::ZeroTrials`]; a negative numerator would pass it
    /// and lower to a proportion outside `[0, 1]`, which the binomial deviance
    /// accepts without complaint (R's `glm` stops with "negative counts").
    NegativeCount {
        /// 0-based row.
        row: usize,
    },
    /// The kernel refused the lowered input ([`fit`](super::fit()),
    /// [`fit_lowered`](super::fit_lowered)), with the [`crate::FitError`]
    /// [`crate::fit`] returns.
    Fit(crate::FitError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(e) => write!(f, "{e}"),
            Error::UnknownColumn(name) => write!(f, "unknown column: {name}"),
            Error::ResponseNotNumeric(name) => {
                write!(f, "response column '{name}' is not numeric")
            }
            Error::WrongColumnKind { name, expected } => {
                write!(f, "column '{name}' is not {expected}")
            }
            Error::SingleLevelFactor { name } => write!(
                f,
                "column '{name}': contrasts can be applied only to factors with 2 or more levels"
            ),
            Error::RandomColumnRepeated { group, column } => write!(
                f,
                "random-effect column '{column}' is in two terms on grouping '{group}'; \
                 put each column in one term, for example (1 + x + f | g) or \
                 (x + f || g) instead of (1 + x | g) + (0 + f | g)"
            ),
            Error::RanefShapeMismatch(detail) => write!(
                f,
                "the fit and the lowered random effects do not describe the same model: {detail}"
            ),
            Error::EmptyDesign => write!(
                f,
                "the fixed design has no columns: '- 1' / '0 +' with no fixed-effect term"
            ),
            Error::TransformNotFinite { term, row } => {
                write!(f, "term '{term}' is not finite at row {row}")
            }
            Error::CbindNeedsBinomial => {
                write!(f, "cbind(successes, failures) needs family = binomial")
            }
            Error::ZeroTrials { row } => write!(
                f,
                "cbind(): successes + failures must be positive and finite at every row; row {row} is not"
            ),
            Error::NegativeCount { row } => write!(
                f,
                "cbind(): successes and failures must be non-negative at every row; row {row} is not"
            ),
            Error::Fit(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for Error {}
impl std::error::Error for ParseError {}

impl From<ParseError> for Error {
    fn from(e: ParseError) -> Self {
        Error::Parse(e)
    }
}
