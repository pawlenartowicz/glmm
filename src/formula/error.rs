//! The crate's error surface. `ParseError`'s six variants and their Display
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
    /// A token on the RHS is not a valid identifier.
    Syntax {
        /// 0-based position of the offending token in the formula string.
        pos: usize,
        /// Description of what's wrong with the token.
        msg: String,
    },
    /// A `-` term removal outside parentheses (unsupported).
    TermRemovalUnsupported,
    /// The same grouping factor appears in two random-effect terms.
    DuplicateGroupingVar {
        /// The grouping factor name that appears more than once.
        name: String,
    },
    /// A slope term `(1+|g)` with no slope variables.
    EmptySlopeTerm {
        /// The grouping factor of the empty slope term.
        group: String,
    },
    /// Intercept suppression `(0+…|g)` / `(-1+…|g)` in a random-effect term.
    RandomInterceptSuppressionUnsupported,
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
            ParseError::DuplicateGroupingVar { name } => {
                write!(f, "duplicate grouping variable: {name}")
            }
            ParseError::EmptySlopeTerm { group } => {
                write!(f, "empty slope term for group {group}")
            }
            ParseError::RandomInterceptSuppressionUnsupported => write!(
                f,
                "a random slope requires a random intercept in this engine version; \
                 intercept suppression ('0 +' / '-1 +') in a random-effects term is \
                 not supported — write '(x | g)' or '(1 + x | g)'"
            ),
        }
    }
}

/// Everything `parse` or `materialize`/`lower` can reject.
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
        /// The kind the column was expected to be (`"numeric"` or `"factor"`).
        expected: &'static str,
    },
    /// A random-slope variable is not present as a (numeric) fixed term, so it
    /// has no `ColumnId` in the design. A future version may allow
    /// random-slope variables with no corresponding fixed-effect term.
    SlopeVarNotInDesign(String),
    /// [`super::label_ranef`] was handed a `Fit` and a `re_groups` that do not
    /// describe the same model. The payload says which count disagreed.
    RanefShapeMismatch(String),
    /// The formula removes the intercept and names no fixed-effect term, so the
    /// design would have zero columns.
    EmptyDesign,
    /// A transform term evaluated to a non-finite value (e.g. `log(x)` on a
    /// non-positive `x`). R would silently drop the row via `na.omit`; here it
    /// is an error naming the row.
    TransformNotFinite {
        /// The term as spelled, e.g. `log(x)`.
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
            Error::SlopeVarNotInDesign(name) => write!(
                f,
                "random-slope variable '{name}' is not a numeric fixed term in the design"
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
