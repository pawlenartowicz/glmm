//! The error every port renders.

use std::fmt;

/// A failed call. Each port maps `code` to its own error class: Python raises
/// `internal` as `RuntimeError` and every other code as `ValueError`, with the
/// code as its `.code`; R raises an error of class `fastglmm_error_<code>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindError {
    /// `invalid_argument` (a bad argument, found by bind or the engine),
    /// `fit_error` (lowering or fitting the caller's data failed),
    /// `compare_error` (two stored fits cannot be compared) or `internal`
    /// (a panic, or a function not implemented yet).
    pub code: &'static str,
    /// The message, the caller's row already in it.
    pub message: String,
    /// The caller's row the error names, in the caller's numbering
    /// (`row_index`, else the row passed plus `row_base`).
    pub row: Option<usize>,
}

impl BindError {
    /// An error with no row.
    #[must_use]
    pub fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            row: None,
        }
    }

    /// A bad argument.
    #[must_use]
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new("invalid_argument", message)
    }

    /// A bug: a panic or a broken invariant.
    #[must_use]
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new("internal", message)
    }
}

impl From<String> for BindError {
    /// An engine function's `String` error: a bad argument.
    fn from(message: String) -> Self {
        Self::invalid_argument(message)
    }
}

impl fmt::Display for BindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for BindError {}
