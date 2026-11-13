//! One `run` function per registered function: read the arguments, call
//! `glmm::orchestrate`, convert the result.

pub(crate) mod compare;
pub(crate) mod fit;
pub(crate) mod formula;
pub(crate) mod inference;
pub(crate) mod predict;

use glmm::orchestrate::PortError;

use crate::error::BindError;
use crate::inputs::{FactorColumns, Inputs, NumericColumns};

/// How an error names a row: `row_index`, the caller's number of each row
/// passed, else the row passed counted from `row_base` (0 from Python, 1
/// from R).
pub(crate) struct Rows {
    index: Option<Vec<i64>>,
    base: usize,
}

impl Rows {
    pub(crate) fn read(inp: &mut Inputs) -> Result<Self, BindError> {
        Ok(Self {
            index: inp.opt_i64s("row_index")?,
            base: inp.usize("row_base")?,
        })
    }

    pub(crate) fn base(&self) -> usize {
        self.base
    }

    fn shown(&self, row: usize) -> usize {
        self.index
            .as_ref()
            .and_then(|ix| ix.get(row))
            .and_then(|&k| usize::try_from(k).ok())
            .unwrap_or(row + self.base)
    }

    /// `e` as an `invalid_argument` (a refused option or start value) or a
    /// `fit_error`, naming the caller's row.
    pub(crate) fn error(&self, e: PortError) -> BindError {
        let code = if e.argument {
            "invalid_argument"
        } else {
            "fit_error"
        };
        match e.row {
            Some(row) => {
                let shown = self.shown(row);
                BindError {
                    row: Some(shown),
                    ..BindError::new(code, e.message_with_row(shown))
                }
            }
            None => BindError::new(code, e.message),
        }
    }
}

/// Refuses data whose columns all have zero rows, with the kernel's text
/// (`fit::common`, and `formula::materialize` repeats it: the three change
/// together), before lowering: a factor column would otherwise reach the
/// lowering's single-level-factor refusal first.
pub(crate) fn check_rows(
    numeric: &NumericColumns,
    factors: &FactorColumns,
) -> Result<(), BindError> {
    let mut lens = numeric
        .values()
        .map(Vec::len)
        .chain(factors.values().map(|(_, codes)| codes.len()));
    match lens.next() {
        Some(0) if lens.all(|n| n == 0) => Err(BindError::new(
            "fit_error",
            "Data has no rows (n = 0); a fit needs at least one row",
        )),
        _ => Ok(()),
    }
}

/// Refuses a formula with nothing before its `~` (or with no `~` or `=` at
/// all), in the parser's words. The parser names the missing response by a
/// placeholder column, which `compare` would otherwise show when it compares
/// the two responses. An empty formula is left to the parser.
pub(crate) fn check_response(formula: &str) -> Result<(), BindError> {
    let lhs = formula
        .split_once('~')
        .or_else(|| formula.split_once('='))
        .map(|(lhs, _)| lhs);
    if !formula.trim().is_empty() && lhs.is_none_or(|l| l.trim().is_empty()) {
        return Err(BindError::invalid_argument(
            "formula syntax error: the formula has no response; write it as y ~ x",
        ));
    }
    Ok(())
}

/// A stored fit's training data, as `fit` took it.
pub(crate) struct ModelData {
    pub formula: String,
    pub numeric: NumericColumns,
    pub factors: FactorColumns,
    pub family: String,
    pub link: String,
    pub weights: Option<Vec<f64>>,
    pub offset: Option<Vec<f64>>,
}

impl ModelData {
    pub(crate) fn read(inp: &mut Inputs) -> Result<Self, BindError> {
        Ok(Self {
            formula: inp.string("formula")?,
            numeric: inp.numeric("numeric")?,
            factors: inp.factors("factors")?,
            family: inp.string("family")?,
            link: inp.string("link")?,
            weights: inp.opt_vec("weights")?,
            offset: inp.opt_vec("offset")?,
        })
    }
}
