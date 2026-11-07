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
/// from R) (D7).
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

    /// `e` as a `fit_error` naming the caller's row.
    pub(crate) fn error(&self, e: PortError) -> BindError {
        match e.row {
            Some(row) => {
                let shown = self.shown(row);
                BindError {
                    row: Some(shown),
                    ..BindError::new("fit_error", e.message_with_row(shown))
                }
            }
            None => BindError::new("fit_error", e.message),
        }
    }
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
