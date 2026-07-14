//! PyO3 binding: builds the `glmm._native` extension module. Almost all logic
//! lives in `convert.rs`/`orchestrate.rs`, which never touch a `pyo3` type and
//! run under plain `cargo test` — this file is only the FFI shim.

mod convert;
mod orchestrate;

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    formula, numeric_columns, factor_columns, family, link, wald_se, nagq,
    dispersion, weights, targets, warm_start,
))]
fn fit(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, Vec<String>>,
    family: &str,
    link: &str,
    wald_se: &str,
    nagq: u8,
    dispersion: Option<f64>,
    weights: Option<Vec<f64>>,
    targets: Option<Vec<String>>,
    warm_start: Option<(Vec<f64>, Vec<f64>)>,
) -> PyResult<orchestrate::FitTuple> {
    orchestrate::run_fit(
        formula,
        numeric_columns,
        factor_columns,
        family,
        link,
        wald_se,
        nagq,
        dispersion,
        weights,
        targets,
        warm_start,
    )
    .map(orchestrate::FitResult::into_tuple)
    .map_err(PyValueError::new_err)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    Ok(())
}
