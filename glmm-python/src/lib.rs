//! PyO3 binding: builds the `glmm._native` extension module. Almost all logic
//! lives in `convert.rs`/`orchestrate.rs`, which never touch a `pyo3` type and
//! run under plain `cargo test` — this file is only the FFI shim.

mod convert;
mod orchestrate;

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// `FitResult` → a `dict` keyed by field name, unpacked in `glmm/__init__.py::fit`.
///
/// A dict rather than a positional tuple, and do not flatten it back to one:
/// PyO3 implements `IntoPyObject` for tuples only up to 12 elements and this
/// carries more, and a positional tuple is precisely what let `re_groups`,
/// `n_eval`, and `deviance` sit on `glmm::Fit` without ever crossing — naming
/// each field at both ends makes an omission a `KeyError` instead of silence.
/// Every field of `glmm::Fit` belongs here; one with no Python home is a bug.
fn fit_dict<'py>(py: Python<'py>, r: orchestrate::FitResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("beta", r.beta)?;
    d.set_item("se", r.se)?;
    d.set_item("vcov", r.vcov)?;
    d.set_item("tau2", r.tau2)?;
    d.set_item("varcorr", r.varcorr)?;
    d.set_item("stddev_se", r.stddev_se)?;
    d.set_item("aliased", r.aliased)?;
    d.set_item("dispersion", r.dispersion)?;
    d.set_item("converged", r.converged)?;
    d.set_item("n_eval", r.n_eval)?;
    d.set_item("deviance", r.deviance)?;
    d.set_item("singular", r.singular)?;
    d.set_item("names", r.names)?;
    d.set_item("re_groups", r.re_groups)?;
    d.set_item("agq_warning", r.agq_warning)?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    formula, numeric_columns, factor_columns, family, link, wald_se, nagq,
    dispersion, weights, warm_start,
))]
fn fit<'py>(
    py: Python<'py>,
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    // Each factor arrives as (levels, per-row codes): `glmm.fit` states the
    // level order, so the caller's reference level survives instead of being
    // re-derived by a sort here. See `glmm::formula::Column::Factor`.
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: &str,
    link: &str,
    wald_se: &str,
    nagq: u8,
    dispersion: Option<f64>,
    weights: Option<Vec<f64>>,
    warm_start: Option<(Vec<f64>, Vec<f64>)>,
) -> PyResult<Bound<'py, PyDict>> {
    let result = orchestrate::run_fit(
        formula,
        numeric_columns,
        factor_columns,
        family,
        link,
        wald_se,
        nagq,
        dispersion,
        weights,
        warm_start,
    )
    .map_err(PyValueError::new_err)?;
    fit_dict(py, result)
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    Ok(())
}
