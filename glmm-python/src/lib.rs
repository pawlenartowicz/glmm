//! PyO3 binding: builds the `glmm._native` extension module. All logic lives
//! in `glmm::orchestrate` (the crate's shared port-orchestration module,
//! behind its `orchestrate` feature) — this file is only the FFI shim.

use std::collections::HashMap;

use glmm::orchestrate;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

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
    d.set_item("loglik", r.loglik)?;
    d.set_item("df", r.df)?;
    d.set_item("reml", r.reml)?;
    d.set_item("fitted", r.fitted)?;
    d.set_item("ranef", r.ranef)?;
    d.set_item("ranef_levels", r.ranef_levels)?;
    d.set_item("y", r.y)?;
    d.set_item("weights", r.weights)?;
    d.set_item("nobs", r.nobs)?;
    // Labelled blocks as their own dicts, same reason the notes below are: the
    // Python layer reshapes each into `(n_levels, n_terms)` and never has to
    // know a positional order.
    let blocks = PyList::empty(py);
    for (group, terms, levels, values) in r.ranef_blocks {
        let bd = PyDict::new(py);
        bd.set_item("group", group)?;
        bd.set_item("terms", terms)?;
        bd.set_item("levels", levels)?;
        bd.set_item("values", values)?;
        blocks.append(bd)?;
    }
    d.set_item("ranef_blocks", blocks)?;
    d.set_item("boundary", r.boundary)?;
    d.set_item("pinned", r.pinned)?;
    // Each note as its own dict, keyed the same way the outer one is: the
    // `kind` string is what `glmm/__init__.py` maps to a warning category, so
    // an unrecognized kind stays readable instead of decoding to a position.
    let notes = PyList::empty(py);
    for n in r.notes {
        let nd = PyDict::new(py);
        nd.set_item("kind", n.kind)?;
        nd.set_item("columns", n.columns)?;
        nd.set_item("pivot", n.pivot)?;
        nd.set_item("evals", n.evals)?;
        nd.set_item("final_eval", n.final_eval)?;
        nd.set_item("detail", n.detail)?;
        nd.set_item("ratio", n.ratio)?;
        notes.append(nd)?;
    }
    d.set_item("notes", notes)?;
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    formula, numeric_columns, factor_columns, family, link, wald_se, nagq,
    dispersion, weights, offset, warm_start,
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
    offset: Option<Vec<f64>>,
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
        offset,
        warm_start,
    )
    .map_err(PyValueError::new_err)?;
    fit_dict(py, result)
}

/// What `formula` reads from a dataset whose column names are `available`, from
/// one parse: `(used, missing, numeric_only)`.
///
/// - `used`: the names in `available` the formula reads as data columns
///   (`glmm::formula::referenced_columns`), so `glmm/__init__.py` never has to
///   re-implement the identifier grammar or the transform/`cbind()`/`offset()`
///   spellings to filter `data` down to the columns a fit needs.
/// - `missing`: names the formula references that resolve against neither
///   `available` nor a transform of one of its members
///   (`glmm::formula::unresolved_columns`) — what `glmm.fit` would fail on as
///   an unknown column. Lets `glmm/__init__.py` raise that error before it
///   checks `weights=`/`offset=` against the row count.
/// - `numeric_only`: names among `available` that always stay plain numbers,
///   the response and the `offset()` column (`glmm::formula::numeric_only_columns`)
///   — a bool predictor or grouping column crosses as a `FALSE`/`TRUE` factor,
///   but these never do.
///
/// A formula that fails to parse raises here exactly as `fit()` itself would,
/// through the same error conversion.
#[pyfunction]
fn formula_columns(
    formula: &str,
    available: Vec<String>,
) -> PyResult<(Vec<String>, Vec<String>, Vec<String>)> {
    let ast = glmm::formula::parse(formula).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        glmm::formula::referenced_columns(&ast, &available),
        glmm::formula::unresolved_columns(&ast, &available),
        glmm::formula::numeric_only_columns(&ast, &available),
    ))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(formula_columns, m)?)?;
    Ok(())
}
