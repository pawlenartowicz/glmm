//! PyO3 binding: builds the `glmm._native` extension module. All logic lives
//! in `glmm::orchestrate` (the crate's shared port-orchestration module,
//! behind its `orchestrate` feature) — this file is only the FFI shim.

use std::collections::HashMap;

use glmm::formula::anova::{self, CompareError, Comparison, ComparisonTest, Differs, Model, Side};
use glmm::inference::{DenDf, Reference};
use glmm::orchestrate;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyBytes, PyDict, PyFloat, PyList, PyString, PyTuple};

/// A per-row vector sent from Python as the raw bytes of a native-endian numpy
/// array (`ndarray.tobytes()`), the counterpart of `values_bytes` below. Bytes
/// rather than a list: building and extracting one Python float per row is a
/// large share of a large Gaussian model's fit time. Refuses a length that is
/// not a whole number of `N`-byte values, which only a caller other than
/// `glmm/__init__.py` can send.
fn from_bytes<T, const N: usize>(
    b: &[u8],
    what: &str,
    decode: fn([u8; N]) -> T,
) -> PyResult<Vec<T>> {
    let chunks = b.chunks_exact(N);
    if !chunks.remainder().is_empty() {
        return Err(PyValueError::new_err(format!(
            "{what}: {} bytes is not a whole number of {N}-byte values",
            b.len()
        )));
    }
    Ok(chunks
        .map(|c| decode(c.try_into().expect("chunks_exact yields N bytes")))
        .collect())
}

type NumericColumns = HashMap<String, Vec<f64>>;
type FactorColumns = HashMap<String, (Vec<String>, Vec<u32>)>;

/// The columns `glmm/__init__.py::_prepare_columns` sends: numeric columns as
/// `float64` bytes, factor codes as `uint32` bytes beside their levels.
fn columns(
    numeric: HashMap<String, PyBackedBytes>,
    factor: HashMap<String, (Vec<String>, PyBackedBytes)>,
) -> PyResult<(NumericColumns, FactorColumns)> {
    let numeric = numeric
        .into_iter()
        .map(|(name, b)| {
            let v = from_bytes(&b, &format!("column {name:?}"), f64::from_ne_bytes)?;
            Ok((name, v))
        })
        .collect::<PyResult<_>>()?;
    let factor = factor
        .into_iter()
        .map(|(name, (levels, b))| {
            let codes = from_bytes(&b, &format!("column {name:?}"), u32::from_ne_bytes)?;
            Ok((name, (levels, codes)))
        })
        .collect::<PyResult<_>>()?;
    Ok((numeric, factor))
}

fn opt_f64s(b: Option<PyBackedBytes>, what: &str) -> PyResult<Option<Vec<f64>>> {
    b.map(|b| from_bytes(&b, what, f64::from_ne_bytes))
        .transpose()
}

/// Per-row output as the raw bytes of a native-endian array, read back by
/// `np.frombuffer` in `glmm/__init__.py::_fit_from_native`, for the same
/// reason inputs arrive as bytes (`from_bytes`).
fn values_bytes<'py, T: Copy, const N: usize>(
    py: Python<'py>,
    v: &[T],
    encode: fn(T) -> [u8; N],
) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, v.len() * N, |buf| {
        for (c, &x) in buf.chunks_exact_mut(N).zip(v) {
            c.copy_from_slice(&encode(x));
        }
        Ok(())
    })
}

fn f64_bytes<'py>(py: Python<'py>, v: &[f64]) -> PyResult<Bound<'py, PyBytes>> {
    values_bytes(py, v, f64::to_ne_bytes)
}

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
    d.set_item("n_grad", r.n_grad)?;
    d.set_item("outer_stop", r.outer_stop)?;
    d.set_item("outer_decrement", r.outer_decrement)?;
    d.set_item("deviance", r.deviance)?;
    d.set_item("singular", r.singular)?;
    d.set_item("names", r.names)?;
    d.set_item("re_groups", r.re_groups)?;
    d.set_item("agq_warning", r.agq_warning)?;
    d.set_item("reml_warning", r.reml_warning)?;
    d.set_item("loglik", r.loglik)?;
    d.set_item("df", r.df)?;
    d.set_item("reml", r.reml)?;
    d.set_item("fitted", f64_bytes(py, &r.fitted)?)?;
    d.set_item("ranef", f64_bytes(py, &r.ranef)?)?;
    d.set_item(
        "ranef_levels",
        values_bytes(py, &r.ranef_levels, |n| (n as u64).to_ne_bytes())?,
    )?;
    d.set_item("y", f64_bytes(py, &r.y)?)?;
    let weights = r.weights.as_deref().map(|w| f64_bytes(py, w)).transpose()?;
    d.set_item("weights", weights)?;
    // `(counts, s, c)` as a dict of three flat lists, `None` outside the
    // `(1 | g)` envelope.
    match r.design_summary {
        Some((counts, s, c)) => {
            let ds = PyDict::new(py);
            ds.set_item("counts", counts)?;
            ds.set_item("s", s)?;
            ds.set_item("c", c)?;
            d.set_item("design_summary", ds)?;
        }
        None => d.set_item("design_summary", py.None())?,
    }
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
        notes.append(nd)?;
    }
    d.set_item("notes", notes)?;
    d.set_item("coef_df", r.coef_df)?;
    d.set_item("coef_p", r.coef_p)?;
    d.set_item("df_method", r.df_method)?;
    d.set_item("rss", r.rss)?;
    d.set_item("r_squared", r.r_squared)?;
    d.set_item("adj_r_squared", r.adj_r_squared)?;
    let offset = r.offset.as_deref().map(|o| f64_bytes(py, o)).transpose()?;
    d.set_item("offset", offset)?;
    let (objective, nodes) = orchestrate::objective_parts(r.objective);
    d.set_item("objective", objective)?;
    d.set_item("objective_nodes", nodes)?;
    d.set_item("term_table_type3", term_dict(py, &r.term_table_type3)?)?;
    d.set_item("term_table_type2", term_dict(py, &r.term_table_type2)?)?;
    // `None` unless the fit asked for the likelihood-ratio tables.
    let lrt = |t: &Option<Result<glmm::formula::anova::TermTable, String>>| {
        t.as_ref().map(|t| term_dict(py, t)).transpose()
    };
    d.set_item("term_table_type3_lrt", lrt(&r.term_table_type3_lrt)?)?;
    d.set_item("term_table_type2_lrt", lrt(&r.term_table_type2_lrt)?)?;
    // What `"auto"` gives on this fit: {"test", "reason"}.
    let (test, reason) = orchestrate::choice_tags(&r.term_test_auto);
    let auto = PyDict::new(py);
    auto.set_item("test", test)?;
    auto.set_item("reason", reason)?;
    d.set_item("term_test_auto", auto)?;
    Ok(d)
}

/// A stored per-term result as a `dict`: `error` is the refusal's message,
/// or `None` beside the table's columns (`orchestrate::TermColumns`, one list
/// entry per row).
fn term_dict<'py>(
    py: Python<'py>,
    result: &Result<glmm::formula::anova::TermTable, String>,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    match result {
        Err(message) => d.set_item("error", message)?,
        Ok(table) => {
            let c = orchestrate::term_columns(table);
            d.set_item("error", py.None())?;
            d.set_item("ss_type", c.ss_type)?;
            d.set_item("test", c.test)?;
            d.set_item("reason", c.reason)?;
            d.set_item("label", c.label)?;
            d.set_item("kind", c.kind)?;
            d.set_item("statistic", c.statistic)?;
            d.set_item("num_df", c.num_df)?;
            d.set_item("den_df", c.den_df)?;
            d.set_item("den_df_kind", c.den_df_kind)?;
            d.set_item("p_value", c.p_value)?;
            d.set_item("sum_sq", c.sum_sq)?;
            d.set_item("mean_sq", c.mean_sq)?;
            d.set_item("row_notes", c.row_notes)?;
            d.set_item("notes", c.notes)?;
        }
    }
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    formula, numeric_columns, factor_columns, family, link, wald_se, nagq,
    dispersion, weights, offset, warm_start, satterthwaite, reml, anova_test = "auto",
))]
fn fit<'py>(
    py: Python<'py>,
    formula: &str,
    numeric_columns: HashMap<String, PyBackedBytes>,
    // Each factor arrives as (levels, per-row codes): `glmm.fit` states the
    // level order, so the caller's reference level survives instead of being
    // re-derived by a sort here. See `glmm::formula::Column::Factor`.
    factor_columns: HashMap<String, (Vec<String>, PyBackedBytes)>,
    family: &str,
    link: &str,
    wald_se: &str,
    nagq: u8,
    dispersion: Option<f64>,
    weights: Option<PyBackedBytes>,
    offset: Option<PyBackedBytes>,
    warm_start: Option<(Vec<f64>, Vec<f64>, Option<f64>)>,
    satterthwaite: bool,
    reml: Option<bool>,
    // "auto", "Wald" or "LRT": "LRT" also stores the likelihood-ratio tables.
    anova_test: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (numeric_columns, factor_columns) = columns(numeric_columns, factor_columns)?;
    let weights = opt_f64s(weights, "weights")?;
    let offset = opt_f64s(offset, "offset")?;
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
        satterthwaite,
        reml,
        anova_test,
    )
    .map_err(PyValueError::new_err)?;
    fit_dict(py, result)
}

/// The crate's refusal of `glmm.anova(m, test=...)` asking for `"Wald"` or
/// `"LRT"` rows of a fit whose per-term rows are F tests, so both ports give
/// one text. `den_df_kind` is the F rows' df source, `"satterthwaite"` or
/// `"residual"`, as a stored table names it.
#[pyfunction]
fn f_table_refusal(asked: &str, den_df_kind: &str) -> PyResult<String> {
    let asked = match asked {
        "Wald" => anova::TermTest::Wald,
        "LRT" => anova::TermTest::Lrt,
        other => {
            return Err(PyValueError::new_err(format!(
                "no refusal for test {other:?}"
            )))
        }
    };
    let den_df = match den_df_kind {
        "satterthwaite" => DenDf::Satterthwaite,
        "residual" => DenDf::Residual,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown df source {other:?}"
            )))
        }
    };
    Ok(anova::TermTestError::FTable { asked, den_df }.to_string())
}

/// `glmm::orchestrate::confint`: per coefficient `[lower, upper]`, the t
/// quantile on a finite df and the normal one elsewhere. `Fit.confint` calls
/// it, so both ports give the same intervals from one rule.
#[pyfunction]
fn confint(beta: Vec<f64>, se: Vec<f64>, coef_df: Vec<f64>, level: f64) -> PyResult<Vec<[f64; 2]>> {
    orchestrate::confint(&beta, &se, &coef_df, level).map_err(PyValueError::new_err)
}

/// One side of `glmm.anova(m0, m1)`: what a stored `glmm.Fit` keeps, keyed as
/// `glmm/anova.py::_side` sends it, read into an `anova::Model` with no refit.
#[derive(FromPyObject)]
#[pyo3(from_item_all)]
struct StoredFit {
    formula: String,
    family: String,
    link: String,
    loglik: f64,
    df: usize,
    nobs: usize,
    reml: bool,
    objective: String,
    objective_nodes: u8,
    converged: bool,
    varcorr: Vec<Vec<f64>>,
    re_groups: Vec<(String, Vec<String>)>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    dispersion_held: Option<f64>,
    design_summary: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    rss: f64,
    coef_names: Vec<String>,
    coef: Vec<f64>,
    vcov: Vec<Vec<f64>>,
    aliased: Vec<bool>,
}

impl StoredFit {
    fn model(self) -> PyResult<Model> {
        Ok(Model {
            family: orchestrate::family_from_str(&self.family, &self.link)
                .map_err(PyValueError::new_err)?,
            objective: orchestrate::objective_from_parts(&self.objective, self.objective_nodes)
                .map_err(PyValueError::new_err)?,
            formula: self.formula,
            loglik: self.loglik,
            df: self.df,
            nobs: self.nobs,
            reml: self.reml,
            converged: self.converged,
            varcorr: self.varcorr,
            re_groups: self.re_groups,
            y: self.y,
            weights: self.weights,
            offset: self.offset,
            dispersion_held: self.dispersion_held,
            design_summary: self.design_summary,
            rss: self.rss,
            coef_names: self.coef_names,
            coef: self.coef,
            vcov: self.vcov,
            aliased: self.aliased,
        })
    }
}

/// A refusal of `glmm.anova(m0, m1)` in Python's words: the crate's text says
/// what is wrong and what to do with no call syntax, and the seven refusals
/// that `glmm.compare` avoids gain the pointer to it (and to `reml=False`).
fn refusal(e: &CompareError) -> String {
    const COMPARE: &str = "glmm.compare(data, f0, f1)";
    let pointer = match e {
        CompareError::DifferentRows { .. } => {
            format!("Or use {COMPARE}, which drops those rows once for both fits.")
        }
        CompareError::DifferentY => format!("Or use {COMPARE}, which fits both to the same data."),
        CompareError::DifferentWeights => {
            format!("Or use {COMPARE}, which takes the weights once.")
        }
        CompareError::DifferentOffset => format!("Or use {COMPARE}, which takes the offset once."),
        CompareError::DifferentFamily { .. } => {
            format!("Or use {COMPARE}, which takes the family and link once.")
        }
        CompareError::MixedEstimators { .. } => {
            format!("Or use {COMPARE}, which picks the estimator itself.")
        }
        CompareError::RemlDifferentFixed => {
            format!("Pass reml=False to both fits, or use {COMPARE}, which fits both by ML.")
        }
        _ => return e.to_string(),
    };
    format!("{e} {pointer}")
}

/// A [`Comparison`] as a `dict` for `glmm/anova.py`: `sides` holds `m0` and
/// `m1` as given (each with its residual sum of squares from `rss`, NaN off
/// OLS), `larger` the index of the larger one, and the test, its reference
/// and notes beside the `nsim` and `seed` the call used.
fn comparison_dict<'py>(
    py: Python<'py>,
    c: &Comparison,
    rss: [f64; 2],
    nsim: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let sides = PyList::empty(py);
    for (info, rss) in [(&c.m0, rss[0]), (&c.m1, rss[1])] {
        let s = PyDict::new(py);
        s.set_item("formula", &info.formula_label)?;
        s.set_item("npar", info.npar)?;
        s.set_item("nobs", info.nobs)?;
        s.set_item("loglik", info.loglik)?;
        s.set_item("aic", info.aic)?;
        s.set_item("bic", info.bic)?;
        s.set_item("rss", rss)?;
        sides.append(s)?;
    }
    d.set_item("sides", sides)?;
    d.set_item(
        "larger",
        match c.larger {
            Side::First => 0,
            Side::Second => 1,
        },
    )?;
    d.set_item(
        "differs",
        match c.differs {
            Differs::Fixed => "fixed",
            Differs::Random => "random",
            Differs::Both => "both",
        },
    )?;
    let (test, statistic, df, p_value, reference) = match &c.test {
        ComparisonTest::Lr(t) => {
            let (objective, nodes) = orchestrate::objective_parts(t.objective);
            d.set_item("objective", objective)?;
            d.set_item("objective_nodes", nodes)?;
            ("lr", t.statistic, t.df as f64, t.p_value, t.reference)
        }
        ComparisonTest::Linear(t) => ("f", t.statistic, t.num_df, t.p_value, t.reference),
        other => {
            return Err(PyValueError::new_err(format!(
                "no Python rendering for the comparison test {other:?}"
            )))
        }
    };
    d.set_item("test", test)?;
    d.set_item("statistic", statistic)?;
    d.set_item("df", df)?;
    d.set_item("p_value", p_value)?;
    // The exact reference's `nsim` is the draws made (0 at a zero statistic),
    // not the draws requested, which `nsim` below carries.
    let r = PyDict::new(py);
    match reference {
        Reference::ChiSquared { df } => {
            r.set_item("kind", "chisq")?;
            r.set_item("df", df)?;
        }
        Reference::ChiBarSquared { df_low } => {
            r.set_item("kind", "chibar")?;
            r.set_item("df", df_low)?;
        }
        Reference::ExactRlrt { nsim, seed } => {
            r.set_item("kind", "exact")?;
            r.set_item("nsim", nsim)?;
            r.set_item("seed", seed)?;
        }
        Reference::F { num_df, den_df, .. } => {
            r.set_item("kind", "f")?;
            r.set_item("df", num_df)?;
            r.set_item("den_df", den_df)?;
        }
        other => {
            return Err(PyValueError::new_err(format!(
                "no Python rendering for the reference {other:?}"
            )))
        }
    }
    d.set_item("reference", r)?;
    d.set_item("notes", c.note_lines())?;
    d.set_item("nsim", nsim)?;
    d.set_item("seed", seed)?;
    Ok(d)
}

/// `glmm.anova(m0, m1)`: the two stored fits compared as they are, with no
/// refit (`glmm::formula::anova::compare`). `nsim` and `seed` are the exact
/// test's, `None` taking `orchestrate::compare_options`' defaults.
#[pyfunction]
#[pyo3(signature = (m0, m1, nsim, seed))]
fn compare_fits<'py>(
    py: Python<'py>,
    m0: StoredFit,
    m1: StoredFit,
    nsim: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    let opts = orchestrate::compare_options(nsim, seed).map_err(PyValueError::new_err)?;
    let rss = [m0.rss, m1.rss];
    let (a, b) = (m0.model()?, m1.model()?);
    let c = anova::compare(&a, &b, &opts).map_err(|e| PyValueError::new_err(refusal(&e)))?;
    comparison_dict(py, &c, rss, opts.nsim, opts.seed)
}

/// `glmm.compare(data, f0, f1)`: `orchestrate::run_compare` on columns the
/// caller has already masked to the rows both formulas can use. The comparison
/// as `comparison_dict` gives it, plus `fit0` and `fit1` as `fit` gives them.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (
    f0, f1, numeric_columns, factor_columns, family, link, nagq, weights, offset, nsim, seed,
))]
fn compare<'py>(
    py: Python<'py>,
    f0: &str,
    f1: &str,
    numeric_columns: HashMap<String, PyBackedBytes>,
    factor_columns: HashMap<String, (Vec<String>, PyBackedBytes)>,
    family: &str,
    link: &str,
    nagq: u8,
    weights: Option<PyBackedBytes>,
    offset: Option<PyBackedBytes>,
    nsim: Option<usize>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    let (numeric_columns, factor_columns) = columns(numeric_columns, factor_columns)?;
    let weights = opt_f64s(weights, "weights")?;
    let offset = opt_f64s(offset, "offset")?;
    let r = orchestrate::run_compare(
        f0,
        f1,
        numeric_columns,
        factor_columns,
        family,
        link,
        nagq,
        weights,
        offset,
        nsim,
        seed,
    )
    .map_err(PyValueError::new_err)?;
    let d = comparison_dict(py, &r.comparison, [r.fit0.rss, r.fit1.rss], r.nsim, r.seed)?;
    d.set_item("fit0", fit_dict(py, r.fit0)?)?;
    d.set_item("fit1", fit_dict(py, r.fit1)?)?;
    Ok(d)
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

/// The per-element pass of `glmm/__init__.py::_plain_column` over a plain list,
/// which decides everything else (the type sets, what is missing, the cutoff,
/// the fallback). Returns:
///
/// - `("num", values)` when every element's type is one of `number_types`:
///   `values` the float64 bytes, each element converted as
///   `np.array(elements, dtype=float)` converts it (`None` is NaN, the rest go
///   through `PyFloat_AsDouble`, which rounds an int correctly).
/// - `("str", codes, keys)` when every element is a string whose type is one of
///   `string_types`, or missing: `None`, or a NaN whose type is a float type in
///   `number_types`. `keys` are the distinct strings as the original objects,
///   in first-seen order; when a row is missing, one `None` key follows them
///   and every missing row takes its code. `codes` are intp bytes.
/// - `None` for anything else, including an int too large for a double and a
///   string with no UTF-8 form (a lone surrogate): the caller then runs its own
///   per-element code, which raises or converts as it always has.
///
/// Types match by identity, as `type(v) in _NUMBER_TYPES` does, so a subclass
/// is never matched and no element's own Python code runs. Strings are keyed
/// by their UTF-8 bytes, so `"a"` and `np.str_("a")` are one key, as they are
/// in a dict.
#[pyfunction]
fn plain_column<'py>(
    py: Python<'py>,
    elements: &Bound<'py, PyList>,
    number_types: &Bound<'py, PyTuple>,
    string_types: &Bound<'py, PyTuple>,
) -> PyResult<Option<Bound<'py, PyTuple>>> {
    if elements.is_empty() {
        return Ok(None);
    }
    let ptrs = |types: &Bound<'py, PyTuple>| types.iter().map(|t| t.as_ptr()).collect::<Vec<_>>();
    let (numbers, strings) = (ptrs(number_types), ptrs(string_types));
    let type_of = |v: &Bound<'py, PyAny>| v.get_type_ptr().cast::<pyo3::ffi::PyObject>();

    let mut values = Vec::with_capacity(elements.len());
    for v in elements.iter() {
        if !numbers.contains(&type_of(&v)) {
            break;
        }
        if v.is_none() {
            values.push(f64::NAN);
        } else if let Ok(x) = v.extract::<f64>() {
            values.push(x);
        } else {
            return Ok(None);
        }
    }
    if values.len() == elements.len() {
        return Ok(Some(("num", f64_bytes(py, &values)?).into_pyobject(py)?));
    }

    // Each distinct string allocates its key once; the codes depend only on
    // first-seen order, so the hasher does not matter.
    let mut index: HashMap<Box<str>, isize> = HashMap::new();
    let keys = PyList::empty(py);
    let mut codes = Vec::with_capacity(elements.len());
    let mut any_missing = false;
    for v in elements.iter() {
        let t = type_of(&v);
        if strings.contains(&t) {
            let Some(s) = v.cast::<PyString>().ok().and_then(|s| s.to_str().ok()) else {
                return Ok(None);
            };
            let code = match index.get(s) {
                Some(&code) => code,
                None => {
                    let code = index.len() as isize;
                    index.insert(s.into(), code);
                    keys.append(&v)?;
                    code
                }
            };
            codes.push(code);
        } else if v.is_none()
            || (numbers.contains(&t) && v.cast::<PyFloat>().is_ok_and(|f| f.value().is_nan()))
        {
            codes.push(-1);
            any_missing = true;
        } else {
            return Ok(None);
        }
    }
    if any_missing {
        let code = keys.len() as isize;
        codes.iter_mut().filter(|c| **c < 0).for_each(|c| *c = code);
        keys.append(py.None())?;
    }
    let codes = values_bytes(py, &codes, isize::to_ne_bytes)?;
    Ok(Some(("str", codes, keys).into_pyobject(py)?))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(formula_columns, m)?)?;
    m.add_function(wrap_pyfunction!(plain_column, m)?)?;
    m.add_function(wrap_pyfunction!(confint, m)?)?;
    m.add_function(wrap_pyfunction!(f_table_refusal, m)?)?;
    m.add_function(wrap_pyfunction!(compare_fits, m)?)?;
    m.add_function(wrap_pyfunction!(compare, m)?)?;
    // The bound `orchestrate::compare_options` puts on `nsim`, for the
    // Python-side message on a value below 1 (which cannot cross as a usize).
    m.add("MAX_NSIM", orchestrate::MAX_NSIM)?;
    Ok(())
}
