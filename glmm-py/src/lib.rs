//! PyO3 binding: builds the `glmm._native` extension module. `call` converts
//! a dict of arguments into a `glmm_bind::Record`, runs the registered
//! function, and converts the result back; argument checks, results and error
//! codes live in `glmm-bind`.

use glmm_bind::{BindError, Record, Value};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

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

/// Per-row output as the raw bytes of a native-endian array, read back by
/// `np.frombuffer` in `glmm/__init__.py`, for the same reason inputs arrive as
/// bytes (`from_bytes`).
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

/// Arguments whose dict maps a column name to its data: their entries are
/// named `column "x"` in a bytes error.
const COLUMN_MAPS: &[&str] = &["numeric", "factors", "new_numeric", "new_factors"];

/// A Python value as a [`Value`]; `what` names it in a bytes error. Bytes are
/// float64 values, except `row_index`'s int64 ones; a `(levels, codes)` tuple
/// whose codes are uint32 bytes is a factor; a list of one scalar type is that
/// type's vector. A numpy array or scalar crosses through its `tolist()`.
fn to_value(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return Ok(Value::F64(f.value()));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::Str(s.to_str()?.to_owned()));
    }
    if obj.is_instance_of::<PyInt>() {
        return Ok(match obj.extract::<i64>() {
            Ok(n) => Value::I64(n),
            // Past i64 a seed up to 2^64 stays exact; a larger int is the
            // double `float()` gives (OverflowError past the double range).
            Err(_) => match obj.extract::<u64>() {
                Ok(n) => Value::U64(n),
                Err(_) => Value::F64(obj.extract()?),
            },
        });
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(if what == "row_index" {
            Value::IntVec(from_bytes(b.as_bytes(), what, i64::from_ne_bytes)?)
        } else {
            Value::Vec(from_bytes(b.as_bytes(), what, f64::from_ne_bytes)?)
        });
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let columns = COLUMN_MAPS.contains(&what);
        let mut rec = Record::new();
        for (k, v) in d.iter() {
            let key: String = k.extract()?;
            let inner = if columns {
                format!("column {key:?}")
            } else {
                key.clone()
            };
            rec.push(&key, to_value(&v, &inner)?);
        }
        return Ok(Value::Record(rec));
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        if t.len() == 2 {
            if let Ok(codes) = t.get_item(1)?.cast::<PyBytes>() {
                let codes = from_bytes(codes.as_bytes(), what, u32::from_ne_bytes)?;
                return Ok(Value::Factor {
                    levels: t.get_item(0)?.extract()?,
                    codes: codes
                        .into_iter()
                        .map(|c| {
                            i32::try_from(c).map_err(|_| {
                                PyValueError::new_err(format!("{what}: code {c} is too large"))
                            })
                        })
                        .collect::<PyResult<_>>()?,
                });
            }
        }
    }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
        let items = obj
            .try_iter()?
            .map(|v| to_value(&v?, what))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(typed_list(items));
    }
    if let Ok(plain) = obj.call_method0("tolist") {
        return to_value(&plain, what);
    }
    Err(PyTypeError::new_err(format!(
        "{what}: cannot pass a {} to glmm's native code",
        obj.get_type().name()?
    )))
}

/// A non-empty list of one scalar type as that type's vector (ints and
/// floats mixed as floats); anything else stays a list.
fn typed_list(items: Vec<Value>) -> Value {
    fn all<T>(items: &[Value], f: impl Fn(&Value) -> Option<T>) -> Option<Vec<T>> {
        if items.is_empty() {
            return None;
        }
        items.iter().map(f).collect()
    }
    if let Some(b) = all(&items, |v| match v {
        Value::Bool(b) => Some(*b),
        _ => None,
    }) {
        return Value::BoolVec(b);
    }
    if let Some(n) = all(&items, |v| match v {
        Value::I64(n) => Some(*n),
        _ => None,
    }) {
        return Value::IntVec(n);
    }
    if let Some(x) = all(&items, |v| match v {
        Value::F64(x) => Some(*x),
        Value::I64(n) => Some(*n as f64),
        Value::U64(n) => Some(*n as f64),
        _ => None,
    }) {
        return Value::Vec(x);
    }
    if !items.is_empty() && items.iter().all(|v| matches!(v, Value::Str(_))) {
        let strs = items.into_iter().map(|v| match v {
            Value::Str(s) => s,
            _ => unreachable!("every item is a string"),
        });
        return Value::StrVec(strs.collect());
    }
    Value::List(items)
}

/// The fit record's per-row fields `glmm/__init__.py` reads as bytes.
const FIT_BYTES: &[&str] = &[
    "fitted",
    "ranef",
    "ranef_levels",
    "y",
    "weights",
    "offset",
    "condvar",
    "condvar_x",
];

/// The fit record's `robust` fields `glmm/__init__.py` reads as bytes: the
/// ones that grow with the rows (HC2 has one cluster per row).
const ROBUST_BYTES: &[&str] = &["er", "row_e", "row_x", "row_phi"];

/// The fields of function `name`'s result that cross back as bytes.
fn bytes_fields(name: &str) -> &'static [&'static str] {
    match name {
        "fit" => FIT_BYTES,
        "predict" => &[
            "fit",
            "se",
            "lower",
            "upper",
            "future_lower",
            "future_upper",
        ],
        "model_matrix" => &["x"],
        "plain_column" => &["values", "codes"],
        _ => &[],
    }
}

/// A [`Value`] as a Python value: a record as a dict, a vector as a list, or
/// as bytes (float64, int64 for an integer vector) with `as_bytes`. `bytes`
/// names the fields of a record that cross as bytes; a fit record nested in
/// it (`compare`'s `fit0` and `fit1`) takes [`FIT_BYTES`], and a fit's
/// `robust` record [`ROBUST_BYTES`].
fn to_py<'py>(
    py: Python<'py>,
    v: Value,
    bytes: &[&str],
    as_bytes: bool,
) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, b).to_owned().into_any(),
        Value::I64(n) => n.into_pyobject(py)?.into_any(),
        Value::U64(n) => n.into_pyobject(py)?.into_any(),
        Value::F64(x) => x.into_pyobject(py)?.into_any(),
        Value::Str(s) => s.into_pyobject(py)?.into_any(),
        Value::Vec(x) if as_bytes => values_bytes(py, &x, f64::to_ne_bytes)?.into_any(),
        Value::IntVec(x) if as_bytes => values_bytes(py, &x, i64::to_ne_bytes)?.into_any(),
        Value::Vec(x) => x.into_pyobject(py)?.into_any(),
        Value::IntVec(x) => x.into_pyobject(py)?.into_any(),
        Value::Index(x) => x.into_pyobject(py)?.into_any(),
        Value::StrVec(x) => x.into_pyobject(py)?.into_any(),
        Value::BoolVec(x) => x.into_pyobject(py)?.into_any(),
        Value::Factor { .. } => unreachable!("glmm-bind returns no factor"),
        Value::Mat { nrow, data, .. } => {
            let rows: Vec<Vec<f64>> = (0..nrow)
                .map(|i| data.iter().skip(i).step_by(nrow).copied().collect())
                .collect();
            rows.into_pyobject(py)?.into_any()
        }
        Value::Record(r) => {
            let d = PyDict::new(py);
            for (k, v) in r.into_fields() {
                let inner = match k.as_str() {
                    "fit0" | "fit1" => FIT_BYTES,
                    "robust" => ROBUST_BYTES,
                    _ => &[],
                };
                let as_bytes = bytes.contains(&k.as_str());
                d.set_item(k, to_py(py, v, inner, as_bytes)?)?;
            }
            d.into_any()
        }
        Value::List(items) => {
            let l = PyList::empty(py);
            for v in items {
                l.append(to_py(py, v, &[], false)?)?;
            }
            l.into_any()
        }
    })
}

/// A refusal of `glmm.anova(m0, m1)` in Python's words: the crate's text says
/// what is wrong and what to do with no call syntax, and the refusals that
/// `glmm.compare` avoids gain the pointer to it (and to `reml=False`).
fn refusal(kind: &str, message: String) -> String {
    const COMPARE: &str = "glmm.compare(data, f0, f1)";
    let pointer = match kind {
        "different_rows" => format!("Or use {COMPARE}, which drops those rows once for both fits."),
        "different_y" => format!("Or use {COMPARE}, which fits both to the same data."),
        "different_weights" => format!("Or use {COMPARE}, which takes the weights once."),
        "different_offset" => format!("Or use {COMPARE}, which takes the offset once."),
        "different_family" => format!("Or use {COMPARE}, which takes the family and link once."),
        "mixed_estimators" => format!("Or use {COMPARE}, which picks the estimator itself."),
        "reml_different_fixed" => {
            format!("Pass reml=False to both fits, or use {COMPARE}, which fits both by ML.")
        }
        "reml_different_fixed_estimated" => format!(
            "Pass reml=False and weight_power=\"estimate\" to both fits, or use {COMPARE}, \
             which fits both by ML."
        ),
        "weight_power_mismatch" => {
            format!("Or use {COMPARE}, which takes weight_power once.")
        }
        _ => return message,
    };
    format!("{message} {pointer}")
}

/// A `glmm-bind` error as a `ValueError`, or a `RuntimeError` for an
/// `internal` one (a caught panic: a crate bug, not a bad argument); its
/// message already names the caller's row.
fn bind_error(e: BindError) -> PyErr {
    if e.code == "internal" {
        return PyRuntimeError::new_err(e.message);
    }
    PyValueError::new_err(match e.kind {
        Some(kind) => refusal(kind, e.message),
        None => e.message,
    })
}

/// `glmm_bind::call(name, inputs)`: `inputs` a dict of the function's
/// arguments, the result as `to_py` gives it.
#[pyfunction]
fn call<'py>(
    py: Python<'py>,
    name: &str,
    inputs: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyAny>> {
    let Value::Record(rec) = to_value(inputs.as_any(), "")? else {
        unreachable!("a dict converts to a record")
    };
    let out = glmm_bind::call(name, rec).map_err(bind_error)?;
    to_py(py, out, bytes_fields(name), name == "simulate")
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(call, m)?)?;
    // The bound `compare_options` puts on `nsim`, for the Python-side message
    // on a value below 1 (which cannot cross as a count).
    m.add("MAX_NSIM", glmm_bind::MAX_NSIM)?;
    Ok(())
}
