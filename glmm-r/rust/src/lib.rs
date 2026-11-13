//! extendr binding: the one native entry point of the `fastglmm` R package.
//! `glmm_call` converts a named list of arguments into a `glmm_bind::Record`,
//! runs the registered function, and converts the result back; argument
//! checks, results and error codes live in `glmm-bind`. Re-exported into the
//! R package's staticlib by `glmm-r/src/rust`
//! (`extendr_module! { mod fastglmm; use glmm_r; }`).

use extendr_api::prelude::*;
use glmm_bind::{BindError, Port, Record, Value};

/// Fields holding positions, counted from 1 in R and from 0 in `glmm-bind`.
/// Only these are read -1; an `Index` result is written +1 whatever
/// its name.
const INDEX_FIELDS: &[&str] = &["retained", "re_term_index", "null_cluster"];

/// Arguments whose list maps a column name to its data: their entries are
/// named `column "x"`, so a column's name is never read as a field's.
const COLUMN_MAPS: &[&str] = &["numeric", "factors", "new_numeric", "new_factors"];

/// Counts a fit object stores as integers; every other count leaves as a
/// double, as the fit objects saved by earlier versions hold them.
const INT_FIELDS: &[&str] = &["objective_nodes", "evals"];

fn bad(what: &str, why: &str) -> BindError {
    BindError::invalid_argument(format!("{what} {why}"))
}

/// An R value as a [`Value`]; `what` is its argument or field name. A named
/// list is a record, any other list a list; a factor keeps its levels in R's
/// order (codes from 0, `NA` as -1); a lone logical, integer or string `NA`
/// is [`Value::Na`]; every other vector stays a vector, as `glmm-bind` reads
/// a one-element vector as a scalar.
fn to_value(x: &Robj, what: &str) -> std::result::Result<Value, BindError> {
    if x.is_null() {
        return Ok(Value::Null);
    }
    if x.inherits("factor") {
        let levels = x
            .get_attrib("levels")
            .and_then(|l| l.as_string_vector())
            .ok_or_else(|| bad(what, "is a factor without levels"))?;
        let codes = x.as_integer_slice().unwrap_or_default();
        return Ok(Value::Factor {
            levels,
            codes: codes
                .iter()
                .map(|&c| if c == i32::MIN { -1 } else { c - 1 })
                .collect(),
        });
    }
    match x.rtype() {
        Rtype::List => {
            let list = List::try_from(x).map_err(|e| bad(what, &e.to_string()))?;
            match x.names() {
                Some(names) => {
                    let columns = COLUMN_MAPS.contains(&what);
                    let mut rec = Record::new();
                    for (key, item) in names.zip(list.values()) {
                        if key.is_empty() {
                            return Err(bad(what, "must name every element of the list"));
                        }
                        let inner = if columns {
                            format!("column {key:?}")
                        } else {
                            key.to_owned()
                        };
                        rec.push(key, to_value(&item, &inner)?);
                    }
                    Ok(Value::Record(rec))
                }
                None => list
                    .values()
                    .map(|item| to_value(&item, what))
                    .collect::<std::result::Result<_, _>>()
                    .map(Value::List),
            }
        }
        Rtype::Doubles => {
            let v = x.as_real_slice().unwrap_or_default();
            if INDEX_FIELDS.contains(&what) {
                return index(v.iter().copied(), what);
            }
            Ok(Value::Vec(v.to_vec()))
        }
        Rtype::Integers => {
            let v = x.as_integer_slice().unwrap_or_default();
            if v == [i32::MIN] {
                return Ok(Value::Na);
            }
            if v.contains(&i32::MIN) {
                return Err(bad(what, "has a missing value"));
            }
            if INDEX_FIELDS.contains(&what) {
                return index(v.iter().map(|&k| f64::from(k)), what);
            }
            Ok(Value::IntVec(v.iter().map(|&k| i64::from(k)).collect()))
        }
        Rtype::Logicals => {
            let v = x.as_logical_slice().unwrap_or_default();
            if let [b] = v {
                if b.is_na() {
                    // `NA` given for an option: its reader names it in its refusal.
                    return Ok(Value::Na);
                }
            }
            if v.iter().any(|b| b.is_na()) {
                return Err(bad(what, "has a missing value"));
            }
            Ok(Value::BoolVec(v.iter().map(|b| b.is_true()).collect()))
        }
        Rtype::Strings => {
            let v: Vec<&str> = x.as_str_iter().map(Iterator::collect).unwrap_or_default();
            if let [s] = v[..] {
                if s.is_na() {
                    return Ok(Value::Na);
                }
            }
            if v.iter().any(|s| s.is_na()) {
                return Err(bad(what, "has a missing value"));
            }
            Ok(Value::StrVec(v.into_iter().map(str::to_owned).collect()))
        }
        other => Err(bad(
            what,
            &format!("cannot pass an R value of type {other:?} to glmm's native code"),
        )),
    }
}

/// 1-based positions as a 0-based [`Value::Index`].
fn index(v: impl Iterator<Item = f64>, what: &str) -> std::result::Result<Value, BindError> {
    v.map(|k| {
        if k >= 1.0 && k.fract() == 0.0 && k <= f64::from(i32::MAX) {
            Ok(k as usize - 1)
        } else {
            Err(bad(what, &format!("must be positions from 1, got {k}")))
        }
    })
    .collect::<std::result::Result<_, _>>()
    .map(Value::Index)
}

/// A [`Value`] as an R value; `what` is its field name. Counts (`I64`,
/// `U64`, `IntVec`) become doubles except the [`INT_FIELDS`]; an `Index`
/// becomes 1-based integers.
fn to_r(v: Value, what: &str) -> Robj {
    match v {
        Value::Null => r!(NULL),
        Value::Bool(b) => r!(b),
        Value::I64(n) if INT_FIELDS.contains(&what) => {
            i32::try_from(n).map_or_else(|_| r!(n as f64), |k| r!(k))
        }
        Value::I64(n) => r!(n as f64),
        Value::U64(n) => r!(n as f64),
        Value::F64(x) => r!(x),
        Value::Str(s) => r!(s),
        Value::Vec(x) => r!(x),
        Value::IntVec(x) => r!(x.into_iter().map(|n| n as f64).collect::<Vec<f64>>()),
        Value::Index(x) => r!(x.into_iter().map(|k| k as i32 + 1).collect::<Vec<i32>>()),
        Value::StrVec(x) => r!(x),
        Value::BoolVec(x) => r!(x),
        Value::Factor { .. } => unreachable!("glmm-bind returns no factor"),
        Value::Na => unreachable!("glmm-bind returns no NA"),
        Value::Mat { nrow, ncol, data } => {
            let mut out = r!(data);
            let _ = out.set_attrib("dim", [nrow as i32, ncol as i32]);
            out
        }
        Value::Record(rec) => {
            let (names, values): (Vec<String>, Vec<Robj>) = rec
                .into_fields()
                .into_iter()
                .map(|(k, v)| {
                    let r = to_r(v, &k);
                    (k, r)
                })
                .unzip();
            List::from_names_and_values(names, values)
                .expect("as many names as values")
                .into()
        }
        Value::List(items) => List::from_values(items.into_iter().map(|v| to_r(v, what))).into(),
    }
}

/// Run `glmm-bind`'s function `name` on the named list `inputs`. Returns
/// `list(value)`, or `list(error, code)` for the R helper
/// `.glmm_call()` to raise, so no R error unwinds through Rust frames.
///
/// @usage NULL
/// @keywords internal
#[extendr]
fn glmm_call(name: &str, inputs: Robj) -> Robj {
    let out = to_value(&inputs, "inputs").and_then(|v| match v {
        Value::Record(rec) => glmm_bind::call(Port::R, name, rec),
        Value::List(items) if items.is_empty() => glmm_bind::call(Port::R, name, Record::new()),
        _ => Err(bad("inputs", "must be a named list")),
    });
    match out {
        Ok(v) => list!(value = to_r(v, "")).into(),
        Err(e) => list!(error = e.message, code = e.code).into(),
    }
}

extendr_module! {
    mod glmm_r;
    fn glmm_call;
}
