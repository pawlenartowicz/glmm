//! Coercion of one value to the type a parameter or field needs. Each
//! function returns the reason as a `String`; [`crate::inputs::Inputs`]
//! prefixes the argument's name.
//!
//! An integer takes `I64`, `U64` or a whole `F64` (R sends counts as
//! doubles); a one-element vector reads as its element and a scalar widens
//! to a one-element vector, because R has no scalar / length-1 distinction.
//! An empty list is an empty vector of any type (Python's `[]` has none).

use glmm::{SeType, WeightPower};

use crate::value::Value;

/// 2^53: every integer up to here is exactly representable as an `f64`.
const F64_EXACT_MAX: f64 = 9_007_199_254_740_992.0;

/// A short description of a value for error messages; a one-element vector
/// is described by its element.
pub(crate) fn describe(v: &Value) -> String {
    match &scalar(v) {
        Value::Null => "null".to_owned(),
        Value::Bool(b) => b.to_string(),
        Value::I64(n) => n.to_string(),
        Value::U64(n) => n.to_string(),
        Value::F64(x) => x.to_string(),
        Value::Str(s) => format!("the string {s:?}"),
        Value::Na => "NA".to_owned(),
        other => {
            let kind = other.kind_name();
            let article = if kind.starts_with(['a', 'e', 'i', 'o', 'u']) {
                "an"
            } else {
                "a"
            };
            format!("{article} {kind}")
        }
    }
}

/// The element of a one-element vector, else the value itself.
fn scalar(v: &Value) -> Value {
    match v {
        Value::Vec(x) if x.len() == 1 => Value::F64(x[0]),
        Value::IntVec(x) if x.len() == 1 => Value::I64(x[0]),
        Value::BoolVec(x) if x.len() == 1 => Value::Bool(x[0]),
        Value::StrVec(x) if x.len() == 1 => Value::Str(x[0].clone()),
        other => other.clone(),
    }
}

fn whole_f64(x: f64) -> bool {
    x.is_finite() && x.fract() == 0.0 && (0.0..=F64_EXACT_MAX).contains(&x)
}

/// A count: non-negative and whole.
pub(crate) fn to_usize(v: &Value) -> Result<usize, String> {
    let bad = || format!("must be a non-negative whole number, got {}", describe(v));
    match scalar(v) {
        Value::I64(n) => usize::try_from(n).map_err(|_| bad()),
        Value::U64(n) => usize::try_from(n).map_err(|_| bad()),
        // Through `u64`, exact up to 2^53.
        Value::F64(x) if whole_f64(x) => usize::try_from(x as u64).map_err(|_| bad()),
        _ => Err(bad()),
    }
}

/// A seed: any `u64`. A float must be whole and at most 2^53.
pub(crate) fn to_seed(v: &Value) -> Result<u64, String> {
    let bad = || format!("must be a whole number in [0, 2^64), got {}", describe(v));
    match scalar(v) {
        Value::U64(n) => Ok(n),
        Value::I64(n) => u64::try_from(n).map_err(|_| bad()),
        Value::F64(x) if whole_f64(x) => Ok(x as u64),
        _ => Err(bad()),
    }
}

/// A float; integers convert.
pub(crate) fn to_f64(v: &Value) -> Result<f64, String> {
    match scalar(v) {
        Value::F64(x) => Ok(x),
        Value::I64(n) => Ok(n as f64),
        Value::U64(n) => Ok(n as f64),
        _ => Err(format!("must be a number, got {}", describe(v))),
    }
}

/// A bool; nothing converts.
pub(crate) fn to_bool(v: &Value) -> Result<bool, String> {
    match scalar(v) {
        Value::Bool(b) => Ok(b),
        _ => Err(format!("must be TRUE/FALSE (a bool), got {}", describe(v))),
    }
}

/// A string.
pub(crate) fn to_string(v: Value) -> Result<String, String> {
    match scalar(&v) {
        Value::Str(s) => Ok(s),
        _ => Err(format!("must be a string, got {}", describe(&v))),
    }
}

/// A weight power: a finite number, or the string `"estimate"`.
pub(crate) fn to_weight_power(v: Value) -> Result<WeightPower, String> {
    match scalar(&v) {
        Value::Str(s) if s == "estimate" => Ok(WeightPower::Estimate),
        Value::F64(x) if x.is_finite() => Ok(WeightPower::Fixed(x)),
        Value::I64(n) => Ok(WeightPower::Fixed(n as f64)),
        Value::U64(n) => Ok(WeightPower::Fixed(n as f64)),
        _ => Err(format!(
            r#"must be a finite number or "estimate", got {}"#,
            describe(&v)
        )),
    }
}

/// A standard-error type: `"model"` or `"robust"`.
pub(crate) fn to_se_type(v: Value) -> Result<SeType, String> {
    match scalar(&v) {
        Value::Str(s) if s == "model" => Ok(SeType::Model),
        Value::Str(s) if s == "robust" => Ok(SeType::Robust),
        _ => Err(format!(
            r#"must be "model" or "robust", got {}"#,
            describe(&v)
        )),
    }
}

/// A float vector. Scalars widen; integer and logical vectors convert
/// (`true` is 1, as numpy casts a bool array).
pub(crate) fn to_vec(v: Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Vec(x) => Ok(x),
        Value::F64(x) => Ok(vec![x]),
        Value::I64(n) => Ok(vec![n as f64]),
        Value::U64(n) => Ok(vec![n as f64]),
        Value::IntVec(xs) => Ok(xs.into_iter().map(|n| n as f64).collect()),
        Value::BoolVec(bs) => Ok(bs.into_iter().map(|b| f64::from(u8::from(b))).collect()),
        Value::List(x) if x.is_empty() => Ok(Vec::new()),
        other => Err(format!(
            "must be a numeric vector, got {}",
            describe(&other)
        )),
    }
}

/// Whole numbers, each read as [`to_i64s`] reads it.
fn whole_ints(v: Value) -> Result<Vec<i64>, String> {
    let what = describe(&v);
    let bad = || format!("must be whole numbers, got {what}");
    match v {
        Value::IntVec(x) => Ok(x),
        Value::List(x) if x.is_empty() => Ok(Vec::new()),
        Value::I64(n) => Ok(vec![n]),
        Value::U64(n) => i64::try_from(n).map(|n| vec![n]).map_err(|_| bad()),
        Value::Index(x) => x
            .into_iter()
            .map(|n| i64::try_from(n).map_err(|_| bad()))
            .collect(),
        Value::F64(_) | Value::Vec(_) => {
            let xs = to_vec(v)?;
            if xs
                .iter()
                .all(|x| x.is_finite() && x.fract() == 0.0 && x.abs() <= F64_EXACT_MAX)
            {
                Ok(xs.into_iter().map(|x| x as i64).collect())
            } else {
                Err(bad())
            }
        }
        _ => Err(bad()),
    }
}

/// Integers taken as they are (the caller's row numbers).
pub(crate) fn to_i64s(v: Value) -> Result<Vec<i64>, String> {
    whole_ints(v)
}

/// Non-negative whole numbers: an `Index` (0-based), or counts.
pub(crate) fn to_usizes(v: Value) -> Result<Vec<usize>, String> {
    if let Value::Index(x) = v {
        return Ok(x);
    }
    whole_ints(v)?
        .into_iter()
        .map(|n| usize::try_from(n).map_err(|_| format!("must be non-negative, got {n}")))
        .collect()
}

/// A string vector; a string widens.
pub(crate) fn to_strs(v: Value) -> Result<Vec<String>, String> {
    match v {
        Value::StrVec(x) => Ok(x),
        Value::Str(s) => Ok(vec![s]),
        Value::List(items) => items.into_iter().map(to_string).collect(),
        other => Err(format!("must be a string vector, got {}", describe(&other))),
    }
}

/// A logical vector; a bool widens.
pub(crate) fn to_bools(v: Value) -> Result<Vec<bool>, String> {
    match v {
        Value::BoolVec(x) => Ok(x),
        Value::Bool(b) => Ok(vec![b]),
        Value::List(x) if x.is_empty() => Ok(Vec::new()),
        other => Err(format!(
            "must be a logical vector, got {}",
            describe(&other)
        )),
    }
}

/// A list of float vectors (`varcorr`, `jac_theta`).
pub(crate) fn to_vecs(v: Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::List(items) => items.into_iter().map(to_vec).collect(),
        other => Err(format!(
            "must be a list of numeric vectors, got {}",
            describe(&other)
        )),
    }
}

/// A `width`-column matrix as its rows: a list of rows (Python) or a flat
/// row-major vector (R).
pub(crate) fn to_rows(v: Value, width: usize) -> Result<Vec<Vec<f64>>, String> {
    let rows = match v {
        Value::List(_) => to_vecs(v)?,
        flat => {
            let flat = to_vec(flat)?;
            if flat.len() % width.max(1) != 0 {
                return Err(format!(
                    "has {} entries, not a whole number of rows of {width}",
                    flat.len()
                ));
            }
            flat.chunks(width.max(1)).map(<[f64]>::to_vec).collect()
        }
    };
    match rows.iter().find(|r| r.len() != width) {
        Some(r) => Err(format!(
            "has a row of {} entries, expected {width}",
            r.len()
        )),
        None => Ok(rows),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_take_whole_doubles_and_refuse_the_rest() {
        assert_eq!(to_usize(&Value::F64(3.0)), Ok(3));
        assert_eq!(to_usize(&Value::Vec(vec![3.0])), Ok(3));
        for bad in [
            Value::F64(3.5),
            Value::I64(-1),
            Value::F64(f64::NAN),
            Value::Str("3".into()),
        ] {
            assert!(to_usize(&bad).is_err(), "{bad:?} accepted");
        }
    }

    #[test]
    fn a_matrix_reads_as_rows_in_each_form() {
        let want = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let flat = Value::Vec(vec![1.0, 2.0, 3.0, 4.0]);
        let list = Value::List(vec![Value::Vec(vec![1.0, 2.0]), Value::Vec(vec![3.0, 4.0])]);
        for v in [flat, list] {
            assert_eq!(to_rows(v, 2).unwrap(), want);
        }
        assert!(to_rows(Value::Vec(vec![1.0, 2.0, 3.0]), 2).is_err());
    }

    #[test]
    fn a_weight_power_is_a_finite_number_or_estimate_and_one_text_refuses_the_rest() {
        assert_eq!(
            to_weight_power(Value::F64(0.5)),
            Ok(WeightPower::Fixed(0.5))
        );
        assert_eq!(to_weight_power(Value::I64(2)), Ok(WeightPower::Fixed(2.0)));
        assert_eq!(
            to_weight_power(Value::Vec(vec![0.0])),
            Ok(WeightPower::Fixed(0.0))
        );
        assert_eq!(
            to_weight_power(Value::Str("estimate".into())),
            Ok(WeightPower::Estimate)
        );
        for bad in [
            Value::F64(f64::NAN),
            Value::F64(f64::INFINITY),
            Value::Bool(true),
            Value::Str("Estimate".into()),
            Value::Vec(vec![1.0, 2.0]),
        ] {
            let e = to_weight_power(bad.clone()).unwrap_err();
            assert!(
                e.starts_with(r#"must be a finite number or "estimate", got"#),
                "{bad:?}: {e}"
            );
        }
        assert_eq!(to_se_type(Value::Str("robust".into())), Ok(SeType::Robust));
        assert_eq!(to_se_type(Value::Str("model".into())), Ok(SeType::Model));
        let e = to_se_type(Value::Str("CR2".into())).unwrap_err();
        assert_eq!(e, r#"must be "model" or "robust", got the string "CR2""#);
    }

    #[test]
    fn an_empty_list_is_an_empty_vector_of_any_type() {
        assert_eq!(to_vec(Value::List(vec![])), Ok(vec![]));
        assert_eq!(to_usizes(Value::List(vec![])), Ok(vec![]));
        assert_eq!(to_bools(Value::List(vec![])), Ok(vec![]));
        assert_eq!(to_strs(Value::List(vec![])), Ok(Vec::<String>::new()));
    }
}
