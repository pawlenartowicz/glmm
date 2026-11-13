use std::collections::HashMap;

use glmm::formula;
use glmm::orchestrate;

use crate::coerce;
use crate::error::BindError;
use crate::fns::Rows;
use crate::inputs::Inputs;
use crate::records::int;
use crate::value::{Record, Value};

/// `orchestrate::term_labels`: the fixed-term `labels` and `has_intercept`.
pub(crate) fn term_labels(inp: &mut Inputs) -> Result<Value, BindError> {
    let (labels, has_intercept) = orchestrate::term_labels(&inp.string("formula")?)?;
    Ok(Value::Record(
        Record::new()
            .with("labels", Value::StrVec(labels))
            .with("has_intercept", Value::Bool(has_intercept)),
    ))
}

/// `orchestrate::same_fixed_part`.
pub(crate) fn same_fixed_part(inp: &mut Inputs) -> Result<Value, BindError> {
    let same = orchestrate::same_fixed_part(&inp.string("f0")?, &inp.string("f1")?)?;
    Ok(Value::Bool(same))
}

fn columns_record((used, missing, numeric_only): (Vec<String>, Vec<String>, Vec<String>)) -> Value {
    Value::Record(
        Record::new()
            .with("used", Value::StrVec(used))
            .with("missing", Value::StrVec(missing))
            .with("numeric_only", Value::StrVec(numeric_only)),
    )
}

/// What `formula` reads from a dataset whose column names are `available`:
/// `used` (the data columns it reads), `missing` (names it references that
/// resolve against no column nor a transform of one) and `numeric_only`
/// (the response and the `offset()` column, which never cross as factors).
pub(crate) fn formula_columns(inp: &mut Inputs) -> Result<Value, BindError> {
    let ast = formula::parse(&inp.string("formula")?)
        .map_err(|e| BindError::invalid_argument(e.to_string()))?;
    let available = inp.strs("available")?;
    Ok(columns_record((
        formula::referenced_columns(&ast, &available),
        formula::unresolved_columns(&ast, &available),
        formula::numeric_only_columns(&ast, &available),
    )))
}

/// `formula_columns` for a design formula (`orchestrate::model_matrix_columns`).
pub(crate) fn model_matrix_columns(inp: &mut Inputs) -> Result<Value, BindError> {
    let cols = orchestrate::model_matrix_columns(&inp.string("formula")?, &inp.strs("available")?)?;
    Ok(columns_record(cols))
}

/// `orchestrate::run_model_matrix`: `x` the row-major `n x p` design, `n`,
/// `p` and the coefficient `names`.
pub(crate) fn model_matrix(inp: &mut Inputs) -> Result<Value, BindError> {
    let formula = inp.string("formula")?;
    let numeric = inp.numeric("numeric")?;
    let factors = inp.factors("factors")?;
    let rows = Rows::read(inp)?;
    let (x, n, p, names) =
        orchestrate::run_model_matrix(&formula, numeric, factors, inp.usize("n_rows")?)
            .map_err(|e| rows.error(e))?;
    Ok(Value::Record(
        Record::new()
            .with("x", Value::Vec(x))
            .with("n", int(n))
            .with("p", int(p))
            .with("names", Value::StrVec(names)),
    ))
}

/// A plain column's elements read in one pass:
///
/// - `{kind: "num", values}` when every element is a number, a bool or
///   `Null` (missing, NaN);
/// - `{kind: "str", codes, keys}` when every element is a string or missing
///   (`Null`, or a NaN number): `keys` the distinct strings in first-seen
///   order, then one `Null` key when a row is missing, which every missing
///   row's code names;
/// - `Null` for anything else, so the port runs its own per-element code.
///
/// A port decides which of its element types reach here as numbers and
/// strings; any other value (a record, say) makes the answer `Null`.
pub(crate) fn plain_column(inp: &mut Inputs) -> Result<Value, BindError> {
    let elements: Vec<Value> = match inp.value("elements") {
        Value::List(x) => x,
        Value::Vec(x) => x.into_iter().map(Value::F64).collect(),
        Value::IntVec(x) => x.into_iter().map(Value::I64).collect(),
        Value::BoolVec(x) => x.into_iter().map(Value::Bool).collect(),
        Value::StrVec(x) => x.into_iter().map(Value::Str).collect(),
        other => {
            return Err(BindError::invalid_argument(format!(
                "elements must be a list, got {}",
                coerce::describe(&other)
            )))
        }
    };
    if elements.is_empty() {
        return Ok(Value::Null);
    }
    let number = |v: &Value| match v {
        Value::Null => Some(f64::NAN),
        Value::Bool(b) => Some(f64::from(u8::from(*b))),
        Value::I64(_) | Value::U64(_) | Value::F64(_) => coerce::to_f64(v).ok(),
        _ => None,
    };
    if let Some(values) = elements.iter().map(number).collect::<Option<Vec<f64>>>() {
        return Ok(Value::Record(
            Record::new()
                .with("kind", Value::Str("num".into()))
                .with("values", Value::Vec(values)),
        ));
    }
    let mut index: HashMap<&str, i64> = HashMap::new();
    let mut keys = Vec::new();
    let mut codes = Vec::with_capacity(elements.len());
    let mut any_missing = false;
    for v in &elements {
        match v {
            Value::Str(s) => {
                let next = index.len() as i64;
                let code = *index.entry(s.as_str()).or_insert_with(|| {
                    keys.push(Value::Str(s.clone()));
                    next
                });
                codes.push(code);
            }
            Value::Null => {
                any_missing = true;
                codes.push(-1);
            }
            Value::F64(x) if x.is_nan() => {
                any_missing = true;
                codes.push(-1);
            }
            _ => return Ok(Value::Null),
        }
    }
    if any_missing {
        let code = keys.len() as i64;
        codes.iter_mut().filter(|c| **c < 0).for_each(|c| *c = code);
        keys.push(Value::Null);
    }
    Ok(Value::Record(
        Record::new()
            .with("kind", Value::Str("str".into()))
            .with("codes", Value::IntVec(codes))
            .with("keys", Value::List(keys)),
    ))
}
