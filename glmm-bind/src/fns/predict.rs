use glmm::orchestrate;
use glmm::Scale;

use crate::error::BindError;
use crate::fns::{ModelData, Rows};
use crate::inputs::Inputs;
use crate::notes;
use crate::options;
use crate::port::spell;
use crate::records::read_parts;
use crate::value::{Record, Value};

/// `orchestrate::predict` on the training data, the stored `parts`, and new
/// rows when `new_numeric` or `new_factors` is given (`n_rows` their count).
/// `fit`, `se`, `lower`, `upper`, `future_lower`, `future_upper` (a part not
/// asked for empty) and `notes`, each `kind` and the `message` the port
/// raises, its rows counted from `row_base`.
pub(crate) fn predict(inp: &mut Inputs) -> Result<Value, BindError> {
    let port = inp.port();
    let scale = options::choice(
        inp,
        "scale",
        "type",
        &[("link", Scale::Link), ("response", Scale::Response)],
    )?;
    let population = options::population(inp)?;
    let allow_new_levels = options::flag(inp, "allow_new_levels", spell(port, "allow_new_levels"))?;
    let se = options::flag(inp, "se", spell(port, "se_fit"))?;
    let level = inp.f64("level")?;
    if !(level > 0.0 && level < 1.0) {
        return Err(BindError::invalid_argument(format!(
            "level must be in (0, 1), got {level}"
        )));
    }
    let future = options::flag(inp, "future", "future")?;
    let d = ModelData::read(inp)?;
    let parts = read_parts(inp.record("parts")?)?;
    let new = if inp.has("new_numeric") || inp.has("new_factors") {
        Some((
            inp.numeric("new_numeric")?,
            inp.factors("new_factors")?,
            inp.usize("n_rows")?,
        ))
    } else {
        None
    };
    let rows = Rows::read(inp)?;
    let family = d.family().map_err(|e| rows.error(e.into()))?;
    let r = orchestrate::predict(
        &d.formula,
        d.numeric,
        d.factors,
        family,
        d.weights,
        d.offset,
        &parts,
        new,
        scale,
        population,
        allow_new_levels,
        spell(port, "allow_new_levels_true"),
        se,
        level,
        future,
    )
    .map_err(|e| rows.error(e))?;
    let notes = r
        .notes
        .into_iter()
        .map(|n| {
            Value::Record(
                Record::new()
                    .with("kind", Value::Str(n.kind.to_owned()))
                    .with(
                        "message",
                        notes::predict_message(&n, rows.base()).map_or(Value::Null, Value::Str),
                    ),
            )
        })
        .collect();
    Ok(Value::Record(
        Record::new()
            .with("fit", Value::Vec(r.fit))
            .with("se", Value::Vec(r.se))
            .with("lower", Value::Vec(r.lower))
            .with("upper", Value::Vec(r.upper))
            .with("future_lower", Value::Vec(r.future_lower))
            .with("future_upper", Value::Vec(r.future_upper))
            .with("notes", Value::List(notes)),
    ))
}

/// `orchestrate::simulate`: the draws, draw after draw.
pub(crate) fn simulate(inp: &mut Inputs) -> Result<Value, BindError> {
    let nsim = options::nsim(inp)?;
    let seed = options::seed(inp)?;
    let d = ModelData::read(inp)?;
    let parts = read_parts(inp.record("parts")?)?;
    let rows = Rows::read(inp)?;
    let family = d.family().map_err(|e| rows.error(e.into()))?;
    let draws = orchestrate::simulate(
        &d.formula, d.numeric, d.factors, family, d.weights, d.offset, &parts, nsim, seed,
    )
    .map_err(|e| rows.error(e))?;
    Ok(Value::Vec(draws))
}

/// `orchestrate::predict_columns`: the names of `available` the formula
/// reads on new rows, the response left out.
pub(crate) fn predict_columns(inp: &mut Inputs) -> Result<Value, BindError> {
    let names = orchestrate::predict_columns(&inp.string("formula")?, &inp.strs("available")?)?;
    Ok(Value::StrVec(names))
}
