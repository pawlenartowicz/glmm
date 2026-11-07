use glmm::orchestrate;

use crate::error::BindError;
use crate::fns::{ModelData, Rows};
use crate::inputs::Inputs;
use crate::records::read_parts;
use crate::value::{Record, Value};

/// A note's `detail` with the rows after `": "` counted from `base`
/// (`new_grouping_levels` gives `"<grouping index>: <rows>"`, the grouping
/// index stays 0-based).
fn detail_rows_from(detail: String, base: usize) -> String {
    match detail.split_once(": ") {
        Some((g, rows)) if base != 0 => {
            let rows: Vec<String> = rows
                .split(", ")
                .map(|i| {
                    i.parse::<usize>()
                        .map_or(i.to_string(), |i| (i + base).to_string())
                })
                .collect();
            format!("{g}: {}", rows.join(", "))
        }
        _ => detail,
    }
}

/// `orchestrate::predict` on the training data, the stored `parts`, and new
/// rows when `new_numeric` or `new_factors` is given (`n_rows` their count).
/// `fit`, `se`, `lower`, `upper`, `future_lower`, `future_upper` (a part not
/// asked for empty) and `notes`, each `kind` and `detail`, the rows in
/// `detail` counted from `row_base`.
pub(crate) fn predict(inp: &mut Inputs) -> Result<Value, BindError> {
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
    let r = orchestrate::predict(
        &d.formula,
        d.numeric,
        d.factors,
        &d.family,
        &d.link,
        d.weights,
        d.offset,
        &parts,
        new,
        &inp.string("scale")?,
        inp.bool("population")?,
        inp.bool("allow_new_levels")?,
        inp.bool("se")?,
        inp.f64("level")?,
        inp.bool("future")?,
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
                        "detail",
                        Value::Str(detail_rows_from(n.detail, rows.base())),
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
    let d = ModelData::read(inp)?;
    let parts = read_parts(inp.record("parts")?)?;
    let rows = Rows::read(inp)?;
    let draws = orchestrate::simulate(
        &d.formula,
        d.numeric,
        d.factors,
        &d.family,
        &d.link,
        d.weights,
        d.offset,
        &parts,
        inp.usize("nsim")?,
        inp.seed("seed")?,
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

#[cfg(test)]
mod tests {
    use super::detail_rows_from;

    #[test]
    fn note_rows_count_from_the_base_and_the_grouping_index_does_not() {
        assert_eq!(detail_rows_from("1: 0, 4".into(), 1), "1: 1, 5");
        assert_eq!(detail_rows_from("1: 0, 4".into(), 0), "1: 0, 4");
        assert_eq!(detail_rows_from("no rows".into(), 1), "no rows");
    }
}
