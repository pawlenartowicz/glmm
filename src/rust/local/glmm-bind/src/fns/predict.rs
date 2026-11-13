use glmm::orchestrate;

use crate::error::BindError;
use crate::fns::{ModelData, Rows};
use crate::inputs::Inputs;
use crate::options;
use crate::port::spell;
use crate::records::read_parts;
use crate::value::{Record, Value};

/// A note's `detail` with its rows counted from `base`: the whole `detail`
/// of `rows_not_estimable`, the rows after `": "` of every other kind
/// (`new_grouping_levels` gives `"<grouping index>: <rows>"`, the grouping
/// index stays 0-based).
fn detail_rows_from(kind: &str, detail: String, base: usize) -> String {
    let shift = |rows: &str| -> String {
        rows.split(", ")
            .map(|i| {
                i.parse::<usize>()
                    .map_or(i.to_string(), |i| (i + base).to_string())
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    if base == 0 {
        return detail;
    }
    if kind == "rows_not_estimable" {
        return shift(&detail);
    }
    match detail.split_once(": ") {
        Some((g, rows)) => format!("{g}: {}", shift(rows)),
        None => detail,
    }
}

/// `orchestrate::predict` on the training data, the stored `parts`, and new
/// rows when `new_numeric` or `new_factors` is given (`n_rows` their count).
/// `fit`, `se`, `lower`, `upper`, `future_lower`, `future_upper` (a part not
/// asked for empty) and `notes`, each `kind` and `detail`, the rows in
/// `detail` counted from `row_base`.
pub(crate) fn predict(inp: &mut Inputs) -> Result<Value, BindError> {
    let port = inp.port();
    let scale = options::choice(inp, "scale", "type", &["link", "response"])?;
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
        &scale,
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
                        "detail",
                        Value::Str(detail_rows_from(n.kind, n.detail, rows.base())),
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
    let draws = orchestrate::simulate(
        &d.formula, d.numeric, d.factors, &d.family, &d.link, d.weights, d.offset, &parts, nsim,
        seed,
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
        let new_levels = "new_grouping_levels";
        assert_eq!(detail_rows_from(new_levels, "1: 0, 4".into(), 1), "1: 1, 5");
        assert_eq!(detail_rows_from(new_levels, "1: 0, 4".into(), 0), "1: 0, 4");
        assert_eq!(detail_rows_from(new_levels, "no rows".into(), 1), "no rows");
        assert_eq!(
            detail_rows_from("rows_not_estimable", "0, 4".into(), 1),
            "1, 5"
        );
    }
}
