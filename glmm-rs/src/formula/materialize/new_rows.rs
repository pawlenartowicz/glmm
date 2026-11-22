use std::collections::HashMap;

use super::random::{grouping_labels, layout_node, re_group_name, re_only_columns};
use super::{
    check_table, design_vars, fixed_design, kept_nodes, offset_column, Column, FixedDesign,
    Lowered, Model, Table,
};
use crate::formula::error::Error;
use crate::formula::parse::{parse, ParsedFormula};
use crate::{Design, ErrorMessage, FitError, GroupIds, PredictOptions, Prediction};

/// A new table lowered against a fit's lowering ([`lower_new`]): the fit's
/// design columns, random-effect-only columns, ids and offset on the new
/// rows. No response and no weights.
pub(crate) struct NewRows {
    /// Row-major `n × p`, in [`Lowered::col_names`]' order and coding.
    pub x: Vec<f64>,
    /// Row count.
    pub n: usize,
    /// Design width.
    pub p: usize,
    /// Row-major `n × re_p`, as [`Lowered::re_columns`]. All zero when `ids`
    /// is `None`: population mode reads no random effect, and the block only
    /// keeps the design the shape the model's slope indices need.
    pub re_columns: Vec<f64>,
    /// Width of `re_columns`.
    pub re_p: usize,
    /// The rows' slot per grouping, or `None` when the table has none of the
    /// model's grouping columns. A level the fit did not see gets the slot
    /// count of its grouping, past every slot.
    pub ids: Option<GroupIds>,
    /// The `offset()` term on the new rows, `None` when the formula has none.
    pub offset: Option<Vec<f64>>,
}

impl NewRows {
    /// The rows as a [`Design`] for [`crate::predict`].
    pub(crate) fn design(&self) -> Design<'_> {
        let mut design = Design::new(&self.x, self.n, self.p);
        if self.re_p > 0 {
            design = design.re_columns(&self.re_columns, self.re_p);
        }
        if let Some(ids) = &self.ids {
            design = design.ids(ids);
        }
        if let Some(o) = &self.offset {
            design = design.offset(o);
        }
        design
    }
}

/// Lowers the rows of `table` as `lowered`, the lowering of `formula` that a
/// model was fitted on, lowered its own rows: each factor coded with the
/// fit's levels by label, the fixed design ([`fixed_design`]) and the
/// random-effect-only columns ([`re_only_columns`]) then in the fit's order
/// by construction, the offset evaluated on the new rows, and each grouping
/// label matched to its slot in [`ReGroupInfo::slot_labels`](super::ReGroupInfo::slot_labels). The response
/// is not read. With `population` (a population prediction) no grouping
/// column and no random-effect-only column is read: `ids` is `None` and
/// `re_columns` all zero, as for a table with none of the grouping columns.
///
/// # Errors
/// [`Error::Fit`] with [`FitError::Shape`] for a level of a fixed factor or
/// a factor slope that the fit did not see, naming column, level and row (lme4
/// refuses it too); [`Error::WrongColumnKind`] for a variable whose kind is
/// not the one it had in the fit; the errors of [`fixed_design`] and the
/// offset as [`materialize`](super::materialize) returns them. In conditional mode only (not
/// `population`): the errors of [`re_only_columns`], a grouping label that
/// names two of the fit's levels, and [`Error::UnknownColumn`] for a grouping
/// column missing while another is present.
pub(crate) fn lower_new(
    formula: &str,
    lowered: &Lowered,
    table: &Table,
    population: bool,
) -> Result<NewRows, Error> {
    let ast = parse(formula)?;
    check_table(table)?;
    let groupings = &lowered.re_groups;
    let (
        table,
        FixedDesign {
            x,
            p,
            mut numeric_main_col,
            mut factor_main_cols,
            ..
        },
    ) = new_fixed_design(&ast, lowered, table)?;
    let n = table.n;
    let present = !population
        && groupings
            .iter()
            .any(|g| g.name.split(':').any(|c| table.get(c).is_some()));
    let (re_columns, re_p, ids) = if present {
        let (re_columns, re_p) = re_only_columns(
            kept_nodes(&ast, groupings.iter().map(|g| &g.name)),
            &table,
            p,
            &mut numeric_main_col,
            &mut factor_main_cols,
        )?;
        if re_p != lowered.re_p {
            return Err(not_the_lowering());
        }
        let mut ids = groupings.iter().map(|g| {
            let nodes = ast
                .random_effects
                .iter()
                .filter(|re| re_group_name(re) == g.name);
            let node = layout_node(nodes).ok_or_else(not_the_lowering)?;
            // `None` marks a label two slots share: the fit keeps the pairs
            // ("a:b", "c") and ("a", "b:c") apart, but both join to "a:b:c".
            let mut slot: HashMap<&str, Option<u32>> = HashMap::new();
            for (k, l) in g.slot_labels.iter().enumerate() {
                if let Some(l) = l.as_deref() {
                    slot.entry(l)
                        .and_modify(|v| *v = None)
                        .or_insert(Some(k as u32));
                }
            }
            let unseen = g.slot_labels.len() as u32;
            grouping_labels(node, &table)?
                .iter()
                .enumerate()
                .map(|(row, l)| match slot.get(l.as_str()) {
                    Some(Some(k)) => Ok(*k),
                    Some(None) => Err(Error::Fit(FitError::Shape(ErrorMessage::at_row(
                        format!("grouping {:?} has the label {l:?} at row ", g.name),
                        row,
                        ", which names two of the fit's levels; relabel its columns without ':'",
                    )))),
                    None => Ok(unseen),
                })
                .collect::<Result<Vec<u32>, Error>>()
        });
        let primary = ids.next().ok_or_else(not_the_lowering)??;
        let extra = ids.collect::<Result<_, _>>()?;
        (re_columns, re_p, Some(GroupIds { primary, extra }))
    } else {
        (vec![0.0; n * lowered.re_p], lowered.re_p, None)
    };
    Ok(NewRows {
        x,
        n,
        p,
        re_columns,
        re_p,
        ids,
        offset: offset_column(&ast, &table)?,
    })
}

/// The rows of `table` coded as the fit's own rows were: each factor with
/// the fit's levels by label, so the fixed design ([`fixed_design`]) has the
/// fit's columns by construction. Returns the recoded table and its fixed
/// design. Grouping columns are copied as they are.
///
/// # Errors
/// The errors of [`lower_new`] on a factor's level, a column's kind and the
/// fixed design.
pub(crate) fn new_fixed_design<'a>(
    ast: &'a ParsedFormula,
    lowered: &Lowered,
    table: &Table,
) -> Result<(Table, FixedDesign<'a>), Error> {
    let vars = design_vars(ast, &lowered.re_groups);
    let fit_levels = |name: &str| {
        lowered
            .factor_levels
            .iter()
            .find(|(f, _)| f == name)
            .map(|(_, l)| l)
    };
    let mut columns = Vec::with_capacity(table.columns.len());
    for (name, col) in &table.columns {
        let col = match (col, fit_levels(name)) {
            (Column::Factor { levels, codes }, Some(fit)) => {
                let index: HashMap<&str, u32> = fit
                    .iter()
                    .enumerate()
                    .map(|(k, l)| (l.as_str(), k as u32))
                    .collect();
                let codes = codes
                    .iter()
                    .enumerate()
                    .map(|(row, &c)| {
                        let label = &levels[c as usize];
                        index.get(label.as_str()).copied().ok_or_else(|| {
                            Error::Fit(FitError::Shape(ErrorMessage::at_row(
                                format!("column {name:?} has the level {label:?} at row "),
                                row,
                                ", which the fit did not see",
                            )))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                Column::Factor {
                    levels: fit.clone(),
                    codes,
                }
            }
            (Column::Numeric(_), Some(_)) => {
                return Err(Error::WrongColumnKind {
                    name: name.clone(),
                    expected: "a factor, as in the fit",
                })
            }
            (Column::Factor { .. }, None) if vars.contains(&name.as_str()) => {
                return Err(Error::WrongColumnKind {
                    name: name.clone(),
                    expected: "numeric",
                })
            }
            (Column::Factor { levels, codes }, None) => Column::Factor {
                levels: levels.clone(),
                codes: codes.clone(),
            },
            (Column::Numeric(v), None) => Column::Numeric(v.clone()),
        };
        columns.push((name.clone(), col));
    }
    let table = Table {
        columns,
        n: table.n,
    };
    let design = fixed_design(ast, &table)?;
    // The fit's levels and column kinds give the fit's columns, unless the
    // formula is not the lowering's.
    if design.col_names != lowered.col_names {
        return Err(not_the_lowering());
    }
    Ok((table, design))
}

/// The refusal of a formula whose new rows do not lower to its lowering's
/// design.
fn not_the_lowering() -> Error {
    Error::Fit(FitError::Shape(
        "the formula does not lower to the fit's design; predict with the formula it was \
         fitted with"
            .into(),
    ))
}

/// Predicts `model` on the rows of `table` ([`crate::predict`]): the table
/// is lowered as the fit's own rows were, with the fit's factor levels,
/// column order, random-effect-only columns, transforms and offset, and its
/// grouping labels matched to the fit's levels. A grouping label the fit did
/// not see is an unseen level ([`PredictOptions::allow_new_levels`]). A table
/// with none of the model's grouping columns gives no ids, which population
/// mode does not need. The response and any weights are not read, so a
/// binomial future-observation bound is for one trial.
///
/// # Errors
/// [`Error::Fit`] with [`FitError::Shape`] for a level of a fixed factor that
/// the fit did not see, naming column, level and row; the lowering errors of
/// [`lower`](super::lower) that apply to new rows ([`Error::UnknownColumn`],
/// [`Error::WrongColumnKind`], [`Error::TransformNotFinite`]); and
/// [`Error::Fit`] with the errors of [`crate::predict`].
pub fn predict(model: &Model, table: &Table, opts: &PredictOptions) -> Result<Prediction, Error> {
    let rows = lower_new(&model.formula, &model.lowered, table, opts.population)?;
    crate::predict(&model.fit, model.lowered.spec(), &rows.design(), opts).map_err(Error::Fit)
}
