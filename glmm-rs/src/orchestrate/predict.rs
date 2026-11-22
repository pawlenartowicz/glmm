//! `predict` and `simulate` on what a port stored of one fit, on its training
//! rows or on new data.

use std::collections::HashMap;

use crate::fit::{detect_aliased, null_basis, CondvarCross};
use crate::formula::{design_columns, lower_new, parse, Lowered};
use crate::predict::predict_parts;
use crate::{
    ErrorMessage, Family, FitError, FitOptions, PredictOptions, Prediction, Scale, Simulator,
};

use super::design::table_of;
use super::fit::{lower_with, note_infos, NoteInfo, MAX_NSIM};
use super::stored::StoredParts;
use super::PortError;

/// `formula` lowered on a port's stored training columns as [`run_fit`]
/// lowered them: [`table_of`], then [`lower_with`] with the `weights=` and
/// `offset=` arguments the fit was given. The lowering is deterministic, so
/// this is the fit's own, with its rows, trial counts, factor levels and
/// grouping labels.
pub(super) fn relower(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
) -> Result<Lowered, String> {
    let table = table_of(numeric_columns, factor_columns)?;
    let prepared = lower_with(
        formula,
        &table,
        family,
        FitOptions::default(),
        weights,
        offset,
    )?;
    Ok(prepared.lowered)
}

/// What [`predict`] reads from a new table whose column names are
/// `available`: the names the formula reads there, as
/// [`model_matrix_columns`](super::model_matrix_columns)' `used` gives them, without the response. A
/// grouping column is among them when the table has it.
///
/// # Errors
///
/// A formula that does not parse.
pub fn predict_columns(formula: &str, available: &[String]) -> Result<Vec<String>, String> {
    let ast = parse(formula).map_err(|e| e.to_string())?;
    Ok(design_columns(&ast, available, true).0)
}

/// What [`predict`] returns: the parts of [`crate::Prediction`], with its
/// notes flattened as a fit's are. A part that was not asked for is empty.
#[derive(Debug)]
pub struct PredictResult {
    /// The prediction on the requested scale.
    pub fit: Vec<f64>,
    /// Its standard error, when asked for.
    pub se: Vec<f64>,
    /// The lower end of the interval, when the standard error is asked for.
    pub lower: Vec<f64>,
    /// The upper end of the interval.
    pub upper: Vec<f64>,
    /// The lower bound of a future observation, when asked for.
    pub future_lower: Vec<f64>,
    /// The upper bound of a future observation.
    pub future_upper: Vec<f64>,
    /// One `"new_grouping_levels"` note per grouping with levels the fit did
    /// not see, then one `"rows_not_estimable"` note when a row is outside the
    /// row space of a rank-deficient fit.
    pub notes: Vec<NoteInfo>,
}

/// The ports' `predict`: [`crate::predict`] of a stored fit. The training
/// columns, `family`, `weights` and `offset` are what the fit was made
/// from, as [`run_fit`](super::run_fit) took them, and are lowered again (`relower`);
/// `parts` are the numbers the port stored. `new` is the new table's columns
/// and its row count (read only when it has no column), or `None` to predict
/// the training rows. `scale`, `population`, `allow_new_levels`, `se`,
/// `level` and `future` are those of [`crate::PredictOptions`].
/// `new_levels_arg` is the caller's spelling of turning `allow_new_levels`
/// on, which the unseen-level refusal names.
///
/// A new table is lowered with the fit's factor levels, column order and
/// grouping labels; a table with none of the model's grouping columns
/// predicts in population mode only.
///
/// # Errors
///
/// A `level` outside `(0, 1)`; `future` on a model that is
/// not a non-Gaussian mixed model; new rows for a fit made with an `offset=`
/// argument (its values for the new rows are unknown); conditional mode on a
/// new table without the grouping columns; a new row with a grouping level
/// the fit did not see while `allow_new_levels` is off ([`FitError::Ids`],
/// naming the grouping and the row); the lowering errors of the new table;
/// and every error of [`crate::predict`].
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn predict(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    parts: &StoredParts,
    new: Option<(
        HashMap<String, Vec<f64>>,
        HashMap<String, (Vec<String>, Vec<u32>)>,
        usize,
    )>,
    scale: Scale,
    population: bool,
    allow_new_levels: bool,
    new_levels_arg: &str,
    se: bool,
    level: f64,
    future: bool,
) -> Result<PredictResult, PortError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(format!("level must be in (0, 1), got {level}").into());
    }
    let offset_argument = offset.is_some();
    let lowered = relower(
        formula,
        numeric_columns,
        factor_columns,
        family,
        weights,
        offset,
    )?;
    let mixed = lowered.model.re.is_some();
    if future && !(mixed && lowered.model.family != Family::Gaussian) {
        return Err("future applies only to a non-Gaussian mixed model"
            .to_string()
            .into());
    }
    let opts = PredictOptions {
        population,
        scale,
        allow_new_levels,
        se,
        level,
        future,
    };
    let cross = parts.condvar_cross.as_ref().map(CondvarCross::from);
    let mut fp = parts.fit_parts(lowered.model.family, cross.as_ref());
    // The fit's null space, from the training design it was fitted on, which
    // the port does not store.
    let aliased = detect_aliased(&lowered.x, lowered.n, lowered.p);
    if aliased.iter().any(|&a| a) {
        fp.null_basis =
            std::borrow::Cow::Owned(null_basis(&lowered.x, lowered.n, lowered.p, &aliased));
    }
    let pred = (|| -> Result<Prediction, PortError> {
        let fitted = |design: &crate::Design<'_>| {
            predict_parts(&fp, lowered.spec(), design, &opts).map_err(PortError::from)
        };
        let Some((numeric, factor, n_rows)) = new else {
            return fitted(&lowered.design());
        };
        if offset_argument {
            return Err(
                "this model was fitted with the offset= argument, whose values for new rows \
                 are unknown; to predict new rows, fit with an offset() term in the formula"
                    .to_string()
                    .into(),
            );
        }
        let mut table = table_of(numeric, factor)?;
        if table.columns.is_empty() {
            table.n = n_rows;
        }
        let rows = lower_new(formula, &lowered, &table, population)?;
        if mixed && !population && rows.ids.is_none() {
            let names: Vec<&str> = lowered.re_groups.iter().map(|g| g.name.as_str()).collect();
            return Err(format!(
                "the new data has none of the grouping columns ({}), so it can be predicted \
                 from the fixed effects only",
                names.join(", ")
            )
            .into());
        }
        if let (false, Some(ids)) = (allow_new_levels, &rows.ids) {
            // The kernel refuses the same rows in the same order, but names
            // the grouping by its index. Mirrors the check in
            // `crate::predict`; change together.
            let columns: Vec<&[u32]> = std::iter::once(&ids.primary)
                .chain(&ids.extra)
                .map(Vec::as_slice)
                .collect();
            for i in 0..rows.n {
                for (g, col) in lowered.re_groups.iter().zip(&columns) {
                    if col[i] as usize >= g.slot_labels.len() {
                        return Err(FitError::Ids(ErrorMessage::at_row(
                            format!(
                                "grouping {:?} has a level the fit did not see at row ",
                                g.name
                            ),
                            i,
                            &format!(
                                "; set {new_levels_arg} to predict it with a zero random effect"
                            ),
                        ))
                        .into());
                    }
                }
            }
        }
        fitted(&rows.design())
    })()?;
    Ok(PredictResult {
        fit: pred.fit,
        se: pred.se,
        lower: pred.lower,
        upper: pred.upper,
        future_lower: pred.future_lower,
        future_upper: pred.future_upper,
        notes: note_infos(pred.notes, &lowered.re_groups),
    })
}

/// The ports' `simulate`: `nsim` draws of the response of a stored fit on its
/// training rows ([`crate::Simulator`]), draw `d` the Philox stream `d` of
/// `seed`. The training columns, `family`, `weights` and `offset` are
/// read as [`predict`] reads them, so a binomial draw uses the fit's trial
/// counts. Returned draw after draw: draw `d` is
/// `out[d·nobs..(d + 1)·nobs]`.
///
/// # Errors
///
/// An `nsim` of 0 or above [`MAX_NSIM`], or draws too many to allocate; and
/// the errors of [`crate::Simulator::new`] (a fit that did not converge has no
/// dispersion to draw with).
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    parts: &StoredParts,
    nsim: usize,
    seed: u64,
) -> Result<Vec<f64>, PortError> {
    if !(1..=MAX_NSIM).contains(&nsim) {
        return Err(format!("nsim must be a whole number from 1 to {MAX_NSIM}, got {nsim}").into());
    }
    let lowered = relower(
        formula,
        numeric_columns,
        factor_columns,
        family,
        weights,
        offset,
    )?;
    let cross = parts.condvar_cross.as_ref().map(CondvarCross::from);
    let fp = parts.fit_parts(lowered.model.family, cross.as_ref());
    let design = lowered.design();
    let sim = Simulator::from_parts(&fp, lowered.spec(), &design)?;
    let n = lowered.n;
    // An allocation failure aborts the process instead of panicking, so a
    // draw count too large for memory is refused here.
    let mut out = Vec::new();
    out.try_reserve_exact(n * nsim)
        .map_err(|_| format!("nsim = {nsim} draws of {n} rows do not fit in memory"))?;
    out.resize(n * nsim, 0.0);
    if n > 0 {
        for (d, y) in out.chunks_mut(n).enumerate() {
            sim.simulate_into(seed, d as u64, y)?;
        }
    }
    Ok(out)
}
