//! A port's columns as a table, and a formula's fixed-effects design and terms
//! without a fit.

use std::collections::HashMap;

use crate::formula::{
    design_columns, drop_unused_levels, fixed_design, fixed_part_relation, parse, r_terms, Column,
    FixedPartRelation, ParsedFormula, Table,
};

use super::PortError;

/// `formula` parsed for a fixed-effects design alone ([`model_matrix_columns`],
/// [`run_model_matrix`]), with whether it has no response. Refused: a formula
/// with no `~` (the parser would take its default response name for it), a
/// random-effect term (a fit drops the rows missing a grouping value, which
/// can remove a factor level and change the design), and an `offset()` term
/// (it adds no design column).
///
/// Both checks on `~` read the string, because the parsed formula cannot tell
/// `~ x` from `x`, nor `y = x` from `y ~ x`.
fn design_formula(formula: &str) -> Result<(ParsedFormula, bool), String> {
    if !formula.contains('~') {
        return Err(format!(
            "a design formula needs ~, as in \"~ x + f\" or \"y ~ x + f\"; got {formula:?}"
        ));
    }
    let one_sided = formula.trim_start().starts_with('~');
    let ast = parse(formula).map_err(|e| e.to_string())?;
    if !ast.random_effects.is_empty() {
        return Err(
            "a design formula takes no random-effect terms; give the fixed part only".to_string(),
        );
    }
    if ast.offset.is_some() {
        return Err(
            "a design formula takes no offset() term: an offset adds no design column".to_string(),
        );
    }
    Ok((ast, one_sided))
}

/// The columns of a dataset a formula reads ([`model_matrix_columns`]).
#[derive(Debug, PartialEq, Eq)]
pub struct FormulaColumns {
    /// The columns of the dataset the formula reads.
    pub used: Vec<String>,
    /// The names the formula reads that the dataset lacks.
    pub missing: Vec<String>,
    /// The columns the formula reads as plain numbers, which stay numeric
    /// even when their values look boolean: the response of a two-sided
    /// formula (`crate::formula::numeric_only_columns`).
    pub numeric_only: Vec<String>,
}

/// What [`run_model_matrix`] reads from a dataset whose column names are
/// `available`, as the Python port's `formula_columns` gives them for a fit.
/// A response (`y ~ x`) is read, so it must exist, though it is not part of
/// the design; a one-sided formula (`~ x`) reads none.
///
/// # Errors
///
/// The refusals of a design formula (no `~`, a random-effect term, an
/// `offset()` term) and a formula that does not parse.
pub fn model_matrix_columns(formula: &str, available: &[String]) -> Result<FormulaColumns, String> {
    let (ast, one_sided) = design_formula(formula)?;
    let (used, missing, numeric_only) = design_columns(&ast, available, one_sided);
    Ok(FormulaColumns {
        used,
        missing,
        numeric_only,
    })
}

/// The fixed-effects design of `formula` on pre-marshalled columns, as
/// `(x, n, p, col_names)`: the row-major `n × p` design and the coefficient
/// names [`run_fit`](super::run_fit) gives the same formula, every column kept (an aliased one
/// too), unused factor levels dropped, as `fit` does. `n_rows` is the
/// caller's row count, read only when the formula reads no column (`~ 1`);
/// otherwise the columns give it. The caller drops no row: a missing value is
/// refused before the call.
///
/// # Errors
///
/// The refusals of [`model_matrix_columns`], the column errors of
/// [`run_fit`](super::run_fit) (mismatched lengths, a bad factor code), and the design errors
/// of [`crate::formula::lower`] (an unknown column, a single-level factor, an
/// empty design, a non-finite value).
pub fn run_model_matrix(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    n_rows: usize,
) -> Result<(Vec<f64>, usize, usize, Vec<String>), PortError> {
    let (ast, _) = design_formula(formula)?;
    let mut table = table_of(numeric_columns, factor_columns)?;
    // `table_of` takes `n` from the columns, 0 when there are none.
    if table.columns.is_empty() {
        table.n = n_rows;
    }
    let owned = drop_unused_levels(&table);
    let table = owned.as_ref().unwrap_or(&table);
    let design = fixed_design(&ast, table)?;
    Ok((design.x, table.n, design.p, design.col_names))
}

/// The ports' columns as a [`Table`], every column the same length.
pub(super) fn table_of(
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
) -> Result<Table, String> {
    let mut columns: Vec<(String, Column)> = Vec::new();
    let mut lengths: Vec<(String, usize)> = Vec::new();
    for (name, values) in numeric_columns {
        lengths.push((name.clone(), values.len()));
        columns.push((name, Column::Numeric(values)));
    }
    // Factors arrive pre-coded: the port supplies the level order (e.g. a
    // pandas Categorical's `categories`, or the sorted distinct labels of a
    // plain string column), so the caller's reference level survives to here
    // rather than being re-derived by a sort. An out-of-range code would index
    // past `levels` inside `materialize`, so it is rejected at the boundary.
    for (name, (levels, codes)) in factor_columns {
        if let Some(&bad) = codes.iter().find(|&&c| c as usize >= levels.len()) {
            return Err(format!(
                "factor column {name:?}: code {bad} is out of range for {} levels",
                levels.len()
            ));
        }
        lengths.push((name.clone(), codes.len()));
        columns.push((name, Column::Factor { levels, codes }));
    }
    // Require every column to agree on row count rather than taking the max:
    // a ragged column would otherwise be silently zero-padded by `lower()`'s
    // materialize step, converging to wrong coefficients with no error.
    // `numeric_columns`/`factor_columns` are HashMaps, so their iteration
    // order (and thus which entry lands first in `lengths`) is not stable
    // across calls -- the error message names every column instead of
    // picking out a single "offending" one, so it stays deterministic.
    let n = match lengths.first() {
        Some((_, first_len)) => {
            let n = *first_len;
            if lengths.iter().any(|(_, len)| *len != n) {
                let mut detail: Vec<String> = lengths
                    .iter()
                    .map(|(name, len)| format!("{name:?}: {len}"))
                    .collect();
                detail.sort();
                return Err(format!(
                    "columns have mismatched lengths ({}); all columns must have the same length",
                    detail.join(", ")
                ));
            }
            n
        }
        None => 0,
    };
    Ok(Table { columns, n })
}

/// Whether two formula strings have the same fixed part, for the ports'
/// `AIC` and `BIC` on two fits: `crate::formula::fixed_part_relation`
/// answers `FixedPartRelation::Same` (the same fixed terms, compared as sets
/// of variable names, and the same fixed intercept). The response, the offset
/// and the random part are not read.
///
/// # Errors
///
/// When either formula does not parse; the message quotes it.
pub fn same_fixed_part(f0: &str, f1: &str) -> Result<bool, String> {
    let parsed = |f: &str| parse(f).map_err(|e| format!("formula {f:?} does not parse: {e}"));
    Ok(fixed_part_relation(&parsed(f0)?, &parsed(f1)?) == FixedPartRelation::Same)
}

/// The fixed-term labels of `formula` and whether it has an intercept, read
/// from its parse alone, with no data: R's `attr(terms(f), "term.labels")`.
/// Each term's variables are joined by `:` in the order they first appear in
/// the fixed part, as the lowering labels a term (`FixedTerm::label`), and the
/// terms are sorted by degree, in the written order within a degree.
/// Random-effect terms and `offset()` are not among them.
///
/// # Errors
///
/// A formula that does not parse, with the parser's message.
pub fn term_labels(formula: &str) -> Result<(Vec<String>, bool), String> {
    let ast = parse(formula).map_err(|e| e.to_string())?;
    let labels = r_terms(&ast).iter().map(|vars| vars.join(":")).collect();
    Ok((labels, ast.has_intercept))
}
