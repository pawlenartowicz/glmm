use std::borrow::Cow;
use std::collections::BTreeSet;

use super::fixed::sorted_levels_and_codes;
use crate::formula::error::Error;
use crate::FitError;

/// A single data column. A `Factor` carries its level order from the caller —
/// see [`Column::factor_from_labels`] for the no-declared-order default.
pub enum Column {
    /// A numeric column, used verbatim.
    Numeric(Vec<f64>),
    /// A categorical column, as levels + per-row codes.
    Factor {
        /// Distinct category levels **in the caller's intended order**; a
        /// repeated label is refused. Not
        /// required to be sorted, and not required to be exhausted by `codes`:
        /// the lowering drops the levels no row has, for every factor the
        /// formula reads (fixed, random slope and grouping), as R's model frame
        /// does for `lm`, `glm` and lme4 (`drop.unused.levels = TRUE`). The
        /// first level that has rows is the treatment-contrast base, so this
        /// order picks the reference level.
        levels: Vec<String>,
        /// Per-row index into `levels` (length `n`).
        codes: Vec<u32>,
    },
}

impl Column {
    /// A factor from per-row labels, levels ordered **lexicographically** (by
    /// code point) — R's `factor()` default for a character vector in the C
    /// locale (R in another locale can order mixed-case labels differently),
    /// and the right default for a
    /// caller that has no declared order to state (a plain string column).
    ///
    /// A caller that *does* have one — a pandas `Categorical`, an R
    /// `factor(levels = …)` — must build [`Column::Factor`] directly, so that
    /// order survives into the treatment-contrast base. Going through this
    /// constructor would discard it.
    pub fn factor_from_labels(labels: &[String]) -> Column {
        let (levels, codes) = sorted_levels_and_codes(labels);
        Column::Factor { levels, codes }
    }
}

/// The crate's minimal columnar input — no Arrow, no pandas. Consumers
/// (SDOC/Arrow, the Python port/pandas) convert into this at their boundary, once
/// per model.
///
/// # Errors
/// [`lower`](super::lower) refuses a table whose column names repeat, whose columns are not
/// all `n` long, or whose factor has a code past its levels or a level label
/// given more than once.
pub struct Table {
    /// Name → column, each of length `n`, in column order; each name once.
    pub columns: Vec<(String, Column)>,
    /// Row count.
    pub n: usize,
}

impl Table {
    pub(super) fn get(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }
}

/// `data`'s columns checked against it: each name once, each `n` long, each
/// factor code a level and each level label once. A longer column would index
/// past the design, a shorter one would be fitted as zeros, a bad code would
/// index past the levels, and a repeated name or label would be read as its
/// first copy.
pub(super) fn check_table(data: &Table) -> Result<(), Error> {
    let mut names = BTreeSet::new();
    for (name, col) in &data.columns {
        if !names.insert(name.as_str()) {
            return Err(Error::Fit(FitError::Shape(
                format!("duplicate column name {name:?}").into(),
            )));
        }
        let len = match col {
            Column::Numeric(v) => v.len(),
            Column::Factor { codes, .. } => codes.len(),
        };
        if len != data.n {
            return Err(Error::Fit(FitError::Shape(
                format!("column {name:?} has {len} rows, the table {}", data.n).into(),
            )));
        }
        if let Column::Factor { levels, codes } = col {
            if let Some(&bad) = codes.iter().find(|&&c| c as usize >= levels.len()) {
                return Err(Error::Fit(FitError::Shape(
                    format!(
                        "factor column {name:?}: code {bad} is out of range for {} levels",
                        levels.len()
                    )
                    .into(),
                )));
            }
            let mut seen = BTreeSet::new();
            if let Some(label) = levels.iter().find(|l| !seen.insert(l.as_str())) {
                return Err(Error::Fit(FitError::Shape(
                    format!("factor column {name:?} has the level {label:?} more than once").into(),
                )));
            }
        }
    }
    Ok(())
}

/// A grouping column's levels and per-row codes: borrowed from a factor column,
/// built for a whole-number one (see [`grouping_factor`](super::random::grouping_factor)).
pub(super) type LevelsAndCodes<'a> = (Cow<'a, [String]>, Cow<'a, [u32]>);

/// `levels` without the ones no code uses (R's `droplevels`), in their order,
/// and `codes` renumbered to match; both borrowed when every level is used.
pub(super) fn used_levels<'a>(levels: &'a [String], codes: &'a [u32]) -> LevelsAndCodes<'a> {
    let mut used = vec![false; levels.len()];
    for &c in codes {
        used[c as usize] = true;
    }
    if used.iter().all(|&u| u) {
        return (Cow::Borrowed(levels), Cow::Borrowed(codes));
    }
    let mut remap = vec![0u32; levels.len()];
    let mut kept = Vec::new();
    for (l, _) in used.iter().enumerate().filter(|&(_, &u)| u) {
        remap[l] = kept.len() as u32;
        kept.push(levels[l].clone());
    }
    let codes = codes.iter().map(|&c| remap[c as usize]).collect();
    (Cow::Owned(kept), Cow::Owned(codes))
}

/// `data` with each factor column's unused levels dropped by [`used_levels`],
/// as R's model frame does (`drop.unused.levels = TRUE`) before `lm`, `glm`
/// and lme4 code a factor; `None` when every factor uses all its levels. The
/// codes must be in range ([`check_table`]). A new table for `predict` is not
/// passed here: it is coded by the fit's levels.
pub(crate) fn drop_unused_levels(data: &Table) -> Option<Table> {
    let compact: Vec<Option<(Vec<String>, Vec<u32>)>> = data
        .columns
        .iter()
        .map(|(_, col)| match col {
            Column::Factor { levels, codes } => match used_levels(levels, codes) {
                (Cow::Owned(levels), Cow::Owned(codes)) => Some((levels, codes)),
                _ => None,
            },
            Column::Numeric(_) => None,
        })
        .collect();
    if compact.iter().all(Option::is_none) {
        return None;
    }
    // shortcut: copies the whole table when a factor has an unused level;
    // drop per column on read if that copy shows in a profile.
    let columns = data
        .columns
        .iter()
        .zip(compact)
        .map(|((name, col), compact)| {
            let col = match (col, compact) {
                (_, Some((levels, codes))) => Column::Factor { levels, codes },
                (Column::Factor { levels, codes }, None) => Column::Factor {
                    levels: levels.clone(),
                    codes: codes.clone(),
                },
                (Column::Numeric(v), None) => Column::Numeric(v.clone()),
            };
            (name.clone(), col)
        })
        .collect();
    Some(Table { columns, n: data.n })
}
