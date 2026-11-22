//! Data-dependent lowering — the only stage that touches data. Turns a
//! [`ParsedFormula`] + a [`Table`] into the numeric inputs [`crate::fit`]
//! consumes: a row-major design matrix, a structure-only [`ModelSpec`], per-row
//! [`GroupIds`], and the weights and offset the formula itself carries.
//!
//! Conventions (validated against R `model.matrix` / `lme4`):
//! - **Treatment contrasts**, base = first level that has rows — the caller's
//!   [`Column::Factor`] level order picks the reference level, and levels with
//!   no rows are dropped first, as R's model frame does.
//!   [`Column::factor_from_labels`] supplies the lexicographic order R's
//!   `factor()` defaults to when the caller has none. Dummy names follow `paste0(var, level)`, e.g. `period2`. A factor keeps its
//!   base level (full indicator set) where R's `model.matrix` keeps it — see
//!   [`full_indicator_flags`](fixed::full_indicator_flags).
//! - **Column order** is R's (`terms.formula`, then `model.matrix`): the
//!   intercept, then the terms sorted by degree (main effects, then two-way
//!   interactions, and so on), keeping the written order within a degree.
//!   [`Lowered::col_names`] and the columns of [`Lowered::x`] follow it.
//! - **Interaction columns** are the elementwise product of their components'
//!   expanded columns. The components are taken in the order their variables
//!   first appear in the fixed part of the formula, not as the term spells them
//!   (`y ~ x + f:x` gives `x:fb`); names are those components joined by `:`,
//!   and the first component's contrasts vary fastest (R's `model.matrix`).
//! - **Random effects**: the first grouping in the AST is primary (its width
//!   lives in `ReStructure`); the rest are `extra_groupings`. The terms that
//!   name one grouping are one block, the intercept term first, with
//!   [`crate::ReTerm`]s when there is more than one term or none holds the
//!   intercept. The spec carries no level counts — the kernel derives them
//!   from `GroupIds`.

use std::sync::OnceLock;

use super::error::{Error, ParseError};
use super::parse::{parse, parse_transform, ParsedFormula, RandomEffect, DEFAULT_RESPONSE};
use crate::{Data, Design, Family, Fit, FitError, FitOptions, GroupIds, ModelSpec};

#[cfg(feature = "orchestrate")]
mod columns;
mod fixed;
mod new_rows;
mod random;
mod ranef;
mod table;

#[cfg(feature = "orchestrate")]
pub(crate) use columns::design_columns;
#[cfg(feature = "orchestrate")]
pub use columns::{numeric_only_columns, referenced_columns, unresolved_columns};
pub(super) use fixed::margin_present;
pub(crate) use fixed::{fixed_design, r_terms};
use fixed::{numeric_column, rdx_cross, FixedDesign};
#[cfg(any(test, feature = "orchestrate"))]
pub(crate) use new_rows::lower_new;
pub(crate) use new_rows::new_fixed_design;
pub use new_rows::predict;
use random::{lower_random_effects, re_group_name, LoweredRandom};
pub use ranef::{label_ranef, RanefBlock};
use table::check_table;
pub(crate) use table::drop_unused_levels;
pub use table::{Column, Table};

/// One grouping factor's random-effect column names, in the order of the
/// block of `Fit::varcorr(g)` (`"(Intercept)"` first, then any slopes), plus
/// the term each column belongs to and the label of every slot of its RE
/// block.
pub struct ReGroupInfo {
    /// Grouping factor name (`"Subject"`, or `"A:B"` for a nested inner factor).
    pub name: String,
    /// RE column names for this grouping, e.g. `["(Intercept)", "Days"]`. An
    /// intercept-free block (`(0 + x | g)`) has no `"(Intercept)"`.
    pub terms: Vec<String>,
    /// Per entry of `terms`, the index of its term among the grouping's terms,
    /// which come in block order, the intercept's term first:
    /// `(x || g)` gives `[0, 1]`, `(1 + x | g)` gives `[0, 0]`. Columns of two
    /// terms are uncorrelated by the model.
    pub term_index: Vec<usize>,
    /// Level label per slot of this grouping's RE block, in kernel block order,
    /// `None` for a padded nested slot that belongs to no level. Emitted by the
    /// same layer that builds the layout (`grouping_ids`), because the padding
    /// of a nested grouping depends on how many children each parent has in
    /// the data. The labeller behind [`Model::ranef`] is the only reader.
    pub slot_labels: Vec<Option<String>>,
}

/// The lowered fit inputs — a one-shot front door to [`crate::fit`]:
/// [`Lowered::data`] and [`Lowered::spec`] are its first two arguments.
///
/// `Lowered` is `#[non_exhaustive]`: outside this crate it cannot be built by
/// struct literal, not even by struct update from a lowered value. Build one
/// with [`lower`]. A caller may assign [`Lowered::y`], [`Lowered::weights`]
/// and [`Lowered::offset`] before [`fit_lowered`]; the other public fields
/// record the lowering and are read, not assigned. [`fit_lowered`] refuses a
/// lowering of another formula. It also carries private
/// fields (the data cross-products the hypothesis matrices need), which on
/// their own forbid construction outside the crate, so the example below fails
/// whatever the attribute says; the attribute keeps a new public field an
/// additive change.
///
/// ```compile_fail
/// use glmm::formula::{lower, Column, Lowered, Table};
/// use glmm::Family;
///
/// let table = Table {
///     columns: vec![("y".into(), Column::Numeric(vec![0.5, 1.5]))],
///     n: 2,
/// };
/// let lo = lower("y ~ 1", &table, Family::Gaussian).unwrap();
/// let _copy = Lowered { ..lo };
/// ```
// Non-exhaustive so a new field is an additive change, not a break.
#[non_exhaustive]
pub struct Lowered {
    /// Row-major n×p design (intercept column first when the formula has one).
    pub x: Vec<f64>,
    /// The response column, or `s/(s+f)` for a `cbind` response.
    pub y: Vec<f64>,
    /// Row count.
    pub n: usize,
    /// Design width (emitted column count).
    pub p: usize,
    /// Coefficient name per design column, in column order (`"(Intercept)"`
    /// first when the formula has one). The R-matching handle the
    /// end-to-end/contrast oracles assert against.
    pub col_names: Vec<String>,
    /// Row-major `n × re_p` random-effect-only columns
    /// ([`crate::Data::re_columns`]): the columns of the random-slope
    /// variables that are in no fixed main term, one set per variable (a
    /// numeric column, or a factor's treatment dummies) in order of first
    /// appearance among the random-effect terms. A slope reads column `j` as
    /// slope index `p + j`. Empty when every slope variable is a fixed term.
    pub re_columns: Vec<f64>,
    /// Width of [`Lowered::re_columns`].
    pub re_p: usize,
    /// Structure-only model spec (the kernel derives the level counts from
    /// `ids`).
    pub model: ModelSpec,
    /// Per-row level ids for every grouping.
    pub ids: GroupIds,
    /// Per-grouping name + RE term names + slot labels, in the block order of
    /// `Fit::varcorr(g)` (primary first, then extras in declaration order).
    /// Empty when the formula declares no random effects.
    pub re_groups: Vec<ReGroupInfo>,
    /// Observations the LOWERING made about the data, which no solver can make
    /// because no solver sees the labels —
    /// [`crate::Note::SingleLevelGroupingDropped`] and
    /// [`crate::Note::FewRowsForRandomEffects`]. Empty on a clean
    /// lowering. Carried here rather than on `Fit` because it is decided
    /// before any fit runs; the ports fold it into the same warning channel as
    /// `Fit`'s own notes.
    pub notes: Vec<crate::Note>,
    /// Per-row prior weights: the trial counts of a `cbind()` response, else
    /// `None`. Caller may set them ([`crate::Data::weights`]).
    pub weights: Option<Vec<f64>>,
    /// Per-row offset from an `offset()` formula term, else `None`. Caller may
    /// set it ([`crate::Data::offset`]).
    pub offset: Option<Vec<f64>>,
    /// The fixed terms in R's order (`attr(terms, "term.labels")`), the
    /// intercept excluded. Their `columns` partition the design columns after
    /// the intercept: R's `assign`, as ranges.
    pub terms: Vec<FixedTerm>,
    /// The formula keeps the intercept (column 0 of [`Lowered::x`]).
    pub has_intercept: bool,
    /// `Xᵀ·rdX` for the Type III hypotheses (`formula::hypotheses::type3`).
    pub(crate) rdx: RdxCross,
    /// `XᵀX` over all `p` columns, filled by the first hypothesis builder
    /// that reads it (`formula::hypotheses`), so the Type II and Type III
    /// tables of one fit, and every fit of the design in a loop, share one.
    pub(crate) xtx: OnceLock<faer::Mat<f64>>,
    /// Each factor the design reads, a fixed term's or a random slope's, with
    /// its used levels (those with rows) in the fit's order. [`new_rows::lower_new`] codes a new table's labels
    /// by them; a grouping factor is matched by its slot labels instead.
    pub(crate) factor_levels: Vec<(String, Vec<String>)>,
    /// Each numeric variable the fixed part reads, the `offset()` term's
    /// included, with its unweighted mean over the rows, in order of first
    /// appearance. A transform's variable is listed raw: `log(x)` gives the
    /// entry `x`, so a row at the mean reads `log(x̄)`, as emmeans' reference
    /// grid does.
    pub numeric_means: Vec<(String, f64)>,
    /// The parsed formula it was lowered from, which [`fit_lowered`] compares
    /// with the formula it is given: [`super::anova::nesting`] and
    /// [`predict`] read the formula string, the fit reads this lowering.
    pub(crate) ast: ParsedFormula,
}

impl Lowered {
    /// The data of the fit: [`Lowered::x`], [`Lowered::y`], `n` and `p`, the
    /// random-effect-only columns when there are any, the ids when the model
    /// has random effects, and the weights and offset when set.
    pub fn data(&self) -> Data<'_> {
        let mut data = Data::new(&self.x, &self.y, self.n, self.p);
        if self.re_p > 0 {
            data = data.re_columns(&self.re_columns, self.re_p);
        }
        if self.model.re.is_some() {
            data = data.ids(&self.ids);
        }
        if let Some(w) = &self.weights {
            data = data.weights(w);
        }
        if let Some(o) = &self.offset {
            data = data.offset(o);
        }
        data
    }

    /// The rows of the fit without the response: [`Lowered::data`]'s design,
    /// with the same random-effect-only columns, ids, weights and offset. A
    /// `cbind` response's trial counts are its weights, so a
    /// [`crate::Simulator`] built on it draws from the trials the fit saw.
    pub fn design(&self) -> Design<'_> {
        self.data().design
    }

    /// The model of the fit, [`Lowered::model`].
    pub fn spec(&self) -> &ModelSpec {
        &self.model
    }
}

/// One fixed term of the formula, as R's `terms.formula` lists it.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct FixedTerm {
    /// R's term label: the variables joined by `:` (`"g:h"`, `"x:f"`).
    pub label: String,
    /// The term's variables, in the order of their first appearance in the
    /// fixed part of the formula (R's order inside a term label).
    pub vars: Vec<String>,
    /// Per variable of `vars`: `true` for a numeric column (or a transform of
    /// one), `false` for a factor.
    pub numeric: Vec<bool>,
    /// The term's design columns in [`Lowered::x`] and [`Lowered::col_names`].
    pub columns: std::ops::Range<usize>,
}

/// lmerTest's `rdX` (`get_rdX`: the intercept, then per term the full
/// indicator coding `model.matrix(~ 0 + term)`), kept as its cross-product
/// with the design, since the data are gone after lowering.
pub(crate) struct RdxCross {
    /// The `rdX` column names (`"(Intercept)"` first when the formula has one).
    pub(crate) names: Vec<String>,
    /// `Xᵀ·rdX`, `p × names.len()`.
    pub(crate) xt_rdx: faer::Mat<f64>,
}

/// Parses `formula` and lowers it against `data` in one call, producing the
/// design matrix, structure-only [`ModelSpec`], per-row [`GroupIds`], and the
/// formula's own weights and offset that [`crate::fit`] consumes.
///
/// `formula` is an R-style formula string (e.g. `"y ~ x1 * x2 + (1 + x1 | g)"`).
/// `data` supplies one column per name referenced in `formula`; `family`
/// selects the response distribution recorded in the returned `ModelSpec`.
///
/// # Errors
/// Returns [`Error::Parse`] for a malformed formula (see [`super::ParseError`]), or
/// one of the data-dependent variants ([`Error::UnknownColumn`],
/// [`Error::ResponseNotNumeric`], [`Error::WrongColumnKind`],
/// [`Error::RandomColumnRepeated`],
/// [`Error::SingleLevelFactor`], [`Error::EmptyDesign`],
/// [`Error::TransformNotFinite`], [`Error::CbindNeedsBinomial`],
/// [`Error::ZeroTrials`], [`Error::NegativeCount`]) when `data` doesn't match
/// what the formula requires. [`Error::Fit`] with [`FitError::Shape`] when
/// `data` has no rows or is not a valid [`Table`]: a repeated column name, a column that is
/// not `n` long, a factor code past its levels or a factor level label given
/// more than once.
///
/// # Examples
/// ```
/// use glmm::formula::{lower, Column, Table};
/// use glmm::Family;
///
/// let data = Table {
///     columns: vec![
///         ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0])),
///         ("x".into(), Column::Numeric(vec![0.5, 1.0, 1.5])),
///     ],
///     n: 3,
/// };
/// let result = lower("y ~ x", &data, Family::Gaussian);
/// assert!(result.is_ok());
/// let lo = result.unwrap();
/// assert_eq!(lo.n, 3);
/// assert_eq!(lo.p, 2); // intercept + x
/// ```
pub fn lower(formula: &str, data: &Table, family: Family) -> Result<Lowered, Error> {
    let ast = parse(formula)?;
    materialize(&ast, data, family)
}

/// A model fitted through the formula door: the formula, its lowering, the
/// options it was fitted with and the fit. Built by [`fit`] or
/// [`fit_lowered`]; [`super::anova::compare`] and
/// [`super::anova::term_tests`] take it.
pub struct Model {
    formula: String,
    lowered: Lowered,
    /// Read by the comparison and the per-term tests, whose refits take the
    /// fit's own node count and held dispersion.
    pub(super) opts: FitOptions,
    fit: Fit,
}

impl Model {
    /// The formula as given.
    pub fn formula(&self) -> &str {
        &self.formula
    }

    /// The lowering the model was fitted on.
    pub fn lowered(&self) -> &Lowered {
        &self.lowered
    }

    /// The fit.
    pub fn fit(&self) -> &Fit {
        &self.fit
    }

    /// The fit's random-effect conditional modes, labelled: one [`RanefBlock`]
    /// per grouping in declaration order, rows the grouping's levels as the
    /// data spells them, columns its terms, padded nested slots dropped. Empty
    /// when the fit has no conditional modes (no random part, or a fit that did
    /// not converge, whose [`Fit::ranef`] is NaN).
    ///
    /// # Errors
    /// [`Error::RanefShapeMismatch`] when the lowering's `re_groups` do not
    /// describe the fit's stored modes, which only a lowering edited after
    /// [`lower`] can cause.
    pub fn ranef(&self) -> Result<Vec<RanefBlock>, Error> {
        label_ranef(&self.fit, &self.lowered.re_groups)
    }
}

/// Lowers `formula` against `table` ([`lower`]) and fits it with `opts`
/// ([`crate::fit`]).
///
/// # Errors
/// The errors of [`lower`] (a repeated column name or factor level label in
/// `table` among them), and [`Error::Fit`] when the kernel refuses the
/// lowered input.
pub fn fit(
    formula: &str,
    table: &Table,
    family: Family,
    opts: &FitOptions,
) -> Result<Model, Error> {
    fit_lowered(formula, lower(formula, table, family)?, opts)
}

/// Fits a lowering of `formula` with `opts` ([`crate::fit`]): [`fit`] for a
/// caller that sets [`Lowered::weights`] or [`Lowered::offset`] after
/// [`lower`].
///
/// # Errors
/// The parse errors of `formula`; [`Error::Fit`] with [`FitError::Shape`]
/// when `lowered` is not a lowering of `formula` (it was lowered from a
/// formula that parses differently: other fixed terms, intercept, groupings,
/// random terms, response or offset); [`Error::Fit`] when the kernel refuses
/// the lowered input.
pub fn fit_lowered(formula: &str, lowered: Lowered, opts: &FitOptions) -> Result<Model, Error> {
    if parse(formula)? != lowered.ast {
        return Err(Error::Fit(FitError::Shape(
            "the lowering is not the formula's (it was lowered from another formula); lower \
             the formula it is fitted with"
                .into(),
        )));
    }
    let fit = crate::fit(&lowered.data(), lowered.spec(), opts).map_err(Error::Fit)?;
    Ok(Model {
        formula: formula.to_string(),
        lowered,
        opts: opts.clone(),
        fit,
    })
}

/// The data-dependent half alone (caller already holds a [`ParsedFormula`]).
///
/// # Errors
/// Returns the same data-dependent error variants as [`lower`]: [`Error::UnknownColumn`],
/// [`Error::ResponseNotNumeric`], [`Error::WrongColumnKind`],
/// [`Error::RandomColumnRepeated`],
/// [`Error::SingleLevelFactor`], [`Error::EmptyDesign`],
/// [`Error::TransformNotFinite`], [`Error::CbindNeedsBinomial`],
/// [`Error::ZeroTrials`], and [`Error::NegativeCount`]. [`Error::Fit`] with
/// [`FitError::Shape`] when `data` has no rows or is not a valid [`Table`]:
/// a repeated column name, a column that is not `n` long, a factor code past
/// its levels or a factor level label given more than once.
pub fn materialize(ast: &ParsedFormula, data: &Table, family: Family) -> Result<Lowered, Error> {
    check_table(data)?;
    // Not in `check_table`, which `lower_new` shares: a zero-row predict is
    // legal. The text is the kernel's (`fit::common::validate`) - change
    // together.
    if data.n == 0 {
        return Err(Error::Fit(FitError::Shape(
            "Data has no rows (n = 0); a fit needs at least one row".into(),
        )));
    }
    let owned = drop_unused_levels(data);
    let data = owned.as_ref().unwrap_or(data);
    let n = data.n;

    // 1. Response. `cbind(s, f)` lowers to lme4's own objective: proportion
    // as `y`, trial count as prior weights — the path the frozen cbpp golden
    // already validates.
    let mut weights = None;
    let y = match &ast.cbind {
        Some((s, f)) => {
            if !matches!(family, Family::Binomial { .. }) {
                return Err(Error::CbindNeedsBinomial);
            }
            let s = numeric_column(s, data)?;
            let f = numeric_column(f, data)?;
            if let Some(row) = s.iter().zip(&f).position(|(a, b)| *a < 0.0 || *b < 0.0) {
                return Err(Error::NegativeCount { row });
            }
            let trials: Vec<f64> = s.iter().zip(&f).map(|(a, b)| a + b).collect();
            if let Some(row) = trials.iter().position(|t| !(t.is_finite() && *t > 0.0)) {
                return Err(Error::ZeroTrials { row });
            }
            let y = s.iter().zip(&trials).map(|(a, t)| a / t).collect();
            weights = Some(trials);
            y
        }
        None => match data.get(&ast.dependent) {
            Some(Column::Numeric(v)) => v.clone(),
            Some(Column::Factor { .. }) => {
                return Err(Error::ResponseNotNumeric(ast.dependent.clone()))
            }
            // A formula literally naming `explained_variable` on data without
            // that column reads as one with no response.
            None if ast.dependent == DEFAULT_RESPONSE => {
                return Err(ParseError::Syntax {
                    pos: 0,
                    msg: "the formula has no response; write it as y ~ x".into(),
                }
                .into())
            }
            None => return Err(Error::UnknownColumn(ast.dependent.clone())),
        },
    };
    // Refused here, by name, rather than by the fit entry's check, which
    // names `y` and not the formula's response. NaN too: a missing value is
    // the caller's to drop before lowering, as both ports do.
    if let Some(row) = y.iter().position(|v| !v.is_finite()) {
        return Err(Error::TransformNotFinite {
            term: ast.dependent.clone(),
            row,
        });
    }

    // 2. Fixed design.
    let FixedDesign {
        x,
        p,
        col_names,
        terms,
        fixed_terms,
        numeric_main_col,
        factor_main_cols,
    } = fixed_design(ast, data)?;

    let rdx = rdx_cross(&terms, ast.has_intercept, data, &x, n, p)?;

    // 3. Random effects → ReStructure + GroupIds (declaration order; first = primary).
    let LoweredRandom {
        spec: model,
        ids,
        re_groups,
        notes,
        re_columns,
        re_width: re_p,
    } = lower_random_effects(ast, data, family, n, p, numeric_main_col, factor_main_cols)?;

    // `weights`/`offset` are set here ONLY when the formula itself carries
    // them (`cbind()` response, `offset()` term). `orchestrate::run_fit` is
    // the double-specification check site for both — change together.
    let offset = offset_column(ast, data)?;
    let factor_levels: Vec<(String, Vec<String>)> = design_vars(ast, &re_groups)
        .into_iter()
        .filter_map(|v| match data.get(v) {
            Some(Column::Factor { levels, .. }) => Some((v.to_string(), levels.clone())),
            _ => None,
        })
        .collect();
    let numeric_means = numeric_means(ast, data);
    Ok(Lowered {
        x,
        y,
        n,
        p,
        col_names,
        re_columns,
        re_p,
        model,
        ids,
        re_groups,
        notes,
        weights,
        offset,
        terms: fixed_terms,
        has_intercept: ast.has_intercept,
        rdx,
        xtx: OnceLock::new(),
        factor_levels,
        numeric_means,
        ast: ast.clone(),
    })
}

/// [`Lowered::numeric_means`] of `ast` on `data`.
fn numeric_means(ast: &ParsedFormula, data: &Table) -> Vec<(String, f64)> {
    let mut out: Vec<(String, f64)> = Vec::new();
    for spelling in r_terms(ast)
        .into_iter()
        .flatten()
        .chain(ast.offset.as_deref())
    {
        // The variable `numeric_column` reads for this spelling.
        let name = match data.get(spelling) {
            Some(_) => spelling,
            None => match parse_transform(spelling) {
                Some((_, col)) => col,
                None => continue,
            },
        };
        if let Some(Column::Numeric(v)) = data.get(name) {
            if !out.iter().any(|(n, _)| n == name) {
                out.push((name.to_string(), v.iter().sum::<f64>() / v.len() as f64));
            }
        }
    }
    out
}

/// The `offset()` term of `ast` on `data`, `None` when the formula has none.
fn offset_column(ast: &ParsedFormula, data: &Table) -> Result<Option<Vec<f64>>, Error> {
    let Some(expr) = &ast.offset else {
        return Ok(None);
    };
    let o = numeric_column(expr, data)?;
    // Refused here, by term, rather than by the fit entry's `Data offset`
    // check, which names a field the formula user never set.
    if let Some(row) = o.iter().position(|v| !v.is_finite()) {
        return Err(Error::TransformNotFinite {
            term: format!("offset({expr})"),
            row,
        });
    }
    Ok(Some(o))
}

/// The random-effect terms of the groupings named `kept`, in formula order.
fn kept_nodes<'a, 'k>(
    ast: &'a ParsedFormula,
    kept: impl Iterator<Item = &'k String> + Clone,
) -> impl Iterator<Item = &'a RandomEffect> {
    ast.random_effects
        .iter()
        .filter(move |re| kept.clone().any(|g| *g == re_group_name(re)))
}

/// The variables the design reads, each once: the fixed terms' in R's term
/// order, then the slope variables of the kept groupings in formula order.
fn design_vars<'a>(ast: &'a ParsedFormula, re_groups: &'a [ReGroupInfo]) -> Vec<&'a str> {
    let slopes = kept_nodes(ast, re_groups.iter().map(|g| &g.name)).flat_map(|re| match re {
        RandomEffect::Intercept { .. } => &[][..],
        RandomEffect::Slope { vars, .. }
        | RandomEffect::NoIntercept { vars, .. }
        | RandomEffect::ZeroCorr { vars, .. } => &vars[..],
    });
    let mut out: Vec<&str> = Vec::new();
    for v in r_terms(ast)
        .into_iter()
        .flatten()
        .chain(slopes.map(String::as_str))
    {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}

#[cfg(test)]
mod tests;
