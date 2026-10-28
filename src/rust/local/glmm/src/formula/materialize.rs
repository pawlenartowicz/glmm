//! Data-dependent lowering — the only stage that touches data. Turns a
//! [`ParsedFormula`] + a [`Table`] into the numeric inputs [`crate::fit`]
//! consumes: a row-major design matrix, a structure-only [`ModelSpec`], per-row
//! [`GroupIds`], and the weights and offset the formula itself carries.
//!
//! Conventions (validated against R `model.matrix` / `lme4`):
//! - **Treatment contrasts**, base = first level — the caller's [`Column::Factor`]
//!   level order picks the reference level. [`Column::factor_from_labels`] supplies
//!   the lexicographic order R's `factor()` defaults to when the caller has none.
//!   Dummy names follow `paste0(var, level)`, e.g. `period2`. A factor keeps its
//!   base level (full indicator set) where R's `model.matrix` keeps it — see
//!   [`full_indicator_flags`].
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

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use super::error::Error;
use super::parse::{parse, parse_transform, ParsedFormula, RandomEffect, Transform};
use crate::{
    ColumnId, Data, Design, Family, Fit, FitError, FitOptions, GroupIds, Grouping,
    GroupingRelation, ModelSpec, PredictOptions, Prediction, ReStructure, ReTerm,
};

/// A single data column. A `Factor` carries its level order from the caller —
/// see [`Column::factor_from_labels`] for the no-declared-order default.
pub enum Column {
    /// A numeric column, used verbatim.
    Numeric(Vec<f64>),
    /// A categorical column, as levels + per-row codes.
    Factor {
        /// Distinct category levels **in the caller's intended order**. Level 0
        /// is the treatment-contrast base, so this order picks the reference
        /// level. Not required to be sorted, and not required to be exhausted
        /// by `codes`: an unused level yields an all-zero dummy column and thus
        /// an aliased (NA) coefficient, matching R's `model.matrix` on a factor
        /// with an unused level. A *grouping* factor drops its unused levels, as
        /// lme4 does: they get no random-effect slot and no `ranef` row.
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
pub struct Table {
    /// Name → column, each of length `n`, in column order.
    pub columns: Vec<(String, Column)>,
    /// Row count.
    pub n: usize,
}

impl Table {
    fn get(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|(n, _)| n == name).map(|(_, c)| c)
    }
}

/// One grouping factor's random-effect column names, in the order the kernel's
/// `varcorr`/`tau2` blocks use them — `"(Intercept)"` first, then any slopes —
/// plus the term each column belongs to and the label of every slot of its
/// RE block.
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
    /// same layer that CHOOSES the layout (`grouping_ids`), because that
    /// choice is data-dependent: a flat `(1|A) + (1|B)` routes nested or crossed
    /// depending on how balanced the data is (see `detect_flat_nesting`), so a
    /// consumer that inferred the layout would be wrong on a dataset nobody
    /// tested. [`label_ranef`] is the only reader.
    pub slot_labels: Vec<Option<String>>,
}

/// The lowered fit inputs — a one-shot front door to [`crate::fit`]:
/// [`Lowered::data`] and [`Lowered::spec`] are its first two arguments.
///
/// `Lowered` is `#[non_exhaustive]`: outside this crate it cannot be built by
/// struct literal, not even by struct update from a lowered value. Build one
/// with [`lower`] and assign its public fields. It also carries private
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
    /// Per-grouping name + RE term names + slot labels, in `varcorr`/`tau2`
    /// block order (primary first, then extras in declaration order). Empty when
    /// the formula declares no random effects.
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
    /// its levels in the fit's order. [`lower_new`] codes a new table's labels
    /// by them; a grouping factor is matched by its slot labels instead.
    pub(crate) factor_levels: Vec<(String, Vec<String>)>,
    /// The levels of a [`Self::factor_levels`] factor that no row of the fit
    /// has, per factor that has one. Their columns are aliased, so
    /// [`lower_new`] refuses them as levels the fit did not see.
    pub(crate) unseen_levels: Vec<(String, Vec<String>)>,
    /// Each numeric variable the fixed part reads, the `offset()` term's
    /// included, with its unweighted mean over the rows, in order of first
    /// appearance. A transform's variable is listed raw: `log(x)` gives the
    /// entry `x`, so a row at the mean reads `log(x̄)`, as emmeans' reference
    /// grid does.
    pub numeric_means: Vec<(String, f64)>,
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
/// what the formula requires.
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
}

/// Lowers `formula` against `table` ([`lower`]) and fits it with `opts`
/// ([`crate::fit`]).
///
/// # Errors
/// The errors of [`lower`], and [`Error::Fit`] when the kernel refuses the
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
/// [`Error::Fit`] when the kernel refuses the lowered input.
pub fn fit_lowered(formula: &str, lowered: Lowered, opts: &FitOptions) -> Result<Model, Error> {
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
/// [`Error::ZeroTrials`], and [`Error::NegativeCount`].
pub fn materialize(ast: &ParsedFormula, data: &Table, family: Family) -> Result<Lowered, Error> {
    check_table(data)?;
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
    let (model, ids, re_groups, notes, (re_columns, re_p)) =
        lower_random_effects(ast, data, family, n, p, numeric_main_col, factor_main_cols)?;

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
    let unseen_levels = factor_levels
        .iter()
        .filter_map(|(v, levels)| {
            let Some(Column::Factor { codes, .. }) = data.get(v) else {
                return None;
            };
            let mut seen = vec![false; levels.len()];
            for &c in codes {
                seen[c as usize] = true;
            }
            let unseen: Vec<String> = levels
                .iter()
                .zip(seen)
                .filter(|(_, s)| !s)
                .map(|(l, _)| l.clone())
                .collect();
            (!unseen.is_empty()).then(|| (v.clone(), unseen))
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
        unseen_levels,
        numeric_means,
    })
}

/// `data`'s columns checked against it: each `n` long and each factor code a
/// level. A longer column would index past the design, a shorter one would be
/// fitted as zeros, and a bad code would index past the levels.
fn check_table(data: &Table) -> Result<(), Error> {
    for (name, col) in &data.columns {
        let len = match col {
            Column::Numeric(v) => v.len(),
            Column::Factor { codes, .. } => codes.len(),
        };
        if len != data.n {
            return Err(Error::Fit(FitError::Shape(format!(
                "column {name:?} has {len} rows, the table {}",
                data.n
            ))));
        }
        if let Column::Factor { levels, codes } = col {
            if let Some(&bad) = codes.iter().find(|&&c| c as usize >= levels.len()) {
                return Err(Error::Fit(FitError::Shape(format!(
                    "factor column {name:?}: code {bad} is out of range for {} levels",
                    levels.len()
                ))));
            }
        }
    }
    Ok(())
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

/// The random-effect terms of the groupings a lowering kept (`re_groups`),
/// in formula order.
fn kept_nodes<'a>(
    ast: &'a ParsedFormula,
    re_groups: &'a [ReGroupInfo],
) -> impl Iterator<Item = &'a RandomEffect> {
    ast.random_effects
        .iter()
        .filter(|re| re_groups.iter().any(|g| g.name == re_group_name(re)))
}

/// The variables the design reads, each once: the fixed terms' in R's term
/// order, then the slope variables of the kept groupings in formula order.
fn design_vars<'a>(ast: &'a ParsedFormula, re_groups: &'a [ReGroupInfo]) -> Vec<&'a str> {
    let slopes = kept_nodes(ast, re_groups).flat_map(|re| match re {
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
/// label matched to its slot in [`ReGroupInfo::slot_labels`]. The response
/// is not read.
///
/// # Errors
/// [`Error::Fit`] with [`FitError::Shape`] for a level of a fixed factor or
/// a factor slope that the fit did not see, naming column, level and row (lme4
/// refuses it too); [`Error::WrongColumnKind`] for a variable whose kind is
/// not the one it had in the fit; the errors of [`fixed_design`],
/// [`re_only_columns`] and the offset as [`materialize`] returns them; and
/// [`Error::UnknownColumn`] for a grouping column missing while another is
/// present.
pub(crate) fn lower_new(formula: &str, lowered: &Lowered, table: &Table) -> Result<NewRows, Error> {
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
    let present = groupings
        .iter()
        .any(|g| g.name.split(':').any(|c| table.get(c).is_some()));
    let (re_columns, re_p, ids) = if present {
        let (re_columns, re_p) = re_only_columns(
            kept_nodes(&ast, groupings),
            &table,
            p,
            &mut numeric_main_col,
            &mut factor_main_cols,
        )?;
        assert_eq!(re_p, lowered.re_p, "the fit's random-effect-only columns");
        let mut ids = groupings.iter().map(|g| {
            // The node `lower_random_effects` takes the layout from.
            let mut nodes = ast
                .random_effects
                .iter()
                .filter(|re| re_group_name(re) == g.name);
            let first = nodes.clone().next().expect("a kept grouping has a term");
            let node = nodes
                .find(|re| {
                    matches!(
                        re,
                        RandomEffect::Intercept {
                            parent: Some(_),
                            ..
                        }
                    )
                })
                .unwrap_or(first);
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
                    Some(None) => Err(Error::Fit(FitError::Shape(format!(
                        "grouping {:?} has the label {l:?} at row {row}, which names two of \
                         the fit's levels; relabel its columns without ':'",
                        g.name
                    )))),
                    None => Ok(unseen),
                })
                .collect::<Result<Vec<u32>, Error>>()
        });
        let primary = ids.next().expect("a model with random effects")?;
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
                let unseen = lowered
                    .unseen_levels
                    .iter()
                    .find(|(f, _)| f == name)
                    .map_or(&[][..], |(_, l)| l);
                let index: HashMap<&str, u32> = fit
                    .iter()
                    .enumerate()
                    .filter(|(_, l)| !unseen.contains(l))
                    .map(|(k, l)| (l.as_str(), k as u32))
                    .collect();
                let codes = codes
                    .iter()
                    .enumerate()
                    .map(|(row, &c)| {
                        let label = &levels[c as usize];
                        index.get(label.as_str()).copied().ok_or_else(|| {
                            Error::Fit(FitError::Shape(format!(
                                "column {name:?} has the level {label:?} at row {row}, \
                                 which the fit did not see"
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
    assert_eq!(
        design.col_names, lowered.col_names,
        "the fit's levels and column kinds give the fit's columns"
    );
    Ok((table, design))
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
/// [`lower`] that apply to new rows ([`Error::UnknownColumn`],
/// [`Error::WrongColumnKind`], [`Error::TransformNotFinite`]); and
/// [`Error::Fit`] with the errors of [`crate::predict`].
pub fn predict(model: &Model, table: &Table, opts: &PredictOptions) -> Result<Prediction, Error> {
    let rows = lower_new(&model.formula, &model.lowered, table)?;
    crate::predict(&model.fit, model.lowered.spec(), &rows.design(), opts).map_err(Error::Fit)
}

/// The fixed-effects design of a formula on its data ([`fixed_design`]), and
/// what [`materialize`] reads from it to build the rest of [`Lowered`].
pub(crate) struct FixedDesign<'a> {
    /// Row-major `data.n × p` design, as [`Lowered::x`].
    pub(crate) x: Vec<f64>,
    /// Design width.
    pub(crate) p: usize,
    /// As [`Lowered::col_names`].
    pub(crate) col_names: Vec<String>,
    /// The fixed terms of [`r_terms`].
    terms: Vec<Vec<&'a str>>,
    /// As [`Lowered::terms`].
    fixed_terms: Vec<FixedTerm>,
    /// The random-slope resolution maps, described where [`fixed_design`]
    /// fills them.
    numeric_main_col: HashMap<String, ColumnId>,
    factor_main_cols: HashMap<String, Vec<(String, ColumnId)>>,
}

/// The fixed-effects design of `ast` on `data`: the intercept first (when the
/// formula has one), then each term's expanded columns, in R's term order.
/// The response, the random effects and the offset are not read.
///
/// # Errors
/// [`Error::UnknownColumn`], [`Error::WrongColumnKind`],
/// [`Error::SingleLevelFactor`], [`Error::EmptyDesign`] and
/// [`Error::TransformNotFinite`], as [`materialize`] returns them.
pub(crate) fn fixed_design<'a>(
    ast: &'a ParsedFormula,
    data: &Table,
) -> Result<FixedDesign<'a>, Error> {
    let n = data.n;
    let mut col_names: Vec<String> = Vec::new();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    if ast.has_intercept {
        col_names.push("(Intercept)".to_string());
        cols.push(vec![1.0; n]);
    }
    let terms = r_terms(ast);
    let full = full_indicator_flags(&terms, ast.has_intercept, data);
    // name → ColumnId for numeric main effects — the resolution map for
    // numeric random-slope variables.
    let mut numeric_main_col: HashMap<String, ColumnId> = HashMap::new();
    // name → its dummy (name, ColumnId) pairs, in `factor_dummies` emission
    // order — the resolution map for factor random-slope variables (a slope on
    // a categorical var expands to all its dummy columns, same treatment
    // contrasts the fixed-effect side already computed).
    let mut factor_main_cols: HashMap<String, Vec<(String, ColumnId)>> = HashMap::new();

    let mut fixed_terms: Vec<FixedTerm> = Vec::with_capacity(terms.len());
    for (vars, keep_base) in terms.iter().zip(&full) {
        let start = cols.len();
        match vars[..] {
            [name] => match data.get(name) {
                Some(Column::Factor { levels, codes }) => {
                    let promoted = keep_base[0];
                    let mut dummies = Vec::new();
                    for (suffix, col) in factor_dummies(name, levels, codes, promoted)? {
                        dummies.push((suffix.clone(), cols.len() as ColumnId));
                        col_names.push(suffix);
                        cols.push(col);
                    }
                    // The promotion (`keep_base`) is a FIXED-DESIGN coding
                    // convention only: R's contrast promotion recodes the
                    // bare factor term to a full indicator set on the fixed
                    // side, but a random slope on the same factor still
                    // mirrors lme4's own `model.matrix(~f)` for the term —
                    // intercept + treatment dummies — regardless of what the
                    // fixed side did. So the RE-slope resolution map
                    // (`factor_main_cols`) keeps only the non-base dummies
                    // here (skip the promoted base level); the fixed design
                    // above still emitted all of them.
                    let re_dummies = if promoted {
                        dummies[1..].to_vec()
                    } else {
                        dummies
                    };
                    factor_main_cols.insert(name.to_string(), re_dummies);
                }
                _ => {
                    let col = numeric_column(name, data)?;
                    numeric_main_col.insert(name.to_string(), cols.len() as ColumnId);
                    col_names.push(name.to_string());
                    cols.push(col);
                }
            },
            _ => {
                for (name, col) in interaction_columns(vars, keep_base, data)? {
                    col_names.push(name);
                    cols.push(col);
                }
            }
        }
        fixed_terms.push(FixedTerm {
            label: vars.join(":"),
            vars: vars.iter().map(|v| v.to_string()).collect(),
            numeric: vars
                .iter()
                .map(|v| !matches!(data.get(v), Some(Column::Factor { .. })))
                .collect(),
            columns: start..cols.len(),
        });
    }

    if cols.is_empty() {
        return Err(Error::EmptyDesign);
    }
    // Same refusal as the response's, per design column: a bare numeric
    // column, or an interaction product that overflowed.
    for (name, col) in col_names.iter().zip(&cols) {
        if let Some(row) = col.iter().position(|v| !v.is_finite()) {
            return Err(Error::TransformNotFinite {
                term: name.clone(),
                row,
            });
        }
    }
    Ok(FixedDesign {
        x: row_major(&cols, n),
        p: cols.len(),
        col_names,
        terms,
        fixed_terms,
        numeric_main_col,
        factor_main_cols,
    })
}

/// Columns of length `n` flattened to the row-major layout the fit expects:
/// element `(i, j)` at `i * cols.len() + j`.
fn row_major(cols: &[Vec<f64>], n: usize) -> Vec<f64> {
    let w = cols.len();
    let mut out = vec![0.0; n * w];
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            out[i * w + j] = v;
        }
    }
    out
}

/// lmerTest's `rdX` columns (see [`RdxCross`]) multiplied into the row-major
/// `n × p` design `x`. A factor is coded with every level in every term,
/// which is what `model.matrix(~ 0 + term)` does for a lone term; names
/// follow [`interaction_columns`], so each design column's name is also an
/// `rdX` name.
fn rdx_cross(
    terms: &[Vec<&str>],
    has_intercept: bool,
    data: &Table,
    x: &[f64],
    n: usize,
    p: usize,
) -> Result<RdxCross, Error> {
    let mut names = Vec::new();
    let mut cols: Vec<Vec<f64>> = Vec::new();
    if has_intercept {
        names.push("(Intercept)".to_string());
        cols.push(vec![1.0; n]);
    }
    for vars in terms {
        for (name, col) in interaction_columns(vars, &vec![true; vars.len()], data)? {
            names.push(name);
            cols.push(col);
        }
    }
    let mut xt_rdx = faer::Mat::<f64>::zeros(p, cols.len());
    for (c, col) in cols.iter().enumerate() {
        for (i, &r) in col.iter().enumerate() {
            for (j, &v) in x[i * p..(i + 1) * p].iter().enumerate() {
                xt_rdx[(j, c)] += v * r;
            }
        }
    }
    Ok(RdxCross { names, xt_rdx })
}

/// Sorted-unique levels and per-row codes — the lexicographic ordering
/// [`Column::factor_from_labels`] applies for a caller with no declared order,
/// and the one [`grouping_ids`]'s crossed-interaction arm applies to its
/// composite `(label, A, B)` keys.
fn sorted_levels_and_codes<T: Ord + std::hash::Hash + Clone>(labels: &[T]) -> (Vec<T>, Vec<u32>) {
    let levels: Vec<T> = labels
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let index: HashMap<&T, u32> = levels
        .iter()
        .enumerate()
        .map(|(i, l)| (l, i as u32))
        .collect();
    let codes = labels.iter().map(|l| index[l]).collect();
    (levels, codes)
}

/// Treatment-coded dummy columns for a factor (base = level 0, `levels-1`
/// columns), or with `keep_base` the full indicator set (`levels` columns,
/// base included) wherever [`full_indicator_flags`] says R keeps the base.
/// Names follow the module-header treatment-contrasts convention.
///
/// # Errors
/// [`Error::SingleLevelFactor`] when `levels` has fewer than 2 entries — R's
/// `contrasts<-` refuses the same thing, regardless of `keep_base`: a
/// single-level factor never carries information for the fixed part, whether
/// it would be treatment-coded or given a full indicator.
fn factor_dummies(
    var: &str,
    levels: &[String],
    codes: &[u32],
    keep_base: bool,
) -> Result<Vec<(String, Vec<f64>)>, Error> {
    if levels.len() < 2 {
        return Err(Error::SingleLevelFactor {
            name: var.to_string(),
        });
    }
    Ok(levels
        .iter()
        .enumerate()
        .skip(if keep_base { 0 } else { 1 })
        .map(|(li, lvl)| {
            let col: Vec<f64> = codes
                .iter()
                .map(|&c| if c as usize == li { 1.0 } else { 0.0 })
                .collect();
            (format!("{var}{lvl}"), col)
        })
        .collect())
}

/// A numeric design column by spelling: a bare column, or a whitelisted
/// transform of one (`parse_transform`). A bare name that IS a column wins
/// over a transform reading, so a caller who already computed `log(x)` into a
/// column of that name gets it verbatim.
///
/// Evaluation mirrors R's arithmetic on the same libm so `tests/contrasts_oracle.rs`
/// can compare within `|a-b| < 1e-12`: `log`/`sqrt`/`exp` are the libm calls R
/// makes; `x^2` is `x*x` in R (`R_POW` special-cases 2) and every other integer
/// power goes through libm `pow`, hence `powf`, not `powi`.
fn numeric_column(name: &str, data: &Table) -> Result<Vec<f64>, Error> {
    match data.get(name) {
        Some(Column::Numeric(v)) => return Ok(v.clone()),
        Some(Column::Factor { .. }) => {
            return Err(Error::WrongColumnKind {
                name: name.to_string(),
                expected: "numeric",
            })
        }
        None => {}
    }
    let Some((t, col)) = parse_transform(name) else {
        return Err(Error::UnknownColumn(name.to_string()));
    };
    let v = match data.get(col) {
        Some(Column::Numeric(v)) => v,
        Some(Column::Factor { .. }) => {
            return Err(Error::WrongColumnKind {
                name: col.to_string(),
                expected: "numeric",
            })
        }
        None => return Err(Error::UnknownColumn(col.to_string())),
    };
    let out: Vec<f64> = v
        .iter()
        .map(|&x| match t {
            Transform::Log => x.ln(),
            Transform::Sqrt => x.sqrt(),
            Transform::Exp => x.exp(),
            Transform::Pow(2) => x * x,
            Transform::Pow(k) => x.powf(f64::from(k)),
        })
        .collect();
    if let Some(i) = out.iter().position(|x| !x.is_finite()) {
        return Err(Error::TransformNotFinite {
            term: name.to_string(),
            row: i,
        });
    }
    Ok(out)
}

/// The fixed terms as R lays them out (`TermsForm` in R's `model.c`, behind
/// `terms.formula`), one variable list per term. Variables are numbered by
/// their first appearance in the fixed part (`ast.predictors`; lme4 builds its
/// fixed design from `nobars(formula)`, so a variable seen only in a random
/// term takes no place), and each term lists its variables in that order.
/// Terms are then stably sorted by degree: main effects as written, then the
/// two-way interactions as written, and so on. `y ~ x + c:b:a + a*b*c` thus
/// becomes `x a b c b:a c:a c:b c:b:a`.
pub(crate) fn r_terms(ast: &ParsedFormula) -> Vec<Vec<&str>> {
    use super::parse::Term;
    let rank = |v: &str| {
        ast.predictors
            .iter()
            .position(|p| p == v)
            .unwrap_or(usize::MAX)
    };
    let mut terms: Vec<Vec<&str>> = ast
        .terms
        .iter()
        .map(|t| {
            let mut vars: Vec<&str> = match t {
                Term::Main { name } => vec![name.as_str()],
                Term::Interaction { vars } => vars.iter().map(String::as_str).collect(),
            };
            vars.sort_by_key(|v| rank(v));
            vars
        })
        .collect();
    terms.sort_by_key(Vec::len);
    terms
}

/// Per fixed term of [`r_terms`], one flag per variable: `true` where a factor
/// is coded with its full indicator set, `false` where it gets treatment
/// contrasts. Flags on numeric variables are never read.
///
/// This is R's rule (`TermCode` in R's `model.c`; Chambers & Hastie 1992,
/// *Statistical Models in S*): a factor in a term gets contrasts when the
/// term with that factor removed is empty (the intercept's place) or is
/// contained in a term placed before it. Without an intercept, R's
/// `model.matrix` also gives full indicators to the first factor, in variable
/// order, of the first term that has one. `y ~ f + f:g` thus codes `g` with
/// contrasts and `f` with indicators inside `f:g` (`fa:gq fb:gq`), and
/// `y ~ x:f` keeps every level of `f` (`x:fa x:fb`).
fn full_indicator_flags(terms: &[Vec<&str>], has_intercept: bool, data: &Table) -> Vec<Vec<bool>> {
    let is_factor = |v: &str| matches!(data.get(v), Some(Column::Factor { .. }));
    let mut full: Vec<Vec<bool>> = terms.iter().map(|v| vec![false; v.len()]).collect();
    for (t, vars) in terms.iter().enumerate() {
        for (k, &v) in vars.iter().enumerate() {
            if !is_factor(v) {
                continue;
            }
            let rest: Vec<&str> = vars.iter().copied().filter(|&u| u != v).collect();
            let covered = rest.is_empty()
                || terms[..t]
                    .iter()
                    .any(|e| rest.iter().all(|u| e.contains(u)));
            full[t][k] = !covered;
        }
    }
    if !has_intercept {
        if let Some((t, k)) = terms
            .iter()
            .enumerate()
            .find_map(|(t, vars)| vars.iter().position(|v| is_factor(v)).map(|k| (t, k)))
        {
            full[t][k] = true;
        }
    }
    full
}

/// Expand one variable to its design columns: a numeric (or transform) → one
/// column named after the variable; a factor → its treatment dummies, or all
/// its indicators with `keep_base`.
fn expand_var(name: &str, keep_base: bool, data: &Table) -> Result<Vec<(String, Vec<f64>)>, Error> {
    match data.get(name) {
        Some(Column::Factor { levels, codes }) => factor_dummies(name, levels, codes, keep_base),
        _ => Ok(vec![(name.to_string(), numeric_column(name, data)?)]),
    }
}

/// Interaction columns formed from the module-header conventions: elementwise
/// product across the vars' expanded sets, earliest var varying fastest, names
/// joined by `:`. `vars` is one term of [`r_terms`] (variable order) and
/// `keep_base` its row of [`full_indicator_flags`].
fn interaction_columns(
    vars: &[&str],
    keep_base: &[bool],
    data: &Table,
) -> Result<Vec<(String, Vec<f64>)>, Error> {
    let mut acc = expand_var(vars[0], keep_base[0], data)?;
    for (var, &kb) in vars[1..].iter().zip(&keep_base[1..]) {
        let next = expand_var(var, kb, data)?;
        let mut out = Vec::with_capacity(acc.len() * next.len());
        // Later var outer, earlier var inner → earliest contrasts vary fastest.
        for (n2, c2) in &next {
            for (n1, c1) in &acc {
                let col: Vec<f64> = c1.iter().zip(c2).map(|(a, b)| a * b).collect();
                out.push((format!("{n1}:{n2}"), col));
            }
        }
        acc = out;
    }
    Ok(acc)
}

/// A grouping column's levels and per-row codes: borrowed from a factor column,
/// built for a whole-number one (see [`grouping_factor`]).
type LevelsAndCodes<'a> = (Cow<'a, [String]>, Cow<'a, [u32]>);

/// A grouping column's `(levels, codes)`, the levels no row carries dropped (R's
/// `droplevels`, which lme4 applies to a grouping factor). The codes ARE the
/// per-row ids, dense in `0..levels.len()` and so each below `n`: a level with no
/// row would cost a random-effect slot and get a fully shrunk `ranef` row. A
/// grouping's level order is arbitrary as far as the fit is concerned (its RE
/// levels are exchangeable), so a factor's own order is kept.
///
/// A numeric column whose values are all whole numbers (an integer column, as
/// `read.csv` and `pd.read_csv` load an id) is taken as a factor of its distinct
/// values in ascending order, labelled as integers — the levels lme4 gets from
/// `factor()` on that column. Any other numeric column is refused.
fn grouping_factor<'a>(name: &str, data: &'a Table) -> Result<LevelsAndCodes<'a>, Error> {
    match data.get(name) {
        Some(Column::Factor { levels, codes }) => {
            let mut used = vec![false; levels.len()];
            for &c in codes {
                used[c as usize] = true;
            }
            if used.iter().all(|&u| u) {
                return Ok((Cow::Borrowed(levels), Cow::Borrowed(codes)));
            }
            let mut remap = vec![0u32; levels.len()];
            let mut kept = Vec::new();
            for (l, _) in used.iter().enumerate().filter(|&(_, &u)| u) {
                remap[l] = kept.len() as u32;
                kept.push(levels[l].clone());
            }
            let codes = codes.iter().map(|&c| remap[c as usize]).collect();
            Ok((Cow::Owned(kept), Cow::Owned(codes)))
        }
        Some(Column::Numeric(v)) => {
            if !v.iter().all(|x| x.is_finite() && x.fract() == 0.0) {
                return Err(Error::WrongColumnKind {
                    name: name.to_string(),
                    expected: "a factor or a column of whole numbers (grouping variable)",
                });
            }
            // `+ 0.0` folds -0.0 into 0.0, which `total_cmp` would keep apart.
            let mut values: Vec<f64> = v.iter().map(|x| x + 0.0).collect();
            values.sort_by(f64::total_cmp);
            values.dedup();
            let codes = v
                .iter()
                .map(|x| {
                    values
                        .binary_search_by(|u| u.total_cmp(&(x + 0.0)))
                        .unwrap() as u32
                })
                .collect();
            let levels = values.iter().map(|u| format!("{u}")).collect();
            Ok((Cow::Owned(levels), Cow::Owned(codes)))
        }
        None => Err(Error::UnknownColumn(name.to_string())),
    }
}

/// Per-row grouping labels, rebuilt as `levels[codes[i]]` — owned, because
/// [`Column::Factor`] stores levels + codes and has no per-row label slice to
/// borrow. Only the two arms that need label *identity across two columns* (the
/// nested and crossed composites) pay this; the plain arm reads `codes` directly.
fn grouping_row_labels(name: &str, data: &Table) -> Result<Vec<String>, Error> {
    let (levels, codes) = grouping_factor(name, data)?;
    Ok(codes.iter().map(|&c| levels[c as usize].clone()).collect())
}

/// Padded-per-parent child ids: given already-dense `parent_ids` (one per row)
/// and per-row child `labels`, lay the children out as CONTIGUOUS PER-PARENT
/// BLOCKS in the same order the parent's own ids use — parent id `p`'s children
/// occupy `[p·W, p·W + k_p)` where `k_p` is that parent's distinct child count
/// and `W = max_p k_p` (the true max — every block gets the same width, since
/// the kernel's `NestedWithin` sizing is a fixed-width rectangle: `src/lmm/kernel.rs`'s
/// `add_rows_multi` computes a child's RE column as
/// `extra_offsets[e] + id·extra_q[e]`, which only lands in the right parent's
/// block under this padded-block layout — a fresh global lexicographic sort of
/// the joined label does not preserve per-parent contiguity for unbalanced
/// nesting). Slots `k_p..W` in a shorter parent's block are never assigned to
/// any row (padding). Assumes each child label occurs under a SINGLE parent —
/// the caller must have verified nesting (explicit `parent:child` syntax, or
/// [`detect_flat_nesting`] for the flat idiom).
///
/// Returns the ids alongside one label per slot of the padded rectangle
/// (`n_parents · W`, slot `p·W + k`): the child label at every assigned slot and
/// `None` at every padded one. The caller joins the parent's own label in when
/// the formula named a parent (`(1|A/B)`); the flat idiom keeps the child label
/// bare, because `re_group_info` names that block for the child alone and the
/// route it takes is data-dependent.
fn nested_padded_ids(parent_ids: &[u32], labels: &[String]) -> (Vec<u32>, Vec<Option<String>>) {
    let n_parents = parent_ids
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(1);
    // Distinct child labels observed under each parent id, lexicographic
    // (BTreeSet) — the within-parent dense order; the specific order among
    // children of one parent doesn't matter statistically (an exchangeable RE),
    // only that it is dense over `0..k_p`.
    let mut children_per_parent: Vec<BTreeSet<&str>> = vec![BTreeSet::new(); n_parents];
    for (&p, c) in parent_ids.iter().zip(labels) {
        children_per_parent[p as usize].insert(c.as_str());
    }
    let n_per_parent = children_per_parent
        .iter()
        .map(BTreeSet::len)
        .max()
        .unwrap_or(1)
        .max(1);
    let local_index: Vec<HashMap<&str, u32>> = children_per_parent
        .iter()
        .map(|set| {
            set.iter()
                .enumerate()
                .map(|(i, &l)| (l, i as u32))
                .collect()
        })
        .collect();
    let mut slot_labels: Vec<Option<String>> = vec![None; n_parents * n_per_parent];
    for (p, set) in children_per_parent.iter().enumerate() {
        for (k, &child) in set.iter().enumerate() {
            slot_labels[p * n_per_parent + k] = Some(child.to_string());
        }
    }
    let ids = parent_ids
        .iter()
        .zip(labels)
        .map(|(&p, c)| p * n_per_parent as u32 + local_index[p as usize][c.as_str()])
        .collect();
    (ids, slot_labels)
}

/// Max padding-inflation factor for flat-nesting detection: detect nested only
/// while `n_parents · max_children ≤ NESTING_INFLATION_BOUND ·
/// n_distinct_child_levels`. The guard protects the padded rectangle from
/// inflating the dense RE block — not balance per se: near-balanced nesting
/// (children-per-parent ∈ {1,2,3}, inflation ~1.6×) pads a few empty slots and
/// stays far cheaper than routing its tens of thousands of levels crossed
/// (dense tail is cubic in crossed levels), while an observation-level factor
/// like grouseticks' INDEX (each row its own level, inflation ≫ 2) still fails
/// closed to `Crossed` (measured there: nested 44 s vs crossed 0.16 s).
const NESTING_INFLATION_BOUND: usize = 2;

/// Detect nesting of a flat extra grouping (the lme4 idiom
/// `(1|parent)+(1|child)`, no explicit `parent:child` syntax) from the observed
/// id structure, returning its [`nested_padded_ids`] layout iff routing it
/// nested is both correct AND a win. Two guards, both fail closed to `Crossed`:
///
/// 1. **Genuine nesting** — every distinct `child_labels` value must fall under a
///    single `primary_ids` (parent). A label spanning two parents is genuinely
///    crossed; routing it nested would corrupt the padded family-block Cholesky.
/// 2. **Bounded padding inflation** — the padded rectangle
///    `n_parents · max_children` must not exceed [`NESTING_INFLATION_BOUND`] ×
///    the distinct child-level count (see the constant's rationale). Explicit
///    `parent:child` syntax still pads unbounded — there the user asked for the
///    nested layout outright.
fn detect_flat_nesting(
    primary_ids: &[u32],
    child_labels: &[String],
) -> Option<(Vec<u32>, Vec<Option<String>>)> {
    let n_parents = primary_ids
        .iter()
        .copied()
        .max()
        .map(|m| m as usize + 1)
        .unwrap_or(1);
    let mut parent_of: HashMap<&str, u32> = HashMap::new();
    let mut children_per_parent = vec![0usize; n_parents];
    for (&p, c) in primary_ids.iter().zip(child_labels) {
        match parent_of.insert(c.as_str(), p) {
            Some(prev) if prev != p => return None, // spans two parents → crossed
            Some(_) => {}                           // repeat obs of a seen child
            None => children_per_parent[p as usize] += 1, // first sighting
        }
    }
    let w = children_per_parent.iter().copied().max().unwrap_or(0);
    if n_parents * w > NESTING_INFLATION_BOUND * parent_of.len() {
        return None; // padded rectangle would inflate the dense RE block
    }
    Some(nested_padded_ids(primary_ids, child_labels))
}

/// One grouping's lowered level layout. `slot_labels` is [`ReGroupInfo`]'s
/// field.
struct GroupingLayout {
    ids: Vec<u32>,
    slot_labels: Vec<Option<String>>,
}

impl GroupingLayout {
    /// A layout whose every slot is an observed level, in the order `labels`
    /// gives them — the crossed-interaction and plain cases.
    fn all_observed(ids: Vec<u32>, labels: Vec<String>) -> Self {
        GroupingLayout {
            ids,
            slot_labels: labels.into_iter().map(Some).collect(),
        }
    }
}

/// A random-effect term's own grouping factor's distinct OBSERVED level
/// count — what lme4 checks against `> 1` ("grouping factors must have > 1
/// sampled level"). This is independent of how [`grouping_ids`] will end up
/// routing the term (nested, crossed, or plain): a plain grouping's count is
/// its own column's distinct observed code, and an explicit `parent:child`
/// or crossed `A:B` grouping's is the distinct observed `(A, B)` pair
/// count — the same composite factor lme4 builds via `interaction()` before
/// counting its levels, not the padded rectangle a nested layout later sizes
/// its RE block to. Pairs are counted as pairs, never as the joined `"A:B"`
/// string, which two different pairs share when a label contains `:`.
fn grouping_level_count(re: &RandomEffect, data: &Table) -> Result<usize, Error> {
    let group = re_group_name(re);
    if let RandomEffect::Intercept {
        parent: Some(parent),
        ..
    } = re
    {
        let child = group.strip_prefix(&format!("{parent}:")).unwrap_or(&group);
        let a = grouping_row_labels(parent, data)?;
        let b = grouping_row_labels(child, data)?;
        let distinct: BTreeSet<(&String, &String)> = a.iter().zip(&b).collect();
        return Ok(distinct.len());
    }
    if let Some((lhs, rhs)) = group.split_once(':') {
        let a = grouping_row_labels(lhs, data)?;
        let b = grouping_row_labels(rhs, data)?;
        let distinct: BTreeSet<(&String, &String)> = a.iter().zip(&b).collect();
        return Ok(distinct.len());
    }
    Ok(grouping_factor(&group, data)?.0.len())
}

/// Per-row level labels of `re`'s grouping, spelled as its
/// [`ReGroupInfo::slot_labels`] spell them, so [`grouping_ids`] labels its
/// slots with them and [`lower_new`] matches a new row to its slot by them.
/// A nested inner factor (`(1 | A/B)`) and a crossed interaction grouping
/// (`(1 | A:B)`) join the two labels as `a:b`, any other grouping is its own
/// label. The nested order is PARENT-FIRST: lme4 spells the same level
/// child-first (`b1:a1`); both label the same thing, and this order matches
/// the one the formula and the grouping's own name already use.
fn grouping_labels(re: &RandomEffect, data: &Table) -> Result<Vec<String>, Error> {
    let pair = match re {
        RandomEffect::Intercept {
            group,
            parent: Some(parent),
        } => Some((
            parent.as_str(),
            group.strip_prefix(&format!("{parent}:")).unwrap_or(group),
        )),
        RandomEffect::Intercept {
            group,
            parent: None,
        } => group.split_once(':'),
        _ => None,
    };
    let Some((lhs, rhs)) = pair else {
        return grouping_row_labels(&re_group_name(re), data);
    };
    let a = grouping_row_labels(lhs, data)?;
    let b = grouping_row_labels(rhs, data)?;
    Ok(a.iter().zip(&b).map(|(x, y)| format!("{x}:{y}")).collect())
}

/// Dense per-row ids for one grouping, plus the label of every slot of its RE
/// block. A nested inner factor `A:B` (explicit parent `A`) routes through
/// [`nested_padded_ids`]; a crossed interaction and a plain grouping use a flat
/// global lexicographic code.
fn grouping_ids(re: &RandomEffect, data: &Table) -> Result<GroupingLayout, Error> {
    match re {
        RandomEffect::Intercept {
            group,
            parent: Some(parent),
        } => {
            // Nested inner: strip the `parent:` prefix to get the child column.
            let child = group.strip_prefix(&format!("{parent}:")).unwrap_or(group);
            // Parent ids: identical computation to the plain-grouping arm below
            // applied to `parent` — guarantees this matches the primary's own ids
            // exactly when `parent` names the primary grouping.
            let (_, parent_ids) = grouping_factor(parent, data)?;
            let (ids, slots) = nested_padded_ids(&parent_ids, &grouping_row_labels(child, data)?);
            // Each slot owned by a row takes that row's `parent:child` label;
            // a padded one stays `None`.
            let mut slot_labels = vec![None; slots.len()];
            for (&id, label) in ids.iter().zip(grouping_labels(re, data)?) {
                slot_labels[id as usize] = Some(label);
            }
            Ok(GroupingLayout { ids, slot_labels })
        }
        RandomEffect::Intercept {
            group,
            parent: None,
        } if group.contains(':') => {
            // Crossed interaction grouping, e.g. `(1|recipe:replicate)`: `group`
            // itself (not a `parent:child` relation — `parent` is None) names two
            // existing factor columns joined by `:`. Every observed `(A,B)` pair
            // is its own level — composite-label materialization like the nested
            // case above, but with no parent grouping created.
            //
            // The composite's own level order is lexicographic on the joined
            // label, NOT inherited from either component's declared order: the
            // pair has no order the caller ever stated, and a grouping's level
            // order is inference-invariant anyway (exchangeable RE levels).
            //
            // A level is the `(A, B)` pair, not the joined label: with a `:`
            // inside a label, `("a:b", "c")` and `("a", "b:c")` both join to
            // `"a:b:c"` but are two levels. The pair breaks ties after the
            // joined label, so such levels share a display label and stay apart.
            let (lhs, rhs) = group
                .split_once(':')
                .expect("group contains ':' per guard above");
            let a = grouping_row_labels(lhs, data)?;
            let b = grouping_row_labels(rhs, data)?;
            let keys: Vec<(String, &str, &str)> = grouping_labels(re, data)?
                .into_iter()
                .zip(a.iter().zip(&b))
                .map(|(label, (x, y))| (label, x.as_str(), y.as_str()))
                .collect();
            let (levels, codes) = sorted_levels_and_codes(&keys);
            let labels = levels.into_iter().map(|(label, _, _)| label).collect();
            Ok(GroupingLayout::all_observed(codes, labels))
        }
        RandomEffect::Intercept { group, .. }
        | RandomEffect::Slope { group, .. }
        | RandomEffect::NoIntercept { group, .. }
        | RandomEffect::ZeroCorr { group, .. } => {
            // The codes ARE the ids, so slot `l` is level `l`.
            let (levels, codes) = grouping_factor(group, data)?;
            Ok(GroupingLayout::all_observed(
                codes.into_owned(),
                levels.into_owned(),
            ))
        }
    }
}

/// Resolve random-slope variables to their columns, each with its label: a
/// numeric slope var to its single column; a factor slope var (not in
/// `numeric_cols`) to ALL of its dummy columns, in `factor_dummies` order and
/// named as there — so the returned vec can be longer than `vars` — and sets
/// the returned flag. The maps hold the fixed main terms' design columns and,
/// from [`re_only_columns`], every other slope variable's columns at `p + j`,
/// so every slope variable is in one of them.
fn slope_cols(
    vars: &[String],
    numeric_cols: &HashMap<String, ColumnId>,
    factor_cols: &HashMap<String, Vec<(String, ColumnId)>>,
) -> (Vec<(String, ColumnId)>, bool) {
    let mut out = Vec::new();
    let mut has_factor = false;
    for v in vars {
        if let Some(&cid) = numeric_cols.get(v) {
            out.push((v.clone(), cid));
        } else {
            out.extend(factor_cols[v].iter().cloned());
            has_factor = true;
        }
    }
    (out, has_factor)
}

/// The random-effect-only columns of the terms `nodes` ([`Lowered::re_columns`]),
/// row-major `n × s`, and `s`: each slope variable that is in neither map
/// (no fixed main term; a variable in `X` only inside an interaction
/// included, as lme4's `model.matrix` of the term would build it) gets its
/// column, or a factor its K − 1 treatment dummies, the first time a term
/// names it. Each is added to its map at `p + j`, `j` its first column here,
/// so every grouping that names the variable reads the same columns.
fn re_only_columns<'a>(
    nodes: impl IntoIterator<Item = &'a RandomEffect>,
    data: &Table,
    p: usize,
    numeric_cols: &mut HashMap<String, ColumnId>,
    factor_cols: &mut HashMap<String, Vec<(String, ColumnId)>>,
) -> Result<(Vec<f64>, usize), Error> {
    let mut cols: Vec<Vec<f64>> = Vec::new();
    for re in nodes {
        let (RandomEffect::Slope { vars, .. }
        | RandomEffect::NoIntercept { vars, .. }
        | RandomEffect::ZeroCorr { vars, .. }) = re
        else {
            continue;
        };
        for v in vars {
            if numeric_cols.contains_key(v) || factor_cols.contains_key(v) {
                continue;
            }
            let set = expand_var(v, false, data)?;
            // The design columns' refusal, by name (`fixed_design`).
            for (name, col) in &set {
                if let Some(row) = col.iter().position(|v| !v.is_finite()) {
                    return Err(Error::TransformNotFinite {
                        term: name.clone(),
                        row,
                    });
                }
            }
            let at = |k: usize| (p + cols.len() + k) as ColumnId;
            if matches!(data.get(v), Some(Column::Factor { .. })) {
                let ids = set
                    .iter()
                    .enumerate()
                    .map(|(k, (name, _))| (name.clone(), at(k)));
                factor_cols.insert(v.clone(), ids.collect());
            } else {
                numeric_cols.insert(v.clone(), at(0));
            }
            cols.extend(set.into_iter().map(|(_, col)| col));
        }
    }
    Ok((row_major(&cols, data.n), cols.len()))
}

/// One random-effect term after factor coding: whether it holds the
/// intercept, and its slope columns with their labels.
struct LoweredTerm {
    intercept: bool,
    cols: Vec<(String, ColumnId)>,
}

/// The terms that name one grouping (`nodes`, in AST order), lowered to the
/// block's terms in block order: the intercept's term first, the others as
/// written.
///
/// A factor slope lowers as the crate codes it, with K − 1 dummies, so a term
/// without the intercept that holds a factor, `(0 + f | g)`, is lowered as
/// `(1 + f | g)`, the same unstructured model. An intercept-only term on the
/// grouping merges into it: lme4 fits `(f || g)`, its `(1 | g) + (0 + f | g)`,
/// with an extra intercept variance that the factor's full indicator set
/// leaves unidentified, and the model space is the same. `zerocorr` gives the
/// intercept and each design column its own term.
///
/// # Errors
/// [`Error::RandomColumnRepeated`] when a column, the intercept included, is
/// in two terms; and a syntax error for `zerocorr(0 + f | g)` on a factor `f`, whose diagonal over
/// the factor's levels the K − 1 coding cannot write.
fn lower_terms(
    group: &str,
    nodes: &[&RandomEffect],
    numeric_cols: &HashMap<String, ColumnId>,
    factor_cols: &HashMap<String, Vec<(String, ColumnId)>>,
) -> Result<Vec<LoweredTerm>, Error> {
    let resolve = |vars: &[String]| slope_cols(vars, numeric_cols, factor_cols);
    let mut terms: Vec<LoweredTerm> = Vec::new();
    let mut promoted = false;
    for node in nodes {
        match node {
            RandomEffect::Intercept { .. } => terms.push(LoweredTerm {
                intercept: true,
                cols: Vec::new(),
            }),
            RandomEffect::Slope { vars, .. } => terms.push(LoweredTerm {
                intercept: true,
                cols: resolve(vars).0,
            }),
            RandomEffect::NoIntercept { vars, .. } => {
                let (cols, has_factor) = resolve(vars);
                promoted |= has_factor;
                terms.push(LoweredTerm {
                    intercept: has_factor,
                    cols,
                });
            }
            RandomEffect::ZeroCorr {
                vars, intercept, ..
            } => {
                let (cols, has_factor) = resolve(vars);
                if has_factor && !intercept {
                    return Err(Error::Parse(super::ParseError::Syntax {
                        pos: 0,
                        msg: format!(
                            "zerocorr(0 + {} | {group}): a factor slope in a zerocorr term \
                             needs the intercept; write zerocorr(1 + {} | {group})",
                            vars.join(" + "),
                            vars.join(" + ")
                        ),
                    }));
                }
                if *intercept {
                    terms.push(LoweredTerm {
                        intercept: true,
                        cols: Vec::new(),
                    });
                }
                terms.extend(cols.into_iter().map(|c| LoweredTerm {
                    intercept: false,
                    cols: vec![c],
                }));
            }
        }
    }
    if promoted {
        if let Some(k) = terms.iter().position(|t| t.intercept && t.cols.is_empty()) {
            terms.remove(k);
        }
    }
    let repeated = |column: &str| Error::RandomColumnRepeated {
        group: group.to_string(),
        column: column.to_string(),
    };
    if terms.iter().filter(|t| t.intercept).count() > 1 {
        return Err(repeated("(Intercept)"));
    }
    let mut seen = BTreeSet::new();
    for (label, cid) in terms.iter().flat_map(|t| &t.cols) {
        if !seen.insert(*cid) {
            return Err(repeated(label));
        }
    }
    terms.sort_by_key(|t| !t.intercept);
    Ok(terms)
}

/// A grouping's spec block from its [`lower_terms`]: the terms and the slope
/// columns, the terms' columns in block order. One term holding the
/// intercept is the unstructured block every one-term grouping has, written
/// with no terms.
fn block_of(terms: &[LoweredTerm]) -> (Vec<ReTerm>, Vec<ColumnId>) {
    let slopes = terms
        .iter()
        .flat_map(|t| t.cols.iter().map(|(_, cid)| *cid))
        .collect();
    if let [t] = terms {
        if t.intercept {
            return (Vec::new(), slopes);
        }
    }
    let re_terms = terms
        .iter()
        .map(|t| {
            let term = if t.intercept {
                ReTerm::intercept()
            } else {
                ReTerm::no_intercept()
            };
            term.slopes(t.cols.iter().map(|(_, cid)| *cid))
        })
        .collect();
    (re_terms, slopes)
}

/// Build one grouping's `ReGroupInfo` from its [`lower_terms`] —
/// `"(Intercept)"` first when a term holds it, then the slope columns, in
/// block order (mirrors `varcorr`/`tau2` block term order). A factor slope's
/// columns are its dummy names (`format!("{var}{lvl}")`, same naming
/// `factor_dummies` uses on the fixed-effect side), not the bare var name.
/// `slot_labels` comes from [`grouping_ids`], which is the only place the chosen
/// layout and the level labels are both in scope.
fn re_group_info(
    name: &str,
    terms: &[LoweredTerm],
    slot_labels: Vec<Option<String>>,
) -> ReGroupInfo {
    let mut labels = Vec::new();
    let mut term_index = Vec::new();
    for (k, t) in terms.iter().enumerate() {
        if t.intercept {
            labels.push("(Intercept)".to_string());
            term_index.push(k);
        }
        for (label, _) in &t.cols {
            labels.push(label.clone());
            term_index.push(k);
        }
    }
    ReGroupInfo {
        name: name.to_string(),
        terms: labels,
        term_index,
        slot_labels,
    }
}

/// A random-effect term's random effects per level: its columns after factor
/// coding, the intercept included, as lme4's `check.nobs.vs.nRE` counts them.
fn term_width(
    re: &RandomEffect,
    numeric_cols: &HashMap<String, ColumnId>,
    factor_cols: &HashMap<String, Vec<(String, ColumnId)>>,
) -> usize {
    let resolve = |vars: &[String]| slope_cols(vars, numeric_cols, factor_cols);
    match re {
        RandomEffect::Intercept { .. } => 1,
        RandomEffect::Slope { vars, .. } => 1 + resolve(vars).0.len(),
        RandomEffect::NoIntercept { vars, .. } => {
            let (cols, has_factor) = resolve(vars);
            cols.len() + usize::from(has_factor)
        }
        RandomEffect::ZeroCorr {
            vars, intercept, ..
        } => usize::from(*intercept) + resolve(vars).0.len(),
    }
}

/// What [`lower_random_effects`] gives [`Lowered`]: the model, the ids,
/// [`Lowered::re_groups`], the notes, and [`Lowered::re_columns`] with its
/// width.
type LoweredRandomEffects = (
    ModelSpec,
    GroupIds,
    Vec<ReGroupInfo>,
    Vec<crate::Note>,
    (Vec<f64>, usize),
);

fn lower_random_effects(
    ast: &ParsedFormula,
    data: &Table,
    family: Family,
    n: usize,
    p: usize,
    mut numeric_cols: HashMap<String, ColumnId>,
    mut factor_cols: HashMap<String, Vec<(String, ColumnId)>>,
) -> Result<LoweredRandomEffects, Error> {
    if ast.random_effects.is_empty() {
        return Ok((
            ModelSpec { family, re: None },
            GroupIds::default(),
            Vec::new(),
            Vec::new(),
            (Vec::new(), 0),
        ));
    }

    // The terms of each grouping, groupings in order of first appearance.
    let mut groups: Vec<(String, Vec<&RandomEffect>)> = Vec::new();
    for re in &ast.random_effects {
        let name = re_group_name(re);
        match groups.iter_mut().find(|(g, _)| *g == name) {
            Some((_, nodes)) => nodes.push(re),
            None => groups.push((name, vec![re])),
        }
    }

    // A grouping with only one observed level carries no information about a
    // between-group variance (lme4: "grouping factors must have > 1 sampled
    // level"). Checked up front, independent of the nested/crossed routing
    // decided below (see `grouping_level_count`), and dropped with a warning
    // rather than an error — the rest of the model, including any OTHER
    // random-effect term, still fits. Filtering the grouping list before
    // anything below picks a primary means a dropped primary correctly
    // promotes the next surviving grouping, exactly as if the formula had
    // never declared the dropped one.
    let mut notes: Vec<crate::Note> = Vec::new();
    let mut kept: Vec<(String, Vec<&RandomEffect>)> = Vec::with_capacity(groups.len());
    // Observed level count per kept grouping, for the row check at the end.
    let mut level_counts: Vec<usize> = Vec::with_capacity(groups.len());
    for (name, nodes) in groups {
        let levels = grouping_level_count(nodes[0], data)?;
        if levels < 2 {
            notes.push(crate::Note::SingleLevelGroupingDropped { grouping: name });
        } else {
            kept.push((name, nodes));
            level_counts.push(levels);
        }
    }
    let Some(((name0, nodes0), rest)) = kept.split_first() else {
        // Every random-effect term was dropped: fit as though the formula had
        // declared none, keeping the warnings raised above.
        return Ok((
            ModelSpec { family, re: None },
            GroupIds::default(),
            Vec::new(),
            notes,
            (Vec::new(), 0),
        ));
    };

    // Only the kept groupings' slopes, so that every column is read.
    let kept_nodes = ast
        .random_effects
        .iter()
        .filter(|re| kept.iter().any(|(g, _)| *g == re_group_name(re)));
    let re_columns = re_only_columns(kept_nodes, data, p, &mut numeric_cols, &mut factor_cols)?;
    let primary_terms = lower_terms(name0, nodes0, &numeric_cols, &factor_cols)?;
    let (primary_re_terms, primary_slopes) = block_of(&primary_terms);
    let primary_layout = grouping_ids(nodes0[0], data)?;
    let primary_ids = primary_layout.ids;
    debug_assert_eq!(primary_ids.len(), n);

    let mut extra_groupings = Vec::new();
    let mut extra_ids = Vec::new();
    let mut re_groups = vec![re_group_info(
        name0,
        &primary_terms,
        primary_layout.slot_labels,
    )];
    // The kernel holds ONE nested slot (`LmmGroupings.nested: Option<_>`), so at
    // most one extra may carry `NestedWithin`. Flat-nesting detection therefore
    // only fires while no nested extra exists yet — later flat candidates fail
    // closed to `Crossed` (statistically the same model, see the route-invariance
    // lever). Explicit `parent:child` syntax is not gated here; a second explicit
    // nested extra trips the engine's `assert_model_shape` instead.
    let mut have_nested = false;
    for (name, nodes) in rest {
        let terms = lower_terms(name, nodes, &numeric_cols, &factor_cols)?;
        let (re_terms, slopes) = block_of(&terms);
        // Explicit `parent:child` syntax → nested. A flat grouping with one
        // scalar-intercept term and no `:` may STILL nest within the primary
        // (the lme4 idiom `(1|batch)+(1|sample)`); `detect_flat_nesting`
        // recognizes that from the id structure and fails closed to Crossed on
        // any parent conflict. The level counts are the kernel's, derived from
        // the ids.
        let explicit_nested = nodes.iter().find(|re| {
            matches!(
                re,
                RandomEffect::Intercept {
                    parent: Some(_),
                    ..
                }
            )
        });
        let (relation, layout) = match (explicit_nested, &nodes[..]) {
            (Some(&re), _) => (GroupingRelation::NestedWithin, grouping_ids(re, data)?),
            (
                None,
                [re @ RandomEffect::Intercept {
                    group,
                    parent: None,
                }],
            ) if !group.contains(':') && !have_nested => {
                match detect_flat_nesting(&primary_ids, &grouping_row_labels(group, data)?) {
                    // The flat idiom writes no parent, and `re_group_info` names
                    // this block for the child alone, so the labels stay bare
                    // child labels: joining the parent in here would make the
                    // SPELLING move with the dataset, since the same formula
                    // routes nested or crossed depending on balance.
                    Some((padded, slot_labels)) => (
                        GroupingRelation::NestedWithin,
                        GroupingLayout {
                            ids: padded,
                            slot_labels,
                        },
                    ),
                    None => (GroupingRelation::Crossed, grouping_ids(re, data)?),
                }
            }
            (None, _) => (GroupingRelation::Crossed, grouping_ids(nodes[0], data)?),
        };
        have_nested |= matches!(relation, GroupingRelation::NestedWithin);
        extra_groupings.push(Grouping {
            terms: re_terms,
            relation,
            slopes,
        });
        extra_ids.push(layout.ids);
        re_groups.push(re_group_info(name, &terms, layout.slot_labels));
    }

    // lme4's `check.nobs.vs.nRE`: each term's random effects, its observed
    // levels times its columns (the intercept, one per numeric slope, one per
    // non-base level of a factor slope), against all rows, not summed across
    // terms. A Gaussian model is noted at `n <= count` (`lmer`), any other
    // family at `n < count` (`glmer`, which allows an observation-level
    // intercept).
    let gaussian = matches!(family, Family::Gaussian);
    for ((name, nodes), &levels) in kept.iter().zip(&level_counts) {
        for &re in nodes {
            let random_effects = levels * term_width(re, &numeric_cols, &factor_cols);
            if n < random_effects || (gaussian && n == random_effects) {
                notes.push(crate::Note::FewRowsForRandomEffects {
                    terms: re_terms_spelling(re),
                    grouping: name.clone(),
                    random_effects: u32::try_from(random_effects).unwrap_or(u32::MAX),
                    rows: u32::try_from(n).unwrap_or(u32::MAX),
                });
            }
        }
    }

    // No envelope cap here: the MAX_* caps are the engine's NoZ↔Sparse ROUTING
    // boundary (`fit::classify_design`), not a model ceiling — over-envelope
    // designs (any family) fit through the sparse-Z solvers, so the frontend
    // passes them through.
    let re = ReStructure {
        terms: primary_re_terms,
        slopes: primary_slopes,
        extra_groupings,
    };

    let ids = GroupIds {
        primary: primary_ids,
        extra: extra_ids,
    };
    Ok((
        ModelSpec {
            family,
            re: Some(re),
        },
        ids,
        re_groups,
        notes,
        re_columns,
    ))
}

/// One grouping's random-effect conditional modes, labelled — the shape every
/// consumer renders, and the ONLY place the kernel's RE block layout is
/// interpreted. Rows are levels, columns are terms; padded nested slots are
/// dropped, so `levels.len()` is the row count and `values.len() = levels.len()
/// · terms.len()`.
pub struct RanefBlock {
    /// Grouping factor name — [`ReGroupInfo::name`].
    pub group: String,
    /// Column names, `"(Intercept)"` first — [`ReGroupInfo::terms`].
    pub terms: Vec<String>,
    /// Row labels, one per retained level, in kernel block order.
    pub levels: Vec<String>,
    /// Row-major `levels.len() × terms.len()` conditional modes.
    pub values: Vec<f64>,
    /// Row-major `levels.len() × q(q+1)/2`, `q = terms.len()`: each level's
    /// conditional covariance, the vech of [`crate::Fit::condvar`]'s block
    /// (column-major lower triangle, `varcorr`'s order). Empty where
    /// `Fit::condvar` is.
    pub condvar: Vec<f64>,
}

/// Label a fit's random-effect conditional modes: [`crate::Fit::ranef`]'s flat
/// blocks zipped against the lowering's slot labels, padded slots dropped, one
/// [`RanefBlock`] per grouping in declaration order.
///
/// This is the whole of the RE block-layout knowledge, written once. Every
/// consumer — the Python and R packages, a Rust caller — goes through here
/// rather than re-deriving the slicing, because the layout is a data-dependent
/// SPEED decision (see `detect_flat_nesting`) that no consumer can infer from
/// the formula.
///
/// A fit with no conditional modes (fixed-only, or non-converged, where
/// `Fit::ranef` is empty) yields an empty vec — not an error.
///
/// # Errors
/// [`Error::RanefShapeMismatch`] when `re_groups` does not describe this fit:
/// a different grouping count, a slot-label count that disagrees with
/// [`crate::Fit::ranef_levels`], or a total that disagrees with `ranef.len()`
/// (or with a non-empty `condvar`'s length).
/// Partial results are never returned — a mislabelled mode reads as a wrong
/// answer, not a cosmetic slip.
pub fn label_ranef(fit: &crate::Fit, re_groups: &[ReGroupInfo]) -> Result<Vec<RanefBlock>, Error> {
    if fit.ranef.is_empty() {
        return Ok(Vec::new());
    }
    let mismatch = |what: &str| Error::RanefShapeMismatch(what.to_string());
    if fit.ranef_levels.len() != re_groups.len() {
        return Err(mismatch(&format!(
            "fit has {} grouping(s), the formula lowered {}",
            fit.ranef_levels.len(),
            re_groups.len()
        )));
    }
    let total: usize = fit
        .ranef_levels
        .iter()
        .zip(re_groups)
        .map(|(&l, g)| l * g.terms.len())
        .sum();
    if total != fit.ranef.len() {
        return Err(mismatch(&format!(
            "ranef holds {} value(s), the lowered blocks span {total}",
            fit.ranef.len()
        )));
    }
    let vech = |q: usize| q * (q + 1) / 2;
    let total_cv: usize = fit
        .ranef_levels
        .iter()
        .zip(re_groups)
        .map(|(&l, g)| l * vech(g.terms.len()))
        .sum();
    if !fit.condvar.is_empty() && total_cv != fit.condvar.len() {
        return Err(mismatch(&format!(
            "condvar holds {} value(s), the lowered blocks span {total_cv}",
            fit.condvar.len()
        )));
    }
    let mut out = Vec::with_capacity(re_groups.len());
    let mut base = 0usize;
    let mut base_cv = 0usize;
    for (g, info) in re_groups.iter().enumerate() {
        let n_levels = fit.ranef_levels[g];
        let q = info.terms.len();
        if info.slot_labels.len() != n_levels {
            return Err(mismatch(&format!(
                "grouping {:?} has {} slot label(s) for {n_levels} level(s)",
                info.name,
                info.slot_labels.len()
            )));
        }
        let mut levels = Vec::new();
        let mut values = Vec::new();
        let mut condvar = Vec::new();
        let v = vech(q);
        for (l, label) in info.slot_labels.iter().enumerate() {
            // A padded nested slot is not a level: no row is assigned to it and
            // its mode is zero by construction, so it is dropped rather than
            // reported (lme4 has no counterpart to it either).
            let Some(label) = label else { continue };
            levels.push(label.clone());
            values.extend_from_slice(&fit.ranef[base + l * q..base + (l + 1) * q]);
            if !fit.condvar.is_empty() {
                condvar.extend_from_slice(&fit.condvar[base_cv + l * v..base_cv + (l + 1) * v]);
            }
        }
        out.push(RanefBlock {
            group: info.name.clone(),
            terms: info.terms.clone(),
            levels,
            values,
            condvar,
        });
        base += n_levels * q;
        base_cv += n_levels * v;
    }
    Ok(out)
}

/// One term's raw spelling (`"x"`, `"log(x)"`, a grouping factor's own name,
/// …) resolved to the real column name lowering would read for it, or `None`
/// when it resolves against neither. Mirrors `numeric_column`'s own
/// tie-break: a literal column by that exact spelling wins over reading it as
/// a transform, so a caller who already has a column named `"log(x)"` gets
/// that column, never `"x"`. `available` stands in for `data` here — this
/// runs before any data column is actually read, in [`referenced_columns`].
fn resolve_available<'a>(raw: &'a str, available: &[String]) -> Option<&'a str> {
    if available.iter().any(|a| a == raw) {
        return Some(raw);
    }
    let (_, col) = parse_transform(raw)?;
    available.iter().any(|a| a == col).then_some(col)
}

fn push_resolved(out: &mut Vec<String>, raw: &str, available: &[String]) {
    if let Some(col) = resolve_available(raw, available) {
        out.push(col.to_string());
    }
}

/// Grouping factor names are never a transform spelling — the RE grammar
/// admits no parens around a group name — so a `parent:child` or `lhs:rhs`
/// composite (`grouping_ids`'s nested and crossed arms) is always exactly its
/// two literal column names joined by `:`; splitting on it recovers them with
/// no tie-break needed. A plain group with no `:` is that single literal
/// name, the split's only piece.
fn push_group(out: &mut Vec<String>, group: &str, available: &[String]) {
    for part in group.split(':') {
        if available.iter().any(|a| a == part) {
            out.push(part.to_string());
        }
    }
}

/// The names in `available` that `materialize` would actually read off `data`
/// to lower `ast` — a caller with its own columns to fetch (the Python port;
/// `Table` itself expects every column already in hand) can filter down to
/// just these before paying for a conversion, instead of touching a column
/// the formula never uses.
///
/// A name that resolves against neither `available` nor a transform of one of
/// its members (an unknown column, or a transform spelling nothing in
/// `available` answers to) is left out — the same name will still surface
/// from [`materialize`] itself, with its own precise error, once the caller
/// tries to lower against data that is really missing it. This function never
/// errors; it is a filter, not a validator.
pub fn referenced_columns(ast: &ParsedFormula, available: &[String]) -> Vec<String> {
    resolved_names(ast, available, true)
}

/// [`referenced_columns`], with the response left out unless `with_response`.
fn resolved_names(ast: &ParsedFormula, available: &[String], with_response: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for_each_formula_name(ast, with_response, |raw, is_group| {
        if is_group {
            push_group(&mut out, raw, available)
        } else {
            push_resolved(&mut out, raw, available)
        }
    });
    out.sort();
    out.dedup();
    out
}

/// Every name the formula spells, in formula order: the response (both
/// `cbind()` columns; skipped unless `with_response`), fixed terms, each
/// random effect's grouping (with `is_group` set) and slope variables, then
/// the `offset()` term. The one walk behind [`referenced_columns`],
/// [`unresolved_columns`] and [`design_columns`].
fn for_each_formula_name(
    ast: &ParsedFormula,
    with_response: bool,
    mut visit: impl FnMut(&str, bool),
) {
    use super::parse::Term;

    if with_response {
        match &ast.cbind {
            Some((s, f)) => {
                visit(s, false);
                visit(f, false);
            }
            None => visit(&ast.dependent, false),
        }
    }
    for term in &ast.terms {
        match term {
            Term::Main { name } => visit(name, false),
            Term::Interaction { vars } => vars.iter().for_each(|v| visit(v, false)),
        }
    }
    for re in &ast.random_effects {
        match re {
            RandomEffect::Intercept { group, .. } => visit(group, true),
            RandomEffect::Slope { group, vars }
            | RandomEffect::NoIntercept { group, vars }
            | RandomEffect::ZeroCorr { group, vars, .. } => {
                visit(group, true);
                vars.iter().for_each(|v| visit(v, false));
            }
        }
    }
    if let Some(expr) = &ast.offset {
        visit(expr, false);
    }
}

/// The other side of [`referenced_columns`]'s filter: names the formula
/// references that resolve against NEITHER `available` nor a transform of one
/// of its members — what [`materialize`] itself would raise
/// [`Error::UnknownColumn`] on, for the first one it reaches. A `log(x)`
/// spelling is reported as a missing `x` (matching `numeric_column`'s own
/// error) rather than as the literal call text.
///
/// A caller with `available` = its data's column names can raise the same
/// "unknown column" error [`materialize`] would, before doing anything else
/// with the formula (the Python port needs this: unlike `materialize`, it
/// validates `weights=`/`offset=` against the data's row count before ever
/// calling into Rust, and a formula's genuinely missing column must be
/// reported ahead of an unrelated length mismatch on those arguments).
pub fn unresolved_columns(ast: &ParsedFormula, available: &[String]) -> Vec<String> {
    unresolved_names(ast, available, true)
}

/// [`unresolved_columns`], with the response left out unless `with_response`.
fn unresolved_names(ast: &ParsedFormula, available: &[String], with_response: bool) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for_each_formula_name(ast, with_response, |raw, is_group| {
        if is_group {
            push_missing_group(&mut out, raw, available)
        } else {
            push_missing(&mut out, raw, available)
        }
    });
    let mut unique: Vec<String> = Vec::with_capacity(out.len());
    for name in out {
        if !unique.contains(&name) {
            unique.push(name);
        }
    }
    unique
}

/// `raw`'s [`resolve_available`] miss, reworded to the name an `UnknownColumn`
/// error would name: `raw` itself when it resolves against neither `available`
/// nor a transform spelling, else the transform's underlying column when THAT
/// is what is missing. `None` when `raw` resolves.
fn missing_name(raw: &str, available: &[String]) -> Option<String> {
    if available.iter().any(|a| a == raw) {
        return None;
    }
    match parse_transform(raw) {
        Some((_, col)) => (!available.iter().any(|a| a == col)).then(|| col.to_string()),
        None => Some(raw.to_string()),
    }
}

fn push_missing(out: &mut Vec<String>, raw: &str, available: &[String]) {
    if let Some(name) = missing_name(raw, available) {
        out.push(name);
    }
}

/// [`push_group`]'s miss side: a grouping's literal parts (never a transform
/// spelling) that are not in `available`.
fn push_missing_group(out: &mut Vec<String>, group: &str, available: &[String]) {
    for part in group.split(':') {
        if !available.iter().any(|a| a == part) {
            out.push(part.to_string());
        }
    }
}

/// Names the formula always reads as plain numbers: the response (both count
/// columns of a `cbind()` response), and the `offset()` term's own column, if
/// any — the R port's `all.vars(formula[[2L]])` plus its own `offset()` walk,
/// which read R's formula AST directly (`r/R/fastglmm.R`). A caller with no
/// such AST of its own (the Python port) asks here instead, to know which
/// columns must stay numeric even when their VALUES look boolean — a bool
/// predictor or grouping column crosses as a `FALSE`/`TRUE` factor, but the
/// response and the offset never do.
///
/// Resolved against `available` the same way [`referenced_columns`] resolves
/// every other name, so a transform spelling (`offset(log(exposure))`) names
/// its underlying column, matching what actually appears in that filtered
/// list.
pub fn numeric_only_columns(ast: &ParsedFormula, available: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    match &ast.cbind {
        Some((s, f)) => {
            push_resolved(&mut out, s, available);
            push_resolved(&mut out, f, available);
        }
        None => push_resolved(&mut out, &ast.dependent, available),
    }
    if let Some(expr) = &ast.offset {
        push_resolved(&mut out, expr, available);
    }
    out
}

/// `(used, missing, numeric_only)` for a design-only lowering
/// ([`fixed_design`]): [`referenced_columns`], [`unresolved_columns`] and
/// [`numeric_only_columns`], with the response left out when the formula is
/// `one_sided`. A one-sided formula's parsed response is the parser's
/// placeholder name (`explained_variable`), not a column the caller named.
#[cfg(feature = "orchestrate")]
pub(crate) fn design_columns(
    ast: &ParsedFormula,
    available: &[String],
    one_sided: bool,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let numeric_only = if one_sided {
        Vec::new()
    } else {
        numeric_only_columns(ast, available)
    };
    (
        resolved_names(ast, available, !one_sided),
        unresolved_names(ast, available, !one_sided),
        numeric_only,
    )
}

/// A random-effect term's left-hand side with its intercept written out:
/// `"1"`, `"1 + x + z"` for slopes, `"0 + x"` without the intercept.
fn re_terms_spelling(re: &RandomEffect) -> String {
    match re {
        RandomEffect::Intercept { .. } => "1".to_string(),
        RandomEffect::Slope { vars, .. }
        | RandomEffect::ZeroCorr {
            vars,
            intercept: true,
            ..
        } => format!("1 + {}", vars.join(" + ")),
        RandomEffect::NoIntercept { vars, .. }
        | RandomEffect::ZeroCorr {
            vars,
            intercept: false,
            ..
        } => format!("0 + {}", vars.join(" + ")),
    }
}

/// The grouping factor a random effect is written against — the name
/// `re_group_info` publishes, and the one a lowering note points at.
fn re_group_name(re: &RandomEffect) -> String {
    match re {
        RandomEffect::Intercept { group, .. }
        | RandomEffect::Slope { group, .. }
        | RandomEffect::NoIntercept { group, .. }
        | RandomEffect::ZeroCorr { group, .. } => group.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse::RandomEffect;
    use crate::ReTerm;

    /// Unbalanced nesting — parents "A","B","C" (lexicographic ids 0,1,2) with
    /// 1, 2, 3 distinct children respectively (child labels reused across
    /// parents, as real nested factors do — e.g. cask "1"/"2"/"3" repeats in
    /// every batch). Hand-computed expected layout: contiguous per-parent
    /// blocks padded to `W = max_p k_p = 3` — `A`'s single child lands at `0`
    /// (slots 1,2 unused padding), `B`'s two children at `[3,4)` (slot 5
    /// padding), `C`'s three children fill `[6,9)` exactly.
    #[test]
    fn grouping_ids_nested_unbalanced_pads_to_max_per_parent() {
        let table = Table {
            columns: vec![
                (
                    "g1".into(),
                    Column::factor_from_labels(&strs(&["A", "B", "B", "C", "C", "C"])),
                ),
                (
                    "g2".into(),
                    Column::factor_from_labels(&strs(&["c1", "c1", "c2", "c1", "c2", "c3"])),
                ),
            ],
            n: 6,
        };
        let re = RandomEffect::Intercept {
            group: "g1:g2".to_string(),
            parent: Some("g1".to_string()),
        };
        let layout = grouping_ids(&re, &table).unwrap();
        assert_eq!(layout.ids, vec![0, 3, 4, 6, 7, 8]);
        // The same hand-computed rectangle, read as labels: parent-first joins at
        // every assigned slot, `None` at the two padded ones.
        assert_eq!(
            layout.slot_labels,
            vec![
                Some("A:c1".into()),
                None,
                None,
                Some("B:c1".into()),
                Some("B:c2".into()),
                None,
                Some("C:c1".into()),
                Some("C:c2".into()),
                Some("C:c3".into()),
            ]
        );
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Labels containing `:`: the pairs `("a:b", "c")` and `("a", "b:c")` join
    /// to the same `"a:b:c"` but are two levels, in the level count of both the
    /// nested and the crossed spelling and in the crossed arm's codes. Codes by
    /// hand: equal joined labels, so the pair decides and `"a" < "a:b"` puts
    /// `("a", "b:c")` first.
    #[test]
    fn colon_in_labels_keeps_distinct_pairs_apart() {
        let table = Table {
            columns: vec![
                (
                    "g1".into(),
                    Column::factor_from_labels(&strs(&["a:b", "a", "a:b", "a"])),
                ),
                (
                    "g2".into(),
                    Column::factor_from_labels(&strs(&["c", "b:c", "c", "b:c"])),
                ),
            ],
            n: 4,
        };
        let crossed = RandomEffect::Intercept {
            group: "g1:g2".to_string(),
            parent: None,
        };
        let nested = RandomEffect::Intercept {
            group: "g1:g2".to_string(),
            parent: Some("g1".to_string()),
        };
        assert_eq!(grouping_level_count(&crossed, &table).unwrap(), 2);
        assert_eq!(grouping_level_count(&nested, &table).unwrap(), 2);
        let layout = grouping_ids(&crossed, &table).unwrap();
        assert_eq!(layout.ids, vec![1, 0, 1, 0]);
        assert_eq!(
            layout.slot_labels,
            vec![Some("a:b:c".into()), Some("a:b:c".into())]
        );
    }

    /// Genuinely crossed with matching cardinality — a child label reused across
    /// parents (Penicillin's `sample` spans every `plate`). `detect_flat_nesting`
    /// must fail closed to `Crossed` (`None`), never routing it through the nested padded layout.
    #[test]
    fn detect_flat_nesting_shared_child_stays_crossed() {
        let primary = vec![0, 1, 0, 1];
        let child = strs(&["x", "x", "y", "y"]);
        assert_eq!(detect_flat_nesting(&primary, &child), None);
    }

    /// Genuine nesting but WILDLY uneven (the shape of an observation-level
    /// factor like grouseticks' `INDEX`): one parent holds most of the children
    /// — 5/1/1 — so the padded rectangle `n_parents · max_children = 3·5 = 15`
    /// exceeds `NESTING_INFLATION_BOUND · 7 levels = 14`. Detection fails
    /// closed to `Crossed` (`None`) even though every child nests cleanly.
    #[test]
    fn detect_flat_nesting_high_inflation_stays_crossed() {
        let primary = vec![0, 0, 0, 0, 0, 1, 2];
        let child = strs(&["a", "b", "c", "d", "e", "f", "g"]);
        assert_eq!(detect_flat_nesting(&primary, &child), None);
    }

    /// The inflation bound at equality: 4/1/1 children, so the padded rectangle
    /// `3·4 = 12` equals `NESTING_INFLATION_BOUND · 6 levels = 12`, and the
    /// layout stays nested. Four slots per parent, children in lexicographic
    /// order within each, the unfilled slots unlabelled (`nested_padded_ids`).
    #[test]
    fn detect_flat_nesting_inflation_at_the_bound_stays_nested() {
        let primary = vec![0, 0, 0, 0, 1, 2];
        let child = strs(&["a", "b", "c", "d", "e", "f"]);
        let (ids, labels) = detect_flat_nesting(&primary, &child).unwrap();
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 8]);
        let slot = |s: &str| Some(s.to_string());
        assert_eq!(
            labels,
            vec![
                slot("a"),
                slot("b"),
                slot("c"),
                slot("d"),
                slot("e"),
                None,
                None,
                None,
                slot("f"),
                None,
                None,
                None,
            ]
        );
    }

    /// Two flat extras that BOTH nest cleanly in the primary: the kernel holds
    /// one nested slot, so only the first detects `NestedWithin`; the second
    /// fails closed to `Crossed` (same statistical model either way). Beyond
    /// the `matches!` on the relation tag, this pins the actual
    /// padded ids and slot labels `lower()` built: `g2`'s two children per
    /// parent land at global ids `[0,1]`/`[2,3]` (bare child labels, hand-traced
    /// in `nested_padded_ids`), and `g3`'s crossed codes are its own
    /// lexicographic factor codes, unrelated to `g1`.
    #[test]
    fn second_flat_nesting_candidate_stays_crossed() {
        let table = Table {
            columns: vec![
                (
                    "y".into(),
                    Column::Numeric(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                ),
                (
                    "g1".into(),
                    Column::factor_from_labels(&strs(&["A", "A", "A", "A", "B", "B", "B", "B"])),
                ),
                (
                    "g2".into(),
                    Column::factor_from_labels(&strs(&[
                        "a1", "a1", "a2", "a2", "b1", "b1", "b2", "b2",
                    ])),
                ),
                (
                    "g3".into(),
                    Column::factor_from_labels(&strs(&[
                        "c1", "c1", "c2", "c2", "d1", "d1", "d2", "d2",
                    ])),
                ),
            ],
            n: 8,
        };
        let lo = super::lower("y ~ (1|g1) + (1|g2) + (1|g3)", &table, Family::Gaussian).unwrap();
        let relations: Vec<_> = lo
            .model
            .re
            .as_ref()
            .unwrap()
            .extra_groupings
            .iter()
            .map(|g| g.relation)
            .collect();
        assert!(matches!(relations[0], GroupingRelation::NestedWithin));
        assert!(matches!(relations[1], GroupingRelation::Crossed));
        assert_eq!(
            lo.ids.extra[0],
            vec![0, 0, 1, 1, 2, 2, 3, 3],
            "g2's two children per g1 parent, padded id per row"
        );
        assert_eq!(
            lo.re_groups[1].slot_labels,
            vec![
                Some("a1".into()),
                Some("a2".into()),
                Some("b1".into()),
                Some("b2".into())
            ],
            "g2's slot labels stay bare child labels, not parent:child"
        );
        assert_eq!(
            lo.ids.extra[1],
            vec![0, 0, 1, 1, 2, 2, 3, 3],
            "g3's own lexicographic factor codes (c1,c2,d1,d2), independent of g1"
        );
        assert_eq!(
            lo.re_groups[2].slot_labels,
            vec![
                Some("c1".into()),
                Some("c2".into()),
                Some("d1".into()),
                Some("d2".into())
            ]
        );
    }

    /// A formula with the intercept removed and no other fixed term (`y ~ 0`,
    /// or `y ~ 0 + (1|g)` with only a random intercept) leaves the design with
    /// zero columns — `materialize` refuses this outright rather than handing
    /// the fit a `p == 0` design.
    #[test]
    fn empty_fixed_design_is_refused() {
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
                (
                    "g".into(),
                    Column::factor_from_labels(&strs(&["a", "a", "b", "b"])),
                ),
            ],
            n: 4,
        };
        for formula in ["y ~ 0", "y ~ 0 + (1|g)"] {
            let err = match super::lower(formula, &table, Family::Gaussian) {
                Ok(_) => panic!("{formula}: expected an empty fixed design to be refused"),
                Err(e) => e,
            };
            assert_eq!(err, Error::EmptyDesign, "{formula}");
            assert!(err.to_string().contains("no columns"), "{formula}: {err}");
        }
    }

    /// A fixed-part factor with a single declared level errors, naming the
    /// column — lme4's own "contrasts can be applied only to factors with 2
    /// or more levels", with the offending column named. `y ~ f:x` reaches the
    /// factor only through an interaction term (no main effect, so `f` is not
    /// lowered first): the `interaction_columns` → `expand_var` →
    /// `factor_dummies` route.
    #[test]
    fn one_level_factor_in_fixed_part_errors() {
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0])),
                ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3])),
                (
                    "f".into(),
                    Column::factor_from_labels(&strs(&["z", "z", "z"])),
                ),
            ],
            n: 3,
        };
        for formula in ["y ~ f + x", "y ~ f:x"] {
            assert_eq!(
                super::lower(formula, &table, Family::Gaussian)
                    .err()
                    .unwrap(),
                Error::SingleLevelFactor {
                    name: "f".to_string()
                },
                "{formula}"
            );
        }
    }

    /// A grouping with a single observed level is dropped with a
    /// `SingleLevelGroupingDropped` note rather than fitted, and the other
    /// random-effect term is promoted to primary and still fits.
    #[test]
    fn single_level_grouping_dropped_with_note_keeps_other_groupings() {
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
                ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3, 0.4])),
                (
                    "onegroup".into(),
                    Column::factor_from_labels(&strs(&["only", "only", "only", "only"])),
                ),
                (
                    "h".into(),
                    Column::factor_from_labels(&strs(&["a", "a", "b", "b"])),
                ),
            ],
            n: 4,
        };
        let lo =
            super::lower("y ~ x + (1 | onegroup) + (1 | h)", &table, Family::Gaussian).unwrap();
        assert!(matches!(
            lo.notes.as_slice(),
            [crate::Note::SingleLevelGroupingDropped { grouping }] if grouping == "onegroup"
        ));
        assert_eq!(lo.re_groups.len(), 1);
        assert_eq!(lo.re_groups[0].name, "h");
    }

    /// Every random-effect term is single-level: the fit routes exactly like a
    /// formula that declared no random effects, keeping both warnings.
    #[test]
    fn every_grouping_single_level_drops_to_no_random_effects() {
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
                ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3, 0.4])),
                (
                    "g1".into(),
                    Column::factor_from_labels(&strs(&["only", "only", "only", "only"])),
                ),
                (
                    "g2".into(),
                    Column::factor_from_labels(&strs(&["one", "one", "one", "one"])),
                ),
            ],
            n: 4,
        };
        let lo = super::lower("y ~ x + (1 | g1) + (1 | g2)", &table, Family::Gaussian).unwrap();
        assert!(lo.model.re.is_none());
        assert!(lo.re_groups.is_empty());
        assert_eq!(lo.notes.len(), 2);
        assert!(lo
            .notes
            .iter()
            .all(|n| matches!(n, crate::Note::SingleLevelGroupingDropped { .. })));
    }

    /// [`unresolved_columns`] names a bare missing column and a transform's
    /// missing underlying column (`log(z)` reports `z`, not the call text),
    /// and stays empty once everything resolves.
    #[test]
    fn unresolved_columns_reports_missing_names_including_transform_targets() {
        let ast = parse("y ~ log(z) + (1 | g) + offset(o)").unwrap();
        let available = strs(&["y", "g"]);
        assert_eq!(
            unresolved_columns(&ast, &available),
            vec!["z".to_string(), "o".to_string()]
        );

        let complete = strs(&["y", "z", "g", "o"]);
        assert!(unresolved_columns(&ast, &complete).is_empty());

        // A name spelled in several terms is reported once, in formula order.
        assert_eq!(
            unresolved_columns(&parse("y ~ z + w + z:w").unwrap(), &strs(&["y"])),
            strs(&["z", "w"])
        );
    }

    /// 36 rows: six groups `g` of six, a three-level factor `f` (`a`, `b`,
    /// `c`) varying within each group, a numeric `x` and a response.
    fn factor_table() -> Table {
        let n = 36;
        let (mut y, mut x, mut f, mut g) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..n {
            x.push(((i * 7) % 11) as f64 / 10.0);
            y.push(1.0 + ((i * 5) % 13) as f64 / 6.0);
            f.push(["a", "b", "c"][i % 3].to_string());
            g.push(format!("g{}", i / 6));
        }
        Table {
            columns: vec![
                ("y".into(), Column::Numeric(y)),
                ("x".into(), Column::Numeric(x)),
                ("f".into(), Column::factor_from_labels(&f)),
                ("g".into(), Column::factor_from_labels(&g)),
            ],
            n,
        }
    }

    /// The kernel's grouping layout of a lowering's model.
    fn groupings(lo: &Lowered) -> crate::lmm::LmmGroupings {
        let re = lo.model.re.clone().expect("a mixed model");
        let cols = |s: &[ColumnId]| s.iter().map(|&c| c as usize).collect::<Vec<_>>();
        let primary = cols(&re.slopes);
        let extras: Vec<Vec<usize>> = re.extra_groupings.iter().map(|g| cols(&g.slopes)).collect();
        let sized = crate::fit::SizedSpec {
            extra_counts: vec![3; re.extra_groupings.len()],
            spec: ModelSpec::mixed(lo.model.family, re),
            n_primary: 6,
        };
        crate::lmm::LmmGroupings::from_cluster_spec_ext(&sized, &primary, &extras)
    }

    /// The labels and the term of each label of the first grouping.
    fn labels(lo: &Lowered) -> (Vec<&str>, Vec<usize>) {
        let g = &lo.re_groups[0];
        (
            g.terms.iter().map(String::as_str).collect(),
            g.term_index.clone(),
        )
    }

    /// `(f || g)` on a factor is lme4's `(1 | g) + (0 + f | g)`; in the crate's
    /// K − 1 coding the factor's term already holds the intercept, so the
    /// intercept term merges into it and the block is one unstructured
    /// `(1 + f | g)`. `(0 + f | g)` lowers to the same block. A numeric slope
    /// beside the factor keeps its own term.
    #[test]
    fn factor_double_bar_is_one_unstructured_term() {
        let t = factor_table();
        let unstructured = lower("y ~ f + (1 + f | g)", &t, Family::Gaussian).unwrap();
        for f in [
            "y ~ f + (f || g)",
            "y ~ f + (0 + f | g)",
            "y ~ f + (1 | g) + (0 + f | g)",
        ] {
            let lo = lower(f, &t, Family::Gaussian).unwrap();
            assert_eq!(lo.model, unstructured.model, "{f}");
            assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![1, 2], "{f}");
            assert_eq!(
                labels(&lo),
                (vec!["(Intercept)", "fb", "fc"], vec![0, 0, 0]),
                "{f}"
            );
        }
        // Columns: (Intercept) 0, x 1, fb 2, fc 3.
        let lo = lower("y ~ x + f + (1 + x + f || g)", &t, Family::Gaussian).unwrap();
        assert_eq!(
            lo.model.re.as_ref().unwrap(),
            &ReStructure::intercept().terms([
                ReTerm::intercept().slopes([2, 3]),
                ReTerm::no_intercept().slopes([1]),
            ])
        );
        assert_eq!(
            labels(&lo),
            (vec!["(Intercept)", "fb", "fc", "x"], vec![0, 0, 0, 1])
        );
        assert_eq!(groupings(&lo).structural_theta(), &[3, 6, 8][..]);
    }

    /// `zerocorr(f | g)` is diagonal over every column of the factor's term:
    /// one block of width 3, every covariance held at zero. On a factor this
    /// is a different model from `(f || g)`.
    #[test]
    fn factor_zerocorr_is_diagonal_over_every_column() {
        let t = factor_table();
        let lo = lower("y ~ f + zerocorr(f | g)", &t, Family::Gaussian).unwrap();
        let g = groupings(&lo);
        assert_eq!(g.primary_q, 3);
        assert_eq!(g.structural_theta(), &[1, 2, 4][..]);
        assert_eq!(
            labels(&lo),
            (vec!["(Intercept)", "fb", "fc"], vec![0, 1, 2])
        );
        assert_eq!(
            lo.model.re.as_ref().unwrap(),
            &ReStructure::intercept().slopes([1, 2]).uncorrelated()
        );
        // Without the intercept the factor's K − 1 coding cannot write a
        // diagonal over the factor's levels.
        let err = match lower("y ~ f + zerocorr(0 + f | g)", &t, Family::Gaussian) {
            Ok(_) => panic!("expected a refusal"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("zerocorr"), "{err}");
    }

    fn sleepstudy() -> Table {
        crate::test_support::csv_table(
            include_str!("../../validation/data/empirical/sleepstudy.csv"),
            &["Subject"],
        )
    }

    /// The formula door's `||` and its split spelling build the matrix door's
    /// `Re::intercept().slopes([1]).uncorrelated()` and fit it bit for bit.
    #[test]
    fn split_terms_match_the_matrix_door() {
        let t = sleepstudy();
        let matrix = ModelSpec::mixed(
            Family::Gaussian,
            ReStructure::intercept().slopes([1]).uncorrelated(),
        );
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let opts = FitOptions::default();
        for f in [
            "Reaction ~ Days + (Days || Subject)",
            "Reaction ~ Days + (1 + Days || Subject)",
            "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
            "Reaction ~ Days + zerocorr(Days | Subject)",
        ] {
            let lo = lower(f, &t, Family::Gaussian).unwrap();
            assert_eq!(lo.model, matrix, "{f}");
            assert_eq!(
                labels(&lo),
                (vec!["(Intercept)", "Days"], vec![0, 1]),
                "{f}"
            );
            let by_formula = crate::fit(&lo.data(), lo.spec(), &opts).unwrap();
            let by_matrix = crate::fit(&lo.data(), &matrix, &opts).unwrap();
            assert!(by_formula.converged(), "{f}");
            assert_eq!(
                by_formula.loglik.to_bits(),
                by_matrix.loglik.to_bits(),
                "{f}"
            );
            assert_eq!(bits(&by_formula.beta), bits(&by_matrix.beta), "{f}");
            assert_eq!(bits(&by_formula.se), bits(&by_matrix.se), "{f}");
            assert_eq!(
                bits(&by_formula.varcorr[0]),
                bits(&by_matrix.varcorr[0]),
                "{f}"
            );
            assert_eq!(bits(&by_formula.ranef), bits(&by_matrix.ranef), "{f}");
        }
    }

    /// Term order does not matter: the intercept term goes first and the
    /// labels follow the block. A five-node AGQ fit of an intercept-free
    /// block over three slopes (kernel width 4) is refused, not a panic.
    #[test]
    fn term_order_and_the_agq_width_cap() {
        let t = sleepstudy();
        let a = lower(
            "Reaction ~ Days + (0 + Days | Subject) + (1 | Subject)",
            &t,
            Family::Gaussian,
        )
        .unwrap();
        let b = lower(
            "Reaction ~ Days + (1 | Subject) + (0 + Days | Subject)",
            &t,
            Family::Gaussian,
        )
        .unwrap();
        assert_eq!(a.model, b.model);
        assert_eq!(labels(&a), (vec!["(Intercept)", "Days"], vec![0, 1]));
        assert_eq!(labels(&a), labels(&b));
        // An intercept-free block labels its visible column alone, and its
        // conditional modes have one value per level.
        let lo = lower(
            "Reaction ~ Days + (0 + Days | Subject)",
            &t,
            Family::Gaussian,
        )
        .unwrap();
        assert_eq!(labels(&lo), (vec!["Days"], vec![0]));
        let fit = crate::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
        let blocks = label_ranef(&fit, &lo.re_groups).unwrap();
        assert_eq!(blocks[0].terms, ["Days"]);
        assert_eq!(blocks[0].values.len(), blocks[0].levels.len());

        let mut t = factor_table();
        let x = |k: usize| -> Vec<f64> { (0..t.n).map(|i| ((i * k) % 7) as f64 / 7.0).collect() };
        let (x1, x2, x3) = (x(3), x(5), x(11));
        let yb: Vec<f64> = (0..t.n).map(|i| ((i * 13) % 5 < 2) as u8 as f64).collect();
        t.columns.push(("x1".into(), Column::Numeric(x1)));
        t.columns.push(("x2".into(), Column::Numeric(x2)));
        t.columns.push(("x3".into(), Column::Numeric(x3)));
        t.columns.push(("yb".into(), Column::Numeric(yb)));
        let opts = FitOptions {
            nagq: 5,
            ..FitOptions::default()
        };
        let binomial = Family::Binomial {
            link: crate::BinomialLink::Logit,
        };
        match super::fit(
            "yb ~ x1 + x2 + x3 + (0 + x1 + x2 + x3 | g)",
            &t,
            binomial,
            &opts,
        ) {
            Err(Error::Fit(crate::FitError::Unsupported(msg))) => {
                assert!(msg.contains("q_p=4"), "{msg}")
            }
            Err(e) => panic!("expected FitError::Unsupported, got {e}"),
            Ok(_) => panic!("expected FitError::Unsupported, got a fit"),
        }
    }

    /// `y ~ 1 + (x | g)`: `x` is in no fixed term, so it is the one
    /// random-effect-only column, read by slope index `p + 0`.
    #[test]
    fn slope_outside_the_design_is_a_re_column() {
        let t = factor_table();
        let lo = lower("y ~ 1 + (x | g)", &t, Family::Gaussian).unwrap();
        assert_eq!((lo.p, lo.re_p), (1, 1));
        let Some(Column::Numeric(x)) = t.get("x") else {
            unreachable!()
        };
        assert_eq!(&lo.re_columns, x);
        assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![1]);
        assert_eq!(lo.re_groups[0].terms, ["(Intercept)", "x"]);
    }

    /// One column set per variable: two groupings that name `x` share it, and
    /// a variable in `X` only inside an interaction is still a column of its
    /// own, as in lme4.
    #[test]
    fn one_re_column_per_variable() {
        let t = factor_table();
        let lo = lower("y ~ 1 + (x | g) + (x | f)", &t, Family::Gaussian).unwrap();
        assert_eq!((lo.p, lo.re_p), (1, 1));
        let re = lo.model.re.as_ref().unwrap();
        assert_eq!(re.slopes, vec![1]);
        assert_eq!(re.extra_groupings[0].slopes, vec![1]);

        let lo = lower("y ~ x:f + (x | g)", &t, Family::Gaussian).unwrap();
        assert_eq!(lo.re_p, 1);
        assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![lo.p as ColumnId]);
    }

    /// A column of the wrong length or a factor code past its levels is an
    /// error, not a panic or a fit on zero-padded rows.
    #[test]
    fn ragged_columns_and_bad_codes_are_refused() {
        let y = || ("y".to_string(), Column::Numeric(vec![1.0, 2.0, 4.0]));
        for xlen in [2, 4] {
            let t = Table {
                columns: vec![y(), ("x".into(), Column::Numeric(vec![1.0; xlen]))],
                n: 3,
            };
            match lower("y ~ x", &t, Family::Gaussian) {
                Err(Error::Fit(FitError::Shape(m))) => assert!(m.contains("\"x\""), "{m}"),
                other => panic!("x of {xlen} rows: {:?}", other.err()),
            }
        }
        let t = Table {
            columns: vec![
                y(),
                (
                    "f".into(),
                    Column::Factor {
                        levels: vec!["a".into(), "b".into()],
                        codes: vec![0, 1, 2],
                    },
                ),
            ],
            n: 3,
        };
        match lower("y ~ f", &t, Family::Gaussian) {
            Err(Error::Fit(FitError::Shape(m))) => assert!(m.contains("code 2"), "{m}"),
            other => panic!("bad code: {:?}", other.err()),
        }
    }

    /// `y ~ 1 + (f | g)`: the factor's K − 1 treatment dummies are the
    /// random-effect-only columns, the same columns and names `y ~ f` puts in
    /// its fixed design, and `(0 + f | g)` takes the intercept as it does with
    /// `f` a fixed term.
    #[test]
    fn factor_slope_outside_the_design() {
        let t = factor_table();
        let lo = lower("y ~ 1 + (f | g)", &t, Family::Gaussian).unwrap();
        assert_eq!((lo.p, lo.re_p), (1, 2));
        let fixed = lower("y ~ f", &t, Family::Gaussian).unwrap();
        let dummies: Vec<f64> = (0..t.n)
            .flat_map(|i| fixed.x[i * 3 + 1..i * 3 + 3].to_vec())
            .collect();
        assert_eq!(lo.re_columns, dummies);
        assert_eq!(fixed.col_names[1..], ["fb", "fc"]);
        assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![1, 2]);
        assert_eq!(
            labels(&lo),
            (vec!["(Intercept)", "fb", "fc"], vec![0, 0, 0])
        );
        let no_intercept = lower("y ~ 1 + (0 + f | g)", &t, Family::Gaussian).unwrap();
        assert_eq!(no_intercept.model, lo.model);
    }

    /// [`numeric_only_columns`] names the response and the offset column, and
    /// both sides of a `cbind()` response.
    #[test]
    fn numeric_only_columns_names_response_and_offset() {
        let ast = parse("y ~ x + (1 | g) + offset(o)").unwrap();
        let available = strs(&["y", "x", "g", "o"]);
        let mut names = numeric_only_columns(&ast, &available);
        names.sort();
        assert_eq!(names, vec!["o".to_string(), "y".to_string()]);

        let cbind_ast = parse("cbind(s, f) ~ x").unwrap();
        let cbind_available = strs(&["s", "f", "x"]);
        let mut cbind_names = numeric_only_columns(&cbind_ast, &cbind_available);
        cbind_names.sort();
        assert_eq!(cbind_names, vec!["f".to_string(), "s".to_string()]);
    }
}
