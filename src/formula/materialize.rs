//! Data-dependent lowering — the only stage that touches data. Turns a
//! [`ParsedFormula`] + a [`Table`] into the numeric inputs `crate::fit_cold`
//! consumes: a row-major design matrix, a structure-only [`ModelSpec`], per-row
//! [`GroupIds`], and a defaulted [`FitOptions`].
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
//! - **Random effects**: the first grouping in the AST is primary (its width/
//!   sizing live in `ReStructure`); the rest are `extra_groupings`. All count
//!   fields are placeholders — the kernel re-derives real level counts from
//!   `GroupIds`.

use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};

use super::error::Error;
use super::parse::{parse, parse_transform, ParsedFormula, RandomEffect, Transform};
use crate::{
    ColumnId, Family, FitOptions, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure,
    Sizing,
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
        /// with an unused level. An unused level of a *grouping* factor is an
        /// empty cluster, which contributes nothing to the likelihood. It costs
        /// RE width only when an observed level follows it: block width is
        /// `max(code) + 1` (`level_count` in `fit/common.rs`), so a level
        /// declared after the last observed one occupies no slot at all.
        /// Callers who care should drop unused levels (R's `droplevels`).
        levels: Vec<String>,
        /// Per-row index into `levels` (length `n`).
        codes: Vec<u32>,
    },
}

impl Column {
    /// A factor from per-row labels, levels ordered **lexicographically** — R's
    /// `factor()` default for a character vector, and the right default for a
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

/// One grouping factor's random-effect term names, in the order the kernel's
/// `varcorr`/`tau2` blocks use them — `"(Intercept)"` first, then any slopes —
/// plus the label of every slot of its RE block.
pub struct ReGroupInfo {
    /// Grouping factor name (`"Subject"`, or `"A:B"` for a nested inner factor).
    pub name: String,
    /// RE term names for this grouping, e.g. `["(Intercept)", "Days"]`.
    pub terms: Vec<String>,
    /// Level label per slot of this grouping's RE block, in kernel block order,
    /// `None` for a padded nested slot that belongs to no level. Emitted by the
    /// same layer that CHOOSES the layout (`grouping_ids`), because that
    /// choice is data-dependent: a flat `(1|A) + (1|B)` routes nested or crossed
    /// depending on how balanced the data is (see `detect_flat_nesting`), so a
    /// consumer that inferred the layout would be wrong on a dataset nobody
    /// tested. [`label_ranef`] is the only reader.
    pub slot_labels: Vec<Option<String>>,
}

/// The lowered fit inputs — a one-shot front door to `crate::fit_cold`.
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
    /// Structure-only model spec (counts are placeholders; the kernel re-derives
    /// real level counts from `ids`).
    pub model: ModelSpec,
    /// Per-row level ids for every grouping.
    pub ids: GroupIds,
    /// Per-grouping name + RE term names + slot labels, in `varcorr`/`tau2`
    /// block order (primary first, then extras in declaration order). Empty when
    /// the formula declares no random effects.
    pub re_groups: Vec<ReGroupInfo>,
    /// Observations the LOWERING made about the data, which no solver can make
    /// because no solver sees the labels or the un-scaled design —
    /// [`crate::Note::UnusedGroupingLevels`], [`crate::Note::ReDesignScaleSpread`]
    /// and [`crate::Note::SingleLevelGroupingDropped`]. Empty on a clean
    /// lowering. Carried here rather than on `Fit` because it is decided
    /// before any fit runs; the ports fold it into the same warning channel as
    /// `Fit`'s own notes.
    pub notes: Vec<crate::Note>,
    /// `target_indices = 0..p`; `offset` set from an `offset()` formula term
    /// when present, else `None`; other knobs defaulted. Caller may override.
    pub opts: FitOptions,
}

/// Parses `formula` and lowers it against `data` in one call, producing the
/// design matrix, structure-only [`ModelSpec`], per-row [`GroupIds`], and
/// defaulted [`FitOptions`] that `crate::fit_cold` consumes.
///
/// `formula` is an R-style formula string (e.g. `"y ~ x1 * x2 + (1 + x1 | g)"`).
/// `data` supplies one column per name referenced in `formula`; `family`
/// selects the response distribution recorded in the returned `ModelSpec`.
///
/// # Errors
/// Returns [`Error::Parse`] for a malformed formula (see [`super::ParseError`]), or
/// one of the data-dependent variants ([`Error::UnknownColumn`],
/// [`Error::ResponseNotNumeric`], [`Error::WrongColumnKind`],
/// [`Error::SlopeVarNotInDesign`], [`Error::SingleLevelFactor`], [`Error::EmptyDesign`],
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

/// The data-dependent half alone (caller already holds a [`ParsedFormula`]).
///
/// # Errors
/// Returns the same data-dependent error variants as [`lower`]: [`Error::UnknownColumn`],
/// [`Error::ResponseNotNumeric`], [`Error::WrongColumnKind`],
/// [`Error::SlopeVarNotInDesign`], [`Error::SingleLevelFactor`], [`Error::EmptyDesign`],
/// [`Error::TransformNotFinite`], [`Error::CbindNeedsBinomial`],
/// [`Error::ZeroTrials`], and [`Error::NegativeCount`].
pub fn materialize(ast: &ParsedFormula, data: &Table, family: Family) -> Result<Lowered, Error> {
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

    // 2. Fixed design — intercept first (when the formula has one), then each
    // term's expanded columns, in R's term order.
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

    for (vars, keep_base) in terms.iter().zip(&full) {
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
    }

    if cols.is_empty() {
        return Err(Error::EmptyDesign);
    }
    let p = cols.len();
    // Flatten column-major buffers to the row-major layout fit_cold expects
    // (element (i,j) at x[i*p + j]).
    let mut x = vec![0.0; n * p];
    for (j, col) in cols.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            x[i * p + j] = v;
        }
    }

    // 3. Random effects → ReStructure + GroupIds (declaration order; first = primary).
    let xm = faer::MatRef::from_row_major_slice(&x, n, p);
    let (model, ids, re_groups, notes) = lower_random_effects(
        ast,
        data,
        family,
        n,
        xm,
        &numeric_main_col,
        &factor_main_cols,
    )?;

    // `weights`/`offset` are set here ONLY when the formula itself carries
    // them (`cbind()` response, `offset()` term). `orchestrate::run_fit` is
    // the double-specification check site for both — change together.
    let offset = match &ast.offset {
        Some(expr) => Some(numeric_column(expr, data)?),
        None => None,
    };
    let opts = FitOptions {
        target_indices: (0..p as u32).collect(),
        offset,
        weights,
        ..FitOptions::default()
    };

    Ok(Lowered {
        x,
        y,
        n,
        p,
        col_names,
        model,
        ids,
        re_groups,
        notes,
        opts,
    })
}

/// Sorted-unique levels (lexicographic) and per-row codes — the ordering
/// [`Column::factor_from_labels`] applies for a caller with no declared order,
/// and the one [`grouping_ids`]'s crossed-interaction arm applies to its
/// composite `"A:B"` labels.
fn sorted_levels_and_codes(labels: &[String]) -> (Vec<String>, Vec<u32>) {
    let levels: Vec<String> = labels
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let index: HashMap<&str, u32> = levels
        .iter()
        .enumerate()
        .map(|(i, l)| (l.as_str(), i as u32))
        .collect();
    let codes = labels.iter().map(|l| index[l.as_str()]).collect();
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
fn r_terms(ast: &ParsedFormula) -> Vec<Vec<&str>> {
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

/// A grouping column's `(levels, codes)`. The codes ARE the dense per-row ids:
/// a grouping's level order is arbitrary as far as the fit is concerned (its RE
/// levels are exchangeable), so a factor's own order is taken as-is rather than
/// re-derived.
///
/// A numeric column whose values are all whole numbers (an integer column, as
/// `read.csv` and `pd.read_csv` load an id) is taken as a factor of its distinct
/// values in ascending order, labelled as integers — the levels lme4 gets from
/// `factor()` on that column. Any other numeric column is refused.
fn grouping_factor<'a>(name: &str, data: &'a Table) -> Result<LevelsAndCodes<'a>, Error> {
    match data.get(name) {
        Some(Column::Factor { levels, codes }) => Ok((Cow::Borrowed(levels), Cow::Borrowed(codes))),
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
/// field; `unused` names the declared levels that own a slot but no row (see
/// [`crate::Note::UnusedGroupingLevels`]) and is empty for every arm but the
/// plain one, where alone it can happen.
struct GroupingLayout {
    ids: Vec<u32>,
    slot_labels: Vec<Option<String>>,
    unused: Vec<String>,
}

impl GroupingLayout {
    /// A layout whose every slot is an observed level, in the order `labels`
    /// gives them — the crossed-interaction and (unused-free) plain cases.
    fn all_observed(ids: Vec<u32>, labels: Vec<String>) -> Self {
        GroupingLayout {
            ids,
            slot_labels: labels.into_iter().map(Some).collect(),
            unused: Vec::new(),
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
/// its RE block to.
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
        let distinct: BTreeSet<String> =
            a.iter().zip(&b).map(|(x, y)| format!("{x}:{y}")).collect();
        return Ok(distinct.len());
    }
    if let Some((lhs, rhs)) = group.split_once(':') {
        let a = grouping_row_labels(lhs, data)?;
        let b = grouping_row_labels(rhs, data)?;
        let distinct: BTreeSet<String> =
            a.iter().zip(&b).map(|(x, y)| format!("{x}:{y}")).collect();
        return Ok(distinct.len());
    }
    let (_, codes) = grouping_factor(&group, data)?;
    Ok(codes.iter().collect::<BTreeSet<_>>().len())
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
            let (parent_levels, parent_ids) = grouping_factor(parent, data)?;
            let (ids, child_labels) =
                nested_padded_ids(&parent_ids, &grouping_row_labels(child, data)?);
            // Explicit `parent:child` syntax: the user named the parent, so the
            // level is spelled with it. PARENT-FIRST — lme4 spells the same level
            // child-first (`b1:a1`); both label the same thing, and this order
            // matches the one the formula and the grouping's own name already use.
            // `nested_padded_ids` lays out `max(parent_ids)+1` blocks, which is
            // ≤ the parent's declared level count when trailing levels go
            // unobserved — recover its own block width, not the declared one.
            let n_parents = parent_ids
                .iter()
                .copied()
                .max()
                .map(|m| m as usize + 1)
                .unwrap_or(1);
            let w = (child_labels.len() / n_parents).max(1);
            let slot_labels = child_labels
                .into_iter()
                .enumerate()
                .map(|(slot, c)| c.map(|c| format!("{}:{c}", parent_levels[slot / w])))
                .collect();
            Ok(GroupingLayout {
                ids,
                slot_labels,
                unused: Vec::new(),
            })
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
            let (lhs, rhs) = group
                .split_once(':')
                .expect("group contains ':' per guard above");
            let a = grouping_row_labels(lhs, data)?;
            let b = grouping_row_labels(rhs, data)?;
            let joined: Vec<String> = a.iter().zip(&b).map(|(x, y)| format!("{x}:{y}")).collect();
            let (levels, codes) = sorted_levels_and_codes(&joined);
            Ok(GroupingLayout::all_observed(codes, levels))
        }
        RandomEffect::Intercept { group, .. } | RandomEffect::Slope { group, .. } => {
            let (levels, codes) = grouping_factor(group, data)?;
            // The codes ARE the ids, so slot `l` is level `l` — but the block is
            // only `max(code)+1` wide (`fit::common::spec_sized_from_ids`), not
            // `levels.len()`: both ports marshal `levels()` wholesale after row
            // filtering, so a level declared after the last observed one has no
            // slot to label. A level with a slot but no row DOES cost RE width
            // and gets a labelled (fully shrunk) row plus a note.
            let width = codes
                .iter()
                .copied()
                .max()
                .map(|m| m as usize + 1)
                .unwrap_or(0);
            let mut observed = vec![false; width];
            for &c in codes.iter() {
                observed[c as usize] = true;
            }
            Ok(GroupingLayout {
                ids: codes.to_vec(),
                slot_labels: levels[..width].iter().cloned().map(Some).collect(),
                unused: (0..width)
                    .filter(|&l| !observed[l])
                    .map(|l| levels[l].clone())
                    .collect(),
            })
        }
    }
}

/// Resolve a random effect's slope variables to their design `ColumnId`s. A
/// numeric slope var resolves to its single fixed-effect column; a factor
/// slope var (not in `numeric_main_col`) expands to ALL of its dummy
/// `ColumnId`s, in `factor_dummies` order — so the returned vec can be longer
/// than `vars`. A slope var absent from both maps (never a fixed-effect main
/// term) is `SlopeVarNotInDesign` — the crate does not support slope-only
/// factors (factors appearing only in random slopes, not fixed main effects),
/// as that would require on-demand dummy computation outside the main-effects
/// pass. This is unreached by the current validation corpus where every factor
/// slope variable is also a fixed main term.
fn slope_cols(
    re: &RandomEffect,
    numeric_main_col: &HashMap<String, ColumnId>,
    factor_main_cols: &HashMap<String, Vec<(String, ColumnId)>>,
) -> Result<Vec<ColumnId>, Error> {
    match re {
        RandomEffect::Slope { vars, .. } => {
            let mut out = Vec::new();
            for v in vars {
                if let Some(&cid) = numeric_main_col.get(v) {
                    out.push(cid);
                } else if let Some(dummies) = factor_main_cols.get(v) {
                    out.extend(dummies.iter().map(|(_, cid)| *cid));
                } else {
                    return Err(Error::SlopeVarNotInDesign(v.clone()));
                }
            }
            Ok(out)
        }
        RandomEffect::Intercept { .. } => Ok(Vec::new()),
    }
}

/// Build this random effect's `ReGroupInfo` — `"(Intercept)"` first, then any
/// slope term names (mirrors `varcorr`/`tau2` block term order). A factor
/// slope var expands to its dummy names (`format!("{var}{lvl}")`, same naming
/// `factor_dummies` uses on the fixed-effect side), not the bare var name.
/// `slot_labels` comes from [`grouping_ids`], which is the only place the chosen
/// layout and the level labels are both in scope.
fn re_group_info(
    re: &RandomEffect,
    factor_main_cols: &HashMap<String, Vec<(String, ColumnId)>>,
    slot_labels: Vec<Option<String>>,
) -> ReGroupInfo {
    match re {
        RandomEffect::Intercept { group, .. } => ReGroupInfo {
            name: group.clone(),
            terms: vec!["(Intercept)".to_string()],
            slot_labels,
        },
        RandomEffect::Slope { group, vars } => {
            let mut terms = vec!["(Intercept)".to_string()];
            for v in vars {
                if let Some(dummies) = factor_main_cols.get(v) {
                    terms.extend(dummies.iter().map(|(name, _)| name.clone()));
                } else {
                    terms.push(v.clone());
                }
            }
            ReGroupInfo {
                name: group.clone(),
                terms,
                slot_labels,
            }
        }
    }
}

/// The lowering's [`crate::Note::UnusedGroupingLevels`] for one grouping, or
/// `None` when every slot of its block owns a row.
fn unused_levels_note(name: &str, unused: Vec<String>) -> Option<crate::Note> {
    (!unused.is_empty()).then(|| crate::Note::UnusedGroupingLevels {
        grouping: name.to_string(),
        levels: unused,
    })
}

/// [`crate::Note::ReDesignScaleSpread`] fires above this max/min column-RMS
/// ratio — the same 1000 lme4's `lme4:::checkScaleX(tol = 1000)` uses to warn
/// "Some predictor variables are on very different scales: consider
/// rescaling".
const RE_SCALE_SPREAD_WARN: f64 = 1e3;

/// [`crate::Note::ReDesignScaleSpread`] for one grouping: the max/min ratio of
/// [`crate::lmm::rms_column_scale`] over its random-slope design columns, the
/// implicit intercept counted as a constant-1 column (RMS exactly 1.0). `None`
/// when the grouping has no slopes (nothing to compare the intercept against)
/// or the ratio does not clear [`RE_SCALE_SPREAD_WARN`].
///
/// Unweighted (`weights: None`): `FitOptions::weights` is filled in by the
/// ports AFTER `lower()` returns (see `orchestrate::run_fit`), so no per-row
/// weight is in scope at lowering time.
fn scale_spread_note(
    name: &str,
    x: faer::MatRef<'_, f64>,
    slope_cols: &[ColumnId],
) -> Option<crate::Note> {
    if slope_cols.is_empty() {
        return None;
    }
    let mut lo = 1.0f64;
    let mut hi = 1.0f64;
    for &c in slope_cols {
        let s = crate::lmm::rms_column_scale(x, c as usize, None);
        lo = lo.min(s);
        hi = hi.max(s);
    }
    let ratio = hi / lo;
    (ratio > RE_SCALE_SPREAD_WARN).then(|| crate::Note::ReDesignScaleSpread {
        grouping: name.to_string(),
        ratio,
    })
}

fn lower_random_effects(
    ast: &ParsedFormula,
    data: &Table,
    family: Family,
    n: usize,
    xm: faer::MatRef<'_, f64>,
    numeric_main_col: &HashMap<String, ColumnId>,
    factor_main_cols: &HashMap<String, Vec<(String, ColumnId)>>,
) -> Result<(ModelSpec, GroupIds, Vec<ReGroupInfo>, Vec<crate::Note>), Error> {
    if ast.random_effects.is_empty() {
        return Ok((
            ModelSpec { family, re: None },
            GroupIds::default(),
            Vec::new(),
            Vec::new(),
        ));
    }

    // A grouping with only one observed level carries no information about a
    // between-group variance (lme4: "grouping factors must have > 1 sampled
    // level"). Checked up front, independent of the nested/crossed routing
    // decided below (see `grouping_level_count`), and dropped with a warning
    // rather than an error — the rest of the model, including any OTHER
    // random-effect term, still fits. Filtering the term list before anything
    // below picks a primary means a dropped primary correctly promotes the
    // next surviving term, exactly as if the formula had never declared the
    // dropped one.
    let mut notes: Vec<crate::Note> = Vec::new();
    let mut terms: Vec<&RandomEffect> = Vec::with_capacity(ast.random_effects.len());
    for re in &ast.random_effects {
        if grouping_level_count(re, data)? < 2 {
            notes.push(crate::Note::SingleLevelGroupingDropped {
                grouping: re_group_name(re),
            });
        } else {
            terms.push(re);
        }
    }
    let Some((&re0, rest)) = terms.split_first() else {
        // Every random-effect term was dropped: fit as though the formula had
        // declared none, keeping the warnings raised above.
        return Ok((
            ModelSpec { family, re: None },
            GroupIds::default(),
            Vec::new(),
            notes,
        ));
    };

    let primary_slopes = slope_cols(re0, numeric_main_col, factor_main_cols)?;
    let primary_layout = grouping_ids(re0, data)?;
    let primary_ids = primary_layout.ids;
    debug_assert_eq!(primary_ids.len(), n);

    let mut extra_groupings = Vec::new();
    let mut extra_ids = Vec::new();
    notes.extend(unused_levels_note(
        &re_group_name(re0),
        primary_layout.unused,
    ));
    notes.extend(scale_spread_note(&re_group_name(re0), xm, &primary_slopes));
    let mut re_groups = vec![re_group_info(
        re0,
        factor_main_cols,
        primary_layout.slot_labels,
    )];
    // The kernel holds ONE nested slot (`LmmGroupings.nested: Option<_>`), so at
    // most one extra may carry `NestedWithin`. Flat-nesting detection therefore
    // only fires while no nested extra exists yet — later flat candidates fail
    // closed to `Crossed` (statistically the same model, see the route-invariance
    // lever). Explicit `parent:child` syntax is not gated here; a second explicit
    // nested extra trips the engine's `assert_model_shape` instead.
    let mut have_nested = false;
    for &re in rest {
        let slopes = slope_cols(re, numeric_main_col, factor_main_cols)?;
        // Explicit `parent:child` syntax → nested. A flat scalar-intercept
        // grouping with no `:` may STILL nest within the primary (the lme4 idiom
        // `(1|batch)+(1|sample)`); `detect_flat_nesting` recognizes that from the
        // id structure and fails closed to Crossed on any parent conflict. Both relation counts are
        // placeholders — the kernel re-derives real level counts from the ids.
        let (relation, layout) = match re {
            RandomEffect::Intercept {
                parent: Some(_), ..
            } => (
                GroupingRelation::NestedWithin { n_per_parent: 1 },
                grouping_ids(re, data)?,
            ),
            RandomEffect::Intercept {
                group,
                parent: None,
            } if !group.contains(':') && !have_nested => {
                match detect_flat_nesting(&primary_ids, &grouping_row_labels(group, data)?) {
                    // The flat idiom writes no parent, and `re_group_info` names
                    // this block for the child alone, so the labels stay bare
                    // child labels: joining the parent in here would make the
                    // SPELLING move with the dataset, since the same formula
                    // routes nested or crossed depending on balance.
                    Some((padded, slot_labels)) => (
                        GroupingRelation::NestedWithin { n_per_parent: 1 },
                        GroupingLayout {
                            ids: padded,
                            slot_labels,
                            unused: Vec::new(),
                        },
                    ),
                    None => (
                        GroupingRelation::Crossed { n_clusters: 1 },
                        grouping_ids(re, data)?,
                    ),
                }
            }
            _ => (
                GroupingRelation::Crossed { n_clusters: 1 },
                grouping_ids(re, data)?,
            ),
        };
        have_nested |= matches!(relation, GroupingRelation::NestedWithin { .. });
        notes.extend(unused_levels_note(&re_group_name(re), layout.unused));
        notes.extend(scale_spread_note(&re_group_name(re), xm, &slopes));
        extra_groupings.push(Grouping { relation, slopes });
        extra_ids.push(layout.ids);
        re_groups.push(re_group_info(re, factor_main_cols, layout.slot_labels));
    }

    // No envelope cap here: the MAX_* caps are the engine's NoZ↔Sparse ROUTING
    // boundary (`fit::classify_design`), not a model ceiling — over-envelope
    // designs (any family) fit through the sparse-Z solvers, so the frontend
    // passes them through.
    let re = ReStructure {
        sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — kernel derives from ids
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
/// [`crate::Fit::ranef_levels`], or a total that disagrees with `ranef.len()`.
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
    let mut out = Vec::with_capacity(re_groups.len());
    let mut base = 0usize;
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
        for (l, label) in info.slot_labels.iter().enumerate() {
            // A padded nested slot is not a level: no row is assigned to it and
            // its mode is zero by construction, so it is dropped rather than
            // reported (lme4 has no counterpart to it either).
            let Some(label) = label else { continue };
            levels.push(label.clone());
            values.extend_from_slice(&fit.ranef[base + l * q..base + (l + 1) * q]);
        }
        out.push(RanefBlock {
            group: info.name.clone(),
            terms: info.terms.clone(),
            levels,
            values,
        });
        base += n_levels * q;
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
    let mut out: Vec<String> = Vec::new();
    for_each_formula_name(ast, |raw, is_group| {
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
/// `cbind()` columns), fixed terms, each random effect's grouping (with
/// `is_group` set) and slope variables, then the `offset()` term. The one walk
/// behind [`referenced_columns`] and [`unresolved_columns`].
fn for_each_formula_name(ast: &ParsedFormula, mut visit: impl FnMut(&str, bool)) {
    use super::parse::Term;

    match &ast.cbind {
        Some((s, f)) => {
            visit(s, false);
            visit(f, false);
        }
        None => visit(&ast.dependent, false),
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
            RandomEffect::Slope { group, vars } => {
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
    let mut out: Vec<String> = Vec::new();
    for_each_formula_name(ast, |raw, is_group| {
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

/// The grouping factor a random effect is written against — the name
/// `re_group_info` publishes, and the one a lowering note points at.
fn re_group_name(re: &RandomEffect) -> String {
    match re {
        RandomEffect::Intercept { group, .. } | RandomEffect::Slope { group, .. } => group.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::parse::RandomEffect;

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

    /// Flat lme4 idiom `(1|parent)+(1|child)`, BALANCED — each parent has the
    /// same distinct-child count and every label is unique to one parent (Pastes'
    /// `sample = batch:cask`, 3 casks per batch). `detect_flat_nesting` detects
    /// the nesting and returns the zero-waste padded layout. Parent ids [0,0,1,1] each with 2
    /// distinct children (W=2, no padding) → ids [0,1,2,3].
    #[test]
    fn detect_flat_nesting_balanced_is_nested() {
        let primary = vec![0, 0, 1, 1];
        let child = strs(&["a", "b", "c", "d"]);
        let (ids, labels) = detect_flat_nesting(&primary, &child).expect("nesting detected");
        assert_eq!(ids, vec![0, 1, 2, 3]);
        // Bare child labels, not `parent:child` — see the flat arm in
        // `lower_random_effects` for why this route must not join the parent in.
        assert_eq!(
            labels,
            vec![
                Some("a".into()),
                Some("b".into()),
                Some("c".into()),
                Some("d".into())
            ]
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

    /// Genuine nesting, NEAR-balanced — parents 0/1/2 with 3/2/3 distinct
    /// children (the grid generator's `sample(1:3)` shape: children-per-parent
    /// ∈ {1,2,3}). Padded rectangle `3·3 = 9` vs `8` levels — inflation 1.125
    /// ≤ NESTING_INFLATION_BOUND, so detection returns the padded layout
    /// (parent 1's slot `[3+2, 6)` stays unassigned padding).
    #[test]
    fn detect_flat_nesting_near_balanced_is_nested() {
        let primary = vec![0, 0, 0, 1, 1, 2, 2, 2];
        let child = strs(&["a", "b", "c", "d", "e", "f", "g", "h"]);
        let (ids, labels) = detect_flat_nesting(&primary, &child).expect("nesting detected");
        assert_eq!(ids, vec![0, 1, 2, 3, 4, 6, 7, 8]);
        assert_eq!(labels[5], None, "parent 1's third slot is padding");
        assert_eq!(labels[4], Some("e".into()));
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

    /// Two flat extras that BOTH nest cleanly in the primary: the kernel holds
    /// one nested slot, so only the first detects `NestedWithin`; the second
    /// fails closed to `Crossed` (same statistical model either way). Beyond
    /// the `matches!` on the relation tag — whose `n_per_parent`/`n_clusters`
    /// are placeholders the kernel re-derives from `ids` — this pins the actual
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
            .map(|g| g.relation.clone())
            .collect();
        assert!(matches!(
            relations[0],
            GroupingRelation::NestedWithin { .. }
        ));
        assert!(matches!(relations[1], GroupingRelation::Crossed { .. }));
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

    /// A formula with the intercept removed and no other fixed term (`y ~ 0`)
    /// leaves the design with zero columns — `materialize` refuses this
    /// outright rather than handing `fit_cold` a `p == 0` design.
    #[test]
    fn empty_fixed_design_is_refused() {
        let table = Table {
            columns: vec![("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0]))],
            n: 3,
        };
        assert!(matches!(
            super::lower("y ~ 0", &table, Family::Gaussian),
            Err(Error::EmptyDesign)
        ));
    }

    /// Fixture for the `label_ranef` mismatch tests below: 3 groups of 4 rows
    /// each, one numeric predictor, random intercept on `g`.
    fn ranef_mismatch_fixture() -> (crate::Fit, Vec<ReGroupInfo>) {
        let g_labels = strs(&[
            "g1", "g1", "g1", "g1", "g2", "g2", "g2", "g2", "g3", "g3", "g3", "g3",
        ]);
        let group_effect = [0.6, -0.4, 0.9];
        let mut seed = 7u64;
        let x: Vec<f64> = (0..12)
            .map(|_| crate::fit::common_tests::lcg(&mut seed))
            .collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                1.0 + 0.5 * v
                    + group_effect[i / 4]
                    + 0.05 * crate::fit::common_tests::lcg(&mut seed)
            })
            .collect();
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(y)),
                ("x".into(), Column::Numeric(x)),
                ("g".into(), Column::factor_from_labels(&g_labels)),
            ],
            n: 12,
        };
        let lo = super::lower("y ~ x + (1 | g)", &table, Family::Gaussian).unwrap();
        let fit = crate::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        (fit, lo.re_groups)
    }

    /// The total-length branch: grouping count agrees with the fit, but an
    /// inflated `terms` count for that grouping makes the block width
    /// `label_ranef` computes disagree with `fit.ranef.len()`.
    #[test]
    fn label_ranef_refuses_a_total_length_mismatch() {
        let (fit, mut re_groups) = ranef_mismatch_fixture();
        assert!(
            !fit.ranef.is_empty(),
            "fixture must produce conditional modes"
        );
        re_groups[0].terms.push("(Intercept)".to_string());
        assert!(matches!(
            label_ranef(&fit, &re_groups),
            Err(Error::RanefShapeMismatch(_))
        ));
    }

    /// The slot-label-count branch: grouping and term counts both agree with
    /// the fit (so the total-length check passes), but the slot-label vector
    /// for that grouping is a level short of `fit.ranef_levels`.
    #[test]
    fn label_ranef_refuses_a_slot_label_count_mismatch() {
        let (fit, mut re_groups) = ranef_mismatch_fixture();
        assert!(
            !fit.ranef.is_empty(),
            "fixture must produce conditional modes"
        );
        re_groups[0].slot_labels.pop();
        assert!(matches!(
            label_ranef(&fit, &re_groups),
            Err(Error::RanefShapeMismatch(_))
        ));
    }

    /// A fixed-part factor with a single declared level errors, naming the
    /// column — lme4's own "contrasts can be applied only to factors with 2
    /// or more levels", with the offending column named.
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
        assert_eq!(
            super::lower("y ~ f + x", &table, Family::Gaussian)
                .err()
                .unwrap(),
            Error::SingleLevelFactor {
                name: "f".to_string()
            }
        );
    }

    /// The same one-level factor inside an interaction term also errors —
    /// `factor_dummies` is the one call site the main-effect and interaction
    /// paths both go through.
    #[test]
    fn one_level_factor_inside_interaction_errors() {
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
        assert_eq!(
            super::lower("y ~ f * x", &table, Family::Gaussian)
                .err()
                .unwrap(),
            Error::SingleLevelFactor {
                name: "f".to_string()
            }
        );
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
