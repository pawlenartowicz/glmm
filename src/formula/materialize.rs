//! Data-dependent lowering — the only stage that touches data. Turns a
//! [`ParsedFormula`] + a [`Table`] into the numeric inputs `crate::fit_cold`
//! consumes: a row-major design matrix, a structure-only [`ModelSpec`], per-row
//! [`GroupIds`], and a defaulted [`FitOptions`].
//!
//! Conventions (validated against R `model.matrix` / `lme4`):
//! - **Treatment contrasts**, base = first level — the caller's [`Column::Factor`]
//!   level order picks the reference level. [`Column::factor_from_labels`] supplies
//!   the lexicographic order R's `factor()` defaults to when the caller has none.
//!   Dummy names follow `paste0(var, level)`, e.g. `period2`.
//! - **Interaction columns** are the elementwise product of their components'
//!   expanded columns, with the earliest component's contrasts varying fastest
//!   (R's `model.matrix` order); names are the component names joined by `:`.
//! - **Random effects**: the first grouping in the AST is primary (its width/
//!   sizing live in `ReStructure`); the rest are `extra_groupings`. All count
//!   fields are placeholders — the kernel re-derives real level counts from
//!   `GroupIds`.

use std::collections::{BTreeSet, HashMap};

use super::error::Error;
use super::parse::{parse, ParsedFormula, RandomEffect};
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
        /// empty cluster, which contributes nothing to the likelihood but does
        /// cost RE width — callers who care should drop it (R's `droplevels`).
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
/// `varcorr`/`tau2` blocks use them — `"(Intercept)"` first, then any slopes.
pub struct ReGroupInfo {
    /// Grouping factor name (`"Subject"`, or `"A:B"` for a nested inner factor).
    pub name: String,
    /// RE term names for this grouping, e.g. `["(Intercept)", "Days"]`.
    pub terms: Vec<String>,
}

/// The lowered fit inputs — a one-shot front door to `crate::fit_cold`.
pub struct Lowered {
    /// Row-major n×p design (intercept column first).
    pub x: Vec<f64>,
    /// Response column, passed through as-is.
    pub y: Vec<f64>,
    /// Row count.
    pub n: usize,
    /// Design width (emitted column count).
    pub p: usize,
    /// Coefficient name per design column, in column order (`"(Intercept)"` first).
    /// The R-matching handle the end-to-end/contrast oracles assert against.
    pub col_names: Vec<String>,
    /// Structure-only model spec (counts are placeholders; the kernel re-derives
    /// real level counts from `ids`).
    pub model: ModelSpec,
    /// Per-row level ids for every grouping.
    pub ids: GroupIds,
    /// Per-grouping name + RE term names, in `varcorr`/`tau2` block order
    /// (primary first, then extras in declaration order). Empty when the
    /// formula declares no random effects.
    pub re_groups: Vec<ReGroupInfo>,
    /// `target_indices = 0..p`; other knobs defaulted. Caller may override.
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
/// [`Error::SlopeVarNotInDesign`]) when `data` doesn't match what the formula
/// requires.
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
/// [`Error::ResponseNotNumeric`], [`Error::WrongColumnKind`], and
/// [`Error::SlopeVarNotInDesign`].
pub fn materialize(ast: &ParsedFormula, data: &Table, family: Family) -> Result<Lowered, Error> {
    let n = data.n;

    // 1. Response — single numeric column, passed through unchanged.
    let y = match data.get(&ast.dependent) {
        Some(Column::Numeric(v)) => v.clone(),
        Some(Column::Factor { .. }) => {
            return Err(Error::ResponseNotNumeric(ast.dependent.clone()))
        }
        None => return Err(Error::UnknownColumn(ast.dependent.clone())),
    };

    // 2. Fixed design — intercept first, then each term's expanded columns.
    let mut col_names: Vec<String> = vec!["(Intercept)".to_string()];
    let mut cols: Vec<Vec<f64>> = vec![vec![1.0; n]];
    // name → ColumnId for numeric main effects — the resolution map for
    // numeric random-slope variables.
    let mut numeric_main_col: HashMap<String, ColumnId> = HashMap::new();
    // name → its dummy (name, ColumnId) pairs, in `factor_dummies` emission
    // order — the resolution map for factor random-slope variables (a slope on
    // a categorical var expands to all its dummy columns, same treatment
    // contrasts the fixed-effect side already computed).
    let mut factor_main_cols: HashMap<String, Vec<(String, ColumnId)>> = HashMap::new();

    for term in &ast.terms {
        use super::parse::Term;
        match term {
            Term::Main { name } => match data.get(name) {
                Some(Column::Numeric(v)) => {
                    numeric_main_col.insert(name.clone(), cols.len() as ColumnId);
                    col_names.push(name.clone());
                    cols.push(v.clone());
                }
                Some(Column::Factor { levels, codes }) => {
                    let mut dummies = Vec::new();
                    for (suffix, col) in factor_dummies(name, levels, codes) {
                        dummies.push((suffix.clone(), cols.len() as ColumnId));
                        col_names.push(suffix);
                        cols.push(col);
                    }
                    factor_main_cols.insert(name.clone(), dummies);
                }
                None => return Err(Error::UnknownColumn(name.clone())),
            },
            Term::Interaction { vars } => {
                for (name, col) in interaction_columns(vars, data)? {
                    col_names.push(name);
                    cols.push(col);
                }
            }
        }
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
    let (model, ids, re_groups) =
        lower_random_effects(ast, data, family, n, &numeric_main_col, &factor_main_cols)?;

    let opts = FitOptions {
        target_indices: (0..p as u32).collect(),
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
/// columns). Names follow the module-header treatment-contrasts convention.
fn factor_dummies(var: &str, levels: &[String], codes: &[u32]) -> Vec<(String, Vec<f64>)> {
    levels
        .iter()
        .enumerate()
        .skip(1) // drop the base level
        .map(|(li, lvl)| {
            let col: Vec<f64> = codes
                .iter()
                .map(|&c| if c as usize == li { 1.0 } else { 0.0 })
                .collect();
            (format!("{var}{lvl}"), col)
        })
        .collect()
}

/// Expand one variable to its design columns: a numeric → one column named after
/// the variable; a factor → its treatment dummies.
fn expand_var(name: &str, data: &Table) -> Result<Vec<(String, Vec<f64>)>, Error> {
    match data.get(name) {
        Some(Column::Numeric(v)) => Ok(vec![(name.to_string(), v.clone())]),
        Some(Column::Factor { levels, codes }) => Ok(factor_dummies(name, levels, codes)),
        None => Err(Error::UnknownColumn(name.to_string())),
    }
}

/// Interaction columns formed from the module-header conventions: elementwise
/// product across the vars' expanded sets, earliest var varying fastest, names
/// joined by `:`.
fn interaction_columns(vars: &[String], data: &Table) -> Result<Vec<(String, Vec<f64>)>, Error> {
    let mut acc = expand_var(&vars[0], data)?;
    for var in &vars[1..] {
        let next = expand_var(var, data)?;
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

/// A grouping column's `(levels, codes)` (grouping vars must be factors). The
/// codes ARE the dense per-row ids: a grouping's level order is arbitrary as far
/// as the fit is concerned (its RE levels are exchangeable), so the caller's
/// order is taken as-is rather than re-derived.
fn grouping_factor<'a>(name: &str, data: &'a Table) -> Result<(&'a [String], &'a [u32]), Error> {
    match data.get(name) {
        Some(Column::Factor { levels, codes }) => Ok((levels, codes)),
        Some(Column::Numeric(_)) => Err(Error::WrongColumnKind {
            name: name.to_string(),
            expected: "a factor (grouping variable)",
        }),
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
/// the kernel's `NestedWithin` sizing is a fixed-width rectangle: `src/lmm.rs`'s
/// `add_rows_multi` computes a child's RE column as
/// `extra_offsets[e] + id·extra_q[e]`, which only lands in the right parent's
/// block under this padded-block layout — a fresh global lexicographic sort of
/// the joined label does not preserve per-parent contiguity for unbalanced
/// nesting). Slots `k_p..W` in a shorter parent's block are never assigned to
/// any row (padding). Assumes each child label occurs under a SINGLE parent —
/// the caller must have verified nesting (explicit `parent:child` syntax, or
/// [`detect_flat_nesting`] for the flat idiom).
fn nested_padded_ids(parent_ids: &[u32], labels: &[String]) -> Vec<u32> {
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
    parent_ids
        .iter()
        .zip(labels)
        .map(|(&p, c)| p * n_per_parent as u32 + local_index[p as usize][c.as_str()])
        .collect()
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
fn detect_flat_nesting(primary_ids: &[u32], child_labels: &[String]) -> Option<Vec<u32>> {
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

/// Dense per-row ids for one grouping. A nested inner factor `A:B` (explicit
/// parent `A`) routes through [`nested_padded_ids`]; a crossed interaction and a
/// plain grouping use a flat global lexicographic code.
fn grouping_ids(re: &RandomEffect, data: &Table) -> Result<Vec<u32>, Error> {
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
            Ok(nested_padded_ids(
                parent_ids,
                &grouping_row_labels(child, data)?,
            ))
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
            Ok(sorted_levels_and_codes(&joined).1)
        }
        RandomEffect::Intercept { group, .. } | RandomEffect::Slope { group, .. } => {
            Ok(grouping_factor(group, data)?.1.to_vec())
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
fn re_group_info(
    re: &RandomEffect,
    factor_main_cols: &HashMap<String, Vec<(String, ColumnId)>>,
) -> ReGroupInfo {
    match re {
        RandomEffect::Intercept { group, .. } => ReGroupInfo {
            name: group.clone(),
            terms: vec!["(Intercept)".to_string()],
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
            }
        }
    }
}

fn lower_random_effects(
    ast: &ParsedFormula,
    data: &Table,
    family: Family,
    n: usize,
    numeric_main_col: &HashMap<String, ColumnId>,
    factor_main_cols: &HashMap<String, Vec<(String, ColumnId)>>,
) -> Result<(ModelSpec, GroupIds, Vec<ReGroupInfo>), Error> {
    if ast.random_effects.is_empty() {
        return Ok((
            ModelSpec { family, re: None },
            GroupIds::default(),
            Vec::new(),
        ));
    }

    let re0 = &ast.random_effects[0];
    let primary_slopes = slope_cols(re0, numeric_main_col, factor_main_cols)?;
    let primary_ids = grouping_ids(re0, data)?;
    debug_assert_eq!(primary_ids.len(), n);

    let mut extra_groupings = Vec::new();
    let mut extra_ids = Vec::new();
    let mut re_groups = vec![re_group_info(re0, factor_main_cols)];
    // The kernel holds ONE nested slot (`LmmGroupings.nested: Option<_>`), so at
    // most one extra may carry `NestedWithin`. Flat-nesting detection therefore
    // only fires while no nested extra exists yet — later flat candidates fail
    // closed to `Crossed` (statistically the same model, see the route-invariance
    // lever). Explicit `parent:child` syntax is not gated here; a second explicit
    // nested extra trips the engine's `assert_model_shape` instead.
    let mut have_nested = false;
    for re in &ast.random_effects[1..] {
        let slopes = slope_cols(re, numeric_main_col, factor_main_cols)?;
        // Explicit `parent:child` syntax → nested. A flat scalar-intercept
        // grouping with no `:` may STILL nest within the primary (the lme4 idiom
        // `(1|batch)+(1|sample)`, T3); detect that from the id structure and fail
        // closed to Crossed on any parent conflict. Both relation counts are
        // placeholders — the kernel re-derives real level counts from the ids.
        let (relation, ids) = match re {
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
                    Some(padded) => (GroupingRelation::NestedWithin { n_per_parent: 1 }, padded),
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
        extra_groupings.push(Grouping { relation, slopes });
        extra_ids.push(ids);
        re_groups.push(re_group_info(re, factor_main_cols));
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
    ))
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
        let ids = grouping_ids(&re, &table).unwrap();
        assert_eq!(ids, vec![0, 3, 4, 6, 7, 8]);
    }

    fn strs(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    /// Flat lme4 idiom `(1|parent)+(1|child)`, BALANCED — each parent has the
    /// same distinct-child count and every label is unique to one parent (Pastes'
    /// `sample = batch:cask`, 3 casks per batch). T3 detects the nesting and
    /// returns the zero-waste padded layout. Parent ids [0,0,1,1] each with 2
    /// distinct children (W=2, no padding) → ids [0,1,2,3].
    #[test]
    fn detect_flat_nesting_balanced_is_nested() {
        let primary = vec![0, 0, 1, 1];
        let child = strs(&["a", "b", "c", "d"]);
        assert_eq!(
            detect_flat_nesting(&primary, &child),
            Some(vec![0, 1, 2, 3])
        );
    }

    /// Genuinely crossed with matching cardinality — a child label reused across
    /// parents (Penicillin's `sample` spans every `plate`). T3 must fail closed
    /// to `Crossed` (`None`), never routing it through the nested padded layout.
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
        assert_eq!(
            detect_flat_nesting(&primary, &child),
            Some(vec![0, 1, 2, 3, 4, 6, 7, 8])
        );
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
    /// fails closed to `Crossed` (same statistical model either way).
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
    }
}
