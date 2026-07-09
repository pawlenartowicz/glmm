//! Data-dependent lowering — the only stage that touches data. Turns a
//! [`ParsedFormula`] + a [`Table`] into the numeric inputs `glmm::fit_cold`
//! consumes: a row-major design matrix, a structure-only [`ModelSpec`], per-row
//! [`GroupIds`], and a defaulted [`FitOptions`].
//!
//! Conventions (validated against R `model.matrix` / `lme4`):
//! - **Treatment contrasts**, base = first level; factor levels ordered
//!   lexicographically (R `factor()` default). Dummy names are `paste0(var, level)`.
//! - **Interaction columns** are the elementwise product of their components'
//!   expanded columns, with the earliest component's contrasts varying fastest
//!   (R's `model.matrix` order); names are the component names joined by `:`.
//! - **Random effects**: the first grouping in the AST is primary (its width/
//!   sizing live in `ReStructure`); the rest are `extra_groupings`. All count
//!   fields are placeholders — the kernel re-derives real level counts from
//!   `GroupIds`.

use std::collections::{BTreeSet, HashMap};

use crate::error::Error;
use crate::parse::{parse, ParsedFormula, RandomEffect};
use glmm::{
    ColumnId, Family, FitOptions, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure,
    Sizing,
};

/// A single data column. `materialize` derives dense factor codes + R-ordered
/// levels from `Factor` labels itself — the caller supplies raw per-row labels.
pub enum Column {
    /// A numeric column, used verbatim.
    Numeric(Vec<f64>),
    /// A categorical column: per-row category labels.
    Factor {
        /// Per-row category labels (length `n`).
        labels: Vec<String>,
    },
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

/// The lowered fit inputs — a one-shot front door to `glmm::fit_cold`.
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
    /// The R-parity handle the end-to-end/contrast oracles assert against.
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
/// defaulted [`FitOptions`] that `glmm::fit_cold` consumes.
///
/// `formula` is an R-style formula string (e.g. `"y ~ x1 * x2 + (1 + x1 | g)"`).
/// `data` supplies one column per name referenced in `formula`; `family`
/// selects the response distribution recorded in the returned `ModelSpec`.
///
/// # Errors
/// Returns [`Error::Parse`] for a malformed formula (see [`crate::ParseError`]), or
/// one of the data-dependent variants ([`Error::UnknownColumn`],
/// [`Error::ResponseNotNumeric`], [`Error::WrongColumnKind`],
/// [`Error::SlopeVarNotInDesign`]) when `data` doesn't match what the formula
/// requires.
pub fn lower(formula: &str, data: &Table, family: Family) -> Result<Lowered, Error> {
    let ast = parse(formula)?;
    materialize(&ast, data, family)
}

/// The data-dependent half alone (caller already holds a [`ParsedFormula`]).
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
        use crate::parse::Term;
        match term {
            Term::Main { name } => match data.get(name) {
                Some(Column::Numeric(v)) => {
                    numeric_main_col.insert(name.clone(), cols.len() as ColumnId);
                    col_names.push(name.clone());
                    cols.push(v.clone());
                }
                Some(Column::Factor { labels }) => {
                    let mut dummies = Vec::new();
                    for (suffix, col) in factor_dummies(name, labels) {
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

/// Sorted-unique levels (lexicographic — R `factor()` default) and per-row codes.
fn factor_levels(labels: &[String]) -> (Vec<String>, Vec<u32>) {
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

/// Treatment-coded dummy columns for a factor (base = first level, `levels-1`
/// columns). Each is named `paste0(var, level)` — e.g. `period2`.
fn factor_dummies(var: &str, labels: &[String]) -> Vec<(String, Vec<f64>)> {
    let (levels, codes) = factor_levels(labels);
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
        Some(Column::Factor { labels }) => Ok(factor_dummies(name, labels)),
        None => Err(Error::UnknownColumn(name.to_string())),
    }
}

/// Interaction columns: the elementwise product across the vars' expanded column
/// sets, earliest var varying fastest (R `model.matrix` order). Names join the
/// component names with `:`.
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

/// Per-row grouping labels for a factor column (grouping vars must be factors).
fn grouping_labels<'a>(name: &str, data: &'a Table) -> Result<&'a [String], Error> {
    match data.get(name) {
        Some(Column::Factor { labels }) => Ok(labels),
        Some(Column::Numeric(_)) => Err(Error::WrongColumnKind {
            name: name.to_string(),
            expected: "a factor (grouping variable)",
        }),
        None => Err(Error::UnknownColumn(name.to_string())),
    }
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
        .map(|set| set.iter().enumerate().map(|(i, &l)| (l, i as u32)).collect())
        .collect();
    parent_ids
        .iter()
        .zip(labels)
        .map(|(&p, c)| p * n_per_parent as u32 + local_index[p as usize][c.as_str()])
        .collect()
}

/// Detect nesting of a flat extra grouping (the lme4 idiom
/// `(1|parent)+(1|child)`, no explicit `parent:child` syntax) from the observed
/// id structure, returning its [`nested_padded_ids`] layout iff routing it
/// nested is both correct AND a win. Two guards, both fail closed to `Crossed`:
///
/// 1. **Genuine nesting** — every distinct `child_labels` value must fall under a
///    single `primary_ids` (parent). A label spanning two parents is genuinely
///    crossed; routing it nested would corrupt the padded family-block Cholesky.
/// 2. **Balance** — every parent must have the same distinct-child count, so the
///    padded-per-parent layout adds NO empty slots (padded dim == level count).
///    Unbalanced flat nesting — an observation-level factor like grouseticks'
///    `INDEX` (each row its own level, wildly uneven per parent) — genuinely
///    nests but its padded dim (`n_parents · max_children`) far exceeds the level
///    count, inflating the DENSE RE block and running orders of magnitude slower
///    than the crossed path (measured: grouseticks 0.16s→44s). Crossed is
///    correct for it anyway. Explicit `parent:child` syntax still pads unbalanced
///    nesting — there the user asked for nested layout outright.
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
    if children_per_parent.iter().any(|&c| c != w) {
        return None; // unbalanced → padding would inflate the dense block
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
            let (_, parent_ids) = factor_levels(grouping_labels(parent, data)?);
            Ok(nested_padded_ids(&parent_ids, grouping_labels(child, data)?))
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
            let (lhs, rhs) = group
                .split_once(':')
                .expect("group contains ':' per guard above");
            let a = grouping_labels(lhs, data)?;
            let b = grouping_labels(rhs, data)?;
            let joined: Vec<String> = a.iter().zip(b).map(|(x, y)| format!("{x}:{y}")).collect();
            Ok(factor_levels(&joined).1)
        }
        RandomEffect::Intercept { group, .. } | RandomEffect::Slope { group, .. } => {
            Ok(factor_levels(grouping_labels(group, data)?).1)
        }
    }
}

/// Resolve a random effect's slope variables to their design `ColumnId`s. A
/// numeric slope var resolves to its single fixed-effect column; a factor
/// slope var (not in `numeric_main_col`) expands to ALL of its dummy
/// `ColumnId`s, in `factor_dummies` order — so the returned vec can be longer
/// than `vars`. A slope var absent from both maps (never a fixed-effect main
/// term) is `SlopeVarNotInDesign` — this crate does not compute on-demand
/// dummies for a slope-only factor (see design doc point 4: unreached by the
/// current parity corpus, where every factor slope var is also a fixed main).
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
            RandomEffect::Intercept { group, parent: None } if !group.contains(':') => {
                match detect_flat_nesting(&primary_ids, grouping_labels(group, data)?) {
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
    use crate::parse::RandomEffect;

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
                    Column::Factor {
                        labels: vec!["A", "B", "B", "C", "C", "C"]
                            .into_iter()
                            .map(String::from)
                            .collect(),
                    },
                ),
                (
                    "g2".into(),
                    Column::Factor {
                        labels: vec!["c1", "c1", "c2", "c1", "c2", "c3"]
                            .into_iter()
                            .map(String::from)
                            .collect(),
                    },
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
        assert_eq!(detect_flat_nesting(&primary, &child), Some(vec![0, 1, 2, 3]));
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

    /// Genuine nesting but UNBALANCED — parent 0 has 2 children, parent 1 has 1
    /// (the shape of an observation-level factor like grouseticks' `INDEX`,
    /// uneven per parent). Padding to `W=2` would inflate the dense RE block, so
    /// T3 stays `Crossed` (`None`) even though every child nests cleanly.
    #[test]
    fn detect_flat_nesting_unbalanced_stays_crossed() {
        let primary = vec![0, 0, 1];
        let child = strs(&["a", "b", "c"]);
        assert_eq!(detect_flat_nesting(&primary, &child), None);
    }
}
