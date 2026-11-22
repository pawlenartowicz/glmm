use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap, HashSet};

use super::fixed::{expand_var, row_major};
use super::table::{used_levels, LevelsAndCodes};
use super::{kept_nodes, Column, ReGroupInfo, Table};
use crate::fit::nested_padded_ids;
use crate::formula::error::Error;
use crate::formula::parse::{ParsedFormula, RandomEffect};
use crate::{
    ColumnId, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, ReTerm,
};

/// A grouping column's `(levels, codes)`, the levels no row carries dropped by
/// [`used_levels`] (R's `droplevels`, which lme4 applies to a grouping factor;
/// [`materialize`](super::materialize) has already dropped them, a new table has not). The codes
/// ARE the per-row ids, dense in `0..levels.len()` and so each below `n`: a
/// level with no row would cost a random-effect slot and get a fully shrunk `ranef` row. A
/// grouping's level order is arbitrary as far as the fit is concerned (its RE
/// levels are exchangeable), so a factor's own order is kept.
///
/// A numeric column whose values are all whole numbers (an integer column, as
/// `read.csv` and `pd.read_csv` load an id) is taken as a factor of its distinct
/// values in ascending order, labelled as integers — the levels lme4 gets from
/// `factor()` on that column. Any other numeric column is refused.
pub(super) fn grouping_factor<'a>(
    name: &str,
    data: &'a Table,
) -> Result<LevelsAndCodes<'a>, Error> {
    match data.get(name) {
        Some(Column::Factor { levels, codes }) => Ok(used_levels(levels, codes)),
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

/// A two-factor grouping's two columns, each as [`grouping_factor`] reads it:
/// `(1 | a/b)` (nested, `a` the parent) or `(1 | a:b)` (crossed).
struct FactorPair<'a> {
    nested: bool,
    a: LevelsAndCodes<'a>,
    b: LevelsAndCodes<'a>,
}

impl FactorPair<'_> {
    /// Per-row `(a, b)` codes. A column's level labels are distinct, so two
    /// rows share a code pair exactly when they share a label pair.
    fn codes(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.a.1.iter().copied().zip(self.b.1.iter().copied())
    }

    /// The `a:b` label of the code pair `(x, y)`.
    fn label(&self, (x, y): (u32, u32)) -> String {
        format!("{}:{}", self.a.0[x as usize], self.b.0[y as usize])
    }
}

/// `re`'s two-factor grouping, `None` for a grouping of one column. Only a
/// random intercept has one: the parser refuses a slope on `a:b` or `a/b`.
fn factor_pair<'a>(re: &RandomEffect, data: &'a Table) -> Result<Option<FactorPair<'a>>, Error> {
    let (nested, lhs, rhs) = match re {
        RandomEffect::Intercept {
            group,
            parent: Some(parent),
        } => (
            true,
            parent.as_str(),
            group.strip_prefix(&format!("{parent}:")).unwrap_or(group),
        ),
        RandomEffect::Intercept {
            group,
            parent: None,
        } => match group.split_once(':') {
            Some((lhs, rhs)) => (false, lhs, rhs),
            None => return Ok(None),
        },
        _ => return Ok(None),
    };
    Ok(Some(FactorPair {
        nested,
        a: grouping_factor(lhs, data)?,
        b: grouping_factor(rhs, data)?,
    }))
}

/// One grouping's lowered level layout. `slot_labels` is [`ReGroupInfo`]'s
/// field.
pub(super) struct GroupingLayout {
    pub(super) ids: Vec<u32>,
    pub(super) slot_labels: Vec<Option<String>>,
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
pub(super) fn grouping_level_count(re: &RandomEffect, data: &Table) -> Result<usize, Error> {
    match factor_pair(re, data)? {
        Some(pair) => Ok(pair.codes().collect::<HashSet<_>>().len()),
        None => Ok(grouping_factor(&re_group_name(re), data)?.0.len()),
    }
}

/// Per-row level labels of `re`'s grouping, spelled as its
/// [`ReGroupInfo::slot_labels`] spell them, so [`grouping_ids`] labels its
/// slots with them and [`lower_new`](super::new_rows::lower_new) matches a new row to its slot by them.
/// A nested inner factor (`(1 | A/B)`) and a crossed interaction grouping
/// (`(1 | A:B)`) join the two labels as `a:b`, any other grouping is its own
/// label. The nested order is PARENT-FIRST: lme4 spells the same level
/// child-first (`b1:a1`); both label the same thing, and this order matches
/// the one the formula and the grouping's own name already use.
pub(super) fn grouping_labels(re: &RandomEffect, data: &Table) -> Result<Vec<String>, Error> {
    if let Some(pair) = factor_pair(re, data)? {
        return Ok(pair.codes().map(|c| pair.label(c)).collect());
    }
    let (levels, codes) = grouping_factor(&re_group_name(re), data)?;
    Ok(codes.iter().map(|&c| levels[c as usize].clone()).collect())
}

/// Dense per-row ids for one grouping, plus the label of every slot of its RE
/// block. A nested inner factor `A:B` (explicit parent `A`) routes through
/// [`nested_padded_ids`]; a crossed interaction and a plain grouping use a flat
/// global lexicographic code.
pub(super) fn grouping_ids(re: &RandomEffect, data: &Table) -> Result<GroupingLayout, Error> {
    let Some(pair) = factor_pair(re, data)? else {
        // The codes ARE the ids, so slot `l` is level `l`.
        let (levels, codes) = grouping_factor(&re_group_name(re), data)?;
        return Ok(GroupingLayout::all_observed(
            codes.into_owned(),
            levels.into_owned(),
        ));
    };
    if pair.nested {
        // The parent's codes are its ids, as in the plain case — so they match
        // the primary's own ids exactly when the parent is the primary grouping.
        // Within a parent the children are ordered by label.
        let (child_levels, child_codes) = &pair.b;
        let child: Vec<&str> = child_codes
            .iter()
            .map(|&c| child_levels[c as usize].as_str())
            .collect();
        let (ids, slots) = nested_padded_ids(&pair.a.1, &child);
        // Each slot owned by a row takes that row's `parent:child` label;
        // a padded one stays `None`.
        let mut slot_labels = vec![None; slots.len()];
        for (&id, c) in ids.iter().zip(pair.codes()) {
            slot_labels[id as usize].get_or_insert_with(|| pair.label(c));
        }
        return Ok(GroupingLayout { ids, slot_labels });
    }
    // Crossed interaction grouping, e.g. `(1|recipe:replicate)`: every
    // observed `(A,B)` pair is its own level, with no parent grouping created.
    //
    // The composite's own level order is lexicographic on the joined label,
    // NOT inherited from either component's declared order: the pair has no
    // order the caller ever stated, and a grouping's level order is
    // inference-invariant anyway (exchangeable RE levels).
    //
    // A level is the `(A, B)` pair, not the joined label: with a `:` inside a
    // label, `("a:b", "c")` and `("a", "b:c")` both join to `"a:b:c"` but are
    // two levels. The pair's labels break ties after the joined label, so such
    // levels share a display label and stay apart.
    let mut levels: Vec<(String, &str, &str, (u32, u32))> = pair
        .codes()
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|c| {
            let a = pair.a.0[c.0 as usize].as_str();
            let b = pair.b.0[c.1 as usize].as_str();
            (pair.label(c), a, b, c)
        })
        .collect();
    levels.sort_unstable();
    let id_of: HashMap<(u32, u32), u32> = levels
        .iter()
        .enumerate()
        .map(|(id, level)| (level.3, id as u32))
        .collect();
    let ids = pair.codes().map(|c| id_of[&c]).collect();
    let labels = levels.into_iter().map(|level| level.0).collect();
    Ok(GroupingLayout::all_observed(ids, labels))
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

/// The random-effect-only columns of the terms `nodes` ([`Lowered::re_columns`](super::Lowered::re_columns)),
/// row-major `n × s`, and `s`: each slope variable that is in neither map
/// (no fixed main term; a variable in `X` only inside an interaction
/// included, as lme4's `model.matrix` of the term would build it) gets its
/// column, or a factor its K − 1 treatment dummies, the first time a term
/// names it. Each is added to its map at `p + j`, `j` its first column here,
/// so every grouping that names the variable reads the same columns.
pub(super) fn re_only_columns<'a>(
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
///
/// [`term_width`] counts each term's columns by the same rules (the
/// intercept a factor slope gains included); change together.
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
/// block order (mirrors the term order of `Fit::varcorr(g)`). A factor slope's
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
/// Mirrors [`lower_terms`]' rules, `(0 + f | g)` gaining the intercept; change
/// together.
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

/// Whether `re` is the child term of an explicit `parent/child` nesting.
fn is_nested(re: &RandomEffect) -> bool {
    matches!(
        re,
        RandomEffect::Intercept {
            parent: Some(_),
            ..
        }
    )
}

/// The term a grouping's level layout is read from, `nodes` being that
/// grouping's terms in formula order: its nested intercept when it has one,
/// else its first term. `None` for no terms. [`lower_random_effects`] lays a
/// grouping out from it and [`lower_new`](super::lower_new) lays the new rows
/// out the same way.
pub(super) fn layout_node<'a>(
    mut nodes: impl Iterator<Item = &'a RandomEffect> + Clone,
) -> Option<&'a RandomEffect> {
    let first = nodes.clone().next();
    nodes.find(|&re| is_nested(re)).or(first)
}

/// What [`lower_random_effects`] gives [`Lowered`](super::Lowered): the model, the ids,
/// [`Lowered::re_groups`](super::Lowered::re_groups), the notes, and [`Lowered::re_columns`](super::Lowered::re_columns) with its
/// width.
pub(super) struct LoweredRandom {
    pub(super) spec: ModelSpec,
    pub(super) ids: GroupIds,
    pub(super) re_groups: Vec<ReGroupInfo>,
    pub(super) notes: Vec<crate::Note>,
    pub(super) re_columns: Vec<f64>,
    pub(super) re_width: usize,
}

pub(super) fn lower_random_effects(
    ast: &ParsedFormula,
    data: &Table,
    family: Family,
    n: usize,
    p: usize,
    mut numeric_cols: HashMap<String, ColumnId>,
    mut factor_cols: HashMap<String, Vec<(String, ColumnId)>>,
) -> Result<LoweredRandom, Error> {
    if ast.random_effects.is_empty() {
        return Ok(LoweredRandom {
            spec: ModelSpec { family, re: None },
            ids: GroupIds::default(),
            re_groups: Vec::new(),
            notes: Vec::new(),
            re_columns: Vec::new(),
            re_width: 0,
        });
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
        return Ok(LoweredRandom {
            spec: ModelSpec { family, re: None },
            ids: GroupIds::default(),
            re_groups: Vec::new(),
            notes,
            re_columns: Vec::new(),
            re_width: 0,
        });
    };

    // Only the kept groupings' slopes, so that every column is read.
    let (re_columns, re_width) = re_only_columns(
        kept_nodes(ast, kept.iter().map(|(g, _)| g)),
        data,
        p,
        &mut numeric_cols,
        &mut factor_cols,
    )?;
    let primary_terms = lower_terms(name0, nodes0, &numeric_cols, &factor_cols)?;
    let (primary_re_terms, primary_slopes) = block_of(&primary_terms);
    // Every grouping in `groups` has a term.
    let primary_node = layout_node(nodes0.iter().copied()).expect("a grouping has a term");
    let primary_layout = grouping_ids(primary_node, data)?;
    let primary_ids = primary_layout.ids;
    debug_assert_eq!(primary_ids.len(), n);

    let mut extra_groupings = Vec::new();
    let mut extra_ids = Vec::new();
    let mut re_groups = vec![re_group_info(
        name0,
        &primary_terms,
        primary_layout.slot_labels,
    )];
    for (name, nodes) in rest {
        let terms = lower_terms(name, nodes, &numeric_cols, &factor_cols)?;
        let (re_terms, slopes) = block_of(&terms);
        // Only explicit `parent:child` syntax lowers nested. A flat pair that
        // happens to nest lowers crossed; the fit finds the nest in the ids.
        let node = layout_node(nodes.iter().copied()).expect("a grouping has a term");
        let relation = if is_nested(node) {
            GroupingRelation::NestedWithin
        } else {
            GroupingRelation::Crossed
        };
        let layout = grouping_ids(node, data)?;
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
    Ok(LoweredRandom {
        spec: ModelSpec {
            family,
            re: Some(re),
        },
        ids,
        re_groups,
        notes,
        re_columns,
        re_width,
    })
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
pub(super) fn re_group_name(re: &RandomEffect) -> String {
    match re {
        RandomEffect::Intercept { group, .. }
        | RandomEffect::Slope { group, .. }
        | RandomEffect::NoIntercept { group, .. }
        | RandomEffect::ZeroCorr { group, .. } => group.clone(),
    }
}
