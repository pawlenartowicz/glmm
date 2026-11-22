use std::collections::{BTreeSet, HashMap};

use super::{Column, FixedTerm, RdxCross, Table};
use crate::formula::error::Error;
use crate::formula::parse::{parse_transform, ParsedFormula, Transform};
use crate::ColumnId;

/// The fixed-effects design of a formula on its data ([`fixed_design`]), and
/// what [`materialize`](super::materialize) reads from it to build the rest of [`Lowered`](super::Lowered).
pub(crate) struct FixedDesign<'a> {
    /// Row-major `data.n × p` design, as [`Lowered::x`](super::Lowered::x).
    pub(crate) x: Vec<f64>,
    /// Design width.
    pub(crate) p: usize,
    /// As [`Lowered::col_names`](super::Lowered::col_names).
    pub(crate) col_names: Vec<String>,
    /// The fixed terms of [`r_terms`].
    pub(super) terms: Vec<Vec<&'a str>>,
    /// As [`Lowered::terms`](super::Lowered::terms).
    pub(super) fixed_terms: Vec<FixedTerm>,
    /// The random-slope resolution maps, described where [`fixed_design`]
    /// fills them.
    pub(super) numeric_main_col: HashMap<String, ColumnId>,
    pub(super) factor_main_cols: HashMap<String, Vec<(String, ColumnId)>>,
}

/// The fixed-effects design of `ast` on `data`: the intercept first (when the
/// formula has one), then each term's expanded columns, in R's term order.
/// The response, the random effects and the offset are not read.
///
/// # Errors
/// [`Error::UnknownColumn`], [`Error::WrongColumnKind`],
/// [`Error::SingleLevelFactor`], [`Error::EmptyDesign`] and
/// [`Error::TransformNotFinite`], as [`materialize`](super::materialize) returns them.
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
pub(super) fn row_major(cols: &[Vec<f64>], n: usize) -> Vec<f64> {
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
pub(super) fn rdx_cross(
    terms: &[Vec<&str>],
    has_intercept: bool,
    data: &Table,
    x: &[f64],
    n: usize,
    p: usize,
) -> Result<RdxCross, Error> {
    let mut names = Vec::new();
    // One `Xᵀ·rdX` column per `rdX` column, added in a term at a time so that
    // only one term's `rdX` columns are alive at once.
    let mut xt_cols: Vec<Vec<f64>> = Vec::new();
    let mut add = |name: String, col: &[f64]| {
        let mut xt_col = vec![0.0; p];
        for (i, &r) in col.iter().enumerate() {
            // x is finite here, so a zero adds only ±0.0.
            if r == 0.0 {
                continue;
            }
            for (acc, &v) in xt_col.iter_mut().zip(&x[i * p..(i + 1) * p]) {
                *acc += v * r;
            }
        }
        names.push(name);
        xt_cols.push(xt_col);
    };
    if has_intercept {
        add("(Intercept)".to_string(), &vec![1.0; n]);
    }
    for vars in terms {
        for (name, col) in interaction_columns(vars, &vec![true; vars.len()], data)? {
            add(name, &col);
        }
    }
    let xt_rdx = faer::Mat::from_fn(p, xt_cols.len(), |j, c| xt_cols[c][j]);
    Ok(RdxCross { names, xt_rdx })
}

/// Sorted-unique levels and per-row codes — the lexicographic ordering
/// [`Column::factor_from_labels`] applies for a caller with no declared order.
pub(super) fn sorted_levels_and_codes<T: Ord + std::hash::Hash + Clone>(
    labels: &[T],
) -> (Vec<T>, Vec<u32>) {
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
pub(super) fn numeric_column(name: &str, data: &Table) -> Result<Vec<f64>, Error> {
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
    use crate::formula::parse::Term;
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

/// R's `TermCode` for variable `v` of a term with variables `vars`: whether
/// the term without `v` is empty (the intercept's place) or lies inside one
/// of the `earlier` terms (code 1; code 2 when that margin is absent). Both
/// the contrast coding ([`full_indicator_flags`]) and lmerTest's containment
/// (`hypotheses::Design::new`) read it, so the two always agree.
pub(in crate::formula) fn margin_present<'a, S: AsRef<str> + 'a>(
    vars: &[S],
    v: &str,
    earlier: impl IntoIterator<Item = &'a [S]>,
) -> bool {
    let rest: Vec<&str> = vars.iter().map(AsRef::as_ref).filter(|&u| u != v).collect();
    rest.is_empty()
        || earlier
            .into_iter()
            .any(|e| rest.iter().all(|&u| e.iter().any(|w| w.as_ref() == u)))
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
pub(super) fn full_indicator_flags(
    terms: &[Vec<&str>],
    has_intercept: bool,
    data: &Table,
) -> Vec<Vec<bool>> {
    let is_factor = |v: &str| matches!(data.get(v), Some(Column::Factor { .. }));
    let mut full: Vec<Vec<bool>> = terms.iter().map(|v| vec![false; v.len()]).collect();
    for (t, vars) in terms.iter().enumerate() {
        for (k, &v) in vars.iter().enumerate() {
            if !is_factor(v) {
                continue;
            }
            full[t][k] = !margin_present(vars, v, terms[..t].iter().map(Vec::as_slice));
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
pub(super) fn expand_var(
    name: &str,
    keep_base: bool,
    data: &Table,
) -> Result<Vec<(String, Vec<f64>)>, Error> {
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
