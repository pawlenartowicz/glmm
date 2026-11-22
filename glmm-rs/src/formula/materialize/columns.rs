use crate::formula::parse::{parse_transform, ParsedFormula, RandomEffect, DEFAULT_RESPONSE};

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
/// from [`materialize`](super::materialize) itself, with its own precise error, once the caller
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
    use crate::formula::parse::Term;

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
/// of its members — what [`materialize`](super::materialize) itself would raise
/// [`Error::UnknownColumn`](crate::formula::Error::UnknownColumn) on, for the first one it reaches. A `log(x)`
/// spelling is reported as a missing `x` (matching `numeric_column`'s own
/// error) rather than as the literal call text.
///
/// A caller with `available` = its data's column names can raise the same
/// "unknown column" error [`materialize`](super::materialize) would, before doing anything else
/// with the formula (the Python port needs this: unlike `materialize`, it
/// validates `weights=`/`offset=` against the data's row count before ever
/// calling into Rust, and a formula's genuinely missing column must be
/// reported ahead of an unrelated length mismatch on those arguments).
pub fn unresolved_columns(ast: &ParsedFormula, available: &[String]) -> Vec<String> {
    unresolved_names(ast, available, true)
}

/// [`unresolved_columns`], with the response left out unless `with_response`.
/// A formula with no response is left to [`materialize`](super::materialize)'s syntax error, so
/// the parser's placeholder name is never reported as a missing column.
fn unresolved_names(ast: &ParsedFormula, available: &[String], with_response: bool) -> Vec<String> {
    let no_response = ast.cbind.is_none() && ast.dependent == DEFAULT_RESPONSE;
    let mut out: Vec<String> = Vec::new();
    for_each_formula_name(ast, with_response && !no_response, |raw, is_group| {
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
/// which read R's formula AST directly (`glmm-r/R/fastglmm.R`). A caller with no
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
/// ([`fixed_design`](super::fixed_design)): [`referenced_columns`], [`unresolved_columns`] and
/// [`numeric_only_columns`], with the response left out when the formula is
/// `one_sided`. A one-sided formula's parsed response is the parser's
/// placeholder name (`explained_variable`), not a column the caller named.
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
