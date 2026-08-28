//! Data-free formula parser. The AST shape and the [`ParseError`] Display
//! strings (`error.rs`) are a fixed contract: the parse test suite matches
//! error substrings against them, so both must stay stable across changes to
//! this file.
//!
//! `*` is desugared into main effects + all interactions and `A/B` into a
//! nesting relation *while parsing* — there is no separate normalize pass.

use super::error::ParseError;
use regex::Regex;
use std::sync::LazyLock;

// Random-effect regex tower — applied in this order so `(0|g)` suppression is
// caught before `(1|g)`, slopes before plain intercepts, and the `A:B`
// crossed-interaction grouping (`RE_INTERACTION`) before the plain intercept
// (`RE_INT`) — RE_INT's group-name class excludes `:` so it can never match an
// interaction grouping, but the interaction step still runs first for clarity.
static RE_SUPPRESS: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\((?:0|-1)(?:\+[^|]*)?\|[^)]*\)").unwrap());
static RE_NESTED: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*)/([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});
// The slope-list class admits one paren level so `(1 + log(x) | g)` matches,
// and refuses to cross a `)` so a fixed transform term followed by an RE term
// (`log(x)+(1|g)`) cannot be swallowed as `(x)+(1|g)`.
static RE_SLOPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(1\+((?:[^|()]|\([^|()]*\))+?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});
static RE_ISLOPE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(([_A-Za-z](?:[^|()]|\([^|()]*\))*?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});
static RE_INT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap());
static RE_INTERACTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*):([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});

// `cbind(successes, failures)` LHS — a data-free syntax check only; the
// two names are resolved against `data` in `materialize`.
static CBIND_LHS: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^cbind\(([_A-Za-z][_A-Za-z0-9]*),([_A-Za-z][_A-Za-z0-9]*)\)$").unwrap()
});

// `offset(...)` term, anchored to a term boundary (start of the RHS or right
// after a `+`) so `foffset(e)` (a different identifier that merely ends in
// "offset") and `x*offset(e)` / `a:offset(e)` (offset used inside a `*`/`:`
// combination, which the grammar does not support) are rejected as syntax
// errors instead of silently matching mid-token. One paren level admitted
// inside the capture (mirrors `RE_SLOPE`'s class) so a whitelisted transform
// spelling like `log(exposure)` matches; the extracted text is then
// re-checked by `is_identifier` / `parse_transform` rather than trusted, so
// `offset(a+b)` or `offset(poly(a,2))` still fail as syntax errors.
static OFFSET_TERM: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(^|\+)offset\(((?:[^()]|\([^()]*\))*)\)").unwrap());

// Fixed-intercept removal. Whitespace is already stripped, so `x - 1`
// arrives as `x-1`: the `-1` is matched at a term boundary (end of string
// or before `+`) so `x-10` is untouched and still reaches
// `find_term_removal`. `0` is a whole term (`0+x`, `x+0`). The `regex` crate
// has no look-around, so the boundary is a normal capture group that gets
// put back in the replacement rather than merely asserted.
static NO_INTERCEPT_MINUS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"-1(\+|$)").unwrap());
static NO_INTERCEPT_ZERO: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:^|\+)0(\+|$)").unwrap());

/// The parsed formula AST. A frozen contract — its shape must stay stable
/// since the parse test suite depends on it. The suite proves a superset of
/// MCPower's grammar, not a mirror: fields like `has_intercept` widen the
/// grammar past what MCPower's own parser accepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedFormula {
    /// Dependent-variable name (`"y"`; `"explained_variable"` when the LHS is empty).
    pub dependent: String,
    /// Predictor names in formula order. Duplicates are dropped. Grouping vars are
    /// NOT predictors.
    pub predictors: Vec<String>,
    /// Effect terms in formula order. Main effects appear as `Term::Main`;
    /// interactions (from `:` or the `*` expansion) as `Term::Interaction`.
    pub terms: Vec<Term>,
    /// Random effects in formula order, with one exception: a nested `(1|A/B)`
    /// pair always sorts first (parent immediately before child), because the
    /// kernel interprets `NestedWithin` relative to the primary grouping.
    /// Empty when the formula declares none. The first entry's grouping is the
    /// primary grouping.
    pub random_effects: Vec<RandomEffect>,
    /// `false` when the fixed design carries no intercept (`- 1` / `0 +`).
    /// Random-effect intercepts are unaffected — they are always present.
    pub has_intercept: bool,
    /// The expression inside an `offset(...)` term — a bare column name or a
    /// whitelisted transform spelling. Never a predictor and never a term:
    /// an offset adds no design column.
    pub offset: Option<String>,
    /// `cbind(successes, failures)` on the LHS: the two column names. `dependent`
    /// keeps the full spelling. Lowered by `materialize` onto the proportion +
    /// trial-count form the kernel already fits.
    pub cbind: Option<(String, String)>,
}

/// A fixed-effect term. `Interaction` holds its component variable names in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// A single main effect, e.g. `Main { name: "x1" }`.
    Main {
        /// The predictor's name.
        name: String,
    },
    /// An interaction, e.g. `Interaction { vars: ["x1", "x2"] }` (`x1:x2`).
    Interaction {
        /// Component variable names in interaction order.
        vars: Vec<String>,
    },
}

/// A random-effect term. `Intercept.parent` is set when the grouping factor is
/// nested — `(1|A/B)` yields `Intercept{group:"A",parent:None}` then
/// `Intercept{group:"A:B",parent:Some("A")}`. A crossed (non-nested) grouping
/// keyed by two factors' combination — `(1|A:B)` — yields
/// `Intercept{group:"A:B",parent:None}`: same `"A:B"` group-name shape as the
/// nested inner, but `parent: None` marks it as crossed, not nested.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RandomEffect {
    /// Random intercept for `group`.
    Intercept {
        /// Grouping factor name (`"A:B"` for a nested inner factor).
        group: String,
        /// Parent grouping name when nested, else `None`.
        parent: Option<String>,
    },
    /// Random slope(s) `vars` for grouping factor `group` (intercept implicit).
    Slope {
        /// Grouping factor name.
        group: String,
        /// Slope variable names (the implicit `1` intercept is dropped from this list).
        vars: Vec<String>,
    },
}

/// Parse a formula string into its data-free AST.
///
/// # Errors
/// Returns a [`ParseError`] for malformed formulas:
/// - [`ParseError::EmptyFormula`] if the formula is empty or has no RHS.
/// - [`ParseError::Syntax`] if a token is not a valid identifier (e.g. contains
///   parentheses or other invalid characters).
/// - [`ParseError::TermRemovalUnsupported`] if the formula contains a `-` term
///   removal (e.g. `y ~ x - z`).
/// - [`ParseError::DuplicateGroupingVar`] if the same grouping factor appears
///   in multiple random-effect terms.
/// - [`ParseError::EmptySlopeTerm`] if a random-slope term has no slope
///   variables (e.g. `(1+ |g)` with nothing between `+` and `|`).
/// - [`ParseError::RandomInterceptSuppressionUnsupported`] if a random-effect
///   term suppresses the intercept (e.g. `(0+x|g)` or `(-1+x|g)`).
pub fn parse(input: &str) -> Result<ParsedFormula, ParseError> {
    let cleaned: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.is_empty() {
        return Err(ParseError::EmptyFormula);
    }

    let (mut dep, rhs) = split_at_separator(&cleaned);
    // `cbind(s,f)` LHS check first: an empty LHS never matches it, so the
    // `explained_variable` fallback below still applies. The canonical parse
    // suite's LHS forms are `y ~`, `y =`, and none — never a non-identifier
    // that already parses — so the third branch below is safe.
    let cbind = if let Some(m) = CBIND_LHS.captures(&dep) {
        Some((
            m.get(1).unwrap().as_str().to_string(),
            m.get(2).unwrap().as_str().to_string(),
        ))
    } else if dep.starts_with("cbind(") {
        return Err(ParseError::Syntax {
            pos: 0,
            msg: format!("cbind() takes exactly two column names, got '{dep}'"),
        });
    } else if !dep.is_empty() && !is_identifier(&dep) {
        return Err(ParseError::Syntax {
            pos: 0,
            msg: format!("expected a response column name, got '{dep}'"),
        });
    } else {
        None
    };
    if dep.is_empty() {
        // Empty LHS (e.g. `~X` or `=X`) — fall back to the same default name we
        // use when there is no separator at all.
        dep = "explained_variable".to_string();
    }
    if rhs.is_empty() {
        return Err(ParseError::EmptyFormula);
    }

    // Extract random-effect terms first so the remaining RHS is a plain
    // fixed-effects string.
    let (random_effects, rhs_stripped) = extract_random_effects(&rhs)?;

    // `offset()` extraction runs before the `- 1` stripping below so those
    // regexes never see `offset(...)`'s contents, and after RE extraction so
    // an RE term cannot contain `offset(` (the paren-aware slope classes
    // stop short at a `)`, so `offset(x)+(1|g)` is never swallowed as an RE
    // term either).
    let mut offset = None;
    let mut rhs_stripped = rhs_stripped;
    let offsets: Vec<String> = OFFSET_TERM
        .captures_iter(&rhs_stripped)
        .map(|m| m.get(2).unwrap().as_str().to_string())
        .collect();
    if offsets.len() > 1 {
        return Err(ParseError::Syntax {
            pos: 0,
            msg: "at most one offset() term is allowed".into(),
        });
    }
    if let Some(expr) = offsets.into_iter().next() {
        if !(is_identifier(&expr) || parse_transform(&expr).is_some()) {
            return Err(ParseError::Syntax {
                pos: 0,
                msg: format!(
                    "offset() takes a column name or a whitelisted transform of one, got '{expr}'"
                ),
            });
        }
        offset = Some(expr);
        rhs_stripped = clean_residual_plusses(&OFFSET_TERM.replace(&rhs_stripped, "+"));
    }

    let mut has_intercept = true;
    if NO_INTERCEPT_MINUS.is_match(&rhs_stripped) || NO_INTERCEPT_ZERO.is_match(&rhs_stripped) {
        has_intercept = false;
        // `$1` restores the trailing `+`/end-of-string the match consumed (no
        // look-around available); `clean_residual_plusses` mops up the
        // doubled `+` the zero-branch's literal `"+"` replacement can leave.
        rhs_stripped = NO_INTERCEPT_MINUS
            .replace_all(&rhs_stripped, "$1")
            .into_owned();
        rhs_stripped = NO_INTERCEPT_ZERO
            .replace_all(&rhs_stripped, "+$1")
            .into_owned();
        rhs_stripped = clean_residual_plusses(&rhs_stripped);
    }

    // Reject term removal: a '-' that isn't inside parens and isn't a digit sign.
    if find_term_removal(&rhs_stripped).is_some() {
        return Err(ParseError::TermRemovalUnsupported);
    }

    let (predictors, terms) = if rhs_stripped.is_empty() {
        // RE-only RHS is valid (e.g. `y ~ (1|g)`).
        (Vec::new(), Vec::new())
    } else {
        parse_rhs(&rhs_stripped)?
    };

    Ok(ParsedFormula {
        dependent: dep,
        predictors,
        terms,
        random_effects,
        has_intercept,
        offset,
        cbind,
    })
}

/// Repeatedly matches `regex` against `work`, hands each match's captures to
/// `make` to produce zero or more `(seen-key, effect)` pairs, dedup-inserts
/// each key into `seen` (erroring on a repeat grouping factor), pushes the
/// effects keyed `(rank, source position)`, and strips the first match out of
/// `work` — until `regex` no longer matches. This is the skeleton shared by
/// all five RE-extraction stages in [`extract_random_effects`]; everything
/// stage-specific (how many effects one match yields,
/// `EmptySlopeTerm`/fallback logic) lives in `make`.
///
/// The position is the match text's byte offset in the original `rhs`, so a
/// final stable sort can restore formula order after the stages run. For
/// well-formed input the match text is a verbatim substring of `rhs`, and its
/// first occurrence is the right one: duplicate grouping factors are rejected
/// (`DuplicateGroupingVar`), so every match text contains a distinct group
/// name and occurs exactly once. The `unwrap_or` is not decoration: degenerate
/// input can embed one RE term inside another (e.g. `(1+x+(1|a/b)|g)`), and an
/// earlier stage's removal then leaves a seam a later stage matches across —
/// the seam match's text is absent from `rhs` and `find` returns `None`. Such
/// input parses without error today and must not start panicking; `usize::MAX`
/// parks any seam match at the end, in stage order via the stable sort.
fn extract_stage(
    rhs: &str,
    rank: u8,
    work: &mut String,
    seen: &mut std::collections::BTreeSet<String>,
    effects: &mut Vec<((u8, usize), RandomEffect)>,
    regex: &Regex,
    mut make: impl FnMut(&regex::Captures) -> Result<Vec<(String, RandomEffect)>, ParseError>,
) -> Result<(), ParseError> {
    loop {
        let snapshot = work.clone();
        let Some(m) = regex.captures(&snapshot) else {
            break;
        };
        let pos = rhs.find(m.get(0).unwrap().as_str()).unwrap_or(usize::MAX);
        for (name, effect) in make(&m)? {
            if !seen.insert(name.clone()) {
                return Err(ParseError::DuplicateGroupingVar { name });
            }
            effects.push(((rank, pos), effect));
        }
        *work = regex.replacen(&snapshot, 1, "").into_owned();
    }
    Ok(())
}

fn extract_random_effects(rhs: &str) -> Result<(Vec<RandomEffect>, String), ParseError> {
    use std::collections::BTreeSet;

    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut effects: Vec<((u8, usize), RandomEffect)> = Vec::new();
    let mut work = rhs.to_string();

    // 0. Reject intercept suppression: (0+…|g), (-1+…|g), (0|g), (-1|g).
    if RE_SUPPRESS.is_match(rhs) {
        return Err(ParseError::RandomInterceptSuppressionUnsupported);
    }

    // Stage order below is load-bearing for MATCHING only (suppression before
    // intercept, interaction before plain intercept, …); the final sort keyed
    // `(rank, source position)` restores formula order. Rank 0 keeps a nested
    // pair first regardless of where it was written: `NestedWithin` is
    // interpreted relative to the PRIMARY grouping (`materialize.rs`), so a
    // nested term must supply the primary — pure formula order would lower
    // `y ~ (1|c1) + (1|g1/g2)` with `c1` primary and hand the kernel a nested
    // extra whose parent is not the primary. Every other stage gets rank 1.

    // 1. Nested intercept: (1|A/B) → two entries, parent then joined-with-parent.
    extract_stage(
        rhs,
        0,
        &mut work,
        &mut seen,
        &mut effects,
        &RE_NESTED,
        |m| {
            let parent_name = m.get(1).unwrap().as_str().to_string();
            let child_name = m.get(2).unwrap().as_str().to_string();
            let joined = format!("{parent_name}:{child_name}");
            Ok(vec![
                (
                    parent_name.clone(),
                    RandomEffect::Intercept {
                        group: parent_name.clone(),
                        parent: None,
                    },
                ),
                (
                    joined.clone(),
                    RandomEffect::Intercept {
                        group: joined,
                        parent: Some(parent_name),
                    },
                ),
            ])
        },
    )?;

    // 2. Random slope: (1+x+y|g) — falls back to a plain Intercept if every
    // token was "1" (no non-"1" slope vars survive the filter).
    extract_stage(rhs, 1, &mut work, &mut seen, &mut effects, &RE_SLOPE, |m| {
        let var_list_raw = m.get(1).unwrap().as_str();
        let group = m.get(2).unwrap().as_str().to_string();
        let raw_tokens: Vec<&str> = var_list_raw
            .split('+')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if raw_tokens.is_empty() {
            return Err(ParseError::EmptySlopeTerm { group });
        }
        let vars: Vec<String> = raw_tokens
            .iter()
            .filter(|s| **s != "1")
            .map(|s| String::from(*s))
            .collect();
        let effect = if vars.is_empty() {
            RandomEffect::Intercept {
                group: group.clone(),
                parent: None,
            }
        } else {
            RandomEffect::Slope {
                group: group.clone(),
                vars,
            }
        };
        Ok(vec![(group, effect)])
    })?;

    // 2.5. Implicit-intercept slope: (x|g), (x+z|g) — equivalent to (1+x|g).
    // Unlike stage 2, always yields a Slope (never falls back to Intercept):
    // an empty `vars` here means the term was `(1|g)`, which RE_ISLOPE's
    // `[_A-Za-z]`-starting group excludes from matching in the first place.
    extract_stage(
        rhs,
        1,
        &mut work,
        &mut seen,
        &mut effects,
        &RE_ISLOPE,
        |m| {
            let var_list_raw = m.get(1).unwrap().as_str();
            let group = m.get(2).unwrap().as_str().to_string();
            let vars: Vec<String> = var_list_raw
                .split('+')
                .map(str::trim)
                .filter(|s| !s.is_empty() && *s != "1")
                .map(String::from)
                .collect();
            Ok(vec![(group.clone(), RandomEffect::Slope { group, vars })])
        },
    )?;

    // 2.7. Crossed interaction grouping: (1|A:B) — a random intercept keyed by
    // the combination of two existing factor columns (NOT nesting: parent stays
    // None, unlike step 1's `(1|A/B)` inner term).
    extract_stage(
        rhs,
        1,
        &mut work,
        &mut seen,
        &mut effects,
        &RE_INTERACTION,
        |m| {
            let lhs = m.get(1).unwrap().as_str().to_string();
            let rhs = m.get(2).unwrap().as_str().to_string();
            let joined = format!("{lhs}:{rhs}");
            Ok(vec![(
                joined.clone(),
                RandomEffect::Intercept {
                    group: joined,
                    parent: None,
                },
            )])
        },
    )?;

    // 3. Random intercept: (1|g)
    extract_stage(rhs, 1, &mut work, &mut seen, &mut effects, &RE_INT, |m| {
        let name = m.get(1).unwrap().as_str().to_string();
        Ok(vec![(
            name.clone(),
            RandomEffect::Intercept {
                group: name,
                parent: None,
            },
        )])
    })?;

    // Restore formula order (stable: a nested pair's parent stays immediately
    // before its child — both carry one match's position).
    effects.sort_by_key(|(key, _)| *key);
    let effects = effects.into_iter().map(|(_, e)| e).collect();

    // Clean stray "+" — collapse "++", trim leading/trailing "+".
    let cleaned = clean_residual_plusses(&work);
    Ok((effects, cleaned))
}

fn clean_residual_plusses(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_plus = false;
    for ch in s.chars() {
        if ch == '+' {
            if !prev_plus && !out.is_empty() {
                out.push('+');
                prev_plus = true;
            }
        } else if !ch.is_whitespace() {
            out.push(ch);
            prev_plus = false;
        }
    }
    while out.starts_with('+') {
        out.remove(0);
    }
    while out.ends_with('+') {
        out.pop();
    }
    out
}

fn split_at_separator(s: &str) -> (String, String) {
    if let Some((l, r)) = s.split_once('~') {
        (l.to_string(), r.to_string())
    } else if let Some((l, r)) = s.split_once('=') {
        (l.to_string(), r.to_string())
    } else {
        ("explained_variable".to_string(), s.to_string())
    }
}

// The only legitimate `-` is the fixed-intercept removal (`- 1`), which the
// caller strips via `NO_INTERCEPT_MINUS`/`NO_INTERCEPT_ZERO` before this runs
// — so by the time `find_term_removal` sees a `-`, it is always unsupported,
// including a leftover `-1` that wasn't at a term boundary (`x-10`, `x-1-z`).
fn find_term_removal(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b'-' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn parse_rhs(rhs: &str) -> Result<(Vec<String>, Vec<Term>), ParseError> {
    use std::collections::BTreeSet;

    let mut predictors: Vec<String> = Vec::new();
    let mut seen_pred: BTreeSet<String> = BTreeSet::new();
    let mut terms: Vec<Term> = Vec::new();
    let mut seen_term: BTreeSet<String> = BTreeSet::new();

    for raw_term in rhs.split('+') {
        let term = raw_term.trim();
        if term.is_empty() {
            continue;
        }

        if term.contains('*') {
            let vars = parse_identifier_list(term, &['*'])?;
            register_vars(&vars, &mut predictors, &mut seen_pred);
            for v in &vars {
                if seen_term.insert(v.clone()) {
                    terms.push(Term::Main { name: v.clone() });
                }
            }
            for r in 2..=vars.len() {
                for combo in combinations(&vars, r) {
                    let key = combo.join(":");
                    if seen_term.insert(key) {
                        terms.push(Term::Interaction { vars: combo });
                    }
                }
            }
        } else if term.contains(':') {
            let vars = parse_identifier_list(term, &[':'])?;
            register_vars(&vars, &mut predictors, &mut seen_pred);
            let key = vars.join(":");
            if seen_term.insert(key) {
                terms.push(Term::Interaction { vars });
            }
        } else {
            let name = parse_single_identifier(term)?;
            if seen_pred.insert(name.clone()) {
                predictors.push(name.clone());
            }
            if seen_term.insert(name.clone()) {
                terms.push(Term::Main { name });
            }
        }
    }

    Ok((predictors, terms))
}

fn parse_single_identifier(s: &str) -> Result<String, ParseError> {
    if is_identifier(s) || parse_transform(s).is_some() {
        Ok(s.to_string())
    } else {
        Err(ParseError::Syntax {
            pos: 0,
            msg: format!("expected identifier, got '{s}'"),
        })
    }
}

/// A whitelisted single-column transform. The whitelist is deliberately
/// closed: R's `poly()` defaults to orthogonal polynomials, so emitting raw
/// powers under its name would claim R's column name for a different column
/// (`docs/parity_gaps.md` #6 in the wrapper repo); `I(x^k)` covers the same
/// models with an honest name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Transform {
    Log,
    Sqrt,
    Exp,
    /// `I(x^k)`, `k ≥ 2`.
    Pow(u32),
}

static TRANSFORM_CALL: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(log|sqrt|exp)\(([_A-Za-z][_A-Za-z0-9]*)\)$").unwrap());
static TRANSFORM_POW: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^I\(([_A-Za-z][_A-Za-z0-9]*)\^([2-9]|[1-9][0-9])\)$").unwrap());

/// `log(x)` / `sqrt(x)` / `exp(x)` / `I(x^k)` with a single bare column name
/// inside → the transform and that name. The spelling is used verbatim as the
/// design column name, which equals what R's `model.matrix` deparses
/// (whitespace was stripped by `parse`).
pub(super) fn parse_transform(s: &str) -> Option<(Transform, &str)> {
    if let Some(m) = TRANSFORM_CALL.captures(s) {
        let t = match m.get(1).unwrap().as_str() {
            "log" => Transform::Log,
            "sqrt" => Transform::Sqrt,
            _ => Transform::Exp,
        };
        return Some((t, m.get(2).unwrap().as_str()));
    }
    let m = TRANSFORM_POW.captures(s)?;
    let k: u32 = m.get(2).unwrap().as_str().parse().ok()?;
    Some((Transform::Pow(k), m.get(1).unwrap().as_str()))
}

fn parse_identifier_list(s: &str, seps: &[char]) -> Result<Vec<String>, ParseError> {
    let mut parts: Vec<&str> = vec![s];
    for sep in seps {
        parts = parts.into_iter().flat_map(|p| p.split(*sep)).collect();
    }
    parts
        .into_iter()
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(parse_single_identifier)
        .collect()
}

fn is_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_alphanumeric() || c == '_')
}

fn register_vars(
    vars: &[String],
    predictors: &mut Vec<String>,
    seen: &mut std::collections::BTreeSet<String>,
) {
    for v in vars {
        if seen.insert(v.clone()) {
            predictors.push(v.clone());
        }
    }
}

fn combinations<T: Clone>(items: &[T], r: usize) -> Vec<Vec<T>> {
    let n = items.len();
    if r == 0 || r > n {
        return vec![];
    }
    let mut idx: Vec<usize> = (0..r).collect();
    let mut out: Vec<Vec<T>> = Vec::new();
    loop {
        out.push(idx.iter().map(|&i| items[i].clone()).collect());
        let mut i = r;
        while i > 0 && idx[i - 1] == n - r + (i - 1) {
            i -= 1;
        }
        if i == 0 {
            break;
        }
        idx[i - 1] += 1;
        for j in i..r {
            idx[j] = idx[j - 1] + 1;
        }
    }
    out
}
