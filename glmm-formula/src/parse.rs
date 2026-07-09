//! Data-free formula parser. The AST shape and the [`ParseError`] Display
//! strings (`error.rs`) are a fixed contract: the parse test suite matches
//! error substrings against them, so both must stay stable across changes to
//! this file.
//!
//! `*` is desugared into main effects + all interactions and `A/B` into a
//! nesting relation *while parsing* — there is no separate normalize pass.

use crate::error::ParseError;
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
static RE_SLOPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(1\+([^|]+?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap());
static RE_ISLOPE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(([_A-Za-z][^|]*?)\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap());
static RE_INT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*)\)").unwrap());
static RE_INTERACTION: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\(1\|([_A-Za-z][_A-Za-z0-9]*):([_A-Za-z][_A-Za-z0-9]*)\)").unwrap()
});

/// The parsed formula AST. A frozen contract — its shape must stay stable
/// since the parse test suite depends on it.
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
    /// Random effects in parser-extraction order (nested → explicit slopes →
    /// implicit slopes → crossed interaction groupings → plain intercepts — NOT
    /// formula order); empty when the formula declares none. The first entry's
    /// grouping is the primary grouping.
    pub random_effects: Vec<RandomEffect>,
}

/// A fixed-effect term. `Interaction` holds its component variable names in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// A single main effect, e.g. `Main { name: "x1" }`.
    Main { name: String },
    /// An interaction, e.g. `Interaction { vars: ["x1", "x2"] }` (`x1:x2`).
    Interaction { vars: Vec<String> },
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
pub fn parse(input: &str) -> Result<ParsedFormula, ParseError> {
    let cleaned: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.is_empty() {
        return Err(ParseError::EmptyFormula);
    }

    let (mut dep, rhs) = split_at_separator(&cleaned);
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
    })
}

fn extract_random_effects(rhs: &str) -> Result<(Vec<RandomEffect>, String), ParseError> {
    use std::collections::BTreeSet;

    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut effects: Vec<RandomEffect> = Vec::new();
    let mut work = rhs.to_string();

    // 0. Reject intercept suppression: (0+…|g), (-1+…|g), (0|g), (-1|g).
    if RE_SUPPRESS.is_match(rhs) {
        return Err(ParseError::RandomInterceptSuppressionUnsupported);
    }

    // 1. Nested intercept: (1|A/B) → push pair, replace with empty.
    let re_nested = &*RE_NESTED;
    loop {
        let snapshot = work.clone();
        let Some(m) = re_nested.captures(&snapshot) else {
            break;
        };
        let parent_name = m.get(1).unwrap().as_str().to_string();
        let child_name = m.get(2).unwrap().as_str().to_string();
        let joined = format!("{parent_name}:{child_name}");
        for name in [&parent_name, &joined] {
            if !seen.insert(name.clone()) {
                return Err(ParseError::DuplicateGroupingVar { name: name.clone() });
            }
        }
        effects.push(RandomEffect::Intercept {
            group: parent_name.clone(),
            parent: None,
        });
        effects.push(RandomEffect::Intercept {
            group: joined,
            parent: Some(parent_name),
        });
        work = re_nested.replacen(&snapshot, 1, "").into_owned();
    }

    // 2. Random slope: (1+x+y|g)
    let re_slope = &*RE_SLOPE;
    loop {
        let snapshot = work.clone();
        let Some(m) = re_slope.captures(&snapshot) else {
            break;
        };
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
        if !seen.insert(group.clone()) {
            return Err(ParseError::DuplicateGroupingVar {
                name: group.clone(),
            });
        }
        if vars.is_empty() {
            effects.push(RandomEffect::Intercept {
                group,
                parent: None,
            });
        } else {
            effects.push(RandomEffect::Slope { group, vars });
        }
        work = re_slope.replacen(&snapshot, 1, "").into_owned();
    }

    // 2.5. Implicit-intercept slope: (x|g), (x+z|g) — equivalent to (1+x|g).
    let re_islope = &*RE_ISLOPE;
    loop {
        let snapshot = work.clone();
        let Some(m) = re_islope.captures(&snapshot) else {
            break;
        };
        let var_list_raw = m.get(1).unwrap().as_str();
        let group = m.get(2).unwrap().as_str().to_string();
        let vars: Vec<String> = var_list_raw
            .split('+')
            .map(str::trim)
            .filter(|s| !s.is_empty() && *s != "1")
            .map(String::from)
            .collect();
        if !seen.insert(group.clone()) {
            return Err(ParseError::DuplicateGroupingVar {
                name: group.clone(),
            });
        }
        effects.push(RandomEffect::Slope { group, vars });
        work = re_islope.replacen(&snapshot, 1, "").into_owned();
    }

    // 2.7. Crossed interaction grouping: (1|A:B) — a random intercept keyed by
    // the combination of two existing factor columns (NOT nesting: parent stays
    // None, unlike step 1's `(1|A/B)` inner term).
    let re_interaction = &*RE_INTERACTION;
    loop {
        let snapshot = work.clone();
        let Some(m) = re_interaction.captures(&snapshot) else {
            break;
        };
        let lhs = m.get(1).unwrap().as_str().to_string();
        let rhs = m.get(2).unwrap().as_str().to_string();
        let joined = format!("{lhs}:{rhs}");
        if !seen.insert(joined.clone()) {
            return Err(ParseError::DuplicateGroupingVar { name: joined });
        }
        effects.push(RandomEffect::Intercept {
            group: joined,
            parent: None,
        });
        work = re_interaction.replacen(&snapshot, 1, "").into_owned();
    }

    // 3. Random intercept: (1|g)
    let re_int = &*RE_INT;
    loop {
        let snapshot = work.clone();
        let Some(m) = re_int.captures(&snapshot) else {
            break;
        };
        let name = m.get(1).unwrap().as_str().to_string();
        if !seen.insert(name.clone()) {
            return Err(ParseError::DuplicateGroupingVar { name });
        }
        effects.push(RandomEffect::Intercept {
            group: name,
            parent: None,
        });
        work = re_int.replacen(&snapshot, 1, "").into_owned();
    }

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

fn find_term_removal(s: &str) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => depth -= 1,
            b'-' if depth == 0 => {
                let next = bytes.get(i + 1).copied().unwrap_or(b' ');
                if !next.is_ascii_digit() {
                    return Some(i);
                }
            }
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
    if is_identifier(s) {
        Ok(s.to_string())
    } else {
        Err(ParseError::Syntax {
            pos: 0,
            msg: format!("expected identifier, got '{s}'"),
        })
    }
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
