#![cfg(feature = "formula")]
//! Proves the parser is a superset of MCPower's — the 28 canonical cases parse
//! identically; everything below the canonical suite is glmm-only grammar.
//! Two corpora:
//!   - the 28-case canonical suite (`configs/formula-fixtures/canonical-suite.json`),
//!     inlined here as Rust data and checked via the same canonical normalization
//!     the app-spec `formula_suite.rs` harness uses;
//!   - the random-effects cases (`random_effects_parse.rs`) as direct AST asserts.
//!
//! Pure — no data table, except `repeated_random_column_is_refused`, whose
//! refusal needs the lowering.

use glmm::formula::{parse, ParseError, ParsedFormula, RandomEffect, Term};

/// (id, formula, dependent, fixed effects, random effects).
type SuccessCase = (
    &'static str,
    &'static str,
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
);

/// Normalize an AST to the port-neutral canonical shape (mirrors app-spec's
/// `canonical`): each term → its coefficient-name string, each RE → `intercept|g`
/// / `slope(v1,v2)|g`.
fn canonical(p: &ParsedFormula) -> (String, Vec<String>, Vec<String>) {
    let fixed = p
        .terms
        .iter()
        .map(|t| match t {
            Term::Main { name } => name.clone(),
            Term::Interaction { vars } => vars.join(":"),
        })
        .collect();
    let re = p
        .random_effects
        .iter()
        .map(|r| match r {
            RandomEffect::Intercept { group, .. } => format!("intercept|{group}"),
            RandomEffect::Slope { group, vars } => format!("slope({})|{group}", vars.join(",")),
            RandomEffect::NoIntercept { group, vars } => {
                format!("nointercept({})|{group}", vars.join(","))
            }
            RandomEffect::ZeroCorr {
                group,
                vars,
                intercept,
            } => format!(
                "zerocorr({}{})|{group}",
                if *intercept { "1," } else { "0," },
                vars.join(",")
            ),
        })
        .collect();
    (p.dependent.clone(), fixed, re)
}

/// (id, formula, dependent, fixed effects, random effects) — the success cases of
/// the canonical suite, verbatim from `canonical-suite.json`.
const SUCCESS: &[SuccessCase] = &[
    ("001_ols_simple", "y ~ x1 + x2", "y", &["x1", "x2"], &[]),
    (
        "002_star_expansion",
        "y ~ x1*x2*x3",
        "y",
        &["x1", "x2", "x3", "x1:x2", "x1:x3", "x2:x3", "x1:x2:x3"],
        &[],
    ),
    (
        "003_colon_interaction",
        "y ~ x1 + x2 + x1:x2",
        "y",
        &["x1", "x2", "x1:x2"],
        &[],
    ),
    (
        "008_default_dep",
        "x1 + x2",
        "explained_variable",
        &["x1", "x2"],
        &[],
    ),
    (
        "009_equals_separator",
        "y = x1 + x2",
        "y",
        &["x1", "x2"],
        &[],
    ),
    (
        "011_star_two",
        "y ~ x1*x2",
        "y",
        &["x1", "x2", "x1:x2"],
        &[],
    ),
    (
        "012_mixed_star_colon",
        "y ~ a*b + c:d",
        "y",
        &["a", "b", "a:b", "c:d"],
        &[],
    ),
    (
        "013_colon_three_way",
        "y ~ x1:x2:x3",
        "y",
        &["x1:x2:x3"],
        &[],
    ),
    ("014_dup_main", "y ~ x1 + x1 + x2", "y", &["x1", "x2"], &[]),
    (
        "015_star_plus_main",
        "y ~ x1*x2 + z",
        "y",
        &["x1", "x2", "x1:x2", "z"],
        &[],
    ),
    (
        "004_random_intercept",
        "y ~ x + (1|g)",
        "y",
        &["x"],
        &["intercept|g"],
    ),
    (
        "005_random_slope_one_var",
        "y ~ x + (1+x|g)",
        "y",
        &["x"],
        &["slope(x)|g"],
    ),
    (
        "006_random_slope_multi",
        "y ~ x + z + (1+x+z|g)",
        "y",
        &["x", "z"],
        &["slope(x,z)|g"],
    ),
    (
        "007_nested",
        "y ~ x + (1|A/B)",
        "y",
        &["x"],
        &["intercept|A", "intercept|A:B"],
    ),
    (
        "010_re_only_no_fixed",
        "y ~ (1|g)",
        "y",
        &[],
        &["intercept|g"],
    ),
    (
        "017_two_groups",
        "y ~ x + (1|g) + (1|h)",
        "y",
        &["x"],
        &["intercept|g", "intercept|h"],
    ),
    (
        "018_star_with_re",
        "y ~ x1*x2 + (1|g)",
        "y",
        &["x1", "x2", "x1:x2"],
        &["intercept|g"],
    ),
    (
        "019_nested_with_fixed",
        "y ~ x + z + (1|school/class)",
        "y",
        &["x", "z"],
        &["intercept|school", "intercept|school:class"],
    ),
    (
        "020_slope_on_interaction_var",
        "y ~ x1 + x2 + x1:x2 + (1+x1|g)",
        "y",
        &["x1", "x2", "x1:x2"],
        &["slope(x1)|g"],
    ),
    (
        "021_multivar_slope_and_intercept",
        "y ~ x + z + (1|g) + (1+x+z|h)",
        "y",
        &["x", "z"],
        // Formula order: the intercept term is written first, so it lowers
        // first (and becomes the primary grouping) even though the parser's
        // slope stage runs before its intercept stage.
        &["intercept|g", "slope(x,z)|h"],
    ),
    (
        "022_implicit_intercept_slope",
        "y ~ x + (x|g)",
        "y",
        &["x"],
        &["slope(x)|g"],
    ),
];

/// (id, formula, error substring) — the error cases of the canonical suite.
const ERRORS: &[(&str, &str, &str)] = &[
    ("err_empty", "", "formula is empty"),
    ("err_bad_identifier", "y ~ 1x", "formula syntax error"),
    ("err_term_removal", "y ~ x1 - x2", "term removal"),
];

/// The canonical suite's other error cases, which this parser accepts: several
/// terms on one grouping and an intercept-free term parse, and the lowering
/// refuses a column in two terms on one grouping
/// (`repeated_random_column_is_refused`).
const ERRORS_THAT_PARSE: &[SuccessCase] = &[
    (
        "016_intercept_and_slope_same_group",
        "y ~ x + (1|g) + (1+x|g)",
        "y",
        &["x"],
        &["intercept|g", "slope(x)|g"],
    ),
    (
        "err_duplicate_group",
        "y ~ x + (1|g) + (1|g)",
        "y",
        &["x"],
        &["intercept|g", "intercept|g"],
    ),
    (
        "err_intercept_suppression_zero",
        "y ~ x + (0+x|g)",
        "y",
        &["x"],
        &["nointercept(x)|g"],
    ),
    (
        "err_intercept_suppression_minus_one",
        "y ~ x + (-1+x|g)",
        "y",
        &["x"],
        &["nointercept(x)|g"],
    ),
];

#[test]
fn canonical_suite_matches_mirror() {
    for &(id, formula, dep, fixed, re) in SUCCESS.iter().chain(ERRORS_THAT_PARSE) {
        let parsed = parse(formula).unwrap_or_else(|e| panic!("case {id} should parse, got {e}"));
        let (got_dep, got_fixed, got_re) = canonical(&parsed);
        assert_eq!(got_dep, dep, "case {id} dependent");
        assert_eq!(got_fixed, fixed, "case {id} fixed effects");
        assert_eq!(got_re, re, "case {id} random effects");
    }
    for &(id, formula, needle) in ERRORS {
        let msg = parse(formula)
            .expect_err(&format!("case {id} should error"))
            .to_string();
        assert!(msg.contains(needle), "case {id}: {msg:?} !~ {needle:?}");
    }
}

// ── The random-effects cases (random_effects_parse.rs), as direct AST asserts ──

#[test]
fn interaction_grouping_combined_with_plain_intercept() {
    let f = parse("y ~ x + (1|recipe:replicate) + (1|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![
            RandomEffect::Intercept {
                group: "recipe:replicate".into(),
                parent: None,
            },
            RandomEffect::Intercept {
                group: "g".into(),
                parent: None,
            },
        ]
    );
}

/// A column in two terms on one grouping, the intercept included, is refused
/// by the lowering (the parser cannot see a factor's columns), naming the
/// grouping and the column.
#[test]
fn repeated_random_column_is_refused() {
    use glmm::formula::{lower, Column, Error, Table};
    let labels = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(vec![1.0, 2.0, 0.5, 1.5, 2.5, 0.2]),
            ),
            (
                "x".into(),
                Column::Numeric(vec![0.1, 0.4, 0.2, 0.9, 0.3, 0.7]),
            ),
            (
                "g".into(),
                Column::factor_from_labels(&labels(&["a", "a", "b", "b", "c", "c"])),
            ),
            (
                "recipe".into(),
                Column::factor_from_labels(&labels(&["r", "s", "r", "s", "r", "s"])),
            ),
            (
                "replicate".into(),
                Column::factor_from_labels(&labels(&["1", "1", "2", "2", "3", "3"])),
            ),
        ],
        n: 6,
    };
    for (f, group, column) in [
        ("y ~ x + (1 | g) + (1 | g)", "g", "(Intercept)"),
        ("y ~ x + (x | g) + (0 + x | g)", "g", "x"),
        ("y ~ x + (1 | g) + (1 + x | g)", "g", "(Intercept)"),
        (
            "y ~ x + (1|recipe:replicate) + (1|recipe:replicate)",
            "recipe:replicate",
            "(Intercept)",
        ),
    ] {
        let err = match lower(f, &table, glmm::Family::Gaussian) {
            Ok(_) => panic!("{f}: expected a refusal"),
            Err(e) => e,
        };
        assert_eq!(
            err,
            Error::RandomColumnRepeated {
                group: group.into(),
                column: column.into()
            },
            "{f}"
        );
        let msg = err.to_string();
        assert!(
            msg.contains("(1 + x + f | g)") && msg.contains("(x + f || g)"),
            "{f}: {msg}"
        );
    }
}

/// `(lhs || g)` is one term per variable, the intercept its own term, as
/// lme4's `expandDoubleVerts` (reformulas 0.4.1) writes `(1 + x + f || g)`:
/// `(1 | g) + (0 + x | g) + (0 + f | g)`. A factor's term stays one term.
#[test]
fn double_bar_expands_like_lme4() {
    let int = |g: &str| RandomEffect::Intercept {
        group: g.into(),
        parent: None,
    };
    let no_int = |g: &str, v: &str| RandomEffect::NoIntercept {
        group: g.into(),
        vars: vec![v.into()],
    };
    for (f, want) in [
        (
            "y ~ x + f + (1 + x + f || g)",
            vec![int("g"), no_int("g", "x"), no_int("g", "f")],
        ),
        ("y ~ x + (x || g)", vec![int("g"), no_int("g", "x")]),
        (
            "y ~ x + z + (0 + x + z || g)",
            vec![no_int("g", "x"), no_int("g", "z")],
        ),
        (
            "y ~ x + (1 | h) + (x || g)",
            vec![int("h"), int("g"), no_int("g", "x")],
        ),
        ("y ~ x + (1 || g)", vec![int("g")]),
    ] {
        assert_eq!(parse(f).unwrap().random_effects, want, "{f}");
    }
    assert_eq!(
        parse("y ~ x + (0 || g)"),
        Err(ParseError::EmptySlopeTerm { group: "g".into() })
    );
}

/// `0 +` and `- 1` drop the random intercept from a term; a term with nothing
/// else left is an empty slope term.
#[test]
fn intercept_suppression_parses() {
    for f in [
        "y ~ x + (0 + x | g)",
        "y ~ x + (-1 + x | g)",
        "y ~ x + (x - 1 | g)",
        "y ~ x + (x + 0 | g)",
    ] {
        assert_eq!(
            parse(f).unwrap().random_effects,
            vec![RandomEffect::NoIntercept {
                group: "g".into(),
                vars: vec!["x".into()]
            }],
            "{f}"
        );
    }
    for f in ["y ~ x + (0 | g)", "y ~ x + (-1 | g)"] {
        assert_eq!(
            parse(f),
            Err(ParseError::EmptySlopeTerm { group: "g".into() }),
            "{f}"
        );
    }
    let msg = parse("y ~ x + (1 + x - 1 | g)").unwrap_err().to_string();
    assert!(msg.contains("both writes"), "{msg}");
}

/// `zerocorr(...)` is one node: only the lowering knows a factor's columns.
#[test]
fn zerocorr_parses() {
    let zc = |vars: &[&str], intercept| RandomEffect::ZeroCorr {
        group: "g".into(),
        vars: vars.iter().map(|v| v.to_string()).collect(),
        intercept,
    };
    for (f, want) in [
        ("y ~ x + zerocorr(1 + x | g)", zc(&["x"], true)),
        ("y ~ x + zerocorr(x | g)", zc(&["x"], true)),
        (
            "y ~ x + z + zerocorr(0 + x + z | g)",
            zc(&["x", "z"], false),
        ),
    ] {
        let p = parse(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(p.random_effects, vec![want], "{f}");
    }
    assert_eq!(
        parse("y ~ x + zerocorr(1 | g)").unwrap().random_effects,
        vec![RandomEffect::Intercept {
            group: "g".into(),
            parent: None
        }]
    );
}

/// `||` and `zerocorr` take one grouping factor, not a nesting.
#[test]
fn double_bar_on_nested_is_a_syntax_error() {
    for f in [
        "y ~ x + (1 + x || A/B)",
        "y ~ x + (x || A/B)",
        "y ~ x + zerocorr(1 + x | A/B)",
    ] {
        let err = parse(f).unwrap_err();
        assert!(matches!(err, ParseError::Syntax { .. }), "{f}: {err:?}");
        assert!(err.to_string().contains("A/B"), "{f}: {err}");
    }
}

/// Whitespace is stripped before matching, so `(1+ + |g)` is the same
/// term as `(1++|g)`.
#[test]
fn a_slope_term_of_only_plus_signs_is_an_empty_slope_term() {
    for f in ["y ~ x + (1++|g)", "y ~ x + (1+ + |g)"] {
        let err = parse(f).unwrap_err();
        assert_eq!(err, ParseError::EmptySlopeTerm { group: "g".into() }, "{f}");
        assert_eq!(err.to_string(), "empty slope term for group g", "{f}");
    }
}

#[test]
fn implicit_intercept_multivar_slope() {
    let p = parse("y ~ x + z + (x+z|g)").unwrap();
    assert_eq!(
        p.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into(), "z".into()]
        }]
    );
}

#[test]
fn redundant_explicit_one_in_implicit_form_is_dropped() {
    let p = parse("y ~ x + (x+1|g)").unwrap();
    assert_eq!(
        p.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()]
        }]
    );
}

#[test]
fn nested_term_sorts_first_even_when_written_last() {
    // `NestedWithin` is interpreted relative to the primary grouping, so a
    // nested pair must supply the primary — it outranks formula order.
    let f = parse("y ~ x + (1|c1) + (1|g1/g2)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![
            RandomEffect::Intercept {
                group: "g1".into(),
                parent: None,
            },
            RandomEffect::Intercept {
                group: "g1:g2".into(),
                parent: Some("g1".into()),
            },
            RandomEffect::Intercept {
                group: "c1".into(),
                parent: None,
            },
        ]
    );
}

/// Degenerate input: an RE term embedded in another. The nested stage removes
/// `(1|a/b)` and the slope stage then matches across the seam (`(1+x+|g)`),
/// whose text does not occur in the original RHS: the position lookup must
/// fall back (parse.rs `extract_stage`), not panic. What this input parses
/// to is not a contract.
#[test]
fn seam_match_from_embedded_re_term_does_not_panic() {
    let _ = parse("y ~ (1+x+(1|a/b)|g)");
}

// ── Fixed-intercept removal ─────────────────────────────────────────────────

#[test]
fn minus_one_and_zero_plus_drop_the_intercept() {
    for f in [
        "y ~ x - 1",
        "y ~ x -1",
        "y ~ -1 + x",
        "y ~ 0 + x",
        "y ~ x + 0",
        "y ~ 0 + x + (1|g)",
    ] {
        let p = parse(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert!(!p.has_intercept, "{f}");
        assert_eq!(p.predictors, vec!["x".to_string()], "{f}");
    }
    assert!(parse("y ~ x").unwrap().has_intercept);
    assert!(parse("y ~ x + (1|g)").unwrap().has_intercept);
}

#[test]
fn other_term_removal_still_rejected() {
    for f in ["y ~ x - z", "y ~ x - 10", "y ~ x - 1 - z"] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(msg.contains("term removal"), "{f}: {msg}");
    }
}

#[test]
fn intercept_only_removal_leaves_no_terms() {
    let p = parse("y ~ -1 + (1|g)").unwrap();
    assert!(!p.has_intercept);
    assert!(p.terms.is_empty());
    assert_eq!(p.random_effects.len(), 1);
}

// ── Explicit fixed intercept ────────────────────────────────────────────────

#[test]
fn explicit_one_is_the_implied_intercept() {
    for (written, implied) in [
        ("y ~ 1 + x", "y ~ x"),
        ("y ~ x + 1", "y ~ x"),
        ("y ~ 1 + x + (1 + x | g)", "y ~ x + (1 + x | g)"),
        ("y ~ 1 + offset(log(e)) + x", "y ~ offset(log(e)) + x"),
        ("y ~ 1 + (1|g)", "y ~ (1|g)"),
        ("y ~ 1", "y ~ (1|g)"),
    ] {
        let p = parse(written).unwrap_or_else(|e| panic!("{written}: {e}"));
        let q = parse(implied).unwrap();
        assert!(p.has_intercept, "{written}");
        assert_eq!(p.predictors, q.predictors, "{written}");
        assert_eq!(p.terms, q.terms, "{written}");
    }
}

#[test]
fn explicit_one_beside_an_intercept_removal_is_rejected() {
    for f in ["y ~ 1 + x - 1", "y ~ 0 + 1 + x", "y ~ 1 + x + 0"] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(msg.contains("intercept"), "{f}: {msg}");
    }
}

// ── Whitelist transforms ────────────────────────────────────────────────────

#[test]
fn whitelist_transforms_are_terms_named_as_r_spells_them() {
    let p = parse("y ~ log(x) + I(z ^ 2) + sqrt(w):f + exp(x)*z").unwrap();
    let (_, fixed, _) = canonical(&p);
    assert_eq!(
        fixed,
        ["log(x)", "I(z^2)", "sqrt(w):f", "exp(x)", "z", "exp(x):z"].map(String::from)
    );
    assert_eq!(p.predictors[0], "log(x)");
}

#[test]
fn transform_random_slope_parses() {
    let p = parse("y ~ log(x) + (1 + log(x) | g) + (I(z^2) | h)").unwrap();
    assert_eq!(
        p.random_effects,
        vec![
            RandomEffect::Slope {
                group: "g".into(),
                vars: vec!["log(x)".into()]
            },
            RandomEffect::Slope {
                group: "h".into(),
                vars: vec!["I(z^2)".into()]
            },
        ]
    );
}

#[test]
fn transform_term_beside_a_random_intercept_does_not_confuse_re_extraction() {
    // The slope regexes exclude parens, so `(x)+(1|g)` is not an
    // implicit-intercept slope term.
    let p = parse("y ~ log(x) + (1|g)").unwrap();
    let (_, fixed, re) = canonical(&p);
    assert_eq!(fixed, vec!["log(x)".to_string()]);
    assert_eq!(re, vec!["intercept|g".to_string()]);
}

#[test]
fn non_whitelist_calls_are_rejected() {
    for f in [
        "y ~ poly(x, 2)",
        "y ~ log(x, 10)",
        "y ~ log(sqrt(x))",
        "y ~ I(x^2 + z)",
        "y ~ I(x^1)",
        "y ~ I(x^2.5)",
        "y ~ log(x + z)",
        "y ~ abs(x)",
    ] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(msg.contains("formula syntax error"), "{f}: {msg}");
    }
}

// ── offset() term ───────────────────────────────────────────────────────────

#[test]
fn offset_term_is_recorded_and_removed_from_the_design() {
    let p = parse("y ~ x + offset(log(exposure)) + (1|g)").unwrap();
    assert_eq!(p.offset.as_deref(), Some("log(exposure)"));
    let (_, fixed, re) = canonical(&p);
    assert_eq!(fixed, vec!["x".to_string()]);
    assert_eq!(re, vec!["intercept|g".to_string()]);
    assert_eq!(p.predictors, vec!["x".to_string()]);

    let p = parse("y ~ offset(e) + x").unwrap();
    assert_eq!(p.offset.as_deref(), Some("e"));
    assert!(parse("y ~ x").unwrap().offset.is_none());
}

#[test]
fn offset_term_rejections() {
    for (f, needle) in [
        ("y ~ x + offset(a) + offset(b)", "one offset"),
        ("y ~ x + offset(a + b)", "formula syntax error"),
        ("y ~ x + offset(poly(a, 2))", "formula syntax error"),
        ("y ~ x + offset()", "formula syntax error"),
        ("y ~ foffset(e) + x", "formula syntax error"),
        ("y ~ x*offset(e)", "formula syntax error"),
        ("y ~ a:offset(e)", "formula syntax error"),
    ] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(msg.contains(needle), "{f}: {msg}");
    }
}

// ── cbind() response ────────────────────────────────────────────────────────

#[test]
fn cbind_response_is_recorded() {
    let p = parse("cbind(inc, size - inc) ~ period + (1|herd)");
    assert!(p.is_err(), "arithmetic inside cbind is not accepted: {p:?}");
    let p = parse("cbind(inc, fail) ~ period + (1|herd)").unwrap();
    assert_eq!(p.cbind, Some(("inc".to_string(), "fail".to_string())));
    assert_eq!(p.dependent, "cbind(inc,fail)");
    assert!(parse("y ~ x").unwrap().cbind.is_none());
}

#[test]
fn cbind_rejections() {
    for f in [
        "cbind(a) ~ x",
        "cbind(a, b, c) ~ x",
        "cbind(a, log(b)) ~ x",
        "y ~ cbind(a, b)",
    ] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(msg.contains("formula syntax error"), "{f}: {msg}");
    }
}

// ── Repeated terms ──────────────────────────────────────────────────────────

#[test]
fn an_interaction_repeated_in_another_order_is_one_term() {
    // R's `terms.formula` keys a term by its set of variables: `g:f` is the
    // term `f:g`, and the first spelling keeps its place.
    for (f, want) in [
        ("y ~ f*g + g:f", &["f", "g", "f:g"][..]),
        ("y ~ f*g + g*f", &["f", "g", "f:g"]),
        ("y ~ f:g + g:f", &["f:g"]),
        ("y ~ g:f + f:g", &["g:f"]),
        (
            "y ~ x + c:b:a + a*b*c",
            &["x", "c:b:a", "a", "b", "c", "a:b", "a:c", "b:c"],
        ),
    ] {
        let (_, fixed, _) = canonical(&parse(f).unwrap());
        assert_eq!(fixed, want, "{f}");
    }
}

#[test]
fn a_variable_repeated_within_a_term_counts_once() {
    // R codes a term as its set of variables: `x:x` is `x`, `f:g:f` is `f:g`.
    for (f, want) in [
        ("y ~ x:x", &["x"][..]),
        ("y ~ x + x:x", &["x"]),
        ("y ~ x*x", &["x"]),
        ("y ~ f:g:f", &["f:g"]),
        ("y ~ x*z*x", &["x", "z", "x:z"]),
    ] {
        let p = parse(f).unwrap();
        let (_, fixed, _) = canonical(&p);
        assert_eq!(fixed, want, "{f}");
    }
    assert_eq!(
        parse("y ~ x:x").unwrap().terms,
        [Term::Main { name: "x".into() }]
    );
}

#[test]
fn an_empty_name_in_an_interaction_is_a_syntax_error() {
    for f in [
        "y ~ :",
        "y ~ *",
        "y ~ x:",
        "y ~ :x",
        "y ~ x::z",
        "y ~ x + * + z",
    ] {
        assert!(
            matches!(parse(f), Err(glmm::formula::ParseError::Syntax { .. })),
            "expected a syntax error for {f}, got {:?}",
            parse(f)
        );
    }
}

#[test]
fn a_random_slope_variable_repeated_counts_once() {
    // Mirrors `a_variable_repeated_within_a_term_counts_once` on the
    // random-effects side: lme4's `lFormula` also collapses `(1+x+x|g)` to a
    // single `x` slope.
    for (f, want) in [
        ("y ~ x + (1 + x + x | g)", "slope(x)|g"),
        ("y ~ x + (x + x | g)", "slope(x)|g"),
    ] {
        let p = parse(f).unwrap();
        let (_, _, re) = canonical(&p);
        assert_eq!(re, [want], "{f}");
    }
}

#[test]
fn star_lists_interactions_in_r_order() {
    // R expands `a*b*c*d` as `((a*b)*c)*d`, so within a degree the
    // interactions come ordered by their last variable (`CrossTerms`, R's
    // `model.c`); `model.matrix` gives `a:b a:c b:c a:d b:d c:d`.
    let (_, fixed, _) = canonical(&parse("y ~ a*b*c*d").unwrap());
    assert_eq!(
        fixed,
        [
            "a", "b", "c", "d", "a:b", "a:c", "b:c", "a:d", "b:d", "c:d", "a:b:c", "a:b:d",
            "a:c:d", "b:c:d", "a:b:c:d"
        ]
        .map(String::from)
    );
}

// ── Non-ASCII names ─────────────────────────────────────────────────────────

#[test]
fn non_ascii_names_parse_wherever_a_bare_name_does() {
    let p = parse("y ~ log(café) + I(ł^2) + ł + (1 + ł | grupa_ł) + (ź | g)").unwrap();
    let (_, fixed, re) = canonical(&p);
    assert_eq!(fixed, ["log(café)", "I(ł^2)", "ł"].map(String::from));
    assert_eq!(re, ["slope(ł)|grupa_ł", "slope(ź)|g"].map(String::from));

    let (_, _, re) = canonical(&parse("y ~ x + (1 | szkoła) + (1 | pień:gałąź)").unwrap());
    assert_eq!(
        re,
        ["intercept|szkoła", "intercept|pień:gałąź"].map(String::from)
    );

    let p = parse("y ~ x + (1 | szkoła/klasa_ę)").unwrap();
    assert_eq!(
        p.random_effects[1],
        RandomEffect::Intercept {
            group: "szkoła:klasa_ę".into(),
            parent: Some("szkoła".into())
        }
    );

    let p = parse("cbind(sukces, porażka) ~ x + offset(log(ekspozycja_ś))").unwrap();
    assert_eq!(p.cbind, Some(("sukces".into(), "porażka".into())));
    assert_eq!(p.offset.as_deref(), Some("log(ekspozycja_ś)"));
}

// ── A dangling `+` and the `^` operator ─────────────────────────────────────

/// `pos` is the character index of the `+` in the formula string, not its
/// byte offset (`ł` is two bytes).
fn syntax_error(pos: usize, msg: &str) -> Result<ParsedFormula, ParseError> {
    Err(ParseError::Syntax {
        pos,
        msg: msg.to_string(),
    })
}

#[test]
fn a_formula_ending_in_plus_is_a_syntax_error() {
    let msg = "the formula ends with '+': a term is missing after it";
    for (f, pos) in [
        ("y ~ x + ", 6),
        ("y ~ x + (1 | g) + ", 16),
        ("y ~ x + offset(z) +", 18),
        ("y ~ ł + ", 6),
    ] {
        assert_eq!(parse(f), syntax_error(pos, msg), "{f}");
    }
    // R accepts a leading and a doubled `+`.
    assert_eq!(parse("y ~ + x").unwrap(), parse("y ~ x").unwrap());
    assert_eq!(parse("y ~ x + + z").unwrap(), parse("y ~ x + z").unwrap());
}

/// `^` crosses terms as R does: `(a + b + c)^2` is every interaction of the
/// base's terms up to order 2, `a^2` is `a`, and `I(a^2)` stays a power. The
/// labels are `attr(terms(f), "term.labels")` from R 4.5.3, the random term
/// left out as lme4's `nobars` does.
#[test]
fn crossing_operator_follows_r_terms() {
    for (f, r_labels) in [
        (
            "y ~ (a + b + c)^2",
            &["a", "b", "c", "a:b", "a:c", "b:c"][..],
        ),
        ("y ~ (a + b)^3", &["a", "b", "a:b"]),
        ("y ~ a^2", &["a"]),
        ("y ~ (a + b)^2 + (1 | g)", &["a", "b", "a:b"]),
        (
            "y ~ x + (a + b + c)^3",
            &["x", "a", "b", "c", "a:b", "a:c", "b:c", "a:b:c"],
        ),
        ("y ~ (a + b)^2:c", &["a:c", "b:c", "a:b:c"]),
        (
            "y ~ (a*b + c)^2",
            &["a", "b", "c", "a:b", "a:c", "b:c", "a:b:c"],
        ),
        ("y ~ (ł + b)^2", &["ł", "b", "ł:b"]),
    ] {
        let p = parse(f).unwrap_or_else(|e| panic!("{f}: {e}"));
        assert_eq!(canonical(&p).1, r_labels, "{f}");
    }
    let (_, fixed, _) = canonical(&parse("y ~ I(a^2)").unwrap());
    assert_eq!(fixed, ["I(a^2)"]);
    // R refuses a power below 2 ("invalid power in formula").
    for f in ["y ~ (a + b)^1", "y ~ (a + b)^0", "y ~ (a + b)^x", "y ~ a^"] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(msg.contains("power"), "{f}: {msg}");
    }
}

// ── Grouping parentheses in the fixed part ──────────────────────────────────

/// A grouped formula parses to its written-out form, terms in R's order:
/// the term labels are `attr(terms(f), "term.labels")` from R 4.5.
#[test]
fn a_grouped_formula_parses_like_its_written_out_form() {
    for (grouped, written, r_labels) in [
        (
            "y ~ a*(b + c)",
            "y ~ a + b + c + a:b + a:c",
            &["a", "b", "c", "a:b", "a:c"][..],
        ),
        ("y ~ a:(b + c)", "y ~ a:b + a:c", &["a:b", "a:c"]),
        (
            "y ~ (a + b)*(c + d)",
            "y ~ a + b + c + d + a:c + a:d + b:c + b:d",
            &["a", "b", "c", "d", "a:c", "a:d", "b:c", "b:d"],
        ),
        (
            "y ~ (a + b)*c*d",
            "y ~ a + b + c + d + a:c + b:c + a:d + b:d + c:d + a:c:d + b:c:d",
            &[
                "a", "b", "c", "d", "a:c", "b:c", "a:d", "b:d", "c:d", "a:c:d", "b:c:d",
            ],
        ),
        ("y ~ (a + b)", "y ~ a + b", &["a", "b"]),
        (
            "y ~ a*(b + c*d)",
            "y ~ a + b + c + d + c:d + a:b + a:c + a:d + a:c:d",
            &["a", "b", "c", "d", "c:d", "a:b", "a:c", "a:d", "a:c:d"],
        ),
        ("y ~ c:(a + b)", "y ~ c:a + c:b", &["c:a", "c:b"]),
        (
            "y ~ x + a*(b + c) + a:b",
            "y ~ x + a + b + c + a:b + a:c",
            &["x", "a", "b", "c", "a:b", "a:c"],
        ),
        (
            "y ~ a*(log(x) + z)",
            "y ~ a + log(x) + z + a:log(x) + a:z",
            &["a", "log(x)", "z", "a:log(x)", "a:z"],
        ),
        (
            "y ~ (a*b*c)",
            "y ~ a*b*c",
            &["a", "b", "c", "a:b", "a:c", "b:c", "a:b:c"],
        ),
        (
            "y ~ a*((b + c))",
            "y ~ a*(b + c)",
            &["a", "b", "c", "a:b", "a:c"],
        ),
        (
            "y ~ a*(b + c) - 1",
            "y ~ a + b + c + a:b + a:c - 1",
            &["a", "b", "c", "a:b", "a:c"],
        ),
        (
            "y ~ 0 + a*(b + c)",
            "y ~ 0 + a + b + c + a:b + a:c",
            &["a", "b", "c", "a:b", "a:c"],
        ),
        (
            "y ~ a*(b + c) + (1 + b | g)",
            "y ~ a + b + c + a:b + a:c + (1 + b | g)",
            &["a", "b", "c", "a:b", "a:c"],
        ),
    ] {
        let p = parse(grouped).unwrap_or_else(|e| panic!("{grouped}: {e}"));
        assert_eq!(p, parse(written).unwrap(), "{grouped}");
        assert_eq!(canonical(&p).1, r_labels, "{grouped}");
    }
}

/// R numbers the variables by first appearance as written, where no
/// written-out form of the terms can: `a:c` would number `c` before `b`.
/// A term lists its variables in that order even where the degree sort
/// moves it ahead of one written before it: in `(a:b + c)*d` the term
/// `a:b:d` keeps `a` and `b` first, as R labels it.
#[test]
fn a_grouped_terms_variables_keep_their_written_order() {
    for (f, r_labels) in [
        ("y ~ (a + b):(c + d)", &["a:c", "a:d", "b:c", "b:d"][..]),
        ("y ~ (a:b + c)*d", &["c", "d", "a:b", "c:d", "a:b:d"]),
    ] {
        let p = parse(f).unwrap();
        assert_eq!(p.predictors, ["a", "b", "c", "d"], "{f}");
        assert_eq!(canonical(&p).1, r_labels, "{f}");
    }
}

#[test]
fn an_unbalanced_or_empty_group_is_a_syntax_error() {
    for (f, msg) in [
        ("y ~ a*(b + c", "unbalanced parentheses in term 'a*(b+c'"),
        (
            "y ~ a*(b + c))",
            "unbalanced parentheses in term 'a*(b+c))'",
        ),
        ("y ~ a*()", "empty parentheses in term 'a*()'"),
        ("y ~ a*(b + ())", "empty parentheses in term 'a*(b+())'"),
    ] {
        assert_eq!(parse(f), syntax_error(0, msg), "{f}");
    }
}

#[test]
fn a_group_refuses_what_the_fixed_part_refuses() {
    for f in ["y ~ a*(b - c)", "y ~ (a + b - c):d"] {
        assert_eq!(parse(f), Err(ParseError::TermRemovalUnsupported), "{f}");
    }
    let msg = "the intercept (`1`, `0` or `- 1`) cannot be written inside parentheses";
    for f in [
        "y ~ a*(b + 1)",
        "y ~ a*(0 + b)",
        "y ~ a*(b - 1 + c)",
        "y ~ (1 + a)*b",
    ] {
        assert_eq!(parse(f), syntax_error(0, msg), "{f}");
    }
    for (f, needle) in [
        (
            "y ~ a*(b + poly(x, 2))",
            "expected identifier, got 'poly(x,2)'",
        ),
        ("y ~ a*(b + 2c)", "expected identifier, got '2c'"),
        ("y ~ a*(b + c:)", "missing variable name in term 'a*(b+c:)'"),
    ] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(msg.contains(needle), "{f}: {msg}");
    }
}

// ── Random-effect terms the grammar refuses ─────────────────────────────────

/// A `( … | g)` written right after a name would otherwise parse as that name
/// plus the term: `zerocor(1 + x | g)` would become a fixed term `zerocor` and
/// a correlated slope.
#[test]
fn a_random_effect_term_after_a_name_is_a_syntax_error() {
    let msg = |term: &str| {
        format!("'{term}': a random-effect term `( … | g)` must stand alone, not after a name")
    };
    for (f, term) in [
        ("y ~ x + foo(1 | g)", "foo(1|g)"),
        ("y ~ x + foo(x | g)", "foo(x|g)"),
        ("y ~ x + foo(0 + x | g)", "foo(0+x|g)"),
        ("y ~ x + foo(1 + x || g)", "foo(1+x||g)"),
        ("y ~ x + foo(1 | a/b)", "foo(1|a/b)"),
        ("y ~ x + foo(1 | a:b)", "foo(1|a:b)"),
        ("y ~ x*foo(1 + x | g)", "foo(1+x|g)"),
        ("y ~ x + zerocor(1 + x | g)", "zerocor(1+x|g)"),
        ("y ~ zerocor(1 + x | g)", "zerocor(1+x|g)"),
    ] {
        assert_eq!(parse(f), syntax_error(0, &msg(term)), "{f}");
    }
    for f in [
        "y ~ x + zerocorr(1 + x | g)",
        "y ~ zerocorr(1 + x | g) + x",
        "y ~ log(x) + (1 | g)",
        "y ~ (1 | g) - 1",
    ] {
        assert!(parse(f).is_ok(), "{f}: {:?}", parse(f));
    }
}

/// A piece's term count and nesting depth are bounded: `*` and `^` double the
/// terms per variable, and each `(` is one level of recursion.
#[test]
fn huge_expansions_and_deep_nesting_are_syntax_errors() {
    let vars: Vec<String> = (0..16).map(|i| format!("x{i}")).collect();
    for f in [
        format!("y ~ {}", vars.join("*")),
        format!("y ~ ({})^16", vars.join("+")),
        format!("y ~ ({})", vars.join("*")),
    ] {
        let msg = parse(&f).expect_err(&f).to_string();
        assert!(
            msg.contains("expands to more than 10000 terms"),
            "{f}: {msg}"
        );
    }
    let deep = format!("y ~ {}x{}", "(".repeat(20_000), ")".repeat(20_000));
    let msg = parse(&deep).expect_err("deep").to_string();
    assert!(msg.contains("more than 200 nested parentheses"), "{msg}");
    let ok = format!("y ~ {}x{}", "(".repeat(100), ")".repeat(100));
    assert!(parse(&ok).is_ok(), "{:?}", parse(&ok));
}

/// Nor glued to the term after it: `(1|g)x` would parse as `x + (1|g)`.
#[test]
fn a_random_effect_term_before_a_term_without_a_plus_is_a_syntax_error() {
    for f in [
        "y ~ (1 | g)x",
        "y ~ x + (1 | g)(1 | h)",
        "y ~ x + (1 | g) x",
    ] {
        let msg = parse(f).expect_err(f).to_string();
        assert!(
            msg.contains("must stand alone, not before another term"),
            "{f}: {msg}"
        );
    }
}

/// A grouping is one factor, `a:b` or `a/b`; only a random intercept takes
/// the two-factor ones. The three-factor check runs first, so a slope on
/// three factors gets the three-factor message.
#[test]
fn a_slope_on_two_factors_or_a_grouping_of_three_is_a_syntax_error() {
    let slope = |term: &str, group: &str| {
        format!(
            "'{term}': a random slope on an interaction or nested grouping ('{group}') is not \
             supported; only a random intercept, `(1 | a:b)` or `(1 | a/b)`"
        )
    };
    for (f, term, group) in [
        ("y ~ x + (1 + x | a:b)", "(1+x|a:b)", "a:b"),
        ("y ~ x + (x | a:b)", "(x|a:b)", "a:b"),
        ("y ~ x + (0 + x | a:b)", "(0+x|a:b)", "a:b"),
        ("y ~ x + (1 + x | a/b)", "(1+x|a/b)", "a/b"),
        ("y ~ x + (x | a/b)", "(x|a/b)", "a/b"),
        ("y ~ x + (1 + log(x) | a:b)", "(1+log(x)|a:b)", "a:b"),
    ] {
        assert_eq!(parse(f), syntax_error(0, &slope(term, group)), "{f}");
    }
    let three =
        |term: &str| format!("'{term}': a grouping takes at most two factors (`a:b` or `a/b`)");
    for (f, term) in [
        ("y ~ x + (1 | a/b/c)", "(1|a/b/c)"),
        ("y ~ x + (1 | a:b:c)", "(1|a:b:c)"),
        ("y ~ x + (1 | a:b/c)", "(1|a:b/c)"),
        ("y ~ x + (1 + x | a/b/c)", "(1+x|a/b/c)"),
    ] {
        assert_eq!(parse(f), syntax_error(0, &three(term)), "{f}");
    }
    for f in ["y ~ x + (1 | a/b)", "y ~ x + (1 | a:b)"] {
        assert!(parse(f).is_ok(), "{f}: {:?}", parse(f));
    }
}
