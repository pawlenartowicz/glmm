#![cfg(feature = "formula")]
//! Proves the mirrored parser is faithful to MCPower's. Two corpora, both copied
//! from the MCPower repo (d3 §6a):
//!   - the 28-case canonical suite (`configs/formula-fixtures/canonical-suite.json`),
//!     inlined here as Rust data and checked via the same canonical normalization
//!     the app-spec `formula_suite.rs` harness uses;
//!   - the 11 random-effects cases (`random_effects_parse.rs`) as direct AST asserts.
//!
//! Pure — no data table.

use glmm::formula::{parse, ParsedFormula, RandomEffect, Term};

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
    (
        "016_intercept_and_slope_same_group",
        "y ~ x + (1|g) + (1+x|g)",
        "duplicate grouping variable",
    ),
    ("err_empty", "", "formula is empty"),
    ("err_bad_identifier", "y ~ 1x", "formula syntax error"),
    (
        "err_duplicate_group",
        "y ~ x + (1|g) + (1|g)",
        "duplicate grouping variable",
    ),
    ("err_term_removal", "y ~ x1 - x2", "term removal"),
    (
        "err_intercept_suppression_zero",
        "y ~ x + (0+x|g)",
        "intercept suppression",
    ),
    (
        "err_intercept_suppression_minus_one",
        "y ~ x + (-1+x|g)",
        "intercept suppression",
    ),
];

#[test]
fn canonical_suite_matches_mirror() {
    for &(id, formula, dep, fixed, re) in SUCCESS {
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

// ── The 11 random-effects cases (random_effects_parse.rs), as direct AST asserts ──

#[test]
fn parses_random_intercept() {
    let f = parse("y ~ x + (1|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![RandomEffect::Intercept {
            group: "g".into(),
            parent: None,
        }]
    );
    assert_eq!(f.predictors, vec!["x".to_string()]); // group var is NOT a predictor
}

#[test]
fn parses_random_slope_single_var() {
    let f = parse("y ~ x + (1+x|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()],
        }]
    );
}

#[test]
fn parses_random_slope_multi_var() {
    let f = parse("y ~ x + z + (1+x+z|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into(), "z".into()],
        }]
    );
}

#[test]
fn parses_nested_intercept_expands_to_pair() {
    let f = parse("y ~ x + (1|A/B)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![
            RandomEffect::Intercept {
                group: "A".into(),
                parent: None,
            },
            RandomEffect::Intercept {
                group: "A:B".into(),
                parent: Some("A".into()),
            },
        ]
    );
}

#[test]
fn parses_interaction_grouping() {
    let f = parse("y ~ x + (1|recipe:replicate)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![RandomEffect::Intercept {
            group: "recipe:replicate".into(),
            parent: None,
        }]
    );
}

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

#[test]
fn duplicate_interaction_grouping_var_errors() {
    let err = parse("y ~ x + (1|recipe:replicate) + (1|recipe:replicate)").unwrap_err();
    assert!(matches!(
        err,
        glmm::formula::ParseError::DuplicateGroupingVar { .. }
    ));
}

#[test]
fn duplicate_grouping_var_errors() {
    let err = parse("y ~ x + (1|g) + (1|g)").unwrap_err();
    assert!(matches!(
        err,
        glmm::formula::ParseError::DuplicateGroupingVar { .. }
    ));
}

#[test]
fn rhs_after_re_extraction_has_clean_plusses() {
    let f = parse("y ~ x + (1|g)").unwrap();
    assert_eq!(f.terms.len(), 1); // only "x" — RE term doesn't appear in `terms`
    assert_eq!(f.terms[0], Term::Main { name: "x".into() });
}

#[test]
fn intercept_suppression_rejected() {
    for f in [
        "y ~ x + (0+x|g)",
        "y ~ x + (-1+x|g)",
        "y ~ x + (0|g)",
        "y ~ x + (-1|g)",
    ] {
        assert!(
            matches!(
                parse(f),
                Err(glmm::formula::ParseError::RandomInterceptSuppressionUnsupported)
            ),
            "expected suppression error for {f}, got {:?}",
            parse(f)
        );
    }
}

#[test]
fn implicit_intercept_slope_equals_explicit() {
    let imp = parse("y ~ x + (x|g)").unwrap();
    let exp = parse("y ~ x + (1+x|g)").unwrap();
    assert_eq!(imp.random_effects, exp.random_effects);
    assert_eq!(
        imp.random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()]
        }]
    );
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

#[test]
fn seam_match_from_embedded_re_term_still_parses() {
    // Degenerate input: an RE term embedded in another. The nested stage
    // removes `(1|a/b)` and the slope stage then matches across the seam
    // (`(1+x+|g)`), whose text does not occur in the original RHS — the
    // position lookup must fall back (sort to the end), not panic.
    let f = parse("y ~ (1+x+(1|a/b)|g)").unwrap();
    assert_eq!(
        f.random_effects,
        vec![
            RandomEffect::Intercept {
                group: "a".into(),
                parent: None,
            },
            RandomEffect::Intercept {
                group: "a:b".into(),
                parent: Some("a".into()),
            },
            RandomEffect::Slope {
                group: "g".into(),
                vars: vec!["x".into()],
            },
        ]
    );
}

#[test]
fn explicit_intercept_forms_unchanged() {
    assert_eq!(
        parse("y ~ (1|g)").unwrap().random_effects,
        vec![RandomEffect::Intercept {
            group: "g".into(),
            parent: None
        }]
    );
    assert_eq!(
        parse("y ~ x + (1+x|g)").unwrap().random_effects,
        vec![RandomEffect::Slope {
            group: "g".into(),
            vars: vec!["x".into()]
        }]
    );
}
