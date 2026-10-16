#![cfg(feature = "formula")]
//! materialize's fixed design vs R `model.matrix`. The fixtures are
//! frozen R output (`fixtures/contrasts_fixtures.rs`, regenerate with
//! `gen_contrasts_fixtures.R`); the oracle is sacred — a mismatch is a
//! formula-frontend bug, never a relaxed fixture.

use glmm::formula::{lower, Column, Table};
use glmm::Family;

include!("fixtures/contrasts_fixtures.rs");

/// The shared fixture frame — mirrors `dat` in the R generator verbatim.
fn dat() -> Table {
    Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(vec![0.1, 1.2, -0.3, 2.0, 0.7, -1.1]),
            ),
            (
                "x".into(),
                Column::Numeric(vec![1.0, 2.5, -1.0, 0.0, 3.0, 2.0]),
            ),
            (
                "z".into(),
                Column::Numeric(vec![0.5, -2.0, 1.5, 4.0, -0.5, 1.0]),
            ),
            (
                "f".into(),
                Column::factor_from_labels(&["a", "b", "c", "a", "b", "c"].map(String::from)),
            ),
            (
                "g".into(),
                Column::factor_from_labels(&["p", "q", "p", "q", "p", "q"].map(String::from)),
            ),
            (
                // Same labels as f with a declared non-lexicographic order —
                // mirrors R's factor(..., levels = c("c","a","b")); level 0
                // (here c) is the treatment-contrast reference.
                "h".into(),
                Column::Factor {
                    levels: vec!["c".into(), "a".into(), "b".into()],
                    codes: vec![1, 2, 0, 1, 2, 0],
                },
            ),
            (
                "w".into(),
                Column::Numeric(vec![1.5, 2.0, 0.5, 4.0, 3.0, 2.5]),
            ),
            (
                "xz".into(),
                Column::Numeric(vec![0.5, -5.0, -1.5, 0.0, -1.5, 2.0]),
            ),
            (
                "a".into(),
                Column::Numeric(vec![0.3, -1.0, 2.0, 1.5, 0.0, -0.7]),
            ),
            (
                "b".into(),
                Column::Numeric(vec![1.1, 0.4, -0.2, 2.2, -1.3, 0.9]),
            ),
            (
                "c".into(),
                Column::Numeric(vec![-0.5, 1.7, 0.8, -2.0, 1.0, 0.2]),
            ),
            (
                "d".into(),
                Column::Numeric(vec![2.5, -0.6, 0.1, 1.3, -1.8, 0.7]),
            ),
        ],
        n: 6,
    }
}

/// The numeric 3-way frame — mirrors `dat3` in the R generator.
fn dat3() -> Table {
    Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(vec![0.1, 1.2, -0.3, 2.0, 0.7, -1.1]),
            ),
            (
                "x1".into(),
                Column::Numeric(vec![1.0, 2.5, -1.0, 0.0, 3.0, 2.0]),
            ),
            (
                "x2".into(),
                Column::Numeric(vec![0.5, -2.0, 1.5, 4.0, -0.5, 1.0]),
            ),
            (
                "x3".into(),
                Column::Numeric(vec![2.0, 1.0, 0.0, -1.0, 0.5, 3.0]),
            ),
        ],
        n: 6,
    }
}

#[test]
fn fixed_design_matches_model_matrix() {
    // Pinned so a regenerated-empty fixture file (a missing R dependency, a
    // manifest path that moved) fails loudly instead of leaving this loop
    // running zero times and reporting green.
    assert_eq!(
        FIXTURES.len(),
        59,
        "the frozen contrast fixture set changed size"
    );
    for fx in FIXTURES {
        let table = if fx.formula.contains("x1") {
            dat3()
        } else {
            dat()
        };
        // The parser has no `-` term removal; `y ~ f:g + g` is the same model,
        // with `f` written first as in `f*g`, so the interaction is named `fb:gp`
        // as R names it.
        let formula = match fx.formula {
            "y ~ f*g - f" => "y ~ f:g + g",
            other => other,
        };
        let lo = lower(formula, &table, Family::Gaussian)
            .unwrap_or_else(|e| panic!("{}: lower failed: {e}", fx.formula));

        let want_names: Vec<String> = fx.names.iter().map(|s| s.to_string()).collect();
        assert_eq!(lo.col_names, want_names, "{} column names", fx.formula);
        assert_eq!(lo.p, fx.p, "{} width", fx.formula);
        assert_eq!(lo.n, fx.n, "{} rows", fx.formula);
        assert_eq!(lo.x.len(), fx.x.len(), "{} design length", fx.formula);
        for (k, (a, b)) in lo.x.iter().zip(fx.x).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "{}: design[{k}] = {a} but R has {b}",
                fx.formula
            );
        }
    }
}

#[test]
fn an_interaction_repeated_in_another_order_is_coded_once() {
    // R's `model.matrix` gives `y ~ f*g + g:f` exactly the design of
    // `y ~ f*g`, and `y ~ f:g + g:f` that of `y ~ f:g` (checked with
    // `identical()` in R 4.5.3): a term is its set of variables, so the
    // repeat adds no column.
    for (formula, same_as) in [("y ~ f*g + g:f", "y ~ f*g"), ("y ~ f:g + g:f", "y ~ f:g")] {
        let fx = FIXTURES.iter().find(|fx| fx.formula == same_as).unwrap();
        let lo = lower(formula, &dat(), Family::Gaussian).unwrap();
        assert_eq!(lo.col_names, fx.names, "{formula} column names");
        assert_eq!(lo.x, fx.x, "{formula} design");
    }
}
