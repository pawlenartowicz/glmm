#![cfg(feature = "formula")]
//! materialize's fixed design vs R `model.matrix` (d3 §6b). The fixtures are
//! frozen R output (`fixtures/contrasts_fixtures.rs`, regenerate with
//! `gen_contrasts_fixtures.R`); the oracle is sacred — a mismatch is a
//! formula-frontend bug, never a relaxed fixture. Interactions are checked in the
//! marginal-present (treatment-dummy) form only; see the generator header.

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
    for fx in FIXTURES {
        let table = if fx.formula.contains("x1") {
            dat3()
        } else {
            dat()
        };
        let lo = lower(fx.formula, &table, Family::Gaussian)
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
