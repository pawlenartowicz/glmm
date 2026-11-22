//! Tests of `predict` and `simulate` on a port's stored numbers.

use super::design::table_of;
use super::fit_tests::{csv_columns, run, table};
use super::*;
use crate::formula::lower;
use crate::{BinomialLink, Family, FitOptions, PoissonLink, PredictOptions, Scale, Simulator};
use std::collections::HashMap;

// ── predict and simulate on a port's stored numbers ───────────────────

/// What a port stores of `r` for `predict` and `simulate`.
pub(super) fn stored(r: &FitResult) -> StoredParts {
    StoredParts {
        beta: r.beta.clone(),
        vcov: r.vcov.clone(),
        model_vcov: r.model_vcov.clone(),
        varcorr: r.varcorr.clone(),
        ranef: r.ranef.clone(),
        ranef_levels: r.ranef_levels.clone(),
        condvar: r.condvar.clone(),
        condvar_x: r.condvar_x.clone(),
        condvar_cross: r.condvar_cross.clone(),
        dispersion: r.dispersion,
        weight_power: r.weight_power,
    }
}

fn bitwise(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `predict` and `simulate` on the numbers a port stored, the training
/// columns lowered again, give the kernel's `predict` and `Simulator` on
/// the `Fit` bit for bit: crossed (Penicillin) and nested (Pastes)
/// groupings, whose conditional standard error reads the covariance
/// across groupings, a correlated slope, a binomial GLMM and a Poisson
/// GLMM with an offset term.
#[test]
fn predict_and_simulate_on_stored_parts_match_the_kernel() {
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    // (CSV, factor columns, formula, family, and its port family and link)
    type Case = (
        &'static str,
        &'static [&'static str],
        &'static str,
        Family,
        &'static str,
        &'static str,
    );
    let cases: [Case; 5] = [
        (
            include_str!("../../../validation/data/empirical/Penicillin.csv"),
            &["plate", "sample"],
            "diameter ~ 1 + (1 | plate) + (1 | sample)",
            Family::Gaussian,
            "gaussian",
            "identity",
        ),
        (
            include_str!("../../../validation/data/empirical/Pastes.csv"),
            &["batch", "cask", "sample"],
            "strength ~ 1 + (1 | batch/cask)",
            Family::Gaussian,
            "gaussian",
            "identity",
        ),
        (
            include_str!("../../../validation/data/empirical/sleepstudy.csv"),
            &["Subject"],
            "Reaction ~ Days + (1 + Days | Subject)",
            Family::Gaussian,
            "gaussian",
            "identity",
        ),
        (
            include_str!("../../../validation/data/simulated/sim_factorial_bern.csv"),
            &["subj", "g", "h"],
            "y ~ x + g + (1 | subj)",
            binomial,
            "binomial",
            "logit",
        ),
        (
            include_str!("../../../validation/data/simulated/sim_poisson_offset.csv"),
            &["cluster"],
            "y ~ x + offset(log_exposure) + (1 | cluster)",
            poisson,
            "poisson",
            "log",
        ),
    ];
    let mut crossed = 0;
    for (csv, factors, formula, fam, family, link) in cases {
        let cols = csv_columns(csv, factors);
        let r = run(formula, cols.clone(), family, link, 1, None, None);
        let lo = lower(formula, &table(&cols), fam).expect("lowers");
        let opts = FitOptions {
            condvar: true,
            ..FitOptions::default()
        };
        let fit = crate::fit(&lo.data(), lo.spec(), &opts).expect("fits");
        assert!(fit.converged(), "{formula}");
        let parts = stored(&r);
        let future = fam != Family::Gaussian;
        let ours = |parts: &StoredParts, population: bool| {
            predict(
                formula,
                cols.0.clone(),
                cols.1.clone(),
                family_from_str(family, link).unwrap(),
                None,
                None,
                parts,
                None,
                Scale::Response,
                population,
                true,
                "allow_new_levels=True",
                true,
                0.9,
                future,
            )
            .unwrap_or_else(|e| panic!("{formula}: {e}"))
        };
        for population in [false, true] {
            let opts = PredictOptions {
                population,
                scale: Scale::Response,
                se: true,
                level: 0.9,
                future,
                ..PredictOptions::default()
            };
            let want = crate::predict(&fit, lo.spec(), &lo.data().design, &opts).unwrap();
            let got = ours(&parts, population);
            for (what, g, w) in [
                ("fit", &got.fit, &want.fit),
                ("se", &got.se, &want.se),
                ("lower", &got.lower, &want.lower),
                ("upper", &got.upper, &want.upper),
                ("future_lower", &got.future_lower, &want.future_lower),
                ("future_upper", &got.future_upper, &want.future_upper),
            ] {
                assert_eq!(bitwise(g), bitwise(w), "{formula} {what} pop={population}");
            }
            assert!(got.notes.is_empty(), "{formula}");
        }
        // The covariance across groupings moves the conditional SE, so a
        // port that dropped it would be seen here.
        if r.condvar_cross.is_some() {
            crossed += 1;
            let dropped = StoredParts {
                condvar_cross: None,
                ..parts.clone()
            };
            let (a, b) = (ours(&parts, false).se, ours(&dropped, false).se);
            assert!(
                a.iter().zip(&b).any(|(a, b)| (a - b).abs() > 1e-6 * a),
                "{formula}"
            );
        }
        let sim = Simulator::new(&fit, lo.spec(), &lo.design()).unwrap();
        let draws = simulate(
            formula,
            cols.0.clone(),
            cols.1.clone(),
            family_from_str(family, link).unwrap(),
            None,
            None,
            &parts,
            3,
            7,
        )
        .unwrap();
        assert_eq!(draws.len(), 3 * lo.n, "{formula}");
        for (d, got) in draws.chunks(lo.n).enumerate() {
            assert_eq!(
                bitwise(got),
                bitwise(&sim.simulate(7, d as u64)),
                "{formula} draw {d}"
            );
        }
    }
    assert_eq!(crossed, 2, "Penicillin and Pastes carry a cross record");
}

/// On a cell-means fit with an absent cell, the port path finds the
/// fit's null space from the relowered training table: the absent cell's
/// row is NaN and named in a note, a cell with data predicts a number.
#[test]
fn predict_names_the_rows_a_rank_deficient_fit_cannot_estimate() {
    let formula = "y ~ 0 + f1:f2 + (1 | g)";
    let cols = csv_columns(
        &crate::predict::predict_tests::absent_cell_csv(),
        &["f1", "f2", "g"],
    );
    let r = run(formula, cols.clone(), "gaussian", "identity", 1, None, None);
    let parts = stored(&r);
    let labels = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let numeric = HashMap::from([("y".to_string(), vec![0.0; 2])]);
    let factor = HashMap::from([
        ("f1".to_string(), (labels(&["a", "b"]), vec![0, 1])),
        ("f2".to_string(), (labels(&["A", "B"]), vec![0, 1])),
    ]);
    let got = predict(
        formula,
        cols.0,
        cols.1,
        Family::Gaussian,
        None,
        None,
        &parts,
        Some((numeric, factor, 2)),
        Scale::Link,
        true,
        true,
        "allow_new_levels=True",
        true,
        0.95,
        false,
    )
    .unwrap();
    assert!(
        got.fit[0].is_nan() && got.fit[1].is_finite(),
        "{:?}",
        got.fit
    );
    let notes: Vec<(&str, &[String], &[u32])> = got
        .notes
        .iter()
        .map(|n| (n.kind, &n.names[..], &n.rows[..]))
        .collect();
    assert_eq!(notes, [("rows_not_estimable", &[][..], &[0][..])]);
}

/// A new table: lowered with the fit's levels and grouping labels, an
/// unseen grouping label a note with its rows, the formula door's
/// prediction bit for bit; and the refusals the glue adds.
#[test]
fn predict_lowers_a_new_table_as_the_formula_door_does() {
    let csv = include_str!("../../../validation/data/empirical/Penicillin.csv");
    let formula = "diameter ~ 1 + (1 | plate) + (1 | sample)";
    let cols = csv_columns(csv, &["plate", "sample"]);
    let r = run(formula, cols.clone(), "gaussian", "identity", 1, None, None);
    let parts = stored(&r);
    // Three rows: a seen pair, an unseen plate, and both unseen, with
    // levels coded in another order than the fit's.
    let mut numeric = HashMap::new();
    numeric.insert("diameter".to_string(), vec![0.0; 3]);
    let mut factor = HashMap::new();
    let labels = |l: &[&str]| l.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    factor.insert("plate".to_string(), (labels(&["zz", "b"]), vec![1, 0, 0]));
    factor.insert(
        "sample".to_string(),
        (labels(&["F", "A", "Q"]), vec![0, 1, 2]),
    );
    let new_table = table_of(numeric.clone(), factor.clone()).unwrap();
    let model = crate::formula::fit(
        formula,
        &table(&cols),
        Family::Gaussian,
        &FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
    )
    .unwrap();
    let opts = PredictOptions {
        se: true,
        ..PredictOptions::default()
    };
    let want = crate::formula::predict(&model, &new_table, &opts).unwrap();
    let ours = |new, population, level| {
        predict(
            formula,
            cols.0.clone(),
            cols.1.clone(),
            Family::Gaussian,
            None,
            None,
            &parts,
            new,
            Scale::Link,
            population,
            true,
            "allow_new_levels=True",
            true,
            level,
            false,
        )
    };
    let got = ours(Some((numeric.clone(), factor.clone(), 3)), false, 0.95).unwrap();
    assert_eq!(bitwise(&got.fit), bitwise(&want.fit));
    assert_eq!(bitwise(&got.se), bitwise(&want.se));
    let notes: Vec<(&str, Vec<&str>, &[u32])> = got
        .notes
        .iter()
        .map(|n| {
            (
                n.kind,
                n.names.iter().map(String::as_str).collect(),
                &n.rows[..],
            )
        })
        .collect();
    assert_eq!(
        notes,
        [
            ("new_grouping_levels", vec!["plate"], &[1, 2][..]),
            ("new_grouping_levels", vec!["sample"], &[2][..])
        ]
    );
    // No grouping column: population mode.
    let fixed_only = Some((numeric.clone(), HashMap::new(), 3));
    assert!(ours(fixed_only, true, 0.95).is_ok());
    // An `offset=` argument has no values for new rows.
    let off = Some(vec![0.0; r.nobs]);
    let r_off = run(
        formula,
        cols.clone(),
        "gaussian",
        "identity",
        1,
        off.clone(),
        None,
    );
    let err = predict(
        formula,
        cols.0.clone(),
        cols.1.clone(),
        Family::Gaussian,
        None,
        off,
        &stored(&r_off),
        Some((numeric, factor, 3)),
        Scale::Link,
        false,
        true,
        "allow_new_levels=True",
        false,
        0.95,
        false,
    )
    .unwrap_err();
    assert!(err.contains("offset= argument"), "{err}");
    for nsim in [0, MAX_NSIM + 1] {
        assert!(simulate(
            formula,
            cols.0.clone(),
            cols.1.clone(),
            Family::Gaussian,
            None,
            None,
            &parts,
            nsim,
            1
        )
        .unwrap_err()
        .contains("nsim must be a whole number from 1 to"));
    }
}

#[test]
fn predict_columns_leave_out_the_response() {
    let available: Vec<String> = ["y", "x", "g", "o", "w"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let used = predict_columns("y ~ x + offset(log(o)) + (1 | g)", &available).unwrap();
    assert_eq!(used, ["g", "o", "x"]);
    assert!(predict_columns("y ~ x + (1 | g", &available).is_err());
}
