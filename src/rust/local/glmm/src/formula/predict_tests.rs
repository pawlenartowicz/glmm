//! `predict` on the formula door: a new table lowered with the fit's levels,
//! columns, random-effect-only columns, offset and transforms, against the
//! matrix door on the same rows.

use super::materialize::lower_new;
use super::{fit, label_ranef, lower, predict, Column, Error, Model, Table};
use crate::predict::predict_tests::SLEEP;
use crate::test_support::csv_table;
use crate::{Family, FitError, FitOptions, GroupIds, Note, PoissonLink, PredictOptions, Scale};

const FACTORIAL: &str = include_str!("../../validation/data/simulated/sim_factorial.csv");
const PASTES: &str = include_str!("../../validation/data/empirical/Pastes.csv");
const POISSON_OFFSET: &str = include_str!("../../validation/data/simulated/sim_poisson_offset.csv");

fn model(csv: &str, factors: &[&str], formula: &str, family: Family) -> Model {
    let m = fit(
        formula,
        &csv_table(csv, factors),
        family,
        &FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
    )
    .unwrap();
    assert!(m.fit().converged(), "{formula}: not converged");
    m
}

/// The CSV rows whose `keep` holds, as a table: the factors' levels are the
/// labels those rows hold, so a dropped label is not a level.
fn csv_rows(csv: &str, factors: &[&str], keep: impl Fn(&str) -> bool) -> Table {
    let mut lines = csv.lines();
    let header = lines.next().unwrap();
    let kept: Vec<&str> = lines.filter(|l| keep(l)).collect();
    csv_table(&format!("{header}\n{}", kept.join("\n")), factors)
}

fn factor_mut<'a>(table: &'a mut Table, name: &str) -> (&'a mut Vec<String>, &'a mut Vec<u32>) {
    match table.columns.iter_mut().find(|(n, _)| n == name) {
        Some((_, Column::Factor { levels, codes })) => (levels, codes),
        _ => panic!("{name} is not a factor column"),
    }
}

fn labels(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn new_table_levels_are_matched_by_label() {
    let formula = "y ~ x * g + h + (1 | subj)";
    let factors = ["subj", "g", "h"];
    let lo = lower(formula, &csv_table(FACTORIAL, &factors), Family::Gaussian).unwrap();
    let p = lo.p;

    // The same rows with `g`'s levels declared in reverse order.
    let mut reversed = csv_table(FACTORIAL, &factors);
    let (levels, codes) = factor_mut(&mut reversed, "g");
    let k = levels.len() as u32;
    levels.reverse();
    codes.iter_mut().for_each(|c| *c = k - 1 - *c);
    let rows = lower_new(formula, &lo, &reversed).unwrap();
    assert_eq!(rows.x, lo.x, "reversed levels");

    // A subset of the levels: the rows without `g = a`, and the rows with
    // `g = b` alone (a one-level factor in the new table).
    let column = |l: &str| l.split(',').nth(1).unwrap().to_string();
    for subset in [&["\"b\"", "\"c\""][..], &["\"b\""][..]] {
        let table = csv_rows(FACTORIAL, &factors, |l| {
            subset.contains(&column(l).as_str())
        });
        let rows = lower_new(formula, &lo, &table).unwrap();
        let mut expected = Vec::new();
        for (i, line) in FACTORIAL.lines().skip(1).enumerate() {
            if subset.contains(&column(line).as_str()) {
                expected.extend_from_slice(&lo.x[i * p..(i + 1) * p]);
            }
        }
        assert_eq!(rows.n, table.n);
        assert_eq!(rows.x, expected, "subset {subset:?}");
    }

    // An unseen level of a fixed factor is refused by column, level and row.
    let mut unseen = csv_table(FACTORIAL, &factors);
    let (levels, codes) = factor_mut(&mut unseen, "g");
    levels.push("z".to_string());
    codes[5] = levels.len() as u32 - 1;
    match lower_new(formula, &lo, &unseen) {
        Err(Error::Fit(FitError::Shape(m))) => {
            assert!(
                m.contains("\"g\"") && m.contains("\"z\"") && m.contains("row 5"),
                "{m}"
            );
        }
        other => panic!(
            "expected Error::Fit(FitError::Shape(..)), got {:?}",
            other.err()
        ),
    }

    // So is a level the fit declared but had no row of: its column is aliased,
    // and reading it as 0 would predict the base level.
    let mut declared = csv_table(FACTORIAL, &factors);
    factor_mut(&mut declared, "g").0.push("z".to_string());
    let lo_declared = lower(formula, &declared, Family::Gaussian).unwrap();
    match lower_new(formula, &lo_declared, &unseen) {
        Err(Error::Fit(FitError::Shape(m))) => {
            assert!(m.contains("\"z\"") && m.contains("row 5"), "{m}");
        }
        other => panic!(
            "expected Error::Fit(FitError::Shape(..)), got {:?}",
            other.err()
        ),
    }

    // A joined label that names two of the fit's levels, ("a:b", "c") and
    // ("a", "b:c"), is refused rather than matched to either.
    let ab = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            (
                "A".into(),
                Column::factor_from_labels(&labels(&["a:b", "a:b", "a", "a"])),
            ),
            (
                "B".into(),
                Column::factor_from_labels(&labels(&["c", "c", "b:c", "b:c"])),
            ),
        ],
        n: 4,
    };
    let lo_ab = lower("y ~ 1 + (1 | A:B)", &ab, Family::Gaussian).unwrap();
    match lower_new("y ~ 1 + (1 | A:B)", &lo_ab, &ab) {
        Err(Error::Fit(FitError::Shape(m))) => {
            assert!(m.contains("\"a:b:c\"") && m.contains("row 0"), "{m}");
        }
        other => panic!(
            "expected Error::Fit(FitError::Shape(..)), got {:?}",
            other.err()
        ),
    }

    // Pastes, `(1 | batch/cask)`: a seen cask, a new cask in a seen batch, and
    // a new batch. The new table's levels are not the training ones.
    let m = model(
        PASTES,
        &["batch", "cask", "sample"],
        "strength ~ 1 + (1 | batch/cask)",
        Family::Gaussian,
    );
    let new = Table {
        columns: vec![
            (
                "batch".into(),
                Column::factor_from_labels(&labels(&["B", "B", "Z", "A"])),
            ),
            (
                "cask".into(),
                Column::factor_from_labels(&labels(&["b", "z", "a", "c"])),
            ),
        ],
        n: 4,
    };
    let pred = predict(&m, &new, &PredictOptions::default()).unwrap();
    let blocks = label_ranef(m.fit(), &m.lowered().re_groups).unwrap();
    let mode = |g: usize, level: &str| {
        let at = blocks[g].levels.iter().position(|l| l == level).unwrap();
        blocks[g].values[at]
    };
    let b0 = m.fit().beta[0];
    let expected = [
        b0 + mode(0, "B") + mode(1, "B:b"),
        b0 + mode(0, "B"),
        b0,
        b0 + mode(0, "A") + mode(1, "A:c"),
    ];
    for (i, (a, e)) in pred.fit.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= 1e-12 * (1.0 + e.abs()),
            "row {i}: {a} vs {e}"
        );
    }
    assert!(
        matches!(
            &pred.notes[..],
            [
                Note::NewGroupingLevels { grouping: 0, rows: r0 },
                Note::NewGroupingLevels { grouping: 1, rows: r1 },
            ] if r0 == &[2] && r1 == &[1, 2]
        ),
        "{:?}",
        pred.notes
    );
}

#[test]
fn re_only_slope_on_new_data() {
    let m = model(
        SLEEP,
        &["Subject"],
        "Reaction ~ 1 + (Days | Subject)",
        Family::Gaussian,
    );
    let lo = m.lowered();
    assert_eq!((lo.p, lo.re_p), (1, 1));
    // New days, three subjects in an order of their own, one unseen.
    let subjects = labels(&["372", "308", "999", "372", "308"]);
    let days = vec![0.5, 2.5, 4.0, 11.0, 7.25];
    let new = Table {
        columns: vec![
            ("Days".into(), Column::Numeric(days.clone())),
            ("Subject".into(), Column::factor_from_labels(&subjects)),
        ],
        n: 5,
    };
    let slots = &lo.re_groups[0].slot_labels;
    let ids = GroupIds::new(
        subjects
            .iter()
            .map(|s| {
                slots
                    .iter()
                    .position(|l| l.as_deref() == Some(s.as_str()))
                    .unwrap_or(slots.len()) as u32
            })
            .collect(),
    );
    let x = vec![1.0; 5];
    let design = crate::Design::new(&x, 5, 1).re_columns(&days, 1).ids(&ids);
    let opts = PredictOptions {
        se: true,
        ..PredictOptions::default()
    };
    let door = predict(&m, &new, &opts).unwrap();
    let matrix = crate::predict(m.fit(), lo.spec(), &design, &opts).unwrap();
    assert_eq!(door.fit, matrix.fit);
    assert_eq!(door.se, matrix.se);
    assert!(
        matches!(
            &door.notes[..],
            [Note::NewGroupingLevels { grouping: 0, rows }] if rows == &[2]
        ),
        "{:?}",
        door.notes
    );
    assert_eq!(format!("{:?}", door.notes), format!("{:?}", matrix.notes));
}

#[test]
fn offset_and_transforms_on_new_data() {
    let m = model(
        POISSON_OFFSET,
        &["cluster"],
        "y ~ x + I(x^2) + offset(log_exposure) + (1 | cluster)",
        Family::Poisson {
            link: PoissonLink::Log,
        },
    );
    let lo = m.lowered();
    assert_eq!(lo.col_names, ["(Intercept)", "x", "I(x^2)"]);
    let x = [-1.5, 0.25, 2.0];
    let exposure = [0.3, -0.7, 1.1];
    let new = Table {
        columns: vec![
            ("x".into(), Column::Numeric(x.to_vec())),
            ("log_exposure".into(), Column::Numeric(exposure.to_vec())),
            (
                "cluster".into(),
                Column::factor_from_labels(&labels(&["1", "2", "3"])),
            ),
        ],
        n: 3,
    };
    let rows = lower_new(m.formula(), lo, &new).unwrap();
    let expected_x: Vec<f64> = x.iter().flat_map(|&v| [1.0, v, v * v]).collect();
    assert_eq!(rows.x, expected_x);
    assert_eq!(rows.offset.as_deref(), Some(&exposure[..]));

    let opts = PredictOptions {
        population: true,
        ..PredictOptions::default()
    };
    let pred = predict(&m, &new, &opts).unwrap();
    let b = &m.fit().beta;
    for i in 0..3 {
        let e = b[0] + b[1] * x[i] + b[2] * x[i] * x[i] + exposure[i];
        let a = pred.fit[i];
        assert!(
            (a - e).abs() <= 1e-12 * (1.0 + e.abs()),
            "row {i}: {a} vs {e}"
        );
    }
}

#[test]
fn formula_predict_equals_the_matrix_door() {
    let m = model(
        SLEEP,
        &["Subject"],
        "Reaction ~ Days + (1 + Days | Subject)",
        Family::Gaussian,
    );
    let lo = m.lowered();
    let table = csv_table(SLEEP, &["Subject"]);
    let rows = lower_new(m.formula(), lo, &table).unwrap();
    assert_eq!(rows.x, lo.x);
    assert_eq!(rows.ids.as_ref(), Some(&lo.ids));
    let mut cases = 0;
    for population in [false, true] {
        for scale in [Scale::Link, Scale::Response] {
            let opts = PredictOptions {
                population,
                scale,
                se: true,
                ..PredictOptions::default()
            };
            let door = predict(&m, &table, &opts).unwrap();
            let matrix = crate::predict(m.fit(), lo.spec(), &lo.design(), &opts).unwrap();
            assert_eq!(door.fit, matrix.fit);
            assert_eq!(door.se, matrix.se);
            assert_eq!(door.lower, matrix.lower);
            assert_eq!(door.upper, matrix.upper);
            assert!(door.notes.is_empty());
            cases += 1;
        }
    }
    assert_eq!(cases, 4);
}

/// The CI copy of part of the frozen `predict_sleepstudy` reference (`oracle-tests`
/// compares all of it): three new rows' population and conditional link
/// values, two population standard errors and two levels' conditional SDs,
/// under the bands of `formula::predict_oracle_tests`. Values copied from
/// `validation/goldens/inference/predict/predict_sleepstudy.json`
/// (`goldens_predict.R`, lme4 2.0.6, R 4.5.3): `Reaction ~ 1 + Days +
/// (1 + Days | Subject)` by REML, its new-data rows 0, 3 and 4.
#[test]
fn predict_sleepstudy_matches_its_frozen_reference() {
    // Mirrors `SE_REL` in `formula::predict_oracle_tests`; change together.
    const SE_REL: f64 = 1e-3;
    // lme4's `fixef` and the `ranef` rows of subjects 309 and 308.
    let beta = [251.405104848485, 10.467285959596];
    let modes = [
        ("309", [-40.3987380788051, -8.61968061524321]),
        ("308", [2.25855094990116, 9.1989757655659]),
    ];
    let days = [0.0, 13.5, 0.0];
    let subjects = ["309", "308", "new1"];
    let population = [251.405104848485, 392.713465303031, 251.405104848485];
    let conditional = [211.00636676968, 519.158189088072, 251.405104848485];
    // `predict(se.fit = TRUE, re.form = NA)` at 0 and 13.5 days.
    let se_population = [6.8245966949549, 21.0446155258105];
    // `as.data.frame(ranef(m))$condsd`: subject 308's intercept, 309's slope.
    let condsd = [
        ("308", "(Intercept)", 12.0708569505143),
        ("309", "Days", 2.30483902090028),
    ];

    let m = model(
        SLEEP,
        &["Subject"],
        "Reaction ~ 1 + Days + (1 + Days | Subject)",
        Family::Gaussian,
    );
    let blocks = label_ranef(m.fit(), &m.lowered().re_groups).unwrap();
    let mode = |s: &str| {
        let at = blocks[0].levels.iter().position(|l| l == s).unwrap();
        &blocks[0].values[at * 2..at * 2 + 2]
    };
    let new = Table {
        columns: vec![
            ("Days".into(), Column::Numeric(days.to_vec())),
            (
                "Subject".into(),
                Column::factor_from_labels(&labels(&subjects)),
            ),
        ],
        n: 3,
    };
    let b = &m.fit().beta;
    for (population_mode, want) in [(true, population), (false, conditional)] {
        let opts = PredictOptions {
            population: population_mode,
            ..PredictOptions::default()
        };
        let got = predict(&m, &new, &opts).unwrap();
        for i in 0..3 {
            let z = [1.0, days[i]];
            // `Σ|x_j||Δβ_j| + Σ|z_k||Δb_k| + (p + q)·ε·Σ|term|` at lme4's
            // parameters; an unseen subject has no effect.
            let mut delta: f64 = (0..2).map(|j| (z[j] * (b[j] - beta[j])).abs()).sum();
            let mut sum: f64 = (0..2).map(|j| (z[j] * beta[j]).abs()).sum();
            if let (false, Some((_, lme4))) = (
                population_mode,
                modes.iter().find(|(s, _)| *s == subjects[i]),
            ) {
                let ours = mode(subjects[i]);
                for k in 0..2 {
                    delta += (z[k] * (ours[k] - lme4[k])).abs();
                    sum += (z[k] * lme4[k]).abs();
                }
            }
            let band = delta + 4.0 * f64::EPSILON * sum;
            let gap = (got.fit[i] - want[i]).abs();
            assert!(
                gap <= band,
                "population {population_mode} row {i}: {} against lme4 {} (gap {gap:e}, band {band:e})",
                got.fit[i],
                want[i]
            );
        }
    }
    let opts = PredictOptions {
        population: true,
        se: true,
        ..PredictOptions::default()
    };
    let got = predict(&m, &new, &opts).unwrap();
    for (i, want) in se_population.iter().enumerate() {
        let rel = (got.se[i] - want).abs() / want;
        assert!(
            rel <= SE_REL,
            "population SE row {i}: {} against lme4 {want}",
            got.se[i]
        );
    }
    for (subject, term, want) in condsd {
        let at = blocks[0].levels.iter().position(|l| l == subject).unwrap();
        // The vech of a 2 × 2 block is `(0,0), (1,0), (1,1)`.
        let diag = if term == "Days" { 2 } else { 0 };
        let got = blocks[0].condvar[at * 3 + diag].sqrt();
        let rel = (got - want).abs() / want;
        assert!(
            rel <= SE_REL,
            "condsd {subject} {term}: {got} against lme4 {want}"
        );
    }
}
