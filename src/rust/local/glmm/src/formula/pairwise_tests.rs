//! `pairwise` and its intervals where the frozen emmeans references do not
//! reach: the adjustments' NaN rule, the Tukey p and the interval of each
//! adjustment on every df route, a transformed covariate, an empty cell, a
//! fit without df, a numeric `by`, the level check and the refusals.

use super::{adjust_p, pairwise, Adjust, By, Pairwise, PairwiseIntervals, FACTOR, VALUES};
use crate::formula::{fit, Column, Error, Model, Table};
use crate::inference::contrast::NoDfReason;
use crate::inference::tails::{normal_quantile, t_quantile, tukey_quantile, tukey_upper_p};
use crate::inference::LinearNote;
use crate::test_support::csv_table;
use crate::{BinomialLink, Family, FitError, FitOptions};

const FACTORIAL: &str = include_str!("../../validation/data/simulated/sim_factorial.csv");
const FACTORIAL_EMPTY: &str =
    include_str!("../../validation/data/simulated/sim_factorial_empty.csv");
const FACTORIAL_BERN: &str = include_str!("../../validation/data/simulated/sim_factorial_bern.csv");

fn model(table: &Table, formula: &str, opts: &FitOptions) -> Model {
    let m = fit(formula, table, Family::Gaussian, opts).unwrap();
    assert!(m.fit().converged(), "{formula}: not converged");
    m
}

fn coef(m: &Model, name: &str) -> f64 {
    let lo = m.lowered();
    let j = lo.col_names.iter().position(|c| c == name).unwrap();
    m.fit().beta[j]
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * b.abs().max(1.0)
}

/// R's `p.adjust`: `[0.01, 0.04, 0.03]` gives Holm `[0.03, 0.06, 0.06]` and
/// Bonferroni `[0.03, 0.12, 0.09]`; a NaN stays NaN and is not counted.
#[test]
fn holm_and_bonferroni_follow_p_adjust() {
    let check = |p: &[f64], adjust, want: &[f64]| {
        let got = adjust_p(p, adjust);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!(
                (g.is_nan() && w.is_nan()) || (g - w).abs() <= 1e-15,
                "{adjust:?} of {p:?}: {got:?}, want {want:?}"
            );
        }
    };
    check(&[0.01, 0.04, 0.03], Adjust::Holm, &[0.03, 0.06, 0.06]);
    check(&[0.01, 0.04, 0.03], Adjust::Bonferroni, &[0.03, 0.12, 0.09]);
    check(
        &[0.01, f64::NAN, 0.04],
        Adjust::Holm,
        &[0.02, f64::NAN, 0.04],
    );
    check(
        &[0.01, f64::NAN, 0.04],
        Adjust::Bonferroni,
        &[0.02, f64::NAN, 0.08],
    );
    check(&[0.4, 0.7], Adjust::Bonferroni, &[0.8, 1.0]);
    check(&[0.01, f64::NAN], Adjust::None, &[0.01, f64::NAN]);
}

/// One model per df route, each with whether its rows are z rows:
/// Satterthwaite's df on an LMM, `n − p` on OLS, and a binomial GLMM's z
/// rows, all `~ g * h + x` on `sim_factorial`.
fn df_routes() -> [(Model, bool); 3] {
    let table = csv_table(FACTORIAL, &["subj", "g", "h"]);
    let bern = csv_table(FACTORIAL_BERN, &["subj", "g", "h"]);
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let glmm = fit(
        "y ~ g * h + x + (1 | subj)",
        &bern,
        logit,
        &FitOptions::default(),
    )
    .unwrap();
    assert!(glmm.fit().converged());
    [
        (
            model(&table, "y ~ g * h + x + (1 | subj)", &FitOptions::default()),
            false,
        ),
        (
            model(&table, "y ~ g * h + x", &FitOptions::default()),
            false,
        ),
        (glmm, true),
    ]
}

/// Each row's Tukey p is the studentized range of the factor's three levels
/// at `√2·|t|` on the row's own df, on every df route.
#[test]
fn tukey_row() {
    for (m, z) in &df_routes() {
        let pw = pairwise(m, "g", None, Adjust::Tukey).unwrap();
        assert_eq!(pw.rows.len(), 3);
        for row in &pw.rows {
            assert_eq!(row.df == f64::INFINITY, *z, "{row:?}");
            let q = std::f64::consts::SQRT_2 * row.statistic.abs();
            let want = tukey_upper_p(q, 3.0, row.df);
            assert!(want.is_finite() && want >= row.p_value, "{row:?}");
            assert_eq!(row.p_adjusted, want, "{row:?}");
        }
    }
}

/// `y ~ g * log(x)`: the grid holds `x` at its mean, so a cell reads
/// `log(x̄)`, not the mean of `log(x)`.
#[test]
fn transformed_covariate_uses_the_mean_of_the_variable() {
    let mut table = csv_table(FACTORIAL, &["subj", "g", "h"]);
    let x = match table.columns.iter_mut().find(|(n, _)| n == "x") {
        Some((_, Column::Numeric(v))) => v,
        _ => panic!("x is a numeric column"),
    };
    x.iter_mut().for_each(|v| *v = v.exp());
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    let mean_log = x.iter().map(|v| v.ln()).sum::<f64>() / x.len() as f64;
    let m = model(&table, "y ~ g * log(x)", &FitOptions::default());
    assert_eq!(m.lowered().numeric_means, vec![("x".to_string(), mean)]);

    let pw = pairwise(&m, "g", None, Adjust::None).unwrap();
    let row = &pw.rows[0];
    assert_eq!(row.contrast, "a - b");
    let at = |v: f64| -(coef(&m, "gb") + coef(&m, "gb:log(x)") * v);
    assert!(
        close(row.estimate, at(mean.ln())),
        "{} vs {}",
        row.estimate,
        at(mean.ln())
    );
    assert!(!close(row.estimate, at(mean_log)));
}

/// An empty cell (`g = c`, `h = q`) of `g * h`: the comparisons that read
/// its aliased column are NaN with the note, and each family is adjusted over
/// its tested rows only.
#[test]
fn aliased_cell() {
    let table = csv_table(FACTORIAL_EMPTY, &["subj", "g", "h"]);
    let m = model(&table, "y ~ g * h + (1 | subj)", &FitOptions::default());
    let column = m
        .lowered()
        .col_names
        .iter()
        .position(|c| c == "gc:hq")
        .unwrap();
    assert!(m.fit().aliased()[column]);
    let pw = pairwise(&m, "g", Some(By::Levels("h")), Adjust::Bonferroni).unwrap();
    assert_eq!(pw.rows.len(), 6);
    for row in &pw.rows {
        let by = row.by_level.as_deref().unwrap();
        let touches = by == "q" && row.contrast.ends_with("- c");
        if touches {
            assert!(row.estimate.is_nan() && row.se.is_nan() && row.p_adjusted.is_nan());
            assert_eq!(row.notes, vec![LinearNote::AliasedColumn { column }]);
        } else {
            assert!(row.p_value.is_finite() && row.df.is_finite(), "{row:?}");
            assert!(row.notes.is_empty());
            let m = if by == "q" { 1.0 } else { 3.0 };
            assert_eq!(row.p_adjusted, (m * row.p_value).min(1.0), "{row:?}");
        }
    }
}

/// A Gaussian LMM fitted without Satterthwaite df gives NaN df, statistic
/// and p-values with the note, and a numeric `by` with chosen values gives
/// one family per value, each with the variable at that value.
#[test]
fn no_df_lmm_and_by_values() {
    let table = csv_table(FACTORIAL, &["subj", "g", "h"]);
    let opts = FitOptions {
        satterthwaite: false,
        ..FitOptions::default()
    };
    let m = model(&table, "y ~ g * x + (1 | subj)", &opts);
    let pw = pairwise(&m, "g", Some(By::Values("x", &[1.0, 2.0])), Adjust::Holm).unwrap();
    assert_eq!(pw.by.as_deref(), Some("x"));
    let by: Vec<&str> = pw
        .rows
        .iter()
        .map(|r| r.by_level.as_deref().unwrap())
        .collect();
    assert_eq!(by, ["1", "1", "1", "2", "2", "2"]);
    for row in &pw.rows {
        assert!(row.estimate.is_finite() && row.se.is_finite(), "{row:?}");
        assert!(row.df.is_nan() && row.statistic.is_nan(), "{row:?}");
        assert!(row.p_value.is_nan() && row.p_adjusted.is_nan(), "{row:?}");
        assert_eq!(
            row.notes,
            vec![LinearNote::NoDf {
                reason: NoDfReason::Disabled
            }]
        );
    }
    for (row, v) in [(&pw.rows[0], 1.0), (&pw.rows[3], 2.0)] {
        assert_eq!(row.contrast, "a - b");
        let want = -(coef(&m, "gb") + coef(&m, "gb:x") * v);
        assert!(close(row.estimate, want), "{} vs {want}", row.estimate);
    }
}

/// A `factor` or `by` outside the fixed part, a numeric `factor`, a factor
/// `by` with values, a numeric `by` without them, and a `by` that is the
/// factor.
#[test]
fn refusals() {
    let table = csv_table(FACTORIAL, &["subj", "g", "h"]);
    let m = model(&table, "y ~ g * h + x + (1 | subj)", &FitOptions::default());
    let kind = |r: Result<Pairwise, Error>, name: &str, expected: &str| match r {
        Err(Error::WrongColumnKind {
            name: n,
            expected: e,
        }) => n == name && e == expected,
        _ => false,
    };
    let cases = [
        matches!(
            pairwise(&m, "subj", None, Adjust::Holm),
            Err(Error::UnknownColumn(n)) if n == "subj"
        ),
        matches!(
            pairwise(&m, "g", Some(By::Levels("nope")), Adjust::Holm),
            Err(Error::UnknownColumn(n)) if n == "nope"
        ),
        kind(pairwise(&m, "x", None, Adjust::Holm), "x", FACTOR),
        kind(
            pairwise(&m, "g", Some(By::Values("h", &[1.0])), Adjust::Holm),
            "h",
            VALUES,
        ),
        kind(
            pairwise(&m, "g", Some(By::Levels("x")), Adjust::Holm),
            "x",
            FACTOR,
        ),
        matches!(
            pairwise(&m, "g", Some(By::Levels("g")), Adjust::Holm),
            Err(Error::Fit(FitError::OptionValue(_)))
        ),
    ];
    assert_eq!(cases, [true; 6]);
}

/// The interval of each adjustment on every df route is `estimate ± c·se`:
/// `c` the t quantile at `1 − α/2` on the row's df (normal on z rows) without
/// adjustment, at `1 − α/(2m)` with Bonferroni, and the studentized-range
/// quantile at `α` over `√2` with Tukey; on the logit link the odds-ratio
/// bounds are `exp` of the bounds.
#[test]
fn confint_follows_the_adjustment() {
    let alpha = 1.0 - 0.95;
    for (m, z) in &df_routes() {
        let two_sided = |a: f64, df: f64| {
            let p = 1.0 - a / 2.0;
            if *z {
                normal_quantile(p)
            } else {
                t_quantile(p, df)
            }
        };
        for (adjust, crit) in [
            (
                Adjust::None,
                &(|df| two_sided(alpha, df)) as &dyn Fn(f64) -> f64,
            ),
            (Adjust::Bonferroni, &|df| two_sided(alpha / 3.0, df)),
            (Adjust::Tukey, &|df| {
                tukey_quantile(alpha, 3.0, df) / std::f64::consts::SQRT_2
            }),
        ] {
            let pw = pairwise(m, "g", None, adjust).unwrap();
            let ci = pw.confint(0.95).unwrap();
            assert_eq!((ci.level, ci.adjust), (0.95, adjust));
            assert_eq!(ci.rows.len(), 3);
            for (row, iv) in pw.rows.iter().zip(&ci.rows) {
                let c = crit(row.df);
                assert!(c.is_finite() && c > 0.0, "{adjust:?}: {row:?}");
                assert_eq!(iv.lower, row.estimate - c * row.se, "{adjust:?}: {row:?}");
                assert_eq!(iv.upper, row.estimate + c * row.se, "{adjust:?}: {row:?}");
                if *z {
                    assert_eq!(
                        (iv.ratio_lower, iv.ratio_upper),
                        (iv.lower.exp(), iv.upper.exp())
                    );
                } else {
                    assert!(iv.ratio_lower.is_nan() && iv.ratio_upper.is_nan());
                }
            }
        }
    }
}

/// Holm has no simultaneous interval: a Holm result gets the Bonferroni
/// intervals of each `by` family, and says so.
#[test]
fn holm_gives_bonferroni_intervals() {
    let table = csv_table(FACTORIAL, &["subj", "g", "h"]);
    let m = model(&table, "y ~ g * h + x", &FitOptions::default());
    let by = Some(By::Levels("h"));
    let holm = pairwise(&m, "g", by, Adjust::Holm)
        .unwrap()
        .confint(0.9)
        .unwrap();
    let bonferroni = pairwise(&m, "g", by, Adjust::Bonferroni)
        .unwrap()
        .confint(0.9)
        .unwrap();
    assert_eq!((holm.level, holm.adjust), (0.9, Adjust::Bonferroni));
    assert_eq!(holm.rows.len(), 6);
    let bounds = |ci: &PairwiseIntervals| -> Vec<(f64, f64)> {
        ci.rows.iter().map(|iv| (iv.lower, iv.upper)).collect()
    };
    assert_eq!(bounds(&holm), bounds(&bonferroni));
}

/// An empty cell's comparisons get NaN intervals and leave the Bonferroni
/// family size alone: the one tested row of its family gets the unadjusted
/// interval. A `level` of 0, 1 or NaN is refused.
#[test]
fn nan_rows_and_level() {
    let table = csv_table(FACTORIAL_EMPTY, &["subj", "g", "h"]);
    let m = model(&table, "y ~ g * h + (1 | subj)", &FitOptions::default());
    let by = Some(By::Levels("h"));
    let pw = pairwise(&m, "g", by, Adjust::Bonferroni).unwrap();
    let ci = pw.confint(0.95).unwrap();
    let none = pairwise(&m, "g", by, Adjust::None)
        .unwrap()
        .confint(0.95)
        .unwrap();
    let mut nan = 0;
    for ((row, iv), plain) in pw.rows.iter().zip(&ci.rows).zip(&none.rows) {
        if row.p_value.is_nan() {
            assert!(iv.lower.is_nan() && iv.upper.is_nan(), "{row:?}");
            nan += 1;
        } else if row.by_level.as_deref() == Some("q") {
            assert_eq!((iv.lower, iv.upper), (plain.lower, plain.upper), "{row:?}");
        } else {
            assert!(iv.lower < plain.lower && iv.upper > plain.upper, "{row:?}");
        }
    }
    assert_eq!(nan, 2);
    for level in [0.0, 1.0, f64::NAN] {
        assert!(
            matches!(pw.confint(level), Err(FitError::OptionValue(_))),
            "level {level}"
        );
    }
}
