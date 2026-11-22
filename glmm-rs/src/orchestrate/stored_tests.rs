//! Tests of the readers of a port's stored fit.

use super::fit::den_df_fields;
use super::fit_tests::{csv_columns, direct, run, sim, table, Columns};
use super::predict_tests::stored;
use super::*;
use crate::formula::anova::{linear_note_line, TermTable, TermTest};
use crate::formula::{Adjust, By, Column, PairwiseIntervals};
use crate::inference::contrast::NoDfReason;
use crate::inference::wald::wald_chisq;
use crate::inference::{DenDf, LinearNote, LinearTest};
use crate::{BinomialLink, Family, FitOptions, GammaLink, PoissonLink, SeType, WeightPower};

#[test]
fn confint_follows_the_df_method() {
    // qt(0.975, 8), qnorm(0.975) and qnorm(0.95) in R.
    let (qt8, qn, qn90) = (2.3060041352041654, 1.9599639845400534, 1.6448536269514715);
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
    let nan = |ci: [f64; 2]| ci[0].is_nan() && ci[1].is_nan();
    let beta = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let se = [0.5, 0.5, 0.5, 0.5, 0.5, f64::NAN];
    let df = [8.0, f64::NAN, 0.0, -1.0, f64::INFINITY, 8.0];
    for method in [
        "residual",
        "satterthwaite",
        "bell_mccaffrey",
        "pustejovsky_tipton",
    ] {
        let ci = confint(&beta, &se, &df, DfMethod::parse(method).unwrap(), 0.95).unwrap();
        assert!(close(ci[0][0], 1.0 - qt8 * 0.5) && close(ci[0][1], 1.0 + qt8 * 0.5));
        // A lost df on a fit that tests with t has no interval.
        for (i, c) in ci.iter().enumerate().skip(1) {
            assert!(nan(*c), "{method} row {i}: {c:?}");
        }
    }
    for method in ["none", "disabled", "failed"] {
        let method = DfMethod::parse(method).unwrap();
        let ci = confint(&beta, &se, &df, method, 0.95).unwrap();
        assert!(close(ci[0][0], 1.0 - qt8 * 0.5) && close(ci[0][1], 1.0 + qt8 * 0.5));
        assert!(close(ci[1][0], 2.0 - qn * 0.5) && close(ci[1][1], 2.0 + qn * 0.5));
        assert!(close(ci[2][0], 3.0 - qn * 0.5) && close(ci[2][1], 3.0 + qn * 0.5));
        assert!(nan(ci[5]), "{method:?}");
        let ci90 = confint(&beta, &se, &df, method, 0.9).unwrap();
        assert!(close(ci90[1][1], 2.0 + qn90 * 0.5));
    }
    for bad in [0.0, 1.0, 1.5, -0.1, f64::NAN] {
        assert!(
            confint(&beta, &se, &df, DfMethod::NoDf, bad).is_err(),
            "level {bad}"
        );
    }
    assert!(confint(&beta, &se[..2], &df, DfMethod::NoDf, 0.95).is_err());
}

/// `effect_sizes` on what a port stored of a `run_fit` record, the
/// training columns lowered again, is `Fit::r2` and `Fit::icc` of the fit
/// on its own `Lowered`, bit for bit: an LMM with a random slope, a
/// binomial GLMM with a `cbind()` response (trial counts made by the
/// lowering) and a Poisson GLMM (values from a null refit). Without the
/// null refit the Poisson GLMM gives `None` and the others the same
/// values; a fit without random effects gives NaN.
#[test]
fn effect_sizes_on_stored_parts_are_the_fit_r2_and_icc() {
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let mut cbpp = csv_columns(
        include_str!("../../../validation/data/empirical/cbpp.csv"),
        &["herd", "period"],
    );
    let fail: Vec<f64> = cbpp.0["size"]
        .iter()
        .zip(&cbpp.0["incidence"])
        .map(|(n, k)| n - k)
        .collect();
    cbpp.0.insert("fail".to_string(), fail);
    let ours = |formula: &str, cols: &Columns, family, link, r: &FitResult, null_refit| {
        effect_sizes(
            formula,
            cols.0.clone(),
            cols.1.clone(),
            family_from_str(family, link).unwrap(),
            None,
            None,
            &stored(r),
            &r.aliased,
            r.converged,
            r.objective,
            null_refit,
        )
        .unwrap_or_else(|e| panic!("{formula}: {e}"))
    };
    for (formula, cols, fam, family, link) in [
        (
            "y ~ g * h + x + (1 + x | subj)",
            sim("sim_factorial"),
            Family::Gaussian,
            "gaussian",
            "identity",
        ),
        (
            "cbind(incidence, fail) ~ period + (1 | herd)",
            cbpp,
            binomial,
            "binomial",
            "logit",
        ),
        (
            "y ~ x + (1 | g)",
            sim("sim_poisson_bigsd"),
            poisson,
            "poisson",
            "log",
        ),
    ] {
        let r = run(formula, cols.clone(), family, link, 1, None, None);
        let (lo, _, fit) = direct(formula, &cols, fam, 1, None, None);
        let want = (
            fit.r2(&lo.data(), lo.spec()).expect("a mixed model"),
            fit.icc(&lo.data(), lo.spec()).expect("a mixed model"),
        );
        assert!(want.0.marginal.is_finite(), "{formula}");
        let got = ours(formula, &cols, family, link, &r, true);
        assert_eq!(format!("{got:?}"), format!("{:?}", Some(want)), "{formula}");
        let free = ours(formula, &cols, family, link, &r, false);
        let want_free = (fam != poisson).then_some(want);
        assert_eq!(format!("{free:?}"), format!("{want_free:?}"), "{formula}");
    }
    let cols = sim("sim_factorial");
    let ols = run(
        "y ~ g * h + x",
        cols.clone(),
        "gaussian",
        "identity",
        1,
        None,
        None,
    );
    let (r2, icc) =
        ours("y ~ g * h + x", &cols, "gaussian", "identity", &ols, false).expect("no refit on OLS");
    assert!([r2.marginal, r2.conditional, icc.adjusted, icc.unadjusted]
        .iter()
        .all(|v| v.is_nan()));
}

/// A stored `beta` that does not match the lowered model is an error, not
/// the NaN values of a model `Fit::r2` does not cover.
#[test]
fn effect_sizes_on_a_wrong_length_beta_is_an_error() {
    let formula = "y ~ g * h + x + (1 + x | subj)";
    let cols = sim("sim_factorial");
    let r = run(formula, cols.clone(), "gaussian", "identity", 1, None, None);
    let mut parts = stored(&r);
    parts.beta.pop();
    let got = effect_sizes(
        formula,
        cols.0,
        cols.1,
        Family::Gaussian,
        None,
        None,
        &parts,
        &r.aliased,
        r.converged,
        r.objective,
        false,
    );
    let e = got.expect_err("a beta one short");
    assert!(e.contains("the fit was made on"), "{e}");
}

/// `contrast_test` on the fields `run_fit` stores gives the kernel's own
/// test of the same `L` bit for bit: `Fit::contrast_f` on a Gaussian LMM,
/// on OLS (rank-deficient design included) and on a Gamma GLM,
/// `wald_chisq` on a binomial GLMM. The stored record carries everything the test reads.
#[test]
fn contrast_test_on_the_stored_fields_is_the_kernel_test_bitwise() {
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let (mut numeric, factor) = sim("sim_factorial");
    let x2: Vec<f64> = numeric["x"].iter().map(|v| 2.0 * v).collect();
    numeric.insert("x2".to_string(), x2);
    let with_x2 = (numeric, factor);
    let cases = [
        (
            "y ~ g + h + x + (1 + x | subj)",
            sim("sim_factorial"),
            "gaussian",
            "identity",
            Family::Gaussian,
            "satterthwaite",
        ),
        (
            "y ~ g + h + x",
            sim("sim_factorial"),
            "gaussian",
            "identity",
            Family::Gaussian,
            "residual",
        ),
        (
            "y ~ g + x + x2 + h",
            with_x2,
            "gaussian",
            "identity",
            Family::Gaussian,
            "residual",
        ),
        (
            "y ~ x + grp",
            sim("glm_gamma"),
            "gamma",
            "log",
            Family::Gamma {
                link: GammaLink::Log,
            },
            "residual",
        ),
        (
            "y ~ g + h + x + (1 | subj)",
            sim("sim_factorial_bern"),
            "binomial",
            "logit",
            binomial,
            "none",
        ),
    ];
    let mut tested = 0;
    for (formula, data, family, link, fam, df_method) in cases {
        let r = run(formula, data.clone(), family, link, 1, None, None);
        let (_, _, fit) = direct(formula, &data, fam, 1, None, None);
        assert_eq!(r.df_method.as_str(), df_method, "{formula}");
        assert_eq!(
            r.satterthwaite_record.is_some(),
            df_method == "satterthwaite"
        );
        let p = r.beta.len();
        let unit = |j: usize| {
            (0..p)
                .map(|k| f64::from(u8::from(k == j)))
                .collect::<Vec<_>>()
        };
        let mixed: Vec<f64> = (0..p).map(|k| 0.5 * k as f64 - 1.0).collect();
        // Two rows, the same two plus their sum (rank 2 of 3), one row
        // touching every column (the aliased `x2` on the third case).
        let ls: Vec<Vec<f64>> = vec![
            [unit(1), unit(2)].concat(),
            [
                unit(1),
                mixed.clone(),
                unit(1).iter().zip(&mixed).map(|(a, b)| a + b).collect(),
            ]
            .concat(),
            mixed.clone(),
        ];
        for l in ls {
            let rows = l.len() / p;
            let table = contrast_test(
                r.df_method,
                &r.beta,
                &r.vcov,
                &r.model_vcov,
                &r.aliased,
                r.nobs,
                r.dispersion,
                r.satterthwaite_record.clone(),
                None,
                &r.names,
                &l,
                rows,
            )
            .expect("a test");
            let got = term_columns(&table);
            let want = match df_method {
                "none" => wald_chisq(&fit.beta, &fit.vcov_rows(), fit.aliased(), &l, rows),
                _ => LinearTest::from_contrast_f(
                    fit.contrast_f(&l, rows).expect("a df record"),
                    if df_method == "satterthwaite" {
                        DenDf::Satterthwaite
                    } else {
                        DenDf::Residual
                    },
                ),
            };
            let (den_df, den_df_kind) = den_df_fields(want.reference);
            assert_eq!(
                [
                    got.statistic[0],
                    got.num_df[0],
                    got.den_df[0],
                    got.p_value[0]
                ]
                .map(f64::to_bits),
                [want.statistic, want.num_df, den_df, want.p_value].map(f64::to_bits),
                "{formula}: {l:?}"
            );
            assert_eq!(
                (got.kind[0], got.den_df_kind[0]),
                (kind_tag(want.kind), den_df_kind)
            );
            let notes: Vec<String> = want
                .notes
                .iter()
                .map(|n| linear_note_line(*n, &r.names))
                .collect();
            assert_eq!(got.row_notes[0], notes, "{formula}: {l:?}");
            tested += 1;
        }
    }
    assert_eq!(tested, 15);
}

#[test]
fn contrast_test_refuses_a_missing_or_bad_record() {
    let r = run(
        "y ~ g + h + x + (1 | subj)",
        sim("sim_factorial"),
        "gaussian",
        "identity",
        1,
        None,
        None,
    );
    let e = vec![1.0; r.beta.len()];
    let test = |df_method: &str, record: Option<SatterthwaiteRecord>| {
        contrast_test(
            DfMethod::parse(df_method)?,
            &r.beta,
            &r.vcov,
            &[],
            &r.aliased,
            r.nobs,
            r.dispersion,
            record,
            None,
            &r.names,
            &e,
            1,
        )
    };
    let rec = r.satterthwaite_record.clone();
    assert!(test("satterthwaite", rec.clone()).is_ok());
    let err = |res: Result<TermTable, String>| res.unwrap_err();
    assert!(err(test("satterthwaite", None)).contains("needs its satterthwaite_record"));
    let mut short = rec.expect("a record");
    short.c.pop();
    assert!(err(test("satterthwaite", Some(short))).contains("is malformed"));
    assert!(err(test("other", None)).contains("unknown df_method"));
    // A Gaussian LMM with no pass: the NaN F row with its reason.
    let row = term_columns(&test("failed", None).expect("a row"));
    assert_eq!((row.kind[0], row.den_df_kind[0]), ("F", "satterthwaite"));
    assert!([
        row.statistic[0],
        row.num_df[0],
        row.den_df[0],
        row.p_value[0],
        row.sum_sq[0]
    ]
    .iter()
    .all(|v| v.is_nan()));
    assert_eq!(
        row.row_notes[0],
        [linear_note_line(
            LinearNote::NoDf {
                reason: NoDfReason::Failed
            },
            &r.names
        )]
    );
}

/// `contrast_test` and `pairwise` on the fields `run_fit` stores of a
/// robust fit give the kernel's tests of that fit: a one-row `L`
/// the robust t² on its df, two rows the model-based F with its note,
/// each comparison its robust t; CR2 on a random intercept, HC2 on WLS.
/// Each formula also with an aliased column (`x2 = 2x`), whose stored
/// record's `t` is `p × p_Z`.
#[test]
fn contrast_test_and_pairwise_of_a_stored_robust_fit() {
    use crate::test_support::csv_table;
    let close = |a: f64, b: f64| a == b || (a / b - 1.0).abs() < 1e-12;
    for (formula, df_method, aliased) in [
        ("y ~ g + x + (1 | subj)", "pustejovsky_tipton", false),
        ("y ~ g + x", "bell_mccaffrey", false),
        ("y ~ g + x + x2 + (1 | subj)", "pustejovsky_tipton", true),
        ("y ~ g + x + x2", "bell_mccaffrey", true),
    ] {
        let (mut numeric, factor) = sim("sim_factorial");
        let n = numeric["y"].len();
        let x2: Vec<f64> = numeric["x"].iter().map(|v| 2.0 * v).collect();
        numeric.insert("x2".to_string(), x2.clone());
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 / 2.0).collect();
        let r = run_fit(
            formula,
            numeric.clone(),
            factor.clone(),
            Family::Gaussian,
            FitOptions {
                condvar: true,
                weight_power: Some(WeightPower::Fixed(1.0)),
                se_type: Some(SeType::Robust),
                ..FitOptions::default()
            },
            Some(w.clone()),
            None,
            None,
            TermTest::Wald,
        )
        .unwrap_or_else(|e| panic!("{formula}: {e}"));
        assert_eq!(r.df_method.as_str(), df_method);
        assert_eq!(
            r.satterthwaite_record.is_some(),
            df_method == "pustejovsky_tipton"
        );
        let p = r.beta.len();
        if aliased {
            let rec = r.robust.clone().expect("a robust record");
            assert_eq!(r.aliased.iter().filter(|a| **a).count(), 1, "{formula}");
            assert_eq!(rec.t.len(), p * (p - 1), "{formula}");
        }

        let csv = include_str!("../../../validation/data/simulated/sim_factorial.csv");
        let mut table = csv_table(csv, &["subj", "g", "h"]);
        table.columns.push(("x2".to_string(), Column::Numeric(x2)));
        let mut lo = crate::formula::lower(formula, &table, crate::Family::Gaussian)
            .unwrap_or_else(|e| panic!("{formula}: {e}"));
        lo.weights = Some(w);
        let opts = crate::FitOptions {
            weight_power: Some(WeightPower::Fixed(1.0)),
            se_type: Some(SeType::Robust),
            ..crate::FitOptions::default()
        };
        let m = crate::formula::fit_lowered(formula, lo, &opts)
            .unwrap_or_else(|e| panic!("{formula}: {e}"));
        let fit = m.fit();
        assert!(fit.converged(), "{formula}: not converged");
        let unit = |j: usize| {
            (0..p)
                .map(|k| f64::from(u8::from(k == j)))
                .collect::<Vec<_>>()
        };
        let one: Vec<f64> = (0..p).map(|k| [0.0, 1.0, -1.0, 0.5, 0.0][k]).collect();
        for (l, rows) in [(one, 1), ([unit(1), unit(3)].concat(), 2)] {
            let table = contrast_test(
                r.df_method,
                &r.beta,
                &r.vcov,
                &r.model_vcov,
                &r.aliased,
                r.nobs,
                r.dispersion,
                r.satterthwaite_record.clone(),
                r.robust.clone(),
                &r.names,
                &l,
                rows,
            )
            .unwrap_or_else(|e| panic!("{formula} {rows}: {e}"));
            let got = term_columns(&table);
            let want = fit
                .contrast_f(&l, rows)
                .unwrap_or_else(|e| panic!("{formula} {rows}: {e}"));
            for (a, b) in [
                (got.statistic[0], want.f),
                (got.num_df[0], want.num_df),
                (got.den_df[0], want.den_df),
                (got.p_value[0], want.p_value),
            ] {
                assert!(close(a, b), "{formula} {rows}: {a} vs {b}");
            }
            let notes: Vec<String> = want
                .notes
                .iter()
                .map(|n| linear_note_line(crate::inference::wald::linear_note(*n), &r.names))
                .collect();
            assert_eq!(got.row_notes[0], notes, "{formula} {rows}");
            assert_eq!(
                rows == 2,
                notes.contains(&linear_note_line(LinearNote::ModelBasedJointTest, &r.names)),
                "{formula} {rows}"
            );
        }
        let got = pairwise(
            formula,
            numeric,
            factor,
            Family::Gaussian,
            r.df_method,
            &r.beta,
            &r.vcov,
            &r.model_vcov,
            &r.aliased,
            r.nobs,
            r.satterthwaite_record.clone(),
            r.robust.clone(),
            "g",
            None,
            None,
            Adjust::None,
        )
        .unwrap_or_else(|e| panic!("{formula}: {e}"));
        let want = crate::formula::pairwise(&m, "g", None, Adjust::None)
            .unwrap_or_else(|e| panic!("{formula}: {e}"));
        assert_eq!(got.rows.len(), want.rows.len(), "{formula}");
        for (a, b) in got.rows.iter().zip(&want.rows) {
            for (x, y) in [
                (a.estimate, b.estimate),
                (a.se, b.se),
                (a.df, b.df),
                (a.statistic, b.statistic),
                (a.p_value, b.p_value),
            ] {
                assert!(close(x, y), "{formula} {}: {x} vs {y}", a.contrast);
            }
            assert_eq!(a.notes, b.notes, "{formula} {}", a.contrast);
        }
    }
}

// ── pairwise on a port's stored numbers ───────────────────────────────

/// One `pairwise` case: the formula, its data, the port family and link,
/// the kernel family, and the comparison asked for.
type PairwiseCase = (
    &'static str,
    Columns,
    &'static str,
    &'static str,
    Family,
    Option<(&'static str, Option<Vec<f64>>)>,
    Adjust,
);

/// A Satterthwaite LMM within the levels of `h`, OLS at two values of
/// `x` and with an empty cell (aliased columns, NaN rows), and a binomial
/// GLMM's z rows with odds ratios: every df route and both `by` kinds.
fn pairwise_cases() -> [PairwiseCase; 4] {
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    [
        (
            "y ~ g * h + x + (1 | subj)",
            sim("sim_factorial"),
            "gaussian",
            "identity",
            Family::Gaussian,
            Some(("h", None)),
            Adjust::Holm,
        ),
        (
            "y ~ g * h + x",
            sim("sim_factorial"),
            "gaussian",
            "identity",
            Family::Gaussian,
            Some(("x", Some(vec![-1.0, 0.5]))),
            Adjust::Tukey,
        ),
        (
            "y ~ g * h + x",
            sim("sim_factorial_empty"),
            "gaussian",
            "identity",
            Family::Gaussian,
            Some(("h", None)),
            Adjust::Bonferroni,
        ),
        (
            "y ~ g * h + x + (1 | subj)",
            sim("sim_factorial_bern"),
            "binomial",
            "logit",
            binomial,
            None,
            Adjust::Holm,
        ),
    ]
}

/// `pairwise` on the fields `run_fit` stores and its training columns
/// gives `formula::pairwise` on the same model bit for bit (`{:?}` of an
/// `f64` round-trips), and `pairwise_columns` carries every row field.
#[test]
fn stored_pairwise_equals_formula_pairwise() {
    for (formula, cols, family, link, fam, by, adjust) in pairwise_cases() {
        let r = run(formula, cols.clone(), family, link, 1, None, None);
        let model =
            crate::formula::fit(formula, &table(&cols), fam, &FitOptions::default()).expect("fits");
        let by_arg = by.as_ref().map(|(name, values)| match values {
            Some(v) => By::Values(name, v),
            None => By::Levels(name),
        });
        let want = crate::formula::pairwise(&model, "g", by_arg, adjust).expect("pairs");
        let got = pairwise(
            formula,
            cols.0,
            cols.1,
            family_from_str(family, link).unwrap(),
            r.df_method,
            &r.beta,
            &r.vcov,
            &r.model_vcov,
            &r.aliased,
            r.nobs,
            r.satterthwaite_record.clone(),
            None,
            "g",
            by.as_ref().map(|(name, _)| *name),
            by.and_then(|(_, values)| values),
            adjust,
        )
        .unwrap_or_else(|e| panic!("{formula}: {e}"));
        assert_eq!(format!("{got:?}"), format!("{want:?}"), "{formula}");
        let c = pairwise_columns(&got, &r.names);
        assert_eq!(c.adjust, adjust_tag(adjust));
        assert_eq!(c.levels, 3);
        for (i, row) in got.rows.iter().enumerate() {
            assert_eq!(
                format!(
                    "{:?}",
                    [
                        c.estimate[i],
                        c.se[i],
                        c.df[i],
                        c.statistic[i],
                        c.p_value[i],
                        c.p_adjusted[i],
                        c.ratio[i],
                        c.ratio_se[i]
                    ]
                ),
                format!(
                    "{:?}",
                    [
                        row.estimate,
                        row.se,
                        row.df,
                        row.statistic,
                        row.p_value,
                        row.p_adjusted,
                        row.ratio,
                        row.ratio_se
                    ]
                )
            );
            assert_eq!(c.contrast[i], row.contrast);
            assert_eq!(c.by_level[i], row.by_level.clone().unwrap_or_default());
            assert_eq!(c.row_notes[i].len(), row.notes.len());
        }
    }
}

/// `pairwise_confint` on the columns a port stores of a pairwise table
/// gives `Pairwise::confint` of the kernel's table bit for bit, under the
/// table's own adjustment and under each of the others.
#[test]
fn stored_confint_equals_pairwise_confint() {
    let mut nan_rows = 0;
    for (formula, cols, _, _, fam, by, _) in pairwise_cases() {
        let model =
            crate::formula::fit(formula, &table(&cols), fam, &FitOptions::default()).expect("fits");
        for adjust in [
            Adjust::Holm,
            Adjust::Bonferroni,
            Adjust::Tukey,
            Adjust::None,
        ] {
            let by_arg = by.as_ref().map(|(name, values)| match values {
                Some(v) => By::Values(name, v),
                None => By::Levels(name),
            });
            let pw = crate::formula::pairwise(&model, "g", by_arg, adjust).expect("pairs");
            let want = pw.confint(0.9).expect("intervals");
            let c = pairwise_columns(&pw, &[]);
            let got = pairwise_confint(
                &c.estimate,
                &c.se,
                &c.df,
                &c.p_value,
                pw.adjust,
                c.levels,
                pw.ratio,
                0.9,
            )
            .expect("intervals");
            assert_eq!(format!("{got:?}"), format!("{want:?}"), "{formula}");
            nan_rows += got.rows.iter().filter(|r| r.lower.is_nan()).count();
        }
    }
    // The empty cell's rows, under each of the four adjustments.
    assert!(nan_rows > 0);
    let err = |r: Result<PairwiseIntervals, String>| r.unwrap_err();
    let one = [0.5];
    assert!(err(pairwise_confint(
        &one,
        &one,
        &one,
        &one,
        Adjust::None,
        3,
        None,
        0.95
    ))
    .contains("malformed"));
    assert!(err(pairwise_confint(
        &one,
        &one,
        &one,
        &one,
        Adjust::None,
        2,
        None,
        1.0
    ))
    .contains("level must be"));
}
