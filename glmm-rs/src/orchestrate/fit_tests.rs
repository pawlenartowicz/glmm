//! Tests of `run_fit`, `run_compare` and the flattening of a fit.

use super::design::table_of;
use super::fit::{fit_lowered, flatten, lower_with, note_infos};
use super::*;
use crate::formula::anova::{
    compare_stored, linear_note_line, term_tests_cached, CompareError, CompareOptions,
    ComparisonTest, Model, RefitCache, Side, SsType, TermTable, TermTest, TermTestOptions,
    WeightPowerMismatch,
};
use crate::formula::{lower, Column, Lowered, ReGroupInfo, Table};
use crate::inference::tails::t_two_sided_p;
use crate::inference::{LrNote, Objective, Reference};
use crate::test_support::{csv_table, fit_warm_wo};
use crate::{
    BinomialLink, Dispersion, Family, Fit, FitOptions, Note, PoissonLink, SeType, StartValues,
    WaldSe, WeightPower,
};
use std::collections::HashMap;

#[test]
fn note_payloads_survive_flattening() {
    let group = |name: &str, terms: &[&str]| ReGroupInfo {
        name: name.to_string(),
        terms: terms.iter().map(|t| t.to_string()).collect(),
        term_index: vec![0; terms.len()],
        slot_labels: Vec::new(),
    };
    let re_groups = [
        group("word", &["(Intercept)", "catB", "catC"]),
        group("g2", &["(Intercept)"]),
    ];
    let notes = note_infos(
        vec![
            Note::SingleLevelGrouping { grouping: 1 },
            Note::RepeatedGrouping {
                first: 0,
                second: 1,
            },
            Note::SlopeConstantWithinLevels {
                grouping: 0,
                slopes: vec![0, 1],
            },
            Note::SeparatedByGrouping { grouping: 0 },
        ],
        &re_groups,
    );
    let flat: Vec<(&str, Vec<&str>)> = notes
        .iter()
        .map(|n| (n.kind, n.names.iter().map(String::as_str).collect()))
        .collect();
    assert_eq!(
        flat,
        [
            ("single_level_grouping", vec!["g2"]),
            ("repeated_grouping", vec!["word", "g2"]),
            ("slope_constant_within_levels", vec!["catB", "catC", "word"]),
            ("separated_by_grouping", vec!["word"]),
        ]
    );
    let notes = note_infos(
        vec![
            Note::SingleLevelGroupingDropped {
                grouping: "Subject".to_string(),
            },
            Note::HessianSeFallback,
            Note::NbShapeUnsettled { rounds: 25 },
            Note::ExactProfileFallback,
            // No known dataset reaches `final_eval == true` end to end, so the
            // ports' message branch is pinned from constructed notes.
            Note::PirlsExhausted {
                evals: 3,
                final_eval: false,
            },
            Note::PirlsExhausted {
                evals: 0,
                final_eval: true,
            },
        ],
        &[],
    );
    assert_eq!(notes[0].kind, "single_level_grouping_dropped");
    assert_eq!(notes[0].names, ["Subject"]);
    assert_eq!(notes[1].kind, "hessian_se_fallback");
    assert!(notes[1].names.is_empty());
    assert_eq!(notes[2].kind, "nb_shape_unsettled");
    assert_eq!(notes[2].evals, 25);
    assert_eq!(notes[3].kind, "exact_profile_fallback");
    assert!(notes[3].names.is_empty());
    assert_eq!(notes[4].kind, "pirls_exhausted");
    assert_eq!((notes[4].evals, notes[4].final_eval), (3, false));
    assert_eq!(notes[5].kind, "pirls_exhausted");
    assert_eq!((notes[5].evals, notes[5].final_eval), (0, true));
}

/// A factor column in the `(levels, codes)` form `run_fit` takes, with the
/// lexicographic level order a plain string column gets from the ports.
fn factor_col(labels: &[&str]) -> (Vec<String>, Vec<u32>) {
    let mut levels: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
    levels.sort();
    levels.dedup();
    let codes = labels
        .iter()
        .map(|l| levels.iter().position(|v| v == l).unwrap() as u32)
        .collect();
    (levels, codes)
}

#[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
fn toy_ols() -> (
    HashMap<String, Vec<f64>>,
    HashMap<String, (Vec<String>, Vec<u32>)>,
) {
    let y = vec![1.0, 2.0, 2.9, 4.1, 5.0, 6.2, 6.8, 8.1, 9.0, 10.2];
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mut numeric = HashMap::new();
    numeric.insert("y".to_string(), y);
    numeric.insert("x".to_string(), x);
    (numeric, HashMap::new())
}

#[test]
fn unknown_column_is_a_clean_error() {
    let (numeric, factor) = toy_ols();
    let err = run_fit(
        "y ~ z",
        numeric,
        factor,
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .unwrap_err();
    assert_eq!(err, "unknown column: z");
}

#[test]
fn offset_given_both_ways_is_a_clean_error() {
    let (numeric, factor) = toy_ols();
    let n = numeric["y"].len();
    let err = run_fit(
        "y ~ x + offset(x)",
        numeric,
        factor,
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        Some(vec![0.0; n]),
        None,
        TermTest::Wald,
    )
    .unwrap_err();
    assert_eq!(
        err,
        "offset given both as an offset() formula term and as the offset= argument; use one"
    );
}

#[test]
fn weights_given_both_ways_is_a_clean_error() {
    let mut numeric = HashMap::new();
    numeric.insert("s".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
    numeric.insert("f".to_string(), vec![3.0, 2.0, 1.0, 0.5]);
    numeric.insert("x".to_string(), vec![0.0, 1.0, 2.0, 3.0]);
    let n = numeric["s"].len();
    let err = run_fit(
        "cbind(s, f) ~ x",
        numeric,
        HashMap::new(),
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        Some(vec![1.0; n]),
        None,
        None,
        TermTest::Wald,
    )
    .unwrap_err();
    assert_eq!(
        err,
        "weights given both as a cbind() response and as the weights= argument; use one"
    );
}

#[test]
fn ineligible_nagq_is_stripped_with_a_warning_not_an_error() {
    // Each shape must strip to Laplace and report the warn-and-strip
    // message, never surface the kernel's shape panic.
    let (numeric, mut factor) = toy_ols();
    let g: Vec<&str> = ["a", "b", "c", "d", "e"]
        .iter()
        .copied()
        .cycle()
        .take(10)
        .collect();
    factor.insert("g".to_string(), factor_col(&g));
    // Gaussian: ineligible family.
    let result = run_fit(
        "y ~ x + (1 | g)",
        numeric,
        factor,
        Family::Gaussian,
        FitOptions {
            nagq: 3,
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .expect("ineligible nagq must be stripped, not an error");
    assert!(result.agq_warning);

    // Binomial with q_p = 1 + 3 slopes = 4 > 3: over the random-effect cap.
    let n = 40;
    let mut numeric = HashMap::new();
    let mut y = Vec::new();
    for i in 0..n {
        y.push(if (i * 7 + i / 3) % 3 == 0 { 1.0 } else { 0.0 });
    }
    numeric.insert("y".to_string(), y);
    for (k, name) in ["x1", "x2", "x3"].iter().enumerate() {
        let col: Vec<f64> = (0..n)
            .map(|i| (((i * (k + 3) + k) % 11) as f64) / 11.0 - 0.5)
            .collect();
        numeric.insert(name.to_string(), col);
    }
    let mut factor = HashMap::new();
    let labels: Vec<&str> = ["a", "b", "c", "d", "e"]
        .iter()
        .copied()
        .cycle()
        .take(n)
        .collect();
    factor.insert("g".to_string(), factor_col(&labels));
    let result = run_fit(
        "y ~ x1 + x2 + x3 + (1 + x1 + x2 + x3 | g)",
        numeric,
        factor,
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        FitOptions {
            nagq: 3,
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .expect("over-cap nagq must be stripped, not an error");
    assert!(result.agq_warning);
}

/// A message without a row is the same whatever the port shows.
#[test]
fn a_message_without_a_row_is_shown_unchanged() {
    let plain = PortError::from("no row here".to_string());
    assert_eq!(
        (plain.row, plain.message_shown(4, 0)),
        (None, plain.message.clone())
    );
}

/// The lowering's own row refusal (`formula::Error`) keeps its row too, on
/// the fit and on the comparison.
#[test]
fn a_lowering_refusal_keeps_its_row() {
    let (mut numeric, factor) = toy_ols();
    numeric.get_mut("x").unwrap()[2] = f64::INFINITY;
    let err = run_fit(
        "y ~ x",
        numeric.clone(),
        factor.clone(),
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .unwrap_err();
    assert_eq!(err.row, Some(2));
    assert_eq!(err.message_shown(3, 0), "term 'x' is not finite at row 3");
    let err = run_compare(
        "y ~ 1",
        "y ~ x",
        numeric,
        factor,
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(err.row, Some(2));
    assert_eq!(err.message_shown(3, 0), "term 'x' is not finite at row 3");
}

#[test]
fn mismatched_column_lengths_is_a_clean_error() {
    let mut numeric = std::collections::HashMap::new();
    numeric.insert("y".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    numeric.insert("x".to_string(), vec![0.0, 1.0]); // ragged: 2 vs 5
    let err = run_fit(
        "y ~ x",
        numeric,
        std::collections::HashMap::new(),
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .unwrap_err();
    assert!(err.contains("x"), "{err}");
    assert!(err.contains('5') && err.contains('2'), "{err}");
}

/// Warm-start values reach the mixed-model fit: a start far from the
/// optimum takes a different number of evaluations than a cold start yet
/// lands on the same estimates. Fixed-only fits ignore `StartValues`, so
/// this needs a random effect.
#[test]
fn mixed_warm_start_reaches_same_optimum_with_different_evals() {
    let (numeric, mut factor) = toy_ols();
    let g: Vec<&str> = ["a", "b", "c", "d", "e"]
        .iter()
        .copied()
        .cycle()
        .take(10)
        .collect();
    factor.insert("g".to_string(), factor_col(&g));
    let cold = run_fit(
        "y ~ x + (1 | g)",
        numeric.clone(),
        factor.clone(),
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .expect("cold mixed fit should converge");
    assert!(cold.converged);

    // θ₀ = 5 sits far from this fit's optimum, so the warm search runs
    // well past the cold one's count (25 evaluations against 11, x86_64).
    // A nearer start can tie the cold count by chance, and a tie says
    // nothing about whether the start reached the fit.
    let warm = run_fit(
        "y ~ x + (1 | g)",
        numeric,
        factor,
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        Some(StartValues {
            beta: cold.beta.clone(),
            theta: vec![5.0],
            dispersion: None,
        }),
        TermTest::Wald,
    )
    .expect("warm-started mixed fit should converge");
    assert!(warm.converged);
    for (c, w) in cold.beta.iter().zip(warm.beta.iter()) {
        assert!((c - w).abs() < 1e-6, "beta moved: {c} vs {w}");
    }
    assert_ne!(
        warm.n_eval, cold.n_eval,
        "a far-off warm start must change the evaluation count"
    );
}

/// A Gaussian random-intercept fixture: 8 groups of 6 rows, `y = 1 + 0.5x +
/// b_g + e` with deterministic `b_g` and `e`.
#[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
fn toy_lmm() -> (
    HashMap<String, Vec<f64>>,
    HashMap<String, (Vec<String>, Vec<u32>)>,
) {
    let (mut x, mut y, mut g) = (Vec::new(), Vec::new(), Vec::new());
    let labels = ["a", "b", "c", "d", "e", "f", "g", "h"];
    for (k, label) in labels.iter().enumerate() {
        let b = ((k * 5) % 8) as f64 / 4.0 - 0.9;
        for j in 0..6 {
            let xi = j as f64 - 2.5 + 0.1 * k as f64;
            let e = (((k * 7 + j * 3) % 11) as f64 - 5.0) / 6.0;
            x.push(xi);
            y.push(1.0 + 0.5 * xi + b + e);
            g.push(*label);
        }
    }
    let mut numeric = HashMap::new();
    numeric.insert("y".to_string(), y);
    numeric.insert("x".to_string(), x);
    let mut factor = HashMap::new();
    factor.insert("g".to_string(), factor_col(&g));
    (numeric, factor)
}

/// Three rows for three fixed coefficients, two groups.
#[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
fn three_row_lmm() -> (
    HashMap<String, Vec<f64>>,
    HashMap<String, (Vec<String>, Vec<u32>)>,
) {
    let mut numeric = HashMap::new();
    numeric.insert("y".to_string(), vec![1.0, 2.0, 4.0]);
    numeric.insert("x1".to_string(), vec![0.0, 1.0, 3.0]);
    numeric.insert("x2".to_string(), vec![1.0, 0.0, 2.0]);
    let mut factor = HashMap::new();
    factor.insert("g".to_string(), factor_col(&["a", "b", "a"]));
    (numeric, factor)
}

#[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
fn fit_with(
    formula: &str,
    (numeric, factor): (
        HashMap<String, Vec<f64>>,
        HashMap<String, (Vec<String>, Vec<u32>)>,
    ),
    family: &str,
    link: &str,
    dispersion: Option<f64>,
    satterthwaite: bool,
) -> FitResult {
    run_fit(
        formula,
        numeric,
        factor,
        family_from_str(family, link).unwrap(),
        FitOptions {
            dispersion: dispersion.map_or(Dispersion::Default, Dispersion::Fixed),
            satterthwaite,
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .expect("fit should succeed")
}

fn has_curvature_note(r: &FitResult) -> bool {
    r.notes.iter().any(|n| n.kind == "satterthwaite_curvature")
}

/// `coef_p` is the t test on each row's df (`contrast_t`'s df agrees with
/// `coef_df` to rounding).
fn assert_t_p(r: &FitResult) {
    for i in 0..r.beta.len() {
        let want = t_two_sided_p(r.beta[i] / r.se[i], r.coef_df[i]);
        assert!(
            (r.coef_p[i] - want).abs() < 1e-12,
            "coef_p[{i}] = {}, want {want}",
            r.coef_p[i]
        );
    }
}

#[test]
fn lmm_carries_satterthwaite_df_and_p_unless_switched_off() {
    let on = fit_with(
        "y ~ x + (1 | g)",
        toy_lmm(),
        "gaussian",
        "identity",
        None,
        true,
    );
    assert!(on.converged);
    assert_eq!(on.df_method.as_str(), "satterthwaite");
    assert_eq!(on.coef_df.len(), 2);
    assert!(on.coef_df.iter().all(|d| d.is_finite() && *d > 0.0));
    assert_t_p(&on);
    assert!(on.rss.is_nan() && on.r_squared.is_nan() && on.adj_r_squared.is_nan());

    let off = fit_with(
        "y ~ x + (1 | g)",
        toy_lmm(),
        "gaussian",
        "identity",
        None,
        false,
    );
    assert_eq!(off.df_method.as_str(), "disabled");
    assert!(off.coef_df.iter().all(|d| d.is_nan()));
    assert!(off.coef_p.iter().all(|v| v.is_nan()));
    assert!(!has_curvature_note(&off));
    assert_eq!(off.beta, on.beta);
    assert_eq!(off.se, on.se);
}

#[test]
fn df_method_text_round_trips() {
    for m in DfMethod::ALL {
        assert_eq!(DfMethod::parse(m.as_str()), Ok(m));
    }
    let err = DfMethod::parse("other").unwrap_err();
    assert_eq!(err, "unknown df_method \"other\"");
}

#[test]
fn ols_and_dispersion_glm_keep_residual_df_whatever_the_switch() {
    for switch in [true, false] {
        let ols = fit_with("y ~ x", toy_ols(), "gaussian", "identity", None, switch);
        assert_eq!(ols.df_method.as_str(), "residual");
        assert_eq!(ols.coef_df, vec![8.0, 8.0]);
        assert_t_p(&ols);

        let gamma = fit_with("y ~ x", toy_gamma(), "gamma", "log", None, switch);
        assert_eq!(gamma.df_method.as_str(), "residual");
        assert_eq!(gamma.coef_df, vec![98.0, 98.0]);
        assert_t_p(&gamma);
    }
    // A held φ keeps z, as `summary.glm(dispersion = v)` does.
    let held = fit_with("y ~ x", toy_gamma(), "gamma", "log", Some(2.5), true);
    assert!(held.converged);
    assert_eq!(
        held.dispersion, 2.5,
        "dispersion: Some(v) holds phi at v, not estimated"
    );
    assert_eq!(held.df_method.as_str(), "none");
    assert!(held.coef_df.iter().chain(&held.coef_p).all(|v| v.is_nan()));
    let glmm = fit_with(
        "cbind(incidence, healthy) ~ period + (1 | herd)",
        cbpp_columns(),
        "binomial",
        "logit",
        None,
        true,
    );
    assert_eq!(glmm.df_method.as_str(), "none");
    assert!(glmm.coef_df.iter().chain(&glmm.coef_p).all(|v| v.is_nan()));
    assert!(glmm.rss.is_nan() && glmm.r_squared.is_nan());
}

/// A Gaussian LMM with no finite endpoint (three rows for three
/// coefficients) carries the failed record and the note; switched off,
/// neither.
#[test]
fn degenerate_lmm_is_failed_with_the_curvature_note() {
    let formula = "y ~ x1 + x2 + (1 | g)";
    let r = fit_with(formula, three_row_lmm(), "gaussian", "identity", None, true);
    assert!(!r.converged);
    assert_eq!(r.df_method.as_str(), "failed");
    let note = r
        .notes
        .iter()
        .find(|n| n.kind == "satterthwaite_curvature")
        .expect("the failed pass carries the note");
    assert!(note.nonfinite, "{note:?}");
    let off = fit_with(
        formula,
        three_row_lmm(),
        "gaussian",
        "identity",
        None,
        false,
    );
    assert_eq!(off.df_method.as_str(), "disabled");
    assert!(!has_curvature_note(&off));
}

/// `summary.lm`'s R² on closed forms: simple regression (R² is the squared
/// correlation of x and y), intercept only (0), no intercept (uncentred).
/// The weighted and offset fixtures are checked against `summary.lm` itself
/// in the R port's tests.
#[test]
fn ols_r_squared_follows_summary_lm() {
    let (numeric, _) = toy_ols();
    let (x, y) = (numeric["x"].clone(), numeric["y"].clone());
    let n = y.len() as f64;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mx, my) = (mean(&x), mean(&y));
    let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
    let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);

    let r = fit_with("y ~ x", toy_ols(), "gaussian", "identity", None, true);
    assert_eq!(r.names, vec!["(Intercept)".to_string(), "x".to_string()]);
    assert!(r.converged);
    assert!(close(r.beta[1], sxy / sxx), "slope {}", r.beta[1]);
    let r2 = sxy * sxy / (sxx * syy);
    assert!(close(r.r_squared, r2), "{} vs {r2}", r.r_squared);
    let adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - 2.0);
    assert!(close(r.adj_r_squared, adj), "{} vs {adj}", r.adj_r_squared);
    assert!(close(r.rss, syy - sxy * sxy / sxx), "{}", r.rss);

    let one = fit_with("y ~ 1", toy_ols(), "gaussian", "identity", None, true);
    assert_eq!((one.r_squared, one.adj_r_squared), (0.0, 0.0));
    assert!(close(one.rss, syy), "{}", one.rss);

    let origin = fit_with("y ~ 0 + x", toy_ols(), "gaussian", "identity", None, true);
    let (sxx0, syy0) = (
        x.iter().map(|v| v * v).sum::<f64>(),
        y.iter().map(|v| v * v).sum::<f64>(),
    );
    let sxy0: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
    let r2 = sxy0 * sxy0 / (sxx0 * syy0);
    assert!(close(origin.r_squared, r2), "{} vs {r2}", origin.r_squared);
    let adj = 1.0 - (1.0 - r2) * n / (n - 1.0);
    assert!(
        close(origin.adj_r_squared, adj),
        "{} vs {adj}",
        origin.adj_r_squared
    );

    let mut tiny = HashMap::new();
    tiny.insert("y".to_string(), vec![1.0, 3.0]);
    tiny.insert("x".to_string(), vec![0.0, 1.0]);
    let unfit = fit_with(
        "y ~ x",
        (tiny, HashMap::new()),
        "gaussian",
        "identity",
        None,
        true,
    );
    assert!(!unfit.converged);
    assert!(unfit.rss.is_nan() && unfit.r_squared.is_nan() && unfit.adj_r_squared.is_nan());
}

/// A constant response leaves R² as a ratio of round-off, so both R² are
/// NaN; an exactly fitted response that does vary keeps R² = 1.
#[test]
fn ols_r_squared_is_nan_when_nothing_varies() {
    let x: Vec<f64> = (0..50).map(|i| i as f64 / 7.0).collect();
    let data = |y: Vec<f64>| {
        let mut numeric = HashMap::new();
        numeric.insert("y".to_string(), y);
        numeric.insert("x".to_string(), x.clone());
        (numeric, HashMap::new())
    };

    let flat = fit_with(
        "y ~ x",
        data(vec![3.7; 50]),
        "gaussian",
        "identity",
        None,
        true,
    );
    assert!(flat.converged);
    assert!(flat.r_squared.is_nan(), "{}", flat.r_squared);
    assert!(flat.adj_r_squared.is_nan(), "{}", flat.adj_r_squared);

    let line = data(x.iter().map(|v| 2.0 + 3.0 * v).collect());
    let exact = fit_with("y ~ x", line, "gaussian", "identity", None, true);
    assert_eq!((exact.r_squared, exact.adj_r_squared), (1.0, 1.0));
}

/// Deterministic fixed-effects-only Gamma(log) fixture: `y = exp(1 +
/// 0.5·x) · jitter` on a centered `x`, `jitter` a 7-cycle deterministic
/// wobble around 1 (same shape as `glm.rs`'s Gamma fixtures) so the
/// estimated-vs-fixed dispersion paths are both exercised on real data.
#[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
fn toy_gamma() -> (
    HashMap<String, Vec<f64>>,
    HashMap<String, (Vec<String>, Vec<u32>)>,
) {
    let n = 100;
    let mut x = vec![0.0f64; n];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let xi = (i as f64) / (n as f64) - 0.5;
        let mu = (1.0 + 0.5 * xi).exp();
        let jitter = 1.0 + 0.05 * (((i % 7) as f64) - 3.0) / 3.0;
        x[i] = xi;
        y[i] = mu * jitter;
    }
    let mut numeric = HashMap::new();
    numeric.insert("y".to_string(), y);
    numeric.insert("x".to_string(), x);
    (numeric, HashMap::new())
}

/// The committed `cbpp` herd dataset (`validation/data/empirical/cbpp.csv`)
/// through `run_fit`'s `cbind()` aggregated-binomial path: `incidence`
/// successes out of `size` trials per herd/period. `healthy = size −
/// incidence` is precomputed since `cbind()` only accepts two bare column
/// names, not an expression.
#[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
fn cbpp_columns() -> (
    HashMap<String, Vec<f64>>,
    HashMap<String, (Vec<String>, Vec<u32>)>,
) {
    let csv = include_str!("../../../validation/data/empirical/cbpp.csv");
    let mut herd_labels: Vec<String> = Vec::new();
    let mut period_labels: Vec<String> = Vec::new();
    let mut incidence = Vec::new();
    let mut healthy = Vec::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let inc: f64 = f[1].parse().unwrap();
        let size: f64 = f[2].parse().unwrap();
        herd_labels.push(f[0].to_string());
        incidence.push(inc);
        healthy.push(size - inc);
        period_labels.push(f[3].to_string());
    }
    let herd_refs: Vec<&str> = herd_labels.iter().map(String::as_str).collect();
    let period_refs: Vec<&str> = period_labels.iter().map(String::as_str).collect();
    let mut numeric = HashMap::new();
    numeric.insert("incidence".to_string(), incidence);
    numeric.insert("healthy".to_string(), healthy);
    let mut factor = HashMap::new();
    factor.insert("herd".to_string(), factor_col(&herd_refs));
    factor.insert("period".to_string(), factor_col(&period_refs));
    (numeric, factor)
}

/// `wald_se="rx"` is the only route by which the Python/R ports select
/// `WaldSe::Rx`; a no-op enum arm would leave `se` identical to the
/// `WaldSe::Hessian` default and pass silently.
/// Same herd/period random-intercept model `fit_glmm_cbpp_hessian_rx_agree_on_fitted`
/// (`fit/glmm_tests.rs`) checks at the kernel-struct layer — this is the
/// orchestration layer above it.
#[test]
fn cbpp_glmm_wald_se_rx_differs_from_hessian() {
    let (numeric, factor) = cbpp_columns();
    let hessian = run_fit(
        "cbind(incidence, healthy) ~ period + (1 | herd)",
        numeric.clone(),
        factor.clone(),
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .expect("hessian-SE fit should converge");
    assert!(hessian.converged);
    assert_eq!(hessian.beta.len(), 4);

    let rx = run_fit(
        "cbind(incidence, healthy) ~ period + (1 | herd)",
        numeric,
        factor,
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        FitOptions {
            wald_se: WaldSe::Rx,
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        TermTest::Wald,
    )
    .expect("rx-SE fit should converge");
    assert!(rx.converged);

    for (h, r) in hessian.beta.iter().zip(rx.beta.iter()) {
        assert!(
            (h - r).abs() < 1e-6,
            "wald_se must select an SE denominator, not move the fitted optimum: {h} vs {r}"
        );
    }
    let se_differs = hessian
        .se
        .iter()
        .zip(rx.se.iter())
        .any(|(h, r)| (h - r).abs() > 1e-6);
    assert!(
        se_differs,
        "hessian={:?} rx={:?} must differ, or wald_se=\"rx\" is a silent no-op",
        hessian.se, rx.se
    );
}

// ── run_compare, the stored per-term tables, the offset ───────────────

pub(super) type Columns = (
    HashMap<String, Vec<f64>>,
    HashMap<String, (Vec<String>, Vec<u32>)>,
);

/// A committed simulated dataset (`validation/data/simulated/<name>.csv`)
/// as the column maps `run_fit` takes, `subj`, `g` and `h` as factors.
pub(super) fn sim(name: &str) -> Columns {
    let path = format!(
        "{}/../validation/data/simulated/{name}.csv",
        env!("CARGO_MANIFEST_DIR")
    );
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    csv_columns(&raw, &["subj", "g", "h"])
}

/// A CSV's columns as the maps `run_fit` takes, the `factors` as factors.
pub(super) fn csv_columns(raw: &str, factors: &[&str]) -> Columns {
    let mut numeric = HashMap::new();
    let mut factor = HashMap::new();
    for (col, values) in csv_table(raw, factors).columns {
        match values {
            Column::Numeric(v) => {
                numeric.insert(col, v);
            }
            Column::Factor { levels, codes } => {
                factor.insert(col, (levels, codes));
            }
        }
    }
    (numeric, factor)
}

/// The same columns as a `Table`, for the direct kernel path.
pub(super) fn table((numeric, factor): &Columns) -> Table {
    table_of(numeric.clone(), factor.clone()).expect("a table")
}

/// `formula` lowered and fitted straight through the kernel, with the
/// options `run_fit` and `run_compare` set (`wald_se = "hessian"`,
/// `satterthwaite = true`): the reference the stored results must equal.
pub(super) fn direct(
    formula: &str,
    data: &Columns,
    family: Family,
    nagq: u8,
    reml: Option<bool>,
    offset: Option<Vec<f64>>,
) -> (Lowered, FitOptions, Fit) {
    let mut lo = lower(formula, &table(data), family).expect("lowers");
    let fit_opts = FitOptions {
        wald_se: WaldSe::Hessian,
        nagq,
        reml,
        satterthwaite: true,
        ..FitOptions::default()
    };
    if offset.is_some() {
        lo.offset = offset;
    }
    let fit = fit_warm_wo(
        &lo.x,
        &lo.y,
        lo.n,
        lo.p,
        &lo.model,
        &lo.ids,
        None,
        lo.weights.as_deref(),
        lo.offset.as_deref(),
        &fit_opts,
    );
    (lo, fit_opts, fit)
}

pub(super) fn run(
    formula: &str,
    (numeric, factor): Columns,
    family: &str,
    link: &str,
    nagq: u8,
    offset: Option<Vec<f64>>,
    reml: Option<bool>,
) -> FitResult {
    run_fit(
        formula,
        numeric,
        factor,
        family_from_str(family, link).unwrap(),
        FitOptions {
            nagq,
            condvar: true,
            reml,
            ..FitOptions::default()
        },
        None,
        offset,
        None,
        TermTest::Wald,
    )
    .unwrap_or_else(|e| panic!("{formula}: {e}"))
}

/// Every `FitResult` field read off the kernel's `Fit`, as one string
/// (`{:?}` of an `f64` round-trips, so equal strings are equal bits).
fn kernel_fields(r: &FitResult) -> String {
    format!(
        "{:?}",
        (
            (&r.beta, &r.se, &r.vcov, &r.tau2, &r.varcorr, &r.stddev_se),
            (&r.aliased, r.dispersion, r.converged, r.n_eval, r.deviance),
            (r.loglik, r.df, r.reml, &r.fitted, &r.ranef, &r.ranef_levels),
            (&r.coef_df, &r.design_summary, r.nobs),
        )
    )
}

fn fit_fields(lo: &Lowered, fit: &Fit) -> String {
    let ds = fit.design_summary.as_ref().map(|d| DesignSummary {
        counts: d.counts.clone(),
        s: d.s.clone(),
        c: d.c.clone(),
    });
    format!(
        "{:?}",
        (
            (
                &fit.beta,
                &fit.se,
                &fit.vcov_rows(),
                &fit.tau2,
                &fit.varcorr,
                &fit.stddev_se
            ),
            (
                fit.aliased(),
                fit.dispersion,
                fit.converged(),
                fit.n_eval,
                fit.deviance
            ),
            (
                fit.loglik,
                fit.df,
                fit.reml,
                &fit.fitted,
                &fit.ranef,
                &fit.ranef_levels
            ),
            (&fit.coef_df, &ds, lo.n),
        )
    )
}

fn bits(v: &Option<Vec<f64>>) -> Option<Vec<u64>> {
    v.as_ref().map(|v| v.iter().map(|x| x.to_bits()).collect())
}

/// `run_fit` hands on the kernel's fit bit for bit, with its objective
/// (an AGQ fit's node count included) and the offset it fitted with,
/// from `offset=` or from `offset()` in the formula.
#[test]
fn run_fit_keeps_the_kernel_fit_bitwise_with_its_objective_and_offset() {
    let (mut numeric, factor) = toy_ols();
    let off: Vec<f64> = (0..10).map(|i| 0.1 * i as f64).collect();
    numeric.insert("o".to_string(), off.clone());
    let ols = (numeric, factor);
    let gaussian = Family::Gaussian;
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let cbpp = "cbind(incidence, healthy) ~ period + (1 | herd)";
    let lmm = "y ~ g * h + x + (1 + x | subj)";
    let cases = [
        (
            lmm,
            sim("sim_factorial"),
            "gaussian",
            gaussian,
            1,
            None,
            Objective::Reml,
        ),
        (
            cbpp,
            cbpp_columns(),
            "binomial",
            binomial,
            5,
            None,
            Objective::Agq { nodes: 5 },
        ),
        (
            "y ~ x + (1 | g)",
            toy_lmm(),
            "gaussian",
            gaussian,
            1,
            None,
            Objective::Reml,
        ),
        (
            "y ~ x",
            ols.clone(),
            "gaussian",
            gaussian,
            1,
            Some(off.clone()),
            Objective::LeastSquares,
        ),
        (
            "y ~ x + offset(o)",
            ols,
            "gaussian",
            gaussian,
            1,
            None,
            Objective::LeastSquares,
        ),
    ];
    for (formula, data, family, fam, nagq, offset, objective) in cases {
        let link = if family == "binomial" {
            "logit"
        } else {
            "identity"
        };
        let want = (offset.is_some() || formula.contains("offset(")).then(|| off.clone());
        let r = run(
            formula,
            data.clone(),
            family,
            link,
            nagq,
            offset.clone(),
            None,
        );
        let (lo, _, fit) = direct(formula, &data, fam, nagq, None, offset);
        assert_eq!(kernel_fields(&r), fit_fields(&lo, &fit), "{formula}");
        // Only the unweighted `(1 | g)` row carries a design summary.
        assert_eq!(
            r.design_summary.is_some(),
            formula.contains("(1 | g)"),
            "{formula}"
        );
        assert_eq!(r.objective, fit.objective, "{formula}");
        assert_eq!(r.objective, objective, "{formula}");
        assert_eq!(bits(&r.offset), bits(&lo.offset), "{formula}");
        assert_eq!(bits(&r.offset), bits(&want), "{formula}");
    }
}

fn table_string(t: &Result<TermTable, TermTableError>) -> String {
    format!("{t:?}")
}

/// The stored tables are `term_tests` on the fit's own `Lowered` and
/// `Fit`, both types, bit for bit: an LMM, its OLS twin, a GLMM; the
/// refusal of a formula without its marginal term is stored for both
/// types; an empty cell puts the missing-cell note on the Type III table.
#[test]
fn run_fit_stores_both_per_term_tables_as_term_tests_gives_them() {
    let gaussian = Family::Gaussian;
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let cases = [
        ("y ~ g * h + x + (1 + x | subj)", "sim_factorial", gaussian),
        ("y ~ g * h + x", "sim_factorial", gaussian),
        ("y ~ g * h + x + (1 | subj)", "sim_factorial_bern", binomial),
        ("y ~ x + x:g", "sim_factorial", gaussian),
        (
            "y ~ g * h + x + (1 + x | subj)",
            "sim_factorial_empty",
            gaussian,
        ),
    ];
    for (formula, data, fam) in cases {
        let (family, link) = if fam == gaussian {
            ("gaussian", "identity")
        } else {
            ("binomial", "logit")
        };
        let r = run(formula, sim(data), family, link, 1, None, None);
        let (lo, fit_opts, fit) = direct(formula, &sim(data), fam, 1, None, None);
        for (stored, ss_type) in [
            (&r.term_table_type3, SsType::Type3),
            (&r.term_table_type2, SsType::Type2),
        ] {
            let opts = TermTestOptions {
                ss_type,
                test: TermTest::Auto,
            };
            let want = term_tests_cached(&lo, &fit_opts, &fit, &opts, &mut RefitCache::default())
                .map_err(|e| TermTableError::Refused(e.to_string()));
            assert_eq!(
                table_string(stored),
                table_string(&want),
                "{formula} on {data}"
            );
        }
    }
    // What the cases above hold: a refusal for both types, and the note.
    let nonmarginal = run(
        "y ~ x + x:g",
        sim("sim_factorial"),
        "gaussian",
        "identity",
        1,
        None,
        None,
    );
    assert!(nonmarginal.term_table_type3.is_err() && nonmarginal.term_table_type2.is_err());
    let empty = run(
        "y ~ g * h + x + (1 + x | subj)",
        sim("sim_factorial_empty"),
        "gaussian",
        "identity",
        1,
        None,
        None,
    );
    let t3 = empty.term_table_type3.as_ref().expect("a table");
    assert!(
        t3.notes.iter().any(|n| n.contains("Missing cells")),
        "{:?}",
        t3.notes
    );
    let lmm = run(
        "y ~ g * h + x + (1 + x | subj)",
        sim("sim_factorial"),
        "gaussian",
        "identity",
        1,
        None,
        None,
    );
    let cols = term_columns(lmm.term_table_type3.as_ref().expect("a table"));
    assert_eq!(cols.label, ["g", "h", "x", "g:h"]);
    assert!(cols.den_df_kind.iter().all(|k| *k == "satterthwaite"));
    assert_eq!(choice_tags(&lmm.term_test_auto), ("F", "f_test"));
}

/// `run_fit` as [`run`] makes it at `nagq = 1`, with `anova_test` given.
fn run_asking(
    formula: &str,
    (numeric, factor): Columns,
    family: Family,
    anova_test: TermTest,
) -> FitResult {
    run_fit(
        formula,
        numeric,
        factor,
        family,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        anova_test,
    )
    .unwrap_or_else(|e| panic!("{formula}: {e}"))
}

/// With `anova_test = "LRT"` the stored likelihood-ratio tables are
/// `term_tests` under `TermTest::Lrt` on the fit's own `Lowered` and
/// `Fit`, both types, bit for bit, although the two tables share their
/// refits: a GLMM (two of its four Type III refits are Type II designs),
/// its GLM twin, and a Gaussian LMM, whose two slots hold the refusal of
/// a fit with F rows. `term_columns` names the table's test and reason.
#[test]
fn run_fit_stores_the_likelihood_ratio_tables_as_term_tests_gives_them() {
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let glmm = "y ~ g * h + x + (1 | subj)";
    let cases = [
        (glmm, "sim_factorial_bern", binomial),
        ("y ~ g * h + x", "sim_factorial_bern", binomial),
        (
            "y ~ g * h + x + (1 + x | subj)",
            "sim_factorial",
            Family::Gaussian,
        ),
    ];
    for (formula, data, fam) in cases {
        let r = run_asking(formula, sim(data), fam, TermTest::Lrt);
        let (lo, fit_opts, fit) = direct(formula, &sim(data), fam, 1, None, None);
        for (stored, ss_type) in [
            (&r.term_table_type3_lrt, SsType::Type3),
            (&r.term_table_type2_lrt, SsType::Type2),
        ] {
            let stored = stored.as_ref().expect("asked for");
            assert_eq!(stored.is_ok(), fam != Family::Gaussian, "{formula}");
            let opts = TermTestOptions {
                ss_type,
                test: TermTest::Lrt,
            };
            let want = term_tests_cached(&lo, &fit_opts, &fit, &opts, &mut RefitCache::default())
                .map_err(|e| TermTableError::Refused(e.to_string()));
            assert_eq!(
                table_string(stored),
                table_string(&want),
                "{formula} on {data}"
            );
        }
    }
    let r = run_asking(glmm, sim("sim_factorial_bern"), binomial, TermTest::Lrt);
    let t3 = r.term_table_type3_lrt.as_ref().expect("asked for");
    let cols = term_columns(t3.as_ref().expect("a table"));
    assert_eq!((cols.test, cols.reason), ("lrt", "requested"));
    assert!(cols.kind.iter().all(|k| *k == "lrt"), "{:?}", cols.kind);
}

/// A fit other than a Poisson GLMM (here a binomial GLMM and GLM, whose
/// `"auto"` is the Wald χ²) stores no likelihood-ratio table unless it
/// asks with `anova_test = "LRT"`, and asking moves nothing else: the fit
/// and its two Wald tables are the same bits under `"auto"`, `"Wald"` and
/// `"LRT"`.
#[test]
fn no_fit_stores_a_likelihood_ratio_table_unless_asked() {
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let glm = run_asking(
        "y ~ g * h + x",
        sim("sim_factorial_bern"),
        binomial,
        TermTest::Auto,
    );
    assert_eq!(
        choice_tags(&glm.term_test_auto),
        ("chisq", "no_random_effects")
    );
    assert!(glm.term_table_type3_lrt.is_none() && glm.term_table_type2_lrt.is_none());
    let formula = "y ~ g * h + x + (1 | subj)";
    let data = || sim("sim_factorial_bern");
    let asked = run_asking(formula, data(), binomial, TermTest::Lrt);
    assert!(asked.term_table_type3_lrt.is_some() && asked.term_table_type2_lrt.is_some());
    assert_eq!(
        choice_tags(&asked.term_test_auto),
        ("chisq", "binomial_mixed")
    );
    for anova_test in [TermTest::Auto, TermTest::Wald] {
        let r = run_asking(formula, data(), binomial, anova_test);
        assert_eq!(
            choice_tags(&r.term_test_auto),
            ("chisq", "binomial_mixed"),
            "{anova_test:?}"
        );
        assert!(
            r.term_table_type3_lrt.is_none() && r.term_table_type2_lrt.is_none(),
            "{anova_test:?}"
        );
        assert_eq!(kernel_fields(&r), kernel_fields(&asked), "{anova_test:?}");
        for (t, u) in [
            (&r.term_table_type3, &asked.term_table_type3),
            (&r.term_table_type2, &asked.term_table_type2),
        ] {
            assert_eq!(table_string(t), table_string(u), "{anova_test:?}");
        }
    }
}

/// A Poisson GLMM's `"auto"` is the likelihood-ratio χ², so the fit
/// stores the likelihood-ratio tables under `"auto"` (built under
/// `TermTest::Auto`) as under `"LRT"`, with the same rows; `"Wald"`
/// stores none. Its two other slots hold the Wald tables under every
/// value, and the fit is the same bits under all three.
#[test]
fn auto_stores_the_likelihood_ratio_tables_of_a_poisson_glmm() {
    let formula = "y ~ x + (1 | g)";
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let (lo, fit_opts, fit) = direct(formula, &sim("glmm_poisson"), poisson, 1, None, None);
    let want = |ss_type, test| {
        let opts = TermTestOptions { ss_type, test };
        table_string(
            &term_tests_cached(&lo, &fit_opts, &fit, &opts, &mut RefitCache::default())
                .map_err(|e| TermTableError::Refused(e.to_string())),
        )
    };
    let values = [TermTest::Auto, TermTest::Wald, TermTest::Lrt];
    let fits = values.map(|t| run_asking(formula, sim("glmm_poisson"), poisson, t));
    for (r, anova_test) in fits.iter().zip(values) {
        assert_eq!(
            choice_tags(&r.term_test_auto),
            ("lrt", "poisson_mixed"),
            "{anova_test:?}"
        );
        assert_eq!(kernel_fields(r), kernel_fields(&fits[0]), "{anova_test:?}");
        for (wald, lrt, ss_type) in [
            (&r.term_table_type3, &r.term_table_type3_lrt, SsType::Type3),
            (&r.term_table_type2, &r.term_table_type2_lrt, SsType::Type2),
        ] {
            assert_eq!(
                table_string(wald),
                want(ss_type, TermTest::Wald),
                "{anova_test:?}"
            );
            if anova_test == TermTest::Wald {
                assert!(lrt.is_none(), "{anova_test:?}");
                continue;
            }
            let lrt = lrt.as_ref().expect("stored");
            assert_eq!(
                table_string(lrt),
                want(ss_type, anova_test),
                "{anova_test:?}"
            );
        }
    }
    let cols =
        |r: &FitResult| term_columns(r.term_table_type3_lrt.as_ref().unwrap().as_ref().unwrap());
    let (auto, asked) = (cols(&fits[0]), cols(&fits[2]));
    assert_eq!((auto.test, auto.reason), ("lrt", "poisson_mixed"));
    assert_eq!((asked.test, asked.reason), ("lrt", "requested"));
    let rows = |c: &TermColumns| format!("{:?}", (&c.label, &c.statistic, &c.p_value, &c.notes));
    assert_eq!(rows(&auto), rows(&asked));
}

/// `compare()`'s two fits are made as `"Wald"` makes them: on a Poisson
/// GLMM pair, whose `"auto"` is still the likelihood-ratio test, neither
/// stores a likelihood-ratio table.
#[test]
fn run_compare_makes_its_fits_as_wald() {
    let (numeric, factor) = sim("glmm_poisson");
    let r = run_compare(
        "y ~ 1 + (1 | g)",
        "y ~ x + (1 | g)",
        numeric,
        factor,
        Family::Poisson {
            link: PoissonLink::Log,
        },
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        None,
    )
    .expect("compares");
    for f in [&r.fit0, &r.fit1] {
        assert_eq!(choice_tags(&f.term_test_auto), ("lrt", "poisson_mixed"));
        assert!(f.term_table_type3_lrt.is_none() && f.term_table_type2_lrt.is_none());
    }
}

/// The four estimator cases of `run_compare`: each pair fitted the way
/// its case needs, and the comparison equal, bit for bit, to the two-fit
/// `compare` on the same two fits made directly.
#[test]
fn run_compare_picks_the_estimator_from_what_differs() {
    let gaussian = Family::Gaussian;
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    // (f0, f1, data, family, the fits' `reml` argument, `Fit::reml`)
    let cases = [
        (
            "y ~ g * h + x",
            "y ~ g * h + x + (1 | subj)",
            "sim_factorial",
            gaussian,
            Some(true),
            true,
        ),
        (
            "y ~ g * h + x + (1 | subj)",
            "y ~ g + h + x + (1 | subj)",
            "sim_factorial",
            gaussian,
            Some(false),
            false,
        ),
        (
            "y ~ g + h + x + (1 | subj)",
            "y ~ g * h + x + (1 + x | subj)",
            "sim_factorial",
            gaussian,
            Some(false),
            false,
        ),
        (
            "y ~ g * h + x + (1 | subj)",
            "y ~ g * h + x",
            "sim_factorial_bern",
            binomial,
            None,
            false,
        ),
    ];
    for (f0, f1, data, fam, reml, fit_reml) in cases {
        let (family, link) = if fam == gaussian {
            ("gaussian", "identity")
        } else {
            ("binomial", "logit")
        };
        let (numeric, factor) = sim(data);
        let got = run_compare(
            f0,
            f1,
            numeric,
            factor,
            family_from_str(family, link).unwrap(),
            FitOptions {
                condvar: true,
                ..FitOptions::default()
            },
            None,
            None,
            None,
            None,
        )
        .unwrap_or_else(|e| panic!("{f0} vs {f1}: {e}"));
        for fit in [&got.fit0, &got.fit1] {
            assert_eq!(fit.reml, fit_reml, "{f0} vs {f1}");
            assert!(!fit.reml_warning);
            // satterthwaite = true: a Gaussian mixed side's table has its F rows.
            if fam == gaussian && !fit.varcorr.is_empty() {
                let cols = term_columns(fit.term_table_type3.as_ref().expect("a table"));
                assert!(cols.den_df_kind.iter().all(|k| *k == "satterthwaite"));
            }
        }
        let (lo0, opts0, fit0) = direct(f0, &sim(data), fam, 1, reml, None);
        let (lo1, opts1, fit1) = direct(f1, &sim(data), fam, 1, reml, None);
        let want = compare_stored(
            &Model::from_fit(f0, &lo0, &opts0, &fit0),
            &Model::from_fit(f1, &lo1, &opts1, &fit1),
            &compare_options(None, None).unwrap(),
        )
        .expect("comparable");
        assert_eq!(
            format!("{:?}", got.comparison),
            format!("{want:?}"),
            "{f0} vs {f1}"
        );
    }
}

/// The formulas swapped give the same test; the fits follow the
/// formulas as given.
#[test]
fn run_compare_takes_the_formulas_in_either_order() {
    let (f0, f1) = (
        "y ~ g + h + x + (1 | subj)",
        "y ~ g * h + x + (1 + x | subj)",
    );
    let go = |a: &str, b: &str| {
        let (numeric, factor) = sim("sim_factorial");
        run_compare(
            a,
            b,
            numeric,
            factor,
            Family::Gaussian,
            FitOptions {
                condvar: true,
                ..FitOptions::default()
            },
            None,
            None,
            None,
            None,
        )
        .expect("compares")
    };
    let (ab, ba) = (go(f0, f1), go(f1, f0));
    assert_eq!(
        format!("{:?}", ab.comparison.test),
        format!("{:?}", ba.comparison.test)
    );
    assert_eq!(ab.comparison.larger, Side::Second);
    assert_eq!(ba.comparison.larger, Side::First);
    assert_eq!(ba.comparison.m0.formula_label, f1);
    assert_eq!(kernel_fields(&ab.fit0), kernel_fields(&ba.fit1));
    assert_eq!(kernel_fields(&ab.fit1), kernel_fields(&ba.fit0));
}

/// Formulas that are not nested, or have the same fixed and random parts,
/// refuse before either fit (an unknown column would be the fit's error);
/// formulas that differ only by an `offset()` term have the same parts, but
/// the refusal names the offsets, as `compare` does; a fit that does not
/// converge refuses the comparison.
#[test]
fn run_compare_refuses_unnested_or_equal_parts_and_an_unconverged_fit() {
    let not_nested = ("y ~ g + nosuch", "y ~ h");
    for (f0, f1, want) in [
        (
            not_nested.0,
            not_nested.1,
            CompareError::NotNested {
                m0: not_nested.0.to_string(),
                m1: not_nested.1.to_string(),
            },
        ),
        (
            "y ~ g + nosuch + (1 | subj)",
            "y ~ nosuch + g + (1 | subj)",
            CompareError::SameParts,
        ),
        (
            "y ~ g + offset(x) + (1 | subj)",
            "y ~ g + (1 | subj)",
            CompareError::DifferentOffset,
        ),
    ] {
        let (numeric, factor) = sim("sim_factorial");
        let err = run_compare(
            f0,
            f1,
            numeric,
            factor,
            Family::Gaussian,
            FitOptions {
                condvar: true,
                ..FitOptions::default()
            },
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err, want.to_string(), "{f0} against {f1}");
    }

    let (numeric, factor) = three_row_lmm();
    let err = run_compare(
        "y ~ 1 + (1 | g)",
        "y ~ x1 + x2 + (1 | g)",
        numeric,
        factor,
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        None,
    )
    .unwrap_err();
    assert_eq!(
        err,
        CompareError::NotConverged {
            m0: false,
            m1: true
        }
        .to_string()
    );
}

/// A `cbind()` response sets the weights without tripping the
/// double specification, and `nagq` is stripped per model as `run_fit`
/// strips it: `Agq{5}` against `Laplace`, with the node-count note.
#[test]
fn run_compare_strips_nagq_per_model_and_reads_a_cbind_response() {
    let (numeric, mut factor) = cbpp_columns();
    let blk: Vec<String> = factor["herd"]
        .1
        .iter()
        .map(|c| format!("b{}", c % 3))
        .collect();
    let blk: Vec<&str> = blk.iter().map(String::as_str).collect();
    factor.insert("blk".to_string(), factor_col(&blk));
    let f0 = "cbind(incidence, healthy) ~ period + (1 | herd)";
    let f1 = "cbind(incidence, healthy) ~ period + (1 | herd) + (1 | blk)";
    let got = run_compare(
        f0,
        f1,
        numeric,
        factor,
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        FitOptions {
            nagq: 5,
            condvar: true,
            ..FitOptions::default()
        },
        None,
        None,
        None,
        None,
    )
    .expect("compares");
    assert_eq!(got.fit0.objective, Objective::Agq { nodes: 5 });
    assert!(!got.fit0.agq_warning);
    assert_eq!(got.fit1.objective, Objective::Laplace);
    assert!(got.fit1.agq_warning);
    assert!(got.fit0.weights.is_some());
    let ComparisonTest::Lr(t) = &got.comparison.test else {
        panic!("an LR test")
    };
    assert!(
        t.notes.contains(&LrNote::NodeCounts { m0: 5, m1: 1 }),
        "{:?}",
        t.notes
    );
}

/// `seed = None` takes `DEFAULT_SEED`: the same p-value as passing it.
/// The result carries `nsim` and the seed.
#[test]
fn run_compare_defaults_the_seed_to_the_constant() {
    let (f0, f1) = ("y ~ g * h + x", "y ~ g * h + x + (1 | subj)");
    let go = |seed: Option<u64>| {
        let (numeric, factor) = sim("sim_factorial");
        run_compare(
            f0,
            f1,
            numeric,
            factor,
            Family::Gaussian,
            FitOptions {
                condvar: true,
                ..FitOptions::default()
            },
            None,
            None,
            Some(999),
            seed,
        )
        .expect("compares")
    };
    let p = |r: &CompareResult| match &r.comparison.test {
        ComparisonTest::Lr(t) => (t.p_value, t.reference),
        other => panic!("{other:?}"),
    };
    let (default, given) = (go(None), go(Some(DEFAULT_SEED)));
    assert_eq!(p(&default).0.to_bits(), p(&given).0.to_bits());
    assert_eq!((default.nsim, default.seed), (999, DEFAULT_SEED));
    assert_eq!(
        p(&default).1,
        Reference::ExactRlrt {
            nsim: 999,
            seed: DEFAULT_SEED
        }
    );
}

/// `x` and `y` are the same `f64`, a NaN equal to any NaN.
pub(super) fn same(x: f64, y: f64) -> bool {
    x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
}

/// `term_columns` puts each `FixedRow` field in its column, row by row,
/// on an F table with Satterthwaite df, an F table with the residual df,
/// a Wald χ² table, and a table with row notes and a table note.
#[test]
fn term_columns_flattens_each_row_into_its_columns() {
    let gaussian = Family::Gaussian;
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let lmm = "y ~ g * h + x + (1 + x | subj)";
    // `x2` is `x` doubled: its column is aliased, its row untested.
    let (mut numeric, factor) = sim("sim_factorial");
    let x2: Vec<f64> = numeric["x"].iter().map(|v| 2.0 * v).collect();
    numeric.insert("x2".to_string(), x2);
    let aliased = (numeric, factor);
    // (formula, data, family, ss type, each row's kind and den_df_kind)
    let cases = [
        (
            lmm,
            "sim_factorial",
            gaussian,
            SsType::Type3,
            "F",
            "satterthwaite",
        ),
        (
            "y ~ g * h + x",
            "sim_factorial",
            gaussian,
            SsType::Type2,
            "F",
            "residual",
        ),
        (
            "y ~ g * h + x + (1 | subj)",
            "sim_factorial_bern",
            binomial,
            SsType::Type3,
            "chisq",
            "",
        ),
        (
            lmm,
            "sim_factorial_empty",
            gaussian,
            SsType::Type3,
            "F",
            "satterthwaite",
        ),
        (
            "y ~ g * h + x + x2",
            "",
            gaussian,
            SsType::Type3,
            "F",
            "residual",
        ),
    ];
    let (mut multi_df_f_rows, mut row_notes, mut table_notes) = (0, 0, 0);
    for (formula, data, fam, ss_type, kind, den_df_kind) in cases {
        let columns = if data.is_empty() {
            aliased.clone()
        } else {
            sim(data)
        };
        let (lo, fit_opts, fit) = direct(formula, &columns, fam, 1, None, None);
        let opts = TermTestOptions {
            ss_type,
            test: TermTest::Auto,
        };
        let t = term_tests_cached(&lo, &fit_opts, &fit, &opts, &mut RefitCache::default())
            .expect("a table");
        let c = term_columns(&t);
        let reason = if kind == "F" {
            "f_test"
        } else {
            "binomial_mixed"
        };
        assert_eq!((c.test, c.reason), (kind, reason), "{formula} on {data}");
        let n = t.fixed.len();
        assert!(n > 0, "{formula} on {data}");
        let ss = if ss_type == SsType::Type3 {
            "III"
        } else {
            "II"
        };
        assert_eq!(c.ss_type, ss);
        let lens = [
            c.label.len(),
            c.kind.len(),
            c.statistic.len(),
            c.num_df.len(),
            c.den_df.len(),
            c.den_df_kind.len(),
            c.p_value.len(),
            c.sum_sq.len(),
            c.mean_sq.len(),
            c.eta2_partial.len(),
            c.row_notes.len(),
        ];
        assert_eq!(lens, [n; 11], "{formula} on {data}");
        for (i, row) in t.fixed.iter().enumerate() {
            let at = format!("{formula} on {data}, row {i}");
            assert_eq!(c.label[i], row.label, "{at}");
            assert_eq!(c.kind[i], kind, "{at}");
            assert_eq!(c.den_df_kind[i], den_df_kind, "{at}");
            assert!(same(c.statistic[i], row.test.statistic), "{at}");
            assert!(same(c.num_df[i], row.test.num_df), "{at}");
            assert!(same(c.p_value[i], row.test.p_value), "{at}");
            assert!(same(c.eta2_partial[i], row.eta2_partial()), "{at}");
            match row.test.reference {
                Reference::F { den_df, .. } => {
                    assert_eq!(kind, "F", "{at}");
                    assert!(same(c.den_df[i], den_df), "{at}");
                }
                _ => {
                    assert_eq!(kind, "chisq", "{at}");
                    assert!(c.den_df[i].is_nan(), "{at}");
                }
            }
            match (row.sum_sq, row.mean_sq) {
                (Some(ss), Some(ms)) => {
                    assert!(same(c.sum_sq[i], ss) && same(c.mean_sq[i], ms), "{at}");
                    multi_df_f_rows += usize::from(row.test.num_df > 1.0 && ss != ms);
                }
                (None, None) => {
                    assert!(c.sum_sq[i].is_nan() && c.mean_sq[i].is_nan(), "{at}");
                }
                other => panic!("{at}: {other:?}"),
            }
            let lines: Vec<String> = row
                .test
                .notes
                .iter()
                .map(|note| linear_note_line(*note, &t.coef_names))
                .collect();
            assert_eq!(c.row_notes[i], lines, "{at}");
            row_notes += lines.len();
        }
        assert_eq!(c.notes, t.notes, "{formula} on {data}");
        table_notes += t.notes.len();
    }
    // What the cases hold: an F row on more than one df (its Sum Sq is
    // not its Mean Sq), a row note and a table note.
    assert!(
        multi_df_f_rows > 0 && row_notes > 0 && table_notes > 0,
        "{multi_df_f_rows} multi-df F rows, {row_notes} row notes, {table_notes} table notes"
    );
}

/// `nsim` from 1 to `MAX_NSIM` is taken; 0 and `MAX_NSIM + 1` are refused.
#[test]
fn compare_options_takes_nsim_from_1_to_max_nsim() {
    assert_eq!(
        compare_options(None, None),
        Ok(CompareOptions {
            nsim: DEFAULT_NSIM,
            seed: DEFAULT_SEED
        })
    );
    for nsim in [1, MAX_NSIM] {
        assert_eq!(
            compare_options(Some(nsim), Some(7)),
            Ok(CompareOptions { nsim, seed: 7 })
        );
    }
    for nsim in [0, MAX_NSIM + 1] {
        assert_eq!(
            compare_options(Some(nsim), None),
            Err(format!(
                "nsim must be a whole number from 1 to {MAX_NSIM}, got {nsim}"
            ))
        );
    }
}

/// `run_compare` on `sim_factorial` with the weights `1 + (i mod 7)/2`.
fn weighted_compare(
    f0: &str,
    f1: &str,
    weight_power: Option<WeightPower>,
) -> Result<CompareResult, PortError> {
    let (numeric, factor) = sim("sim_factorial");
    let w = (0..numeric["y"].len())
        .map(|i| 1.0 + (i % 7) as f64 / 2.0)
        .collect();
    run_compare(
        f0,
        f1,
        numeric,
        factor,
        Family::Gaussian,
        FitOptions {
            condvar: true,
            weight_power,
            ..FitOptions::default()
        },
        Some(w),
        None,
        None,
        None,
    )
}

/// With weights and a shape both models take, `None` fits the
/// power by ML on both sides when the fixed parts differ, and the test is
/// the likelihood-ratio test on the fixed columns; a named power is used
/// as given; a second grouping factor on one side resolves `None` to the
/// weights as given on both. Only the fitted power notes the test's
/// liberalness.
#[test]
fn run_compare_fits_the_weight_power_by_ml_on_both_sides() {
    for (f0, f1) in [
        ("y ~ x", "y ~ x + g"),
        ("y ~ x + (1 | subj)", "y ~ x + g + (1 | subj)"),
    ] {
        let r = weighted_compare(f0, f1, None).expect("compares");
        for f in [&r.fit0, &r.fit1] {
            assert!(f.weight_power_estimated && !f.reml, "{f1}");
            assert_eq!(f.se_type, SeType::Model);
        }
        let ComparisonTest::Lr(t) = &r.comparison.test else {
            panic!("{f1}: not a likelihood-ratio test")
        };
        assert_eq!(t.df, r.fit1.beta.len() - r.fit0.beta.len());
        assert_eq!(r.comparison.m1.npar, r.fit1.df);
        assert!(t.notes.contains(&LrNote::EstimatedWeightPower), "{f1}");
        let claimed = weighted_compare(f0, f1, Some(WeightPower::Fixed(1.0))).unwrap();
        for f in [&claimed.fit0, &claimed.fit1] {
            assert!(!f.weight_power_estimated && f.weight_power == 1.0, "{f1}");
        }
        if let ComparisonTest::Lr(t) = &claimed.comparison.test {
            assert!(!t.notes.contains(&LrNote::EstimatedWeightPower), "{f1}");
        }
    }
    let crossed = weighted_compare(
        "y ~ x + (1 | subj)",
        "y ~ x + h + (1 | subj) + (1 | g)",
        None,
    )
    .expect("compares");
    for f in [&crossed.fit0, &crossed.fit1] {
        assert!(!f.weight_power_estimated && f.weight_power == 1.0);
    }
}

/// With the same fixed parts a resolved or named fitted power is
/// refused before either fit; the weights as given are compared.
#[test]
fn run_compare_refuses_fitted_powers_on_a_random_part_alone() {
    let refusal = CompareError::WeightPowerMismatch(WeightPowerMismatch::DeltaRandomPartOnly);
    for power in [None, Some(WeightPower::Estimate)] {
        let err = weighted_compare("y ~ x", "y ~ x + (1 | subj)", power);
        assert_eq!(err.unwrap_err(), refusal.to_string());
    }
    let claimed = weighted_compare("y ~ x", "y ~ x + (1 | subj)", Some(WeightPower::Fixed(1.0)));
    assert!(claimed.is_ok());
}

/// `anova_test = "LRT"` on a δ fit stores the refusal of its
/// likelihood-ratio tables (a Gaussian fit's rows are F tests), and the
/// profile intervals refuse a fitted power and a fixed power other than 1.
#[test]
fn a_delta_fit_refuses_lrt_tables_and_profile_intervals() {
    let (numeric, factor) = sim("sim_factorial");
    let w: Vec<f64> = (0..numeric["y"].len())
        .map(|i| 1.0 + (i % 7) as f64 / 2.0)
        .collect();
    let r = run_fit(
        "y ~ x + g + (1 | subj)",
        numeric.clone(),
        factor.clone(),
        Family::Gaussian,
        FitOptions {
            condvar: true,
            ..FitOptions::default()
        },
        Some(w.clone()),
        None,
        None,
        TermTest::Lrt,
    )
    .unwrap();
    assert!(r.weight_power_estimated);
    for t in [&r.term_table_type3_lrt, &r.term_table_type2_lrt] {
        assert!(
            matches!(t, Some(Err(TermTableError::Refused(e))) if e.contains("F tests")),
            "{t:?}"
        );
    }
    for (power, estimated) in [(r.weight_power, true), (0.5, false)] {
        let err = profile_intervals(
            "y ~ x + g + (1 | subj)",
            numeric.clone(),
            factor.clone(),
            Family::Gaussian,
            Some(w.clone()),
            None,
            true,
            power,
            estimated,
            0.95,
            None,
        )
        .unwrap_err();
        assert!(
            err.contains("weight power") || err.contains("weight_power"),
            "{err}"
        );
    }
}

/// A panic while a per-term table is built, a likelihood-ratio table
/// included, is stored as that table's internal error, never as a
/// refusal, and the fit stands.
#[test]
fn a_panic_in_the_term_tables_is_stored_as_an_internal_error() {
    let data = table(&sim("sim_factorial_bern"));
    let prepared = lower_with(
        "y ~ g + x",
        &data,
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        FitOptions::default(),
        None,
        None,
    )
    .expect("lowers");
    let mut fit = fit_lowered(&prepared.lowered, &prepared.opts, None).expect("fits");
    // `term_tests` reads `aliased` at each column, past its end now.
    fit.diagnostics.aliased.clear();
    let r = flatten(prepared, fit, TermTest::Lrt).expect("the fit stands");
    let lrt =
        [&r.term_table_type3_lrt, &r.term_table_type2_lrt].map(|t| t.as_ref().expect("asked for"));
    for t in [&r.term_table_type3, &r.term_table_type2]
        .into_iter()
        .chain(lrt)
    {
        let Err(TermTableError::Internal(err)) = t else {
            panic!("{t:?}")
        };
        assert!(
            err.starts_with(
                "internal error while building the per-term table (please report it): "
            ),
            "{err}"
        );
    }
}
