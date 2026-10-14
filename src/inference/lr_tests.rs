//! The likelihood-ratio arithmetic against the frozen `kind = "anova"`
//! references: every pair's statistic and p-values recomputed from its frozen
//! log-likelihoods and parameter counts, and the 50:50 mixture on the frozen
//! grid of points; plus the floor, `T = 0`, NaN and `−∞` rules, which the
//! references do not cover.

use std::collections::BTreeSet;

use serde_json::Value;

use super::{chibar_upper_p, lr_p_value, lr_statistic, LrNote, NEGATIVE_LR_ROUNDOFF};
use crate::inference::test_refs::{self, Kind};
use crate::inference::{DenDf, Reference};

/// Statistics, relative to the frozen value.
const STAT_REL: f64 = 1e-12;
/// p-values, relative in log space: `|ln p − ln p_ref| ≤ P_LOG_REL·max(1, |ln p_ref|)`.
const P_LOG_REL: f64 = 1e-10;

/// Half a unit in the 15th significant digit of `x`: how far the JSON's
/// 15-digit rounding may have moved a stored value.
fn stored_rounding(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        0.5 * 10f64.powi(x.abs().log10().floor() as i32 - 14)
    }
}

/// The statistic recomputed from the two stored log-likelihoods carries
/// their rounding too, which on a small statistic is more than `STAT_REL` of
/// it, so that rounding widens the band.
fn assert_stat(got: f64, want: f64, ll0: f64, ll1: f64, what: &str) {
    let band = STAT_REL * want.abs()
        + 2.0 * (stored_rounding(ll0) + stored_rounding(ll1))
        + stored_rounding(want);
    assert!(
        (got - want).abs() <= band,
        "{what}: statistic {got} vs frozen {want} (band {band:e})"
    );
}

fn assert_p(got: f64, want: f64, what: &str) {
    assert!(
        (f64::MIN_POSITIVE..=1.0).contains(&want),
        "{what}: frozen p {want} is outside the gated range"
    );
    let (lg, lw) = (got.ln(), want.ln());
    let band = P_LOG_REL * lw.abs().max(1.0);
    assert!(
        (lg - lw).abs() <= band,
        "{what}: p {got:e} vs frozen {want:e} (ln diff {:e}, band {band:e})",
        (lg - lw).abs()
    );
}

fn num(v: &Value, key: &str, what: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("{what}: `{key}` is not a number"))
}

fn count(v: &Value, key: &str, what: &str) -> usize {
    v[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{what}: `{key}` is not a count")) as usize
}

/// One frozen comparison: the smaller and the larger side's `logLik` and
/// `npar`, and the frozen statistic, df and p-values.
fn check_pair(
    what: &str,
    small: &Value,
    large: &Value,
    stat_key: &str,
    df_key: &str,
    frozen: &Value,
) {
    let (ll0, ll1) = (num(small, "logLik", what), num(large, "logLik", what));
    let (t, note) = lr_statistic(ll0, ll1);
    assert_eq!(note, None, "{what}: no note on a positive statistic");
    assert_stat(t, num(frozen, stat_key, what), ll0, ll1, what);

    let df = count(large, "npar", what) - count(small, "npar", what);
    assert_eq!(df, count(frozen, df_key, what), "{what}: Δnpar");
    let p = lr_p_value(t, &Reference::ChiSquared { df });
    assert_p(p, num(frozen, "p_chisq", what), &format!("{what} χ²"));
    if frozen.get("p_mixture").is_some() {
        let df_low = count(frozen, "mixture_k", what);
        let p = lr_p_value(t, &Reference::ChiBarSquared { df_low });
        assert_p(
            p,
            num(frozen, "p_mixture", what),
            &format!("{what} mixture"),
        );
    }
}

#[test]
fn pairs_reproduce_the_frozen_statistics_and_p_values() {
    // The cases read here, by name, and the cases skipped on purpose (the
    // per-term tables, the linear hypothesis and the fit-free grid have other
    // layouts). Any other case panics, so a new one is never skipped silently.
    const PAIR_CASES: [&str; 7] = [
        "gamma_glmm_vs_glm",
        "glm_pair",
        "glmm_pair",
        "glmm_vs_glm",
        "ml_pair",
        "ols_pair",
        "random_pair",
    ];
    const SKIPPED_CASES: [&str; 7] = [
        "car_glm",
        "car_glmm",
        "car_lm",
        "linear_hypothesis",
        "lrt_terms",
        "mixture_points",
        "nonmarginal",
    ];
    let mut cases = BTreeSet::new();
    for e in test_refs::entries(Kind::Anova) {
        assert!(!e.is_pending(), "{}: pending", e.name);
        let json = test_refs::load(Kind::Anova, &e.name);
        let case = json["case"].as_str().expect("case").to_string();
        if case == "ranova" {
            // Each row drops one term from the full model.
            let rows = json["rows"].as_array().expect("rows");
            for (i, row) in rows.iter().enumerate() {
                let what = format!("{}[{i}]", e.name);
                check_pair(&what, row, &json["full"], "LRT", "Df", row);
            }
        } else if PAIR_CASES.contains(&case.as_str()) {
            // The first two members are the compared pair, smaller first.
            let m = json["members"].as_array().expect("members");
            check_pair(&e.name, &m[0], &m[1], "statistic", "df", &json);
        } else {
            assert!(
                SKIPPED_CASES.contains(&case.as_str()),
                "{}: case `{case}` is neither read here nor listed as skipped",
                e.name
            );
            continue;
        }
        cases.insert(case);
    }
    let expected: BTreeSet<String> = PAIR_CASES
        .into_iter()
        .chain(["ranova"])
        .map(String::from)
        .collect();
    assert_eq!(cases, expected);
}

#[test]
fn mixture_matches_the_frozen_grid() {
    let json = test_refs::load(Kind::Anova, "anova_mixture_points");
    let ts: Vec<f64> = serde_json::from_value(json["t"].clone()).expect("t");
    let ks: Vec<usize> = serde_json::from_value(json["k"].clone()).expect("k");
    let ps: Vec<Vec<f64>> = serde_json::from_value(json["p_mixture"].clone()).expect("p_mixture");
    assert_eq!(ps.len(), ks.len());
    for (&k, row) in ks.iter().zip(&ps) {
        assert_eq!(row.len(), ts.len());
        for (&t, &want) in ts.iter().zip(row) {
            let what = format!("mixture k = {k}, T = {t}");
            assert_p(chibar_upper_p(t, k), want, &what);
            let via_reference = lr_p_value(t, &Reference::ChiBarSquared { df_low: k });
            assert_eq!(
                via_reference.to_bits(),
                chibar_upper_p(t, k).to_bits(),
                "{what}"
            );
        }
    }
}

#[test]
fn zero_statistic_gives_p_one() {
    assert_eq!(lr_statistic(-12.5, -12.5), (0.0, None));
    for k in 0..5 {
        assert_eq!(chibar_upper_p(0.0, k), 1.0, "mixture df_low = {k}");
        assert_eq!(
            lr_p_value(0.0, &Reference::ChiSquared { df: k + 1 }),
            1.0,
            "χ²({})",
            k + 1
        );
    }
}

#[test]
fn negative_raw_value_is_floored_with_a_note_beyond_round_off() {
    assert_eq!(NEGATIVE_LR_ROUNDOFF, 8e-5);
    let ll0 = -250.0;

    // Within round-off: 0, no note.
    let ll1 = ll0 - 0.4 * NEGATIVE_LR_ROUNDOFF;
    assert!(2.0 * (ll1 - ll0) < 0.0);
    assert_eq!(lr_statistic(ll0, ll1), (0.0, None));

    // Beyond it: 0, with a note carrying the raw value.
    let ll1 = ll0 - NEGATIVE_LR_ROUNDOFF;
    let raw = 2.0 * (ll1 - ll0);
    assert!(raw < -NEGATIVE_LR_ROUNDOFF);
    let (t, note) = lr_statistic(ll0, ll1);
    assert_eq!(t, 0.0);
    assert_eq!(note, Some(LrNote::NegativeStatistic { raw }));
    assert_eq!(lr_p_value(t, &Reference::ChiSquared { df: 1 }), 1.0);
    assert_eq!(chibar_upper_p(t, 0), 1.0);
}

#[test]
fn nan_or_a_collapsed_fit_gives_nan_never_the_floor() {
    for (ll0, ll1) in [
        (f64::NAN, -3.0),
        (-3.0, f64::NAN),
        (-3.0, f64::NEG_INFINITY),
        (f64::NEG_INFINITY, -3.0),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
    ] {
        let (t, note) = lr_statistic(ll0, ll1);
        assert!(t.is_nan(), "({ll0}, {ll1}) gave {t}");
        assert_eq!(note, None);
    }
    assert!(lr_p_value(f64::NAN, &Reference::ChiSquared { df: 2 }).is_nan());
    for k in 0..3 {
        assert!(chibar_upper_p(f64::NAN, k).is_nan(), "df_low = {k}");
    }
}

#[test]
fn exact_and_f_references_have_no_lr_p_value() {
    let exact = Reference::ExactRlrt {
        nsim: 10_000,
        seed: 1,
    };
    let f = Reference::F {
        num_df: 1.0,
        den_df: 10.0,
        den_df_kind: DenDf::Residual,
    };
    assert!(lr_p_value(3.0, &exact).is_nan());
    assert!(lr_p_value(3.0, &f).is_nan());
}
