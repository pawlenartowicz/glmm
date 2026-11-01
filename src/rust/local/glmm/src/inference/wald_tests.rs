//! The Wald χ² against the frozen `kind = "anova"` references, recomputed
//! from each fit's frozen β̂, vcov and `L`: car's χ² tables on the binomial
//! GLM and GLMM, `W/r` against car's F on the OLS twin and against
//! `linearHypothesis`'s F on the Gamma GLM; the rank rule and the aliased
//! columns, which the references do not cover; and the `ContrastF` wrapper.

use serde_json::Value;

use super::{wald_chisq, LinearKind, LinearNote, LinearTest};
use crate::inference::contrast::{ContrastF, ContrastNote};
use crate::inference::tails::f_upper_p;
use crate::inference::test_refs::{self, Kind};
use crate::inference::{DenDf, Reference};

/// Statistics, relative to the frozen value.
const STAT_REL: f64 = 1e-12;
/// p-values, relative in log space: `|ln p − ln p_ref| ≤ P_LOG_REL·max(1, |ln p_ref|)`.
const P_LOG_REL: f64 = 1e-10;

fn assert_stat(got: f64, want: f64, what: &str) {
    let band = STAT_REL * want.abs();
    assert!(
        (got - want).abs() <= band,
        "{what}: statistic {got} vs frozen {want} (rel {:e})",
        (got - want).abs() / want.abs()
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

fn vector(v: &Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what}: not an array"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("{what}: not a number")))
        .collect()
}

fn matrix(v: &Value, what: &str) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what}: not an array"))
        .iter()
        .map(|row| vector(row, what))
        .collect()
}

/// A frozen fit's `(β̂, V)`.
fn fit_numbers(fit: &Value, what: &str) -> (Vec<f64>, Vec<Vec<f64>>) {
    (vector(&fit["coef"], what), matrix(&fit["vcov"], what))
}

/// Every row of the frozen car tables of `name` (Type II, Type III on the
/// treatment fit, Type III on the `contr.sum` fit), each with the Wald χ²
/// recomputed from its fit's β̂ and vcov and its `L`: `(what, test, row)`.
fn car_rows(name: &str) -> Vec<(String, LinearTest, Value)> {
    let d = test_refs::load(Kind::Anova, name);
    let mut out = Vec::new();
    for table in ["type2", "type3", "type3_sum"] {
        let t = &d[table];
        let fit_key = t["fit"].as_str().expect("`fit` names the fit");
        let (beta, vcov) = fit_numbers(&d[fit_key], name);
        let p = beta.len();
        for row in t["rows"].as_array().expect("`rows` is an array") {
            let what = format!("{name} {table} {}", row["term"]);
            let l: Vec<f64> = matrix(&row["L"], &what).concat();
            let test = wald_chisq(&beta, &vcov, &vec![false; p], &l, l.len() / p);
            out.push((what, test, row.clone()));
        }
    }
    out
}

#[test]
fn wald_chisq_matches_car_anova_on_glm_and_glmm() {
    for name in [
        "anova_car_glm_sim_factorial_bern",
        "anova_car_glmm_sim_factorial_bern",
    ] {
        let rows = car_rows(name);
        assert_eq!(rows.len(), 14, "{name}: 4 + 5 + 5 rows");
        for (what, t, row) in rows {
            let df = num(&row, "df", &what);
            assert_eq!(t.kind, LinearKind::ChiSquared, "{what}");
            assert_eq!(t.num_df, df, "{what}");
            assert_eq!(
                t.reference,
                Reference::ChiSquared { df: df as usize },
                "{what}"
            );
            assert!(t.notes.is_empty(), "{what}: {:?}", t.notes);
            assert_stat(t.statistic, num(&row, "statistic", &what), &what);
            assert_p(t.p_value, num(&row, "p", &what), &what);
        }
    }
}

#[test]
fn wald_over_rank_matches_car_anova_f_on_ols() {
    let rows = car_rows("anova_car_lm_sim_factorial");
    assert_eq!(rows.len(), 14);
    for (what, t, row) in rows {
        let df = num(&row, "df", &what);
        assert_eq!(t.num_df, df, "{what}");
        let f = t.statistic / t.num_df;
        assert_stat(f, num(&row, "statistic", &what), &what);
        let p = f_upper_p(f, df, num(&row, "res_df", &what));
        assert_p(p, num(&row, "p", &what), &what);
    }
}

#[test]
fn wald_over_rank_matches_linear_hypothesis_f_on_gamma_glm() {
    let name = "anova_linear_hypothesis_glm_gamma";
    let d = test_refs::load(Kind::Anova, name);
    let (beta, vcov) = fit_numbers(&d, name);
    let l = matrix(&d["L"], name).concat();
    let t = wald_chisq(&beta, &vcov, &[false; 3], &l, 1);
    assert_eq!(t.num_df, num(&d, "num_df", name));
    let f = t.statistic / t.num_df;
    assert_stat(f, num(&d, "F", name), name);
    let p = f_upper_p(f, t.num_df, num(&d, "den_df", name));
    assert_p(p, num(&d, "p", name), name);
}

/// A duplicated row of `L` adds no direction: `L V Lᵀ` is singular, the
/// rank is the number of distinct directions and the statistic is the
/// unduplicated one.
#[test]
fn duplicated_row_is_tested_on_its_rank() {
    let name = "anova_car_glm_sim_factorial_bern";
    let d = test_refs::load(Kind::Anova, name);
    let (beta, vcov) = fit_numbers(&d["treatment"], name);
    let row = &d["type2"]["rows"][0];
    assert_eq!(row["term"], "g");
    let mut l = matrix(&row["L"], name);
    assert_eq!(l.len(), 2);
    l.push(l[0].clone());
    let t = wald_chisq(&beta, &vcov, &[false; 7], &l.concat(), 3);
    assert_eq!(t.num_df, 2.0);
    assert_eq!(t.reference, Reference::ChiSquared { df: 2 });
    assert_eq!(
        t.notes,
        vec![LinearNote::RankDeficientL { rank: 2, rows: 3 }]
    );
    assert_stat(t.statistic, num(row, "statistic", name), name);
    assert_p(t.p_value, num(row, "p", name), name);
}

/// The rank threshold, `1e-8` of `λ_max`, against `g`'s two Type II rows
/// `row0`, `row1` (`λ_min/λ_max` near 0.2). A third row `row0 + 1e-6·e`,
/// `e` the `gb:hq` column (outside the span of `g`'s rows), adds an
/// eigenvalue near `1e-12·λ_max`, below the threshold: rank 2 of 3. The rows
/// `row0`, `row0 + 1e-2·row1` span `g`'s rows with `λ_min/λ_max` near
/// `1.6e-5`, above it: rank 2, the statistic `g`'s.
#[test]
fn near_duplicate_rows_split_at_the_rank_threshold() {
    let name = "anova_car_glm_sim_factorial_bern";
    let d = test_refs::load(Kind::Anova, name);
    let (beta, vcov) = fit_numbers(&d["treatment"], name);
    let row = &d["type2"]["rows"][0];
    assert_eq!(row["term"], "g");
    let l = matrix(&row["L"], name);
    let (row0, row1) = (&l[0], &l[1]);
    let e = d["treatment"]["coef_names"]
        .as_array()
        .unwrap()
        .iter()
        .position(|c| c == "gb:hq")
        .expect("a gb:hq column");

    let mut third = row0.clone();
    third[e] += 1e-6;
    let near: Vec<f64> = [row0.clone(), row1.clone(), third].concat();
    let t = wald_chisq(&beta, &vcov, &[false; 7], &near, 3);
    assert_eq!(t.num_df, 2.0);
    assert_eq!(t.reference, Reference::ChiSquared { df: 2 });
    assert_eq!(
        t.notes,
        vec![LinearNote::RankDeficientL { rank: 2, rows: 3 }]
    );

    let close: Vec<f64> = row0.iter().zip(row1).map(|(a, b)| a + 1e-2 * b).collect();
    let t = wald_chisq(
        &beta,
        &vcov,
        &[false; 7],
        &[row0.clone(), close].concat(),
        2,
    );
    assert_eq!(t.num_df, 2.0);
    assert!(t.notes.is_empty(), "{:?}", t.notes);
    // `L V Lᵀ` has condition near 6e4 here, so the band is wider than
    // `STAT_REL`.
    let want = num(row, "statistic", name);
    assert!(
        (t.statistic - want).abs() <= 1e-10 * want,
        "{name}: statistic {} vs frozen {want}",
        t.statistic
    );
}

/// A non-finite `V` entry among the columns tested: not tested, no note.
/// `L` zero on every column tested: rank 0, statistic 0 and a NaN p-value.
#[test]
fn nonfinite_vcov_and_rank_zero_outcomes() {
    let name = "anova_linear_hypothesis_glm_gamma";
    let d = test_refs::load(Kind::Anova, name);
    let (beta, mut vcov) = fit_numbers(&d, name);
    let t = wald_chisq(&beta, &vcov, &[false; 3], &[0.0; 3], 1);
    assert_eq!(t.statistic, 0.0);
    assert_eq!(t.num_df, 0.0);
    assert_eq!(t.reference, Reference::ChiSquared { df: 0 });
    assert!(t.p_value.is_nan());
    assert_eq!(
        t.notes,
        vec![LinearNote::RankDeficientL { rank: 0, rows: 1 }]
    );

    vcov[1][1] = f64::INFINITY;
    let t = wald_chisq(&beta, &vcov, &[false; 3], &[0.0, 1.0, 0.0], 1);
    assert!(t.statistic.is_nan() && t.num_df.is_nan() && t.p_value.is_nan());
    assert_eq!(t.reference, Reference::ChiSquared { df: 0 });
    assert!(t.notes.is_empty(), "{:?}", t.notes);
}

/// An aliased column's β̂ and vcov entries are NaN on a fit and are never
/// read: `L` with a non-zero entry there is not tested, and `L` zero there
/// is tested on the other columns.
#[test]
fn aliased_column_with_a_nonzero_entry_is_not_tested() {
    let name = "anova_linear_hypothesis_glm_gamma";
    let d = test_refs::load(Kind::Anova, name);
    let (mut beta, mut vcov) = fit_numbers(&d, name);
    let want = beta[1] * beta[1] / vcov[1][1];
    beta[2] = f64::NAN;
    for (i, row) in vcov.iter_mut().enumerate() {
        row[2] = f64::NAN;
        if i == 2 {
            row.fill(f64::NAN);
        }
    }
    let aliased = [false, false, true];

    let t = wald_chisq(&beta, &vcov, &aliased, &[0.0, 1.0, 1.0], 1);
    assert!(t.statistic.is_nan() && t.num_df.is_nan() && t.p_value.is_nan());
    assert_eq!(t.reference, Reference::ChiSquared { df: 0 });
    assert_eq!(t.notes, vec![LinearNote::AliasedColumn { column: 2 }]);

    let t = wald_chisq(&beta, &vcov, &aliased, &[0.0, 1.0, 0.0], 1);
    assert_eq!(t.num_df, 1.0);
    assert!(t.notes.is_empty());
    assert_stat(t.statistic, want, "column 1 alone");
}

#[test]
fn contrast_f_wrapper_copies_the_test_and_maps_the_notes() {
    let cf = ContrastF {
        f: 3.5,
        num_df: 2.0,
        den_df: 17.25,
        p_value: 0.0548,
        notes: vec![ContrastNote::RankDeficientL { rank: 2, rows: 3 }],
    };
    for den_df_kind in [DenDf::Satterthwaite, DenDf::Residual] {
        let t = LinearTest::from_contrast_f(cf.clone(), den_df_kind);
        assert_eq!(
            t,
            LinearTest {
                statistic: 3.5,
                kind: LinearKind::F,
                num_df: 2.0,
                reference: Reference::F {
                    num_df: 2.0,
                    den_df: 17.25,
                    den_df_kind,
                },
                p_value: 0.0548,
                notes: vec![LinearNote::RankDeficientL { rank: 2, rows: 3 }],
            }
        );
    }

    let aliased = ContrastF {
        f: f64::NAN,
        num_df: f64::NAN,
        den_df: f64::NAN,
        p_value: f64::NAN,
        notes: vec![ContrastNote::AliasedColumn { column: 4 }],
    };
    let t = LinearTest::from_contrast_f(aliased, DenDf::Residual);
    assert!(t.statistic.is_nan() && t.num_df.is_nan() && t.p_value.is_nan());
    assert_eq!(t.kind, LinearKind::F);
    assert_eq!(t.notes, vec![LinearNote::AliasedColumn { column: 4 }]);
}
