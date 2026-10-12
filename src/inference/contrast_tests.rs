//! The stable path's Satterthwaite wiring and the contrasts on top of it:
//! `coef_df` is the pass at the fitted θ̂, the contrasts reproduce it, and
//! every route without a df says why.

use super::{ContrastError, ContrastNote, NoDfReason};
use crate::fit::common_tests::lcg;
use crate::fit::fit_cold_keep_workspace;
use crate::inference::tails::t_two_sided_p;
use crate::lmm::satterthwaite::pass;
use crate::test_support::intercept_only_spec;
use crate::{
    fit_cold, BinomialLink, Family, Fit, FitOptions, GroupIds, ModelSpec, Note, ReStructure, Sizing,
};

/// `y ~ 1 + x1 + x2 + (1 | g)`: 10 groups of 4 to 9 rows, an interior REML
/// fit whose df lie well inside `(0, N − p)`. With `dup_at = Some(k)`, a
/// fourth column `2·x1` inserted at index `k`, which the alias gate drops.
/// Returns `(x, y, ids, n, p)`.
fn lmm_data(dup_at: Option<usize>) -> (Vec<f64>, Vec<f64>, GroupIds, usize, usize) {
    let mut s = 5u64;
    let (mut x, mut y, mut g) = (vec![], vec![], vec![]);
    for grp in 0..10u32 {
        let b = 0.9 * lcg(&mut s);
        for _ in 0..(4 + grp % 6) {
            let x1 = 2.0 * lcg(&mut s);
            let x2 = lcg(&mut s);
            let mut row = vec![1.0, x1, x2];
            if let Some(k) = dup_at {
                row.insert(k, 2.0 * x1);
            }
            x.extend(row);
            y.push(0.5 + 0.7 * x1 - 0.3 * x2 + b + 0.6 * lcg(&mut s));
            g.push(grp);
        }
    }
    let n = y.len();
    let ids = GroupIds {
        primary: g,
        extra: vec![],
    };
    (x, y, ids, n, if dup_at.is_some() { 4 } else { 3 })
}

fn lmm_model() -> ModelSpec {
    intercept_only_spec(Sizing::FixedClusters { n_clusters: 1 })
}

fn all_targets(p: usize) -> FitOptions {
    FitOptions {
        target_indices: (0..p as u32).collect(),
        ..FitOptions::default()
    }
}

fn lmm_fit(dup_at: Option<usize>, opts: &FitOptions) -> Fit {
    let (x, y, ids, n, p) = lmm_data(dup_at);
    fit_cold(&x, &y, n, p, &lmm_model(), &ids, opts)
}

fn unit(p: usize, j: usize) -> Vec<f64> {
    let mut e = vec![0.0; p];
    e[j] = 1.0;
    e
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `fit_cold` fills `coef_df` with the pass at the θ̂ it fitted, bit for bit:
/// the same pass run on the workspace of the same fit, done the loop tier's
/// way.
#[test]
fn coef_df_equals_pass_at_fitted_theta() {
    let (x, y, ids, n, p) = lmm_data(None);
    let opts = all_targets(p);
    let fit = fit_cold(&x, &y, n, p, &lmm_model(), &ids, &opts);
    let (_, mut ws) = fit_cold_keep_workspace(&x, &y, n, p, &lmm_model(), &ids, &opts);
    let (lmm, map) = ws.lmm_parts().expect("a Gaussian mixed fit");
    let want = pass(lmm, map).coef_df();
    assert_eq!(bits(&fit.coef_df), bits(&want));
    let nu = (n - p) as f64;
    assert!(
        fit.coef_df.iter().all(|&d| d > 1.0 && d < nu),
        "{:?}",
        fit.coef_df
    );
    assert!(!fit
        .diagnostics
        .notes
        .iter()
        .any(|n| matches!(n, Note::SatterthwaiteCurvature { .. })));
}

/// The t test of `e_j` is coefficient `j`'s own: `β̂_j`, `se_j`, `coef_df_j`
/// (to rounding: the contest form `2v²/(gᵀAg)` against the pass's
/// `v²/(g_pᵀH_p⁺g_p + v²/ν)`), and its p on that df.
#[test]
fn contrast_t_unit_vector_equals_coef_df() {
    let p = 3;
    let fit = lmm_fit(None, &all_targets(p));
    for j in 0..p {
        let t = fit.contrast_t(&unit(p, j)).expect("a Gaussian LMM has df");
        assert_eq!(t.estimate, fit.beta[j]);
        assert!(
            rel(t.se, fit.se[j]) < 1e-12,
            "se {j}: {} {}",
            t.se,
            fit.se[j]
        );
        assert!(
            rel(t.df, fit.coef_df[j]) < 1e-12,
            "df {j}: {} {}",
            t.df,
            fit.coef_df[j]
        );
        assert_eq!(t.t, t.estimate / t.se);
        assert_eq!(t.p_value, t_two_sided_p(t.t, t.df));
        assert!(t.notes.is_empty());
    }
}

/// One row of `L` is the t test: `F = t²` on `(1, df)`, the same p.
#[test]
fn contrast_f_one_row_is_t_squared() {
    let fit = lmm_fit(None, &all_targets(3));
    let l = [0.0, 1.0, -0.5];
    let t = fit.contrast_t(&l).unwrap();
    let f = fit.contrast_f(&l, 1).unwrap();
    assert_eq!(f.f, t.t * t.t);
    assert_eq!(f.num_df, 1.0);
    assert_eq!(f.den_df, t.df);
    assert_eq!(f.p_value, t.p_value);
    assert!(f.notes.is_empty());
}

/// A non-zero entry on an aliased column is not tested: NaN fields and the
/// column named. A zero entry there tests the retained columns.
#[test]
fn contrast_t_aliased_nonzero_is_nan_with_note() {
    let fit = lmm_fit(Some(3), &all_targets(4));
    assert_eq!(fit.aliased(), [false, false, false, true]);
    let t = fit.contrast_t(&[0.0, 1.0, 0.0, 1.0]).unwrap();
    for v in [t.estimate, t.se, t.t, t.df, t.p_value] {
        assert!(v.is_nan());
    }
    assert_eq!(t.notes, [ContrastNote::AliasedColumn { column: 3 }]);
    let f = fit
        .contrast_f(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0], 2)
        .unwrap();
    for v in [f.f, f.num_df, f.den_df, f.p_value] {
        assert!(v.is_nan());
    }
    assert_eq!(f.notes, [ContrastNote::AliasedColumn { column: 3 }]);

    let ok = fit.contrast_t(&[0.0, 1.0, 0.0, 0.0]).unwrap();
    assert_eq!(ok.estimate, fit.beta[1]);
    assert!(rel(ok.df, fit.coef_df[1]) < 1e-12);
    assert!(ok.notes.is_empty());
}

/// A GLMM has no Satterthwaite df: a typed error, never a z test.
#[test]
fn contrast_on_glmm_is_no_df() {
    let mut s = 9u64;
    let (mut x, mut y, mut g) = (vec![], vec![], vec![]);
    for grp in 0..12u32 {
        let b = 0.8 * lcg(&mut s);
        for _ in 0..10 {
            let x1 = lcg(&mut s);
            let prob = 1.0 / (1.0 + (-(0.2 + 0.9 * x1 + b)).exp());
            x.extend([1.0, x1]);
            y.push(if 0.5 * (lcg(&mut s) + 1.0) < prob {
                1.0
            } else {
                0.0
            });
            g.push(grp);
        }
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: g,
        extra: vec![],
    };
    let fit = fit_cold(&x, &y, n, 2, &model, &ids, &all_targets(2));
    assert!(fit.converged());
    let no_df = ContrastError::NoDf(NoDfReason::NotGaussianMixed);
    assert_eq!(fit.contrast_t(&[0.0, 1.0]), Err(no_df));
    assert_eq!(fit.contrast_f(&[1.0, 0.0, 0.0, 1.0], 2), Err(no_df));
    assert!(fit.coef_df.iter().all(|d| d.is_nan()));
}

/// `satterthwaite = false`: no pass, the switched-off record, no note, and
/// every other field as the default fit has it.
#[test]
fn satterthwaite_false_skips_the_pass() {
    let on = lmm_fit(None, &all_targets(3));
    let off = lmm_fit(
        None,
        &FitOptions {
            satterthwaite: false,
            ..all_targets(3)
        },
    );
    assert!(off.coef_df.iter().all(|d| d.is_nan()));
    assert!(off
        .satterthwaite
        .as_ref()
        .is_some_and(|st| st.is_disabled()));
    assert_eq!(
        off.contrast_t(&unit(3, 1)),
        Err(ContrastError::NoDf(NoDfReason::Disabled))
    );
    assert!(off.diagnostics.notes.is_empty());
    assert_eq!(bits(&off.beta), bits(&on.beta));
    assert_eq!(bits(&off.se), bits(&on.se));
    assert_eq!(off.deviance.to_bits(), on.deviance.to_bits());
}

/// A Gaussian LMM `Fit` built by `FitView::into_fit` without the stable
/// path's pass carries the not-run record, on a REML and on an ML fit alike
/// (`Fit::reml` does not decide the reason).
#[test]
fn into_fit_without_pass_is_state_missing() {
    let (x, y, ids, n, p) = lmm_data(None);
    for reml in [None, Some(false)] {
        let opts = FitOptions {
            reml,
            ..all_targets(p)
        };
        let (fit, _) = fit_cold_keep_workspace(&x, &y, n, p, &lmm_model(), &ids, &opts);
        assert_eq!(fit.reml, reml.is_none(), "reml {reml:?}");
        assert!(
            fit.satterthwaite.as_ref().is_some_and(|st| st.is_not_run()),
            "reml {reml:?}: the not-run record"
        );
        assert!(fit.coef_df.iter().all(|d| d.is_nan()), "reml {reml:?}");
        assert_eq!(
            fit.contrast_t(&unit(p, 0)),
            Err(ContrastError::NoDf(NoDfReason::StateMissing)),
            "reml {reml:?}"
        );
    }
}

/// The aliased column's df is NaN; the retained columns carry the reduced
/// fit's df bit for bit.
#[test]
fn rank_deficient_coef_df_scatters() {
    let full = lmm_fit(Some(3), &all_targets(4));
    let reduced = lmm_fit(None, &all_targets(3));
    assert!(full.coef_df[3].is_nan());
    assert_eq!(bits(&full.coef_df[..3]), bits(&reduced.coef_df));
    let st = full.satterthwaite.as_ref().unwrap();
    assert_eq!(st.retained.as_deref(), Some(&[0, 1, 2][..]));
}

/// `coef_df` is NaN off the target columns, and the target's value does not
/// depend on which other columns are targets; the contrasts still cover
/// every column.
#[test]
fn coef_df_nan_outside_targets() {
    let all = lmm_fit(None, &all_targets(3));
    let one = lmm_fit(
        None,
        &FitOptions {
            target_indices: vec![1],
            ..FitOptions::default()
        },
    );
    assert!(one.coef_df[0].is_nan() && one.coef_df[2].is_nan());
    assert_eq!(one.coef_df[1].to_bits(), all.coef_df[1].to_bits());
    let t0 = one.contrast_t(&unit(3, 0)).unwrap();
    assert!(rel(t0.df, all.coef_df[0]) < 1e-12);
}

/// Two identical rows of `L` have rank 1: the F test is the one-row test on
/// that direction, with the rank deficiency noted.
#[test]
fn contrast_f_rank_deficient_l_is_noted() {
    let fit = lmm_fit(None, &all_targets(3));
    let l = [0.0, 1.0, -0.5];
    let one = fit.contrast_f(&l, 1).unwrap();
    let two = fit.contrast_f(&[l, l].concat(), 2).unwrap();
    assert_eq!(two.num_df, 1.0);
    assert_eq!(
        two.notes,
        [ContrastNote::RankDeficientL { rank: 1, rows: 2 }]
    );
    assert!(rel(two.f, one.f) < 1e-12, "{} {}", two.f, one.f);
    assert!(rel(two.den_df, one.den_df) < 1e-12);
}

/// The aliased column in the middle (`1, x1, 2·x1, x2`): the state covers
/// columns 0, 1 and 3, and every df and contrast is the reduced fit's,
/// read through that map, not through the first three columns.
#[test]
fn rank_deficient_middle_column_maps_through_retained() {
    let full = lmm_fit(Some(2), &all_targets(4));
    let reduced = lmm_fit(None, &all_targets(3));
    assert_eq!(full.aliased(), [false, false, true, false]);
    let st = full.satterthwaite.as_ref().unwrap();
    assert_eq!(st.retained.as_deref(), Some(&[0, 1, 3][..]));
    assert!(full.coef_df[2].is_nan());
    let kept = [full.coef_df[0], full.coef_df[1], full.coef_df[3]];
    assert_eq!(bits(&kept), bits(&reduced.coef_df));

    let t3 = full.contrast_t(&unit(4, 3)).unwrap();
    assert_eq!(t3.estimate, full.beta[3]);
    assert!(rel(t3.se, full.se[3]) < 1e-12);
    assert!(rel(t3.df, full.coef_df[3]) < 1e-12);
    assert_eq!(
        full.contrast_t(&[0.0, 1.0, 0.0, -1.0]).unwrap(),
        reduced.contrast_t(&[0.0, 1.0, -1.0]).unwrap()
    );

    let t = full.contrast_t(&[0.0, 0.0, 1.0, 1.0]).unwrap();
    for v in [t.estimate, t.se, t.t, t.df, t.p_value] {
        assert!(v.is_nan());
    }
    assert_eq!(t.notes, [ContrastNote::AliasedColumn { column: 2 }]);
}

/// A Gaussian LMM with no finite endpoint records a failed pass without
/// running one: the `p = 0` refusal through `fit_warm`, and a design whose
/// aliased column is a random slope (`fit_rank_deficient`'s unfittable
/// return). Both give the note with `nonfinite` and `NoDf(Failed)`; switched
/// off, the latter gives `Disabled` and no note.
#[test]
fn no_endpoint_lmm_records_failed_pass() {
    let failed_note = |fit: &Fit| {
        fit.diagnostics
            .notes
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    Note::SatterthwaiteCurvature {
                        dropped: 0,
                        negative: 0,
                        converged: false,
                        nonfinite: true,
                    }
                )
            })
            .count()
    };
    let failed = ContrastError::NoDf(NoDfReason::Failed);

    let (_, y, ids, n, _) = lmm_data(None);
    let empty = fit_cold(&[], &y, n, 0, &lmm_model(), &ids, &FitOptions::default());
    assert!(!empty.converged() && empty.deviance.is_nan());
    assert_eq!(failed_note(&empty), 1);
    assert_eq!(empty.contrast_t(&[]), Err(failed));

    let (x, y, ids, n, p) = lmm_data(Some(2));
    let mut slope_on_dup = lmm_model();
    slope_on_dup.re.as_mut().unwrap().slopes = vec![2];
    let unfit = fit_cold(&x, &y, n, p, &slope_on_dup, &ids, &all_targets(p));
    assert!(!unfit.converged() && unfit.deviance.is_nan());
    assert_eq!(failed_note(&unfit), 1);
    assert!(unfit.coef_df.iter().all(|d| d.is_nan()));
    assert_eq!(unfit.contrast_t(&unit(p, 1)), Err(failed));
    assert_eq!(unfit.contrast_f(&unit(p, 1), 1), Err(failed));

    let off = fit_cold(
        &x,
        &y,
        n,
        p,
        &slope_on_dup,
        &ids,
        &FitOptions {
            satterthwaite: false,
            ..all_targets(p)
        },
    );
    assert_eq!(failed_note(&off), 0);
    assert_eq!(
        off.contrast_t(&unit(p, 1)),
        Err(ContrastError::NoDf(NoDfReason::Disabled))
    );
}

/// The OLS F test is the classical one: every direction's df is `n − p`, so
/// the mean rule gives a DenDF of `n − p` exactly, and
/// `F = (Lβ̂)ᵀ(L·C·Lᵀ)⁻¹(Lβ̂)/r`.
#[test]
fn ols_contrast_f_is_classical_f() {
    let mut s = 31u64;
    let (n, p) = (40, 4);
    let (mut x, mut y) = (vec![], vec![]);
    for _ in 0..n {
        let (x1, x2, x3) = (lcg(&mut s), lcg(&mut s), lcg(&mut s));
        x.extend([1.0, x1, x2, x3]);
        y.push(0.2 + 0.5 * x1 - 0.4 * x2 + 0.1 * x3 + 0.7 * lcg(&mut s));
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let fit = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &all_targets(p));
    assert!(fit.converged());
    let nu = (n - p) as f64;
    // L = [e_1; e_2 + e_3].
    let l = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
    let got = fit.contrast_f(&l, 2).expect("OLS has a residual df");
    assert_eq!(got.num_df, 2.0);
    assert_eq!(got.den_df, nu);
    assert!(got.notes.is_empty());
    // (Lβ̂)ᵀ(L·C·Lᵀ)⁻¹(Lβ̂)/2 with the 2×2 inverse written out.
    let c = &fit.vcov;
    let lb = [fit.beta[1], fit.beta[2] + fit.beta[3]];
    let (a, b, d) = (
        c[1][1],
        c[1][2] + c[1][3],
        c[2][2] + c[3][3] + 2.0 * c[2][3],
    );
    let det = a * d - b * b;
    let want = (d * lb[0] * lb[0] - 2.0 * b * lb[0] * lb[1] + a * lb[1] * lb[1]) / det / 2.0;
    assert!(rel(got.f, want) < 1e-12, "F {} against {want}", got.f);
    assert_eq!(
        got.p_value,
        crate::inference::tails::f_upper_p(got.f, 2.0, nu)
    );
}
