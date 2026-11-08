//! Profile-likelihood intervals of a Gaussian LMM: each bound is where the
//! signed root of the ML deviance change reaches the normal quantile, a
//! variance at zero gets a one-sided interval, a correlation stops at ±1, an
//! unidentified correlation is undetermined, the sparse kernel profiles too,
//! and the refusals come before any work. On synthetic `|ζ|` paths, a held
//! minimum below the fit is a better optimum, and a side that never brackets
//! or whose `|ζ|` falls is undetermined.

use super::profile::{
    bound, coord_params, path, ProfileError, ProfileInterval, ProfileIntervals, ProfileParam,
    Toward,
};
use crate::fit::common_tests::lcg;
use crate::fit::{fit_cold_keep_workspace, Perm};
use crate::lmm::profile::ProfileKernel;
use crate::lmm::profile_tests::{
    design, levels, ml_opts, num, primary, sleepstudy_ml, slope_model, table, Case,
};
use crate::{
    BinomialLink, Family, Fit, FitError, FitOptions, GroupIds, Grouping, ModelSpec, ReStructure,
};

/// `Φ⁻¹(0.975)`.
const Z95: f64 = 1.959963984540054;

/// The case fitted by `crate::fit`, REML or ML.
fn fit_case(case: &Case, reml: bool) -> Fit {
    let opts = FitOptions {
        reml: Some(reml),
        ..FitOptions::default()
    };
    let fit = crate::fit(&case.data(), &case.model, &opts).expect("the case fits");
    assert!(fit.converged());
    fit
}

/// The case's profile intervals, from a fit by `crate::fit`.
fn profile(
    case: &Case,
    reml: bool,
    level: f64,
    which: Option<&[ProfileParam]>,
) -> ProfileIntervals {
    fit_case(case, reml)
        .profile_intervals(&case.data(), &case.model, level, which)
        .expect("the profile runs")
}

/// `|ζ|` at `value` of `param`: the held minimum from the estimate (θ̂ for a
/// coefficient), on a fresh ML fit of the case.
fn zeta_at(case: &Case, param: ProfileParam, value: f64) -> f64 {
    // No case that calls this is reordered by the size rule (each
    // multi-grouping case has a slope).
    case.with_ml(|_, ws, map| {
        let hidden = ws.kernel.groupings().hidden.clone();
        let mut k = ProfileKernel::new(ws, map);
        let base = k.kernel_deviance_at_hat();
        let (d, ok) = match param {
            ProfileParam::Beta(j) => {
                let mut start = k.theta_hat().to_vec();
                k.hold_beta(j, value, &mut start)
            }
            _ => {
                let i = coord_params(k.coords(), &hidden, Perm::IDENTITY)
                    .iter()
                    .position(|&q| q == param)
                    .expect("a variance parameter of the model");
                let mut start = k.estimate();
                k.hold_var(i, value, &mut start)
            }
        };
        assert!(ok, "{param:?} held at {value}");
        (d - base).max(0.0).sqrt()
    })
}

/// Checks `|ζ| = z` at every bound of `rows` that is off its limit (finite,
/// not 0 on an SD, not ±1 on a correlation); returns how many it checked.
fn check_bounds(case: &Case, rows: &[ProfileInterval], z: f64) -> usize {
    let mut checked = 0;
    for r in rows {
        for b in [r.lower, r.upper] {
            let at_limit = match r.param {
                ProfileParam::Sd { .. } => b == 0.0,
                ProfileParam::Cor { .. } => b.abs() == 1.0,
                _ => false,
            };
            if !b.is_finite() || at_limit {
                continue;
            }
            let zeta = zeta_at(case, r.param, b);
            assert!(
                (zeta - z).abs() <= 1e-5,
                "{:?} at {b}: |ζ| {zeta} against {z}",
                r.param
            );
            checked += 1;
        }
    }
    checked
}

/// Asserts both bounds finite, off their limits and around the estimate.
fn assert_two_sided(r: &ProfileInterval) {
    assert!(
        r.lower.is_finite() && r.upper.is_finite(),
        "{:?}: [{}, {}]",
        r.param,
        r.lower,
        r.upper
    );
    assert!(
        r.lower < r.estimate && r.estimate < r.upper,
        "{:?}: {} not inside [{}, {}]",
        r.param,
        r.estimate,
        r.lower,
        r.upper
    );
    match r.param {
        ProfileParam::Sd { .. } => assert!(r.lower > 0.0, "{:?}", r.param),
        ProfileParam::Cor { .. } => {
            assert!(r.lower > -1.0 && r.upper < 1.0, "{:?}", r.param)
        }
        _ => {}
    }
}

fn sd(group: usize, term: usize) -> ProfileParam {
    ProfileParam::Sd { group, term }
}

/// sleepstudy's six parameters, in glmm's row order.
fn sleepstudy_params() -> Vec<ProfileParam> {
    vec![
        sd(0, 0),
        ProfileParam::Cor {
            group: 0,
            row: 1,
            col: 0,
        },
        sd(0, 1),
        ProfileParam::Sigma,
        ProfileParam::Beta(0),
        ProfileParam::Beta(1),
    ]
}

#[test]
fn bounds_sit_where_zeta_reaches_the_quantile() {
    let case = sleepstudy_ml();
    let out = profile(&case, true, 0.95, None);
    assert_eq!(out.level, 0.95);
    let params: Vec<ProfileParam> = out.rows.iter().map(|r| r.param).collect();
    assert_eq!(params, sleepstudy_params());
    out.rows.iter().for_each(assert_two_sided);
    assert_eq!(check_bounds(&case, &out.rows, Z95), 12);
}

#[test]
fn a_lower_level_nests_inside() {
    let case = sleepstudy_ml();
    let wide = profile(&case, true, 0.95, None);
    let narrow = profile(&case, true, 0.8, None);
    assert_eq!(narrow.level, 0.8);
    let mut nested = 0;
    for (w, n) in wide.rows.iter().zip(&narrow.rows) {
        assert_eq!(w.param, n.param);
        assert!(
            w.lower < n.lower && n.upper < w.upper,
            "{:?}: [{}, {}] not inside [{}, {}]",
            w.param,
            n.lower,
            n.upper,
            w.lower,
            w.upper
        );
        nested += 1;
    }
    assert_eq!(nested, 6);
}

/// Dyestuff2, `Yield ~ 1 + (1 | Batch)`: the batch SD is estimated at 0.
fn dyestuff2() -> Case {
    let t = table(include_str!(
        "../../../validation/data/empirical/Dyestuff2.csv"
    ));
    let y = num(&t, "Yield");
    Case {
        x: vec![1.0; y.len()],
        y,
        p: 1,
        ids: primary(levels(&t, "Batch")),
        w: None,
        model: crate::test_support::intercept_only_spec(),
    }
}

#[test]
fn an_estimate_at_zero_gets_a_one_sided_interval() {
    let case = dyestuff2();
    let out = profile(&case, true, 0.95, None);
    let params: Vec<ProfileParam> = out.rows.iter().map(|r| r.param).collect();
    assert_eq!(
        params,
        [sd(0, 0), ProfileParam::Sigma, ProfileParam::Beta(0)]
    );
    let batch = out.rows[0];
    assert_eq!(batch.estimate.to_bits(), 0.0f64.to_bits());
    assert_eq!(batch.lower.to_bits(), 0.0f64.to_bits());
    assert!(batch.upper.is_finite() && batch.upper > 0.0, "{batch:?}");
    out.rows[1..].iter().for_each(assert_two_sided);
    // The upper SD bound and the four of σ and β.
    assert_eq!(check_bounds(&case, &out.rows, Z95), 5);
}

#[test]
fn a_correlation_at_its_limit_stays_there() {
    let t = table(include_str!(
        "../../../validation/data/simulated/sim_corr_boundary.csv"
    ));
    let case = Case {
        x: design(&[&num(&t, "x")]),
        y: num(&t, "y"),
        p: 2,
        ids: primary(levels(&t, "g")),
        w: None,
        model: slope_model(vec![1]),
    };
    let cor = ProfileParam::Cor {
        group: 0,
        row: 1,
        col: 0,
    };
    let out = profile(&case, true, 0.95, Some(&[cor]));
    assert_eq!(out.rows.len(), 1);
    let r = out.rows[0];
    assert!(r.estimate != 0.0 && r.estimate.abs() <= 1.0, "{r:?}");
    let toward = if r.estimate > 0.0 { r.upper } else { r.lower };
    assert_eq!(toward, r.estimate.signum(), "{r:?}");
}

/// `y ~ x + (1 + x | g)` on seeded data with no random intercept and no
/// random slope.
fn no_group_variance(seed: u64) -> Case {
    let (groups, per) = (12u32, 8usize);
    let mut st = seed;
    let (mut x, mut y, mut g) = (vec![], vec![], vec![]);
    for grp in 0..groups {
        for _ in 0..per {
            let xi = lcg(&mut st);
            x.extend([1.0, xi]);
            y.push(1.0 + 0.5 * xi + 0.3 * lcg(&mut st));
            g.push(grp);
        }
    }
    Case {
        x,
        y,
        p: 2,
        ids: primary(g),
        w: None,
        model: slope_model(vec![1]),
    }
}

#[test]
fn a_correlation_next_to_a_zero_sd_is_undetermined() {
    let case = no_group_variance(6);
    let out = profile(&case, false, 0.95, None);
    let params: Vec<ProfileParam> = out.rows.iter().map(|r| r.param).collect();
    assert_eq!(params, sleepstudy_params());
    // The fit pins a slope variance with its intercept's (the whole block
    // at 0) or keeps the slope's row of the factor off 0 (a correlation of
    // ±1): seed 6 is the first case.
    assert_eq!(out.rows[2].estimate, 0.0, "the slope SD is estimated at 0");
    assert_eq!(out.rows[0].estimate, 0.0, "so is the intercept's");
    let cor = out.rows[1];
    assert!(cor.lower.is_nan() && cor.upper.is_nan(), "{cor:?}");
    let mut finite = 0;
    for r in out.rows.iter().filter(|r| r.param != cor.param) {
        assert!(r.lower.is_finite() && r.upper.is_finite(), "{r:?}");
        finite += 1;
    }
    assert_eq!(finite, 5);
}

#[test]
fn the_sparse_route_profiles_too() {
    let t = table(include_str!(
        "../../../validation/data/simulated/sim_slope_extra.csv"
    ));
    let case = Case {
        x: design(&[&num(&t, "x")]),
        y: num(&t, "y"),
        p: 2,
        ids: GroupIds {
            primary: levels(&t, "g1"),
            extra: vec![levels(&t, "g2")],
        },
        w: None,
        model: ModelSpec::mixed(
            Family::Gaussian,
            ReStructure::intercept()
                .slopes([1])
                .extra(Grouping::crossed().slopes([1])),
        ),
    };
    let (_, fws) = fit_cold_keep_workspace(&case.data(), &case.model, &ml_opts());
    assert!(
        fws.is_lmm_sparse(),
        "a slope on an extra grouping routes sparse"
    );
    let which = [ProfileParam::Sigma, ProfileParam::Beta(1), sd(1, 1)];
    let out = profile(&case, true, 0.95, Some(&which));
    // glmm's row order: the variance blocks, σ, then β.
    let params: Vec<ProfileParam> = out.rows.iter().map(|r| r.param).collect();
    assert_eq!(
        params,
        [sd(1, 1), ProfileParam::Sigma, ProfileParam::Beta(1)]
    );
    out.rows[1..].iter().for_each(assert_two_sided);
    assert!(out.rows[0].upper.is_finite(), "{:?}", out.rows[0]);
    let off_limit = out
        .rows
        .iter()
        .flat_map(|r| [r.lower, r.upper].map(|b| (r.param, b)))
        .filter(|&(q, b)| !(matches!(q, ProfileParam::Sd { .. }) && b == 0.0))
        .count();
    assert_eq!(check_bounds(&case, &out.rows, Z95), off_limit);
}

#[test]
fn an_ml_fit_is_its_own_base() {
    let case = sleepstudy_ml();
    let ml_fit = fit_case(&case, false);
    let ml = ml_fit
        .profile_intervals(&case.data(), &case.model, 0.95, None)
        .expect("the profile runs");
    assert_eq!(ml.deviance.to_bits(), (-2.0 * ml_fit.loglik).to_bits());
    let reml = profile(&case, true, 0.95, None);
    assert_eq!(reml.deviance.to_bits(), ml.deviance.to_bits());
    let mut same = 0;
    for (a, b) in reml.rows.iter().zip(&ml.rows) {
        assert_eq!(a.param, b.param);
        for (u, v) in [
            (a.estimate, b.estimate),
            (a.lower, b.lower),
            (a.upper, b.upper),
        ] {
            assert_eq!(u.to_bits(), v.to_bits(), "{:?}", a.param);
        }
        same += 1;
    }
    assert_eq!(same, 6);
    // The estimates are the ML fit's.
    assert_eq!(ml.rows[4].estimate, ml_fit.beta[0]);
    assert_eq!(ml.rows[5].estimate, ml_fit.beta[1]);
}

#[test]
fn which_selects_rows() {
    let case = sleepstudy_ml();
    let full = profile(&case, true, 0.95, None);
    let which = [ProfileParam::Sigma, ProfileParam::Beta(1)];
    let some = profile(&case, true, 0.95, Some(&which));
    assert_eq!(some.rows.len(), 2);
    let mut matched = 0;
    for r in &some.rows {
        let f = full
            .rows
            .iter()
            .find(|f| f.param == r.param)
            .expect("a row");
        for (u, v) in [
            (r.estimate, f.estimate),
            (r.lower, f.lower),
            (r.upper, f.upper),
        ] {
            assert_eq!(u.to_bits(), v.to_bits(), "{:?}", r.param);
        }
        matched += 1;
    }
    assert_eq!(matched, 2);
    assert_eq!(some.deviance.to_bits(), full.deviance.to_bits());
}

#[test]
fn refusals() {
    let case = sleepstudy_ml();
    let fit = fit_case(&case, true);
    let data = case.data();
    let mut refused = 0;

    let err = fit
        .profile_intervals(&data, &case.model, 1.0, None)
        .unwrap_err();
    assert_eq!(
        err,
        ProfileError::Fit(FitError::OptionValue(
            "level must be a number strictly between 0 and 1, got 1".into()
        ))
    );
    refused += 1;

    let bad = [sd(3, 0)];
    let err = fit
        .profile_intervals(&data, &case.model, 0.95, Some(&bad))
        .unwrap_err();
    assert!(
        matches!(err, ProfileError::Fit(FitError::OptionValue(_))),
        "{err}"
    );
    refused += 1;

    // OLS.
    let ols_spec = ModelSpec::fixed(Family::Gaussian);
    let ols_data = crate::Data::new(&case.x, &case.y, case.y.len(), case.p);
    let ols = crate::fit(&ols_data, &ols_spec, &FitOptions::default()).expect("OLS fits");
    let err = ols
        .profile_intervals(&ols_data, &ols_spec, 0.95, None)
        .unwrap_err();
    assert!(
        matches!(err, ProfileError::Fit(FitError::Unsupported(_))),
        "{err}"
    );
    refused += 1;

    // A binomial GLMM.
    let mut st = 11u64;
    let yb: Vec<f64> = case
        .y
        .iter()
        .map(|_| f64::from(u8::from(lcg(&mut st) > 0.0)))
        .collect();
    let bin = Case {
        y: yb,
        model: ModelSpec::mixed(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            ReStructure::intercept(),
        ),
        x: case.x.clone(),
        p: case.p,
        ids: case.ids.clone(),
        w: None,
    };
    let bin_fit = crate::fit(&bin.data(), &bin.model, &FitOptions::default()).expect("fits");
    let err = bin_fit
        .profile_intervals(&bin.data(), &bin.model, 0.95, None)
        .unwrap_err();
    assert!(
        matches!(err, ProfileError::Fit(FitError::Unsupported(_))),
        "{err}"
    );
    refused += 1;

    // An aliased column: Days twice.
    let aliased = Case {
        x: case.x.chunks(2).flat_map(|r| [r[0], r[1], r[1]]).collect(),
        p: 3,
        y: case.y.clone(),
        ids: case.ids.clone(),
        w: None,
        model: slope_model(vec![1]),
    };
    let al_fit = fit_case(&aliased, true);
    assert!(al_fit.aliased()[2]);
    let err = al_fit
        .profile_intervals(&aliased.data(), &aliased.model, 0.95, None)
        .unwrap_err();
    assert!(
        matches!(err, ProfileError::Fit(FitError::Unsupported(_))),
        "{err}"
    );
    refused += 1;

    assert_eq!(refused, 5);
}

#[test]
fn the_parameters_follow_the_spec() {
    // `(1 + x1 | g) + (0 + x2 | g)` and a crossed intercept: the term
    // between the blocks has no correlation, and the slope-only term's SD
    // is column 0 of its own reported block when it stands alone.
    let spec = ModelSpec::mixed(
        Family::Gaussian,
        ReStructure::intercept()
            .terms([
                crate::ReTerm::intercept().slopes([1]),
                crate::ReTerm::no_intercept().slopes([2]),
            ])
            .extra(Grouping::crossed().terms([crate::ReTerm::no_intercept().slopes([1])])),
    );
    assert_eq!(
        super::profile::profile_params(&spec, 3),
        [
            sd(0, 0),
            ProfileParam::Cor {
                group: 0,
                row: 1,
                col: 0
            },
            sd(0, 1),
            sd(0, 2),
            sd(1, 0),
            ProfileParam::Sigma,
            ProfileParam::Beta(0),
            ProfileParam::Beta(1),
            ProfileParam::Beta(2),
        ]
    );
}

#[test]
fn split_and_slope_only_terms_report_their_own_columns() {
    // sleepstudy with `(Days || Subject)`, whose zero covariance has no row,
    // and with `(0 + Days | Subject)`, whose block reports the slope alone
    // while the kernel keeps an idle intercept column.
    let split = Case {
        model: ModelSpec::mixed(
            Family::Gaussian,
            ReStructure::intercept().slopes([1]).uncorrelated(),
        ),
        ..sleepstudy_ml()
    };
    let slope_only = Case {
        model: ModelSpec::mixed(
            Family::Gaussian,
            ReStructure::intercept().terms([crate::ReTerm::no_intercept().slopes([1])]),
        ),
        ..sleepstudy_ml()
    };
    let tail = [
        ProfileParam::Sigma,
        ProfileParam::Beta(0),
        ProfileParam::Beta(1),
    ];
    let mut checked = 0;
    for (case, sds) in [
        (split, vec![sd(0, 0), sd(0, 1)]),
        (slope_only, vec![sd(0, 0)]),
    ] {
        let out = profile(&case, true, 0.95, None);
        let params: Vec<ProfileParam> = out.rows.iter().map(|r| r.param).collect();
        let want: Vec<ProfileParam> = sds.iter().chain(&tail).copied().collect();
        assert_eq!(params, want);
        let (stddev, _) = fit_case(&case, false).stddev_corr(0);
        for (r, s) in out.rows.iter().zip(&stddev) {
            assert!(((r.estimate - s) / s).abs() <= 1e-12, "{r:?} {s}");
            assert_two_sided(r);
            checked += 1;
        }
    }
    assert_eq!(checked, 3);
}

/// A held minimum more than the tolerance below the fit's deviance is a
/// better optimum, reported with its drop; a smaller drop reads as `ζ = 0`.
#[test]
fn a_held_minimum_below_the_fit_is_a_better_optimum() {
    let (base, param) = (100.0, ProfileParam::Sigma);
    let side = |drop: f64| {
        let zeta = path(Z95, base, param, vec![0.0], |_, _| (base - drop, true));
        bound(Z95, 0.0, Toward::Expand(&|t| t), zeta)
    };
    match side(2e-6) {
        Err(ProfileError::BetterOptimum { param: q, drop }) => {
            assert_eq!(q, param);
            assert!((drop - 2e-6).abs() < 1e-12, "drop {drop}");
        }
        other => panic!("expected BetterOptimum, got {other:?}"),
    }
    // ζ = 0 at every point never brackets the root.
    assert!(side(5e-7).expect("within the tolerance").is_nan());
}

/// The message names the parameter in words, not as the Rust enum.
#[test]
fn a_better_optimum_message_names_the_parameter() {
    let msg = |param| ProfileError::BetterOptimum { param, drop: 2e-6 }.to_string();
    let sd = msg(ProfileParam::Sd { group: 0, term: 1 });
    assert!(
        sd.contains("profiling the SD of column 1 of grouping 0 found"),
        "{sd}"
    );
    assert!(!sd.contains("Sd {"), "{sd}");
    let sigma = msg(ProfileParam::Sigma);
    assert!(
        sigma.contains("profiling the residual SD (sigma) found"),
        "{sigma}"
    );
}

/// A side whose `|ζ|` rises but never reaches the quantile is undetermined.
#[test]
fn a_side_that_never_brackets_is_nan() {
    let mut calls = 0;
    let zeta = |v: f64| {
        calls += 1;
        Ok(Some(1.0 - 1.0 / (1.0 + v)))
    };
    let b = bound(Z95, 0.0, Toward::Expand(&|t| t), zeta).expect("no error");
    assert!(b.is_nan() && calls > 1, "{b} after {calls} points");
}

/// A side whose `|ζ|` falls by more than the tolerance between expansion
/// steps is undetermined, though a later step would bracket the root; a
/// smaller fall is kept and the root found.
#[test]
fn a_falling_zeta_makes_the_side_nan() {
    // Expansion points v = 1, 2, 4, ...: |ζ| rises to 1, falls by `drop` up
    // to 2, then climbs past z before 4.
    let side = |drop: f64| {
        let zeta = |v: f64| {
            Ok(Some(if v <= 1.0 {
                v
            } else if v <= 2.0 {
                1.0 - drop * (v - 1.0)
            } else {
                1.0 - drop + 2.0 * (v - 2.0)
            }))
        };
        bound(Z95, 0.0, Toward::Expand(&|t| t), zeta).expect("no error")
    };
    assert!(side(2e-5).is_nan());
    let kept = side(5e-7);
    assert!(
        (kept - (2.0 + (Z95 - 1.0 + 5e-7) / 2.0)).abs() < 1e-6,
        "the root past the small fall, got {kept}"
    );
}

/// §6, D17 (G9): profile intervals refit on the weights as given, so a fit
/// that estimated its weight power, or used a fixed power other than 1, is
/// refused; a power of 1 is the fit the profiles refit.
#[test]
fn a_weight_power_other_than_one_is_refused() {
    use crate::WeightPower::{Estimate, Fixed};
    let mut case = sleepstudy_ml();
    case.w = Some(
        (0..case.y.len())
            .map(|i| 1.0 + (i % 7) as f64 / 2.0)
            .collect(),
    );
    for power in [Estimate, Fixed(0.5)] {
        let opts = FitOptions {
            weight_power: Some(power),
            ..FitOptions::default()
        };
        let fit = crate::fit(&case.data(), &case.model, &opts).expect("fits");
        let err = fit
            .profile_intervals(&case.data(), &case.model, 0.95, None)
            .unwrap_err();
        assert!(
            matches!(err, ProfileError::Fit(FitError::Unsupported(_))),
            "{power:?}: {err}"
        );
        assert!(err.to_string().contains("weight"), "{err}");
    }
    let claimed = FitOptions {
        weight_power: Some(Fixed(1.0)),
        ..FitOptions::default()
    };
    let fit = crate::fit(&case.data(), &case.model, &claimed).expect("fits");
    let which = [ProfileParam::Sigma];
    assert!(fit
        .profile_intervals(&case.data(), &case.model, 0.95, Some(&which))
        .is_ok());
}

/// Two crossed intercept-only groupings, a (6 levels) declared before b (40):
/// the size rule makes b the kernel's primary, so the kernel's coordinates
/// come in the order b, a. Each row must still carry its own grouping's
/// interval, the same numbers as when b is declared first.
#[test]
fn size_rule_reorder_keeps_each_interval_on_its_grouping() {
    let n = 600;
    let mut st = 7u64;
    let ua: Vec<f64> = (0..6).map(|_| 2.0 * lcg(&mut st)).collect();
    let ub: Vec<f64> = (0..40).map(|_| 0.3 * lcg(&mut st)).collect();
    let (mut x, mut y, mut a, mut b) = (vec![], vec![], vec![], vec![]);
    for _ in 0..n {
        let ia = ((lcg(&mut st) + 1.0) * 0.5 * 6.0) as usize % 6;
        let ib = ((lcg(&mut st) + 1.0) * 0.5 * 40.0) as usize % 40;
        let x1 = lcg(&mut st);
        x.extend([1.0, x1]);
        y.push(0.3 + 0.5 * x1 + ua[ia] + ub[ib] + lcg(&mut st));
        a.push(ia as u32);
        b.push(ib as u32);
    }
    let case = |ids: GroupIds| Case {
        x: x.clone(),
        y: y.clone(),
        p: 2,
        ids,
        w: None,
        model: ModelSpec::mixed(
            Family::Gaussian,
            ReStructure::intercept().extra(Grouping::crossed()),
        ),
    };
    let ab_case = case(GroupIds::new(a.clone()).extra(b.clone()));
    let ba_case = case(GroupIds::new(b).extra(a));
    let reordered = |c: &Case| {
        !fit_cold_keep_workspace(&c.data(), &c.model, &ml_opts())
            .1
            .perm()
            .is_identity()
    };
    assert!(reordered(&ab_case) && !reordered(&ba_case));

    let ab = profile(&ab_case, false, 0.95, None);
    let ba = profile(&ba_case, false, 0.95, None);
    let vals = |r: &ProfileInterval| (r.estimate, r.lower, r.upper);
    assert_eq!(ab.rows[0].param, sd(0, 0));
    assert_eq!(ab.rows[1].param, sd(1, 0));
    // The kernel sees the same input both ways, and one row's search does not
    // depend on the rows before it (see `which_selects_rows`).
    assert_eq!(vals(&ab.rows[0]), vals(&ba.rows[1]), "a's row");
    assert_eq!(vals(&ab.rows[1]), vals(&ba.rows[0]), "b's row");
    assert_eq!(ab.rows[2..], ba.rows[2..]);
    let sd_a = fit_case(&ab_case, false).varcorr(0)[0].sqrt();
    assert!(
        (ab.rows[0].estimate - sd_a).abs() <= 1e-8 * sd_a.abs(),
        "{} vs {sd_a}",
        ab.rows[0].estimate
    );
    let only_a = profile(&ab_case, false, 0.95, Some(&[sd(0, 0)])).rows;
    assert_eq!(only_a.len(), 1);
    assert_eq!(vals(&only_a[0]), vals(&ab.rows[0]), "which picks a's row");
}
