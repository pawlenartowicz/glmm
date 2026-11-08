//! The fitted weight power (§3.1, §3.1c, §3.3 of the δ fit spec): the search
//! against a dense grid of the crate's own profile, its identities, notes and
//! the `None` resolution.

use super::common::resolve_weight_power;
use super::common_tests::lcg;
use super::delta::{delta_bracket, implausibility_note, probe_options, profile};
use super::*;
use crate::{Grouping, PoissonLink, ReStructure};

/// δ̂'s deviance over the best of the 2001-point grid on a random-intercept
/// dataset (P4). Measured on [`mixed_sets`]: δ̂ beat the grid on every one (the
/// closest by 1.35e-4), so the bound is the WLS one.
const MIXED_TOL: f64 = 1e-7;

/// One seeded dataset: `[1, x]` rows, weights spanning `[scale, scale·spread]`
/// (both ends attained), noise variance `(w/scale)^(−delta0)`, and with
/// `groups > 0` a random intercept of SD 0.8 on `i % groups`.
struct Set {
    x: Vec<f64>,
    y: Vec<f64>,
    w: Vec<f64>,
    ids: GroupIds,
    n: usize,
    groups: usize,
}

fn normal(st: &mut u64) -> f64 {
    let u1 = 0.5 * (lcg(st) + 1.0) + 1e-12;
    let u2 = 0.5 * (lcg(st) + 1.0);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn dataset(seed: u64, n: usize, groups: usize, scale: f64, spread: f64, delta0: f64) -> Set {
    let mut st = seed;
    let u: Vec<f64> = (0..groups.max(1)).map(|_| 0.8 * normal(&mut st)).collect();
    let mut x = Vec::with_capacity(2 * n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = lcg(&mut st);
        let r = match i {
            0 => 1.0,
            1 => spread,
            _ => spread.powf(0.5 * (lcg(&mut st) + 1.0)),
        };
        let g = if groups > 0 { u[i % groups] } else { 0.0 };
        x.extend_from_slice(&[1.0, x1]);
        y.push(1.0 + 0.5 * x1 + g + normal(&mut st) * r.powf(-0.5 * delta0));
        w.push(scale * r);
    }
    let ids = GroupIds::new((0..n).map(|i| (i % groups.max(1)) as u32).collect());
    Set {
        x,
        y,
        w,
        ids,
        n,
        groups,
    }
}

impl Set {
    fn spec(&self) -> ModelSpec {
        if self.groups > 0 {
            ModelSpec::mixed(Family::Gaussian, ReStructure::intercept())
        } else {
            ModelSpec::fixed(Family::Gaussian)
        }
    }

    fn data<'a>(&'a self, w: Option<&'a [f64]>) -> Data<'a> {
        let mut d = Data::new(&self.x, &self.y, self.n, 2);
        if self.groups > 0 {
            d = d.ids(&self.ids);
        }
        match w {
            Some(w) => d.weights(w),
            None => d,
        }
    }

    fn fit(&self, opts: &FitOptions) -> Fit {
        crate::fit(&self.data(Some(&self.w)), &self.spec(), opts).unwrap()
    }

    /// The best deviance (`−2·loglik`) of the profile on 2001 points over the
    /// bracket, by REML or ML.
    fn grid_best(&self, reml: bool) -> f64 {
        let (lo, hi) = delta_bracket(&self.w);
        let probe = probe_options(&FitOptions::default(), reml);
        let data = self.data(Some(&self.w));
        (0..2001)
            .map(|k| {
                let d = lo + (hi - lo) * k as f64 / 2000.0;
                -2.0 * profile(&data, &self.spec(), &probe, None, d).0
            })
            .filter(|v| v.is_finite())
            .fold(f64::INFINITY, f64::min)
    }
}

/// WLS datasets: spreads 20 to 1e16, and weights near zero (1e-12 to 1e-8).
fn wls_sets() -> Vec<Set> {
    vec![
        dataset(1, 50, 0, 1.0, 20.0, 1.0),
        dataset(2, 100, 0, 1.0, 1e6, 0.5),
        dataset(3, 60, 0, 1.0, 1e16, 0.0),
        dataset(4, 80, 0, 1e-12, 1e4, 2.0),
    ]
}

/// Random-intercept datasets, the same spreads.
fn mixed_sets() -> Vec<Set> {
    vec![
        dataset(11, 120, 12, 1.0, 20.0, 1.0),
        dataset(12, 200, 20, 1.0, 1e6, 0.5),
        dataset(13, 150, 10, 1.0, 1e16, 0.5),
        dataset(14, 100, 10, 1e-12, 1e3, 0.0),
    ]
}

fn has_search_note(fit: &Fit) -> bool {
    fit.diagnostics.notes.iter().any(|n| {
        matches!(
            n,
            Note::WeightPowerAtBound { .. } | Note::WeightPowerSearchFailures { .. }
        )
    })
}

#[test]
fn delta_bracket_follows_the_weight_spread() {
    for (s, want) in [
        (20.0, 21.52),
        (100.0, 14.0),
        (1e6, 4.67),
        (1.5, 30.0),
        (1e40, 2.0),
    ] {
        let (lo, hi) = delta_bracket(&[3.0, 3.0 * s, 3.0 * s.sqrt()]);
        assert!((hi - want).abs() < 0.01 && lo == -hi, "s = {s}: {lo}, {hi}");
    }
}

/// G1 by REML, through the default call (which resolves to the δ fit on these
/// shapes): δ̂'s deviance is within 1e-7 (WLS) or `MIXED_TOL` (random
/// intercept) of the grid's best; no fit fails at spreads up to 1e16; the fit
/// reports REML, its log-likelihood is the profile at δ̂, and δ̂ counts in
/// `df` (D6). Past what `f64` resolves, the search notes its probes.
#[test]
fn reml_search_reaches_the_grid_best() {
    for (sets, tol) in [(wls_sets(), 1e-7), (mixed_sets(), MIXED_TOL)] {
        for set in sets {
            let fit = set.fit(&FitOptions::default());
            let delta = fit.weight_power();
            assert!(fit.converged() && fit.loglik().is_finite(), "δ̂ = {delta}");
            assert!(fit.weight_power_estimated && fit.reml());
            let gap = -2.0 * fit.loglik() - set.grid_best(true);
            assert!(gap <= tol, "groups {}: gap {gap:e}", set.groups);
            let probe = probe_options(&FitOptions::default(), true);
            let at = profile(&set.data(Some(&set.w)), &set.spec(), &probe, None, delta).0;
            assert!((fit.loglik() - at).abs() <= 1e-12 * at.abs(), "{at}");
            let claimed = set.fit(&FitOptions {
                weight_power: Some(WeightPower::Fixed(1.0)),
                ..FitOptions::default()
            });
            assert_eq!(fit.df(), claimed.df() + 1);
            assert!(!has_search_note(&fit));
        }
    }
    // Spread 1e16 with the profile peaking near δ = 1.5: from δ ≈ 0.6 the
    // inner fits end on the θ limit or on the exact-fit floor, below the
    // profile by 0.2 to 1300, so the fit at 1.5 and the fit at δ̂ (0.83) say so.
    let rough = dataset(13, 150, 10, 1.0, 1e16, 1.5);
    let at = rough.fit(&FitOptions {
        weight_power: Some(WeightPower::Fixed(1.5)),
        ..FitOptions::default()
    });
    let notes = &at.diagnostics.notes;
    assert!(
        notes.iter().any(|n| matches!(n, Note::PrecisionLimited)),
        "{notes:?}"
    );
    let fit = rough.fit(&FitOptions::default());
    assert!(fit.converged() && fit.loglik().is_finite());
    let notes = &fit.diagnostics.notes;
    assert!(
        notes.iter().any(|n| matches!(n, Note::PrecisionLimited)),
        "{notes:?}"
    );
}

/// Spread 1e16 at δ = −1.5: the internal θ̂ (about 7e-5) is below `PIN_THETA`,
/// but zeroing it costs about 281 log-likelihood units, so the incumbent check
/// keeps it. A pinned fit would equal the WLS fit, the same model with no
/// random intercept.
#[test]
fn a_pin_that_costs_deviance_is_undone() {
    let set = dataset(13, 150, 10, 1.0, 1e16, 1.5);
    let opts = FitOptions {
        weight_power: Some(WeightPower::Fixed(-1.5)),
        reml: Some(true),
        ..FitOptions::default()
    };
    let mixed = set.fit(&opts);
    let wls = crate::fit(
        &Data::new(&set.x, &set.y, set.n, 2).weights(&set.w),
        &ModelSpec::fixed(Family::Gaussian),
        &opts,
    )
    .unwrap();
    assert!(mixed.converged() && mixed.varcorr(0)[0] > 0.0);
    assert!(
        mixed.loglik() - wls.loglik() > 200.0,
        "{} vs {}",
        mixed.loglik(),
        wls.loglik()
    );
}

/// The LMM's rounding estimate reads the response uncentred: a mean 1e6 times
/// the residual SD at n = 1000 takes `PrecisionLimited`, a mean 1e2 times it
/// does not.
#[test]
fn a_response_far_from_zero_is_precision_limited() {
    for (shift, limited) in [(1e6, true), (1e2, false)] {
        let mut set = dataset(21, 1000, 20, 1.0, 1.0, 0.0);
        set.y.iter_mut().for_each(|y| *y += shift);
        let fit = crate::fit(&set.data(None), &set.spec(), &FitOptions::default()).unwrap();
        let notes = &fit.diagnostics.notes;
        assert_eq!(
            notes.iter().any(|n| matches!(n, Note::PrecisionLimited)),
            limited,
            "shift {shift}: {notes:?}"
        );
    }
}

/// Spread 1e16 at δ = −2 by REML: from a warm start θ = 10 or 100 the search
/// stops at θ̂ ≈ 1e9 (caller units). There the intercept's pivot of X′V⁻¹X is
/// the difference of two numbers near 2.9e16 and comes out as a few units in
/// their last place, so the deviance is noise (loglik −2450.78 and −2474.29
/// against −2311.64 at the optimum). r² is fine there, so only the pivot term
/// of the estimate sees it. The δ = −1.5 cold fit (loglik −1685.894956) is a
/// near-optimum fit with a small θ and gets no note. Measured 2026-11-08: pivot
/// terms 20.7 (θ = 10) and 10.4 (θ = 100); before the pivot term neither fit
/// carried a note.
#[test]
fn a_fit_stopped_in_the_pivot_noise_is_precision_limited() {
    let set = dataset(13, 150, 10, 1.0, 1e16, 1.5);
    for (delta, start, limited) in [
        (-2.0, Some(10.0), true),
        (-2.0, Some(100.0), true),
        (-1.5, None, false),
    ] {
        let opts = FitOptions {
            weight_power: Some(WeightPower::Fixed(delta)),
            reml: Some(true),
            ..FitOptions::default()
        };
        let fit = match start {
            Some(t) => crate::fit_with_start(
                &set.data(Some(&set.w)),
                &set.spec(),
                &StartValues::new(vec![], vec![t]),
                &opts,
            )
            .unwrap(),
            None => set.fit(&opts),
        };
        let notes = &fit.diagnostics.notes;
        assert_eq!(
            notes.iter().any(|n| matches!(n, Note::PrecisionLimited)),
            limited,
            "δ {delta} start {start:?}: {notes:?}"
        );
    }
}

/// The δ dataset (spread 1e16) with a second, crossed random intercept on
/// `i % 7`, at δ = −2 by REML. The crossed shape keeps the Gram kernel, so
/// this note is its only guard. Intercept-only crossed extras route to the
/// dense kernel (`Solver::NoZ` in `classify_design`). Measured 2026-11-08:
/// pivot term 3.7, r-term 1.9e-9; the loglik depends on the start (−2204.93
/// cold, −2349.31 from θ = 10, −2386.48 from θ = 100).
#[test]
fn a_crossed_fit_under_wide_weights_is_precision_limited() {
    let set = dataset(13, 150, 10, 1.0, 1e16, 1.5);
    let ids = set
        .ids
        .clone()
        .extra((0..150).map(|i| (i % 7) as u32).collect());
    let spec = ModelSpec::mixed(
        Family::Gaussian,
        ReStructure::intercept().extra(Grouping::crossed()),
    );
    let opts = FitOptions {
        weight_power: Some(WeightPower::Fixed(-2.0)),
        reml: Some(true),
        ..FitOptions::default()
    };
    let data = Data::new(&set.x, &set.y, set.n, 2)
        .ids(&ids)
        .weights(&set.w);
    let fit = crate::fit(&data, &spec, &opts).unwrap();
    let notes = &fit.diagnostics.notes;
    assert!(
        notes.iter().any(|n| matches!(n, Note::PrecisionLimited)),
        "{notes:?}"
    );
}

/// The crossed model of the test above routes to the dense kernel
/// (`classify_design`), so this one forces the same model through the sparse
/// kernel (`fit_mle_sparse_pub`), whose `cxy` holds the X̃ Gram the pivot term
/// reads, and through the dense one (`fit_mle_noz_pub`), on the working
/// weights `w^−2` by REML. Both fits stop in the pivot noise and both carry
/// `PrecisionLimited`.
#[test]
fn a_sparse_routed_fit_computes_the_pivot_noise() {
    use super::common::spec_sized_from_ids;
    use super::{fit_mle_noz_pub, fit_mle_sparse_pub};
    let set = dataset(13, 150, 10, 1.0, 1e16, 1.5);
    let ids = set
        .ids
        .clone()
        .extra((0..150).map(|i| (i % 7) as u32).collect());
    let spec = ModelSpec::mixed(
        Family::Gaussian,
        ReStructure::intercept().extra(Grouping::crossed()),
    );
    let (sized, ids, _) = spec_sized_from_ids(&spec, &ids);
    let w: Vec<f64> = set.w.iter().map(|w| w.powf(-2.0)).collect();
    let opts = FitOptions {
        reml: Some(true),
        ..FitOptions::default()
    };
    for kernel in ["dense", "sparse"] {
        let f = if kernel == "sparse" {
            fit_mle_sparse_pub
        } else {
            fit_mle_noz_pub
        };
        let fit = f(
            &set.x,
            &set.y,
            Some(&w),
            None,
            set.n,
            2,
            &sized,
            &ids.primary,
            &ids.extra,
            None,
            &opts,
        );
        let notes = &fit.diagnostics.notes;
        assert!(
            notes.iter().any(|n| matches!(n, Note::PrecisionLimited)),
            "{kernel}: loglik {}, {notes:?}",
            fit.loglik()
        );
    }
}

/// G1: at δ = 1 and δ = 0 the profile is the REML log-likelihood of the
/// claimed fit and of the fit without weights.
#[test]
fn profile_at_one_and_zero_is_the_claimed_and_the_unweighted_fit() {
    let probe = probe_options(&FitOptions::default(), true);
    let reml = |power| FitOptions {
        weight_power: power,
        reml: Some(true),
        ..FitOptions::default()
    };
    for set in wls_sets().into_iter().chain(mixed_sets()) {
        let data = set.data(Some(&set.w));
        let claimed = set.fit(&reml(Some(WeightPower::Fixed(1.0)))).loglik();
        let plain = crate::fit(&set.data(None), &set.spec(), &reml(None))
            .unwrap()
            .loglik();
        for (d, want) in [(1.0, claimed), (0.0, plain)] {
            let got = profile(&data, &set.spec(), &probe, None, d).0;
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "δ = {d}: {got} vs {want}"
            );
        }
    }
}

/// §10.3: `w → 10·w` moves σ² only.
#[test]
fn scaling_the_weights_leaves_delta_and_beta() {
    for set in [
        dataset(1, 50, 0, 1.0, 20.0, 1.0),
        dataset(11, 120, 12, 1.0, 20.0, 1.0),
    ] {
        let a = set.fit(&FitOptions::default());
        let w10: Vec<f64> = set.w.iter().map(|w| 10.0 * w).collect();
        let b = crate::fit(&set.data(Some(&w10)), &set.spec(), &FitOptions::default()).unwrap();
        assert!((a.weight_power() - b.weight_power()).abs() <= 1e-8);
        for (x, y) in a.beta().iter().zip(b.beta()) {
            assert!((x - y).abs() <= 1e-8 * y.abs().max(1.0), "{x} vs {y}");
        }
    }
}

/// §3.1c: a named `Estimate` with `reml = Some(false)` maximizes the ML
/// log-likelihood (the same grid check) and reports ML.
#[test]
fn ml_search_reaches_the_ml_grid_best() {
    let opts = FitOptions {
        weight_power: Some(WeightPower::Estimate),
        reml: Some(false),
        ..FitOptions::default()
    };
    for (set, tol) in [
        (dataset(2, 100, 0, 1.0, 1e6, 0.5), 1e-7),
        (dataset(12, 200, 20, 1.0, 1e6, 0.5), MIXED_TOL),
    ] {
        let fit = set.fit(&opts);
        assert!(!fit.reml() && fit.weight_power_estimated);
        let gap = -2.0 * fit.loglik() - set.grid_best(false);
        assert!(gap <= tol, "groups {}: gap {gap:e}", set.groups);
    }
}

/// With at most `p` rows off the common weight the profile goes flat; on
/// this dataset it rises to the bracket's end at rounding level.
#[test]
fn a_flat_profile_ends_at_the_bound_with_a_note() {
    let mut set = dataset(7, 40, 0, 1.0, 1.0, 0.0);
    set.w = vec![1.0; 40];
    set.w[0] = 3.0;
    let fit = set.fit(&FitOptions::default());
    let note = fit
        .diagnostics
        .notes
        .iter()
        .find_map(|n| match n {
            Note::WeightPowerAtBound {
                delta,
                lower,
                upper,
            } => Some((*delta, *lower, *upper)),
            _ => None,
        })
        .expect("a note");
    assert_eq!(note, (fit.weight_power(), -30.0, 30.0));
    assert!(fit.weight_power() > 20.0 && fit.converged());
}

/// The fit's notes of the search: (whether it is at the bound, whether
/// probes failed).
fn search_notes(fit: &Fit) -> (bool, bool) {
    let notes = &fit.diagnostics.notes;
    (
        notes
            .iter()
            .any(|n| matches!(n, Note::WeightPowerAtBound { .. })),
        notes
            .iter()
            .any(|n| matches!(n, Note::WeightPowerSearchFailures { .. })),
    )
}

/// Weights near 1e100 on noise steeper than any power keeping `wᵢ^δ` within
/// 1e±250: the bracket narrows to ±2.4 (the spread alone gives ±7), no probe
/// fails, and the fit ends at the bound with its note and finite model and
/// robust SEs (before, XᵀWX overflowed at δ̂ near 3: NaN SEs or a failed refit).
#[test]
fn huge_weights_narrow_the_bracket_and_the_fit_ends_at_the_bound() {
    for (seed, delta0) in [(6, 5.0), (13, 4.0), (0, 3.0)] {
        let set = dataset(seed, 60, 0, 1e100, 1e4, delta0);
        let (lower, upper) = delta_bracket(&set.w);
        assert!(lower == -upper && upper > 2.0 && upper < 2.5, "{upper}");
        for se_type in [SeType::Model, SeType::Robust] {
            let fit = set.fit(&FitOptions {
                se_type: Some(se_type),
                ..FitOptions::default()
            });
            assert!(fit.converged() && fit.loglik().is_finite(), "seed {seed}");
            assert!(
                fit.se().iter().all(|v| v.is_finite()),
                "seed {seed}: {:?}",
                fit.se()
            );
            assert_eq!(search_notes(&fit), (true, false), "seed {seed}");
            assert!(
                (fit.weight_power() - upper).abs() < 1e-3,
                "{}",
                fit.weight_power()
            );
        }
    }
}

/// Weights in [1e12, 1e13] on noise steeper than the bracket: the spread alone
/// gives ±28, past the overflow of `w^δ` near 23.6, where the bound probe was
/// NaN and only the failures note fired; the bracket now stops at
/// ln(1e250)/ln(1e13) ≈ 19.2, and the fit there carries the bound note alone.
#[test]
fn large_weights_keep_the_bound_probe_finite() {
    let set = dataset(3, 60, 0, 1e12, 10.0, 25.0);
    let (_, upper) = delta_bracket(&set.w);
    assert!(
        (upper - 1e250_f64.ln() / (1e13_f64).ln()).abs() < 1e-3,
        "{upper}"
    );
    let fit = set.fit(&FitOptions::default());
    assert!(fit.converged() && fit.se().iter().all(|v| v.is_finite()));
    assert_eq!(search_notes(&fit), (true, false));
}

/// §3.3 (G9): `None` resolves to the δ fit only on a Gaussian REML fit with
/// weights not all equal, at most one grouping factor and at least 20 rows.
#[test]
fn none_resolves_to_estimate_only_in_scope() {
    let ols = dataset(7, 40, 0, 1.0, 20.0, 1.0);
    let mixed = dataset(17, 40, 4, 1.0, 20.0, 1.0);
    let crossed_ids = mixed
        .ids
        .clone()
        .extra((0..40).map(|i| (i % 3) as u32).collect());
    let crossed = ModelSpec::mixed(
        Family::Gaussian,
        ReStructure::intercept().extra(Grouping::crossed()),
    );
    let poisson = ModelSpec::fixed(Family::Poisson {
        link: PoissonLink::Log,
    });
    let counts: Vec<f64> = ols.y.iter().map(|v| v.abs().round()).collect();
    let equal = vec![2.0; 40];
    let reml = |r| FitOptions {
        reml: r,
        ..FitOptions::default()
    };
    let (ols_spec, mixed_spec) = (ols.spec(), mixed.spec());
    // (name, data, model, options, whether `None` resolves to the δ fit)
    type Row<'a> = (&'a str, Data<'a>, &'a ModelSpec, FitOptions, bool);
    let rows: Vec<Row> = vec![
        (
            "19 rows",
            Data::new(&ols.x[..38], &ols.y[..19], 19, 2).weights(&ols.w[..19]),
            &ols_spec,
            reml(None),
            false,
        ),
        (
            "weights all equal",
            ols.data(Some(&equal)),
            &ols_spec,
            reml(None),
            false,
        ),
        ("no weights", ols.data(None), &ols_spec, reml(None), false),
        (
            "two grouping factors",
            mixed.data(Some(&mixed.w)).ids(&crossed_ids),
            &crossed,
            reml(None),
            false,
        ),
        (
            "Poisson",
            Data::new(&ols.x, &counts, 40, 2).weights(&ols.w),
            &poisson,
            reml(None),
            false,
        ),
        (
            "ML",
            ols.data(Some(&ols.w)),
            &ols_spec,
            reml(Some(false)),
            false,
        ),
        (
            "OLS, reml None",
            ols.data(Some(&ols.w)),
            &ols_spec,
            reml(None),
            true,
        ),
        (
            "OLS, REML",
            ols.data(Some(&ols.w)),
            &ols_spec,
            reml(Some(true)),
            true,
        ),
        (
            "random intercept",
            mixed.data(Some(&mixed.w)),
            &mixed_spec,
            reml(None),
            true,
        ),
    ];
    for (name, data, spec, opts, estimate) in rows {
        let want = if estimate {
            WeightPower::Estimate
        } else {
            WeightPower::Fixed(1.0)
        };
        assert_eq!(resolve_weight_power(spec, &data, &opts), Ok(want), "{name}");
        let fit = crate::fit(&data, spec, &opts).unwrap();
        assert_eq!(fit.weight_power_estimated, estimate, "{name}");
    }
    // The in-scope call fits the power; the 19-row call takes the weights as given.
    assert_ne!(ols.fit(&FitOptions::default()).weight_power(), 1.0);
}

/// The weight-power notes as `(level, δ, factor)`: 1 for
/// `WeightPowerOutsideUnit` (factor NaN), 2 for `WeightPowerImplausible`.
fn levels(notes: &[Note]) -> Vec<(u8, f64, f64)> {
    notes
        .iter()
        .filter_map(|n| match *n {
            Note::WeightPowerOutsideUnit { delta } => Some((1, delta, f64::NAN)),
            Note::WeightPowerImplausible { delta, factor } => Some((2, delta, factor)),
            _ => None,
        })
        .collect()
}

/// [`implausibility_note`] as [`levels`] reads it.
fn level(delta: f64, w: &[f64], at_bound: bool) -> Vec<(u8, f64, f64)> {
    levels(
        &implausibility_note(delta, w, at_bound)
            .into_iter()
            .collect::<Vec<_>>(),
    )
}

/// The two levels on δ̂ alone: [0, 1] and [−1, 2] are closed, the factor
/// `s^|δ̂ − 1|` must pass 10, Level 2 replaces Level 1, and `WeightPowerAtBound`
/// silences Level 1 only.
#[test]
fn implausibility_levels_follow_delta_the_spread_and_the_bound() {
    let wide = [2.0, 2e6, 2e3]; // s = 1e6
    let narrow = [2.0, 6.0]; // s = 3
    for d in [0.0, 1.0, 0.5] {
        assert!(level(d, &wide, false).is_empty(), "{d}");
    }
    for d in [-1.0, 2.0, -0.5, 1.5] {
        let got = level(d, &wide, false);
        assert!(matches!(got[..], [(1, x, _)] if x == d), "{d}: {got:?}");
        assert!(level(d, &wide, true).is_empty(), "{d}");
    }
    // Outside [−1, 2]: factor 3^2 = 9 stays Level 1, 3^2.2 = 11.2 is Level 2.
    assert!(matches!(level(3.0, &narrow, false)[..], [(1, ..)]));
    assert!(level(3.0, &narrow, true).is_empty());
    let got = level(-1.2, &narrow, false);
    assert!(
        matches!(got[..], [(2, d, f)] if d == -1.2 && (f - 3f64.powf(2.2)).abs() < 1e-12),
        "{got:?}"
    );
    for at_bound in [false, true] {
        let got = level(3.0, &wide, at_bound);
        assert!(
            matches!(got[..], [(2, _, f)] if (f / 1e12 - 1.0).abs() < 1e-12),
            "{got:?}"
        );
    }
    // Far past the double range: the comparison holds in logs.
    let got = level(-30.0, &[1.0, 1e300], false);
    assert!(
        matches!(got[..], [(2, _, f)] if f == f64::INFINITY),
        "{got:?}"
    );
}

/// Through the fit: δ̂ well past 2 on weights spread 20 sets Level 2 with the
/// factor on the caller's weights (scaled near 1e-12); δ̂ in (1, 2] sets Level 1;
/// a fixed power outside [0, 1] sets neither.
#[test]
fn the_fit_sets_the_level_its_estimate_reaches() {
    let far = dataset(5, 200, 0, 1e-12, 20.0, 5.0);
    let fit = far.fit(&FitOptions::default());
    let delta = fit.weight_power();
    let got = levels(&fit.diagnostics.notes);
    let want = 20f64.powf((delta - 1.0).abs());
    assert!(
        matches!(got[..], [(2, d, f)] if d == delta && (f - want).abs() <= 1e-9 * want),
        "δ̂ = {delta}: {got:?}"
    );

    let near = dataset(5, 200, 0, 1.0, 20.0, 1.5);
    let fit = near.fit(&FitOptions::default());
    let delta = fit.weight_power();
    let got = levels(&fit.diagnostics.notes);
    assert!(
        delta > 1.0 && delta <= 2.0 && matches!(got[..], [(1, d, _)] if d == delta),
        "δ̂ = {delta}: {got:?}"
    );

    for power in [3.0, -0.5] {
        let fixed = far.fit(&FitOptions {
            weight_power: Some(WeightPower::Fixed(power)),
            ..FitOptions::default()
        });
        assert!(levels(&fixed.diagnostics.notes).is_empty(), "{power}");
    }
}
