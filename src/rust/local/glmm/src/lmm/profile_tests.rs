//! The profile objectives against the kernel they sit on: the lme4-scale
//! vector round-trips through θ̃ (prior weights included), a held coefficient
//! is the fit with that column moved into the offset, a zero SD on a centered
//! slope maps to a semidefinite θ̃, and a correlation matrix that is not PSD is
//! infeasible.

use super::profile::ProfileKernel;
use crate::fit::{fit_cold_keep_workspace, varcorr_block};
use crate::inference::ProfileParam;
use crate::lmm::LmmWorkspace;
use crate::standardize::{ColumnScaling, Units};
use crate::{Data, Family, Fit, FitOptions, GroupIds, ModelSpec, ReStructure};

/// The header and the rows of a quoted CSV, every field unquoted.
pub(crate) fn table(csv: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let split = |l: &str| -> Vec<String> {
        l.split(',')
            .map(|f| f.trim().trim_matches('"').to_string())
            .collect()
    };
    let head = split(lines.next().expect("a header"));
    (head, lines.map(split).collect())
}

/// Column `name` of `t` as numbers.
pub(crate) fn num(t: &(Vec<String>, Vec<Vec<String>>), name: &str) -> Vec<f64> {
    let c = t.0.iter().position(|h| h == name).expect("a column");
    t.1.iter().map(|r| r[c].parse().unwrap()).collect()
}

/// Column `name` of `t` as dense level ids, in order of first appearance.
pub(crate) fn levels(t: &(Vec<String>, Vec<Vec<String>>), name: &str) -> Vec<u32> {
    let c = t.0.iter().position(|h| h == name).expect("a column");
    let mut seen: Vec<&str> = Vec::new();
    t.1.iter()
        .map(|r| match seen.iter().position(|s| *s == r[c]) {
            Some(i) => i as u32,
            None => {
                seen.push(&r[c]);
                (seen.len() - 1) as u32
            }
        })
        .collect()
}

/// Row-major `n × p` design `[1, cols…]`.
pub(crate) fn design(cols: &[&[f64]]) -> Vec<f64> {
    let n = cols[0].len();
    let mut x = Vec::with_capacity(n * (cols.len() + 1));
    for i in 0..n {
        x.push(1.0);
        x.extend(cols.iter().map(|c| c[i]));
    }
    x
}

/// `y ~ x + (1 + slopes | g)` with `slopes` design columns.
pub(crate) fn slope_model(slopes: Vec<u32>) -> ModelSpec {
    ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes,
            extra_groupings: vec![],
        }),
    }
}

/// The ML options the profile fits under.
pub(crate) fn ml_opts() -> FitOptions {
    FitOptions {
        reml: Some(false),
        satterthwaite: false,
        condvar: false,
        ..FitOptions::default()
    }
}

/// One ML fit's inputs.
pub(crate) struct Case {
    pub(crate) x: Vec<f64>,
    pub(crate) y: Vec<f64>,
    pub(crate) p: usize,
    pub(crate) ids: GroupIds,
    pub(crate) w: Option<Vec<f64>>,
    pub(crate) model: ModelSpec,
}

impl Case {
    /// The case's rows, ids and weights.
    pub(crate) fn data(&self) -> Data<'_> {
        let data = Data::new(&self.x, &self.y, self.y.len(), self.p).ids(&self.ids);
        match &self.w {
            Some(w) => data.weights(w),
            None => data,
        }
    }

    /// Fits the case by ML as `fit` does and hands `f` the fit with the LMM
    /// workspace and column map it left.
    pub(crate) fn with_ml<R>(
        &self,
        f: impl FnOnce(&Fit, &mut LmmWorkspace, &ColumnScaling, &Units) -> R,
    ) -> R {
        let (fit, mut fws) = fit_cold_keep_workspace(&self.data(), &self.model, &ml_opts());
        assert!(fit.converged(), "the ML fit converges");
        let (ws, map, units) = fws.lmm_parts().expect("a Gaussian mixed fit");
        f(&fit, ws, map, units)
    }
}

/// The kernel over a case's workspace, slots in declaration order, since no
/// case here is reordered by the size rule.
pub(crate) fn kernel<'a>(
    ws: &'a mut LmmWorkspace,
    map: &'a ColumnScaling,
    units: &'a Units,
) -> ProfileKernel<'a> {
    let slots: Vec<usize> = (0..1 + ws.kernel.groupings().extra_q.len()).collect();
    ProfileKernel::new(ws, map, units, &slots)
}

pub(crate) fn primary(g: Vec<u32>) -> GroupIds {
    GroupIds {
        primary: g,
        extra: vec![],
    }
}

/// sleepstudy, `Reaction ~ 1 + Days + (1 + Days | Subject)`; the fit centers
/// `Days` inside.
pub(crate) fn sleepstudy_ml() -> Case {
    let t = table(include_str!(
        "../../../validation/data/empirical/sleepstudy.csv"
    ));
    Case {
        x: design(&[&num(&t, "Days")]),
        y: num(&t, "Reaction"),
        p: 2,
        ids: primary(levels(&t, "Subject")),
        w: None,
        model: slope_model(vec![1]),
    }
}

/// `lmm_slope.csv`, `y ~ 1 + x + (1 + x | g)` with prior weights `10·w`. The
/// file's `w` average about 1, which leaves the kernel's weight scale `s` at
/// 1; ten times them gives `s = 8`, so `σ_int² = σ²/s` is tested.
fn lmm_slope_weighted_ml() -> Case {
    let t = table(include_str!(
        "../../../validation/data/simulated/lmm_slope.csv"
    ));
    Case {
        x: design(&[&num(&t, "x")]),
        y: num(&t, "y"),
        p: 2,
        ids: primary(levels(&t, "g")),
        w: Some(num(&t, "w").iter().map(|w| 10.0 * w).collect()),
        model: slope_model(vec![1]),
    }
}

#[test]
fn lme4_scale_round_trips_through_the_kernel_theta() {
    let mut ran = 0;
    for case in [sleepstudy_ml(), lmm_slope_weighted_ml()] {
        case.with_ml(|fit, ws, map, units| {
            if case.w.is_some() {
                assert_eq!(units.s(), 8.0, "the weights move the kernel's scale");
            }
            let (sd, corr) = fit.stddev_corr(0);
            let want = [sd[0], corr[1][0], sd[1], fit.dispersion.sqrt()];
            let mut k = kernel(ws, map, units);
            // lme4's θ order within the term: sd, cor, sd, then σ.
            assert_eq!(
                k.params(),
                [
                    ProfileParam::Sd { group: 0, term: 0 },
                    ProfileParam::Cor {
                        group: 0,
                        row: 1,
                        col: 0
                    },
                    ProfileParam::Sd { group: 0, term: 1 },
                    ProfileParam::Sigma,
                ]
            );
            let v = k.estimate();
            assert_eq!(v.len(), want.len());
            for (g, w) in v.iter().zip(want) {
                assert!(((g - w) / w).abs() <= 1e-13, "{g} {w}");
            }
            let mut th = vec![0.0; v.len() - 1];
            assert!(k.theta_of(&v, &mut th));
            assert!(
                th.iter()
                    .zip(k.theta_hat())
                    .all(|(a, b)| (a - b).abs() <= 1e-10),
                "{th:?} {:?}",
                k.theta_hat()
            );
            let base = k.kernel_deviance_at_hat();
            let got = k.var_deviance(&v);
            assert!((got - base).abs() <= 1e-9 * base.abs(), "{got} {base}");
        });
        ran += 1;
    }
    assert_eq!(ran, 2);
}

#[test]
fn a_fixed_coefficient_is_the_dropped_column_fit() {
    let t = table(include_str!(
        "../../../validation/data/simulated/lmm_intercept.csv"
    ));
    let (xv, y, w, g) = (num(&t, "x"), num(&t, "y"), num(&t, "w"), levels(&t, "g"));
    let n = y.len();
    let ones = vec![1.0; n];
    let full = Case {
        x: design(&[&xv]),
        y: y.clone(),
        p: 2,
        ids: primary(g.clone()),
        w: Some(w.clone()),
        model: crate::test_support::intercept_only_spec(),
    };
    let mut points = 0;
    full.with_ml(|fit, ws, map, units| {
        let mut k = kernel(ws, map, units);
        let hat = k.theta_hat().to_vec();
        for j in 0..2 {
            // Column j moves into the offset; the other column stays.
            let (xj, other) = if j == 0 { (&ones, &xv) } else { (&xv, &ones) };
            for b in [fit.beta[j] - 2.0 * fit.se[j], fit.beta[j] + 3.0 * fit.se[j]] {
                let dropped = Case {
                    x: other.clone(),
                    y: y.iter().zip(xj).map(|(yi, xi)| yi - b * xi).collect(),
                    p: 1,
                    ids: primary(g.clone()),
                    w: Some(w.clone()),
                    model: crate::test_support::intercept_only_spec(),
                };
                dropped.with_ml(|_, ws2, _, units2| {
                    // Each kernel's deviance is on its own response scale:
                    // the caller's is `+2N·ln c` (ML).
                    let to_caller = |c: f64| 2.0 * n as f64 * c.ln();
                    for f in [1.0, 0.5, 2.0] {
                        let th: Vec<f64> = hat.iter().map(|t| f * t).collect();
                        let got = k.beta_deviance(&th, j, b) + to_caller(units.c());
                        let want = ws2.kernel.deviance(&th) + to_caller(units2.c());
                        assert!(
                            (got - want).abs() <= 1e-9 * want.abs(),
                            "j {j} b {b} θ×{f}: {got} {want}"
                        );
                        points += 1;
                    }
                });
            }
        }
    });
    assert_eq!(points, 12);
}

#[test]
fn an_sd_at_zero_on_a_centered_slope_block() {
    sleepstudy_ml().with_ml(|_, ws, map, units| {
        let s = units.s();
        let g = ws.kernel.groupings().clone();
        assert!(g.block_has_shift(0, map), "Days is centered inside");
        let mut k = kernel(ws, map, units);
        let mut v = k.estimate();
        v[2] = 0.0; // sd_Days|Subject
        let mut th = vec![0.0; 3];
        assert!(k.theta_of(&v, &mut th));
        let sigma_int_sq = v[3] * v[3] / s;
        let vc = varcorr_block(&th, 2, sigma_int_sq, &g, 0, map);
        let trace = vc[0] + vc[2];
        assert!(vc[2] <= 1e-12 * trace, "Var(Days) {}", vc[2]);
        let want = v[0] * v[0];
        assert!(((vc[0] - want) / want).abs() <= 1e-12, "{} {want}", vc[0]);
        let base = k.kernel_deviance_at_hat();
        let d = k.var_deviance(&v);
        assert!(d.is_finite() && d > base, "{d} {base}");
    });
}

#[test]
fn a_non_psd_correlation_is_infeasible() {
    let t = table(include_str!(
        "../../../validation/data/empirical/Machines_dummy.csv"
    ));
    let case = Case {
        x: design(&[&num(&t, "MachineB"), &num(&t, "MachineC")]),
        y: num(&t, "score"),
        p: 3,
        ids: primary(levels(&t, "Worker")),
        w: None,
        model: slope_model(vec![1, 2]),
    };
    case.with_ml(|_, ws, map, units| {
        let mut k = kernel(ws, map, units);
        let mut v = k.estimate();
        // sd0, cor10, cor20, sd1, cor21, sd2, σ.
        assert_eq!(v.len(), 7);
        for i in [0, 3, 5] {
            assert!(v[i] > 0.0, "sd at {i} is {}", v[i]);
        }
        v[1] = 0.95;
        v[2] = 0.95;
        v[4] = -0.95;
        let mut th = vec![0.0; 6];
        assert!(!k.theta_of(&v, &mut th));
        assert_eq!(k.var_deviance(&v), f64::INFINITY);
    });
}

#[test]
fn a_parameter_held_at_its_estimate_gives_the_base() {
    sleepstudy_ml().with_ml(|fit, ws, map, units| {
        let mut k = kernel(ws, map, units);
        let base = k.kernel_deviance_at_hat();
        let v = k.estimate();
        let mut held = 0;
        for i in 0..v.len() {
            let mut start = v.clone();
            let (d, ok) = k.hold_var(i, v[i], &mut start);
            assert!(ok && (d - base).abs() <= 1e-6, "coordinate {i}: {d} {base}");
            assert_eq!(start[i], v[i]);
            held += 1;
        }
        let hat = k.theta_hat().to_vec();
        for j in 0..2 {
            let mut start = hat.clone();
            let (d, ok) = k.hold_beta(j, fit.beta[j], &mut start);
            assert!(ok && (d - base).abs() <= 1e-6, "β{j}: {d} {base}");
            // Off the estimate the held minimum lies between the base and the
            // value at θ̂, where it started.
            let b = fit.beta[j] + 2.0 * fit.se[j];
            let at_hat = k.beta_deviance(&hat, j, b);
            let mut start = hat.clone();
            let (d, ok) = k.hold_beta(j, b, &mut start);
            assert!(
                ok && base < d && d <= at_hat,
                "β{j} at {b}: {base} {d} {at_hat}"
            );
            held += 2;
        }
        assert_eq!(held, 8);
    });
}

/// A small SD that the fit did not pin can follow a held parameter past
/// 1000 times its estimate: its box is the fit's, `THETA_HI·σ̂_int` in caller
/// units (`σ̂` here, unweighted). Balanced one-way, 6 groups of 4, within-group residuals ±1 and
/// group means ±c: the ML estimates are `σ̂² = SSW/(K(n − 1)) = 4/3` and
/// `τ̂² = c² − σ̂²/n`, with `c` set so that `τ̂ = 3e-4·σ̂`. With σ held at
/// `σ̂/10` the group means' spread goes to τ, which then wants about
/// `√(1/3)`, 1.7 times `1000·τ̂`.
#[test]
fn a_small_sd_is_not_boxed_at_a_multiple_of_its_estimate() {
    let (k_groups, n_per) = (6u32, 4usize);
    let sigma_sq: f64 = 4.0 / 3.0;
    let tau = 3e-4 * sigma_sq.sqrt();
    let c = (sigma_sq / n_per as f64 + tau * tau).sqrt();
    let (mut y, mut g) = (vec![], vec![]);
    for grp in 0..k_groups {
        let mean = if grp % 2 == 0 { c } else { -c };
        for i in 0..n_per {
            y.push(mean + if i % 2 == 0 { 1.0 } else { -1.0 });
            g.push(grp);
        }
    }
    let case = Case {
        x: vec![1.0; y.len()],
        y,
        p: 1,
        ids: primary(g),
        w: None,
        model: crate::test_support::intercept_only_spec(),
    };
    case.with_ml(|_, ws, map, units| {
        let theta_hat = ws.theta[0];
        assert!(
            theta_hat > crate::lmm::PIN_THETA && theta_hat < 1e-3,
            "θ̂ {theta_hat} is small and unpinned"
        );
        let mut k = kernel(ws, map, units);
        let v = k.estimate();
        let (sd_hat, sigma_hat) = (v[0], v[1]);
        let mut start = v.clone();
        let (_, ok) = k.hold_var(1, sigma_hat / 10.0, &mut start);
        assert!(ok);
        assert!(
            start[0] > 1.5 * 1000.0 * sd_hat,
            "sd {} against 1000·ŝd {}",
            start[0],
            1000.0 * sd_hat
        );
    });
}

/// A random-slope SD's box is the fit's for that slope, which divides by the
/// column's standardization scale. sleepstudy with `Days·1e-4`: the column's
/// scale is about 2.9e-4, so the slope SD estimate (about 5.7e4) is past
/// `THETA_HI·σ̂` (about 2.6e4) although the fit's θ̃ is about 0.7. Holding any
/// coordinate at its estimate gives the base, with the slope SD at its
/// estimate in the argmin, past `THETA_HI·σ̂`.
#[test]
fn a_slope_sd_keeps_the_fits_box_on_a_small_scale_column() {
    let mut case = sleepstudy_ml();
    for row in case.x.chunks_mut(2) {
        row[1] *= 1e-4;
    }
    case.with_ml(|_, ws, map, units| {
        let mut k = kernel(ws, map, units);
        let base = k.kernel_deviance_at_hat();
        let v = k.estimate();
        let (sd_days, sigma_hat) = (v[2], v[3]);
        assert!(
            sd_days > crate::lmm::THETA_HI * sigma_hat,
            "{sd_days} {sigma_hat}"
        );
        let mut held = 0;
        for i in 0..v.len() {
            let mut start = v.clone();
            let (d, ok) = k.hold_var(i, v[i], &mut start);
            assert!(ok && (d - base).abs() <= 1e-6, "coordinate {i}: {d} {base}");
            assert!(
                start[2] > crate::lmm::THETA_HI * sigma_hat,
                "coordinate {i}: sd_Days {} capped",
                start[2]
            );
            held += 1;
        }
        assert_eq!(held, 4);
    });
}
