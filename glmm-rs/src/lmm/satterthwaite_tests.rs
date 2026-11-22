//! The pass against closed forms: ν with no free coordinate, `K − 1` on a
//! balanced one-way design, the small-θ̂ designs' closed forms, `C` against
//! `Fit::vcov`, and the assembly's failure path. Then the gates: the exact
//! pass against the value route on the same fits, the value route's step
//! sweep that pins [`ETA`] and [`DF_FD_REL`], dense against sparse and QR
//! against Gram at one θ, the invariances, and the boundary cases (the
//! canonical correlation boundary, a wide block pinned without
//! canonicalization).

use super::{
    assemble, free_coordinates, pass, pass_at, residual_nu, Collected, DerivRoute,
    SatterthwaiteState,
};
#[cfg(feature = "formula")]
use super::{restore_response, shift_response, ETA};
use crate::fit::common_tests::{lcg, sim_clustered};
use crate::fit::fit_keep_workspace;
use crate::fit::SizedSpec;
#[cfg(feature = "formula")]
use crate::lmm::fit_lmm_unhinted;
#[cfg(feature = "formula")]
use crate::lmm::LmmKernel;
use crate::lmm::{LmmGroupings, LmmWorkspace};
use crate::standardize::{ColumnScaling, Units};
use crate::test_support::intercept_only_spec;
use crate::SeFor;
use crate::{Data, Family, Fit, FitOptions, GroupIds, ModelSpec, ReStructure};
use faer::Mat;

/// Two routes to the same df that differ only in rounding: `C` read off the
/// mapped factor against `C` read off the unmapped one through `M = B·U⁻¹`,
/// amplified by the `1/(2h)` of the central difference (about `1e-15/ETA`).
#[cfg(feature = "formula")]
const ROUTE_REL: f64 = 1e-9;

/// `df_fd_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`): the value route's
/// finite-difference band on a df, measured by
/// `formula_gates::step_sweep_is_flat_at_eta`.
/// Change together.
#[cfg(feature = "formula")]
pub(crate) const DF_FD_REL: f64 = 4e-5;

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs()
}

/// Every coefficient's df, each checked to lie in `(0, ν]` (gate 5).
fn coef_df_in_range(st: &SatterthwaiteState) -> Vec<f64> {
    let df = st.coef_df();
    for (i, &d) in df.iter().enumerate() {
        assert!(d > 0.0 && d <= st.nu, "df[{i}] {d} outside (0, {}]", st.nu);
    }
    df
}

/// Fits `y ~ x + (1 | g)` (row-major `x`, `n × p`, intercept included) the way
/// `fit_cold` does and hands `f` the fit with the LMM workspace and column map
/// it left.
#[allow(clippy::too_many_arguments)] // one fixture's arrays plus the callback
fn with_fit<R>(
    x: &[f64],
    y: &[f64],
    g: &[u32],
    n: usize,
    p: usize,
    weights: Option<Vec<f64>>,
    f: impl FnOnce(&Fit, &mut LmmWorkspace, &ColumnScaling, &Units) -> R,
) -> R {
    let model = intercept_only_spec();
    let ids = GroupIds {
        primary: g.to_vec(),
        extra: vec![],
    };
    let opts = FitOptions {
        se_for: SeFor::Only((0..p as u32).collect()),
        ..FitOptions::default()
    };
    let mut data = Data::new(x, y, n, p).ids(&ids);
    if let Some(w) = &weights {
        data = data.weights(w);
    }
    let (fit, mut fws) = fit_keep_workspace(&data, &model, None, &opts).unwrap();
    let (ws, map, units) = fws
        .lmm_parts()
        .expect("a Gaussian mixed fit has an LMM workspace");
    f(&fit, ws, map, units)
}

/// `y ~ 1 + x1 + x2 + (1 | g)`: 12 groups of 3 to 8 rows, `x1` far from
/// standardized and `x2` correlated with it, so the column map `B` and the
/// orthogonalization `U` are both far from the identity. Returns
/// `(x, y, g, n, p)`.
fn mixed_fixture() -> (Vec<f64>, Vec<f64>, Vec<u32>, usize, usize) {
    let mut s = 7u64;
    let (mut x, mut y, mut g) = (vec![], vec![], vec![]);
    for grp in 0..12u32 {
        let b = 0.8 * lcg(&mut s);
        for _ in 0..(3 + grp % 6) {
            let x1 = 50.0 + 20.0 * lcg(&mut s);
            let x2 = 0.05 * x1 + 0.5 * lcg(&mut s);
            x.extend([1.0, x1, x2]);
            y.push(1.0 + 0.03 * x1 - 0.4 * x2 + b + 0.5 * lcg(&mut s));
            g.push(grp);
        }
    }
    let n = y.len();
    (x, y, g, n, 3)
}

/// Prior weights in `[10, 40]`, so `family::weight_scale` is not 1.
fn weights(n: usize) -> Vec<f64> {
    let mut s = 11u64;
    (0..n).map(|_| 25.0 + 15.0 * lcg(&mut s)).collect()
}

/// Dyestuff2 (lme4): six batches of five, where the REML fit is singular, so
/// θ̂ pins at exactly 0, no coordinate is free and every df is `N − p`.
#[test]
fn n_free_zero_gives_nu() {
    let csv = include_str!("../../../validation/data/empirical/Dyestuff2.csv");
    let (mut y, mut g) = (vec![], vec![]);
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let (batch, yield_) = line.split_once(',').unwrap();
        g.push(u32::from(batch.trim_matches('"').as_bytes()[0] - b'A'));
        y.push(yield_.parse::<f64>().unwrap());
    }
    let n = y.len();
    assert_eq!(n, 30);
    with_fit(&vec![1.0; n], &y, &g, n, 1, None, |fit, ws, map, units| {
        assert!(fit.converged());
        assert_eq!(ws.theta, [0.0], "Dyestuff2's θ̂ pins at exactly 0");
        let st = pass(ws, map, units);
        assert!(st.jac_theta.is_empty());
        assert_eq!(coef_df_in_range(&st), [29.0]);
    });
}

/// Balanced `y ~ 1 + (1 | g)`: the intercept's Satterthwaite df is `K − 1`
/// exactly, at the REML optimum `σ̂² = MSW`, `σ̂_b² = (MSB − MSW)/m`.
#[test]
fn balanced_one_way_intercept_df_is_k_minus_one() {
    const K: usize = 10;
    const M: usize = 6;
    let mut s = 3u64;
    let (mut y, mut g) = (vec![], vec![]);
    for grp in 0..K as u32 {
        let b = lcg(&mut s);
        for _ in 0..M {
            y.push(1.5 * b + lcg(&mut s));
            g.push(grp);
        }
    }
    let n = K * M;
    let mean = y.iter().sum::<f64>() / n as f64;
    let group_means: Vec<f64> = y
        .chunks(M)
        .map(|c| c.iter().sum::<f64>() / M as f64)
        .collect();
    let msb =
        M as f64 * group_means.iter().map(|m| (m - mean).powi(2)).sum::<f64>() / (K - 1) as f64;
    let msw = y
        .chunks(M)
        .zip(&group_means)
        .map(|(c, m)| c.iter().map(|v| (v - m).powi(2)).sum::<f64>())
        .sum::<f64>()
        / (K * (M - 1)) as f64;
    assert!(msb > msw, "the fixture needs an interior REML optimum");
    let theta_caller = ((msb - msw) / M as f64 / msw).sqrt();

    with_fit(&vec![1.0; n], &y, &g, n, 1, None, |fit, ws, map, units| {
        assert!(fit.converged());
        let mut theta = vec![theta_caller];
        let groupings = ws.kernel.groupings();
        let mut scratch = vec![0.0; groupings.re_map_scratch_len()];
        groupings.theta_caller_to_design(&mut theta, &mut scratch, map);
        let at_reml = coef_df_in_range(&pass_at(ws, &theta, map, units))[0];
        assert!(
            rel(at_reml, (K - 1) as f64) <= 1e-6,
            "df at the analytic θ̂ {at_reml}"
        );
        // The fitted θ̂ is within RHO_END = 1e-6 of it, which moves df at first
        // order.
        let at_fit = coef_df_in_range(&pass(ws, map, units))[0];
        assert!(
            rel(at_fit, (K - 1) as f64) <= 1e-5,
            "df at the fitted θ̂ {at_fit}"
        );
    });
}

/// The deviance and the diagonal of `C` at `θ`, `C` read off the factor mapped
/// to the caller's columns and units (the route `Fit::vcov` takes).
#[cfg(feature = "formula")]
fn deviance_and_var(
    kernel: &mut LmmKernel,
    theta: &[f64],
    map: &ColumnScaling,
    units: &Units,
    p: usize,
) -> (f64, Vec<f64>) {
    let f = kernel.deviance(theta);
    kernel.map_factor_to_design();
    let (l, s2) = (kernel.factor(), kernel.sigma_sq() * units.var_scale());
    let var = (0..p)
        .map(|t| {
            let mut u = vec![0.0; p];
            for i in 0..p {
                let acc = map.rhs(t, i) - (0..i).map(|k| l[(i, k)] * u[k]).sum::<f64>();
                u[i] = acc / l[(i, i)];
            }
            s2 * u.iter().map(|v| v * v).sum::<f64>()
        })
        .collect();
    (f, var)
}

/// A canonical correlation-boundary block (`q = 2`, trailing column zero)
/// drops that column's coordinate; a column with any non-zero entry is kept,
/// even under a zero diagonal.
#[test]
fn free_coordinates_drop_exact_zero_columns() {
    let spec = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: 4,
        extra_counts: vec![],
        tail_fill: 0.0,
    };
    // vech column-major: θ̃ = (Λ₀₀, Λ₁₀, Λ₁₁).
    let g = LmmGroupings::from_cluster_spec(&spec, &[1]);
    assert_eq!(free_coordinates(&g, &[0.774, 0.505, 0.0]), [0, 1]);
    assert_eq!(free_coordinates(&g, &[0.774, 0.505, 0.2]), [0, 1, 2]);
    assert_eq!(free_coordinates(&g, &[0.0, 0.4, 0.3]), [0, 1, 2]);
    assert_eq!(free_coordinates(&g, &[0.0, 0.0, 0.3]), [2]);
    assert!(free_coordinates(&g, &[0.0, 0.0, 0.0]).is_empty());
}

/// The θ entries a grouping's terms hold at zero are never coordinates of the
/// pass: `(x || g)` differentiates its two variances, not the covariance
/// between the terms, `(1 + x1 | g) + (0 + x2 | g)` the four entries of its
/// two terms, `(0 + x | g)` the slope alone. On a fitted `(d || g)` (`d` a
/// 0/1 column, so it is scaled without being centered) the state carries one
/// `∂C/∂θ` per free variance and every df lies in `(0, ν]`.
#[test]
fn satterthwaite_skips_structural_coordinates() {
    use crate::ReTerm;
    let groupings = |re: ReStructure, slope_cols: &[usize]| {
        let spec = SizedSpec {
            spec: ModelSpec {
                family: Family::Gaussian,
                re: Some(re),
            },
            n_primary: 4,
            extra_counts: vec![],
            tail_fill: 0.0,
        };
        LmmGroupings::from_cluster_spec(&spec, slope_cols)
    };
    let uncorr = groupings(ReStructure::intercept().slopes([1]).uncorrelated(), &[1]);
    assert_eq!(free_coordinates(&uncorr, &[0.7, 0.0, 0.4]), [0, 2]);
    assert_eq!(free_coordinates(&uncorr, &[0.7, 0.0, 0.0]), [0]);
    let split = groupings(
        ReStructure::intercept().terms([
            ReTerm::intercept().slopes([1]),
            ReTerm::no_intercept().slopes([2]),
        ]),
        &[1, 2],
    );
    // vech column-major, q = 3: (0,0) (1,0) (2,0) (1,1) (2,1) (2,2).
    assert_eq!(
        free_coordinates(&split, &[0.7, 0.2, 0.0, 0.5, 0.0, 0.3]),
        [0, 1, 3, 5]
    );
    let no_int = groupings(
        ReStructure::intercept().terms([ReTerm::no_intercept().slopes([1])]),
        &[1],
    );
    assert_eq!(free_coordinates(&no_int, &[0.0, 0.0, 0.4]), [2]);

    let mut s = 23u64;
    let (mut x, mut y, mut g) = (vec![], vec![], vec![]);
    for grp in 0..16u32 {
        let (b0, b1) = (0.8 * lcg(&mut s), 0.8 * lcg(&mut s));
        for i in 0..10 {
            let d = f64::from(i % 2);
            x.extend([1.0, d]);
            y.push(1.0 + 0.5 * d + b0 + b1 * d + 0.5 * lcg(&mut s));
            g.push(grp);
        }
    }
    let (n, p) = (y.len(), 2);
    let ids = GroupIds {
        primary: g,
        extra: vec![],
    };
    let model = ModelSpec::mixed(
        Family::Gaussian,
        ReStructure::intercept().slopes([1]).uncorrelated(),
    );
    let opts = FitOptions {
        se_for: SeFor::Only(vec![0, 1]),
        ..FitOptions::default()
    };
    let (fit, mut fws) =
        fit_keep_workspace(&Data::new(&x, &y, n, p).ids(&ids), &model, None, &opts).unwrap();
    let (ws, map, units) = fws.lmm_parts().expect("an LMM workspace");
    assert!(fit.converged());
    assert_eq!(ws.theta[1], 0.0);
    assert!(ws.theta[0] != 0.0 && ws.theta[2] != 0.0, "{:?}", ws.theta);
    let st = pass(ws, map, units);
    assert!(!st.is_failed());
    assert_eq!(st.jac_theta.len(), 2);
    coef_df_in_range(&st);
}

/// `C` at θ̂ is `Fit::vcov` (the full target block), unweighted and weighted:
/// the `M·C̃·Mᵀ` map needs no prior-weight scale factor, and τ̂ is the
/// dispersion the fit reports.
#[test]
fn state_c_matches_fit_vcov() {
    let (x, y, g, n, p) = mixed_fixture();
    for w in [None, Some(weights(n))] {
        let weighted = w.is_some();
        with_fit(&x, &y, &g, n, p, w, |fit, ws, map, units| {
            assert!(fit.converged());
            assert_eq!(weighted, units.s() != 1.0);
            let st = pass(ws, map, units);
            assert!(!st.is_failed());
            assert_eq!(st.route, DerivRoute::Exact);
            assert_eq!((st.dropped, st.negative), (0, 0), "an interior fit");
            // τ̂ is on the scale the fit reports σ̂² (the weight scale undone).
            assert!(rel(st.tau, fit.dispersion) <= 1e-12, "τ̂ {}", st.tau);
            for a in 0..p {
                for b in 0..p {
                    let want = fit.vcov()[a * p + b];
                    let scale = (fit.vcov()[a * p + a] * fit.vcov()[b * p + b]).sqrt();
                    assert!(
                        (st.c[(a, b)] - want).abs() <= 1e-12 * scale,
                        "weighted {weighted}: C[{a},{b}] {} against vcov {want}",
                        st.c[(a, b)]
                    );
                }
            }
        });
    }
}

/// `ws.theta` is read, never written, and the kernel's statistics come back
/// from the response shift bit for bit.
#[test]
fn pass_reads_theta_and_never_writes() {
    let (x, y, g, n, p) = mixed_fixture();
    with_fit(&x, &y, &g, n, p, None, |_, ws, map, units| {
        let theta = ws.theta.clone();
        let f_before = ws.kernel.deviance(&theta).to_bits();
        let st = pass(ws, map, units);
        assert!(!st.is_failed());
        let after: Vec<u64> = ws.theta.iter().map(|t| t.to_bits()).collect();
        let before: Vec<u64> = theta.iter().map(|t| t.to_bits()).collect();
        assert_eq!(after, before);
        // The shifted response is put back: the kernel evaluates as before.
        assert_eq!(ws.kernel.deviance(&theta).to_bits(), f_before);
    });
}

/// Two free coordinates on the quadratic `f(θ) = ½θᵀHθ`, where central
/// differences are exact up to rounding, with `C(θ) = I + diag(θ)` (`p = 2`,
/// so `∂C/∂θ_k = e_k e_kᵀ` and coefficient k's `g` is `e_k`), τ̂ = 1, ν = 10.
fn toy_collected(hess: [[f64; 2]; 2], h: [f64; 2]) -> Collected {
    let f =
        |a: f64, b: f64| 0.5 * (hess[0][0] * a * a + 2.0 * hess[0][1] * a * b + hess[1][1] * b * b);
    let c = |a: f64, b: f64| Mat::from_fn(2, 2, |i, j| if i == j { 1.0 + [a, b][i] } else { 0.0 });
    let [h0, h1] = h;
    Collected {
        free: vec![0, 1],
        h: h.to_vec(),
        f0: f(0.0, 0.0),
        f_axis: vec![[f(h0, 0.0), f(-h0, 0.0)], [f(0.0, h1), f(0.0, -h1)]],
        f_mixed: vec![[f(h0, h1), f(-h0, -h1)]],
        exact: false,
        hess: Mat::zeros(0, 0),
        jac: vec![],
        c0: c(0.0, 0.0),
        c_axis: vec![[c(h0, 0.0), c(-h0, 0.0)], [c(0.0, h1), c(0.0, -h1)]],
        tau: 1.0,
        nu: 10.0,
    }
}

/// The `n_free = 2` assembly against its closed form: `A_eff = 2·blockdiag(H⁻¹,
/// τ̂²/ν)` and `df_k = 1/((H⁻¹)_kk + 1/ν)`. With `H = [[2, 1], [1, 2]]` both df
/// are `1/(2/3 + 1/10) = 30/23`; with `H = [[2, 1], [1, 3]]` and unequal steps
/// they are `1/(3/5 + 1/10) = 10/7` and `1/(2/5 + 1/10) = 2`, so a mixed
/// divisor other than `2hᵢhⱼ` or a swapped `i`/`j` fails.
#[test]
fn assemble_two_coordinates_matches_closed_form() {
    for (hess, h, h_inv, want) in [
        (
            [[2.0, 1.0], [1.0, 2.0]],
            [0.1, 0.1],
            [[2.0 / 3.0, -1.0 / 3.0], [-1.0 / 3.0, 2.0 / 3.0]],
            [30.0 / 23.0, 30.0 / 23.0],
        ),
        (
            [[2.0, 1.0], [1.0, 3.0]],
            [0.1, 0.2],
            [[3.0 / 5.0, -1.0 / 5.0], [-1.0 / 5.0, 2.0 / 5.0]],
            [10.0 / 7.0, 2.0],
        ),
    ] {
        let st = assemble(&toy_collected(hess, h));
        assert!(!st.is_failed());
        assert_eq!((st.dropped, st.negative), (0, 0));
        for (i, row) in h_inv.iter().enumerate() {
            for (j, &hij) in row.iter().enumerate() {
                let a = st.a_eff[(i, j)];
                assert!((a - 2.0 * hij).abs() <= 1e-12, "A_eff[{i},{j}] {a}");
            }
            assert_eq!((st.a_eff[(i, 2)], st.a_eff[(2, i)]), (0.0, 0.0));
        }
        assert!((st.a_eff[(2, 2)] - 0.2).abs() <= 1e-15);
        for (k, (df, w)) in coef_df_in_range(&st).iter().zip(want).enumerate() {
            assert!(rel(*df, w) <= 1e-12, "H {hess:?}: df[{k}] {df} against {w}");
        }
    }
}

#[test]
fn assemble_nonfinite_point_gives_nan_df() {
    let toy = || toy_collected([[2.0, 1.0], [1.0, 3.0]], [0.1, 0.2]);
    let mut col = toy();
    col.f_mixed[0][1] = f64::INFINITY;
    let st = assemble(&col);
    assert!(st.is_failed());
    assert!(st.coef_df().iter().all(|d| d.is_nan()));

    let mut col = toy();
    col.c_axis[1][0][(1, 1)] = f64::NAN;
    let st = assemble(&col);
    assert!(st.is_failed());
    assert!(st.coef_df().iter().all(|d| d.is_nan()));
}

// ---------------------------------------------------------------------------
// Gate 5: invariances
// ---------------------------------------------------------------------------

/// Gate 5 on a dense and a sparse route: rescaling a slope covariate by 10³
/// refits the same model, and every df stays within the route's band of
/// `src/fit/standardize_tests.rs::unit_invariance_on_every_route` (1e-6 dense,
/// 1e-4 sparse; measured 2026-10-11: 2.7e-7 and 3.5e-6, the refit's θ̂ moving
/// within the search's tolerance). With every coordinate pinned the pass
/// returns ν exactly. The dense fixture is that test's `lmm_dense` (a `q = 2`
/// block whose fit is on the correlation boundary), the sparse one its
/// `lmm_sparse` (a crossed `q = 5` block beside a `q = 2` primary).
#[test]
fn slope_rescale_keeps_df_and_all_pinned_gives_nu() {
    let (x, y, ids, _) = sim_clustered(include_str!(
        "../../../validation/data/simulated/sim_gamma.csv"
    ));
    let dense = (
        x,
        y,
        3,
        ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        GroupIds {
            primary: ids,
            extra: vec![],
        },
    );
    let (x, y, _, p, mut model, ids) = crate::sparse::tests::sparse_gamma_design();
    model.re.as_mut().unwrap().slopes = vec![1];
    let model = ModelSpec {
        family: Family::Gaussian,
        ..model
    };
    for (sparse, band, (x, y, p, model, ids)) in
        [(false, 1e-6, dense), (true, 1e-4, (x, y, p, model, ids))]
    {
        let n = y.len();
        let df_at = |scale: f64| {
            let mut x = x.clone();
            for row in x.chunks_exact_mut(p) {
                row[1] *= scale;
            }
            let opts = FitOptions {
                se_for: SeFor::Only((0..p as u32).collect()),
                ..FitOptions::default()
            };
            let (fit, mut fws) =
                fit_keep_workspace(&Data::new(&x, &y, n, p).ids(&ids), &model, None, &opts)
                    .unwrap();
            assert!(fit.converged(), "sparse {sparse}, x1·{scale}");
            assert_eq!(fws.is_lmm_sparse(), sparse);
            let (ws, map, units) = fws.lmm_parts().expect("an LMM workspace");
            let df = coef_df_in_range(&pass(ws, map, units));
            let pinned = vec![0.0; ws.theta.len()];
            let nu = residual_nu(ws);
            assert_eq!(
                coef_df_in_range(&pass_at(ws, &pinned, map, units)),
                vec![nu; p]
            );
            df
        };
        let (base, scaled) = (df_at(1.0), df_at(1e3));
        for (i, (&s, &b)) in scaled.iter().zip(&base).enumerate() {
            assert!(
                rel(s, b) <= band,
                "sparse {sparse}: df[{i}] {s} against {b}"
            );
        }
    }
}

/// Gates 3 and 4 and the boundary cases whose fixtures are built through the
/// formula frontend (the `formula` feature).
#[cfg(feature = "formula")]
mod formula_gates {
    use super::*;
    use crate::formula::{lower, Column, Table};
    use crate::linalg::SeqLlt;
    use crate::lmm::per_group_gradient_armed;
    use crate::lmm::satterthwaite::{collect, hessian, pass_value_at};
    use crate::sparse::SparseLmmWorkspace;
    use crate::test_support::csv_table;
    use crate::{Grouping, GroupingRelation};

    // ---------------------------------------------------------------------------
    // Gate 3: step stability
    // ---------------------------------------------------------------------------

    /// The step sweep of gate 3, [`ETA`] its third step.
    const SWEEP: [f64; 4] = [3e-5, 1e-4, 3e-4, 1e-3];

    /// The band of `dense_and_sparse_kernels_agree` on the exact pass, two
    /// exact passes that differ only in rounding: twice the largest measured
    /// gap, rounded up to one significant figure (measured 2026-11-21,
    /// x86_64: 8.7e-14, the design with 150 crossed levels).
    const DENSE_SPARSE_REL: f64 = 2e-13;

    /// The band of `qr_pass_df_match_the_gram_pass`, two exact passes that
    /// differ only in rounding: twice the largest measured gap, rounded up to
    /// one significant figure (measured 2026-11-22, plain wasm32-wasip1:
    /// 2.7e-14, `nest2` df[0]; x86_64: 3.9e-15).
    const QR_GRAM_REL: f64 = 6e-14;

    /// The pass's `collect` on the value route at the step factor `eta`, or
    /// with `exact` on the exact pass where it covers the kernel.
    pub(super) fn collect_eta(
        ws: &mut LmmWorkspace,
        theta: &[f64],
        map: &ColumnScaling,
        units: &Units,
        eta: f64,
        exact: bool,
    ) -> Collected {
        let nu = residual_nu(ws);
        collect(
            &mut ws.kernel,
            theta,
            &ws.recovery.beta_tilde,
            units,
            map,
            nu,
            eta,
            exact,
        )
    }

    /// Lowers `formula` on `table`, fits it the way `fit_cold` does and hands `f`
    /// the fit with the LMM workspace and column map it left.
    pub(super) fn with_formula_fit<R>(
        formula: &str,
        table: &Table,
        f: impl FnOnce(&Fit, &mut LmmWorkspace, &ColumnScaling, &Units) -> R,
    ) -> R {
        let lo = lower(formula, table, Family::Gaussian).expect("the fixture lowers");
        let (fit, mut fws) =
            fit_keep_workspace(&lo.data(), &lo.model, None, &FitOptions::default()).unwrap();
        let (ws, map, units) = fws
            .lmm_parts()
            .expect("a Gaussian mixed fit has an LMM workspace");
        f(&fit, ws, map, units)
    }

    /// [`with_formula_fit`] with the QR kernel started on the fit's rows.
    fn with_formula_fit_qr<R>(
        formula: &str,
        table: &Table,
        f: impl FnOnce(&Fit, &mut LmmWorkspace, &ColumnScaling, &Units) -> R,
    ) -> R {
        let lo = lower(formula, table, Family::Gaussian).expect("the fixture lowers");
        let data = lo.data();
        let (fit, mut fws) =
            fit_keep_workspace(&data, &lo.model, None, &FitOptions::default()).unwrap();
        assert!(fws.lmm_start_qr(&data), "a shape the QR kernel covers");
        let (ws, map, units) = fws
            .lmm_parts()
            .expect("a Gaussian mixed fit has an LMM workspace");
        f(&fit, ws, map, units)
    }

    /// A uniform draw with unit variance.
    fn unit(s: &mut u64) -> f64 {
        3f64.sqrt() * lcg(s)
    }

    /// A draw of a Gaussian shape of the speed campaign
    /// (`validation/campaigns/speed-campaign/prep.R`, `sim_cell`), balanced:
    /// 300 rows over 60 primary groups of 5, one block per entry of `q`. The
    /// first is the primary block, `(1 + x1 + … | g1)`; every later one is a
    /// random intercept, nested in the one before when `nested` (each parent
    /// level split into 3), crossed otherwise (30 levels drawn per row). The sd of
    /// block `k`'s coordinate `d` is `0.9^d·[1, 0.8, 0.7, 0.6, 0.6, 0.5][k]`
    /// (`nearzero`: the primary block's last one is 0.02, the campaign's
    /// near-singular regime), with correlation 0.2 inside a block; the fixed
    /// effects are `0.5 + 0.8·x1 − 0.5·x2`, the residual sd 0.6. Uniform draws
    /// with unit variance stand in for the campaign's normal ones. Returns the
    /// table and its formula.
    pub(super) fn speed_shape(
        q: &[usize],
        nested: bool,
        nearzero: bool,
        seed: u64,
    ) -> (Table, String) {
        const N: usize = 300;
        const GROUPS: usize = 60;
        const CROSSED: f64 = 30.0;
        let mut s = seed;
        let nx = q.iter().max().unwrap().saturating_sub(1).max(1);
        let x: Vec<Vec<f64>> = (0..nx)
            .map(|_| (0..N).map(|_| unit(&mut s)).collect())
            .collect();
        let mut y: Vec<f64> = (0..N)
            .map(|i| {
                0.5 + [0.8, -0.5]
                    .iter()
                    .zip(&x)
                    .map(|(b, xj)| b * xj[i])
                    .sum::<f64>()
            })
            .collect();
        let (mut columns, mut terms) = (vec![], vec![]);
        let mut level: Vec<usize> = (0..N).map(|i| i % GROUPS).collect();
        for (k, &qk) in q.iter().enumerate() {
            if k > 0 {
                // `lcg` is U(−1, 1).
                level = if nested {
                    level
                        .iter()
                        .map(|&parent| 3 * parent + ((lcg(&mut s) + 1.0) * 1.5) as usize)
                        .collect()
                } else {
                    (0..N)
                        .map(|_| ((lcg(&mut s) + 1.0) * 0.5 * CROSSED) as usize)
                        .collect()
                };
            }
            let mut sd: Vec<f64> = (0..qk)
                .map(|d| 0.9f64.powi(d as i32) * [1.0, 0.8, 0.7, 0.6, 0.6, 0.5][k])
                .collect();
            if nearzero && k == 0 {
                sd[qk - 1] = 0.02;
            }
            let b: Vec<Vec<f64>> = (0..=*level.iter().max().unwrap())
                .map(|_| {
                    let common = unit(&mut s);
                    sd.iter()
                        .map(|sd| sd * (0.2f64.sqrt() * common + 0.8f64.sqrt() * unit(&mut s)))
                        .collect()
                })
                .collect();
            for (i, yi) in y.iter_mut().enumerate() {
                let bl = &b[level[i]];
                *yi += bl[0] + (1..qk).map(|d| bl[d] * x[d - 1][i]).sum::<f64>();
            }
            let name = format!("g{}", k + 1);
            let slopes: String = (1..qk).map(|d| format!(" + x{d}")).collect();
            terms.push(format!("(1{slopes} | {name})"));
            let labels: Vec<String> = level.iter().map(|l| l.to_string()).collect();
            columns.push((name, Column::factor_from_labels(&labels)));
        }
        for yi in &mut y {
            *yi += 0.6 * unit(&mut s);
        }
        let fixed: String = (1..=nx).map(|j| format!(" + x{j}")).collect();
        let formula = format!("y ~ 1{fixed} + {}", terms.join(" + "));
        for (j, xj) in x.into_iter().enumerate() {
            columns.push((format!("x{}", j + 1), Column::Numeric(xj)));
        }
        columns.push(("y".to_string(), Column::Numeric(y)));
        (Table { columns, n: N }, formula)
    }

    /// A Gaussian design with crossed random intercepts: `rows` rows over
    /// `groups` primary levels (row `i` in level `i mod groups`), and one
    /// crossed factor per entry of `levels`, each row's level drawn uniformly,
    /// `y ~ 1 + x + (1 | g1) + (1 | c1) + …`. Random-effect sds 1 (primary)
    /// and 0.7 (each crossed factor), fixed effects `0.5 + 0.8·x`, residual sd
    /// 0.6; uniform draws with unit variance. Returns the table and its
    /// formula.
    pub(super) fn crossed_design(
        groups: usize,
        levels: &[usize],
        rows: usize,
        seed: u64,
    ) -> (Table, String) {
        let mut s = seed;
        let x: Vec<f64> = (0..rows).map(|_| unit(&mut s)).collect();
        let mut y: Vec<f64> = x.iter().map(|xi| 0.5 + 0.8 * xi).collect();
        let mut columns = vec![];
        let mut terms = vec!["(1 | g1)".to_string()];
        let b: Vec<f64> = (0..groups).map(|_| unit(&mut s)).collect();
        let g: Vec<String> = (0..rows)
            .map(|i| {
                y[i] += b[i % groups];
                (i % groups).to_string()
            })
            .collect();
        columns.push(("g1".to_string(), Column::factor_from_labels(&g)));
        for (k, &l) in levels.iter().enumerate() {
            let b: Vec<f64> = (0..l).map(|_| 0.7 * unit(&mut s)).collect();
            let lab: Vec<String> = (0..rows)
                .map(|i| {
                    // Every level holds at least one row: the first `l` rows
                    // take each level once.
                    let lv = if i < l {
                        i
                    } else {
                        (((lcg(&mut s) + 1.0) * 0.5 * l as f64) as usize).min(l - 1)
                    };
                    y[i] += b[lv];
                    lv.to_string()
                })
                .collect();
            let name = format!("c{}", k + 1);
            terms.push(format!("(1 | {name})"));
            columns.push((name, Column::factor_from_labels(&lab)));
        }
        for yi in &mut y {
            *yi += 0.6 * unit(&mut s);
        }
        columns.push(("x".to_string(), Column::Numeric(x)));
        columns.push(("y".to_string(), Column::Numeric(y)));
        (
            Table { columns, n: rows },
            format!("y ~ 1 + x + {}", terms.join(" + ")),
        )
    }

    /// A balanced design whose REML θ̂ is `theta` (a lower-triangular `Θ` by
    /// columns: one entry for `y ~ x + (1 | g)`, three for `y ~ x + (1 + x | g)`)
    /// up to the search's tolerance, for a block just above `PIN_THETA`. `K`
    /// groups of `M` rows share one centered `x` of unit variance, so every
    /// group's `G = Z′Z` is `M·I`. The residual is projected off `[1, x]` inside
    /// each group, and the group effects are whitened across groups and scaled
    /// to the sample covariance `σ̂²·(ΘΘ′ + G⁻¹)`, `σ̂²` the residual mean square
    /// REML reads: the between-group estimate less `σ̂²·G⁻¹` is then `σ̂²·ΘΘ′`
    /// with no sampling noise. Fixed effects `0.5 + 0.8·x`.
    pub(super) fn small_theta_shape(theta: &[f64], seed: u64) -> (Table, String) {
        const K: usize = 30;
        const M: usize = 1000;
        let q = if theta.len() == 1 { 1 } else { 2 };
        let mut s = seed;
        let xs: Vec<f64> = (0..M)
            .map(|j| (j as f64 - (M as f64 - 1.0) / 2.0) / ((M * M - 1) as f64 / 12.0).sqrt())
            .collect();
        let mut e = vec![];
        for _ in 0..K {
            let mut eg: Vec<f64> = (0..M).map(|_| unit(&mut s)).collect();
            let mean = eg.iter().sum::<f64>() / M as f64;
            let slope = eg.iter().zip(&xs).map(|(v, x)| v * x).sum::<f64>() / M as f64;
            for (v, x) in eg.iter_mut().zip(&xs) {
                *v -= mean + slope * x;
            }
            e.extend(eg);
        }
        // Residual df on this design: n − K·q − (p − q), p = 2.
        let s2 = e.iter().map(|v| v * v).sum::<f64>() / (K * M - K * q - (2 - q)) as f64;
        // 2×2 lower Cholesky; a 1×1 matrix is padded with an identity row.
        let chol = |a: [[f64; 2]; 2]| {
            let l00 = a[0][0].sqrt();
            let l10 = a[1][0] / l00;
            [[l00, 0.0], [l10, (a[1][1] - l10 * l10).sqrt()]]
        };
        let (t0, t1, t2) = match *theta {
            [t] => (t, 0.0, 0.0),
            [a, b, c] => (a, b, c),
            _ => panic!("one or three θ entries"),
        };
        let pad = if q == 1 { 1.0 } else { 0.0 };
        let gi = 1.0 / M as f64;
        let lt = chol([
            [s2 * (t0 * t0 + gi), s2 * t0 * t1],
            [
                s2 * t0 * t1,
                s2 * (t1 * t1 + t2 * t2 + gi) * (1.0 - pad) + pad,
            ],
        ]);
        let mut z: Vec<[f64; 2]> = (0..K)
            .map(|_| [unit(&mut s), if q == 2 { unit(&mut s) } else { 0.0 }])
            .collect();
        let zbar = [0, 1].map(|d| z.iter().map(|v| v[d]).sum::<f64>() / K as f64);
        for v in &mut z {
            v[0] -= zbar[0];
            v[1] -= zbar[1];
        }
        let cov = |a: usize, b: usize| z.iter().map(|v| v[a] * v[b]).sum::<f64>() / (K - 1) as f64;
        let ls = chol([[cov(0, 0), cov(0, 1)], [cov(1, 0), cov(1, 1) + pad]]);
        let (mut y, mut x, mut g) = (vec![], vec![], vec![]);
        for (k, v) in z.iter().enumerate() {
            let w0 = v[0] / ls[0][0];
            let w1 = (v[1] - ls[1][0] * w0) / ls[1][1];
            let b0 = lt[0][0] * w0;
            let b1 = if q == 2 {
                lt[1][0] * w0 + lt[1][1] * w1
            } else {
                0.0
            };
            for (j, &xj) in xs.iter().enumerate() {
                y.push(0.5 + 0.8 * xj + b0 + b1 * xj + e[k * M + j]);
                x.push(xj);
                g.push(k.to_string());
            }
        }
        let formula = if q == 1 {
            "y ~ x + (1 | g)"
        } else {
            "y ~ x + (1 + x | g)"
        };
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(y)),
                ("x".into(), Column::Numeric(x)),
                ("g".into(), Column::factor_from_labels(&g)),
            ],
            n: K * M,
        };
        (table, formula.into())
    }

    /// Lowers `formula` on `table`, with the prior weights of column `weights`
    /// when given and `FitOptions::reml` set to `reml`, fits it the way
    /// `fit_cold` does and hands `f` the fit with the LMM workspace and column
    /// map it left.
    pub(super) fn with_fit_w<R>(
        formula: &str,
        table: &Table,
        weights: Option<&str>,
        reml: Option<bool>,
        f: impl FnOnce(&Fit, &mut LmmWorkspace, &ColumnScaling, &Units) -> R,
    ) -> R {
        let mut lo = lower(formula, table, Family::Gaussian).expect("the fixture lowers");
        let fit_opts = FitOptions {
            reml,
            ..FitOptions::default()
        };
        if let Some(w) = weights {
            let w = table.columns.iter().find_map(|(name, col)| match col {
                Column::Numeric(v) if name == w => Some(v.clone()),
                _ => None,
            });
            lo.weights = Some(w.expect("a numeric weight column"));
        }
        let (fit, mut fws) = fit_keep_workspace(&lo.data(), &lo.model, None, &fit_opts).unwrap();
        let (ws, map, units) = fws
            .lmm_parts()
            .expect("a Gaussian mixed fit has an LMM workspace");
        f(&fit, ws, map, units)
    }

    /// The per-group fixtures, `(name, table, formula, weight column)`:
    /// sleepstudy, the speed campaign's `q3s` draw and its near-singular draw,
    /// the `q = 8` `sim_max_q_slope` design, the weighted `lmm_slope` and
    /// `all_ones` fixtures and the correlation boundary.
    pub(super) fn per_group_fixtures() -> Vec<(String, Table, String, Option<&'static str>)> {
        let mut v = vec![];
        for (name, csv, formula, factor, weights) in [
            (
                "sleepstudy",
                include_str!("../../../validation/data/empirical/sleepstudy.csv"),
                "Reaction ~ Days + (Days | Subject)",
                "Subject",
                None,
            ),
            (
                "sim_max_q_slope",
                include_str!("../../../validation/data/simulated/sim_max_q_slope.csv"),
                "y ~ 1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 \
                 + (1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 | g1)",
                "g1",
                None,
            ),
            (
                "lmm_slope (weighted)",
                include_str!("../../../validation/data/simulated/lmm_slope.csv"),
                "y ~ 1 + x + (1 + x | g)",
                "g",
                Some("w"),
            ),
            (
                "all_ones (weighted)",
                include_str!("../../../validation/data/simulated/all_ones.csv"),
                "y ~ 1 + x + (1 + x | g)",
                "g",
                Some("w"),
            ),
            (
                "sim_corr_boundary",
                include_str!("../../../validation/data/simulated/sim_corr_boundary.csv"),
                "y ~ x + (x | g)",
                "g",
                None,
            ),
        ] {
            v.push((
                name.to_string(),
                csv_table(csv, &[factor]),
                formula.to_string(),
                weights,
            ));
        }
        for (name, nearzero, seed) in [
            ("q3s base", false, 811_001),
            ("q3s near-zero", true, 811_002),
        ] {
            let (table, formula) = speed_shape(&[3], false, nearzero, seed);
            v.push((name.to_string(), table, formula, None));
        }
        v
    }

    /// The df of [`small_theta_shape`]'s two designs against their closed
    /// forms: one-way `[K − 1, n − K − 1] = [29, 29969]`, `(1 + x | g)` with
    /// block maximum 1e-3 `[29, 29]`. Both hold exactly at the REML optimum of
    /// these balanced designs (the between-group estimate is the sample
    /// covariance, Wishart on `K − 1` df). At the generating θ̂ mapped into
    /// kernel units the band is 1e-9 relative: the exact terms cancel by
    /// about `4KM/H_p ≈ 1e3`, so the round-off is about `1e3·ε`. At the fitted
    /// θ̂ the band is 1e-2: the df move at first order with θ̂ (θ̂ × 1.001
    /// moves them by 1e-3 relative), and the search stops within about
    /// `RHO_END` = 1e-6 of θ̂ = 1e-3, a relative 1e-3.
    #[test]
    fn small_theta_df_match_the_closed_form() {
        const K: f64 = 30.0;
        const N: f64 = 30_000.0;
        let mut failures = vec![];
        for (name, theta, want) in [
            ("one-way θ 1e-3", &[1e-3][..], [K - 1.0, N - K - 1.0]),
            (
                "slope block max 1e-3",
                &[1e-3, 3e-4, 5e-4][..],
                [K - 1.0, K - 1.0],
            ),
        ] {
            let (table, formula) = small_theta_shape(theta, 811_003);
            with_formula_fit(&formula, &table, |fit, ws, map, units| {
                assert!(fit.converged(), "{name}");
                let mut th = theta.to_vec();
                let groupings = ws.kernel.groupings();
                let mut scratch = vec![0.0; groupings.re_map_scratch_len()];
                groupings.theta_caller_to_design(&mut th, &mut scratch, map);
                let root_s = units.s().sqrt();
                th.iter_mut().for_each(|t| *t *= root_s);
                let st = pass_at(ws, &th, map, units);
                let at_closed = coef_df_in_range(&st);
                let at_fit = coef_df_in_range(&pass(ws, map, units));
                for (at, df, band) in [
                    ("closed-form θ̂", &at_closed, 1e-9),
                    ("fitted θ̂", &at_fit, 1e-2),
                ] {
                    for (i, (&d, &w)) in df.iter().zip(&want).enumerate() {
                        let r = rel(d, w);
                        eprintln!(
                            "SMALL-THETA {name} {at} ({:?} route): df[{i}] {d} closed form {w} rel {r:.2e} band {band:e}",
                            st.route
                        );
                        if r > band {
                            failures.push(format!("{name} at the {at}: df[{i}] {d} against {w} (rel {r:.2e} > {band:e})"));
                        }
                    }
                }
            });
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// A balanced two-way crossed design `y ~ 1 + (1 | a) + (1 | b)`, `A = 20`
    /// levels of `a` by `B = 30` of `b` with `M = 40` rows per cell, built so
    /// that the mean squares are exactly `MS_E` (from the draws), `MS_A =
    /// MS_E·(1 + M·B·θ_a²)` and `MS_B = MS_E·(1 + M·A·θ_b²)`: the residual is
    /// projected off both main effects, and each factor's effects are centered
    /// and scaled to their mean square. At that interior optimum the REML θ̂
    /// is the ANOVA estimate `θ_a`, `θ_b`. `b`, with more levels, is the
    /// primary grouping and `a` the crossed one. Returns the table, the three
    /// mean squares and the residual df.
    fn two_way_shape(theta_a: f64, theta_b: f64, seed: u64) -> (Table, [f64; 3], f64) {
        const A: usize = 20;
        const B: usize = 30;
        const M: usize = 40;
        let mut s = seed;
        let mut e: Vec<f64> = (0..A * B * M).map(|_| unit(&mut s)).collect();
        let at = |i: usize, j: usize, k: usize| (i * B + j) * M + k;
        let mean_a: Vec<f64> = (0..A)
            .map(|i| (0..B * M).map(|t| e[at(i, t / M, t % M)]).sum::<f64>() / (B * M) as f64)
            .collect();
        let mean_b: Vec<f64> = (0..B)
            .map(|j| (0..A * M).map(|t| e[at(t / M, j, t % M)]).sum::<f64>() / (A * M) as f64)
            .collect();
        let grand = e.iter().sum::<f64>() / e.len() as f64;
        for i in 0..A {
            for j in 0..B {
                for k in 0..M {
                    e[at(i, j, k)] -= mean_a[i] + mean_b[j] - grand;
                }
            }
        }
        let df_e = (A * B * M - A - B + 1) as f64;
        let ms_e = e.iter().map(|v| v * v).sum::<f64>() / df_e;
        // Effects centered, with Σ eff² = target·(L − 1)/(rows per level).
        let effects = |l: usize, per: usize, ms: f64, s: &mut u64| {
            let mut v: Vec<f64> = (0..l).map(|_| unit(s)).collect();
            let m = v.iter().sum::<f64>() / l as f64;
            v.iter_mut().for_each(|x| *x -= m);
            let ss = v.iter().map(|x| x * x).sum::<f64>();
            let k = (ms * (l - 1) as f64 / per as f64 / ss).sqrt();
            v.iter_mut().for_each(|x| *x *= k);
            v
        };
        let ms_a = ms_e * (1.0 + (M * B) as f64 * theta_a * theta_a);
        let ms_b = ms_e * (1.0 + (M * A) as f64 * theta_b * theta_b);
        let (alpha, beta) = (
            effects(A, B * M, ms_a, &mut s),
            effects(B, A * M, ms_b, &mut s),
        );
        let (mut y, mut a, mut b) = (vec![], vec![], vec![]);
        for i in 0..A {
            for j in 0..B {
                for k in 0..M {
                    y.push(alpha[i] + beta[j] + e[at(i, j, k)]);
                    a.push(format!("a{i}"));
                    b.push(format!("b{j}"));
                }
            }
        }
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(y)),
                ("a".into(), Column::factor_from_labels(&a)),
                ("b".into(), Column::factor_from_labels(&b)),
            ],
            n: A * B * M,
        };
        (table, [ms_a, ms_b, ms_e], df_e)
    }

    /// Check 1b: the intercept df of [`two_way_shape`] against the classical
    /// Satterthwaite df of `MS_A + MS_B − MS_E`, `(MS_A + MS_B − MS_E)² /
    /// (MS_A²/(A − 1) + MS_B²/(B − 1) + MS_E²/df_E)`, which the REML df equal
    /// at the ANOVA optimum. The crossed factor `a` has `θ̂_a` = 1e-3 next to
    /// its level size `M·B` = 1200 (`M·B·θ̂²` = 1.2e-3, as the one-way design
    /// of `small_theta_df_match_the_closed_form`). On the dense kernel (the
    /// fit's own, `a` crossed) and the sparse kernel (fitted on the design as
    /// given, so its θ̃ is the caller's, with `a` its primary and `b` crossed),
    /// with the bands of the one-way design: 1e-9 at the closed-form θ̂, 1e-2
    /// at the fitted θ̂.
    #[test]
    fn two_way_crossed_df_match_the_closed_form() {
        let (theta_a, theta_b) = (1e-3, 0.5);
        let (table, [ms_a, ms_b, ms_e], df_e) = two_way_shape(theta_a, theta_b, 811_005);
        let want = (ms_a + ms_b - ms_e).powi(2)
            / (ms_a * ms_a / 19.0 + ms_b * ms_b / 29.0 + ms_e * ms_e / df_e);
        let formula = "y ~ 1 + (1 | a) + (1 | b)";
        let mut failures = vec![];
        let mut check = |kernel: &str, at: &str, df: f64, band: f64, route: DerivRoute| {
            let r = rel(df, want);
            eprintln!(
                "TWO-WAY {kernel} {at} ({route:?} route): df {df} closed form {want} rel {r:.2e} band {band:e}"
            );
            assert_eq!(route, DerivRoute::Exact, "{kernel}");
            if r > band {
                failures.push(format!(
                    "{kernel} at the {at}: df {df} against {want} (rel {r:.2e} > {band:e})"
                ));
            }
        };
        with_formula_fit(formula, &table, |fit, ws, map, units| {
            assert!(fit.converged());
            assert!(matches!(ws.kernel, LmmKernel::Dense { .. }));
            assert_eq!(
                ws.kernel.groupings().n_primary,
                30,
                "b is the primary grouping"
            );
            // Kernel θ order: the primary `b`, then `a`.
            let mut th = vec![theta_b, theta_a];
            let groupings = ws.kernel.groupings();
            let mut scratch = vec![0.0; groupings.re_map_scratch_len()];
            groupings.theta_caller_to_design(&mut th, &mut scratch, map);
            let root_s = units.s().sqrt();
            th.iter_mut().for_each(|t| *t *= root_s);
            let st = pass_at(ws, &th, map, units);
            check(
                "dense",
                "closed-form θ̂",
                coef_df_in_range(&st)[0],
                1e-9,
                st.route,
            );
            let st = pass(ws, map, units);
            check(
                "dense",
                "fitted θ̂",
                coef_df_in_range(&st)[0],
                1e-2,
                st.route,
            );
        });
        // `both_kernels` keeps the formula's order: `a` is its primary.
        let (_, mut sparse) = both_kernels(&table, formula);
        assert!(matches!(sparse.kernel, LmmKernel::Sparse { .. }));
        assert_eq!(
            sparse.kernel.groupings().n_primary,
            20,
            "a is the primary grouping"
        );
        let st = pass_at(
            &mut sparse,
            &[theta_a, theta_b],
            ColumnScaling::IDENTITY,
            Units::IDENTITY,
        );
        check(
            "sparse",
            "closed-form θ̂",
            coef_df_in_range(&st)[0],
            1e-9,
            st.route,
        );
        let theta = sparse.theta.clone();
        let st = pass_at(
            &mut sparse,
            &theta,
            ColumnScaling::IDENTITY,
            Units::IDENTITY,
        );
        check(
            "sparse",
            "fitted θ̂",
            coef_df_in_range(&st)[0],
            1e-2,
            st.route,
        );
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// The step sweep's fixtures, `(name, table, formula)`, in the order of
    /// [`step_sweep_is_flat_at_eta`]'s table.
    fn sweep_fixtures() -> Vec<(String, Table, String)> {
        // The balanced one-way design of `balanced_one_way_intercept_df_is_k_minus_one`,
        // uncentered: 300 + y/4.
        const K: usize = 10;
        const M: usize = 6;
        let mut s = 3u64;
        let (mut y, mut g) = (vec![], vec![]);
        for grp in 0..K {
            let b = lcg(&mut s);
            for _ in 0..M {
                y.push(300.0 + (1.5 * b + lcg(&mut s)) / 4.0);
                g.push(grp.to_string());
            }
        }
        let toy = Table {
            columns: vec![
                ("y".into(), Column::Numeric(y)),
                ("g".into(), Column::factor_from_labels(&g)),
            ],
            n: K * M,
        };
        let mut v = vec![("toy 300 + y/4".into(), toy, "y ~ 1 + (1 | g)".into())];
        for (name, csv, formula, factors) in [
            (
                "sim_dynrange_lmm",
                include_str!("../../../validation/data/simulated/sim_dynrange_lmm.csv"),
                "y ~ 1 + u + w + (1 | g)",
                &["g"][..],
            ),
            (
                "sim_nested_crossed_mix",
                include_str!("../../../validation/data/simulated/sim_nested_crossed_mix.csv"),
                "y ~ 1 + x + (1 | g1/g2) + (1 | c1)",
                &["g1", "g2", "c1"][..],
            ),
            (
                "sim_near_singular",
                include_str!("../../../validation/data/simulated/sim_near_singular.csv"),
                "y ~ x + (1 | g)",
                &["g"][..],
            ),
            (
                "sim_corr_boundary",
                include_str!("../../../validation/data/simulated/sim_corr_boundary.csv"),
                "y ~ x + (x | g)",
                &["g"][..],
            ),
        ] {
            v.push((name.into(), csv_table(csv, factors), formula.into()));
        }
        for (name, q, nested) in [
            ("int1", &[1][..], false),
            ("nest2", &[1, 1][..], true),
            ("q2s", &[2][..], false),
            ("q3s", &[3][..], false),
            ("cross4", &[1; 4][..], false),
            ("cross6", &[1; 6][..], false),
        ] {
            // The near-zero seed is one whose draw keeps a free coordinate.
            for (regime, nearzero, seed) in [("base", false, 811_001), ("near-zero", true, 811_002)]
            {
                let (table, formula) = speed_shape(q, nested, nearzero, seed);
                v.push((format!("{name} {regime}"), table, formula));
            }
        }
        v
    }

    /// Gate 3: the df is flat in the step at [`ETA`], the band [`DF_FD_REL`].
    ///
    /// Each fixture is fitted, then its df computed at the fitted θ̂ with every
    /// step of [`SWEEP`]. The spread between two steps is the largest relative
    /// change of a coefficient's df; `|f₀|` is the deviance the pass re-evaluates
    /// at θ̂ (on the shifted response); `min θ̃` the smallest free coordinate.
    /// Measured 2026-11-18 (macOS arm64) with each step factor times its
    /// block's largest `|θ̃|`:
    ///
    /// | fixture | n_free | min θ̃ | \|f₀\| | 3e-5↔1e-4 | 1e-4↔3e-4 | 3e-4↔1e-3 |
    /// |---|---|---|---|---|---|---|
    /// | toy `300 + y/4` | 1 | 1.64 | 4.64e1 | 2.2e-6 | 2.7e-7 | 1.0e-6 |
    /// | sim_dynrange_lmm | 1 | 14.7 | 4.99e3 | 3.2e-4 | 6.6e-5 | 1.0e-5 |
    /// | sim_nested_crossed_mix | 3 | 1.45 | 6.84e2 | 1.3e-3 | 3.8e-5 | 1.9e-5 |
    /// | sim_near_singular | 1 | 0.196 | 1.83e1 | 7.2e-6 | 4.6e-7 | 1.1e-7 |
    /// | sim_corr_boundary | 2 | 0.289 | 1.48e1 | 7.4e-7 | 3.3e-7 | 1.8e-7 |
    /// | int1 base | 1 | 1.67 | 1.36e2 | 1.4e-6 | 1.2e-7 | 8.3e-7 |
    /// | int1 near-zero | 1 | 0.177 | 1.40e2 | 2.3e-5 | 1.8e-6 | 3.2e-7 |
    /// | nest2 base | 2 | 1.30 | 8.34e0 | 4.2e-6 | 1.6e-7 | 4.1e-7 |
    /// | nest2 near-zero | 2 | 0.378 | 6.14e1 | 5.6e-5 | 5.1e-7 | 2.4e-7 |
    /// | q2s base | 3 | 0.112 | 4.38e2 | 1.0e-6 | 1.9e-7 | 1.1e-6 |
    /// | q2s near-zero | 2 | 0.0816 | 1.16e2 | 4.5e-7 | 7.1e-8 | 8.0e-7 |
    /// | q3s base | 6 | 0.462 | 3.48e2 | 5.6e-7 | 1.1e-7 | 1.4e-6 |
    /// | q3s near-zero | 6 | 0.0207 | 1.01e1 | 1.4e-6 | 1.7e-6 | 1.7e-5 |
    /// | cross4 base | 4 | 0.808 | 3.35e2 | 3.4e-6 | 3.2e-7 | 1.1e-6 |
    /// | cross4 near-zero | 3 | 1.09 | 4.71e2 | 4.8e-7 | 1.4e-7 | 1.1e-6 |
    /// | cross6 base | 6 | 0.828 | 2.67e2 | 1.1e-5 | 1.6e-6 | 1.3e-6 |
    /// | cross6 near-zero | 6 | 0.0721 | 3.59e2 | 3.3e-5 | 1.9e-5 | 1.9e-6 |
    ///
    /// Across every fixture the flattest pair is 3e-4↔1e-3 (largest spread
    /// 1.9e-5, against 6.6e-5 for 1e-4↔3e-4 and 1.3e-3 for 3e-5↔1e-4). Below 3e-4
    /// the deviance's round-off dominates where it comes from the random effects,
    /// which the response shift cannot remove (sim_dynrange_lmm,
    /// sim_nested_crossed_mix); above it truncation grows as η². So `ETA = 3e-4`,
    /// the lower step of that pair, and the gate is the spread over the pair.
    /// `DF_FD_REL = 4e-5` is the house rule (twice the maximum, rounded up to one
    /// significant figure) on the largest spread here, 1.9e-5
    /// (sim_nested_crossed_mix); the largest df gap of `exact_pass_agrees_with_value_route`,
    /// 1.75e-5 on plain wasm32-wasip1 (2026-11-22), is below it. As a
    /// diagnostic only, Richardson extrapolation from
    /// the steps 5e-4 and 1e-3 puts the error of the df at 3e-4 at 1.8e-5 or below
    /// on every fixture (sim_nested_crossed_mix), against 1.9e-5 at 1e-3 (q3s
    /// near-zero).
    #[test]
    fn step_sweep_is_flat_at_eta() {
        assert_eq!(SWEEP[2], ETA, "ETA is the sweep's third step");
        let bits = |v: &[f64]| v.iter().map(|d| d.to_bits()).collect::<Vec<_>>();
        for (name, table, formula) in sweep_fixtures() {
            with_formula_fit(&formula, &table, |fit, ws, map, units| {
                assert!(fit.converged(), "{name}");
                let theta = ws.theta.clone();
                let st = crate::lmm::satterthwaite::pass_value_at(ws, &theta, map, units);
                assert!(!st.jac_theta.is_empty(), "{name}: no free coordinate");
                let df: Vec<Vec<f64>> = SWEEP
                    .iter()
                    .map(|&eta| {
                        coef_df_in_range(&assemble(&collect_eta(
                            ws, &theta, map, units, eta, false,
                        )))
                    })
                    .collect();
                assert_eq!(bits(&df[2]), bits(&coef_df_in_range(&st)), "{name}");
                for (i, (&at, &up)) in df[2].iter().zip(&df[3]).enumerate() {
                    assert!(
                        rel(up, at) <= DF_FD_REL,
                        "{name}: df[{i}] {at} at η = {}, {up} at {}",
                        SWEEP[2],
                        SWEEP[3]
                    );
                }
            });
        }
    }

    /// The fixture whose `H_p` is not compared, because the value route's own
    /// round-off puts an entry outside the band: on `cross6 near-zero` the primary
    /// coordinate is θ̃ = 0.072, and its curvature at the step `ETA·θ̃` is off
    /// by 4.3e-4 (REML) and 3.6e-4 (ML) relative. The exact `H_00` 8.908827 is
    /// what the value route converges to at larger steps (8.908848 at η =
    /// 3e-3, a truncation error of 2.3e-6), while the round-off grows as the
    /// step shrinks (8.904981 at η = 3e-4, 8.899516 at 1e-4, 8.732160 at
    /// 3e-5); measured 2026-11-21, x86_64. Its df agree within the band.
    const H_VALUE_ROUNDOFF: &str = "cross6 near-zero";

    /// The band on `H_p` entries in `exact_pass_agrees_with_value_route`: twice
    /// the largest measured gap, rounded up to one significant figure (measured
    /// 2026-11-22, plain wasm32-wasip1: 2.2e-5, `sim_nested_crossed_mix` ML
    /// H[0,0]; x86_64: 1.8e-5). The gap is the value route's round-off at
    /// `ETA`: on that entry 1.1e-5 on x86_64 and 2.2e-5 on plain wasm32,
    /// falling to 2.6e-6 and 1.6e-6 at η = 1e-3; from η = 3e-3 up only the
    /// truncation error remains, equal on both targets (1.26e-5 at 3e-3).
    const H_FD_REL: f64 = 5e-5;

    /// The exact pass against the value route at the fitted θ̂, each under
    /// REML and ML: on the per-group fixtures and the step sweep's fixtures
    /// (the dense kernel, crossed fits included), `sim_slope_extra` (the
    /// sparse kernel, a slope on the crossed grouping) and `q2s nested` (a
    /// primary slope with a nested child); and under REML on the
    /// sparse kernel fitted on the design as given for `q2s` (no crossed
    /// factor), `cross4` (a dense tail, 90 crossed levels) and a design with
    /// 150 crossed levels (the sparse kernel's fill-reducing tail), and
    /// `sim_nested_crossed_mix` with a slope on `c1` (a nested child on the
    /// sparse kernel). Every df
    /// within [`DF_FD_REL`] of the value route's, and every entry `(i, j)` of
    /// `H_p` within [`H_FD_REL`]`·√(H_ii·H_jj)` of it, except on
    /// [`H_VALUE_ROUNDOFF`]. The value route's own error is the limit:
    /// `step_sweep_is_flat_at_eta` measured its df error into [`DF_FD_REL`]. Fits with no
    /// free coordinate are not compared; the count of compared fits is pinned.
    /// Measured 2026-11-21 (x86_64), the largest over the 56 fits: df 1.7e-5
    /// and, outside [`H_VALUE_ROUNDOFF`], `H_p` 1.8e-5, both on
    /// `sim_nested_crossed_mix` with the slope (sparse kernel); plain
    /// wasm32-wasip1 (2026-11-22): df 1.75e-5 (`sim_dynrange_lmm` REML) and
    /// `H_p` 2.2e-5 (`sim_nested_crossed_mix` ML).
    #[test]
    fn exact_pass_agrees_with_value_route() {
        let (mut compared, mut worst_df, mut worst_h) =
            (0usize, (0.0f64, String::new()), (0.0f64, String::new()));
        let mut check = |name: &str, ws: &mut LmmWorkspace, map: &ColumnScaling, units: &Units| {
            let theta = ws.theta.clone();
            let exact = collect_eta(ws, &theta, map, units, ETA, true);
            assert!(exact.exact, "{name}: the exact pass covers the fit");
            if exact.free.is_empty() {
                eprintln!("EXACT-VALUE {name}: not compared (n_free 0)");
                return;
            }
            let value = collect_eta(ws, &theta, map, units, ETA, false);
            assert!(!value.exact, "{name}");
            let (st, sv) = (assemble(&exact), assemble(&value));
            assert_eq!((st.route, sv.route), (DerivRoute::Exact, DerivRoute::Value));
            let (de, dv) = (coef_df_in_range(&st), coef_df_in_range(&sv));
            for (i, (&e, &v)) in de.iter().zip(&dv).enumerate() {
                let r = rel(e, v);
                if r > worst_df.0 {
                    worst_df = (r, format!("{name} df[{i}]"));
                }
                assert!(r <= DF_FD_REL, "{name}: df[{i}] {e} exact, {v} value");
            }
            let (he, hv) = (hessian(&exact), hessian(&value));
            let n = hv.nrows();
            assert_eq!(n, he.nrows(), "{name}");
            let h_compared = !name.starts_with(H_VALUE_ROUNDOFF);
            if h_compared {
                for i in 0..n {
                    for j in 0..n {
                        let scale = (hv[(i, i)] * hv[(j, j)]).abs().sqrt();
                        let r = (he[(i, j)] - hv[(i, j)]).abs() / scale;
                        if r > worst_h.0 {
                            worst_h = (r, format!("{name} H[{i},{j}]"));
                        }
                        assert!(
                            r <= H_FD_REL,
                            "{name}: H[{i},{j}] {} exact, {} value",
                            he[(i, j)],
                            hv[(i, j)]
                        );
                    }
                }
            }
            eprintln!("EXACT-VALUE {name}: compared, n_free {n}, H_p compared {h_compared}");
            compared += 1;
        };
        let mut fixtures = per_group_fixtures();
        fixtures.extend(
            sweep_fixtures()
                .into_iter()
                .map(|(name, table, formula)| (name, table, formula, None)),
        );
        fixtures.push((
            "sim_slope_extra".to_string(),
            csv_table(
                include_str!("../../../validation/data/simulated/sim_slope_extra.csv"),
                &["g1", "g2"],
            ),
            "y ~ 1 + x + (1 + x | g1) + (1 + x | g2)".to_string(),
            None,
        ));
        // A primary slope with a nested child (the child↔slope cross terms).
        let (table, formula) = speed_shape(&[2, 1], true, false, 811_001);
        fixtures.push(("q2s nested".to_string(), table, formula, None));
        for (name, table, formula, weights) in &fixtures {
            for reml in [None, Some(false)] {
                let name = format!("{name} (reml {reml:?})");
                with_fit_w(formula, table, *weights, reml, |fit, ws, map, units| {
                    assert!(fit.converged(), "{name}");
                    assert_eq!(fit.reml, reml.is_none(), "{name}: criterion");
                    check(&name, ws, map, units);
                });
            }
        }
        for (name, table, formula) in sparse_fixtures() {
            let (_, mut ws) = both_kernels(&table, &formula);
            assert!(matches!(ws.kernel, LmmKernel::Sparse { .. }), "{name}");
            check(
                &format!("{name} (sparse kernel)"),
                &mut ws,
                ColumnScaling::IDENTITY,
                Units::IDENTITY,
            );
        }
        // A nested child on the sparse kernel (its child blocks); the slope on
        // the crossed grouping keeps it off the dense kernel.
        let table = csv_table(
            include_str!("../../../validation/data/simulated/sim_nested_crossed_mix.csv"),
            &["g1", "g2", "c1"],
        );
        let formula = "y ~ 1 + x + (1 | g1/g2) + (1 + x | c1)";
        let mut ws = accumulated(&table, formula, true);
        assert!(matches!(ws.kernel, LmmKernel::Sparse { .. }));
        let targets: Vec<u32> = (0..2).collect();
        assert!(fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY).converged);
        check(
            "sim_nested_crossed_mix slope (sparse kernel)",
            &mut ws,
            ColumnScaling::IDENTITY,
            Units::IDENTITY,
        );
        eprintln!(
            "EXACT-VALUE largest gaps: df {:.2e} ({}), H {:.2e} ({}); {compared} fits compared",
            worst_df.0, worst_df.1, worst_h.0, worst_h.1
        );
        assert_eq!(compared, 56, "fits compared");
    }

    /// The designs gate 4 and the comparison with the value route fit on the
    /// sparse kernel, `(name, table, formula)`: `q2s` (no crossed factor),
    /// `cross4` (90 crossed levels, the dense tail) and 150 crossed levels
    /// (the fill-reducing sparse tail, past `TAIL_SPARSE_MIN`).
    fn sparse_fixtures() -> Vec<(String, Table, String)> {
        let mut v = vec![];
        for (name, q) in [("q2s", &[2][..]), ("cross4", &[1; 4][..])] {
            let (table, formula) = speed_shape(q, false, false, 811_001);
            v.push((name.to_string(), table, formula));
        }
        let (table, formula) = crossed_design(200, &[75, 75], 1200, 7);
        v.push(("crossed 150".to_string(), table, formula));
        v
    }

    // ---------------------------------------------------------------------------
    // Gate 4: dense against sparse
    // ---------------------------------------------------------------------------

    /// One design on both kernels, `(dense, sparse)`, each fitted on the design as
    /// given (the identity column map, so the two share their θ̃). Extra
    /// groupings are crossed.
    fn both_kernels(table: &Table, formula: &str) -> (LmmWorkspace, LmmWorkspace) {
        let targets: Vec<u32> =
            (0..lower(formula, table, Family::Gaussian).unwrap().p as u32).collect();
        let fitted = |sparse: bool| {
            let mut ws = accumulated(table, formula, sparse);
            let fit = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);
            assert!(fit.converged, "sparse kernel {sparse}");
            ws
        };
        (fitted(false), fitted(true))
    }

    /// One design's rows accumulated on the dense or the sparse kernel, as
    /// [`both_kernels`] fits them, with no fit: θ̃ and β̂ as the workspace was
    /// built (β̂ zero, of length `p`).
    fn accumulated(table: &Table, formula: &str, sparse: bool) -> LmmWorkspace {
        let lo = lower(formula, table, Family::Gaussian).expect("the fixture lowers");
        let (n, p) = (lo.n, lo.p);
        let x = Mat::from_fn(n, p, |i, j| lo.x[i * p + j]);
        let levels = |ids: &[u32]| ids.iter().max().map_or(0, |&m| m + 1);
        let re = lo.model.re.expect("a mixed model");
        let slopes: Vec<usize> = re.slopes.iter().map(|&s| s as usize).collect();
        let extra_slopes: Vec<Vec<usize>> = re
            .extra_groupings
            .iter()
            .map(|g| g.slopes.iter().map(|&s| s as usize).collect())
            .collect();
        let model = SizedSpec {
            spec: ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: re.slopes,
                    extra_groupings: re
                        .extra_groupings
                        .into_iter()
                        .map(|g| Grouping {
                            terms: Vec::new(),
                            relation: GroupingRelation::Crossed,
                            slopes: g.slopes,
                        })
                        .collect(),
                }),
            },
            n_primary: levels(&lo.ids.primary) as usize,
            extra_counts: lo
                .ids
                .extra
                .iter()
                .map(|ids| levels(ids) as usize)
                .collect(),
            tail_fill: 0.0,
        };
        let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &model, &slopes, &extra_slopes, sparse);
        {
            let (ids, extra) = (&lo.ids.primary, &lo.ids.extra);
            match &mut ws.kernel {
                LmmKernel::Dense { suff, .. } => {
                    suff.add_rows_multi(
                        x.as_ref(),
                        x.as_ref().subcols(0, 0),
                        &lo.y,
                        ids,
                        extra,
                        None,
                    );
                }
                LmmKernel::Sparse { g, ws: slot } => {
                    *slot = Some(Box::new(
                        SparseLmmWorkspace::new(
                            g,
                            x.as_ref(),
                            x.as_ref().subcols(0, 0),
                            ids,
                            extra,
                            &lo.y,
                            n,
                            p,
                            None,
                            crate::lmm::Criterion::Reml,
                        )
                        .unwrap(),
                    ));
                }
            }
        }
        ws.recovery.betas = vec![0.0; p];
        ws.recovery.beta_tilde = vec![0.0; p];
        ws
    }

    /// Gate 4: on the designs of [`sparse_fixtures`] (`q2s`, `cross4` and 150
    /// crossed levels, the last on the sparse kernel's fill-reducing tail) the
    /// two kernels run the pass at one shared θ, the dense fit's: the two
    /// exact passes give every df within [`DENSE_SPARSE_REL`], and the two
    /// value routes within the value band [`DF_FD_REL`]. Their deviances agree
    /// to round-off there first, so the two hold the same design.
    #[test]
    fn dense_and_sparse_kernels_agree() {
        let mut worst = 0.0f64;
        for (name, table, formula) in sparse_fixtures() {
            let (mut dense, mut sparse) = both_kernels(&table, &formula);
            let theta = dense.theta.clone();
            let (fd, fs) = (
                dense.kernel.deviance(&theta),
                sparse.kernel.deviance(&theta),
            );
            assert!(
                rel(fs, fd) <= 1e-12,
                "{name}: deviance {fd} dense, {fs} sparse"
            );
            for (route, band) in [
                (DerivRoute::Exact, DENSE_SPARSE_REL),
                (DerivRoute::Value, DF_FD_REL),
            ] {
                let on = |ws: &mut LmmWorkspace| {
                    let st = match route {
                        DerivRoute::Exact => {
                            pass_at(ws, &theta, ColumnScaling::IDENTITY, Units::IDENTITY)
                        }
                        DerivRoute::Value => {
                            pass_value_at(ws, &theta, ColumnScaling::IDENTITY, Units::IDENTITY)
                        }
                    };
                    assert_eq!(st.route, route, "{name}");
                    coef_df_in_range(&st)
                };
                let (want, got) = (on(&mut dense), on(&mut sparse));
                for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                    let r = rel(g, w);
                    eprintln!(
                        "DENSE-SPARSE {name} {route:?}: df[{i}] sparse {g} dense {w} rel {r:.2e}"
                    );
                    if route == DerivRoute::Exact {
                        worst = worst.max(r);
                    }
                    assert!(
                        r <= band,
                        "{name}: df[{i}] {g} sparse, {w} dense ({route:?})"
                    );
                }
            }
        }
        eprintln!("DENSE-SPARSE largest exact gap {worst:.2e}");
    }

    /// The tail width: a design with `EXACT_TAIL_MAX` crossed levels takes the
    /// exact pass and one with a level more the value route, on both kernels.
    /// Built relative to the constant, so moving it moves the fixture. The
    /// workspaces are not fitted: the route depends on the design alone.
    #[test]
    fn exact_pass_stops_at_the_tail_width() {
        use crate::lmm::satterthwaite::exact::EXACT_TAIL_MAX;
        for (e, want) in [
            (EXACT_TAIL_MAX, DerivRoute::Exact),
            (EXACT_TAIL_MAX + 1, DerivRoute::Value),
        ] {
            let (table, formula) = crossed_design(e + 100, &[e], 2 * e + 200, 811_006);
            for sparse in [false, true] {
                let mut ws = accumulated(&table, &formula, sparse);
                assert_eq!(ws.kernel.groupings().k_crossed(), e);
                let theta = vec![1.0; ws.theta.len()];
                let st = pass_at(&mut ws, &theta, ColumnScaling::IDENTITY, Units::IDENTITY);
                assert!(!st.is_failed(), "e {e}, sparse {sparse}");
                assert_eq!(st.route, want, "e {e}, sparse {sparse}");
            }
        }
    }

    // ---------------------------------------------------------------------------
    // Boundary
    // ---------------------------------------------------------------------------

    /// A real fit of a `q = 2` block (a random intercept and slope) beside a
    /// crossed random intercept, every coordinate free: each df is the profiled
    /// formula built by hand from the same stencil over the four coordinates,
    /// `H` with its mixed points `θ̂ ± (h_i e_i + h_j e_j)` solved by a Cholesky
    /// factor, `g` read off the mapped factor. A coordinate's step is `ETA`
    /// times the largest `|θ̃|` of its block, so the two blocks' steps differ: a
    /// mixed point that steps one coordinate's `h` on both axes fails.
    #[test]
    fn q2_fit_matches_hand_built_stencil() {
        let (table, formula) = speed_shape(&[2, 1], false, false, 811_001);
        with_formula_fit(&formula, &table, |fit, ws, map, units| {
            assert!(fit.converged());
            let th = ws.theta.clone();
            let n = th.len();
            let st = pass_value_at(ws, &th, map, units);
            assert_eq!(st.jac_theta.len(), n, "every coordinate free: θ̂ {th:?}");
            assert_eq!((st.dropped, st.negative), (0, 0));
            let df = coef_df_in_range(&st);

            let p = df.len();
            let m_b = |b: &[f64]| ETA * b.iter().fold(0.0, |m: f64, t| m.max(t.abs()));
            let h = [vec![m_b(&th[..3]); 3], vec![m_b(&th[3..])]].concat();
            assert!(h[3] != h[0], "unequal steps: θ̂ {th:?}");
            let at = |steps: &[(usize, f64)]| {
                let mut t = th.clone();
                for &(k, s) in steps {
                    t[k] += s;
                }
                t
            };
            let saved = shift_response(&mut ws.kernel, &ws.recovery.beta_tilde);
            let f0 = ws.kernel.deviance(&th);
            let (mut f_axis, mut c_axis) = (vec![], vec![]);
            for (k, &hk) in h.iter().enumerate().take(n) {
                let (fp, cp) = deviance_and_var(&mut ws.kernel, &at(&[(k, hk)]), map, units, p);
                let (fm, cm) = deviance_and_var(&mut ws.kernel, &at(&[(k, -hk)]), map, units, p);
                f_axis.push([fp, fm]);
                c_axis.push([cp, cm]);
            }
            let mut hess = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                hess[(i, i)] = (f_axis[i][0] - 2.0 * f0 + f_axis[i][1]) / (h[i] * h[i]);
                for j in i + 1..n {
                    let fpp = ws.kernel.deviance(&at(&[(i, h[i]), (j, h[j])]));
                    let fmm = ws.kernel.deviance(&at(&[(i, -h[i]), (j, -h[j])]));
                    let axis = f_axis[i][0] + f_axis[i][1] + f_axis[j][0] + f_axis[j][1];
                    let b = (fpp + fmm - axis + 2.0 * f0) / (2.0 * h[i] * h[j]);
                    hess[(i, j)] = b;
                    hess[(j, i)] = b;
                }
            }
            restore_response(&mut ws.kernel, &saved);
            let llt = SeqLlt::new(hess.as_ref()).expect("an interior fit's H is PD");
            for t in 0..p {
                let v = fit.vcov()[t * p + t];
                let g: Vec<f64> = (0..n)
                    .map(|k| (c_axis[k][0][t] - c_axis[k][1][t]) / (2.0 * h[k]))
                    .collect();
                let mut x = Mat::from_fn(n, 1, |k, _| g[k]);
                llt.solve_in_place(x.as_mut());
                let ghg: f64 = (0..n).map(|k| g[k] * x[(k, 0)]).sum();
                let want = v * v / (ghg + v * v / st.nu);
                assert!(
                    rel(df[t], want) <= ROUTE_REL,
                    "coefficient {t}: df {} against {want}",
                    df[t]
                );
            }
        });
    }

    /// The response statistics' bits (the `y` row of the Gram, the sparse
    /// kernel's `y` column too, the `y` entry of each random-effect column's
    /// `Z'[X̃ y]`, and the QR kernel's `y` column while it is on) and the
    /// deviance's at θ̂.
    fn response_bits(ws: &mut LmmWorkspace) -> Vec<u64> {
        let mut v: Vec<f64> = match &mut ws.kernel {
            LmmKernel::Dense { suff, fit } => {
                let p = suff.m - 1;
                (0..=p)
                    .map(|j| suff.c[(p, j)])
                    .chain((0..suff.s.ncols()).map(|a| suff.s[(p, a)]))
                    .chain(
                        fit.qr
                            .as_deref_mut()
                            .filter(|q| q.on)
                            .map_or(vec![], |q| q.y_column()),
                    )
                    .collect()
            }
            LmmKernel::Sparse { ws: sp, .. } => {
                let sp = LmmKernel::sparse_ws(sp);
                let p = sp.p;
                (0..=p)
                    .map(|j| sp.cxy[(p, j)])
                    .chain((0..=p).map(|i| sp.cxy[(i, p)]))
                    .chain(sp.ztxy.chunks(sp.m).map(|r| r[p]))
                    .collect()
            }
        };
        let theta = ws.theta.clone();
        v.push(ws.kernel.deviance(&theta));
        v.into_iter().map(f64::to_bits).collect()
    }

    /// The pass puts the response back bit for bit on each kernel route it
    /// meets, on the exact pass and on the value route (which shifts the
    /// response): the balanced collapse (`int1`, five rows in each of 60
    /// groups), the QR kernel (`int1` again), the per-group regressions
    /// (`q2s`) and the sparse kernel (`cross4`). The
    /// deviance at θ̂ reads the re-armed precomputes, so a stale one shows
    /// there.
    #[test]
    fn pass_restores_the_response_on_every_route() {
        let both = |ws: &mut LmmWorkspace, map: &ColumnScaling, units: &Units, name: &str| {
            let theta = ws.theta.clone();
            let before = response_bits(ws);
            let st = pass(ws, map, units);
            assert!(!st.is_failed(), "{name}");
            assert_eq!(st.route, DerivRoute::Exact, "{name}");
            assert_eq!(response_bits(ws), before, "{name}, exact pass");
            let st = pass_value_at(ws, &theta, map, units);
            assert!(!st.is_failed(), "{name}");
            assert_eq!(st.route, DerivRoute::Value, "{name}");
            assert_eq!(response_bits(ws), before, "{name}, value route");
        };
        let (table, formula) = speed_shape(&[1], false, false, 811_001);
        with_formula_fit(&formula, &table, |fit, ws, map, units| {
            assert!(fit.converged());
            assert!(
                matches!(&ws.kernel, LmmKernel::Dense { fit, .. } if fit.collapse_n_active > 0),
                "int1: the balanced collapse is armed"
            );
            both(ws, map, units, "int1");
        });
        with_formula_fit_qr(&formula, &table, |fit, ws, map, units| {
            assert!(fit.converged());
            assert!(ws.kernel.qr_on(), "int1: the QR kernel is on");
            both(ws, map, units, "int1, QR kernel");
        });

        let (table, formula) = speed_shape(&[2], false, false, 811_001);
        with_formula_fit(&formula, &table, |fit, ws, map, units| {
            assert!(fit.converged());
            assert!(per_group_gradient_armed(&ws.kernel), "q2s: per-group route");
            both(ws, map, units, "q2s");
        });

        let (table, formula) = speed_shape(&[1; 4], false, false, 811_001);
        let (_, mut sparse) = both_kernels(&table, &formula);
        assert!(matches!(sparse.kernel, LmmKernel::Sparse { .. }));
        both(
            &mut sparse,
            ColumnScaling::IDENTITY,
            Units::IDENTITY,
            "cross4, sparse kernel",
        );
    }

    /// The exact pass on the QR kernel gives the Gram kernel's df at the same
    /// θ̂ within [`QR_GRAM_REL`] (`q2s`, random slopes on the primary, and
    /// `nest2`, a nested intercept under the primary one).
    #[test]
    fn qr_pass_df_match_the_gram_pass() {
        let mut worst = 0.0f64;
        for (name, q, nested) in [("q2s", &[2][..], false), ("nest2", &[1, 1][..], true)] {
            let (table, formula) = speed_shape(q, nested, false, 811_001);
            let lo = lower(&formula, &table, Family::Gaussian).expect("the fixture lowers");
            let data = lo.data();
            let (fit, mut fws) =
                fit_keep_workspace(&data, &lo.model, None, &FitOptions::default()).unwrap();
            assert!(fit.converged());
            let (ws, map, units) = fws.lmm_parts().unwrap();
            let theta = ws.theta.clone();
            let st = pass_at(ws, &theta, map, units);
            assert_eq!(st.route, DerivRoute::Exact, "{name}");
            let gram = st.coef_df();
            assert!(fws.lmm_start_qr(&data), "{name}: a covered shape");
            let (ws, map, units) = fws.lmm_parts().unwrap();
            assert!(ws.kernel.qr_on());
            let st = pass_at(ws, &theta, map, units);
            assert_eq!(st.route, DerivRoute::Exact, "{name}");
            let qr = st.coef_df();
            assert_eq!(qr.len(), lo.p);
            assert_eq!(gram.len(), qr.len());
            for (t, (&q, &g)) in qr.iter().zip(&gram).enumerate() {
                eprintln!(
                    "QR-GRAM {name}: df[{t}] QR {q} Gram {g} rel {:.2e}",
                    rel(q, g)
                );
                worst = worst.max(rel(q, g));
                assert!(
                    rel(q, g) <= QR_GRAM_REL,
                    "{name}: coefficient {t}: QR df {q} against Gram {g}"
                );
            }
        }
        eprintln!("QR-GRAM largest gap {worst:.2e}");
    }

    /// The canonical correlation boundary (`sim_corr_boundary`, `y ~ x + (x | g)`):
    /// the fit pins the block's trailing diagonal and canonicalizes the block, so
    /// that column is exactly zero. The pass drops its coordinate, and each df is
    /// the profiled formula built by hand over the two free coordinates: `H`
    /// written out from the same stencil and inverted as a 2 × 2, `g` read off the
    /// mapped factor. Both free coordinates step `ETA` times the larger `|θ̃|`.
    #[test]
    fn correlation_boundary_matches_dropped_coordinate_formula() {
        let table = csv_table(
            include_str!("../../../validation/data/simulated/sim_corr_boundary.csv"),
            &["g"],
        );
        with_formula_fit("y ~ x + (x | g)", &table, |fit, ws, map, units| {
            assert!(fit.converged());
            let th = ws.theta.clone();
            assert!(th[0] != 0.0 && th[1] != 0.0 && th[2] == 0.0, "θ̂ {th:?}");
            let st = crate::lmm::satterthwaite::pass_value_at(ws, &th, map, units);
            assert_eq!(st.jac_theta.len(), 2);
            assert_eq!((st.dropped, st.negative), (0, 0));
            let df = coef_df_in_range(&st);

            let p = df.len();
            let h = [ETA * th[0].abs().max(th[1].abs()); 2];
            let at = |d0: f64, d1: f64| [th[0] + d0, th[1] + d1, 0.0];
            let saved = shift_response(&mut ws.kernel, &ws.recovery.beta_tilde);
            let f0 = ws.kernel.deviance(&th);
            let (f0p, c0p) = deviance_and_var(&mut ws.kernel, &at(h[0], 0.0), map, units, p);
            let (f0m, c0m) = deviance_and_var(&mut ws.kernel, &at(-h[0], 0.0), map, units, p);
            let (f1p, c1p) = deviance_and_var(&mut ws.kernel, &at(0.0, h[1]), map, units, p);
            let (f1m, c1m) = deviance_and_var(&mut ws.kernel, &at(0.0, -h[1]), map, units, p);
            let fpp = ws.kernel.deviance(&at(h[0], h[1]));
            let fmm = ws.kernel.deviance(&at(-h[0], -h[1]));
            restore_response(&mut ws.kernel, &saved);
            let a = (f0p - 2.0 * f0 + f0m) / (h[0] * h[0]);
            let c = (f1p - 2.0 * f0 + f1m) / (h[1] * h[1]);
            // Summed in the pass's order: at these steps another order moves df by 2e-9.
            let axis = f0p + f0m + f1p + f1m;
            let b = (fpp + fmm - axis + 2.0 * f0) / (2.0 * h[0] * h[1]);
            for t in 0..p {
                let v = fit.vcov()[t * p + t];
                let g0 = (c0p[t] - c0m[t]) / (2.0 * h[0]);
                let g1 = (c1p[t] - c1m[t]) / (2.0 * h[1]);
                let ghg = (c * g0 * g0 - 2.0 * b * g0 * g1 + a * g1 * g1) / (a * c - b * b);
                let want = v * v / (ghg + v * v / st.nu);
                assert!(
                    rel(df[t], want) <= ROUTE_REL,
                    "coefficient {t}: df {} against {want}",
                    df[t]
                );
            }
        });
    }

    /// `y ~ 1 + x1..x8 + (1 + x1..x8 | g)` as a table, `x` the eight covariates.
    fn wide_fixture(x: &[Vec<f64>], y: &[f64], g: &[String]) -> (Table, String) {
        let mut columns = vec![("y".to_string(), Column::Numeric(y.to_vec()))];
        for (j, xj) in x.iter().enumerate() {
            columns.push((format!("x{}", j + 1), Column::Numeric(xj.clone())));
        }
        columns.push(("g".to_string(), Column::factor_from_labels(g)));
        let slopes = (1..=8)
            .map(|j| format!("x{j}"))
            .collect::<Vec<_>>()
            .join(" + ");
        let formula = format!("y ~ 1 + {slopes} + (1 + {slopes} | g)");
        (
            Table {
                columns,
                n: y.len(),
            },
            formula,
        )
    }

    /// A wide primary block (`q = 9 > MAX_PRIMARY_Q`, so the sparse route, which
    /// `canonicalize_pinned_blocks` skips) with a pinned diagonal above a non-zero
    /// entry: that column stays free, and `H_p` goes through the pseudo-inverse,
    /// which sets aside at least one direction (dropped or negative, either of
    /// which raises the note).
    ///
    /// The design puts the REML optimum there. A first fit (40 groups of 25, slope
    /// 4's true variance 0) stops on the boundary, its `Σ̂` singular with null
    /// vector `v`, mostly along `x4`. The refit puts `x4` first among the slopes
    /// and shears the others, `x_m − (v_m/v_4)·x4`: the same model with a null
    /// vector supported on the intercept and `x4`, now column 1, so the leading
    /// 2 × 2 of its `Σ̂` is singular: `Λ̂₁₁ = 0` at the optimum, the entries below
    /// it free (rotating columns 1 and 2 leaves `ΛΛᵀ` unchanged). The shear stays
    /// small (`|v_m/v_4| ≤ 0.1` here). Shearing onto `x1` instead takes factors
    /// up to `|v_m/v_1| ≈ 45`, and that refit used up its 45,000 evaluations on
    /// 5 of 7 draws on aarch64.
    ///
    /// `Σ̂` counts as singular at `λ_min ≤ 1e-10·λ_max`. Whether the first fit's
    /// trailing diagonal pins is platform rounding: at seed 3·7919 it pinned on
    /// x86_64 and stopped at 1.45e-4, above `PIN_THETA`, on aarch64 (`λ_min/λ_max`
    /// 4.5e-12, as on wasm32). `v` is the exact null vector of `Σ̂ − λ_min·vvᵀ`,
    /// within `λ_min` of `Σ̂`, so the band turns `v` by at most `λ_min/λ₂ ≈ 7e-10`,
    /// far below the first fit's own stopping error. Were `λ_min` a real variance,
    /// the refit's optimum would have `Λ̂₁₁ ≲ √λ_min/σ̂ ≈ 2e-5` at the band's edge
    /// (`λ_max ≈ 0.88`, `σ̂² ≈ 0.24`, unit-variance columns): a fifth of
    /// `PIN_THETA`, so still pinned. `PIN_THETA` is reached at about 3e-9.
    ///
    /// Probed 2026-10-11 on aarch64. At this draw both fits pin: `λ_min/λ_max`
    /// 1.2e-16, then `Λ̃₁₁ = 0` after 1,337 of 45,000 evaluations. The pass counts
    /// `negative` 3: `H_p`'s lowest eigenvalues are −9.6e-7, −4.5e-7 and −1.3e-7
    /// of `λ_max`, against the pseudo-inverse's 1e-8. The point is not
    /// stationary, so they need not be ≥ 0. Six perturbations of `y` at 1e-14
    /// relative keep both pins, the refit at 1.3k to 1.5k evaluations and
    /// `negative` at 3 or 4. Over seeds `k·7919`, `k = 1..7`, the refit converges
    /// on every draw, with `dropped` 0 or 1 and `negative` 1 to 5. A search that
    /// stops above `PIN_THETA` (up to 3.3e-3 on those draws) is pinned by the
    /// test, as `fit_lmm` would pin it.
    #[test]
    fn wide_pinned_block_goes_through_the_pseudo_inverse() {
        const GROUPS: usize = 40;
        const PER: usize = 25;
        const ZERO: usize = 4;
        let n = GROUPS * PER;
        let mut s = 7919u64;
        let x: Vec<Vec<f64>> = (0..8)
            .map(|_| (0..n).map(|_| unit(&mut s)).collect())
            .collect();
        let mut y: Vec<f64> = (0..n).map(|i| 0.5 + 0.3 * x[0][i]).collect();
        for (grp, rows) in y.chunks_mut(PER).enumerate() {
            let b: Vec<f64> = (0..9)
                .map(|d| match d {
                    ZERO => 0.0,
                    0 => unit(&mut s),
                    _ => 0.5 * unit(&mut s),
                })
                .collect();
            for (r, yi) in rows.iter_mut().enumerate() {
                let i = grp * PER + r;
                *yi += b[0] + (1..9).map(|d| b[d] * x[d - 1][i]).sum::<f64>();
            }
        }
        for yi in &mut y {
            *yi += 0.5 * unit(&mut s);
        }
        let g: Vec<String> = (0..n).map(|i| (i / PER).to_string()).collect();

        let (table, formula) = wide_fixture(&x, &y, &g);
        let d = with_formula_fit(&formula, &table, |fit, _, _, _| {
            assert!(fit.converged());
            fit.varcorr[0].clone()
        });
        // `varcorr` is the column-major lower vech of Σ̂.
        let mut sigma = Mat::<f64>::zeros(9, 9);
        let mut k = 0;
        for c in 0..9 {
            for r in c..9 {
                (sigma[(r, c)], sigma[(c, r)]) = (d[k], d[k]);
                k += 1;
            }
        }
        let eig = crate::inference::evd::sym_eigen(sigma.as_ref(), true).expect("a symmetric Σ̂");
        assert!(
            eig.values[0].abs() <= 1e-10 * eig.values[8],
            "Σ̂ {:?}",
            eig.values
        );
        let vectors = eig.vectors.expect("vectors");
        let v: Vec<f64> = (0..9).map(|r| vectors[(r, 0)]).collect();
        let z = &x[ZERO - 1];
        let x2: Vec<Vec<f64>> = std::iter::once(z.clone())
            .chain((1..9).filter(|&m| m != ZERO).map(|m| {
                (0..n)
                    .map(|i| x[m - 1][i] - v[m] / v[ZERO] * z[i])
                    .collect()
            }))
            .collect();

        let (table, formula) = wide_fixture(&x2, &y, &g);
        with_formula_fit(&formula, &table, |fit, ws, map, units| {
            assert!(fit.converged());
            assert!(matches!(ws.kernel, LmmKernel::Sparse { .. }));
            let groupings = ws.kernel.groupings();
            let (diag, len) = (groupings.diagonal_theta()[1], groupings.diagonal_run_len[1]);
            let mut theta = ws.theta.clone();
            assert!(theta[diag].abs() <= 1e-2, "Λ̃₁₁ {}", theta[diag]);
            assert!(theta[diag + 1..diag + len].iter().any(|&t| t != 0.0));
            theta[diag] = 0.0;
            assert_eq!(free_coordinates(groupings, &theta).len(), 45);
            let st = pass_at(ws, &theta, map, units);
            assert!(!st.is_failed());
            assert_eq!(st.jac_theta.len(), 45);
            assert!(
                st.dropped + st.negative >= 1,
                "({}, {})",
                st.dropped,
                st.negative
            );
            coef_df_in_range(&st);
        });
    }
}

// ---------------------------------------------------------------------------
// Gate 1 at lme4's θ̂ (oracle tier)
// ---------------------------------------------------------------------------

/// Gate 1 at lme4's θ̂: the pass run at lme4's θ̂, mapped to the kernel's θ̃,
/// against lmerTest's frozen df (`validation/goldens/inference/lmertest/`).
/// The per-coefficient gate is here; the per-term rows
/// (`formula::hypotheses_oracle_tests`) share the mapping and the band.
#[cfg(feature = "oracle-tests")]
pub(crate) mod gate1 {
    use super::*;
    use crate::fit::spec_sized_from_ids;
    use crate::formula::hypotheses_tests::table_of;
    use crate::formula::{lower, Column, Lowered};
    use crate::inference::evd::sym_eigen;
    use crate::inference::test_refs::{self, Coverage, DivEntry, Entry, Kind};
    use serde_json::Value;
    use std::collections::BTreeSet;

    /// `df_lmertest_floor_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`):
    /// the floor of lmerTest's own finite-difference spread in [`band`].
    /// Change together.
    pub(crate) const DF_LMERTEST_FLOOR_REL: f64 = 1e-6;

    /// Gate 1's band at lme4's θ̂ on `route`, relative to lmerTest's value.
    /// The error sources add: lmerTest's own finite-difference `spread` (the
    /// frozen second-setting values) floored at [`DF_LMERTEST_FLOOR_REL`], the
    /// value route's own band [`DF_FD_REL`] (the exact pass has no step and
    /// no such term), and `nonstat`, how far the df moves over one Newton
    /// step from lme4's θ̂ toward the crate's stationary point
    /// ([`newton_point`]): the identity with lmerTest's coordinates holds only
    /// at a stationary θ̂, and lme4's stops at its optimizer's tolerance.
    pub(crate) fn band(route: DerivRoute, spread: f64, nonstat: f64) -> f64 {
        let route_rel = match route {
            DerivRoute::Exact => 0.0,
            DerivRoute::Value => DF_FD_REL,
        };
        spread.max(DF_LMERTEST_FLOOR_REL) + route_rel + nonstat
    }

    /// The references whose fit the exact pass covers (a crossed tail within
    /// its width), without the `_ml` suffix.
    const GATE1_EXACT: &[&str] = &[
        "lmertest_Dyestuff",
        "lmertest_Dyestuff2",
        "lmertest_Machines",
        "lmertest_Oats",
        "lmertest_Pastes",
        "lmertest_Penicillin",
        "lmertest_all_ones",
        "lmertest_cake",
        "lmertest_lmm_crossed",
        "lmertest_lmm_intercept",
        "lmertest_lmm_slope",
        "lmertest_path_huge_int",
        "lmertest_sim_collinear_lmm",
        "lmertest_sim_corr_boundary",
        "lmertest_sim_dynrange_lmm",
        "lmertest_sim_factorial",
        "lmertest_sim_max_q_slope",
        "lmertest_sim_near_singular",
        "lmertest_sim_nested_crossed_mix",
        "lmertest_sim_slope",
        "lmertest_sim_slope_extra",
        "lmertest_sim_three_level",
        "lmertest_sim_unbalanced_nested",
        "lmertest_sim_wide_crossed",
        "lmertest_sim_wide_slopes",
        "lmertest_sleepstudy",
    ];

    /// The derivative routes gate 1 runs on the fit `ws` holds: the value
    /// route always, and the exact pass where it covers the fit's kernel.
    pub(crate) fn routes(ws: &LmmWorkspace) -> Vec<DerivRoute> {
        let mut routes = vec![DerivRoute::Value];
        if crate::lmm::satterthwaite::exact::covers(&ws.kernel) {
            routes.push(DerivRoute::Exact);
        }
        routes
    }

    /// The pass on `route` at the kernel's θ̃, the route asserted: every gate-1
    /// pass here and in the per-term rows goes through this one helper.
    pub(crate) fn route_pass_at(
        ws: &mut LmmWorkspace,
        theta: &[f64],
        map: &ColumnScaling,
        units: &Units,
        route: DerivRoute,
    ) -> SatterthwaiteState {
        let st = match route {
            DerivRoute::Value => crate::lmm::satterthwaite::pass_value_at(ws, theta, map, units),
            DerivRoute::Exact => pass_at(ws, theta, map, units),
        };
        assert_eq!(st.route, route, "the pass took another route");
        st
    }

    /// One registry driver of the oracle tier. A reference name may be
    /// compared by several drivers, each under its own quantity labels (gate 1
    /// at lme4's θ̂ under `df_at_lme4`, `dendf_at_lme4` and `f_at_lme4`, end to
    /// end under `df`, `f`, `p_value`, `lrt` and `confint`), so a driver sees
    /// only the `oracle-tier` entries under its labels, both when a row is over
    /// its band and in the stale check: an entry sized for one part never
    /// excuses the other. A label may belong to two drivers on disjoint
    /// references (`df` and `f` are the per-term rows' end-to-end labels and
    /// the integration gate's), so the one-driver rule is checked per compared
    /// reference, in [`Driver::finish`].
    pub(crate) struct Driver {
        quantities: &'static [&'static str],
        registry: Vec<DivEntry>,
        compared: BTreeSet<String>,
        fired: BTreeSet<String>,
    }

    impl Driver {
        /// The driver of `quantities`: the `oracle-tier` entries that list one
        /// of them.
        pub(crate) fn new(quantities: &'static [&'static str]) -> Driver {
            let mine = |q: &String| quantities.contains(&q.as_str());
            let registry: Vec<DivEntry> = test_refs::registry_entries()
                .into_iter()
                .filter(|e| e.quantities.iter().any(mine))
                .collect();
            Driver {
                quantities,
                registry,
                compared: BTreeSet::new(),
                fired: BTreeSet::new(),
            }
        }

        /// `got` against lmerTest's `want` within `band` relative; past it, an
        /// entry on `name` listing `quantity` decides. A NaN never passes.
        pub(crate) fn check(
            &mut self,
            name: &str,
            quantity: &str,
            what: &str,
            (got, want): (f64, f64),
            band: f64,
        ) {
            assert!(
                self.quantities.contains(&quantity),
                "{quantity} is not one of this driver's quantities {:?}",
                self.quantities
            );
            let rel = (got - want).abs() / want.abs();
            if rel <= band {
                return;
            }
            match test_refs::covers(&self.registry, name, quantity, rel) {
                Coverage::Within { id } => {
                    eprintln!(
                        "documented divergence {id}: {name} {what}: {got} against lmerTest {want} (rel {rel:.3e})"
                    );
                    self.fired.insert(id);
                }
                Coverage::NotCovered => panic!(
                    "{name} {what}: {got} against lmerTest {want} (rel {rel:.3e} > band {band:.3e})"
                ),
                Coverage::Exceeds { id, max_rel } => panic!(
                    "{name} {what}: {got} against lmerTest {want} (rel {rel:.3e}); documented divergence `{id}` covers only {max_rel:.1e}"
                ),
            }
        }

        /// Records that `name` was compared.
        pub(crate) fn compared(&mut self, name: &str) {
            self.compared.insert(name.to_string());
        }

        /// The registry's teeth on the references this driver compared: an
        /// entry there under one of its quantities lists only its quantities
        /// (so no entry on a reference belongs to two drivers), and fired.
        pub(crate) fn finish(&self) {
            let mine = |q: &String| self.quantities.contains(&q.as_str());
            for e in self
                .registry
                .iter()
                .filter(|e| self.compared.contains(&e.cell))
            {
                assert!(
                    e.quantities.iter().all(mine),
                    "registry entry {} mixes this driver's quantities {:?} with others: {:?}",
                    e.id,
                    self.quantities,
                    e.quantities
                );
            }
            let stale = test_refs::stale_entries(&self.registry, &self.compared, &self.fired);
            assert!(
                stale.is_empty(),
                "inference divergence registry is out of date: these entries on a compared \
                 reference no longer fire and must be deleted: {stale:?}"
            );
        }
    }

    /// The largest relative distance of the frozen second-setting values from
    /// lmerTest's own.
    pub(crate) fn spread(want: f64, second: &[f64]) -> f64 {
        second
            .iter()
            .map(|s| ((s - want) / want).abs())
            .fold(0.0, f64::max)
    }

    pub(crate) fn numbers(v: &Value, what: &str) -> Vec<f64> {
        v.as_array()
            .unwrap_or_else(|| panic!("{what}: not an array: {v}"))
            .iter()
            .map(|x| {
                x.as_f64()
                    .unwrap_or_else(|| panic!("{what}: not a number: {x}"))
            })
            .collect()
    }

    pub(crate) fn strings(v: &Value, what: &str) -> Vec<String> {
        v.as_array()
            .unwrap_or_else(|| panic!("{what}: not an array: {v}"))
            .iter()
            .map(|x| {
                x.as_str()
                    .unwrap_or_else(|| panic!("{what}: not a string: {x}"))
                    .to_string()
            })
            .collect()
    }

    /// A matrix written by jsonlite: an array of rows.
    pub(crate) fn matrix(v: &Value, what: &str) -> Mat<f64> {
        let rows: Vec<Vec<f64>> = v
            .as_array()
            .unwrap_or_else(|| panic!("{what}: not an array of rows: {v}"))
            .iter()
            .map(|r| numbers(r, what))
            .collect();
        let ncols = rows.first().map_or(0, Vec::len);
        assert!(rows.iter().all(|r| r.len() == ncols), "{what}: ragged rows");
        Mat::from_fn(rows.len(), ncols, |i, j| rows[i][j])
    }

    /// The entries of kind `lmertest` whose `proposal_dependent` is
    /// `per_term` (`false` the per-coefficient layout, `true` the per-term
    /// one) and whose `reml` is `reml` (`false` the `_ml` entries, fitted with
    /// `REML = FALSE`). Never empty.
    pub(crate) fn lmertest_entries(per_term: bool, reml: bool) -> Vec<Entry> {
        let out: Vec<Entry> = test_refs::entries(Kind::Lmertest)
            .into_iter()
            .filter(|e| {
                let flag = e.raw["proposal_dependent"].as_bool();
                let r = e.raw["reml"].as_bool();
                assert!(
                    flag.is_some() && r.is_some(),
                    "{}: proposal_dependent or reml is not a bool",
                    e.name
                );
                flag == Some(per_term) && r == Some(reml)
            })
            .collect();
        assert!(
            !out.is_empty(),
            "no lmertest entry (per_term {per_term}, reml {reml})"
        );
        assert!(
            out.iter().all(|e| !e.is_pending()),
            "an entry is not frozen"
        );
        out
    }

    /// The entry's formula lowered on its data, with its prior weights (the
    /// entry's `weights` column), and the fit options with its criterion
    /// (`FitOptions::reml` from the entry's `reml`, `None` on an entry without
    /// one).
    pub(crate) fn lowered(e: &Entry) -> (Lowered, FitOptions) {
        let table = table_of(e);
        let formula = e
            .r_formula
            .as_ref()
            .expect("an lmertest entry has a formula");
        let mut lo = lower(formula, &table, Family::Gaussian)
            .unwrap_or_else(|err| panic!("{}: {err}", e.name));
        if let Some(w) = e.raw["weights"].as_str() {
            let column = table.columns.iter().find(|(n, _)| n == w);
            let Some((_, Column::Numeric(v))) = column else {
                panic!("{}: no numeric weights column {w}", e.name);
            };
            lo.weights = Some(v.clone());
        }
        let fit_opts = FitOptions {
            reml: e.raw["reml"].as_bool(),
            ..FitOptions::default()
        };
        (lo, fit_opts)
    }

    /// Fits the design of `lo` on lme4's retained columns (the design minus
    /// the reference's `cols_dropped`, lmerTest's column-dropped `X`) to
    /// accumulate the workspace, maps the reference's θ̂ to the kernel's θ̃ and
    /// hands `f` the retained caller columns, the workspace, the column map
    /// and θ̃. `fit_json` carries the fit fields: the per-coefficient
    /// reference itself, or a per-term reference's twin.
    pub(crate) fn at_lme4_theta<R>(
        name: &str,
        lo: &Lowered,
        fit_opts: &FitOptions,
        fit_json: &Value,
        f: impl FnOnce(&[usize], &mut LmmWorkspace, &ColumnScaling, &Units, &[f64]) -> R,
    ) -> R {
        let dropped = strings(&fit_json["cols_dropped"], name);
        for d in &dropped {
            assert!(lo.col_names.contains(d), "{name}: no design column {d}");
        }
        let kept: Vec<usize> = (0..lo.p)
            .filter(|&j| !dropped.contains(&lo.col_names[j]))
            .collect();
        let kept_names: Vec<&String> = kept.iter().map(|&j| &lo.col_names[j]).collect();
        assert_eq!(
            kept_names,
            strings(&fit_json["coef_names"], name)
                .iter()
                .collect::<Vec<_>>(),
            "{name}: coefficient names"
        );
        let p = kept.len();
        let x: Vec<f64> =
            lo.x.chunks(lo.p)
                .flat_map(|row| kept.iter().map(move |&j| row[j]))
                .collect();
        let opts = FitOptions {
            se_for: SeFor::Only((0..p as u32).collect()),
            ..fit_opts.clone()
        };
        let mut data = Data::new(&x, &lo.y, lo.n, p).ids(&lo.ids);
        if let Some(w) = &lo.weights {
            data = data.weights(w);
        }
        let (_, mut fws) = fit_keep_workspace(&data, &lo.model, None, &opts).unwrap();
        let (ws, map, units) = fws
            .lmm_parts()
            .expect("a Gaussian mixed fit has an LMM workspace");
        let theta = lme4_theta_to_kernel(name, lo, fit_json, ws, map, units);
        f(&kept, ws, map, units, &theta)
    }

    /// lme4's θ̂ (`theta`, blocks in lme4's grouping order) as the kernel's θ̃:
    /// the blocks matched by grouping name into the lowered declaration order
    /// (lme4 sorts its terms by level count and names the inner grouping of
    /// `a/b` `b:a`, the crate `a:b`, so names match as sets of factors), into
    /// slot order through the workspace's `Perm`, through
    /// `LmmGroupings::theta_caller_to_design`, and times `√s` (the kernel runs
    /// on the prior weights over their scale `s`, which scales Λ̃ by `√s`).
    fn lme4_theta_to_kernel(
        name: &str,
        lo: &Lowered,
        fit_json: &Value,
        ws: &LmmWorkspace,
        map: &ColumnScaling,
        units: &Units,
    ) -> Vec<f64> {
        let groups = strings(&fit_json["group_names"], name);
        let cnms: Vec<Vec<String>> = fit_json["cnms"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: cnms"))
            .iter()
            .map(|c| strings(c, name))
            .collect();
        let theta = numbers(&fit_json["theta"], name);
        let mut start = vec![0];
        for c in &cnms {
            start.push(start.last().unwrap() + c.len() * (c.len() + 1) / 2);
        }
        assert_eq!(start.last(), Some(&theta.len()), "{name}: theta length");
        assert_eq!(groups.len(), lo.re_groups.len(), "{name}: groupings");
        let factors = |g: &str| {
            let mut v: Vec<&str> = g.split(':').collect();
            v.sort_unstable();
            v.join(":")
        };
        let mut blocks: Vec<Vec<f64>> = lo
            .re_groups
            .iter()
            .map(|rg| {
                let g = groups
                    .iter()
                    .position(|n| factors(n) == factors(&rg.name))
                    .unwrap_or_else(|| panic!("{name}: grouping {} not in {groups:?}", rg.name));
                assert_eq!(cnms[g], rg.terms, "{name} {}: terms", rg.name);
                theta[start[g]..start[g + 1]].to_vec()
            })
            .collect();
        let (_, _, perm) = spec_sized_from_ids(&lo.model, &lo.ids, true);
        perm.swap_slots(&mut blocks);
        let mut out: Vec<f64> = blocks.concat();
        let groupings = ws.kernel.groupings();
        let mut scratch = vec![0.0; groupings.re_map_scratch_len()];
        groupings.theta_caller_to_design(&mut out, &mut scratch, map);
        let root_s = units.s().sqrt();
        out.iter_mut().for_each(|t| *t *= root_s);
        out
    }

    /// `θ̃₀ − H_p⁺ĝ` on the free coordinates, `ĝ` the central gradient at the
    /// value route's axis points (its step [`ETA`] on either route, on the
    /// shifted response) and `H_p⁺` the pass's (`A_eff`'s θ block over 2). The
    /// coordinates of a zero Λ column stay 0.
    pub(crate) fn newton_point(
        ws: &mut LmmWorkspace,
        st: &SatterthwaiteState,
        theta: &[f64],
    ) -> Vec<f64> {
        let free = free_coordinates(ws.kernel.groupings(), theta);
        let n = free.len();
        assert_eq!(n, st.jac_theta.len(), "the pass's free coordinates");
        let mut h = Vec::new();
        crate::lmm::satterthwaite::steps_into(ws.kernel.groupings(), theta, &free, ETA, &mut h);
        let saved = shift_response(&mut ws.kernel, &ws.recovery.beta_tilde);
        let mut point = theta.to_vec();
        let grad: Vec<f64> = free
            .iter()
            .zip(&h)
            .map(|(&k, &h)| {
                point[k] = theta[k] + h;
                let fp = ws.kernel.deviance(&point);
                point[k] = theta[k] - h;
                let fm = ws.kernel.deviance(&point);
                point[k] = theta[k];
                (fp - fm) / (2.0 * h)
            })
            .collect();
        restore_response(&mut ws.kernel, &saved);
        let mut out = theta.to_vec();
        for (a, &k) in free.iter().enumerate() {
            out[k] -= (0..n)
                .map(|b| 0.5 * st.a_eff[(a, b)] * grad[b])
                .sum::<f64>();
        }
        out
    }

    /// The pseudo-inverse direction check on a boundary fixture (does the
    /// crate's relative eigenvalue rule drop the directions lmerTest's absolute
    /// one drops?): lmerTest keeps the eigenvalues above 1e-8 of its full
    /// `(θ, σ)` Hessian `h`, frozen on the boundary references; with the
    /// coordinates of every Λ column that is exactly zero at lme4's θ̂ removed,
    /// the count of eigenvalues `≤ 1e-8` equals the crate's
    /// `dropped + negative` on `H_p` at lme4's θ̂ (its relative rule, on `H_p`
    /// scaled to a unit diagonal).
    fn pseudo_inverse_drops_what_lmertest_drops(name: &str, json: &Value, st: &SatterthwaiteState) {
        let h = matrix(&json["h"], name);
        let theta = numbers(&json["theta"], name);
        let mut kept = Vec::new();
        let mut start = 0;
        for c in json["cnms"].as_array().expect("cnms") {
            let q = c.as_array().expect("cnms block").len();
            let mut at = start;
            for col in 0..q {
                let run = at..at + q - col;
                if theta[run.clone()].iter().any(|&t| t != 0.0) {
                    kept.extend(run.clone());
                }
                at = run.end;
            }
            start = at;
        }
        kept.push(theta.len());
        assert_eq!(h.nrows(), theta.len() + 1, "{name}: h is (θ, σ) square");
        let reduced = Mat::from_fn(kept.len(), kept.len(), |i, j| {
            0.5 * (h[(kept[i], kept[j])] + h[(kept[j], kept[i])])
        });
        let eig = sym_eigen(reduced.as_ref(), false).expect("h is finite");
        let small = eig.values.iter().filter(|&&l| l <= 1e-8).count();
        eprintln!(
            "GATE1-PINV {name}: h eigenvalues {:?}, lmerTest drops {small}, crate (dropped, negative) ({}, {})",
            eig.values, st.dropped, st.negative
        );
        assert_eq!(
            small,
            st.dropped + st.negative,
            "{name}: lmerTest's rule drops {small} directions, the crate's {} + {}",
            st.dropped,
            st.negative
        );
    }

    /// Gate 1 at lme4's θ̂, per coefficient, on every per-coefficient
    /// reference of both criteria (the REML ones and the `_ml` ones, each
    /// fitted under its own `reml`) and on each route the fit can take
    /// ([`routes`]: the value route everywhere, the exact pass where it covers
    /// the kernel): `|df − df_lmerTest| / df_lmerTest` within the route's
    /// [`band`], an over-band row excused only by an entry under `df_at_lme4`
    /// (the end-to-end gate on the same references keeps `df`). Also the
    /// pseudo-inverse direction check on the boundary references, the
    /// mixed-point stencil with unequal steps on a real fit with a `q = 2`
    /// block (`h_k` differs between blocks whose largest `|θ̃|` differ), and
    /// the direct check of ν on ML: `lmertest_Dyestuff2_ml`, pinned at θ = 0,
    /// gets `ν = N` exactly.
    #[test]
    fn gate1_at_lme4_theta() {
        let mut driver = Driver::new(&["df_at_lme4"]);
        for reml in [true, false] {
            let suffix = if reml { "" } else { "_ml" };
            let entries = lmertest_entries(false, reml);
            eprintln!(
                "GATE1 reml {reml}: {} per-coefficient references compared",
                entries.len()
            );
            let mut unequal_steps = Vec::new();
            let mut exact = Vec::new();
            let mut boundary = 0;
            let mut pinned_nu = None;
            for e in entries {
                let name = e.name.as_str();
                let json = test_refs::load(Kind::Lmertest, name);
                let (lo, fit_opts) = lowered(&e);
                let want = numbers(&json["df"], name);
                let second: Vec<Vec<f64>> = json["df_second"]
                    .as_array()
                    .unwrap_or_else(|| panic!("{name}: df_second"))
                    .iter()
                    .map(|r| numbers(r, name))
                    .collect();
                let coef = strings(&json["coef_names"], name);
                let q_max = json["cnms"]
                    .as_array()
                    .expect("cnms")
                    .iter()
                    .map(|c| c.as_array().expect("cnms block").len())
                    .max()
                    .unwrap_or(0);
                at_lme4_theta(name, &lo, &fit_opts, &json, |_, ws, map, units, theta| {
                    assert_eq!(
                        ws.kernel.criterion() == crate::lmm::Criterion::Reml,
                        reml,
                        "{name}: the workspace's criterion"
                    );
                    let routes = routes(ws);
                    if routes.contains(&DerivRoute::Exact) {
                        exact.push(name.to_string());
                    }
                    let mut value_state = None;
                    for route in routes {
                        let st = route_pass_at(ws, theta, map, units, route);
                        let df = coef_df_in_range(&st);
                        let theta1 = newton_point(ws, &st, theta);
                        let moved = route_pass_at(ws, &theta1, map, units, route).coef_df();
                        assert_eq!(df.len(), want.len(), "{name}: coefficients");
                        for i in 0..want.len() {
                            let err = (df[i] - want[i]).abs() / want[i];
                            let sp = spread(want[i], &second[i]);
                            let nonstat = (df[i] - moved[i]).abs() / want[i];
                            let b = band(route, sp, nonstat);
                            eprintln!(
                                "GATE1 {name} {route:?} {}: df {} lmerTest {} rel {err:.2e} spread {sp:.2e} nonstat {nonstat:.2e} band {b:.2e} n_free {}",
                                coef[i], df[i], want[i], st.jac_theta.len()
                            );
                            let what = format!(
                                "{} at lme4's theta, {route:?} route (spread {sp:.2e}, nonstat {nonstat:.2e})",
                                coef[i]
                            );
                            driver.check(name, "df_at_lme4", &what, (df[i], want[i]), b);
                        }
                        if route == DerivRoute::Value {
                            value_state = Some(st);
                        }
                    }
                    let st = value_state.expect("the value route always runs");
                    if name == "lmertest_Dyestuff2_ml" {
                        assert!(st.jac_theta.is_empty(), "{name}: θ is pinned");
                        assert_eq!(st.nu, lo.n as f64, "{name}: ν = N on ML");
                        assert_eq!(st.coef_df(), [lo.n as f64], "{name}: df = N");
                        pinned_nu = Some(st.nu);
                    }
                    let free = free_coordinates(ws.kernel.groupings(), theta);
                    let mut steps = Vec::new();
                    crate::lmm::satterthwaite::steps_into(
                        ws.kernel.groupings(),
                        theta,
                        &free,
                        1.0,
                        &mut steps,
                    );
                    if q_max >= 2 && steps.iter().any(|&s| s != steps[0]) {
                        unequal_steps.push(name.to_string());
                    }
                    if json["boundary"] != "none" {
                        pseudo_inverse_drops_what_lmertest_drops(name, &json, &st);
                        boundary += 1;
                    }
                });
                driver.compared(name);
            }
            eprintln!(
                "GATE1 reml {reml}: mixed points with unequal steps on q >= 2 fits: {unequal_steps:?}"
            );
            eprintln!("GATE1 reml {reml}: exact pass also run on: {exact:?}");
            exact.sort();
            let mut want_exact: Vec<String> =
                GATE1_EXACT.iter().map(|n| format!("{n}{suffix}")).collect();
            want_exact.sort();
            assert_eq!(
                exact, want_exact,
                "reml {reml}: the references whose fit the exact pass covers"
            );
            assert!(
                !unequal_steps.is_empty(),
                "reml {reml}: no q >= 2 reference exercises the mixed stencil with unequal steps"
            );
            assert_eq!(
                boundary, 3,
                "reml {reml}: the boundary references (singular, correlation, near-singular)"
            );
            assert_eq!(
                pinned_nu.is_some(),
                !reml,
                "the ML pinned reference is compared on ML only"
            );
        }
        driver.finish();
    }
}
