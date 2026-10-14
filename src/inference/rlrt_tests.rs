//! The design eigenvalues, the observed statistic and the refusals of the
//! exact RLRT: the Z-free route against `ZᵀP₀Z` formed densely from the raw
//! design, the routing of a `(1 | g)` model, every refusal the core returns,
//! the floor and NaN rules of the statistic, the grid kernel's profile and
//! maximum, and the simulated null: against the balanced design's F test, its
//! determinism, the zero-eigenvalue merge, the zero-tail branch and the
//! p-value rules; and `exact_rlrt`: its zero, floor and NaN results, its
//! fields, its refusals, and Dyestuff's p-value against the F test. Also the
//! cost sweep (`rlrt_cost_sweep`, ignored), which prints wall times.

use super::evd::{sym_eigen, EvdError};
use super::rlrt::{
    exact_rlrt, max_on_grid, null_draws, profile, rlrt_design, rlrt_null, rlrt_statistic,
    GridTables, RlrtConfig, RlrtDesign, RlrtNull, RlrtRefusal, RlrtResult, LAMBDA_GRID,
    MU_ZERO_REL,
};
use super::rlrt_refs::{bands, csv};
use crate::fit::common_tests::lcg;
use crate::fit::{classify_design, Solver};
use crate::test_support::intercept_only_spec;
use crate::{fit_cold, Family, Fit, FitOptions, GroupIds, ModelSpec, Sizing};
use commonstats::dist::{ChiSquared, ContinuousCdf, FisherF, Normal, Sampler};
use commonstats::rng::{CommonStatsRng, STREAM_TAG_RLRT};
use faer::linalg::matmul::matmul;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::{Accum, Mat, Par};
use std::time::Instant;

/// A `(1 | g)` design: `sizes[a]` rows in level `a`, each row's design
/// columns from `row(a, state)`, and a response with a level effect of scale
/// `effect`. Returns row-major `x`, `y`, the primary ids and `p`.
fn design(
    sizes: &[usize],
    effect: f64,
    row: impl Fn(usize, &mut u64) -> Vec<f64>,
) -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
    let mut st = 29u64;
    let (mut x, mut y, mut ids, mut p) = (vec![], vec![], vec![], 0);
    for (a, &na) in sizes.iter().enumerate() {
        let b = effect * lcg(&mut st);
        for _ in 0..na {
            let r = row(a, &mut st);
            p = r.len();
            y.push(0.4 + 0.3 * r[1..].iter().sum::<f64>() + b + 0.5 * lcg(&mut st));
            x.extend(r);
            ids.push(a as u32);
        }
    }
    (x, y, ids, p)
}

/// The `(1 | g)` fit of a design.
fn fit_of(x: &[f64], y: &[f64], ids: &[u32], p: usize, opts: &FitOptions) -> Fit {
    let model = intercept_only_spec(Sizing::FixedClusters { n_clusters: 1 });
    let gids = GroupIds {
        primary: ids.to_vec(),
        extra: vec![],
    };
    fit_cold(x, y, ids.len(), p, &model, &gids, opts)
}

/// `rlrt_design` on a fit's design summary.
fn design_of(tag: &str, fit: &Fit) -> Result<RlrtDesign, RlrtRefusal> {
    let ds = fit
        .design_summary
        .as_ref()
        .unwrap_or_else(|| panic!("{tag}: no design summary"));
    rlrt_design(&ds.counts, &ds.s, &ds.c)
}

/// The spectrum of `ZᵀP₀Z` formed densely from the raw design: `Z` the
/// `n × K` level indicators, `P₀ = I − QQᵀ` with `Q` the thin Householder Q
/// of `x`, the eigenvalues by `sym_eigen`, then clamped, sorted descending and
/// normalized by the largest. Returns the values at or above `MU_ZERO_REL`,
/// the count below it, and `κ(x)`.
fn dense_spectrum(x: &[f64], p: usize, ids: &[u32]) -> (Vec<f64>, usize, f64) {
    let n = ids.len();
    let k = ids.iter().map(|&a| a as usize + 1).max().unwrap();
    let xm = Mat::<f64>::from_fn(n, p, |i, j| x[i * p + j]);
    let q = xm.as_ref().qr().compute_thin_Q();
    let p0 = Mat::<f64>::from_fn(n, n, |i, j| {
        let id = if i == j { 1.0 } else { 0.0 };
        id - (0..p).map(|l| q[(i, l)] * q[(j, l)]).sum::<f64>()
    });
    let z = Mat::<f64>::from_fn(n, k, |i, a| if ids[i] as usize == a { 1.0 } else { 0.0 });
    let m = z.transpose() * &p0 * &z;
    let m = Mat::<f64>::from_fn(k, k, |a, b| 0.5 * (m[(a, b)] + m[(b, a)]));
    let mut mu = sym_eigen(m.as_ref(), false).expect("EVD of ZᵀP₀Z").values;
    mu.iter_mut().for_each(|v| *v = v.max(0.0));
    mu.sort_by(|a, b| b.total_cmp(a));
    let top = mu[0];
    mu.iter_mut().for_each(|v| *v /= top);
    let nonzero: Vec<f64> = mu.iter().copied().filter(|&v| v >= MU_ZERO_REL).collect();
    let sv = xm.as_ref().singular_values().expect("SVD of the design");
    let (lo, hi) = sv.iter().fold((f64::INFINITY, 0.0f64), |(lo, hi), &v| {
        (lo.min(v), hi.max(v))
    });
    let zeros = k - nonzero.len();
    (nonzero, zeros, hi / lo)
}

/// `rlrt_design` on the fit's summary against the dense spectrum
/// ([`compare_with_dense`]).
fn check_against_dense(tag: &str, fit: &Fit, x: &[f64], p: usize, ids: &[u32]) {
    assert!(
        fit.diagnostics.aliased.iter().all(|&a| !a),
        "{tag}: aliased"
    );
    let d = design_of(tag, fit).unwrap_or_else(|e| panic!("{tag}: {}", e.reason()));
    let k_levels = fit.design_summary.as_ref().unwrap().counts.len();
    compare_with_dense(tag, &d, k_levels, x, p, ids);
}

/// `d`, from a summary of `k_levels` levels, against the dense spectrum: the
/// same nonzero count and zero count, every normalized `μ` within
/// `100·ε·κ(X)` (the two routes differ by about `ε·κ(X)`), and `n − p_eff`
/// and the tail df from the raw design, which checks `n = Σ counts`. Prints
/// `κ(X)` beside the measured gap.
fn compare_with_dense(
    tag: &str,
    d: &RlrtDesign,
    k_levels: usize,
    x: &[f64],
    p: usize,
    ids: &[u32],
) {
    let (want, zeros, kappa) = dense_spectrum(x, p, ids);
    assert_eq!(d.mu.len(), want.len(), "{tag}: nonzero count");
    assert_eq!(k_levels - d.mu.len(), zeros, "{tag}: zero count");
    let band = 100.0 * f64::EPSILON * kappa;
    let worst =
        d.mu.iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
    eprintln!(
        "{tag}: kappa(X) = {kappa:.3e}, k = {}, max |dmu| = {worst:.3e}, band {band:.3e}",
        d.mu.len()
    );
    assert!(worst <= band, "{tag}: max |dmu| {worst:e} > {band:e}");
    let n = ids.len();
    assert_eq!(d.n_minus_p, n - p, "{tag}: n - p_eff");
    assert_eq!(d.tail_df, n - p - d.mu.len(), "{tag}: tail df");
}

#[test]
fn explicit_z_matches_z_free_route() {
    let opts = FitOptions::default();
    let x1 = |_: usize, st: &mut u64| vec![1.0, lcg(st)];

    let (x, y, ids, p) = design(&[2, 7, 3, 9, 4], 1.0, x1);
    let fit = fit_of(&x, &y, &ids, p, &opts);
    check_against_dense("unbalanced", &fit, &x, p, &ids);

    // Level 2 has no rows.
    let (x, y, ids, p) = design(&[4, 6, 0, 5, 3], 1.0, x1);
    let fit = fit_of(&x, &y, &ids, p, &opts);
    check_against_dense("empty level", &fit, &x, p, &ids);

    let (x, y, ids, p) = design(&[4, 6, 5, 3, 7, 5], 1.0, |a, st| {
        vec![1.0, lcg(st), (a as f64 - 2.5) * 0.7]
    });
    let fit = fit_of(&x, &y, &ids, p, &opts);
    check_against_dense("group-level covariate", &fit, &x, p, &ids);

    let (x, y, ids, p) = design(&[3, 8, 5, 6, 4], 1.0, x1);
    let with_offset = FitOptions {
        offset: Some((0..ids.len()).map(|i| 0.1 * (i % 7) as f64).collect()),
        ..FitOptions::default()
    };
    let fit = fit_of(&x, &y, &ids, p, &with_offset);
    check_against_dense("offset", &fit, &x, p, &ids);

    // `x2 = x1 + 1e-4·noise`: a pivot ratio far under `GRAM_PIVOT_MIN`, so
    // the fit takes the Givens pass.
    let (x, y, ids, p) = design(&[6, 8, 5, 7, 6, 9], 1.0, |_, st| {
        let v = lcg(st);
        vec![1.0, v, v + 1e-4 * lcg(st)]
    });
    let n = ids.len();
    let gram = Mat::<f64>::from_fn(p, p, |i, j| {
        (0..n).map(|r| x[r * p + i] * x[r * p + j]).sum()
    });
    let l = gram.as_ref().llt(faer::Side::Lower).expect("full rank");
    let pivot = (0..p)
        .map(|j| l.L()[(j, j)].powi(2) / gram[(j, j)])
        .fold(f64::INFINITY, f64::min);
    assert!(pivot < crate::lmm::GRAM_PIVOT_MIN, "pivot ratio {pivot:e}");
    let fit = fit_of(&x, &y, &ids, p, &opts);
    check_against_dense("givens", &fit, &x, p, &ids);

    // The raw `XᵀZ` and `XᵀX`, no fit: `rlrt_design` holds on any basis of
    // the column space, so a `c` far from diagonal gives the dense spectrum
    // too (a fit's orthogonalized `c` is diagonal to round-off).
    let (x, _, ids, p) = design(&[4, 6, 5, 3, 7, 5], 1.0, |a, st| {
        vec![1.0, 2.0 + lcg(st), (a as f64 - 1.0) * 0.7]
    });
    let k = ids.iter().map(|&a| a as usize + 1).max().unwrap();
    let (mut counts, mut s, mut c) = (vec![0.0; k], vec![0.0; p * k], vec![0.0; p * p]);
    for (row, &a) in x.chunks(p).zip(&ids) {
        let a = a as usize;
        counts[a] += 1.0;
        for (j, &xj) in row.iter().enumerate() {
            s[a * p + j] += xj;
            for (i, &xi) in row.iter().enumerate() {
                c[j * p + i] += xi * xj;
            }
        }
    }
    let corr = (0..p)
        .flat_map(|j| (0..j).map(move |i| (i, j)))
        .map(|(i, j)| c[j * p + i].abs() / (c[i * p + i] * c[j * p + j]).sqrt())
        .fold(0.0f64, f64::max);
    assert!(
        corr > 0.5,
        "X'X nearly diagonal: largest correlation {corr}"
    );
    let d = rlrt_design(&counts, &s, &c).unwrap_or_else(|e| panic!("raw: {}", e.reason()));
    compare_with_dense("raw X'Z and X'X", &d, k, &x, p, &ids);
}

#[test]
fn one_random_intercept_routes_noz() {
    let model = intercept_only_spec(Sizing::FixedClusters { n_clusters: 5000 });
    assert_eq!(classify_design(&model, 1), Solver::NoZ);
}

/// The hand-built refusal cases, `(counts, s, c)` and the refusal, each an
/// input that reaches it: [`refusals`] checks them on `rlrt_design` and
/// [`exact_rlrt_refusals`] on `exact_rlrt`.
type RefusalCase = (Vec<f64>, Vec<f64>, Vec<f64>, RlrtRefusal);
fn refusal_cases() -> Vec<RefusalCase> {
    use RlrtRefusal::*;
    let eye = vec![1.0, 0.0, 0.0, 1.0];
    vec![
        // `n = p_eff`: two rows, two columns.
        (vec![1.0, 1.0], eye.clone(), eye.clone(), NoResidualDf),
        // Nothing at all.
        (vec![], vec![], vec![], NoResidualDf),
        // A positive `ZᵀP₀Z` at round-off scale, `M = 4 − (2 − 1e-13)² ≈
        // 4e-13`, under `MU_ZERO_REL` times the count: refused, never
        // normalized to 1.
        (vec![4.0], vec![2.0 - 1e-13], vec![1.0], NoGroupVariation),
        // Sizes that disagree: `s` not `p × K`, `c` not square, `c` 2 × 2
        // with a one-row `s`.
        (vec![3.0, 4.0], vec![3.0], vec![7.0], MalformedSummary),
        (
            vec![3.0, 4.0],
            vec![3.0, 4.0],
            vec![7.0, 0.0],
            MalformedSummary,
        ),
        (
            vec![3.0, 4.0],
            vec![3.0, 4.0],
            eye.clone(),
            MalformedSummary,
        ),
        // A non-finite entry in each array.
        (
            vec![3.0, f64::INFINITY],
            vec![3.0, 4.0],
            vec![7.0],
            MalformedSummary,
        ),
        (
            vec![3.0, 4.0],
            vec![f64::NAN, 4.0],
            vec![7.0],
            MalformedSummary,
        ),
        (
            vec![3.0, 4.0],
            vec![3.0, 4.0],
            vec![f64::NAN],
            MalformedSummary,
        ),
        // Counts that are not row counts.
        (vec![3.5, 4.0], vec![3.5, 4.0], vec![7.5], MalformedSummary),
        (vec![-3.0, 4.0], vec![3.0, 4.0], vec![7.0], MalformedSummary),
        // A `c` the Cholesky refuses (indefinite).
        (
            vec![3.0, 4.0],
            eye,
            vec![1.0, 2.0, 2.0, 1.0],
            MalformedSummary,
        ),
        // More nonzero `μ` than `n − p_eff`, which no real design gives:
        // `M = I₃` with `n − p_eff = 2`.
        (
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0],
            MalformedSummary,
        ),
        // Finite entries whose `ZᵀP₀Z` overflows: the EVD refuses it.
        (
            vec![3.0, 4.0],
            vec![1e200, 1e200],
            vec![1.0],
            Eigen(EvdError::NonFinite),
        ),
    ]
}

/// Every refusal the core returns, each from an input that reaches it, and
/// every `reason()` non-empty and distinct. Each call returns, so no input
/// here panics.
#[test]
fn refusals() {
    use RlrtRefusal::*;

    // An intercept-only summary: `X̃ = 1`, so `s` holds the counts and `c = n`.
    let good = rlrt_design(&[3.0, 4.0], &[3.0, 4.0], &[7.0]).expect("a valid summary");
    assert_eq!((good.mu.len(), good.n_minus_p, good.tail_df), (1, 6, 5));

    for (i, (counts, s, c, want)) in refusal_cases().into_iter().enumerate() {
        assert_eq!(rlrt_design(&counts, &s, &c), Err(want), "case {i}");
    }

    // An intercept plus a dummy for every level: the fit drops one column,
    // and the rest span `Z`.
    let (x, y, ids, p) = design(&[5, 6, 4, 7], 1.0, |a, st| {
        let mut r = vec![1.0, lcg(st)];
        r.extend((0..4).map(|b| if a == b { 1.0 } else { 0.0 }));
        r
    });
    let fit = fit_of(&x, &y, &ids, p, &FitOptions::default());
    assert_eq!(fit.diagnostics.aliased.iter().filter(|&&a| a).count(), 1);
    assert_eq!(design_of("dummies", &fit), Err(NoGroupVariation));

    // No fixed effects: `M = diag(counts)`.
    let free = rlrt_design(&[2.0, 3.0], &[], &[]).expect("p = 0 is a design");
    assert_eq!(free.mu, vec![1.0, 2.0 / 3.0]);
    assert_eq!((free.n_minus_p, free.tail_df), (5, 3));

    // One arm per variant, so a new variant fails to compile here.
    let reasons: Vec<&str> = [
        NoResidualDf,
        NoGroupVariation,
        Eigen(EvdError::NoConvergence),
        MalformedSummary,
    ]
    .iter()
    .map(|r| match r {
        NoResidualDf | NoGroupVariation | Eigen(_) | MalformedSummary => r.reason(),
    })
    .collect();
    for (i, a) in reasons.iter().enumerate() {
        assert!(!a.trim().is_empty(), "an empty reason");
        assert!(
            reasons[i + 1..].iter().all(|b| a != b),
            "a repeated reason: {a}"
        );
    }
    for e in [EvdError::NotSquare, EvdError::NonFinite] {
        assert_eq!(Eigen(e).reason(), reasons[2]);
    }
}

/// The log-likelihood pairs `(loglik_full, loglik_reduced)` whose statistic
/// is NaN, pinned or not: either side NaN, `loglik_full = −∞` (the `(1 | g)`
/// fit collapsed), or `loglik_reduced = −∞` (the OLS fit collapsed).
const NAN_PAIRS: [(f64, f64); 6] = [
    (f64::NAN, -10.0),
    (-10.0, f64::NAN),
    (f64::NAN, f64::INFINITY),
    (f64::NEG_INFINITY, -10.0),
    (f64::NEG_INFINITY, f64::INFINITY),
    (-10.0, f64::NEG_INFINITY),
];

#[test]
fn statistic_rules() {
    for (l1, l0) in NAN_PAIRS {
        for pinned in [false, true] {
            let t = rlrt_statistic(l1, l0, pinned);
            assert!(t.is_nan(), "({l1}, {l0}), pinned {pinned}: {t}");
        }
    }
    // The pinned rule, on log-likelihoods that are not NaN or −∞.
    for (l1, l0) in [
        (-10.0, -12.5),
        (-12.5, -10.0),
        (3.0, 3.0),
        (1.0, f64::INFINITY),
        (f64::INFINITY, -4.0),
    ] {
        assert_eq!(rlrt_statistic(l1, l0, true).to_bits(), 0.0f64.to_bits());
    }
    assert_eq!(rlrt_statistic(-10.0, -12.5, false), 5.0);
    // The floor: exactly +0.0, never a negative value or -0.0.
    assert_eq!(
        rlrt_statistic(-12.5, -10.0, false).to_bits(),
        0.0f64.to_bits()
    );
    assert_eq!(rlrt_statistic(3.0, 3.0, false).to_bits(), 0.0f64.to_bits());
    // An exact OLS fit reports `+∞`.
    assert_eq!(
        rlrt_statistic(-4.0, f64::INFINITY, false).to_bits(),
        0.0f64.to_bits()
    );
    // The mirror: `loglik_full = +∞` stays `+∞` unpinned (0 pinned, above).
    assert_eq!(rlrt_statistic(f64::INFINITY, -4.0, false), f64::INFINITY);

    // Real fits against the OLS fit made with `reml = Some(true)`.
    let ols_of = |x: &[f64], y: &[f64], p: usize| {
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let opts = FitOptions {
            reml: Some(true),
            ..FitOptions::default()
        };
        let fit = fit_cold(x, y, y.len(), p, &model, &GroupIds::default(), &opts);
        assert!(fit.reml);
        fit
    };
    let pinned_of = |fit: &Fit| fit.diagnostics.pinned.iter().flatten().any(|&b| b);

    // Every level holds the same five values, so the level means agree and
    // the between-level variance is pinned at zero.
    let base = [0.3, -1.1, 0.8, 2.0, -0.4];
    let ids: Vec<u32> = (0..30).map(|i| (i / 5) as u32).collect();
    let y: Vec<f64> = (0..30).map(|i| base[(i + i / 5) % 5]).collect();
    let x = vec![1.0; 30];
    let full = fit_of(&x, &y, &ids, 1, &FitOptions::default());
    assert!(full.converged() && pinned_of(&full), "a pinned fit");
    let t = rlrt_statistic(full.loglik, ols_of(&x, &y, 1).loglik, pinned_of(&full));
    assert_eq!(t.to_bits(), 0.0f64.to_bits());

    // An exact fit, `y` constant on `x = 1`: the OLS fit reports `+∞`, and the
    // `(1 | g)` fit does not converge and reports a NaN log-likelihood, so the
    // pair tests as NaN and never reaches the `+∞` rule above.
    let y = vec![1.0; 30];
    let full = fit_of(&x, &y, &ids, 1, &FitOptions::default());
    let reduced = ols_of(&x, &y, 1);
    assert_eq!(reduced.loglik, f64::INFINITY, "an exact OLS fit");
    assert!(
        !full.converged() && full.loglik.is_nan() && !pinned_of(&full),
        "the (1 | g) fit of an exact fit: loglik {}",
        full.loglik
    );
    assert!(rlrt_statistic(full.loglik, reduced.loglik, false).is_nan());

    let (x, y, ids, p) = design(&[5, 6, 4, 7, 5, 6], 3.0, |_, st| vec![1.0, lcg(st)]);
    let full = fit_of(&x, &y, &ids, p, &FitOptions::default());
    assert!(full.converged() && !pinned_of(&full), "an interior fit");
    let t = rlrt_statistic(full.loglik, ols_of(&x, &y, p).loglik, pinned_of(&full));
    assert!(t > 0.0, "interior statistic {t}");
}

/// A draw for `k` eigenvalues: `μ` in `(0, 1]` with `μ[0] = 1` (normalized,
/// any order), and positive `χ²₁` values.
fn draw(k: usize, st: &mut u64) -> (Vec<f64>, Vec<f64>) {
    let mut mu: Vec<f64> = (0..k).map(|_| 0.05 + 0.95 * lcg(st).abs()).collect();
    mu[0] = 1.0;
    let chi1 = (0..k).map(|_| (2.0 * lcg(st)).powi(2) + 1e-3).collect();
    (mu, chi1)
}

/// `f_0` is `+0.0` to the bit on any draw (`λ = 0` gives `N = 0` and
/// `SL = 0`): with zero `μ` among the entries, a zero tail, zero `χ²₁`
/// entries, and on an explicit grid as well as [`LAMBDA_GRID`]. And a draw
/// whose profile falls from `λ = 0` has its maximum at index 0, exactly
/// `+0.0`: an exact zero of the null.
#[test]
fn profile_at_zero_is_exactly_zero() {
    let mut st = 7u64;
    let explicit = [0.0, 1e-3, 0.4, 2.0, 50.0];
    for k in [1, 2, 5, 40] {
        for n_minus_p in [k, k + 1, 3 * k + 7] {
            let (mut mu, mut chi1) = draw(k, &mut st);
            if k > 2 {
                mu[k - 1] = 0.0;
                chi1[1] = 0.0;
            }
            for tail in [0.0, 0.7, 30.0] {
                for grid in [&LAMBDA_GRID[..], &explicit[..]] {
                    let f = profile(&mu, grid, &chi1, tail, n_minus_p);
                    assert_eq!(f.len(), grid.len());
                    assert_eq!(f[0].to_bits(), 0.0f64.to_bits(), "k {k}, tail {tail}");
                }
            }
        }
    }
    // Small group draws against a large tail: the profile falls from λ = 0.
    let mu = [1.0, 0.5, 0.25];
    let chi1 = [1e-4, 2e-4, 1e-4];
    let f = profile(&mu, &LAMBDA_GRID[..], &chi1, 40.0, 30);
    assert!(f[1..].iter().all(|&v| v < 0.0));
    let (t, at) = max_on_grid(&GridTables::new(&mu, &LAMBDA_GRID[..]), &chi1, 40.0, 30);
    assert_eq!((t.to_bits(), at), (0.0f64.to_bits(), 0));
}

/// A two-eigenvalue case worked by hand: `μ = (1, 1/4)`, `χ²₁ = (2, 1/2)`,
/// tail 3, `n − p = 7`. At `λ = 1/2`, `fN = (1/3, 1/9)`, `fD = (2/3, 8/9)`,
/// so `N = 13/18`, `D = 43/9`, `N/D = 13/86`, and `SL = log1p(1/2) +
/// log1p(1/8)`. At `λ = 4`, `fN = (4/5, 1/2)`, `fD = (1/5, 1/2)`, so
/// `N = 1.85`, `D = 3.65`, and `SL = log1p(4) + log1p(1)`. Within a few ulps
/// of the `(n − p)·log1p` term.
#[test]
fn profile_matches_hand_values() {
    let want = [
        0.0,
        7.0 * (13.0f64 / 86.0).ln_1p() - 0.5f64.ln_1p() - 0.125f64.ln_1p(),
        7.0 * (1.85f64 / 3.65).ln_1p() - 4.0f64.ln_1p() - 1.0f64.ln_1p(),
    ];
    let f = profile(&[1.0, 0.25], &[0.0, 0.5, 4.0], &[2.0, 0.5], 3.0, 7);
    assert_eq!(f.len(), 3);
    for (i, (&got, &w)) in f.iter().zip(&want).enumerate() {
        assert!((got - w).abs() <= 1e-14, "λ index {i}: {got} against {w}");
    }
    // The global maximum is the larger of the two positive values.
    let (t, at) = max_on_grid(
        &GridTables::new(&[1.0, 0.25], &[0.0, 0.5, 4.0]),
        &[2.0, 0.5],
        3.0,
        7,
    );
    assert_eq!((t, at), (f[2], 2));
}

/// Decision 1: the maximum is the global one, never the first decrease. The
/// hand-values draw on the grid `[0, 1/2, 1/10, 4]` has the profile about
/// `[0, 0.4622, 0.1314, 0.5676]`: a walk that stops at the first decrease
/// ends at index 1, the global maximum is index 3.
#[test]
fn global_maximum_not_first_decrease() {
    let grid = [0.0, 0.5, 0.1, 4.0];
    let f = profile(&[1.0, 0.25], &grid, &[2.0, 0.5], 3.0, 7);
    assert!(
        f[2] < f[1] && f[1] < f[3],
        "not the non-monotone profile: {f:?}"
    );
    let (t, at) = max_on_grid(&GridTables::new(&[1.0, 0.25], &grid), &[2.0, 0.5], 3.0, 7);
    assert_eq!((t.to_bits(), at), (f[3].to_bits(), 3));
}

/// The grid's shape: 200 points, `λ = 0` to the bit, then `exp` of 199
/// points from `−10` to `8` with both ends exact and every log step
/// `18/198`. The tolerance on a step, 1e-13, is far above its round-off and
/// far below an off-by-one grid's step error (4.6e-4): each log point carries
/// a few ulps of 10 from R's `seq` arithmetic (about 2e-15), `exp` and `ln`
/// one ulp each (measured worst 2.7e-15).
#[test]
fn lambda_grid_shape() {
    assert_eq!(LAMBDA_GRID.len(), 200);
    assert_eq!(LAMBDA_GRID[0].to_bits(), 0.0f64.to_bits());
    assert_eq!(LAMBDA_GRID[1], (-10.0f64).exp());
    assert_eq!(LAMBDA_GRID[199], 8.0f64.exp());
    let step = 18.0 / 198.0;
    let worst = LAMBDA_GRID[1..]
        .windows(2)
        .map(|w| ((w[1] / w[0]).ln() - step).abs())
        .fold(0.0f64, f64::max);
    eprintln!("LAMBDA_GRID: worst log-step error {worst:.3e}");
    assert!(worst <= 1e-13, "a log step off by {worst:e}");
}

/// A zero `μ` adds its `χ²₁` to `D` and nothing else, so appending zero
/// eigenvalues and moving their draws into the tail leaves the profile
/// unchanged to 1e-15: `N` and `SL` gain only `+0.0` terms, and `D` moves by
/// its rounding, as the tail joins the sum in another order (the tables are
/// private, so only `f` is compared). `f_0` is `+0.0` on both sides, to the
/// bit.
#[test]
fn zero_mu_entries_are_inert() {
    let mut st = 11u64;
    for k in [1, 3, 12] {
        let (mu, chi1) = draw(k, &mut st);
        let extra = [1.3, 0.2, 2.7];
        let (mut mu_z, mut chi1_z) = (mu.clone(), chi1.clone());
        mu_z.extend([0.0; 3]);
        chi1_z.extend(extra);
        let tail = 5.3;
        let n_minus_p = k + 3 + 6;
        let merged = profile(
            &mu,
            &LAMBDA_GRID[..],
            &chi1,
            tail + extra.iter().sum::<f64>(),
            n_minus_p,
        );
        let kept = profile(&mu_z, &LAMBDA_GRID[..], &chi1_z, tail, n_minus_p);
        assert_eq!(merged[0].to_bits(), 0.0f64.to_bits());
        assert_eq!(kept[0].to_bits(), 0.0f64.to_bits());
        let mut worst = 0.0f64;
        for (i, (a, b)) in merged.iter().zip(&kept).enumerate() {
            let d = (a - b).abs() / a.abs().max(1.0);
            // Written so that a NaN fails.
            assert!(d <= 1e-15, "k {k}, λ index {i}: {a} against {b}");
            worst = worst.max(d);
        }
        eprintln!("k {k}: worst |df| {worst:.3e}");
    }
}

/// A draw whose profile rises over the whole grid: one `μ = 1` with
/// `χ²₁ = 1`, a tail of `1e-3` and `n − p = 10`. Then `df/dλ > 0` while
/// `1 + λ < (n − p − 1)·χ²₁/tail = 9000`, past the grid's top `e⁸ ≈ 2981`, so
/// the maximum is the last grid point, which the null counts as an edge hit.
#[test]
fn edge_hit_counted() {
    let last = LAMBDA_GRID.len() - 1;
    let f = profile(&[1.0], &LAMBDA_GRID[..], &[1.0], 1e-3, 10);
    assert!(
        f.windows(2).all(|w| w[1] > w[0]),
        "the profile rises throughout"
    );
    let (t, at) = max_on_grid(&GridTables::new(&[1.0], &LAMBDA_GRID[..]), &[1.0], 1e-3, 10);
    assert_eq!(at, last);
    assert_eq!(t.to_bits(), f[last].to_bits());
}

/// The summary of a balanced intercept-only design, built by hand from the
/// raw design: `K` levels of `J` rows, `X = 1`, so `counts = s = J` per level
/// and `c = n` (`M` depends on `X` only through its column space).
fn balanced(k: usize, j: usize) -> RlrtDesign {
    let jf = j as f64;
    rlrt_design(&vec![jf; k], &vec![jf; k], &[(k * j) as f64]).expect("a balanced design")
}

fn config(nsim: usize, seed: u64, parallel_inner: bool) -> RlrtConfig {
    RlrtConfig {
        nsim,
        seed,
        parallel_inner,
    }
}

/// The draws to the bit.
fn bits(draws: &[(f64, usize)]) -> Vec<(u64, usize)> {
    draws.iter().map(|&(t, i)| (t.to_bits(), i)).collect()
}

/// Gate 1, its fit-free part. On a balanced intercept-only design the RLRT is
/// a function of the one-way ANOVA F statistic, `F ~ F(K − 1, n − K)` under
/// the null: `g(F) = (n − 1)·ln((n − K + (K − 1)F)/(n − 1)) − (K − 1)·ln F`
/// for `F > 1`, increasing there, and `0` for `F ≤ 1`. So the null's mass at
/// zero is `P(F ≤ 1)`, and above that atom `P(T ≤ g(F_q)) = q`. Both within
/// the DKW band `√(ln(2/α)/(2B))`, α = [`bands::RLRT_DKW_ALPHA`], at
/// `B = 10⁵` (0.0062).
#[test]
fn balanced_null_matches_anova_f() {
    const B: usize = 100_000;
    let band = ((2.0 / bands::RLRT_DKW_ALPHA).ln() / (2.0 * B as f64)).sqrt();
    let mut worst = 0.0f64;
    for ((k, j), seed) in [(6, 5), (3, 4), (4, 5), (10, 3)]
        .into_iter()
        .zip(20261011u64..)
    {
        let n = k * j;
        let null = rlrt_null(&balanced(k, j), &config(B, seed, true));
        let f = FisherF::new((k - 1) as f64, (n - k) as f64).expect("F(K - 1, n - K)");
        let atom = f.cdf(1.0);
        let (n1, k1) = ((n - 1) as f64, (k - 1) as f64);
        let g = |x: f64| n1 * (((n - k) as f64 + k1 * x) / n1).ln() - k1 * x.ln();
        let ecdf =
            |x: f64| (null.n_zero + null.draws_pos.partition_point(|&t| t <= x)) as f64 / B as f64;
        let gap = (null.n_zero as f64 / B as f64 - atom).abs();
        eprintln!("(K, J) = ({k}, {j}): atom {atom:.4}, mass at zero off by {gap:.2e}");
        assert!(gap <= band, "({k}, {j}): mass at zero off by {gap:e}");
        worst = worst.max(gap);
        for q in [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] {
            if q <= atom {
                continue;
            }
            let fq = f.quantile(q).expect("an F quantile");
            let gap = (ecdf(g(fq)) - q).abs();
            eprintln!("({k}, {j}) q {q}: ECDF off by {gap:.2e}");
            assert!(gap <= band, "({k}, {j}) q {q}: ECDF off by {gap:e}");
            worst = worst.max(gap);
        }
    }
    eprintln!("balanced: worst gap {worst:.3e}, band {band:.3e}");
}

/// Gate 8. The same seed gives the same draws to the bit, serial or over
/// rayon (`parallel_inner`, live under `--features parallel` at any
/// `RAYON_NUM_THREADS`); the first `B` draws of a `2B` run are a `B` run's
/// draws; another seed gives other draws; the null carries its `nsim` and
/// seed. Then the pins on Dyestuff's design (`K = 6`, `J = 5`), compared
/// within 1e-12 relative, not to the bit: `μ` comes from faer's EVD and the
/// grid from the platform's `ln_1p`, whose last ulp may differ on another
/// machine or on wasm32.
#[test]
fn determinism() {
    let d = balanced(6, 5);
    // Two full 1024-draw chunks and a partial one.
    let (b, seed) = (2500, 20261031);
    let serial = null_draws(&d, &config(b, seed, false));
    assert_eq!(serial.len(), b);
    let same = null_draws(&d, &config(b, seed, false));
    assert_eq!(bits(&serial), bits(&same), "the same seed");
    let par = null_draws(&d, &config(b, seed, true));
    assert_eq!(bits(&serial), bits(&par), "serial against parallel_inner");
    let longer = null_draws(&d, &config(2 * b, seed, true));
    assert_eq!(bits(&serial), bits(&longer[..b]), "the first B of 2B");
    let other = null_draws(&d, &config(b, seed + 1, false));
    assert_ne!(bits(&serial), bits(&other), "another seed");

    let summary = |n: &RlrtNull| {
        let pos: Vec<u64> = n.draws_pos.iter().map(|t| t.to_bits()).collect();
        (pos, n.n_zero, n.edge_hits, n.nsim, n.seed)
    };
    let null = rlrt_null(&d, &config(b, seed, false));
    assert_eq!(
        summary(&null),
        summary(&rlrt_null(&d, &config(b, seed, false)))
    );
    assert_eq!(
        summary(&null),
        summary(&rlrt_null(&d, &config(b, seed, true)))
    );
    assert_eq!((null.nsim, null.seed), (b, seed));
    // The null is the draws, sorted and counted.
    let mut pos: Vec<f64> = serial.iter().map(|d| d.0).filter(|&t| t > 0.0).collect();
    pos.sort_by(f64::total_cmp);
    assert_eq!(null.draws_pos, pos);
    assert_eq!(null.n_zero, serial.iter().filter(|d| d.0 == 0.0).count());
    assert_eq!(null.n_zero + pos.len(), b);
    let last = LAMBDA_GRID.len() - 1;
    assert_eq!(
        null.edge_hits,
        serial.iter().filter(|d| d.1 == last).count()
    );

    // The pins move with the stream: a moved stream is re-pinned with its
    // provenance line, never absorbed by a band. There is no direct
    // rand_philox pin, so a `cargo update` may move them.
    let close = |got: f64, want: f64| (got - want).abs() <= 1e-12 * want.abs();
    let pinned = null_draws(&d, &config(1000, seed, false));
    let sum: f64 = rlrt_null(&d, &config(1000, seed, false))
        .draws_pos
        .iter()
        .sum();
    eprintln!("pins: first three {:?}, sum {sum:?}", &pinned[..3]);
    // Pinned 2026-10-12 on pawel (x86_64), commonstats 0.2.2 and rand_philox
    // 0.2.1 (Cargo.lock). The third draw is an exact zero of the null.
    let first = [0.9322514488828157, 0.5595149104770827, 0.0];
    for (i, (&(t, _), &want)) in pinned.iter().zip(&first).enumerate() {
        assert!(close(t, want), "T_{} = {t:e} against {want:e}", i + 1);
    }
    // Pinned 2026-10-12 on pawel (x86_64), commonstats 0.2.2 and rand_philox
    // 0.2.1 (Cargo.lock).
    assert!(
        close(sum, 411.7842766035683),
        "the sorted positive draws sum to {sum:e}"
    );
}

/// A known zero `μ` merges into the tail before any draw. `X` the indicator
/// of level 2 alone, `J = 4` (so `√4` is exact): `M = diag(4, 4, 0, 4)`
/// exactly, one zero `μ`. RLRsim's form keeps it, with its own `χ²₁` and a
/// tail of `n − p − K = 11`; removed, the tail is raised to `12`. The design
/// equals that merged one and gives the same null to the bit.
#[test]
fn zero_eigenvalue_merge() {
    let with_zero = rlrt_design(&[4.0; 4], &[0.0, 0.0, 4.0, 0.0], &[4.0]).expect("a design");
    let merged = RlrtDesign {
        mu: vec![1.0; 3],
        n_minus_p: 15,
        tail_df: 12,
    };
    assert_eq!(with_zero, merged);
    let cfg = config(3000, 20261031, false);
    assert_eq!(
        bits(&null_draws(&with_zero, &cfg)),
        bits(&null_draws(&merged, &cfg))
    );
}

/// `k = n − p`: one row per level and an intercept, so `k = K − 1 = n − 1`
/// and the tail has no degrees of freedom. The null is finite and every draw
/// is the replay of its stream with no tail drawn: `k` normals squared and a
/// tail of `0.0` (a `ChiSquared::new(0.0)` would be an error).
#[test]
fn zero_tail_df_branch() {
    let d = rlrt_design(&[1.0; 5], &[1.0; 5], &[5.0]).expect("a design");
    assert_eq!((d.mu.len(), d.n_minus_p, d.tail_df), (4, 4, 0));
    let cfg = config(500, 20261031, false);
    let tables = GridTables::new(&d.mu, &LAMBDA_GRID[..]);
    let normal = Normal::new(0.0, 1.0).expect("N(0, 1)");
    for (b, &(t, i)) in null_draws(&d, &cfg).iter().enumerate() {
        assert!(t.is_finite() && t >= 0.0, "draw {b}: {t}");
        let mut rng = CommonStatsRng::new_tagged(cfg.seed, b as u64, STREAM_TAG_RLRT);
        let chi1: Vec<f64> = (0..d.mu.len())
            .map(|_| {
                let z = normal.sample(&mut rng);
                z * z
            })
            .collect();
        let (want, at) = max_on_grid(&tables, &chi1, 0.0, d.n_minus_p);
        assert_eq!((t.to_bits(), i), (want.to_bits(), at), "draw {b}");
    }
    let null = rlrt_null(&d, &cfg);
    assert_eq!(null.n_zero + null.draws_pos.len(), cfg.nsim);
}

/// `p = (1 + #{T_b ≥ t})/(B + 1)` for `t > 0` against a brute-force count,
/// so never below `1/(B + 1)`, exactly that above every draw, ties counted;
/// `p(0) = 1`; a NaN `t` and a null of no draws give NaN, without a panic.
#[test]
fn p_value_rules() {
    let b = 1000;
    let floor = 1.0 / (b as f64 + 1.0);
    let cfg = config(b, 20261031, false);
    let d = balanced(6, 5);
    let draws: Vec<f64> = null_draws(&d, &cfg).iter().map(|d| d.0).collect();
    let null = rlrt_null(&d, &cfg);
    assert_eq!(null.p_value(0.0), 1.0);
    assert_eq!(null.p_value(-1.0), 1.0);
    assert!(null.p_value(f64::NAN).is_nan());
    let top = *null.draws_pos.last().expect("positive draws");
    let mut probes = vec![f64::MIN_POSITIVE, 1e-3, 0.5, 2.0, top];
    probes.extend(null.draws_pos.iter().step_by(37));
    for &t in &probes {
        let p = null.p_value(t);
        let count = draws.iter().filter(|&&x| x >= t).count();
        assert_eq!(p, (1 + count) as f64 / (b + 1) as f64, "t = {t}");
        assert!((floor..=1.0).contains(&p), "t = {t}: p = {p}");
    }
    for t in [f64::from_bits(top.to_bits() + 1), 2.0 * top, f64::INFINITY] {
        assert_eq!(null.p_value(t), floor, "t = {t}");
    }
    let empty = rlrt_null(&d, &config(0, 20261031, false));
    assert_eq!((empty.n_zero, empty.draws_pos.len()), (0, 0));
    assert!(empty.p_value(1.0).is_nan());
    assert_eq!(empty.p_value(0.0), 1.0);
}

/// The OLS fit of `x` and `y` made with `reml = Some(true)`: the REML
/// log-likelihood at `θ = 0`.
fn ols_reml(x: &[f64], y: &[f64], p: usize) -> Fit {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        reml: Some(true),
        ..FitOptions::default()
    };
    let fit = fit_cold(x, y, y.len(), p, &model, &GroupIds::default(), &opts);
    assert!(fit.reml && fit.loglik.is_finite());
    fit
}

/// The `(1 | g)` fit pinned its one component at zero.
fn is_pinned(fit: &Fit) -> bool {
    fit.diagnostics.pinned.iter().flatten().any(|&b| b)
}

/// `exact_rlrt` on the `(1 | g)` fit `full` and the OLS fit `reduced`.
fn exact_of(full: &Fit, reduced: &Fit, cfg: &RlrtConfig) -> Result<RlrtResult, RlrtRefusal> {
    let ds = full
        .design_summary
        .as_ref()
        .expect("a (1 | g) fit keeps its design summary");
    exact_rlrt(
        &ds.counts,
        &ds.s,
        &ds.c,
        full.loglik,
        reduced.loglik,
        is_pinned(full),
        cfg,
    )
}

/// Every field to the bit, so two results with a NaN field compare.
fn result_bits(r: &RlrtResult) -> [u64; 8] {
    [
        r.statistic.to_bits(),
        r.p_value.to_bits(),
        r.nsim as u64,
        r.seed,
        r.mass_at_zero.to_bits(),
        r.mc_se.to_bits(),
        r.k_eigen as u64,
        r.edge_hits as u64,
    ]
}

/// The result of a zero statistic: `+0.0`, `p = 1`, no draws.
fn assert_zero_result(tag: &str, r: &RlrtResult, cfg: &RlrtConfig, k_eigen: usize) {
    assert_eq!(r.statistic.to_bits(), 0.0f64.to_bits(), "{tag}: statistic");
    assert_eq!(r.p_value, 1.0, "{tag}: p-value");
    assert_eq!((r.nsim, r.edge_hits), (0, 0), "{tag}: draws");
    assert!(r.mass_at_zero.is_nan(), "{tag}: mass at zero");
    assert_eq!(r.mc_se, 0.0, "{tag}: Monte Carlo SE");
    assert_eq!((r.seed, r.k_eigen), (cfg.seed, k_eigen), "{tag}: seed, k");
}

/// A pinned fit tests as an exact zero: `p = 1` with no draws, whatever the
/// two log-likelihoods say short of the NaN rules (`exact_rlrt_floor`).
#[test]
fn exact_rlrt_zero_statistic() {
    // Every level holds the same five values, so the between-level variance
    // is pinned at zero.
    let base = [0.3, -1.1, 0.8, 2.0, -0.4];
    let ids: Vec<u32> = (0..30).map(|i| (i / 5) as u32).collect();
    let y: Vec<f64> = (0..30).map(|i| base[(i + i / 5) % 5]).collect();
    let x = vec![1.0; 30];
    let full = fit_of(&x, &y, &ids, 1, &FitOptions::default());
    assert!(full.converged() && is_pinned(&full), "a pinned fit");
    let k = design_of("pinned", &full).expect("a valid design").mu.len();
    let cfg = config(1000, 20261041, false);
    let r = exact_of(&full, &ols_reml(&x, &y, 1), &cfg).expect("no refusal");
    assert_zero_result("pinned fit", &r, &cfg, k);

    let ds = full.design_summary.as_ref().unwrap();
    let r = exact_rlrt(&ds.counts, &ds.s, &ds.c, -10.0, -12.5, true, &cfg).unwrap();
    assert_zero_result("pinned, full above reduced", &r, &cfg, k);
}

/// The floor: a full fit no better than `θ = 0`, not pinned, gives the zero
/// statistic's result, and so does an exact OLS fit, pinned or not; every
/// pair of [`NAN_PAIRS`], pinned or not, gives a NaN statistic and p-value
/// with no draws.
#[test]
fn exact_rlrt_floor() {
    let (counts, s, c) = (vec![5.0; 6], vec![5.0; 6], [30.0]);
    let cfg = config(1000, 20261041, false);
    let zero = exact_rlrt(&counts, &s, &c, -10.0, -10.0, true, &cfg).unwrap();
    assert_zero_result("pinned", &zero, &cfg, 5);
    for (l1, l0, pinned) in [
        (-12.5, -10.0, false),
        (-10.0, -10.0, false),
        (-4.0, f64::INFINITY, false),
        (-4.0, f64::INFINITY, true),
    ] {
        let r = exact_rlrt(&counts, &s, &c, l1, l0, pinned, &cfg).unwrap();
        assert_eq!(
            result_bits(&r),
            result_bits(&zero),
            "({l1}, {l0}, {pinned})"
        );
    }
    for (l1, l0) in NAN_PAIRS {
        for pinned in [false, true] {
            let tag = format!("({l1}, {l0}), pinned {pinned}");
            let r = exact_rlrt(&counts, &s, &c, l1, l0, pinned, &cfg).unwrap();
            assert!(
                r.statistic.is_nan() && r.p_value.is_nan(),
                "{tag}: statistic {}, p {}",
                r.statistic,
                r.p_value
            );
            assert_eq!((r.nsim, r.edge_hits), (0, 0), "{tag}: no draws");
            assert!(r.mass_at_zero.is_nan() && r.mc_se.is_nan(), "{tag}");
            assert_eq!((r.seed, r.k_eigen), (cfg.seed, 5), "{tag}");
        }
    }
}

/// An interior fit: the statistic, the null's p-value, mass at zero and edge
/// hits as the parts give them, the caller's `nsim` and seed, the design's
/// `k`, and `mc_se = √(p(1 − p)/nsim)`. `nsim = 0` gives a NaN p-value.
#[test]
fn exact_rlrt_fields() {
    let (x, y, ids, p) = design(&[5, 6, 4, 7, 5, 6], 1.0, |_, st| vec![1.0, lcg(st)]);
    let full = fit_of(&x, &y, &ids, p, &FitOptions::default());
    assert!(full.converged() && !is_pinned(&full), "an interior fit");
    let reduced = ols_reml(&x, &y, p);
    let cfg = config(3000, 20261042, false);
    let r = exact_of(&full, &reduced, &cfg).expect("no refusal");

    let d = design_of("interior", &full).unwrap();
    let t = rlrt_statistic(full.loglik, reduced.loglik, false);
    let null = rlrt_null(&d, &cfg);
    eprintln!("interior: t = {t}, p = {}, k = {}", r.p_value, d.mu.len());
    assert!(t > 0.0, "interior statistic {t}");
    assert_eq!(r.statistic.to_bits(), t.to_bits());
    assert_eq!((r.nsim, r.seed), (cfg.nsim, cfg.seed));
    assert_eq!(r.k_eigen, d.mu.len());
    assert_eq!(r.p_value.to_bits(), null.p_value(t).to_bits());
    assert_eq!(r.mass_at_zero, null.n_zero as f64 / cfg.nsim as f64);
    assert_eq!(r.edge_hits, null.edge_hits);
    let pv = r.p_value;
    assert!(pv > 0.0 && pv <= 1.0);
    assert_eq!(r.mc_se, (pv * (1.0 - pv) / cfg.nsim as f64).sqrt());

    let r = exact_of(&full, &reduced, &config(0, cfg.seed, false)).unwrap();
    assert!(r.p_value.is_nan() && r.mc_se.is_nan() && r.mass_at_zero.is_nan());
    assert_eq!((r.nsim, r.edge_hits), (0, 0));
}

/// Every refusal of `refusals` also comes out of `exact_rlrt`, before the
/// statistic is read: pinned or not, with a positive or a NaN statistic.
#[test]
fn exact_rlrt_refusals() {
    use RlrtRefusal::*;
    let cfg = config(1000, 20261043, false);
    let logliks = [
        (-10.0, -12.5, false),
        (-10.0, -12.5, true),
        (f64::NAN, -12.5, false),
    ];
    for (i, (counts, s, c, want)) in refusal_cases().into_iter().enumerate() {
        for (l1, l0, pinned) in logliks {
            let got = exact_rlrt(&counts, &s, &c, l1, l0, pinned, &cfg);
            assert_eq!(got, Err(want), "case {i}, ({l1}, {l0}, {pinned})");
        }
    }

    // An intercept plus a dummy for every level: the groups are absorbed.
    let (x, y, ids, p) = design(&[5, 6, 4, 7], 1.0, |a, st| {
        let mut r = vec![1.0, lcg(st)];
        r.extend((0..4).map(|b| if a == b { 1.0 } else { 0.0 }));
        r
    });
    let full = fit_of(&x, &y, &ids, p, &FitOptions::default());
    let ds = full.design_summary.as_ref().unwrap();
    for (l1, l0, pinned) in logliks {
        let got = exact_rlrt(&ds.counts, &ds.s, &ds.c, l1, l0, pinned, &cfg);
        assert_eq!(
            got,
            Err(NoGroupVariation),
            "dummies, ({l1}, {l0}, {pinned})"
        );
    }
}

/// Gate 1, the Dyestuff clause. Dyestuff (`K = 6` batches of `J = 5`) read
/// by hand, `x = 1`, the `(1 | Batch)` fit and the OLS fit made with
/// `reml = Some(true)`, both with `fit_cold`. On a balanced intercept-only
/// design the RLRT is an increasing function of the one-way ANOVA F above
/// `F = 1` (`balanced_null_matches_anova_f`), so the exact p-value is the F
/// test's: `exact_rlrt`'s p at `B = 10⁵` against `P(F_{K−1, n−K} ≥ F_obs)`
/// within [`bands::RLRT_P_SE`] binomial standard errors plus `1/(B + 1)`.
#[test]
fn rlrt_dyestuff_p_matches_anova_f() {
    const B: usize = 100_000;
    let text = csv("data/empirical/Dyestuff.csv");
    let (mut levels, mut ids, mut y) = (Vec::<String>::new(), vec![], vec![]);
    for line in text.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let (batch, value) = line.split_once(',').expect("Batch,Yield");
        let batch = batch.trim().trim_matches('"');
        let a = match levels.iter().position(|l| l == batch) {
            Some(a) => a,
            None => {
                levels.push(batch.to_string());
                levels.len() - 1
            }
        };
        ids.push(a as u32);
        y.push(value.trim().parse::<f64>().expect("a numeric Yield"));
    }
    let (n, k) = (y.len(), levels.len());
    assert_eq!((n, k), (30, 6), "Dyestuff's size");
    let x = vec![1.0; n];
    let full = fit_of(&x, &y, &ids, 1, &FitOptions::default());
    assert!(full.converged() && !is_pinned(&full), "an interior fit");
    let cfg = config(B, 20261015, true);
    let r = exact_of(&full, &ols_reml(&x, &y, 1), &cfg).expect("no refusal");

    let (mut sum, mut rows) = (vec![0.0; k], vec![0.0; k]);
    for (&a, &v) in ids.iter().zip(&y) {
        sum[a as usize] += v;
        rows[a as usize] += 1.0;
    }
    let grand = y.iter().sum::<f64>() / n as f64;
    let means: Vec<f64> = sum.iter().zip(&rows).map(|(s, r)| s / r).collect();
    let between: f64 = means
        .iter()
        .zip(&rows)
        .map(|(m, r)| r * (m - grand).powi(2))
        .sum();
    let within: f64 = ids
        .iter()
        .zip(&y)
        .map(|(&a, v)| (v - means[a as usize]).powi(2))
        .sum();
    let (df1, df2) = ((k - 1) as f64, (n - k) as f64);
    let f_obs = (between / df1) / (within / df2);
    let p_f = FisherF::new(df1, df2).expect("F(K - 1, n - K)").sf(f_obs);
    let bf = B as f64;
    let band = bands::RLRT_P_SE * (p_f * (1.0 - p_f) / bf).sqrt() + 1.0 / (bf + 1.0);
    let gap = (r.p_value - p_f).abs();
    eprintln!(
        "dyestuff: F = {f_obs:.6}, F-test p = {p_f:.6e}, statistic {:.6}, p = {:.6e}, \
         |dp| = {gap:.3e}, band {band:.3e} ({:.0}%)",
        r.statistic,
        r.p_value,
        100.0 * gap / band
    );
    assert!(gap <= band, "p {} against the F test's {p_f}", r.p_value);
}

/// Milliseconds since `t0`.
fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

/// `ZᵀP₀Z = diag(counts) − GᵀG`, `G = L⁻¹s`, `c = LLᵀ`: the matrix
/// [`rlrt_design`] forms before its EVD, by the same steps (the mirror of the
/// lower triangle included, the input checks left out), so the sweep can
/// time the two apart. Every level is non-empty.
fn zp0z(counts: &[f64], s: &[f64], c: &[f64]) -> Mat<f64> {
    let (levels, p) = (counts.len(), c.len().isqrt());
    let chol = crate::linalg::SeqLlt::new(Mat::from_fn(p, p, |i, j| c[j * p + i]).as_ref())
        .expect("c is positive definite");
    let mut g = Mat::<f64>::from_fn(p, levels, |j, a| s[a * p + j]);
    solve_lower_triangular_in_place(chol.L(), g.as_mut(), Par::Seq);
    let mut m = Mat::<f64>::zeros(levels, levels);
    matmul(
        m.as_mut(),
        Accum::Replace,
        g.transpose(),
        g.as_ref(),
        -1.0,
        Par::Seq,
    );
    for b in 0..levels {
        m[(b, b)] += counts[b];
        for a in 0..b {
            m[(a, b)] = m[(b, a)];
        }
    }
    m
}

/// The cost sweep: the design matrix, its EVD, the null's draws and its grid
/// scan priced against one fit, and the full-grid scan against a
/// first-decrease scan. On balanced designs `y ~ 1 + x + (1 | g)` with `J = 3` rows per level and
/// `K = k + 1` levels (the intercept takes one eigenvalue to zero), per `k`:
/// the wall of one `fit_cold` with `satterthwaite: false`, and with it on (the
/// default) for the record; `ZᵀP₀Z` and its EVD apart, and `rlrt_design`
/// whole; `rlrt_null` at `B = 10⁴` (serial) whole, and split into the draws
/// (the same streams), the grid tables and the full-grid scan
/// ([`max_on_grid`]); and on the same draws a scan that stops at the first
/// decrease, with the share of grid points it visits and the draws where it
/// stops strictly below the global maximum (a tie at another index is not
/// counted). The split draws are checked to the bit against the null's. Each part also as a multiple of the first
/// fit's wall. Wall times by `Instant` only, one run each; it asserts no
/// time. Run by hand:
/// `cargo test --release rlrt_cost_sweep -- --ignored --nocapture --test-threads=1`.
#[test]
#[ignore = "cost sweep: prints wall times, run by hand in release"]
fn rlrt_cost_sweep() {
    const B: usize = 10_000;
    const CHUNK: usize = 1024;
    const SEED: u64 = 20261081;
    let no_satt = FitOptions {
        satterthwaite: false,
        ..FitOptions::default()
    };
    println!(
        "| k | K | n | fit ms | fit+satt ms | M ms | EVD ms | design ms | null ms | draws ms \
         | tables ms | grid ms | first-dec ms | first-dec visits | first-dec below | M/fit \
         | EVD/fit | draws/fit | grid/fit | null/fit |"
    );
    println!("|{}", "---|".repeat(20));
    for k in [5, 20, 45, 100, 500, 1000, 2000, 5000] {
        let levels = k + 1;
        let (x, y, ids, p) = design(&vec![3; levels], 1.0, |_, st| vec![1.0, lcg(st)]);
        let n = ids.len();

        let t0 = Instant::now();
        let fit = fit_of(&x, &y, &ids, p, &no_satt);
        let fit_ms = ms_since(t0);
        let t0 = Instant::now();
        let fit_satt = fit_of(&x, &y, &ids, p, &FitOptions::default());
        let fit_satt_ms = ms_since(t0);
        assert!(fit.converged() && fit_satt.converged(), "k {k}: converged");
        let ds = fit.design_summary.as_ref().expect("a design summary");

        let (m_ms, evd_ms) = {
            let t0 = Instant::now();
            let m = zp0z(&ds.counts, &ds.s, &ds.c);
            let m_ms = ms_since(t0);
            let t0 = Instant::now();
            sym_eigen(m.as_ref(), false).expect("EVD of ZᵀP₀Z");
            (m_ms, ms_since(t0))
        };
        let t0 = Instant::now();
        let d = rlrt_design(&ds.counts, &ds.s, &ds.c).expect("a valid design");
        let design_ms = ms_since(t0);
        assert_eq!(d.mu.len(), k, "k {k}: nonzero eigenvalues");

        let t0 = Instant::now();
        let null = rlrt_null(&d, &config(B, SEED, false));
        let null_ms = ms_since(t0);

        let t0 = Instant::now();
        let tables = GridTables::new(&d.mu, &LAMBDA_GRID[..]);
        let tables_ms = ms_since(t0);
        // The first-decrease walk needs the profile point by point, and the
        // tables' accessor is private: the same arithmetic, test-local.
        let (mut fn_, mut fd, mut sl) = (vec![], vec![], vec![]);
        for &lambda in LAMBDA_GRID.iter() {
            let mut log_sum = 0.0;
            for &mu in &d.mu {
                let v = lambda * mu;
                fn_.push(v / (1.0 + v));
                fd.push(1.0 / (1.0 + v));
                log_sum += v.ln_1p();
            }
            sl.push(log_sum);
        }
        let n_minus_p = d.n_minus_p as f64;
        let at = |i: usize, chi1: &[f64], tail: f64| {
            let (mut num, mut den) = (0.0, 0.0);
            for ((&a, &b), &c) in fn_[i * k..(i + 1) * k]
                .iter()
                .zip(&fd[i * k..(i + 1) * k])
                .zip(chi1)
            {
                num += a * c;
                den += b * c;
            }
            n_minus_p * (num / (den + tail)).ln_1p() - sl[i]
        };

        let normal = Normal::new(0.0, 1.0).expect("N(0, 1)");
        let tail = (d.tail_df > 0).then(|| ChiSquared::new(d.tail_df as f64).expect("df > 0"));
        let (mut chi1, mut tails) = (vec![0.0; CHUNK * k], vec![0.0; CHUNK]);
        let (mut draws_ms, mut grid_ms, mut first_ms) = (0.0, 0.0, 0.0);
        let (mut visits, mut below, mut positive) = (0usize, 0usize, vec![]);
        for start in (0..B).step_by(CHUNK) {
            let len = CHUNK.min(B - start);
            let t0 = Instant::now();
            for (i, chunk) in chi1.chunks_mut(k).take(len).enumerate() {
                let mut rng = CommonStatsRng::new_tagged(SEED, (start + i) as u64, STREAM_TAG_RLRT);
                for c in chunk.iter_mut() {
                    let z = normal.sample(&mut rng);
                    *c = z * z;
                }
                tails[i] = tail.as_ref().map_or(0.0, |t| t.sample(&mut rng));
            }
            draws_ms += ms_since(t0);

            let t0 = Instant::now();
            let full: Vec<(f64, usize)> = (0..len)
                .map(|i| max_on_grid(&tables, &chi1[i * k..(i + 1) * k], tails[i], d.n_minus_p))
                .collect();
            grid_ms += ms_since(t0);

            let t0 = Instant::now();
            let first: Vec<(f64, usize)> = (0..len)
                .map(|i| {
                    let draw = &chi1[i * k..(i + 1) * k];
                    let mut best = (at(0, draw, tails[i]), 0);
                    for g in 1..LAMBDA_GRID.len() {
                        let f = at(g, draw, tails[i]);
                        if f < best.0 {
                            break;
                        }
                        best = (f, g);
                    }
                    best
                })
                .collect();
            first_ms += ms_since(t0);

            for (&(t, g), &(tf, gf)) in full.iter().zip(&first) {
                // The walk stops at `gf + 1`, or runs off the grid's end.
                visits += (gf + 2).min(LAMBDA_GRID.len());
                if t > 0.0 {
                    positive.push(t);
                }
                if gf == g {
                    assert_eq!(tf.to_bits(), t.to_bits(), "k {k}: the local arithmetic");
                } else {
                    assert!(tf <= t, "k {k}: the walk above the global maximum");
                    below += usize::from(tf < t);
                }
            }
        }
        positive.sort_unstable_by(f64::total_cmp);
        assert!(
            positive
                .iter()
                .map(|t| t.to_bits())
                .eq(null.draws_pos.iter().map(|t| t.to_bits())),
            "k {k}: the split draws are the null's"
        );

        let visit_share = visits as f64 / (B * LAMBDA_GRID.len()) as f64;
        let of_fit = |v: f64| v / fit_ms;
        println!(
            "| {k} | {levels} | {n} | {fit_ms:.2} | {fit_satt_ms:.2} | {m_ms:.2} | {evd_ms:.2} \
             | {design_ms:.2} | {null_ms:.1} | {draws_ms:.1} | {tables_ms:.2} | {grid_ms:.1} \
             | {first_ms:.1} | {:.1}% | {below} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
            100.0 * visit_share,
            of_fit(m_ms),
            of_fit(evd_ms),
            of_fit(draws_ms),
            of_fit(grid_ms),
            of_fit(null_ms),
        );
    }
}
