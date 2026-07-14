//! OLS estimator tests (`Family::Gaussian`, `re: None`).

use super::*;
use crate::{Family, GroupIds, ModelSpec};

#[test]
fn fit_ols_recovers_slope() {
    // y = 2*x + noise-free → beta[1] ≈ 2
    let n = 20;
    let p = 2;
    let x: Vec<f64> = (0..n).flat_map(|i| [1.0, i as f64]).collect(); // [intercept, x]
    let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![1],
            ..FitOptions::default()
        },
    );
    assert!(f.converged);
    assert!((f.beta[1] - 2.0).abs() < 1e-6);
}

/// WLS through the stable surface, gated against R `lm(weights=)`.
/// Convention: σ̂² = Σwᵢrᵢ²/(n−p) with raw-row-count df (R's summary.lm).
#[test]
fn fit_ols_weighted_matches_r_lm() {
    // R 4.5.3 oracle, data as in the vectors below:
    //   f <- lm(y ~ x, weights = w); print(coef(summary(f)), digits = 15)
    // REF_BETA/REF_SE are the Estimate / Std. Error columns.
    let xv = [
        0.2, 1.4, -0.8, 2.1, 0.5, -1.3, 1.9, 0.0, -0.6, 1.1, 2.4, -1.7,
    ];
    let w = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 3.0, 2.0];
    let y = vec![
        0.8, 3.1, -1.2, 4.6, 1.4, -2.0, 4.1, 0.3, -0.9, 2.6, 5.2, -3.1,
    ];
    let n = 12;
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &xv {
        x.extend_from_slice(&[1.0, xi]);
    }
    const REF_BETA: [f64; 2] = [0.371528122456273, 1.996237765292144];
    const REF_SE: [f64; 2] = [0.0289002251717619, 0.0195893362923423];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &opts);
    assert!(f.converged);
    for j in 0..2 {
        assert!((f.beta[j] - REF_BETA[j]).abs() < 1e-9, "beta[{j}]");
        assert!((f.se[j] - REF_SE[j]).abs() < 1e-9, "se[{j}]");
    }
}

/// Constant weights w≡c must reproduce the unweighted fit exactly:
/// β̂ is scale-invariant and σ̂²(X'WX)⁻¹ cancels the c.
#[test]
fn fit_ols_constant_weights_invariant() {
    let xv = [0.2, 1.4, -0.8, 2.1, 0.5, -1.3, 1.9, 0.0];
    // A tiny perturbation on one point keeps this off the exact-fit (RSS≈0)
    // edge, where closed-form RSS = y'y − β̂'X'y catastrophically cancels
    // and the sign of the residual float noise (not weighting) decides
    // whether `var_diag` clears its `>= 0` finite guard.
    let y: Vec<f64> = xv
        .iter()
        .enumerate()
        .map(|(i, v)| 1.0 + 2.0 * v + if i == 0 { 0.01 } else { 0.0 })
        .collect();
    let n = 8;
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &xv {
        x.extend_from_slice(&[1.0, xi]);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let base = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let f0 = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &base);
    let opts = FitOptions {
        weights: Some(vec![3.0; n]),
        ..base
    };
    let f1 = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &opts);
    for j in 0..2 {
        assert!((f0.beta[j] - f1.beta[j]).abs() < 1e-12);
        assert!((f0.se[j] - f1.se[j]).abs() < 1e-12);
    }
}
