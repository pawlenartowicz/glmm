//! OLS estimator tests (`Family::Gaussian`, `re: None`).

use super::ols::{fit_ols, fit_ols_prebuilt, ols_view_to_fit, OlsWorkspace};
use super::*;
use crate::test_support::assert_near;
use crate::{Family, GroupIds, ModelSpec};

/// A small fixed OLS dataset (n=20, p=3: intercept + two predictors) used by the
/// workspace-reuse gate. Deterministic, no RNG.
fn ols_hand_dataset() -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n = 20;
    let p = 3;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64;
        let b = ((i * 7) % 11) as f64 - 5.0;
        x.extend_from_slice(&[1.0, a, b]);
        y.push(0.5 + 1.3 * a - 0.7 * b + ((i % 3) as f64 - 1.0));
    }
    (x, y, n, p)
}

/// A reused `OlsWorkspace` must give a near-identical `Fit` to a throwaway one,
/// and a second fit on the SAME ws must match the first — guards stale-buffer
/// leakage the goldens (which only fit throwaway workspaces) cannot see.
#[test]
fn fit_ols_prebuilt_reused_ws_near_identical_to_fresh() {
    let (x, y, n, p) = ols_hand_dataset();
    let opts = FitOptions {
        target_indices: vec![1, 2],
        ..FitOptions::default()
    };

    let fresh = fit_ols(&x, &y, n, p, &opts);

    let mut ws = OlsWorkspace::new(n, p, opts.target_indices.len(), opts.weights.is_some());
    let x_mat = super::common::to_col_major(&x, n, p);
    let reused = {
        let v = fit_ols_prebuilt(&mut ws, x_mat.as_ref().subrows(0, n), &y, n, p, &opts);
        ols_view_to_fit(&v, &x, &y, n, p, &opts)
    };
    let reused2 = {
        let v = fit_ols_prebuilt(&mut ws, x_mat.as_ref().subrows(0, n), &y, n, p, &opts);
        ols_view_to_fit(&v, &x, &y, n, p, &opts)
    };

    assert_near(&fresh.beta, &reused.beta, "beta reused vs fresh");
    assert_near(&fresh.se, &reused.se, "se reused vs fresh");
    assert_near(&[fresh.dispersion], &[reused.dispersion], "dispersion");
    assert_near(&reused.beta, &reused2.beta, "beta second reuse");
}

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
    // OLS reports deviance: NaN, singular: false unconditionally (fit/ols.rs).
    assert!(f.deviance.is_nan());
    assert!(!f.singular);
    assert!(f.tau2.is_empty());
    // y = 2*i is an exact fit (no noise) → RSS/(n-p) ≈ 0, not the GLM φ≡1 convention.
    assert!(f.dispersion >= 0.0 && f.dispersion < 1e-9);
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
    // vcov(f)[1,2] from the same R run — the only external pin of a vcov
    // off-diagonal at the fit surface (se checks only see the diagonal).
    const REF_VCOV_01: f64 = -0.0002002132676736411;
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
    for (i, j) in [(0, 1), (1, 0)] {
        assert!(
            (f.vcov[i][j] - REF_VCOV_01).abs() < 1e-12,
            "vcov[{i}][{j}] = {}, R = {REF_VCOV_01}",
            f.vcov[i][j]
        );
    }
    // logLik(f) / attr(logLik(f), "df") from the same R run — pins the weighted
    // ML Gaussian log-likelihood (½(Σlog wᵢ − n(ln 2π + 1 − ln n + ln Σwᵢrᵢ²)))
    // and the p+σ² parameter count.
    const REF_LOGLIK: f64 = 11.7602761183173;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-9,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 3);
    assert!(!f.reml);
    // fitted = Xβ̂ on the RAW rows (weights are a solver device, not the mean).
    assert_eq!(f.fitted.len(), n);
    for (i, &xi) in xv.iter().enumerate() {
        let eta = f.beta[0] + f.beta[1] * xi;
        assert!((f.fitted[i] - eta).abs() < 1e-12, "fitted[{i}]");
    }
}

/// Weighted OLS with a per-row offset, vs R `lm(weights=, offset=)`.
/// The identity-link offset is the exact `y − o` shift; `fitted` reports
/// `o + Xβ̂` on the original scale. Oracle (R 4.5.3, same xv/w/y as
/// `fit_ols_weighted_matches_r_lm`, `o <- 0.3 * ((seq_along(y) - 1) %% 4)`):
///   f <- lm(y ~ xv, weights = w, offset = o)
///   print(coef(f), digits = 15); print(logLik(f), digits = 15)
#[test]
fn fit_ols_offset_matches_r_lm() {
    let xv = [
        0.2, 1.4, -0.8, 2.1, 0.5, -1.3, 1.9, 0.0, -0.6, 1.1, 2.4, -1.7,
    ];
    let w = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 3.0, 2.0];
    let y = vec![
        0.8, 3.1, -1.2, 4.6, 1.4, -2.0, 4.1, 0.3, -0.9, 2.6, 5.2, -3.1,
    ];
    let n = 12;
    let o: Vec<f64> = (0..n).map(|i| 0.3 * (i % 4) as f64).collect();
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &xv {
        x.extend_from_slice(&[1.0, xi]);
    }
    const REF_BETA: [f64; 2] = [-0.159548531835057, 1.964134686017193];
    const REF_LOGLIK: f64 = -5.65572764128147;
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        offset: Some(o.clone()),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &opts);
    assert!(f.converged);
    for (j, (&b, &r)) in f.beta.iter().zip(&REF_BETA).enumerate() {
        assert!((b - r).abs() < 1e-9, "beta[{j}] {b} vs R {r}");
    }
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-9,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    // fitted on the ORIGINAL y scale: o + Xβ̂.
    for (i, &xi) in xv.iter().enumerate() {
        let eta = o[i] + f.beta[0] + f.beta[1] * xi;
        assert!((f.fitted[i] - eta).abs() < 1e-12, "fitted[{i}]");
    }
    // A zero offset must be BIT-identical to no offset (the None path is the
    // same code with an all-zeros shift).
    let f_none = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &GroupIds::default(),
        &FitOptions {
            offset: None,
            ..opts.clone()
        },
    );
    let f_zero = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &GroupIds::default(),
        &FitOptions {
            offset: Some(vec![0.0; n]),
            ..opts
        },
    );
    assert_eq!(f_none.beta, f_zero.beta);
    assert_eq!(f_none.loglik.to_bits(), f_zero.loglik.to_bits());
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
