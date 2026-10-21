//! OLS estimator tests (`Family::Gaussian`, `re: None`).

use super::common_tests::weighted_collinear_ols_fixture;
use super::ols::{fit_ols_prebuilt, OlsWorkspace};
use super::*;
use crate::{BinomialLink, Family, GroupIds, ModelSpec};

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
    assert!(f.converged());
    assert!((f.beta[1] - 2.0).abs() < 1e-6);
    // OLS reports deviance: NaN, singular: false unconditionally (fit/ols.rs).
    assert!(f.deviance.is_nan());
    assert!(!f.singular());
    assert!(f.tau2.is_empty());
    // y = 2*i is an exact fit (no noise) → RSS/(n-p) ≈ 0, not the GLM φ≡1 convention.
    assert!(f.dispersion.abs() < 1e-9);
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
    assert!(f.converged());
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
    assert!(f.converged());
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

/// Weighted-collinear OLS: full rank in the RAW design, near-singular once
/// weighted. The fit comes back, the pivot records that it is barely identified,
/// and the standard error says so out loud.
///
/// This class is invisible to every check upstream of the solve. The
/// pre-dispatch alias gate (`detect_aliased`) tests the UNWEIGHTED `X'X`, where
/// column 2 has a healthy pivot ratio — the two predictor columns differ by a
/// full unit on the last third of the rows. But that third carries weight
/// `WSMALL` (1e-11, `weighted_collinear_ols_fixture`), so in `X'WX` — the
/// matrix β̂ and its SEs actually come from — the columns are very nearly the
/// same column, and the pivot ratio falls below 1e-12.
///
/// The `min|L_ii| / max|L_ii|` statistic conflates collinearity with column
/// scale and never fires here (measured 4.8e-8, four orders above its own
/// 1e-12 threshold), so the route does not guard on it. It is not fixed by refusing on the
/// right statistic either: the 2026-07-31 1-ULP sweep showed this route's
/// standard errors stay stable to 1.3e-13 relative and never understate the
/// error, all the way past total loss of β̂. So the deliverable is the
/// combination asserted below — the estimates are returned, and the SE beside
/// them is orders larger than the coefficient, which is what tells the caller
/// the number is worthless.
#[test]
fn fit_ols_weighted_collinear_fits_with_an_honest_se() {
    let (x, y, w, n, p) = weighted_collinear_ols_fixture();
    let opts = FitOptions {
        target_indices: vec![0, 1, 2],
        weights: Some(w),
        ..FitOptions::default()
    };
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let ids = GroupIds {
        primary: vec![],
        extra: vec![],
    };

    // The raw design really is full rank: the alias gate passes it through, so
    // nothing before the solve can see the problem.
    assert!(
        !super::common::detect_aliased(&x, n, p).iter().any(|&a| a),
        "the RAW design is full rank — this test is only meaningful if the \
         alias gate lets it through"
    );

    // The recorded pivot is on the WEIGHTED Gram and must be in the flagging
    // band; the raw Gram's own pivot is orders above it, which is the whole
    // point of measuring the matrix the fit actually used.
    let mut ws = OlsWorkspace::new(n, p, opts.target_indices.len(), true);
    let x_mat = super::common::to_col_major(&x, n, p);
    let view = fit_ols_prebuilt(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
        &y,
        n,
        p,
        &opts,
        crate::standardize::ColumnScaling::IDENTITY,
    );
    assert!(
        view.converged,
        "the fit is computable and must be returned, not refused"
    );
    assert!(
        view.pivot < crate::ols::PIVOT_MIN,
        "the weighted pivot must land in the ill-conditioned band, got {}",
        view.pivot
    );

    let f = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "the fit is computable and must be returned");
    assert_eq!(
        f.aliased(),
        vec![false; p],
        "nothing is redundant in the raw design, so no column is dropped"
    );
    // The honest report: the two entangled coefficients carry standard errors
    // orders larger than themselves. A caller reading `se` cannot mistake these
    // for estimates.
    for j in [1usize, 2] {
        assert!(
            f.se[j] > 100.0 * f.beta[j].abs(),
            "β[{j}] = {} must carry an SE orders above it, got {}",
            f.beta[j],
            f.se[j]
        );
    }
}

/// Fixed-only (`re: None`) counterpart to `fit_glmm_degenerate_width_never_panics`
/// (`src/fit/common_tests.rs`): that test covers the mixed route's own `p == 0`
/// short-circuit, but `re: None` takes a completely separate one in
/// `fit_ols_prebuilt` and `fit_glm`, so a regression there is invisible to it. Covers `p == 0`,
/// `n == p`, `n == 0`, and an all-zero column, for the OLS route
/// (`Family::Gaussian`) and one GLM route (`Family::Binomial { link: Logit }`).
#[test]
fn fit_fixed_degenerate_width_never_panics() {
    let families = [
        Family::Gaussian,
        Family::Binomial {
            link: BinomialLink::Logit,
        },
    ];
    for family in families {
        let model = ModelSpec { family, re: None };
        let ids = GroupIds::default();

        // p == 0.
        let n = 4;
        let fit = fit_cold(
            &[],
            &vec![1.0f64; n],
            n,
            0,
            &model,
            &ids,
            &FitOptions::default(),
        );
        assert!(
            !fit.converged(),
            "{family:?} p=0: degenerate fit does not report converged"
        );

        // n == 0, p == 1.
        let fit = fit_cold(&[], &[], 0, 1, &model, &ids, &FitOptions::default());
        assert!(
            !fit.converged(),
            "{family:?} n=0: degenerate fit does not report converged"
        );

        // n == p: four independent columns (lower-triangular ones) on four
        // rows. The `n <= p` guard fires before rank is ever examined, so
        // even this full-rank design comes back NaN rather than fitted.
        let (n, p) = (4, 4);
        let x: Vec<f64> = (0..n * p)
            .map(|t| if t % p <= t / p { 1.0 } else { 0.0 })
            .collect();
        let fit = fit_cold(
            &x,
            &vec![1.0f64; n],
            n,
            p,
            &model,
            &ids,
            &FitOptions::default(),
        );
        assert!(
            !fit.converged() && fit.beta.iter().all(|b| b.is_nan()),
            "{family:?} n == p: degenerate fit is NaN and does not report converged"
        );

        // p == 1 with the one fixed column all zero. A mixed 0/1 response
        // (rather than a constant one) keeps binomial off its OWN all-0/all-1
        // short-circuit, so this exercises the shared zero-column path (X'WX
        // singular, Cholesky refuses) instead.
        let n = 4;
        let x = vec![0.0f64; n];
        let y = vec![1.0, 0.0, 1.0, 0.0];
        let fit = fit_cold(&x, &y, n, 1, &model, &ids, &FitOptions::default());
        assert!(
            !fit.converged() && fit.beta.iter().all(|b| b.is_nan()),
            "{family:?} all-zero column: degenerate fit does not report converged"
        );
    }
}

/// OLS on a nearly collinear design keeps its digits: on
/// `entangled_dyadic_designs` the entangled coefficient and its SE equal the
/// well-conditioned fit's mapped through the exact `1/δ`. A solve from the
/// Gram of `X` squares `cond(X) ≈ 2e6` and misses by about 1e-4; the
/// `DesignQr` route in `fit_ols_prebuilt` is accurate to round-off times
/// `cond(X)`.
#[test]
fn fit_ols_entangled_columns_keep_their_digits() {
    let n = 200;
    let (x, xr, delta) = super::common_tests::entangled_dyadic_designs(n);
    let mut st = 5u64;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            0.5 + 0.3 * xr[3 * i + 1] + 0.2 * xr[3 * i + 2] + super::common_tests::lcg(&mut st)
        })
        .collect();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1, 2],
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, 3, &model, &GroupIds::default(), &opts);
    let r = fit_cold(&xr, &y, n, 3, &model, &GroupIds::default(), &opts);
    assert!(f.converged() && r.converged());
    assert!(!f.aliased().iter().any(|&a| a), "nothing is dropped");
    let (beta, se) = (r.beta[2] / delta, r.se[2] / delta);
    assert!(
        (f.beta[2] - beta).abs() <= 1e-9 * beta.abs(),
        "β₂ {} against {beta}",
        f.beta[2]
    );
    assert!(
        (f.se[2] - se).abs() <= 1e-9 * se,
        "se₂ {} against {se}",
        f.se[2]
    );
}

/// `y ~ 1 + x1 + x2` on 30 rows, with optional weights and an optional
/// fourth column `2·x1` (aliased). Returns `(x, y, weights, n, p)`.
fn residual_df_data(dup: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
    let mut s = 17u64;
    let n = 30;
    let (mut x, mut y, mut w) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let x1 = super::common_tests::lcg(&mut s);
        let x2 = 2.0 * super::common_tests::lcg(&mut s);
        x.extend([1.0, x1, x2]);
        if dup {
            x.push(2.0 * x1);
        }
        y.push(0.4 + 1.1 * x1 - 0.6 * x2 + 0.5 * super::common_tests::lcg(&mut s));
        w.push(1.5 + super::common_tests::lcg(&mut s));
    }
    (x, y, w, n, if dup { 4 } else { 3 })
}

/// OLS reports `coef_df = n − p` on the target columns (raw rows, `p` the
/// retained columns) and its t tests take that df, weighted or not and
/// whatever `FitOptions::satterthwaite` says. The contrasts read the full
/// `C`, so a non-target column is still testable.
#[test]
fn ols_coef_df_is_n_minus_p() {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let all = |p: usize| FitOptions {
        target_indices: (0..p as u32).collect(),
        ..FitOptions::default()
    };
    let (x, y, w, n, p) = residual_df_data(false);
    let nu = (n - p) as f64;
    let weighted = FitOptions {
        weights: Some(w.clone()),
        ..all(p)
    };
    let switched_off = FitOptions {
        satterthwaite: false,
        ..all(p)
    };
    // ν is the residual df on either criterion: `reml` moves OLS's
    // log-likelihood, never its df.
    let criterion = |reml| FitOptions { reml, ..all(p) };
    for opts in [
        all(p),
        weighted,
        switched_off,
        criterion(Some(false)),
        criterion(Some(true)),
    ] {
        let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts);
        assert!(f.converged());
        assert_eq!(f.coef_df, vec![nu; p]);
        for j in 0..p {
            let mut e = vec![0.0; p];
            e[j] = 1.0;
            let t = f.contrast_t(&e).expect("OLS has a residual df");
            assert_eq!(t.df, nu);
            assert_eq!(t.estimate, f.beta[j]);
            assert!((t.se - f.se[j]).abs() <= 1e-12 * f.se[j], "se {j}");
            assert_eq!(t.p_value, crate::inference::tails::t_two_sided_p(t.t, nu));
            assert!(t.notes.is_empty());
        }
    }

    // One target: coef_df on it only, the contrast on another column from
    // the full C.
    let full = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &all(p));
    let one = FitOptions {
        target_indices: vec![1],
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &one);
    assert!(f.coef_df[0].is_nan() && f.coef_df[2].is_nan());
    assert_eq!(f.coef_df[1], nu);
    let t = f.contrast_t(&[1.0, 0.0, 0.0]).expect("full C");
    assert!((t.se - full.se[0]).abs() <= 1e-12 * full.se[0]);
    assert_eq!(t.df, nu);

    // Rank-deficient: the aliased column has no df and `p` is the retained
    // count, as in σ̂².
    let (x, y, _, n, p) = residual_df_data(true);
    let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &all(p));
    assert!(f.converged());
    assert_eq!(f.diagnostics.aliased, vec![false, false, false, true]);
    let nu = (n - 3) as f64;
    assert_eq!(&f.coef_df[..3], &[nu; 3]);
    assert!(f.coef_df[3].is_nan());
    let t = f
        .contrast_t(&[0.0, 1.0, 0.0, 0.0])
        .expect("retained column");
    assert_eq!(t.df, nu);
    let t = f.contrast_t(&[0.0, 0.0, 0.0, 1.0]).expect("aliased column");
    assert!(t.df.is_nan());
    assert_eq!(
        t.notes,
        vec![crate::inference::contrast::ContrastNote::AliasedColumn { column: 3 }]
    );

    // n = p does not converge: no coef_df (as se), but the residual record
    // stays, so the contrasts are NaN numbers on ν = 0, never `NoDf`.
    let x = [1.0, 0.3, -1.2, 1.0, -0.7, 0.4, 1.0, 1.1, 0.9];
    let y = [0.5, -0.2, 1.3];
    let f = fit_cold(&x, &y, 3, 3, &model, &GroupIds::default(), &all(3));
    assert!(!f.converged());
    assert!(f.coef_df.iter().all(|d| d.is_nan()));
    let t = f
        .contrast_t(&[0.0, 1.0, 0.0])
        .expect("the residual record, not NoDf");
    assert!(t.estimate.is_nan() && t.se.is_nan() && t.t.is_nan() && t.p_value.is_nan());
    assert_eq!(t.df, 0.0);
    let fs = f
        .contrast_f(&[0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 2)
        .expect("the residual record, not NoDf");
    assert!(fs.f.is_nan() && fs.p_value.is_nan());
}

// --- `reml = Some(true)`: the REML log-likelihood ---

/// The closed-form REML deviance `ln|XᵀWX| + (n − p)·ln(RSS_w/(n − p)) − Σ ln wᵢ`
/// on the caller's X, from the normal equations solved here with a plain
/// Cholesky (no standardization, no fit). `x` is row-major `n × p`.
fn closed_form_reml_deviance(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    w: Option<&[f64]>,
    o: Option<&[f64]>,
) -> f64 {
    let wt = |i: usize| w.map_or(1.0, |w| w[i]);
    let z = |i: usize| y[i] - o.map_or(0.0, |o| o[i]);
    let (mut a, mut b) = (vec![0.0; p * p], vec![0.0; p]);
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        for j in 0..p {
            b[j] += wt(i) * row[j] * z(i);
            for k in 0..p {
                a[j * p + k] += wt(i) * row[j] * row[k];
            }
        }
    }
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        for i in j..p {
            let s = a[i * p + j] - (0..j).map(|k| l[i * p + k] * l[j * p + k]).sum::<f64>();
            l[i * p + j] = if i == j { s.sqrt() } else { s / l[j * p + j] };
        }
    }
    for i in 0..p {
        b[i] = (b[i] - (0..i).map(|k| l[i * p + k] * b[k]).sum::<f64>()) / l[i * p + i];
    }
    for i in (0..p).rev() {
        b[i] = (b[i] - (i + 1..p).map(|k| l[k * p + i] * b[k]).sum::<f64>()) / l[i * p + i];
    }
    let rss: f64 = (0..n)
        .map(|i| {
            let r = z(i) - (0..p).map(|j| x[i * p + j] * b[j]).sum::<f64>();
            wt(i) * r * r
        })
        .sum();
    let nu = (n - p) as f64;
    let log_det: f64 = (0..p).map(|j| 2.0 * l[j * p + j].ln()).sum();
    log_det + nu * (rss / nu).ln() - (0..n).map(|i| wt(i).ln()).sum::<f64>()
}

/// `−½·(d + (n − p)(1 + ln 2π))`: a REML deviance on the `logLik()` scale.
fn reml_loglik(d: f64, n: usize, p: usize) -> f64 {
    -0.5 * (d + (n - p) as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln()))
}

/// `reml = Some(true)` on OLS moves only the log-likelihood and the flag: β̂,
/// the SEs, `vcov` and `σ̂² = RSS/(n − p)` are the `None` fit's bit for bit,
/// `Some(false)` gives the `None` fit bit for bit, `reml` is `true`,
/// `deviance` stays NaN, and `loglik` is R's `logLik(lm, REML = TRUE)`, the
/// closed form computed here. Plain, weighted and with an offset, on a
/// standardized design (so the standardization shift is exercised); then once
/// on the `DesignQr` route.
#[test]
fn ols_reml_reports_the_closed_form_reml_loglik() {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let bits = |v: &[f64]| v.iter().map(|t| t.to_bits()).collect::<Vec<_>>();
    let (x, y, w, n, p) = residual_df_data(false);
    let o: Vec<f64> = (0..n).map(|i| 0.3 * (i % 4) as f64).collect();
    let base = FitOptions {
        target_indices: (0..p as u32).collect(),
        ..FitOptions::default()
    };
    let variants = [
        ("plain", base.clone()),
        (
            "weights",
            FitOptions {
                weights: Some(w),
                ..base.clone()
            },
        ),
        (
            "offset",
            FitOptions {
                offset: Some(o),
                ..base
            },
        ),
    ];
    for (tag, opts) in variants {
        let ml = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts);
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                reml: Some(true),
                ..opts.clone()
            },
        );
        assert!(ml.converged() && f.converged(), "{tag}");
        // `Some(false)` is the `None` path: every field bit for bit.
        let ml_false = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                reml: Some(false),
                ..opts.clone()
            },
        );
        super::core::core_tests::assert_fits_bitwise_equal(
            &ml_false,
            &ml,
            &format!("{tag}: reml = Some(false)"),
        );
        assert_eq!(bits(&f.beta), bits(&ml.beta), "{tag}: beta");
        assert_eq!(bits(&f.se), bits(&ml.se), "{tag}: se");
        assert_eq!(
            bits(&f.vcov.concat()),
            bits(&ml.vcov.concat()),
            "{tag}: vcov"
        );
        assert_eq!(f.dispersion.to_bits(), ml.dispersion.to_bits(), "{tag}: σ̂²");
        assert!(f.reml && !ml.reml, "{tag}: reml");
        assert!(f.deviance.is_nan(), "{tag}: deviance");
        let d = closed_form_reml_deviance(
            &x,
            &y,
            n,
            p,
            opts.weights.as_deref(),
            opts.offset.as_deref(),
        );
        let want = reml_loglik(d, n, p);
        assert!(
            (f.loglik - want).abs() <= 1e-10 * want.abs(),
            "{tag}: loglik {} vs the closed form {want}",
            f.loglik
        );
    }

    // The `DesignQr` route (`fit_ols_entangled_columns_keep_their_digits`):
    // `X = Xr·T` with `det T = δ`, so `ln|XᵀX|` is `ln|XrᵀXr| + 2 ln δ`, the
    // RSS is the same, and the REML log-likelihood moves by exactly `−ln δ`.
    let n = 200;
    let (x, xr, delta) = super::common_tests::entangled_dyadic_designs(n);
    let mut st = 5u64;
    let y: Vec<f64> = (0..n)
        .map(|i| {
            0.5 + 0.3 * xr[3 * i + 1] + 0.2 * xr[3 * i + 2] + super::common_tests::lcg(&mut st)
        })
        .collect();
    let opts = FitOptions {
        target_indices: vec![0, 1, 2],
        reml: Some(true),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, 3, &model, &GroupIds::default(), &opts);
    let r = fit_cold(&xr, &y, n, 3, &model, &GroupIds::default(), &opts);
    let want = r.loglik - delta.ln();
    assert!(
        (f.loglik - want).abs() <= 1e-10 * want.abs(),
        "DesignQr route: loglik {} vs {want}",
        f.loglik
    );
}

/// An aliased OLS design under `Some(true)` reports `reml` true and the
/// reduced fit's REML log-likelihood (`fit_rank_deficient` passes both through).
#[test]
fn ols_reml_survives_an_aliased_column() {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = |p: usize| FitOptions {
        target_indices: (0..p as u32).collect(),
        reml: Some(true),
        ..FitOptions::default()
    };
    let (x, y, _, n, p) = residual_df_data(true);
    let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts(p));
    assert_eq!(f.diagnostics.aliased, vec![false, false, false, true]);
    assert!(f.reml);
    let (xr, yr, _, nr, pr) = residual_df_data(false);
    let r = fit_cold(&xr, &yr, nr, pr, &model, &GroupIds::default(), &opts(pr));
    assert!(r.loglik.is_finite());
    assert_eq!(f.loglik.to_bits(), r.loglik.to_bits());
}

/// The scale test: the OLS fit's REML log-likelihood, put on the deviance
/// scale by the inverse of `fit::common::lmm_loglik`, equals the deviance of
/// the mixed fit `y ~ 1 + (1 | Batch)` pinned at zero variance, to round-off.
/// Dyestuff2 (lme4's data), where the REML fit is singular. Three variants:
/// plain; prior weights `1 + 0.5·k`; an offset `0.4·(k − 2)`, with `k` the row's
/// place in its batch (0 to 4). lme4 pins every variant too (R 2026-10-12,
/// lme4 2.0.6, `getME(m, "theta")` 0). The fixed part is the intercept
/// alone, so the standardization shift is 0 here;
/// `ols_reml_reports_the_closed_form_reml_loglik` is the test that catches a
/// missing shift.
#[test]
fn ols_reml_loglik_is_on_the_lmm_deviance_scale() {
    // Band: an error bound, not the measured gap. Measured 2026-10-12 on
    // validation/data/empirical/Dyestuff2.csv, relative to the deviance (about
    // 80): 3.6e-16 plain, 3.5e-16 weights, 5.5e-16 offset (2 to 3 ulps), the
    // same bits on x86_64 Linux and on wasm32-wasip1 (plain and simd128). The
    // round-off of the two routes (the LMM tail's cancellation and the OLS
    // route's log-determinant and `ln`) bounds the gap at about 1.4e-15
    // relative; the band is that bound, rounded up.
    const BAND: f64 = 2e-15;
    let csv = include_str!("../../validation/data/empirical/Dyestuff2.csv");
    let (mut y, mut g, mut k) = (vec![], vec![], vec![]);
    let mut seen = [0u32; 6];
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let (batch, yield_) = line.split_once(',').unwrap();
        let b = u32::from(batch.trim_matches('"').as_bytes()[0] - b'A');
        g.push(b);
        k.push(f64::from(seen[b as usize]));
        seen[b as usize] += 1;
        y.push(yield_.parse::<f64>().unwrap());
    }
    let (n, p) = (y.len(), 1);
    assert_eq!((n, seen), (30, [5; 6]));
    let x = vec![1.0; n];
    let mixed = ModelSpec {
        family: Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing: crate::Sizing::FixedClusters { n_clusters: 6 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ols = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let ids = GroupIds {
        primary: g,
        extra: vec![],
    };
    let base = FitOptions {
        target_indices: vec![0],
        ..FitOptions::default()
    };
    let variants = [
        ("plain", base.clone()),
        (
            "weights",
            FitOptions {
                weights: Some(k.iter().map(|k| 1.0 + 0.5 * k).collect()),
                ..base.clone()
            },
        ),
        (
            "offset",
            FitOptions {
                offset: Some(k.iter().map(|k| 0.4 * (k - 2.0)).collect()),
                ..base
            },
        ),
    ];
    for (tag, opts) in variants {
        let m = fit_cold(&x, &y, n, p, &mixed, &ids, &opts);
        assert!(m.converged(), "{tag}: the mixed fit converges");
        assert_eq!(
            m.tau2,
            [0.0],
            "{tag}: the mixed fit is pinned at zero variance"
        );
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &ols,
            &GroupIds::default(),
            &FitOptions {
                reml: Some(true),
                ..opts
            },
        );
        assert!(f.converged() && f.reml, "{tag}: the OLS REML fit");
        let closed = -2.0 * f.loglik - (n - p) as f64 * (1.0 + (2.0 * std::f64::consts::PI).ln());
        let gap = (closed - m.deviance).abs() / m.deviance.abs();
        assert!(
            gap <= BAND,
            "{tag}: closed form {closed} vs the pinned fit's deviance {} (relative gap {gap:e})",
            m.deviance
        );
    }
}

/// `nobs` is the row count on every OLS fit, a non-converged one included, and
/// `objective` is least squares whatever `reml` asks for: an OLS fit made with
/// `reml = Some(true)` reports the REML `loglik` (`Fit::reml`) under the same
/// label. A weighted fit counts its rows, not the sum of its weights.
#[test]
fn ols_fit_reports_nobs_and_least_squares() {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let (x, y, w, n, p) = residual_df_data(false);
    for (reml, weights) in [(None, None), (Some(true), Some(w))] {
        let opts = FitOptions {
            target_indices: (0..p as u32).collect(),
            reml,
            weights,
            ..FitOptions::default()
        };
        let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts);
        assert!(f.converged(), "reml = {reml:?}");
        assert_eq!(
            (f.nobs, f.objective),
            (n, Objective::LeastSquares),
            "reml = {reml:?}"
        );
    }
    // `n == p`: refused before any solve.
    let (n, p) = (4, 4);
    let x: Vec<f64> = (0..n * p)
        .map(|t| if t % p <= t / p { 1.0 } else { 0.0 })
        .collect();
    let f = fit_cold(
        &x,
        &[1.0; 4],
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions::default(),
    );
    assert!(!f.converged());
    assert_eq!((f.nobs, f.objective), (n, Objective::LeastSquares));
}
