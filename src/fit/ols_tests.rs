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
/// full unit on the last third of the rows. But that third carries weight 1e-14,
/// so in `X'WX` — the matrix β̂ and its SEs actually come from — the columns are
/// very nearly the same column, and the pivot ratio falls below 1e-12.
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
    for opts in [all(p), weighted, switched_off] {
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
