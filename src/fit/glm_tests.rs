//! GLM estimator tests (fixed-effects binomial/Poisson/Gamma/negative-binomial,
//! `re: None`).

use super::glm::{fit_glm, fit_glm_prebuilt, glm_view_to_fit, GlmScratchBuf};
use super::*;
use crate::test_support::assert_near;
use crate::{BinomialLink, Family, GroupIds, ModelSpec};

use super::common_tests::{lcg, sim_clustered};

/// A small non-separable binomial(logit) dataset (n=30, p=2) for the view-mapper
/// equivalence gate. Deterministic, no RNG.
fn glm_logit_hand_dataset() -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n = 30;
    let p = 2;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) / 10.0 - 1.5;
        x.extend_from_slice(&[1.0, xi]);
        y.push(if (i * 3 + 1) % 5 < 3 { 1.0 } else { 0.0 });
    }
    (x, y, n, p)
}

/// `fit_glm_prebuilt` + `glm_view_to_fit` must reproduce the `Fit` that the
/// throwaway `fit_glm` path produces — pins the view/assembly split as
/// behavior-preserving for a non-Gaussian family.
#[test]
fn fit_glm_prebuilt_view_maps_to_same_fit() {
    let (x, y, n, p) = glm_logit_hand_dataset();
    let opts = FitOptions {
        target_indices: vec![1],
        ..FitOptions::default()
    };
    let family = Family::Binomial {
        link: BinomialLink::Logit,
    };

    let direct = fit_glm(family, f64::NAN, &x, &y, n, p, &opts);

    let mut buf = GlmScratchBuf::new(n, p, opts.target_indices.len());
    let x_mat = super::common::to_col_major(&x, n, p);
    let via_view = {
        let v = fit_glm_prebuilt(
            family,
            f64::NAN,
            x_mat.as_ref().subrows(0, n),
            &y,
            &opts,
            &mut buf,
        );
        glm_view_to_fit(&v, &y, family, f64::NAN, n, p, &opts)
    };
    assert!(direct.converged() && via_view.converged());
    assert_near(&direct.beta, &via_view.beta, "beta");
    assert_near(&direct.se, &via_view.se, "se");
    assert_near(&[direct.loglik], &[via_view.loglik], "loglik");
}

/// Weighted Gamma(log) GLM vs R glm(weights=). Convention: prior weight
/// multiplies the IRLS working weight and deviance; Pearson dispersion
/// φ = Σwᵢrᵢ²/(n−p), raw-row df (R summary.glm).
#[test]
fn fit_glm_gamma_weighted_matches_r() {
    // R 4.5.3 oracle (set.seed(42), n = 40):
    //   x1 <- round(rnorm(n), 4); w <- sample(1:4, n, replace = TRUE)
    //   eta <- 0.4 + 0.8 * x1
    //   yg <- round(rgamma(n, shape = 2, scale = exp(eta) / 2), 6)
    //   fg <- glm(yg ~ x1, family = Gamma("log"), weights = w)
    //   print(coef(summary(fg)), digits = 15); print(summary(fg)$dispersion, digits = 15)
    let x1: [f64; 40] = [
        1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627, 1.3049,
        2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201, -0.3066,
        -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64, 0.4555, 0.7048,
        1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
    ];
    let w: Vec<f64> = vec![
        4.0, 1.0, 2.0, 1.0, 1.0, 4.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 1.0, 4.0, 4.0, 2.0, 1.0, 4.0,
        2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 4.0, 3.0, 4.0, 1.0, 4.0, 1.0, 4.0, 3.0, 2.0, 2.0,
        3.0, 1.0, 1.0, 2.0,
    ];
    let yg: Vec<f64> = vec![
        2.421196, 0.850101, 1.188318, 0.917668, 1.895064, 2.717167, 4.391082, 0.266883, 1.853922,
        1.838375, 5.959549, 19.008523, 0.121882, 1.544704, 1.422566, 0.758422, 1.264496, 0.147806,
        0.06751, 2.907132, 0.3538, 0.223494, 0.297625, 5.273375, 12.534684, 0.514577, 1.473477,
        0.485665, 0.962023, 1.043896, 1.771311, 1.926229, 7.592099, 1.298714, 0.675125, 0.201756,
        1.814679, 1.104297, 0.434436, 0.470596,
    ];
    const REF_BETA: [f64; 2] = [0.423197712262065, 0.845082014360343];
    const REF_SE: [f64; 2] = [0.0960484092896012, 0.0763975129700953];
    const REF_DISPERSION: f64 = 0.885577425465437;
    let n = 40;
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &x1 {
        x.extend_from_slice(&[1.0, xi]);
    }
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &yg, n, 2, &model, &GroupIds::default(), &opts);
    assert!(f.converged());
    for j in 0..2 {
        assert!((f.beta[j] - REF_BETA[j]).abs() < 1e-6, "beta[{j}]");
        assert!((f.se[j] - REF_SE[j]).abs() < 1e-6, "se[{j}]");
    }
    assert!((f.dispersion - REF_DISPERSION).abs() / REF_DISPERSION < 1e-6);
    // logLik(fg)/df from the same R run — R's Gamma()$aic convention, whose
    // dispersion is profiled as dev/Σwᵢ (NOT the Pearson φ̂ above).
    const REF_LOGLIK: f64 = -118.213036736182;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-6,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 3); // β0, β1, φ
    assert_eq!(f.fitted.len(), 40);
}

/// Weighted binomial-logit GLM on aggregated (proportion, trial-count)
/// rows vs R glm(weights=). Exercises the weighted-logit fallthrough to
/// the general IRLS arm (the fused SIMD logit kernel cannot take
/// per-row weights, see `glm_irls_fit`'s `prior_w` doc).
#[test]
fn fit_glm_binomial_weighted_aggregated_matches_r() {
    // R 4.5.3 oracle (same x1/eta as the Gamma golden above, set.seed(42)):
    //   m <- sample(2:6, n, replace = TRUE)
    //   s <- rbinom(n, m, plogis(eta)); yp <- s / m
    //   fb <- glm(yp ~ x1, family = binomial, weights = m)
    //   print(coef(summary(fb)), digits = 15)
    let x1: [f64; 40] = [
        1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627, 1.3049,
        2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201, -0.3066,
        -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64, 0.4555, 0.7048,
        1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
    ];
    let m: Vec<f64> = vec![
        5.0, 2.0, 6.0, 5.0, 2.0, 2.0, 2.0, 5.0, 3.0, 4.0, 6.0, 6.0, 5.0, 2.0, 4.0, 5.0, 6.0, 3.0,
        2.0, 2.0, 2.0, 4.0, 3.0, 6.0, 5.0, 5.0, 6.0, 2.0, 5.0, 2.0, 2.0, 6.0, 4.0, 2.0, 3.0, 5.0,
        3.0, 6.0, 6.0, 4.0,
    ];
    let yp: Vec<f64> = vec![
        0.800000000000000,
        0.000000000000000,
        0.833333333333333,
        0.600000000000000,
        0.000000000000000,
        0.500000000000000,
        0.500000000000000,
        0.400000000000000,
        0.666666666666667,
        0.250000000000000,
        1.000000000000000,
        1.000000000000000,
        0.200000000000000,
        0.500000000000000,
        0.500000000000000,
        0.800000000000000,
        0.666666666666667,
        0.333333333333333,
        0.000000000000000,
        1.000000000000000,
        1.000000000000000,
        0.500000000000000,
        0.666666666666667,
        1.000000000000000,
        1.000000000000000,
        0.200000000000000,
        0.666666666666667,
        0.500000000000000,
        0.400000000000000,
        0.500000000000000,
        0.500000000000000,
        1.000000000000000,
        1.000000000000000,
        0.500000000000000,
        0.666666666666667,
        0.400000000000000,
        1.000000000000000,
        0.500000000000000,
        0.166666666666667,
        0.500000000000000,
    ];
    const REF_BETA: [f64; 2] = [0.512593391575506, 0.822576961628648];
    const REF_SE: [f64; 2] = [0.181425472435286, 0.170693131259756];
    let n = 40;
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &x1 {
        x.extend_from_slice(&[1.0, xi]);
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(m),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &yp, n, 2, &model, &GroupIds::default(), &opts);
    assert!(f.converged());
    for j in 0..2 {
        assert!((f.beta[j] - REF_BETA[j]).abs() < 1e-6, "beta[{j}]");
        assert!((f.se[j] - REF_SE[j]).abs() < 1e-6, "se[{j}]");
    }
    // logLik(fb)/df from the same R run — includes the ln C(mᵢ,sᵢ) binomial
    // coefficients (dbinom on the aggregated counts), the exact quantity the
    // saturated-constant restoration must reproduce under weights.
    const REF_LOGLIK: f64 = -46.8334270981151;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-6,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 2); // β only; binomial has no free dispersion
}

#[test]
fn fit_glm_smoke() {
    // Logistic data: P(y=1) = σ(0.4 + 1.0·x), x ~ U(−1, 1), Bernoulli sampled
    // from a second LCG draw → non-separable, so IRLS converges to a finite β̂.
    let n = 400;
    let p = 2;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let xi = lcg(&mut st); // U(−1, 1)
        x[i * p] = 1.0;
        x[i * p + 1] = xi;
        let prob = 1.0 / (1.0 + (-(0.4 + 1.0 * xi)).exp());
        let u = (lcg(&mut st) + 1.0) / 2.0; // U(0, 1)
        y[i] = if u < prob { 1.0 } else { 0.0 };
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
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
    assert!(f.converged(), "GLM should converge on clean logistic data");
    assert!(
        f.beta.iter().all(|b| b.is_finite()),
        "β̂ must be finite, got {:?}",
        f.beta
    );
    assert!(
        f.beta[1] > 0.0,
        "slope sign should recover positive, got {}",
        f.beta[1]
    );
    assert!(
        f.se[1].is_finite() && f.se[1] > 0.0,
        "target SE must be finite positive, got {}",
        f.se[1]
    );
    assert!(f.tau2.is_empty(), "GLM has no variance components");
}

/// Poisson GLM through stable `fit` (re: None), gated against the frozen R
/// `glm(family=poisson)` oracle (`validation/goldens/grouseticks_glm.json`):
/// `TICKS ~ 1 + YEAR + cHEIGHT` on grouseticks, canonical log link. Dispersion
/// is fixed `φ≡1`, so SE = √((XᵀWX)⁻¹). Routes the Poisson canonical-shortcut
/// branch of `family.rs`. The oracle is sacred (RULE 0).
#[test]
fn fit_glm_poisson_matches_r() {
    const REF_BETA: [f64; 4] = [
        1.61599798052329,
        0.409645768793675,
        -1.68514104774929,
        -0.0214518421117811,
    ];
    const REF_SE: [f64; 4] = [
        0.0401455805199035,
        0.0453477934183976,
        0.0898007150621173,
        0.000710396896273056,
    ];
    // grouseticks.csv cols: INDEX,TICKS,BROOD,HEIGHT,YEAR,LOCATION,cHEIGHT.
    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    let p = 4; // [intercept, YEAR96, YEAR97, cHEIGHT]; YEAR base level 95.
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let ticks: f64 = f[1].parse().unwrap();
        let year: u32 = f[4].parse().unwrap();
        let cheight: f64 = f[6].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            cheight,
        ]);
        y.push(ticks);
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
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
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "poisson GLM must converge");
    assert!((f.dispersion - 1.0).abs() < 1e-12, "poisson φ≡1");
    assert!(f.tau2.is_empty(), "GLM has no variance components");
    for j in 0..p {
        let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
        assert!(
            b_rel < 1e-3,
            "β[{j}] = {} vs R {} (rel {b_rel})",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < 3e-2,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
    // R logLik on the same fit: glm(TICKS ~ factor(YEAR) + cHEIGHT, poisson)
    // on validation/data/empirical/grouseticks.csv → logLik −2187.40552083455,
    // df 4 (φ≡1: no dispersion parameter).
    const REF_LOGLIK: f64 = -2187.40552083455;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 4);
}

/// Poisson GLM with a per-row offset (the canonical exposure use case), vs R
/// `glm(offset=)` on grouseticks: `TICKS ~ factor(YEAR) + cHEIGHT` with
/// `o_i = 0.1·((i−1) mod 7)` (0-based CSV row order in Rust). Oracle (R 4.5.3):
///   fp <- glm(TICKS ~ YEAR + cHEIGHT, family = poisson, data = gt, offset = og)
///   print(coef(fp), digits = 15); print(logLik(fp), digits = 15)
#[test]
fn fit_glm_poisson_offset_matches_r() {
    const REF_BETA: [f64; 4] = [
        1.3026444328289024,
        0.4002824077793738,
        -1.6756905837047817,
        -0.0213150417946447,
    ];
    const REF_LOGLIK: f64 = -2233.81176722254;
    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let year: u32 = f[4].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            f[6].parse().unwrap(), // cHEIGHT
        ]);
        y.push(f[1].parse().unwrap()); // TICKS
    }
    let n = y.len();
    let o: Vec<f64> = (0..n).map(|i| 0.1 * (i % 7) as f64).collect();
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
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
            target_indices: vec![0, 1, 2, 3],
            offset: Some(o.clone()),
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "poisson GLM with offset must converge");
    for (j, (&b, &r)) in f.beta.iter().zip(&REF_BETA).enumerate() {
        let b_rel = (b - r).abs() / r.abs();
        assert!(b_rel < 1e-3, "β[{j}] = {b} vs R {r} (rel {b_rel})");
    }
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    // fitted = exp(o + Xβ̂): the offset is part of the mean, not of β.
    for i in 0..n.min(20) {
        let eta: f64 = o[i] + (0..p).map(|j| x[i * p + j] * f.beta[j]).sum::<f64>();
        assert!(
            (f.fitted[i] - eta.exp()).abs() < 1e-8 * eta.exp().max(1.0),
            "fitted[{i}]"
        );
    }
}

/// High-mean Poisson GLM through stable `fit` (re: None), gated against the
/// frozen R `glm(family=poisson)` oracle
/// (`validation/goldens/sim_poisson_highmean_glm.json`): `y ~ 1 + x + grp` on
/// sim_poisson_highmean (ȳ ≈ 85). Regression gate for the IRLS log-link cold
/// start: from the old μ = 1 seed (η = 0) any count data with ȳ ≳ ~25–30 made
/// the first WLS step overshoot and IRLS run away (β → ~9e304,
/// `converged = false`); the μ₀ = y + 0.1 seed (R's family `initialize`)
/// converges here. The oracle is sacred (RULE 0).
#[test]
fn fit_glm_poisson_highmean_matches_r() {
    const REF_BETA: [f64; 3] = [4.27614930354405, 0.299823553158498, 0.220101964251659];
    const REF_SE: [f64; 3] = [0.00955233157557028, 0.00587180175968696, 0.0125653819843501];
    // sim_poisson_highmean.csv cols: x,grp,y — grp ∈ {a,b}, base level a.
    let csv = include_str!("../../validation/data/simulated/sim_poisson_highmean.csv");
    let p = 3; // [intercept, x, grpb]
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let xv: f64 = f[0].parse().unwrap();
        let grp_b = f[1] == "b";
        let yv: f64 = f[2].parse().unwrap();
        x.extend_from_slice(&[1.0, xv, f64::from(u32::from(grp_b))]);
        y.push(yv);
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
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
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "high-mean poisson GLM must converge");
    assert!((f.dispersion - 1.0).abs() < 1e-12, "poisson φ≡1");
    for j in 0..p {
        let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
        assert!(
            b_rel < 1e-3,
            "β[{j}] = {} vs R {} (rel {b_rel})",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < 3e-2,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// Probit binomial GLM through stable `fit` (re: None), gated against frozen
/// R `glm(binomial("probit"))` (`validation/goldens/cbpp_probit_glm.json`): cbpp
/// `cbind(incidence, size−incidence) ~ period`, expanded to 0/1 rows (same
/// MLE + Fisher information as the aggregated fit). Probit is non-canonical →
/// the general Fisher-scoring branch; `φ≡1`. The oracle is sacred (RULE 0).
#[test]
fn fit_glm_probit_matches_r() {
    const REF_BETA: [f64; 4] = [
        -0.774138451538547,
        -0.629665092013555,
        -0.693759371053835,
        -0.919560095621316,
    ];
    const REF_SE: [f64; 4] = [
        0.0839559752851447,
        0.150778883932662,
        0.158774972086234,
        0.194512024389745,
    ];
    let csv = include_str!("../../validation/data/empirical/cbpp.csv");
    let p = 4; // [intercept, period2, period3, period4]
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let incidence: u32 = f[1].parse().unwrap();
        let size: u32 = f[2].parse().unwrap();
        let period: u32 = f[3].parse().unwrap();
        let row = [
            1.0,
            f64::from(u32::from(period == 2)),
            f64::from(u32::from(period == 3)),
            f64::from(u32::from(period == 4)),
        ];
        for k in 0..size {
            x.extend_from_slice(&row);
            y.push(if k < incidence { 1.0 } else { 0.0 });
        }
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Probit,
        },
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
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "probit GLM must converge");
    assert!((f.dispersion - 1.0).abs() < 1e-12, "probit φ≡1");
    for j in 0..p {
        let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
        assert!(
            b_rel < 1e-3,
            "β[{j}] = {} vs R {} (rel {b_rel})",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < 3e-2,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// `y ~ 1 + x + grp` design from the committed `sim_gamma.csv`
/// (cluster,x,grp,y); X = [intercept, x, grp=="b"]. Shared by the Gamma
/// goldens.
fn sim_gamma_xy() -> (Vec<f64>, Vec<f64>, usize) {
    let csv = include_str!("../../validation/data/simulated/sim_gamma.csv");
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let xv: f64 = f[1].parse().unwrap();
        let grp_b = f64::from(u32::from(f[2] == "b"));
        let yv: f64 = f[3].parse().unwrap();
        x.extend_from_slice(&[1.0, xv, grp_b]);
        y.push(yv);
    }
    let n = y.len();
    (x, y, n)
}

/// Gamma log-link GLM, gated against frozen R `glm(family=Gamma("log"))`
/// (`validation/goldens/sim_gamma_glm.json`). φ is the post-fit Pearson moment
/// estimator (`dispersion: None`); SE is √φ-scaled, matching R's
/// `summary()$dispersion`. The oracle is sacred (RULE 0).
#[test]
fn fit_glm_gamma_log_matches_r() {
    const REF_BETA: [f64; 3] = [0.449945830683142, 0.565796931228723, 0.526238083012209];
    const REF_SE: [f64; 3] = [0.0818215272793177, 0.0596141419705928, 0.119864153173617];
    const REF_DISP: f64 = 1.0286627876062;
    let (x, y, n) = sim_gamma_xy();
    let p = 3;
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
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
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "gamma-log GLM must converge");
    let disp_rel = (f.dispersion - REF_DISP).abs() / REF_DISP;
    assert!(disp_rel < 5e-3, "φ = {} vs R {REF_DISP}", f.dispersion);
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs R {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(se_rel < 3e-2, "se[{j}] = {} vs R {}", f.se[j], REF_SE[j]);
    }
}

/// Gamma inverse-link GLM, gated against frozen R `glm(family=Gamma("inverse"))`
/// (`validation/goldens/sim_gamma_inv_glm.json`). Inverse is non-canonical (η=1/μ is
/// −θ): the general branch + the 1/y cold-start seed. The oracle is sacred.
#[test]
fn fit_glm_gamma_inverse_matches_r() {
    const REF_BETA: [f64; 3] = [0.629151640871097, -0.198980738259224, -0.176508060896549];
    const REF_SE: [f64; 3] = [0.0432089466672347, 0.0187898149082593, 0.04188122263817];
    const REF_DISP: f64 = 1.0354907206002;
    let (x, y, n) = sim_gamma_xy();
    let p = 3;
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Inverse,
        },
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
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "gamma-inverse GLM must converge");
    let disp_rel = (f.dispersion - REF_DISP).abs() / REF_DISP;
    assert!(disp_rel < 5e-3, "φ = {} vs R {REF_DISP}", f.dispersion);
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs R {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(se_rel < 3e-2, "se[{j}] = {} vs R {}", f.se[j], REF_SE[j]);
    }
}

/// `dispersion: Some(v)` holds φ=v fixed (skips the Pearson estimate) and
/// scales SE by √v. Fitting at Some(1.0) vs Some(2.0) on identical data must
/// give the same β and SE in the exact ratio √2, with `dispersion` reported
/// as the held value.
#[test]
fn fit_glm_gamma_fixed_dispersion_scales_se() {
    let (x, y, n) = sim_gamma_xy();
    let p = 3;
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: None,
    };
    // φ directive lives in FitOptions, not the Family payload.
    let opts = |phi: f64| FitOptions {
        target_indices: vec![0, 1, 2],
        dispersion: Some(phi),
        ..FitOptions::default()
    };
    let f1 = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts(1.0));
    let f2 = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts(2.0));
    assert!(f1.converged() && f2.converged());
    assert!((f2.dispersion - 2.0).abs() < 1e-12, "held φ must be 2.0");
    assert!((f1.dispersion - 1.0).abs() < 1e-12);
    for j in 0..p {
        assert!((f1.beta[j] - f2.beta[j]).abs() < 1e-12, "β φ-independent");
        // SE(φ=2) = √2 · SE(φ=1) exactly (same (XᵀWX)⁻¹, different √φ).
        assert!(
            (f2.se[j] - 2.0_f64.sqrt() * f1.se[j]).abs() < 1e-12,
            "se ratio at j={j}: {} vs {}",
            f2.se[j],
            2.0_f64.sqrt() * f1.se[j]
        );
    }
}

/// Negative-binomial GLM via the alternating outer-θ loop, gated against
/// frozen R `MASS::glm.nb` (`validation/goldens/sim_nb_glm.json`):
/// `y ~ 1 + x + grp` on sim_nb. `dispersion = θ̂` (the estimated shape); β SE
/// conditions on θ̂. The oracle is sacred (RULE 0).
#[test]
fn fit_glm_nb_matches_mass() {
    const REF_BETA: [f64; 3] = [0.144166077871857, 0.619826870647895, 0.633686899496841];
    const REF_SE: [f64; 3] = [0.120690561977139, 0.0756442004078213, 0.155714256322938];
    const REF_THETA: f64 = 1.01052181546876;
    // sim_nb.csv: cluster,x,grp,y (y integer counts).
    let csv = include_str!("../../validation/data/simulated/sim_nb.csv");
    let p = 3;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let xv: f64 = f[1].parse().unwrap();
        let grp_b = f64::from(u32::from(f[2] == "b"));
        let yv: f64 = f[3].parse().unwrap();
        x.extend_from_slice(&[1.0, xv, grp_b]);
        y.push(yv);
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
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
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "NB GLM must converge");
    let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
    assert!(
        th_rel < 2e-2,
        "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
        f.dispersion
    );
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs MASS {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(se_rel < 3e-2, "se[{j}] = {} vs MASS {}", f.se[j], REF_SE[j]);
    }
}

/// Weighted negative-binomial GLM vs `MASS::glm.nb(weights=)`. Convention:
/// prior weight multiplies both the IRLS working weight (β/SE, per Task 2)
/// and the per-row θ profile term (`nb_profile_loglik`'s `weights` — matches
/// `theta.ml`'s weighted profile, the outer loop MASS::glm.nb alternates on).
#[test]
#[expect(
    clippy::approx_constant,
    reason = "R-generated x1 datum 0.3183, not a use of the std FRAC_1_PI constant"
)]
fn fit_glm_nb_weighted_matches_mass() {
    // R 4.5.3 oracle:
    //   library(MASS); set.seed(7); n <- 60
    //   x1 <- round(rnorm(n), 4); w <- sample(1:3, n, TRUE)
    //   mu <- exp(0.5 + 0.6 * x1); y <- rnbinom(n, size = 1.8, mu = mu)
    //   f <- glm.nb(y ~ x1, weights = w)
    //   print(coef(summary(f)), digits = 15); print(f$theta, digits = 15)
    let x1: [f64; 60] = [
        2.2872, -1.1968, -0.6943, -0.4123, -0.9707, -0.9473, 0.7481, -0.117, 0.1527, 2.19, 0.357,
        2.7168, 2.2815, 0.324, 1.8961, 0.4677, -0.8938, -0.3073, -0.0048, 0.9882, 0.8398, 0.7053,
        1.306, -1.388, 1.2729, 0.1842, 0.7523, 0.5917, -0.9831, -0.2761, -0.8709, 0.7187, 0.1107,
        -0.0785, -0.4205, -0.5621, 0.9975, -1.1051, -0.1423, 0.315, 1.2186, -0.6993, -0.2854,
        -1.3116, -0.391, -0.4015, 1.3505, 0.5912, 0.1005, 0.9311, -0.2627, -0.0077, 0.3672, 1.7072,
        0.7237, 0.481, -1.5679, 0.3183, 0.166, -0.8999,
    ];
    let w: [f64; 60] = [
        3.0, 2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0,
        3.0, 2.0, 1.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.0, 3.0, 3.0,
        1.0, 3.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.0, 3.0, 2.0, 3.0, 2.0, 1.0,
        3.0, 2.0, 2.0, 2.0, 2.0, 1.0,
    ];
    let y: [f64; 60] = [
        7.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 7.0, 0.0, 3.0, 4.0, 15.0, 0.0, 12.0, 0.0, 1.0, 0.0,
        3.0, 2.0, 0.0, 4.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0, 5.0, 1.0,
        7.0, 1.0, 3.0, 3.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0,
        1.0, 0.0, 0.0, 7.0, 1.0, 2.0,
    ];
    const REF_BETA: [f64; 2] = [0.448681810160982, 0.593940842956464];
    const REF_SE: [f64; 2] = [0.119405783091442, 0.112801176259142];
    const REF_THETA: f64 = 1.23453054082489;
    let n = 60;
    let p = 2;
    let mut x = Vec::with_capacity(n * p);
    for &xi in &x1 {
        x.extend_from_slice(&[1.0, xi]);
    }
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w.to_vec()),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts);
    assert!(f.converged(), "weighted NB GLM must converge");
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs MASS {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(se_rel < 3e-2, "se[{j}] = {} vs MASS {}", f.se[j], REF_SE[j]);
    }
    let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
    assert!(
        th_rel < 1e-4,
        "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
        f.dispersion
    );
    // logLik(f)/df from the same R run (`logLik.glm` on the glm.nb fit) — the
    // full NB density including the −lnΓ(yᵢ+1) count terms, weighted.
    const REF_LOGLIK: f64 = -220.733217667106;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 5e-4,
        "loglik {} vs MASS {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 3); // β0, β1, θ
}

/// NB GLM with a per-row offset has no R oracle (`MASS::glm.nb` takes no
/// `offset=`), so this pins the offset threaded through the NB outer loop by
/// its structural invariant instead: a CONSTANT offset `c` on the log link
/// shifts only the intercept, by `−c`, leaving the slopes and the estimated
/// dispersion θ̂ untouched (μ̂ = exp(c + β₀ + …) = exp((β₀−c) + …)). The Rust
/// twin of Python's `test_offset_shifts_poisson_intercept_by_minus_constant`.
#[test]
fn fit_glm_nb_constant_offset_shifts_intercept() {
    let csv = include_str!("../../validation/data/simulated/sim_nb.csv");
    let p = 3;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let xv: f64 = f[1].parse().unwrap();
        let grp_b = f64::from(u32::from(f[2] == "b"));
        let yv: f64 = f[3].parse().unwrap();
        x.extend_from_slice(&[1.0, xv, grp_b]);
        y.push(yv);
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: None,
    };
    let base = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    let c = 1.3;
    let shifted = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1, 2],
            offset: Some(vec![c; n]),
            ..FitOptions::default()
        },
    );
    assert!(
        base.converged() && shifted.converged(),
        "NB fits must converge"
    );
    assert!(
        (shifted.beta[0] - (base.beta[0] - c)).abs() < 2e-3,
        "intercept {} vs base {} − {c}",
        shifted.beta[0],
        base.beta[0]
    );
    for j in 1..p {
        assert!(
            (shifted.beta[j] - base.beta[j]).abs() < 2e-3,
            "β[{j}] shifted {} vs base {}",
            shifted.beta[j],
            base.beta[j]
        );
    }
    // θ̂ is offset-invariant: the shape depends on the (unchanged) fitted means.
    assert!(
        (shifted.dispersion - base.dispersion).abs() / base.dispersion < 1e-2,
        "θ̂ shifted {} vs base {}",
        shifted.dispersion,
        base.dispersion
    );
}

/// Parse an `x,grp,y` NB-edge sim CSV → (X=[1,x,grp_b], y, n).
fn nb_edge_data(csv: &str) -> (Vec<f64>, Vec<f64>, usize) {
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        x.extend_from_slice(&[
            1.0,
            f[0].parse().unwrap(),
            f64::from(u32::from(f[1] == "b")),
        ]);
        y.push(f[2].parse().unwrap());
    }
    let n = y.len();
    (x, y, n)
}

/// Fit the NB GLM on an edge dataset and gate against the frozen MASS
/// reference (β rel 1e-3, SE rel 3e-2 — the `fit_glm_nb_matches_mass`
/// bands). Returns the fit so the caller can pin its edge-specific θ̂
/// assertions. Shared by the two θ-bracket-edge tests.
fn nb_edge_fit(csv: &str, ref_beta: &[f64; 3], ref_se: &[f64; 3]) -> Fit {
    let (x, y, n) = nb_edge_data(csv);
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: None,
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        3,
        &model,
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "NB edge GLM must converge");
    for j in 0..3 {
        assert!(
            (f.beta[j] - ref_beta[j]).abs() / ref_beta[j].abs() < 1e-3,
            "β[{j}] = {} vs MASS {}",
            f.beta[j],
            ref_beta[j]
        );
        let se_rel = (f.se[j] - ref_se[j]).abs() / ref_se[j];
        assert!(se_rel < 3e-2, "se[{j}] = {} vs MASS {}", f.se[j], ref_se[j]);
    }
    f
}

/// θ-bracket LOW edge: heavily overdispersed NB GLM (θ̂ ≈ 4.1e-3, half an
/// order above `NB_THETA_LO` = 1e-3), gated against frozen `MASS::glm.nb`
/// (`validation/goldens/sim_nb_lowtheta_glm.json`; glm.nb converged with zero
/// warnings on the committed CSV — the reference is trustworthy this close
/// to the edge, not past it). Pins that the golden-section θ search stays
/// interior and matches MASS near its lower bracket end. The oracle is
/// sacred.
#[test]
fn fit_glm_nb_theta_low_edge_matches_mass() {
    const REF_BETA: [f64; 3] = [0.392948589321679, -1.19642377752834, 0.820910978622294];
    const REF_SE: [f64; 3] = [1.12744374740952, 0.781254798362756, 1.57118324159906];
    const REF_THETA: f64 = 0.00409762150621296;
    let f = nb_edge_fit(
        include_str!("../../validation/data/simulated/sim_nb_lowtheta.csv"),
        &REF_BETA,
        &REF_SE,
    );
    // Sane boundary behavior: inside the bracket, near (but not AT) the low end.
    assert!(
        f.dispersion > super::glm::NB_THETA_LO && f.dispersion < 1e-2,
        "θ̂ = {} must sit interior near NB_THETA_LO",
        f.dispersion
    );
    let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
    assert!(
        th_rel < 2e-2,
        "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
        f.dispersion
    );
}

/// θ-bracket HIGH edge: near-Poisson NB GLM (θ̂ ≈ 5.3e2, pushed toward
/// `NB_THETA_HI` = 1e4), gated against frozen `MASS::glm.nb`
/// (`validation/goldens/sim_nb_hightheta_glm.json`; zero glm.nb warnings on the
/// committed CSV — cells with θ̂ nearer the edge all put `theta.ml` at its
/// iteration/alternation limits, and count size is separately capped by the
/// IRLS cold-start divergence; both constraints are documented at the
/// generator, `prep/export_data.R`). The profile is nearly flat in θ up
/// here, yet both engines maximise the same profile on the same data, so
/// θ̂ still gates at 1e-2 (measured ~2e-9); β/SE stay at the standard bands
/// (β is θ-insensitive near the Poisson limit). The oracle is sacred.
#[test]
fn fit_glm_nb_theta_high_edge_matches_mass() {
    const REF_BETA: [f64; 3] = [2.00540691601978, 0.596522354278958, 0.385922258588444];
    const REF_SE: [f64; 3] = [
        0.00809529402417648,
        0.00479352480867935,
        0.00985794518911199,
    ];
    const REF_THETA: f64 = 534.632483746729;
    let f = nb_edge_fit(
        include_str!("../../validation/data/simulated/sim_nb_hightheta.csv"),
        &REF_BETA,
        &REF_SE,
    );
    // Sane boundary behavior: large but interior (not clamped at NB_THETA_HI).
    assert!(
        f.dispersion > 1e2 && f.dispersion < super::glm::NB_THETA_HI,
        "θ̂ = {} must sit interior, pushed toward NB_THETA_HI",
        f.dispersion
    );
    let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
    assert!(
        th_rel < 1e-2,
        "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
        f.dispersion
    );
}

/// `NB_MAX_OUTER` cap semantics via the `fit_glm_nb_capped` seam, seeded at
/// `NB_THETA_LO` (far from θ̂ ≈ 1.01 on sim_nb) so one alternation cannot
/// meet `NB_THETA_TOL`. Pins that cap exhaustion is SILENT: the capped fit
/// reports `converged = true` (the flag reflects only the last inner IRLS
/// fit, not the θ alternation), β/se stay at the stale pre-update θ, and
/// `dispersion` carries the newer θ. `max_outer = 0` is the degenerate
/// never-ran case: the all-NaN `converged = false` placeholder.
#[test]
fn fit_glm_nb_outer_cap_semantics() {
    // Fixed-only fit; sim_clustered's cluster ids are unused here.
    let (x, y, _ids, _nc) =
        sim_clustered(include_str!("../../validation/data/simulated/sim_nb.csv"));
    let (n, p) = (y.len(), 3);
    let opts = FitOptions {
        target_indices: vec![0, 1, 2],
        ..FitOptions::default()
    };
    let seed = Some(super::glm::NB_THETA_LO);

    let f0 = super::glm::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, 0);
    assert!(
        !f0.converged(),
        "cap 0: never-ran placeholder is converged=false"
    );
    assert!(f0.beta.iter().all(|b| b.is_nan()), "cap 0: β all NaN");

    let f1 = super::glm::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, 1);
    let full = super::glm::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, super::glm::NB_MAX_OUTER);
    // Cap exhaustion is silent: the inner IRLS converged, so the flag is true
    // even though the θ alternation was cut off mid-flight.
    assert!(
        f1.converged(),
        "capped fit reports the INNER convergence flag"
    );
    assert!(full.converged());
    // The single alternation moved θ off the seed (the profile step ran) …
    assert!(
        (f1.dispersion - super::glm::NB_THETA_LO).abs() / super::glm::NB_THETA_LO > 1.0,
        "θ after one alternation ({}) must leave the seed",
        f1.dispersion
    );
    // … but β/se were fit at the stale seed θ = 1e-3, whose NB variance
    // V = μ + μ²/θ is ~10³ wider than the converged fit's — the capped SE
    // must visibly disagree with the fully-alternated one.
    assert!(
        (f1.se[0] - full.se[0]).abs() / full.se[0] > 0.5,
        "capped se[0] = {} vs full {} must reflect the stale θ",
        f1.se[0],
        full.se[0]
    );
    // Sanity: the uncapped path from the same seed reaches the MASS optimum
    // (`fit_glm_nb_matches_mass`'s reference θ̂).
    assert!(
        (full.dispersion - 1.01052181546876).abs() / 1.01052181546876 < 2e-2,
        "full θ̂ = {} vs MASS 1.0105",
        full.dispersion
    );
}

/// The same logistic model in three unit systems, gated against frozen R
/// `glm(family=binomial)` (`validation/goldens/sim_scale_logit_glm.json`,
/// `..._small_glm.json`, `..._big_glm.json`): `y ~ x`, `y ~ x/1000`,
/// `y ~ x*1000` on sim_scale_logit. R fits all three identically — same
/// deviance to 10 digits, same iteration count, coefficients and standard errors
/// scaling exactly — because `glm.fit` has no coefficient cap. glmm used to
/// reject the middle one: its divergence guard bounded |β|, so the accept/reject
/// decision moved with the caller's choice of units. The guard bounds |η| now,
/// which is invariant. The oracle is sacred (RULE 0).
#[test]
fn fit_glm_scale_variation_matches_r() {
    // From validation/goldens/sim_scale_logit_glm.json (estimates.beta / .se).
    const REF_BETA: [f64; 2] = [-0.311403810670574, 2.0819815005051];
    const REF_SE: [f64; 2] = [0.209439712908431, 0.268038942174326];

    // sim_scale_logit.csv cols: y,x,x_small,x_big
    let csv = include_str!("../../validation/data/simulated/sim_scale_logit.csv");
    let mut cols: [Vec<f64>; 4] = Default::default();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        for (k, f) in line.split(',').enumerate() {
            cols[k].push(f.trim_matches('"').parse().unwrap());
        }
    }
    let y = cols[0].clone();
    let n = y.len();
    let p = 2;
    let model = ModelSpec {
        family: Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        re: None,
    };

    // col 1 = x (scale 1), col 2 = x/1000, col 3 = x*1000. The frozen reference
    // for each is the same fit; only the slope's units differ.
    for (col, scale) in [(1usize, 1.0f64), (2, 1e-3), (3, 1e3)] {
        let mut x = Vec::<f64>::with_capacity(n * p);
        for &xi in &cols[col] {
            x.extend_from_slice(&[1.0, xi]);
        }
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1],
                ..FitOptions::default()
            },
        );
        assert!(
            f.converged(),
            "scale {scale}: must converge — R converges on all three"
        );
        let expect = [REF_BETA[0], REF_BETA[1] / scale];
        let expect_se = [REF_SE[0], REF_SE[1] / scale];
        for j in 0..p {
            let b_rel = (f.beta[j] - expect[j]).abs() / expect[j].abs();
            assert!(
                b_rel < 1e-3,
                "scale {scale} β[{j}] = {} vs R {} (rel {b_rel})",
                f.beta[j],
                expect[j]
            );
            let se_rel = (f.se[j] - expect_se[j]).abs() / expect_se[j];
            assert!(
                se_rel < 1e-3,
                "scale {scale} se[{j}] = {} vs R {} (rel {se_rel})",
                f.se[j],
                expect_se[j]
            );
        }
    }
}

/// Complete separation, gated against frozen R
/// (`validation/goldens/sim_scale_sep_glm.json`): `y = 1[x > 0]`, where R
/// reports `converged: FALSE` after exhausting `maxit`. glmm must also refuse.
/// Only the FLAG is compared, not the coefficients: both engines stop at an
/// arbitrary point on a path to infinity, and R's own stopping point depends on
/// its iteration budget (25) which differs from glmm's (50). The oracle is
/// sacred (RULE 0).
#[test]
fn fit_glm_separated_rejected_like_r() {
    let csv = include_str!("../../validation/data/simulated/sim_scale_sep.csv");
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        x.extend_from_slice(&[1.0, f[1].parse().unwrap()]);
    }
    let n = y.len();
    let f = fit_cold(
        &x,
        &y,
        n,
        2,
        &ModelSpec {
            family: Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            re: None,
        },
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    assert!(
        !f.converged(),
        "completely separated data must be refused, as R's glm.fit refuses it"
    );
}

/// Gamma inverse link with a small mean, gated against frozen R
/// `glm(Gamma(link="inverse"))` (`validation/goldens/sim_scale_gamma_inv_glm.json`):
/// `y ~ x` on sim_scale_gamma_inv, where μ ≈ 0.01 so η = 1/μ ≈ 100. A flat
/// divergence cap of 30 on |η| would refuse this honest fit, which is why the
/// GLM guard skips this family/link pair. The oracle is sacred (RULE 0).
#[test]
fn fit_glm_gamma_inverse_small_mean_matches_r() {
    // From validation/goldens/sim_scale_gamma_inv_glm.json.
    const REF_BETA: [f64; 2] = [99.8813244077148, -19.7574641575102];
    const REF_SE: [f64; 2] = [0.204076575867777, 0.711212166103063];

    let csv = include_str!("../../validation/data/simulated/sim_scale_gamma_inv.csv");
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        x.extend_from_slice(&[1.0, f[1].parse().unwrap()]);
    }
    let n = y.len();
    let f = fit_cold(
        &x,
        &y,
        n,
        2,
        &ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Inverse,
            },
            re: None,
        },
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "small-mean Gamma inverse fit must converge");
    for j in 0..2 {
        let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
        assert!(
            b_rel < 1e-3,
            "β[{j}] = {} vs R {} (rel {b_rel})",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < 1e-3,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
}
