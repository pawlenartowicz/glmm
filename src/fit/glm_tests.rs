//! GLM estimator tests (fixed-effects binomial/Poisson/Gamma/negative-binomial,
//! `re: None`).

use super::glm::fit_glm;
use super::*;
use crate::{
    BinomialLink, Family, GroupIds, ModelSpec, NegBinomialLink, Note, ReStructure, Sizing,
};

use super::common_tests::{assert_pinned, lcg, sim_clustered, PIN_REL_OLS};

/// SE agreement band against R/MASS for the φ≡1 GLM families (binomial,
/// Poisson): dispersion is held fixed at 1, so SE is the closed-form
/// √((XᵀWX)⁻¹) off the converged IRLS/Fisher-scoring weights with no moment
/// estimate in the way. `3e-5` = ceil-to-one-significant-figure(2 × 1.4441e-5),
/// the worst measured relative gap (`fit_glm_cloglog_matches_r`); the gap comes
/// from the IRLS/Fisher-scoring stopping tolerance, not from rounding, so it
/// sits well above machine epsilon.
const SE_REL_PHI1: f64 = 3e-5;

/// SE agreement band against R/MASS for GLM families whose dispersion is
/// estimated rather than fixed — Gamma and inverse-Gaussian (post-fit Pearson
/// φ̂) and negative-binomial (profile-likelihood θ̂): SE scales through the
/// estimated dispersion on top of the same IRLS stopping-tolerance gap
/// `SE_REL_PHI1` already carries, so it needs its own, looser floor. `7e-5` =
/// ceil-to-one-significant-figure(2 × 3.2793e-5), the worst measured relative
/// gap (`fit_glm_nb_theta_low_edge_matches_mass`, the heavily-overdispersed
/// θ-bracket edge).
const SE_REL_DISPERSION: f64 = 7e-5;

/// Weighted Gamma(log) GLM vs R glm(weights=). Precision weights: row `i` has
/// variance `φ·V(μᵢ)/wᵢ`. β is the same either way (the mean model doesn't
/// see the convention), but SE and `dispersion` are `summary(glm)`'s Pearson
/// moment `φ̂ = Σwᵢrᵢ²/(n−p)`, and `loglik` is the maximised precision
/// log-likelihood (Σwᵢ, the case-weight ML value, differs).
#[test]
fn fit_glm_gamma_weighted_matches_r() {
    // R 4.5.3 oracle (set.seed(42), n = 40):
    //   x1 <- round(rnorm(n), 4); w <- sample(1:4, n, replace = TRUE)
    //   eta <- 0.4 + 0.8 * x1
    //   yg <- round(rgamma(n, shape = 2, scale = exp(eta) / 2), 6)
    //   fg <- glm(yg ~ x1, family = Gamma("log"), weights = w)
    //   print(coef(fg), digits = 17)
    //   print(coef(summary(fg))[, 2], digits = 17); print(summary(fg)$dispersion, digits = 17)
    //   # precision loglik: row i has variance phi*V(mu_i)/w_i (shape a_i = w_i/phi)
    //   mu <- fitted(fg)
    //   prec_ll <- function(phi) sum(dgamma(yg, shape = w / phi, rate = (w / phi) / mu, log = TRUE))
    //   opt <- optimize(prec_ll, c(1e-6, 100), maximum = TRUE, tol = 1e-12)
    //   print(opt$maximum, digits = 17); print(opt$objective, digits = 17)
    //   # cross-check: glmmTMB(yg ~ x1, family = Gamma("log"),
    //   #                      dispformula = ~ offset(log(w))) -> same beta, logLik
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
    const REF_BETA: [f64; 2] = [0.4231977122620654, 0.8450820143603432];
    const REF_SE: [f64; 2] = [0.09604840928960116, 0.07639751297009528];
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
    // The maximised precision log-likelihood, cross-checked against
    // glmmTMB(dispformula = ~ offset(log(w))) at -47.268246110587896.
    const REF_LOGLIK: f64 = -47.26824611058739;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-6,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 3); // β0, β1, φ
    assert_eq!(f.fitted.len(), 40);
}

/// Precision weights make the fit invariant to the overall SCALE of `w`:
/// `wᵢ → c·wᵢ` leaves every row's shape `aᵢ = wᵢ/φ` unchanged once φ absorbs
/// the factor. The mean-model IRLS fit runs on raw, unnormalised `w` (its
/// weighted-least-squares argmin does not depend on the overall scale in
/// exact arithmetic, for any positive `c`), and stops on `glm::DEVIANCE_TOL`'s
/// RELATIVE rule, which is scale-free for `|deviance| ≫ 0.1` (true at every
/// tested `c` here) — β/SE/dispersion agree to 1e-9 relative at every tested
/// `c` (measured worst case ~2e-10). `loglik` — the
/// closed-form ML φ̂ solve, which runs on the internal `ŵ` instead — agrees to
/// 1e-9 relative too, since powers of two keep `ŵ` bit-exact. `2^20` also
/// exercises the ML solver's start. Same x1/w/yg fixture as
/// `fit_glm_gamma_weighted_matches_r`.
#[test]
fn fit_glm_gamma_weight_scale_invariant() {
    let x1: [f64; 40] = [
        1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627, 1.3049,
        2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201, -0.3066,
        -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64, 0.4555, 0.7048,
        1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
    ];
    let w: [f64; 40] = [
        4.0, 1.0, 2.0, 1.0, 1.0, 4.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 1.0, 4.0, 4.0, 2.0, 1.0, 4.0,
        2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 4.0, 3.0, 4.0, 1.0, 4.0, 1.0, 4.0, 3.0, 2.0, 2.0,
        3.0, 1.0, 1.0, 2.0,
    ];
    let yg: [f64; 40] = [
        2.421196, 0.850101, 1.188318, 0.917668, 1.895064, 2.717167, 4.391082, 0.266883, 1.853922,
        1.838375, 5.959549, 19.008523, 0.121882, 1.544704, 1.422566, 0.758422, 1.264496, 0.147806,
        0.06751, 2.907132, 0.3538, 0.223494, 0.297625, 5.273375, 12.534684, 0.514577, 1.473477,
        0.485665, 0.962023, 1.043896, 1.771311, 1.926229, 7.592099, 1.298714, 0.675125, 0.201756,
        1.814679, 1.104297, 0.434436, 0.470596,
    ];
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
    let fit_at = |c: f64| {
        let wc: Vec<f64> = w.iter().map(|&wi| c * wi).collect();
        fit_cold(
            &x,
            &yg,
            n,
            2,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1],
                weights: Some(wc),
                ..FitOptions::default()
            },
        )
    };
    let base = fit_at(1.0);
    assert!(base.converged());
    for &c in &[8.0_f64, 2.0_f64.powi(-6), 2.0_f64.powi(20)] {
        let f = fit_at(c);
        assert!(f.converged(), "c = {c}");
        for j in 0..2 {
            let b_rel = (f.beta[j] - base.beta[j]).abs() / base.beta[j].abs();
            assert!(
                b_rel < 1e-9,
                "c = {c}: β[{j}] {} vs {}",
                f.beta[j],
                base.beta[j]
            );
            let se_rel = (f.se[j] - base.se[j]).abs() / base.se[j].abs();
            assert!(
                se_rel < 1e-9,
                "c = {c}: se[{j}] {} vs {}",
                f.se[j],
                base.se[j]
            );
        }
        let ll_rel = (f.loglik - base.loglik).abs() / base.loglik.abs();
        assert!(
            ll_rel < 1e-9,
            "c = {c}: loglik {} vs {}",
            f.loglik,
            base.loglik
        );
        let disp_rel = (f.dispersion - c * base.dispersion).abs() / (c * base.dispersion);
        assert!(
            disp_rel < 1e-9,
            "c = {c}: dispersion {} vs {c}·{}",
            f.dispersion,
            base.dispersion
        );
    }
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
    // Rust-vs-Rust pin, not an R oracle — logit accuracy against R is covered
    // separately by `fit_glm_binomial_weighted_aggregated_matches_r`. This
    // catches a 2x-scaled or otherwise-wrong-but-finite-positive β̂ that the
    // old finiteness/sign-only checks let through.
    const REF_BETA: [f64; 2] = [0.491837142677357, 0.872230971652398];
    const REF_SE1: f64 = 0.187677151494303;
    assert_pinned(&f.beta, &REF_BETA, PIN_REL_OLS, "beta");
    assert_pinned(&f.se[1..2], &[REF_SE1], PIN_REL_OLS, "se");
    assert!(f.tau2.is_empty(), "GLM has no variance components");
}

/// Poisson GLM through stable `fit` (re: None), gated against the frozen R
/// `glm(family=poisson)` oracle (`validation/goldens/grouseticks_glm.json`):
/// `TICKS ~ 1 + YEAR + cHEIGHT` on grouseticks, canonical log link. Dispersion
/// is fixed `φ≡1`, so SE = √((XᵀWX)⁻¹). Routes the Poisson canonical-shortcut
/// branch of `family.rs`. The oracle is sacred.
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
            se_rel < SE_REL_PHI1,
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

/// Weighted Poisson(log) GLM vs R `glm(weights=)`. Prior weight multiplies the
/// IRLS working weight and deviance the same way it does in the Gamma/NB arms
/// (`fit_glm_gamma_weighted_matches_r`); φ stays fixed at 1.
#[test]
fn fit_glm_poisson_weighted_matches_r() {
    // R 4.5.3 oracle (set.seed(43), n = 40):
    //   x1 <- round(rnorm(n), 4); w <- sample(1:4, n, replace = TRUE)
    //   y <- rpois(n, lambda = exp(0.3 + 0.5 * x1))
    //   fp <- glm(y ~ x1, family = poisson, weights = w)
    //   print(coef(summary(fp)), digits = 15); print(logLik(fp), digits = 15)
    let x1: [f64; 40] = [
        -0.0375, -1.5746, -0.486, 0.4652, -0.9041, -0.2774, 0.3864, -0.0604, -0.6862, -1.9061,
        1.8038, -0.9669, -0.3531, 1.1069, 0.5663, 2.0643, 1.4693, -1.6515, 0.2026, -0.721, -0.159,
        0.7342, -0.3633, -0.0101, 0.5828, -0.2933, -1.3944, -0.0851, -0.6881, -0.7765, 1.7442,
        0.4566, -0.1182, 1.6754, -1.159, -0.0406, 1.0889, 1.5121, 0.8857, 0.3146,
    ];
    let w: Vec<f64> = vec![
        1.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 4.0, 2.0, 4.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0,
        3.0, 1.0, 1.0, 2.0, 4.0, 1.0, 1.0, 4.0, 2.0, 1.0, 4.0, 2.0, 4.0, 4.0, 1.0, 4.0, 2.0, 4.0,
        1.0, 3.0, 2.0, 3.0,
    ];
    let y: Vec<f64> = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 4.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 1.0, 1.0,
        2.0, 0.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0, 3.0, 1.0, 3.0, 7.0, 6.0, 1.0, 5.0, 0.0, 4.0,
        2.0, 2.0, 3.0, 1.0,
    ];
    const REF_BETA: [f64; 2] = [0.543560457232364, 0.506189423172992];
    const REF_SE: [f64; 2] = [0.0863317392574266, 0.0761557350301788];
    const REF_LOGLIK: f64 = -151.640748208627;
    let n = 40;
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &x1 {
        x.extend_from_slice(&[1.0, xi]);
    }
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &opts);
    assert!(f.converged(), "weighted poisson GLM must converge");
    assert!((f.dispersion - 1.0).abs() < 1e-12, "poisson φ≡1");
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
            se_rel < SE_REL_PHI1,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 2); // β only; poisson has no free dispersion
}

/// Poisson(log) GLM with prior weights AND a per-row offset together:
/// only `fit_ols_offset_matches_r_lm` combines the two, and only for Gaussian.
/// Vs R `glm(weights=, offset=)`.
#[test]
fn fit_glm_poisson_weighted_offset_matches_r() {
    // R 4.5.3 oracle (set.seed(44), n = 40):
    //   x1 <- round(rnorm(n), 4); w <- sample(1:4, n, replace = TRUE)
    //   o <- 0.1 * ((seq_len(n) - 1) %% 7)
    //   y <- rpois(n, lambda = exp(0.3 + 0.5 * x1 + o))
    //   fp <- glm(y ~ x1, family = poisson, weights = w, offset = o)
    //   print(coef(summary(fp)), digits = 15); print(logLik(fp), digits = 15)
    let x1: [f64; 40] = [
        0.6539, 0.0191, -1.8495, -0.1328, -1.1988, -1.3297, 0.9165, -0.163, -1.6021, -0.8352,
        0.3619, -0.3238, -0.7299, -0.7046, -0.3622, 0.1341, 1.6394, -1.3996, 3.0322, 1.1984,
        0.0559, 0.4144, -0.9294, -0.6573, -0.0341, -2.2582, -0.5237, 1.188, 1.5156, 0.1834,
        -0.0686, 1.6507, 1.426, -1.4339, 0.1258, 0.6734, 0.1094, 0.1095, -0.1591, -0.1998,
    ];
    let w: Vec<f64> = vec![
        2.0, 3.0, 2.0, 3.0, 3.0, 4.0, 2.0, 4.0, 3.0, 3.0, 4.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,
        4.0, 1.0, 1.0, 1.0, 4.0, 4.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0, 4.0, 3.0, 4.0, 4.0, 1.0, 2.0,
        3.0, 1.0, 1.0, 4.0,
    ];
    let y: Vec<f64> = vec![
        3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 5.0, 2.0, 2.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0,
        13.0, 4.0, 3.0, 3.0, 0.0, 4.0, 0.0, 0.0, 2.0, 6.0, 4.0, 4.0, 1.0, 3.0, 3.0, 1.0, 2.0, 2.0,
        0.0, 2.0, 3.0, 1.0,
    ];
    const REF_BETA: [f64; 2] = [0.0940812560237927, 0.6691944996210337];
    const REF_SE: [f64; 2] = [0.0859428695879596, 0.0494290057981252];
    const REF_LOGLIK: f64 = -157.855689341192;
    let n = 40;
    let o: Vec<f64> = (0..n).map(|i| 0.1 * (i % 7) as f64).collect();
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &x1 {
        x.extend_from_slice(&[1.0, xi]);
    }
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(w),
        offset: Some(o),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &opts);
    assert!(f.converged(), "weighted+offset poisson GLM must converge");
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
            se_rel < SE_REL_PHI1,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
}

/// High-mean Poisson GLM through stable `fit` (re: None), gated against the
/// frozen R `glm(family=poisson)` oracle
/// (`validation/goldens/sim_poisson_highmean_glm.json`): `y ~ 1 + x + grp` on
/// sim_poisson_highmean (ȳ ≈ 85). Regression gate for the IRLS log-link cold
/// start: from the old μ = 1 seed (η = 0) any count data with ȳ ≳ ~25–30 made
/// the first WLS step overshoot and IRLS run away (β → ~9e304,
/// `converged = false`); the μ₀ = y + 0.1 seed (R's family `initialize`)
/// converges here. The oracle is sacred.
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
            se_rel < SE_REL_PHI1,
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
/// the general Fisher-scoring branch; `φ≡1`. The oracle is sacred.
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
            se_rel < SE_REL_PHI1,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// Weighted, aggregated, non-canonical-link (probit) binomial GLM vs R
/// `glm(weights=)`. Prior weights only reach the canonical logit path
/// (`fit_glm_binomial_weighted_aggregated_matches_r`) elsewhere in this file, so
/// this exercises the weighted general Fisher-scoring branch instead.
#[test]
fn fit_glm_probit_weighted_matches_r() {
    // R 4.5.3 oracle (set.seed(43), n = 40):
    //   x2 <- round(rnorm(n), 4); m <- sample(2:6, n, replace = TRUE)
    //   s <- rbinom(n, m, pnorm(0.2 + 0.6 * x2)); yp <- s / m
    //   fb <- glm(yp ~ x2, family = binomial(link = "probit"), weights = m)
    //   print(coef(summary(fb)), digits = 15)
    let x2: [f64; 40] = [
        -0.0375, -1.5746, -0.486, 0.4652, -0.9041, -0.2774, 0.3864, -0.0604, -0.6862, -1.9061,
        1.8038, -0.9669, -0.3531, 1.1069, 0.5663, 2.0643, 1.4693, -1.6515, 0.2026, -0.721, -0.159,
        0.7342, -0.3633, -0.0101, 0.5828, -0.2933, -1.3944, -0.0851, -0.6881, -0.7765, 1.7442,
        0.4566, -0.1182, 1.6754, -1.159, -0.0406, 1.0889, 1.5121, 0.8857, 0.3146,
    ];
    let m: Vec<f64> = vec![
        2.0, 2.0, 3.0, 4.0, 6.0, 6.0, 5.0, 2.0, 3.0, 4.0, 4.0, 2.0, 6.0, 6.0, 4.0, 2.0, 6.0, 3.0,
        6.0, 6.0, 3.0, 6.0, 5.0, 6.0, 3.0, 6.0, 5.0, 3.0, 2.0, 2.0, 5.0, 4.0, 3.0, 3.0, 5.0, 2.0,
        4.0, 4.0, 6.0, 6.0,
    ];
    let yp: Vec<f64> = vec![
        0.500000000000000,
        0.500000000000000,
        0.666666666666667,
        0.500000000000000,
        0.333333333333333,
        0.333333333333333,
        0.600000000000000,
        0.000000000000000,
        1.000000000000000,
        0.750000000000000,
        1.000000000000000,
        1.000000000000000,
        0.166666666666667,
        0.500000000000000,
        0.750000000000000,
        1.000000000000000,
        0.833333333333333,
        0.000000000000000,
        0.666666666666667,
        0.500000000000000,
        0.666666666666667,
        1.000000000000000,
        0.200000000000000,
        0.166666666666667,
        1.000000000000000,
        0.333333333333333,
        0.400000000000000,
        0.666666666666667,
        1.000000000000000,
        1.000000000000000,
        0.600000000000000,
        0.500000000000000,
        0.333333333333333,
        0.666666666666667,
        0.000000000000000,
        1.000000000000000,
        1.000000000000000,
        1.000000000000000,
        0.500000000000000,
        0.666666666666667,
    ];
    const REF_BETA: [f64; 2] = [0.148570258330836, 0.386380163829481];
    const REF_SE: [f64; 2] = [0.100794909899017, 0.108744926783315];
    let n = 40;
    let mut x = Vec::with_capacity(n * 2);
    for &xi in &x2 {
        x.extend_from_slice(&[1.0, xi]);
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Probit,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        weights: Some(m),
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &yp, n, 2, &model, &GroupIds::default(), &opts);
    assert!(f.converged(), "weighted probit GLM must converge");
    assert!((f.dispersion - 1.0).abs() < 1e-12, "probit φ≡1");
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
            se_rel < SE_REL_PHI1,
            "se[{j}] = {} vs R {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// Cloglog GLM through stable `fit` (re: None): the fitted means must reproduce
/// the link's own inverse at the converged η, and the log-likelihood (−½ the
/// binomial `dev_resid` sum for 0/1 y) must be the Bernoulli sum at those means. Non-canonical → general
/// Fisher-scoring branch. The R-gated version is
/// `fit_glm_cloglog_matches_r` (validation golden `sim_cloglog_glm`).
#[test]
fn fit_glm_cloglog_is_self_consistent() {
    // Small separable-free design: 200 rows, one continuous predictor.
    let n = 200usize;
    let p = 2usize;
    let mut x = Vec::<f64>::with_capacity(n * p);
    let mut y = Vec::<f64>::with_capacity(n);
    for i in 0..n {
        let xi = -2.0 + 4.0 * (i as f64) / (n as f64);
        x.push(1.0);
        x.push(xi);
        // Deterministic 0/1 pattern with both classes present at every x range.
        y.push(f64::from(u32::from(i % 3 == 0)));
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Cloglog,
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
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "cloglog GLM must converge");
    assert_eq!(f.dispersion, 1.0, "binomial holds φ≡1");
    for (i, &mu) in f.fitted.iter().enumerate() {
        let eta = f.beta[0] + f.beta[1] * x[i * p + 1];
        let want = crate::family::link_inv(model.family, eta);
        assert!((mu - want).abs() < 1e-9, "μ[{i}] = {mu} vs {want}");
        assert!(mu > 0.0 && mu < 1.0);
    }
    // y is 0/1, so the saturated log-likelihood is 0 and logLik is the Bernoulli
    // sum at the fitted means (`Fit.deviance` is NaN on the GLM route).
    let want_ll: f64 = y
        .iter()
        .zip(&f.fitted)
        .map(|(&yi, &mu)| yi * mu.ln() + (1.0 - yi) * (1.0 - mu).ln())
        .sum();
    assert!(
        (f.loglik - want_ll).abs() <= 1e-9 * want_ll.abs(),
        "logLik {} vs Bernoulli sum at the fitted means {want_ll}",
        f.loglik
    );
}

/// Cloglog binomial GLM through stable `fit` (re: None), gated against frozen R
/// `glm(binomial("cloglog"))` (`validation/goldens/sim_cloglog_glm.json`) on the
/// 9,600-row `sim_probit_large` fixture. Cloglog is non-canonical → the general
/// Fisher-scoring branch; `φ≡1`. The oracle is sacred.
#[test]
fn fit_glm_cloglog_matches_r() {
    const REF_BETA: [f64; 5] = [
        0.0284899934174711,
        0.432010019796386,
        -0.345842927736584,
        0.192639099878277,
        -0.51719986105143,
    ];
    const REF_SE: [f64; 5] = [
        0.0199775612287933,
        0.0156164313872971,
        0.0152923154008697,
        0.0148074298207126,
        0.029541138884397,
    ];
    let csv = include_str!("../../validation/data/simulated/sim_probit_large.csv");
    let p = 5; // [intercept, x1, x2, x3, z]
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        x.extend_from_slice(&[
            1.0,
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
            f[4].parse().unwrap(),
        ]);
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Cloglog,
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
            target_indices: vec![0, 1, 2, 3, 4],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "cloglog GLM must converge");
    assert!((f.dispersion - 1.0).abs() < 1e-12, "cloglog φ≡1");
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
            se_rel < SE_REL_PHI1,
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
/// (`validation/goldens/sim_gamma_glm.json`): β from `stats::glm`, φ̂ its
/// Pearson moment, SE `summary(fg)`'s, logLik the maximised precision one
/// (`validation/goldens/sim_gamma_glm_ml.json`'s — unweighted, so the ML and
/// precision likelihoods coincide). The oracle is sacred.
#[test]
fn fit_glm_gamma_log_matches_r() {
    const REF_BETA: [f64; 3] = [0.449945830683142, 0.565796931228723, 0.526238083012209];
    const REF_SE: [f64; 3] = [0.0818215272793177, 0.0596141419705928, 0.119864153173617];
    const REF_DISP: f64 = 1.0286627876062;
    const REF_LOGLIK: f64 = -489.6218795591715;
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
    assert!(
        disp_rel < SE_REL_DISPERSION,
        "φ = {} vs R {REF_DISP}",
        f.dispersion
    );
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-8,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs R {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// The Gamma-log and inverse-Gaussian-log goldens with the response in other
/// units, `y → c·y`, which puts |ln μ̂| past 30. On a log link the model is
/// exactly equivariant: the intercept moves by `ln c`, the slopes and SEs stay,
/// the Gamma φ̂ stays and the inverse-Gaussian φ̂ scales by `1/c`. So the frozen
/// R fits of `fit_glm_gamma_log_matches_r` and `fit_glm_igauss_matches_r` are
/// the reference, and a divergence guard on |η| itself refuses these honest
/// fits.
#[test]
fn fit_glm_log_link_converges_in_any_response_units() {
    const GAMMA_BETA: [f64; 3] = [0.449945830683142, 0.565796931228723, 0.526238083012209];
    const GAMMA_SE: [f64; 3] = [0.0818215272793177, 0.0596141419705928, 0.119864153173617];
    const GAMMA_DISP: f64 = 1.0286627876062;
    const IG_BETA: [f64; 3] = [0.388918465368179, 0.0547374103199151, 0.0848741285568016];
    const IG_SE: [f64; 3] = [0.020676393123407, 0.0149473668365438, 0.0298734973282543];
    const IG_DISP: f64 = 0.289727973895683;
    let gamma = Family::Gamma {
        link: crate::GammaLink::Log,
    };
    let ig = Family::InverseGaussian {
        link: crate::InverseGaussianLink::Log,
    };
    let cases = [
        (gamma, 1e-14, GAMMA_BETA, GAMMA_SE, GAMMA_DISP),
        (gamma, 1e14, GAMMA_BETA, GAMMA_SE, GAMMA_DISP),
        (ig, 1e-14, IG_BETA, IG_SE, IG_DISP),
        (ig, 1e14, IG_BETA, IG_SE, IG_DISP),
    ];
    for (family, c, ref_beta, ref_se, ref_disp) in cases {
        let (x, y, n) = match family {
            Family::Gamma { .. } => sim_gamma_xy(),
            _ => sim_igauss_xy(),
        };
        let y: Vec<f64> = y.iter().map(|v| v * c).collect();
        let f = fit_cold(
            &x,
            &y,
            n,
            3,
            &ModelSpec { family, re: None },
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged(), "{family:?}, y·{c:e}: must converge");
        let disp = match family {
            Family::Gamma { .. } => f.dispersion,
            _ => f.dispersion * c,
        };
        assert!(
            (disp - ref_disp).abs() / ref_disp < 5e-3,
            "{family:?}, y·{c:e}: φ = {} vs R {ref_disp}",
            f.dispersion
        );
        for j in 0..3 {
            let b = if j == 0 {
                f.beta[0] - c.ln()
            } else {
                f.beta[j]
            };
            assert!(
                (b - ref_beta[j]).abs() / ref_beta[j].abs() < 1e-3,
                "{family:?}, y·{c:e}: β[{j}] = {} vs R {} (intercept shifted by ln c)",
                f.beta[j],
                ref_beta[j]
            );
            let se_rel = (f.se[j] - ref_se[j]).abs() / ref_se[j];
            assert!(
                se_rel < SE_REL_DISPERSION,
                "{family:?}, y·{c:e}: se[{j}] = {} vs R {}",
                f.se[j],
                ref_se[j]
            );
        }
    }
}

/// The relative deviance rule `|ΔD| / (|D| + f)` on the two dispersion
/// families, with the response or the precision weights in other units. The
/// model is exactly equivariant: on the log link `y → c·y` moves only the
/// intercept, by `ln c`, and `w → k·w` moves no β. The deviance scales by `1/c`
/// (inverse-Gaussian) and by `k`, so with R's fixed `f = 0.1` the rule turned
/// absolute and stopped early: inverse-Gaussian at `y·1e3` 1e-6 off, Gamma with
/// every weight 1e-9 2e-3 off, both reported converged. The reference is the
/// same fit in the fixture's own units, to 1e-10.
#[test]
fn fit_glm_dispersion_family_stop_rule_is_scale_free() {
    let gamma = Family::Gamma {
        link: crate::GammaLink::Log,
    };
    let ig = Family::InverseGaussian {
        link: crate::InverseGaussianLink::Log,
    };
    let fit_with = |family: Family, c: f64, k: Option<f64>| {
        let (x, y, n) = match family {
            Family::Gamma { .. } => sim_gamma_xy(),
            _ => sim_igauss_xy(),
        };
        let y: Vec<f64> = y.iter().map(|v| v * c).collect();
        fit_cold(
            &x,
            &y,
            n,
            3,
            &ModelSpec { family, re: None },
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                weights: k.map(|k| vec![k; n]),
                ..FitOptions::default()
            },
        )
    };
    let cases = [
        (gamma, 1.0, Some(1e-9)),
        (ig, 1.0, Some(1e-9)),
        (ig, 1e4, None),
        (ig, 1e14, None),
    ];
    for (family, c, k) in cases {
        let base = fit_with(family, 1.0, None);
        let f = fit_with(family, c, k);
        assert!(
            f.converged(),
            "{family:?}, y·{c:e}, w = {k:?}: must converge"
        );
        for j in 0..3 {
            let b = if j == 0 {
                f.beta[0] - c.ln()
            } else {
                f.beta[j]
            };
            assert!(
                (b - base.beta[j]).abs() / base.beta[j].abs() < 1e-10,
                "{family:?}, y·{c:e}, w = {k:?}: β[{j}] = {} vs {} (intercept shifted by ln c)",
                f.beta[j],
                base.beta[j]
            );
        }
    }
}

/// Poisson with non-integer y in tiny units (a rate without exposure weights):
/// the deviance scales with ȳ, so R's absolute `+ 0.1` floor would stop the
/// fit early while it reports converged (1.7e-2 off in β at y·1e-13). The
/// floor carries ȳ's units below ȳ = 1, so the fit is the y·1 fit with the
/// intercept shifted by ln c. The cold seed and the divergence guard's centre
/// are ln ȳ with no `+ 0.1` either: centred on ln(ȳ + 0.1) ≈ ln 0.1, the fit at
/// y·1e-16 (ln μ̂ near −36) sat more than 30 from it and was refused. The data
/// are `sim_gamma_xy`'s positive y.
#[test]
fn fit_glm_poisson_small_mean_stop_rule_is_scale_free() {
    let (x, y, n) = sim_gamma_xy();
    let fit_with = |c: f64| {
        let y: Vec<f64> = y.iter().map(|v| v * c).collect();
        fit_cold(
            &x,
            &y,
            n,
            3,
            &ModelSpec {
                family: Family::Poisson {
                    link: crate::PoissonLink::Log,
                },
                re: None,
            },
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        )
    };
    let base = fit_with(1.0);
    assert!(base.converged(), "Poisson on y·1 must converge");
    for c in [1e-7, 1e-10, 1e-13, 1e-16, 1e-20] {
        let f = fit_with(c);
        assert!(f.converged(), "Poisson on y·{c:e} must converge");
        for j in 0..3 {
            let b = if j == 0 {
                f.beta[0] - c.ln()
            } else {
                f.beta[j]
            };
            assert!(
                (b - base.beta[j]).abs() / base.beta[j].abs() < 1e-10,
                "Poisson on y·{c:e}: β[{j}] = {} vs {} (intercept shifted by ln c)",
                f.beta[j],
                base.beta[j]
            );
        }
    }
}

/// The post-fit saturation guard is a binomial test: a fitted probability
/// pinned at 0 or 1. Its bound on the IRLS weight is absolute, and on the
/// other families that weight carries the response's units, so these honest
/// fits were refused:
/// - Poisson with exposure 1e-7 on 30 of 40 rows: the fitted means there are
///   about 3e-7, below the bound on the Poisson weight μ. Frozen R 4.5.3 fit:
///   `glm(y ~ x1 + offset(log(e)), poisson, epsilon = 1e-15)` on the fixture
///   below (`i = 0..39`, `x1 = (7i mod 40)/39`, `e = 1e-7` for `i < 30`).
/// - Gamma/inverse with `y·1e-4` (weight μ²) and inverse-Gaussian `1/μ²` with
///   `y·1e-2` (weight μ³/4). Both are exactly equivariant, β scaling by `1/c`
///   and `1/c²`, so the references are the frozen R fits of
///   `fit_glm_gamma_inverse_matches_r` and
///   `fit_glm_igauss_inverse_squared_matches_r`.
#[test]
fn fit_glm_saturation_guard_is_binomial_only() {
    const POIS_BETA: [f64; 2] = [1.68065523018842, -0.602386665393579];
    const POIS_SE: [f64; 2] = [0.352733819647417, 0.61603058055554];
    let n = 40;
    let mut x = Vec::with_capacity(n * 2);
    let mut offset = Vec::with_capacity(n);
    for i in 0..n {
        x.extend_from_slice(&[1.0, ((i * 7) % 40) as f64 / 39.0]);
        offset.push(if i < 30 { 1e-7_f64.ln() } else { 0.0 });
    }
    let mut y = vec![0.0f64; 30];
    y.extend_from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]);
    let f = fit_cold(
        &x,
        &y,
        n,
        2,
        &ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: None,
        },
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1],
            offset: Some(offset),
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "Poisson with exposure 1e-7: must converge");
    for j in 0..2 {
        assert!(
            (f.beta[j] - POIS_BETA[j]).abs() / POIS_BETA[j].abs() < 1e-10,
            "Poisson: β[{j}] = {} vs R {}",
            f.beta[j],
            POIS_BETA[j]
        );
        assert!(
            (f.se[j] - POIS_SE[j]).abs() / POIS_SE[j] < SE_REL_PHI1,
            "Poisson: se[{j}] = {} vs R {}",
            f.se[j],
            POIS_SE[j]
        );
    }

    const GAMMA_INV_BETA: [f64; 3] = [0.629151640871097, -0.198980738259224, -0.176508060896549];
    const IG_INV_SQ_BETA: [f64; 3] = [0.459456921305253, -0.0419956854008115, -0.0679438262942649];
    let cases = [
        (
            Family::Gamma {
                link: crate::GammaLink::Inverse,
            },
            1e-4,
            1e4,
            GAMMA_INV_BETA,
        ),
        (
            Family::InverseGaussian {
                link: crate::InverseGaussianLink::InverseSquared,
            },
            1e-2,
            1e4,
            IG_INV_SQ_BETA,
        ),
    ];
    for (family, c, beta_scale, ref_beta) in cases {
        let (x, y, n) = match family {
            Family::Gamma { .. } => sim_gamma_xy(),
            _ => sim_igauss_xy(),
        };
        let y: Vec<f64> = y.iter().map(|v| v * c).collect();
        let f = fit_cold(
            &x,
            &y,
            n,
            3,
            &ModelSpec { family, re: None },
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged(), "{family:?}, y·{c:e}: must converge");
        for (j, (&b, &r)) in f.beta.iter().zip(&ref_beta).enumerate() {
            let want = r * beta_scale;
            assert!(
                (b - want).abs() / want.abs() < 1e-3,
                "{family:?}, y·{c:e}: β[{j}] = {b} vs R {want}"
            );
        }
    }
}

/// Log-link fits whose exposures differ by many decades between rows. The
/// divergence guard and the cold seed are centred on the null model with the
/// offset, oᵢ + ln(Σy / Σe^o); centred on ln ȳ alone, rows with exposure
/// 1e-16 of the rest sit about 37 from it and the fit was refused.
/// - Poisson on the `fit_glm_saturation_guard_is_binomial_only` fixture with
///   exposure 1e-16 and 1e16 on its 30 zero rows. Frozen R 4.5.3 fits:
///   `glm(y ~ x1 + offset(log(e)), poisson, epsilon = 1e-14)`.
/// - Gamma/log on `sim_gamma.csv` with exposure `e` on the odd rows and their
///   y scaled by `e`. The Gamma unit deviance is unchanged when a row's y and μ
///   scale together, so β is exactly the frozen R fit of
///   `fit_glm_gamma_log_matches_r`.
#[test]
fn fit_glm_log_link_offset_units_do_not_matter() {
    let n = 40;
    let mut x = Vec::with_capacity(n * 2);
    for i in 0..n {
        x.extend_from_slice(&[1.0, ((i * 7) % 40) as f64 / 39.0]);
    }
    let mut y = vec![0.0f64; 30];
    y.extend_from_slice(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0]);
    let pois_cases = [
        (
            1e-16_f64,
            [1.68065573993521, -0.602387049125178],
            [0.352733841333684, 0.616030628260301],
        ),
        (
            1e16,
            [-36.7249672461556, 0.293967798814331],
            [0.312495121342981, 0.525981485832405],
        ),
    ];
    for (e, ref_beta, ref_se) in pois_cases {
        let offset: Vec<f64> = (0..n).map(|i| if i < 30 { e.ln() } else { 0.0 }).collect();
        let f = fit_cold(
            &x,
            &y,
            n,
            2,
            &ModelSpec {
                family: Family::Poisson {
                    link: crate::PoissonLink::Log,
                },
                re: None,
            },
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1],
                offset: Some(offset),
                ..FitOptions::default()
            },
        );
        assert!(f.converged(), "Poisson with exposure {e:e}: must converge");
        for j in 0..2 {
            assert!(
                (f.beta[j] - ref_beta[j]).abs() / ref_beta[j].abs() < 1e-10,
                "Poisson, exposure {e:e}: β[{j}] = {} vs R {}",
                f.beta[j],
                ref_beta[j]
            );
            assert!(
                (f.se[j] - ref_se[j]).abs() / ref_se[j] < SE_REL_PHI1,
                "Poisson, exposure {e:e}: se[{j}] = {} vs R {}",
                f.se[j],
                ref_se[j]
            );
        }
    }

    const GAMMA_BETA: [f64; 3] = [0.449945830683142, 0.565796931228723, 0.526238083012209];
    let (x, y, n) = sim_gamma_xy();
    for e in [1e-16_f64, 1e16] {
        let offset: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 1 { e.ln() } else { 0.0 })
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 1 { y[i] * e } else { y[i] })
            .collect();
        let f = fit_cold(
            &x,
            &y,
            n,
            3,
            &ModelSpec {
                family: Family::Gamma {
                    link: crate::GammaLink::Log,
                },
                re: None,
            },
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                offset: Some(offset),
                ..FitOptions::default()
            },
        );
        assert!(
            f.converged(),
            "Gamma/log with exposure {e:e}: must converge"
        );
        for (j, (&b, &r)) in f.beta.iter().zip(&GAMMA_BETA).enumerate() {
            assert!(
                (b - r).abs() / r.abs() < 1e-4,
                "Gamma/log, exposure {e:e}: β[{j}] = {b} vs R {r}"
            );
        }
    }
}

/// A Gamma GLM whose mean model reproduces the data exactly (constant `y`,
/// intercept only): the deviance is rounding noise and the maximum-likelihood
/// φ̂ goes to ~1e-16, where the log-likelihood rises without bound. The
/// reported `loglik` is the value at that φ̂, so it must be at least the
/// log-likelihood at any held φ. R's `logLik` is NaN here (its `Gamma()$aic`
/// at a deviance that rounds below zero).
#[test]
fn fit_glm_gamma_exact_fit_loglik_is_the_maximum() {
    let n = 24;
    let x = vec![1.0f64; n];
    let y = vec![3.0f64; n];
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: None,
    };
    let fit = |dispersion: Option<f64>| {
        fit_cold(
            &x,
            &y,
            n,
            1,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0],
                dispersion,
                ..FitOptions::default()
            },
        )
    };
    let ml = fit(None);
    assert!(ml.converged(), "exact Gamma fit must converge");
    for phi in [1e-4, 1e-8, 1e-12] {
        let held = fit(Some(phi));
        assert!(
            ml.loglik >= held.loglik,
            "ML loglik {} below the loglik {} at held φ = {phi:e}",
            ml.loglik,
            held.loglik
        );
    }
}

/// Gamma inverse-link GLM, gated against frozen R `glm(family=Gamma("inverse"))`
/// (`validation/goldens/sim_gamma_inv_glm.json`, the Pearson-φ̂ convention of
/// `fit_glm_gamma_log_matches_r`). Inverse is non-canonical (η=1/μ is −θ): the
/// general branch + the 1/y cold-start seed. The oracle is sacred.
#[test]
fn fit_glm_gamma_inverse_matches_r() {
    const REF_BETA: [f64; 3] = [0.629151640871097, -0.198980738259224, -0.176508060896549];
    const REF_SE: [f64; 3] = [0.0432089466672347, 0.0187898149082593, 0.04188122263817];
    const REF_DISP: f64 = 1.0354907206002;
    const REF_LOGLIK: f64 = -493.0941438528882;
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
    assert!(
        disp_rel < SE_REL_DISPERSION,
        "φ = {} vs R {REF_DISP}",
        f.dispersion
    );
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-8,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs R {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// Gamma(log) GLM with a per-row offset, vs R `glm(Gamma("log"),
/// offset=)` on `sim_gamma.csv`, `o_i = 0.1·(i mod 7)` (the same offset shape
/// as `fit_glm_poisson_offset_matches_r`). Log is non-canonical for Gamma in
/// this crate, so this exercises the general Fisher-scoring branch's offset
/// fold, which Binomial/Poisson offset coverage never touches.
#[test]
fn fit_glm_gamma_offset_matches_r() {
    // R 4.5.3 oracle:
    //   gt <- read.csv("sim_gamma.csv"); o <- 0.1 * ((seq_len(nrow(gt))-1) %% 7)
    //   fg <- glm(y ~ x + grp, data = gt, family = Gamma("log"), offset = o)
    //   print(coef(summary(fg)), digits = 15); print(summary(fg)$dispersion, digits = 15)
    //   the maximised logLik is the precision ML φ̂'s, computed exactly as in
    //   `fit_glm_gamma_weighted_matches_r` (unweighted here, so it equals R's own)
    const REF_BETA: [f64; 3] = [0.200540179916345, 0.569394040617305, 0.455205206563064];
    const REF_SE: [f64; 3] = [0.0828872899140855, 0.0603906433037824, 0.1214254383261718];
    const REF_DISP: f64 = 1.0556349151192;
    const REF_LOGLIK: f64 = -495.620364037027;
    let (x, y, n) = sim_gamma_xy();
    let p = 3;
    let o: Vec<f64> = (0..n).map(|i| 0.1 * (i % 7) as f64).collect();
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
            offset: Some(o),
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "gamma GLM with offset must converge");
    let disp_rel = (f.dispersion - REF_DISP).abs() / REF_DISP;
    assert!(
        disp_rel < SE_REL_DISPERSION,
        "φ = {} vs R {REF_DISP}",
        f.dispersion
    );
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs R {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
}

/// `y ~ 1 + x + grp` design from the committed `sim_igauss.csv` (y,x,grp);
/// X = [intercept, x, grp=="b"]. Shared by the inverse-Gaussian goldens.
fn sim_igauss_xy() -> (Vec<f64>, Vec<f64>, usize) {
    let csv = include_str!("../../validation/data/simulated/sim_igauss.csv");
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let yv: f64 = f[0].parse().unwrap();
        let xv: f64 = f[1].parse().unwrap();
        let grp_b = f64::from(u32::from(f[2] == "b"));
        x.extend_from_slice(&[1.0, xv, grp_b]);
        y.push(yv);
    }
    let n = y.len();
    (x, y, n)
}

/// Inverse-Gaussian GLM, log link, gated against frozen R
/// `glm(family=inverse.gaussian("log"))` (`validation/goldens/sim_igauss_glm.json`).
/// V(μ)=μ³, φ̂ Pearson post-fit (`summary(glm)`'s). The oracle is sacred.
#[test]
fn fit_glm_igauss_matches_r() {
    const REF_BETA: [f64; 3] = [0.388918465368179, 0.0547374103199151, 0.0848741285568016];
    const REF_SE: [f64; 3] = [0.020676393123407, 0.0149473668365438, 0.0298734973282543];
    const REF_DISP: f64 = 0.289727973895683;
    let (x, y, n) = sim_igauss_xy();
    let p = 3;
    let model = ModelSpec {
        family: Family::InverseGaussian {
            link: crate::InverseGaussianLink::Log,
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
    assert!(f.converged(), "igauss-log GLM must converge");
    // The fitted means are the log link's inverse at the converged η.
    for (i, &mu) in f.fitted.iter().enumerate() {
        let eta: f64 = (0..p).map(|j| f.beta[j] * x[i * p + j]).sum();
        let want = eta.exp();
        assert!((mu - want).abs() / want < 1e-9, "μ[{i}] = {mu} vs {want}");
    }
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
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// Inverse-Gaussian GLM, 1/μ² link, gated against frozen R
/// `glm(family=inverse.gaussian("1/mu^2"))`
/// (`validation/goldens/sim_igauss_inv_sq_glm.json`). The link is canonical only
/// up to sign and scale, so it takes the general Fisher-scoring branch, with the
/// η₀ = 1/y² cold start. The oracle is sacred.
#[test]
fn fit_glm_igauss_inverse_squared_matches_r() {
    const REF_BETA: [f64; 3] = [0.459456921305253, -0.0419956854008115, -0.0679438262942649];
    const REF_SE: [f64; 3] = [0.0189525417126228, 0.0125108528341764, 0.0251747245203818];
    const REF_DISP: f64 = 0.29048859329441;
    let (x, y, n) = sim_igauss_xy();
    let p = 3;
    let model = ModelSpec {
        family: Family::InverseGaussian {
            link: crate::InverseGaussianLink::InverseSquared,
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
    assert!(f.converged(), "igauss-inverse_squared GLM must converge");
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
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// Same inverse-Gaussian `1/μ²` fixture with the response scaled by 0.1, which
/// puts μ̂ in [0.128, 0.203] and so η̂ = 1/μ̂² in [24.2, 60.8] — above
/// `ETA_DIVERGENCE_CAP`, where the fit is still the honest small-mean one and
/// must converge. The model is exactly equivariant under `y → c·y` here
/// (η = 1/μ² scales by `c⁻²` and so does every β), so the reference is the
/// frozen R β of `fit_glm_igauss_inverse_squared_matches_r` times 100, which R
/// reproduces to 15 digits on the scaled column; φ̂ scales by `c⁻¹`.
#[test]
fn fit_glm_igauss_inverse_squared_converges_at_small_mu() {
    const REF_BETA: [f64; 3] = [45.9456921305253, -4.19956854008116, -6.79438262942649];
    const REF_SE: [f64; 3] = [1.89525417126228, 1.25108528341764, 2.51747245203818];
    const REF_DISP: f64 = 2.9048859329441;
    let (x, y, n) = sim_igauss_xy();
    let y: Vec<f64> = y.iter().map(|v| v * 0.1).collect();
    let p = 3;
    let model = ModelSpec {
        family: Family::InverseGaussian {
            link: crate::InverseGaussianLink::InverseSquared,
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
    assert!(
        f.converged(),
        "small-mean igauss-inverse_squared GLM must converge; β = {:?}",
        f.beta
    );
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
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
}

/// Inverse-Gaussian(log) GLM with a per-row offset, vs R
/// `glm(inverse.gaussian("log"), offset=)` on `sim_igauss.csv`, the same
/// `o_i = 0.1·(i mod 7)` offset shape as the Poisson/Gamma offset goldens. Log
/// is non-canonical for inverse-Gaussian, exercising the general
/// Fisher-scoring branch's offset fold on the second family that never
/// touched it before this test.
#[test]
fn fit_glm_igauss_offset_matches_r() {
    // R 4.5.3 oracle:
    //   it <- read.csv("sim_igauss.csv"); o <- 0.1 * ((seq_len(nrow(it))-1) %% 7)
    //   fi <- glm(y ~ x + grp, data = it, family = inverse.gaussian("log"), offset = o)
    //   print(coef(summary(fi)), digits = 15); print(summary(fi)$dispersion, digits = 15)
    //   print(logLik(fi), digits = 15)
    const REF_BETA: [f64; 3] = [0.1538085257325026, 0.0622414458057679, 0.0737976388898105];
    const REF_SE: [f64; 3] = [0.0220286758020475, 0.0158884686612027, 0.0317295003334939];
    const REF_DISP: f64 = 0.314469305573548;
    const REF_LOGLIK: f64 = -2458.48472319725;
    let (x, y, n) = sim_igauss_xy();
    let p = 3;
    let o: Vec<f64> = (0..n).map(|i| 0.1 * (i % 7) as f64).collect();
    let model = ModelSpec {
        family: Family::InverseGaussian {
            link: crate::InverseGaussianLink::Log,
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
            offset: Some(o),
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "igauss GLM with offset must converge");
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
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs R {REF_LOGLIK}",
        f.loglik
    );
}

/// `dispersion: Some(v)` holds φ=v fixed (skips the ML estimate) and
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
    // logLik is evaluated at the held φ: holding the ML φ̂ (`free.dispersion`
    // is the Pearson moment, a different value — see
    // validation/goldens/sim_gamma_glm_ml.json) reproduces the free fit's
    // (maximised) logLik, and any other held value is lower.
    const ML_PHI_HAT: f64 = 0.7857435708451531;
    let free = fit_cold(
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
    let at_hat = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &opts(ML_PHI_HAT),
    );
    assert!(
        (at_hat.loglik - free.loglik).abs() < 1e-9 * free.loglik.abs(),
        "loglik at the held φ̂ {} vs the free fit's {}",
        at_hat.loglik,
        free.loglik
    );
    assert!(f1.loglik < free.loglik && f2.loglik < free.loglik);
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

/// `Family::Gamma { .. } | Family::InverseGaussian { .. }` share one
/// `dispersion: Some(v)` directive: the Gamma half is covered by
/// `fit_glm_gamma_fixed_dispersion_scales_se` above; this is its
/// inverse-Gaussian twin, so an IG-specific φ-scaling regression cannot hide
/// behind the Gamma-only coverage.
#[test]
fn fit_glm_igauss_fixed_dispersion_scales_se() {
    let (x, y, n) = sim_igauss_xy();
    let p = 3;
    let model = ModelSpec {
        family: Family::InverseGaussian {
            link: crate::InverseGaussianLink::Log,
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
    // `loglik` must actually READ the held φ, not silently profile its own
    // `D/n` regardless of what is held — reconstruct D independently from the
    // converged means and check the closed form at both held values.
    let dev: f64 = (0..n)
        .map(|i| crate::family::dev_resid(model.family, f64::NAN, y[i], f1.fitted[i]))
        .sum();
    let ln_y: f64 = y[..n].iter().map(|v| v.ln()).sum();
    let expect_ll = |phi: f64| {
        -0.5 * (dev / phi + n as f64 * (2.0 * std::f64::consts::PI * phi).ln() + 3.0 * ln_y)
    };
    assert!(
        (f1.loglik - expect_ll(1.0)).abs() < 1e-6 * expect_ll(1.0).abs(),
        "loglik(φ=1) {} vs closed form {}",
        f1.loglik,
        expect_ll(1.0)
    );
    assert!(
        (f2.loglik - expect_ll(2.0)).abs() < 1e-6 * expect_ll(2.0).abs(),
        "loglik(φ=2) {} vs closed form {}",
        f2.loglik,
        expect_ll(2.0)
    );
    // Pinned against R (`sim_igauss`'s design, μ̂ from this fit's own `f1.fitted`):
    //   sum(statmod::dinvgauss(y, mean = mu_hat, dispersion = 1, log = TRUE))
    //   sum(statmod::dinvgauss(y, mean = mu_hat, dispersion = 2, log = TRUE))
    // give −2853.830350967997 and −3389.827736208988 respectively; AIC at
    // `dispersion = 1` (`-2*loglik + 2*3`, df = 3: 2 β + the held φ costs no
    // extra df) is 5713.660701935994.
    const REF_LOGLIK_PHI1: f64 = -2853.830350967997;
    const REF_LOGLIK_PHI2: f64 = -3389.827736208988;
    assert!(
        (f1.loglik - REF_LOGLIK_PHI1).abs() < 1e-6,
        "loglik(φ=1) {} vs R {REF_LOGLIK_PHI1}",
        f1.loglik
    );
    assert!(
        (f2.loglik - REF_LOGLIK_PHI2).abs() < 1e-6,
        "loglik(φ=2) {} vs R {REF_LOGLIK_PHI2}",
        f2.loglik
    );
    const REF_AIC_PHI1: f64 = 5713.660701935994;
    assert!(
        (-2.0 * f1.loglik + 2.0 * f1.df as f64 - REF_AIC_PHI1).abs() < 1e-6,
        "AIC(φ=1) {} vs R {REF_AIC_PHI1}",
        -2.0 * f1.loglik + 2.0 * f1.df as f64
    );
    assert!(
        f1.loglik != f2.loglik,
        "held φ must change loglik, not be ignored"
    );
}

/// Negative-binomial GLM via the alternating outer-θ loop, gated against
/// frozen R `MASS::glm.nb` (`validation/goldens/sim_nb_glm.json`):
/// `y ~ 1 + x + grp` on sim_nb. `dispersion = θ̂` (the estimated shape); β SE
/// conditions on θ̂. The oracle is sacred.
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
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs MASS {}",
            f.se[j],
            REF_SE[j]
        );
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
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs MASS {}",
            f.se[j],
            REF_SE[j]
        );
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

/// Fixed-θ NB GLM whose MLE repels plain Fisher scoring: at β̂ the observed
/// information exceeds twice the expected one along one direction (largest
/// eigenvalue of F⁻¹H is 2.18), so undamped IRLS steps back and forth across
/// β̂ with growing amplitude and runs out `MAX_IRLS_ITERS`, as R's `glm.fit`
/// does on the same call. The period-2 damping in `glm_irls_fit` halves the
/// steps once they reverse, which maps that eigenvalue of the iteration map to
/// (1 + (1 − 2.18))/2 ≈ −0.09, and the fit converges to the MLE.
#[test]
fn fit_glm_nb_two_cycle_is_damped_to_the_mle() {
    // R 4.5.3, MASS 7.3-65:
    //   library(MASS); set.seed(1967); n <- sample(c(12, 15, 20), 1)
    //   th <- sample(c(0.3, 0.5, 1), 1); lmu <- sample(c(3, 5, 7), 1)
    //   x <- round(rnorm(n), 1); y <- rnbinom(n, size = th, mu = exp(lmu + 0.5 * x))
    //   # n = 12, th = 0.5, lmu = 7
    //   f0 <- glm(y ~ x, family = negative.binomial(0.5),
    //             control = glm.control(maxit = 50, epsilon = 1e-12))
    //   f0$converged                              # FALSE: a two-cycle
    //   X <- cbind(1, x); th <- 0.5; b <- c(log(mean(y)), 0)
    //   for (it in 1:100) {                       # Newton, observed information
    //     m <- drop(exp(X %*% b)); s <- colSums(X * (th * (y - m) / (th + m)))
    //     H <- crossprod(X * (th * m * (y + th) / (th + m)^2), X)
    //     b <- b + solve(H, s)
    //   }
    //   m <- drop(exp(X %*% b))
    //   print(b, digits = 15)
    //   print(sqrt(diag(solve(crossprod(X * (th * m / (th + m)), X)))), digits = 15)
    //   print(sum(dnbinom(y, size = th, mu = m, log = TRUE)), digits = 15)
    // The SEs are the Fisher ones at φ = 1, `summary(glm, dispersion = 1)`'s.
    let x1: [f64; 12] = [
        0.4, 0.2, -0.8, 1.0, 0.0, -1.1, -0.7, 1.2, 1.8, 1.1, -1.6, 1.0,
    ];
    let y: [f64; 12] = [
        106.0, 235.0, 1373.0, 268.0, 437.0, 1546.0, 591.0, 144.0, 9551.0, 678.0, 4913.0, 629.0,
    ];
    const REF_BETA: [f64; 2] = [7.417292663027711, 0.0817550188556138];
    const REF_SE: [f64; 2] = [0.416751106933391, 0.400555016260832];
    const REF_LOGLIK: f64 = -100.808778157978;
    let (n, p) = (12, 2);
    let mut x = Vec::with_capacity(n * p);
    for &xi in &x1 {
        x.extend_from_slice(&[1.0, xi]);
    }
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let f = fit_glm(family, 0.5, &x, &y, n, p, &opts);
    assert!(f.converged(), "damped IRLS must converge");
    // Bands are 2 × the measured gap (β 9.6e-8, loglik 3.3e-12 relative). The
    // halved steps converge linearly, so the relative-deviance stopping rule
    // leaves more β residual than on an undamped fit. The SEs sit 2.2e-10 away.
    for j in 0..p {
        let rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
        assert!(
            rel < 2e-7,
            "β[{j}] = {} vs R {} (rel {rel:e})",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < SE_REL_PHI1,
            "se[{j}] = {} vs R {}",
            f.se[j],
            REF_SE[j]
        );
    }
    let ll_rel = (f.loglik - REF_LOGLIK).abs() / REF_LOGLIK.abs();
    assert!(
        ll_rel < 7e-12,
        "loglik {} vs R {REF_LOGLIK} (rel {ll_rel:e})",
        f.loglik
    );
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
/// reference (β rel 1e-3, SE rel `SE_REL_DISPERSION` — the
/// `fit_glm_nb_matches_mass` bands). Returns the fit so the caller can pin its
/// edge-specific θ̂ assertions. Shared by the two θ-bracket-edge tests.
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
        assert!(
            se_rel < SE_REL_DISPERSION,
            "se[{j}] = {} vs MASS {}",
            f.se[j],
            ref_se[j]
        );
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
/// generator, `validation/tools/prep/export_data.R`). The profile is nearly flat in θ up
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

/// [`super::glm::nb_profile_loglik`]'s two forms, evaluated one row past
/// [`super::glm::Y_SUM_MAX`] where the function itself takes the `ln_gamma`
/// branch, against the finite sum computed independently right here — the
/// same statement the function makes below the switch, done by hand above it.
#[test]
fn nb_profile_loglik_matches_finite_sum_at_the_switch() {
    let y = (super::glm::Y_SUM_MAX + 1) as f64;
    for &theta in &[1e-3, 1.0, 1e2, 1e3] {
        let mut want = 0.0;
        for k in 0..(y.round() as u64) {
            want += (theta + k as f64).ln();
        }
        // `mu = 0.0` takes the saturated-mean guard, isolating the switched
        // term from the mean-model piece `nb_profile_loglik` adds on top.
        let got = super::glm::nb_profile_loglik(&[y], &[0.0], theta, None);
        let rel = (got - want).abs() / want.abs();
        // 1e-13 is `Y_SUM_MAX`'s own switch criterion (see its doc); measured
        // worst case right at the switch is ~1.55e-14.
        assert!(
            rel < 1e-13,
            "theta={theta}: got {got} want {want} rel {rel}"
        );
    }
}

/// `nb_profile_loglik` on a single row of `y = 1e10` costs `O(1)` above
/// `Y_SUM_MAX`. Below the cap this same call is the finite sum, `O(y)`
/// unconditionally, so on `y = 1e10` it does not return in any bounded time —
/// this is the direct check the cap exists for. `50 ms` is generous (measured
/// cost of the lgamma path is around 14 µs).
#[test]
fn nb_profile_loglik_on_a_huge_count_is_fast() {
    let start = std::time::Instant::now();
    let ll = super::glm::nb_profile_loglik(&[1e10], &[1e10], 1.0, None);
    let elapsed = start.elapsed();
    assert!(ll.is_finite(), "loglik must be finite, got {ll}");
    assert!(elapsed.as_secs_f64() < 0.05, "took {elapsed:?}");
}

/// A fit whose IRLS actually converges (unlike a single `y = 1e10` outlier,
/// which trips IRLS's own divergence guards before any θ-profile evaluation
/// runs) but whose counts are ALL above `Y_SUM_MAX`, so the θ golden-section
/// search's every evaluation touches the switched code on every row, not just
/// one outlier row among many small ones. Below the cap this fit costs `O(y)`
/// per row per evaluation — see the measured comparison at the bound below.
#[test]
fn fit_glm_nb_with_large_counts_converges_and_is_fast() {
    let n = 30;
    let p = 2;
    let mut st = 11u64;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = lcg(&mut st);
        x.push(1.0);
        x.push(xi);
        // An exact log-linear mean, no added noise: every count clears
        // `Y_SUM_MAX` (mu ranges from about 148,000 to 270,000), and the
        // outer θ alternation settles in two rounds (round 1 at the
        // method-of-moments seed, round 2 at the golden-section optimum),
        // running the θ search's evaluations on every row each round.
        let mu = 2.0e5 * (0.3 * xi).exp();
        y.push(mu.round());
    }
    assert!(
        y.iter().all(|&yi| yi > super::glm::Y_SUM_MAX as f64),
        "every count must clear Y_SUM_MAX for this test to exercise the switch"
    );
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let start = std::time::Instant::now();
    let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts);
    let elapsed = start.elapsed();
    assert!(f.converged(), "must converge");
    assert!(
        f.dispersion.is_finite(),
        "θ̂ must be finite, got {}",
        f.dispersion
    );
    // Measured: ~7 ms. Measured with `Y_SUM_MAX` raised above every count here
    // (forcing the finite-sum branch unconditionally): ~1.7 s, since every one
    // of this fixture's ~50 evaluations then sums a ~200,000-term series on
    // every row.
    assert!(elapsed.as_secs_f64() < 0.5, "took {elapsed:?}");
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

    let f0 = super::glm::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, 0).0;
    assert!(
        !f0.converged(),
        "cap 0: never-ran placeholder is converged=false"
    );
    assert!(f0.beta.iter().all(|b| b.is_nan()), "cap 0: β all NaN");
    assert!(f0.dispersion.is_nan(), "cap 0: dispersion NaN, never ran");

    let f1 = super::glm::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, 1).0;
    let full = super::glm::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, super::glm::NB_MAX_OUTER).0;
    // Cap exhaustion is reported, not silent: the inner IRLS converged, so the
    // fit keeps `converged = true`, but it carries a NbShapeUnsettled note with
    // the number of rounds it ran.
    assert!(
        f1.converged(),
        "capped fit keeps the inner convergence flag"
    );
    let unsettled: Vec<_> = f1
        .diagnostics
        .notes
        .iter()
        .filter(|n| matches!(n, Note::NbShapeUnsettled { .. }))
        .collect();
    assert_eq!(
        unsettled.len(),
        1,
        "one NbShapeUnsettled note: {:?}",
        f1.diagnostics.notes
    );
    assert!(matches!(unsettled[0], Note::NbShapeUnsettled { rounds: 1 }));
    // A fit whose alternation settles carries no such note.
    assert!(full.converged());
    assert!(
        !full
            .diagnostics
            .notes
            .iter()
            .any(|n| matches!(n, Note::NbShapeUnsettled { .. })),
        "a settled alternation raises nothing"
    );
    // One model, not two: β̂/SE are refit at the θ the fit reports, so a single
    // fixed-θ GLM at `f1.dispersion` (the test-only `fit_glm`, cold IRLS at β = 0,
    // as every alternation round is) reproduces them.
    assert!(f1.dispersion.is_finite());
    let family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let at_theta = super::glm::fit_glm(family, f1.dispersion, &x, &y, n, p, &opts);
    assert_eq!(
        f1.beta, at_theta.beta,
        "β̂ must be the fit at the reported θ"
    );
    assert_eq!(f1.se, at_theta.se, "SE must be the fit at the reported θ");
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
/// scaling exactly — because `glm.fit` has no coefficient cap. glmm's
/// divergence guard bounds |η| rather than |β|, so the accept/reject decision
/// is invariant to the caller's choice of units. The oracle is sacred.
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
/// sacred.
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
    assert!(
        f.dispersion.is_nan(),
        "dispersion must be NaN on a refused fit, not binomial's structural 1.0: {}",
        f.dispersion
    );
}

/// Failed Gamma GLM (no random effects), **inverse** link, perfectly
/// separated: `y ~ x`, `y ∈ {1e-6, 1e6}` split exactly on `x ∈ {0, 1}`. Under
/// `η = 1/μ` the exact fit needs `η` near `1e6` for one group and near `1e-6`
/// for the other, so the intercept and slope must grow without settling; R's
/// own `glm(Gamma(link="inverse"))` does not converge on this data either
/// (checked in R 4.5.3, `glm.control(maxit=2000, epsilon=1e-12)`: still
/// `converged=FALSE` at iteration 2000, deviance crawling from 6.3e-07 to
/// 1.6e-07, coefficients stuck at ±999957.9 — a real non-convergent fit, not
/// an artifact of one solver's cold start). `dispersion` must be NaN, not the
/// Gamma exponential special case `φ=1`, which a caller cannot tell from a
/// real estimate. Also with a caller-held φ (`FitOptions::dispersion =
/// Some(2.0)`): the held value is neither an estimate off this fit nor honored
/// when the fit never reached an endpoint, so `2.0` would be as dishonest here
/// as the unheld case's `1.0`.
#[test]
fn fit_glm_gamma_failed_fit_dispersion_is_nan() {
    let n = 24;
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = if i < 12 { 0.0 } else { 1.0 };
        y[i] = if i < 12 { 1e-6 } else { 1e6 };
    }
    for held in [None, Some(2.0)] {
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &ModelSpec {
                family: Family::Gamma {
                    link: crate::GammaLink::Inverse,
                },
                re: None,
            },
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1],
                dispersion: held,
                ..FitOptions::default()
            },
        );
        assert!(
            !f.converged(),
            "perfectly separated Gamma(inverse) GLM must not converge (held φ {held:?})"
        );
        assert!(
            f.dispersion.is_nan(),
            "dispersion must be NaN on a failed fit, not the Gamma exponential special case 1.0 \
         or a held φ (held φ {held:?}): {}",
            f.dispersion
        );
    }
}

/// Gamma inverse link with a small mean, gated against frozen R
/// `glm(Gamma(link="inverse"))` (`validation/goldens/sim_scale_gamma_inv_glm.json`):
/// `y ~ x` on sim_scale_gamma_inv, where μ ≈ 0.01 so η = 1/μ ≈ 100. A flat
/// divergence cap of 30 on |η| would refuse this honest fit, which is why the
/// GLM guard skips this family/link pair. The oracle is sacred.
#[test]
fn fit_glm_gamma_inverse_small_mean_matches_r() {
    // From validation/goldens/sim_scale_gamma_inv_glm.json (SE at the Pearson φ̂).
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

/// Inverse-Gaussian precision weights: `wᵢ → c·wᵢ` leaves `loglik` invariant.
/// `D` scales by `c` (the mean-model IRLS fit doesn't move under an overall
/// weight rescale, so `D` is unchanged in shape and only inherits `c` from
/// the weights), so `φ̂ = D/n` scales by `c` too, `n·ln(2πφ̂)` moves by
/// `+n·ln c`, and `inv_gaussian_aic`'s `−Σ ln wᵢ` term moves by exactly
/// `−n·ln c` — the two shifts cancel. `dispersion` is the raw-weight Pearson
/// moment (`family::pearson_dispersion`) on this family; β/SE/dispersion are
/// checked too, as a sanity check alongside the loglik invariance this test
/// targets. `glm::DEVIANCE_TOL`'s relative stopping rule is scale-free for
/// `|deviance| ≫ 0.1` (true at every tested `c` here), so all three agree to
/// 1e-9 relative at every tested `c` (see
/// `fit_glm_gamma_weight_scale_invariant`'s doc for the mechanism). The
/// data are 300 rows of `y = 1 + 0.5·x + 0.05·(i mod 7)`, `x = i/300`, with a
/// row-varying `w`.
#[test]
fn fit_glm_inverse_gaussian_weight_scale_invariant() {
    let n = 300usize;
    let p = 2usize;
    let mut x = Vec::<f64>::with_capacity(n * p);
    let mut y = Vec::<f64>::with_capacity(n);
    let mut w = Vec::<f64>::with_capacity(n);
    for i in 0..n {
        let xi = (i as f64) / (n as f64);
        x.push(1.0);
        x.push(xi);
        y.push(1.0 + 0.5 * xi + 0.05 * ((i % 7) as f64));
        w.push(1.0 + 0.3 * ((i % 5) as f64));
    }
    let model = ModelSpec {
        family: Family::InverseGaussian {
            link: crate::InverseGaussianLink::Log,
        },
        re: None,
    };
    let fit_at = |c: f64| {
        let wc: Vec<f64> = w.iter().map(|&wi| c * wi).collect();
        fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1],
                weights: Some(wc),
                ..FitOptions::default()
            },
        )
    };
    let base = fit_at(1.0);
    assert!(base.converged());
    for &c in &[8.0_f64, 2.0_f64.powi(-6), 2.0_f64.powi(20)] {
        let f = fit_at(c);
        assert!(f.converged(), "c = {c}");
        let ll_rel = (f.loglik - base.loglik).abs() / base.loglik.abs();
        assert!(
            ll_rel < 1e-9,
            "c = {c}: loglik {} vs {}",
            f.loglik,
            base.loglik
        );
        for j in 0..p {
            let b_rel = (f.beta[j] - base.beta[j]).abs() / base.beta[j].abs();
            // Raw-weight IRLS mean-model fit, same tolerance note as
            // `fit_glm_gamma_weight_scale_invariant`.
            assert!(
                b_rel < 1e-9,
                "c = {c}: β[{j}] {} vs {} rel {b_rel}",
                f.beta[j],
                base.beta[j]
            );
            let se_rel = (f.se[j] - base.se[j]).abs() / base.se[j].abs();
            assert!(
                se_rel < 1e-9,
                "c = {c}: se[{j}] {} vs {} rel {se_rel}",
                f.se[j],
                base.se[j]
            );
        }
        let disp_rel = (f.dispersion - c * base.dispersion).abs() / (c * base.dispersion);
        assert!(
            disp_rel < 1e-9,
            "c = {c}: dispersion {} vs {c}·{} rel {disp_rel}",
            f.dispersion,
            base.dispersion
        );
    }
}

/// Inverse-Gaussian with random effects is not wired (the profiled
/// `inverse.gaussian()$aic` objective term is not built) and must fault at the
/// model-shape gate, before any workspace is allocated.
#[test]
#[should_panic(expected = "inverse-Gaussian mixed models are not implemented")]
fn fit_inverse_gaussian_mixed_faults() {
    let n = 8usize;
    let p = 1usize;
    let x = vec![1.0; n];
    let y: Vec<f64> = (1..=n).map(|k| k as f64).collect();
    let ids = GroupIds {
        primary: vec![0, 0, 0, 0, 1, 1, 1, 1],
        extra: vec![],
    };
    let model = ModelSpec {
        family: Family::InverseGaussian {
            link: crate::InverseGaussianLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let _ = fit_cold(&x, &y, n, p, &model, &ids, &FitOptions::default());
}
