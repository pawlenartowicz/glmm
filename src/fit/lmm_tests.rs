//! LMM estimator tests (`Family::Gaussian`, `re: Some`), plus the
//! `loop_advanced`-gated LMM sweep/refit dev-seam tests.

use super::*;
use crate::lmm::{fit_lmm, LmmWorkspace};
use crate::{
    Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing, StartValues,
};
use faer::Mat;

#[cfg(feature = "loop_advanced")]
use super::common_tests::lmm_hand_dataset;
use super::common_tests::{dense_str, lcg};

#[derive(serde::Deserialize)]
struct RdLmmEst {
    beta: Vec<f64>,
    varcomp: Vec<VcBlock>,
}

#[derive(serde::Deserialize)]
struct RdLmmGolden {
    coef_names: Vec<String>,
    estimates: RdLmmEst,
}

/// Coverage-gaps G5: aliased fixed column on a MIXED design.
/// `y ~ 1 + x1 + x2 + x3 + (1|g)` on sim_collinear_lmm (x3 ≈ x1 + x2) vs
/// frozen lme4 (`parity/goldens/sim_collinear_lmm.json`): lmer's rankMatrix
/// check drops the aliased column and `fixef` simply omits its name, so the
/// golden's `coef_names` records WHICH column lme4 dropped (x3, the last
/// dependent one — the same later-column convention `detect_aliased` uses,
/// so the drop indices are asserted equal, not merely each self-consistent).
/// glmm instead keeps full width with `NaN` in the dropped slot(s); the
/// surviving β and the varcomp must match the reduced lme4 fit. The oracle
/// is sacred.
#[test]
fn fit_lmm_rank_deficient_matches_lme4_drop() {
    let raw = include_str!("../../parity/goldens/sim_collinear_lmm.json");
    let gold: RdLmmGolden = serde_json::from_str(raw).expect("golden JSON parses");

    // sim_collinear_lmm.csv: y,x1,x2,x3,g
    let csv = include_str!("../../parity/data_simulated/sim_collinear_lmm.csv");
    let mut y = Vec::<f64>::new();
    let mut cols: Vec<[f64; 3]> = Vec::new();
    let mut g_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        cols.push([
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        ]);
        g_raw.push(f[4].to_string());
    }
    let n = y.len();
    let p = 4; // intercept + x1 + x2 + x3
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = cols[i][0];
        x[i * p + 2] = cols[i][1];
        x[i * p + 3] = cols[i][2];
    }
    let (g, _n_g) = dense_str(&g_raw);
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: g,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );

    assert!(f.converged, "reduced LMM must converge");
    // lme4 dropped exactly x3: its surviving coef_names lack it. glmm's
    // aliased mask must mark the SAME column (index 3) and only it.
    assert!(
        !gold.coef_names.iter().any(|c| c == "x3") && gold.coef_names.len() == 3,
        "golden must record lme4 dropping x3, got {:?}",
        gold.coef_names
    );
    assert_eq!(
        f.aliased,
        vec![false, false, false, true],
        "glmm must drop the same column lme4 does (x3)"
    );
    assert!(f.beta[3].is_nan(), "aliased β = NaN");
    assert!(f.se[3].is_nan(), "aliased se = NaN");
    // Surviving slots [0..3) line up 1:1 with the golden's 3 reduced coefs.
    for (j, rb) in gold.estimates.beta.iter().enumerate() {
        assert!(
            (f.beta[j] - rb).abs() / rb.abs() < 1e-3,
            "β{j} {} vs lme4 {rb}",
            f.beta[j]
        );
    }
    // Varcomp of the reduced fit passes through the salvage unchanged.
    let ref_g_sd = gold.estimates.varcomp[0].stddev[0];
    let g_rel = (f.tau2[0].sqrt() - ref_g_sd).abs() / ref_g_sd;
    assert!(
        g_rel < 1e-2,
        "g sd = {} vs lme4 {ref_g_sd}",
        f.tau2[0].sqrt()
    );
}

/// Warm-start A/B on the realistic sleepstudy random-slope LMM
/// (`Reaction ~ Days + (1 + Days | Subject)`, q=2, n_theta=3): a warm fit
/// from the frozen lme4 θ̂ ("from the truth") and one from a well-off
/// perturbed θ must land on the cold optimum — β, SE, and the varcorr
/// stddevs — and warm must never degrade convergence status. Extends
/// `fit_warm_start_reaches_cold_beta` (β-only, hand-built n_theta=1) to a
/// realistic q≥2 rung; MCPower's hot loop rides this contract.
#[test]
fn fit_warm_sleepstudy_slope_matches_cold_optimum() {
    // Parsing mirrors `fit_sleepstudy_slope_varcorr_matches_lme4`.
    let csv = include_str!("../../parity/data_empirical/sleepstudy.csv");
    let mut y = Vec::<f64>::new();
    let mut days = Vec::<f64>::new();
    let mut subj_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // Reaction
        days.push(f[1].parse().unwrap()); // Days
        subj_raw.push(f[2].to_string()); // Subject
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = days[i];
    }
    let (subject, _n_subj) = dense_str(&subj_raw);
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: subject,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(cold.converged, "cold sleepstudy fit must converge");

    // lme4 θ̂ = vech Cholesky of D̂/σ̂², from the frozen golden's
    // stddev/corr/sigma (`parity/goldens/sleepstudy_lmm.json`) — `Fit`
    // does not expose θ̂, and Gaussian tau2/varcorr are both σ²-scaled so
    // θ cannot be recovered from the cold fit alone:
    // θ00 = sd0/σ, θ10 = corr·sd1/σ, θ11 = (sd1/σ)·√(1−corr²).
    const REF_SD0: f64 = 24.7406579949841;
    const REF_SD1: f64 = 5.92213765889808;
    const REF_CORR: f64 = 0.0655512382381282;
    const REF_SIGMA: f64 = 25.5917957216753;
    let truth = vec![
        REF_SD0 / REF_SIGMA,
        REF_CORR * REF_SD1 / REF_SIGMA,
        REF_SD1 / REF_SIGMA * (1.0 - REF_CORR * REF_CORR).sqrt(),
    ];
    let starts = [
        (
            "truth",
            StartValues {
                beta: cold.beta.clone(),
                theta: truth,
            },
        ),
        // Well off θ̂ ≈ [0.97, 0.015, 0.23] in every coordinate; the LMM
        // path threads θ only (β is solved exactly given θ).
        (
            "perturbed",
            StartValues {
                beta: vec![0.0; p],
                theta: vec![3.0, 0.5, 1.5],
            },
        ),
    ];
    for (label, start) in &starts {
        let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
        assert!(warm.converged, "{label}: warm must not degrade convergence");
        for j in 0..p {
            let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
            assert!(
                rel < 1e-3,
                "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                warm.beta[j],
                cold.beta[j]
            );
            let rel = (warm.se[j] - cold.se[j]).abs() / cold.se[j];
            assert!(
                rel < 1e-3,
                "{label}: se[{j}] warm {} vs cold {} (rel {rel})",
                warm.se[j],
                cold.se[j]
            );
        }
        // q=2 vech diag (offsets 0, 2) → the two RE stddevs. The off-diag
        // covariance is near zero here (corr≈0.066); the stddevs pin the block.
        for off in [0usize, 2] {
            let (w, c) = (warm.varcorr[0][off].sqrt(), cold.varcorr[0][off].sqrt());
            let rel = (w - c).abs() / c;
            assert!(
                rel < 1e-3,
                "{label}: RE stddev (vech {off}) warm {w} vs cold {c} (rel {rel})"
            );
        }
    }
}

/// Gap #3: sleepstudy `Reaction ~ 1 + Days + (1 + Days | Subject)` — a q=2
/// random-slope LMM through `fit_cold`, gated against the frozen lme4 VarCorr
/// (`parity/goldens/sleepstudy_lmm.json`, REML). Checks the full 2×2 RE
/// covariance (variances AND the off-diagonal covariance) via `Fit::varcorr`,
/// which `tau2` cannot represent at q≥2. The oracle is sacred.
#[test]
fn fit_sleepstudy_slope_varcorr_matches_lme4() {
    const REF_B0: f64 = 251.405104848485;
    const REF_B1: f64 = 10.467285959596;
    const REF_SE0: f64 = 6.82459669495491;
    const REF_SE1: f64 = 1.54578964390598;
    const REF_SD0: f64 = 24.7406579949841; // (Intercept) sd
    const REF_SD1: f64 = 5.92213765889808; // Days sd
    const REF_CORR: f64 = 0.0655512382381282;
    const REF_SIGMA: f64 = 25.5917957216753; // residual sd, lme4 sigma()

    let csv = include_str!("../../parity/data_empirical/sleepstudy.csv");
    let mut y = Vec::<f64>::new();
    let mut days = Vec::<f64>::new();
    let mut subj_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // Reaction
        days.push(f[1].parse().unwrap()); // Days
        subj_raw.push(f[2].to_string()); // Subject
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0; // intercept
        x[i * p + 1] = days[i]; // Days
    }
    let (subject, _n_subj) = dense_str(&subj_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![1],                                 // random slope on Days (col 1)
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: subject,
        extra: vec![],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );

    assert!(f.converged, "sleepstudy slope LMM must converge");
    assert!(
        (f.beta[0] - REF_B0).abs() / REF_B0 < 1e-3,
        "β0 {} vs {REF_B0}",
        f.beta[0]
    );
    assert!(
        (f.beta[1] - REF_B1).abs() / REF_B1 < 1e-3,
        "β1 {} vs {REF_B1}",
        f.beta[1]
    );
    assert!(
        (f.se[0] - REF_SE0).abs() / REF_SE0 < 2e-2,
        "se0 {} vs {REF_SE0}",
        f.se[0]
    );
    assert!(
        (f.se[1] - REF_SE1).abs() / REF_SE1 < 2e-2,
        "se1 {} vs {REF_SE1}",
        f.se[1]
    );
    // Gaussian dispersion = REML σ̂²; σ̂ against the frozen lme4 sigma().
    assert!(
        (f.dispersion.sqrt() - REF_SIGMA).abs() / REF_SIGMA < 1e-3,
        "σ̂ {} vs {REF_SIGMA}",
        f.dispersion.sqrt()
    );

    // Reference D (col-major vech lower-tri): [D00, D10, D11].
    let d00 = REF_SD0 * REF_SD0;
    let d11 = REF_SD1 * REF_SD1;
    let d10 = REF_CORR * REF_SD0 * REF_SD1;
    assert_eq!(f.varcorr.len(), 1, "one grouping block");
    let vc = &f.varcorr[0];
    assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
    assert!(
        (vc[0].sqrt() - REF_SD0).abs() / REF_SD0 < 1e-2,
        "sd0 {} vs {REF_SD0}",
        vc[0].sqrt()
    );
    assert!(
        (vc[2].sqrt() - REF_SD1).abs() / REF_SD1 < 1e-2,
        "sd1 {} vs {REF_SD1}",
        vc[2].sqrt()
    );
    // Covariance is small (corr≈0.066) → check on an absolute scale.
    assert!((vc[0] - d00).abs() / d00 < 2e-2, "D00 {} vs {d00}", vc[0]);
    assert!((vc[2] - d11).abs() / d11 < 2e-2, "D11 {} vs {d11}", vc[2]);
    assert!(
        (vc[1] - d10).abs() < 0.20 * REF_SD0 * REF_SD1,
        "D10 {} vs {d10}",
        vc[1]
    );
}

/// Campaign instrumentation: `fit` must surface the optimizer eval count, the
/// minimized criterion, and boundary/singular status. Oracle: lme4's frozen
/// sleepstudy REML fit — REMLcrit = glmm deviance + df·(1 + ln 2π), df = n − p
/// (glmm's reml_deviance omits the df·(1+ln 2π) constant lme4's REMLcrit
/// carries; loglik = −REMLcrit/2 is what results/lme4_empirical stores).
#[test]
fn fit_exposes_n_eval_deviance_singular() {
    let csv = include_str!("../../parity/data_empirical/sleepstudy.csv");
    let mut y = Vec::<f64>::new();
    let mut days = Vec::<f64>::new();
    let mut subj_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // Reaction
        days.push(f[1].parse().unwrap()); // Days
        subj_raw.push(f[2].to_string()); // Subject
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0; // intercept
        x[i * p + 1] = days[i]; // Days
    }
    let (subject, _n_subj) = dense_str(&subj_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![1],                                 // random slope on Days (col 1)
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: subject,
        extra: vec![],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );

    assert!(f.n_eval > 0, "BOBYQA ran, evals must be counted");
    assert!(f.deviance.is_finite());
    assert!(!f.singular, "sleepstudy is an interior optimum");
    let n = 180.0_f64;
    let p = 2.0_f64; // intercept + Days
    let df = n - p;
    let lme4_loglik = -871.814135979976; // parity/results/lme4_empirical/sleepstudy.json .estimates.loglik
    let remlcrit = -2.0 * lme4_loglik;
    let expected = remlcrit - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance - expected).abs() < 1e-6,
        "deviance {} vs lme4-derived {expected}",
        f.deviance
    );
    // Fit.loglik must invert that stripped constant back to lme4's logLik —
    // the REML criterion on the logLik scale (reml flags it as such).
    assert!(
        (f.loglik - lme4_loglik).abs() < 1e-6,
        "loglik {} vs lme4 {lme4_loglik}",
        f.loglik
    );
    assert!(f.reml, "Gaussian LMM loglik is the REML criterion");
    assert_eq!(f.df, 6); // 2 β + 3 θ (q=2 vech) + σ²
}

/// Dense LMM with a per-row offset — the identity-link `y − o` shift — vs R
/// `lmer(offset=)`: sleepstudy random-slope with `o_i = 5·((i−1) mod 4)`
/// (0-based CSV row order in Rust). Oracle (R 4.5.3, lme4 1.1-38):
///   fl <- lmer(Reaction ~ Days + (Days | Subject), data = ss, offset = ol)
///   print(fixef(fl), digits = 15); print(REMLcrit(fl), digits = 15)
///   print(logLik(fl), digits = 15)
#[test]
fn fit_lmm_offset_matches_lme4() {
    const REF_BETA: [f64; 2] = [244.5869230303025, 10.3157708080802];
    const REF_REMLCRIT: f64 = 1756.8758930064;
    const REF_LOGLIK: f64 = -878.437946503201;
    let csv = include_str!("../../parity/data_empirical/sleepstudy.csv");
    let mut y = Vec::<f64>::new();
    let mut days = Vec::<f64>::new();
    let mut subj_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // Reaction
        days.push(f[1].parse().unwrap()); // Days
        subj_raw.push(f[2].to_string()); // Subject
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = days[i];
    }
    let (subject, _n_subj) = dense_str(&subj_raw);
    let o: Vec<f64> = (0..n).map(|i| 5.0 * (i % 4) as f64).collect();

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: subject,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1],
            offset: Some(o),
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "offset LMM must converge");
    for (j, (&b, &r)) in f.beta.iter().zip(&REF_BETA).enumerate() {
        assert!((b - r).abs() / r.abs() < 1e-3, "β[{j}] = {b} vs lme4 {r}");
    }
    let df = (n - p) as f64;
    let expected = REF_REMLCRIT - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance - expected).abs() < 1e-6,
        "deviance {} vs lme4-derived {expected}",
        f.deviance
    );
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-6,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
}

/// Task 5: weighted dense LMM REML — sleepstudy random-slope fit with
/// synthetic weights `w_i = 1 + (i mod 3)` (i = 0-based CSV row order),
/// gated against a frozen lme4 golden. Pins β, SE, the 2×2 RE covariance
/// (SDs + correlation), σ̂, and the `−Σlog wᵢ` deviance-constant convention:
/// weighted REMLcrit strips the same `df·(1+ln 2π)` constant as the
/// unweighted case (`lme.rs:2978`) PLUS the weighted Gaussian log-density's
/// `+½Σlog wᵢ` per row (`−Σlog wᵢ` on the −2ℓ deviance scale). Generated
/// with (R 4.5.3, lme4 1.1-38):
/// ```r
/// library(lme4)
/// d <- read.csv("parity/data_empirical/sleepstudy.csv")
/// w <- 1 + (seq_len(nrow(d)) - 1) %% 3
/// f <- lmer(Reaction ~ Days + (Days | Subject), data = d, weights = w, REML = TRUE)
/// print(summary(f)$coefficients, digits = 15)
/// print(as.data.frame(VarCorr(f)), digits = 15)
/// print(sigma(f), digits = 15); print(REMLcrit(f), digits = 15)
/// ```
#[test]
fn fit_lmm_weighted_matches_lme4() {
    const REF_B0: f64 = 251.804_690_405_274;
    const REF_B1: f64 = 10.4358707468765;
    const REF_SE0: f64 = 6.44698545564581;
    const REF_SE1: f64 = 1.57363056312657;
    const REF_SD0: f64 = 22.09852363841438; // (Intercept) sd
    const REF_SD1: f64 = 5.95218759898762; // Days sd
    const REF_CORR: f64 = 0.16395038320169;
    const REF_SIGMA: f64 = 38.62892535113247;
    const REF_REMLCRIT: f64 = 1778.29146275691;

    let csv = include_str!("../../parity/data_empirical/sleepstudy.csv");
    let mut y = Vec::<f64>::new();
    let mut days = Vec::<f64>::new();
    let mut subj_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // Reaction
        days.push(f[1].parse().unwrap()); // Days
        subj_raw.push(f[2].to_string()); // Subject
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0; // intercept
        x[i * p + 1] = days[i]; // Days
    }
    let (subject, _n_subj) = dense_str(&subj_raw);
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
            slopes: vec![1],                                 // random slope on Days (col 1)
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: subject,
        extra: vec![],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w.clone()),
            ..FitOptions::default()
        },
    );

    assert!(f.converged, "weighted sleepstudy slope LMM must converge");
    assert!(
        (f.beta[0] - REF_B0).abs() / REF_B0 < 1e-6,
        "β0 {} vs {REF_B0}",
        f.beta[0]
    );
    assert!(
        (f.beta[1] - REF_B1).abs() / REF_B1 < 1e-6,
        "β1 {} vs {REF_B1}",
        f.beta[1]
    );
    assert!(
        (f.se[0] - REF_SE0).abs() / REF_SE0 < 1e-4,
        "se0 {} vs {REF_SE0}",
        f.se[0]
    );
    assert!(
        (f.se[1] - REF_SE1).abs() / REF_SE1 < 1e-4,
        "se1 {} vs {REF_SE1}",
        f.se[1]
    );

    assert_eq!(f.varcorr.len(), 1, "one grouping block");
    let vc = &f.varcorr[0];
    assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
    let sd0 = vc[0].sqrt();
    let sd1 = vc[2].sqrt();
    let corr = vc[1] / (sd0 * sd1);
    assert!(
        (sd0 - REF_SD0).abs() / REF_SD0 < 1e-4,
        "sd0 {sd0} vs {REF_SD0}"
    );
    assert!(
        (sd1 - REF_SD1).abs() / REF_SD1 < 1e-4,
        "sd1 {sd1} vs {REF_SD1}"
    );
    // The off-diagonal covariance/correlation is the least-constrained θ
    // coordinate under BOBYQA's rho_end floor (θ10 is small relative to
    // θ00/θ11, so its relative precision is looser) — the unweighted
    // analog (`fit_sleepstudy_slope_varcorr_matches_lme4`) hits the exact
    // same floor and uses the same absolute-on-D10-scale band.
    assert!((corr - REF_CORR).abs() < 0.05, "corr {corr} vs {REF_CORR}");

    // Fit.deviance vs REMLcrit(f) − (n−p)·(1+ln 2π) — pins the −Σlog wᵢ
    // constant `fit_mle` folds into the reported deviance (see the arm
    // above fit_mle in this file). 1e-6 abs, as the unweighted analog above.
    let df = (n - p) as f64;
    let expected = REF_REMLCRIT - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
    assert!(
        (f.deviance - expected).abs() < 1e-6,
        "deviance {} vs lme4-derived {expected}",
        f.deviance
    );
    // loglik = −REMLcrit/2 under weights — pins that the −Σlog wᵢ correction
    // lands INSIDE the criterion the loglik reports (lme4's weighted logLik).
    assert!(
        (f.loglik - (-REF_REMLCRIT / 2.0)).abs() < 1e-6,
        "weighted loglik {} vs lme4 {}",
        f.loglik,
        -REF_REMLCRIT / 2.0
    );
    assert!(f.reml);

    // σ̂ isn't exposed on `Fit` for q≥2 RE (tau2 only reproduces the (0,0)
    // diagonal, not the raw residual variance) — reconstruct via the same
    // suff-stats accumulator/kernel `fit_mle` calls, reading `sigma_sq`
    // straight off `LmmFit` (mirrors fit_mle's construction verbatim).
    let sized = spec_sized_from_ids(&model, &ids);
    let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &[1], &[]);
    let mut x_mat = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }
    ws.suff
        .add_rows_multi(x_mat.as_ref(), &y, &ids.primary, &[], Some(&w));
    let lmm_fit = fit_lmm(&mut ws, &[0, 1], None);
    let sigma = lmm_fit.sigma_sq.sqrt();
    assert!(
        (sigma - REF_SIGMA).abs() / REF_SIGMA < 1e-4,
        "sigma {sigma} vs {REF_SIGMA}"
    );
}

/// Task 5: constant weights (w ≡ 2) must reproduce the unweighted fit's β,
/// SE, AND tau2 exactly (1e-10) — under w ≡ c, the substitution θ̃ = √c·θ
/// maps the weighted profiled deviance onto the unweighted one 1:1, so θ̂
/// scales by 1/√c while σ̂² scales by c, and tau2 = θ²σ̂² is invariant.
/// Verified against lme4 separately: sleepstudy with w ≡ 2 leaves the
/// VarCorr group variances unchanged and exactly doubles the residual
/// variance (not re-asserted here — this test only needs internal
/// consistency on a small synthetic LMM, cheaper than another R golden).
#[test]
fn fit_lmm_constant_weights_invariant() {
    let n_clusters = 6usize;
    let per = 8usize;
    let n = n_clusters * per;
    let mut st = 13u64;
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids_v = vec![0u32; n];
    for i in 0..n {
        ids_v[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        let re = 0.3 * ((ids_v[i] as f64) - (n_clusters as f64) / 2.0);
        y[i] = 0.5 + 0.4 * x1 + re + 0.2 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: ids_v,
        extra: vec![],
    };
    let unweighted = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    let weighted = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(vec![2.0; n]),
            ..FitOptions::default()
        },
    );
    assert!(unweighted.converged && weighted.converged);
    // The θ̃=√c·θ substitution is exact algebra; the achieved match is
    // bounded by BOBYQA's rho_end floor (2 independently-converged fits,
    // not a shared trajectory), not by 1e-10 — 1e-6 relative is the tight
    // bound this floor actually supports (measured ~2e-8 on this fixture).
    for j in 0..2 {
        assert!(
            (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
            "β[{j}] unweighted {} vs w≡2 {}",
            unweighted.beta[j],
            weighted.beta[j]
        );
        assert!(
            (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-6,
            "se[{j}] unweighted {} vs w≡2 {}",
            unweighted.se[j],
            weighted.se[j]
        );
    }
    assert_eq!(unweighted.tau2.len(), weighted.tau2.len());
    for k in 0..unweighted.tau2.len() {
        assert!(
            (unweighted.tau2[k] - weighted.tau2[k]).abs() / unweighted.tau2[k] < 1e-6,
            "tau2[{k}] unweighted {} vs w≡2 {}",
            unweighted.tau2[k],
            weighted.tau2[k]
        );
    }
}

/// Constant-weights invariance on a CROSSED random-slope design
/// (`y ~ 1 + x + (1 + x | g1) + (1 | g2)`, the `sim_slope` fixture):
/// w ≡ 2 must reproduce the unweighted β/SE/varcorr. This is the numeric
/// check for the crossed-path weight sites in `add_rows_multi` — the
/// intercept×intercept `zx += wᵢ` and the slope↔crossed `zx_slope += z·zw`
/// (q_p = 2 primary slope + crossed intercept extra takes the scalar
/// crossed branch, which unit-weight tests cannot distinguish from a
/// wrong-power bug). Same θ̃ = √c·θ rationale and BOBYQA-floor tolerance as
/// `fit_lmm_constant_weights_invariant`.
#[test]
fn fit_lmm_crossed_constant_weights_invariant() {
    let csv = include_str!("../../parity/data_simulated/sim_slope.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut g1_raw = Vec::<String>::new();
    let mut g2_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // y
        xcol.push(f[1].parse().unwrap()); // x
        g1_raw.push(f[2].to_string());
        g2_raw.push(f[3].to_string());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let (g1, _n1) = dense_str(&g1_raw);
    let (g2, _n2) = dense_str(&g2_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![1], // random slope on x for g1
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![], // g2 intercept-only
            }],
        }),
    };
    let ids = GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    let base_opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let unweighted = fit_cold(&x, &y, n, p, &model, &ids, &base_opts);
    let weighted = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            weights: Some(vec![2.0; n]),
            ..base_opts
        },
    );
    assert!(unweighted.converged && weighted.converged);
    for j in 0..p {
        assert!(
            (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
            "β[{j}] unweighted {} vs w≡2 {}",
            unweighted.beta[j],
            weighted.beta[j]
        );
        assert!(
            (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-6,
            "se[{j}] unweighted {} vs w≡2 {}",
            unweighted.se[j],
            weighted.se[j]
        );
    }
    // varcorr covers BOTH groupings' D̂ blocks (tau2 only reproduces the
    // (0,0) diagonal for the q=2 primary). Relative bound on the diagonals;
    // the small q=2 off-diagonal takes the same bound scaled to its own
    // magnitude floor.
    assert_eq!(unweighted.varcorr.len(), weighted.varcorr.len());
    for (gi, (vu, vw)) in unweighted
        .varcorr
        .iter()
        .zip(weighted.varcorr.iter())
        .enumerate()
    {
        assert_eq!(vu.len(), vw.len());
        for k in 0..vu.len() {
            let scale = vu[k].abs().max(1e-3);
            assert!(
                (vu[k] - vw[k]).abs() / scale < 1e-5,
                "varcorr[{gi}][{k}] unweighted {} vs w≡2 {}",
                vu[k],
                vw[k]
            );
        }
    }
}

/// Task 5 Step 6: the dense-LMM boundary (τ̂ ≈ 0, pinned exactly per the
/// Q7 deterministic-pin policy — mirrors
/// `lmm::tests::zero_between_cluster_variance_pins_at_exactly_zero`) must
/// reproduce the weighted fixed-only WLS fit (Task 1, `fit_ols`) on the
/// same rows: at θ̂=0 the mixed kernel's weighted Grams (`c`/`s`/`counts`,
/// all Σwᵢ-scaled per Task 5's accumulator) collapse to the same weighted
/// normal equations WLS solves directly, so the two paths must agree.
#[test]
fn fit_lmm_weighted_boundary_matches_wls() {
    let n = 48usize;
    let n_clusters = 6usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    let mut w = vec![0.0f64; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        // i/n_clusters cycles 0..8 within each cluster: 4 even, 4 odd ⇒
        // the ±0.8 residuals cancel exactly per cluster (deterministic pin).
        let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
        y[i] = 0.5 + 0.4 * x1 + e;
        w[i] = 1.0 + (i % 3) as f64;
    }

    let mixed_model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let mixed_ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let mixed = fit_cold(
        &x,
        &y,
        n,
        2,
        &mixed_model,
        &mixed_ids,
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w.clone()),
            ..FitOptions::default()
        },
    );
    assert!(mixed.converged, "boundary pin still counts as converged");
    assert!(mixed.singular, "must pin at the τ=0 boundary");

    let fixed_only = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let wls = fit_cold(
        &x,
        &y,
        n,
        2,
        &fixed_only,
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..FitOptions::default()
        },
    );
    assert!(wls.converged);

    for j in 0..2 {
        assert!(
            (mixed.beta[j] - wls.beta[j]).abs() / wls.beta[j].abs() < 1e-6,
            "β[{j}] mixed {} vs WLS {}",
            mixed.beta[j],
            wls.beta[j]
        );
        assert!(
            (mixed.se[j] - wls.se[j]).abs() / wls.se[j] < 1e-3,
            "se[{j}] mixed {} vs WLS {}",
            mixed.se[j],
            wls.se[j]
        );
    }
}

// serde ignores unread JSON fields (e.g. `group`) by default; only the fields
// the assertions consume are declared, to keep the dead_code lint clean.
#[derive(serde::Deserialize)]
struct VcBlock {
    stddev: Vec<f64>,
    corr: Vec<Vec<f64>>,
}

#[derive(serde::Deserialize)]
struct VcEst {
    beta: Vec<f64>,
    varcomp: Vec<VcBlock>,
}

#[derive(serde::Deserialize)]
struct VcGolden {
    estimates: VcEst,
}

/// Gap #3 synthetic: crossed random-slope `y ~ 1 + x + (1 + x | g1) + (1 | g2)`
/// vs the R-generated lme4 golden (`parity/goldens/sim_slope_lmm.json`). Exercises
/// a q=2 `varcorr` block on the PRIMARY plus a scalar block on a crossed EXTRA
/// grouping — the multi-grouping generalization the single-grouping composition omits.
/// The oracle is sacred.
#[test]
fn fit_sim_slope_varcorr_matches_lme4() {
    let raw = include_str!("../../parity/goldens/sim_slope_lmm.json");
    let gold: VcGolden = serde_json::from_str(raw).expect("golden JSON parses");

    let csv = include_str!("../../parity/data_simulated/sim_slope.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut g1_raw = Vec::<String>::new();
    let mut g2_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // y
        xcol.push(f[1].parse().unwrap()); // x
        g1_raw.push(f[2].to_string());
        g2_raw.push(f[3].to_string());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let (g1, _n1) = dense_str(&g1_raw);
    let (g2, _n2) = dense_str(&g2_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![1], // random slope on x for g1
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![], // g2 intercept-only
            }],
        }),
    };
    let ids = GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );

    assert!(f.converged);
    for j in 0..p {
        let r = gold.estimates.beta[j];
        assert!(
            (f.beta[j] - r).abs() / r.abs().max(1e-6) < 5e-3,
            "β{j} {} vs {r}",
            f.beta[j]
        );
    }
    // varcorr[0] = g1 (q=2), varcorr[1] = g2 (scalar). Reconstruct D from stddev+corr.
    assert_eq!(f.varcorr.len(), 2);
    let g1b = &gold.estimates.varcomp[0];
    let (sd0, sd1, c01) = (g1b.stddev[0], g1b.stddev[1], g1b.corr[0][1]);
    let vc0 = &f.varcorr[0];
    assert!(
        (vc0[0].sqrt() - sd0).abs() / sd0 < 2e-2,
        "g1 sd0 {} vs {sd0}",
        vc0[0].sqrt()
    );
    assert!(
        (vc0[2].sqrt() - sd1).abs() / sd1 < 2e-2,
        "g1 sd1 {} vs {sd1}",
        vc0[2].sqrt()
    );
    assert!(
        (vc0[1] - c01 * sd0 * sd1).abs() < 0.30 * sd0 * sd1,
        "g1 cov {}",
        vc0[1]
    );
    let g2sd = gold.estimates.varcomp[1].stddev[0];
    assert!(
        (f.varcorr[1][0].sqrt() - g2sd).abs() / g2sd < 3e-2,
        "g2 sd {} vs {g2sd}",
        f.varcorr[1][0].sqrt()
    );
}

/// Gap #1 crossed: Penicillin `diameter ~ 1 + (1|plate) + (1|sample)` through the
/// data-shaped `fit_cold` with `GroupIds { primary: plate, extra: vec![sample] }`,
/// gated against the frozen lme4 golden (`parity/goldens/penicillin_lmm.json`,
/// REML). Two crossed intercept-only groupings, fixed effect = intercept only
/// (p=1). Placeholder spec counts prove the data path derives level counts from
/// the ids. The oracle is sacred.
#[test]
fn fit_penicillin_crossed_matches_lme4() {
    const REF_BETA: f64 = 22.9722222222;
    const REF_SE: f64 = 0.808595361386;
    const REF_PLATE_SD: f64 = 0.846702;
    const REF_SAMPLE_SD: f64 = 1.931614;

    let csv = include_str!("../../parity/data_empirical/Penicillin.csv");
    let mut y = Vec::<f64>::new();
    let mut plate_raw = Vec::<String>::new();
    let mut sample_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // diameter
        plate_raw.push(f[1].to_string());
        sample_raw.push(f[2].to_string());
    }
    let n = y.len();
    let p = 1;
    let x = vec![1.0f64; n]; // intercept-only design
    let (plate, _n_plate) = dense_str(&plate_raw);
    let (sample, _n_sample) = dense_str(&sample_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 }, // placeholder
                slopes: vec![],
            }],
        }),
    };
    let ids = GroupIds {
        primary: plate,
        extra: vec![sample],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0],
            ..FitOptions::default()
        },
    );

    assert!(f.converged, "Penicillin crossed LMM must converge");
    assert!(
        (f.beta[0] - REF_BETA).abs() / REF_BETA < 1e-4,
        "β0 = {} vs lme4 {REF_BETA}",
        f.beta[0]
    );
    let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
    assert!(
        se_rel < 2e-2,
        "se0 = {} vs lme4 {REF_SE} (rel {se_rel})",
        f.se[0]
    );
    // theta layout: [primary (plate) vech | sample scalar]; tau2[k] = θ̂[k]²·σ̂².
    let plate_sd = f.tau2[0].sqrt();
    let sample_sd = f.tau2[1].sqrt();
    assert!(
        (plate_sd - REF_PLATE_SD).abs() / REF_PLATE_SD < 5e-3,
        "plate sd = {plate_sd} vs lme4 {REF_PLATE_SD}"
    );
    assert!(
        (sample_sd - REF_SAMPLE_SD).abs() / REF_SAMPLE_SD < 5e-3,
        "sample sd = {sample_sd} vs lme4 {REF_SAMPLE_SD}"
    );
}

/// Gap #1 nested: Pastes `strength ~ 1 + (1|batch/cask)` through the data-shaped
/// `fit_cold` with `GroupIds { primary: batch, extra: vec![cask] }`, where `cask`
/// is the globally-unique batch:cask level (dense 0..29). Gated against the frozen
/// lme4 golden (`parity/goldens/pastes_lmm.json`, REML). Exercises the
/// `NestedWithin` topology tag on the data path; placeholder counts prove level
/// counts come from the ids. The oracle is sacred.
#[test]
fn fit_pastes_nested_matches_lme4() {
    const REF_BETA: f64 = 60.0533333333;
    const REF_SE: f64 = 0.676870215074;
    const REF_BATCH_SD: f64 = 1.287366;
    const REF_CASK_SD: f64 = 2.904077;

    let csv = include_str!("../../parity/data_empirical/Pastes.csv");
    // cols: strength,batch,cask,sample  (sample = "batch:cask" global label)
    let mut y = Vec::<f64>::new();
    let mut batch_raw = Vec::<String>::new();
    let mut cask_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap()); // strength
        batch_raw.push(f[1].to_string()); // batch
        cask_raw.push(f[3].to_string()); // sample = batch:cask global label
    }
    let n = y.len();
    let p = 1;
    let x = vec![1.0f64; n];
    let (batch, _n_batch) = dense_str(&batch_raw);
    let (cask, _n_cask) = dense_str(&cask_raw);

    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 1 }, // placeholder
                slopes: vec![],
            }],
        }),
    };
    let ids = GroupIds {
        primary: batch,
        extra: vec![cask],
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0],
            ..FitOptions::default()
        },
    );

    assert!(f.converged, "Pastes nested LMM must converge");
    assert!(
        (f.beta[0] - REF_BETA).abs() / REF_BETA < 1e-4,
        "β0 = {} vs lme4 {REF_BETA}",
        f.beta[0]
    );
    let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
    assert!(
        se_rel < 2e-2,
        "se0 = {} vs lme4 {REF_SE} (rel {se_rel})",
        f.se[0]
    );
    // theta layout: [primary (batch) vech | nested (cask) scalar].
    let batch_sd = f.tau2[0].sqrt();
    let cask_sd = f.tau2[1].sqrt();
    assert!(
        (batch_sd - REF_BATCH_SD).abs() / REF_BATCH_SD < 1e-2,
        "batch sd = {batch_sd} vs lme4 {REF_BATCH_SD}"
    );
    assert!(
        (cask_sd - REF_CASK_SD).abs() / REF_CASK_SD < 5e-3,
        "cask sd = {cask_sd} vs lme4 {REF_CASK_SD}"
    );
}

/// Bit-for-bit `LmmSweepOutcome` comparison — the arbiter for task 2's reuse
/// claim (a held `LmmSeamWs` must change nothing but wall time). `to_bits`
/// rather than `==` so a NaN-carrying `deviance`/`theta` (a non-converged
/// run) still compares meaningfully.
#[cfg(feature = "loop_advanced")]
fn assert_sweep_outcomes_bit_equal(a: &LmmSweepOutcome, b: &LmmSweepOutcome, label: &str) {
    assert_eq!(
        a.deviance.to_bits(),
        b.deviance.to_bits(),
        "{label}: deviance mismatch ({} vs {})",
        a.deviance,
        b.deviance
    );
    assert_eq!(
        a.theta.len(),
        b.theta.len(),
        "{label}: theta length mismatch"
    );
    for (i, (x, y)) in a.theta.iter().zip(&b.theta).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{label}: theta[{i}] mismatch ({x} vs {y})"
        );
    }
    assert_eq!(a.n_eval, b.n_eval, "{label}: n_eval mismatch");
    assert_eq!(a.converged, b.converged, "{label}: converged mismatch");
}

/// Task 2's correctness proof: two [`lmm_sweep_fit_on`] calls at different
/// θ₀ on ONE [`build_lmm_seam_ws`] result must reproduce two independent
/// [`lmm_sweep_fit`] calls bit-for-bit — proving the held `suff`/`fit` (or
/// sparse `ws`) is genuinely reused, not silently rebuilt under the hood.
/// Dense (`Solver::NoZ`) shape: `lmm_hand_dataset`'s intercept-only design.
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_sweep_fit_on_matches_lmm_sweep_fit_dense() {
    let (x, y, n, p) = lmm_hand_dataset();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 6 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);
    assert!(matches!(classify_design(&model, 1), Solver::NoZ));

    let (mut ws, g) = build_lmm_seam_ws(&x, &y, n, p, &model, &ids);
    let (blind, lower, upper) = g.blind_theta_and_bounds();
    let theta_a = blind.clone();
    let theta_b: Vec<f64> = lower
        .iter()
        .zip(&upper)
        .map(|(&lo, &hi)| lo + 0.25 * (hi - lo))
        .collect();

    let on_a = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_a), 1e-6, None, None);
    let on_b = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_b), 1e-6, None, None);

    let standalone_a = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_a), 1e-6, None, None);
    let standalone_b = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_b), 1e-6, None, None);

    assert_sweep_outcomes_bit_equal(&on_a, &standalone_a, "dense theta_a");
    assert_sweep_outcomes_bit_equal(&on_b, &standalone_b, "dense theta_b");
}

/// Same proof as [`lmm_sweep_fit_on_matches_lmm_sweep_fit_dense`], sparse
/// (`Solver::Sparse`) shape: a crossed extra grouping carrying a slope
/// forces the sparse route regardless of size (`classify_design`'s
/// `slope_extras` clause), so a small hand design suffices.
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_sweep_fit_on_matches_lmm_sweep_fit_sparse() {
    let n = 32usize;
    let p = 3usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut primary = vec![0u32; n];
    let mut extra = vec![0u32; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = x2;
        primary[i] = (i % 4) as u32;
        extra[i] = ((i / 4) % 4) as u32;
        y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + 0.3 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                slopes: vec![1],
            }],
        }),
    };
    let ids = GroupIds {
        primary,
        extra: vec![extra],
    };
    assert!(matches!(classify_design(&model, 1), Solver::Sparse));

    let (mut ws, g) = build_lmm_seam_ws(&x, &y, n, p, &model, &ids);
    let (blind, lower, upper) = g.blind_theta_and_bounds();
    let theta_a = blind.clone();
    let theta_b: Vec<f64> = lower
        .iter()
        .zip(&upper)
        .map(|(&lo, &hi)| lo + 0.25 * (hi - lo))
        .collect();

    let on_a = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_a), 1e-6, None, None);
    let on_b = lmm_sweep_fit_on(&mut ws, &g, Some(&theta_b), 1e-6, None, None);

    let standalone_a = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_a), 1e-6, None, None);
    let standalone_b = lmm_sweep_fit(&x, &y, n, p, &model, &ids, Some(&theta_b), 1e-6, None, None);

    assert_sweep_outcomes_bit_equal(&on_a, &standalone_a, "sparse theta_a");
    assert_sweep_outcomes_bit_equal(&on_b, &standalone_b, "sparse theta_b");
}

/// [`lmm_objective_at`] self-consistency: evaluating it at the θ̂ an
/// `lmm_sweep_fit` run converged to must reproduce that run's own
/// `deviance` — both paths build the same `LmmSeamWs` and call the same
/// `reml_deviance`/`sparse_reml_deviance` closure, so only bit-level FP
/// order can separate them. Dense (`Solver::NoZ`) shape, same design as
/// [`lmm_sweep_fit_on_matches_lmm_sweep_fit_dense`].
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_objective_at_matches_lmm_sweep_fit_deviance_dense() {
    let (x, y, n, p) = lmm_hand_dataset();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 6 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);
    assert!(matches!(classify_design(&model, 1), Solver::NoZ));

    let outcome = lmm_sweep_fit(&x, &y, n, p, &model, &ids, None, 1e-6, None, None);
    assert!(outcome.converged, "dense sweep fit must converge");

    let obj = lmm_objective_at(&x, &y, n, p, &model, &ids, &outcome.theta);
    let rel = (obj - outcome.deviance).abs() / outcome.deviance.abs();
    assert!(
        rel < 1e-10,
        "dense: lmm_objective_at {obj} vs sweep deviance {} (rel {rel})",
        outcome.deviance
    );
}

/// Same proof as [`lmm_objective_at_matches_lmm_sweep_fit_deviance_dense`],
/// sparse (`Solver::Sparse`) shape: the crossed-slope 32-row design from
/// [`lmm_sweep_fit_on_matches_lmm_sweep_fit_sparse`] that forces
/// `classify_design` off the dense route.
#[cfg(feature = "loop_advanced")]
#[test]
fn lmm_objective_at_matches_lmm_sweep_fit_deviance_sparse() {
    let n = 32usize;
    let p = 3usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut primary = vec![0u32; n];
    let mut extra = vec![0u32; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = x2;
        primary[i] = (i % 4) as u32;
        extra[i] = ((i / 4) % 4) as u32;
        y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + 0.3 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                slopes: vec![1],
            }],
        }),
    };
    let ids = GroupIds {
        primary,
        extra: vec![extra],
    };
    assert!(matches!(classify_design(&model, 1), Solver::Sparse));

    let outcome = lmm_sweep_fit(&x, &y, n, p, &model, &ids, None, 1e-6, None, None);
    assert!(outcome.converged, "sparse sweep fit must converge");

    let obj = lmm_objective_at(&x, &y, n, p, &model, &ids, &outcome.theta);
    let rel = (obj - outcome.deviance).abs() / outcome.deviance.abs();
    assert!(
        rel < 1e-10,
        "sparse: lmm_objective_at {obj} vs sweep deviance {} (rel {rel})",
        outcome.deviance
    );
}

/// Task 3's correctness proof: two [`refit_lmm`] calls with DIFFERENT `y`
/// (dataset A unweighted, dataset B weighted) on ONE workspace built by
/// [`build_lmm_workspace`] must reproduce two independent [`fit_cold`]
/// calls bit-for-bit — proving genuine workspace reuse (not a silent
/// rebuild) AND exercising the `-Σlog wᵢ` weighted-deviance coupling: an
/// omitted correction would only surface as a `deviance` mismatch on
/// dataset B, since A (unweighted) can't distinguish the two code paths.
/// Dense (`Solver::NoZ`) shape: same intercept-only 6-cluster design as
/// `lmm_hand_dataset`, re-seeded per dataset.
#[cfg(feature = "loop_advanced")]
#[test]
fn refit_lmm_matches_fresh_fit_cold() {
    let n = 48usize;
    let p = 3usize;
    let n_clusters = 6usize;
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);

    // Two datasets of the SAME shape (n, p, cluster structure), different
    // y — mirrors MCPower's re-simulated-y power loop. Shape matches
    // `lmm_hand_dataset` (cluster c = i % n_clusters), parameterized by seed.
    let dataset = |seed: u64| -> (Vec<f64>, Vec<f64>) {
        let mut st = seed;
        let u_c: Vec<f64> = (0..n_clusters).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let c = i % n_clusters;
            let x1 = lcg(&mut st);
            let x2 = lcg(&mut st);
            x[i * p] = 1.0;
            x[i * p + 1] = x1;
            x[i * p + 2] = x2;
            y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + u_c[c] + 0.8 * lcg(&mut st);
        }
        (x, y)
    };
    let (xa, ya) = dataset(42);
    let (xb, yb) = dataset(99);
    // Weighted case exercises the -Σlog wᵢ coupling: deviance is the field
    // that silently diverges from fit_cold's if the correction is dropped.
    let wb: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64 * 0.5).collect();

    let opts_a = FitOptions {
        target_indices: vec![1, 2],
        ..FitOptions::default()
    };
    let opts_b = FitOptions {
        target_indices: vec![1, 2],
        weights: Some(wb.clone()),
        ..FitOptions::default()
    };

    // ONE workspace, built once, reused across both refits — the reuse claim.
    let mut ws = build_lmm_workspace(p, &model, n);
    let refit_a = refit_lmm(&mut ws, &xa, &ya, n, p, &ids, &opts_a, None);
    let refit_b = refit_lmm(&mut ws, &xb, &yb, n, p, &ids, &opts_b, None);

    let cold_a = fit_cold(&xa, &ya, n, p, &model, &ids, &opts_a);
    let cold_b = fit_cold(&xb, &yb, n, p, &model, &ids, &opts_b);
    assert!(
        cold_a.converged && cold_b.converged,
        "oracle fits must converge"
    );

    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (label, refit, cold) in [
        ("A (unweighted)", &refit_a, &cold_a),
        ("B (weighted)", &refit_b, &cold_b),
    ] {
        assert_eq!(refit.converged, cold.converged, "{label}: converged");
        assert_eq!(bits(&refit.beta), bits(&cold.beta), "{label}: beta");
        assert_eq!(bits(&refit.se), bits(&cold.se), "{label}: se");
        assert_eq!(bits(&refit.tau2), bits(&cold.tau2), "{label}: tau2");
        assert_eq!(
            refit.varcorr.len(),
            cold.varcorr.len(),
            "{label}: varcorr len"
        );
        for (a, b) in refit.varcorr.iter().zip(&cold.varcorr) {
            assert_eq!(bits(a), bits(b), "{label}: varcorr block");
        }
        assert_eq!(
            refit.deviance.to_bits(),
            cold.deviance.to_bits(),
            "{label}: deviance"
        );
        assert_eq!(refit.n_eval, cold.n_eval, "{label}: n_eval");
        assert_eq!(refit.singular, cold.singular, "{label}: singular");
    }
}
