//! GLMM estimator tests (Binomial/Poisson/Gamma/negative-binomial,
//! `re: Some`, dense + sparse-Schur equivalence + AGQ + two-stage).

use super::*;
use crate::glmm::{build_z, glmm_laplace_deviance, GlmmWorkspace, StructuredSchur};
use crate::test_support::assert_near;
use crate::{
    BinomialLink, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing,
    StartValues, WaldSe,
};
use faer::Mat;

use super::common_tests::{assert_pinned, dense_ids, dense_str, sim_clustered, PIN_REL_ITER};

/// `fit_glmm` (now `run_glmm_on` + `glmm_view_to_fit`) must reproduce the `Fit`
/// that the full `fit_cold` dispatch produces for a clustered binomial GLMM —
/// pins the view/assembly split as behavior-preserving. The mu_hat/deviance
/// tuple that the NB marginal-θ loop reads must also stay populated.
#[test]
fn glmm_view_maps_to_same_fit_as_fit_cold() {
    let (x, y, cluster_ids, n) = cbpp_design();
    let p = 4;
    let model = cbpp_model();
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1, 2, 3],
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let (via, mu, dev) = super::glmm::fit_glmm(
        &x,
        &y,
        n,
        p,
        &model,
        &ids.primary,
        &ids.extra,
        f64::NAN,
        None,
        &opts,
    );
    assert!(cold.converged && via.converged);
    assert_near(&cold.beta, &via.beta, "beta");
    assert_near(&cold.se, &via.se, "se");
    assert_near(&[cold.deviance], &[via.deviance], "deviance");
    assert_eq!(mu.len(), n);
    assert!(dev.is_finite());
}

/// Warm-start A/B on the realistic cbpp binomial GLMM (dense joint-BOBYQA
/// path, scalar herd intercept): warm from the cold fit's own solution
/// (θ̂ = √tau2 — σ²≡1 binomial — and β̂ verbatim) and from a perturbed
/// (θ, β) must land on the cold optimum — β, SE, herd SD — and never
/// degrade convergence. Unlike the LMM path, the GLMM start threads β
/// verbatim (bypassing `glm_warm_start_beta`), so both arms also exercise
/// PIRLS opening away from the GLM seed.
#[test]
fn fit_warm_glmm_cbpp_matches_cold_optimum() {
    let (x, y, cluster_ids, n) = cbpp_design();
    let p = 4;
    let model = cbpp_model();
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1, 2, 3],
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(cold.converged, "cold cbpp GLMM must converge");
    let starts = [
        (
            "truth",
            StartValues {
                beta: cold.beta.clone(),
                theta: vec![cold.tau2[0].sqrt()],
            },
        ),
        // Halved β̂ + θ=3 (θ̂ ≈ 0.64): far enough to move the joint
        // optimizer, near enough that PIRLS opens in a sane weight regime
        // from the verbatim β start.
        (
            "perturbed",
            StartValues {
                beta: cold.beta.iter().map(|b| 0.5 * b).collect(),
                theta: vec![3.0],
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
        let (w, c) = (warm.tau2[0].sqrt(), cold.tau2[0].sqrt());
        let rel = (w - c).abs() / c;
        assert!(
            rel < 1e-3,
            "{label}: herd SD warm {w} vs cold {c} (rel {rel})"
        );
    }
}

/// Committed cbpp design, expanded to `size` Bernoulli 0/1 rows per record:
/// `(x [n·4 row-major], y, herd cluster_ids, n)`. Shared by the cbpp oracle
/// test and `fit_grouped_honors_opts_wald_se`.
fn cbpp_design() -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
    let csv = include_str!("../../validation/data/empirical/cbpp.csv");
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut cluster_ids = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let herd: u32 = f[0].parse::<u32>().unwrap() - 1; // herds 1..15 → ids 0..14
        let incidence: u32 = f[1].parse().unwrap();
        let size: u32 = f[2].parse().unwrap();
        let period: u32 = f[3].parse().unwrap();
        let row = [
            1.0,
            f64::from(u32::from(period == 2)),
            f64::from(u32::from(period == 3)),
            f64::from(u32::from(period == 4)),
        ];
        // Expand to `size` Bernoulli trials: `incidence` ones, rest zeros.
        for k in 0..size {
            x.extend_from_slice(&row);
            y.push(if k < incidence { 1.0 } else { 0.0 });
            cluster_ids.push(herd);
        }
    }
    let n = y.len();
    (x, y, cluster_ids, n)
}

/// Structure-only cbpp model: `Binomial{Logit}` + a single intercept herd
/// grouping (15 clusters; explicit ids place each row). Method knobs live in
/// `FitOptions` now, not here.
fn cbpp_model() -> ModelSpec {
    ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 15 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    }
}

/// `opts.wald_se` (not `model.wald_se`) selects the GLMM Wald-SE denominator:
/// `Hessian` and `Rx` on the same cbpp fit must produce different SEs. Guards
/// that the knob lives on `FitOptions`, not `ModelSpec`.
#[test]
fn fit_grouped_honors_opts_wald_se() {
    let (x, y, cluster_ids, n) = cbpp_design();
    let p = 4;
    let model = cbpp_model();
    let hess = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            wald_se: WaldSe::Hessian,
            ..FitOptions::default()
        },
    );
    let rx = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            wald_se: WaldSe::Rx,
            ..FitOptions::default()
        },
    );
    assert!(hess.converged && rx.converged);
    assert!(
        (hess.se[1] - rx.se[1]).abs() > 1e-6,
        "Rx vs Hessian SE must differ"
    );
}

// Frozen lme4 1.1-38 cbpp reference (validation/results/lme4_empirical/cbpp.json,
// which records tolPwrss = 1e-13). ONE definition, shared by the expanded and
// aggregated gates below: they assert the same fit through two entry points, and
// keeping two copies is exactly how both drifted onto lme4's DEFAULT-tolPwrss
// (1e-7) numbers while still citing this file — SE read 0.231213976143225
// against the file's 0.232473254781808. That gap is lme4's lagged-ldL2 artifact
// (glmer builds log|A| from working weights one PIRLS iteration behind the mode
// — see src/glmm/se.rs), and carrying it forced a 3e-2 SE band that hid a real
// ~1.3% disagreement. Corrected 2026-07-21; glmm now agrees to 6.0e-6.
const CBPP_REF_BETA: [f64; 4] = [
    -1.39853204368263,
    -0.992315880328946,
    -1.12866414695346,
    -1.58031559790095,
];
const CBPP_REF_SE: [f64; 4] = [
    0.232473254781808,
    0.306641326429934,
    0.326637242566145,
    0.427437244644503,
];
/// √τ̂²(herd intercept).
const CBPP_REF_HERD_SD: f64 = 0.642269888687578;
const CBPP_REF_LOGLIK: f64 = -92.0262818745091;

/// cbpp binomial GLMM through the stable `fit_cold` surface with explicit
/// `GroupIds` (single grouping), gated against the frozen R `lme4::glmer` oracle
/// (`validation/results/lme4_empirical/cbpp.json`). cbpp is
/// `cbind(incidence, size−incidence) ~ period + (1 | herd)`; the kernel is
/// Bernoulli-logit, so each `(incidence, size)` row is expanded to `size` 0/1
/// rows sharing its design row and herd — value-identical MLE to the aggregated
/// binomial fit. Herds are unbalanced, so the positional `Sizing` layout cannot
/// express them: this is the data-shaped-ids path's reason to exist.
/// SE is compared to **lme4 only** (its Hessian denom keeps the θ–β coupling;
/// MixedModels.jl drops it ~3% — RULE 6). The oracle is sacred: on
/// disagreement glmm is presumed wrong (RULE 0).
#[test]
fn fit_glmm_cbpp_matches_lme4() {
    let (x, y, cluster_ids, n) = cbpp_design();
    let p = 4; // [intercept, period2, period3, period4]
    let model = cbpp_model();
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );

    assert!(f.converged, "cbpp GLMM must converge");
    // Bands are validation/tol.R's cross-engine numbers (beta_rel, se_hessian_rel,
    // stddev_rel = 1e-3) — change together with that file. This is a glmm↔lme4
    // claim, so tol.R's calibration is the one that applies. Measured agreement
    // against the artifact-free reference is far inside them: SE worst 6.0e-6.
    // The SE band was 3e-2 only because the constants above were the
    // default-tolPwrss ones; with the citation corrected, that band no longer
    // has a reason to exist. The oracle is sacred — these bound glmm to lme4,
    // never the reverse (RULE 0).
    for j in 0..p {
        let b_rel = (f.beta[j] - CBPP_REF_BETA[j]).abs() / CBPP_REF_BETA[j].abs();
        assert!(
            b_rel < 1e-3,
            "β[{j}] = {} vs lme4 {} (rel {b_rel})",
            f.beta[j],
            CBPP_REF_BETA[j]
        );
        let se_rel = (f.se[j] - CBPP_REF_SE[j]).abs() / CBPP_REF_SE[j];
        assert!(
            se_rel < 1e-3,
            "se[{j}] = {} vs lme4 {} (rel {se_rel})",
            f.se[j],
            CBPP_REF_SE[j]
        );
    }
    // Herd random-intercept SD = √τ̂²; tau2[0] = θ̂² = τ̂² (σ² = 1 binomial).
    let herd_sd = f.tau2[0].sqrt();
    let sd_rel = (herd_sd - CBPP_REF_HERD_SD).abs() / CBPP_REF_HERD_SD;
    assert!(
        sd_rel < 1e-3,
        "herd SD = {herd_sd} vs lme4 {CBPP_REF_HERD_SD} (rel {sd_rel})"
    );
}

/// cbpp AGGREGATED: 56 rows, y = incidence/size, weights = size. Mirrors
/// `cbpp_design`'s parsing verbatim; only the Bernoulli expansion loop is
/// replaced by one row per CSV record.
fn cbpp_design_aggregated() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>, usize) {
    let csv = include_str!("../../validation/data/empirical/cbpp.csv");
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut w = Vec::<f64>::new();
    let mut cluster_ids = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let herd: u32 = f[0].parse::<u32>().unwrap() - 1; // herds 1..15 → ids 0..14
        let incidence: u32 = f[1].parse().unwrap();
        let size: u32 = f[2].parse().unwrap();
        let period: u32 = f[3].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(period == 2)),
            f64::from(u32::from(period == 3)),
            f64::from(u32::from(period == 4)),
        ]);
        y.push(f64::from(incidence) / f64::from(size));
        w.push(f64::from(size));
        cluster_ids.push(herd);
    }
    let n = y.len();
    (x, y, w, cluster_ids, n)
}

/// Aggregated cbpp through the DENSE (NoZ) path with prior weights must
/// reproduce the same frozen lme4 oracle as the expanded fit — lme4 itself
/// fits cbind(incidence, size−incidence), i.e. the aggregated objective.
/// Matches lme4 1.1-38 (validation/results/lme4_empirical/cbpp.json freeze).
#[test]
fn fit_glmm_cbpp_aggregated_matches_lme4() {
    // Same frozen reference and bands as fit_glmm_cbpp_matches_lme4 — it asserts
    // the same lme4 fit through the expanded entry point, so both read the one
    // CBPP_REF_* definition above rather than each keeping a copy.
    let (x, y, w, cluster_ids, n) = cbpp_design_aggregated();
    let p = 4; // [intercept, period2, period3, period4]
    let model = cbpp_model();
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            weights: Some(w),
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "aggregated cbpp GLMM must converge");
    for j in 0..p {
        let b_rel = (f.beta[j] - CBPP_REF_BETA[j]).abs() / CBPP_REF_BETA[j].abs();
        assert!(
            b_rel < 1e-3,
            "β[{j}] = {} vs lme4 {} (rel {b_rel})",
            f.beta[j],
            CBPP_REF_BETA[j]
        );
        let se_rel = (f.se[j] - CBPP_REF_SE[j]).abs() / CBPP_REF_SE[j];
        assert!(
            se_rel < 1e-3,
            "se[{j}] = {} vs lme4 {} (rel {se_rel})",
            f.se[j],
            CBPP_REF_SE[j]
        );
    }
    let herd_sd = f.tau2[0].sqrt();
    let sd_rel = (herd_sd - CBPP_REF_HERD_SD).abs() / CBPP_REF_HERD_SD;
    assert!(
        sd_rel < 1e-3,
        "herd SD = {herd_sd} vs lme4 {CBPP_REF_HERD_SD} (rel {sd_rel})"
    );
    // lme4 logLik on the same cbind(incidence, size−incidence) fit
    // (validation/results/lme4_empirical/cbpp.json .estimates.loglik) — the
    // aggregated-binomial saturated constant (incl. ln C(mᵢ,sᵢ)) restored
    // under prior weights, the engine-spec §7.6 gate.
    assert!(
        (f.loglik - CBPP_REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs lme4 {CBPP_REF_LOGLIK}",
        f.loglik
    );
    assert!(!f.reml);
    assert_eq!(f.df, 5); // 4 β + herd-intercept θ; binomial has no dispersion
                         // fitted/ranef consistency: b̂ = θ̂û on the natural scale must reproduce μ̂
                         // through the logit link — pins the dense-path ranef layout AND scale.
    assert_eq!(f.ranef_levels, vec![15]);
    assert_eq!(f.ranef.len(), 15);
    assert_eq!(f.fitted.len(), n);
    for i in 0..n {
        let eta: f64 = (0..p).map(|j| x[i * p + j] * f.beta[j]).sum::<f64>()
            + f.ranef[cluster_ids[i] as usize];
        let mu = 1.0 / (1.0 + (-eta).exp());
        assert!(
            (f.fitted[i] - mu).abs() < 1e-8,
            "fitted[{i}] = {} vs Xβ̂+Zb̂ → {mu}",
            f.fitted[i]
        );
    }
}

/// `FitOptions::offset` on the aggregated cbpp binomial GLMM: a constant
/// per-row offset `o` shifts `η = o + Xβ + Zb`, so at the same argmin the
/// fitted intercept must absorb it (`β̂₀(offset) ≈ β̂₀(no offset) − o`) while
/// every other coefficient and the RE variance are unchanged — two
/// independent BOBYQA runs of the same-argmin-up-to-a-shift objectives, so
/// the tolerance is optimizer-scatter-sized (5e-4), not the tight oracle
/// gate above. A zero offset must reproduce the no-offset fit bit-for-bit:
/// `refresh_eta_fixed`'s `if let Some(o)` gate still runs (unlike `None`),
/// so this is the one case that actually exercises the offset-add arithmetic
/// while proving it is a no-op at `o=0`.
#[test]
fn fit_glmm_offset_constant_shifts_intercept() {
    let (x, y, w, cluster_ids, n) = cbpp_design_aggregated();
    let p = 4;
    let model = cbpp_model();
    let ids = GroupIds {
        primary: cluster_ids.clone(),
        extra: vec![],
    };
    let base_opts = FitOptions {
        target_indices: vec![0, 1, 2, 3],
        weights: Some(w.clone()),
        ..FitOptions::default()
    };

    let f0 = fit_cold(&x, &y, n, p, &model, &ids, &base_opts);
    assert!(f0.converged, "no-offset aggregated cbpp GLMM must converge");

    const OFFSET_VAL: f64 = 0.7;
    let f_off = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            offset: Some(vec![OFFSET_VAL; n]),
            ..base_opts.clone()
        },
    );
    assert!(f_off.converged, "offset aggregated cbpp GLMM must converge");

    assert!(
        (f_off.beta[0] - (f0.beta[0] - OFFSET_VAL)).abs() < 5e-4,
        "intercept: offset fit {} vs no-offset {} shifted by -{OFFSET_VAL}",
        f_off.beta[0],
        f0.beta[0]
    );
    for j in 1..p {
        assert!(
            (f_off.beta[j] - f0.beta[j]).abs() < 5e-4,
            "β[{j}]: offset fit {} vs no-offset {}",
            f_off.beta[j],
            f0.beta[j]
        );
    }
    let herd_sd_diff = (f_off.tau2[0].sqrt() - f0.tau2[0].sqrt()).abs();
    assert!(
        herd_sd_diff < 5e-4,
        "herd SD: offset fit {} vs no-offset {}",
        f_off.tau2[0].sqrt(),
        f0.tau2[0].sqrt()
    );

    // Logit consistency: fitted[i] must equal plogis(offset + Xβ̂ + b̂) at the
    // offset fit's own (β̂, b̂) — mirrors the fitted/ranef check on the
    // no-offset oracle test above.
    assert_eq!(f_off.fitted.len(), n);
    for i in 0..n {
        let eta: f64 = OFFSET_VAL
            + (0..p).map(|j| x[i * p + j] * f_off.beta[j]).sum::<f64>()
            + f_off.ranef[cluster_ids[i] as usize];
        let mu = 1.0 / (1.0 + (-eta).exp());
        assert!(
            (f_off.fitted[i] - mu).abs() < 1e-8,
            "fitted[{i}] = {} vs offset+Xβ̂+b̂ → {mu}",
            f_off.fitted[i]
        );
    }

    // An all-zeros offset must be bit-identical to no offset: the
    // `if let Some(o)` gate in `refresh_eta_fixed` runs and adds 0.0 to every
    // eta_fixed entry, which must not perturb the converged optimum at all.
    let f_zero = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            offset: Some(vec![0.0; n]),
            ..base_opts
        },
    );
    assert_eq!(f_zero.deviance, f0.deviance, "zero-offset deviance");
    assert_eq!(f_zero.beta, f0.beta, "zero-offset beta");
}

/// Prior-weight fit-level equivalence on the DENSE (NoZ) path: `fit_cold`
/// on aggregated cbpp proportions with `weights = size` matches the
/// expanded Bernoulli fit on β/SE/τ² for both `WaldSe` arms. Two
/// independent BOBYQA runs of same-argmin objectives, so the bounds are
/// optimizer-scatter-sized (the oracle test above is the tight anchor).
/// Dense twin of `sparse_weighted_binomial_fit_matches_expanded`.
#[test]
fn fit_glmm_cbpp_aggregated_matches_expanded() {
    let (xe, ye, ids_e, n_e) = cbpp_design();
    let (xa, ya, wa, ids_a, n_a) = cbpp_design_aggregated();
    let p = 4;
    let model = cbpp_model();
    for wald_se in [WaldSe::Hessian, WaldSe::Rx] {
        let fe = fit_cold(
            &xe,
            &ye,
            n_e,
            p,
            &model,
            &GroupIds {
                primary: ids_e.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                wald_se,
                ..FitOptions::default()
            },
        );
        let fa = fit_cold(
            &xa,
            &ya,
            n_a,
            p,
            &model,
            &GroupIds {
                primary: ids_a.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                wald_se,
                weights: Some(wa.clone()),
                ..FitOptions::default()
            },
        );
        let tag = format!("{wald_se:?}");
        assert!(
            fe.converged && fa.converged,
            "{tag}: both fits must converge"
        );
        for j in 0..p {
            assert!(
                (fa.beta[j] - fe.beta[j]).abs() < 2e-3 * (1.0 + fe.beta[j].abs()),
                "{tag} β[{j}]: agg={} exp={}",
                fa.beta[j],
                fe.beta[j]
            );
            assert!(
                (fa.se[j] - fe.se[j]).abs() < 2e-2 * (1.0 + fe.se[j].abs()),
                "{tag} se[{j}]: agg={} exp={}",
                fa.se[j],
                fe.se[j]
            );
        }
        assert_eq!(fa.tau2.len(), fe.tau2.len(), "{tag}: tau2 length");
        for (a, b) in fa.tau2.iter().zip(fe.tau2.iter()) {
            assert!(
                (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                "{tag} tau2: agg={a} exp={b}"
            );
        }
    }
}

/// Shared 12-cluster × 10-row single-grouping design for the weighted
/// dense-GLMM goldens: `(x [n·2 row-major: intercept, x1], ids, n, p)`
/// assembled from the R-exported x1 slice; y and w are family-specific.
fn weighted_glmm_design(x1: &[f64]) -> (Vec<f64>, Vec<u32>, usize, usize) {
    let n = x1.len();
    let mut x = Vec::with_capacity(n * 2);
    for &v in x1 {
        x.push(1.0);
        x.push(v);
    }
    let ids: Vec<u32> = (0..n as u32).map(|i| i / 10).collect();
    (x, ids, n, 2)
}

/// Weighted dense Poisson GLMM vs the frozen lme4 golden. Generated with
/// (R 4.5.3, lme4 1.1-38):
/// ```r
/// library(lme4); set.seed(11)
/// g <- rep(1:12, each = 10); n <- 120
/// x1 <- round(rnorm(n), 4); w <- sample(1:4, n, TRUE)
/// b <- rnorm(12, 0, 0.5)
/// y <- rpois(n, exp(0.3 + 0.5 * x1 + b[g]))
/// f <- glmer(y ~ x1 + (1 | g), family = poisson, weights = w)
/// print(summary(f)$coefficients, digits = 15)
/// print(as.data.frame(VarCorr(f)), digits = 15)
/// ```
/// Tolerances mirror `fit_glmm_cbpp_matches_lme4` (2e-3 abs β, 3e-2 rel SE,
/// 3e-3 rel RE SD).
#[test]
fn fit_glmm_poisson_weighted_matches_lme4() {
    const X1: [f64; 120] = [
        -0.591, 0.0266, -1.5166, -1.3627, 1.1785, -0.9342, 1.3236, 0.6249, -0.0457, -1.0041,
        -0.8284, -0.3484, -1.5383, -0.2556, -1.1499, 0.0123, -0.223, 0.8878, -0.5922, -0.6557,
        -0.6825, -0.0159, -0.4426, 0.3526, 0.0732, 0.0072, -0.1876, -0.7657, -0.2211, -0.9836,
        -1.1043, -0.9382, 0.6786, -1.5775, -0.8699, 0.4847, -0.1861, 1.5456, -0.6114, -0.3478,
        -1.6365, 0.0204, 0.8917, -0.8727, 0.8901, -0.3439, -2.1868, 0.8801, 0.7239, 0.2199, 0.7899,
        -0.23, -0.8185, 0.4997, 0.1592, 0.5426, -0.1566, 0.4388, 1.4879, 0.0602, -0.849, 2.3397,
        -0.1212, -1.9502, 0.5387, 1.6935, -0.791, -1.0753, -0.6079, 0.7544, 0.4535, -0.1234,
        -0.7631, 0.2283, 1.1195, 0.1566, -0.6888, 0.4529, -1.0675, 0.4016, -0.0648, 0.3155,
        -0.6057, -0.9076, 2.2616, -0.6032, -1.2979, 0.5065, -0.8533, -1.506, 1.2023, -1.0279,
        0.9383, -0.5432, 0.5131, -0.3526, 1.3265, -1.1402, 1.4131, -0.6022, -0.4417, 0.2436,
        0.5968, -0.12, -2.0697, 0.5856, 0.4894, -1.0066, 1.2697, 1.1239, 0.8425, 1.6206, 0.4477,
        -2.2989, -0.0792, -0.5231, -0.4176, 0.3049, -0.0314, 0.1051,
    ];
    const W: [f64; 120] = [
        4., 1., 3., 2., 3., 2., 3., 1., 3., 1., 1., 1., 2., 4., 4., 4., 1., 1., 4., 3., 4., 4., 3.,
        4., 4., 1., 1., 1., 3., 4., 3., 3., 3., 2., 1., 3., 2., 2., 2., 3., 2., 1., 4., 1., 1., 1.,
        2., 1., 3., 4., 2., 4., 1., 1., 4., 2., 4., 1., 1., 3., 2., 1., 1., 3., 4., 3., 2., 3., 2.,
        1., 3., 4., 1., 4., 1., 3., 3., 1., 2., 4., 2., 4., 1., 2., 1., 4., 1., 4., 3., 4., 3., 2.,
        4., 2., 2., 4., 3., 3., 1., 3., 4., 1., 1., 3., 3., 4., 3., 1., 4., 3., 3., 4., 3., 1., 2.,
        4., 4., 1., 1., 2.,
    ];
    const Y: [f64; 120] = [
        2., 2., 0., 0., 3., 0., 8., 3., 3., 1., 0., 0., 1., 2., 2., 0., 2., 3., 3., 1., 0., 1., 3.,
        1., 3., 0., 2., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 3., 1., 0., 1., 6., 5., 3., 10.,
        6., 1., 14., 4., 3., 3., 0., 0., 1., 1., 0., 1., 1., 3., 1., 1., 4., 0., 0., 0., 1., 1.,
        0., 1., 0., 2., 3., 0., 1., 1., 3., 2., 2., 1., 1., 0., 0., 1., 0., 0., 0., 0., 3., 0., 1.,
        5., 1., 1., 1., 3., 1., 5., 0., 4., 2., 3., 1., 3., 0., 2., 3., 0., 1., 2., 4., 2., 2., 0.,
        1., 0., 2., 0., 1., 1., 0.,
    ];
    const REF_BETA: [f64; 2] = [0.235954720439220, 0.547941515043755];
    const REF_SE: [f64; 2] = [0.1756494873870279, 0.0594199356963711];
    const REF_G_SD: f64 = 0.575359686811311;

    let (x, ids, n, p) = weighted_glmm_design(&X1);
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 12 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let f = fit_cold(
        &x,
        &Y,
        n,
        p,
        &model,
        &GroupIds {
            primary: ids,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(W.to_vec()),
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "weighted Poisson GLMM must converge");
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() < 2e-3,
            "β[{j}] = {} vs lme4 {} (Δ {})",
            f.beta[j],
            REF_BETA[j],
            (f.beta[j] - REF_BETA[j]).abs()
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < 3e-2,
            "se[{j}] = {} vs lme4 {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
    let g_sd = f.tau2[0].sqrt();
    let sd_rel = (g_sd - REF_G_SD).abs() / REF_G_SD;
    assert!(
        sd_rel < 3e-3,
        "g SD = {g_sd} vs lme4 {REF_G_SD} (rel {sd_rel})"
    );
}

/// Weighted dense Gamma GLMM vs the frozen lme4 golden — this is what pins
/// the weighted `gamma_aic` (profiled dispersion over Σwᵢ), the weighted
/// `glmm_sigma_sq` (σ̂² = pwrss/n with wᵢrᵢ², raw-n denominator: lme4's
/// VarCorr vcov below only reproduces under raw n), and the weighted
/// Pearson dispersion. Generated with (R 4.5.3, lme4 1.1-38):
/// ```r
/// library(lme4); set.seed(21)
/// g <- rep(1:12, each = 10); n <- 120
/// x1 <- round(rnorm(n), 4); w <- sample(1:4, n, TRUE)
/// b <- rnorm(12, 0, 0.4)
/// mu <- exp(0.8 + 0.4 * x1 + b[g])
/// y <- round(rgamma(n, shape = 3, scale = mu / 3), 6)
/// f <- glmer(y ~ x1 + (1 | g), family = Gamma("log"), weights = w)
/// print(summary(f)$coefficients, digits = 15)
/// print(as.data.frame(VarCorr(f)), digits = 15); print(sigma(f)^2, digits = 15)
/// ```
/// τ² is compared on lme4's VarCorr vcov scale (σ̂²·θ̂²). SE tolerance is the
/// cbpp 3e-2; β mirrors `fit_glmm_gamma_sim_matches_lme4`'s relative gate.
#[test]
fn fit_glmm_gamma_weighted_matches_lme4() {
    // R-generated covariate data; 1.1283 coincidentally approximates 2/√π.
    #[allow(clippy::approx_constant)]
    const X1: [f64; 120] = [
        0.793, 0.5223, 1.7462, -1.2713, 2.1974, 0.4331, -1.5702, -0.9349, 0.0635, -0.0024, -2.2768,
        0.7574, -0.5484, 0.1725, 0.5629, 1.5118, 0.659, 1.122, -0.7846, -0.4257, 0.393, 0.0368,
        -1.0321, -1.2649, -0.227, 0.7456, 0.3328, -1.124, -0.7061, -0.7275, -1.8343, -0.4077,
        0.0269, 0.9116, 1.6343, 0.0607, 1.8476, 0.0801, 1.4186, 1.4586, 0.0559, -1.5172, -0.0486,
        -0.2144, 2.0958, 0.2023, 0.5177, 1.6781, 0.3852, -1.2819, -0.5822, 1.7741, -0.2107,
        -0.3521, 0.5852, 1.0137, -0.0226, -0.9032, 0.9078, 1.1619, -0.458, 0.928, -2.1029, -1.6772,
        1.7657, 0.7944, -0.4839, 1.9284, -0.3841, -1.5867, 0.2143, -1.1383, 0.4894, -1.7526, 0.501,
        0.0868, 0.1911, 0.8318, -0.679, 0.2959, 1.1122, 0.3626, -0.2709, -0.1969, 0.067, -0.8678,
        -0.362, -1.1396, -0.8154, 1.3102, -0.2584, 0.6063, 0.3134, 0.0536, 1.1283, -0.5581, 1.536,
        -0.0624, 0.0216, -2.0898, -0.8109, -2.9438, -0.0188, -0.3547, 0.0356, 0.4941, -0.6598,
        1.0011, 1.0721, 0.7558, -1.4555, 0.9429, -1.8703, -0.2533, -0.2926, 0.2188, -1.3551,
        -0.1227, -0.4519, 0.0972,
    ];
    const W: [f64; 120] = [
        2., 2., 2., 1., 2., 4., 2., 3., 3., 3., 4., 4., 4., 4., 2., 4., 3., 3., 2., 2., 1., 1., 4.,
        1., 1., 1., 4., 4., 3., 4., 3., 2., 4., 3., 4., 2., 4., 2., 2., 2., 1., 1., 1., 1., 1., 3.,
        2., 1., 2., 2., 4., 4., 2., 3., 4., 4., 4., 3., 2., 4., 4., 2., 3., 4., 2., 4., 2., 2., 2.,
        1., 1., 1., 4., 4., 4., 1., 3., 4., 4., 3., 2., 1., 1., 4., 4., 4., 1., 2., 2., 2., 4., 3.,
        3., 1., 1., 1., 4., 3., 4., 3., 3., 2., 2., 3., 4., 4., 3., 4., 2., 1., 3., 1., 3., 2., 3.,
        3., 4., 3., 1., 3.,
    ];
    const Y: [f64; 120] = [
        1.027885, 3.568778, 5.059958, 1.829256, 7.572745, 1.888244, 0.638556, 1.352118, 6.460123,
        1.431433, 0.491063, 1.808875, 1.736458, 2.965294, 4.171528, 2.554423, 2.217066, 0.48551,
        1.646985, 3.758326, 3.388564, 2.795867, 0.780591, 1.495213, 1.664063, 3.445218, 2.973526,
        1.700702, 1.031139, 1.852452, 2.514445, 1.04869, 1.757371, 2.407751, 1.232387, 1.211173,
        7.507012, 3.516693, 3.209465, 1.575613, 1.416005, 0.324474, 1.528727, 1.941835, 9.305071,
        0.960217, 1.934011, 1.54724, 1.326433, 1.255908, 2.665283, 4.779793, 1.830826, 0.990174,
        1.892684, 11.248398, 1.851022, 1.273189, 3.905656, 0.905928, 3.315271, 1.126161, 0.465568,
        1.937359, 4.986676, 5.506185, 0.636041, 5.615351, 0.473084, 0.831148, 1.471093, 2.344402,
        0.680976, 1.026012, 1.43575, 2.919631, 5.756904, 4.804391, 1.699487, 0.706556, 3.551593,
        2.787834, 2.280541, 1.685016, 3.503679, 3.911159, 0.424846, 3.080594, 0.663857, 4.361308,
        3.329871, 3.137527, 7.377112, 2.457973, 4.633516, 3.899755, 5.727707, 1.813578, 2.754815,
        1.84022, 0.753663, 0.331312, 0.870051, 2.412794, 3.001372, 1.099695, 4.98129, 4.075331,
        4.525327, 5.201431, 1.504496, 5.951359, 1.258666, 5.439477, 2.243875, 0.603161, 1.000063,
        2.337211, 0.981631, 0.914213,
    ];
    const REF_BETA: [f64; 2] = [0.863125471935252, 0.372178348714047];
    const REF_SE: [f64; 2] = [0.0654914008493939, 0.0333300007320761];
    const REF_G_VCOV: f64 = 0.0510221486396947; // σ̂²·θ̂² (lme4 VarCorr vcov)

    let (x, ids, n, p) = weighted_glmm_design(&X1);
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 12 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let f = fit_cold(
        &x,
        &Y,
        n,
        p,
        &model,
        &GroupIds {
            primary: ids,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1],
            weights: Some(W.to_vec()),
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "weighted Gamma GLMM must converge");
    for j in 0..p {
        let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
        assert!(
            b_rel < 2e-3,
            "β[{j}] = {} vs lme4 {} (rel {b_rel})",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(
            se_rel < 3e-2,
            "se[{j}] = {} vs lme4 {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
    // varcorr[0][0] = σ̂²·θ̂² — directly lme4's VarCorr vcov for the g
    // intercept, via the public block (σ̂²-scaled like tau2, B1 fix).
    let vc_rel = (f.varcorr[0][0] - REF_G_VCOV).abs() / REF_G_VCOV;
    assert!(
        vc_rel < 1e-2,
        "g vcov = {} vs lme4 {REF_G_VCOV} (rel {vc_rel})",
        f.varcorr[0][0]
    );
    assert!(
        (f.varcorr[0][0] - f.tau2[0]).abs() < 1e-12,
        "varcorr and tau2 must report the same σ̂²-scaled variance"
    );
}

/// Poisson GLMM `TICKS ~ 1 + YEAR + cHEIGHT + (1|INDEX)` on grouseticks
/// (observation-level INDEX = 403 size-1 clusters), gated against frozen
/// `lme4::glmer(family=poisson, nAGQ=1)` (`validation/goldens/grouseticks_agq_k1.json`).
/// Exercises the blocked PIRLS path for a non-binomial family. lme4-only SE
/// (RULE 6). The oracle is sacred.
#[test]
fn fit_glmm_poisson_grouseticks_matches_lme4() {
    const REF_BETA: [f64; 4] = [
        0.43997315657,
        1.10082823356,
        -0.988047711093,
        -0.0236982108735,
    ];
    const REF_SE: [f64; 4] = [
        0.140882438904,
        0.168795499457,
        0.197654140578,
        0.00211151961592,
    ];
    const REF_INDEX_SD: f64 = 1.129369439;
    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        raw.push(f[0].parse().unwrap()); // INDEX
        let year: u32 = f[4].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            f[6].parse().unwrap(), // cHEIGHT
        ]);
        y.push(f[1].parse().unwrap()); // TICKS
    }
    let (cluster_ids, n_clusters) = dense_ids(&raw);
    let n = y.len();
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
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
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "poisson GLMM must converge");
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs lme4 {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(se_rel < 3e-2, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
    }
    let sd_rel = (f.tau2[0].sqrt() - REF_INDEX_SD).abs() / REF_INDEX_SD;
    assert!(
        sd_rel < 3e-3,
        "INDEX sd = {} vs lme4 {REF_INDEX_SD}",
        f.tau2[0].sqrt()
    );
    // lme4 logLik (grouseticks_agq_k1.json .estimates.loglik) — the Poisson
    // saturated constant Σ(yᵢln yᵢ − yᵢ − ln yᵢ!) restored.
    const REF_LOGLIK: f64 = -957.399741174491;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 5); // 4 β + INDEX θ; Poisson has no dispersion
                         // fitted/ranef consistency through the log link (403 size-1 clusters).
    assert_eq!(f.ranef_levels, vec![n_clusters]);
    assert_eq!(f.fitted.len(), n);
    for i in 0..n {
        let eta: f64 = (0..p).map(|j| x[i * p + j] * f.beta[j]).sum::<f64>()
            + f.ranef[cluster_ids[i] as usize];
        assert!(
            (f.fitted[i] - eta.exp()).abs() < 1e-6 * eta.exp().max(1.0),
            "fitted[{i}] = {} vs exp(Xβ̂+Zb̂) = {}",
            f.fitted[i],
            eta.exp()
        );
    }
}

/// Poisson GLMM with a per-row offset vs R `glmer(offset=)`: grouseticks
/// `TICKS ~ YEAR + cHEIGHT + (1|INDEX)` with `o_i = 0.1·((i−1) mod 7)`
/// (0-based CSV row order in Rust). Oracle (R 4.5.3, lme4 1.1-38,
/// `glmerControl(tolPwrss = 1e-13)`):
///   fg <- glmer(TICKS ~ YEAR + cHEIGHT + (1|INDEX), poisson, data = gt, offset = og)
///   print(fixef(fg), digits = 15); print(logLik(fg), digits = 15)
///   print(as.data.frame(VarCorr(fg))$sdcor[1], digits = 15)
#[test]
fn fit_glmm_poisson_offset_matches_lme4() {
    const REF_BETA: [f64; 4] = [
        0.128483161410054,
        1.10179195638099,
        -0.982969256355447,
        -0.023819614546972,
    ];
    const REF_LOGLIK: f64 = -960.701615612628;
    const REF_INDEX_SD: f64 = 1.14913810893358;
    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        raw.push(f[0].parse().unwrap()); // INDEX
        let year: u32 = f[4].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            f[6].parse().unwrap(), // cHEIGHT
        ]);
        y.push(f[1].parse().unwrap()); // TICKS
    }
    let (cluster_ids, n_clusters) = dense_ids(&raw);
    let n = y.len();
    let o: Vec<f64> = (0..n).map(|i| 0.1 * (i % 7) as f64).collect();
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
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
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            offset: Some(o.clone()),
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "offset poisson GLMM must converge");
    for (j, (&b, &r)) in f.beta.iter().zip(&REF_BETA).enumerate() {
        // Intercept is near zero under this offset — absolute band there.
        let ok = if r.abs() > 0.1 {
            (b - r).abs() / r.abs() < 2e-3
        } else {
            (b - r).abs() < 2e-3
        };
        assert!(ok, "β[{j}] = {b} vs lme4 {r}");
    }
    let sd_rel = (f.tau2[0].sqrt() - REF_INDEX_SD).abs() / REF_INDEX_SD;
    assert!(
        sd_rel < 3e-3,
        "INDEX sd = {} vs lme4 {REF_INDEX_SD}",
        f.tau2[0].sqrt()
    );
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
    // fitted folds the offset: μ̂ = exp(o + Xβ̂ + b̂[cluster]).
    for i in 0..n {
        let eta: f64 = o[i]
            + (0..p).map(|j| x[i * p + j] * f.beta[j]).sum::<f64>()
            + f.ranef[cluster_ids[i] as usize];
        assert!(
            (f.fitted[i] - eta.exp()).abs() < 1e-6 * eta.exp().max(1.0),
            "fitted[{i}] = {} vs exp(o+Xβ̂+Zb̂) = {}",
            f.fitted[i],
            eta.exp()
        );
    }
}

/// Parses `validation/data/empirical/grouseticks.csv` into the 3-crossed `TICKS ~ YEAR +
/// cHEIGHT + (1|INDEX) + (1|BROOD) + (1|LOCATION)` design (observation-level
/// INDEX + crossed BROOD, LOCATION). Shared by the lme4 fit gate below and the
/// both-paths sparse-vs-dense Schur cross-checks (`sparse_schur_*`), which need
/// direct `GlmmWorkspace`/`StructuredSchur` access that `fit_cold` doesn't expose.
fn grouseticks_3crossed_inputs() -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, GroupIds) {
    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    // cols: INDEX,TICKS,BROOD,HEIGHT,YEAR,LOCATION,cHEIGHT
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut index_raw = Vec::<u32>::new();
    let mut brood_raw = Vec::<String>::new();
    let mut loc_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        index_raw.push(f[0].parse().unwrap());
        let year: u32 = f[4].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            f[6].parse().unwrap(), // cHEIGHT
        ]);
        y.push(f[1].parse().unwrap()); // TICKS
        brood_raw.push(f[2].to_string());
        loc_raw.push(f[5].to_string());
    }
    let n = y.len();
    let (index_ids, n_index) = dense_ids(&index_raw);
    let (brood_ids, _n_brood) = dense_str(&brood_raw);
    let (loc_ids, _n_loc) = dense_str(&loc_raw);

    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_index as u32,
            },
            slopes: vec![],
            extra_groupings: vec![
                Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                },
                Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                },
            ],
        }),
    };
    let ids = GroupIds {
        primary: index_ids,
        extra: vec![brood_ids, loc_ids],
    };
    (x, y, n, p, model, ids)
}

/// Poisson GLMM, **three crossed groupings**: grouseticks
/// `TICKS ~ YEAR + cHEIGHT + (1|INDEX) + (1|BROOD) + (1|LOCATION)` (observation-
/// level INDEX + crossed BROOD, LOCATION), gated against the frozen
/// `lme4::glmer(family=poisson)` reference (`validation/results/lme4_empirical/grouseticks.json`).
/// Exercises the structured crossed-extras PIRLS/Schur path (`pirls_solve_blocked_
/// extras` / `structured_factor`) that the single-grouping test above does not.
/// This is the regression guard for the degenerate-fit bug: from a β=0 cold start
/// the first PIRLS step overshot into a ~1e30 weight regime, the crossed Schur
/// went non-PD, and the fit returned start values reported as converged. The GLM
/// warm-start of β (`glm_warm_start_beta`) opens PIRLS near the mean and removes
/// the overshoot; the converged-deviance guard (`glmm/mod.rs`) is the backstop.
/// The oracle is sacred.
#[test]
fn fit_glmm_poisson_grouseticks_3crossed_matches_lme4() {
    // Frozen lme4 reference (validation/results/lme4_empirical/grouseticks.json).
    const REF_BETA: [f64; 4] = [
        0.372776372908808,
        1.18041688638813,
        -0.978684717829623,
        -0.0237606272596611,
    ];
    const REF_INDEX_SD: f64 = 0.541508524819898;
    const REF_BROOD_SD: f64 = 0.750027963921318;
    const REF_LOCATION_SD: f64 = 0.52872140071578;
    let (x, y, n, p, model, ids) = grouseticks_3crossed_inputs();
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );

    assert!(
        f.converged,
        "3-crossed poisson GLMM must converge (not the degenerate start fit)"
    );
    // Bands are `validation/tol.R`'s cross-engine ones (beta_rel/stddev_rel 1e-3),
    // since every constant here is lme4's — mirrors tol.R, change together.
    // Measured worst against this reference: β 6.9e-5, RE stddev 2.2e-5. The
    // 3e-2/5e-2 these replaced predate the tol.R calibration and could not fail.
    // β[3] (cHEIGHT, ~-0.024) took an absolute band while the others took a
    // relative one; at 2.3e-6 relative it does not need the exception.
    for (j, &rb) in REF_BETA.iter().enumerate() {
        assert!(
            (f.beta[j] - rb).abs() / rb.abs() < 1e-3,
            "β[{j}] = {} vs lme4 {rb}",
            f.beta[j]
        );
    }
    // tau2 layout [primary(INDEX) | BROOD | LOCATION].
    for (k, refsd) in [REF_INDEX_SD, REF_BROOD_SD, REF_LOCATION_SD]
        .into_iter()
        .enumerate()
    {
        let sd = f.tau2[k].sqrt();
        assert!(
            (sd - refsd).abs() / refsd < 1e-3,
            "grouping {k} sd = {sd} vs lme4 {refsd}"
        );
    }
}

/// Both-paths cross-check: the sparse-S Laplace deviance equals
/// the dense-Schur deviance at the same θ on the grouseticks 3-crossed design. If
/// they disagree, exactly one factor path is wrong (the +0.5·logdet_llt convention
/// is the prime suspect). Not bitwise-equal (AMD reorders the sparse elimination),
/// so a tight numeric gate, orders below the ~1.5e-4 lme4 β gap we must preserve.
#[test]
fn sparse_schur_deviance_equals_dense_grouseticks() {
    let (x, y, n, p, model, ids) = grouseticks_3crossed_inputs();
    let model = spec_sized_from_ids_pub(&model, &ids);
    let slope_cols: Vec<usize> = vec![];
    let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, 1);
    // Column-major x + build_z + StructuredSchur, as fit_glmm does.
    let mut xm = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xm[(i, j)] = x[i * p + j];
        }
    }
    build_z(
        &mut ws,
        xm.as_ref().subrows(0, n),
        &ids.primary,
        &ids.extra,
        n,
    );
    ws.structured_schur = StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n);
    // A representative interior θ (the blind start θ₀ for the 3 groupings) + a β
    // (the GLM warm start, matching what `fit_glmm` would open PIRLS at).
    let params: Vec<f64> = {
        let mut prm = ws.params.clone();
        let beta = glm_warm_start_beta(
            model.family,
            f64::NAN,
            xm.as_ref().subrows(0, n),
            &y,
            n,
            p,
            None,
        );
        prm[ws.n_theta..ws.n_theta + p].copy_from_slice(&beta);
        prm
    };

    ws.force_dense_schur = true;
    let dev_dense = glmm_laplace_deviance(
        &params,
        &mut ws,
        xm.as_ref().subrows(0, n),
        &y,
        &ids.primary,
        n,
    );
    ws.force_dense_schur = false;
    let dev_sparse = glmm_laplace_deviance(
        &params,
        &mut ws,
        xm.as_ref().subrows(0, n),
        &y,
        &ids.primary,
        n,
    );

    assert!(
        dev_dense.is_finite() && dev_sparse.is_finite(),
        "both deviances finite"
    );
    let rel = (dev_dense - dev_sparse).abs() / (1.0 + dev_dense.abs());
    assert!(
        rel < 1e-9,
        "dense {dev_dense} vs sparse {dev_sparse} (rel {rel})"
    );
}

/// SE cross-check: the structured_schur_fill SE (sparse solve) equals the dense-Schur
/// SE at the converged fit (se.rs routes through structured_ainv_solve).
/// Unlike `sparse_schur_deviance_equals_dense_grouseticks` (one eval at a fixed θ,
/// gated at 1e-9), this runs the FULL BOBYQA optimization twice — dense and sparse
/// factor paths disagree by ~1e-9 per eval (AMD reorders the sparse elimination), so
/// each run's θ̂ drifts by a path-dependent amount within BOBYQA's `rho_end` trust
/// region before the Wald SE nonlinearly amplifies it. Gated at 1e-4: orders above
/// the observed ~6.6e-7 noise floor, still tight enough to catch a real convention
/// bug (a flipped 0.5×/1.0× logdet would show as a gap orders of magnitude larger).
#[test]
fn sparse_schur_se_equals_dense_grouseticks() {
    let (x, y, n, p, model, ids) = grouseticks_3crossed_inputs();
    let model = spec_sized_from_ids_pub(&model, &ids);
    let slope_cols: Vec<usize> = vec![];
    let mut xm = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xm[(i, j)] = x[i * p + j];
        }
    }
    let beta_start = glm_warm_start_beta(
        model.family,
        f64::NAN,
        xm.as_ref().subrows(0, n),
        &y,
        n,
        p,
        None,
    );

    let run = |force_dense: bool| -> (Vec<f64>, bool) {
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, 1);
        build_z(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &ids.primary,
            &ids.extra,
            n,
        );
        ws.structured_schur = if ws.groupings.structured_extras_eligible() {
            StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
        } else {
            None
        };
        ws.force_dense_schur = force_dense;
        let fit = crate::glmm::fit_glmm(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &y,
            &ids.primary,
            &[0, 1, 2, 3],
            None,
            &beta_start,
            n,
            WaldSe::Rx,
        );
        (ws.var_diag[..p].to_vec(), fit.converged)
    };

    let (var_dense, conv_dense) = run(true);
    let (var_sparse, conv_sparse) = run(false);
    assert!(
        conv_dense && conv_sparse,
        "both dense and sparse fits must converge"
    );
    for (j, (&vd, &vs)) in var_dense.iter().zip(&var_sparse).enumerate() {
        assert!(
            vd.is_finite() && vs.is_finite(),
            "var_diag[{j}] finite (dense {vd}, sparse {vs})"
        );
        let rel = (vd - vs).abs() / (1.0 + vd.abs());
        assert!(
            rel < 1e-4,
            "var_diag[{j}] dense {vd} vs sparse {vs} (rel {rel})"
        );
    }
}

/// Small-`e` guard (no regression on small-`e` GLMMs): a synthetic
/// crossed binomial GLMM `y ~ x + (1|g1) + (1|g2)`, primary g1 = 4 levels,
/// extra crossed g2 = 6 levels ⇒ e = 6 — orders below grouseticks' e = 181,
/// the scale the other `sparse_schur_*_equals_dense_grouseticks` cross-checks
/// exercise. Runs the full BOBYQA fit twice (dense-forced vs sparse,
/// mirroring `sparse_schur_se_equals_dense_grouseticks`'s pattern) and
/// compares both β and the Wald SE. Gated at 1e-7 (tighter than that e=181
/// test's 1e-4): a 6-wide Schur gives AMD far less elimination-order
/// freedom, so the dense/sparse per-eval float noise that drives BOBYQA
/// path-dependent drift is negligible at this scale.
#[test]
fn sparse_schur_small_e_matches_dense() {
    // 4-level primary × 6-level crossed extra, 2 obs/cell ⇒ e = 6, n = 48.
    let (n_prim, n_extra, reps) = (4usize, 6usize, 2usize);
    let n = n_prim * n_extra * reps;
    let p = 2;
    let prim_eff = [0.4, -0.3, 0.5, -0.2];
    let extra_eff = [0.3, -0.4, 0.2, -0.1, 0.35, -0.25];
    let mut xm = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    let mut cl = vec![0u32; n];
    let mut cr = vec![0u32; n];
    let mut st = 42u64;
    let mut i = 0;
    for (pi, &pe) in prim_eff.iter().enumerate() {
        for (ei, &ee) in extra_eff.iter().enumerate() {
            for _ in 0..reps {
                let cov = crate::sparse::test_lcg(&mut st);
                let eta = 0.2 + 0.6 * cov + pe + ee;
                let prob = 1.0 / (1.0 + (-eta).exp());
                let draw = (crate::sparse::test_lcg(&mut st) + 1.0) / 2.0;
                xm[(i, 0)] = 1.0;
                xm[(i, 1)] = cov;
                cl[i] = pi as u32;
                cr[i] = ei as u32;
                y[i] = if draw < prob { 1.0 } else { 0.0 };
                i += 1;
            }
        }
    }
    let ids = GroupIds {
        primary: cl,
        extra: vec![cr],
    };
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_prim as u32,
            },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: n_extra as u32,
                },
                slopes: vec![],
            }],
        }),
    };
    let model = spec_sized_from_ids_pub(&model, &ids);
    let slope_cols: Vec<usize> = vec![];
    let beta_start = glm_warm_start_beta(
        model.family,
        f64::NAN,
        xm.as_ref().subrows(0, n),
        &y,
        n,
        p,
        None,
    );

    let run = |force_dense: bool| -> (Vec<f64>, Vec<f64>, bool) {
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, 1);
        build_z(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &ids.primary,
            &ids.extra,
            n,
        );
        ws.structured_schur = if ws.groupings.structured_extras_eligible() {
            StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
        } else {
            None
        };
        ws.force_dense_schur = force_dense;
        let fit = crate::glmm::fit_glmm(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &y,
            &ids.primary,
            &[0, 1],
            None,
            &beta_start,
            n,
            WaldSe::Rx,
        );
        (ws.betas.clone(), ws.var_diag[..p].to_vec(), fit.converged)
    };

    let (beta_dense, var_dense, conv_dense) = run(true);
    let (beta_sparse, var_sparse, conv_sparse) = run(false);
    assert!(
        conv_dense && conv_sparse,
        "both dense and sparse fits must converge"
    );
    for j in 0..p {
        let rel_b = (beta_dense[j] - beta_sparse[j]).abs() / (1.0 + beta_dense[j].abs());
        assert!(
            rel_b < 1e-7,
            "β[{j}] dense {} vs sparse {} (rel {rel_b})",
            beta_dense[j],
            beta_sparse[j]
        );
        let vd = var_dense[j];
        let vs = var_sparse[j];
        assert!(
            vd.is_finite() && vs.is_finite(),
            "var_diag[{j}] finite (dense {vd}, sparse {vs})"
        );
        let rel_v = (vd - vs).abs() / (1.0 + vd.abs());
        assert!(
            rel_v < 1e-7,
            "var_diag[{j}] dense {vd} vs sparse {vs} (rel {rel_v})"
        );
    }
}

/// Adaptive GH quadrature, binomial GLMM: cbpp `cbind(incidence, size−incidence)
/// ~ period + (1|herd)` (expanded 0/1) at nAGQ ∈ {1,7,11}, gated against frozen
/// `glmer(nAGQ=k)` (`validation/goldens/cbpp_agq_k{1,7,11}.json`). nAGQ=1 is Laplace
/// (≡ `fit_glmm_cbpp_matches_lme4`); k>1 shifts β/varcomp off it as the Laplace
/// bias is integrated out (herd sd 0.642→0.648). β + varcomp only — the AGQ
/// goldens don't freeze SE (AGQ changes the integral, not the SE convention).
/// The oracle is sacred.
#[test]
fn fit_glmm_binomial_agq_matches_lme4() {
    // (nAGQ, β, herd sd) per frozen glmer(nAGQ=k).
    let refs: [(u8, [f64; 4], f64); 3] = [
        (
            1,
            [
                -1.3983428644712,
                -0.991924974975699,
                -1.12821621594328,
                -1.57974541364914,
            ],
            0.642069927729443,
        ),
        (
            7,
            [
                -1.39923514006289,
                -0.991393555379478,
                -1.12782137776524,
                -1.57947295789128,
            ],
            0.647518692435348,
        ),
        (
            11,
            [
                -1.39921944386306,
                -0.991408657432828,
                -1.12781283713842,
                -1.57948777358155,
            ],
            0.647517861083539,
        ),
    ];
    let csv = include_str!("../../validation/data/empirical/cbpp.csv");
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut cluster_ids = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let herd: u32 = f[0].parse::<u32>().unwrap() - 1;
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
            cluster_ids.push(herd);
        }
    }
    let n = y.len();
    for (nagq, refb, refsd) in refs {
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 15 },
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
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                nagq,
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "binomial AGQ k={nagq} must converge");
        for (j, (&b, &rb)) in f.beta.iter().zip(&refb).enumerate() {
            assert!(
                (b - rb).abs() / rb.abs() < 1e-3,
                "k={nagq} β[{j}] = {b} vs lme4 {rb}"
            );
        }
        let sd_rel = (f.tau2[0].sqrt() - refsd).abs() / refsd;
        assert!(
            sd_rel < 1e-3,
            "k={nagq} herd sd = {} vs lme4 {refsd}",
            f.tau2[0].sqrt()
        );
    }
}

/// Aggregated-form cbpp (56 rows, y = incidence/size, weights = size) through the
/// weighted AGQ path must reproduce the SAME frozen `glmer(nAGQ=k)` goldens as the
/// expanded fixture above — the goldens were produced from aggregated cbpp on the
/// R side, so they are reusable as-is; only the Rust encoding changes. This is the
/// weighted-AGQ validation rung: it exercises `prior_w` flowing through the AGQ
/// kernel's PIRLS mode and per-row dev_resid sums end-to-end at nAGQ ∈ {1,7,11}.
/// The oracle is sacred.
#[test]
fn fit_glmm_cbpp_aggregated_agq_matches_lme4() {
    // Same frozen (nAGQ, β, herd sd) constants as fit_glmm_binomial_agq_matches_lme4.
    let refs: [(u8, [f64; 4], f64); 3] = [
        (
            1,
            [
                -1.3983428644712,
                -0.991924974975699,
                -1.12821621594328,
                -1.57974541364914,
            ],
            0.642069927729443,
        ),
        (
            7,
            [
                -1.39923514006289,
                -0.991393555379478,
                -1.12782137776524,
                -1.57947295789128,
            ],
            0.647518692435348,
        ),
        (
            11,
            [
                -1.39921944386306,
                -0.991408657432828,
                -1.12781283713842,
                -1.57948777358155,
            ],
            0.647517861083539,
        ),
    ];
    let (x, y, w, cluster_ids, n) = cbpp_design_aggregated();
    let p = 4;
    let model = cbpp_model();
    for (nagq, refb, refsd) in refs {
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                nagq,
                weights: Some(w.clone()),
                ..FitOptions::default()
            },
        );
        assert!(
            f.converged,
            "aggregated binomial AGQ k={nagq} must converge"
        );
        for (j, (&b, &rb)) in f.beta.iter().zip(&refb).enumerate() {
            assert!(
                (b - rb).abs() / rb.abs() < 1e-3,
                "k={nagq} β[{j}] = {b} vs lme4 {rb}"
            );
        }
        let sd_rel = (f.tau2[0].sqrt() - refsd).abs() / refsd;
        assert!(
            sd_rel < 1e-3,
            "k={nagq} herd sd = {} vs lme4 {refsd}",
            f.tau2[0].sqrt()
        );
    }
}

/// `FitOptions::parallel_inner` gates the AGQ cluster-outer restructuring
/// (`agq::agq_deviance`'s `cluster_rows` path) but must never change the fitted
/// result: cluster-outer and node-outer visit the same operands in the same
/// per-accumulator order (`ClusterRowIndex`'s ascending-row guarantee), so a
/// full cbpp AGQ fit through the stable `fit_cold` surface is bit-identical
/// with the knob on vs off. Exact equality, not tolerance — this is the
/// end-to-end witness for the same safety argument
/// `agq_cluster_outer_bit_identical_to_node_outer` (glmm/tests.rs) checks at
/// the kernel level.
#[test]
fn fit_glmm_binomial_agq_parallel_inner_knob_is_bit_identical() {
    let (x, y, cluster_ids, n) = cbpp_design();
    let p = 4;
    let model = cbpp_model();
    for nagq in [7u8, 11] {
        let ids = GroupIds {
            primary: cluster_ids.clone(),
            extra: vec![],
        };
        let f_on = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                nagq,
                parallel_inner: true,
                ..FitOptions::default()
            },
        );
        let f_off = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                nagq,
                parallel_inner: false,
                ..FitOptions::default()
            },
        );
        assert!(f_on.converged && f_off.converged, "nagq={nagq}");
        for (j, (&b_on, &b_off)) in f_on.beta.iter().zip(&f_off.beta).enumerate() {
            assert_eq!(
                b_on.to_bits(),
                b_off.to_bits(),
                "nagq={nagq} β[{j}]: on={b_on} off={b_off}"
            );
        }
        for (j, (&s_on, &s_off)) in f_on.se.iter().zip(&f_off.se).enumerate() {
            assert_eq!(
                s_on.to_bits(),
                s_off.to_bits(),
                "nagq={nagq} se[{j}]: on={s_on} off={s_off}"
            );
        }
        for (j, (&t_on, &t_off)) in f_on.tau2.iter().zip(&f_off.tau2).enumerate() {
            assert_eq!(
                t_on.to_bits(),
                t_off.to_bits(),
                "nagq={nagq} tau2[{j}]: on={t_on} off={t_off}"
            );
        }
    }
}

/// Adaptive GH quadrature, Poisson GLMM: grouseticks single-grouping `TICKS ~
/// YEAR + cHEIGHT + (1|INDEX)` at nAGQ ∈ {1,7,11}, gated against frozen
/// `glmer(family=poisson, nAGQ=k)` (`validation/goldens/grouseticks_agq_k{1,7,11}.json`).
/// nAGQ=1 ≡ `fit_glmm_poisson_grouseticks_matches_lme4`; k>1 shifts the fit as the
/// Laplace bias is integrated out. β + varcomp only. The oracle is sacred.
#[test]
fn fit_glmm_poisson_agq_matches_lme4() {
    let refs: [(u8, [f64; 4], f64); 3] = [
        (
            1,
            [
                0.439973156570138,
                1.10082823355748,
                -0.988047711092655,
                -0.0236982108735122,
            ],
            1.1293694390126,
        ),
        (
            7,
            [
                0.443726696423487,
                1.09738146557843,
                -0.988798870848502,
                -0.0236841397694784,
            ],
            1.13482415039616,
        ),
        (
            11,
            [
                0.444137982539483,
                1.09717523260645,
                -0.9889317811938,
                -0.0236832339939658,
            ],
            1.13407867482264,
        ),
    ];
    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        raw.push(f[0].parse().unwrap());
        let year: u32 = f[4].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            f[6].parse().unwrap(),
        ]);
        y.push(f[1].parse().unwrap());
    }
    let (cluster_ids, n_clusters) = dense_ids(&raw);
    let n = y.len();
    for (nagq, refb, refsd) in refs {
        let model = ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
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
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                nagq,
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "poisson AGQ k={nagq} must converge");
        for (j, (&b, &rb)) in f.beta.iter().zip(&refb).enumerate() {
            assert!(
                (b - rb).abs() / rb.abs() < 1e-3,
                "k={nagq} β[{j}] = {b} vs lme4 {rb}"
            );
        }
        let sd_rel = (f.tau2[0].sqrt() - refsd).abs() / refsd;
        assert!(
            sd_rel < 1e-3,
            "k={nagq} INDEX sd = {} vs lme4 {refsd}",
            f.tau2[0].sqrt()
        );
    }
}

/// Probit binomial GLMM `cbind(incidence, size−incidence) ~ period + (1|herd)`
/// on cbpp (expanded 0/1), gated against frozen `glmer(binomial("probit"))`
/// (`validation/goldens/cbpp_probit_glmm.json`). lme4-only SE. The oracle is sacred.
// FD-Hessian SE (use.hessian=TRUE) for this non-canonical link needs a
// smooth deviance: probit is Fisher-scoring (linear convergence), so PIRLS at
// the canonical 1e-6 tolerance left the deviance noisy to ~1e-4 and the FD
// second differences amplified it into a 7–41%-wrong SE. `pirls_tol` gives
// non-canonical links the tight `PIRLS_TOL_REL_NONCANON` (1e-8); β and
// se_hessian now match lme4 to ~1e-4. (The Φ accuracy — `phi_hp`, Cody erfc —
// is a separate genuine fix but was NOT the SE cause; verified by spike.)
#[test]
fn fit_glmm_probit_cbpp_matches_lme4() {
    const REF_BETA: [f64; 4] = [
        -0.835474929637,
        -0.528032739718,
        -0.616854298164,
        -0.799572598137,
    ];
    const REF_SE: [f64; 4] = [
        0.126232795983,
        0.160588369843,
        0.169457682932,
        0.204681153481,
    ];
    const REF_HERD_SD: f64 = 0.3379893465;
    let csv = include_str!("../../validation/data/empirical/cbpp.csv");
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut cluster_ids = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let herd: u32 = f[0].parse::<u32>().unwrap() - 1;
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
            cluster_ids.push(herd);
        }
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Probit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 15 },
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
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "probit GLMM must converge");
    let sd_rel = (f.tau2[0].sqrt() - REF_HERD_SD).abs() / REF_HERD_SD;
    assert!(
        sd_rel < 3e-3,
        "herd sd = {} vs lme4 {REF_HERD_SD}",
        f.tau2[0].sqrt()
    );
    for ((&b, &rb), (&s, &rs)) in f.beta.iter().zip(&REF_BETA).zip(f.se.iter().zip(&REF_SE)) {
        assert!((b - rb).abs() / rb.abs() < 2e-3, "β = {b} vs lme4 {rb}");
        assert!((s - rs).abs() / rs < 3e-2, "se = {s} vs lme4 {rs}");
    }
}

/// Gamma INVERSE-link GLMM `y ~ 1 + x + grp + (1|cluster)` on sim_gamma, gated
/// against frozen `glmer(family=Gamma("inverse"))`
/// (`validation/goldens/sim_gamma_inv_glmm.json`, `tolPwrss = 1e-13`). Same data and
/// formula as the log-link test below — only the link differs, which is what
/// makes it a controlled pair.
///
/// Regression guard for the FD-Hessian seeding bug: `fd_hessian_cov` used to
/// re-derive the random-effect mode û(γ̂) by a COLD PIRLS solve rather than
/// reusing the one the fit had just converged to. Where the mode problem has more
/// than one basin the cold solve lands in a different one, and the inverse link is
/// where that shows: the fit reaches deviance 936.7683 and the cold re-eval at the
/// same γ̂ returned 1034.5678. The finite differences then straddled the two
/// branches, the joint Hessian came out indefinite, the RX fallback's Schur was
/// indefinite at the same wrong mode, and the whole fit was reported failed for
/// want of a standard error — `converged` was `false` and every estimate NaN.
///
/// So `converged` is the assertion that would have caught it, and `loglik` is the
/// one that keeps catching it: landing on the wrong branch moves the
/// log-likelihood by ~49, which no band here tolerates. Bands are `validation/tol.R`'s
/// cross-engine ones throughout — every constant below is lme4's.
#[test]
fn fit_glmm_gamma_inverse_link_matches_lme4() {
    const REF_BETA: [f64; 3] = [0.75205795080653, -0.187572954875194, -0.140275024148733];
    const REF_SE: [f64; 3] = [0.0757958253868104, 0.0174915990199966, 0.0340264529562919];
    const REF_CLUSTER_SD: f64 = 0.243083546786158;
    const REF_DISP: f64 = 0.578838313863376;
    const REF_LOGLIK: f64 = -468.38415378098;

    let (x, y, cluster_ids, n_clusters) = sim_clustered(include_str!(
        "../../validation/data/simulated/sim_gamma.csv"
    ));
    let (n, p) = (y.len(), 3);
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Inverse,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    // Default `WaldSe::Hessian` deliberately — the Rx arm never touches
    // `fd_hessian_cov` and stayed green all through the bug.
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: cluster_ids,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );

    assert!(
        f.converged,
        "gamma-inverse GLMM must converge (the FD-Hessian must anchor on the fit's own mode)"
    );
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
            "β[{j}] = {} vs lme4 {}",
            f.beta[j],
            REF_BETA[j]
        );
        assert!(
            (f.se[j] - REF_SE[j]).abs() / REF_SE[j] < 1e-3,
            "se_hessian[{j}] = {} vs lme4 {}",
            f.se[j],
            REF_SE[j]
        );
    }
    let (sd, _corr) = f.stddev_corr(0);
    assert!(
        (sd[0] - REF_CLUSTER_SD).abs() / REF_CLUSTER_SD < 1e-3,
        "cluster sd = {} vs lme4 {REF_CLUSTER_SD}",
        sd[0]
    );
    assert!(
        (f.dispersion - REF_DISP).abs() / REF_DISP < 1e-3,
        "φ̂ = {} vs lme4 {REF_DISP}",
        f.dispersion
    );
    // The branch check: the discarded mode sat ~98 deviance units above this one,
    // so a wrong-basin fit misses here by ~49 even if it manages to report SEs.
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
}

/// Gamma log-link GLMM `y ~ 1 + x + grp + (1|cluster)` on sim_gamma, gated
/// against frozen `glmer(family=Gamma("log"))` (`validation/goldens/sim_gamma_glmm.json`).
/// φ̂ is the post-fit Pearson moment on conditional-mode residuals (matches the
/// oracle's hand-computed `Σpearson²/(n−p)`). lme4-only SE. The oracle is sacred.
//
// The dispersion enters glmer's Gamma fit ONLY through the family `aic` term in
// the Laplace objective (profiled `disp=D/n`), not via 1/φ-weighted PIRLS or a
// φ-ridge (confirmed against lme4 src/glmFamily.cpp; MixedModels.jl decouples
// entirely and PQL/glmmPQL uses a φ-ridge — both are *different* estimators).
// The kernel swaps `D → gamma_aic` in `laplace_deviance`, so β̂/τ̂ and the
// FD-Hessian SE pick up the coupling. See `family::gamma_aic`.
#[test]
fn fit_glmm_gamma_sim_matches_lme4() {
    const REF_BETA: [f64; 3] = [0.308930805779, 0.577841416651, 0.455706877075];
    const REF_SE: [f64; 3] = [0.139098615851, 0.0427935407665, 0.0883045165218];
    // Golden's `se_rx` = lme4 `vcov(use.hessian=FALSE)`, σ̂²-scaled for Gamma —
    // gates the kernel's `WaldSe::Rx` σ̂² factor (`family::glmm_sigma_sq`).
    const REF_SE_RX: [f64; 3] = [0.116924273630386, 0.0453773644154408, 0.0929163554683392];
    const REF_CLUSTER_SD: f64 = 0.4851167757;
    const REF_DISP: f64 = 0.5265553674;
    let (x, y, cluster_ids, n_clusters) = sim_clustered(include_str!(
        "../../validation/data/simulated/sim_gamma.csv"
    ));
    let (n, p) = (y.len(), 3);
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
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
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "gamma GLMM must converge");
    let disp_rel = (f.dispersion - REF_DISP).abs() / REF_DISP;
    assert!(disp_rel < 2e-2, "φ̂ = {} vs lme4 {REF_DISP}", f.dispersion);
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 2e-3,
            "β[{j}] = {} vs lme4 {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(se_rel < 3e-2, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
    }
    // Via stddev_corr/varcorr — σ̂²-scaled like tau2 (B1 fix), so it gates
    // the public accessor directly against lme4's VarCorr stddev.
    let (sd, _corr) = f.stddev_corr(0);
    let sd_rel = (sd[0] - REF_CLUSTER_SD).abs() / REF_CLUSTER_SD;
    assert!(
        sd_rel < 5e-3,
        "cluster sd (stddev_corr) = {} vs lme4 {REF_CLUSTER_SD}",
        sd[0]
    );
    assert!(
        (sd[0] - f.tau2[0].sqrt()).abs() < 1e-12,
        "stddev_corr and tau2 must report the same σ̂-scaled sd"
    );
    // lme4 logLik (validation/results/lme4_simulated/sim_gamma.json) — pins the
    // Gamma rule loglik = −½·deviance verbatim: lme4's glmer logLik is
    // −devfun/2 with gamma_aic's +2 left inside (1 below Σ log f).
    const REF_LOGLIK: f64 = -445.173519506374;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 5e-3,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 5); // 3 β + cluster θ + φ

    // Rx arm on the same design vs the golden's σ̂²-scaled `se_rx`.
    let f_rx = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: cluster_ids,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2],
            wald_se: WaldSe::Rx,
            ..FitOptions::default()
        },
    );
    assert!(f_rx.converged, "gamma GLMM (Rx) must converge");
    #[allow(clippy::needless_range_loop)]
    for j in 0..p {
        let se_rel = (f_rx.se[j] - REF_SE_RX[j]).abs() / REF_SE_RX[j];
        assert!(
            se_rel < 3e-2,
            "rx se[{j}] = {} vs lme4 {}",
            f_rx.se[j],
            REF_SE_RX[j]
        );
    }
}

/// NB GLMM `y ~ 1 + x + grp + (1|cluster)` on sim_nb via the outer-θ loop,
/// gated against frozen `lme4::glmer.nb` (`validation/goldens/sim_nb_glmm.json`).
/// `dispersion = θ̂`. lme4-only SE. The oracle is sacred.
#[test]
fn fit_glmm_nb_sim_matches_lme4() {
    const REF_BETA: [f64; 3] = [-0.0207782143496, 0.593950952004, 0.59944069353];
    const REF_SE: [f64; 3] = [0.163165315799, 0.0721272221837, 0.141480120735];
    const REF_CLUSTER_SD: f64 = 0.5742029807;
    const REF_THETA: f64 = 1.783620004;
    let (x, y, cluster_ids, n_clusters) =
        sim_clustered(include_str!("../../validation/data/simulated/sim_nb.csv"));
    let (n, p) = (y.len(), 3);
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
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
            primary: cluster_ids.clone(),
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "NB GLMM must converge");
    let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
    assert!(th_rel < 5e-2, "θ̂ = {} vs lme4 {REF_THETA}", f.dispersion);
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 5e-3
                || (f.beta[j] - REF_BETA[j]).abs() < 5e-3,
            "β[{j}] = {} vs lme4 {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
        assert!(se_rel < 5e-2, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
    }
    let sd_rel = (f.tau2[0].sqrt() - REF_CLUSTER_SD).abs() / REF_CLUSTER_SD;
    assert!(
        sd_rel < 2e-2,
        "cluster sd = {} vs lme4 {REF_CLUSTER_SD}",
        f.tau2[0].sqrt()
    );
    // lme4 logLik (validation/goldens/sim_nb_glmm.json) — the θ-dependent NB
    // saturated constant (incl. −ln yᵢ!) restored at θ̂. Wider band than the
    // φ≡1 families: the constant itself moves with θ̂ (5e-2 rel above).
    const REF_LOGLIK: f64 = -481.455529976646;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-2,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
    assert_eq!(f.df, 5); // 3 β + cluster θ_RE + NB θ
}

/// NB GLMM on an UNBALANCED NESTED design: `y ~ 1 + x + (1|g1/g2)` on
/// sim_nb_nested (per-g1 sizes 8..120 on an exp ladder), gated against
/// frozen `lme4::glmer.nb` (`validation/goldens/sim_nb_nested_glmm.json`).
/// The nested extra rides the Pastes convention: `GroupIds.extra` carries
/// the globally-unique g1:g2 level, `NestedWithin` is the topology tag,
/// placeholder counts prove sizing comes from the ids. `dispersion = θ̂`;
/// lme4-only SE (Hessian, glmm's default). tau2 layout: [primary g1 |
/// nested g2:g1] — the golden's varcomp lists g2:g1 first (lme4 orders by
/// descending level count). The oracle is sacred.
#[test]
fn fit_glmm_nb_nested_unbalanced_matches_lme4() {
    const REF_BETA: [f64; 2] = [0.584998228282064, 0.507364808670142];
    const REF_SE_HESSIAN: [f64; 2] = [0.204822249488268, 0.0539927793867315];
    const REF_G1_SD: f64 = 0.629024806733981;
    const REF_NEST_SD: f64 = 0.355202234990849;
    const REF_THETA: f64 = 1.43012979314052;
    // sim_nb_nested.csv: y,x,g1,g2 (g2 labels reused across g1 parents).
    let csv = include_str!("../../validation/data/simulated/sim_nb_nested.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut g1_raw = Vec::<String>::new();
    let mut nest_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        xcol.push(f[1].parse().unwrap());
        g1_raw.push(f[2].to_string());
        // Globally-unique nested level, the Pastes "sample" convention.
        nest_raw.push(format!("{}:{}", f[2], f[3]));
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let (g1, _n_g1) = dense_str(&g1_raw);
    let (nest, _n_nest) = dense_str(&nest_raw);
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 1 }, // placeholder
                slopes: vec![],
            }],
        }),
    };
    let f = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds {
            primary: g1,
            extra: vec![nest],
        },
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "nested NB GLMM must converge");
    let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
    assert!(th_rel < 5e-2, "θ̂ = {} vs lme4 {REF_THETA}", f.dispersion);
    for j in 0..p {
        assert!(
            (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 5e-3
                || (f.beta[j] - REF_BETA[j]).abs() < 5e-3,
            "β[{j}] = {} vs lme4 {}",
            f.beta[j],
            REF_BETA[j]
        );
        let se_rel = (f.se[j] - REF_SE_HESSIAN[j]).abs() / REF_SE_HESSIAN[j];
        assert!(
            se_rel < 5e-2,
            "se[{j}] = {} vs lme4 {}",
            f.se[j],
            REF_SE_HESSIAN[j]
        );
    }
    let g1_rel = (f.tau2[0].sqrt() - REF_G1_SD).abs() / REF_G1_SD;
    assert!(
        g1_rel < 2e-2,
        "g1 sd = {} vs lme4 {REF_G1_SD}",
        f.tau2[0].sqrt()
    );
    let nest_rel = (f.tau2[1].sqrt() - REF_NEST_SD).abs() / REF_NEST_SD;
    assert!(
        nest_rel < 2e-2,
        "g2:g1 sd = {} vs lme4 {REF_NEST_SD}",
        f.tau2[1].sqrt()
    );
}

/// AGQ-bypass canary (the `GLMM_RHO_END` canary's two-stage counterpart).
/// nAGQ>1 fits bypass stage 1: the `two_stage && nagq == 1` gate
/// excludes them (Profile deviance is undefined on the AGQ early-return path,
/// `debug_assert!(!profile_beta || nagq == 1)`), so setting `ws.two_stage = true`
/// on an AGQ fit must be a strict no-op. Runs the Poisson grouseticks AGQ fixture
/// (nAGQ=7) through `crate::glmm::fit_glmm` both ways and asserts β̂, θ̂, τ̂², and
/// n_eval are BIT-identical — the bypass is clean. (A Laplace-pass warm start for
/// AGQ was measured on the diligent AGQ cells (2026-07-14) and reverted as a wash;
/// the bypass this canary pins is the shipped state.)
#[test]
fn two_stage_agq_bypass_is_bit_identical() {
    let csv = include_str!("../../validation/data/empirical/grouseticks.csv");
    let p = 4;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        raw.push(f[0].parse().unwrap());
        let year: u32 = f[4].parse().unwrap();
        x.extend_from_slice(&[
            1.0,
            f64::from(u32::from(year == 96)),
            f64::from(u32::from(year == 97)),
            f[6].parse().unwrap(),
        ]);
        y.push(f[1].parse().unwrap());
    }
    let (cluster_ids, n_clusters) = dense_ids(&raw);
    let n = y.len();
    let nagq = 7u8;
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let sized = spec_sized_from_ids_pub(
        &model,
        &GroupIds {
            primary: cluster_ids.clone(),
            extra: vec![],
        },
    );
    let mut xm = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xm[(i, j)] = x[i * p + j];
        }
    }
    let beta_start = glm_warm_start_beta(
        sized.family,
        f64::NAN,
        xm.as_ref().subrows(0, n),
        &y,
        n,
        p,
        None,
    );

    let run = |two_stage: bool| -> (Vec<f64>, Vec<f64>, f64, usize) {
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &sized, n, &[], nagq);
        build_z(&mut ws, xm.as_ref().subrows(0, n), &cluster_ids, &[], n);
        ws.two_stage = two_stage;
        let fit = crate::glmm::fit_glmm(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &y,
            &cluster_ids,
            &[0, 1, 2, 3],
            None,
            &beta_start,
            n,
            WaldSe::Rx,
        );
        assert!(
            fit.converged,
            "AGQ fit (two_stage={two_stage}) must converge"
        );
        (
            ws.betas[..p].to_vec(),
            ws.params[..ws.n_theta].to_vec(),
            fit.tau_squared_hat,
            fit.n_eval,
        )
    };
    let (b1, t1, tau1, ne1) = run(false);
    let (b2, t2, tau2, ne2) = run(true);
    for j in 0..p {
        assert_eq!(
            b1[j].to_bits(),
            b2[j].to_bits(),
            "AGQ bypass: β[{j}] must be bit-identical"
        );
    }
    for t in 0..t1.len() {
        assert_eq!(
            t1[t].to_bits(),
            t2[t].to_bits(),
            "AGQ bypass: θ[{t}] must be bit-identical"
        );
    }
    assert_eq!(
        tau1.to_bits(),
        tau2.to_bits(),
        "AGQ bypass: τ̂² must be bit-identical"
    );
    assert_eq!(
        ne1, ne2,
        "AGQ bypass: n_eval must be identical (stage 1 skipped)"
    );
}

/// Two-stage A/B for a fixture whose data/model helpers live only in fit.rs's
/// private `#[cfg(test)]` module (unreachable from glmm/tests.rs). Mirrors
/// glmm/tests.rs `assert_two_stage_matches_single`: two fresh workspaces —
/// single- vs two-stage — must land on the same optimum at ORACLE tolerances
/// (β_rel 1e-3; θ abs+rel 1e-3 band; τ² rel 1e-3). Prints the
/// `(n_eval_single, n_eval_two)` pair for the baseline doc; NO n_eval assertion —
/// the eval-count win is a separate, measured concern. Drives
/// `crate::glmm::fit_glmm` directly so `ws.two_stage` is settable.
fn assert_two_stage_matches_single_local(
    label: &str,
    model: &ModelSpec,
    x: &[f64],
    y: &[f64],
    ids: &GroupIds,
    n: usize,
    p: usize,
) -> (usize, usize) {
    let sized = spec_sized_from_ids_pub(model, ids);
    let mut xm = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xm[(i, j)] = x[i * p + j];
        }
    }
    let beta_start = glm_warm_start_beta(
        sized.family,
        f64::NAN,
        xm.as_ref().subrows(0, n),
        y,
        n,
        p,
        None,
    );
    let targets: Vec<u32> = (0..p as u32).collect();

    let run = |two_stage: bool| -> (Vec<f64>, Vec<f64>, f64, usize) {
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &sized, n, &[], 1);
        ws.nb_theta = f64::NAN; // non-NB families ignore it (mirrors fit_glmm_impl)
        build_z(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &ids.primary,
            &ids.extra,
            n,
        );
        ws.structured_schur = if ws.groupings.structured_extras_eligible() {
            StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
        } else {
            None
        };
        ws.two_stage = two_stage;
        let fit = crate::glmm::fit_glmm(
            &mut ws,
            xm.as_ref().subrows(0, n),
            y,
            &ids.primary,
            &targets,
            None,
            &beta_start,
            n,
            WaldSe::Rx,
        );
        assert!(
            fit.converged,
            "{label}: {} fit must converge",
            if two_stage {
                "two-stage"
            } else {
                "single-stage"
            }
        );
        (
            ws.betas[..p].to_vec(),
            ws.params[..ws.n_theta].to_vec(),
            fit.tau_squared_hat,
            fit.n_eval,
        )
    };
    let (b1, t1, tau1, ne1) = run(false);
    let (b2, t2, tau2, ne2) = run(true);
    for j in 0..p {
        let rel = (b1[j] - b2[j]).abs() / b1[j].abs().max(1e-6);
        assert!(
            rel < 1e-3,
            "{label}: β[{j}] single {} vs two-stage {} (rel {rel})",
            b1[j],
            b2[j]
        );
    }
    for t in 0..t1.len() {
        assert!(
            (t1[t] - t2[t]).abs() < 1e-3 * (1.0 + t1[t].abs()),
            "{label}: θ[{t}] single {} vs two-stage {}",
            t1[t],
            t2[t]
        );
    }
    let trel = (tau1 - tau2).abs() / tau1.abs().max(1e-6);
    assert!(
        trel < 1e-3,
        "{label}: τ² single {tau1} vs two-stage {tau2} (rel {trel})"
    );
    println!("{label} n_eval: single {ne1} vs two {ne2}");
    (ne1, ne2)
}

/// Two-stage A/B on the two GLMM fixtures whose helpers are private to this
/// module — the cbpp probit binomial GLMM (non-canonical link, blocked path,
/// lme4-validated) and the sim_gamma log-link mixed model (a distinct
/// non-canonical / dispersion PIRLS path with zero prior two-stage coverage).
/// `#[ignore]`: part of the explicit two-stage corpus proof, out of the fast
/// suite (like the glmm/tests.rs corpus sweep).
#[test]
#[ignore]
fn two_stage_matches_single_stage_cbpp_probit_and_gamma() {
    // cbpp probit binomial GLMM (blocked, non-canonical probit link).
    {
        let (x, y, cluster_ids, n) = cbpp_design();
        let mut model = cbpp_model();
        model.family = Family::Binomial {
            link: BinomialLink::Probit,
        };
        let ids = GroupIds {
            primary: cluster_ids,
            extra: vec![],
        };
        assert_two_stage_matches_single_local("cbpp_probit", &model, &x, &y, &ids, n, 4);
    }
    // Gamma log-link mixed model (blocked, non-canonical + dispersion PIRLS path).
    {
        let (x, y, cluster_ids, n_clusters) = sim_clustered(include_str!(
            "../../validation/data/simulated/sim_gamma.csv"
        ));
        let n = y.len();
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let ids = GroupIds {
            primary: cluster_ids,
            extra: vec![],
        };
        assert_two_stage_matches_single_local("gamma_sim", &model, &x, &y, &ids, n, 3);
    }
}

// ── Vector-RE AGQ goldens (GLMMadaptive oracle) ─────────────────────────────

/// One vector-AGQ pin: parse a `y,<covariates...>,g` CSV, fit at `nagq`
/// (primary grouping factor, random slopes on every covariate → q_p = 1 + n_x,
/// routed through `agq_deviance_vec`), and check β̂ / Hessian SE / RE stddev /
/// RE correlation against values recorded from glmm.
///
/// `corr_lower` is the strict lower triangle in row order — (1,0) for q=2;
/// (1,0), (2,0), (2,1) for q=3. The diagonal is 1 by construction and the
/// upper triangle mirrors, so pinning them would assert nothing.
///
/// Cross-engine validation of every one of these fits lives in the
/// `sim_{binomial,poisson}_slope{1,2}_agq_k{7,11}` cells, against frozen
/// `GLMMadaptive::mixed_model(nAGQ=k)`. Those cells run at the wider `agq_*`
/// bands from `validation/tol.R`, because GLMMadaptive's quadrature details differ
/// from ours (per-step re-adaptation, a different RE-covariance
/// parameterization) — matched-k agreement is tight but not machine-precision.
/// That is a fact about the two engines and has no bearing on how tightly glmm
/// reproduces its own answer, which is what this pins.
///
/// No deviance pin: the deviance scale is owned by the in-crate k-convergence
/// invariants in `glmm/tests.rs`.
#[allow(clippy::too_many_arguments)]
fn check_vector_agq_pin(
    name: &str,
    csv: &str,
    nagq: u8,
    n_x: usize,
    family: Family,
    ref_beta: &[f64],
    ref_se: &[f64],
    ref_stddev: &[f64],
    corr_lower: &[f64],
) {
    let mut y = Vec::<f64>::new();
    let mut xc: Vec<Vec<f64>> = vec![vec![]; n_x];
    let mut g_raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        for (k, col) in xc.iter_mut().enumerate() {
            col.push(f[1 + k].parse().unwrap());
        }
        g_raw.push(f[1 + n_x].parse().unwrap());
    }
    let n = y.len();
    let p = 1 + n_x;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        for k in 0..n_x {
            x[i * p + 1 + k] = xc[k][i];
        }
    }
    let (primary, n_clusters) = dense_ids(&g_raw);
    let ids = GroupIds {
        primary,
        extra: vec![],
    };
    let model = ModelSpec {
        family,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: (1..=n_x as u32).collect(),
            extra_groupings: vec![],
        }),
    };
    let opts = FitOptions {
        target_indices: (0..p as u32).collect(),
        nagq,
        ..FitOptions::default() // WaldSe::Hessian — the se_hessian partner
    };
    let f = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(f.converged, "{name} k={nagq} must converge");

    let what = format!("{name} k={nagq}");
    assert_pinned(&f.beta, ref_beta, PIN_REL_ITER, &format!("{what} beta"));
    assert_pinned(&f.se, ref_se, PIN_REL_ITER, &format!("{what} se"));
    let (stddev, corr) = f.stddev_corr(0);
    assert_pinned(&stddev, ref_stddev, PIN_REL_ITER, &format!("{what} stddev"));
    let lower: Vec<f64> = (0..stddev.len())
        .flat_map(|t| (0..t).map(move |u| (t, u)))
        .map(|(t, u)| corr[t][u])
        .collect();
    assert_pinned(&lower, corr_lower, PIN_REL_ITER, &format!("{what} corr"));
}

/// Vector AGQ (q=2), binomial: `y ~ x + (1 + x | g)` on sim_binomial_slope1 at
/// nAGQ ∈ {7, 11}. Validated cross-engine by the
/// `sim_binomial_slope1_agq_k{7,11}` cells against GLMMadaptive.
#[test]
fn fit_glmm_binomial_slope1_vector_agq_is_pinned() {
    let csv = include_str!("../../validation/data/simulated/sim_binomial_slope1.csv");
    let family = Family::Binomial {
        link: BinomialLink::Logit,
    };
    check_vector_agq_pin(
        "sim_binomial_slope1",
        csv,
        7,
        1,
        family,
        &[0.42897645003558754, 0.43458315523970653],
        &[0.1562674670018699, 0.1310475868542789],
        &[0.8758219936062426, 0.3726397459266911],
        &[0.5283132632086392],
    );
    check_vector_agq_pin(
        "sim_binomial_slope1",
        csv,
        11,
        1,
        family,
        &[0.42897959050352286, 0.43458759206119546],
        &[0.15627012841005886, 0.13105001203556532],
        &[0.8758453743269463, 0.3726611962891021],
        &[0.5283433982271978],
    );
}

/// Vector AGQ (q=2), Poisson: `y ~ x + (1 + x | g)` on sim_poisson_slope1 at
/// nAGQ ∈ {7, 11}. Validated cross-engine by the
/// `sim_poisson_slope1_agq_k{7,11}` cells against GLMMadaptive.
#[test]
fn fit_glmm_poisson_slope1_vector_agq_is_pinned() {
    let csv = include_str!("../../validation/data/simulated/sim_poisson_slope1.csv");
    let family = Family::Poisson {
        link: crate::PoissonLink::Log,
    };
    check_vector_agq_pin(
        "sim_poisson_slope1",
        csv,
        7,
        1,
        family,
        &[-1.3538764484761825, 0.49328234955414413],
        &[0.21833946383750474, 0.18544416147664192],
        &[1.0117923068022499, 0.33570973669422155],
        &[0.03689193891989874],
    );
    check_vector_agq_pin(
        "sim_poisson_slope1",
        csv,
        11,
        1,
        family,
        &[-1.353312567495963, 0.493094585284533],
        &[0.21807780917884897, 0.18530325400802838],
        &[1.0106390790816442, 0.33617266187700573],
        &[0.037212955698941674],
    );
}

/// Vector AGQ (q=3), binomial: `y ~ x1 + x2 + (1 + x1 + x2 | g)` on
/// sim_binomial_slope2 at nAGQ ∈ {7, 11} — the q_p ≤ 3 cap surface and the
/// kernel's dimensional generality. Validated cross-engine by the
/// `sim_binomial_slope2_agq_k{7,11}` cells against GLMMadaptive.
#[test]
fn fit_glmm_binomial_slope2_vector_agq_is_pinned() {
    let csv = include_str!("../../validation/data/simulated/sim_binomial_slope2.csv");
    let family = Family::Binomial {
        link: BinomialLink::Logit,
    };
    check_vector_agq_pin(
        "sim_binomial_slope2",
        csv,
        7,
        2,
        family,
        &[0.3730517271148301, 0.538098978038483, -0.3654854470965549],
        &[
            0.13111390541787663,
            0.10465308883561775,
            0.10556860651205546,
        ],
        &[1.0640265640371298, 0.6217295819798455, 0.6420938363939653],
        &[
            0.21906654236295983,
            0.13220156109374376,
            -0.030986539519051545,
        ],
    );
    check_vector_agq_pin(
        "sim_binomial_slope2",
        csv,
        11,
        2,
        family,
        &[0.3730629587832013, 0.5381087834502509, -0.3654883076141031],
        &[
            0.13112095270455848,
            0.10465667876714824,
            0.10557212730472322,
        ],
        &[1.0640974804776098, 0.6217673168423941, 0.6421365960857665],
        &[
            0.21910550941911974,
            0.13222240391317389,
            -0.03101919706448947,
        ],
    );
}

/// Boundary singular flag on a GLMM: a scalar random-intercept
/// binomial-logit fit with NO cluster signal (`y` drawn from a
/// fixed-effects-only logit; the grouping factor is present but no cluster
/// deviation is added to `eta`) must pin θ̂ ≈ 0 and set `Fit::singular`,
/// mirroring the LMM boundary case
/// (`fit_lmm_weighted_boundary_matches_wls`'s `mixed.singular` assert) but
/// for the GLMM path, which sets `singular` from `boundary_hit` OR
/// `has_negligible_component` (`fit/glmm.rs`'s `SINGULAR_REL_TOL` check) —
/// neither of which any existing GLMM test exercises. 40 clusters × 10 reps
/// (n=400): fewer clusters/reps left tau2[0] at a small positive REML
/// estimate instead of pinning at the boundary (finite-sample cluster-mean
/// noise still readable as signal) — verified empirically, not a guess.
#[test]
fn fit_glmm_binomial_no_cluster_signal_is_singular() {
    let n_clusters = 40u32;
    let reps = 10usize;
    let n = n_clusters as usize * reps;
    let p = 2;
    let mut xm = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    let mut cl = vec![0u32; n];
    let mut st = 11u64;
    let mut i = 0;
    for c in 0..n_clusters {
        for _ in 0..reps {
            let cov = crate::sparse::test_lcg(&mut st);
            // Fixed-effects-only logit — no per-cluster deviation added, so
            // the true random-intercept variance is exactly zero.
            let eta = 0.3 + 0.5 * cov;
            let prob = 1.0 / (1.0 + (-eta).exp());
            let draw = (crate::sparse::test_lcg(&mut st) + 1.0) / 2.0;
            xm[(i, 0)] = 1.0;
            xm[(i, 1)] = cov;
            cl[i] = c;
            y[i] = if draw < prob { 1.0 } else { 0.0 };
            i += 1;
        }
    }
    let mut x = vec![0.0f64; n * p];
    for row in 0..n {
        for col in 0..p {
            x[row * p + col] = xm[(row, col)];
        }
    }
    let ids = GroupIds {
        primary: cl,
        extra: vec![],
    };
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters },
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
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    assert!(f.converged, "no-signal GLMM must still converge");
    assert!(f.singular, "must flag the θ≈0 boundary as singular");
    assert!(
        f.tau2[0] < 1e-4,
        "tau2[0] must pin near zero, got {}",
        f.tau2[0]
    );
}

/// The loop tier must fit the SAME model `fit_cold` does when an extra grouping
/// carries a random slope. The dense GLMM kernel builds intercept-only extras, so
/// reaching it with such a design fits a reduced model and reports it as a normal
/// success; `build_workspace`'s `classify_design` call is what keeps the loop tier
/// off it. Nothing else pins the loop-tier entry itself — the classifier test
/// (`classify_routes_slope_extras_to_sparse_all_families`) pins only the routing
/// decision, not that the built workspace acts on it.
#[test]
fn loop_tier_honours_extra_grouping_slope() {
    let (n_g1, n_g2, per) = (8usize, 6usize, 10usize);
    let n = n_g1 * per;
    let mut st = 7u64;
    let (mut x, mut y) = (vec![0.0f64; n * 2], vec![0.0f64; n]);
    let (mut g1, mut g2) = (vec![0u32; n], vec![0u32; n]);
    for i in 0..n {
        g1[i] = (i % n_g1) as u32;
        g2[i] = (i % n_g2) as u32;
        let x1 = crate::sparse::test_lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        // Extra-grouping slope variance ≈ 0.9 in the fitted model — big enough that
        // dropping the slope would move β̂ well past any optimizer tolerance.
        let eta = 0.5 + 0.4 * x1 + 0.25 * (g1[i] as f64 - 4.0) + 0.6 * x1 * (g2[i] as f64 - 3.0);
        y[i] = eta.exp().round();
    }
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![1],
            }],
        }),
    };
    let ids = GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };

    let cold = fit_cold(&x, &y, n, 2, &model, &ids, &opts);
    assert!(cold.converged, "reference fit must converge");
    // tau2 layout: primary intercept, then the extra grouping's 2×2 vech block.
    assert!(
        cold.tau2.len() == 4 && cold.tau2[2] > 0.1,
        "the draw must actually carry an extra-grouping slope variance: tau2 {:?}",
        cold.tau2
    );

    let sized = super::common::spec_sized_from_ids(&model, &ids);
    let mut ws = super::core::build_workspace(&sized, n, 2, &opts);
    let view = super::core::fit_on(&mut ws, &x, &y, &ids, None, &opts);
    for j in 0..2 {
        assert_eq!(
            view.betas()[j],
            cold.beta[j],
            "loop tier must reach fit_cold's β exactly: β[{j}]"
        );
    }
}
