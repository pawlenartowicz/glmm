//! GLMM estimator tests (Binomial/Poisson/Gamma/negative-binomial,
//! `re: Some`, dense + sparse-Schur equivalence + AGQ + two-stage).

use super::*;
use crate::glmm::{build_z, glmm_laplace_deviance, GlmmWorkspace, OuterSearch, StructuredSchur};
use crate::test_support::assert_near;
use crate::{
    BinomialLink, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing,
    StartValues, WaldSe,
};
use faer::Mat;

use super::common_tests::{assert_pinned, dense_ids, dense_str, sim_clustered};

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
    assert!(cold.converged() && via.converged());
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
    assert!(cold.converged(), "cold cbpp GLMM must converge");
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
        assert!(
            warm.converged(),
            "{label}: warm must not degrade convergence"
        );
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

/// Per-component cold start on the cbpp binomial GLMM: an EMPTY `beta` or
/// `theta` cold-starts that component alone. The ports need this — lme4's
/// `start = list(theta = …)` supplies θ and nothing else, and neither the R nor
/// the Python wrapper can synthesize the missing β (the cold seed is a no-RE GLM
/// fit computed inside the kernel).
///
/// Both-empty is the strict arm: it must be BIT-identical to `fit_cold`, since
/// it takes every cold branch. The one-sided arms only have to land on the cold
/// optimum, like the warm arms above.
#[test]
fn fit_warm_glmm_partial_start_cold_starts_the_missing_component() {
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
    assert!(cold.converged(), "cold cbpp GLMM must converge");

    let both_empty = StartValues {
        beta: vec![],
        theta: vec![],
    };
    let empty = fit_warm(&x, &y, n, p, &model, &ids, Some(&both_empty), &opts);
    // Bitwise (not PartialEq): non-target SE slots are NaN and NaN != NaN.
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&cold.beta), bits(&empty.beta));
    assert_eq!(bits(&cold.se), bits(&empty.se));
    assert_eq!(bits(&cold.tau2), bits(&empty.tau2));

    let starts = [
        // θ only (lme4's `start = list(theta = …)`), θ̂ ≈ 0.64 perturbed to 3.
        (
            "theta-only",
            StartValues {
                beta: vec![],
                theta: vec![3.0],
            },
        ),
        // β only: halved β̂, θ falls back to the THETA0 blind start.
        (
            "beta-only",
            StartValues {
                beta: cold.beta.iter().map(|b| 0.5 * b).collect(),
                theta: vec![],
            },
        ),
    ];
    for (label, start) in &starts {
        let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
        assert!(warm.converged(), "{label}: must converge");
        for j in 0..p {
            let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
            assert!(
                rel < 1e-3,
                "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                warm.beta[j],
                cold.beta[j]
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
    assert!(hess.converged() && rx.converged());
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
/// MixedModels.jl drops it ~3%). The oracle is sacred: on
/// disagreement glmm is presumed wrong.
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

    assert!(f.converged(), "cbpp GLMM must converge");
    assert!(
        f.diagnostics.notes.is_empty(),
        "a well-behaved fit carries no PirlsExhausted note, got {:?}",
        f.diagnostics.notes
    );
    // Bands are validation/tol.R's cross-engine numbers (beta_rel, se_hessian_rel,
    // stddev_rel = 1e-3) — change together with that file. This is a glmm↔lme4
    // claim, so tol.R's calibration is the one that applies. Measured agreement
    // against the artifact-free reference is far inside them: SE worst 6.0e-6.
    // The SE band was 3e-2 only because the constants above were the
    // default-tolPwrss ones; with the citation corrected, that band no longer
    // has a reason to exist. The oracle is sacred — these bound glmm to lme4,
    // never the reverse.
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
    assert!(f.converged(), "aggregated cbpp GLMM must converge");
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
    // under prior weights.
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
    assert!(
        f0.converged(),
        "no-offset aggregated cbpp GLMM must converge"
    );

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
    assert!(
        f_off.converged(),
        "offset aggregated cbpp GLMM must converge"
    );

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
            fe.converged() && fa.converged(),
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
    assert!(f.converged(), "weighted Poisson GLMM must converge");
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
    assert!(f.converged(), "weighted Gamma GLMM must converge");
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
/// Exercises the blocked PIRLS path for a non-binomial family. lme4-only SE.
/// The oracle is sacred.
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
    assert!(f.converged(), "poisson GLMM must converge");
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
    assert!(f.converged(), "offset poisson GLMM must converge");
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
        f.converged(),
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
    let (model, ids, _perm) = spec_sized_from_ids_pub(&model, &ids);
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
        &ids.extra,
        n,
    );
    ws.force_dense_schur = false;
    let dev_sparse = glmm_laplace_deviance(
        &params,
        &mut ws,
        xm.as_ref().subrows(0, n),
        &y,
        &ids.primary,
        &ids.extra,
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
    let (model, ids, _perm) = spec_sized_from_ids_pub(&model, &ids);
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
            &ids.extra,
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
    let (model, ids, _perm) = spec_sized_from_ids_pub(&model, &ids);
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
            &ids.extra,
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
/// bias is integrated out (herd sd 0.642→0.648). The oracle is sacred.
///
/// **β + varcomp only, deliberately.** The `cbpp_agq_k*` goldens *do* carry
/// `se_hessian` (we agree with them to 6.0e-6 / 1.1e-5 / 1.4e-5 at k = 1/7/11),
/// so this is a choice about what each test owns, not a gap in the reference:
/// what k > 1 changes here is the integral, and β/varcomp are where that shows.
/// The FD-Hessian SE machinery is gated at nAGQ = 1 by
/// `fit_glmm_cbpp_matches_lme4` on this very fit, and at nAGQ = 7/11 by
/// `fit_glmm_binomial_bigsd_agq_matches_lme4`.
///
/// AGQ does not leave the SE *convention* alone: `joint_hessian_cov` differentiates
/// the deviance through `ws.nagq`, so at k > 1 it differences the AGQ deviance,
/// not the Laplace one — the SE is a property of the quadrature order like
/// everything else. What is true, measured
/// 2026-07-30 across the 0.1.4 FD-θ-step fix, is the *step rule*'s behaviour: the
/// θ-profile of the AGQ deviance obeys the same O(h²) truncation law as the
/// Laplace one, with the same constant. Dropping the `max(1, |θ̂|)` scaling
/// divides the error by θ̂² to within 6% at nAGQ = 1 and 7 and 11 alike, over
/// θ̂ ∈ [1.13, 5.16]. That is why one step rule serves every k, and why cbpp
/// (herd sd 0.647, below the `max(1, ·)` floor) is bit-identical across that fix
/// at all three orders.
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
        assert!(f.converged(), "binomial AGQ k={nagq} must converge");
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
            f.converged(),
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
        assert!(f_on.converged() && f_off.converged(), "nagq={nagq}");
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
        assert!(f.converged(), "poisson AGQ k={nagq} must converge");
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
    assert!(f.converged(), "probit GLMM must converge");
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

/// Cloglog binomial GLMM `y ~ 1 + x1 + x2 + x3 + z + (1 | g)` on the 9,600-row
/// `sim_probit_large` fixture, gated against frozen
/// `glmer(binomial("cloglog"), tolPwrss = 1e-13)`
/// (`validation/goldens/sim_cloglog_glmm.json`). lme4-only SE. No kernel change
/// was needed for this arm: `build_workspace`'s `(family, Some(re))` branch
/// already catch-alls to the dense GLMM route and PIRLS reaches the link
/// through `family_pass`. The oracle is sacred.
#[test]
fn fit_glmm_cloglog_matches_lme4() {
    const REF_BETA: [f64; 5] = [
        0.0719116500194013,
        0.523012683780339,
        -0.43319759691006,
        0.259765436565002,
        -0.631825419394664,
    ];
    const REF_SE: [f64; 5] = [
        0.0771902212322184,
        0.0174613948792221,
        0.0169682053629786,
        0.0163105409930296,
        0.032588334380417,
    ];
    const REF_STDDEV: f64 = 0.738958645035249;
    const REF_LOGLIK: f64 = -4924.21139386758;
    let csv = include_str!("../../validation/data/simulated/sim_probit_large.csv");
    let p = 5; // [intercept, x1, x2, x3, z]
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut raw_g = Vec::<String>::new();
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
        raw_g.push(f[5].to_string());
    }
    let n = y.len();
    let (cluster_ids, n_clusters) = dense_str(&raw_g);
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Cloglog,
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
            primary: cluster_ids,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2, 3, 4],
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "cloglog GLMM must converge");
    for ((&b, &rb), (&s, &rs)) in f.beta.iter().zip(&REF_BETA).zip(f.se.iter().zip(&REF_SE)) {
        assert!((b - rb).abs() / rb.abs() < 2e-3, "β = {b} vs lme4 {rb}");
        assert!((s - rs).abs() / rs < 3e-2, "se = {s} vs lme4 {rs}");
    }
    let (sd, _corr) = f.stddev_corr(0);
    assert!(
        (sd[0] - REF_STDDEV).abs() / REF_STDDEV < 3e-3,
        "g sd = {} vs lme4 {REF_STDDEV}",
        sd[0]
    );
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-3,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
}

/// Gamma INVERSE-link GLMM `y ~ 1 + x + grp + (1|cluster)` on sim_gamma, gated
/// against frozen `glmer(family=Gamma("inverse"))`
/// (`validation/goldens/sim_gamma_inv_glmm.json`, `tolPwrss = 1e-13`). Same data and
/// formula as the log-link test below — only the link differs, which is what
/// makes it a controlled pair.
///
/// Regression guard for the FD-Hessian seeding bug: `joint_hessian_cov` used to
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
    // `joint_hessian_cov` and stayed green all through the bug.
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
        f.converged(),
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
    assert!(f.converged(), "gamma GLMM must converge");
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
    assert!(f_rx.converged(), "gamma GLMM (Rx) must converge");
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

/// `WaldSe::Hessian` and `WaldSe::Rx` must report the same fitted point on
/// `sim_gamma`: `joint_hessian_cov`'s tail re-eval used to leave `ws.prob` at
/// its own re-solve while `ws.u` was put back to the pinned re-eval's mode,
/// so Gamma's σ̂² (`family::glmm_sigma_sq`) was built from a mismatched pair.
/// `tau2`/`varcorr`/`dispersion` derive from that σ̂², and `fitted` IS `ws.prob`.
#[test]
fn fit_glmm_gamma_hessian_rx_agree_on_fitted() {
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
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let f_hess = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1, 2],
            wald_se: WaldSe::Hessian,
            ..FitOptions::default()
        },
    );
    let f_rx = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1, 2],
            wald_se: WaldSe::Rx,
            ..FitOptions::default()
        },
    );
    assert!(f_hess.converged() && f_rx.converged());
    assert_eq!(
        f_hess.tau2[0].to_bits(),
        f_rx.tau2[0].to_bits(),
        "tau2[0]: {} vs {}",
        f_hess.tau2[0],
        f_rx.tau2[0]
    );
    assert_eq!(
        f_hess.varcorr[0][0].to_bits(),
        f_rx.varcorr[0][0].to_bits(),
        "varcorr[0][0]: {} vs {}",
        f_hess.varcorr[0][0],
        f_rx.varcorr[0][0]
    );
    assert_eq!(
        f_hess.dispersion.to_bits(),
        f_rx.dispersion.to_bits(),
        "dispersion: {} vs {}",
        f_hess.dispersion,
        f_rx.dispersion
    );
    for i in 0..n {
        assert_eq!(
            f_hess.fitted[i].to_bits(),
            f_rx.fitted[i].to_bits(),
            "fitted[{i}]: {} vs {}",
            f_hess.fitted[i],
            f_rx.fitted[i]
        );
    }
}

/// Same fitted-point check as [`fit_glmm_gamma_hessian_rx_agree_on_fitted`] on
/// a φ≡1 family (cbpp, binomial-logit): `glmm_sigma_sq` is a literal `1.0`
/// here, so only `fitted` can move — the split was never Gamma-specific, it
/// is just too small to see at canonical links' quadratic PIRLS convergence.
#[test]
fn fit_glmm_cbpp_hessian_rx_agree_on_fitted() {
    let (x, y, cluster_ids, n) = cbpp_design();
    let p = 4;
    let model = cbpp_model();
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let f_hess = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            wald_se: WaldSe::Hessian,
            ..FitOptions::default()
        },
    );
    let f_rx = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1, 2, 3],
            wald_se: WaldSe::Rx,
            ..FitOptions::default()
        },
    );
    assert!(f_hess.converged() && f_rx.converged());
    for i in 0..n {
        assert_eq!(
            f_hess.fitted[i].to_bits(),
            f_rx.fitted[i].to_bits(),
            "fitted[{i}]: {} vs {}",
            f_hess.fitted[i],
            f_rx.fitted[i]
        );
    }
}

/// NB GLMM `y ~ 1 + x + grp + (1|cluster)` on sim_nb via the outer-θ loop,
/// gated against frozen `lme4::glmer.nb` (`validation/goldens/sim_nb_glmm.json`).
/// `dispersion = θ̂`. lme4-only SE. The oracle is sacred.
///
/// **Additive Rust-vs-Rust pin (2026-08-06).** Gates `golden_max_ln_theta`
/// (`src/fit/glm.rs`) and the NB marginal-θ path (`fit_glmm_nb`,
/// `src/fit/glmm.rs`) directly — the lme4 bands above (5e-3 β, 5e-2 se/θ̂) are
/// too loose to tell a regression from rounding: a knife-edge
/// in the inner fixed-θ fit that a 1e-8-wide
/// golden-section stopping width could land on either side of; loosening the
/// width to 1e-4 (`glm.rs`'s own provenance comment) removed the two-branch
/// behavior. `BAND = 1e-7` is 10-50× the measured worst-case drift on this
/// fixture at the new width: 6.17e-9 relative under a 128-draw 1-ULP sweep on
/// `sim_nb`'s inputs (K=64 on `x`, K=64 on `y`), 5.16e-11 under the committed
/// `pulp` lane-width probe (scalar-forced vs normal dispatch) — both far
/// inside the band, both probes agree on the order of magnitude.
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
    assert!(f.converged(), "NB GLMM must converge");
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

    // Additive bit-exact pin (see doc comment above). Frozen at the same
    // `golden_max_ln_theta` stopping width (1e-4) this fixture's fit above
    // just ran at.
    const BAND: f64 = 1e-7;
    // re-pinned 2026-09-03: exact β-profile, θ-only outer search (was
    // [-0.02075568116330833, 0.5939541494671592, 0.5994666840076409]). This
    // shape (blocked, no extras, nAGQ=1, non-Gamma) now routes through
    // `OuterSearch::ExactProfile`.
    //
    // Re-pinned a second time the same day, once the exact-mode merit's accept
    // band was widened to carry its own correction residual
    // (`pirls_solve_blocked`): the first re-pin recorded values the warm-start
    // deadlock had produced, where uphill θ probes came back non-finite and the
    // outer search stopped early — β₀ landed at -0.020603 with θ̂_NB = 1.77368.
    // At the fixture's own θ̂_NB (1.783599975969345) the two routes now agree,
    // Δdev = exact − joint = -7.512e-8 (deviance 327.90301823213406 vs
    // 327.90301830725440), well inside `GLMM_RHO_END` (3e-6).
    const REF_BETA_PIN: [f64; 3] = [
        -0.020772684227064696,
        0.5939577927279746,
        0.5994409570567505,
    ];
    const REF_SE_PIN: [f64; 3] = [0.1631695206794065, 0.07212850508079749, 0.1414816654249278];
    const REF_TAU2_PIN: [f64; 1] = [0.32973894328360587];
    const REF_THETA_PIN: f64 = 1.783599975969345;
    assert_pinned(&f.beta, &REF_BETA_PIN, BAND, "sim_nb pinned beta");
    assert_pinned(&f.se, &REF_SE_PIN, BAND, "sim_nb pinned se");
    assert_pinned(&f.tau2, &REF_TAU2_PIN, BAND, "sim_nb pinned tau2");
    assert_pinned(
        &[f.dispersion],
        &[REF_THETA_PIN],
        BAND,
        "sim_nb pinned theta",
    );
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
///
/// **Additive Rust-vs-Rust pin (2026-08-06).** Same treatment and same
/// reasoning as `fit_glmm_nb_sim_matches_lme4`'s pin — see its doc comment.
/// This fixture is not redundant with that one: two variance blocks instead
/// of one, and its own independently measured drift, never copied from
/// `sim_nb`'s. `BAND = 1e-7` is 10-50× the measured worst case at the 1e-4
/// stopping width: 7.84e-9 relative under the 128-draw 1-ULP sweep (K=64 on
/// `x`, K=64 on `y`), 7.64e-10 under the lane-width probe.
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
    assert!(f.converged(), "nested NB GLMM must converge");
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

    // Additive bit-exact pin (see doc comment above). Frozen at the same
    // `golden_max_ln_theta` stopping width (1e-4) this fixture's fit above
    // just ran at.
    const BAND: f64 = 1e-7;
    const REF_BETA_PIN: [f64; 2] = [0.5849258898680778, 0.5073630231065611];
    // Re-pinned 2026-09-02: the exact hyper-dual joint (θ, β) Hessian replaced
    // the finite-difference stencil on the STRUCTURED extras path too, so this
    // nested fixture now takes the arm the blocked NB fixture took on
    // 2026-09-01. The fit is unchanged — β̂, τ̂², θ̂ and the deviance keep their
    // pins — so the movement is the FD stencil's own truncation-plus-noise
    // error, 9.58e-6 rel (se[0], worst) against the `se_hessian_rel` band of
    // 1e-3. Regenerate by running this test and reading the reported value; the
    // value is glmm's own, not a reference (the lme4 comparison in this same
    // test keeps its 5e-2 band and did not move).
    const REF_SE_PIN: [f64; 2] = [0.2048279321099271, 0.0539936652016425];
    const REF_TAU2_PIN: [f64; 2] = [0.39569392783021273, 0.12615822089462564];
    const REF_THETA_PIN: f64 = 1.430_063_029_750_896_3;
    assert_pinned(&f.beta, &REF_BETA_PIN, BAND, "sim_nb_nested pinned beta");
    assert_pinned(&f.se, &REF_SE_PIN, BAND, "sim_nb_nested pinned se");
    assert_pinned(&f.tau2, &REF_TAU2_PIN, BAND, "sim_nb_nested pinned tau2");
    assert_pinned(
        &[f.dispersion],
        &[REF_THETA_PIN],
        BAND,
        "sim_nb_nested pinned theta",
    );
}

/// AGQ-bypass canary (the `GLMM_RHO_END` canary's two-stage counterpart).
/// nAGQ>1 fits bypass stage 1: the `stage1_mode.filter(|_| nagq == 1)` gate
/// excludes them (Profile deviance is undefined on the AGQ early-return path,
/// `debug_assert!(beta_mode == BetaMode::Fixed || nagq == 1)`), so setting
/// `ws.outer_search = OuterSearch::PqlThenJoint` on an AGQ fit must be a strict
/// no-op. Runs the Poisson grouseticks AGQ fixture
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
    let (sized, _ids, _perm) = spec_sized_from_ids_pub(
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
        ws.outer_search = if two_stage {
            OuterSearch::PqlThenJoint
        } else {
            OuterSearch::Joint
        };
        let fit = crate::glmm::fit_glmm(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &y,
            &cluster_ids,
            &[],
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
/// `outer_search = OuterSearch::Joint` vs forced `OuterSearch::PqlThenJoint` —
/// must land on the same optimum at ORACLE tolerances
/// (β_rel 1e-3; θ abs+rel 1e-3 band; τ² rel 1e-3). Prints the
/// `(n_eval_single, n_eval_two)` pair for the baseline doc; NO n_eval assertion —
/// the eval-count win is a separate, measured concern. Drives
/// `crate::glmm::fit_glmm` directly so `ws.outer_search` is settable.
fn assert_two_stage_matches_single_local(
    label: &str,
    model: &ModelSpec,
    x: &[f64],
    y: &[f64],
    ids: &GroupIds,
    n: usize,
    p: usize,
) -> (usize, usize) {
    let (sized, ids, _perm) = spec_sized_from_ids_pub(model, ids);
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
        // false: pin the single-stage reference (`OuterSearch::Joint`). true:
        // force `PqlThenJoint` explicitly rather than trust the constructor's
        // default — `gamma_sim` (n_θ=1, p=3) now falls into the
        // `n_theta <= 2 && p <= 4` skip and defaults to `Joint`, which would
        // make this A/B compare two identical configurations and leave
        // `PqlThenJoint` with no other end-to-end optimum check.
        ws.outer_search = if two_stage {
            OuterSearch::PqlThenJoint
        } else {
            OuterSearch::Joint
        };
        let fit = crate::glmm::fit_glmm(
            &mut ws,
            xm.as_ref().subrows(0, n),
            y,
            &ids.primary,
            &ids.extra,
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
    // Serialized under alloc-tests so its allocations can't land in a
    // concurrent dhat profiler window on an `-- --ignored` run.
    #[cfg(feature = "alloc-tests")]
    let _serial = crate::test_support::alloc_test_guard();
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
    band: f64,
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
    assert!(f.converged(), "{name} k={nagq} must converge");

    let what = format!("{name} k={nagq}");
    assert_pinned(&f.beta, ref_beta, band, &format!("{what} beta"));
    assert_pinned(&f.se, ref_se, band, &format!("{what} se"));
    let (stddev, corr) = f.stddev_corr(0);
    assert_pinned(&stddev, ref_stddev, band, &format!("{what} stddev"));
    let lower: Vec<f64> = (0..stddev.len())
        .flat_map(|t| (0..t).map(move |u| (t, u)))
        .map(|(t, u)| corr[t][u])
        .collect();
    assert_pinned(&lower, corr_lower, band, &format!("{what} corr"));
}

/// Vector AGQ (q=2), binomial: `y ~ x + (1 + x | g)` on sim_binomial_slope1 at
/// nAGQ ∈ {7, 11}. Validated cross-engine by the
/// `sim_binomial_slope1_agq_k{7,11}` cells against GLMMadaptive.
///
/// Relative-tolerance, not bit-equal. These values reproduce BIT-EXACTLY on the
/// anchor machine (see `assert_pinned`'s "which machine the pins are frozen on");
/// BAND is margin for aarch64-apple-darwin, where the k=11 β drifts 1.51e-7
/// (`beta[0]`) from architecture-dependent SIMD/FMA contraction on this kernel's
/// long reductions. 5e-6 is ~33x that: loose enough to absorb cross-arch
/// reassociation, tight enough that a real change in the fit still trips it.
///
/// **k=7 `ref_se` re-anchored 2026-07-31.** Both of its elements had been frozen
/// off-anchor and missed by 3.6e-15 / 1.5e-13 — invisible under BAND, and old
/// enough that the responsible toolchain or faer bump is not identifiable. Not a
/// numerical finding at that size; re-pinned only so "bit-exact on the anchor"
/// holds without exceptions and stays checkable. β, stddev and corr were already
/// exact and are untouched, as is k=11.
#[test]
fn fit_glmm_binomial_slope1_vector_agq_is_pinned() {
    const BAND: f64 = 5e-6;
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
        // Re-pinned 2026-09-01 (was the 2026-07-31 re-anchor): the exact
        // hyper-dual joint (θ, β) Hessian replaced the FD stencil on the
        // blocked GLMM path (W3), which covers this vector-AGQ shape. The fit
        // is unchanged — β/stddev/corr keep their constants — so the movement
        // is the FD stencil's own truncation-plus-noise error, measured at
        // 1.32e-5 rel (k=7 se[1], worst) against the `se_hessian_rel` band of
        // 1e-3. Regenerate by running this test and reading the reported value;
        // the value is glmm's own, not a reference.
        &[0.1562674235527202, 0.13104931577314224],
        &[0.8758219936062426, 0.3726397459266911],
        &[0.5283132632086392],
        BAND,
    );
    check_vector_agq_pin(
        "sim_binomial_slope1",
        csv,
        11,
        1,
        family,
        &[0.42897959050352286, 0.43458759206119546],
        // Re-pinned 2026-09-01, same W3 exact-Hessian movement as the k=7 row
        // above (1.32e-5 rel worst) — see the provenance comment there.
        &[0.15627017032650084, 0.13105173822010058],
        &[0.8758453743269463, 0.3726611962891021],
        &[0.5283433982271978],
        BAND,
    );
}

/// Vector AGQ (q=2), Poisson: `y ~ x + (1 + x | g)` on sim_poisson_slope1 at
/// nAGQ ∈ {7, 11}. Validated cross-engine by the
/// `sim_poisson_slope1_agq_k{7,11}` cells against GLMMadaptive.
///
/// Relative-tolerance, not bit-equal: these values were frozen on a
/// different machine and reproduce here (aarch64-apple-darwin) only to rel
/// ~2.05e-6 (k=7 `beta[1]`) and, the binding one, ~7.07e-5 (k=7 `corr[0]`) —
/// not bit-exactly — architecture-dependent SIMD/FMA contraction on this
/// kernel's long reductions, not a regression. 1e-3 is ~14x the worst
/// observed drift: loose enough to absorb cross-arch reassociation, tight
/// enough that a real change in the fit still trips it.
///
/// **`ref_se` re-pinned 2026-07-30 (0.1.4 FD-θ-step fix), then re-anchored
/// 2026-07-31; β / stddev / corr are untouched throughout.** This fit's leading
/// Cholesky diagonal is 1.0118, the only θ coordinate here above 1, so dropping
/// the `max(1, |θ̂|)` scaling from the FD Hessian's θ step (`glmm::FD_STEP_BASE`,
/// and the step-construction comment in `glmm/se.rs`) shrank exactly one of its
/// steps, by 1.2%. Across that edit `se_hessian` moves 8.68e-7 (k=7) / 7.80e-7
/// (k=11) relative — the SAME figure on both machines, which is the cleanest
/// evidence available that the move is the fix and not the port — and **nothing
/// else in the fit moves by a bit**. β, the RE stddevs and the correlations are
/// bit-identical on both sides of the edit, which is why only `se` moved.
///
/// The 07-30 re-pin took its values from aarch64, splitting this test across two
/// reference machines; 07-31 replaced them with the anchor's, which differ from
/// the aarch64 ones by 5.2e-7 (k=7) / 7.4e-7 (k=11). Both are equally valid
/// arithmetic and both sit far inside BAND — the point of preferring the anchor
/// is corpus-wide, not local (`assert_pinned`, "re-freezing rule"). β/stddev/corr
/// keep their original anchor constants and remain what sizes BAND.
#[test]
fn fit_glmm_poisson_slope1_vector_agq_is_pinned() {
    const BAND: f64 = 1e-3;
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
        &[0.21833927439261516, 0.1854441204256203],
        &[1.0117923068022499, 0.33570973669422155],
        &[0.03689193891989874],
        BAND,
    );
    check_vector_agq_pin(
        "sim_poisson_slope1",
        csv,
        11,
        1,
        family,
        &[-1.353312567495963, 0.493094585284533],
        &[0.21807763915845102, 0.1853032173193218],
        &[1.0106390790816442, 0.33617266187700573],
        &[0.037212955698941674],
        BAND,
    );
}

/// Vector AGQ (q=3), binomial: `y ~ x1 + x2 + (1 + x1 + x2 | g)` on
/// sim_binomial_slope2 at nAGQ ∈ {7, 11} — the q_p ≤ 3 cap surface and the
/// kernel's dimensional generality. Validated cross-engine by the
/// `sim_binomial_slope2_agq_k{7,11}` cells against GLMMadaptive.
///
/// Relative-tolerance, not bit-equal. These values reproduce BIT-EXACTLY on the
/// anchor machine (see `assert_pinned`'s "which machine the pins are frozen on");
/// BAND is margin for aarch64-apple-darwin, where the k=7 β drifts 3.60e-6
/// (`beta[2]`) from architecture-dependent SIMD/FMA contraction on this kernel's
/// long reductions. 5e-5 is ~14x that: loose enough to absorb cross-arch
/// reassociation, tight enough that a real change in the fit still trips it.
///
/// **`ref_se` re-pinned 2026-07-30 (0.1.4 FD-θ-step fix), then re-anchored
/// 2026-07-31; β / stddev / corr are untouched throughout.** Same mechanism as
/// the `sim_poisson_slope1` sibling above and documented there: only this fit's
/// leading Cholesky diagonal (1.0640) is above 1, so exactly one FD θ step shrank,
/// by 6.4%, and `se_hessian` moves 1.20e-7 relative at both k — again the same
/// figure on both machines. Everything else in the fit is bit-identical across
/// the edit. As on the sibling, the 07-30 re-pin took aarch64 values and 07-31
/// replaced them with the anchor's (differing by 1.78e-6 at k=7, 9.7e-9 at k=11).
/// The θ-step move is ~15x smaller than that cross-arch spread, so BAND is
/// unchanged and still sized by `beta[2]`'s 3.60e-6 — both re-pins are
/// bookkeeping, not a tolerance question.
#[test]
fn fit_glmm_binomial_slope2_vector_agq_is_pinned() {
    const BAND: f64 = 5e-5;
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
            0.13111388966197157,
            0.10465309428546639,
            0.10556860947993794,
        ],
        &[1.0640265640371298, 0.6217295819798455, 0.6420938363939653],
        &[
            0.21906654236295983,
            0.13220156109374376,
            -0.030986539519051545,
        ],
        BAND,
    );
    check_vector_agq_pin(
        "sim_binomial_slope2",
        csv,
        11,
        2,
        family,
        &[0.3730629587832013, 0.5381087834502509, -0.3654883076141031],
        &[
            0.13112093692608306,
            0.10465668422853727,
            0.10557213027825517,
        ],
        &[1.0640974804776098, 0.6217673168423941, 0.6421365960857665],
        &[
            0.21910550941911974,
            0.13222240391317389,
            -0.03101919706448947,
        ],
        BAND,
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
    assert!(f.converged(), "no-signal GLMM must still converge");
    assert!(f.singular(), "must flag the θ≈0 boundary as singular");
    assert!(
        f.tau2[0] < 1e-4,
        "tau2[0] must pin near zero, got {}",
        f.tau2[0]
    );
}

/// R1 of the large-θ̂ coverage spec, AGQ arm: binomial GLMM
/// `y ~ 1 + x + z + (1 | g)` on `sim_binomial_bigsd` at nAGQ ∈ {7, 11}, gated
/// against the frozen `glmer(nAGQ = k, tolPwrss = 1e-13)` goldens
/// `validation/goldens/sim_binomial_bigsd_agq_k{7,11}.json`. lme4-only SE
/// (MixedModels computes no `se_hessian`). The oracle is sacred.
///
/// **Why this rung exists.** It is the only in-crate gate that reaches the FD
/// Hessian at a random-effect SD well above 1 — θ̂ = 4.85 (k=7) and 5.16 (k=11),
/// against a corpus that otherwise tops out at 1.13. The θ step is
/// `FD_STEP_BASE`, unscaled by θ̂, so these two fits difference the deviance
/// over ±0.01 and carry only 1.9e-7 and 1.8e-7 relative truncation error off
/// our own h→0 stencil limit even at this θ̂ range. Nothing else in the crate
/// could see a step that scaled with θ̂: every other GLMM rung has θ̂ at or
/// near 1.
///
/// **BAND = 2e-5, and it is reference-limited, not ours.** Measured post-fix
/// through this test's own lowering on aarch64-apple-darwin 2026-07-30, worst
/// coordinate of each quantity: `se_hessian` 7.18e-6 (k=7) / 8.35e-6 (k=11),
/// β 2.13e-6 / 5.51e-6, RE stddev 2.35e-6 / 2.15e-6. `tol.R`'s convention is
/// ceil-to-one-significant-figure of ~2× the measured worst, which the binding
/// `se_hessian` figure (2 × 8.35e-6 = 1.67e-5) puts at 2e-5; β and stddev clear
/// that with ≥3× to spare, so one band serves all three. The residual is **the
/// golden's own**: lme4's `vcov(use.hessian = TRUE)` is `lme4:::deriv12` at an
/// ABSOLUTE δ = 1e-4, which carries 8.34e-6 / 8.36e-6 relative error here — i.e.
/// post-fix the whole remaining disagreement is accounted for by the reference,
/// and tightening the band further would pin a number lme4 cannot itself
/// reproduce (two runs of its own stencil differ by 4.5e-7…1.8e-6). Do not read
/// the band as our accuracy.
///
/// **This band is not fail-before/pass-after evidence, and must not be read
/// as it.** What it measures is against our own h→0 limit, not against lme4.
/// The release's fail-before/pass-after rung is `sim_poisson_bigsd` in
/// `validation/tol.R`'s `TOL_PER_RUNG`, not this test.
///
/// Sizing note: k = 7 → k = 11 still moves
/// θ̂ by 6.3% on this dataset, so **k = 11 is not the AGQ limit here** and the two
/// goldens are not expected to agree closely with each other — `se_hessian`
/// differs between them by up to 6.2% on the intercept. That is the fit moving
/// with the quadrature order, not disagreement, which is why each k is pinned
/// against its own golden rather than against the other.
#[test]
fn fit_glmm_binomial_bigsd_agq_matches_lme4() {
    // ceil₁(2 × 8.35e-6 = 1.67e-5); see the doc comment for why it is the reference's floor.
    const BAND: f64 = 2e-5;
    // (nAGQ, β, se_hessian, RE stddev) per the frozen golden.
    let refs: [(u8, [f64; 3], [f64; 3], f64); 2] = [
        (
            7,
            [0.786420873395614, 0.903437833126084, -0.616408180886449],
            [0.333946011198284, 0.112745333379985, 0.195512460550797],
            4.85249696634019,
        ),
        (
            11,
            [0.824994474498218, 0.913561093491045, -0.622307953920597],
            [0.354539726379712, 0.113789698253031, 0.196816594693318],
            5.15796068325673,
        ),
    ];
    // Columns are y, x, z, g; `z` is numeric 0/1 and `g` (the grouping) is the
    // only factor — the same lowering `validation/manifest.json` declares.
    let csv = include_str!("../../validation/data/simulated/sim_binomial_bigsd.csv");
    let p = 3;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut g_raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        x.extend_from_slice(&[1.0, f[1].parse().unwrap(), f[2].parse().unwrap()]);
        g_raw.push(f[3].parse().unwrap());
    }
    let n = y.len();
    let (cluster_ids, n_clusters) = dense_ids(&g_raw);
    assert_eq!((n, n_clusters), (1800, 300), "R1 fixture shape");
    for (nagq, ref_beta, ref_se, ref_sd) in refs {
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
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
                nagq,
                ..FitOptions::default() // WaldSe::Hessian — the whole point of the rung
            },
        );
        assert!(f.converged(), "bigsd AGQ k={nagq} must converge");
        assert!(!f.singular(), "bigsd AGQ k={nagq} is an interior fit");
        let what = format!("sim_binomial_bigsd k={nagq}");
        assert_pinned(&f.beta, &ref_beta, BAND, &format!("{what} beta"));
        assert_pinned(&f.se, &ref_se, BAND, &format!("{what} se_hessian"));
        let (stddev, _) = f.stddev_corr(0);
        assert_pinned(&stddev, &[ref_sd], BAND, &format!("{what} stddev"));
    }
}

/// R3 of the large-θ̂ coverage spec: the
/// θ = 0 end of the same axis, on the **committed** `sim_binomial_zerosd`
/// fixture rather than a synthetic draw.
///
/// This gates a *documented behavioural divergence*, not a number, and it is why
/// R3 is deliberately NOT a curated `datasets` rung (`//large_theta_rungs` in
/// `validation/manifest.json`). What diverges from lme4 is only the reporting:
///
/// - **glmm** pins the component and reports `converged = true` with
///   `singular = true`.
/// - **lme4** emits `boundary (singular) fit`, which lands in
///   `m@optinfo$conv$lme4$messages` — so `engines/lme4.R:296`'s rule
///   (`converged = length(messages) == 0`) records `converged = FALSE` for the
///   very same fit. `isSingular()` is `TRUE` on both sides.
///
/// That difference-in-default had no test behind it.
/// `compare.R` cannot supply one: it compares β, SEs, stddevs, loglik and
/// coefficient names and reads no convergence flag at all — and both engines land
/// on a **bit-exact 0.0** stddev, so every numeric gate it does run reports
/// perfect agreement. Two further reasons the rung track is closed to R3, both
/// measured 2026-07-30 rather than assumed: lme4's
/// `vcov(m, use.hessian = TRUE)` — exactly what `engines/lme4.R:269` and `:146`
/// call — **hard-errors** on this fit (`'use.hessian'=TRUE specified, but Hessian
/// is unavailable`; `m@optinfo$derivs` is `NULL` on a boundary fit), so R3 as a
/// rung would abort the whole oracle run; and at θ̂ = 0 the θ↔β coupling block
/// vanishes, so `se_hessian` and `se_rx` collapse onto each other (9.4e-6 apart
/// here, against 9.7e-2 on R1) — R3 gates nothing about the coupling term.
///
/// Hence: in-crate, no oracle JSON, asserting the flags and the exact zero.
/// The exact zero is the assert that has to be `==`, not a band: `rel_max` floors
/// its denominator at 1e-12 (`validation/tol.R`), so a *tiny nonzero* θ̂ against
/// lme4's exact 0.0 would read as a relative difference of exactly 1.0. A
/// seed sweep found Bernoulli-shaped cells that returned 1.5e-8 / 3.9e-8 **while
/// still flagging singular** — which is why this fixture is the aggregated
/// (`incidence`/`size`) shape and why "pinned" is checked as bit-equality.
///
/// Sibling of `fit_glmm_binomial_no_cluster_signal_is_singular` above, which makes
/// the same claim on synthetic data with a `< 1e-4` band; this one is the committed
/// fixture and the exact pin.
#[test]
fn glmm_zerosd_boundary_reports_converged_and_singular() {
    // Aggregated binomial, lowered the way `validation/engines/common.rs`'s
    // `lower_dataset_generic` does for a manifest `weights` rung: the response is
    // `prop = incidence/size` and the trial counts enter as prior weights, one row
    // per aggregate observation. X = [1, x, z] — `z` is numeric 0/1 in the CSV and
    // the only declared factor is the grouping `g`.
    let csv = include_str!("../../validation/data/simulated/sim_binomial_zerosd.csv");
    let p = 3;
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut w = Vec::<f64>::new();
    let mut cl = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        let incidence: f64 = f[0].parse().unwrap();
        let size: f64 = f[1].parse().unwrap();
        x.extend_from_slice(&[1.0, f[2].parse().unwrap(), f[3].parse().unwrap()]);
        y.push(incidence / size);
        w.push(size);
        // Groups are labelled 1..=20 in the CSV; ids are 0-based and dense.
        cl.push(f[4].parse::<u32>().unwrap() - 1);
    }
    let n = y.len();
    let n_clusters = cl.iter().max().unwrap() + 1;
    assert_eq!((n, n_clusters), (160, 20), "committed R3 fixture shape");

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
        &GroupIds {
            primary: cl,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1, 2],
            weights: Some(w),
            ..FitOptions::default()
        },
    );

    // The two flags the divergence is about, asserted together: reporting a pinned
    // boundary as a *converged* fit is the whole claim, so `converged` alone or
    // `singular` alone would each be satisfiable by the wrong behaviour.
    assert!(
        f.converged(),
        "glmm must report the pinned θ=0 boundary as CONVERGED (lme4 records \
         converged=FALSE on the same fit — documented divergence)"
    );
    assert!(
        f.singular(),
        "and must flag it singular, as lme4's isSingular does"
    );

    // Exact, not near: see the rel_max 1e-12-floor note above.
    let (stddev, _) = f.stddev_corr(0);
    assert_eq!(stddev.len(), 1, "one scalar grouping");
    assert_eq!(
        stddev[0].to_bits(),
        0.0f64.to_bits(),
        "RE stddev must be bit-exact 0.0, got {} (bits 0x{:016x})",
        stddev[0],
        stddev[0].to_bits()
    );
    assert_eq!(
        f.tau2[0].to_bits(),
        0.0f64.to_bits(),
        "tau2[0] must be bit-exact 0.0, got {}",
        f.tau2[0]
    );
    // A pinned boundary is still a reportable fit: the estimates must be finite,
    // not the NaN-fill a numerical failure would leave behind.
    assert!(
        f.beta.iter().chain(&f.se).all(|v| v.is_finite()) && f.loglik.is_finite(),
        "β/SE/loglik must be finite at the boundary: β {:?} se {:?} loglik {}",
        f.beta,
        f.se,
        f.loglik
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
    assert!(cold.converged(), "reference fit must converge");
    // The guard is on the extra grouping's SLOPE VARIANCE, read off `varcorr` —
    // `D[1][1]`, the last entry of that block's q=2 vech. The θ coordinates are
    // the wrong place to read it: this draw's extra-grouping intercept variance
    // is zero by construction (the generating η carries `x1·(g2−3)` and no `g2`
    // main effect), and once `Λ[0][0] = 0` the Cholesky no longer identifies how
    // the slope variance splits between `Λ[1][0]` and `Λ[1][1]` — only their sum
    // of squares, which is exactly `D[1][1]`.
    assert_eq!(cold.varcorr.len(), 2, "primary + one extra grouping");
    let d_slope = *cold.varcorr[1].last().unwrap();
    assert!(
        d_slope > 0.1,
        "the draw must actually carry an extra-grouping slope variance: varcorr {:?}",
        cold.varcorr
    );

    let (sized, ids, perm) = super::common::spec_sized_from_ids(&model, &ids);
    let mut ws = super::core::build_workspace(&sized, perm, n, 2, &opts);
    let view = super::core::fit_on(&mut ws, &x, &y, &ids, None, &opts);
    for j in 0..2 {
        assert_eq!(
            view.betas()[j],
            cold.beta[j],
            "loop tier must reach fit_cold's β exactly: β[{j}]"
        );
    }
}

// ---------------------------------------------------------------------------
// Internal random-effect column scaling (`LmmGroupings::set_slope_scales`) —
// dense GLMM rescale test
//
// GOVERNING IDEA, shared with `lmm_tests.rs`'s and `sparse/tests.rs`'s rescale
// tests: multiply a random-slope design column by an exact power of two `C`
// and refit. A dropped back-map shows up unmistakably as a ratio of 1 instead
// of the predicted `1/C` or `1/C²` (see `lmm_tests.rs`'s rescale test for the
// full `Z~ = Z·diag(1/s)`, `Λ~ = diag(s)·Λ` derivation, which applies
// unchanged here).
//
// `C` is smaller here (4.0, not the LMM tests' 1024.0) and the band looser,
// for a reason specific to the GLMM route: the LMM solver optimizes θ ALONE
// (β is recovered in closed form at each θ), so scaling a column that is also
// a fixed effect leaves the θ-search's internal problem bit-identical between
// the two fits. The GLMM solver optimizes the JOINT vector `[θ | β]` with one
// shared BOBYQA trust radius and a `BETA_BOX` of ±30 on the raw (unrescaled)
// β coordinates. Column-scaling shifts β's position inside that fixed box
// differently in the two fits (β̂ᵪ ≈ β̂/C sits closer to 0 than β̂ does), so the
// two fits' internal trust-region paths genuinely differ — they are two
// separate optimizations of equivalent objectives, not one bit-identical
// search read twice. A small `C` keeps both fits' β inside the same box
// region; a loose band absorbs the resulting path difference.
// ---------------------------------------------------------------------------

/// Parses `validation/data/simulated/sim_binomial_slope1.csv` into the q=2
/// random-slope design `y ~ 1 + x + (1 + x | g)` — the same fixture
/// `check_vector_agq_pin` uses for the AGQ pins above, reused here because it
/// is already known to converge under a random primary slope. Column 1 (`x`)
/// is both the fixed-effect covariate and the primary random-slope covariate.
fn sim_binomial_slope1_design() -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, GroupIds) {
    let csv = include_str!("../../validation/data/simulated/sim_binomial_slope1.csv");
    let mut y = Vec::<f64>::new();
    let mut xcol = Vec::<f64>::new();
    let mut g_raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        xcol.push(f[1].parse().unwrap());
        g_raw.push(f[2].parse().unwrap());
    }
    let n = y.len();
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = xcol[i];
    }
    let (primary, n_clusters) = dense_ids(&g_raw);
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![1],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary,
        extra: vec![],
    };
    (x, y, n, p, model, ids)
}

/// Dense GLMM rescale identity, `C = 4.0`, `WaldSe::Hessian`. Same predicted
/// moves as the LMM rescale test for `beta`/`se`/`varcorr`/`tau2` (see that
/// test's doc comment for the derivation), EXCEPT:
///
/// - **No REML Jacobian.** The GLMM criterion is the marginal Laplace
///   deviance, which carries no `log|X'V⁻¹X|` term (that term is a Gaussian-
///   REML-only artifact of profiling β out of a linear-Gaussian likelihood).
///   So unlike the LMM test, `deviance` here must be UNCHANGED — a genuine
///   reparameterization, not a rescale, of the same marginal likelihood.
/// - **`stddev_se` moves by the Lambda-row scales, not their squares.**
///   `stddev_se` is the SE of θ itself (the θ-Hessian block of the joint
///   covariance), not of `θ²`, so it carries exactly ONE power of the row
///   scale — `[se0, se1/C, se2/C]` — unlike `tau2`, which is `θ²·σ̂²` and so
///   carries the row scale SQUARED.
///
/// `BAND` is margin over the worst relative spread measured between the two
/// independent joint-BOBYQA fits on 2026-08-23 (the anchor machine — see
/// `assert_pinned`'s doc comment). Looser than the LMM test's band for the
/// reason in this section's header comment: these are two genuinely different
/// internal optimizations, not the same search read twice.
#[test]
fn glmm_rescaling_slope_column_moves_stddev_se_by_the_predicted_power_of_c() {
    const C: f64 = 4.0;
    const BAND: f64 = 3e-4;
    const DEV_ABS: f64 = 1e-9;

    let (x, y, n, p, model, ids) = sim_binomial_slope1_design();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        wald_se: WaldSe::Hessian,
        ..FitOptions::default()
    };

    let base = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(base.converged(), "base binomial slope GLMM must converge");
    assert!(
        base.stddev_se.iter().all(|v| v.is_finite()),
        "base stddev_se must be finite on a converged Hessian fit: {:?}",
        base.stddev_se
    );

    let mut x_c = x.clone();
    for i in 0..n {
        x_c[i * p + 1] *= C;
    }
    let scaled = fit_cold(&x_c, &y, n, p, &model, &ids, &opts);
    assert!(scaled.converged(), "column-scaled fit must converge");
    assert!(
        scaled.stddev_se.iter().all(|v| v.is_finite()),
        "scaled stddev_se must be finite on a converged Hessian fit: {:?}",
        scaled.stddev_se
    );

    assert_pinned(&[scaled.beta[0]], &[base.beta[0]], BAND, "beta[0]");
    assert_pinned(&[scaled.beta[1]], &[base.beta[1] / C], BAND, "beta[1]");
    assert_pinned(&[scaled.se[0]], &[base.se[0]], BAND, "se[0]");
    assert_pinned(&[scaled.se[1]], &[base.se[1] / C], BAND, "se[1]");

    // varcorr vech [D00, D10, D11].
    assert_eq!(scaled.varcorr.len(), 1, "one grouping block");
    assert_pinned(
        &scaled.varcorr[0],
        &[
            base.varcorr[0][0],
            base.varcorr[0][1] / C,
            base.varcorr[0][2] / (C * C),
        ],
        BAND,
        "varcorr vech",
    );

    // tau2[0] = Lambda row 0 (intercept); tau2[1], tau2[2] = Lambda row 1 (slope).
    assert_pinned(
        &scaled.tau2,
        &[base.tau2[0], base.tau2[1] / (C * C), base.tau2[2] / (C * C)],
        BAND,
        "tau2",
    );

    // ranef, per level [b0, b1] — the assertion that caught `assemble_ranef_dense`
    // reporting its slope modes on the internal scale (ratio 1 instead of 1/C).
    assert_eq!(scaled.ranef.len(), base.ranef.len());
    assert_eq!(scaled.ranef_levels, base.ranef_levels);
    let n_levels = scaled.ranef_levels[0];
    let mut want_ranef = Vec::with_capacity(scaled.ranef.len());
    for l in 0..n_levels {
        want_ranef.push(base.ranef[l * 2]);
        want_ranef.push(base.ranef[l * 2 + 1] / C);
    }
    assert_pinned(&scaled.ranef, &want_ranef, BAND, "ranef");

    // stddev_se — the item this test exists for: ONE power of the row scale
    // (θ-scale SE), not squared like tau2. Length 3: [row0, row1, row1].
    assert_eq!(scaled.stddev_se.len(), 3, "one grouping, q=2 vech");
    assert_pinned(
        &scaled.stddev_se,
        &[
            base.stddev_se[0],
            base.stddev_se[1] / C,
            base.stddev_se[2] / C,
        ],
        BAND,
        "stddev_se",
    );

    // deviance — no REML Jacobian on the GLMM route, so this is a genuine
    // reparameterization: the marginal criterion is invariant.
    assert!(
        (scaled.deviance - base.deviance).abs() < DEV_ABS,
        "deviance moved under a column reparameterization: {} vs {}",
        scaled.deviance,
        base.deviance
    );
}

/// The stage split must add up to the reported eval count, and the shrink
/// count must be a real subset of stage 2's evals — counters 1 and 2 in
/// `crate::counters`' module header, on the dense GLMM route. cbpp (n_theta=1,
/// p=4) is a logit-link blocked shape, so its default `outer_search` is
/// `OuterSearch::ExactProfile` (`exact_profile_shape`), which has no stage-2
/// solve to split evals against. The stage split this test measures only
/// exists on `PqlThenJoint`, so this drives the kernel entry `crate::glmm::fit_glmm`
/// directly with `ws.outer_search` forced to `PqlThenJoint`,
/// mirroring `assert_two_stage_matches_single_local` (glmm_tests.rs:2676).
#[cfg(feature = "counters")]
#[test]
fn dense_glmm_counters_split_stages_and_count_shrink_evals() {
    use crate::counters::Stage;
    let (x, y, cluster_ids, n) = cbpp_design();
    let model = cbpp_model();
    let p = 4;
    let ids = crate::GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let (sized, ids, _perm) = spec_sized_from_ids_pub(&model, &ids);
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
    let targets: Vec<u32> = (0..p as u32).collect();

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
    ws.outer_search = OuterSearch::PqlThenJoint;
    let f = crate::glmm::fit_glmm(
        &mut ws,
        xm.as_ref().subrows(0, n),
        &y,
        &ids.primary,
        &ids.extra,
        &targets,
        None,
        &beta_start,
        n,
        WaldSe::Rx,
    );
    assert!(f.converged, "cbpp (two-stage forced) must converge");
    let c = f.counters;
    assert!(
        c.stage_evals[0] > 0,
        "two-stage path must record stage-1 evals"
    );
    assert!(c.stage_evals[1] > 0, "stage 2 always runs");
    assert_eq!(
        (c.stage_evals[0] + c.stage_evals[1]) as usize,
        f.n_eval,
        "stage split must reconstruct n_eval"
    );
    assert!(
        c.evals_after_last_improve(Stage::Two) < c.stage_evals[1],
        "shrink evals are a strict subset of stage-2 evals"
    );
}

/// One histogram entry per fit-path outer evaluation, and none from the
/// FD-Hessian SE pass — the same fit-path-vs-SE-eval discriminator
/// `Note::PirlsExhausted` already uses. PIRLS never converges in zero
/// iterations, so bucket 0 must stay empty.
#[cfg(feature = "counters")]
#[test]
fn dense_glmm_counters_histogram_one_entry_per_outer_eval() {
    let (x, y, ids, n) = cbpp_design();
    let model = cbpp_model();
    let ids = crate::GroupIds {
        primary: ids,
        extra: vec![],
    };
    let opts = crate::FitOptions {
        target_indices: vec![1],
        ..crate::FitOptions::default() // WaldSe::Hessian — the FD pass runs
    };
    let f = crate::fit_cold(&x, &y, n, 4, &model, &ids, &opts);
    assert!(f.converged(), "cbpp must converge");
    let c = f.counters;
    assert_eq!(
        c.pirls_hist.iter().sum::<u32>() as usize,
        f.n_eval,
        "one PIRLS histogram entry per fit-path eval, SE evals excluded"
    );
    assert_eq!(
        c.pirls_hist[0], 0,
        "no eval solves PIRLS in zero iterations"
    );
}

/// An AGQ fit must report one AGQ evaluation per outer eval and the node cost
/// they carry: clusters x nagq^q per evaluation. A Laplace fit records none.
#[cfg(feature = "counters")]
#[test]
fn agq_counters_report_evals_times_nodes() {
    let (x, y, ids, n) = cbpp_design();
    let model = cbpp_model();
    let n_clusters = (ids.iter().copied().max().unwrap() as u64) + 1;
    let ids = crate::GroupIds {
        primary: ids,
        extra: vec![],
    };

    let laplace = crate::fit_cold(
        &x,
        &y,
        n,
        4,
        &model,
        &ids,
        &crate::FitOptions {
            target_indices: vec![1],
            ..crate::FitOptions::default()
        },
    );
    assert_eq!(
        laplace.counters.agq_evals, 0,
        "nagq == 1 records no AGQ eval"
    );
    assert_eq!(laplace.counters.agq_node_evals, 0);

    let agq = crate::fit_cold(
        &x,
        &y,
        n,
        4,
        &model,
        &ids,
        &crate::FitOptions {
            target_indices: vec![1],
            nagq: 7,
            ..crate::FitOptions::default()
        },
    );
    assert!(agq.converged(), "cbpp nAGQ=7 must converge");
    assert_eq!(
        agq.counters.agq_evals as usize, agq.n_eval,
        "every AGQ outer eval evaluates the quadrature"
    );
    assert_eq!(
        agq.counters.agq_node_evals,
        agq.counters.agq_evals as u64 * n_clusters * 7,
        "node cost is evals x clusters x nagq^1"
    );
}
