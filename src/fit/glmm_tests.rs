//! GLMM estimator tests (Binomial/Poisson/Gamma/negative-binomial,
//! `re: Some`, dense + sparse-Schur equivalence + AGQ + two-stage).

use super::*;
use crate::glmm::{glmm_laplace_deviance, GlmmWorkspace, OuterSearch, StructuredSchur};
use crate::test_support::assert_near;
use crate::{
    BinomialLink, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, PoissonLink,
    ReStructure, Sizing, StartValues, WaldSe,
};
use faer::Mat;

use super::common_tests::{
    assert_pinned, dense_ids, dense_str, inf_plateau_exp1, inf_plateau_lcg_next,
    inf_plateau_normal, inf_plateau_poisson, lcg, sim_clustered,
};

/// `run_glmm_on` + `glmm_view_to_fit` must reproduce the `Fit`
/// that the full `fit_cold` dispatch produces for a clustered binomial GLMM —
/// pins the view/assembly split as behavior-preserving. The mu_hat/deviance
/// tuple the route-comparison tests read must also stay populated.
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
/// test, `fit_grouped_honors_opts_wald_se`, and (`pub(crate)`)
/// `core_tests::counters_show_the_outer_search_stage_split`.
pub(crate) fn cbpp_design() -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
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
/// `FitOptions` now, not here. `pub(crate)`: also used by
/// `core_tests::counters_show_the_outer_search_stage_split`.
pub(crate) fn cbpp_model() -> ModelSpec {
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
    // A looser SE band has no reason to exist: the reference constants above
    // are the citation-corrected ones, not the default-tolPwrss ones. The
    // oracle is sacred — these bound glmm to lme4, never the reverse.
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

/// The presence half of the `PirlsExhausted` contract `fit_glmm_cbpp_matches_lme4`
/// above asserts the absence of: a real solve whose opening trial points run the
/// full `PIRLS_MAX_ITERS` cap, so the note's payload is produced by the fit path
/// rather than by a hand-built `FitDiagnostics` carrier.
///
/// The design itself is benign — a 4-cluster Gamma-log GLMM that cold-fits with
/// no note at all — and the pathology is the warm start alone. θ₀ = 1e5 puts
/// BOBYQA's opening trial points at a random-effect scale where the penalized
/// deviance has no mode reachable inside the cap; each such eval scores
/// `+INFINITY` and is rejected, and the search walks back to the cold optimum.
/// So `final_eval` stays false and the reported estimates are the cold ones
/// (measured: worst β gap 7.9e-6 relative, deviance 2.4e-12) — this is exactly
/// the "a note with `final_eval == false` costs nothing observable" case.
///
/// The second arm re-fits the same warm start under the RX/Schur SE denominator
/// instead of the FD joint Hessian. Both arms share one fit path and differ only
/// in the post-fit SE pass, so an `evals` that moved between them would mean the
/// SE pass's own tight-tolerance PIRLS solves had leaked into the fit-path
/// counter that the note is defined over.
#[test]
fn pirls_exhausted_note_counts_fit_path_evals_only() {
    let (n, n_clusters, p) = (24usize, 4usize, 2usize);
    // Cluster offsets live on the log-mean scale and span ±3, which is what
    // makes the far-out θ trial points unsolvable rather than merely slow;
    // y is Gamma(shape 1) about exp(η), so the cold fit is ordinary.
    let mut st = 7u64;
    let u: Vec<f64> = (0..n_clusters).map(|_| 3.0 * lcg(&mut st)).collect();
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        y[i] = (0.5 + 0.8 * x1 + u[i % n_clusters]).exp() * inf_plateau_exp1(&mut st);
    }
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
        primary: (0..n).map(|i| (i % n_clusters) as u32).collect(),
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };

    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(
        cold.converged(),
        "the cold fit of this design must converge"
    );
    assert!(
        cold.diagnostics.notes.is_empty(),
        "cold-started, no trial point reaches the cap, got {:?}",
        cold.diagnostics.notes
    );

    let start = StartValues {
        beta: vec![],
        theta: vec![1e5],
    };
    let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(&start), &opts);
    assert!(
        warm.converged(),
        "the search must still reach the optimum from the absurd start"
    );
    let [note] = &warm.diagnostics.notes[..] else {
        panic!(
            "expected exactly one note, got {:?}",
            warm.diagnostics.notes
        );
    };
    let Note::PirlsExhausted { evals, final_eval } = note else {
        panic!("expected PirlsExhausted, got {note:?}");
    };
    // Not pinned to the observed 17: how many of BOBYQA's trial points land in
    // the unsolvable region moves with the optimizer. What is pinned is that
    // fit-path evals are counted at all, and that the solve behind the reported
    // estimates is not one of them.
    assert!(*evals > 0, "the exhausted trial evals must be counted");
    assert!(
        !*final_eval,
        "the re-evaluation at γ̂ converges, so the reported estimates are untruncated"
    );
    for j in 0..p {
        let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
        assert!(
            rel < 1e-4,
            "β[{j}] warm {} vs cold {} (rel {rel})",
            warm.beta[j],
            cold.beta[j]
        );
    }
    let dev_rel = (warm.deviance - cold.deviance).abs() / cold.deviance.abs();
    assert!(
        dev_rel < 1e-9,
        "deviance warm {} vs cold {} (rel {dev_rel})",
        warm.deviance,
        cold.deviance
    );

    let rx = fit_warm(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        Some(&start),
        &FitOptions {
            wald_se: WaldSe::Rx,
            ..opts.clone()
        },
    );
    let Some(Note::PirlsExhausted {
        evals: rx_evals, ..
    }) = rx.diagnostics.notes.first()
    else {
        panic!(
            "expected PirlsExhausted on the Rx arm, got {:?}",
            rx.diagnostics.notes
        );
    };
    assert_eq!(
        rx_evals, evals,
        "the FD-Hessian SE pass must not add to the fit-path eval count"
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
/// f <- glmer(y ~ x1 + (1 | g), family = poisson, weights = w,
///            control = glmerControl(tolPwrss = 1e-13))
/// print(summary(f)$coefficients, digits = 15)
/// print(as.data.frame(VarCorr(f)), digits = 15)
/// ```
/// β (2e-3 abs) and the RE SD (3e-3 rel) mirror `fit_glmm_cbpp_matches_lme4`.
/// SE band 1e-3, like the other goldens in this file: `tolPwrss = 1e-13` holds
/// `glmer` to the mode, so its log|A| comes from working weights at the mode
/// (see src/glmm/se.rs).
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
    const REF_BETA: [f64; 2] = [0.235917526156541, 0.547943727436689];
    const REF_SE: [f64; 2] = [0.1758115943204218, 0.0597574539823975];
    const REF_G_SD: f64 = 0.575373557412504;

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
        // `validation/tol.R`'s se_hessian_rel, the same band the artifact-free
        // goldens in this file hold.
        assert!(
            se_rel < 1e-3,
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
/// f <- glmer(y ~ x1 + (1 | g), family = Gamma("log"), weights = w,
///            control = glmerControl(tolPwrss = 1e-13))
/// print(summary(f)$coefficients, digits = 15)
/// print(as.data.frame(VarCorr(f)), digits = 15); print(sigma(f)^2, digits = 15)
/// ```
/// τ² is compared on lme4's VarCorr vcov scale (σ̂²·θ̂²). SE tolerance is the
/// cbpp 1e-3; β mirrors `fit_glmm_gamma_sim_matches_lme4`'s relative gate.
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
    const REF_BETA: [f64; 2] = [0.863096050047979, 0.372185978193556];
    const REF_SE: [f64; 2] = [0.0654910409193553, 0.0333306334749443];
    const REF_G_VCOV: f64 = 0.0510270265232477; // σ̂²·θ̂² (lme4 VarCorr vcov)

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
        // `validation/tol.R`'s se_hessian_rel; measured worst here 8.8e-6.
        assert!(
            se_rel < 1e-3,
            "se[{j}] = {} vs lme4 {} (rel {se_rel})",
            f.se[j],
            REF_SE[j]
        );
    }
    // varcorr[0][0] = σ̂²·θ̂² — directly lme4's VarCorr vcov for the g
    // intercept, via the public block (σ̂²-scaled like tau2).
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
        // `validation/tol.R`'s se_hessian_rel; measured worst here 1.5e-4.
        assert!(se_rel < 1e-3, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
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
    // Column-major x + StructuredSchur, as fit_glmm does.
    let mut xm = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            xm[(i, j)] = x[i * p + j];
        }
    }
    ws.pattern.structured_schur = StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n);
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

    ws.pattern.force_dense_schur = true;
    let dev_dense = glmm_laplace_deviance(
        &params,
        &mut ws,
        xm.as_ref().subrows(0, n),
        &y,
        &ids.primary,
        &ids.extra,
        n,
    );
    ws.pattern.force_dense_schur = false;
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
        ws.pattern.structured_schur = if ws.groupings.structured_extras_eligible() {
            StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
        } else {
            None
        };
        ws.pattern.force_dense_schur = force_dense;
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
        (ws.inference.var_diag[..p].to_vec(), fit.converged)
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
                let cov = lcg(&mut st);
                let eta = 0.2 + 0.6 * cov + pe + ee;
                let prob = 1.0 / (1.0 + (-eta).exp());
                let draw = (lcg(&mut st) + 1.0) / 2.0;
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
        ws.pattern.structured_schur = if ws.groupings.structured_extras_eligible() {
            StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
        } else {
            None
        };
        ws.pattern.force_dense_schur = force_dense;
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
        (
            ws.betas.clone(),
            ws.inference.var_diag[..p].to_vec(),
            fit.converged,
        )
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
/// 2026-07-30 across the FD-θ-step fix, is the *step rule*'s behaviour: the
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
    // SE band is `validation/tol.R`'s se_hessian_rel; measured worst 1.1e-4,
    // the ~1e-4 agreement the header attributes to `PIRLS_TOL_REL_NONCANON`.
    for ((&b, &rb), (&s, &rs)) in f.beta.iter().zip(&REF_BETA).zip(f.se.iter().zip(&REF_SE)) {
        assert!((b - rb).abs() / rb.abs() < 2e-3, "β = {b} vs lme4 {rb}");
        assert!((s - rs).abs() / rs < 1e-3, "se = {s} vs lme4 {rs}");
    }
}

/// Cloglog binomial GLMM `y ~ 1 + x1 + x2 + x3 + z + (1 | g)` on the 9,600-row
/// `sim_probit_large` fixture, gated against frozen
/// `glmer(binomial("cloglog"), tolPwrss = 1e-13)`
/// (`validation/goldens/sim_cloglog_glmm.json`). lme4-only SE. This arm needs no
/// kernel change: `build_workspace`'s `(family, Some(re))` branch
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
    // SE band is `validation/tol.R`'s se_hessian_rel; measured worst 6.9e-6.
    for ((&b, &rb), (&s, &rs)) in f.beta.iter().zip(&REF_BETA).zip(f.se.iter().zip(&REF_SE)) {
        assert!((b - rb).abs() / rb.abs() < 2e-3, "β = {b} vs lme4 {rb}");
        assert!((s - rs).abs() / rs < 1e-3, "se = {s} vs lme4 {rs}");
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
/// Regression guard for the FD-Hessian seeding bug: `joint_hessian_cov` reuses
/// the random-effect mode û(γ̂) the fit already converged to, rather than
/// re-deriving it by a COLD PIRLS solve. Where the mode problem has more
/// than one basin a cold solve can land in a different one, and the inverse link is
/// where that shows: the fit reaches deviance 936.7683 while a cold re-eval at the
/// same γ̂ returns 1034.5678. Finite differences that straddle the two
/// branches leave the joint Hessian indefinite, the RX fallback's Schur
/// indefinite at the same wrong mode, and the whole fit reported failed for
/// want of a standard error — `converged` false and every estimate NaN.
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
        // `validation/tol.R`'s se_hessian_rel; measured worst here 1.9e-4.
        assert!(se_rel < 1e-3, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
    }
    // Via stddev_corr/varcorr — σ̂²-scaled like tau2, so it gates
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
        // Method-matched arm, so `validation/tol.R`'s se_rel, not se_hessian_rel;
        // measured worst here 1.7e-4.
        assert!(
            se_rel < 1e-3,
            "rx se[{j}] = {} vs lme4 {}",
            f_rx.se[j],
            REF_SE_RX[j]
        );
    }
}

/// The `dispersion: Some(v)` directive on the GLMM route: φ̂ is reported as the
/// held value instead of the Pearson moment estimate, and φ stops costing a
/// degree of freedom, so `df` drops to 3 β + 1 θ. Same `sim_gamma` design as
/// `fit_glmm_gamma_sim_matches_lme4`, whose estimate-φ `df` is 5.
#[test]
fn fit_glmm_gamma_fixed_dispersion_is_reported_and_costs_no_df() {
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
    const PHI: f64 = 0.75;
    let held = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1, 2],
            dispersion: Some(PHI),
            ..FitOptions::default()
        },
    );
    let free = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );
    assert!(
        held.converged() && free.converged(),
        "gamma GLMM must converge"
    );
    assert_eq!(held.dispersion, PHI, "held φ must be reported verbatim");
    assert!(
        (free.dispersion - PHI).abs() > 1e-3,
        "the held value must differ from the Pearson estimate {} or this proves nothing",
        free.dispersion
    );
    assert_eq!(free.df, 5); // 3 β + cluster θ + φ
    assert_eq!(held.df, free.df - 1, "a held φ costs no degree of freedom");
}

/// `WaldSe::Hessian` and `WaldSe::Rx` must report the same fitted point on
/// `sim_gamma`: `joint_hessian_cov`'s tail re-eval keeps `ws.pirls.prob` and `ws.pirls.u`
/// at the same pinned re-eval mode, so Gamma's σ̂² (`family::glmm_sigma_sq`)
/// is built from a matched pair — leaving `ws.pirls.prob` at its own re-solve while
/// `ws.pirls.u` is put back to the pinned mode would mismatch them.
/// `tau2`/`varcorr`/`dispersion` derive from that σ̂², and `fitted` IS `ws.pirls.prob`.
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
/// **Additive Rust-vs-Rust pin (2026-08-06; scope updated 2026-09-06).** Gates
/// the NB coordinate's read-back (`fit_glmm_nb` → `run_glmm_on` →
/// `glmm_view_to_fit`) and the marginal objective directly — the lme4 bands
/// above (5e-3 β, 5e-2 se/θ̂) are too loose to tell a regression from rounding.
/// Before 2026-09-06 this pin instead gated `golden_max_ln_theta`
/// (`src/fit/glm.rs`)'s golden-section bracket: a knife-edge in the inner
/// fixed-θ fit that a 1e-8-wide stopping width could land on either side of,
/// fixed by loosening the width to 1e-4. `BAND = 1e-7` is 10-50× the measured
/// worst-case drift on this fixture at that width: 6.17e-9 relative under a
/// 128-draw 1-ULP sweep on `sim_nb`'s inputs (K=64 on `x`, K=64 on `y`),
/// 5.16e-11 under the committed `pulp` lane-width probe (scalar-forced vs
/// normal dispatch) — both far inside the band, both probes agree on the
/// order of magnitude.
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

    const BAND: f64 = 1e-7;
    // Additive bit-exact pin (see doc comment above). What produces these
    // values: θ_NB is a coordinate of the outer BOBYQA on the marginal
    // objective — the incumbent is the answer, there is no final fit at θ̂ —
    // started cold from the no-RE GLM-NB's own θ̂ (a method-of-moments seed
    // charges the RE variance to the dispersion, which breaks random-slope NB
    // shapes), and PIRLS returns the data term, `log|A|` and the factor at the
    // returned iterate. The point meets this test's own lme4 bands (β 5e-3,
    // se 5e-2, sd 2e-2, θ̂ 5e-2) with 2.8 to four orders to spare: β within
    // 8.3e-6 absolute, se within 1.3e-5 relative, cluster SD 1.5e-5, θ̂ 4.5e-6.
    const REF_BETA_PIN: [f64; 3] = [
        -0.020769959407223825,
        0.5939568607057298,
        0.5994414310028326,
    ];
    const REF_SE_PIN: [f64; 3] = [0.1631665854267635, 0.0721281096331149, 0.14148098492027553];
    const REF_TAU2_PIN: [f64; 1] = [0.32971870514707935];
    const REF_THETA_PIN: f64 = 1.7836281070311075;
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

/// On an NB GLMM, `Fit::deviance` is the outer θ search's own objective —
/// `dev(θ̂) − 2·saturated_loglik(θ̂)`, equal to `−2·logLik` exactly (see
/// `fit_glmm`'s NB deviance and `fit::common::glmm_loglik`). Same `sim_nb`
/// fixture and reference log-likelihood as `fit_glmm_nb_sim_matches_lme4`;
/// this test only adds the deviance identity.
#[test]
fn fit_glmm_nb_deviance_is_search_objective() {
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
    assert!(
        (f.deviance + 2.0 * f.loglik).abs() < 1e-9 * f.deviance.abs(),
        "deviance {} vs -2*loglik {}",
        f.deviance,
        -2.0 * f.loglik
    );
    // Same reference and band as `fit_glmm_nb_sim_matches_lme4`'s loglik pin —
    // shows loglik did not move.
    const REF_LOGLIK: f64 = -481.455529976646;
    assert!(
        (f.loglik - REF_LOGLIK).abs() < 1e-2,
        "loglik {} vs lme4 {REF_LOGLIK}",
        f.loglik
    );
}

/// The same `deviance = −2·logLik` identity under prior weights. The NB
/// saturated term enters the reported deviance through
/// `saturated_loglik(θ̂, y, weights)`, whose weighted branch no other NB GLMM
/// test reaches. The identity is the whole test: no reference is needed, only
/// that the deviance's correction and `loglik`'s are the same weighted sum.
#[test]
fn fit_glmm_nb_weighted_deviance_is_search_objective() {
    let (x, y, cluster_ids, n_clusters) =
        sim_clustered(include_str!("../../validation/data/simulated/sim_nb.csv"));
    let (n, p) = (y.len(), 3);
    let weights: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * ((i % 4) as f64)).collect();
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
            weights: Some(weights),
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "weighted NB GLMM must converge");
    assert!(
        (f.deviance + 2.0 * f.loglik).abs() < 1e-9 * f.deviance.abs(),
        "weighted deviance {} vs -2*loglik {}",
        f.deviance,
        -2.0 * f.loglik
    );
}

/// The `ln θ_NB` coordinate of the outer BOBYQA must reach the same optimum
/// whichever `A`-layout evaluates the marginal objective under it. `sim_nb` is
/// in the dense envelope, so the same design is fit twice: routed (blocked)
/// through `fit_cold`, and forced onto the packed-row layout. `loglik` to 1e-5
/// absolute (optimizer-noise deviance gaps on rungs of this size measured 1e-7
/// to 1e-10 when the exact β-profile landed, 2026-09-04; two decades of margin)
/// and θ̂ to 1e-3 relative.
#[test]
fn nb_coordinate_agrees_across_layouts() {
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
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1, 2],
        wald_se: crate::WaldSe::Rx,
        ..FitOptions::default()
    };
    let blocked = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    let packed = crate::fit::fit_glmm_packed(
        &x,
        &y,
        n,
        p,
        &sized,
        &ids.primary,
        &ids.extra,
        f64::NAN,
        None,
        &opts,
    )
    .0;
    assert!(blocked.converged() && packed.converged());
    assert!(
        (blocked.loglik - packed.loglik).abs() < 1e-5,
        "loglik blocked {} vs packed {}",
        blocked.loglik,
        packed.loglik
    );
    assert!(
        (blocked.dispersion - packed.dispersion).abs() < 1e-3 * packed.dispersion,
        "θ̂ blocked {} vs packed {}",
        blocked.dispersion,
        packed.dispersion
    );
}

/// The GLM outer loop's cold-start seed, pinned against its own formula. The
/// GLMM `ln θ_NB` coordinate reaches it only through `fit_glm_nb`'s start.
#[test]
fn nb_theta_moment_seed_is_the_glm_seed() {
    let y = [0.0, 4.0, 1.0, 7.0, 2.0, 0.0, 3.0, 9.0];
    let ybar = y.iter().sum::<f64>() / 8.0;
    let var = y.iter().map(|v| (v - ybar).powi(2)).sum::<f64>() / 7.0;
    let expect = (ybar * ybar / (var - ybar).max(1e-6))
        .clamp(crate::fit::NB_THETA_LO, crate::fit::NB_THETA_HI);
    assert_eq!(crate::fit::nb_theta_moment_seed(&y, 8), expect);
    // Underdispersed input pins to the ε floor and the top of the box.
    let flat = [2.0; 8];
    assert_eq!(
        crate::fit::nb_theta_moment_seed(&flat, 8),
        crate::fit::NB_THETA_HI
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

    const BAND: f64 = 1e-7;
    // Additive bit-exact pin (see doc comment above). Re-pinned 2026-09-06 (twice).
    // First: θ_NB became a coordinate of the outer BOBYQA on the marginal
    // objective, not a golden-section bracket over re-fits. This fixture takes
    // `OuterSearch::Joint` (nested extra grouping, non-canonical NB-log,
    // n_theta = 2, p = 2) rather than the `ExactProfile` arm `sim_nb` takes —
    // it is the joint-closure pin, where `sim_nb`
    // (`fit_glmm_nb_sim_matches_lme4`) is the stage-1 pin for the same
    // coordinate mechanism. That move was 1.3e-5 relative on θ̂ — inside the
    // bracket's own 1e-4 `ln θ` resolution. Second: the coordinate's cold
    // start moved from the method-of-moments seed to the no-RE GLM-NB's own
    // θ̂, for the same reason as `sim_nb`'s second re-pin (the moment seed
    // charges the RE variance to the dispersion, which breaks random-slope NB
    // shapes elsewhere in the corpus). That move was 1.79e-4 relative on θ̂,
    // still inside the bracket's own resolution. Third (2026-09-10): every
    // θ coordinate is boxed [−THETA_HI, THETA_HI] (`blind_theta_and_bounds`).
    // This fixture has no off-diagonal, so no sign is in play, but the box
    // changes BOBYQA's interpolation set and trust-region steps on every
    // fit. The deviance moved DOWN by 8.6e-3 (605.667764 → 605.659188, 94 →
    // 110 evaluations, both converged, KKT norm 9.9e-3 → 8.8e-3): the old
    // endpoint was the less converged one on the flat ln θ_NB direction.
    // θ̂ moved 3.6e-5 relative, β[0] 2.3e-4, and β̂ sits closer to lme4's
    // (`REF_BETA`) than before. Fourth (2026-09-13): the structured-extras
    // kernel carries an observed-information twin of its factor, so this
    // shape's exact Laplace β-profile is available and the outer search is
    // θ-only on that profile instead of the joint `[θ | β]` search — this
    // fixture moves from `OuterSearch::Joint` to `OuterSearch::ExactProfile`.
    // The deviance moved UP by 4.686e-3 (605.659188 → 605.663874, 110 → 53
    // evaluations, both converged), a smaller swing than the Third re-pin's
    // own 8.6e-3 move on this fixture's flat ln θ_NB direction. The relative
    // move stays tiny everywhere it is checked: θ̂ 2.0e-5, β[0] 6.7e-5, β[1]
    // 3.1e-5, the g1 SD 2.0e-4, the g2:g1 SD 5.5e-6 — all far inside this
    // test's own 5e-3/2e-2/5e-2 lme4 bands. The pin is re-taken at the new
    // point.
    // Fifth (2026-09-15): PIRLS returns the data term, `log|A|` and the factor
    // at the returned iterate, where before the factor and `log|A|` were one
    // Newton step behind. β[0] moved 5.5e-6, β[1] 8.4e-7, the SEs 1.6e-5 and
    // 5.0e-6, τ̂² 3.1e-5 and 9.0e-5, θ̂ 2.1e-5 relative — an order smaller than
    // the Fourth re-pin's own moves on this fixture, and far inside the same
    // 5e-3/2e-2/5e-2 lme4 bands, which the new point meets at β 1.3e-5
    // absolute, se 1.2e-5 relative, the two SDs 8.8e-6 and 5.8e-6, and θ̂
    // 1.9e-6 (2.0e-5 before, so the fit sits closer to lme4's θ̂).
    const REF_BETA_PIN: [f64; 2] = [0.5850115414592737, 0.5073652244060262];
    const REF_SE_PIN: [f64; 2] = [0.20481981166967742, 0.05399297004417363];
    const REF_TAU2_PIN: [f64; 2] = [0.3956652125784431, 0.12616717407427647];
    const REF_THETA_PIN: f64 = 1.430132504448455;
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
/// n_eval are BIT-identical — the bypass is clean. (AGQ fits do not use a
/// Laplace-pass warm start: measured 2026-07-14 on the diligent AGQ cells as a
/// wash. The bypass this canary pins is the shipped state.)
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
        ws.pattern.structured_schur = if ws.groupings.structured_extras_eligible() {
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
        // Re-pinned 2026-09-01: the exact hyper-dual joint (θ, β) Hessian
        // replaced the FD stencil on the
        // blocked GLMM path, which covers this vector-AGQ shape. The fit
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
        // Re-pinned 2026-09-01, same exact-Hessian movement as the k=7 row
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
/// **`ref_se` re-pinned 2026-07-30 (FD-θ-step fix), then re-anchored
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
/// **`ref_se` re-pinned 2026-07-30 (FD-θ-step fix), then re-anchored
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
            let cov = lcg(&mut st);
            // Fixed-effects-only logit — no per-cluster deviation added, so
            // the true random-intercept variance is exactly zero.
            let eta = 0.3 + 0.5 * cov;
            let prob = 1.0 / (1.0 + (-eta).exp());
            let draw = (lcg(&mut st) + 1.0) / 2.0;
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

/// The ρ = −1 trap on a 3-term block, dense GLMM `ExactProfile` route.
/// `tests/fixtures/glmm_npt_trap.csv` is one 800-row Bernoulli draw —
/// `y ~ x1 + x2 + (1 + x1 + x2 | g)`, 40 clusters × 20 rows, true RE
/// correlation ≈ −0.76 between the intercept and `x1`. Nothing in the 48-rung
/// validation corpus exits at a boundary, so without this fixture the trap
/// mechanism would be untested in-crate on this route.
///
/// With the diagonal θ boxed at `[0, THETA_HI]` the search stopped on the face
/// Λ_jj = 0 at 910.3787 with the `x1` diagonal pinned and the entries below
/// it of the wrong sign — a first-order stationary point of the boxed
/// problem, since Σ = ΛΛᵀ is even in each block column and the two sign
/// halves meet only on that face. Under the signed box
/// (`blind_theta_and_bounds`) the face is an interior point and the search
/// walks through it to 910.2127 in one search of 148 evaluations (the
/// retired `npt = n + 2` re-run reached the same basin at 284). lme4 on the
/// same draw: 910.2151 by default and 910.3823 with `nAGQ0initStep = FALSE`
/// — the same two basins, so this is a property of the surface, not of one
/// optimizer. `DEV_BAND` is loose because only the basin is being asserted;
/// the gap between the basins is 170× it.
///
/// The RE stddevs, not the pinned-component flags, are what identifies the
/// basin here. Which diagonal reads as pinned is not stable: the escaped
/// optimum leaves that diagonal within a few 1e-5 of `PIN_THETA`, and merely
/// renumbering the 40 clusters (numeric instead of lexicographic label order —
/// the same model, the same optimum to 1e-9 in deviance) moves it to the other
/// side of the threshold. The stddevs agree to 5e-5 relative across that
/// relabelling and separate the two basins by 8–23%, and they also reproduce
/// lme4's `0.89153 / 0.10321 / 0.23743` on this draw to 0.1%.
#[test]
fn fit_glmm_signed_box_escapes_the_singular_circle() {
    const DEV_BAND: f64 = 1e-3;
    const SD_BAND: f64 = 5e-3;
    const TRAPPED_DEVIANCE: f64 = 910.3787035640;
    const ESCAPED_DEVIANCE: f64 = 910.2127359705;
    let csv = include_str!("../../tests/fixtures/glmm_npt_trap.csv");
    let mut y = Vec::<f64>::new();
    let mut x1 = Vec::<f64>::new();
    let mut x2 = Vec::<f64>::new();
    let mut g_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        x1.push(f[1].parse().unwrap());
        x2.push(f[2].parse().unwrap());
        g_raw.push(f[3].to_string());
    }
    let n = y.len();
    let p = 3;
    let mut x = vec![0.0f64; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = x1[i];
        x[i * p + 2] = x2[i];
    }
    let mut labels: Vec<&str> = g_raw.iter().map(String::as_str).collect();
    labels.sort_unstable();
    labels.dedup();
    let primary: Vec<u32> = g_raw
        .iter()
        .map(|l| labels.iter().position(|x| x == l).unwrap() as u32)
        .collect();
    let n_clusters = labels.len();
    let ids = GroupIds {
        primary,
        extra: vec![],
    };
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![1, 2],
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
            target_indices: (0..p as u32).collect(),
            ..FitOptions::default()
        },
    );
    assert!(f.converged(), "trap draw must converge");
    assert!(
        (f.deviance - ESCAPED_DEVIANCE).abs() < DEV_BAND,
        "deviance {} is not the interior basin {ESCAPED_DEVIANCE} (trapped basin is {TRAPPED_DEVIANCE})",
        f.deviance
    );
    assert!(
        f.deviance < TRAPPED_DEVIANCE - DEV_BAND,
        "the signed box must beat the boxed search's face, got {} vs {TRAPPED_DEVIANCE}",
        f.deviance
    );
    assert_pinned(
        &f.stddev_corr(0).0,
        &[0.8922556154657538, 0.10371641873371715, 0.23757089840392914],
        SD_BAND,
        "trap draw stddev",
    );
}

/// One measured sign-trap draw per dense GLMM route shape, each a simulated
/// 300-row draw (30 groups × 10 rows, `g1` the one grouping column) frozen
/// as a fixture from the 2026-09-10 sign-trap simulation study. `stopped` is
/// where the search
/// ended with the diagonal θ boxed at `[0, THETA_HI]` — a diagonal on the
/// face Λ_jj = 0 with entries below it of the wrong sign, or, on the
/// "walk-back" and "empty-column" draws, a stop the retired flip / push /
/// kick re-runs were needed for — and `reached` is what the signed box
/// reaches in one search. Both are asserted: the fit must land within
/// `DEV_BAND` of `reached` and below `stopped` by more than `DEV_BAND`.
/// Convergence, not just the deviance, is checked on every draw.
#[cfg(feature = "formula")]
fn assert_sign_trap_escaped(
    what: &str,
    csv: &str,
    formula: &str,
    family: Family,
    nagq: u8,
    stopped: f64,
    reached: f64,
) {
    const DEV_BAND: f64 = 1e-3;
    let lo = super::common_tests::fixture_lowered(csv, &["g1"], formula, family);
    let opts = FitOptions {
        nagq,
        ..lo.opts.clone()
    };
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &opts);
    assert!(
        f.converged(),
        "{what}: must converge ({:?})",
        f.diagnostics.boundary
    );
    let dev = -2.0 * f.loglik;
    assert!(
        (dev - reached).abs() < DEV_BAND,
        "{what}: −2·loglik {dev} is not the escaped basin {reached} (boxed search stopped at {stopped}; {} evaluations)",
        f.n_eval
    );
    assert!(
        dev < stopped - DEV_BAND,
        "{what}: −2·loglik {dev} does not beat the boxed search's stop {stopped}"
    );
}

/// Dense GLMM `ExactProfile`: Bernoulli `y ~ x1 + (1 + x1 | g1)`, generated
/// with a zero random-slope SD. Boxed, the search stopped on Λ₀₀ = 0 with
/// λ₁₀ = 0.094 (internal) below it, deviance 397.6085, the intercept pinned
/// (57 evaluations, 105 with the retired re-run); the signed box reaches
/// 397.4577 in 55, with the slope diagonal at the pin instead.
#[cfg(feature = "formula")]
#[test]
fn fit_glmm_sign_trap_dense_exact_profile() {
    assert_sign_trap_escaped(
        "dense ExactProfile",
        include_str!("../../tests/fixtures/sign_trap_glmm_dense_slope.csv"),
        "y ~ x1 + (1 + x1 | g1)",
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        1,
        397.6085441897,
        397.4576798411,
    );
}

/// Dense GLMM `Joint` under AGQ (nAGQ = 3 on a 3-term block, the vector
/// kernel): Bernoulli `y ~ x1 + x2 + (1 + x1 + x2 | g1)`, generated with a
/// rank-1 random-effect covariance. Boxed, the search stopped at 387.3085 in
/// 270 evaluations; the signed box
/// reaches 387.2703 in 316. lme4 refuses nAGQ > 1 on a vector random effect,
/// so this shape has no external reference — the two basins are glmm's own.
#[cfg(feature = "formula")]
#[test]
fn fit_glmm_sign_trap_dense_joint_agq() {
    assert_sign_trap_escaped(
        "dense Joint AGQ",
        include_str!("../../tests/fixtures/sign_trap_glmm_agq_q3.csv"),
        "y ~ x1 + x2 + (1 + x1 + x2 | g1)",
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        3,
        387.3084738547,
        387.2702811551,
    );
}

/// The walk-back case: Poisson `y ~ x1 + x2 + (1 + x1 + x2 | g1)`, generated
/// with a rank-1 random-effect covariance, dense `ExactProfile`. Boxed, the
/// first search stopped
/// at 998.0913 with the `x1` diagonal on the face and both entries below it
/// non-zero; a flipped warm start alone did not reach the optimum (it needed
/// the diagonal pushed off the face first). The signed box reaches 994.7496
/// in 211 evaluations (368 with the retired re-run).
#[cfg(feature = "formula")]
#[test]
fn fit_glmm_sign_trap_walk_back() {
    assert_sign_trap_escaped(
        "walk-back",
        include_str!("../../tests/fixtures/sign_trap_glmm_walkback_q3.csv"),
        "y ~ x1 + x2 + (1 + x1 + x2 | g1)",
        Family::Poisson {
            link: PoissonLink::Log,
        },
        1,
        998.0913393396,
        994.7495903063,
    );
}

/// The empty-column case: Bernoulli, same formula as the walk-back draw,
/// generated with random-effect correlations of 0.95. Boxed, the first
/// search stopped at 384.1806 with the
/// `x1` diagonal on the face and nothing below it to flip (both entries at
/// or under the pin threshold), so no sign-flip re-run could move it. The
/// signed box reaches 384.0195 in 110 evaluations (189 with the retired
/// re-run).
#[cfg(feature = "formula")]
#[test]
fn fit_glmm_sign_trap_empty_column() {
    assert_sign_trap_escaped(
        "empty column",
        include_str!("../../tests/fixtures/sign_trap_glmm_emptycol_q3.csv"),
        "y ~ x1 + x2 + (1 + x1 + x2 | g1)",
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        1,
        384.1806428649,
        384.0194701453,
    );
}

/// Large-θ̂ coverage, AGQ arm: binomial GLMM
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

/// The θ = 0 end of the large-θ̂ coverage axis, on the **committed**
/// `sim_binomial_zerosd` fixture rather than a synthetic draw.
///
/// This gates a *documented behavioural divergence*, not a number, and it is why
/// this test is deliberately NOT a curated `datasets` rung (`//large_theta_rungs` in
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
/// perfect agreement. Two further reasons the rung track is closed to this test, both
/// measured 2026-07-30 rather than assumed: lme4's
/// `vcov(m, use.hessian = TRUE)` — exactly what `engines/lme4.R:269` and `:146`
/// call — **hard-errors** on this fit (`'use.hessian'=TRUE specified, but Hessian
/// is unavailable`; `m@optinfo$derivs` is `NULL` on a boundary fit), so running it as
/// a rung would abort the whole oracle run; and at θ̂ = 0 the θ↔β coupling block
/// vanishes, so `se_hessian` and `se_rx` collapse onto each other (9.4e-6 apart
/// here, against 9.7e-2 on `fit_glmm_binomial_bigsd_agq_matches_lme4`) — this test
/// gates nothing about the coupling term.
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
        let x1 = lcg(&mut st);
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
    ws.pattern.structured_schur = if ws.groupings.structured_extras_eligible() {
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

/// Uniform(0,1) off the shared LCG (`lcg` is uniform on (-1,1)).
fn nb_slope_uniform01(st: &mut u64) -> f64 {
    (super::common_tests::lcg(st) + 1.0) / 2.0
}

/// Standard normal via a Box-Muller transform of two `nb_slope_uniform01` draws.
fn nb_slope_std_normal(st: &mut u64) -> f64 {
    let u1 = nb_slope_uniform01(st).max(1e-12);
    let u2 = nb_slope_uniform01(st);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gamma(shape, scale) via Marsaglia-Tsang (2000): direct rejection sampling
/// for `shape >= 1`, boosted by one unit and corrected by a
/// `Uniform(0,1)^(1/shape)` factor below that.
fn nb_slope_gamma(st: &mut u64, shape: f64, scale: f64) -> f64 {
    if shape < 1.0 {
        let u = nb_slope_uniform01(st);
        return nb_slope_gamma(st, shape + 1.0, scale) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let mut x;
        let mut v;
        loop {
            x = nb_slope_std_normal(st);
            v = 1.0 + c * x;
            if v > 0.0 {
                break;
            }
        }
        v = v * v * v;
        let u = nb_slope_uniform01(st);
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            return d * v * scale;
        }
    }
}

/// Poisson(lambda) via Knuth's multiplicative sampler — exact, and cheap at
/// the small means this fixture draws.
fn nb_slope_poisson(st: &mut u64, lambda: f64) -> f64 {
    let l = (-lambda).exp();
    let mut k = 0.0;
    let mut p = 1.0;
    loop {
        k += 1.0;
        p *= nb_slope_uniform01(st);
        if p <= l {
            break;
        }
    }
    k - 1.0
}

/// Negative-binomial counts on an imbalanced correlated random-INTERCEPT-AND-
/// SLOPE design: `y ~ 1 + x + (1 + x | g)`, 15 groups over 300 rows, 20% of
/// the groups carrying 80% of the rows. Each group's `(intercept, slope)`
/// pair is a correlated bivariate normal (correlation 0.2, sd 0.6 on the
/// intercept and 0.54 on the slope) built from two `nb_slope_std_normal`
/// draws; `x` is standard normal; counts are a genuine Gamma(θ, μ/θ)-Poisson
/// mixture at θ = 1.5 (`nb_slope_gamma` + `nb_slope_poisson`), which is
/// exactly how `MASS::rnegbin` draws negative-binomial counts. Every draw
/// comes off one seeded LCG, so the fixture is exactly reproducible.
///
/// The group-size imbalance is what makes the naive method-of-moments
/// dispersion seed collapse far enough below θ̂ for a fixed seed to still
/// reliably trip the bug this file guards against (see the test below): a
/// balanced version of the same random-slope design usually seeds close
/// enough to θ̂ that PIRLS still converges from there, so it doesn't
/// reproduce the failure on every run.
fn sim_nb_slope_dataset() -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
    const N: usize = 300;
    const N_CLUSTERS: usize = 15;
    const BETA0: f64 = 0.4;
    const BETA1: f64 = 0.8;
    const SD0: f64 = 0.6;
    const SD1: f64 = 0.54;
    const RHO: f64 = 0.2;
    const THETA_TRUE: f64 = 1.5;
    const N_HEAVY: usize = 3; // 20% of 15 groups carry 80% of the rows

    let mut st = 900012u64;
    let mut cluster_ids = vec![0u32; N];
    for (i, cid) in cluster_ids.iter_mut().enumerate() {
        // 80% of rows round-robin over the N_HEAVY heavy groups, the
        // remaining 20% round-robin over the rest — deterministic stand-in
        // for the weighted multinomial group draw an R simulation would use.
        let u = nb_slope_uniform01(&mut st);
        *cid = if u < 0.8 {
            (i % N_HEAVY) as u32
        } else {
            (N_HEAVY + i % (N_CLUSTERS - N_HEAVY)) as u32
        };
    }
    let mut b0 = [0.0f64; N_CLUSTERS];
    let mut b1 = [0.0f64; N_CLUSTERS];
    for g in 0..N_CLUSTERS {
        let z0 = nb_slope_std_normal(&mut st);
        let z1 = nb_slope_std_normal(&mut st);
        b0[g] = SD0 * z0;
        b1[g] = SD1 * (RHO * z0 + (1.0 - RHO * RHO).sqrt() * z1);
    }
    let mut x = vec![0.0f64; N * 2];
    let mut y = vec![0.0f64; N];
    for i in 0..N {
        let g = cluster_ids[i] as usize;
        let xi = nb_slope_std_normal(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = xi;
        let mu = (BETA0 + BETA1 * xi + b0[g] + b1[g] * xi).exp();
        let lambda = nb_slope_gamma(&mut st, THETA_TRUE, mu / THETA_TRUE);
        y[i] = nb_slope_poisson(&mut st, lambda);
    }
    (x, y, cluster_ids, N_CLUSTERS)
}

/// Pins the fix for the NB dispersion coordinate's cold start on a
/// random-slope shape. `fit_glmm_nb` searches `ln θ_NB` as a trailing
/// coordinate of the outer BOBYQA over the marginal objective, and that
/// coordinate needs a starting value where the exact-profile PIRLS inside it
/// actually converges. The naive method-of-moments seed
/// (`nb_theta_moment_seed`, `ȳ²/(s²−ȳ)`) charges every source of variance in
/// `y` — including random-slope variance — to the NB dispersion, so on a
/// random-slope shape it can land one to two orders of magnitude below the
/// fitted θ̂, on a start where PIRLS does not converge. That handed the outer
/// BOBYQA a `+∞` incumbent it could not climb out of: on this exact fixture,
/// reverting to the naive seed makes the fit report `converged = false` with
/// `loglik = NaN` after only a handful of evaluations, stuck at the seed
/// value. The fix seeds the coordinate from the no-RE NB GLM's own θ̂
/// (`fit_glm_nb`) instead, which starts inside the basin PIRLS can converge
/// in.
///
/// Every other negative-binomial fixture in this file is intercept-shaped
/// (one scalar grouping, or two scalar-intercept blocks), so none of them
/// exercise the random-slope case. This fixture uses `y ~ 1 + x + (1 + x | g)`
/// — two random-effect variances and a correlation (`n_theta = 3`), `p = 2`
/// fixed effects — the shape the naive seed breaks. The test checks the
/// fixture actually reproduces that precondition (the moment seed at least
/// 10x below θ̂) before trusting anything else, so a fixture whose
/// random-effect variance or group imbalance drifted over time cannot pass
/// while silently no longer covering the bug. It then checks the routed fit
/// converges and agrees with the same design forced onto the packed-row
/// layout — an independent `A`-layout under the same outer search, on the same
/// marginal objective.
#[test]
fn fit_glmm_nb_random_slope_seed_lands_in_convergent_basin() {
    let (x, y, cluster_ids, n_clusters) = sim_nb_slope_dataset();
    let (n, p) = (y.len(), 2);
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
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
        primary: cluster_ids,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        wald_se: crate::WaldSe::Rx,
        ..FitOptions::default()
    };

    let blocked = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(blocked.converged(), "NB random-slope GLMM must converge");

    // The precondition the original bug needed: the naive moment seed must
    // land at least an order of magnitude below θ̂. A fixture that lost this
    // property would no longer start the coordinate outside the convergent
    // basin, so it would stop being a regression test for the bug at all.
    let moment_seed = crate::fit::nb_theta_moment_seed(&y, n);
    assert!(
        moment_seed < blocked.dispersion / 10.0,
        "fixture must put the moment seed >=10x below θ̂: seed={moment_seed} θ̂={}",
        blocked.dispersion
    );

    let (sized, ids, _perm) = crate::fit::spec_sized_from_ids_pub(&model, &ids);
    let packed = crate::fit::fit_glmm_packed(
        &x,
        &y,
        n,
        p,
        &sized,
        &ids.primary,
        &ids.extra,
        f64::NAN,
        None,
        &opts,
    )
    .0;
    assert!(
        packed.converged(),
        "packed NB random-slope reference must converge"
    );
    assert!(
        (blocked.loglik - packed.loglik).abs() < 1e-5,
        "loglik blocked {} vs packed {}",
        blocked.loglik,
        packed.loglik
    );
    assert!(
        (blocked.dispersion - packed.dispersion).abs() < 1e-3 * packed.dispersion,
        "θ̂ blocked {} vs packed {}",
        blocked.dispersion,
        packed.dispersion
    );
}

/// Negative-binomial counts (as a Poisson-Exponential(1) mixture, i.e. NB with
/// shape 1) on a random-INTERCEPT-only design: `y ~ 1 + x + (1 | g)`, 10
/// clusters x 3 rows. This is the pathological-sweep cell that reproduces the
/// `+INF` plateau: 15 of the 16 gating-stage BOBYQA evaluations
/// diverge in PIRLS (`laplace_deviance` returns `+INFINITY`), one evaluation
/// is finite, and — before the finite-eval-count guard — the run exits
/// `Status::Converged` at the untouched θ_RE cold start (`THETA0 = 1.0`),
/// with a KKT residual nowhere near a stationary point. Every draw comes off
/// one seeded LCG, so the fixture is exactly reproducible.
fn sim_nb_inf_plateau_dataset() -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
    const N_CLUSTERS: usize = 10;
    const PER: usize = 3;
    const SD_INT: f64 = 1.5;
    const SD_SLOPE: f64 = 4.0;
    const SEED: u64 = 4;

    let mut state = SEED
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0x1234_5678 ^ (N_CLUSTERS as u64) << 20);
    let n = N_CLUSTERS * PER;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    let mut cluster_ids = Vec::with_capacity(n);
    for c in 0..N_CLUSTERS {
        let ui = SD_INT * inf_plateau_normal(&mut state);
        let us = SD_SLOPE * inf_plateau_normal(&mut state);
        for _ in 0..PER {
            let xv = inf_plateau_lcg_next(&mut state) * 2.0 - 1.0;
            let eta = 0.5 + 0.8 * xv + ui + us * xv;
            let mu = eta.exp().clamp(1e-8, 1e6);
            let e = inf_plateau_exp1(&mut state);
            x.push(1.0);
            x.push(xv);
            y.push(inf_plateau_poisson(&mut state, mu * e));
            cluster_ids.push(c as u32);
        }
    }
    (x, y, cluster_ids, N_CLUSTERS)
}

/// Pins the fix for the `+INF` plateau: BOBYQA's `moderatef` maps both `NaN`
/// and `+inf` to `FUNCMAX`, so a gating-stage search where PIRLS diverges on
/// (almost) every evaluation is a flat *finite* surface — it shrinks to
/// `rho_end` and exits `Status::Converged` having compared close to nothing.
/// On this fixture only 1 of 16 stage-1 evaluations is genuinely finite, so
/// before the fix the fit reported `converged = true` with `theta_RE` still
/// at its cold start and a KKT residual nowhere near zero. The fix requires
/// at least 2 finite evaluations before trusting `Status::Converged`.
#[test]
fn fit_glmm_nb_random_intercept_inf_plateau_does_not_converge() {
    let (x, y, cluster_ids, n_clusters) = sim_nb_inf_plateau_dataset();
    let n = y.len();
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
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let opts = FitOptions::default();

    let f = fit_cold(&x, &y, n, 2, &model, &ids, &opts);

    assert!(
        !f.converged(),
        "the +INF plateau fit must not report converged"
    );
    assert!(
        f.varcorr.iter().all(|row| row.iter().all(|v| v.is_nan())),
        "varcorr must be NaN-filled on a non-converged fit: {:?}",
        f.varcorr
    );
    assert!(
        f.tau2.iter().all(|t| t.is_nan()),
        "tau2 must be NaN-filled on a non-converged fit: {:?}",
        f.tau2
    );
}

/// Perfectly separated data forced through the dense GLMM route:
/// `y ~ x + (1|g)`, `y ∈ {1e-6, 1e6}` split exactly on `x ∈ {0, 1}` — no
/// finite Gamma-log fit exists. `dispersion` must be NaN, not the Gamma
/// exponential special case `φ=1`, which a caller cannot tell from a real
/// estimate.
#[test]
fn fit_glmm_gamma_failed_fit_dispersion_is_nan() {
    let n = 24;
    let p = 2;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        x[i * p] = 1.0;
        x[i * p + 1] = if i < 12 { 0.0 } else { 1.0 };
        y[i] = if i < 12 { 1e-6 } else { 1e6 };
    }
    let cluster_ids: Vec<u32> = (0..n as u32).map(|i| i % 4).collect();
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let f = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(
        !f.converged(),
        "perfectly separated Gamma GLMM must not converge"
    );
    assert!(
        f.dispersion.is_nan(),
        "dispersion must be NaN on a failed fit, not the Gamma exponential special case 1.0: {}",
        f.dispersion
    );
}

/// Negative-binomial log link, dense route: `dispersion` must be NaN rather
/// than whatever θ coordinate the outer BOBYQA search happened to be
/// standing on when it gave up. The perfectly-separated reproducer above
/// (`y ∈ {1e-6, 1e6}` on `x ∈ {0, 1}`) converges fine under NB — the extra θ
/// coordinate absorbs the separation that starves Gamma, parking at
/// `NB_THETA_HI` and reporting `converged = true` (boundary counts as
/// converged) — so this instead reuses [`sim_nb_inf_plateau_dataset`]'s LCG
/// (random-slope-generated counts fit as random-INTERCEPT-only) at a much
/// larger slope spread (`sd_slope = 8`, vs. that fixture's `4`) so that NO
/// evaluation is ever finite, not just "almost every" one. That is
/// deliberately outside the reach of the separate `+INF`-plateau finite-eval-
/// count guard (`fit_glmm_nb_random_intercept_inf_plateau_
/// does_not_converge`): `best` never turns finite in the first place, so this
/// reports `converged = false` on the guard's ORIGINAL logic already, with no
/// dependency on that separate fix's state.
#[test]
fn fit_glmm_nb_failed_fit_dispersion_is_nan() {
    const N_CLUSTERS: usize = 10;
    const PER: usize = 3;
    const SD_INT: f64 = 1.5;
    const SD_SLOPE: f64 = 8.0;
    const SEED: u64 = 4;
    let mut state = SEED
        .wrapping_mul(0x9E3779B97F4A7C15)
        .wrapping_add(0x1234_5678 ^ (N_CLUSTERS as u64) << 20);
    let n = N_CLUSTERS * PER;
    let mut x = Vec::with_capacity(n * 2);
    let mut y = Vec::with_capacity(n);
    let mut cluster_ids = Vec::with_capacity(n);
    for c in 0..N_CLUSTERS {
        let ui = SD_INT * inf_plateau_normal(&mut state);
        let us = SD_SLOPE * inf_plateau_normal(&mut state);
        for _ in 0..PER {
            let xv = inf_plateau_lcg_next(&mut state) * 2.0 - 1.0;
            let eta = 0.5 + 0.8 * xv + ui + us * xv;
            let mu = eta.exp().clamp(1e-8, 1e6);
            let e = inf_plateau_exp1(&mut state);
            x.push(1.0);
            x.push(xv);
            y.push(inf_plateau_poisson(&mut state, mu * e));
            cluster_ids.push(c as u32);
        }
    }
    let model = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: N_CLUSTERS as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: cluster_ids,
        extra: vec![],
    };
    let f = fit_cold(&x, &y, n, 2, &model, &ids, &FitOptions::default());
    assert!(
        !f.converged(),
        "random-slope counts fit as random-intercept-only at this slope spread must not converge"
    );
    assert!(
        f.dispersion.is_nan(),
        "dispersion must be NaN on a failed fit, not the θ the outer search stood on: {}",
        f.dispersion
    );
}

/// One dense Laplace GLMM rung of the validation corpus, driven from the
/// committed dataset the harness fits: the CSV, the formula, the family/link
/// and the per-rung weights/offset the manifest records.
///
/// `formula` is the manifest's own formula with the two mechanical rewrites the
/// validation harness applies before lowering: the `@formula(...)` wrapper and
/// the explicit `1` intercept term come off a `jl_formula` (this parser treats
/// the intercept as implicit and has no term for a literal `1`), and an
/// aggregated-binomial `cbind(s, n - s)` response becomes the synthesized
/// `prop` column, which with the trial count as prior weights IS lme4's own
/// objective for that response.
struct DenseLaplaceRung {
    rung: u32,
    csv: &'static str,
    formula: &'static str,
    family: Family,
    /// Columns lowered as categorical. Everything else is numeric, except a
    /// column that fails to parse as `f64` anywhere, which falls back to a
    /// factor — a CSV can carry a categorical helper column no formula names.
    factors: &'static [&'static str],
    /// Aggregated binomial `(successes, trials)` column names: the response
    /// becomes `successes/trials` in a synthesized `prop` column and the trials
    /// become prior weights.
    agg: Option<(&'static str, &'static str)>,
    /// Per-row prior weights read off this column.
    weights_col: Option<&'static str>,
    /// Per-row known additive term on the linear-predictor scale.
    offset_col: Option<&'static str>,
}

/// Every non-Gaussian dense-routed rung of the validation corpus, at Laplace
/// (`nagq = 1`) — the set whose standard errors the exact joint Hessian
/// produces today. The AGQ orders some of these carry in the manifest are not
/// applied: quadrature keeps the packed second-order pass and is out of this
/// comparison's scope.
const DENSE_LAPLACE_RUNGS: &[DenseLaplaceRung] = &[
    DenseLaplaceRung {
        rung: 5,
        csv: include_str!("../../validation/data/empirical/cbpp.csv"),
        formula: "prop ~ period + (1 | herd)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["herd", "period"],
        agg: Some(("incidence", "size")),
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 6,
        csv: include_str!("../../validation/data/empirical/grouseticks.csv"),
        formula: "TICKS ~ YEAR + cHEIGHT + (1 | BROOD) + (1 | INDEX) + (1 | LOCATION)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["BROOD", "INDEX", "LOCATION", "YEAR"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 12,
        csv: include_str!("../../validation/data/empirical/VerbAgg.csv"),
        formula: "y ~ Anger + Gender + btype + situ + mode + (1|id) + (1|item)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["Gender", "btype", "situ", "mode", "id", "item"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 14,
        csv: include_str!("../../validation/data/empirical/Arabidopsis.csv"),
        formula: "total_fruits ~ nutrient + amd + rack + status + (1 | popu/gen)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["nutrient", "rack", "amd", "status", "popu", "gen"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 17,
        csv: include_str!("../../validation/data/simulated/sim_crossed_at_cap.csv"),
        formula: "y ~ x + (1|g1) + (1|c1) + (1|c2) + (1|c3) + (1|c4) + (1|c5) + (1|c6)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["g1", "c1", "c2", "c3", "c4", "c5", "c6"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 19,
        csv: include_str!("../../validation/data/simulated/sim_poisson_nested.csv"),
        formula: "y ~ x + (1 | g1/g2)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["g1", "g2"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 22,
        csv: include_str!("../../validation/data/empirical/cbpp.csv"),
        formula: "prop ~ period + (1 | herd)",
        family: Family::Binomial {
            link: BinomialLink::Probit,
        },
        factors: &["herd", "period"],
        agg: Some(("incidence", "size")),
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 23,
        csv: include_str!("../../validation/data/simulated/sim_gamma.csv"),
        formula: "y ~ x + grp + (1 | cluster)",
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        factors: &["cluster", "grp"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 25,
        csv: include_str!("../../validation/data/simulated/sim_binomial_slope1.csv"),
        formula: "y ~ x + (1 + x | g)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["g"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 26,
        csv: include_str!("../../validation/data/simulated/sim_poisson_slope1.csv"),
        formula: "y ~ x + (1 + x | g)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["g"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 27,
        csv: include_str!("../../validation/data/simulated/sim_binomial_slope2.csv"),
        formula: "y ~ x1 + x2 + (1 + x1 + x2 | g)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["g"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 28,
        csv: include_str!("../../validation/data/simulated/sim_poisson_offset.csv"),
        formula: "y ~ x + (1 | cluster)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["cluster"],
        agg: None,
        weights_col: None,
        offset_col: Some("log_exposure"),
    },
    DenseLaplaceRung {
        rung: 37,
        csv: include_str!("../../validation/data/simulated/glmm_poisson.csv"),
        formula: "y ~ x + (1 | g)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["g"],
        agg: None,
        weights_col: Some("w"),
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 44,
        csv: include_str!("../../validation/data/simulated/sim_binomial_bigsd.csv"),
        formula: "y ~ x + z + (1 | g)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["g"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 45,
        csv: include_str!("../../validation/data/simulated/sim_poisson_bigsd.csv"),
        formula: "y ~ x + z + (1 | g)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["g"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 48,
        csv: include_str!("../../validation/data/simulated/sim_probit_large.csv"),
        formula: "y ~ x1 + x2 + x3 + z + (1 | g)",
        family: Family::Binomial {
            link: BinomialLink::Probit,
        },
        factors: &["g"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 49,
        csv: include_str!("../../validation/data/simulated/sim_cloglog_nested_crossed.csv"),
        formula: "y ~ x + (1 | g1/g2) + (1 | c1)",
        family: Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        factors: &["g1", "g2", "c1"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
];

/// Header + row cells of an embedded CSV, quotes stripped. Mirrors the
/// validation harness's own reader; the corpus CSVs carry no embedded commas.
fn rung_csv(csv: &'static str) -> (Vec<String>, Vec<Vec<String>>) {
    let unquote = |s: &str| s.trim().trim_matches('"').to_string();
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .expect("csv header")
        .split(',')
        .map(unquote)
        .collect();
    let rows: Vec<Vec<String>> = lines.map(|l| l.split(',').map(unquote).collect()).collect();
    (header, rows)
}

/// The rung's CSV as a lowering `Table`, plus the prior weights an aggregated
/// binomial response implies. A column named in `factors`, or one that fails to
/// parse as `f64` anywhere, becomes a factor at its lexicographic level order —
/// which is what R's own `factor()` did when the references were frozen.
/// Header dots become underscores, as this parser cannot read a dot inside an
/// identifier (`Arabidopsis`'s `total.fruits`).
#[cfg(feature = "formula")]
fn rung_table(r: &DenseLaplaceRung) -> (crate::formula::Table, Option<Vec<f64>>, Option<Vec<f64>>) {
    use crate::formula::{Column, Table};
    let (header, rows) = rung_csv(r.csv);
    let n = rows.len();
    let numeric =
        |j: usize| -> Vec<f64> { rows.iter().map(|row| row[j].parse().unwrap()).collect() };
    let mut columns: Vec<(String, Column)> = header
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let is_factor = r.factors.contains(&name.as_str())
                || rows.iter().any(|row| row[j].parse::<f64>().is_err());
            let col = if is_factor {
                let labels: Vec<String> = rows.iter().map(|row| row[j].clone()).collect();
                Column::factor_from_labels(&labels)
            } else {
                Column::Numeric(numeric(j))
            };
            (name.replace('.', "_"), col)
        })
        .collect();
    let col_of = |name: &str| -> Vec<f64> {
        let j = header
            .iter()
            .position(|h| h == name)
            .unwrap_or_else(|| panic!("rung {}: column {name:?} not in header", r.rung));
        numeric(j)
    };
    let weights = match r.agg {
        Some((succ, trials)) => {
            let s = col_of(succ);
            let t = col_of(trials);
            let prop: Vec<f64> = s.iter().zip(&t).map(|(a, b)| a / b).collect();
            columns.push(("prop".into(), Column::Numeric(prop)));
            Some(t)
        }
        None => r.weights_col.map(col_of),
    };
    let offset = r.offset_col.map(col_of);
    (Table { columns, n }, weights, offset)
}

/// Rows the kernel's own clamps hold at the workspace's current point:
/// `(μ at `family::clamp_mu`'s bound, η at the link's `clamp_eta` bound)`.
///
/// The μ count comes from `assembled::mu_clamped_rows`, which is what the
/// assembled engine itself refuses on, so this census and that refusal cannot
/// drift apart; `weighted` is what tells the two routes apart on the
/// unweighted-logit exemption. The η count reads `family::clamp_eta_bounds`,
/// the same table `clamp_eta` holds η inside, for the same reason. No corpus
/// rung reaches an η clamp at γ̂, and one appearing is a regime change worth
/// failing on.
#[cfg(feature = "formula")]
fn clamp_census(ws: &GlmmWorkspace, family: Family, weighted: bool, n: usize) -> (usize, usize) {
    let (lo_eta, hi_eta) = crate::family::clamp_eta_bounds(family);
    let mu_clamped = crate::glmm::mu_clamped_rows(family, weighted, &ws.pirls.prob[..n]);
    let mut eta_clamped = 0usize;
    for i in 0..n {
        if ws.pirls.eta[i] <= lo_eta || ws.pirls.eta[i] >= hi_eta {
            eta_clamped += 1;
        }
    }
    (mu_clamped, eta_clamped)
}

/// One rung lowered, fitted to its own γ̂ with `WaldSe::Rx` (so the fit runs
/// neither Hessian pass), and handed back as the workspace sitting at that γ̂
/// with everything the two Hessian engines need.
#[cfg(feature = "formula")]
#[allow(clippy::type_complexity)]
fn rung_at_gamma_hat(
    r: &DenseLaplaceRung,
) -> (
    GlmmWorkspace,
    Mat<f64>,
    Vec<f64>,
    Vec<u32>,
    Vec<Vec<u32>>,
    usize,
    usize,
    f64,
) {
    let (table, weights, offset) = rung_table(r);
    let lo = crate::formula::lower(r.formula, &table, r.family)
        .unwrap_or_else(|e| panic!("rung {}: lower: {e}", r.rung));
    let opts = FitOptions {
        target_indices: lo.opts.target_indices.clone(),
        wald_se: WaldSe::Rx,
        weights: weights.or(lo.opts.weights.clone()),
        offset: offset.or(lo.opts.offset.clone()),
        nagq: 1,
        ..FitOptions::default()
    };
    // Level counts come from the ids, exactly as the shipped dispatch derives
    // them before it builds a workspace; the lowering leaves placeholders.
    let (sized_model, sized_ids, _perm) = super::spec_sized_from_ids(&lo.model, &lo.ids);
    let (mut ws, x_mat) = super::glmm::fit_glmm_build(
        &lo.x,
        lo.n,
        lo.p,
        &sized_model,
        &sized_ids.primary,
        &sized_ids.extra,
        &opts,
    )
    .unwrap_or_else(|_| panic!("rung {}: degenerate design", r.rung));
    // The shipped dispatch's own inner call, so the fit reaches the same γ̂ the
    // corpus is fitted at — β seeded from the no-RE GLM, θ from the kernel's
    // blind start.
    let deviance = {
        let view = super::glmm::run_glmm_on(
            &mut ws,
            x_mat.as_ref(),
            &lo.y,
            lo.n,
            lo.p,
            &sized_model,
            &sized_ids.primary,
            &sized_ids.extra,
            f64::NAN,
            None,
            &opts,
        );
        let (converged, deviance) = view.converged_deviance();
        assert!(converged, "rung {}: fit must converge", r.rung);
        deviance
    };
    // Either exact-derivative owner will do: `supports_exact_shape` for a
    // layout with a dual kernel, `assembly_routes` for the packed-row layout,
    // which has only the assembled engine. Each rung list fixes which.
    assert!(
        crate::glmm::supports_exact_shape(ws.layout, &ws.groupings)
            || crate::glmm::assembly_routes(&ws, lo.n),
        "rung {}: exact-derivative shape expected",
        r.rung
    );
    let sized_ids = sized_ids.into_owned();
    (
        ws,
        x_mat,
        lo.y,
        sized_ids.primary,
        sized_ids.extra,
        lo.p,
        lo.n,
        deviance,
    )
}

/// One row of the corpus gate's printed table.
#[cfg(feature = "formula")]
struct RungReport {
    rung: u32,
    m: usize,
    /// Worst relative gap between the `f64` assembled gradient and the dual one.
    worst_grad: f64,
    /// Worst relative gap between the two Hessians, per entry.
    worst: f64,
    /// The assembled pass's own pre-symmetrization asymmetry.
    asym: f64,
    /// Each pass's exit `‖u − u_prev‖`.
    step_asm: f64,
    step_hd: f64,
    /// Rows on the μ clamp at γ̂.
    mu_clamped: usize,
}

/// Richardson-extrapolated central second difference of the f64 Laplace
/// deviance, `(4·D_{base/2} − D_base)/3` — an independent arbiter for
/// the `CLOGLOG_LARGE` fixture, checked against the assembled Hessian without
/// going through the hyper-dual pass. Step absolute on θ (`k < n_theta`),
/// relative × `max(|β|, 1)` on β, mirroring `packed_assembled_gradient_matches_richardson_fd`'s
/// FD shape one derivative order down. `params` is γ̂, captured ONCE by the
/// caller before any evaluation: `glmm_laplace_deviance` copies its argument
/// into `ws.params` and leaves it there, so re-reading `ws.params` as the
/// centre mid-sweep would walk the centre off γ̂ after the first perturbed
/// call. `ws.pirls.u` is zeroed before every evaluation, so each one is a
/// cold PIRLS solve at its own perturbed point, never warm-started from the
/// last.
#[cfg(feature = "formula")]
#[allow(clippy::too_many_arguments)]
fn richardson_fd_hessian(
    ws: &mut GlmmWorkspace,
    x: faer::MatRef<f64>,
    y: &[f64],
    ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
    m: usize,
    n_theta: usize,
    base: f64,
    params: &[f64],
) -> Mat<f64> {
    let params: Vec<f64> = params.to_vec();
    let step = |k: usize, b: f64| -> f64 {
        if k < n_theta {
            b
        } else {
            b * params[k].abs().max(1.0)
        }
    };
    let mut at = |shifts: &[(usize, f64)]| -> f64 {
        let mut q = params.clone();
        for &(k, s) in shifts {
            q[k] += s;
        }
        ws.pirls.u.fill(0.0);
        glmm_laplace_deviance(&q, ws, x, y, ids, extra_ids, n)
    };
    let f0 = at(&[]);
    let mut raw = |b: f64| -> Mat<f64> {
        let mut h = Mat::<f64>::zeros(m, m);
        for i in 0..m {
            let hi = step(i, b);
            h[(i, i)] = (at(&[(i, hi)]) - 2.0 * f0 + at(&[(i, -hi)])) / (hi * hi);
            for j in 0..i {
                let hj = step(j, b);
                let v =
                    (at(&[(i, hi), (j, hj)]) - at(&[(i, hi), (j, -hj)]) - at(&[(i, -hi), (j, hj)])
                        + at(&[(i, -hi), (j, -hj)]))
                        / (4.0 * hi * hj);
                h[(i, j)] = v;
                h[(j, i)] = v;
            }
        }
        h
    };
    let h1 = raw(base);
    let h2 = raw(0.5 * base);
    let mut out = Mat::<f64>::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            out[(i, j)] = (4.0 * h2[(i, j)] - h1[(i, j)]) / 3.0;
        }
    }
    out
}

/// The assembled joint Hessian against the hyper-dual one, entry by entry, at
/// the same converged γ̂, on every dense Laplace GLMM rung of the validation
/// corpus — the real datasets, lowered from the manifest's own formula through
/// the crate's formula frontend, not a stand-in — plus [`CLOGLOG_LARGE`], the
/// 9,600-row cloglog fixture whose mode state carries a μ-clamped row, run at
/// the same bands as everything else.
///
/// Two independent exact routes to the same matrix: one differentiates the
/// objective twice through packed second-order lanes, the other differentiates
/// an explicit `F`/`G` adjoint once and reads the second order off first-order
/// lanes. They share the `f64` mode solve and the PIRLS tolerance and nothing
/// else.
///
/// The run also reports, per rung: the assembled pass's own pre-symmetrization
/// asymmetry (`max|H_ij − H_ji|` relative, over columns built by different
/// chunks — a consistency check the packed pass cannot offer, its triangle
/// being symmetric by construction), and each pass's exit `‖u − u_prev‖`, which
/// is the size of the iterate mix each one differentiates. `mu_clamped` is a
/// reported count now, not a switch: every rung is compared the same way
/// whether or not any row is pinned, since a pinned row reads its deviance
/// slope, observed weight and `dw/dη` off closed forms instead of breaking
/// the comparison.
#[cfg(feature = "formula")]
#[test]
fn assembled_hessian_matches_hyperdual_per_entry() {
    // One band per comparison, both fixed from the first run on the real
    // datasets and neither tuned since. Gradient: every compared rung agrees to
    // 8.94e-9 or better, the worst being rung 23 (`sim_gamma`), and fifteen of
    // the sixteen are at or below 1.75e-11. Hessian: every compared rung agrees
    // to 6.11e-12 or better, the worst being rung 14 (`Arabidopsis`).
    const GRAD_BAND: f64 = 1e-7;
    const BAND: f64 = 1e-10;

    let mut rows: Vec<RungReport> = Vec::new();
    for r in DENSE_LAPLACE_RUNGS
        .iter()
        .chain(std::iter::once(&CLOGLOG_LARGE))
    {
        let (mut ws, x, y, ids, extra_ids, p, n, dev) = rung_at_gamma_hat(r);
        let m = ws.n_theta + p;
        let (mu_clamped, eta_clamped) = clamp_census(&ws, r.family, ws.weighted, n);
        assert_eq!(
            eta_clamped, 0,
            "rung {}: a clamped η at γ̂ is a new regime, not a band",
            r.rung
        );

        // Stage 1 on the real data: the `f64` assembled gradient against
        // `laplace_gradient`, per coordinate. Two ways of differentiating the
        // same objective once — an explicit `F`/`G` adjoint against the dual
        // kernel's own lanes — so they must agree to round-off, and a rung
        // where they do not is a rung where the Hessian comparison below is
        // measuring the gradient, not the second derivative.
        let mut g_dual = vec![0.0; m];
        let st = crate::glmm::laplace_gradient(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut g_dual,
        );
        assert!(
            matches!(st, crate::glmm::DerivStatus::Ok(_)),
            "rung {}: the dual gradient declined",
            r.rung
        );
        let mut g_f64 = vec![0.0; m];
        crate::glmm::gradient_f64(&mut ws, x.as_ref(), &y, &ids, &extra_ids, p, n, &mut g_f64)
            .unwrap_or_else(|| panic!("rung {}: the assembled f64 gradient declined", r.rung));
        let mut worst_grad = 0.0f64;
        for c in 0..m {
            let gap = (g_f64[c] - g_dual[c]).abs() / g_dual[c].abs().max(1.0);
            worst_grad = worst_grad.max(gap);
            assert!(
                gap <= GRAD_BAND,
                "rung {} coord {c}: assembled {} vs dual {} (relative gap {gap:e})",
                r.rung,
                g_f64[c],
                g_dual[c]
            );
        }

        let mut h_asm = Mat::<f64>::zeros(m, m);
        let mut g_asm = vec![0.0; m];
        let st = crate::glmm::joint_hessian_columns(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut g_asm,
            &mut h_asm,
        );
        let v_asm = match st {
            crate::glmm::DerivStatus::Ok(v) => v,
            _ => panic!("rung {}: the assembled Hessian declined", r.rung),
        };
        let step_asm = ws
            .dual_scratch
            .as_deref()
            .expect("the assembled pass sizes the dual scratch")
            .exit_mode_step();
        let mut asym = 0.0f64;
        for i in 0..m {
            for j in 0..i {
                asym =
                    asym.max((h_asm[(i, j)] - h_asm[(j, i)]).abs() / h_asm[(i, j)].abs().max(1.0));
                let v = 0.5 * (h_asm[(i, j)] + h_asm[(j, i)]);
                h_asm[(i, j)] = v;
                h_asm[(j, i)] = v;
            }
        }

        let mut h_hd = Mat::<f64>::zeros(m, m);
        let mut g_hd = vec![0.0; m];
        let st = crate::glmm::laplace_hessian(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut g_hd,
            &mut h_hd,
        );
        let v_hd = match st {
            crate::glmm::DerivStatus::Ok(v) => v,
            _ => panic!("rung {}: the hyper-dual Hessian declined", r.rung),
        };
        let step_hd = ws
            .hyper_scratch
            .as_deref()
            .expect("the hyper-dual pass sizes its own scratch")
            .exit_mode_step();

        // The objective value each pass reports is the same function at the
        // same point whichever chunk it came out of, and it is the fit's own
        // minimized deviance. Asserted in release, not only in debug: on a
        // chunked pass this is what says every chunk differentiated the same
        // objective.
        for (what, v) in [("assembled", v_asm), ("hyper-dual", v_hd)] {
            assert!(
                (v - dev).abs() <= 1e-6 * (1.0 + dev.abs()),
                "rung {}: {what} pass reports objective {v}, the fit's deviance is {dev}",
                r.rung
            );
        }

        let mut worst = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                let gap = (h_asm[(i, j)] - h_hd[(i, j)]).abs() / h_hd[(i, j)].abs().max(1.0);
                worst = worst.max(gap);
                assert!(
                    gap <= BAND,
                    "rung {} entry ({i},{j}): assembled {} vs hyper-dual {} (relative gap {gap:e})",
                    r.rung,
                    h_asm[(i, j)],
                    h_hd[(i, j)]
                );
            }
        }
        // The asymmetry compares the assembled pass with itself: columns built
        // by different chunks against each other.
        assert!(
            asym <= BAND,
            "rung {}: pre-symmetrization asymmetry {asym:e} above the band",
            r.rung
        );

        // `CLOGLOG_LARGE` only: a second, independent arbiter — the
        // Richardson-extrapolated central second difference of the f64
        // Laplace deviance, at base steps 4e-3 and 8e-3 — so the assembled
        // Hessian is checked against something other than the hyper-dual
        // pass too. `FD_BAND` is the arbiter's own reproducibility between
        // those two base steps, not a statement about either engine: nothing
        // tighter than the arbiter's own drift would be. Measured on this
        // tree: reproducibility (base 4e-3 vs 8e-3) 2.08e-7 over all entries,
        // 3.95e-8 on the diagonal; assembled vs the base-4e-3 estimate
        // 3.20e-7 worst entry — `FD_BAND` sits above that measured worst and
        // below twice it.
        if std::ptr::eq(r, &CLOGLOG_LARGE) {
            const FD_BAND: f64 = 4e-7;
            let n_theta = ws.n_theta;
            ws.fd.pirls_tol_override = Some(1e-12);
            let gamma_hat: Vec<f64> = ws.params[..m].to_vec();
            let fd4 = richardson_fd_hessian(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &extra_ids,
                n,
                m,
                n_theta,
                4e-3,
                &gamma_hat,
            );
            let fd8 = richardson_fd_hessian(
                &mut ws,
                x.as_ref(),
                &y,
                &ids,
                &extra_ids,
                n,
                m,
                n_theta,
                8e-3,
                &gamma_hat,
            );
            ws.fd.pirls_tol_override = None;
            let mut repro_all = 0.0f64;
            let mut repro_diag = 0.0f64;
            for i in 0..m {
                for j in 0..m {
                    let gap = (fd4[(i, j)] - fd8[(i, j)]).abs()
                        / fd4[(i, j)].abs().min(fd8[(i, j)].abs()).max(1.0);
                    repro_all = repro_all.max(gap);
                    if i == j {
                        repro_diag = repro_diag.max(gap);
                    }
                }
            }
            let mut worst_fd = 0.0f64;
            for i in 0..m {
                for j in 0..m {
                    let gap = (h_asm[(i, j)] - fd4[(i, j)]).abs()
                        / h_asm[(i, j)].abs().min(fd4[(i, j)].abs()).max(1.0);
                    worst_fd = worst_fd.max(gap);
                    assert!(
                        gap <= FD_BAND,
                        "CLOGLOG_LARGE entry ({i},{j}): assembled {} vs Richardson FD(4e-3) {} \
                         (relative gap {gap:e})",
                        h_asm[(i, j)],
                        fd4[(i, j)]
                    );
                }
            }
            println!(
                "CLOGLOG_LARGE: FD reproducibility (4e-3 vs 8e-3) all entries {repro_all:e}, \
                 diagonal {repro_diag:e}; assembled vs FD(4e-3) worst {worst_fd:e}"
            );
        }

        rows.push(RungReport {
            rung: r.rung,
            m,
            worst_grad,
            worst,
            asym,
            step_asm,
            step_hd,
            mu_clamped,
        });
    }
    assert_eq!(
        rows.len(),
        DENSE_LAPLACE_RUNGS.len() + 1,
        "every rung, plus CLOGLOG_LARGE, must run"
    );
    for r in &rows {
        println!(
            "rung {}: m {}, gradient {:e}, worst entry {:e}, asymmetry {:e}, \
             exit |u-u_prev| assembled {:e} hyper-dual {:e}, clamped rows {}",
            r.rung, r.m, r.worst_grad, r.worst, r.asym, r.step_asm, r.step_hd, r.mu_clamped
        );
    }
}

/// A small unweighted Bernoulli-logit design whose raw sigmoid sits at (or
/// past) `family::clamp_mu`'s upper bound on a few rows at γ̂, without any
/// kernel ever calling `clamp_mu` there — one primary grouping, no extras, so
/// the fit is dense (the blocked route), not packed. `y = 1` and a `+30`
/// offset on three rows out of eighty forces `σ(η) ≥ 1 − PROB_EPS` there while
/// leaving the rest of the design an ordinary random-intercept logit fit.
#[cfg(feature = "formula")]
#[allow(clippy::type_complexity)]
fn unweighted_logit_saturated_fixture() -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
) {
    let (n_g, per) = (8usize, 10usize);
    let n = n_g * per;
    let p = 2;
    let mut st = 11u64;
    let (mut x, mut y) = (vec![0.0f64; n * p], vec![0.0f64; n]);
    let mut g = vec![0u32; n];
    for i in 0..n {
        g[i] = (i % n_g) as u32;
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        let eta = 0.3 + 0.5 * x1 + 0.2 * (g[i] as f64 - 3.5);
        y[i] = if eta.exp() / (1.0 + eta.exp()) > 0.5 {
            1.0
        } else {
            0.0
        };
    }
    let mut offset = vec![0.0f64; n];
    for &i in &[0usize, 1, 2] {
        offset[i] = 30.0;
        y[i] = 1.0;
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_g as u32,
            },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: g,
        extra: vec![],
    };
    (x, y, offset, n, p, model, ids)
}

/// The unweighted-logit exemption, end to end. On [`unweighted_logit_saturated_fixture`]
/// the raw sigmoid sits at `family::clamp_mu`'s bound on a few rows while no
/// kernel ever calls `clamp_mu` there — `family::pinned_mu_bounds` (what
/// `assembled::mu_clamped_rows`, `pirls::clamped_row_present`, and the
/// per-row pinned test in `assemble`/`packed_assemble` all read) must report
/// those rows as ordinary, not pinned, so the assembled engine takes the fit
/// and its Hessian/gradient agree with the hyper-dual pass at the per-entry
/// gate's own bands — the one end-to-end check of that route.
#[cfg(feature = "formula")]
#[test]
fn unweighted_logit_saturated_rows_are_not_pinned() {
    const GRAD_BAND: f64 = 1e-7;
    const BAND: f64 = 1e-10;

    let (xf, y, offset, n, p, model, ids) = unweighted_logit_saturated_fixture();
    let (mut ws, x, ids_p, extra_ids) = ws_at_gamma_hat(
        &xf,
        &y,
        Some(offset),
        n,
        p,
        &model,
        &ids,
        "unweighted logit saturated fixture",
    );
    assert!(!ws.weighted, "fixture must be unweighted");
    let family = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let (mu_lo, mu_hi) = crate::family::clamp_mu_bounds(family);
    let raw_saturated = ws.pirls.prob[..n]
        .iter()
        .filter(|&&mu| mu <= mu_lo || mu >= mu_hi)
        .count();
    assert!(
        raw_saturated > 0,
        "fixture must reach the raw clamp_mu bound"
    );
    assert_eq!(
        crate::glmm::mu_clamped_rows(family, false, &ws.pirls.prob[..n]),
        0,
        "unweighted logit's census must stay 0 despite the raw-bound rows"
    );

    let m = ws.n_theta + p;
    let mut g_dual = vec![0.0; m];
    let st = crate::glmm::laplace_gradient(
        &mut ws,
        x.as_ref(),
        &y,
        &ids_p,
        &extra_ids,
        p,
        n,
        &mut g_dual,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "dual gradient declined"
    );
    let mut g_f64 = vec![0.0; m];
    crate::glmm::gradient_f64(
        &mut ws,
        x.as_ref(),
        &y,
        &ids_p,
        &extra_ids,
        p,
        n,
        &mut g_f64,
    )
    .expect("assembled f64 gradient declined");
    for c in 0..m {
        let gap = (g_f64[c] - g_dual[c]).abs() / g_dual[c].abs().max(1.0);
        assert!(
            gap <= GRAD_BAND,
            "coord {c}: assembled {} vs dual {} (relative gap {gap:e})",
            g_f64[c],
            g_dual[c]
        );
    }

    let mut h_asm = Mat::<f64>::zeros(m, m);
    let mut g_asm = vec![0.0; m];
    let st = crate::glmm::joint_hessian_columns(
        &mut ws,
        x.as_ref(),
        &y,
        &ids_p,
        &extra_ids,
        p,
        n,
        &mut g_asm,
        &mut h_asm,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "assembled Hessian declined"
    );
    let mut h_hd = Mat::<f64>::zeros(m, m);
    let mut g_hd = vec![0.0; m];
    let st = crate::glmm::laplace_hessian(
        &mut ws,
        x.as_ref(),
        &y,
        &ids_p,
        &extra_ids,
        p,
        n,
        &mut g_hd,
        &mut h_hd,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "hyper-dual Hessian declined"
    );
    for i in 0..m {
        for j in 0..m {
            let gap = (h_asm[(i, j)] - h_hd[(i, j)]).abs() / h_hd[(i, j)].abs().max(1.0);
            assert!(
                gap <= BAND,
                "entry ({i},{j}): assembled {} vs hyper-dual {} (relative gap {gap:e})",
                h_asm[(i, j)],
                h_hd[(i, j)]
            );
        }
    }
}

/// Rung 48's dataset read under the cloglog link — the 9,600-row model
/// `fit_glmm_cloglog_matches_lme4` fits, expressed as a rung so the harness
/// above can drive it to its own γ̂. Not in `DENSE_LAPLACE_RUNGS`: the corpus
/// gate's set is the manifest's, one entry per dataset and link, and this is a
/// second link on a dataset already in it.
#[cfg(feature = "formula")]
static CLOGLOG_LARGE: DenseLaplaceRung = DenseLaplaceRung {
    rung: 48,
    csv: include_str!("../../validation/data/simulated/sim_probit_large.csv"),
    formula: "y ~ x1 + x2 + x3 + z + (1 | g)",
    family: Family::Binomial {
        link: BinomialLink::Cloglog,
    },
    factors: &["g"],
    agg: None,
    weights_col: None,
    offset_col: None,
};

/// The corpus rung carrying this number.
#[cfg(feature = "formula")]
fn rung_by_number(rung: u32) -> &'static DenseLaplaceRung {
    DENSE_LAPLACE_RUNGS
        .iter()
        .find(|r| r.rung == rung)
        .unwrap_or_else(|| panic!("rung {rung} is in the corpus"))
}

/// One fixture of [`laplace_gradient_lanes_settle_on_a_clamped_mode_state`]:
/// the point to drive, how many clamped rows its mode state is expected to
/// carry, whether the dual kernel should still be reporting one-call
/// exactness there, and the band the lanes must meet.
#[cfg(feature = "formula")]
struct LaneFixture {
    what: &'static str,
    rung: &'static DenseLaplaceRung,
    mu_clamped: usize,
    exact: bool,
    band: f64,
}

/// The dual Laplace gradient's lanes against a Richardson-extrapolated central
/// difference of the very objective they claim to differentiate, on the one
/// point whose mode state carries a clamped row and on three that do not.
///
/// **What can go wrong here.** The dual PIRLS kernels take the Hessian step
/// (`pirls::DualStep`), so the lanes normally reach the implicit-function
/// answer in one kernel call and the kernels say so through `DualStep::exact`;
/// the caller then skips its refinement loop. On a row sitting on one of the
/// kernel's clamps the step matrix is no longer the Jacobian of the map the
/// iteration walks (`pirls::clamped_row_present` carries the derivation), the
/// contraction is not zero, and lanes read after one call are part-converged —
/// wrong in a way nothing downstream can see, because they are smooth,
/// plausible, and only a few digits short.
///
/// So both halves are asserted. On the clamped point the lanes must meet the
/// band a settled kernel reaches, which is far tighter than a single call
/// gets there. On the three clean points `exact` must still be true, so the
/// refinement loop stays off where it costs and buys nothing.
///
/// The arbiter is the `f64` objective's own central difference, Richardson
/// extrapolated from base `h` and `h/2`, with `ws.pirls.u` zeroed before every
/// evaluation so each one is the same cold function of γ. It is an independent
/// route to the gradient — no dual arithmetic, no adjoint — and its own
/// resolution is what sets the bands, not taste.
#[cfg(feature = "formula")]
#[test]
fn laplace_gradient_lanes_settle_on_a_clamped_mode_state() {
    // Tight enough that the mode solve's last step is at round-off, so the
    // differentiated objective and the differenced one are the same function
    // rather than two nearby ones.
    const TOL: f64 = 1e-12;
    // Base FD step: absolute on θ, relative × max(|β|, 1) on β.
    const FD_BASE: f64 = 1e-3;
    // The two bands, both set from the first run on these four points and
    // neither tuned since. `SMALL_BAND` covers the two few-hundred-row points,
    // rung 49 and rung 22, where the arbiter resolves the gradient to ~5e-10
    // and the lanes land at 7.6e-12 to 4.6e-10. `BIG_BAND` covers the two
    // 9,600-row ones, where the arbiter's own drift between base steps is up
    // to 6.4e-8 — nothing tighter than the arbiter's resolution is a
    // statement about the lanes. There the lanes land at 1.8e-9 to 5.5e-8 on
    // rung 48 and, settled by the refinement loop, at 9.5e-9 to 1.3e-7 on the
    // clamped cloglog fixture.
    const SMALL_BAND: f64 = 1e-8;
    const BIG_BAND: f64 = 2e-7;

    let fixtures = [
        LaneFixture {
            what: "rung 49 sim_cloglog_nested_crossed",
            rung: rung_by_number(49),
            mu_clamped: 0,
            exact: true,
            band: SMALL_BAND,
        },
        LaneFixture {
            what: "cloglog 9,600-row fixture",
            rung: &CLOGLOG_LARGE,
            mu_clamped: 3,
            exact: false,
            band: BIG_BAND,
        },
        LaneFixture {
            what: "rung 22 cbpp_probit",
            rung: rung_by_number(22),
            mu_clamped: 0,
            exact: true,
            band: SMALL_BAND,
        },
        LaneFixture {
            what: "rung 48 sim_probit_large",
            rung: rung_by_number(48),
            mu_clamped: 0,
            exact: true,
            band: BIG_BAND,
        },
    ];

    for f in &fixtures {
        let (mut ws, x, y, ids, extra_ids, p, n, _dev) = rung_at_gamma_hat(f.rung);
        let m = ws.n_theta + p;
        let n_theta = ws.n_theta;
        let (mu_clamped, eta_clamped) = clamp_census(&ws, f.rung.family, ws.weighted, n);
        assert_eq!(
            (mu_clamped, eta_clamped),
            (f.mu_clamped, 0),
            "{}: the clamp census at γ̂ moved — this fixture is here for its clamp state",
            f.what
        );

        ws.fd.pirls_tol_override = Some(TOL);
        let mut g_dual = vec![0.0; m];
        let st = crate::glmm::laplace_gradient(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut g_dual,
        );
        assert!(
            matches!(st, crate::glmm::DerivStatus::Ok(_)),
            "{}: the dual gradient declined",
            f.what
        );
        assert_eq!(
            ws.dual_scratch
                .as_deref()
                .expect("the dual gradient sizes the dual scratch")
                .exit_exact(),
            f.exact,
            "{}: one-call exactness is not what this mode state supports",
            f.what
        );

        let params: Vec<f64> = ws.params[..m].to_vec();
        let at = |ws: &mut GlmmWorkspace, k: usize, step: f64| -> f64 {
            let mut q = params.clone();
            q[k] += step;
            ws.pirls.u.fill(0.0);
            glmm_laplace_deviance(&q, ws, x.as_ref(), &y, &ids, &extra_ids, n)
        };
        let mut worst = 0.0f64;
        let mut gaps = vec![0.0; m];
        for k in 0..m {
            let h = if k < n_theta {
                FD_BASE
            } else {
                FD_BASE * params[k].abs().max(1.0)
            };
            // Central difference at h and at h/2, combined as
            // (4·D_{h/2} − D_h)/3 — the O(h⁴) Richardson step for a stencil
            // whose own error is O(h²).
            let c1 = (at(&mut ws, k, h) - at(&mut ws, k, -h)) / (2.0 * h);
            let c2 = (at(&mut ws, k, 0.5 * h) - at(&mut ws, k, -0.5 * h)) / h;
            let fd = (4.0 * c2 - c1) / 3.0;
            gaps[k] = (g_dual[k] - fd).abs();
            worst = worst.max(gaps[k]);
        }
        println!(
            "{}: m {}, clamped rows (μ {mu_clamped}), exact {}, \
             per-coordinate |lane − FD| {:?}, worst {:e}",
            f.what, m, f.exact, gaps, worst
        );
        for k in 0..m {
            assert!(
                gaps[k] <= f.band,
                "{} coord {k}: lane {} vs central difference (absolute gap {:e}, band {:e})",
                f.what,
                g_dual[k],
                gaps[k],
                f.band
            );
        }
    }
}

/// The dense assembled path on a clamped mode state, end to end: on a rung
/// whose mode state carries a clamped row the assembled engine takes the
/// fit, and `joint_hessian_cov` ships its covariance for it.
///
/// `CLOGLOG_LARGE`, the cloglog fixture on the 9,600-row `sim_probit_large`
/// dataset, is that rung — three μ-clamped rows — and the corpus gate above
/// already holds the per-entry agreement with the hyper-dual pass. What is
/// unexercised without this test is the RESULT through the production entry
/// point: that `joint_hessian_cov` reports `FdHessianStatus::Ok`, and that
/// the covariance and the θ-block standard errors it ships are the assembled
/// arm's own, bit for bit, rather than the hyper-dual pass's or a stencil's.
///
/// The reference side drives `joint_hessian` directly at the same γ̂ and
/// inverts its matrix the way `joint_hessian_cov` inverts it — `cov = 2·H⁻¹`
/// on the β block, `sqrt(2·H⁻¹_kk)` on the θ diagonal — so the comparison is
/// bitwise and needs no band.
///
/// The take is asserted at the engine, not through
/// `assembled::ASSEMBLED_OK_COUNT`: that counter is process-wide and any
/// concurrently-running test's `WaldSe::Hessian` fit advances it too, so it
/// cannot say this call in particular is what advanced it.
#[cfg(feature = "formula")]
#[test]
fn clamped_dense_rung_ships_the_assembled_covariance() {
    use faer::linalg::solvers::Solve;

    let r = &CLOGLOG_LARGE;
    let (mut ws, x, y, ids, extra_ids, p, n, _dev) = rung_at_gamma_hat(r);
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let (mu_clamped, _eta) = clamp_census(&ws, r.family, ws.weighted, n);
    assert!(
        mu_clamped > 0,
        "rung {}: this fixture is here for its clamped mode state (μ {mu_clamped})",
        r.rung
    );

    let mut g = vec![0.0; m];
    let mut hess = Mat::<f64>::zeros(m, m);
    let st = crate::glmm::joint_hessian_columns(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut g,
        &mut hess,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "rung {}: the assembled engine must take a clamped mode state now",
        r.rung
    );

    // The assembled arm on its own, symmetrized, at the same γ̂ — the
    // reference the production entry point below is compared against.
    let st = crate::glmm::joint_hessian(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut g,
        &mut hess,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "rung {}: the assembled arm must answer",
        r.rung
    );
    let chol = hess
        .as_ref()
        .llt(faer::Side::Lower)
        .expect("the assembled joint Hessian is PD at this γ̂");
    let mut inv = Mat::<f64>::identity(m, m);
    chol.solve_in_place(inv.as_mut());
    let want_cov = Mat::<f64>::from_fn(p, p, |a, b| 2.0 * inv[(n_theta + a, n_theta + b)]);
    let want_theta_se: Vec<f64> = (0..n_theta)
        .map(|k| (2.0 * inv[(k, k)]).max(0.0).sqrt())
        .collect();

    let mut got_cov = Mat::<f64>::zeros(p, p);
    let st = crate::glmm::joint_hessian_cov(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut got_cov,
    );
    assert!(
        matches!(st, crate::glmm::FdHessianStatus::Ok),
        "rung {}: expected Ok ({st:?})",
        r.rung
    );
    for a in 0..p {
        for b in 0..p {
            assert_eq!(
                got_cov[(a, b)].to_bits(),
                want_cov[(a, b)].to_bits(),
                "rung {}: cov[({a},{b})] {} is not the assembled arm's {}",
                r.rung,
                got_cov[(a, b)],
                want_cov[(a, b)]
            );
        }
    }
    for (k, &want) in want_theta_se.iter().enumerate() {
        assert_eq!(
            ws.inference.theta_se[k].to_bits(),
            want.to_bits(),
            "rung {}: theta_se[{k}] {} is not the assembled arm's {want}",
            r.rung,
            ws.inference.theta_se[k]
        );
    }
}

/// One timed rep of a rung, construction-inclusive: a fresh lowering plus one
/// public `fit_cold` call at the given [`WaldSe`], timed end to end — the same
/// shape of measurement as the bit-identity dump's `_full` fields
/// (`validation/summarize_timing.R:47-59`), which prefer the wall that
/// includes formula lowering and workspace construction over a fit-only
/// wall. Re-lowering every rep (rather than lowering once and timing only
/// `fit_cold`) is what makes each rep independent and comparable to a real
/// caller's cold entry.
#[cfg(feature = "formula")]
fn timed_rung_fit(r: &DenseLaplaceRung, wald_se: WaldSe) -> (std::time::Duration, Fit) {
    let (table, weights, offset) = rung_table(r);
    let lo = crate::formula::lower(r.formula, &table, r.family)
        .unwrap_or_else(|e| panic!("rung {}: lower: {e}", r.rung));
    let opts = FitOptions {
        target_indices: lo.opts.target_indices.clone(),
        wald_se,
        weights: weights.or(lo.opts.weights.clone()),
        offset: offset.or(lo.opts.offset.clone()),
        nagq: lo.opts.nagq,
        ..FitOptions::default()
    };
    let t0 = std::time::Instant::now();
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &opts);
    (t0.elapsed(), f)
}

/// `REPS` runs of [`timed_rung_fit`] at one `wald_se`, first discarded: returns
/// the min elapsed across the kept reps, the LAST rep's `Fit` (every rep fits
/// the same deterministic problem, so any rep's diagnostics represent the
/// series), and how many of the `reps` runs converged.
#[cfg(feature = "formula")]
fn timed_series(
    r: &DenseLaplaceRung,
    wald_se: WaldSe,
    reps: usize,
) -> (std::time::Duration, Fit, usize) {
    assert!(
        reps >= 2,
        "REPS must be >= 2 so the first rep can be discarded"
    );
    let mut best: Option<std::time::Duration> = None;
    let mut last: Option<Fit> = None;
    let mut n_converged = 0usize;
    for rep in 0..reps {
        let (elapsed, f) = timed_rung_fit(r, wald_se);
        if f.converged() {
            n_converged += 1;
        }
        if rep > 0 {
            best = Some(best.map_or(elapsed, |b| b.min(elapsed)));
        }
        last = Some(f);
    }
    (best.unwrap(), last.unwrap(), n_converged)
}

/// The Hessian-inclusive fit wall against the Rx-only fit wall, paired within
/// one session, on grouseticks (rung 6) and VerbAgg (rung 12) — the large-N,
/// complex-model cells the speed comparison is judged on. Two arms for the
/// Hessian wall, on the same binary: the
/// assembled pass as the tree stands, and the hyper-dual pass reached by
/// forcing the assembled pass to decline through the test-only
/// `assembled::FORCE_DECLINE` switch — never a `FitOptions` flag. Prints one
/// table with both arms' wall, the ratio `hess/rx − 1`, and the convergence
/// axes (`converged`, `n_eval`, `deviance`) alongside `ASSEMBLED_OK_COUNT`'s
/// movement, which is this crate's only signal that a given arm actually took
/// its intended path (`Diagnostics` carries no exact-vs-fallback field).
///
/// Asserts only that both arms converge and report finite SEs — this is a
/// measurement driver, not a speed gate; the numbers it prints are read by a
/// human against the locked-run report.
#[cfg(feature = "formula")]
#[test]
#[ignore]
fn assembled_vs_hyperdual_paired_timing() {
    use std::sync::atomic::Ordering;

    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);

    struct Row {
        cell: &'static str,
        arm: &'static str,
        rx_wall: std::time::Duration,
        hess_wall: std::time::Duration,
        n_converged: usize,
        reps: usize,
        f: Fit,
        assembled_delta: usize,
    }
    let mut rows: Vec<Row> = Vec::new();

    for (cell, rung) in [("grouseticks", 6u32), ("VerbAgg", 12u32)] {
        let r = DENSE_LAPLACE_RUNGS.iter().find(|r| r.rung == rung).unwrap();

        // Rx wall does not touch the joint-Hessian machinery at all — one
        // series serves both arms' ratio.
        let (rx_wall, rx_fit, rx_converged) = timed_series(r, WaldSe::Rx, reps);
        assert!(rx_fit.converged(), "{cell}: Rx-only fit must converge");

        for (arm, force_decline) in [("assembled", false), ("hyper-dual", true)] {
            crate::glmm::FORCE_DECLINE.store(force_decline, Ordering::Relaxed);
            let before = crate::glmm::ASSEMBLED_OK_COUNT.load(Ordering::Relaxed);
            let (hess_wall, hess_fit, hess_converged) = timed_series(r, WaldSe::Hessian, reps);
            let after = crate::glmm::ASSEMBLED_OK_COUNT.load(Ordering::Relaxed);
            crate::glmm::FORCE_DECLINE.store(false, Ordering::Relaxed);

            assert!(
                hess_fit.converged(),
                "{cell} {arm}: Hessian fit must converge"
            );
            assert!(
                hess_fit.se.iter().all(|v| v.is_finite() || v.is_nan()),
                "{cell} {arm}: se must be finite or NaN, never garbage"
            );
            assert!(
                hess_fit.se.iter().any(|v| v.is_finite()),
                "{cell} {arm}: at least one target SE must be finite"
            );
            // The construction of the switch guarantees this by itself for
            // the forced arm; assert it anyway so a future edit that breaks
            // the routing fails here rather than silently mismeasuring.
            if force_decline {
                assert_eq!(
                    after, before,
                    "{cell} {arm}: FORCE_DECLINE must keep the assembled arm from running"
                );
            } else {
                assert!(
                    after > before,
                    "{cell} {arm}: the assembled arm must actually run when not forced off"
                );
            }

            rows.push(Row {
                cell,
                arm,
                rx_wall,
                hess_wall,
                n_converged: hess_converged.min(rx_converged),
                reps,
                f: hess_fit,
                assembled_delta: after - before,
            });
        }
    }

    // The driver does not read the machine's clock state: a run is a
    // measurement only when the caller locked the CPU clock first.
    println!("paired timing, REPS={reps} (first rep discarded, min of the rest); lock state not read here.");
    println!(
        "{:<12} {:<11} {:>12} {:>12} {:>9} {:>10} {:>9} {:>13} {:>10}",
        "cell",
        "arm",
        "rx_wall_s",
        "hess_wall_s",
        "ratio",
        "converged",
        "n_eval",
        "deviance",
        "asm_delta"
    );
    for row in &rows {
        let ratio = row.hess_wall.as_secs_f64() / row.rx_wall.as_secs_f64() - 1.0;
        println!(
            "{:<12} {:<11} {:>12.6} {:>12.6} {:>9.4} {:>8}/{:<2}{:>9} {:>13.4} {:>10}",
            row.cell,
            row.arm,
            row.rx_wall.as_secs_f64(),
            row.hess_wall.as_secs_f64(),
            ratio,
            row.n_converged,
            row.reps,
            row.f.n_eval,
            row.f.deviance,
            row.assembled_delta,
        );
    }
}

/// How much the `f64` assembled gradient (the Laplace objective's explicit
/// `F`/`G` adjoint that `src/glmm/assembled.rs` builds) moves between the
/// PIRLS exit band the fit ships and a band shrunk until the penalized
/// deviance stops changing, alongside the exact mode residual `‖G_u‖`,
/// `G_u = D_u + 2u` (`assembled.rs`'s own `G(γ,u)`), at each band — the
/// quantity every formula in that construction assumes is zero, and the one
/// the assembled gradient's own error is first order in.
///
/// `sim_sparse_gamma` is a sparse-route rung: `gradient_f64` only reaches the
/// dense blocked and structured routes (`assembly_routes`), so it cannot
/// evaluate there at all. `sim_gamma` (rung 23, Gamma-log, dense) is the
/// nearest dense Gamma rung and stands in for it; `sim_poisson_nested` (rung
/// 19) is the canonical-link rung run alongside it.
///
/// The reference band is picked the way `fd_margin.rs` picks its own FD
/// reference: shrink `pirls_tol_override` down the same seven-value exit-band
/// ladder that module tries (`sparse/fd_margin.rs`'s `REF_TOL_LADDER`), and
/// gate the choice on step-freeness exactly as that module hard-gates
/// `pick_ref_tol` on its own precondition rather than merely reporting it —
/// the tightest finite rung is taken only once it agrees with its
/// next-loosest neighbor to within `STEP_FREE_BAND`, falling back down the
/// ladder when it does not, and panicking if no neighboring pair ever agrees.
/// `fd_margin.rs` itself scans ±3δ in γ because it is protecting an FD
/// stencil built by perturbing γ; this probe evaluates the gradient at one
/// fixed γ̂ with no such stencil, so neighboring-rung agreement in tolerance
/// space is the applicable form of the same check.
#[cfg(feature = "formula")]
#[test]
#[ignore]
fn assembled_gradient_mode_residual_probe() {
    // The same seven values `sparse/fd_margin.rs`'s `REF_TOL_LADDER` tries, in
    // the same order, tightest last.
    const REF_TOL_LADDER: [f64; 7] = [0.0, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10];
    // "Near round-off" for a penalized deviance summed over up to a few
    // thousand rows: comfortably above bit-level noise, comfortably below the
    // FD stencil's own 2.0e-5 margin this probe's result is judged against.
    const STEP_FREE_BAND: f64 = 1e-9;

    println!("sim_gamma stands in for sim_sparse_gamma, which the dense assembly cannot take.");

    for rung in [23u32, 19] {
        let r = DENSE_LAPLACE_RUNGS.iter().find(|r| r.rung == rung).unwrap();
        let (mut ws, x, y, ids, extra_ids, p, n, _dev) = rung_at_gamma_hat(r);
        let m = ws.n_theta + p;

        // (a) production band: pirls_tol_override unset, so gradient_f64 falls
        // back to pirls_tol_fd(family) (glmm/mod.rs:173), PIRLS_TOL_REL_FD =
        // 1e-8 unless the family's own fit tolerance is tighter.
        ws.fd.pirls_tol_override = None;
        let prod_tol = crate::glmm::pirls_tol_fd(r.family);
        let mut g_prod = vec![0.0; m];
        let resid_prod = crate::glmm::gradient_f64_mode_residual(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut g_prod,
        )
        .unwrap_or_else(|| panic!("rung {}: production-band gradient declined", r.rung));

        // (b) reference band: shrink pirls_tol_override along the ladder,
        // reading the penalized deviance glmm_laplace_deviance leaves at γ̂,
        // until it stops moving.
        let params: Vec<f64> = ws.params[..m].to_vec();
        let mut ladder: Vec<(f64, f64)> = Vec::new();
        for &tol in &REF_TOL_LADDER {
            ws.fd.pirls_tol_override = Some(tol);
            let d = glmm_laplace_deviance(&params, &mut ws, x.as_ref(), &y, &ids, &extra_ids, n);
            if d.is_finite() {
                ladder.push((tol, d));
            }
        }
        assert!(
            !ladder.is_empty(),
            "rung {}: no ladder rung gave a finite penalized deviance at γ̂",
            r.rung
        );
        // Climb down from the tightest finite rung until neighboring rungs
        // agree inside STEP_FREE_BAND — mirrors fd_margin.rs's pick_ref_tol
        // hard-gating on its own precondition instead of merely reporting it.
        let mut ref_idx = ladder.len() - 1;
        let step_free_gap = loop {
            assert!(
                ref_idx > 0,
                "rung {}: no ladder rung neighbor pair agrees within {STEP_FREE_BAND:e} \
                 — the reference band cannot be established",
                r.rung
            );
            let (_, dev_here) = ladder[ref_idx];
            let (_, dev_prev) = ladder[ref_idx - 1];
            let gap = (dev_here - dev_prev).abs() / dev_here.abs().max(1.0);
            if gap <= STEP_FREE_BAND {
                break gap;
            }
            ref_idx -= 1;
        };
        let (ref_tol, _) = ladder[ref_idx];

        ws.fd.pirls_tol_override = Some(ref_tol);
        let mut g_ref = vec![0.0; m];
        let resid_ref = crate::glmm::gradient_f64_mode_residual(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut g_ref,
        )
        .unwrap_or_else(|| panic!("rung {}: reference-band gradient declined", r.rung));

        let gaps: Vec<f64> = (0..m)
            .map(|c| (g_prod[c] - g_ref[c]).abs() / g_ref[c].abs().max(1.0))
            .collect();
        let worst_gap = gaps.iter().copied().fold(0.0, f64::max);

        println!(
            "rung {}: m {}, production tol {:e} (‖G_u‖ {:e}), reference tol {:e} \
             (‖G_u‖ {:e}, ladder rungs {}, step-free gap {:e}), \
             per-coordinate relative gradient gap {:?}, worst {:e}",
            r.rung,
            m,
            prod_tol,
            resid_prod,
            ref_tol,
            resid_ref,
            ladder.len(),
            step_free_gap,
            gaps,
            worst_gap
        );
    }
}

/// Every packed-row-routed GLMM rung of the validation corpus this instrument
/// can drive, at Laplace — rungs 8, 9, 18, 24, 38 and 46, the manifest's
/// `Solver::Sparse` rows that are also non-Gaussian and so reach the GLMM
/// driver rather than the LMM one. Same two mechanical formula rewrites
/// [`DenseLaplaceRung`] documents.
///
/// The manifest's other three `Solver::Sparse` rungs (7, 36, 47) are excluded,
/// not proxied: they are Gaussian and take the LMM REML path, so no GLMM
/// engine touches them.
///
/// One packed cell is covered by proxy instead of appearing here. The
/// negative-binomial golden `sim_sparse_nb` is packed-routed and in scope,
/// but `rung_at_gamma_hat` cannot reach it: an NB fit carries a trailing
/// `ln θ_NB` coordinate that `run_glmm_on` expects seeded from a no-RE GLM-NB
/// θ̂, which only the NB dispatch (`fit::glmm::fit_glmm_nb`) supplies, and
/// with `nb_theta` NaN the outer search rejects its own box. Its packed
/// engine is exercised through the full dispatch instead — by the
/// `NegativeBinomial` cell of `sparse::tests`'s cross-engine envelope check,
/// and by `sparse::tests::fit_sparse_nb_glmm_is_pinned`.
///
/// Every rung here is unweighted or carries its prior weights through an
/// aggregated-binomial response (rungs 8, 18, 38), all three canonical, so
/// prior weights on a NON-canonical packed fit are not gated from this list.
/// That combination is covered by the weighted Gamma-log cell of
/// `sparse::tests::packed_and_dense_assembled_hessians_agree`, against the
/// dense assembled engine, and corroborated by
/// `sparse::tests::fit_sparse_gamma_glmm_weighted_matches_lme4`.
#[cfg(feature = "formula")]
const PACKED_LAPLACE_RUNGS: &[DenseLaplaceRung] = &[
    DenseLaplaceRung {
        rung: 8,
        csv: include_str!("../../validation/data/simulated/sim_sparse_binomial.csv"),
        formula: "prop ~ x + (1|g1) + (1|c1) + (1|c2) + (1|c3) + (1|c4) + (1|c5) + (1|c6) + (1|c7)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["g1", "c1", "c2", "c3", "c4", "c5", "c6", "c7"],
        agg: Some(("incidence", "size")),
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 9,
        csv: include_str!("../../validation/data/simulated/sim_sparse_poisson.csv"),
        formula: "y ~ x + (1|g1) + (1|c1) + (1|c2) + (1|c3) + (1|c4) + (1|c5) + (1|c6) + (1|c7)",
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        factors: &["g1", "c1", "c2", "c3", "c4", "c5", "c6", "c7"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 18,
        csv: include_str!("../../validation/data/simulated/sim_binomial_slope_crossed.csv"),
        formula: "prop ~ x + (1 + x | g1) + (1 + x | g2)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["g1", "g2"],
        agg: Some(("incidence", "size")),
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 24,
        csv: include_str!("../../validation/data/simulated/sim_sparse_gamma.csv"),
        formula: "y ~ x1 + x2 + x3 + x4 + (1 | gp) + (1 + x1 + x2 + x3 + x4 | ge)",
        family: Family::Gamma {
            link: crate::GammaLink::Log,
        },
        factors: &["gp", "ge"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 38,
        csv: include_str!("../../validation/data/simulated/glmm_binomial.csv"),
        formula: "prop ~ x + (1|g1) + (1|c1) + (1|c2) + (1|c3) + (1|c4) + (1|c5) + (1|c6) + (1|c7)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["g1", "c1", "c2", "c3", "c4", "c5", "c6", "c7"],
        agg: Some(("incidence", "size")),
        weights_col: None,
        offset_col: None,
    },
    DenseLaplaceRung {
        rung: 46,
        csv: include_str!("../../validation/data/simulated/sim_sparse_binomial_bigsd.csv"),
        formula:
            "y ~ x + z + (1 | g1) + (1 | c1) + (1 | c2) + (1 | c3) + (1 | c4) + (1 | c5) + (1 | c6) + (1 | c7)",
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        factors: &["g1", "c1", "c2", "c3", "c4", "c5", "c6", "c7"],
        agg: None,
        weights_col: None,
        offset_col: None,
    },
];

/// The packed assembled gradient at `T = f64` against a Richardson-
/// extrapolated central difference of the Laplace deviance it differentiates,
/// at every packed rung's own γ̂ on the corpus's own data.
///
/// The oracle differences `glmm_laplace_deviance` on the same workspace with
/// `ws.fd.pirls_tol_override = Some(1e-12)`, which is where the packed arm of
/// the deviance cold-seeds `û = 0` on every evaluation, so each `f(γ)` is a
/// pure function of γ whatever order the coordinates run in. The engine's own
/// mode solve warm-starts from the fit's `û(γ̂)` instead; at that tolerance
/// both reach the same mode, so the two sides differentiate one function.
///
/// Central differences at `h` and `h/2`, combined as `(4·D(h/2) − D(h))/3`.
/// That cancels the plain central difference's `O(h²)` truncation and leaves
/// the deviance's own reproducibility divided by `h` as the floor, so the
/// step is deliberately LARGE: `h = 1e-3·max(1, |γ̂_k|)`.
///
/// Band `1e-6` relative (to `max(1, |FD|)`), fixed from the first
/// measurement: worst coordinate 2.75e-8, rung 46
/// (`sim_sparse_binomial_bigsd`, coordinate 1), with rung 24
/// (`sim_sparse_gamma`) next at 2.57e-8 and every other rung at or below
/// 2.0e-9. The ~36× margin over the worst is the FD oracle's own scatter,
/// not a statement about the engine: the engine's own accuracy is pinned
/// against exact references — the Hessian supply check below, and the
/// both-layouts Hessian cross-check in `sparse::tests`, which puts this
/// engine within 2.6e-13 of the dense assembled one on the shapes the dense
/// engine can also fit (`q_p ≤ 2`, `q_g = 1`, weighted and unweighted).
#[cfg(feature = "formula")]
#[test]
fn packed_assembled_gradient_matches_richardson_fd() {
    const BAND: f64 = 1e-6;
    for r in PACKED_LAPLACE_RUNGS {
        let (mut ws, x, y, ids, extra_ids, p, n, _dev) = rung_at_gamma_hat(r);
        assert_eq!(
            ws.layout,
            crate::glmm::GlmmLayout::Packed,
            "rung {}: packed-row layout expected",
            r.rung
        );
        let n_theta = ws.n_theta;
        let m = n_theta + p;
        let kk = ws.k.max(1);
        let saved: Vec<f64> = ws.params[..m].to_vec();
        let u_hat: Vec<f64> = ws.pirls.u[..kk].to_vec();
        ws.fd.pirls_tol_override = Some(1e-12);

        let mut g = vec![0.0; m];
        ws.pirls.u[..kk].copy_from_slice(&u_hat);
        let st = crate::glmm::packed_gradient(&mut ws, x.as_ref(), &y, p, n, &mut g);
        assert!(
            matches!(st, crate::glmm::DerivStatus::Ok(_)),
            "rung {}: the packed assembled gradient declined",
            r.rung
        );

        let mut worst = 0.0f64;
        let mut worst_coord = 0usize;
        for coord in 0..m {
            let h = 1e-3 * saved[coord].abs().max(1.0);
            let mut at = saved.clone();
            let ev = |ws: &mut GlmmWorkspace, at: &mut Vec<f64>, d: f64| -> f64 {
                at[coord] = saved[coord] + d;
                glmm_laplace_deviance(at, ws, x.as_ref(), &y, &ids, &extra_ids, n)
            };
            let d1 = (ev(&mut ws, &mut at, h) - ev(&mut ws, &mut at, -h)) / (2.0 * h);
            let d2 = (ev(&mut ws, &mut at, 0.5 * h) - ev(&mut ws, &mut at, -0.5 * h)) / h;
            let fd = (4.0 * d2 - d1) / 3.0;
            let gap = (g[coord] - fd).abs() / fd.abs().max(1.0);
            if gap > worst {
                worst = gap;
                worst_coord = coord;
            }
            assert!(
                gap <= BAND,
                "rung {} coord {coord}: assembled {} vs Richardson FD {fd} (relative gap {gap:e})",
                r.rung,
                g[coord]
            );
        }
        ws.params[..m].copy_from_slice(&saved);
        ws.fd.pirls_tol_override = None;
        println!(
            "packed gradient rung {}: m {m}, k {}, worst relative gap {worst:e} at coord {worst_coord}",
            r.rung, ws.k
        );
    }
}

/// The packed assembled Hessian's columns against a central difference of the
/// packed assembled gradient at `T = f64` — the supply check that every
/// quantity the assembly reads is differentiated rather than lifted, on the
/// corpus's own data at each packed rung's γ̂.
///
/// Both sides come from one body at two scalars, so a lane-plumbing mistake —
/// a quantity carried in with `from_f64` that should have been differentiated,
/// `û`'s response `U` solved against the wrong factor — shows here as a whole
/// column missing. The pre-symmetrization asymmetry is checked alongside: the
/// two triangles come out of different chunks, so their agreement is a free
/// consistency check on the chunk bookkeeping.
///
/// Richardson-extrapolated over `h` and `h/2`, as the gradient gate above is:
/// without that, the plain central difference's `O(h²)` truncation is what
/// the gate measures.
///
/// **The step and the mode tolerance are chosen together, because the
/// oracle's error is the mode solve's own reproducibility divided by `h`.** A
/// step too small is that noise; a step too large is what Richardson's
/// `O(h⁴)` remainder leaves; `h = 1e-3·max(1, |γ̂_k|)` under a `1e-14` mode
/// solve is the flat middle. Worst gap over all cells of `sim_sparse_gamma`
/// (rung 24), the corpus's hardest packed shape, scanned over both:
///
/// ```text
///   mode tol \ h_rel      1e-2      1e-3      1e-4      1e-5
///   pirls_tol_fd (1e-8)  5.35e-2   2.59e-1   8.21e-1   2.73e0
///   1e-12                3.35e-6   1.49e-4   9.97e-4   3.02e-3
///   1e-14                3.32e-6   4.03e-8   5.06e-5   5.39e-5
/// ```
///
/// Along each tolerance row `gap × h` is roughly constant and the whole row
/// drops when the mode solve is tightened — round-off over step. Truncation
/// would go the other way: a decade wider `h` multiplies an `O(h⁴)` remainder
/// by 1e4, which is the `1e-2` column.
///
/// That the ENGINE is not what moves there is settled by an arbiter that
/// never touches the assembled gradient: a Richardson-extrapolated second
/// difference of `glmm_laplace_deviance` itself, cold-seeded per evaluation,
/// against the engine's symmetrized Hessian. On rung 24's four worst θθ cells
/// that arbiter is stable on `h_rel ∈ [1e-2, 3e-3]` and agrees with the
/// engine there to 3e-9…6e-7 — its own stability — at mode tolerance `1e-12`
/// and `1e-14` alike: at `h_rel = 1e-2`, cell (10,9) 1.02e-7, (13,0) 4.28e-8,
/// (5,7) 5.73e-7, (10,10) 4.75e-8. The engine itself moves 1.17e-9 between
/// those two tolerances; the oracle moves four orders.
///
/// Bands fixed from that measurement. Columns: worst 4.03e-8 (rung 24), with
/// rungs 8, 9, 18, 38 and 46 at 1.4e-10 to 4.0e-10, so one `BAND = 1e-6`
/// clears the worst by 25× and every other rung by ~3000×. Asymmetry: worst
/// 1.6e-13 (rung 24), unchanged by the tolerance, so `ASYM_BAND = 1e-11`
/// clears it by ~60×.
#[cfg(feature = "formula")]
#[test]
fn packed_assembled_hessian_columns_match_fd_of_f64_gradient() {
    const BAND: f64 = 1e-6;
    const ASYM_BAND: f64 = 1e-11;
    for r in PACKED_LAPLACE_RUNGS {
        let (mut ws, x, y, ids, extra_ids, p, n, _dev) = rung_at_gamma_hat(r);
        let n_theta = ws.n_theta;
        let m = n_theta + p;
        let kk = ws.k.max(1);
        let saved: Vec<f64> = ws.params[..m].to_vec();
        let u_hat: Vec<f64> = ws.pirls.u[..kk].to_vec();
        ws.fd.pirls_tol_override = Some(1e-14);

        let mut hess = Mat::<f64>::zeros(m, m);
        let mut hgrad = vec![0.0; m];
        ws.pirls.u[..kk].copy_from_slice(&u_hat);
        let st = crate::glmm::joint_hessian_columns(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut hgrad,
            &mut hess,
        );
        assert!(
            matches!(st, crate::glmm::DerivStatus::Ok(_)),
            "rung {}: the packed assembled Hessian declined",
            r.rung
        );
        let mut asym = 0.0f64;
        for i in 0..m {
            for j in 0..m {
                let d = (hess[(i, j)] - hess[(j, i)]).abs();
                asym = asym.max(d / hess[(i, j)].abs().max(1.0));
            }
        }
        assert!(
            asym <= ASYM_BAND,
            "rung {}: pre-symmetrization asymmetry {asym:e}",
            r.rung
        );

        // The value part of the chunked pass IS the gradient, entry by entry.
        let mut gref = vec![0.0; m];
        ws.params[..m].copy_from_slice(&saved);
        ws.pirls.u[..kk].copy_from_slice(&u_hat);
        assert!(
            matches!(
                crate::glmm::packed_gradient(&mut ws, x.as_ref(), &y, p, n, &mut gref),
                crate::glmm::DerivStatus::Ok(_)
            ),
            "rung {}: the reference gradient declined",
            r.rung
        );
        for a in 0..m {
            let gap = (hgrad[a] - gref[a]).abs() / gref[a].abs().max(1.0);
            assert!(
                gap <= 1e-12,
                "rung {} coord {a}: the chunked pass's gradient {} vs the f64 gradient {} \
                 (relative gap {gap:e})",
                r.rung,
                hgrad[a],
                gref[a]
            );
        }

        let mut worst = 0.0f64;
        let mut worst_cell = (0usize, 0usize);
        for coord in 0..m {
            let h = 1e-3 * saved[coord].abs().max(1.0);
            let eval = |ws: &mut GlmmWorkspace, d: f64, out: &mut [f64]| {
                let mut at = saved.clone();
                at[coord] = saved[coord] + d;
                ws.params[..m].copy_from_slice(&at);
                ws.pirls.u[..kk].copy_from_slice(&u_hat);
                assert!(
                    matches!(
                        crate::glmm::packed_gradient(ws, x.as_ref(), &y, p, n, out),
                        crate::glmm::DerivStatus::Ok(_)
                    ),
                    "rung {} coord {coord}: the assembled gradient declined",
                    r.rung
                );
            };
            let mut gp = vec![0.0; m];
            eval(&mut ws, h, &mut gp);
            let mut gm = vec![0.0; m];
            eval(&mut ws, -h, &mut gm);
            let mut gp2 = vec![0.0; m];
            eval(&mut ws, 0.5 * h, &mut gp2);
            let mut gm2 = vec![0.0; m];
            eval(&mut ws, -0.5 * h, &mut gm2);
            for a in 0..m {
                let d1 = (gp[a] - gm[a]) / (2.0 * h);
                let d2 = (gp2[a] - gm2[a]) / h;
                let fd = (4.0 * d2 - d1) / 3.0;
                let gap = (hess[(a, coord)] - fd).abs() / fd.abs().max(1.0);
                if gap > worst {
                    worst = gap;
                    worst_cell = (a, coord);
                }
                assert!(
                    gap <= BAND,
                    "rung {} column {coord} entry {a}: lane {} vs fd {fd} (relative gap {gap:e})",
                    r.rung,
                    hess[(a, coord)]
                );
            }
        }
        ws.params[..m].copy_from_slice(&saved);
        ws.fd.pirls_tol_override = None;
        println!(
            "packed hessian rung {}: m {m}, worst relative gap {worst:e} at {worst_cell:?}, \
             asymmetry {asym:e}",
            r.rung
        );
    }
}

/// The packed assembled engine's shipped covariance against the packed FD
/// stencil's, at every packed rung's γ̂: the `p` fixed-effect standard errors
/// off `joint_hessian_cov`'s own `out_cov`, and the `n_θ` θ-block standard
/// errors off `ws.inference.theta_se` (which `stddev_se` is, divided by the
/// fixed positive Λ-row scales, so a relative gap here is a relative gap
/// there).
///
/// The stencil is reached through `ws.fd.force_fd_hessian`, the workspace
/// switch the other layouts' FD tests already use, so no global state moves
/// and the two arms run on one workspace at one γ̂.
///
/// A MEASUREMENT with a guard rail, not an equality: the stencil is a
/// single-step central difference at `SPARSE_FD_STEP_REL = 1e-4` relative and
/// carries its own step error. `src/sparse/fd_margin.rs`'s header records the
/// size of that error on this very layout — δ-vs-δ/2 standard-error agreement
/// of 2.0e-5 on `sim_sparse_gamma`, and 1.4e-4 on `sim_sparse_binomial_bigsd`,
/// whose sequence is truncation-dominated with its two extrapolations 20%
/// apart.
///
/// Measured gaps, worst per rung: 1.65e-4 `se` / 2.16e-5 `theta_se` on rung
/// 46, 1.43e-5 / 2.50e-6 on rung 24, and ≤1.5e-6 / ≤4.1e-7 on rungs 8, 9, 18
/// and 38. The two large ones land on exactly the two rungs `fd_margin`
/// records the stencil's own step error for, and at the size it records — so
/// the whole gap is the stencil's, and this comparison says nothing about the
/// assembled engine's accuracy (`sparse::tests`'s both-layouts cross-check
/// does, at 2.6e-13, on the shapes the dense engine can also fit).
/// `BAND = 1e-3` clears the worst by ~6× and is a tripwire
/// for a routing or sign error: a gap far above what the stencil's own step
/// error explains is a finding to chase, not a band to widen.
#[cfg(feature = "formula")]
#[test]
fn packed_assembled_se_matches_the_packed_stencil() {
    use std::sync::atomic::Ordering;
    const BAND: f64 = 1e-3;
    for r in PACKED_LAPLACE_RUNGS {
        let (mut ws, x, y, ids, extra_ids, p, n, _dev) = rung_at_gamma_hat(r);
        let n_theta = ws.n_theta;
        assert!(
            crate::glmm::assembly_routes(&ws, n),
            "rung {}: the assembled engine must route this shape",
            r.rung
        );
        let before = crate::glmm::ASSEMBLED_OK_COUNT.load(Ordering::Relaxed);
        let mut cov_exact = Mat::<f64>::zeros(p, p);
        let st = crate::glmm::joint_hessian_cov(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut cov_exact,
        );
        assert!(
            matches!(st, crate::glmm::FdHessianStatus::Ok),
            "rung {}: the exact arm must not fall back",
            r.rung
        );
        assert!(
            crate::glmm::ASSEMBLED_OK_COUNT.load(Ordering::Relaxed) > before,
            "rung {}: the assembled arm must actually run",
            r.rung
        );
        let se_exact: Vec<f64> = (0..p).map(|j| cov_exact[(j, j)].sqrt()).collect();
        let th_exact: Vec<f64> = ws.inference.theta_se[..n_theta].to_vec();

        ws.fd.force_fd_hessian = true;
        let mut cov_fd = Mat::<f64>::zeros(p, p);
        let st_fd = crate::glmm::joint_hessian_cov(
            &mut ws,
            x.as_ref(),
            &y,
            &ids,
            &extra_ids,
            p,
            n,
            &mut cov_fd,
        );
        ws.fd.force_fd_hessian = false;
        assert!(
            matches!(st_fd, crate::glmm::FdHessianStatus::Ok),
            "rung {}: the stencil arm must not fall back ({st_fd:?})",
            r.rung
        );
        let se_fd: Vec<f64> = (0..p).map(|j| cov_fd[(j, j)].sqrt()).collect();
        let th_fd: Vec<f64> = ws.inference.theta_se[..n_theta].to_vec();

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-300);
        let worst_se = (0..p)
            .map(|j| rel(se_exact[j], se_fd[j]))
            .fold(0.0, f64::max);
        let worst_th = (0..n_theta)
            .map(|j| rel(th_exact[j], th_fd[j]))
            .fold(0.0, f64::max);
        println!(
            "packed se rung {}: worst relative gap vs the stencil — se {worst_se:e}, \
             theta_se {worst_th:e}",
            r.rung
        );
        assert!(
            worst_se <= BAND && worst_th <= BAND,
            "rung {}: se gap {worst_se:e}, theta_se gap {worst_th:e} — both must sit inside \
             the stencil's own step error",
            r.rung
        );
    }
}

/// A dense-routed design whose joint coordinate count sits above the dual
/// kernel's lane cap: `q_p = 4` — an intercept and three random slopes, so
/// `n_theta = 10` — with `p = 4`, giving `m = 14 > MAX_DUAL_N`. Inside the
/// dense envelope on every axis (`MAX_PRIMARY_Q` is 8), so the layout stays
/// blocked and both Hessian passes are asked the same question about it.
#[cfg(feature = "formula")]
fn wide_slope_fixture() -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, GroupIds) {
    let (n_clusters, per) = (15usize, 20usize);
    let n = n_clusters * per;
    let p = 4;
    let mut st = 20261u64;
    let sd = [0.5, 0.4, 0.3, 0.3];
    let u: Vec<[f64; 4]> = (0..n_clusters)
        .map(|_| std::array::from_fn(|q| sd[q] * lcg(&mut st)))
        .collect();
    let (mut x, mut y) = (vec![0.0f64; n * p], vec![0.0f64; n]);
    let mut g = vec![0u32; n];
    for i in 0..n {
        g[i] = (i % n_clusters) as u32;
        let uc = &u[g[i] as usize];
        x[i * p] = 1.0;
        for j in 1..p {
            x[i * p + j] = lcg(&mut st);
        }
        let fixed = [0.3, 0.5, -0.4, 0.3];
        let eta: f64 = (0..p).map(|j| (fixed[j] + uc[j]) * x[i * p + j]).sum();
        y[i] = eta.exp().round();
    }
    let model = ModelSpec {
        family: Family::Poisson {
            link: PoissonLink::Log,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![1, 2, 3],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: g,
        extra: vec![],
    };
    (x, y, n, p, model, ids)
}

/// The assembled engine covers a dense shape the hyper-dual pass refuses for
/// its lane count — the sibling of
/// `glmm::derivative::tests::laplace_hessian_m_above_cap_is_unsupported`,
/// which holds the refusal itself.
///
/// Both halves run on ONE workspace at ONE γ̂, so the two passes differ in
/// nothing but the pass. `laplace_hessian` refuses because a cross-chunk
/// second-derivative block would need both coordinates' first-order lanes
/// live in one call; the assembled pass reads second order off FIRST-order
/// lanes, which chunk, so nothing about `m` reaches its routing gate. What
/// this asserts is the consequence: at `m = 14` the shipped covariance is the
/// exact one, not the FD stencil's.
#[cfg(feature = "formula")]
#[test]
fn assembled_hessian_covers_an_m_above_the_dual_lane_cap() {
    use std::sync::atomic::Ordering;

    let (xf, y, n, p, model, ids) = wide_slope_fixture();
    let (mut ws, x, ids_p, extra_ids) = ws_at_gamma_hat(
        &xf,
        &y,
        None,
        n,
        p,
        &model,
        &ids,
        "the wide-slope dense fixture",
    );
    let ids = ids_p;
    let m = ws.n_theta + p;
    // γ̂ captured ONCE, before any evaluation walks `ws.params` off it.
    let gamma_hat: Vec<f64> = ws.params[..m].to_vec();
    assert!(
        m > crate::glmm::MAX_DUAL_N,
        "this fixture is here for its coordinate count: m {m}"
    );
    assert!(
        crate::glmm::supports_exact_shape(ws.layout, &ws.groupings),
        "the shape itself must be one the dual kernel has a twin for, so the refusal \
         below is the lane cap and nothing else"
    );

    let mut g = vec![0.0; m];
    let mut hess = Mat::<f64>::zeros(m, m);
    let st = crate::glmm::laplace_hessian(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut g,
        &mut hess,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Unsupported),
        "the hyper-dual pass still refuses above its lane cap"
    );
    let st = crate::glmm::joint_hessian_columns(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut g,
        &mut hess,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "the assembled engine must not refuse a dense shape for its lane count"
    );

    let before = crate::glmm::ASSEMBLED_OK_COUNT.load(Ordering::Relaxed);
    let mut cov = Mat::<f64>::zeros(p, p);
    let st =
        crate::glmm::joint_hessian_cov(&mut ws, x.as_ref(), &y, &ids, &extra_ids, p, n, &mut cov);
    assert!(
        matches!(st, crate::glmm::FdHessianStatus::Ok),
        "the exact arm must answer at this m ({st:?})"
    );
    assert!(
        crate::glmm::ASSEMBLED_OK_COUNT.load(Ordering::Relaxed) > before,
        "the shipped covariance at this m must come from the assembled arm"
    );
    // Arbiter: the Richardson-extrapolated central second difference of the f64
    // Laplace deviance, the same independent route
    // `assembled_hessian_matches_hyperdual_per_entry` puts `CLOGLOG_LARGE`
    // through. Needed here because this is the ONE dense shape the hyper-dual
    // pass cannot cross-check, so without it the assembled `hess` and the `cov`
    // built from it are only asserted to exist.
    //
    // Both bands sit above their measured worst and below twice it, the shape
    // the `CLOGLOG_LARGE` arbiter there uses. Measured on this tree: the
    // arbiter reproduces itself between base steps 4e-3 and 8e-3 to 9.36e-6,
    // the assembled Hessian agrees with the base-4e-3 estimate to 6.31e-7, and
    // the shipped covariance to 1.11e-8 of its own scale. The covariance is the
    // tighter comparison because inverting the joint Hessian damps the FD step
    // error the Hessian entries still carry.
    const FD_BAND: f64 = 1e-6;
    // Relative to √(cov_aa·cov_bb) — a correlation scale, so the near-zero
    // off-diagonal entries are held to the same standard as the diagonal
    // instead of passing on their own smallness.
    const COV_BAND: f64 = 2e-8;
    let n_theta = ws.n_theta;
    ws.fd.pirls_tol_override = Some(1e-12);
    let fd4 = richardson_fd_hessian(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        n,
        m,
        n_theta,
        4e-3,
        &gamma_hat,
    );
    let fd8 = richardson_fd_hessian(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        n,
        m,
        n_theta,
        8e-3,
        &gamma_hat,
    );
    ws.fd.pirls_tol_override = None;
    let rel = |a: f64, b: f64| (a - b).abs() / a.abs().min(b.abs()).max(1.0);
    let mut repro = 0.0f64;
    let mut worst_fd = 0.0f64;
    for i in 0..m {
        for j in 0..m {
            repro = repro.max(rel(fd4[(i, j)], fd8[(i, j)]));
            let gap = rel(hess[(i, j)], fd4[(i, j)]);
            worst_fd = worst_fd.max(gap);
            assert!(
                gap <= FD_BAND,
                "entry ({i},{j}): assembled {} vs Richardson FD(4e-3) {} (relative gap {gap:e})",
                hess[(i, j)],
                fd4[(i, j)]
            );
        }
    }

    // `joint_hessian_cov` ships `cov = 2·(H⁻¹)_ββ` (`src/glmm/se.rs`), so invert
    // the arbiter's Hessian the same way and compare the shipped covariance
    // itself — a wrong scale factor, an off-diagonal sign slip or a missing
    // link-derivative term all survive a positive-diagonal check.
    let fd_cov = {
        use faer::linalg::solvers::Solve;
        let chol = fd4
            .as_ref()
            .llt(faer::Side::Lower)
            .expect("the arbiter Hessian at γ̂ must be PD");
        let mut inv = Mat::<f64>::identity(m, m);
        chol.solve_in_place(inv.as_mut());
        inv
    };
    let mut worst_cov = 0.0f64;
    for a in 0..p {
        for b in 0..p {
            let want = 2.0 * fd_cov[(n_theta + a, n_theta + b)];
            let scale = (cov[(a, a)] * cov[(b, b)]).sqrt();
            let gap = (cov[(a, b)] - want).abs() / scale;
            worst_cov = worst_cov.max(gap);
            assert!(
                gap <= COV_BAND,
                "cov[({a},{b})] {} vs Richardson FD {want} (relative gap {gap:e})",
                cov[(a, b)]
            );
        }
    }
    println!(
        "wide-slope fixture: n {n}, m {m}, n_theta {n_theta}; FD reproducibility (4e-3 vs 8e-3) \
         {repro:e}, assembled vs FD(4e-3) {worst_fd:e}, cov vs FD {worst_cov:e}"
    );
}

/// Family of [`packed_clamped_fixture`], named once so the fixture, its γ̂
/// driver and the fit that reads its diagnostics cannot drift apart.
#[cfg(feature = "formula")]
const PACKED_CLAMPED_FAMILY: Family = Family::Poisson {
    link: PoissonLink::Log,
};

/// A packed-row design whose converged mode state carries a μ-clamped row:
/// `(x, y, offset, n, p, model, ids)`. No corpus rung is in that state on
/// this layout, so the refusal path needs a constructed cell.
///
/// Packed by construction: a random slope on an EXTRA grouping is a shape
/// `classify_design` answers `Solver::Sparse` for, so `GlmmLayout::for_design`
/// takes the packed-row layout whatever the row count. The draw is the
/// extra-grouping-slope Poisson design `loop_tier_honours_extra_grouping_slope`
/// fits.
///
/// The clamp comes from a large negative offset on row 0 and nothing else — no
/// edited response, no outlying covariate. On the log link μ = exp(η), so an
/// offset of −30 against an η of order 1 leaves exp(η) ≈ 1e-13, under
/// `family::MU_FLOOR` (1e-10), and μ is held at that bound — which is what
/// makes the census read one μ-clamped row. That row's own deviance
/// contribution is `2(μ − y·…) ≈ 2e-10` at `y = 0`, so it moves the census
/// without moving the optimum.
#[cfg(feature = "formula")]
#[allow(clippy::type_complexity)]
fn packed_clamped_fixture() -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
) {
    let (n_g1, n_g2, per) = (8usize, 6usize, 10usize);
    let n = n_g1 * per;
    let p = 2;
    let mut st = 7u64;
    let (mut x, mut y) = (vec![0.0f64; n * p], vec![0.0f64; n]);
    let (mut g1, mut g2) = (vec![0u32; n], vec![0u32; n]);
    for i in 0..n {
        g1[i] = (i % n_g1) as u32;
        g2[i] = (i % n_g2) as u32;
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        let eta = 0.5 + 0.4 * x1 + 0.25 * (g1[i] as f64 - 4.0) + 0.6 * x1 * (g2[i] as f64 - 3.0);
        y[i] = eta.exp().round();
    }
    let mut offset = vec![0.0f64; n];
    offset[0] = -30.0;
    y[0] = 0.0;
    let model = ModelSpec {
        family: PACKED_CLAMPED_FAMILY,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_g1 as u32,
            },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: n_g2 as u32,
                },
                slopes: vec![1],
            }],
        }),
    };
    let ids = GroupIds {
        primary: g1,
        extra: vec![g2],
    };
    (x, y, offset, n, p, model, ids)
}

/// One design built in memory, fitted to its own γ̂ with `WaldSe::Rx` (so the
/// fit runs no Hessian pass) and handed back as the workspace sitting at that
/// γ̂ with the column-major `X` and the sized ids: `rung_at_gamma_hat`'s
/// contract for a fixture that is not a corpus CSV.
#[cfg(feature = "formula")]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn ws_at_gamma_hat(
    x: &[f64],
    y: &[f64],
    offset: Option<Vec<f64>>,
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    what: &str,
) -> (GlmmWorkspace, Mat<f64>, Vec<u32>, Vec<Vec<u32>>) {
    let opts = FitOptions {
        target_indices: (0..p as u32).collect(),
        wald_se: WaldSe::Rx,
        offset,
        nagq: 1,
        ..FitOptions::default()
    };
    let (sized_model, sized_ids, _perm) = super::spec_sized_from_ids(model, ids);
    let (mut ws, x_mat) = super::glmm::fit_glmm_build(
        x,
        n,
        p,
        &sized_model,
        &sized_ids.primary,
        &sized_ids.extra,
        &opts,
    )
    .unwrap_or_else(|_| panic!("{what}: degenerate design"));
    let view = super::glmm::run_glmm_on(
        &mut ws,
        x_mat.as_ref(),
        y,
        n,
        p,
        &sized_model,
        &sized_ids.primary,
        &sized_ids.extra,
        f64::NAN,
        None,
        &opts,
    );
    assert!(view.converged_deviance().0, "{what}: fit must converge");
    let sized_ids = sized_ids.into_owned();
    (ws, x_mat, sized_ids.primary, sized_ids.extra)
}

/// The packed assembled path on a clamped mode state, end to end: on a
/// packed-row fit whose mode state carries a μ-clamped row the assembled
/// engine takes the fit, and `joint_hessian_cov` ships its
/// covariance for it.
///
/// The reference side drives `joint_hessian` directly at the same γ̂ and
/// inverts its matrix the way `joint_hessian_cov` inverts it — `cov = 2·H⁻¹`
/// on the β block, `sqrt(2·H⁻¹_kk)` on the θ diagonal — so the comparison is
/// bitwise and needs no band. `packed_gradient` at this γ̂ is checked
/// separately against a Richardson central difference of
/// `glmm_laplace_deviance`, the shape
/// `packed_assembled_gradient_matches_richardson_fd` uses.
///
/// The take is asserted at the engine, not through
/// `assembled::ASSEMBLED_OK_COUNT`: that counter is process-wide and any
/// concurrently-running test's `WaldSe::Hessian` fit advances it too, so it
/// cannot say this call in particular is what advanced it.
#[cfg(feature = "formula")]
#[test]
fn clamped_packed_fit_ships_the_assembled_covariance() {
    use faer::linalg::solvers::Solve;

    let (xf, y, offset, n, p, model, ids) = packed_clamped_fixture();
    let (mut ws, x, ids_p, extra_ids) = ws_at_gamma_hat(
        &xf,
        &y,
        Some(offset),
        n,
        p,
        &model,
        &ids,
        "the packed clamped fixture",
    );
    let ids = ids_p;
    assert_eq!(
        ws.layout,
        crate::glmm::GlmmLayout::Packed,
        "an extra-grouping slope must take the packed-row layout"
    );
    let n_theta = ws.n_theta;
    let m = n_theta + p;
    let (mu_clamped, eta_clamped) = clamp_census(&ws, PACKED_CLAMPED_FAMILY, ws.weighted, n);
    println!(
        "packed clamped fixture: n {n}, m {m}, k {}, census (μ {mu_clamped}, η {eta_clamped})",
        ws.k
    );
    assert!(
        mu_clamped > 0,
        "this fixture is here for its clamped mode state (μ {mu_clamped})"
    );
    assert!(
        crate::glmm::assembly_routes(&ws, n),
        "the shape and the memory guard must both admit the assembled engine"
    );

    let mut g = vec![0.0; m];
    let mut hess = Mat::<f64>::zeros(m, m);
    let st = crate::glmm::joint_hessian_columns(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut g,
        &mut hess,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "the packed assembled engine must take a clamped mode state now"
    );

    // The assembled arm on its own, symmetrized, at the same γ̂ — the
    // reference `joint_hessian_cov` below is compared against.
    let st = crate::glmm::joint_hessian(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut g,
        &mut hess,
    );
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "the assembled arm must answer"
    );
    let chol = hess
        .as_ref()
        .llt(faer::Side::Lower)
        .expect("the assembled joint Hessian is PD at this γ̂");
    let mut inv = Mat::<f64>::identity(m, m);
    chol.solve_in_place(inv.as_mut());
    let want_cov = Mat::<f64>::from_fn(p, p, |a, b| 2.0 * inv[(n_theta + a, n_theta + b)]);
    let want_theta_se: Vec<f64> = (0..n_theta)
        .map(|k| (2.0 * inv[(k, k)]).max(0.0).sqrt())
        .collect();

    let mut got_cov = Mat::<f64>::zeros(p, p);
    let st = crate::glmm::joint_hessian_cov(
        &mut ws,
        x.as_ref(),
        &y,
        &ids,
        &extra_ids,
        p,
        n,
        &mut got_cov,
    );
    assert!(
        matches!(st, crate::glmm::FdHessianStatus::Ok),
        "expected Ok ({st:?})"
    );
    for a in 0..p {
        for b in 0..p {
            assert_eq!(
                got_cov[(a, b)].to_bits(),
                want_cov[(a, b)].to_bits(),
                "cov[({a},{b})] {} is not the assembled arm's {}",
                got_cov[(a, b)],
                want_cov[(a, b)]
            );
        }
    }
    for (k, &want) in want_theta_se.iter().enumerate() {
        assert_eq!(
            ws.inference.theta_se[k].to_bits(),
            want.to_bits(),
            "theta_se[{k}] {} is not the assembled arm's {want}",
            ws.inference.theta_se[k]
        );
    }

    // `packed_gradient` at this fixture's γ̂ against a Richardson central
    // difference of `glmm_laplace_deviance`, the same shape
    // `packed_assembled_gradient_matches_richardson_fd` uses.
    const GRAD_BAND: f64 = 1e-6;
    let kk = ws.k.max(1);
    let saved: Vec<f64> = ws.params[..m].to_vec();
    let u_hat: Vec<f64> = ws.pirls.u[..kk].to_vec();
    ws.fd.pirls_tol_override = Some(1e-12);
    let mut g_pg = vec![0.0; m];
    ws.pirls.u[..kk].copy_from_slice(&u_hat);
    let st = crate::glmm::packed_gradient(&mut ws, x.as_ref(), &y, p, n, &mut g_pg);
    assert!(
        matches!(st, crate::glmm::DerivStatus::Ok(_)),
        "the packed assembled gradient declined"
    );
    let mut worst_grad_fd = 0.0f64;
    for coord in 0..m {
        let h = 1e-3 * saved[coord].abs().max(1.0);
        let mut at = saved.clone();
        let ev = |ws: &mut GlmmWorkspace, at: &mut Vec<f64>, d: f64| -> f64 {
            at[coord] = saved[coord] + d;
            glmm_laplace_deviance(at, ws, x.as_ref(), &y, &ids, &extra_ids, n)
        };
        let d1 = (ev(&mut ws, &mut at, h) - ev(&mut ws, &mut at, -h)) / (2.0 * h);
        let d2 = (ev(&mut ws, &mut at, 0.5 * h) - ev(&mut ws, &mut at, -0.5 * h)) / h;
        let fd = (4.0 * d2 - d1) / 3.0;
        let gap = (g_pg[coord] - fd).abs() / fd.abs().max(1.0);
        worst_grad_fd = worst_grad_fd.max(gap);
        assert!(
            gap <= GRAD_BAND,
            "coord {coord}: packed gradient {} vs Richardson FD {fd} (relative gap {gap:e})",
            g_pg[coord]
        );
    }
    ws.params[..m].copy_from_slice(&saved);
    ws.fd.pirls_tol_override = None;
    println!(
        "packed clamped fixture gradient: worst relative gap vs Richardson FD {worst_grad_fd:e}"
    );

    // The same fixture through the shipped dispatch — a converged fit with
    // every target SE finite.
    let (_, _, offset, _, _, _, ids) = packed_clamped_fixture();
    let opts = FitOptions {
        target_indices: vec![0, 1],
        wald_se: WaldSe::Hessian,
        offset: Some(offset),
        ..FitOptions::default()
    };
    let f = fit_cold(&xf, &y, n, p, &model, &ids, &opts);
    assert!(f.converged(), "the packed clamped fit must converge");
    assert!(
        f.se.iter().all(|v| v.is_finite()),
        "every target SE is finite: {:?}",
        f.se
    );

    // The clean control, unaffected by this flip: the same design and code
    // path with the offset dropped still converges.
    let clean = FitOptions {
        offset: None,
        ..opts
    };
    let fc = fit_cold(&xf, &y, n, p, &model, &ids, &clean);
    assert!(fc.converged(), "the unclamped control fit must converge");
}

/// Gamma-inverse design that drives `joint_hessian_cov` off its main path: 12
/// rows in 4 size-3 clusters whose means span nine decades, with the second
/// cluster's three responses scaled by `big_scale`. The spread is what makes
/// the fitted joint (θ,β) Hessian non-PD; `big_scale` moves the fitted point
/// along that flank and so decides whether the RX/Schur fallback survives the
/// Hessian (1e-2) or fails with it (1.0).
fn gamma_inverse_adversarial(big_scale: f64) -> (Vec<f64>, Vec<f64>, ModelSpec, GroupIds) {
    const X1: [f64; 12] = [
        -0.2884326052962697,
        0.15675840297260962,
        -0.29990566105127503,
        0.3053648563367106,
        -0.608001199771249,
        1.451281361998219,
        -0.12817918698856592,
        0.2873778743945057,
        1.2188666482896826,
        -1.2520230368147756,
        -0.01938374443978608,
        1.0354827975218956,
    ];
    const Y: [f64; 12] = [
        0.013147681629338537,
        0.022571111060908807,
        0.009380481267257667,
        13915684.733974772,
        16148934.63843452,
        47449285.58840935,
        0.03659612272549357,
        0.03340213953300986,
        0.16526466878896995,
        2288.1399088594685,
        5963.477254271042,
        19049.155692047607,
    ];
    let x: Vec<f64> = X1.iter().flat_map(|&v| [1.0, v]).collect();
    // Rows 3..6 are the second cluster — the one the scale moves.
    let y: Vec<f64> = Y
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if (3..6).contains(&i) {
                v * big_scale
            } else {
                v
            }
        })
        .collect();
    let model = ModelSpec {
        family: Family::Gamma {
            link: crate::GammaLink::Inverse,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: (0..Y.len() as u32).map(|i| i / 3).collect(),
        extra: vec![],
    };
    (x, y, model, ids)
}

/// A Gamma-inverse fit where the joint Hessian and the RX fallback's Schur are
/// both non-PD at the returned point: 12 rows in 4 clusters whose means span
/// nine decades. `joint_hessian_cov` NaN-fills the covariance there and the fit
/// comes back as a failed fit, not a panic.
#[test]
fn gamma_inverse_double_se_failure_returns_a_failed_fit() {
    let (x, y, model, ids) = gamma_inverse_adversarial(1.0);
    let (n, p) = (12, 2);
    let fit = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &ids,
        &FitOptions {
            target_indices: vec![0, 1],
            wald_se: WaldSe::Hessian,
            ..FitOptions::default()
        },
    );
    assert!(!fit.converged(), "the double SE failure is a failed fit");
    assert!(fit.se.iter().all(|s| s.is_nan()), "se = {:?}", fit.se);
}

/// The single SE failure on the same shape: the joint (θ,β) Hessian is non-PD
/// but the β-only RX/Schur block stays PD, so `joint_hessian_cov` reports
/// `FdHessianStatus::NonPdFellBackToRx` and the fit ships an RX covariance
/// under `Note::HessianSeFallback` instead of failing.
///
/// That the SEs come from RX is asserted against a `WaldSe::Rx` fit of the same
/// design: same kernel, same converged point, so `se` must match bit for bit
/// and `vcov` to a few ULP. A band any wider would hide a scale slip — the
/// Gamma σ̂² factor the fallback reapplies by hand is exactly what this pins.
/// `stddev_se` must be NaN: the fallback never assembles a joint Hessian, so
/// there is no θ-block SE to report, and a stale value from an earlier fit on
/// the reused workspace must not leak out in its place.
#[test]
fn gamma_inverse_non_pd_hessian_falls_back_to_rx_se() {
    let (x, y, model, ids) = gamma_inverse_adversarial(1e-2);
    let (n, p) = (12, 2);
    let opts = |wald_se| FitOptions {
        target_indices: vec![0, 1],
        wald_se,
        ..FitOptions::default()
    };
    let fit = fit_cold(&x, &y, n, p, &model, &ids, &opts(WaldSe::Hessian));
    let rx = fit_cold(&x, &y, n, p, &model, &ids, &opts(WaldSe::Rx));
    assert!(
        fit.converged() && rx.converged(),
        "both arms must converge for the comparison to mean anything"
    );
    assert!(
        fit.diagnostics
            .notes
            .iter()
            .any(|note| matches!(note, crate::Note::HessianSeFallback)),
        "expected HessianSeFallback, got {:?}",
        fit.diagnostics.notes
    );
    assert_eq!(fit.se, rx.se, "fallback se must BE the RX se");
    // The covariance only meets to a few ULP off the diagonal: the production
    // Rx arm accumulates the off-diagonal from `vcov_cols` products while the
    // fallback reads it off `rx_cov_into`'s LLT inverse, a different
    // reassociation of the same quantity. The diagonal IS bit-identical — that
    // is the `se` assertion above.
    for i in 0..p {
        for j in 0..p {
            let (got, want) = (fit.vcov[i][j], rx.vcov[i][j]);
            assert!(
                want.is_finite(),
                "the fallback ships a real covariance, not the double-failure NaN fill"
            );
            assert!(
                (got - want).abs() <= 8.0 * f64::EPSILON * want.abs(),
                "vcov[{i}][{j}] = {got} vs rx {want}"
            );
        }
    }
    assert!(
        fit.stddev_se.iter().all(|v| v.is_nan()),
        "no joint Hessian ⇒ no θ-block SE: {:?}",
        fit.stddev_se
    );
}

/// The plateau policy on the GLMM route — the mirror of `src/lmm/tests.rs`'s
/// `maxfun_cap_reports_honest_endpoint`. A `Status::MaxFunReached` exit reports
/// its finite incumbent (β̂/SE/vcov/deviance) instead of NaN-filling, with
/// `converged() == false`, `Boundary::NoOptimum` and `df == 0`.
///
/// The cap is forced by swapping in a joint solver whose `max_fun` is the legal
/// minimum (`npt + 1`) — one evaluation past the initial interpolation model,
/// nowhere near cbpp's optimum. `LMM_MAX_FUN_FORMULA` would do the same, but it
/// is read once per process, so a test using it would race every other fit in
/// the binary.
#[test]
fn glmm_maxfun_cap_reports_honest_endpoint() {
    use bobyqa::{Bobyqa, Config};

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
    let (mut ws, x_mat) =
        super::glmm::fit_glmm_build(&x, n, p, &model, &ids.primary, &ids.extra, &opts)
            .unwrap_or_else(|_| panic!("cbpp design must build"));

    // Both outer solvers are capped: which one carries the fit's status depends
    // on `ws.outer_search` (cbpp routes `ExactProfile`, where stage 1 IS the
    // search), and the contract under test is the same either way. Dimensions are
    // the workspace's own: n_theta + p joint, n_theta stage 1 — no `ln θ_NB`
    // coordinate on binomial.
    let cap = |dim: usize| {
        let mut c = Config::new(dim);
        c.npt = 2 * dim + 1; // PRIMA's default, the legal minimum at dim = 1
        c.max_fun = c.npt + 1;
        Bobyqa::new(dim, c).expect("legal minimal config")
    };
    ws.solver = cap(ws.n_theta + p);
    ws.solver_stage1 = cap(ws.n_theta);

    let view = super::glmm::run_glmm_on(
        &mut ws,
        x_mat.as_ref(),
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
    let (fit, _mu, _dev) = super::glmm::glmm_view_to_fit(&view, &y, n, p, &model, &opts);

    assert!(!fit.converged(), "a capped fit must not report converged");
    assert_eq!(
        fit.diagnostics.boundary,
        Boundary::NoOptimum,
        "a capped endpoint is not an accepted boundary"
    );
    assert_eq!(fit.df, 0, "df is gated on convergence");
    assert!(
        fit.deviance.is_finite(),
        "plateau policy: capped endpoint must report a finite deviance, got {}",
        fit.deviance
    );
    for &tj in &opts.target_indices {
        let j = tj as usize;
        assert!(
            fit.beta[j].is_finite() && fit.se[j].is_finite(),
            "plateau policy: capped endpoint must not NaN-fill β/se at j={j}: β {} se {}",
            fit.beta[j],
            fit.se[j]
        );
        assert!(
            fit.vcov[j][j].is_finite(),
            "plateau policy: capped endpoint must not NaN-fill vcov at j={j}"
        );
    }
}
