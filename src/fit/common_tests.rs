//! Dispatch-level tests shared across estimators: `fit_cold`/`fit_warm`
//! marshalling (weights shape/AGQ rejection, rank-deficiency salvage),
//! θ-width/varcorr bookkeeping, `spec_sized_from_ids` sizing, and
//! `classify_design` routing. Also hosts the golden-data generators
//! (`lcg`, `lmm_hand_dataset`, `dense_ids`, `dense_str`, `sim_clustered`)
//! shared by 2+ estimator test files.

use super::*;
use crate::{
    BinomialLink, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, NegBinomialLink,
    ReStructure, Sizing, StartValues,
};

/// Band for pins on the closed-form paths — OLS and GLM/IRLS.
///
/// These are Rust-vs-Rust pins, NOT cross-engine agreement bands: they say the
/// number has not moved, and nothing about lme4. Cross-engine bands live in
/// `validation/tol.R` and are asserted by the `oracle-tests` tier.
pub(crate) const PIN_REL_OLS: f64 = 1e-9;

/// Band for pins on the iterative paths — LMM/GLMM, where BOBYQA stops inside
/// its own convergence tolerance rather than on a fixed point, so a shifted
/// last bit in the objective moves the reported optimum further than it does on
/// the closed-form paths.
///
/// Measured, not chosen — see [`assert_pinned`] for which machine measured it
/// and what the second machine cost. Bit-exact pins would be the tighter claim
/// and are wrong for exactly that reason: they would go red on a host drawing a
/// different microarchitecture, for no bug.
pub(crate) const PIN_REL_ITER: f64 = 1e-7;

/// Assert a fitted vector against its pinned values, elementwise relative.
/// `what` names the quantity so a failure says which one moved. Length is
/// asserted too: a pin that silently covers fewer elements than the fit
/// produces is the same nothing-asserted failure the pins exist to prevent.
///
/// The failure names the WORST element, not the first one over the band. That
/// matters for the band-measurement run below: stopping at the first element
/// would report whichever came first in index order rather than the spread the
/// band has to cover.
///
/// # Which machine the pins are frozen on
///
/// This is the one place that says it; every pin site cites this function rather
/// than re-explaining. If you are re-freezing a pin, read this first.
///
/// **The anchor is x86_64-unknown-linux-gnu**, Intel Core Ultra 7 265H (Arrow
/// Lake-H, AVX2 + FMA, no AVX-512). Every Rust-vs-Rust pin in the crate — the
/// `PIN_REL_OLS` / `PIN_REL_ITER` pins and the per-test `BAND` pins in
/// `glmm_tests.rs` and `sparse/tests.rs` — reproduces there BIT-EXACTLY, worst
/// relative spread 0.0, across all four feature configs (default,
/// `loop_advanced`, `parallel`, `--no-default-features`) in both debug and
/// release: eight legs, zero non-exact pins. Verified 2026-07-31 by rewriting
/// every band to `1e-300` and re-running the suite, so a bit-exact pin passes
/// and one off by a single ULP fails. `1e-300` and not `0.0`: at `0.0` even an
/// exact pin fails, so each test aborts on its first quantity and a run reports
/// every test's `beta` and nothing about its `se`. That is the `pin-bands`
/// workflow's procedure, and it rewrites this band list — keep the two in step.
///
/// **The bands are margin for a different CPU, and that cost is now measured.**
/// The second machine is aarch64-apple-darwin, where `simd_transcendental.rs`
/// dispatches differently and the compiler contracts multiply-adds into FMA at
/// other points in this kernel's long reductions. Drift off the anchor values,
/// worst per test: 1.4e-7 (`fit_wide_crossed_sparse_is_pinned`), 1.5e-7
/// (`fit_glmm_binomial_slope1_vector_agq_is_pinned`), 1.4e-6
/// (`fit_wide_slopes_sparse_is_pinned`), 3.6e-6
/// (`fit_glmm_binomial_slope2_vector_agq_is_pinned`), 7.7e-6
/// (`fit_sparse_binomial_slope_crossed_is_pinned`), 7.1e-5
/// (`fit_glmm_poisson_slope1_vector_agq_is_pinned`), 7.9e-4
/// (`fit_sparse_nb_glmm_is_pinned`).
///
/// That four-order spread is not noise in the measurement — it tracks how much
/// ITERATION sits downstream of the reduction that rounds differently. A single
/// solve lands at ~1e-7; the NB fit's marginal golden-section θ search compounds
/// the same last-bit difference into 7.9e-4. So a band is sized per test against
/// its own measured drift, never corpus-wide.
///
/// **Re-freezing rule: re-pin on the anchor.** A value harvested on the second
/// machine is as arithmetically valid as the anchor's, but mixing the two splits
/// the corpus across two references and makes "bit-exact on the anchor" stop
/// being checkable — which is the property that lets a real regression be told
/// apart from a port. Four `se` vectors were re-anchored on 2026-07-31 for
/// exactly that reason; each says so at its own site.
///
/// **Not covered by any of this: pins asserted against a frozen R/Julia
/// reference** (`fit_glmm_binomial_bigsd_agq_matches_lme4`, and everything in
/// `tests/validation_oracle.rs`). Those bands are cross-ENGINE agreement, sized
/// by the reference's own reproducibility, and re-pinning them to any machine of
/// ours would be re-pinning the oracle. They are never bit-exact and are not
/// supposed to be.
pub(crate) fn assert_pinned(got: &[f64], want: &[f64], band: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (f64::NEG_INFINITY, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let rel = (g - w).abs() / w.abs();
        if rel > worst.0 {
            worst = (rel, i);
        }
    }
    let (rel, i) = worst;
    assert!(
        rel < band,
        "{what}[{i}] = {} vs pinned {} (rel {rel:.2e}, worst of {})",
        got[i],
        want[i],
        got.len()
    );
}

/// Parse a `cluster,x,grp,y` sim CSV → (X=[1,x,grp_b], y, dense cluster ids, n_clusters).
/// `pub(crate)` so `src/sparse/tests.rs` can load the same fixtures.
pub(crate) fn sim_clustered(csv: &str) -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
    let mut x = Vec::<f64>::new();
    let mut y = Vec::<f64>::new();
    let mut raw = Vec::<u32>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        raw.push(f[0].parse().unwrap());
        x.extend_from_slice(&[
            1.0,
            f[1].parse().unwrap(),
            f64::from(u32::from(f[2] == "b")),
        ]);
        y.push(f[3].parse().unwrap());
    }
    let (ids, nc) = dense_ids(&raw);
    (x, y, ids, nc)
}

/// Shape asserts (length + positivity) landed in Task 1 and never moved —
/// this pins the wrong-length case still faults on an otherwise-open path
/// (fixed-only OLS), independent of the family/RE capability map above.
#[test]
#[should_panic(expected = "n elements")]
fn weights_shape_still_asserted() {
    let n = 4;
    let x = vec![1.0f64; n];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        weights: Some(vec![1.0; n - 1]),
        ..FitOptions::default()
    };
    let _ = fit_cold(&x, &y, n, 1, &model, &GroupIds::default(), &opts);
}

#[test]
fn fit_rank_deficient_drops_and_matches_reduced() {
    let n = 30;
    let p = 3;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = 1.0 + x1; // col2 = col0 + col1 exactly
        y[i] = 0.5 + 0.4 * x1 + 0.3 * lcg(&mut st);
    }
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
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "reduced OLS must converge");
    assert_eq!(
        f.aliased(),
        vec![false, false, true],
        "later collinear column dropped"
    );
    assert!(f.beta[2].is_nan(), "aliased β = NaN");
    assert!(f.se[2].is_nan(), "aliased se = NaN");

    // Direct fit on the 2-column reduced design.
    let mut xr = vec![0.0f64; n * 2];
    for i in 0..n {
        xr[i * 2] = x[i * p];
        xr[i * 2 + 1] = x[i * p + 1];
    }
    let fr = fit_cold(
        &xr,
        &y,
        n,
        2,
        &model,
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    assert!(
        (f.beta[0] - fr.beta[0]).abs() < 1e-9,
        "β0 {} vs reduced {}",
        f.beta[0],
        fr.beta[0]
    );
    assert!(
        (f.beta[1] - fr.beta[1]).abs() < 1e-9,
        "β1 {} vs reduced {}",
        f.beta[1],
        fr.beta[1]
    );
}

/// Gap B — an aliased fixed column that is ALSO an RE random slope must report a
/// non-converged `Fit`, not panic.
///
/// `y ~ 1 + x1 + x2 + (0 + x2 | g)` with `x2 = 1 + x1` exactly, so `x2` is
/// aliased at `ALIAS_EPS` and `fit_warm` routes into `fit_rank_deficient` — which
/// then has to remap the RE slope index through the kept-columns map and finds
/// the slope's column gone. Dropping the random slope alongside the fixed column
/// would be a different model, so the model is genuinely unfittable; the question
/// is only how that is reported. It used to `assert!`, which takes the caller's
/// whole process down — an R/Python user got an abort instead of an inspectable
/// fit, and a loop caller lost the entire run over one degenerate draw.
///
/// Deliberately written with `catch_unwind` rather than `#[should_panic]`: the
/// assertion under test is that NO panic happens, so a regression must surface as
/// a test FAILURE with the panic message attached, not as an abort that takes the
/// rest of the suite's output with it. `AssertUnwindSafe` is sound here because
/// nothing observed after the call is shared with the closure — the inputs are
/// read-only and the `Fit` is moved out.
#[test]
fn rank_deficient_random_slope_returns_nonconverged_instead_of_panicking() {
    let (n_clusters, per) = (8usize, 12usize);
    let n = n_clusters * per;
    let p = 3usize;
    let mut st = 29u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        x[i * p + 2] = 1.0 + x1; // col2 = col0 + col1 exactly ⇒ aliased at 1e-14
        y[i] = 0.5 + 0.4 * x1 + 0.3 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_clusters as u32,
            },
            slopes: vec![2], // the aliased column, as a random slope
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: ids,
        extra: vec![],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1, 2],
        ..FitOptions::default()
    };

    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_cold(&x, &y, n, p, &model, &ids, &opts)
    }));
    let f = match caught {
        Ok(f) => f,
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".to_string());
            panic!("fit_cold panicked on a rank-deficient random slope: {msg}");
        }
    };

    assert!(
        !f.converged(),
        "an unfittable model must not report converged"
    );
    assert!(
        f.beta.iter().all(|b| b.is_nan()) && f.se.iter().all(|s| s.is_nan()),
        "β/se must be NaN-filled: beta {:?} se {:?}",
        f.beta,
        f.se
    );
    assert!(
        f.vcov.iter().all(|row| row.iter().all(|v| v.is_nan())),
        "vcov must be NaN-filled"
    );
    // The mask is the one diagnostic the caller can act on, so it is reported
    // rather than blanked — the same field the successful salvage fills.
    assert_eq!(
        f.aliased(),
        vec![false, false, true],
        "the aliased column is still named"
    );
    // θ width for q_p = 2 (intercept + one slope) is vech = 3; NaN-filled, not
    // dropped, because the RE structure was never reduced.
    assert_eq!(f.tau2.len(), 3, "tau2 keeps the unreduced θ width");
    assert!(f.tau2.iter().all(|t| t.is_nan()), "tau2 NaN-filled");
    assert!(f.dispersion.is_nan() && f.deviance.is_nan() && f.loglik.is_nan());
    assert_eq!(f.df, 0, "no parameters were estimated");
    assert!(
        !f.singular(),
        "singular is a fitted-boundary flag, not a failure flag"
    );
    assert!(f.varcorr.is_empty() && f.fitted.is_empty() && f.ranef.is_empty());
}

/// Rank-deficiency salvage on a fixed-only design: near-collinear
/// `y ~ 1 + x1 + x2 + x3` (x3 ≈ x1+x2) must drop x3, mark it in `Fit::aliased`,
/// and fit the reduced model.
///
/// Values recorded from glmm. They are validated by `sim_collinear_glm`, whose
/// cross-engine cell checks the same fit against R's `stats::glm` — including
/// that R drops the SAME column (R keeps the name and writes `NA`; glmm keeps
/// the column and writes NaN). That agreement claim lives there, not here, so
/// this test can pin at a band the closed-form OLS path actually holds.
#[test]
fn fit_sim_collinear_drops_the_aliased_column() {
    // Surviving coefficients of the reduced fit; index 3 is the dropped x3.
    const REF_BETA: [f64; 3] = [0.9521742640860978, 0.7192032159906663, -0.43352501721286113];
    const REF_SE: [f64; 3] = [0.05307754868563462, 0.0526001540491167, 0.05132117947475763];

    let csv = include_str!("../../validation/data/simulated/sim_collinear.csv");
    let mut y = Vec::<f64>::new();
    let mut cols: Vec<[f64; 3]> = Vec::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        cols.push([
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
            f[3].parse().unwrap(),
        ]);
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
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        },
    );

    assert!(f.converged(), "reduced fit converges");
    assert_eq!(
        f.aliased(),
        vec![false, false, false, true],
        "x3 is the dependent column and the only one dropped"
    );
    assert!(f.beta[3].is_nan(), "aliased β = NaN");
    assert!(f.se[3].is_nan(), "aliased se = NaN");
    assert_pinned(&f.beta[..3], &REF_BETA, PIN_REL_OLS, "beta");
    assert_pinned(&f.se[..3], &REF_SE, PIN_REL_OLS, "se");
}

/// Deterministic pseudo-data (NR LCG), uniform in (−1, 1). Mirrors the
/// LCG in `src/lmm/tests.rs` so the smoke dataset behaves the same way.
pub(super) fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

/// n=48, p=3, 6 clusters — same shape as `src/lmm/tests.rs`'s `hand_dataset`, adapted
/// to the row-major f64 layout the friendly API expects.
pub(super) fn lmm_hand_dataset() -> (Vec<f64>, Vec<f64>, usize, usize) {
    let n = 48usize;
    let p = 3;
    let n_clusters = 6usize;
    let mut st = 42u64;
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
    (x, y, n, p)
}

#[test]
fn theta_width_counts_vech_blocks() {
    // intercept-only primary + 1 slope → q_p=2 → 3; one intercept-only crossed → +1.
    let re = ReStructure {
        sizing: Sizing::FixedClusters { n_clusters: 4 },
        slopes: vec![1],
        extra_groupings: vec![Grouping {
            relation: GroupingRelation::Crossed { n_clusters: 3 },
            slopes: vec![],
        }],
    };
    assert_eq!(super::common::theta_width(Some(&re)), 3 + 1);
    assert_eq!(super::common::theta_width(None), 0);
}

/// varcorr_block computes vech(scale·ΛΛ') for a hand θ. For q=2 with Λ
/// (col-major lower-tri vech) [2.0, 0.5, 1.0] → Λ=[[2,0],[0.5,1]],
/// D=ΛΛ'=[[4,1],[1,1.25]], vech col-major = [4, 1, 1.25]; ×scale.
#[test]
fn varcorr_block_is_scaled_lambda_gram() {
    let vech = super::common::varcorr_block(&[2.0, 0.5, 1.0], 2, 1.0, &[1.0, 1.0]);
    assert_eq!(vech.len(), 3);
    assert!((vech[0] - 4.0).abs() < 1e-12, "D00 {}", vech[0]);
    assert!((vech[1] - 1.0).abs() < 1e-12, "D10 {}", vech[1]);
    assert!((vech[2] - 1.25).abs() < 1e-12, "D11 {}", vech[2]);
    let scaled = super::common::varcorr_block(&[2.0, 0.5, 1.0], 2, 3.0, &[1.0, 1.0]);
    assert!((scaled[0] - 12.0).abs() < 1e-12);
    assert!((scaled[2] - 3.75).abs() < 1e-12);
}

fn fit_with_varcorr(vech: Vec<f64>) -> Fit {
    Fit {
        beta: vec![],
        se: vec![],
        vcov: vec![],
        tau2: vec![],
        dispersion: 1.0,
        diagnostics: crate::Diagnostics::from_flags(true, false, 0),
        varcorr: vec![vech],
        stddev_se: vec![],
        n_eval: 0,
        deviance: f64::NAN,
        loglik: f64::NAN,
        df: 0,
        reml: false,
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
    }
}

/// q=1: a scalar block has no off-diagonal — stddev is just sqrt(variance)
/// and the 1x1 "correlation matrix" is the trivial [[1.0]].
#[test]
fn stddev_corr_q1_trivial() {
    let f = fit_with_varcorr(vec![9.0]);
    let (sd, corr) = f.stddev_corr(0);
    assert_eq!(sd, vec![3.0]);
    assert_eq!(corr, vec![vec![1.0]]);
}

/// q=2 hand math, mirroring `varcorr_block_is_scaled_lambda_gram`'s D:
/// D=[[4,1],[1,1.25]] → vech(col-major lower-tri)=[4,1,1.25].
/// stddev=[2, sqrt(1.25)]; corr01 = 1/(2*sqrt(1.25)).
#[test]
fn stddev_corr_q2_hand_math() {
    let f = fit_with_varcorr(vec![4.0, 1.0, 1.25]);
    let (sd, corr) = f.stddev_corr(0);
    let sd1 = 1.25_f64.sqrt();
    assert!((sd[0] - 2.0).abs() < 1e-12);
    assert!((sd[1] - sd1).abs() < 1e-12);
    assert_eq!(corr[0][0], 1.0);
    assert_eq!(corr[1][1], 1.0);
    let rho = 1.0 / (2.0 * sd1);
    assert!((corr[0][1] - rho).abs() < 1e-12);
    assert!((corr[1][0] - rho).abs() < 1e-12);
}

/// q=3 hand-computed, chosen specifically to catch a vech-indexing bug:
/// D = [[4,1,2],[1,9,3],[2,3,16]] (sd = [2,3,4], all off-diagonal terms
/// distinct so a transposed/misindexed vech would mismatch). Column-major
/// lower-tri vech walk: c=0 → (D00,D10,D20)=(4,1,2); c=1 → (D11,D21)=(9,3);
/// c=2 → (D22)=(16). vech = [4,1,2,9,3,16], len=6 ⇒ q=3.
#[test]
fn stddev_corr_q3_hand_math() {
    let f = fit_with_varcorr(vec![4.0, 1.0, 2.0, 9.0, 3.0, 16.0]);
    let (sd, corr) = f.stddev_corr(0);
    assert_eq!(sd, vec![2.0, 3.0, 4.0]);
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        assert_eq!(corr[i][i], 1.0);
    }
    // corr(0,1) = D10/(sd0*sd1) = 1/(2*3)
    assert!((corr[0][1] - 1.0 / 6.0).abs() < 1e-12);
    assert!((corr[1][0] - 1.0 / 6.0).abs() < 1e-12);
    // corr(0,2) = D20/(sd0*sd2) = 2/(2*4) = 0.25
    assert!((corr[0][2] - 0.25).abs() < 1e-12);
    assert!((corr[2][0] - 0.25).abs() < 1e-12);
    // corr(1,2) = D21/(sd1*sd2) = 3/(3*4) = 0.25
    assert!((corr[1][2] - 0.25).abs() < 1e-12);
    assert!((corr[2][1] - 0.25).abs() < 1e-12);
}

/// assemble_varcorr emits one block per grouping in declaration order:
/// primary q=2 (vech [2,0.5,1]) then one scalar extra (θ=0.7, q=1) → D=0.49.
#[test]
fn assemble_varcorr_one_block_per_grouping() {
    let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(
        &ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 3 },
                    slopes: vec![],
                }],
            }),
        },
        16,
        &[1],
        &[],
    );
    let theta = [2.0, 0.5, 1.0, 0.7];
    let vc = super::assemble_varcorr(&theta, &g, 1.0);
    assert_eq!(vc.len(), 2);
    assert_eq!(vc[0], vec![4.0, 1.0, 1.25]);
    assert!((vc[1][0] - 0.49).abs() < 1e-12, "extra D {}", vc[1][0]);
}

#[test]
fn spec_sized_from_ids_derives_counts() {
    let re = ReStructure {
        sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
        slopes: vec![],
        extra_groupings: vec![
            Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![],
            },
            Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 1 },
                slopes: vec![],
            },
        ],
    };
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(re),
    };
    let ids = GroupIds {
        primary: vec![0, 1, 2, 0, 1, 2], // 3 primary levels
        extra: vec![vec![0, 0, 1, 1, 2, 2], vec![0, 1, 2, 3, 4, 5]], // crossed 3, nested 6 children
    };
    let (sized, _ids, _perm) = super::spec_sized_from_ids(&model, &ids);
    let sre = sized.re.unwrap();
    assert_eq!(sre.sizing, Sizing::FixedClusters { n_clusters: 3 });
    assert_eq!(
        sre.extra_groupings[0].relation,
        GroupingRelation::Crossed { n_clusters: 3 }
    );
    // 6 nested children / 3 parents = 2 per parent.
    assert_eq!(
        sre.extra_groupings[1].relation,
        GroupingRelation::NestedWithin { n_per_parent: 2 }
    );
}

/// UNBALANCED nesting: 3 parents with 1, 2, and 3 distinct children
/// respectively, primary ids ascending with the widest parent last
/// (`0 → {0}`, `1 → {3,4}`, `2 → {6,7,8}` — the contiguous-per-parent-block
/// layout the formula frontend's `grouping_ids` now emits, padded to width 3).
#[test]
fn spec_sized_from_ids_nested_unbalanced_uses_true_max_per_parent() {
    let re = ReStructure {
        sizing: Sizing::FixedClusters { n_clusters: 1 },
        slopes: vec![],
        extra_groupings: vec![Grouping {
            relation: GroupingRelation::NestedWithin { n_per_parent: 1 },
            slopes: vec![],
        }],
    };
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(re),
    };
    // primary id 0: 1 row → child {0}; id 1: 2 rows → children {3,4};
    // id 2: 3 rows → children {6,7,8}.
    let ids = GroupIds {
        primary: vec![0, 1, 1, 2, 2, 2],
        extra: vec![vec![0, 3, 4, 6, 7, 8]],
    };
    let (sized, _ids, _perm) = super::spec_sized_from_ids(&model, &ids);
    let sre = sized.re.unwrap();
    assert_eq!(sre.sizing, Sizing::FixedClusters { n_clusters: 3 });
    assert_eq!(
        sre.extra_groupings[0].relation,
        GroupingRelation::NestedWithin { n_per_parent: 3 }
    );
}

/// Same unbalanced shape, but the WIDEST parent is first in primary order
/// (id 0 → 3 children `{0,1,2}`, id 1 → 2 `{3,4}`, id 2 → 1 `{5}`). The old
/// `children.div_ceil(n_primary)` formula computed `⌈6/3⌉ = 2` from the
/// global `max(extra)+1 = 6` — under-sizing parent 0's true 3-wide block
/// because the fullest parent isn't the one that sets the global max id.
/// The true per-parent count (grouping rows by `(primary, extra)` pairs)
/// gets this right regardless of which parent is fullest.
#[test]
fn spec_sized_from_ids_nested_unbalanced_first_parent_widest() {
    let re = ReStructure {
        sizing: Sizing::FixedClusters { n_clusters: 1 },
        slopes: vec![],
        extra_groupings: vec![Grouping {
            relation: GroupingRelation::NestedWithin { n_per_parent: 1 },
            slopes: vec![],
        }],
    };
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(re),
    };
    let ids = GroupIds {
        primary: vec![0, 0, 0, 1, 1, 2],
        extra: vec![vec![0, 1, 2, 3, 4, 5]],
    };
    let (sized, _ids, _perm) = super::spec_sized_from_ids(&model, &ids);
    let sre = sized.re.unwrap();
    assert_eq!(
        sre.extra_groupings[0].relation,
        GroupingRelation::NestedWithin { n_per_parent: 3 }
    );
}

/// A `q_g = 5` (intercept + 4 slopes) extra grouping is over the `MAX_EXTRA_Q = 4`
/// NoZ-scratch envelope and routes to Sparse (d1 #2). The sparse numeric path
/// makes this a *supported* design — it routes to Sparse and
/// `fit_cold` runs the sparse solver (returning a degenerate non-converged `Fit`
/// on this n=0 input) instead of hitting the removed stub.
#[test]
fn fit_extra_grouping_q_too_large_routes_sparse() {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q
            }],
        }),
    };
    assert!(matches!(classify_design(&model, 1), Solver::Sparse));
    // n=0: bypasses rank-deficiency detection (guarded by `if n > 0 && p > 0`)
    // so routing reaches classify_design → Sparse → fit_mle_sparse, which now
    // runs (no PD data at n=0 ⇒ non-converged) rather than panicking.
    let n = 0;
    let p = 5; // slopes [1,2,3,4] in-bounds; q_g=5 over the NoZ envelope
    let fit = fit_cold(
        &[],
        &[],
        n,
        p,
        &model,
        &GroupIds::from_sizing(model.re.as_ref().unwrap(), n),
        &FitOptions {
            target_indices: vec![1],
            ..FitOptions::default()
        },
    );
    assert!(!fit.converged());
}

/// d1 #2: 7 crossed extras exceed `MAX_EXTRA_GROUPINGS = 6`, the NoZ-scratch
/// envelope. Over-envelope-by-count designs are now supported: `classify_design`
/// routes them to Sparse, and the sparse path builds its own cap-free structures
/// (`SparseLmmWorkspace::new` no longer calls `add_rows_multi`, and
/// `from_cluster_spec_ext`'s `n_extras <= MAX_EXTRA_GROUPINGS` guard is gone).
/// So `fit_cold` runs the sparse solver rather than panicking. Mirrors the
/// sibling `fit_extra_grouping_q_too_large_routes_sparse` (over-envelope by width).
#[test]
fn fit_too_many_extra_groupings_routes_sparse() {
    let extra_groupings: Vec<Grouping> = (0..7)
        .map(|_| Grouping {
            relation: GroupingRelation::Crossed { n_clusters: 2 },
            slopes: vec![],
        })
        .collect();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings,
        }),
    };
    assert!(matches!(classify_design(&model, 1), Solver::Sparse));
    let (n, p) = (8, 1);
    let x = vec![1.0f64; n * p];
    let y = vec![0.0f64; n];
    let ids = GroupIds {
        primary: vec![0; n],
        extra: vec![vec![0; n]; 7],
    };
    // Runs the sparse solver end-to-end without panic; this degenerate
    // (all-zero y) input is non-converged but must return a `Fit`.
    let fit = fit_cold(
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
    assert_eq!(fit.beta.len(), p);
}

/// Anti-panic floor: an over-envelope NON-GAUSSIAN mixed model must return a
/// `Fit`, never panic — for both over-envelope shapes (over-count: 7 crossed
/// extras > MAX_EXTRA_GROUPINGS; over-width: a q_g=5 slope block >
/// MAX_EXTRA_Q) across the wired non-Gaussian families. Holds whether the
/// over-envelope design routes to a real sparse solver or a non-converged
/// placeholder — either way, no panic.
#[test]
fn fit_over_envelope_non_gaussian_never_panics() {
    let families = [
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
    ];
    for family in families {
        // Over-count: 7 crossed intercept-only extras.
        let extra_groupings: Vec<Grouping> = (0..7)
            .map(|_| Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 2 },
                slopes: vec![],
            })
            .collect();
        let model = ModelSpec {
            family,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings,
            }),
        };
        assert!(matches!(classify_design(&model, 1), Solver::Sparse));
        let (n, p) = (8, 1);
        let x = vec![1.0f64; n * p];
        let y = vec![1.0f64; n];
        let ids = GroupIds {
            primary: vec![0; n],
            extra: vec![vec![0; n]; 7],
        };
        let fit = fit_cold(
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
        assert_eq!(fit.beta.len(), p, "{family:?} over-count returns a Fit");

        // Over-width: one crossed extra with a width-5 slope block (q_g = 5).
        let model_w = ModelSpec {
            family,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 4 },
                    slopes: vec![1, 2, 3, 4],
                }],
            }),
        };
        assert!(matches!(classify_design(&model_w, 1), Solver::Sparse));
        let (n, p) = (16, 5);
        let mut st = 11u64;
        let x: Vec<f64> = (0..n)
            .flat_map(|_| {
                let mut r = [0.0f64; 5];
                r[0] = 1.0;
                for v in r[1..].iter_mut() {
                    *v = lcg(&mut st);
                }
                r
            })
            .collect();
        let y = vec![1.0f64; n];
        let ids = GroupIds {
            primary: (0..n as u32).map(|i| i % 4).collect(),
            extra: vec![(0..n as u32).map(|i| (i / 4) % 4).collect()],
        };
        let fit = fit_cold(
            &x,
            &y,
            n,
            p,
            &model_w,
            &ids,
            &FitOptions {
                target_indices: vec![1],
                ..FitOptions::default()
            },
        );
        assert_eq!(fit.beta.len(), p, "{family:?} over-width returns a Fit");
    }
}

/// Warm-path wrapper equivalence: `fit_cold(..)` and `fit_warm(.., None, ..)`
/// return a byte-identical `Fit` — locks "one implementation, two names".
/// Uses the intercept-only 6-cluster LMM.
#[test]
fn fit_cold_equals_fit_warm_none() {
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
    let opts = FitOptions {
        target_indices: vec![1, 2],
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let warm_none = fit_warm(&x, &y, n, p, &model, &ids, None, &opts);
    // Bitwise equality (not PartialEq): non-target SE slots are NaN, and
    // NaN != NaN under `==` — but the two Fits share one code path, so their
    // bit patterns (NaNs included) must match exactly.
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&cold.beta), bits(&warm_none.beta));
    assert_eq!(bits(&cold.se), bits(&warm_none.se));
    assert_eq!(bits(&cold.tau2), bits(&warm_none.tau2));
    assert_eq!(cold.dispersion.to_bits(), warm_none.dispersion.to_bits());
    assert_eq!(cold.converged(), warm_none.converged());
}

/// Warm-path start-independence: on an LMM the MLE is start-independent, so a
/// warm fit from a perturbed `StartValues.theta` reaches the same β̂ as the cold
/// fit to optimizer tolerance — a good start shortens the path without moving the
/// MLE. n_theta=1 (intercept-only 6-cluster), so theta=[5.0] is
/// well off the THETA0 blind start.
#[test]
fn fit_warm_start_reaches_cold_beta() {
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
    let opts = FitOptions {
        target_indices: vec![1, 2],
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let start = StartValues {
        beta: vec![0.0; p],
        theta: vec![5.0],
    };
    let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(&start), &opts);
    assert!(
        cold.converged() && warm.converged(),
        "both fits must converge"
    );
    for j in [1usize, 2] {
        let (a, b) = (cold.beta[j], warm.beta[j]);
        let d = (a - b).abs();
        assert!(
            d <= 1e-7 || d <= 1e-6 * a.abs().max(b.abs()),
            "LMM MLE must be start-independent: β[{j}] cold {a} vs warm {b}"
        );
    }
}

/// Same start-independence on the LMM, but from a θ-only start (`beta` empty —
/// the per-component cold marker the ports need for lme4's
/// `start = list(theta = …)`). The LMM ignores a β start anyway (β is solved
/// exactly given θ), so this pins that an empty β is accepted rather than
/// faulting the entry assert, and that θ still threads through.
#[test]
fn fit_warm_theta_only_start_reaches_cold_beta() {
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
    let opts = FitOptions {
        target_indices: vec![1, 2],
        ..FitOptions::default()
    };
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let start = StartValues {
        beta: vec![],
        theta: vec![5.0],
    };
    let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(&start), &opts);
    assert!(
        cold.converged() && warm.converged(),
        "both fits must converge"
    );
    for j in [1usize, 2] {
        let (a, b) = (cold.beta[j], warm.beta[j]);
        let d = (a - b).abs();
        assert!(
            d <= 1e-7 || d <= 1e-6 * a.abs().max(b.abs()),
            "LMM MLE must be start-independent: β[{j}] cold {a} vs warm {b}"
        );
    }
}

/// Map raw cluster labels to dense 0-based ids (first-seen order) + the count.
pub(super) fn dense_ids(raw: &[u32]) -> (Vec<u32>, usize) {
    use std::collections::HashMap;
    let mut map: HashMap<u32, u32> = HashMap::new();
    let mut next = 0u32;
    let ids: Vec<u32> = raw
        .iter()
        .map(|&r| {
            *map.entry(r).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect();
    (ids, next as usize)
}

/// Map string factor labels to dense 0-based ids (first-seen order) + the count.
pub(super) fn dense_str(raw: &[String]) -> (Vec<u32>, usize) {
    use std::collections::HashMap;
    let mut map: HashMap<String, u32> = HashMap::new();
    let mut next = 0u32;
    let ids: Vec<u32> = raw
        .iter()
        .map(|r| {
            *map.entry(r.clone()).or_insert_with(|| {
                let v = next;
                next += 1;
                v
            })
        })
        .collect();
    (ids, next as usize)
}

/// In-envelope designs route NoZ (byte-identical fast path); over-envelope
/// route Sparse. The boundary is the cap edge.
#[test]
fn classify_routes_at_the_cap_edge() {
    // A scalar-intercept LMM (q_p=1, no extras) — deep in-envelope.
    let in_env = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 10 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    assert!(matches!(
        super::classify_design_pub(&in_env, 1),
        super::Solver::NoZ
    ));

    // MAX_EXTRA_GROUPINGS+1 crossed groupings — over-envelope ⇒ Sparse.
    let over = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 10 },
            slopes: vec![],
            extra_groupings: (0..(crate::consts::MAX_EXTRA_GROUPINGS + 1))
                .map(|_| Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 4 },
                    slopes: vec![],
                })
                .collect(),
        }),
    };
    assert!(matches!(
        super::classify_design_pub(&over, 1),
        super::Solver::Sparse
    ));

    // Wide primary slope block past MAX_PRIMARY_Q ⇒ Sparse.
    let wide = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 10 },
            slopes: (1..=crate::consts::MAX_PRIMARY_Q as u32).collect(), // q_p = 1 + MAX_PRIMARY_Q
            extra_groupings: vec![],
        }),
    };
    assert!(matches!(
        super::classify_design_pub(&wide, 1),
        super::Solver::Sparse
    ));
}

/// Total `Crossed` level count past MAX_CROSSED_LEVELS routes Sparse (the
/// dense tail is cubic in the SUM of crossed levels, so two half-cap
/// factors trip it together); at the cap exactly, intercept-only crossed
/// extras stay NoZ. Nested extras don't count toward the sum (they live in
/// the per-family elimination path, not the dense tail).
#[test]
fn classify_routes_many_crossed_levels_to_sparse() {
    let cap = crate::consts::MAX_CROSSED_LEVELS as u32;
    let spec = |extras: Vec<Grouping>| ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 10 },
            slopes: vec![],
            extra_groupings: extras,
        }),
    };
    let crossed = |n_clusters: u32| Grouping {
        relation: GroupingRelation::Crossed { n_clusters },
        slopes: vec![],
    };
    // One factor over the cap ⇒ Sparse.
    let over = spec(vec![crossed(cap + 1)]);
    assert!(matches!(
        super::classify_design_pub(&over, 1),
        super::Solver::Sparse
    ));
    // Sum over factors trips the cap even when each is under it.
    let sum_over = spec(vec![crossed(cap / 2 + 1), crossed(cap / 2 + 1)]);
    assert!(matches!(
        super::classify_design_pub(&sum_over, 1),
        super::Solver::Sparse
    ));
    // Exactly at the cap ⇒ NoZ unchanged.
    let at_cap = spec(vec![crossed(cap)]);
    assert!(matches!(
        super::classify_design_pub(&at_cap, 1),
        super::Solver::NoZ
    ));
    // A many-level NESTED extra doesn't count toward the crossed sum.
    let nested = spec(vec![Grouping {
        relation: GroupingRelation::NestedWithin {
            n_per_parent: cap + 1,
        },
        slopes: vec![],
    }]);
    assert!(matches!(
        super::classify_design_pub(&nested, 1),
        super::Solver::NoZ
    ));
}

/// The measured q_g performance boundary (d2 Phase-1 crossover sweep) for
/// Gaussian, and the only-implemented-route boundary for non-Gaussian: ANY
/// slope-carrying extra grouping routes Sparse. Gaussian intercept-only
/// extras (q_g = 1) stay NoZ (NoZ won 12–15× on the measured slice); the
/// dense NoZ GLMM kernel builds intercept-only extras exclusively, so for
/// non-Gaussian families Sparse is a correctness route, not a perf choice.
#[test]
fn classify_routes_slope_extras_to_sparse_all_families() {
    let spec = |family: Family, extra_slopes: Vec<u32>| ModelSpec {
        family,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 10 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 5 },
                slopes: extra_slopes,
            }],
        }),
    };
    // Gaussian, slope-carrying extra (q_g = 2, in-envelope) ⇒ Sparse.
    let g_slope = spec(Family::Gaussian, vec![1]);
    assert!(matches!(
        super::classify_design_pub(&g_slope, 1),
        super::Solver::Sparse
    ));
    // Gaussian, intercept-only extra ⇒ NoZ.
    let g_int = spec(Family::Gaussian, vec![]);
    assert!(matches!(
        super::classify_design_pub(&g_int, 1),
        super::Solver::NoZ
    ));
    // Non-Gaussian, slope-carrying extra ⇒ Sparse (the only kernel that fits it).
    let p_slope = spec(
        Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        vec![1],
    );
    assert!(matches!(
        super::classify_design_pub(&p_slope, 1),
        super::Solver::Sparse
    ));
}

/// A fixed-only model always routes NoZ (no RE to make sparse).
#[test]
fn classify_fixed_only_is_noz() {
    let ols = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    assert!(matches!(
        super::classify_design_pub(&ols, 1),
        super::Solver::NoZ
    ));
}

/// Tier-0 short-circuit: a fixed-only
/// fit (no random effects, no θ to search) must run the direct GLM/OLS path and
/// enter the BOBYQA search ZERO times. `fit_cold`'s `(family, None)` dispatch arms
/// route straight to `fit_ols`/`fit_glm`/`fit_glm_nb`, each of which hard-sets
/// `n_eval: 0`; this pins that invariant end-to-end (through the whole cold path,
/// not just `classify_design`) across every wired fixed-only family, so a future
/// change that accidentally sent a no-Z model through the optimizer would trip it.
/// NB's outer θ↔β alternation is a golden-section search, not BOBYQA — it also
/// leaves `n_eval` at the inner IRLS 0.
#[test]
fn fixed_only_fit_runs_zero_bobyqa_evals() {
    let n = 24;
    let p = 2;
    let mut st = 3u64;
    let mut x = vec![0.0f64; n * p];
    let mut xv = vec![0.0f64; n];
    for i in 0..n {
        let x1 = lcg(&mut st);
        x[i * p] = 1.0;
        x[i * p + 1] = x1;
        xv[i] = x1;
    }
    // Per-family responses in each family's valid range so the direct fit is a
    // real convergent fit, not a degenerate one — the n_eval==0 invariant holds
    // regardless, but a healthy fit makes the guard exercise the live path.
    let families: [(Family, Vec<f64>); 5] = [
        (
            Family::Gaussian,
            (0..n).map(|i| 0.5 + 0.4 * xv[i]).collect(),
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            (0..n).map(|i| f64::from(u32::from(xv[i] > 0.0))).collect(),
        ),
        (
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            (0..n).map(|i| f64::from(1 + (i % 4) as u32)).collect(),
        ),
        (
            Family::Gamma {
                link: crate::GammaLink::Log,
            },
            (0..n).map(|i| 1.0 + 0.5 * (xv[i] + 1.0)).collect(),
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            (0..n).map(|i| f64::from(1 + (i % 4) as u32)).collect(),
        ),
    ];
    for (family, y) in families {
        let model = ModelSpec { family, re: None };
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
        assert_eq!(
            f.n_eval, 0,
            "{family:?} fixed-only fit must enter BOBYQA zero times"
        );
    }
}

// ---------------------------------------------------------------------------
// `Fit::vcov` — gating tests for the `targets`-subset carve-out.
// ---------------------------------------------------------------------------

/// `vcov` must be a symmetric p×p whose diagonal IS `se²`, on every estimator
/// path — the spec's plan gate 2: no path may report a finite `se[j]` next to a
/// NaN `vcov[j][j]`. Covers OLS / GLM / LMM / GLMM (both `WaldSe` arms), since
/// each sources `vcov` from a different matrix.
fn assert_vcov_agrees_with_se(fit: &Fit, p: usize, ctx: &str) {
    assert_eq!(fit.vcov.len(), p, "{ctx}: vcov is not p rows");
    for row in &fit.vcov {
        assert_eq!(row.len(), p, "{ctx}: vcov is not p×p");
    }
    for j in 0..p {
        if fit.se[j].is_finite() {
            assert!(
                fit.vcov[j][j].is_finite(),
                "{ctx}: finite se[{j}] alongside NaN vcov[{j}][{j}]"
            );
            // se = sqrt(diag) — same quantity, so this is tight, not a tolerance.
            let want = fit.se[j] * fit.se[j];
            assert!(
                (fit.vcov[j][j] - want).abs() <= 1e-9 * want.abs().max(1e-12),
                "{ctx}: vcov[{j}][{j}] = {} vs se[{j}]² = {want}",
                fit.vcov[j][j]
            );
        }
    }
    for i in 0..p {
        for j in 0..p {
            if fit.vcov[i][j].is_finite() || fit.vcov[j][i].is_finite() {
                assert_eq!(fit.vcov[i][j], fit.vcov[j][i], "{ctx}: vcov not symmetric");
            }
        }
    }
}

#[test]
fn vcov_diagonal_is_se_squared_on_every_path() {
    let (x, y, n, p) = lmm_hand_dataset();
    let all: Vec<u32> = (0..p as u32).collect();
    let opts = FitOptions {
        target_indices: all.clone(),
        ..FitOptions::default()
    };

    // OLS — Gaussian, no RE.
    let ols = fit_cold(
        &x,
        &y,
        n,
        p,
        &ModelSpec {
            family: Family::Gaussian,
            re: None,
        },
        &GroupIds::default(),
        &opts,
    );
    assert!(ols.converged());
    assert_vcov_agrees_with_se(&ols, p, "ols");

    // LMM — Gaussian, 6 clusters.
    let ids = GroupIds {
        primary: (0..n).map(|i| (i % 6) as u32).collect(),
        extra: vec![],
    };
    let lmm_model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 6 },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let lmm = fit_cold(&x, &y, n, p, &lmm_model, &ids, &opts);
    assert!(lmm.converged());
    assert_vcov_agrees_with_se(&lmm, p, "lmm");

    // GLM / GLMM — Bernoulli response off the same design.
    let yb: Vec<f64> = y.iter().map(|&v| f64::from(v > 0.5)).collect();
    let binom = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let glm = fit_cold(
        &x,
        &yb,
        n,
        p,
        &ModelSpec {
            family: binom,
            re: None,
        },
        &GroupIds::default(),
        &opts,
    );
    assert!(glm.converged());
    assert_vcov_agrees_with_se(&glm, p, "glm");

    // GLMM on both SE arms — each sources vcov from a different matrix
    // (FD-Hessian β block vs Schur inverse).
    for wald_se in [WaldSe::Hessian, WaldSe::Rx] {
        let glmm = fit_cold(
            &x,
            &yb,
            n,
            p,
            &ModelSpec {
                family: binom,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 6 },
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            },
            &ids,
            &FitOptions {
                target_indices: all.clone(),
                wald_se,
                ..FitOptions::default()
            },
        );
        assert!(glmm.converged(), "glmm {wald_se:?} did not converge");
        assert_vcov_agrees_with_se(&glmm, p, &format!("glmm {wald_se:?}"));
    }
}

/// The sanctioned `targets=` carve-out (gate 2's one exception): under a target
/// subset, `vcov` is finite exactly on the target block and NaN outside it —
/// never a finite `se` next to a NaN variance, and never a fabricated
/// covariance for a coefficient whose variance was never computed.
#[test]
fn vcov_is_nan_outside_the_target_block() {
    let (x, y, n, p) = lmm_hand_dataset();
    let fit = fit_cold(
        &x,
        &y,
        n,
        p,
        &ModelSpec {
            family: Family::Gaussian,
            re: None,
        },
        &GroupIds::default(),
        &FitOptions {
            target_indices: vec![2], // only the last predictor
            ..FitOptions::default()
        },
    );
    assert!(fit.converged());
    assert!(fit.se[2].is_finite() && fit.vcov[2][2].is_finite());
    assert!(fit.se[0].is_nan() && fit.se[1].is_nan());
    for j in [0usize, 1] {
        for i in 0..p {
            assert!(
                fit.vcov[i][j].is_nan() && fit.vcov[j][i].is_nan(),
                "vcov must be NaN outside the target block at ({i},{j})"
            );
        }
    }
    assert_vcov_agrees_with_se(&fit, p, "targets subset");
}

/// An aliased (rank-deficient) column has no coefficient, so it has no
/// covariance either: its whole `vcov` row/column is NaN, exactly as its `se`
/// slot is, while the surviving block stays finite and self-consistent.
#[test]
fn vcov_rows_are_nan_for_aliased_columns() {
    let (x, y, n, _) = lmm_hand_dataset();
    // Widen the design with a duplicate of column 1 → aliased on it.
    let p = 4;
    let mut xa = vec![0.0f64; n * p];
    for i in 0..n {
        xa[i * p] = x[i * 3];
        xa[i * p + 1] = x[i * 3 + 1];
        xa[i * p + 2] = x[i * 3 + 2];
        xa[i * p + 3] = x[i * 3 + 1]; // exact copy of column 1
    }
    let fit = fit_cold(
        &xa,
        &y,
        n,
        p,
        &ModelSpec {
            family: Family::Gaussian,
            re: None,
        },
        &GroupIds::default(),
        &FitOptions {
            target_indices: (0..p as u32).collect(),
            ..FitOptions::default()
        },
    );
    assert!(
        fit.aliased()[3],
        "duplicate column must be detected aliased"
    );
    for i in 0..p {
        assert!(
            fit.vcov[i][3].is_nan(),
            "aliased column keeps a NaN vcov col"
        );
        assert!(
            fit.vcov[3][i].is_nan(),
            "aliased column keeps a NaN vcov row"
        );
    }
    assert_vcov_agrees_with_se(&fit, p, "aliased");
}

// ---------------------------------------------------------------------------
// The public `Diagnostics` surface (0.2.0): the three moved fields, the two
// reshaped ones, and the notes channel.
// ---------------------------------------------------------------------------

/// A clean OLS design and a rank-deficient one, each read through BOTH paths —
/// `fit.diagnostics.<field>` and the forwarding accessor. The point is not the
/// values (other tests pin those) but that the two paths are the same storage:
/// if a future change ever re-adds a top-level copy of one of these, one of the
/// four `assert_eq!`s below stops holding.
#[test]
fn diagnostics_moved_fields_agree_through_both_paths() {
    let (n, p) = (12usize, 3usize);
    let mut st = 11u64;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = lcg(&mut st);
        let b = lcg(&mut st);
        x.extend_from_slice(&[1.0, a, b]);
        y.push(0.3 + 1.1 * a - 0.7 * b + 0.05 * ((i % 3) as f64 - 1.0));
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        target_indices: (0..p as u32).collect(),
        ..FitOptions::default()
    };
    let fit = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts);
    assert_eq!(fit.diagnostics.converged, fit.converged());
    assert_eq!(fit.diagnostics.singular, fit.singular());
    assert_eq!(fit.diagnostics.aliased, fit.aliased());
    assert!(fit.converged() && !fit.singular());
    assert_eq!(fit.aliased(), vec![false; p]);

    // Column 2 duplicated onto column 1 ⇒ the alias gate drops it. `aliased` is
    // the one moved field the alias gate fills from ABOVE the fitting routes,
    // so it needs its own true-valued case.
    let mut xd = Vec::with_capacity(n * p);
    for i in 0..n {
        let a = x[i * p + 1];
        xd.extend_from_slice(&[1.0, a, a]);
    }
    let dup = fit_cold(&xd, &y, n, p, &model, &GroupIds::default(), &opts);
    assert_eq!(dup.diagnostics.aliased, dup.aliased());
    assert_eq!(dup.aliased(), vec![false, false, true]);
    assert!(dup.converged(), "the reduced model fits");
}

/// `Boundary` at both ends of the range the dense LMM route can report:
/// the deterministic τ̂=0 pin fixture (`fit_lmm_weighted_boundary_matches_wls`'s
/// design, same construction) lands `AtBoundary`, sleepstudy lands `Interior`.
/// Also pins the one place `singular` and `boundary` are NOT interchangeable —
/// `singular` additionally carries the post-hoc negligible-component check.
#[test]
fn diagnostics_boundary_reports_both_ends() {
    let n = 48usize;
    let n_clusters = 6usize;
    let mut st = 7u64;
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        // ±0.8 cancels exactly per cluster ⇒ the between-cluster variance MLE
        // is 0 and the optimizer pins it there.
        let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
        y[i] = 0.5 + 0.4 * x1 + e;
    }
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
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    let pinned = fit_cold(
        &x,
        &y,
        n,
        2,
        &model,
        &GroupIds {
            primary: ids,
            extra: vec![],
        },
        &opts,
    );
    assert!(pinned.converged(), "a boundary fit still converges");
    assert_eq!(pinned.diagnostics.boundary, Boundary::AtBoundary);
    assert!(pinned.singular());

    let (xs, ys, ns, ps) = lmm_hand_dataset();
    let ids_s: Vec<u32> = (0..ns).map(|i| (i % 6) as u32).collect();
    let interior = fit_cold(
        &xs,
        &ys,
        ns,
        ps,
        &ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 6 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        },
        &GroupIds {
            primary: ids_s,
            extra: vec![],
        },
        &FitOptions {
            target_indices: (0..ps as u32).collect(),
            ..FitOptions::default()
        },
    );
    assert!(interior.converged());
    assert_eq!(interior.diagnostics.boundary, Boundary::Interior);
    assert!(interior.diagnostics.pinned.is_empty(), "nothing pinned");
}

/// `pinned[g][i]` must pair with `stddev_corr(g).0[i]` — the whole point of the
/// alignment. Asserted on the one shape where a wrong
/// mapping is VISIBLE — a q=2 primary block where the SLOPE component pins and
/// the intercept stays interior, so swapping the two bits (or reading the vech
/// off-diagonal as a component) flips the assertion.
///
/// The design is `lmm::tests::zero_slope_variance_pins_slope_component`'s,
/// driven through `fit_cold` instead of the kernel: x1 is a within-cluster
/// antithetic ±1 pattern carrying a real fixed slope but no cluster-varying
/// one, and the ±0.8 period-4 residual is in quadrature against it, so every
/// cluster has Σresid = 0 and Σx1·resid = 0 exactly. The planted u₀ keeps the
/// intercept component off the boundary.
#[test]
fn diagnostics_pinned_aligns_with_varcorr_blocks() {
    let (nc, per) = (16usize, 16usize);
    let n = nc * per;
    let mut st = 5u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
    let mut x = vec![0.0f64; n * 2];
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for (c, &uc) in u0.iter().enumerate() {
        for k in 0..per {
            let i = c * per + k;
            ids[i] = c as u32;
            let x1 = if k % 2 == 0 { 1.0 } else { -1.0 };
            let e = if (k / 2) % 2 == 0 { 0.8 } else { -0.8 };
            x[i * 2] = 1.0;
            x[i * 2 + 1] = x1;
            y[i] = 0.5 + 0.4 * x1 + uc + e;
        }
    }
    let fit = fit_cold(
        &x,
        &y,
        n,
        2,
        &ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: nc as u32,
                },
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        &GroupIds {
            primary: ids,
            extra: vec![],
        },
        &FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        },
    );
    assert!(fit.converged());
    assert_eq!(fit.diagnostics.pinned.len(), fit.varcorr.len());
    let (sd, _) = fit.stddev_corr(0);
    assert_eq!(fit.diagnostics.pinned[0].len(), sd.len());
    assert_eq!(
        fit.diagnostics.pinned[0],
        vec![false, true],
        "the SLOPE component is the pinned one; stddev {sd:?}"
    );
    // The pairing, stated as the invariant rather than as two literals: the
    // pinned slot's stddev collapses and the unpinned one's does not.
    //
    // NOT exactly zero, and that is not slack in the assertion. The pin fixes
    // the DIAGONAL θ (λ₁₁ = 0 exactly), while `stddev_corr(0).0[1]` is
    // √D₁₁ = √(λ₁₀² + λ₁₁²) — so it inherits whatever the off-diagonal λ₁₀
    // settled on, measured here at 4.4e-9 against a 0.6-scale intercept
    // component. A q≥2 pinned component reads as negligible, not as 0.0.
    assert!(
        sd[1] / sd[0] < 1e-6,
        "pinned component's stddev must collapse: {sd:?}"
    );
    assert!(sd[0] > 0.0, "interior component's stddev is positive");
}

/// `Note::IllConditioned` through the STABLE entry, positive and negative.
/// Positive: `x = [1, a, a+δ]` where δ lives entirely on rows carrying weight
/// 1e-11 — full-rank raw (so the alias gate passes it through untouched) and
/// near-singular once weighted, which is the case nothing upstream of the fit
/// can see. Negative: the same design at unit weights raises nothing.
#[test]
fn diagnostics_ill_conditioned_note_through_fit_cold() {
    let (n, p, split) = (60usize, 3usize, 40usize);
    const WSMALL: f64 = 1e-11;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::with_capacity(n);
    for i in 0..n {
        let a = ((i * 13) % 17) as f64 - 8.0;
        let delta = if i < split { 0.0 } else { 1.0 };
        x.extend_from_slice(&[1.0, a, a + delta]);
        y.push(0.5 + 1.3 * a + 0.477 * (a + delta) + ((i % 3) as f64 - 1.0));
        w.push(if i < split { 1.0 } else { WSMALL });
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let targets: Vec<u32> = (0..p as u32).collect();
    let flagged = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            target_indices: targets.clone(),
            weights: Some(w),
            ..FitOptions::default()
        },
    );
    assert!(flagged.converged(), "fit-and-flag: the design is returned");
    assert_eq!(flagged.aliased(), vec![false; p], "nothing was dropped");
    assert_eq!(flagged.diagnostics.notes.len(), 1);
    let Note::IllConditioned { columns, pivot } = &flagged.diagnostics.notes[0] else {
        panic!(
            "expected IllConditioned, got {:?}",
            flagged.diagnostics.notes[0]
        );
    };
    assert_eq!(columns, &vec![2u32], "the later column of the pair");
    assert!(
        *pivot < crate::ols::PIVOT_MIN,
        "the note carries the measured ratio, got {pivot}"
    );

    let clean = fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &GroupIds::default(),
        &FitOptions {
            target_indices: targets,
            ..FitOptions::default()
        },
    );
    assert!(clean.converged());
    assert!(
        clean.diagnostics.notes.is_empty(),
        "unweighted, the same design is well-conditioned"
    );
    assert!(clean.diagnostics.pinned.is_empty(), "no RE, nothing to pin");
}

/// A grouping `g` with a random slope on `x`, `x` built at the requested RMS
/// scale (constant across groups; the note only reads the column, not the
/// fit). Mirrors `sparse::tests::sparse_binomial_bigsd_formula_routes_sparse`'s
/// `formula::lower` setup — the smallest fixture that exercises the real
/// lowering path rather than a hand-built `ModelSpec`.
#[cfg(feature = "formula")]
fn scale_spread_table(x_scale: f64) -> (crate::formula::Table, usize) {
    use crate::formula::{Column, Table};

    let n_groups = 5;
    let per_group = 8;
    let n = n_groups * per_group;
    let mut y = Vec::with_capacity(n);
    let mut x = Vec::with_capacity(n);
    let mut g_labels = Vec::with_capacity(n);
    for gi in 0..n_groups {
        for j in 0..per_group {
            let jitter = j as f64 - (per_group as f64 - 1.0) / 2.0;
            let xv = x_scale + jitter;
            x.push(xv);
            y.push(1.0 + 0.1 * xv / x_scale + 0.05 * gi as f64);
            g_labels.push(format!("g{gi}"));
        }
    }
    let table = Table {
        n,
        columns: vec![
            ("y".to_string(), Column::Numeric(y)),
            ("x".to_string(), Column::Numeric(x)),
            ("g".to_string(), Column::factor_from_labels(&g_labels)),
        ],
    };
    (table, n)
}

#[cfg(feature = "formula")]
#[test]
fn re_design_scale_spread_note_fires_on_mismatched_slope_scale() {
    let (table, _n) = scale_spread_table(1.0e4);
    let lo = crate::formula::lower("y ~ x + (1 + x | g)", &table, Family::Gaussian).unwrap();

    let spread: Vec<&Note> = lo
        .notes
        .iter()
        .filter(|n| matches!(n, Note::ReDesignScaleSpread { .. }))
        .collect();
    assert_eq!(
        spread.len(),
        1,
        "expected exactly one note, got {:?}",
        lo.notes
    );
    match spread[0] {
        Note::ReDesignScaleSpread { grouping, ratio } => {
            assert_eq!(grouping, "g");
            // RMS(x) ~ 1e4 against the implicit intercept's 1.0 — same decade.
            assert!(
                (1.0e3..1.0e5).contains(ratio),
                "ratio {ratio} not in the expected decade"
            );
        }
        other => panic!("expected ReDesignScaleSpread, got {other:?}"),
    }

    // The note is a lowering-time observation, independent of whether the
    // solver converges — the pipeline still runs end to end through fit_cold.
    let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
}

#[cfg(feature = "formula")]
#[test]
fn re_design_scale_spread_note_absent_on_well_scaled_design() {
    let (table, _n) = scale_spread_table(4.0);
    let lo = crate::formula::lower("y ~ x + (1 + x | g)", &table, Family::Gaussian).unwrap();

    assert!(
        !lo.notes
            .iter()
            .any(|n| matches!(n, Note::ReDesignScaleSpread { .. })),
        "well-scaled design should not warn, got {:?}",
        lo.notes
    );
}
