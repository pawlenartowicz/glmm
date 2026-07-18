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

/// Parse a `cluster,x,grp,y` sim CSV → (X=[1,x,grp_b], y, dense cluster ids, n_clusters).
pub(super) fn sim_clustered(csv: &str) -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
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

    assert!(f.converged, "reduced OLS must converge");
    assert_eq!(
        f.aliased,
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

#[derive(serde::Deserialize)]
struct ColEst {
    beta: Vec<Option<f64>>,
}

#[derive(serde::Deserialize)]
struct ColGolden {
    estimates: ColEst,
}

/// Gap #4 oracle: near-collinear `y ~ 1 + x1 + x2 + x3` (x3 ≈ x1+x2) vs R's
/// column-drop-and-fit (`parity/goldens/sim_collinear_glm.json`). glmm must
/// drop the SAME column R marks `NA`, mark it in `Fit::aliased`, and match the
/// retained β. The oracle is sacred.
#[test]
fn fit_sim_collinear_matches_lme4_drop() {
    let raw = include_str!("../../parity/goldens/sim_collinear_glm.json");
    let gold: ColGolden = serde_json::from_str(raw).expect("golden JSON parses");

    let csv = include_str!("../../parity/data_simulated/sim_collinear.csv");
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

    assert!(f.converged, "reduced fit converges");
    // R's aliased column = the beta slot that is null.
    let r_aliased: Vec<bool> = gold.estimates.beta.iter().map(|b| b.is_none()).collect();
    assert_eq!(
        f.aliased, r_aliased,
        "glmm must drop the same column R does"
    );
    for (j, rb) in gold.estimates.beta.iter().enumerate() {
        match rb {
            Some(v) => assert!(
                (f.beta[j] - v).abs() / v.abs().max(1e-6) < 1e-3,
                "β{j} {} vs {v}",
                f.beta[j]
            ),
            None => assert!(f.beta[j].is_nan(), "β{j} must be NaN (aliased)"),
        }
    }
}

/// Deterministic pseudo-data (NR LCG), uniform in (−1, 1). Mirrors the
/// LCG in lmm.rs tests so the smoke dataset behaves the same way.
pub(super) fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

/// n=48, p=3, 6 clusters — same shape as lmm.rs's `hand_dataset`, adapted
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
    let vech = super::common::varcorr_block(&[2.0, 0.5, 1.0], 2, 1.0);
    assert_eq!(vech.len(), 3);
    assert!((vech[0] - 4.0).abs() < 1e-12, "D00 {}", vech[0]);
    assert!((vech[1] - 1.0).abs() < 1e-12, "D10 {}", vech[1]);
    assert!((vech[2] - 1.25).abs() < 1e-12, "D11 {}", vech[2]);
    let scaled = super::common::varcorr_block(&[2.0, 0.5, 1.0], 2, 3.0);
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
        converged: true,
        varcorr: vec![vech],
        stddev_se: vec![],
        aliased: vec![],
        n_eval: 0,
        deviance: f64::NAN,
        singular: false,
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
    let sized = super::spec_sized_from_ids(&model, &ids);
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
    let sized = super::spec_sized_from_ids(&model, &ids);
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
    let sized = super::spec_sized_from_ids(&model, &ids);
    let sre = sized.re.unwrap();
    assert_eq!(
        sre.extra_groupings[0].relation,
        GroupingRelation::NestedWithin { n_per_parent: 3 }
    );
}

/// Mirror of MCPower's `extra_grouping_rejects_too_many_slopes` contract test:
/// a `q_g = 5` (intercept + 4 slopes) extra grouping is over the `MAX_EXTRA_Q = 4`
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
    assert!(!fit.converged);
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
    assert_eq!(cold.converged, warm_none.converged);
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
    assert!(cold.converged && warm.converged, "both fits must converge");
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

/// Tier-0 short-circuit (tiered-optimizer-architecture.md action 1): a fixed-only
/// fit (no random effects, no θ to search) must run the direct GLM/OLS path and
/// enter the BOBYQA search ZERO times. `fit_cold`'s `(family, None)` dispatch arms
/// route straight to `fit_ols`/`fit_glm`/`fit_glm_nb`, each of which hard-sets
/// `n_eval: 0`; this pins that invariant end-to-end (through the whole cold path,
/// not just `classify_design`) across every wired fixed-only family, so a future
/// change that accidentally sent a no-Z model through the optimizer would trip it.
/// NB's outer θ↔β alternation is a scalar Brent search, not BOBYQA — it also
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
    assert!(ols.converged);
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
    assert!(lmm.converged);
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
    assert!(glm.converged);
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
        assert!(glmm.converged, "glmm {wald_se:?} did not converge");
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
    assert!(fit.converged);
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
    assert!(fit.aliased[3], "duplicate column must be detected aliased");
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
