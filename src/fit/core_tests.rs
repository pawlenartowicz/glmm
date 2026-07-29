//! Unified fit-core tests: the reuse gate (`fit_on`-on-reused-ws vs `fit_cold`,
//! near-identity), route pinning (build_workspace classification + fit_on shape
//! panic), and the `FitView` accessor surface. The frozen goldens only ever fit
//! throwaway workspaces at `n == sized shape`, so they cannot catch stale state,
//! `n_max` over-reads, or option-reset misses — this suite is that guard.

use super::{build_workspace, fit_on};
use crate::fit::spec_sized_from_ids_pub;
use crate::test_support::assert_near;
use crate::{
    fit_cold, BinomialLink, Family, FitOptions, GroupIds, Grouping, GroupingRelation, ModelSpec,
    ReStructure, Sizing,
};

/// Tiny deterministic LCG in (−1, 1) — no RNG in the fit path.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

// --- case builders (deterministic, no RNG) ---

fn ols_case() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    FitOptions,
) {
    let n = 30;
    let p = 3;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = i as f64;
        let b = ((i * 5) % 7) as f64 - 3.0;
        x.extend_from_slice(&[1.0, a, b]);
        y.push(0.7 + 1.1 * a - 0.5 * b + ((i % 3) as f64 - 1.0));
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![1, 2],
        ..FitOptions::default()
    };
    (x, y, n, p, model, GroupIds::default(), opts)
}

/// A fixed-effects Poisson(log) dataset (n=30, p=2, `re: None`) — routes to
/// `FitKind::Glm`. Deterministic, no RNG (the LCG is seeded, not random).
fn glm_case() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    FitOptions,
) {
    let n = 30;
    let p = 2;
    let mut st = 7u64;
    let mut x = Vec::with_capacity(n * p);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let x1 = 0.4 * lcg(&mut st);
        x.extend_from_slice(&[1.0, x1]);
        let eta: f64 = 0.5 + 0.6 * x1;
        y.push(eta.exp().round());
    }
    let model = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: None,
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    (x, y, n, p, model, GroupIds::default(), opts)
}

fn lmm_intercept_case() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    FitOptions,
) {
    let n_clusters = 6usize;
    let per = 8usize;
    let n = n_clusters * per;
    let p = 2usize;
    let mut st = 13u64;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids_v = vec![0u32; n];
    for i in 0..n {
        ids_v[i] = (i % n_clusters) as u32;
        // deterministic pseudo-noise via a tiny LCG
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x1 = ((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        let re = 0.3 * ((ids_v[i] as f64) - (n_clusters as f64) / 2.0);
        y[i] = 0.5 + 0.4 * x1 + re + 0.1 * ((i % 5) as f64 - 2.0);
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
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    (x, y, n, p, model, ids, opts)
}

fn glmm_binomial_intercept_case() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    FitOptions,
) {
    let n_clusters = 8usize;
    let per = 16usize;
    let n = n_clusters * per;
    let p = 2usize;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut ids_v = vec![0u32; n];
    for i in 0..n {
        let c = i / per;
        ids_v[i] = c as u32;
        let x1 = ((i % per) as f64) / (per as f64) - 0.5;
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        // Deterministic 0/1 with cluster shift + slope signal; not separable.
        let eta = -0.2 + 0.9 * x1 + 0.4 * (c as f64 - 3.5);
        y[i] = if (eta + ((i * 7 % 11) as f64 - 5.0) * 0.15) > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
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
    let opts = FitOptions {
        target_indices: vec![1],
        ..FitOptions::default()
    };
    (x, y, n, p, model, ids, opts)
}

/// The crossed-extra shape at binomial family: primary random intercept plus one
/// crossed intercept-only extra grouping, inside the dense GLMM envelope.
/// Deterministic 0/1 responses, same construction as
/// [`glmm_binomial_intercept_case`].
fn crossed_extra_glmm_case() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    FitOptions,
) {
    let g1 = 6usize; // primary intercept groups
    let g2 = 4usize; // crossed extra groups
    let n = 96usize;
    let p = 2usize;
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        let c1 = i % g1;
        let c2 = (i / g1) % g2;
        pid[i] = c1 as u32;
        eid[i] = c2 as u32;
        let x1 = ((i % 8) as f64) / 8.0 - 0.5;
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        let eta = -0.1 + 0.8 * x1 + 0.3 * (c1 as f64 - 2.5) + 0.25 * (c2 as f64 - 1.5);
        y[i] = if (eta + ((i * 7 % 11) as f64 - 5.0) * 0.15) > 0.0 {
            1.0
        } else {
            0.0
        };
    }
    let model = ModelSpec {
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: vec![],
            }],
        }),
    };
    let ids = GroupIds {
        primary: pid,
        extra: vec![eid],
    };
    let opts = FitOptions {
        target_indices: vec![1],
        ..FitOptions::default()
    };
    (x, y, n, p, model, ids, opts)
}

/// UNBALANCED nesting, Gaussian: 4 parents, parent 0 owning 3 children and the
/// rest 2, child ids globalized as `3·parent + child`. `spec_sized_from_ids`
/// takes `n_per_parent` from the widest parent (3), so the build capacity is
/// `4·3 = 12` while the draw only reaches child id 10 — 11 used levels. That gap
/// is the under-filled case the capacity pin must accept.
fn nested_unbalanced_case() -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    FitOptions,
) {
    let children_per_parent = [3usize, 2, 2, 2];
    let rows_per_child = 4usize;
    let p = 2usize;
    let mut st = 7u64;
    let mut x = Vec::new();
    let mut y = Vec::new();
    let mut pid = Vec::new();
    let mut eid = Vec::new();
    for (parent, &kids) in children_per_parent.iter().enumerate() {
        let u_p = 0.5 * lcg(&mut st);
        for child in 0..kids {
            let u_c = 0.3 * lcg(&mut st);
            for _ in 0..rows_per_child {
                let x1 = lcg(&mut st);
                x.extend_from_slice(&[1.0, x1]);
                y.push(0.4 + 0.6 * x1 + u_p + u_c + 0.2 * lcg(&mut st));
                pid.push(parent as u32);
                eid.push((3 * parent + child) as u32);
            }
        }
    }
    let n = y.len();
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 1 },
                slopes: vec![],
            }],
        }),
    };
    let ids = GroupIds {
        primary: pid,
        extra: vec![eid],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    (x, y, n, p, model, ids, opts)
}

// --- FitView accessor surface ---

#[test]
fn fitview_accessors_match_fit_for_ols() {
    let (x, y, n, p, model, ids, opts) = ols_case();
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let mut ws = build_workspace(&model, n, p, &opts);
    let v = fit_on(&mut ws, &x, &y, &ids, None, &opts);
    assert_eq!(v.converged(), cold.converged);
    // OLS t_sq is target-compact.
    assert_eq!(v.t_sq().len(), opts.target_indices.len());
    assert_eq!(v.betas().len(), p);
}

// --- build_workspace routing ---

#[test]
fn build_workspace_routes_fixed_only_to_ols_and_mixed_to_lmm() {
    let opts = FitOptions::default();
    let fixed = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let ws_fixed = build_workspace(&fixed, 100, 3, &opts);
    assert!(ws_fixed.is_ols());

    let (_, _, n, p, mixed, ids, _) = lmm_intercept_case();
    let sized = spec_sized_from_ids_pub(&mixed, &ids);
    let ws_mix = build_workspace(&sized, n, p, &FitOptions::default());
    assert!(ws_mix.is_lmm_dense());

    let glm = ModelSpec {
        family: Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        re: None,
    };
    assert!(build_workspace(&glm, 50, 2, &opts).is_glm());

    let nb = ModelSpec {
        family: Family::NegativeBinomial {
            link: crate::NegBinomialLink::Log,
        },
        re: None,
    };
    assert!(build_workspace(&nb, 50, 2, &opts).is_prebuilt());
}

// --- pin the absence of the scalar-Brent route ---

/// The scalar-Brent kernel in `lme.rs` is deliberately not wired: every tier
/// routes the single-random-intercept Gaussian LMM to `LmmDense`/BOBYQA (the
/// reasoning is in `lmm.rs`'s module header). If someone adds a `prefer_scalar`
/// branch to `build_workspace` later, this fails loudly.
#[test]
fn single_intercept_gaussian_routes_to_bobyqa_not_brent() {
    let (_x, _y, n, p, model, ids, opts) = lmm_intercept_case();
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let ws = build_workspace(&sized, n, p, &opts);
    assert!(ws.is_lmm_dense());
}

// --- reuse gate + shape pin ---

#[test]
fn fit_on_ols_reused_ws_near_identical_to_fit_cold() {
    let (x, y, n, p, model, ids, opts) = ols_case();
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let mut ws = build_workspace(&model, n, p, &opts);
    let f1 = fit_on(&mut ws, &x, &y, &ids, None, &opts).into_fit(&x, &y, n, p, &model, &opts);
    let f2 = fit_on(&mut ws, &x, &y, &ids, None, &opts).into_fit(&x, &y, n, p, &model, &opts);
    assert_near(&cold.beta, &f1.beta, "beta vs fit_cold");
    assert_near(&cold.se, &f1.se, "se vs fit_cold");
    assert_near(&f1.beta, &f2.beta, "beta first vs second reuse");
}

#[test]
fn fit_on_reused_ws_near_identical_to_fit_cold_lmm() {
    let (x, y, n, p, model, ids, opts) = lmm_intercept_case();
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    let f1 = fit_on(&mut ws, &x, &y, &ids, None, &opts).into_fit(&x, &y, n, p, &model, &opts);
    assert_near(&cold.beta, &f1.beta, "beta vs fit_cold");
    assert_near(&cold.tau2, &f1.tau2, "tau2 vs fit_cold");
    assert_near(
        &[cold.dispersion],
        &[f1.dispersion],
        "dispersion vs fit_cold",
    );

    // Second draw on the SAME ws — different data, same shape. A repeat of draw 1
    // would agree even with stale accumulator state; a different draw will not.
    let y2: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, &v)| v + 0.3 * ((i % 7) as f64 - 3.0))
        .collect();
    let cold2 = fit_cold(&x, &y2, n, p, &model, &ids, &opts);
    let f2 = fit_on(&mut ws, &x, &y2, &ids, None, &opts).into_fit(&x, &y2, n, p, &model, &opts);
    assert_near(&cold2.beta, &f2.beta, "draw-2 beta vs fit_cold");
    assert_near(&cold2.tau2, &f2.tau2, "draw-2 tau2 vs fit_cold");
    // The perturbation must actually move the fit, or the reuse gate above proves
    // nothing.
    assert!(
        (f1.beta[0] - f2.beta[0]).abs() > 1e-6,
        "draws are degenerate"
    );
}

#[test]
fn fit_on_glmm_dense_matches_fit_cold() {
    let (x, y, n, p, model, ids, opts) = glmm_binomial_intercept_case();
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    assert!(cold.converged);
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    assert!(ws.is_glmm_dense());
    let f1 = fit_on(&mut ws, &x, &y, &ids, None, &opts).into_fit(&x, &y, n, p, &model, &opts);
    assert_near(&cold.beta, &f1.beta, "beta vs fit_cold");
    assert_near(&cold.se, &f1.se, "se vs fit_cold");

    // Second draw on the SAME ws — different responses, same shape. Guards stale
    // Z / PIRLS state that a repeat of draw 1 would agree with anyway.
    let y2: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, &v)| if i % 5 == 0 { 1.0 - v } else { v })
        .collect();
    let cold2 = fit_cold(&x, &y2, n, p, &model, &ids, &opts);
    assert!(cold2.converged);
    let f2 = fit_on(&mut ws, &x, &y2, &ids, None, &opts).into_fit(&x, &y2, n, p, &model, &opts);
    assert_near(&cold2.beta, &f2.beta, "draw-2 beta vs fit_cold");
    assert_near(&cold2.se, &f2.se, "draw-2 se vs fit_cold");
}

#[test]
#[should_panic(expected = "shape")]
fn fit_on_panics_on_level_count_mismatch() {
    let (x, y, n, p, model, ids, opts) = lmm_intercept_case();
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    // Collapse every row to a single cluster → fewer levels than the build shape.
    let fewer = GroupIds {
        primary: vec![0u32; n],
        extra: vec![],
    };
    let _ = fit_on(&mut ws, &x, &y, &fewer, None, &opts);
}

// --- sparse routing (crossed extra-grouping random slope) ---

/// A crossed Gaussian LMM whose EXTRA grouping carries `extra_slopes` as its
/// random-slope x-columns. With `vec![1]` this is the shape `classify_design`
/// routes to `Sparse`, and the shape a dense hot path drops the extra slope from
/// if it fails to route; with `vec![]` (intercept-only extra) the same design
/// stays inside the dense NoZ envelope. Returns a converging design either way.
fn crossed_extra_case(
    extra_slopes: Vec<u32>,
) -> (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    FitOptions,
) {
    let g1 = 8usize; // primary intercept groups
    let g2 = 5usize; // extra groups carrying the random slope
    let n = 80usize;
    let p = 2usize;
    let mut st = 99u64;
    let u1: Vec<f64> = (0..g1).map(|_| 0.5 * lcg(&mut st)).collect();
    let s2: Vec<f64> = (0..g2).map(|_| 0.4 * lcg(&mut st)).collect();
    let mut x = vec![0.0f64; n * p];
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        let c1 = i % g1;
        let c2 = i % g2;
        pid[i] = c1 as u32;
        eid[i] = c2 as u32;
        let x1 = lcg(&mut st);
        x[i * 2] = 1.0;
        x[i * 2 + 1] = x1;
        y[i] = 0.5 + 0.7 * x1 + u1[c1] + s2[c2] * x1 + 0.2 * lcg(&mut st);
    }
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 1 },
                slopes: extra_slopes,
            }],
        }),
    };
    let ids = GroupIds {
        primary: pid,
        extra: vec![eid],
    };
    let opts = FitOptions {
        target_indices: vec![0, 1],
        ..FitOptions::default()
    };
    (x, y, n, p, model, ids, opts)
}

#[test]
fn fit_on_sparse_matches_fit_cold() {
    let (x, y, n, p, model, ids, opts) = crossed_extra_case(vec![1]);
    let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    assert!(ws.is_prebuilt()); // sparse Level 1 routes through the Prebuilt arm
    let via = fit_on(&mut ws, &x, &y, &ids, None, &opts).into_fit(&x, &y, n, p, &model, &opts);
    // Near-identity (both call the same sparse kernel); guards the routing + wrap.
    assert_near(&cold.beta, &via.beta, "sparse Level 1 beta vs fit_cold");
    assert_near(&cold.se, &via.se, "sparse Level 1 se vs fit_cold");
}

/// An extra grouping that outgrows its build capacity must panic (see
/// `FitWorkspace`). The primary pin cannot catch this — the primary level count
/// is unchanged. The pin sits ahead of the dispatch, so it is route-agnostic;
/// the `vec![1]` extra slope routes this case Sparse, where the overflow would
/// instead reach the sparse kernel with an undersized build spec. The two tests
/// below cover the dense LMM and dense GLMM arms.
#[test]
#[should_panic(expected = "extra grouping")]
fn fit_on_panics_on_extra_level_count_overflow() {
    let (x, y, n, p, model, ids, opts) = crossed_extra_case(vec![1]);
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    let mut more = ids.clone();
    // One row promoted past the built level count for that grouping.
    more.extra[0][0] = *more.extra[0].iter().max().unwrap() + 1;
    let _ = fit_on(&mut ws, &x, &y, &more, None, &opts);
}

/// Dense-LMM arm of the capacity pin — the arm where an over-capacity id would
/// scatter into the next grouping's column block instead of panicking. The
/// routing assertion runs FIRST: if routing ever flips to Sparse this must fail
/// there, not pass on the sparse arm's panic.
#[test]
#[should_panic(expected = "extra grouping")]
fn fit_on_panics_on_dense_lmm_extra_level_count_overflow() {
    let (x, y, n, p, model, ids, opts) = crossed_extra_case(vec![]);
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    assert!(ws.is_lmm_dense());
    let mut more = ids.clone();
    more.extra[0][0] = *more.extra[0].iter().max().unwrap() + 1;
    let _ = fit_on(&mut ws, &x, &y, &more, None, &opts);
}

/// Dense-GLMM arm of the capacity pin. Routing asserted before the pin fires,
/// for the same reason as the dense-LMM test above.
#[test]
#[should_panic(expected = "extra grouping")]
fn fit_on_panics_on_dense_glmm_extra_level_count_overflow() {
    let (x, y, n, p, model, ids, opts) = crossed_extra_glmm_case();
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    assert!(ws.is_glmm_dense());
    let mut more = ids.clone();
    more.extra[0][0] = *more.extra[0].iter().max().unwrap() + 1;
    let _ = fit_on(&mut ws, &x, &y, &more, None, &opts);
}

/// The pin is a capacity check (`<=`), not an equality one: an unbalanced nested
/// grouping is sized by its WIDEST parent, so a legitimate draw under-fills its
/// block. Capacity and used-level counts are asserted outright so a later edit to
/// the fixture cannot quietly close the gap and make this vacuous.
#[test]
fn fit_on_accepts_nested_draw_below_capacity() {
    let (x, y, n, p, model, ids, opts) = nested_unbalanced_case();
    let sized = spec_sized_from_ids_pub(&model, &ids);
    let mut ws = build_workspace(&sized, n, p, &opts);
    assert_eq!(ws.build_extra_capacity, vec![12]); // 4 parents × widest parent's 3
    let used = *ids.extra[0].iter().max().unwrap() as usize + 1;
    assert_eq!(used, 11); // max child id 10 (3·3 + 1)
    let _ = fit_on(&mut ws, &x, &y, &ids, None, &opts);
}

/// Target count sizes the OLS/GLM result slots at build, so growing it per call
/// must fault here rather than deep inside the estimator's own bounds check.
#[test]
#[should_panic(expected = "target count is frozen at build")]
fn fit_on_panics_on_grown_target_count() {
    let (x, y, n, p, model, ids, opts) = ols_case();
    let mut ws = build_workspace(&model, n, p, &opts);
    let wider = FitOptions {
        target_indices: vec![0, 1, 2],
        ..opts
    };
    let _ = fit_on(&mut ws, &x, &y, &ids, None, &wider);
}

/// A second `NestedWithin` extra aliases the first's RE-column block
/// (`from_cluster_spec_ext` is last-wins) and `classify_design` does not count
/// nestings, so without the build-time shape gate this shape would fit and
/// return a wrong answer instead of refusing.
#[test]
#[should_panic(expected = "at most one NestedWithin")]
fn build_workspace_rejects_two_nested_groupings() {
    let (_, _, n, p, model, _, opts) = nested_unbalanced_case();
    let mut sized = model;
    let nested = sized.re.as_ref().unwrap().extra_groupings[0].clone();
    sized.re.as_mut().unwrap().extra_groupings.push(nested);
    let _ = build_workspace(&sized, n, p, &opts);
}

// --- reuse-gate completeness (varying n, weighted reuse) ---

#[test]
fn fit_on_varying_n_below_n_max_matches_fit_cold() {
    // Build the ws at n_max; refit at several n < n_max. Each must equal a fresh
    // fit_cold at that n (guards n_max-sized buffers being read past n).
    let (full_x, full_y, n_max, p, model, _ids, opts) = ols_case();
    let mut ws = build_workspace(&model, n_max, p, &opts);
    for &n in &[10usize, 21, n_max] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds::default();
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.se, &via.se, &format!("n={n} se"));
    }
}

#[test]
fn fit_on_weighted_reuse_matches_fit_cold() {
    // A weighted call, then a unit-weight call on the SAME (has_weights=true) ws:
    // both must match fit_cold (guards stale prior_w — weights presence is frozen,
    // so the reset varies the weight VALUES, ending on unit weights).
    let (x, y, n, p, model, ids, mut opts_w) = ols_case();
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 4) as f64).collect();
    opts_w.weights = Some(w);
    let opts_unit = FitOptions {
        weights: Some(vec![1.0; n]),
        ..opts_w.clone()
    };
    let mut ws = build_workspace(&model, n, p, &opts_w);

    let cold_w = fit_cold(&x, &y, n, p, &model, &ids, &opts_w);
    let via_w =
        fit_on(&mut ws, &x, &y, &ids, None, &opts_w).into_fit(&x, &y, n, p, &model, &opts_w);
    assert_near(&cold_w.beta, &via_w.beta, "weighted beta");

    let cold_u = fit_cold(&x, &y, n, p, &model, &ids, &opts_unit);
    let via_u =
        fit_on(&mut ws, &x, &y, &ids, None, &opts_unit).into_fit(&x, &y, n, p, &model, &opts_unit);
    assert_near(&cold_u.beta, &via_u.beta, "unit-weight-after-weighted beta");
    assert_near(&cold_u.se, &via_u.se, "unit-weight-after-weighted se");
}

// --- fit_on alloc reduction (0.1.3): stale-row tripwire per touched arm ---
//
// `Ols`/`Glm`/`LmmDense` each gained a build-once `x_mat` sibling buffer that
// `fit_on` fills in place instead of allocating fresh every call: workspace
// `x_mat` reuse across different n must equal fresh single-shot fits. Rows
// past the CURRENT call's `n` keep the PREVIOUS call's values and are never
// read — the tests above already
// cover ascending n for OLS; the pairs below add the direction most likely to
// surface a stale-row bug (shrinking n leaves more of the buffer stale than
// growing it does), for every arm that gained a buffer.

#[test]
fn fit_on_ols_smaller_then_larger_matches_fit_cold() {
    let (full_x, full_y, n_max, p, model, _ids, opts) = ols_case();
    let mut ws = build_workspace(&model, n_max, p, &opts);
    for &n in &[12usize, n_max] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds::default();
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.se, &via.se, &format!("n={n} se"));
    }
}

#[test]
fn fit_on_ols_larger_then_smaller_matches_fit_cold() {
    let (full_x, full_y, n_max, p, model, _ids, opts) = ols_case();
    let mut ws = build_workspace(&model, n_max, p, &opts);
    for &n in &[n_max, 12usize] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds::default();
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.se, &via.se, &format!("n={n} se"));
    }
}

#[test]
fn fit_on_glm_smaller_then_larger_matches_fit_cold() {
    let (full_x, full_y, n_max, p, model, _ids, opts) = glm_case();
    let mut ws = build_workspace(&model, n_max, p, &opts);
    assert!(ws.is_glm());
    for &n in &[12usize, n_max] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds::default();
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.se, &via.se, &format!("n={n} se"));
    }
}

#[test]
fn fit_on_glm_larger_then_smaller_matches_fit_cold() {
    let (full_x, full_y, n_max, p, model, _ids, opts) = glm_case();
    let mut ws = build_workspace(&model, n_max, p, &opts);
    for &n in &[n_max, 12usize] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds::default();
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.se, &via.se, &format!("n={n} se"));
    }
}

/// LMM sub-slices must still cover every primary cluster level (`fit_on`'s
/// shape pin is an EQUALITY check on level count, not a capacity check like the
/// extra groupings) — `n_clusters` divides every `n` used here.
#[test]
fn fit_on_lmm_dense_smaller_then_larger_matches_fit_cold() {
    let (full_x, full_y, n_max, p, model, ids_full, opts) = lmm_intercept_case();
    let sized = spec_sized_from_ids_pub(&model, &ids_full);
    let mut ws = build_workspace(&sized, n_max, p, &opts);
    assert!(ws.is_lmm_dense());
    for &n in &[24usize, n_max] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds {
            primary: ids_full.primary[..n].to_vec(),
            extra: vec![],
        };
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.tau2, &via.tau2, &format!("n={n} tau2"));
    }
}

#[test]
fn fit_on_lmm_dense_larger_then_smaller_matches_fit_cold() {
    let (full_x, full_y, n_max, p, model, ids_full, opts) = lmm_intercept_case();
    let sized = spec_sized_from_ids_pub(&model, &ids_full);
    let mut ws = build_workspace(&sized, n_max, p, &opts);
    for &n in &[n_max, 24usize] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds {
            primary: ids_full.primary[..n].to_vec(),
            extra: vec![],
        };
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.tau2, &via.tau2, &format!("n={n} tau2"));
    }
}

/// `scaled_x` is allocated only for weighted builds — an unweighted
/// workspace must never read it. `has_weights` is frozen at build, so
/// `OlsWorkspace::scaled_x` is sized 0×0 on an unweighted build (never read)
/// and `n_max×p` on a weighted one (read every call). Fitting both routes to
/// completion — instead of panicking on the 0×0 bounds check — proves the
/// gate reads the right flag.
#[test]
fn fit_on_ols_scaled_x_gate_matches_has_weights() {
    let (x, y, n, p, model, ids, opts) = ols_case();
    assert!(opts.weights.is_none());
    let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();
    let opts_w = FitOptions {
        weights: Some(w),
        ..opts.clone()
    };

    // Unweighted build: has_weights=false ⇒ scaled_x is 0×0, must never be read.
    let cold_u = fit_cold(&x, &y, n, p, &model, &ids, &opts);
    let mut ws_u = build_workspace(&model, n, p, &opts);
    let via_u = fit_on(&mut ws_u, &x, &y, &ids, None, &opts).into_fit(&x, &y, n, p, &model, &opts);
    assert_near(&cold_u.beta, &via_u.beta, "unweighted beta");
    assert_near(&cold_u.se, &via_u.se, "unweighted se");

    // Weighted build: has_weights=true ⇒ scaled_x is n_max×p, read every call.
    let cold_w = fit_cold(&x, &y, n, p, &model, &ids, &opts_w);
    let mut ws_w = build_workspace(&model, n, p, &opts_w);
    let via_w =
        fit_on(&mut ws_w, &x, &y, &ids, None, &opts_w).into_fit(&x, &y, n, p, &model, &opts_w);
    assert_near(&cold_w.beta, &via_w.beta, "weighted beta");
    assert_near(&cold_w.se, &via_w.se, "weighted se");
}

/// The offset path fills the preallocated `y_shifted` instead of collecting:
/// a build-once `n_max`-sized sibling buffer, filled fresh from `y - offset`
/// every call. Round-trips the offset path across varying n (grow then
/// shrink) so a stale trailing entry — left over from a larger previous call
/// — would show up as a wrong shift on the smaller one.
#[test]
fn fit_on_lmm_dense_offset_round_trip_varying_n() {
    // `fit_cold`'s own boundary check requires `offset.len() == n` exactly
    // (`fit_warm`'s shape gate), so each iteration gets its own n-sliced opts
    // for the fit_cold reference; `opts` (full n_max-length offset) is what
    // `fit_on`'s frozen-presence build actually uses, matching how a real
    // build-once/fit-many caller would hold one offset buffer across calls.
    let (full_x, full_y, n_max, p, model, ids_full, mut opts) = lmm_intercept_case();
    let offset: Vec<f64> = (0..n_max).map(|i| 0.05 * ((i % 4) as f64 - 1.5)).collect();
    opts.offset = Some(offset.clone());
    let sized = spec_sized_from_ids_pub(&model, &ids_full);
    let mut ws = build_workspace(&sized, n_max, p, &opts);
    for &n in &[24usize, n_max, 24usize] {
        let x = &full_x[..n * p];
        let y = &full_y[..n];
        let ids = GroupIds {
            primary: ids_full.primary[..n].to_vec(),
            extra: vec![],
        };
        let opts_n = FitOptions {
            offset: Some(offset[..n].to_vec()),
            ..opts.clone()
        };
        let cold = fit_cold(x, y, n, p, &model, &ids, &opts_n);
        let via = fit_on(&mut ws, x, y, &ids, None, &opts).into_fit(x, y, n, p, &model, &opts_n);
        assert_near(&cold.beta, &via.beta, &format!("n={n} beta"));
        assert_near(&cold.tau2, &via.tau2, &format!("n={n} tau2"));
    }
}
