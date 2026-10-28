//! Unit tests for `lmm::kernel` and `lmm` (dispatch, workspace, BOBYQA config, joint Wald).

use super::*;
use crate::fit::common_tests::lcg;
use crate::test_support::{
    cluster_of_row, extra_level_of_row, intercept_only_spec, model_atom, Sizing,
};
use crate::{Family, Grouping, GroupingRelation, ModelSpec, Re, ReStructure, ReTerm};

/// The plain single-intercept grouping at `n_clusters` clusters (no slopes,
/// no extras), as a `ModelSpec` lowers it.
fn single_grouping(n_clusters: usize) -> LmmGroupings {
    let spec = SizedSpec {
        spec: intercept_only_spec(),
        n_primary: n_clusters,
        extra_counts: vec![],
    };
    LmmGroupings::from_cluster_spec(&spec, &[])
}

/// n=48, p=3 (intercept + x1 + x2), 6 clusters,
/// y = 0.5 + 0.4·x1 − 0.2·x2 + u_c + 0.8·e.
fn hand_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let (n, nc) = (48usize, 6usize);
    let mut st = 42u64;
    let u_c: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;
        y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + u_c[c] + 0.8 * lcg(&mut st);
    }
    (x, y, ids)
}

/// q=1 single-grouping deviance against the textbook n×n V oracle
/// ([`brute_force_deviance`]) at nine θ from 0 to 100, under REML and ML;
/// the balanced collapse arm against the general loop at the same θ
/// (reassociation band, ≤ 1e-9 rel). The collapse precompute holds no
/// criterion term, so it serves both criteria.
#[test]
fn deviance_matches_brute_force_across_theta() {
    const THETAS: [f64; 9] = [0.0, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0];
    let (x, y, ids) = hand_dataset();
    let mut suff = LmmSuffStats::with_groupings(3, single_grouping(6));
    suff.add_rows(x.as_ref(), &y, &ids);
    let mut fit = LmmFitScratch::with_groupings(3, &single_grouping(6));
    let mut fit_c = LmmFitScratch::with_groupings(3, &single_grouping(6));
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));
    for criterion in [Criterion::Reml, Criterion::Ml] {
        fit.criterion = criterion;
        fit_c.criterion = criterion;
        for &theta in &THETAS {
            let dev_gen = reml_deviance(&[theta], &suff, &mut fit);
            let oracle = brute_force_deviance(&[theta], &x, &y, &[&ids], criterion);
            assert!(dev_gen.is_finite(), "{criterion:?} θ={theta}");
            assert!(
                (dev_gen - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "{criterion:?} θ={theta}: general {dev_gen} vs oracle {oracle}"
            );
            let dev_c = reml_deviance(&[theta], &suff, &mut fit_c);
            assert!(
                (dev_c - dev_gen).abs() <= 1e-9 * dev_gen.abs().max(1.0),
                "{criterion:?} θ={theta}: collapse {dev_c} vs general {dev_gen}"
            );
        }
    }
}

/// Exercises the plateau policy: a `MaxFunReached` cap-out must still
/// report the honest finite endpoint
/// (β̂/σ̂²/SE/deviance), with `converged = false` and `boundary_hit == 2`
/// (not the accepted-boundary 1). Forces the cap by swapping in a solver
/// whose `max_fun` is the legal minimum (`npt + 1`) — one eval past the
/// initial model build, nowhere near this dataset's optimum.
#[test]
fn maxfun_cap_reports_honest_endpoint() {
    let (x, y, ids) = hand_dataset();
    let targets: Vec<u32> = vec![1, 2];

    let mut ws = LmmWorkspace::new(3, 6);
    ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let n_theta = ws.theta.len();
    let npt = 2 * n_theta + 1; // n_theta == 1 here: PRIMA's minimum npt
    let config = {
        let mut c = Config::new(n_theta);
        c.npt = npt;
        c.max_fun = npt + 1;
        c
    };
    ws.solver = Bobyqa::new(n_theta, config).expect("legal minimal config");

    let fit = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);

    assert!(!fit.converged, "capped fit must not report converged");
    assert_eq!(
        fit.boundary_hit, 2,
        "capped fit must not migrate into the accepted-boundary code"
    );
    assert_eq!(
        fit.pinned_components, 0,
        "a capped endpoint is a point, not an accepted boundary"
    );
    assert!(
        fit.deviance.is_finite(),
        "plateau policy: capped endpoint must report a finite deviance"
    );
    assert!(
        fit.sigma_sq.is_finite(),
        "plateau policy: capped endpoint must report a finite sigma_sq"
    );
    assert!(
        fit.joint_t_sq.is_finite(),
        "plateau policy: capped endpoint must report a finite joint_t_sq"
    );
    for &tj in &targets {
        assert!(
            ws.recovery.betas[tj as usize].is_finite(),
            "plateau policy: capped endpoint must not NaN-fill beta"
        );
    }
    assert!(fit.n_eval <= npt + 1, "n_eval must reflect the forced cap");

    // Pinned values are the deterministic truncated-BOBYQA endpoint (hand_dataset,
    // max_fun = npt+1) — a regression lock, not an external oracle: any solver-path
    // change that moves the honest cap-out endpoint should fail this test.
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs().max(1e-12);
    assert!(
        rel(fit.deviance, -62.08988134487164) < 1e-6,
        "deviance = {}",
        fit.deviance
    );
    assert!(
        rel(fit.sigma_sq, 0.16043347869402982) < 1e-6,
        "sigma_sq = {}",
        fit.sigma_sq
    );
    assert!(
        rel(fit.joint_t_sq, 14.568949550460516) < 1e-6,
        "joint_t_sq = {}",
        fit.joint_t_sq
    );
    let want_betas = [
        0.4691004480864937,
        0.26391548909385104,
        -0.33307894295165125,
    ];
    for (j, &wb) in want_betas.iter().enumerate() {
        assert!(
            rel(ws.recovery.betas[j], wb) < 1e-6,
            "betas[{j}] = {}, want {}",
            ws.recovery.betas[j],
            wb
        );
    }
}

/// End-to-end q=1 parity on the hand dataset: the general machine against
/// frozen literal endpoints, at the amended tolerances (rel 1e-4, abs floors
/// β̂ 1e-5 / stat 1e-4 — the measured Brent θ̂-placement-noise floor).
///
/// `PINNED_BETAS`/`PINNED_STATS`/`PINNED_JOINT_T_SQ` are frozen reference
/// outputs on `hand_dataset` (targets `[1, 2]`), pinned so `fit_lmm`'s
/// general θ-search agrees with a fixed reference at the amended tolerances
/// above.
#[test]
fn fit_matches_pinned_q1_endpoint_on_hand_dataset() {
    const PINNED_BETAS: [f64; 3] = [
        4.699333343472561e-1,
        2.5793782757575945e-1,
        -3.2576278857950314e-1,
    ];
    const PINNED_STATS: [f64; 2] = [2.3733464018128823e0, 3.177463198588152e0];
    const PINNED_JOINT_T_SQ: f64 = 1.3713656562170998e1;

    let (x, y, ids) = hand_dataset();
    let targets: Vec<u32> = vec![1, 2];

    let mut ws = LmmWorkspace::new(3, 6);
    ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let fit = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);
    assert!(fit.converged);
    assert!(fit.boundary_hit <= 1);

    for (j, &want) in PINNED_BETAS.iter().enumerate() {
        let (a, b) = (want, ws.recovery.betas[j]);
        let d = (a - b).abs();
        assert!(
            d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
            "β[{j}]: {a} vs {b}"
        );
    }
    for (idx, &tj) in targets.iter().enumerate() {
        let a = PINNED_STATS[idx];
        let b = ws.recovery.t_sq[tj as usize].sqrt();
        let d = (a - b).abs();
        assert!(
            d <= 1e-4 || d <= 1e-4 * a.abs().max(b.abs()),
            "stat[{tj}]: {a} vs {b}"
        );
    }
    let (a, b) = (PINNED_JOINT_T_SQ, fit.joint_t_sq);
    let d = (a - b).abs();
    assert!(
        d <= 1e-4 || d <= 1e-4 * a.abs().max(b.abs()),
        "joint: {a} vs {b}"
    );
}

/// Rank deficiency is DETECTED, not refused, at kernel level: x2 = 0.1·x1
/// (the scaled-duplicate fixture — exact duplicates can slip through faer's
/// llt grey zone) returns a fit whose `pivot` records the exhaustion and
/// names the offending column.
///
/// This kernel entry point is below the alias gate, which is what actually
/// handles a design like this: through `fit`/`fit_with_start` the duplicate
/// column is dropped before the solver runs and the caller gets a clean
/// `p−1` fit with `aliased[2]`, matching R. `fit_lmm` called directly does
/// not NaN-fill — it hands back the numbers together with the statistic
/// that condemns them, and the standard error it reports (~9e7 on a
/// coefficient of 11.5) is truthful.
#[test]
fn rank_deficient_design_is_flagged_not_refused() {
    let n = 48usize;
    let n_clusters = 6usize;
    let mut st = 11u64;
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = 0.1 * x1; // 0.1-scaled duplicate → guaranteed non-convergence
        y[i] = 0.5 + 0.4 * x1 + 0.8 * lcg(&mut st);
    }
    let mut ws = LmmWorkspace::new(3, n_clusters);
    ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let fit = fit_lmm_unhinted(&mut ws, &[1, 2], None, ColumnScaling::IDENTITY);
    assert!(
        fit.pivot < PIVOT_MIN,
        "the duplicate column must be detected, got pivot {}",
        fit.pivot
    );
    assert_eq!(
        fit.pivot_col, 2,
        "the LATER column of the duplicated pair is the one named"
    );
    // The SE is what makes the returned numbers safe: Var(β̂₂) ~ 8e15, so
    // the coefficient is reported with an error eight orders larger than
    // itself. That is the channel a caller reads, and it does not lie.
    assert!(
        ws.recovery.var_diag[2].sqrt() > 1e6 * ws.recovery.betas[2].abs(),
        "β̂₂ = {} must carry an SE orders above it, got {}",
        ws.recovery.betas[2],
        ws.recovery.var_diag[2].sqrt()
    );
}

/// A truth-started fit (`theta_start: Some`) reaches the same answer as the
/// blind fit on the same bytes — and Some(0.0) exercises the
/// THETA_TRUTH_FLOOR clamp rather than starting on the 0 boundary.
/// Bands are the amended floors: two BOBYQA runs from different
/// starts each place θ̂ within the rho_end band of the same minimum.
#[test]
fn theta_start_some_matches_blind_fit() {
    let (x, y, ids) = hand_dataset();
    let targets: Vec<u32> = vec![1, 2];

    let mut ws_blind = LmmWorkspace::new(3, 6);
    ws_blind.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let blind = fit_lmm_unhinted(&mut ws_blind, &targets, None, ColumnScaling::IDENTITY);
    assert!(blind.converged);

    for start in [[0.0], [0.6]] {
        let mut ws = LmmWorkspace::new(3, 6);
        ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm_unhinted(&mut ws, &targets, Some(&start), ColumnScaling::IDENTITY);
        assert!(fit.converged, "start {start:?}");
        for j in 0..3 {
            let (a, b) = (ws_blind.recovery.betas[j], ws.recovery.betas[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
                "start {start:?} β[{j}]: blind {a} vs started {b}"
            );
        }
    }

    // THETA_TRUTH_FLOOR clamp, directly: two raw starts on opposite sides of
    // 0 but both below the floor (0.0 and 0.005, THETA_TRUTH_FLOOR = 0.01)
    // must collapse to the identical internal starting point and so drive
    // BOBYQA through a bit-for-bit identical search — n_eval, θ̂ and deviance
    // all equal. Without the clamp the two starts differ from each other and
    // from THETA_TRUTH_FLOOR, so BOBYQA's very first interpolation point
    // would not agree between the two runs.
    let mut ws_zero = LmmWorkspace::new(3, 6);
    ws_zero.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let fit_zero = fit_lmm_unhinted(
        &mut ws_zero,
        &targets,
        Some(&[0.0]),
        ColumnScaling::IDENTITY,
    );
    assert!(fit_zero.converged);

    let mut ws_below_floor = LmmWorkspace::new(3, 6);
    ws_below_floor.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let fit_below_floor = fit_lmm_unhinted(
        &mut ws_below_floor,
        &targets,
        Some(&[0.005]),
        ColumnScaling::IDENTITY,
    );
    assert!(fit_below_floor.converged);

    assert_eq!(fit_zero.n_eval, fit_below_floor.n_eval);
    assert_eq!(
        fit_zero.deviance.to_bits(),
        fit_below_floor.deviance.to_bits()
    );
    for (a, b) in ws_zero.theta.iter().zip(&ws_below_floor.theta) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// Bounded-allocation warm-path check for `fit_lmm`.
/// Marked #[ignore] because dhat measures
/// process-wide allocations; `alloc_test_guard` serializes test bodies, but
/// libtest's own per-test thread spawn still needs `--test-threads=1`:
///   RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests lmm_fit_warm_path_bounded_alloc -- --ignored --test-threads=1
///
/// The test locks the measured warm-path block count, 0. LmmWorkspace itself is
/// allocation-free across fits (Bobyqa::new is the only solver allocation,
/// done once), and `chol_lower` takes faer's Cholesky scratch from a stack
/// buffer at these shapes, so a deviance evaluation allocates nothing. A
/// hand-rolled owned-kernel replacement for faer's `llt` is not used: its
/// wasm `f64::ln` takes a different ULP path than the native build (the
/// factorization itself is fine), which breaks cross-platform bit-equality.
/// If a future faer version changes its Cholesky internals, update the bound
/// — do not relax it.
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn lmm_fit_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: usize = 100;
    // Measured 0 (this machine): `chol_lower` keeps the per-eval Cholesky scratch on the stack,
    // the diagonal_theta map is cached on LmmGroupings and the ranef recovery solves in scratch
    // fields, so the q=1 warm loop allocates nothing. Any heap block here is a regression.

    let (x, y, ids) = hand_dataset();
    let targets: Vec<u32> = vec![1, 2];
    let mut ws = LmmWorkspace::new(3, 6);

    // Warmup drives one-time setup outside the profiler window.
    ws.suff_mut().reset();
    ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let _ = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);
    crate::test_support::settle_background_allocs();

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        ws.suff_mut().reset();
        ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
        let _ = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert_eq!(
        stats.total_blocks, 0,
        "fit_lmm allocated {} blocks across {} warm-path calls",
        stats.total_blocks, N_CALLS
    );
}

// -----------------------------------------------------------------------
// Multi-grouping: layout-true datasets, suff-stats, family-blocked
// deviance vs a brute-force n×n oracle, and end-to-end fits.
// -----------------------------------------------------------------------

/// Layout-true multi-grouping dataset: primary S=6, crossed I=4, nested
/// np=2 (optional), p=3, n = n_blocks·atom rows. Ids come from the
/// contract layout helpers — the same functions the workspace uses.
#[allow(clippy::type_complexity)]
fn multi_dataset(
    with_nested: bool,
    n_blocks: usize,
) -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>, SizedSpec) {
    let sizing = Sizing::FixedClusters { n_clusters: 6 };
    let mut extra_counts = vec![4u32];
    let mut cluster = intercept_only_spec();
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        terms: Vec::new(),
        relation: GroupingRelation::Crossed,
        slopes: vec![],
    });
    if with_nested {
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::NestedWithin,
            slopes: vec![],
        });
        extra_counts.push(2);
    }
    let n = n_blocks * model_atom(&sizing, &extra_counts);
    let mut st = 99u64;
    let u_p: Vec<f64> = (0..6).map(|_| 0.5 * lcg(&mut st)).collect();
    let u_x: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
    let u_n: Vec<f64> = (0..12).map(|_| 0.3 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let n_extras = cluster.re.as_ref().unwrap().extra_groupings.len();
    let mut eids: Vec<Vec<u32>> = vec![vec![0u32; n]; n_extras];
    for i in 0..n {
        pid[i] = cluster_of_row(&sizing, i) as u32;
        #[allow(clippy::needless_range_loop)]
        for g in 0..n_extras {
            eids[g][i] = extra_level_of_row(&cluster, &sizing, &extra_counts, g, i) as u32;
        }
        let x1 = lcg(&mut st);
        let x2 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;
        y[i] = 0.5 + 0.4 * x1 - 0.2 * x2
            + u_p[pid[i] as usize]
            + u_x[eids[0][i] as usize]
            + if with_nested {
                u_n[eids[1][i] as usize]
            } else {
                0.0
            }
            + 0.8 * lcg(&mut st);
    }
    let sized = SizedSpec {
        spec: cluster,
        n_primary: 6,
        extra_counts: extra_counts.iter().map(|&c| c as usize).collect(),
    };
    (x, y, pid, eids, sized)
}

/// `diagonal_theta` / `n_theta` / `k_family` at q_p ∈ {1, 2, 3} — locks
/// the column-major vech ordering. q_p=1 must reproduce the intercept-only
/// baseline values; q_p>1 tests the standalone slope branch (no extras, k_crossed=0).
#[test]
fn groupings_vech_layout() {
    let base = SizedSpec {
        spec: intercept_only_spec(),
        n_primary: 4,
        extra_counts: vec![],
    };

    // q_p = 1 (intercept-only): shape must be unchanged.
    let g1 = LmmGroupings::from_cluster_spec(&base, &[]);
    assert_eq!(g1.n_theta(), 1);
    assert_eq!(g1.k_family(), 4); // 4 clusters × 1
    assert_eq!(g1.diagonal_theta(), &[0][..]);

    // q_p = 2 (1 slope): vech([σ_00, σ_10, σ_11]) length 3; diagonals at 0, 2.
    let mut spec2 = base.clone();
    spec2.spec.re.as_mut().unwrap().slopes.push(1);
    let g2 = LmmGroupings::from_cluster_spec(&spec2, &[1]);
    assert_eq!(g2.primary_q, 2);
    assert_eq!(g2.n_theta(), 3); // 2·3/2 = 3
    assert_eq!(g2.k_family(), 8); // 4 clusters × 2
    assert_eq!(g2.k_total, 8);
    assert_eq!(g2.diagonal_theta(), &[0, 2][..]); // off-diagonal vech[1]=1 excluded

    // q_p = 3 (2 slopes): vech([σ_00, σ_10, σ_11, σ_20, σ_21, σ_22]) length 6; diagonals at 0, 3, 5.
    let mut spec3 = base.clone();
    spec3.spec.re.as_mut().unwrap().slopes.push(1);
    spec3.spec.re.as_mut().unwrap().slopes.push(2);
    let g3 = LmmGroupings::from_cluster_spec(&spec3, &[1, 2]);
    assert_eq!(g3.primary_q, 3);
    assert_eq!(g3.n_theta(), 6); // 3·4/2 = 6
    assert_eq!(g3.k_family(), 12); // 4 clusters × 3
    assert_eq!(g3.k_total, 12);
    assert_eq!(g3.diagonal_theta(), &[0, 3, 5][..]);
}

/// The groupings of a Gaussian spec at 4 primary levels and 3 levels per
/// extra, each slope's x column taken as its spec column.
fn groupings_of(re: ReStructure) -> LmmGroupings {
    let cols = |s: &[crate::ColumnId]| s.iter().map(|&c| c as usize).collect::<Vec<_>>();
    let primary = cols(&re.slopes);
    let extras: Vec<Vec<usize>> = re.extra_groupings.iter().map(|g| cols(&g.slopes)).collect();
    let sized = SizedSpec {
        extra_counts: vec![3; re.extra_groupings.len()],
        spec: ModelSpec::mixed(Family::Gaussian, re),
        n_primary: 4,
    };
    LmmGroupings::from_cluster_spec_ext(&sized, &primary, &extras)
}

/// `(x || g)`: vech `[λ00, λ10, λ11]`, the covariance between the two terms
/// held at zero.
#[test]
fn mask_layout_of_uncorrelated_slope() {
    let g = groupings_of(Re::intercept().slopes([1]).uncorrelated());
    assert_eq!(g.primary_q, 2);
    assert_eq!(g.structural_theta(), &[1][..]);
    assert_eq!(g.diagonal_theta(), &[0, 2][..]);
    assert_eq!(g.diagonal_run_len, vec![2, 1]);
    assert_eq!(g.hidden, vec![false]);
    assert_eq!(g.n_free_theta(), 2);
    let mut theta = [0.5, 0.3, 0.7];
    g.zero_structural(&mut theta);
    assert_eq!(theta, [0.5, 0.0, 0.7]);
}

/// `(1 + x1 | g) + (0 + x2 | g)`: vech `[λ00, λ10, λ20, λ11, λ21, λ22]`; the
/// second term's column is uncorrelated with both columns of the first.
#[test]
fn mask_layout_of_split_terms() {
    let g = groupings_of(Re::intercept().terms([
        ReTerm::intercept().slopes([1]),
        ReTerm::no_intercept().slopes([2]),
    ]));
    assert_eq!(g.primary_q, 3);
    assert_eq!(g.structural_theta(), &[2, 4][..]);
    assert_eq!(g.diagonal_theta(), &[0, 3, 5][..]);
    assert_eq!(g.n_free_theta(), 4);
}

/// `(0 + x | g)`: the block keeps its intercept column, held at zero (its
/// diagonal and the entry below it) and hidden.
#[test]
fn mask_layout_of_intercept_free_block() {
    let g = groupings_of(Re::intercept().terms([ReTerm::no_intercept().slopes([1])]));
    assert_eq!(g.primary_q, 2);
    assert_eq!(g.structural_theta(), &[0, 1][..]);
    assert_eq!(g.diagonal_theta(), &[2][..]);
    assert_eq!(g.diagonal_run_len, vec![1]);
    assert_eq!(g.hidden, vec![true]);
    assert_eq!(g.n_free_theta(), 1);
}

/// The mask on crossed extras with slopes: `(1 | g) + (0 + x | h) + (x || k)`.
/// θ = `[λ_g | vech Λ_h at 1..4 | vech Λ_k at 4..7]`; each extra's entries
/// sit at its own `vech_start`.
#[test]
fn mask_reaches_extra_groupings() {
    let g = groupings_of(
        Re::intercept()
            .extra(Grouping::crossed().terms([ReTerm::no_intercept().slopes([1])]))
            .extra(Grouping::crossed().slopes([2]).uncorrelated()),
    );
    let starts: Vec<usize> = g.crossed.iter().map(|cf| cf.vech_start).collect();
    assert_eq!(starts, vec![1, 4]);
    assert_eq!(g.structural_theta(), &[1, 2, 5][..]);
    assert_eq!(g.diagonal_theta(), &[0, 3, 4, 6][..]);
    assert_eq!(g.hidden, vec![false, true, false]);
    assert_eq!(g.n_free_theta(), 4);
}

/// One unstructured term over the intercept and every slope is the layout
/// of the same block without terms, field for field.
#[test]
fn one_full_term_matches_the_termless_layout() {
    let with_terms = groupings_of(Re::intercept().terms([ReTerm::intercept().slopes([1, 2])]));
    let without = groupings_of(Re::intercept().slopes([1, 2]));
    assert_eq!(with_terms, without);
    assert!(without.structural_theta().is_empty());
    assert_eq!(without.hidden, vec![false]);
}

/// `vech(Σ)` (column-major, matching the θ layout) for one `q×q` block's
/// `vech(Λ)` — the quantity `canonicalize_pinned_blocks` must leave alone.
fn sigma_vech(theta: &[f64], q: usize) -> Vec<f64> {
    let mut lam = vec![0.0f64; q * q];
    primary_lambda(theta, q, &mut lam);
    let mut out = Vec::new();
    for c in 0..q {
        for r in c..q {
            let mut s = 0.0;
            for k in 0..=c {
                s += lam[r * q + k] * lam[c * q + k];
            }
            out.push(s);
        }
    }
    out
}

/// `canonicalize_pinned_blocks`: a pinned diagonal's column folds into the
/// trailing diagonals with Σ preserved, a second pass is a no-op, and a block
/// with no pinned diagonal comes back BIT-identical (that last one is what
/// keeps every non-singular fit's dump unmoved).
#[test]
fn canonicalize_pinned_blocks_folds_preserves_sigma_and_is_idempotent() {
    let base = SizedSpec {
        spec: intercept_only_spec(),
        n_primary: 4,
        extra_counts: vec![],
    };

    // q_p = 2, vech(Λ) = [λ00, λ10, λ11]: both diagonals at 0 with the whole
    // variance carried by the unidentified λ10 — the shape a pinned q=2 fit
    // stops at.
    let mut spec2 = base.clone();
    spec2.spec.re.as_mut().unwrap().slopes.push(1);
    let g2 = LmmGroupings::from_cluster_spec(&spec2, &[1]);
    let mut th2 = [0.0, 0.8172902747418043, 0.0];
    let sig2 = sigma_vech(&th2, 2);
    assert!(canonicalize_pinned_blocks(&g2, &mut th2));
    assert_eq!(th2[0], 0.0);
    assert_eq!(th2[1], 0.0); // column 0 zeroed outright
    assert!((th2[2] - 0.8172902747418043).abs() <= 1e-15); // mass on λ11
    for (got, want) in sigma_vech(&th2, 2).iter().zip(&sig2) {
        assert!((got - want).abs() <= 1e-15 * want.abs().max(1.0));
    }
    let again = th2;
    assert!(!canonicalize_pinned_blocks(&g2, &mut th2));
    assert_eq!(th2[0].to_bits(), again[0].to_bits());
    assert_eq!(th2[1].to_bits(), again[1].to_bits());
    assert_eq!(th2[2].to_bits(), again[2].to_bits());

    // No pinned diagonal ⇒ strict no-op, bit-for-bit.
    let mut th_live = [1.3, 0.37, 0.9];
    let before = th_live;
    assert!(!canonicalize_pinned_blocks(&g2, &mut th_live));
    for (a, b) in th_live.iter().zip(&before) {
        assert_eq!(a.to_bits(), b.to_bits());
    }

    // q_p = 3, vech(Λ) = [λ00, λ10, λ20, λ11, λ21, λ22]: the pinned leading
    // column carries two live entries that have to fold into columns 1 and 2.
    let mut spec3 = base.clone();
    spec3.spec.re.as_mut().unwrap().slopes.push(1);
    spec3.spec.re.as_mut().unwrap().slopes.push(2);
    let g3 = LmmGroupings::from_cluster_spec(&spec3, &[1, 2]);
    let mut th3 = [0.0, 0.5, 0.25, 0.0, 0.3, 0.4];
    let sig3 = sigma_vech(&th3, 3);
    assert!(canonicalize_pinned_blocks(&g3, &mut th3));
    assert_eq!(&th3[..3], &[0.0, 0.0, 0.0]); // pinned column 0 gone
    assert!(th3[3] > PIN_THETA); // its variance re-emerges on λ11
    for (got, want) in sigma_vech(&th3, 3).iter().zip(&sig3) {
        assert!(
            (got - want).abs() <= 1e-14 * want.abs().max(1.0),
            "Σ moved: {:?} vs {:?}",
            sigma_vech(&th3, 3),
            sig3
        );
    }
    let again3 = th3;
    assert!(!canonicalize_pinned_blocks(&g3, &mut th3));
    for (a, b) in th3.iter().zip(&again3) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// Suff-stats bookkeeping on a hand-checkable block: counts per RE column,
/// per-column sums, crossed cross-counts.
#[test]
fn suff_stats_multi_accumulators() {
    let (x, y, pid, eids, cluster) = multi_dataset(true, 1); // one atom block, n=48
    let g = LmmGroupings::from_cluster_spec(&cluster, &[]);
    assert_eq!(g.n_primary, 6);
    assert_eq!(g.nested_per_parent, 2);
    assert_eq!(g.k_family(), 18); // 6 + 6·2
    assert_eq!(g.k_total, 22); // + 4 crossed
    assert_eq!(g.n_theta(), 3);
    let mut suff = LmmSuffStats::with_groupings(3, g);
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    // One full-factorial block: every primary level has 8 rows, every
    // child 4, every crossed level 12.
    for f in 0..6 {
        assert_eq!(suff.counts[f], 8.0);
    }
    for c in 6..18 {
        assert_eq!(suff.counts[c], 4.0);
    }
    for b in 18..22 {
        assert_eq!(suff.counts[b], 12.0);
    }
    // Crossed co-occurrence: each (primary, crossed) pair shares exactly
    // 2 rows in a full factorial of 6·4·2.
    assert_eq!(suff.zx[(0, 0)], 2.0);
    assert_eq!(suff.zx[(5, 3)], 2.0);
    // Same-factor crossed pairs never co-occur.
    assert_eq!(suff.zx[(18, 1)], 0.0);
    // Intercept-row sum of `s` = row count per level, on EVERY RE column
    // (primary, nested children, crossed) — a wrong scatter into any one of
    // them would otherwise slip past a check of column 0 alone.
    for col in 0..22 {
        assert!(
            (suff.s[(0, col)] - suff.counts[col]).abs() < 1e-12,
            "col {col}: s[(0,{col})] = {}, counts = {}",
            suff.s[(0, col)],
            suff.counts[col]
        );
    }
}

/// Textbook deviance on the explicit n×n V under `criterion` (see
/// [`profile_from_v`]), the oracle for the family-blocked elimination.
/// `groups[g]` = grouping g's global level ids (primary first); `theta[g]`
/// the matching component.
fn brute_force_deviance(
    theta: &[f64],
    x: &Mat<f64>,
    y: &[f64],
    groups: &[&[u32]],
    criterion: Criterion,
) -> f64 {
    let n = x.nrows();
    let mut v = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = 1.0;
    }
    for (g, ids) in groups.iter().enumerate() {
        let t2 = theta[g] * theta[g];
        for i in 0..n {
            for j in 0..n {
                if ids[i] == ids[j] {
                    v[(i, j)] += t2;
                }
            }
        }
    }
    profile_from_v(&v, x, y, criterion)
}

fn assert_deviance_matches_oracle(with_nested: bool, thetas: &[Vec<f64>]) {
    let (x, y, pid, eids, cluster) = multi_dataset(with_nested, 2);
    let g = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let mut suff = LmmSuffStats::with_groupings(3, g);
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let mut fit = LmmFitScratch::with_groupings(3, &gref);
    let mut fit_c = LmmFitScratch::with_groupings(3, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));
    // Oracle wants global ids per grouping.
    let mut groups: Vec<&[u32]> = vec![&pid];
    for e in &eids {
        groups.push(e);
    }
    // The collapse precompute holds no criterion term, so it serves both.
    for criterion in [Criterion::Reml, Criterion::Ml] {
        fit.criterion = criterion;
        fit_c.criterion = criterion;
        for th in thetas {
            let dev = reml_deviance(th, &suff, &mut fit);
            let oracle = brute_force_deviance(th, &x, &y, &groups, criterion);
            assert!(dev.is_finite(), "{criterion:?} θ={th:?}");
            let tol = 1e-8 * oracle.abs().max(1.0);
            assert!(
                (dev - oracle).abs() <= tol,
                "{criterion:?} θ={th:?}: family-blocked {dev} vs oracle {oracle}"
            );
            // Collapse arm: same θ through the balanced path; reassociation
            // band vs the loop, oracle band absolute.
            let dev_c = reml_deviance(th, &suff, &mut fit_c);
            let band = 1e-9 * dev.abs().max(1.0);
            assert!(
                (dev_c - dev).abs() <= band,
                "{criterion:?} θ={th:?}: collapse {dev_c} vs loop {dev}"
            );
            assert!(
                (dev_c - oracle).abs() <= tol,
                "{criterion:?} θ={th:?}: collapse vs oracle"
            );
        }
    }
}

#[test]
fn crossed_deviance_matches_brute_force() {
    assert_deviance_matches_oracle(
        false,
        &[
            vec![0.5, 0.3],
            vec![1.0, 1.0],
            vec![2.0, 0.1],
            vec![0.0, 0.7],
            vec![1e-3, 1e-3],
        ],
    );
}

#[test]
fn crossed_plus_nested_deviance_matches_brute_force() {
    assert_deviance_matches_oracle(
        true,
        &[
            vec![0.5, 0.3, 0.2],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.5, 0.9],
            vec![2.0, 0.05, 0.4],
        ],
    );
}

/// Nested-only in Regime B — the path with NO crossed tail (zx is 0×0)
/// and parents that grow with N.
#[test]
fn nested_regime_b_deviance_matches_brute_force() {
    let sizing = Sizing::FixedSize { cluster_size: 8 };
    let mut cluster = intercept_only_spec();
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        terms: Vec::new(),
        relation: GroupingRelation::NestedWithin,
        slopes: vec![],
    });
    let n = 4 * model_atom(&sizing, &[2]); // 64
    let mut st = 7u64;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut cid = vec![0u32; n];
    let u_p: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
    let u_c: Vec<f64> = (0..16).map(|_| 0.3 * lcg(&mut st)).collect();
    for i in 0..n {
        pid[i] = cluster_of_row(&sizing, i) as u32;
        cid[i] = extra_level_of_row(&cluster, &sizing, &[2], 0, i) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + u_p[pid[i] as usize] + u_c[cid[i] as usize] + 0.8 * lcg(&mut st);
    }
    let cluster = SizedSpec {
        spec: cluster,
        n_primary: sizing.n_clusters_at(n),
        extra_counts: vec![2],
    };
    let g = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    let eids = vec![cid.clone()];
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));
    for th in [[0.6, 0.4], [1.0, 1.0], [0.2, 0.0], [0.0, 0.0]] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &cid], Criterion::Reml);
        let tol = 1e-8 * oracle.abs().max(1.0);
        assert!((dev - oracle).abs() <= tol, "θ={th:?}: {dev} vs {oracle}");
        // Collapse arm — reassociation band vs the loop incl. the θ=0 edge.
        let dev_c = reml_deviance(&th, &suff, &mut fit_c);
        let band = 1e-9 * dev.abs().max(1.0);
        assert!(
            (dev_c - dev).abs() <= band,
            "θ={th:?}: collapse {dev_c} vs {dev}"
        );
    }
}

/// Balanced-collapse applicability: balanced intercept designs precompute,
/// slope groupings and unbalanced counts fall back, and a refused precompute
/// leaves the general loop's deviance bit-identical.
#[test]
fn balanced_collapse_applicability() {
    // Balanced: the regime-B nested dataset (atom-multiple by construction).
    let sizing = Sizing::FixedSize { cluster_size: 8 };
    let mut cluster = intercept_only_spec();
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        terms: Vec::new(),
        relation: GroupingRelation::NestedWithin,
        slopes: vec![],
    });
    let n = 4 * model_atom(&sizing, &[2]); // 64
    let max_n = 2 * n; // workspace sized for a larger grid top — active PREFIX
    let mut st = 7u64;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut cid = vec![0u32; n];
    for i in 0..n {
        pid[i] = cluster_of_row(&sizing, i) as u32;
        cid[i] = extra_level_of_row(&cluster, &sizing, &[2], 0, i) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + 0.8 * lcg(&mut st);
    }
    let cluster = SizedSpec {
        spec: cluster,
        n_primary: sizing.n_clusters_at(max_n),
        extra_counts: vec![2],
    };
    let g = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let n_primary = g.n_primary;
    let mut suff = LmmSuffStats::with_groupings(2, g);
    let eids = vec![cid.clone()];
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit));
    assert_eq!(fit.collapse_n_active, n / 8);
    assert!(fit.collapse_n_active < n_primary); // genuinely a prefix

    // Unbalanced: drop the last row — the trailing cluster is short.
    let mut suff_u =
        LmmSuffStats::with_groupings(2, LmmGroupings::from_cluster_spec(&cluster, &[]));
    let eids_u = vec![cid[..n - 1].to_vec()];
    suff_u.add_rows_multi(
        x.as_ref().subrows(0, n - 1),
        x.as_ref().subrows(0, n - 1).subcols(0, 0),
        &y[..n - 1],
        &pid[..n - 1],
        &eids_u,
        None,
    );
    assert!(!precompute_balanced_collapse(&suff_u, &mut fit));
    assert_eq!(fit.collapse_n_active, 0);
    // The refused scratch, collapse-armed a moment ago, now runs the general
    // loop bit for bit like a fresh one.
    let mut fit_fresh = LmmFitScratch::with_groupings(2, &gref);
    for th in [[0.6, 0.4], [1.0, 1.0], [0.0, 0.5]] {
        let a = reml_deviance(&th, &suff_u, &mut fit);
        let b = reml_deviance(&th, &suff_u, &mut fit_fresh);
        assert_eq!(a.to_bits(), b.to_bits(), "θ={th:?}");
    }

    // Slope path: never applicable — populated, balanced data, so the
    // rejection is the q_p guard, not the empty-suff early-out (balanced
    // slope counts would otherwise pass the count checks).
    let (xs, ys, ids_s) = slope_dataset();
    let gs = slope_groupings();
    let mut suff_s = LmmSuffStats::with_groupings(2, slope_groupings());
    suff_s.add_rows_multi(
        xs.as_ref(),
        xs.as_ref().subcols(0, 0),
        &ys,
        &ids_s,
        &[],
        None,
    );
    let mut fit_s = LmmFitScratch::with_groupings(2, &gs);
    assert!(!precompute_balanced_collapse(&suff_s, &mut fit_s));
}

/// Balanced collapse with prior weights: constant w ≡ 2 preserves the
/// per-cluster `counts` equality (`counts[f] = 2·n_f`, still exactly equal
/// across the balanced prefix), so the collapse must STILL trigger — and
/// the collapse-taken weighted fit must reproduce the unweighted one's
/// β/SE/tau2 (θ̃ = √c·θ maps the weighted profiled deviance onto the
/// unweighted one; θ̂² scales by 1/c, σ̂² by c, tau2 = θ²σ̂² invariant).
/// Both fits take the collapse branch (asserted below), so agreement is a
/// numeric check of the collapse kernel consuming weighted Grams, not just
/// of the accumulator.
#[test]
fn balanced_collapse_weighted_fit_invariant() {
    let (x, y, ids) = hand_dataset(); // balanced: 6 clusters × 8 rows
    let n = x.nrows();
    let targets: Vec<u32> = vec![1, 2];
    let w = vec![2.0f64; n];

    let mut ws_w = LmmWorkspace::new(3, 6);
    ws_w.suff_mut().add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &ids,
        &[],
        Some(&w),
    );
    let crate::lmm::LmmKernel::Dense { suff, fit } = &mut ws_w.kernel else {
        unreachable!("LmmWorkspace::new builds the dense kernel")
    };
    assert!(
        precompute_balanced_collapse(suff, fit),
        "constant weights keep exact per-cluster counts equality"
    );
    assert_eq!(ws_w.fit_mut().collapse_n_active, 6);
    let fit_w = fit_lmm_unhinted(&mut ws_w, &targets, None, ColumnScaling::IDENTITY);
    assert!(fit_w.converged);

    let mut ws_u = LmmWorkspace::new(3, 6);
    ws_u.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let fit_u = fit_lmm_unhinted(&mut ws_u, &targets, None, ColumnScaling::IDENTITY);
    assert!(fit_u.converged);

    // Two independent BOBYQA runs agree to the rho_end floor, not machine
    // precision — same 1e-6 relative band as the fit.rs invariance tests.
    for j in 0..3 {
        let (a, b) = (ws_u.recovery.betas[j], ws_w.recovery.betas[j]);
        assert!(
            (a - b).abs() / a.abs() < 1e-6,
            "β[{j}] unweighted {a} vs w≡2 {b}"
        );
    }
    for &tj in &targets {
        let (a, b) = (
            ws_u.recovery.var_diag[tj as usize].sqrt(),
            ws_w.recovery.var_diag[tj as usize].sqrt(),
        );
        assert!(
            (a - b).abs() / a < 1e-6,
            "se[{tj}] unweighted {a} vs w≡2 {b}"
        );
    }
    let (tu, tw) = (
        ws_u.theta[0] * ws_u.theta[0] * fit_u.sigma_sq,
        ws_w.theta[0] * ws_w.theta[0] * fit_w.sigma_sq,
    );
    assert!(
        (tu - tw).abs() / tu < 1e-6,
        "tau2 unweighted {tu} vs w≡2 {tw}"
    );
}

/// Two crossed factors — the dense cross-factor coupling block.
#[test]
fn two_crossed_factors_deviance_matches_brute_force() {
    let sizing = Sizing::FixedClusters { n_clusters: 3 };
    let counts = [4u32, 2u32];
    let mut cluster = intercept_only_spec();
    for _ in counts {
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::Crossed,
            slopes: vec![],
        });
    }
    let n = 2 * model_atom(&sizing, &counts); // 48
    let mut st = 21u64;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut e0 = vec![0u32; n];
    let mut e1 = vec![0u32; n];
    let u_p: Vec<f64> = (0..3).map(|_| 0.5 * lcg(&mut st)).collect();
    let u_a: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
    let u_b: Vec<f64> = (0..2).map(|_| 0.3 * lcg(&mut st)).collect();
    for i in 0..n {
        pid[i] = cluster_of_row(&sizing, i) as u32;
        e0[i] = extra_level_of_row(&cluster, &sizing, &counts, 0, i) as u32;
        e1[i] = extra_level_of_row(&cluster, &sizing, &counts, 1, i) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5
            + 0.4 * x1
            + u_p[pid[i] as usize]
            + u_a[e0[i] as usize]
            + u_b[e1[i] as usize]
            + 0.8 * lcg(&mut st);
    }
    let cluster = SizedSpec {
        spec: cluster,
        n_primary: 3,
        extra_counts: vec![4, 2],
    };
    let g = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    let eids = vec![e0.clone(), e1.clone()];
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, &[]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));
    for th in [[0.5, 0.4, 0.3], [1.0, 1.0, 1.0], [0.3, 0.0, 0.8]] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &e0, &e1], Criterion::Reml);
        let tol = 1e-8 * oracle.abs().max(1.0);
        assert!((dev - oracle).abs() <= tol, "θ={th:?}: {dev} vs {oracle}");
        // Collapse arm — reassociation band vs the loop.
        let dev_c = reml_deviance(&th, &suff, &mut fit_c);
        let band = 1e-9 * dev.abs().max(1.0);
        assert!(
            (dev_c - dev).abs() <= band,
            "θ={th:?}: collapse {dev_c} vs {dev}"
        );
    }
}

/// Per-component pin: items carry NO between-level signal by construction
/// (each item sees every subject equally, and the ±0.8 residual pattern is
/// block-constant so item means cancel exactly), while subjects carry a
/// real u_p. The crossed component must pin at exactly 0 (boundary_hit
/// == 1) with the primary component interior.
#[test]
fn zero_crossed_variance_pins_only_that_component() {
    let s_cl = 4usize;
    let i_cl = 3usize;
    let sizing = Sizing::FixedClusters {
        n_clusters: s_cl as u32,
    };
    let counts = [i_cl as u32];
    let mut cluster = SizedSpec {
        spec: intercept_only_spec(),
        n_primary: s_cl,
        extra_counts: vec![i_cl],
    };
    cluster
        .spec
        .re
        .as_mut()
        .unwrap()
        .extra_groupings
        .push(Grouping {
            terms: Vec::new(),
            relation: GroupingRelation::Crossed,
            slopes: vec![],
        });
    let n = 4 * model_atom(&sizing, &counts); // 48: 4 blocks ⇒ ±0.8 cancels per item
    let mut st = 5u64;
    let u_p: Vec<f64> = (0..s_cl).map(|_| 0.8 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        pid[i] = cluster_of_row(&sizing, i) as u32;
        eid[i] = extra_level_of_row(&cluster.spec, &sizing, &counts, 0, i) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let e = if (i / model_atom(&sizing, &counts)) % 2 == 0 {
            0.8
        } else {
            -0.8
        };
        y[i] = 0.5 + 0.4 * x1 + u_p[pid[i] as usize] + e;
    }
    let mut ws = LmmWorkspace::for_cluster_spec(2, &cluster, &[]);
    let eids = vec![eid];
    ws.suff_mut()
        .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    let fit = fit_lmm_unhinted(&mut ws, &[1], None, ColumnScaling::IDENTITY);
    assert!(fit.converged);
    assert_eq!(fit.boundary_hit, 1);
    assert_eq!(ws.theta[1], 0.0, "crossed component must pin at exact 0.0");
    assert!(
        ws.theta[0] > PIN_THETA,
        "primary component must stay interior"
    );
    assert!(fit.joint_t_sq.is_finite());
}

/// End-to-end crossed+nested fit recovers the generating β within wide
/// sanity bands and produces finite Wald machinery: an end-to-end smoke check
/// of the full multi-grouping pipeline.
#[test]
fn crossed_nested_fit_recovers_betas() {
    let (x, y, pid, eids, cluster) = multi_dataset(true, 4); // n = 192
    let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, &[]);
    ws.suff_mut()
        .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    let fit = fit_lmm_unhinted(&mut ws, &[1, 2], None, ColumnScaling::IDENTITY);
    assert!(fit.converged);
    assert!((ws.recovery.betas[1] - 0.4).abs() < 0.15);
    assert!((ws.recovery.betas[2] + 0.2).abs() < 0.15);
    // Deterministic regression lock (lcg-seeded multi_dataset) alongside the
    // planted-value recovers-check above, which documents intent.
    assert!(
        (ws.recovery.betas[1] - 0.40829926961384383).abs() / 0.40829926961384383_f64.abs() < 1e-6
    );
    assert!(
        (ws.recovery.betas[2] - -0.2916210839321183).abs() / 0.2916210839321183_f64.abs() < 1e-6
    );
    assert!(ws.recovery.t_sq[1].is_finite() && ws.recovery.t_sq[2].is_finite());
    assert!(fit.joint_t_sq.is_finite() && fit.joint_t_sq > 0.0);
    assert_eq!(ws.theta.len(), 3);
}

/// General-path twin of lmm_fit_warm_path_bounded_alloc: crossed+nested
/// workspace (the family loop is hand-rolled, zero-alloc). Warm-started from a cold prime fit's fitted θ (the loop tier's production
/// pattern), matching the few-eval regime the production path runs.
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn lmm_fit_general_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: usize = 100;
    // Measured 272 across the 100 calls (this machine), under 3 blocks per fit; the per-eval
    // Cholesky scratch is on the stack (`chol_lower`). The bound adds half a block per fit, so one
    // extra allocation per fit or per evaluation fails.
    const BOUND_GENERAL: u64 = 322;

    let (x, y, pid, eids, cluster) = multi_dataset(true, 2);
    let targets: Vec<u32> = vec![1, 2];
    let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, &[]);

    ws.suff_mut().reset();
    ws.suff_mut()
        .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
    // prime cold, then warm-start subsequent refits from the previous fit's fitted θ
    // (the loop tier's production pattern).
    let _ = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);
    let warm = ws.theta.clone();
    crate::test_support::settle_background_allocs();

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        ws.suff_mut().reset();
        ws.suff_mut()
            .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &pid, &eids, None);
        let _ = fit_lmm_unhinted(&mut ws, &targets, Some(&warm), ColumnScaling::IDENTITY);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert!(
        stats.total_blocks <= BOUND_GENERAL,
        "general fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
        stats.total_blocks,
        N_CALLS,
        BOUND_GENERAL
    );
}

// -----------------------------------------------------------------------
// Standalone primary slopes: q_p×q_p primary block, oracle deviance,
// diagonal-only pin. The kernel and the brute force read the same f64 data,
// so the 1e-8 band covers round-off only.
// -----------------------------------------------------------------------

/// n=64, p=2 (intercept + x1), 8 clusters, y carries u₀ + u₁·x1.
fn slope_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let (n, nc) = (64usize, 8usize);
    let mut st = 71u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + u0[c] + u1[c] * x1 + 0.8 * lcg(&mut st);
    }
    (x, y, ids)
}

/// n=96, p=3 (intercept + x1 + x2), 8 clusters, y carries u₀ + u₁·x1 + u₂·x2.
fn multislope_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let (n, nc) = (96usize, 8usize);
    let mut st = 91u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
    let u2: Vec<f64> = (0..nc).map(|_| 0.25 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        let (x1, x2) = (lcg(&mut st), lcg(&mut st));
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x2;
        y[i] = 0.5 + 0.4 * x1 + 0.2 * x2 + u0[c] + u1[c] * x1 + u2[c] * x2 + 0.8 * lcg(&mut st);
    }
    (x, y, ids)
}

/// Textbook deviance under `criterion` with a q×q D over the slope columns
/// of Z_p. `theta` is the column-major vech of Λ (q×q lower-tri); D_rel =
/// ΛΛ′ (σ-relative); V = I + Z·D_rel·Z′ with Z_i = [1, x[i, slope_cols]].
/// The f32 data is widened to f64 so the oracle reads the same bytes the
/// suff stats accumulated.
fn brute_force_slope_deviance(
    theta: &[f64],
    x: &Mat<f64>,
    y: &[f64],
    ids: &[u32],
    slope_cols: &[usize],
    q: usize,
    criterion: Criterion,
) -> f64 {
    let n = x.nrows();
    // Λ (q×q lower-tri) from column-major vech, then D = ΛΛ′.
    let mut lam = vec![0.0f64; q * q];
    let mut t = 0;
    for c in 0..q {
        for r in c..q {
            lam[r * q + c] = theta[t];
            t += 1;
        }
    }
    let mut d = vec![0.0f64; q * q];
    for i in 0..q {
        for j in 0..q {
            let mut s = 0.0;
            for k in 0..q {
                s += lam[i * q + k] * lam[j * q + k];
            }
            d[i * q + j] = s;
        }
    }
    let zrow = |i: usize| -> Vec<f64> {
        let mut z = vec![1.0];
        for &sc in slope_cols {
            z.push(x[(i, sc)]);
        }
        z
    };
    let mut v = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        v[(i, i)] += 1.0;
    }
    for i in 0..n {
        let zi = zrow(i);
        for j in 0..n {
            if ids[i] == ids[j] {
                let zj = zrow(j);
                let mut acc = 0.0;
                for a in 0..q {
                    for b in 0..q {
                        acc += zi[a] * d[a * q + b] * zj[b];
                    }
                }
                v[(i, j)] += acc;
            }
        }
    }
    profile_from_v(&v, x, y, criterion)
}

fn slope_groupings() -> LmmGroupings {
    LmmGroupings::from_cluster_spec(&slope_sized(), &[1])
}

/// 8 primary clusters, one slope on x_full col 1; no extras.
fn slope_sized() -> SizedSpec {
    SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![0],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    }
}

fn multislope_groupings() -> LmmGroupings {
    LmmGroupings::from_cluster_spec(&multislope_sized(), &[1, 2])
}

/// 8 primary clusters, two slopes on x_full cols 1,2; no extras.
fn multislope_sized() -> SizedSpec {
    SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![0, 1],
                extra_groupings: vec![],
            }),
        },
        n_primary: 8,
        extra_counts: vec![],
    }
}

/// Both slope-primary eliminations against the dense oracle at fixed θ, under
/// each criterion: the family loop (an unarmed scratch) and the per-group
/// regression route (`precompute_group_regressions`, whose precompute holds no
/// criterion term), which must also agree with each other on the tail the
/// β̂/SE code factors.
fn check_slope_routes_against_brute_force(
    x: &Mat<f64>,
    y: &[f64],
    ids: &[u32],
    g: &LmmGroupings,
    slope_cols: &[usize],
    thetas: &[Vec<f64>],
) {
    let (p, q) = (x.ncols(), g.primary_q);
    let mut suff = LmmSuffStats::with_groupings(p, g.clone());
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), y, ids, &[], None);
    let mut loop_fit = LmmFitScratch::with_groupings(p, g);
    let mut route_fit = LmmFitScratch::with_groupings(p, g);
    assert!(precompute_group_regressions(&suff, &mut route_fit));
    for criterion in [Criterion::Reml, Criterion::Ml] {
        loop_fit.criterion = criterion;
        route_fit.criterion = criterion;
        for th in thetas {
            let oracle = brute_force_slope_deviance(th, x, y, ids, slope_cols, q, criterion);
            let d_loop = reml_deviance(th, &suff, &mut loop_fit);
            let d_route = reml_deviance(th, &suff, &mut route_fit);
            for dev in [d_loop, d_route] {
                assert!(dev.is_finite(), "{criterion:?} θ={th:?}");
                assert!(
                    (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                    "{criterion:?} θ={th:?}: {dev} vs {oracle}"
                );
            }
            // The tails, [X y]′V⁻¹[X y], agree to rounding of the Gram they
            // are built from. Not the factors entrywise: at θ near THETA_HI
            // the family loop's `[X y]′[X y] − ΣB′B` cancels to ~1e-9 relative.
            for j in 0..=p {
                for i in j..=p {
                    let (a, b) = (
                        route_fit.tail[j * (p + 1) + i],
                        loop_fit.tail[j * (p + 1) + i],
                    );
                    let scale = (suff.c[(i, i)] * suff.c[(j, j)]).sqrt();
                    assert!(
                        (a - b).abs() <= 1e-12 * scale,
                        "{criterion:?} θ={th:?} tail ({i},{j}): {a} vs {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn slope_deviance_matches_brute_force() {
    let (x, y, ids) = slope_dataset();
    // θ = vech(Λ), q=2: [λ₀₀, λ₁₀, λ₁₁].
    let thetas = [
        vec![1.0, 0.0, 1.0],
        vec![0.5, 0.2, 0.4],
        vec![2.0, -0.5, 0.7],
        vec![1e-3, 1e-3, 1e-3],
        // D = 0 and a rank-one D: the route's identities hold on the
        // semidefinite boundary the fit pins to.
        vec![0.0, 0.0, 0.0],
        vec![0.8, -0.6, 0.0],
        // θ at THETA_HI (BOBYQA's box upper bound): the per-family Crout
        // pivot product must stay finite here — a product accumulated
        // across all families instead of reset per family would overflow.
        vec![THETA_HI, 0.0, THETA_HI],
    ];
    check_slope_routes_against_brute_force(&x, &y, &ids, &slope_groupings(), &[1], &thetas);
}

#[test]
fn multislope_deviance_matches_brute_force() {
    let (x, y, ids) = multislope_dataset();
    // θ = vech(Λ), q=3: [λ₀₀, λ₁₀, λ₂₀, λ₁₁, λ₂₁, λ₂₂].
    let thetas = [
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.6, 0.2, -0.1, 0.4, 0.15, 0.3],
        vec![1.5, -0.4, 0.3, 0.7, -0.2, 0.5],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.9, 0.3, -0.2, 0.0, 0.0, 0.0],
        vec![THETA_HI, 0.0, 0.0, THETA_HI, 0.0, THETA_HI],
    ];
    check_slope_routes_against_brute_force(&x, &y, &ids, &multislope_groupings(), &[1, 2], &thetas);
}

/// Balanced one-way `(1 | g)`, `a` groups of `m`, interior optimum: σ̂² = MSW
/// on both criteria, τ̂² = ((1 − 1/a)·MSB − MSW)/m under ML and
/// (MSB − MSW)/m under REML (Searle, Casella and McCulloch 1992, ch. 3). The
/// kernel at the analytic θ̂ = √(τ̂²/σ̂²) gives σ̂² to round-off. The
/// optimizer's θ̂ is accurate only to `RHO_END`, so it enters through its
/// deviance (equal to the analytic one: the gap is second order in δθ) and
/// through σ̂² and τ̂² in a 1e-5 band.
#[test]
fn balanced_one_way_matches_the_anova_estimators() {
    let (a, m) = (6usize, 5usize);
    let n = a * m;
    let mut st = 29u64;
    let u: Vec<f64> = (0..a).map(|_| 2.0 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 1);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % a) as u32;
        x[(i, 0)] = 1.0;
        y[i] = 1.0 + u[i % a] + 0.5 * lcg(&mut st);
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let mut group_mean = vec![0.0f64; a];
    for i in 0..n {
        group_mean[i % a] += y[i] / m as f64;
    }
    let ssw: f64 = (0..n).map(|i| (y[i] - group_mean[i % a]).powi(2)).sum();
    let ssb: f64 = group_mean
        .iter()
        .map(|g| m as f64 * (g - mean).powi(2))
        .sum();
    let (msw, msb) = (ssw / (n - a) as f64, ssb / (a - 1) as f64);
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    for (criterion, tau2) in [
        (Criterion::Reml, (msb - msw) / m as f64),
        (
            Criterion::Ml,
            ((1.0 - 1.0 / a as f64) * msb - msw) / m as f64,
        ),
    ] {
        assert!(tau2 > 0.0, "{criterion:?}: the optimum is not interior");
        let theta = (tau2 / msw).sqrt();
        let mut ws = LmmWorkspace::new(1, a);
        ws.fit_mut().criterion = criterion;
        ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
        let LmmKernel::Dense { suff, fit } = &mut ws.kernel else {
            unreachable!("LmmWorkspace::new builds the dense kernel")
        };
        let d_analytic = reml_deviance(&[theta], suff, fit);
        assert!(
            rel(fit.sigma_sq, msw) <= 1e-10,
            "{criterion:?}: σ̂² {} vs MSW {msw}",
            fit.sigma_sq
        );
        assert!(
            rel(theta * theta * fit.sigma_sq, tau2) <= 1e-10,
            "{criterion:?}: τ̂² vs {tau2}"
        );

        let lf = fit_lmm_unhinted(&mut ws, &[0], None, ColumnScaling::IDENTITY);
        assert!(lf.converged && lf.boundary_hit == 0, "{criterion:?}");
        let theta_opt = ws.theta[0];
        let LmmKernel::Dense { suff, fit } = &mut ws.kernel else {
            unreachable!()
        };
        let d_opt = reml_deviance(&[theta_opt], suff, fit);
        assert!(
            (d_opt - d_analytic).abs() <= 1e-10 * d_analytic.abs().max(1.0),
            "{criterion:?}: deviance at the optimizer's θ̂ {d_opt} vs analytic {d_analytic}"
        );
        assert!(
            rel(lf.sigma_sq, msw) <= 1e-5,
            "{criterion:?}: optimizer σ̂² {} vs {msw}",
            lf.sigma_sq
        );
        assert!(
            rel(theta_opt * theta_opt * lf.sigma_sq, tau2) <= 1e-5,
            "{criterion:?}: optimizer τ̂² vs {tau2}"
        );
    }
}

/// A group whose random-effect Gram is singular (here: one row against a
/// two-column `Z`) keeps the per-group regression route off, so the fit takes
/// the family loop, which needs no `G_g⁻¹`.
#[test]
fn group_regressions_refuse_a_singular_group_gram() {
    let (x, y, mut ids) = slope_dataset();
    // Cluster 7 keeps only its first row; the rest move to cluster 0.
    let mut seen = false;
    for id in ids.iter_mut() {
        if *id == 7 {
            if seen {
                *id = 0;
            }
            seen = true;
        }
    }
    let mut suff = LmmSuffStats::with_groupings(2, slope_groupings());
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
    let mut fit = LmmFitScratch::with_groupings(2, &slope_groupings());
    assert!(!precompute_group_regressions(&suff, &mut fit));
    assert_eq!(fit.pg_n_active, 0);
}

/// End-to-end single-slope fit recovers the planted structure within BOBYQA
/// bands and pins nothing on a well-identified design.
#[test]
fn slope_fit_converges_interior() {
    let (x, y, ids) = slope_dataset();
    let mut ws = LmmWorkspace::for_cluster_spec(2, &slope_sized(), &[1]);
    ws.suff_mut()
        .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
    let fit = fit_lmm_unhinted(&mut ws, &[1], None, ColumnScaling::IDENTITY);
    assert!(fit.converged);
    // Planted [intercept 0.5, slope 0.4]; small (n=64, 8 clusters) REML draw
    // recovers ≈[0.46, 0.20] — directionally correct, finite-sample attenuated.
    // Pin sign + a band tight enough to catch a sign flip, a collapse to 0, or a
    // blow-up (mere `is_finite` passed any of those).
    assert!(
        (0.2..0.8).contains(&ws.recovery.betas[0]),
        "intercept {}",
        ws.recovery.betas[0]
    );
    assert!(
        (0.05..0.6).contains(&ws.recovery.betas[1]),
        "slope {}",
        ws.recovery.betas[1]
    );
    // Deterministic regression lock alongside the bands above.
    assert!(
        (ws.recovery.betas[0] - 0.46265883331118085).abs() / 0.46265883331118085_f64.abs() < 1e-6
    );
    assert!(
        (ws.recovery.betas[1] - 0.20152611939449563).abs() / 0.20152611939449563_f64.abs() < 1e-6
    );
    assert_eq!(
        fit.pinned_components, 0,
        "a well-identified design pins no component"
    );
}

/// End-to-end two-slope fit (intercept + 2 slopes) whose REML optimum is a
/// rank-deficient Σ̂. On this draw the x2 slope is almost a linear combination of
/// the intercept and the x1 slope (corr(x1, x2) ≈ −0.997), so the trailing
/// Cholesky diagonal λ₂₂, the x2 slope's variance given the other two, pins
/// (bit 2) while the x2 slope's own variance (row 2 of Λ) stays positive. The
/// boundary is the optimum, not an optimizer stop: with the other five θ
/// re-optimized, the deviance rises monotonically in λ₂₂ (+5.9e-4 at 0.01,
/// +0.53 at 0.3), BOBYQA lands on the same pin at npt 2n+1 = 13 and at the
/// shipped ⌈1.5n⌉+1 = 10, and lme4 2.0-6 fits this data to the same θ̂ with
/// `isSingular` TRUE.
#[test]
fn multislope_fit_pins_rank_deficient_slope_block() {
    let (x, y, ids) = multislope_dataset();
    let mut ws = LmmWorkspace::for_cluster_spec(3, &multislope_sized(), &[1, 2]);
    ws.suff_mut()
        .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
    let fit = fit_lmm_unhinted(&mut ws, &[1, 2], None, ColumnScaling::IDENTITY);
    assert!(fit.converged);
    // Planted [0.5, 0.4, 0.2]; recovered ≈[0.51, 0.64, 0.28]. Both slopes positive
    // with β̂₁ > β̂₂ (planted ordering preserved) — pin that, so a β₁/β₂ swap or a
    // scale collapse fails, which a bare finiteness check would not catch.
    assert!(
        (0.2..0.9).contains(&ws.recovery.betas[0]),
        "intercept {}",
        ws.recovery.betas[0]
    );
    assert!(
        (0.2..1.1).contains(&ws.recovery.betas[1]),
        "slope x1 {}",
        ws.recovery.betas[1]
    );
    assert!(
        (0.0..0.7).contains(&ws.recovery.betas[2]),
        "slope x2 {}",
        ws.recovery.betas[2]
    );
    assert!(
        ws.recovery.betas[1] > ws.recovery.betas[2],
        "x1 slope must exceed x2 slope"
    );
    // Deterministic regression lock alongside the bands above.
    assert!(
        (ws.recovery.betas[0] - 0.5129839426148501).abs() / 0.5129839426148501_f64.abs() < 1e-6
    );
    assert!(
        (ws.recovery.betas[1] - 0.6442611282130077).abs() / 0.6442611282130077_f64.abs() < 1e-6
    );
    assert!(
        (ws.recovery.betas[2] - 0.28355377896623535).abs() / 0.28355377896623535_f64.abs() < 1e-6
    );
    assert_eq!(fit.pinned_components, 0b100, "only λ₂₂ pins");
    let x2_row = ws.theta[2].hypot(ws.theta[4]);
    assert!(
        x2_row > 0.1,
        "x2 slope variance collapsed: |Λ row 2| = {x2_row}"
    );
    // Recorded from glmm's own output. lme4 2.0-6 reaches the same optimum:
    // its REMLcrit 154.045763924748 is this plus (n − p)(1 + ln 2π).
    // Equal-or-better passes.
    const DEV_OPT: f64 = -109.876_803_251_320_7;
    assert!(
        fit.deviance <= DEV_OPT + 1e-9,
        "deviance {} above the optimum {DEV_OPT}",
        fit.deviance
    );
    let mut off = ws.theta.clone();
    off[5] = 0.01;
    let dev_off = ws.kernel.deviance(&off);
    assert!(
        dev_off > fit.deviance + 1e-4,
        "moving λ₂₂ off the pin did not raise the deviance: {dev_off} vs {}",
        fit.deviance
    );
}

// -----------------------------------------------------------------------
// Composition: primary slope (1 + x1 | g) co-existing with an
// intercept-only crossed (1 | item) / nested (1 | g:sub) extra. The
// family-blocked deviance must match a brute-force V = I + Z_p D_p Z_p′ +
// τ_e² Z_e Z_e′. The kernel and the oracle read the same f64 data, so the
// 1e-8 band covers round-off only.
// -----------------------------------------------------------------------

/// n=80, p=2 (intercept + x1), 8 primary clusters crossed with 5 items;
/// y carries u₀ + u₁·x1 (primary) + v (item intercept).
fn composed_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
    let (n, nc, ni) = (80usize, 8usize, 5usize);
    let mut st = 41u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
    let v: Vec<f64> = (0..ni).map(|_| 0.4 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let (mut pid, mut iid) = (vec![0u32; n], vec![0u32; n]);
    for i in 0..n {
        let (c, it) = (i % nc, i % ni);
        pid[i] = c as u32;
        iid[i] = it as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + u0[c] + u1[c] * x1 + v[it] + 0.8 * lcg(&mut st);
    }
    (x, y, pid, iid)
}

/// primary (1 + x1 | g), crossed (1 | item); slope on x_full col 1.
fn composed_groupings() -> LmmGroupings {
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![0],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![],
                }],
            }),
        },
        n_primary: 8,
        extra_counts: vec![5],
    };
    LmmGroupings::from_cluster_spec(&cluster, &[1])
}

// --- θ-layout generalization (scalar → vech ranges) ---

/// One intercept-only primary + one crossed grouping of RE width `q_g`
/// (intercept + `q_g−1` slopes), expressed through the slope machinery — the
/// θ-layout fixture. Slope columns are placeholders (layout reads only
/// `slopes.len()`).
fn groupings_primary1_crossed_qg(q_g: usize) -> LmmGroupings {
    let slopes: Vec<crate::ColumnId> = (0..q_g - 1).map(|k| (k + 1) as u32).collect();
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes,
                }],
            }),
        },
        n_primary: 8,
        extra_counts: vec![5],
    };
    LmmGroupings::from_cluster_spec(&cluster, &[])
}

#[test]
fn extra_theta_layout_qg1_and_qg2() {
    // Intercept-only crossed factor through the slope machinery = the plain
    // scalar layout: one primary scalar + one extra scalar.
    let g = groupings_primary1_crossed_qg(1);
    assert_eq!(g.n_theta(), 1 + 1);
    assert_eq!(g.crossed[0].vech_start, 1);
    assert_eq!(g.crossed[0].q, 1);
    assert!(!g.extra_slopes_any);

    let g = groupings_primary1_crossed_qg(2);
    assert_eq!(g.crossed[0].q, 2);
    assert_eq!(g.n_theta(), 1 + 3); // primary scalar + vech(2×2)=3
    assert!(g.extra_slopes_any);
    // The extra block's two diagonal θ indices are vech_start (=1) and
    // vech_start + 2 (=3) under the column-major lower-tri convention.
    let diag = &g.diagonal_theta;
    assert!(diag.contains(&1) && diag.contains(&3));
    // Off-diagonal λ₁₀ at index 2 is NOT a diagonal (signed box).
    assert!(!diag.contains(&2));
}

// --- Extra-slope sufficient statistics ---

/// Brute-force the `s` columns for a crossed factor carrying a slope: the
/// intercept subcol is Σ_{rows∈level} [X y]; the slope subcol is Σ x_slope·[X y].
#[test]
fn extra_crossed_slope_s_columns_match_bruteforce() {
    let n = 6usize;
    let p = 3; // [1, x1, x2]
    let xd = [
        (0.5, -0.2),
        (-0.3, 0.7),
        (0.9, 0.1),
        (-0.6, -0.4),
        (0.2, 0.8),
        (0.4, -0.5),
    ];
    let cluster_ids = [0u32, 1, 0, 1, 0, 1];
    let crossed_ids = [0u32, 1, 2, 0, 1, 2];
    let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = xd[i].0;
        x[(i, 2)] = xd[i].1;
    }
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 2,
        extra_counts: vec![3],
    };
    // crossed slope on x_full col 1.
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, &[], &[vec![1]]);
    // crossed block: q_g=2, offset = prim_width = 2 (n_primary=2, q_p=1).
    assert_eq!(g.extra_offsets[0], 2);
    assert_eq!(g.extra_q[0], 2);
    let mut suff = LmmSuffStats::with_groupings(p, g);
    suff.add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &cluster_ids,
        &[crossed_ids.to_vec()],
        None,
    );
    let m = p + 1;
    for c in 0..3usize {
        let icol = 2 + c * 2;
        let scol = icol + 1;
        let mut s_int = vec![0.0; m];
        let mut s_slope = vec![0.0; m];
        for i in 0..n {
            if crossed_ids[i] as usize == c {
                let w = [x[(i, 0)], x[(i, 1)], x[(i, 2)], y[i]];
                let x1 = x[(i, 1)];
                for j in 0..m {
                    s_int[j] += w[j];
                    s_slope[j] += x1 * w[j];
                }
            }
        }
        for j in 0..m {
            assert!(
                (suff.s[(j, icol)] - s_int[j]).abs() < 1e-12,
                "intercept col level {c} row {j}: got {} want {}",
                suff.s[(j, icol)],
                s_int[j]
            );
            assert!(
                (suff.s[(j, scol)] - s_slope[j]).abs() < 1e-12,
                "slope col level {c} row {j}: got {} want {}",
                suff.s[(j, scol)],
                s_slope[j]
            );
        }
        // counts only on the intercept subcol.
        let n_c = crossed_ids.iter().filter(|&&l| l as usize == c).count() as f64;
        assert_eq!(suff.counts[icol], n_c);
        assert_eq!(suff.counts[scol], 0.0);
    }
}

/// REML deviance on the explicit n×n V for the composed model: the 2×2
/// primary slope block (D_p = ΛΛ′ over [1, x1]) PLUS the extra-grouping
/// intercept block (θ_e² when the extra ids match). The f32 data is widened
/// to f64 so the oracle reads the same bytes the suff stats accumulated.
/// `eid` is the extra grouping's level id per row (item, or nested child).
/// θ = [primary vech λ₀₀, λ₁₀, λ₁₁ ; extra scalar θ_e].
fn brute_force_composed_deviance(
    theta: &[f64],
    x: &Mat<f64>,
    y: &[f64],
    pid: &[u32],
    eid: &[u32],
) -> f64 {
    let n = x.nrows();
    let (a, b, c) = (theta[0], theta[1], theta[2]);
    // D_p = ΛΛ′, Λ = [[a,0],[b,c]] (column-major vech).
    let (d00, d01, d11) = (a * a, a * b, b * b + c * c);
    let te2 = theta[3] * theta[3];
    let mut v = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        v[(i, i)] += 1.0;
    }
    for i in 0..n {
        for j in 0..n {
            if pid[i] == pid[j] {
                let (zi1, zj1) = (x[(i, 1)], x[(j, 1)]);
                v[(i, j)] += d00 + d01 * (zi1 + zj1) + d11 * zi1 * zj1;
            }
            if eid[i] == eid[j] {
                v[(i, j)] += te2;
            }
        }
    }
    profile_from_v(&v, x, y, Criterion::Reml)
}

/// Profiled deviance from an explicit n×n marginal V (in residual-σ²
/// units), r² = y′V⁻¹y − β̂′X′V⁻¹y: under REML `log|V| + log|XᵀV⁻¹X| +
/// (N−P)·log(r²/(N−P))`, under ML `log|V| + N·log(r²/N)`. The shared
/// V→deviance back end for every brute-force oracle (composed, crossed-slope, …).
fn profile_from_v(v: &Mat<f64>, x: &Mat<f64>, y: &[f64], criterion: Criterion) -> f64 {
    use faer::linalg::solvers::Solve;
    let (n, p) = (x.nrows(), x.ncols());
    let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
    let mut ldv = 0.0;
    for i in 0..n {
        ldv += vc.L()[(i, i)].ln();
    }
    let ldv = 2.0 * ldv;
    let mut vix = (*x).clone();
    vc.solve_in_place(vix.as_mut());
    let mut viy = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        viy[(i, 0)] = y[i];
    }
    vc.solve_in_place(viy.as_mut());
    let mut xtvix = Mat::<f64>::zeros(p, p);
    let mut xtviy = vec![0.0; p];
    for aa in 0..p {
        for bb in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, aa)] * vix[(i, bb)];
            }
            xtvix[(aa, bb)] = s;
        }
        let mut s = 0.0;
        for i in 0..n {
            s += x[(i, aa)] * viy[(i, 0)];
        }
        xtviy[aa] = s;
    }
    let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
    let mut ldk = 0.0;
    for aa in 0..p {
        ldk += kc.L()[(aa, aa)].ln();
    }
    let ldk = 2.0 * ldk;
    let mut beta = Mat::<f64>::zeros(p, 1);
    for aa in 0..p {
        beta[(aa, 0)] = xtviy[aa];
    }
    kc.solve_in_place(beta.as_mut());
    let mut ytviy = 0.0;
    for i in 0..n {
        ytviy += y[i] * viy[(i, 0)];
    }
    let mut bxy = 0.0;
    for aa in 0..p {
        bxy += beta[(aa, 0)] * xtviy[aa];
    }
    let r_sq = ytviy - bxy;
    match criterion {
        Criterion::Reml => {
            let df = (n - p) as f64;
            ldv + ldk + df * (r_sq / df).ln()
        }
        Criterion::Ml => ldv + n as f64 * (r_sq / n as f64).ln(),
    }
}

/// Brute-force REML deviance for a CROSSED-SLOPE model
/// `y ~ x1 + (1+x1 | primary) + (1+x1 | crossed)`: V = I + Z_p D_p Z_pᵀ +
/// Z_e D_e Z_eᵀ, each D a 2×2 from its vech θ over [1, x1]. θ =
/// [primary vech (3) ; crossed vech (3)].
fn brute_force_crossed_slope_deviance(
    theta: &[f64],
    x: &Mat<f64>,
    y: &[f64],
    pid: &[u32],
    eid: &[u32],
) -> f64 {
    let n = x.nrows();
    let (ap, bp, cp) = (theta[0], theta[1], theta[2]);
    let (dp00, dp01, dp11) = (ap * ap, ap * bp, bp * bp + cp * cp);
    let (ae, be, ce) = (theta[3], theta[4], theta[5]);
    let (de00, de01, de11) = (ae * ae, ae * be, be * be + ce * ce);
    let mut v = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        v[(i, i)] += 1.0;
    }
    for i in 0..n {
        for j in 0..n {
            let (zi, zj) = (x[(i, 1)], x[(j, 1)]);
            if pid[i] == pid[j] {
                v[(i, j)] += dp00 + dp01 * (zi + zj) + dp11 * zi * zj;
            }
            if eid[i] == eid[j] {
                v[(i, j)] += de00 + de01 * (zi + zj) + de11 * zi * zj;
            }
        }
    }
    profile_from_v(&v, x, y, Criterion::Reml)
}

/// Slope + crossed: the composed deviance matches the brute-force oracle to
/// 1e-8 — the slope-composition gate. zx_slope carries the slope↔crossed
/// coupling; the primary 2×2 block and the item intercept block are coupled
/// through the shared family-blocked tail.
#[test]
fn composed_deviance_matches_brute_force() {
    let (x, y, pid, iid) = composed_dataset();
    let mut suff = LmmSuffStats::with_groupings(2, composed_groupings());
    suff.add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &pid,
        std::slice::from_ref(&iid),
        None,
    ); // item ids as the single extra grouping
    let mut fit = LmmFitScratch::with_groupings(2, &composed_groupings());
    // θ = [λ₀₀, λ₁₀, λ₁₁, θ_c].
    for th in [
        vec![1.0, 0.0, 1.0, 0.5],
        vec![0.6, 0.2, 0.4, 0.3],
        vec![1.5, -0.4, 0.7, 0.8],
    ] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_composed_deviance(&th, &x, &y, &pid, &iid);
        assert!(dev.is_finite(), "θ={th:?}");
        assert!(
            (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
            "θ={th:?}: {dev} vs {oracle}"
        );
    }
}

/// CROSSED SLOPES (the headline lme4-agreement case): `y ~ x1 + (1+x1 | primary)
/// + (1+x1 | item)` — both grouping factors carry a random slope on x1, so the
/// gated blocked path runs. Deviance must match the explicit-V oracle to 1e-7
/// across θ, including the primary-slope↔crossed-slope coupling (the x1²
/// weighted co-occurrence) the blocked `zx` fill captures.
#[test]
fn crossed_slope_deviance_matches_brute_force() {
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 5,
        extra_counts: vec![4],
    };
    let n = 60; // atom = 5·4 = 20 ⇒ 3 balanced blocks
    let mut st = 91u64;
    let u0p: Vec<f64> = (0..5).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1p: Vec<f64> = (0..5).map(|_| 0.3 * lcg(&mut st)).collect();
    let u0e: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
    let u1e: Vec<f64> = (0..4).map(|_| 0.3 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        let sizing = Sizing::FixedClusters { n_clusters: 5 };
        let par = cluster_of_row(&sizing, i);
        let item = extra_level_of_row(&cluster.spec, &sizing, &[4], 0, i);
        pid[i] = par as u32;
        eid[i] = item as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5
            + 0.4 * x1
            + u0p[par]
            + u1p[par] * x1
            + u0e[item]
            + u1e[item] * x1
            + 0.8 * lcg(&mut st);
    }
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, &[1], &[vec![1]]);
    assert!(g.extra_slopes_any, "must route to the blocked path");
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &pid,
        &[eid.clone()],
        None,
    );
    let gref = LmmGroupings::from_cluster_spec_ext(&cluster, &[1], &[vec![1]]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    // θ = [primary vech (λ₀₀,λ₁₀,λ₁₁) ; crossed vech (λ₀₀,λ₁₀,λ₁₁)].
    for th in [
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0.7, 0.2, 0.5, 0.6, 0.1, 0.4],
        vec![1.3, -0.3, 0.6, 0.9, -0.2, 0.5],
    ] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_crossed_slope_deviance(&th, &x, &y, &pid, &eid);
        assert!(dev.is_finite(), "θ={th:?}");
        assert!(
            (dev - oracle).abs() <= 1e-7 * oracle.abs().max(1.0),
            "θ={th:?}: {dev} vs {oracle}"
        );
    }
}

/// NESTED SLOPES: `y ~ x1 + (1+x1 | grp) + (1+x1 | class)`,
/// class nested in grp — both grouping factors carry a random slope on x1, so
/// the gated blocked path runs with a nested factor of q_n = 2. A blocked path
/// that assembles the nested children intercept-only (scalar θ_n) diverges to
/// NaN here. The marginal V is grouping-agnostic (Σ_g Z_g D_g Z_gᵀ
/// over rows sharing a level id), so the crossed-slope oracle is reused with the
/// GLOBAL nested child id as the extra level. Matches the explicit-V oracle to
/// 1e-7 across θ.
#[test]
fn nested_slope_deviance_matches_brute_force() {
    let n_per_parent = 3u32;
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::NestedWithin,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 5,
        extra_counts: vec![n_per_parent as usize],
    };
    let n = 60; // atom = primary 5 · nested 3 = 15 ⇒ 4 balanced blocks
    let n_child = 5 * n_per_parent as usize;
    let mut st = 137u64;
    let u0p: Vec<f64> = (0..5).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1p: Vec<f64> = (0..5).map(|_| 0.3 * lcg(&mut st)).collect();
    let u0e: Vec<f64> = (0..n_child).map(|_| 0.4 * lcg(&mut st)).collect();
    let u1e: Vec<f64> = (0..n_child).map(|_| 0.3 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        let sizing = Sizing::FixedClusters { n_clusters: 5 };
        let par = cluster_of_row(&sizing, i);
        let child = extra_level_of_row(&cluster.spec, &sizing, &[n_per_parent], 0, i); // GLOBAL child id
        pid[i] = par as u32;
        eid[i] = child as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5
            + 0.4 * x1
            + u0p[par]
            + u1p[par] * x1
            + u0e[child]
            + u1e[child] * x1
            + 0.8 * lcg(&mut st);
    }
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, &[1], &[vec![1]]);
    assert!(g.extra_slopes_any, "must route to the blocked path");
    assert!(g.nested.is_some(), "must carry a nested factor");
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &pid,
        &[eid.clone()],
        None,
    );
    let gref = LmmGroupings::from_cluster_spec_ext(&cluster, &[1], &[vec![1]]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    // θ = [primary vech (λ₀₀,λ₁₀,λ₁₁) ; nested vech (λ₀₀,λ₁₀,λ₁₁)].
    for th in [
        vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0.7, 0.2, 0.5, 0.6, 0.1, 0.4],
        vec![1.3, -0.3, 0.6, 0.9, -0.2, 0.5],
    ] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_crossed_slope_deviance(&th, &x, &y, &pid, &eid);
        assert!(dev.is_finite(), "θ={th:?}");
        assert!(
            (dev - oracle).abs() <= 1e-7 * oracle.abs().max(1.0),
            "θ={th:?}: {dev} vs {oracle}"
        );
    }
}

/// General brute-force REML deviance: V = I + Σ_g Z_g D_g Z_gᵀ where each
/// factor `(ids, vech)` contributes a 2×2 D over [1, x1] (an intercept-only
/// factor passes `[θ, 0, 0]`). Used for the multi-crossed-factor oracle.
fn brute_force_slopes_deviance(x: &Mat<f64>, y: &[f64], factors: &[(&[u32], [f64; 3])]) -> f64 {
    let n = x.nrows();
    let mut v = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        v[(i, i)] += 1.0;
    }
    for &(ids, vech) in factors {
        let (a, b, c) = (vech[0], vech[1], vech[2]);
        let (d00, d01, d11) = (a * a, a * b, b * b + c * c);
        for i in 0..n {
            for j in 0..n {
                if ids[i] == ids[j] {
                    let (zi, zj) = (x[(i, 1)], x[(j, 1)]);
                    v[(i, j)] += d00 + d01 * (zi + zj) + d11 * zi * zj;
                }
            }
        }
    }
    profile_from_v(&v, x, y, Criterion::Reml)
}

/// TWO crossed factors with slopes:
/// `y ~ x1 + (1 | primary) + (1+x1 | c1) + (1+x1 | c2)`. Exercises the
/// crossed↔crossed slope coupling (c1's slope column against c2's, the x1²
/// weighted co-occurrence between two distinct crossed factors) — the part
/// neither the composed nor single-crossed test reaches. Matches the
/// explicit-V oracle to 1e-7.
#[test]
fn two_crossed_slopes_deviance_matches_brute_force() {
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![], // primary intercept-only
                extra_groupings: vec![
                    Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![1],
                    },
                    Grouping {
                        terms: Vec::new(),
                        relation: GroupingRelation::Crossed,
                        slopes: vec![1],
                    },
                ],
            }),
        },
        n_primary: 3,
        extra_counts: vec![3, 3],
    };
    let n = 54; // atom = 3·3·3 = 27 ⇒ 2 blocks
    let mut st = 73u64;
    let up: Vec<f64> = (0..3).map(|_| 0.45 * lcg(&mut st)).collect();
    let u0a: Vec<f64> = (0..3).map(|_| 0.4 * lcg(&mut st)).collect();
    let u1a: Vec<f64> = (0..3).map(|_| 0.3 * lcg(&mut st)).collect();
    let u0b: Vec<f64> = (0..3).map(|_| 0.35 * lcg(&mut st)).collect();
    let u1b: Vec<f64> = (0..3).map(|_| 0.28 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut c1 = vec![0u32; n];
    let mut c2 = vec![0u32; n];
    for i in 0..n {
        let sizing = Sizing::FixedClusters { n_clusters: 3 };
        let par = cluster_of_row(&sizing, i);
        let a = extra_level_of_row(&cluster.spec, &sizing, &[3, 3], 0, i);
        let b = extra_level_of_row(&cluster.spec, &sizing, &[3, 3], 1, i);
        pid[i] = par as u32;
        c1[i] = a as u32;
        c2[i] = b as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5
            + 0.4 * x1
            + up[par]
            + u0a[a]
            + u1a[a] * x1
            + u0b[b]
            + u1b[b] * x1
            + 0.8 * lcg(&mut st);
    }
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, &[], &[vec![1], vec![1]]);
    assert!(g.extra_slopes_any);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &pid,
        &[c1.clone(), c2.clone()],
        None,
    );
    let gref = LmmGroupings::from_cluster_spec_ext(&cluster, &[], &[vec![1], vec![1]]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    // θ = [primary scalar ; c1 vech (3) ; c2 vech (3)].
    for th in [
        vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        vec![0.6, 0.7, 0.2, 0.4, 0.6, -0.1, 0.35],
        vec![0.8, 1.2, -0.3, 0.5, 0.9, 0.25, 0.45],
    ] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_slopes_deviance(
            &x,
            &y,
            &[
                (&pid, [th[0], 0.0, 0.0]),
                (&c1, [th[1], th[2], th[3]]),
                (&c2, [th[4], th[5], th[6]]),
            ],
        );
        assert!(dev.is_finite(), "θ={th:?}");
        assert!(
            (dev - oracle).abs() <= 1e-7 * oracle.abs().max(1.0),
            "θ={th:?}: {dev} vs {oracle}"
        );
    }
}

/// Deterministic crossed-slope dataset for the lme4 golden: 8 primary
/// clusters × 6 crossed levels × 2 reps (n=96),
/// `y = 1.0 + 0.8·x1 + u0p + u1p·x1 + u0e + u1e·x1 + ε`. Mirrored operation for
/// operation by `tests/fixtures/gen_crossed_slope_lme4.R`, which rebuilds these
/// doubles for the `lme4::lmer` reference frozen in
/// `crossed_slope_fit_matches_lme4_golden` (change together);
/// `dump_crossed_slope_golden_csv` writes them as a CSV that script can check
/// against.
fn crossed_slope_golden_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
    let (n_prim, n_cross, n) = (8usize, 6usize, 96usize);
    let mut st = 20260629u64;
    let u0p: Vec<f64> = (0..n_prim).map(|_| 0.7 * lcg(&mut st)).collect();
    let u1p: Vec<f64> = (0..n_prim).map(|_| 0.5 * lcg(&mut st)).collect();
    let u0e: Vec<f64> = (0..n_cross).map(|_| 0.6 * lcg(&mut st)).collect();
    let u1e: Vec<f64> = (0..n_cross).map(|_| 0.4 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        let pp = i % n_prim; // FixedClusters primary: i % n_clusters
        let ee = (i / n_prim) % n_cross; // crossed: (i / n_prim) % n_cross
        pid[i] = pp as u32;
        eid[i] = ee as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] =
            1.0 + 0.8 * x1 + u0p[pp] + u1p[pp] * x1 + u0e[ee] + u1e[ee] * x1 + 0.5 * lcg(&mut st);
    }
    (x, y, pid, eid)
}

/// Run once (`cargo test -p ... dump_crossed_slope_golden_csv -- --ignored`)
/// to regenerate the CSV the R reference reads. Not a normal test.
#[test]
#[ignore]
fn dump_crossed_slope_golden_csv() {
    // Serialized under alloc-tests so its allocations can't land in a
    // concurrent dhat profiler window on an `-- --ignored` run.
    #[cfg(feature = "alloc-tests")]
    let _serial = crate::test_support::alloc_test_guard();
    let (x, y, pid, eid) = crossed_slope_golden_dataset();
    let mut s = String::from("x1,y,pid,eid\n");
    for i in 0..y.len() {
        s.push_str(&format!("{},{},{},{}\n", x[(i, 1)], y[i], pid[i], eid[i]));
    }
    std::fs::write("/tmp/crossed_slope_golden.csv", s).unwrap();
}

/// `glmm`'s crossed-slope fit must reproduce `lme4::lmer`'s REML fit
/// of `y ~ x1 + (1+x1|pid) + (1+x1|eid)` on the committed dataset — fixed
/// effects, residual σ², and both 2×2 RE covariances. Frozen from lme4 1.1 with
/// `optimizer = "bobyqa"`; `tests/fixtures/gen_crossed_slope_lme4.R` regenerates
/// them (registered in `validation/manifest.json`, `in_crate_references`).
/// Recovered D_g = σ̂²·Λ_gΛ_gᵀ from θ̂.
#[test]
fn crossed_slope_fit_matches_lme4_golden() {
    // lme4 golden (REML, bobyqa).
    const G_BETA0: f64 = 1.0582083262;
    const G_BETA1: f64 = 0.6334043248;
    const G_SIGMA2: f64 = 0.0921249591;
    const G_PID_V0: f64 = 0.1406815355; // var(intercept)
    const G_PID_V1: f64 = 0.1237856496; // var(x1)
    const G_PID_COV: f64 = 0.0127473486;
    const G_EID_V0: f64 = 0.1828301299;
    const G_EID_V1: f64 = 0.0396985129;
    const G_EID_COV: f64 = -0.0456611171;

    let (x, y, pid, eid) = crossed_slope_golden_dataset();
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 8,
        extra_counts: vec![6],
    };
    let mut ws = LmmWorkspace::for_cluster_spec_ext(2, &cluster, &[1], &[vec![1]], false);
    ws.suff_mut().reset();
    ws.suff_mut().add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &pid,
        std::slice::from_ref(&eid),
        None,
    );
    let fit = fit_lmm_unhinted(&mut ws, &[1], None, ColumnScaling::IDENTITY);
    assert!(fit.converged, "golden fit must converge");
    let s2 = fit.sigma_sq;

    // Fixed effects + residual variance.
    assert!(
        (ws.recovery.betas[0] - G_BETA0).abs() < 1e-4,
        "β0 {} vs {G_BETA0}",
        ws.recovery.betas[0]
    );
    assert!(
        (ws.recovery.betas[1] - G_BETA1).abs() < 1e-4,
        "β1 {} vs {G_BETA1}",
        ws.recovery.betas[1]
    );
    assert!(
        (s2 - G_SIGMA2).abs() <= 1e-3 * G_SIGMA2,
        "σ² {s2} vs {G_SIGMA2}"
    );

    // D_g = σ̂²·Λ_gΛ_gᵀ from θ̂ (primary vech θ[0..3], crossed vech θ[3..6]).
    let dblock = |t: &[f64]| {
        let (a, b, c) = (t[0], t[1], t[2]);
        (s2 * a * a, s2 * (b * b + c * c), s2 * a * b) // (v0, v1, cov)
    };
    let (pv0, pv1, pcov) = dblock(&ws.theta[0..3]);
    let (ev0, ev1, ecov) = dblock(&ws.theta[3..6]);
    let close = |got: f64, want: f64, name: &str| {
        assert!(
            (got - want).abs() <= 2e-3 * want.abs().max(1e-3),
            "{name}: {got} vs {want}"
        );
    };
    close(pv0, G_PID_V0, "pid var0");
    close(pv1, G_PID_V1, "pid var1");
    close(pcov, G_PID_COV, "pid cov");
    close(ev0, G_EID_V0, "eid var0");
    close(ev1, G_EID_V1, "eid var1");
    close(ecov, G_EID_COV, "eid cov");
}

/// Slope + NESTED: `(1 + x1 | g) + (1 | g:sub)` — the composed deviance with
/// a nested child tail (vs the crossed tail above). Exercises the
/// primary-slope↔child off-diagonal (read from `s`) and the shifted nested
/// offset `q_p·n_primary + f·np + c`. The nested child ids are globalized
/// (parent·np + within) — the workspace layout the contract helpers produce.
#[test]
fn composed_nested_deviance_matches_brute_force() {
    // 8 primary clusters × 2 children each, fixed-size 8 ⇒ 64 rows / 4 blocks.
    let sizing = Sizing::FixedSize { cluster_size: 8 };
    let spec = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            terms: Vec::new(),
            slopes: vec![0],
            extra_groupings: vec![Grouping {
                terms: Vec::new(),
                relation: GroupingRelation::NestedWithin,
                slopes: vec![],
            }],
        }),
    };
    let n = 4 * model_atom(&sizing, &[2]); // 64
    let cluster = SizedSpec {
        spec,
        n_primary: sizing.n_clusters_at(n),
        extra_counts: vec![2],
    };
    let mut st = 47u64;
    let u0: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..8).map(|_| 0.3 * lcg(&mut st)).collect();
    let u_c: Vec<f64> = (0..16).map(|_| 0.35 * lcg(&mut st)).collect(); // 8 parents × 2 children
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut cid = vec![0u32; n]; // globalized child id (parent·np + within)
    for i in 0..n {
        let par = cluster_of_row(&sizing, i);
        let child = extra_level_of_row(&cluster.spec, &sizing, &[2], 0, i); // already globalized par·np + within
        pid[i] = par as u32;
        cid[i] = child as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + u0[par] + u1[par] * x1 + u_c[child] + 0.8 * lcg(&mut st);
    }
    let g = LmmGroupings::from_cluster_spec(&cluster, &[1]);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &pid,
        &[cid.clone()],
        None,
    );
    let gref = LmmGroupings::from_cluster_spec(&cluster, &[1]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    // The brute-force oracle is V-shape-agnostic: the nested child block adds
    // θ_n² when the (globalized) child ids match — same form as the crossed.
    for th in [
        vec![1.0, 0.0, 1.0, 0.5],
        vec![0.7, 0.25, 0.5, 0.4],
        vec![1.3, -0.3, 0.6, 0.2],
    ] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_composed_deviance(&th, &x, &y, &pid, &cid);
        assert!(dev.is_finite(), "θ={th:?}");
        assert!(
            (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
            "θ={th:?}: {dev} vs {oracle}"
        );
    }
}

/// Bounded-allocation twin on the standalone slope workspace: the warm
/// `fit_lmm` loop allocates nothing, as on the q=1 path. `alloc_test_guard` serializes
/// test bodies, but libtest's own per-test thread spawn still needs
/// `--test-threads=1`:
///   RAYON_NUM_THREADS=1 cargo test -p glmm --features alloc-tests lmm_fit_slope_warm_path_bounded_alloc -- --ignored --test-threads=1
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn lmm_fit_slope_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: usize = 100;
    // Measured 0 (this machine), as on the q=1 path: the per-eval Cholesky scratch is on the stack
    // and the family loop and the primary Λ/Gram use scratch. Any heap block here is a regression.

    let (x, y, ids) = slope_dataset();
    let targets: Vec<u32> = vec![1];
    let mut ws = LmmWorkspace::for_cluster_spec(2, &slope_sized(), &[1]);

    ws.suff_mut().reset();
    ws.suff_mut()
        .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
    let _ = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);
    crate::test_support::settle_background_allocs();

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        ws.suff_mut().reset();
        ws.suff_mut()
            .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
        let _ = fit_lmm_unhinted(&mut ws, &targets, None, ColumnScaling::IDENTITY);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert_eq!(
        stats.total_blocks, 0,
        "slope fit_lmm allocated {} blocks across {} warm-path calls",
        stats.total_blocks, N_CALLS
    );
}

/// `fix_column_signs`: a block column whose diagonal is negative is negated
/// whole, so the fixed θ has non-negative diagonals and the SAME Σ = ΛΛᵀ to
/// the bit (every entry of Σ is a sum of products of two entries from one
/// column, and `(−a)·(−b)` is the double `a·b`). A θ with no negative
/// diagonal comes back bit-identical and reports no change.
#[test]
fn fix_column_signs_restores_nonnegative_diagonals_and_keeps_sigma_to_the_bit() {
    let base = SizedSpec {
        spec: intercept_only_spec(),
        n_primary: 4,
        extra_counts: vec![],
    };

    // q_p = 2, vech(Λ) = [λ00, λ10, λ11]: both diagonals negative.
    let mut spec2 = base.clone();
    spec2.spec.re.as_mut().unwrap().slopes.push(1);
    let g2 = LmmGroupings::from_cluster_spec(&spec2, &[1]);
    let mut th2 = [-0.7, 0.3, -0.5];
    let sig2 = sigma_vech(&th2, 2);
    assert!(fix_column_signs(&g2, &mut th2));
    assert_eq!(th2, [0.7, -0.3, 0.5]);
    for (got, want) in sigma_vech(&th2, 2).iter().zip(&sig2) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
    let again = th2;
    assert!(!fix_column_signs(&g2, &mut th2));
    assert_eq!(th2, again);

    // q_p = 3, vech(Λ) = [λ00, λ10, λ20, λ11, λ21, λ22]: columns 1 and 2
    // negative, column 0 left alone.
    let mut spec3 = base.clone();
    spec3.spec.re.as_mut().unwrap().slopes.push(1);
    spec3.spec.re.as_mut().unwrap().slopes.push(2);
    let g3 = LmmGroupings::from_cluster_spec(&spec3, &[1, 2]);
    let mut th3 = [0.5, -0.2, 0.1, -0.4, 0.3, -0.6];
    let sig3 = sigma_vech(&th3, 3);
    assert!(fix_column_signs(&g3, &mut th3));
    assert_eq!(th3, [0.5, -0.2, 0.1, 0.4, -0.3, 0.6]);
    for (got, want) in sigma_vech(&th3, 3).iter().zip(&sig3) {
        assert_eq!(got.to_bits(), want.to_bits());
    }

    // No negative diagonal ⇒ strict no-op, bit-for-bit.
    let mut th_live = [1.3, -0.37, 0.9, 0.2, -0.1, 0.4];
    let before = th_live;
    assert!(!fix_column_signs(&g3, &mut th_live));
    for (a, b) in th_live.iter().zip(&before) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

/// `fix_mode_signs`: the u entries of each block column with a negative
/// diagonal are negated at every level of that factor, on both primary
/// layouts, and nothing else moves. The expected indices are written out for
/// two small shapes — 5 primary levels with a slope (primary width 10), then a
/// crossed extra with 4 levels or a nested extra with 3 children per parent,
/// each with a slope — not recomputed from the layout formulas.
#[test]
fn fix_mode_signs_negates_the_modes_of_negative_columns() {
    let spec = |relation: GroupingRelation, count: usize| SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 5,
        extra_counts: vec![count],
    };
    let check = |g: &LmmGroupings, theta: &[f64], slope_major: bool, negated: &[usize]| {
        let mut u: Vec<f64> = (1..=g.k_total).map(|i| i as f64).collect();
        fix_mode_signs(g, theta, &mut u, slope_major);
        for (i, &v) in u.iter().enumerate() {
            let plain = (i + 1) as f64;
            let want = if negated.contains(&i) { -plain } else { plain };
            assert_eq!(v, want, "u[{i}], slope_major = {slope_major}");
        }
    };

    // Crossed, θ = [primary λ00 λ10 λ11 | crossed λ00 λ10 λ11]: primary
    // column 0 and crossed column 1 negative.
    let gc =
        LmmGroupings::from_cluster_spec_ext(&spec(GroupingRelation::Crossed, 4), &[1], &[vec![1]]);
    assert_eq!((gc.k_total, gc.extra_offsets.as_slice()), (18, &[10][..]));
    let th = [-0.7, 0.2, 0.5, 0.6, 0.1, -0.4];
    check(&gc, &th, false, &[0, 2, 4, 6, 8, 11, 13, 15, 17]);
    check(&gc, &th, true, &[0, 1, 2, 3, 4, 11, 13, 15, 17]);

    // Nested, same θ layout: primary column 1 and nested column 0 negative.
    let gn = LmmGroupings::from_cluster_spec_ext(
        &spec(GroupingRelation::NestedWithin, 3),
        &[1],
        &[vec![1]],
    );
    assert_eq!((gn.k_total, gn.extra_offsets.as_slice()), (40, &[10][..]));
    let th = [0.7, 0.2, -0.5, -0.6, 0.1, 0.4];
    let nested_col0 = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38];
    check(
        &gn,
        &th,
        false,
        &[&[1, 3, 5, 7, 9][..], &nested_col0].concat(),
    );
    check(
        &gn,
        &th,
        true,
        &[&[5, 6, 7, 8, 9][..], &nested_col0].concat(),
    );
}

/// A search that exits with a negative diagonal, and the exit sign fix on it.
/// `tests/fixtures/sign_trap_lmm_dense_slope.csv` is a simulated Gaussian
/// draw from the 2026-09-10 sign-trap simulation study: `y ~ x1 +
/// (1 + x1 | g1)`, 300 rows, 30 groups, random-slope SD 0.05, dense LMM
/// route. The raw BOBYQA endpoint under the
/// signed box (`blind_theta_and_bounds`) carries a negative diagonal; the
/// fixed θ has non-negative diagonals and the same REML deviance to the bit,
/// and it is exactly the θ̂ `fit_lmm` reports (no diagonal is near the pin
/// here, so the pin and the canonical block do nothing).
#[test]
fn fit_lmm_negative_diagonal_exit_is_sign_fixed_without_moving_the_deviance() {
    let csv = include_str!("../../tests/fixtures/sign_trap_lmm_dense_slope.csv");
    let mut y = Vec::<f64>::new();
    let mut x1 = Vec::<f64>::new();
    let mut g_raw = Vec::<String>::new();
    for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
        y.push(f[0].parse().unwrap());
        x1.push(f[1].parse().unwrap());
        g_raw.push(f[2].to_string());
    }
    let n = y.len();
    let mut labels: Vec<&str> = g_raw.iter().map(String::as_str).collect();
    labels.sort_unstable();
    labels.dedup();
    let ids: Vec<u32> = g_raw
        .iter()
        .map(|l| labels.iter().position(|x| x == l).unwrap() as u32)
        .collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1[i];
    }
    let model = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: labels.len(),
        extra_counts: vec![],
    };
    let build = || {
        let mut ws = LmmWorkspace::for_cluster_spec(2, &model, &[1]);
        ws.suff_mut()
            .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
        ws
    };

    // The raw search, exactly as `fit_lmm` drives it.
    let mut ws1 = build();
    let LmmWorkspace {
        solver,
        solver_warm,
        config,
        bfgs,
        kernel,
        theta,
        lower,
        upper,
        ..
    } = &mut ws1;
    let crate::lmm::LmmKernel::Dense { suff, fit } = &mut *kernel else {
        unreachable!("for_cluster_spec builds the dense kernel")
    };
    // `fit_lmm` arms the per-group regression route on this shape.
    assert!(precompute_group_regressions(suff, fit));
    let mut raw = theta.clone();
    let mut counters = crate::counters::EvalCounters::new();
    let (out, fell_back, _, _) = search_theta(
        kernel,
        solver,
        solver_warm,
        config,
        bfgs,
        &mut raw,
        lower,
        upper,
        false,
        None,
        &mut counters,
        &mut 0,
    );
    assert!(matches!(out.status, Status::Converged), "{:?}", out.status);
    assert!(!fell_back);
    let crate::lmm::LmmKernel::Dense { suff, fit } = kernel else {
        unreachable!("for_cluster_spec builds the dense kernel")
    };
    let diag = suff.groupings.diagonal_theta();
    assert!(
        diag.iter().any(|&i| raw[i] < 0.0),
        "the search must exit with a negative diagonal on this draw, got {raw:?}"
    );
    let d_raw = reml_deviance(&raw, suff, fit);
    let mut fixed = raw.clone();
    assert!(fix_column_signs(&suff.groupings, &mut fixed));
    assert!(diag.iter().all(|&i| fixed[i] >= 0.0), "{fixed:?}");
    assert_eq!(reml_deviance(&fixed, suff, fit).to_bits(), d_raw.to_bits());

    // The shipped fit reports that fixed θ, bit for bit.
    let mut ws2 = build();
    let f = fit_lmm_unhinted(&mut ws2, &[0, 1], None, ColumnScaling::IDENTITY);
    assert!(f.converged);
    assert_eq!(f.boundary_hit, 0, "no component pins on this draw");
    for (a, b) in ws2.theta.iter().zip(&fixed) {
        assert_eq!(a.to_bits(), b.to_bits(), "θ̂ {:?} vs {fixed:?}", ws2.theta);
    }
    assert_eq!(f.deviance.to_bits(), d_raw.to_bits());
}

/// A search that pins one Cholesky-factor diagonal to (near) zero while a
/// live entry sits below it in the same column: `canonicalize_pinned_blocks`
/// must re-derive Λ from Σ so the reported variance component reflects the
/// mass that is actually there, not the raw endpoint's near-zero coordinate.
/// Synthetic q_p=2 slope draw (10 clusters, 5 rows each, LCG seed 1) with the
/// slope covariate and the noise centred within each cluster, so every
/// cluster mean of y is exactly the grand intercept: the intercept variance
/// sits on its boundary (raw θ₀ lands near 4e-8; the profiled REML deviance
/// rises as ≈45·θ₀², so an endpoint above `PIN_THETA` would be ≥4.5e-7
/// worse), while the slope variance is live. With θ₀ at zero the criterion
/// sees the slope column only through θ₁² + θ₂², so the search leaves mass
/// in θ₁ (≈0.24) that the fold has to move into θ₂.
#[test]
fn fit_lmm_pin_rebuild_folds_the_raw_endpoint() {
    let nc = 10usize;
    let nper = 5usize;
    let n = nc * nper;
    let mut st = 1u64;
    let u1: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
    let mut x1 = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    for i in 0..n {
        x1[i] = lcg(&mut st);
        e[i] = 0.1 * lcg(&mut st);
    }
    for c in 0..nc {
        let rows = || (0..n).filter(move |i| i % nc == c);
        let mx = rows().map(|i| x1[i]).sum::<f64>() / nper as f64;
        let me = rows().map(|i| e[i]).sum::<f64>() / nper as f64;
        for i in rows() {
            x1[i] -= mx;
            e[i] -= me;
        }
    }
    // One more cluster with a single row at the grand intercept (`x₁ = 0`,
    // `y = 0.5`): its mean matches every other cluster's, so the intercept
    // variance stays on its boundary, and its two-column Z has a singular
    // Gram, which refuses the per-group regression route. The fold is the
    // behaviour under test, and BOBYQA's raw endpoint is the one that leaves
    // live mass below the pinned diagonal; the gradient route's endpoint on
    // the same draw already carries it in θ₂.
    let n_rows = n + 1;
    let mut x = Mat::<f64>::zeros(n_rows, 2);
    let mut y = vec![0.0f64; n_rows];
    let mut ids = vec![0u32; n_rows];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1[i];
        y[i] = 0.5 + 0.4 * x1[i] + u1[c] * x1[i] + e[i];
    }
    ids[n] = nc as u32;
    x[(n, 0)] = 1.0;
    y[n] = 0.5;
    // The slope column on the scale a fit of the design as given runs it at,
    // its root mean square (`ColumnScaling::compute`): the scale this draw's
    // pin was found at.
    let rms = ((0..n_rows).map(|i| x[(i, 1)] * x[(i, 1)]).sum::<f64>() / n_rows as f64).sqrt();
    for i in 0..n_rows {
        x[(i, 1)] /= rms;
    }
    let model = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        },
        n_primary: nc + 1,
        extra_counts: vec![],
    };
    let build = || {
        let mut ws = LmmWorkspace::for_cluster_spec(2, &model, &[1]);
        ws.suff_mut()
            .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
        ws
    };

    // The raw search, exactly as `fit_lmm` drives it, then the same
    // sign-fix + pin sequence `fit_lmm` runs after it.
    let mut ws1 = build();
    let LmmWorkspace {
        solver,
        solver_warm,
        config,
        bfgs,
        kernel,
        theta,
        lower,
        upper,
        ..
    } = &mut ws1;
    let crate::lmm::LmmKernel::Dense { suff, fit } = &mut *kernel else {
        unreachable!("for_cluster_spec builds the dense kernel")
    };
    // The single-row cluster keeps the per-group regression route off.
    assert!(!precompute_group_regressions(suff, fit));
    let mut raw = theta.clone();
    let mut counters = crate::counters::EvalCounters::new();
    let (out, fell_back, _, _) = search_theta(
        kernel,
        solver,
        solver_warm,
        config,
        bfgs,
        &mut raw,
        lower,
        upper,
        false,
        None,
        &mut counters,
        &mut 0,
    );
    assert!(matches!(out.status, Status::Converged), "{:?}", out.status);
    assert!(!fell_back);
    let crate::lmm::LmmKernel::Dense { suff, .. } = kernel else {
        unreachable!("for_cluster_spec builds the dense kernel")
    };
    fix_column_signs(&suff.groupings, &mut raw);
    let diag = suff.groupings.diagonal_theta().to_vec();
    assert!(
        raw[diag[0]] <= PIN_THETA,
        "this draw's search must land with the intercept diagonal at the pin floor, got {raw:?}"
    );
    let mut folded = raw.clone();
    for &ti in &diag {
        if folded[ti] <= PIN_THETA {
            folded[ti] = 0.0;
        }
    }
    assert!(canonicalize_pinned_blocks(&suff.groupings, &mut folded));
    // Not a round-off rewrite of an already-canonical Λ: θ₁'s mass moves.
    assert!(
        raw[1].abs() > 1e-2 && folded[1] == 0.0,
        "the pinned diagonal must carry live mass below it that the fold has to move, {raw:?} -> {folded:?}"
    );

    // The shipped fit reaches the same rebuild internally and reports the
    // folded θ̂, bit for bit.
    let mut ws2 = build();
    let f = fit_lmm_unhinted(&mut ws2, &[1], None, ColumnScaling::IDENTITY);
    assert!(f.converged);
    assert_eq!(f.boundary_hit, 1);
    for (a, b) in ws2.theta.iter().zip(&folded) {
        assert_eq!(a.to_bits(), b.to_bits(), "θ̂ {:?} vs {folded:?}", ws2.theta);
    }
}

/// A design with no rows accumulated has `reml_deviance` return `+INFINITY`
/// at every θ BOBYQA probes (the `n_rows <= p` guard, `kernel.rs`). BOBYQA's
/// interpolation model is built entirely from non-finite values, so the
/// search itself degenerates (`Status::ModelDegenerate`) rather than
/// converging or capping out — the only other path into `fit_lmm`'s NaN-fill
/// exit besides a non-finite re-evaluated deviance at a real endpoint.
#[test]
fn fit_lmm_nan_fills_on_model_degenerate_search() {
    let mut ws = LmmWorkspace::new(2, 6);
    let fit = fit_lmm_unhinted(&mut ws, &[1], None, ColumnScaling::IDENTITY);
    assert!(!fit.converged);
    assert_eq!(fit.boundary_hit, 2);
    assert_eq!(fit.pinned_components, 0);
    assert!(fit.sigma_sq.is_nan());
    assert!(fit.deviance.is_nan());
    assert!(fit.joint_t_sq.is_nan());
    assert!(ws.recovery.betas.iter().all(|b| b.is_nan()));
    assert!(ws.recovery.var_diag[1].is_nan());
    assert!(ws.recovery.t_sq[1].is_nan());
}

/// The cheap `+INFINITY` guards of `reml_deviance` (`kernel.rs`): too few
/// rows for the design (`n_rows <= p`), no fixed effects at all (`p == 0`),
/// and an exact fit. All three return the deviance failure sentinel rather
/// than a garbage finite value BOBYQA could accept as an optimum. The exact
/// fit also runs through `sparse_reml_deviance`, which shares the guard.
#[test]
fn reml_deviance_infinity_guards() {
    // n_rows <= p: nothing accumulated, p = 2.
    let suff_empty = LmmSuffStats::with_groupings(2, single_grouping(6));
    let mut fit_empty = LmmFitScratch::with_groupings(2, &single_grouping(6));
    assert_eq!(
        reml_deviance(&[1.0], &suff_empty, &mut fit_empty),
        f64::INFINITY
    );

    // p == 0: m = 1 (y only, no fixed effects), rows present so only the
    // `p == 0` half of the guard can be firing.
    let mut suff_p0 = LmmSuffStats::with_groupings(0, single_grouping(4));
    let x0 = Mat::<f64>::zeros(3, 0);
    suff_p0.add_rows(x0.as_ref(), &[1.0, 2.0, 3.0], &[0, 1, 2]);
    let mut fit_p0 = LmmFitScratch::with_groupings(0, &single_grouping(4));
    assert_eq!(reml_deviance(&[1.0], &suff_p0, &mut fit_p0), f64::INFINITY);

    // Exact fit: y = 2 + 3·x on x = 0.1·i, so r² is round-off. On x86_64 it
    // comes out as a positive ~1e-14 at every θ below (σ̂² ~ 2e-15, deviance
    // ~ −200 without the floor); the floor on r² is what refuses it.
    let n = 8;
    let mut x2 = Mat::<f64>::zeros(n, 2);
    let mut y_exact = vec![0.0; n];
    for i in 0..n {
        x2[(i, 0)] = 1.0;
        x2[(i, 1)] = 0.1 * i as f64;
        y_exact[i] = 2.0 + 3.0 * x2[(i, 1)];
    }
    let ids_exact: Vec<u32> = (0..n).map(|i| (i % 4) as u32).collect();
    let mut suff_exact = LmmSuffStats::with_groupings(2, single_grouping(4));
    suff_exact.add_rows(x2.as_ref(), &y_exact, &ids_exact);
    let mut fit_exact = LmmFitScratch::with_groupings(2, &single_grouping(4));
    let mut ws_sparse = crate::sparse::SparseLmmWorkspace::new(
        &single_grouping(4),
        x2.as_ref(),
        x2.as_ref().subcols(0, 0),
        &ids_exact,
        &[],
        &y_exact,
        n,
        2,
        None,
        crate::lmm::Criterion::Reml,
    );
    for th in [0.0, 0.5, 1.0] {
        assert_eq!(
            reml_deviance(&[th], &suff_exact, &mut fit_exact),
            f64::INFINITY,
            "dense, θ = {th}"
        );
        assert_eq!(
            crate::sparse::sparse_reml_deviance(&[th], &mut ws_sparse),
            f64::INFINITY,
            "sparse, θ = {th}"
        );
    }
}

/// The blocked twin (`reml_deviance_blocked`, crossed/nested random slopes)
/// hits the same failure surface: a perfect fit (X exactly spans y, no
/// residual) makes the augmented `[Λ′ZᵀZΛ+I | Λ′Zᵀ[Xy]; · | [Xy]ᵀ[Xy]]`
/// factor exactly rank-deficient at θ = 0 (Λ = 0 collapses the RE penalty to
/// the identity, so the trailing [X y] block alone is singular at the exact
/// fit). In floating point its last pivot is round-off of either sign
/// (174 − 147 − 27 here), so which guard fires is platform rounding: a
/// non-positive pivot fails `chol_lower` (native), a tiny positive one is
/// caught by the r² round-off floor (wasm32). Either must give +inf.
#[test]
fn reml_deviance_blocked_infinity_guard() {
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    terms: Vec::new(),
                    relation: GroupingRelation::Crossed,
                    slopes: vec![1],
                }],
            }),
        },
        n_primary: 3,
        extra_counts: vec![3],
    };
    let n = 12;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        pid[i] = (i % 3) as u32;
        eid[i] = ((i / 3) % 3) as u32;
        let x1 = (i % 2) as f64;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 2.0 + 3.0 * x1; // exact linear fit, zero residual
    }
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, &[], &[vec![1]]);
    assert!(g.extra_slopes_any, "must route to the blocked path");
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        &y,
        &pid,
        &[eid.clone()],
        None,
    );
    let gref = LmmGroupings::from_cluster_spec_ext(&cluster, &[], &[vec![1]]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    let th = vec![0.0; gref.n_theta()];
    assert_eq!(reml_deviance(&th, &suff, &mut fit), f64::INFINITY);
}

/// With a single target, the joint Wald-χ² collapses to the per-target `t²`
/// (`W = β̂_T'[(K⁻¹)_TT]⁻¹β̂_T/σ̂²` at k=1 is `β̂²/Var(β̂)`, the same quantity
/// `t_sq` already holds). Every multi-grouping fit test checks `joint_t_sq`
/// with `is_finite` alone, so a wrong Σ_T gather in `joint_wald_chi_sq` would
/// pass unnoticed on a single-target design.
#[test]
fn joint_wald_matches_single_target_t_sq() {
    let (x, y, ids) = hand_dataset();
    let mut ws = LmmWorkspace::new(3, 6);
    ws.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let fit = fit_lmm_unhinted(&mut ws, &[1], None, ColumnScaling::IDENTITY);
    assert!(fit.converged);
    let want = ws.recovery.t_sq[1];
    assert!(want.is_finite());
    let band = 1e-9 * want.abs().max(1.0);
    assert!(
        (fit.joint_t_sq - want).abs() <= band,
        "joint {} vs t_sq[1] {}",
        fit.joint_t_sq,
        want
    );
}

/// A near-exact-duplicate design (x2 = x1 + 1e-12·noise) is so ill-conditioned
/// that almost every θ BOBYQA probes fails the Cholesky and moderates to
/// PRIMA's `FUNCMAX` — a flat surface that can still report `Status::Converged`
/// (`SMALL_TR_RADIUS` on a moderated surface, `bobyqa::FUNCMAX`'s own
/// documented case). Here the search sees only one genuinely finite
/// evaluation before reporting Converged; the reported θ̂ happens to
/// re-evaluate finite too, so nothing else would mark this endpoint
/// unusable. `finite_evals >= 2` is what does — deleting that conjunct
/// would report this one-finite-point draw as a converged fit with real
/// (non-NaN) β̂/σ̂².
#[test]
fn converged_status_with_one_finite_eval_is_not_reported_converged() {
    let n = 48usize;
    let nc = 8usize;
    let mut st = 3u64;
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % nc) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x1 + 1e-12 * lcg(&mut st); // near-exact duplicate of x1
        y[i] = 0.5 + 0.4 * x1 + 0.8 * lcg(&mut st);
    }

    // Witness: the raw search, run exactly as `fit_lmm` drives it, converges
    // with fewer than 2 finite evaluations.
    let mut ws1 = LmmWorkspace::new(3, nc);
    ws1.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let LmmWorkspace {
        solver,
        kernel,
        theta,
        lower,
        upper,
        ..
    } = &mut ws1;
    let crate::lmm::LmmKernel::Dense { suff, fit } = kernel else {
        unreachable!("LmmWorkspace::new builds the dense kernel")
    };
    let mut raw = theta.clone();
    let mut finite_evals = 0usize;
    let out = solver.minimize(
        |t| {
            let d = reml_deviance(t, suff, fit);
            if d.is_finite() {
                finite_evals += 1;
            }
            d
        },
        &mut raw,
        lower,
        upper,
    );
    assert!(matches!(out.status, Status::Converged), "{:?}", out.status);
    assert!(
        finite_evals < 2,
        "this draw's search must see fewer than 2 finite evaluations, got {finite_evals}"
    );

    // The shipped fit must not call this converged, even though its θ̂
    // re-evaluates finite and every other field looks like a normal fit.
    let mut ws2 = LmmWorkspace::new(3, nc);
    ws2.suff_mut().add_rows(x.as_ref(), &y, &ids);
    let f = fit_lmm_unhinted(&mut ws2, &[1, 2], None, ColumnScaling::IDENTITY);
    assert!(
        !f.converged,
        "too few finite evaluations must not be reported converged"
    );
    assert!(
        f.deviance.is_finite(),
        "this endpoint is a real fit, not the NaN-fill exit"
    );
    assert!(f.sigma_sq.is_finite());
    assert!(ws2.recovery.betas.iter().all(|b| b.is_finite()));
}

/// `DesignQr` reads its rows through a row-major tile (`load_tile`), which
/// copies contiguous columns by slice and any other view element by element.
/// Both copies must give the same factor, bit for bit: a strided view (the
/// transpose of a `p×n` matrix) against the contiguous `n×p` one, at a width
/// with a fixed-width instance (p = 3) and one without (p = 7), over more rows
/// than one tile holds.
#[test]
fn design_qr_reads_a_strided_design_like_a_contiguous_one() {
    for p in [3usize, 7] {
        let n = 150;
        let mut st = 46u64;
        let x = faer::Mat::<f64>::from_fn(n, p, |_, j| if j == 0 { 1.0 } else { lcg(&mut st) });
        let xt = x.transpose().to_owned();
        let mut v = vec![0.0; p];
        let mut a = DesignQr::identity(p);
        let mut b = DesignQr::identity(p);
        assert!(a.set_from_rows(x.as_ref(), |i| 1.0 + 0.01 * i as f64, &mut v));
        assert!(b.set_from_rows(xt.as_ref().transpose(), |i| 1.0 + 0.01 * i as f64, &mut v));
        let bits = |s: &[f64]| s.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(a.r_diag()), bits(b.r_diag()), "p = {p}: D");
        let mut ra: Vec<f64> = (0..p).map(|j| 1.0 + j as f64).collect();
        let mut rb = ra.clone();
        a.solve_row(&mut ra);
        b.solve_row(&mut rb);
        assert_eq!(bits(&ra), bits(&rb), "p = {p}: x·U⁻¹");
    }
}

// --- per-group route gradient ---

/// A random-slopes dataset with `q − 1` slopes on the primary grouping:
/// `nc` clusters of `per` rows, `x = [1, x₁ … x_{q−1}, extra]` (one extra
/// fixed column), and the groupings that put every slope on the primary.
fn pg_dataset(
    q: usize,
    nc: usize,
    per: usize,
    seed: u64,
) -> (Mat<f64>, Vec<f64>, Vec<u32>, LmmGroupings) {
    let n = nc * per;
    let p = q + 1;
    let mut st = seed;
    let u: Vec<Vec<f64>> = (0..q)
        .map(|k| {
            (0..nc)
                .map(|_| (0.6 / (1.0 + k as f64)) * lcg(&mut st))
                .collect()
        })
        .collect();
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % nc;
        ids[i] = c as u32;
        x[(i, 0)] = 1.0;
        let mut yi = 0.3 + u[0][c];
        for k in 1..q {
            let v = lcg(&mut st);
            x[(i, k)] = v;
            yi += (0.2 + 0.1 * k as f64) * v + u[k][c] * v;
        }
        let e = lcg(&mut st);
        x[(i, q)] = e;
        y[i] = yi + 0.5 * e + 0.7 * lcg(&mut st);
    }
    let cluster = SizedSpec {
        spec: ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                terms: Vec::new(),
                slopes: (0..q as u32 - 1).collect(),
                extra_groupings: vec![],
            }),
        },
        n_primary: nc,
        extra_counts: vec![],
    };
    let slope_cols: Vec<usize> = (1..q).collect();
    let g = LmmGroupings::from_cluster_spec(&cluster, &slope_cols);
    (x, y, ids, g)
}

/// The gradient against central differences of the deviance at each θ,
/// under `criterion`. The differences are read off three evaluators of the
/// same deviance, each checked against the route's value first: the route's
/// own dense scratch, the dense blocked elimination (`reml_deviance_blocked`,
/// reached by setting `extra_slopes_any` on the same groupings) and the
/// sparse workspace. Only free coordinates are differenced: on a masked `g`
/// each θ has its structural entries zeroed first, and those entries are
/// never stepped. Returns the number of coordinate checks.
#[allow(clippy::too_many_arguments)]
fn check_pg_gradient(
    x: &Mat<f64>,
    y: &[f64],
    ids: &[u32],
    w: Option<&[f64]>,
    g: &LmmGroupings,
    thetas: &[Vec<f64>],
    criterion: Criterion,
) -> usize {
    let (n_rows, p) = (x.nrows(), x.ncols());
    let mut suff = LmmSuffStats::with_groupings(p, g.clone());
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), y, ids, &[], w);
    let mut fit = LmmFitScratch::with_groupings(p, g);
    fit.criterion = criterion;
    assert!(precompute_group_regressions(&suff, &mut fit));
    let mut gb = g.clone();
    gb.extra_slopes_any = true;
    let mut suff_b = LmmSuffStats::with_groupings(p, gb.clone());
    suff_b.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), y, ids, &[], w);
    let mut fit_b = LmmFitScratch::with_groupings(p, &gb);
    fit_b.criterion = criterion;
    let sqrt_w: Option<Vec<f64>> = w.map(|w| w.iter().map(|v| v.sqrt()).collect());
    let mut sp = crate::sparse::SparseLmmWorkspace::new(
        g,
        x.as_ref(),
        x.as_ref().subcols(0, 0),
        ids,
        &[],
        y,
        n_rows,
        p,
        sqrt_w.as_deref(),
        criterion,
    );
    let mut checks = 0;
    for th in thetas {
        let mut th = th.clone();
        g.zero_structural(&mut th);
        let th = &th;
        let n = th.len();
        let f0 = reml_deviance(th, &suff, &mut fit);
        assert!(f0.is_finite(), "{criterion:?} θ={th:?}");
        let mut grad = vec![0.0; n];
        assert!(
            pg_grad::pg_gradient(&suff, &mut fit, &mut grad),
            "{criterion:?} θ={th:?}"
        );
        let mut eval = |kernel: usize, t: &[f64]| match kernel {
            0 => reml_deviance(t, &suff, &mut fit),
            1 => reml_deviance(t, &suff_b, &mut fit_b),
            _ => crate::sparse::sparse_reml_deviance(t, &mut sp),
        };
        for (kernel, name) in ["route", "blocked", "sparse"].into_iter().enumerate() {
            let fk = eval(kernel, th);
            assert!(
                (fk - f0).abs() <= 1e-8 * f0.abs().max(1.0),
                "{criterion:?} θ={th:?} {name}: {fk} vs route {f0}"
            );
            for i in (0..n).filter(|i| g.structural_theta().binary_search(i).is_err()) {
                let h = 1e-5 * th[i].abs().max(1.0);
                let (mut tp, mut tm) = (th.clone(), th.clone());
                tp[i] += h;
                tm[i] -= h;
                let fd = (eval(kernel, &tp) - eval(kernel, &tm)) / (2.0 * h);
                assert!(
                    (grad[i] - fd).abs() <= 2e-5 * fd.abs().max(1.0),
                    "{criterion:?} θ={th:?} {name} ∂{i}: analytic {} vs FD {fd}",
                    grad[i]
                );
                checks += 1;
            }
        }
    }
    checks
}

fn vech_identity(q: usize, s: f64) -> Vec<f64> {
    let mut v = Vec::new();
    for c in 0..q {
        for r in c..q {
            v.push(if r == c { s } else { 0.0 });
        }
    }
    v
}

fn vech_random(q: usize, seed: u64) -> Vec<f64> {
    let mut st = seed;
    let mut v = Vec::new();
    for c in 0..q {
        for r in c..q {
            let u = lcg(&mut st);
            v.push(if r == c { 0.4 + 0.5 * u.abs() } else { 0.3 * u });
        }
    }
    v
}

/// One case per block width, plus the masked shape
/// `(1 + x1 | g) + (0 + x2 | g)` on the `q = 3` design (θ entries 2 and 4
/// structural, differenced along the other four). Prints the check count of
/// each case.
#[test]
fn pg_gradient_matches_finite_differences() {
    let mut counts = Vec::new();
    for (q, nc, per, masked) in [
        (2usize, 12usize, 9usize, false),
        (3, 10, 12, false),
        (4, 10, 14, false),
        (8, 10, 24, false),
        (3, 10, 12, true),
    ] {
        let (x, y, ids, mut g) = pg_dataset(q, nc, per, 17 + q as u64);
        if masked {
            let spec = SizedSpec {
                spec: ModelSpec {
                    family: Family::Gaussian,
                    re: Some(Re::intercept().terms([
                        ReTerm::intercept().slopes([1]),
                        ReTerm::no_intercept().slopes([2]),
                    ])),
                },
                n_primary: nc,
                extra_counts: vec![],
            };
            g = LmmGroupings::from_cluster_spec(&spec, &[1, 2]);
            assert_eq!(g.structural_theta(), [2, 4]);
        }
        let mut rank_one = vech_identity(q, 0.0);
        for (k, v) in rank_one.iter_mut().take(q).enumerate() {
            *v = 0.8 - 0.1 * k as f64; // first column only: D = l l′
        }
        let mut near_singular = vech_random(q, 5);
        let last = near_singular.len() - 1;
        near_singular[last] = 1e-4;
        let thetas = vec![
            vech_identity(q, 1.0),
            vech_random(q, 3),
            near_singular,
            rank_one,
            vech_identity(q, 1e-3),
            vech_identity(q, 40.0),
        ];
        let mut checks = 0;
        for criterion in [Criterion::Reml, Criterion::Ml] {
            checks += check_pg_gradient(&x, &y, &ids, None, &g, &thetas, criterion);
        }
        counts.push((q, masked, checks));
    }
    for (q, masked, checks) in &counts {
        println!("pg_gradient FD gate: q {q} masked {masked}: {checks} checks");
    }
    // Two criteria × six θ × three evaluators × four free coordinates.
    assert_eq!(counts.last(), Some(&(3, true, 2 * 6 * 3 * 4)));
}

#[test]
fn pg_gradient_matches_finite_differences_weighted() {
    let (x, y, ids, g) = pg_dataset(3, 10, 12, 29);
    let mut st = 41u64;
    let w: Vec<f64> = (0..y.len()).map(|_| 1.25 + 0.75 * lcg(&mut st)).collect();
    for criterion in [Criterion::Reml, Criterion::Ml] {
        check_pg_gradient(
            &x,
            &y,
            &ids,
            Some(&w),
            &g,
            &[vech_identity(3, 1.0), vech_random(3, 7)],
            criterion,
        );
    }
}

/// The gradient is taken at the LAST evaluation's θ: evaluating θ₁ then θ₂
/// and asking for the gradient gives θ₂'s, bit for bit.
#[test]
fn pg_gradient_reads_the_last_evaluation() {
    let (x, y, ids, g) = pg_dataset(3, 10, 12, 31);
    let p = x.ncols();
    let mut suff = LmmSuffStats::with_groupings(p, g.clone());
    suff.add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
    let mut fit = LmmFitScratch::with_groupings(p, &g);
    assert!(precompute_group_regressions(&suff, &mut fit));
    let (t1, t2) = (vech_random(3, 1), vech_random(3, 2));
    let mut ga = vec![0.0; 6];
    let mut gb = vec![0.0; 6];
    reml_deviance(&t2, &suff, &mut fit);
    assert!(pg_grad::pg_gradient(&suff, &mut fit, &mut ga));
    reml_deviance(&t1, &suff, &mut fit);
    reml_deviance(&t2, &suff, &mut fit);
    assert!(pg_grad::pg_gradient(&suff, &mut fit, &mut gb));
    assert_eq!(
        ga.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        gb.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

/// More groups than one stacked-solve block: the blocked solve gives the
/// same gradient as finite differences (the block boundary is invisible).
#[test]
fn pg_gradient_spans_solve_blocks() {
    let nc = pg_grad::PG_GRAD_BLOCK + 37;
    let (x, y, ids, g) = pg_dataset(2, nc, 4, 43);
    for criterion in [Criterion::Reml, Criterion::Ml] {
        check_pg_gradient(&x, &y, &ids, None, &g, &[vech_random(2, 9)], criterion);
    }
}

/// Under ML the BFGS route's endpoint is equal to or better than the BOBYQA
/// route's on the same data: its deviance is at most BOBYQA's `+ 1e-6`
/// (absolute), the band of
/// `glmm::tests::gradient_route_matches_or_beats_the_bobyqa_route`. Each
/// search runs from the blind start through [`search_theta`], as `fit_lmm`
/// drives it: once with the per-group route armed
/// (`precompute_group_regressions`, so BFGS at `n_θ ≥ 3`) and once unarmed
/// (`pg_n_active = 0`, so BOBYQA). It guards the endpoint only; the gradient
/// itself is pinned by `pg_gradient_matches_finite_differences` (and its
/// weighted and `pg_gradient_spans_solve_blocks` twins).
#[test]
fn ml_bfgs_route_matches_or_beats_the_bobyqa_route() {
    let nc = 30usize;
    for q in [2usize, 3, 4] {
        let (x, y, ids, _) = pg_dataset(q, nc, 10, 53 + q as u64);
        let model = SizedSpec {
            spec: ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    terms: Vec::new(),
                    slopes: (0..q as u32 - 1).collect(),
                    extra_groupings: vec![],
                }),
            },
            n_primary: nc,
            extra_counts: vec![],
        };
        let slope_cols: Vec<usize> = (1..q).collect();
        let search = |armed: bool| -> f64 {
            let mut ws = LmmWorkspace::for_cluster_spec(x.ncols(), &model, &slope_cols);
            ws.suff_mut()
                .add_rows_multi(x.as_ref(), x.as_ref().subcols(0, 0), &y, &ids, &[], None);
            let LmmWorkspace {
                solver,
                solver_warm,
                config,
                bfgs,
                kernel,
                theta,
                lower,
                upper,
                ..
            } = &mut ws;
            let LmmKernel::Dense { suff, fit } = &mut *kernel else {
                unreachable!("for_cluster_spec builds the dense kernel")
            };
            fit.criterion = Criterion::Ml;
            if armed {
                assert!(precompute_group_regressions(suff, fit), "q={q}");
            }
            let mut counters = crate::counters::EvalCounters::new();
            let (out, _, _, report) = search_theta(
                kernel,
                solver,
                solver_warm,
                config,
                bfgs,
                theta,
                lower,
                upper,
                false,
                None,
                &mut counters,
                &mut 0,
            );
            assert_eq!(report.is_some(), armed, "q={q}: the route");
            assert!(
                matches!(out.status, Status::Converged),
                "q={q}: {:?}",
                out.status
            );
            out.f
        };
        let (f_bfgs, f_bobyqa) = (search(true), search(false));
        assert!(
            f_bfgs <= f_bobyqa + 1e-6,
            "q={q}: BFGS route {f_bfgs} vs BOBYQA {f_bobyqa}"
        );
    }
}
