//! Unit tests for `lmm::kernel` and `lmm` (dispatch, workspace, BOBYQA config, joint Wald).

use super::*;
use crate::test_support::{extra_level_of_row, intercept_only_spec, model_atom};
use crate::{Family, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing};

/// Grammar: `<mult>n<add>`, e.g. `2n1` = 2·n+1, `1.5n1` = ⌈1.5·n⌉+1, `1n2` =
/// n+2. Result clamped to BOBYQA's legal `[n+2, (n+1)(n+2)/2]`: flat constants
/// (and small-n underflow) violate the bounds, so the hook clamps rather than
/// panic deep in `Bobyqa::new`.
#[test]
fn npt_formula_parses_and_clamps() {
    assert_eq!(npt_from_formula("2n1", 36), Some(73));
    assert_eq!(npt_from_formula("1.5n1", 36), Some(55)); // ⌈54⌉+1
    assert_eq!(npt_from_formula("1n2", 36), Some(38));
    assert_eq!(npt_from_formula("1.5n1", 2), Some(4)); // ⌈3⌉+1 = 4 = n+2 ✓
    assert_eq!(npt_from_formula("3n0", 2), Some(6)); // 6 = (n+1)(n+2)/2 cap
    assert_eq!(npt_from_formula("1n0", 3), Some(5)); // clamped up to n+2
    assert_eq!(npt_from_formula("500n500", 8), Some(45)); // max_fun grammar reuses
                                                          // the parser; the CLAMP is
                                                          // npt-specific — see Step 3
    assert_eq!(npt_from_formula("garbage", 8), None);
    assert_eq!(npt_from_formula("73", 8), None); // flat constants rejected
}

#[test]
fn formula_eval_unclamped() {
    assert_eq!(eval_formula("500n500", 8), Some(4500));
    assert_eq!(eval_formula("2n1", 36), Some(73));
    assert_eq!(eval_formula("n2", 8), None); // mult is mandatory: write 1n2
}

/// Deterministic pseudo-data (NR LCG), uniform in (−1, 1). NR = Press,
/// Teukolsky, Vetterling & Flannery (2007), *Numerical Recipes: The Art of
/// Scientific Computing*, 3rd ed., Cambridge University Press.
fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
}

/// n=48, p=3 (intercept + x1 + x2), 6 clusters,
/// y = 0.5 + 0.4·x1 − 0.2·x2 + u_c + 0.8·e.
fn hand_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
    let n = 48usize;
    let n_clusters = 6usize;
    let mut st = 42u64;
    let u_c: Vec<f64> = (0..n_clusters).map(|_| 0.6 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        let c = i % n_clusters;
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

/// Same quantity, two factorizations — both return
/// log|V| + log|X'V⁻¹X| + (N−P)·log σ̂², so agreement is FP-level
/// (≤ 1e-9 rel), not up-to-a-constant. THE formulation proof, held on
/// every θ probed.
///
/// `PINNED_DEV` is `profiled_deviance(theta, &mut scratch)` on
/// `hand_dataset` from the scalar Brent kernel retired in this commit
/// (`src/lme.rs`, deleted 2026-08-26, commit 95dfe90 with uncommitted
/// working-tree edits), `scratch` built via `test_support::build_lme_scratch`
/// at that same commit — pinned so the cross-check between the collapse
/// and general REML paths survives the kernel that produced it.
#[test]
fn deviance_matches_pinned_across_theta() {
    const PINNED_DEV: [(f64, f64); 9] = [
        (0.0, -5.133651734340493e1),
        (1e-4, -5.133651830952116e1),
        (1e-2, -5.134617257834838e1),
        (0.1, -5.22466385997638e1),
        (0.5, -6.044958474290513e1),
        (1.0, -6.2089881344871685e1),
        (2.0, -5.8665868559293685e1),
        (10.0, -4.390226095246822e1),
        (100.0, -2.0933925045275885e1),
    ];

    let (x, y, ids) = hand_dataset();
    let mut suff = LmmSuffStats::new(3, 6);
    suff.add_rows(x.as_ref(), &y, &ids);
    let mut fit = LmmFitScratch::new(3, 6);
    let mut fit_c = LmmFitScratch::new(3, 6);
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));

    for &(theta, dev_ship) in &PINNED_DEV {
        let dev_gen = reml_deviance(&[theta], &suff, &mut fit);
        assert!(dev_gen.is_finite(), "θ={theta}");
        let tol = 1e-9 * dev_ship.abs().max(1.0);
        assert!(
            (dev_ship - dev_gen).abs() <= tol,
            "θ={theta}: pinned {dev_ship} vs general {dev_gen}"
        );
        // Collapse arm — reassociation band vs the general loop incl. θ=0.
        let dev_c = reml_deviance(&[theta], &suff, &mut fit_c);
        let band = 1e-9 * dev_gen.abs().max(1.0);
        assert!(
            (dev_c - dev_gen).abs() <= band,
            "θ={theta}: collapse {dev_c} vs general {dev_gen}"
        );
    }
}

/// All scratch is overwritten per call — re-evaluating a θ after an
/// intervening different-θ call reproduces bit-identical deviance and σ̂²
/// (mirrors the retired `lme.rs`'s stale-state test).
#[test]
fn reml_deviance_overwrites_state() {
    let (x, y, ids) = hand_dataset();
    let mut suff = LmmSuffStats::new(3, 6);
    suff.add_rows(x.as_ref(), &y, &ids);
    let mut fit = LmmFitScratch::new(3, 6);

    let dev_a = reml_deviance(&[1.0], &suff, &mut fit);
    let sig_a = fit.sigma_sq;
    let _ = reml_deviance(&[2.0], &suff, &mut fit);
    let dev_b = reml_deviance(&[1.0], &suff, &mut fit);
    let sig_b = fit.sigma_sq;
    assert_eq!(dev_a, dev_b, "deviance(θ=1) must be reproducible");
    assert_eq!(sig_a, sig_b, "σ̂²(θ=1) must be reproducible");
}

/// Correctness prerequisite for workspace reuse across simulation draws:
/// `suff.reset()` followed by a refill on a DIFFERENT dataset (same shape, different `y`)
/// must reproduce a freshly-constructed workspace's fit bit-for-bit. Same
/// buffers + same code path ⇒ identical float reassociation, so the
/// assertion is exact `==`, not a tolerance band. If this fails, `reset()`
/// (src/lmm/kernel.rs) leaves some `LmmSuffStats` field stale across datasets.
#[test]
fn reused_workspace_refill_matches_fresh() {
    let (x, y_a, ids) = hand_dataset();
    // B: same shape/ids as A, deterministically different y (constant shift
    // + a fixed rescale) — not randomized, so the comparison stays exact.
    let y_b: Vec<f64> = y_a.iter().map(|&v| 1.7 - 0.3 * v).collect();
    let targets: Vec<u32> = vec![1, 2];

    // Fresh workspace, fit B directly.
    let mut ws_fresh = LmmWorkspace::new(3, 6);
    ws_fresh.suff.reset();
    ws_fresh.suff.add_rows(x.as_ref(), &y_b, &ids);
    let fit_fresh = fit_lmm(&mut ws_fresh, &targets, None);

    // Reused workspace: fit A first, reset, refill with B, fit again.
    let mut ws_reused = LmmWorkspace::new(3, 6);
    ws_reused.suff.reset();
    ws_reused.suff.add_rows(x.as_ref(), &y_a, &ids);
    let _ = fit_lmm(&mut ws_reused, &targets, None);
    ws_reused.suff.reset();
    ws_reused.suff.add_rows(x.as_ref(), &y_b, &ids);
    let fit_reused = fit_lmm(&mut ws_reused, &targets, None);

    assert_eq!(
        fit_fresh.deviance, fit_reused.deviance,
        "deviance must be bit-identical after reset+refill on new data"
    );
    assert_eq!(
        ws_fresh.fit.betas, ws_reused.fit.betas,
        "betas must be bit-identical after reset+refill on new data"
    );
    assert_eq!(
        ws_fresh.fit.var_diag, ws_reused.fit.var_diag,
        "var_diag must be bit-identical after reset+refill on new data"
    );
}

/// Exercises the plateau policy: a `MaxFunReached` cap-out must still
/// report the honest finite endpoint
/// (β̂/σ̂²/SE/deviance), with `converged = false` and `boundary_hit == 2`
/// (not the accepted-boundary 1). Forces the cap by swapping in a solver
/// whose `max_fun` is the legal minimum (`npt + 1`) — one eval past the
/// initial model build, nowhere near this dataset's optimum — bypassing
/// `LMM_MAX_FUN_FORMULA` entirely so the test carries no process-env race.
#[test]
fn maxfun_cap_reports_honest_endpoint() {
    let (x, y, ids) = hand_dataset();
    let targets: Vec<u32> = vec![1, 2];

    let mut ws = LmmWorkspace::new(3, 6);
    ws.suff.add_rows(x.as_ref(), &y, &ids);
    let n_theta = ws.theta.len();
    let npt = 2 * n_theta + 1; // n_theta == 1 here: PRIMA's minimum npt
    let config = {
        let mut c = Config::new(n_theta);
        c.npt = npt;
        c.max_fun = npt + 1;
        c
    };
    ws.solver = Bobyqa::new(n_theta, config).expect("legal minimal config");

    let fit = fit_lmm(&mut ws, &targets, None);

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
            ws.fit.betas[tj as usize].is_finite(),
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
            rel(ws.fit.betas[j], wb) < 1e-6,
            "betas[{j}] = {}, want {}",
            ws.fit.betas[j],
            wb
        );
    }
}

/// End-to-end q=1 parity on the hand dataset: the general machine against
/// frozen literal endpoints, at the amended tolerances (rel 1e-4, abs floors
/// β̂ 1e-5 / stat 1e-4 — the measured Brent θ̂-placement-noise floor).
///
/// `PINNED_BETAS`/`PINNED_STATS`/`PINNED_JOINT_T_SQ` are the retired scalar
/// `lme_fit`'s output on `hand_dataset` (targets `[1, 2]`) from the Brent
/// kernel retired in this commit (`src/lme.rs`, deleted 2026-08-26,
/// commit 95dfe90 with uncommitted working-tree edits).
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
    ws.suff.add_rows(x.as_ref(), &y, &ids);
    let fit = fit_lmm(&mut ws, &targets, None);
    assert!(fit.converged);
    assert!(fit.boundary_hit <= 1);

    for (j, &want) in PINNED_BETAS.iter().enumerate() {
        let (a, b) = (want, ws.fit.betas[j]);
        let d = (a - b).abs();
        assert!(
            d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
            "β[{j}]: {a} vs {b}"
        );
    }
    for (idx, &tj) in targets.iter().enumerate() {
        let a = PINNED_STATS[idx];
        let b = ws.fit.t_sq[tj as usize].sqrt();
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

/// Deterministic pin: y carries NO between-cluster signal by construction —
/// residuals alternate ±0.8 within each cluster with equal counts, so every
/// cluster's residual sum is exactly 0 and the REML deviance is minimized at
/// θ = 0. The fit must pin (boundary_hit == 1), write θ̂ = exactly 0.0, and
/// count as converged: zero variance is a legitimate boundary optimum, not
/// a failure to fit.
#[test]
fn zero_between_cluster_variance_pins_at_exactly_zero() {
    let n = 48usize;
    let n_clusters = 6usize;
    let mut st = 7u64;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    for i in 0..n {
        ids[i] = (i % n_clusters) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        // i/n_clusters cycles 0..8 within each cluster: 4 even, 4 odd ⇒
        // the ±0.8 residuals cancel exactly per cluster.
        let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
        y[i] = 0.5 + 0.4 * x1 + e;
    }
    let mut ws = LmmWorkspace::new(2, n_clusters);
    ws.suff.add_rows(x.as_ref(), &y, &ids);
    let fit = fit_lmm(&mut ws, &[1], None);
    assert!(fit.converged);
    assert_eq!(fit.boundary_hit, 1);
    assert_eq!(ws.theta[0], 0.0, "pin must be exact 0.0, not merely small");
    assert!(ws.fit.betas[1].is_finite());
}

/// Rank deficiency is DETECTED, not refused, at kernel level: x2 = 0.1·x1
/// (the scaled-duplicate fixture — exact duplicates can slip through faer's
/// llt grey zone) returns a fit whose `pivot` records the exhaustion and
/// names the offending column.
///
/// This kernel entry point is below the alias gate, which is what actually
/// handles a design like this: through `fit_cold`/`fit_warm` the duplicate
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
    ws.suff.add_rows(x.as_ref(), &y, &ids);
    let fit = fit_lmm(&mut ws, &[1, 2], None);
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
        ws.fit.var_diag[2].sqrt() > 1e6 * ws.fit.betas[2].abs(),
        "β̂₂ = {} must carry an SE orders above it, got {}",
        ws.fit.betas[2],
        ws.fit.var_diag[2].sqrt()
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
    ws_blind.suff.add_rows(x.as_ref(), &y, &ids);
    let blind = fit_lmm(&mut ws_blind, &targets, None);
    assert!(blind.converged);

    for start in [[0.0], [0.6]] {
        let mut ws = LmmWorkspace::new(3, 6);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm(&mut ws, &targets, Some(&start));
        assert!(fit.converged, "start {start:?}");
        for j in 0..3 {
            let (a, b) = (ws_blind.fit.betas[j], ws.fit.betas[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
                "start {start:?} β[{j}]: blind {a} vs started {b}"
            );
        }
    }
}

/// Bounded-allocation warm-path check for `fit_lmm`.
/// Marked #[ignore] because dhat measures
/// process-wide allocations; `alloc_test_guard` serializes it against the
/// other `#[ignore]` tests:
///   cargo test -p glmm --features alloc-tests lmm_fit_warm_path_bounded_alloc -- --ignored
///
/// BOUND locks the measured warm-path block count. LmmWorkspace itself is
/// allocation-free across fits (Bobyqa::new is the only solver allocation,
/// done once). On the faer kernel the per-call blocks are `llt` internals —
/// ~2 per deviance evaluation (15.1–15.7 evals/fit at rho_end 1e-6, the
/// measured mean), the same acceptance the shipped path's 26
/// blocks/call carry; if a future faer version changes its Cholesky
/// internals, update the bound — do not relax it. A hand-rolled
/// owned-kernel replacement for faer's `llt` was tried and rejected: its
/// wasm `f64::ln` took a different ULP path than the native build (the
/// factorization itself was fine), which broke cross-platform
/// bit-equality. The faer bound stays the locked steady state.
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn lmm_fit_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: usize = 100;
    const BOUND: u64 = 4800; // Measured 4600 (this machine) — ~46 blocks/fit of faer `llt` internals on the family-blocked q=1 path (one m×m tail llt per eval). `fit_lmm` no longer allocates per fit (the diagonal_theta index map is cached once on LmmGroupings; the ranef recovery pass solves in the ranef_ux/ranef_rhs scratch fields), so this count is purely faer's Cholesky internals — faer-version/machine specific. q=1 deviance is byte-identical to the hand-rolled augmented-factor deviance (held by the lmm_parity corpus + golden_rng), so the eval trajectory is unchanged; the count differs from the prior 3804 only because faer's blocked llt allocates more per eval than the hand-rolled augmented factor. If faer changes its Cholesky internals, update — do not relax.

    let (x, y, ids) = hand_dataset();
    let targets: Vec<u32> = vec![1, 2];
    let mut ws = LmmWorkspace::new(3, 6);

    // Warmup drives one-time setup outside the profiler window.
    ws.suff.reset();
    ws.suff.add_rows(x.as_ref(), &y, &ids);
    let _ = fit_lmm(&mut ws, &targets, None);

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        ws.suff.reset();
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let _ = fit_lmm(&mut ws, &targets, None);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert!(
        stats.total_blocks <= BOUND,
        "fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
        stats.total_blocks,
        N_CALLS,
        BOUND
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
) -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>, ModelSpec) {
    let mut cluster = intercept_only_spec(Sizing::FixedClusters { n_clusters: 6 });
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        relation: GroupingRelation::Crossed { n_clusters: 4 },
        slopes: vec![],
    });
    if with_nested {
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            slopes: vec![],
        });
    }
    let n = n_blocks * model_atom(&cluster);
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
        pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
        #[allow(clippy::needless_range_loop)]
        for g in 0..n_extras {
            eids[g][i] = extra_level_of_row(&cluster, g, i) as u32;
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
    (x, y, pid, eids, cluster)
}

/// `diagonal_theta` / `n_theta` / `k_family` at q_p ∈ {1, 2, 3} — locks
/// the column-major vech ordering. q_p=1 must reproduce the intercept-only
/// baseline values; q_p>1 tests the standalone slope branch (no extras, k_crossed=0).
#[test]
fn groupings_vech_layout() {
    let sizing = Sizing::FixedClusters { n_clusters: 4 };
    let base = intercept_only_spec(sizing.clone());

    // q_p = 1 (intercept-only): shape must be unchanged.
    let g1 = LmmGroupings::from_cluster_spec(&base, 40, &[]);
    assert_eq!(g1.n_theta(), 1);
    assert_eq!(g1.k_family(), 4); // 4 clusters × 1
    assert_eq!(g1.diagonal_theta(), &[0][..]);

    // q_p = 2 (1 slope): vech([σ_00, σ_10, σ_11]) length 3; diagonals at 0, 2.
    let mut spec2 = base.clone();
    spec2.re.as_mut().unwrap().slopes.push(1);
    let g2 = LmmGroupings::from_cluster_spec(&spec2, 40, &[1]);
    assert_eq!(g2.primary_q, 2);
    assert_eq!(g2.n_theta(), 3); // 2·3/2 = 3
    assert_eq!(g2.k_family(), 8); // 4 clusters × 2
    assert_eq!(g2.k_total, 8);
    assert_eq!(g2.diagonal_theta(), &[0, 2][..]); // off-diagonal vech[1]=1 excluded

    // q_p = 3 (2 slopes): vech([σ_00, σ_10, σ_11, σ_20, σ_21, σ_22]) length 6; diagonals at 0, 3, 5.
    let mut spec3 = base.clone();
    spec3.re.as_mut().unwrap().slopes.push(1);
    spec3.re.as_mut().unwrap().slopes.push(2);
    let g3 = LmmGroupings::from_cluster_spec(&spec3, 40, &[1, 2]);
    assert_eq!(g3.primary_q, 3);
    assert_eq!(g3.n_theta(), 6); // 3·4/2 = 6
    assert_eq!(g3.k_family(), 12); // 4 clusters × 3
    assert_eq!(g3.k_total, 12);
    assert_eq!(g3.diagonal_theta(), &[0, 3, 5][..]);
}

/// Suff-stats bookkeeping on a hand-checkable block: counts per RE column,
/// per-column sums, crossed cross-counts.
#[test]
fn suff_stats_multi_accumulators() {
    let (x, y, pid, eids, cluster) = multi_dataset(true, 1); // one atom block, n=48
    let g = LmmGroupings::from_cluster_spec(&cluster, 48, &[]);
    assert_eq!(g.n_primary, 6);
    assert_eq!(g.nested_per_parent, 2);
    assert_eq!(g.k_family(), 18); // 6 + 6·2
    assert_eq!(g.k_total, 22); // + 4 crossed
    assert_eq!(g.n_theta(), 3);
    let mut suff = LmmSuffStats::with_groupings(3, g);
    suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
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
    // Intercept column sum = row count per level.
    assert!((suff.s[(0, 0)] - 8.0).abs() < 1e-12);
}

/// Textbook REML deviance on the explicit n×n V — the oracle for the
/// family-blocked elimination. dev = ln|V| + ln|X'V⁻¹X| + (N−P)·ln σ̂²,
/// σ̂² = (y'V⁻¹y − β̂'X'V⁻¹y)/(N−P).  `groups[g]` = grouping g's global
/// level ids (primary first); `theta[g]` the matching component.
fn brute_force_deviance(theta: &[f64], x: &Mat<f64>, y: &[f64], groups: &[&[u32]]) -> f64 {
    use faer::linalg::solvers::Solve;
    let n = x.nrows();
    let p = x.ncols();
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
    let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
    let mut log_det_v = 0.0;
    for i in 0..n {
        log_det_v += vc.L()[(i, i)].ln();
    }
    let log_det_v = 2.0 * log_det_v;
    let mut vix = (*x).clone();
    vc.solve_in_place(vix.as_mut());
    let mut viy = Mat::<f64>::zeros(n, 1);
    for i in 0..n {
        viy[(i, 0)] = y[i];
    }
    vc.solve_in_place(viy.as_mut());
    let mut xtvix = Mat::<f64>::zeros(p, p);
    let mut xtviy = vec![0.0; p];
    for a in 0..p {
        for b in 0..p {
            let mut acc = 0.0;
            for i in 0..n {
                acc += x[(i, a)] * vix[(i, b)];
            }
            xtvix[(a, b)] = acc;
        }
        let mut acc = 0.0;
        for i in 0..n {
            acc += x[(i, a)] * viy[(i, 0)];
        }
        xtviy[a] = acc;
    }
    let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
    let mut log_det_k = 0.0;
    for a in 0..p {
        log_det_k += kc.L()[(a, a)].ln();
    }
    let log_det_k = 2.0 * log_det_k;
    let mut beta = Mat::<f64>::zeros(p, 1);
    for a in 0..p {
        beta[(a, 0)] = xtviy[a];
    }
    kc.solve_in_place(beta.as_mut());
    let mut ytviy = 0.0;
    for i in 0..n {
        ytviy += y[i] * viy[(i, 0)];
    }
    let mut bxy = 0.0;
    for a in 0..p {
        bxy += beta[(a, 0)] * xtviy[a];
    }
    let df = (n - p) as f64;
    let sigma_sq = (ytviy - bxy) / df;
    log_det_v + log_det_k + df * sigma_sq.ln()
}

fn assert_deviance_matches_oracle(with_nested: bool, thetas: &[Vec<f64>]) {
    let (x, y, pid, eids, cluster) = multi_dataset(with_nested, 2);
    let n = x.nrows();
    let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
    let mut suff = LmmSuffStats::with_groupings(3, g);
    suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
    let mut fit = LmmFitScratch::with_groupings(3, &gref);
    let mut fit_c = LmmFitScratch::with_groupings(3, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));
    // Oracle wants global ids per grouping.
    let mut groups: Vec<&[u32]> = vec![&pid];
    for e in &eids {
        groups.push(e);
    }
    for th in thetas {
        let dev = reml_deviance(th, &suff, &mut fit);
        let oracle = brute_force_deviance(th, &x, &y, &groups);
        assert!(dev.is_finite(), "θ={th:?}");
        let tol = 1e-8 * oracle.abs().max(1.0);
        assert!(
            (dev - oracle).abs() <= tol,
            "θ={th:?}: family-blocked {dev} vs oracle {oracle}"
        );
        // Collapse arm: same θ through the balanced path — reassociation
        // band vs the loop, oracle band absolute.
        let dev_c = reml_deviance(th, &suff, &mut fit_c);
        let band = 1e-9 * dev.abs().max(1.0);
        assert!(
            (dev_c - dev).abs() <= band,
            "θ={th:?}: collapse {dev_c} vs loop {dev}"
        );
        assert!(
            (dev_c - oracle).abs() <= tol,
            "θ={th:?}: collapse vs oracle"
        );
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

/// Unbalanced counts must take the legacy loop byte-for-byte: a failed
/// precompute leaves collapse_n_active = 0 and the eval path untouched.
#[test]
fn unbalanced_counts_fall_back_byte_identical() {
    let (x, y, pid, eids, cluster) = multi_dataset(true, 2);
    let n = x.nrows() - 1; // truncate one row — last cluster short
    let g = LmmGroupings::from_cluster_spec(&cluster, x.nrows(), &[]);
    let mut suff = LmmSuffStats::with_groupings(3, g);
    let eids_t: Vec<Vec<u32>> = eids.iter().map(|e| e[..n].to_vec()).collect();
    suff.add_rows_multi(x.as_ref().subrows(0, n), &y[..n], &pid[..n], &eids_t, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, x.nrows(), &[]);
    let mut fit_a = LmmFitScratch::with_groupings(3, &gref);
    let mut fit_b = LmmFitScratch::with_groupings(3, &gref);
    assert!(!precompute_balanced_collapse(&suff, &mut fit_b));
    for th in [[0.5, 0.3, 0.2], [1.0, 1.0, 1.0], [0.0, 0.5, 0.9]] {
        let a = reml_deviance(&th, &suff, &mut fit_a);
        let b = reml_deviance(&th, &suff, &mut fit_b);
        assert_eq!(a.to_bits(), b.to_bits(), "θ={th:?}");
    }
}

/// Off-grid N under `FixedSize`: row 17 of 18 sits in cluster 4, so five
/// primary levels exist and the nested block must cover all five parents.
#[test]
fn fixed_size_off_grid_n_keeps_the_partial_trailing_cluster() {
    let mut cluster = intercept_only_spec(Sizing::FixedSize { cluster_size: 4 });
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
        slopes: vec![],
    });
    let n = 18;
    let sizing = &cluster.re.as_ref().unwrap().sizing;
    assert_eq!(sizing.cluster_of_row(n - 1), 4);
    assert_eq!(sizing.n_clusters_at(n), 5);
    let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
    assert_eq!(g.n_primary, 5);
    assert_eq!(g.n_primary * g.nested_per_parent, 10);
    assert_eq!(g.k_total, 15);
}

/// Nested-only in Regime B — the path with NO crossed tail (zx is 0×0)
/// and parents that grow with N.
#[test]
fn nested_regime_b_deviance_matches_brute_force() {
    let mut cluster = intercept_only_spec(Sizing::FixedSize { cluster_size: 8 });
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
        slopes: vec![],
    });
    let n = 4 * model_atom(&cluster); // 64
    let mut st = 7u64;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut cid = vec![0u32; n];
    let u_p: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
    let u_c: Vec<f64> = (0..16).map(|_| 0.3 * lcg(&mut st)).collect();
    for i in 0..n {
        pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
        cid[i] = extra_level_of_row(&cluster, 0, i) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + u_p[pid[i] as usize] + u_c[cid[i] as usize] + 0.8 * lcg(&mut st);
    }
    let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    let eids = vec![cid.clone()];
    suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));
    for th in [[0.6, 0.4], [1.0, 1.0], [0.2, 0.0], [0.0, 0.0]] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &cid]);
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
/// slope groupings and unbalanced counts fall back.
#[test]
fn balanced_collapse_applicability() {
    // Balanced: the regime-B nested dataset (atom-multiple by construction).
    let mut cluster = intercept_only_spec(Sizing::FixedSize { cluster_size: 8 });
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
        slopes: vec![],
    });
    let n = 4 * model_atom(&cluster); // 64
    let max_n = 2 * n; // workspace sized for a larger grid top — active PREFIX
    let mut st = 7u64;
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut cid = vec![0u32; n];
    for i in 0..n {
        pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
        cid[i] = extra_level_of_row(&cluster, 0, i) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + 0.8 * lcg(&mut st);
    }
    let g = LmmGroupings::from_cluster_spec(&cluster, max_n, &[]);
    let n_primary = g.n_primary;
    let mut suff = LmmSuffStats::with_groupings(2, g);
    let eids = vec![cid.clone()];
    suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, max_n, &[]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit));
    assert_eq!(fit.collapse_n_active, n / 8);
    assert!(fit.collapse_n_active < n_primary); // genuinely a prefix

    // Unbalanced: drop the last row — the trailing cluster is short.
    let mut suff_u =
        LmmSuffStats::with_groupings(2, LmmGroupings::from_cluster_spec(&cluster, max_n, &[]));
    let eids_u = vec![cid[..n - 1].to_vec()];
    suff_u.add_rows_multi(
        x.as_ref().subrows(0, n - 1),
        &y[..n - 1],
        &pid[..n - 1],
        &eids_u,
        None,
    );
    assert!(!precompute_balanced_collapse(&suff_u, &mut fit));
    assert_eq!(fit.collapse_n_active, 0);

    // Slope path: never applicable — populated, balanced data, so the
    // rejection is the q_p guard, not the empty-suff early-out (balanced
    // slope counts would otherwise pass the count checks).
    let (xs, ys, ids_s) = slope_dataset();
    let gs = slope_groupings();
    let mut suff_s = LmmSuffStats::with_groupings(2, slope_groupings());
    suff_s.add_rows_multi(xs.as_ref(), &ys, &ids_s, &[], None);
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
    ws_w.suff
        .add_rows_multi(x.as_ref(), &y, &ids, &[], Some(&w));
    assert!(
        precompute_balanced_collapse(&ws_w.suff, &mut ws_w.fit),
        "constant weights keep exact per-cluster counts equality"
    );
    assert_eq!(ws_w.fit.collapse_n_active, 6);
    let fit_w = fit_lmm(&mut ws_w, &targets, None);
    assert!(fit_w.converged);

    let mut ws_u = LmmWorkspace::new(3, 6);
    ws_u.suff.add_rows(x.as_ref(), &y, &ids);
    let fit_u = fit_lmm(&mut ws_u, &targets, None);
    assert!(fit_u.converged);

    // Two independent BOBYQA runs agree to the rho_end floor, not machine
    // precision — same 1e-6 relative band as the fit.rs invariance tests.
    for j in 0..3 {
        let (a, b) = (ws_u.fit.betas[j], ws_w.fit.betas[j]);
        assert!(
            (a - b).abs() / a.abs() < 1e-6,
            "β[{j}] unweighted {a} vs w≡2 {b}"
        );
    }
    for &tj in &targets {
        let (a, b) = (
            ws_u.fit.var_diag[tj as usize].sqrt(),
            ws_w.fit.var_diag[tj as usize].sqrt(),
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
    let mut cluster = intercept_only_spec(Sizing::FixedClusters { n_clusters: 3 });
    for k in [4u32, 2u32] {
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            relation: GroupingRelation::Crossed { n_clusters: k },
            slopes: vec![],
        });
    }
    let n = 2 * model_atom(&cluster); // 48
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
        pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
        e0[i] = extra_level_of_row(&cluster, 0, i) as u32;
        e1[i] = extra_level_of_row(&cluster, 1, i) as u32;
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
    let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    let eids = vec![e0.clone(), e1.clone()];
    suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
    let mut fit = LmmFitScratch::with_groupings(2, &gref);
    let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
    assert!(precompute_balanced_collapse(&suff, &mut fit_c));
    for th in [[0.5, 0.4, 0.3], [1.0, 1.0, 1.0], [0.3, 0.0, 0.8]] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &e0, &e1]);
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
    let mut cluster = intercept_only_spec(Sizing::FixedClusters {
        n_clusters: s_cl as u32,
    });
    cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
        relation: GroupingRelation::Crossed {
            n_clusters: i_cl as u32,
        },
        slopes: vec![],
    });
    let n = 4 * model_atom(&cluster); // 48: 4 blocks ⇒ ±0.8 cancels per item
    let mut st = 5u64;
    let u_p: Vec<f64> = (0..s_cl).map(|_| 0.8 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut eid = vec![0u32; n];
    for i in 0..n {
        pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
        eid[i] = extra_level_of_row(&cluster, 0, i) as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        let e = if (i / model_atom(&cluster)) % 2 == 0 {
            0.8
        } else {
            -0.8
        };
        y[i] = 0.5 + 0.4 * x1 + u_p[pid[i] as usize] + e;
    }
    let mut ws = LmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
    let eids = vec![eid];
    ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
    let fit = fit_lmm(&mut ws, &[1], None);
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
/// sanity bands and produces finite Wald machinery — the L1 smoke for the
/// full multi-grouping pipeline (the statistical gates live in L3).
#[test]
fn crossed_nested_fit_recovers_betas() {
    let (x, y, pid, eids, cluster) = multi_dataset(true, 4); // n = 192
    let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);
    ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
    let fit = fit_lmm(&mut ws, &[1, 2], None);
    assert!(fit.converged);
    assert!((ws.fit.betas[1] - 0.4).abs() < 0.15);
    assert!((ws.fit.betas[2] + 0.2).abs() < 0.15);
    // Deterministic regression lock (lcg-seeded multi_dataset) alongside the
    // planted-value recovers-check above, which documents intent.
    assert!((ws.fit.betas[1] - 0.40829926961384383).abs() / 0.40829926961384383_f64.abs() < 1e-6);
    assert!((ws.fit.betas[2] - -0.2916210839321183).abs() / 0.2916210839321183_f64.abs() < 1e-6);
    assert!(ws.fit.t_sq[1].is_finite() && ws.fit.t_sq[2].is_finite());
    assert!(fit.joint_t_sq.is_finite() && fit.joint_t_sq > 0.0);
    assert_eq!(ws.theta.len(), 3);
}

/// General-path twin of lmm_fit_warm_path_bounded_alloc: crossed+nested
/// workspace. Per-call blocks are the tail-llt faer internals (the family
/// loop is hand-rolled, zero-alloc) — the same acceptance class as q=1.
/// Warm-started from a cold prime fit's fitted θ (the loop tier's production
/// pattern), matching the few-eval regime the production path runs.
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn lmm_fit_general_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: usize = 100;
    const BOUND_GENERAL: u64 = 8400; // Measured 8000 (this machine) — ~80 blocks/fit truth-started (scaled rho + spec-derived start; the few-eval regime the production path runs). Per-eval faer `llt` internals only: the family loop is hand-rolled zero-alloc, the cached diagonal_theta map removed the per-fit Vec, and the ranef recovery pass solves in the ranef_ux/ranef_rhs scratch fields, so this count is faer-version/machine specific. If faer changes its Cholesky internals, update — do not relax.

    let (x, y, pid, eids, cluster) = multi_dataset(true, 2);
    let targets: Vec<u32> = vec![1, 2];
    let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);

    ws.suff.reset();
    ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
    // prime cold, then warm-start subsequent refits from the previous fit's fitted θ
    // (the loop tier's production pattern; replaces the deleted spec truth-start).
    let _ = fit_lmm(&mut ws, &targets, None);
    let warm = ws.theta.clone();

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        ws.suff.reset();
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        let _ = fit_lmm(&mut ws, &targets, Some(&warm));
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

/// Crossed-slopes twin of the bounded-alloc gate: the blocked path's only
/// per-eval heap traffic is the faer `llt` internals (everything else lives in
/// `LmmFitScratch.blocked_*`, sized once). Same acceptance class as the other
/// general fits; faer-version/machine specific — if faer changes its Cholesky
/// internals, update the bound, do not relax.
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn lmm_fit_crossed_slope_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: usize = 100;
    // Measured ~46100 (this machine, faer 0.x): ~460 blocks/fit = the dim≈31
    // tail `llt` internals × the ~50–90 BOBYQA evals of a 6-θ fit. ALL faer-
    // internal — `reml_deviance_blocked` itself is zero-alloc (every buffer is
    // in `blocked_*` scratch; only a stack `lam_g`). faer-version/machine
    // specific; if faer changes its Cholesky internals, update — do not relax.
    const BOUND: u64 = 55000;

    let (x, y, pid, eid) = crossed_slope_golden_dataset();
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 6 },
                slopes: vec![1],
            }],
        }),
    };
    let mut ws = LmmWorkspace::for_cluster_spec_ext(2, &cluster, x.nrows(), &[1], &[vec![1]]);
    ws.suff.reset();
    ws.suff
        .add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&eid), None);
    // prime cold, then warm-start subsequent refits from the previous fit's fitted θ
    // (the loop tier's production pattern; replaces the deleted spec truth-start).
    let _ = fit_lmm(&mut ws, &[1], None);
    let warm = ws.theta.clone();

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        ws.suff.reset();
        ws.suff
            .add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&eid), None);
        let _ = fit_lmm(&mut ws, &[1], Some(&warm));
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert!(
        stats.total_blocks <= BOUND,
        "crossed-slope fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
        stats.total_blocks,
        N_CALLS,
        BOUND
    );
}

// -----------------------------------------------------------------------
// Standalone primary slopes: q_p×q_p primary block, oracle deviance,
// diagonal-only pin. Data lives on the engine's f32 plane (mirrors the scalar
// oracle convention); the brute force widens the identical bytes to f64, so
// the 1e-8 match is exact, not modulo an f32↔f64 roundtrip.
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

/// Textbook REML deviance with a q×q D over the slope columns of Z_p.
/// `theta` is the column-major vech of Λ (q×q lower-tri); D_rel = ΛΛ′
/// (σ-relative); V = I + Z·D_rel·Z′ with Z_i = [1, x[i, slope_cols]]. The
/// f32 data is widened to f64 so the oracle reads the same bytes the suff
/// stats accumulated.
fn brute_force_slope_deviance(
    theta: &[f64],
    x: &Mat<f64>,
    y: &[f64],
    ids: &[u32],
    slope_cols: &[usize],
    q: usize,
) -> f64 {
    use faer::linalg::solvers::Solve;
    let (n, p) = (x.nrows(), x.ncols());
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
    // REML profile (unchanged from the scalar oracle): ldv + ldk + df·ln s².
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
    let df = (n - p) as f64;
    let s2 = (ytviy - bxy) / df;
    ldv + ldk + df * s2.ln()
}

fn slope_groupings() -> LmmGroupings {
    // 8 primary clusters, one slope on x_full col 1; no extras.
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![0],
            extra_groupings: vec![],
        }),
    };
    LmmGroupings::from_cluster_spec(&cluster, 64, &[1])
}

fn multislope_groupings() -> LmmGroupings {
    // 8 primary clusters, two slopes on x_full cols 1,2; no extras.
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![0, 1],
            extra_groupings: vec![],
        }),
    };
    LmmGroupings::from_cluster_spec(&cluster, 96, &[1, 2])
}

#[test]
fn slope_deviance_matches_brute_force() {
    let (x, y, ids) = slope_dataset();
    let mut suff = LmmSuffStats::with_groupings(2, slope_groupings());
    suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let mut fit = LmmFitScratch::with_groupings(2, &slope_groupings());
    // θ = vech(Λ), q=2: [λ₀₀, λ₁₀, λ₁₁].
    for th in [
        vec![1.0, 0.0, 1.0],
        vec![0.5, 0.2, 0.4],
        vec![2.0, -0.5, 0.7],
        vec![1e-3, 1e-3, 1e-3],
        // θ at THETA_HI (BOBYQA's box upper bound): the per-family Crout
        // pivot product must stay finite here — a product accumulated
        // across all families instead of reset per family would overflow.
        vec![THETA_HI, 0.0, THETA_HI],
    ] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_slope_deviance(&th, &x, &y, &ids, &[1], 2);
        assert!(dev.is_finite(), "θ={th:?}");
        assert!(
            (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
            "θ={th:?}: {dev} vs {oracle}"
        );
    }
}

#[test]
fn multislope_deviance_matches_brute_force() {
    let (x, y, ids) = multislope_dataset();
    let mut suff = LmmSuffStats::with_groupings(3, multislope_groupings());
    suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let mut fit = LmmFitScratch::with_groupings(3, &multislope_groupings());
    // θ = vech(Λ), q=3: [λ₀₀, λ₁₀, λ₂₀, λ₁₁, λ₂₁, λ₂₂].
    for th in [
        vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
        vec![0.6, 0.2, -0.1, 0.4, 0.15, 0.3],
        vec![1.5, -0.4, 0.3, 0.7, -0.2, 0.5],
    ] {
        let dev = reml_deviance(&th, &suff, &mut fit);
        let oracle = brute_force_slope_deviance(&th, &x, &y, &ids, &[1, 2], 3);
        assert!(dev.is_finite(), "θ={th:?}");
        assert!(
            (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
            "θ={th:?}: {dev} vs {oracle}"
        );
    }
}

/// End-to-end single-slope fit recovers the planted structure within BOBYQA
/// bands and pins nothing on a well-identified design.
#[test]
fn slope_fit_converges_interior() {
    let (x, y, ids) = slope_dataset();
    let mut ws = LmmWorkspace::with_groupings(2, slope_groupings());
    ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let fit = fit_lmm(&mut ws, &[1], None);
    assert!(fit.converged);
    // Planted [intercept 0.5, slope 0.4]; small (n=64, 8 clusters) REML draw
    // recovers ≈[0.46, 0.20] — directionally correct, finite-sample attenuated.
    // Pin sign + a band tight enough to catch a sign flip, a collapse to 0, or a
    // blow-up (mere `is_finite` passed any of those).
    assert!(
        (0.2..0.8).contains(&ws.fit.betas[0]),
        "intercept {}",
        ws.fit.betas[0]
    );
    assert!(
        (0.05..0.6).contains(&ws.fit.betas[1]),
        "slope {}",
        ws.fit.betas[1]
    );
    // Deterministic regression lock alongside the bands above.
    assert!((ws.fit.betas[0] - 0.46265883331118085).abs() / 0.46265883331118085_f64.abs() < 1e-6);
    assert!((ws.fit.betas[1] - 0.20152611939449563).abs() / 0.20152611939449563_f64.abs() < 1e-6);
    assert_eq!(fit.pinned_components & !0b11, 0); // only 2 components exist
}

/// End-to-end two-slope fit: 3 components (intercept + 2 slopes), interior.
#[test]
fn multislope_fit_converges_interior() {
    let (x, y, ids) = multislope_dataset();
    let mut ws = LmmWorkspace::with_groupings(3, multislope_groupings());
    ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let fit = fit_lmm(&mut ws, &[1, 2], None);
    assert!(fit.converged);
    // Planted [0.5, 0.4, 0.2]; recovered ≈[0.51, 0.64, 0.28]. Both slopes positive
    // with β̂₁ > β̂₂ (planted ordering preserved) — pin that, so a β₁/β₂ swap or a
    // scale collapse fails where the old `is_finite` pair passed.
    assert!(
        (0.2..0.9).contains(&ws.fit.betas[0]),
        "intercept {}",
        ws.fit.betas[0]
    );
    assert!(
        (0.2..1.1).contains(&ws.fit.betas[1]),
        "slope x1 {}",
        ws.fit.betas[1]
    );
    assert!(
        (0.0..0.7).contains(&ws.fit.betas[2]),
        "slope x2 {}",
        ws.fit.betas[2]
    );
    assert!(
        ws.fit.betas[1] > ws.fit.betas[2],
        "x1 slope must exceed x2 slope"
    );
    // Deterministic regression lock alongside the bands above.
    assert!((ws.fit.betas[0] - 0.5129839426148501).abs() / 0.5129839426148501_f64.abs() < 1e-6);
    assert!((ws.fit.betas[1] - 0.6442611282130077).abs() / 0.6442611282130077_f64.abs() < 1e-6);
    assert!((ws.fit.betas[2] - 0.28355377896623535).abs() / 0.28355377896623535_f64.abs() < 1e-6);
    assert_eq!(fit.pinned_components & !0b111, 0); // only 3 components exist
}

/// The experimental two-stage warm restart must reach the same
/// optimum as single-stage on a well-behaved rung — stage 1 (npt = n+2,
/// rho_end 1e-3, measured correctness-safe on the validation corpus) finds the
/// basin, stage 2 (npt = 2n+1, shipped rho_end) refines from stage 1's point.
/// Uses the multislope fixture (n_theta = 6) so the shipped mid-npt formula
/// (`n_theta >= 3`) is the one exercised by the single-stage comparator.
#[test]
fn two_stage_matches_single_stage_optimum() {
    let (x, y, ids) = multislope_dataset();
    let mut ws1 = LmmWorkspace::with_groupings(3, multislope_groupings());
    ws1.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let targets = [1u32, 2];
    let f1 = fit_lmm(&mut ws1, &targets, None);

    let (x, y, ids) = multislope_dataset();
    let mut ws2 = LmmWorkspace::with_groupings(3, multislope_groupings());
    ws2.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let f2 = fit_lmm_two_stage(&mut ws2, &targets, None);

    assert!(f2.converged);
    assert!(
        (f1.deviance - f2.deviance).abs() < 1e-6,
        "two-stage must land on the same optimum: {} vs {}",
        f1.deviance,
        f2.deviance
    );
    assert!(f2.n_eval > 0);
}

/// Slope-variance collapse pins the SLOPE component (bit 1), not the
/// intercept. x1 is a within-cluster antithetic ±1 pattern that carries a
/// real fixed slope but ZERO cluster-varying slope, and the residual is a
/// ±0.8 period-4 quadrature block (+,+,−,− against x1's +,−,+,−) so every
/// cluster has Σ resid = 0 AND Σ x1·resid = 0 exactly — the REML
/// slope-variance MLE is 0, so λ₁₁ pins (bit 1) while the planted u₀ keeps
/// λ₀₀ interior. (The original lockstep ±0.8 pattern made resid ≡ 0.8·x1 —
/// collinear with the slope covariate, so σ̂²→0 once large θ₀ absorbed the
/// exactly-identified cluster means, the deviance ran unbounded to the θ₀
/// box bound, and the λ₁₁ pin rode FP noise on the degenerate surface; the
/// quadrature pattern keeps σ̂² positive and θ̂₀ genuinely interior.) Large
/// balanced design (16 clusters × 16 rows) so finite-sample REML does not
/// overfit a spurious slope RE the way a small noisy draw does.
#[test]
fn zero_slope_variance_pins_slope_component() {
    let (nc, per) = (16usize, 16usize);
    let n = nc * per;
    let mut st = 5u64;
    let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut ids = vec![0u32; n];
    #[allow(clippy::needless_range_loop)]
    for c in 0..nc {
        for k in 0..per {
            let i = c * per + k;
            ids[i] = c as u32;
            // x1: identical antithetic pattern in every cluster (±1
            // alternating) — no between-cluster slope signal.
            let x1 = if k % 2 == 0 { 1.0 } else { -1.0 };
            // residual: ±0.8 period-4 quadrature against x1, so per cluster
            // Σ x1·resid = 0 AND Σ resid = 0 (no slope/intercept RE pull
            // from the noise; only the planted u₀ moves intercepts).
            let e = if (k / 2) % 2 == 0 { 0.8 } else { -0.8 };
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5 + 0.4 * x1 + u0[c] + e;
        }
    }
    let mut ws = LmmWorkspace::with_groupings(
        2,
        LmmGroupings::from_cluster_spec(
            &ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters {
                        n_clusters: nc as u32,
                    },
                    slopes: vec![0],
                    extra_groupings: vec![],
                }),
            },
            n,
            &[1],
        ),
    );
    ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let fit = fit_lmm(&mut ws, &[1], None);
    assert!(fit.converged);
    assert!(
        ws.theta[2] == 0.0,
        "slope λ₁₁ must pin to exactly 0, got {:e}",
        ws.theta[2]
    );
    assert!(fit.pinned_components & 0b10 != 0, "slope component bit set");
    assert!(
        ws.theta[0] > PIN_THETA,
        "intercept component must stay interior"
    );
    assert!(
        ws.theta[0] < THETA_HI,
        "intercept component must be off the box bound"
    );
}

// -----------------------------------------------------------------------
// Composition: primary slope (1 + x1 | g) co-existing with an
// intercept-only crossed (1 | item) / nested (1 | g:sub) extra. The
// family-blocked deviance must match a brute-force V = I + Z_p D_p Z_p′ +
// τ_e² Z_e Z_e′. Data on the f32 plane (the suff-stats input convention);
// the oracle widens the identical bytes, so the 1e-8 match is exact.
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
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![0],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 5 },
                slopes: vec![],
            }],
        }),
    };
    LmmGroupings::from_cluster_spec(&cluster, 80, &[1])
}

// --- θ-layout generalization (scalar → vech ranges) ---

/// One intercept-only primary + one crossed grouping of RE width `q_g`
/// (intercept + `q_g−1` slopes), expressed through the slope machinery — the
/// θ-layout fixture. Slope columns are placeholders (layout reads only
/// `slopes.len()`).
fn groupings_primary1_crossed_qg(q_g: usize) -> LmmGroupings {
    let slopes: Vec<crate::ColumnId> = (0..q_g - 1).map(|k| (k + 1) as u32).collect();
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 5 },
                slopes,
            }],
        }),
    };
    LmmGroupings::from_cluster_spec(&cluster, 80, &[])
}

#[test]
fn extra_qg1_theta_layout_matches_scalar() {
    // Intercept-only crossed factor through the slope machinery = the old
    // scalar layout: one primary scalar + one extra scalar.
    let g = groupings_primary1_crossed_qg(1);
    assert_eq!(g.n_theta(), 1 + 1);
    assert_eq!(g.crossed[0].vech_start, 1);
    assert_eq!(g.crossed[0].q, 1);
    assert!(!g.extra_slopes_any);
}

#[test]
fn extra_qg2_theta_packs_vech3() {
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
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 2 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 3 },
                slopes: vec![1],
            }],
        }),
    };
    // crossed slope on x_full col 1.
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[], &[vec![1]]);
    // crossed block: q_g=2, offset = prim_width = 2 (n_primary=2, q_p=1).
    assert_eq!(g.extra_offsets[0], 2);
    assert_eq!(g.extra_q[0], 2);
    let mut suff = LmmSuffStats::with_groupings(p, g);
    suff.add_rows_multi(x.as_ref(), &y, &cluster_ids, &[crossed_ids.to_vec()], None);
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
    reml_profile_from_v(&v, x, y)
}

/// REML profiled deviance from an explicit n×n marginal V (in residual-σ²
/// units): `log|V| + log|XᵀV⁻¹X| + (N−P)·log σ̂²`. The shared V→deviance back
/// end for every brute-force oracle (composed, crossed-slope, …).
fn reml_profile_from_v(v: &Mat<f64>, x: &Mat<f64>, y: &[f64]) -> f64 {
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
    let df = (n - p) as f64;
    let s2 = (ytviy - bxy) / df;
    ldv + ldk + df * s2.ln()
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
    reml_profile_from_v(&v, x, y)
}

/// Slope + crossed: the composed deviance matches the brute-force oracle to
/// 1e-8 — the slope-composition gate. zx_slope carries the slope↔crossed
/// coupling; the primary 2×2 block and the item intercept block are coupled
/// through the shared family-blocked tail.
#[test]
fn composed_deviance_matches_brute_force() {
    let (x, y, pid, iid) = composed_dataset();
    let mut suff = LmmSuffStats::with_groupings(2, composed_groupings());
    suff.add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&iid), None); // item ids as the single extra grouping
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
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 5 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 4 },
                slopes: vec![1],
            }],
        }),
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
        let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
        let item = extra_level_of_row(&cluster, 0, i) as usize;
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
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
    assert!(g.extra_slopes_any, "must route to the blocked path");
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(x.as_ref(), &y, &pid, &[eid.clone()], None);
    let gref = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
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

/// NESTED SLOPES (the nested-slope defect): `y ~ x1 + (1+x1 | grp) + (1+x1 | class)`,
/// class nested in grp — both grouping factors carry a random slope on x1, so
/// the gated blocked path runs with a nested factor of q_n = 2. Before the
/// fix the blocked path assembled the nested children intercept-only (scalar
/// θ_n), diverging to NaN. The marginal V is grouping-agnostic (Σ_g Z_g D_g Z_gᵀ
/// over rows sharing a level id), so the crossed-slope oracle is reused with the
/// GLOBAL nested child id as the extra level. Matches the explicit-V oracle to
/// 1e-7 across θ.
#[test]
fn nested_slope_deviance_matches_brute_force() {
    let n_per_parent = 3u32;
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 5 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent },
                slopes: vec![1],
            }],
        }),
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
        let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
        let child = extra_level_of_row(&cluster, 0, i); // GLOBAL child id
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
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
    assert!(g.extra_slopes_any, "must route to the blocked path");
    assert!(g.nested.is_some(), "must carry a nested factor");
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(x.as_ref(), &y, &pid, &[eid.clone()], None);
    let gref = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
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

/// End-to-end NESTED-SLOPE fit: the original nested-slope symptom was BOBYQA diverging
/// to NaN (`converged = false`) on every seed because the blocked objective was
/// mis-assembled. With the correct objective the full θ-search must converge to
/// a finite interior fit. Asserts `converged`, no numerical failure
/// (`boundary_hit != 2`), finite θ̂/σ̂², and β̂ recovered near the planted
/// [0.5, 0.4].
#[test]
fn nested_slope_fit_converges() {
    let n_per_parent = 3u32;
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 5 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent },
                slopes: vec![1],
            }],
        }),
    };
    let n = 120; // atom = 5·3 = 15 ⇒ 8 balanced blocks
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
        let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
        let child = extra_level_of_row(&cluster, 0, i);
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
    let mut ws = LmmWorkspace::for_cluster_spec_ext(2, &cluster, n, &[1], &[vec![1]]);
    ws.suff.reset();
    ws.suff
        .add_rows_multi(x.as_ref(), &y, &pid, &[eid.clone()], None);
    let fit = fit_lmm(&mut ws, &[1], None);
    assert!(fit.converged, "nested-slope fit must converge");
    assert_ne!(fit.boundary_hit, 2, "must not be a numerical (NaN) failure");
    assert!(
        fit.sigma_sq.is_finite() && fit.sigma_sq > 0.0,
        "σ̂² {}",
        fit.sigma_sq
    );
    assert!(ws.theta.iter().all(|t| t.is_finite()), "θ̂ {:?}", ws.theta);
    assert!(
        (0.2..0.8).contains(&ws.fit.betas[0]),
        "intercept {}",
        ws.fit.betas[0]
    );
    assert!(
        (0.1..0.7).contains(&ws.fit.betas[1]),
        "slope {}",
        ws.fit.betas[1]
    );
    // Deterministic regression lock (seed 137) alongside the wide recovers-check above.
    assert!((ws.fit.betas[0] - 0.6209080774915476).abs() / 0.6209080774915476_f64.abs() < 1e-6);
    assert!((ws.fit.betas[1] - 0.257915422474595).abs() / 0.257915422474595_f64.abs() < 1e-6);
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
    reml_profile_from_v(&v, x, y)
}

/// TWO crossed factors with slopes:
/// `y ~ x1 + (1 | primary) + (1+x1 | c1) + (1+x1 | c2)`. Exercises the
/// crossed↔crossed slope coupling (c1's slope column against c2's, the x1²
/// weighted co-occurrence between two distinct crossed factors) — the part
/// neither the composed nor single-crossed test reaches. Matches the
/// explicit-V oracle to 1e-7.
#[test]
fn two_crossed_slopes_deviance_matches_brute_force() {
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 3 },
            slopes: vec![], // primary intercept-only
            extra_groupings: vec![
                Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 3 },
                    slopes: vec![1],
                },
                Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 3 },
                    slopes: vec![1],
                },
            ],
        }),
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
        let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
        let a = extra_level_of_row(&cluster, 0, i) as usize;
        let b = extra_level_of_row(&cluster, 1, i) as usize;
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
    let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[], &[vec![1], vec![1]]);
    assert!(g.extra_slopes_any);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(x.as_ref(), &y, &pid, &[c1.clone(), c2.clone()], None);
    let gref = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[], &[vec![1], vec![1]]);
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
/// `y = 1.0 + 0.8·x1 + u0p + u1p·x1 + u0e + u1e·x1 + ε`. The Rust generator is
/// the source of truth; `dump_crossed_slope_golden_csv` writes it for the R
/// `lme4::lmer` reference whose fit is frozen in `GOLDEN_LME4_*`.
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

/// L3 golden: `glmm`'s crossed-slope fit must reproduce `lme4::lmer`'s REML fit
/// of `y ~ x1 + (1+x1|pid) + (1+x1|eid)` on the committed dataset — fixed
/// effects, residual σ², and both 2×2 RE covariances. Frozen from
/// `/tmp/golden_fit.R` (lme4 1.1, bobyqa). Recovered D_g = σ̂²·Λ_gΛ_gᵀ from θ̂.
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
    let n = y.len();
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 8 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 6 },
                slopes: vec![1],
            }],
        }),
    };
    let mut ws = LmmWorkspace::for_cluster_spec_ext(2, &cluster, n, &[1], &[vec![1]]);
    ws.suff.reset();
    ws.suff
        .add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&eid), None);
    let fit = fit_lmm(&mut ws, &[1], None);
    assert!(fit.converged, "golden fit must converge");
    let s2 = fit.sigma_sq;

    // Fixed effects + residual variance.
    assert!(
        (ws.fit.betas[0] - G_BETA0).abs() < 1e-4,
        "β0 {} vs {G_BETA0}",
        ws.fit.betas[0]
    );
    assert!(
        (ws.fit.betas[1] - G_BETA1).abs() < 1e-4,
        "β1 {} vs {G_BETA1}",
        ws.fit.betas[1]
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
    let cluster = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedSize { cluster_size: 8 },
            slopes: vec![0],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                slopes: vec![],
            }],
        }),
    };
    let n = 4 * model_atom(&cluster); // 64
    let mut st = 47u64;
    let u0: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
    let u1: Vec<f64> = (0..8).map(|_| 0.3 * lcg(&mut st)).collect();
    let u_c: Vec<f64> = (0..16).map(|_| 0.35 * lcg(&mut st)).collect(); // 8 parents × 2 children
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut pid = vec![0u32; n];
    let mut cid = vec![0u32; n]; // globalized child id (parent·np + within)
    for i in 0..n {
        let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
        let child = extra_level_of_row(&cluster, 0, i); // already globalized par·np + within
        pid[i] = par as u32;
        cid[i] = child as u32;
        let x1 = lcg(&mut st);
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        y[i] = 0.5 + 0.4 * x1 + u0[par] + u1[par] * x1 + u_c[child] + 0.8 * lcg(&mut st);
    }
    let g = LmmGroupings::from_cluster_spec(&cluster, n, &[1]);
    let mut suff = LmmSuffStats::with_groupings(2, g);
    suff.add_rows_multi(x.as_ref(), &y, &pid, &[cid.clone()], None);
    let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[1]);
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

/// Bounded-allocation twin — the standalone slope workspace
/// allocates only faer `llt` internals on the warm `fit_lmm` loop, the same
/// acceptance class as the q=1 / general twins.
///   cargo test -p glmm --features alloc-tests lmm_fit_slope_warm_path_bounded_alloc -- --ignored
#[cfg(feature = "alloc-tests")]
#[test]
#[ignore]
fn lmm_fit_slope_warm_path_bounded_alloc() {
    let _serial = crate::test_support::alloc_test_guard();
    const N_CALLS: usize = 100;
    const BOUND_SLOPE: u64 = 12000; // Measured 11400 (this machine) — ~114 blocks/fit of faer `llt` internals (one m×m tail llt per eval × ~54 evals on the blind 3-D q_p=2 surface; the family loop + primary Λ/Gram are zero-alloc scratch, the cached diagonal_theta map removed the per-fit Vec, and the ranef recovery pass solves in the ranef_ux/ranef_rhs scratch fields). Higher total than q=1's 4600 only via the larger blind eval count, not a richer per-eval alloc — faer-version/machine specific. If faer's Cholesky internals change, update — do not relax.

    let (x, y, ids) = slope_dataset();
    let targets: Vec<u32> = vec![1];
    let mut ws = LmmWorkspace::with_groupings(2, slope_groupings());

    ws.suff.reset();
    ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
    let _ = fit_lmm(&mut ws, &targets, None);

    let profiler = dhat::Profiler::builder().testing().build();
    for _ in 0..N_CALLS {
        ws.suff.reset();
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let _ = fit_lmm(&mut ws, &targets, None);
    }
    let stats = dhat::HeapStats::get();
    drop(profiler);
    assert!(
        stats.total_blocks <= BOUND_SLOPE,
        "slope fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
        stats.total_blocks,
        N_CALLS,
        BOUND_SLOPE
    );
}
