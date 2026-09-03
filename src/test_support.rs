//! Test-only scaffolding shared across the fit-module `#[cfg(test)]` blocks.
//!
//! `TestWs` is a buffer-bag mirroring the field names/reset semantics of the
//! kernel's own workspace, for the subset the fit-module tests borrow, so each test's
//! field-borrowing helper body (`suff_stats`, `glm_scratch`) ports with
//! minimal change. It carries no
//! design-gen / RNG / critval machinery — the tests only use it as pre-sized
//! scratch. The alloc lines + the two reset methods mirror the kernel workspace's
//! `new` / its resets; the cluster count is passed directly.

use crate::ols::PANEL_ROWS;
use faer::Mat;

/// Ceiling on `Diagnostics::kkt_grad_norm` at a converged GLMM optimum,
/// interior or boundary — "at or below what BOBYQA actually leaves", not zero
/// (BOBYQA stops on a trust radius, not on a gradient). Shared by the KKT
/// tests in `src/glmm/tests.rs` and `src/fit/common_tests.rs`.
///
/// Calibrated 2026-09-01 by `kkt_calibration_measurement`
/// (`tests/validation_oracle.rs`, `--features oracle-tests -- --ignored`) over
/// every GLMM golden the cross-engine tier loads plus the committed
/// `glmm_hessian_vcov.json` fixture: worst finite residual 1.1311e-1
/// (`sim_cloglog_glmm`, deviance ≈ 9848; the rest sit between 5.5e-8 and
/// 2.2e-2). Pinned absolute — the residuals do not cleanly track `|deviance|`
/// across rungs (ratio spans ~5 decades) — at ceil-to-one-significant-figure
/// of ten times that worst, the margin convention `validation/tol.R` uses for
/// its own measured bands.
pub(crate) const KKT_INTERIOR_MAX: f64 = 2.0;

/// Near-identity slice comparison shared by the fit-core equivalence tests
/// (`fit_on`-on-reused-ws vs `fit_cold`, view-mapper vs direct). Tolerance is
/// `1e-12 + 1e-9·|want|` — tight enough to catch every failure these gates
/// exist for (stale buffers, `n_max` over-reads, option-reset misses shift
/// results materially, not by an ULP) without demanding bit-identity across a
/// refactor. Scalars compare as one-element slices: `assert_near(&[a], &[b], _)`.
pub(crate) fn assert_near(got: &[f64], want: &[f64], ctx: &str) {
    assert_eq!(
        got.len(),
        want.len(),
        "{ctx}: len {} vs {}",
        got.len(),
        want.len()
    );
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        // Both NaN is near-identical (a non-target SE / non-converged slot is
        // NaN in both fresh and reused); one NaN alone is a real mismatch.
        if g.is_nan() && w.is_nan() {
            continue;
        }
        assert!(
            (g - w).abs() <= 1e-12 + 1e-9 * w.abs(),
            "{ctx}[{i}]: {g} vs {w}"
        );
    }
}

/// Serializes every `#[ignore]` lib test under `--features alloc-tests`.
/// `dhat::Profiler` counts process-wide allocations and permits one live
/// profiler at a time, so concurrent tests attribute each other's allocations
/// (an OLS route with no optimizer once "regressed" this way). The
/// bounded-alloc tests hold this for their whole body; the non-alloc
/// `#[ignore]` tests take it too (feature-gated), since their allocations
/// would otherwise land in a concurrent profiler window on an `-- --ignored`
/// run. This makes `--test-threads=1` unnecessary. Poisoning is deliberately
/// swallowed: one failing test must not cascade into the rest.
#[cfg(feature = "alloc-tests")]
pub(crate) fn alloc_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Blocks until no thread but ours is allocating, so a bounded-alloc test's
/// profiler window measures only its own calls.
///
/// faer pulls in rayon, whose global pool spawns its workers lazily on first
/// use and *asynchronously*: each worker's own startup allocations
/// (crossbeam-epoch `Local::register`, crossbeam-deque `JobFifo`, the
/// stack-overflow handler's `ThreadInfo`) land milliseconds after the fit that
/// triggered the pool, on threads dhat counts all the same. On a 16-core box
/// that is ~30 stray blocks drifting into whatever window is open — enough to
/// blow a tight pin, and timing-dependent, so it looks like the kernel itself
/// allocates non-deterministically. Call this after the warmup fit and before
/// building the profiler.
///
/// Quiet is defined by measurement, not by a fixed sleep: sample short idle
/// windows until one comes back at zero blocks. The cap keeps a genuinely
/// noisy process from hanging the test — it then fails on the real bound.
#[cfg(feature = "alloc-tests")]
pub(crate) fn settle_background_allocs() {
    for _ in 0..200 {
        let probe = dhat::Profiler::builder().testing().build();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let blocks = dhat::HeapStats::get().total_blocks;
        drop(probe);
        if blocks == 0 {
            return;
        }
    }
}

pub(crate) struct TestWs {
    // OLS suff-stats
    pub suff_xtx: Mat<f64>,
    pub suff_xty: Vec<f64>,
    pub suff_yty: f64,
    pub suff_sum_y: f64,
    pub suff_n_rows: usize,
    pub suff_xtx_work: Mat<f64>,
    pub panel_x: Vec<f64>,
    pub panel_y: Vec<f64>,

    // OLS fit scratch
    pub fit_betas: Vec<f64>,
    pub fit_var_diag: Vec<f64>,
    pub fit_t_sq: Vec<f64>,
    pub fit_u_scratch: Vec<f64>,
    pub fit_factor: Mat<f64>,
    pub fit_rhs: Mat<f64>,

    // IRLS scratch
    pub irls_eta: Vec<f64>,
    pub irls_p: Vec<f64>,
    pub irls_w: Vec<f64>,
    pub irls_z: Vec<f64>,
    pub irls_betas: Vec<f64>,
    pub irls_betas_new: Vec<f64>,
    pub irls_var_diag: Vec<f64>,
    pub irls_t_sq: Vec<f64>,
    pub irls_u_scratch: Vec<f64>,
    pub irls_xtwx: Mat<f64>,
    pub irls_xtwz: Vec<f64>,
    pub irls_l: Mat<f64>,
    pub irls_wx: Vec<f64>,
}

impl TestWs {
    /// Allocate scratch sized for `max_n` rows, `n_predictors` columns.
    /// Alloc lines copied verbatim from `SimWorkspace::new`. `_max_n_clusters`
    /// is accepted-but-unused: kept so every existing `TestWs::new` call site
    /// (positional, cluster count still passed) needs no change now that the
    /// LME scratch it used to size is gone.
    pub(crate) fn new(max_n: usize, n_predictors: usize, _max_n_clusters: usize) -> Self {
        Self {
            fit_betas: vec![0.0; n_predictors],
            fit_var_diag: vec![0.0; n_predictors],
            fit_t_sq: vec![0.0; n_predictors],
            fit_u_scratch: vec![0.0; n_predictors],
            fit_factor: Mat::<f64>::zeros(n_predictors, n_predictors),
            fit_rhs: Mat::<f64>::zeros(max_n.max(n_predictors), 1),

            suff_xtx: Mat::<f64>::zeros(n_predictors, n_predictors),
            suff_xty: vec![0.0; n_predictors],
            suff_yty: 0.0,
            suff_sum_y: 0.0,
            suff_n_rows: 0,
            suff_xtx_work: Mat::<f64>::zeros(n_predictors, n_predictors),
            panel_x: vec![0.0f64; PANEL_ROWS * n_predictors],
            panel_y: vec![0.0f64; PANEL_ROWS],

            irls_eta: vec![0.0; max_n],
            irls_p: vec![0.0; max_n],
            irls_w: vec![0.0; max_n],
            irls_z: vec![0.0; max_n],
            irls_betas: vec![0.0; n_predictors],
            irls_betas_new: vec![0.0; n_predictors],
            irls_var_diag: vec![0.0; n_predictors],
            irls_t_sq: vec![0.0; n_predictors],
            irls_u_scratch: vec![0.0; n_predictors],
            irls_xtwx: Mat::<f64>::zeros(n_predictors, n_predictors),
            irls_xtwz: vec![0.0; n_predictors],
            irls_l: Mat::<f64>::zeros(n_predictors, n_predictors),
            irls_wx: vec![0.0; max_n * n_predictors],
        }
    }

    /// Reset the OLS sufficient-statistics accumulator to "no rows seen".
    /// Reuses storage — zeros only the lower triangle of `suff_xtx` (the upper
    /// triangle is never read or written by `add_rows_suff` / Cholesky).
    pub(crate) fn reset_suff_stats(&mut self) {
        let p = self.suff_xty.len();
        for j in 0..p {
            for i in j..p {
                self.suff_xtx[(i, j)] = 0.0;
            }
            self.suff_xty[j] = 0.0;
        }
        self.suff_yty = 0.0;
        self.suff_sum_y = 0.0;
        self.suff_n_rows = 0;
    }
}

/// Intercept-only `ModelSpec` — mirrors `engine_contract::ClusterSpec::intercept_only`.
pub(crate) fn intercept_only_spec(sizing: crate::Sizing) -> crate::ModelSpec {
    crate::ModelSpec {
        family: crate::Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing,
            slopes: vec![],
            extra_groupings: vec![],
        }),
    }
}

/// Levels one grouping contributes to an atom block. Delegates to
/// `ids::block_levels` (the two were a verbatim duplicate) — kept as a
/// separate binding here since the DGP layout helpers below (`model_atom`,
/// `extra_level_of_row`) are `ModelSpec`-shaped while `ids`'s is
/// `ReStructure`-shaped.
pub(crate) fn block_levels(rel: &crate::GroupingRelation) -> usize {
    crate::ids::block_levels(rel)
}

/// Grid atom for the full grouping structure. Verbatim mirror of
/// `engine_contract::ClusterSpec::atom` (test-only DGP layout).
pub(crate) fn model_atom(spec: &crate::ModelSpec) -> usize {
    let re = spec.re.as_ref().expect("model_atom requires re: Some");
    re.extra_groupings
        .iter()
        .fold(re.sizing.atom(), |a, g| a * block_levels(&g.relation))
}

/// Level of extra grouping `g` that row `i` belongs to. `ModelSpec`-shaped
/// wrapper that unwraps `re` and delegates to `ids::extra_level_of_row` (the
/// `ReStructure`-shaped original) — the two bodies were a verbatim duplicate.
pub(crate) fn extra_level_of_row(spec: &crate::ModelSpec, g: usize, i: usize) -> usize {
    let re = spec
        .re
        .as_ref()
        .expect("extra_level_of_row requires re: Some");
    crate::ids::extra_level_of_row(re, g, i) as usize
}
