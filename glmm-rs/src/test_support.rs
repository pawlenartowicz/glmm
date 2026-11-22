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
use crate::{Data, Fit, FitOptions, GroupIds, ModelSpec, StartValues};
use faer::Mat;

/// The unit tests' cold fit: no warm start, no weights, no offset. Unit tests
/// reach the crate's entry point only through this and [`fit_warm_wo`], so a
/// change of entry point edits these two bodies alone.
pub(crate) fn fit_cold(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    opts: &FitOptions,
) -> Fit {
    fit_warm_wo(x, y, n, p, model, ids, None, None, None, opts)
}

/// [`fit_cold`] with an optional warm start, prior weights and offset. The
/// ids reach the fit only when `model` has random effects, so a fixed-only
/// test may pass whatever ids it built. Panics with the [`crate::FitError`]'s
/// message on a refused input.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_warm_wo(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    start: Option<&StartValues>,
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    opts: &FitOptions,
) -> Fit {
    let mut data = Data::new(x, y, n, p);
    if model.re.is_some() {
        data = data.ids(ids);
    }
    if let Some(w) = weights {
        data = data.weights(w);
    }
    if let Some(o) = offset {
        data = data.offset(o);
    }
    let fit = match start {
        Some(s) => crate::fit_with_start(&data, model, s, opts),
        None => crate::fit(&data, model, opts),
    };
    fit.unwrap_or_else(|e| panic!("{e}"))
}

/// Near-identity slice comparison shared by the fit-core equivalence tests
/// (`fit_on`-on-reused-ws vs a fresh fit). Tolerance is
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
/// profiler at a time, so concurrent tests attribute each other's allocations.
/// The bounded-alloc tests hold this for their whole body; the non-alloc
/// `#[ignore]` tests take it too (feature-gated), since their allocations
/// would otherwise land in a concurrent profiler window on an `-- --ignored`
/// run. It serializes test bodies only: libtest spawns each test's OS thread
/// up front and does its own per-thread bookkeeping before that thread ever
/// reaches the guard, so those allocations can still land in another test's
/// open profiler window. The tier is run `--test-threads=1` for that reason.
/// Every alloc test also needs `RAYON_NUM_THREADS=1` — rayon worker startup
/// otherwise adds ~4-24 stray blocks to the count. Poisoning is deliberately
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
    pub irls_broken: Vec<bool>,
}

impl TestWs {
    /// Allocate scratch sized for `max_n` rows, `n_predictors` columns.
    /// Alloc lines copied verbatim from `SimWorkspace::new`. `_max_n_clusters`
    /// is accepted-but-unused: `TestWs` holds no cluster-sized LME scratch, so
    /// the parameter is kept only so every existing `TestWs::new` call site
    /// (positional, cluster count still passed) needs no change.
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
            irls_broken: vec![false; n_predictors],
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

/// n = 40 rows in 8 groups of 5, row-major `[1, x]`, a 0/1 response and a
/// count response built from fixed effects, group effects and a noise
/// sequence; returns `(x, binary, counts, group ids)`.
pub(crate) fn tiny_weight_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>) {
    const U_G: [f64; 8] = [-0.6, 0.4, 0.1, -0.3, 0.7, -0.2, 0.5, -0.4];
    let n = 40;
    let x_seq = |i: usize| (1.7 * i as f64 + 0.3).sin();
    let noise = |i: usize| (2.3 * i as f64 + 0.7).sin();
    let g: Vec<u32> = (0..n as u32).map(|i| i / 5).collect();
    let x = (0..n).flat_map(|i| [1.0, x_seq(i)]).collect();
    let u = |i: usize| U_G[g[i] as usize];
    let binary = (0..n)
        .map(|i| f64::from(u8::from(0.2 + 0.8 * x_seq(i) + u(i) + 0.8 * noise(i) > 0.0)))
        .collect();
    let counts = (0..n)
        .map(|i| {
            (0.5 + 0.3 * x_seq(i) + 0.5 * u(i) + 0.3 * noise(i))
                .exp()
                .round()
        })
        .collect();
    (x, binary, counts, g)
}

/// Intercept-only Gaussian mixed `ModelSpec` — mirrors
/// `engine_contract::ClusterSpec::intercept_only`.
pub(crate) fn intercept_only_spec() -> crate::ModelSpec {
    crate::ModelSpec::mixed(crate::Family::Gaussian, crate::ReStructure::intercept())
}

/// Cluster sizing regime of the positional test layout: a fixed cluster
/// count, or a fixed per-cluster size. The layout helpers below build
/// `GroupIds` for tests from counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Sizing {
    /// Fixed number of clusters; per-cluster size grows with total N.
    FixedClusters {
        /// Cluster count, held constant as N scales.
        n_clusters: u32,
    },
    /// Fixed per-cluster size; cluster count grows with total N.
    FixedSize {
        /// Rows per cluster, held constant as N scales.
        cluster_size: u32,
    },
}

impl Sizing {
    /// Number of clusters when total row count is `n`: the fixed `n_clusters`
    /// itself under `FixedClusters`, or `n / cluster_size` rounded UP under
    /// `FixedSize`. Rounding up matches the positional row layout (row `i` in
    /// cluster `i / cluster_size`): off-grid `n` leaves a partial trailing cluster
    /// whose rows still carry a real id, and that id must be in range.
    pub(crate) fn n_clusters_at(&self, n: usize) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => n.div_ceil((*cluster_size).max(1) as usize),
        }
    }
}

/// Grid atom for the full grouping structure: the primary block times every
/// extra grouping's count. `extra_counts[g]` is the level count of a crossed
/// grouping and the children per parent of a nested one, declaration order.
/// Verbatim mirror of `engine_contract::ClusterSpec::atom` (test-only DGP
/// layout).
pub(crate) fn model_atom(sizing: &Sizing, extra_counts: &[u32]) -> usize {
    let base = match sizing {
        Sizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
        Sizing::FixedSize { cluster_size } => (*cluster_size).max(1) as usize,
    };
    extra_counts
        .iter()
        .fold(base, |a, &c| a * c.max(1) as usize)
}

/// Level of extra grouping `g` that row `i` belongs to under the positional
/// layout; `extra_counts` as in [`model_atom`]. A nested level is the GLOBAL
/// child id (`parent · per_parent + within`).
pub(crate) fn extra_level_of_row(
    spec: &crate::ModelSpec,
    sizing: &Sizing,
    extra_counts: &[u32],
    g: usize,
    i: usize,
) -> usize {
    let re = spec
        .re
        .as_ref()
        .expect("extra_level_of_row requires re: Some");
    let rel = re.extra_groupings[g].relation;
    let levels = |h: usize| extra_counts[h].max(1) as usize;
    match sizing {
        Sizing::FixedClusters { n_clusters } => {
            let s = (*n_clusters).max(1) as usize;
            let stride = (0..g).fold(s, |a, h| a * levels(h));
            let within = (i / stride) % levels(g);
            match rel {
                crate::GroupingRelation::Crossed => within,
                crate::GroupingRelation::NestedWithin => (i % s) * levels(g) + within,
            }
        }
        Sizing::FixedSize { cluster_size } => {
            let cs = (*cluster_size).max(1) as usize;
            (i / cs) * levels(g) + (i % cs) % levels(g)
        }
    }
}

/// Positional `GroupIds` for `n` rows of `spec` under `sizing`, with
/// `extra_counts` as in [`model_atom`]: the balanced simulator layout, built
/// from counts up front and then fitted through the same entry as real data.
pub(crate) fn ids_from_sizing(
    spec: &crate::ModelSpec,
    sizing: &Sizing,
    extra_counts: &[u32],
    n: usize,
) -> GroupIds {
    let primary: Vec<u32> = (0..n).map(|i| cluster_of_row(sizing, i) as u32).collect();
    let extra: Vec<Vec<u32>> = (0..extra_counts.len())
        .map(|g| {
            (0..n)
                .map(|i| extra_level_of_row(spec, sizing, extra_counts, g, i) as u32)
                .collect()
        })
        .collect();
    GroupIds { primary, extra }
}

/// A CSV in the validation corpus's format (`,`-split, quotes trimmed) as a
/// formula `Table`, by [`table_from_rows`].
#[cfg(feature = "formula")]
pub(crate) fn csv_table(csv: &str, factors: &[&str]) -> crate::formula::Table {
    let split = |l: &str| -> Vec<String> {
        l.split(',')
            .map(|c| c.trim().trim_matches('"').to_string())
            .collect()
    };
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header = split(lines.next().expect("a CSV header"));
    let rows: Vec<Vec<String>> = lines.map(split).collect();
    table_from_rows(&header, &rows, factors)
}

/// CSV cells as a formula `Table`: the columns named in `factors`, and any
/// column with a cell that does not parse as `f64`, are factors; every other
/// column is numeric.
#[cfg(feature = "formula")]
pub(crate) fn table_from_rows(
    header: &[String],
    rows: &[Vec<String>],
    factors: &[&str],
) -> crate::formula::Table {
    use crate::formula::Column;
    let columns = header
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let cells: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
            let col = if factors.contains(&name.as_str())
                || cells.iter().any(|c| c.parse::<f64>().is_err())
            {
                Column::factor_from_labels(&cells)
            } else {
                Column::Numeric(cells.iter().map(|c| c.parse().unwrap()).collect())
            };
            (name.clone(), col)
        })
        .collect();
    crate::formula::Table {
        columns,
        n: rows.len(),
    }
}

/// `table` with the label "zzz" inserted into `name`'s levels at `at`, used by
/// no row (codes at or past `at` shift up by one).
#[cfg(feature = "formula")]
pub(crate) fn with_unused(
    table: &crate::formula::Table,
    name: &str,
    at: usize,
) -> crate::formula::Table {
    use crate::formula::Column;
    let columns = table
        .columns
        .iter()
        .map(|(n, col)| {
            let col = match col {
                Column::Factor { levels, codes } if n == name => {
                    let mut levels = levels.clone();
                    levels.insert(at, "zzz".to_string());
                    let codes = codes
                        .iter()
                        .map(|&c| c + u32::from(c as usize >= at))
                        .collect();
                    Column::Factor { levels, codes }
                }
                Column::Factor { levels, codes } => Column::Factor {
                    levels: levels.clone(),
                    codes: codes.clone(),
                },
                Column::Numeric(v) => Column::Numeric(v.clone()),
            };
            (n.clone(), col)
        })
        .collect();
    crate::formula::Table {
        columns,
        n: table.n,
    }
}

/// `sim_factorial` (120 rows, 20 subjects `subj`, factors `g` and `h`, a
/// numeric `x`) with weights `1 + (i mod 7)/2`, fitted on `formula` with the
/// weights as given and `se_type`.
#[cfg(feature = "formula")]
pub(crate) fn weighted_factorial(formula: &str, se_type: crate::SeType) -> crate::formula::Model {
    let csv = include_str!("../../validation/data/simulated/sim_factorial.csv");
    let table = csv_table(csv, &["subj", "g", "h"]);
    let mut lo = crate::formula::lower(formula, &table, crate::Family::Gaussian).unwrap();
    lo.weights = Some((0..lo.n).map(|i| 1.0 + (i % 7) as f64 / 2.0).collect());
    let opts = crate::FitOptions {
        weight_power: Some(crate::WeightPower::Fixed(1.0)),
        se_type: Some(se_type),
        ..crate::FitOptions::default()
    };
    let m = crate::formula::fit_lowered(formula, lo, &opts).unwrap();
    assert!(m.fit().converged(), "{formula}: not converged");
    m
}

/// Primary cluster owning row `i` under the positional layout: rows dealt
/// round-robin (`i % n_clusters`) under `FixedClusters`, contiguous blocks
/// (`i / cluster_size`) under `FixedSize`. Mirrors
/// `engine_contract::ClusterSizing::cluster_of_row` (test-only DGP layout).
pub(crate) fn cluster_of_row(sizing: &Sizing, i: usize) -> usize {
    match sizing {
        Sizing::FixedClusters { n_clusters } => i % (*n_clusters).max(1) as usize,
        Sizing::FixedSize { cluster_size } => i / (*cluster_size).max(1) as usize,
    }
}

/// Profiled deviance from an explicit n×n marginal V (in residual-σ²
/// units), r² = y′V⁻¹y − β̂′X′V⁻¹y: under REML `log|V| + log|XᵀV⁻¹X| +
/// (N−P)·log(r²/(N−P))`, under ML `log|V| + N·log(r²/N)`. The shared
/// V→deviance back end for every brute-force oracle (composed, crossed-slope, …).
pub(crate) fn profile_from_v(
    v: &Mat<f64>,
    x: &Mat<f64>,
    y: &[f64],
    criterion: crate::lmm::Criterion,
) -> f64 {
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
        crate::lmm::Criterion::Reml => {
            let df = (n - p) as f64;
            ldv + ldk + df * (r_sq / df).ln()
        }
        crate::lmm::Criterion::Ml => ldv + n as f64 * (r_sq / n as f64).ln(),
    }
}

/// Brute-force profiled deviance on the explicit n×n marginal
/// `V = diag(1/wᵢ) + Σ_g Z_g D_g Z_gᵀ` (unit weights when `w` is `None`), in
/// residual-σ² units. Each factor `(ids, slope_cols, vech)` is one grouping:
/// rows with equal ids share a level, row i of `Z_g` is
/// `[1, x[i, slope_cols]]`, and `D_g = ΛΛᵀ` from the column-major vech of the
/// lower-triangular `Λ` (`q = 1 + slope_cols.len()`). The kernels work on the
/// √w-scaled rows, so a weighted kernel deviance is this value plus `Σ ln wᵢ`.
pub(crate) fn brute_force_deviance(
    x: &Mat<f64>,
    y: &[f64],
    w: Option<&[f64]>,
    factors: &[(&[u32], &[usize], &[f64])],
    criterion: crate::lmm::Criterion,
) -> f64 {
    let n = x.nrows();
    let mut v = Mat::<f64>::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = w.map_or(1.0, |w| 1.0 / w[i]);
    }
    for &(ids, slope_cols, vech) in factors {
        let q = 1 + slope_cols.len();
        assert_eq!(vech.len(), q * (q + 1) / 2, "vech length for q = {q}");
        let mut lam = vec![0.0f64; q * q];
        let mut t = 0;
        for c in 0..q {
            for r in c..q {
                lam[r * q + c] = vech[t];
                t += 1;
            }
        }
        let mut d = vec![0.0f64; q * q];
        for i in 0..q {
            for j in 0..q {
                d[i * q + j] = (0..q).map(|k| lam[i * q + k] * lam[j * q + k]).sum();
            }
        }
        let zrow = |i: usize| -> Vec<f64> {
            std::iter::once(1.0)
                .chain(slope_cols.iter().map(|&c| x[(i, c)]))
                .collect()
        };
        let z: Vec<Vec<f64>> = (0..n).map(zrow).collect();
        for i in 0..n {
            for j in 0..n {
                if ids[i] == ids[j] {
                    let mut acc = 0.0;
                    for a in 0..q {
                        for b in 0..q {
                            acc += z[i][a] * d[a * q + b] * z[j][b];
                        }
                    }
                    v[(i, j)] += acc;
                }
            }
        }
    }
    profile_from_v(&v, x, y, criterion)
}
