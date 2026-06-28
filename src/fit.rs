//! Friendly stable `fit` entry point for the `glmm` crate.
//!
//! Owns all scratch; dispatches on `ModelSpec::estimator`; returns `Fit`.
//! This is the additive stable public surface described in the carve spec §5 —
//! it never touches any kernel; only marshals data and scratch into kernel
//! calls, then copies results out.
//!
//! # Calling convention
//!
//! `x` is a design matrix in **row-major f64** layout: element `(i, j)` is at
//! `x[i * p + j]`. `y` is the response vector of length `n`. The kernels use
//! faer column-major f64 internally; conversion is done here.
//!
//! Cluster ids and extra-grouping ids are derived from `ModelSpec` using the
//! same row-layout rules as MCPower's data-gen layer (mirrors the DGP's
//! `cluster_of_row` / `extra_level_of_row` helpers).

use faer::Mat;

use crate::lmm::{fit_lmm, LmmWorkspace};
use crate::ols::{OlsScratch, OlsSuffStats, PANEL_ROWS};
use crate::{Estimator, GroupingRelation, ModelSpec, Sizing};

/// Result of `fit`. Fixed-effect estimates cover all p predictors; SE and
/// tau2 have the ranges below. Non-target SE slots are NaN.
pub struct Fit {
    /// Fixed-effect estimates, length p.
    pub beta: Vec<f64>,
    /// Standard errors: `se[j] = sqrt(Var(β̂_j))` for target predictors,
    /// NaN for non-targets. Length p.
    pub se: Vec<f64>,
    /// Per-element Cholesky-scaled values `theta[k]^2 * sigma_sq`. These equal
    /// the random-effect variance components only for diagonal/scalar RE
    /// components (q=1 / scalar-extra — the currently reachable case); slope
    /// (q≥2) models are not yet validated through this field. Empty for OLS.
    pub tau2: Vec<f64>,
    pub converged: bool,
}

/// Options for `fit`.
pub struct FitOptions {
    /// Predictor column indices for which SE is computed.
    pub target_indices: Vec<u32>,
}

/// Thin friendly adapter: own all scratch, dispatch on `model.estimator`,
/// return a `Fit`.
///
/// # Panics
///
/// Panics only on engine invariant violations (e.g., `x.len() != n * p`).
/// All numerical failures (rank deficiency, optimiser failure) are signalled
/// via `Fit { converged: false, .. }` with NaN-filled estimates.
pub fn fit(x: &[f64], y: &[f64], n: usize, p: usize, model: &ModelSpec, opts: &FitOptions) -> Fit {
    assert_eq!(
        x.len(),
        n * p,
        "x must have n*p elements in row-major layout"
    );
    assert_eq!(y.len(), n, "y must have n elements");
    match model.estimator {
        Estimator::Ols => fit_ols(x, y, n, p, opts),
        Estimator::Mle => fit_mle(x, y, n, p, model, opts),
        Estimator::Glm => unimplemented!("Estimator::Glm is not yet wired in glmm::fit (both unclustered GLM and clustered GLMM are unimplemented); use the glmm::mcpower surface for now"),
    }
}

// ---------------------------------------------------------------------------
// OLS dispatch
// ---------------------------------------------------------------------------

fn fit_ols(x: &[f64], y: &[f64], n: usize, p: usize, opts: &FitOptions) -> Fit {
    let t = opts.target_indices.len();

    // --- scratch allocation (mirrors SimWorkspace field sizes in workspace.rs) ---
    let p1 = p.max(1); // guard zero-column degenerate call
    let mut fit_betas = vec![0.0f64; p1];
    let mut fit_var_diag = vec![0.0f64; t.max(1)];
    let mut fit_t_sq = vec![0.0f64; t.max(1)];
    let mut fit_u_scratch = vec![0.0f64; p1];
    let mut fit_factor = Mat::<f64>::zeros(p1, p1);
    let mut fit_rhs = Mat::<f64>::zeros(p1, 1);
    let mut suff_xtx = Mat::<f64>::zeros(p1, p1);
    let mut suff_xty = vec![0.0f64; p1];
    let mut suff_yty = 0.0f64;
    let mut suff_sum_y = 0.0f64;
    let mut suff_n_rows = 0usize;
    let mut suff_xtx_work = Mat::<f64>::zeros(p1, p1);
    // panel buffers: PANEL_ROWS * p1 is always sufficient (see PANEL_ROWS comment)
    let mut panel_x = vec![0.0f64; PANEL_ROWS * p1];
    let mut panel_y = vec![0.0f64; PANEL_ROWS];

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p1);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }

    // --- accumulate suff stats ---
    {
        let mut suff = OlsSuffStats {
            xtx: suff_xtx.as_mut(),
            xty: &mut suff_xty,
            yty: &mut suff_yty,
            sum_y: &mut suff_sum_y,
            n_rows: &mut suff_n_rows,
            panel_x: &mut panel_x,
            panel_y: &mut panel_y,
        };
        if n > 0 && p > 0 {
            suff.add_rows(x_mat.as_ref().subrows(0, n), y);
        }
    }

    // --- fit ---
    let view = {
        let scratch = OlsScratch {
            fit_betas: &mut fit_betas,
            fit_var_diag: &mut fit_var_diag,
            fit_t_sq: &mut fit_t_sq,
            fit_u_scratch: &mut fit_u_scratch,
            fit_factor: fit_factor.as_mut(),
            fit_rhs: fit_rhs.as_mut(),
        };
        crate::ols::fit_suff_stats_t_sq(
            suff_xtx.as_ref(),
            &suff_xty,
            suff_yty,
            suff_sum_y,
            suff_n_rows,
            &opts.target_indices,
            1e-12,
            suff_xtx_work.as_mut(),
            scratch,
        )
    };

    // --- map OlsFitView → Fit ---
    // view.betas is compact [0..p]; view.var_diag is compact [0..t] at target rank
    // (OLS/GLM are target-compact; LME/LMM are predictor-indexed — see batch.rs header)
    let beta = view.betas.to_vec();
    let converged = view.converged;
    let mut se = vec![f64::NAN; p];
    for (i, &ti) in opts.target_indices.iter().enumerate() {
        let vd = view.var_diag[i];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }

    Fit {
        beta,
        se,
        tau2: vec![],
        converged,
    }
}

// ---------------------------------------------------------------------------
// LMM dispatch (Estimator::Mle)
// ---------------------------------------------------------------------------

/// Produce level-0 cluster id for row `i` from the primary sizing.
fn primary_cluster_of_row(sizing: &Sizing, i: usize) -> u32 {
    sizing.cluster_of_row(i) as u32
}

/// Produce the local level id for extra grouping `g` at row `i`.
/// Verbatim logic mirrored from `test_support::extra_level_of_row` — that
/// helper is #[cfg(test)]-gated so it cannot be referenced here.
fn extra_level_of_row(model: &ModelSpec, g: usize, i: usize) -> u32 {
    let rel = &model.extra_groupings[g].relation;
    let level = match &model.sizing {
        Sizing::FixedClusters { n_clusters } => {
            let s = (*n_clusters).max(1) as usize;
            let mut stride = s;
            for h in &model.extra_groupings[..g] {
                stride *= block_levels(&h.relation);
            }
            let within = (i / stride) % block_levels(rel);
            match rel {
                GroupingRelation::Crossed { .. } => within,
                GroupingRelation::NestedWithin { n_per_parent } => {
                    (i % s) * (*n_per_parent).max(1) as usize + within
                }
            }
        }
        Sizing::FixedSize { cluster_size } => {
            let cs = (*cluster_size).max(1) as usize;
            let np = block_levels(rel);
            (i / cs) * np + (i % cs) % np
        }
    };
    level as u32
}

fn block_levels(rel: &GroupingRelation) -> usize {
    match rel {
        GroupingRelation::Crossed { n_clusters } => (*n_clusters).max(1) as usize,
        GroupingRelation::NestedWithin { n_per_parent } => (*n_per_parent).max(1) as usize,
    }
}

fn fit_mle(x: &[f64], y: &[f64], n: usize, p: usize, model: &ModelSpec, opts: &FitOptions) -> Fit {
    // slope_cols: x column indices for the primary RE slopes (empty = intercept-only)
    let slope_cols: Vec<usize> = model.slopes.iter().map(|s| s.column as usize).collect();

    // Build workspace — allocates solver, suff-stats, fit scratch for this model shape
    let mut ws = LmmWorkspace::for_cluster_spec(p, model, n, &slope_cols);

    // Build cluster and extra-grouping id vectors from the model layout
    let cluster_ids: Vec<u32> = (0..n)
        .map(|i| primary_cluster_of_row(&model.sizing, i))
        .collect();
    let extra_ids: Vec<Vec<u32>> = (0..model.extra_groupings.len())
        .map(|g| (0..n).map(|i| extra_level_of_row(model, g, i)).collect())
        .collect();

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let p1 = p.max(1);
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p1);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }

    // Accumulate sufficient statistics
    ws.suff.reset();
    if n > 0 && p > 0 {
        ws.suff
            .add_rows_multi(x_mat.as_ref().subrows(0, n), y, &cluster_ids, &extra_ids);
    }

    // Fit — use truth-start from the workspace (the DGP-derived hint, P1).
    // Copy theta_truth to a local buffer to avoid a borrow conflict with &mut ws.
    let theta_truth = ws.theta_truth.clone();
    let lmm_fit = fit_lmm(&mut ws, &opts.target_indices, Some(&theta_truth));

    // Map LmmFit + workspace state → Fit
    // ws.fit.betas: length p, all fixed effects
    // ws.fit.var_diag: length p, predictor-indexed (LME/LMM are predictor-indexed, unlike OLS)
    let beta = ws.fit.betas.clone();
    let sigma_sq = lmm_fit.sigma_sq;

    let mut se = vec![f64::NAN; p];
    for &ti in &opts.target_indices {
        let vd = ws.fit.var_diag[ti as usize];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }

    // tau2[k] = theta[k]^2 * sigma_sq — the k-th variance component in original scale.
    // ws.theta holds the fitted Cholesky parameters; diagonal entries satisfy
    // theta[k] = sqrt(tau_k / sigma_sq), so theta[k]^2 * sigma_sq = tau_k.
    let tau2: Vec<f64> = if lmm_fit.converged {
        ws.theta.iter().map(|&t| t * t * sigma_sq).collect()
    } else {
        ws.theta.iter().map(|_| f64::NAN).collect()
    };

    Fit {
        beta,
        se,
        tau2,
        converged: lmm_fit.converged,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Estimator, ModelSpec, Sizing, WaldSe};

    #[test]
    fn fit_ols_recovers_slope() {
        // y = 2*x + noise-free → beta[1] ≈ 2
        let n = 20;
        let p = 2;
        let x: Vec<f64> = (0..n).flat_map(|i| [1.0, i as f64]).collect(); // [intercept, x]
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
        let model = ModelSpec {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            tau_squared: 0.0,
            slopes: vec![],
            extra_groupings: vec![],
            estimator: Estimator::Ols,
            wald_se: WaldSe::Hessian,
        };
        let f = fit(
            &x,
            &y,
            n,
            p,
            &model,
            &FitOptions {
                target_indices: vec![1],
            },
        );
        assert!(f.converged);
        assert!((f.beta[1] - 2.0).abs() < 1e-6);
    }

    /// Deterministic pseudo-data (NR LCG), uniform in (−1, 1). Mirrors the
    /// LCG in lmm.rs tests so the smoke dataset behaves the same way.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }

    /// n=48, p=3, 6 clusters — same shape as lmm.rs's `hand_dataset`, adapted
    /// to the row-major f64 layout the friendly API expects.
    fn lmm_hand_dataset() -> (Vec<f64>, Vec<f64>, usize, usize) {
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
    fn fit_lmm_smoke() {
        let (x, y, n, p) = lmm_hand_dataset();
        let model = ModelSpec {
            sizing: Sizing::FixedClusters { n_clusters: 6 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![],
            estimator: Estimator::Mle,
            wald_se: WaldSe::Hessian,
        };
        let f = fit(
            &x,
            &y,
            n,
            p,
            &model,
            &FitOptions {
                target_indices: vec![1, 2],
            },
        );
        assert!(f.converged, "LMM should converge on clean clustered data");
        assert!(
            f.tau2[0].is_finite() && f.tau2[0] >= 0.0,
            "tau2[0] must be a finite non-negative variance, got {}",
            f.tau2[0]
        );
    }
}
