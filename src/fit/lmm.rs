//! LMM (`Family::Gaussian`, `re: Some`) dispatch — marshals `fit_warm`'s
//! inputs into the dense LMM workspace and calls `fit_lmm`. The numerical
//! kernel lives in `src/lmm.rs`; this module only builds the workspace,
//! accumulates sufficient statistics, and maps `LmmFit` back to `Fit`.

use crate::lmm::{fit_lmm, LmmWorkspace};
use crate::{ModelSpec, StartValues};

use super::common::{
    assemble_varcorr, fill_se_by_predictor, nan_vcov, to_col_major, vcov_from_chol,
};
use super::{Fit, FitOptions};

// ---------------------------------------------------------------------------
// LMM dispatch (Estimator::Mle)
// ---------------------------------------------------------------------------

/// Converts row-major `x` into the workspace's column-major buffer and
/// accumulates the θ-independent sufficient statistics (reset + add_rows_multi).
/// Single-sourced across `fit_mle` and `with_lmm_objective`'s dense arm — both
/// were byte-identical copies of this block, differing only in `weights` and
/// whether the guard was written out at the call site.
#[allow(clippy::too_many_arguments)] // marshals the kernel's (x, y, n, p, ids…) surface
pub(super) fn accumulate_lmm_rows(
    ws: &mut LmmWorkspace,
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    weights: Option<&[f64]>,
) {
    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let x_mat = to_col_major(x, n, p);

    ws.suff.reset();
    if n > 0 && p > 0 {
        ws.suff.add_rows_multi(
            x_mat.as_ref().subrows(0, n),
            y,
            cluster_ids,
            extra_ids,
            weights,
        );
    }
}

/// θ-solve + `Fit` assembly from a pre-built, already-accumulated workspace.
/// Caller contract: `ws.suff` holds the accumulated rows (`accumulate_lmm_rows`,
/// reset + add_rows_multi per dataset) — mirrors `fit_lmm`'s own contract.
pub(super) fn fit_lmm_into(
    ws: &mut LmmWorkspace,
    target_indices: &[u32],
    start: Option<&StartValues>,
) -> Fit {
    // Warm start threads `theta` only — the LMM β is solved exactly given θ, so a
    // β start is irrelevant (matches StartValues carrying no LMM β use). `None`
    // (cold) uses the kernel's THETA0 blind start.
    let lmm_fit = fit_lmm(ws, target_indices, start.map(|s| s.theta.as_slice()));

    // Map LmmFit + workspace state → Fit
    // ws.fit.betas: length p, all fixed effects
    // ws.fit.var_diag: length p, predictor-indexed (LME/LMM are predictor-indexed, unlike OLS)
    let beta = ws.fit.betas.clone();
    let p = beta.len();
    let sigma_sq = lmm_fit.sigma_sq;

    let mut se = vec![f64::NAN; p];
    fill_se_by_predictor(&ws.fit.var_diag, target_indices, &mut se);

    // tau2[k] = theta[k]^2 * sigma_sq — the k-th variance component in original scale.
    // ws.theta holds the fitted Cholesky parameters; diagonal entries satisfy
    // theta[k] = sqrt(tau_k / sigma_sq), so theta[k]^2 * sigma_sq = tau_k.
    // Gated on the finite endpoint (`deviance`), not `converged` — the plateau
    // policy: a `MaxFunReached` cap-out reports an honest θ̂ (its finite
    // endpoint) with `converged == false` rather than NaN-filling.
    let has_endpoint = lmm_fit.deviance.is_finite();
    let tau2: Vec<f64> = if has_endpoint {
        ws.theta.iter().map(|&t| t * t * sigma_sq).collect()
    } else {
        ws.theta.iter().map(|_| f64::NAN).collect()
    };

    // varcorr[g] = vech(σ̂²·Λ̂_gΛ̂_g') per grouping — the q≥2-valid
    // covariance; equals tau2's diagonal only at the (0,0) primary entry.
    let varcorr = if has_endpoint {
        assemble_varcorr(&ws.theta, &ws.suff.groupings, sigma_sq)
    } else {
        vec![]
    };

    // Var(β̂) = σ̂²·(X'V⁻¹X)⁻¹ from the top-left p×p block of the augmented
    // factor — the same `L_XX` and the same σ̂² the `var_diag` forward solve
    // above uses, so `vcov`'s diagonal is `var_diag`. Gated on the finite
    // endpoint like `tau2`/`varcorr`: the degenerate return NaN-fills
    // `var_diag` and `sigma_sq` together, so `se` and `vcov` go NaN together.
    let vcov = if has_endpoint {
        vcov_from_chol(ws.fit.factor.as_ref(), p, target_indices, sigma_sq)
    } else {
        nan_vcov(p)
    };

    let mut fit = Fit {
        beta,
        se,
        vcov,
        tau2,
        // REML σ̂² (NaN-filled by the kernel alongside var_diag on the
        // degenerate path, so this stays honest without an endpoint gate).
        dispersion: sigma_sq,
        converged: lmm_fit.converged,
        varcorr,
        stddev_se: vec![], // LMM has no Hessian SE machinery
        aliased: vec![false; p],
        n_eval: lmm_fit.n_eval,
        deviance: lmm_fit.deviance,
        singular: lmm_fit.boundary_hit == 1,
    };
    fit.singular = fit.singular || fit.has_negligible_component();
    fit
}
/// LMM dispatch adapter. `cluster_ids`/`extra_ids` are the per-row level ids from
/// the entry's [`GroupIds`]; `model` is the sizing-corrected spec (counts derived
/// from those ids). Slope x-columns come from `model.re`. Mirrors `fit_glmm`.
#[allow(clippy::too_many_arguments)] // marshals the kernel's (x, y, n, p, spec, ids…) surface
pub(super) fn fit_mle(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Fit {
    let re = model
        .re
        .as_ref()
        .expect("fit_mle requires a mixed model (re: Some)");
    // slope_cols: x column indices for the primary RE slopes (empty = intercept-only)
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    // Extra-grouping slope x-columns, declaration order. On the standalone path the
    // ModelSpec's slope columns ARE x-matrix indices (unlike MCPower, which resolves
    // them separately), so they are read directly here.
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();

    // Build workspace — allocates solver, suff-stats, fit scratch for this model shape
    let mut ws = LmmWorkspace::for_cluster_spec_ext(p, model, n, &slope_cols, &extra_slope_cols);
    accumulate_lmm_rows(
        &mut ws,
        x,
        y,
        n,
        p,
        cluster_ids,
        extra_ids,
        opts.weights.as_deref(),
    );

    let mut fit = fit_lmm_into(&mut ws, &opts.target_indices, start);

    // Weighted Gaussian log-density carries +½Σlog wᵢ per row; on the −2ℓ
    // scale the REML/ML deviance gains −Σlog wᵢ (θ-independent — added
    // post-optimization, argmin unchanged). Matches lme4's weighted REMLcrit
    // up to the engine's documented stripped constant (see lme.rs:2978).
    // Applied here rather than inside `fit_lmm_into` — it needs the caller's raw
    // weights slice, which the workspace doesn't retain past accumulation.
    if let Some(w) = &opts.weights {
        fit.deviance -= w.iter().map(|v| v.ln()).sum::<f64>();
    }
    fit
}
/// Test-only forced-NoZ entry (mirror of forcing `fit_mle_sparse` directly):
/// the NoZ↔Sparse cross-checks and timed sweeps must reach the dense kernel
/// regardless of where `classify_design`'s performance boundary sits — going
/// through `fit_cold` would compare Sparse against itself on any cell the
/// router sends to Sparse. Takes the sizing-corrected spec, like `fit_mle`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // marshals the same surface as fit_mle
pub(crate) fn fit_mle_noz_pub(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    sized: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Fit {
    fit_mle(x, y, n, p, sized, cluster_ids, extra_ids, start, opts)
}
