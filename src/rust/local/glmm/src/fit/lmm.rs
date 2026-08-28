//! LMM (`Family::Gaussian`, `re: Some`) dispatch — marshals `fit_warm`'s
//! inputs into the dense LMM workspace and calls `fit_lmm`. The numerical
//! kernel lives in `src/lmm/mod.rs`; this module only builds the workspace,
//! accumulates sufficient statistics, and maps `LmmFit` back to `Fit`.

use crate::lmm::{fit_lmm, LmmFit, LmmGroupings, LmmWorkspace};
// Used only by the test-only `fit_mle`/`fit_lmm_into` baselines. The stable core
// reaches the LMM path via `accumulate_lmm_rows`/`lmm_run_on`, which take neither.
#[cfg(test)]
use crate::{ModelSpec, StartValues};

use super::common::{
    assemble_varcorr, fill_se_by_predictor, nan_vcov, vcov_from_chol, FitDiagnostics,
};
// Used only by the test-only `fit_mle` baseline, which builds its own throwaway
// `Mat` — the stable core reaches the LMM path via `accumulate_lmm_rows`, which
// takes a caller-filled `x_mat` directly.
#[cfg(test)]
use super::common::{to_col_major, warm_theta};
use super::{Fit, FitOptions};

// ---------------------------------------------------------------------------
// LMM dispatch (Estimator::Mle)
// ---------------------------------------------------------------------------

/// Accumulates the θ-independent sufficient statistics (reset + add_rows_multi)
/// from an already-column-major `x_mat`. Single-sourced across `fit_mle` and
/// `with_lmm_objective`'s dense arm — both were byte-identical copies of this
/// block, differing only in `weights` and whether the guard was written out at
/// the call site. Takes `x_mat` rather than row-major `x` so the `fit_on` hot
/// path can fill a build-once `n_max`-sized buffer and pass a `subrows(0, n)`
/// view in, instead of allocating a fresh `Mat` every call; the three other
/// callers (`fit_mle`, `build_lmm_seam_ws`, `refit_lmm`) each build their own
/// throwaway `Mat` via `to_col_major` at the call site.
#[allow(clippy::too_many_arguments)] // marshals the kernel's (x_mat, y, n, p, ids…) surface
pub(super) fn accumulate_lmm_rows(
    ws: &mut LmmWorkspace,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    n: usize,
    p: usize,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    weights: Option<&[f64]>,
) {
    ws.suff.reset();
    // Before any row is accumulated: the RE design-column scales are per dataset, and
    // every slope value the accumulator stores is already divided by them. The
    // workspace is reused across draws with different `x`, so this is refreshed per
    // call, not cached at construction — and outside the `n > 0 && p > 0` guard,
    // because `SuffStats::reset` does not clear the scales and an empty draw would
    // otherwise leave the previous draw's in place.
    ws.suff.groupings.set_slope_scales(x_mat, weights);
    if n > 0 && p > 0 {
        ws.suff
            .add_rows_multi(x_mat, y, cluster_ids, extra_ids, weights);
    }
}

/// Borrowed result of [`lmm_run_on`]: the [`LmmFit`] summary plus the workspace
/// result slots the `Fit` assembly reads (β̂, per-predictor Var/z², the augmented
/// factor, θ̂, groupings, row count). Lifetime ties back to the [`LmmWorkspace`]
/// that owns the storage.
pub(crate) struct LmmResultView<'a> {
    fit: LmmFit,
    betas: &'a [f64],
    var_diag: &'a [f64],
    // Read only by the loop-tier `t_sq` accessor below (dead in a default build).
    #[allow(dead_code)]
    t_sq: &'a [f64],
    factor: faer::MatRef<'a, f64>,
    theta: &'a [f64],
    groupings: &'a LmmGroupings,
    n_rows: usize,
    /// Spherical conditional modes û at θ̂, or empty when the recovery did not
    /// run or did not succeed (see `LmmFitScratch::ranef_ok`).
    ranef_u: &'a [f64],
}

// Loop-tier read accessors (via the `FitView`/`loop_advanced` surface); the
// stable path reads these slots through `lmm_view_to_fit` instead.
#[allow(dead_code)]
impl LmmResultView<'_> {
    /// Per-target Wald z², predictor-indexed length p — only the
    /// `target_indices` slots are written; a non-target slot reads 0.0 on a
    /// fresh workspace or a previous fit's value on a reused one.
    pub(crate) fn t_sq(&self) -> &[f64] {
        self.t_sq
    }
    /// Fixed-effect estimates β̂, predictor-indexed.
    pub(crate) fn betas(&self) -> &[f64] {
        self.betas
    }
    /// Per-predictor Var(β̂_j), predictor-indexed length p — only the
    /// `target_indices` slots are written; a non-target slot reads 0.0 on a
    /// fresh workspace or a previous fit's value on a reused one.
    pub(crate) fn var_diag(&self) -> &[f64] {
        self.var_diag
    }
    /// This route's [`FitDiagnostics`] — every field is real here: the dense LMM
    /// is the one route that reports θ boundary state, per-component pins, AND a
    /// recorded pivot. The pivot floor is this route's own ([`crate::lmm::PIVOT_MIN`]),
    /// which sits at the same 1e-12 as OLS/GLM but was calibrated separately and
    /// stays a separate constant for that reason.
    pub(crate) fn diagnostics(&self) -> FitDiagnostics {
        FitDiagnostics {
            converged: self.fit.converged,
            boundary_hit: self.fit.boundary_hit,
            pinned_components: self.fit.pinned_components,
            pivot: self.fit.pivot,
            pivot_col: self.fit.pivot_col,
            ill_conditioned: self.fit.pivot < crate::lmm::PIVOT_MIN,
            // No PIRLS on the Gaussian LMM path (profiled REML, no IRLS inner
            // loop) — always 0/false.
            pirls_exhausted: 0,
            final_pirls_exhausted: false,
            // No FD-Hessian SE pass on the Gaussian LMM path (no PIRLS to
            // perturb) — always false.
            hessian_fallback: false,
        }
    }
    /// Joint Wald-χ² over the target set (the omnibus significance read).
    pub(crate) fn joint_t_sq(&self) -> f64 {
        self.fit.joint_t_sq
    }
    /// Objective evaluations the θ-solve spent.
    pub(crate) fn n_eval(&self) -> usize {
        self.fit.n_eval
    }
    /// Residual variance σ̂² (the LMM dispersion).
    pub(crate) fn dispersion(&self) -> f64 {
        self.fit.sigma_sq
    }
    /// Fitted θ̂ vech (primary block then extras, column-major lower-triangular),
    /// in the solver's INTERNAL RE column scale — `FitView::theta` divides it back
    /// into the design's own units for the warm-start carry.
    pub(crate) fn theta(&self) -> &[f64] {
        self.theta
    }
    /// Grouping structure, for the internal RE column scales θ̂ carries.
    pub(crate) fn groupings(&self) -> &LmmGroupings {
        self.groupings
    }
}

/// Runs the θ-solve on a pre-accumulated workspace and returns a borrowed view
/// of the results. Caller contract: `ws.suff` holds the accumulated rows
/// ([`accumulate_lmm_rows`]). Warm start threads `theta` only — the LMM β is
/// solved exactly given θ, so a β start is irrelevant; `None` (cold) uses the
/// kernel's THETA0 blind start.
pub(crate) fn lmm_run_on<'a>(
    ws: &'a mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
) -> LmmResultView<'a> {
    let fit = fit_lmm(ws, target_indices, theta_start);
    LmmResultView {
        ranef_u: if ws.fit.ranef_ok {
            &ws.fit.ranef_u[..ws.suff.groupings.k_total]
        } else {
            &[]
        },
        fit,
        betas: &ws.fit.betas,
        var_diag: &ws.fit.var_diag,
        t_sq: &ws.fit.t_sq,
        factor: ws.fit.factor.as_ref(),
        theta: &ws.theta,
        groupings: &ws.suff.groupings,
        n_rows: ws.suff.n_rows,
    }
}

/// Maps an [`LmmResultView`] to the full stable `Fit`: SE from `var_diag`, tau2
/// from θ̂, varcorr, vcov from the augmented factor, the REML criterion loglik,
/// the conditional modes + per-row means, and — when `opts.weights` is set — the
/// −Σlog wᵢ weighted-deviance correction.
///
/// `x`/`ids` are here for `fitted` alone (`Xβ̂ + Zb̂` needs the design rows and
/// the per-row levels), the same way `ols_view_to_fit` already takes `x`/`y`.
pub(crate) fn lmm_view_to_fit(
    view: &LmmResultView<'_>,
    x: &[f64],
    ids: &crate::GroupIds,
    n: usize,
    p: usize,
    opts: &FitOptions,
) -> Fit {
    let lmm_fit = &view.fit;
    let diag = view.diagnostics();
    // view.betas: length p, all fixed effects.
    // view.var_diag: length p, predictor-indexed (LME/LMM are predictor-indexed, unlike OLS).
    let beta = view.betas.to_vec();
    let sigma_sq = lmm_fit.sigma_sq;

    let mut se = vec![f64::NAN; p];
    fill_se_by_predictor(view.var_diag, &opts.target_indices, &mut se);

    // tau2[k] = theta[k]^2 * sigma_sq — the k-th variance component in original scale.
    // θ̂ holds the fitted Cholesky parameters; diagonal entries satisfy
    // theta[k] = sqrt(tau_k / sigma_sq), so theta[k]^2 * sigma_sq = tau_k.
    // Gated on the finite endpoint (`deviance`), not `converged` — the plateau
    // policy: a `MaxFunReached` cap-out reports an honest θ̂ (its finite
    // endpoint) with `converged == false` rather than NaN-filling.
    let has_endpoint = lmm_fit.deviance.is_finite();
    // θ̂ is in the solver's internal RE units (`LmmGroupings::set_slope_scales`);
    // `theta_row_scales` divides each entry back into the design's own units
    // before it is squared, so `tau2` is a variance in the user's scale.
    let theta_scales = view.groupings.theta_row_scales();
    let tau2: Vec<f64> = if has_endpoint {
        view.theta
            .iter()
            .zip(theta_scales.iter())
            .map(|(&t, &s)| (t / s) * (t / s) * sigma_sq)
            .collect()
    } else {
        view.theta.iter().map(|_| f64::NAN).collect()
    };

    // varcorr[g] = vech(σ̂²·Λ̂_gΛ̂_g') per grouping — the q≥2-valid
    // covariance; equals tau2's diagonal only at the (0,0) primary entry.
    let varcorr = if has_endpoint {
        assemble_varcorr(view.theta, view.groupings, sigma_sq)
    } else {
        vec![]
    };

    // Var(β̂) = σ̂²·(X'V⁻¹X)⁻¹ from the top-left p×p block of the augmented
    // factor — the same `L_XX` and the same σ̂² the `var_diag` forward solve
    // above uses, so `vcov`'s diagonal is `var_diag`. Gated on the finite
    // endpoint like `tau2`/`varcorr`: the degenerate return NaN-fills
    // `var_diag` and `sigma_sq` together, so `se` and `vcov` go NaN together.
    let vcov = if has_endpoint {
        vcov_from_chol(view.factor, p, &opts.target_indices, sigma_sq)
    } else {
        nan_vcov(p)
    };

    // Equal to `n` on every real path; diverges only in the degenerate
    // n > 0 && p == 0 case, where `accumulate_lmm_rows` skips `add_rows_multi`
    // (and so never increments `n_rows`) while `n` still reflects the caller's
    // row count.
    let n_rows = view.n_rows;
    let n_theta = view.theta.len();

    // Level counts depend only on the design, so they are reported on every
    // fit; a header line ("Number of obs: …, groups: …") prints on non-converged
    // fits too. `fitted`/`ranef` stay gated on `converged`: a conditional mode
    // at a non-converged θ̂ is not a BLUP of anything (the same rule the GLMM
    // paths apply). `ranef_u` is empty when the recovery did not run.
    let ranef_levels = super::common::ranef_level_counts(view.groupings);
    let (fitted, ranef) = if lmm_fit.converged && !view.ranef_u.is_empty() {
        let ranef = super::common::assemble_ranef_sparse(view.theta, view.groupings, view.ranef_u);
        let fitted = super::common::lmm_fitted(
            x,
            n,
            p,
            view.betas,
            &ranef,
            view.groupings,
            &ids.primary,
            &ids.extra,
            opts.offset.as_deref(),
        );
        (fitted, ranef)
    } else {
        (vec![], vec![])
    };

    let mut fit = Fit {
        beta,
        se,
        vcov,
        tau2,
        // REML σ̂² (NaN-filled by the kernel alongside var_diag on the
        // degenerate path, so this stays honest without an endpoint gate).
        dispersion: sigma_sq,
        diagnostics: super::common::materialize_diagnostics(&diag, p, &varcorr),
        varcorr,
        stddev_se: vec![], // LMM has no Hessian SE machinery
        n_eval: lmm_fit.n_eval,
        deviance: lmm_fit.deviance,
        // REML criterion on the logLik scale, from the base (unweighted) deviance;
        // the weighted correction below rewrites both together (using `n`, not
        // `n_rows` — see note above `n_rows`).
        loglik: super::common::lmm_loglik(lmm_fit.deviance, n_rows, p),
        df: if has_endpoint { p + n_theta + 1 } else { 0 },
        reml: true,
        fitted,
        ranef,
        ranef_levels,
    };
    fit.diagnostics.singular = fit.diagnostics.singular
        || fit.has_negligible_component(&super::common::re_scale_grid(view.groupings));

    // Weighted Gaussian log-density carries +½Σlog wᵢ per row; on the −2ℓ scale
    // the REML deviance gains −Σlog wᵢ (θ-independent — added post-optimization,
    // argmin unchanged), and the criterion-scale loglik is recomputed from the
    // corrected deviance. Matches lme4's weighted REMLcrit up to the additive
    // constant the engine strips from its deviance convention (documented on
    // `lmm::reml_deviance`). This is the single site every caller reaches, so
    // no caller applies the correction itself.
    if let Some(w) = &opts.weights {
        fit.deviance -= w.iter().map(|v| v.ln()).sum::<f64>();
        fit.loglik = super::common::lmm_loglik(fit.deviance, n, p);
    }
    fit
}

/// θ-solve + `Fit` assembly from a pre-built, already-accumulated workspace.
/// Composes [`lmm_run_on`] + [`lmm_view_to_fit`]. Caller contract: `ws.suff`
/// holds the accumulated rows (`accumulate_lmm_rows`, reset + add_rows_multi per
/// dataset). The weighted-deviance correction is inside `lmm_view_to_fit`, so
/// callers must NOT re-apply it. Only the test-gated `fit_mle` and `refit_lmm`
/// baselines compose through this; the core calls `lmm_run_on`/`lmm_view_to_fit`
/// directly.
#[cfg(test)]
pub(super) fn fit_lmm_into(
    ws: &mut LmmWorkspace,
    x: &[f64],
    ids: &crate::GroupIds,
    n: usize,
    p: usize,
    opts: &FitOptions,
    start: Option<&StartValues>,
) -> Fit {
    let view = lmm_run_on(ws, &opts.target_indices, warm_theta(start));
    lmm_view_to_fit(&view, x, ids, n, p, opts)
}
/// LMM dispatch adapter. `cluster_ids`/`extra_ids` are the per-row level ids from
/// the entry's [`GroupIds`]; `model` is the sizing-corrected spec (counts derived
/// from those ids). Slope x-columns come from `model.re`. Mirrors `fit_glmm`.
/// Test-only baseline since the stable path dispatches through the unified core
/// ([`super::core::fit_on`]) over `accumulate_lmm_rows`/`lmm_run_on`.
#[cfg(test)]
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
    // Identity-link offset is an exact y-shift before accumulation (the suff
    // stats are the only place raw y enters) — mirrors `fit_ols` / the sparse
    // `fit_mle_sparse`; change together.
    let y_shifted: Vec<f64>;
    let y_eff: &[f64] = match &opts.offset {
        Some(o) => {
            y_shifted = y.iter().zip(o).map(|(&yi, &oi)| yi - oi).collect();
            &y_shifted
        }
        None => y,
    };
    let x_mat = to_col_major(x, n, p);
    accumulate_lmm_rows(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
        y_eff,
        n,
        p,
        cluster_ids,
        extra_ids,
        opts.weights.as_deref(),
    );

    let ids = crate::GroupIds {
        primary: cluster_ids.to_vec(),
        extra: extra_ids.to_vec(),
    };
    fit_lmm_into(&mut ws, x, &ids, n, p, opts, start)
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
