//! LMM (`Family::Gaussian`, `re: Some`) dispatch — marshals `fit_warm`'s
//! inputs into the LMM workspace and calls `fit_lmm`. The numerical kernels
//! live in `src/lmm/mod.rs` and `src/sparse/mod.rs`; this module only builds
//! the workspace, accumulates the design, and maps `LmmFit` back to `Fit`.

use crate::inference::Objective;
use crate::lmm::{fit_lmm, Criterion, LmmFit, LmmGroupings, LmmKernel, LmmSuffStats, LmmWorkspace};
use crate::standardize::ColumnScaling;
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
use super::common::{as_given_design, start_to_design, warm_theta};
use super::{DesignSummary, Fit, FitOptions};

// ---------------------------------------------------------------------------
// LMM dispatch (Estimator::Mle)
// ---------------------------------------------------------------------------

/// Accumulates the θ-independent state of the design into whichever kernel the
/// workspace holds: the dense sufficient statistics (reset + add_rows_multi),
/// or a freshly built sparse-Z Gram workspace. Takes `x_mat` rather than row-major `x` so the `fit_on` hot
/// path can fill a build-once `n_max`-sized buffer and pass a `subrows(0, n)`
/// view in, instead of allocating a fresh `Mat` every call; the test-only
/// `fit_mle` baseline builds its own throwaway `Mat` via `as_given_design`.
/// `criterion` is the workspace's frozen one: a sparse kernel built here
/// takes it (the dense scratch has carried it since the build).
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
    same_design: bool,
    criterion: Criterion,
) {
    // The θ-search runs on ŵ = w/s (s = `family::weight_scale`), the same
    // precision-weight renormalisation the Gamma dispersion path applies
    // (`fit::glmm::prep_glmm_design`): it keeps θ's internal box (`PIN_THETA`/
    // `THETA_HI`) meaningful regardless of the raw scale of `w`, since ŵ's
    // scale is pinned near 1 by construction. `weight_scale` returns exactly
    // `1.0` for `None`, and `wᵢ/1.0` is exact in IEEE-754, so an unweighted
    // fit takes this path bit-identical to skipping it.
    // `LmmResultView::dispersion` and `lmm_view_to_fit` undo the substitution
    // on the one reported quantity that actually moves with the weight scale,
    // σ̂² (times `ws.weight_scale`); β̂, SEs, vcov, τ² and varcorr come out
    // already correct because the internal θ̂ and σ̂² carry compensating
    // factors of √s and 1/s that cancel when they are combined
    // (θ̂² · σ̂² = τ², σ̂² · (X'ŴX)⁻¹ = Var(β̂), …).
    let s = crate::family::weight_scale(weights, n);
    ws.weight_scale = s;
    // ŵ goes into the workspace's buffer, refilled in place: a weighted draw
    // allocates here only while the buffer grows to the largest `n` it sees.
    let w_for_rows: Option<&[f64]> = match weights {
        Some(w) => {
            ws.w_scaled.clear();
            ws.w_scaled.extend(w[..n].iter().map(|&wi| wi / s));
            Some(ws.w_scaled.as_slice())
        }
        None => None,
    };

    // The kernel carries the criterion the workspace froze: set at build on the
    // dense kernel, taken from `criterion` below when the sparse one is built.
    debug_assert!(
        match &ws.kernel {
            LmmKernel::Dense { fit, .. } => fit.criterion == criterion,
            LmmKernel::Sparse { ws: sw, .. } => {
                sw.as_ref().is_none_or(|sw| sw.criterion == criterion)
            }
        },
        "accumulate_lmm_rows: the kernel's criterion is not the frozen {criterion:?}"
    );
    match &mut ws.kernel {
        // Fixed-design opt-in: the design part of the statistics is the last
        // call's (same x, ids, weights), only the response part is rebuilt.
        LmmKernel::Dense { suff, .. } if same_design && n > 0 && p > 0 => {
            suff.update_response(x_mat, y, cluster_ids, extra_ids, w_for_rows);
        }
        LmmKernel::Dense { suff, .. } => {
            suff.reset();
            if n > 0 && p > 0 {
                // The fixed-effect factor U (`lmm::DesignQr`): on a
                // well-conditioned design read off the Gram of this one pass
                // and mapped onto what it accumulated
                // (`orthogonalize_from_gram`); otherwise the Givens pass, which
                // must run before any row is accumulated, and a second
                // accumulation through it.
                // The sparse arm computes its U inside its constructor.
                suff.add_rows_multi(x_mat, y, cluster_ids, extra_ids, w_for_rows);
                if !suff.orthogonalize_from_gram() {
                    suff.reset();
                    suff.set_design_qr(x_mat, w_for_rows);
                    suff.add_rows_multi(x_mat, y, cluster_ids, extra_ids, w_for_rows);
                }
            }
        }
        LmmKernel::Sparse { g, ws: sparse_ws } => {
            // WLS-style √ŵᵢ pre-scaling, the same convention `add_rows_multi`
            // applies on the dense arm: threaded through every z-emission and
            // raw x/y read in the constructor, so every packed Gram carries
            // exactly `ŵᵢ` per row. Taken in place: nothing on this kernel
            // reads ŵ itself after the constructor.
            if weights.is_some() {
                for v in ws.w_scaled.iter_mut() {
                    *v = v.sqrt();
                }
            }
            let sqrt_w = weights.map(|_| ws.w_scaled.as_slice());
            // The Gram scatter IS the constructor on this kernel: one pass over
            // the rows builds `Z'Z`, `Z'[X y]` and `[X y]'[X y]` and sizes every
            // per-eval buffer. A draw on the rows the workspace was built from
            // has the same pattern, so it refills the values in place.
            match sparse_ws {
                Some(sw) if crate::sparse::same_rows(&sw.ids, cluster_ids, extra_ids, n) => {
                    sw.fill_values(x_mat, cluster_ids, extra_ids, y, sqrt_w);
                }
                _ => {
                    *sparse_ws = Some(Box::new(crate::sparse::SparseLmmWorkspace::new(
                        g,
                        x_mat,
                        cluster_ids,
                        extra_ids,
                        y,
                        n,
                        p,
                        sqrt_w,
                        criterion,
                    )));
                }
            }
        }
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
    /// run or did not succeed.
    ranef_u: &'a [f64],
    /// The workspace's `weight_scale` s: `fit.sigma_sq` is σ̂²/s.
    weight_scale: f64,
    /// The criterion the kernel minimized. `Fit::reml`, the standardization
    /// shift and the log-likelihood's `ν` follow it, not `FitOptions::reml`,
    /// so a fit cannot report one criterion and have minimized the other.
    criterion: Criterion,
    /// The column standardization the rows were accumulated under
    /// (`β = B·β̃`): `factor` is on the standardized design, so `vcov` is read
    /// through it, and so are θ̂, `varcorr` and the conditional modes.
    map: &'a ColumnScaling,
    /// The dense kernel's accumulated statistics, which `lmm_view_to_fit`
    /// copies the design summary out of; `None` on the sparse kernel.
    suff: Option<&'a LmmSuffStats>,
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
    /// This route's [`FitDiagnostics`] — every field is real here: θ boundary
    /// state, per-component pins and a recorded pivot, on both of its kernels.
    /// The pivot floor is [`crate::lmm::PIVOT_MIN`], shared with the GLMM,
    /// which sits at the same 1e-12 as OLS/GLM but was calibrated separately
    /// and stays a separate constant for that reason.
    pub(crate) fn diagnostics(&self) -> FitDiagnostics {
        FitDiagnostics {
            converged: self.fit.converged,
            boundary_hit: self.fit.boundary_hit,
            pinned_components: self.fit.pinned_components,
            pivot: self.fit.pivot,
            pivot_col: self.fit.pivot_col,
            ill_conditioned: self.fit.pivot < crate::lmm::PIVOT_MIN,
            // No PIRLS on the Gaussian LMM path (profiled REML or ML, no IRLS
            // inner loop): always 0/false.
            pirls_exhausted: 0,
            final_pirls_exhausted: false,
            // No FD-Hessian SE pass on the Gaussian LMM path (no PIRLS to
            // perturb) — always false.
            hessian_fallback: false,
            // No ExactProfile route on the Gaussian LMM path (GLMM-only) —
            // always false.
            exact_profile_fallback: false,
            // The LMM's SEs come off the factor its criterion is evaluated on,
            // so an endpoint always has them.
            se_unavailable: false,
            // The LMM's profiled criterion is scale-free in the response.
            precision_limited: false,
            // No IRLS on the Gaussian LMM path.
            separated: false,
            gradient_search_fallback: self.fit.gradient_search_fallback,
            n_grad: self.fit.grad_report.map_or(0, |r| r.1 as u32),
            outer_stop: self.fit.grad_report.map(|r| r.0.into()),
            outer_decrement: self.fit.grad_report.map_or(f64::NAN, |r| r.2),
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
    /// Residual variance σ̂² (the LMM dispersion) on the caller's weight scale.
    /// Mirrors `lmm_view_to_fit`'s `fit.dispersion *= s` — change together.
    pub(crate) fn dispersion(&self) -> f64 {
        self.fit.sigma_sq * self.weight_scale
    }
    /// Fitted θ̂ vech (primary block then extras, column-major lower-triangular),
    /// in the kernel's coordinates: the internal RE column scale on the
    /// standardized design. `FitView::theta` maps it to the caller's units for
    /// the warm-start carry (`LmmGroupings::theta_kernel_to_caller`).
    pub(crate) fn theta(&self) -> &[f64] {
        self.theta
    }
    /// Grouping structure, for the random-effect maps θ̂ is read through.
    pub(crate) fn groupings(&self) -> &LmmGroupings {
        self.groupings
    }
}

/// Runs the θ-solve on a pre-accumulated workspace and returns a borrowed view
/// of the results. Caller contract: `ws.suff` holds the accumulated rows
/// ([`accumulate_lmm_rows`]), on the design `map` standardized. Warm start
/// threads `theta` only — the LMM β is solved exactly given θ, so a β start is
/// irrelevant; `None` (cold) uses the kernel's THETA0 blind start.
pub(crate) fn lmm_run_on<'a>(
    ws: &'a mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    hint: Option<&crate::optim::CurvatureHint>,
    map: &'a ColumnScaling,
) -> LmmResultView<'a> {
    let fit = fit_lmm(ws, target_indices, theta_start, hint, map);

    LmmResultView {
        weight_scale: ws.weight_scale,
        criterion: ws.kernel.criterion(),
        map,
        ranef_u: ws.kernel.ranef_u(),
        fit,
        betas: &ws.recovery.betas,
        var_diag: &ws.recovery.var_diag,
        t_sq: &ws.recovery.t_sq,
        factor: ws.kernel.factor(),
        theta: &ws.theta,
        groupings: ws.kernel.groupings(),
        n_rows: ws.kernel.n_rows(),
        suff: match &ws.kernel {
            LmmKernel::Dense { suff, .. } => Some(&**suff),
            LmmKernel::Sparse { .. } => None,
        },
    }
}

/// The design summary of [`Fit::design_summary`], copied out of the dense
/// accumulator: `counts` and the `x` rows of `s` over the `n_clusters` levels
/// seen (both are sized for the workspace's level capacity, which can be
/// larger), and the leading `p × p` block of `c` with its upper triangle
/// mirrored from the lower one the accumulator fills.
fn design_summary(suff: &LmmSuffStats) -> DesignSummary {
    let (p, k) = (suff.m - 1, suff.n_clusters);
    DesignSummary {
        counts: suff.counts[..k].to_vec(),
        s: (0..k)
            .flat_map(|a| suff.s.col_as_slice(a)[..p].iter().copied())
            .collect(),
        c: (0..p)
            .flat_map(|j| (0..p).map(move |i| suff.c[(i.max(j), i.min(j))]))
            .collect(),
    }
}

/// Maps an [`LmmResultView`] to the full stable `Fit`: SE from `var_diag`, tau2
/// from θ̂, varcorr, vcov from the augmented factor, the deviance and loglik of
/// the kernel's criterion (REML or ML), the conditional modes + per-row means,
/// and, when `opts.weights` is set, the −Σlog wᵢ weighted-deviance correction.
/// On an unweighted fit whose only random term is one random intercept and
/// which has a fixed-effect column, it also copies the design summary out of
/// the accumulator.
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
    // θ̂ in the caller's coordinates (slot order): the column standardization
    // mapped out (`LmmGroupings::theta_kernel_to_caller`), so `tau2` is a
    // variance in the user's scale.
    let (theta_caller, caller_pins) =
        view.groupings
            .theta_to_caller_vec(view.theta, diag.converged, view.map);
    let tau2: Vec<f64> = if has_endpoint {
        theta_caller.iter().map(|&t| t * t * sigma_sq).collect()
    } else {
        view.theta.iter().map(|_| f64::NAN).collect()
    };

    // varcorr[g] = vech(σ̂²·Λ̂_gΛ̂_g') per grouping — the q≥2-valid
    // covariance; equals tau2's diagonal only at the (0,0) primary entry.
    let varcorr = if has_endpoint {
        assemble_varcorr(view.theta, view.groupings, sigma_sq, view.map)
    } else {
        vec![]
    };

    // Var(β̂) = σ̂²·B(X̃'V⁻¹X̃)⁻¹Bᵀ from the top-left p×p block of the
    // augmented factor, read through the column map B — the same `L_XX`, the
    // same σ̂² and the same right-hand sides the `var_diag` forward solve uses,
    // so `vcov`'s diagonal is `var_diag`. Gated on the finite
    // endpoint like `tau2`/`varcorr`: the degenerate return NaN-fills
    // `var_diag` and `sigma_sq` together, so `se` and `vcov` go NaN together.
    let vcov = if has_endpoint {
        vcov_from_chol(view.factor, p, &opts.target_indices, sigma_sq, view.map)
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
        let ranef = super::common::assemble_ranef_sparse(
            view.theta,
            view.groupings,
            view.ranef_u,
            view.map,
        );
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

    // `pinned` in the caller's coordinates, where they differ from the kernel's.
    let mut diag = diag;
    if let Some(mask) = caller_pins {
        diag.pinned_components = mask;
    }
    let diagnostics = super::common::materialize_diagnostics(&diag, p, &varcorr);
    let criterion = view.criterion;
    // The kernel's deviance on the caller's design. The REML criterion contains
    // ln|X̃ᵀV⁻¹X̃| on the standardized design, which is the caller's ln|XᵀV⁻¹X|
    // minus 2·Σ ln sⱼ (a shift has determinant 1); the shift puts it back. The
    // ML criterion has no such term, and r² and |V| depend on X only through
    // its column space (and on the RE slope scaling through the exact Z̃Λ̃ = ZΛ
    // identity), so it needs none. The weighted correction below then applies
    // to the caller's value.
    let deviance = match criterion {
        Criterion::Reml => lmm_fit.deviance + view.map.reml_deviance_shift(),
        Criterion::Ml => lmm_fit.deviance,
    };
    let mut fit = Fit {
        beta,
        se,
        vcov,
        tau2,
        // The criterion's σ̂², r²/(n − p) under REML and r²/n under ML
        // (NaN-filled by the kernel alongside var_diag on the degenerate path,
        // so this stays honest without an endpoint gate).
        dispersion: sigma_sq,
        diagnostics,
        varcorr,
        stddev_se: vec![], // LMM has no Hessian SE machinery
        n_eval: lmm_fit.n_eval,
        #[cfg(feature = "counters")]
        counters: lmm_fit.counters,
        deviance,
        // The criterion on the logLik scale, from the base (unweighted)
        // deviance; the weighted correction below rewrites both together
        // (using `n`, not `n_rows`: see the note above `n_rows`).
        loglik: super::common::lmm_loglik(deviance, criterion.nu(n_rows, p)),
        df: if has_endpoint { p + n_theta + 1 } else { 0 },
        reml: criterion == Criterion::Reml,
        nobs: n,
        objective: match criterion {
            Criterion::Reml => Objective::Reml,
            Criterion::Ml => Objective::MaximumLikelihood,
        },
        fitted,
        ranef,
        ranef_levels,
        coef_df: vec![f64::NAN; p],
        // No pass has run: the stable path's `attach_satterthwaite` replaces
        // the record; a loop-tier `Fit` keeps it.
        satterthwaite: Some(crate::lmm::satterthwaite::SatterthwaiteState::not_run()),
        // The family is Gaussian on this path; the rest of the envelope is one
        // random intercept, no prior weights and at least one fixed-effect
        // column (with none, the accumulator holds nothing). A `(1 | g)`
        // design always routes to the dense kernel.
        design_summary: match view.suff {
            Some(suff)
                if suff.m > 1
                    && opts.weights.is_none()
                    && view.groupings.primary_q == 1
                    && view.groupings.nested.is_none()
                    && view.groupings.crossed.is_empty() =>
            {
                Some(design_summary(suff))
            }
            _ => None,
        },
    };
    // A capped (`MaxFunReached`) endpoint is reported as a point, not an
    // accepted optimum, even though `varcorr` is finite there — so the
    // post-hoc negligible-component check only runs on a converged fit
    // (mirrors `fit/glmm.rs`, same gate).
    fit.diagnostics.singular = fit.diagnostics.singular
        || (lmm_fit.converged
            && fit.has_negligible_component(&view.groupings.kernel_stddevs(view.theta)));

    // Weighted Gaussian log-density carries +½Σlog wᵢ per row; on the −2ℓ scale
    // the deviance gains −Σlog wᵢ on both criteria (θ-independent: added
    // post-optimization, argmin unchanged), and the loglik is recomputed from
    // the corrected deviance with the criterion's ν. Matches lme4's weighted
    // REMLcrit (REML) and deviance (ML) up to the additive constant the engine
    // strips from its deviance convention (documented on `lmm::reml_deviance`).
    // This is the single site every caller reaches, so no caller applies the
    // correction itself.
    //
    // The kernel was accumulated on ŵ = w/s (`accumulate_lmm_rows`), so its
    // deviance is already stripped of −Σlog ŵᵢ, not −Σlog wᵢ; the two differ
    // by n·ln(s), which exactly cancels the n·ln(s) that `s` introduces into
    // the kernel's criterion (its θ̂ carries √s, so ln|L_ZZ|² does not move).
    // Under ML the cancellation is direct: n·ln σ̂²_int = n·ln σ̂²_raw − n·ln s.
    // Under REML, (n − p)·ln σ̂²_int gives −(n − p)·ln s and ln|L_XX|² (the
    // relative X̃ᵀV⁻¹X̃, scaled by 1/s) the other −p·ln s. So no additional
    // correction is needed beyond using ŵ here; using raw `w` instead would
    // leave a spurious n·ln(s) in `deviance` and `loglik`.
    //
    // `fit.dispersion` (σ̂²_int off the ŵ-accumulated kernel) is
    // converted to the raw scale by the same `s` here and in
    // `LmmResultView::dispersion` (change together) — every other field above
    // (β, se, vcov, tau2, varcorr) is already correct on σ̂²_int, since their
    // formulas cancel the compensating factor of `s`.
    if let Some(w) = &opts.weights {
        let s = view.weight_scale;
        fit.dispersion *= s;
        fit.deviance -= w.iter().map(|v| (v / s).ln()).sum::<f64>();
        fit.loglik = super::common::lmm_loglik(fit.deviance, criterion.nu(n, p));
    }
    fit
}

/// θ-solve + `Fit` assembly from a pre-built, already-accumulated workspace.
/// Composes [`lmm_run_on`] + [`lmm_view_to_fit`]. Caller contract: `ws.suff`
/// holds the accumulated rows (`accumulate_lmm_rows`, reset + add_rows_multi per
/// dataset) of the design `map` scaled; `start` is in the caller's units. The
/// weighted-deviance correction is inside `lmm_view_to_fit`, so callers must
/// NOT re-apply it. Only the test-gated `fit_mle` baseline composes through
/// this; the core calls `lmm_run_on`/`lmm_view_to_fit` directly.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // marshals the kernel's (ws, x, ids, n, p, opts, start, map) surface
fn fit_lmm_into(
    ws: &mut LmmWorkspace,
    x: &[f64],
    ids: &crate::GroupIds,
    n: usize,
    p: usize,
    opts: &FitOptions,
    start: Option<&StartValues>,
    map: &ColumnScaling,
) -> Fit {
    let start = start.map(|s| {
        let mut s = s.clone();
        let g = ws.kernel.groupings();
        start_to_design(&mut s, map, g, &mut vec![0.0; g.re_map_scratch_len()]);
        s
    });
    let view = lmm_run_on(
        ws,
        &opts.target_indices,
        warm_theta(start.as_ref()),
        None,
        map,
    );
    lmm_view_to_fit(&view, x, ids, n, p, opts)
}
/// LMM dispatch adapter. `cluster_ids`/`extra_ids` are the per-row level ids from
/// the entry's [`GroupIds`]; `model` is the sizing-corrected spec (counts derived
/// from those ids). Slope x-columns come from `model.re`. Mirrors `fit_glmm`.
/// `sparse` picks the kernel outright instead of asking `classify_design`, which
/// is what makes this a forced baseline on either side of the router.
/// Test-only baseline since the stable path dispatches through the unified core
/// ([`super::core::fit_on`]) over `accumulate_lmm_rows`/`lmm_run_on`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // marshals the kernel's (x, y, n, p, spec, ids…) surface
fn fit_mle(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    start: Option<&StartValues>,
    opts: &FitOptions,
    sparse: bool,
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

    let criterion = super::common::resolve_criterion(model, opts)
        .expect("fit_mle requires a Gaussian mixed model");
    // Build workspace — allocates solver, suff-stats, fit scratch for this model shape
    let mut ws =
        LmmWorkspace::for_cluster_spec_ext(p, model, n, &slope_cols, &extra_slope_cols, sparse);
    // As `fit::core::build_workspace` sets it; the sparse kernel takes it from
    // `accumulate_lmm_rows`.
    if let LmmKernel::Dense { fit, .. } = &mut ws.kernel {
        fit.criterion = criterion;
    }
    // Identity-link offset is an exact y-shift before accumulation (the Grams
    // are the only place raw y enters) — the same convention `fit_ols_prebuilt` applies.
    // Mirrored by `fit::core`'s `Lmm` arm; change together.
    let y_shifted: Vec<f64>;
    let y_eff: &[f64] = match &opts.offset {
        Some(o) => {
            y_shifted = y.iter().zip(o).map(|(&yi, &oi)| yi - oi).collect();
            &y_shifted
        }
        None => y,
    };
    let (map, x_mat) = as_given_design(x, n, p, ws.kernel.groupings(), opts.weights.as_deref());
    accumulate_lmm_rows(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
        y_eff,
        n,
        p,
        cluster_ids,
        extra_ids,
        opts.weights.as_deref(),
        false,
        criterion,
    );

    let ids = crate::GroupIds {
        primary: cluster_ids.to_vec(),
        extra: extra_ids.to_vec(),
    };
    fit_lmm_into(&mut ws, x, &ids, n, p, opts, start, &map)
}
/// Test-only forced-NoZ entry: the NoZ↔Sparse cross-checks and timed sweeps
/// must reach the dense kernel regardless of where `classify_design`'s
/// performance boundary sits — going through `fit_cold` would compare Sparse
/// against itself on any cell the router sends to Sparse. Takes the
/// sizing-corrected spec, like `fit_mle`.
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
    fit_mle(
        x,
        y,
        n,
        p,
        sized,
        cluster_ids,
        extra_ids,
        start,
        opts,
        false,
    )
}

/// Test-only forced-sparse entry, the twin of [`fit_mle_noz_pub`]: the sparse
/// kernel is a superset of the dense one, so the cross-checks drive it on
/// in-envelope designs the router would otherwise send to the dense kernel.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // marshals the same surface as fit_mle
pub(crate) fn fit_mle_sparse_pub(
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
    fit_mle(x, y, n, p, sized, cluster_ids, extra_ids, start, opts, true)
}
