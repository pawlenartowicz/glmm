//! GLMM (Binomial/Poisson/Gamma/negative-binomial, `re: Some`) dispatch —
//! estimator dispatch, the numerical kernel lives in `src/glmm/`. Builds the
//! `GlmmWorkspace`/RE design `Z`/crossed-Schur symbolic factor, cold-starts β
//! from the no-RE GLM fit (`glm_warm_start_beta`), and maps `GlmmFit` +
//! workspace state back to `Fit`. `fit_glmm_nb` runs the NB **marginal-θ**
//! outer search (`lme4::glmer.nb`) on top of the fixed-θ `fit_glmm`/`fit_glmm_build`
//! pair.

use faer::Mat;

use crate::glm::{glm_irls_fit, GlmScratch};
use crate::glmm::{build_z, GlmmWorkspace, StructuredSchur};
use crate::{Family, ModelSpec, NegBinomialLink, StartValues};

use super::common::{assemble_varcorr, fill_se_by_predictor, nan_vcov, to_col_major};
use super::glm::{golden_max_ln_theta, nb_profile_loglik};
use super::{Fit, FitOptions};

// ---------------------------------------------------------------------------
// GLMM dispatch (Binomial{Logit}, re: Some)
// ---------------------------------------------------------------------------

/// Clustered-logistic GLMM dispatch adapter. Mirrors `fit_mle`: build the GLMM
/// workspace for this model shape, convert the row-major input to a column-major
/// faer `Mat`, build the dense RE design `Z` for the supplied ids, run the kernel
/// (workspace θ truth-start, β cold-start 0), and map `GlmmFit` → `Fit`.
///
/// `tau2[k] = θ̂[k]²` (no σ²; binomial residual scale is 1), mirroring `fit_mle`'s
/// `θ̂[k]²·σ̂²` map — so it carries the **same** `Fit::tau2` caveat: it equals the
/// RE variance component only for diagonal/scalar components (q=1 / scalar-extra);
/// slope (q≥2) models are not yet validated through this field.
/// Returns the mapped `Fit`, the converged conditional means `μ̂` (length `n`, from
/// `ws.prob` after the pinned-γ̂ re-eval), and the minimized marginal Laplace
/// deviance — the NB GLMM marginal-θ loop needs the deviance (μ̂ is now unused by
/// it but kept for any conditional-mean caller); other callers take `.0`.
/// Cold-start β for a GLMM fit: the coefficients of the fixed-effects-only GLM
/// (no random effects), matching lme4/glmer's initialization. Starting the joint
/// [θ|β] BOBYQA (and its inner PIRLS) from η ≈ Xβ̂_glm — the mean already explained
/// by the fixed effects — instead of β = 0 keeps the first PIRLS step small.
/// From β = 0 the linear predictor is η = Zu, so on an observation-level design the
/// conditional modes must absorb the entire mean in one Fisher step and can
/// overshoot into a weight regime (μ = exp(η) ~ 1e30 for Poisson-log) where the
/// structured crossed-Schur factor loses positive-definiteness and the deviance
/// aborts to `inf` (the grouseticks 3-crossed degenerate fit). Falls back to β = 0
/// if the GLM does not converge to finite coefficients. Only the cold path pays this
/// solve; a warm start (the MCPower hot loop) supplies β and never calls this.
///
/// Always calls the kernel with `prior_w: None`, even when the caller's `opts`
/// carries weights: this only seeds β for the GLMM optimizers, and the accept
/// rule + |Δdeviance| fixpoint make the seed irrelevant to the converged
/// answer — only the path to it shortens.
pub(crate) fn glm_warm_start_beta(
    family: Family,
    nb_theta: f64,
    x: faer::MatRef<f64>,
    y: &[f64],
    n: usize,
    p: usize,
    offset: Option<&[f64]>,
) -> Vec<f64> {
    let (n1, p1) = (n.max(1), p.max(1));
    let mut irls_eta = vec![0.0f64; n1];
    let mut irls_p = vec![0.0f64; n1];
    let mut irls_w = vec![0.0f64; n1];
    let mut irls_z = vec![0.0f64; n1];
    let mut irls_betas = vec![0.0f64; p1];
    let mut irls_betas_new = vec![0.0f64; p1];
    let mut irls_u_scratch = vec![0.0f64; p1];
    let mut irls_xtwx = Mat::<f64>::zeros(p1, p1);
    let mut irls_xtwz = vec![0.0f64; p1];
    let mut irls_l = Mat::<f64>::zeros(p1, p1);
    let mut irls_wx = vec![0.0f64; n1 * p1];
    // No target SEs are needed for a seed — only β — so target_indices is empty and
    // the var_diag / t_sq slots stay zero-length.
    let mut irls_var_diag: Vec<f64> = vec![];
    let mut irls_t_sq: Vec<f64> = vec![];
    let view = glm_irls_fit(
        family,
        nb_theta,
        x,
        y,
        &[],
        None,
        None,
        offset,
        GlmScratch {
            irls_eta: &mut irls_eta,
            irls_p: &mut irls_p,
            irls_w: &mut irls_w,
            irls_z: &mut irls_z,
            irls_betas: &mut irls_betas,
            irls_betas_new: &mut irls_betas_new,
            irls_var_diag: &mut irls_var_diag,
            irls_t_sq: &mut irls_t_sq,
            irls_u_scratch: &mut irls_u_scratch,
            irls_xtwx: irls_xtwx.as_mut(),
            irls_xtwz: &mut irls_xtwz,
            irls_l: irls_l.as_mut(),
            irls_wx: &mut irls_wx,
        },
    );
    if view.converged && view.betas.iter().all(|b| b.is_finite()) {
        view.betas.to_vec()
    } else {
        vec![0.0f64; p]
    }
}

/// θ-invariant build half of [`fit_glmm`]: allocates the workspace for this
/// (spec, n) shape, copies the θ-independent options (`parallel_inner`, prior
/// weights), converts `x` to column-major, and populates the RE design `Z` and
/// the crossed-Schur symbolic factor — none of which depend on `nb_theta`.
/// Returns the prebuilt `(ws, x_mat)` for [`fit_glmm_prebuilt`], or (on the
/// degenerate n=0/p=0 short-circuit) `Err` carrying the same NaN `Fit` triple
/// the public path returns. Hoisted so the NB marginal-θ search
/// ([`fit_glmm_nb`]) builds it ONCE and re-solves per θ instead of rebuilding
/// `Z` + the symbolic factor every golden-section eval.
/// θ-invariant build state returned by [`fit_glmm_build`]: the sized workspace
/// and the column-major `X`, both reusable across NB marginal-θ evals.
type BuiltGlmm = (GlmmWorkspace, Mat<f64>);

fn fit_glmm_build(
    x: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    opts: &FitOptions,
) -> Result<BuiltGlmm, Box<(Fit, Vec<f64>, f64)>> {
    let re = model
        .re
        .as_ref()
        .expect("fit_glmm requires a mixed model (re: Some)");
    // slope_cols: x column indices for the primary RE slopes (empty = intercept-only).
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();

    // Workspace for this (spec, n) shape — sizes per-cluster solver buffers off
    // re.sizing's cluster count; the kernels cold-start θ from their blind θ₀.
    let mut ws = GlmmWorkspace::for_cluster_spec(p, model, n, &slope_cols, opts.nagq);
    ws.parallel_inner = opts.parallel_inner;
    if let Some(w) = &opts.weights {
        ws.prior_w[..n].copy_from_slice(w);
        ws.weighted = true;
    }
    ws.offset = opts.offset.clone();

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let x_mat = to_col_major(x, n, p);

    // Degenerate guard (mirrors the kernel's n≤p short-circuit contract).
    if n == 0 || p == 0 {
        return Err(Box::new((
            Fit {
                beta: vec![f64::NAN; p],
                se: vec![f64::NAN; p],
                vcov: nan_vcov(p),
                tau2: vec![f64::NAN; ws.n_theta],
                dispersion: f64::NAN,
                converged: false,
                varcorr: vec![],
                stddev_se: vec![],
                aliased: vec![false; p],
                n_eval: 0,
                deviance: f64::NAN,
                singular: false,
                loglik: f64::NAN,
                df: 0,
                reml: false,
                fitted: vec![],
                ranef: vec![],
                ranef_levels: vec![],
            },
            vec![],
            f64::INFINITY,
        )));
    }

    // Build the dense RE design Z for this (X, ids) before the fit reads it.
    build_z(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
        cluster_ids,
        extra_ids,
        n,
    );

    // Cache the crossed-Schur symbolic factor once per fit. Only the
    // structured crossed path with e > 0 uses it; every other shape leaves it None.
    ws.structured_schur = if ws.groupings.structured_extras_eligible() {
        StructuredSchur::new(&ws.groupings, cluster_ids, extra_ids, n)
    } else {
        None
    };

    Ok((ws, x_mat))
}

#[allow(clippy::too_many_arguments)] // marshals the kernel's (x, y, n, p, spec, ids…) surface
pub(super) fn fit_glmm(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    nb_theta: f64,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> (Fit, Vec<f64>, f64) {
    let (mut ws, x_mat) = match fit_glmm_build(x, n, p, model, cluster_ids, extra_ids, opts) {
        Ok(built) => built,
        Err(degenerate) => return *degenerate,
    };
    fit_glmm_prebuilt(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
        y,
        n,
        p,
        model,
        cluster_ids,
        nb_theta,
        start,
        opts,
    )
}

/// θ-dependent solve half of [`fit_glmm`]: sets the NB dispersion on the
/// prebuilt workspace, cold- or warm-seeds β, runs the GLMM kernel, and maps
/// `GlmmFit` + workspace → `Fit`. The kernel resets all per-fit warm-start state
/// (`params`, `u_seed`, `coup_mask`, `cluster_rows`, `theta_se`) at its top —
/// the workspace is designed for cross-fit reuse (see `glmm::fit_glmm`) — so
/// calling this repeatedly on one prebuilt `ws` (the NB marginal-θ search) is
/// bit-identical to a fresh construction per θ. `Z`, the symbolic factor, and
/// `x_mat` are θ-invariant reads; the numeric factorization the kernel writes
/// into `structured_schur` is recomputed every eval.
#[allow(clippy::too_many_arguments)]
fn fit_glmm_prebuilt(
    ws: &mut GlmmWorkspace,
    x_mat: faer::MatRef<f64>,
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    nb_theta: f64,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> (Fit, Vec<f64>, f64) {
    // NB θ̂ is threaded explicitly (the spec is θ-free); the PIRLS/AGQ variance and
    // deviance read it off the workspace. NaN for every non-NB family (unread).
    ws.nb_theta = nb_theta;
    let n_theta = ws.n_theta;

    // Warm start threads β + θ into the GLMM kernel. A caller-supplied `start` (the
    // MCPower hot loop) uses its β verbatim; a cold start seeds β from the no-RE GLM
    // fit (lme4/glmer initialization — see `glm_warm_start_beta`) instead of 0, so
    // the inner PIRLS opens near the mean and does not overshoot. θ still cold-starts
    // at the kernel's THETA0 blind start.
    let beta_start = match start {
        Some(s) => s.beta.clone(),
        None => glm_warm_start_beta(
            model.family,
            nb_theta,
            x_mat,
            y,
            n,
            p,
            opts.offset.as_deref(),
        ),
    };
    let glmm_fit = crate::glmm::fit_glmm(
        ws,
        x_mat,
        y,
        cluster_ids,
        &opts.target_indices,
        start.map(|s| s.theta.as_slice()),
        &beta_start,
        n,
        opts.wald_se,
    );

    // Map GlmmFit + workspace state → Fit.
    // ws.betas: length p, all fixed effects; ws.var_diag: predictor-indexed.
    let beta = ws.betas.clone();
    let mut se = vec![f64::NAN; p];
    fill_se_by_predictor(&ws.var_diag, &opts.target_indices, &mut se);

    // tau2[k] = σ²·θ̂[k]². lme4 parametrizes the RE covariance as σ²·θθ', so VarCorr
    // reports sd = σ·θ̂; our internal λ̂ = ws.params[..n_theta] IS that relative factor
    // θ̂ (the Laplace penalty is the unit ‖u‖²). For binomial/Poisson/NB the residual
    // scale σ²≡1, but Gamma's σ² = pwrss/n = (Pearson χ² + ‖û‖²)/n ≠ 1, so its
    // variance components carry it. (Distinct from `dispersion` below — that is the
    // Pearson/(n−p) moment lme4 reports separately, a different quantity.) Same
    // q≥2-slope caveat as fit_mle's tau2.
    // σ̂² = pwrss/n (family::glmm_sigma_sq; exactly 1.0 for the φ≡1 families),
    // hoisted so tau2 and varcorr below carry the SAME scale — lme4's VarCorr
    // convention. Only meaningful on a converged fit (reads the converged
    // μ̂/û state).
    let sigma_sq = if glmm_fit.converged {
        crate::family::glmm_sigma_sq(
            model.family,
            &y[..n],
            &ws.prob[..n],
            &ws.u[..ws.k],
            ws.weighted.then(|| &ws.prior_w[..n]),
        )
    } else {
        f64::NAN
    };
    let tau2: Vec<f64> = if glmm_fit.converged {
        ws.params[..n_theta]
            .iter()
            .map(|&t| t * t * sigma_sq)
            .collect()
    } else {
        vec![f64::NAN; n_theta]
    };

    // Dispersion. Binomial/Poisson hold φ≡1. Gamma recovers the (possibly
    // weighted) Pearson moment estimator on the conditional-mode residuals
    // (μ̂ = ws.prob after the pinned-γ̂ re-eval): `φ̂ = Σ wᵢrᵢ²/(n−p)`,
    // `rᵢ = (yᵢ−μ̂ᵢ)/√V(μ̂ᵢ)` (raw `n−p` df, not `Σwᵢ−p`). It does NOT rescale the
    // SE here — the kernel already reports each arm on lme4's convention: Hessian
    // unscaled (`vcov(use.hessian=TRUE)`, oracle-settled) and Rx carrying σ̂² =
    // pwrss/n (`vcov(use.hessian=FALSE)`; `family::glmm_sigma_sq`, a DIFFERENT
    // quantity than this φ̂). NB θ̂ is set by the outer-θ wrapper, not here.
    let dispersion = match model.family {
        Family::Gamma { .. } if glmm_fit.converged => match opts.dispersion {
            Some(v) => v,
            None => crate::family::pearson_dispersion(
                &y[..n],
                &ws.prob[..n],
                model.family,
                nb_theta,
                n,
                p,
                Some(&ws.prior_w[..n]),
            ),
        },
        _ => 1.0,
    };

    // GLMM D̂ = σ̂²·Λ̂Λ̂' — the same σ̂² that scales tau2 above, so the two
    // accessors report the one variance component on one scale (lme4 VarCorr;
    // σ̂² ≡ 1 for binomial/Poisson/NB, so this only bites dispersion families
    // like Gamma). Oracle: `fit_glmm_gamma_sim_matches_lme4` /
    // `parity/goldens/sim_gamma_glmm.json` varcomp stddevs.
    let varcorr = if glmm_fit.converged {
        assemble_varcorr(&ws.params[..n_theta], &ws.groupings, sigma_sq)
    } else {
        vec![]
    };

    // SE of the RE stddevs from the joint-Hessian θ block (`WaldSe::Hessian` only;
    // NaN under Rx / RX fallback / non-converged — `ws.theta_se` is reset per fit
    // and refilled only by `fd_hessian_cov`). Cloned verbatim: for the reachable
    // scalar groupings θ = stddev, so the θ-scale SE is the stddev SE.
    let stddev_se = if glmm_fit.converged {
        ws.theta_se[..n_theta].to_vec()
    } else {
        vec![f64::NAN; n_theta]
    };

    // `ws.vcov` is filled at the same target indices as `ws.var_diag` by
    // whichever SE arm ran, and NaN elsewhere — so `Fit::vcov` is finite exactly
    // where `Fit::se` is, on both `Hessian` and `Rx`.
    let vcov: Vec<Vec<f64>> = (0..p)
        .map(|i| (0..p).map(|j| ws.vcov[(i, j)]).collect())
        .collect();

    let mu_hat = ws.prob[..n].to_vec();
    // Diagnostics off the converged workspace state: μ̂ (the same conditional
    // means the tuple returns), b̂ = Λ̂û from the spherical modes, and the
    // marginal log-likelihood with the saturated constant restored.
    let (fitted, ranef, ranef_levels) = if glmm_fit.converged {
        (
            mu_hat.clone(),
            super::common::assemble_ranef_dense(
                &ws.params[..n_theta],
                &ws.groupings,
                &ws.u[..ws.k],
            ),
            super::common::ranef_level_counts(&ws.groupings),
        )
    } else {
        (vec![], vec![], vec![])
    };
    let loglik = super::common::glmm_loglik(
        model.family,
        nb_theta,
        if glmm_fit.deviance.is_finite() {
            glmm_fit.deviance
        } else {
            f64::NAN
        },
        &y[..n],
        ws.weighted.then(|| &ws.prior_w[..n]),
    );
    let mut fit = Fit {
        beta,
        se,
        vcov,
        tau2,
        dispersion,
        converged: glmm_fit.converged,
        varcorr,
        stddev_se,
        aliased: vec![false; p],
        n_eval: glmm_fit.n_eval,
        deviance: if glmm_fit.deviance.is_finite() {
            glmm_fit.deviance
        } else {
            f64::NAN
        },
        singular: glmm_fit.boundary_hit == 1,
        loglik,
        df: if glmm_fit.converged {
            super::common::model_df(model.family, p, n_theta, opts.dispersion.is_some())
        } else {
            0
        },
        reml: false,
        fitted,
        ranef,
        ranef_levels,
    };
    fit.singular = fit.singular || fit.has_negligible_component();
    (fit, mu_hat, glmm_fit.deviance)
}

/// Negative-binomial GLMM via the **marginal-θ** profile (`lme4::glmer.nb`):
/// optimise the dispersion θ on the *marginal* (Laplace-integrated) likelihood,
/// not the conditional one. For each candidate θ the inner [`fit_glmm`] re-fits
/// the full GLMM (variance components + β) at that fixed θ and returns its
/// minimized marginal Laplace deviance `D(θ)`; the marginal log-likelihood is then
///
/// ```text
///   logL_marginal(θ) = −½·D(θ) + nb_profile_loglik(y, y, θ, weights)
/// ```
///
/// where the second term is the NB **saturated** log-likelihood (the θ-dependent
/// `Σᵢ wᵢ·[lnΓ(yᵢ+θ)−lnΓ(θ)]` normalisation the deviance cancels against its
/// saturated reference — see [`nb_profile_loglik`]'s derivation), `weights =
/// opts.weights` (`None` ⇒ unit weights, matching `D(θ)`'s own weighting since
/// both come from the same fit). Maximising this over `ln θ`
/// by [`golden_max_ln_theta`] reproduces `glmer.nb`'s outer `optimize()`, which
/// likewise re-fits the GLMM per θ. A non-converging inner fit returns
/// `D=∞ ⇒ logL=−∞`, so the maximiser rejects that θ.
///
/// The earlier conditional-μ̂ profile (optimise θ on `nb_profile_loglik(y, μ̂, θ)`
/// at the fitted conditional means) is biased by ~21% on the sim_nb oracle — it
/// treats the conditional modes as data and ignores both the RE-integration and
/// the curvature term's θ-dependence. `dispersion = θ̂`; the reported β/SE come
/// from a final fit at θ̂ (`theta_seed` is irrelevant to the global ln-θ bracket
/// search and unused).
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_glmm_nb(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    _start: Option<&StartValues>,
    opts: &FitOptions,
) -> Fit {
    // θ-free spec; θ̂ is threaded to fit_glmm explicitly per candidate. The NB
    // marginal-θ search is a global ln-θ bracket, so a warm `_start` is irrelevant
    // (matches the former unused `theta_seed`) — the inner fits cold-start.
    let nb_spec = ModelSpec {
        family: Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        re: model.re.clone(),
    };

    // Build the θ-invariant state (workspace, Z, symbolic factor, col-major x)
    // ONCE — every golden-section eval below re-solves on it at a new θ instead
    // of reconstructing it. Degenerate n=0/p=0 returns the NaN Fit directly.
    let (mut ws, x_mat) = match fit_glmm_build(x, n, p, &nb_spec, cluster_ids, extra_ids, opts) {
        Ok(built) => built,
        Err(degenerate) => return degenerate.0,
    };
    let x_ref = x_mat.as_ref().subrows(0, n);

    let theta = golden_max_ln_theta(|t| {
        let th = t.exp();
        let (_fit, _mu, dev) = fit_glmm_prebuilt(
            &mut ws,
            x_ref,
            y,
            n,
            p,
            &nb_spec,
            cluster_ids,
            th,
            None,
            opts,
        );
        // `dev` is already weighted (opts threads through fit_glmm → ws.prior_w,
        // 4c); the saturated-reference term takes the same per-row weights so
        // both halves of `logL_marginal` are on the same weighted scale.
        -0.5 * dev + nb_profile_loglik(y, y, th, opts.weights.as_deref())
    });

    let mut fit_result = fit_glmm_prebuilt(
        &mut ws,
        x_ref,
        y,
        n,
        p,
        &nb_spec,
        cluster_ids,
        theta,
        None,
        opts,
    )
    .0;
    fit_result.dispersion = theta;
    fit_result
}
