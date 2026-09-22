//! GLMM (Binomial/Poisson/Gamma/negative-binomial, `re: Some`) dispatch —
//! estimator dispatch, the numerical kernel lives in `src/glmm/`. Builds the
//! `GlmmWorkspace`/RE design `Z`/crossed-Schur symbolic factor, cold-starts β
//! from the no-RE GLM fit (`glm_warm_start_beta`), and maps `GlmmFit` +
//! workspace state back to `Fit`. `fit_glmm_nb` searches the NB dispersion
//! `θ_NB` as one more coordinate of the kernel's own outer BOBYQA
//! (`glmm::fit_glmm`), so one call fits β, θ_RE and θ_NB together.

use faer::Mat;

use crate::glm::{glm_irls_fit, GlmScratch};
use crate::glmm::{fill_packed_cols, GlmmFit, GlmmLayout, GlmmWorkspace, StructuredSchur};
use crate::{Family, ModelSpec, NegBinomialLink, StartValues};

use super::common::{
    assemble_varcorr, fill_se_by_predictor, nan_vcov, to_col_major, warm_theta, FitDiagnostics,
};
use super::{Diagnostics, Fit, FitOptions};

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
/// `ws.pirls.prob` after the pinned-γ̂ re-eval), and the minimized marginal Laplace
/// deviance; callers take `.0`, the deviance rides along for the tests that
/// compare routes at fixed θ.
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

/// All-NaN, non-converged `Fit` for a degenerate GLMM shape (`n <= p` or
/// `p == 0`), the shapes the OLS, GLM and LMM kernels refuse too: the fixed
/// effects are not estimable, so every estimate is NaN. Shared by
/// [`build_on_workspace`]'s build-time guard (the NB route, which rebuilds its
/// workspace per call) and the unified core's `FitKind::Glmm` arm (the other
/// families, which build the workspace once per shape and so need their own
/// per-call check at the same width).
pub(super) fn degenerate_glmm_fit(p: usize, n_theta: usize) -> Fit {
    Fit {
        beta: vec![f64::NAN; p],
        se: vec![f64::NAN; p],
        vcov: nan_vcov(p),
        tau2: vec![f64::NAN; n_theta],
        dispersion: f64::NAN,
        diagnostics: Diagnostics::from_flags(false, false, p),
        varcorr: vec![],
        stddev_se: vec![],
        n_eval: 0,
        #[cfg(feature = "counters")]
        counters: crate::counters::EvalCounters::new(),
        deviance: f64::NAN,
        loglik: f64::NAN,
        df: 0,
        reml: false,
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
    }
}

/// θ-invariant build half of [`fit_glmm`]: allocates the workspace for this
/// (spec, n) shape, copies the θ-independent options (`parallel_inner`, prior
/// weights), converts `x` to column-major, and populates the RE design `Z` and
/// the crossed-Schur symbolic factor — none of which depend on `nb_theta`.
/// Returns the prebuilt `(ws, x_mat)` for [`fit_glmm_prebuilt`], or (on the
/// degenerate n=0/p=0 short-circuit) `Err` carrying the same NaN `Fit` triple
/// the public path returns. Split from [`fit_glmm_prebuilt`] so a caller with
/// its own workspace policy can build once and solve on it; the stable path
/// composes both.
/// θ-invariant build state returned by [`fit_glmm_build`]: the sized workspace
/// and the column-major `X`.
type BuiltGlmm = (GlmmWorkspace, Mat<f64>);

pub(super) fn fit_glmm_build(
    x: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    opts: &FitOptions,
) -> Result<BuiltGlmm, Box<(Fit, Vec<f64>, f64)>> {
    let (slope_cols, extra_slope_cols) = re_slope_cols(model);
    // Workspace for this (spec, n) shape — sizes per-cluster solver buffers off
    // re.sizing's cluster count; the kernels cold-start θ from their blind θ₀.
    let ws =
        GlmmWorkspace::for_cluster_spec_ext(p, model, n, &slope_cols, &extra_slope_cols, opts.nagq);
    build_on_workspace(ws, x, n, p, cluster_ids, extra_ids, opts)
}

/// `slope_cols`: x column indices for the primary RE slopes (empty =
/// intercept-only). `extra_slope_cols`: the same per extra grouping, in
/// declaration order — read by the packed-row GLMM layout, which applies a full
/// `q_g×q_g` Λ block per extra level, and by the sparse LMM; every other
/// layout's groupings come out identical either way.
pub(super) fn re_slope_cols(model: &ModelSpec) -> (Vec<usize>, Vec<Vec<usize>>) {
    let re = model
        .re
        .as_ref()
        .expect("a mixed model (re: Some) is required here");
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();
    (slope_cols, extra_slope_cols)
}

/// The design-dependent tail of [`fit_glmm_build`], on an already-allocated
/// workspace: the θ-independent options, the column-major `X`, the RE column
/// scales, the packed `M` columns and the crossed-Schur symbolic factors.
#[allow(clippy::too_many_arguments)]
fn build_on_workspace(
    mut ws: GlmmWorkspace,
    x: &[f64],
    n: usize,
    p: usize,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    opts: &FitOptions,
) -> Result<BuiltGlmm, Box<(Fit, Vec<f64>, f64)>> {
    // Degenerate guard (mirrors the kernel's n≤p short-circuit contract).
    if n <= p || p == 0 {
        return Err(Box::new((
            degenerate_glmm_fit(p, ws.n_theta),
            vec![],
            f64::INFINITY,
        )));
    }

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let x_mat = to_col_major(x, n, p);
    prep_glmm_design(&mut ws, x_mat.as_ref(), cluster_ids, extra_ids, n, opts);
    Ok((ws, x_mat))
}

/// The per-call, design-dependent workspace prep: the call-varying options, the
/// RE column scales for this design, the packed `M` columns and the two
/// crossed-Schur symbolic factors. Shared by [`build_on_workspace`] (a fresh
/// workspace per call) and the unified core's `FitKind::Glmm` arm (one
/// workspace reused across draws, hence the `weighted` reset). `x` is the
/// column-major design with at least `n` rows.
pub(super) fn prep_glmm_design(
    ws: &mut GlmmWorkspace,
    x: faer::MatRef<f64>,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
    opts: &FitOptions,
) {
    ws.parallel_inner = opts.parallel_inner;
    if let Some(w) = &opts.weights {
        ws.prior_w[..n].copy_from_slice(w);
        ws.weighted = true;
    } else {
        ws.weighted = false;
    }
    ws.offset = opts.offset.clone();

    // Before the RE design is read: every slope column enters it divided by its
    // internal scale, so the scales have to be current for THIS design first. Per
    // call, not cached on the workspace shape — the same workspace is reused
    // across draws.
    ws.groupings
        .set_slope_scales(x.subrows(0, n), opts.weights.as_deref());

    // Fill the packed M columns for this (X, ids) before the fit reads them.
    fill_packed_cols(ws, cluster_ids, extra_ids, n);

    // Cache the crossed-Schur symbolic factor once per fit. Only the
    // structured layout reads it; every other shape leaves it None.
    ws.pattern.structured_schur = if ws.layout == GlmmLayout::Structured {
        StructuredSchur::new(&ws.groupings, cluster_ids, extra_ids, n)
    } else {
        None
    };
    // Observed twin of the crossed-Schur symbolic factor, so the exact β-profile's
    // adjoint solve can run on `A_obs` without overwriting the Fisher factor every
    // later pass reads. Built only where the exact profile can read it: a
    // non-canonical link on the structured layout — canonical links never read
    // it, so building it there would be pure cost on every warm-path draw.
    ws.exact_prof.obs_schur =
        if ws.layout == GlmmLayout::Structured && !crate::family::is_canonical(ws.family) {
            StructuredSchur::new(&ws.groupings, cluster_ids, extra_ids, n)
        } else {
            None
        };
}

/// Test-only baseline (fixed-θ dense GLMM as a single call). The stable path
/// dispatches through the unified core ([`super::core::fit_on`]) over
/// `run_glmm_on`/`glmm_view_to_fit`; the NB route (`fit_glmm_nb`) composes
/// `fit_glmm_build`/`fit_glmm_prebuilt` directly.
#[cfg(test)]
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
        extra_ids,
        nb_theta,
        start,
        opts,
    )
}

/// Test-only: the same fit on a workspace forced onto the packed-row layout,
/// whatever layout [`crate::glmm::GlmmLayout::for_design`] would pick for this
/// design. Lets an in-envelope design be fit both ways so the packed kernel can
/// be checked against the blocked and structured ones.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glmm_packed(
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
    let (slope_cols, extra_slope_cols) = re_slope_cols(model);
    let groupings =
        crate::lmm::LmmGroupings::from_cluster_spec_ext(model, n, &slope_cols, &extra_slope_cols);
    let ws =
        GlmmWorkspace::from_groupings(groupings, model.family, p, n, opts.nagq, GlmmLayout::Packed);
    let (mut ws, x_mat) = match build_on_workspace(ws, x, n, p, cluster_ids, extra_ids, opts) {
        Ok(built) => built,
        Err(degenerate) => return *degenerate,
    };
    // NB seeds its `ln θ_NB` coordinate from the no-RE GLM-NB's own θ̂, exactly
    // as [`fit_glmm_nb`] does, so the two routes start the same search.
    let nb_theta = match model.family {
        Family::NegativeBinomial { .. } => super::glm::fit_glm_nb(x, y, n, p, None, opts).1,
        _ => nb_theta,
    };
    fit_glmm_prebuilt(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
        y,
        n,
        p,
        model,
        cluster_ids,
        extra_ids,
        nb_theta,
        start,
        opts,
    )
}

/// Borrowed result of [`run_glmm_on`]: the [`GlmmFit`] summary, the θ̂ it was fit
/// at, and a shared borrow of the whole solved [`GlmmWorkspace`] (the assembly
/// reads a dozen of its result slots — β̂, Var, μ̂, û, θ̂, θ_se, vcov, groupings —
/// so borrowing the workspace whole is simpler than enumerating each). Lifetime
/// ties back to the workspace that owns the storage.
pub(crate) struct GlmmResultView<'a> {
    fit: GlmmFit,
    nb_theta: f64,
    ws: &'a GlmmWorkspace,
}

// Loop-tier read accessors (via the `FitView`/`loop_advanced` surface); the
// stable path reads the workspace slots through `glmm_view_to_fit` instead.
#[allow(dead_code)]
impl GlmmResultView<'_> {
    /// Whether the search converged, and the minimized marginal deviance it
    /// reached — the two scalars a caller that only wants the workspace left
    /// at γ̂ needs, without mapping the whole view to a `Fit`.
    #[cfg(test)]
    pub(crate) fn converged_deviance(&self) -> (bool, f64) {
        (self.fit.converged, self.fit.deviance)
    }
    /// Per-target Wald statistic, predictor-indexed length p — only the
    /// `target_indices` slots are written; a non-target slot reads 0.0 on a
    /// fresh workspace or a previous fit's value on a reused one.
    pub(crate) fn t_sq(&self) -> &[f64] {
        &self.ws.inference.t_sq
    }
    /// Fixed-effect estimates β̂, predictor-indexed.
    pub(crate) fn betas(&self) -> &[f64] {
        &self.ws.betas
    }
    /// Per-predictor Var(β̂_j), predictor-indexed length p — only the
    /// `target_indices` slots are written; a non-target slot reads 0.0 on a
    /// fresh workspace or a previous fit's value on a reused one.
    pub(crate) fn var_diag(&self) -> &[f64] {
        &self.ws.inference.var_diag
    }
    /// This route's [`FitDiagnostics`]. θ boundary state and per-component pins
    /// are real; the pivot fields are NOT — the dense GLMM records no pivot, so
    /// they stay at the `fixed_only` NaN and this route never flags
    /// ill-conditioning. Detection here would need its own calibration on the
    /// PIRLS-weighted `X'WX`, which nobody has run.
    pub(crate) fn diagnostics(&self) -> FitDiagnostics {
        FitDiagnostics {
            boundary_hit: self.fit.boundary_hit,
            pinned_components: self.fit.pinned_components,
            pirls_exhausted: self.ws.pirls_exhausted,
            final_pirls_exhausted: self.ws.final_pirls_exhausted,
            hessian_fallback: self.fit.hessian_fallback,
            ..FitDiagnostics::fixed_only(self.fit.converged)
        }
    }
    /// Joint Wald-χ² over the target set (the omnibus significance read).
    pub(crate) fn joint_t_sq(&self) -> f64 {
        self.fit.joint_t_sq
    }
    /// Objective evaluations the joint solve spent.
    pub(crate) fn n_eval(&self) -> usize {
        self.fit.n_eval
    }
    /// Random-intercept variance D̂[0][0] (the GLMM dispersion read).
    pub(crate) fn dispersion(&self) -> f64 {
        self.fit.tau_squared_hat
    }
    /// Fitted θ̂ vech — the leading `n_theta` params of the joint [θ̂ | β̂] block,
    /// in the solver's INTERNAL RE column scale (`FitView::theta` divides it back).
    pub(crate) fn theta(&self) -> &[f64] {
        &self.ws.params[..self.ws.n_theta]
    }
    /// Grouping structure, for the internal RE column scales θ̂ carries.
    pub(crate) fn groupings(&self) -> &crate::lmm::LmmGroupings {
        &self.ws.groupings
    }
}

/// θ-dependent solve half of [`fit_glmm`]: sets the NB dispersion on the
/// prebuilt workspace, cold- or warm-seeds β, runs the GLMM kernel, and returns
/// the borrowed [`GlmmResultView`]. The kernel resets all per-fit warm-start
/// state (`params`, `u_seed`, `coup_mask`, `cluster_rows`) at its top and
/// `theta_se` further down, after the degenerate-fit guard — an early
/// `nan_fit` bail there (`glmm/mod.rs`) skips the `theta_se` reset and leaves
/// the previous fit's values in the workspace; harmless since `stddev_se` is
/// itself gated on `converged`. The workspace is designed for cross-fit reuse
/// (see `glmm::fit_glmm`), so calling this repeatedly on one prebuilt `ws` is
/// bit-identical to a fresh construction per call except for that one
/// stale-on-bail slot. `Z`, the symbolic factor, and `x_mat` are inputs the
/// caller fixes before calling; the numeric factorization the kernel writes
/// into `structured_schur` is recomputed every eval. [`glmm_view_to_fit`] maps
/// the returned view to `Fit`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_glmm_on<'a>(
    ws: &'a mut GlmmWorkspace,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    nb_theta: f64,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> GlmmResultView<'a> {
    // NB θ₀ (the start of the `ln θ_NB` coordinate) is threaded explicitly; the
    // kernel leaves θ̂_NB in `ws.nb_theta`. NaN for every non-NB family (unread).
    ws.nb_theta = nb_theta;

    // Warm start threads β + θ into the GLMM kernel. A caller-supplied `start` (the
    // MCPower hot loop) uses its β verbatim; a cold start seeds β from the no-RE GLM
    // fit (lme4/glmer initialization — see `glm_warm_start_beta`) instead of 0, so
    // the inner PIRLS opens near the mean and does not overshoot. θ still cold-starts
    // at the kernel's THETA0 blind start. An empty field is a per-component cold
    // start (`StartValues`), so it takes the same branch as `start = None`.
    let beta_start = match start {
        Some(s) if !s.beta.is_empty() => s.beta.clone(),
        _ => glm_warm_start_beta(
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
        extra_ids,
        &opts.target_indices,
        warm_theta(start),
        &beta_start,
        n,
        opts.wald_se,
    );
    // θ̂_NB after the search (NaN on every other family, untouched by the kernel).
    let nb_theta = ws.nb_theta;
    GlmmResultView {
        fit: glmm_fit,
        nb_theta,
        ws,
    }
}

/// Maps a [`GlmmResultView`] to the full stable `Fit`, the converged conditional
/// means `μ̂` (length `n`), and the minimized marginal Laplace deviance; callers
/// take `.0`, the deviance rides along for the tests that compare routes at
/// fixed θ. Needs raw `y` (σ̂²/dispersion/loglik read it) and `model` (family
/// selection); θ̂ comes from the view.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glmm_view_to_fit(
    view: &GlmmResultView<'_>,
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    opts: &FitOptions,
) -> (Fit, Vec<f64>, f64) {
    let ws = view.ws;
    let glmm_fit = &view.fit;
    let diag = view.diagnostics();
    let converged = diag.converged;
    let nb_theta = view.nb_theta;
    let n_theta = ws.n_theta;

    // Map GlmmFit + workspace state → Fit.
    // ws.betas: length p, all fixed effects; ws.inference.var_diag: predictor-indexed.
    let beta = ws.betas.clone();
    let mut se = vec![f64::NAN; p];
    fill_se_by_predictor(&ws.inference.var_diag, &opts.target_indices, &mut se);

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
    let sigma_sq = if converged {
        crate::family::glmm_sigma_sq(
            model.family,
            &y[..n],
            &ws.pirls.prob[..n],
            &ws.pirls.u[..ws.k],
            ws.weighted.then(|| &ws.prior_w[..n]),
        )
    } else {
        f64::NAN
    };
    // θ̂ is in the solver's internal RE units (`LmmGroupings::set_slope_scales`);
    // dividing by the Λ-row scales puts every θ-derived magnitude back into the
    // design's own units before it is squared.
    let theta_scales = ws.groupings.theta_row_scales();
    let tau2: Vec<f64> = if converged {
        ws.params[..n_theta]
            .iter()
            .zip(theta_scales.iter())
            .map(|(&t, &s)| (t / s) * (t / s) * sigma_sq)
            .collect()
    } else {
        vec![f64::NAN; n_theta]
    };

    // Dispersion. Binomial/Poisson hold φ≡1. Gamma recovers the (possibly
    // weighted) Pearson moment estimator on the conditional-mode residuals
    // (μ̂ = ws.pirls.prob after the pinned-γ̂ re-eval): `φ̂ = Σ wᵢrᵢ²/(n−p)`,
    // `rᵢ = (yᵢ−μ̂ᵢ)/√V(μ̂ᵢ)` (raw `n−p` df, not `Σwᵢ−p`). It does NOT rescale the
    // SE here — the kernel already reports each arm on lme4's convention: Hessian
    // unscaled (`vcov(use.hessian=TRUE)`, oracle-settled) and Rx carrying σ̂² =
    // pwrss/n (`vcov(use.hessian=FALSE)`; `family::glmm_sigma_sq`, a DIFFERENT
    // quantity than this φ̂).
    let dispersion = if !converged {
        f64::NAN
    } else {
        match model.family {
            Family::Gamma { .. } => match opts.dispersion {
                Some(v) => v,
                None => crate::family::pearson_dispersion(
                    &y[..n],
                    &ws.pirls.prob[..n],
                    model.family,
                    nb_theta,
                    n,
                    p,
                    Some(&ws.prior_w[..n]),
                ),
            },
            Family::NegativeBinomial { .. } => nb_theta,
            _ => 1.0,
        }
    };

    // GLMM D̂ = σ̂²·Λ̂Λ̂' — the same σ̂² that scales tau2 above, so the two
    // accessors report the one variance component on one scale (lme4 VarCorr;
    // σ̂² ≡ 1 for binomial/Poisson/NB, so this only bites dispersion families
    // like Gamma). Oracle: `fit_glmm_gamma_sim_matches_lme4` /
    // `validation/goldens/sim_gamma_glmm.json` varcomp stddevs.
    let varcorr = if converged {
        assemble_varcorr(&ws.params[..n_theta], &ws.groupings, sigma_sq)
    } else {
        vec![]
    };

    // SE of the RE stddevs from the joint-Hessian θ block (`WaldSe::Hessian` only;
    // NaN under Rx / RX fallback / non-converged — `ws.inference.theta_se` is reset per fit
    // and refilled only by `joint_hessian_cov`). For the reachable scalar groupings
    // θ = stddev, so the θ-scale SE is the stddev SE.
    //
    // The joint Hessian is taken in the INTERNAL θ̃ = s·θ on both arms (the FD
    // stencil perturbs it, the exact kernel differentiates w.r.t. it), so
    // `theta_se` is an SE on θ̃ — mirrors §Boundary handling in
    // `documentation/algorithms-glmm.md`, change together.
    // The map is a fixed diagonal linear reparametrization, so its Jacobian is the
    // constant `s` — the back-map is the same plain division θ̂ itself takes, with
    // no delta-method term.
    let stddev_se = if converged {
        ws.inference.theta_se[..n_theta]
            .iter()
            .zip(theta_scales.iter())
            .map(|(&se, &s)| se / s)
            .collect()
    } else {
        vec![f64::NAN; n_theta]
    };

    // `ws.inference.vcov` is filled at the same target indices as `ws.inference.var_diag` by
    // whichever SE arm ran, and NaN elsewhere — so `Fit::vcov` is finite exactly
    // where `Fit::se` is, on both `Hessian` and `Rx`.
    let vcov: Vec<Vec<f64>> = (0..p)
        .map(|i| (0..p).map(|j| ws.inference.vcov[(i, j)]).collect())
        .collect();

    let mu_hat = ws.pirls.prob[..n].to_vec();
    // Diagnostics off the converged workspace state: μ̂ (the same conditional
    // means the tuple returns) and b̂ = Λ̂û from the spherical modes. Level
    // counts are design-only and reported regardless — see `fit/lmm.rs`.
    let ranef_levels = super::common::ranef_level_counts(&ws.groupings);
    let (fitted, ranef) = if converged {
        // The two layouts order û's primary block differently: blocked and
        // structured are level-major (`lvl·q_p + c`), packed is component-major
        // (`c·n_primary + f`). Each assembler walks its own order.
        let u = &ws.pirls.u[..ws.k];
        let ranef = if ws.layout == crate::glmm::GlmmLayout::Packed {
            super::common::assemble_ranef_sparse(&ws.params[..n_theta], &ws.groupings, u)
        } else {
            super::common::assemble_ranef_dense(&ws.params[..n_theta], &ws.groupings, u)
        };
        (mu_hat.clone(), ranef)
    } else {
        (vec![], vec![])
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
    let diagnostics = super::common::materialize_diagnostics(&diag, p, &varcorr);
    let mut fit = Fit {
        beta,
        se,
        vcov,
        tau2,
        dispersion,
        diagnostics,
        varcorr,
        stddev_se,
        n_eval: glmm_fit.n_eval,
        #[cfg(feature = "counters")]
        counters: glmm_fit.counters,
        deviance: if glmm_fit.deviance.is_finite() {
            glmm_fit.deviance
        } else {
            f64::NAN
        },
        loglik,
        df: if converged {
            super::common::model_df(model.family, p, n_theta, opts.dispersion.is_some())
        } else {
            0
        },
        reml: false,
        fitted,
        ranef,
        ranef_levels,
    };
    fit.diagnostics.singular = fit.diagnostics.singular
        || fit.has_negligible_component(&super::common::re_scale_grid(&ws.groupings));
    (fit, mu_hat, glmm_fit.deviance)
}

/// θ-dependent solve + `Fit` assembly on a prebuilt workspace. Composes
/// [`run_glmm_on`] + [`glmm_view_to_fit`] — the same split the unified fit core
/// drives (`fit_on` calls `run_glmm_on`; `FitView::into_fit` calls
/// `glmm_view_to_fit`). Returns `(Fit, μ̂, deviance)`; callers take `.0`, the
/// deviance rides along for the tests that compare routes at fixed θ.
#[allow(clippy::too_many_arguments)]
fn fit_glmm_prebuilt(
    ws: &mut GlmmWorkspace,
    x_mat: faer::MatRef<f64>,
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
    let view = run_glmm_on(
        ws,
        x_mat,
        y,
        n,
        p,
        model,
        cluster_ids,
        extra_ids,
        nb_theta,
        start,
        opts,
    );
    glmm_view_to_fit(&view, y, n, p, model, opts)
}

/// Negative-binomial GLMM: the dispersion θ_NB is a coordinate of the outer
/// search on the marginal objective `dev − 2·nb_profile_loglik(y, y, θ_NB, w)`
/// (`glmm::fit_glmm`), so β, θ_RE and θ_NB come out of one fit — the same optimum
/// `lme4::glmer.nb`'s outer `optimize()` over re-fitted GLMMs reaches, without
/// the re-fits. The coordinate cold-starts at the no-RE GLM-NB's own θ̂ (one
/// extra fixed-effects-only `fit_glm_nb`, itself an IRLS/θ-profile alternation
/// capped at `NB_MAX_OUTER`); a caller's `start` seeds β/θ_RE as on every
/// other family (θ_NB has no start slot). `dispersion = θ̂_NB`; the β SE
/// conditions on θ̂ (lme4/MASS convention).
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_glmm_nb(
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
    let nb_spec = ModelSpec {
        family: Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        re: model.re.clone(),
    };
    let (mut ws, x_mat) = match fit_glmm_build(x, n, p, &nb_spec, cluster_ids, extra_ids, opts) {
        Ok(built) => built,
        Err(degenerate) => return degenerate.0,
    };
    // The method-of-moments seed (`nb_theta_moment_seed`) charges the random-effect
    // variance to the dispersion: on a GLMM it lands one to two orders of magnitude
    // below θ̂, and on random-slope shapes PIRLS never converges at those dispersions,
    // handing the outer BOBYQA a `+∞` plateau it can't escape. The no-RE GLM-NB's own
    // θ̂ is a start, not an estimate (the fixed effects alone under-explain the mean,
    // so it still moves under the outer search), but it starts inside the basin PIRLS
    // can actually converge in. Taken unguarded: on finite `y` the prefit always
    // reports a θ inside `[NB_THETA_LO, NB_THETA_HI]` (its own seed is clamped there
    // and the profile search never leaves the box), so there is nothing a fallback
    // could rescue. `fit_glm_nb` hands back that θ as its second return value —
    // the last θ its alternation stood on, the moment seed if the first inner IRLS
    // failed, the θ reached so far if a later one did — because the `Fit`'s own
    // `dispersion` field is NaN unless the prefit converged.
    let nb_seed = super::glm::fit_glm_nb(x, y, n, p, None, opts).1;
    fit_glmm_prebuilt(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
        y,
        n,
        p,
        &nb_spec,
        cluster_ids,
        extra_ids,
        nb_seed,
        start,
        opts,
    )
    .0
}
