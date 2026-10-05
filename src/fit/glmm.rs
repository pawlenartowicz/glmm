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
use crate::standardize::ColumnScaling;
use crate::{Family, ModelSpec, NegBinomialLink, StartValues};

use super::common::{
    assemble_varcorr, fill_se_by_predictor, nan_vcov, start_to_design, warm_theta, FitDiagnostics,
};
use super::{Diagnostics, Fit, FitOptions};

// ---------------------------------------------------------------------------
// GLMM dispatch (Binomial{Logit}, re: Some)
// ---------------------------------------------------------------------------

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
/// The flag is `GlmFitView::separated`: the GLM found no finite maximum, and β is
/// then its last iterate instead of the fallback, for the caller to report.
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
) -> (Vec<f64>, bool) {
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
    if (view.converged || view.separated) && view.betas.iter().all(|b| b.is_finite()) {
        (view.betas.to_vec(), view.separated)
    } else {
        (vec![0.0f64; p], false)
    }
}

/// All-NaN, non-converged `Fit` for a degenerate GLMM shape (`n <= p` or
/// `p == 0`), the shapes the OLS, GLM and LMM kernels refuse too: the fixed
/// effects are not estimable, so every estimate is NaN. Returned by the
/// per-call guards of [`fit_glmm_nb`] and the unified core's `FitKind::Glmm`
/// arm (both build their workspace once per shape, so the check runs per
/// call), and by the test-only [`build_on_workspace`].
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

/// θ-invariant build half: allocates the workspace for this (spec, n) shape
/// and runs [`build_on_workspace`]: the θ-independent options, the RE design
/// and the crossed-Schur symbolic factor,
/// none of which depend on `nb_theta`. On the degenerate n≤p/p=0
/// short-circuit returns `Err` carrying the same NaN `Fit` triple the public
/// path returns. Split from [`fit_glmm_prebuilt`] so a caller with its own
/// workspace policy can build once and solve on it. Test-only: every fit path
/// builds its workspace in the unified core (`fit::core::build_workspace`).
#[cfg(test)]
fn build_glmm(
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    opts: &FitOptions,
) -> Result<GlmmWorkspace, Box<(Fit, Vec<f64>, f64)>> {
    let (slope_cols, extra_slope_cols) = re_slope_cols(model);
    // Workspace for this (spec, n) shape — sizes per-cluster solver buffers off
    // re.sizing's cluster count; the kernels cold-start θ from their blind θ₀.
    let ws =
        GlmmWorkspace::for_cluster_spec_ext(p, model, n, &slope_cols, &extra_slope_cols, opts.nagq);
    build_on_workspace(ws, n, p, cluster_ids, extra_ids, opts)
}

#[cfg(test)]
type BuiltGlmm = (GlmmWorkspace, Mat<f64>);

/// Test-only: [`build_glmm`] on row-major `x`, returning the column-major copy
/// alongside the workspace.
#[cfg(test)]
pub(super) fn fit_glmm_build(
    x: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    opts: &FitOptions,
) -> Result<BuiltGlmm, Box<(Fit, Vec<f64>, f64)>> {
    let x_mat = super::common::to_col_major(x, n, p);
    let ws = build_glmm(n, p, model, cluster_ids, extra_ids, opts)?;
    Ok((ws, x_mat))
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

/// The design-dependent tail of [`build_glmm`], on an already-allocated
/// workspace: [`prep_glmm_design`], after the degenerate guard. Test-only.
#[cfg(test)]
fn build_on_workspace(
    mut ws: GlmmWorkspace,
    n: usize,
    p: usize,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    opts: &FitOptions,
) -> Result<GlmmWorkspace, Box<(Fit, Vec<f64>, f64)>> {
    // Degenerate guard (mirrors the kernel's n≤p short-circuit contract).
    if n <= p || p == 0 {
        return Err(Box::new((
            degenerate_glmm_fit(p, ws.n_theta),
            vec![],
            f64::INFINITY,
        )));
    }

    prep_glmm_design(&mut ws, cluster_ids, extra_ids, n, opts);
    Ok(ws)
}

/// The per-call, design-dependent workspace prep: the call-varying options, the
/// packed `M` columns and the two crossed-Schur symbolic factors. Shared by
/// [`build_on_workspace`] (a fresh workspace per call, test-only), the unified
/// core's `FitKind::Glmm` arm and [`fit_glmm_nb`] (both reuse one workspace
/// across draws, hence the `weighted` reset).
pub(super) fn prep_glmm_design(
    ws: &mut GlmmWorkspace,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
    opts: &FitOptions,
) {
    ws.parallel_inner = opts.parallel_inner;
    ws.weight_scale = 1.0;
    if let Some(w) = &opts.weights {
        // Gamma runs its outer φ search and dispersion term on `ŵ = w/s`
        // (precision weights are not scale-free on raw `w` — see
        // `family::gamma_dispersion_term`); every other family keeps raw `w`.
        if matches!(ws.family, crate::Family::Gamma { .. }) {
            let s = crate::family::weight_scale(Some(w), n);
            ws.weight_scale = s;
            for (dst, &wi) in ws.prior_w[..n].iter_mut().zip(w.iter()) {
                *dst = wi / s;
            }
        } else {
            ws.prior_w[..n].copy_from_slice(w);
        }
        ws.weighted = true;
    } else {
        ws.weighted = false;
    }
    ws.offset.clone_from(&opts.offset);

    // The packed M columns with their sparse factor, and the crossed-Schur
    // symbolic factor (read by the structured layout only; None on every other
    // shape), depend on the rows' level ids alone. A draw on the rows the last
    // fit was prepared from keeps them; new ids, even at the same `n`, rebuild.
    if !crate::sparse::same_rows(&ws.design_ids, cluster_ids, extra_ids, n) {
        fill_packed_cols(ws, cluster_ids, extra_ids, n);
        ws.pattern.structured_schur = if ws.layout == GlmmLayout::Structured {
            StructuredSchur::new(&ws.groupings, cluster_ids, extra_ids, n)
        } else {
            None
        };
        crate::sparse::record_rows(&mut ws.design_ids, cluster_ids, extra_ids, n);
    }
    // `TailKernel::tail_inverse` writes `S⁻¹` one tail component at a time and
    // leaves the entries between components alone, which is right only while
    // they hold zeros. This design's components can differ from the last
    // design's on a reused workspace, so clear them once per fit.
    ws.exact_prof.tail_inv.fill(0.0);
}

/// Test-only: the same fit on a workspace forced onto the packed-row layout,
/// whatever layout [`crate::glmm::GlmmLayout::for_design`] would pick for this
/// design, fitted as given (`super::common::as_given_design`). Lets an in-envelope design be fit both ways so the packed kernel can
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
    let (map, x_owned) =
        super::common::as_given_design(x, n, p, &groupings, opts.weights.as_deref());
    let ws =
        GlmmWorkspace::from_groupings(groupings, model.family, p, n, opts.nagq, GlmmLayout::Packed);
    let x_mat = x_owned.as_ref().subrows(0, n);
    // `start` is in the caller's units; the workspace fits the scaled design.
    let start = start.map(|s| {
        let mut s = s.clone();
        let mut scratch = vec![0.0; ws.groupings.re_map_scratch_len()];
        start_to_design(&mut s, &map, &ws.groupings, &mut scratch);
        s
    });
    let mut ws = match build_on_workspace(ws, n, p, cluster_ids, extra_ids, opts) {
        Ok(ws) => ws,
        Err(degenerate) => return *degenerate,
    };
    // NB seeds its `ln θ_NB` coordinate from the no-RE GLM-NB's own θ̂, exactly
    // as [`fit_glmm_nb`] does, so the two routes start the same search.
    let nb_theta = match model.family {
        Family::NegativeBinomial { .. } => {
            let (prefit, theta) = super::glm::fit_glm_nb(x_mat, y, n, p, None, opts, &map);
            nb_glmm_seed(prefit.converged(), theta)
        }
        _ => nb_theta,
    };
    fit_glmm_prebuilt(
        &mut ws,
        x_mat,
        y,
        n,
        p,
        model,
        cluster_ids,
        extra_ids,
        nb_theta,
        start.as_ref(),
        opts,
        &map,
    )
}

/// The NB GLMM's `ln θ_NB` start from the no-RE GLM-NB prefit: its θ̂ when the
/// prefit converged at least a factor [`NB_SEED_FLOOR_FACTOR`] above the box
/// floor `NB_THETA_LO`, and [`NB_SEED_FALLBACK`] otherwise. A prefit that fails,
/// or that runs to the floor (MASS's `glm.nb` fails on the same data there),
/// starts the outer search in a region where PIRLS barely converges and the
/// search can settle at a wrong point: measured on grid cell
/// `nb_q2sx2_g3000p5_bal_base`, a 0.014 start converged 3808 logLik below
/// glmmTMB, while θ = 1 reached glmmTMB's optimum in 324 evaluations, and on
/// `nb_q8_g3000p20_bal_base` a 0.009 start took 38143 evaluations against 1145.
pub(crate) fn nb_glmm_seed(prefit_converged: bool, prefit_theta: f64) -> f64 {
    if prefit_converged && prefit_theta >= NB_SEED_FLOOR_FACTOR * super::NB_THETA_LO {
        prefit_theta
    } else {
        NB_SEED_FALLBACK
    }
}

/// See [`nb_glmm_seed`].
const NB_SEED_FLOOR_FACTOR: f64 = 10.0;
/// See [`nb_glmm_seed`]: the measured start that reaches the optimum on both
/// cells the floor guard exists for.
const NB_SEED_FALLBACK: f64 = 1.0;

/// Borrowed result of [`run_glmm_on`]: the [`GlmmFit`] summary, the θ̂ it was fit
/// at, and a shared borrow of the whole solved [`GlmmWorkspace`] (the assembly
/// reads a dozen of its result slots — β̂, Var, μ̂, û, θ̂, θ_se, vcov, groupings —
/// so borrowing the workspace whole is simpler than enumerating each). Lifetime
/// ties back to the workspace that owns the storage.
pub(crate) struct GlmmResultView<'a> {
    fit: GlmmFit,
    nb_theta: f64,
    ws: &'a GlmmWorkspace,
    /// The column standardization `x` carried: β̂ and the SEs are already the
    /// caller's, θ̂ is read through it.
    map: &'a ColumnScaling,
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
    /// This route's [`FitDiagnostics`]. θ boundary state, per-component pins
    /// and the pivot are all real. The pivot is that of the β information the
    /// SE arm factored at the final PIRLS weights, on the fitted design's
    /// columns: the Rx Schur complement under `WaldSe::Rx` and on the Hessian
    /// arm's Rx fallback, the β block of the joint Hessian with θ profiled out
    /// under `WaldSe::Hessian` (see `InferenceScratch::pivot`). Every layout
    /// records it; a fit with no endpoint, or with no covariance at its
    /// endpoint, leaves it NaN and never flags.
    pub(crate) fn diagnostics(&self) -> FitDiagnostics {
        FitDiagnostics {
            pivot: self.ws.inference.pivot,
            pivot_col: self.ws.inference.pivot_col,
            ill_conditioned: self.ws.inference.pivot < crate::lmm::PIVOT_MIN,
            boundary_hit: self.fit.boundary_hit,
            pinned_components: self.fit.pinned_components,
            pirls_exhausted: self.ws.pirls_exhausted,
            final_pirls_exhausted: self.ws.final_pirls_exhausted,
            hessian_fallback: self.fit.hessian_fallback,
            exact_profile_fallback: self.fit.exact_profile_fallback,
            se_unavailable: self.fit.se_unavailable,
            precision_limited: self.fit.precision_limited,
            separated: self.fit.separated,
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
    /// Whether the fit ended on a finite endpoint (the re-eval deviance): a
    /// converged optimum or a `MaxFunReached` incumbent, not a failed search.
    pub(crate) fn has_endpoint(&self) -> bool {
        self.fit.deviance.is_finite()
    }
    /// Fitted θ̂ vech — the leading `n_theta` params of the joint [θ̂ | β̂] block,
    /// on the fitted design (`FitView::theta` maps it back).
    pub(crate) fn theta(&self) -> &[f64] {
        &self.ws.params[..self.ws.n_theta]
    }
    /// Grouping structure, for the random-effect maps θ̂ is read through.
    pub(crate) fn groupings(&self) -> &crate::lmm::LmmGroupings {
        &self.ws.groupings
    }
}

/// θ-dependent solve half of [`fit_glmm`]: sets the NB dispersion on the
/// prebuilt workspace, cold- or warm-seeds β, runs the GLMM kernel, and returns
/// the borrowed [`GlmmResultView`]. The kernel resets all per-fit warm-start
/// state (`params`, `u_seed`, `coup_mask`, `cluster_rows`) at its top and
/// `theta_cov` further down, after the degenerate-fit guard — an early
/// `nan_fit` bail there (`glmm/mod.rs`) skips the `theta_cov` reset and leaves
/// the previous fit's values in the workspace; harmless since `stddev_se` is
/// itself gated on `converged`. The workspace is designed for cross-fit reuse
/// (see `glmm::fit_glmm`), so calling this repeatedly on one prebuilt `ws` is
/// bit-identical to a fresh construction per call except for that one
/// stale-on-bail slot. `Z`, the symbolic factor, and `x_mat` are inputs the
/// caller fixes before calling; the numeric factorization the kernel writes
/// into `structured_schur` is recomputed every eval. [`glmm_view_to_fit`] maps
/// the returned view to `Fit`. `map` is the column standardization `x_mat`
/// carries; `start` is already on that design.
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
    map: &'a ColumnScaling,
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
    let (beta_start, separated) = match start {
        Some(s) if !s.beta.is_empty() => (s.beta.clone(), false),
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
    if separated {
        let fit = crate::glmm::separated_fit(ws, &beta_start, &opts.target_indices, map);
        let nb_theta = ws.nb_theta;
        return GlmmResultView {
            fit,
            nb_theta,
            ws,
            map,
        };
    }
    // A held Gamma φ is a fixed value of the objective, not a coordinate. The
    // outer search runs on the internal `ŵ = w/s` scale (`prep_glmm_design`),
    // so a held raw-scale `v` enters there as `v/s`.
    ws.gamma_phi_held = match model.family {
        Family::Gamma { .. } => opts.dispersion.map(|v| v / ws.weight_scale),
        _ => None,
    };
    // Gamma inverse link: the search runs in units read off the data and is
    // retried in other units when it fails or ends no lower than the GLM
    // point (`gamma_inverse_units`, `gamma_inverse_retry`).
    let units = match model.family {
        Family::Gamma {
            link: crate::GammaLink::Inverse,
        } => Some(gamma_inverse_units(
            ws,
            x_mat,
            y,
            n,
            cluster_ids,
            &beta_start,
            opts,
        )),
        _ => None,
    };
    let c0 = units.as_ref().map_or(1.0, |u| u.fast);
    let fit_in = |ws: &mut GlmmWorkspace, c: f64| {
        fit_glmm_in_units(
            ws,
            x_mat,
            y,
            n,
            p,
            cluster_ids,
            extra_ids,
            &beta_start,
            start,
            opts,
            map,
            c,
        )
    };
    let mut glmm_fit = fit_in(ws, c0);
    if let Some(u) = units {
        glmm_fit = gamma_inverse_retry(ws, glmm_fit, c0, &u, fit_in);
    }
    // θ̂_NB after the search (NaN on every other family, untouched by the kernel).
    let nb_theta = ws.nb_theta;
    GlmmResultView {
        fit: glmm_fit,
        nb_theta,
        ws,
        map,
    }
}

/// What `run_glmm_on` reads off a Gamma-inverse GLMM's data before its first
/// search ([`gamma_inverse_units`]): response rescales `c` (the kernel fits
/// `y/c`, where η, β and θ are `c` times larger) and the objective at the GLM
/// point.
struct GammaInverseUnits {
    /// `c` of the first search: 1 while the largest cluster η lies in
    /// `[1/64, 64]`, [`Self::clusters`] outside.
    fast: f64,
    /// Brings the largest cluster η, `Σw/Σwy` over the rows of a
    /// primary-grouping level, into `[4, 8]`.
    clusters: f64,
    /// Brings the smallest η of the β start's linear predictor into
    /// `[1/2, 64]`.
    glm: f64,
    /// The objective at θ = 0, the β start and its best φ (the held φ when
    /// there is one), in y's units: a point every search could reach. `+∞`
    /// when the β start has an η ≤ 0.
    glm_point: f64,
}

/// Power of two nearest 1 that brings `v` into `[lo, hi]` (both powers of
/// two); 1 for a `v` that is not a positive finite number.
fn units_into_band(v: f64, lo: f64, hi: f64) -> f64 {
    if !(v > 0.0 && v.is_finite()) {
        return 1.0;
    }
    let l = v.log2();
    let e = 0.0_f64.clamp((lo.log2() - l).ceil(), (hi.log2() - l).floor());
    2.0_f64.powi(e.clamp(-1000.0, 1000.0) as i32)
}

/// The [`GammaInverseUnits`] of this fit.
///
/// On the inverse link η = 1/μ is in the reciprocal of y's units, and so are
/// β, the random effects and θ, while every θ constant of the search (the
/// blind start θ₀ = 1, the box, the trust radii, the pin, the
/// finite-difference step) is absolute. The likelihood can have two basins
/// whose θ live on scales decades apart: σ = 0, near the GLM fit, with its
/// structure on the scale of the smallest start η (those rows weigh most,
/// `1/(φη²)`, and a random effect pushes them out of `η > 0` first), and an
/// interior one whose SD is on the scale of the largest cluster η.
///
/// The bands were measured on y·k, k over 2⁻²⁰ to 2²⁰, for a 12-row design
/// whose four cluster means span up to nine decades and for corpus fits with
/// 15 rows. In y's own units the search reaches an interior optimum with the
/// largest cluster η from about 2 to 12 and ends converged short of it, with
/// no sign the retry reads (`gamma_inverse_retry`), mostly where that η is
/// below 1/64 or above 64; there the first search runs on y rescaled into
/// `[4, 8]` instead. The σ = 0 optimum, with an η of order 1e-4, is reached
/// with the smallest start η in `[1/2, 64]`.
fn gamma_inverse_units(
    ws: &GlmmWorkspace,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    n: usize,
    cluster_ids: &[u32],
    beta_start: &[f64],
    opts: &FitOptions,
) -> GammaInverseUnits {
    let fam = ws.family;
    let w = &ws.prior_w[..n];
    let (mut eta_min, mut dev, mut sum_ln_y) = (f64::INFINITY, 0.0, 0.0);
    for i in 0..n {
        let mut eta = opts.offset.as_ref().map_or(0.0, |o| o[i]);
        for (j, &b) in beta_start.iter().enumerate() {
            eta += x_mat[(i, j)] * b;
        }
        eta_min = eta_min.min(eta);
        dev += w[i] * crate::family::dev_resid_eta(fam, f64::NAN, y[i], eta);
        sum_ln_y += y[i].ln();
    }
    let weights = ws.weighted.then_some(w);
    let phi = ws
        .gamma_phi_held
        .unwrap_or_else(|| crate::family::gamma_ml_dispersion(dev, weights, n))
        .clamp(crate::glmm::GAMMA_PHI_LO, crate::glmm::GAMMA_PHI_HI);
    let glm_point = if eta_min > 0.0 {
        dev / phi + crate::family::gamma_dispersion_term(phi.ln(), weights, n, sum_ln_y)
    } else {
        f64::INFINITY
    };
    let levels = cluster_ids[..n]
        .iter()
        .copied()
        .max()
        .map_or(0, |m| m as usize + 1);
    let (mut wy, mut sw) = (vec![0.0; levels], vec![0.0; levels]);
    for i in 0..n {
        wy[cluster_ids[i] as usize] += w[i] * y[i];
        sw[cluster_ids[i] as usize] += w[i];
    }
    let eta_max = (0..levels)
        .filter(|&g| wy[g] > 0.0)
        .map(|g| sw[g] / wy[g])
        .fold(0.0, f64::max);
    let clusters = units_into_band(eta_max, 4.0, 8.0);
    GammaInverseUnits {
        fast: if units_into_band(eta_max, 1.0 / 64.0, 64.0) == 1.0 {
            1.0
        } else {
            clusters
        },
        clusters,
        glm: units_into_band(eta_min, 0.5, 64.0),
        glm_point,
    }
}

/// Relative deviance band of [`gamma_inverse_retry`]: a first search that ends
/// within it of the GLM point, or above it, is retried.
const UNITS_RETRY_TOL: f64 = 1e-6;

/// The retry of a Gamma-inverse GLMM whose first search (`fast`, at `c0`)
/// failed or ended no lower than the GLM point (`GammaInverseUnits::glm_point`
/// less [`UNITS_RETRY_TOL`]). It searches in the units of each basin the first
/// search did not use (see [`gamma_inverse_units`]): `glm` for σ = 0, and for
/// the interior `clusters` and twice it (the largest cluster η in `[8, 16]`),
/// since the interior optimum can be reached in one of those octaves and
/// missed in the other. It keeps the converged search of lowest deviance. When
/// a lower point was seen (the GLM point, or the incumbent of a search that
/// failed) and a search failed, it reports the failed search with the lowest
/// incumbent instead: a converged σ = 0 fit is then not the optimum. When the
/// kept search did not run last, its saved [`SearchState`] is put back on the
/// workspace; `n_eval` counts every search.
fn gamma_inverse_retry(
    ws: &mut GlmmWorkspace,
    fast: GlmmFit,
    c0: f64,
    u: &GammaInverseUnits,
    mut fit_in: impl FnMut(&mut GlmmWorkspace, f64) -> GlmmFit,
) -> GlmmFit {
    let band = |v: f64| UNITS_RETRY_TOL * (1.0 + v.abs());
    if fast.converged && fast.deviance < u.glm_point + band(u.glm_point) {
        return fast;
    }
    // (c, converged, deviance, lowest objective seen, fit) per search.
    let mut runs = vec![(
        c0,
        fast.converged,
        fast.deviance,
        ws.best_objective.min(fast.deviance),
        fast,
    )];
    // The workspace state of each search but the last, which `ws` holds.
    let mut saved = Vec::new();
    for c in [u.glm, u.clusters, 2.0 * u.clusters] {
        if runs.iter().any(|r| r.0 == c) {
            continue;
        }
        saved.push(SearchState::save(ws));
        let fit = fit_in(ws, c);
        runs.push((
            c,
            fit.converged,
            fit.deviance,
            ws.best_objective.min(fit.deviance),
            fit,
        ));
    }
    let spent = runs.iter().map(|r| r.4.n_eval).sum();
    let seen = runs.iter().map(|r| r.3).fold(u.glm_point, f64::min);
    let lowest_converged = runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.1)
        .min_by(|(_, a), (_, b)| a.2.total_cmp(&b.2));
    let lowest_failed = runs
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.1)
        .min_by(|(_, a), (_, b)| a.3.total_cmp(&b.3));
    let pick = match (lowest_converged, lowest_failed) {
        (Some((_, c)), Some((f, _))) if seen < c.2 - band(c.2) => f,
        (Some((c, _)), _) => c,
        (None, Some((f, _))) => f,
        (None, None) => runs.len() - 1,
    };
    if pick < saved.len() {
        saved.swap_remove(pick).restore(ws);
    }
    let mut kept = runs.swap_remove(pick).4;
    kept.n_eval = spent;
    kept
}

/// The workspace slots a search writes that are read after `run_glmm_on`
/// returns (by the [`GlmmResultView`] accessors and [`glmm_view_to_fit`]), so
/// [`gamma_inverse_retry`] can put back a search that did not run last. Every
/// other slot a search writes is scratch that `fit_glmm` rewrites before it
/// reads it. Mirrors those readers — change together.
struct SearchState {
    params: Vec<f64>,
    betas: Vec<f64>,
    gamma_phi: f64,
    var_diag: Vec<f64>,
    t_sq: Vec<f64>,
    vcov: Mat<f64>,
    theta_cov: Mat<f64>,
    pivot: f64,
    pivot_col: u32,
    prob: Vec<f64>,
    u: Vec<f64>,
    pirls_exhausted: u32,
    final_pirls_exhausted: bool,
}

impl SearchState {
    fn save(ws: &GlmmWorkspace) -> Self {
        let inf = &ws.inference;
        Self {
            params: ws.params.clone(),
            betas: ws.betas.clone(),
            gamma_phi: ws.gamma_phi,
            var_diag: inf.var_diag.clone(),
            t_sq: inf.t_sq.clone(),
            vcov: inf.vcov.clone(),
            theta_cov: inf.theta_cov.clone(),
            pivot: inf.pivot,
            pivot_col: inf.pivot_col,
            prob: ws.pirls.prob.clone(),
            u: ws.pirls.u.clone(),
            pirls_exhausted: ws.pirls_exhausted,
            final_pirls_exhausted: ws.final_pirls_exhausted,
        }
    }

    fn restore(self, ws: &mut GlmmWorkspace) {
        ws.params = self.params;
        ws.betas = self.betas;
        ws.gamma_phi = self.gamma_phi;
        let inf = &mut ws.inference;
        inf.var_diag = self.var_diag;
        inf.t_sq = self.t_sq;
        inf.vcov = self.vcov;
        inf.theta_cov = self.theta_cov;
        inf.pivot = self.pivot;
        inf.pivot_col = self.pivot_col;
        ws.pirls.prob = self.prob;
        ws.pirls.u = self.u;
        ws.pirls_exhausted = self.pirls_exhausted;
        ws.final_pirls_exhausted = self.final_pirls_exhausted;
    }
}

/// One search of [`crate::glmm::fit_glmm`] on `y/c` (η, β and θ times `c`),
/// mapped back to y's units on the workspace and the returned fit.
#[allow(clippy::too_many_arguments)]
fn fit_glmm_in_units(
    ws: &mut GlmmWorkspace,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    n: usize,
    p: usize,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    beta_start: &[f64],
    start: Option<&StartValues>,
    opts: &FitOptions,
    map: &ColumnScaling,
    c: f64,
) -> GlmmFit {
    if c == 1.0 {
        return crate::glmm::fit_glmm(
            ws,
            x_mat,
            y,
            cluster_ids,
            extra_ids,
            &opts.target_indices,
            warm_theta(start),
            beta_start,
            n,
            opts.wald_se,
            map,
        );
    }
    let y_fit: Vec<f64> = y[..n].iter().map(|&v| v / c).collect();
    let beta_fit: Vec<f64> = beta_start.iter().map(|&b| b * c).collect();
    let theta_fit: Option<Vec<f64>> = warm_theta(start).map(|t| t.iter().map(|&v| v * c).collect());
    if let Some(o) = ws.offset.as_mut() {
        o.iter_mut().for_each(|v| *v *= c);
    }
    let mut fit = crate::glmm::fit_glmm(
        ws,
        x_mat,
        &y_fit,
        cluster_ids,
        extra_ids,
        &opts.target_indices,
        theta_fit.as_deref(),
        &beta_fit,
        n,
        opts.wald_se,
        map,
    );
    if let Some(o) = ws.offset.as_mut() {
        o.iter_mut().for_each(|v| *v /= c);
    }
    let (n_theta, c2) = (ws.n_theta, c * c);
    ws.params[..n_theta + p].iter_mut().for_each(|v| *v /= c);
    ws.betas[..p].iter_mut().for_each(|v| *v /= c);
    for &t in &opts.target_indices {
        ws.inference.var_diag[t as usize] /= c2;
    }
    for j in 0..p {
        for i in 0..p {
            ws.inference.vcov[(i, j)] /= c2;
        }
    }
    for j in 0..n_theta {
        for i in 0..n_theta {
            ws.inference.theta_cov[(i, j)] /= c2;
        }
    }
    ws.pirls.prob[..n].iter_mut().for_each(|v| *v *= c);
    let shift = 2.0 * n as f64 * c.ln();
    fit.deviance += shift;
    ws.best_objective += shift;
    fit
}

/// Maps a [`GlmmResultView`] to the full stable `Fit`, the converged conditional
/// means `μ̂` (length `n`), and the minimized marginal Laplace deviance; callers
/// take `.0`, the deviance rides along for the tests that compare routes at
/// fixed θ. Needs raw `y` (σ̂²/dispersion/loglik read it) and `model` (family
/// selection); θ̂ comes from the view. Every workspace slot a search writes
/// and this function or a [`GlmmResultView`] accessor reads is in
/// [`SearchState`] — change together.
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
    let map = view.map;
    let glmm_fit = &view.fit;
    let diag = view.diagnostics();
    let converged = diag.converged;
    // Gated on the finite endpoint (the re-eval deviance), not `converged`: the
    // plateau policy `fit/lmm.rs` applies. A `MaxFunReached` exit reports τ̂², D̂
    // and the dispersion at its incumbent with `converged == false`; every other
    // non-convergence goes through `nan_fit`, whose deviance is +∞.
    let has_endpoint = view.has_endpoint();
    let nb_theta = view.nb_theta;
    let n_theta = ws.n_theta;

    // Map GlmmFit + workspace state → Fit.
    // ws.betas: length p, all fixed effects; ws.inference.var_diag: predictor-indexed.
    let beta = ws.betas.clone();
    let mut se = vec![f64::NAN; p];
    fill_se_by_predictor(&ws.inference.var_diag, &opts.target_indices, &mut se);

    // tau2[k] = θ̂[k]², with no residual-scale factor on any family. θ̂ =
    // ws.params[..n_theta] is the factor of the unit ridge ‖u‖², and the
    // dispersion is carried on the DATA side of the Gamma objective — the prior
    // weights PIRLS runs on are divided by φ̂ (see
    // `glmm::deviance::laplace_deviance`) — never on that ridge. θ̂ therefore
    // comes out in the response's own absolute units, and rescaling it here
    // would double-count φ̂. lme4 reports `σ·θ̂` instead because its own
    // φ-free PIRLS leaves θ̂ relative; glmmTMB and GLMMadaptive report RE
    // standard deviations in absolute units, as this does. (`dispersion` below
    // is φ̂ itself.) Same q≥2-slope caveat as fit_mle's tau2.
    let sigma_sq = if has_endpoint { 1.0 } else { f64::NAN };
    // θ̂ in the caller's coordinates (slot order): the column standardization
    // mapped out (`LmmGroupings::theta_kernel_to_caller`), so every θ-derived
    // magnitude is in the design's own units before it is
    // squared.
    let (theta_caller, caller_pins) =
        ws.groupings
            .theta_to_caller_vec(&ws.params[..n_theta], converged, map);
    let tau2: Vec<f64> = if has_endpoint {
        theta_caller.iter().map(|&t| t * t * sigma_sq).collect()
    } else {
        vec![f64::NAN; n_theta]
    };

    // Dispersion. Binomial/Poisson hold φ≡1. Gamma reports φ̂, the maximum-
    // likelihood value of the Laplace objective: the outer search carries
    // `ln φ_int` as a coordinate on the internal `ŵ = w/s` scale and leaves
    // φ̂_int in `ws.gamma_phi` (`glmm::fit_glmm`); the reported φ̂ is `s·φ̂_int`.
    // It is the φ both SE arms already carry on that internal scale (the
    // PIRLS weights are `ŵᵢ/φ̂_int` = `wᵢ/φ̂`), so nothing else is rescaled here.
    let dispersion = if !has_endpoint {
        f64::NAN
    } else {
        match model.family {
            Family::Gamma { .. } => match opts.dispersion {
                Some(v) => v,
                None => ws.weight_scale * ws.gamma_phi,
            },
            Family::NegativeBinomial { .. } => nb_theta,
            _ => 1.0,
        }
    };

    // GLMM D̂ = σ̂²·Λ̂Λ̂' — the same σ̂² (≡ 1, see tau2 above) that scales tau2,
    // so the two accessors report the one variance component on one scale.
    // Oracle on Gamma: `fit_glmm_gamma_sim_matches_grid_reference` /
    // `validation/goldens/sim_gamma_glmm_tmb.json` varcomp stddevs.
    let varcorr = if has_endpoint {
        assemble_varcorr(&ws.params[..n_theta], &ws.groupings, sigma_sq, map)
    } else {
        vec![]
    };

    // SE of the RE stddevs from the joint-Hessian θ block (`WaldSe::Hessian` only;
    // NaN under Rx / RX fallback / non-converged — `ws.inference.theta_cov` is reset per fit
    // and refilled only by `joint_hessian_cov`).
    //
    // The joint Hessian is taken in the INTERNAL θ̃ on both arms (the FD stencil
    // perturbs it, the exact kernel differentiates w.r.t. it) — mirrors
    // §Boundary handling in `documentation/algorithms-glmm.md`, change together.
    // `stddev_se_kernel_to_caller` maps it to the SE of each reported stddev
    // `√Σ_kk` by the delta method on θ̃.
    let stddev_se = if converged {
        let mut out = vec![f64::NAN; n_theta];
        ws.groupings.stddev_se_kernel_to_caller(
            &ws.params[..n_theta],
            ws.inference.theta_cov.as_ref(),
            &mut out,
            map,
        );
        out
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
            super::common::assemble_ranef_sparse(&ws.params[..n_theta], &ws.groupings, u, map)
        } else {
            super::common::assemble_ranef_dense(&ws.params[..n_theta], &ws.groupings, u, map)
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
    // `pinned` in the caller's coordinates, where they differ from the
    // kernel's (where they do not, the kernel's mask stands; on the AGQ route it
    // comes from a canonical copy the search's Λ is not replaced by).
    let mut diag = diag;
    if let Some(mask) = caller_pins {
        diag.pinned_components = mask;
    }
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
    // A capped (`MaxFunReached`) endpoint is reported as a point, not an
    // accepted optimum, even though `varcorr` is now finite there — so the
    // post-hoc negligible-component check only runs on a converged fit
    // (mirrors `fit/lmm.rs`, same gate).
    fit.diagnostics.singular = fit.diagnostics.singular
        || (converged
            && fit.has_negligible_component(&ws.groupings.kernel_stddevs(&ws.params[..n_theta])));
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
    map: &ColumnScaling,
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
        map,
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
/// other family (θ_NB has no start slot). `dispersion = θ̂_NB`; the joint
/// Hessian of `WaldSe::Hessian` carries the `ln θ_NB` coordinate, so the β SE
/// includes θ_NB's uncertainty (unlike the GLM route's `fit_glm_nb`, which
/// conditions on θ̂ — see [`crate::glmm::se::joint_hessian_cov`]).
///
/// `ws` is built once per shape by the unified core
/// (`GlmmWorkspace::for_cluster_spec_ext` on [`nb_spec`] at the build shape)
/// and reused across calls: [`prep_glmm_design`] runs here per call, as the
/// core's `FitKind::Glmm` arm runs it for the other families, and `fit_glmm`
/// resets every per-fit state at its top, so a reused workspace gives the fit
/// a fresh one gives. `x` is the column-major design the kernel fits (`n`
/// rows), standardized through `cs`; `start` (θ in slot order, each component
/// empty or of its full width) and the returned `Fit` are in the caller's
/// units.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_glmm_nb(
    ws: &mut GlmmWorkspace,
    x: faer::MatRef<'_, f64>,
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    start: Option<&StartValues>,
    opts: &FitOptions,
    cs: &ColumnScaling,
) -> Fit {
    // Degenerate guard, the same as `build_on_workspace`'s and the core's
    // `FitKind::Glmm` arm's.
    if n <= p || p == 0 {
        return degenerate_glmm_fit(p, ws.n_theta);
    }
    prep_glmm_design(ws, cluster_ids, extra_ids, n, opts);
    let nb_spec = nb_spec(model);
    // `start` is in the caller's units (θ in slot order); the workspace fits
    // the standardized design.
    let start = start.map(|s| {
        let mut s = s.clone();
        let mut scratch = vec![0.0; ws.groupings.re_map_scratch_len()];
        start_to_design(&mut s, cs, &ws.groupings, &mut scratch);
        s
    });
    // The method-of-moments seed (`nb_theta_moment_seed`) charges the random-effect
    // variance to the dispersion: on a GLMM it lands one to two orders of magnitude
    // below θ̂, and on random-slope shapes PIRLS never converges at those dispersions,
    // handing the outer BOBYQA a `+∞` plateau it can't escape. The no-RE GLM-NB's own
    // θ̂ is a start, not an estimate (the fixed effects alone under-explain the mean,
    // so it still moves under the outer search), but it starts inside the basin PIRLS
    // can actually converge in, except where the prefit itself fails or runs to
    // the box floor — see [`nb_glmm_seed`]. `fit_glm_nb` hands back its θ as its
    // second return value — the last θ its alternation stood on, the moment seed
    // if the first inner IRLS failed, the θ reached so far if a later one did —
    // because the `Fit`'s own `dispersion` field is NaN unless the prefit
    // converged.
    let (prefit, prefit_theta) = super::glm::fit_glm_nb(x, y, n, p, None, opts, cs);
    let nb_seed = nb_glmm_seed(prefit.converged(), prefit_theta);
    fit_glmm_prebuilt(
        ws,
        x,
        y,
        n,
        p,
        &nb_spec,
        cluster_ids,
        extra_ids,
        nb_seed,
        start.as_ref(),
        opts,
        cs,
    )
    .0
}

/// The NB GLMM's spec: the caller's random effects on the NB/log family.
pub(super) fn nb_spec(model: &ModelSpec) -> ModelSpec {
    ModelSpec {
        family: Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        re: model.re.clone(),
    }
}
