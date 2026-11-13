//! GLMM (Binomial/Poisson/Gamma/negative-binomial, `re: Some`) dispatch —
//! estimator dispatch, the numerical kernel lives in `src/glmm/`. Builds the
//! `GlmmWorkspace`/RE design `Z`/crossed-Schur symbolic factor, cold-starts β
//! from the no-RE GLM fit (`glm_warm_start_beta`), and maps `GlmmFit` +
//! workspace state back to `Fit`. `run_glmm_nb_on` searches the NB dispersion
//! `θ_NB` as one more coordinate of the kernel's own outer BOBYQA
//! (`glmm::fit_glmm`), so one call fits β, θ_RE and θ_NB together.

use faer::Mat;

use crate::glm::{glm_irls_fit, GlmScratch};
use crate::glmm::{fill_packed_cols, GlmmFit, GlmmLayout, GlmmWorkspace, StructuredSchur};
use crate::standardize::{ColumnScaling, Units};
use crate::{Family, ModelSpec, NegBinomialLink, StartValues};

use super::common::{
    assemble_varcorr, fill_se_by_predictor, glmm_objective, warm_theta, FitDiagnostics,
};
#[cfg(test)]
use super::SizedSpec;
use super::{Fit, FitOptions, SeType};

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
/// if the GLM does not converge to finite coefficients. The cold path pays this
/// solve; a stable warm start pays it too, for the separation flag only; a
/// loop-tier warm start (the MCPower hot loop) supplies β and never calls this.
///
/// The flag is `GlmFitView::separated`: the GLM found no finite maximum, and β is
/// then its last iterate instead of the fallback, for the caller to report. A
/// binomial response that is all 0 or all 1 is flagged too, with NaN β: the GLM
/// declines it without iterating, and a GLMM run from β = 0 would drift to
/// |β₀| ≈ 50 and report convergence.
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
    // Mirrors the all-0 / all-1 short circuit in `crate::glm::glm_irls_fit_mapped`.
    let y_sum: f64 = y[..n].iter().sum();
    if matches!(family, Family::Binomial { .. }) && (y_sum <= 0.0 || y_sum >= n as f64) {
        return (vec![f64::NAN; p], true);
    }
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
    let mut irls_broken = vec![false; p];
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
            irls_broken: &mut irls_broken,
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
/// per-call guards of the unified core's `FitKind::Glmm` and
/// `FitKind::GlmmNb` arms (both build their workspace once per shape, so the check runs per
/// call), and by the test-only [`build_on_workspace`]. `groupings` and
/// `nagq` are the workspace's, for the per-grouping layout and the route's
/// [`Fit::objective`].
pub(super) fn degenerate_glmm_fit(
    n: usize,
    p: usize,
    groupings: &crate::lmm::LmmGroupings,
    nagq: u8,
) -> Fit {
    let mut fit = Fit::empty(n, p, Some(groupings));
    fit.objective = glmm_objective(nagq);
    fit
}

/// θ-invariant build half: allocates the workspace for this (spec, n) shape
/// and runs [`build_on_workspace`]: the θ-independent options, the RE design
/// and the crossed-Schur symbolic factor,
/// none of which depend on `nb_theta`. On the degenerate n≤p/p=0
/// short-circuit returns `Err` carrying the same NaN `Fit` triple the public
/// path returns. Split from [`fit_glmm_prebuilt`] so a caller with its own
/// workspace policy can build once and solve on it. `weights` are the ones the
/// fit runs on (`Units::weights`). Test-only: every fit path builds its
/// workspace in the unified core (`fit::core::build_workspace`).
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn build_glmm(
    n: usize,
    p: usize,
    sized: &SizedSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    opts: &FitOptions,
) -> Result<GlmmWorkspace, Box<(Fit, Vec<f64>, f64)>> {
    let (slope_cols, extra_slope_cols) = re_slope_cols(&sized.spec);
    // Workspace for this (spec, n) shape — sizes per-cluster solver buffers off
    // the sized cluster count; the kernels cold-start θ from their blind θ₀.
    let ws =
        GlmmWorkspace::for_cluster_spec_ext(p, sized, n, &slope_cols, &extra_slope_cols, opts.nagq);
    build_on_workspace(ws, n, p, cluster_ids, extra_ids, weights, offset, opts)
}

#[cfg(test)]
type BuiltGlmm = (GlmmWorkspace, Mat<f64>, Units);

/// Test-only: [`build_glmm`] on row-major `x`, returning the column-major copy
/// and the draw's units (`fit::core::draw_units`) alongside the workspace.
/// [`run_glmm_on`] given those units searches on the units the shipped fit
/// searches on.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glmm_build(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    sized: &SizedSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    opts: &FitOptions,
) -> Result<BuiltGlmm, Box<(Fit, Vec<f64>, f64)>> {
    let x_mat = super::common::to_col_major(x, n, p);
    let mut units = Units::new(sized.spec.family, n, p);
    let map = ColumnScaling::IDENTITY;
    super::core::draw_units(
        &mut units,
        None,
        true,
        x,
        x_mat.as_ref(),
        y,
        weights,
        offset,
        n,
        p,
        map,
    );
    let fit_weights = units.weights(weights);
    let ws = build_glmm(
        n,
        p,
        sized,
        cluster_ids,
        extra_ids,
        fit_weights,
        offset,
        opts,
    )?;
    Ok((ws, x_mat, units))
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
/// workspace: [`prep_glmm_design`], after the degenerate guard. `weights` are
/// the ones the fit runs on (`Units::weights`). Test-only.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn build_on_workspace(
    mut ws: GlmmWorkspace,
    n: usize,
    p: usize,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    opts: &FitOptions,
) -> Result<GlmmWorkspace, Box<(Fit, Vec<f64>, f64)>> {
    // Degenerate guard (mirrors the kernel's n≤p short-circuit contract).
    if n <= p || p == 0 {
        return Err(Box::new((
            degenerate_glmm_fit(n, p, &ws.groupings, ws.nagq),
            vec![],
            f64::INFINITY,
        )));
    }

    prep_glmm_design(&mut ws, cluster_ids, extra_ids, weights, offset, n, opts);
    Ok(ws)
}

/// The per-call, design-dependent workspace prep: the call-varying options, the
/// packed `M` columns and the two crossed-Schur symbolic factors. Shared by
/// [`build_on_workspace`] (a fresh workspace per call, test-only), the unified
/// core's `FitKind::Glmm` arm and [`run_glmm_nb_on`] (both reuse one workspace
/// across draws, hence the `weighted` reset). `weights` are the ones the fit
/// runs on (`Units::weights`): `ŵ = w/s` on Gamma, whose outer φ search and
/// dispersion term are not scale-free on raw `w` (`family::gamma_dispersion_term`),
/// and the caller's trials or exposure on every other family.
pub(super) fn prep_glmm_design(
    ws: &mut GlmmWorkspace,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    n: usize,
    opts: &FitOptions,
) {
    ws.parallel_inner = opts.parallel_inner;
    if let Some(w) = weights {
        ws.prior_w[..n].copy_from_slice(&w[..n]);
        ws.weighted = true;
    } else {
        // Most consumers read `prior_w` without gating on `weighted` (the
        // PIRLS working-weight and deviance folds, the pass-3 effective
        // residual), so a call after a weighted one restores the unit
        // weights a fresh workspace holds.
        if ws.weighted {
            ws.prior_w.fill(1.0);
        }
        ws.weighted = false;
    }
    // Copied into the held buffer, so a draw on a reused workspace allocates
    // nothing for its offset.
    match (&mut ws.offset, offset) {
        (Some(dst), Some(o)) => o.clone_into(dst),
        (dst, o) => *dst = o.map(<[f64]>::to_vec),
    }

    // The packed M columns with their sparse factor, and the crossed-Schur
    // symbolic factor (read by the structured layout only; None on every other
    // shape), depend on the rows' level ids alone, and a workspace owns its ids:
    // the first draw builds them and every later one keeps them.
    if !ws.design_built {
        fill_packed_cols(ws, cluster_ids, extra_ids, n);
        ws.pattern.structured_schur = if ws.layout == GlmmLayout::Structured {
            StructuredSchur::new(&ws.groupings, cluster_ids, extra_ids, n)
        } else {
            None
        };
        ws.design_built = true;
    }
    // `TailKernel::tail_inverse` writes `S⁻¹` one tail component at a time and
    // leaves the entries between components alone, which is right only while
    // they hold zeros. Its fallbacks write the whole matrix, so clear them once
    // per fit rather than trust what the last fit left.
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
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    sized: &SizedSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    nb_theta: f64,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> (Fit, Vec<f64>, f64) {
    let model = &sized.spec;
    let (slope_cols, extra_slope_cols) = re_slope_cols(model);
    let groupings =
        crate::lmm::LmmGroupings::from_cluster_spec_ext(sized, &slope_cols, &extra_slope_cols);
    let (map, x_owned) = super::common::as_given_design(x, n, p, &groupings, weights);
    let ws =
        GlmmWorkspace::from_groupings(groupings, model.family, p, n, opts.nagq, GlmmLayout::Packed);
    let x_mat = x_owned.as_ref().subrows(0, n);
    // `start` is in the caller's units; the workspace fits the scaled design.
    let start = start.map(|s| {
        let mut s = s.clone();
        let mut scratch = vec![0.0; ws.groupings.re_map_scratch_len()];
        super::common::start_to_design(&mut s, &map, &ws.groupings, &mut scratch);
        s
    });
    let mut units = Units::new(model.family, n, p);
    super::core::draw_units(
        &mut units, None, true, x, x_mat, y, weights, offset, n, p, &map,
    );
    let fit_weights = units.weights(weights);
    let mut ws =
        match build_on_workspace(ws, n, p, cluster_ids, extra_ids, fit_weights, offset, opts) {
            Ok(ws) => ws,
            Err(degenerate) => return *degenerate,
        };
    // NB seeds its `ln θ_NB` coordinate from the no-RE GLM-NB's own θ̂, exactly
    // as [`run_glmm_nb_on`] does, so the two routes start the same search.
    let nb_theta = match model.family {
        Family::NegativeBinomial { .. } => match start.as_ref().and_then(|s| s.dispersion) {
            Some(d) => d,
            None => {
                let (prefit, theta) = super::glm::fit_glm_nb(
                    x_mat,
                    y,
                    weights,
                    units.offset(offset),
                    n,
                    p,
                    None,
                    opts,
                    &map,
                );
                nb_glmm_seed(prefit.converged(), theta)
            }
        },
        _ => nb_theta,
    };
    fit_glmm_prebuilt(
        &mut ws,
        x_mat,
        x_mat.subcols(0, 0),
        y,
        offset,
        n,
        p,
        model,
        cluster_ids,
        extra_ids,
        nb_theta,
        start.as_ref(),
        opts,
        &map,
        &units,
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
    if prefit_converged && prefit_theta >= NB_SEED_FLOOR_FACTOR * crate::consts::NB_THETA_LO {
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
    /// Mutable for the conditional variances alone, whose mode re-solve
    /// (`glmm::condvar_blocks`) writes the workspace's scratch.
    ws: &'a mut GlmmWorkspace,
    /// The column standardization `x` carried: β̂ and the SEs are already the
    /// caller's, θ̂ is read through it.
    map: &'a ColumnScaling,
    /// The units the workspace's weights are in: a Gamma φ̂ is `s` times the
    /// workspace's.
    units: &'a Units,
    /// The fitted design, its random-effect-only columns and the ids, for
    /// the conditional variances.
    x: faer::MatRef<'a, f64>,
    z: faer::MatRef<'a, f64>,
    cluster_ids: &'a [u32],
    extra_ids: &'a [Vec<u32>],
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
    /// The estimated dispersion in the caller's units (`Fit::dispersion`'s
    /// value), the one to carry into [`crate::StartValues::dispersion`]:
    /// θ̂_NB on NB, φ̂ on Gamma. `None` on every other family, on a held φ and
    /// on a fit with no endpoint.
    pub(crate) fn dispersion_estimate(&self) -> Option<f64> {
        match self.ws.family {
            Family::NegativeBinomial { .. } => self.has_endpoint().then_some(self.nb_theta),
            Family::Gamma { .. } => (self.ws.gamma_phi_held.is_none() && self.has_endpoint())
                .then_some(self.units.dispersion_scale() * self.ws.gamma_phi),
            _ => None,
        }
    }
    /// Whether this is the negative-binomial GLMM.
    pub(crate) fn is_nb(&self) -> bool {
        matches!(self.ws.family, Family::NegativeBinomial { .. })
    }
    /// Per-predictor Var(β̂_j), predictor-indexed length p — only the
    /// `target_indices` slots are written; a non-target slot reads 0.0 on a
    /// fresh workspace or a previous fit's value on a reused one.
    pub(crate) fn var_diag(&self) -> &[f64] {
        &self.ws.inference.var_diag
    }
    /// Var(β̂_p − β̂_n) = V_pp + V_nn − 2·V_pn off the workspace's Cov(β̂),
    /// which the `wald_se` arm that set `var_diag` fills on every column pair
    /// of a fit with a covariance (NaN otherwise). The caller checks
    /// `p_col, n_col < p`.
    pub(crate) fn contrast_var(&self, p_col: usize, n_col: usize) -> f64 {
        let v = &self.ws.inference.vcov;
        v[(p_col, p_col)] + v[(n_col, n_col)] - 2.0 * v[(p_col, n_col)]
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
            gradient_search_fallback: self.fit.gradient_search_fallback,
            n_grad: self.ws.grad_report.map_or(0, |r| r.1 as u32),
            outer_stop: self.ws.grad_report.map(|r| r.0.into()),
            outer_decrement: self.ws.grad_report.map_or(f64::NAN, |r| r.2),
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
    /// converged optimum or a capped search's incumbent, not a failed search.
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
/// and `z_mat` (the `n × s` random-effect-only columns) carry; `start` is
/// already on that design. `y` and `offset` are the caller's, and `units` the
/// draw's units: those of the weights [`prep_glmm_design`] stored, and the
/// response scale every search runs on ([`fit_glmm_in_units`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_glmm_on<'a>(
    ws: &'a mut GlmmWorkspace,
    x_mat: faer::MatRef<'a, f64>,
    z_mat: faer::MatRef<'a, f64>,
    y: &[f64],
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &'a [u32],
    extra_ids: &'a [Vec<u32>],
    nb_theta: f64,
    start: Option<&StartValues>,
    targets: &[u32],
    opts: &FitOptions,
    map: &'a ColumnScaling,
    units: &'a Units,
) -> GlmmResultView<'a> {
    // NB θ₀ (the start of the `ln θ_NB` coordinate) is threaded explicitly; the
    // kernel leaves θ̂_NB in `ws.nb_theta`. NaN for every non-NB family (unread).
    ws.nb_theta = nb_theta;
    // The fused logit deviance identity holds only for 0/1 rows
    // (`family::weighted_deviance_needed`); `prior_w` is all 1 here when
    // unweighted (`prep_glmm_design` re-decides the flag per fit).
    ws.weighted = crate::family::weighted_deviance_needed(
        model.family,
        &y[..n],
        ws.weighted.then_some(&ws.prior_w[..n]),
    );

    // Warm start threads β + θ into the GLMM kernel. A caller-supplied `start` (the
    // MCPower hot loop) uses its β verbatim; a cold start seeds β from the no-RE GLM
    // fit (lme4/glmer initialization — see `glm_warm_start_beta`) instead of 0, so
    // the inner PIRLS opens near the mean and does not overshoot. θ still cold-starts
    // at the kernel's THETA0 blind start. An empty field is a per-component cold
    // start (`StartValues`), so it takes the same branch as `start = None`. A fit
    // from a β start that fails is searched again from the cold start (below).
    // The stable door checks a β start for separation with the same GLM and
    // stops as a cold start does when it separates, otherwise keeping the
    // caller's β; the loop tier trusts the start and skips the GLM.
    ws.intercept = units.intercept();
    // The GLM seed runs on the internal response and offset, `y/uc` and
    // `o − m`, and stays in those units (`beta_units` = `Some(uc)`): a search
    // at `uc` then starts from it bit for bit, whatever the caller's units. A
    // caller's β start is in the caller's units (`beta_units` = `None`).
    let uc = units.c();
    let glm_seed = || {
        let (y_int, offset_int) = (units.y(y), units.offset(offset));
        glm_warm_start_beta(model.family, nb_theta, x_mat, y_int, n, p, offset_int)
    };
    let glm_start;
    let (beta_start, beta_units, separated): (&[f64], Option<f64>, bool) = match start {
        Some(s) if !s.beta.is_empty() && ws.check_start_separation => {
            glm_start = glm_seed();
            if glm_start.1 {
                (&glm_start.0, Some(uc), true)
            } else {
                (&s.beta, None, false)
            }
        }
        Some(s) if !s.beta.is_empty() => (&s.beta, None, false),
        _ => {
            glm_start = glm_seed();
            (&glm_start.0, Some(uc), glm_start.1)
        }
    };
    if separated {
        // The GLM's last iterate, in the caller's units.
        let mut beta = beta_start.to_vec();
        units.beta_from_internal(&mut beta);
        let fit = crate::glmm::separated_fit(ws, &beta, targets, map);
        let nb_theta = ws.nb_theta;
        return GlmmResultView {
            fit,
            nb_theta,
            ws,
            map,
            units,
            x: x_mat,
            z: z_mat,
            cluster_ids,
            extra_ids,
        };
    }
    // A held Gamma φ is a fixed value of the objective, not a coordinate. The
    // outer search runs on the internal `ŵ = w/s` scale (`prep_glmm_design`),
    // so a held raw-scale `v` enters there as `v/s`.
    ws.gamma_phi_held = match model.family {
        Family::Gamma { .. } => opts.dispersion.held().map(|v| units.held_dispersion_in(v)),
        _ => None,
    };
    // A carried Gamma φ (`StartValues::dispersion`) seeds the `ln φ`
    // coordinate, on the same internal scale as a held one. Set on every
    // call, so a reused workspace never keeps a previous draw's.
    ws.gamma_phi_start = match model.family {
        Family::Gamma { .. } => start
            .and_then(|s| s.dispersion)
            .map(|v| units.held_dispersion_in(v)),
        _ => None,
    };
    // Every search runs on `y/c` with `c` the draw's response scale
    // (`Units::c`). On the Gamma inverse link `c` takes a further factor read
    // off the data, and the search is retried at other factors when it fails
    // or ends no lower than the GLM point (`gamma_inverse_units`,
    // `gamma_inverse_retry`).
    let inverse_units = match model.family {
        Family::Gamma {
            link: crate::GammaLink::Inverse,
        } => Some(gamma_inverse_units(
            ws,
            x_mat,
            y,
            offset,
            n,
            cluster_ids,
            beta_start,
            beta_units.unwrap_or(1.0),
            uc,
        )),
        _ => None,
    };
    let c0 = inverse_units.as_ref().map_or(uc, |u| u.fast);
    // A curvature hint holds a search's BFGS state in that search's θ units
    // (`GlmmWorkspace::search_units`): a hint taken in other units than this
    // fit's first search is dropped.
    if ws
        .grad_hint_in
        .as_ref()
        .is_some_and(|h| h.units != theta_units(ws.family, c0))
    {
        ws.grad_hint_in = None;
    }
    let fit_in = |ws: &mut GlmmWorkspace, c: f64| {
        fit_glmm_in_units(
            ws,
            x_mat,
            z_mat,
            y,
            offset,
            n,
            p,
            cluster_ids,
            extra_ids,
            beta_start,
            beta_units,
            start,
            targets,
            opts,
            map,
            units,
            c,
        )
    };
    let restart_in = |ws: &mut GlmmWorkspace, restart: &StartValues| {
        fit_glmm_in_units(
            ws,
            x_mat,
            z_mat,
            y,
            offset,
            n,
            p,
            cluster_ids,
            extra_ids,
            &restart.beta,
            None,
            Some(restart),
            targets,
            opts,
            map,
            units,
            uc,
        )
    };
    let mut glmm_fit = fit_in(ws, c0);
    let has_units = inverse_units.is_some();
    if let Some(u) = inverse_units {
        glmm_fit = gamma_inverse_retry(ws, glmm_fit, c0, &u, fit_in);
    } else {
        glmm_fit = capped_restart(ws, glmm_fit, restart_in);
    }
    // A β start the search cannot leave is not caught by the start guard, which
    // compares θ candidates at the caller's β₀: a fit from one that failed,
    // stopped short or ended precision-limited is searched again from the cold
    // start (the GLM β, the blind θ, no carried Gamma φ), as a cold fit is, and
    // the cold search is kept when it ended lower without losing convergence.
    if !has_units
        && start.is_some_and(|s| !s.beta.is_empty())
        && (!glmm_fit.converged || !glmm_fit.deviance.is_finite() || glmm_fit.precision_limited)
    {
        let (cold_beta, sep) = glm_seed();
        if !sep {
            let (saved, nb_saved, phi_saved) =
                (SearchState::save(ws), ws.nb_theta, ws.gamma_phi_start);
            ws.nb_theta = nb_theta;
            ws.gamma_phi_start = None;
            let cold = fit_glmm_in_units(
                ws,
                x_mat,
                z_mat,
                y,
                offset,
                n,
                p,
                cluster_ids,
                extra_ids,
                &cold_beta,
                Some(uc),
                None,
                targets,
                opts,
                map,
                units,
                uc,
            );
            let mut cold = capped_restart(ws, cold, restart_in);
            let spent = glmm_fit.n_eval + cold.n_eval;
            let keep = cold.deviance.is_finite()
                && (cold.converged || !glmm_fit.converged)
                && (!glmm_fit.deviance.is_finite() || cold.deviance < glmm_fit.deviance);
            if keep {
                cold.n_eval = spent;
                glmm_fit = cold;
            } else {
                saved.restore(ws);
                ws.nb_theta = nb_saved;
                ws.gamma_phi_start = phi_saved;
                ws.grad_hint_ready = false;
                glmm_fit.n_eval = spent;
            }
        }
    }
    // θ̂_NB after the search (NaN on every other family, untouched by the kernel).
    let nb_theta = ws.nb_theta;
    GlmmResultView {
        fit: glmm_fit,
        nb_theta,
        ws,
        map,
        units,
        x: x_mat,
        z: z_mat,
        cluster_ids,
        extra_ids,
    }
}

/// One more search from the endpoint of a search its budget stopped (`fit`,
/// not converged at a finite deviance), warm-started there. A search that ran
/// out walking a long, flat valley (an intercept SD from about 0.7 to 80 on
/// precise Poisson counts) reaches its optimum from its own endpoint: the
/// gradient route takes the endpoint, whose free coordinates are already off
/// the trough that starved the first search. The restart is kept when it
/// converged or ended lower; otherwise the first search's state is put back.
/// `n_eval` counts both searches.
fn capped_restart(
    ws: &mut GlmmWorkspace,
    fit: GlmmFit,
    mut search: impl FnMut(&mut GlmmWorkspace, &StartValues) -> GlmmFit,
) -> GlmmFit {
    if fit.converged || !fit.deviance.is_finite() {
        return fit;
    }
    let (n_theta, p) = (ws.n_theta, ws.p);
    let restart = StartValues::new(
        ws.params[n_theta..n_theta + p].to_vec(),
        ws.params[..n_theta].to_vec(),
    );
    let (saved, nb_theta) = (SearchState::save(ws), ws.nb_theta);
    let mut again = search(ws, &restart);
    let spent = fit.n_eval + again.n_eval;
    if again.converged || again.deviance < fit.deviance {
        again.n_eval = spent;
        return again;
    }
    saved.restore(ws);
    ws.nb_theta = nb_theta;
    ws.grad_hint_ready = false;
    GlmmFit {
        n_eval: spent,
        ..fit
    }
}

/// What `run_glmm_on` reads off a Gamma-inverse GLMM's data before its first
/// search ([`gamma_inverse_units`]): response rescales `c` (the kernel fits
/// `y/c`, where η, β and θ are `c` times larger), each the draw's
/// `Units::c` times a factor read off the internal η, and the objective at
/// the GLM point.
struct GammaInverseUnits {
    /// `c` of the first search: brings the largest cluster η into `[8, 16]`.
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
/// below 1/64 or above 64. The first search runs on y rescaled into
/// `[8, 16]`, inside that band at any scale of y: on the grid's 24
/// Gamma-inverse cells it took the fewest evaluations there (`[4, 8]` and
/// `[16, 32]` took more). The σ = 0 optimum, with an η of order 1e-4, is
/// reached with the smallest start η in `[1/2, 64]`.
///
/// Read on `y/uc`, `uc` the draw's `Units::c`, so every `c` returned is
/// `uc` times its factor. `y` and `offset` are the caller's, `beta_start` in
/// the units of `y/beta_units`; the GLM point is in the caller's units.
#[allow(clippy::too_many_arguments)]
fn gamma_inverse_units(
    ws: &GlmmWorkspace,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    offset: Option<&[f64]>,
    n: usize,
    cluster_ids: &[u32],
    beta_start: &[f64],
    beta_units: f64,
    uc: f64,
) -> GammaInverseUnits {
    let fam = ws.family;
    let w = &ws.prior_w[..n];
    // η scales as `c` on this link; powers of two, so these maps are exact.
    let kb = uc / beta_units;
    let (mut eta_min, mut dev, mut sum_ln_y) = (f64::INFINITY, 0.0, 0.0);
    for i in 0..n {
        let mut xb = 0.0;
        for (j, &b) in beta_start.iter().enumerate() {
            xb += x_mat[(i, j)] * b;
        }
        let eta = offset.map_or(0.0, |o| uc * o[i]) + kb * xb;
        let yi = y[i] / uc;
        eta_min = eta_min.min(eta);
        dev += w[i] * crate::family::dev_resid_eta(fam, f64::NAN, yi, eta);
        sum_ln_y += yi.ln();
    }
    let weights = ws.weighted.then_some(w);
    // The stable doors refuse a held φ outside the box
    // (`fit::common::validate`); the clamp bounds the ML estimate, and a held
    // one from the unvalidated `loop_advanced` tier.
    let phi = ws
        .gamma_phi_held
        .unwrap_or_else(|| crate::family::gamma_ml_dispersion(dev, weights, n))
        .clamp(crate::consts::GAMMA_PHI_LO, crate::consts::GAMMA_PHI_HI);
    let glm_point = if eta_min > 0.0 {
        dev / phi
            + crate::family::gamma_dispersion_term(phi.ln(), weights, n, sum_ln_y)
            + 2.0 * n as f64 * uc.ln()
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
        wy[cluster_ids[i] as usize] += w[i] * (y[i] / uc);
        sw[cluster_ids[i] as usize] += w[i];
    }
    let eta_max = (0..levels)
        .filter(|&g| wy[g] > 0.0)
        .map(|g| sw[g] / wy[g])
        .fold(0.0, f64::max);
    let clusters = units_into_band(eta_max, 4.0, 8.0);
    GammaInverseUnits {
        fast: uc * units_into_band(eta_max, 8.0, 16.0),
        clusters: uc * clusters,
        glm: uc * units_into_band(eta_min, 0.5, 64.0),
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
/// workspace. `n_eval` counts every search; the gradient count `n_grad` does
/// too when the kept search ran the gradient search, whose stop and decrement
/// are reported, and is 0 otherwise.
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
    let mut n_grad = ws.grad_report.map_or(0, |r| r.1);
    for c in [u.glm, u.clusters, 2.0 * u.clusters] {
        if runs.iter().any(|r| r.0 == c) {
            continue;
        }
        saved.push(SearchState::save(ws));
        let fit = fit_in(ws, c);
        n_grad += ws.grad_report.map_or(0, |r| r.1);
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
    if let Some(r) = ws.grad_report.as_mut() {
        r.1 = n_grad;
    }
    // The BFGS state left is the last search's, which need not be the kept
    // one, so no curvature hint is taken from it.
    ws.grad_hint_ready = false;
    let mut kept = runs.swap_remove(pick).4;
    kept.n_eval = spent;
    kept
}

/// The workspace slots a search writes that are read after `run_glmm_on`
/// returns (by the [`GlmmResultView`] accessors and [`glmm_view_to_fit`]), so
/// [`gamma_inverse_retry`] and the cold fallback of a failed β start in
/// [`run_glmm_on`] can put back a search that did not run last. Every
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
    grad_report: Option<(crate::optim::bfgs::BfgsStop, usize, f64, u8)>,
    search_units: f64,
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
            grad_report: ws.grad_report,
            search_units: ws.search_units,
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
        ws.grad_report = self.grad_report;
        ws.search_units = self.search_units;
    }
}

/// The factor θ (and β, η and the offset) of a search on `y/c` take: `c` on
/// the Gamma inverse link, 1 on every other link, where the intercept absorbs
/// the rescale ([`fit_glmm_in_units`]).
fn theta_units(family: Family, c: f64) -> f64 {
    match family {
        Family::Gamma {
            link: crate::GammaLink::Inverse,
        } => c,
        _ => 1.0,
    }
}

/// One search of [`crate::glmm::fit_glmm`] on `y/c`, mapped back to y's units
/// on the workspace and the returned fit (`standardize::Units` derives each
/// map). On the Gamma inverse link η, β, θ and the offset are `c` times the
/// caller's. On the log link `y/c` moves η by `−ln c` and the centred offset
/// `o − m` (`Units::m`) by `−m`, which the intercept takes
/// (`GlmmWorkspace::intercept_shift`; a log-link fit without an intercept has
/// `c = 1` and `m = 0`), and θ does not move. The deviance takes `+2n·ln c`
/// and μ̂ the factor `c`. `y` and `offset` are the caller's, `start` is in the
/// caller's units and `beta_start` in those of a search on `y/beta_units`
/// (`None`: the caller's).
#[allow(clippy::too_many_arguments)]
fn fit_glmm_in_units(
    ws: &mut GlmmWorkspace,
    x_mat: faer::MatRef<'_, f64>,
    z_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    beta_start: &[f64],
    beta_units: Option<f64>,
    start: Option<&StartValues>,
    targets: &[u32],
    opts: &FitOptions,
    map: &ColumnScaling,
    units: &Units,
    c: f64,
) -> GlmmFit {
    let k = theta_units(ws.family, c);
    let m = units.m();
    if c == 1.0 && m == 0.0 && beta_units.is_none_or(|b| b == 1.0) {
        return crate::glmm::fit_glmm(
            ws,
            x_mat,
            z_mat,
            y,
            cluster_ids,
            extra_ids,
            targets,
            warm_theta(start),
            beta_start,
            n,
            opts.wald_se,
            map,
        );
    }
    let inverse = matches!(
        ws.family,
        Family::Gamma {
            link: crate::GammaLink::Inverse
        }
    );
    // What the caller's intercept less the internal one is on a search at
    // `c` (`ln c − m` on the log link), here and in `beta_start`'s units.
    let log_shift = |c: f64| if inverse { 0.0 } else { c.ln() - m };
    let shift = log_shift(c);
    let start_shift = beta_units.map_or(0.0, log_shift);
    let intercept = ws.intercept.filter(|_| shift != 0.0);
    debug_assert!(
        shift == 0.0 || intercept.is_some(),
        "a log-link search on y/c needs an intercept"
    );
    // Taken out of the workspace for the call, which borrows it whole; put
    // back below, so no draw allocates.
    let (mut y_fit, mut beta_fit, mut theta_fit) = (
        std::mem::take(&mut ws.y_units),
        std::mem::take(&mut ws.beta_units),
        std::mem::take(&mut ws.theta_units),
    );
    for (d, &v) in y_fit[..n].iter_mut().zip(y) {
        *d = v / c;
    }
    let kb = k / theta_units(ws.family, beta_units.unwrap_or(1.0));
    for (d, &b) in beta_fit[..p].iter_mut().zip(beta_start) {
        *d = b * kb;
    }
    if let Some(i) = ws.intercept.filter(|_| start_shift != shift) {
        beta_fit[i] -= shift - start_shift;
    }
    let theta = match warm_theta(start) {
        Some(t) => {
            for (d, &v) in theta_fit[..t.len()].iter_mut().zip(t) {
                *d = v * k;
            }
            Some(&theta_fit[..t.len()])
        }
        None => None,
    };
    if let Some(o) = ws.offset.as_mut().filter(|_| k != 1.0) {
        o.iter_mut().for_each(|v| *v *= k);
    }
    if let (Some(dst), Some(o)) = (
        ws.offset.as_mut().filter(|_| m != 0.0),
        units.offset(offset),
    ) {
        dst[..n].copy_from_slice(&o[..n]);
    }
    ws.intercept_shift = shift;
    let mut fit = crate::glmm::fit_glmm(
        ws,
        x_mat,
        z_mat,
        &y_fit[..n],
        cluster_ids,
        extra_ids,
        targets,
        theta,
        &beta_fit[..p],
        n,
        opts.wald_se,
        map,
    );
    ws.intercept_shift = 0.0;
    ws.search_units = k;
    (ws.y_units, ws.beta_units, ws.theta_units) = (y_fit, beta_fit, theta_fit);
    if let Some(i) = intercept {
        ws.params[ws.n_theta + i] += shift;
    }
    // Copied back rather than shifted by `+m`, which would round.
    if let (Some(dst), Some(o)) = (ws.offset.as_mut().filter(|_| m != 0.0), offset) {
        dst[..n].copy_from_slice(&o[..n]);
    }
    if k != 1.0 {
        if let Some(o) = ws.offset.as_mut() {
            o.iter_mut().for_each(|v| *v /= k);
        }
        let (n_theta, k2) = (ws.n_theta, k * k);
        ws.params[..n_theta + p].iter_mut().for_each(|v| *v /= k);
        ws.betas[..p].iter_mut().for_each(|v| *v /= k);
        for &t in targets {
            ws.inference.var_diag[t as usize] /= k2;
        }
        for j in 0..p {
            for i in 0..p {
                ws.inference.vcov[(i, j)] /= k2;
            }
        }
        for j in 0..n_theta {
            for i in 0..n_theta {
                ws.inference.theta_cov[(i, j)] /= k2;
            }
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
    view: &mut GlmmResultView<'_>,
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    opts: &FitOptions,
) -> (Fit, Vec<f64>, f64) {
    let diag = view.diagnostics();
    let converged = diag.converged;
    // First, as it re-solves the mode on the workspace (and puts back the
    // slots read below). In the search's units (`fit_glmm_in_units`): on the
    // Gamma inverse link the caller's η = 1/μ can lie outside the PIRLS
    // range. `k` is a power of two, so the scaling there and back is exact;
    // the blocks of `A⁻¹` do not move with it and its `A⁻¹MᵀWx` rows take `k`.
    let cond = if converged && opts.condvar {
        let ws = &mut *view.ws;
        let (k, m) = (ws.search_units, ws.n_theta + p);
        let mut y_fit = std::mem::take(&mut ws.y_units);
        let rescale = |ws: &mut GlmmWorkspace, f: f64| {
            ws.params[..m].iter_mut().for_each(|v| *v *= f);
            if let Some(o) = ws.offset.as_mut() {
                o.iter_mut().for_each(|v| *v *= f);
            }
        };
        if k != 1.0 {
            for (d, &v) in y_fit[..n].iter_mut().zip(y) {
                *d = v / k;
            }
            rescale(ws, k);
        }
        let out = crate::glmm::condvar_blocks(
            ws,
            view.x,
            view.z,
            if k != 1.0 { &y_fit[..n] } else { y },
            view.cluster_ids,
            view.extra_ids,
            p,
            n,
        );
        if k != 1.0 {
            rescale(ws, 1.0 / k);
        }
        ws.y_units = y_fit;
        out.map(|(blocks, mut rows, cross)| {
            rows.iter_mut().for_each(|v| *v *= k);
            (blocks, rows, cross)
        })
    } else {
        None
    };
    let ws = &*view.ws;
    let map = view.map;
    let glmm_fit = &view.fit;
    // Gated on the finite endpoint (the re-eval deviance), not `converged`: the
    // plateau policy `fit/lmm.rs` applies. A capped exit reports τ̂², D̂ and the
    // dispersion at its incumbent with `converged == false`; every other
    // non-convergence goes through `nan_fit`, whose deviance is +∞.
    let has_endpoint = view.has_endpoint();
    let nb_theta = view.nb_theta;
    let n_theta = ws.n_theta;

    // Map GlmmFit + workspace state → Fit.
    // ws.betas: length p, all fixed effects; ws.inference.var_diag: predictor-indexed.
    let beta = ws.betas.clone();
    let mut se = vec![f64::NAN; p];
    fill_se_by_predictor(&ws.inference.var_diag, &opts.se_for.columns(p), &mut se);

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
    let mut tau2: Vec<f64> = if has_endpoint {
        theta_caller.iter().map(|&t| t * t * sigma_sq).collect()
    } else {
        vec![f64::NAN; n_theta]
    };
    ws.groupings.drop_hidden(&mut tau2);

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
            Family::Gamma { .. } => match opts.dispersion.held() {
                Some(v) => v,
                None => view.units.dispersion_scale() * ws.gamma_phi,
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
        super::common::nan_varcorr(&ws.groupings)
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
    let mut stddev_se = if converged {
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
    ws.groupings.drop_hidden(&mut stddev_se);

    // `ws.inference.vcov` holds every column pair (the loop tier's contrasts
    // read it), while `ws.inference.var_diag` is written on the targets only.
    // Masking to the target block keeps `Fit::vcov` finite exactly where
    // `Fit::se` is, on both `Hessian` and `Rx`.
    let mut on = vec![false; p];
    for &t in opts.se_for.columns(p).iter() {
        on[t as usize] = true;
    }
    let vcov: Vec<f64> = (0..p * p)
        .map(|k| {
            let (i, j) = (k / p, k % p);
            if on[i] && on[j] {
                ws.inference.vcov[(i, j)]
            } else {
                f64::NAN
            }
        })
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
        (vec![f64::NAN; n], super::common::nan_ranef(&ws.groupings))
    };
    let (condvar, condvar_x, condvar_cross) = match cond {
        Some((blocks, rows, cross)) => {
            let theta = &ws.params[..n_theta];
            (
                super::common::assemble_condvar(theta, &ws.groupings, &blocks, 1.0, map),
                super::common::assemble_condvar_x(theta, &ws.groupings, &rows, p, map),
                cross.map(|c| {
                    super::common::assemble_condvar_cross(theta, &ws.groupings, c, 1.0, map)
                }),
            )
        }
        None => (vec![], vec![], None),
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
    let mut diagnostics = super::common::materialize_diagnostics(&diag, p, &varcorr);
    if has_endpoint {
        // The box applies to the search's θ, `search_units` times the θ in
        // y's units held here (`fit_glmm_in_units`).
        super::common::note_theta_at_limit(
            &mut diagnostics,
            &ws.params[..n_theta],
            ws.search_units,
        );
        if let Family::NegativeBinomial { .. } = model.family {
            super::common::note_nb_shape_at_limit(&mut diagnostics, nb_theta);
        }
    }
    if let Some(edge) = glmm_fit.phi_at_limit {
        diagnostics.notes.push(super::Note::DispersionAtLimit {
            limit: edge * view.units.dispersion_scale(),
            edge,
        });
    }
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
            super::common::model_df(
                model.family,
                p,
                ws.groupings.n_free_theta(),
                opts.dispersion.held().is_some(),
            )
        } else {
            0
        },
        reml: false,
        weight_power: f64::NAN,
        weight_power_estimated: false,
        se_type: SeType::Model,
        model_vcov: Vec::new(),
        robust: None,
        nobs: n,
        objective: glmm_objective(ws.nagq),
        fitted,
        ranef,
        ranef_levels,
        condvar,
        condvar_x,
        condvar_cross,
        coef_df: vec![f64::NAN; p],
        satterthwaite: None,
        design_summary: None,
        null_basis: Vec::new(),
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
/// Test-only, for [`fit_glmm_packed`].
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn fit_glmm_prebuilt(
    ws: &mut GlmmWorkspace,
    x_mat: faer::MatRef<f64>,
    z_mat: faer::MatRef<f64>,
    y: &[f64],
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    nb_theta: f64,
    start: Option<&StartValues>,
    opts: &FitOptions,
    map: &ColumnScaling,
    units: &Units,
) -> (Fit, Vec<f64>, f64) {
    let mut view = run_glmm_on(
        ws,
        x_mat,
        z_mat,
        y,
        offset,
        n,
        p,
        model,
        cluster_ids,
        extra_ids,
        nb_theta,
        start,
        &opts.se_for.columns(p),
        opts,
        map,
        units,
    );
    glmm_view_to_fit(&mut view, y, n, p, model, opts)
}

/// Negative-binomial GLMM: the dispersion θ_NB is a coordinate of the outer
/// search on the marginal objective `dev − 2·nb_profile_loglik(y, y, θ_NB, w)`
/// (`glmm::fit_glmm`), so β, θ_RE and θ_NB come out of one fit — the same optimum
/// `lme4::glmer.nb`'s outer `optimize()` over re-fitted GLMMs reaches, without
/// the re-fits. The coordinate cold-starts at the no-RE GLM-NB's own θ̂ (one
/// extra fixed-effects-only `fit_glm_nb`, itself an IRLS/θ-profile alternation
/// capped at `NB_MAX_OUTER`), or at a carried `start.dispersion`; a caller's
/// `start` seeds β/θ_RE as on every other family. θ̂_NB is the view's
/// dispersion; the joint Hessian of `WaldSe::Hessian` carries the `ln θ_NB`
/// coordinate, so the β SE includes θ_NB's uncertainty (unlike the GLM route's
/// `fit_glm_nb`, which conditions on θ̂ — see
/// [`crate::glmm::se::joint_hessian_cov`]).
///
/// `ws` is built once per shape by the unified core
/// (`GlmmWorkspace::for_cluster_spec_ext` on [`nb_spec`] at the build shape)
/// and reused across calls: [`prep_glmm_design`] runs here per call, as the
/// core's `FitKind::Glmm` arm runs it for the other families, and `fit_glmm`
/// resets every per-fit state at its top, so a reused workspace gives the fit
/// a fresh one gives. `x` is the column-major design the kernel fits (`n`
/// rows) and `z` its random-effect-only columns, standardized through `cs`;
/// `start` is already on that design. The caller runs the degenerate guard.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_glmm_nb_on<'a>(
    ws: &'a mut GlmmWorkspace,
    x: faer::MatRef<'a, f64>,
    z: faer::MatRef<'a, f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    model: &ModelSpec,
    cluster_ids: &'a [u32],
    extra_ids: &'a [Vec<u32>],
    start: Option<&StartValues>,
    targets: &[u32],
    opts: &FitOptions,
    cs: &'a ColumnScaling,
    units: &'a Units,
) -> GlmmResultView<'a> {
    // Exposure weights, which the NB units leave as given.
    prep_glmm_design(ws, cluster_ids, extra_ids, weights, offset, n, opts);
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
    // A carried shape (`StartValues::dispersion`) is the seed itself, and the
    // prefit that exists only to produce one is skipped.
    let nb_seed = match start.and_then(|s| s.dispersion) {
        Some(d) => d,
        None => {
            let (prefit, prefit_theta) =
                super::glm::fit_glm_nb(x, y, weights, units.offset(offset), n, p, None, opts, cs);
            nb_glmm_seed(prefit.converged(), prefit_theta)
        }
    };
    run_glmm_on(
        ws,
        x,
        z,
        y,
        offset,
        n,
        p,
        &nb_spec(model),
        cluster_ids,
        extra_ids,
        nb_seed,
        start,
        targets,
        opts,
        cs,
        units,
    )
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

#[cfg(test)]
mod retry_tests {
    use super::*;
    use crate::SeFor;

    /// What [`gamma_inverse_retry`] reports when it runs more than the first
    /// search. The routed `gamma_inverse_slope_design`, `(1 + x₁ | g)`
    /// (n_θ = 3), runs its real searches in each unit; only the reported
    /// deviances are set, so that the search in `glm` units (second of four,
    /// put back) or the last one is kept. Every search leaves a hint; after
    /// the retry there is none in either case, since the BFGS state left is
    /// the last search's, which need not be the kept one. The gradient
    /// search's stop and decrement are the
    /// kept search's own, and `n_grad` is the sum over all four searches.
    #[test]
    fn gamma_inverse_retry_reports_the_kept_search_and_drops_the_hint() {
        let (x, y, ids, model) = super::super::glmm_tests::gamma_inverse_slope_design();
        let (n, p) = (y.len(), 2usize);
        let opts = FitOptions {
            se_for: SeFor::Only(vec![0, 1]),
            ..FitOptions::default()
        };
        let (sized, sids, _) = super::super::common::spec_sized_from_ids(&model, &ids);
        let beta = [0.8, -0.3];
        let u = GammaInverseUnits {
            fast: 1.0,
            clusters: 2.0,
            glm: 0.5,
            glm_point: f64::INFINITY,
        };
        for kept in [u.glm, 2.0 * u.clusters] {
            let Ok((mut ws, x_mat, _)) =
                fit_glmm_build(&x, &y, None, None, n, p, &sized, &sids.primary, &[], &opts)
            else {
                panic!("the design did not build");
            };
            let mut reports = Vec::new();
            let mut fit_in = |ws: &mut GlmmWorkspace, c: f64| {
                let mut f = fit_glmm_in_units(
                    ws,
                    x_mat.as_ref(),
                    x_mat.as_ref().subcols(0, 0),
                    &y,
                    None,
                    n,
                    p,
                    &sids.primary,
                    &[],
                    &beta,
                    None,
                    None,
                    &opts.se_for.columns(p),
                    &opts,
                    ColumnScaling::IDENTITY,
                    Units::IDENTITY,
                    c,
                );
                assert!(ws.grad_hint_ready, "c = {c}: the search left no hint");
                reports.push((c, ws.grad_report.expect("the search routes")));
                f.converged = true;
                // Far below any real objective here, so the kept search is
                // also the lowest point seen and no failed search is preferred.
                f.deviance = if c == kept { -1e6 } else { 0.0 };
                f
            };
            let mut fast = fit_in(&mut ws, u.fast);
            fast.converged = false;
            let _ = gamma_inverse_retry(&mut ws, fast, u.fast, &u, fit_in);
            assert_eq!(reports.len(), 4, "kept c = {kept}");
            assert_eq!(ws.search_units, kept, "kept c = {kept}: search units");
            assert!(
                crate::glmm::glmm_curvature_hint(&ws).is_none(),
                "kept c = {kept}: a hint is left"
            );
            let own = reports.iter().find(|r| r.0 == kept).expect("kept ran").1;
            let last = reports[3].1;
            if kept != 2.0 * u.clusters {
                assert_ne!(
                    own.2.to_bits(),
                    last.2.to_bits(),
                    "the kept and the last search end alike, so the check is empty"
                );
            }
            let got = ws.grad_report.expect("a report");
            assert_eq!(got.0, own.0, "kept c = {kept}: stop");
            assert_eq!(
                got.2.to_bits(),
                own.2.to_bits(),
                "kept c = {kept}: decrement"
            );
            let sum: usize = reports.iter().map(|r| r.1 .1).sum();
            assert_eq!(got.1, sum, "kept c = {kept}: n_grad");
        }
    }
}
