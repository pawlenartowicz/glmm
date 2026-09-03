//! Unified fit core: one dispatch body shared by `fit_cold`/`fit_warm` and the
//! `loop_advanced` hot path, so a change to glmm's model/routing lands in every
//! caller identically and no caller can re-implement the routing and drift.
//!
//! Two reusable pieces, mirroring the existing `fit_glmm_build`/`_prebuilt`
//! split: [`build_workspace`] classifies the design once (family × RE × solver)
//! and allocates the per-shape [`FitWorkspace`]; [`fit_on`] accumulates + solves
//! on that workspace and returns a lean borrowed [`FitView`]. `fit_warm` is
//! re-expressed over these (build a throwaway ws, `fit_on`, `FitView::into_fit`);
//! the loop tier builds once per shape and calls `fit_on` per draw.
//!
//! Always compiled (stable `fit_warm` is re-expressed over it); only the
//! re-exports through `crate::loop_advanced` are feature-gated.

use faer::Mat;

use crate::glmm::{build_z, GlmmWorkspace, StructuredSchur};
use crate::lmm::LmmWorkspace;
use crate::{BinomialLink, Family, GroupIds, GroupingRelation, ModelSpec, StartValues};

use super::common::{
    assert_group_ids, assert_model_shape, fill_col_major, unpermute_fit, warm_theta,
    FitDiagnostics, Perm,
};
use super::glm::GlmScratchBuf;
use super::lmm::LmmResultView;
use super::ols::OlsWorkspace;
use super::{classify_design, Fit, FitOptions, Solver};

// ---------------------------------------------------------------------------
// FitView — the lean, caller-owned result
// ---------------------------------------------------------------------------

/// Borrowed result of [`fit_on`]. Reads (`t_sq`/`converged`/`betas`/`var_diag`)
/// come straight off the workspace slots the fit wrote — no `Fit` allocation on
/// the hot path. Call [`FitView::into_fit`] for the full stable `Fit` (vcov,
/// loglik, varcorr, fitted, ranef).
pub struct FitView<'a> {
    kind: FitViewKind<'a>,
    /// Grouping reorder the workspace was built under ([`FitWorkspace::perm`]).
    perm: Perm,
    /// θ̂ mapped back to the caller's declaration order AND out of the solver's
    /// internal RE column scaling. Filled only when `perm` reorders something or
    /// some RE design column carries a scale other than exactly 1 — empty
    /// otherwise, and then [`FitView::theta`] hands back the kernel's own slice
    /// with no copy.
    ///
    /// Borrowed from [`FitWorkspace::theta_declared_buf`] rather than owned: the
    /// scales are a function of the DRAW's design (`set_slope_scales` recomputes
    /// them per call), so they cannot be frozen at build — but the storage can,
    /// and that is what keeps a warm random-slope loop off the heap.
    theta_declared: &'a [f64],
}

// The `Prebuilt` arm is ~2× the next-largest variant because it owns a whole
// assembled `Fit` — which is the arm's entire reason to exist, and got wider
// again when `Fit`'s diagnostics moved behind `Diagnostics`. Clippy's fix is to
// box it, and that is the wrong trade here: the enum lives for one call frame,
// while `Box::new` would add a heap block PER DRAW on the loop tier, which is
// the one cost this crate spent 0.1.3 removing.
#[allow(clippy::large_enum_variant)]
enum FitViewKind<'a> {
    Ols(crate::ols::OlsFitView<'a>),
    Glm(crate::glm::GlmFitView<'a>),
    Lmm(LmmResultView<'a>),
    Glmm(super::glmm::GlmmResultView<'a>),
    /// NB outer-loops (GLM-NB, GLMM-NB) and every sparse-routed design: the
    /// kernel already assembled a `Fit`. `t_sq`/`var_diag` are reconstructed
    /// predictor-indexed from β̂/se for the accessor surface (Wald t²_j =
    /// (β̂_j/se_j)², Var = se_j²) — a best-effort convenience; `into_fit` returns
    /// the kernel's `Fit` verbatim.
    Prebuilt {
        fit: Fit,
        // Loop-tier read surface (see the accessor impl) — dead in a default
        // single-fit build, live under `loop_advanced`/tests.
        #[allow(dead_code)]
        t_sq: Vec<f64>,
        #[allow(dead_code)]
        var_diag: Vec<f64>,
    },
}

// `t_sq`/`converged`/`betas`/`var_diag` are the loop-tier hot-path reads
// (MCPower's warm loop, via the `loop_advanced` re-export). Unused by the stable
// single-fit path (which calls `into_fit`), so they read as dead code in a
// default build; `into_fit` in the same surface stays live.
#[allow(dead_code)]
impl FitView<'_> {
    /// Per-target Wald statistic — the hot-loop significance read. Layout follows
    /// each estimator's native convention (OLS/GLM target-compact `[0..t]`;
    /// LMM/GLMM/Prebuilt predictor-indexed `[0..p]`), as elsewhere in the crate.
    pub fn t_sq(&self) -> &[f64] {
        match &self.kind {
            FitViewKind::Ols(v) => v.t_sq,
            FitViewKind::Glm(v) => v.t_sq,
            FitViewKind::Lmm(v) => v.t_sq(),
            FitViewKind::Glmm(v) => v.t_sq(),
            FitViewKind::Prebuilt { t_sq, .. } => t_sq,
        }
    }

    /// Every diagnostic this fit produced, in one carrier — the single channel
    /// each route fills and everything downstream reads (see [`FitDiagnostics`]
    /// for what each field means per route, and for the two placeholder fields).
    /// Allocation-free, so a warm loop can read it per draw.
    ///
    /// This is the loop tier's ONLY window onto rank state, and reading it every
    /// draw is the tier's price of admission. The warm-loop entry points skip
    /// the pre-dispatch alias gate `fit_cold`/`fit_warm` run — the ONLY place a
    /// column is ever dropped — deliberately, to buy the speed the tier exists
    /// for. The design therefore reaches the solver as-is, and one of two things
    /// comes back: refusal (NaN β̂, `converged: false`, no pivot recorded), or
    /// acceptance (`converged: true`, a finite β̂, an enormous standard error,
    /// `ill_conditioned` set).
    ///
    /// Which one is settled by the arithmetic on that draw. It is not
    /// predictable from the design or from the route: one route, handed the same
    /// kind of duplicated column at different sample sizes, has been measured
    /// landing on opposite sides.
    ///
    /// So the obligation transfers to the caller. Read `converged` AND
    /// `ill_conditioned` on every draw and decide what the draw is worth — a
    /// returned number is not evidence of a good fit, and counting only
    /// non-converged draws misses every draw that fit through badly conditioned.
    pub fn diagnostics(&self) -> FitDiagnostics {
        match &self.kind {
            FitViewKind::Ols(v) => v.diagnostics(),
            FitViewKind::Glm(v) => v.diagnostics(),
            FitViewKind::Lmm(v) => v.diagnostics(),
            FitViewKind::Glmm(v) => v.diagnostics(),
            // The kernel already assembled a `Fit`, so the carrier is read back
            // off it. `boundary_hit` is back-derived from `singular`, which is
            // lossy in the documented way: this route cannot report a cap-out
            // (2), and the carrier holds no `Vec`, so the per-component detail
            // stays on the assembled `Fit` (`Diagnostics::pinned`, filled by
            // the sparse routes) and does not reach here. Its `singular` also
            // carries the negligible-component check `into_fit` applies to the
            // other arms afterwards, so it is at least as inclusive.
            FitViewKind::Prebuilt { fit, .. } => FitDiagnostics {
                boundary_hit: fit.singular() as u8,
                ..FitDiagnostics::fixed_only(fit.converged())
            },
        }
    }

    /// Whether the fit reached its convergence criterion. Forwards to
    /// [`FitView::diagnostics`] — the most-read diagnostic keeps its one-hop
    /// accessor rather than a five-arm match of its own.
    pub fn converged(&self) -> bool {
        self.diagnostics().converged
    }

    /// Fixed-effect estimates β̂ (length `p`).
    pub fn betas(&self) -> &[f64] {
        match &self.kind {
            FitViewKind::Ols(v) => v.betas,
            FitViewKind::Glm(v) => v.betas,
            FitViewKind::Lmm(v) => v.betas(),
            FitViewKind::Glmm(v) => v.betas(),
            FitViewKind::Prebuilt { fit, .. } => &fit.beta,
        }
    }

    /// Per-coefficient Var(β̂). Same compact/indexed layout split as `t_sq`.
    pub fn var_diag(&self) -> &[f64] {
        match &self.kind {
            FitViewKind::Ols(v) => v.var_diag,
            FitViewKind::Glm(v) => v.var_diag,
            FitViewKind::Lmm(v) => v.var_diag(),
            FitViewKind::Glmm(v) => v.var_diag(),
            FitViewKind::Prebuilt { var_diag, .. } => var_diag,
        }
    }

    /// Joint Wald-χ² over the target set (the omnibus significance read). Only
    /// the mixed arms compute it; OLS/GLM (never read here) and the `Prebuilt`
    /// arm (`Fit` carries no joint statistic) report NaN.
    pub fn joint_t_sq(&self) -> f64 {
        match &self.kind {
            FitViewKind::Ols(_) | FitViewKind::Glm(_) => f64::NAN,
            FitViewKind::Lmm(v) => v.joint_t_sq(),
            FitViewKind::Glmm(v) => v.joint_t_sq(),
            FitViewKind::Prebuilt { .. } => f64::NAN,
        }
    }

    /// Objective evaluations the solve spent. OLS is closed-form (0); the
    /// `Prebuilt` arm forwards the kernel's `Fit.n_eval`. GLM reports 0 by
    /// choice, not by omission — it has no objective to evaluate, and its
    /// `GlmFitView::n_iter` counts IRLS steps, which is a different quantity
    /// from the θ-search evaluation count every other arm reports here.
    pub fn n_eval(&self) -> usize {
        match &self.kind {
            FitViewKind::Ols(_) | FitViewKind::Glm(_) => 0,
            FitViewKind::Lmm(v) => v.n_eval(),
            FitViewKind::Glmm(v) => v.n_eval(),
            FitViewKind::Prebuilt { fit, .. } => fit.n_eval,
        }
    }

    /// Estimator dispersion: LMM σ̂², GLMM D̂[0][0], `Prebuilt` `Fit.dispersion`.
    /// OLS/GLM (never read here) report NaN.
    pub fn dispersion(&self) -> f64 {
        match &self.kind {
            FitViewKind::Ols(_) | FitViewKind::Glm(_) => f64::NAN,
            FitViewKind::Lmm(v) => v.dispersion(),
            FitViewKind::Glmm(v) => v.dispersion(),
            FitViewKind::Prebuilt { fit, .. } => fit.dispersion,
        }
    }

    /// Fitted θ̂ vech (primary block then extras, column-major lower-triangular) —
    /// feeds the grid-sequential warm-start carry. Empty for the `Prebuilt` arm
    /// (sparse route holds no exposed θ̂) and for OLS/GLM.
    ///
    /// In the caller's DECLARATION order, matching what [`crate::StartValues`]
    /// is read in, so carrying θ̂ from one fit into the next needs no mapping.
    pub fn theta(&self) -> &[f64] {
        if self.theta_declared.is_empty() {
            self.kernel_theta()
        } else {
            self.theta_declared
        }
    }

    /// The θ-carrying routes' grouping structure — the only thing that knows the
    /// internal RE column scales θ̂ has to be divided by. `None` where there is no
    /// θ (OLS/GLM) or none exposed (`Prebuilt`).
    fn kernel_groupings(&self) -> Option<&crate::lmm::LmmGroupings> {
        match &self.kind {
            FitViewKind::Lmm(v) => Some(v.groupings()),
            FitViewKind::Glmm(v) => Some(v.groupings()),
            FitViewKind::Ols(_) | FitViewKind::Glm(_) | FitViewKind::Prebuilt { .. } => None,
        }
    }

    /// θ̂ as the kernel holds it — in `perm`'s slot order, which is declaration
    /// order only when `perm` is the identity.
    fn kernel_theta(&self) -> &[f64] {
        match &self.kind {
            FitViewKind::Ols(_) | FitViewKind::Glm(_) => &[],
            FitViewKind::Lmm(v) => v.theta(),
            FitViewKind::Glmm(v) => v.theta(),
            FitViewKind::Prebuilt { .. } => &[],
        }
    }

    /// Build the full stable `Fit` (vcov, loglik, varcorr, fitted, ranef). This
    /// is the allocating stable-API path; the hot loop reads the accessors above
    /// instead. `model` selects the family for the GLM/GLMM mappers.
    #[allow(clippy::too_many_arguments)] // marshals (x, y, ids, n, p, model, opts)
    pub fn into_fit(
        self,
        x: &[f64],
        y: &[f64],
        ids: &GroupIds,
        n: usize,
        p: usize,
        model: &ModelSpec,
        opts: &FitOptions,
    ) -> Fit {
        let perm = self.perm;
        let mut fit = match self.kind {
            FitViewKind::Ols(v) => super::ols::ols_view_to_fit(&v, x, y, n, p, opts),
            FitViewKind::Glm(v) => {
                super::glm::glm_view_to_fit(&v, y, model.family, f64::NAN, n, p, opts)
            }
            FitViewKind::Lmm(v) => super::lmm::lmm_view_to_fit(&v, x, ids, n, p, opts),
            FitViewKind::Glmm(v) => super::glmm::glmm_view_to_fit(&v, y, n, p, model, opts).0,
            FitViewKind::Prebuilt { fit, .. } => fit,
        };
        // `ids` above is the SLOT-order companion of the sized spec (`fitted`
        // sums over all groupings, so it is order-blind), while `model` is the
        // caller's own declaration-order spec — which is the order every
        // grouping-indexed field of the returned `Fit` is expected in.
        unpermute_fit(perm, &mut fit);
        fit
    }
}

impl<'a> FitView<'a> {
    /// Wrap a solver's borrowed result under the workspace's grouping reorder,
    /// materializing the declaration-order θ̂ once here so the hot-path accessor
    /// stays a plain slice read.
    fn new(kind: FitViewKind<'a>, perm: Perm, buf: &'a mut [f64]) -> Self {
        let mut view = FitView {
            kind,
            perm,
            theta_declared: &[],
        };
        // The kernels minimize over the internally scaled θ̃ = s·θ
        // (`LmmGroupings::set_slope_scales`); a caller carries θ̂ forward as a warm
        // start, which is read in the design's own units, so the carry has to come
        // out divided. Nothing is materialized when every scale is exactly 1 and
        // the reorder is the identity — the intercept-only case, which is most of
        // them — and then `theta()` hands back the kernel's own slice.
        let scaled = view
            .kernel_groupings()
            .is_some_and(|g| g.any_slope_scaled());
        if scaled || !perm.is_identity() {
            let n_theta = view.kernel_theta().len();
            let theta = &mut buf[..n_theta];
            theta.copy_from_slice(view.kernel_theta());
            if scaled {
                // Stack-sized off the θ ceiling every dense route is bounded by
                // (`classify_design` sends anything wider to the sparse path,
                // which exposes no θ here), so the divide costs no heap block.
                let mut scales = [0.0f64; crate::consts::MAX_THETA];
                let scales = &mut scales[..n_theta];
                view.kernel_groupings()
                    .expect("scaled implies a θ-carrying route")
                    .fill_theta_row_scales(scales);
                for (t, &sc) in theta.iter_mut().zip(scales.iter()) {
                    *t /= sc;
                }
            }
            // Scales are in kernel slot order, so they apply before the reorder.
            perm.swap_slots(theta);
            view.theta_declared = &buf[..n_theta];
        }
        view
    }
}

/// Predictor-indexed Wald t² and Var(β̂_j) reconstructed from an assembled `Fit`
/// for the [`FitViewKind::Prebuilt`] accessor surface. Slots whose `se` is not
/// finite (non-target, non-converged) stay NaN.
fn prebuilt_stats(fit: &Fit) -> (Vec<f64>, Vec<f64>) {
    let p = fit.beta.len();
    let mut t_sq = vec![f64::NAN; p];
    let mut var_diag = vec![f64::NAN; p];
    for j in 0..p {
        let se = fit.se[j];
        if se.is_finite() {
            var_diag[j] = se * se;
            t_sq[j] = (fit.beta[j] / se).powi(2);
        }
    }
    (t_sq, var_diag)
}

// ---------------------------------------------------------------------------
// FitWorkspace — classify once, allocate the variant
// ---------------------------------------------------------------------------

/// Per-shape fit workspace produced by [`build_workspace`]. The route is pinned
/// to the build shape (spec, `n_max`, `p`) and the frozen options below, and
/// `fit_on` hard-panics on any per-call mismatch. The panic is the point: the
/// accumulators index the RE buffers by GLOBALIZED level id
/// (`extra_offsets[g] + id·q_g`), so a draw whose level counts outgrow the build
/// shape scatters into a neighbouring grouping's column block — a silently wrong
/// answer, not a bounds panic. Only ids past the whole buffer trip the `Vec`
/// bounds check; the per-id guard in `LmmSuffStats::add_rows_multi` is
/// `debug_assert`-only.
pub struct FitWorkspace {
    n_max: usize,
    p: usize,
    /// Level-count-baked spec (from `spec_sized_from_ids`); fixed by the build.
    sized: ModelSpec,
    /// The grouping reorder `sized` carries, so every result read off this
    /// workspace can be reported in the caller's declaration order. Frozen at
    /// build like the spec itself — the two are one unit, and pairing them here
    /// is what stops a caller sizing a spec and then reading θ̂ as if it had not
    /// been reordered.
    perm: Perm,
    /// Primary RE level count baked at build (0 for fixed-only). The per-call
    /// shape pin compares each draw's `max(ids.primary)+1` against this.
    build_primary_levels: usize,
    /// Per-extra-grouping level capacity baked at build, declaration order.
    /// Empty for fixed-only and for specs with no extra groupings.
    build_extra_capacity: Vec<usize>,
    // Frozen options — a per-call `opts` must match these (buffer presence / grid
    // size are baked into the allocated workspace).
    nagq: u8,
    /// `opts.target_indices.len()`: the OLS/GLM result slots are sized off it, so
    /// a later call asking for more targets would overrun them.
    n_targets: usize,
    has_weights: bool,
    has_offset: bool,
    parallel_inner: bool,
    /// Storage for [`FitView::theta_declared`], sized once at build. Sibling of
    /// `kind` for the same borrow reason `x_mat` is (a slice of the solver
    /// workspace could not also be handed out alongside `&mut` to it).
    theta_declared_buf: Vec<f64>,
    /// Slot-order copy of a caller's warm start, refilled per call on a
    /// reordering workspace. Its capacities are the only reason permuting a
    /// start costs no heap block, so never shrink or reallocate them.
    start_permuted: StartValues,
    kind: FitKind,
}

// The variants hold build-once per-shape solver workspaces (real buffer state),
// so their sizes differ by design; a `FitWorkspace` is held singly per shape,
// never in a `Vec`, so boxing the large arms would add a hot-path deref for no
// memory win.
#[allow(clippy::large_enum_variant)]
enum FitKind {
    // `x_mat` is a sibling of the workspace, not a field of it: both
    // `fit_ols_prebuilt`/`fit_glm_prebuilt` take `&mut Ols/GlmScratchBuf` and
    // return a view borrowing from it, so a `MatRef` sourced from a field of
    // that same workspace would be a second mutable borrow for the whole call
    // — a hard error, not a lifetime puzzle. Sibling fields give `fit_on`
    // disjoint borrows instead, exactly as `GlmmDense` already does below.
    Ols {
        ws: OlsWorkspace,
        x_mat: Mat<f64>,
    },
    Glm {
        buf: GlmScratchBuf,
        x_mat: Mat<f64>,
    },
    LmmDense {
        ws: LmmWorkspace,
        x_mat: Mat<f64>,
        // Offset-shifted y, filled only when `opts.offset` is set (per-call,
        // not frozen at build — unlike `x_mat` this buffer sits unused and
        // untouched on every offset-less call). Sibling of `ws` for the same
        // borrow reason as `x_mat`: `accumulate_lmm_rows` takes `ws` mutably,
        // so a slice of `ws` cannot also be its `y` argument.
        y_shifted: Vec<f64>,
    },
    GlmmDense {
        ws: GlmmWorkspace,
        x_mat: Mat<f64>,
    },
    /// Routes whose kernel allocates per call and returns a fully-assembled
    /// `Fit`: the NB outer-θ loops (GLM-NB, GLMM-NB) and every sparse-routed
    /// design. `build_workspace` pins the exact kernel here (classify once), but
    /// the buffers are still allocated per call inside `fit_on` — these routes
    /// get the routing guarantee without the workspace-reuse win.
    Prebuilt(PrebuiltRoute),
}

#[derive(Clone, Copy)]
enum PrebuiltRoute {
    GlmNb,
    GlmmNbDense,
    LmmSparse,
    GlmmSparse,
    GlmmNbSparse,
}

#[cfg(test)]
impl FitWorkspace {
    pub(crate) fn is_ols(&self) -> bool {
        matches!(self.kind, FitKind::Ols { .. })
    }
    pub(crate) fn is_glm(&self) -> bool {
        matches!(self.kind, FitKind::Glm { .. })
    }
    pub(crate) fn is_lmm_dense(&self) -> bool {
        matches!(self.kind, FitKind::LmmDense { .. })
    }
    pub(crate) fn is_glmm_dense(&self) -> bool {
        matches!(self.kind, FitKind::GlmmDense { .. })
    }
    pub(crate) fn is_prebuilt(&self) -> bool {
        matches!(self.kind, FitKind::Prebuilt(_))
    }
}

/// Classify the (already level-count-sized) design once and allocate its
/// workspace. `sized` and `perm` must be the first and third values
/// `spec_sized_from_ids(model, ids)` returned together (or, for a spec whose RE
/// level counts already match the data and was never reordered, that spec and
/// [`Perm::IDENTITY`]). Fixed-only (`re: None`) bypasses `classify_design`,
/// exactly as `fit_warm`'s `(_, None)` arm does.
pub fn build_workspace(
    sized: &ModelSpec,
    perm: Perm,
    n_max: usize,
    p: usize,
    opts: &FitOptions,
) -> FitWorkspace {
    // The same spec-shape gate `fit_warm` runs at its entry, so the loop tier
    // cannot build a workspace for a shape the kernels do not support. The
    // load-bearing one here is the ≤1 `NestedWithin` rule: `classify_design` does
    // not count nestings, so without this a second nested grouping would route
    // `NoZ` and silently share the first one's RE-column block.
    assert_model_shape(sized, p, opts.nagq);
    let t = opts.target_indices.len();
    let kind = match (&sized.family, sized.re.as_ref()) {
        (Family::Gaussian, None) => FitKind::Ols {
            ws: OlsWorkspace::new(n_max, p, t, opts.weights.is_some()),
            x_mat: Mat::<f64>::zeros(n_max.max(1), p.max(1)),
        },
        (Family::NegativeBinomial { .. }, None) => FitKind::Prebuilt(PrebuiltRoute::GlmNb),
        // Spelled out rather than `(_, None)` so adding a family (or a binomial
        // link) is a compile error here instead of silently routing to IRLS.
        (
            Family::Poisson { .. }
            | Family::Gamma { .. }
            | Family::InverseGaussian { .. }
            | Family::Binomial {
                link: BinomialLink::Probit | BinomialLink::Logit | BinomialLink::Cloglog,
            },
            None,
        ) => FitKind::Glm {
            buf: GlmScratchBuf::new(n_max, p, t),
            x_mat: Mat::<f64>::zeros(n_max.max(1), p.max(1)),
        },
        (family, Some(re)) => match classify_design(sized, opts.nagq) {
            Solver::NoZ => match family {
                Family::Gaussian => {
                    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
                    let extra_slope_cols: Vec<Vec<usize>> = re
                        .extra_groupings
                        .iter()
                        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
                        .collect();
                    FitKind::LmmDense {
                        ws: LmmWorkspace::for_cluster_spec_ext(
                            p,
                            sized,
                            n_max,
                            &slope_cols,
                            &extra_slope_cols,
                        ),
                        x_mat: Mat::<f64>::zeros(n_max.max(1), p.max(1)),
                        y_shifted: vec![0.0f64; n_max.max(1)],
                    }
                }
                Family::NegativeBinomial { .. } => FitKind::Prebuilt(PrebuiltRoute::GlmmNbDense),
                _ => {
                    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
                    let ws =
                        GlmmWorkspace::for_cluster_spec(p, sized, n_max, &slope_cols, opts.nagq);
                    let x_mat = Mat::<f64>::zeros(n_max.max(1), p.max(1));
                    FitKind::GlmmDense { ws, x_mat }
                }
            },
            Solver::Sparse => match family {
                Family::Gaussian => FitKind::Prebuilt(PrebuiltRoute::LmmSparse),
                Family::NegativeBinomial { .. } => FitKind::Prebuilt(PrebuiltRoute::GlmmNbSparse),
                _ => FitKind::Prebuilt(PrebuiltRoute::GlmmSparse),
            },
        },
    };
    // `n_clusters_at` — NOT the raw sizing field: under `FixedSize` the field is
    // rows-per-cluster, so reading it directly would pin the shape against a row
    // count. `spec_sized_from_ids` only ever emits `FixedClusters`, so this arm
    // is reachable only from a loop-tier caller that sizes its own spec.
    let build_primary_levels = sized
        .re
        .as_ref()
        .map(|re| re.sizing.n_clusters_at(n_max))
        .unwrap_or(0);
    // Level capacity of each extra grouping's RE-column block, derived the same
    // way `spec_sized_from_ids` sizes the spec: a crossed factor owns its own
    // level count, a nested one owns parents × children-per-parent. The `.max(1)`
    // mirrors the builders' own clamps (`LmmGroupings::from_cluster_spec_ext`), so
    // a hand-built spec declaring 0 does not get a capacity below its real width.
    // The pin is route-agnostic even though only the dense routes preallocate: the
    // sparse/NB `Prebuilt` routes build per call, but from `ws.sized` — the BUILD
    // shape — so an over-capacity draw is undersized there too.
    let build_extra_capacity: Vec<usize> = sized
        .re
        .as_ref()
        .map(|re| {
            re.extra_groupings
                .iter()
                .map(|g| match g.relation {
                    GroupingRelation::Crossed { n_clusters } => n_clusters.max(1) as usize,
                    GroupingRelation::NestedWithin { n_per_parent } => {
                        build_primary_levels * n_per_parent.max(1) as usize
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    FitWorkspace {
        n_max,
        p,
        sized: sized.clone(),
        perm,
        build_primary_levels,
        build_extra_capacity,
        nagq: opts.nagq,
        n_targets: t,
        has_weights: opts.weights.is_some(),
        has_offset: opts.offset.is_some(),
        parallel_inner: opts.parallel_inner,
        theta_declared_buf: vec![0.0; crate::consts::MAX_THETA],
        start_permuted: StartValues {
            beta: Vec::with_capacity(p),
            theta: Vec::with_capacity(crate::consts::MAX_THETA),
        },
        kind,
    }
}

// ---------------------------------------------------------------------------
// fit_on — shape pin, option check, dispatch, return FitView
// ---------------------------------------------------------------------------

/// Fit `(x, y, ids)` on a prebuilt workspace and return the lean [`FitView`].
/// The route is pinned by the build shape: `fit_on` hard-panics on any per-call
/// shape (n, p, RE level counts) or frozen-option (nagq, weights/offset
/// presence, parallel_inner) mismatch. The solver is the one `fit_warm` runs;
/// only the workspace lifecycle differs by caller.
///
/// The data, however, is the caller's responsibility — `fit_warm` validates and
/// salvages where `fit_on` trusts:
/// - **Start values.** A short `start.theta` is zipped against θ and stops at the
///   shorter, so the trailing entries keep the PREVIOUS fit's θ̂ (`ws.theta` is the
///   in/out buffer) where a fresh workspace would hold the blind `THETA0`. Same
///   for a short β start on the GLMM route. The call then answers differently
///   depending on the workspace's age, through a different BOBYQA start.
/// - **Weights and offsets.** A weights vector longer than `n` is accepted, but
///   `lmm_view_to_fit`'s `−Σ log wᵢ` correction sums the whole vector, so the LMM
///   deviance and loglik come out wrong. Non-finite weights or offsets propagate
///   as NaN; the GLMM route panics on a length mismatch instead.
/// - **Rank deficiency.** `fit_warm` drops aliased fixed-effect columns and refits
///   (lme4 behaviour); `fit_on` hands the design to the solver whole, and whether
///   it comes back NaN-filled or fitted-and-flagged is settled by the arithmetic
///   on that draw. Checking is the caller's job, per draw: read `converged` and
///   `ill_conditioned` off [`FitView::diagnostics`].
pub fn fit_on<'a>(
    ws: &'a mut FitWorkspace,
    x: &[f64],
    y: &[f64],
    ids: &GroupIds,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> FitView<'a> {
    let n = y.len();
    let p = ws.p;
    assert!(
        n <= ws.n_max,
        "fit_on: shape mismatch — n={n} exceeds build n_max={}",
        ws.n_max
    );
    assert_eq!(
        x.len(),
        n * p,
        "fit_on: shape mismatch — x.len()={} vs n*p={}",
        x.len(),
        n * p
    );
    assert_eq!(opts.nagq, ws.nagq, "fit_on: nagq is frozen at build");
    assert_eq!(
        opts.target_indices.len(),
        ws.n_targets,
        "fit_on: target count is frozen at build"
    );
    assert_eq!(
        opts.weights.is_some(),
        ws.has_weights,
        "fit_on: weights presence is frozen at build"
    );
    assert_eq!(
        opts.offset.is_some(),
        ws.has_offset,
        "fit_on: offset presence is frozen at build"
    );
    assert_eq!(
        opts.parallel_inner, ws.parallel_inner,
        "fit_on: parallel_inner is frozen at build"
    );

    // Mixed-arm shape pin: the RE level count must match the build shape, or the
    // route was pinned wrong and this draw indexes the wrong buffers (see
    // `FitWorkspace`).
    let mixed = ws.sized.re.is_some();
    if mixed {
        let re = ws.sized.re.as_ref().unwrap();
        assert_group_ids(re, ids, n);
        // Same level-count formula as `spec_sized_from_ids` (empty ids → 1), so
        // the degenerate n=0 draw matches its own build shape instead of panicking.
        let call_primary = ids
            .primary
            .iter()
            .copied()
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(1);
        assert_eq!(
            call_primary, ws.build_primary_levels,
            "fit_on: shape mismatch — primary level count {call_primary} != build {}",
            ws.build_primary_levels
        );
        // Extras are a capacity check, not an equality one: a nested grouping's
        // block is sized parents × children-per-parent, which a single draw need
        // not fill. Over capacity is the dangerous direction (see `FitWorkspace`).
        for (g, e) in ids.extra.iter().enumerate() {
            let levels = e.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
            assert!(
                levels <= ws.build_extra_capacity[g],
                "fit_on: shape mismatch — extra grouping {g} needs {levels} levels, build capacity {}",
                ws.build_extra_capacity[g]
            );
        }
    }

    // A `StartValues.theta` arrives in the caller's DECLARATION order — the
    // order `FitView::theta` reports θ̂ in, so a warm-start carry round-trips —
    // while the kernels index θ by slot. This is the input side of that pair;
    // `unpermute_fit`/`FitView::theta` are the output side.
    let perm = ws.perm;
    let start = match start {
        Some(s) if !perm.is_identity() && !s.theta.is_empty() => {
            // Refilled in place rather than cloned: on a reordering workspace
            // this runs on every draw of a warm loop.
            let dst = &mut ws.start_permuted;
            dst.beta.clear();
            dst.beta.extend_from_slice(&s.beta);
            dst.theta.clear();
            dst.theta.extend_from_slice(&s.theta);
            perm.swap_slots(&mut dst.theta);
            Some(&*dst)
        }
        unchanged => unchanged,
    };

    let theta_buf: &mut [f64] = &mut ws.theta_declared_buf;
    match &mut ws.kind {
        FitKind::Ols { ws: ols_ws, x_mat } => {
            fill_col_major(x_mat, x, n, p);
            let v =
                super::ols::fit_ols_prebuilt(ols_ws, x_mat.as_ref().subrows(0, n), y, n, p, opts);
            FitView::new(FitViewKind::Ols(v), perm, theta_buf)
        }
        FitKind::Glm { buf, x_mat } => {
            fill_col_major(x_mat, x, n, p);
            let v = super::glm::fit_glm_prebuilt(
                ws.sized.family,
                f64::NAN,
                x_mat.as_ref().subrows(0, n),
                y,
                opts,
                buf,
            );
            FitView::new(FitViewKind::Glm(v), perm, theta_buf)
        }
        FitKind::LmmDense {
            ws: lmm_ws,
            x_mat,
            y_shifted,
        } => {
            fill_col_major(x_mat, x, n, p);
            // Identity-link offset as the exact y-shift before accumulation
            // (mirrors `fit_mle`); weights fold into the Gram accumulators.
            // The identical collect lives at `fit/lmm.rs`, `loop_advanced_seam.rs`,
            // and `sparse/mod.rs`, each carrying a "change together"
            // comment — what those comments pin is the numerics (offset as an
            // exact y-shift applied before accumulation), which is unchanged
            // here; this site only stops allocating, filling the build-once
            // `y_shifted` buffer with a plain loop instead of a fresh `collect()`.
            let y_eff: &[f64] = match &opts.offset {
                Some(o) => {
                    for i in 0..n {
                        y_shifted[i] = y[i] - o[i];
                    }
                    &y_shifted[..n]
                }
                None => y,
            };
            super::lmm::accumulate_lmm_rows(
                lmm_ws,
                x_mat.as_ref().subrows(0, n),
                y_eff,
                n,
                p,
                &ids.primary,
                &ids.extra,
                opts.weights.as_deref(),
            );
            let v = super::lmm::lmm_run_on(
                lmm_ws,
                &opts.target_indices,
                warm_theta(start),
                opts.boundary_score,
            );
            FitView::new(FitViewKind::Lmm(v), perm, theta_buf)
        }
        FitKind::GlmmDense { ws: glmm_ws, x_mat } => {
            fill_col_major(x_mat, x, n, p);
            // Reset call-varying option state (mirrors `fit_glmm_build`'s one-time
            // set, made per-call so a reused ws is correct for the new draw).
            glmm_ws.parallel_inner = opts.parallel_inner;
            if let Some(w) = &opts.weights {
                glmm_ws.prior_w[..n].copy_from_slice(w);
                glmm_ws.weighted = true;
            } else {
                glmm_ws.weighted = false;
            }
            glmm_ws.offset = opts.offset.clone();
            // Refresh the RE column scales for THIS draw's design before Z is
            // rebuilt from it — mirrors `fit_glmm_build`, and mirrors what
            // `accumulate_lmm_rows` does on the LMM arm above.
            glmm_ws
                .groupings
                .set_slope_scales(x_mat.as_ref().subrows(0, n), opts.weights.as_deref());
            // Rebuild Z + the crossed-Schur symbolic factor for this (x, ids)
            // draw (both are ids-dependent; the ws buffers are reused).
            build_z(
                glmm_ws,
                x_mat.as_ref().subrows(0, n),
                &ids.primary,
                &ids.extra,
                n,
            );
            glmm_ws.structured_schur = if glmm_ws.groupings.structured_extras_eligible() {
                StructuredSchur::new(&glmm_ws.groupings, &ids.primary, &ids.extra, n)
            } else {
                None
            };
            let v = super::glmm::run_glmm_on(
                glmm_ws,
                x_mat.as_ref().subrows(0, n),
                y,
                n,
                p,
                &ws.sized,
                &ids.primary,
                &ids.extra,
                f64::NAN,
                start,
                opts,
            );
            FitView::new(FitViewKind::Glmm(v), perm, theta_buf)
        }
        FitKind::Prebuilt(route) => {
            let route = *route;
            let sized = &ws.sized;
            let fit = match route {
                PrebuiltRoute::GlmNb => super::glm::fit_glm_nb(x, y, n, p, None, opts),
                PrebuiltRoute::GlmmNbDense => super::glmm::fit_glmm_nb(
                    x,
                    y,
                    n,
                    p,
                    sized,
                    &ids.primary,
                    &ids.extra,
                    start,
                    opts,
                ),
                PrebuiltRoute::LmmSparse => crate::sparse::fit_mle_sparse(
                    x,
                    y,
                    n,
                    p,
                    sized,
                    &ids.primary,
                    &ids.extra,
                    start,
                    opts,
                ),
                PrebuiltRoute::GlmmSparse => {
                    crate::sparse::fit_glmm_sparse(
                        x,
                        y,
                        n,
                        p,
                        sized,
                        &ids.primary,
                        &ids.extra,
                        f64::NAN,
                        start,
                        opts,
                    )
                    .0
                }
                PrebuiltRoute::GlmmNbSparse => crate::sparse::fit_glmm_nb_sparse(
                    x,
                    y,
                    n,
                    p,
                    sized,
                    &ids.primary,
                    &ids.extra,
                    start,
                    opts,
                ),
            };
            let (t_sq, var_diag) = prebuilt_stats(&fit);
            FitView::new(
                FitViewKind::Prebuilt {
                    fit,
                    t_sq,
                    var_diag,
                },
                perm,
                theta_buf,
            )
        }
    }
}

#[cfg(test)]
#[path = "core_tests.rs"]
mod core_tests;
