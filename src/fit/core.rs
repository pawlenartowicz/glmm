//! Unified fit core: one dispatch body shared by `fit_cold`/`fit_warm` and the
//! `loop_advanced` hot path, so a change to glmm's model/routing lands in every
//! caller identically and no caller can re-implement the routing and drift.
//!
//! Two reusable pieces, mirroring the existing `build_glmm`/`fit_glmm_prebuilt`
//! split: [`build_workspace`] classifies the design once (family × RE × solver)
//! and allocates the per-shape [`FitWorkspace`]; [`fit_on`] accumulates + solves
//! on that workspace and returns a lean borrowed [`FitView`]. `fit_warm` is
//! re-expressed over these (build a throwaway ws, `fit_on`, `FitView::into_fit`);
//! the loop tier builds once per shape and calls `fit_on` per draw.
//!
//! Always compiled (stable `fit_warm` is re-expressed over it); only the
//! re-exports through `crate::loop_advanced` are feature-gated.

use faer::Mat;

use crate::glmm::GlmmWorkspace;
use crate::lmm::{Criterion, LmmGroupings, LmmWorkspace};
use crate::standardize::ColumnScaling;
use crate::{BinomialLink, Family, GroupIds, GroupingRelation, ModelSpec, Note, StartValues};

use super::common::{
    assert_group_ids, assert_model_shape, fill_col_major_std, resolve_criterion, start_to_design,
    theta_width, unpermute_fit, warm_theta, FitDiagnostics, Perm,
};
use super::glm::GlmScratchBuf;
use super::lmm::LmmResultView;
use super::ols::OlsWorkspace;
use super::{classify_design, Fit, FitOptions, Solver};
use crate::optim::CurvatureHint;

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
    /// θ̂ mapped back to the caller's declaration order AND out of the column
    /// standardization. Filled only when `perm` reorders something or a slope
    /// column is standardized; empty otherwise, and then [`FitView::theta`]
    /// hands back the kernel's own slice with no copy.
    ///
    /// Borrowed from [`FitWorkspace::theta_declared_buf`] rather than owned: the
    /// maps are a function of the DRAW's design (recomputed per call), so they
    /// cannot be frozen at build — but the storage can, and that is what keeps a
    /// warm random-slope loop off the heap.
    theta_declared: &'a [f64],
    /// The caller's β̂ on the OLS/GLM routes, whose kernels keep β̃ on the
    /// fitted design; borrowed from [`FitWorkspace::beta_caller`]. Empty on
    /// the other routes, whose kernels report the caller's β̂ themselves.
    betas_caller: &'a [f64],
    /// `pinned` in the caller's coordinates, when a centered slope moved the
    /// zero pattern of Λ away from the kernel's.
    pinned_override: Option<u64>,
    /// The fitted design's column map (`β = B·β̃`), for `into_fit`.
    map: &'a ColumnScaling,
}

// The `Prebuilt` arm is ~2× the next-largest variant because it owns a whole
// assembled `Fit` — which is the arm's entire reason to exist, and got wider
// again when `Fit`'s diagnostics moved behind `Diagnostics`. Clippy's fix is to
// box it, and that is the wrong trade here: the enum lives for one call frame,
// while `Box::new` would add a heap block PER DRAW on the loop tier, and that
// per-draw allocation is the one cost this crate keeps off the loop tier.
#[allow(clippy::large_enum_variant)]
enum FitViewKind<'a> {
    Ols(crate::ols::OlsFitView<'a>),
    Glm(crate::glm::GlmFitView<'a>),
    Lmm(LmmResultView<'a>),
    Glmm(super::glmm::GlmmResultView<'a>),
    /// The NB routes (GLM-NB, GLMM-NB): the kernel searches `ln θ_NB` as an
    /// outer coordinate and hands back an assembled `Fit`. `t_sq`/`var_diag`
    /// are reconstructed
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
        let mut diag = self.kernel_diagnostics();
        if let Some(mask) = self.pinned_override {
            diag.pinned_components = mask;
        }
        diag
    }

    /// [`Self::diagnostics`] as the kernel reported them, in its coordinates.
    fn kernel_diagnostics(&self) -> FitDiagnostics {
        match &self.kind {
            FitViewKind::Ols(v) => v.diagnostics(),
            FitViewKind::Glm(v) => v.diagnostics(),
            FitViewKind::Lmm(v) => v.diagnostics(),
            FitViewKind::Glmm(v) => v.diagnostics(),
            // The NB kernels already assembled a `Fit`, so the carrier is read
            // back off it. `boundary_hit` is back-derived from `singular`, which
            // is lossy in the documented way: this arm cannot report a cap-out
            // (2), and the carrier holds no `Vec`, so the per-component detail
            // stays on the assembled `Fit` (`Diagnostics::pinned`, which the
            // kernel's own view mapper filled) and does not reach here. Its
            // `singular` also carries the negligible-component check `into_fit`
            // applies to the other arms afterwards, so it is at least as
            // inclusive.
            FitViewKind::Prebuilt { fit, .. } => {
                let mut d = FitDiagnostics {
                    boundary_hit: fit.singular() as u8,
                    n_grad: fit.diagnostics.n_grad as u32,
                    outer_stop: fit.diagnostics.outer_stop,
                    outer_decrement: fit.diagnostics.outer_decrement,
                    ..FitDiagnostics::fixed_only(fit.converged())
                };
                // The inverse of `common::materialize_diagnostics`: every note it
                // raises off a carrier field sets that field back — change
                // together. Walking the notes reads the `Vec`, it allocates
                // nothing.
                for note in &fit.diagnostics.notes {
                    match note {
                        Note::IllConditioned { columns, pivot } => {
                            d.ill_conditioned = true;
                            d.pivot = *pivot;
                            d.pivot_col = columns[0];
                        }
                        Note::PirlsExhausted { evals, final_eval } => {
                            d.pirls_exhausted = *evals;
                            d.final_pirls_exhausted = *final_eval;
                        }
                        Note::HessianSeFallback => d.hessian_fallback = true,
                        Note::ExactProfileFallback => d.exact_profile_fallback = true,
                        Note::GradientSearchFallback => d.gradient_search_fallback = true,
                        Note::SeUnavailable => d.se_unavailable = true,
                        Note::PrecisionLimited => d.precision_limited = true,
                        Note::Separated => d.separated = true,
                        // No carrier field.
                        Note::NbShapeUnsettled { .. }
                        | Note::UnusedGroupingLevels { .. }
                        | Note::NonIntegerResponse { .. }
                        | Note::SingleLevelGroupingDropped { .. }
                        | Note::FewRowsForRandomEffects { .. }
                        | Note::VarianceAtLimit
                        | Note::SatterthwaiteCurvature { .. } => {}
                    }
                }
                d
            }
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
            FitViewKind::Ols(_) | FitViewKind::Glm(_) => self.betas_caller,
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

    /// Estimator dispersion: LMM σ̂², GLMM `D̂[0][0]`, `Prebuilt` `Fit.dispersion`.
    /// OLS/GLM (never read here) report NaN.
    ///
    /// The GLMM's `D̂[0][0]` is the primary grouping's random-intercept
    /// variance in the caller's units, `Λ₀₀²` of the caller's lower factor `Λ`
    /// ([`Self::theta`]'s first entry): with a centered slope the kernel's own
    /// `Λ₀₀` is the intercept at the column mean, not at `x = 0`. NaN when the
    /// search ended without an endpoint.
    pub fn dispersion(&self) -> f64 {
        match &self.kind {
            FitViewKind::Ols(_) | FitViewKind::Glm(_) => f64::NAN,
            FitViewKind::Lmm(v) => v.dispersion(),
            FitViewKind::Glmm(v) if v.has_endpoint() => self.theta()[0].powi(2),
            FitViewKind::Glmm(_) => f64::NAN,
            FitViewKind::Prebuilt { fit, .. } => fit.dispersion,
        }
    }

    /// The estimated dispersion to carry into the next draw's
    /// [`crate::StartValues::dispersion`]: θ̂_NB on the negative-binomial routes,
    /// φ̂ on a Gamma GLMM (both in [`Fit::dispersion`]'s units). `None` where
    /// no dispersion is estimated (every other family, a held Gamma φ) and on a
    /// fit with no endpoint. Not [`Self::dispersion`], which on the GLMM arm is
    /// `D̂[0][0]`.
    pub fn dispersion_start(&self) -> Option<f64> {
        let d = match &self.kind {
            FitViewKind::Glmm(v) => v.dispersion_estimate(),
            FitViewKind::Prebuilt { fit, .. } => Some(fit.dispersion),
            FitViewKind::Ols(_) | FitViewKind::Glm(_) | FitViewKind::Lmm(_) => None,
        };
        d.filter(|v| v.is_finite() && *v > 0.0)
    }

    /// Fitted θ̂ vech (primary block then extras, column-major lower-triangular) —
    /// feeds the grid-sequential warm-start carry. Empty for the `Prebuilt` arm
    /// (GLM-NB has no θ, and the GLMM-NB kernel exposes none through the
    /// assembled `Fit`) and for OLS/GLM.
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
        let map = self.map;
        // OLS / GLM: the caller's β̂ `FitView::new` already mapped.
        let beta = self.betas_caller.to_vec();
        let mut fit = match self.kind {
            FitViewKind::Ols(v) => super::ols::ols_view_to_fit(&v, beta, x, y, n, p, opts, map),
            FitViewKind::Glm(v) => {
                super::glm::glm_view_to_fit(&v, beta, y, model.family, f64::NAN, n, p, opts, map)
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
    /// Wrap a solver's borrowed result under the workspace's grouping reorder
    /// and the draw's column map, materializing the caller's β̂ and the
    /// declaration-order θ̂ once here so the hot-path accessors stay plain slice
    /// reads.
    fn new(
        kind: FitViewKind<'a>,
        perm: Perm,
        theta_buf: &'a mut [f64],
        beta_buf: &'a mut [f64],
        re_scratch: &mut [f64],
        map: &'a ColumnScaling,
    ) -> Self {
        let mut view = FitView {
            kind,
            perm,
            theta_declared: &[],
            betas_caller: &[],
            pinned_override: None,
            map,
        };
        // OLS / GLM kernels keep β̃ (the NB alternation and the GLMM seed reuse
        // it on the fitted design); the caller's β goes into a workspace buffer.
        let kernel_betas = match &view.kind {
            FitViewKind::Ols(v) => Some(v.betas),
            FitViewKind::Glm(v) => Some(v.betas),
            _ => None,
        };
        if let Some(bt) = kernel_betas {
            let p = bt.len();
            beta_buf[..p].copy_from_slice(bt);
            map.beta_to_caller(&mut beta_buf[..p]);
            view.betas_caller = &beta_buf[..p];
        }
        // θ̂ out of the column standardization, then into declaration order: a
        // caller carries θ̂ forward as a warm start, which is read in the
        // design's own units. Nothing is materialized when no slope column is
        // standardized and the reorder is the identity — the intercept-only
        // case, which is most of them — and then `theta()` hands back the
        // kernel's own slice.
        let mapped = view
            .kernel_groupings()
            .is_some_and(|g| g.any_column_std(map));
        if mapped || !perm.is_identity() {
            let n_theta = view.kernel_theta().len();
            // Sized off this workspace's own θ width, so an over-envelope vech (a
            // primary block past `MAX_PRIMARY_Q`, say) is as bounded as an
            // intercept-only one, and the map costs the warm loop no heap block.
            let theta = &mut theta_buf[..n_theta];
            view.pinned_override = match view.kernel_groupings() {
                // The mask is read in slot order, before the reorder below.
                Some(g) if mapped => {
                    let converged = view.kernel_diagnostics().converged;
                    g.theta_to_caller(view.kernel_theta(), converged, theta, re_scratch, map)
                }
                _ => {
                    theta.copy_from_slice(view.kernel_theta());
                    None
                }
            };
            // The maps are in kernel slot order, so they apply before the reorder.
            perm.swap_slots(theta);
            view.theta_declared = &theta_buf[..n_theta];
        }
        view
    }
}

/// An assembled `Fit` as a [`FitViewKind::Prebuilt`], with the predictor-indexed
/// Wald t² and Var(β̂_j) of its accessor surface reconstructed from it. Slots
/// whose `se` is not finite (non-target, non-converged) stay NaN.
fn prebuilt_kind(fit: Fit) -> FitViewKind<'static> {
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
    FitViewKind::Prebuilt {
        fit,
        t_sq,
        var_diag,
    }
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
    /// `opts.reml` resolved on `sized` ([`resolve_criterion`]): the LMM
    /// kernel's criterion, `None` off the LMM path.
    criterion: Option<Criterion>,
    /// `opts.target_indices.len()`: the OLS/GLM result slots are sized off it, so
    /// a later call asking for more targets would overrun them.
    n_targets: usize,
    /// The last `fit_on` call's `opts.target_indices`, refilled in place (the
    /// count is frozen at build): [`satterthwaite`] fills these columns.
    targets: Vec<u32>,
    has_weights: bool,
    has_offset: bool,
    parallel_inner: bool,
    /// Storage for [`FitView::theta_declared`], `n_theta` long. Sized once at
    /// build off the built kernel's own θ width, so the widest vech a design can
    /// carry is the widest this holds. Sibling of `kind` for the same borrow
    /// reason `x_mat` is (a slice of the solver workspace could not also be
    /// handed out alongside `&mut` to it).
    theta_declared_buf: Vec<f64>,
    /// This draw's column statistics (`x̃ = (x − m)/s`), recomputed by every
    /// `fit_on` call; sized `p` at build.
    scaling: ColumnScaling,
    /// Skip the per-draw standardization ([`FitWorkspace::set_prestandardized`]).
    prestandardized: bool,
    /// The caller's fixed-design promise ([`FitWorkspace::set_fixed_design`]).
    fixed_design: bool,
    /// Row count of the design the workspace holds the setup of (column
    /// statistics, `x_mat`, the LMM design statistics), set by a full setup
    /// while `fixed_design` is on; `None` when there is none to reuse.
    design_rows: Option<usize>,
    /// Debug builds: a hash of the `x`, ids and weights of the last full setup
    /// under the fixed-design promise ([`FitWorkspace::check_fixed_design`]).
    #[cfg(debug_assertions)]
    design_hash: u64,
    /// Storage for [`FitView::betas_caller`], as long as the OLS/GLM kernels'
    /// β slices (`p.max(1)`).
    beta_caller: Vec<f64>,
    /// Scratch of the random-effect maps (`LmmGroupings::re_map_scratch_len`;
    /// empty without θ): the start's forward map and `FitView::new`'s back-map.
    re_map_scratch: Vec<f64>,
    /// Column-major `x̃`, `n_max × p`, refilled by every `fit_on` call and read
    /// by every route. A sibling of `kind`, not a field of a route's workspace:
    /// `fit_ols_prebuilt`/`fit_glm_prebuilt` take `&mut Ols/GlmScratchBuf` and
    /// return a view borrowing from it, so a `MatRef` sourced from a field of
    /// that same workspace would be a second mutable borrow for the whole call.
    x_mat: Mat<f64>,
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
    Ols {
        ws: OlsWorkspace,
    },
    Glm {
        buf: GlmScratchBuf,
    },
    Lmm {
        ws: LmmWorkspace,
        // Offset-shifted y, filled only when `opts.offset` is set (per-call,
        // not frozen at build — unlike `x_mat` this buffer sits unused and
        // untouched on every offset-less call). Sibling of `ws` for the same
        // borrow reason as `FitWorkspace::x_mat`: `accumulate_lmm_rows` takes
        // `ws` mutably, so a slice of `ws` cannot also be its `y` argument.
        y_shifted: Vec<f64>,
    },
    Glmm {
        ws: GlmmWorkspace,
    },
    /// Routes whose kernel returns a fully-assembled `Fit`: the NB routes
    /// (GLM-NB, GLMM-NB), which search `ln θ_NB` as an outer coordinate.
    /// `build_workspace` pins the exact kernel here (classify once). The
    /// GLMM-NB route also keeps its `GlmmWorkspace` here, built once; the
    /// GLM-NB route still allocates its buffers per call inside `fit_on`.
    Prebuilt(PrebuiltRoute),
}

enum PrebuiltRoute {
    GlmNb,
    /// The GLMM-NB kernel's `GlmmWorkspace`, built once here and reused by
    /// every draw (`fit::glmm::fit_glmm_nb`), as the `Glmm` arm reuses its
    /// own; the fit still comes back as an assembled `Fit`.
    GlmmNb(Box<GlmmWorkspace>),
}

impl FitKind {
    /// The θ routes' grouping structure, where the random-effect maps live.
    fn groupings(&self) -> Option<&LmmGroupings> {
        match self {
            FitKind::Lmm { ws, .. } => Some(ws.kernel.groupings()),
            FitKind::Glmm { ws, .. } => Some(&ws.groupings),
            FitKind::Ols { .. } | FitKind::Glm { .. } | FitKind::Prebuilt(_) => None,
        }
    }
}

impl FitWorkspace {
    /// Skip the per-draw column standardization: the design is fitted as given,
    /// except that a random-slope column is divided by its prior-weighted root
    /// mean square (as a 0/1 slope column is when standardizing), so the
    /// random-effect search is not sensitive to its units. Same model; on
    /// data far from standardized the fit is otherwise as sensitive to units
    /// as the raw design makes it. Default `false`. Loop tier only;
    /// `fit_cold`/`fit_warm` always standardize.
    pub fn set_prestandardized(&mut self, on: bool) {
        self.prestandardized = on;
        self.design_rows = None;
    }

    /// Promise that every following [`fit_on`] call passes the same design
    /// `x`, the same group ids and the same weights as the call that ran the
    /// last full setup, with only `y` (and an offset) changing: the per-draw
    /// setup that depends on the design alone (column statistics, the
    /// standardized copy of `x`, and on the dense LMM route the fixed-effect
    /// factor and the design part of the sufficient statistics) is then
    /// computed once and reused. The first call after this one runs the full
    /// setup, and so does a call with another row count `n`, which then
    /// becomes the new reference. The answer is bit-identical to a fit
    /// without the promise. A release build does not check it: a caller that
    /// changes `x`, the ids or the weights at the same `n` while it is on gets
    /// a wrong fit (the old column statistics and design factor mixed with the
    /// new rows). A debug build hashes them on every call and panics on a
    /// change. `false` (the default) drops the cached setup. Loop tier only.
    pub fn set_fixed_design(&mut self, on: bool) {
        self.fixed_design = on;
        self.design_rows = None;
    }

    /// Whiten the fixed effects of a GLMM's gradient search by their
    /// information at the start instead of by the design, which is far off
    /// for coefficients the random effects absorb (the intercept). It costs
    /// one PQL mode solve per fit, shared with the start guard's when the
    /// guard keeps a warm start. Measured in evaluations against the default
    /// (2026-10-21): about 2% to 6% fewer on warm fits, up to a third fewer on
    /// crossed and nested shapes, but about 21% more on cold fits of data
    /// built unlike the accuracy grid. Binomial, Poisson and Gamma Laplace
    /// fits that search with BFGS and get no curvature hint; no effect on
    /// other routes. Default `false`. Loop tier only; `fit_cold`/`fit_warm`
    /// never use it.
    pub fn set_beta_info_scaling(&mut self, on: bool) {
        if let FitKind::Glmm { ws } = &mut self.kind {
            ws.beta_info_scaling = on;
        }
    }

    /// Debug builds: under the fixed-design promise, panics when this call's
    /// `x`, ids or weights differ from those of the last full setup, compared
    /// by a hash (no copy of the design is kept).
    #[cfg(debug_assertions)]
    fn check_fixed_design(&mut self, x: &[f64], ids: &GroupIds, weights: Option<&[f64]>, n: usize) {
        use std::hash::{Hash, Hasher};
        if !self.fixed_design {
            return;
        }
        let mut h = std::hash::DefaultHasher::new();
        x.iter().for_each(|v| v.to_bits().hash(&mut h));
        ids.primary.hash(&mut h);
        ids.extra.hash(&mut h);
        if let Some(w) = weights {
            w.iter().for_each(|v| v.to_bits().hash(&mut h));
        }
        let h = h.finish();
        if self.design_rows == Some(n) {
            assert_eq!(
                h, self.design_hash,
                "fit_on: set_fixed_design(true) promises the x, ids and weights of the last full setup, and this call's differ"
            );
        }
        self.design_hash = h;
    }
}

#[cfg(test)]
impl FitWorkspace {
    pub(crate) fn is_ols(&self) -> bool {
        matches!(self.kind, FitKind::Ols { .. })
    }
    pub(crate) fn is_glm(&self) -> bool {
        matches!(self.kind, FitKind::Glm { .. })
    }
    pub(crate) fn is_lmm(&self) -> bool {
        matches!(self.kind, FitKind::Lmm { .. })
    }
    /// Which LMM kernel the LMM arm built: the routing bit `classify_design`
    /// handed [`build_workspace`], readable again after the fact. Both are
    /// `false` on every non-LMM workspace.
    pub(crate) fn is_lmm_dense(&self) -> bool {
        matches!(
            self.kind,
            FitKind::Lmm {
                ws: LmmWorkspace {
                    kernel: crate::lmm::LmmKernel::Dense { .. },
                    ..
                },
                ..
            }
        )
    }
    pub(crate) fn is_lmm_sparse(&self) -> bool {
        matches!(
            self.kind,
            FitKind::Lmm {
                ws: LmmWorkspace {
                    kernel: crate::lmm::LmmKernel::Sparse { .. },
                    ..
                },
                ..
            }
        )
    }
    pub(crate) fn is_glmm(&self) -> bool {
        matches!(self.kind, FitKind::Glmm { .. })
    }
    /// The criterion the LMM kernel carries: `None` on every non-LMM
    /// workspace, and on a sparse one no `fit_on` has built the kernel of.
    pub(crate) fn kernel_criterion(&self) -> Option<Criterion> {
        let FitKind::Lmm { ws, .. } = &self.kind else {
            return None;
        };
        match &ws.kernel {
            crate::lmm::LmmKernel::Dense { fit, .. } => Some(fit.criterion),
            crate::lmm::LmmKernel::Sparse { ws, .. } => ws.as_ref().map(|s| s.criterion),
        }
    }
    /// The LMM workspace and this draw's column map, for tests that read what
    /// a fit left there. `None` on every non-LMM workspace.
    pub(crate) fn lmm_parts(&mut self) -> Option<(&mut LmmWorkspace, &ColumnScaling)> {
        match &mut self.kind {
            FitKind::Lmm { ws, .. } => Some((ws, &self.scaling)),
            _ => None,
        }
    }
    pub(crate) fn is_prebuilt(&self) -> bool {
        matches!(self.kind, FitKind::Prebuilt(_))
    }
    /// Whether a GLMM fit on this workspace had its warm θ₀ kept by the start
    /// guard: only then does `fit_glmm` build the warm stage-1 solver
    /// (`GlmmWorkspace::solver_stage1_warm`). `false` on every non-GLMM
    /// workspace.
    pub(crate) fn glmm_guard_kept_warm_start(&self) -> bool {
        match &self.kind {
            FitKind::Glmm { ws } => ws.solver_stage1_warm.is_some(),
            FitKind::Prebuilt(PrebuiltRoute::GlmmNb(ws)) => ws.solver_stage1_warm.is_some(),
            _ => false,
        }
    }
}

/// Every column a random-effect block of `model` reads as a slope (none on a
/// fixed-only spec), for [`ColumnScaling::with_re_slopes`].
fn re_slope_columns(model: &ModelSpec) -> impl Iterator<Item = usize> + '_ {
    model
        .re
        .iter()
        .flat_map(|re| {
            let extra = re.extra_groupings.iter().flat_map(|g| g.slopes.iter());
            re.slopes.iter().chain(extra)
        })
        .map(|&c| c as usize)
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
    let criterion = resolve_criterion(sized, opts);
    let t = opts.target_indices.len();
    let kind = match (&sized.family, sized.re.as_ref()) {
        (Family::Gaussian, None) => FitKind::Ols {
            ws: OlsWorkspace::new(n_max, p, t, opts.weights.is_some()),
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
        },
        (family, Some(_)) => {
            let (slope_cols, extra_slope_cols) = super::glmm::re_slope_cols(sized);
            let solver = classify_design(sized, opts.nagq);
            match family {
                // Every mixed design reaches one entry point per family and
                // picks its kernel behind it: `fit_lmm` over the dense or the
                // sparse-Z profiled objective, `fit_glmm` over its three
                // `A`-layouts.
                Family::Gaussian => {
                    let mut ws = LmmWorkspace::for_cluster_spec_ext(
                        p,
                        sized,
                        n_max,
                        &slope_cols,
                        &extra_slope_cols,
                        solver == Solver::Sparse,
                    );
                    // The sparse kernel takes it when `accumulate_lmm_rows`
                    // builds it.
                    if let crate::lmm::LmmKernel::Dense { fit, .. } = &mut ws.kernel {
                        fit.criterion = criterion.expect("a Gaussian mixed model has a criterion");
                    }
                    FitKind::Lmm {
                        ws,
                        y_shifted: vec![0.0f64; n_max.max(1)],
                    }
                }
                Family::NegativeBinomial { .. } => FitKind::Prebuilt(PrebuiltRoute::GlmmNb(
                    Box::new(GlmmWorkspace::for_cluster_spec_ext(
                        p,
                        &super::glmm::nb_spec(sized),
                        n_max,
                        &slope_cols,
                        &extra_slope_cols,
                        opts.nagq,
                    )),
                )),
                _ => {
                    let ws = GlmmWorkspace::for_cluster_spec_ext(
                        p,
                        sized,
                        n_max,
                        &slope_cols,
                        &extra_slope_cols,
                        opts.nagq,
                    );
                    FitKind::Glmm { ws }
                }
            }
        }
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
    // The pin is route-agnostic even though only the dense routes and the GLMM-NB
    // route preallocate: the sparse and GLM-NB routes build per call, but from
    // `ws.sized` (the BUILD shape), so an over-capacity draw is undersized there
    // too.
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
    // θ width of the kernel this build picked — read off the built groupings, not
    // off the spec, so it is the length the solver will actually hand back. The
    // fixed-only and `Prebuilt` routes carry no θ and get an empty buffer.
    let n_theta = match &kind {
        FitKind::Lmm { ws, .. } => ws.kernel.groupings().n_theta(),
        FitKind::Glmm { ws, .. } => ws.n_theta,
        FitKind::Ols { .. } | FitKind::Glm { .. } | FitKind::Prebuilt(_) => 0,
    };
    let re_len = kind.groupings().map_or(0, |g| g.re_map_scratch_len());
    FitWorkspace {
        n_max,
        p,
        sized: sized.clone(),
        perm,
        build_primary_levels,
        build_extra_capacity,
        nagq: opts.nagq,
        criterion,
        n_targets: t,
        targets: opts.target_indices.clone(),
        has_weights: opts.weights.is_some(),
        has_offset: opts.offset.is_some(),
        parallel_inner: opts.parallel_inner,
        theta_declared_buf: vec![0.0; n_theta],
        scaling: ColumnScaling::with_re_slopes(p, re_slope_columns(sized)),
        prestandardized: false,
        fixed_design: false,
        design_rows: None,
        #[cfg(debug_assertions)]
        design_hash: 0,
        beta_caller: vec![0.0; p.max(1)],
        re_map_scratch: vec![0.0; re_len],
        x_mat: Mat::<f64>::zeros(n_max.max(1), p.max(1)),
        start_permuted: StartValues {
            beta: Vec::with_capacity(p),
            theta: Vec::with_capacity(crate::consts::MAX_THETA),
            dispersion: None,
        },
        kind,
    }
}

// ---------------------------------------------------------------------------
// fit_on — shape pin, option check, dispatch, return FitView
// ---------------------------------------------------------------------------

/// Fit `(x, y, ids)` on a prebuilt workspace and return the lean [`FitView`].
/// The route is pinned by the build shape: `fit_on` hard-panics on any per-call
/// shape (n, p, RE level counts) or frozen-option (nagq, the criterion `reml`
/// resolves to, weights/offset presence, parallel_inner) mismatch. The solver
/// is the one `fit_warm` runs; only the workspace lifecycle differs by caller.
///
/// The data, however, is the caller's responsibility — `fit_warm` validates and
/// salvages where `fit_on` trusts:
/// - **Start values.** A `start.beta` or `start.theta` of the wrong width is
///   dropped: that component cold-starts.
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
    fit_on_hinted(ws, x, y, ids, start, None, opts)
}

/// Test-only: route a GLMM workspace's outer search to BOBYQA on both start
/// kinds (the gradient route off), so a test of a BOBYQA-route property runs
/// on that route whatever the shape. A no-op on every other workspace kind.
#[cfg(test)]
pub(crate) fn force_bobyqa_route(ws: &mut FitWorkspace) {
    if let FitKind::Glmm { ws, .. } = &mut ws.kind {
        ws.route_plan = crate::optim::RoutePlan::BOBYQA;
    }
}

/// The [`CurvatureHint`] at the optimum of the last fit on `ws`, for passing
/// to [`fit_on_hinted`] on later draws. `None` when that fit did not search
/// with BFGS, or when the curvature there is not positive definite. BFGS
/// searches LMMs whose random slopes are all on one grouping (the hint then
/// costs `n_θ + 1` deviance and gradient evaluations), and binomial,
/// Poisson and Gamma Laplace GLMMs with at least two θ on the blocked or
/// structured layout (the hint is the pilot search's own final estimate and
/// costs nothing). Call it once, on a pilot fit, right after it.
pub fn curvature_hint(ws: &mut FitWorkspace) -> Option<CurvatureHint> {
    match &mut ws.kind {
        FitKind::Lmm { ws, .. } => crate::lmm::lmm_curvature_hint(ws),
        FitKind::Glmm { ws } => crate::glmm::glmm_curvature_hint(ws),
        _ => None,
    }
}

/// The Satterthwaite df of each target coefficient of the last fit on `ws`,
/// into `df_out` (length `p`, NaN off the last `fit_on` call's
/// `target_indices`): the stable path's [`Fit::coef_df`], run on demand. Call
/// it after [`fit_on`] or [`fit_on_hinted`], with the view dropped; nothing
/// runs it otherwise. It reads θ̂ and never writes it, and it leaves the
/// kernel's factor and `σ̂²` off θ̂, at its last stencil point, as
/// [`curvature_hint`] does, so the two give the same answers in either order
/// and the next `fit_on` refits from scratch. The pass's buffers live on the
/// workspace, built on the first call, so each later call allocates only the
/// pseudo-inverse's blocks: the same count whichever variance parameters the
/// draw pins at zero while at least one stays free, and none when the draw
/// pins them all (its df is then ν). On a rank-deficient draw the df are as
/// untrustworthy as β̂ (they can be finite on aliased columns, where the
/// stable path refits the reduced design and gives NaN): read
/// `ill_conditioned` off [`FitView::diagnostics`] first, as for [`fit_on`].
///
/// # Errors
///
/// `df_out` is then all NaN: [`NoDfReason::NotGaussianMixed`] on any
/// workspace but a Gaussian LMM's (OLS included); [`NoDfReason::Failed`]
/// when the pass failed (a stencil deviance, `C` or `σ̂²` not finite) or the
/// fit had no finite endpoint to run it at.
///
/// # Panics
///
/// When `df_out.len()` is not the workspace's `p`, or when `ws` is a sparse
/// Gaussian LMM's and no `fit_on` has run on it. A dense one no `fit_on` has
/// filled gives [`NoDfReason::Failed`], as a fit with no endpoint does.
///
/// [`NoDfReason::NotGaussianMixed`]: crate::inference::contrast::NoDfReason::NotGaussianMixed
/// [`NoDfReason::Failed`]: crate::inference::contrast::NoDfReason::Failed
pub fn satterthwaite(
    ws: &mut FitWorkspace,
    df_out: &mut [f64],
) -> Result<(), crate::inference::contrast::ContrastError> {
    use crate::inference::contrast::{ContrastError, NoDfReason};
    assert_eq!(
        df_out.len(),
        ws.p,
        "satterthwaite: df_out.len() must be the workspace's p"
    );
    df_out.fill(f64::NAN);
    let FitKind::Lmm { ws: lmm, .. } = &mut ws.kind else {
        return Err(ContrastError::NoDf(NoDfReason::NotGaussianMixed));
    };
    // β̂ holds a NaN whenever `fit_lmm` found no finite endpoint (it NaN-fills
    // β̂ there), the case the stable path records as a failed pass without
    // running one. `n_rows ≤ p` (a dense `p = 0` design accumulates no rows,
    // and neither does a workspace no `fit_on` has filled) has no endpoint
    // either (no pass can run there).
    if lmm.recovery.betas.iter().any(|b| b.is_nan())
        || lmm.kernel.n_rows() <= lmm.recovery.betas.len()
    {
        return Err(ContrastError::NoDf(NoDfReason::Failed));
    }
    if !crate::lmm::satterthwaite::pass_targets(lmm, &ws.scaling, &ws.targets, df_out) {
        return Err(ContrastError::NoDf(NoDfReason::Failed));
    }
    Ok(())
}

/// The stable path's Satterthwaite step, run by `fit_warm` right after
/// [`FitView::into_fit`] while `ws` still holds the fit's kernel: on a
/// Gaussian LMM workspace it stores the switched-off record when
/// `opts.satterthwaite` is `false`; otherwise, when the fit has a finite
/// deviance, it runs the pass at θ̂, fills [`Fit::coef_df`] on the target
/// columns, stores the state and pushes [`Note::SatterthwaiteCurvature`] when
/// [`curvature_note`] says so; without one, it records a failed pass
/// ([`record_failed_pass`]). A no-op on every other workspace kind. It
/// writes no other field of `fit`.
pub(crate) fn attach_satterthwaite(ws: &mut FitWorkspace, fit: &mut Fit, opts: &FitOptions) {
    use crate::lmm::satterthwaite::{pass, SatterthwaiteState};
    let FitKind::Lmm { ws: lmm, .. } = &mut ws.kind else {
        return;
    };
    if !opts.satterthwaite {
        fit.satterthwaite = Some(SatterthwaiteState::switched_off());
        return;
    }
    // A finite deviance is the fit's re-evaluation at `ws.theta`, the θ̂ the
    // pass runs at (a `MaxFunReached` endpoint included), and the one place
    // `beta_tilde` is written, so the pass's response shift uses this fit's
    // β̂. A NaN-filled fit (no endpoint, or a non-finite re-evaluation there)
    // has neither: no pass runs, and the fit records a failed one.
    if !fit.deviance.is_finite() {
        record_failed_pass(fit);
        return;
    }
    let state = pass(lmm, &ws.scaling);
    let df = state.coef_df();
    for &t in &opts.target_indices {
        fit.coef_df[t as usize] = df[t as usize];
    }
    if let Some(note) = curvature_note(&state, fit.converged()) {
        fit.diagnostics.notes.push(note);
    }
    fit.satterthwaite = Some(state);
}

/// Record on a Gaussian LMM fit with no finite endpoint (a non-finite
/// endpoint, in the note's terms) a failed pass without running one: the
/// failed record and its note, `nonfinite` set.
pub(crate) fn record_failed_pass(fit: &mut Fit) {
    let record = crate::lmm::satterthwaite::SatterthwaiteState::failed_record();
    if let Some(note) = curvature_note(&record, fit.converged()) {
        fit.diagnostics.notes.push(note);
    }
    fit.satterthwaite = Some(record);
}

/// The note a pass's state raises at an endpoint that did or did not
/// converge: when a direction of `H_p` was dropped or negative, when the
/// endpoint did not converge, or when the pass failed; `None` otherwise.
pub(crate) fn curvature_note(
    state: &crate::lmm::satterthwaite::SatterthwaiteState,
    converged: bool,
) -> Option<Note> {
    let raise = state.dropped > 0 || state.negative > 0 || !converged || state.is_failed();
    raise.then_some(Note::SatterthwaiteCurvature {
        dropped: state.dropped as u32,
        negative: state.negative as u32,
        converged,
        nonfinite: state.is_failed(),
    })
}

/// [`fit_on`] with a [`CurvatureHint`] for the gradient search, taken from a
/// pilot fit by [`curvature_hint`] and passed with every draw. The fit only
/// reads it, so results depend on (data, start, hint) and not on the order of
/// the calls. It is used where the fit searches with BFGS (LMMs whose random
/// slopes are all on one grouping, when the per-group route is armed for the
/// draw; binomial, Poisson and Gamma Laplace GLMMs with at least two θ on the
/// blocked or structured layout) and ignored where it searches with BOBYQA.
///
/// # Panics
///
/// As [`fit_on`], and when `hint` was taken on a workspace of another kind or
/// search dimension (a hint applies to LMM workspaces of its own `n_θ`, and
/// to GLMM workspaces of its own `n_θ + p`, + 1 on a Gamma fit whose
/// dispersion is not held).
pub fn fit_on_hinted<'a>(
    ws: &'a mut FitWorkspace,
    x: &[f64],
    y: &[f64],
    ids: &GroupIds,
    start: Option<&StartValues>,
    hint: Option<&CurvatureHint>,
    opts: &FitOptions,
) -> FitView<'a> {
    if let Some(h) = hint {
        let dim = match &ws.kind {
            FitKind::Lmm { ws, .. } => ws.kernel.groupings().n_theta(),
            // `opts.dispersion`: the workspace's held φ is set later, by the fit.
            FitKind::Glmm { ws } => {
                crate::glmm::grad_search_dim(ws.family, ws.n_theta, ws.p, opts.dispersion.is_some())
            }
            _ => panic!("fit_on_hinted: a curvature hint applies to LMM and GLMM workspaces only"),
        };
        assert_eq!(
            h.dim(),
            dim,
            "fit_on_hinted: curvature hint dimension {} != the workspace's search dimension {dim}",
            h.dim()
        );
    }
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
        resolve_criterion(&ws.sized, opts),
        ws.criterion,
        "fit_on: reml is frozen at build"
    );
    assert_eq!(
        opts.target_indices.len(),
        ws.n_targets,
        "fit_on: target count is frozen at build"
    );
    ws.targets.clear();
    ws.targets.extend_from_slice(&opts.target_indices);
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

    #[cfg(debug_assertions)]
    ws.check_fixed_design(x, ids, opts.weights.as_deref(), n);
    let FitWorkspace {
        kind,
        scaling,
        prestandardized,
        fixed_design,
        design_rows,
        theta_declared_buf,
        beta_caller,
        re_map_scratch,
        x_mat,
        start_permuted,
        perm,
        sized,
        criterion,
        ..
    } = ws;
    let perm = *perm;
    let sized: &ModelSpec = sized;

    // Column statistics for this draw, once, before any route reads X: the fit
    // runs on x̃ = (x − m)/s and reports through B and N (`standardize`).
    // Under the fixed-design promise both, and the LMM design statistics
    // below, are the previous call's.
    let same_design = *fixed_design && *design_rows == Some(n);
    if !same_design {
        scaling.compute(x, n, p, opts.weights.as_deref(), !*prestandardized);
        fill_col_major_std(x_mat, x, n, p, scaling);
        *design_rows = fixed_design.then_some(n);
    }
    let scaling: &'a ColumnScaling = scaling;
    let x_mat = x_mat.as_ref().subrows(0, n);

    // Starts arrive in the caller's units and DECLARATION order — the order
    // `FitView::theta` reports θ̂ in, so a warm-start carry round-trips — while
    // the kernels index θ by slot and fit the standardized design. A component
    // of the wrong width is dropped (a cold start for it), θ is moved to slot
    // order, and on the θ routes both are mapped onto the standardized design
    // (`start_to_design`; the GLMM-NB route maps inside `fit_glmm_nb`). This is
    // the input side of the pair; `unpermute_fit`/`FitView::theta` are the
    // output side.
    let n_theta = theta_width(sized.re.as_ref());
    let maps_start = kind.groupings().filter(|_| !scaling.is_identity());
    let as_given = |s: &StartValues| {
        (s.beta.is_empty() || s.beta.len() == p)
            && (s.theta.is_empty() || (s.theta.len() == n_theta && perm.is_identity()))
    };
    let start = match start {
        Some(s) if maps_start.is_some() || !as_given(s) => {
            // Refilled in place rather than cloned: on a reordering or
            // standardizing workspace this runs on every draw of a warm loop.
            let dst = start_permuted;
            dst.beta.clear();
            if s.beta.len() == p {
                dst.beta.extend_from_slice(&s.beta);
            }
            dst.theta.clear();
            if s.theta.len() == n_theta {
                dst.theta.extend_from_slice(&s.theta);
                perm.swap_slots(&mut dst.theta);
            }
            dst.dispersion = s.dispersion;
            if let Some(g) = maps_start {
                start_to_design(dst, scaling, g, re_map_scratch);
            }
            Some(&*dst)
        }
        unchanged => unchanged,
    };

    let kind = match kind {
        FitKind::Ols { ws: ols_ws } => FitViewKind::Ols(super::ols::fit_ols_prebuilt(
            ols_ws, x_mat, y, n, p, opts, scaling,
        )),
        FitKind::Glm { buf } => FitViewKind::Glm(super::glm::fit_glm_prebuilt(
            sized.family,
            f64::NAN,
            x_mat,
            y,
            opts,
            buf,
            scaling,
            None,
        )),
        FitKind::Lmm {
            ws: lmm_ws,
            y_shifted,
        } => {
            // Identity-link offset as the exact y-shift before accumulation
            // (mirrors `fit_mle`); weights fold into the Gram accumulators.
            // The identical collect lives at `fit/lmm.rs`'s `fit_mle`, carrying a
            // "change together" comment — what that comment pins is the numerics
            // (offset as an exact y-shift applied before accumulation), which is
            // unchanged here; this site only stops allocating, filling the
            // build-once `y_shifted` buffer with a plain loop instead of a fresh
            // `collect()`.
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
                x_mat,
                y_eff,
                n,
                p,
                &ids.primary,
                &ids.extra,
                opts.weights.as_deref(),
                same_design,
                criterion.expect("an LMM workspace has a criterion"),
            );
            FitViewKind::Lmm(super::lmm::lmm_run_on(
                lmm_ws,
                &opts.target_indices,
                warm_theta(start),
                hint,
                scaling,
            ))
        }
        // Degenerate guard, the twin of `build_on_workspace`'s: the PIRLS
        // kernels take no width-0 fixed block, and at `n <= p` the fixed
        // effects are not estimable.
        FitKind::Glmm { ws: glmm_ws } if n <= p || p == 0 => prebuilt_kind(
            super::glmm::degenerate_glmm_fit(n, p, glmm_ws.n_theta, glmm_ws.nagq),
        ),
        FitKind::Glmm { ws: glmm_ws } => {
            // Per-draw option state, packed M columns and crossed-Schur
            // symbolic factors — what `accumulate_lmm_rows` does
            // for the LMM arm above.
            super::glmm::prep_glmm_design(glmm_ws, &ids.primary, &ids.extra, n, opts);
            glmm_ws.grad_hint_in = hint.cloned();
            FitViewKind::Glmm(super::glmm::run_glmm_on(
                glmm_ws,
                x_mat,
                y,
                n,
                p,
                sized,
                &ids.primary,
                &ids.extra,
                f64::NAN,
                start,
                opts,
                scaling,
            ))
        }
        // `start` is in the caller's units here: this route has no groupings
        // on the `FitWorkspace`, and `fit_glmm_nb` maps it.
        FitKind::Prebuilt(route) => prebuilt_kind(match route {
            PrebuiltRoute::GlmNb => {
                let seed = start.and_then(|s| s.dispersion);
                super::glm::fit_glm_nb(x_mat, y, n, p, seed, opts, scaling).0
            }
            PrebuiltRoute::GlmmNb(nb_ws) => super::glmm::fit_glmm_nb(
                nb_ws,
                x_mat,
                y,
                n,
                p,
                sized,
                &ids.primary,
                &ids.extra,
                start,
                opts,
                scaling,
            ),
        }),
    };
    FitView::new(
        kind,
        perm,
        theta_declared_buf,
        beta_caller,
        re_map_scratch,
        scaling,
    )
}

#[cfg(test)]
#[path = "core_tests.rs"]
pub(super) mod core_tests;
