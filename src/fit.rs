//! Friendly stable `fit` entry point for the `glmm` crate.
//!
//! Owns all scratch; dispatches on `ModelSpec::estimator`; returns `Fit`.
//! This is the additive stable public surface —
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

use crate::consts::{MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q, MAX_PRIMARY_Q};
use crate::glm::{glm_irls_fit, GlmScratch};
use crate::glmm::{build_z, GlmmWorkspace, StructuredSchur};
use crate::lmm::{fit_lmm, LmmWorkspace};
use crate::ols::{OlsScratch, OlsSuffStats, PANEL_ROWS};
use crate::{
    BinomialLink, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, NegBinomialLink,
    ReStructure, Sizing, StartValues, WaldSe,
};

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
    /// Estimated dispersion: `φ` for Gamma (Pearson moment estimator), the
    /// estimated shape `θ` for negative-binomial, and `1.0` for
    /// Gaussian/binomial/Poisson (where dispersion is fixed, not estimated).
    pub dispersion: f64,
    pub converged: bool,
    /// RE (co)variance per grouping: one **vech-packed
    /// lower-triangular** covariance block `D̂` per grouping, in declaration
    /// order (primary, then each extra). `D̂ = σ̂²·Λ̂Λ̂'` (LMM) / `Λ̂Λ̂'` (GLMM,
    /// dispersion ≡ 1 on the link scale). Vech order is column-major
    /// lower-triangular (matching the θ vech convention): for a `q×q` block,
    /// `(0,0),(1,0),…,(q-1,0),(1,1),…,(q-1,q-1)`. Validated against lme4
    /// `VarCorr` (`parity/goldens/sleepstudy_lmm.json`). Empty for OLS/GLM
    /// (no random effects). This is the q≥2-valid replacement for `tau2`'s
    /// per-component variances; `tau2` is retained for back-compat.
    pub varcorr: Vec<Vec<f64>>,
    /// SE of each RE standard deviation, laid out like `tau2` (per θ coordinate,
    /// length `n_theta`; primary block then each extra, in declaration order).
    /// Populated ONLY on a converged GLMM `WaldSe::Hessian` fit, from the θ block
    /// of the joint (θ,β) FD-Hessian covariance that `fd_hessian_cov` inverts and
    /// otherwise discards. NaN under `WaldSe::Rx`, on the Hessian RX fallback, and
    /// for OLS/LMM (no Hessian machinery). Correct for SCALAR groupings only (the
    /// reachable GLMM case), where the RE stddev equals its θ so the θ-scale SE is
    /// the stddev SE directly; a q≥2 block would need a delta-method Jacobian.
    pub stddev_se: Vec<f64>,
    /// Rank-deficiency mask, length `p`: `true` for a fixed-effect
    /// column dropped because it is aliased (linearly dependent) on an earlier
    /// column, mirroring lme4's `NA`-coefficient behavior. The corresponding
    /// `beta`/`se` slots are `NaN` and `converged` stays `true` (the reduced
    /// model fits). All-`false` when the design is full-rank.
    pub aliased: Vec<bool>,
}

impl Fit {
    /// Reduce `varcorr[group_idx]` — a vech-packed (column-major lower-triangular,
    /// see [`Fit::varcorr`]) `q×q` covariance block — into per-dimension standard
    /// deviations and a full symmetric `q×q` correlation matrix, mirroring lme4's
    /// `VarCorr` stddev/corr split. `q` is recovered from `vech.len() = q(q+1)/2`
    /// via the quadratic formula. Column `c`'s entries start at vech offset
    /// `c*q - c*(c-1)/2` (the same running cursor `varcorr_block` builds — for
    /// `r >= c`, `idx(r,c) = c*q - c*(c-1)/2 + (r-c)`); change together.
    pub fn stddev_corr(&self, group_idx: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let vech = &self.varcorr[group_idx];
        let len = vech.len();
        let q = (((1 + 8 * len) as f64).sqrt() as usize - 1) / 2;
        debug_assert_eq!(
            q * (q + 1) / 2,
            len,
            "varcorr[{group_idx}] is not a valid vech"
        );

        // `c*(c-1)/2` written as `(c*c - c)/2` to avoid a `c - 1` underflow at
        // c=0 in usize arithmetic (c*c >= c holds for all c >= 0).
        let idx = |r: usize, c: usize| -> usize { c * q - (c * c - c) / 2 + (r - c) };
        let stddev: Vec<f64> = (0..q).map(|i| vech[idx(i, i)].sqrt()).collect();

        let mut corr = vec![vec![0.0; q]; q];
        for i in 0..q {
            corr[i][i] = 1.0;
        }
        for c in 0..q {
            for r in (c + 1)..q {
                let rho = vech[idx(r, c)] / (stddev[r] * stddev[c]);
                corr[r][c] = rho;
                corr[c][r] = rho;
            }
        }
        (stddev, corr)
    }
}

/// Options for `fit`. Carries the safe, defaulted method knobs: unlike
/// a warm start these do not silently move the estimate based on a caller's guess.
pub struct FitOptions {
    /// Predictor column indices for which SE is computed.
    pub target_indices: Vec<u32>,
    /// Wald-SE denominator (relocated from `ModelSpec`). Default `Hessian`.
    pub wald_se: WaldSe,
    /// Adaptive Gauss–Hermite node count (relocated from `ModelSpec`). Default 1
    /// (= Laplace). Must be odd and in `1..=MAX_NAGQ`; `>1` is honored only on a
    /// single scalar-intercept binomial/Poisson GLMM (M3), other shapes ignore it.
    pub nagq: u8,
    /// Gamma dispersion directive (relocated from `Family::Gamma`). `None` =
    /// estimate φ post-fit (Pearson); `Some(v)` = hold φ fixed at `v`. Ignored by
    /// non-Gamma families.
    pub dispersion: Option<f64>,
    /// Per-row prior (case) weights `wᵢ` — lme4's `weights=`. `None` = unit
    /// weights. For an aggregated binomial, `y` is the success PROPORTION and
    /// `wᵢ` the trial count: this is exactly lme4's `cbind(s, m−s)` objective,
    /// whose deviance differs from the expanded-Bernoulli one only by a
    /// data-only saturated constant (same argmin — same β/SE/varcomp).
    /// Currently honored ONLY on the sparse binomial GLMM path
    /// (`Solver::Sparse`, `Family::Binomial`); every other dispatch faults at
    /// the boundary rather than silently fit unweighted. Validated against the
    /// `sim_sparse_binomial` parity rung (lme4/MixedModels weighted fits) and
    /// the in-crate `sparse_weighted_binomial_*` expanded-vs-aggregated tests.
    pub weights: Option<Vec<f64>>,
    /// Opt into the restructured inner-fit kernels (cluster-outer AGQ, per-`(i,j)`
    /// FD-Hessian) tracked in `docs/GLMM/ideas/inner-fit-parallelism.md`, instead of
    /// today's sequential ones. Two effects, one flag, resolved differently: (1)
    /// switches to the restructured algorithm — live regardless of the `parallel`
    /// Cargo feature, since e.g. AGQ's cluster-outer loop's row-locality benefit
    /// doesn't need threads; (2) ADDITIONALLY dispatches via `rayon` — only when
    /// built with the `parallel` feature on a non-`wasm32` target (compile-time
    /// exclusion, not a runtime check, so `wasm32` builds never pull `rayon` in
    /// regardless of this flag).
    ///
    /// Default `true`: a standalone fit through this stable surface should use
    /// whatever the build offers. Batch callers driving many fits at once (e.g.
    /// MCPower's `loop_advanced` hot loop) should set this `false` — that caller
    /// already owns the outer parallelism, and stacking a second parallel axis
    /// inside every fit adds per-split overhead (and, for AGQ, a real per-fit CSR
    /// preprocessing cost) for no benefit once the outer loop saturates the cores.
    ///
    /// **Not yet wired to any code path** — this field is scaffolding; setting it
    /// currently has no effect. See the design doc above for the full rationale,
    /// including why nesting this under an outer `rayon` batch loop is safe (one
    /// shared work-stealing pool) in a way naive OS-thread or BLAS-thread nesting
    /// is not.
    pub parallel_inner: bool,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            target_indices: vec![],
            wald_se: WaldSe::Hessian,
            nagq: 1,
            dispersion: None,
            weights: None,
            parallel_inner: true,
        }
    }
}

/// Cold single fit — the default real-data entry. Random-effect
/// level ids are supplied per-row via `ids` ([`GroupIds`]); every level count is
/// derived from them, so `ModelSpec` stays structure-only. Exactly
/// `fit_warm(.., None, ..)`.
///
/// Dispatches on `(family, re.is_some())` (`Gaussian`→OLS/LMM,
/// `Binomial`→GLM/GLMM, Poisson/Gamma/NB likewise; `re: None`→fixed-only,
/// `Some`→mixed). Binomial counts are fit as expanded 0/1 Bernoulli rows (the
/// kernel is Bernoulli) — except on the sparse binomial GLMM path, where
/// aggregated proportions with [`FitOptions::weights`] trial counts fit
/// directly without the expansion.
///
/// # Panics
///
/// Panics only on engine invariant violations (`x.len() != n*p`, malformed
/// [`GroupIds`], over-envelope model). Numerical failures signal via
/// `Fit { converged: false, .. }` with NaN-filled estimates.
///
/// # Examples
///
/// Fixed-only OLS (`re: None`): `y = 2x` exactly, so the fitted slope is 2.
///
/// ```rust
/// use glmm::{fit_cold, Family, FitOptions, GroupIds, ModelSpec};
///
/// let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]; // n=3, p=2, row-major [intercept, x]
/// let y = vec![0.0, 2.0, 4.0];
/// let model = ModelSpec { family: Family::Gaussian, re: None };
/// let opts = FitOptions { target_indices: vec![1], ..FitOptions::default() };
/// let fit = fit_cold(&x, &y, 3, 2, &model, &GroupIds::default(), &opts);
/// assert!(fit.converged);
/// assert!((fit.beta[1] - 2.0).abs() < 1e-9);
/// ```
pub fn fit_cold(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    opts: &FitOptions,
) -> Fit {
    fit_warm(x, y, n, p, model, ids, None, opts)
}

/// Warm single fit — identical to [`fit_cold`] but accepts an optional
/// [`StartValues`] warm start. `Some` warm-starts the optimizer;
/// `None` cold-starts (byte-identical to `fit_cold`). One fit, one answer — the
/// start only shortens the path. The start threads `beta`+`theta` into the
/// LMM/GLMM kernels; fixed-only (OLS/GLM) and the NB global-θ search ignore it
/// (see [`crate::StartValues`]).
///
/// # Panics
///
/// As [`fit_cold`], plus: with `start = Some`, `start.beta.len() != p` or
/// `start.theta.len() != n_theta` (the model's RE θ width) — a malformed stable
/// input faults at the entry, not deep in a kernel.
///
/// # Examples
///
/// An intercept-only Gaussian LMM (3 clusters, `n_theta = 1`) warm-started off a
/// perturbed θ:
///
/// ```rust
/// use glmm::{fit_warm, Family, FitOptions, GroupIds, ModelSpec, ReStructure, Sizing, StartValues};
///
/// let x = vec![1.0; 6]; // n=6, p=1 (intercept-only design)
/// let y = vec![1.0, 2.0, 1.5, 1.2, 2.1, 1.4];
/// let model = ModelSpec {
///     family: Family::Gaussian,
///     re: Some(ReStructure {
///         sizing: Sizing::FixedClusters { n_clusters: 3 },
///         slopes: vec![],
///         extra_groupings: vec![],
///     }),
/// };
/// let ids = GroupIds { primary: vec![0, 1, 2, 0, 1, 2], extra: vec![] };
/// let start = StartValues { beta: vec![0.0], theta: vec![1.0] };
/// let fit = fit_warm(&x, &y, 6, 1, &model, &ids, Some(&start), &FitOptions::default());
/// assert_eq!(fit.beta.len(), 1);
/// ```
#[allow(clippy::too_many_arguments)] // marshals (x, y, n, p, spec, ids, start, opts)
pub fn fit_warm(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Fit {
    assert_eq!(
        x.len(),
        n * p,
        "x must have n*p elements in row-major layout"
    );
    assert_eq!(y.len(), n, "y must have n elements");
    assert_model_shape(model, p, opts.nagq);
    if let Some(s) = start {
        assert_eq!(s.beta.len(), p, "StartValues.beta must have p elements");
        assert_eq!(
            s.theta.len(),
            theta_width(model.re.as_ref()),
            "StartValues.theta must have n_theta elements for this RE structure"
        );
    }
    // Prior weights: shape-check at the boundary and reject unsupported paths
    // here (not deep in a kernel) — only the sparse binomial GLMM honors them.
    if let Some(w) = &opts.weights {
        assert_eq!(w.len(), n, "FitOptions.weights must have n elements");
        assert!(
            w.iter().all(|&v| v.is_finite() && v > 0.0),
            "FitOptions.weights must be finite and > 0"
        );
        let sparse_binomial = matches!(model.family, Family::Binomial { .. })
            && model.re.is_some()
            && matches!(classify_design(model, opts.nagq), Solver::Sparse);
        assert!(
            sparse_binomial,
            "FitOptions.weights is currently supported only on the sparse binomial GLMM path"
        );
    }
    // Rank-deficiency salvage: drop fixed-effect columns aliased on
    // earlier columns and fit the reduced model (lme4 behavior — NA coefficient,
    // still converges). Path-agnostic: preprocesses the fixed design X before the
    // solver dispatch, so it serves OLS/GLM/LMM/GLMM alike. Runs after the shape
    // asserts (full p still valid) and short-circuits into a reduced re-entry.
    if n > 0 && p > 0 {
        let aliased = detect_aliased(x, n, p);
        if aliased.iter().any(|&a| a) {
            return fit_rank_deficient(x, y, n, p, model, ids, start, opts, &aliased);
        }
    }
    match (&model.family, model.re.as_ref()) {
        // Fixed-only (OLS/GLM/NB-GLM): `ids` and `start` are unused — a warm start
        // is a no-op for fixed-only fits. No GroupIds shape check.
        (Family::Gaussian, None) => fit_ols(x, y, n, p, opts),
        (
            Family::Poisson { .. }
            | Family::Gamma { .. }
            | Family::Binomial {
                link: BinomialLink::Probit | BinomialLink::Logit,
            },
            None,
        ) => fit_glm(model.family, f64::NAN, x, y, n, p, opts),
        (Family::NegativeBinomial { .. }, None) => fit_glm_nb(x, y, n, p, None, opts),
        // Mixed (LMM/GLMM): validate the ids, size the spec from them, dispatch.
        (family, Some(re)) => {
            assert_group_ids(re, ids, n);
            let sized = spec_sized_from_ids(model, ids);
            match classify_design(model, opts.nagq) {
                Solver::NoZ => match family {
                    Family::Gaussian => {
                        fit_mle(x, y, n, p, &sized, &ids.primary, &ids.extra, start, opts)
                    }
                    Family::NegativeBinomial { .. } => {
                        fit_glmm_nb(x, y, n, p, &sized, &ids.primary, &ids.extra, start, opts)
                    }
                    Family::Binomial { .. } | Family::Poisson { .. } | Family::Gamma { .. } => {
                        fit_glmm(
                            x,
                            y,
                            n,
                            p,
                            &sized,
                            &ids.primary,
                            &ids.extra,
                            f64::NAN,
                            start,
                            opts,
                        )
                        .0
                    }
                },
                // Over-envelope designs: the Gaussian sparse-Z REML
                // path, the sparse NB marginal-θ wrapper, and the sparse non-Gaussian
                // PIRLS driver — every wired family fits, no reachable panic.
                Solver::Sparse => match family {
                    Family::Gaussian => crate::sparse::fit_mle_sparse(
                        x,
                        y,
                        n,
                        p,
                        &sized,
                        &ids.primary,
                        &ids.extra,
                        start,
                        opts,
                    ),
                    Family::NegativeBinomial { .. } => crate::sparse::fit_glmm_nb_sparse(
                        x,
                        y,
                        n,
                        p,
                        &sized,
                        &ids.primary,
                        &ids.extra,
                        start,
                        opts,
                    ),
                    Family::Binomial { .. } | Family::Poisson { .. } | Family::Gamma { .. } => {
                        crate::sparse::fit_glmm_sparse(
                            x,
                            y,
                            n,
                            p,
                            &sized,
                            &ids.primary,
                            &ids.extra,
                            f64::NAN,
                            start,
                            opts,
                        )
                        .0
                    }
                },
            }
        }
    }
}

/// RE θ length for a mixed model, from topology ALONE (independent of level
/// counts): primary vech `q_p(q_p+1)/2` plus a `vech(Λ_g)` block `q_g(q_g+1)/2`
/// per extra grouping. Equals `LmmGroupings::n_theta()` for the same spec; used to
/// validate a `StartValues.theta` at the stable boundary before the workspace
/// (which is where `n_theta` otherwise first materializes) is built. `None`
/// (fixed-only) → 0.
fn theta_width(re: Option<&ReStructure>) -> usize {
    let Some(re) = re else { return 0 };
    let q_p = 1 + re.slopes.len();
    let mut w = q_p * (q_p + 1) / 2;
    for g in &re.extra_groupings {
        let q_g = 1 + g.slopes.len();
        w += q_g * (q_g + 1) / 2;
    }
    w
}

/// vech(column-major lower-tri) of `D̂ = scale·Λ̂Λ̂'` for one `q×q` grouping.
/// `theta_block` is that grouping's column-major lower-tri `vech(Λ)` prefix
/// (as `primary_lambda` reads it). `D[r][c] = Σ_{k≤min(r,c)} Λ[r][k]·Λ[c][k]`.
fn varcorr_block(theta_block: &[f64], q: usize, scale: f64) -> Vec<f64> {
    let mut lam = vec![0.0f64; q * q];
    crate::lmm::primary_lambda(theta_block, q, &mut lam); // Λ lower-tri, row-major
    let mut vech = Vec::with_capacity(q * (q + 1) / 2);
    for c in 0..q {
        for r in c..q {
            let mut d = 0.0;
            for k in 0..=c {
                d += lam[r * q + k] * lam[c * q + k];
            }
            vech.push(scale * d);
        }
    }
    vech
}

/// Assemble `Fit::varcorr`: one vech-packed `D̂ = scale·Λ̂Λ̂'` block
/// per grouping, declaration order (primary, then each extra). `scale` is σ̂²
/// (LMM) or 1.0 (GLMM link scale). Path-independent — a function of θ̂ only.
/// The primary-then-extras vech walk mirrors `LmmGroupings::from_cluster_spec_ext`'s
/// `vech_start` layout (`lmm.rs:252-267`) — change together.
/// `pub(crate)` so the sparse-Z path (`sparse::fit_mle_sparse`) recovers varcorr
/// from θ̂ through the same path-independent assembly as the NoZ `fit_mle`.
pub(crate) fn assemble_varcorr(
    theta: &[f64],
    groupings: &crate::lmm::LmmGroupings,
    scale: f64,
) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(1 + groupings.extra_q.len());
    let mut cursor = 0usize;
    for &q in std::iter::once(&groupings.primary_q).chain(groupings.extra_q.iter()) {
        let vech = q * (q + 1) / 2;
        out.push(varcorr_block(&theta[cursor..cursor + vech], q, scale));
        cursor += vech;
    }
    out
}

/// A `ModelSpec` copy whose RE **counts** are derived from the supplied `GroupIds`
/// (`n_primary = max(primary)+1`; per crossed grouping `n_clusters =
/// max(extra[g])+1`; per nested grouping `n_per_parent = ⌈children/n_primary⌉`,
/// `children = max(extra[g])+1`). Topology tags (`Crossed`/`NestedWithin`),
/// family, and slope columns are preserved. This is the data path's mechanism for
/// "derive every level count from the ids": the sizing-corrected copy
/// feeds the existing workspace builders unchanged, so no builder/kernel signature
/// changes. `re: None` is returned as-is (fixed-only carries no counts).
fn spec_sized_from_ids(model: &ModelSpec, ids: &GroupIds) -> ModelSpec {
    let Some(re) = model.re.as_ref() else {
        return model.clone();
    };
    let level_count = |v: &[u32]| v.iter().copied().max().map(|m| m as usize + 1).unwrap_or(1);
    let n_primary = level_count(&ids.primary);
    let extra_groupings: Vec<Grouping> = re
        .extra_groupings
        .iter()
        .enumerate()
        .map(|(g, gr)| {
            let relation = match gr.relation {
                GroupingRelation::Crossed { .. } => {
                    let children = level_count(&ids.extra[g]);
                    GroupingRelation::Crossed {
                        n_clusters: children as u32,
                    }
                }
                GroupingRelation::NestedWithin { .. } => {
                    // Per-parent distinct-child-id sets, counted directly from the
                    // (primary, extra) row pairs — layout-agnostic (works whether
                    // or not the frontend's ids are contiguous per-parent blocks),
                    // unlike deriving from a single global `max(extra)+1` divided
                    // by `n_primary`, which under-sizes any parent with an
                    // above-average child count (the true fix for unbalanced
                    // nesting: `n_per_parent` must be the TRUE max, not a
                    // `children.div_ceil(n_primary)` global average).
                    let mut per_parent: Vec<std::collections::HashSet<u32>> =
                        vec![Default::default(); n_primary];
                    for (&p, &c) in ids.primary.iter().zip(&ids.extra[g]) {
                        per_parent[p as usize].insert(c);
                    }
                    let n_per_parent = per_parent.iter().map(|s| s.len()).max().unwrap_or(1).max(1);
                    GroupingRelation::NestedWithin {
                        n_per_parent: n_per_parent as u32,
                    }
                }
            };
            Grouping {
                relation,
                slopes: gr.slopes.clone(),
            }
        })
        .collect();
    ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_primary as u32,
            },
            slopes: re.slopes.clone(),
            extra_groupings,
        }),
    }
}

/// Lower-tri Gram `G = XᵀX` (from row-major X) → `aliased_columns` mask. `p` is
/// tiny, so the extra `O(N·p²)` reduction for the drop decision is negligible
/// (the reduced fit recomputes its own suff stats).
fn detect_aliased(x: &[f64], n: usize, p: usize) -> Vec<bool> {
    let mut gram = Mat::<f64>::zeros(p, p);
    for i in 0..n {
        for a in 0..p {
            let xa = x[i * p + a];
            for b in 0..=a {
                gram[(a, b)] += xa * x[i * p + b];
            }
        }
    }
    crate::ols::aliased_columns(gram.as_ref(), p, crate::ols::ALIAS_EPS)
}

/// Remap a spec's RE slope x-column indices through the kept-columns map
/// (`to_reduced[orig]` = reduced index, or `usize::MAX` if that column was
/// dropped). An aliased column that is ALSO an RE slope is a rank-deficient
/// random slope — out of scope for #4; fault honestly rather than silently
/// mis-index.
fn remap_spec_slopes(model: &ModelSpec, to_reduced: &[usize]) -> ModelSpec {
    let Some(re) = model.re.as_ref() else {
        return model.clone();
    };
    let remap = |cols: &[u32]| -> Vec<u32> {
        cols.iter()
            .map(|&c| {
                let r = to_reduced[c as usize];
                assert!(
                    r != usize::MAX,
                    "rank-deficient random-slope column {c}: an aliased fixed column is used as an RE slope (unsupported)"
                );
                r as u32
            })
            .collect()
    };
    let extra_groupings = re
        .extra_groupings
        .iter()
        .map(|g| Grouping {
            relation: g.relation.clone(),
            slopes: remap(&g.slopes),
        })
        .collect();
    ModelSpec {
        family: model.family,
        re: Some(ReStructure {
            sizing: re.sizing.clone(),
            slopes: remap(&re.slopes),
            extra_groupings,
        }),
    }
}

/// Fit the reduced (aliased-columns-dropped) model and scatter β/se back to full
/// width: retained slots take the reduced fit, aliased slots are NaN,
/// `converged` follows the reduced fit, `tau2`/`varcorr`/`dispersion` pass through
/// (the RE structure is unchanged; only fixed-column indices are remapped). The
/// reduced design is full-rank, so the recursive `fit_warm` never re-enters here.
#[allow(clippy::too_many_arguments)]
fn fit_rank_deficient(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    start: Option<&StartValues>,
    opts: &FitOptions,
    aliased: &[bool],
) -> Fit {
    let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
    let pk = kept.len();
    let mut to_reduced = vec![usize::MAX; p];
    for (r, &orig) in kept.iter().enumerate() {
        to_reduced[orig] = r;
    }
    // Reduced design (drop aliased columns), row-major.
    let mut xr = vec![0.0f64; n * pk];
    for i in 0..n {
        for (r, &orig) in kept.iter().enumerate() {
            xr[i * pk + r] = x[i * p + orig];
        }
    }
    let model_r = remap_spec_slopes(model, &to_reduced);
    // StartValues.beta is p-wide → reduce it; theta is RE-only, unchanged.
    let start_r = start.map(|s| StartValues {
        beta: kept.iter().map(|&o| s.beta[o]).collect(),
        theta: s.theta.clone(),
    });
    // Targets: drop aliased targets, reindex survivors into the reduced design.
    let targets_r: Vec<u32> = opts
        .target_indices
        .iter()
        .filter(|&&t| !aliased[t as usize])
        .map(|&t| to_reduced[t as usize] as u32)
        .collect();
    let opts_r = FitOptions {
        target_indices: targets_r,
        wald_se: opts.wald_se,
        nagq: opts.nagq,
        dispersion: opts.dispersion,
        // Per-row, so dropping fixed-effect COLUMNS leaves them untouched.
        weights: opts.weights.clone(),
        parallel_inner: opts.parallel_inner,
    };
    let fr = fit_warm(&xr, y, n, pk, &model_r, ids, start_r.as_ref(), &opts_r);
    // Scatter reduced β/se to full width; aliased slots stay NaN.
    let mut beta = vec![f64::NAN; p];
    let mut se = vec![f64::NAN; p];
    for (r, &orig) in kept.iter().enumerate() {
        beta[orig] = fr.beta[r];
        se[orig] = fr.se[r];
    }
    Fit {
        beta,
        se,
        tau2: fr.tau2,
        dispersion: fr.dispersion,
        converged: fr.converged,
        varcorr: fr.varcorr,
        stddev_se: fr.stddev_se,
        aliased: aliased.to_vec(),
    }
}

/// Engine-invariant shape check for the data-shaped ids (mirrors `fit_grouped`'s
/// former `cluster_ids.len()==n` panic): the primary id vector is length `n`, the
/// extra vectors align 1:1 with the declared extra groupings, and each is length
/// `n`. A malformed `GroupIds` is an engine invariant violation, so this panics
/// (consistent with the `fit` panic convention), not a `converged:false` return.
fn assert_group_ids(re: &ReStructure, ids: &GroupIds, n: usize) {
    assert_eq!(
        ids.primary.len(),
        n,
        "GroupIds.primary must have n elements"
    );
    assert_eq!(
        ids.extra.len(),
        re.extra_groupings.len(),
        "GroupIds.extra must align 1:1 with re.extra_groupings (declaration order)"
    );
    for (g, e) in ids.extra.iter().enumerate() {
        assert_eq!(e.len(), n, "GroupIds.extra[{g}] must have n elements");
    }
}

/// Mirror of MCPower's contract invariants 19/21 for the standalone `fit` path:
/// the kernel's stack scratch is sized off `MAX_PRIMARY_Q`/`MAX_EXTRA_Q`, so a
/// `q` over the cap would overflow it, and every slope column must index into the
/// `p`-wide design. A malformed spec is an engine invariant violation (see the
/// `fit` panic convention), so this asserts rather than returning a `Fit`.
/// Fixed-only models (`re: None`) carry no RE caps to check.
fn assert_model_shape(model: &ModelSpec, p: usize, nagq: u8) {
    // nAGQ: odd, 1..=MAX_NAGQ; >1 only on a single scalar-intercept binomial/
    // Poisson GLMM. Checked before the RE early-return so even fixed-only specs
    // can't smuggle a bad nagq through. Sourced from `FitOptions` (M3.5), not the
    // spec.
    assert!(
        (1..=crate::consts::MAX_NAGQ).contains(&nagq) && nagq % 2 == 1,
        "nagq={} must be odd in 1..={}",
        nagq,
        crate::consts::MAX_NAGQ
    );
    if nagq > 1 {
        let re = model
            .re
            .as_ref()
            .expect("nagq>1 requires a mixed model (re: Some)");
        let scalar_intercept = re.slopes.is_empty() && re.extra_groupings.is_empty();
        let agq_family = matches!(
            model.family,
            Family::Binomial { .. } | Family::Poisson { .. }
        );
        assert!(
            scalar_intercept && agq_family,
            "nagq>1 legal only on a single scalar-intercept binomial/Poisson GLMM"
        );
    }
    let Some(re) = model.re.as_ref() else {
        return;
    };
    // d1 #2: the RE-envelope caps (extra-grouping count, q_p, q_g) that used to
    // panic here are now `classify_design`'s routing boundary — over-envelope
    // designs route to the sparse path instead of aborting. Only the column-
    // bounds validity asserts remain below (they hold regardless of solver).
    for &col in &re.slopes {
        assert!(
            (col as usize) < p,
            "primary slope column {col} out of range (p={p})"
        );
    }
    for g in &re.extra_groupings {
        for &col in &g.slopes {
            assert!(
                (col as usize) < p,
                "extra-grouping slope column {col} out of range (p={p})"
            );
        }
    }
}

/// Test-only re-export of [`assert_model_shape`] so `spec.rs` can exercise the
/// `nagq` shape-check without routing through a full `fit` call.
#[cfg(test)]
pub fn assert_model_shape_pub(model: &ModelSpec, p: usize, nagq: u8) {
    assert_model_shape(model, p, nagq);
}

/// Test-only re-export of [`spec_sized_from_ids`] so the sparse-Z path's
/// equivalence test (`sparse::tests`) can size a spec from ids exactly as the
/// stable `fit_warm` entry does, then force the sparse path directly.
#[cfg(test)]
pub(crate) fn spec_sized_from_ids_pub(model: &ModelSpec, ids: &GroupIds) -> ModelSpec {
    spec_sized_from_ids(model, ids)
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

/// Which LMM/GLMM solver a design routes to. `NoZ` is the dense
/// no-Z fast path with bounded stack scratch (the `MAX_*` envelope), kept
/// byte-identical. `Sparse` is the heap sparse-Z path that lifts the caps. The
/// boundary is currently hard-coded at the cap edge; pulling in-envelope-but-large
/// designs into `Sparse` for speed is future, measured work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Solver {
    NoZ,
    Sparse,
}

/// Route a design to `NoZ` (in-envelope) or `Sparse` (over-envelope). The
/// RE-envelope caps — extra-grouping count, primary width `q_p`, per-extra
/// width `q_g` — are a scratch ceiling, not a model ceiling: over the caps
/// routes to `Sparse` instead of failing. Fixed-only and nAGQ legality are
/// handled by `assert_model_shape` (which still runs first and still panics on
/// true invariant violations).
///
/// On top of the structural caps, slope-carrying extra groupings always route
/// `Sparse` — see the comment at the clause for the per-family rationale.
pub(crate) fn classify_design(model: &ModelSpec, _nagq: u8) -> Solver {
    let Some(re) = model.re.as_ref() else {
        return Solver::NoZ; // no random effects ⇒ nothing to make sparse
    };
    let q_p = 1 + re.slopes.len();
    let over = re.extra_groupings.len() > MAX_EXTRA_GROUPINGS
        || q_p > MAX_PRIMARY_Q
        || re
            .extra_groupings
            .iter()
            .any(|g| 1 + g.slopes.len() > MAX_EXTRA_Q);
    // Extra-grouping random slopes always route Sparse. Gaussian: measured
    // (d2 Phase-1 crossover sweep, 2026-07-02, locked machine — Sparse wins
    // 4–13× across q_g ∈ {2,3,4}, n_extra ∈ {2,4,6}; intercept-only extras
    // stay NoZ, 2–1500× the other way). Non-Gaussian: the dense NoZ GLMM
    // kernel builds intercept-only extras exclusively (`glmm::build_z` emits
    // no slope columns for extras; `apply_lambda`/`build_packed_m` carry
    // debug_asserts), so Sparse — whose PIRLS applies full q_g×q_g Λ-blocks
    // per extra level — is the only implemented route.
    let slope_extras = re.extra_groupings.iter().any(|g| !g.slopes.is_empty());
    if over || slope_extras {
        Solver::Sparse
    } else {
        Solver::NoZ
    }
}

#[cfg(test)]
pub(crate) fn classify_design_pub(model: &ModelSpec, nagq: u8) -> Solver {
    classify_design(model, nagq)
}

/// Placeholder for the M3 families/links not yet wired through `fit` (Tasks 3–7
/// replace these arms with real kernels). Returns the standard non-converged NaN
/// signal — truthful "not yet fittable", never reached by a test until its
/// family's kernel lands.
fn fit_unsupported_family(p: usize) -> Fit {
    Fit {
        beta: vec![f64::NAN; p],
        se: vec![f64::NAN; p],
        tau2: vec![],
        dispersion: f64::NAN,
        converged: false,
        varcorr: vec![],
        stddev_se: vec![],
        aliased: vec![false; p],
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
    // (OLS/GLM are target-compact; LME/LMM are predictor-indexed)
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
        dispersion: 1.0,
        converged,
        varcorr: vec![],
        stddev_se: vec![],
        aliased: vec![false; p],
    }
}

// ---------------------------------------------------------------------------
// LMM dispatch (Estimator::Mle)
// ---------------------------------------------------------------------------

/// LMM dispatch adapter. `cluster_ids`/`extra_ids` are the per-row level ids from
/// the entry's [`GroupIds`]; `model` is the sizing-corrected spec (counts derived
/// from those ids). Slope x-columns come from `model.re`. Mirrors `fit_glmm`.
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

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let p1 = p.max(1);
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p1);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }

    ws.suff.reset();
    if n > 0 && p > 0 {
        ws.suff
            .add_rows_multi(x_mat.as_ref().subrows(0, n), y, cluster_ids, extra_ids);
    }

    // Warm start threads `theta` only — the LMM β is solved exactly given θ, so a
    // β start is irrelevant (matches StartValues carrying no LMM β use). `None`
    // (cold) uses the kernel's THETA0 blind start.
    let lmm_fit = fit_lmm(
        &mut ws,
        &opts.target_indices,
        start.map(|s| s.theta.as_slice()),
    );

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

    // varcorr[g] = vech(σ̂²·Λ̂_gΛ̂_g') per grouping — the q≥2-valid
    // covariance; equals tau2's diagonal only at the (0,0) primary entry.
    let varcorr = if lmm_fit.converged {
        assemble_varcorr(&ws.theta, &ws.suff.groupings, sigma_sq)
    } else {
        vec![]
    };

    Fit {
        beta,
        se,
        tau2,
        dispersion: 1.0,
        converged: lmm_fit.converged,
        varcorr,
        stddev_se: vec![], // LMM has no Hessian SE machinery
        aliased: vec![false; p],
    }
}

// ---------------------------------------------------------------------------
// GLM dispatch (Binomial{Logit}, re: None)
// ---------------------------------------------------------------------------

/// GLM dispatch adapter. Owns the `irls_*` scratch inline (the analog of
/// `fit_ols`'s `OlsScratch` allocation; on the simulation path these live in
/// `SimWorkspace`), converts the row-major input to a column-major faer `Mat`,
/// runs the `family`-selected IRLS kernel cold-started at β=0, and maps the view
/// to `Fit`. No random effects ⇒ `tau2` empty. Binomial/Poisson keep
/// `dispersion = 1.0`; Gamma/NB recover their dispersion in their own arms.
fn fit_glm(
    family: Family,
    nb_theta: f64,
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    opts: &FitOptions,
) -> Fit {
    let t = opts.target_indices.len();
    let n1 = n.max(1);
    let p1 = p.max(1);
    let t1 = t.max(1);

    // --- irls_* scratch, sized off n/p/t exactly as GlmScratch documents ---
    let mut irls_eta = vec![0.0f64; n1];
    let mut irls_p = vec![0.0f64; n1];
    let mut irls_w = vec![0.0f64; n1];
    let mut irls_z = vec![0.0f64; n1];
    let mut irls_betas = vec![0.0f64; p1];
    let mut irls_betas_new = vec![0.0f64; p1];
    let mut irls_var_diag = vec![0.0f64; t1];
    let mut irls_t_sq = vec![0.0f64; t1];
    let mut irls_u_scratch = vec![0.0f64; p1];
    let mut irls_xtwx = Mat::<f64>::zeros(p1, p1);
    let mut irls_xtwz = vec![0.0f64; p1];
    let mut irls_l = Mat::<f64>::zeros(p1, p1);
    let mut irls_wx = vec![0.0f64; n1 * p1]; // column-major W∘X, needs ≥ n·p

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let mut x_mat = Mat::<f64>::zeros(n1, p1);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }

    let view = {
        let scratch = GlmScratch {
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
        };
        // None = β=0 cold start (no spec-derived truth on the standalone path).
        glm_irls_fit(
            family,
            nb_theta,
            x_mat.as_ref().subrows(0, n),
            y,
            &opts.target_indices,
            None,
            scratch,
        )
    };

    // --- map GlmFitView → Fit ---
    // view.betas is full [0..p]; view.var_diag is target-compact [0..t] (like OLS).
    let beta = view.betas.to_vec();
    let converged = view.converged;
    let mut se = vec![f64::NAN; p];
    for (i, &ti) in opts.target_indices.iter().enumerate() {
        let vd = view.var_diag[i];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }

    // Dispersion. Binomial/Poisson hold φ≡1 (the kernel's `(XᵀWX)⁻¹` is the full
    // covariance). Gamma recovers φ post-fit — the mean model β is φ-independent,
    // so φ stays out of the IRLS — and scales the SE by √φ (the kernel folded
    // φ=1, so Var(β̂)=φ·(XᵀWX)⁻¹). `dispersion: Some(v)` holds φ=v fixed; `None`
    // estimates the Pearson moment `φ̂=Σ rᵢ²/(n−p)`, `rᵢ=(yᵢ−μ̂ᵢ)/√V(μ̂ᵢ)` —
    // exactly `summary(glm(family=Gamma))$dispersion`.
    let dispersion = match family {
        Family::Gamma { .. } if converged => {
            let phi = match opts.dispersion {
                Some(v) => v,
                None => {
                    let mut s = 0.0;
                    for i in 0..n {
                        let mut eta = 0.0;
                        for j in 0..p {
                            eta += x[i * p + j] * beta[j];
                        }
                        let mu = crate::family::link_inv(family, eta);
                        let r = (y[i] - mu) / crate::family::variance(family, nb_theta, mu).sqrt();
                        s += r * r;
                    }
                    s / (n - p) as f64
                }
            };
            let sqrt_phi = phi.sqrt();
            for v in se.iter_mut() {
                if v.is_finite() {
                    *v *= sqrt_phi;
                }
            }
            phi
        }
        _ => 1.0,
    };

    Fit {
        beta,
        se,
        tau2: vec![],
        dispersion,
        converged,
        varcorr: vec![],
        stddev_se: vec![],
        aliased: vec![false; p],
    }
}

// ---------------------------------------------------------------------------
// Negative-binomial GLM — alternating outer-θ loop (MASS::glm.nb style)
// ---------------------------------------------------------------------------

/// NB θ outer-loop caps. Convergence on `|Δθ|/θ < NB_THETA_TOL`; `NB_MAX_OUTER`
/// alternations between the β fit (fixed θ) and the 1-D θ optimisation.
const NB_MAX_OUTER: usize = 25;
const NB_THETA_TOL: f64 = 1e-6;
/// θ search bracket (on ln θ): `θ ∈ [1e-3, 1e4]`. Below → ≈Poisson (huge θ);
/// above is implausible overdispersion for the parity datasets.
const NB_THETA_LO: f64 = 1e-3;
const NB_THETA_HI: f64 = 1e4;

/// NB profile log-likelihood in θ at fixed μ̂, up to the θ-independent `−ln(yᵢ!)`:
/// `Σ[ lnΓ(yᵢ+θ) − lnΓ(θ) + θ·ln(θ/(θ+μ̂ᵢ)) + yᵢ·ln(μ̂ᵢ/(θ+μ̂ᵢ)) ]`. Counts are
/// integers, so `lnΓ(y+θ)−lnΓ(θ) = Σ_{k=0}^{y−1} ln(θ+k)` exactly — no lgamma,
/// and identical to `MASS::theta.ml`'s objective. (`Σ_{k}` is `O(Σy)`; fine at
/// parity scale.)
///
/// The `μᵢ==0` guard makes this finite when evaluated at the **saturated** mean
/// `μ=y` (so `nb_profile_loglik(y, y, θ)` = the NB saturated log-likelihood, the
/// term the NB GLMM marginal-θ objective adds back to the Laplace deviance): at a
/// zero count `μ=y=0`, both mean terms vanish in the limit, but `0·ln(0)` would
/// otherwise produce a `NaN`. For the GLM caller `μ̂>0` always, so the guard never
/// fires there.
pub(crate) fn nb_profile_loglik(y: &[f64], mu: &[f64], theta: f64) -> f64 {
    let mut ll = 0.0;
    for (&yi, &mi) in y.iter().zip(mu.iter()) {
        let mut s = 0.0;
        for k in 0..(yi.round() as u64) {
            s += (theta + k as f64).ln();
        }
        if mi > 0.0 {
            s += theta * (theta / (theta + mi)).ln() + yi * (mi / (theta + mi)).ln();
        }
        ll += s;
    }
    ll
}

/// Maximise `g(ln θ)` over `ln θ ∈ [ln NB_THETA_LO, ln NB_THETA_HI]` by
/// golden-section (the NB likelihood is far more symmetric in ln θ than in θ).
/// Returns `θ̂ = exp(argmax)`. Shared by the GLM conditional θ profile
/// ([`optimize_nb_theta`]) and the GLMM marginal-θ objective in [`fit_glmm_nb`];
/// `g` is the log-likelihood to maximise as a function of `ln θ`.
pub(crate) fn golden_max_ln_theta(g: impl Fn(f64) -> f64) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9; // 1/golden ratio
    let (mut a, mut b) = (NB_THETA_LO.ln(), NB_THETA_HI.ln());
    let mut c = b - (b - a) * INV_PHI;
    let mut d = a + (b - a) * INV_PHI;
    let (mut fc, mut fd) = (g(c), g(d));
    for _ in 0..200 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * INV_PHI;
            fc = g(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * INV_PHI;
            fd = g(d);
        }
        if (b - a).abs() < 1e-8 {
            break;
        }
    }
    (0.5 * (a + b)).exp()
}

/// Maximise [`nb_profile_loglik`] over θ at fixed μ̂. Returns θ̂.
fn optimize_nb_theta(y: &[f64], mu: &[f64]) -> f64 {
    golden_max_ln_theta(|t| nb_profile_loglik(y, mu, t.exp()))
}

/// Negative-binomial GLM via the alternating outer-θ loop (`MASS::glm.nb`):
/// (1) fit the GLM at fixed θ; (2) 1-D maximise the NB profile log-likelihood
/// over θ holding β̂/μ̂; (3) repeat to convergence. `theta_seed = Some(v)` seeds
/// step 1; `None` cold-starts from a method-of-moments estimate
/// `θ₀ = ȳ²/max(s²−ȳ, ε)`. The β SE conditions on θ̂ (lme4/MASS convention;
/// θ-uncertainty is out of scope), so `dispersion = θ̂` and the SE comes straight
/// from the final fixed-θ fit (NB has `φ≡1`; overdispersion lives in `V=μ+μ²/θ`).
fn fit_glm_nb(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    theta_seed: Option<f64>,
    opts: &FitOptions,
) -> Fit {
    let mut theta = theta_seed.unwrap_or_else(|| {
        let ybar = y.iter().sum::<f64>() / n as f64;
        let var = y.iter().map(|&yi| (yi - ybar).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64;
        (ybar * ybar / (var - ybar).max(1e-6)).clamp(NB_THETA_LO, NB_THETA_HI)
    });

    let family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let mut fit_result = fit_unsupported_family(p);
    for _ in 0..NB_MAX_OUTER {
        // θ is fixed for this β fit and threaded explicitly (the spec is θ-free).
        fit_result = fit_glm(family, theta, x, y, n, p, opts);
        if !fit_result.converged {
            break;
        }
        // μ̂ = exp(Xβ̂) for the θ optimisation.
        let mu: Vec<f64> = (0..n)
            .map(|i| {
                let eta: f64 = (0..p).map(|j| x[i * p + j] * fit_result.beta[j]).sum();
                crate::family::link_inv(family, eta)
            })
            .collect();
        let new_theta = optimize_nb_theta(y, &mu);
        let converged = (new_theta - theta).abs() / theta < NB_THETA_TOL;
        theta = new_theta;
        if converged {
            // Final β/SE at the converged θ for consistency.
            fit_result = fit_glm(family, theta, x, y, n, p, opts);
            break;
        }
    }
    fit_result.dispersion = theta;
    fit_result
}

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
pub(crate) fn glm_warm_start_beta(
    family: Family,
    nb_theta: f64,
    x: faer::MatRef<f64>,
    y: &[f64],
    n: usize,
    p: usize,
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

#[allow(clippy::too_many_arguments)] // marshals the kernel's (x, y, n, p, spec, ids…) surface
fn fit_glmm(
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
    let re = model
        .re
        .as_ref()
        .expect("fit_glmm requires a mixed model (re: Some)");
    // slope_cols: x column indices for the primary RE slopes (empty = intercept-only).
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();

    // Workspace for this (spec, n) shape — sizes per-cluster solver buffers off
    // re.sizing's cluster count; the kernels cold-start θ from their blind θ₀.
    let mut ws = GlmmWorkspace::for_cluster_spec(p, model, n, &slope_cols, opts.nagq);
    // NB θ̂ is threaded explicitly (the spec is θ-free); the PIRLS/AGQ variance and
    // deviance read it off the workspace. NaN for every non-NB family (unread).
    ws.nb_theta = nb_theta;
    let n_theta = ws.n_theta;

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    let p1 = p.max(1);
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p1);
    for i in 0..n {
        for j in 0..p {
            x_mat[(i, j)] = x[i * p + j];
        }
    }

    // Degenerate guard (mirrors the kernel's n≤p short-circuit contract).
    if n == 0 || p == 0 {
        return (
            Fit {
                beta: vec![f64::NAN; p],
                se: vec![f64::NAN; p],
                tau2: vec![f64::NAN; n_theta],
                dispersion: f64::NAN,
                converged: false,
                varcorr: vec![],
                stddev_se: vec![],
                aliased: vec![false; p],
            },
            vec![],
            f64::INFINITY,
        );
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
            x_mat.as_ref().subrows(0, n),
            y,
            n,
            p,
        ),
    };
    let glmm_fit = crate::glmm::fit_glmm(
        &mut ws,
        x_mat.as_ref().subrows(0, n),
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
    for &ti in &opts.target_indices {
        let vd = ws.var_diag[ti as usize];
        if vd.is_finite() && vd >= 0.0 {
            se[ti as usize] = vd.sqrt();
        }
    }

    // tau2[k] = σ²·θ̂[k]². lme4 parametrizes the RE covariance as σ²·θθ', so VarCorr
    // reports sd = σ·θ̂; our internal λ̂ = ws.params[..n_theta] IS that relative factor
    // θ̂ (the Laplace penalty is the unit ‖u‖²). For binomial/Poisson/NB the residual
    // scale σ²≡1, but Gamma's σ² = pwrss/n = (Pearson χ² + ‖û‖²)/n ≠ 1, so its
    // variance components carry it. (Distinct from `dispersion` below — that is the
    // Pearson/(n−p) moment lme4 reports separately, a different quantity.) Same
    // q≥2-slope caveat as fit_mle's tau2.
    let tau2: Vec<f64> = if glmm_fit.converged {
        let sigma_sq =
            crate::family::glmm_sigma_sq(model.family, &y[..n], &ws.prob[..n], &ws.u[..ws.k]);
        ws.params[..n_theta]
            .iter()
            .map(|&t| t * t * sigma_sq)
            .collect()
    } else {
        vec![f64::NAN; n_theta]
    };

    // Dispersion. Binomial/Poisson hold φ≡1. Gamma recovers the Pearson moment
    // estimator on the conditional-mode residuals (μ̂ = ws.prob after the pinned-γ̂
    // re-eval): `φ̂ = Σ rᵢ²/(n−p)`, `rᵢ = (yᵢ−μ̂ᵢ)/√V(μ̂ᵢ)`. It does NOT rescale the
    // SE here — the kernel already reports each arm on lme4's convention: Hessian
    // unscaled (`vcov(use.hessian=TRUE)`, oracle-settled) and Rx carrying σ̂² =
    // pwrss/n (`vcov(use.hessian=FALSE)`; `family::glmm_sigma_sq`, a DIFFERENT
    // quantity than this φ̂). NB θ̂ is set by the outer-θ wrapper, not here.
    let dispersion = match model.family {
        Family::Gamma { .. } if glmm_fit.converged => match opts.dispersion {
            Some(v) => v,
            None => {
                let mut s = 0.0;
                for (&yi, &mu) in y[..n].iter().zip(ws.prob[..n].iter()) {
                    let r = (yi - mu) / crate::family::variance(model.family, nb_theta, mu).sqrt();
                    s += r * r;
                }
                s / (n - p) as f64
            }
        },
        _ => 1.0,
    };

    // GLMM D̂ = Λ̂Λ̂' (dispersion ≡ 1 on the link scale; the
    // glmm/mod.rs:502 case generalized to the full block).
    let varcorr = if glmm_fit.converged {
        assemble_varcorr(&ws.params[..n_theta], &ws.groupings, 1.0)
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

    let mu_hat = ws.prob[..n].to_vec();
    (
        Fit {
            beta,
            se,
            tau2,
            dispersion,
            converged: glmm_fit.converged,
            varcorr,
            stddev_se,
            aliased: vec![false; p],
        },
        mu_hat,
        glmm_fit.deviance,
    )
}

/// Negative-binomial GLMM via the **marginal-θ** profile (`lme4::glmer.nb`):
/// optimise the dispersion θ on the *marginal* (Laplace-integrated) likelihood,
/// not the conditional one. For each candidate θ the inner [`fit_glmm`] re-fits
/// the full GLMM (variance components + β) at that fixed θ and returns its
/// minimized marginal Laplace deviance `D(θ)`; the marginal log-likelihood is then
///
/// ```text
///   logL_marginal(θ) = −½·D(θ) + nb_profile_loglik(y, y, θ)
/// ```
///
/// where the second term is the NB **saturated** log-likelihood (the θ-dependent
/// `lnΓ(y+θ)−lnΓ(θ)` normalisation the deviance cancels against its saturated
/// reference — see [`nb_profile_loglik`]'s derivation). Maximising this over `ln θ`
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
fn fit_glmm_nb(
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

    let theta = golden_max_ln_theta(|t| {
        let th = t.exp();
        let (_fit, _mu, dev) =
            fit_glmm(x, y, n, p, &nb_spec, cluster_ids, extra_ids, th, None, opts);
        -0.5 * dev + nb_profile_loglik(y, y, th)
    });

    let mut fit_result = fit_glmm(
        x,
        y,
        n,
        p,
        &nb_spec,
        cluster_ids,
        extra_ids,
        theta,
        None,
        opts,
    )
    .0;
    fit_result.dispersion = theta;
    fit_result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glmm::glmm_laplace_deviance;
    use crate::{BinomialLink, Family, ModelSpec, ReStructure, Sizing, WaldSe};

    #[test]
    fn fit_ols_recovers_slope() {
        // y = 2*x + noise-free → beta[1] ≈ 2
        let n = 20;
        let p = 2;
        let x: Vec<f64> = (0..n).flat_map(|i| [1.0, i as f64]).collect(); // [intercept, x]
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![1],
                ..FitOptions::default()
            },
        );
        assert!(f.converged);
        assert!((f.beta[1] - 2.0).abs() < 1e-6);
    }

    /// Gap #4: an exactly-collinear OLS design (col2 = col0 + col1). glmm must
    /// drop the LATER column (2), mark it in `Fit::aliased`, keep `converged`, and
    /// the retained β must equal a direct fit on the 2-column reduced design
    /// (self-consistency — no external oracle needed for exact collinearity).
    #[test]
    #[should_panic(expected = "weights")]
    fn weights_rejected_off_sparse_binomial_path() {
        // Prior weights are honored only on the sparse binomial GLMM path; every
        // other dispatch must fault at the stable boundary rather than silently
        // fit unweighted (an OLS here).
        let n = 8;
        let x = vec![1.0f64; n];
        let y: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let opts = FitOptions {
            weights: Some(vec![2.0; n]),
            ..FitOptions::default()
        };
        let _ = fit_cold(&x, &y, n, 1, &model, &GroupIds::default(), &opts);
    }

    #[test]
    fn fit_rank_deficient_drops_and_matches_reduced() {
        let n = 30;
        let p = 3;
        let mut st = 7u64;
        let mut x = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let x1 = lcg(&mut st);
            x[i * p] = 1.0;
            x[i * p + 1] = x1;
            x[i * p + 2] = 1.0 + x1; // col2 = col0 + col1 exactly
            y[i] = 0.5 + 0.4 * x1 + 0.3 * lcg(&mut st);
        }
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "reduced OLS must converge");
        assert_eq!(
            f.aliased,
            vec![false, false, true],
            "later collinear column dropped"
        );
        assert!(f.beta[2].is_nan(), "aliased β = NaN");
        assert!(f.se[2].is_nan(), "aliased se = NaN");

        // Direct fit on the 2-column reduced design.
        let mut xr = vec![0.0f64; n * 2];
        for i in 0..n {
            xr[i * 2] = x[i * p];
            xr[i * 2 + 1] = x[i * p + 1];
        }
        let fr = fit_cold(
            &xr,
            &y,
            n,
            2,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1],
                ..FitOptions::default()
            },
        );
        assert!(
            (f.beta[0] - fr.beta[0]).abs() < 1e-9,
            "β0 {} vs reduced {}",
            f.beta[0],
            fr.beta[0]
        );
        assert!(
            (f.beta[1] - fr.beta[1]).abs() < 1e-9,
            "β1 {} vs reduced {}",
            f.beta[1],
            fr.beta[1]
        );
    }

    #[derive(serde::Deserialize)]
    struct ColEst {
        beta: Vec<Option<f64>>,
    }
    #[derive(serde::Deserialize)]
    struct ColGolden {
        estimates: ColEst,
    }

    /// Gap #4 oracle: near-collinear `y ~ 1 + x1 + x2 + x3` (x3 ≈ x1+x2) vs R's
    /// column-drop-and-fit (`parity/goldens/sim_collinear_glm.json`). glmm must
    /// drop the SAME column R marks `NA`, mark it in `Fit::aliased`, and match the
    /// retained β. The oracle is sacred.
    #[test]
    fn fit_sim_collinear_matches_lme4_drop() {
        let raw = include_str!("../parity/goldens/sim_collinear_glm.json");
        let gold: ColGolden = serde_json::from_str(raw).expect("golden JSON parses");

        let csv = include_str!("../parity/data/sim_collinear.csv");
        let mut y = Vec::<f64>::new();
        let mut cols: Vec<[f64; 3]> = Vec::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            cols.push([
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
            ]);
        }
        let n = y.len();
        let p = 4; // intercept + x1 + x2 + x3
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            x[i * p + 1] = cols[i][0];
            x[i * p + 2] = cols[i][1];
            x[i * p + 3] = cols[i][2];
        }
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "reduced fit converges");
        // R's aliased column = the beta slot that is null.
        let r_aliased: Vec<bool> = gold.estimates.beta.iter().map(|b| b.is_none()).collect();
        assert_eq!(
            f.aliased, r_aliased,
            "glmm must drop the same column R does"
        );
        for (j, rb) in gold.estimates.beta.iter().enumerate() {
            match rb {
                Some(v) => assert!(
                    (f.beta[j] - v).abs() / v.abs().max(1e-6) < 1e-3,
                    "β{j} {} vs {v}",
                    f.beta[j]
                ),
                None => assert!(f.beta[j].is_nan(), "β{j} must be NaN (aliased)"),
            }
        }
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

    use crate::{Grouping, GroupingRelation};

    #[test]
    fn theta_width_counts_vech_blocks() {
        // intercept-only primary + 1 slope → q_p=2 → 3; one intercept-only crossed → +1.
        let re = ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 3 },
                slopes: vec![],
            }],
        };
        assert_eq!(super::theta_width(Some(&re)), 3 + 1);
        assert_eq!(super::theta_width(None), 0);
    }

    /// varcorr_block computes vech(scale·ΛΛ') for a hand θ. For q=2 with Λ
    /// (col-major lower-tri vech) [2.0, 0.5, 1.0] → Λ=[[2,0],[0.5,1]],
    /// D=ΛΛ'=[[4,1],[1,1.25]], vech col-major = [4, 1, 1.25]; ×scale.
    #[test]
    fn varcorr_block_is_scaled_lambda_gram() {
        let vech = super::varcorr_block(&[2.0, 0.5, 1.0], 2, 1.0);
        assert_eq!(vech.len(), 3);
        assert!((vech[0] - 4.0).abs() < 1e-12, "D00 {}", vech[0]);
        assert!((vech[1] - 1.0).abs() < 1e-12, "D10 {}", vech[1]);
        assert!((vech[2] - 1.25).abs() < 1e-12, "D11 {}", vech[2]);
        let scaled = super::varcorr_block(&[2.0, 0.5, 1.0], 2, 3.0);
        assert!((scaled[0] - 12.0).abs() < 1e-12);
        assert!((scaled[2] - 3.75).abs() < 1e-12);
    }

    fn fit_with_varcorr(vech: Vec<f64>) -> Fit {
        Fit {
            beta: vec![],
            se: vec![],
            tau2: vec![],
            dispersion: 1.0,
            converged: true,
            varcorr: vec![vech],
            stddev_se: vec![],
            aliased: vec![],
        }
    }

    /// q=1: a scalar block has no off-diagonal — stddev is just sqrt(variance)
    /// and the 1x1 "correlation matrix" is the trivial [[1.0]].
    #[test]
    fn stddev_corr_q1_trivial() {
        let f = fit_with_varcorr(vec![9.0]);
        let (sd, corr) = f.stddev_corr(0);
        assert_eq!(sd, vec![3.0]);
        assert_eq!(corr, vec![vec![1.0]]);
    }

    /// q=2 hand math, mirroring `varcorr_block_is_scaled_lambda_gram`'s D:
    /// D=[[4,1],[1,1.25]] → vech(col-major lower-tri)=[4,1,1.25].
    /// stddev=[2, sqrt(1.25)]; corr01 = 1/(2*sqrt(1.25)).
    #[test]
    fn stddev_corr_q2_hand_math() {
        let f = fit_with_varcorr(vec![4.0, 1.0, 1.25]);
        let (sd, corr) = f.stddev_corr(0);
        let sd1 = 1.25_f64.sqrt();
        assert!((sd[0] - 2.0).abs() < 1e-12);
        assert!((sd[1] - sd1).abs() < 1e-12);
        assert_eq!(corr[0][0], 1.0);
        assert_eq!(corr[1][1], 1.0);
        let rho = 1.0 / (2.0 * sd1);
        assert!((corr[0][1] - rho).abs() < 1e-12);
        assert!((corr[1][0] - rho).abs() < 1e-12);
    }

    /// q=3 hand-computed, chosen specifically to catch a vech-indexing bug:
    /// D = [[4,1,2],[1,9,3],[2,3,16]] (sd = [2,3,4], all off-diagonal terms
    /// distinct so a transposed/misindexed vech would mismatch). Column-major
    /// lower-tri vech walk: c=0 → (D00,D10,D20)=(4,1,2); c=1 → (D11,D21)=(9,3);
    /// c=2 → (D22)=(16). vech = [4,1,2,9,3,16], len=6 ⇒ q=3.
    #[test]
    fn stddev_corr_q3_hand_math() {
        let f = fit_with_varcorr(vec![4.0, 1.0, 2.0, 9.0, 3.0, 16.0]);
        let (sd, corr) = f.stddev_corr(0);
        assert_eq!(sd, vec![2.0, 3.0, 4.0]);
        for i in 0..3 {
            assert_eq!(corr[i][i], 1.0);
        }
        // corr(0,1) = D10/(sd0*sd1) = 1/(2*3)
        assert!((corr[0][1] - 1.0 / 6.0).abs() < 1e-12);
        assert!((corr[1][0] - 1.0 / 6.0).abs() < 1e-12);
        // corr(0,2) = D20/(sd0*sd2) = 2/(2*4) = 0.25
        assert!((corr[0][2] - 0.25).abs() < 1e-12);
        assert!((corr[2][0] - 0.25).abs() < 1e-12);
        // corr(1,2) = D21/(sd1*sd2) = 3/(3*4) = 0.25
        assert!((corr[1][2] - 0.25).abs() < 1e-12);
        assert!((corr[2][1] - 0.25).abs() < 1e-12);
    }

    /// assemble_varcorr emits one block per grouping in declaration order:
    /// primary q=2 (vech [2,0.5,1]) then one scalar extra (θ=0.7, q=1) → D=0.49.
    #[test]
    fn assemble_varcorr_one_block_per_grouping() {
        let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(
            &ModelSpec {
                family: Family::Gaussian,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 4 },
                    slopes: vec![1],
                    extra_groupings: vec![Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 3 },
                        slopes: vec![],
                    }],
                }),
            },
            16,
            &[1],
            &[],
        );
        let theta = [2.0, 0.5, 1.0, 0.7];
        let vc = super::assemble_varcorr(&theta, &g, 1.0);
        assert_eq!(vc.len(), 2);
        assert_eq!(vc[0], vec![4.0, 1.0, 1.25]);
        assert!((vc[1][0] - 0.49).abs() < 1e-12, "extra D {}", vc[1][0]);
    }

    #[test]
    fn spec_sized_from_ids_derives_counts() {
        let re = ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
            slopes: vec![],
            extra_groupings: vec![
                Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![],
                },
                Grouping {
                    relation: GroupingRelation::NestedWithin { n_per_parent: 1 },
                    slopes: vec![],
                },
            ],
        };
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(re),
        };
        let ids = GroupIds {
            primary: vec![0, 1, 2, 0, 1, 2], // 3 primary levels
            extra: vec![vec![0, 0, 1, 1, 2, 2], vec![0, 1, 2, 3, 4, 5]], // crossed 3, nested 6 children
        };
        let sized = super::spec_sized_from_ids(&model, &ids);
        let sre = sized.re.unwrap();
        assert_eq!(sre.sizing, Sizing::FixedClusters { n_clusters: 3 });
        assert_eq!(
            sre.extra_groupings[0].relation,
            GroupingRelation::Crossed { n_clusters: 3 }
        );
        // 6 nested children / 3 parents = 2 per parent.
        assert_eq!(
            sre.extra_groupings[1].relation,
            GroupingRelation::NestedWithin { n_per_parent: 2 }
        );
    }

    /// UNBALANCED nesting: 3 parents with 1, 2, and 3 distinct children
    /// respectively, primary ids ascending with the widest parent last
    /// (`0 → {0}`, `1 → {3,4}`, `2 → {6,7,8}` — the contiguous-per-parent-block
    /// layout `glmm-formula`'s `grouping_ids` now emits, padded to width 3).
    #[test]
    fn spec_sized_from_ids_nested_unbalanced_uses_true_max_per_parent() {
        let re = ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 1 },
                slopes: vec![],
            }],
        };
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(re),
        };
        // primary id 0: 1 row → child {0}; id 1: 2 rows → children {3,4};
        // id 2: 3 rows → children {6,7,8}.
        let ids = GroupIds {
            primary: vec![0, 1, 1, 2, 2, 2],
            extra: vec![vec![0, 3, 4, 6, 7, 8]],
        };
        let sized = super::spec_sized_from_ids(&model, &ids);
        let sre = sized.re.unwrap();
        assert_eq!(sre.sizing, Sizing::FixedClusters { n_clusters: 3 });
        assert_eq!(
            sre.extra_groupings[0].relation,
            GroupingRelation::NestedWithin { n_per_parent: 3 }
        );
    }

    /// Same unbalanced shape, but the WIDEST parent is first in primary order
    /// (id 0 → 3 children `{0,1,2}`, id 1 → 2 `{3,4}`, id 2 → 1 `{5}`). The old
    /// `children.div_ceil(n_primary)` formula computed `⌈6/3⌉ = 2` from the
    /// global `max(extra)+1 = 6` — under-sizing parent 0's true 3-wide block
    /// because the fullest parent isn't the one that sets the global max id.
    /// The true per-parent count (grouping rows by `(primary, extra)` pairs)
    /// gets this right regardless of which parent is fullest.
    #[test]
    fn spec_sized_from_ids_nested_unbalanced_first_parent_widest() {
        let re = ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 1 },
            slopes: vec![],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 1 },
                slopes: vec![],
            }],
        };
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(re),
        };
        let ids = GroupIds {
            primary: vec![0, 0, 0, 1, 1, 2],
            extra: vec![vec![0, 1, 2, 3, 4, 5]],
        };
        let sized = super::spec_sized_from_ids(&model, &ids);
        let sre = sized.re.unwrap();
        assert_eq!(
            sre.extra_groupings[0].relation,
            GroupingRelation::NestedWithin { n_per_parent: 3 }
        );
    }

    /// Mirror of MCPower's `extra_grouping_rejects_too_many_slopes` contract test:
    /// a `q_g = 5` (intercept + 4 slopes) extra grouping is over the `MAX_EXTRA_Q = 4`
    /// NoZ-scratch envelope and routes to Sparse (d1 #2). The sparse numeric path
    /// makes this a *supported* design — it routes to Sparse and
    /// `fit_cold` runs the sparse solver (returning a degenerate non-converged `Fit`
    /// on this n=0 input) instead of hitting the removed stub.
    #[test]
    fn fit_extra_grouping_q_too_large_routes_sparse() {
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 4 },
                    slopes: vec![1, 2, 3, 4], // q_g = 5 > MAX_EXTRA_Q
                }],
            }),
        };
        assert!(matches!(classify_design(&model, 1), Solver::Sparse));
        // n=0: bypasses rank-deficiency detection (guarded by `if n > 0 && p > 0`)
        // so routing reaches classify_design → Sparse → fit_mle_sparse, which now
        // runs (no PD data at n=0 ⇒ non-converged) rather than panicking.
        let n = 0;
        let p = 5; // slopes [1,2,3,4] in-bounds; q_g=5 over the NoZ envelope
        let fit = fit_cold(
            &[],
            &[],
            n,
            p,
            &model,
            &GroupIds::from_sizing(model.re.as_ref().unwrap(), n),
            &FitOptions {
                target_indices: vec![1],
                ..FitOptions::default()
            },
        );
        assert!(!fit.converged);
    }

    /// d1 #2: 7 crossed extras exceed `MAX_EXTRA_GROUPINGS = 6`, the NoZ-scratch
    /// envelope. Over-envelope-by-count designs are now supported: `classify_design`
    /// routes them to Sparse, and the sparse path builds its own cap-free structures
    /// (`SparseLmmWorkspace::new` no longer calls `add_rows_multi`, and
    /// `from_cluster_spec_ext`'s `n_extras <= MAX_EXTRA_GROUPINGS` guard is gone).
    /// So `fit_cold` runs the sparse solver rather than panicking. Mirrors the
    /// sibling `fit_extra_grouping_q_too_large_routes_sparse` (over-envelope by width).
    #[test]
    fn fit_too_many_extra_groupings_routes_sparse() {
        let extra_groupings: Vec<Grouping> = (0..7)
            .map(|_| Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 2 },
                slopes: vec![],
            })
            .collect();
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings,
            }),
        };
        assert!(matches!(classify_design(&model, 1), Solver::Sparse));
        let (n, p) = (8, 1);
        let x = vec![1.0f64; n * p];
        let y = vec![0.0f64; n];
        let ids = GroupIds {
            primary: vec![0; n],
            extra: vec![vec![0; n]; 7],
        };
        // Runs the sparse solver end-to-end without panic; this degenerate
        // (all-zero y) input is non-converged but must return a `Fit`.
        let fit = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0],
                ..FitOptions::default()
            },
        );
        assert_eq!(fit.beta.len(), p);
    }

    /// Anti-panic floor: an over-envelope NON-GAUSSIAN mixed model must return a
    /// `Fit`, never panic — for both over-envelope shapes (over-count: 7 crossed
    /// extras > MAX_EXTRA_GROUPINGS; over-width: a q_g=5 slope block >
    /// MAX_EXTRA_Q) across the wired non-Gaussian families. Holds whether the
    /// over-envelope design routes to a real sparse solver or a non-converged
    /// placeholder — either way, no panic.
    #[test]
    fn fit_over_envelope_non_gaussian_never_panics() {
        let families = [
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            Family::Gamma {
                link: crate::GammaLink::Log,
            },
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
        ];
        for family in families {
            // Over-count: 7 crossed intercept-only extras.
            let extra_groupings: Vec<Grouping> = (0..7)
                .map(|_| Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 2 },
                    slopes: vec![],
                })
                .collect();
            let model = ModelSpec {
                family,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 2 },
                    slopes: vec![],
                    extra_groupings,
                }),
            };
            assert!(matches!(classify_design(&model, 1), Solver::Sparse));
            let (n, p) = (8, 1);
            let x = vec![1.0f64; n * p];
            let y = vec![1.0f64; n];
            let ids = GroupIds {
                primary: vec![0; n],
                extra: vec![vec![0; n]; 7],
            };
            let fit = fit_cold(
                &x,
                &y,
                n,
                p,
                &model,
                &ids,
                &FitOptions {
                    target_indices: vec![0],
                    ..FitOptions::default()
                },
            );
            assert_eq!(fit.beta.len(), p, "{family:?} over-count returns a Fit");

            // Over-width: one crossed extra with a width-5 slope block (q_g = 5).
            let model_w = ModelSpec {
                family,
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 4 },
                    slopes: vec![],
                    extra_groupings: vec![Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 4 },
                        slopes: vec![1, 2, 3, 4],
                    }],
                }),
            };
            assert!(matches!(classify_design(&model_w, 1), Solver::Sparse));
            let (n, p) = (16, 5);
            let mut st = 11u64;
            let x: Vec<f64> = (0..n)
                .flat_map(|_| {
                    let mut r = [0.0f64; 5];
                    r[0] = 1.0;
                    for v in r[1..].iter_mut() {
                        *v = lcg(&mut st);
                    }
                    r
                })
                .collect();
            let y = vec![1.0f64; n];
            let ids = GroupIds {
                primary: (0..n as u32).map(|i| i % 4).collect(),
                extra: vec![(0..n as u32).map(|i| (i / 4) % 4).collect()],
            };
            let fit = fit_cold(
                &x,
                &y,
                n,
                p,
                &model_w,
                &ids,
                &FitOptions {
                    target_indices: vec![1],
                    ..FitOptions::default()
                },
            );
            assert_eq!(fit.beta.len(), p, "{family:?} over-width returns a Fit");
        }
    }

    /// Warm-path wrapper equivalence: `fit_cold(..)` and `fit_warm(.., None, ..)`
    /// return a byte-identical `Fit` — locks "one implementation, two names".
    /// Uses the intercept-only 6-cluster LMM.
    #[test]
    fn fit_cold_equals_fit_warm_none() {
        let (x, y, n, p) = lmm_hand_dataset();
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 6 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);
        let opts = FitOptions {
            target_indices: vec![1, 2],
            ..FitOptions::default()
        };
        let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
        let warm_none = fit_warm(&x, &y, n, p, &model, &ids, None, &opts);
        // Bitwise equality (not PartialEq): non-target SE slots are NaN, and
        // NaN != NaN under `==` — but the two Fits share one code path, so their
        // bit patterns (NaNs included) must match exactly.
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&cold.beta), bits(&warm_none.beta));
        assert_eq!(bits(&cold.se), bits(&warm_none.se));
        assert_eq!(bits(&cold.tau2), bits(&warm_none.tau2));
        assert_eq!(cold.dispersion.to_bits(), warm_none.dispersion.to_bits());
        assert_eq!(cold.converged, warm_none.converged);
    }

    /// Warm-path start-independence: on an LMM the MLE is start-independent, so a
    /// warm fit from a perturbed `StartValues.theta` reaches the same β̂ as the cold
    /// fit to optimizer tolerance — a good start shortens the path without moving the
    /// MLE. n_theta=1 (intercept-only 6-cluster), so theta=[5.0] is
    /// well off the THETA0 blind start.
    #[test]
    fn fit_warm_start_reaches_cold_beta() {
        let (x, y, n, p) = lmm_hand_dataset();
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 6 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let ids = GroupIds::from_sizing(model.re.as_ref().unwrap(), n);
        let opts = FitOptions {
            target_indices: vec![1, 2],
            ..FitOptions::default()
        };
        let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
        let start = StartValues {
            beta: vec![0.0; p],
            theta: vec![5.0],
        };
        let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(&start), &opts);
        assert!(cold.converged && warm.converged, "both fits must converge");
        for j in [1usize, 2] {
            let (a, b) = (cold.beta[j], warm.beta[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-7 || d <= 1e-6 * a.abs().max(b.abs()),
                "LMM MLE must be start-independent: β[{j}] cold {a} vs warm {b}"
            );
        }
    }

    #[test]
    fn fit_lmm_smoke() {
        let (x, y, n, p) = lmm_hand_dataset();
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 6 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::from_sizing(model.re.as_ref().unwrap(), n),
            &FitOptions {
                target_indices: vec![1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "LMM should converge on clean clustered data");
        assert!(
            f.tau2[0].is_finite() && f.tau2[0] >= 0.0,
            "tau2[0] must be a finite non-negative variance, got {}",
            f.tau2[0]
        );
    }

    #[test]
    fn fit_glm_smoke() {
        // Logistic data: P(y=1) = σ(0.4 + 1.0·x), x ~ U(−1, 1), Bernoulli sampled
        // from a second LCG draw → non-separable, so IRLS converges to a finite β̂.
        let n = 400;
        let p = 2;
        let mut st = 7u64;
        let mut x = vec![0.0f64; n * p];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let xi = lcg(&mut st); // U(−1, 1)
            x[i * p] = 1.0;
            x[i * p + 1] = xi;
            let prob = 1.0 / (1.0 + (-(0.4 + 1.0 * xi)).exp());
            let u = (lcg(&mut st) + 1.0) / 2.0; // U(0, 1)
            y[i] = if u < prob { 1.0 } else { 0.0 };
        }
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![1],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "GLM should converge on clean logistic data");
        assert!(
            f.beta.iter().all(|b| b.is_finite()),
            "β̂ must be finite, got {:?}",
            f.beta
        );
        assert!(
            f.beta[1] > 0.0,
            "slope sign should recover positive, got {}",
            f.beta[1]
        );
        assert!(
            f.se[1].is_finite() && f.se[1] > 0.0,
            "target SE must be finite positive, got {}",
            f.se[1]
        );
        assert!(f.tau2.is_empty(), "GLM has no variance components");
    }

    /// Committed cbpp design, expanded to `size` Bernoulli 0/1 rows per record:
    /// `(x [n·4 row-major], y, herd cluster_ids, n)`. Shared by the cbpp oracle
    /// test and `fit_grouped_honors_opts_wald_se`.
    fn cbpp_design() -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
        let csv = include_str!("../parity/data/cbpp.csv");
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut cluster_ids = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let herd: u32 = f[0].parse::<u32>().unwrap() - 1; // herds 1..15 → ids 0..14
            let incidence: u32 = f[1].parse().unwrap();
            let size: u32 = f[2].parse().unwrap();
            let period: u32 = f[3].parse().unwrap();
            let row = [
                1.0,
                f64::from(u32::from(period == 2)),
                f64::from(u32::from(period == 3)),
                f64::from(u32::from(period == 4)),
            ];
            // Expand to `size` Bernoulli trials: `incidence` ones, rest zeros.
            for k in 0..size {
                x.extend_from_slice(&row);
                y.push(if k < incidence { 1.0 } else { 0.0 });
                cluster_ids.push(herd);
            }
        }
        let n = y.len();
        (x, y, cluster_ids, n)
    }

    /// Structure-only cbpp model: `Binomial{Logit}` + a single intercept herd
    /// grouping (15 clusters; explicit ids place each row). Method knobs live in
    /// `FitOptions` now, not here.
    fn cbpp_model() -> ModelSpec {
        ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 15 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        }
    }

    /// `opts.wald_se` (not `model.wald_se`) selects the GLMM Wald-SE denominator:
    /// `Hessian` and `Rx` on the same cbpp fit must produce different SEs. Guards
    /// that the knob lives on `FitOptions`, not `ModelSpec`.
    #[test]
    fn fit_grouped_honors_opts_wald_se() {
        let (x, y, cluster_ids, n) = cbpp_design();
        let p = 4;
        let model = cbpp_model();
        let hess = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                wald_se: WaldSe::Hessian,
                ..FitOptions::default()
            },
        );
        let rx = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                wald_se: WaldSe::Rx,
                ..FitOptions::default()
            },
        );
        assert!(hess.converged && rx.converged);
        assert!(
            (hess.se[1] - rx.se[1]).abs() > 1e-6,
            "Rx vs Hessian SE must differ"
        );
    }

    /// cbpp binomial GLMM through the stable `fit_cold` surface with explicit
    /// `GroupIds` (single grouping), gated against the frozen R `lme4::glmer` oracle
    /// (`parity/results/lme4/cbpp.json`). cbpp is
    /// `cbind(incidence, size−incidence) ~ period + (1 | herd)`; the kernel is
    /// Bernoulli-logit, so each `(incidence, size)` row is expanded to `size` 0/1
    /// rows sharing its design row and herd — value-identical MLE to the aggregated
    /// binomial fit. Herds are unbalanced, so the positional `Sizing` layout cannot
    /// express them: this is the data-shaped-ids path's reason to exist.
    /// SE is compared to **lme4 only** (its Hessian denom keeps the θ–β coupling;
    /// MixedModels.jl drops it ~3% — parity §6). The oracle is sacred: on
    /// disagreement glmm is presumed wrong (parity §1).
    #[test]
    fn fit_glmm_cbpp_matches_lme4() {
        // Frozen lme4 1.1-38 reference (parity/results/lme4/cbpp.json).
        const REF_BETA: [f64; 4] = [
            -1.3983428644712,
            -0.991924974975699,
            -1.12821621594328,
            -1.57974541364914,
        ];
        const REF_SE: [f64; 4] = [
            0.231213976143225,
            0.303150526138057,
            0.32283000769806,
            0.42204890650355,
        ];
        const REF_HERD_SD: f64 = 0.642069927729443; // √τ̂²(herd intercept)

        let (x, y, cluster_ids, n) = cbpp_design();
        let p = 4; // [intercept, period2, period3, period4]
        let model = cbpp_model();
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "cbpp GLMM must converge");
        // Tolerances reflect the measured glmm↔lme4 agreement with margin for
        // optimizer float non-determinism: β abs Δ ≤ 5.7e-4 (rel ~4e-4 — both
        // Laplace/nAGQ=1), herd SD rel 3.0e-4, SE rel ≤ 1.3% (glmm FD-Hessian vs
        // lme4's numerical-differentiation Hessian). The oracle is sacred — these
        // bound glmm to lme4, never the reverse (parity §1).
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() < 2e-3,
                "β[{j}] = {} vs lme4 {} (Δ {})",
                f.beta[j],
                REF_BETA[j],
                (f.beta[j] - REF_BETA[j]).abs()
            );
            // SE vs lme4 only (Hessian denom): relative tolerance covers the
            // FD-Hessian vs lme4 numerical-differentiation difference.
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(
                se_rel < 3e-2,
                "se[{j}] = {} vs lme4 {} (rel {se_rel})",
                f.se[j],
                REF_SE[j]
            );
        }
        // Herd random-intercept SD = √τ̂²; tau2[0] = θ̂² = τ̂² (σ² = 1 binomial).
        let herd_sd = f.tau2[0].sqrt();
        let sd_rel = (herd_sd - REF_HERD_SD).abs() / REF_HERD_SD;
        assert!(
            sd_rel < 3e-3,
            "herd SD = {herd_sd} vs lme4 {REF_HERD_SD} (rel {sd_rel})"
        );
    }

    /// Poisson GLM through stable `fit` (re: None), gated against the frozen R
    /// `glm(family=poisson)` oracle (`parity/goldens/grouseticks_glm.json`):
    /// `TICKS ~ 1 + YEAR + cHEIGHT` on grouseticks, canonical log link. Dispersion
    /// is fixed `φ≡1`, so SE = √((XᵀWX)⁻¹). Routes the Poisson canonical-shortcut
    /// branch of `family.rs`. The oracle is sacred (parity §1).
    #[test]
    fn fit_glm_poisson_matches_r() {
        const REF_BETA: [f64; 4] = [
            1.61599798052329,
            0.409645768793675,
            -1.68514104774929,
            -0.0214518421117811,
        ];
        const REF_SE: [f64; 4] = [
            0.0401455805199035,
            0.0453477934183976,
            0.0898007150621173,
            0.000710396896273056,
        ];
        // grouseticks.csv cols: INDEX,TICKS,BROOD,HEIGHT,YEAR,LOCATION,cHEIGHT.
        let csv = include_str!("../parity/data/grouseticks.csv");
        let p = 4; // [intercept, YEAR96, YEAR97, cHEIGHT]; YEAR base level 95.
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let ticks: f64 = f[1].parse().unwrap();
            let year: u32 = f[4].parse().unwrap();
            let cheight: f64 = f[6].parse().unwrap();
            x.extend_from_slice(&[
                1.0,
                f64::from(u32::from(year == 96)),
                f64::from(u32::from(year == 97)),
                cheight,
            ]);
            y.push(ticks);
        }
        let n = y.len();
        let model = ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "poisson GLM must converge");
        assert!((f.dispersion - 1.0).abs() < 1e-12, "poisson φ≡1");
        assert!(f.tau2.is_empty(), "GLM has no variance components");
        for j in 0..p {
            let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
            assert!(
                b_rel < 1e-3,
                "β[{j}] = {} vs R {} (rel {b_rel})",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(
                se_rel < 3e-2,
                "se[{j}] = {} vs R {} (rel {se_rel})",
                f.se[j],
                REF_SE[j]
            );
        }
    }

    /// Probit binomial GLM through stable `fit` (re: None), gated against frozen
    /// R `glm(binomial("probit"))` (`parity/goldens/cbpp_probit_glm.json`): cbpp
    /// `cbind(incidence, size−incidence) ~ period`, expanded to 0/1 rows (same
    /// MLE + Fisher information as the aggregated fit). Probit is non-canonical →
    /// the general Fisher-scoring branch; `φ≡1`. The oracle is sacred (parity §1).
    #[test]
    fn fit_glm_probit_matches_r() {
        const REF_BETA: [f64; 4] = [
            -0.774138451538547,
            -0.629665092013555,
            -0.693759371053835,
            -0.919560095621316,
        ];
        const REF_SE: [f64; 4] = [
            0.0839559752851447,
            0.150778883932662,
            0.158774972086234,
            0.194512024389745,
        ];
        let csv = include_str!("../parity/data/cbpp.csv");
        let p = 4; // [intercept, period2, period3, period4]
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let incidence: u32 = f[1].parse().unwrap();
            let size: u32 = f[2].parse().unwrap();
            let period: u32 = f[3].parse().unwrap();
            let row = [
                1.0,
                f64::from(u32::from(period == 2)),
                f64::from(u32::from(period == 3)),
                f64::from(u32::from(period == 4)),
            ];
            for k in 0..size {
                x.extend_from_slice(&row);
                y.push(if k < incidence { 1.0 } else { 0.0 });
            }
        }
        let n = y.len();
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Probit,
            },
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "probit GLM must converge");
        for j in 0..p {
            let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
            assert!(
                b_rel < 1e-3,
                "β[{j}] = {} vs R {} (rel {b_rel})",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(
                se_rel < 3e-2,
                "se[{j}] = {} vs R {} (rel {se_rel})",
                f.se[j],
                REF_SE[j]
            );
        }
    }

    /// `y ~ 1 + x + grp` design from the committed `sim_gamma.csv`
    /// (cluster,x,grp,y); X = [intercept, x, grp=="b"]. Shared by the Gamma
    /// goldens.
    fn sim_gamma_xy() -> (Vec<f64>, Vec<f64>, usize) {
        let csv = include_str!("../parity/data/sim_gamma.csv");
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let xv: f64 = f[1].parse().unwrap();
            let grp_b = f64::from(u32::from(f[2] == "b"));
            let yv: f64 = f[3].parse().unwrap();
            x.extend_from_slice(&[1.0, xv, grp_b]);
            y.push(yv);
        }
        let n = y.len();
        (x, y, n)
    }

    /// Gamma log-link GLM, gated against frozen R `glm(family=Gamma("log"))`
    /// (`parity/goldens/sim_gamma_glm.json`). φ is the post-fit Pearson moment
    /// estimator (`dispersion: None`); SE is √φ-scaled, matching R's
    /// `summary()$dispersion`. The oracle is sacred (parity §1).
    #[test]
    fn fit_glm_gamma_log_matches_r() {
        const REF_BETA: [f64; 3] = [0.449945830683142, 0.565796931228723, 0.526238083012209];
        const REF_SE: [f64; 3] = [0.0818215272793177, 0.0596141419705928, 0.119864153173617];
        const REF_DISP: f64 = 1.0286627876062;
        let (x, y, n) = sim_gamma_xy();
        let p = 3;
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "gamma-log GLM must converge");
        let disp_rel = (f.dispersion - REF_DISP).abs() / REF_DISP;
        assert!(disp_rel < 5e-3, "φ = {} vs R {REF_DISP}", f.dispersion);
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
                "β[{j}] = {} vs R {}",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(se_rel < 3e-2, "se[{j}] = {} vs R {}", f.se[j], REF_SE[j]);
        }
    }

    /// Gamma inverse-link GLM, gated against frozen R `glm(family=Gamma("inverse"))`
    /// (`parity/goldens/sim_gamma_inv_glm.json`). Inverse is non-canonical (η=1/μ is
    /// −θ): the general branch + the 1/y cold-start seed. The oracle is sacred.
    #[test]
    fn fit_glm_gamma_inverse_matches_r() {
        const REF_BETA: [f64; 3] = [0.629151640871097, -0.198980738259224, -0.176508060896549];
        const REF_SE: [f64; 3] = [0.0432089466672347, 0.0187898149082593, 0.04188122263817];
        const REF_DISP: f64 = 1.0354907206002;
        let (x, y, n) = sim_gamma_xy();
        let p = 3;
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Inverse,
            },
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "gamma-inverse GLM must converge");
        let disp_rel = (f.dispersion - REF_DISP).abs() / REF_DISP;
        assert!(disp_rel < 5e-3, "φ = {} vs R {REF_DISP}", f.dispersion);
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
                "β[{j}] = {} vs R {}",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(se_rel < 3e-2, "se[{j}] = {} vs R {}", f.se[j], REF_SE[j]);
        }
    }

    /// `dispersion: Some(v)` holds φ=v fixed (skips the Pearson estimate) and
    /// scales SE by √v. Fitting at Some(1.0) vs Some(2.0) on identical data must
    /// give the same β and SE in the exact ratio √2, with `dispersion` reported
    /// as the held value.
    #[test]
    fn fit_glm_gamma_fixed_dispersion_scales_se() {
        let (x, y, n) = sim_gamma_xy();
        let p = 3;
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: None,
        };
        // φ directive lives in FitOptions, not the Family payload.
        let opts = |phi: f64| FitOptions {
            target_indices: vec![0, 1, 2],
            dispersion: Some(phi),
            ..FitOptions::default()
        };
        let f1 = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts(1.0));
        let f2 = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts(2.0));
        assert!(f1.converged && f2.converged);
        assert!((f2.dispersion - 2.0).abs() < 1e-12, "held φ must be 2.0");
        assert!((f1.dispersion - 1.0).abs() < 1e-12);
        for j in 0..p {
            assert!((f1.beta[j] - f2.beta[j]).abs() < 1e-12, "β φ-independent");
            // SE(φ=2) = √2 · SE(φ=1) exactly (same (XᵀWX)⁻¹, different √φ).
            assert!(
                (f2.se[j] - 2.0_f64.sqrt() * f1.se[j]).abs() < 1e-12,
                "se ratio at j={j}: {} vs {}",
                f2.se[j],
                2.0_f64.sqrt() * f1.se[j]
            );
        }
    }

    /// Negative-binomial GLM via the alternating outer-θ loop, gated against
    /// frozen R `MASS::glm.nb` (`parity/goldens/sim_nb_glm.json`):
    /// `y ~ 1 + x + grp` on sim_nb. `dispersion = θ̂` (the estimated shape); β SE
    /// conditions on θ̂. The oracle is sacred (parity §1).
    #[test]
    fn fit_glm_nb_matches_mass() {
        const REF_BETA: [f64; 3] = [0.144166077871857, 0.619826870647895, 0.633686899496841];
        const REF_SE: [f64; 3] = [0.120690561977139, 0.0756442004078213, 0.155714256322938];
        const REF_THETA: f64 = 1.01052181546876;
        // sim_nb.csv: cluster,x,grp,y (y integer counts).
        let csv = include_str!("../parity/data/sim_nb.csv");
        let p = 3;
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let xv: f64 = f[1].parse().unwrap();
            let grp_b = f64::from(u32::from(f[2] == "b"));
            let yv: f64 = f[3].parse().unwrap();
            x.extend_from_slice(&[1.0, xv, grp_b]);
            y.push(yv);
        }
        let n = y.len();
        let model = ModelSpec {
            family: Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            re: None,
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "NB GLM must converge");
        let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
        assert!(
            th_rel < 2e-2,
            "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
            f.dispersion
        );
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
                "β[{j}] = {} vs MASS {}",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(se_rel < 3e-2, "se[{j}] = {} vs MASS {}", f.se[j], REF_SE[j]);
        }
    }

    /// Map raw cluster labels to dense 0-based ids (first-seen order) + the count.
    fn dense_ids(raw: &[u32]) -> (Vec<u32>, usize) {
        use std::collections::HashMap;
        let mut map: HashMap<u32, u32> = HashMap::new();
        let mut next = 0u32;
        let ids: Vec<u32> = raw
            .iter()
            .map(|&r| {
                *map.entry(r).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                })
            })
            .collect();
        (ids, next as usize)
    }

    /// Map string factor labels to dense 0-based ids (first-seen order) + the count.
    fn dense_str(raw: &[String]) -> (Vec<u32>, usize) {
        use std::collections::HashMap;
        let mut map: HashMap<String, u32> = HashMap::new();
        let mut next = 0u32;
        let ids: Vec<u32> = raw
            .iter()
            .map(|r| {
                *map.entry(r.clone()).or_insert_with(|| {
                    let v = next;
                    next += 1;
                    v
                })
            })
            .collect();
        (ids, next as usize)
    }

    /// Gap #3: sleepstudy `Reaction ~ 1 + Days + (1 + Days | Subject)` — a q=2
    /// random-slope LMM through `fit_cold`, gated against the frozen lme4 VarCorr
    /// (`parity/goldens/sleepstudy_lmm.json`, REML). Checks the full 2×2 RE
    /// covariance (variances AND the off-diagonal covariance) via `Fit::varcorr`,
    /// which `tau2` cannot represent at q≥2. The oracle is sacred.
    #[test]
    fn fit_sleepstudy_slope_varcorr_matches_lme4() {
        const REF_B0: f64 = 251.405104848485;
        const REF_B1: f64 = 10.467285959596;
        const REF_SE0: f64 = 6.82459669495491;
        const REF_SE1: f64 = 1.54578964390598;
        const REF_SD0: f64 = 24.7406579949841; // (Intercept) sd
        const REF_SD1: f64 = 5.92213765889808; // Days sd
        const REF_CORR: f64 = 0.0655512382381282;

        let csv = include_str!("../parity/data/sleepstudy.csv");
        let mut y = Vec::<f64>::new();
        let mut days = Vec::<f64>::new();
        let mut subj_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap()); // Reaction
            days.push(f[1].parse().unwrap()); // Days
            subj_raw.push(f[2].to_string()); // Subject
        }
        let n = y.len();
        let p = 2;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0; // intercept
            x[i * p + 1] = days[i]; // Days
        }
        let (subject, _n_subj) = dense_str(&subj_raw);

        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
                slopes: vec![1],                                 // random slope on Days (col 1)
                extra_groupings: vec![],
            }),
        };
        let ids = GroupIds {
            primary: subject,
            extra: vec![],
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0, 1],
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "sleepstudy slope LMM must converge");
        assert!(
            (f.beta[0] - REF_B0).abs() / REF_B0 < 1e-3,
            "β0 {} vs {REF_B0}",
            f.beta[0]
        );
        assert!(
            (f.beta[1] - REF_B1).abs() / REF_B1 < 1e-3,
            "β1 {} vs {REF_B1}",
            f.beta[1]
        );
        assert!(
            (f.se[0] - REF_SE0).abs() / REF_SE0 < 2e-2,
            "se0 {} vs {REF_SE0}",
            f.se[0]
        );
        assert!(
            (f.se[1] - REF_SE1).abs() / REF_SE1 < 2e-2,
            "se1 {} vs {REF_SE1}",
            f.se[1]
        );

        // Reference D (col-major vech lower-tri): [D00, D10, D11].
        let d00 = REF_SD0 * REF_SD0;
        let d11 = REF_SD1 * REF_SD1;
        let d10 = REF_CORR * REF_SD0 * REF_SD1;
        assert_eq!(f.varcorr.len(), 1, "one grouping block");
        let vc = &f.varcorr[0];
        assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
        assert!(
            (vc[0].sqrt() - REF_SD0).abs() / REF_SD0 < 1e-2,
            "sd0 {} vs {REF_SD0}",
            vc[0].sqrt()
        );
        assert!(
            (vc[2].sqrt() - REF_SD1).abs() / REF_SD1 < 1e-2,
            "sd1 {} vs {REF_SD1}",
            vc[2].sqrt()
        );
        // Covariance is small (corr≈0.066) → check on an absolute scale.
        assert!((vc[0] - d00).abs() / d00 < 2e-2, "D00 {} vs {d00}", vc[0]);
        assert!((vc[2] - d11).abs() / d11 < 2e-2, "D11 {} vs {d11}", vc[2]);
        assert!(
            (vc[1] - d10).abs() < 0.20 * REF_SD0 * REF_SD1,
            "D10 {} vs {d10}",
            vc[1]
        );
    }

    // serde ignores unread JSON fields (e.g. `group`) by default; only the fields
    // the assertions consume are declared, to keep the dead_code lint clean.
    #[derive(serde::Deserialize)]
    struct VcBlock {
        stddev: Vec<f64>,
        corr: Vec<Vec<f64>>,
    }
    #[derive(serde::Deserialize)]
    struct VcEst {
        beta: Vec<f64>,
        varcomp: Vec<VcBlock>,
    }
    #[derive(serde::Deserialize)]
    struct VcGolden {
        estimates: VcEst,
    }

    /// Gap #3 synthetic: crossed random-slope `y ~ 1 + x + (1 + x | g1) + (1 | g2)`
    /// vs the R-generated lme4 golden (`parity/goldens/sim_slope_lmm.json`). Exercises
    /// a q=2 `varcorr` block on the PRIMARY plus a scalar block on a crossed EXTRA
    /// grouping — the multi-grouping generalization the single-grouping composition omits.
    /// The oracle is sacred.
    #[test]
    fn fit_sim_slope_varcorr_matches_lme4() {
        let raw = include_str!("../parity/goldens/sim_slope_lmm.json");
        let gold: VcGolden = serde_json::from_str(raw).expect("golden JSON parses");

        let csv = include_str!("../parity/data/sim_slope.csv");
        let mut y = Vec::<f64>::new();
        let mut xcol = Vec::<f64>::new();
        let mut g1_raw = Vec::<String>::new();
        let mut g2_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap()); // y
            xcol.push(f[1].parse().unwrap()); // x
            g1_raw.push(f[2].to_string());
            g2_raw.push(f[3].to_string());
        }
        let n = y.len();
        let p = 2;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            x[i * p + 1] = xcol[i];
        }
        let (g1, _n1) = dense_str(&g1_raw);
        let (g2, _n2) = dense_str(&g2_raw);

        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 },
                slopes: vec![1], // random slope on x for g1
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 },
                    slopes: vec![], // g2 intercept-only
                }],
            }),
        };
        let ids = GroupIds {
            primary: g1,
            extra: vec![g2],
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0, 1],
                ..FitOptions::default()
            },
        );

        assert!(f.converged);
        for j in 0..p {
            let r = gold.estimates.beta[j];
            assert!(
                (f.beta[j] - r).abs() / r.abs().max(1e-6) < 5e-3,
                "β{j} {} vs {r}",
                f.beta[j]
            );
        }
        // varcorr[0] = g1 (q=2), varcorr[1] = g2 (scalar). Reconstruct D from stddev+corr.
        assert_eq!(f.varcorr.len(), 2);
        let g1b = &gold.estimates.varcomp[0];
        let (sd0, sd1, c01) = (g1b.stddev[0], g1b.stddev[1], g1b.corr[0][1]);
        let vc0 = &f.varcorr[0];
        assert!(
            (vc0[0].sqrt() - sd0).abs() / sd0 < 2e-2,
            "g1 sd0 {} vs {sd0}",
            vc0[0].sqrt()
        );
        assert!(
            (vc0[2].sqrt() - sd1).abs() / sd1 < 2e-2,
            "g1 sd1 {} vs {sd1}",
            vc0[2].sqrt()
        );
        assert!(
            (vc0[1] - c01 * sd0 * sd1).abs() < 0.30 * sd0 * sd1,
            "g1 cov {}",
            vc0[1]
        );
        let g2sd = gold.estimates.varcomp[1].stddev[0];
        assert!(
            (f.varcorr[1][0].sqrt() - g2sd).abs() / g2sd < 3e-2,
            "g2 sd {} vs {g2sd}",
            f.varcorr[1][0].sqrt()
        );
    }

    /// Gap #1 crossed: Penicillin `diameter ~ 1 + (1|plate) + (1|sample)` through the
    /// data-shaped `fit_cold` with `GroupIds { primary: plate, extra: vec![sample] }`,
    /// gated against the frozen lme4 golden (`parity/goldens/penicillin_lmm.json`,
    /// REML). Two crossed intercept-only groupings, fixed effect = intercept only
    /// (p=1). Placeholder spec counts prove the data path derives level counts from
    /// the ids. The oracle is sacred.
    #[test]
    fn fit_penicillin_crossed_matches_lme4() {
        const REF_BETA: f64 = 22.9722222222;
        const REF_SE: f64 = 0.808595361386;
        const REF_PLATE_SD: f64 = 0.846702;
        const REF_SAMPLE_SD: f64 = 1.931614;

        let csv = include_str!("../parity/data/Penicillin.csv");
        let mut y = Vec::<f64>::new();
        let mut plate_raw = Vec::<String>::new();
        let mut sample_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap()); // diameter
            plate_raw.push(f[1].to_string());
            sample_raw.push(f[2].to_string());
        }
        let n = y.len();
        let p = 1;
        let x = vec![1.0f64; n]; // intercept-only design
        let (plate, _n_plate) = dense_str(&plate_raw);
        let (sample, _n_sample) = dense_str(&sample_raw);

        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 1 }, // placeholder
                    slopes: vec![],
                }],
            }),
        };
        let ids = GroupIds {
            primary: plate,
            extra: vec![sample],
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0],
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "Penicillin crossed LMM must converge");
        assert!(
            (f.beta[0] - REF_BETA).abs() / REF_BETA < 1e-4,
            "β0 = {} vs lme4 {REF_BETA}",
            f.beta[0]
        );
        let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
        assert!(
            se_rel < 2e-2,
            "se0 = {} vs lme4 {REF_SE} (rel {se_rel})",
            f.se[0]
        );
        // theta layout: [primary (plate) vech | sample scalar]; tau2[k] = θ̂[k]²·σ̂².
        let plate_sd = f.tau2[0].sqrt();
        let sample_sd = f.tau2[1].sqrt();
        assert!(
            (plate_sd - REF_PLATE_SD).abs() / REF_PLATE_SD < 5e-3,
            "plate sd = {plate_sd} vs lme4 {REF_PLATE_SD}"
        );
        assert!(
            (sample_sd - REF_SAMPLE_SD).abs() / REF_SAMPLE_SD < 5e-3,
            "sample sd = {sample_sd} vs lme4 {REF_SAMPLE_SD}"
        );
    }

    /// Gap #1 nested: Pastes `strength ~ 1 + (1|batch/cask)` through the data-shaped
    /// `fit_cold` with `GroupIds { primary: batch, extra: vec![cask] }`, where `cask`
    /// is the globally-unique batch:cask level (dense 0..29). Gated against the frozen
    /// lme4 golden (`parity/goldens/pastes_lmm.json`, REML). Exercises the
    /// `NestedWithin` topology tag on the data path; placeholder counts prove level
    /// counts come from the ids. The oracle is sacred.
    #[test]
    fn fit_pastes_nested_matches_lme4() {
        const REF_BETA: f64 = 60.0533333333;
        const REF_SE: f64 = 0.676870215074;
        const REF_BATCH_SD: f64 = 1.287366;
        const REF_CASK_SD: f64 = 2.904077;

        let csv = include_str!("../parity/data/Pastes.csv");
        // cols: strength,batch,cask,sample  (sample = "batch:cask" global label)
        let mut y = Vec::<f64>::new();
        let mut batch_raw = Vec::<String>::new();
        let mut cask_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap()); // strength
            batch_raw.push(f[1].to_string()); // batch
            cask_raw.push(f[3].to_string()); // sample = batch:cask global label
        }
        let n = y.len();
        let p = 1;
        let x = vec![1.0f64; n];
        let (batch, _n_batch) = dense_str(&batch_raw);
        let (cask, _n_cask) = dense_str(&cask_raw);

        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::NestedWithin { n_per_parent: 1 }, // placeholder
                    slopes: vec![],
                }],
            }),
        };
        let ids = GroupIds {
            primary: batch,
            extra: vec![cask],
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0],
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "Pastes nested LMM must converge");
        assert!(
            (f.beta[0] - REF_BETA).abs() / REF_BETA < 1e-4,
            "β0 = {} vs lme4 {REF_BETA}",
            f.beta[0]
        );
        let se_rel = (f.se[0] - REF_SE).abs() / REF_SE;
        assert!(
            se_rel < 2e-2,
            "se0 = {} vs lme4 {REF_SE} (rel {se_rel})",
            f.se[0]
        );
        // theta layout: [primary (batch) vech | nested (cask) scalar].
        let batch_sd = f.tau2[0].sqrt();
        let cask_sd = f.tau2[1].sqrt();
        assert!(
            (batch_sd - REF_BATCH_SD).abs() / REF_BATCH_SD < 1e-2,
            "batch sd = {batch_sd} vs lme4 {REF_BATCH_SD}"
        );
        assert!(
            (cask_sd - REF_CASK_SD).abs() / REF_CASK_SD < 5e-3,
            "cask sd = {cask_sd} vs lme4 {REF_CASK_SD}"
        );
    }

    /// Parse a `cluster,x,grp,y` sim CSV → (X=[1,x,grp_b], y, dense cluster ids, n_clusters).
    fn sim_clustered(csv: &str) -> (Vec<f64>, Vec<f64>, Vec<u32>, usize) {
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut raw = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            raw.push(f[0].parse().unwrap());
            x.extend_from_slice(&[
                1.0,
                f[1].parse().unwrap(),
                f64::from(u32::from(f[2] == "b")),
            ]);
            y.push(f[3].parse().unwrap());
        }
        let (ids, nc) = dense_ids(&raw);
        (x, y, ids, nc)
    }

    /// Poisson GLMM `TICKS ~ 1 + YEAR + cHEIGHT + (1|INDEX)` on grouseticks
    /// (observation-level INDEX = 403 size-1 clusters), gated against frozen
    /// `lme4::glmer(family=poisson, nAGQ=1)` (`parity/goldens/grouseticks_agq_k1.json`).
    /// Exercises the blocked PIRLS path for a non-binomial family. lme4-only SE
    /// (parity §6). The oracle is sacred.
    #[test]
    fn fit_glmm_poisson_grouseticks_matches_lme4() {
        const REF_BETA: [f64; 4] = [
            0.43997315657,
            1.10082823356,
            -0.988047711093,
            -0.0236982108735,
        ];
        const REF_SE: [f64; 4] = [
            0.140882438904,
            0.168795499457,
            0.197654140578,
            0.00211151961592,
        ];
        const REF_INDEX_SD: f64 = 1.129369439;
        let csv = include_str!("../parity/data/grouseticks.csv");
        let p = 4;
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut raw = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            raw.push(f[0].parse().unwrap()); // INDEX
            let year: u32 = f[4].parse().unwrap();
            x.extend_from_slice(&[
                1.0,
                f64::from(u32::from(year == 96)),
                f64::from(u32::from(year == 97)),
                f[6].parse().unwrap(), // cHEIGHT
            ]);
            y.push(f[1].parse().unwrap()); // TICKS
        }
        let (cluster_ids, n_clusters) = dense_ids(&raw);
        let n = y.len();
        let model = ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "poisson GLMM must converge");
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 1e-3,
                "β[{j}] = {} vs lme4 {}",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(se_rel < 3e-2, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
        }
        let sd_rel = (f.tau2[0].sqrt() - REF_INDEX_SD).abs() / REF_INDEX_SD;
        assert!(
            sd_rel < 3e-3,
            "INDEX sd = {} vs lme4 {REF_INDEX_SD}",
            f.tau2[0].sqrt()
        );
    }

    /// Parses `parity/data/grouseticks.csv` into the 3-crossed `TICKS ~ YEAR +
    /// cHEIGHT + (1|INDEX) + (1|BROOD) + (1|LOCATION)` design (observation-level
    /// INDEX + crossed BROOD, LOCATION). Shared by the lme4 fit gate below and the
    /// both-paths sparse-vs-dense Schur cross-checks (`sparse_schur_*`), which need
    /// direct `GlmmWorkspace`/`StructuredSchur` access that `fit_cold` doesn't expose.
    fn grouseticks_3crossed_inputs() -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, GroupIds) {
        let csv = include_str!("../parity/data/grouseticks.csv");
        // cols: INDEX,TICKS,BROOD,HEIGHT,YEAR,LOCATION,cHEIGHT
        let p = 4;
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut index_raw = Vec::<u32>::new();
        let mut brood_raw = Vec::<String>::new();
        let mut loc_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            index_raw.push(f[0].parse().unwrap());
            let year: u32 = f[4].parse().unwrap();
            x.extend_from_slice(&[
                1.0,
                f64::from(u32::from(year == 96)),
                f64::from(u32::from(year == 97)),
                f[6].parse().unwrap(), // cHEIGHT
            ]);
            y.push(f[1].parse().unwrap()); // TICKS
            brood_raw.push(f[2].to_string());
            loc_raw.push(f[5].to_string());
        }
        let n = y.len();
        let (index_ids, n_index) = dense_ids(&index_raw);
        let (brood_ids, _n_brood) = dense_str(&brood_raw);
        let (loc_ids, _n_loc) = dense_str(&loc_raw);

        let model = ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_index as u32,
                },
                slopes: vec![],
                extra_groupings: vec![
                    Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    },
                    Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 1 },
                        slopes: vec![],
                    },
                ],
            }),
        };
        let ids = GroupIds {
            primary: index_ids,
            extra: vec![brood_ids, loc_ids],
        };
        (x, y, n, p, model, ids)
    }

    /// Poisson GLMM, **three crossed groupings**: grouseticks
    /// `TICKS ~ YEAR + cHEIGHT + (1|INDEX) + (1|BROOD) + (1|LOCATION)` (observation-
    /// level INDEX + crossed BROOD, LOCATION), gated against the frozen
    /// `lme4::glmer(family=poisson)` reference (`parity/results/lme4/grouseticks.json`).
    /// Exercises the structured crossed-extras PIRLS/Schur path (`pirls_solve_blocked_
    /// extras` / `structured_factor`) that the single-grouping test above does not.
    /// This is the regression guard for the degenerate-fit bug: from a β=0 cold start
    /// the first PIRLS step overshot into a ~1e30 weight regime, the crossed Schur
    /// went non-PD, and the fit returned start values reported as converged. The GLM
    /// warm-start of β (`glm_warm_start_beta`) opens PIRLS near the mean and removes
    /// the overshoot; the converged-deviance guard (`glmm/mod.rs`) is the backstop.
    /// The oracle is sacred.
    #[test]
    fn fit_glmm_poisson_grouseticks_3crossed_matches_lme4() {
        // Frozen lme4 reference (parity/results/lme4/grouseticks.json).
        const REF_BETA: [f64; 4] = [
            0.372776372908808,
            1.18041688638813,
            -0.978684717829623,
            -0.0237606272596611,
        ];
        const REF_INDEX_SD: f64 = 0.541508524819898;
        const REF_BROOD_SD: f64 = 0.750027963921318;
        const REF_LOCATION_SD: f64 = 0.52872140071578;
        let (x, y, n, p, model, ids) = grouseticks_3crossed_inputs();
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );

        assert!(
            f.converged,
            "3-crossed poisson GLMM must converge (not the degenerate start fit)"
        );
        // β[0..2] are O(1); rel-compare. β[3] (cHEIGHT) is ~-0.024; abs-compare.
        for (j, &rb) in REF_BETA.iter().enumerate().take(3) {
            assert!(
                (f.beta[j] - rb).abs() / rb.abs() < 3e-2,
                "β[{j}] = {} vs lme4 {rb}",
                f.beta[j]
            );
        }
        assert!(
            (f.beta[3] - REF_BETA[3]).abs() < 3e-3,
            "β[3] = {} vs lme4 {}",
            f.beta[3],
            REF_BETA[3]
        );
        // tau2 layout [primary(INDEX) | BROOD | LOCATION].
        for (k, refsd) in [REF_INDEX_SD, REF_BROOD_SD, REF_LOCATION_SD]
            .into_iter()
            .enumerate()
        {
            let sd = f.tau2[k].sqrt();
            assert!(
                (sd - refsd).abs() / refsd < 5e-2,
                "grouping {k} sd = {sd} vs lme4 {refsd}"
            );
        }
    }

    /// Both-paths cross-check: the sparse-S Laplace deviance equals
    /// the dense-Schur deviance at the same θ on the grouseticks 3-crossed design. If
    /// they disagree, exactly one factor path is wrong (the +0.5·logdet_llt convention
    /// is the prime suspect). Not bitwise-equal (AMD reorders the sparse elimination),
    /// so a tight numeric gate, orders below the ~1.5e-4 lme4 β gap we must preserve.
    #[test]
    fn sparse_schur_deviance_equals_dense_grouseticks() {
        let (x, y, n, p, model, ids) = grouseticks_3crossed_inputs();
        let model = spec_sized_from_ids_pub(&model, &ids);
        let slope_cols: Vec<usize> = vec![];
        let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, 1);
        // Column-major x + build_z + StructuredSchur, as fit_glmm does.
        let mut xm = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xm[(i, j)] = x[i * p + j];
            }
        }
        build_z(
            &mut ws,
            xm.as_ref().subrows(0, n),
            &ids.primary,
            &ids.extra,
            n,
        );
        ws.structured_schur = StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n);
        // A representative interior θ (the blind start θ₀ for the 3 groupings) + a β
        // (the GLM warm start, matching what `fit_glmm` would open PIRLS at).
        let params: Vec<f64> = {
            let mut prm = ws.params.clone();
            let beta =
                glm_warm_start_beta(model.family, f64::NAN, xm.as_ref().subrows(0, n), &y, n, p);
            prm[ws.n_theta..ws.n_theta + p].copy_from_slice(&beta);
            prm
        };

        ws.force_dense_schur = true;
        let dev_dense = glmm_laplace_deviance(
            &params,
            &mut ws,
            xm.as_ref().subrows(0, n),
            &y,
            &ids.primary,
            n,
        );
        ws.force_dense_schur = false;
        let dev_sparse = glmm_laplace_deviance(
            &params,
            &mut ws,
            xm.as_ref().subrows(0, n),
            &y,
            &ids.primary,
            n,
        );

        assert!(
            dev_dense.is_finite() && dev_sparse.is_finite(),
            "both deviances finite"
        );
        let rel = (dev_dense - dev_sparse).abs() / (1.0 + dev_dense.abs());
        assert!(
            rel < 1e-9,
            "dense {dev_dense} vs sparse {dev_sparse} (rel {rel})"
        );
    }

    /// SE cross-check: the structured_schur_fill SE (sparse solve) equals the dense-Schur
    /// SE at the converged fit (se.rs routes through structured_ainv_solve).
    /// Unlike `sparse_schur_deviance_equals_dense_grouseticks` (one eval at a fixed θ,
    /// gated at 1e-9), this runs the FULL BOBYQA optimization twice — dense and sparse
    /// factor paths disagree by ~1e-9 per eval (AMD reorders the sparse elimination), so
    /// each run's θ̂ drifts by a path-dependent amount within BOBYQA's `rho_end` trust
    /// region before the Wald SE nonlinearly amplifies it. Gated at 1e-4: orders above
    /// the observed ~6.6e-7 noise floor, still tight enough to catch a real convention
    /// bug (a flipped 0.5×/1.0× logdet would show as a gap orders of magnitude larger).
    #[test]
    fn sparse_schur_se_equals_dense_grouseticks() {
        let (x, y, n, p, model, ids) = grouseticks_3crossed_inputs();
        let model = spec_sized_from_ids_pub(&model, &ids);
        let slope_cols: Vec<usize> = vec![];
        let mut xm = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xm[(i, j)] = x[i * p + j];
            }
        }
        let beta_start =
            glm_warm_start_beta(model.family, f64::NAN, xm.as_ref().subrows(0, n), &y, n, p);

        let run = |force_dense: bool| -> (Vec<f64>, bool) {
            let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, 1);
            build_z(
                &mut ws,
                xm.as_ref().subrows(0, n),
                &ids.primary,
                &ids.extra,
                n,
            );
            ws.structured_schur = if ws.groupings.structured_extras_eligible() {
                StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
            } else {
                None
            };
            ws.force_dense_schur = force_dense;
            let fit = crate::glmm::fit_glmm(
                &mut ws,
                xm.as_ref().subrows(0, n),
                &y,
                &ids.primary,
                &[0, 1, 2, 3],
                None,
                &beta_start,
                n,
                WaldSe::Rx,
            );
            (ws.var_diag[..p].to_vec(), fit.converged)
        };

        let (var_dense, conv_dense) = run(true);
        let (var_sparse, conv_sparse) = run(false);
        assert!(
            conv_dense && conv_sparse,
            "both dense and sparse fits must converge"
        );
        for (j, (&vd, &vs)) in var_dense.iter().zip(&var_sparse).enumerate() {
            assert!(
                vd.is_finite() && vs.is_finite(),
                "var_diag[{j}] finite (dense {vd}, sparse {vs})"
            );
            let rel = (vd - vs).abs() / (1.0 + vd.abs());
            assert!(
                rel < 1e-4,
                "var_diag[{j}] dense {vd} vs sparse {vs} (rel {rel})"
            );
        }
    }

    /// Small-`e` guard (no regression on small-`e` GLMMs): a synthetic
    /// crossed binomial GLMM `y ~ x + (1|g1) + (1|g2)`, primary g1 = 4 levels,
    /// extra crossed g2 = 6 levels ⇒ e = 6 — orders below grouseticks' e = 181,
    /// the scale the other `sparse_schur_*_equals_dense_grouseticks` cross-checks
    /// exercise. Runs the full BOBYQA fit twice (dense-forced vs sparse,
    /// mirroring `sparse_schur_se_equals_dense_grouseticks`'s pattern) and
    /// compares both β and the Wald SE. Gated at 1e-7 (tighter than that e=181
    /// test's 1e-4): a 6-wide Schur gives AMD far less elimination-order
    /// freedom, so the dense/sparse per-eval float noise that drives BOBYQA
    /// path-dependent drift is negligible at this scale.
    #[test]
    fn sparse_schur_small_e_matches_dense() {
        // 4-level primary × 6-level crossed extra, 2 obs/cell ⇒ e = 6, n = 48.
        let (n_prim, n_extra, reps) = (4usize, 6usize, 2usize);
        let n = n_prim * n_extra * reps;
        let p = 2;
        let prim_eff = [0.4, -0.3, 0.5, -0.2];
        let extra_eff = [0.3, -0.4, 0.2, -0.1, 0.35, -0.25];
        let mut xm = Mat::<f64>::zeros(n, p);
        let mut y = vec![0.0f64; n];
        let mut cl = vec![0u32; n];
        let mut cr = vec![0u32; n];
        let mut st = 42u64;
        let mut i = 0;
        for (pi, &pe) in prim_eff.iter().enumerate() {
            for (ei, &ee) in extra_eff.iter().enumerate() {
                for _ in 0..reps {
                    let cov = crate::sparse::test_lcg(&mut st);
                    let eta = 0.2 + 0.6 * cov + pe + ee;
                    let prob = 1.0 / (1.0 + (-eta).exp());
                    let draw = (crate::sparse::test_lcg(&mut st) + 1.0) / 2.0;
                    xm[(i, 0)] = 1.0;
                    xm[(i, 1)] = cov;
                    cl[i] = pi as u32;
                    cr[i] = ei as u32;
                    y[i] = if draw < prob { 1.0 } else { 0.0 };
                    i += 1;
                }
            }
        }
        let ids = GroupIds {
            primary: cl,
            extra: vec![cr],
        };
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_prim as u32,
                },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed {
                        n_clusters: n_extra as u32,
                    },
                    slopes: vec![],
                }],
            }),
        };
        let model = spec_sized_from_ids_pub(&model, &ids);
        let slope_cols: Vec<usize> = vec![];
        let beta_start =
            glm_warm_start_beta(model.family, f64::NAN, xm.as_ref().subrows(0, n), &y, n, p);

        let run = |force_dense: bool| -> (Vec<f64>, Vec<f64>, bool) {
            let mut ws = GlmmWorkspace::for_cluster_spec(p, &model, n, &slope_cols, 1);
            build_z(
                &mut ws,
                xm.as_ref().subrows(0, n),
                &ids.primary,
                &ids.extra,
                n,
            );
            ws.structured_schur = if ws.groupings.structured_extras_eligible() {
                StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
            } else {
                None
            };
            ws.force_dense_schur = force_dense;
            let fit = crate::glmm::fit_glmm(
                &mut ws,
                xm.as_ref().subrows(0, n),
                &y,
                &ids.primary,
                &[0, 1],
                None,
                &beta_start,
                n,
                WaldSe::Rx,
            );
            (ws.betas.clone(), ws.var_diag[..p].to_vec(), fit.converged)
        };

        let (beta_dense, var_dense, conv_dense) = run(true);
        let (beta_sparse, var_sparse, conv_sparse) = run(false);
        assert!(
            conv_dense && conv_sparse,
            "both dense and sparse fits must converge"
        );
        for j in 0..p {
            let rel_b = (beta_dense[j] - beta_sparse[j]).abs() / (1.0 + beta_dense[j].abs());
            assert!(
                rel_b < 1e-7,
                "β[{j}] dense {} vs sparse {} (rel {rel_b})",
                beta_dense[j],
                beta_sparse[j]
            );
            let vd = var_dense[j];
            let vs = var_sparse[j];
            assert!(
                vd.is_finite() && vs.is_finite(),
                "var_diag[{j}] finite (dense {vd}, sparse {vs})"
            );
            let rel_v = (vd - vs).abs() / (1.0 + vd.abs());
            assert!(
                rel_v < 1e-7,
                "var_diag[{j}] dense {vd} vs sparse {vs} (rel {rel_v})"
            );
        }
    }

    /// Adaptive GH quadrature, binomial GLMM: cbpp `cbind(incidence, size−incidence)
    /// ~ period + (1|herd)` (expanded 0/1) at nAGQ ∈ {1,7,11}, gated against frozen
    /// `glmer(nAGQ=k)` (`parity/goldens/cbpp_agq_k{1,7,11}.json`). nAGQ=1 is Laplace
    /// (≡ `fit_glmm_cbpp_matches_lme4`); k>1 shifts β/varcomp off it as the Laplace
    /// bias is integrated out (herd sd 0.642→0.648). β + varcomp only — the AGQ
    /// goldens don't freeze SE (AGQ changes the integral, not the SE convention).
    /// The oracle is sacred.
    #[test]
    fn fit_glmm_binomial_agq_matches_lme4() {
        // (nAGQ, β, herd sd) per frozen glmer(nAGQ=k).
        let refs: [(u8, [f64; 4], f64); 3] = [
            (
                1,
                [
                    -1.3983428644712,
                    -0.991924974975699,
                    -1.12821621594328,
                    -1.57974541364914,
                ],
                0.642069927729443,
            ),
            (
                7,
                [
                    -1.39923514006289,
                    -0.991393555379478,
                    -1.12782137776524,
                    -1.57947295789128,
                ],
                0.647518692435348,
            ),
            (
                11,
                [
                    -1.39921944386306,
                    -0.991408657432828,
                    -1.12781283713842,
                    -1.57948777358155,
                ],
                0.647517861083539,
            ),
        ];
        let csv = include_str!("../parity/data/cbpp.csv");
        let p = 4;
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut cluster_ids = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let herd: u32 = f[0].parse::<u32>().unwrap() - 1;
            let incidence: u32 = f[1].parse().unwrap();
            let size: u32 = f[2].parse().unwrap();
            let period: u32 = f[3].parse().unwrap();
            let row = [
                1.0,
                f64::from(u32::from(period == 2)),
                f64::from(u32::from(period == 3)),
                f64::from(u32::from(period == 4)),
            ];
            for k in 0..size {
                x.extend_from_slice(&row);
                y.push(if k < incidence { 1.0 } else { 0.0 });
                cluster_ids.push(herd);
            }
        }
        let n = y.len();
        for (nagq, refb, refsd) in refs {
            let model = ModelSpec {
                family: Family::Binomial {
                    link: BinomialLink::Logit,
                },
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters { n_clusters: 15 },
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            };
            let f = fit_cold(
                &x,
                &y,
                n,
                p,
                &model,
                &GroupIds {
                    primary: cluster_ids.clone(),
                    extra: vec![],
                },
                &FitOptions {
                    target_indices: vec![0, 1, 2, 3],
                    nagq,
                    ..FitOptions::default()
                },
            );
            assert!(f.converged, "binomial AGQ k={nagq} must converge");
            for (j, (&b, &rb)) in f.beta.iter().zip(&refb).enumerate() {
                assert!(
                    (b - rb).abs() / rb.abs() < 1e-3,
                    "k={nagq} β[{j}] = {b} vs lme4 {rb}"
                );
            }
            let sd_rel = (f.tau2[0].sqrt() - refsd).abs() / refsd;
            assert!(
                sd_rel < 1e-3,
                "k={nagq} herd sd = {} vs lme4 {refsd}",
                f.tau2[0].sqrt()
            );
        }
    }

    /// Adaptive GH quadrature, Poisson GLMM: grouseticks single-grouping `TICKS ~
    /// YEAR + cHEIGHT + (1|INDEX)` at nAGQ ∈ {1,7,11}, gated against frozen
    /// `glmer(family=poisson, nAGQ=k)` (`parity/goldens/grouseticks_agq_k{1,7,11}.json`).
    /// nAGQ=1 ≡ `fit_glmm_poisson_grouseticks_matches_lme4`; k>1 shifts the fit as the
    /// Laplace bias is integrated out. β + varcomp only. The oracle is sacred.
    #[test]
    fn fit_glmm_poisson_agq_matches_lme4() {
        let refs: [(u8, [f64; 4], f64); 3] = [
            (
                1,
                [
                    0.439973156570138,
                    1.10082823355748,
                    -0.988047711092655,
                    -0.0236982108735122,
                ],
                1.1293694390126,
            ),
            (
                7,
                [
                    0.443726696423487,
                    1.09738146557843,
                    -0.988798870848502,
                    -0.0236841397694784,
                ],
                1.13482415039616,
            ),
            (
                11,
                [
                    0.444137982539483,
                    1.09717523260645,
                    -0.9889317811938,
                    -0.0236832339939658,
                ],
                1.13407867482264,
            ),
        ];
        let csv = include_str!("../parity/data/grouseticks.csv");
        let p = 4;
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut raw = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            raw.push(f[0].parse().unwrap());
            let year: u32 = f[4].parse().unwrap();
            x.extend_from_slice(&[
                1.0,
                f64::from(u32::from(year == 96)),
                f64::from(u32::from(year == 97)),
                f[6].parse().unwrap(),
            ]);
            y.push(f[1].parse().unwrap());
        }
        let (cluster_ids, n_clusters) = dense_ids(&raw);
        let n = y.len();
        for (nagq, refb, refsd) in refs {
            let model = ModelSpec {
                family: Family::Poisson {
                    link: crate::PoissonLink::Log,
                },
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters {
                        n_clusters: n_clusters as u32,
                    },
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            };
            let f = fit_cold(
                &x,
                &y,
                n,
                p,
                &model,
                &GroupIds {
                    primary: cluster_ids.clone(),
                    extra: vec![],
                },
                &FitOptions {
                    target_indices: vec![0, 1, 2, 3],
                    nagq,
                    ..FitOptions::default()
                },
            );
            assert!(f.converged, "poisson AGQ k={nagq} must converge");
            for (j, (&b, &rb)) in f.beta.iter().zip(&refb).enumerate() {
                assert!(
                    (b - rb).abs() / rb.abs() < 1e-3,
                    "k={nagq} β[{j}] = {b} vs lme4 {rb}"
                );
            }
            let sd_rel = (f.tau2[0].sqrt() - refsd).abs() / refsd;
            assert!(
                sd_rel < 1e-3,
                "k={nagq} INDEX sd = {} vs lme4 {refsd}",
                f.tau2[0].sqrt()
            );
        }
    }

    /// Probit binomial GLMM `cbind(incidence, size−incidence) ~ period + (1|herd)`
    /// on cbpp (expanded 0/1), gated against frozen `glmer(binomial("probit"))`
    /// (`parity/goldens/cbpp_probit_glmm.json`). lme4-only SE. The oracle is sacred.
    // FD-Hessian SE (use.hessian=TRUE) for this non-canonical link needs a
    // smooth deviance: probit is Fisher-scoring (linear convergence), so PIRLS at
    // the canonical 1e-6 tolerance left the deviance noisy to ~1e-4 and the FD
    // second differences amplified it into a 7–41%-wrong SE. `pirls_tol` gives
    // non-canonical links the tight `PIRLS_TOL_REL_NONCANON` (1e-8); β and
    // se_hessian now match lme4 to ~1e-4. (The Φ accuracy — `phi_hp`, Cody erfc —
    // is a separate genuine fix but was NOT the SE cause; verified by spike.)
    #[test]
    fn fit_glmm_probit_cbpp_matches_lme4() {
        const REF_BETA: [f64; 4] = [
            -0.835474929637,
            -0.528032739718,
            -0.616854298164,
            -0.799572598137,
        ];
        const REF_SE: [f64; 4] = [
            0.126232795983,
            0.160588369843,
            0.169457682932,
            0.204681153481,
        ];
        const REF_HERD_SD: f64 = 0.3379893465;
        let csv = include_str!("../parity/data/cbpp.csv");
        let p = 4;
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut cluster_ids = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let herd: u32 = f[0].parse::<u32>().unwrap() - 1;
            let incidence: u32 = f[1].parse().unwrap();
            let size: u32 = f[2].parse().unwrap();
            let period: u32 = f[3].parse().unwrap();
            let row = [
                1.0,
                f64::from(u32::from(period == 2)),
                f64::from(u32::from(period == 3)),
                f64::from(u32::from(period == 4)),
            ];
            for k in 0..size {
                x.extend_from_slice(&row);
                y.push(if k < incidence { 1.0 } else { 0.0 });
                cluster_ids.push(herd);
            }
        }
        let n = y.len();
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Probit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 15 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "probit GLMM must converge");
        let sd_rel = (f.tau2[0].sqrt() - REF_HERD_SD).abs() / REF_HERD_SD;
        assert!(
            sd_rel < 3e-3,
            "herd sd = {} vs lme4 {REF_HERD_SD}",
            f.tau2[0].sqrt()
        );
        for ((&b, &rb), (&s, &rs)) in f.beta.iter().zip(&REF_BETA).zip(f.se.iter().zip(&REF_SE)) {
            assert!((b - rb).abs() / rb.abs() < 2e-3, "β = {b} vs lme4 {rb}");
            assert!((s - rs).abs() / rs < 3e-2, "se = {s} vs lme4 {rs}");
        }
    }

    /// Gamma log-link GLMM `y ~ 1 + x + grp + (1|cluster)` on sim_gamma, gated
    /// against frozen `glmer(family=Gamma("log"))` (`parity/goldens/sim_gamma_glmm.json`).
    /// φ̂ is the post-fit Pearson moment on conditional-mode residuals (matches the
    /// oracle's hand-computed `Σpearson²/(n−p)`). lme4-only SE. The oracle is sacred.
    //
    // The dispersion enters glmer's Gamma fit ONLY through the family `aic` term in
    // the Laplace objective (profiled `disp=D/n`), not via 1/φ-weighted PIRLS or a
    // φ-ridge (confirmed against lme4 src/glmFamily.cpp; MixedModels.jl decouples
    // entirely and PQL/glmmPQL uses a φ-ridge — both are *different* estimators).
    // The kernel swaps `D → gamma_aic` in `laplace_deviance`, so β̂/τ̂ and the
    // FD-Hessian SE pick up the coupling. See `family::gamma_aic`.
    #[test]
    fn fit_glmm_gamma_sim_matches_lme4() {
        const REF_BETA: [f64; 3] = [0.308930805779, 0.577841416651, 0.455706877075];
        const REF_SE: [f64; 3] = [0.139098615851, 0.0427935407665, 0.0883045165218];
        // Golden's `se_rx` = lme4 `vcov(use.hessian=FALSE)`, σ̂²-scaled for Gamma —
        // gates the kernel's `WaldSe::Rx` σ̂² factor (`family::glmm_sigma_sq`).
        const REF_SE_RX: [f64; 3] = [0.116924273630386, 0.0453773644154408, 0.0929163554683392];
        const REF_CLUSTER_SD: f64 = 0.4851167757;
        const REF_DISP: f64 = 0.5265553674;
        let (x, y, cluster_ids, n_clusters) =
            sim_clustered(include_str!("../parity/data/sim_gamma.csv"));
        let (n, p) = (y.len(), 3);
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "gamma GLMM must converge");
        let disp_rel = (f.dispersion - REF_DISP).abs() / REF_DISP;
        assert!(disp_rel < 2e-2, "φ̂ = {} vs lme4 {REF_DISP}", f.dispersion);
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 2e-3,
                "β[{j}] = {} vs lme4 {}",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(se_rel < 3e-2, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
        }
        let sd_rel = (f.tau2[0].sqrt() - REF_CLUSTER_SD).abs() / REF_CLUSTER_SD;
        assert!(
            sd_rel < 5e-3,
            "cluster sd = {} vs lme4 {REF_CLUSTER_SD}",
            f.tau2[0].sqrt()
        );

        // Rx arm on the same design vs the golden's σ̂²-scaled `se_rx`.
        let f_rx = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids,
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2],
                wald_se: WaldSe::Rx,
                ..FitOptions::default()
            },
        );
        assert!(f_rx.converged, "gamma GLMM (Rx) must converge");
        for j in 0..p {
            let se_rel = (f_rx.se[j] - REF_SE_RX[j]).abs() / REF_SE_RX[j];
            assert!(
                se_rel < 3e-2,
                "rx se[{j}] = {} vs lme4 {}",
                f_rx.se[j],
                REF_SE_RX[j]
            );
        }
    }

    /// NB GLMM `y ~ 1 + x + grp + (1|cluster)` on sim_nb via the outer-θ loop,
    /// gated against frozen `lme4::glmer.nb` (`parity/goldens/sim_nb_glmm.json`).
    /// `dispersion = θ̂`. lme4-only SE. The oracle is sacred.
    #[test]
    fn fit_glmm_nb_sim_matches_lme4() {
        const REF_BETA: [f64; 3] = [-0.0207782143496, 0.593950952004, 0.59944069353];
        const REF_SE: [f64; 3] = [0.163165315799, 0.0721272221837, 0.141480120735];
        const REF_CLUSTER_SD: f64 = 0.5742029807;
        const REF_THETA: f64 = 1.783620004;
        let (x, y, cluster_ids, n_clusters) =
            sim_clustered(include_str!("../parity/data/sim_nb.csv"));
        let (n, p) = (y.len(), 3);
        let model = ModelSpec {
            family: Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "NB GLMM must converge");
        let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
        assert!(th_rel < 5e-2, "θ̂ = {} vs lme4 {REF_THETA}", f.dispersion);
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs() < 5e-3
                    || (f.beta[j] - REF_BETA[j]).abs() < 5e-3,
                "β[{j}] = {} vs lme4 {}",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(se_rel < 5e-2, "se[{j}] = {} vs lme4 {}", f.se[j], REF_SE[j]);
        }
        let sd_rel = (f.tau2[0].sqrt() - REF_CLUSTER_SD).abs() / REF_CLUSTER_SD;
        assert!(
            sd_rel < 2e-2,
            "cluster sd = {} vs lme4 {REF_CLUSTER_SD}",
            f.tau2[0].sqrt()
        );
    }

    /// In-envelope designs route NoZ (byte-identical fast path); over-envelope
    /// route Sparse. The boundary is the cap edge.
    #[test]
    fn classify_routes_at_the_cap_edge() {
        // A scalar-intercept LMM (q_p=1, no extras) — deep in-envelope.
        let in_env = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 10 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        assert!(matches!(
            super::classify_design_pub(&in_env, 1),
            super::Solver::NoZ
        ));

        // MAX_EXTRA_GROUPINGS+1 crossed groupings — over-envelope ⇒ Sparse.
        let over = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 10 },
                slopes: vec![],
                extra_groupings: (0..(crate::consts::MAX_EXTRA_GROUPINGS + 1))
                    .map(|_| Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 4 },
                        slopes: vec![],
                    })
                    .collect(),
            }),
        };
        assert!(matches!(
            super::classify_design_pub(&over, 1),
            super::Solver::Sparse
        ));

        // Wide primary slope block past MAX_PRIMARY_Q ⇒ Sparse.
        let wide = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 10 },
                slopes: (1..=crate::consts::MAX_PRIMARY_Q as u32).collect(), // q_p = 1 + MAX_PRIMARY_Q
                extra_groupings: vec![],
            }),
        };
        assert!(matches!(
            super::classify_design_pub(&wide, 1),
            super::Solver::Sparse
        ));
    }

    /// The measured q_g performance boundary (d2 Phase-1 crossover sweep) for
    /// Gaussian, and the only-implemented-route boundary for non-Gaussian: ANY
    /// slope-carrying extra grouping routes Sparse. Gaussian intercept-only
    /// extras (q_g = 1) stay NoZ (NoZ won 12–15× on the measured slice); the
    /// dense NoZ GLMM kernel builds intercept-only extras exclusively, so for
    /// non-Gaussian families Sparse is a correctness route, not a perf choice.
    #[test]
    fn classify_routes_slope_extras_to_sparse_all_families() {
        let spec = |family: Family, extra_slopes: Vec<u32>| ModelSpec {
            family,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 10 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 5 },
                    slopes: extra_slopes,
                }],
            }),
        };
        // Gaussian, slope-carrying extra (q_g = 2, in-envelope) ⇒ Sparse.
        let g_slope = spec(Family::Gaussian, vec![1]);
        assert!(matches!(
            super::classify_design_pub(&g_slope, 1),
            super::Solver::Sparse
        ));
        // Gaussian, intercept-only extra ⇒ NoZ.
        let g_int = spec(Family::Gaussian, vec![]);
        assert!(matches!(
            super::classify_design_pub(&g_int, 1),
            super::Solver::NoZ
        ));
        // Non-Gaussian, slope-carrying extra ⇒ Sparse (the only kernel that fits it).
        let p_slope = spec(
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            vec![1],
        );
        assert!(matches!(
            super::classify_design_pub(&p_slope, 1),
            super::Solver::Sparse
        ));
    }

    /// A fixed-only model always routes NoZ (no RE to make sparse).
    #[test]
    fn classify_fixed_only_is_noz() {
        let ols = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        assert!(matches!(
            super::classify_design_pub(&ols, 1),
            super::Solver::NoZ
        ));
    }

    /// AGQ-bypass canary (the `GLMM_RHO_END` canary's two-stage counterpart).
    /// nAGQ>1 fits bypass stage 1: the `two_stage && nagq == 1` gate
    /// excludes them (Profile deviance is undefined on the AGQ early-return path,
    /// `debug_assert!(!profile_beta || nagq == 1)`), so setting `ws.two_stage = true`
    /// on an AGQ fit must be a strict no-op. Runs the Poisson grouseticks AGQ fixture
    /// (nAGQ=7) through `crate::glmm::fit_glmm` both ways and asserts β̂, θ̂, τ̂², and
    /// n_eval are BIT-identical — the bypass is clean.
    #[test]
    fn two_stage_agq_bypass_is_bit_identical() {
        let csv = include_str!("../parity/data/grouseticks.csv");
        let p = 4;
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut raw = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            raw.push(f[0].parse().unwrap());
            let year: u32 = f[4].parse().unwrap();
            x.extend_from_slice(&[
                1.0,
                f64::from(u32::from(year == 96)),
                f64::from(u32::from(year == 97)),
                f[6].parse().unwrap(),
            ]);
            y.push(f[1].parse().unwrap());
        }
        let (cluster_ids, n_clusters) = dense_ids(&raw);
        let n = y.len();
        let nagq = 7u8;
        let model = ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters {
                    n_clusters: n_clusters as u32,
                },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let sized = spec_sized_from_ids_pub(
            &model,
            &GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            },
        );
        let mut xm = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xm[(i, j)] = x[i * p + j];
            }
        }
        let beta_start =
            glm_warm_start_beta(sized.family, f64::NAN, xm.as_ref().subrows(0, n), &y, n, p);

        let run = |two_stage: bool| -> (Vec<f64>, Vec<f64>, f64, usize) {
            let mut ws = GlmmWorkspace::for_cluster_spec(p, &sized, n, &[], nagq);
            build_z(&mut ws, xm.as_ref().subrows(0, n), &cluster_ids, &[], n);
            ws.two_stage = two_stage;
            let fit = crate::glmm::fit_glmm(
                &mut ws,
                xm.as_ref().subrows(0, n),
                &y,
                &cluster_ids,
                &[0, 1, 2, 3],
                None,
                &beta_start,
                n,
                WaldSe::Rx,
            );
            assert!(
                fit.converged,
                "AGQ fit (two_stage={two_stage}) must converge"
            );
            (
                ws.betas[..p].to_vec(),
                ws.params[..ws.n_theta].to_vec(),
                fit.tau_squared_hat,
                fit.n_eval,
            )
        };
        let (b1, t1, tau1, ne1) = run(false);
        let (b2, t2, tau2, ne2) = run(true);
        for j in 0..p {
            assert_eq!(
                b1[j].to_bits(),
                b2[j].to_bits(),
                "AGQ bypass: β[{j}] must be bit-identical"
            );
        }
        for t in 0..t1.len() {
            assert_eq!(
                t1[t].to_bits(),
                t2[t].to_bits(),
                "AGQ bypass: θ[{t}] must be bit-identical"
            );
        }
        assert_eq!(
            tau1.to_bits(),
            tau2.to_bits(),
            "AGQ bypass: τ̂² must be bit-identical"
        );
        assert_eq!(
            ne1, ne2,
            "AGQ bypass: n_eval must be identical (stage 1 skipped)"
        );
    }

    /// Two-stage A/B for a fixture whose data/model helpers live only in fit.rs's
    /// private `#[cfg(test)]` module (unreachable from glmm/tests.rs). Mirrors
    /// glmm/tests.rs `assert_two_stage_matches_single`: two fresh workspaces —
    /// single- vs two-stage — must land on the same optimum at ORACLE tolerances
    /// (β_rel 1e-3; θ abs+rel 1e-3 band; τ² rel 1e-3). Prints the
    /// `(n_eval_single, n_eval_two)` pair for the baseline doc; NO n_eval assertion —
    /// the eval-count win is a separate, measured concern. Drives
    /// `crate::glmm::fit_glmm` directly so `ws.two_stage` is settable.
    fn assert_two_stage_matches_single_local(
        label: &str,
        model: &ModelSpec,
        x: &[f64],
        y: &[f64],
        ids: &GroupIds,
        n: usize,
        p: usize,
    ) -> (usize, usize) {
        let sized = spec_sized_from_ids_pub(model, ids);
        let mut xm = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                xm[(i, j)] = x[i * p + j];
            }
        }
        let beta_start =
            glm_warm_start_beta(sized.family, f64::NAN, xm.as_ref().subrows(0, n), y, n, p);
        let targets: Vec<u32> = (0..p as u32).collect();

        let run = |two_stage: bool| -> (Vec<f64>, Vec<f64>, f64, usize) {
            let mut ws = GlmmWorkspace::for_cluster_spec(p, &sized, n, &[], 1);
            ws.nb_theta = f64::NAN; // non-NB families ignore it (mirrors fit_glmm_impl)
            build_z(
                &mut ws,
                xm.as_ref().subrows(0, n),
                &ids.primary,
                &ids.extra,
                n,
            );
            ws.structured_schur = if ws.groupings.structured_extras_eligible() {
                StructuredSchur::new(&ws.groupings, &ids.primary, &ids.extra, n)
            } else {
                None
            };
            ws.two_stage = two_stage;
            let fit = crate::glmm::fit_glmm(
                &mut ws,
                xm.as_ref().subrows(0, n),
                y,
                &ids.primary,
                &targets,
                None,
                &beta_start,
                n,
                WaldSe::Rx,
            );
            assert!(
                fit.converged,
                "{label}: {} fit must converge",
                if two_stage {
                    "two-stage"
                } else {
                    "single-stage"
                }
            );
            (
                ws.betas[..p].to_vec(),
                ws.params[..ws.n_theta].to_vec(),
                fit.tau_squared_hat,
                fit.n_eval,
            )
        };
        let (b1, t1, tau1, ne1) = run(false);
        let (b2, t2, tau2, ne2) = run(true);
        for j in 0..p {
            let rel = (b1[j] - b2[j]).abs() / b1[j].abs().max(1e-6);
            assert!(
                rel < 1e-3,
                "{label}: β[{j}] single {} vs two-stage {} (rel {rel})",
                b1[j],
                b2[j]
            );
        }
        for t in 0..t1.len() {
            assert!(
                (t1[t] - t2[t]).abs() < 1e-3 * (1.0 + t1[t].abs()),
                "{label}: θ[{t}] single {} vs two-stage {}",
                t1[t],
                t2[t]
            );
        }
        let trel = (tau1 - tau2).abs() / tau1.abs().max(1e-6);
        assert!(
            trel < 1e-3,
            "{label}: τ² single {tau1} vs two-stage {tau2} (rel {trel})"
        );
        println!("{label} n_eval: single {ne1} vs two {ne2}");
        (ne1, ne2)
    }

    /// Two-stage A/B on the two GLMM fixtures whose helpers are private to this
    /// module — the cbpp probit binomial GLMM (non-canonical link, blocked path,
    /// lme4-validated) and the sim_gamma log-link mixed model (a distinct
    /// non-canonical / dispersion PIRLS path with zero prior two-stage coverage).
    /// `#[ignore]`: part of the explicit two-stage corpus proof, out of the fast
    /// suite (like the glmm/tests.rs corpus sweep).
    #[test]
    #[ignore]
    fn two_stage_matches_single_stage_cbpp_probit_and_gamma() {
        // cbpp probit binomial GLMM (blocked, non-canonical probit link).
        {
            let (x, y, cluster_ids, n) = cbpp_design();
            let mut model = cbpp_model();
            model.family = Family::Binomial {
                link: BinomialLink::Probit,
            };
            let ids = GroupIds {
                primary: cluster_ids,
                extra: vec![],
            };
            assert_two_stage_matches_single_local("cbpp_probit", &model, &x, &y, &ids, n, 4);
        }
        // Gamma log-link mixed model (blocked, non-canonical + dispersion PIRLS path).
        {
            let (x, y, cluster_ids, n_clusters) =
                sim_clustered(include_str!("../parity/data/sim_gamma.csv"));
            let n = y.len();
            let model = ModelSpec {
                family: Family::Gamma {
                    link: crate::GammaLink::Log,
                },
                re: Some(ReStructure {
                    sizing: Sizing::FixedClusters {
                        n_clusters: n_clusters as u32,
                    },
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
            };
            let ids = GroupIds {
                primary: cluster_ids,
                extra: vec![],
            };
            assert_two_stage_matches_single_local("gamma_sim", &model, &x, &y, &ids, n, 3);
        }
    }
}
