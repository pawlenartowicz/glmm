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
    /// lower-triangular** covariance block `D̂ = σ̂²·Λ̂Λ̂'` per grouping, in
    /// declaration order (primary, then each extra). σ̂² is the residual scale
    /// for an LMM and the free GLMM scale `pwrss/n` for dispersion families
    /// (Gamma) — exactly the factor lme4's `VarCorr` stddevs carry; it is ≡ 1
    /// for binomial/Poisson/NB, and the same scale `tau2` reports, so the two
    /// accessors agree. Vech order is column-major
    /// lower-triangular (matching the θ vech convention): for a `q×q` block,
    /// `(0,0),(1,0),…,(q-1,0),(1,1),…,(q-1,q-1)`. Validated against lme4
    /// `VarCorr` (`parity/goldens/sleepstudy_lmm.json`; Gamma scale:
    /// `parity/goldens/sim_gamma_glmm.json`). Empty for OLS/GLM
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
    ///
    /// Scale caveat for dispersion families (Gamma): this SE stays on the
    /// **θ scale** — lme4's θ-Hessian convention — while `varcorr`/`tau2` carry
    /// the σ̂² factor. Deliberately NOT multiplied by σ̂: the joint Hessian does
    /// not carry cov(σ̂, θ̂), so a σ̂-rescaled SE would be a delta-method value
    /// that matches no oracle. For the φ≡1 families the two scales coincide and
    /// there is no split. (The parity harness skips `sd_se` gating for
    /// dispersion families accordingly.)
    pub stddev_se: Vec<f64>,
    /// Rank-deficiency mask, length `p`: `true` for a fixed-effect
    /// column dropped because it is aliased (linearly dependent) on an earlier
    /// column, mirroring lme4's `NA`-coefficient behavior. The corresponding
    /// `beta`/`se` slots are `NaN` and `converged` stays `true` (the reduced
    /// model fits). All-`false` when the design is full-rank.
    pub aliased: Vec<bool>,
    /// Objective evaluations consumed by the θ (LMM) / joint [θ|β] (GLMM)
    /// BOBYQA search — GLMM counts both stages. 0 where no derivative-free
    /// optimizer runs (OLS/GLM closed-form or IRLS paths). Deterministic and
    /// clock-independent; the optimizer-grid campaign's primary metric.
    pub n_eval: usize,
    /// Minimized optimizer criterion at the accepted point. LMM: the profiled
    /// REML deviance as computed by `reml_deviance` — equals lme4's REMLcrit
    /// minus the data-independent constant df·(1 + ln 2π), df = n − p
    /// (validated against the frozen lme4 sleepstudy reference). GLMM: the
    /// marginal Laplace deviance `d(y,ũ) + ‖ũ‖² + log|A|` (see
    /// `GlmmFit::deviance`), which differs from −2·logLik by a data-only
    /// saturated constant. NaN for OLS/GLM and on non-converged fits
    /// (GLMM non-convergence surfaces as +∞ internally; mapped to NaN here).
    pub deviance: f64,
    /// `true` iff the fit converged onto the θ boundary (≥ 1 diagonal variance
    /// component pinned at 0 — `boundary_hit == 1` internally), the same
    /// condition lme4's `isSingular` reports. `false` for OLS/GLM.
    pub singular: bool,
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
        #[allow(clippy::needless_range_loop)]
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
    /// weights. `wᵢ` scales each row's loglik/deviance contribution (the
    /// Aitken/M&N prior-weight convention — not an inverse-variance analytic
    /// weight). For an aggregated binomial, `y` is the success PROPORTION and
    /// `wᵢ` the trial count: this is exactly lme4's `cbind(s, m−s)` objective,
    /// whose deviance differs from the expanded-Bernoulli one only by a
    /// data-only saturated constant (same argmin — same β/SE/varcomp).
    /// Dispersion/σ̂² always divide by the raw row count `n−p`, never `Σwᵢ−p`.
    ///
    /// Support matrix: every (family, RE structure, solver) combination, at
    /// `nagq == 1` only — AGQ (`nagq > 1`) rejects weights at the boundary (see
    /// `fit_warm`'s capability map), since the three per-row `dev_resid` sums
    /// AGQ's quadrature threads through (`glmm/agq.rs`) assume unit weights.
    ///
    /// - Gaussian fixed-only: WLS via √wᵢ row pre-scaling, `σ̂² = Σwᵢrᵢ²/(n−p)`;
    ///   matches R `lm(weights=)` (`fit_ols_weighted_matches_r_lm`).
    /// - GLM fixed-only, any family (Binomial/Poisson/Gamma/NB): weighted IRLS —
    ///   `wᵢ` multiplies the working weight and deviance; Gamma's Pearson φ is
    ///   `Σwᵢrᵢ²/(n−p)`; the null deviance uses the weighted mean. Matches R
    ///   `glm(weights=)` (`fit_glm_gamma_weighted_matches_r`,
    ///   `fit_glm_binomial_weighted_aggregated_matches_r`,
    ///   `glm_weighted_deviance_null_golden_value`) and `MASS::glm.nb(weights=)`
    ///   (`fit_glm_nb_weighted_matches_mass`, weighted θ profile).
    /// - Sparse (`Solver::Sparse`) non-Gaussian GLMM, every family: `wᵢ` weights
    ///   the sparse PIRLS working weight/deviance/score identically to the
    ///   dense mixed path below; Gamma's profiled dispersion (`gamma_aic`) and
    ///   `vcov(use.hessian=FALSE)` scale (`glmm_sigma_sq`) take `ws.prior_w`,
    ///   its post-fit Pearson φ̂ sums `wᵢrᵢ²` over raw `n−p` df, and NB's
    ///   marginal-θ profile (`nb_profile_loglik`) takes `opts.weights`.
    ///   Validated against `fit_sparse_gamma_glmm_weighted_matches_lme4` (lme4
    ///   `glmer`), the `sparse_weighted_binomial_*` expanded-vs-aggregated
    ///   tests, and the `sparse_weighted_{poisson,gamma,nb}_matches_replicated`
    ///   weighted-vs-replicated-row equivalence tests.
    /// - Dense (`Solver::NoZ`) mixed GLMM, every non-Gaussian family
    ///   (Binomial/Poisson/Gamma/NB): `wᵢ` weights the working weight,
    ///   deviance, and β-gradient score identically to the sparse binomial path
    ///   above; Gamma's profiled dispersion (`family::gamma_aic`) and its
    ///   `vcov(use.hessian=FALSE)` scale (`family::glmm_sigma_sq`) both take
    ///   `Σwᵢ` in place of `n`. Validated against
    ///   `fit_glmm_cbpp_aggregated_matches_lme4`,
    ///   `fit_glmm_poisson_weighted_matches_lme4`, and
    ///   `fit_glmm_gamma_weighted_matches_lme4` (all lme4 `glmer`, dense).
    /// - Dense (`Solver::NoZ`) mixed Gaussian LMM (weighted REML): every
    ///   row of `[X y Z]` is conceptually √wᵢ-scaled before hitting the unit-
    ///   weight suff-stats/deviance kernel (`LmmSuffStats::add_rows_multi`),
    ///   which folds `wᵢ` straight into the Gram accumulators (`c`, `s`,
    ///   `counts` — the last now `Σ z²·wᵢ`, not a row count; `df = n − p` still
    ///   comes from the raw row count). `σ̂² = pwrss_w/(n−p)` (raw df) falls out
    ///   of the weighted Grams automatically; `tau2 = θ²·σ̂²` and SEs inherit.
    ///   The reported deviance carries an extra `−Σlog wᵢ` (the weighted
    ///   Gaussian log-density's `+½Σlog wᵢ` per row, on the −2ℓ scale) atop the
    ///   usual stripped-constant convention, matching lme4's weighted
    ///   `lmer(weights=)` REMLcrit up to that same stripped constant.
    ///   Validated against `fit_lmm_weighted_matches_lme4` (lme4 `lmer`, dense),
    ///   `fit_lmm_constant_weights_invariant`, and
    ///   `fit_lmm_weighted_boundary_matches_wls`.
    /// - Sparse (`Solver::Sparse`) Gaussian LMM: identical √wᵢ scaling threaded
    ///   through the sparse-Z accumulator instead — `for_each_z_entry`'s
    ///   emitted z values and every raw x/y read in `SparseLmmWorkspace::new`'s
    ///   pass 2 each carry one `√wᵢ` factor, so every Gram product (`ztxy`,
    ///   `cxy`, the packed `pk_*` raw-Gram streams) ends up carrying exactly
    ///   `wᵢ`. Deviance/df stay raw (`n − p`); `fit_mle_sparse` adds the same
    ///   `−Σlog wᵢ` constant as `fit_mle` above. Validated against
    ///   `fit_sparse_lmm_weighted_matches_lme4` (lme4 `lmer`, sparse) and
    ///   `sparse_lmm_constant_weights_invariant`.
    pub weights: Option<Vec<f64>>,
    /// **Experimental.** Opt into the parallel inner-fit kernels (cluster-outer
    /// AGQ, per-`(i,j)` FD-Hessian grid) — live ONLY when built with the
    /// `parallel` Cargo feature on a non-`wasm32` target. The exclusion is
    /// compile-time, not a runtime check: serial and `wasm32` builds run the
    /// original sequential kernels byte-for-byte regardless of this flag, and
    /// never pull `rayon` in.
    /// (Cluster-outer AGQ is deliberately NOT offered serially: its per-cluster
    /// overhead regresses many-tiny-cluster shapes — measured +12–16% on
    /// observation-level REs — so it exists only as the rayon substrate.)
    ///
    /// Default `false`: parallel results are bit-identical to serial (tested),
    /// but the kernels are new and their perf envelope isn't characterized yet,
    /// so a fit uses them only on explicit opt-in. Batch callers driving many
    /// fits at once (e.g. MCPower's `loop_advanced` hot loop) should keep this
    /// `false` even after opting builds into `parallel` — that caller already
    /// owns the outer parallelism, and stacking a second parallel axis inside
    /// every fit adds per-split overhead (and, for AGQ, a real per-fit CSR
    /// preprocessing cost) for no benefit once the outer loop saturates the
    /// cores.
    ///
    /// Wired to the AGQ cluster-outer restructuring (`agq::agq_deviance`'s
    /// `cluster_rows` path): `false` skips the per-fit `ClusterRowIndex` build and
    /// runs the original node-outer loop. Also gates the FD-Hessian grids (dense and
    /// sparse): `false` keeps the serial cell-by-cell loop. See the design doc above
    /// for the full rationale, including why nesting this under an outer `rayon`
    /// batch loop is safe (one shared work-stealing pool) in a way naive OS-thread
    /// or BLAS-thread nesting is not.
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
            parallel_inner: false,
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
/// kernel is Bernoulli) — except when `y` is an aggregated success PROPORTION
/// and [`FitOptions::weights`] carries the trial count, which fits directly
/// without the expansion. That aggregated form is now supported on every
/// (family, RE, solver) path at `nagq == 1` (see `FitOptions::weights`'s
/// support matrix); `nagq > 1` (AGQ) rejects weights entirely.
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
    // Prior weights: shape-check at the boundary, then reject unsupported paths
    // here (not deep in a kernel) via the capability map below.
    if let Some(w) = &opts.weights {
        assert_eq!(w.len(), n, "FitOptions.weights must have n elements");
        assert!(
            w.iter().all(|&v| v.is_finite() && v > 0.0),
            "FitOptions.weights must be finite and > 0"
        );
        // AGQ (nAGQ>1) with prior weights: no wired use case — the three
        // per-row dev_resid sums in glmm/agq.rs assume unit weights.
        assert!(
            opts.nagq == 1,
            "FitOptions.weights with nAGQ > 1 is not supported"
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
            // Classify the SIZED spec: the level-count clause needs real
            // `n_clusters`, and the frontend passes placeholders (`sized ==
            // model` for callers who already pass real counts).
            match classify_design(&sized, opts.nagq) {
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
        n_eval: fr.n_eval,
        deviance: fr.deviance,
        singular: fr.singular,
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
    // The kernel holds ONE nested slot (`LmmGroupings.nested`) and
    // `from_cluster_spec_ext` would silently let a later `NestedWithin` extra
    // overwrite an earlier one (last wins). The formula frontend detects at most
    // one; this guards the explicit `parent:child` route (defense in depth).
    let n_nested = re
        .extra_groupings
        .iter()
        .filter(|g| matches!(g.relation, GroupingRelation::NestedWithin { .. }))
        .count();
    assert!(
        n_nested <= 1,
        "at most one NestedWithin extra grouping is supported (got {n_nested})"
    );
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
/// `Sparse`, and so does a total `Crossed` level count past
/// `MAX_CROSSED_LEVELS` — see the comments at the clauses.
///
/// The level-count clause reads `Crossed { n_clusters }`, so callers must pass
/// a spec with REAL counts (`spec_sized_from_ids`) — the formula frontend's
/// placeholders (`n_clusters: 1`) would never trip it.
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
    // Many-level crossed extras route Sparse: every crossed level lands in the
    // dense tail (`reml_deviance`'s `t_dim`; the dense GLMM path's Schur
    // complement likewise), which is cubic in the TOTAL crossed level count —
    // hence the sum over factors, not a per-factor cap. See MAX_CROSSED_LEVELS
    // for the measured blowup and the placeholder threshold's rationale.
    let crossed_levels: usize = re
        .extra_groupings
        .iter()
        .map(|g| match g.relation {
            GroupingRelation::Crossed { n_clusters } => n_clusters as usize,
            GroupingRelation::NestedWithin { .. } => 0,
        })
        .sum();
    if over || slope_extras || crossed_levels > crate::consts::MAX_CROSSED_LEVELS {
        Solver::Sparse
    } else {
        Solver::NoZ
    }
}

#[cfg(test)]
pub(crate) fn classify_design_pub(model: &ModelSpec, nagq: u8) -> Solver {
    classify_design(model, nagq)
}

// ---------------------------------------------------------------------------
// Dev-only adjudication seam (loop_advanced) — mismatch-oracle spec 2026-07-11.
// NOT semver-covered. Gaussian LMM only: exposes the exact profiled-REML
// closure `fit` minimizes, (a) evaluated at a caller-fixed θ and (b) minimized
// under a caller-configured schedule. The shipped path is untouched.
// ---------------------------------------------------------------------------

/// The θ ↦ profiled-REML-deviance closure the dev seam hands out.
#[cfg(feature = "loop_advanced")]
type LmmObjective<'a> = dyn FnMut(&[f64]) -> f64 + 'a;

/// Per-eval trace hook for [`lmm_sweep_fit`]: `(k, θ, f)` per objective call.
#[cfg(feature = "loop_advanced")]
pub type LmmTrace<'a> = dyn FnMut(usize, &[f64], f64) + 'a;

/// Marshal the LMM inputs exactly as `fit_mle` does — sized spec, slope
/// columns, workspace, suff-stats, balanced-collapse arming — and hand the
/// ready-to-evaluate workspace to `f`. Shared by the two dev entries so the
/// closure they see is the one BOBYQA sees inside `fit_cold` (dense route);
/// the sparse route builds `SparseLmmWorkspace` the way `fit_mle_sparse` does.
#[cfg(feature = "loop_advanced")]
fn with_lmm_objective<R>(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    f: impl FnOnce(&mut LmmObjective<'_>, &crate::lmm::LmmGroupings) -> R,
) -> R {
    assert!(
        matches!(model.family, Family::Gaussian) && model.re.is_some(),
        "dev objective seam covers Gaussian LMM only"
    );
    assert_group_ids(model.re.as_ref().unwrap(), ids, n);
    let sized = spec_sized_from_ids(model, ids);
    let re = sized.re.as_ref().unwrap();
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();
    match classify_design(&sized, 1) {
        Solver::NoZ => {
            let mut ws =
                LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &slope_cols, &extra_slope_cols);
            let p1 = p.max(1);
            let mut x_mat = Mat::<f64>::zeros(n.max(1), p1);
            for i in 0..n {
                for j in 0..p {
                    x_mat[(i, j)] = x[i * p + j];
                }
            }
            ws.suff.reset();
            ws.suff
                .add_rows_multi(x_mat.as_ref().subrows(0, n), y, &ids.primary, &ids.extra, None);
            let LmmWorkspace { suff, fit, .. } = &mut ws;
            crate::lmm::precompute_balanced_collapse(suff, fit);
            let mut obj = |theta: &[f64]| crate::lmm::reml_deviance(theta, suff, fit);
            f(&mut obj, &suff.groupings)
        }
        Solver::Sparse => {
            let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(
                &sized,
                n,
                &slope_cols,
                &extra_slope_cols,
            );
            let xm = faer::MatRef::from_row_major_slice(x, n, p);
            let mut ws = crate::sparse::SparseLmmWorkspace::new(
                &g,
                xm,
                &ids.primary,
                &ids.extra,
                y,
                n,
                p,
                None,
            );
            let mut obj = |theta: &[f64]| crate::sparse::sparse_reml_deviance(theta, &mut ws);
            f(&mut obj, &g)
        }
    }
}

/// Profiled REML deviance of the LMM objective at a fixed θ (glmm's own vech
/// layout: primary column-major lower triangle, then extras in declaration
/// order). Raw optimizer scale — the same value `Fit::deviance` reports for an
/// unweighted fit.
#[cfg(feature = "loop_advanced")]
pub fn lmm_objective_at(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    theta: &[f64],
) -> f64 {
    with_lmm_objective(x, y, n, p, model, ids, |obj, g| {
        assert_eq!(theta.len(), g.n_theta(), "theta length must match the model");
        obj(theta)
    })
}

/// Outcome of [`lmm_sweep_fit`]: the accepted point and objective, plus the
/// eval count and raw convergence bit (no pinning, no β recovery).
#[cfg(feature = "loop_advanced")]
pub struct LmmSweepOutcome {
    pub deviance: f64,
    pub theta: Vec<f64>,
    pub n_eval: usize,
    pub converged: bool,
}

/// Minimize the LMM objective under a caller-configured BOBYQA schedule.
/// `theta0` is used VERBATIM (`None` → the shipped blind start) — unlike
/// `fit`'s warm start, which clamps every component to `THETA_TRUTH_FLOOR`
/// and so cannot express a negative off-diagonal start. npt and rho_begin
/// are derived exactly as the shipped sites derive them (mid npt ⌈1.5n⌉+1
/// from n ≥ 3, rho_begin = min(0.1·min diag θ₀, RHO_BEGIN) floored at
/// 10·rho_end), so `(theta0 = None, rho_end = RHO_END, max_fun = None)`
/// replays a shipped grid fit trajectory-identically; `trace` then observes
/// every (k, θ, f) evaluation without any hook in the shipped path.
#[cfg(feature = "loop_advanced")]
#[allow(clippy::too_many_arguments)] // dev seam, marshals the fit_mle surface + schedule
pub fn lmm_sweep_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    theta0: Option<&[f64]>,
    rho_end: f64,
    max_fun: Option<usize>,
    mut trace: Option<&mut LmmTrace<'_>>,
) -> LmmSweepOutcome {
    use bobyqa::{Bobyqa, Config, Status};
    with_lmm_objective(x, y, n, p, model, ids, |obj, g| {
        let n_theta = g.n_theta();
        let (blind, lower, upper) = g.blind_theta_and_bounds();
        let mut theta = match theta0 {
            Some(t) => {
                assert_eq!(t.len(), n_theta, "theta0 length must match the model");
                t.to_vec()
            }
            // Mirror `fit_lmm`'s cold arm exactly (replay fidelity depends on
            // this): the blind seed — diagonals THETA0, off-diagonals 0 —
            // adopted by the shipped LMM paths in the 2026-07-11 basin fix.
            None => blind,
        };
        let min_diag = g
            .diagonal_theta()
            .iter()
            .map(|&i| theta[i])
            .fold(f64::INFINITY, f64::min);
        let rho_begin = (0.1 * min_diag)
            .min(crate::lmm::RHO_BEGIN)
            .max(10.0 * rho_end);
        let npt = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        let mut config = Config {
            rho_begin,
            rho_end,
            npt,
            ..Config::new(n_theta)
        };
        crate::lmm::apply_campaign_overrides(&mut config, n_theta);
        if let Some(mf) = max_fun {
            config.max_fun = mf;
        }
        let mut solver = Bobyqa::new(n_theta, config).expect("dev sweep config valid");
        let mut k = 0usize;
        let out = solver.minimize(
            |xs| {
                let v = obj(xs);
                k += 1;
                if let Some(t) = trace.as_mut() {
                    t(k, xs, v);
                }
                v
            },
            &mut theta,
            &lower,
            &upper,
        );
        LmmSweepOutcome {
            deviance: obj(&theta),
            theta,
            n_eval: out.n_eval,
            converged: matches!(out.status, Status::Converged),
        }
    })
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
        n_eval: 0,
        deviance: f64::NAN,
        singular: false,
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
    // WLS by √wᵢ row scaling: the unweighted accumulator then yields
    // X'WX / X'Wy / y'Wy, so RSS = y'Wy − β̂'X'Wy is the weighted RSS and
    // σ̂² = RSS/(n−p) matches R lm(weights=) (raw-row df). `sum_y`/`sst`
    // are NOT weight-consistent under this scaling but are never mapped
    // into `Fit` — leave them unwired.
    let sqrt_w: Option<Vec<f64>> = opts
        .weights
        .as_ref()
        .map(|w| w.iter().map(|v| v.sqrt()).collect());
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p1);
    for i in 0..n {
        let s = sqrt_w.as_ref().map_or(1.0, |sw| sw[i]);
        for j in 0..p {
            x_mat[(i, j)] = s * x[i * p + j];
        }
    }
    let y_scaled: Vec<f64>;
    let y_eff: &[f64] = match &sqrt_w {
        Some(sw) => {
            y_scaled = y.iter().zip(sw).map(|(&yi, &s)| yi * s).collect();
            &y_scaled
        }
        None => y,
    };

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
            suff.add_rows(x_mat.as_ref().subrows(0, n), y_eff);
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
        n_eval: 0,
        deviance: f64::NAN,
        singular: false,
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
        ws.suff.add_rows_multi(
            x_mat.as_ref().subrows(0, n),
            y,
            cluster_ids,
            extra_ids,
            opts.weights.as_deref(),
        );
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

    // Weighted Gaussian log-density carries +½Σlog wᵢ per row; on the −2ℓ
    // scale the REML/ML deviance gains −Σlog wᵢ (θ-independent — added
    // post-optimization, argmin unchanged). Matches lme4's weighted REMLcrit
    // up to the engine's documented stripped constant (see lme.rs:2978).
    let dev = match &opts.weights {
        Some(w) => lmm_fit.deviance - w.iter().map(|v| v.ln()).sum::<f64>(),
        None => lmm_fit.deviance,
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
        n_eval: lmm_fit.n_eval,
        deviance: dev,
        singular: lmm_fit.boundary_hit == 1,
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
            opts.weights.as_deref(),
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
    // estimates the Pearson moment `φ̂=Σ wᵢrᵢ²/(n−p)`, `rᵢ=(yᵢ−μ̂ᵢ)/√V(μ̂ᵢ)`,
    // raw-row df — exactly `summary(glm(family=Gamma, weights=w))$dispersion`.
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
                        let pw = opts.weights.as_ref().map_or(1.0, |w| w[i]);
                        s += pw * r * r;
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
        n_eval: 0,
        deviance: f64::NAN,
        singular: false,
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
///
/// `weights: Some(w)` multiplies each row's contribution by the prior weight
/// `wᵢ` (matches `MASS::glm.nb(weights=)`'s profile — `theta.ml` weights the
/// per-row deviance terms the same way `glm.fit` weights IRLS); `None` is unit
/// weights.
pub(crate) fn nb_profile_loglik(y: &[f64], mu: &[f64], theta: f64, weights: Option<&[f64]>) -> f64 {
    let mut ll = 0.0;
    for (i, (&yi, &mi)) in y.iter().zip(mu.iter()).enumerate() {
        let mut s = 0.0;
        for k in 0..(yi.round() as u64) {
            s += (theta + k as f64).ln();
        }
        if mi > 0.0 {
            s += theta * (theta / (theta + mi)).ln() + yi * (mi / (theta + mi)).ln();
        }
        ll += weights.map_or(1.0, |w| w[i]) * s;
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

/// Maximise [`nb_profile_loglik`] over θ at fixed μ̂, weighted by `weights`
/// (`FitOptions::weights`, forwarded from the caller). Returns θ̂.
fn optimize_nb_theta(y: &[f64], mu: &[f64], weights: Option<&[f64]>) -> f64 {
    golden_max_ln_theta(|t| nb_profile_loglik(y, mu, t.exp(), weights))
}

/// Negative-binomial GLM via the alternating outer-θ loop (`MASS::glm.nb`):
/// (1) fit the GLM at fixed θ; (2) 1-D maximise the NB profile log-likelihood
/// over θ holding β̂/μ̂; (3) repeat to convergence. `theta_seed = Some(v)` seeds
/// step 1; `None` cold-starts from a method-of-moments estimate
/// `θ₀ = ȳ²/max(s²−ȳ, ε)` (seed only, left unweighted — the outer loop's
/// weighted θ optimisation below washes out the seed's influence). The β SE
/// conditions on θ̂ (lme4/MASS convention; θ-uncertainty is out of scope), so
/// `dispersion = θ̂` and the SE comes straight from the final fixed-θ fit (NB
/// has `φ≡1`; overdispersion lives in `V=μ+μ²/θ`).
fn fit_glm_nb(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    theta_seed: Option<f64>,
    opts: &FitOptions,
) -> Fit {
    fit_glm_nb_capped(x, y, n, p, theta_seed, opts, NB_MAX_OUTER)
}

/// [`fit_glm_nb`]'s loop body with the outer-iteration cap as a parameter —
/// a seam so the cap-exhaustion semantics are testable without engineering a
/// dataset that legitimately burns all `NB_MAX_OUTER` alternations. Production
/// code only ever calls it through [`fit_glm_nb`] (cap = `NB_MAX_OUTER`).
///
/// Cap-exhaustion semantics (pinned by `fit_glm_nb_outer_cap_semantics`): the
/// exit is SILENT — `converged` reflects only the last inner IRLS fit, β/se
/// stay at the stale pre-update θ, and `dispersion` carries the newer θ.
#[allow(clippy::too_many_arguments)]
fn fit_glm_nb_capped(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    theta_seed: Option<f64>,
    opts: &FitOptions,
    max_outer: usize,
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
    for _ in 0..max_outer {
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
        let new_theta = optimize_nb_theta(y, &mu, opts.weights.as_deref());
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
    ws.parallel_inner = opts.parallel_inner;
    if let Some(w) = &opts.weights {
        ws.prior_w[..n].copy_from_slice(w);
        ws.weighted = true;
    }
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
                n_eval: 0,
                deviance: f64::NAN,
                singular: false,
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
            None => {
                let mut s = 0.0;
                for (i, (&yi, &mu)) in y[..n].iter().zip(ws.prob[..n].iter()).enumerate() {
                    let r = (yi - mu) / crate::family::variance(model.family, nb_theta, mu).sqrt();
                    s += ws.prior_w[i] * r * r;
                }
                s / (n - p) as f64
            }
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
            n_eval: glmm_fit.n_eval,
            deviance: if glmm_fit.deviance.is_finite() {
                glmm_fit.deviance
            } else {
                f64::NAN
            },
            singular: glmm_fit.boundary_hit == 1,
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
        // `dev` is already weighted (opts threads through fit_glmm → ws.prior_w,
        // 4c); the saturated-reference term takes the same per-row weights so
        // both halves of `logL_marginal` are on the same weighted scale.
        -0.5 * dev + nb_profile_loglik(y, y, th, opts.weights.as_deref())
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

    /// WLS through the stable surface, gated against R `lm(weights=)`.
    /// Convention: σ̂² = Σwᵢrᵢ²/(n−p) with raw-row-count df (R's summary.lm).
    #[test]
    fn fit_ols_weighted_matches_r_lm() {
        // R 4.5.3 oracle, data as in the vectors below:
        //   f <- lm(y ~ x, weights = w); print(coef(summary(f)), digits = 15)
        // REF_BETA/REF_SE are the Estimate / Std. Error columns.
        let xv = [
            0.2, 1.4, -0.8, 2.1, 0.5, -1.3, 1.9, 0.0, -0.6, 1.1, 2.4, -1.7,
        ];
        let w = vec![1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 4.0, 2.0, 1.0, 3.0, 2.0];
        let y = vec![
            0.8, 3.1, -1.2, 4.6, 1.4, -2.0, 4.1, 0.3, -0.9, 2.6, 5.2, -3.1,
        ];
        let n = 12;
        let mut x = Vec::with_capacity(n * 2);
        for &xi in &xv {
            x.extend_from_slice(&[1.0, xi]);
        }
        const REF_BETA: [f64; 2] = [0.371528122456273, 1.996237765292144];
        const REF_SE: [f64; 2] = [0.0289002251717619, 0.0195893362923423];
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let opts = FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..FitOptions::default()
        };
        let f = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &opts);
        assert!(f.converged);
        for j in 0..2 {
            assert!((f.beta[j] - REF_BETA[j]).abs() < 1e-9, "beta[{j}]");
            assert!((f.se[j] - REF_SE[j]).abs() < 1e-9, "se[{j}]");
        }
    }

    /// Constant weights w≡c must reproduce the unweighted fit exactly:
    /// β̂ is scale-invariant and σ̂²(X'WX)⁻¹ cancels the c.
    #[test]
    fn fit_ols_constant_weights_invariant() {
        let xv = [0.2, 1.4, -0.8, 2.1, 0.5, -1.3, 1.9, 0.0];
        // A tiny perturbation on one point keeps this off the exact-fit (RSS≈0)
        // edge, where closed-form RSS = y'y − β̂'X'y catastrophically cancels
        // and the sign of the residual float noise (not weighting) decides
        // whether `var_diag` clears its `>= 0` finite guard.
        let y: Vec<f64> = xv
            .iter()
            .enumerate()
            .map(|(i, v)| 1.0 + 2.0 * v + if i == 0 { 0.01 } else { 0.0 })
            .collect();
        let n = 8;
        let mut x = Vec::with_capacity(n * 2);
        for &xi in &xv {
            x.extend_from_slice(&[1.0, xi]);
        }
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let base = FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        };
        let f0 = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &base);
        let opts = FitOptions {
            weights: Some(vec![3.0; n]),
            ..base
        };
        let f1 = fit_cold(&x, &y, n, 2, &model, &GroupIds::default(), &opts);
        for j in 0..2 {
            assert!((f0.beta[j] - f1.beta[j]).abs() < 1e-12);
            assert!((f0.se[j] - f1.se[j]).abs() < 1e-12);
        }
    }

    /// Terminal boundary: nAGQ>1 is the ONLY closed shape once Tasks 1-7 wired
    /// every (family, RE, solver) combination at nAGQ=1. The three per-row
    /// `dev_resid` sums AGQ's quadrature threads through (`glmm/agq.rs`) assume
    /// unit weights, so `fit_warm`'s capability map rejects `weights.is_some()`
    /// whenever `nagq > 1` rather than silently dropping them. Fixture: a
    /// scalar-intercept binomial GLMM (the one shape nAGQ=3 is otherwise legal
    /// on, per `assert_model_shape`) with weights — must still panic.
    #[test]
    #[should_panic(expected = "nAGQ > 1")]
    fn weights_rejected_with_agq() {
        let n = 8;
        let x = vec![1.0f64; n];
        let y: Vec<f64> = (0..n).map(|i| f64::from(u32::from(i % 3 == 0))).collect();
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let ids = GroupIds {
            primary: vec![0, 0, 0, 0, 1, 1, 1, 1],
            extra: vec![],
        };
        let opts = FitOptions {
            weights: Some(vec![2.0; n]),
            nagq: 3,
            ..FitOptions::default()
        };
        let _ = fit_cold(&x, &y, n, 1, &model, &ids, &opts);
    }

    /// Shape asserts (length + positivity) landed in Task 1 and never moved —
    /// this pins the wrong-length case still faults on an otherwise-open path
    /// (fixed-only OLS), independent of the family/RE capability map above.
    #[test]
    #[should_panic(expected = "n elements")]
    fn weights_shape_still_asserted() {
        let n = 4;
        let x = vec![1.0f64; n];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let model = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let opts = FitOptions {
            weights: Some(vec![1.0; n - 1]),
            ..FitOptions::default()
        };
        let _ = fit_cold(&x, &y, n, 1, &model, &GroupIds::default(), &opts);
    }

    /// Weighted Gamma(log) GLM vs R glm(weights=). Convention: prior weight
    /// multiplies the IRLS working weight and deviance; Pearson dispersion
    /// φ = Σwᵢrᵢ²/(n−p), raw-row df (R summary.glm).
    #[test]
    fn fit_glm_gamma_weighted_matches_r() {
        // R 4.5.3 oracle (set.seed(42), n = 40):
        //   x1 <- round(rnorm(n), 4); w <- sample(1:4, n, replace = TRUE)
        //   eta <- 0.4 + 0.8 * x1
        //   yg <- round(rgamma(n, shape = 2, scale = exp(eta) / 2), 6)
        //   fg <- glm(yg ~ x1, family = Gamma("log"), weights = w)
        //   print(coef(summary(fg)), digits = 15); print(summary(fg)$dispersion, digits = 15)
        let x1: [f64; 40] = [
            1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627,
            1.3049, 2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201,
            -0.3066, -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64,
            0.4555, 0.7048, 1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
        ];
        let w: Vec<f64> = vec![
            4.0, 1.0, 2.0, 1.0, 1.0, 4.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 1.0, 4.0, 4.0, 2.0, 1.0,
            4.0, 2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 4.0, 3.0, 4.0, 1.0, 4.0, 1.0, 4.0, 3.0,
            2.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ];
        let yg: Vec<f64> = vec![
            2.421196, 0.850101, 1.188318, 0.917668, 1.895064, 2.717167, 4.391082, 0.266883,
            1.853922, 1.838375, 5.959549, 19.008523, 0.121882, 1.544704, 1.422566, 0.758422,
            1.264496, 0.147806, 0.06751, 2.907132, 0.3538, 0.223494, 0.297625, 5.273375, 12.534684,
            0.514577, 1.473477, 0.485665, 0.962023, 1.043896, 1.771311, 1.926229, 7.592099,
            1.298714, 0.675125, 0.201756, 1.814679, 1.104297, 0.434436, 0.470596,
        ];
        const REF_BETA: [f64; 2] = [0.423197712262065, 0.845082014360343];
        const REF_SE: [f64; 2] = [0.0960484092896012, 0.0763975129700953];
        const REF_DISPERSION: f64 = 0.885577425465437;
        let n = 40;
        let mut x = Vec::with_capacity(n * 2);
        for &xi in &x1 {
            x.extend_from_slice(&[1.0, xi]);
        }
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: None,
        };
        let opts = FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w),
            ..FitOptions::default()
        };
        let f = fit_cold(&x, &yg, n, 2, &model, &GroupIds::default(), &opts);
        assert!(f.converged);
        for j in 0..2 {
            assert!((f.beta[j] - REF_BETA[j]).abs() < 1e-6, "beta[{j}]");
            assert!((f.se[j] - REF_SE[j]).abs() < 1e-6, "se[{j}]");
        }
        assert!((f.dispersion - REF_DISPERSION).abs() / REF_DISPERSION < 1e-6);
    }

    /// Weighted binomial-logit GLM on aggregated (proportion, trial-count)
    /// rows vs R glm(weights=). Exercises the weighted-logit fallthrough to
    /// the general IRLS arm (the fused SIMD logit kernel cannot take
    /// per-row weights, see `glm_irls_fit`'s `prior_w` doc).
    #[test]
    fn fit_glm_binomial_weighted_aggregated_matches_r() {
        // R 4.5.3 oracle (same x1/eta as the Gamma golden above, set.seed(42)):
        //   m <- sample(2:6, n, replace = TRUE)
        //   s <- rbinom(n, m, plogis(eta)); yp <- s / m
        //   fb <- glm(yp ~ x1, family = binomial, weights = m)
        //   print(coef(summary(fb)), digits = 15)
        let x1: [f64; 40] = [
            1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627,
            1.3049, 2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201,
            -0.3066, -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64,
            0.4555, 0.7048, 1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
        ];
        let m: Vec<f64> = vec![
            5.0, 2.0, 6.0, 5.0, 2.0, 2.0, 2.0, 5.0, 3.0, 4.0, 6.0, 6.0, 5.0, 2.0, 4.0, 5.0, 6.0,
            3.0, 2.0, 2.0, 2.0, 4.0, 3.0, 6.0, 5.0, 5.0, 6.0, 2.0, 5.0, 2.0, 2.0, 6.0, 4.0, 2.0,
            3.0, 5.0, 3.0, 6.0, 6.0, 4.0,
        ];
        let yp: Vec<f64> = vec![
            0.800000000000000,
            0.000000000000000,
            0.833333333333333,
            0.600000000000000,
            0.000000000000000,
            0.500000000000000,
            0.500000000000000,
            0.400000000000000,
            0.666666666666667,
            0.250000000000000,
            1.000000000000000,
            1.000000000000000,
            0.200000000000000,
            0.500000000000000,
            0.500000000000000,
            0.800000000000000,
            0.666666666666667,
            0.333333333333333,
            0.000000000000000,
            1.000000000000000,
            1.000000000000000,
            0.500000000000000,
            0.666666666666667,
            1.000000000000000,
            1.000000000000000,
            0.200000000000000,
            0.666666666666667,
            0.500000000000000,
            0.400000000000000,
            0.500000000000000,
            0.500000000000000,
            1.000000000000000,
            1.000000000000000,
            0.500000000000000,
            0.666666666666667,
            0.400000000000000,
            1.000000000000000,
            0.500000000000000,
            0.166666666666667,
            0.500000000000000,
        ];
        const REF_BETA: [f64; 2] = [0.512593391575506, 0.822576961628648];
        const REF_SE: [f64; 2] = [0.181425472435286, 0.170693131259756];
        let n = 40;
        let mut x = Vec::with_capacity(n * 2);
        for &xi in &x1 {
            x.extend_from_slice(&[1.0, xi]);
        }
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: None,
        };
        let opts = FitOptions {
            target_indices: vec![0, 1],
            weights: Some(m),
            ..FitOptions::default()
        };
        let f = fit_cold(&x, &yp, n, 2, &model, &GroupIds::default(), &opts);
        assert!(f.converged);
        for j in 0..2 {
            assert!((f.beta[j] - REF_BETA[j]).abs() < 1e-6, "beta[{j}]");
            assert!((f.se[j] - REF_SE[j]).abs() < 1e-6, "se[{j}]");
        }
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

        let csv = include_str!("../parity/data_simulated/sim_collinear.csv");
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

    #[derive(serde::Deserialize)]
    struct RdLmmEst {
        beta: Vec<f64>,
        varcomp: Vec<VcBlock>,
    }
    #[derive(serde::Deserialize)]
    struct RdLmmGolden {
        coef_names: Vec<String>,
        estimates: RdLmmEst,
    }

    /// Coverage-gaps G5: aliased fixed column on a MIXED design.
    /// `y ~ 1 + x1 + x2 + x3 + (1|g)` on sim_collinear_lmm (x3 ≈ x1 + x2) vs
    /// frozen lme4 (`parity/goldens/sim_collinear_lmm.json`): lmer's rankMatrix
    /// check drops the aliased column and `fixef` simply omits its name, so the
    /// golden's `coef_names` records WHICH column lme4 dropped (x3, the last
    /// dependent one — the same later-column convention `detect_aliased` uses,
    /// so the drop indices are asserted equal, not merely each self-consistent).
    /// glmm instead keeps full width with `NaN` in the dropped slot(s); the
    /// surviving β and the varcomp must match the reduced lme4 fit. The oracle
    /// is sacred.
    #[test]
    fn fit_lmm_rank_deficient_matches_lme4_drop() {
        let raw = include_str!("../parity/goldens/sim_collinear_lmm.json");
        let gold: RdLmmGolden = serde_json::from_str(raw).expect("golden JSON parses");

        // sim_collinear_lmm.csv: y,x1,x2,x3,g
        let csv = include_str!("../parity/data_simulated/sim_collinear_lmm.csv");
        let mut y = Vec::<f64>::new();
        let mut cols: Vec<[f64; 3]> = Vec::new();
        let mut g_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            cols.push([
                f[1].parse().unwrap(),
                f[2].parse().unwrap(),
                f[3].parse().unwrap(),
            ]);
            g_raw.push(f[4].to_string());
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
        let (g, _n_g) = dense_str(&g_raw);
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
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
                primary: g,
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1, 2, 3],
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "reduced LMM must converge");
        // lme4 dropped exactly x3: its surviving coef_names lack it. glmm's
        // aliased mask must mark the SAME column (index 3) and only it.
        assert!(
            !gold.coef_names.iter().any(|c| c == "x3") && gold.coef_names.len() == 3,
            "golden must record lme4 dropping x3, got {:?}",
            gold.coef_names
        );
        assert_eq!(
            f.aliased,
            vec![false, false, false, true],
            "glmm must drop the same column lme4 does (x3)"
        );
        assert!(f.beta[3].is_nan(), "aliased β = NaN");
        assert!(f.se[3].is_nan(), "aliased se = NaN");
        // Surviving slots [0..3) line up 1:1 with the golden's 3 reduced coefs.
        for (j, rb) in gold.estimates.beta.iter().enumerate() {
            assert!(
                (f.beta[j] - rb).abs() / rb.abs() < 1e-3,
                "β{j} {} vs lme4 {rb}",
                f.beta[j]
            );
        }
        // Varcomp of the reduced fit passes through the salvage unchanged.
        let ref_g_sd = gold.estimates.varcomp[0].stddev[0];
        let g_rel = (f.tau2[0].sqrt() - ref_g_sd).abs() / ref_g_sd;
        assert!(
            g_rel < 1e-2,
            "g sd = {} vs lme4 {ref_g_sd}",
            f.tau2[0].sqrt()
        );
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
            n_eval: 0,
            deviance: f64::NAN,
            singular: false,
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
        #[allow(clippy::needless_range_loop)]
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

    /// Warm-start A/B on the realistic sleepstudy random-slope LMM
    /// (`Reaction ~ Days + (1 + Days | Subject)`, q=2, n_theta=3): a warm fit
    /// from the frozen lme4 θ̂ ("from the truth") and one from a well-off
    /// perturbed θ must land on the cold optimum — β, SE, and the varcorr
    /// stddevs — and warm must never degrade convergence status. Extends
    /// `fit_warm_start_reaches_cold_beta` (β-only, hand-built n_theta=1) to a
    /// realistic q≥2 rung; MCPower's hot loop rides this contract.
    #[test]
    fn fit_warm_sleepstudy_slope_matches_cold_optimum() {
        // Parsing mirrors `fit_sleepstudy_slope_varcorr_matches_lme4`.
        let csv = include_str!("../parity/data_empirical/sleepstudy.csv");
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
            x[i * p] = 1.0;
            x[i * p + 1] = days[i];
        }
        let (subject, _n_subj) = dense_str(&subj_raw);
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — data path derives it
                slopes: vec![1],
                extra_groupings: vec![],
            }),
        };
        let ids = GroupIds {
            primary: subject,
            extra: vec![],
        };
        let opts = FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        };
        let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(cold.converged, "cold sleepstudy fit must converge");

        // lme4 θ̂ = vech Cholesky of D̂/σ̂², from the frozen golden's
        // stddev/corr/sigma (`parity/goldens/sleepstudy_lmm.json`) — `Fit`
        // does not expose θ̂, and Gaussian tau2/varcorr are both σ²-scaled so
        // θ cannot be recovered from the cold fit alone:
        // θ00 = sd0/σ, θ10 = corr·sd1/σ, θ11 = (sd1/σ)·√(1−corr²).
        const REF_SD0: f64 = 24.7406579949841;
        const REF_SD1: f64 = 5.92213765889808;
        const REF_CORR: f64 = 0.0655512382381282;
        const REF_SIGMA: f64 = 25.5917957216753;
        let truth = vec![
            REF_SD0 / REF_SIGMA,
            REF_CORR * REF_SD1 / REF_SIGMA,
            REF_SD1 / REF_SIGMA * (1.0 - REF_CORR * REF_CORR).sqrt(),
        ];
        let starts = [
            (
                "truth",
                StartValues {
                    beta: cold.beta.clone(),
                    theta: truth,
                },
            ),
            // Well off θ̂ ≈ [0.97, 0.015, 0.23] in every coordinate; the LMM
            // path threads θ only (β is solved exactly given θ).
            (
                "perturbed",
                StartValues {
                    beta: vec![0.0; p],
                    theta: vec![3.0, 0.5, 1.5],
                },
            ),
        ];
        for (label, start) in &starts {
            let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
            assert!(warm.converged, "{label}: warm must not degrade convergence");
            for j in 0..p {
                let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
                assert!(
                    rel < 1e-3,
                    "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                    warm.beta[j],
                    cold.beta[j]
                );
                let rel = (warm.se[j] - cold.se[j]).abs() / cold.se[j];
                assert!(
                    rel < 1e-3,
                    "{label}: se[{j}] warm {} vs cold {} (rel {rel})",
                    warm.se[j],
                    cold.se[j]
                );
            }
            // q=2 vech diag (offsets 0, 2) → the two RE stddevs. The off-diag
            // covariance is near zero here (corr≈0.066); the stddevs pin the block.
            for off in [0usize, 2] {
                let (w, c) = (warm.varcorr[0][off].sqrt(), cold.varcorr[0][off].sqrt());
                let rel = (w - c).abs() / c;
                assert!(
                    rel < 1e-3,
                    "{label}: RE stddev (vech {off}) warm {w} vs cold {c} (rel {rel})"
                );
            }
        }
    }

    /// Warm-start A/B on the realistic cbpp binomial GLMM (dense joint-BOBYQA
    /// path, scalar herd intercept): warm from the cold fit's own solution
    /// (θ̂ = √tau2 — σ²≡1 binomial — and β̂ verbatim) and from a perturbed
    /// (θ, β) must land on the cold optimum — β, SE, herd SD — and never
    /// degrade convergence. Unlike the LMM path, the GLMM start threads β
    /// verbatim (bypassing `glm_warm_start_beta`), so both arms also exercise
    /// PIRLS opening away from the GLM seed.
    #[test]
    fn fit_warm_glmm_cbpp_matches_cold_optimum() {
        let (x, y, cluster_ids, n) = cbpp_design();
        let p = 4;
        let model = cbpp_model();
        let ids = GroupIds {
            primary: cluster_ids,
            extra: vec![],
        };
        let opts = FitOptions {
            target_indices: vec![0, 1, 2, 3],
            ..FitOptions::default()
        };
        let cold = fit_cold(&x, &y, n, p, &model, &ids, &opts);
        assert!(cold.converged, "cold cbpp GLMM must converge");
        let starts = [
            (
                "truth",
                StartValues {
                    beta: cold.beta.clone(),
                    theta: vec![cold.tau2[0].sqrt()],
                },
            ),
            // Halved β̂ + θ=3 (θ̂ ≈ 0.64): far enough to move the joint
            // optimizer, near enough that PIRLS opens in a sane weight regime
            // from the verbatim β start.
            (
                "perturbed",
                StartValues {
                    beta: cold.beta.iter().map(|b| 0.5 * b).collect(),
                    theta: vec![3.0],
                },
            ),
        ];
        for (label, start) in &starts {
            let warm = fit_warm(&x, &y, n, p, &model, &ids, Some(start), &opts);
            assert!(warm.converged, "{label}: warm must not degrade convergence");
            for j in 0..p {
                let rel = (warm.beta[j] - cold.beta[j]).abs() / cold.beta[j].abs();
                assert!(
                    rel < 1e-3,
                    "{label}: β[{j}] warm {} vs cold {} (rel {rel})",
                    warm.beta[j],
                    cold.beta[j]
                );
                let rel = (warm.se[j] - cold.se[j]).abs() / cold.se[j];
                assert!(
                    rel < 1e-3,
                    "{label}: se[{j}] warm {} vs cold {} (rel {rel})",
                    warm.se[j],
                    cold.se[j]
                );
            }
            let (w, c) = (warm.tau2[0].sqrt(), cold.tau2[0].sqrt());
            let rel = (w - c).abs() / c;
            assert!(
                rel < 1e-3,
                "{label}: herd SD warm {w} vs cold {c} (rel {rel})"
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
        let csv = include_str!("../parity/data_empirical/cbpp.csv");
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
    /// (`parity/results/lme4_empirical/cbpp.json`). cbpp is
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
        // Frozen lme4 1.1-38 reference (parity/results/lme4_empirical/cbpp.json).
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

    /// cbpp AGGREGATED: 56 rows, y = incidence/size, weights = size. Mirrors
    /// `cbpp_design`'s parsing verbatim; only the Bernoulli expansion loop is
    /// replaced by one row per CSV record.
    fn cbpp_design_aggregated() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>, usize) {
        let csv = include_str!("../parity/data_empirical/cbpp.csv");
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        let mut w = Vec::<f64>::new();
        let mut cluster_ids = Vec::<u32>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let herd: u32 = f[0].parse::<u32>().unwrap() - 1; // herds 1..15 → ids 0..14
            let incidence: u32 = f[1].parse().unwrap();
            let size: u32 = f[2].parse().unwrap();
            let period: u32 = f[3].parse().unwrap();
            x.extend_from_slice(&[
                1.0,
                f64::from(u32::from(period == 2)),
                f64::from(u32::from(period == 3)),
                f64::from(u32::from(period == 4)),
            ]);
            y.push(f64::from(incidence) / f64::from(size));
            w.push(f64::from(size));
            cluster_ids.push(herd);
        }
        let n = y.len();
        (x, y, w, cluster_ids, n)
    }

    /// Aggregated cbpp through the DENSE (NoZ) path with prior weights must
    /// reproduce the same frozen lme4 oracle as the expanded fit — lme4 itself
    /// fits cbind(incidence, size−incidence), i.e. the aggregated objective.
    /// Matches lme4 1.1-38 (parity/results/lme4_empirical/cbpp.json freeze).
    #[test]
    fn fit_glmm_cbpp_aggregated_matches_lme4() {
        // Same frozen constants and tolerances as fit_glmm_cbpp_matches_lme4
        // (2e-3 abs β, 3e-2 rel SE, 3e-3 rel herd SD).
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

        let (x, y, w, cluster_ids, n) = cbpp_design_aggregated();
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
                weights: Some(w),
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "aggregated cbpp GLMM must converge");
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() < 2e-3,
                "β[{j}] = {} vs lme4 {} (Δ {})",
                f.beta[j],
                REF_BETA[j],
                (f.beta[j] - REF_BETA[j]).abs()
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(
                se_rel < 3e-2,
                "se[{j}] = {} vs lme4 {} (rel {se_rel})",
                f.se[j],
                REF_SE[j]
            );
        }
        let herd_sd = f.tau2[0].sqrt();
        let sd_rel = (herd_sd - REF_HERD_SD).abs() / REF_HERD_SD;
        assert!(
            sd_rel < 3e-3,
            "herd SD = {herd_sd} vs lme4 {REF_HERD_SD} (rel {sd_rel})"
        );
    }

    /// Prior-weight fit-level equivalence on the DENSE (NoZ) path: `fit_cold`
    /// on aggregated cbpp proportions with `weights = size` matches the
    /// expanded Bernoulli fit on β/SE/τ² for both `WaldSe` arms. Two
    /// independent BOBYQA runs of same-argmin objectives, so the bounds are
    /// optimizer-scatter-sized (the oracle test above is the tight anchor).
    /// Dense twin of `sparse_weighted_binomial_fit_matches_expanded`.
    #[test]
    fn fit_glmm_cbpp_aggregated_matches_expanded() {
        let (xe, ye, ids_e, n_e) = cbpp_design();
        let (xa, ya, wa, ids_a, n_a) = cbpp_design_aggregated();
        let p = 4;
        let model = cbpp_model();
        for wald_se in [WaldSe::Hessian, WaldSe::Rx] {
            let fe = fit_cold(
                &xe,
                &ye,
                n_e,
                p,
                &model,
                &GroupIds {
                    primary: ids_e.clone(),
                    extra: vec![],
                },
                &FitOptions {
                    target_indices: vec![0, 1, 2, 3],
                    wald_se,
                    ..FitOptions::default()
                },
            );
            let fa = fit_cold(
                &xa,
                &ya,
                n_a,
                p,
                &model,
                &GroupIds {
                    primary: ids_a.clone(),
                    extra: vec![],
                },
                &FitOptions {
                    target_indices: vec![0, 1, 2, 3],
                    wald_se,
                    weights: Some(wa.clone()),
                    ..FitOptions::default()
                },
            );
            let tag = format!("{wald_se:?}");
            assert!(
                fe.converged && fa.converged,
                "{tag}: both fits must converge"
            );
            for j in 0..p {
                assert!(
                    (fa.beta[j] - fe.beta[j]).abs() < 2e-3 * (1.0 + fe.beta[j].abs()),
                    "{tag} β[{j}]: agg={} exp={}",
                    fa.beta[j],
                    fe.beta[j]
                );
                assert!(
                    (fa.se[j] - fe.se[j]).abs() < 2e-2 * (1.0 + fe.se[j].abs()),
                    "{tag} se[{j}]: agg={} exp={}",
                    fa.se[j],
                    fe.se[j]
                );
            }
            assert_eq!(fa.tau2.len(), fe.tau2.len(), "{tag}: tau2 length");
            for (a, b) in fa.tau2.iter().zip(fe.tau2.iter()) {
                assert!(
                    (a - b).abs() < 2e-2 * (1.0 + b.abs()),
                    "{tag} tau2: agg={a} exp={b}"
                );
            }
        }
    }

    /// Shared 12-cluster × 10-row single-grouping design for the weighted
    /// dense-GLMM goldens: `(x [n·2 row-major: intercept, x1], ids, n, p)`
    /// assembled from the R-exported x1 slice; y and w are family-specific.
    fn weighted_glmm_design(x1: &[f64]) -> (Vec<f64>, Vec<u32>, usize, usize) {
        let n = x1.len();
        let mut x = Vec::with_capacity(n * 2);
        for &v in x1 {
            x.push(1.0);
            x.push(v);
        }
        let ids: Vec<u32> = (0..n as u32).map(|i| i / 10).collect();
        (x, ids, n, 2)
    }

    /// Weighted dense Poisson GLMM vs the frozen lme4 golden. Generated with
    /// (R 4.5.3, lme4 1.1-38):
    /// ```r
    /// library(lme4); set.seed(11)
    /// g <- rep(1:12, each = 10); n <- 120
    /// x1 <- round(rnorm(n), 4); w <- sample(1:4, n, TRUE)
    /// b <- rnorm(12, 0, 0.5)
    /// y <- rpois(n, exp(0.3 + 0.5 * x1 + b[g]))
    /// f <- glmer(y ~ x1 + (1 | g), family = poisson, weights = w)
    /// print(summary(f)$coefficients, digits = 15)
    /// print(as.data.frame(VarCorr(f)), digits = 15)
    /// ```
    /// Tolerances mirror `fit_glmm_cbpp_matches_lme4` (2e-3 abs β, 3e-2 rel SE,
    /// 3e-3 rel RE SD).
    #[test]
    fn fit_glmm_poisson_weighted_matches_lme4() {
        const X1: [f64; 120] = [
            -0.591, 0.0266, -1.5166, -1.3627, 1.1785, -0.9342, 1.3236, 0.6249, -0.0457, -1.0041,
            -0.8284, -0.3484, -1.5383, -0.2556, -1.1499, 0.0123, -0.223, 0.8878, -0.5922, -0.6557,
            -0.6825, -0.0159, -0.4426, 0.3526, 0.0732, 0.0072, -0.1876, -0.7657, -0.2211, -0.9836,
            -1.1043, -0.9382, 0.6786, -1.5775, -0.8699, 0.4847, -0.1861, 1.5456, -0.6114, -0.3478,
            -1.6365, 0.0204, 0.8917, -0.8727, 0.8901, -0.3439, -2.1868, 0.8801, 0.7239, 0.2199,
            0.7899, -0.23, -0.8185, 0.4997, 0.1592, 0.5426, -0.1566, 0.4388, 1.4879, 0.0602,
            -0.849, 2.3397, -0.1212, -1.9502, 0.5387, 1.6935, -0.791, -1.0753, -0.6079, 0.7544,
            0.4535, -0.1234, -0.7631, 0.2283, 1.1195, 0.1566, -0.6888, 0.4529, -1.0675, 0.4016,
            -0.0648, 0.3155, -0.6057, -0.9076, 2.2616, -0.6032, -1.2979, 0.5065, -0.8533, -1.506,
            1.2023, -1.0279, 0.9383, -0.5432, 0.5131, -0.3526, 1.3265, -1.1402, 1.4131, -0.6022,
            -0.4417, 0.2436, 0.5968, -0.12, -2.0697, 0.5856, 0.4894, -1.0066, 1.2697, 1.1239,
            0.8425, 1.6206, 0.4477, -2.2989, -0.0792, -0.5231, -0.4176, 0.3049, -0.0314, 0.1051,
        ];
        const W: [f64; 120] = [
            4., 1., 3., 2., 3., 2., 3., 1., 3., 1., 1., 1., 2., 4., 4., 4., 1., 1., 4., 3., 4., 4.,
            3., 4., 4., 1., 1., 1., 3., 4., 3., 3., 3., 2., 1., 3., 2., 2., 2., 3., 2., 1., 4., 1.,
            1., 1., 2., 1., 3., 4., 2., 4., 1., 1., 4., 2., 4., 1., 1., 3., 2., 1., 1., 3., 4., 3.,
            2., 3., 2., 1., 3., 4., 1., 4., 1., 3., 3., 1., 2., 4., 2., 4., 1., 2., 1., 4., 1., 4.,
            3., 4., 3., 2., 4., 2., 2., 4., 3., 3., 1., 3., 4., 1., 1., 3., 3., 4., 3., 1., 4., 3.,
            3., 4., 3., 1., 2., 4., 4., 1., 1., 2.,
        ];
        const Y: [f64; 120] = [
            2., 2., 0., 0., 3., 0., 8., 3., 3., 1., 0., 0., 1., 2., 2., 0., 2., 3., 3., 1., 0., 1.,
            3., 1., 3., 0., 2., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 3., 1., 0., 1., 6., 5., 3.,
            10., 6., 1., 14., 4., 3., 3., 0., 0., 1., 1., 0., 1., 1., 3., 1., 1., 4., 0., 0., 0.,
            1., 1., 0., 1., 0., 2., 3., 0., 1., 1., 3., 2., 2., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
            3., 0., 1., 5., 1., 1., 1., 3., 1., 5., 0., 4., 2., 3., 1., 3., 0., 2., 3., 0., 1., 2.,
            4., 2., 2., 0., 1., 0., 2., 0., 1., 1., 0.,
        ];
        const REF_BETA: [f64; 2] = [0.235954720439220, 0.547941515043755];
        const REF_SE: [f64; 2] = [0.1756494873870279, 0.0594199356963711];
        const REF_G_SD: f64 = 0.575359686811311;

        let (x, ids, n, p) = weighted_glmm_design(&X1);
        let model = ModelSpec {
            family: Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 12 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let f = fit_cold(
            &x,
            &Y,
            n,
            p,
            &model,
            &GroupIds {
                primary: ids,
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1],
                weights: Some(W.to_vec()),
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "weighted Poisson GLMM must converge");
        for j in 0..p {
            assert!(
                (f.beta[j] - REF_BETA[j]).abs() < 2e-3,
                "β[{j}] = {} vs lme4 {} (Δ {})",
                f.beta[j],
                REF_BETA[j],
                (f.beta[j] - REF_BETA[j]).abs()
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(
                se_rel < 3e-2,
                "se[{j}] = {} vs lme4 {} (rel {se_rel})",
                f.se[j],
                REF_SE[j]
            );
        }
        let g_sd = f.tau2[0].sqrt();
        let sd_rel = (g_sd - REF_G_SD).abs() / REF_G_SD;
        assert!(
            sd_rel < 3e-3,
            "g SD = {g_sd} vs lme4 {REF_G_SD} (rel {sd_rel})"
        );
    }

    /// Weighted dense Gamma GLMM vs the frozen lme4 golden — this is what pins
    /// the weighted `gamma_aic` (profiled dispersion over Σwᵢ), the weighted
    /// `glmm_sigma_sq` (σ̂² = pwrss/n with wᵢrᵢ², raw-n denominator: lme4's
    /// VarCorr vcov below only reproduces under raw n), and the weighted
    /// Pearson dispersion. Generated with (R 4.5.3, lme4 1.1-38):
    /// ```r
    /// library(lme4); set.seed(21)
    /// g <- rep(1:12, each = 10); n <- 120
    /// x1 <- round(rnorm(n), 4); w <- sample(1:4, n, TRUE)
    /// b <- rnorm(12, 0, 0.4)
    /// mu <- exp(0.8 + 0.4 * x1 + b[g])
    /// y <- round(rgamma(n, shape = 3, scale = mu / 3), 6)
    /// f <- glmer(y ~ x1 + (1 | g), family = Gamma("log"), weights = w)
    /// print(summary(f)$coefficients, digits = 15)
    /// print(as.data.frame(VarCorr(f)), digits = 15); print(sigma(f)^2, digits = 15)
    /// ```
    /// τ² is compared on lme4's VarCorr vcov scale (σ̂²·θ̂²). SE tolerance is the
    /// cbpp 3e-2; β mirrors `fit_glmm_gamma_sim_matches_lme4`'s relative gate.
    #[test]
    fn fit_glmm_gamma_weighted_matches_lme4() {
        // R-generated covariate data; 1.1283 coincidentally approximates 2/√π.
        #[allow(clippy::approx_constant)]
        const X1: [f64; 120] = [
            0.793, 0.5223, 1.7462, -1.2713, 2.1974, 0.4331, -1.5702, -0.9349, 0.0635, -0.0024,
            -2.2768, 0.7574, -0.5484, 0.1725, 0.5629, 1.5118, 0.659, 1.122, -0.7846, -0.4257,
            0.393, 0.0368, -1.0321, -1.2649, -0.227, 0.7456, 0.3328, -1.124, -0.7061, -0.7275,
            -1.8343, -0.4077, 0.0269, 0.9116, 1.6343, 0.0607, 1.8476, 0.0801, 1.4186, 1.4586,
            0.0559, -1.5172, -0.0486, -0.2144, 2.0958, 0.2023, 0.5177, 1.6781, 0.3852, -1.2819,
            -0.5822, 1.7741, -0.2107, -0.3521, 0.5852, 1.0137, -0.0226, -0.9032, 0.9078, 1.1619,
            -0.458, 0.928, -2.1029, -1.6772, 1.7657, 0.7944, -0.4839, 1.9284, -0.3841, -1.5867,
            0.2143, -1.1383, 0.4894, -1.7526, 0.501, 0.0868, 0.1911, 0.8318, -0.679, 0.2959,
            1.1122, 0.3626, -0.2709, -0.1969, 0.067, -0.8678, -0.362, -1.1396, -0.8154, 1.3102,
            -0.2584, 0.6063, 0.3134, 0.0536, 1.1283, -0.5581, 1.536, -0.0624, 0.0216, -2.0898,
            -0.8109, -2.9438, -0.0188, -0.3547, 0.0356, 0.4941, -0.6598, 1.0011, 1.0721, 0.7558,
            -1.4555, 0.9429, -1.8703, -0.2533, -0.2926, 0.2188, -1.3551, -0.1227, -0.4519, 0.0972,
        ];
        const W: [f64; 120] = [
            2., 2., 2., 1., 2., 4., 2., 3., 3., 3., 4., 4., 4., 4., 2., 4., 3., 3., 2., 2., 1., 1.,
            4., 1., 1., 1., 4., 4., 3., 4., 3., 2., 4., 3., 4., 2., 4., 2., 2., 2., 1., 1., 1., 1.,
            1., 3., 2., 1., 2., 2., 4., 4., 2., 3., 4., 4., 4., 3., 2., 4., 4., 2., 3., 4., 2., 4.,
            2., 2., 2., 1., 1., 1., 4., 4., 4., 1., 3., 4., 4., 3., 2., 1., 1., 4., 4., 4., 1., 2.,
            2., 2., 4., 3., 3., 1., 1., 1., 4., 3., 4., 3., 3., 2., 2., 3., 4., 4., 3., 4., 2., 1.,
            3., 1., 3., 2., 3., 3., 4., 3., 1., 3.,
        ];
        const Y: [f64; 120] = [
            1.027885, 3.568778, 5.059958, 1.829256, 7.572745, 1.888244, 0.638556, 1.352118,
            6.460123, 1.431433, 0.491063, 1.808875, 1.736458, 2.965294, 4.171528, 2.554423,
            2.217066, 0.48551, 1.646985, 3.758326, 3.388564, 2.795867, 0.780591, 1.495213,
            1.664063, 3.445218, 2.973526, 1.700702, 1.031139, 1.852452, 2.514445, 1.04869,
            1.757371, 2.407751, 1.232387, 1.211173, 7.507012, 3.516693, 3.209465, 1.575613,
            1.416005, 0.324474, 1.528727, 1.941835, 9.305071, 0.960217, 1.934011, 1.54724,
            1.326433, 1.255908, 2.665283, 4.779793, 1.830826, 0.990174, 1.892684, 11.248398,
            1.851022, 1.273189, 3.905656, 0.905928, 3.315271, 1.126161, 0.465568, 1.937359,
            4.986676, 5.506185, 0.636041, 5.615351, 0.473084, 0.831148, 1.471093, 2.344402,
            0.680976, 1.026012, 1.43575, 2.919631, 5.756904, 4.804391, 1.699487, 0.706556,
            3.551593, 2.787834, 2.280541, 1.685016, 3.503679, 3.911159, 0.424846, 3.080594,
            0.663857, 4.361308, 3.329871, 3.137527, 7.377112, 2.457973, 4.633516, 3.899755,
            5.727707, 1.813578, 2.754815, 1.84022, 0.753663, 0.331312, 0.870051, 2.412794,
            3.001372, 1.099695, 4.98129, 4.075331, 4.525327, 5.201431, 1.504496, 5.951359,
            1.258666, 5.439477, 2.243875, 0.603161, 1.000063, 2.337211, 0.981631, 0.914213,
        ];
        const REF_BETA: [f64; 2] = [0.863125471935252, 0.372178348714047];
        const REF_SE: [f64; 2] = [0.0654914008493939, 0.0333300007320761];
        const REF_G_VCOV: f64 = 0.0510221486396947; // σ̂²·θ̂² (lme4 VarCorr vcov)

        let (x, ids, n, p) = weighted_glmm_design(&X1);
        let model = ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 12 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let f = fit_cold(
            &x,
            &Y,
            n,
            p,
            &model,
            &GroupIds {
                primary: ids,
                extra: vec![],
            },
            &FitOptions {
                target_indices: vec![0, 1],
                weights: Some(W.to_vec()),
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "weighted Gamma GLMM must converge");
        for j in 0..p {
            let b_rel = (f.beta[j] - REF_BETA[j]).abs() / REF_BETA[j].abs();
            assert!(
                b_rel < 2e-3,
                "β[{j}] = {} vs lme4 {} (rel {b_rel})",
                f.beta[j],
                REF_BETA[j]
            );
            let se_rel = (f.se[j] - REF_SE[j]).abs() / REF_SE[j];
            assert!(
                se_rel < 3e-2,
                "se[{j}] = {} vs lme4 {} (rel {se_rel})",
                f.se[j],
                REF_SE[j]
            );
        }
        // varcorr[0][0] = σ̂²·θ̂² — directly lme4's VarCorr vcov for the g
        // intercept, via the public block (σ̂²-scaled like tau2, B1 fix).
        let vc_rel = (f.varcorr[0][0] - REF_G_VCOV).abs() / REF_G_VCOV;
        assert!(
            vc_rel < 1e-2,
            "g vcov = {} vs lme4 {REF_G_VCOV} (rel {vc_rel})",
            f.varcorr[0][0]
        );
        assert!(
            (f.varcorr[0][0] - f.tau2[0]).abs() < 1e-12,
            "varcorr and tau2 must report the same σ̂²-scaled variance"
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
        let csv = include_str!("../parity/data_empirical/grouseticks.csv");
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

    /// High-mean Poisson GLM through stable `fit` (re: None), gated against the
    /// frozen R `glm(family=poisson)` oracle
    /// (`parity/goldens/sim_poisson_highmean_glm.json`): `y ~ 1 + x + grp` on
    /// sim_poisson_highmean (ȳ ≈ 85). Regression gate for the IRLS log-link cold
    /// start: from the old μ = 1 seed (η = 0) any count data with ȳ ≳ ~25–30 made
    /// the first WLS step overshoot and IRLS run away (β → ~9e304,
    /// `converged = false`); the μ₀ = y + 0.1 seed (R's family `initialize`)
    /// converges here. The oracle is sacred (parity §1).
    #[test]
    fn fit_glm_poisson_highmean_matches_r() {
        const REF_BETA: [f64; 3] = [4.27614930354405, 0.299823553158498, 0.220101964251659];
        const REF_SE: [f64; 3] = [
            0.00955233157557028,
            0.00587180175968696,
            0.0125653819843501,
        ];
        // sim_poisson_highmean.csv cols: x,grp,y — grp ∈ {a,b}, base level a.
        let csv = include_str!("../parity/data_simulated/sim_poisson_highmean.csv");
        let p = 3; // [intercept, x, grpb]
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let xv: f64 = f[0].parse().unwrap();
            let grp_b = f[1] == "b";
            let yv: f64 = f[2].parse().unwrap();
            x.extend_from_slice(&[1.0, xv, f64::from(u32::from(grp_b))]);
            y.push(yv);
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
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "high-mean poisson GLM must converge");
        assert!((f.dispersion - 1.0).abs() < 1e-12, "poisson φ≡1");
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
        let csv = include_str!("../parity/data_empirical/cbpp.csv");
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
        let csv = include_str!("../parity/data_simulated/sim_gamma.csv");
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
        let csv = include_str!("../parity/data_simulated/sim_nb.csv");
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

    /// Weighted negative-binomial GLM vs `MASS::glm.nb(weights=)`. Convention:
    /// prior weight multiplies both the IRLS working weight (β/SE, per Task 2)
    /// and the per-row θ profile term (`nb_profile_loglik`'s `weights` — matches
    /// `theta.ml`'s weighted profile, the outer loop MASS::glm.nb alternates on).
    #[test]
    #[expect(
        clippy::approx_constant,
        reason = "R-generated x1 datum 0.3183, not a use of the std FRAC_1_PI constant"
    )]
    fn fit_glm_nb_weighted_matches_mass() {
        // R 4.5.3 oracle:
        //   library(MASS); set.seed(7); n <- 60
        //   x1 <- round(rnorm(n), 4); w <- sample(1:3, n, TRUE)
        //   mu <- exp(0.5 + 0.6 * x1); y <- rnbinom(n, size = 1.8, mu = mu)
        //   f <- glm.nb(y ~ x1, weights = w)
        //   print(coef(summary(f)), digits = 15); print(f$theta, digits = 15)
        let x1: [f64; 60] = [
            2.2872, -1.1968, -0.6943, -0.4123, -0.9707, -0.9473, 0.7481, -0.117, 0.1527, 2.19,
            0.357, 2.7168, 2.2815, 0.324, 1.8961, 0.4677, -0.8938, -0.3073, -0.0048, 0.9882,
            0.8398, 0.7053, 1.306, -1.388, 1.2729, 0.1842, 0.7523, 0.5917, -0.9831, -0.2761,
            -0.8709, 0.7187, 0.1107, -0.0785, -0.4205, -0.5621, 0.9975, -1.1051, -0.1423, 0.315,
            1.2186, -0.6993, -0.2854, -1.3116, -0.391, -0.4015, 1.3505, 0.5912, 0.1005, 0.9311,
            -0.2627, -0.0077, 0.3672, 1.7072, 0.7237, 0.481, -1.5679, 0.3183, 0.166, -0.8999,
        ];
        let w: [f64; 60] = [
            3.0, 2.0, 1.0, 2.0, 1.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0,
            1.0, 3.0, 2.0, 1.0, 3.0, 3.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 2.0,
            3.0, 3.0, 1.0, 3.0, 2.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 3.0, 1.0, 3.0, 2.0,
            3.0, 2.0, 1.0, 3.0, 2.0, 2.0, 2.0, 2.0, 1.0,
        ];
        let y: [f64; 60] = [
            7.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 7.0, 0.0, 3.0, 4.0, 15.0, 0.0, 12.0, 0.0, 1.0,
            0.0, 3.0, 2.0, 0.0, 4.0, 0.0, 1.0, 3.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 2.0,
            5.0, 1.0, 7.0, 1.0, 3.0, 3.0, 0.0, 2.0, 1.0, 0.0, 3.0, 0.0, 3.0, 3.0, 0.0, 0.0, 0.0,
            0.0, 4.0, 2.0, 1.0, 0.0, 0.0, 7.0, 1.0, 2.0,
        ];
        const REF_BETA: [f64; 2] = [0.448681810160982, 0.593940842956464];
        const REF_SE: [f64; 2] = [0.119405783091442, 0.112801176259142];
        const REF_THETA: f64 = 1.23453054082489;
        let n = 60;
        let p = 2;
        let mut x = Vec::with_capacity(n * p);
        for &xi in &x1 {
            x.extend_from_slice(&[1.0, xi]);
        }
        let model = ModelSpec {
            family: Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            re: None,
        };
        let opts = FitOptions {
            target_indices: vec![0, 1],
            weights: Some(w.to_vec()),
            ..FitOptions::default()
        };
        let f = fit_cold(&x, &y, n, p, &model, &GroupIds::default(), &opts);
        assert!(f.converged, "weighted NB GLM must converge");
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
        let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
        assert!(
            th_rel < 1e-4,
            "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
            f.dispersion
        );
    }

    /// Parse an `x,grp,y` NB-edge sim CSV → (X=[1,x,grp_b], y, n).
    fn nb_edge_data(csv: &str) -> (Vec<f64>, Vec<f64>, usize) {
        let mut x = Vec::<f64>::new();
        let mut y = Vec::<f64>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            x.extend_from_slice(&[
                1.0,
                f[0].parse().unwrap(),
                f64::from(u32::from(f[1] == "b")),
            ]);
            y.push(f[2].parse().unwrap());
        }
        let n = y.len();
        (x, y, n)
    }

    /// Fit the NB GLM on an edge dataset and gate against the frozen MASS
    /// reference (β rel 1e-3, SE rel 3e-2 — the `fit_glm_nb_matches_mass`
    /// bands). Returns the fit so the caller can pin its edge-specific θ̂
    /// assertions. Shared by the two θ-bracket-edge tests.
    fn nb_edge_fit(csv: &str, ref_beta: &[f64; 3], ref_se: &[f64; 3]) -> Fit {
        let (x, y, n) = nb_edge_data(csv);
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
            3,
            &model,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1, 2],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "NB edge GLM must converge");
        for j in 0..3 {
            assert!(
                (f.beta[j] - ref_beta[j]).abs() / ref_beta[j].abs() < 1e-3,
                "β[{j}] = {} vs MASS {}",
                f.beta[j],
                ref_beta[j]
            );
            let se_rel = (f.se[j] - ref_se[j]).abs() / ref_se[j];
            assert!(se_rel < 3e-2, "se[{j}] = {} vs MASS {}", f.se[j], ref_se[j]);
        }
        f
    }

    /// θ-bracket LOW edge: heavily overdispersed NB GLM (θ̂ ≈ 4.1e-3, half an
    /// order above `NB_THETA_LO` = 1e-3), gated against frozen `MASS::glm.nb`
    /// (`parity/goldens/sim_nb_lowtheta_glm.json`; glm.nb converged with zero
    /// warnings on the committed CSV — the reference is trustworthy this close
    /// to the edge, not past it). Pins that the golden-section θ search stays
    /// interior and matches MASS near its lower bracket end. The oracle is
    /// sacred.
    #[test]
    fn fit_glm_nb_theta_low_edge_matches_mass() {
        const REF_BETA: [f64; 3] = [0.392948589321679, -1.19642377752834, 0.820910978622294];
        const REF_SE: [f64; 3] = [1.12744374740952, 0.781254798362756, 1.57118324159906];
        const REF_THETA: f64 = 0.00409762150621296;
        let f = nb_edge_fit(
            include_str!("../parity/data_simulated/sim_nb_lowtheta.csv"),
            &REF_BETA,
            &REF_SE,
        );
        // Sane boundary behavior: inside the bracket, near (but not AT) the low end.
        assert!(
            f.dispersion > super::NB_THETA_LO && f.dispersion < 1e-2,
            "θ̂ = {} must sit interior near NB_THETA_LO",
            f.dispersion
        );
        let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
        assert!(
            th_rel < 2e-2,
            "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
            f.dispersion
        );
    }

    /// θ-bracket HIGH edge: near-Poisson NB GLM (θ̂ ≈ 5.3e2, pushed toward
    /// `NB_THETA_HI` = 1e4), gated against frozen `MASS::glm.nb`
    /// (`parity/goldens/sim_nb_hightheta_glm.json`; zero glm.nb warnings on the
    /// committed CSV — cells with θ̂ nearer the edge all put `theta.ml` at its
    /// iteration/alternation limits, and count size is separately capped by the
    /// IRLS cold-start divergence; both constraints are documented at the
    /// generator, `prep/export_data.R`). The profile is nearly flat in θ up
    /// here, yet both engines maximise the same profile on the same data, so
    /// θ̂ still gates at 1e-2 (measured ~2e-9); β/SE stay at the standard bands
    /// (β is θ-insensitive near the Poisson limit). The oracle is sacred.
    #[test]
    fn fit_glm_nb_theta_high_edge_matches_mass() {
        const REF_BETA: [f64; 3] = [2.00540691601978, 0.596522354278958, 0.385922258588444];
        const REF_SE: [f64; 3] = [0.00809529402417648, 0.00479352480867935, 0.00985794518911199];
        const REF_THETA: f64 = 534.632483746729;
        let f = nb_edge_fit(
            include_str!("../parity/data_simulated/sim_nb_hightheta.csv"),
            &REF_BETA,
            &REF_SE,
        );
        // Sane boundary behavior: large but interior (not clamped at NB_THETA_HI).
        assert!(
            f.dispersion > 1e2 && f.dispersion < super::NB_THETA_HI,
            "θ̂ = {} must sit interior, pushed toward NB_THETA_HI",
            f.dispersion
        );
        let th_rel = (f.dispersion - REF_THETA).abs() / REF_THETA;
        assert!(
            th_rel < 1e-2,
            "θ̂ = {} vs MASS {REF_THETA} (rel {th_rel})",
            f.dispersion
        );
    }

    /// `NB_MAX_OUTER` cap semantics via the `fit_glm_nb_capped` seam, seeded at
    /// `NB_THETA_LO` (far from θ̂ ≈ 1.01 on sim_nb) so one alternation cannot
    /// meet `NB_THETA_TOL`. Pins that cap exhaustion is SILENT: the capped fit
    /// reports `converged = true` (the flag reflects only the last inner IRLS
    /// fit, not the θ alternation), β/se stay at the stale pre-update θ, and
    /// `dispersion` carries the newer θ. `max_outer = 0` is the degenerate
    /// never-ran case: the all-NaN `converged = false` placeholder.
    #[test]
    fn fit_glm_nb_outer_cap_semantics() {
        // Fixed-only fit; sim_clustered's cluster ids are unused here.
        let (x, y, _ids, _nc) =
            sim_clustered(include_str!("../parity/data_simulated/sim_nb.csv"));
        let (n, p) = (y.len(), 3);
        let opts = FitOptions {
            target_indices: vec![0, 1, 2],
            ..FitOptions::default()
        };
        let seed = Some(super::NB_THETA_LO);

        let f0 = super::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, 0);
        assert!(!f0.converged, "cap 0: never-ran placeholder is converged=false");
        assert!(f0.beta.iter().all(|b| b.is_nan()), "cap 0: β all NaN");

        let f1 = super::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, 1);
        let full = super::fit_glm_nb_capped(&x, &y, n, p, seed, &opts, super::NB_MAX_OUTER);
        // Cap exhaustion is silent: the inner IRLS converged, so the flag is true
        // even though the θ alternation was cut off mid-flight.
        assert!(f1.converged, "capped fit reports the INNER convergence flag");
        assert!(full.converged);
        // The single alternation moved θ off the seed (the profile step ran) …
        assert!(
            (f1.dispersion - super::NB_THETA_LO).abs() / super::NB_THETA_LO > 1.0,
            "θ after one alternation ({}) must leave the seed",
            f1.dispersion
        );
        // … but β/se were fit at the stale seed θ = 1e-3, whose NB variance
        // V = μ + μ²/θ is ~10³ wider than the converged fit's — the capped SE
        // must visibly disagree with the fully-alternated one.
        assert!(
            (f1.se[0] - full.se[0]).abs() / full.se[0] > 0.5,
            "capped se[0] = {} vs full {} must reflect the stale θ",
            f1.se[0],
            full.se[0]
        );
        // Sanity: the uncapped path from the same seed reaches the MASS optimum
        // (`fit_glm_nb_matches_mass`'s reference θ̂).
        assert!(
            (full.dispersion - 1.01052181546876).abs() / 1.01052181546876 < 2e-2,
            "full θ̂ = {} vs MASS 1.0105",
            full.dispersion
        );
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

        let csv = include_str!("../parity/data_empirical/sleepstudy.csv");
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

    /// Campaign instrumentation: `fit` must surface the optimizer eval count, the
    /// minimized criterion, and boundary/singular status. Oracle: lme4's frozen
    /// sleepstudy REML fit — REMLcrit = glmm deviance + df·(1 + ln 2π), df = n − p
    /// (glmm's reml_deviance omits the df·(1+ln 2π) constant lme4's REMLcrit
    /// carries; loglik = −REMLcrit/2 is what results/lme4_empirical stores).
    #[test]
    fn fit_exposes_n_eval_deviance_singular() {
        let csv = include_str!("../parity/data_empirical/sleepstudy.csv");
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

        assert!(f.n_eval > 0, "BOBYQA ran, evals must be counted");
        assert!(f.deviance.is_finite());
        assert!(!f.singular, "sleepstudy is an interior optimum");
        let n = 180.0_f64;
        let p = 2.0_f64; // intercept + Days
        let df = n - p;
        let lme4_loglik = -871.814135979976; // parity/results/lme4_empirical/sleepstudy.json .estimates.loglik
        let remlcrit = -2.0 * lme4_loglik;
        let expected = remlcrit - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
        assert!(
            (f.deviance - expected).abs() < 1e-6,
            "deviance {} vs lme4-derived {expected}",
            f.deviance
        );
    }

    /// Task 5: weighted dense LMM REML — sleepstudy random-slope fit with
    /// synthetic weights `w_i = 1 + (i mod 3)` (i = 0-based CSV row order),
    /// gated against a frozen lme4 golden. Pins β, SE, the 2×2 RE covariance
    /// (SDs + correlation), σ̂, and the `−Σlog wᵢ` deviance-constant convention:
    /// weighted REMLcrit strips the same `df·(1+ln 2π)` constant as the
    /// unweighted case (`lme.rs:2978`) PLUS the weighted Gaussian log-density's
    /// `+½Σlog wᵢ` per row (`−Σlog wᵢ` on the −2ℓ deviance scale). Generated
    /// with (R 4.5.3, lme4 1.1-38):
    /// ```r
    /// library(lme4)
    /// d <- read.csv("parity/data_empirical/sleepstudy.csv")
    /// w <- 1 + (seq_len(nrow(d)) - 1) %% 3
    /// f <- lmer(Reaction ~ Days + (Days | Subject), data = d, weights = w, REML = TRUE)
    /// print(summary(f)$coefficients, digits = 15)
    /// print(as.data.frame(VarCorr(f)), digits = 15)
    /// print(sigma(f), digits = 15); print(REMLcrit(f), digits = 15)
    /// ```
    #[test]
    fn fit_lmm_weighted_matches_lme4() {
        const REF_B0: f64 = 251.804_690_405_274;
        const REF_B1: f64 = 10.4358707468765;
        const REF_SE0: f64 = 6.44698545564581;
        const REF_SE1: f64 = 1.57363056312657;
        const REF_SD0: f64 = 22.09852363841438; // (Intercept) sd
        const REF_SD1: f64 = 5.95218759898762; // Days sd
        const REF_CORR: f64 = 0.16395038320169;
        const REF_SIGMA: f64 = 38.62892535113247;
        const REF_REMLCRIT: f64 = 1778.29146275691;

        let csv = include_str!("../parity/data_empirical/sleepstudy.csv");
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
        let w: Vec<f64> = (0..n).map(|i| 1.0 + (i % 3) as f64).collect();

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
                weights: Some(w.clone()),
                ..FitOptions::default()
            },
        );

        assert!(f.converged, "weighted sleepstudy slope LMM must converge");
        assert!(
            (f.beta[0] - REF_B0).abs() / REF_B0 < 1e-6,
            "β0 {} vs {REF_B0}",
            f.beta[0]
        );
        assert!(
            (f.beta[1] - REF_B1).abs() / REF_B1 < 1e-6,
            "β1 {} vs {REF_B1}",
            f.beta[1]
        );
        assert!(
            (f.se[0] - REF_SE0).abs() / REF_SE0 < 1e-4,
            "se0 {} vs {REF_SE0}",
            f.se[0]
        );
        assert!(
            (f.se[1] - REF_SE1).abs() / REF_SE1 < 1e-4,
            "se1 {} vs {REF_SE1}",
            f.se[1]
        );

        assert_eq!(f.varcorr.len(), 1, "one grouping block");
        let vc = &f.varcorr[0];
        assert_eq!(vc.len(), 3, "q=2 vech has 3 entries");
        let sd0 = vc[0].sqrt();
        let sd1 = vc[2].sqrt();
        let corr = vc[1] / (sd0 * sd1);
        assert!(
            (sd0 - REF_SD0).abs() / REF_SD0 < 1e-4,
            "sd0 {sd0} vs {REF_SD0}"
        );
        assert!(
            (sd1 - REF_SD1).abs() / REF_SD1 < 1e-4,
            "sd1 {sd1} vs {REF_SD1}"
        );
        // The off-diagonal covariance/correlation is the least-constrained θ
        // coordinate under BOBYQA's rho_end floor (θ10 is small relative to
        // θ00/θ11, so its relative precision is looser) — the unweighted
        // analog (`fit_sleepstudy_slope_varcorr_matches_lme4`) hits the exact
        // same floor and uses the same absolute-on-D10-scale band.
        assert!((corr - REF_CORR).abs() < 0.05, "corr {corr} vs {REF_CORR}");

        // Fit.deviance vs REMLcrit(f) − (n−p)·(1+ln 2π) — pins the −Σlog wᵢ
        // constant `fit_mle` folds into the reported deviance (see the arm
        // above fit_mle in this file). 1e-6 abs, as the unweighted analog above.
        let df = (n - p) as f64;
        let expected = REF_REMLCRIT - df * (1.0 + (2.0 * std::f64::consts::PI).ln());
        assert!(
            (f.deviance - expected).abs() < 1e-6,
            "deviance {} vs lme4-derived {expected}",
            f.deviance
        );

        // σ̂ isn't exposed on `Fit` for q≥2 RE (tau2 only reproduces the (0,0)
        // diagonal, not the raw residual variance) — reconstruct via the same
        // suff-stats accumulator/kernel `fit_mle` calls, reading `sigma_sq`
        // straight off `LmmFit` (mirrors fit_mle's construction verbatim).
        let sized = spec_sized_from_ids(&model, &ids);
        let mut ws = LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &[1], &[]);
        let mut x_mat = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            for j in 0..p {
                x_mat[(i, j)] = x[i * p + j];
            }
        }
        ws.suff
            .add_rows_multi(x_mat.as_ref(), &y, &ids.primary, &[], Some(&w));
        let lmm_fit = fit_lmm(&mut ws, &[0, 1], None);
        let sigma = lmm_fit.sigma_sq.sqrt();
        assert!(
            (sigma - REF_SIGMA).abs() / REF_SIGMA < 1e-4,
            "sigma {sigma} vs {REF_SIGMA}"
        );
    }

    /// Task 5: constant weights (w ≡ 2) must reproduce the unweighted fit's β,
    /// SE, AND tau2 exactly (1e-10) — under w ≡ c, the substitution θ̃ = √c·θ
    /// maps the weighted profiled deviance onto the unweighted one 1:1, so θ̂
    /// scales by 1/√c while σ̂² scales by c, and tau2 = θ²σ̂² is invariant.
    /// Verified against lme4 separately: sleepstudy with w ≡ 2 leaves the
    /// VarCorr group variances unchanged and exactly doubles the residual
    /// variance (not re-asserted here — this test only needs internal
    /// consistency on a small synthetic LMM, cheaper than another R golden).
    #[test]
    fn fit_lmm_constant_weights_invariant() {
        let n_clusters = 6usize;
        let per = 8usize;
        let n = n_clusters * per;
        let mut st = 13u64;
        let mut x = vec![0.0f64; n * 2];
        let mut y = vec![0.0f64; n];
        let mut ids_v = vec![0u32; n];
        for i in 0..n {
            ids_v[i] = (i % n_clusters) as u32;
            let x1 = lcg(&mut st);
            x[i * 2] = 1.0;
            x[i * 2 + 1] = x1;
            let re = 0.3 * ((ids_v[i] as f64) - (n_clusters as f64) / 2.0);
            y[i] = 0.5 + 0.4 * x1 + re + 0.2 * lcg(&mut st);
        }
        let model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let ids = GroupIds {
            primary: ids_v,
            extra: vec![],
        };
        let unweighted = fit_cold(
            &x,
            &y,
            n,
            2,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0, 1],
                ..FitOptions::default()
            },
        );
        let weighted = fit_cold(
            &x,
            &y,
            n,
            2,
            &model,
            &ids,
            &FitOptions {
                target_indices: vec![0, 1],
                weights: Some(vec![2.0; n]),
                ..FitOptions::default()
            },
        );
        assert!(unweighted.converged && weighted.converged);
        // The θ̃=√c·θ substitution is exact algebra; the achieved match is
        // bounded by BOBYQA's rho_end floor (2 independently-converged fits,
        // not a shared trajectory), not by 1e-10 — 1e-6 relative is the tight
        // bound this floor actually supports (measured ~2e-8 on this fixture).
        for j in 0..2 {
            assert!(
                (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
                "β[{j}] unweighted {} vs w≡2 {}",
                unweighted.beta[j],
                weighted.beta[j]
            );
            assert!(
                (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-6,
                "se[{j}] unweighted {} vs w≡2 {}",
                unweighted.se[j],
                weighted.se[j]
            );
        }
        assert_eq!(unweighted.tau2.len(), weighted.tau2.len());
        for k in 0..unweighted.tau2.len() {
            assert!(
                (unweighted.tau2[k] - weighted.tau2[k]).abs() / unweighted.tau2[k] < 1e-6,
                "tau2[{k}] unweighted {} vs w≡2 {}",
                unweighted.tau2[k],
                weighted.tau2[k]
            );
        }
    }

    /// Constant-weights invariance on a CROSSED random-slope design
    /// (`y ~ 1 + x + (1 + x | g1) + (1 | g2)`, the `sim_slope` fixture):
    /// w ≡ 2 must reproduce the unweighted β/SE/varcorr. This is the numeric
    /// check for the crossed-path weight sites in `add_rows_multi` — the
    /// intercept×intercept `zx += wᵢ` and the slope↔crossed `zx_slope += z·zw`
    /// (q_p = 2 primary slope + crossed intercept extra takes the scalar
    /// crossed branch, which unit-weight tests cannot distinguish from a
    /// wrong-power bug). Same θ̃ = √c·θ rationale and BOBYQA-floor tolerance as
    /// `fit_lmm_constant_weights_invariant`.
    #[test]
    fn fit_lmm_crossed_constant_weights_invariant() {
        let csv = include_str!("../parity/data_simulated/sim_slope.csv");
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
        let base_opts = FitOptions {
            target_indices: vec![0, 1],
            ..FitOptions::default()
        };
        let unweighted = fit_cold(&x, &y, n, p, &model, &ids, &base_opts);
        let weighted = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &ids,
            &FitOptions {
                weights: Some(vec![2.0; n]),
                ..base_opts
            },
        );
        assert!(unweighted.converged && weighted.converged);
        for j in 0..p {
            assert!(
                (unweighted.beta[j] - weighted.beta[j]).abs() / unweighted.beta[j].abs() < 1e-6,
                "β[{j}] unweighted {} vs w≡2 {}",
                unweighted.beta[j],
                weighted.beta[j]
            );
            assert!(
                (unweighted.se[j] - weighted.se[j]).abs() / unweighted.se[j] < 1e-6,
                "se[{j}] unweighted {} vs w≡2 {}",
                unweighted.se[j],
                weighted.se[j]
            );
        }
        // varcorr covers BOTH groupings' D̂ blocks (tau2 only reproduces the
        // (0,0) diagonal for the q=2 primary). Relative bound on the diagonals;
        // the small q=2 off-diagonal takes the same bound scaled to its own
        // magnitude floor.
        assert_eq!(unweighted.varcorr.len(), weighted.varcorr.len());
        for (gi, (vu, vw)) in unweighted
            .varcorr
            .iter()
            .zip(weighted.varcorr.iter())
            .enumerate()
        {
            assert_eq!(vu.len(), vw.len());
            for k in 0..vu.len() {
                let scale = vu[k].abs().max(1e-3);
                assert!(
                    (vu[k] - vw[k]).abs() / scale < 1e-5,
                    "varcorr[{gi}][{k}] unweighted {} vs w≡2 {}",
                    vu[k],
                    vw[k]
                );
            }
        }
    }

    /// Task 5 Step 6: the dense-LMM boundary (τ̂ ≈ 0, pinned exactly per the
    /// Q7 deterministic-pin policy — mirrors
    /// `lmm::tests::zero_between_cluster_variance_pins_at_exactly_zero`) must
    /// reproduce the weighted fixed-only WLS fit (Task 1, `fit_ols`) on the
    /// same rows: at θ̂=0 the mixed kernel's weighted Grams (`c`/`s`/`counts`,
    /// all Σwᵢ-scaled per Task 5's accumulator) collapse to the same weighted
    /// normal equations WLS solves directly, so the two paths must agree.
    #[test]
    fn fit_lmm_weighted_boundary_matches_wls() {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 7u64;
        let mut x = vec![0.0f64; n * 2];
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            ids[i] = (i % n_clusters) as u32;
            let x1 = lcg(&mut st);
            x[i * 2] = 1.0;
            x[i * 2 + 1] = x1;
            // i/n_clusters cycles 0..8 within each cluster: 4 even, 4 odd ⇒
            // the ±0.8 residuals cancel exactly per cluster (deterministic pin).
            let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
            y[i] = 0.5 + 0.4 * x1 + e;
            w[i] = 1.0 + (i % 3) as f64;
        }

        let mixed_model = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let mixed_ids = GroupIds {
            primary: ids,
            extra: vec![],
        };
        let mixed = fit_cold(
            &x,
            &y,
            n,
            2,
            &mixed_model,
            &mixed_ids,
            &FitOptions {
                target_indices: vec![0, 1],
                weights: Some(w.clone()),
                ..FitOptions::default()
            },
        );
        assert!(mixed.converged, "boundary pin still counts as converged");
        assert!(mixed.singular, "must pin at the τ=0 boundary");

        let fixed_only = ModelSpec {
            family: Family::Gaussian,
            re: None,
        };
        let wls = fit_cold(
            &x,
            &y,
            n,
            2,
            &fixed_only,
            &GroupIds::default(),
            &FitOptions {
                target_indices: vec![0, 1],
                weights: Some(w),
                ..FitOptions::default()
            },
        );
        assert!(wls.converged);

        for j in 0..2 {
            assert!(
                (mixed.beta[j] - wls.beta[j]).abs() / wls.beta[j].abs() < 1e-6,
                "β[{j}] mixed {} vs WLS {}",
                mixed.beta[j],
                wls.beta[j]
            );
            assert!(
                (mixed.se[j] - wls.se[j]).abs() / wls.se[j] < 1e-3,
                "se[{j}] mixed {} vs WLS {}",
                mixed.se[j],
                wls.se[j]
            );
        }
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

        let csv = include_str!("../parity/data_simulated/sim_slope.csv");
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

        let csv = include_str!("../parity/data_empirical/Penicillin.csv");
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

        let csv = include_str!("../parity/data_empirical/Pastes.csv");
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
        let csv = include_str!("../parity/data_empirical/grouseticks.csv");
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

    /// Parses `parity/data_empirical/grouseticks.csv` into the 3-crossed `TICKS ~ YEAR +
    /// cHEIGHT + (1|INDEX) + (1|BROOD) + (1|LOCATION)` design (observation-level
    /// INDEX + crossed BROOD, LOCATION). Shared by the lme4 fit gate below and the
    /// both-paths sparse-vs-dense Schur cross-checks (`sparse_schur_*`), which need
    /// direct `GlmmWorkspace`/`StructuredSchur` access that `fit_cold` doesn't expose.
    fn grouseticks_3crossed_inputs() -> (Vec<f64>, Vec<f64>, usize, usize, ModelSpec, GroupIds) {
        let csv = include_str!("../parity/data_empirical/grouseticks.csv");
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
    /// `lme4::glmer(family=poisson)` reference (`parity/results/lme4_empirical/grouseticks.json`).
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
        // Frozen lme4 reference (parity/results/lme4_empirical/grouseticks.json).
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
        let csv = include_str!("../parity/data_empirical/cbpp.csv");
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

    /// `FitOptions::parallel_inner` gates the AGQ cluster-outer restructuring
    /// (`agq::agq_deviance`'s `cluster_rows` path) but must never change the fitted
    /// result: cluster-outer and node-outer visit the same operands in the same
    /// per-accumulator order (`ClusterRowIndex`'s ascending-row guarantee), so a
    /// full cbpp AGQ fit through the stable `fit_cold` surface is bit-identical
    /// with the knob on vs off. Exact equality, not tolerance — this is the
    /// end-to-end witness for the same safety argument
    /// `agq_cluster_outer_bit_identical_to_node_outer` (glmm/tests.rs) checks at
    /// the kernel level.
    #[test]
    fn fit_glmm_binomial_agq_parallel_inner_knob_is_bit_identical() {
        let (x, y, cluster_ids, n) = cbpp_design();
        let p = 4;
        let model = cbpp_model();
        for nagq in [7u8, 11] {
            let ids = GroupIds {
                primary: cluster_ids.clone(),
                extra: vec![],
            };
            let f_on = fit_cold(
                &x,
                &y,
                n,
                p,
                &model,
                &ids,
                &FitOptions {
                    target_indices: vec![0, 1, 2, 3],
                    nagq,
                    parallel_inner: true,
                    ..FitOptions::default()
                },
            );
            let f_off = fit_cold(
                &x,
                &y,
                n,
                p,
                &model,
                &ids,
                &FitOptions {
                    target_indices: vec![0, 1, 2, 3],
                    nagq,
                    parallel_inner: false,
                    ..FitOptions::default()
                },
            );
            assert!(f_on.converged && f_off.converged, "nagq={nagq}");
            for (j, (&b_on, &b_off)) in f_on.beta.iter().zip(&f_off.beta).enumerate() {
                assert_eq!(
                    b_on.to_bits(),
                    b_off.to_bits(),
                    "nagq={nagq} β[{j}]: on={b_on} off={b_off}"
                );
            }
            for (j, (&s_on, &s_off)) in f_on.se.iter().zip(&f_off.se).enumerate() {
                assert_eq!(
                    s_on.to_bits(),
                    s_off.to_bits(),
                    "nagq={nagq} se[{j}]: on={s_on} off={s_off}"
                );
            }
            for (j, (&t_on, &t_off)) in f_on.tau2.iter().zip(&f_off.tau2).enumerate() {
                assert_eq!(
                    t_on.to_bits(),
                    t_off.to_bits(),
                    "nagq={nagq} tau2[{j}]: on={t_on} off={t_off}"
                );
            }
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
        let csv = include_str!("../parity/data_empirical/grouseticks.csv");
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
        let csv = include_str!("../parity/data_empirical/cbpp.csv");
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
            sim_clustered(include_str!("../parity/data_simulated/sim_gamma.csv"));
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
        // Via stddev_corr/varcorr — σ̂²-scaled like tau2 (B1 fix), so it gates
        // the public accessor directly against lme4's VarCorr stddev.
        let (sd, _corr) = f.stddev_corr(0);
        let sd_rel = (sd[0] - REF_CLUSTER_SD).abs() / REF_CLUSTER_SD;
        assert!(
            sd_rel < 5e-3,
            "cluster sd (stddev_corr) = {} vs lme4 {REF_CLUSTER_SD}",
            sd[0]
        );
        assert!(
            (sd[0] - f.tau2[0].sqrt()).abs() < 1e-12,
            "stddev_corr and tau2 must report the same σ̂-scaled sd"
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
        #[allow(clippy::needless_range_loop)]
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
            sim_clustered(include_str!("../parity/data_simulated/sim_nb.csv"));
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

    /// NB GLMM on an UNBALANCED NESTED design: `y ~ 1 + x + (1|g1/g2)` on
    /// sim_nb_nested (per-g1 sizes 8..120 on an exp ladder), gated against
    /// frozen `lme4::glmer.nb` (`parity/goldens/sim_nb_nested_glmm.json`).
    /// The nested extra rides the Pastes convention: `GroupIds.extra` carries
    /// the globally-unique g1:g2 level, `NestedWithin` is the topology tag,
    /// placeholder counts prove sizing comes from the ids. `dispersion = θ̂`;
    /// lme4-only SE (Hessian, glmm's default). tau2 layout: [primary g1 |
    /// nested g2:g1] — the golden's varcomp lists g2:g1 first (lme4 orders by
    /// descending level count). The oracle is sacred.
    #[test]
    fn fit_glmm_nb_nested_unbalanced_matches_lme4() {
        const REF_BETA: [f64; 2] = [0.584998228282064, 0.507364808670142];
        const REF_SE_HESSIAN: [f64; 2] = [0.204822249488268, 0.0539927793867315];
        const REF_G1_SD: f64 = 0.629024806733981;
        const REF_NEST_SD: f64 = 0.355202234990849;
        const REF_THETA: f64 = 1.43012979314052;
        // sim_nb_nested.csv: y,x,g1,g2 (g2 labels reused across g1 parents).
        let csv = include_str!("../parity/data_simulated/sim_nb_nested.csv");
        let mut y = Vec::<f64>::new();
        let mut xcol = Vec::<f64>::new();
        let mut g1_raw = Vec::<String>::new();
        let mut nest_raw = Vec::<String>::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            y.push(f[0].parse().unwrap());
            xcol.push(f[1].parse().unwrap());
            g1_raw.push(f[2].to_string());
            // Globally-unique nested level, the Pastes "sample" convention.
            nest_raw.push(format!("{}:{}", f[2], f[3]));
        }
        let n = y.len();
        let p = 2;
        let mut x = vec![0.0f64; n * p];
        for i in 0..n {
            x[i * p] = 1.0;
            x[i * p + 1] = xcol[i];
        }
        let (g1, _n_g1) = dense_str(&g1_raw);
        let (nest, _n_nest) = dense_str(&nest_raw);
        let model = ModelSpec {
            family: Family::NegativeBinomial {
                link: crate::NegBinomialLink::Log,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 1 }, // placeholder — ignored on data path
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::NestedWithin { n_per_parent: 1 }, // placeholder
                    slopes: vec![],
                }],
            }),
        };
        let f = fit_cold(
            &x,
            &y,
            n,
            p,
            &model,
            &GroupIds {
                primary: g1,
                extra: vec![nest],
            },
            &FitOptions {
                target_indices: vec![0, 1],
                ..FitOptions::default()
            },
        );
        assert!(f.converged, "nested NB GLMM must converge");
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
            let se_rel = (f.se[j] - REF_SE_HESSIAN[j]).abs() / REF_SE_HESSIAN[j];
            assert!(
                se_rel < 5e-2,
                "se[{j}] = {} vs lme4 {}",
                f.se[j],
                REF_SE_HESSIAN[j]
            );
        }
        let g1_rel = (f.tau2[0].sqrt() - REF_G1_SD).abs() / REF_G1_SD;
        assert!(
            g1_rel < 2e-2,
            "g1 sd = {} vs lme4 {REF_G1_SD}",
            f.tau2[0].sqrt()
        );
        let nest_rel = (f.tau2[1].sqrt() - REF_NEST_SD).abs() / REF_NEST_SD;
        assert!(
            nest_rel < 2e-2,
            "g2:g1 sd = {} vs lme4 {REF_NEST_SD}",
            f.tau2[1].sqrt()
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

    /// Total `Crossed` level count past MAX_CROSSED_LEVELS routes Sparse (the
    /// dense tail is cubic in the SUM of crossed levels, so two half-cap
    /// factors trip it together); at the cap exactly, intercept-only crossed
    /// extras stay NoZ. Nested extras don't count toward the sum (they live in
    /// the per-family elimination path, not the dense tail).
    #[test]
    fn classify_routes_many_crossed_levels_to_sparse() {
        let cap = crate::consts::MAX_CROSSED_LEVELS as u32;
        let spec = |extras: Vec<Grouping>| ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 10 },
                slopes: vec![],
                extra_groupings: extras,
            }),
        };
        let crossed = |n_clusters: u32| Grouping {
            relation: GroupingRelation::Crossed { n_clusters },
            slopes: vec![],
        };
        // One factor over the cap ⇒ Sparse.
        let over = spec(vec![crossed(cap + 1)]);
        assert!(matches!(
            super::classify_design_pub(&over, 1),
            super::Solver::Sparse
        ));
        // Sum over factors trips the cap even when each is under it.
        let sum_over = spec(vec![crossed(cap / 2 + 1), crossed(cap / 2 + 1)]);
        assert!(matches!(
            super::classify_design_pub(&sum_over, 1),
            super::Solver::Sparse
        ));
        // Exactly at the cap ⇒ NoZ unchanged.
        let at_cap = spec(vec![crossed(cap)]);
        assert!(matches!(
            super::classify_design_pub(&at_cap, 1),
            super::Solver::NoZ
        ));
        // A many-level NESTED extra doesn't count toward the crossed sum.
        let nested = spec(vec![Grouping {
            relation: GroupingRelation::NestedWithin {
                n_per_parent: cap + 1,
            },
            slopes: vec![],
        }]);
        assert!(matches!(
            super::classify_design_pub(&nested, 1),
            super::Solver::NoZ
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
        let csv = include_str!("../parity/data_empirical/grouseticks.csv");
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
                sim_clustered(include_str!("../parity/data_simulated/sim_gamma.csv"));
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
