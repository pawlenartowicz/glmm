//! Adaptive IRLS logistic regression kernel; outputs `z_sq_j = β̂_j² / Var(β̂_j)`
//! for hot-loop comparison against a caller-supplied precomputed squared-
//! critical-value table — no z-CDF calls, no SE sqrt, no per-coefficient
//! square roots anywhere in the loop.
//!
//! All per-fit buffers live in `SimWorkspace` (the `irls_*` fields, see workspace.rs).
//! The kernel takes a `GlmScratch<'w>` built inline at the call site (NLL split-borrow)
//! and returns a borrowed `GlmFitView<'a>`. No owned result struct.
//!
//! Algorithm — guards and tolerances:
//!   - Adaptive convergence: `|Δdeviance| < DEVIANCE_TOL = 1e-8`
//!   - Safety cap: `MAX_IRLS_ITERS = 50`
//!   - ETA_DIVERGENCE_CAP divergence guard: `iter ≥ 3 ∧ ‖η‖_∞ > 30 →
//!     non-converged` (skipped under the Gamma inverse link)
//!   - All-0 / all-1 short circuit
//!   - Post-fit saturation guard (50% of weights < 1e-5 ⇒ non-converged)
//!   - No step-halving: β_new is accepted directly
//!
//! Two `beta_start` modes: `None` seeds β = 0 with a family-specific η seed
//! (η = 0 for logit; R's `initialize` μ-start for Gamma-inverse; the null-model
//! μ₀ = ȳ + 0.1 for the log-link count families — see the cold-start arm), a
//! fixed reproducible cold
//! start; `Some(spec.effect_sizes)` — what the shipped hot loop passes — seeds
//! the spec-derived truth-start (Y is synthetic, so the true β on the logit
//! scale is known; mirrors `lmm::fit_lmm`'s `theta_start`). Either way the
//! accept rule and the |Δdev| < 1e-8 fixpoint are unchanged — only the path
//! to it shortens.
//!
//! Working-response IRLS form and the canonical-link weights follow McCullagh &
//! Nelder (1989), *Generalized Linear Models*, 2nd ed., Chapman & Hall — as MN89.

use faer::linalg::matmul::triangular::BlockStructure;
use faer::linalg::matmul::{matmul, triangular};
use faer::reborrow::{IntoConst, Reborrow, ReborrowMut};
use faer::{Accum, MatMut, MatRef, Par};

use crate::ols::{nan_fill_ols_scratch, triangular_solve_norm_sq};
use crate::spec::{BinomialLink, Family};
use crate::FLOAT_NEAR_ZERO;

/// IRLS safety cap.
pub const MAX_IRLS_ITERS: u32 = 50;
/// Adaptive convergence tolerance on `|Δdeviance|`.
pub const DEVIANCE_TOL: f64 = 1e-8;
/// Divergence guard: any |η_i| > ETA_DIVERGENCE_CAP at iter ≥ 3 marks
/// non-converged. The bound is on the LINEAR PREDICTOR, not on β: η = Xβ is
/// unchanged when a predictor column is rescaled (the compensating change in β̂
/// is exact), so the accept/reject decision does not depend on the caller's
/// choice of units — height in metres and height in kilometres give the same
/// verdict. 30 is the number the physical argument actually supports: on the
/// logit scale |η| = 30 is already p ≈ 1 − 1e-13, which is separation, not
/// signal. Skipped under the Gamma inverse link, where η = 1/μ makes a large
/// |η| an honest small-mean fit (see the guard site).
pub const ETA_DIVERGENCE_CAP: f64 = 30.0;
/// Floor on per-row IRLS weight `W_i = p_i (1-p_i)` to avoid division by zero
/// in the working response.
pub const WEIGHT_CLAMP: f64 = 1e-6;
/// Saturation post-fit guard: rows with `p_i(1-p_i) < SATURATION_W` count as
/// saturated. If the fraction exceeds `SATURATION_FRAC`, the fit is marked
/// non-converged.
pub const SATURATION_W: f64 = 1e-5;
/// Saturation post-fit guard: fraction of rows with `W_i < SATURATION_W`
/// above which the fit is marked non-converged.
pub const SATURATION_FRAC: f64 = 0.5;

/// Borrowed view into the workspace `irls_*` scratch produced by `glm_irls_fit`.
/// Lifetime ties back to the workspace.
///
/// `t_sq` field name reuses the OLS slot for uniformity — values are z² under
/// Logit (Wald-z²); the threshold comparison `stat_sq > crit_sq` is family-
/// agnostic so writeback code in `batch.rs` stays uniform.
pub struct GlmFitView<'a> {
    /// length P — fitted coefficients (NaN-filled on non-converged paths).
    pub betas: &'a [f64],
    /// length T — `((X'WX)⁻¹)_jj` for each target.
    pub var_diag: &'a [f64],
    /// length T — Wald z² for each target. Compared against `z_crit²` from
    /// `CritValueTable`.
    pub t_sq: &'a [f64],
    /// Cached lower-triangular Cholesky factor L of the last accepted X'WX.
    /// Valid only when `converged == true` (stale-or-zero otherwise — same
    /// staleness contract as `OlsFitView::factor`).
    pub l: MatRef<'a, f64>,
    /// Number of IRLS iterations completed (0 on the short-circuit / non-converged paths).
    pub n_iter: u32,
    /// Whether the deviance fixpoint `|Δdeviance| < DEVIANCE_TOL` was reached before `MAX_IRLS_ITERS`.
    pub converged: bool,
    /// Final-iteration Bernoulli deviance −2·Σ[y log p̂ + (1−y) log(1−p̂)].
    /// `NaN` on every non-converged / short-circuit return.
    pub deviance: f64,
    /// Null-model deviance −2·(Σy · log ȳ + (n − Σy) · log(1 − ȳ)).
    /// `NaN` whenever `Σy ∈ {0, n}` (the short-circuit path) or any other
    /// non-converged path.
    pub deviance_null: f64,
    /// Fitted means μ̂ = link⁻¹(η̂), length `n`. Valid only when `converged`
    /// (stale/uninitialised otherwise — same staleness contract as `l`). Carried
    /// on the view so `glm_view_to_fit` can assemble `fitted`/`loglik`/Gamma
    /// dispersion without a second borrow of the IRLS scratch the view holds.
    pub mu: &'a [f64],
    /// Scale-invariant per-column pivot ratio of the CONVERGED `X'WX`
    /// (`crate::ols::min_pivot_ratio`), with `pivot_col` the column that attains
    /// it. **Detection only** — nothing in this route reads it to accept or
    /// reject a design, and nothing may start to: the IRLS weights come out of
    /// the fit itself, so an ill-conditioned-but-computable design already fits
    /// here and turning that into a refusal needs its own calibration against
    /// `glm.fit`. The value exists so a diagnostics channel can tell the caller
    /// their coefficients are barely identified, which is the one thing nothing
    /// upstream can see: a design full-rank in raw `x` can be singular under the
    /// converged weights, and the pre-dispatch alias gate cannot predict them
    /// even in principle.
    ///
    /// `NaN` / `0` on every non-converged and short-circuit return — there is no
    /// converged `X'WX` to measure.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
}

/// Caller-owned scratch borrowed from `SimWorkspace` field-by-field. Built
/// inline at the call site (NLL split-borrow with simultaneous shared borrows
/// of `ws.x_full` / `ws.y_full`). **Do not** wrap in a helper method
/// `ws.glm_scratch()` — that re-introduces the whole-struct exclusive borrow
/// problem (NLL cannot split borrow a method receiver from its fields).
pub struct GlmScratch<'w> {
    /// Linear predictor η = Xβ, length `n`, one entry per row.
    pub irls_eta: &'w mut [f64],
    /// Fitted mean μ (= p under logit link), length `n`, one entry per row.
    pub irls_p: &'w mut [f64],
    /// IRLS working weight per row, length `n`, clamped to `WEIGHT_CLAMP`.
    pub irls_w: &'w mut [f64],
    /// IRLS working response per row, length `n`.
    pub irls_z: &'w mut [f64],
    /// Current-iteration coefficient vector β, length `p`, in x-matrix column order.
    pub irls_betas: &'w mut [f64],
    /// Next-iteration coefficient vector produced by the WLS solve, length `p`.
    pub irls_betas_new: &'w mut [f64],
    /// Output slot for `((X'WX)⁻¹)_jj` per target, length `t` (one per requested target index).
    pub irls_var_diag: &'w mut [f64],
    /// Output slot for Wald z² per target, length `t`.
    pub irls_t_sq: &'w mut [f64],
    /// Forward-substitution scratch `u` for `L·u = e_j`, length `p`; reused per target when computing `var_diag`.
    pub irls_u_scratch: &'w mut [f64],
    /// Weighted normal-equations matrix X'WX, `p × p`.
    pub irls_xtwx: MatMut<'w, f64>,
    /// Weighted right-hand side X'Wz, length `p`; overwritten with the WLS solution in place.
    pub irls_xtwz: &'w mut [f64],
    /// Lower-triangular Cholesky factor L of X'WX, `p × p`.
    pub irls_l: MatMut<'w, f64>,
    /// Per-iteration W∘X scratch (column-major, stride n); needs `len ≥ n·p`.
    pub irls_wx: &'w mut [f64],
}

// ---------------------------------------------------------------------------
// Numerically stable helpers
// ---------------------------------------------------------------------------

/// Numerically stable sigmoid: branches on the sign of `eta` so `exp` never
/// overflows and the ratio avoids cancellation. Production paths now compute `p`
/// in the vectorized `simd_transcendental`
/// kernels (fit: `pw_and_log1pexp_sum`; generation: `sigmoid_fill`, ≤2 ULP of this
/// form); this stays as the libm reference for tests.
#[cfg_attr(not(test), allow(dead_code))]
#[inline]
pub fn sigmoid_stable(eta: f64) -> f64 {
    if eta >= 0.0 {
        let z = (-eta).exp();
        1.0 / (1.0 + z)
    } else {
        let z = eta.exp();
        z / (1.0 + z)
    }
}

// ---------------------------------------------------------------------------
// IRLS kernel
// ---------------------------------------------------------------------------

/// Fit a GLM via adaptive IRLS. All buffers live in `scratch`; the returned view
/// borrows from the same storage.
///
/// `family` selects the IRLS math, all of it behind one batched call into
/// `simd_transcendental::family_pass`. Unweighted `Family::Binomial { link:
/// Logit }` reaches the fused kernel `pw_and_log1pexp_sum` there — the MCPower
/// hot loop, byte-identical to before that kernel was shared; every other
/// family/link runs a vectorized arm written against [`crate::family`]'s scalar
/// formulas. Gamma/NB dispersion is handled by the caller (`fit.rs`),
/// not here — this kernel folds `φ=1`. `nb_theta` is the NB shape θ̂ the caller's
/// outer-θ loop fixes for this fit (only the NB family reads it via
/// `family::variance`/`dev_resid`; pass `f64::NAN` for every other family).
///
/// - `x`: `n × p` design (column-major faer).
/// - `y`: length `n`; {0.0, 1.0} for binomial, counts for Poisson/NB, positive for Gamma.
/// - `target_indices`: per-coefficient indices to compute `z²` for.
/// - `beta_start`: `None` → β = 0 cold start (bit-identical pre-warm-start
///   path); `Some(β₀)` (length `p`) → spec-derived truth start — seeds β and
///   computes η = X·β₀ once. A per-scenario constant, so determinism and
///   chunk merging are unaffected.
/// - `prior_w`: per-row prior (case) weight `wᵢ`, McCullagh & Nelder's prior-
///   weight sense (MN89 §2.2.2) — `None` = unit weight. Multiplies the IRLS
///   working weight (`irls_w[i] = (wᵢ·W_raw).max(WEIGHT_CLAMP)`) and the
///   per-row deviance contribution; the working response `z = η + r` is
///   untouched (prior weights don't scale the working residual). The fused
///   `log1pexp` deviance identity holds only for unweighted Bernoulli rows, so
///   `Some(_)` routes `Family::Binomial { link: Logit }` to the weighted arm
///   instead (unweighted logit keeps the fused kernel). Matches R
///   `glm(weights=)` (`fit_glm_gamma_weighted_matches_r`,
///   `fit_glm_binomial_weighted_aggregated_matches_r`).
/// - `scratch`: borrowed mutable slots from `SimWorkspace.irls_*`.
///
/// IRLS weight: `W = diag(μ(1−μ))` under canonical logit (`Family::Binomial
/// { link: Logit }`); every other family/link uses the general Fisher-scoring
/// weight `W_i = 1 / (Var(μ_i) · g'(μ_i)²)` from [`crate::family`].
///
/// `deviance_null` matches R's `glm(family=binomial)$null.deviance` (see
/// `glm_deviance_null_golden_value`). The full β̂/deviance path is pinned
/// against R for the weighted branch (`glm_weighted_deviance_null_golden_value`
/// here; `fit_glm_gamma_weighted_matches_r`,
/// `fit_glm_binomial_weighted_aggregated_matches_r` in `fit/glm_tests.rs`),
/// which is what unweighted-binomial aggregated-proportion data (R's
/// `cbind(successes, failures) ~ ...`, e.g. the `cbpp_logit_glm` validation golden)
/// also routes through. The one path still without a full β̂ oracle is the
/// unweighted canonical-logit fast path (`Family::Binomial { link: Logit }`,
/// `prior_w: None`) — its only external check is the null-deviance golden
/// above, which depends solely on ȳ, not the fitted β.
#[allow(clippy::too_many_arguments)] // marshals (family, nb_theta, x, y, target_indices, beta_start, prior_w, offset, scratch)
pub fn glm_irls_fit<'a>(
    family: Family,
    nb_theta: f64,
    x: MatRef<'_, f64>,
    y: &[f64],
    target_indices: &[u32],
    beta_start: Option<&[f64]>,
    prior_w: Option<&[f64]>,
    // Per-row additive linear-predictor offset `oᵢ` (R `glm(offset=)`): the
    // carried η is `o + Xβ` throughout, and the WLS solve regresses `z − o` on
    // X (β must not absorb the offset). `None` leaves every loop structurally
    // unchanged — byte-identity for offset-free fits.
    offset: Option<&[f64]>,
    scratch: GlmScratch<'a>,
) -> GlmFitView<'a> {
    let n = x.nrows();
    let p = x.ncols();
    let t = target_indices.len();

    debug_assert_eq!(n, y.len(), "glm_irls_fit: y length must match X.nrows()");

    let GlmScratch {
        irls_eta,
        irls_p,
        irls_w,
        irls_z,
        irls_betas,
        irls_betas_new,
        irls_var_diag,
        irls_t_sq,
        irls_u_scratch,
        mut irls_xtwx,
        irls_xtwz,
        mut irls_l,
        irls_wx,
    } = scratch;

    debug_assert!(p <= irls_betas.len(), "scratch sized for fewer predictors");
    debug_assert!(t <= irls_var_diag.len());
    debug_assert!(n <= irls_eta.len());

    // NaN-fill on early-return / non-converged paths so callers can detect
    // non-convergence from NaN outputs alone. Successful paths overwrite
    // every populated slot.
    nan_fill_ols_scratch(irls_betas, irls_var_diag, irls_t_sq, p, t);

    // Short-circuit: n ≤ p, or no predictors.
    if n <= p || p == 0 {
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter: 0,
            converged: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
            mu: &irls_p[..n],
            pivot: f64::NAN,
            pivot_col: 0,
        };
    }

    // All-0 / all-1 short circuit — Bernoulli-only: without it the binomial
    // working response divides by zero on iter 1 (W collapses, p collapses).
    // Poisson/Gamma/NB have no such degeneracy (y_sum ≥ n is ordinary for
    // counts), so the guard is gated behind the binomial family; their n≤p
    // guard above still holds.
    let mut y_sum = 0.0;
    for &yi in &y[..n] {
        y_sum += yi;
    }
    if matches!(family, Family::Binomial { .. }) && (y_sum <= 0.0 || y_sum >= n as f64) {
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter: 0,
            converged: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
            mu: &irls_p[..n],
            pivot: f64::NAN,
            pivot_col: 0,
        };
    }

    // Null-model deviance — computed once, reused at the batch site for the
    // LRT. Stored in a local so the per-iteration deviance tracker doesn't
    // clobber it. The all-0 / all-1 branch above guarantees `y_bar ∉ {0, 1}`
    // here, so no `ln(0)` risk.
    // Intercept-only MLE is μ̂₀=ȳ for any (family, link) with unit weights; under
    // prior weights it is the WEIGHTED mean μ̂₀ = Σwᵢyᵢ/Σwᵢ (R's
    // glm(weights=)$null.deviance), so the null deviance is Σ wᵢdᵢ(y, μ̂₀).
    // Logit keeps the closed-form Bernoulli expression verbatim (byte-identity)
    // only when unweighted — a weighted fit routes through the same general
    // fold the per-iteration deviance uses (see the `prior_w` guard below), so
    // `deviance_null` stays on the same weighting convention as `deviance`.
    // (The all-0/all-1 short circuit above keys off raw y, which is equivalent
    // under positive weights: the weighted mean is 0/1 exactly when ȳ is.)
    let y_bar = match prior_w {
        Some(w) => {
            let (mut wy_sum, mut w_sum) = (0.0, 0.0);
            for (i, &yi) in y[..n].iter().enumerate() {
                wy_sum += w[i] * yi;
                w_sum += w[i];
            }
            wy_sum / w_sum
        }
        None => y_sum / n as f64,
    };
    let deviance_null = match family {
        Family::Binomial {
            link: BinomialLink::Logit,
        } if prior_w.is_none() => {
            -2.0 * (y_sum * y_bar.ln() + (n as f64 - y_sum) * (1.0 - y_bar).ln())
        }
        other => {
            let mu0 = crate::family::clamp_mu(other, y_bar);
            let mut d = 0.0;
            for (i, &yi) in y[..n].iter().enumerate() {
                let pw = prior_w.map_or(1.0, |w| w[i]);
                d += pw * crate::family::dev_resid(other, nb_theta, yi, mu0);
            }
            d
        }
    };

    // Seed β and η here; the IRLS loop carries η forward from each
    // iteration's post-accept recompute (the β-accept step) instead of
    // recomputing X·β at the top of every iteration.
    match beta_start {
        // Truth start: β ← β₀, η = X·β₀ computed once (O(np)). The η loop
        // matches the β-accept step's summation order (column-sweep axpy) so a warm fit
        // walks the same arithmetic the accept path uses — change together.
        Some(b0) => {
            debug_assert_eq!(b0.len(), p, "beta_start length must match X.ncols()");
            irls_betas[..p].copy_from_slice(&b0[..p]);
            irls_eta[..n].fill(0.0);
            for j in 0..p {
                let b_j = irls_betas[j];
                for i in 0..n {
                    irls_eta[i] += x[(i, j)] * b_j;
                }
            }
            // η = o + Xβ₀ — mirrors the accept-step recompute below; change together.
            if let Some(o) = offset {
                for i in 0..n {
                    irls_eta[i] += o[i];
                }
            }
        }
        // Cold start: β ← 0 with a family-specific η seed. Logit keeps η = 0
        // (μ=0.5, bit-identical to the pre-warm-start behavior). The Gamma
        // **inverse** link cannot start at η=0 (μ=1/0): seed η = 1/y per row
        // (R's `mustart=y`, `etastart=1/y`). Log-link count families
        // (Poisson/NB) seed the null model, μ₀ = ȳ + 0.1 ⇒ η = ln(ȳ+0.1) on
        // every row: from η = 0 (μ=1) on data with ȳ ≳ ~25–30 the first WLS
        // step overshoots and IRLS runs away (β → ~9e304) — there is
        // deliberately no step-halving to catch it (see the accept step).
        // NOT R's per-row μ₀ = y + 0.1 `initialize`: on
        // zero-heavy small-θ NB data the per-row seed's first step fits the
        // log-count mean (intercept ≈ ln 0.1), the second explodes past
        // the divergence cap, and recovery crawls at ~1 unit/iter past the iteration
        // budget — R's own glm.fit fails identically there (glm.nb only
        // survives by warm-starting each alternation from a Poisson fit); the
        // constant ȳ seed converges on both regimes. β stays 0 in all seeded
        // cases; the first solve overwrites it, so X·β consistency resumes
        // from iter 1.
        None => {
            irls_betas[..p].fill(0.0);
            match family {
                Family::Gamma {
                    link: crate::spec::GammaLink::Inverse,
                    ..
                } => {
                    for i in 0..n {
                        irls_eta[i] = 1.0 / crate::family::clamp_mu(family, y[i]);
                    }
                }
                // η = 1/μ² with μ₀ = y (R's `mustart = y` for
                // inverse.gaussian): η = 0 would put μ at ∞ for this link, and
                // η ≤ 0 is outside its domain entirely. Mirrors the
                // Gamma-inverse seed above. The IG **log** link needs no seed —
                // η = 0 gives μ = 1, the same treatment Gamma-log gets.
                Family::InverseGaussian {
                    link: crate::spec::InverseGaussianLink::InverseSquared,
                } => {
                    for i in 0..n {
                        let mu0 = crate::family::clamp_mu(family, y[i]);
                        irls_eta[i] = 1.0 / (mu0 * mu0);
                    }
                }
                Family::Poisson { .. } | Family::NegativeBinomial { .. } => {
                    let ybar = y[..n].iter().sum::<f64>() / n.max(1) as f64;
                    irls_eta[..n].fill((ybar + 0.1).ln());
                }
                _ => irls_eta[..n].fill(0.0),
            }
        }
    }

    // Cholesky factor of the last accepted X'WX. The lower-triangular L is
    // materialised once after the loop (deferred from per-iteration so each
    // IRLS step skips an owned-Mat allocation). `Llt` owns its storage, so
    // keeping the last one past the next iter's irls_xtwx rebuild is safe.
    let mut last_chol = None;

    let mut deviance_prev = f64::INFINITY;
    let mut deviance_final = f64::NAN;
    let mut converged = false;
    let mut had_pd_failure = false;
    let mut n_iter: u32 = 0;

    // IRLS: each pass is a weighted least squares solve
    // β_new = (X'WX)⁻¹ X'Wz on the working response z = η + (y − μ)/W, μ = σ(η).
    // W = diag(μ(1−μ)) under canonical logit; the general Fisher-scoring
    // weight (see the `///` doc above) applies otherwise. Iterated to the
    // |Δdeviance| fixpoint it is the maximum-likelihood β. MN89 §4.
    // `0..=`: pass k's top checks convergence of the deviance that pass k−1's
    // solve produced (off the carried η), so the final solve still gets its
    // check without the extra pass buying another factorization. The deviance
    // check reads the top-of-pass fused kernel's output directly — no separate
    // `log1pexp` sweep — so `deviance_final` is always one pass behind the β
    // that produced it, by construction, not by an accident of ordering.
    //
    // The divergence guard bounds |η|, which under the Gamma inverse link is
    // 1/μ — a legitimate small-mean fit sits far above the threshold there. That
    // family/link pair falls through to the other exits instead: clamp_eta's
    // ±700 (src/family.rs), the non-finite guard on β_new, and MAX_IRLS_ITERS.
    // Loop-invariant, so it is evaluated once.
    let eta_guard_active = !matches!(
        family,
        Family::Gamma {
            link: crate::spec::GammaLink::Inverse
        }
    );

    for iter in 0..=MAX_IRLS_ITERS {
        // η = X · β is already in `irls_eta`: seeded to 0 (β = 0) or the
        // truth start before the loop, then refreshed by the accept step below
        // after each β update. Reusing it here is bit-identical to recomputing
        // X·β (same β, same summation order).

        // p, W, z, and the deviance for this β — one batched call into the shared
        // family kernel (`simd_transcendental::family_pass`), which owns the
        // clamps, the inverse link, the Fisher weight with its prior-weight
        // multiply and `WEIGHT_CLAMP` floor, the working response `z = η + r`
        // (prior weights don't scale it — MN89 §2.2.2), and the `Σ wᵢdᵢ` fold.
        // `Σ y·η` is only read by the unweighted-Bernoulli-logit deviance
        // identity `2·(Σ log1pexp(η) − Σ y·η)`; it is accumulated here because
        // that arm's kernel streams η without the response.
        let mut yeta = 0.0;
        for i in 0..n {
            yeta += y[i] * irls_eta[i];
        }
        let (deviance, _infeasible) = crate::simd_transcendental::family_pass(
            family,
            nb_theta,
            &mut irls_eta[..n],
            &y[..n],
            // Sliced to n so the kernel's SIMD head/tail split lines up with
            // η's; an empty slice is its "unit weights" spelling.
            prior_w.map_or(&[][..], |w| &w[..n]),
            prior_w.is_some(),
            yeta,
            &mut irls_p[..n],
            &mut irls_w[..n],
            &mut irls_z[..n],
        );
        // z − o: the WLS below solves β from X alone, so the offset's fixed
        // contribution must leave the working response.
        if let Some(o) = offset {
            for i in 0..n {
                irls_z[i] -= o[i];
            }
        }

        // Adaptive early exit on |Δdeviance| at the CURRENT β — the one the
        // previous pass's solve accepted. Pass 0 sees the seed β, which has no
        // prior deviance to compare against: skip the check and the tracker so
        // the first real comparison is β₂-vs-β₁.
        if iter > 0 {
            deviance_final = deviance;
            if (deviance - deviance_prev).abs() < DEVIANCE_TOL {
                converged = true;
                break;
            }
            deviance_prev = deviance;
        }
        // Solve budget spent — the `..=` pass exists only for the check above.
        if iter == MAX_IRLS_ITERS {
            break;
        }
        n_iter = iter + 1;

        // WX = W∘X into irls_wx (column-major, mirrors x), then
        // X′WX (lower triangle — Cholesky reads Side::Lower only) and X′Wz
        // through faer GEMM (`Par::Seq`; per-fit parallelism is the outer
        // rayon loop). GEMM's blocked accumulation keeps the FP-add chain off
        // the critical path on wide p, unlike a per-entry row-order dot
        // product, which serializes one add per row.
        {
            let wx = &mut irls_wx[..n * p];
            for j in 0..p {
                let wxj = &mut wx[j * n..(j + 1) * n];
                for i in 0..n {
                    wxj[i] = irls_w[i] * x[(i, j)];
                }
            }
        }
        let wx_ref = MatRef::from_column_major_slice(&irls_wx[..n * p], n, p);
        triangular::matmul(
            irls_xtwx.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            x.transpose(),
            BlockStructure::Rectangular,
            wx_ref,
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        matmul(
            MatMut::from_column_major_slice_mut(&mut irls_xtwz[..p], p, 1),
            Accum::Replace,
            wx_ref.transpose(),
            MatRef::from_column_major_slice(&irls_z[..n], n, 1),
            1.0,
            Par::Seq,
        );

        // Cholesky of X'WX on the lower triangle. faer's high-level API
        // returns an owned factor (it does not consume irls_xtwx, which is
        // rebuilt from scratch each iter anyway). The factor drives the in-place
        // β solve below; its L is materialised once after the loop.
        let chol = match irls_xtwx.rb().llt(faer::Side::Lower) {
            Ok(c) => c,
            Err(_) => {
                had_pd_failure = true;
                break;
            }
        };

        // Solve β_new = L⁻ᵀ L⁻¹ · X'Wz using chol.solve_in_place.
        // First write xtwz into a 1-col view; chol.solve_in_place expects a
        // MatMut. Since irls_xtwz is &mut [f64] of length p, build a
        // temporary MatMut via faer's `from_column_major_slice_mut`.
        {
            use faer::linalg::solvers::Solve;
            let mut rhs = MatMut::from_column_major_slice_mut(irls_xtwz, p, 1usize);
            chol.solve_in_place(rhs.rb_mut());
        }
        irls_betas_new[..p].copy_from_slice(&irls_xtwz[..p]);

        // Stash this iteration's factor; the cached L is materialised once
        // after the loop (only the converged path reads it).
        last_chol = Some(chol);

        // Non-finite guard on β_new.
        let mut all_finite = true;
        for &b in &irls_betas_new[..p] {
            if !b.is_finite() {
                all_finite = false;
                break;
            }
        }
        if !all_finite {
            break;
        }

        // Accept β_new unconditionally and compute new deviance.
        //
        // β_new is accepted unconditionally — no step-halving. The
        // DEVIANCE_TOL early-exit and MAX_IRLS_ITERS cap are sufficient
        // divergence guards.
        irls_betas[..p].copy_from_slice(&irls_betas_new[..p]);
        // η = X·β as a column sweep (axpy over x columns) — each η_i still
        // accumulates in the same j order from 0.0, bit-identical to the
        // strided per-row form. Mirrors the truth-start seed loop above —
        // change together.
        irls_eta[..n].fill(0.0);
        for j in 0..p {
            let b_j = irls_betas[j];
            for i in 0..n {
                irls_eta[i] += x[(i, j)] * b_j;
            }
        }
        // η = o + Xβ — mirrors the truth-start seed; change together. (Cold
        // seeds deliberately omit o, like R's η₀ = link(mustart): the seed is
        // a start, not a fixpoint constraint; consistency begins at iter 1.)
        if let Some(o) = offset {
            for i in 0..n {
                irls_eta[i] += o[i];
            }
        }

        // Divergence guard at iter ≥ 3, on the linear predictor just recomputed
        // above. Fires before the next pass's convergence check — a capped fit
        // never reports converged. The sweep is over n rather than p; that is a
        // longer pass than the old |β| one, and negligible beside the GEMM that
        // builds X'WX each iteration.
        if eta_guard_active && iter >= 3 {
            let mut max_abs: f64 = 0.0;
            for &e in &irls_eta[..n] {
                let ae = e.abs();
                if ae > max_abs {
                    max_abs = ae;
                }
            }
            if max_abs > ETA_DIVERGENCE_CAP {
                break;
            }
        }
    }

    if had_pd_failure {
        converged = false;
    }

    // Post-fit saturation guard. Evaluate p_i = σ(η_i) from the final irls_eta
    // to catch the case where the loop early-exits but β has drifted into a
    // saturated region.
    if converged {
        // irls_w already holds the FINAL η's weights: convergence only breaks
        // right after the top-of-pass fused kernel refilled p/W from the carried
        // η — no recompute needed. The clamp floor (WEIGHT_CLAMP = 1e-6) sits
        // below SATURATION_W (1e-5), so `w < SATURATION_W` is equivalent to the
        // raw `p(1-p) < SATURATION_W` test the scalar guard used.
        //
        // Weighted case: irls_w carries wᵢ·W_raw, so the threshold scales with
        // wᵢ too — otherwise a legitimately small prior weight would masquerade
        // as saturation. The guard tests the FAMILY weight μ(1−μ) (or its
        // generalization), not the case weight, so it compares against
        // SATURATION_W·wᵢ rather than a fixed floor. Sub-unit-weight edge:
        // for wᵢ < WEIGHT_CLAMP/SATURATION_W = 0.1 the clamp floor (1e-6)
        // exceeds the scaled threshold SATURATION_W·wᵢ, so a truly saturated
        // row escapes the guard — accepted edge; case weights are typically ≥ 1.
        let saturated = (0..n)
            .filter(|&i| {
                let pw = prior_w.map_or(1.0, |w| w[i]);
                irls_w[i] < SATURATION_W * pw
            })
            .count();
        if (saturated as f64) / (n as f64) > SATURATION_FRAC {
            converged = false;
        }
    }

    if !converged {
        // Partial re-NaN: inference outputs only — `betas` are deliberately
        // left at the loop's last fitted value, so the 3-array
        // `nan_fill_ols_scratch` must NOT be used here.
        irls_var_diag[..t].fill(f64::NAN);
        irls_t_sq[..t].fill(f64::NAN);
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter,
            converged: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
            mu: &irls_p[..n],
            pivot: f64::NAN,
            pivot_col: 0,
        };
    }

    // Materialise the cached lower-triangular L of the last accepted X'WX —
    // once, now that the fit has converged (deferred from the IRLS loop). On
    // the converged path `last_chol` is always `Some`: `converged` is only set
    // after a successful factorization stashed it.
    if let Some(chol) = last_chol {
        let l = chol.L();
        for j in 0..p {
            for i in 0..p {
                irls_l[(i, j)] = if i >= j { l[(i, j)] } else { 0.0 };
            }
        }
    }

    // Per-target Var(β̂_j) = ((X'WX)⁻¹)_jj — see `ols::triangular_solve_norm_sq`
    // for the forward-solve identity (no σ̂² scaling here: under Bernoulli the
    // score covariance is (X'WX)⁻¹ directly).
    for (out_idx, &tj) in target_indices.iter().enumerate() {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        let norm_sq = triangular_solve_norm_sq(
            irls_l.rb(),
            |i| if i == tj { 1.0 } else { 0.0 },
            irls_u_scratch,
            p,
            false, // lower-triangular L: row access factor[(i, k)]
        );
        irls_var_diag[out_idx] = norm_sq;
        if norm_sq > FLOAT_NEAR_ZERO && norm_sq.is_finite() {
            let beta_j = irls_betas[tj];
            irls_t_sq[out_idx] = (beta_j * beta_j) / norm_sq;
        } else {
            irls_t_sq[out_idx] = f64::NAN;
        }
    }

    // Ill-conditioning DETECTION on the converged X'WX, read off the L just
    // materialised. Purely observational: no branch below depends on it. The
    // matrix is the one the coefficients and their SEs actually came from, which
    // is why it is measured here and not on the raw design.
    let (pivot, pivot_col) = crate::ols::min_pivot_ratio(irls_l.rb(), p);

    GlmFitView {
        betas: &irls_betas[..p],
        var_diag: &irls_var_diag[..t],
        t_sq: &irls_t_sq[..t],
        l: irls_l.into_const(),
        n_iter,
        converged: true,
        deviance: deviance_final,
        deviance_null,
        mu: &irls_p[..n],
        pivot,
        pivot_col: pivot_col as u32,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::TestWs;
    use faer::Mat;

    /// Build a `GlmScratch` borrowing every IRLS field of `ws`. Used by tests
    /// to avoid duplicating the inline struct literal at each call site.
    fn glm_scratch(ws: &mut TestWs) -> GlmScratch<'_> {
        GlmScratch {
            irls_eta: &mut ws.irls_eta,
            irls_p: &mut ws.irls_p,
            irls_w: &mut ws.irls_w,
            irls_z: &mut ws.irls_z,
            irls_betas: &mut ws.irls_betas,
            irls_betas_new: &mut ws.irls_betas_new,
            irls_var_diag: &mut ws.irls_var_diag,
            irls_t_sq: &mut ws.irls_t_sq,
            irls_u_scratch: &mut ws.irls_u_scratch,
            irls_xtwx: ws.irls_xtwx.as_mut(),
            irls_xtwz: &mut ws.irls_xtwz,
            irls_l: ws.irls_l.as_mut(),
            irls_wx: &mut ws.irls_wx,
        }
    }

    #[test]
    fn glm_all_zero_y_short_circuits() {
        let n = 100;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f64) / (n as f64) - 0.5;
        }
        let y = vec![0.0f64; n];
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(!fit.converged);
        assert_eq!(fit.n_iter, 0);
    }

    #[test]
    fn glm_all_one_y_short_circuits() {
        let n = 100;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f64) / (n as f64) - 0.5;
        }
        let y = vec![1.0f64; n];
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(!fit.converged);
        assert_eq!(fit.n_iter, 0);
    }

    #[test]
    fn glm_rank_deficient_design() {
        // X with two identical columns → X'WX is singular.
        let n = 100;
        let p = 3;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            // Column 1 and column 2 identical.
            x[(i, 1)] = ((i as f64) / (n as f64)) - 0.5;
            x[(i, 2)] = x[(i, 1)];
        }
        // Build y with mixed 0/1 to avoid the all-0/all-1 short circuit.
        let y: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![1, 2];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(
            !fit.converged,
            "rank-deficient design must report non-converged"
        );
    }

    /// GLM Wald z² is NaN on a non-converged fit (the variance is not
    /// recoverable). Error path for the z² shape rule — a broken kernel that
    /// emitted a finite garbage z² when the fit failed would be caught.
    #[test]
    fn glm_z_sq_nan_on_non_converged() {
        // All-zero y short-circuits to non-converged.
        let n = 100;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f64) / (n as f64) - 0.5;
        }
        let y = vec![0.0f64; n];
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(!fit.converged);
        for &t in fit.t_sq.iter() {
            assert!(t.is_nan(), "z² must be NaN on non-converged fit, got {t}");
        }
    }

    #[test]
    fn glm_deviance_nan_on_non_converged() {
        // All-0 y short-circuit → non-converged.
        let n = 100;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f64) / (n as f64) - 0.5;
        }
        let y = vec![0.0f64; n];
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(!fit.converged);
        assert!(
            fit.deviance.is_nan(),
            "deviance must be NaN on non-converged"
        );
        assert!(
            fit.deviance_null.is_nan(),
            "deviance_null must be NaN on all-0 short-circuit (sum_y = 0)"
        );
    }

    #[test]
    fn glm_separation_marks_non_converged() {
        // Build a fully-separated dataset: y = (x > 0). Logistic regression
        // diverges (β → ∞, so η → ∞); the divergence cap or the saturation guard should fire.
        let n = 200;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            x[(i, 0)] = 1.0;
            let xi = (i as f64 - n as f64 / 2.0) / 10.0; // wide span
            x[(i, 1)] = xi;
            y[i] = if xi > 0.0 { 1.0 } else { 0.0 };
        }
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![1];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(
            !fit.converged,
            "fully separated data must report non-converged"
        );
    }

    /// Well-conditioned logistic design whose slope column is multiplied by
    /// `scale`. The response is generated from the UNSCALED column, so the three
    /// scalings are literally the same model in three unit systems. The LCG is
    /// local to this builder — the fit path itself stays RNG-free.
    fn scaled_logit_design(n: usize, scale: f64) -> (Mat<f64>, Vec<f64>) {
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut s: u64 = 12345;
        for i in 0..n {
            let xu = ((i as f64) / (n as f64) - 0.5) * 4.0;
            x[(i, 0)] = 1.0;
            x[(i, 1)] = xu * scale;
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / ((1u64 << 53) as f64);
            let p = 1.0 / (1.0 + (0.3 - 2.0 * xu).exp());
            y[i] = f64::from(u8::from(u < p));
        }
        (x, y)
    }

    /// Same idea for a Poisson log-link design: μ = exp(0.5 + 0.8·x), counts
    /// spread deterministically around it.
    fn scaled_poisson_design(n: usize, scale: f64) -> (Mat<f64>, Vec<f64>) {
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let xu = ((i as f64) / (n as f64) - 0.5) * 4.0;
            x[(i, 0)] = 1.0;
            x[(i, 1)] = xu * scale;
            let mu = (0.5 + 0.8 * xu).exp();
            let jitter = 0.7 + 0.2 * ((i % 4) as f64);
            y[i] = (mu * jitter).round();
        }
        (x, y)
    }

    /// η = Xβ does not move when a predictor column is rescaled — the
    /// compensating change in β̂ is exact — so a guard that bounds η gives the
    /// same accept/reject decision in every unit system. Under the old
    /// `|β_j| > 30` cap the x/1000 fit was rejected with NaN variances while the
    /// identical x and x·1000 fits were accepted.
    #[test]
    fn glm_logit_guard_is_scale_invariant() {
        let n = 200;
        let p = 2;
        let targets: Vec<u32> = vec![0, 1];
        let fit_at = |scale: f64| {
            let (x, y) = scaled_logit_design(n, scale);
            let mut ws = TestWs::new(n, p, 0);
            let f = glm_irls_fit(
                crate::Family::Binomial {
                    link: crate::BinomialLink::Logit,
                },
                f64::NAN,
                x.as_ref(),
                &y,
                &targets,
                None,
                None,
                None,
                glm_scratch(&mut ws),
            );
            (
                f.converged,
                f.betas.to_vec(),
                f.deviance,
                f.var_diag.to_vec(),
            )
        };

        let (c1, b1, d1, v1) = fit_at(1.0);
        let (c2, b2, d2, v2) = fit_at(1e-3);
        let (c3, b3, d3, v3) = fit_at(1e3);

        assert!(
            c1 && c2 && c3,
            "the same model in three unit systems must give the same convergence \
             verdict: x={c1} x/1000={c2} x*1000={c3}"
        );

        // Tolerance floor is ~1e-12 relative, not 0: `xu * 1e-3` and `xu * 1e3`
        // are rounded doubles, so the three designs are not literally the same
        // numbers even though they are the same model.
        for (b, d, s) in [(&b2, d2, 1e-3f64), (&b3, d3, 1e3f64)] {
            assert!(
                (b[0] - b1[0]).abs() <= 1e-9 * b1[0].abs().max(1.0),
                "intercept must not depend on the slope column's units: {} vs {}",
                b[0],
                b1[0]
            );
            let rescaled = b[1] * s;
            assert!(
                (rescaled - b1[1]).abs() <= 1e-9 * b1[1].abs().max(1.0),
                "slope must scale exactly with the unit change: {rescaled} vs {}",
                b1[1]
            );
            assert!(
                (d - d1).abs() <= 1e-9 * d1.abs(),
                "deviance must not depend on units: {d} vs {d1}"
            );
        }
        for v in [&v1, &v2, &v3] {
            assert!(
                v.iter().all(|q| q.is_finite()),
                "variances must be finite in every unit system: {v:?}"
            );
        }
    }

    /// The Poisson log link repeats the logistic invariance property. Under the
    /// old `|β_j| > 30` cap, `y ~ x/1000` returned `converged: false` with
    /// β̂ = (0.4915, 803.09) and NaN standard errors.
    #[test]
    fn glm_poisson_guard_is_scale_invariant() {
        let n = 200;
        let p = 2;
        let targets: Vec<u32> = vec![0, 1];
        let fit_at = |scale: f64| {
            let (x, y) = scaled_poisson_design(n, scale);
            let mut ws = TestWs::new(n, p, 0);
            let f = glm_irls_fit(
                crate::Family::Poisson {
                    link: crate::PoissonLink::Log,
                },
                f64::NAN,
                x.as_ref(),
                &y,
                &targets,
                None,
                None,
                None,
                glm_scratch(&mut ws),
            );
            (
                f.converged,
                f.betas.to_vec(),
                f.deviance,
                f.var_diag.to_vec(),
            )
        };

        let (c1, b1, d1, v1) = fit_at(1.0);
        let (c2, b2, d2, v2) = fit_at(1e-3);

        assert!(c1 && c2, "poisson: x={c1} x/1000={c2}");
        assert!(
            (b2[0] - b1[0]).abs() <= 1e-9 * b1[0].abs().max(1.0),
            "poisson intercept: {} vs {}",
            b2[0],
            b1[0]
        );
        let rescaled = b2[1] * 1e-3;
        assert!(
            (rescaled - b1[1]).abs() <= 1e-9 * b1[1].abs().max(1.0),
            "poisson slope must scale exactly: {rescaled} vs {}",
            b1[1]
        );
        assert!(
            (d2 - d1).abs() <= 1e-9 * d1.abs(),
            "poisson deviance: {d2} vs {d1}"
        );
        assert!(v1.iter().chain(v2.iter()).all(|q| q.is_finite()));
    }

    /// Under the Gamma inverse link η = 1/μ, so a legitimate small-mean fit sits
    /// far above the divergence threshold on the η scale — μ = 0.01 gives
    /// η = 100. The guard is skipped for that family/link pair; this fit pins
    /// that it is, and that the fit is still accepted.
    #[test]
    fn glm_gamma_inverse_large_eta_still_converges() {
        let n = 200;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let xu = (i as f64) / (n as f64) - 0.5;
            x[(i, 0)] = 1.0;
            x[(i, 1)] = xu;
            // η = 100 − 20·x exactly, i.e. μ ≈ 0.01: honest, and more than three
            // times the divergence threshold.
            let eta = 100.0 - 20.0 * xu;
            let jitter = 1.0 + 0.05 * (((i % 7) as f64) - 3.0) / 3.0;
            y[i] = jitter / eta;
        }
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(
            crate::Family::Gamma {
                link: crate::GammaLink::Inverse,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(
            fit.converged,
            "a small-mean Gamma inverse-link fit carries |η| ≈ 100 honestly and \
             must not be rejected as divergence"
        );
        assert!(
            (fit.betas[0] - 100.0).abs() < 5.0,
            "intercept on the 1/μ scale should land near 100, got {}",
            fit.betas[0]
        );
    }

    // -----------------------------------------------------------------
    // GLM deviance_null golden value (external oracle: R glm()$null.deviance)
    // -----------------------------------------------------------------

    #[test]
    fn glm_deviance_null_golden_value() {
        // null deviance = -2*(y_sum*ln(y_bar) + (n-y_sum)*ln(1-y_bar))
        // y_sum=40, n=100, y_bar=0.4
        // → expected = -2*(40*ln(0.4) + 60*ln(0.6)) ≈ 134.6023334
        // R: glm(c(rep(1,40),rep(0,60)) ~ 1, family=binomial)$null.deviance = 134.6023
        //
        // y pattern: every 5th group, first 2 are 1 and last 3 are 0.
        // This gives exactly 40 ones spread throughout, so there is no separation
        // between y and the linearly-increasing x predictor.
        let n = 100;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f64) / (n as f64) - 0.5; // arbitrary predictor
        }
        // y[i] = 1 for i % 5 < 2, else 0 → exactly 40 ones, not separated from x.
        let mut y = vec![0.0f64; n];
        for (i, v) in y.iter_mut().enumerate() {
            if i % 5 < 2 {
                *v = 1.0;
            }
        }

        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![1];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(fit.converged, "must converge: non-separated y pattern");

        // External oracle: R glm(y ~ x, family=binomial)$null.deviance with y_sum=40/n=100
        // null deviance = -2*(40*ln(0.4) + 60*ln(0.6)) = 134.6023334 (depends only on y_bar)
        let expected = 134.6023334f64;
        let abs_err = (fit.deviance_null - expected).abs();
        assert!(
            abs_err < 0.001,
            "deviance_null = {}, expected {expected}, err = {abs_err}",
            fit.deviance_null
        );
    }

    /// Weighted Gamma(log) null/residual deviance vs R glm(weights=).
    /// Convention: null mean is the WEIGHTED mean μ̂₀ = Σwᵢyᵢ/Σwᵢ and both
    /// deviances accumulate Σwᵢdᵢ. Same fixture as fit.rs's
    /// `fit_glm_gamma_weighted_matches_r` (β/SE/φ gated there).
    #[test]
    fn glm_weighted_deviance_null_golden_value() {
        // R 4.5.3 oracle (set.seed(42), n = 40):
        //   x1 <- round(rnorm(n), 4); w <- sample(1:4, n, replace = TRUE)
        //   eta <- 0.4 + 0.8 * x1
        //   yg <- round(rgamma(n, shape = 2, scale = exp(eta) / 2), 6)
        //   fg <- glm(yg ~ x1, family = Gamma("log"), weights = w)
        //   print(fg$null.deviance, digits = 15); print(fg$deviance, digits = 15)
        let x1: [f64; 40] = [
            1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627,
            1.3049, 2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201,
            -0.3066, -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64,
            0.4555, 0.7048, 1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
        ];
        let w: [f64; 40] = [
            4.0, 1.0, 2.0, 1.0, 1.0, 4.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 1.0, 4.0, 4.0, 2.0, 1.0,
            4.0, 2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 4.0, 3.0, 4.0, 1.0, 4.0, 1.0, 4.0, 3.0,
            2.0, 2.0, 3.0, 1.0, 1.0, 2.0,
        ];
        let y: [f64; 40] = [
            2.421196, 0.850101, 1.188318, 0.917668, 1.895064, 2.717167, 4.391082, 0.266883,
            1.853922, 1.838375, 5.959549, 19.008523, 0.121882, 1.544704, 1.422566, 0.758422,
            1.264496, 0.147806, 0.06751, 2.907132, 0.3538, 0.223494, 0.297625, 5.273375, 12.534684,
            0.514577, 1.473477, 0.485665, 0.962023, 1.043896, 1.771311, 1.926229, 7.592099,
            1.298714, 0.675125, 0.201756, 1.814679, 1.104297, 0.434436, 0.470596,
        ];
        const REF_DEV_NULL: f64 = 140.09428224081;
        const REF_DEV: f64 = 39.1211374203115;
        let n = 40;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1[i];
        }
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(
            crate::Family::Gamma {
                link: crate::GammaLink::Log,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            Some(&w),
            None,
            glm_scratch(&mut ws),
        );
        assert!(fit.converged, "weighted gamma GLM must converge");
        assert!(
            (fit.deviance_null - REF_DEV_NULL).abs() / REF_DEV_NULL < 1e-8,
            "deviance_null = {} vs R {REF_DEV_NULL}",
            fit.deviance_null
        );
        assert!(
            (fit.deviance - REF_DEV).abs() / REF_DEV < 1e-6,
            "deviance = {} vs R {REF_DEV}",
            fit.deviance
        );
    }
}
