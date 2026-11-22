//! Adaptive IRLS GLM kernel (every `Family`); outputs `z_sq_j = β̂_j² / Var(β̂_j)`
//! for hot-loop comparison against a caller-supplied precomputed squared-
//! critical-value table — no z-CDF calls, no SE sqrt, no per-coefficient
//! square roots anywhere in the loop.
//!
//! All per-fit buffers live in the caller's workspace.
//! The kernel takes a `GlmScratch<'w>` built inline at the call site (NLL split-borrow)
//! and returns a borrowed `GlmFitView<'a>`. No owned result struct.
//!
//! Algorithm — guards and tolerances:
//!   - Adaptive convergence: `|Δdeviance| / (|deviance| + f) < DEVIANCE_TOL = 1e-12`
//!     (R's `glm.fit` rule, where `f = 0.1`).
//!     The rule is relative while `|deviance| ≫ f`; the floor `f` makes it
//!     absolute once the deviance itself is tiny (an exact fit). On Gamma and
//!     inverse-Gaussian `f` is 0.1 of a deviance whose units the core has
//!     taken out of the weights and the response (`standardize::Units`), so
//!     their stopping point does not depend on either's overall scale (see
//!     `dev_floor`). On binomial, Poisson and NB `f` is R's 0.1 in units of
//!     one case: weights below one case scale it down with them.
//!   - Both stops read the step that produced the current β, `max_i |Δηᵢ|`:
//!     the fit also stops as converged when that step moved no η by more than
//!     `STEP_ULPS` ULPs of max |η|, or of the size of the terms η sums once
//!     a pass has solved below the conditioning bound (round-off), and a stop
//!     whose step still moved some η by more than `SEPARATION_STEP` is
//!     non-converged (separation; skipped under the Gamma inverse and
//!     inverse-Gaussian 1/μ² links). A stop whose round-off bound exceeds
//!     `SEPARATION_STEP` is non-converged too, and separated when some η is
//!     past `ETA_DIVERGENCE_CAP`: its step cannot show the move.
//!   - Safety cap: `MAX_IRLS_ITERS = 50`
//!   - ETA_DIVERGENCE_CAP divergence guard: a fit that ends unsettled (the
//!     safety cap, an η its link does not evaluate, a failed solve), with
//!     `‖η − η₀‖_∞ > 30` and a last step that still moves some η by more than
//!     `SEPARATION_STEP`, is separated; crossing 30 alone stops nothing.
//!     η₀ᵢ = oᵢ on the binomial links and the null model
//!     η₀ᵢ = oᵢ + ln(Σwy / Σw·e^o) on the log links (skipped under the Gamma
//!     inverse and inverse-Gaussian 1/μ² links)
//!   - A seed whose raw η falls outside the link's clamp bounds where the
//!     data pull it back (`family::eta_infeasible`), or a pass that halving
//!     cannot bring back, ends the fit non-converged
//!   - All-0 / all-1 short circuit
//!   - Step-halving on a deviance increase (above `CANONICAL_RISE` on the
//!     canonical links), a non-finite deviance or an η the link does not
//!     evaluate (R package glm2's rule; R's `glm.fit` halves only on the last
//!     two), at most `PIRLS_MAX_HALVINGS` times per pass; on a
//!     link whose observed curvature differs from the Fisher weight, steps are
//!     also halved once β has stepped back and forth three times running
//!     (period-2 damping, see the accept step)
//!   - On those links every pass with an accepted β to fall back to steps by
//!     Newton's method (observed weights), with the Fisher step as its
//!     fallback (see the Newton step); the SEs keep the Fisher factor
//!
//! Two `beta_start` modes: `None` seeds a family-specific cold start
//! (η = 0 for logit; the null-model β, or without an intercept R's
//! `initialize` μ-start, for the Gamma-inverse and inverse-Gaussian 1/μ²
//! links; the null model η₀ᵢ = oᵢ + ln(Σwy / Σw·e^o) for the log links — see
//! the cold-start arm; a log-link fit with an offset that fails from that
//! seed is run once more from R's per-row seed, and reports that run), a
//! fixed reproducible cold
//! start; `Some(spec.effect_sizes)` — what the shipped hot loop passes — seeds
//! the spec-derived truth-start (Y is synthetic, so the true β on the logit
//! scale is known; mirrors `lmm::fit_lmm`'s `theta_start`). Either way the
//! accept rule and the relative-|Δdev| fixpoint are unchanged — only the path
//! to it shortens.
//!
//! Working-response IRLS form and the canonical-link weights follow McCullagh &
//! Nelder (1989), *Generalized Linear Models*, 2nd ed., Chapman & Hall — as MN89.

use faer::linalg::matmul::triangular::BlockStructure;
use faer::linalg::matmul::{matmul, triangular};
use faer::reborrow::{IntoConst, Reborrow, ReborrowMut};
use faer::{Accum, MatMut, MatRef, Par};

use crate::fit::StopReason;
use crate::ols::{
    contrast_t_sq_of, nan_fill_ols_scratch, pair_contrast_norm_sq, triangular_solve_norm_sq,
};
use crate::spec::{BinomialLink, Family};
use crate::standardize::{ColumnScaling, Units};

/// IRLS safety cap.
pub const MAX_IRLS_ITERS: u32 = 50;
/// Adaptive convergence tolerance on the RELATIVE deviance change
/// `|Δdeviance| / (|deviance| + f)` — R's `glm.fit` rule. The floor `f`
/// keeps the rule well-behaved as deviance approaches 0 (an exact or
/// near-exact fit), the same role R's `+ 0.1` plays in its own stopping test;
/// `f` is that 0.1 carried into the deviance's units (see `dev_floor` in
/// `glm_irls_fit`).
///
/// R's own default is `1e-8`, but Fisher scoring on a NON-canonical link
/// (probit, cloglog, Gamma/log, inverse-Gaussian/log, NB/log) converges only
/// LINEARLY, not quadratically — the same mechanism `glmm::PIRLS_TOL_REL_NONCANON`
/// documents for PIRLS — so a relative tolerance loose enough for a
/// quadratically-convergent canonical link leaves a visible residual there
/// (this IRLS takes Newton steps on those links, with Fisher steps as the
/// fallback; see the Newton step).
/// The R/MASS goldens this crate gates against are not immune: they are `stats::glm`
/// at R's own default `1e-8`, and R's own last IRLS step can overshoot that
/// tolerance by a wide margin (measured on `cbpp_probit_glm`'s R trace: deviance
/// 115.6074 → 114.1043 → 114.1017 → 114.1017 over 4 iterations, the golden SEs
/// still 3.6e-7 to 9.7e-6 away from R at `epsilon = 1e-14`) — so matching R here
/// means landing at the MLE, not at R's own stopping point. Measured on this
/// crate's own R/MASS-gated goldens: at `1e-8` the worst observed gap against a
/// frozen R fit is 1.5e-4 relative (probit SE); at `1e-10` the worst gap left is
/// 6.7e-6, against a self-pinned `se_rx` value (a 1e-7 band, not an R fit); at
/// `1e-12` every R/MASS-gated golden is back inside its own band, and going one
/// decade tighter still (`1e-13`) changes nothing further on ordinary data while
/// flipping the outcome of `gamma_inverse_double_se_failure_returns_a_failed_fit`
/// (`src/fit/glmm_tests.rs`), an adversarial fixture whose cluster means span ten
/// decades. `1e-12` is the tightest of `{1e-8, 1e-10, 1e-12, 1e-13}` that clears
/// every existing golden without that regression.
pub const DEVIANCE_TOL: f64 = 1e-12;
/// Divergence verdict: a fit that ends without settling (at `MAX_IRLS_ITERS`,
/// on an η its link does not evaluate, a non-finite solve or a non-PD X'WX)
/// with some |η_i − η₀ᵢ| >
/// ETA_DIVERGENCE_CAP and the last step still moving some η by more than
/// `SEPARATION_STEP`, is running off to infinity and reports `separated`.
/// Crossing the cap alone stops nothing: a fit with a finite maximum whose η
/// passes ±30 at extreme x (logit β = 1.6 over x in ±25) settles and
/// converges as usual. η₀ᵢ is the link's null-model location (oᵢ on the
/// binomial links, oᵢ + ln r on the log links, r = Σwy / Σw·e^o the null-model rate, ȳ
/// without an offset). An offset of ±30 on a binomial row puts that row's
/// probability within 1e-13 of 0 or 1 by itself; that is the caller's model,
/// not separation, so the bound is on η − oᵢ. The bound is on the LINEAR
/// PREDICTOR, not on β: η = Xβ is unchanged when a predictor column is rescaled (the
/// compensating change in β̂ is exact), so the accept/reject decision does not
/// depend on the caller's choice of units — height in metres and height in
/// kilometres give the same verdict. 30 is the number the physical argument
/// actually supports: on the logit scale |η| = 30 is already p ≈ 1 − 1e-13,
/// so a fit still moving out there after the full budget is separation. On a
/// log link the same bound reads μᵢ/μ₀ᵢ outside `[e^−30, e^30] ≈
/// [1e-13, 1e13]`: a fitted rate that many decades off the data's own rate,
/// where a y = 0 row runs when its mean is driven to 0. Centring on oᵢ + ln r keeps the verdict free of the units of
/// the response, the weights and the offset too — Gamma/inverse-Gaussian y in
/// grams or tonnes, Poisson counts near e^30 or near 1e-20, and rows whose
/// exposure is 1e-15 of the rest all shift η̂ᵢ by the same amount as η₀ᵢ.
/// Skipped under the two
/// reciprocal links — Gamma `1/μ` and inverse-Gaussian `1/μ²` — where a large
/// |η| is an honest small-mean fit (see the guard site).
pub const ETA_DIVERGENCE_CAP: f64 = 30.0;
/// Round-off stop, in ULPs of the linear predictor: when the step that
/// produced the current β moved every η by at most
/// `STEP_ULPS · ε · max_i |ηᵢ|` (max |η| floored at 1 where the divergence
/// guard runs; from the first pass solved below the conditioning bound on,
/// the size of the terms η sums in place of max |η|, see the step sweep), the
/// fit stops as converged. At large means
/// (Poisson or NB at μ ≳ 1e5) the deviance evaluated at float β carries
/// round-off above `DEVIANCE_TOL`: a row's dᵢ moves by 2|yᵢ − μᵢ|·δηᵢ when its
/// η rounds differently, and |yᵢ − μᵢ| grows like √μ. Measured spread over
/// ±1-ULP moves of β at a 50-digit MLE: ~1e-11 relative at μ ≈ 1.6e5, ~4e-9 at
/// 6.6e7, ~1e-6 at 2.6e10. The iterates then wander within a few ULPs of the
/// MLE while the deviance rule never fires. An ordinary fit stops on the
/// deviance rule at the same pass: a step this small changes the deviance by
/// O(step²·information), far below `DEVIANCE_TOL`, unless round-off dominates.
pub const STEP_ULPS: f64 = 16.0;
/// Step-halving threshold on the canonical links, relative as in the stop
/// rule. There Fisher scoring is Newton's method, but from a far start on a
/// nearly singular X'WX a Newton step can overshoot and the fit runs off (an
/// ill-conditioned logit design: +18% on the first bad pass, then without
/// bound). The deviance's own round-off reaches about 1e-6 relative at
/// Poisson counts near 2.6e10 (see `STEP_ULPS`), and halving at
/// `DEVIANCE_TOL` moved 14 well-conditioned fits, so a canonical link halves
/// only above this.
pub const CANONICAL_RISE: f64 = 1e-3;
/// Separation stop: a deviance stop whose last step still moved some η by more
/// than this is non-converged. Without a finite maximum (complete or
/// quasi-complete separation on a binomial link, a predictor category with
/// only zero counts on Poisson or NB) the iterates walk off along a direction
/// in which the deviance only creeps down, and the separated rows' deviance
/// terms fall below `DEVIANCE_TOL` while their η still moves: by about 1 per
/// step on logit, on Poisson/NB and on the cloglog lower tail, by about 1/|η|
/// (0.14 at the stop) on probit, and by about e^{−η} on the cloglog upper
/// tail, where 1 − μ = exp(−e^η) vanishes so fast that the stop comes at
/// η ≈ 3.2 to 3.4 with steps of 0.037 to 0.040 (measured on factor-level
/// separation, 30 to 15000 rows). At the stop the m separated rows'
/// deviance, 2m·exp(−e^η), is still at least DEVIANCE_TOL times the 0.1
/// floor, so e^η ≤ 30.6 + ln m and the step e^{−η} stays above
/// 1/(30.6 + ln m): 0.02 even at m = 1e9. At a finite maximum the last
/// step is of order √DEVIANCE_TOL: on the cold-start GLM of every
/// non-Gaussian grid cell and on simulated fits on all six families, the
/// largest measured was 3.2e-5. Applied only where the divergence guard
/// runs: on the two reciprocal links η carries the response's units and an
/// absolute bound on it has no meaning.
pub const SEPARATION_STEP: f64 = 0.01;
/// Floor on the stored per-row IRLS weight `W_i = p_i (1-p_i)`, keeping it
/// strictly positive. This keeps the logit fast path's working response
/// `z = η + (y − p)/w` and the GLM's `X'WX` finite on a row where `p(1 − p)`
/// rounds to 0. The GLMM kernels never divide by `w`.
pub const WEIGHT_CLAMP: f64 = 1e-300;

/// Borrowed view into the workspace `irls_*` scratch produced by `glm_irls_fit`.
/// Lifetime ties back to the workspace.
///
/// `t_sq` field name reuses the OLS slot for uniformity — values are z² under
/// Logit (Wald-z²); the threshold comparison `stat_sq > crit_sq` is family-
/// agnostic so a caller's writeback code stays uniform.
pub struct GlmFitView<'a> {
    /// length P — fitted coefficients (NaN-filled on non-converged paths), on
    /// the design `x` held (β̃ when the fit ran through a column map, see
    /// `glm_irls_fit_mapped`). Under `se_unavailable` all finite, the columns
    /// of `broken` at their start value (0 from a cold start): those are not
    /// estimates.
    pub betas: &'a [f64],
    /// length T — `((X'WX)⁻¹)_jj` for each target, in the caller's units
    /// under a column map: `(B(X̃'WX̃)⁻¹Bᵀ)ₜₜ`.
    pub var_diag: &'a [f64],
    /// length T — Wald z² for each target. Compared against the caller's
    /// squared critical values.
    pub t_sq: &'a [f64],
    /// Cached lower-triangular Cholesky factor L of the last accepted X'WX.
    /// Valid only when `converged == true` and `se_unavailable == false`
    /// (stale-or-zero otherwise — same staleness contract as
    /// `OlsFitView::factor`).
    pub l: MatRef<'a, f64>,
    /// Number of IRLS iterations completed (0 on the short-circuit / non-converged paths).
    pub n_iter: u32,
    /// Whether the deviance fixpoint `|Δdeviance| / (|deviance| + f) < DEVIANCE_TOL`, or the
    /// round-off stop (`STEP_ULPS`), was reached before `MAX_IRLS_ITERS` without the separation
    /// stop (`SEPARATION_STEP`) firing, with a round-off bound no larger than `SEPARATION_STEP`.
    pub converged: bool,
    /// Whether the fit ended because no finite maximum seems to exist: the separation stop or
    /// the divergence guard (`ETA_DIVERGENCE_CAP`) judged the unsettled end of a binomial,
    /// Poisson or negative-binomial fit as running off to infinity. Always
    /// `false` on Gamma and inverse-Gaussian: their response has no boundary value (y > 0), so
    /// there is no separation to report, and such a fit is only non-converged.
    pub separated: bool,
    /// Final-iteration weighted deviance of the family, `Σ wᵢdᵢ` (unit
    /// deviances `dᵢ`, prior weights `wᵢ`).
    /// `NaN` on every non-converged / short-circuit return.
    pub deviance: f64,
    /// The null-model value of `deviance`: `Σ wᵢdᵢ(y, μ̂₀)` at the weighted mean
    /// `μ̂₀ = Σwᵢyᵢ/Σwᵢ`.
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
    /// converged `X'WX` to measure. `0.0` under `se_unavailable`, with
    /// `pivot_col` the first of the columns in `broken`.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
    /// The fit converged, but the QR of `√W·X` at its last solve had a zero or
    /// non-finite diagonal at the columns of `broken` while `X'WX` there is
    /// positive: each is not separated from the ones before it under the
    /// converged weights. `betas`, `deviance` and `mu` are the fit's; `var_diag`
    /// and `t_sq` are NaN, `pivot` is 0. The IRLS step along such a column is
    /// zero, so it stays at its start value. `false` on every other return.
    pub se_unavailable: bool,
    /// Under `se_unavailable`, length P with `true` at each broken column.
    /// Empty on every other return.
    pub broken: &'a [bool],
    /// Why the fit did not converge, set at the exit that stopped it: the
    /// short circuits and a first solve that failed are
    /// [`StopReason::FirstSolveFailed`] (every β NaN), every other
    /// non-converged end [`StopReason::IrlsDiverged`]. `None` when `converged`.
    pub(crate) stop: Option<StopReason>,
}

/// Caller-owned scratch borrowed from the caller's workspace field-by-field. Built
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
    /// Mask of the QR's broken columns, length `p`; read back as
    /// `GlmFitView::broken`.
    pub irls_broken: &'w mut [bool],
}

// ---------------------------------------------------------------------------
// Numerically stable helpers
// ---------------------------------------------------------------------------

/// Numerically stable sigmoid: branches on the sign of `eta` so `exp` never
/// overflows and the ratio avoids cancellation. The logit IRLS and PIRLS rows
/// compute `p` in the vectorized `simd_transcendental` kernels
/// (`pw_and_log1pexp_sum`, ≤2 ULP of this form); this libm form is the `f64`
/// and dual-number `sigmoid` of the generic `Scalar` paths and the tests'
/// reference.
#[inline]
pub(crate) fn sigmoid_stable(eta: f64) -> f64 {
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

/// Whether some row's η sits more than `ETA_DIVERGENCE_CAP` from its
/// null-model location `oᵢ + eta_null`. A NaN η counts as past the cap.
fn past_cap(eta: &[f64], offset: Option<&[f64]>, eta_null: f64) -> bool {
    eta.iter().enumerate().any(|(i, &e)| {
        let d = (e - offset.map_or(0.0, |o| o[i]) - eta_null).abs();
        d.is_nan() || d > ETA_DIVERGENCE_CAP
    })
}

/// The divergence verdict on a fit that ended without settling: some row's
/// η is past the cap (`past_cap`) and the last step still moved some η by
/// more than `SEPARATION_STEP`, so the fit is running off to infinity. A NaN
/// step is not a moving step, and neither is the infinite placeholder of a fit
/// that ended before any step was accepted (a warm start already past the cap
/// whose first solve failed is not separation).
fn runs_off(eta: &[f64], offset: Option<&[f64]>, eta_null: f64, step_max: f64) -> bool {
    step_max > SEPARATION_STEP && step_max.is_finite() && past_cap(eta, offset, eta_null)
}

/// η = o + Xβ as a column sweep (axpy over x's columns), each ηᵢ accumulated
/// in the same j order from 0.0, bit-identical to the strided per-row form.
/// The truth start, the accept step and the step-halving all form η here, so a
/// warm fit walks the arithmetic of the accept path.
fn fill_eta(x: MatRef<'_, f64>, beta: &[f64], offset: Option<&[f64]>, eta: &mut [f64]) {
    eta.fill(0.0);
    for (j, &b_j) in beta.iter().enumerate() {
        for (i, e) in eta.iter_mut().enumerate() {
            *e += x[(i, j)] * b_j;
        }
    }
    if let Some(o) = offset {
        for (e, &oi) in eta.iter_mut().zip(o) {
            *e += oi;
        }
    }
}

/// The size of the terms `fill_eta` sums, `maxᵢ (|oᵢ| + Σⱼ |xᵢⱼβⱼ|)`, in the
/// same column sweep; `buf` (length n) is scratch. ηᵢ's round-off scales with
/// it, not with |ηᵢ|: terms that cancel leave their rounding behind (Higham,
/// *Accuracy and Stability of Numerical Algorithms*, 2nd ed., §4.2).
fn term_size(x: MatRef<'_, f64>, beta: &[f64], offset: Option<&[f64]>, buf: &mut [f64]) -> f64 {
    match offset {
        Some(o) => {
            for (t, &oi) in buf.iter_mut().zip(o) {
                *t = oi.abs();
            }
        }
        None => buf.fill(0.0),
    }
    for (j, &b_j) in beta.iter().enumerate() {
        for (i, t) in buf.iter_mut().enumerate() {
            *t += (x[(i, j)] * b_j).abs();
        }
    }
    buf.iter().fold(0.0, |m, &t| m.max(t))
}

/// Fit a GLM via adaptive IRLS. All buffers live in `scratch`; the returned view
/// borrows from the same storage.
///
/// `family` selects the IRLS math, all of it behind one batched call into
/// `simd_transcendental::family_pass`. Unweighted `Family::Binomial { link:
/// Logit }` on a 0/1 response reaches the fused kernel `pw_and_log1pexp_sum`
/// there; every other family/link runs a vectorized arm written against
/// `family`'s scalar formulas. Gamma/NB dispersion is handled by the caller (`fit/glm.rs`),
/// not here — this kernel folds `φ=1`. `nb_theta` is the NB shape θ̂ the caller's
/// outer-θ loop fixes for this fit (only the NB family reads it via
/// `family::variance`/`dev_resid`; pass `f64::NAN` for every other family).
///
/// - `x`: `n × p` design (column-major faer).
/// - `y`: length `n`; proportions in [0, 1] for binomial (0/1 for Bernoulli
///   rows), counts for Poisson/NB, positive for Gamma.
/// - `target_indices`: per-coefficient indices to compute `z²` for.
/// - `beta_start`: `None` → β = 0 cold start; `Some(β₀)` (length `p`) → spec-derived truth start — seeds β and
///   computes η = X·β₀ once. A per-scenario constant, so determinism and
///   chunk merging are unaffected.
/// - `prior_w`: per-row precision weight `wᵢ`, McCullagh & Nelder's prior-
///   weight sense (MN89 §2.2.2) — `None` = unit weight. Multiplies the IRLS
///   working weight (`irls_w[i] = (wᵢ·W_raw).max(WEIGHT_CLAMP)`) and the
///   per-row deviance contribution; the working response `z = η + r` is
///   untouched (prior weights don't scale the working residual). The fused
///   `log1pexp` deviance identity holds only for unweighted Bernoulli rows, so
///   `Some(_)`, or a binomial y with a value other than 0 and 1, routes
///   `Family::Binomial { link: Logit }` to the weighted arm instead
///   (unweighted logit on a 0/1 y keeps the fused kernel). Matches R
///   `glm(weights=)` (`fit_glm_gamma_weighted_matches_r`,
///   `fit_glm_binomial_weighted_aggregated_matches_r`).
/// - `scratch`: borrowed mutable slots from the caller's workspace.
///
/// IRLS weight: `W = diag(μ(1−μ))` under canonical logit (`Family::Binomial
/// { link: Logit }`); every other family/link uses the general Fisher-scoring
/// weight `W_i = 1 / (Var(μ_i) · g'(μ_i)²)` from `family`.
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
/// `prior_w: None`, a 0/1 y) — its only external check is the null-deviance golden
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
    glm_irls_fit_mapped(
        family,
        nb_theta,
        x,
        y,
        target_indices,
        beta_start,
        prior_w,
        offset,
        scratch,
        ColumnScaling::IDENTITY,
        Units::IDENTITY,
    )
}

/// Wald z² of the pairwise contrast `β_p − β_n` on a [`glm_irls_fit`] view,
/// the GLM twin of `ols_contrast_t_sq`:
///
/// ```text
/// c = e_p − e_n,   L · u = c,   var(β_p − β_n) = ‖u‖²,   z² = (β̂_p − β̂_n)² / var
/// ```
///
/// with `L` the view's factor `l` of `X'WX`. The variance is on the view's own
/// `var_diag` scale, φ = 1: on Gamma and inverse-Gaussian, whose dispersion
/// the kernel does not estimate, a caller multiplies the variance by φ̂, that
/// is, divides the returned value by φ̂.
///
/// Returns `NaN` when the fit did not converge, under `se_unavailable`, when
/// an index is ≥ p or the two are equal, when `scratch` is shorter than p, or
/// when the variance is not finite and positive.
///
/// Valid only on a view fitted without a column map ([`glm_irls_fit`]): it
/// reads `betas` and `l` as the caller's own coefficients and information.
///
/// `scratch` must be length ≥ `p`; it is overwritten and not read on output.
pub fn glm_contrast_t_sq(fit: &GlmFitView<'_>, p_col: u32, n_col: u32, scratch: &mut [f64]) -> f64 {
    if !fit.converged || fit.se_unavailable {
        return f64::NAN;
    }
    let var = pair_contrast_norm_sq(
        fit.l,
        ColumnScaling::IDENTITY,
        fit.betas.len(),
        p_col,
        n_col,
        scratch,
    );
    // NaN here also covers the index and scratch checks, before `betas` is read.
    if var.is_nan() {
        return f64::NAN;
    }
    contrast_t_sq_of(fit.betas[p_col as usize] - fit.betas[n_col as usize], var)
}

/// [`glm_irls_fit`] on a design fitted through a column map `B`
/// (`x̃ = X·B`, `β = B·β̃`), with `y`, `prior_w` and `offset` those of `units`
/// (`Units::y`, `Units::weights`, `Units::offset`). The returned `betas` are
/// β̃ in the internal units, on the design `x` holds; `var_diag` and `t_sq`
/// are the caller's: `Var(βₜ) = v·‖L⁻¹bₜ‖²` with `bₜ` row `t` of `B` and `v` =
/// [`Units::glm_var_scale`], `t² = βₜ²/Var(βₜ)` with the caller's `βₜ`
/// ([`Units::beta_to_caller`]).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_irls_fit_mapped<'a>(
    family: Family,
    nb_theta: f64,
    x: MatRef<'_, f64>,
    y: &[f64],
    target_indices: &[u32],
    beta_start: Option<&[f64]>,
    prior_w: Option<&[f64]>,
    offset: Option<&[f64]>,
    scratch: GlmScratch<'a>,
    map: &ColumnScaling,
    units: &Units,
) -> GlmFitView<'a> {
    glm_irls_fit_seeded(
        family,
        nb_theta,
        x,
        y,
        target_indices,
        beta_start,
        prior_w,
        offset,
        scratch,
        map,
        units,
        false,
    )
}

/// The fit's inputs, fixed for the whole fit.
#[derive(Clone, Copy)]
struct IrlsInputs<'x> {
    family: Family,
    nb_theta: f64,
    x: MatRef<'x, f64>,
    y: &'x [f64],
    target_indices: &'x [u32],
    beta_start: Option<&'x [f64]>,
    prior_w: Option<&'x [f64]>,
    offset: Option<&'x [f64]>,
    map: &'x ColumnScaling,
    units: &'x Units,
    per_row_seed: bool,
    n: usize,
    p: usize,
    t: usize,
    weighted: bool,
}

/// The intercept-only model and what the loop reads off it.
struct NullModel {
    y_bar: f64,
    deviance_null: f64,
    log_link: bool,
    eta_null: f64,
    dev_floor: f64,
}

/// The IRLS loop's mutable state, carried across passes.
struct IrlsState {
    // Cholesky factor of the last accepted X'WX. `SeqLlt` allocates its
    // factor on every call (each pass's Fisher factor, and the Newton
    // factor beside it), so this one owns its storage and outlives the next
    // pass's irls_xtwx rebuild; its L is copied into `irls_l` once, after the
    // loop.
    last_chol: Option<crate::linalg::SeqLlt>,
    deviance_prev: f64,
    // Deviance at the accepted β, the step-halving's reference; ∞ while there
    // is none. `deviance_prev` skips the seed, see the stop check.
    deviance_accepted: f64,
    deviance_final: f64,
    converged: bool,
    // Whether some solve's β was accepted. Until then β is the start (0 on a
    // cold start), not an estimate.
    stepped: bool,
    had_pd_failure: bool,
    // Set when the separation stop fires or the divergence guard judges an
    // unsettled end (`runs_off`).
    diverged: bool,
    n_iter: u32,
    // The step that produced the current β, `max_i |Δηᵢ|`, and the bound of the
    // round-off stop; both read by the stop check at the top of the next pass
    // and refreshed after each accept. Pass 0 has no step behind it.
    step_max: f64,
    step_tiny: f64,
    step_sq_prev: f64,
    // Whether the accepted β came from a Newton step, its Fisher twin held in
    // `irls_betas_new` (see the Newton step).
    newton_pending: bool,
    osc_flips: usize,
    damp: bool,
    // A nearly collinear X'WX: a pivot ratio below `lmm::GRAM_PIVOT_MIN` (whose
    // comment says why that bound), or a Cholesky that fails. The Gram squares
    // cond(X), so a solve on it carries an error of about cond(X)²·ε relative
    // to β, and on `[1, a, a + δb]` at cond(X) ≈ 2e6 that moves η by 1e-3 per
    // pass: the deviance never settles and the fit runs out its budget. Such a
    // pass solves on `X̃ = X·U⁻¹` instead, `U` from a QR of √W·X at that pass's
    // weights (`lmm::DesignQr`, as the OLS route and the GLMM's Schur take it):
    // the Cholesky of X̃'WX̃ = U⁻ᵀX'WXU⁻¹ is well conditioned, and β = U⁻¹β̃.
    // From the first such pass on, every pass also solves for the step,
    // β_new − β from the working residual z − η (= z − o − Xβ), so the solve's
    // relative error falls on a step that shrinks to zero and the fixpoint is
    // the one the residual defines. The factor the SEs read is then Uᵀ·L̃, L̃
    // the Cholesky factor of X̃'WX̃. Where the QR has a zero or non-finite
    // diagonal on a column that is not zero under the weights, the column has
    // broken down: the pass takes a zero step along it, and a fit whose last
    // solve broke down settles with its SEs withheld (`se_unavailable`).
    // Where that column is zero under the weights the pass keeps the Gram's
    // own factor, or fails with it. A design that never crosses the bound
    // keeps the direct solve and allocates nothing more.
    solve_step: bool,
    // The QR; `irls_broken` holds its mask of broken columns.
    design_qr: Option<crate::lmm::DesignQr>,
    // Whether `last_chol` factors X̃'WX̃ on `design_qr`'s `U`.
    l_from_qr: bool,
    // The first broken column of the last solve pass, if any.
    broken_col: Option<usize>,
}

/// [`glm_irls_fit_mapped`]'s body. `per_row_seed` replaces the log links'
/// null-model cold seed by R's per-row one (see the cold-start arm in [`seed_eta`]); it is set
/// only on the second attempt of a failed log-link fit with an offset.
#[allow(clippy::too_many_arguments)]
fn glm_irls_fit_seeded<'a>(
    family: Family,
    nb_theta: f64,
    x: MatRef<'_, f64>,
    y: &[f64],
    target_indices: &[u32],
    beta_start: Option<&[f64]>,
    prior_w: Option<&[f64]>,
    offset: Option<&[f64]>,
    mut scratch: GlmScratch<'a>,
    map: &ColumnScaling,
    units: &Units,
    per_row_seed: bool,
) -> GlmFitView<'a> {
    let n = x.nrows();
    let p = x.ncols();
    let t = target_indices.len();

    debug_assert_eq!(n, y.len(), "glm_irls_fit: y length must match X.nrows()");
    // The fused logit deviance `2·(Σ log1pexp(η) − Σ yη)` and the closed-form
    // null deviance hold only on 0/1 rows: a fractional binomial y takes the
    // weighted arm (unit weights) instead.
    let weighted = crate::family::weighted_deviance_needed(family, &y[..n], prior_w);

    debug_assert!(
        p <= scratch.irls_betas.len(),
        "scratch sized for fewer predictors"
    );
    debug_assert!(t <= scratch.irls_var_diag.len());
    debug_assert!(n <= scratch.irls_eta.len());

    // NaN-fill on early-return / non-converged paths so callers can detect
    // non-convergence from NaN outputs alone. Successful paths overwrite
    // every populated slot.
    nan_fill_ols_scratch(
        scratch.irls_betas,
        scratch.irls_var_diag,
        scratch.irls_t_sq,
        p,
        t,
    );

    // Short-circuit: n ≤ p, or no predictors.
    if n <= p || p == 0 {
        let GlmScratch {
            irls_p,
            irls_betas,
            irls_var_diag,
            irls_t_sq,
            irls_l,
            ..
        } = scratch;
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter: 0,
            converged: false,
            separated: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
            mu: &irls_p[..n],
            pivot: f64::NAN,
            pivot_col: 0,
            se_unavailable: false,
            broken: &[],
            stop: Some(StopReason::FirstSolveFailed),
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
    // Mirrored by `crate::fit::glmm::glm_warm_start_beta` — change together.
    if matches!(family, Family::Binomial { .. }) && (y_sum <= 0.0 || y_sum >= n as f64) {
        let GlmScratch {
            irls_p,
            irls_betas,
            irls_var_diag,
            irls_t_sq,
            irls_l,
            ..
        } = scratch;
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter: 0,
            converged: false,
            separated: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
            mu: &irls_p[..n],
            pivot: f64::NAN,
            pivot_col: 0,
            se_unavailable: false,
            broken: &[],
            stop: Some(StopReason::FirstSolveFailed),
        };
    }

    let inp = IrlsInputs {
        family,
        nb_theta,
        x,
        y,
        target_indices,
        beta_start,
        prior_w,
        offset,
        map,
        units,
        per_row_seed,
        n,
        p,
        t,
        weighted,
    };
    let nm = null_model(&inp, y_sum);
    let seed_is_beta = seed_eta(&inp, &nm, &mut scratch);

    let mut st = IrlsState {
        last_chol: None,
        deviance_prev: f64::INFINITY,
        deviance_accepted: f64::INFINITY,
        deviance_final: f64::NAN,
        converged: false,
        stepped: false,
        had_pd_failure: false,
        diverged: false,
        n_iter: 0,
        step_max: f64::INFINITY,
        step_tiny: 0.0f64,
        step_sq_prev: f64::NAN,
        newton_pending: false,
        osc_flips: 0usize,
        damp: false,
        solve_step: false,
        design_qr: None,
        l_from_qr: false,
        broken_col: None,
    };
    // Period-2 damping state; see the detector at the accept step. It runs
    // only where the observed curvature differs from the Fisher weight: on the
    // other links Fisher scoring is Newton's method, whose map has a zero
    // Jacobian at the MLE. `irls_u_scratch` holds the last accepted β step; it
    // is free until the post-loop SE solve.
    let osc_guard_active = crate::family::exact_curvature_differs(family);
    if osc_guard_active {
        scratch.irls_u_scratch[..p].fill(0.0);
    }

    // On the canonical links (logit, Poisson/log) the deviance is convex in β
    // and Fisher scoring is Newton's method, so a pass rarely raises the
    // deviance; a rise there halves only above `CANONICAL_RISE`, where it is
    // an overshoot rather than round-off. An ordinary pass, and so the
    // unweighted-logit hot path, stays one family pass per IRLS pass.
    let canonical = crate::family::is_canonical(family);

    // IRLS: each pass is a weighted least squares solve
    // β_new = (X'WX)⁻¹ X'Wz on the working response z = η + r,
    // r = (y − μ)·dη/dμ.
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
    // The divergence guard reads |η − η₀|, which the two reciprocal links invert:
    // η = 1/μ under the Gamma inverse link, η = 1/μ² under the inverse-Gaussian
    // 1/μ² link. On both, η grows without bound as μ → 0, so a legitimate
    // small-mean fit sits far above the threshold (μ = 0.18 already gives
    // η = 30 under 1/μ²). Those two family/link pairs end only non-converged
    // at the other exits: an η past clamp_eta's bounds `[1e-75, 1e75]`
    // (src/family.rs), the non-finite guard on β_new and MAX_IRLS_ITERS.
    // Loop-invariant, so it is evaluated once.
    let eta_guard_active = !crate::family::is_reciprocal_link(family);

    for iter in 0..=MAX_IRLS_ITERS {
        // η = X · β is already in `irls_eta`: set from the start β before
        // the loop, then refreshed by the accept step below
        // after each β update. Reusing it here is bit-identical to recomputing
        // X·β (same β, same summation order).

        let (deviance, infeasible) = evaluate_with_halving(
            &inp,
            &nm,
            canonical,
            eta_guard_active,
            &mut scratch,
            &mut st,
        );
        // A raw η the link does not evaluate (clamped by the pass, so this
        // deviance is not the objective) that halving could not bring back, or
        // that has no accepted β to halve back to, ends the fit not converged,
        // as does a non-finite deviance that halving could not bring back.
        // On the inverse-Gaussian 1/μ² link, which skips the divergence guard,
        // such a point otherwise leads every row to the domain edge η → 0
        // (μ → ∞), where each row's deviance flattens to 1/y, and the relative
        // rule stops on that plateau and reports converged.
        if infeasible {
            break;
        }
        // A per-row or constant η seed is not Xβ (β = 0 under it), so there is
        // no β to halve back to until the first solve's β is accepted, as in glm2.
        if iter > 0 || seed_is_beta {
            st.deviance_accepted = deviance;
        }
        // z − o: the WLS below solves β from X alone, so the offset's fixed
        // contribution must leave the working response. Solving for the step,
        // the right-hand side is z − η instead: `solve_step` is only set by a
        // pass's solve, and from the first accept on η = o + Xβ.
        let step_form = st.solve_step;
        if step_form {
            for i in 0..n {
                scratch.irls_z[i] -= scratch.irls_eta[i];
            }
        } else if let Some(o) = offset {
            for i in 0..n {
                scratch.irls_z[i] -= o[i];
            }
        }

        // Adaptive early exit on the RELATIVE |Δdeviance| at the CURRENT β —
        // the one the previous pass's solve accepted. Pass 0 sees the seed β,
        // which has no prior deviance to compare against: skip the check and
        // the tracker so the first real comparison is β₂-vs-β₁.
        //
        // The step that produced this β decides what a stop means. Moving no η
        // by more than round-off (`STEP_ULPS`) stops the fit as converged even
        // when the deviance's own round-off keeps the relative rule from
        // firing. A deviance stop with the step still moving some η by more
        // than `SEPARATION_STEP` is separation, not convergence. A round-off
        // bound above `SEPARATION_STEP` means the step cannot show a move of
        // that size, so the stop cannot tell the two apart: the fit is not
        // converged, and separated when some η is past the cap. That is where a
        // fit without a finite maximum ends once a solve on a nearly singular
        // X'WX throws η out to 1e15 or beyond: η then absorbs every later step
        // (o + Xβ rounds to the same η), the step reads 0 and the deviance
        // repeats exactly, so both stops fire on a fit that is running off.
        if iter > 0 {
            st.deviance_final = deviance;
            let settled = (deviance - st.deviance_prev).abs() / (deviance.abs() + nm.dev_floor)
                < DEVIANCE_TOL;
            if settled || (st.step_max <= st.step_tiny && deviance.is_finite()) {
                if eta_guard_active && st.step_tiny > SEPARATION_STEP {
                    st.diverged = past_cap(&scratch.irls_eta[..n], offset, nm.eta_null);
                } else {
                    st.diverged = eta_guard_active && st.step_max > SEPARATION_STEP;
                    st.converged = !st.diverged;
                }
                break;
            }
            st.deviance_prev = deviance;
        }
        // Solve budget spent — the `..=` pass exists only for the check above.
        if iter == MAX_IRLS_ITERS {
            break;
        }
        st.n_iter = iter + 1;

        let Some(chol) = fisher_solve(&inp, step_form, &mut scratch, &mut st) else {
            break;
        };
        let newton_used = newton_step(&inp, osc_guard_active, step_form, &mut scratch, &mut st);

        // Stash this iteration's factor; the cached L is materialised once
        // after the loop (only the converged path reads it).
        st.last_chol = Some(chol);

        if !accept_step(
            &inp,
            osc_guard_active,
            eta_guard_active,
            newton_used,
            &mut scratch,
            &mut st,
        ) {
            break;
        }
    }

    finish(&inp, &nm, eta_guard_active, scratch, st)
}

/// The null-model deviance and location, and the floor of the relative
/// deviance rule.
fn null_model(inp: &IrlsInputs<'_>, y_sum: f64) -> NullModel {
    let IrlsInputs {
        family,
        nb_theta,
        y,
        prior_w,
        offset,
        n,
        weighted,
        ..
    } = *inp;
    // Null-model deviance — computed once, reused at the batch site for the
    // LRT. Stored in a local so the per-iteration deviance tracker doesn't
    // clobber it. The all-0 / all-1 branch in `glm_irls_fit_seeded` guarantees `y_bar ∉ {0, 1}`
    // here, so no `ln(0)` risk.
    // Intercept-only MLE is μ̂₀=ȳ for any (family, link) with unit weights; under
    // prior weights it is the WEIGHTED mean μ̂₀ = Σwᵢyᵢ/Σwᵢ (R's
    // glm(weights=)$null.deviance), so the null deviance is Σ wᵢdᵢ(y, μ̂₀).
    // Logit keeps the closed-form Bernoulli expression verbatim (byte-identity)
    // only when unweighted on a 0/1 y — a weighted or fractional fit routes
    // through the same general fold the per-iteration deviance uses (the
    // `weighted` flag), so
    // `deviance_null` stays on the same weighting convention as `deviance`.
    // (The all-0/all-1 short circuit in `glm_irls_fit_seeded` keys off raw y, which is equivalent
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
        } if !weighted => -2.0 * (y_sum * y_bar.ln() + (n as f64 - y_sum) * (1.0 - y_bar).ln()),
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

    // Null-model location of the link, η₀ᵢ = oᵢ + `eta_null`: on the log links
    // `eta_null` = ln r with r = Σwᵢyᵢ / Σwᵢe^{oᵢ} (ln ȳ without an offset), 0 on
    // every other link. r is the Poisson intercept-only MLE with that offset, a
    // weighted mean of the per-row rates yᵢ/e^{oᵢ}, so it moves with the units of
    // y, of the weights and of the offset exactly as the fitted means do. The
    // log links' cold start seeds every row at η₀ᵢ, and the divergence guard
    // measures |ηᵢ − η₀ᵢ| from it. Not floored: y > 0 on Gamma and
    // inverse-Gaussian, and a floor such as R's `y + 0.1` would put a units limit
    // back into both the seed and the guard. Only an all-zero count response
    // (ȳ = 0, ln r = −∞) takes ȳ = 0.1: its maximum sits at μ = 0, and there is
    // no data scale left to be free of.
    let log_link = crate::family::is_log_link(family);
    let eta_null = if log_link {
        let mut ln_rate = if y_bar > 0.0 { y_bar } else { 0.1 }.ln();
        if let Some(o) = offset {
            // ln r = ln ȳ − ln(Σwᵢe^{oᵢ} / Σwᵢ), ȳ the weighted mean above.
            let (mut we_sum, mut w_sum) = (0.0, 0.0);
            for (i, &oi) in o[..n].iter().enumerate() {
                let pw = prior_w.map_or(1.0, |w| w[i]);
                we_sum += pw * oi.exp();
                w_sum += pw;
            }
            ln_rate -= (we_sum / w_sum).ln();
        }
        ln_rate
    } else {
        0.0
    };

    // Floor of the relative deviance rule: R's 0.1. The floor is an absolute
    // deviance, so a fixed 0.1 stops a fit early whenever the deviance is
    // small only because of units. On Gamma and inverse-Gaussian the prior
    // weights are precision weights with no scale of their own, and
    // D = Σ wᵢdᵢ scales with them; the inverse-Gaussian unit deviance
    // (y−μ)²/(yμ²) also scales as 1/y. The core fits both on internal units
    // (`standardize::Units`: `ŵ = w/s` and `y/c`, geometric means near 1), so
    // 0.1 is in those units there. On raw data (the loop tier's
    // `glm_irls_fit`) it is R's 0.1: an inverse-Gaussian/log fit with D = 19
    // at y × 1e3 stops with β 1e-6 (relative) short of the maximum, reported
    // converged. On binomial,
    // Poisson and NB, with φ ≡ 1, the deviance is a
    // likelihood-ratio statistic of counts, and R's 0.1 is in units of one
    // case: weights of a case or more (trial counts, frequency weights) keep
    // it exactly, and weights below one case scale it down with them
    // (`case_scale`). With every weight 1e-12 the fixed 0.1
    // stopped a binomial fit after its first compared pass, read as
    // separation. Poisson and NB also carry ȳ's units below ȳ = 1: a
    // non-integer y in tiny units (a rate without exposure weights) has
    // D ∝ ȳ, and R's 0.1 stopped such a fit 1.7e-2 short in β at y × 1e-13,
    // reported converged. Not a plain 0.1·ȳ: an NB deviance with θ ≪ ȳ does
    // not grow with ȳ, and at ȳ = 1e6 that floor would ask |ΔD| < 1e-7 of a
    // linearly converging fit. An all-zero count response (ȳ = 0) keeps
    // 0.1·`case_scale`. Unweighted and off ȳ < 1 counts,
    // the floor is exactly 0.1.
    let case_scale = crate::family::weight_scale(prior_w, n).min(1.0);
    let dev_floor = match family {
        Family::Gamma { .. } | Family::InverseGaussian { .. } => 0.1,
        Family::Poisson { .. } | Family::NegativeBinomial { .. } if y_bar > 0.0 => {
            0.1 * y_bar.min(1.0) * case_scale
        }
        _ => 0.1 * case_scale,
    };
    NullModel {
        y_bar,
        deviance_null,
        log_link,
        eta_null,
        dev_floor,
    }
}

/// Seed β and η; returns whether η = o + Xβ holds at the seed (`seed_is_beta`).
fn seed_eta(inp: &IrlsInputs<'_>, nm: &NullModel, scratch: &mut GlmScratch<'_>) -> bool {
    let IrlsInputs {
        family,
        x,
        y,
        beta_start,
        offset,
        per_row_seed,
        n,
        p,
        ..
    } = *inp;
    let NullModel {
        y_bar,
        log_link,
        eta_null,
        ..
    } = *nm;
    let GlmScratch {
        irls_eta,
        irls_betas,
        ..
    } = scratch;
    // Seed β and η here; the IRLS loop carries η forward from each
    // iteration's post-accept recompute (the β-accept step) instead of
    // recomputing X·β at the top of every iteration. `seed_is_beta`: η = o + Xβ
    // holds at the seed, so its deviance is the step-halving's first reference.
    let mut seed_is_beta = beta_start.is_some();
    match beta_start {
        // Truth start: β ← β₀, η = X·β₀ computed once (O(np)). The η loop
        // matches the β-accept step's summation order (column-sweep axpy) so a warm fit
        // walks the same arithmetic the accept path uses — change together.
        Some(b0) => {
            debug_assert_eq!(b0.len(), p, "beta_start length must match X.ncols()");
            irls_betas[..p].copy_from_slice(&b0[..p]);
            fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
        }
        // Cold start: β ← 0 with a family-specific η seed. Logit keeps η = 0
        // (μ=0.5). The Gamma
        // **inverse** link cannot start at η=0 (μ=1/0): seed η = 1/y per row
        // (R's `mustart=y`, `etastart=1/y`). Log-link families (Poisson/NB,
        // Gamma, inverse-Gaussian) seed the null model, η₀ᵢ = oᵢ + ln r
        // (`eta_null`): from
        // η = 0 (μ=1) the first WLS step overshoots on data whose mean is far
        // from 1 and IRLS runs away (β → ~9e304). The seed carries the
        // offset for the same reason: rows whose exposure is 1e-6 of the rest,
        // seeded at the data's mean, send a Gamma/log fit off to ~1e304. NOT
        // R's per-row μ₀ = y + 0.1 `initialize`: on
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
                // The reciprocal links, η = 1/μ (Gamma) and η = 1/μ²
                // (inverse-Gaussian): η = 0 puts μ at ∞ and η ≤ 0 is outside
                // the domain. With an intercept column the seed is the
                // null-model β (intercept link(ȳ), every slope 0), a real β:
                // its deviance is accepted at pass 0, so a first Fisher step
                // that leaves the domain is halved back toward it like any
                // later step. On some data that first step leaves the domain
                // while the maximum is interior (R's `glm` stops there with
                // "no valid set of coefficients" from its own start). Without
                // an intercept, a column whose entries all share one sign
                // gives a β in the domain the same way: βⱼ = link(ȳ)/x̄ⱼ, so
                // ηᵢ = link(ȳ)·xᵢⱼ/x̄ⱼ > 0, link(ȳ) at the column's mean (the
                // intercept is the case xᵢⱼ = 1). With neither, no β is known
                // to lie in the domain (there may be no β with Xβ > 0 on every
                // row). Then, and when o + Xβ at that β leaves the domain on
                // some row (an offset can put it there), η seeds per row at μ₀ = y (R's
                // `mustart = y`) with β = 0. y > 0 on both families, so
                // neither seed floors y, which would make the path depend on
                // y's units; the link's own η bounds keep the per-row seed in
                // its domain.
                _ if crate::family::is_reciprocal_link(family) => {
                    let link = |mu: f64| match family {
                        Family::Gamma { .. } => 1.0 / mu,
                        _ => 1.0 / (mu * mu),
                    };
                    let one_sign = |j: usize| {
                        (0..n).all(|i| x[(i, j)] > 0.0) || (0..n).all(|i| x[(i, j)] < 0.0)
                    };
                    let col_mean = |j: usize| (0..n).map(|i| x[(i, j)]).sum::<f64>() / n as f64;
                    let seed_col = (0..p)
                        .find(|&j| (0..n).all(|i| x[(i, j)] == 1.0))
                        .map(|j| (j, 1.0))
                        .or_else(|| (0..p).find(|&j| one_sign(j)).map(|j| (j, col_mean(j))));
                    if let Some((j, x_bar)) = seed_col {
                        irls_betas[j] = link(y_bar) / x_bar;
                        fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
                        seed_is_beta = (0..n)
                            .all(|i| !crate::family::eta_infeasible(family, y[i], irls_eta[i]));
                    }
                    if !seed_is_beta {
                        irls_betas[..p].fill(0.0);
                        for i in 0..n {
                            irls_eta[i] = crate::family::clamp_eta(family, link(y[i]));
                        }
                    }
                }
                // The log links seed the null model η₀ᵢ = oᵢ + ln r (`eta_null`),
                // the binomial links η₀ = 0 with no offset. From η = 0 (μ = 1) the Fisher
                // weight is 1 on Gamma-log, so the first step is ordinary least
                // squares of y − 1 on X; at a mean of ~150 that puts the intercept
                // near 150, the rows with the smallest η then carry working
                // responses of order y·e^{−η}, and IRLS runs off to ~1e306. R
                // starts both families at μ₀ = y (`Gamma()$initialize`,
                // `inverse.gaussian()$initialize`).
                //
                // The second attempt of a failed log-link fit with an offset
                // takes R's per-row seed instead, ηᵢ = ln μ₀ᵢ with no offset
                // (`glm.fit`'s `eta <- linkfun(mustart)`): μ₀ = y + 0.1 on
                // Poisson and NB (`poisson()$initialize`), μ₀ = y on Gamma and
                // inverse-Gaussian (y > 0 there). One row with an offset far
                // above the rest dominates Σw·e^o, and the null-model seed then
                // puts every other row many log units from its data.
                _ if per_row_seed => {
                    let count = matches!(
                        family,
                        Family::Poisson { .. } | Family::NegativeBinomial { .. }
                    );
                    for i in 0..n {
                        irls_eta[i] = if count { y[i] + 0.1 } else { y[i] }.ln();
                    }
                }
                _ => {
                    irls_eta[..n].fill(eta_null);
                    if let (true, Some(o)) = (log_link, offset) {
                        for i in 0..n {
                            irls_eta[i] += o[i];
                        }
                    }
                }
            }
        }
    }
    seed_is_beta
}

/// The step-bound update after a step: the round-off bound `step_tiny` from
/// the sweep's `e_max = max |η|`, or from `term_size` on a pass that solves for
/// the step, floored at 1 where the divergence guard runs (see the accept
/// step). Returns `(step_max, step_tiny)`; `buf` (length n) is scratch.
#[allow(clippy::too_many_arguments)]
fn step_bound(
    s_max: f64,
    mut e_max: f64,
    solve_step: bool,
    eta_guard_active: bool,
    x: MatRef<'_, f64>,
    betas: &[f64],
    offset: Option<&[f64]>,
    buf: &mut [f64],
) -> (f64, f64) {
    if solve_step {
        e_max = term_size(x, betas, offset, buf);
    }
    if eta_guard_active {
        e_max = e_max.max(1.0);
    }
    (s_max, STEP_ULPS * f64::EPSILON * e_max)
}

/// One IRLS pass's family evaluation at the current β, with the step-halving.
/// Returns the deviance and whether the pass's raw η was infeasible.
fn evaluate_with_halving(
    inp: &IrlsInputs<'_>,
    nm: &NullModel,
    canonical: bool,
    eta_guard_active: bool,
    scratch: &mut GlmScratch<'_>,
    st: &mut IrlsState,
) -> (f64, bool) {
    let IrlsInputs {
        family,
        nb_theta,
        x,
        y,
        prior_w,
        offset,
        n,
        p,
        weighted,
        ..
    } = *inp;
    let dev_floor = nm.dev_floor;
    let GlmScratch {
        irls_eta,
        irls_p,
        irls_w,
        irls_z,
        irls_betas,
        irls_betas_new,
        irls_xtwz,
        ..
    } = scratch;
    let IrlsState {
        deviance_accepted,
        newton_pending,
        solve_step,
        step_max,
        step_tiny,
        ..
    } = st;
    // p, W, z, and the deviance for this β — one batched call into the shared
    // family kernel (`simd_transcendental::family_pass`), which owns the
    // clamps, the inverse link, the Fisher weight with its prior-weight
    // multiply and `WEIGHT_CLAMP` floor, the working response `z = η + r`
    // (prior weights don't scale it — MN89 §2.2.2), and the `Σ wᵢdᵢ` fold.
    // `Σ y·η` is only read by the unweighted-Bernoulli-logit deviance
    // identity `2·(Σ log1pexp(η) − Σ y·η)`; it is accumulated here, on that
    // arm only (0 elsewhere), because that arm's kernel streams η without the
    // response. `fused` is `family_pass`'s test for that arm — change together.
    //
    // Step-halving (Marschner 2011, R package glm2), in PIRLS's
    // retrospective form: when the deviance at the β the last solve
    // produced rises above the accepted one by more than the stop rule's
    // band (`CANONICAL_RISE` on the canonical links), is not finite, or
    // comes from a raw η the link does not evaluate, β moves half way back
    // toward the accepted β (held in `irls_xtwz`, free until this pass's
    // GEMM) and the pass is redone, up to `PIRLS_MAX_HALVINGS` times. Fisher steps raise the deviance where
    // the observed curvature is far from the expected one: an
    // inverse-Gaussian/log fit from the null-model start can step from
    // deviance 32 to 37 to 79 to 1575 and run off to β ~ 1e74 while a
    // finite maximum exists. A finite rise that outlasts every halving
    // within the PIRLS band (`glmm::pirls_tol`) puts β back at the accepted
    // β, and the round-off stop ends the fit there as converged: the Fisher
    // step is a descent direction, so a deviance still that close above
    // after a 2⁻¹⁶ step is the deviance's own round-off, which at large NB
    // counts sits above `DEVIANCE_TOL` (see `STEP_ULPS`). A larger leftover
    // rise is kept: the solve itself lost the direction (an X'WX near
    // singular under its weights) and the fit goes on. A halved step moves every η
    // by half as much, so `step_max` halves with it. A fit whose deviance
    // never rises is untouched.
    let fused = matches!(
        family,
        Family::Binomial {
            link: BinomialLink::Logit
        }
    ) && !weighted;
    let mut halvings = 0;
    loop {
        let mut yeta = 0.0;
        if fused {
            for i in 0..n {
                yeta += y[i] * irls_eta[i];
            }
        }
        let (dev, inf) = crate::simd_transcendental::family_pass(
            family,
            nb_theta,
            &mut irls_eta[..n],
            &y[..n],
            // Sliced to n so the kernel's SIMD head/tail split lines up with
            // η's; an empty slice is its "unit weights" spelling.
            prior_w.map_or(&[][..], |w| &w[..n]),
            weighted,
            yeta,
            &mut irls_p[..n],
            &mut irls_w[..n],
            &mut irls_z[..n],
        );
        let rise = (dev - *deviance_accepted) / (dev.abs() + dev_floor);
        let rise_tol = if canonical {
            CANONICAL_RISE
        } else {
            DEVIANCE_TOL
        };
        let worse = inf || !dev.is_finite() || rise > rise_tol;
        if !worse || !deviance_accepted.is_finite() {
            break (dev, inf);
        }
        // A Newton step that raised the deviance gives way to the Fisher
        // step from the same accepted β (in `irls_betas_new`, see the
        // accept step), which then halves as usual: far from the maximum,
        // where a row's W_obs is near 0 (Gamma/log at μ ≫ y), the Newton
        // step can be 1e7 long, past what the halvings bring back.
        if *newton_pending && irls_betas_new[..p].iter().all(|b| b.is_finite()) {
            *newton_pending = false;
            irls_betas[..p].copy_from_slice(&irls_betas_new[..p]);
            fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
            let (mut s_max, mut e_max) = (0.0f64, 0.0f64);
            for i in 0..n {
                let mut d = 0.0;
                for j in 0..p {
                    d += x[(i, j)] * (irls_betas[j] - irls_xtwz[j]);
                }
                if d.is_nan() || d.abs() > s_max {
                    s_max = d.abs();
                }
                e_max = e_max.max(irls_eta[i].abs());
            }
            // `irls_z` is free: the redone pass rewrites it.
            (*step_max, *step_tiny) = step_bound(
                s_max,
                e_max,
                *solve_step,
                eta_guard_active,
                x,
                &irls_betas[..p],
                offset,
                &mut irls_z[..n],
            );
            continue;
        }
        // The halving cap and the band are PIRLS's (`glmm::PIRLS_MAX_HALVINGS`,
        // `glmm::pirls_tol`), with no measurement of their own here: a change
        // to either moves GLM convergence verdicts too — change together.
        if halvings == crate::glmm::PIRLS_MAX_HALVINGS {
            // `step_max > 0`: the redone pass at the accepted β breaks
            // here even when its deviance is not bit-identical.
            if rise <= crate::glmm::pirls_tol(family) && *step_max > 0.0 {
                irls_betas[..p].copy_from_slice(&irls_xtwz[..p]);
                fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
                *step_max = 0.0;
                continue;
            }
            break (dev, inf || !dev.is_finite());
        }
        halvings += 1;
        for j in 0..p {
            irls_betas[j] = 0.5 * (irls_betas[j] + irls_xtwz[j]);
        }
        fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
        *step_max *= 0.5;
    }
}

/// The pass's Fisher solve: X'WX and X'Wz, the switch to the QR of √W·X below
/// the pivot bound, the broken columns, and β_new into `irls_betas_new`.
/// Returns the factor, or `None` (and `had_pd_failure`) when it fails.
fn fisher_solve(
    inp: &IrlsInputs<'_>,
    step_form: bool,
    scratch: &mut GlmScratch<'_>,
    st: &mut IrlsState,
) -> Option<crate::linalg::SeqLlt> {
    let IrlsInputs { x, n, p, .. } = *inp;
    let GlmScratch {
        irls_w,
        irls_z,
        irls_betas,
        irls_betas_new,
        irls_xtwx,
        irls_xtwz,
        irls_wx,
        irls_broken,
        ..
    } = scratch;
    let IrlsState {
        had_pd_failure,
        solve_step,
        design_qr,
        l_from_qr,
        broken_col,
        ..
    } = st;
    let mut irls_xtwx = irls_xtwx.rb_mut();

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

    // Cholesky of X'WX on the lower triangle. `SeqLlt` returns an owned
    // factor (it does not consume irls_xtwx, which is
    // rebuilt from scratch each iter anyway). The factor drives the in-place
    // β solve below; its L is materialised once after the loop. A factor
    // that fails or whose pivot ratio is below the bound hands the solve to
    // `X̃ = X·U⁻¹` (see `solve_step`); the ratio is read before any solve,
    // so a pass at or above the bound runs the direct solve unchanged.
    let gram_chol = crate::linalg::SeqLlt::new(irls_xtwx.rb());
    let below = gram_chol
        .as_ref()
        .is_none_or(|c| crate::ols::min_pivot_ratio(c.L(), p).0 < crate::lmm::GRAM_PIVOT_MIN);
    // `irls_betas_new` is free until the copy below: the Givens and row scratch.
    // A column whose `R_jj` is zero or not finite while `(X'WX)_jj` is
    // positive and finite has broken down: the pass solves on `X̃` with a
    // zero step along it (below). A broken column whose `(X'WX)_jj` is zero
    // or not finite (a column zero under the weights) keeps the Gram's path.
    let w = &irls_w[..n];
    *l_from_qr = false;
    let mut breakdown = None;
    if below {
        let broken = &mut irls_broken[..p];
        let qr = design_qr.get_or_insert_with(|| crate::lmm::DesignQr::identity(p));
        if qr.set_from_rows_marking(x, |i| w[i].sqrt(), irls_betas_new, broken) {
            *l_from_qr = true;
        } else if (0..p).all(|j| {
            let g = irls_xtwx[(j, j)];
            !broken[j] || (g.is_finite() && g > 0.0)
        }) {
            *l_from_qr = true;
            breakdown = broken.iter().position(|&b| b);
        }
    }
    let chol = if let (true, Some(qr)) = (*l_from_qr, &design_qr) {
        // √W·X̃ into `irls_wx` (W∘X is spent once X'WX and X'Wz are
        // formed), its Gram and X̃'Wz = (√W·X̃)'(√W·z); `irls_z` is dead
        // after this right-hand side.
        {
            let wx = &mut irls_wx[..n * p];
            let row = &mut irls_betas_new[..p];
            for i in 0..n {
                let sw = w[i].sqrt();
                for (j, r) in row.iter_mut().enumerate() {
                    *r = sw * x[(i, j)];
                }
                qr.solve_row(row);
                for (j, &r) in row.iter().enumerate() {
                    wx[j * n + i] = r;
                }
                irls_z[i] *= sw;
            }
        }
        let wxt = MatRef::from_column_major_slice(&irls_wx[..n * p], n, p);
        triangular::matmul(
            irls_xtwx.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            wxt.transpose(),
            BlockStructure::Rectangular,
            wxt,
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        matmul(
            MatMut::from_column_major_slice_mut(&mut irls_xtwz[..p], p, 1),
            Accum::Replace,
            wxt.transpose(),
            MatRef::from_column_major_slice(&irls_z[..n], n, 1),
            1.0,
            Par::Seq,
        );
        // The zero step along a broken column `j`: row and column `j` of
        // X̃'WX̃ become `e_j` (the column of X̃ is rounding there), and the
        // right-hand side is 0 for the step, or β̃_j = β_j (row `j` of `U`
        // is `e_j`) for β, so the solve leaves β_j where it is.
        if breakdown.is_some() {
            for j in (0..p).filter(|&j| irls_broken[j]) {
                for k in 0..p {
                    irls_xtwx[(j, k)] = 0.0;
                    irls_xtwx[(k, j)] = 0.0;
                }
                irls_xtwx[(j, j)] = 1.0;
                irls_xtwz[j] = if step_form { 0.0 } else { irls_betas[j] };
            }
        }
        crate::linalg::SeqLlt::new(irls_xtwx.rb())
    } else {
        gram_chol
    };
    let Some(chol) = chol else {
        *had_pd_failure = true;
        return None;
    };
    *solve_step |= below;
    *broken_col = breakdown;

    // Solve β_new = L⁻ᵀ L⁻¹ · X'Wz using chol.solve_in_place.
    // First write xtwz into a 1-col view; chol.solve_in_place expects a
    // MatMut. Since irls_xtwz is &mut [f64] of length p, build a
    // temporary MatMut via faer's `from_column_major_slice_mut`.
    {
        let mut rhs = MatMut::from_column_major_slice_mut(irls_xtwz, p, 1usize);
        chol.solve_in_place(rhs.rb_mut());
    }
    if let (true, Some(qr)) = (*l_from_qr, &design_qr) {
        qr.map_beta(&mut irls_xtwz[..p]);
    }
    irls_betas_new[..p].copy_from_slice(&irls_xtwz[..p]);
    Some(chol)
}

/// The Newton step on a link whose exact curvature differs from Fisher, and
/// β_new in `irls_betas_new` (the Fisher β in `irls_xtwz` when it was used).
/// Returns whether the Newton step was used.
fn newton_step(
    inp: &IrlsInputs<'_>,
    osc_guard_active: bool,
    step_form: bool,
    scratch: &mut GlmScratch<'_>,
    st: &mut IrlsState,
) -> bool {
    let IrlsInputs {
        family,
        nb_theta,
        x,
        y,
        prior_w,
        offset,
        n,
        p,
        ..
    } = *inp;
    let GlmScratch {
        irls_eta,
        irls_p,
        irls_w,
        irls_z,
        irls_betas,
        irls_betas_new,
        irls_xtwx,
        irls_xtwz,
        irls_wx,
        ..
    } = scratch;
    let IrlsState {
        deviance_accepted,
        broken_col: breakdown,
        ..
    } = st;
    let mut irls_xtwx = irls_xtwx.rb_mut();

    // Newton step on a link whose exact curvature differs from Fisher:
    // the same WLS on the observed weight `W_obs` (`family::observed_weight`)
    // and the score ρ (`family::row_score`), right-hand side
    // X'(W_obs·(η − o) + ρ), or X'ρ when solving for the step. Fisher
    // scoring converges only linearly there, and on a cloglog y = 0 row
    // past η ≈ 6.6 its weight underflows while the score is −e^η, so its
    // working response overflows; `W_obs` is e^η on that row. The SEs keep
    // the Fisher factor above (expected information, as `glm.fit`), and an
    // X'W_obs X that is not positive definite (an inverse-Gaussian/log
    // row with μ > 2y has W_obs < 0) keeps the Fisher step. Fisher W and
    // z are dead once solved: `irls_w` takes W_obs, `irls_z` the
    // right-hand side's rows. Mirrors the PIRLS Newton step
    // (`glmm::pirls::observed_weights_in_place`).
    // Not on a pass with a broken column: the raw Gram moves it.
    let mut newton_used = false;
    if osc_guard_active && deviance_accepted.is_finite() && breakdown.is_none() {
        // Probit takes φ(η) from `ProbitRows`' SIMD `exp` tile, not a libm `exp` per row.
        let mut probit = crate::family::ProbitRows::new(family, &irls_eta[..n], &irls_p[..n]);
        for i in 0..n {
            let pw = prior_w.map_or(1.0, |w| w[i]);
            let (e, mu) = (irls_eta[i], irls_p[i]);
            let (wo, rho) = match probit.as_mut() {
                Some(rows) => (
                    rows.observed_weight(i, y[i], pw, irls_w[i]),
                    rows.score(i, y[i], pw),
                ),
                None => (
                    crate::family::observed_weight(family, nb_theta, y[i], pw, e, mu, irls_w[i]),
                    crate::family::row_score(family, nb_theta, y[i], pw, e, mu),
                ),
            };
            irls_z[i] = if step_form {
                rho
            } else {
                wo * (e - offset.map_or(0.0, |o| o[i])) + rho
            };
            irls_w[i] = wo;
        }
        let wx = &mut irls_wx[..n * p];
        for j in 0..p {
            let wxj = &mut wx[j * n..(j + 1) * n];
            for i in 0..n {
                wxj[i] = irls_w[i] * x[(i, j)];
            }
        }
        triangular::matmul(
            irls_xtwx.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Replace,
            x.transpose(),
            BlockStructure::Rectangular,
            MatRef::from_column_major_slice(&irls_wx[..n * p], n, p),
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        if let Some(newton) = crate::linalg::SeqLlt::new(irls_xtwx.rb()) {
            let mut rhs = MatMut::from_column_major_slice_mut(&mut irls_xtwz[..p], p, 1);
            matmul(
                rhs.rb_mut(),
                Accum::Replace,
                x.transpose(),
                MatRef::from_column_major_slice(&irls_z[..n], n, 1),
                1.0,
                Par::Seq,
            );
            newton.solve_in_place(rhs);
            newton_used = irls_xtwz[..p].iter().all(|b| b.is_finite());
        }
    }
    if step_form {
        for j in 0..p {
            irls_betas_new[j] += irls_betas[j];
            if newton_used {
                irls_xtwz[j] += irls_betas[j];
            }
        }
    }
    // The Newton β goes to `irls_betas_new`, the Fisher β to `irls_xtwz`.
    if newton_used {
        for j in 0..p {
            std::mem::swap(&mut irls_betas_new[j], &mut irls_xtwz[j]);
        }
    }
    newton_used
}

/// Accept β_new: the non-finite guard, the period-2 detector, the accept with
/// its damping, η at the new β and the step bound. Returns false when β_new is
/// not finite.
fn accept_step(
    inp: &IrlsInputs<'_>,
    osc_guard_active: bool,
    eta_guard_active: bool,
    newton_used: bool,
    scratch: &mut GlmScratch<'_>,
    st: &mut IrlsState,
) -> bool {
    let IrlsInputs {
        x, offset, n, p, ..
    } = *inp;
    let GlmScratch {
        irls_eta,
        irls_z,
        irls_betas,
        irls_betas_new,
        irls_u_scratch,
        irls_xtwz,
        ..
    } = scratch;
    let IrlsState {
        stepped,
        step_max,
        step_tiny,
        step_sq_prev,
        newton_pending,
        osc_flips,
        damp,
        solve_step,
        ..
    } = st;
    // Non-finite guard on β_new.
    let mut all_finite = true;
    for &b in &irls_betas_new[..p] {
        if !b.is_finite() {
            all_finite = false;
            break;
        }
    }
    if !all_finite {
        return false;
    }

    // Period-2 detector on the β step, with the PIRLS constants
    // (`crate::glmm::PIRLS_OSC_RATIO`): after `PIRLS_OSC_TRIGGER` passes in
    // a row whose step reverses the one before (negative dot product) at no
    // less than `PIRLS_OSC_RATIO` of its length, every later step is halved.
    // Fisher scoring's map has Jacobian I − F⁻¹H at the MLE (F expected, H
    // observed information), so where H exceeds 2F in some direction the
    // MLE repels and the iterates step back and forth across it with
    // growing amplitude (NB/log: per row H/F = (y + θ)/(μ + θ)); the
    // deviance then rises on every pass instead of alternating, which is
    // why the step, not the deviance, is watched. At very large counts the
    // iterates can also two-cycle between neighbouring floats at the MLE,
    // the deviance's round-off (NB rows cancel terms of order y) sitting
    // above `DEVIANCE_TOL`. Halving maps an eigenvalue λ of the undamped map
    // to (1 + λ)/2. A fit the detector never fires on is bit-identical.
    if osc_guard_active && !*damp {
        let (mut dot, mut step_sq) = (0.0, 0.0);
        for j in 0..p {
            let step = irls_betas_new[j] - irls_betas[j];
            dot += step * irls_u_scratch[j];
            step_sq += step * step;
            irls_u_scratch[j] = step;
        }
        let ratio = crate::glmm::PIRLS_OSC_RATIO;
        if dot < 0.0 && step_sq > ratio * ratio * *step_sq_prev {
            *osc_flips += 1;
        } else {
            *osc_flips = 0;
        }
        *damp = *osc_flips >= crate::glmm::PIRLS_OSC_TRIGGER;
        *step_sq_prev = step_sq;
    }
    // Accept β_new; the next pass halves it back if the deviance rises.
    // The accepted β goes to `irls_xtwz` for that: the solve result is
    // already in `irls_betas_new`, and the next GEMM rewrites it. Once the
    // detector has fired, every step moves half way from the current β.
    // After a Newton step the Fisher β, moved the same way, waits in
    // `irls_betas_new` for the halving loop's fallback.
    if newton_used {
        for j in 0..p {
            let (old, fisher) = (irls_betas[j], irls_xtwz[j]);
            irls_xtwz[j] = old;
            if *damp {
                irls_betas[j] += 0.5 * (irls_betas_new[j] - old);
                irls_betas_new[j] = old + 0.5 * (fisher - old);
            } else {
                irls_betas[j] = irls_betas_new[j];
                irls_betas_new[j] = fisher;
            }
        }
    } else {
        irls_xtwz[..p].copy_from_slice(&irls_betas[..p]);
        if *damp {
            for j in 0..p {
                irls_betas[j] += 0.5 * (irls_betas_new[j] - irls_betas[j]);
            }
        } else {
            irls_betas[..p].copy_from_slice(&irls_betas_new[..p]);
        }
    }
    *newton_pending = newton_used;
    *stepped = true;
    // The old η goes to `irls_z` for the step sweep below: z is dead once
    // X'Wz has been formed, and the next pass's family kernel rewrites it.
    irls_z[..n].copy_from_slice(&irls_eta[..n]);
    // (The binomial and per-row reciprocal-link cold seeds omit o, like R's
    // η₀ = link(mustart): the seed is a start, not a fixpoint constraint;
    // consistency begins at iter 1.)
    fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
    // The accepted step `max_i |Δηᵢ|` and the round-off bound. A NaN Δη
    // sticks in `step_max` (`d > s_max` is false for NaN, hence its own
    // arm), so neither stop reads a NaN η as a small step. The bound is
    // relative to max |η|, floored at 1 where the divergence guard runs:
    // there η is unit-free up to a shift (a probability scale, or ln μ), so
    // |η| ≈ 0 is not small. On the two reciprocal links η carries the
    // response's units, and a floor would turn into a unit-dependent
    // absolute bound. On a pass that solves for the step (see
    // `solve_step`) the bound is relative to the size of the terms η sums
    // (`term_size`) instead: on `[1, a, a + δb]` at δ = 2⁻²⁷ η sums terms
    // near 1e8 that cancel to order 1, so η carries about 1e-8 of rounding,
    // which moves the deviance by more than `DEVIANCE_TOL` from pass to
    // pass while max |η| puts the bound near 1e-14. A fit that never
    // crosses the bound keeps max |η| and skips the extra sweep: with a
    // weighted pivot ratio above 1e-3 its columns are far from collinear,
    // so its terms cannot cancel by much.
    let (mut s_max, mut e_max) = (0.0f64, 0.0f64);
    for i in 0..n {
        let d = (irls_eta[i] - irls_z[i]).abs();
        if d.is_nan() || d > s_max {
            s_max = d;
        }
        e_max = e_max.max(irls_eta[i].abs());
    }
    // The old η in `irls_z` is spent.
    (*step_max, *step_tiny) = step_bound(
        s_max,
        e_max,
        *solve_step,
        eta_guard_active,
        x,
        &irls_betas[..p],
        offset,
        &mut irls_z[..n],
    );
    true
}

/// The fit's exits: the divergence verdict, the per-row-seed retry, the
/// non-converged and broken-column returns, `l`, the per-target SEs and the
/// pivot.
fn finish<'a>(
    inp: &IrlsInputs<'_>,
    nm: &NullModel,
    eta_guard_active: bool,
    scratch: GlmScratch<'a>,
    st: IrlsState,
) -> GlmFitView<'a> {
    let IrlsInputs {
        family,
        nb_theta,
        x,
        y,
        target_indices,
        beta_start,
        prior_w,
        offset,
        map,
        units,
        per_row_seed,
        n,
        p,
        t,
        ..
    } = *inp;
    let NullModel {
        deviance_null,
        log_link,
        eta_null,
        ..
    } = *nm;
    let IrlsState {
        last_chol,
        deviance_final,
        mut converged,
        stepped,
        had_pd_failure,
        mut diverged,
        n_iter,
        step_max,
        design_qr,
        l_from_qr,
        broken_col,
        ..
    } = st;
    if had_pd_failure {
        converged = false;
    }
    // Every unsettled end (the budget, an η the link does not evaluate, a
    // non-finite solve, a non-PD X'WX) gets the divergence verdict.
    if !converged && !diverged && eta_guard_active {
        diverged = runs_off(&scratch.irls_eta[..n], offset, eta_null, step_max);
    }

    if !converged {
        // The null-model seed is what keeps exposures decades apart working, so
        // it stays first; R's per-row seed is only a second try, and its run
        // is the one reported.
        if !per_row_seed && beta_start.is_none() && log_link && offset.is_some() {
            return glm_irls_fit_seeded(
                family,
                nb_theta,
                x,
                y,
                target_indices,
                beta_start,
                prior_w,
                offset,
                scratch,
                map,
                units,
                true,
            );
        }
    }
    let GlmScratch {
        irls_p,
        irls_betas,
        irls_var_diag,
        irls_t_sq,
        irls_u_scratch,
        mut irls_l,
        irls_broken,
        ..
    } = scratch;
    if !converged {
        // Partial re-NaN: inference outputs only — `betas` are deliberately
        // left at the loop's last fitted value, so the 3-array
        // `nan_fill_ols_scratch` must NOT be used here. A fit whose first solve
        // failed (X'WX not positive definite on the Gram or its QR rebuild, a
        // non-finite solve) has no fitted value: β is NaN, as on the n ≤ p
        // short circuit, not the start reported as a last step. A start the
        // link cannot evaluate stops before any solve (`n_iter` 0) and keeps it.
        if !stepped && n_iter > 0 {
            irls_betas[..p].fill(f64::NAN);
        }
        irls_var_diag[..t].fill(f64::NAN);
        irls_t_sq[..t].fill(f64::NAN);
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter,
            converged: false,
            separated: diverged
                && matches!(
                    family,
                    Family::Binomial { .. }
                        | Family::Poisson { .. }
                        | Family::NegativeBinomial { .. }
                ),
            deviance: f64::NAN,
            deviance_null: f64::NAN,
            mu: &irls_p[..n],
            pivot: f64::NAN,
            pivot_col: 0,
            se_unavailable: false,
            broken: &[],
            stop: Some(if !stepped && n_iter > 0 {
                StopReason::FirstSolveFailed
            } else {
                StopReason::IrlsDiverged
            }),
        };
    }

    // The last solve broke down: the estimates and the deviance stand, the SEs
    // are withheld. `l` is not materialised.
    if let Some(j) = broken_col {
        irls_var_diag[..t].fill(f64::NAN);
        irls_t_sq[..t].fill(f64::NAN);
        return GlmFitView {
            betas: &irls_betas[..p],
            var_diag: &irls_var_diag[..t],
            t_sq: &irls_t_sq[..t],
            l: irls_l.into_const(),
            n_iter,
            converged: true,
            separated: false,
            deviance: deviance_final,
            deviance_null,
            mu: &irls_p[..n],
            pivot: 0.0,
            pivot_col: j as u32,
            se_unavailable: true,
            broken: &irls_broken[..p],
            stop: None,
        };
    }

    // Materialise the cached lower-triangular L of the last accepted X'WX —
    // once, now that the fit has converged (deferred from the IRLS loop). On
    // the converged path `last_chol` is always `Some`: `converged` is only set
    // after a successful factorization stashed it. On a pass solved on X̃ the
    // stashed factor is X̃'WX̃'s, and L = Uᵀ·L̃ (see `solve_step`).
    if let Some(chol) = last_chol {
        let l = chol.L();
        for j in 0..p {
            for i in 0..p {
                irls_l[(i, j)] = if i >= j { l[(i, j)] } else { 0.0 };
            }
        }
        if let (true, Some(qr)) = (l_from_qr, &design_qr) {
            qr.map_factor(irls_l.rb_mut());
        }
    }

    // Per-target (B(X̃'WX̃)⁻¹Bᵀ)ₜₜ = ‖L⁻¹bₜ‖², bₜ row t of the column map B (eₜ on
    // an unmapped design) — see `ols::triangular_solve_norm_sq` for the
    // forward-solve identity — in the caller's units (`glm_var_scale`). That
    // is Var(β̂ₜ) at φ = 1, and φ stays out of the kernel: Var(β̂) = φ·(X'WX)⁻¹
    // on Gamma and inverse-Gaussian, with φ̂ read after the fit
    // (`fit::glm::glm_dispersion`). t² reads the caller's βₜ.
    for (out_idx, &tj) in target_indices.iter().enumerate() {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        let norm_sq = triangular_solve_norm_sq(
            irls_l.rb(),
            |i| map.rhs(tj, i),
            irls_u_scratch,
            p,
            false, // lower-triangular L: row access factor[(i, k)]
        ) * units.glm_var_scale();
        irls_var_diag[out_idx] = norm_sq;
        if norm_sq > 0.0 && norm_sq.is_finite() {
            let beta_j =
                units.beta_scale() * (map.beta_coef(tj, &irls_betas[..p]) + units.shift_caller(tj));
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
        separated: false,
        deviance: deviance_final,
        deviance_null,
        mu: &irls_p[..n],
        pivot,
        pivot_col: pivot_col as u32,
        se_unavailable: false,
        broken: &[],
        stop: None,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "glm_tests.rs"]
mod tests;
