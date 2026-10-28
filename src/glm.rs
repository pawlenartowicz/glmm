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
//!   - Adaptive convergence: `|Δdeviance| / (|deviance| + f) < DEVIANCE_TOL = 1e-12`
//!     (R's `glm.fit` rule, `src/library/stats/R/glm.R`, where `f = 0.1`).
//!     The rule is relative while `|deviance| ≫ f`; the floor `f` makes it
//!     absolute once the deviance itself is tiny (an exact fit). On Gamma and
//!     inverse-Gaussian `f` is 0.1 in the deviance's own units: times the
//!     precision weights' scale, and over ȳ on inverse-Gaussian, whose
//!     deviance scales as 1/y. So their stopping point does not depend on the
//!     overall scale of the weights or on the response's units (see
//!     `dev_floor`). The other families keep `f = 0.1`.
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
//! the cold-start arm), a
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

use crate::ols::{nan_fill_ols_scratch, triangular_solve_norm_sq};
use crate::spec::{BinomialLink, Family};
use crate::standardize::ColumnScaling;

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
/// agnostic so writeback code in `batch.rs` stays uniform.
pub struct GlmFitView<'a> {
    /// length P — fitted coefficients (NaN-filled on non-converged paths), on
    /// the design `x` held (β̃ when the fit ran through a column map, see
    /// `glm_irls_fit_mapped`). Under `se_unavailable` all finite, the column
    /// `pivot_col` at its start value (0 from a cold start): it is not an
    /// estimate.
    pub betas: &'a [f64],
    /// length T — `((X'WX)⁻¹)_jj` for each target, in the caller's units
    /// under a column map: `(B(X̃'WX̃)⁻¹Bᵀ)ₜₜ`.
    pub var_diag: &'a [f64],
    /// length T — Wald z² for each target. Compared against `z_crit²` from
    /// `CritValueTable`.
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
    /// converged `X'WX` to measure. `0.0` under `se_unavailable`, with
    /// `pivot_col` the first column that broke down.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
    /// The fit converged, but the QR of `√W·X` at its last solve had a zero or
    /// non-finite diagonal at column `pivot_col` (and possibly others after
    /// it) while `X'WX` there is positive: the column is not separated from
    /// the ones before it under the converged weights. `betas`, `deviance`
    /// and `mu` are the fit's; `var_diag` and `t_sq` are NaN, `pivot` is 0.
    /// The IRLS step along such a column is zero, so it stays at its start
    /// value. `false` on every other return.
    pub se_unavailable: bool,
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
/// Logit }` reaches the fused kernel `pw_and_log1pexp_sum` there, byte-identical
/// to the MCPower hot loop's own computation; every other
/// family/link runs a vectorized arm written against `family`'s scalar
/// formulas. Gamma/NB dispersion is handled by the caller (`fit.rs`),
/// not here — this kernel folds `φ=1`. `nb_theta` is the NB shape θ̂ the caller's
/// outer-θ loop fixes for this fit (only the NB family reads it via
/// `family::variance`/`dev_resid`; pass `f64::NAN` for every other family).
///
/// - `x`: `n × p` design (column-major faer).
/// - `y`: length `n`; {0.0, 1.0} for binomial, counts for Poisson/NB, positive for Gamma.
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
///   `Some(_)` routes `Family::Binomial { link: Logit }` to the weighted arm
///   instead (unweighted logit keeps the fused kernel). Matches R
///   `glm(weights=)` (`fit_glm_gamma_weighted_matches_r`,
///   `fit_glm_binomial_weighted_aggregated_matches_r`).
/// - `scratch`: borrowed mutable slots from `SimWorkspace.irls_*`.
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
    )
}

/// [`glm_irls_fit`] on a design fitted through a column map `B`
/// (`x̃ = X·B`, `β = B·β̃`). The returned `betas` are β̃, on the design `x`
/// holds; `var_diag` and `t_sq` are the caller's: `Var(βₜ) = ‖L⁻¹bₜ‖²` with
/// `bₜ` row `t` of `B`, `t² = βₜ²/Var(βₜ)`.
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
            separated: false,
            deviance: f64::NAN,
            deviance_null: f64::NAN,
            mu: &irls_p[..n],
            pivot: f64::NAN,
            pivot_col: 0,
            se_unavailable: false,
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
    let log_link = matches!(
        family,
        Family::Poisson { .. }
            | Family::NegativeBinomial { .. }
            | Family::Gamma {
                link: crate::spec::GammaLink::Log,
            }
            | Family::InverseGaussian {
                link: crate::spec::InverseGaussianLink::Log,
            }
    );
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

    // Floor of the relative deviance rule: R's 0.1, carried into the
    // deviance's units on the two dispersion families. The floor is an
    // absolute deviance, so a fixed 0.1 stops a fit early whenever the
    // deviance is small only because of units. On Gamma and inverse-Gaussian
    // the prior weights are precision weights with no scale of their own, and
    // D = Σ wᵢdᵢ scales with them; the inverse-Gaussian unit deviance
    // (y−μ)²/(yμ²) also scales as 1/y. With R's 0.1, an inverse-Gaussian/log
    // fit with D = 19 stopped with β 1e-6 (relative) short of the maximum at
    // y × 1e3, and a Gamma/log fit with every weight 1e-9 2e-3 short, both
    // reported converged.
    // `weight_scale` is an exact power of two (1 unweighted) and ȳ carries y's
    // units. The binomial deviance stays on R's 0.1: with φ ≡ 1 it is a
    // likelihood-ratio statistic of counts, whose weights count cases. Poisson
    // and NB do too while ȳ ≥ 1, but a non-integer y in tiny units (a rate
    // without exposure weights) has D ∝ ȳ, and R's 0.1 stopped such a fit
    // 1.7e-2 short in β at y × 1e-13, reported converged; so below ȳ = 1 the
    // floor carries ȳ's units. Not a plain 0.1·ȳ: an NB deviance with θ ≪ ȳ
    // does not grow with ȳ, and at ȳ = 1e6 that floor would ask |ΔD| < 1e-7 of
    // a linearly converging fit. An all-zero count response (ȳ = 0) keeps 0.1.
    // Unweighted, off inverse-Gaussian and ȳ < 1 counts, the floor is exactly 0.1.
    let dev_floor = match family {
        Family::Gamma { .. } => 0.1 * crate::family::weight_scale(prior_w, n),
        Family::InverseGaussian { .. } => 0.1 * crate::family::weight_scale(prior_w, n) / y_bar,
        Family::Poisson { .. } | Family::NegativeBinomial { .. } if y_bar > 0.0 => {
            0.1 * y_bar.min(1.0)
        }
        _ => 0.1,
    };

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
                Family::Gamma {
                    link: crate::spec::GammaLink::Inverse,
                    ..
                }
                | Family::InverseGaussian {
                    link: crate::spec::InverseGaussianLink::InverseSquared,
                } => {
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

    // Cholesky factor of the last accepted X'WX. The lower-triangular L is
    // materialised once after the loop (deferred from per-iteration so each
    // IRLS step skips an owned-Mat allocation). `Llt` owns its storage, so
    // keeping the last one past the next iter's irls_xtwx rebuild is safe.
    let mut last_chol = None;

    let mut deviance_prev = f64::INFINITY;
    // Deviance at the accepted β, the step-halving's reference; ∞ while there
    // is none. `deviance_prev` skips the seed, see the stop check.
    let mut deviance_accepted = f64::INFINITY;
    let mut deviance_final = f64::NAN;
    let mut converged = false;
    let mut had_pd_failure = false;
    // Set when the separation stop fires or the divergence guard judges an
    // unsettled end (`runs_off`).
    let mut diverged = false;
    let mut n_iter: u32 = 0;
    // The step that produced the current β, `max_i |Δηᵢ|`, and the bound of the
    // round-off stop; both read by the stop check at the top of the next pass
    // and refreshed after each accept. Pass 0 has no step behind it.
    let mut step_max = f64::INFINITY;
    let mut step_tiny = 0.0f64;
    // Period-2 damping state; see the detector at the accept step. It runs
    // only where the observed curvature differs from the Fisher weight: on the
    // other links Fisher scoring is Newton's method, whose map has a zero
    // Jacobian at the MLE. `irls_u_scratch` holds the last accepted β step; it
    // is free until the post-loop SE solve.
    let osc_guard_active = crate::family::exact_curvature_differs(family);
    let mut step_sq_prev = f64::NAN;
    // Whether the accepted β came from a Newton step, its Fisher twin held in
    // `irls_betas_new` (see the Newton step).
    let mut newton_pending = false;
    let mut osc_flips = 0usize;
    let mut damp = false;
    if osc_guard_active {
        irls_u_scratch[..p].fill(0.0);
    }

    // On the canonical links (logit, Poisson/log) the deviance is convex in β
    // and Fisher scoring is Newton's method, so a pass rarely raises the
    // deviance; a rise there halves only above `CANONICAL_RISE`, where it is
    // an overshoot rather than round-off. An ordinary pass, and so the
    // unweighted-logit hot path, stays one family pass per IRLS pass.
    let canonical = crate::family::is_canonical(family);

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
    let mut solve_step = false;
    // The QR and its mask of broken columns.
    let mut design_qr: Option<(crate::lmm::DesignQr, Vec<bool>)> = None;
    // Whether `last_chol` factors X̃'WX̃ on `design_qr`'s `U`.
    let mut l_from_qr = false;
    // The first broken column of the last solve pass, if any.
    let mut broken_col: Option<usize> = None;

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
    // The divergence guard reads |η − η₀|, which the two reciprocal links invert:
    // η = 1/μ under the Gamma inverse link, η = 1/μ² under the inverse-Gaussian
    // 1/μ² link. On both, η grows without bound as μ → 0, so a legitimate
    // small-mean fit sits far above the threshold (μ = 0.18 already gives
    // η = 30 under 1/μ²). Those two family/link pairs end only non-converged
    // at the other exits: an η past clamp_eta's bounds `[1e-75, 1e75]`
    // (src/family.rs), the non-finite guard on β_new and MAX_IRLS_ITERS.
    // Loop-invariant, so it is evaluated once.
    let eta_guard_active = !matches!(
        family,
        Family::Gamma {
            link: crate::spec::GammaLink::Inverse
        } | Family::InverseGaussian {
            link: crate::spec::InverseGaussianLink::InverseSquared
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
        let mut halvings = 0;
        let (deviance, infeasible) = loop {
            let mut yeta = 0.0;
            for i in 0..n {
                yeta += y[i] * irls_eta[i];
            }
            let (dev, inf) = crate::simd_transcendental::family_pass(
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
            let rise = (dev - deviance_accepted) / (dev.abs() + dev_floor);
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
            if newton_pending && irls_betas_new[..p].iter().all(|b| b.is_finite()) {
                newton_pending = false;
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
                if solve_step {
                    e_max = term_size(x, &irls_betas[..p], offset, &mut irls_z[..n]);
                }
                if eta_guard_active {
                    e_max = e_max.max(1.0);
                }
                step_max = s_max;
                step_tiny = STEP_ULPS * f64::EPSILON * e_max;
                continue;
            }
            if halvings == crate::glmm::PIRLS_MAX_HALVINGS {
                // `step_max > 0`: the redone pass at the accepted β breaks
                // here even when its deviance is not bit-identical.
                if rise <= crate::glmm::pirls_tol(family) && step_max > 0.0 {
                    irls_betas[..p].copy_from_slice(&irls_xtwz[..p]);
                    fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
                    step_max = 0.0;
                    continue;
                }
                break (dev, inf || !dev.is_finite());
            }
            halvings += 1;
            for j in 0..p {
                irls_betas[j] = 0.5 * (irls_betas[j] + irls_xtwz[j]);
            }
            fill_eta(x, &irls_betas[..p], offset, &mut irls_eta[..n]);
            step_max *= 0.5;
        };
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
            deviance_accepted = deviance;
        }
        // z − o: the WLS below solves β from X alone, so the offset's fixed
        // contribution must leave the working response. Solving for the step,
        // the right-hand side is z − η instead: `solve_step` is only set by a
        // pass's solve, and from the first accept on η = o + Xβ.
        let step_form = solve_step;
        if step_form {
            for i in 0..n {
                irls_z[i] -= irls_eta[i];
            }
        } else if let Some(o) = offset {
            for i in 0..n {
                irls_z[i] -= o[i];
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
            deviance_final = deviance;
            let settled =
                (deviance - deviance_prev).abs() / (deviance.abs() + dev_floor) < DEVIANCE_TOL;
            if settled || (step_max <= step_tiny && deviance.is_finite()) {
                if eta_guard_active && step_tiny > SEPARATION_STEP {
                    diverged = past_cap(&irls_eta[..n], offset, eta_null);
                } else {
                    diverged = eta_guard_active && step_max > SEPARATION_STEP;
                    converged = !diverged;
                }
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
        l_from_qr = false;
        let mut breakdown = None;
        if below {
            let (qr, broken) = design_qr
                .get_or_insert_with(|| (crate::lmm::DesignQr::identity(p), vec![false; p]));
            if qr.set_from_rows_marking(x, |i| w[i].sqrt(), irls_betas_new, broken) {
                l_from_qr = true;
            } else if (0..p).all(|j| {
                let g = irls_xtwx[(j, j)];
                !broken[j] || (g.is_finite() && g > 0.0)
            }) {
                l_from_qr = true;
                breakdown = broken.iter().position(|&b| b);
            }
        }
        let chol = if let (true, Some((qr, broken))) = (l_from_qr, &design_qr) {
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
                for j in (0..p).filter(|&j| broken[j]) {
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
            had_pd_failure = true;
            break;
        };
        solve_step |= below;
        broken_col = breakdown;

        // Solve β_new = L⁻ᵀ L⁻¹ · X'Wz using chol.solve_in_place.
        // First write xtwz into a 1-col view; chol.solve_in_place expects a
        // MatMut. Since irls_xtwz is &mut [f64] of length p, build a
        // temporary MatMut via faer's `from_column_major_slice_mut`.
        {
            let mut rhs = MatMut::from_column_major_slice_mut(irls_xtwz, p, 1usize);
            chol.solve_in_place(rhs.rb_mut());
        }
        if let (true, Some((qr, _))) = (l_from_qr, &design_qr) {
            qr.map_beta(&mut irls_xtwz[..p]);
        }
        irls_betas_new[..p].copy_from_slice(&irls_xtwz[..p]);
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
            for i in 0..n {
                let pw = prior_w.map_or(1.0, |w| w[i]);
                let (e, mu) = (irls_eta[i], irls_p[i]);
                let wo =
                    crate::family::observed_weight(family, nb_theta, y[i], pw, e, mu, irls_w[i]);
                let rho = crate::family::row_score(family, nb_theta, y[i], pw, e, mu);
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
        if osc_guard_active && !damp {
            let (mut dot, mut step_sq) = (0.0, 0.0);
            for j in 0..p {
                let step = irls_betas_new[j] - irls_betas[j];
                dot += step * irls_u_scratch[j];
                step_sq += step * step;
                irls_u_scratch[j] = step;
            }
            let ratio = crate::glmm::PIRLS_OSC_RATIO;
            if dot < 0.0 && step_sq > ratio * ratio * step_sq_prev {
                osc_flips += 1;
            } else {
                osc_flips = 0;
            }
            damp = osc_flips >= crate::glmm::PIRLS_OSC_TRIGGER;
            step_sq_prev = step_sq;
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
                if damp {
                    irls_betas[j] += 0.5 * (irls_betas_new[j] - old);
                    irls_betas_new[j] = old + 0.5 * (fisher - old);
                } else {
                    irls_betas[j] = irls_betas_new[j];
                    irls_betas_new[j] = fisher;
                }
            }
        } else {
            irls_xtwz[..p].copy_from_slice(&irls_betas[..p]);
            if damp {
                for j in 0..p {
                    irls_betas[j] += 0.5 * (irls_betas_new[j] - irls_betas[j]);
                }
            } else {
                irls_betas[..p].copy_from_slice(&irls_betas_new[..p]);
            }
        }
        newton_pending = newton_used;
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
        if solve_step {
            e_max = term_size(x, &irls_betas[..p], offset, &mut irls_z[..n]);
        }
        if eta_guard_active {
            e_max = e_max.max(1.0);
        }
        step_max = s_max;
        step_tiny = STEP_ULPS * f64::EPSILON * e_max;
    }

    if had_pd_failure {
        converged = false;
    }
    // Every unsettled end (the budget, an η the link does not evaluate, a
    // non-finite solve, a non-PD X'WX) gets the divergence verdict.
    if !converged && !diverged && eta_guard_active {
        diverged = runs_off(&irls_eta[..n], offset, eta_null, step_max);
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
        if let (true, Some((qr, _))) = (l_from_qr, &design_qr) {
            qr.map_factor(irls_l.rb_mut());
        }
    }

    // Per-target Var(β̂ₜ) = (B(X̃'WX̃)⁻¹Bᵀ)ₜₜ = ‖L⁻¹bₜ‖², bₜ row t of the column
    // map B (eₜ on an unmapped design) — see `ols::triangular_solve_norm_sq`
    // for the forward-solve identity (no σ̂² scaling here: under Bernoulli the
    // score covariance is (X'WX)⁻¹ directly).
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
        );
        irls_var_diag[out_idx] = norm_sq;
        if norm_sq > 0.0 && norm_sq.is_finite() {
            let beta_j = map.beta_coef(tj, &irls_betas[..p]);
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

    /// All-0 and all-1 Bernoulli y short-circuit (n_iter 0, not converged)
    /// with z², deviance and null deviance NaN.
    #[test]
    fn glm_z_sq_and_deviances_nan_on_non_converged() {
        let n = 100;
        let p = 2;
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = (i as f64) / (n as f64) - 0.5;
        }
        for y_val in [0.0f64, 1.0] {
            let y = vec![y_val; n];
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
            assert!(!fit.converged, "y={y_val}");
            assert_eq!(fit.n_iter, 0, "y={y_val}");
            for &t in fit.t_sq.iter() {
                assert!(
                    t.is_nan(),
                    "y={y_val}: z² must be NaN on non-converged fit, got {t}"
                );
            }
            assert!(
                fit.deviance.is_nan(),
                "y={y_val}: deviance must be NaN on non-converged"
            );
            assert!(
                fit.deviance_null.is_nan(),
                "y={y_val}: deviance_null must be NaN on the all-0/all-1 short circuit"
            );
        }
    }

    #[test]
    fn glm_separation_marks_non_converged() {
        // Build a fully-separated dataset: y = (x > 0). Logistic regression
        // diverges (β → ∞, so η → ∞); the divergence cap or the separation stop should fire.
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
        assert!(fit.separated, "separation must be reported as such");
        for (k, (&t, &v)) in fit.t_sq.iter().zip(fit.var_diag.iter()).enumerate() {
            assert!(
                t.is_nan() && v.is_nan(),
                "target {k}: t_sq {t}, var_diag {v}"
            );
        }
        assert!(fit.deviance.is_nan(), "deviance {}", fit.deviance);
        assert!(
            fit.deviance_null.is_nan(),
            "deviance_null {}",
            fit.deviance_null
        );
        // β is left at the last iterate, not NaN-filled.
        assert!(fit.betas.iter().all(|b| b.is_finite()), "{:?}", fit.betas);
    }

    /// Separation that crosses the bound: `[1, x, x + d]` with `d = ±1` on a
    /// block of ten rows whose outcomes separate along `d` (`y = 1` where
    /// `d = 1`), pushed to the edge by an offset of −12, and `d = 0` elsewhere.
    /// As the block's weights vanish `X'WX` turns singular, the solve on the
    /// QR factor throws β₁ and β₂ out to ∓3e34, and η then absorbs every later
    /// step: the step reads 0 and the deviance repeats, so both stops fire on
    /// a fit that is running off. Their round-off bound is far above
    /// `SEPARATION_STEP`, so the step cannot show a move and η decides, as on
    /// a fit that runs out its budget.
    #[test]
    fn glm_separation_past_the_round_off_bound_is_not_converged() {
        let (n, m) = (60, 10);
        let mut st = 1u64;
        let mut x = Mat::<f64>::zeros(n, 3);
        let (mut y, mut offset) = (vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            let x1 = 2.0 * crate::fit::common_tests::lcg(&mut st);
            let u = 0.5 * (crate::fit::common_tests::lcg(&mut st) + 1.0);
            let d = match (i < m, i % 2) {
                (true, 0) => 1.0,
                (true, _) => -1.0,
                _ => 0.0,
            };
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            x[(i, 2)] = x1 + d;
            y[i] = if i < m {
                f64::from(u8::from(d > 0.0))
            } else {
                f64::from(u8::from(u < 1.0 / (1.0 + (-0.3 * x1).exp())))
            };
            if i < m {
                offset[i] = -12.0;
            }
        }
        let mut ws = TestWs::new(n, 3, 0);
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &[0, 1, 2],
            None,
            None,
            Some(&offset),
            glm_scratch(&mut ws),
        );
        assert!(
            !fit.converged && fit.separated,
            "converged {}, separated {}, β {:?}",
            fit.converged,
            fit.separated,
            fit.betas
        );
    }

    /// Quasi-separation on the cloglog upper tail: one category whose rows are
    /// all 1. There 1 − μ = exp(−e^η) vanishes so fast that the deviance rule
    /// fires at η ≈ 3.3 while the category's η still moves about e^{−η} ≈ 0.04
    /// per step, well short of the divergence guard. The fit has no finite
    /// maximum and must say so (`SEPARATION_STEP`).
    #[test]
    fn glm_cloglog_upper_tail_separation_marks_non_converged() {
        let (m, p) = (20usize, 3usize);
        let n = 3 * m;
        let mut x = Mat::<f64>::zeros(n, p);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let g = i / m;
            x[(i, 0)] = 1.0;
            x[(i, 1)] = f64::from(u8::from(g == 1));
            x[(i, 2)] = f64::from(u8::from(g == 2));
            y[i] = match g {
                0 => f64::from(u8::from(i % 3 == 0)),
                1 => f64::from(u8::from(i % 3 != 0)),
                _ => 1.0,
            };
        }
        let mut ws = TestWs::new(n, p, 0);
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Cloglog,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &[],
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(
            !fit.converged && fit.separated,
            "converged {}, separated {}",
            fit.converged,
            fit.separated
        );
        assert!(fit.betas.iter().all(|b| b.is_finite()), "{:?}", fit.betas);
    }

    /// A cloglog fit started where a y = 0 row sits at η ≈ 20.2 (a leverage row
    /// at x = 10, start β = (0.2, 2, 0)). There the Fisher weight underflows
    /// while the score is −e^η, so the working response overflows and the
    /// Fisher step is not finite; the Newton step's weight on that row is e^η.
    /// A finite maximum exists. Reference: R 4.5 `optim(c(0, 0, 0), D,
    /// gradient, method = "BFGS", reltol = 1e-16)`, restarted at its own
    /// optimum until it stops moving, on D(β) = −2Σ[y·ln(−expm1(−t)) −
    /// (1 − y)·t], t = exp(Xβ): D = 40.8525618518509, gradient norm 1e-6.
    #[test]
    fn glm_cloglog_start_with_a_y0_row_deep_in_the_upper_tail_converges() {
        const REF_BETA: [f64; 3] = [0.063854992759243, -0.0109381919145387, 0.187123718391584];
        const REF_DEV: f64 = 40.8525618518509;
        #[rustfmt::skip]
        let y_head: [f64; 30] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
            1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        // 30 rows on x ∈ [−2, 2], then a y = 0 row at x = 10 and a y = 1 row at
        // x = −10.
        let n = 32;
        let mut x = Mat::<f64>::zeros(n, 3);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let (xi, x2, yi) = match i {
                30 => (10.0, 0.0, 0.0),
                31 => (-10.0, 0.0, 1.0),
                _ => (
                    -2.0 + 4.0 * i as f64 / 29.0,
                    (1.7 * i as f64).cos(),
                    y_head[i],
                ),
            };
            x[(i, 0)] = 1.0;
            x[(i, 1)] = xi;
            x[(i, 2)] = x2;
            y[i] = yi;
        }
        let mut ws = TestWs::new(n, 3, 0);
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Cloglog,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &[],
            Some(&[0.2, 2.0, 0.0]),
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(fit.converged, "must converge; β = {:?}", fit.betas);
        assert!(
            (fit.deviance - REF_DEV).abs() / REF_DEV < 1e-12,
            "deviance {} vs optim {REF_DEV}",
            fit.deviance
        );
        for (j, &r) in REF_BETA.iter().enumerate() {
            let rel = (fit.betas[j] - r).abs() / r.abs();
            assert!(rel < 1e-6, "β[{j}] = {} vs optim {r}", fit.betas[j]);
        }
    }

    /// `beta_start` seeds β and η directly instead of the family's cold-start
    /// rule, so warm-starting at an already-converged optimum should settle
    /// in far fewer IRLS iterations than a cold start and land on the same β.
    #[test]
    fn glm_irls_fit_warm_start_reaches_same_optimum_in_fewer_iterations() {
        let n = 200;
        let p = 2;
        let targets: Vec<u32> = vec![0, 1];
        let (x, y) = scaled_logit_design(n, 1.0);

        let mut ws_cold = TestWs::new(n, p, 0);
        let cold = glm_irls_fit(
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
            glm_scratch(&mut ws_cold),
        );
        assert!(cold.converged, "cold fit must converge");
        let cold_betas = cold.betas.to_vec();
        let cold_iter = cold.n_iter;

        let mut ws_warm = TestWs::new(n, p, 0);
        let warm = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            Some(&cold_betas),
            None,
            None,
            glm_scratch(&mut ws_warm),
        );
        assert!(warm.converged, "warm fit must converge");
        assert!(
            warm.n_iter < cold_iter,
            "warm-starting at the converged optimum must take fewer IRLS \
             iterations than the cold start: warm={} cold={cold_iter}",
            warm.n_iter
        );
        for (w, c) in warm.betas.iter().zip(cold_betas.iter()) {
            assert!(
                (w - c).abs() < 1e-9,
                "warm start must land on the same optimum as the cold fit: {w} vs {c}"
            );
        }
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
    /// same accept/reject decision in every unit system. A guard bounding
    /// `|β_j| > 30` instead would reject the x/1000 fit with NaN variances while
    /// accepting the identical x and x·1000 fits — scale-dependent, unlike the η-based guard here.
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
        // Column scaling D = diag(1, s) turns X'WX into D(X'WX)D, so the
        // covariance (X'WX)⁻¹ picks up D⁻¹ on both sides: the intercept
        // variance is exactly unchanged and the slope variance is divided by
        // s². A `|β_j| > 30`-style guard that let a scale-dependent variance
        // through would fail this, not just the finiteness check above.
        for (v, s) in [(&v2, 1e-3f64), (&v3, 1e3f64)] {
            assert!(
                (v[0] - v1[0]).abs() <= 1e-9 * v1[0].abs(),
                "intercept variance must not depend on the slope column's units: {} vs {}",
                v[0],
                v1[0]
            );
            let rescaled = v[1] * s * s;
            assert!(
                (rescaled - v1[1]).abs() <= 1e-9 * v1[1].abs(),
                "slope variance must scale by s²: {rescaled} vs {}",
                v1[1]
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
        // Same D = diag(1, s) argument as the logit guard above: intercept
        // variance is unchanged, slope variance is divided by s².
        assert!(
            (v2[0] - v1[0]).abs() <= 1e-9 * v1[0].abs(),
            "poisson intercept variance: {} vs {}",
            v2[0],
            v1[0]
        );
        let rescaled_var = v2[1] * 1e-3 * 1e-3;
        assert!(
            (rescaled_var - v1[1]).abs() <= 1e-9 * v1[1].abs(),
            "poisson slope variance must scale by s²: {rescaled_var} vs {}",
            v1[1]
        );
    }

    /// A GLM whose `X'WX` factor fails solves its step on `X·U⁻¹`, `U` from the
    /// QR of `√W·X`, and reaches the maximum. On `entangled_dyadic_designs` at
    /// `δ = 2⁻²⁷` (pivot ratio about 5e-17) the Gram is singular to rounding,
    /// and a solve from it ends every arm not converged with β₂ off by 16% to
    /// 100%. The `[1, a, b]` twin spans the same space, so the entangled
    /// coefficient is the twin's divided by `δ`, its variance by `δ²`. Arms:
    /// logit; probit, whose Newton step keeps the raw Gram; Poisson with an
    /// offset and prior weights, whose right-hand side must stay `z − o` and
    /// then `z − η`. Logit and Poisson also exercise the round-off stop on the
    /// size of the terms: η = Xβ sums terms near 1e8 (β₂ ≈ 3e6, |a| ≤ 50) that
    /// cancel to order 1, so η carries about 1e-8 of rounding and the deviance
    /// moves by more than the relative rule allows from one pass to the next.
    /// Measured against the twin (relative): logit deviance 1.5e-10, β₂ 1.0e-9,
    /// `var_diag₂` 1.0e-9; probit 5.6e-10, 9.3e-6 (6e-5 of its SE, the stop
    /// rule's slack on a linearly converging link) and 2.1e-6; Poisson 6.4e-11,
    /// 1.4e-9 and 5.9e-7.
    #[test]
    fn glm_steps_through_the_design_qr_when_the_gram_factor_fails() {
        let n = 400;
        let (xv, xrv, delta) = crate::fit::common_tests::entangled_dyadic_designs(n, 27);
        let col_major = |v: &[f64]| Mat::<f64>::from_fn(n, 3, |i, j| v[3 * i + j]);
        let (x, xr) = (col_major(&xv), col_major(&xrv));
        let offset: Vec<f64> = (0..n).map(|i| 0.1 * ((i % 7) as f64 - 3.0)).collect();
        let w: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * (i % 5) as f64).collect();
        // (family, and the relative tolerances on β₂, the deviance and var₂).
        let arms = [
            (
                Family::Binomial {
                    link: BinomialLink::Logit,
                },
                1e-8,
                1e-9,
                1e-8,
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Probit,
                },
                1e-4,
                1e-8,
                1e-5,
            ),
            (
                Family::Poisson {
                    link: crate::PoissonLink::Log,
                },
                1e-8,
                1e-9,
                1e-5,
            ),
        ];
        let targets = [0u32, 1, 2];
        for (family, beta_tol, dev_tol, var_tol) in arms {
            let poisson = matches!(family, Family::Poisson { .. });
            let (off, pw) = if poisson {
                (Some(&offset[..]), Some(&w[..]))
            } else {
                (None, None)
            };
            let mut st = 7u64;
            let y: Vec<f64> = (0..n)
                .map(|i| {
                    let eta = 0.2 + 0.02 * xrv[3 * i + 1] + 0.02 * xrv[3 * i + 2];
                    let u = 0.5 * (crate::fit::common_tests::lcg(&mut st) + 1.0);
                    if poisson {
                        // Inverse-CDF Poisson draw.
                        let mu = (eta + offset[i]).exp();
                        let (mut k, mut pk) = (0.0, (-mu).exp());
                        let mut cdf = pk;
                        while u > cdf {
                            k += 1.0;
                            pk *= mu / k;
                            cdf += pk;
                        }
                        k
                    } else {
                        f64::from(u8::from(u < 1.0 / (1.0 + (-eta).exp())))
                    }
                })
                .collect();
            let fit = |x: MatRef<'_, f64>| {
                let mut ws = TestWs::new(n, 3, 0);
                let v = glm_irls_fit(
                    family,
                    f64::NAN,
                    x,
                    &y,
                    &targets,
                    None,
                    pw,
                    off,
                    glm_scratch(&mut ws),
                );
                (v.converged, v.pivot, v.deviance, v.betas[2], v.var_diag[2])
            };
            let (conv, pivot, dev, beta, var) = fit(x.as_ref());
            let (conv_r, _, dev_r, beta_r, var_r) = fit(xr.as_ref());
            assert!(conv_r, "{family:?}: the twin converges");
            let want_beta = beta_r / delta;
            assert!(
                (beta - want_beta).abs() <= beta_tol * want_beta.abs(),
                "{family:?}: β₂ {beta} against {want_beta}"
            );
            assert!(conv, "{family:?}: converges");
            assert!(
                pivot < crate::lmm::GRAM_PIVOT_MIN,
                "{family:?}: pivot {pivot:e}"
            );
            assert!(
                (dev - dev_r).abs() <= dev_tol * dev_r,
                "{family:?}: deviance {dev} against {dev_r}"
            );
            let want_var = var_r / (delta * delta);
            assert!(
                (var - want_var).abs() <= var_tol * want_var,
                "{family:?}: var₂ {var} against {want_var}"
            );
        }
    }

    /// `[1, x, e, e]` (`n` rows, `e` the indicator of the last row) when
    /// `dup`, else `[1, x, e]`, with a response for `family`: the Givens pass
    /// over `√W·X` leaves `R_33` exactly zero, since no rotation reaches the
    /// duplicate once the first copy has absorbed the last row.
    fn breakdown_case(family: Family, dup: bool) -> (Mat<f64>, Vec<f64>) {
        let n = 50;
        let p = if dup { 4 } else { 3 };
        let x = Mat::<f64>::from_fn(n, p, |i, j| match j {
            0 => 1.0,
            1 => i as f64 / n as f64 - 0.5,
            _ => f64::from(u8::from(i == n - 1)),
        });
        let y = (0..n)
            .map(|i| match family {
                Family::Gamma { .. } => 1.0 + 0.25 * ((3 * i) % 7) as f64,
                _ if i == n - 1 => 3.0,
                _ => ((7 * i) % 5) as f64,
            })
            .collect();
        (x, y)
    }

    /// A breakdown at the last solve (`R_33 = 0` on `[1, x, e, e]`) keeps the
    /// fit: converged, with the deviance of the `[1, x, e]` fit (same column
    /// space) and the SEs withheld. The duplicate takes a zero step, so from a
    /// cold start it stays at 0. Gamma/log also takes Newton steps, which skip
    /// a pass with a broken column (from the raw Gram the duplicate would move
    /// to −0.17). Measured against `[1, x, e]` (relative deviance): Poisson 0,
    /// Gamma 2.1e-16.
    #[test]
    fn glm_breakdown_keeps_the_deviance_and_withholds_the_ses() {
        let arms = [
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            Family::Gamma {
                link: crate::GammaLink::Log,
            },
        ];
        for family in arms {
            let (x, y) = breakdown_case(family, true);
            let mut ws = TestWs::new(50, 4, 0);
            let v = glm_irls_fit(
                family,
                f64::NAN,
                x.as_ref(),
                &y,
                &[0, 1, 2, 3],
                None,
                None,
                None,
                glm_scratch(&mut ws),
            );
            let (xr, _) = breakdown_case(family, false);
            let mut ws_r = TestWs::new(50, 3, 0);
            let r = glm_irls_fit(
                family,
                f64::NAN,
                xr.as_ref(),
                &y,
                &[0, 1, 2],
                None,
                None,
                None,
                glm_scratch(&mut ws_r),
            );
            assert!(r.converged, "{family:?}: the [1, x, e] fit converges");
            assert!(v.converged && v.se_unavailable, "{family:?}");
            assert!(!v.separated, "{family:?}");
            assert_eq!((v.pivot, v.pivot_col), (0.0, 3), "{family:?}");
            assert!(v.var_diag.iter().all(|d| d.is_nan()), "{family:?}");
            assert!(v.t_sq.iter().all(|t| t.is_nan()), "{family:?}");
            assert!(v.betas.iter().all(|b| b.is_finite()), "{family:?}");
            assert_eq!(v.betas[3], 0.0, "{family:?}");
            let rel = (v.deviance - r.deviance).abs() / r.deviance;
            assert!(
                rel <= 1e-14,
                "{family:?}: deviance {} against {}",
                v.deviance,
                r.deviance
            );
        }
    }

    /// From a warm start the broken-down column stays at its start value: the
    /// loop tier's call shape, a draw seeded from the truth.
    #[test]
    fn glm_breakdown_holds_the_column_at_its_warm_start() {
        let family = Family::Poisson {
            link: crate::PoissonLink::Log,
        };
        let (x, y) = breakdown_case(family, true);
        let mut ws = TestWs::new(50, 4, 0);
        let v = glm_irls_fit(
            family,
            f64::NAN,
            x.as_ref(),
            &y,
            &[0, 1, 2, 3],
            Some(&[0.5, 0.1, 0.3, 0.25]),
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(v.converged && v.se_unavailable);
        assert_eq!(v.betas[3], 0.25);
    }

    /// A column zero under the weights is not a breakdown: the Gram has a zero
    /// pivot there and the fit ends not converged, with no deviance. The logit
    /// arm's alternating 0/1 response keeps it off the all-0/all-1 short circuit.
    #[test]
    fn glm_zero_weighted_column_stays_not_converged() {
        let arms = [
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
        ];
        for family in arms {
            let (mut x, mut y) = breakdown_case(family, false);
            for i in 0..50 {
                x[(i, 2)] = 0.0;
            }
            if matches!(family, Family::Binomial { .. }) {
                y = (0..50).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
            }
            let mut ws = TestWs::new(50, 3, 0);
            let v = glm_irls_fit(
                family,
                f64::NAN,
                x.as_ref(),
                &y,
                &[0, 1, 2],
                None,
                None,
                None,
                glm_scratch(&mut ws),
            );
            assert!(!v.converged && !v.se_unavailable, "{family:?}");
            assert!(v.deviance.is_nan(), "{family:?}");
        }
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
        // (R: 134.6023); null deviance = -2*(40*ln(0.4) + 60*ln(0.6)) depends only on y_bar.
        let expected = -2.0 * (40.0 * 0.4f64.ln() + 60.0 * 0.6f64.ln());
        let rel = (fit.deviance_null - expected).abs() / expected;
        assert!(
            rel <= 1e-12,
            "deviance_null = {}, expected {expected} (rel {rel:e})",
            fit.deviance_null
        );
    }

    /// Weighted Gamma(log) null/residual deviance vs R glm(weights=).
    /// Convention: null mean is the WEIGHTED mean μ̂₀ = Σwᵢyᵢ/Σwᵢ and both
    /// deviances accumulate Σwᵢdᵢ. Same fixture as fit.rs's
    /// `fit_glm_gamma_weighted_matches_r` (β/SE/φ gated there). The Gamma
    /// deviance does not change when y is rescaled, so the same references hold
    /// at y × 2⁻⁴⁰, where ȳ ≈ 2e-12 sits below `family::MU_FLOOR`.
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
        let targets: Vec<u32> = vec![0, 1];
        for scale in [1.0, 2f64.powi(-40)] {
            let ys: Vec<f64> = y.iter().map(|v| v * scale).collect();
            let mut ws = TestWs::new(n, p, 0);
            let fit = glm_irls_fit(
                crate::Family::Gamma {
                    link: crate::GammaLink::Log,
                },
                f64::NAN,
                x.as_ref(),
                &ys,
                &targets,
                None,
                Some(&w),
                None,
                glm_scratch(&mut ws),
            );
            assert!(
                fit.converged,
                "weighted gamma GLM must converge (y × {scale})"
            );
            assert!(
                (fit.deviance_null - REF_DEV_NULL).abs() / REF_DEV_NULL < 1e-8,
                "deviance_null = {} vs R {REF_DEV_NULL} (y × {scale})",
                fit.deviance_null
            );
            assert!(
                (fit.deviance - REF_DEV).abs() / REF_DEV < 1e-6,
                "deviance = {} vs R {REF_DEV} (y × {scale})",
                fit.deviance
            );
        }
    }
}
