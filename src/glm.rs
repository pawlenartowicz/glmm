//! Adaptive IRLS logistic regression kernel; outputs z² = β̂²/Var(β̂) for hot-loop comparison against a precomputed squared-critical-value table supplied by the caller — no CDF calls, no SE sqrt.
//!
//! Hot-loop invariant: no z-CDF calls, no SE sqrt, no per-coefficient
//! square roots. Inference outputs `z_sq_j = β̂_j² / Var(β̂_j)` against the
//! caller-supplied precomputed `z_crit_sq` table.
//!
//! All per-fit buffers live in `SimWorkspace` (the `irls_*` fields, see workspace.rs).
//! The kernel takes a `GlmScratch<'w>` built inline at the call site (NLL split-borrow)
//! and returns a borrowed `GlmFitView<'a>`. No owned result struct.
//!
//! Algorithm — guards and tolerances:
//!   - Adaptive convergence: `|Δdeviance| < DEVIANCE_TOL = 1e-8`
//!   - Safety cap: `MAX_IRLS_ITERS = 50`
//!   - BETA_CAP divergence guard: `iter ≥ 3 ∧ ‖β‖_∞ > 30 → non-converged`
//!   - All-0 / all-1 short circuit
//!   - Post-fit saturation guard (50% of weights < 1e-5 ⇒ non-converged)
//!   - No step-halving: β_new is accepted directly (step-halving drifts
//!     power ~3.5% at N=50 — see the accept step in the IRLS loop)
//!
//! Two `beta_start` modes: `None` seeds β = 0, a fixed reproducible cold
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

use crate::ols::nan_fill_ols_scratch;
use crate::spec::{BinomialLink, Family};
use crate::FLOAT_NEAR_ZERO;

/// IRLS safety cap.
pub const MAX_IRLS_ITERS: u32 = 50;
/// Adaptive convergence tolerance on `|Δdeviance|`.
pub const DEVIANCE_TOL: f64 = 1e-8;
/// Divergence guard: any |β_j| > BETA_CAP at iter ≥ 3 marks non-converged.
pub const BETA_CAP: f64 = 30.0;
/// Floor on per-row IRLS weight `W_i = p_i (1-p_i)` to avoid division by zero
/// in the working response.
pub const WEIGHT_CLAMP: f64 = 1e-6;
/// Saturation post-fit guard: rows with `p_i(1-p_i) < SATURATION_W` count as
/// saturated. If the fraction exceeds `SATURATION_FRAC`, the fit is marked
/// non-converged.
pub const SATURATION_W: f64 = 1e-5;
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
    pub n_iter: u32,
    pub converged: bool,
    /// Final-iteration Bernoulli deviance −2·Σ[y log p̂ + (1−y) log(1−p̂)].
    /// `NaN` on every non-converged / short-circuit return.
    pub deviance: f64,
    /// Null-model deviance −2·(Σy · log ȳ + (n − Σy) · log(1 − ȳ)).
    /// `NaN` whenever `Σy ∈ {0, n}` (the short-circuit path) or any other
    /// non-converged path.
    pub deviance_null: f64,
}

/// Caller-owned scratch borrowed from `SimWorkspace` field-by-field. Built
/// inline at the call site (NLL split-borrow with simultaneous shared borrows
/// of `ws.x_full` / `ws.y_full`). **Do not** wrap in a helper method
/// `ws.glm_scratch()` — that re-introduces the whole-struct exclusive borrow
/// problem (NLL cannot split borrow a method receiver from its fields).
pub struct GlmScratch<'w> {
    pub irls_eta: &'w mut [f64],
    pub irls_p: &'w mut [f64],
    pub irls_w: &'w mut [f64],
    pub irls_z: &'w mut [f64],
    pub irls_betas: &'w mut [f64],
    pub irls_betas_new: &'w mut [f64],
    pub irls_var_diag: &'w mut [f64],
    pub irls_t_sq: &'w mut [f64],
    pub irls_u_scratch: &'w mut [f64],
    pub irls_xtwx: MatMut<'w, f64>,
    pub irls_xtwz: &'w mut [f64],
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
/// `family` selects the IRLS math. `Family::Binomial { link: Logit }` runs the
/// **verbatim** canonical fused-SIMD path (byte-identity, the MCPower hot loop);
/// every other family routes the scalar general Fisher-scoring branch through
/// [`crate::family`]. Gamma/NB dispersion is handled by the caller (`fit.rs`),
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
/// - `scratch`: borrowed mutable slots from `SimWorkspace.irls_*`.
///
/// `deviance_null` matches R's `glm(family=binomial)$null.deviance` (see
/// `glm_deviance_null_golden_value`); no external oracle validates the full
/// β̂/deviance path yet.
pub fn glm_irls_fit<'a>(
    family: Family,
    nb_theta: f64,
    x: MatRef<'_, f64>,
    y: &[f64],
    target_indices: &[u32],
    beta_start: Option<&[f64]>,
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
        };
    }

    // Null-model deviance — computed once, reused at the batch site for the
    // LRT. Stored in a local so the per-iteration deviance tracker doesn't
    // clobber it. The all-0 / all-1 branch above guarantees `y_bar ∉ {0, 1}`
    // here, so no `ln(0)` risk.
    let y_bar = y_sum / n as f64;
    // Intercept-only MLE is μ̂=ȳ for any (family, link), so the null deviance is
    // Σ dᵢ(y, ȳ). Logit keeps the closed-form Bernoulli expression verbatim
    // (byte-identity); other families fold it generically.
    let deviance_null = match family {
        Family::Binomial {
            link: BinomialLink::Logit,
        } => -2.0 * (y_sum * y_bar.ln() + (n as f64 - y_sum) * (1.0 - y_bar).ln()),
        other => {
            let mu0 = crate::family::clamp_mu(other, y_bar);
            let mut d = 0.0;
            for &yi in &y[..n] {
                d += crate::family::dev_resid(other, nb_theta, yi, mu0);
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
        }
        // Cold start: β ← 0, so η = X·β = 0 — bit-identical for logit (μ=0.5) and
        // fine for Poisson/Gamma-log (μ=1). The Gamma **inverse** link cannot
        // start there (η=0 ⇒ μ=1/0): seed η = 1/y per row (R's `mustart=y`,
        // `etastart=1/y`) so the first IRLS step has a valid μ>0. β stays 0; the
        // first solve overwrites it, so X·β consistency resumes from iter 1.
        None => {
            irls_betas[..p].fill(0.0);
            if matches!(
                family,
                Family::Gamma {
                    link: crate::spec::GammaLink::Inverse,
                    ..
                }
            ) {
                for i in 0..n {
                    irls_eta[i] = 1.0 / crate::family::clamp_mu(family, y[i]);
                }
            } else {
                irls_eta[..n].fill(0.0);
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
    // β_new = (X'WX)⁻¹ X'Wz on the working response z = η + (y − μ)/W with
    // weights W = diag(μ(1−μ)), μ = σ(η). Iterated to the |Δdeviance| fixpoint
    // it is the maximum-likelihood β. MN89 §4.
    // `0..=`: pass k's top checks convergence of the deviance that pass k−1's
    // solve produced (off the carried η), so the final solve still gets its
    // check without the extra pass buying another factorization. The deviance
    // moved from a post-accept `bernoulli_deviance` call to the top-of-pass
    // fused kernel — same values one pass later, no standalone `log1pexp` sweep
    // (deviance fold; bit-identical, the fused Σ equals the lean one).
    for iter in 0..=MAX_IRLS_ITERS {
        // η = X · β is already in `irls_eta`: seeded to 0 (β = 0) or the
        // truth start before the loop, then refreshed by the accept step below
        // after each β update. Reusing it here is bit-identical to recomputing
        // X·β (same β, same summation order).

        // p, W, z, and the deviance for this β. Family-branched: logit keeps the
        // VERBATIM fused-SIMD path (p, W, Σ log1pexp(η) in one vectorized pass,
        // then the scalar z follow-up that folds in the Σ y·η deviance half) for
        // byte-identity; new families take the scalar general Fisher-scoring
        // branch through `family.rs` (φ folded as 1). Both leave irls_p / irls_w
        // / irls_z filled identically in shape for the WLS solve below.
        let deviance = match family {
            Family::Binomial {
                link: BinomialLink::Logit,
            } => {
                let lp_sum = crate::simd_transcendental::pw_and_log1pexp_sum(
                    &irls_eta[..n],
                    &mut irls_p[..n],
                    &mut irls_w[..n],
                );
                let mut yeta = 0.0;
                for i in 0..n {
                    let yi = y[i];
                    yeta += yi * irls_eta[i];
                    irls_z[i] = irls_eta[i] + (yi - irls_p[i]) / irls_w[i];
                }
                2.0 * (lp_sum - yeta)
            }
            other => {
                // Per row: clamp η; (μ, W_raw, working_resid) from family.rs;
                // floor W at WEIGHT_CLAMP; z = η + r; accumulate Σ dᵢ (the
                // residual deviance — the |Δ| convergence metric, dispersion-free).
                let mut dev = 0.0;
                for i in 0..n {
                    let e = crate::family::clamp_eta(other, irls_eta[i]);
                    let (mu, w_raw, r) =
                        crate::family::irls_weight_and_resid(other, nb_theta, y[i], e);
                    irls_p[i] = mu;
                    irls_w[i] = w_raw.max(WEIGHT_CLAMP);
                    irls_z[i] = e + r;
                    dev += crate::family::dev_resid(other, nb_theta, y[i], mu);
                }
                dev
            }
        };

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
        // rayon loop). GEMM accumulation order, deliberately NOT the old
        // per-entry row-order dots: the serial FP-add chain was the latency
        // floor on wide p (measured 0.94× glm_wide).
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
        // β_new is accepted unconditionally — no step-halving: it causes
        // systematic power drift at small N (measured ~3.5% at N=50). The
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

        // BETA_CAP divergence guard at iter ≥ 3. Fires before the next
        // pass's convergence check — a capped β never reports converged.
        if iter >= 3 {
            let mut max_abs: f64 = 0.0;
            for &b in &irls_betas[..p] {
                let ab = b.abs();
                if ab > max_abs {
                    max_abs = ab;
                }
            }
            if max_abs > BETA_CAP {
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
        let saturated = irls_w[..n]
            .iter()
            .filter(|&&w_i| w_i < SATURATION_W)
            .count();
        if (saturated as f64) / (n as f64) > SATURATION_FRAC {
            converged = false;
        }
    }

    // If not converged: NaN the inference outputs but leave betas at whatever
    // value the loop last computed.
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

    // Per-target Var(β̂_j) = ((X'WX)⁻¹)_jj via forward solve on L:
    //   L · u = e_{tj} → var_diag = ‖u‖²
    // (No σ̂² scaling — under Bernoulli the score covariance is (X'WX)⁻¹.)
    for (out_idx, &tj) in target_indices.iter().enumerate() {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        irls_u_scratch[..p].fill(0.0);
        for i in 0..p {
            let b_i = if i == tj { 1.0 } else { 0.0 };
            let mut acc = b_i;
            for k in 0..i {
                acc -= irls_l[(i, k)] * irls_u_scratch[k];
            }
            let l_ii = irls_l[(i, i)];
            irls_u_scratch[i] = if l_ii.abs() < FLOAT_NEAR_ZERO {
                f64::NAN
            } else {
                acc / l_ii
            };
        }
        let mut norm_sq = 0.0;
        for &v in &irls_u_scratch[..p] {
            norm_sq += v * v;
        }
        irls_var_diag[out_idx] = norm_sq;
        if norm_sq > FLOAT_NEAR_ZERO && norm_sq.is_finite() {
            let beta_j = irls_betas[tj];
            irls_t_sq[out_idx] = (beta_j * beta_j) / norm_sq;
        } else {
            irls_t_sq[out_idx] = f64::NAN;
        }
    }

    GlmFitView {
        betas: &irls_betas[..p],
        var_diag: &irls_var_diag[..t],
        t_sq: &irls_t_sq[..t],
        l: irls_l.into_const(),
        n_iter,
        converged: true,
        deviance: deviance_final,
        deviance_null,
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
        // diverges (β → ∞); BETA_CAP or saturation guard should fire.
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
            glm_scratch(&mut ws),
        );
        assert!(
            !fit.converged,
            "fully separated data must report non-converged"
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
}
