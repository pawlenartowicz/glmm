//! Packed-row PIRLS solve (`pirls_solve_packed`).

use super::*;

/// The packed-row Laplace layout: `M = ZΛ` is stored as `width` nonzeros per
/// row (`width = q_p + Σ q_g`, since every row loads exactly one level of every
/// grouping), and `A = M'WM + I` is a dense `k×k` accumulated by scattering each
/// row's `width²` outer-product entries. Borrowed from the buffers
/// `pirls_solve_packed` destructures.
///
/// The dense `k×k` factor rather than a block kernel: `W` changes every inner
/// iteration, so a θ-independent packing of `Λ'Z'ZΛ` would be rebuilt per step
/// anyway. The `O(n·width²)` weighted Gram accumulation dominates and stays
/// sparse; only the `k×k` factor is dense.
pub(crate) struct PackedFactor<'a> {
    /// len `n·width`: the RE column of each nonzero, row `i` at `i·width`.
    pub(crate) m_cols: &'a [u32],
    /// len `n·width`: `(ZΛ)` at that column, same indexing as `m_cols`.
    pub(crate) m_vals: &'a [f64],
    /// `k×k`, full symmetric — both triangles are written. `factor_logdet`
    /// leaves it holding `A` itself (`+I` applied, un-factored) and factors the
    /// separate `a_chol` copy, so `packed_schur_fill` can re-read it.
    pub(crate) a: &'a mut Mat<f64>,
    /// Copy-then-factor target for `a`'s lower triangle.
    pub(crate) a_chol: &'a mut Mat<f64>,
    /// Scratch for `a_chol`'s in-place factor and back-solve.
    pub(crate) a_llt_mem: &'a mut MemBuffer,
    /// len `k`: `M'r`, then `A⁻¹M'r` in place.
    pub(crate) a_rhs: &'a mut [f64],
    /// len `n`: `(Mu)ᵢ` at the current iterate, written by `eta_from_mode` and
    /// read by `scatter`'s IRLS right-hand side. It is NOT overwritten by the
    /// working residual — the scatter folds `w·Mu + ρ` per row as it goes.
    pub(crate) mu: &'a mut [f64],
    pub(crate) family: Family,
    pub(crate) nb_theta: f64,
    /// Nonzeros per packed row.
    pub(crate) width: usize,
    /// Total RE columns, the side of `a`.
    pub(crate) k: usize,
}

impl LaplaceFactor<f64> for PackedFactor<'_> {
    fn eta_from_mode(
        &mut self,
        u: &[f64],
        eta_fixed: &[f64],
        eta: &mut [f64],
        y: &[f64],
        n: usize,
    ) -> f64 {
        let (m_cols, m_vals, width) = (self.m_cols, self.m_vals, self.width);
        let mut yeta = 0.0;
        for i in 0..n {
            let base = i * width;
            let mut acc = 0.0;
            for t in base..base + width {
                acc += m_vals[t] * u[m_cols[t] as usize];
            }
            self.mu[i] = acc;
            eta[i] = eta_fixed[i] + acc;
            yeta += y[i] * eta[i];
        }
        yeta
    }

    fn scatter(
        &mut self,
        w: &[f64],
        prob: &[f64],
        eta: &[f64],
        y: &[f64],
        prior_w: &[f64],
        _weighted: bool,
        n: usize,
        dual: Option<&mut DualStep<f64>>,
    ) {
        // This layout carries no observed-information twin: it serves only the
        // `f64` fit path, whose derivative requests are refused by
        // `derivative::supports_shape`.
        debug_assert!(dual.is_none());
        let (m_cols, m_vals, width) = (self.m_cols, self.m_vals, self.width);
        let (family, nb_theta, k) = (self.family, self.nb_theta, self.k);
        let a = &mut *self.a;
        let a_rhs = &mut *self.a_rhs;
        for c in 0..k {
            for r in 0..k {
                a[(r, c)] = 0.0;
            }
            a_rhs[c] = 0.0;
        }
        // `wᵢmᵢmᵢ'` into A and `mᵢ·(wᵢ(Mu)ᵢ + ρᵢ)` into the RHS, from each row's
        // `width` nonzeros. This pass is scalar per row for every family — no
        // logit shortcut and no batched kernel: the effective residual
        // `ρ = w·(dμ/dη)·(y−μ)/V` is formed here beside the gather. The exit
        // refresh's μ/W/deviance pass is `Scalar::family_pass`, which does take
        // the fused logit and SIMD arms.
        for i in 0..n {
            let wi = w[i];
            let dmu = crate::family::mu_eta(family, eta[i]);
            let v = crate::family::variance(family, nb_theta, prob[i]);
            // `0/0` on an unweighted logit row whose `prob` is exactly 0 or 1. Only
            // the exit refresh passes an unclamped `prob`, and it reads `A` alone:
            // `a_rhs` is zeroed at the top of the next scatter before any solve.
            let rho = prior_w[i] * dmu * (y[i] - prob[i]) / v;
            let q_i = wi * self.mu[i] + rho;
            let base = i * width;
            for ta in base..base + width {
                let ca = m_cols[ta] as usize;
                let va = m_vals[ta];
                let wva = wi * va;
                for tb in base..base + width {
                    let cb = m_cols[tb] as usize;
                    a[(ca, cb)] += wva * m_vals[tb];
                }
                a_rhs[ca] += va * q_i;
            }
        }
    }

    fn factor_logdet(&mut self) -> Option<f64> {
        let k = self.k;
        for r in 0..k {
            self.a[(r, r)] += 1.0;
        }
        self.a_chol.copy_from_triangular_lower(self.a.as_ref());
        if cholesky_in_place(
            self.a_chol.as_mut(),
            LltRegularization::default(),
            Par::Seq,
            MemStack::new(self.a_llt_mem),
            Spec::default(),
        )
        .is_err()
        {
            return None;
        }
        let mut logdet = 0.0;
        for r in 0..k {
            logdet += self.a_chol[(r, r)].ln();
        }
        Some(logdet)
    }

    fn solve_in_place(&mut self) {
        let k = self.k;
        solve_in_place(
            self.a_chol.as_ref(),
            MatMut::from_column_major_slice_mut(&mut self.a_rhs[..k], k, 1),
            Par::Seq,
            MemStack::new(self.a_llt_mem),
        );
    }
}

/// Penalized-IRLS inner solve on the packed `M` rows by Fisher scoring on the
/// penalized likelihood, with `M = ZΛ` the scaled RE design and a `+I` ridge
/// (the nAGQ=1 reparameterization, `u ~ N(0, I)`). Each step assembles
/// `A = M'WM + I` and the IRLS right-hand side `M'(W·Mu + W·r)` from the
/// `width` nonzeros per row and takes `u ← A⁻¹M'r` through a dense `k×k`
/// Cholesky. Returns `(deviance, ‖ũ‖², log|L|` at the returned iterate,
/// converged`)`; a Cholesky failure surfaces as `(NaN, NaN, NaN, false)`.
/// A converged solve ends by re-evaluating η/μ/W at the returned `u` and
/// rebuilding and refactoring `A` there ([`evaluate_at_mode`]), so `log|L|`
/// (`log|A| = 2 log|L|`) and the deviance describe that iterate, and
/// `packed.a` is left holding its raw symmetric `A` for
/// `se::packed_schur_fill` to re-factor.
///
/// The `beta_step` mode sets what moves: `BetaStep::Fixed` holds β at the
/// caller's input and solves for the conditional modes ũ(β) alone (the
/// objective stays a function of the caller's β — required by the joint-Hessian
/// SE path and BOBYQA stage 2); `BetaStep::Profile` adds a joint δβ Schur-border
/// step each iteration (`T = A⁻¹B`, `S_β = C − B'T`, `δβ = S_β⁻¹(X'ρ − B'δu₀)`,
/// then `u ← u_new − T·δβ`) so the returned `(ũ, β̂)` is jointly PQL-optimal for
/// this θ, writing β̂ back through `beta`.
///
/// **Step-halving (lme4 `pwrssUpdate`, retrospective form):** each iteration
/// evaluates the trial `u` first; only if the same-point penalized deviance
/// `dev + ‖u‖²` rose above the last accepted value BY MORE than the tol band
/// does it halve `δu = u − u_prev` and re-evaluate, up to `PIRLS_MAX_HALVINGS`
/// times. A within-band rise is FP noise near the optimum and is accepted — it
/// never burns a halving. In `Profile` mode the joint `(u, β)` step is
/// backtracked in lockstep, halving β toward `beta_prev` alongside u. A
/// domain-infeasible trial η (`family::eta_infeasible`, which names
/// Gamma-inverse and inverse-Gaussian-inverse-squared)
/// halves regardless of the band. Convergence is the mixed
/// `dev(uⱼ) + ‖uⱼ₊₁‖²` band on successive steps, checked after the step.
///
/// Iterates from whatever `scratch.u` holds on entry — the caller owns reset
/// vs. warm start.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pirls_solve_packed(
    family: Family,
    nb_theta: f64,
    k: usize,
    p: usize,
    x: MatRef<f64>,
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    beta: &mut [f64],
    mut beta_step: BetaStep,
    scratch: &mut PirlsScratch<f64>,
    // Packed-row layout scratch — see [`PackedScratch`]. `m_vals` must be
    // refilled at this θ before the call; `a` must survive it holding the raw
    // symmetric `M'WM + I`, so the Cholesky runs on the separate `a_chol`.
    packed: &mut PackedScratch,
    // n × p = W∘X GEMM scratch for the Profile β-Schur border's C = X'WX
    // (rebuilt fresh from this iteration's `w` each Profile step, so no
    // stale-value hazard across PIRLS iterations).
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`FitOptions::offset`), added into
    // `eta_fixed` by every `refresh_eta_fixed` call. `None` ⇒ no offset.
    offset: Option<&[f64]>,
    pirls_tol_override: Option<f64>,
    n: usize,
    // Observation-only: the iteration index of the solve in progress, written
    // every iteration so an early return still leaves the right value pending.
    counters: &mut crate::counters::EvalCounters,
) -> (f64, f64, f64, bool) {
    // The exact Laplace β-profile differentiates `log|A|` through the layout's
    // own factor; this layout has no such pass. Loud, not silent: a silently
    // dropped correction would be a wrong objective, not a slow one.
    assert!(
        !matches!(beta_step, BetaStep::Profile { exact: Some(_), .. }),
        "the packed-row layout has no exact Laplace β-profile"
    );
    let PirlsScratch {
        eta,
        prob,
        w,
        u,
        u_prev,
        eta_fixed,
        mu,
        a_rhs,
        ..
    } = scratch;
    let width = packed.width;
    let mut layout = PackedFactor {
        m_cols: &packed.m_cols[..n * width],
        m_vals: &packed.m_vals[..n * width],
        a: &mut packed.a,
        a_chol: &mut packed.a_chol,
        a_llt_mem: &mut packed.a_llt_mem,
        a_rhs: &mut a_rhs[..],
        mu: &mut mu[..],
        family,
        nb_theta,
        width,
        k,
    };
    // η_fixed,ᵢ = Σ_j x[i,j]·β[j]. In Fixed mode β is invariant across iterations
    // so this once-at-entry fill stands for the whole solve; in Profile mode the
    // δβ step re-fills it after every β update.
    refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
    // Backtrack seeds for the FIRST trial iterate (which has no accepted
    // predecessor): u_prev = 0 so an infeasible first trial halves toward
    // η = eta_fixed (the canonical cold seed), beta_prev = the caller's β. Dead
    // for the overshoot trigger — it cannot fire before an accept — so only the
    // domain-infeasibility trigger ever reads these seeds.
    u_prev[..k].fill(0.0);
    if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
        beta_prev[..p].copy_from_slice(&beta[..p]);
    }
    let mut pen_accepted = f64::INFINITY; // same-point penalized deviance at the last ACCEPTED iterate
    let mut mixed_prev = f64::INFINITY; // the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` from the previous step
    let mut halvings = 0usize;
    let mut converged = false;
    let mut dev = f64::NAN;
    let mut pen = f64::NAN; // ‖u‖² at the returned (post-step) iterate
    let mut logdet = 0.0;
    let tol = pirls_tol_override.unwrap_or_else(|| super::super::pirls_tol(family));
    for it in 0..PIRLS_MAX_ITERS {
        counters.set_pirls_iters(it + 1);
        // --- trial evaluation at the CURRENT u: (Mu)ᵢ and the raw η, then
        // μ/W/deviance per row. On a fresh accept this is the newly-stepped u;
        // after a halving `continue` it is the backtracked u. `infeasible` flags
        // any RAW η outside the link's open domain — the two
        // `family::eta_infeasible` names, Gamma-inverse and
        // inverse-Gaussian-inverse-squared. ---
        // The returned `Σ yᵢηᵢ` is dropped: this loop forms the deviance row by
        // row below instead of calling `T::family_pass`, so nothing consumes it.
        layout.eta_from_mode(&u[..], &eta_fixed[..], &mut eta[..], y, n);
        dev = 0.0;
        let mut infeasible = false;
        for i in 0..n {
            let raw = eta[i];
            infeasible |= crate::family::eta_infeasible(family, raw);
            let e = crate::family::clamp_eta(family, raw);
            eta[i] = e;
            // Canonical-link shortcut (Poisson-log) lives inside this call — see
            // `irls_weight_and_resid`'s doc comment.
            let (mui, wi, _) = crate::family::irls_weight_and_resid(family, nb_theta, y[i], e);
            prob[i] = mui;
            w[i] = (prior_w[i] * wi).max(crate::glm::WEIGHT_CLAMP);
            dev += prior_w[i] * crate::family::dev_resid(family, nb_theta, y[i], mui);
        }
        // Band-tolerant retrospective step-halving: the convergence band is
        // consulted before any halving, because near the optimum Fisher scoring
        // is not strictly monotone — a step can land ε above `pen_accepted` yet
        // inside the tol band, and that must converge rather than burn all ten
        // halvings against FP noise. Only a rise EXCEEDING the band is a genuine
        // overshoot. A domain-infeasible trial is a step failure regardless of
        // the band: accepting it would let `clamp_eta`'s boundary projection
        // define the converged answer.
        let pen_u: f64 = u[..k].iter().map(|v| v * v).sum();
        let penalized = dev + pen_u;
        if infeasible || penalized - pen_accepted > tol * (1.0 + penalized.abs()) {
            if halvings < PIRLS_MAX_HALVINGS {
                halvings += 1;
                for c in 0..k {
                    u[c] = 0.5 * (u[c] + u_prev[c]);
                }
                if let BetaStep::Profile { beta_prev, .. } = &beta_step {
                    for j in 0..p {
                        beta[j] = 0.5 * (beta[j] + beta_prev[j]);
                    }
                    refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
                }
                continue;
            }
            return (f64::NAN, f64::NAN, f64::NAN, false); // halvings exhausted
        }
        // Accept this iterate, snapshot it for the next backtrack, and take a
        // fresh full Fisher step from it (cold start: pen_accepted = ∞ ⇒ always
        // accepts).
        halvings = 0;
        pen_accepted = penalized;
        u_prev[..k].copy_from_slice(&u[..k]);
        if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
            beta_prev[..p].copy_from_slice(&beta[..p]);
        }
        layout.scatter(&w[..], &prob[..], &eta[..], y, prior_w, weighted, n, None);
        // Profile mode: accumulate the β-gradient X'ρ (ρ = effective residual)
        // into `beta_rhs` — the joint system's bottom-block RHS. Its own pass off
        // the fresh prob/eta, so the Fixed path never pays for it.
        if let BetaStep::Profile { beta_rhs, .. } = &mut beta_step {
            for v in beta_rhs[..p].iter_mut() {
                *v = 0.0;
            }
            for i in 0..n {
                let dmu = crate::family::mu_eta(family, eta[i]);
                let v = crate::family::variance(family, nb_theta, prob[i]);
                let rho = prior_w[i] * dmu * (y[i] - prob[i]) / v;
                for j in 0..p {
                    beta_rhs[j] += x[(i, j)] * rho;
                }
            }
        }
        // `+I`, factor, `log|A|` off the factor that produces this step's u_new.
        // The exit refresh below replaces both with their values at the
        // returned u.
        logdet = match layout.factor_logdet() {
            Some(l) => l,
            None => return (f64::NAN, f64::NAN, f64::NAN, false),
        };
        layout.solve_in_place();
        pen = 0.0;
        for (uc, &a) in u[..k].iter_mut().zip(layout.a_rhs.iter()) {
            *uc = a;
            pen += a * a;
        }
        // --- Profile-mode joint δβ step (β-Schur border), taken while the LLT of
        // A is still alive. `u` currently holds u_new = u_prev + δu₀, so
        // δu₀ = u − u_prev. ---
        if let BetaStep::Profile {
            xtwx,
            xtwm,
            ainv_mtwx,
            schur,
            beta_rhs,
            schur_llt_mem,
            ..
        } = &mut beta_step
        {
            // B' = X'WM (p×k) by per-row scatter over the packed nonzeros.
            for r in 0..p {
                for c in 0..k {
                    xtwm[(r, c)] = 0.0;
                }
            }
            for i in 0..n {
                let wi = w[i];
                let base = i * width;
                for r in 0..p {
                    let xw = x[(i, r)] * wi;
                    for t in base..base + width {
                        xtwm[(r, layout.m_cols[t] as usize)] += xw * layout.m_vals[t];
                    }
                }
            }
            // C = X'WX (p×p) via the W∘X GEMM scratch `wx`, recomputed each
            // iteration because W changes with the working weights. Kept
            // full-symmetric — the border below reads the whole matrix.
            for r in 0..p {
                for i in 0..n {
                    wx[(i, r)] = w[i] * x[(i, r)];
                }
            }
            faer::linalg::matmul::matmul(
                xtwx.as_mut(),
                faer::Accum::Replace,
                x.transpose(),
                wx.as_ref(),
                1.0,
                Par::Seq,
            );
            // T = A⁻¹B = A⁻¹(M'WX): transpose-gather B' then solve with this
            // iteration's factor.
            for r in 0..k {
                for c in 0..p {
                    ainv_mtwx[(r, c)] = xtwm[(c, r)];
                }
            }
            solve_in_place(
                layout.a_chol.as_ref(),
                ainv_mtwx.as_mut(),
                Par::Seq,
                MemStack::new(layout.a_llt_mem),
            );
            // S_β = C − B'·T.
            for r in 0..p {
                for c in 0..p {
                    let mut s = xtwx[(r, c)];
                    for j in 0..k {
                        s -= xtwm[(r, j)] * ainv_mtwx[(j, c)];
                    }
                    schur[(r, c)] = s;
                }
            }
            // rhs = X'ρ − B'·δu₀ (beta_rhs holds X'ρ; δu₀ = u − u_prev).
            for r in 0..p {
                let mut acc = 0.0;
                for c in 0..k {
                    acc += xtwm[(r, c)] * (u[c] - u_prev[c]);
                }
                beta_rhs[r] -= acc;
            }
            // δβ = S_β⁻¹·rhs in place. Non-PD S_β ⇒ the (NaN,…,false) failure surface.
            if cholesky_in_place(
                schur.as_mut(),
                LltRegularization::default(),
                Par::Seq,
                MemStack::new(schur_llt_mem),
                Spec::default(),
            )
            .is_err()
            {
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
            solve_in_place(
                schur.as_ref(),
                MatMut::from_column_major_slice_mut(&mut beta_rhs[..p], p, 1),
                Par::Seq,
                MemStack::new(schur_llt_mem),
            );
            // Apply: β += δβ; u = u_joint = u_new − T·δβ.
            for j in 0..p {
                beta[j] += beta_rhs[j];
            }
            for c in 0..k {
                let mut acc = 0.0;
                for j in 0..p {
                    acc += ainv_mtwx[(c, j)] * beta_rhs[j];
                }
                u[c] -= acc;
            }
            // η_fixed depends on β; refresh it for the next trial evaluation.
            // `pen` must track the moved u (‖u_joint‖²), so recompute it.
            refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
            pen = u[..k].iter().map(|v| v * v).sum();
        }
        // The stopping rule: the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` band on successive
        // steps, read off the loop's own assembly point — the exit refresh
        // runs after this decision and never enters it.
        let mixed = dev + pen;
        if (mixed - mixed_prev).abs() < tol * (1.0 + mixed.abs()) {
            converged = true;
            break;
        }
        mixed_prev = mixed;
    }
    // The returned `dev`, `log|A|` and factor at the returned iterate, so the
    // objective's three terms and the factor `packed_schur_fill` inherits
    // describe one point — see [`evaluate_at_mode`].
    if converged {
        match evaluate_at_mode(
            &mut layout,
            family,
            nb_theta,
            y,
            prior_w,
            weighted,
            &eta_fixed[..],
            &u[..],
            &mut eta[..],
            &mut prob[..],
            &mut w[..],
            n,
        ) {
            Some((d, ld)) => {
                dev = d;
                logdet = ld;
            }
            None => return (f64::NAN, f64::NAN, f64::NAN, false),
        }
    }
    (dev, pen, logdet, converged)
}

/// Refill the packed `M` values at the current Λ (`packed.lam_small` must be
/// filled for this θ): entry `c` of a row's block is the lower-tri fold
/// `Σ_{r≥c} z[r]·Λ[r,c]` with `z = [1, x[slope cols…]]`, the same `ZΛ` sandwich
/// every other layout writes. The Z-side factor is indexed by the Λ ROW `r`,
/// not the column `c`: it takes the RE column's internal scale
/// (`LmmGroupings::set_slope_scales`) while `lam_small` is the θ side and takes
/// none. The intercept's scale is exactly 1.
pub(crate) fn fill_m_vals(
    packed: &mut PackedScratch,
    g: &crate::lmm::LmmGroupings,
    x: MatRef<f64>,
    n: usize,
) {
    let q_p = g.primary_q;
    let width = packed.width;
    for i in 0..n {
        let mut t = i * width;
        for c in 0..q_p {
            let mut acc = 0.0;
            for r in c..q_p {
                let z = if r == 0 {
                    1.0
                } else {
                    x[(i, g.primary_slope_cols[r - 1])] / g.primary_slope_scales[r - 1]
                };
                acc += z * packed.lam_small[r * q_p + c];
            }
            packed.m_vals[t] = acc;
            t += 1;
        }
        for e in 0..g.extra_offsets.len() {
            let q_g = g.extra_q[e];
            let lo = packed.lam_off_decl[e];
            for c in 0..q_g {
                let mut acc = 0.0;
                for r in c..q_g {
                    let z = if r == 0 {
                        1.0
                    } else {
                        x[(i, g.extra_slope_cols[e][r - 1])] / g.extra_slope_scales[e][r - 1]
                    };
                    acc += z * packed.lam_small[lo + r * q_g + c];
                }
                packed.m_vals[t] = acc;
                t += 1;
            }
        }
    }
}

/// `∂m_vals[i·width + t]/∂θ_a` for one packed row, written into `out`
/// (`width` long) in [`fill_m_vals`]'s own slot order.
///
/// `Λ` is linear in θ, so this is a selection rather than a derivative of
/// anything: θ_a is exactly one `(r, c)` vech slot of exactly one grouping's
/// `Λ`, so the fold can carry it into exactly one slot of the row — that
/// grouping's slot `c`, holding the `z_r` the fill multiplies it by. `z` is
/// indexed by the Λ ROW, as in the fill, so it carries that RE column's
/// internal scale (`LmmGroupings::set_slope_scales`) while the θ side takes
/// none. Every other slot is zero.
///
/// One walk finds both the owning grouping and its slot base because the three
/// orders agree: θ is the primary vech then each extra grouping's vech in
/// DECLARATION order ([`crate::lmm::LmmGroupings::n_theta`]), which is the
/// order `sparse::fill_lambda_small` slices θ in and the order [`fill_m_vals`]
/// walks a row's blocks in. No grouping is skipped: on this layout both of
/// those fills treat a θ-pinned grouping like any other — its `Λ` block is
/// simply zero and its packed slot stays in the row — so, unlike the
/// structured packer's `workspace::packed_m_theta_deriv`, there is no pin arm
/// here and no scalar-dependent column order to pair against.
///
/// Mirrors [`fill_m_vals`] — change together.
pub(crate) fn packed_m_vals_theta_deriv(
    g: &crate::lmm::LmmGroupings,
    a: usize,
    x: MatRef<f64>,
    i: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    let q_p = g.primary_q;
    let base_theta = q_p * (q_p + 1) / 2;
    if a < base_theta {
        // Invert the column-major vech enumeration `lmm::primary_lambda`
        // writes (`c` outer, `r` inner from `r == c`).
        let mut t = 0;
        #[allow(clippy::needless_range_loop)]
        for c in 0..q_p {
            for r in c..q_p {
                if t == a {
                    out[c] = if r == 0 {
                        1.0
                    } else {
                        x[(i, g.primary_slope_cols[r - 1])] / g.primary_slope_scales[r - 1]
                    };
                    return;
                }
                t += 1;
            }
        }
        return;
    }
    let mut slot = q_p;
    let mut vech = base_theta;
    for (e, &q_g) in g.extra_q.iter().enumerate() {
        let len = q_g * (q_g + 1) / 2;
        if a < vech + len {
            let local = a - vech;
            let mut t = 0;
            for c in 0..q_g {
                for r in c..q_g {
                    if t == local {
                        out[slot + c] = if r == 0 {
                            1.0
                        } else {
                            x[(i, g.extra_slope_cols[e][r - 1])] / g.extra_slope_scales[e][r - 1]
                        };
                        return;
                    }
                    t += 1;
                }
            }
            return;
        }
        slot += q_g;
        vech += len;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lmm::LmmGroupings;

    /// [`packed_m_vals_theta_deriv`] must reproduce a central difference of
    /// [`fill_m_vals`] in every θ coordinate and every slot of every row, on
    /// a packed shape carrying a random slope on the PRIMARY grouping and a
    /// random slope on an EXTRA one — the two blocks whose vech walks, `Λ`
    /// offsets and `z` columns the selection has to keep apart. The slope
    /// scales are set from the design (`set_slope_scales`), so both columns
    /// carry a scale other than 1 and the `z_r / s_r` division is visible
    /// rather than an identity.
    #[test]
    fn packed_m_vals_theta_deriv_matches_fill_m_vals_difference() {
        let n = 16;
        let mut x = Mat::<f64>::zeros(n, 3);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = 0.7 + 0.31 * i as f64;
            x[(i, 2)] = 2.5 - 0.17 * (i % 7) as f64;
        }
        let spec = crate::ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![1],
                extra_groupings: vec![crate::Grouping {
                    relation: crate::GroupingRelation::Crossed { n_clusters: 3 },
                    slopes: vec![2],
                }],
            }),
        };
        let mut g = LmmGroupings::from_cluster_spec_ext(&spec, n, &[1], &[vec![2]]);
        g.set_slope_scales(x.as_ref(), None);
        assert!(
            g.primary_slope_scales[0] != 1.0 && g.extra_slope_scales[0][0] != 1.0,
            "both slope columns must carry a non-unit scale for this test to see it"
        );
        let mut packed = PackedScratch::for_shape(&g, n, g.k_total);
        let width = packed.width;
        let n_theta = g.n_theta();
        let mut theta: Vec<f64> = (0..n_theta).map(|k| 0.4 + 0.11 * k as f64).collect();
        let eps = 1e-6;

        for a in 0..n_theta {
            let base = theta[a];
            theta[a] = base + eps;
            crate::sparse::fill_lambda_small(&theta, &g, &mut packed.lam_small);
            fill_m_vals(&mut packed, &g, x.as_ref(), n);
            let plus = packed.m_vals[..n * width].to_vec();
            theta[a] = base - eps;
            crate::sparse::fill_lambda_small(&theta, &g, &mut packed.lam_small);
            fill_m_vals(&mut packed, &g, x.as_ref(), n);
            let minus = packed.m_vals[..n * width].to_vec();
            theta[a] = base;

            let mut d = vec![0.0; width];
            for i in 0..n {
                packed_m_vals_theta_deriv(&g, a, x.as_ref(), i, &mut d);
                for t in 0..width {
                    let want = (plus[i * width + t] - minus[i * width + t]) / (2.0 * eps);
                    assert!(
                        (d[t] - want).abs() < 1e-8,
                        "a={a} i={i} t={t}: selection {} vs central diff {want}",
                        d[t]
                    );
                }
            }
        }
    }

    /// The exit refresh evaluates the loop's post-step iterate, which no trial
    /// evaluation has passed, so a raw η outside the link's open domain can
    /// reach [`evaluate_at_mode`]. It must refuse such a point instead of
    /// reporting the `clamp_eta`-projected boundary point's deviance and
    /// `log|A|` as the converged answer. Gamma-inverse here: `η = 1/μ`, so
    /// `η ≤ 0` is outside the domain (`family::eta_infeasible`).
    #[test]
    fn evaluate_at_mode_refuses_an_infeasible_eta() {
        let (n, s) = (6usize, 2usize);
        let x = Mat::<f64>::from_fn(n, 1, |_, _| 1.0);
        let spec = crate::ModelSpec {
            family: Family::Gamma {
                link: crate::GammaLink::Inverse,
            },
            re: Some(crate::ReStructure {
                sizing: crate::Sizing::FixedClusters {
                    n_clusters: s as u32,
                },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        let g = LmmGroupings::from_cluster_spec_ext(&spec, n, &[], &[]);
        let k = g.k_total;
        let mut packed = PackedScratch::for_shape(&g, n, k);
        crate::sparse::fill_lambda_small(&[0.5], &g, &mut packed.lam_small);
        fill_m_vals(&mut packed, &g, x.as_ref(), n);
        // Intercept-only primary: one nonzero per row, at that row's cluster.
        for i in 0..n {
            packed.m_cols[i] = (i % s) as u32;
        }
        let width = packed.width;
        let y = vec![1.0; n];
        let prior_w = vec![1.0; n];
        let u = vec![0.0; k];
        // u = 0, so η is η_fixed: every row feasible but one sitting past the
        // boundary, the shape a post-step iterate takes when the step overshoots.
        let mut eta_fixed = vec![1.0; n];
        eta_fixed[2] = -0.25;
        let mut eta = vec![0.0; n];
        let mut prob = vec![0.0; n];
        let mut w = vec![0.0; n];
        let mut a_rhs = vec![0.0; k];
        let mut mu = vec![0.0; n];
        let mut layout = PackedFactor {
            m_cols: &packed.m_cols[..n * width],
            m_vals: &packed.m_vals[..n * width],
            a: &mut packed.a,
            a_chol: &mut packed.a_chol,
            a_llt_mem: &mut packed.a_llt_mem,
            a_rhs: &mut a_rhs[..],
            mu: &mut mu[..],
            family: spec.family,
            nb_theta: f64::NAN,
            width,
            k,
        };
        assert!(
            evaluate_at_mode(
                &mut layout,
                spec.family,
                f64::NAN,
                &y,
                &prior_w,
                false,
                &eta_fixed,
                &u,
                &mut eta,
                &mut prob,
                &mut w,
                n,
            )
            .is_none(),
            "a refreshed iterate with η outside the link's domain must not be reported as an evaluation"
        );
    }
}
