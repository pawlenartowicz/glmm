//! Packed-row PIRLS solve (`pirls_solve_packed`).

use super::*;

/// The packed-row Laplace layout: `M = ZΛ` is stored as `width` nonzeros per
/// row (`width = q_p + Σ q_g`, since every row loads exactly one level of every
/// grouping), and `A = M'WM + I` is a dense `k×k` accumulated by scattering each
/// row's `width²` outer-product entries. Borrowed from the buffers
/// `pirls_solve_packed` destructures.
///
/// `A` is factored whole, either dense (`a_chol`) or, when `fill_packed_cols`
/// built one, through the sparse Cholesky on `A`'s fixed pattern
/// ([`crate::glmm::workspace::PackedSparse`]): `W` changes every inner
/// iteration but the pattern does not, so one symbolic analysis per design
/// serves every numeric refactor. The `O(n·width²)` weighted Gram
/// accumulation scatters straight into whichever storage is in use.
pub(crate) struct PackedFactor<'a> {
    /// len `n·width`: the RE column of each nonzero, row `i` at `i·width`.
    pub(crate) m_cols: &'a [u32],
    /// len `n·width`: `(ZΛ)` at that column, same indexing as `m_cols`.
    pub(crate) m_vals: &'a [f64],
    /// `k×k`, full symmetric — both triangles are written. `factor_logdet`
    /// leaves it holding `A` itself (`+I` applied, un-factored) and factors the
    /// separate `a_chol` copy, so `packed_schur_fill` can re-read it. Dense
    /// factor only: `0×0` and never touched when `sp` is `Some`.
    pub(crate) a: &'a mut Mat<f64>,
    /// Copy-then-factor target for `a`'s lower triangle (dense factor only).
    pub(crate) a_chol: &'a mut Mat<f64>,
    /// Scratch for `a_chol`'s in-place factor and back-solve.
    pub(crate) a_llt_mem: &'a mut MemBuffer,
    /// len `k`: `M'ρ`, then the increment right-hand side `M'ρ − u`, then
    /// `A⁻¹(M'ρ − u)` in place.
    pub(crate) a_rhs: &'a mut [f64],
    pub(crate) family: Family,
    pub(crate) nb_theta: f64,
    /// Nonzeros per packed row.
    pub(crate) width: usize,
    /// Total RE columns, the side of `a`.
    pub(crate) k: usize,
    /// When present, `scatter` writes `A` into its CSC values and
    /// `factor_logdet` and the solves run through this sparse Cholesky instead
    /// of the dense `a`/`a_chol`, which are then `0×0`. After a converged
    /// solve `axx` holds the final `A` and `l_values` its completed factor
    /// (`factor_ok`), and `packed_schur_fill` solves through that factor
    /// directly, with no dense expansion of `A`.
    pub(crate) sp: Option<&'a mut crate::glmm::workspace::PackedSparse>,
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
        if let Some(sp) = self.sp.as_deref_mut() {
            let a_rhs = &mut *self.a_rhs;
            a_rhs[..k].fill(0.0);
            let row_slots = &sp.row_slots;
            let vals = sp.axx.val_mut();
            vals.fill(0.0);
            let mut probit_rows = crate::family::ProbitRows::new(family, eta, prob);
            for i in 0..n {
                let wi = w[i];
                let rho = super::row_rho(
                    family,
                    nb_theta,
                    i,
                    y[i],
                    prior_w[i],
                    eta[i],
                    prob[i],
                    probit_rows.as_mut(),
                );
                let base = i * width;
                for ta in base..base + width {
                    let ca = m_cols[ta] as usize;
                    let va = m_vals[ta];
                    let wva = wi * va;
                    let slots = &row_slots[ta * width..(ta + 1) * width];
                    for (tb, &s) in (base..base + width).zip(slots) {
                        if s != u32::MAX {
                            vals[s as usize] += wva * m_vals[tb];
                        }
                    }
                    a_rhs[ca] += va * rho;
                }
            }
            return;
        }
        let a = &mut *self.a;
        let a_rhs = &mut *self.a_rhs;
        for c in 0..k {
            for r in 0..k {
                a[(r, c)] = 0.0;
            }
            a_rhs[c] = 0.0;
        }
        // `wᵢmᵢmᵢ'` into A and `mᵢ·ρᵢ` into the RHS, from each row's
        // `width` nonzeros. This pass is scalar per row, with no batched kernel
        // but probit's `φ`, a tile at a time (`family::ProbitRows`): the row
        // score ρᵢ (`row_rho`) is formed here beside the gather. The
        // μ/W/deviance pass before it is `Scalar::family_pass`, which does take
        // the fused logit and SIMD arms.
        let mut probit_rows = crate::family::ProbitRows::new(family, eta, prob);
        for i in 0..n {
            let wi = w[i];
            let rho = super::row_rho(
                family,
                nb_theta,
                i,
                y[i],
                prior_w[i],
                eta[i],
                prob[i],
                probit_rows.as_mut(),
            );
            let base = i * width;
            for ta in base..base + width {
                let ca = m_cols[ta] as usize;
                let va = m_vals[ta];
                let wva = wi * va;
                for tb in base..base + width {
                    let cb = m_cols[tb] as usize;
                    a[(ca, cb)] += wva * m_vals[tb];
                }
                a_rhs[ca] += va * rho;
            }
        }
    }

    fn factor_logdet(&mut self) -> Option<f64> {
        let k = self.k;
        if let Some(sp) = self.sp.as_deref_mut() {
            // Cleared before the factorization: a failed one leaves `l_values`
            // partly stale, and only this flag tells `packed_schur_fill` so.
            sp.factor_ok = false;
            {
                let vals = sp.axx.val_mut();
                for &s in &sp.diag_slots {
                    vals[s as usize] += 1.0;
                }
            }
            sp.symbolic
                .factorize_numeric_llt(
                    &mut sp.l_values,
                    sp.axx.as_ref(),
                    faer::Side::Lower,
                    LltRegularization::default(),
                    Par::Seq,
                    MemStack::new(&mut sp.fac_mem),
                    Spec::default(),
                )
                .ok()?;
            // `logdet_llt` is `log|A| = 2·Σ ln L_ii`; this method returns `Σ ln L_ii`.
            let ld = crate::sparse::logdet_llt(&sp.symbolic, &sp.l_values);
            sp.factor_ok = ld.is_finite();
            return sp.factor_ok.then_some(0.5 * ld);
        }
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
        if let Some(sp) = self.sp.as_deref_mut() {
            faer::sparse::linalg::cholesky::LltRef::new(&sp.symbolic, &sp.l_values)
                .solve_in_place_with_conj(
                    faer::Conj::No,
                    MatMut::from_column_major_slice_mut(&mut self.a_rhs[..k], k, 1),
                    Par::Seq,
                    MemStack::new(&mut sp.solve_mem),
                );
            return;
        }
        solve_in_place(
            self.a_chol.as_ref(),
            MatMut::from_column_major_slice_mut(&mut self.a_rhs[..k], k, 1),
            Par::Seq,
            MemStack::new(self.a_llt_mem),
        );
    }
}

/// Penalized-IRLS inner solve on the packed `M` rows by Newton's method on the
/// penalized likelihood (the step weight is the exact curvature, see
/// `pirls_solve_blocked`), with `M = ZΛ` the scaled RE design and a `+I` ridge
/// (the nAGQ=1 reparameterization, `u ~ N(0, I)`). Each step assembles
/// `A = M'WM + I` and the right-hand side `M'ρ − u`, with the score
/// `ρ = prior_w·μ'(y−μ)/V`, from the `width` nonzeros per row and takes
/// `u ← u + A⁻¹(M'ρ − u)` through a dense `k×k` (or sparse) Cholesky (the
/// increment form, see `pirls_solve_blocked`). Returns `(deviance, ‖ũ‖², log|L|` at the returned iterate,
/// converged`)`; a Cholesky failure surfaces as `(NaN, NaN, NaN, false)`.
/// A converged solve ends by re-evaluating η/μ/W at the returned `u` and
/// rebuilding and refactoring `A` there ([`evaluate_at_mode`]), so `log|L|`
/// (`log|A| = 2 log|L|`) and the deviance describe that iterate, and the
/// storage `se::packed_schur_fill` reads describes it too: on the dense
/// factor `packed.a` is left holding its raw symmetric `A` (re-factored
/// there); on the sparse factor `axx` holds that same `A` and `l_values` its
/// completed factor, which the Schur fill solves through directly.
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
/// trial whose raw η passes the link's clamp (`family::eta_infeasible`) is
/// rejected regardless of the band and pulled back, in one evaluation, to the
/// largest power-of-two fraction of the step that is inside the clamp
/// (`clamp_jump`), which counts as one halving. Convergence is the mixed
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
    // refilled at this θ before the call; on the dense factor `a` must survive
    // it holding the raw symmetric `M'WM + I`, so the Cholesky runs on the
    // separate `a_chol`.
    packed: &mut PackedScratch,
    // n×p scratch: the design residual rows `rᵢ = xᵢ − Tᵀmᵢ` of the Profile β
    // border (`residual_border`), rebuilt each Profile step, so no stale-value
    // hazard across PIRLS iterations.
    wx: &mut Mat<f64>,
    // Per-row linear-predictor offset (`Data::offset`), added into
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
        family,
        nb_theta,
        width,
        k,
        sp: packed.sparse.as_deref_mut(),
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
    // Period-2 damping state — see `PIRLS_OSC_RATIO`.
    let mut dmix_prev = f64::NAN;
    let mut osc_flips = 0usize;
    let mut damp = false;
    let mut dev = f64::NAN;
    let mut pen = f64::NAN; // ‖u‖² at the returned (post-step) iterate
    let mut logdet = 0.0;
    let tol = pirls_tol_override.unwrap_or_else(|| super::super::pirls_tol(family));
    for it in 0..PIRLS_MAX_ITERS {
        counters.set_pirls_iters(it + 1);
        // --- trial evaluation at the CURRENT u: (Mu)ᵢ and the raw η, then
        // μ/W/deviance through the family kernel the exit refresh
        // (`evaluate_at_mode`) also runs, so the band and the stop rule read
        // the deviance it returns. On a fresh accept this is the newly-stepped
        // u; after a halving `continue` it is the backtracked u. `infeasible`
        // flags any RAW η past the link's clamp bounds
        // (`family::eta_infeasible`) or NaN on a link that clamps η. ---
        let yeta = layout.eta_from_mode(&u[..], &eta_fixed[..], &mut eta[..], y, n);
        let (d, infeasible) = f64::family_pass(
            family,
            nb_theta,
            &mut eta[..n],
            &y[..n],
            &prior_w[..n],
            weighted,
            yeta,
            &mut prob[..n],
            &mut w[..n],
            &mut [],
        );
        dev = d;
        // Band-tolerant retrospective step-halving: the convergence band is
        // consulted before any halving, because near the optimum the iteration
        // is not strictly monotone — a step can land ε above `pen_accepted` yet
        // inside the tol band, and that must converge rather than burn the
        // halving budget (`PIRLS_MAX_HALVINGS`) against FP noise. Only a rise EXCEEDING the band is a genuine
        // overshoot. A domain-infeasible trial is a step failure regardless of
        // the band: accepting it would let `clamp_eta`'s boundary projection
        // define the converged answer.
        let pen_u: f64 = u[..k].iter().map(|v| v * v).sum();
        let penalized = dev + pen_u;
        if infeasible
            || !penalized.is_finite()
            || penalized - pen_accepted > tol * (1.0 + penalized.abs())
        {
            if halvings < PIRLS_MAX_HALVINGS {
                halvings += 1;
                if infeasible {
                    // Past the η clamp: one jump back inside it.
                    jump_back(
                        &mut layout,
                        family,
                        y,
                        &FixedPart {
                            x,
                            rows: None,
                            n,
                            p,
                            offset,
                        },
                        &beta_step,
                        u,
                        u_prev,
                        beta,
                        eta_fixed,
                        prob,
                        eta,
                        w,
                        k,
                    );
                    continue;
                }
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
        // fresh full step from it (cold start: pen_accepted = ∞ ⇒ always
        // accepts).
        halvings = 0;
        pen_accepted = penalized;
        u_prev[..k].copy_from_slice(&u[..k]);
        if let BetaStep::Profile { beta_prev, .. } = &mut beta_step {
            beta_prev[..p].copy_from_slice(&beta[..p]);
        }
        // Newton step on a link whose exact curvature differs from Fisher —
        // reasoning in `pirls_solve_blocked`, change together.
        observed_weights_in_place(
            family,
            nb_theta,
            y,
            prior_w,
            &eta[..],
            &prob[..],
            &mut w[..],
            n,
        );
        layout.scatter(&w[..], &prob[..], &eta[..], y, prior_w, weighted, n, None);
        // The increment right-hand side M'ρ − u (see `pirls_solve_blocked`,
        // change together).
        for (r, &uc) in layout.a_rhs[..k].iter_mut().zip(&u[..k]) {
            *r -= uc;
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
            *uc += a;
            pen += *uc * *uc;
        }
        // --- Profile-mode joint δβ step (β-Schur border), taken while the LLT of
        // A is still alive. `u` currently holds u_new = u_prev + δu₀. ---
        if let BetaStep::Profile {
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
            // T = A⁻¹B = A⁻¹(M'WX): transpose-gather B' then solve with this
            // iteration's factor.
            for r in 0..k {
                for c in 0..p {
                    ainv_mtwx[(r, c)] = xtwm[(c, r)];
                }
            }
            if let Some(sp) = layout.sp.as_deref_mut() {
                faer::sparse::linalg::cholesky::LltRef::new(&sp.symbolic, &sp.l_values)
                    .solve_in_place_with_conj(
                        faer::Conj::No,
                        ainv_mtwx.as_mut(),
                        Par::Seq,
                        MemStack::new(&mut sp.solve_mem),
                    );
            } else {
                solve_in_place(
                    layout.a_chol.as_ref(),
                    ainv_mtwx.as_mut(),
                    Par::Seq,
                    MemStack::new(layout.a_llt_mem),
                );
            }
            // Residual rows rᵢ = xᵢ − Tᵀmᵢ into `wx` over the row's packed
            // nonzeros, then S_β and the right-hand side X'ρ − B'·δu₀ off them
            // (`residual_border`).
            for i in 0..n {
                let base = i * width;
                for c in 0..p {
                    let mut r = x[(i, c)];
                    for t in base..base + width {
                        r -= layout.m_vals[t] * ainv_mtwx[(layout.m_cols[t] as usize, c)];
                    }
                    wx[(i, c)] = r;
                }
            }
            let scores = ProfileScores::new(family, nb_theta, y, prior_w, eta, prob);
            residual_border(
                |i| w[i],
                |i| scores.at(i),
                wx.as_ref().subrows(0, n),
                ainv_mtwx.as_ref().subrows(0, k),
                |j| u_prev[j],
                schur,
                &mut beta_rhs[..p],
            );
            // δβ = S_β⁻¹·rhs in place. Non-PD S_β ⇒ the (NaN,…,false) failure surface.
            if !border_solve(schur, &mut beta_rhs[..p], schur_llt_mem, None) {
                return (f64::NAN, f64::NAN, f64::NAN, false);
            }
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
        // Relaxed step once the period-2 detector has fired (`PIRLS_OSC_RATIO`):
        // move half way from the pre-step iterate, β in lockstep with u.
        if damp {
            for c in 0..k {
                u[c] = u_prev[c] + 0.5 * (u[c] - u_prev[c]);
            }
            if let BetaStep::Profile { beta_prev, .. } = &beta_step {
                for j in 0..p {
                    beta[j] = beta_prev[j] + 0.5 * (beta[j] - beta_prev[j]);
                }
                refresh_eta_fixed(x, beta, eta_fixed, n, p, offset);
            }
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
        // Period-2 detector — see `PIRLS_OSC_RATIO`.
        let dmix = mixed - mixed_prev;
        if dmix.is_finite()
            && dmix_prev.is_finite()
            && dmix * dmix_prev < 0.0
            && dmix.abs() > PIRLS_OSC_RATIO * dmix_prev.abs()
        {
            osc_flips += 1;
        } else {
            osc_flips = 0;
        }
        damp |= osc_flips >= PIRLS_OSC_TRIGGER;
        dmix_prev = dmix;
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
        // Nothing more on the sparse factor: `evaluate_at_mode` left the final
        // `A` in `axx` and its factor in `l_values`, and `packed_schur_fill`
        // solves through that factor; the dense `a` is `0×0` and stays so.
    }
    (dev, pen, logdet, converged)
}

/// Refill the packed `M` values at the current Λ (`packed.lam_small` must be
/// filled for this θ): each grouping's block of a row is [`super::zlam_row`],
/// the same `ZΛ` fold every other layout writes. The slopes come from `z_buf`
/// (`workspace::fill_z_f64`, whose row order this walk shares), never from the
/// fixed-effect design.
pub(crate) fn fill_m_vals(
    packed: &mut PackedScratch,
    g: &crate::lmm::LmmGroupings,
    z_buf: &[f64],
    n: usize,
) {
    let q_p = g.primary_q;
    let width = packed.width;
    let mut zt = 0;
    for i in 0..n {
        let mut t = i * width;
        super::zlam_row(
            &z_buf[zt..],
            &packed.lam_small,
            q_p,
            &mut packed.m_vals[t..t + q_p],
        );
        t += q_p;
        zt += q_p - 1;
        for e in 0..g.extra_offsets.len() {
            let q_g = g.extra_q[e];
            let lo = packed.lam_off_decl[e];
            super::zlam_row(
                &z_buf[zt..],
                &packed.lam_small[lo..],
                q_g,
                &mut packed.m_vals[t..t + q_g],
            );
            t += q_g;
            zt += q_g - 1;
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
/// indexed by the Λ ROW, as in the fill. Every other slot is zero.
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
    z_buf: &[f64],
    i: usize,
    out: &mut [f64],
) {
    out.fill(0.0);
    let q_p = g.primary_q;
    let base_theta = q_p * (q_p + 1) / 2;
    // Row `i`'s slopes in `z_buf`, and the running offset of a grouping's own.
    let mut zt = i * (q_p - 1 + g.extra_q.iter().map(|&q| q - 1).sum::<usize>());
    if a < base_theta {
        // Invert the column-major vech enumeration `lmm::primary_lambda`
        // writes (`c` outer, `r` inner from `r == c`).
        let mut t = 0;
        #[allow(clippy::needless_range_loop)]
        for c in 0..q_p {
            for r in c..q_p {
                if t == a {
                    out[c] = if r == 0 { 1.0 } else { z_buf[zt + r - 1] };
                    return;
                }
                t += 1;
            }
        }
        return;
    }
    let mut slot = q_p;
    let mut vech = base_theta;
    zt += q_p - 1;
    for &q_g in &g.extra_q {
        let len = q_g * (q_g + 1) / 2;
        if a < vech + len {
            let local = a - vech;
            let mut t = 0;
            for c in 0..q_g {
                for r in c..q_g {
                    if t == local {
                        out[slot + c] = if r == 0 { 1.0 } else { z_buf[zt + r - 1] };
                        return;
                    }
                    t += 1;
                }
            }
            return;
        }
        slot += q_g;
        vech += len;
        zt += q_g - 1;
    }
}

#[cfg(test)]
#[path = "packed_tests.rs"]
mod tests;
