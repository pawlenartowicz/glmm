//! The accept / halve / careful / trust-region state machine of the two
//! blocked PIRLS solves, `pirls_solve_blocked` and `pirls_solve_blocked_extras`.
//! [`PirlsControl`] holds every scalar of that state and makes each decision;
//! the solves keep their layout work (scatter, factor, observed twin, the exact
//! passes, the border rows) and apply each decision to their buffers through the
//! helpers in `pirls/mod.rs`. Every decision reads `.value()`s, so the state is
//! `f64` on every instantiation.

use super::*;

/// What the solve does with a trial after [`PirlsControl::trial`].
pub(super) enum Trial {
    /// Not rejected: take the step from it.
    Proceed,
    /// A careful-mode u-only trial past the η clamp: one jump back inside it,
    /// toward `u_prev` at the current β ([`jump_back`] with `BetaStep::Fixed`).
    RefineJump,
    /// A careful-mode u-only trial rose: halve toward `u_prev` ([`halve_u`]).
    RefineHalve,
    /// An overshooting trial past the η clamp: one jump back inside it
    /// ([`jump_back`]).
    Jump,
    /// An overshooting trial: halve back toward the halving target.
    Halve,
    /// Halvings exhausted: the solve fails ([`fail`]).
    Fail,
}

/// What the solve does with an exact-mode trial after [`PirlsControl::merit`].
pub(super) enum Merit {
    /// Re-step u alone at the trial's β (careful mode).
    Refine,
    /// Halve back toward the last accepted `(u, β)` ([`halve_prev_to_acc`],
    /// [`halve_beta`]).
    Reject,
    /// Accepted: store the trial as the last accepted point.
    Accept,
    /// Halvings exhausted: the solve fails ([`fail`]).
    Fail,
}

pub(super) struct PirlsControl {
    tol: f64,
    /// Exact-Laplace β-profile (`BetaStep::Profile { exact: Some(_) }`).
    pub(super) exact: bool,
    /// Where the exact curvature differs from Fisher, the objective's `log|A|`
    /// is `log|A_obs|` (`evaluate_at_mode`), and the Newton step already builds
    /// `A` from `W_obs`: the factor the solve takes is `A_obs`'s. The exact
    /// profile is `f64`-only (`dual` is `None`), so this holds on every
    /// exact-mode solve. It picks `dW_obs/dη` over the Fisher `dw/dη`, in pass A
    /// and in the border's curvature.
    pub(super) exact_obj: bool,
    min_iters: usize,
    /// A `DualStep` is present.
    dual: bool,
    /// Same-point penalized deviance at the last ACCEPTED iterate.
    pen_accepted: f64,
    /// The mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` from the previous step.
    mixed_prev: f64,
    halvings: usize,
    // Period-2 damping state — see `PIRLS_OSC_RATIO`.
    dmix_prev: f64,
    osc_flips: usize,
    /// The period-2 detector has fired: every later step moves half way from
    /// the pre-step iterate ([`damp_step`]).
    pub(super) damp: bool,
    // Exact-Laplace β-profile: the accept/halve decision moves from the
    // penalized-deviance band ([`PirlsControl::trial`]) to the FULL Laplace
    // merit `dev + pen_u + 2·log|A|` after the factor, since log|A| depends on
    // this iteration's A and is not known until the factor is formed. `l_acc`
    // is that merit at the last ACCEPTED (u, β) — cold start accepts
    // unconditionally.
    l_acc: f64,
    // `|g_u'·δu₀|` at the point `l_acc` was measured at — how far that stored
    // merit may itself be off. One half of the accept band in
    // [`PirlsControl::merit`]; see it for why both endpoints of the comparison
    // are charged. Unread while `l_acc = ∞` (the first trial accepts
    // unconditionally).
    l_acc_slack: f64,
    // Careful-mode state (see [`PirlsControl::merit`]). `acc_judged` says
    // whether the stored `l_acc` was itself accepted by a comparison (the
    // first trial is accepted unjudged); `refine` marks an iteration that
    // re-steps u alone at the trial's β; `l_ref` is the merit at the previous
    // iterate of that β and `pen_ref` its `dev + ‖u‖²`, the merit that u-only
    // step backtracks on. `trust` is the border step's trust region.
    careful: bool,
    acc_judged: bool,
    refine: bool,
    l_ref: f64,
    pen_ref: f64,
    ref_halvings: usize,
    pub(super) trust: BorderTrust,
}

impl PirlsControl {
    pub(super) fn new<T: Scalar>(
        family: Family,
        beta_step: &BetaStep,
        dual: Option<&mut DualStep<T>>,
        pirls_tol_override: Option<f64>,
    ) -> Self {
        let has_dual = dual.is_some();
        let min_iters = match dual {
            Some(d) => {
                // Optimistic seed: paired with `observed = !canonical` the step
                // the kernel then takes IS the Hessian step, so the caller may
                // trust the lanes after one call regardless of which link this
                // is, unless a non-PD observed factor further down takes it
                // back (per block in `pirls_solve_blocked`, the whole iteration
                // in `pirls_solve_blocked_extras`; see `DualStep::exact`).
                d.exact = true;
                d.min_iters
            }
            None => 0,
        };
        let exact = matches!(beta_step, BetaStep::Profile { exact: Some(_), .. });
        PirlsControl {
            tol: pirls_tol_override.unwrap_or_else(|| super::super::pirls_tol(family)),
            exact,
            exact_obj: exact && crate::family::exact_curvature_differs(family),
            min_iters,
            dual: has_dual,
            pen_accepted: f64::INFINITY,
            mixed_prev: f64::INFINITY,
            halvings: 0,
            dmix_prev: f64::NAN,
            osc_flips: 0,
            damp: false,
            l_acc: f64::INFINITY,
            l_acc_slack: 0.0,
            careful: false,
            acc_judged: false,
            refine: false,
            l_ref: f64::INFINITY,
            pen_ref: f64::INFINITY,
            ref_halvings: 0,
            trust: BorderTrust::new(),
        }
    }

    /// Judge the trial at the top of the loop, on its same-point penalized
    /// deviance `dev + ‖u‖²` (`penalized`) and the family pass's `infeasible`
    /// flag (any raw η past the link's clamp bounds, `family::eta_infeasible`).
    /// On `Proceed` outside exact mode the trial is the new accepted point.
    pub(super) fn trial(&mut self, penalized: f64, infeasible: bool) -> Trial {
        let tol = self.tol;
        // A careful-mode u-only step (β fixed) is judged the way lme4's
        // `pwrssUpdate` judges a u-only PIRLS step: on `dev + ‖u‖²` against its
        // origin, halving toward that origin with β untouched, never toward the
        // accepted (u, β) the β line search backtracks to.
        if self.refine
            && (infeasible
                || !penalized.is_finite()
                || penalized - self.pen_ref > tol * (1.0 + penalized.abs()))
        {
            if self.ref_halvings < PIRLS_MAX_HALVINGS {
                self.ref_halvings += 1;
                if infeasible {
                    return Trial::RefineJump;
                }
                return Trial::RefineHalve;
            }
            return Trial::Fail;
        }
        // Retrospective step-halving (lme4 `pwrssUpdate`, mirrors
        // `pirls_solve_packed`): convergence band checked BEFORE the overshoot
        // test (near the optimum the iteration is not strictly monotone — a step
        // can land ε above `pen_accepted` yet inside the tol band, and that must
        // converge, not burn the halving budget (`PIRLS_MAX_HALVINGS`) against
        // FP noise).
        // BAND-TOLERANT overshoot test, mirrors `pirls_solve_packed` (see its
        // comments for why a within-band rise is accepted rather than
        // converged-on or halved): only a rise EXCEEDING the tol band
        // backtracks. A domain-infeasible trial halves regardless of the band
        // (see `pirls_solve_packed`'s comment).
        // Exact mode: the retrospective band is on `dev + pen_u` alone, which
        // is not the profiled objective (missing 2·log|A|) — only a
        // domain-infeasible trial halves here; an overshoot on `dev + pen_u` is
        // judged later, after log|A| is known, against the FULL merit
        // ([`PirlsControl::merit`]).
        if infeasible
            || !penalized.is_finite()
            || (!self.exact && penalized - self.pen_accepted > tol * (1.0 + penalized.abs()))
        {
            self.trust.fail(self.acc_judged);
            if self.halvings < PIRLS_MAX_HALVINGS {
                self.halvings += 1;
                if infeasible {
                    return Trial::Jump;
                }
                return Trial::Halve;
            }
            return Trial::Fail;
        }
        // Accept this iterate (cold start: pen_accepted = ∞ ⇒ always accepts).
        // Exact mode defers `halvings`/`pen_accepted`/`beta_prev` bookkeeping
        // to the merit test ([`PirlsControl::merit`]), since this trial has not
        // been judged against the full Laplace objective yet.
        if !self.exact {
            self.halvings = 0;
            self.pen_accepted = penalized;
        }
        Trial::Proceed
    }

    /// The factor on the trial `u` failed (not positive definite). True: halve
    /// toward the last accepted `(u, β)` ([`halve_prev_to_acc`],
    /// [`halve_beta`]) and re-enter the top; false: the solve fails.
    ///
    /// Exact mode has no retrospective `dev + pen_u` band (the merit test owns
    /// the accept decision), so nothing stands between a wildly overshooting
    /// joint (u, β) step and the factor. On grouseticks the cold-start Newton
    /// step lands at dev ≈ 3e26, where the crossed-tail Schur loses
    /// positive-definiteness to cancellation and the solve would end there —
    /// while the same step under the PQL band is simply halved away. An
    /// unevaluable trial is a REJECTED trial, not a failed solve: backtrack
    /// toward the last accepted (u, β) exactly as the merit rejection does, and
    /// surface the failure only once the halvings are spent.
    pub(super) fn factor_failed(&mut self) -> bool {
        if self.exact {
            self.trust.fail(self.acc_judged);
            if self.halvings < PIRLS_MAX_HALVINGS {
                self.halvings += 1;
                self.refine = false;
                self.l_ref = f64::INFINITY;
                return true;
            }
        }
        false
    }

    /// Exact mode: judge the trial on its Laplace merit `l_trial` (the solve's
    /// `dev + ‖u‖² + 2·log|A| + g_u'·δu₀`), with `gu_dot_du` its
    /// mode-consistency correction `g_u'·δu₀` and `penalized` its
    /// `dev + ‖u‖²`.
    pub(super) fn merit(&mut self, l_trial: f64, gu_dot_du: f64, penalized: f64) -> Merit {
        // Being first-order, the mode-consistency correction leaves its own
        // O(‖δu₀‖²) residual behind, either sign, so a merit is only good to
        // about the size of the correction it carries. The comparison has TWO
        // endpoints and each carries its own — hence both are charged:
        // `gu_dot_du.abs()` for this trial, `l_acc_slack` for the stored
        // `l_acc`. Charging the trial alone would only accidentally cover
        // the failure, since the deficit that deadlocks the loop belongs to
        // the ACCEPTED point: a warm (u, β) is accepted unjudged as the first
        // trial, and if its merit sits under the value the solve can reach,
        // every later (correct) iterate is rejected until the iteration cap
        // turns the whole evaluation into a non-convergence. Measured on the
        // single-intercept NB-log shape: a warm start carried from θ = 0.55674
        // to θ = 0.55692 left `l_acc` 3.2e-5 under the value the cold solve
        // reaches, every iteration went to halving, and the θ-only outer
        // search read the `inf` that came back as a wall — with the seed's own
        // 1.4e-4 correction charged, that deficit is inside the band. Both
        // terms shrink with their δu₀, so at the mode the test is the value
        // band again, and the convergence test ([`PirlsControl::stop`]) is
        // untouched.
        // A non-finite merit (an overflowed step) is an overshoot: every
        // comparison with NaN is false, so without this test it would be
        // accepted and poison `l_acc` for the rest of the solve.
        // Careful mode. The slack above trusts `|g_u'·δu₀|` to size a merit's
        // error, which holds only while δu₀ is small. Far from the mode it is
        // not: at a large θ with few clusters the Laplace profile is nearly
        // flat in the β direction the random effects nearly span, the border
        // step can move β by several units, and u lands a long Newton step off
        // its mode. The slack then grows with that step and lets real rises
        // through, while a first-order
        // merit can also sit far below the value its β reaches once u has
        // converged (measured on a 4-cluster NB-log fit at θ = 50: 2.0 under
        // it, with a correction of 0.07), so a stored `l_acc` can refuse every
        // later trial. So the first time the slack alone lets through a rise
        // over a stored merit that was itself judged (a rise over the unjudged
        // first trial is the warm-start case the slack was written for) and the
        // trial's own correction is past the band, the solve stops comparing
        // first-order merits for good: it drops `l_acc`, and judges each later
        // trial only once u-only steps at that trial's β (lme4's `pwrssUpdate`
        // with `uOnly`, backtracking on `dev + ‖u‖²` in
        // [`PirlsControl::trial`]) have made its merit
        // stationary, against a stored merit that was converged the same way,
        // on the plain tol band. lme4's `pwrssUpdate` and MixedModels.jl's
        // `pirls!` accept a step only on a real decrease of the objective,
        // which they can evaluate exactly at every iterate; here it is exact
        // only where u sits at its mode, and the u-only steps are what buy
        // that.
        let band = self.tol * (1.0 + l_trial.abs());
        if self.careful && (l_trial - self.l_ref).abs() > band {
            self.refine = true;
            self.l_ref = l_trial;
            self.pen_ref = penalized;
            self.ref_halvings = 0;
            return Merit::Refine;
        }
        let allow = if self.careful {
            band
        } else {
            band + gu_dot_du.abs() + self.l_acc_slack
        };
        let reject = !l_trial.is_finite() || l_trial - self.l_acc > allow;
        if !self.careful
            && self.acc_judged
            && !reject
            && l_trial - self.l_acc > band
            && gu_dot_du.abs() > band
        {
            self.careful = true;
            self.l_acc = f64::INFINITY;
            self.refine = true;
            self.l_ref = l_trial;
            self.pen_ref = penalized;
            self.ref_halvings = 0;
            return Merit::Refine;
        }
        self.refine = false;
        self.l_ref = f64::INFINITY;
        // The trust region reads the merit change the border step's trial
        // made against its model (`BorderTrust`), under the same allowance
        // for the merits' own error as the accept test.
        self.trust
            .judge(self.l_acc, self.acc_judged, l_trial, allow);
        if reject {
            if self.halvings < PIRLS_MAX_HALVINGS {
                self.halvings += 1;
                return Merit::Reject;
            }
            return Merit::Fail;
        }
        self.halvings = 0;
        self.acc_judged = self.l_acc.is_finite();
        self.l_acc = l_trial;
        self.l_acc_slack = gu_dot_du.abs();
        Merit::Accept
    }

    /// Whether this border step computes `logdet_beta_curvature`
    /// ([`BorderTrust::wants_curvature`]).
    pub(super) fn wants_curvature(&mut self, it: usize, memory: &mut bool) -> bool {
        self.trust
            .wants_curvature(it, self.careful, self.damp, memory)
    }

    /// The stopping rule after step `it`, on `dev_pen = (dev + ‖u‖²).value()`
    /// and `logdet = log|A|.value()`; true = converged. Otherwise feeds the
    /// period-2 detector.
    ///
    /// The stopping rule: the mixed `dev(uⱼ) + ‖uⱼ₊₁‖²` band on successive
    /// steps. The same-point band cannot be a converge trigger: on a solve that
    /// cycles it stops at one point of the cycle rather than at the mode, and
    /// it breaks the AGQ(k=1) ≡ Laplace reduction.
    /// Exact mode: the exit band must track the same merit the accept/halve
    /// decision uses (dev + pen + 2·log|A|), or the loop could settle
    /// on a point that is a fixed point of dev+pen alone but still moving in
    /// log|A|. The merit's
    /// mode-consistency term is deliberately absent here: it is proportional to
    /// δu₀, which the band already forces to zero, so including it would change
    /// no fixed point and only the iterate count.
    pub(super) fn stop(&mut self, it: usize, dev_pen: f64, logdet: f64) -> bool {
        let mixed = dev_pen + if self.exact { 2.0 * logdet } else { 0.0 };
        if it + 1 >= self.min_iters
            && (mixed - self.mixed_prev).abs() < self.tol * (1.0 + mixed.abs())
        {
            return true;
        }
        // Period-2 detector — see `PIRLS_OSC_RATIO`. Off on a dual
        // solve: a halved step is not the one-step Hessian step `DualStep::exact`
        // promises.
        let dmix = mixed - self.mixed_prev;
        if !self.dual
            && dmix.is_finite()
            && self.dmix_prev.is_finite()
            && dmix * self.dmix_prev < 0.0
            && dmix.abs() > PIRLS_OSC_RATIO * self.dmix_prev.abs()
        {
            self.osc_flips += 1;
        } else {
            self.osc_flips = 0;
        }
        self.damp |= self.osc_flips >= PIRLS_OSC_TRIGGER;
        self.dmix_prev = dmix;
        self.mixed_prev = mixed;
        false
    }
}
