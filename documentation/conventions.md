# Conventions

The reporting conventions every `glmm` fit follows, regardless of which surface
(Rust, Python, R) called it: how estimation works, where standard errors come
from, how dispersion and variance components are scaled, how factors are
coded, and what the two boundary/convergence flags on the result mean.

## Estimation

Gaussian mixed models (LMM) are fit by **profiled REML only** — there is no ML
switch on this path. The fixed effects β and the residual scale σ² are
profiled out analytically; the optimizer searches only the relative covariance
parameter θ. See [`algorithms-lmm.md`](algorithms-lmm.md) for the objective
and the BOBYQA schedule that minimizes it.

Non-Gaussian mixed models (GLMM) are fit by **PIRLS + Laplace** by default
(`nAGQ = 1`, `glmer`-faithful). Adaptive Gauss–Hermite quadrature is opt-in via
`nagq`/`nAGQ` (an odd integer up to 25), and is honored on binomial and
Poisson GLMMs with a single grouping factor and up to 3 random effects per
group; on any other shape the Python and R surfaces warn and fall back to
Laplace. See
[`algorithms-glmm.md`](algorithms-glmm.md) for the PIRLS inner loop and the
AGQ gate.

Negative binomial fits carry an extra **outer loop for the shape parameter**
θ (not to be confused with the covariance θ above): the GLM path alternates a
fixed-θ inner fit with a profile-θ update, and the GLMM path maximizes the
marginal log-likelihood over `ln θ` by golden-section search around a full
inner GLMM fit at each candidate. Either way, θ̂ is reported as the fit's
`dispersion`.

## Standard errors

All reported standard errors are **Wald** — the square root of a diagonal of
an estimated covariance matrix, not a profile or bootstrap interval.

For OLS, GLM, and LMM, the covariance comes from the same Cholesky factor the
fit's normal equations already produce: `Var(β̂_j) = σ̂²·‖L_XX⁻¹e_j‖²`, one
method throughout.

For GLMM, two genuinely different Wald covariances are offered, selected by
`wald_se`/`WaldSe`:

- **Hessian** (the default, matching `glmer`'s `vcov(use.hessian = TRUE)`):
  the β-block of `2·H_dev⁻¹`, where `H_dev` is a finite-difference Hessian of
  the joint `(θ, β)` Laplace deviance at the converged point. If the joint
  Hessian is non-positive-definite, or a perturbed deviance is non-finite, it
  falls back to the Rx covariance below.
- **Rx** (conditional on θ̂): the expected-information Schur complement of the
  β block, inverted directly from the factors PIRLS already left behind. This
  is cheaper but assumes β–θ orthogonality — exact for the Gaussian LMM, but
  anticonservative for a GLMM, where the IRLS weights couple β and θ.

Coefficient tables report a Wald z-statistic and a two-sided `Pr(>|z|)`
p-value (the R and Python ports; see `TUTORIAL-R.md` §3).

SEs are computed only for the columns the caller asks for: in Rust,
`FitOptions::target_indices` selects which predictor columns get an SE (every
other slot is NaN). The formula frontend defaults to all columns, and the
Python and R surfaces go through it, so the distinction is invisible there.

## Dispersion

Which families carry an estimated dispersion parameter, and on what scale it
is reported, is a family-level convention documented in full in
[`supported_families.md`](supported_families.md). In short: Gamma estimates a
dispersion φ (a Pearson moment estimator, `φ̂ = Σrᵢ²/(n−p)`, unless the caller
pins it), negative binomial estimates a shape θ threaded through its own
outer loop, and dispersion is fixed at `φ ≡ 1` for Binomial, Poisson, and NB
(NB's overdispersion lives in θ, not φ).

## Variance components scale

Random-effect (co)variances are reported on lme4's **SD/correlation** scale —
standard deviations per random-effect dimension, plus a correlation matrix
between dimensions within the same grouping — not as raw variances. In R this
is what `VarCorr(fit)` prints, one block per grouping factor, with a
`Residual` row (`sigma()`) for a Gaussian mixed fit; the underlying attributes
(`attr(vc$group, "stddev")`, `attr(vc$group, "correlation")`) are available to
pull the numbers back out (see `TUTORIAL-R.md` §3).

## Factor coding

Factors are coded with treatment contrasts, base level = the column's first
level. See [`formula.md#factor-coding`](formula.md#factor-coding) for how the
base level is determined for string, factor, and categorical columns across
the three language surfaces.

## Flags on the result

`converged` reports whether the optimizer reached its convergence criterion.
`false` means the SE, covariance, and dispersion fields are NaN-filled rather
than trustworthy numbers — see the `Fit` doc comment for the exact per-field
fallback. An LMM fit that hits its evaluation cap is a partial exception: it
still reports its finite endpoint deviance, with `converged == false` marking
it as not fully converged.

`singular` (R's `isSingular(fit)`) reports a boundary fit — the same
condition lme4 flags: at least one diagonal random-effect variance component
collapsed to (or landed negligibly close to) zero. It is computed by the
kernel, not left to the caller to infer from the variance components
themselves.

If a fit comes back with `converged == false` or `singular == true`, see
[`troubleshooting.md`](troubleshooting.md) for what each flag implies and what
to try next.
