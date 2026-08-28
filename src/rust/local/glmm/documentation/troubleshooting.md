# Troubleshooting

## The fit is singular (isSingular is TRUE)

`singular` means a random-effect variance component landed at, or negligibly
close to, the boundary (zero) — the same condition lme4 flags with
`isSingular()`. See
[`conventions.md#flags-on-the-result`](conventions.md#flags-on-the-result)
for exactly how it's computed. It is not an error: the fit is a valid point
estimate, just one where the data can't support the random-effect structure
you asked for. Simplify the RE structure — drop a random slope, drop a
grouping factor — and refit.

## Warning: ill-conditioned / entangled columns

Two or more fixed-effect columns are near-collinear but not exactly
redundant — the design is computable and the fit is real. The estimates are
honest and the standard errors are large, because the data genuinely cannot
separate the entangled columns. This is not a failure: `converged` stays
`true`, and no column is dropped (contrast an *aliased* column, which is
exactly redundant and is dropped, with `beta`/`se` reported as `NaN`).

The warning names one column — the one the pivot search happened to reach —
but its entangled partners are **not** named; the search finds the
worst-conditioned column, not the full set it is confounded with. To fix it:
rescale predictors onto comparable magnitudes, or drop or combine the
predictors that are actually measuring the same thing. See
[`conventions.md#flags-on-the-result`](conventions.md#flags-on-the-result) for
where this sits in the `diagnostics` channel.

## Warning: random-effect design columns on very different scales

A grouping's random-effect design columns span very different magnitudes —
the ratio between the largest and smallest column RMS (the implicit
intercept counted as 1.0) exceeds 1000, the same threshold lme4's
`checkScaleX` warns on. Fitting itself is unaffected: the kernel scales the
columns internally before solving. What the warning is telling you is that
the reported random-effect standard deviation for that grouping sits on the
raw variable's own scale, so a large spread between components makes them
hard to compare by eye. Rescale the offending variable (e.g. divide a
day-of-experiment slope by 100, or convert a raw count to thousands) so the
reported stddevs land on comparable magnitudes.

## Warning: Hessian-based standard errors fell back to RX

`wald_se="hessian"` (Python) / `wald.se = "hessian"` (R) was requested, but
the finite-difference joint Hessian this GLMM's SE pass builds was not
usable — either it was not positive definite, or a perturbed deviance
evaluation it needed was non-finite. The fit falls back to the RX/Schur
standard errors instead: `se`/`vcov` are still filled (from the fallback),
but `stddev_se` (the standard errors of the random-effect standard
deviations) stays `NaN`, since only the joint-Hessian route can fill it. This
never fires under `wald_se="rx"`, which never attempts the joint Hessian in
the first place. If you need `stddev_se`, try simplifying the random-effect
structure or switching to `wald_se="rx"` to avoid the failed attempt
entirely.

## converged is false

`converged` reports whether the optimizer reached its convergence criterion.
`false` means `se`, `vcov`, and `dispersion` come back NaN-filled — see
[`conventions.md#flags-on-the-result`](conventions.md#flags-on-the-result)
for the exact per-field fallback (an LMM that hits its evaluation cap is a
partial exception: it still reports a finite endpoint `deviance`).

First things to check: the scale of your predictors (wildly different
magnitudes across columns make the optimizer's job harder), and whether the
model is too rich for the data — too many random-effect parameters for the
number of clusters/groups you have. There is no multistart or retry knob to
reach for here; `fit` runs one optimization from one starting point and
reports honestly if it didn't converge.

## NotImplementedError: family/link/knob

Two combinations have an approved design but no kernel support yet, and
raise a clean `NotImplementedError` rather than silently falling back to
something close:

- quasi-likelihood `dispersion=` on binomial/Poisson
- a float `init_theta=` seed (only the default `init_theta=None` cold start
  is supported)

`family="inversegaussian"` (fixed-effect GLM only — mixed models fault with a
message naming the deferral) and `link="cloglog"` (GLM and GLMM) are both
implemented.

If you hit one of the two remaining gaps, there's nothing to configure around
it today — wait for the knob to land, or restructure the model to avoid it
(e.g. drop the `dispersion=` request).

## Warning: falling back to Laplace

`nagq`/`nAGQ` only takes effect on one narrow shape: a binomial or Poisson
GLMM with a single grouping factor and up to 3 random effects per group,
with an odd node count up to 25. Any other shape — multiple grouping
factors, more REs per group, a non-binomial/Poisson family — warns and
silently falls back to Laplace (`nAGQ = 1`) instead of honoring the
requested node count.

lme4 errors on these shapes rather than falling back, so a script that
worked in lme4 with an ineligible `nAGQ` won't fail the same way here — it
will fit, but on Laplace, and the answer will differ slightly from what an
actual adaptive-quadrature run would give. Watch for the warning; if you see
it, either narrow the model to the eligible shape or treat the result as a
Laplace fit.

## My formula is rejected

The formula parser accepts bare column names, `- 1`/`0 +`, a small whitelist
of transforms (`log(x)`, `sqrt(x)`, `exp(x)`, `I(x^2)`), `offset()`, and
`cbind(s, f)` on the response — but not R's general function-call syntax:
`poly(x, 2)`, nested calls, arithmetic inside a call, `(x || g)`,
`(0 + x | g)`, `.`, and `contrasts=` are all clear parse-time errors, never a
silent reinterpretation. See
[`formula.md#not-accepted-and-the-workaround`](formula.md#not-accepted-and-the-workaround)
for the full list and the workaround for each.

## predict()/residuals()/coef() error

These aren't missing by oversight — the ports error, naming the reason, on
anything the kernel can't compute honestly rather than returning a
fixed-effects-only or otherwise silently different answer. `ranef()`,
`fitted()` and `logLik()`/`AIC()`/`BIC()` used to be on this list and now
work. See
[`coming-from-lme4.md#what-is-deliberately-missing`](coming-from-lme4.md#what-is-deliberately-missing)
for what's blocked and why.
