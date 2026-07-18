# Coming from lme4 (or statsmodels)

## Call mapping

| lme4 | fastglmm (R) | glmm (Python) |
|---|---|---|
| `lmer(f, d)` / `glmer(f, d, family)` | `fastglmm(f, d, family = ...)` | `glmm.fit(d, "f", family)` |
| `summary(fit)` | `summary(fit)` — Wald z table + variance components; no `logLik`/`AIC`/`BIC`/`deviance` line | `fit.summary()` — coefficient table (name, estimate, std. error, z, p), printed and returned as a string; a footer carries `dispersion` and `converged` |
| `fixef(fit)` | `fixef(fit)` — named fixed-effect estimates; aliased columns `NA` | `fit.beta` (estimates) with `fit.names` (aligned coefficient names); `fit.aliased` flags rank-deficient columns |
| `vcov(fit)` | `vcov(fit)` — full `p × p` Wald covariance | `fit.vcov` — `(p, p)` full Cov(β̂) |
| `VarCorr(fit)` | `VarCorr(fit)` — SD/correlation-scale variance components, lme4-shaped, with a `Residual` row for a Gaussian mixed fit | `fit.varcorr` (vech-packed per grouping) plus `fit.stddev_corr(group_idx)`, which splits a grouping's block into a stddev vector and a correlation matrix |
| `confint(fit)` | `confint(fit)` — Wald intervals off `vcov()`; `method = "profile"`/`"boot"` are not available and say so | Build from `fit.beta` and `fit.se` (or `fit.vcov` for joint contrasts) — no dedicated `confint` call |
| `isSingular(fit)` | `isSingular(fit)` — boundary-fit flag, lme4's condition, computed by the kernel | `fit.singular` |

## What is deliberately missing

`fastglmm` is deliberately scoped to **fast fitting**: fixed effects, Wald
standard errors, and variance components on the SD/correlation scale.
Anything the engine cannot compute honestly today — `ranef`, `predict`,
`fitted`, `residuals`, `logLik`/`AIC`, profiling — is an error naming the
reason, never a silently different answer. `coef()` is the clearest case: it
also errors, because lme4's `coef()` means fixed + random effects per group,
which needs `ranef()` — engine-blocked — so it points you at `fixef()`
instead of quietly dropping the random part.

This is the same discipline the kernel applies everywhere: it only surfaces
what it can compute honestly. See
[`validation.md`](validation.md) for how that claim is checked against the
lme4/MixedModels.jl oracles.

## Behavioral differences to watch

- `nAGQ = k` only turns on adaptive quadrature for a binomial/Poisson GLMM
  with a single grouping factor and up to 3 random effects per group; any
  other shape warns and falls back to Laplace instead of erroring the way
  lme4 does — watch for the warning (`r/README.md`).
- R's `Gamma()` family object means `link = "inverse"` (R semantics win when
  you pass the object), but the string `"gamma"` means the port's own
  default, `link = "log"` — see
  [`formula.md#language-notes`](formula.md#language-notes).
- There is no `cbind(successes, failures)` response: pass the proportion as
  the response column and the trial count as `weights=` instead — see
  [`formula.md`](formula.md#not-accepted-and-the-workaround).
- Factors are always coded with treatment contrasts (base = the column's
  first level); `contrasts=` is not an accepted argument — relevel the
  factor instead. See [`formula.md`](formula.md#not-accepted-and-the-workaround).
- REML is the only fit mode for LMMs (no ML switch), and all reported
  standard errors are Wald, not profile or bootstrap — see
  [`conventions.md`](conventions.md#flags-on-the-result) for how the
  `converged`/`singular` flags mark when those numbers should be trusted.

## statsmodels

The Python surface is exactly two names, `glmm.fit` and `glmm.Fit` — there is
no model object to construct, unlike `statsmodels`' construct-then-fit API
(`sm.MixedLM(...).fit()`). You hand `fit` a data table and a formula string,
and it parses, builds the design matrix, fits, and returns a `Fit`. Mixed
models beyond random intercepts — correlated random slopes, crossed and
nested grouping factors, GLMMs with Laplace or adaptive quadrature — are
first-class here, where `statsmodels.MixedLM` only covers Gaussian linear
mixed models.
