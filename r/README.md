# fastglmm

**`fastglmm`: lme4-style formulas on the `glmm` engine.**

**glmm** fits mixed models in Rust, Python and R, and in the browser through WASM. One `fit`
covers ordinary least squares, generalized linear models, linear mixed models and
generalized linear mixed models.

- **Accuracy first.** Gives the same or a more accurate result than lme4, glmmTMB,
  MixedModels.jl and GLMMadaptive, and converges on every model that has a solution.
- **Multi-solver.** Uses a multi-solver paradigm to get the best result more robustly:
  closed form for least squares, IRLS for GLMs, and BFGS or BOBYQA
  for mixed models, all written from scratch in Rust.
- **Fast.** Faster than every other engine on every model, usually 10 to 100 times,
  without RAM blowup.

On all tested models: a 923-model validation grid, checked for every release.

Deliberately scoped to **fast fitting**: fixed effects, Wald standard errors,
variance components on the SD/correlation scale, conditional modes, fitted
values, log-likelihood, and tests of fixed terms and of two nested models
(`anova()`, `compare()`; see
[`hypothesis-tests.md`](../documentation/hypothesis-tests.md)). Anything the
engine cannot compute honestly (`predict`, `coef`, profiling) is an error
naming the reason, never a silently different answer.

## Installation

From r-universe:

```r
install.packages("fastglmm", repos = c("https://pawlenartowicz.r-universe.dev", getOption("repos")))
```

From a checkout (needs Rust — `cargo` and `rustc >= 1.85` on the PATH):

```r
# in GLMM/r/
install.packages(".", repos = NULL, type = "source")
```

## Usage

A full four-section walkthrough — formula in / fit out, families and knobs,
reading the result, warm starts — is in
[`tutorial-r.md`](../documentation/tutorial-r.md).

```r
library(fastglmm)

# OLS: fixed effects only
fit <- fastglmm(y ~ x1 * x2, data)

# LMM: correlated random intercept and slope (REML)
fit <- fastglmm(y ~ x1 + (1 + x1 | subject), data)

# The same LMM by maximum likelihood, as lmer(REML = FALSE)
fit_ml <- fastglmm(y ~ x1 + (1 + x1 | subject), data, REML = FALSE)

# GLMM: binomial with a probit link, 7-node adaptive quadrature
fit <- fastglmm(passed ~ x1 + (1 | school), data,
                family = binomial(link = "probit"), nAGQ = 7)

summary(fit)     # coefficient table (lmerTest's Satterthwaite t on an LMM) + variance components
fixef(fit)       # fixed effects
vcov(fit)        # full Wald covariance
VarCorr(fit)     # SD/correlation-scale variance components, lme4-shaped
confint(fit)     # Wald intervals on the summary's t or z quantiles
isSingular(fit)  # boundary-fit flag, lme4's condition
anova(fit)       # tests of the fixed terms (Type III)
compare(data, y ~ x1, y ~ x1 + (1 | subject))  # two nested models, fitted and compared
```

`nAGQ = k` (odd, up to 25) turns on adaptive quadrature for binomial, Poisson,
negative-binomial and Gamma models with a single grouping factor and up to 3
random effects per group; any other shape **warns and falls back to Laplace**
(lme4 errors instead — watch for the warning).

## Development

The Rust surface lives in the repo's `glmm-r` crate; `src/rust` is a thin
staticlib re-export the R build drives. Path dependencies resolve only inside
the repo checkout — a distributable, self-contained source tarball (with the
in-repo crates materialized and, optionally, all cargo dependencies vendored
for offline/CRAN builds) is produced by:

```sh
tools/cran-tarball.sh [--vendor]
```

`R CMD check --as-cran` is clean on that tarball. Tests: `testthat`, with an
acceptance suite comparing fixed effects and variance components against
`lme4::glmer` on the benchmark shapes (skipped when lme4 is absent).

## Citation

If you use fastglmm in research, please cite:

Lenartowicz, P. (2026). glmm: Mixed-model fitting in Rust, Python and R [Computer software]. https://github.com/pawlenartowicz/glmm

```bibtex
@software{glmm2026,
  author = {Lenartowicz, Pawe{\l}},
  title  = {{glmm}: Mixed-Model Fitting in Rust, Python and R},
  year   = {2026},
  url    = {https://github.com/pawlenartowicz/glmm}
}
```
