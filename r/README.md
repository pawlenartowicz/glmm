# fastglmm

R bindings for the [`glmm`](https://github.com/pawlenartowicz/glmm) Rust
kernel: OLS, GLM (binomial, Poisson, Gamma, negative binomial), REML linear
mixed models, and binomial/Poisson GLMMs with Laplace or adaptive
Gauss-Hermite quadrature. lme4-style formulas, estimates validated against
lme4 and MixedModels.jl.

Deliberately scoped to **fast fitting**: fixed effects, Wald standard errors,
and variance components on the SD/correlation scale. Anything the engine
cannot compute honestly today (`ranef`, `predict`, `fitted`, `residuals`,
`logLik`/`AIC`, profiling) is an error naming the reason — never a silently
different answer.

## Installation

From r-universe (once the first `r-*` tag has built):

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
[`TUTORIAL-R.md`](../documentation/TUTORIAL-R.md).

```r
library(fastglmm)

fit <- fastglmm(y ~ t + d + t:d + (1 + t | g), data, family = binomial())

summary(fit)     # Wald z table + variance components; no fake AIC line
fixef(fit)       # fixed effects
vcov(fit)        # full Wald covariance
VarCorr(fit)     # SD/correlation-scale variance components, lme4-shaped
confint(fit)     # Wald intervals
isSingular(fit)  # boundary-fit flag, lme4's condition
```

`nAGQ = k` (odd, up to 25) turns on adaptive quadrature for binomial/Poisson
models with a single grouping factor and up to 3 random effects per group;
any other shape **warns and falls back to Laplace** (lme4 errors instead —
watch for the warning).

## What the formula accepts

The formula is parsed by the same Rust parser the Python port uses: bare
column names, `+`, `:`, `*`, `A/B` nesting, and `(1 + x | g)` random effects
with a full correlation structure. Not accepted (each is a clear error with a
workaround): `log(x)`/`I()`/`poly()` (compute the column first),
`cbind(s, f)` (pass the proportion as response and trials as `weights=`),
`- 1`/`0 +`, `(x || g)`, `offset()`, `.`, and `contrasts=` (relevel the
factor instead). R's `Gamma()` object means `link = "inverse"` (R semantics
win); the string `"gamma"` means the glmm default `link = "log"`.

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
