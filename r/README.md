# fastglmm

**`fastglmm`: lme4-style formulas on the `glmm` engine.**

**fastglmm** fits ordinary least squares, generalized linear models, linear
mixed models and generalized linear mixed models from an lme4-style formula. It
is the R package of the [glmm engine](https://github.com/pawlenartowicz/glmm),
which also serves Rust and Python. It is called fastglmm because CRAN already
has a different package named `glmm`.

## The idea

- **Accuracy first.** The glmm engine reaches the same log-likelihood as lme4,
  glmmTMB, MixedModels.jl and GLMMadaptive, or a better one.
- **Multi-solver.** Each model gets a method that suits it. Least squares uses
  a closed form, and GLMs use IRLS. Mixed models use BFGS where the engine has
  a cheap exact gradient, and BOBYQA, which needs no gradient, elsewhere.
- **Fast.** The engine is faster than these four packages on every model in the
  speed tests, and it uses little memory.

More in the [project README](https://github.com/pawlenartowicz/glmm#readme).

## Validation

The glmm engine is checked on 923 test models before every release. They cover
the Gaussian, binomial, Poisson, Gamma and negative binomial families, random
intercepts, correlated slopes, nested and crossed groupings, and data from 60 to
30,000 rows. Each model is also fitted by lme4, glmmTMB, MixedModels.jl and
GLMMadaptive, where they support it. The engine is compared with the best of
their results, not with one chosen package. If it reaches a worse
log-likelihood than that best result, the release fails. If its estimates
differ from the other packages, the difference passes only once it is
explained in writing.

Standard errors of mixed models come from the exact second derivatives of the
likelihood, not from a finite-difference approximation. On the test models they
differ from a high-precision lme4 run by at most 0.002% (relative).

The engine was also stress-tested on many hard and borderline models. On such
models it returns a result with flags that say what happened, instead of only
printing a warning. A variance that falls to zero is set to exactly zero and
flagged. A redundant column is dropped and marked, as in lme4. When a fit
fails, the `converged` flag says so. The engine does not crash.

The package versions, the checks and how to rerun them are in
[`validation.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/validation.md).

## Multi-solver paradigm

The glmm engine combines several optimizers for mixed models. Where it has a
cheap exact gradient, it uses BFGS. Elsewhere it uses BOBYQA, which needs no
gradient. BOBYQA also refines the BFGS result and takes over when BFGS fails.

On larger GLMMs, a fast approximate pass first finds a good starting point, and
the full search starts from there. When the main search does not converge, the
fit tries a second search strategy from the same start and keeps the better
answer.

The linear algebra also uses more than one method. Common model shapes use
small dense blocks, one per group. Larger or more complex designs go to a
sparse solver instead of being refused. Given the same variance parameters,
both return the same answer. The choice affects speed only.

## Speed

The glmm engine moves as little data as it can. One pass over the data builds
a small summary per group, and every later step of the search works on those
summaries. The cost of each step therefore grows with the number of groups, not
the number of rows.

The engine is also designed for refitting the same model many times. Its
working memory depends only on the shape of the model. So refitting the same
model on new data, as in a power simulation or a bootstrap, needs no new
memory. Each refit can also start from the previous answer.

The search also needs few steps. Its settings were tuned on the 923 test models
and kept only where they cut the number of steps without changing any result.
With a cheap exact gradient it needs even fewer.

Each of these choices, and how it differs from lme4 and MixedModels.jl, is in
[`glmm-design.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/glmm-design.md).

## Install

From r-universe:

```r
install.packages("fastglmm", repos = c("https://pawlenartowicz.r-universe.dev", getOption("repos")))
```

Building from source, for example from a clone of the repository, needs Rust:
`cargo` and `rustc >= 1.85` on the `PATH`.

```r
# in the r/ folder of a clone of github.com/pawlenartowicz/glmm
install.packages(".", repos = NULL, type = "source")
```

## Quick start

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

summary(fit)     # coefficients + variance components; Satterthwaite t on an LMM
fixef(fit)       # fixed effects
vcov(fit)        # full Wald covariance
VarCorr(fit)     # SD/correlation-scale variance components, lme4-shaped
confint(fit)     # Wald intervals on the summary's t or z quantiles
isSingular(fit)  # boundary-fit flag, lme4's condition
anova(fit)       # tests of the fixed terms (Type III)
compare(data, y ~ x1, y ~ x1 + (1 | subject))  # two nested models, fitted and compared
```

`nAGQ = k` (odd, up to 25) turns on adaptive quadrature for binomial, Poisson,
negative binomial and Gamma models with one grouping factor and up to 3 random
effects per group. For any other model, fastglmm **warns and uses Laplace
instead**. lme4 stops with an error here, so watch for the warning.

## Documentation

- [R tutorial](https://github.com/pawlenartowicz/glmm/blob/main/documentation/tutorial-r.md):
  formula in, fit out, every option, reading the result, and warm starts.
- [Worked examples](https://github.com/pawlenartowicz/glmm/blob/main/documentation/examples-r.md)
  on real datasets.
- [Coming from lme4](https://github.com/pawlenartowicz/glmm/blob/main/documentation/coming-from-lme4.md).
- [What the formula accepts](https://github.com/pawlenartowicz/glmm/blob/main/documentation/formula.md).
- [Tests of fixed terms and model comparisons](https://github.com/pawlenartowicz/glmm/blob/main/documentation/hypothesis-tests.md).
- [Fixes for common errors](https://github.com/pawlenartowicz/glmm/blob/main/documentation/troubleshooting.md).
- [All documentation](https://github.com/pawlenartowicz/glmm/blob/main/documentation/index.md).

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

## License

`LGPL-3.0-or-later`.

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
