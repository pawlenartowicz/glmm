# glmm

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/license-LGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

**`glmm`: mixed models from an R-style formula, with NumPy as the only dependency.**

**glmm** fits ordinary least squares, generalized linear models, linear mixed
models and generalized linear mixed models from an R-style formula. It is the
Python package of the [glmm engine](https://github.com/pawlenartowicz/glmm),
which also serves Rust and R.

## The idea

- **Accuracy first.** glmm reaches the same log-likelihood as lme4, glmmTMB,
  MixedModels.jl and GLMMadaptive, or a better one.
- **Multi-solver.** Each model gets a method that suits it. Least squares uses
  a closed form, and GLMs use IRLS. Mixed models use BFGS where glmm has a cheap
  exact gradient, and BOBYQA, which needs no gradient, elsewhere.
- **Fast.** glmm is faster than these four packages on every model in the speed
  tests, and it uses little memory.

More in the [project README](https://github.com/pawlenartowicz/glmm#readme).

## Install

```bash
pip install glmm
```

Requires Python 3.10+. The only runtime dependency is NumPy. glmm also runs in
the browser under Pyodide: a Pyodide wheel is attached to each
[GitHub release](https://github.com/pawlenartowicz/glmm/releases).

## Validation

glmm is checked on 947 test models before every release. They cover the
Gaussian, binomial, Poisson, Gamma and negative binomial families, random
intercepts, correlated slopes, nested and crossed groupings, and data from 60 to
30,000 rows. Each model is also fitted by lme4, glmmTMB, MixedModels.jl and
GLMMadaptive, where they support it. glmm is compared with the best of their
results, not with one chosen package. If glmm reaches a worse log-likelihood
than that best result, the release fails. If its estimates differ from the
other packages, the difference passes only once it is explained in writing.

Standard errors of mixed models come from the exact second derivatives of the
likelihood, not from a finite-difference approximation. On the test models they
differ from a high-precision lme4 run by at most 0.002% (relative).

glmm was also stress-tested on many hard and borderline models. On such models
it returns a result with flags that say what happened, instead of only printing
a warning. A variance that falls to zero is set to exactly zero and flagged. A
redundant column is dropped and marked, as in lme4. When a fit fails, the
`converged` flag says so. glmm does not crash.

The package versions, the checks and how to rerun them are in
[`validation.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/validation.md).

## Multi-solver paradigm

glmm combines several optimizers for mixed models. Where it has a cheap exact
gradient, it uses BFGS. Elsewhere it uses BOBYQA, which needs no gradient.
BOBYQA also refines the BFGS result and takes over when BFGS fails.

On larger GLMMs, a fast approximate pass first finds a good starting point, and
the full search starts from there. When the main search does not converge, the
fit tries a second search strategy from the same start and keeps the better
answer.

The linear algebra also uses more than one method. Common model shapes use
small dense blocks, one per group. Larger or more complex designs go to a
sparse solver instead of being refused. Given the same variance parameters,
both return the same answer. The choice affects speed only.

## Speed

glmm moves as little data as it can. One pass over the data builds a small
summary per group, and every later step of the search works on those summaries.
The cost of each step therefore grows with the number of groups, not the number
of rows.

glmm is also designed for refitting the same model many times. Its working
memory depends only on the shape of the model. So refitting the same model on
new data, as in a power simulation or a bootstrap, needs no new memory. Each
refit can also start from the previous answer.

The search also needs few steps. Its settings were tuned on the 923 test models
and kept only where they cut the number of steps without changing any result.
With a cheap exact gradient it needs even fewer.

Each of these choices, and how it differs from lme4 and MixedModels.jl, is in
[`glmm-design.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/glmm-design.md).

## Quick start

`data` is any data table: a dict of columns, or a pandas / polars DataFrame.

```python
import glmm

# OLS: fixed effects only
fit = glmm.fit(data, "y ~ x1 * x2")

# LMM: correlated random intercept and slope (REML)
fit = glmm.fit(data, "y ~ x1 + (1 + x1 | subject)")

# The same LMM by maximum likelihood, as lme4's REML = FALSE
fit_ml = glmm.fit(data, "y ~ x1 + (1 + x1 | subject)", reml=False)

# GLMM: binomial with a probit link, 7-node adaptive quadrature
fit = glmm.fit(data, "passed ~ x1 + (1 | school)", "binomial", link="probit", nagq=7)

fit.summary()                     # coefficient table + variance components
fit.beta, fit.se                  # estimates and standard errors
fit.coef_df, fit.confint()        # t-test df (Satterthwaite on an LMM), intervals
fit.converged, fit.singular       # result flags
glmm.anova(fit)                   # tests of the fixed terms (Type III)
glmm.compare(data, "y ~ x1", "y ~ x1 + (1 | subject)")  # two nested models, fitted and compared
glmm.pairwise(fit, "f")           # pairwise comparisons of the levels of factor f (Holm)
fit.predict(new_data)             # predictions on new rows
```

## Documentation

- [Python tutorial](https://github.com/pawlenartowicz/glmm/blob/main/documentation/tutorial-python.md):
  every option, the `Fit` object, and warm starts.
- [Worked examples](https://github.com/pawlenartowicz/glmm/blob/main/documentation/examples-python.md)
  on real datasets.
- [Coming from statsmodels](https://github.com/pawlenartowicz/glmm/blob/main/documentation/coming-from-statsmodels.md)
  and [coming from lme4](https://github.com/pawlenartowicz/glmm/blob/main/documentation/coming-from-lme4.md).
- [What the formula accepts](https://github.com/pawlenartowicz/glmm/blob/main/documentation/formula.md).
- [Tests of fixed terms and model comparisons](https://github.com/pawlenartowicz/glmm/blob/main/documentation/hypothesis-tests.md).
- [Fixes for common errors](https://github.com/pawlenartowicz/glmm/blob/main/documentation/troubleshooting.md).
- [All documentation](https://github.com/pawlenartowicz/glmm/blob/main/documentation/index.md).

## Citation

If you use glmm in research, please cite:

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
