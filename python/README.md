# glmm

**`glmm`: mixed models from an R-style formula, with NumPy as the only dependency.**

[![License: LGPL-3.0-or-later](https://img.shields.io/badge/license-LLGPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

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

## Install

```bash
pip install glmm
```

Requires Python 3.10+. The only runtime dependency is NumPy.

## Quick examples

`df` is any data table: a dict of columns, or a pandas / polars DataFrame.

```python
import glmm

# OLS: fixed effects only
fit = glmm.fit(df, "y ~ x1 * x2")

# LMM: correlated random intercept and slope (REML)
fit = glmm.fit(df, "y ~ x1 + (1 + x1 | subject)")

# The same LMM by maximum likelihood, as lme4's REML = FALSE
fit_ml = glmm.fit(df, "y ~ x1 + (1 + x1 | subject)", reml=False)

# GLMM: binomial with a probit link, 7-node adaptive quadrature
fit = glmm.fit(df, "passed ~ x1 + (1 | school)", "binomial", link="probit", nagq=7)

fit.summary()                     # coefficient table + variance components
fit.beta, fit.se                  # estimates and standard errors
fit.coef_df, fit.confint()        # t-test df (Satterthwaite on an LMM), intervals
fit.converged, fit.singular       # result flags
glmm.anova(fit)                   # tests of the fixed terms (Type III)
glmm.compare(df, "y ~ x1", "y ~ x1 + (1 | subject)")  # two nested models, fitted and compared
```

## Documentation

Full walkthrough — every knob, the `Fit` object, and warm starts:
[`tutorial-python.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/tutorial-python.md).

The algorithm map (dispatch graph, solver paths, tuning knobs) is traced to
code in
[`documentation/algorithms.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/algorithms.md).

What the formula accepts:
[`formula.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/formula.md).
Tests of fixed terms and model comparisons:
[`hypothesis-tests.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/hypothesis-tests.md).
Fixes for common errors:
[`troubleshooting.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/troubleshooting.md).

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
