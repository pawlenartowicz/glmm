# glmm

**Linear-regression family — OLS → GLM → LMM → GLMM — in pure Rust, with a six-name Python API.**

[![License: GPL-3.0-or-later](https://img.shields.io/badge/license-GPL--3.0--or--later-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

Fits fixed-effect and mixed (random-intercept / random-slope) models for
Gaussian, Binomial (logit/probit), Poisson, Gamma, and Negative-Binomial
outcomes. The numerics are the [`glmm`](https://crates.io/crates/glmm) Rust
crate — no BLAS/LAPACK system dependency, no `unsafe`, validated against
R/lme4 and Julia/MixedModels.jl goldens.

## Install

```bash
pip install glmm
```

Requires Python 3.10+. The only runtime dependency is NumPy.

## One call

You hand `fit` a data table and an R-style formula. It parses the formula
against the table's columns, builds the design matrix, fits, and returns a
`Fit`. There is no model object to construct.

```python
import glmm

data = {
    "y":     [1.02, 1.05, 1.11, 1.13, 1.21, 1.24, 1.30, 1.33, 1.42, 1.44, 1.51, 1.53],
    "x1":    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "group": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
}

# y ~ x1 + (1 | group) — Gaussian, random intercept.
fit = glmm.fit(data, "y ~ x1 + (1 | group)")

assert fit.converged
fit.summary()   # prints the coefficient table (and returns it as a string)
```

The public surface is six names: `glmm.fit`, `glmm.Fit`, and the four warning
categories the diagnostics channel raises — `glmm.DiagnosticWarning` (the
base) plus `glmm.IllConditionedWarning`, `glmm.PirlsExhaustedWarning` and
`glmm.UnusedGroupingLevelsWarning`. Families, links, and knobs are string or
scalar arguments, not types.

`data` is documented as a `dict[str, array-like]`, but anything
column-addressable works — pandas / polars DataFrame, pyarrow Table — by
duck-typing, so there is no dataframe dependency.

## Model routing

Routing is inferred from the formula and family — fixed-only means OLS/GLM,
adding a `(… | g)` term makes it LMM/GLMM.

| `family` | default link | other links | fixed-only | with `(… \| g)` |
|---|---|---|---|---|
| `gaussian` | identity | — | OLS | LMM |
| `binomial` | `logit` | `probit` | GLM | GLMM |
| `poisson` | `log` | — | GLM | GLMM |
| `gamma` | `log` | `inverse` | GLM | GLMM |
| `negativebinomial` | `log` | — | GLM | GLMM |

```python
fit = glmm.fit(data, "s ~ x1 + (1 | group)", "binomial", link="probit", nagq=7)
```

The formula follows R conventions: `*` desugars to main effects + interaction,
`A/B` to nesting, `(1 + x | g)` is a correlated random intercept + slope, and
additional `(… | g2)` terms add crossed or nested grouping factors. Treatment
contrasts (R's default) code the factors against the column's first level; the
full syntax, rejected constructs and workarounds, and factor-coding rules are
in
[`formula.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/formula.md).

### Not yet implemented

Four combinations have an approved design but no kernel support, and raise a
clean `NotImplementedError`: `family="inversegaussian"`, `link="cloglog"`,
quasi-likelihood `dispersion=` on binomial/poisson, and a float `init_theta=`
seed (only the default `init_theta=None` cold start is supported). See
[`troubleshooting.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/troubleshooting.md)
for fixes to common errors.

## Documentation

Full walkthrough — every knob, the `Fit` object, and warm starts:
[`tutorial-python.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/tutorial-python.md).

The algorithm map (dispatch graph, solver paths, tuning knobs) is traced to
code in
[`documentation/algorithms.md`](https://github.com/pawlenartowicz/glmm/blob/main/documentation/algorithms.md).

## License

`GPL-3.0-or-later`.

---
**Paweł Lenartowicz** — [Freestyler Scientist](https://freestylerscientist.pl) · [GitHub](https://github.com/pawlenartowicz/) · [ORCID](https://orcid.org/0000-0002-6906-7217)
