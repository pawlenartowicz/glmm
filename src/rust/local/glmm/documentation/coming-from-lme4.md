# Coming from lme4

Migrating from `statsmodels` instead? See
[`coming-from-statsmodels.md`](coming-from-statsmodels.md) — that migration
has its own shape (patsy formulas, a construct-then-fit API) and lives on its
own page.

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
| `ranef(fit)` | `ranef(fit)` — named list of data frames, one per grouping, rows named by level | `fit.ranef_blocks` — a list of `{group, terms, levels, values}`, `values` an `(n_levels, n_terms)` array (not a DataFrame; the package's only dependency is numpy) |
| `fitted(fit)` | `fitted(fit)` — named numeric vector over the model frame's rows | `fit.fitted` — `(n,)` array |

## What is deliberately missing

`fastglmm` is deliberately scoped to **fast fitting**: fixed effects, Wald
standard errors, variance components on the SD/correlation scale, and the
conditional modes with the per-row means they imply. Anything the engine
cannot compute honestly today — `predict`, `residuals`, profiling — is an
error naming the reason, never a silently different answer. `predict()` needs
a design matrix built from rows the fit never saw, and the formula machinery
is Rust-side; `residuals()` would have to guess which of lme4's four `type=`
residuals you meant, and guessing wrong is worse than erroring. `coef()`
errors for the same reason it always did — lme4's `coef()` means fixed +
random effects per group, and returning the fixed part alone would silently
differ — but you can now build it yourself from `fixef()` and `ranef()`.

Conditional variances are the one gap inside the `ranef` surface:
`ranef(condVar = TRUE)` and lme4's `condsd` column have no counterpart, so the
modes come with no uncertainty attached. Read them accordingly.

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
- `cbind(successes, failures) ~ …` works with `family = binomial`, lowered to
  the same proportion + trial-count-weights objective lme4 uses underneath it
  — see [`formula.md`](formula.md).
- Factors are always coded with treatment contrasts (base = the column's
  first level); `contrasts=` is not an accepted argument — relevel the
  factor instead. See [`formula.md`](formula.md#not-accepted-and-the-workaround).
- **Composite grouping levels are ordered and spelled differently.** For an
  explicit nesting `(1 | A/B)`, lme4 names the inner block `B:A` and labels its
  rows child-first (`b1:a1`, `b1:a2`, …, first component varying fastest);
  `glmm` names it `A:B` and labels the rows parent-first (`a1:b1`), grouped by
  parent. The values are identical and every row is labelled — only the
  printout's row order and label spelling differ. A crossed interaction
  `(1 | A:B)` matches lme4 exactly, since both sort lexicographically on the
  joined label.
- **An unused grouping level gets a row where lme4 has none.** A factor level
  declared between two observed ones costs random-effect width even with no
  rows, so `ranef` reports it with its mode shrunk to zero and the fit warns
  (`fastglmm_unused_grouping_levels` / `UnusedGroupingLevelsWarning`).
  `droplevels()` removes both the row and the wasted width. A level declared
  after the last observed one costs nothing and never appears.
- REML is the only fit mode for LMMs (no ML switch), and all reported
  standard errors are Wald, not profile or bootstrap — see
  [`conventions.md`](conventions.md#flags-on-the-result) for how the
  `converged`/`singular` flags mark when those numbers should be trusted.

## Three models, side by side

The same three fits, in lme4, `fastglmm` (R), and `glmm.fit` (Python). These
are recipes 1, 4 and 2 from the worked examples
([`examples-python.md`](examples-python.md),
[`examples-r.md`](examples-r.md)) — the `fastglmm`/`glmm.fit` code below is
copied verbatim from those pages, already run against real data and checked
against the frozen lme4 goldens there. The lme4 column is the formula that
produced each rung's frozen golden (`validation/manifest.json`'s `r_formula`
for that rung), fit the ordinary lme4 way, for direct comparison — see the
linked recipe for the actual printed output and the oracle cross-check. Every
lme4 formula below spells its intercept explicitly (`1 + …`), lme4's own
convention; the `fastglmm`/`glmm.fit` column drops that leading `1 +`, since
the shared parser has no bare-`1` fixed-effect term and errors on it
(`expected identifier, got '1'`) — the intercept is never optional there,
only ever implicit. See [`formula.md`](formula.md) for what the parser
accepts.

### Correlated random slope (`sleepstudy`)

lme4:

```r
library(lme4)

data(sleepstudy)

fit <- lmer(Reaction ~ 1 + Days + (1 + Days | Subject), sleepstudy)

summary(fit)
```

`fastglmm` (R):

```r
library(lme4)
library(fastglmm)

data(sleepstudy)

fit <- fastglmm(Reaction ~ Days + (1 + Days | Subject), sleepstudy)

summary(fit)
```

`glmm.fit` (Python):

```python
import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "sleepstudy.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "Reaction": [float(r["Reaction"]) for r in rows],
    "Days": [float(r["Days"]) for r in rows],
    "Subject": [r["Subject"] for r in rows],
}

fit = glmm.fit(data, "Reaction ~ Days + (1 + Days | Subject)")

assert fit.converged
fit.summary()
```

Full recipe, with output and the goldens cross-check:
[`examples-python.md#1-correlated-random-slope-sleepstudy`](examples-python.md#1-correlated-random-slope-sleepstudy) /
[`examples-r.md#1-correlated-random-slope-sleepstudy`](examples-r.md#1-correlated-random-slope-sleepstudy).

### Aggregated binomial via `weights=` (`cbpp`)

lme4:

```r
library(lme4)

data(cbpp)

fit <- glmer(cbind(incidence, size - incidence) ~ 1 + period + (1 | herd),
             cbpp, family = binomial)

summary(fit)
```

`fastglmm` (R):

```r
library(lme4)
library(fastglmm)

data(cbpp)
cbpp$prop <- cbpp$incidence / cbpp$size

fit <- fastglmm(prop ~ period + (1 | herd), cbpp, family = binomial(), weights = size)

summary(fit)
```

`glmm.fit` (Python):

```python
import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "cbpp.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

incidence = [float(r["incidence"]) for r in rows]
size = [float(r["size"]) for r in rows]
data = {
    "prop": [i / s for i, s in zip(incidence, size)],
    "period": [r["period"] for r in rows],
    "herd": [r["herd"] for r in rows],
}

fit = glmm.fit(data, "prop ~ period + (1 | herd)", family="binomial", weights=size)

assert fit.converged
fit.summary()
```

Full recipe, with output and the goldens cross-check:
[`examples-python.md#4-aggregated-binomial-via-weights-cbpp`](examples-python.md#4-aggregated-binomial-via-weights-cbpp) /
[`examples-r.md#4-aggregated-binomial-via-weights-cbpp`](examples-r.md#4-aggregated-binomial-via-weights-cbpp).

### Crossed grouping factors (`Penicillin`)

lme4:

```r
library(lme4)

data(Penicillin)

fit <- lmer(diameter ~ 1 + (1 | plate) + (1 | sample), Penicillin)

summary(fit)
```

`fastglmm` (R):

```r
library(lme4)
library(fastglmm)

data(Penicillin)

# NOTE: no bare "1" fixed-effect term (the intercept is always implicit).
fit <- fastglmm(diameter ~ (1 | plate) + (1 | sample), Penicillin)

summary(fit)
```

`glmm.fit` (Python):

```python
import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "Penicillin.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "diameter": [float(r["diameter"]) for r in rows],
    "plate": [r["plate"] for r in rows],
    "sample": [r["sample"] for r in rows],
}

# NOTE: no bare "1" fixed-effect term (the intercept is always implicit).
fit = glmm.fit(data, "diameter ~ (1 | plate) + (1 | sample)")

assert fit.converged
fit.summary()
```

Full recipe, with output and the goldens cross-check:
[`examples-python.md#2-crossed-grouping-factors-penicillin`](examples-python.md#2-crossed-grouping-factors-penicillin) /
[`examples-r.md#2-crossed-grouping-factors-penicillin`](examples-r.md#2-crossed-grouping-factors-penicillin).

## The `cbind()` migration

lme4 writes an aggregated binomial — one row per group of trials rather than
one row per trial — as a two-column response built with `cbind()`:

```r
cbind(incidence, size - incidence) ~ period + (1 | herd)
```

`cbind()` here takes two column names, not an arithmetic expression: compute
the failures column yourself first (`size - incidence`), then pass both as
plain columns, with `family = binomial`:

```r
cbpp$failures <- cbpp$size - cbpp$incidence
fastglmm(cbind(incidence, failures) ~ period + (1 | herd), cbpp, family = binomial())
```

The kernel lowers it to the proportion + trial-count-weights objective
internally, so there is no further data-prep arithmetic to do by hand beyond
that one subtraction.
