# Coming from lme4

Migrating from `statsmodels` instead? See
[`coming-from-statsmodels.md`](coming-from-statsmodels.md) — that migration
has its own shape (patsy formulas, a construct-then-fit API) and lives on its
own page.

## Call mapping

| lme4 | fastglmm (R) | glmm (Python) |
|---|---|---|
| `lmer(f, d)` / `glmer(f, d, family)` | `fastglmm(f, d, family = ...)` | `glmm.fit(d, "f", family)` |
| `summary(fit)` (lmerTest's on an LMM) | `summary(fit)`: lme4-shaped, with the REML criterion (REML fits) or the `AIC`/`BIC`/`logLik`/`deviance`/`df.resid` row (ML fits), scaled residuals, variance components, and a coefficient table as lmerTest, `lm` and `glm` print it: t with a Satterthwaite `df` column on a Gaussian mixed model, t on `n - p` on OLS and on the Gamma and inverse-Gaussian GLMs with an estimated dispersion, Wald z elsewhere | `fit.summary()`: the same layout, printed and returned as a string; the df and p-values are also `fit.coef_df` and `fit.coef_p` |
| `lmer(f, d, REML = FALSE)` | `fastglmm(f, d, REML = FALSE)` | `glmm.fit(d, "f", reml=False)` |
| `AIC(m0, m1)`, `BIC(m0, m1)` | `AIC(m0, m1)`, `BIC(m0, m1)`: the `stats` table, except that a REML fit beside an ML fit, and REML fits with different fixed effects, stop with an error | `m.aic`, `m.bic` (properties, one fit each); the `AIC` and `BIC` columns of `glmm.anova(m0, m1)` on ML fits |
| `anova(fit)` (lmerTest's), `car::Anova(fit)` | `anova(fit)`: one row per fixed term, built at fit time, with no refit when called; Type III by lmerTest's construction by default, `type = "II"` on request; lmerTest's F with Satterthwaite df on a Gaussian mixed model, F on `n - p` on OLS and the Gamma and inverse-Gaussian GLMs with an estimated dispersion, car's Wald chi-squared elsewhere, but likelihood-ratio chi-squared on a Poisson GLMM (next row) | `glmm.anova(fit)`, `glmm.anova(fit, type="II")`: the same table, a `TermTable` |
| `afex::mixed(f, d, family = ..., method = "LRT")` on a GLMM; `car::Anova(glm_fit, type = 2, test.statistic = "LR")` on a GLM (on a `glmer` fit car gives only Wald chi-squared rows) | `fastglmm(f, d, family = ..., anova_test = "LRT")`, then `anova(fit, test = "LRT")` (Type III, afex's default type) or `anova(fit, type = "II", test = "LRT")` (car's Type II rule, on a GLM or a GLMM); a Poisson GLMM stores these rows by default, and `anova(fit)` gives them. Not on a Gaussian model, whose rows are Satterthwaite F tests: for a likelihood-ratio test of fixed terms there, use `compare(data, f0, f1)`, which fits both models by ML | `glmm.fit(d, "f", family, anova_test="LRT")`, then `glmm.anova(fit, test="LRT")` or `glmm.anova(fit, type="II", test="LRT")` |
| `anova(m0, m1)` | `anova(m0, m1)`: the two fits compared as they are, never refitted; lme4's columns on a likelihood-ratio test, `anova.lm`'s on two OLS fits | `glmm.anova(m0, m1)`: the same table, a `ComparisonTable` |
| `anova(m0, m1)` with lme4's ML refit; `lmerTest::ranova(fit)`; `RLRsim::exactRLRT()` | `compare(data, f0, f1)`: fits both formulas on the same rows by the estimator the comparison needs (REML when only the random part differs, ML when the fixed part does), with a boundary-aware reference for a variance component (the exact RLRT for a Gaussian model's single random intercept, else the 50:50 mixture) | `glmm.compare(data, f0, f1)`: the same, a `Comparison` with `table`, `fit0`, `fit1` |
| `fixef(fit)` | `fixef(fit)` — named fixed-effect estimates; aliased columns `NA` | `fit.beta` (estimates) with `fit.names` (aligned coefficient names); `fit.aliased` flags rank-deficient columns |
| `vcov(fit)` | `vcov(fit)` — full `p × p` Wald covariance | `fit.vcov` — `(p, p)` full Cov(β̂) |
| `VarCorr(fit)` | `VarCorr(fit)` — SD/correlation-scale variance components, lme4-shaped, with a `Residual` row for a Gaussian mixed fit | `fit.varcorr` (vech-packed per grouping) plus `fit.stddev_corr(group_idx)`, which splits a grouping's block into a stddev vector and a correlation matrix |
| `confint(fit)` | `confint(fit)`: Wald intervals off `vcov()`, on the t quantile where `summary()` tests with t (lmerTest's `contest1D(..., confint = TRUE)` on a Gaussian mixed model) and the normal quantile where it tests with z; `method = "profile"`/`"boot"` are not available and say so | `fit.confint(level=0.95)`: the same intervals as a `(p, 2)` array in `fit.names` order, no `parm` argument |
| `isSingular(fit)` | `isSingular(fit)` — boundary-fit flag, lme4's condition, computed by the kernel | `fit.singular` |
| `ranef(fit)`, `ranef(fit, condVar = TRUE)` | `ranef(fit)` — named list of data frames, one per grouping, rows named by level, with lme4's `postVar` attribute (terms × terms × levels) unless `condVar = FALSE` | `fit.ranef_blocks` — a list of `{group, terms, levels, values, postvar, condsd}`, `values` an `(n_levels, n_terms)` array, `postvar` an `(n_levels, n_terms, n_terms)` array and `condsd` its diagonal's square roots, the two `None` on a fit that forms no conditional variances (not a DataFrame; the package's only dependency is numpy) |
| `fitted(fit)` | `fitted(fit)` — named numeric vector over the model frame's rows | `fit.fitted` — `(n,)` array |
| `predict(fit, newdata, re.form, type, allow.new.levels, se.fit)` | `predict(fit, newdata, re.form = NULL, type = "link", allow.new.levels = TRUE, se.fit = FALSE, level = 0.95, future = FALSE)`: a named vector, or with `se.fit` or `future` a list of `fit`, `se.fit`, `lower`, `upper`, `future.lower`, `future.upper`; `re.form` takes `NULL` or `NA` only | `fit.predict(newdata, type="link", re_form=None, allow_new_levels=True, se_fit=False, level=0.95, future=False)`: an array, or a dict of arrays; `re_form="NA"` for the fixed effects only |
| `simulate(fit, nsim, seed)` | `simulate(fit, nsim = 1, seed = NULL)`: a data frame `sim_1`, …, on the model frame's rows | `fit.simulate(nsim=1, seed=None)`: an `(nobs, nsim)` array |
| `performance::r2_nakagawa(fit)`, `performance::icc(fit)` | `r2(fit)` gives a named vector with `marginal` and `conditional`, `icc(fit)` one with `adjusted` and `unadjusted`; `summary(fit)` prints the marginal and conditional R² and the adjusted ICC in one line. Computed when called, as performance computes them (on a GLMM the lognormal approximation only), for all groupings together, not per grouping. On a Poisson or negative binomial GLMM each `r2()` or `icc()` call refits the null model once, and `summary()` prints a line that points to them instead of the values. `r2()` and `icc()` are plain functions, not methods: when both packages are attached, the one attached last wins, so call `fastglmm::r2()` and `fastglmm::icc()` | `fit.r2()` gives `{"marginal": .., "conditional": ..}`, `fit.icc()` gives `{"adjusted": .., "unadjusted": ..}`; the same line in `fit.summary()` |
| `effectsize::eta_squared(anova(fit))`, `effectsize::F_to_eta2(F, df1, df2)` | the `Eta2_partial` column of `anova(fit)` on F tables, `F_to_eta2`'s formula on the table's own F and df; approximate on a Gaussian mixed model, whose df are Satterthwaite's | the same column of `glmm.anova(fit)` |
| `pairs(emmeans(fit, ~ f))`, `pairs(emmeans(fit, ~ f \| h))`, `confint(pairs(..))` | `pairwise(fit, "f")`, `pairwise(fit, "f", by = "h")`, `pairwise(fit, "f", by = "x", at = c(1, 2))`, then `confint(pw)`: emmeans' columns and reference grid (other factors averaged with equal weights, numeric variables at their means), Satterthwaite df on a Gaussian mixed model (emmeans' `lmer.df = "satterthwaite"`), odds ratios or ratios on a logit or log link. **The default adjustment is Holm, not emmeans' Tukey**: pass `adjust = "tukey"` for emmeans' p-values. One factor per call, one `by`; no custom grids, weights or simple slopes, and no emmeans methods for fastglmm | `glmm.pairwise(fit, "f", by=None, at=None, adjust="holm")`: the same table, a `PairwiseTable`; `pw.confint(level=0.95)` |

## What is deliberately missing

`fastglmm` is deliberately scoped to **fast fitting**: fixed effects, Wald
standard errors, variance components on the SD/correlation scale, the
conditional modes with their conditional variances and the per-row means they
imply, predictions on new rows, simulated responses, the pairwise
comparisons of a factor's levels (`pairwise()`), and the tests of fixed
terms and of two nested models (`anova()`, `compare()`; see
[`hypothesis-tests.md`](hypothesis-tests.md)). There is no parametric
bootstrap test and no `ranova()`-style table of every random term at once.
Anything the engine cannot compute honestly (profiling) is an error naming
the reason, never a silently different answer. `residuals()` gives response
and Pearson residuals; deviance and working residuals are not offered.
`coef()` errors — lme4's `coef()` means fixed + random effects per group, and
returning the fixed part alone would silently differ — but you can build it
yourself from `fixef()` and `ranef()`.

This is the same discipline the kernel applies everywhere: it only surfaces
what it can compute honestly. See
[`validation.md`](validation.md) for how that claim is checked against the
four oracles (lme4, glmmTMB, MixedModels.jl, GLMMadaptive).

## Behavioral differences to watch

- `nAGQ = k` only turns on adaptive quadrature for a binomial, Poisson,
  negative-binomial or Gamma GLMM with a single grouping factor and up to 3
  random effects per group (lme4 has no `nAGQ` for `glmer.nb` beyond what
  `glmer` offers, and none for Gamma with φ estimated); any
  other shape warns and falls back to Laplace instead of erroring the way
  lme4 does — watch for the warning (`r/README.md`). A random slope without
  a random intercept, `(0 + x | g)`, counts one random effect more than lme4
  counts, since the kernel keeps the intercept column at zero variance, so
  the limit of 3 is reached one slope sooner.
- **`(f || g)` on a factor `f` keeps the factor's block correlated**, as in
  lme4, which expands it to `(1 | g) + (0 + f | g)`. glmm codes the factor
  with K − 1 dummies, so it fits the factor's term as `(1 + f | g)` and
  merges the `(1 | g)` into it: the same model, without the intercept
  variance that lme4's fit cannot identify. The deviance and the fixed
  effects match lme4; the variance components are laid out differently. For
  no correlation between any two columns, write `zerocorr(f | g)` (as in
  MixedModels.jl). On a numeric slope the two syntaxes are the same. See
  [`formula.md`](formula.md#uncorrelated-and-intercept-free-random-effects).
- **Several terms on one grouping print as one group.** `summary()` and
  `VarCorr()` list all columns of `(x || g)` under `g`, where lme4 prints one
  group per term (`g`, `g.1`). As in lme4, a correlation is printed only
  between two columns of one term.
- **A random-effect column in two terms on one grouping is an error**, after
  factor coding, where lme4 fits the model. `(1 + x | g) + (0 + f | g)` with
  a factor `f` repeats the intercept, since the factor's term holds it; write
  `(1 + x + f | g)` or `(x + f || g)`. lme4's fit of that model is not
  identified either.
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
- An LMM is fitted by REML unless you ask for ML, as in lme4: `REML = FALSE`
  (Python `reml=False`) fits it by maximum likelihood, and that summary prints
  "Note: p-values can be somewhat anticonservative." under its title. On the
  frozen lmerTest `REML = FALSE` references (26 models) the ML log-likelihood
  agrees with lme4's to within 1.5e-9 relative, and the Satterthwaite df with
  lmerTest's to within 3.5e-4. Unlike lme4, `REML = TRUE` on a model that is
  not Gaussian warns that it was ignored, and `REML = TRUE` on a Gaussian model
  without random effects reports the REML log-likelihood, as
  `logLik(lm, REML = TRUE)`. `logLik()` of a REML fit is the REML criterion,
  as in lme4. `AIC()` and `BIC()` refuse what lme4 lets through: REML fits
  with different fixed effects (their REML criteria are not comparable; use
  `compare()` or fit both with `REML = FALSE`) and a REML fit beside an ML fit.
- **`anova(m0, m1)` never refits.** lme4's `anova()` refits two REML fits by
  ML, even when only their random parts differ. Here two REML fits with
  different fixed parts are an error that points to `compare(data, f0, f1)`
  (which fits both by ML) or to `REML = FALSE`, and two REML fits with the
  same fixed part are compared on the REML criterion, as `ranova()` and
  `anova(refit = FALSE)` do. To test `(1 | g)` against no random effects, make
  the OLS fit with `REML = TRUE` (lmerTest's `ranova()` uses that REML value;
  lme4's `anova(lmer, lm)` uses the `lm` fit's ML one).
- **A variance component is not tested against plain chi-squared.** lme4's
  `anova()` and `ranova()` use chi-squared on the parameter difference, which
  ignores the boundary at zero and roughly doubles the p-value. Here a
  Gaussian model whose only random term is one unweighted random intercept
  with at most 100 grouping levels with rows, compared by REML against the
  same fixed part, gets the exact restricted likelihood-ratio test (RLRsim's
  `exactRLRT()` method, with a fixed default seed), any other single dropped
  random effect the 50:50 mixture of chi-squared distributions (still
  conservative, and labelled so), and only a comparison that drops more than
  one random effect falls back to chi-squared, labelled "ignores the boundary;
  conservative". When the random parts alone differ, a component the larger
  fit puts at exactly zero gives a statistic of exactly 0 and p = 1.
- **Per-term tables follow lmerTest on Gaussian mixed models and car on the
  rest.** Type III is lmerTest's construction on every family, so it does not
  depend on the treatment coding (car's Type III on a treatment-coded `glm`
  tests a different hypothesis); Type II uses lmerTest's `X'X` metric on
  Gaussian mixed models and car's covariance metric on OLS, GLMs and GLMMs.
  An interaction without its marginal terms gets no table.
- **Likelihood-ratio rows of the fixed terms are refitted at fit time, and
  Type II is car's.** afex's `mixed(method = "LRT")` on a GLMM, and car's
  `Anova(test.statistic = "LR")` on a GLM, refit when they are called; here
  the refits are made when the model is fitted, with `anova_test = "LRT"` (or
  by default on a Poisson GLMM), and `anova(fit, test = "LRT")` reads them.
  The Type III rows are afex's at its default `type = 3`. For Type II glmm
  follows the rule of car's `Anova(type = 2, test.statistic = "LR")`, which
  car computes on a GLM only (on a `glmer` fit car gives only Wald
  chi-squared rows); glmm applies the same rule to GLMMs. It is not
  afex's `type = 2`, which is sequential: on `y ~ g * h + x` afex tests `x`
  as `g + h + x` against `g + h`, glmm (car's rule) as `g * h + x` against
  `g * h`. The statistic column is `Chisq`, as in the Wald table (car
  names it `LR Chisq`). The default differs from car's: `Anova()` on a
  `glmer` fit gives Wald rows, while `anova(fit)` here gives a Poisson GLMM
  the likelihood-ratio rows and every other fit its Wald or F rows (see
  [`hypothesis-tests.md`](hypothesis-tests.md#wald-or-likelihood-ratio-rows-test-and-anova_test)).
- **A model with no more observations than random effects is fitted.** lme4
  stops on such a fit by default (`check.nobs.vs.nRE = "stop"`; for example a
  random intercept and slope on two observations per subject); glmm fits it
  and says nothing. The model is weakly identified there; see
  [`validation.md`](validation.md#what-is-gated).
- All reported standard errors are Wald, not profile or bootstrap — see
  [`conventions.md`](conventions.md#flags-on-the-result) for how the
  `converged`/`singular` flags mark when those numbers should be trusted.

## Predictions, simulation and conditional variances

- **A grouping level the fit did not see is accepted by default.** lme4's
  `predict()` stops on it unless you pass `allow.new.levels = TRUE`; here
  `allow.new.levels = TRUE` (Python `allow_new_levels=True`) is the default.
  Such a row gets a random effect of 0 for that grouping, its other groupings
  keep theirs, and the call warns (`fastglmm_new_grouping_levels` /
  `NewGroupingLevelsWarning`). Pass `FALSE` to make it an error, as in lme4.
  A level of a fixed-effect factor that the fit did not see is always an
  error, as in lme4.
- **`re.form` is `NULL` or `NA` only**: every grouping's random effects, or
  none. A formula that keeps some groupings is not accepted.
- **`se.fit` on a GLMM uses the fit's covariance of the fixed effects.** That
  is the Hessian one by default, the same as `vcov(fit)`. lme4's `predict()`
  uses the RX covariance even when its `vcov()` is the Hessian one, so on a
  GLMM the two `se.fit` differ. A fit with the RX covariance matches lme4:
  `wald.se = "rx"` in R, `wald_se="rx"` in Python, `FitOptions::wald_se =
  WaldSe::Rx` in Rust. On a Gaussian model the two agree. Like lme4's, the
  standard error holds the variance parameters at their estimates and adds
  nothing for a new level; glmmTMB's `se.fit` adds both, so it is larger.
  On a nested model such as `(1 | batch/cask)`, lme4 2.0-6's `se.fit` reads
  the wrong rows of its covariance; glmm's is the exact value.
- **On a Gamma GLMM the conditional variances differ from lme4's.** lme4's
  `ranef(condVar = TRUE)` there uses the expected information, which is the
  same for every level of the same size; glmm uses the observed curvature at
  each level's conditional mode, which depends on the level's data. On a
  binomial logit or a Poisson log GLMM the two are the same. On a Gamma GLMM
  lme4's estimates also differ from the maximum-likelihood ones that glmm and
  glmmTMB report, so its predictions differ too.
- **Some fits have no conditional variances**: a fit made with
  `condvar = FALSE` (Python `condvar=False`), which skips them to save time
  (up to about the time of the fit on a large crossed model), a model with a
  random slope on a grouping other than the first one in the formula, a model over the
  dense size limits (more than 500 crossed levels, for example), and the
  larger GLMMs that use the packed layout. There `ranef(fit)` has no `postVar`
  attribute (an explicit `condVar = TRUE` is an error), Python's `postvar` and
  `condsd` are `None`, and `predict()` gives a standard error, or a future
  bound, with `re.form = NA` only. The predictions themselves work on every
  fit.
- **A missing value in `newdata` is an error.** lme4's `predict()` keeps the
  row and returns `NA` for it (`na.action = na.pass`); here, drop those rows
  first.
- **A model fitted with the `offset=` argument predicts its own rows only.**
  The values of that offset for new rows are unknown. Write the offset as an
  `offset()` term in the formula instead; it is then read from `newdata`.
- **Levels of `A/B` and `A:B` groupings are matched by their label.** A new
  row's level is the two labels joined by `:`, parent first (`a1:b1`). When a
  label itself contains `:`, two different pairs can give the same joined
  label, for example `("a:b", "c")` and `("a", "b:c")`. The fit keeps them as
  two levels, but `predict()` cannot tell them apart and gives both the
  conditional mode of one of them. Avoid `:` in such labels.
- **A variable used only in a random slope** (`y ~ 1 + (x | g)`) is read from
  `newdata` whenever `newdata` has a grouping column, also with
  `re.form = NA`, where it does not change the prediction. Without any of the
  grouping columns it is not needed.
- **`future = TRUE` gives bounds for a future observation**, which lme4 does
  not offer: on a non-Gaussian mixed model, the family's quantiles mixed over
  the uncertainty of the linear predictor (`se.fit`), on the response scale.
  With `re.form = NA` that uncertainty is the fixed effects' only, `x'Vx`;
  the random-effect variance is not added, so the bounds are for an
  observation in a group whose random effect is 0, not for one in a new group
  drawn at random. The same holds for a row with a new level. On new rows a
  binomial bound is a proportion of one trial, because `newdata` carries no
  trial counts; on the fit's own rows (no `newdata`) it uses the fit's trial
  counts.
- **`simulate()` draws on the fit's own rows, with new random effects** in
  every draw, as lme4's default `use.u = FALSE` does. There is no `newdata`,
  `use.u = TRUE`, `re.form` or `newparams`. A seed gives the same draws on
  every platform and in both ports; it is not R's random number stream, so
  the draws differ from lme4's.

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
convention; the `fastglmm`/`glmm.fit` column leaves that `1 +` out. Both
spellings give the same model. See [`formula.md`](formula.md) for what the
parser accepts.

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
