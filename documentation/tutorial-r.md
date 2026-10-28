# Using `glmm` from R (the `fastglmm` package)

One page, four sections. The first walks the single entry point — `fastglmm()` —
end to end; the next two go deeper into the knobs and the object it returns; the
fourth is a short note on warm starts. The R surface is deliberately small:
**one function** — `fastglmm()` — plus the `"fastglmm"` object it returns, read
through lme4-shaped accessors (`summary`, `fixef`, `vcov`, `VarCorr`, `confint`,
`isSingular`). There is no `lmer`/`glmer` split — `fastglmm()` dispatches on the
family the way the kernel does. Three more calls test hypotheses: `anova()`,
on one fit or two, `compare()`, on one data set and two formulas, and
`pairwise()`, on the levels of one factor (§3).

Two R conventions differ from the Python port. The **formula comes first**,
`data` second (as in `lm`/`lme4`), the reverse of `glmm.fit(data, formula)`. And
`family` is an R **family object / function / string** (`binomial()`,
`poisson`, `"gamma"`), not a bare string — which brings one trap, the `Gamma()`
link default (§2).

> **Status:** this release ships the full API surface — `fastglmm()`, argument
> validation, every working accessor — wired end to end through the extendr
> binding: a valid call parses the formula, fits, and returns a real
> `"fastglmm"` object. One narrow combination is an open gap and raises a
> clean error naming the reason instead of fitting: quasi-likelihood
> `dispersion=` on binomial/poisson.

The package is not on CRAN yet. Install from a checkout (needs Rust — `cargo`
and `rustc >= 1.85` on the `PATH`):

```r
# in GLMM/r/
install.packages(".", repos = NULL, type = "source")
```

or from r-universe:

```r
install.packages("fastglmm",
                 repos = c("https://pawlenartowicz.r-universe.dev", getOption("repos")))
```

(Same models from other surfaces: the Python port —
[`tutorial-python.md`](tutorial-python.md) — and the Rust crate this package
binds — [`tutorial-rust.md`](tutorial-rust.md), hand-built inputs instead of a
formula.)

## 1. One call — formula in, fit out

`fastglmm()` is the only entry. You hand it a formula and a data frame; it parses
the formula against the frame's columns, builds the design, fits, and returns a
`"fastglmm"` object. There is no model object to assemble and no `n`/`p` to pass —
everything is inferred from the formula and the data.

```r
library(fastglmm)

data <- data.frame(
  y     = c(1.02, 1.05, 1.11, 1.13, 1.21, 1.24, 1.30, 1.33, 1.42, 1.44, 1.51, 1.53),
  x1    = c(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1),
  s     = c(0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1),
  group = factor(rep(letters[1:6], each = 2))
)

# y ~ x1 + (1 | group) — Gaussian, random intercept.
fit <- fastglmm(y ~ x1 + (1 | group), data)

fit$converged
summary(fit)   # lme4-shaped: criterion, random effects, coefficient table
```

Points worth knowing at this layer:

- `data` is a `data.frame` (or anything `as.data.frame()` accepts). Response and
  fixed-effect columns are read as `double`; grouping columns (the `g` in
  `(1 + x | g)`) may be factor, character, integer, or logical and are
  factorized to level ids. A character or logical predictor becomes a factor
  with **lexicographic** level order (exactly what `factor()` does, so a
  logical `x` has levels `FALSE`/`TRUE` and fits a coefficient `xTRUE`, as in
  lme4); a factor's declared level order is honored.
- The default `family = gaussian()` with no `(… | g)` term fits **OLS**; adding a
  random-effect term makes it an **LMM** (REML by default, maximum likelihood
  with `REML = FALSE`). The same split holds for every family: fixed-only ⇒
  GLM, `(… | g)` present ⇒ GLMM.
- The formula follows R conventions: `+`, `:`, `*` (main effects +
  interaction), `A/B` nesting, `- 1`/`0 +` (drops the intercept),
  `log()`/`sqrt()`/`exp()`/`I(x^k)` on one bare column, `offset()`,
  `cbind(s, f) ~ …` (with `family = binomial()`), and `(1 + x | g)` for a
  correlated random intercept + slope, `(x || g)` or `zerocorr(1 + x | g)`
  for an uncorrelated one, `(0 + x | g)` for a slope without a random
  intercept, with extra `(… | g2)` terms adding crossed/nested groupings. Not
  accepted — each a clear error with the fix: `poly()` and other general
  function calls (compute the column first), and `.`.
- Contrasts are always treatment coding based on the **first factor level**; to
  change the base, `relevel()` the factor (there is deliberately no `contrasts=`
  argument).
- A misspelled column name errors at the call (`column(s) not found in data`),
  and a misspelled argument errors too — `fastglmm()` intercepts unknown
  arguments rather than swallowing them, so a typo can't become a silent no-op.

## 2. Families and knobs

`family` accepts a family **object** (`binomial()`), a family **function**
(`binomial`), or a **string** (`"binomial"`); `link` rides on the object or
falls back to the family's default. Pass a link only where the kernel offers a
choice:

| `family` argument | fits | default link | other links | distribution param |
|---|---|---|---|---|
| `gaussian()` / `"gaussian"` | gaussian | identity | — | — |
| `binomial()` / `"binomial"` | binomial | logit | probit, cloglog | — |
| `poisson()` / `"poisson"` | poisson | log | — | — |
| `Gamma()` **(object)** | gamma | **inverse** | log | dispersion |
| `"gamma"` **(string)** | gamma | **log** | inverse | dispersion |
| `"negativebinomial"` | negative binomial | log | — | theta (estimated) |
| `inverse.gaussian()` **(object)** | inverse-Gaussian | **1/mu^2** | log | dispersion; **fixed-effect GLM only** |
| `"inversegaussian"` **(string)** | inverse-Gaussian | **log** | 1/mu^2 | dispersion; **fixed-effect GLM only** |

```r
fit <- fastglmm(s ~ x1 + (1 | group), data, family = binomial(link = "probit"), nAGQ = 7)
```

**The `Gamma()` link trap.** R's `Gamma()` family *object* defaults to
`link = "inverse"`, and an object is honored exactly as given — R semantics win.
The **string** `family = "gamma"` uses the glmm default `link = "log"` instead.
The two forms fit **different models**; `Gamma(link = "log")` and `"gamma"` are
the same, `Gamma()` and `"gamma"` are not. Choose deliberately.

The knobs:

- `nAGQ` — adaptive Gauss–Hermite node count; `1` = Laplace (default). Must be an
  **odd** integer `≤ 25`. `> 1` applies to binomial, Poisson, negative-binomial
  or Gamma mixed models with a single grouping factor and `≤ 3` random effects
  per group; **any other shape warns and falls back to Laplace** rather than
  erroring the way `lme4::glmer` does — the fit you get is a Laplace fit, and the warning is the only notice.
- `dispersion` — Gamma dispersion directive: `NULL` (estimate φ̂ by maximum
  likelihood, the default, on both a GLM and a GLMM, to place `logLik` at
  its maximum — but the two routes report and scale the SE by different
  numbers: a GLM's reported dispersion is the Pearson moment,
  `summary(glm)`'s convention, a GLMM's is that ML φ̂ itself; on
  inverse-Gaussian, GLM only, it is the Pearson moment), `"estimate"`
  (same), or a single number to hold φ fixed.
- `wald.se` — fixed-effect Wald-SE mode: `"hessian"` (default) or `"rx"`.
- `weights` — per-row prior weights, `lme4::glmer`'s `weights=`. On gaussian,
  gamma and inverse-gaussian these are precision weights: row `i` gets
  dispersion `φ/wᵢ`, the same convention as `lm`, `summary(glm)` and lme4,
  not glmmTMB's, which multiplies each row's log-density by `wᵢ` instead
  (see [`conventions.md`](conventions.md#prior-weights)). For an aggregated
  binomial, either write `cbind(successes, failures) ~ …` directly in the
  formula, or pass the success **proportion** as the response and the trial
  count here — the same model, spelled by hand. Weights must be strictly
  positive.
- `offset` — per-row known additive term on the linear-predictor scale,
  `glm`'s `offset=`: `η = offset + Xβ (+ Zb)`, with no coefficient estimated
  for it and no column added to the design. The canonical use is a Poisson
  rate model against a known exposure, `offset = log(exposure)`. Honored for
  every family and every solver path; `NULL` (default) means no offset.
  Evaluated in `data`, so an expression works. The formula also accepts an
  `offset(expr)` term (see [`formula.md`](formula.md)); passing both this
  argument and a formula `offset()` term is an error asking you to use one.
- `satterthwaite`: `TRUE` (default) computes lmerTest's Satterthwaite
  degrees of freedom for a Gaussian mixed model's t tests, in one pass after
  the fit. `FALSE` skips that pass: `summary()` then tests with z and says
  "df not computed (satterthwaite = FALSE)", `confint()` gives normal
  intervals with the same message, and `fit$df_method` is `"disabled"`. OLS
  and the Gamma and inverse-Gaussian GLMs keep their residual df `n - p`
  whatever it says; no other family reads it.
- `REML`: `NULL` (default) or `TRUE` fits a Gaussian mixed model by REML,
  `FALSE` by maximum likelihood, as `lmer(REML = FALSE)`. The summary of an ML
  fit is titled "Linear mixed model fit by maximum likelihood", prints the
  `AIC BIC logLik deviance df.resid` row in place of the REML criterion, and
  prints one line under the title: "Note: p-values can be somewhat
  anticonservative." On OLS (also when every grouping was dropped as
  single-level) the estimates are the same under every value: `TRUE` reports
  the REML log-likelihood, as `logLik(lm, REML = TRUE)`, and `NULL` and `FALSE`
  the ML one. Every other family is fitted by maximum likelihood; `FALSE` there
  changes nothing, and `TRUE` is ignored with an `argument_ignored` warning.
  `fit$reml` says which criterion the fit's log-likelihood is.
- `anova_test`: which per-term tables the fit stores for `anova()`. Every fit
  stores its Wald or F tables. `"auto"` (default) also stores the
  likelihood-ratio tables of a Poisson model with random effects, the rows
  `anova(fit)` gives it by default; `"LRT"` stores them on every fit with
  chi-squared rows, so that `anova(fit, test = "LRT")` can read them;
  `"Wald"` stores none. A likelihood-ratio table costs one refit per reduced
  model, made after the fit; the fit itself is the same under every value.
- `start` — warm start; see §4.
- `init.theta` — negative-binomial shape seed, named for
  `MASS::glm.nb(init.theta=)`: the starting value of the shape search, the
  same slot as `start$dispersion` (give one of the two). **Distinct** from
  `start$theta` (the random-effect Cholesky start): unrelated knobs sharing a
  Greek letter.

**Error vs. warning.** An *invalid* value — unknown family, a link the family
doesn't offer, a non-odd or out-of-range `nAGQ`, a `start` that isn't a list —
**errors**. A *valid but inapplicable* option (`dispersion=` on gaussian,
`init.theta=` off negative-binomial, quasi-dispersion on a mixed
binomial/Poisson formula, `REML = TRUE` off the Gaussian family) **warns and
is stripped**, so it never reaches the kernel: loud enough to catch the
mistake, lenient enough for exploration.

Three things error *naming the reason* rather than fitting a silently different
model. Known lme4 arguments passed through `...` (`control=`, `verbose=`,
`contrasts=`) each explain why they can't be honored. A random-effect term
on `family = inverse.gaussian()` errors — the kernel is GLM-only for that
family — and the remaining open gap, quasi-likelihood
dispersion on binomial/Poisson, errors until the kernel implements it.
(`init.theta=` off negative-binomial is the harmless warn-and-strip case
above.)

## 3. Reading the result

The returned object is class `"fastglmm"`, read through accessors named and
shaped like lme4's. These **work**:

| accessor | what it returns |
|---|---|
| `summary(fit)` | lme4's blocks in lme4's order: the criterion (`REML criterion at convergence` on a REML fit, an `AIC BIC logLik deviance df.resid` row on an ML fit; an LMM fitted with `REML = FALSE` is titled "Linear mixed model fit by maximum likelihood" and prints "Note: p-values can be somewhat anticonservative." under the title), scaled residuals, the RE block, the coefficient table, the correlation of fixed effects, and a dispersion/shape footer. The coefficient table follows lmerTest, `lm` and `glm`: on a Gaussian mixed model `Estimate`, `Std. Error`, `df` (Satterthwaite), `t value`, `Pr(>|t|)`; on OLS and the Gamma and inverse-Gaussian GLMs with an estimated dispersion, t on the residual df `n - p`, printed in the footer (on OLS with `summary.lm`'s residual standard error and R-squared lines); everywhere else Wald **z** and `Pr(>|z|)`. When the fit raised any warnings, the printed summary ends with a `Warnings:` section listing each one's tier, title and message. |
| `fixef(fit)` | named fixed-effect estimates; aliased (rank-deficient) columns are `NA`, as in `lm`/lme4. |
| `vcov(fit)` | full `p × p` Wald covariance of β̂. |
| `VarCorr(fit)` | variance components on the **SD/correlation** scale, one covariance per grouping, lme4-shaped; a `Residual` row (= `sigma()`) is printed for a gaussian mixed fit. Groups are listed in the order they appear in the formula — lme4's own `VarCorr` instead sorts by decreasing number of levels. |
| `confint(fit)` | Wald intervals off `vcov()` that follow `summary()` row by row: the t quantile on the row's df where the summary tests with t (Satterthwaite df on a Gaussian mixed model, `n - p` as `confint.lm` on OLS and the dispersion GLMs), the normal quantile where it tests with z. `parm` and `level` as in `stats::confint`. The Python port's `fit.confint()` gives the same intervals. `method = "profile"`/`"boot"` are not available and say so. |
| `isSingular(fit)` | boundary-fit flag — lme4's condition, computed by the kernel. |
| `sigma(fit)` | residual SD for gaussian fits, `sqrt(phi)` for Gamma/inverse-Gaussian; `1` for binomial/Poisson/negative-binomial (fixed scale, as in lme4). |
| `nobs`, `formula`, `family`, `model.frame`, `print` | the usual; `formula()` returns the model formula as given, as a formula object, so `update(m, . ~ . + z)` works (the parser is Rust-side, so there is no R `terms` object to hand back). |
| `ranef(fit)` | conditional modes (BLUPs), one data frame per grouping, lme4-shaped. With `condVar = TRUE` (the default) each data frame carries lme4's `postVar` attribute, terms × terms × levels; a fit on the sparse route has none. |
| `predict(fit, newdata)` | predictions on new rows (the training rows without `newdata`), on the link scale by default; `re.form = NA` for the population prediction, `se.fit = TRUE` for standard errors and intervals. A level the fit did not see gets a random effect of 0 and a `fastglmm_new_grouping_levels` warning. See [`coming-from-lme4.md`](coming-from-lme4.md). |
| `simulate(fit, nsim, seed)` | `nsim` simulated responses on the rows the fit used, one column each; the same `seed` gives the same draws on every platform. |
| `fitted(fit)` | conditional means per row, including the random-effect contribution and any offset. |
| `residuals(fit)` | `type = "response"` (default, `y - fitted`) or `type = "pearson"`. Deviance and working residuals are not offered. |
| `tidy(fit)`, `glance(fit)` | broom-shaped data frames: one row per fixed effect (`term`, `estimate`, `std.error`, `statistic`, then `df` where `summary()` has a df column, then `p.value`; the statistic and p-value are the summary's, t or z), and one row of fit statistics (`nobs`, `logLik`, `AIC`, `BIC`, `deviance`, `df.residual`, `REML`). |
| `fit$coef_df`, `fit$coef_p`, `fit$df_method` | per coefficient, the df of its t test (Satterthwaite on a Gaussian mixed model, `n - p` on OLS and the dispersion GLMs, `NaN` where the test is z and on an OLS or dispersion-GLM fit that did not converge) and that test's p-value; `df_method` says which (`"satterthwaite"`, `"residual"`, `"none"`, `"disabled"`, `"failed"`). |
| `fit$rss`, `fit$r_squared`, `fit$adj_r_squared` | OLS only: the weighted residual sum of squares and `summary.lm`'s multiple and adjusted R-squared (0 on an intercept-only model); `NaN` on every other fit and on a non-converged one. |
| `logLik(fit)` | log-likelihood on `stats::logLik`'s scale; on a REML fit (a Gaussian mixed model with the default `REML`, or OLS with `REML = TRUE`) this is the REML criterion (`REML = TRUE` attribute), on every other fit the ML log-likelihood (`REML = FALSE`). |
| `AIC(fit)`, `BIC(fit)` | `stats`'s values and tables, read from `logLik()`. Two comparisons stop with an error instead: a REML fit beside an ML fit (the error names which is which), and REML fits with different fixed effects, whose REML criteria are not comparable (the error points to `compare()` and to `REML = FALSE`). ML fits compare across fixed effects as in `stats`. See `?AIC.fastglmm` for the uses it cannot check. |
| `anova(fit)` | the per-term tests, one row per fixed term, built when the model was fitted (no refit when called): Type III by default, `type = "II"` on request; lmerTest's F with Satterthwaite df on a Gaussian mixed model, F on `n - p` on OLS and the Gamma and inverse-Gaussian GLMs with an estimated dispersion, likelihood-ratio chi-squared on a Poisson GLMM, Wald chi-squared elsewhere; `test = "Wald"` or `test = "LRT"` picks the test on a fit with chi-squared rows. See the end of this section. |
| `anova(m0, m1)` | two fits compared as they are, with no refit; see the end of this section. |
| `pairwise(fit, "f")` | every pairwise comparison of the levels of a factor of the fixed part, emmeans' `pairs(emmeans(fit, ~ f))` with Holm's adjustment by default (emmeans' default is Tukey); `confint()` of the table gives the intervals. See the end of this section. |
| `r2(fit)`, `icc(fit)` | a mixed model's marginal and conditional R-squared and its adjusted and unadjusted ICC, as `performance::r2_nakagawa` and `performance::icc` compute them. They are computed when called; on a Poisson or negative binomial GLMM each call refits the null model once. `summary()` prints the R-squared pair and the adjusted ICC in one line, or, on a Poisson or negative binomial GLMM, a line that points to `r2()` and `icc()`. |
| `fit$offset` | the offset the model was fitted with, `NULL` without one. |
| `fit$diagnostics` | list with `converged`, `singular`, `aliased`, `boundary`, `pinned`, `notes` — the solver's own report; the top-level `fit$converged`/`fit$singular`/`fit$aliased` are unchanged and mirror the same values. |

On the Gaussian mixed model of §1:

```r
summary(fit)                  # Estimate, Std. Error, df, t value, Pr(>|t|)
fit$coef_df                   # the df column, named by coefficient
confint(fit)                  # 2.5 % and 97.5 % bounds on t quantiles
confint(fit, "x1", level = 0.9)
tidy(fit)                     # term, estimate, std.error, statistic, df, p.value
```

**Diagnostic conditions.** `fit` raises a warning for anything worth knowing
about the fit (a kernel note such as ill-conditioning, an ignored argument,
or a non-convergence), each a classed condition inheriting
`"fastglmm_diagnostic"`, so one handler catches the whole channel. The same
information is also stored as a row (`tier`, `kind`, `title`, `message`) in
`fit$warnings`, a data frame with one row per warning, in the order raised.
`warnings.md` lists every kind; match on `kind` in scripts, since the title
and message text can change between versions. A design that is merely
ill-conditioned (near-collinear but still distinguishable) is fitted and
returns real numbers; the warning is how you find out its standard errors are
honest but large. Select on the class rather than matching message text:

```r
withCallingHandlers(
  fit <- fastglmm(y ~ x1 + x2 + (1 | group), data),
  fastglmm_diagnostic = function(w) invokeRestart("muffleWarning")
)

# or, to just silence the channel:
suppressWarnings(fastglmm(y ~ x1 + x2 + (1 | group), data), classes = "fastglmm_diagnostic")
```

These **error, naming the reason** — the package's defining choice, that anything
the kernel cannot compute honestly is a documented error, never a silently
different answer: `coef` and `terms`. `coef()` is
instructive: lme4's `coef()` means fixed + random effects per group, and
returning fixed effects only would silently differ from what the same call
gives an lme4 user, so it errors and points you at `fixef()` and `ranef()`
instead.

**`vcov` and `se` carry different information.** The `Std. Error` column is just
the square root of `diag(vcov(fit))`, so it cannot answer anything about two
coefficients jointly. A contrast like β₁ − β₂ needs
`Var(β₁) + Var(β₂) − 2·Cov(β₁, β₂)`, and that covariance lives only in `vcov()`.

To work with a grouping's variance components numerically rather than as printed
text, pull the attributes `VarCorr()` hangs on each block:

```r
vc <- VarCorr(fit)
attr(vc$group, "stddev")       # per-term random-effect standard deviations
attr(vc$group, "correlation")  # their correlation matrix
```

**Tests: `anova()` and `compare()`.** `anova(fit)` tests each fixed term of
one fit, from the table the fit stored when it was made. Two models are
compared in one of two ways:

```r
# Fixed-term tests of one fit (Type III unless type = "II").
anova(fit)

# Two formulas: compare() fits both on the same rows, picking the estimator.
cmp <- compare(data, y ~ x1, y ~ x1 + (1 | group))
cmp            # the comparison table
cmp$fit1       # the fit of the second formula

# Two fits you already have, compared as they are (no refit).
m0 <- fastglmm(y ~ x1, data, REML = TRUE)   # REML = TRUE: see below
m1 <- fastglmm(y ~ x1 + (1 | group), data)
anova(m0, m1)
# On this toy data the group variance is estimated at exactly 0, so the
# statistic is 0 and p = 1 (no draws are needed); hypothesis-tests.md shows a
# comparison on sleepstudy.
```

On a fit with chi-squared rows, `anova(fit, test = "Wald")` or
`anova(fit, test = "LRT")` picks the test of the rows. The default,
`test = "auto"`, gives a Poisson model with random effects the
likelihood-ratio rows and every other fit its Wald or F rows. The
likelihood-ratio rows are refitted when the model is fitted, by default only
on a Poisson GLMM; on any other fit, ask for them with
`fastglmm(..., anova_test = "LRT")` first.

`compare()` has no `REML` argument: on a Gaussian model it fits both models
by REML when only their random parts differ and by ML when their fixed parts
differ. `anova(m0, m1)` never refits, so the two fits must be comparable as
they stand: two REML fits with different fixed parts are an error there
(lme4 refits them by ML; use `compare()`, or refit both with `REML = FALSE`),
and a REML mixed fit compares with an OLS fit only when that one was made with
`REML = TRUE`. A test of a variance component uses a reference that respects
the boundary at zero: the exact restricted likelihood-ratio test for a
Gaussian model's single random intercept, otherwise the 50:50 mixture of
chi-squared distributions,
which is conservative. Both calls take `nsim` and `seed` for the exact test;
the default seed is fixed, so a call gives the same p-value every time and
`set.seed()` does not change it. The full rules, every reference and every
refusal are in [`hypothesis-tests.md`](hypothesis-tests.md).

**Pairwise comparisons: `pairwise()`.** On a model with a factor `f` in its
fixed part, `pairwise(fit, "f")` compares every pair of its levels, as
emmeans' `pairs(emmeans(fit, ~ f))`, and returns a data frame with emmeans'
columns. Nothing is refitted. The default adjustment is Holm; emmeans'
default is Tukey, so pass `adjust = "tukey"` to get emmeans' p-values:

```r
pw <- pairwise(fit, "f")                  # Holm
pairwise(fit, "f", adjust = "tukey")      # emmeans' default
pairwise(fit, "f", by = "h")              # within each level of h
confint(pw, level = 0.95)                 # lower.CL, upper.CL
```

## 4. Warm starts

If you already hold optimizer state from a previous fit of the same model shape,
`start` seeds the next one — it changes how fast the optimizer converges, never
the answer:

```r
fit2 <- fastglmm(y ~ x1 + (1 | group), data2,
                 start = list(beta = fixef(fit1), theta = theta_prev))
```

`start` takes lme4's name and shape: a list with `beta` (length `p`) and `theta`
(the random-effect **Cholesky** vector, `lme4::getME(fit, "theta")`'s layout;
empty for a fixed-only model, where a warm start is a no-op anyway), plus an
optional `dispersion`: a previous fit's `fit$dispersion` (the negative-binomial
shape or the Gamma φ), a starting value for the search, never a held value
(`dispersion=` holds φ fixed). Any element may be left out to cold-start that
part. Unknown list elements warn and are dropped. `start$theta` is unrelated to
`init.theta` (§2), the negative-binomial shape seed — the two just share the
letter, which is why the RE start is `start$theta`; the shape seed is
`init.theta` or `start$dispersion` (give one of the two).
