# Using `glmm` from Python

One page, four sections. The first walks the single entry point — `glmm.fit` —
end to end; the next two go deeper into the knobs and the returned `Fit`; the
fourth is a short note on warm starts. The Python surface is deliberately
tiny: `glmm.fit`, `glmm.Fit`, the tests `glmm.anova` and `glmm.compare` with
their result classes (§3), and a family of warning categories, all
subclasses of `glmm.DiagnosticWarning`, that `fit` can raise. Every warning it
raises is also appended to `fit.warnings`; see
[`warnings.md`](warnings.md) for the full list of kinds, tiers and messages
(that page is the single source, not this one). Everything else (families,
links, knobs) is a string or scalar argument, not a type.

> **Status:** this release ships the full API surface — signatures, argument
> validation, `Fit`, `summary()` — wired end to end through the PyO3 binding:
> a valid `fit(...)` call parses the formula, fits, and returns a real `Fit`.
> One narrow combination is an open gap and raises a clean
> `NotImplementedError` instead: quasi-likelihood `dispersion=` on
> binomial/poisson.

Install from PyPI:

```bash
pip install glmm
```

(For the Rust crate this package ports, see [`tutorial-rust.md`](tutorial-rust.md)
— same models, hand-built inputs instead of a formula — and for the R port,
[`tutorial-r.md`](tutorial-r.md).)

## 1. One call — formula in, `Fit` out

`glmm.fit` is the only entry. You hand it a data table and an R-style formula;
it parses the formula against the table's columns, builds the design matrix,
fits, and returns a `Fit`. There is no model object to construct and no
`n`/`p` to pass — everything is inferred from the formula and the data.

```python
import glmm

data = {
    "y":     [1.02, 1.05, 1.11, 1.13, 1.21, 1.24, 1.30, 1.33, 1.42, 1.44, 1.51, 1.53],
    "x1":    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    "s":     [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
    "group": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
}

# y ~ x1 + (1 | group) — Gaussian, random intercept.
fit = glmm.fit(data, "y ~ x1 + (1 | group)")

print(fit.converged)
fit.summary()   # prints the coefficient table (and returns it as a string)
```

Points worth knowing at this layer:

- `data` is a `dict[str, array-like]` (the documented form), but anything
  column-addressable works — pandas / polars DataFrame, pyarrow Table —
  duck-typed, no dataframe dependency. Response and fixed-effect columns are
  read as 1-D float64; grouping columns (the `g` in `(1 + x | g)`) accept
  string / integer / categorical values and are factorized to level ids.
- The default `family="gaussian"` with no `(… | g)` term fits OLS; adding a
  random-effect term makes it an LMM (REML by default, maximum likelihood with
  `reml=False`). The same split holds for every family: fixed-only ⇒ GLM,
  `(… | g)` present ⇒ GLMM.
- The formula follows R conventions: `*` desugars to main effects +
  interaction, `A/B` to nesting, `(1 + x | g)` is a correlated random
  intercept + slope, and additional `(… | g2)` terms add crossed/nested
  grouping factors. Treatment contrasts (R's default) code the factors, based
  on the column's **first level** — a `pandas.Categorical` uses the first
  category you declare; a plain string column declares no order and is sorted
  lexicographically, as R's `factor()` does.
- Misspell a keyword and you get a `TypeError` at the call site — the
  signature is explicit keywords, no options bag, so typos can't become
  silent no-ops.

## 2. Families and knobs

`family` is a string; `link=None` resolves to the family's default. Pass a
string only where the kernel offers a choice:

| `family` | default link | other links | distribution params |
|---|---|---|---|
| `gaussian` | identity | — | — |
| `binomial` | `logit` | `probit`, `cloglog` | — |
| `poisson` | `log` | — | — |
| `gamma` | `log` | `inverse` | `dispersion` |
| `negativebinomial` | `log` | — | `theta` |
| `inversegaussian` | `log` | `inverse_squared` (fixed-effect GLM only — mixed models raise `ValueError`) | — |

```python
fit = glmm.fit(data, "s ~ x1 + (1 | group)", "binomial", link="probit", nagq=7)
```

- `dispersion` — three states. `None` (default): gamma estimates φ̂ by
  maximum likelihood on both a GLM and a GLMM, to place `fit.loglik` at its
  maximum, but the two routes report and scale the SE by different numbers:
  a GLM's `fit.dispersion` is the Pearson moment `Σ wᵢrᵢ²/(n−p)`
  (`summary(glm)`'s convention), a GLMM's is that ML φ̂ itself.
  Inverse-Gaussian (GLM only) estimates φ̂ by the Pearson moment and scales
  its SE by it; other families hold φ ≡ 1. `"estimate"`:
  the same estimate as `None` on gamma and inverse-Gaussian; on
  binomial/poisson it would force a Pearson estimate — this *is*
  quasi-binomial/quasi-Poisson, GLM only. A float: hold φ fixed (still
  scales SE). Fix-vs-estimate, not a warm start.
- `nagq` — adaptive Gauss–Hermite node count; `1` = Laplace (default). Must
  be odd and ≤ 25. `>1` applies to binomial, Poisson, negative-binomial or
  Gamma models with one grouping factor and ≤ 3 random effects per group
  (intercept + slopes, temporary cap); any other shape warns and falls back
  to Laplace.
- `wald_se` — fixed-effect Wald-SE denominator: `"hessian"` (default) or
  `"rx"`.
- `init_theta` — negative-binomial shape seed, named for
  `MASS::glm.nb(init.theta=)`: the starting value of the shape search, the
  same slot as `warm_start["dispersion"]` (give one of the two). `None`
  (default) cold-starts it from the no-random-effect negative-binomial fit.
  Estimation always runs. Distinct from `warm_start["theta"]`, the
  random-effect Cholesky start (§4): unrelated knobs that happen to share a
  Greek letter, which is why this one is not just called `theta`. Both may be
  passed in one call.
- `weights` — per-row prior weights, lme4's `weights=`. On gaussian, gamma
  and inverse-gaussian these are precision weights: row `i` gets dispersion
  `φ/wᵢ`, the same convention as `lm`, `summary(glm)` and lme4, not
  glmmTMB's, which multiplies each row's log-density by `wᵢ` instead (see
  [`conventions.md`](conventions.md#prior-weights)). For an aggregated
  binomial, `y` is the success *proportion* and `weights` the trial count
  (lme4's `cbind(s, m−s)`).
- `offset` — per-row known additive term on the linear-predictor scale, R's
  `offset=`: `eta = offset + X*beta (+ Z*b)`, with no coefficient estimated
  for it and no column added to the design. The canonical use is a Poisson
  rate model against a known exposure, `offset = np.log(exposure)`. Honored
  for every family and every solver path; `None` (default) means no offset.
  The formula also accepts an `offset(expr)` term (see
  [`formula.md`](formula.md)); passing both this argument and a formula
  `offset()` term is an error asking you to use one.
- `satterthwaite`: `True` (default) computes lmerTest's Satterthwaite degrees
  of freedom for a Gaussian mixed model's t tests, in one pass after the fit.
  `False` skips that pass: the summary then tests with z and says "df not
  computed (satterthwaite=False)", `fit.confint()` gives normal intervals, and
  `fit.df_method` is `"disabled"`. OLS and the gamma and inverse-gaussian GLMs
  keep their residual df `n − p` whatever it says; no other family reads it.
- `reml`: `None` (default) or `True` fits a Gaussian mixed model by REML,
  `False` by maximum likelihood, as lme4's `REML = FALSE`. The summary of an
  ML fit is titled "Linear mixed model fit by maximum likelihood [glmm]",
  prints the `AIC BIC logLik deviance df.resid` row in place of the REML
  criterion, and prints one line under the title: "Note: p-values can be
  somewhat anticonservative." On OLS (also when every grouping was dropped as
  single-level) the estimates are the same under every value: `True` reports
  the REML log-likelihood, as R's `logLik(lm, REML = TRUE)`, and `None` and
  `False` the ML one. Every other family is fitted by maximum likelihood;
  `False` there changes nothing, and `True` is ignored with an
  `ArgumentIgnoredWarning`. `fit.reml` says which criterion `fit.loglik` is.

**Error vs. warning.** An *invalid* value — unknown family, a link the family
doesn't offer, even `nagq` (`ValueError`), a non-dict `warm_start`
(`TypeError`) — raises. A *valid but inapplicable* option (`dispersion=` on
gaussian, `init_theta=` off negative-binomial, quasi-dispersion on a mixed
binomial/poisson formula, `reml=True` off the Gaussian family) warns
(`UserWarning`) and is stripped, so it never reaches the kernel: loud enough to catch the mistake, lenient enough for
exploration. `family="inversegaussian"` with a random-effect term raises
`ValueError` before the call reaches the kernel — mixed models are fixed-effect
GLM only for this family (see status above).

## 3. Reading the result — `Fit`

`Fit` mirrors the Rust `Fit`, plus coefficient names (the formula supplies
them). It is returned by `fit`, never constructed by callers.

| field | what it holds |
|---|---|
| `beta` | `(p,)` fixed-effect estimates |
| `se` | `(p,)` standard errors; `NaN` where unavailable |
| `coef_df` | `(p,)` degrees of freedom of each coefficient's t test: lmerTest's Satterthwaite df on a Gaussian mixed model, `n − p` on OLS and on a gamma or inverse-gaussian GLM with φ estimated; `NaN` where the test is z, on aliased columns, and on an OLS or dispersion-GLM fit that did not converge. Not `df`, which is the AIC parameter count |
| `coef_p` | `(p,)` two-sided p-value of that t test, computed by the kernel; `NaN` where there is no df |
| `df_method` | where `coef_df` comes from: `"satterthwaite"`, `"residual"` (OLS and the dispersion GLMs), `"none"` (the families with no df: z), `"disabled"` (a Gaussian mixed model fitted with `satterthwaite=False`: z) or `"failed"` (its Satterthwaite pass failed: z, with a `SatterthwaiteCurvatureWarning`) |
| `rss`, `r_squared`, `adj_r_squared` | OLS only: the weighted residual sum of squares and `summary.lm`'s multiple and adjusted R² (0 on an intercept-only model); `NaN` on every other fit and on a non-converged one |
| `vcov` | `(p, p)` full Cov(β̂) — `se` is the sqrt of its diagonal; use it for contrasts/confidence intervals, where the off-diagonals matter |
| `names` | coefficient names, aligned with `beta` |
| `aliased` | `(p,)` bool — rank-deficient columns dropped (lme4's `NA` coefficients) |
| `varcorr` | per grouping, in the order the groupings appear in the formula (lme4 instead sorts by decreasing number of levels): vech-packed lower-triangular RE covariance D̂ |
| `tau2` | legacy per-element RE variances (q=1 only) — prefer `varcorr` |
| `stddev_se` | SE of each RE stddev, θ layout (not beta-aligned): at the stddev's diagonal slot, `NaN` at the off-diagonal slots and where unavailable; not printed by `summary()` |
| `dispersion` | φ (gamma / inverse-gaussian) / θ (negbin) / residual σ² (gaussian) / 1.0 (binomial, poisson) |
| `re_groups` | per grouping, in `varcorr` order (formula order): `(name, [term names])` — what `summary()` labels the RE block with |
| `n_eval` | optimizer objective evaluations (0 on the closed-form/IRLS paths) |
| `n_grad`, `outer_stop`, `outer_decrement` | the gradient (BFGS) search, on the models `GradientSearchFallbackWarning` names: its gradient evaluations (not in `n_eval`), why it stopped (`"fdelta"`, `"pgrad"`, `"noise_floor"`, `"line_search"`, `"max_iter"`, `"grad_unavailable"`, `"non_finite"`), and its estimate of the deviance decrease left at the endpoint; `0` / `None` / `NaN` on every other fit |
| `deviance` | minimized optimizer criterion — **not** comparable across models, and not an AIC input (see below) |
| `loglik` | log-likelihood on `logLik()`'s scale (R/lme4); the REML criterion on a REML fit (see `reml`), the ordinary log-likelihood on every other fit; `NaN` wherever `deviance`'s failure modes apply |
| `df` | parameters counted for AIC/BIC: retained fixed effects + RE parameters + 1 if the family estimates a dispersion/scale; `0` on degenerate NaN-fill paths |
| `reml` | `True` iff `loglik` is a REML criterion rather than an ML log-likelihood: a Gaussian mixed model fitted with `reml=None` or `True`, or OLS fitted with `reml=True`. AIC/LRT comparisons across fits with different fixed effects are invalid when this is set |
| `aic`, `bic` | properties, computed on every fit from `loglik`, `df` and `nobs`: `-2·loglik + 2·df` and `-2·loglik + df·ln(nobs)`. On a REML fit they compare only fits with the same fixed effects (see below) |
| `fitted` | `(n,)` fitted means μ̂ per row; empty on non-converged fits |
| `ranef` | random-effect conditional modes b̂, one block per grouping in `varcorr`/`re_groups` order, level-major; empty on non-converged fits — prefer `ranef_blocks` for the labelled form, and do not slice this array yourself |
| `ranef_levels` | level count per grouping, for slicing `ranef`; empty exactly when `ranef` is |
| `ranef_blocks` | the same conditional modes, labelled: a list of dicts per grouping (`group`, `terms`, `levels`, `values`); empty exactly when `ranef` is |
| `converged` | numerical failure signals here (not an exception) — check before trusting `beta`/`se` |
| `singular` | boundary (singular) fit — `>=1` RE variance component pinned at 0; mirrors lme4's `isSingular` |
| `offset` | `(n,)` the offset the model was fitted with; `None` without one |
| `objective`, `objective_nodes` | what the fit maximised: `"least_squares"` (OLS), `"maximum_likelihood"` (a GLM, or a Gaussian mixed model fitted by ML), `"reml"`, `"laplace"` or `"agq"`, and the quadrature node count (`0` off a GLMM); `glmm.anova(m0, m1)` labels its test with it |
| `term_table_type3`, `term_table_type2`, `design_summary` | what `glmm.anova` reads: the per-term tables built when the fit was made, and the design summary the exact random-intercept test needs (`None` off its envelope); read them through `glmm.anova` |
| `dispersion_held` | the number passed as `dispersion=` (gamma / inverse-gaussian only); `None` when φ was estimated or the argument does not apply. When it is set, `summary()` labels the dispersion "fixed" |
| `diagnostics` | dict with `converged`, `singular`, `aliased`, `boundary`, `pinned`, `notes` — the solver's own report; `converged`/`singular`/`aliased` above are `@property` forwarders over `diagnostics[...]`, kept at the top level for the most-read fields |

**Warnings.** `fit` raises a warning for anything worth knowing about the fit
(a kernel note such as ill-conditioning, an ignored argument, or a
non-convergence), each as its own subclass of `DiagnosticWarning`, and stores
the same information as a dict (`tier`, `kind`, `title`, `message`) in
`fit.warnings`, in the order raised. `warnings.md` lists every kind; match on
`kind` in scripts, since the title and message text can change between
versions. A design that is merely ill-conditioned (near-collinear but still
distinguishable in f64) is fitted and returns real numbers; the warning is how
you find out its standard errors are honest but large. Filter the whole
channel, or just one category:

```python
import warnings

warnings.filterwarnings("ignore", category=glmm.DiagnosticWarning)      # all diagnostics
warnings.filterwarnings("ignore", category=glmm.IllConditionedWarning)  # just this one
```

**`deviance` is not a model-comparison statistic.** It is the criterion the
optimizer minimized, on that fit's own scale: for an LMM it is lme4's
`REMLcrit` (on an ML fit, lme4's ML deviance) minus a data-independent
constant; for a GLMM it is the marginal
Laplace deviance, which differs from −2·logLik by a data-only saturated
constant on binomial/Poisson fits and equals −2·logLik on Gamma and
negative-binomial fits. Those constants do not cancel between two different
models, so differencing `deviance` across fits — or feeding it to an AIC — is a mistake.
It is `NaN` for OLS/GLM and on numerical failure.

**`aic` and `bic` of a REML fit.** They are computed on every fit, and on a
REML fit they are built on the REML criterion, as lme4's `AIC()` of a REML
fit is. The REML criterion depends on the fixed-effects design, so two REML
fits compare by `aic` or `bic` only when their fixed effects are the same
(models that differ in their random effects alone). To compare fixed effects,
fit both models with `reml=False`. The `aic` property cannot see the other
fit, so nothing checks this for you there (R's `AIC()` refuses such a pair);
`glmm.anova(m0, m1)` does check, and refuses two REML fits with different
fixed effects.
`summary()` of a REML fit prints the REML criterion and no AIC for the same
reason. On a fit that did not converge the value is not an AIC; check
`converged` first.

**`vcov` and `se` carry different information.** `se` is only the diagonal, so
it cannot answer anything about two coefficients jointly. A contrast like
β₁ − β₂ needs `Var(β₁) + Var(β₂) − 2·Cov(β₁, β₂)`, and that covariance lives
only in `vcov`. Both are `NaN` in the same places.

`summary()` builds the coefficient table, prints it, and returns it as a
string. Its test follows lmerTest, `lm` and `glm`:

- **Gaussian mixed model:** `Estimate`, `Std. Error`, `df`, `t value`,
  `Pr(>|t|)`, with each coefficient's Satterthwaite df (`coef_df`) and the
  kernel's p (`coef_p`), on lmerTest's layout. The footer says "t tests use
  Satterthwaite degrees of freedom (finite-difference θ-Hessian).".
- **OLS, and gamma or inverse-gaussian GLMs with φ estimated:** `t value` and
  `Pr(>|t|)` on the residual df `n − p`, printed in the footer (on OLS as
  `summary.lm`'s "Residual standard error: … on n − p degrees of freedom" and
  its R² line).
- **Everything else** (binomial, Poisson, negative binomial, a held
  dispersion, every GLMM): `z value` and `Pr(>|z|)`, a Wald test on the
  normal, as `glm` and `glmer` print. A Gaussian mixed model fitted with
  `satterthwaite=False`, or one whose pass failed, is tested with z too; the
  first says "df not computed (satterthwaite=False)" in the footer.

```python
fit = glmm.fit(data, "y ~ x1 + (1 | group)")
fit.summary()     # Estimate, Std. Error, df, t value, Pr(>|t|)
fit.coef_df       # the df column, aligned with fit.names
fit.coef_p        # the Pr(>|t|) column

fit.confint()            # (p, 2) array of 95% bounds, rows in fit.names order
fit.confint(level=0.9)   # any level strictly between 0 and 1
```

`confint(level=0.95)` follows `summary()` coefficient by coefficient: the t
quantile on the row's `coef_df` where the summary tests with t, the normal
quantile where it tests with z. It returns a plain `(p, 2)` array (lower,
upper) in the order of `names`, with no `parm` argument; index the rows you
want. Aliased coefficients give `NaN` rows. The R port's `confint()` computes
the same intervals through the same kernel function.

Aliased columns show `NaN` estimates, as lme4 prints `NA`. A footer carries
`dispersion`, `converged` and `singular`, and when `varcorr` is non-empty an
RE block shows each grouping by name with its per-term variance, stddev and
correlation (lme4's `VarCorr` layout; the names come from `re_groups`).
`stddev_se` is not printed; read it off the fit. When the fit raised any
warnings, the text summary ends with a `Warnings:` section listing each one's
tier, title and message; the HTML, LaTeX and Typst renderers do not include
it.

To work with a grouping's covariance numerically rather than as printed text,
`stddev_corr(group_idx)` splits its vech-packed block into a `(q,)` stddev
vector and a `(q, q)` correlation matrix:

```python
sd, corr = fit.stddev_corr(0)   # grouping 0: stddevs + correlation matrix
```

**Tests: `glmm.anova` and `glmm.compare`.** `glmm.anova(fit)` tests each
fixed term of one fit, from the table the fit stored when it was made, and
returns a `TermTable`. Two models are compared in one of two ways:

```python
# Fixed-term tests of one fit (Type III unless type="II").
glmm.anova(fit)

# Two formulas: glmm.compare fits both on the same rows, picking the estimator.
cmp = glmm.compare(data, "y ~ x1", "y ~ x1 + (1 | group)")
cmp.table                  # a ComparisonTable; cmp.table["Pr(>Chisq)"]
cmp.fit1                   # the Fit of the second formula

# Two fits you already have, compared as they are (no refit).
m0 = glmm.fit(data, "y ~ x1", reml=True)   # reml=True: see below
m1 = glmm.fit(data, "y ~ x1 + (1 | group)")
glmm.anova(m0, m1)
# On this toy data the group variance is estimated at exactly 0, so the
# statistic is 0 and p = 1 (no draws are needed); hypothesis-tests.md shows a
# comparison on sleepstudy.
```

Both tables are indexed by R's column names, print as text, render to
`html()`, `latex()` and `typst()`, and convert with `to_pandas()` (pandas is
imported only then). `glmm.compare` has no `reml` argument: on a Gaussian
model it fits both models by REML when only their random parts differ and by
ML when their fixed parts differ. `glmm.anova(m0, m1)` never refits, so the
two fits must be comparable as they stand: two REML fits with different fixed
parts raise `ValueError` (lme4 refits them by ML; use `glmm.compare`, or
refit both with `reml=False`), and a REML mixed fit compares with an OLS fit
only when that one was made with `reml=True`. A test of a variance component
uses a reference that respects the boundary at zero: the exact restricted
likelihood-ratio test for a Gaussian model's single random intercept,
otherwise the 50:50 mixture of chi-squared distributions, which is
conservative. Both calls take `nsim`
and `seed` for the exact test; the default seed is fixed, so a call gives the
same p-value every time. The full rules, every reference and every refusal
are in [`hypothesis-tests.md`](hypothesis-tests.md).

## 4. Warm starts

If you already hold optimizer state from a previous fit of the same model
shape, `warm_start` seeds the next one — it changes how fast the optimizer
converges, never the answer:

```python
fit2 = glmm.fit(data2, "y ~ x1 + (1 | group)",
                warm_start={"beta": list(fit1.beta), "theta": [...]})
```

The dict takes up to three keys, mirroring the Rust `StartValues`: `"beta"`
(length p), `"theta"` (Cholesky-scaled RE start; empty for fixed-only
models, where a warm start is a no-op anyway) and `"dispersion"` (a previous
fit's `Fit.dispersion`: the negative-binomial shape or the Gamma φ, a starting
value for the search, never a held value; `dispersion=` holds φ fixed).
Any key may be left out to cold-start that part. Unknown keys warn and are
dropped. For negative binomial `"dispersion"` and the `init_theta=` kwarg
(§2) are the same seed; give one of the two. `warm_start` is always explicit, caller-owned —
a previous fit's state is never threaded forward automatically.

**Batch processing** (many fits with same formula behind one call — bootstrap,
simulation) is planned.
