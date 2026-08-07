# Coming from statsmodels

Python-only: nobody moves from `statsmodels` to R for this, so this page
doesn't try to cover `fastglmm`. If you're coming from lme4 instead — R or
Python — see [`coming-from-lme4.md`](coming-from-lme4.md).

`statsmodels` gives you `MixedLM` (Gaussian linear mixed models only) and the
ordinary `GLM`/`GLMM`-adjacent estimators via `smf.glm`. `glmm.fit` covers
both from one call. The two libraries part ways in three places: the formula
grammar is not patsy, there is no construct-then-fit split, and the
attribute surface on the result is smaller — deliberately, and each gap below
was checked against the installed wheel rather than assumed.

## Formulas are not patsy

The formula string looks like patsy's — `~` separates response from
predictors, `+` adds terms — but the grammar underneath is a separate, much
smaller parser (`glmm::formula`, shared by Rust, Python and R) that accepts
**bare column names only**, never a function call. Every claim below is the
literal, verified output of `glmm.fit` on the installed wheel.

| patsy construct | Result | Workaround |
|---|---|---|
| `C(group)` | `ValueError: formula syntax error at position 0: expected identifier, got 'C(group)'` | Don't wrap it. A plain string (or `pandas.Categorical`) column is *always* treatment-coded — just write the bare column name, `y ~ group`. Control the base level by sorting (`pandas.Categorical(x, categories=[...])`); its first category is the base. See [`formula.md#factor-coding`](formula.md#factor-coding). |
| `np.log(x)` (or bare `log(x)`) | `ValueError: formula syntax error at position 0: expected identifier, got 'np.log(x2)'` (same shape for `log(x2)`: `got 'log(x2)'`) | No transforms inside the formula, dotted or not. Compute the column yourself — `data["log_x"] = np.log(data["x"])` — and reference `log_x` as a bare name. |
| `y ~ x - 1` (intercept suppression) | `ValueError: formula syntax error at position 0: expected identifier, got 'x-1'` | Not available at all, in either direction: the model always carries an intercept, and there is no way to remove it. Don't write `- 1` or `0 +`. |
| `bs(x, df=3)` (spline basis) | `ValueError: formula syntax error at position 0: expected identifier, got 'bs(x,df=3)'` | No basis-expansion support in the formula. Build the basis columns yourself outside the fit (e.g. `patsy.dmatrix("bs(x, df=3)", data)` or `scipy.interpolate`) and add each resulting column as its own bare fixed-effect term. |
| `x:y` interaction | **Works unchanged.** `glmm.fit(data, "y ~ x:x2")` fits — `:` means pure interaction (no main effects added) in both patsy and this parser, and `*` desugars to main effects + interaction in both too. | None needed. |

One more, not a patsy construct but the single most common first error for
anyone pasting a formula from an R or statsmodels tutorial: a bare `1` on the
fixed-effects side is also a parse error —
`glmm.fit(data, "y ~ 1 + x")` raises `expected identifier, got '1'`.
The intercept is always implicit; write `y ~ x`, never `y ~ 1 + x`.

See [`formula.md`](formula.md) for the full accepted/rejected grammar.

## Call mapping

| statsmodels | `glmm.fit` |
|---|---|
| `sm.MixedLM(endog, exog, groups=g).fit()` — construct the model from arrays, then fit it as a second step | `glmm.fit(data, "y ~ x + (1 \| g)")` — one call. There is no model object to build first: `fit` parses the formula, builds the design, and fits in the same step. |
| `smf.mixedlm("y ~ x", data, groups=data["g"]).fit()` — formula for the fixed effects, but grouping is still a separate `groups=` argument | `glmm.fit(data, "y ~ x + (1 \| g)")` — grouping is a formula term, `(1 \| g)`, not a side argument. Passing `groups=` to `glmm.fit` is a hard error: `TypeError: fit() got an unexpected keyword argument 'groups'` (verified). |
| `smf.mixedlm("y ~ x", data, groups=data["g"], re_formula="~x")` — a correlated random slope needs a second formula, `re_formula` | `glmm.fit(data, "y ~ x + (1 + x \| g)")` — the random-slope term lives in the same formula string as everything else. |
| `smf.glm("y ~ x", data, family=sm.families.Poisson()).fit()` — construct-then-fit again, family from `statsmodels.api.families` | `glmm.fit(data, "y ~ x", family="poisson")` — `family` is a string on the same call, from `{"gaussian", "binomial", "poisson", "gamma", "negativebinomial", "inversegaussian"}`. |
| no equivalent (`MixedLM` is Gaussian-only) | `glmm.fit(data, "y ~ x + (1 \| g)", family="binomial")` (or any other family) — a GLMM in the same call shape as the LMM above; nothing about the call changes except `family`. |

`glmm`'s Python surface is small on purpose: `glmm.fit`, `glmm.Fit` (the
result type — returned by `fit`, never constructed by the caller), and the
four warning categories the diagnostics channel raises
(`DiagnosticWarning`, `IllConditionedWarning`, `PirlsExhaustedWarning`,
`UnusedGroupingLevelsWarning`).

## `MixedLM` is REML by default too — but it *can* switch to ML, and glmm can't

It's tempting to assume `statsmodels.MixedLM` is an ML estimator and `glmm`'s
LMM path is REML, so the two never agree. Checked against the installed
wheel, that's not quite the shape of the difference. `MixedLM.fit()`
defaults to `reml=True` — the same objective `glmm.fit`'s LMM path always
uses. Fit recipe 1's `sleepstudy` model
(`smf.mixedlm("Reaction ~ Days", df, groups=df["Subject"], re_formula="~Days").fit()`,
no `reml=` argument, so the statsmodels default) and its log-likelihood is
`-871.8141359795748` — recipe 1's own REML criterion, printed in
[`examples-python.md`](examples-python.md#1-correlated-random-slope-sleepstudy),
is `-871.814135979251`. Same fixed effects too: `251.405`/`10.467` both
places. A bare, default `MixedLM` call is not the trap.

The trap is the opposite one: `MixedLM.fit(reml=False)` is a real, supported
switch to ordinary ML — and `glmm.fit` has no argument that reproduces it.
`glmm`'s LMM path is **profiled REML only, with no ML mode at all** (see
[`conventions.md`](conventions.md#estimation)). Code that was written
against, or copied from, a `reml=False` `MixedLM` fit will not match
`glmm.fit` on the variance components or the log-likelihood, and there is no
knob to make it match — REML is the only answer `glmm` gives.

One more thing this same check turned up: under its own REML default,
`MixedLM`'s `.aic` is `nan` (confirmed on the wheel above) — AIC isn't a
well-posed number under REML in general. The "no AIC line" in item 6 below
isn't `glmm` being unusually restrictive; it's the same restriction
`statsmodels` places on itself under the objective the two estimators share.

## What you lose

Checked against the installed `glmm.Fit` object on a real fit — not every
missing statsmodels name means the underlying number doesn't exist; some are
just named, or shaped, differently. Where that's the case, this says so
rather than lumping everything into one "blocked" bucket.

| statsmodels | `glmm.Fit` | What actually happens |
|---|---|---|
| `.random_effects` (dict of per-group BLUPs) | `.ranef` / `.ranef_levels` | `fit.random_effects` raises `AttributeError: 'Fit' object has no attribute 'random_effects'` (verified) — the name doesn't exist. The conditional modes themselves do, under `.ranef`/`.ranef_levels`, and are populated for GLM/GLMM fits — but come back as an **empty array** on the Gaussian LMM path, which is fit through sufficient statistics and keeps no per-row/per-group closed form. |
| `.fittedvalues` | `.fitted` | Same story: `AttributeError` on the statsmodels name (verified); `.fitted` holds real in-sample μ̂ per row for GLM/GLMM fits, empty for the Gaussian LMM path. |
| `.predict(new_data)` | — | `AttributeError: 'Fit' object has no attribute 'predict'` (verified) — no method under this or any other name. This is the one clean, total loss in this table: there is no out-of-sample prediction on any path, converged or not. |
| `.bse` | `.se` | Exists, just renamed — the `(p,)` array of Wald standard errors. |
| `.pvalues` | — | No stored field. `fit.summary()` computes Wald z/p on the fly (`z = beta/se`, `p = erfc(\|z\|/sqrt(2))`) and prints them, but doesn't save them anywhere on `Fit`; recompute the same two-line formula from `.beta`/`.se` if you need the numbers back. |
| `.aic` | — | No stored field, and (per the section above) not really comparable across fixed-effect specifications while `glmm`'s LMM path is REML-only anyway. `-2 * fit.loglik + 2 * fit.df` reproduces the usual formula for GLM/GLMM fits (`fit.reml` is `False` there). |
| `.llf` | `.loglik` | Exists, renamed, with the same caveat as above: for an LMM, `.loglik` is the **REML criterion**, not an ordinary log-likelihood — check `fit.reml` before comparing it to anything. |
| `.conf_int()` | — | No method. Build a Wald interval from `fit.beta ± z_crit * fit.se` (or the full `fit.vcov` for a joint contrast) — the same recipe `coming-from-lme4.md`'s `confint()` row gives for `lme4`. |

## What you gain

- **Non-Gaussian mixed models at all.** `MixedLM` is Gaussian-only; `glmm.fit`
  covers binomial (logit/probit), Poisson, Gamma, and negative-binomial GLMMs
  through the same call, just `family=...`.
- **Crossed and nested groupings as ordinary formula terms.** `(1 | g1) + (1 | g2)`
  for crossed factors, `(1 | g1/g2)` for nesting — see recipes 2 and 3 in
  [`examples-python.md`](examples-python.md). `statsmodels` has no formula
  shorthand for either shape on `MixedLM`.
- **Adaptive Gauss–Hermite quadrature.** `nagq=<odd integer, 1..25>` on
  binomial/Poisson GLMMs with a single grouping factor and up to 3 random
  effects per group (`nagq=1`, the default, is Laplace) — `statsmodels` has
  no quadrature option for mixed models.

## Reading `.summary()` against statsmodels'

`fit.summary()`'s coefficient table has the same four columns you'd expect
from `MixedLMResults.summary()` or `GLMResults.summary()` — name, estimate,
standard error, and a test statistic with its p-value — but three things are
different by design, all verified against the printed output in
[`examples-python.md`](examples-python.md):

- **No residual-degrees-of-freedom line.** statsmodels' summary reports
  group/observation counts; `glmm`'s doesn't carry a residual-df concept at
  all — there's no `t`-based inference to justify one (next point).
- **The test column is always Wald z, never `t`.** `glmm.fit`'s own docstring
  is explicit about why: *"Wald-z (not t) matches the GLM/GLMM convention and
  the absence of a residual-df field on the kernel output."* This applies
  uniformly, including on the Gaussian LMM path, where `MixedLMResults`
  already also reports `z`/`P>|z|` (not `t`) — so this one is not actually a
  difference from `MixedLM`, only from a plain OLS/GLM `t`-table intuition.
- **No AIC/BIC/log-likelihood line in the printed table.** `fit.loglik` and
  `fit.df` are on the `Fit` object (see the table above for the caveats on
  building an AIC from them by hand); they just don't appear in
  `summary()`'s own output the way `MixedLMResults.summary()` prints
  `Log-Likelihood`/converged status inline.

The footer line `summary()` does print —
`dispersion: <value>   converged: <bool>   singular: <bool>` — has no
statsmodels analogue at all: `dispersion` is φ/θ/σ² depending on family (see
[`conventions.md`](conventions.md)), and `converged`/`singular` are the two
flags [`coming-from-lme4.md`](coming-from-lme4.md#behavioral-differences-to-watch)
covers in more depth.
