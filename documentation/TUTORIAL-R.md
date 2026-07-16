# Using `glmm` from R (the `fastglmm` package)

One page, four sections. The first walks the single entry point — `fastglmm()` —
end to end; the next two go deeper into the knobs and the object it returns; the
fourth is a short note on warm starts. The R surface is deliberately small:
**one function** — `fastglmm()` — plus the `"fastglmm"` object it returns, read
through lme4-shaped accessors (`summary`, `fixef`, `vcov`, `VarCorr`, `confint`,
`isSingular`). There is no `lmer`/`glmer` split — `fastglmm()` dispatches on the
family the way the kernel does.

Two R conventions differ from the Python port. The **formula comes first**,
`data` second (as in `lm`/`lme4`), the reverse of `glmm.fit(data, formula)`. And
`family` is an R **family object / function / string** (`binomial()`,
`poisson`, `"gamma"`), not a bare string — which brings one trap, the `Gamma()`
link default (§2).

> **Status:** this release ships the full API surface — `fastglmm()`, argument
> validation, every working accessor — wired end to end through the extendr
> binding: a valid call parses the formula, fits, and returns a real
> `"fastglmm"` object. Four narrow combinations are GLMM 0.1.1 gaps and raise a
> clean error naming the reason instead of fitting: `family = inverse.gaussian()`,
> `binomial("cloglog")`, quasi-likelihood `dispersion=` on binomial/poisson, and
> an `init.theta=` shape seed (see §2).

The package is not on CRAN yet. Install from a checkout (needs Rust — `cargo`
and `rustc >= 1.85` on the `PATH`):

```r
# in GLMM/r/
install.packages(".", repos = NULL, type = "source")
```

or, once the first `r-*` tag has built, from r-universe:

```r
install.packages("fastglmm",
                 repos = c("https://pawlenartowicz.r-universe.dev", getOption("repos")))
```

(Same models from other surfaces: the Python port —
[`TUTORIAL-PYTHON.md`](TUTORIAL-PYTHON.md) — and the Rust crate this package
binds — [`TUTORIAL-RUST.md`](TUTORIAL-RUST.md), hand-built inputs instead of a
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
  group = factor(rep(letters[1:6], each = 2))
)

# y ~ x1 + (1 | group) — Gaussian, random intercept.
fit <- fastglmm(y ~ x1 + (1 | group), data)

fit$converged
summary(fit)   # coefficient table + variance components; no fake AIC line
```

Points worth knowing at this layer:

- `data` is a `data.frame` (or anything `as.data.frame()` accepts). Response and
  fixed-effect columns are read as `double`; grouping columns (the `g` in
  `(1 + x | g)`) may be factor, character, integer, or logical and are
  factorized to level ids. A character column becomes a factor with
  **lexicographic** level order (exactly what `factor()` does); a factor's
  declared level order is honored.
- The default `family = gaussian()` with no `(… | g)` term fits **OLS**; adding a
  random-effect term makes it an **LMM** (REML). The same split holds for every
  family: fixed-only ⇒ GLM, `(… | g)` present ⇒ GLMM.
- The formula follows R conventions but takes **bare column names only**: `+`,
  `:`, `*` (main effects + interaction), `A/B` nesting, and `(1 + x | g)` for a
  correlated random intercept + slope, with extra `(… | g2)` terms adding
  crossed/nested groupings. Not accepted — each a clear error with the fix:
  `I()`/`poly()`/`log(x)` (compute the column first), `cbind(s, f)` (pass the
  proportion as response and trials as `weights=`), `offset()`, `.`, `- 1`/`0 +`,
  and `(x || g)`.
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
| `binomial()` / `"binomial"` | binomial | logit | probit | — |
| `poisson()` / `"poisson"` | poisson | log | — | — |
| `Gamma()` **(object)** | gamma | **inverse** | log | dispersion |
| `"gamma"` **(string)** | gamma | **log** | inverse | dispersion |
| `"negativebinomial"` | negative binomial | log | — | theta (estimated) |

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
  **odd** integer `≤ 25`. `> 1` applies to binomial/Poisson mixed models with a
  single grouping factor and `≤ 3` random effects per group; **any other shape
  warns and falls back to Laplace** rather than erroring the way `lme4::glmer`
  does — the fit you get is a Laplace fit, and the warning is the only notice.
- `dispersion` — Gamma dispersion directive: `NULL` (estimate φ̂ by Pearson, the
  default), `"estimate"` (same), or a single number to hold φ fixed.
- `wald.se` — fixed-effect Wald-SE mode: `"hessian"` (default) or `"rx"`.
- `weights` — per-row prior (case) weights, `lme4::glmer`'s `weights=`. For an
  aggregated binomial, pass the success **proportion** as the response and the
  trial count here — the same model as `cbind(successes, failures)`, whose syntax
  the parser does not accept. Weights must be strictly positive.
- `start` — warm start; see §4.
- `init.theta` — negative-binomial shape seed, named for
  `MASS::glm.nb(init.theta=)`, and **distinct** from `start$theta` (the
  random-effect Cholesky start): unrelated knobs sharing a Greek letter.

**Error vs. warning.** An *invalid* value — unknown family, a link the family
doesn't offer, a non-odd or out-of-range `nAGQ`, a `start` that isn't a list —
**errors**. A *valid but inapplicable* option — `dispersion=` on gaussian,
`init.theta=` off negative-binomial, quasi-dispersion on a mixed
binomial/Poisson formula — **warns and is stripped**, so it never reaches the
kernel: loud enough to catch the mistake, lenient enough for exploration.

Three things error *naming the reason* rather than fitting a silently different
model. Known lme4 arguments passed through `...` (`REML = FALSE`, `control=`,
`verbose=`, `contrasts=`, `offset=`) each explain why they can't be honored
(`REML = FALSE`, for instance, because the LMM path is REML-only by design). The
GLMM 0.1.1 gaps (`inverse.gaussian`, `cloglog`, quasi-likelihood dispersion on
binomial/Poisson) error until the kernel implements them. And `init.theta=` with
an actual value errors — there is no kernel hook to seed the shape search yet, so
only the default cold start runs (off negative-binomial the same argument is the
harmless warn-and-strip case above).

## 3. Reading the result

The returned object is class `"fastglmm"`, read through accessors named and
shaped like lme4's. These **work**:

| accessor | what it returns |
|---|---|
| `summary(fit)` | coefficient table — estimate, std. error, Wald **z**, `Pr(>|z|)` — plus the RE block and a dispersion/shape footer. **No** `AIC`/`BIC`/`logLik`/`deviance` line: the kernel surfaces no comparable log-likelihood, and a fake one would be worse than none. |
| `fixef(fit)` | named fixed-effect estimates; aliased (rank-deficient) columns are `NA`, as in `lm`/lme4. |
| `vcov(fit)` | full `p × p` Wald covariance of β̂. |
| `VarCorr(fit)` | variance components on the **SD/correlation** scale, one covariance per grouping, lme4-shaped; a `Residual` row (= `sigma()`) is printed for a gaussian mixed fit. |
| `confint(fit)` | Wald intervals off `vcov()`. `method = "profile"`/`"boot"` are not available and say so. |
| `isSingular(fit)` | boundary-fit flag — lme4's condition, computed by the kernel. |
| `sigma(fit)` | residual SD for gaussian/Gamma fits; `1` for binomial/Poisson/negative-binomial (fixed scale, as in lme4). |
| `nobs`, `formula`, `family`, `model.frame`, `print` | the usual; `formula()` returns the formula **string** as given (the parser is Rust-side, so there is no R `terms` object to hand back). |

These **error, naming the reason** — the package's defining choice, that anything
the kernel cannot compute honestly is a documented error, never a silently
different answer:

```r
ranef(fit)
#> Error: ranef() is engine-blocked: glmm::Fit does not surface the
#>   conditional modes / linear predictor yet ...
```

`ranef`, `predict`, `fitted`, `residuals`, `coef`, `logLik` (and with it
`AIC`/`BIC`), and `terms` all error. `coef()` is instructive: lme4's `coef()`
means fixed + random effects per group, which needs `ranef()` (engine-blocked) —
returning fixed effects only would silently differ from what the same call gives
an lme4 user, so it errors and points you at `fixef()`.

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
empty for a fixed-only model, where a warm start is a no-op anyway). Unknown list
elements warn and are dropped. This is unrelated to `init.theta` (§2), the
negative-binomial shape seed — the two just share the letter, which is why the RE
start is `start$theta` and the shape seed is a separate argument.
