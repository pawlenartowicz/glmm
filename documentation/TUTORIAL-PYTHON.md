# Using `glmm` from Python

One page, four sections. The first walks the single entry point — `glmm.fit` —
end to end; the next two go deeper into the knobs and the returned `Fit`; the
fourth is a short note on warm starts. The Python surface is deliberately tiny:
**two public names**, `glmm.fit` and `glmm.Fit`. Everything else (families,
links, knobs) is a string or scalar argument, not a type.

> **Status:** this release ships the full API surface — signatures, argument
> validation, `Fit`, `summary()` — wired end to end through the PyO3 binding:
> a valid `fit(...)` call parses the formula, fits, and returns a real `Fit`.
> Four narrow combinations are GLMM 0.1.1 gaps and raise a clean
> `NotImplementedError` instead: `family="inversegaussian"`, `link="cloglog"`,
> quasi-likelihood `dispersion=` on binomial/poisson, and an `init_theta=`
> float seed (see §2).

The package is not on PyPI yet. Install from the repo:

```bash
pip install ./python            # from the repo root; add -e for development
```

(For the Rust crate this package ports, see [`TUTORIAL-RUST.md`](TUTORIAL-RUST.md)
— same models, hand-built inputs instead of a formula — and for the R port,
[`TUTORIAL-R.md`](TUTORIAL-R.md).)

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
    "group": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
}

# y ~ x1 + (1 | group) — Gaussian, random intercept.
fit = glmm.fit(data, "y ~ x1 + (1 | group)")

assert fit.converged
fit.summary()   # prints the coefficient table (and returns it as a string)
```

Points worth knowing at this layer:

- `data` is a `dict[str, array-like]` (the documented form), but anything
  column-addressable works — pandas / polars DataFrame, pyarrow Table —
  duck-typed, no dataframe dependency. Response and fixed-effect columns are
  read as 1-D float64; grouping columns (the `g` in `(1 + x | g)`) accept
  string / integer / categorical values and are factorized to level ids.
- The default `family="gaussian"` with no `(… | g)` term fits OLS; adding a
  random-effect term makes it an LMM. The same split holds for every family:
  fixed-only ⇒ GLM, `(… | g)` present ⇒ GLMM.
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
| `inversegaussian` | `log` | `inverse_squared` | `dispersion` |

```python
fit = glmm.fit(data, "s ~ x1 + (1 | group)", "binomial", link="probit", nagq=7)
```

- `dispersion` — three states. `None` (default): gamma / inverse-gaussian
  estimate φ̂ post-fit by Pearson and scale SE by √φ̂; other families hold
  φ ≡ 1. `"estimate"`: force the Pearson estimate — on binomial/poisson this
  *is* quasi-binomial/quasi-Poisson, GLM only. A float: hold φ fixed (still
  scales SE). Fix-vs-estimate, not a warm start.
- `nagq` — adaptive Gauss–Hermite node count; `1` = Laplace (default). Must
  be odd and ≤ 25. `>1` applies to binomial/Poisson models with a single
  grouping factor and ≤ 3 random effects per group (intercept + slopes,
  temporary cap); any other shape warns and falls back to Laplace.
- `wald_se` — fixed-effect Wald-SE denominator: `"hessian"` (default) or
  `"rx"`.
- `init_theta` — negative-binomial shape seed, named for
  `MASS::glm.nb(init.theta=)`. `None` (default, the only value the kernel
  currently accepts) cold-starts the θ search; a float raises
  `NotImplementedError` — there is no kernel hook yet to seed it. Estimation
  always runs. Distinct from `warm_start["theta"]`, the random-effect Cholesky
  start (§4): unrelated knobs that happen to share a Greek letter, which is why
  this one is not just called `theta`. Both may be passed in one call.
- `weights` — per-row prior (case) weights, lme4's `weights=`. For an
  aggregated binomial, `y` is the success *proportion* and `weights` the
  trial count (lme4's `cbind(s, m−s)`).

**Error vs. warning.** An *invalid* value — unknown family, a link the family
doesn't offer, even `nagq` (`ValueError`), a non-dict `warm_start`
(`TypeError`) — raises. A *valid but inapplicable* option — `dispersion=` on
gaussian, `init_theta=` off negative-binomial, quasi-dispersion on a mixed
binomial/poisson formula — warns (`UserWarning`) and is stripped, so it never
reaches the kernel: loud enough to catch the mistake, lenient enough for
exploration. One combination is a hard error rather than a warning:
`inversegaussian` is GLM-only, so a mixed formula with it raises.

## 3. Reading the result — `Fit`

`Fit` mirrors the Rust `Fit`, plus coefficient names (the formula supplies
them). It is returned by `fit`, never constructed by callers.

| field | what it holds |
|---|---|
| `beta` | `(p,)` fixed-effect estimates |
| `se` | `(p,)` standard errors; `NaN` where unavailable |
| `vcov` | `(p, p)` full Cov(β̂) — `se` is the sqrt of its diagonal; use it for contrasts/confidence intervals, where the off-diagonals matter |
| `names` | coefficient names, aligned with `beta` |
| `aliased` | `(p,)` bool — rank-deficient columns dropped (lme4's `NA` coefficients) |
| `varcorr` | per grouping: vech-packed lower-triangular RE covariance D̂ |
| `tau2` | legacy per-element RE variances (q=1 only) — prefer `varcorr` |
| `stddev_se` | SE of each RE stddev, θ layout (not beta-aligned); `NaN` where unavailable |
| `dispersion` | φ (gamma / inverse-gaussian) / θ (negbin) / 1.0 otherwise |
| `re_groups` | per grouping, in `varcorr` order: `(name, [term names])` — what `summary()` labels the RE block with |
| `n_eval` | optimizer objective evaluations (0 on the closed-form/IRLS paths) |
| `deviance` | minimized optimizer criterion — **not** comparable across models, and not an AIC input (see below) |
| `converged` | numerical failure signals here (not an exception) — check before trusting `beta`/`se` |
| `singular` | boundary (singular) fit — `>=1` RE variance component pinned at 0; mirrors lme4's `isSingular` |

**`deviance` is not a model-comparison statistic.** It is the criterion the
optimizer minimized, on that fit's own scale: for an LMM it is lme4's
`REMLcrit` minus a data-independent constant; for a GLMM it is the marginal
Laplace deviance, which differs from −2·logLik by a data-only saturated
constant. Those constants do not cancel between two different models, so
differencing `deviance` across fits — or feeding it to an AIC — is a mistake.
It is `NaN` for OLS/GLM and on numerical failure.

**`vcov` and `se` carry different information.** `se` is only the diagonal, so
it cannot answer anything about two coefficients jointly. A contrast like
β₁ − β₂ needs `Var(β₁) + Var(β₂) − 2·Cov(β₁, β₂)`, and that covariance lives
only in `vcov`. Both are `NaN` in the same places.

`summary()` builds the coefficient table — **name, estimate, std. error, z,
p** — prints it, and returns it as a string. Aliased columns show `NaN`
estimates, as lme4 prints `NA`. A footer carries `dispersion`,
`converged` and `singular`, and when `varcorr` is non-empty an RE block shows each grouping
by name with its per-term stddev / correlation (lme4's `VarCorr` layout) and
`stddev_se` alongside where populated — the names come from `re_groups`. The z/p columns are derived in Python from `beta`/`se` as a
Wald test (`z = beta/se`, `p = 2·(1 − Φ(|z|))`); Wald-z (not t) matches the
GLM/GLMM convention and the absence of a residual-df field on the kernel
output.

To work with a grouping's covariance numerically rather than as printed text,
`stddev_corr(group_idx)` splits its vech-packed block into a `(q,)` stddev
vector and a `(q, q)` correlation matrix:

```python
sd, corr = fit.stddev_corr(0)   # grouping 0: stddevs + correlation matrix
```

## 4. Warm starts

If you already hold optimizer state from a previous fit of the same model
shape, `warm_start` seeds the next one — it changes how fast the optimizer
converges, never the answer:

```python
fit2 = glmm.fit(data2, "y ~ x1 + (1 | group)",
                warm_start={"beta": list(fit1.beta), "theta": [...]})
```

The dict takes exactly two keys, mirroring the Rust `StartValues`: `"beta"`
(length p) and `"theta"` (Cholesky-scaled RE start; empty for fixed-only
models, where a warm start is a no-op anyway). Unknown keys warn and are
dropped. The gamma φ is a post-fit estimate, and the negative-binomial θ seed
is the `init_theta=` kwarg (§2). `warm_start` is always explicit, caller-owned —
a previous fit's state is never threaded forward automatically.

**Batch processing** (many fits with same formula behind one call — bootstrap,
simulation) is planned.
