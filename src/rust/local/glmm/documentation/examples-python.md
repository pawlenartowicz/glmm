# Worked examples (Python)

Nine recipes, each against a dataset already frozen under
[`validation/data/`](../validation/data/) — that is the point of using those
datasets rather than inventing new ones: the printed output below is
checkable against pinned lme4 results instead of taken on faith. Every code
block here is trimmed to the essential call; the full, runnable script —
including the frozen-golden cross-check that produced the "oracle
cross-check" lines in each output block — lives under
[`documentation/examples/python/`](examples/python/), one file per recipe.
Every output block on this page is pasted verbatim from running that script;
none of the numbers below were typed by hand.

The same nine recipes, in R, are [`examples-r.md`](examples-r.md).

**Where the data lives.** Recipes 1–7 read a CSV out of
[`validation/data/empirical/`](../validation/data/empirical/) by relative
path — the same lme4 datasets the [R examples](examples-r.md) load via
`library(lme4); data(...)`, just without the R-specific dataset registry.
Recipes 8–9 have no lme4-bundled equivalent (negative binomial, an offset
column) and instead read a fixture from
[`validation/data/simulated/`](../validation/data/simulated/), generated for
the validation harness. No recipe bundles a second copy of any dataset.

**Oracle cross-check, and where it stops.** Where a recipe's dataset and
formula match a manifest rung, its fixed effects and variance components are
checked against the frozen lme4 JSON under
[`validation/goldens/`](../validation/goldens/), inside the tolerance bands
in [`validation/tol.R`](../validation/tol.R) — six of the nine do (recipes
1, 2, 3, 4, 6, 8). Three do not, and say so plainly rather than implying a
match that isn't there: recipe 5 (grouseticks) uses a simpler, more
pedagogical formula than manifest rung 6's, so rung 6's golden numbers do not
apply; recipe 7 (cake) is manifest rung 13, but no lme4 reference was ever
frozen under `validation/goldens/` for it; recipe 9 (the offset) has no
golden at all. Each says "a run, not an oracle-pinned result" at the point it
applies.

**One formula gotcha that hits every recipe below.** The shared formula
parser always carries an intercept implicitly and has no bare `1` term on
the fixed-effects side — `y ~ 1 + x` is a parse error; write `y ~ x`. lme4
formulas (including the ones in `validation/manifest.json`) are conventionally
written `y ~ 1 + x`, so every formula string below drops that leading `1 +`
relative to its lme4 original. See [`formula.md`](formula.md).

## 1. Correlated random slope (`sleepstudy`)

Reaction time slows over sleep-deprived days, and both the starting point and
the rate of slowing vary by subject — a random intercept alone would force
every subject onto parallel lines, so the model needs a random slope too, and
`(1 + Days | Subject)` lets the two covary rather than assuming
independence.

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

Full script, plus the goldens cross-check:
[`examples/python/01_sleepstudy_random_slope.py`](examples/python/01_sleepstudy_random_slope.py).

```
converged: True  singular: False
name             estimate    std.error          z          p
(Intercept)         251.4        6.825      36.84 4.497e-297
Days                10.47        1.546      6.771  1.275e-11

Random effects:
  Subject:
    (Intercept)  sd      24.74  se        nan  corr    1.000
    Days         sd      5.922  se        nan  corr    0.066    1.000

dispersion: 654.941   converged: True   singular: False
Subject stddevs: [24.74043132  5.9221323 ]
Subject corr: [[1.0, 0.06555162566840655], [0.06555162566840655, 1.0]]
loglik (REML crit): -871.814135979251  reml: True
dispersion (residual sigma^2): 654.9411291155503

oracle cross-check vs goldens/sleepstudy_lmm.json (manifest rung 2):
  beta[Intercept]              got=251.4051048        ref=251.4051048        rel_err=0  tol=0.001  [PASS]
  beta[Days]                   got=10.46728596        ref=10.46728596        rel_err=1.26e-14  tol=0.001  [PASS]
  se[Intercept]                got=6.824552619        ref=6.824596695        rel_err=6.46e-06  tol=0.001  [PASS]
  se[Days]                     got=1.545788748        ref=1.545789644        rel_err=5.79e-07  tol=0.001  [PASS]
  Subject sd[Intercept]        got=24.74043132        ref=24.74065799        rel_err=9.16e-06  tol=0.001  [PASS]
  Subject sd[Days]             got=5.922132304        ref=5.922137659        rel_err=9.04e-07  tol=0.001  [PASS]
  loglik (REML crit)           got=-871.814136        ref=-871.814136        abs_err=7.25e-10  tol=2e-06  [PASS]
```

The number a newcomer misreads is the `0.066` correlation between the
intercept and the slope, printed as `corr` on the `Days` row. It says
subjects who start slower (`(Intercept)`) barely differ from subjects who
start faster in *how fast their reaction time then degrades* — a small,
near-zero correlation, not the two variance components themselves (24.74 and
5.92, on very different scales already). Reading it as "the slope is 0.066"
or as a fraction of variance explained is the mistake; it is a correlation
coefficient between two random effects, bounded in [-1, 1].

## 2. Crossed grouping factors (`Penicillin`)

Every `sample` was tested on every `plate` — the two grouping factors are
crossed, not one nested inside the other, so the model needs two independent
`(1 | g)` terms rather than lme4's `/` nesting shorthand (recipe 3 is the
nested case, for contrast).

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

Full script, plus the goldens cross-check:
[`examples/python/02_penicillin_crossed.py`](examples/python/02_penicillin_crossed.py).

```
converged: True  singular: False
name             estimate    std.error          z          p
(Intercept)         22.97       0.8086      28.41 1.487e-177

Random effects:
  plate:
    (Intercept)  sd     0.8467  se        nan  corr    1.000
  sample:
    (Intercept)  sd      1.932  se        nan  corr    1.000

dispersion: 0.302415   converged: True   singular: False
plate stddev: 0.8467044317
sample stddev: 1.931558312
loglik (REML crit): -165.4302944955325  reml: True

oracle cross-check vs goldens/penicillin_lmm.json (manifest rung 3):
  beta[Intercept]              got=22.97222222        ref=22.97222222        rel_err=4.55e-14  tol=0.001  [PASS]
  se[Intercept]                got=0.8085733584       ref=0.8085953614       rel_err=2.72e-05  tol=0.001  [PASS]
  plate stddev                 got=0.8467044317       ref=0.8467025103       rel_err=2.27e-06  tol=0.001  [PASS]
  sample stddev                got=1.931558312        ref=1.931613792        rel_err=2.87e-05  tol=0.001  [PASS]
  loglik (REML crit)           got=-165.4302945       ref=-165.4302945       abs_err=4.27e-09  tol=2e-06  [PASS]
```

The part worth reading twice: `sample`'s standard deviation (1.93) is over
twice `plate`'s (0.85), even though both are single-term `(1 | g)` blocks
that look symmetric in the formula. Crossed does not mean symmetric — it
means independent — and there is no reason for two unrelated sources of
variation to come out the same size just because they're written the same
way.

## 3. Nested grouping factors (`Pastes`)

`cask` only means something inside its `batch` — cask `"a"` of batch `"A"`
is unrelated to cask `"a"` of batch `"B"` — so the model is
`(1 | batch/cask)`, R's nesting shorthand, and it produces two variance
components rather than one.

```python
import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "Pastes.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "strength": [float(r["strength"]) for r in rows],
    "batch": [r["batch"] for r in rows],
    "cask": [r["cask"] for r in rows],
}

fit = glmm.fit(data, "strength ~ (1 | batch/cask)")

assert fit.converged
fit.summary()
```

Full script, plus the goldens cross-check:
[`examples/python/03_pastes_nested.py`](examples/python/03_pastes_nested.py).

```
converged: True  singular: False
name             estimate    std.error          z          p
(Intercept)         60.05       0.6769      88.72          0

Random effects:
  batch:
    (Intercept)  sd      1.287  se        nan  corr    1.000
  batch:cask:
    (Intercept)  sd      2.904  se        nan  corr    1.000

dispersion: 0.678001   converged: True   singular: False
loglik (REML crit): -123.49537292706944

oracle cross-check vs goldens/pastes_lmm.json (manifest rung 4):
  beta[Intercept]              got=60.05333333        ref=60.05333333        rel_err=1.61e-14  tol=0.001  [PASS]
  se[Intercept]                got=0.676868586        ref=0.6768702151       rel_err=2.41e-06  tol=0.001  [PASS]
  batch stddev                 got=1.287358607        ref=1.287365881        rel_err=5.65e-06  tol=0.001  [PASS]
  batch:cask stddev            got=2.90407562         ref=2.904077466        rel_err=6.36e-07  tol=0.001  [PASS]
  loglik (REML crit)           got=-123.4953729       ref=-123.4953729       abs_err=3.25e-10  tol=2e-06  [PASS]
```

The two variance components here are easy to mix up: `batch` (10 levels,
sd 1.29) is the coarser grouping, `batch:cask` (30 levels — one per cask
actually observed within its batch, sd 2.90) is the finer one *nested
inside* it. `re_groups` names them by their factors, `batch` vs `batch:cask`
— the one with the colon is always the finer, nested grouping, never the
coarser one, regardless of which order the fit happens to list them in.

## 4. Aggregated binomial via `weights=` (`cbpp`)

Each row of `cbpp` is a herd-period, not a single animal: `incidence` cases
out of `size` at risk. lme4 writes this as
`cbind(incidence, size - incidence) ~ ...`. The shared parser also accepts
`cbind()` directly with `family = "binomial"`, but both arguments must be
columns — compute the failures column first (`failures = size - incidence`;
arithmetic inside `cbind()` itself is not accepted) and pass
`cbind(incidence, failures) ~ ...`. This recipe instead spells the same model
as the success *proportion* as the response plus the trial count as
`weights=` — exactly lme4's objective underneath `cbind()`, spelled
differently.

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

Full script, plus the goldens cross-check:
[`examples/python/04_cbpp_binomial_weights.py`](examples/python/04_cbpp_binomial_weights.py).

```
converged: True  singular: False
name             estimate    std.error          z          p
(Intercept)        -1.399       0.2325     -6.016  1.789e-09
period2           -0.9923       0.3066     -3.236   0.001212
period3            -1.129       0.3266     -3.455  0.0005494
period4             -1.58       0.4274     -3.697   0.000218

Random effects:
  herd:
    (Intercept)  sd     0.6423  se     0.1786  corr    1.000

dispersion: 1   converged: True   singular: False
herd stddev: 0.6422614366557076
loglik: -92.02628186475727

oracle cross-check vs goldens/cbpp_agq_k1.json (manifest rung 5, nagq=1):
  beta[(Intercept)]            got=-1.398532067       ref=-1.398532044       rel_err=1.64e-08  tol=0.001  [PASS]
  beta[period2]                got=-0.9923328153      ref=-0.9923158803      rel_err=1.71e-05  tol=0.001  [PASS]
  beta[period3]                got=-1.128672188       ref=-1.128664147       rel_err=7.12e-06  tol=0.001  [PASS]
  beta[period4]                got=-1.580313861       ref=-1.580315598       rel_err=1.1e-06  tol=0.001  [PASS]
  se_hessian[(Intercept)]      got=0.2324738116       ref=0.2324732548       rel_err=2.4e-06  tol=0.001  [PASS]
  se_hessian[period2]          got=0.3066431739       ref=0.3066413264       rel_err=6.02e-06  tol=0.001  [PASS]
  se_hessian[period3]          got=0.3266383584       ref=0.3266372426       rel_err=3.42e-06  tol=0.001  [PASS]
  se_hessian[period4]          got=0.427436017        ref=0.4274372446       rel_err=2.87e-06  tol=0.001  [PASS]
  herd stddev                  got=0.6422614367       ref=0.6422698887       rel_err=1.32e-05  tol=0.001  [PASS]
  loglik                       got=-92.02628186       ref=-92.02628187       abs_err=9.75e-09  tol=0.001  [PASS]
```

The trap here is passing `incidence` itself as `weights=` instead of `size`:
`weights=` is the *denominator* (trials), not the count of successes — get
that backwards and every row is weighted by how many cases it had, which is
correlated with the very thing being modeled.

## 5. Poisson GLMM (`grouseticks`)

Tick counts per chick, with a random intercept per `BROOD`: chicks raised
together share unmeasured causes of infestation (nest site, parent
condition) that a fixed effect on `YEAR` and `HEIGHT` alone cannot capture.

```python
import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "empirical" / "grouseticks.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "TICKS": [float(r["TICKS"]) for r in rows],
    "YEAR": [r["YEAR"] for r in rows],
    "HEIGHT": [float(r["HEIGHT"]) for r in rows],
    "BROOD": [r["BROOD"] for r in rows],
}

fit = glmm.fit(data, "TICKS ~ YEAR + HEIGHT + (1 | BROOD)", family="poisson")

assert fit.converged
fit.summary()
```

Full script:
[`examples/python/05_grouseticks_poisson.py`](examples/python/05_grouseticks_poisson.py).
**No manifest rung matches this exact formula** — rung 6 (`grouseticks`) fits
the centered `cHEIGHT` against all three crossed grouping factors (`BROOD`,
`INDEX`, `LOCATION`) together, a different model from this recipe's single
`(1 | BROOD)` on raw `HEIGHT`. Dropping two grouping factors and swapping the
height variable changes what each variance component absorbs, so rung 6's
golden numbers are not a valid comparison target for this fit. This output
is a run, not an oracle-pinned result.

```
converged: True  singular: False
name             estimate    std.error          z          p
(Intercept)         11.25         1.39      8.097   5.63e-16
YEAR96              1.137       0.2397      4.745  2.088e-06
YEAR97             -1.018        0.269     -3.784  0.0001545
HEIGHT           -0.02323     0.002985     -7.781   7.21e-15

Random effects:
  BROOD:
    (Intercept)  sd     0.9457  se        nan  corr    1.000

dispersion: 1   converged: True   singular: False
BROOD stddev: 0.945727072876557
loglik: -989.0649432849432

(no manifest rung matches this formula -- a run, not an oracle-pinned result)
```

The coefficients on `YEAR96` and `YEAR97` are both contrasts against
`YEAR95` (the first level), not against each other — `YEAR96`'s positive
1.14 and `YEAR97`'s negative -1.02 do not mean 1996 and 1997 are opposite in
sign relative to *each other*; they mean 1996 had noticeably more ticks than
1995, and 1997 noticeably fewer, each measured against the same 1995
baseline.

## 6. Adaptive quadrature (`cbpp` at `nagq=7`)

Recipe 4's model again, but integrated over the random effect with a
7-point adaptive Gauss-Hermite quadrature instead of the `nagq=1` Laplace
default. Eligible here because the model has a single grouping factor
(`herd`) with one random effect per level (`q=1`) — AGQ's current cap is
`q<=3` on a single binomial/Poisson grouping factor.

```python
fit_laplace = glmm.fit(data, "prop ~ period + (1 | herd)", family="binomial", weights=size, nagq=1)
fit_agq = glmm.fit(data, "prop ~ period + (1 | herd)", family="binomial", weights=size, nagq=7)
```

(`data` is built exactly as in recipe 4.) Full script, plus the goldens
cross-check:
[`examples/python/06_cbpp_adaptive_quadrature.py`](examples/python/06_cbpp_adaptive_quadrature.py).
The cross-check below compares beta, `se_hessian`, and the herd standard
deviation only — the crate's own oracle test for this exact golden
(`fit_glmm_binomial_agq_matches_lme4`, `src/fit/glmm_tests.rs`) gates the
same three quantities and deliberately not log-likelihood, because
log-likelihood does not even agree with itself across `nAGQ` in lme4's own
output for this fit (see the reading below).

```
=== nagq=1 (Laplace) ===
name             estimate    std.error          z          p
(Intercept)        -1.399       0.2325     -6.016  1.789e-09
period2           -0.9923       0.3066     -3.236   0.001212
period3            -1.129       0.3266     -3.455  0.0005494
period4             -1.58       0.4274     -3.697   0.000218

Random effects:
  herd:
    (Intercept)  sd     0.6423  se     0.1786  corr    1.000

dispersion: 1   converged: True   singular: False
loglik: -92.02628186475727

=== nagq=7 (adaptive Gauss-Hermite) ===
name             estimate    std.error          z          p
(Intercept)        -1.399       0.2335     -5.992  2.072e-09
period2           -0.9914       0.3068     -3.232    0.00123
period3            -1.128       0.3268     -3.451  0.0005576
period4            -1.579       0.4276     -3.694  0.0002209

Random effects:
  herd:
    (Intercept)  sd     0.6475  se     0.1805  corr    1.000

dispersion: 1   converged: True   singular: False
loglik: -91.98335426918706

Laplace -> AGQ(7) movement on this data:
  beta[(Intercept)]: laplace=-1.398532067 agq7=-1.399232151 delta=-0.0007 rel=0.000501
  beta[period2]: laplace=-0.9923328153 agq7=-0.9914028429 delta=0.00093 rel=0.000937
  beta[period3]: laplace=-1.128672188 agq7=-1.127818208 delta=0.000854 rel=0.000757
  beta[period4]: laplace=-1.580313861 agq7=-1.579469189 delta=0.000845 rel=0.000534
  loglik: laplace=-92.02628186 agq7=-91.98335427 delta=0.0429

oracle cross-check vs goldens/cbpp_agq_k7.json (manifest rung 5 at nagq=7):
(beta, se_hessian, herd stddev only -- see module docstring on why loglik is excluded)
  beta[(Intercept)]            got=-1.399232151       ref=-1.399229755       rel_err=1.71e-06  tol=0.001  [PASS]
  beta[period2]                got=-0.9914028429      ref=-0.9913961245      rel_err=6.78e-06  tol=0.001  [PASS]
  beta[period3]                got=-1.127818208       ref=-1.127832892       rel_err=1.3e-05  tol=0.001  [PASS]
  beta[period4]                got=-1.579469189       ref=-1.579443908       rel_err=1.6e-05  tol=0.001  [PASS]
  se_hessian[(Intercept)]      got=0.2335141791       ref=0.2335116086       rel_err=1.1e-05  tol=0.001  [PASS]
  se_hessian[period2]          got=0.30676867         ref=0.3067673681       rel_err=4.24e-06  tol=0.001  [PASS]
  se_hessian[period3]          got=0.3267688578       ref=0.3267697654       rel_err=2.78e-06  tol=0.001  [PASS]
  se_hessian[period4]          got=0.42759394         ref=0.4275906557       rel_err=7.68e-06  tol=0.001  [PASS]
  herd stddev                  got=0.6475205669       ref=0.6475178183       rel_err=4.24e-06  tol=0.001  [PASS]
```

The thing a newcomer misreads: the answer *does* move relative to Laplace,
but by very little on this data — every beta shifts by under 0.1% and the
herd standard deviation moves from 0.642 to 0.648, exactly the direction
theory predicts (Laplace's variance-component bias runs low). What is easy
to over-read is `loglik`: it shifts by only 0.043 within `glmm`'s own two
fits above, but if you go compare against lme4's own `nAGQ=1` vs `nAGQ=7`
refits of this exact model, lme4's *own* reported log-likelihood jumps from
-92.0 to -50.0 — a ~42-unit change despite beta barely moving. That is an
artifact of how lme4 computes `logLik()` at different quadrature orders for
an aggregated-trials binomial, not a sign that either fit is wrong, and it's
exactly why this recipe's cross-check does not touch log-likelihood at all.

## 7. Factors and interactions (`cake`), and changing the base level

`recipe*temp` desugars to `recipe + temp + recipe:temp`: a main effect per
recipe, a slope on `temp`, and a per-recipe deviation from that slope.
Treatment contrasts code `recipe` against its first level — alphabetically
`"A"`, since a plain string column declares no order — and
`pandas.Categorical`'s declared category order is how you pick a different
base without relabeling the data.

```python
import pandas as pd

# recipe_labels, temp, replicate, angle already loaded as plain lists.

# Default: no declared order -> lexicographic -> base = "A".
data = {"angle": angle, "recipe": recipe_labels, "temp": temp, "replicate": replicate}
fit_a = glmm.fit(data, "angle ~ recipe*temp + (1 | recipe:replicate)")

# Same model, base = "B": a pandas.Categorical with "B" listed first.
data_b = dict(data)
data_b["recipe"] = pd.Categorical(recipe_labels, categories=["B", "A", "C"])
fit_b = glmm.fit(data_b, "angle ~ recipe*temp + (1 | recipe:replicate)")
```

Full script:
[`examples/python/07_cake_factors_interactions.py`](examples/python/07_cake_factors_interactions.py).
**No manifest rung's golden covers this dataset** — `cake` is manifest rung
13, but no lme4 reference JSON was ever frozen under `validation/goldens/`
for it. This output is a run, not an oracle-pinned result.

```
=== base = A (default, no declared order) ===
name             estimate    std.error          z          p
(Intercept)         2.379        5.903     0.4031     0.6869
recipeB            -3.649        8.348    -0.4371      0.662
recipeC            -1.941        8.348    -0.2325     0.8161
temp               0.1537      0.02821      5.449  5.054e-08
recipeB:temp      0.01086      0.03989     0.2722     0.7855
recipeC:temp     0.002095      0.03989    0.05252     0.9581

Random effects:
  recipe:replicate:
    (Intercept)  sd      6.463  se        nan  corr    1.000

dispersion: 20.8861   converged: True   singular: False

=== base = B (pandas.Categorical(categories=['B', 'A', 'C'])) ===
name             estimate    std.error          z          p
(Intercept)         -1.27        5.903    -0.2151     0.8297
recipeA             3.649        8.348     0.4371      0.662
recipeC             1.708        8.348     0.2046     0.8379
temp               0.1646      0.02821      5.834  5.401e-09
recipeA:temp     -0.01086      0.03989    -0.2722     0.7855
recipeC:temp    -0.008762      0.03989    -0.2196     0.8261

Random effects:
  recipe:replicate:
    (Intercept)  sd      6.463  se        nan  corr    1.000

dispersion: 20.8861   converged: True   singular: False

Same fit, different parameterization: fitted values and loglik agree (loglik A=-851.63374, loglik B=-851.63374, delta=1.02e-12); only which contrasts are directly readable off beta changes.

(cake carries no goldens/ entry -- a run, not an oracle-pinned result)
```

The number that looks alarming but isn't: `temp`'s coefficient changes from
0.1537 (base A) to 0.1646 (base B) — a real, non-trivial shift, not a
rounding artifact. Under treatment coding with interactions, the main-effect
slope on `temp` is specifically *recipe A's* slope in the first fit and
*recipe B's* slope in the second; it was never a shared, recipe-independent
slope in either parameterization, `recipe*temp` includes the interaction
precisely so each recipe can have its own. The `loglik` agreement (to 1e-12)
is what confirms it's the same fit looked at two ways, not two different
models.

## 8. Negative binomial (`sim_nb`)

Overdispersed counts that a Poisson model would under-estimate the variance
of: `dispersion` on the returned `Fit` is theta-hat, the NB shape parameter
(`MASS::glm.nb`'s `theta`), not phi — unlike gamma or inverse-Gaussian, where
`dispersion` *is* the Pearson phi. Overdispersion relative to Poisson is
`1/theta`, so a *large* theta means "close to Poisson", not "a lot of extra
variance".

```python
import csv
from pathlib import Path

import glmm

DATA_PATH = Path(__file__).resolve().parents[3] / "validation" / "data" / "simulated" / "sim_nb.csv"

with open(DATA_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

data = {
    "y": [float(r["y"]) for r in rows],
    "x": [float(r["x"]) for r in rows],
    "grp": [r["grp"] for r in rows],
}

fit = glmm.fit(data, "y ~ x + grp", family="negativebinomial")

assert fit.converged
fit.summary()
```

Full script, plus the goldens cross-check:
[`examples/python/08_negative_binomial.py`](examples/python/08_negative_binomial.py).
No lme4-bundled dataset exercises negative binomial, so this reads
`validation/data/simulated/sim_nb.csv` rather than an lme4 dataset — a
fixture generated for the validation harness.

```
converged: True  singular: False
name             estimate    std.error          z          p
(Intercept)        0.1442       0.1207      1.195     0.2323
x                  0.6198      0.07564      8.194  2.527e-16
grpb               0.6337       0.1557       4.07   4.71e-05

dispersion: 1.01052   converged: True   singular: False
theta (NB shape): 1.0105218893928158
loglik: -497.32514268667205

oracle cross-check vs goldens/sim_nb_glm.json (GLM, no random effects):
  beta[(Intercept)]            got=0.1441656485       ref=0.1441660779       rel_err=2.98e-06  tol=0.001  [PASS]
  beta[x]                      got=0.6198259071       ref=0.6198268706       rel_err=1.55e-06  tol=0.001  [PASS]
  beta[grpb]                   got=0.6336876107       ref=0.6336868995       rel_err=1.12e-06  tol=0.001  [PASS]
  se[(Intercept)]              got=0.1206905798       ref=0.120690562        rel_err=1.48e-07  tol=0.001  [PASS]
  se[x]                        got=0.07564414504      ref=0.07564420041      rel_err=7.32e-07  tol=0.001  [PASS]
  se[grpb]                     got=0.1557142496       ref=0.1557142563       rel_err=4.31e-08  tol=0.001  [PASS]
  theta                        got=1.010521889        ref=1.010521815        rel_err=7.32e-08  tol=0.001  [PASS]
  loglik                       got=-497.3251427       ref=-497.3251427       abs_err=8.2e-11  tol=0.001  [PASS]
```

Note this is a GLM, not a GLMM: `sim_nb.csv` carries a `cluster` column that
this recipe's formula does not use. The manifest also registers a
`sim_nb_glmm` golden with a `(1 | cluster)` term, but this recipe deliberately
fits the simpler GLM — the one thing every negative-binomial fit shares
regardless of random effects, `dispersion` being theta rather than phi, is
already visible without adding a grouping factor.

## 9. Offset (`sim_poisson_offset`)

A Poisson rate model against a known exposure: each row's expected count
scales with `exposure`, so `offset = log(exposure)` enters the linear
predictor with a fixed coefficient of 1 rather than an estimated one.

```python
data = {"y": y, "x": x, "cluster": cluster}          # already loaded as lists
log_exposure = [...]                                  # log(exposure), precomputed in the CSV

fit_with = glmm.fit(data, "y ~ x + (1 | cluster)", family="poisson", offset=log_exposure)
fit_without = glmm.fit(data, "y ~ x + (1 | cluster)", family="poisson")
```

Full script:
[`examples/python/09_poisson_offset.py`](examples/python/09_poisson_offset.py).
Data: `validation/data/simulated/sim_poisson_offset.csv`, a fixture generated
for the validation harness with a `log_exposure` column already computed —
`offset=` takes the log-exposure directly, not the raw exposure. No
`goldens/` entry covers this dataset (it is registered at manifest rung 28
with a real `offset` field, but no lme4 reference JSON was frozen for it).
This output is a run, not an oracle-pinned result.

```
=== with offset = log(exposure) ===
name             estimate    std.error          z          p
(Intercept)        0.3269       0.0969      3.373  0.0007435
x                  0.4935     0.009219      53.53          0

Random effects:
  cluster:
    (Intercept)  sd     0.5269  se    0.06913  corr    1.000

dispersion: 1   converged: True   singular: False
cluster stddev: 0.5269486042406111

=== without the offset (exposure variation folded into the fit) ===
name             estimate    std.error          z          p
(Intercept)         1.323      0.09886      13.38  7.455e-41
x                  0.5224     0.009289      56.23          0

Random effects:
  cluster:
    (Intercept)  sd     0.5377  se    0.07053  corr    1.000

dispersion: 1   converged: True   singular: False
cluster stddev: 0.5377180043580608

Dropping a real offset does not just bias the intercept: here the slope on x moves from 0.4935 (with offset) to 0.5224 (without), and the cluster standard deviation moves from 0.5269 to 0.5377 -- the unexplained exposure variation is absorbed by both the fixed and the random-effect side, not cleanly by either alone.

(no goldens/ entry for sim_poisson_offset -- a run, not an oracle-pinned result)
```

The easy misreading: expecting a missing offset to only bias the intercept
(since `offset=log(exposure)` most resembles an intercept shift). It doesn't
stay contained there — dropping it here also moves the slope on `x` (0.494
→ 0.522) and the cluster standard deviation (0.527 → 0.538), because the
model has no other way to explain the exposure-driven part of the count
variation except by routing it through whatever terms are left.
