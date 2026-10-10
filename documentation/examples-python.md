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
in [`validation/grid/tol.R`](../validation/grid/tol.R) — six of the nine do (recipes
1, 2, 3, 4, 6, 8). Three do not, and say so plainly rather than implying a
match that isn't there: recipe 5 (grouseticks) uses a simpler, more
pedagogical formula than manifest rung 6's, so rung 6's golden numbers do not
apply; recipe 7 (cake) is manifest rung 13, but no lme4 reference was ever
frozen under `validation/goldens/` for it; recipe 9 (the offset) has no
golden at all. Each says "a run, not an oracle-pinned result" at the point it
applies.

**The intercept in the formulas below.** lme4 formulas (including the ones
in `validation/manifest.json`) usually write the intercept out, `y ~ 1 + x`.
The formula strings below leave that `1 +` out. Both spellings give the same
model: the intercept is always there unless `- 1` or `0 +` removes it. See
[`formula.md`](formula.md).

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
Linear mixed model fit by REML [glmm]
Formula: Reaction ~ Days + (1 + Days | Subject)
 Family: gaussian (identity)

REML criterion at convergence: 1743.6

Scaled residuals:
    Min      1Q  Median     3Q   Max
 -3.954 -0.4634 0.02312 0.4634 5.179

Random effects:
 Groups   Name        Variance Std.Dev. se Corr
 Subject  (Intercept) 612.1    24.74    NA     
          Days        35.07    5.922    NA 0.07
 Residual             654.9    25.59           
Number of obs: 180, groups:  Subject, 18

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)         251.4        6.825      36.84     <2e-16 ***
Days                10.47        1.546      6.771   1.27e-11 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)
Days              -0.138

Optimizer evaluations: 12, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
Subject stddevs: [24.74045614  5.92213187]
Subject corr: [[1.0, 0.06555150641110637], [0.06555150641110637, 1.0]]
loglik (REML crit): -871.8141359792539  reml: True
dispersion (residual sigma^2): 654.9410288970543

oracle cross-check vs goldens/sleepstudy_lmm.json (manifest rung 2):
  beta[Intercept]              got=251.4051048        ref=251.4051048        rel_err=1.36e-15  tol=0.001  [PASS]
  beta[Days]                   got=10.46728596        ref=10.46728596        rel_err=4.07e-15  tol=0.001  [PASS]
  se[Intercept]                got=6.824557476        ref=6.824596695        rel_err=5.75e-06  tol=0.001  [PASS]
  se[Days]                     got=1.545788634        ref=1.545789644        rel_err=6.53e-07  tol=0.001  [PASS]
  Subject sd[Intercept]        got=24.74045614        ref=24.74065799        rel_err=8.16e-06  tol=0.001  [PASS]
  Subject sd[Days]             got=5.922131871        ref=5.922137659        rel_err=9.77e-07  tol=0.001  [PASS]
  loglik (REML crit)           got=-871.814136        ref=-871.814136        abs_err=7.22e-10  tol=2e-06  [PASS]
```

The number a newcomer misreads is the `0.07` correlation between the
intercept and the slope, in the `Corr` column on the `Days` row (`Subject
corr` below the summary prints it unrounded, 0.0656). It says
subjects who start slower (`(Intercept)`) barely differ from subjects who
start faster in *how fast their reaction time then degrades* — a small,
near-zero correlation, not the two variance components themselves (24.74 and
5.92, on very different scales already). Reading it as "the slope is 0.07"
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

fit = glmm.fit(data, "diameter ~ (1 | plate) + (1 | sample)")

assert fit.converged
fit.summary()
```

Full script, plus the goldens cross-check:
[`examples/python/02_penicillin_crossed.py`](examples/python/02_penicillin_crossed.py).

```
converged: True  singular: False
Linear mixed model fit by REML [glmm]
Formula: diameter ~ (1 | plate) + (1 | sample)
 Family: gaussian (identity)

REML criterion at convergence: 330.86

Scaled residuals:
    Min      1Q  Median     3Q  Max
 -2.079 -0.6714 0.06292 0.5838 2.98

Random effects:
 Groups   Name        Variance Std.Dev. se Corr
 plate    (Intercept) 0.7169   0.8467   NA     
 sample   (Intercept) 3.731    1.932    NA     
 Residual             0.3024   0.5499          
Number of obs: 144, groups:  plate, 24; sample, 6

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)         22.97       0.8086      28.41     <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Optimizer evaluations: 46, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
plate stddev: 0.8467045342
sample stddev: 1.931557938
loglik (REML crit): -165.43029449553177  reml: True

oracle cross-check vs goldens/penicillin_lmm.json (manifest rung 3):
  beta[Intercept]              got=22.97222222        ref=22.97222222        rel_err=4.76e-14  tol=0.001  [PASS]
  se[Intercept]                got=0.8085732141       ref=0.8085953614       rel_err=2.74e-05  tol=0.001  [PASS]
  plate stddev                 got=0.8467045342       ref=0.8467025103       rel_err=2.39e-06  tol=0.001  [PASS]
  sample stddev                got=1.931557938        ref=1.931613792        rel_err=2.89e-05  tol=0.001  [PASS]
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
Linear mixed model fit by REML [glmm]
Formula: strength ~ (1 | batch/cask)
 Family: gaussian (identity)

REML criterion at convergence: 246.99

Scaled residuals:
   Min      1Q   Median    3Q  Max
 -1.48 -0.5156 0.009498 0.472 1.39

Random effects:
 Groups     Name        Variance Std.Dev. se Corr
 batch      (Intercept) 1.657    1.287    NA     
 batch:cask (Intercept) 8.434    2.904    NA     
 Residual               0.678    0.8234          
Number of obs: 60, groups:  batch, 10; batch:cask, 30

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)         60.05       0.6769      88.72     <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Optimizer evaluations: 31, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
loglik (REML crit): -123.49537292741952

oracle cross-check vs goldens/pastes_lmm.json (manifest rung 4):
  beta[Intercept]              got=60.05333333        ref=60.05333333        rel_err=7.58e-14  tol=0.001  [PASS]
  se[Intercept]                got=0.6768731631       ref=0.6768702151       rel_err=4.36e-06  tol=0.001  [PASS]
  batch stddev                 got=1.287383546        ref=1.287365881        rel_err=1.37e-05  tol=0.001  [PASS]
  batch:cask stddev            got=2.904074666        ref=2.904077466        rel_err=9.64e-07  tol=0.001  [PASS]
  loglik (REML crit)           got=-123.4953729       ref=-123.4953729       abs_err=6.76e-10  tol=2e-06  [PASS]
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
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmm]
Formula: prop ~ period + (1 | herd)
 Family: binomial (logit)

    AIC    BIC  logLik deviance df.resid
 194.05 204.18 -92.026   184.05       51

Scaled residuals:
    Min      1Q  Median     3Q  Max
 -2.382 -0.7888 -0.2024 0.5146 2.88

Random effects:
 Groups Name        Variance Std.Dev. se     Corr
 herd   (Intercept) 0.4125   0.6423   0.1786     
Number of obs: 56, groups:  herd, 15

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)        -1.399       0.2325     -6.016   1.79e-09 ***
period2           -0.9923       0.3066     -3.236    0.00121 **
period3            -1.129       0.3266     -3.455   0.000549 ***
period4             -1.58       0.4274     -3.697   0.000218 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)     period2     period3
period2           -0.364
period3           -0.340       0.278
period4           -0.261       0.212       0.196

Optimizer evaluations: 12, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
herd stddev: 0.6422614489970626
loglik: -92.02628186475712

oracle cross-check vs goldens/cbpp_agq_k1.json (manifest rung 5, nagq=1):
  beta[(Intercept)]            got=-1.398532142       ref=-1.398532044       rel_err=7.07e-08  tol=0.001  [PASS]
  beta[period2]                got=-0.9923327348      ref=-0.9923158803      rel_err=1.7e-05  tol=0.001  [PASS]
  beta[period3]                got=-1.128672084       ref=-1.128664147       rel_err=7.03e-06  tol=0.001  [PASS]
  beta[period4]                got=-1.580313885       ref=-1.580315598       rel_err=1.08e-06  tol=0.001  [PASS]
  se_hessian[(Intercept)]      got=0.2324719924       ref=0.2324732548       rel_err=5.43e-06  tol=0.001  [PASS]
  se_hessian[period2]          got=0.3066424979       ref=0.3066413264       rel_err=3.82e-06  tol=0.001  [PASS]
  se_hessian[period3]          got=0.3266378505       ref=0.3266372426       rel_err=1.86e-06  tol=0.001  [PASS]
  se_hessian[period4]          got=0.4274365866       ref=0.4274372446       rel_err=1.54e-06  tol=0.001  [PASS]
  herd stddev                  got=0.642261449        ref=0.6422698887       rel_err=1.31e-05  tol=0.001  [PASS]
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
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmm]
Formula: TICKS ~ YEAR + HEIGHT + (1 | BROOD)
 Family: poisson (log)

    AIC    BIC  logLik deviance df.resid
 1988.1 2008.1 -989.04   1978.1      398

Scaled residuals:
    Min      1Q  Median     3Q   Max
 -3.044 -0.7817 -0.4198 0.6507 5.647

Random effects:
 Groups Name        Variance Std.Dev. se      Corr
 BROOD  (Intercept) 0.9019   0.9497   0.08458     
Number of obs: 403, groups:  BROOD, 118

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)         11.54        1.399      8.252     <2e-16 ***
YEAR96              1.136       0.2424      4.686   2.78e-06 ***
YEAR97             -1.001       0.2697     -3.712   0.000206 ***
HEIGHT           -0.02387     0.003011     -7.927   2.24e-15 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)      YEAR96      YEAR97
YEAR96            -0.135
YEAR97            -0.152       0.513
HEIGHT            -0.991       0.034       0.064

Optimizer evaluations: 11, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
BROOD stddev: 0.9496951397960821
loglik: -989.0377405264944

(no manifest rung matches this formula -- a run, not an oracle-pinned result)
```

The coefficients on `YEAR96` and `YEAR97` are both contrasts against
`YEAR95` (the first level), not against each other — `YEAR96`'s positive
1.14 and `YEAR97`'s negative -1.00 do not mean 1996 and 1997 are opposite in
sign relative to *each other*; they mean 1996 had noticeably more ticks than
1995, and 1997 noticeably fewer, each measured against the same 1995
baseline.

## 6. Adaptive quadrature (`cbpp` at `nagq=7`)

Recipe 4's model again, but integrated over the random effect with a
7-point adaptive Gauss-Hermite quadrature instead of the `nagq=1` Laplace
default. Eligible here because the model has a single grouping factor
(`herd`) with one random effect per level (`q=1`) — AGQ's current cap is
`q<=3` on a single binomial/Poisson/negative-binomial/Gamma grouping factor.

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
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmm]
Formula: prop ~ period + (1 | herd)
 Family: binomial (logit)

    AIC    BIC  logLik deviance df.resid
 194.05 204.18 -92.026   184.05       51

Scaled residuals:
    Min      1Q  Median     3Q  Max
 -2.382 -0.7888 -0.2024 0.5146 2.88

Random effects:
 Groups Name        Variance Std.Dev. se     Corr
 herd   (Intercept) 0.4125   0.6423   0.1786     
Number of obs: 56, groups:  herd, 15

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)        -1.399       0.2325     -6.016   1.79e-09 ***
period2           -0.9923       0.3066     -3.236    0.00121 **
period3            -1.129       0.3266     -3.455   0.000549 ***
period4             -1.58       0.4274     -3.697   0.000218 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)     period2     period3
period2           -0.364
period3           -0.340       0.278
period4           -0.261       0.212       0.196

Optimizer evaluations: 12, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
loglik: -92.02628186475712

=== nagq=7 (adaptive Gauss-Hermite) ===
Generalized linear mixed model fit by maximum likelihood (Adaptive Gauss-Hermite Quadrature, nAGQ = 7) [glmm]
Formula: prop ~ period + (1 | herd)
 Family: binomial (logit)

    AIC    BIC  logLik deviance df.resid
 193.97 204.09 -91.983   183.97       51

Scaled residuals:
    Min      1Q  Median     3Q   Max
 -2.381 -0.7906 -0.2002 0.5112 2.873

Random effects:
 Groups Name        Variance Std.Dev. se     Corr
 herd   (Intercept) 0.4193   0.6475   0.1805     
Number of obs: 56, groups:  herd, 15

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)        -1.399       0.2335     -5.992   2.07e-09 ***
period2           -0.9914       0.3068     -3.232    0.00123 **
period3            -1.128       0.3268     -3.451   0.000558 ***
period4            -1.579       0.4276     -3.694   0.000221 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)     period2     period3
period2           -0.362
period3           -0.339       0.279
period4           -0.260       0.212       0.197

Optimizer evaluations: 59, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
loglik: -91.98335426919778

Laplace -> AGQ(7) movement on this data:
  beta[(Intercept)]: laplace=-1.398532142 agq7=-1.399232215 delta=-0.0007 rel=0.000501
  beta[period2]: laplace=-0.9923327348 agq7=-0.9914025657 delta=0.00093 rel=0.000937
  beta[period3]: laplace=-1.128672084 agq7=-1.127819278 delta=0.000853 rel=0.000756
  beta[period4]: laplace=-1.580313885 agq7=-1.579468937 delta=0.000845 rel=0.000535
  loglik: laplace=-92.02628186 agq7=-91.98335427 delta=0.0429

oracle cross-check vs goldens/cbpp_agq_k7.json (manifest rung 5 at nagq=7):
(beta, se_hessian, herd stddev only -- see module docstring on why loglik is excluded)
  beta[(Intercept)]            got=-1.399232215       ref=-1.399229755       rel_err=1.76e-06  tol=0.001  [PASS]
  beta[period2]                got=-0.9914025657      ref=-0.9913961245      rel_err=6.5e-06  tol=0.001  [PASS]
  beta[period3]                got=-1.127819278       ref=-1.127832892       rel_err=1.21e-05  tol=0.001  [PASS]
  beta[period4]                got=-1.579468937       ref=-1.579443908       rel_err=1.58e-05  tol=0.001  [PASS]
  se_hessian[(Intercept)]      got=0.2335124545       ref=0.2335116086       rel_err=3.62e-06  tol=0.001  [PASS]
  se_hessian[period2]          got=0.3067679782       ref=0.3067673681       rel_err=1.99e-06  tol=0.001  [PASS]
  se_hessian[period3]          got=0.3267684637       ref=0.3267697654       rel_err=3.98e-06  tol=0.001  [PASS]
  se_hessian[period4]          got=0.4275944626       ref=0.4275906557       rel_err=8.9e-06  tol=0.001  [PASS]
  herd stddev                  got=0.6475210585       ref=0.6475178183       rel_err=5e-06  tol=0.001  [PASS]
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
Linear mixed model fit by REML [glmm]
Formula: angle ~ recipe*temp + (1 | recipe:replicate)
 Family: gaussian (identity)

REML criterion at convergence: 1703.3

Scaled residuals:
   Min      1Q  Median     3Q   Max
 -2.46 -0.5909 -0.0967 0.6103 2.658

Random effects:
 Groups           Name        Variance Std.Dev. se Corr
 recipe:replicate (Intercept) 41.77    6.463    NA     
 Residual                     20.89    4.57            
Number of obs: 270, groups:  recipe:replicate, 45

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)         2.379        5.903     0.4031      0.687 
recipeB            -3.649        8.348    -0.4371      0.662 
recipeC            -1.941        8.348    -0.2325      0.816 
temp               0.1537      0.02821      5.449   5.05e-08 ***
recipeB:temp      0.01086      0.03989     0.2722      0.785 
recipeC:temp     0.002095      0.03989    0.05252      0.958 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
              (Intercept)      recipeB      recipeC         temp recipeB:temp
recipeB            -0.707
recipeC            -0.707        0.500
temp               -0.956        0.676        0.676
recipeB:temp        0.676       -0.956       -0.478       -0.707
recipeC:temp        0.676       -0.478       -0.956       -0.707        0.500

Optimizer evaluations: 12, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.

=== base = B (pandas.Categorical(categories=['B', 'A', 'C'])) ===
Linear mixed model fit by REML [glmm]
Formula: angle ~ recipe*temp + (1 | recipe:replicate)
 Family: gaussian (identity)

REML criterion at convergence: 1703.3

Scaled residuals:
   Min      1Q  Median     3Q   Max
 -2.46 -0.5909 -0.0967 0.6103 2.658

Random effects:
 Groups           Name        Variance Std.Dev. se Corr
 recipe:replicate (Intercept) 41.77    6.463    NA     
 Residual                     20.89    4.57            
Number of obs: 270, groups:  recipe:replicate, 45

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)         -1.27        5.903    -0.2151       0.83 
recipeA             3.649        8.348     0.4371      0.662 
recipeC             1.708        8.348     0.2046      0.838 
temp               0.1646      0.02821      5.834    5.4e-09 ***
recipeA:temp     -0.01086      0.03989    -0.2722      0.785 
recipeC:temp    -0.008762      0.03989    -0.2196      0.826 
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
              (Intercept)      recipeA      recipeC         temp recipeA:temp
recipeA            -0.707
recipeC            -0.707        0.500
temp               -0.956        0.676        0.676
recipeA:temp        0.676       -0.956       -0.478       -0.707
recipeC:temp        0.676       -0.478       -0.956       -0.707        0.500

Optimizer evaluations: 12, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.

Same fit, different parameterization: fitted values and loglik agree (loglik A=-851.63374, loglik B=-851.63374, delta=-6.82e-13); only which contrasts are directly readable off beta changes.

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
`dispersion` *is* phi: maximum likelihood on a gamma GLMM, the Pearson moment
on a gamma GLM or on inverse-Gaussian (GLM only). Overdispersion relative to
Poisson is `1/theta`, so a *large* theta means "close to Poisson", not "a lot
of extra variance".

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
Generalized linear model fit by IRLS [glmm]
Formula: y ~ x + grp
 Family: negativebinomial (log)

    AIC    BIC  logLik deviance df.resid
 1002.7 1017.3 -497.33   994.65      284

Scaled residuals:
     Min      1Q  Median     3Q   Max
 -0.9511 -0.6861 -0.3748 0.3193 6.701

Number of obs: 288

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)        0.1442       0.1207      1.195      0.232 
x                  0.6198      0.07564      8.194   2.53e-16 ***
grpb               0.6337       0.1557       4.07   4.71e-05 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)           x
x                 -0.120
grpb              -0.769       0.043

Shape (theta): 1.01052
Optimizer evaluations: 0, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
theta (NB shape): 1.010521783381409
loglik: -497.3251426865909

oracle cross-check vs goldens/sim_nb_glm.json (GLM, no random effects):
  beta[(Intercept)]            got=0.1441660774       ref=0.1441660779       rel_err=3.6e-09  tol=0.001  [PASS]
  beta[x]                      got=0.6198268694       ref=0.6198268706       rel_err=1.93e-09  tol=0.001  [PASS]
  beta[grpb]                   got=0.6336869002       ref=0.6336868995       rel_err=1.13e-09  tol=0.001  [PASS]
  se[(Intercept)]              got=0.1206905628       ref=0.120690562        rel_err=7.2e-09  tol=0.001  [PASS]
  se[x]                        got=0.07564420094      ref=0.07564420041      rel_err=7.06e-09  tol=0.001  [PASS]
  se[grpb]                     got=0.1557142575       ref=0.1557142563       rel_err=7.85e-09  tol=0.001  [PASS]
  theta                        got=1.010521783        ref=1.010521815        rel_err=3.18e-08  tol=0.001  [PASS]
  loglik                       got=-497.3251427       ref=-497.3251427       abs_err=8.53e-13  tol=0.001  [PASS]
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
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmm]
Formula: y ~ x + (1 | cluster)
 Family: poisson (log)

    AIC    BIC logLik deviance df.resid
 9478.1 9495.4  -4736   9472.1     2397

Scaled residuals:
    Min      1Q  Median    3Q   Max
 -2.533 -0.7337 -0.1417 0.606 5.202

Random effects:
 Groups  Name        Variance Std.Dev. se      Corr
 cluster (Intercept) 0.2777   0.5269   0.06916     
Number of obs: 2400, groups:  cluster, 30

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)        0.3269       0.0969      3.373   0.000743 ***
x                  0.4935     0.009219      53.53     <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)
x                 -0.047

Optimizer evaluations: 15, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
cluster stddev: 0.5269484696264434

=== without the offset (exposure variation folded into the fit) ===
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmm]
Formula: y ~ x + (1 | cluster)
 Family: poisson (log)

   AIC   BIC  logLik deviance df.resid
 12276 12294 -6135.2    12270     2397

Scaled residuals:
    Min     1Q  Median     3Q   Max
 -4.101 -1.057 -0.2177 0.8943 7.854

Random effects:
 Groups  Name        Variance Std.Dev. se      Corr
 cluster (Intercept) 0.2891   0.5377   0.07056     
Number of obs: 2400, groups:  cluster, 30

Fixed effects:
                 Estimate   Std. Error    z value   Pr(>|z|)
(Intercept)         1.323      0.09886      13.38     <2e-16 ***
x                  0.5224     0.009289      56.23     <2e-16 ***
---
Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

Correlation of Fixed Effects:
             (Intercept)
x                 -0.047

Optimizer evaluations: 14, converged: True
z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual df is reported.
cluster stddev: 0.5377163600555648

Dropping a real offset does not just bias the intercept: here the slope on x moves from 0.4935 (with offset) to 0.5224 (without), and the cluster standard deviation moves from 0.5269 to 0.5377 -- the unexplained exposure variation is absorbed by both the fixed and the random-effect side, not cleanly by either alone.

(no goldens/ entry for sim_poisson_offset -- a run, not an oracle-pinned result)
```

The easy misreading: expecting a missing offset to only bias the intercept
(since `offset=log(exposure)` most resembles an intercept shift). It doesn't
stay contained there — dropping it here also moves the slope on `x` (0.494
→ 0.522) and the cluster standard deviation (0.527 → 0.538), because the
model has no other way to explain the exposure-driven part of the count
variation except by routing it through whatever terms are left.
