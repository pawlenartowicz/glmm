# Worked examples (R)

The same nine recipes as [`examples-python.md`](examples-python.md), against
the same datasets, in `fastglmm`. Every code block here is trimmed to the
essential call; the full, runnable script — including the frozen-golden
cross-check that produced the "oracle cross-check" lines in each output
block — lives under
[`documentation/examples/r/`](examples/r/), one file per recipe, same
numbering as the Python page. Every output block on this page is pasted
verbatim from running that script; none of the numbers below were typed by
hand.

**Where the data lives.** Recipes 1–7 use `library(lme4); data(...)` — no
file path, since these are lme4's own bundled datasets (the same CSVs the
[Python examples](examples-python.md) read out of
`validation/data/empirical/`). Recipes 8–9 (negative binomial, an offset
column) have no lme4-bundled equivalent and instead `read.csv()` a fixture
under [`validation/data/simulated/`](../validation/data/simulated/),
generated for the validation harness, with a comment at the point each script
does it.

**Loading both packages.** `lme4` and `fastglmm` both export `fixef`,
`VarCorr`, `isSingular`, and `ranef` as generics. Load `lme4` first (for its
bundled datasets) and `fastglmm` second, so `fastglmm`'s methods are the ones
in scope — R warns about the masking either way, but only that order gives
you the right dispatch:

```r
library(lme4)
library(fastglmm)
```

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

```r
library(lme4)
library(fastglmm)

data(sleepstudy)

fit <- fastglmm(Reaction ~ Days + (1 + Days | Subject), sleepstudy)

summary(fit)
```

Full script, plus the goldens cross-check:
[`examples/r/01_sleepstudy_random_slope.R`](examples/r/01_sleepstudy_random_slope.R).

```
converged: TRUE  singular: FALSE 
Linear mixed model fit by REML [fastglmm] 
Formula: Reaction ~ Days + (1 + Days | Subject)
 Family: gaussian (identity)
Random effects:
 Groups   Name        Std.Dev. Corr  
 Subject  (Intercept) 24.74          
          Days        5.922    0.0656
 Residual             25.59          
Number of obs: 180, groups: Subject, 18
Fixed effects:
(Intercept)        Days 
   251.4051     10.4673 

--- summary ---
Linear mixed model fit by REML [fastglmm] 
Formula: Reaction ~ Days + (1 + Days | Subject)
 Family: gaussian (identity)

Random effects:
 Groups   Name        Std.Dev. Corr  
 Subject  (Intercept) 24.74          
          Days        5.922    0.0656
 Residual             25.59          
Number of obs: 180, groups: Subject, 18

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  251.405      6.825  36.838  < 2e-16 ***
Days          10.467      1.546   6.771 1.27e-11 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 61, converged: TRUE

Subject stddevs: 24.74043 5.922132 
Subject corr[1,2]: 0.06555163 
loglik (REML crit): -871.8141  reml: TRUE 

oracle cross-check vs goldens/sleepstudy_lmm.json (manifest rung 2):
  beta[Intercept]              got=251.4051048        ref=251.4051048        rel_err=0  tol=0.001  [PASS]
  beta[Days]                   got=10.46728596        ref=10.46728596        rel_err=1.26e-14  tol=0.001  [PASS]
  se[Intercept]                got=6.824552619        ref=6.824596695        rel_err=6.46e-06  tol=0.001  [PASS]
  se[Days]                     got=1.545788748        ref=1.545789644        rel_err=5.79e-07  tol=0.001  [PASS]
  Subject sd[Intercept]        got=24.74043132        ref=24.74065799        rel_err=9.16e-06  tol=0.001  [PASS]
  Subject sd[Days]             got=5.922132304        ref=5.922137659        rel_err=9.04e-07  tol=0.001  [PASS]
  loglik (REML crit)           got=-871.814136        ref=-871.814136        abs_err=7.25e-10  tol=2e-06  [PASS]
```

The number a newcomer misreads is the `0.0656` correlation between the
intercept and the slope, on the `Days` row of `summary()`'s random-effects
block. It says subjects who start slower (`(Intercept)`) barely differ from
subjects who start faster in *how fast their reaction time then degrades* —
a small, near-zero correlation, not the two variance components themselves
(24.74 and 5.92, on very different scales already). Reading it as "the slope
is 0.0656" or as a fraction of variance explained is the mistake; it is a
correlation coefficient between two random effects, bounded in [-1, 1].

## 2. Crossed grouping factors (`Penicillin`)

Every `sample` was tested on every `plate` — the two grouping factors are
crossed, not one nested inside the other, so the model needs two independent
`(1 | g)` terms rather than lme4's `/` nesting shorthand (recipe 3 is the
nested case, for contrast).

```r
library(lme4)
library(fastglmm)

data(Penicillin)

# NOTE: no bare "1" fixed-effect term (the intercept is always implicit).
fit <- fastglmm(diameter ~ (1 | plate) + (1 | sample), Penicillin)

summary(fit)
```

Full script, plus the goldens cross-check:
[`examples/r/02_penicillin_crossed.R`](examples/r/02_penicillin_crossed.R).

```
converged: TRUE  singular: FALSE 
Linear mixed model fit by REML [fastglmm] 
Formula: diameter ~ (1 | plate) + (1 | sample)
 Family: gaussian (identity)

Random effects:
 Groups   Name        Std.Dev. Corr
 plate    (Intercept) 0.8467       
 sample   (Intercept) 1.932        
 Residual             0.5499       
Number of obs: 144, groups: plate, 24; sample, 6

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  22.9722     0.8086   28.41   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 48, converged: TRUE

plate stddev: 0.8467044 
sample stddev: 1.931558 
loglik (REML crit): -165.4303  reml: TRUE 

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

```r
library(lme4)
library(fastglmm)

data(Pastes)

fit <- fastglmm(strength ~ (1 | batch/cask), Pastes)

summary(fit)
```

Full script, plus the goldens cross-check:
[`examples/r/03_pastes_nested.R`](examples/r/03_pastes_nested.R).

```
converged: TRUE  singular: FALSE 
Linear mixed model fit by REML [fastglmm] 
Formula: strength ~ (1 | batch/cask)
 Family: gaussian (identity)

Random effects:
 Groups     Name        Std.Dev. Corr
 batch      (Intercept) 1.287        
 batch:cask (Intercept) 2.904        
 Residual               0.8234       
Number of obs: 60, groups: batch, 10; batch:cask, 30

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  60.0533     0.6769   88.72   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 74, converged: TRUE

loglik (REML crit): -123.4954 

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
inside* it. `VarCorr()`'s `Groups` column names them by their factors,
`batch` vs `batch:cask` — the one with the colon is always the finer, nested
grouping, never the coarser one, regardless of which order the print
happens to list them in.

## 4. Aggregated binomial via `weights=` (`cbpp`)

Each row of `cbpp` is a herd-period, not a single animal: `incidence` cases
out of `size` at risk. lme4 writes this as
`cbind(incidence, size - incidence) ~ ...`. The shared parser also accepts
`cbind()` directly with `family = binomial`, but both arguments must be
columns — compute the failures column first (`failures <- size - incidence`;
arithmetic inside `cbind()` itself is not accepted) and pass
`cbind(incidence, failures) ~ ...`. This recipe instead spells the same model
as the success *proportion* as the response plus the trial count as
`weights=` — exactly lme4's objective underneath `cbind()`, spelled
differently.

```r
library(lme4)
library(fastglmm)

data(cbpp)
cbpp$prop <- cbpp$incidence / cbpp$size

fit <- fastglmm(prop ~ period + (1 | herd), cbpp, family = binomial(), weights = size)

summary(fit)
```

Full script, plus the goldens cross-check:
[`examples/r/04_cbpp_binomial_weights.R`](examples/r/04_cbpp_binomial_weights.R).

```
converged: TRUE  singular: FALSE 
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [fastglmm] 
Formula: prop ~ period + (1 | herd)
 Family: binomial (logit)

Random effects:
 Groups Name        Std.Dev. Corr
 herd   (Intercept) 0.6423       
Number of obs: 56, groups: herd, 15

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  -1.3985     0.2325  -6.016 1.79e-09 ***
period2      -0.9923     0.3066  -3.236 0.001212 ** 
period3      -1.1287     0.3266  -3.455 0.000549 ***
period4      -1.5803     0.4274  -3.697 0.000218 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 87, converged: TRUE

herd stddev: 0.6422614 
loglik: -92.02628 

oracle cross-check vs goldens/cbpp_agq_k1.json (manifest rung 5, nAGQ=1):
  beta[(Intercept)]            got=-1.398532066       ref=-1.398532044       rel_err=1.63e-08  tol=0.001  [PASS]
  beta[period2]                got=-0.9923328154      ref=-0.9923158803      rel_err=1.71e-05  tol=0.001  [PASS]
  beta[period3]                got=-1.128672189       ref=-1.128664147       rel_err=7.13e-06  tol=0.001  [PASS]
  beta[period4]                got=-1.580313861       ref=-1.580315598       rel_err=1.1e-06  tol=0.001  [PASS]
  se_hessian[(Intercept)]      got=0.2324738116       ref=0.2324732548       rel_err=2.4e-06  tol=0.001  [PASS]
  se_hessian[period2]          got=0.3066431739       ref=0.3066413264       rel_err=6.02e-06  tol=0.001  [PASS]
  se_hessian[period3]          got=0.3266383585       ref=0.3266372426       rel_err=3.42e-06  tol=0.001  [PASS]
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

```r
library(lme4)
library(fastglmm)

data(grouseticks)

fit <- fastglmm(TICKS ~ YEAR + HEIGHT + (1 | BROOD), grouseticks, family = poisson())

summary(fit)
```

Full script:
[`examples/r/05_grouseticks_poisson.R`](examples/r/05_grouseticks_poisson.R).
**No manifest rung matches this exact formula** — rung 6 (`grouseticks`) fits
the centered `cHEIGHT` against all three crossed grouping factors (`BROOD`,
`INDEX`, `LOCATION`) together, a different model from this recipe's single
`(1 | BROOD)` on raw `HEIGHT`. Dropping two grouping factors and swapping the
height variable changes what each variance component absorbs, so rung 6's
golden numbers are not a valid comparison target for this fit. This output
is a run, not an oracle-pinned result.

```
converged: TRUE  singular: FALSE 
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [fastglmm] 
Formula: TICKS ~ YEAR + HEIGHT + (1 | BROOD)
 Family: poisson (log)

Random effects:
 Groups Name        Std.Dev. Corr
 BROOD  (Intercept) 0.9457       
Number of obs: 403, groups: BROOD, 118

Fixed effects:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept) 11.252422   1.389691   8.097 5.63e-16 ***
YEAR96       1.137235   0.239683   4.745 2.09e-06 ***
YEAR97      -1.017944   0.269038  -3.784 0.000155 ***
HEIGHT      -0.023229   0.002985  -7.781 7.21e-15 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 882, converged: TRUE

BROOD stddev: 0.9457271 
loglik: -989.0649 

(no manifest rung matches this formula -- a run, not an oracle-pinned result)
```

The coefficients on `YEAR96` and `YEAR97` are both contrasts against
`YEAR95` (the first level), not against each other — `YEAR96`'s positive
1.14 and `YEAR97`'s negative -1.02 do not mean 1996 and 1997 are opposite in
sign relative to *each other*; they mean 1996 had noticeably more ticks than
1995, and 1997 noticeably fewer, each measured against the same 1995
baseline.

## 6. Adaptive quadrature (`cbpp` at `nAGQ=7`)

Recipe 4's model again, but integrated over the random effect with a
7-point adaptive Gauss-Hermite quadrature instead of the `nAGQ=1` Laplace
default. Eligible here because the model has a single grouping factor
(`herd`) with one random effect per level (`q=1`) — AGQ's current cap is
`q<=3` on a single binomial/Poisson grouping factor.

```r
fit_laplace <- fastglmm(prop ~ period + (1 | herd), cbpp, family = binomial(), weights = size, nAGQ = 1L)
fit_agq <- fastglmm(prop ~ period + (1 | herd), cbpp, family = binomial(), weights = size, nAGQ = 7L)
```

(`cbpp` is prepared exactly as in recipe 4.) Full script, plus the goldens
cross-check:
[`examples/r/06_cbpp_adaptive_quadrature.R`](examples/r/06_cbpp_adaptive_quadrature.R).
The cross-check below compares beta, `se_hessian`, and the herd standard
deviation only — the crate's own oracle test for this exact golden
(`fit_glmm_binomial_agq_matches_lme4`, `src/fit/glmm_tests.rs`) gates the
same three quantities and deliberately not log-likelihood, because
log-likelihood does not even agree with itself across `nAGQ` in lme4's own
output for this fit (see the reading below).

```
=== nAGQ=1 (Laplace) ===
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [fastglmm] 
Formula: prop ~ period + (1 | herd)
 Family: binomial (logit)

Random effects:
 Groups Name        Std.Dev. Corr
 herd   (Intercept) 0.6423       
Number of obs: 56, groups: herd, 15

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  -1.3985     0.2325  -6.016 1.79e-09 ***
period2      -0.9923     0.3066  -3.236 0.001212 ** 
period3      -1.1287     0.3266  -3.455 0.000549 ***
period4      -1.5803     0.4274  -3.697 0.000218 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 87, converged: TRUE
loglik: -92.02628 

=== nAGQ=7 (adaptive Gauss-Hermite) ===
Generalized linear mixed model fit by maximum likelihood (Adaptive Gauss-Hermite Quadrature, nAGQ = 7) [fastglmm] 
Formula: prop ~ period + (1 | herd)
 Family: binomial (logit)

Random effects:
 Groups Name        Std.Dev. Corr
 herd   (Intercept) 0.6475       
Number of obs: 56, groups: herd, 15

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  -1.3992     0.2335  -5.992 2.07e-09 ***
period2      -0.9914     0.3068  -3.232 0.001230 ** 
period3      -1.1278     0.3268  -3.451 0.000558 ***
period4      -1.5795     0.4276  -3.694 0.000221 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 94, converged: TRUE
loglik: -91.98335 

Laplace -> AGQ(7) movement on this data:
  beta[(Intercept)]: laplace=-1.398532066 agq7=-1.399232148 delta=-0.0007 rel=0.000501
  beta[period2]: laplace=-0.9923328154 agq7=-0.9914028439 delta=0.00093 rel=0.000937
  beta[period3]: laplace=-1.128672189 agq7=-1.127818212 delta=0.000854 rel=0.000757
  beta[period4]: laplace=-1.580313861 agq7=-1.579469194 delta=0.000845 rel=0.000534
  loglik: laplace=-92.02628186 agq7=-91.98335427 delta=0.0429

oracle cross-check vs goldens/cbpp_agq_k7.json (manifest rung 5 at nAGQ=7):
(beta, se_hessian, herd stddev only -- see header comment on why loglik is excluded)
  beta[(Intercept)]            got=-1.399232148       ref=-1.399229755       rel_err=1.71e-06  tol=0.001  [PASS]
  beta[period2]                got=-0.9914028439      ref=-0.9913961245      rel_err=6.78e-06  tol=0.001  [PASS]
  beta[period3]                got=-1.127818212       ref=-1.127832892       rel_err=1.3e-05  tol=0.001  [PASS]
  beta[period4]                got=-1.579469194       ref=-1.579443908       rel_err=1.6e-05  tol=0.001  [PASS]
  se_hessian[(Intercept)]      got=0.2335141792       ref=0.2335116086       rel_err=1.1e-05  tol=0.001  [PASS]
  se_hessian[period2]          got=0.3067686699       ref=0.3067673681       rel_err=4.24e-06  tol=0.001  [PASS]
  se_hessian[period3]          got=0.326768858        ref=0.3267697654       rel_err=2.78e-06  tol=0.001  [PASS]
  se_hessian[period4]          got=0.4275939405       ref=0.4275906557       rel_err=7.68e-06  tol=0.001  [PASS]
  herd stddev                  got=0.6475205677       ref=0.6475178183       rel_err=4.25e-06  tol=0.001  [PASS]
```

The thing a newcomer misreads: the answer *does* move relative to Laplace,
but by very little on this data — every beta shifts by under 0.1% and the
herd standard deviation moves from 0.642 to 0.648, exactly the direction
theory predicts (Laplace's variance-component bias runs low). What is easy
to over-read is `loglik`: it shifts by only 0.043 within `fastglmm`'s own
two fits above, but if you go compare against lme4's own `nAGQ=1` vs
`nAGQ=7` refits of this exact model, lme4's *own* reported log-likelihood
jumps from -92.0 to -50.0 — a ~42-unit change despite beta barely moving.
That is an artifact of how lme4 computes `logLik()` at different quadrature
orders for an aggregated-trials binomial, not a sign that either fit is
wrong, and it's exactly why this recipe's cross-check does not touch
log-likelihood at all.

## 7. Factors and interactions (`cake`), and changing the base level

`recipe*temp` desugars to `recipe + temp + recipe:temp`: a main effect per
recipe, a slope on `temp`, and a per-recipe deviation from that slope.
Treatment contrasts code `recipe` against its first level — `"A"`, cake's
factor level order as shipped — and `relevel()` is how you pick a different
base without touching the data; there is deliberately no `contrasts=`
argument.

```r
library(lme4)
library(fastglmm)

data(cake)

fit_a <- fastglmm(angle ~ recipe * temp + (1 | recipe:replicate), cake)

cake_b <- cake
cake_b$recipe <- relevel(cake_b$recipe, ref = "B")
fit_b <- fastglmm(angle ~ recipe * temp + (1 | recipe:replicate), cake_b)
```

Full script:
[`examples/r/07_cake_factors_interactions.R`](examples/r/07_cake_factors_interactions.R).
**No manifest rung's golden covers this dataset** — `cake` is manifest rung
13, but no lme4 reference JSON was ever frozen under `validation/goldens/`
for it. This output is a run, not an oracle-pinned result.

```
=== base = A (cake's factor level order as shipped) ===
Linear mixed model fit by REML [fastglmm] 
Formula: angle ~ recipe * temp + (1 | recipe:replicate)
 Family: gaussian (identity)

Random effects:
 Groups           Name        Std.Dev. Corr
 recipe:replicate (Intercept) 6.463        
 Residual                     4.57         
Number of obs: 270, groups: recipe:replicate, 45

Fixed effects:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept)   2.379365   5.902802   0.403    0.687    
recipeB      -3.649206   8.347822  -0.437    0.662    
recipeC      -1.941270   8.347822  -0.233    0.816    
temp          0.153714   0.028207   5.449 5.05e-08 ***
recipeB:temp  0.010857   0.039891   0.272    0.785    
recipeC:temp  0.002095   0.039891   0.053    0.958    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 16, converged: TRUE

=== base = B (relevel(cake$recipe, ref = "B")) ===
Linear mixed model fit by REML [fastglmm] 
Formula: angle ~ recipe * temp + (1 | recipe:replicate)
 Family: gaussian (identity)

Random effects:
 Groups           Name        Std.Dev. Corr
 recipe:replicate (Intercept) 6.463        
 Residual                     4.57         
Number of obs: 270, groups: recipe:replicate, 45

Fixed effects:
              Estimate Std. Error z value Pr(>|z|)    
(Intercept)  -1.269841   5.902802  -0.215    0.830    
recipeA       3.649206   8.347822   0.437    0.662    
recipeC       1.707937   8.347822   0.205    0.838    
temp          0.164571   0.028207   5.834  5.4e-09 ***
recipeA:temp -0.010857   0.039891  -0.272    0.785    
recipeC:temp -0.008762   0.039891  -0.220    0.826    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 15, converged: TRUE

Same fit, different parameterization: fitted values and loglik agree (loglik A=-851.63374, loglik B=-851.63374, delta=1.02e-12); only which contrasts are directly readable off fixef() changes.

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
of: `dispersion` on the returned object is theta-hat, the NB shape parameter
(`MASS::glm.nb`'s `theta`), not phi — unlike `Gamma()`, where `dispersion`
*is* the Pearson phi. Overdispersion relative to Poisson is `1/theta`, so a
*large* theta means "close to Poisson", not "a lot of extra variance".

```r
library(fastglmm)

sim_nb <- read.csv("validation/data/simulated/sim_nb.csv")

fit <- fastglmm(y ~ x + grp, sim_nb, family = "negativebinomial")

summary(fit)
```

Full script, plus the goldens cross-check:
[`examples/r/08_negative_binomial.R`](examples/r/08_negative_binomial.R). No
lme4-bundled dataset exercises negative binomial, so this reads
`validation/data/simulated/sim_nb.csv` rather than `library(lme4);
data(...)` — a fixture generated for the validation harness.

```
converged: TRUE  singular: FALSE 
Generalized linear model fit by IRLS [fastglmm] 
Formula: y ~ x + grp
 Family: negativebinomial (log)

Number of obs: 288 

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept)  0.14417    0.12069   1.195    0.232    
x            0.61983    0.07564   8.194 2.53e-16 ***
grpb         0.63369    0.15571   4.070 4.71e-05 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Shape (theta): 1.011
Optimizer evaluations: 0, converged: TRUE
theta (NB shape): 1.010522 
loglik: -497.3251 

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

```r
library(fastglmm)

sim_poisson_offset <- read.csv("validation/data/simulated/sim_poisson_offset.csv")
sim_poisson_offset$cluster <- factor(sim_poisson_offset$cluster)

fit_with <- fastglmm(y ~ x + (1 | cluster), sim_poisson_offset,
                      family = poisson(), offset = log_exposure)
fit_without <- fastglmm(y ~ x + (1 | cluster), sim_poisson_offset, family = poisson())
```

Full script:
[`examples/r/09_poisson_offset.R`](examples/r/09_poisson_offset.R). Data:
`validation/data/simulated/sim_poisson_offset.csv`, a fixture generated for
the validation harness with a `log_exposure` column already computed —
`offset=` takes the log-exposure directly, not the raw exposure. No
`goldens/` entry covers this dataset (it is registered at manifest rung 28
with a real `offset` field, but no lme4 reference JSON was frozen for it).
This output is a run, not an oracle-pinned result.

```
=== with offset = log(exposure) ===
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [fastglmm] 
Formula: y ~ x + (1 | cluster)
 Family: poisson (log)

Random effects:
 Groups  Name        Std.Dev. Corr
 cluster (Intercept) 0.5269       
Number of obs: 2400, groups: cluster, 30

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept) 0.326851   0.096902   3.373 0.000743 ***
x           0.493545   0.009219  53.533  < 2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 47, converged: TRUE
cluster stddev: 0.5269486 

=== without the offset (exposure variation folded into the fit) ===
Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [fastglmm] 
Formula: y ~ x + (1 | cluster)
 Family: poisson (log)

Random effects:
 Groups  Name        Std.Dev. Corr
 cluster (Intercept) 0.5377       
Number of obs: 2400, groups: cluster, 30

Fixed effects:
            Estimate Std. Error z value Pr(>|z|)    
(Intercept) 1.323181   0.098860   13.38   <2e-16 ***
x           0.522360   0.009289   56.23   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
Optimizer evaluations: 50, converged: TRUE
cluster stddev: 0.537718 

Dropping a real offset does not just bias the intercept: here the slope on x moves from 0.4935 (with offset) to 0.5224 (without), and the cluster standard deviation moves from 0.5269 to 0.5377 -- the unexplained exposure variation is absorbed by both the fixed and the random-effect side, not cleanly by either alone.

(no goldens/ entry for sim_poisson_offset -- a run, not an oracle-pinned result)
```

The easy misreading: expecting a missing offset to only bias the intercept
(since `offset=log(exposure)` most resembles an intercept shift). It doesn't
stay contained there — dropping it here also moves the slope on `x` (0.494
→ 0.522) and the cluster standard deviation (0.527 → 0.538), because the
model has no other way to explain the exposure-driven part of the count
variation except by routing it through whatever terms are left.
