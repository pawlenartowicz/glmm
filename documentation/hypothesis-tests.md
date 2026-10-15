# Hypothesis tests: `anova()` and `compare()`

Three calls test hypotheses about a fitted model, in both ports:

| Call | R | Python | Takes | Refits? |
|---|---|---|---|---|
| per-term tests | `anova(m)` | `glmm.anova(m)` | one fit | no: reads the tables the fit stored when it was made |
| two fits compared | `anova(m0, m1)` | `glmm.anova(m0, m1)` | two existing fits | no, and reads no data |
| two formulas compared | `compare(data, f0, f1)` | `glmm.compare(data, f0, f1)` | one data set, two formulas | fits both models itself |

The Rust crate computes every statistic and every p-value; the two ports only
lay the numbers out, with R's column names, so both give the same tables. A
comparison always takes two models: a third fit or formula is an error. The
names of these calls and of their result classes are provisional until 1.0.0.

## The per-term tests of one fit: `anova(m)`

`anova(m)` gives one row per fixed term. The rows are built when the model is
fitted and stored on the fit object, so the call computes nothing and refits
nothing: F and Wald rows come from that fit alone, likelihood-ratio rows from
refits made with it
([below](#wald-or-likelihood-ratio-rows-test-and-anova_test)). There are no
random-effect rows: a variance component is tested by comparing two models
(below).

```r
m <- fastglmm(Reaction ~ Days + (1 + Days | Subject), sleepstudy)
anova(m)                 # Type III, the default
anova(m, type = "II")
```

```python
m = glmm.fit(data, "Reaction ~ Days + (1 + Days | Subject)")
glmm.anova(m)            # a TermTable; prints as text
glmm.anova(m, type="II")
```

```
Type III Analysis of Variance Table with Satterthwaite's method
     Sum Sq Mean Sq NumDF DenDF F value    Pr(>F)
Days  30031   30031     1    17  45.853 3.264e-06 ***
```

Which test a row gets depends on the model, as in lmerTest, `lm` and `glm`:

| Model | Row | Columns |
|---|---|---|
| Gaussian mixed model (REML or ML) | F on lmerTest's Satterthwaite denominator df (its multi-df construction) | `Sum Sq`, `Mean Sq`, `NumDF`, `DenDF`, `F value`, `Pr(>F)` |
| OLS; Gamma and inverse-Gaussian GLMs with φ estimated | F on the residual df `n − p` | the same |
| binomial, Poisson and negative-binomial GLMs, a Gamma or inverse-Gaussian GLM with φ held, every GLMM but a Poisson one | Wald χ² on the rank of the term's hypothesis | `Chisq`, `Df`, `Pr(>Chisq)` |
| Poisson GLMM | likelihood-ratio χ² on the rank of the term's hypothesis, from a refit of the model under it | the same |

These are the rows of the default, `test = "auto"`; on a fit with χ² rows,
`test` picks the other test
([below](#wald-or-likelihood-ratio-rows-test-and-anova_test)).

The heading names the type and the method, for example "Type III Analysis of
Variance Table with Satterthwaite's method", or "Type III Analysis of
Variance Table" on an OLS fit or a GLM with φ estimated. A χ² table's heading
is "Type III Analysis of Deviance Table" with a second line naming the test
and why it ran (below), or the one line "Type III Analysis of Deviance Table
(Wald χ² tests)" when `test = "Wald"` was asked for.
Each row's reference distribution follows from its own df columns, `F(NumDF,
DenDF)` or `χ²(Df)`, so the table carries no separate reference column (R's
`print.anova` would print a text column as codes). Python's html, latex and
typst renders add a `Reference` column with that text. `Sum Sq` is
`F · σ̂² · NumDF` and `Mean Sq` is `F · σ̂²` (φ̂ in place of σ̂² on the Gamma
and inverse-Gaussian GLMs), lmerTest's columns. In R the
table is a `c("anova", "data.frame")` with a `heading` attribute, so `$`
indexing and `broom::tidy()` work (on an F table broom keeps `NumDF` and
`DenDF` under their own names and warns that it did not recognize them, as it
does on lmerTest's table); in Python a `TermTable` is indexed by
column name (`tab["Pr(>F)"]`), and `to_pandas()` returns a DataFrame (pandas
is imported only then).

**Type III** (the default) is built the way lmerTest builds it: each term's
hypothesis starts from the term's full indicator coding, expressed in the
fit's coefficients, and is made orthogonal to the terms that contain it, from
the highest-order term down. This makes the test independent of the contrast
coding. It matters here because glmm codes factors with treatment contrasts
only: car's `Anova(type = 3)` on a treatment-coded fit tests a main effect at
the reference level of the other factor, which is a different hypothesis.
A term with an empty cell carries lmerTest's note, "Missing cells for: ….
Interpret type III hypotheses with care."

**Type II** (`type = "II"`) is built in one of two metrics, chosen by model:

- the Gaussian mixed model uses lmerTest's construction, from `XᵀX`
  (`anova(type = 2)` on an `lmer` fit);
- OLS, every GLM, every GLMM and the Gamma and inverse-Gaussian GLMs use
  car's, from the fit's own covariance matrix (`car::Anova(type = 2)` on an
  `lm`, `glm` or `glmer` fit).

The two coincide on OLS without weights, where the covariance is proportional
to `(XᵀX)⁻¹`, but not on OLS with prior weights, where it is proportional to
`(XᵀWX)⁻¹`; they differ on a mixed model (on one unbalanced design, a one-df
term's statistic is 9.75 with lmerTest's metric, an F, and 10.00 with car's, a
Wald χ²; on one df the two share a scale, so the gap is the metric alone). Each
table matches the tool an R user would reach for on that model. Type I is not
offered.

**Row order** is R's term order: by degree, then by position in the formula.
A term whose columns are all aliased keeps its place, with a NaN row and a
note naming each aliased column (lmerTest moves such a row last).

**Interactions need their margins.** A formula whose interaction lacks one of
its marginal terms (`y ~ x + x:g` lacks `g`) is fitted as usual, but gets no
table: `anova(m)` stops (Python: `ValueError`) with "per-term tests need every
marginal term of each interaction in the formula; it lacks `g` (a margin of
`x:g`)". Both constructions assume the margins are present; without them the
rows would test hypotheses that depend on the coding. Add the missing terms.

A Gaussian mixed model fitted with `satterthwaite = FALSE` (Python
`satterthwaite=False`), or one whose Satterthwaite pass failed, has no
denominator df: its F rows are NA (NaN) with the reason, never a z test. A
fit that did not converge gets its table with a first note saying so.

### Wald or likelihood-ratio rows: `test` and `anova_test`

On a fit whose rows are χ² tests (every fit but the F fits above), each row
can be the Wald test or the likelihood-ratio test of its term. Two arguments,
spelled the same in both ports, choose which:

| Argument | Call | Values | What it chooses |
|---|---|---|---|
| `test` | `anova(m, test = )`; Python `glmm.anova(m, test=)` | `"auto"` (default), `"Wald"`, `"LRT"` | which of the stored tables `anova(m)` returns |
| `anova_test` | `fastglmm(..., anova_test = )`; Python `glmm.fit(..., anova_test=)` | `"auto"` (default), `"Wald"`, `"LRT"` | which tables the fit stores |

```r
data(grouseticks, package = "lme4")
grouseticks$cHEIGHT <- grouseticks$HEIGHT - mean(grouseticks$HEIGHT)
m <- fastglmm(TICKS ~ YEAR + cHEIGHT + (1 | BROOD), grouseticks, family = poisson)
anova(m)                  # a Poisson GLMM: likelihood-ratio rows by default
anova(m, test = "Wald")   # the Wald rows of the same fit

data(cbpp, package = "lme4")
cbpp$failures <- cbpp$size - cbpp$incidence
b <- fastglmm(cbind(incidence, failures) ~ period + (1 | herd), cbpp,
              family = binomial, anova_test = "LRT")
anova(b)                  # a binomial GLMM: Wald rows by default
anova(b, test = "LRT")    # stored because the fit asked for them
```

```python
# grouseticks and cbpp: dicts of the columns the R block builds (cHEIGHT, failures)
m = glmm.fit(grouseticks, "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)", "poisson")
glmm.anova(m)
glmm.anova(m, test="Wald")
b = glmm.fit(cbpp, "cbind(incidence, failures) ~ period + (1 | herd)", "binomial",
             anova_test="LRT")
glmm.anova(b, test="LRT")
```

R prints the first table as (Python prints the same rows with fewer digits):

```
Type III Analysis of Deviance Table
Likelihood-ratio chi-squared tests (each term's model refitted without it) on the reduced design; test = "auto": Poisson model with random effects
         Chisq Df Pr(>Chisq)
YEAR    60.399  2  7.665e-14 ***
cHEIGHT 52.512  1  4.276e-13 ***
```

The Wald rows of the same fit are 72.800 (`YEAR`) and 62.839 (`cHEIGHT`).

**What `"auto"` picks.** One test for the whole table, chosen from the family
and whether the model has random effects, never from the size of the data:

- a Poisson model with random effects: the likelihood-ratio rows;
- every other model with χ² rows: the Wald rows. That is a binomial GLMM
  (Bernoulli or with trials), a negative binomial GLMM, a GLMM of any other
  family (such as Gamma), and every GLM;
- a model with F rows (a Gaussian mixed model, OLS, a Gamma or
  inverse-Gaussian GLM with φ estimated): its F rows.

The measurements behind this choice are in
[How well calibrated are these tests](#how-well-calibrated-are-these-tests).

**The heading says which test ran.** Its first line is "Type III Analysis of
Deviance Table" ("Type II …" with `type = "II"`); the second line is:

- `test = "auto"` on a Poisson GLMM: `Likelihood-ratio chi-squared tests
  (each term's model refitted without it) on the reduced design; test =
  "auto": Poisson model with random effects`;
- `test = "auto"` on any other fit with χ² rows: `Wald chi-squared tests;
  test = "auto": ` and the reason, one of "binomial model with random
  effects", "negative binomial model with random effects", "model with random
  effects, not binomial, Poisson or negative binomial" and "model without
  random effects";
- `test = "LRT"`: `Likelihood-ratio chi-squared tests (each term's model
  refitted without it) on the reduced design; test = "LRT"`;
- `test = "Wald"`: none; the heading is the one line "Type III Analysis of
  Deviance Table (Wald χ² tests)".

A Type II likelihood-ratio heading has no " on the reduced design". Each
second line is one line of text, wrapped here. Python's html, latex and typst
renders print both lines.

**How a likelihood-ratio row is built.** Each row compares the fitted model
with the model refitted under the term's hypothesis: the statistic is
`2(ℓ₁ − ℓ₀)`, floored at 0, with ℓ₁ the fit's log-likelihood and ℓ₀ the
refit's, referred to χ² on the term's df (the Wald row's `Df`).

- **Type III** refits the model on the reduced design: the fit's design
  restricted to the null space of the hypothesis its Wald row tests. Like the
  Wald row, the test does not depend on the contrast coding. These are the
  rows afex's `mixed(method = "LRT")` gives at its default `type = 3` (afex
  sets sum-to-zero contrasts), and on a GLM those of car's `Anova(type = 3,
  test.statistic = "LR")` on a fit with sum-to-zero contrasts.
- **Type II** follows the rule of car's `Anova(type = 2, test.statistic =
  "LR")`: a term is tested in the model of every term that does not contain
  it, by fitting that model with and without the term. car computes these
  rows on a GLM only (on a `glmer` fit it gives only Wald χ² rows); glmm
  applies the same rule to GLMMs, and those rows are checked against pairs
  of lme4 `glmer` fits that follow it. afex's `type = 2` is sequential and
  can differ: on `y ~ g * h + x` its row `x` compares `g + h + x` with
  `g + h`, leaving `g:h` out, where car and glmm compare `g * h + x` with
  `g * h`.
- Each refit starts from the fit's estimates and uses the fit's own settings:
  the Laplace approximation or adaptive quadrature with the same node count,
  the weights, the offset and a held dispersion; a negative binomial shape is
  estimated again. A reduced model that both types need is fitted once.
- The table keeps the Wald table's columns, `Chisq`, `Df` and `Pr(>Chisq)`
  (car names its statistic `LR Chisq`), so only the heading tells a
  likelihood-ratio table from a Wald one.
- A term whose columns carry a random slope keeps its Wald row, with a note:
  the refit cannot leave out a column the random part uses. A refit that does
  not converge makes its row NaN, with a note; on a fit that did not converge
  nothing is refitted and every row is NaN. The notes are listed
  [below](#the-notes-a-table-can-carry).

**When the refits are made: `anova_test`.** The likelihood-ratio rows are
built when the model is fitted, as the Wald rows are, so `anova(m)` still
computes nothing. Every fit stores its Wald (or F) tables. `anova_test =
"auto"` (the default) also stores the likelihood-ratio tables of a Poisson
GLMM, whose default table they are; `"LRT"` stores them on every fit with χ²
rows; `"Wald"` stores none. The fit itself (estimates, standard errors,
log-likelihood) is the same under every value. What changes is the time:
each reduced model is one more fit, so a default fit of a Poisson GLMM takes
longer than the same fit with `anova_test = "Wald"`. `compare()` makes its
two fits with `anova_test = "Wald"`, so they store no likelihood-ratio
tables: `anova()` of a Poisson GLMM it fitted (`cmp$fit1`, Python
`cmp.fit1`) needs `test = "Wald"`, and the likelihood-ratio rows come from
`fastglmm(..., anova_test = "LRT")` (Python
`glmm.fit(..., anova_test="LRT")`). On a fit with F rows, `anova_test =
"LRT"` leaves the fit as it is, and `anova(m, test = "LRT")` is refused.

**What `test` refuses.** Each refusal names its fix; the texts are in
[`warnings.md`](warnings.md#test-and-comparison-errors).

- `test = "LRT"` on a fit that stored no likelihood-ratio tables: refit with
  `anova_test = "LRT"`.
- `test = "auto"` on a Poisson GLMM fitted with `anova_test = "Wald"`: an
  error, not a silent switch to Wald. Ask for `test = "Wald"`, or refit with
  `anova_test = "auto"`.
- `test = "Wald"` or `test = "LRT"` on a fit with F rows: leave `test` at its
  default.
- `test` with two fits: a comparison's test follows from the two models.

**Simulations: name the test in both calls.** `"auto"` reads only the family
and whether the model has random effects, so one model gets the same test at
every sample size. A simulation that fits more than one kind of model (a
power analysis such as MCPower's over Poisson and binomial GLMMs, or a GLMM
against its GLM) gets likelihood-ratio rows for some models and Wald rows for
others, under the same column names. To keep one test, name it at fit time
and in `anova()`: `anova_test = "Wald"` with `test = "Wald"` (no refits, the
fastest), or `anova_test = "LRT"` with `test = "LRT"`.

## Comparing two models

### `compare(data, f0, f1)`: two formulas

`compare()` takes one data set and two formulas, fits both models on the same
rows, and compares them. It returns the comparison table and the two fits: in
R a `fastglmm_comparison` list with `$table`, `$fit0` and `$fit1`; in Python a
`Comparison` with `table`, `fit0` and `fit1`. `fit0` is the fit of `f0` and
`fit1` that of `f1`, as given; the table lists the smaller model first, under
its own label (`f0` or `f1`), so the formulas can be given in either order.

```r
cmp <- compare(sleepstudy, Reaction ~ Days, Reaction ~ Days + (1 | Subject))
cmp            # prints the table
cmp$fit1       # the fastglmm fit of the second formula
```

```python
cmp = glmm.compare(data, "Reaction ~ Days", "Reaction ~ Days + (1 | Subject)")
cmp.table["Pr(>Chisq)"]
cmp.fit1
```

```
Model comparison: likelihood ratio test on the REML criterion
f0: Reaction ~ Days
f1: Reaction ~ Days + (1 | Subject)
Reference: exact RLRT null distribution (Crainiceanu & Ruppert 2004), 10000 draws.
   npar  logLik Chisq Df Pr(>Chisq)
f0    3 -946.83
f1    4 -893.23 107.2     9.999e-05 ***
```

`family` (Python also `link`), `weights`, `offset` and `nAGQ` (Python `nagq`)
are given once and used for both fits; each fit is then made as `fastglmm()` (Python
`glmm.fit`) makes it with those arguments. Rows are dropped once, for both
fits: a row with a missing value in any variable of either formula, in the
weights or in the offset (where `fastglmm()` refuses a missing weight or
offset). The warning says so: "Dropped `<n>` of `<m>` row(s): a column either
formula uses, the weights or the offset had a missing value there."

There is no `REML` argument: the estimator follows from what differs between
the two formulas.

| What differs | Both models fitted by | Test |
|---|---|---|
| the random part only, Gaussian | REML (the model with no random effects fitted with `REML = TRUE`) | likelihood-ratio test on the REML criterion; reference by the boundary rule below |
| the fixed part only, Gaussian mixed model | ML | likelihood-ratio test, χ²(Δnpar), with `AIC` and `BIC` columns |
| both parts, Gaussian | ML | likelihood-ratio test, reference by the boundary rule, with `AIC` and `BIC` |
| the fixed part, two OLS models | least squares | `anova.lm`'s exact F |
| the fixed part, two Gamma or inverse-Gaussian GLMs with φ estimated | ML | Wald F on `(r, n − p)` |
| any other family (GLM or GLMM) | ML (Laplace or AGQ for a GLMM) | likelihood-ratio test labelled with the objective: χ²(Δnpar) when the random parts are equal, else the boundary rule |

A Gaussian pair fitted by ML (not two OLS fits) carries the note "Both models
were fitted by ML because their fixed parts differ." When `nAGQ` applies to one model and not
the other (a second grouping in `f1`), the other falls back to Laplace with
the usual warning, and the comparison notes the two node counts.

### `anova(m0, m1)`: two existing fits

`anova(m0, m1)` compares two fits as they are, with no refit and without
reading their data, in either order. It gives the same test and p-value
`compare()` gives when the two fits were made the way `compare()` makes them,
the exact p-value included (same `nsim` and `seed`); the rows are labelled `m0`
and `m1`, and the note that both models were fitted by ML is `compare()`'s
alone. In R the result is the classic
`c("anova", "data.frame")` with a `heading` attribute and the attributes
`nsim` and `seed`; in Python a `ComparisonTable`, indexed by column name, with
`nsim`, `seed`, `reference` and `notes`.

Before any test, the two fits must pass these checks, in this order; each
failure is an error that names the fix:

1. both fits converged;
2. the same data, as far as the fits can tell: the same row count, a
   bitwise-equal response, equal weights and offsets, the same family and
   link, and the same held dispersion. A predictor changed under the same name
   while the response stayed the same is not caught;
3. one formula nested in the other (below);
4. the estimator: two OLS fits pass whatever their `REML`; otherwise one REML
   fit and one ML fit are refused, and so are two REML fits with different
   fixed parts;
5. something to test: the two models differ in their parameter count.

Then the pair decides the test:

| The two fits | Test |
|---|---|
| two OLS fits (any `REML`) | `anova.lm`'s exact F on `(p₁ − p₀, n − p₁)`, with `anova.lm`'s columns `Res.Df`, `RSS`, `Df`, `Sum of Sq`, `F`, `Pr(>F)` |
| two Gamma or inverse-Gaussian GLMs with φ estimated | Wald F of the larger fit's extra columns on `(r, n − p₁)` (columns `Res.Df`, `Df`, `F`, `Pr(>F)`) |
| any other pair of ML fits, every GLM and GLMM pair included | likelihood-ratio test, with lme4's columns `npar`, `AIC`, `BIC`, `logLik`, `Chisq`, `Df`, `Pr(>Chisq)` |
| two REML fits with the same fixed part | likelihood-ratio test on the REML criterion, with no `AIC` or `BIC` column |
| two REML fits with different fixed parts, not two OLS fits | refused |
| one REML fit and one ML fit, not two OLS fits | refused, naming which is which |

**Two REML fits with different fixed parts are an error here, where lme4
refits both with ML.** The REML criterion depends on the fixed-effects
design, so two such criteria do not compare; lme4's `anova()` refits both
models by ML and says so in a message (and it refits even when the fixed
parts are the same, unless `refit = FALSE`). glmm never refits a fit you
hold. `compare(data, f0, f1)` fits both models by ML when their fixed parts
differ, or refit both yourself with `REML = FALSE` (Python `reml=False`).

**Testing a random intercept against no random effects.** For `y ~ x` against
`y ~ x + (1 | g)` on Gaussian data, the smaller model's log-likelihood must be
its REML one, so make the OLS fit with `REML = TRUE` (Python `reml=True`):

```r
m0 <- fastglmm(y ~ x, d, REML = TRUE)
m1 <- fastglmm(y ~ x + (1 | g), d)        # REML by default
anova(m0, m1)
```

An OLS fit made with the default reports the ML log-likelihood and counts as
an ML fit, so beside a REML mixed fit it is refused as "one REML, one ML".
`compare()` makes the OLS fit with `REML = TRUE` itself. (lmerTest's
`ranova()` uses the same REML value; lme4's `anova(lmer, lm)` uses the `lm`
fit's ML log-likelihood.)

**Nesting** is read from the two formulas alone, with no data and no
numeric check. One formula is nested in the other when the response is the
same, every fixed term of the smaller one is in the larger one, and every
grouping of the smaller one is in the larger one with the same or more
slopes. Terms, groupings and slopes are compared as sets of variable names,
so no order matters (`a:b` is `b:a`). The fixed intercept is a term:
`y ~ 0 + x` is nested in `y ~ x`. One consequence: `y ~ 1` against
`y ~ 0 + f` is refused, although with a factor `f` the column spaces are
nested.

### The statistic

The likelihood-ratio statistic is `T = max(0, 2(ℓ₁ − ℓ₀))`, with 1 the larger
model and 0 the smaller (whatever the argument order), as lme4 floors it.
`Δnpar` is the difference of the parameter counts (`npar`, the AIC count).

- **A raw value below zero** beyond round-off (`8e-5`) means the larger fit
  did not reach its optimum. The statistic is still 0, and a note names that
  fit, shows the raw value and says to refit it.
- **A dropped component at exactly zero.** When the random parts alone differ
  and every variance the smaller model drops is exactly 0 in the larger fit
  (the fit pinned it on the boundary), the larger optimum lies in the smaller
  model's space, and `T` is exactly 0, so p = 1 on every reference, whatever
  the round-off between the two log-likelihoods. Without this rule a
  statistic of `1e-14` would give p near ½ under the mixture. A correlation
  at ±1 with a non-zero slope variance is not this case.
- **A failed fit never reads as p = 1.** A log-likelihood that is NaN or `−∞`
  on either side makes the statistic and the p-value NaN, with one note naming
  the side; the notes about the quality of the p-value are dropped then.

### The reference: the boundary rule

A variance component tested at zero sits on the boundary of its parameter
space, so the χ² reference of the likelihood-ratio test does not hold there.
The reference follows how the two random parts differ, for REML and ML pairs,
Gaussian or not:

| The random parts | Reference |
|---|---|
| are the same | χ²(Δnpar) |
| differ by one dimension, inside the exact test's envelope (below) | the exact RLRT null distribution |
| differ by one dimension, otherwise | the 50:50 mixture of χ²(d − 1) and χ²(d), `d = Δnpar` |
| differ by more than one dimension | χ²(Δnpar), labelled "ignores the boundary; conservative" |

One dimension is one random effect: `(1 | g)` added to a model, or one slope
added to a grouping, with its covariances (`(1 | g)` against `(1 + x | g)`
drops one variance and one covariance, `d = 2`, so the mixture is of χ²(1)
and χ²(2)). When the fixed parts differ too, the fixed columns that differ
count in `d − 1`. χ²(0) is a point mass at zero.

**The exact restricted likelihood-ratio test** (Crainiceanu & Ruppert 2004)
simulates the finite-sample null of the REML statistic from the spectral
representation of the design, with no refit. It applies when both fits are
Gaussian REML fits (the smaller an OLS fit made with `REML = TRUE`), the
larger model's only random term is one random intercept `(1 | g)` (an
interaction grouping such as `(1 | a:b)` counts as one), the larger fit has
no prior weights (weights all equal to 1 count as weights) and at least one
fixed-effect column (`p ≥ 1`), the two fixed parts are the same, and at most
100 levels of the grouping factor have rows. An offset is allowed. Outside that
envelope a pair that differs by one dimension gets the 50:50 mixture, with a
note that says why. The method is in
[`algorithms-lmm.md`](algorithms-lmm.md#exact-rlrt-for-one-random-intercept).

- `nsim` (default 10000, from 1 to 1,000,000) is the number of simulated
  draws; the p-value is `(1 + #{draws ≥ T}) / (nsim + 1)`, so it is never
  below `1/(nsim + 1)`, and its Monte Carlo standard error near p = 0.05 is
  about 0.002 at the default.
- `seed` has a fixed default (20261008), so the same call gives the same
  p-value on every run and in both ports; R's `set.seed()` does not change
  it. R takes a whole number from 0 to 2^53 (a double holds integers exactly
  up to there), Python one from 0 to 2^64 − 1.
- At `T = 0` the p-value is 1 and nothing is drawn; the reference then says
  "no draws needed". The table has no `Df` for this reference.

`nsim` and `seed` apply to the two-model calls only (an error with one fit).

**Which reference is which, in the heading.** The heading names the test:
"likelihood ratio test on the REML criterion", "(ML)", "(Laplace)" or
"(AGQ k)" for a GLMM, or "F test". A GLMM pair is labelled with the larger
model's objective; two mixed fits with different node counts are compared,
with the note "The two fits use different quadrature node counts, k₀ and k₁;
the likelihood-ratio test mixes two approximation errors." A fit without
random effects against a GLMM gets no such note.

**The F pairs.** Two OLS fits get `anova.lm`'s F from the two residual sums
of squares, so `y ~ 0 + f` against `y ~ f + x` works too. Two Gamma or
inverse-Gaussian GLMs with φ estimated get the Wald F of the larger fit's
columns whose names the smaller fit lacks, `F = β̂ₛᵀ V̂ₛₛ⁻¹ β̂ₛ / r` on
`(r, n − p₁)`, all from the larger fit, as `car::linearHypothesis(test = "F")`
gives it. When the smaller fit's coefficient names are not among the larger
fit's (an aliased column can break that), the pair falls back to the
likelihood-ratio test with an asymptotic χ² reference and a note. A pair that
holds φ fixed gets the likelihood-ratio test.

## How far to trust each reference

**The 50:50 mixture is conservative.** It assumes the statistic is exactly
zero with probability ½. In finite samples the true probability is larger,
so the mixture puts too little mass at zero and its p-values come out too
large. For the REML statistic of a random intercept, the probability of an
exact zero is 0.561 on Dyestuff's design (6 groups of 5) and 0.52 to 0.64 on
the four designs the exact test was checked on. For the ML statistic it is
larger still: Crainiceanu & Ruppert (2004) give, for a one-way model with K
groups and many observations per group, 0.71, 0.65, 0.61 and 0.55 at K = 5,
10, 20 and 100. In critical values, the 0.99 quantile of the ML statistic at
K = 5 is 5.41 under the mixture and 3.48 under the correct distribution.
Greven et al. (2008) found the mixture conservative in their simulations,
undersized and less powerful than the exact test. The error is large at 5 to
20 groups and small past about 100. The mixture rows carry "Conservative: the
true p-value is usually smaller."; with other groupings in the model, or
another variance component at zero, they add "Derived for one variance
component at zero with every other parameter away from its boundary; with
other groupings or components at zero it is approximate."

**χ² that ignores the boundary is more conservative.** This is what lme4's
`anova()` and lmerTest's `ranova()` print for a random-effect comparison; its
p-value is roughly twice what it should be for one dropped variance. Here it
is used only when the random parts differ by more than one dimension, and
labelled so.

**The exact RLRT has the correct level** inside its envelope, up to the Monte
Carlo error above. The REML statistic is also the better one to test with: on
some designs the ML statistic is almost always zero (Crainiceanu & Ruppert
report a mass at zero of 0.68 for the REML statistic and above 0.99 for the
ML one on a 20-group spline design). lme4's `anova()` refits two REML fits by
ML even when only their random parts differ; `compare()` and `anova(m0, m1)`
keep the REML statistic there.

**A GLMM likelihood-ratio test carries the approximation error.** A GLMM
log-likelihood is a Laplace or adaptive-quadrature approximation, and that
error does not cancel between the two models. When the tested component is
the model's only random term, the smaller model is a GLM, whose likelihood
has no integral and is exact, so the statistic carries the larger model's
whole approximation error, in one direction. It is largest where GLMMs are
hardest: few observations per group, binary responses. More quadrature nodes
(`nAGQ`, where the model allows it) shrink it. The reference distributions
above do not account for it.

A Gamma GLMM and its Gamma GLM sit on the same likelihood (both agree with
glmmTMB's to within 1e-9 on the validation design), so that pair is compared like
any GLMM against its GLM, with the 50:50 mixture. lme4's Gamma `glmer`
log-likelihood is on a different scale, so an lme4 comparison of the same
pair is not a reference for this one.

## How well calibrated are these tests

The rates in this section were measured on simulated data with a known truth,
2000 datasets per design;
[`validation.md`](validation.md#calibration-against-the-truth) describes the
simulation. A design is named `k<K>_m<M>_<response>_<random>`: `K` clusters;
`M` the most visits per cluster (mean cluster size 2, 5 and 9 at `M` = 3, 9,
17); the response, binary with prevalence 0.54, 0.22 or 0.10 (`bern54`,
`bern22`, `bern10`), counts with mean 1.65 or 0.34 (`pois165`, `pois034`), or
Gaussian (`gauss`); and the random part, `(1 | id)` (`rdi`) or
`(1 + time | id)` (`rdis`). A rate is the rejection rate of a true null
hypothesis at α = 0.05 unless stated.

**Gaussian mixed models.** The Satterthwaite t and F rows and the
Satterthwaite intervals are close to nominal from 10 clusters on: the rows
reject between 0.032 (`k20_m17_gauss_rdis`, the `time:group` F) and 0.066
(`k10_m3_gauss_rdis`, the joint F of `group` and `time:group`), and the 0.95
intervals cover 0.9365 (`k20_m3_gauss_rdi`) to 0.971 (`k10_m17_gauss_rdis`).
The exception is the F of a factor with more than two levels that varies
between clusters, under a random slope, at few clusters: a three-level subject
factor rejects 0.098 on `k10_m3_gauss_rdis`, 0.073 on `k20_m17_gauss_rdis` and
0.078 on `k30_m9_gauss_rdis`, and 0.044 (`k200_m17_gauss_rdis`) to 0.0625
(`k100_m3_gauss_rdis`) from 50 clusters on. lmerTest's Satterthwaite F gives
the same rates on the same datasets (0.101 on `k10_m3_gauss_rdis`, 0.078 on
`k30_m9_gauss_rdis`), so the error is the method's. Kenward and Roger's F
(Kenward & Roger 1997, *Biometrics* 53(3), 983–997), which glmm does not
compute, is conservative there at 10 clusters (0.027 on `k10_m3_gauss_rdis`)
and slightly above nominal at 30 (0.0615 on `k30_m9_gauss_rdis`). A one-df
term between clusters is not affected (`group`: 0.039 on `k30_m17_gauss_rdis`
to 0.060 on `k30_m3_gauss_rdis`).

The likelihood-ratio test of a Gaussian fixed part (`compare()` fits both
models by ML) is anti-conservative at 10 clusters, 0.079 (`k10_m9_gauss_rdis`)
to 0.116 (`k10_m3_gauss_rdis`), still above nominal at 20 (up to 0.069,
`k20_m3_gauss_rdi`), and within the declared band from 30 on. At 10 clusters
the Satterthwaite F of the same two terms rejects 0.048 (`k10_m9_gauss_rdis`)
to 0.066 (`k10_m3_gauss_rdis`), so for a fixed-effect hypothesis at few
clusters the F table of one fit is the better test.

**GLMMs at few clusters.** Wald and the likelihood-ratio test are both
asymptotic in the number of clusters, and below about 50 clusters they can err
in opposite directions. On binary data Wald is conservative and the
likelihood-ratio test anti-conservative: pooled over the four one-df null
terms measured and the 18 binary designs at each `K`, Wald rejects 0.030,
0.038 and 0.040 at `K` = 10, 20 and 30, the likelihood-ratio test 0.055, 0.056
and 0.054. On counts the two agree within 0.004 on the same pooled terms, but
Wald rejects a term that varies between clusters too often: on
`k10_m17_pois165_rdi` the Type II row of `group` rejects 0.121 under Wald and
0.097 under the likelihood-ratio test, and a two-df factor 0.252 and 0.179.

Where Wald is calibrated: read as a rule over the designs, at `K ≥ 50` with at
least about 0.5 minority events per cluster (the minority events are the
smaller of the events and non-events, for counts the total count). Above that
line every one-df Wald test and every Wald interval measured is within the
declared band (a relative error of at most 25% plus Monte Carlo noise). The
line's `K` is set by the two-df rows: a two-df Wald row at `K = 50` can still
be anti-conservative (6 of 26 designs: 0.0245 to 0.030 at α = 0.01, and on 3
of them up to 0.0915 at α = 0.05, on `k50_m17_bern22_rdis`). Without those
rows the best-fitting rule moves to `K ≥ 30`, though a few one-df rows and
intervals still miss there (5 of the 104 designs with `K ≥ 30` above the
events floor; the Type II row of `group` on `k30_m17_pois165_rdis` rejects
0.0295 at α = 0.01). The events floor comes from the binary designs; no count
design measured has fewer than 0.66 events per cluster. The same analysis on
the likelihood-ratio test gives the same line. On a GLM (no random effects)
Wald is within the band at 12 or more minority events; below that 11 of the 14
designs miss it, 10 of them only on the conservative side (0.002 on
`k20_m3_bern10_rdi`, measured on a response drawn without random effects on
the same design).

**What `test = "auto"` does, and why.** In every design both tests were
decided on the same datasets, and the per-term tables' default follows what
that showed:

- A Poisson GLMM gets the likelihood-ratio rows, at every size (glmm fits
  Poisson models with the log link only, the link measured here). In none of
  the 72 count designs is Wald closer to nominal than the likelihood-ratio
  test by more than twice the Monte Carlo standard error, and the
  likelihood-ratio test is closer in 48. The gap is large at few clusters
  (above), still there at
  `K = 50` (the likelihood-ratio test closer in 17 of the 36 count designs
  with `K ≥ 50`; Wald differs from the likelihood-ratio test by up to 0.014,
  mostly on the conservative side: the Type III row of `group` on
  `k50_m3_pois034_rdi` rejects 0.031 under Wald, 0.045 under the
  likelihood-ratio test), and by about 0.01 at most at `K ≥ 100` (0.011, the
  same row on `k100_m3_pois034_rdi`).
- A binomial GLMM gets Wald, because neither test is closer to nominal across
  designs: Wald in 29 of the 108 binary designs, the likelihood-ratio test in
  25, a tie in 54. Wald tends to be the better test with about two
  observations per cluster at prevalence 0.22 or more (`time:group` on
  `k30_m3_bern54_rdi`: Wald 0.033, likelihood ratio 0.069); the
  likelihood-ratio test with very sparse data under a random slope (the Type
  III row of `group` on `k30_m3_bern10_rdis`: Wald 0.003, likelihood ratio
  0.018) and with many observations per cluster on a multi-df term (the two-df
  factor on `k30_m17_bern54_rdi`: Wald 0.094, likelihood ratio 0.076).
- A GLM gets Wald: the two tests tie in 79 of the 90 GLM designs (70 under the
  most pessimistic standard error); the likelihood-ratio test is better in 9
  (small sparse designs, where GLM Wald is conservative) and Wald in 2.
- Every other model keeps its table: the Wald χ² for a negative binomial GLMM
  and a GLMM of any other family (such as Gamma), whose calibration was not
  measured, and the F table for a linear mixed model, a linear model and a GLM
  with an estimated dispersion.

Every χ² table under `test = "auto"` names its test and the reason in its
heading.

`test = "Wald"` and `test = "LRT"` remain available on every fit with χ² rows;
on any fit other than a Poisson GLMM fitted with the default
`anova_test = "auto"`, the likelihood-ratio rows need `anova_test = "LRT"` at
fit time.

**Sparse binary data with small clusters.** With about two observations per
cluster and prevalence 0.10, and under a random slope at 0.22, the Laplace fit
itself fails, whatever the number of clusters: the random-effect standard
deviation runs far too large (its mean is 24 above the true 0.8 on
`k200_m3_bern10_rdis`), the 0.99 Wald interval of the intercept covers 0.730
(`k200_m3_bern10_rdi`) and 0.076 (`k200_m3_bern10_rdis`), the random-slope
variance test rejects 0.932 (`k200_m3_bern10_rdi`), and the random-intercept
test against a GLM up to 0.324 (`k100_m3_bern10_rdi`). Many fits also fail to
converge there, mostly by separation, where no maximum-likelihood estimate
exists (78% of the non-aliased datasets on `k10_m3_bern10_rdi`). The failure
is the Laplace approximation's: lme4's `glmer` reached the same
log-likelihood, or a lower one, with the same large variances, on 30 such
datasets.

On such data, fit with `nAGQ` greater than 1 where the model allows it (one
grouping factor with at most three random effects;
[`algorithms-glmm.md`](algorithms-glmm.md)), and check a table with
`test = "LRT"`. What that buys was measured on the `time:group` test of the 90
`(1 | id)` GLMM designs with 11 nodes, and it is not calibration:

- It removes the anti-conservative failure. Outside the declared band, the
  Laplace likelihood-ratio test is anti-conservative on 8 designs, all binary
  with about two observations per cluster, and under AGQ on none
  (`k50_m3_bern10_rdi`: 0.113 under Laplace, 0.044 under AGQ;
  `k200_m3_bern10_rdi`: Wald 0.073 and 0.048, the likelihood-ratio test 0.090
  and 0.060).
- It leaves Wald conservative at few clusters, more so than under Laplace (on
  `k30_m3_bern10_rdi` 0.036 under Laplace, 0.009 under AGQ). On the sparse
  designs the likelihood-ratio test under AGQ is near nominal or conservative
  (0.035 there, 0.085 under Laplace); elsewhere it keeps the Laplace test's
  small excess at few clusters. Outside the band AGQ is conservative on 8
  designs (7 Wald, 1 likelihood-ratio), Laplace on 1.
- Counted both ways at 4 standard errors with no margin, 24 of the 90 designs
  miss under Laplace (12 with an anti-conservative rate, 14 with a
  conservative one, 2 of them both ways) and 23 under AGQ (5 and 20, again 2
  both ways); the 5 left are designs where AGQ and Laplace agree within 0.003.
- It does not rescue the fits that fail: over the 90 designs AGQ failed on
  10,837 datasets and Laplace on 10,833. Intervals, variance-component tests
  and random-slope models were not measured under AGQ.

## Refusals, and how to fix each

Every refusal stops the call (Python: `ValueError`, or `TypeError` for a wrong
kind of argument) with a message that says what is wrong and what to do. The
full texts are in [`warnings.md`](warnings.md#test-and-comparison-errors).

| Refusal | Fix |
|---|---|
| a fit did not converge | refit until it converges (a simpler random part often helps) |
| different row counts, responses, weights or offsets | fit both models to the same rows; `compare(data, f0, f1)` drops missing rows once for both |
| different families or links | refit both with the same family and link, or use `compare()`, which takes them once |
| different held dispersions | refit both with the same `dispersion` argument (`compare()` cannot hold the dispersion) |
| different responses, or neither formula nested in the other | compare two nested models of the same response (two non-nested models can be compared by AIC and BIC) |
| one REML fit and one ML fit | fit both the same way, or use `compare()`, which picks the estimator |
| two REML fits with different fixed parts | `compare()`, which fits both by ML, or refit both with `REML = FALSE` (`reml=False`) |
| the same model twice, or the same parameter count | compare two models that differ by at least one parameter |
| an interaction without its marginal terms (`anova(m)`) | add the missing terms |
| `test = "LRT"` on a fit that stored no likelihood-ratio tables | refit with `anova_test = "LRT"` (Python `anova_test="LRT"`) |
| `test = "auto"` on a Poisson GLMM fitted with `anova_test = "Wald"` | ask for `test = "Wald"`, or refit with the default `anova_test = "auto"` |
| `test = "Wald"` or `test = "LRT"` on a fit with F rows | leave `test` at its default |
| `test` with two fits | leave it out: a comparison's test follows from the two models |
| three or more fits or formulas | compare two at a time |
| a fit object made by a version without these tests | refit the model |
| `nsim` outside 1 to 1,000,000, or `seed` outside its port's range | pass a whole number in range, or leave them at their defaults |

## The notes a table can carry

A note about one row is printed after that row's label, for example
"x2: The coefficient x2 is aliased, so the hypothesis was not tested."

| Note | When |
|---|---|
| The fit did not converge, so its estimates are not at a maximum and these tests are not reliable. … | `anova(m)` on a fit that did not converge (first note) |
| Missing cells for: …. Interpret type III hypotheses with care. | a Type III table whose design has an empty cell |
| The coefficient `<name>` is aliased, so the hypothesis was not tested. | a term with an aliased column (NaN row), one note per aliased coefficient, named as in the fit's coefficient names |
| No denominator df (`<reason>`), so the hypothesis was not tested. | an F row with no df: Satterthwaite switched off or failed |
| The `<rows>` tested coefficients span `<rank>` directions of their covariance; the F is on those `<rank>`. | a rank-deficient hypothesis |
| The model refitted for this term's likelihood-ratio test did not converge, so the statistic and its p-value are NaN. | a likelihood-ratio row whose refit did not converge, or whose log-likelihood is not finite |
| Likelihood-ratio test not available: the term's column carries a random slope, so the row is the Wald chi-squared test. | a likelihood-ratio table's row for a term with a random slope |
| The model refitted under the term's hypothesis reached a higher log-likelihood than the larger model, by more than round-off, so the statistic is set to 0: the larger model's fit did not reach its optimum; refit it. | a likelihood-ratio row whose statistic came out below zero |
| The fit did not converge, so no model was refitted and the likelihood-ratio statistics are NaN. | a likelihood-ratio table of a fit that did not converge (after the first note) |
| The model under the hypothesis of `<term>` could not be built (the hypothesis is too ill-conditioned), so its likelihood-ratio statistic is NaN. | a Type III likelihood-ratio row whose reduced design could not be built |
| Conservative: the true p-value is usually smaller. | every 50:50 mixture reference |
| Derived for one variance component at zero with every other parameter away from its boundary; with other groupings or components at zero it is approximate. | a mixture reference with other groupings or components at zero |
| The random parts differ by more than one dimension: the chi-squared reference ignores the boundary; conservative. | χ²(Δnpar) on random parts more than one dimension apart |
| The exact restricted likelihood-ratio test needs the larger model to be a Gaussian REML fit whose only random term is one unweighted (1 \| g) … so the 50:50 mixture is used. | a one-dimension REML pair outside the exact test's envelope |
| The exact restricted likelihood-ratio test cannot be computed: `<reason>`. The 50:50 mixture is used. | the grouping has more than 100 levels with rows, the design leaves no residual df, or the fixed effects absorb the grouping |
| `<side>`, the larger model, did not reach its optimum: … The statistic is set to 0; refit `<side>`. | a raw statistic below zero beyond round-off |
| The two fits use different quadrature node counts, … | two GLMM fits with different `nAGQ` |
| The Wald F cannot be formed from the two fits' coefficients: … The likelihood-ratio test is used, with an asymptotic chi-squared reference. | a dispersion-GLM pair whose coefficient names do not nest |
| The log-likelihood of `<side>` is not finite (a fit that failed or collapsed), so the statistic and its p-value are NaN. | a NaN or infinite log-likelihood |
| Both models were fitted by ML because their fixed parts differ. | `compare()` on a Gaussian pair whose fixed parts differ (not two OLS fits) |

## From Rust

The comparison surface lives in `glmm::formula::anova` (feature `formula`):
`nesting(f0, f1)` (the formula check), `compare(&Model, &Model,
&CompareOptions)` (two fitted models as plain numbers, `Model::from_fit` builds
one from a `Fit` and its `Lowered`), and `term_tests(&Lowered, &Fit,
&TermTestOptions)` (the per-term table). They return `Comparison` (with
`ComparisonTest::{Lr(LrTest), Linear(LinearTest)}`) and `TermTable`, whose
statistics and references are the `glmm::inference` types `LrTest`,
`LinearTest`, `Reference` and `Objective`. `Fit::nobs` and `Fit::objective`
carry the row count and the label a comparison reads. The ports' entry points
are in `glmm::orchestrate` (feature `orchestrate`): `run_compare`, with the
constants `DEFAULT_SEED`, `DEFAULT_NSIM` and `MAX_NSIM`, and the per-term
tables `run_fit` stores on `FitResult`. `TermTestOptions::test` takes
`TermTest::{Auto, Wald, Lrt}`, and `TermTable::choice` (a `TestChoice`) says
which test ran and why (`ChoiceReason`); `run_fit`'s `anova_test` (`"auto"`,
`"Wald"` or `"LRT"`) decides whether `FitResult::term_table_type3_lrt` and
`term_table_type2_lrt` are built, and `FitResult::term_test_auto` records
what `"auto"` gives. None of these is under the semver promise before 1.0.

## References

- Crainiceanu, C. M. & Ruppert, D. (2004). Likelihood ratio tests in linear
  mixed models with one variance component. *Journal of the Royal Statistical
  Society B* 66(1), 165–185.
- Greven, S., Crainiceanu, C. M., Küchenhoff, H. & Peters, A. (2008).
  Restricted likelihood ratio testing for zero variance components in linear
  mixed models. *Journal of Computational and Graphical Statistics* 17(4),
  870–891.
- Self, S. G. & Liang, K.-Y. (1987). Asymptotic properties of maximum
  likelihood estimators and likelihood ratio tests under nonstandard
  conditions. *JASA* 82(398), 605–610.
- Stram, D. O. & Lee, J. W. (1994). Variance components testing in the
  longitudinal mixed effects model. *Biometrics* 50(4), 1171–1177.
- Kuznetsova, A., Brockhoff, P. B. & Christensen, R. H. B. (2017). lmerTest
  package: tests in linear mixed effects models. *Journal of Statistical
  Software* 82(13).
- Fox, J. & Weisberg, S. (2019). *An R Companion to Applied Regression*, 3rd
  ed. Sage.
