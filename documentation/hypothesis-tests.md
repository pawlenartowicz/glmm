# Hypothesis tests: `anova()` and `compare()`

Three calls test hypotheses about a fitted model, in both ports:

| Call | R | Python | Takes | Refits? |
|---|---|---|---|---|
| per-term tests | `anova(m)` | `glmm.anova(m)` | one fit | no: reads the table the fit stored |
| two fits compared | `anova(m0, m1)` | `glmm.anova(m0, m1)` | two existing fits | no, and reads no data |
| two formulas compared | `compare(data, f0, f1)` | `glmm.compare(data, f0, f1)` | one data set, two formulas | fits both models itself |

The Rust crate computes every statistic and every p-value; the two ports only
lay the numbers out, with R's column names, so both give the same tables. A
comparison always takes two models: a third fit or formula is an error. The
names of these calls and of their result classes are provisional until 1.0.0.

## The per-term tests of one fit: `anova(m)`

`anova(m)` gives one row per fixed term. The rows are built when the model is
fitted, from that fit alone, and stored on the fit object, so the call
computes nothing and refits nothing. There are no random-effect rows: a
variance component is tested by comparing two models (below).

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
| binomial, Poisson and negative-binomial GLMs, a Gamma or inverse-Gaussian GLM with φ held, every GLMM | Wald χ² on the rank of the term's hypothesis | `Chisq`, `Df`, `Pr(>Chisq)` |

The heading names the type and the method, for example "Type III Analysis of
Variance Table with Satterthwaite's method", "Type III Analysis of Deviance
Table (Wald χ² tests)", or "Type III Analysis of Variance Table" on an OLS fit
or a GLM with φ estimated.
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
fixed-effect column (`p ≥ 1`), and the two fixed parts are the same. An
offset is allowed. Outside that envelope a pair that differs by one dimension
gets the 50:50 mixture, with a note that says why. The method is in
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
| Conservative: the true p-value is usually smaller. | every 50:50 mixture reference |
| Derived for one variance component at zero with every other parameter away from its boundary; with other groupings or components at zero it is approximate. | a mixture reference with other groupings or components at zero |
| The random parts differ by more than one dimension: the chi-squared reference ignores the boundary; conservative. | χ²(Δnpar) on random parts more than one dimension apart |
| The exact restricted likelihood-ratio test needs the larger model to be a Gaussian REML fit whose only random term is one unweighted (1 \| g) … so the 50:50 mixture is used. | a one-dimension REML pair outside the exact test's envelope |
| The exact restricted likelihood-ratio test cannot be computed: `<reason>`. The 50:50 mixture is used. | the design leaves no residual df, or the fixed effects absorb the grouping |
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
tables `run_fit` stores on `FitResult`. None of these is under the semver
promise before 1.0.

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
