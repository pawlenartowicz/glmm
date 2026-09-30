# Changelog

All notable changes to the `glmm` crate are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The Python package (`glmm` on PyPI) is versioned in lockstep with the crate and
shares these entries; Python-specific notes are called out where they differ.

## [0.4.1] — Unreleased

GLMMs on non-canonical links and Gamma GLMMs now reach the maximum of the
Laplace likelihood itself, which is glmmTMB's objective and not lme4's. A
Gamma GLMM's dispersion is maximum likelihood. Prior weights are precision
weights on every family with a dispersion, and a Gamma or inverse-Gaussian
GLM's `Fit::loglik` is the maximised precision log-likelihood, not R's
case-weight one.
Negative-binomial and Gamma GLMMs take `nagq > 1`. PIRLS no longer stalls in
a two-cycle, and the packed layout factors a sparse matrix. Every warning a
fit raises is now also
kept on the fit, with a tier, and a fit that did not converge always raises one
severe warning.

### Changed

- **Fixed-effect columns come in R's order, with R's interaction names.**
  The formula frontend used to lay the terms out as written and to name an
  interaction's parts as the term spells them. It now follows R's
  `terms.formula` and `model.matrix`: the intercept, then the main effects,
  then the two-way interactions, and so on, each degree in written order.
  An interaction's parts are named in the order their variables first appear
  in the fixed part of the formula. `y ~ x*z + f` gave `x z x:z fb fc` and now
  gives `x z fb fc x:z`; `y ~ x + f:x` gave `x fb:x fc:x` and now gives
  `x x:fb x:fc`. `a*b*c*d` lists its interactions in R's order
  (`a:b a:c b:c a:d …`). A variable repeated inside a term counts once, as in
  R: `x:x` is `x`, where it used to be a column of `x²`. Otherwise the model
  is the same and only the columns move, except when two columns are exactly
  collinear: the later one is dropped, so `y ~ x:z + xz` (with `xz = x * z`)
  now reports `x:z` as aliased, as lme4 does, where it used to report `xz`.

  **Semver note (0.4.1).** The formula frontend is outside the
  semver-covered surface (`fit_cold`/`fit_warm`, `ModelSpec`, `GroupIds`),
  so this ships in a patch release, but callers see it:
  `Lowered::col_names`, the column order of `Lowered::x` (and so
  `target_indices` positions), and the coefficient order in the Python and R
  packages (`fit.beta`, `fit.names`, `fixef()`, `coef()`, `vcov()`, the
  summary table) change for every formula whose terms were not already
  written main effects first, and interaction names change wherever a term
  spells its variables in another order than their first appearance. Code
  that reads coefficients by position must switch to names, and a β saved
  from an earlier version and passed back as a warm start must be reordered.
  Formulas already written in degree order with consistently ordered
  interactions (`y ~ x*z`, `y ~ f + x + f:x`) are unchanged.

  The bit-identity dumps are re-pinned. Rung 47 (InstEval,
  `y ~ service * dept + studage + lectage`) is the only record that moves,
  in all three configs: `studage` and `lectage` now come before the
  `service:dept` block, and the permuted X changes the rounding. Matched by
  name, β moves by at most 1.6e-7 relative (1.6e-9 absolute), the Hessian
  SEs by 9.1e-9, τ² by 2.3e-8; the REML criterion drops by 5.5e-10 and the
  search takes 32 evaluations instead of 33.
- **The Laplace log-determinant uses the observed curvature on non-canonical
  links** (probit, cloglog, negative binomial, Gamma with the log link). It used
  the expected (Fisher) weight, as lme4 does, which is not the curvature of the
  integrand at the mode. On the eight affected validation goldens glmm now
  matches glmmTMB's deviance to 8e-5 (to 2e-9 on five of them) and its
  parameters inside the cross-engine bands; against lme4 the fixed effects move by up to 7.9 % on
  cloglog and more on Gamma. Canonical links do not change. `WaldSe::Rx` and
  the AGQ node scale use the observed curvature too.
- **A Gamma GLMM estimates φ by maximum likelihood,** as one more coordinate of
  the outer search. It profiled φ as `D/n` inside the objective before, which
  is not a maximum. `Fit::dispersion` is that φ̂, `Fit::loglik` the maximised
  Laplace log-likelihood, the joint-Hessian SEs carry φ's uncertainty, and
  `WaldSe::Rx` is `φ̂·RX⁻¹`. The reported deviance carries every normalising
  constant of the Gamma log-density, so `deviance = −2·logLik` exactly. A fit
  whose φ̂ lands on the edge of its search box (`[1e-6, 1e6]`, reached by data
  the fixed effects reproduce almost exactly) is reported as not converged. On
  a weighted fit the box applies to `φ/w̄`, the normalised scale, so the edge
  sits at `w̄·[1e-6, 1e6]` in raw units.
- **A held Gamma dispersion (`FitOptions::dispersion = Some(v)`) is now held
  on a GLMM.** The fit used to estimate φ anyway and only report `v`; it now
  maximises over θ and β at φ = v, and the SEs carry no φ row.
- **Prior weights are precision weights on every family with an estimated
  dispersion** — Gaussian, Gamma, inverse-Gaussian: row `i` has dispersion
  `φ/wᵢ`, the same convention as `lm`, `summary(glm)` and lme4. Multiplying
  every weight by the same constant does not change the fit (φ̂ scales with
  it). glmmTMB instead multiplies each row's log-likelihood by `wᵢ`; see its
  `dispformula` offset recipe in `conventions.md` to reproduce glmm's fit
  there. Binomial weights are trial counts, Poisson gives the same fit either
  way, and NB weights multiply the log-likelihood.
- **A Gamma or inverse-Gaussian GLM's `Fit::loglik` is the maximised
  log-likelihood of the precision model,** at the ML φ̂. It was
  `logLik.glm`'s: each row's log-density multiplied by `wᵢ`, with φ plugged
  in at `D/Σwᵢ`. A Gamma GLM's SEs and `Fit::dispersion` stay
  `summary(glm)`'s Pearson moment, `Σ wᵢrᵢ²/(n − p)`, unchanged.
- **Negative-binomial GLMM standard errors carry the uncertainty in θ_NB.** The
  joint Hessian gains the `ln θ_NB` row, as glmmTMB's has; before, the SEs
  conditioned on θ̂_NB as lme4's do. On `sim_nb` the intercept SE moves from
  0.16317 to 0.16386.
- **PIRLS damps a two-cycle.** When the step overshoots the mode on alternate
  iterations it is halved, and a trial with a non-finite deviance counts as an
  overshoot. The fifteen accuracy-grid cells that ended `PirlsExhausted` now
  converge.
- **The packed layout factors its PIRLS matrix sparsely** when it is large and
  sparse enough. The switch point is provisional until it is measured on a
  locked machine.
- **A negative-binomial GLMM whose no-RE prefit failed seeds θ_NB at 1.**
- **Python and R print Gamma's dispersion as `Dispersion (phi, Pearson)` on
  a GLM and `Dispersion (phi, ML)` on a GLMM,** and `Dispersion (phi, fixed)`
  when `dispersion=`/`dispersion` holds φ, on any family that takes one.
- **Every warning text was rewritten** in plain words, as
  `<Tier>: <title>. <message>`. Code that matches on message text (for example
  "boundary (singular) fit") must match on `kind` or the class instead.
- **The singular, AGQ-fallback and ignored-argument warnings now have classes**
  under `DiagnosticWarning` / `fastglmm_diagnostic`, so filtering that channel
  silences them too.
- **`nagq` / `nAGQ` above 1 on a Gaussian model or one without random effects**
  is an ignored-argument note, not an AGQ-fallback caution.
- **A GLMM stopped at its evaluation budget reports `tau2`, `varcorr` and
  dispersion** at its best point instead of NaN, as the LMM already did.
- **`singular` is no longer set on a fit that did not converge** (LMM and GLMM).
- **A negative-binomial GLM whose shape search hits its 25-round cap** now
  reports β and SEs refit at the reported θ; before, they came from the
  previous θ.
- **PIRLS takes a full Newton step on every non-canonical link** (probit,
  cloglog, negative-binomial with the log link, Gamma with the log link),
  instead of the linearly-converging Fisher weight. The 30000-row Gamma
  false convergence and the 17 wide/30000-row cells that stopped short are
  fixed: every non-canonical GLMM now lands inside the deviance band against
  glmmTMB. The two-cycle damping stays as a guard, and fires far less often
  now that the step itself is Newton's.
- **The GLM IRLS loop stops on R's relative rule,**
  `|Δdeviance| / (|deviance| + 0.1) < 1e-12`, not an absolute change in the
  deviance. Every GLM fit moves at round-off level. A weighted GLM fit no
  longer depends on the scale of the weights, and a fit whose deviance
  alternates between two points is now reported as not converged instead of
  stopping early.
- **The negative-binomial profile log-likelihood uses `lnΓ` above a count of
  100000,** instead of summing every term one by one, so a fit with huge
  counts no longer costs seconds per row.

- **A fixed-effect factor with only one level is an error.** A text, logical
  or bool column with a single value used to be dropped from the design
  without a message, in both ports. It now fails with "column 'f': contrasts
  can be applied only to factors with 2 or more levels", as lme4 does, both
  as a main effect and inside an interaction.
- **A grouping with only one level is dropped with a warning.** A term like
  `(1 | g)` where g has one value used to be fitted, with a variance the data
  cannot determine. The term is now dropped and the rest of the model is
  fitted, with the new `single_level_grouping_dropped` warning (Python
  `SingleLevelGroupingDroppedWarning`, R class
  `fastglmm_single_level_grouping_dropped`). lme4 stops with an error here.
  When every random-effect term is dropped, the model is fitted as the same
  formula without random effects: OLS for Gaussian, a GLM otherwise. The
  `nagq` fallback and non-convergence messages in both ports now check
  whether the fitted model has random effects, not whether the formula
  asked for them.
- **Python and R give the same error and warning text for the same problem.**
  Only argument names keep each language's spelling (`nagq`/`nAGQ`,
  `warm_start`/`start`, `wald_se`/`wald.se`, `init_theta`/`init.theta`). Both
  ports run their argument checks in the same order, so the same bad input
  fails with the same message. A missing formula column now reads
  `column(s) not found in data: <names>` in both, listing every missing
  column. Python now refuses `(x || g)` and a bare `.` term with the same
  message as R. R shows the kernel's own message for an intercept-free
  random-effect term, as Python does. `documentation/warnings.md` now also
  lists every error.

### Added

- **`nagq > 1` on negative-binomial and Gamma GLMMs.** Adaptive quadrature was
  refused for both before.
- **`m.warnings` / `m$warnings`:** every warning a fit raised, with `tier`,
  `kind`, `title` and `message`. The list of warnings is
  `documentation/warnings.md`. The text summary in both ports ends with a
  `Warnings:` section.
- **Severe warnings for non-converged fits:** `search_limit`, `fit_failed`,
  `glm_diverged`, `design_unsolvable`, `too_few_rows`, `no_coefficients` and
  `constant_response`. Before, a non-converged fit raised nothing.
- **`Note::NbShapeUnsettled`** (Rust, additive; `Note` is `#[non_exhaustive]`)
  and its caution `nb_shape_unsettled` in both ports.
- **Python: 11 warning categories** (`SearchLimitWarning`, `FitFailedWarning`,
  `GlmDivergedWarning`, `DesignUnsolvableWarning`, `ConstantResponseWarning`,
  `TooFewRowsWarning`, `NoCoefficientsWarning`, `SingularFitWarning`,
  `AgqFallbackWarning`, `ArgumentIgnoredWarning`, `NbShapeUnsettledWarning`) and
  `Fit.warnings`. R: the matching `fastglmm_<kind>` condition classes.
- **`Fit.dispersion_held` (Python) / `$dispersion_held` (R):** the numeric
  `dispersion=`/`dispersion` argument the caller held φ at, `None`/`NULL`
  when it was estimated. `summary()`'s dispersion label reads it to print
  `(phi, fixed)` instead of `(phi, ML)`/`(phi, Pearson)`.
- **`Note::NonIntegerResponse`** (Rust, additive) and its caution
  `non_integer_response` in both ports: a response that misses the nearest
  integer by more than R's own tolerance still fits, with this caution
  instead of fitting silently.
- **`rows_dropped_na` caution (both ports):** a row with a missing value in
  a column the formula uses is dropped and reported, the same rule as R's
  `na.omit`. Python raises `glmm.RowsDroppedWarning`; R raises the matching
  `fastglmm_rows_dropped_na` condition.
- **`glmm::formula::referenced_columns`** (the `formula` feature, on by
  default): the names a formula actually reads as data, so a caller can
  filter its columns down before conversion instead of touching one the
  formula never uses. The Python port's `fit()` calls it through
  `_native.formula_columns`.

### Removed

Breaking. Nothing in the crate, the ports or MCPower calls these.

- **`Sizing::atom` and `Sizing::cluster_of_row`.** The row layout they
  describe belongs to the caller's data generator, not to the fit.
- **`Scalar::abs` and `Scalar::mul_add`** on the doc-hidden
  `glmm::scalar::Scalar` trait.
- **`loop_advanced`: `LmmGroupings::single` and `LmmSuffStats::new`.** Use
  `LmmGroupings::from_cluster_spec` and `LmmSuffStats::with_groupings`.

### Fixed

- **A Gaussian mixed model with nearly collinear predictors keeps its
  digits.** The LMM worked from `X'X` built on the raw predictors, which
  squares how badly conditioned X is. With two predictors collinear to 3e-6
  (`sim_entangled_pair_lmm`), the REML criterion was 6.5e-4 too high and β
  was 4e-4 off. Both LMM kernels (dense and sparse) now make the fixed-effect
  columns orthogonal once per fit, using a QR factorisation of X, and map β̂
  and its covariance back. On that design the criterion is now within 2e-10
  of a 60-digit reference and β within 1e-11. Every LMM with more than one
  fixed-effect column moves in its last digits: the deviance by at most 3e-9
  on the validation rungs, the variance parameters within the optimizer's
  stopping tolerance. Models with only an intercept are unchanged.
  `LmmSuffStats` (`loop_advanced`) gains `set_design_qr`. A caller that
  accumulates rows by hand without calling it gets the old behaviour.
- **R: an inverse-Gaussian model with random effects gives an error message.**
  It used to stop with a raw kernel panic. It now says `family
  'inversegaussian' is GLM-only: random-effect terms are not supported`, as
  Python does.
- **An interaction written twice in a different order is one term.**
  `y ~ f*g + g:f` built the `f:g` columns twice and reported the copies as
  aliased. R treats `f:g` and `g:f` as the same term, and so does glmm now;
  the first spelling keeps its place and its column names.
- **A non-ASCII column name parses everywhere a bare one does.** `y ~ café`
  already fitted, but the same kind of name failed as a grouping factor
  (`(1 | grupa_ł)`, `(1 | a/b)`, `(1 | a:b)`), as the argument of `log()`,
  `sqrt()`, `exp()` or `I(x^k)`, and inside `cbind()`.
- **A held inverse-Gaussian dispersion (`dispersion=`) now enters the
  log-likelihood and AIC.** It only scaled the standard errors before; the
  log-likelihood and AIC were still computed at the profiled dispersion, so
  holding φ silently had no effect on them.
- **Gamma and inverse-Gaussian GLMs on the log link start at the null model,
  `μ₀ = ȳ`.** They used to start at `η = 0` (`μ = 1`), and on data whose mean
  sits far from 1 the first IRLS step overshot and the fit diverged. Poisson
  and negative binomial already started this way. Every Gamma-log fit that
  converged before reaches the same optimum; the inverse-Gaussian-log
  deviance is not convex, so a few of its fits now end at a different local
  optimum, better on some data and worse on other.
- **A vector AGQ fit on the boundary reports the deviance at its own θ.**
  When a fit with `nagq > 1` and a random slope ends with a variance
  component at 0, it used to rotate the covariance factor into a canonical
  form with the same covariance matrix and recompute the deviance there.
  Under AGQ that gives a slightly different value, because the quadrature
  grid depends on the factor and not only on the covariance matrix. The fit
  now keeps the factor the optimizer reached, so θ, the deviance and the
  standard errors all come from one point. Which components are reported as
  on the boundary does not change, and fits off the boundary are unchanged.
- **The Gaussian LMM is weight-scale invariant at extreme scales.**
  Multiplying every weight by the same huge or tiny constant used to push
  the internal search variable into its boundary and corrupt the standard
  errors, variance components and deviance.
- **A formula with a bare factor inside an interaction is coded the same way
  as R's `model.matrix`.** `y ~ x:f`, `y ~ f:g`, `y ~ f + f:g` and similar
  formulas used to drop the factor's base level even where R keeps it,
  fitting a different model with no warning.
- **A grouping column of whole numbers, including a logical 0/1 column, is
  accepted as a factor,** ordered by value, the way R's `factor()` treats an
  integer column. It was refused before.
- **The response is checked against the family's domain, and
  `dispersion=`/`dispersion` is checked before fitting.** A negative Poisson
  or negative-binomial count, a binomial value outside `[0, 1]`, a
  non-positive Gamma or inverse-Gaussian value, and a negative, zero or
  non-finite dispersion are refused, instead of fitting to a wrong or
  undefined answer.
- **Python's `fit()` reads only the columns the formula uses, and drops a
  row with a missing value in one of them,** the same rule as R's
  `na.omit`. It used to try to convert every column in the data, which could
  crash on an unrelated column it could not convert, and could turn a
  missing value in a plain column into a literal factor level `"nan"`.
- **R: `fixef`, `ranef` and `VarCorr` no longer mask lme4's own generics in
  either load order, and `isSingular` works whether lme4 is loaded or not.**
  Loading both packages used to break whichever one lost the masking fight,
  most visibly `isSingular` raising "no applicable method."
- **R: a non-default `contrasts=`, an ordered factor, or a factor carrying
  its own `contrasts` attribute, used as a fixed effect, is now refused**
  instead of fitting silently with the wrong coding. A factor used only as a
  grouping variable is unaffected.
- **Python: a summary's group count no longer counts a declared grouping
  level that has no rows.** It used to read the kernel's raw per-grouping
  slot count, one too many per such level; R's summary already excluded
  them.
- **Python: `weights=` and `offset=` are checked before fitting, the way the
  R port checks them.** A wrong length, a NaN, inf or missing entry, or a
  non-positive weight is now a plain `ValueError` naming the argument. It
  used to reach the kernel's entry check and come back as the text of a
  Rust panic, printed to stderr as well.
- **Python: a pyarrow Table's dictionary column keeps its declared level
  order.** `Table.column()` returns a `ChunkedArray`, which was read as plain
  strings and sorted, so the base level could change. A column whose chunks
  carry different dictionaries is unified first, and a null entry drops its
  row.
- **A GLMM whose search stops next to a failed inner solve is no longer
  reported as converged.** When the PIRLS solves around a point fail, the outer
  search can shrink onto the edge of that region and report convergence there.
  Warm-started negative-binomial fits from a far start (θ₀ = 200) did this
  21 to 26 deviance above the optimum. Such a fit now reports
  `converged = false` with no estimates (`fit_failed`). Fits whose failed
  evaluations lie farther away, and fits with none, are unchanged.
- **R: a logical predictor or grouping column is a factor, as in lme4.**
  It used to cross as a 0/1 number, so grouping levels printed as `0`/`1`
  instead of `FALSE`/`TRUE` and a fixed effect was named `b` instead of
  `bTRUE`. In `y ~ 0 + b`, `y ~ x:b` and `b*f + b:x` the design itself
  differed from lme4's; it now matches. A logical response or `offset()`
  variable stays 0/1.
- **A GLM whose Fisher-scoring steps oscillate around the MLE is damped.**
  On links where Fisher scoring is not Newton (probit, cloglog, Gamma/log,
  negative-binomial/log, inverse-Gaussian/log) the IRLS step could jump back
  and forth across the MLE with growing amplitude and end not converged. The
  kernel now watches the β step for sign reversals and halves later steps,
  with the same constants as the PIRLS period-2 damping. Logit, Poisson/log,
  Gamma/inverse and inverse-Gaussian/1/μ² fits run the old code unchanged.
- **Negative-binomial and other log-link GLMM warm starts far from the mode
  reach the optimum.** On small designs a `fit_warm` from a large θ₀ ended
  `NoOptimum` or converged above the cold optimum: the joint (u, β) step was
  too long where the β curvature is small, and the merit test accepted
  trials that raised the objective. Once that happens the solve now takes
  u-only steps at fixed β and accepts a trial only on a real decrease, as
  lme4's `pwrssUpdate` and MixedModels.jl's `pirls!` do, halving the β step
  after each rejection. Cold fits and fits that never reach this path are
  unchanged.
- **Python: a bool column is a factor with levels `FALSE` and `TRUE`,** as in
  lme4 and the R port. It used to be passed as a 0/1 number, so a bool
  predictor was named `b` instead of `bTRUE`, a bool grouping had levels
  `0`/`1`, and `y ~ 0 + b` and `y ~ x:b` fitted a different design from lme4's.
  This covers Python `bool`, `numpy.bool_`, pandas `bool` and `boolean`, and
  pyarrow `bool_`. The response and an `offset()` column stay 0/1.
- **Python: a missing formula column is reported before a `weights=` or
  `offset=` length error.** With a formula column missing from `data` and
  `weights=` of the wrong length, the weights error used to hide the real
  problem. The fit now raises `unknown column: z` first, as the R port
  already did.
- **No more flat objective past the μ clamps.** A row whose μ sat on its
  clamp (`1e-10` on the log links, `1e-12` from 0 or 1 on binomial) had a
  deviance that stopped changing with η and a score near 0, so PIRLS could
  report converged at a point that is not a mode, and a Gamma or NB fit on
  small-scale y could land on the clamp and report success. Such rows are now
  tail rows: their deviance, score and weights are computed from η in a form
  that stays finite, as exact derivatives of one deviance. Log-link μ carries
  no floor and logit μ no bound; probit and cloglog keep the bound on the
  stored μ only. Cloglog η is bounded at ±700 (was ln 700 above), the Gamma
  inverse and inverse-Gaussian `1/μ²` links at `[1e-75, 1e75]` (was
  `[1e-10, 700]`), and a PIRLS step that puts a row past such a bound against
  its data is refused. A Gamma/log GLM with y scaled by 1e-12 now gives the
  unscaled slopes.
- **The exact β-profile's border step is Newton on the whole Laplace profile, inside a trust
  region.** The step used to leave out the curvature of log|A| in β. Where that curvature
  dominates (a few clusters at a large θ), every full step overshot, and PIRLS could cycle
  until its iteration cap. The step now includes it. Far from the mode even the exact
  model can be wrong, so each step's result is compared with what the model predicted, and
  the step length shrinks or grows back by the standard trust-region rule. Where the
  curvature is not positive definite, the step is damped (Levenberg–Marquardt) instead of
  dropping the curvature. Negative-binomial warm starts from θ₀ = 200 to 3000 now reach the
  cold optimum on all 24 small test designs, blocked and crossed (before: 12 to 21 of 24). In
  a scan of 1200 far warm starts over five families, 1089 reach the cold optimum (before:
  768). Over the accuracy grid's exact-profile cells, PIRLS takes 12 % fewer iterations.
  Converged results move by round-off to tolerance: deviance by at most 1.8e-6, β by at most
  1.7e-5 relative. The curvature is computed only in solves where the plain step shows trouble
  (a shrunk trust radius, careful mode, period-2 damping or eight iterations), and on designs
  with crossed random effects it leaves out the crossed tail's terms, which gives an upper
  bound on the curvature and so a step no longer than the Newton step.
- **Log-link GLMs are no longer refused because of the response's units.** The divergence
  guard stopped a fit once any |η| passed 30. On a log link η = ln μ, so a Poisson fit with
  counts near e^30, or a Gamma or inverse-Gaussian fit with y in very small or very large
  units, was refused. A Gamma/log fit with y scaled by 1e-14 ended at β₀ = −30.53, not
  converged; R gives −31.10. The guard now measures η from the null-model value ln ȳ on
  log links, so it still stops a fitted mean that runs 1e13 times away from the data, but
  not a fit in other units. The Gamma and inverse-Gaussian log-link start no longer floors
  ȳ at 1e-10. Binomial fits are unchanged. A Poisson or NB fit with an all-zero group is
  now refused whenever the zero rows' mean passes 1e-13 of the data mean; before, that
  depended on the count scale.
- **A Gamma fit with a very small dispersion reports the right log-likelihood.** The
  dispersion term cancelled badly for φ below about 0.05: 1e-10 relative error at
  φ = 1e-6, and no correct digit near φ = 1e-16. A Gamma GLM that fits the data almost
  exactly reported a log-likelihood below its value at a fixed φ. The term now uses
  Stirling's series there. A fit whose deviance rounds to zero still reports `+∞`: the
  likelihood has no finite maximum, and R's finite value there comes from rounding.
- **GLMM fixed effects are no longer held inside ±30.** The joint search over θ and β used a
  fixed box of ±30 on every β. That box is in the units of y and of the X columns. A
  Gamma/log GLMM with y in very small or very large units needs an intercept past ±30. The
  fit then stopped on the box and still said `converged = true`. With y·1e-14 on
  `sim_gamma` it gave τ² = 3.85 instead of 0.32, and a log-likelihood 28.4 too low. A
  predictor in large units (x·1e-3) was held the same way on AGQ and Gamma fits. The box is
  gone: β is now unbounded, as it already was on the Laplace exact-profile route. Scaled
  fits now match the unscaled fit. Under complete separation, an AGQ or Gamma fit used to
  stop at β = 30 and report convergence. It now uses its evaluation budget and reports
  `converged = false`. Fits that did not touch the box are unchanged, bit for bit.
- **GLMs are no longer refused or stopped early because of the units of y or of the
  weights.** Two checks in the GLM fit used absolute numbers. The saturation check refused
  a fit when more than half the rows had an IRLS weight below 1e-5. That means "fitted
  probability pinned at 0 or 1" only on binomial. On other families the weight is μ, 1/μ or
  μ², so honest fits were refused: a Poisson fit with exposure 1e-7 on most rows, an
  inverse-Gaussian/log fit with y above about 1e5, a Gamma/inverse fit with y below about
  1e-3. The check now runs on binomial fits only. The stopping rule
  `|ΔD| / (|D| + 0.1)` has an absolute floor. On Gamma and inverse-Gaussian the deviance
  scales with the precision weights, and on inverse-Gaussian also with 1/y. So small
  weights or large y stopped the fit early while it reported converged: with every Gamma
  weight 1e-9, β was 2e-3 off. On these two families the floor now scales the same way.
  On Poisson and negative-binomial the floor is `0.1·min(1, ȳ)`: R's rule while
  ȳ ≥ 1, and scaled with y below that, so a Poisson fit of non-integer y in tiny units
  (y × 1e-13 was 1.7e-2 off in β, reported converged) now gives the unscaled fit.
  Binomial fits, unweighted Gamma fits and count fits with ȳ ≥ 1 are unchanged.
- **A warm GLMM fit now checks whether its own starting point is worth
  starting from.** Before searching, the fit compares the objective at the
  caller's θ₀ against the blind cold start and begins from whichever is
  lower — a tie or a non-finite warm value goes to cold. This costs two extra
  evaluations on a warm fit, outside the reported count. Separately, when the
  `ExactProfile` route (binomial, Poisson, and some negative-binomial shapes
  at the default Laplace approximation) ends not converged, the fit reruns
  once on the `PqlThenJoint` route from the same start; the rerun's result is
  reported when it converges, and a new note (`exact_profile_fallback`) says
  so. In a scan of 1200 far warm starts, both fixes together bring warm fits
  that reach the cold optimum from 1089 to 1173 (of 1200) and eliminate every
  remaining outright failure (52 to 0); negative-binomial warm starts from
  θ₀=1000 now reach the cold optimum on 80 of 80 small test designs (was 79).
  Cold fits are unaffected.
- **A GLMM's answer no longer depends on the units of its predictors.** The
  joint search over θ and β, used by every AGQ fit, every Gamma GLMM and every
  model on the packed layout, stepped β in the units of the X columns. A
  predictor in large units, or one far from zero, left the fit short of the
  optimum while it still reported `converged`. On `sim_gamma`, x·1000 gave a
  log-likelihood 0.37 too low, x + 1000 0.04 too low, and x·10⁶ failed. The
  search now steps the fixed effects in coordinates where the design is centred
  and scaled, the way it already scaled the random-effect columns, so rescaled
  and shifted predictors give the same fit to the optimizer's tolerance. Other
  fits move only by that tolerance (deviance by at most 1.3e-7 on the accuracy
  grid, same convergence). The Laplace fits that profile β (binomial, Poisson
  and negative binomial at `nagq = 1` on the dense layouts) are unchanged.
- **A log-link GLM with very different exposures between rows is no longer
  refused.** The divergence guard measured the linear predictor with the offset
  inside, from ln ȳ, so a row whose exposure is 1e-14 of the others tripped it:
  a Poisson fit with exposure 1e-14 on 30 of 40 rows ended at β₀ = 4.02, not
  converged, where R gives 1.084. The guard now measures from the null model
  with the offset, oᵢ + ln(Σwy / Σw·e^o), and the log-link start uses the same
  value; before, the start ignored the offset, and a Gamma/log fit with exposure
  1e-8 on most rows ran off to β ≈ 1e304. The count families' start and guard
  centre also drop R's `ȳ + 0.1` for ln ȳ, so Poisson and negative-binomial
  fits of a mean below about 1e-14 are no longer refused; only an all-zero
  response keeps 0.1. Other count fits move by round-off (deviance at most
  1e-9, β at most 1.5e-6 relative on the validation rungs), and a weighted Gamma
  or inverse-Gaussian GLM now starts at the weighted mean.

## [0.4.0] — Unreleased

The default standard errors of a GLMM are much cheaper. The sparse route now
tries an exact Hessian first and keeps the finite-difference Hessian as its
fallback. The sparse route is a backend of the same fit code, not a second
copy of it. Three bugs in the Laplace fit are fixed.

### Changed

- **The default GLMM standard errors (`WaldSe::Hessian`) come from a new exact
  Hessian pass.** It differentiates the assembled Laplace gradient once, and
  it has no limit on the number of parameters. On the sparse route it runs
  first, and the finite-difference Hessian stays as the fallback. The pass
  also takes a fit with a row whose fitted mean is held at its limit. The
  fallback takes a fit with a row at the limit of the link's linear
  predictor, a weighted logit fit with a saturated row, a fit whose observed
  factor is not positive definite, and a model above the pass's 256 MiB
  memory guard.
  Measured on a locked machine, the pass is 4.3× faster on grouseticks, 1.8×
  on VerbAgg, and 6.5× to 51× faster on the sparse test models. The standard
  errors move at round-off level on dense models. On sparse models that take
  the new pass they move within the error of the finite difference it
  replaces. AGQ fits and the LMM keep their earlier Hessian pass.
- **The floor on the IRLS weight is `1e-300`, was `1e-6`,** in the GLM and in
  the GLMM. It only keeps the weight positive. A row with a weight under
  `1e-6` (a fitted probability or mean very close to its limit) now enters
  the fit with its own weight, so a fit with such a row can move. On the test
  models one fit moved: a cloglog model with one such row, deviance down by
  2.0e-7.
- **Crossed and nested GLMMs with a probit, cloglog or negative-binomial log
  link use the θ-only exact-profile search.** They used the joint search
  before. The objective is the same, the search path is not, so the last
  digits of the estimates can move on these models.
- **One fit routine per model class.** The sparse route shares the dense
  route's PIRLS, deviance and standard-error code. `n_eval` and the last
  digits of the deviance can move on sparse fits; no fit in the validation
  corpus changed its `converged` or `singular` flag. Sparse fits are 17–39 %
  faster.
- **A sparse LMM no longer refuses a badly conditioned design.** It used to
  return NaN estimates with `converged: false` below a pivot ratio of `6e-10`.
  It now fits the design and adds an `IllConditioned` note below a pivot
  ratio of `1e-12`, the same floor as the dense LMM.
- **The formula accepts the intercept written out.** `y ~ 1 + x` is the same
  model as `y ~ x`; it was a syntax error before. A formula that both writes
  the intercept and removes it (`y ~ 1 + x - 1`, `y ~ 0 + 1 + x`) is an error.
- **Large crossed models are faster per evaluation:** 3.3× on a crossed
  simulated model at the dense size limit, 2.7× on VerbAgg. The results move
  at round-off level.
- **The PIRLS iteration-cap warning in Python and R now names one of four
  cases:** a rejected trial point, a converged fit that rests on a capped
  solve, a search that ran out of budget, or a failed fit.

### Fixed

- **A Laplace fit could return all NaN when the search ran out of
  evaluations.** The inner PIRLS cap is now 200 iterations, was 50. A fit
  that runs out of budget reports the best point it found, with
  `converged: false`.
- **The Laplace deviance is evaluated at the last PIRLS iterate.** Before, its
  log-determinant came from the iterate one step earlier. The deviance moves
  by at most 2.8e-5 on the test models. Two registered divergences from lme4
  stopped firing and were removed; `validation/divergences.json` is empty.
- **`deviance` of a negative-binomial GLMM is now exactly `−2·loglik`.**
- **R: `VarCorr()` no longer fails on a fit that found no optimum.**
- **An inverse-Gaussian GLM with the `1/μ²` link and a small mean now
  converges.** The `|η| > 30` divergence rule stopped it, because `η = 1/μ²` is
  above 30 for any mean under 0.18. The rule now skips this link, as it does
  the Gamma inverse link.
- **A GLMM with no usable fixed-effect column no longer panics.** This happened
  with zero fixed columns, and with one fixed column that is all zeros. The fit
  now comes back with NaN estimates and `converged: false`.
- **A GLMM with no more rows than fixed-effect columns (`n <= p`) now comes back
  with NaN estimates and `converged: false`**, as the OLS, GLM and LMM routes
  already did. Before, it reported a converged fit of a saturated model.
- **A Gamma inverse or inverse-Gaussian `1/μ²` GLMM could report the deviance of
  a point outside the link's domain.** The last PIRLS step could move `η` past
  the edge; a debug build panicked there. That evaluation now counts as failed.

### Removed

- **Python: the `faststats.glmm` import alias.** Use `import glmm`.
- `src/sparse/glmm.rs` and `src/glmm/pirls/dense.rs`;
  `src/glmm/pirls/packed.rs` replaces both.
- **`Diagnostics::kkt_grad_norm`, `Diagnostics::boundary_score` and
  `FitOptions::boundary_score`.** Neither diagnostic fed a fitting decision.
  `LmmGroupings::diagonal_has_nonzero_below` is also removed from the
  `loop_advanced` surface (no semver guarantee).

### Changed — `counters` (no semver guarantee)

- `nb_nodes` and `nb_evals_total` are always 0. Every route searches
  `ln θ_NB` inside the outer search, so `n_eval` already counts the whole fit.

## [0.3.3] — 2026-09-11

The random-effect search no longer stops at a false boundary where a Cholesky
diagonal is 0. The stage-1 re-run that 0.3.2 added for the same trap is
removed, and the outer search gets twice the evaluation budget.

### Changed

- **The outer BOBYQA search may spend up to `1000·n` evaluations, was PRIMA's
  default `500·n`.** This holds on every route (dense and sparse, LMM and
  GLMM); `n` is the number of search coordinates. The budget also sets when
  the search restarts (after 1/8 of it), so a long fit now restarts later: at
  750 evaluations for six coordinates, was 375. A fit that stops before that
  point is bit-identical. Measured 2026-09-11 on boundary-heavy simulation
  sets: on a set of 4,500 fits, 17 fits reach a deviance more than 1e-6 lower
  (the largest by 3.5e-4), 8 end higher (the largest by 6.6e-5), the one fit
  that used to hit the cap now converges at 796 evaluations, and total
  evaluations are 0.985× of before. On a set of 12,600 fits and two AGQ cells
  of 3,000 fits no fit moves by more than 3e-6. No fit reaches the new cap. A
  fit that never converges now spends twice as many evaluations before it
  reports that.

### Removed

- **The pinned-exit stage-1 re-run added in 0.3.2.** It ran the θ-only
  exact-profile search a second time when the first run ended at a pinned
  diagonal, and kept the better run. The signed search box (under Fixed)
  escapes the same trap in one search: on the
  `tests/fixtures/glmm_npt_trap.csv` draw the fit reaches 910.2127 in 148
  evaluations, where the re-run needed 284. `n_eval` no longer counts a
  second run.

### Fixed

- **The random-effect search no longer stops at a false boundary where a
  Cholesky diagonal is 0.** θ holds the Cholesky factor Λ of each
  random-effect covariance, and its diagonals were boxed at `[0, THETA_HI]`.
  Σ = ΛΛ′ does not change when a whole column of Λ changes sign, so the two
  sign choices of a column meet only where its diagonal is 0. A search that
  reached that point with the wrong sign on the entries below the diagonal
  saw the deviance rise in every allowed direction and stopped there, a
  deviance unit or more above the optimum, at a point no local check can tell
  apart from a real boundary. Now every θ entry, diagonals included, is
  searched in `[−THETA_HI, THETA_HI]` on every route, so the search walks
  through that point. At exit, each block column whose diagonal ended
  negative has its sign flipped: Σ and the deviance do not change, and the
  reported θ keeps non-negative diagonals. On GLMM routes the matching
  conditional modes are flipped with it, so the final mode solve starts at
  the mode. MixedModels.jl searches the same unbounded box; lme4 keeps the
  diagonals at `≥ 0`. Examples: a Bernoulli `y ~ x1 + (1 + x1 | g1)` draw
  used to stop at deviance 397.6085 and now reaches 397.4577; a Gaussian
  `y ~ x1 + (1 | g1) + (1 + x1 | g2)` draw on the sparse route used to stop
  2.14 above lme4 and MixedModels.jl, which agree, and now reaches 841.7010.
  Seven new fixtures (`tests/fixtures/sign_trap_*.csv`) pin the escape on
  dense and sparse, LMM and GLMM, Laplace and AGQ routes.

  The wider box changes the search path on every fit, not only near a
  boundary, so `n_eval` and the last digits of θ̂, β̂ and the deviance can
  move on any fit; the bit-identity dumps are re-pinned. The oracle rung
  `sim_sparse_nb` now ends 8.35e-5 deviance above lme4 (was 1.7e-6), inside
  the 2e-4 gate, and is registered in `validation/divergences.json`.
  `Diagnostics::kkt_grad_norm` projects the gradient onto the new box.

### Changed — `loop_advanced` (no semver guarantee)

- `LmmSweepOutcome.theta` is the raw search output and can now hold a
  negative diagonal; Σ is the same as with that column's sign flipped.
  `lmm_sweep_fit` and `lmm_sweep_fit_on` size the start radius from the
  absolute values of the start's diagonals.

## [0.3.2] — 2026-09-10

Published to crates.io, PyPI and R-universe. Evaluation counters, a
generic-scalar kernel, dual-number derivatives, exact-Hessian standard errors,
the θ-only exact-profile search, the negative-binomial θ search change and the
relicense.

### Added

- **Two observation-only convergence numbers on `Diagnostics`.**
  `kkt_grad_norm`: the ∞-norm of the exact deviance gradient in θ, projected
  onto the box the optimizer searched, at the accepted θ̂; NaN wherever no
  exact gradient exists (every non-GLMM route, structured extras above the
  tail bound, the dense fallback, the sparse routes, any non-converged fit).
  `boundary_score`: per pinned variance component, whether the pin is the
  constrained optimum of its basin — requested through the new
  `FitOptions::boundary_score` (off by default), empty otherwise. Nothing
  branches on either. Rust surface only: the Python and R wrappers do not
  expose them yet.


- **Dev-only `counters` Cargo feature.** With the feature on, `Fit` carries an
  `EvalCounters` struct recording four observation-only quantities per fit: the
  stage-1/stage-2 evaluation split of the two-stage GLMM search, evaluations
  after the last strict improvement of a stage's incumbent (the trust-radius
  shrink phase), a PIRLS-iterations-per-evaluation histogram, and AGQ
  evaluations with the node evaluations they cost. Off by default and — like
  `loop_advanced` — not part of the semver-covered surface. With the feature
  off the code compiles out entirely and fit output is bit-identical; a new CI
  leg runs the test suite with it on. The speed-grid campaign gained a
  counters pass (`validation/campaigns/speed-grid/counters.R`) that reads
  these counters instead of wall time.

- **Forward-mode dual numbers and exact derivative entry points.** New
  `src/dual.rs`: `Dual<N>` (value + `N` first-derivative lanes) and
  `HyperDual<N, H>` (adds the packed second-derivative triangle), both
  implementing the sealed `Scalar` trait, with crate-own `digamma`/`trigamma`
  for the Gamma-family derivative. On top of them, exact gradients and
  Hessians of the fit criteria, obtained by running the existing generic
  kernels at a dual scalar — no finite differences, no hand-written adjoints:
  - GLMM (new `src/glmm/derivative.rs`): `laplace_gradient`/`laplace_hessian`
    differentiate the joint Laplace or AGQ deviance in `(θ, β)`, with
    dual-typed twins of the θ-dependent PIRLS buffers sized at compile-time
    lane counts `N ∈ {4, 8, 12}` and allocated only on the first derivative
    request (`GlmmWorkspace::dual_scratch`, `None` on every `f64`-only fit).
    Models off the blocked path (nested/crossed extras) or with
    `n_theta + p > 12` return `Unsupported`, and the caller keeps its current
    fallback (FD Hessian, or BOBYQA on the objective).
  - LMM (`src/lmm/kernel.rs`): `reml_gradient`/`reml_hessian` differentiate
    the family-blocked REML criterion in θ — one closed-form dual evaluation,
    no mode solve. Same lane set; crossed/nested-slopes designs and
    `n_theta > 12` return `Unsupported`.

  The exact-Hessian SE pass and the θ-only exact-profile search above are
  their first callers. The `f64` fit path is bit-identical and pays no memory
  for scratch it never requests.

### Changed

- **License is `LGPL-3.0-or-later`, was `GPL-3.0-or-later`.** Crate, Python
  wheel and R package alike. `LICENSE` now carries the LGPL text; the GPL
  text it extends ships alongside as `LICENSE-GPL`.

- **Negative-binomial θ is a coordinate of the outer BOBYQA search.** The
  dense NB GLMM used to wrap a golden-section bracket over `ln θ_NB` around
  the whole fit, running a complete GLMM fit (outer θ search, PIRLS inside)
  at each of about 28 bracket nodes and once more at θ̂. Now `ln θ_NB` is one
  more trailing coordinate of the single outer search over the random-effect
  parameters, on the same marginal objective, and the bracket is deleted.
  Measured on the locked 2026-09-08 paired run over the 36 speed-grid
  `negbin` cells and the 3 NB validation rungs: per-subfamily median
  evaluation and wall ratios 0.04–0.07 of before; every cell converged in
  both arms; converged `−2 logL` differs by at most 7.5e-5 and θ̂_NB by at
  most 6.2e-4 on `ln θ`. The sparse NB route is unchanged (identical
  evaluations and θ̂). `Diagnostics::kkt_grad_norm` holds `ln θ_NB` fixed and
  covers the random-effect coordinates only.

- **`WaldSe::Hessian`, the default, is an exact Hessian where the dual kernel
  reaches.** On the blocked path and on the structured-extras shapes inside
  the measured dense-tail bound (nested and small crossed designs), the joint
  `(θ, β)` Hessian behind `se_hessian`, `vcov` and `stddev_se` is the
  hyper-dual Hessian of the Laplace deviance, no longer an `m(m+1)/2`-cell
  finite-difference stencil of PIRLS solves. The stencil still runs on the
  oversized-core dense fallback, on the sparse driver, and above the lane cap
  (`n_θ + p > 12`). θ̂, β̂, deviance and `n_eval` are untouched; `se_hessian`
  moves within its bands on the affected shapes, and the re-pinned goldens
  carry provenance comments.

- **θ-only exact-profile search on the blocked path.** Where the exact
  derivative kernel runs (the blocked path, and structured extras on
  canonical links), the outer BOBYQA searches θ alone on the exact Laplace
  β-profile, and the joint `[θ | β]` stage no longer runs for those shapes.
  Same objective and optimum, a different search path: `n_eval` and the
  iterate sequence change there. The joint search stays for AGQ,
  non-canonical structured extras, Gamma, the dense fallback and the sparse
  route.


- **The blocked fit kernel is generic over a scalar type.** Family primitives,
  blocked PIRLS, the Laplace/AGQ deviance, and the family-blocked REML kernel
  now take a sealed, `#[doc(hidden)]` `Scalar` trait (new `src/scalar.rs`)
  instead of bare `f64`. `f64` is the production instantiation and is
  bit-identical to the pre-generic kernel (bit-identity dumps and frozen
  goldens unchanged); its trait overrides call the existing SIMD and faer code
  verbatim, so the shipped path pays nothing. Every kernel branch compares the
  value part, so the iterate path a fit takes is the same at every scalar
  type. No public API change: the trait is sealed and hidden, and existing
  callers infer `f64`. This is the substrate the dual-number derivatives
  above are built on.

### Fixed

- **`Fit::dispersion` is NaN on every non-converged fit.** Before, a failed
  Gamma / inverse-Gaussian fit reported `1.0`, a failed binomial / Poisson fit
  reported `1.0`, a failed negative-binomial fit reported the last θ the search
  stood on (often the `NB_THETA_HI` box edge), and a φ held through
  `FitOptions::dispersion` came back as `1.0` — on the GLM, dense GLMM and
  sparse NB routes alike. `Diagnostics::converged`'s doc already promised the
  NaN fill; the four mapping sites now keep it. `fit_glm_nb` /
  `fit_glm_nb_capped` return `(Fit, f64)` so the dense NB GLMM still seeds its
  θ_NB search from the alternation's last θ. Python `Fit.dispersion` and R
  `$dispersion` / `sigma()` move to NaN on the same fits. Bit-identity dumps
  and goldens carry no dispersion field, so nothing recorded moves.

- **`converged` needs at least two finite objective evaluations.** BOBYQA
  maps a `+INF` objective (PIRLS non-convergence) to a finite ceiling, so a
  search whose probes all diverged saw a flat finite surface, exhausted its
  trust-region ladder and exited `Converged` with the random-effect
  coordinates still at their cold start. The convergence read on every route
  (dense GLMM stages 1 and 2, sparse GLMM, dense and sparse LMM) now also
  requires that the gating stage evaluated the objective to a finite value at
  least twice; a fit below that bar is NaN-filled as non-converged, the same
  treatment the all-diverged case already had. No in-crate test, oracle rung
  or bit-identity rung flips; on a 3888-cell pathological sweep it flips 1 of
  3053 previously-`converged` fits.

- **A pinned diagonal now leaves its RE block in canonical Λ.** At `Λ_jj = 0`
  the entries below it in that column are unidentified (Σ sees only their
  squares summed with the trailing diagonals), so BOBYQA stopped anywhere on
  a flat circle and the reported θ̂, `pinned` and `tau2` depended on where.
  After the pin loop each block with a pinned diagonal is rewritten as the
  lower Cholesky of its own Σ (redundant column exactly zero, its variance
  folded into the trailing diagonals), then the pin test is re-run. Σ is
  preserved, so `deviance`, `loglik`, `beta`, `se`, `vcov`, `varcorr`,
  `stddev_corr`, `fitted` and `ranef` do not move; on q ≥ 2 fits with a
  pinned diagonal `theta`, `tau2`, `stddev_se` and `pinned` move to the
  canonical representative (`pinned` no longer flags a component whose
  stddev is non-zero), and `boundary_score` is now reported at pinned
  diagonals that were previously withheld as NaN because their column
  carried a live off-diagonal. Non-singular fits are a bit-for-bit no-op;
  the bit-identity dumps are unchanged. `boundary_score` on the θ-only
  exact-profile route stays the raw joint-Hessian diagonal: with the column
  below the pinned diagonal zero, the deviance is even in that coordinate
  for every β, so its Hessian row vanishes and the raw and β-profiled
  diagonals are the same number (recorded in the source comment).

- **A pinned stage-1 exit on the θ-only exact-profile route re-runs the search
  once with the minimum interpolation set.** At `Λ_jj = 0` the entries below
  it in that column are unidentified, so BOBYQA's quadratic model is singular
  along that circle and the trust region can collapse there before escaping —
  a binary GLMM with an intercept and two random slopes (now
  `tests/fixtures/glmm_npt_trap.csv`) stopped at deviance 910.379 with the
  (intercept, slope 1) correlation at −1 where lme4 reaches 910.215 in the
  interior; the
  interpolation-set size alone decided which basin (npt 9–11 trapped, 8 and
  12+ escaped), and no single npt rule wins on every draw. Now, when the
  `ExactProfile` search exits with a diagonal at or below `PIN_THETA`, stage 1
  runs once more from the same blind start at `npt = n_theta + 2` (BOBYQA's
  minimum, minqa/lme4's default) and the fit keeps the arm with the strictly
  lower deviance (margin 1e-8 relative; the second arm must also meet the
  same convergence bar). `n_eval` counts both arms. Measured 2026-09-08: the
  gate never fires on the 48-rung validation grid (no rung pins), so goldens
  and bit-identity dumps are unchanged; on the 60-draw simulation cell the
  fixture came from it fixes all four trapped draws and worsens none, at +86%
  evaluations on that cell.
  `LMM_NPT_FORMULA` still governs the first arm only. Sparse GLMM is not
  affected: its θ-only stage is a warm-start accelerant, the joint stage is
  the search there. The new fixture pins the escape.

## [0.3.1] — 2026-08-27

### Added

- Formula terms `log(x)`, `sqrt(x)`, `exp(x)` and `I(x^k)` (`k` an integer
  ≥ 2), each on a single bare column, in both the fixed-effect and offset
  positions. The spelling is used verbatim as the design column name, and the
  values are computed with the same libm calls R makes (`x^2` as `x*x`,
  matching R's `R_POW` special case; every other power through `powf`), so
  they agree with R's `model.matrix` to the contrasts oracle's tolerance. A
  non-finite result (e.g. `log(x)` on a non-positive `x`) is now
  `Error::TransformNotFinite`, naming the term and the first bad row, rather
  than silently propagating into the fit. `poly()`, arithmetic inside a call
  (`log(x+1)`), and a transform on a formula's LHS are still formula syntax
  errors.
- `cbind(successes, failures)` as a binomial response. Lowers onto the same
  proportion-plus-prior-weights objective `weights=` already fits (lme4's own
  `cbind()` convention), sharing its argmin with the expanded-Bernoulli form.
  A non-binomial family is `Error::CbindNeedsBinomial`; a row whose trial
  count `successes + failures` is not a positive finite number is
  `Error::ZeroTrials`; a negative success or failure count is
  `Error::NegativeCount`. `weights=` given together with a `cbind()` response,
  or `offset=` given together with an `offset()` term, is now a clean error
  naming both sources (`orchestrate::run_fit`) instead of one silently
  overwriting the other.
- `offset(...)` as a formula term — a bare column name or one of the
  whitelisted transforms above, e.g. `offset(log(exposure))` — lowering onto
  the existing `FitOptions::offset` field. At most one `offset()` term is
  allowed per formula.
- Fixed-intercept removal, `- 1` and `0 +`. `ParsedFormula::has_intercept` is
  `false` when either is present; random-effect intercepts are unaffected.
  Follows R's contrast promotion: in an intercept-free design the first
  factor main effect in term order gets the full indicator set (all levels,
  no base dropped) while later factors and every interaction keep treatment
  contrasts, matching `model.matrix(y ~ x + f - 1)`. A formula with `- 1`/
  `0 +` and no fixed-effect term (empty design) is now `Error::EmptyDesign`.
  An intercept-free random-effect term (`(0+x|g)`, `(-1+x|g)`) is still a
  formula syntax error, as is `(x || g)`, `poly()`, and the `.` formula
  shorthand.
- Python `Fit.summary_object()` returns an lme4-shaped `Summary` (in the new
  `python/glmm/summary.py`) with `.text()`, `.html()`, `.latex()` and
  `.typst()` renderers sharing one number formatter, so the same fit reports
  the same numbers in every output. `Fit.summary()` is unchanged: it still
  prints and returns the coefficient-table `str`, now built by calling
  `summary_object().text()`. `Fit.residuals(type="response"|"pearson")` is
  new; deviance and working residuals are not offered because they need
  per-family formulas the package does not carry, and a picked default would
  silently disagree with lme4's `residuals(type=)`. `Fit` also carries the
  header inputs the summary needs: `formula`, `family`, `link` (resolved,
  after the family default), `nagq` (as actually run, 1 after a
  warn-and-strip), `nobs`, `y` (the response as the kernel fitted it — a
  `cbind(s, f)` response comes back as `s/(s+f)`) and `weights` (the prior
  weights the kernel fitted with, `None` when unweighted; for a `cbind()`
  response these are the trial counts the lowering computed).
- R `tidy()` and `glance()` (registered on `generics::tidy`/`generics::glance`,
  a new `Imports: generics` — not `broom`, which drags 21 further recursive
  dependencies for functions nothing here calls). `residuals.fastglmm()` is
  no longer a hard error: `type = "response"` or `"pearson"` now returns the
  residuals directly. `summary.fastglmm`/`print.summary.fastglmm` gained the
  blocks lme4 prints and this port didn't: the data name, the REML criterion
  (LMM) or the `AIC BIC logLik deviance df.resid` row (ML fit — `glance()`
  returns both AIC and BIC regardless), scaled (Pearson) residual quantiles,
  the correlation of fixed effects, and the Wald-z footnote. `VarCorr`'s
  printer takes a `variance` argument adding a `Variance` column ahead of
  `Std.Dev.`, on by default inside `summary.fastglmm` (lme4's
  `print.summary.merMod` shape) and off in the bare `print.fastglmm` header
  (lme4's `print.merMod` shape). `inversegaussian` is now in `sigma()`'s and
  the summary dispersion label's family lists alongside `gamma`.
- (Internal) `orchestrate::FitResult` gained `y`, `weights` and `nobs`, backing
  the Python and R residuals/summary work above; both FFI shims flatten them
  unchanged.
- Binomial complementary log-log link (`BinomialLink::Cloglog`), fixed-effect
  GLM and GLMM. Non-canonical: general Fisher-scoring branch, with an upper η
  clamp at `ln(ETA_MAX)` the other two binomial links do not need. Validated
  against R `binomial(link="cloglog")` and `lme4::glmer` (validation goldens
  `sim_cloglog_glm`, `sim_cloglog_glmm`). Reachable from the Python port as
  `link="cloglog"` and from `fastglmm` as `binomial(link = "cloglog")`.
- Inverse-Gaussian family (`Family::InverseGaussian`), fixed-effect GLM only,
  with the `1/μ²` and log links. `V(μ)=μ³`; dispersion is the post-fit Pearson
  moment estimator scaling the SE by `√φ̂`, and the log-likelihood follows R's
  `inverse.gaussian()$aic` convention with the dispersion profiled inside the
  term. Mixed models are **not** supported and fault with a message naming the
  deferral. Validated against R `glm(family=inverse.gaussian(link))`
  (validation goldens `sim_igauss_glm`, `sim_igauss_inv_sq_glm`). Reachable as
  `family="inversegaussian"` (Python) / `inverse.gaussian()` (R).

### Changed

- (Internal, no fitted number moves) The Brent scalar-kernel fitter
  (`src/lme.rs`, 3127 lines) is retired: nothing in the stable `fit` dispatch
  used it since the unified fit core, and it stayed only as a `loop_advanced`
  re-export for MCPower, which is pinned to 0.3.0 until 1.0.0. Its
  `joint_wald_chi_sq` helper moved to `src/lmm/mod.rs`, its only remaining
  caller family; the two tests that pinned the scalar kernel's deviance
  against the general path now assert against a recorded literal instead.
  `src/lmm.rs` (5411 lines) split into `src/lmm/{mod.rs, kernel.rs, tests.rs}`
  and `src/glmm/pirls.rs` (1840 lines) split into
  `src/glmm/pirls/{mod.rs, dense.rs, blocked.rs, blocked_extras.rs}`, both
  pure moves. A bit-identity dump (`validation/bit_identity/`, one JSON per
  feature configuration) now fits every manifest rung plus a set of in-crate
  fixtures and records deviance/theta/beta/SEs/eval counts at full `f64`
  precision, to make a future refactor's byte-identity easy to check in one
  diff.

### Fixed

- Pearson residuals and the summary's scaled residuals for a `cbind(s, f)`
  response were computed at unit weights in both ports, dropping the
  `√trials` factor. The weights the kernel fitted with now come back as
  `Fit.weights` (Python) / `fit$weights` (R): the `cbind()` trial counts, or
  the caller's `weights=`, `None`/`NULL` when unweighted.
- `cbind()` with a negative success or failure count is now an error
  (`Error::NegativeCount`) instead of lowering to a proportion outside
  `[0, 1]` and fitting it.

## [0.3.0] — 2026-08-24

The θ search no longer depends on the units a random-slope covariate happens to
be measured in. Random-effect design columns are scaled internally before the
optimizer sees them and mapped back to the caller's units afterwards, which
fixes cold fits that used to settle on a worse local optimum than lme4 on a
badly scaled design, and removes the false `singular` verdicts those designs
produced. Two diagnostic notes arrive alongside it. Nothing breaks: the
`#[non_exhaustive]` marking 0.2.0 paid for is what makes both new `Note`
variants additive.

### Changed

- **Random-effect design columns are scaled internally, so the θ search no
  longer depends on the units of a random-slope covariate.** Each random-slope
  column now enters the random-effect design divided by its own weighted RMS,
  with the matching rows of the relative Cholesky factor multiplied by the same
  factor — an exact model identity, so the criterion surface is unchanged and
  only the coordinate BOBYQA searches moves. Everything derived from θ̂ is mapped
  back to the caller's units (`varcorr`, `tau2`, `ranef`, `fitted`, the GLMM
  `stddev_se`), and a caller's warm-start θ is read in those units too. Always
  on, no trigger.

  What it fixes: on the `lme4/lme4-convergence` N400 subset, whose random-slope
  covariate has sd 1582 against an intercept of 1, the cold fit used to land on a
  different local optimum from lme4 — restricted logLik 238 units worse, the item
  covariance block collapsed. It now reaches REML criterion 395213.55, which is
  the well-scaled optimum (395184.08) plus the `4·ln(1582.449)` the REML
  `log|X'V⁻¹X|` term picks up because the same column is a fixed effect; lme4's
  own default fit on the unscaled data reaches 396013.25. A seeded crossed
  random-slope sweep that used to lose up to 1054 criterion units at a 1e6 column
  ratio now shows zero optimizer loss at every ratio through 1e6, and the GLMM
  sweep reaches the same optimum through a 1e2 ratio with the FD-Hessian
  standard-error route staying live where it used to fall back silently.

  Two diagnostics move with it. `PIN_THETA` and the `SINGULAR_REL_TOL`
  negligible-component check now test the internal (scaled) θ and standard
  deviations, which is what makes their verdicts independent of a covariate's
  units — the false `singular` those sweeps used to report is gone. This diverges
  from lme4's user-scale `isSingular` on a badly scaled design.

  Designs whose random-effect columns are all at scale exactly `1.0` — every
  intercept-only model — are bit-identical to before. Every design carrying a real
  scale factor moves within the floating-point reassociation band.

### Added

- **`Note::ReDesignScaleSpread { grouping, ratio }`** names a grouping whose
  random-effect design columns sit on very different scales (mirrors lme4's
  `checkScaleX`). Fitting is unaffected — that is what the internal scaling
  above is for — but the reported stddev is easier to misread, so the note
  says so. Raised by the formula frontend (`formula::Lowered::notes`), so a
  caller building `x`/`ModelSpec` by hand never sees it.

- **`Note::HessianSeFallback`** says `WaldSe::Hessian` was requested but the
  finite-difference joint Hessian was not usable, so the standard-error pass
  fell back to RX/Schur. Comes from the GLMM routes, dense and sparse, and only
  under `WaldSe::Hessian` — the fallback used to be silent.

- **Python: `glmm.ReDesignScaleWarning` and `glmm.HessianSeFallbackWarning`**,
  one per new note variant, both subclasses of `glmm.DiagnosticWarning` and
  filterable the same way. The public surface goes from six names to eight.

- **R: `fastglmm_re_design_scale_spread` and `fastglmm_hessian_se_fallback`**,
  two more classed conditions inheriting `fastglmm_diagnostic`. A port that did
  not know these variants would have routed them to `fastglmm_unknown_note`.

## [0.2.0] — 2026-08-07

A correctness release that breaks one thing on purpose. Near-collinearity is a
spectrum, and the crate treated most of it as failure: a design whose columns
were merely hard to separate came back all-NaN or quietly one column short, and
on the weighted-OLS route it came back wrong. The rank guards measured
`min|L_ii| / max|L_ii|`, which tracks column *scale*, not near-dependence. They
now measure the scale-invariant per-column pivot ratio, and below the floor the
dense LMM, OLS and GLM routes **fit and flag** — real coefficients, a large
standard error, a machine-readable note naming the column — rather than refuse.
Sparse LMM still refuses. Genuine redundancy is untouched: the alias gate keeps
dropping exactly-dependent columns at `ALIAS_EPS`, bit-identically.

The break is the diagnostics consolidation riding with it: `converged`,
`singular` and `aliased` — plus two internal channels that never reached the
result and the new ill-conditioned marker — now sit behind one
`fit.diagnostics`, and `Fit` plus the three new types become `#[non_exhaustive]`
in the same change, so the next diagnostic is additive instead of the next
break. Rust callers change field access to an accessor or one extra hop; Python
and R callers see additions only.

### Changed

- Raised the sparse PIRLS step-halving cap (`PIRLS_MAX_HALVINGS`) from 10 to
  16. The sparse GLMM's Wald-Hessian standard-error step cold-starts its
  finite-difference deviance evaluations from `û = 0`, and on a large-θ̂,
  many-crossed-grouping design that cold start needed one more halving than
  the cap allowed to walk back to the mode, hard-failing a fit that had
  otherwise converged cleanly. The cap now carries margin above the measured
  floor; no other PIRLS behavior changed, and no existing validation result
  moved (full corpus re-fit, bit-identical).

- **BREAKING — `converged`, `singular` and `aliased` moved off `Fit` into
  `fit.diagnostics`.** `Fit::converged()`, `Fit::singular()` and
  `Fit::aliased()` forward to them, so most call sites change by one character
  or not at all; what breaks is field access, struct literals and exhaustive
  destructuring. One storage location — the accessors read the same
  `Diagnostics`. `Fit::aliased()` returns `&[bool]`; ownership is
  `fit.diagnostics.aliased`.

- **BREAKING — `Fit`, `Diagnostics`, `Boundary` and `Note` are
  `#[non_exhaustive]`.** A one-time break so every future diagnostic, boundary
  state or note variant is additive. `Fit` is no longer constructible outside
  the crate.

- **The rank guards measure a different statistic, and refusal became
  flagging.** The condemning control: `y ~ 1 + u + w + (1|g)` with nothing
  collinear, varying only one column's scale — the old statistic fell six
  decades (5.4e-8 → 5.4e-14) while β̂ moved less than one part in 1e10, and
  three of those four fits were discarded. Every guard now uses
  `min_pivot_ratio` (per-column Schur pivot ÷ that column's own Gram diagonal,
  the alias gate's own basis). Floors: dense LMM and OLS/GLM flag below
  `1e-12`, sparse LMM refuses below `6e-10`. The `EPS_RANK` reveal-and-retry
  that dropped a column after the solver tripped is gone with its constant; the
  alias gate (`detect_aliased` on the raw `X'X` at `ALIAS_EPS`) is now the only
  place a column is ever dropped. `chol_rank_deficient` survives narrowed to
  `src/lme.rs`'s private Brent kernel, which fits off sufficient statistics and
  has no design in hand to take a pivot from.

- **The FD-Hessian θ step is absolute, not SD-scaled** (dense GLMM
  `se_hessian`). `fd_hessian_cov` stepped every joint coordinate at
  `FD_STEP_REL · max(1, |γ̂_k|)` — right for β, backwards for θ, where a larger
  random-effect SD flattens the profile and the rule let the O(h²) truncation
  error grow as θ̂². θ now steps absolutely (constant renamed
  `FD_STEP_BASE`, value unchanged at 1e-2); β keeps the relative form. Eleven
  fits moved, all dense GLMM `se_hessian` or θ-block SEs off the same Hessian,
  each named in advance and re-pinned with provenance; fits with
  `max(1, |θ̂|) = 1` are bit-identical. Against `glmer`
  `vcov(use.hessian = TRUE)` at `tolPwrss = 1e-13`: 3.9e-5 → 1.4e-5 (θ̂ = 2.97)
  and 1.3e-5 → 7.4e-6 (θ̂ = 4.51). glmm sits within 7.2e-6 of its own h→0
  stencil limit on sim_poisson_bigsd and under 4e-7 on sim_binomial_bigsd;
  lme4's value is itself a δ = 1e-4 finite difference carrying 5–9e-6.

- **The singular-fit warning stops asserting things the numbers contradict.**
  Both ports said `sd(term | group) = 0`; they now say `pinned at the variance
  boundary` — the pin fixes the Cholesky *diagonal*, so a q ≥ 2 block's
  reported stddev keeps the off-diagonal (measured 2.2e-3, which prints in a
  `VarCorr`). The `corr(a, b | group) = ±1` clause is removed from both ports:
  a q ≥ 2 pin *is* that event and `pinned` reports it reliably, while the old
  exact `abs(corr) == 1` test never fired (measured 1.0000000000000002). The
  ill-conditioned message likewise says "*b* is entangled with one or more
  other columns" — entanglement is symmetric and the kernel names whichever
  column its pivot search reached. See the corrected 0.1.1 entry below.

- **The NB θ-search stopping width is `1e-4` on `ln θ`, was `1e-8`**
  (`golden_max_ln_theta`, shared by the GLM conditional θ profile and the
  GLMM marginal-θ route). Every evaluation the search makes is a full inner
  fit converged only to its own noise floor (`GLMM_RHO_END = 3e-6`), and the
  old width asked for four decades more resolution than that inner fit
  could supply — the last ~13 of ~45 iterations were picking a side of a
  knife-edge in the inner fit's own basin selection, not resolving curvature.
  Traced on the dense NB GLMM fixture: a 1-ULP input perturbation used to
  flip the reported β by 9.5e-5 relative through exactly this mechanism; at
  the new width the same perturbation converges to a bit-identical β. Moves
  β/θ̂/varcorr on every NB fit that reaches the search (dense and sparse
  GLMM, the fixed-effects GLM θ profile) by amounts inside their existing
  lme4-facing bands; no cross-engine golden moved.

- **`bobyqa` bumped `0.1.3` → `0.2.0`** — the optimizer every LMM and GLMM
  θ-search runs on. The crate's own call sites are unchanged: `Config`,
  `Bobyqa`, `RestartConfig`, `Status` and `Outcome` are used exactly as
  before.

### Added

- **`Diagnostics`, `Boundary` and `Note`**, re-exported from the crate root.
  `Diagnostics` carries `converged`, `singular`, `aliased`, `boundary`,
  `pinned`, `notes`. `Boundary` is `Interior` / `AtBoundary` / `NoOptimum` —
  the last is the one fact nothing exposed before (optimizer cap-out,
  previously inferable only from a finite `deviance` with `converged` false).
  `Note` has three variants. `IllConditioned { columns, pivot }` carries the
  measured pivot so callers can rank severity.
  `PirlsExhausted { evals, final_eval }` says a GLMM's inner PIRLS solve ran
  the full 50-iteration cap without meeting its band; `final_eval` separates
  the case that matters (the solve behind the reported estimates) from a
  rejected BOBYQA trial point. `UnusedGroupingLevels { grouping, levels }` is
  raised by the formula frontend, not a solver, and names declared levels that
  carry no row but still occupy random-effect columns. A fourth, anticipated
  variant was dropped: measurement found no regime where β is noise *and* the
  standard error lies about it, so there is nothing for a refusal to protect.

- **`Diagnostics::pinned`** — which variance components collapsed, as flags
  aligned with the `varcorr` blocks (`pinned[g][i]` pairs with
  `stddev_corr(g).0[i]`), replacing an internal bitmask keyed to a non-public
  order. Empty means "nothing to report"; `AtBoundary` with empty `pinned`
  means something pinned on a route that cannot say which.

- **Coverage is documented rather than assumed.** `converged` and `aliased`
  are filled on every route; `boundary` and `pinned` are real wherever
  variance components exist; `notes` is per-variant. `IllConditioned` can only
  be raised by OLS, GLM and dense LMM — dense GLMM records no pivot, sparse
  refuses instead of flagging. `PirlsExhausted` comes from the GLMM routes,
  dense and sparse. `UnusedGroupingLevels` comes from the formula frontend
  (`formula::Lowered::notes`), so a caller building `x`/`ModelSpec` by hand
  never sees it. An absent note means "not detected", never "checked and
  clean".

- **Python: `fit.diagnostics`** (dict, same six keys) plus one warning class
  per note variant — `glmm.DiagnosticWarning` (base, a `UserWarning`),
  `glmm.IllConditionedWarning`, `glmm.PirlsExhaustedWarning` and
  `glmm.UnusedGroupingLevelsWarning`. Filter the whole channel or one variant
  with `warnings.filterwarnings`. `columns` are 0-based indices into
  `Fit.names`.

- **R: `fit$diagnostics`** (list, same six names); every existing top-level
  name is unchanged and `isSingular()` is unaffected. Notes arrive as classed
  conditions — `fastglmm_ill_conditioned`, `fastglmm_pirls_exhausted`,
  `fastglmm_unused_grouping_levels` and `fastglmm_unknown_note` (the forward
  compatibility arm for a variant this version of the port does not know),
  all inheriting `fastglmm_diagnostic` — selectable without matching message
  text. `columns` are 1-based on this side.

- **The Gaussian LMM paths now report `ranef` and `fitted`.** Both were empty
  there since 0.1.1 — those paths fit off sufficient statistics and never
  formed per-row quantities. The conditional modes are now recovered at θ̂
  after the fit, and the fitted means built from them (offset restored), so
  `Fit::ranef`/`Fit::ranef_levels`/`Fit::fitted` are filled on every converged
  route. Checked against a brute-force solve of the penalized normal equations
  that shares no code with the recovery (`tests/lmm_ranef.rs`). This is what
  the R `ranef()`/`fitted()` methods and Python `ranef_blocks` rest on for an
  LMM.

- **Labelled conditional modes.** The kernel has carried `ranef` and
  `ranef_levels` as flat numbers since 0.1.1, but the layout a grouping lands
  in is a data-dependent decision inside the kernel, so slicing them by hand
  was never safe. `formula::label_ranef` resolves those numbers back to
  `RanefBlock`s — grouping name, term names, level labels, values — using the
  per-slot labels the lowering now keeps (`ReGroupInfo::slot_labels`), and
  drops a nested grouping's padded slots so `levels` is exactly the levels that
  exist. It returns the new `formula::Error::RanefShapeMismatch` rather than
  panicking when a `Fit` and a lowering do not belong together.

- **Python `Fit.ranef_blocks`** — the same labelled form, a list of dicts with
  `group` / `terms` / `levels` / `values`. Not a DataFrame: NumPy is the
  package's only dependency. Empty exactly when `ranef` is.

- **R `ranef()` and `fitted()` work.** Both were hard errors. `ranef()` returns
  lme4's shape — a named list of data frames, one per grouping, rows labelled
  by level and columns by term. `fitted()` returns the conditional means μ̂ per
  row of the model frame, named by its row names, including the random-effect
  contribution and any offset. `predict()` and `residuals()` still error with
  their reasons; `residuals()`' reason is that lme4's `type` argument picks
  between four different quantities and guessing would silently disagree.

- **The `orchestrate` cargo feature** (off by default) — the string-typed fit
  orchestration both FFI ports need: one definition of the family/link string
  vocabulary, the formula-and-data lowering, and the flattened result they
  publish, plus the panic-to-`Err` boundary. `glmm-python` and `glmm-r` each
  carried a mirrored copy (`orchestrate.rs` + `convert.rs`, ~500 lines apiece);
  both copies are deleted and both ports now call `glmm::orchestrate`. Like
  `loop_advanced` and for the same reason, this is **not** part of the
  semver-covered surface — its shape follows the ports' needs.

- **Two frozen lme4 references for designs the crate used to discard.**
  `validation/prep/gen_illcond_data.R` emits `sim_dynrange_lmm` and
  `sim_entangled_pair_lmm`, bit-identical to the crate's builders, CSVs exact
  at 17 digits. The first is a registered cross-engine golden (β within
  6e-11); the second bands the entangled pair itself — 5.7e-4 on the two
  entangled coefficients, 4.8e-7 on their identified sum, against 1e-3.

- **Large-θ̂ validation coverage.** Every GLMM rung's RE SD sat in
  [0.34, 1.34], so the suite had never seen the Laplace approximation's weak
  regime. Three committed datasets — `sim_binomial_bigsd` (θ̂ = 4.51),
  `sim_poisson_bigsd` (θ̂ = 2.97), `sim_binomial_zerosd` (θ̂ exactly 0) — plus
  two frozen lme4 rung references, two AGQ goldens at nAGQ = 7 and 11, a
  validated per-rung tighten-only `se_hessian` band (the two rungs gate at
  3e-5, where the default 1e-3 would have caught nothing), and an in-crate
  convergence-flag assertion at the θ → 0 boundary, where lme4 says
  `isSingular` and glmm says `converged = true, singular = true` — a real
  divergence `compare.R` cannot express.

- **Runnable examples and a documentation index.** `documentation/index.md`
  plus `examples-python.md` / `examples-r.md` and the nine paired scripts each
  walks through (`documentation/examples/{python,r}/`), which check themselves
  against lme4 values via a small oracle helper, and
  `coming-from-statsmodels.md`. The three tutorials moved to lowercase
  filenames (`tutorial-python.md`, `tutorial-r.md`, `tutorial-rust.md`).

- **`logLik()`, `AIC()` and `BIC()` work in the R package.** The kernel has
  carried `loglik`/`df`/`reml` since 0.1.1 and Python exposed them; the R fit
  object dropped all three on the way out behind a stale error. `logLik()` now
  returns a `"logLik"` object with `df`, `nobs`, `REML` attributes. On an LMM
  the value is the REML criterion (`REML = TRUE`), comparable only across
  identical fixed effects — which is why `summary()` still prints no
  `AIC BIC logLik` line. R-side plumbing only.

- **`offset=` works in the R package.** Rejected with a message whose stated
  reason (`the kernel has no offset field`) stopped being true in 0.1.1.
  `fastglmm(..., offset =)` now takes a per-row additive term on the
  linear-predictor scale, following `weights=` at every site: evaluated in
  `data` (so `offset = log(exposure)` works) and parked in the model frame so
  `subset=`/`na.action` drop entries row-locked. The `offset()` *formula term*
  is still an error, now saying only that. The new formal sits after
  `na.action` (where `stats::glm` puts it), so a call passing `nAGQ` or later
  arguments positionally shifts by one. The R port now fits the offset rung,
  covered by the port gate at 9e-16.

### Fixed

- **Weighted OLS returned silently wrong coefficients and called them
  converged.** On a design full-rank on the raw `x` but singular once prior
  weights apply, the old guard's statistic bottomed out four orders *above*
  its own threshold even where `X'WX` was numerically indefinite, and the
  pre-dispatch alias gate tests the raw `x` (healthy at every rung). The crate
  returned `β = −1527` for a true 0.477 with `converged: true`. The root cause
  was the statistic, not the threshold. Such a design now fits with an SE more
  than 100× its coefficient and an `IllConditioned` note.

- **The GLM divergence guard rejected well-behaved fits in the wrong units.**
  It bounded `|β_j| > 30` at iteration ≥ 3, which is not a property of the
  model: rescaling a predictor column rescales β̂ exactly, so the same fit was
  accepted or refused depending on the caller's units. `y ~ x/1000` came back
  `converged: false` with β̂ = (0.4915, 803.09) and NaN standard errors while
  the identical model at unit scale converged, on both the logit and the
  Poisson log link. The bound is now on the linear predictor
  (`ETA_DIVERGENCE_CAP`, same value 30), which `η = Xβ` leaves unchanged under
  a unit change — and where 30 is the number the argument supports (on the
  logit scale `|η| = 30` is p ≈ 1 − 1e-13, separation rather than signal). The
  guard is skipped under the Gamma inverse link, where `η = 1/μ` makes a large
  `|η|` an honest small-mean fit; that pairing falls through to `clamp_eta`,
  the non-finite β check and the iteration cap as before. `BETA_BOX` (the
  joint BOBYQA's β box) was aliased to the old cap and is now its own
  constant, unchanged at 30, so the two stop moving together by accident.

- **A pinned variance component on a q ≥ 2 grouping went unreported in both
  ports.** The ports reconstructed the pin set by scanning `varcorr` for exact
  zeros — which a q ≥ 2 pin never produces (stddev keeps the off-diagonal,
  measured 2.2e-10; the corr fallback missed at 1.0000000000000002), so the
  user got the bare lme4 text. Both reconstructions are deleted; the ports
  read `Diagnostics::pinned`, the kernel's own record.

- **Sparse fits now name their collapsed components.** Both sparse routes ran
  a per-component pin loop but recorded one "something pinned" bool; they now
  build the same mask the dense routes do. Diagnostics only — no fitted number
  moved.

- **A subnormal response punched a `-inf` hole in the Poisson and NB
  objective.** Both deviance residuals computed `y·ln(y/μ)`; for subnormal `y`
  and `μ ≥ 2` the quotient underflows to exactly `0.0` before the log, so the
  term evaluated to `-inf` where the true value is finite and tiny. Written
  `y·(ln y − ln μ)` instead, which is the same quantity with each log taken on
  a representable argument (`src/family.rs`). Only reachable with a response
  at the bottom of the exponent range; no fitted number on any normal-range
  data moved.

- **An aliased column used as a random slope returns a fit instead of
  panicking.** `remap_spec_slopes` asserted; it now returns a non-converged
  `Fit` (NaN β/se). Actually fitting the reduced model — dropping the random
  slope with its aliased fixed column — is a different model with its own
  oracle, and is not implemented here.

### Removed

- **`LmeScratch::ols_scratch`** (`loop_advanced`, no semver guarantee):
  provisioned for a τ̂ ≈ 0 OLS fallback never implemented — the boundary case
  pins θ and runs the same profiled-deviance path. Written and read by nobody.

### Changed — `loop_advanced` (no semver guarantee)

Every changed item, named because MCPower consumes this tier.

- `FitView::diagnostics() -> FitDiagnostics` **added** — a `Copy` struct read
  off borrowed state, no per-draw allocation.
- `FitView::boundary_hit()` / `FitView::pinned_components()` **removed**; read
  them off `diagnostics()`. `converged()` is unchanged and forwards there.
- `fit_suff_stats_t_sq` **lost its `eps_rank` parameter** — once OLS stopped
  refusing it guarded nothing.
- **The dense LMM route lost its rank guard at the θ-search endpoint** (the
  `EPS_RANK = 1e-8` test on `X'V⁻¹X` at θ̂ that NaN-filled the fit); it fits
  and flags there instead. `fit_on` drives this route — see Migration. The
  third copy of the predicate, on `src/lme.rs`'s Brent kernel, is untouched.
- **`pinned_components` is always `0` on the sparse and NB (`Prebuilt`)
  draws** — indistinguishable from "nothing pinned". Those routes do fill
  `Diagnostics::pinned` on the assembled `Fit`; the `Copy` carrier holds no
  `Vec`. Read the cold surface when per-component pinning matters.
- `GlmFitView`, `OlsFitView`, `LmmFit` **gained `pivot` / `pivot_col`**.
- `LmmFit::eps_rank_aliased` **removed** with the reveal-and-retry gate.

### Migration

- **Rust:** `fit.converged` → `fit.converged()` or
  `fit.diagnostics.converged`; same for `singular`/`aliased`. `Fit` literals
  and exhaustive matches need rewriting; `..` covers the match case.
- **Python — one real removal, sharper than it looks.** Attribute reads are
  unchanged (`fit.converged` etc. remain as properties), but the three left
  the dataclass **field list**: `glmm.Fit(converged=...)` no longer
  constructs, and `dataclasses.asdict(fit)` silently stops carrying the three
  most-read diagnostics — nothing raises, the keys are simply absent. Code
  round-tripping a `Fit` through `asdict` must read `fit.diagnostics`.
- **R:** nothing breaks; every current name reads the same value.
- **Loop-tier callers: draws that used to arrive as NaN now arrive as
  numbers, on every dense route including OLS. Aggregate without checking the
  flag and your results move.** Three causes: the alias gate runs in
  `fit_warm`, so `fit_on` bypasses it and a rank-deficient draw reaches the
  solver whole (NaN vs fitted-and-flagged is then settled by the arithmetic on
  that draw, not predictable from the design); `fit_suff_stats_t_sq` lost its
  non-scale-invariant guard, so full-rank draws it wrongly discarded (a merely
  rescaled column crossing 1e-12) now come back; and the dense LMM endpoint
  guard is gone, at a threshold four decades looser, so badly-scaled LMM draws
  start moving at column scale 1e-10, not 1e-12. Differentially measured (6
  sizes × 3 seeds, old and new trees in one binary): duplicate-column LMM
  draws that arrived NaN now arrive fitted on 16/18 designs (11/18 with the
  duplicate rescaled); nothing anywhere moved fitted → NaN. Screening is the
  caller's job: read `FitView::diagnostics()` on every draw and check
  `converged` **and** `ill_conditioned` before it enters an aggregate
  (`pivot_col` names the column). A filter dropping only non-converged draws
  silently admits the newly fitted ones.

### Not changed — measured null result

- **Sparse FD-Hessian seeding was implemented on a scratch tree and
  rejected.** The dense path's cold-start defect does not fire on the sparse
  arm (cold `f0` reproduces the converged deviance to ≤7.9e-7 on eight
  fixtures, non-vacuity proven by injection), and at the shipped
  `SPARSE_FD_STEP_REL = 1e-4` the warm seed is actively harmful — PIRLS's
  relative-increment exit trips early, moving `sim_sparse_gamma` `se_hessian`
  by −27% and `sim_sparse_nb` by −61%. Step-coupled (works at h = 1e-2, fails
  at h ≤ 1e-3); blocked on a sparse step recalibration. No code landed;
  `src/sparse/glmm.rs` deliberately keeps its `max(1, |θ̂|)` scaling, which is
  calibrated on the noise side there.

### Internal — test pins

- Seven in-crate pins moved off the flat `PIN_REL_ITER` onto per-test bands
  (5e-6 to 3e-3) sized from measured aarch64-apple-darwin drift; on the
  reference machine every one is bit-exact across all four feature configs.
  `assert_pinned` names the machine and holds the account.
- The `alloc-tests` bounded-allocation tests serialize themselves through
  `test_support::alloc_test_guard` (dhat counts process-wide), so they no
  longer need `--test-threads=1`.
- The `pending_reference` golden-exclusion flag's dead paths are now driven by
  synthetic specs; proven non-vacuous by inverting the predicate.
- **`fit_glmm_nb_sim_matches_lme4` and `fit_glmm_nb_nested_unbalanced_matches_lme4`
  gained an additive bit-exact Rust-vs-Rust pin** (`BAND = 1e-7`, alongside their
  existing lme4 bands): entry 9 found neither dense NB test could tell a
  regression from rounding, since their only assertions were lme4-facing
  bands wide enough to absorb the old θ-search instability. Both fixtures
  clear a 1e-5 conditioning gate by 3+ orders of magnitude at the new
  stopping width, under both a 1-ULP sweep and the lane-width probe.
  **`fit_sparse_nb_glmm_is_pinned` dropped its bit-exact pin** (the crate's
  thinnest-ever margin, on its worst-conditioned NB fit) for oracle
  agreement against the frozen `sim_sparse_nb.json` golden instead — the
  sparse route's own coverage comes from two live both-paths cross-checks
  against dense, not a second frozen-Rust value.

## [0.1.3] — 2026-07-29

An allocation release. Nothing moves an answer and nothing on the public
surface changes shape: the full validation manifest refits bit-identically
before and after every change. Two changes — the dense GLMM workspace stops
allocating large matrices its common routes never read, and the
`loop_advanced` build-once/fit-many tier stops allocating per draw — plus the
memory harness that measured them.

### Changed

- **The dense GLMM `Z` matrices are allocated only on the route that reads
  them.** The dense GLMM workspace allocated three `n × k_total` matrices
  (`Z`, `M`, `WM`) unconditionally, but the two common solve routes never
  read `Z`: the blocked route reconstructs per-cluster blocks on the fly,
  and the structured route needed it only inside `build_packed_m`, which now
  builds its packed products directly from the column structure. The
  workspace now sizes those buffers to 0×0 unless the model routes to the
  dense fallback (the one route that genuinely reads all of them, unchanged).
  Measured peak RSS on large models, Rust binary: a 50,000-row random-intercept
  fit with 800 levels drops 952 → 27 MB, an observation-level-RE fit
  (10,000 rows, 10,000 levels) 3827 → 12 MB, four correlated slopes plus a
  crossed grouping 2765 → 51 MB; on the validation manifest the multi-grouping
  rungs shrink the same way (VerbAgg 54 → 16 MB) and everything else is flat.
  Net of each runtime's baseline, the kernel's fit cost on the blocked shape
  is now below lme4's.

- **The `fit_on` loop tier no longer allocates per draw** (`loop_advanced`
  feature). The `Ols`, `Glm` and dense-LMM arms each built a fresh
  column-major `n × p` copy of `x` on every call; they now fill a buffer
  preallocated once in `build_workspace`, the same pattern the dense-GLMM arm
  already used. The LMM offset path likewise reuses a preallocated `y − o`
  buffer, and unweighted OLS workspaces reclaim the `scaled_x` matrix they
  never touch. Per-draw heap traffic on those arms is gone. Measured on a
  locked clock (pinned P-core, min over repeats, both versions producing
  bit-identical estimates): 2–3 % per draw on OLS, ≤1 % on dense LMM, flat
  on GLM (IRLS iteration cost dwarfs one allocation), at n = 1000–10000.
  An allocation-hygiene change, not a speedup headline.

### Added

- `validation/memory/` — a peak-RSS measurement harness over the 43 manifest
  rungs plus 13 large synthetic models, with cross-engine baselines (Rust
  binary, Python and R ports, lme4, MixedModels.jl) and a summariser
  (`validation/summarize_memory.R`). Measurement tooling only; no gate, no
  golden.

## [0.1.2] — 2026-07-22

A fix release with an internal restructure. Two changes move an answer, both
narrow: the Gamma inverse-link PIRLS boundary fix (a cell that reported false
convergence now lands on lme4's optimum) and the formula random-effect ordering
fix (a formula that writes a plain-intercept term before a slope term now takes
the written order as its primary grouping). Everything else — the whole log-link
and binomial corpus, single-RE and slope-first formulas — refits bit-identically,
and the oracle goldens hold at their existing tolerances.

### Changed

- Restructured the validation suite: `parity/` is now `validation/` (package
  `validation`, example `validation_fit`, test file `tests/validation_oracle.rs`);
  the prior-weights suite merged in as manifest rungs 29–43 (`tier: "weights"`);
  the finished grid/diligent/accuracy studies archived under
  `validation/campaigns/{speed-grid,estimate-grid,monte_carlo}/`. No gate,
  tolerance, golden, or dataset changed.

- **The formula frontend now lowers random effects in formula order.** The
  parser used to emit random effects in its internal extraction order (slope
  terms before plain intercept terms), so in a formula like
  `y ~ x + (1|g) + (1+x|h)` the slope grouping `h` became the primary grouping
  even though `g` was written first. Random effects now follow the order they
  are written, with one exception: a nested `(1|A/B)` pair still sorts first,
  because the kernel interprets nesting relative to the primary grouping. For
  formulas that write a plain-intercept term before a slope term this changes
  the primary grouping, and with it: which solver the model routes to (a slope
  on an extra grouping is a sparse-routing trigger), the packing order of the
  θ variance components, and the order of `ReGroupInfo` blocks. Single-RE
  models, all-intercept models, slope-first models, and anything with a nested
  term lower exactly as before. This fix re-landed parity rung 24
  (`sim_sparse_gamma`) at unchanged tolerances: the misordering had routed it
  to the dense kernel, whose optimizer stops ~2e-3 deviance short on that
  shape, while the formula-order orientation routes sparse and lands 2e-4
  from lme4's optimum.

### Fixed

- **Gamma inverse-link PIRLS could converge on the η > 0 domain boundary.**
  `clamp_eta` projects a trial iterate with η ≤ 0 (where μ = 1/η is undefined)
  onto η = 1e-10, and the projected row's working weight μ² ≈ 1e20 then
  dominates the WLS solve, so PIRLS kept returning the boundary and reported
  convergence there. Routed through the sparse solver, the `sim_gamma`
  inverse-link cell returned `converged = true` at an optimum ~937 deviance
  units above lme4's; the same mechanism put a ~98-unit discontinuity in the θ
  surface BOBYQA minimizes on the dense path, which had been reaching the right
  optimum only because its warm-start chain stayed feasible. All four PIRLS
  drivers now treat a domain-infeasible trial iterate as a failed step and
  halve toward the last accepted feasible iterate (R `glm.fit`'s
  `valideta`-style step-halving); a first trial with no accepted predecessor
  backtracks toward the u = 0 seed, and an infeasible η_fixed itself surfaces
  as an honest non-converged NaN. Every family/link whose η domain is all of ℝ
  — the whole log-link and binomial corpus — refits bit-identically
  before/after the change; the repaired sparse cell is pinned against the dense
  fit and the frozen lme4 golden (`sim_gamma_inv_glmm`).

- **`Sizing::n_clusters_at` under-counted clusters off-grid.** Under
  `Sizing::FixedSize` it divided `n / cluster_size` rounding down, while its
  neighbour `Sizing::cluster_of_row` sends row `i` to cluster `i / cluster_size`.
  With `n = 18, cluster_size = 4` row 17 lands in cluster 4, so five clusters
  exist and the function reported four — the trailing partial cluster is real
  and its id must be in range. It now rounds up, matching the workspace
  allocator, which had been carrying its own private copy of the corrected
  formula. Off-grid `n` only; on an atom multiple the two agree, so no shipped
  path changes answer.

## [0.1.1] — 2026-07-18

Additive release: an offset term and post-fit reporting fields (log-likelihood,
AIC/BIC df, fitted means, conditional modes). Nothing on the stable surface
changed shape, so every 0.1.0 fit keeps its result up to optimizer tolerance;
the oracle goldens hold at their existing tolerances.

### Added

- **`FitOptions::offset`** — a per-row additive term on the linear-predictor
  scale, `η = offset + Xβ (+ Zb)`, matching R's `glm(offset=)` / `glmer(offset=)`.
  A fixed known contribution, not a parameter (β must not absorb it); the
  canonical use is a Poisson exposure, `offset = log(exposure)`. Supported on
  every path — OLS, GLM, LMM, GLMM (dense and sparse) — with identity-link
  paths applying it as an exact `y − o` shift and `Fit::fitted` still reporting
  means on the original `y` scale. Also on the Python `fit(offset=)`. A new
  `sim_poisson_offset` parity rung (28) pins it against `glmer(offset=)`. The R
  port (`fastglmm`) still rejects `offset=` / `offset()` by design.
- **`Fit::loglik`, `Fit::df`, `Fit::reml`** — the log-likelihood at the fitted
  parameters, `deviance` with its dropped data-only constants restored onto
  lme4's `logLik()` scale, the AIC/BIC parameter count, and a flag marking the
  LMM REML criterion. Together they give `AIC = 2·df − 2·loglik` and
  `BIC = df·ln(n) − 2·loglik` on every path. `loglik` matches `lme4::logLik`
  including the aggregated-binomial `cbind(s, m−s)` form under `weights=`; on
  the Gaussian LMM paths it is the REML criterion `−REMLcrit/2`, comparable only
  between models with identical fixed effects — `reml` is set there, mirroring
  lme4's REML-fit `anova` warning. `df` counts retained fixed effects (lme4's
  NA-coefficient handling for aliased columns) + RE θ parameters + 1 where the
  family estimates a dispersion/scale.
- **`Fit::fitted`** — fitted means `μ̂` per row through the inverse link (lme4
  `fitted()`). Empty on non-converged fits and on the Gaussian LMM paths, which
  fit via sufficient statistics and never materialize per-row means.
- **`Fit::ranef`, `Fit::ranef_levels`** — random-effect conditional modes `b̂`
  (BLUPs), one block per grouping in `varcorr`/`re_groups` order, level-major,
  with `ranef_levels` giving each grouping's level count for slicing. Empty on
  the same paths as `fitted`.
- All six new fields cross the Python and R shims onto their fit results
  (`Fit.loglik`/`df`/`reml`/`fitted`/`ranef`/`ranef_levels` in Python).
- Seven user-facing guides under `documentation/`: `installation.md`,
  `formula.md`, `conventions.md`, `coming-from-lme4.md`, `glmm-design.md`,
  `validation.md`, `troubleshooting.md`. The Python and R READMEs now link them
  instead of inlining the formula and factor-coding rules.

### Changed

- **The boundary (singular) fit warning now names the degenerate components.**
  lme4's exact text (`boundary (singular) fit: see help('isSingular')`) is
  extended with `sd(term | group) = 0` per collapsed variance and
  `corr(a, b | group) = ±1` per degenerate correlation. Exact comparisons are
  safe because the kernel pins boundary components to exact 0 / ±1; the bare
  lme4 text is kept when only the relative-tolerance singular check fired. The
  Python and R ports emit the same extended message.

  > **Corrected 2026-08-01, and both halves of the claim were wrong.** The
  > `corr(a, b | group) = ±1` clause no longer exists in either port — see the
  > 0.2.0 entry above — and the exactness argument it rested on was never
  > true. What the kernel pins is the Cholesky diagonal, not the reported
  > standard deviation or correlation: on a q ≥ 2 block the stddev keeps the
  > off-diagonal and lands at ~1e-10, and the correlation was measured at
  > 1.0000000000000002. The `corr(...)` clause's exact comparison therefore
  > never fired on the designs it was written for, which is why it was deleted
  > rather than reworded. The `sd(...)` half did ship and does fire — its
  > scan-for-zero catches a q = 1 pin, where the pinned value really is exact
  > 0.0, and misses a q ≥ 2 one for the same off-diagonal reason; that is the
  > bug fixed in the 0.2.0 entry, along with the wording.

## [0.1.0] — 2026-07-16

First release of the Python package (`glmm` on PyPI), and the first crate
release since `0.0.2`. The breaking changes below are real breaks against the
published `0.0.2` — `ModelSpec` is now structure-only and the `mcpower` feature
is renamed `loop_advanced`. `0.0.3` was never published, so it is not a
migration source.

All four estimators are wired into the stable `fit` dispatch: OLS; GLM
(Gaussian, binomial logit/probit, Poisson, Gamma, negative binomial); LMM
(closed-form single-intercept + BOBYQA general); GLMM (dense and sparse-Z, all
families including NB), with AGQ (nAGQ > 1) for up to 3 random effects per
group (single grouping factor, binomial/Poisson).
Validated against R/lme4 and Julia/MixedModels.jl across a 23-rung dataset
parity manifest plus a 15-rung prior-weights harness.

### Fixed

- **A factor's level order is no longer silently discarded.** `glmm::formula`
  sorted every factor's levels lexicographically, so the treatment-contrast base
  was whichever label sorted first, regardless of what the caller asked for. A
  deliberately ordered categorical — `pd.Categorical(x, categories=["low",
  "med", "high"])` — was refactored to base `"high"`, returning a different
  coefficient for a different question with nothing in the output to reveal it.
  `Column::Factor` now takes `{ levels, codes }`, so the caller states the order
  and level 0 is the base. Python passes a `Categorical`'s
  `categories`/`codes` through; a plain string column has no declared order and
  is sorted by `Column::factor_from_labels` — the same lexicographic default as
  R's `factor()`, now a default rather than an imposition.
- Python: a categorical of non-strings (`pd.Categorical([1, 2, 3])`) was
  classified numeric and fit as one continuous slope instead of expanding to
  dummies. Column classification now checks the dtype before sniffing values.
- Python: `summary()` printed `group 0` instead of the grouping's name, and its
  per-term rows carried no labels — `Lowered::re_groups` was never carried
  across the PyO3 shim. It now is, and `summary()` prints e.g. `Subject:` with
  `(Intercept)`/`Days` rows.

### Added

- `glmm::formula` — the R-style formula frontend is now part of the crate,
  behind the `formula` feature (on by default). `lower("y ~ x + (1|g)", &table,
  family)` builds the kernel's inputs from a formula string and a data table.
  Previously an unpublished companion crate, so it was unreachable for anyone
  installing from crates.io.
- `default-features = false` gives the formula-free kernel, which links no
  `regex` — the configuration for parse-once/fit-many hot paths.
- **`Fit::vcov`** — the full `p×p` fixed-effect covariance `Cov(β̂)`, on every
  path. `Fit::se` is its diagonal and cannot answer anything about two
  coefficients jointly, so a contrast, a confidence interval, or anything of
  `vcov()`/`confint`/`glht`/`emmeans`'s shape needed off-diagonals that were
  being computed and thrown away (GLMM) or never formed (OLS/GLM/LMM). It is
  finite exactly where `se` is. Also on the Python `Fit`, as a `(p, p)` array.
- Python `Fit` gained `n_eval` (optimizer evaluation count), `deviance` (the
  minimized criterion — **not** comparable across models, see the docs), and
  `re_groups`. All three were already on the Rust `Fit`; none crossed the shim.

### Changed

- **Python: `theta=` is renamed `init_theta=`.** One call had two unrelated
  parameters named `theta`: the negative-binomial shape seed and, inside
  `warm_start={"theta": …}`, the random-effect Cholesky vector. The seed takes
  the name R already uses for it (`MASS::glm.nb(init.theta=)`);
  `warm_start["theta"]` is unchanged, matching lme4's `start=list(theta=)`.
- **Python: `targets=` is removed.** It exposed `FitOptions::target_indices`, a
  performance knob for MCPower's hot path that leaves non-target SEs `NaN`. That
  hot path drives the Rust surface directly, where the option is unchanged; no
  Python caller wants `summary()` printing `NA` for standard errors it could
  have computed.
- Python: the native call returns a dict keyed by field name rather than a
  positional tuple. Internal, but it is why `re_groups`/`n_eval`/`deviance`
  could go missing unnoticed.

### LMM cold start

#### Changed

- **LMM cold starts now use the unit-diagonal blind seed** (diagonal θ at 1,
  off-diagonal vech entries at 0 — the lme4/MixedModels convention), on both
  the dense (`fit_lmm`) and sparse (`fit_mle_sparse`) Gaussian paths. The
  former start set *every* component to 1; on wide-slope designs (q ≥ 4 with
  correlated slopes) that start funneled BOBYQA into a second-best local
  optimum on 8 of 9 adjudicated grid cells (deviance gaps +0.23 to +57.4 vs
  the best-known optima, now frozen as goldens under `parity/goldens/optima/`).
  With the new seed the fitted optimum matches or beats MixedModels on all 9.
  Intercept-only and uncorrelated-slope models have no off-diagonal
  components and are bit-identical. Full-grid effect vs MixedModels on the
  gaussian slope stratum: worse-than-MM cells drop 8 → 2 — the two
  remaining are *new* coin-flips where the old start happened to hold the
  best-known basin (`lmm_q6_g300p5_bal_base` +0.008,
  `lmm_q8_g3000p5_bal_lowsnr` +2.03; goldens frozen for both). It also fixes
  the dense-vs-sparse basin disagreement behind the `noz_sparse_grid_agrees`
  cell-20 failure. Eval counts on affected wide-slope fits move both ways
  (grid-wide gaussian-slope total −10%). The sparse non-Gaussian GLMM joint
  seed already used this shape; the Gaussian paths now match it.

### Prior weights

#### Added

- **`FitOptions::weights`** — per-row prior (case) weights, lme4's `weights=`.
  An aggregated binomial (y = success proportion, weight = trial count) now
  fits directly — lme4's `cbind(s, m−s)` objective, which shares its argmin
  (and so β/SE/varcomp) with the expanded-Bernoulli fit — letting the
  `sim_sparse_binomial` parity rung fit its 240 aggregated rows instead of the
  3,059-row Bernoulli expansion. Parity holds at unchanged tolerances; the
  per-solve O(n·width²) cost collapses accordingly.

#### Changed

- `FitOptions.weights` now supported on all paths (was: sparse binomial GLMM
  only); nAGQ>1 with weights rejected.

### Two-stage GLMM optimizer

#### Changed

- **GLMM fits now use a two-stage optimizer** (lme4's structure, Bates et al. 2015
  §3): a fast θ-only search profiles the fixed effects β out per PIRLS iteration,
  then a short joint (θ, β) polish on the exact Laplace objective warm-started from
  it. The converged (θ̂, β̂) and all standard errors are unchanged up to optimizer
  tolerance — the parity goldens hold at their existing tolerances — but the outer
  evaluation count drops materially (roughly 2× fewer BOBYQA evaluations on the
  grouseticks 3-crossed Poisson fixture). The prior single-stage joint solve remains
  available as an internal A/B toggle. `Fit::n_eval` now includes stage-1
  evaluations, so eval counts are not directly comparable to versions before this
  change.

#### Added

- **PIRLS step-halving.** The inner penalized-IRLS loop now backtracks (halves the
  step, up to 10 times) when a full Fisher-scoring step raises the penalized
  deviance, hardening convergence on ill-scaled joint (u, β) steps; an exhausted
  backtrack surfaces as the existing non-converged/NaN failure state.

### M3.5 — warm-start entry-split

The fit surface now separates model *structure*, optimizer *warm-start state*, and
method *knobs* into three distinct places (`docs/GLMM/api.md`, Layers A–C). The
stable `fit`/`fit_grouped` signatures are unchanged; the breakage is in the shapes
they consume.

#### Changed (breaking)

- **`ModelSpec` is structure-only.** Removed the method knobs `wald_se` and `nagq`
  and every magnitude payload — a `ModelSpec` can no longer carry a start estimate.
  - `ReStructure` and `Grouping` lost `tau_squared` and now hold
    `slopes: Vec<ColumnId>` (the `SlopeTerm` struct, which bundled a column with its
    variance/correlation magnitudes, is deleted along with the `re_correlation_*`
    helpers).
  - `Family::Gamma` lost its `dispersion: Option<f64>` payload;
    `Family::NegativeBinomial` lost its `theta: Option<f64>` payload.
- **`FitOptions` gained the relocated knobs:** `wald_se`, `nagq`, and `dispersion`
  (the Gamma fix-vs-estimate directive). All are defaulted — construct with
  `..FitOptions::default()` (Wald SE `Hessian`, `nagq` 1 = Laplace, `dispersion`
  `None` = estimate φ post-fit). `FitOptions` now implements `Default`.
- **The stable `fit`/`fit_grouped` cold-start the optimizer** — they no longer derive
  a warm start from spec magnitudes; the kernels use their `THETA0` blind start. The
  converged MLE is unchanged up to optimizer tolerance (start-independent), so the
  oracle goldens stay green at their existing tolerances.
- **Cargo feature `mcpower` renamed to `loop_advanced`.** Capability-named rather
  than consumer-named; still off by default, still the unstable scratch-explicit
  loop-tier surface with no semver guarantees. The `cluster_theta_truth` re-export is
  removed (truth-start magnitudes no longer live in `ModelSpec`).

#### Added

- **`StartValues { beta, theta }`** — the warm-start primitive (api.md Layer B): raw
  optimizer state (`beta` = fixed-effect start, `theta` = RE Cholesky parameters),
  not high-level variances. Exported `pub` only behind the `loop_advanced` feature;
  the stable tier never takes it. Carries no `phi`/`nb_theta`: Gamma φ is profiled and
  the GLMM neg-binomial θ search is a global bracket, so neither warm-starts anything
  reachable through the loop surface.

#### Migration — MCPower pin-bump action

MCPower consumes a pinned published `glmm`, so this rename is not a live break. When
MCPower next bumps its pinned `glmm`:

- switch its feature selection `mcpower` → `loop_advanced`;
- build any spec-derived start as a raw `StartValues.theta` (column-major vech of the
  RE Cholesky parameters) instead of relying on the removed `cluster_theta_truth` /
  `ModelSpec` magnitude fields.
