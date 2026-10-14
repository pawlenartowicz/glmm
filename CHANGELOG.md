# Changelog

All notable changes to the `glmm` crate are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

The Python package (`glmm` on PyPI) is versioned in lockstep with the crate and
shares these entries; Python-specific notes are called out where they differ.

## [Unreleased]

### Added

- **The `commonstats` dependency** (`0.2`, floor 0.2.2, `default-features =
  false`, features `dist` and `rng`; it brings `rand_philox` 0.2 with it):
  statistical distributions and the counter-based RNG the inference tests
  use. The t and F tests and `confint` below take their tails and quantiles
  from it; no estimate depends on it.
- **Satterthwaite degrees of freedom for the fixed effects of a Gaussian LMM,
  as lmerTest computes them.** One derivative pass at θ̂ after the fit, with
  no refit. lmerTest's `2v²/(gᵀAg)` over `(θ, σ)` is computed exactly from the
  profiled deviance the search minimized, as `v²/(g_pᵀH_p⁺g_p + v²/ν)`, `ν`
  the divisor of σ̂² (`N − p` on REML). The Hessian `H_p` comes from central
  differences of the deviance on every kernel, or of the analytic θ-gradient
  where the per-group route is armed (the fits the θ search runs BFGS on),
  which falls back to the deviance route when a gradient is unavailable. The
  coordinates of a Λ column that is exactly zero (a pinned boundary) are
  dropped, and the rest of `H_p` is inverted over its eigenvalues above
  `1e-8` of the largest. The method is in `documentation/algorithms-lmm.md`,
  "Satterthwaite degrees of freedom".
  - `Fit::coef_df` (length p): the df of each target coefficient. NaN on
    non-target and aliased columns, when the pass was switched off or
    failed, on a non-converged OLS or GLM fit, and on the families without a
    df (binomial, Poisson, negative binomial, every GLMM, a Gamma or
    inverse-Gaussian GLM with φ held). Not `Fit::df`, which stays the AIC
    parameter count.
  - `Fit::contrast_t(l)` and `Fit::contrast_f(l, rows)`: lmerTest's
    `contest1D` and `contestMD` (with `get_Fstat_ddf`) for any contrast, `l`
    over all p caller columns, row-major for `L`. They return
    `inference::contrast::{ContrastT, ContrastF}`, with
    `ContrastNote::AliasedColumn` (NaN fields) and
    `ContrastNote::RankDeficientL`, or `ContrastError::NoDf(NoDfReason)`
    (`NotGaussianMixed`, `Disabled`, `StateMissing`, `Failed`), never a z
    test. The `inference` module is outside the semver promise until 1.0.
  - `FitOptions::satterthwaite`, default `true`: `false` skips the pass.
  - `loop_advanced::satterthwaite(&mut ws, &mut df_out)` (no semver
    guarantee): the same df for the last `fit_on` draw, on demand; the loop
    tier runs no pass otherwise. Its buffers live on the workspace, so a call
    after the first allocates only the pseudo-inverse's blocks.
  - `Note::SatterthwaiteCurvature { dropped, negative, converged, nonfinite }`
    (kind `satterthwaite_curvature`, Python `SatterthwaiteCurvatureWarning`, R
    `fastglmm_satterthwaite_curvature`, tier caution): raised when the
    pseudo-inverse left out flat or negative directions, when the df were
    computed at an endpoint that did not converge, and when the pass failed
    (a non-finite deviance or covariance on its stencil, or no finite
    endpoint), which leaves every df NaN.
  - OLS and the Gamma and inverse-Gaussian GLMs with φ estimated carry the
    residual df `n − p` in `coef_df` whatever `satterthwaite` says, and their
    contrasts are the classical t and F.
  - Checked against frozen lmerTest 3.2-1 references
    (`validation/goldens/inference/lmertest/`), and on the accuracy grid by a
    new `lmertest` engine, whose per-coefficient df `compare.R` checks on the
    130 Gaussian mixed cells and their 129 ML twins.
- **The ports carry the df.** `orchestrate::FitResult` gains `coef_df`,
  `coef_p` (the kernel's t-test p-value, NaN where there is no df),
  `df_method` (`"satterthwaite"`, `"residual"`, `"none"`, `"disabled"` or
  `"failed"`), and `rss`, `r_squared` and `adj_r_squared` (OLS only);
  `orchestrate::run_fit` takes `satterthwaite: bool` after `warm_start`; and
  `orchestrate::confint` computes both ports' intervals. Python:
  `glmm.fit(..., satterthwaite=True)`, `Fit.coef_df`, `Fit.coef_p`,
  `Fit.df_method`, `Fit.rss`, `Fit.r_squared`, `Fit.adj_r_squared`, and
  `Fit.confint(level=0.95)`, a `(p, 2)` array of bounds in the order of
  `Fit.names`, with no `parm` argument. R: `fastglmm(..., satterthwaite =
  TRUE)` and `$coef_df`, `$coef_p`, `$df_method`, `$rss`, `$r_squared`,
  `$adj_r_squared`.
- **Gaussian mixed models fit by maximum likelihood on request: a `reml`
  argument, REML by default.** `FitOptions::reml: Option<bool>`, default
  `None`: `None` and `Some(true)` fit a Gaussian mixed model by REML, bit for
  bit as before, and `Some(false)` fits it by ML, on every kernel (dense,
  blocked, sparse) and every route (the balanced collapse, BOBYQA, and BFGS on
  the per-group gradient, whose ML form drops the `log|X'V⁻¹X|` term). The
  value is frozen on the workspace at `build_workspace`, and `Fit::reml`
  reports the criterion the kernel minimized. On an ML fit `Fit::deviance`,
  `Fit::loglik` and `Fit::dispersion` are the ML ones (`σ̂² = r²/N`, no
  standardization shift), and the Satterthwaite pass divides by `ν = N`, as
  lmerTest does on a `REML = FALSE` fit. Python: `glmm.fit(..., reml=None)`;
  R: `fastglmm(..., REML = NULL)`. The method is in
  `documentation/algorithms-lmm.md`, "Profiled REML and ML objectives".
  - The summary of an ML fit is titled "Linear mixed model fit by maximum
    likelihood [glmm]" (R `[fastglmm]`), prints the AIC, BIC, logLik,
    deviance and df.resid row, and prints "Note: p-values can be somewhat
    anticonservative." directly under the title, in Python's text, HTML (a
    callout), LaTeX and Typst (a boxed paragraph) renders and in R's
    `print()` and `summary()`. The note cannot be switched off; REML, OLS,
    GLM and GLMM fits print none.
  - Checked against frozen lmerTest 3.2-1 `REML = FALSE` references (26
    models and 6 per-term `anova` twins, `validation/goldens/inference/lmertest/`):
    the ML deviance at glmm's own θ̂ is within 1.5e-9 (relative) of lme4's and
    the Satterthwaite df within 3.5e-4 end to end, every row inside the REML
    bands. The accuracy grid gains ML twins of 129 of its 130 Gaussian mixed
    cells (tag `ml`), fitted by lme4, glmmTMB, MixedModels.jl and lmerTest with
    their own ML switch; `validation/grid/run.sh --ml` adds them to a `--fast`
    or `--timed` run.
- **`reml = Some(true)` on OLS reports the REML log-likelihood**, as R's
  `logLik(lm, REML = TRUE)`, and sets `Fit::reml`; β̂, the SEs and σ̂² do not
  move. It is on the scale of an LMM pinned at zero variance (5.5e-16
  relative on Dyestuff2). A formula whose every grouping was dropped as
  single-level is OLS here too.
- **The ports warn when `reml` is ignored.** `reml = Some(true)` on a GLM or
  GLMM is the ML fit, as `None` is: the kernel ignores it without a note,
  and `orchestrate::run_fit`, which gains a trailing `reml: Option<bool>`,
  strips it and sets the new `FitResult::reml_warning`. The ports raise it as
  an `argument_ignored` note, "reml=True applies only to Gaussian models; it
  was ignored and the model was fitted by maximum likelihood." (R spelling
  `REML = TRUE`; Python `ArgumentIgnoredWarning`, R class
  `fastglmm_argument_ignored`). `reml = Some(false)` there changes nothing
  and raises nothing.
- **R `AIC.fastglmm` and `BIC.fastglmm`.** On one fit, and on ML fits, the
  `stats` values and tables. A REML fit beside an ML fit (another package's
  fit is read as lme4's `isREML()`, an `lm` or `glm` as ML, anything else by
  its `logLik()`'s `REML` attribute) and fastglmm REML fits with different
  fixed effects stop with an error naming the fits. The fixed effects are
  compared as the sets of fixed terms, intercept included, of the stored
  formulas, by the new `orchestrate::same_fixed_part`. Both errors point to
  `compare()`. `?AIC.fastglmm` lists the four uses the methods cannot check.
- **Python `Fit.aic` and `Fit.bic`**: properties, `-2·loglik + 2·df` and
  `-2·loglik + df·ln(nobs)`, on every fit; the summary's ML criterion row
  prints the same numbers. On a REML fit they are built on the REML
  criterion and compare only fits with the same fixed effects.
- **`formula::Lowered::terms` and `Lowered::has_intercept`**: the fixed terms
  in R's order as `formula::FixedTerm { label, vars, numeric, columns }` (R's
  `assign`, as column ranges, the intercept excluded), and whether the formula
  keeps the intercept.
- **Hypothesis tests in both ports: `anova` and `compare`.** The crate computes
  every statistic and p-value; the ports lay out the same tables, with R's
  column names. The names of the calls and result classes are provisional
  until 1.0. The rules are in `documentation/hypothesis-tests.md`.
  - **Per-term tests of one fit**, R `anova(m, type = c("III", "II"))`,
    Python `glmm.anova(m, type="III")`: one row per fixed term, in R's term
    order, with no random-effect rows. lmerTest's F on Satterthwaite df on a
    Gaussian mixed model (REML or ML), F on `n − p` on OLS and the Gamma and
    inverse-Gaussian GLMs with φ estimated, Wald χ² everywhere else. Type III
    is lmerTest's construction on every family, so it does not depend on the
    treatment coding. Type II uses lmerTest's `XᵀX` metric on Gaussian mixed
    models and car's covariance metric on OLS, GLMs, GLMMs and the dispersion
    GLMs (on OLS without weights the two agree). Both tables are built after
    every fit by
    `orchestrate::run_fit` and stored on the fit (`FitResult::term_table_type3`
    and `term_table_type2`; Python `Fit.term_table_type3` / `type2`, R
    `$term_table_type3` / `$term_table_type2`), so the call computes nothing.
    A formula whose interaction lacks a marginal term is fitted as before
    but gets no table: `anova(m)` stops with the reason. A Type III table on
    a design with an empty cell carries lmerTest's "Missing cells" note.
  - **Two existing fits compared**, R `anova(m0, m1)`, Python
    `glmm.anova(m0, m1)`, in either order, with no refit and without reading
    the data. The fits must have converged, share the row count, the
    response (bitwise), the weights, the offset, the family and link and the
    held dispersion, be nested by their formulas, and differ in parameter
    count. Two OLS fits get `anova.lm`'s F, two Gamma or inverse-Gaussian GLMs
    with φ estimated the Wald F on `(r, n − p)`, every other pair the
    likelihood-ratio test `max(0, 2(ℓ₁ − ℓ₀))` (with lme4's columns, and AIC
    and BIC on ML fits). Two REML fits with different fixed parts, and a REML
    fit beside an ML fit (not two OLS fits), stop with an error pointing to
    `compare`.
  - **Two formulas compared**, R `compare(data, f0, f1, ...)`, Python
    `glmm.compare(data, f0, f1, ...)`: fits both models on the same rows
    (a row with a missing value in either formula, the weights or the offset
    is dropped from both) with `family`, `weights`, `offset` and `nAGQ` given
    once, and picks the estimator: REML when only the random part of a
    Gaussian model differs (the OLS side made with `reml = TRUE`), ML when
    the fixed part differs. It returns the table and both fits (R class
    `fastglmm_comparison` with `$table`, `$fit0`, `$fit1`; Python
    `Comparison` with `table`, `fit0`, `fit1`).
  - **The boundary rule for variance components.** Random parts that differ
    by one dimension get the 50:50 mixture of χ²(d − 1) and χ²(d), labelled
    conservative, or, for a Gaussian REML fit whose only random term is one
    unweighted `(1 | g)` with at least one fixed-effect column against its
    OLS reduction, the exact restricted likelihood-ratio test of
    Crainiceanu & Ruppert (2004): its null is simulated from the design
    summary the fit keeps, with no refit. Random parts more than one
    dimension apart get χ²(Δnpar), labelled as ignoring the boundary. When
    the random parts alone differ, a component the larger fit puts at exactly
    zero gives `T = 0` and p = 1 on every reference, and a non-finite
    log-likelihood gives NaN, never p = 1.
    `nsim` (default 10000, at most 1,000,000) and `seed` (a fixed default,
    so one call gives the same p-value on every run, in both ports and
    whatever R's `set.seed()`; R takes 0 to 2^53, Python 0 to 2^64 − 1) are
    arguments of both comparison calls. The exact test is checked against
    frozen RLRsim 3.1-9 references (`validation/goldens/inference/rlrt/`).
  - Python: `glmm.anova`, `glmm.compare` and the classes `TermTable`,
    `ComparisonTable` and `Comparison`; both tables are indexed by column
    name, print as text, render to `html()`, `latex()` and `typst()` (with a
    `Reference` column) and convert with `to_pandas()` (pandas imported only
    there). R: `anova.fastglmm`, returning a `c("anova", "data.frame")`
    table with a `heading` attribute, and `compare`, returning a
    `fastglmm_comparison` whose `$table` is one.
  - Every fit now stores the offset it was fitted with (`FitResult::offset`,
    Python `Fit.offset`, R `$offset`), its objective (Python `Fit.objective`
    and `Fit.objective_nodes`, R `$objective` and `$objective_nodes`) and,
    inside the exact test's envelope, its design summary
    (`FitResult::design_summary`, Python `Fit.design_summary`, R
    `$design_summary`).
  - Checked against frozen lme4 2.0-6, lmerTest 3.2-1, car 3.1-5 and glmmTMB
    1.1-14 references (`validation/goldens/inference/anova/`), and the two ports'
    tables against the Rust ones in CI.
- **The Rust comparison surface** (outside the semver promise until 1.0):
  `formula::anova::{nesting, compare, term_tests}` with `Model` (a fitted
  model as plain numbers, `Model::from_fit`), `CompareOptions`, `Comparison`,
  `ComparisonTest::{Lr, Linear}`, `ModelInfo`, `CompareError`, `TermTable`,
  `FixedRow`, `TermTestOptions`, `SsType`, `TermTestError`, `Side`, `Differs`
  and `Nesting`; `inference::{LrTest, LrNote, LinearTest, LinearKind,
  LinearNote, Reference, DenDf, Objective}`; and, under `orchestrate`,
  `run_compare`, `CompareResult`, `compare_options`, `DEFAULT_SEED`,
  `DEFAULT_NSIM` and `MAX_NSIM`. `TermTable` and `ModelInfo` carry the
  fit's `coef_names`, one per fixed-effect column, and a note on an aliased
  column names the coefficient, never an index: "The coefficient x2 is
  aliased, so the hypothesis was not tested." (the ports print it after the
  row label, "x2: The coefficient x2 is aliased, ...").
- **`Fit::nobs` and `Fit::objective`**: the row count the fit was given (on
  every route, a non-converged fit included) and what it maximised
  (`Objective::{LeastSquares, MaximumLikelihood, Reml, Laplace, Agq { nodes }}`).
  Both are new fields of the `#[non_exhaustive]` `Fit`; no fitted number
  moves.
- **`loop_advanced` wrappers for build-once callers** (no semver promise,
  `#[doc(hidden)]`): `lr_p_value_pub` (the p-value of a likelihood-ratio
  statistic on a `Reference`), `wald_chisq_pub` (the Wald χ² of an `L` set on
  β̂ and its covariance) and `type3_pub` (the Type III hypothesis matrices of
  a design, with `TermHypothesis`), so a power simulation builds the
  hypotheses once per design and pays only the Wald form per draw.

### Changed

- **R's `REML` is a formal argument of `fastglmm()`, and its refusals are
  gone.** It is no longer intercepted in `...`. `REML = FALSE` on a Gaussian
  mixed model fits by ML instead of stopping; `REML = TRUE` on a GLM or GLMM
  warns and fits by ML instead of stopping; `REML = TRUE` on a Gaussian
  formula whose every grouping was dropped reports the OLS REML
  log-likelihood instead of stopping after the fit. A value that is not
  `NULL` or one `TRUE` or `FALSE` still errors.
- **R's `AIC()` and `BIC()` on REML fits with different fixed effects, or on a
  REML fit beside an ML one, now stop with an error**, where 0.5.0 returned
  the `stats` table.
- **Coefficient tables test with t where lmerTest, `lm` and `glm` do.** A
  Gaussian mixed model prints `Estimate`, `Std. Error`, `df`, `t value` and
  `Pr(>|t|)`, with the footer "t tests use Satterthwaite degrees of freedom
  (finite-difference θ-Hessian).". OLS prints `t value` and `Pr(>|t|)` on
  `n − p`, with `summary.lm`'s "Residual standard error: … on n − p degrees
  of freedom" and "Multiple R-squared: …, Adjusted R-squared: …" lines. A Gamma or
  inverse-Gaussian GLM with φ estimated prints t on `n − p`, the df on its
  dispersion line. Binomial, Poisson, negative binomial, a held dispersion
  and every GLMM keep Wald z. A Gaussian mixed model fitted with
  `satterthwaite = FALSE` (Python `satterthwaite=False`) prints z and "df not
  computed", and one whose pass failed prints z with the warning. The footnote
  "z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or residual
  df is reported" is gone. Both ports, all four Python renders.
- **R's `confint()` uses the t quantile where `summary()` tests with t**: each
  row's Satterthwaite df on a Gaussian mixed model, `n − p` on OLS and the
  dispersion GLMs (as `confint.lm`), and the normal quantile on the z rows. So
  those intervals are wider than 0.5.0's normal ones. With `satterthwaite =
  FALSE` the intervals stay normal and a message says the df were not
  computed. Python's new `Fit.confint()` gives the same intervals.
- **R's `tidy()` gains a `df` column**, between `statistic` and `p.value`,
  where `summary()` has one; its statistic and p-value are the summary's, so
  they move to t with it.
- **Binomial and Poisson GLMMs with a random intercept only and small groups
  search with BFGS under adaptive quadrature too: about a fifth less time per
  fit (0.783 of it cold, 0.825 warm, geometric means).** An AGQ fit
  (`nAGQ > 1`) of a binomial GLMM on the logit or probit link, or of a Poisson
  GLMM, with one scalar random intercept, no other grouping and fewer than 50
  rows per group on average now searches `[θ | β]` with projected BFGS on the
  exact gradient of the AGQ deviance, instead of the joint BOBYQA search. The
  gradient is assembled from the modes each value evaluation leaves and costs
  about half a value evaluation. As on the Laplace route, a search that does
  not certify its endpoint is polished by BOBYQA, and one that fails hands
  over to the joint BOBYQA search (`Note::GradientSearchFallback`). The joint
  search's check for working weights that span decades, with its second pass,
  also runs after the gradient search. These fits keep the joint BOBYQA
  search: a warm start with a variance below 0.05 (it ends at or near zero,
  where the endpoint is always polished, and was up to 1.57x slower), a fit at
  50 rows per group or more (a 500-row cell was 1.27x slower), and cloglog,
  NB, Gamma and random-effect vectors under AGQ.
  - Measured as whole `fit_cold` / `fit_warm` calls on a clock-locked x86
    core, against 0.5.0, on the covered AGQ cells of the accuracy grid and the
    stress set below 50 rows per group: 25 cold rows and 38 warm ones (from
    the truth and from 20 % off). No row is slower by more than 1.25x; the
    slowest are the probit `prba_int1_g3000p20_bal_base_agq7`, 1.10x cold and
    1.23x warm from the truth. Cold, the grid's 5-row cells and cbpp take
    0.58x to 0.73x and the stress set's 5-row Poisson cells 0.51x and 0.52x.
  - The search path changes, so results on these fits move by round-off of
    the optimum: on the accuracy grid's 15 covered AGQ cells the deviance
    moves by −1.3e-10 to +4.9e-11, with no `converged` or `singular` flag
    change, and `n_eval` (value evaluations only) is 2 to 6 times lower. The nAGQ = 7
    binomial and Poisson R acceptance pins were re-recorded; their
    log-likelihoods moved by 1.9e-11 and 3.2e-11.
  - `Diagnostics::n_grad`, `outer_stop` and `outer_decrement` (Python
    `Fit.n_grad` / `outer_stop` / `outer_decrement`, R `$n_grad` /
    `$outer_stop` / `$outer_decrement`) are now filled on these fits, and
    `loop_advanced::curvature_hint` returns a hint after one.
  - After an AGQ fit hands over to the joint search, when that search's check
    starts its second pass, one more pass runs from the check's re-solved point and the lower
    of the two is kept. Where the working weights span many decades the
    deviance's rounding noise stops BOBYQA where its start leads it: on 125
    such designs the fits that ended away from the optimum fell from 31 to 16
    on x86_64 and from 30 to 17 on aarch64, none ended worse, and a hand-over
    fit takes about 29 more evaluations. Fits without a hand-over are
    unchanged by it.
  - On a Poisson AGQ fit whose working weights span about twelve decades, the
    gradient search's line search fails, so the fit now raises
    `Note::GradientSearchFallback` (Python `GradientSearchFallbackWarning`, R
    `fastglmm_gradient_search_fallback`) where 0.5.0 raised nothing. The fit
    then ends 4.6e-5 deviance units (3e-15 relative) from 0.5.0's optimum.
- **`FitOptions` and `formula::Lowered` are `#[non_exhaustive]`.** Build
  `FitOptions` from `FitOptions::default()` and assign fields; struct
  literals and `..` update syntax no longer compile outside the crate. No
  result changes.
- **The dev-only `oracle-tests` feature turns on `orchestrate`** (and so
  `formula`): the inference tier gates the ports' interval function, which
  lives there.

### Removed

- **Python's `summary()` no longer prints the `se` column of the
  random-effects block**, in the text output or the HTML, LaTeX and Typst
  renders, so both ports print the same block. `Fit.stddev_se` keeps the
  numbers.

## [0.5.0] — 2026-10-10

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

- **A Gamma GLMM with the Laplace approximation now profiles the fixed effects
  out exactly**, as binomial, Poisson and negative-binomial GLMMs already did:
  the outer search runs over the variance parameters and `ln φ` only. On the
  accuracy grid's Gamma cells the deviance moves by at most 2.6e-7 and the
  search takes about a third fewer evaluations. On a Gamma-inverse design whose
  cluster means span several decades, the joint search over the fixed effects
  reported `converged` 21 to 30 deviance units above the optimum; the profiled
  search reaches the optimum on the milder of these designs and reports
  `converged = false` on the others. AGQ and packed-layout Gamma fits keep the
  joint search.
- **`stddev_se` on a random-slope block (q ≥ 2) is the SE of the reported
  stddev `√Σ_kk`**, at each stddev's diagonal slot of the θ layout, where it
  was the SE of that Cholesky entry; the off-diagonal slots, and a stddev of
  exactly 0, are `NaN`. Scalar groupings and the intercept of a block with no
  centered slope are unchanged.
- **Every fit standardizes its design internally, so a predictor's units and
  location no longer move the answer.** At the start of each fit, every column of
  the fixed-effect design is centered and scaled, `x̃ = (x − m)/s`, with the
  prior-weighted mean and standard deviation of that draw's data. The intercept
  and every 0/1 column (factor dummies, binary covariates) are left as they
  are, and without an intercept nothing is centered, only scaled. A 0/1 column
  used as a random slope is scaled by its prior-weighted root mean square but
  not centered, so its zeros stay exact zeros. A random
  slope reads the same transformed column, and its block's own random
  intercept absorbs the shift. The response, the offset and the weights are
  not transformed. Every result is mapped back exactly: β = Bβ̃ and its SEs,
  `vcov`, t² and the joint Wald statistic through the rows of B; `varcorr` as
  N·Σ̃·Nᵀ, `ranef` as N·b̃ and θ (`tau2`, the loop tier's `FitView::theta`) as
  the lower Cholesky factor of N·Λ̃Λ̃ᵀ·Nᵀ; `stddev_se` by the delta method on
  θ̃; the REML deviance and
  log-likelihood add back 2·Σ ln sⱼ. Measured on the fits that motivated it:
  `sim_gamma` at x·1e6 reports se(β₁) 3.7e-11 relative off where it was 7e-4;
  the Gamma random-slope fit on x + 1000 takes 122 evaluations, as on x,
  where it took 3300; sparse Gamma with x1 + 1000 converges in 731
  evaluations to within 7.1e-9 of the unshifted log-likelihood, where it ran
  out of its 39000-evaluation budget 3.67e-4 short; and over 399 seeds of the
  pinned-diagonal vector-AGQ fixture no seed changes boundary basin between x
  and x + 1000 (largest deviance difference 3.1e-10), where 12 did when the β
  coordinates changed, by up to 0.298.

  Behaviour that changes with it. Vector AGQ (`nagq > 1` on a block with a
  slope) takes its product Gauss–Hermite grid in the standardized coordinates: the
  same model, a different quadrature approximation of it, so its deviance
  differs from an uncentered fit's by the quadrature error. `singular`
  compares the fit's internal standard deviations, on the standardized
  design: on a predictor whose mean is far from zero the intercept's is taken
  at the mean, and a slope variance that was hidden by the predictor's distance
  from zero is now flagged. `pinned` is reported in the caller's coordinates,
  and under a centered slope a zero can move from the intercept diagonal to
  the slope diagonal. `ill_conditioned` is measured on the standardized design,
  so a column that is ill-conditioned only through its distance from zero is no
  longer flagged. On the loop tier, a warm-start β or θ of the wrong width is
  dropped, a cold start for that component, on every workspace: it can no
  longer be mapped, and a short θ used to be zipped against the previous
  fit's θ̂ left in the workspace.

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

  **Semver note.** The formula frontend is outside the semver-covered
  surface (`fit_cold`/`fit_warm`, `ModelSpec`, `GroupIds`), so semver alone
  would allow this in a patch release, but callers see it:
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
- **A sparse-factored packed GLMM no longer holds a dense `k×k` matrix.** The
  dense `A` and its Cholesky target are allocated only once the packed layout
  settles on the dense factor, and the Rx Schur fill solves through the sparse
  factor the converged PIRLS left instead of expanding `A` into a dense buffer
  and factoring it again. On a binomial design with two crossed intercepts and
  k = 2,000 (k = 5,000) random-effect columns the fit's peak RSS falls from
  94 MB to 29 MB (485 MB to 84 MB) with Hessian standard errors and from
  128 MB to 30 MB (690 MB to 85 MB) with Rx standard errors. The saving is
  about 16k² bytes, not 8k²: `Mat::zeros` wrote the pages of the never-read
  Cholesky target as well. Deviances are unchanged.
- **A negative-binomial GLMM whose no-RE prefit failed seeds θ_NB at 1.**
- **Python and R print Gamma's dispersion as `Dispersion (phi, Pearson)` on
  a GLM and `Dispersion (phi, ML)` on a GLMM,** and `Dispersion (phi, fixed)`
  when `dispersion=`/`dispersion` holds φ, on any family that takes one.
- **Every warning text was rewritten** in plain words, as
  `<Tier>: <title>. <message>`; none advises rescaling predictors any more,
  since the fit no longer depends on their scale. Code that matches on message text (for example
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
- **The per-draw column statistics and design fill take fewer instructions**, with
  every result bit-identical. The 0/1, all-ones and constant tests of each column
  stop at the first row that fails them, the 0/1 test is one comparison without a
  branch, and the two summation passes keep every column's row order. On the loop
  tier (`fit_on`, warm start, instructions per fit on x86, Mac M4 in brackets):
  OLS at n = 5000 −28% (−37%), a logit GLM at n = 5000 −15% (−19%), a
  random-intercept LMM at n = 4000 −11% (−10%), random-slope and crossed LMMs
  −5% and −7% (−5%).
- **Each outer search starts at a radius chosen for its kind of start**, where
  every search started at 0.1: the LMM from the blind start at 0.5, GLMM stage 1
  and the `Joint` route at 0.2, the joint polish after stage 1 at 0.03 (0.1 when
  stage 1 ends with a variance component near zero), and a θ-only search from a
  caller's start at 0.1 as before. Fits reach the same optimum in fewer
  evaluations: on x86, 0.81 of them in total on 244 cold fits of the accuracy
  grid (LMM cells 0.86, GLMM families 0.46 to 0.93) and 0.85 on its fast
  subset, with no convergence status changed. Results move within the
  optimizer's stopping tolerance, and the pinned tests that moved are re-pinned.
  The radius of the non-finite-evaluation convergence check stays 0.1.
- **The negative-binomial GLMM route reuses its workspace on the loop tier,
  and a serial blocked or structured GLMM fit no longer starts a rayon thread
  pool.** `fit_on` on an NB GLMM keeps one kernel workspace across draws, as
  the other GLMM families do; serial vector AGQ builds its per-cluster row
  index once per fit; the Cholesky factorizations of the GLMM post-fit step on
  the blocked and structured layouts, the GLM IRLS and OLS β solves and the
  LMM joint Wald test run serially with stack scratch, where they read faer's
  global parallelism setting and so started rayon's pool in every process.
  Every result is bit-identical.
- **The GLMM PIRLS solve takes about 17 to 46% fewer instructions per fit
  (x86) on binomial, Poisson and negative-binomial random-intercept and
  random-slope fits that take the exact β-profile route**, with every result
  bit-identical. Gamma fits and adaptive-quadrature fits do not take this
  route and do not change. The β-profiled solve reads a row-major copy of the
  design and forms the random-effect blocks, the β gradient and the border's
  `M'WX` in one pass over the rows, with the per-row work unrolled for up to 8
  fixed effects. Measured on the loop tier (x86, instructions per warm fit):
  logistic random intercept −46%, Poisson −39%, random slope −27%, negative
  binomial −17%, nested −13%.
- **The GLMM PIRLS border forms `X'WX` in its row pass instead of a matrix
  product**, a further 5.6 points of instructions on logistic
  random-intercept fits and 3.8 on random-slope fits (2.7 on Poisson; x86).
  The products are associated and summed in another order, so results move
  by round-off (objectives by at most 5.0e-10 on the loop-tier shapes,
  standard errors by 1.2e-6 relative).
- **The exact Hessian of a crossed GLMM forms its crossed-effect terms once
  per cluster instead of once per row**, through the per-cluster products
  the PIRLS solve already uses. On a 50 × 40 crossed logistic fit a warm fit
  takes about 36% fewer instructions (x86). Standard errors move by round-off
  (4.4e-16 relative).
- **The negative-binomial GLMM's `ln θ_NB` row of the Hessian takes its mixed
  entries from the exact gradient**: a central difference in `ln θ_NB` of the
  Laplace gradient replaces the four-point difference of the objective, so the
  row costs two mode solves and two gradient passes instead of `4m` solves
  (about 24% of a warm NB GLMM fit's instructions, x86), and its round-off
  error is divided by the step rather than by its square. The diagonal entry
  and the fallback where the exact gradient is not available (AGQ and the
  packed layout) are unchanged. Standard errors move by at most 2.1e-7
  relative on the loop-tier shapes (x86).
- **On log links the PIRLS score and curvature read `dμ/dη = μ` from the
  fitted mean** instead of a second exponential per row (Poisson, negative
  binomial, Gamma and inverse-Gaussian with a log link): Poisson GLMMs take
  about 13% fewer instructions per fit, Gamma about 14% (x86). The mean comes
  from the vectorized exponential, so results move by round-off (objectives by
  at most 2.8e-8 on the loop-tier shapes, a Gamma fit, where the search stops
  elsewhere inside its tolerance).
- **The negative-binomial GLMM evaluates its saturated log-likelihood once per
  distinct count**, carrying the finite sum `Σ ln(θ+k)` up the sorted counts,
  instead of once per row: each count's term is unchanged to the last bit and
  only the sum over rows is regrouped, so objectives move by round-off.
- **The negative-binomial deviance takes its logarithms from a vectorized
  `ln`** (within 2 ULP of the system `ln`, on the normal range), three
  per row, so an NB GLMM fit takes about 15% fewer instructions (x86).
  Results move by round-off. A NaN mean still gives a NaN deviance.
- **A dense LMM fit spends far fewer instructions orthogonalizing and
  accumulating its design**, with every result bit-identical. The Givens pass
  and the sufficient-statistics accumulation read the rows through a small
  row-major tile, keep the hot state on the stack, and have fixed-width
  instances for the common design widths and a unit-weight instance; every
  rotation and every sum keeps its order. On the loop tier at n = 4000 (x86,
  instructions per redraw fit): a random-intercept LMM about −55% (0.448×),
  random-slope LMMs about −30% (0.695×), crossed LMMs about −33% (0.672×); at
  the small sizes 0.69× to 0.93×. On a Mac M4: a random-intercept LMM about
  −60 to −65%, random-slope and crossed LMMs about −35%.
- **A dense LMM fit on a well-conditioned design reads its fixed-effect factor
  off the Gram it accumulates anyway**, instead of a separate Givens pass over
  the rows. In exact arithmetic the two give the same factor; results move by
  round-off (at most 6.0e-10 in deviance on the accuracy grid's fast cells, no
  flag changes). A design with a pivot ratio under `GRAM_PIVOT_MIN` = 1e-3
  keeps the Givens pass and its results bit for bit. Loop tier at n = 4000
  (x86, instructions per redraw fit): about −27% more on a random-intercept
  LMM, −8% on a random-slope LMM, −10% on a crossed LMM. On a Mac M4: about
  −25% on a random-intercept LMM, −6% on random-slope and crossed LMMs.
- **A warm GLMM fit continues from the start guard's own solve.** The guard
  solves the caller's θ₀ and the blind start each the way a fit's first
  evaluation is solved, and the search now takes the winning solve, with the
  curvature state that solve left, as its first evaluation instead of solving
  the point again. A warm fit with a stage 1 does one Laplace evaluation less
  (x86: 3 to 9% of the instructions of a warm loop-tier fit on the exact-route
  shapes), and a warm fit the guard sends to the blind start now runs the
  cold fit's own search. Warm results move by round-off; on the
  far-warm-start scan two of 1200 far starts end at another converged point,
  documented next to the scan (`src/fit/glmm_tests.rs`). Cold fits are
  unchanged.
- **A warm GLMM fit whose θ is the blind start skips the start guard.** The
  guard compares the caller's θ₀ with the blind θ₀ at the same β₀, so when the
  two are the same point there is nothing to compare; the fit now runs as a
  cold fit from the caller's β₀ (for example a random-intercept fit started at
  `[1; n_θ]`). The check reads θ on the fitted design, after random-slope
  columns are rescaled, so on a random-slope model a start of ones is in
  general not the blind start. x86: 2.5 to 8% fewer instructions on such
  loop-tier fits (logit RI 7 to 8%, probit 7%, Poisson 7 to 8%, crossed and
  nested 5%, NB 3%, Gamma 2.5%), the same answers bit for bit on every
  loop-tier draw. Starts at any other θ are unchanged.
- **The `ExactProfile` search solves its early evaluations to a looser PIRLS
  tolerance, tied to the optimizer's trust radius**: `clamp(1e-2·ρ², pirls_tol,
  1e-5)` while the radius ρ is large, full tolerance on the first evaluation and
  once ρ is within 10× of its end value, and the incumbent re-solved at full
  tolerance before it is reported. x86: 5 to 22% fewer instructions per
  loop-tier fit on the exact route from a cold start, 4 to 21% from a warm
  one, about the same number of outer evaluations (0.98 to 1.02 on the loop
  shapes, x86), results moved by round-off. Needs `bobyqa` 0.3.1
  (`Bobyqa::minimize_with_radius`).
- **One-dimensional BOBYQA searches use Powell's original RESCUE test**
  (`bobyqa` 0.3.2, `Config::prima_parity = false`), not PRIMA's, which at a
  fully determined model is decided by rounding alone. On the accuracy grid this
  moves only the cells with one variance parameter (65 of 794): about 15% fewer
  evaluations, deviance equal to round-off, no flag flipped. `bobyqa` 0.3.2 is
  now the floor.
- **Separation is reported the same way everywhere.** A GLM whose deviance
  stops changing while its last step still moves some row's linear predictor by
  more than 0.01 now reports `converged = false`: the data are separated, or a
  category has only zero counts. The coefficients of the last step are kept and
  the standard errors are NaN. A GLMM whose fixed effects alone separate the data
  now stops before its search with the same outcome, and the Python and R
  packages give it the `glm_diverged` message. Before, depending on `nagq`, such
  a GLMM could report `converged = true` with a huge coefficient, or use its
  whole evaluation budget.
- **Strong but ordinary binomial fits are no longer refused.** The check that
  refused a binomial GLM when more than half its rows had a Fisher weight below
  1e-5 is removed. It refused ordinary cloglog fits, and logit or probit fits
  with a strong predictor, that R fits without trouble.
- **An exact-profile GLMM search that ends next to a zero random-effect column
  checks for a saddle there.** The column is doubled while the deviance falls,
  and the search restarts from the better point. On the accuracy grid one cell
  moves (`pois_q2sx2_g3000p20_bal_boundary`, 4.0e-6 lower deviance) and 50
  near-singular cells take 1 to 4 more evaluations; no flags change.
- **A GLMM whose standard errors cannot be computed keeps its estimates.**
  When the information matrix of the fixed effects is not positive definite
  to rounding at the search's end point (the RX Schur
  complement fails, and under `WaldSe::Hessian` the joint Hessian before it),
  the fit used to be thrown away whole: β̂, θ̂ and the deviance were NaN and
  `converged` was false. Now only `se`, `vcov`, `stddev_se` and the Wald
  statistics are NaN; the estimates, the deviance and `converged` are the
  search's own, and the new `Note::SeUnavailable` says why (`kind`
  `se_unavailable`; Python `glmm.SeUnavailableWarning`, R class
  `fastglmm_se_unavailable`). `GlmmFit::se_unavailable` and
  `FitDiagnostics::se_unavailable` (`loop_advanced`) carry it.

- **AGQ fits are 2–3.6× faster on the log links and the logit.** Both AGQ
  kernels now sum each row's change in deviance from its cluster's mode in
  closed form, in SIMD tiles, instead of evaluating the unit deviance afresh
  at every node; with a scalar intercept, Poisson-log and Gamma-log clusters
  reduce to two sums per cluster. On the grid's `nAGQ = 7` cells a fit takes
  0.28–0.48 of its former time on x86 and 0.48–0.55 under WASM, with the same
  `n_eval`. The results move at round-off level. Probit has its own form (next
  entry); cloglog keeps the generic node loop.
- **AGQ probit fits with a scalar intercept take about three quarters of their
  former time on x86.** The scalar AGQ kernel now takes each row's `ln Φ` from
  one SIMD `erfc` and one SIMD `ln` and sums its change from the cluster's
  mode, instead of running the inverse link, `erfc` and μ-form deviance per row
  at every node. On the clock-locked machine below a fit takes 0.73–0.75 of its
  former time on the grid's `prbb` and `prba` `nAGQ = 7` cells and at 30k to
  300k rows (7.94 s to 5.85 s at 300k), and 0.82 on `cbpp_probit_agq7`; on an
  Apple M4, 0.87–0.96 (1.04 on cbpp's 56 rows, a 1.7 ms fit). `n_eval` is
  unchanged; the deviance moves by at most 3.6e-12 on the grid cells and 1e-9
  at 300k rows, with no `converged` or `singular` change, native and WASM.
  Vector random effects keep the generic node loop.
- **Probit GLMM fits take 0.64–0.86 of their former time on the blocked
  layout, and the binomial deviance takes a SIMD `ln`.** On a probit fit the
  PIRLS loops of every layout now take `φ(η) = dμ/dη` for the row score, the
  observed weight and the exact profile's `dW_obs/dη` from the SIMD `exp`, 64
  rows at a time, instead of one libm `exp` per row and per function. The
  family pass's binomial deviance takes its two logs through the SIMD `ln` a
  chunk of rows at a time, for probit, cloglog and weighted logit. On the
  clock-locked machine below a probit Laplace fit takes 0.64–0.79 of its former
  time on the grid's blocked cells (`int1`, `q2s`, `q3s`, `q4`; 0.86 on `q8`),
  0.82–0.84 on two-level nested ones, 0.93–0.98 on the deeper nested and crossed
  ones, and 0.69–0.74 at 30k to 300k rows; the probit GLM 0.81, AGQ probit 0.80–0.91.
  Cloglog fits take 0.93–0.96 (1.005 on `cllb_cross4`). On an Apple M4 probit
  fits take 0.82–0.97 (1.06 on cbpp's 56 rows, three more evaluations on a
  0.3 ms fit) and cloglog fits 0.96–1.12. The results move at round-off level:
  over the grid's 96 probit and cloglog cells the deviance moves by at most
  2.6e-7 (in both directions; 18 cells take more evaluations, 25 fewer), with
  no `converged` or `singular` change, serial and parallel identical. Logit
  without weights, the dual derivative kernels and the SE path keep their
  arithmetic.
- **A Gamma GLMM whose dispersion seed is already close searches `ln φ` on a
  matched scale.** Two probe solves at the start measure the curvature of
  `ln φ`; when its conditional optimum lies within one cold start radius of
  the seed, the search runs on `ln φ` divided by a fixed power of two, so the
  coordinate is about as stiff as θ. On the grid this holds on every
  Gamma-inverse cell and on no Gamma-log cell: on
  `gami_cross4_g3000p20_bal_base` the search takes 97 evaluations, was 365,
  and the fit 0.23 of its former time. Every other Gamma GLMM pays the two
  probes (a later call at a probed point reuses its solve) and searches
  unscaled. The search path changes, so the last digits of the estimates can
  move. The test is the cold fit's on a warm start too: the seed's distance is
  measured against the cold radius, and a carried dispersion (below), like the
  point where a gradient search hands over to BOBYQA, is judged by
  where the cold seed lies, not by its own distance. Measured against the
  warm radius, a warm start from the optimum perturbed by 5% took 1578
  evaluations on `gami_q4_g3000p20_bal_base` (cold 364); it takes 368. Over
  the grid's 35 Gamma cells warm starts without a dispersion take 1.04 of the
  cold fit's evaluations on Gamma-inverse (geometric mean, 18 starts; 1.62
  with the warm radius) and 0.98 on Gamma-log (87 starts; 0.97 with the warm
  radius). Cold fits are unchanged.
- **Crossed GLMMs on the structured layout are 21–24 % faster per
  evaluation**, with bit-identical results: the crossed tail's Schur downdate
  walks contiguous rows, and its `tail_h`/`tail_g` build reads `S⁻¹` by column
  when `S⁻¹` is exactly symmetric.
- **The exact-profile β border is about 1.6× faster per evaluation on wide
  random-effect blocks.** On the blocked layout the residual rows are built a
  tile at a time and summed into the border while in cache, and the block
  solve takes all fixed-effect columns at once. On the grid's `q8` cells
  (eight random effects per group) an evaluation takes 0.61 of its former
  time. The sums associate differently, so the results move at round-off
  level and `n_eval` follows a different search path, in either direction:
  over 73 wide cells the evaluation-count ratio is 1.008 (95 % CI
  0.992–1.024).

  Timings above are on a clock-locked Intel Core Ultra 7 265H; an Apple M4
  gives ratios within about 0.1 of them, and WASM fits stay bit-identical
  across hosts. Over the 794 cells of the accuracy grid the four changes
  together move the deviance by at most 4.6e-7 (worse by more than 1e-9 on 60
  cells, better on 77; 362 identical or at round-off), change no `converged`
  or `singular` flag, and take 11 % fewer evaluations in total.

- **An LMM with random slopes on its only grouping evaluates the REML
  criterion per group in closed form,** 2.5 to 2.7× faster per evaluation on
  the grid's 36-parameter cells. Two Woodbury steps (Henderson & Searle 1981)
  split each group's contribution into a part fixed for the whole fit and a
  small `q×q` solve per evaluation, read from the existing sufficient
  statistics. The route applies to any random-slopes block on the only
  grouping, weighted or not, when every level's random-effect Gram is well
  conditioned; otherwise the fit runs the former loop with identical results.
  A fit on `lmm_q8_g3000p20_bal_base` takes 0.48 of its former time and on
  `sim_max_q_slope` 0.63. Every grid cell that takes the route is faster
  except Machines, which takes 15 % more evaluations and 10 % more time. The
  route is also more accurate at large variance parameters: against an
  80-bit reference at `THETA_HI` its deviance is off by 3e-11, the former
  loop's by 1e-7. On the cells that take it the deviance moves by at most
  7.3e-9 (8e-13 relative), with no `converged`, `singular` or `pinned` flag
  change. Same clock-locked machine as above.

- **LMMs on that route search their variance parameters with a gradient
  method (BFGS): a fit takes about a quarter of its former time, and 0.03 to
  0.06 of it on 36-parameter designs.** The route's criterion has an
  analytic gradient, read off the state of the evaluation it follows (no
  second solve), so `fit_lmm` searches θ there with a projected BFGS
  instead of BOBYQA; every other LMM keeps BOBYQA. An endpoint is accepted
  when its predicted remaining decrease is below 1e-8 and no variance sits
  near zero; otherwise BOBYQA polishes it, and a search that fails (a failed
  line search, an unavailable gradient, the iteration cap) hands its best
  point to BOBYQA and raises `Note::GradientSearchFallback`. Measured on the
  grid's 23 route cells, whole `fit_on` calls (cold and warm from the truth,
  5% and 20% away), on one clock-locked core: cold fits take 0.23 of their
  former time (geometric mean; summed 0.11), warm fits 0.31 to 0.33; the
  36-parameter cells 0.03 to 0.06. The slowest case is
  `lmm_q2s_g3000p20_bal_boundary` from a warm start, 1.03 to 1.16×, where a
  variance at its boundary sends the endpoint to the BOBYQA polish. The
  deviance is worse by at most 9.6e-10 and better by up to 7.6e-9, BOBYQA
  started from every BFGS endpoint gains at most 9.1e-10, and no
  `converged`, `singular` or `pinned` flag changes; on the stress set's 20
  LMM cells every fit is equal to or below the best oracle. `n_eval` counts
  value evaluations only (a gradient reuses its evaluation), so it falls
  with the time.

- **Binomial and Poisson GLMMs search with BFGS: about half the time of a
  whole fit.** A Laplace fit (`nAGQ = 1`) of a binomial (any link) or Poisson
  GLMM with at least two variance parameters on the blocked or structured
  layout now searches `[θ | β]` with projected BFGS on the analytic gradient,
  instead of BOBYQA on the exact β profile. The gradient at each accepted
  point is assembled from that point's own mode solve, so a gradient costs
  about one more value evaluation, not a second mode solve. A search that
  does not certify its endpoint is polished by BOBYQA, and one that fails
  hands over to the BOBYQA route (`Note::GradientSearchFallback`). NB, AGQ,
  the packed layout, one-θ fits, and warm starts far out (a θ above 10, or no
  random slope at 50 rows per level or more) keep BOBYQA.
  - On the grid's fast subset, routed fits take 0.48 of their former time
    (geometric mean, whole fits with Hessian SEs, clock-locked x86), and none
    is slower. The crossed binomial `binb_cross4_g3000p20` goes from 286 to
    116 ms.
  - Over 263 routed grid cells the deviance moves by at most 5.0e-7 and is
    nowhere worse by more than 1e-6, with no `converged` or `singular` flag
    change. Evaluations fall to 0.33 (median).
  - A 3M-row binomial fit with four random effects per group takes 57 s
    instead of 216 s, at the same peak memory.
  - The search path changes, so results on routed fits move at the
    optimizer's tolerance. Two R acceptance pins were re-recorded at higher
    log-likelihood.
  - The search stops when one step lowers the deviance by less than 1e-10
    (LMMs use 1e-9). Where a variance goes to zero, the deviance can be almost
    flat along a curved valley, and the search goes down it in waves; a stop
    at 1e-9 ended such a Gamma-inverse fit 2.4e-5 above its optimum.
    Compared with a stop at 1e-9, this costs about +1–3% time on binomial
    (logit 1.020x, probit 1.030x, summed over the routed grid cells) and
    about +1% on Poisson (1.008x); cloglog is 1.4% faster (0.986x). The
    deviance is equal or better, within 1.1e-9.

- **Gamma GLMMs search with BFGS too: about a quarter of the time of a
  whole fit.** A Laplace fit of a Gamma GLMM (log or inverse link) with
  at least two variance parameters on the blocked or structured layout now
  takes the same BFGS search as binomial and Poisson. The dispersion `ln φ`
  is one more search coordinate, with an exact gradient in closed form. A
  held dispersion (`FitOptions::dispersion`) stays fixed, and the search then
  runs over `[θ | β]` only. On the inverse link the search starts with the
  known curvature of `ln φ`, so its first steps are not spent on `ln φ`
  alone.
  - Whole fits on a clock-locked x86 core, against the BOBYQA search: 0.29x
    of the time summed over the 88 routed Gamma/log grid cells, and 0.21x
    over the 14 Gamma/inverse cells. No cell is slower.
  - WASM, on the timed fast cells: 0.34x on Gamma/log and 0.47x on
    Gamma/inverse. `gaml_cross4_g30000p20` goes from 10.7 s to 3.7 s.
  - `n_eval` falls on these fits (a gradient reuses its value evaluation, so
    it is not counted there). Cold grid fits, median: 82 to 25 on Gamma/log,
    113 to 82 on Gamma/inverse.
  - From cold and warm starts the deviance is never worse than the BOBYQA
    search's by more than 9.4e-10, and no `converged` or `singular` flag
    changes.

- **Hessian standard errors on large GLMMs take a fraction of the memory
  and less time.** The dual-number scratch of the exact Hessian no longer
  holds buffers its shape never reads: the structured layout's row buffers
  exist only on that layout, the crossed values of `M` are kept once per
  grouping instead of once per row, the observed-information factors only on
  a non-canonical link, the crossed layout's coupling buffers hold the
  design's own coupling pattern instead of a bound six times the row count,
  and the assembled pass forms its per-row values in one walk over the rows
  instead of storing them for three. On a single grouping above 4,096 rows
  the assembled Hessian then runs one block of whole clusters at a time, so
  its scratch stays the same size whatever the number of rows; on crossed
  designs above 4,096 rows it forms its per-row values one block of rows at a
  time and keeps only the per-cluster and per-pair sums whole. It also picks
  its lane width from a measured table: two passes of 8 lanes at 13 to 16
  parameters and three of 6 at 17 or 18, instead of two of 12, and two of 5
  at 9 or 10 instead of one of 12 (54% less scratch there; faster on x86, up
  to 12% slower on an Apple M4). On a binomial fit with four random effects
  per group (14 parameters) the fit with Hessian SEs peaks at 1.33 GB of
  allocations at 3M rows (formerly 10.7 GB; the same fit with `WaldSe::Rx`
  peaks at 1.31 GB) and 210 MB at 300k (formerly 1.14 GB); its SE step takes
  15 s instead of 47 s and 1.5 s instead of 3.3 s. A crossed binomial with
  four groupings at 3M rows peaks at 2.1 GB (formerly 6.4 GB; 1.25 GB with
  `WaldSe::Rx`) and 280 MB at 300k (formerly 714 MB). On an Apple M4
  with 16 GB the 3M-row single-grouping fit peaks at 1.25 GB and no longer
  swaps. Estimates are bit-identical, native and WASM, over the stress cells
  and the grid's fast subset; the Hessian standard errors of a single-grouping
  fit above 4,096 rows move at round-off (at most 3e-13 relative), because
  each block sums its own share, and every other standard error is
  bit-identical. Timings on the clock-locked machine above.
- **A Gamma GLMM's Hessian standard errors take one assembled gradient for the
  dispersion row instead of a central difference over every coordinate.**
  The `ln φ` row of the joint Hessian is built from the exact gradient the
  assembled pass already forms, where it used to take 2m + 1 further PIRLS
  solves; the central difference stays as the fallback. On a Gamma fit with a
  scalar intercept the whole fit with Hessian SEs takes 3.78 s instead of
  4.08 s at 300k rows and 12.7 s instead of 13.9 s at 1M. Estimates,
  deviance and `n_eval` are unchanged; the standard errors of the grid's
  Gamma cells move by at most 2.8e-9 relative.
- **Finite-difference Hessians under `parallel_inner` keep one worker
  workspace per thread**, instead of one per rayon split. With 12 SE
  parameters a fit builds 6 workers instead of 67 and its SE step takes 8% to
  11% less time; on a packed shape with `k = 2,000` the peak falls from 549 MB
  to 471 MB. Results are bit-identical.

- **AGQ standard errors on a scalar random intercept are faster and use a
  fraction of the memory.** On binomial logit and probit and Poisson-log AGQ
  fits with one scalar random intercept, the joint Hessian is now built from
  central differences of an exact gradient of the AGQ deviance, instead of a
  hyper-dual pass that carried second-derivative lanes on every row. A fit
  with Hessian SEs takes 0.83 to 0.95 of its former time at 1,200 to 300,000
  rows (`nAGQ = 7`), and at 300,000 rows it peaks at 63 to 84 MB instead of
  296 to 464 MB. With more than 12 parameters, where the hyper-dual pass could
  not run and a finite-difference stencil of the deviance took its place, a
  fit takes 0.25 to 0.36 of its former time and the SEs move by about 3e-6
  relative, the stencil's own error. Everywhere else the SEs agree with the
  former ones to about 1e-11 relative. Other AGQ shapes (cloglog, negative
  binomial, Gamma, vector random effects) keep the hyper-dual pass. Timings on
  the clock-locked machine above.

### Added

- **Warm starts carry the dispersion: `StartValues::dispersion`.** A
  negative-binomial shape θ or a Gamma φ, in the units `Fit::dispersion`
  reports, starts the search's dispersion coordinate on a GLMM and the shape
  search on a negative-binomial GLM, so a previous fit's value carries over
  as is. A carried NB shape also skips the no-random-effect GLM-NB prefit
  that only produced a seed. Over the grid's Gamma and NB cells, warm fits
  from the truth or from the cold optimum perturbed by 5% and 20%, with the
  dispersion carried, take 0.73 of the cold fit's evaluations on Gamma-log
  (geometric mean over 87 starts; 0.98 from β and θ alone), 0.76 on negative
  binomial (90 starts; 0.88) and 0.98 on Gamma-inverse (18 starts; 1.04), with
  no `converged` or `singular` flag change and deviances within 2.2e-7 of the
  cold fit's. Python takes it as `warm_start["dispersion"]` and R as
  `start = list(dispersion = …)`; `init_theta=` / `init.theta=`, which raised
  for want of a kernel hook, now seed the negative-binomial shape through the
  same slot (give one of the two).

  **Semver note.** `StartValues` is a struct with public fields, so code that
  builds it with a struct literal must add `dispersion: None`, or end the
  literal with `..StartValues::default()` (the struct now derives `Default`).
- **`Note::GradientSearchFallback`** (Rust, additive; `Note` is
  `#[non_exhaustive]`), the caution `gradient_search_fallback` in both ports
  (Python `GradientSearchFallbackWarning`, R class
  `fastglmm_gradient_search_fallback`), and
  `FitDiagnostics::gradient_search_fallback` on the loop tier: an LMM's or
  GLMM's gradient search did not converge and BOBYQA finished from its best
  point.
  **Semver note:** `FitDiagnostics` is a plain struct of the unstable loop
  tier (`loop_advanced`); code that builds one with a struct literal adds the
  new field.
- **`Diagnostics::n_grad`, `outer_stop` and `outer_decrement`** (Rust,
  additive; `Diagnostics` is `#[non_exhaustive]`), with the new enum
  `OuterStop`: on LMM and GLMM fits that search with BFGS, the search's
  gradient count, why it stopped, and its estimate of the deviance decrease
  left at the endpoint. Python `Fit.n_grad` / `outer_stop` /
  `outer_decrement` and R `$n_grad` / `$outer_stop` / `$outer_decrement`
  carry them, and so does `FitDiagnostics` on the loop tier.
- **Curvature hints for simulation loops** (`loop_advanced`, no semver
  guarantee): `curvature_hint(&mut ws)` takes a `CurvatureHint` after a pilot
  fit (the inverse of a finite-difference Hessian of the gradient at its θ̂),
  and `fit_on_hinted` passes it with each later draw so the gradient search
  starts from that curvature instead of the identity. The fit only reads the
  hint: results are a function of the data, the start and the hint, the same
  in any call order. A hint of another θ dimension panics; on a draw that
  searches with BOBYQA it is ignored. On 20-draw loops from the truth warm
  start (10 route cells, whole `fit_on` calls), the gradient search takes
  0.33 of BOBYQA's time without a hint and 0.25 with it (0.75 of the
  unhinted time), and no draw is more than 1.25× slower with the hint.
  `fit_on` is `fit_on_hinted` with no hint. On a routed GLMM workspace,
  `curvature_hint` after a pilot fit returns the pilot search's own final
  inverse-Hessian estimate, at no cost, and a hint's dimension is `n_θ + p`
  (`n_θ + p + 1` on a Gamma fit with a free dispersion); a hint of another
  dimension panics, as on an LMM workspace.
  **Semver note:** all of it is on
  the unstable loop tier (`loop_advanced`, outside the semver-covered
  surface); `fit_on`, `fit_cold` and `fit_warm` keep their signatures.

- **GLMM fits raise `Note::IllConditioned`** (kind `ill_conditioned`, the
  "Nearly collinear columns" caution in both ports), as OLS, GLM and LMM fits
  already did. The pivot is that of the fixed-effect information the standard
  errors come from: the Rx Schur complement under `WaldSe::Rx`, the
  fixed-effect block of the joint Hessian under `WaldSe::Hessian`. The floor is
  the LMM's 1e-12. On the loop tier, `FitDiagnostics::pivot`, `pivot_col` and
  `ill_conditioned` now carry real values on the GLMM route.
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
- **GLMM fits raise `Note::PrecisionLimited`** (kind `precision_limited`, the
  "Numbers too large for full precision" caution; Python
  `glmm.PrecisionLimitedWarning`, R class `fastglmm_precision_limited`) when
  the deviance is too large for f64 to resolve the optimum. The rounding error
  of the deviance is estimated as `ε·(|D| + tr_x)`, where `tr_x` is the trace
  of `M'WM` over the extra grouping factors' columns and bounds the rounding of
  `log|A|`. The note is raised above 0.01 deviance units, about a tenth of a
  standard error in θ̂. That happens when the deviance is near 1e13 or more (a
  model that misses a large effect at counts in the billions), and on crossed
  or nested intercepts at counts near e³⁰. Estimates and `converged` are
  unchanged. On the loop tier, `GlmmFit::precision_limited` and
  `FitDiagnostics::precision_limited` carry it.
- **GLM and GLMM fits raise `Note::Separated`** (kind `separated`) when the
  fixed-effects IRLS stops on separation. On a GLMM that is the GLM the search
  starts from, so the search never runs. The ports raise no warning of their
  own for it: they read it to choose the non-convergence message. On the loop
  tier, `GlmmFit::separated` and `FitDiagnostics::separated` carry it.

### Removed

Breaking. MCPower calls none of these, and the crate and the ports no longer do.

- **`Sizing::atom` and `Sizing::cluster_of_row`.** The row layout they
  describe belongs to the caller's data generator, not to the fit.
- **`Scalar::abs` and `Scalar::mul_add`** on the doc-hidden
  `glmm::scalar::Scalar` trait.
- **`loop_advanced`: `LmmGroupings::single` and `LmmSuffStats::new`.** Use
  `LmmGroupings::from_cluster_spec` and `LmmSuffStats::with_groupings`.
- **`loop_advanced`: `GlmmFit::tau_squared_hat`.** `FitView::dispersion()` on
  a GLMM reports the same random-intercept variance, in the caller's units.
- **`loop_advanced`: the LMM objective seam** (`lmm_objective_at`,
  `lmm_sweep_fit`, `lmm_sweep_fit_on`, `build_lmm_seam_ws`, `LmmSeamWs`,
  `LmmSweepOutcome`), a dev surface for adjudicating θ-searches.
- **`Note::ReDesignScaleSpread`** (kind `re_design_scale_spread`, Python
  `glmm.ReDesignScaleWarning`, R class `fastglmm_re_design_scale_spread`) and
  `orchestrate::NoteInfo::ratio`. Since every fit standardizes its design
  internally, the fit no longer depends on predictor scale; to make the
  reported random-effect stddevs easier to read, center or rescale the
  predictor before fitting.

### Added — `loop_advanced` (no semver guarantee)

- **`FitView::dispersion_start()`**: the estimated dispersion to carry into
  the next draw's `StartValues::dispersion`, θ̂_NB on the negative-binomial
  routes and φ̂ on a Gamma GLMM (where `FitView::dispersion` is the
  random-intercept variance `D̂[0][0]`), `None` where no dispersion is
  estimated.

- **`FitWorkspace::set_prestandardized(bool)`**, default `false`: `true`
  skips the per-draw standardization and fits the design as given, with every
  map the identity except the random-slope columns, which are scaled by their
  prior-weighted root mean square. The same model; on data far from standardized the search
  is as sensitive to units as the raw design makes it. `fit_cold` and
  `fit_warm` always standardize.
- **`FitWorkspace::set_fixed_design(bool)`**, default `false`: a promise that
  the following `fit_on` calls pass the same design, group ids and weights,
  with only the response (and offset) changing. The design-only setup is then
  computed once: on a random-intercept LMM at n = 4000 a draw takes about a
  quarter fewer instructions than without the promise (x86). Results are
  bit-identical; a debug build panics if the design changes under the promise.
- **`GlmFitView::separated`**: true when a binomial, Poisson or
  negative-binomial GLM ended on the separation stop, or ended without settling
  with a linear predictor more than 30 from its null-model value that was still
  moving.

### Fixed

- **R: `REML =` is checked against the model.** `REML = TRUE` on a model with
  no REML objective (OLS, a GLM, a GLMM) was silently accepted; it is now an
  error saying the model is fitted by maximum likelihood. `REML = FALSE` on
  those models was refused as "REML-only by design"; it now fits the default
  model. On a Gaussian mixed model nothing changes, unless every grouping has
  a single level and is dropped: the model is then fitted by maximum likelihood
  and `REML = TRUE` errors. A `REML` that is not one `TRUE` or `FALSE` is an
  error.

- **A GLMM whose pinned variance starved the search no longer reports a
  converged fit above its nested model's optimum.** A variance whose optimum
  is 0 puts a trough `log(1 + θ²·Σw)` in its parameter, and under very
  precise data the search's trust radius shrinks to its end inside it with
  the other variance parameters still off. The trough polish that re-searches
  them from the pinned point ran only when the pin itself lowered the
  objective, which an endpoint at the trough's bottom does not do, and where
  the endpoint lands inside the trough is round-off: one crossed Poisson fit
  at counts near e¹⁴ ended 2.6 deviance units above its `(1 | g)` submodel on
  aarch64 and at the optimum on x86. Every kept pin now also probes its free
  variance parameters at the pinned point (two evaluations each) and runs the
  polish when the probe predicts a gain, and a polish that finds nothing
  lower re-evaluates the pinned point so the reported modes are its own.
  On the grid's fast subset every deviance is unchanged bit for bit (116
  cells, no `converged` or `singular` change) and its 12 pinned GLMM cells
  take the probe's two evaluations per free variance parameter (4 to 6 on
  most; 70 on the 36-parameter `binb_q8_g3000p20_bal_base`, 226 to 296);
  none of them runs the polish.
- **GLMs at very large counts now converge.** Above means of about 1e5,
  rounding error in the deviance kept the stopping rule from firing, so Poisson
  and negative-binomial fits reported not converged at the right answer. The fit
  now also stops when the last step moved no linear predictor by more than
  rounding error.
- **Poisson, negative-binomial, binomial and Gamma GLMMs keep converging at very
  large counts or trial weights.** The unit deviance took its logs as
  `ln y − ln μ`, whose round-off, times a count near 1e10, was larger than the
  inner solve's stopping band and made the outer search's objective noisy. It now
  takes them as `ln(1 + (y − μ)/μ)`. Poisson GLMMs at means near e²³ to e²⁸ that
  failed now converge; other fits move by round-off.
- **Negative-binomial GLMMs at counts near 1e10 now converge, and NB fits keep
  their digits at large counts.** The NB unit deviance was the difference of
  two terms, each about the size of `y − μ`, and at counts near 1e10 their
  round-off was larger than the inner solve's stopping band: 2 of 3 test fits
  failed. The deviance is now written so that neither term is larger than
  needed. The NB log-likelihood's `lnΓ(y + θ) − lnΓ(θ) − lnΓ(y + 1)` was built
  from parts the size of `y·ln y`, which left the outer search's objective no
  finer than 1.6e-2 on 200 counts near 1e10; it now comes from Stirling's
  series, with every part the size of `ln y`. Every NB fit moves by round-off.
  A non-integer NB response now enters `lnΓ` as it is, as in `MASS`; it was
  rounded to a count there before.
- **Binomial offsets near ±30 no longer trip the divergence check.** It now
  measures the linear predictor from the offset, as on the log links.
- **A GLM whose linear predictor passes ±30 at extreme predictor values now
  fits.** The check used to stop any fit once some row's linear predictor
  moved more than 30 from its null-model value, which also stopped good fits
  over a wide predictor range (logit β = 1.6 over x in ±25), and the GLMMs
  seeded from them. Crossing 30 now stops nothing. A fit that ends without
  settling, with a linear predictor past 30 that is still moving, is reported as
  separated, as before.
- **A GLM on nearly collinear predictors converges and keeps its digits.** Each
  IRLS pass solved for β from the Gram `X'WX`, which squares how badly
  conditioned X is: with two predictors entangled to about 1e-6 (condition
  number 2e6 after standardization), logit and Poisson fits ended not converged
  with NaN SEs, and a probit fit converged 4e-4 off. When the Gram's pivot ratio
  is below 1e-3, each pass now solves for the step instead, and the SEs come
  from a QR of the weighted design, as on the OLS route: the same fits match a
  well-conditioned reparameterization to about 1e-10. Better-conditioned
  designs take the old path unchanged, bit for bit.
- **A negative-binomial GLM at very large counts settles its θ.** Each
  alternation round restarted the β fit from scratch and, at counts near 1e5,
  stopped up to one Fisher step short of the maximum at a point that differed
  between two nearby θ, so θ̂ alternated between two values until the round cap
  and the fit carried an `NbShapeUnsettled` note (8 of 400 simulated fits).
  Each round now starts from the last β, as `MASS::glm.nb` does. NB fits move
  within the stopping tolerance.
- **Gamma 1/μ and inverse-Gaussian 1/μ² GLMs with an intercept start at the
  null-model β.** They started from a per-row linear predictor with no matching
  β, so a first step that left the link's domain could not be halved back and
  the fit ended not converged, although the maximum lies inside the domain. The
  fit now starts with the intercept at link(ȳ) and every slope 0, as R's
  `glm` does when given that start, and reaches R's maximum. Without an
  intercept the per-row start stays.
- **Standard errors and test statistics no longer turn NaN in small units.** A
  Cholesky pivot below 1e-30, or a variance or σ̂² below 1e-30, was treated as
  zero. Those numbers carry the units of the data: an inverse-Gaussian/log GLM
  at y·1e120 had the right β and φ̂ but NaN SEs, as did a Gamma/log GLM with
  every weight 1e-70; the OLS and GLM loop-tier t² and contrast t², and the
  LMM joint Wald statistic, were NaN with y in units of 1e-20. Only an exact
  zero now counts as zero. Other fits are unchanged.
- **A GLM halves a step that makes the deviance worse**, up to 16 times per
  iteration, as R package glm2 does. Before, a Fisher step could run off, and
  an inverse-Gaussian or Gamma fit on the log link ended not converged although
  a finite maximum exists. Logit and Poisson fits are unchanged; probit,
  cloglog, Gamma, inverse-Gaussian and negative-binomial fits that converged
  before move by round-off.
- **An inverse-Gaussian GLM on the 1/μ² link no longer reports converged at the
  edge of the link's domain.** A pass whose linear predictor the link cannot
  evaluate now ends the fit not converged.
- **Gamma and inverse-Gaussian GLMs no longer depend on the response's units
  below 1e-10.** The null deviance and the reciprocal-link starting values no
  longer floor the mean at 1e-10, and the Pearson dispersion no longer over- or
  underflows at extreme means.
- **Poisson and aggregated-binomial `logLik` (and AIC/BIC) keep their digits
  at very large counts.** The saturated constant was built from terms the size of
  `y·ln y`, about 1e-5 off per row at y = 1e10. It is now accurate to round-off
  at any count; on ordinary data the value moves by round-off only. Deviance and
  the fit are unchanged.
- **`loop_advanced`: `FitView::diagnostics()` on the negative-binomial routes
  reports the fit's notes.** `ill_conditioned`, `pivot`, `pivot_col`,
  `pirls_exhausted`, `final_pirls_exhausted`, `hessian_fallback`,
  `exact_profile_fallback` and `se_unavailable` are now read back from the
  assembled fit; before, they were always unset there.
- **Gamma-inverse GLMMs no longer depend on the units of y, and the joint
  search no longer stops early when the working weights span decades.** A
  Gamma-inverse fit whose cluster means are far out of scale is searched on a
  rescaled response. A fit that fails, or ends no lower than the Gamma GLM, is
  retried in the units of each of the two possible optima (σ = 0 and
  interior), and reports `converged = false` when a lower point was seen but
  not reached. After a joint (AGQ or packed) search, one PQL re-solve of β
  checks the end point; when it is lower, the search runs again with the fixed
  effects scaled by their information. Before, such fits could report
  converged several to hundreds of deviance units above the optimum (Gamma
  inverse with y in other units, Poisson with cluster means 8 to 12 decades
  apart), or fail. On the accuracy grid's Gamma, AGQ and packed cells, results
  and evaluation counts are unchanged.
- **A GLMM whose pinned variance sits in a narrow trough gets one more search.**
  When pinning a variance component to 0 lowers the deviance by more than
  1e-7, the fit runs one more joint search from the pinned point with that
  component held at 0, and keeps it when its deviance is lower. That search
  scales the fixed effects by their information at the pinned point (one PQL
  solve). With very precise data (for example Poisson counts near e¹² or
  higher with a crossed grouping that has no variance, or a random slope whose
  intercept has no variance), the search could stop with the other variances
  off their optimum by up to several deviance units (10.6 on the worst
  simulated random-slope fit, now 0.06), reported as converged. Fits whose pin
  does not lower the deviance that much, which includes every accuracy-grid
  cell, are unchanged.
- **A GLMM variance left at or below 1e-4 is set to 0 only when that does not
  make the fit worse.** If pinning it raises the deviance more than 1e-6 above
  the search's best point, the fit is reported at the search's point,
  converged and not singular. Before, very precise data (for example Poisson
  counts near e¹⁶ with a true SD of 9e-5) could report a converged, singular
  fit tens of deviance units worse than the optimum, and worse than the GLM
  without random effects.
- **Negative-binomial GLMs resolve θ̂ more closely.** The profile search over
  `ln θ` now stops at width 1e-7 instead of 1e-4, so θ̂ lands within about 3e-7
  of the joint maximum-likelihood estimate (up to 3e-5 before). Every NB GLM
  result moves slightly, and so does every NB GLMM, whose θ search starts from
  that θ̂. No convergence or singular flag changes on the accuracy grid.
- **A Gamma 1/μ or inverse-Gaussian 1/μ² GLM without an intercept starts from
  a column whose values all share one sign**, so a first step that leaves the
  link's domain is halved back instead of ending the fit. On 400 simulated fits
  with an interior maximum, the 60 that ended not converged now converge. Fits
  with an intercept are unchanged.
- **Python/R: a Gamma 1/μ or inverse-Gaussian 1/μ² fit with response values
  outside the link's range says so.** When 1/y or 1/y² falls outside 1e-75 to
  1e75, the `glm_diverged` and `fit_failed` warnings name that cause and suggest
  rescaling the response or using the log link.
- **GLMMs at huge counts or trial weights keep their digits in PIRLS.** The
  conditional-mode step is solved as an increment (`A·δu = M'ρ − u`), and the
  β-profile border builds `S_β` and its right-hand side from design residual
  rows instead of subtracting terms of order Σw. Crossed `(1|g)+(1|h)` fits no
  longer fail with an infinite deviance at counts near e³³, and fits with a
  cluster-level covariate at such counts converge on the exact profile instead
  of falling back or ending not converged. Every GLMM result moves by round-off;
  no convergence or singular flag changes on the accuracy grid.
- **Binomial rows near μ = 1 keep their digits.** Probit, cloglog and
  weighted-logit rows with 1 − μ below 1e-4 now take their deviance, score and
  weights from η. Above μ ≈ 1 − 1e-8 the old μ form lost up to four digits,
  enough to leave probit and cloglog fits converged up to 6e-5 off the maximum
  in β.
- **GLMs on probit, cloglog, Gamma/log, negative-binomial/log and
  inverse-Gaussian/log take Newton steps** (observed weights), with a Fisher
  fallback. Fits that Fisher scoring left slow or overflowing now reach the
  maximum; on simulated sets the not-converged count fell from 19 to 0 of 200
  (Gamma/log), 21 to 0 (NB/log), 17 to 1 (IG/log) and 15 to 1 (cloglog with
  high-leverage rows), with no fit going from converged to not converged.
- **The negative-binomial θ starting value takes the offset into account.**
  With exposures that differ by decades it used to fall to the lower bound.
  Results move only by round-off.
- **A negative-binomial GLMM fitted to data with no overdispersion reaches the
  upper bound of θ** (the Poisson limit) instead of stopping short of it at θ̂
  in the thousands with a slightly lower log-likelihood. No accuracy-grid result
  moves; exact-profile NB fits take one more evaluation.
- **An exact-fit Gamma GLMM whose search stops just above the dispersion floor
  is refused** like one that reaches the floor. It used to report
  `converged = true` with a log-likelihood up to 3.6 below the value at the floor.
- **Clearer errors for bad input.** A formula with an empty name in an
  interaction (`y ~ :`, `y ~ *`, `y ~ x:`) is a syntax error; `y ~ :` used to
  crash with an index error, and the others were fitted as a smaller model.
  `dispersion`, warm-start values of the wrong length and a non-finite
  `offset()` term are refused with a message that names the argument, not an
  internal struct. Python refuses dict columns of different lengths (they were
  cut to the shortest one when it held a missing value), and R refuses a list
  column that `as.data.frame` would recycle.

- **OLS and the GLMM Rx standard errors keep their digits on nearly collinear
  predictors.** Both built their solve from the Gram `X'WX`, which squares how
  badly conditioned X is: with two predictors entangled to about 1e-6 (condition
  number 2e6 after standardization), an OLS coefficient and its SE were about
  3e-4 off and a GLMM Rx variance 9e-5 off a high-precision reference. When the
  Gram's pivot ratio is below 1e-3, both now refactor on `X·U⁻¹` from a QR of the
  (prior-weighted) design, as the LMM kernels do, and map back: the same designs
  are now within 5e-12 (OLS SE) and 2e-12 (Rx variance). Better-conditioned
  designs take the old path unchanged, bit for bit.
- **The GLMM Rx standard errors keep their digits at very large counts or
  trial weights.** The Rx covariance formed the Schur complement
  `X'WX − X'WZ(Z'WZ+I)⁻¹Z'WX` by subtraction, which cancels when a
  cluster-level covariate meets a large random-effect variance times a large
  weight: at counts near e³³ the Rx SEs were about 50% off with no note, and
  above that the complement could stop being positive definite. It is now
  summed from the residual rows `x − T'm`, and the same fits match a 60-digit
  reference to 1e-13. Other fits move by round-off.
- **The GLMM Hessian standard errors keep their digits on nearly collinear
  predictors**, on every family and both exact and finite-difference Hessians.
  The β block of the joint Hessian is a Gram in X: at a condition number of 1e6
  the Hessian variances were about 5e-4 off, and a finite-difference Hessian
  (the last fallback) was off by its whole value. When `X'WX` has a pivot ratio
  below 1e-3, the Hessian is now taken in the fit's own orthogonalized β
  coordinates and mapped back: the same designs are within 3e-11.
  Better-conditioned designs take the old path unchanged, bit for bit.
- **The joint Wald statistic keeps its digits on nearly collinear
  predictors.** It re-formed the information matrix from its factor, factored
  it again, inverted it and factored the target block of that inverse: at a
  condition number of 2e6 it was 2e-4 off for two targets. It is now read from a
  QR of the per-target columns the standard errors are computed from, on the
  LMM and on both GLMM SE methods, including the Hessian method's fallback to
  the Rx covariance: 3e-11 on the same design. On every design the statistic
  moves by round-off.
- **A weighted dense LMM fit on a reused loop-tier workspace no longer
  allocates an n-sized vector of rescaled weights per call**; the workspace
  keeps it. Results are bit-identical.
- **A GLMM on a reused loop-tier workspace no longer rebuilds its sparse
  structures on every draw** when the grouping ids are unchanged (packed and
  structured layouts, and the negative-binomial route). Allocations per draw on
  a 40×30 crossed Poisson fit drop from 222 to 23 blocks. Results are
  bit-identical.
- **An exact-fit Gaussian mixed model reports an infinite REML criterion on
  every platform.** When the response is an exact linear function of the
  predictors, the last pivot of the blocked factorization is pure round-off, and
  its sign differs between platforms: on wasm32 it came out slightly positive and
  the fit reported a finite criterion (−345.2 on the test fixture) instead of
  `+inf`. A residual at or below the round-off floor (dim·ε·yᵀy) now counts as an
  exact fit everywhere. No other fit is affected.
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
