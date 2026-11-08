# fastglmm() - the package's one entry point. The R side does row filtering
# (subset/na.action), family normalization, argument validation, and column
# marshalling; the formula is parsed and lowered by the Rust side
# (glmm::formula - one parser shared with the Python port, spec section 3), so this
# file deliberately contains no model.matrix / terms machinery.

`%||%` <- function(x, y) if (is.null(x)) y else x

# A bad argument value, formatted for an error message the same way the
# Python port's `{value!r}` does: a single string quoted, anything else
# deparsed. Mirrors glmm-py/glmm/__init__.py's use of `!r` in its argument
# checks - change together.
.arg_repr <- function(x) {
  if (is.character(x) && length(x) == 1L) sprintf("'%s'", x) else deparse(x)[1]
}

# Family/link table - mirrors glmm-py/glmm/__init__.py::_FAMILIES and
# GLMM/glmm-rs/src/family.rs; change together. Links are the port vocabulary, not R's
# (R's Gamma "inverse" maps to "inverse", "1/mu^2" to "inverse_squared").
.FAMILIES <- list(
  gaussian         = list(default_link = "identity", links = "identity"),
  binomial         = list(default_link = "logit", links = c("logit", "probit", "cloglog")),
  poisson          = list(default_link = "log", links = "log"),
  gamma            = list(default_link = "log", links = c("log", "inverse")),
  negativebinomial = list(default_link = "log", links = "log"),
  inversegaussian  = list(default_link = "log", links = c("log", "inverse_squared"))
)

# Families where dispersion= is meaningful - mirrors
# glmm-py/glmm/__init__.py::_DISPERSION_FAMILIES; change together.
.DISPERSION_FAMILIES <- c("binomial", "poisson", "gamma", "inversegaussian")

# Mirrors GLMM/glmm-rs/src/consts.rs::MAX_NAGQ - change together.
.MAX_NAGQ <- 25L

#' Fit a (generalized) linear or linear mixed model with the glmm Rust kernel
#'
#' One entry point for the whole `glmm` engine, dispatching on `family` the way
#' the kernel itself does: OLS (`gaussian` without random effects), GLM
#' (binomial/poisson/gamma/negative-binomial without random effects), LMM
#' (`gaussian` with random effects; REML by default, ML with `REML = FALSE`),
#' and GLMM (Laplace or adaptive Gauss-Hermite) - there is no `lmer`/`glmer`
#' split.
#'
#' The formula is parsed by the same Rust parser the Python port uses. Besides
#' bare column names, it accepts `log(x)`, `sqrt(x)`, `exp(x)`, and `I(x^k)`
#' (`k >= 2`) of a single column; `cbind(successes, failures)` on the
#' left-hand side with `family = binomial`; an `offset()` term; and term
#' removal (`- 1` / `0 +`). Not accepted: `poly()`, arithmetic inside a call,
#' nested calls, and `.` - compute those columns first and pass them by name.
#' `weights=` together with a `cbind()` formula, and `offset=` together with
#' an `offset()` formula term, both error naming "use one". Fixed and random
#' effects support `+`, `:`, `*`, `A/B` nesting, and `(a + b)^k` crossing.
#' Random-effect terms may be correlated `(1 + x | g)`, uncorrelated
#' `(x || g)` or `zerocorr(1 + x | g)`, intercept-free `(0 + x | g)`, or
#' several terms on one grouping such as `(1 | g) + (0 + x | g)`; a column in
#' two terms on one grouping is an error. On a factor slope, `(f || g)` keeps
#' the factor's block correlated, as lme4 does, while `zerocorr(f | g)` makes
#' every column uncorrelated. Contrasts are
#' always treatment coding with the **first factor level** with rows as base
#' (unused levels are dropped, as in `lm` and lme4); to change
#' the base, `relevel()` the factor (a `contrasts` argument is deliberately
#' absent). Anything that would silently mean something else - an ordered
#' factor, a factor with its own `contrasts` attribute, or the unordered slot
#' of `options(contrasts = )` set away from `"contr.treatment"` - is a clear
#' error when the column is used in the fixed-effects part of the formula
#' (random-effect grouping and slope columns are not checked); character and
#' logical columns are checked the same way as an unordered factor, since
#' `model.matrix()` codes them the same way. Character and logical columns
#' are converted to factors with lexicographic level order (as `factor()`
#' does, so a logical `x` gives levels `FALSE`/`TRUE` and a coefficient
#' `xTRUE`, as in lme4); a factor's declared level order is honored. A
#' logical response or `offset()` variable stays a 0/1 numeric.
#'
#' **`Gamma()` link trap:** R's `Gamma()` family object defaults to
#' `link = "inverse"`, and a family *object* is honored as given - R semantics
#' win. The string form `family = "gamma"` uses the glmm default
#' `link = "log"` instead. The two forms therefore fit different models;
#' choose deliberately.
#'
#' **`nAGQ` fallback (louder than lme4):** `nAGQ > 1` is honored on binomial,
#' Poisson, negative-binomial and Gamma mixed models with a single grouping
#' factor and at most 3 random
#' effects per group. Any other shape **warns and falls back to Laplace**
#' (`nAGQ = 1`) instead of erroring the way `lme4::glmer` does - the fit you
#' get is a Laplace fit, and the warning is the only notice. This mirrors the
#' Python port so the two ports agree.
#'
#' **Diagnostic warnings carry a condition class.** Each solver note is raised
#' as a warning of class `"fastglmm_diagnostic"` with a per-kind subclass
#' (`"fastglmm_ill_conditioned"`, `"fastglmm_pirls_exhausted"`,
#' `"fastglmm_single_level_grouping_dropped"`,
#' `"fastglmm_few_rows_for_random_effects"`, `"fastglmm_repeated_grouping"`,
#' `"fastglmm_slope_constant_within_levels"`, `"fastglmm_separated_by_grouping"`,
#' `"fastglmm_hessian_se_fallback"`,
#' `"fastglmm_se_unavailable"`, `"fastglmm_exact_profile_fallback"`,
#' `"fastglmm_gradient_search_fallback"`,
#' `"fastglmm_precision_limited"`, `"fastglmm_non_integer_response"`, `"fastglmm_rows_dropped_na"`,
#' `"fastglmm_satterthwaite_curvature"`, `"fastglmm_variance_at_limit"`,
#' `"fastglmm_weight_power_at_bound"`, `"fastglmm_weight_power_search_failures"`,
#' `"fastglmm_weight_power_outside_unit"`, `"fastglmm_weight_power_implausible"`,
#' `"fastglmm_robust_with_random_slope"`, `"fastglmm_robust_se_unidentified"`,
#' `"fastglmm_robust_df_small"`; a note from
#' a newer kernel than this package arrives as `"fastglmm_unknown_note"`;
#' [contrast_test()], [pairwise()] and [predict()][predict.fastglmm] on a
#' robust fit add `"fastglmm_model_based_joint_test"` and
#' `"fastglmm_predict_se_model_based"`),
#' so `withCallingHandlers()` and `suppressWarnings(classes = )` can select
#' them without matching message text.
#'
#' Anything the engine cannot do is an error naming the reason - never a
#' silently different model. That includes `control=`/`verbose=` and
#' quasi-likelihood `dispersion=` on binomial/poisson. `REML = TRUE` on a model
#' that is not Gaussian does not apply (the model is fitted by maximum
#' likelihood): it is ignored with a warning, as an inapplicable `nAGQ` is.
#'
#' @param formula an lme4-style model formula (or a string), e.g.
#'   `y ~ t + d + t:d + (1 + t | g)`. See Details for what the shared parser
#'   accepts beyond bare column names.
#' @param data a `data.frame` (or something coercible) holding every column
#'   the formula names.
#' @param family a family object, family function, or string: one of
#'   `gaussian`, `binomial` (logit/probit/cloglog), `poisson` (log), `Gamma`
#'   (log/inverse), `inverse.gaussian` (log/`1/mu^2`), `"negativebinomial"`
#'   (log; the shape `theta` is estimated).
#' @param weights optional per-row prior weights - `lme4::glmer`'s
#'   `weights=`. On a family with an estimated dispersion (gaussian, gamma,
#'   inverse-Gaussian) these are PRECISION weights, the same convention as
#'   `lm`, `glm` and lme4: row `i` has dispersion `phi / weights[i]`, so
#'   multiplying every weight by the same constant leaves the fit unchanged.
#'   For an aggregated binomial, pass the success **proportion** as the
#'   response and the trial count here, or equivalently write
#'   `cbind(successes, failures)` as the formula's left-hand side; combining
#'   both is an error.
#' @param subset optional row filter, evaluated in `data` like `lm`'s
#'   `subset=`. Applied before fitting - row filtering, not parsing.
#' @param na.action how to handle `NA`s in the model columns (default
#'   `getOption("na.action")`, normally [stats::na.omit]). `NA`s must be
#'   resolved before the kernel sees the data; an action that leaves them in
#'   place (`na.pass`) is an error.
#' @param offset optional per-row known additive term on the
#'   linear-predictor scale, `glm`'s `offset=`: `eta = offset + X b (+ Z u)`,
#'   with no coefficient estimated for it. The usual use is a Poisson rate
#'   model with a known exposure, `offset = log(exposure)`. Evaluated in
#'   `data`, so an expression works; `subset=` and `na.action` drop its
#'   entries with the rows. Positioned after `na.action` as in [stats::glm].
#'   Equivalent to an `offset()` term in the formula; combining both is an
#'   error.
#' @param nAGQ adaptive Gauss-Hermite node count: an odd integer in
#'   `1..=25` (`1` = Laplace, the default). More permissive than lme4 in range
#'   but see the fallback note in Details.
#' @param start optional warm start, lme4's name and shape:
#'   `list(beta =, theta =)` with `theta` the random-effect Cholesky vector,
#'   plus an optional `dispersion =`: the estimated dispersion as a previous
#'   fit reports it in `fit$dispersion` (the shape for negative binomial, phi
#'   for Gamma), used as the search's starting value. Any element may be left
#'   out to cold-start that part.
#'   `theta` is distinct from `init.theta`, which is the negative-binomial
#'   shape (the same slot as `start$dispersion` there).
#' @param wald.se Wald standard-error mode: `"hessian"` (default) or `"rx"`.
#' @param dispersion Gamma/inverse-Gaussian dispersion directive: `NULL`
#'   (estimate it, the default: the Pearson moment on a Gamma GLM, maximum
#'   likelihood on a Gamma GLMM, the Pearson moment on inverse-Gaussian),
#'   `"estimate"` (same), or a number to
#'   hold it fixed. Non-`NULL` on binomial/poisson would mean
#'   quasi-likelihood - not implemented, errors.
#' @param init.theta negative-binomial shape seed, named for
#'   `MASS::glm.nb(init.theta=)`: the starting value of the shape search
#'   (the same slot as `start$dispersion`; give one of the two). `NULL`
#'   cold-starts it from the no-random-effect negative-binomial fit.
#' @param satterthwaite compute lmerTest's Satterthwaite degrees of freedom
#'   for a Gaussian mixed model's t tests (default `TRUE`). `FALSE` skips that
#'   pass: [summary()][summary.fastglmm] and [confint()][confint.fastglmm]
#'   then use z and normal quantiles and say "df not computed", and
#'   `$df_method` is `"disabled"`. OLS and the Gamma and inverse-Gaussian GLMs
#'   keep their residual df whatever it says.
#' @param condvar compute the conditional variances of the random effects
#'   (default `TRUE`), which [ranef()][ranef.fastglmm]'s `postVar` attribute
#'   and a conditional [predict()][predict.fastglmm] with `se.fit` or `interval`
#'   read. `FALSE` skips them, which saves up to about the time of the fit
#'   itself on a crossed model with many levels: `ranef()` then leaves
#'   `postVar` off, and `ranef(fit, condVar = TRUE)` and such a `predict()` are
#'   errors.
#' @param REML `NULL` (the default), `TRUE` or `FALSE`; `$reml` reports the
#'   value used. A Gaussian mixed model is fitted by REML under `NULL` and
#'   `TRUE` and by maximum likelihood under `FALSE`. A Gaussian model with no
#'   random effects (OLS, including a formula whose every grouping was dropped
#'   as single-level) without weights, or with a fixed `weight_power`, has the
#'   same estimates under every value: `TRUE` reports the REML log-likelihood
#'   (as `logLik(lm, REML = TRUE)`), `NULL` and `FALSE` the ML one. Where
#'   `weight_power = NULL` estimates the weight power, `NULL` and `TRUE`
#'   estimate it by REML (`$reml` is `TRUE`, the log-likelihood is the REML
#'   one) and `FALSE` takes the weights as given (power 1), so the estimates
#'   differ; `weight_power = "estimate"` with `FALSE` estimates the power by
#'   ML. Every other family is fitted by maximum likelihood, and
#'   `TRUE` there is ignored with a `"fastglmm_argument_ignored"` warning.
#' @param anova_test when the likelihood-ratio tables of
#'   [anova()][anova.fastglmm] are made. Every fit stores its Wald chi-squared
#'   (or F) tables, with no refit. `"auto"` (the default) makes the
#'   likelihood-ratio tables the first time `anova()` needs them, which on a
#'   Poisson model with random effects is `anova(m)` itself; `"LRT"` makes them
#'   with the fit, on every fit with chi-squared rows (any GLMM, and every GLM
#'   but a Gamma or inverse-Gaussian one with an estimated dispersion);
#'   `"Wald"` never makes them, and `anova()` then refuses a test that needs
#'   them. The tables refit the model under each term's hypothesis, and under
#'   `"auto"` once more for the fit itself. The fit itself is the same under
#'   every value.
#' @param weight_power `NULL`, `"estimate"` or a finite number: the power on
#'   `weights`, so row `i` has variance `sigma^2 * weights[i]^-weight_power`.
#'   `"estimate"` fits the power by REML (by ML with `REML = FALSE`) and refits
#'   at it; the power counts as one parameter in the fit's df, and the standard
#'   errors treat it as known. It needs a Gaussian model with weights not all
#'   equal, at most one grouping factor and at least 20 rows. `NULL` is
#'   `"estimate"` on such a model fitted by REML (`REML` `NULL` or `TRUE`), and
#'   1 everywhere else. A number fits on `weights^weight_power`: 1 takes the
#'   weights as given, 0 ignores them; a value other than 0 and 1 has no
#'   simulation evidence behind it. A number other than 1 needs a Gaussian model
#'   when there are weights; without weights it changes nothing and warns.
#'   `fit$weight_power` reports the value used, the estimate on a fitted power.
#' @param se_type `NULL`, `"model"` or `"robust"`, the standard error the fit
#'   reports. `NULL` is `"robust"` on a REML fit that estimates the weight power
#'   with one grouping factor and no random slope (CR2, Pustejovsky-Tipton df)
#'   or with none (HC2, Bell-McCaffrey df), and `"model"` everywhere else.
#'   `"robust"` works on any Gaussian model with at most one grouping factor,
#'   by REML when it has one or estimates the weight power, and warns on a
#'   random slope (`"fastglmm_robust_with_random_slope"`). `fit$se_type`
#'   reports the value used, and `fit$df_method` the df.
#' @param ... intercepted, never silently swallowed. The known lme4 arguments
#'   (`control`, `verbose`, `contrasts`) raise errors saying why they cannot
#'   be honored; unknown names error as unused arguments.
#'
#' @return An object of class `"fastglmm"`: fixed effects ([fixef]), Wald
#'   covariance (`vcov()`), variance components on the SD/correlation scale
#'   ([VarCorr]), `converged` and `singular` flags ([isSingular]), a
#'   `diagnostics` list (the solver's own report: `boundary`, `pinned`,
#'   `notes`, plus the three flags above), a `warnings` data.frame (one row
#'   per warning the call raised: `tier`, `kind`, `title`, `message`; match on
#'   `kind`), `n_eval` (the optimizer's objective evaluations, 0 on the
#'   closed-form and IRLS paths), `n_grad`, `outer_stop` and `outer_decrement`
#'   (the BFGS gradient search's gradient evaluations, not counted in
#'   `n_eval`; why it stopped, `"fdelta"`, `"pgrad"` or `"noise_floor"` when
#'   it converged, otherwise `"line_search"`, `"max_iter"`,
#'   `"grad_unavailable"` or `"non_finite"`; its estimate of the deviance
#'   decrease left; it runs on the models the
#'   `"fastglmm_gradient_search_fallback"` warning names, and they describe it
#'   even when the derivative-free search finished the fit; `0`,
#'   `NA_character_` and `NaN` on every fit that ran no gradient search), the
#'   per-coefficient `coef_df` (the t test's degrees of freedom,
#'   `NaN` where the test is z and on an OLS or dispersion-GLM fit that did not
#'   converge) and `coef_p` (its p-value), `df_method`
#'   (`"satterthwaite"`, `"residual"`, `"none"`, `"disabled"`, `"failed"`, and
#'   on a robust fit `"pustejovsky_tipton"` or `"bell_mccaffrey"`),
#'   `weight_power` (the power used on `weights`: the estimate on a fitted
#'   power, 1 on the weights as given, `NaN` without weights),
#'   `estimated_weight_power` (`TRUE` when the fit estimated it), `se_type`
#'   (`"model"` or `"robust"`, the standard error behind `se`, `vcov` and
#'   `coef_df`), `model_vcov` (on a robust fit, the model-based covariance
#'   that [predict()][predict.fastglmm] and the Type II hypotheses read;
#'   `NULL` otherwise), `robust` (on a robust fit, the CR2 or HC2 pieces its
#'   tests read; `NULL` otherwise),
#'   `rss`, `r_squared` and `adj_r_squared` (OLS fits only, `NaN` otherwise),
#'   `offset` (the offset fitted with, `NULL` without one),
#'   `satterthwaite_record` (what [contrast_test()] and [pairwise()] read on
#'   a fit whose `df_method` is `"satterthwaite"`, `NULL` on every other fit),
#'   plus [anova()][anova.fastglmm] (the per-term tests, or a comparison of
#'   two fits), [pairwise()] (the pairwise comparisons of a
#'   factor's levels), [r2()] and [icc()] (a mixed model's R-squared pair and
#'   intraclass correlation), `print()`, [summary()][summary.fastglmm],
#'   [confint()][confint.fastglmm] (Wald), `nobs()`,
#'   [formula()][formula.fastglmm] (returns the model formula as given, so
#'   `update()` works), `family()`, `model.frame()`, [ranef()][ranef.fastglmm],
#'   `fitted()`, `residuals()`, [predict()][predict.fastglmm],
#'   [simulate()][simulate.fastglmm], [logLik()][logLik.fastglmm], and
#'   [AIC()][AIC.fastglmm] and `BIC()` (which refuse a REML fit beside an ML
#'   one, and REML fits with different fixed effects). The blocked accessors
#'   (`coef`, `terms`) error with the reason.
#'
#' @examples
#' set.seed(1)
#' n_g <- 30L
#' g <- factor(rep(seq_len(n_g), each = 10))
#' t <- rep(seq(0, 0.9, by = 0.1), n_g)
#' d <- rbinom(n_g * 10L, 1L, 0.4)
#' u <- rnorm(n_g, sd = 0.8)[as.integer(g)]
#' y <- rbinom(n_g * 10L, 1L, plogis(-0.5 + 0.7 * t + 0.4 * d + u))
#' fit <- fastglmm(y ~ t + d + t:d + (1 | g), data.frame(y, t, d, g),
#'                 family = binomial())
#' fixef(fit)
#' VarCorr(fit)
#' @export
fastglmm <- function(formula, data, family = gaussian(),
                     weights = NULL, subset = NULL,
                     na.action = getOption("na.action"),
                     offset = NULL,
                     nAGQ = 1L,
                     start = NULL,
                     wald.se = "hessian",
                     dispersion = NULL,
                     init.theta = NULL,
                     satterthwaite = TRUE,
                     condvar = TRUE,
                     REML = NULL,
                     anova_test = c("auto", "Wald", "LRT"),
                     weight_power = NULL,
                     se_type = NULL,
                     ...) {
  .check_dots(...)
  if (!(is.logical(satterthwaite) && length(satterthwaite) == 1L && !is.na(satterthwaite))) {
    stop(sprintf("satterthwaite must be TRUE or FALSE, got %s", .arg_repr(satterthwaite)),
         call. = FALSE)
  }
  if (!(is.logical(condvar) && length(condvar) == 1L && !is.na(condvar))) {
    stop(sprintf("condvar must be TRUE or FALSE, got %s", .arg_repr(condvar)),
         call. = FALSE)
  }
  .check_reml(REML)
  anova_test <- match.arg(anova_test)
  # Captured before `data` is touched: the header's `Data:` line, as lme4
  # deparses it.
  data_name <- paste(deparse(substitute(data), width.cutoff = 500L), collapse = " ")

  if (is.character(formula)) formula <- stats::as.formula(formula)
  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula or a formula string", call. = FALSE)
  }
  f_str <- paste(deparse(formula, width.cutoff = 500L), collapse = " ")
  .check_formula(formula)

  fam <- .normalize_family(family)

  mixed <- grepl("|", f_str, fixed = TRUE)
  if (identical(fam$name, "inversegaussian") && mixed) {
    stop("family 'inversegaussian' is GLM-only: random-effect terms are not supported",
         call. = FALSE)
  }

  if (!(is.character(wald.se) && length(wald.se) == 1L && wald.se %in% c("hessian", "rx"))) {
    stop(sprintf("wald.se must be 'hessian' or 'rx', got %s", .arg_repr(wald.se)),
         call. = FALSE)
  }

  nAGQ <- .check_nagq(nAGQ)

  store <- new.env(parent = emptyenv())
  store$rows <- list()
  ignored <- c("fastglmm_argument_ignored", "fastglmm_diagnostic")

  # Valid-but-inapplicable options: warn and strip, mirroring the Python port
  # (nothing inapplicable may reach the kernel - its checks are assert!s).
  if (!is.null(dispersion) && !(fam$name %in% .DISPERSION_FAMILIES)) {
    .warn_keep(store, "argument_ignored",
               sprintf("dispersion= has no effect for family '%s'.", fam$name), ignored)
    dispersion <- NULL
  }
  if (!is.null(dispersion)) {
    ok <- identical(dispersion, "estimate") ||
      (is.numeric(dispersion) && length(dispersion) == 1L && is.finite(dispersion))
    if (!ok) {
      stop(sprintf("dispersion must be NULL, 'estimate', or a number, got %s",
                    .arg_repr(dispersion)),
           call. = FALSE)
    }
    if (fam$name %in% c("binomial", "poisson") && mixed) {
      .warn_keep(store, "argument_ignored", paste(
        "Quasi-likelihood dispersion= is not supported yet for binomial or Poisson",
        "models. The default dispersion of 1 was used."), ignored)
      dispersion <- NULL
    }
  }
  if (identical(dispersion, "estimate") && fam$name %in% c("gamma", "inversegaussian")) {
    # The phi families' default (NULL) already estimates phi.
    dispersion <- NULL
  }
  if (!is.null(dispersion) && fam$name %in% c("binomial", "poisson")) {
    # mirrors glmm-py/glmm/__init__.py's NotImplementedError block - change together.
    stop("quasi-likelihood dispersion on family '", fam$name,
         "' is not yet implemented in the kernel",
         call. = FALSE)
  }
  if (!is.null(init.theta) && fam$name != "negativebinomial") {
    .warn_keep(store, "argument_ignored",
               sprintf("init.theta= is not used for family '%s'.", fam$name), ignored)
    init.theta <- NULL
  }
  if (!is.null(start)) {
    if (!is.list(start)) {
      stop(sprintf(paste(
        "start must be a list with elements 'beta', 'theta' and/or 'dispersion' (theta",
        "is the random-effect Cholesky vector, not the negative-binomial shape - that is",
        "init.theta or dispersion), got %s"), class(start)[1L]), call. = FALSE)
    }
    unknown <- setdiff(names(start), c("beta", "theta", "dispersion"))
    if (length(unknown)) {
      .warn_keep(store, "argument_ignored", paste0(
        "start accepts only 'beta', 'theta' and 'dispersion'; these entries were ",
        "ignored: ", paste(unknown, collapse = ", "), "."), ignored)
      start <- start[intersect(names(start), c("beta", "theta", "dispersion"))]
    }
  }
  # init.theta is the negative-binomial shape seed: the same kernel slot as
  # start$dispersion, so giving both is ambiguous.
  if (!is.null(init.theta)) {
    if (!is.null(start$dispersion)) {
      stop("give the negative-binomial shape seed once: init.theta= or ",
           "start$dispersion, not both", call. = FALSE)
    }
    start <- c(start, list(dispersion = init.theta))
  }
  if (!is.null(start$dispersion) &&
      !(is.numeric(start$dispersion) && length(start$dispersion) == 1L &&
        is.finite(start$dispersion) && start$dispersion > 0)) {
    stop("start$dispersion (or init.theta) must be one finite number > 0",
         call. = FALSE)
  }

  # --- data prep: row filtering (subset, na.action) happens HERE, on the
  # data.frame, before marshalling - data prep, not parsing (spec section 3.1). ---
  vars <- all.vars(formula)
  data <- .as_data_frame(data, vars)
  frame <- .data_columns(data, vars)
  .check_contrasts(formula, frame)
  e <- substitute(weights)
  if (!is.null(e)) frame <- .add_weights(frame, eval(e, data, parent.frame()), nrow(data))
  e <- substitute(offset)
  if (!is.null(e)) frame <- .add_offset(frame, eval(e, data, parent.frame()), nrow(data))

  # Each row's 1-based position in `data`, carried through subset= and
  # na.action as a column, so an error from the engine names the caller's row
  # (`row_index` of the native fit) whatever the subset or na.action drops.
  frame[["(row)"]] <- seq_len(nrow(frame))
  e <- substitute(subset)
  s <- if (is.null(e)) NULL else eval(e, data, parent.frame())
  if (!is.null(s)) {
    # An NA in a logical subset drops that row, as in lm; `[` would make it an
    # all-NA row that na.action then reports as a missing value.
    if (is.logical(s)) s <- s & !is.na(s)
    frame <- frame[s, , drop = FALSE]
  }

  naf <- if (is.character(na.action)) {
    get(na.action, mode = "function")
  } else {
    na.action
  }
  if (!is.function(naf)) stop("invalid na.action", call. = FALSE)
  n_before_na <- nrow(frame)
  if (!(identical(naf, stats::na.omit) && .na_omit_is_identity(frame))) {
    frame <- naf(frame)
    if (anyNA(frame[vars])) {
      stop("missing values remain in the model columns after na.action; ",
           "the kernel cannot fit NA - use na.action = na.omit or complete ",
           "the data", call. = FALSE)
    }
  }
  if (nrow(frame) == 0L) {
    # Mirrors the Python port's fit(), which raises the same text when its own
    # missing-value row drop empties the data - change together
    # (glmm-py/glmm/__init__.py, the na_mask block). An empty data.frame gets
    # the kernel's text, as Python passes it through. A subset= that keeps no
    # row has no Python counterpart, so it keeps the plain message.
    if (n_before_na > 0L) {
      stop("every row has a missing value in a column the formula uses; no rows left to fit",
           call. = FALSE)
    }
    if (nrow(data) == 0L) {
      stop("Data has no rows (n = 0); a fit needs at least one row", call. = FALSE)
    }
    stop("no rows left to fit", call. = FALSE)
  }
  if (nrow(frame) < n_before_na) {
    # Mirrors the Python port's fit(), which does its own row-drop bookkeeping
    # since it has no model.frame/na.action to do this for free - change
    # together (glmm-py/glmm/__init__.py, the na_mask block).
    .warn_keep(store, "rows_dropped_na", sprintf(
      "Dropped %d of %d row(s): a column the formula uses had a missing value there.",
      n_before_na - nrow(frame), n_before_na),
      c("fastglmm_rows_dropped_na", "fastglmm_diagnostic"))
  }
  row_index <- frame[["(row)"]]
  frame[["(row)"]] <- NULL
  w <- frame[["(weights)"]]
  o <- frame[["(offset)"]]

  cols <- .marshal(frame, vars, c(all.vars(formula[[2L]]), .offset_vars(formula)))

  r <- .glmm_call("fit", list(
    formula = f_str, numeric = cols$numeric, factors = cols$factors,
    family = fam$name, link = fam$link, wald_se = wald.se, nagq = nAGQ,
    dispersion = .optional(dispersion), weights = .optional(w), offset = .optional(o),
    warm_start = .warm_start(start), satterthwaite = satterthwaite, condvar = condvar,
    reml = REML, anova_test = if (identical(anova_test, "auto")) "Wald" else anova_test,
    weight_power = weight_power, se_type = se_type,
    row_index = row_index, row_base = 1L
  ))
  # The native call has refused anything but NULL, "estimate" and a number.
  if (is.numeric(weight_power) && weight_power != 1 && is.na(r$weight_power)) {
    .warn_keep(store, "argument_ignored", sprintf(
      "weight_power = %s has no effect without weights.", format(weight_power)), ignored)
  }

  fit <- .fit_object(r, store, fam, nAGQ, dispersion, f_str, frame, row_index, match.call(),
                     data_name)
  if (identical(anova_test, "auto")) {
    # An environment, so the tables `.lrt_tables` makes stay with every copy
    # of the fit.
    fit$.lrt <- list2env(list(call = list(wald.se = wald.se, nAGQ = nAGQ,
                                          dispersion = dispersion, start = start)),
                         parent = emptyenv())
  }
  fit
}

# The fit object from the kernel's raw list `r`: raises the fit's warnings
# (into `store`, which already holds the data-prep ones) and builds the
# "fastglmm" list. fastglmm() and compare() share it, so a fit made by
# compare() is the object fastglmm() makes. `row_index` is each frame row's
# 1-based position in the caller's data.
.fit_object <- function(r, store, fam, nAGQ, dispersion, f_str, frame, row_index, call,
                        data_name) {
  ignored <- c("fastglmm_argument_ignored", "fastglmm_diagnostic")
  # Whether the FITTED model actually has random effects - not `mixed`, which
  # only says the formula asked for some: a single-level grouping is dropped
  # after lowering ("single_level_grouping_dropped"), and once every
  # random-effect term is dropped this way, `r$re_group_names` comes back
  # empty even though `mixed` (read off the formula text before fitting)
  # still says TRUE.
  fitted_mixed <- length(r$re_group_names) > 0L
  if (!is.null(r$agq_warning) && (identical(fam$name, "gaussian") || !fitted_mixed)) {
    # nAGQ changes nothing here; see the Python port's fit() for why.
    reason <- if (identical(fam$name, "gaussian")) {
      "a Gaussian model"
    } else {
      "a model without random effects"
    }
    .warn_keep(store, "argument_ignored", sprintf(
      "nAGQ=%d has no effect for %s, because nothing is approximated.", nAGQ, reason),
      ignored)
  } else if (!is.null(r$agq_warning)) {
    # Built here, not taken from `agq_warning`: that string says nagq=, not nAGQ=.
    .warn_keep(store, "agq_fallback", sprintf(paste(
      "nAGQ=%d works only for %s models whose random effects are in one",
      "grouping factor, with at most 3 random effects in it. This model was fitted",
      "without adaptive quadrature."), nAGQ, .AGQ_FAMILIES),
      c("fastglmm_agq_fallback", "fastglmm_diagnostic"))
  }
  if (!is.null(r$reml_warning)) {
    # Decided by the shared rule in run_fit (src/orchestrate.rs), on the model
    # actually fitted; worded here, as for nAGQ above, in this port's spelling.
    # Mirrors the Python port's reml_warning block - change together.
    .warn_keep(store, "argument_ignored", paste(
      "REML = TRUE applies only to Gaussian models; it was ignored and the model was",
      "fitted by maximum likelihood."), ignored)
  }
  # Singularity is not assessed on a fit that did not converge: the kernel never sets
  # `singular` there (the post-hoc check and the boundary flags are gated on
  # `converged`), so `r$singular` reads FALSE on a non-converged fit. `isTRUE(r$converged)`
  # below is a defensive guard, not load-bearing on the current kernel.
  if (r$singular && isTRUE(r$converged)) {
    affected <- .pinned_detail(r$pinned, r$re_group_names, r$re_group_terms)
    .warn_keep(store, "singular", paste0(
      "The random effects are too complex for the data: a variance is estimated at or ",
      "near zero, or a correlation at or near \u22121 or 1. Consider removing the ",
      "affected random effect.",
      if (length(affected)) paste0(" Affected: ", paste(affected, collapse = ", "), ".")),
      c("fastglmm_singular", "fastglmm_diagnostic"))
  }
  # The notes and the non-convergence warning come with their text
  # (glmm-bind/src/notes.rs); a note without a message raises nothing.
  for (note in r$notes) {
    if (!is.null(note$message)) .warn_entry(store, note, .note_class(note$kind))
  }
  if (!is.null(r$failure)) .warn_entry(store, r$failure, .note_class(r$failure$kind))

  p <- length(r$beta)
  beta <- r$beta; names(beta) <- r$names
  se <- r$se; names(se) <- r$names
  aliased <- r$aliased; names(aliased) <- r$names
  coef_df <- r$coef_df; names(coef_df) <- r$names
  coef_p <- r$coef_p; names(coef_p) <- r$names
  # Aliased (rank-deficient) slots print as NA, lme4/lm-style, not NaN.
  beta[aliased] <- NA_real_
  se[aliased] <- NA_real_
  vc <- matrix(as.double(unlist(r$vcov)), p, p, byrow = TRUE,
               dimnames = list(r$names, r$names))

  structure(list(
    beta = beta,
    se = se,
    vcov = vc,
    varcorr = r$varcorr,
    stddev_se = r$stddev_se,
    aliased = aliased,
    # Per coefficient: the t test's degrees of freedom (Satterthwaite on a
    # Gaussian LMM, n - p on OLS and the dispersion GLMs; NaN where the test is
    # z) and its two-sided p-value, computed by the kernel. `df_method` says
    # which: "satterthwaite", "residual", "none", "disabled" (satterthwaite =
    # FALSE) or "failed". Mirrors the Python port's Fit fields - change together.
    coef_df = coef_df,
    coef_p = coef_p,
    df_method = r$df_method,
    # The Satterthwaite record contrast_test() reads, on a fit whose df_method
    # is "satterthwaite"; NULL on every other fit, whose test reads only beta,
    # vcov, aliased, nobs and dispersion. list(a_eff, jac_theta, c, tau, nu,
    # retained), every matrix a flat row-major double vector over the p_r
    # retained columns: a_eff ((n_free + 1)^2, 2 blockdiag(H_p^+, tau^2 / nu)
    # over the n_free free theta coordinates and tau), jac_theta (a list of
    # p_r^2 vectors, dC/dtheta_k per free coordinate), c (p_r^2, Cov(beta-hat)),
    # tau (sigma-hat^2), nu (its divisor) and retained (1-based retained
    # columns, NULL when every column was retained). The free theta coordinates
    # are in the kernel's order, which differs from the order the groupings
    # were written in when glmm makes the grouping with the most levels the
    # primary; the df and the contrast tests use them only in sums over all
    # coordinates, so the order does not change any result. About p^2 (n_free + 1)
    # doubles, kept by saveRDS(). Mirrors the Python port's
    # Fit.satterthwaite_record - change together.
    satterthwaite_record = r$satterthwaite_record,
    # OLS fits only (NaN otherwise): weighted RSS and summary.lm's R-squared pair.
    rss = r$rss,
    r_squared = r$r_squared,
    adj_r_squared = r$adj_r_squared,
    dispersion = r$dispersion,
    # The numeric dispersion= argument the caller passed (Gamma or
    # inverse-Gaussian only; NULL when left to estimate). print.summary's
    # dispersion label reads this to print "fixed" instead of "ML"/"Pearson".
    dispersion_held = dispersion,
    converged = r$converged,
    singular = r$singular,
    # Everything the solver reports about the fit itself, mirroring the Rust
    # `Diagnostics` (src/fit/mod.rs) and the Python port's `Fit$diagnostics` -
    # change together. `converged`, `singular` and `aliased` stay at the top
    # level as well; this element is additive.
    #   boundary  "interior" / "at_boundary" / "no_optimum" / "upper_limit":
    #             where the accepted theta sits ("upper_limit": on the upper
    #             limit of the search, otherwise interior). Every theta-carrying
    #             route distinguishes all four (LMM over either kernel, GLMM over
    #             every layout, negative binomial included). OLS and GLM (no
    #             theta) always report "interior", and so does a fit that
    #             failed before any search ran (a degenerate guard), reported
    #             through converged = FALSE.
    #   pinned    one logical vector per grouping, in varcorr order, one entry
    #             per variance component: pinned[[g]][i] pairs with
    #             .stddev_corr(varcorr[[g]])$stddev[i]. ON A CONVERGED FIT,
    #             EMPTY MEANS NOTHING WAS PINNED - a model with no variance
    #             components (OLS, GLM, fixed-effect-only negative binomial)
    #             reports empty for the same reason: there was nothing to pin.
    #             fastglmm() does not error on converged = FALSE: pinned is
    #             empty on every non-converged fit - a failed fit, and a fit
    #             stopped at its evaluation budget, where nothing is pinned at
    #             the capped endpoint.
    #   notes     list of list(kind=, tier=, title=, message= (NULL raises
    #             nothing), columns=, pivot=, evals=, final_eval=,
    #             detail=, delta=, lower=, upper=, failed=, not_converged=);
    #             `columns` is
    #             1-based into `names`. Each is raised as a classed warning by
    #             the call above. An absent note means "not detected", never
    #             "checked and clean": a fit that formed no factor records no
    #             pivot, and every route flags `IllConditioned` rather than
    #             refuse.
    diagnostics = list(
      converged = r$converged,
      singular = r$singular,
      aliased = aliased,
      boundary = r$boundary,
      pinned = r$pinned,
      notes = r$notes
    ),
    # Every warning this call raised, in raise order, one row each: tier
    # ("severe" / "caution" / "note"), kind (stable; match on this), title and
    # message. Texts are in documentation/warnings.md. Mirrors the Python port's
    # `Fit.warnings` - change together.
    warnings = .warnings_frame(store),
    n_eval = r$n_eval,
    # The gradient (BFGS) theta search, on the models the
    # gradient_search_fallback warning names; they describe that search even
    # when the derivative-free search finished the fit after it. n_grad counts
    # its gradient evaluations (n_eval does not); outer_stop is why it
    # stopped: "fdelta", "pgrad" or "noise_floor" (converged), "line_search",
    # "max_iter", "grad_unavailable" or "non_finite"; outer_decrement is its
    # estimate of the deviance decrease left at the endpoint. 0 /
    # NA_character_ / NaN on every fit that ran no gradient search.
    n_grad = r$n_grad,
    outer_stop = r$outer_stop %||% NA_character_,
    outer_decrement = r$outer_decrement,
    deviance = r$deviance,
    # logLik()/AIC()/BIC() inputs. `reml` says the `loglik` is a REML criterion
    # (a Gaussian mixed model fitted by REML, or an OLS fit made with
    # REML = TRUE) rather than an ML one; see logLik.fastglmm().
    loglik = r$loglik,
    df = as.integer(r$df),
    reml = r$reml,
    re_group_names = r$re_group_names,
    re_group_terms = r$re_group_terms,
    # Per grouping, the 1-based term of each name in `re_group_terms`;
    # print.VarCorr.fastglmm prints a correlation only within one term.
    re_term_index = r$re_term_index,
    # Labelled conditional modes straight from the kernel - ranef() reshapes
    # these into lme4's data.frame list. Nothing here slices the flat vector:
    # only the kernel knows which RE block layout the data routed to.
    ranef_blocks = r$ranef_blocks,
    # What predict() and simulate() read besides the public fields, private:
    # the flat conditional modes and their level counts, and the conditional
    # variances in the kernel's layout (glmm-bind's fit record). The
    # training columns are `frame`, marshalled again on each call. NULL on a
    # fit saved before them. Mirrors the Python port's `Fit._kernel_parts` -
    # change together.
    .kernel_parts = r[c("ranef", "ranef_levels", "condvar", "condvar_x", "condvar_cross")],
    # Each frame row's 1-based position in the caller's data, so an error of
    # predict() or simulate() on the frame's rows names the caller's row. NULL
    # when no row was dropped, and on a fit saved before it. Mirrors the
    # Python port's `Fit._kernel_parts["row_index"]` - change together.
    .row_index = if (identical(row_index, seq_len(nrow(frame)))) NULL else row_index,
    fitted = r$fitted,
    # The response as the kernel fitted it (after lowering: a cbind() LHS is
    # the proportion). residuals() reads this rather than re-resolving the
    # response column from the formula, which cbind() makes ambiguous.
    y = r$y,
    # The caller's prior weights (NULL when unweighted); the fit used them
    # raised to `weight_power`. From the kernel, not `weights=`: a cbind() LHS
    # lowers to trial-count weights the caller never passed, and Pearson
    # residuals need them.
    weights = r$weights,
    # The power the fit put on `weights` (row i has variance
    # sigma^2 * weights[i]^-weight_power): 1 on the weights as given, NaN
    # without weights. Pearson residuals, rss, r_squared and simulate() read
    # the weights raised to it; a fit saved before it reads 1. Mirrors the
    # Python port's Fit.weight_power - change together.
    weight_power = r$weight_power,
    # Whether the fit estimated weight_power rather than holding it fixed;
    # anova(m0, m1) and confint(method = "profile") read it. A fit saved before
    # it reads FALSE. Mirrors the Python port's Fit.estimated_weight_power. Not
    # named weight_power_*: `$` would partially match it for weight_power on a
    # fit saved without that field.
    estimated_weight_power = isTRUE(r$weight_power_estimated),
    # The standard error behind se, vcov and coef_df: "model" or "robust". A
    # fit saved before it reads "model".
    se_type = r$se_type,
    # On a robust fit, the model-based covariance of every retained
    # coefficient (p x p, NaN on an aliased one), which predict() and the Type
    # II hypotheses read there; NULL on a model fit, whose vcov is it, and on a
    # fit saved before it. Mirrors the Python port's Fit.model_vcov - change
    # together.
    model_vcov = if (length(r$model_vcov)) {
      matrix(as.double(unlist(r$model_vcov)), p, p, byrow = TRUE,
             dimnames = list(r$names, r$names))
    },
    # On a robust fit, its CR2 or HC2 pieces, which the tests of a robust fit
    # read back: list(er, f, epe, row_e, row_x, row_phi, null_cluster, null),
    # every matrix a flat column-major double vector over the p coefficients
    # (zero on an aliased one), cluster k the k-th level of the grouping factor
    # with rows, or row k without one: er (p * G, E_k r_k per cluster), f and
    # epe (lists of p * p, F_k and E_k Phi_k E_k' per cluster of more than one
    # row), row_e and row_x (p * n1, e_k and x_k per one-row cluster), row_phi
    # (n1, its scalar Phi_k), null_cluster (1-based cluster of each null
    # direction) and null (a list of p-vectors). NULL on a model fit and on a
    # fit saved before it. Mirrors the Python port's Fit.robust - change
    # together.
    robust = r$robust,
    # The design summary of a Gaussian fit whose only random term is one
    # random intercept (1 | g), with no prior weights and at least one
    # fixed-effect column; NULL on every other fit. list(counts, s, c), with
    # K = the number of groups that have rows (so K can be less than
    # nlevels(g)) and p retained fixed-effect columns, on the fit's
    # orthogonalized design X~ (it spans the column space of the fixed-effect
    # design): counts (K, rows per level), s (p * K, the
    # p x K matrix X~'Z column-major, so matrix(s, p, K)) and c (p * p, the
    # p x p matrix X~'X~, symmetric). Mirrors the Python port's
    # Fit.design_summary - change together.
    design_summary = r$design_summary,
    # The offset the kernel fitted with (offset= or an offset() term), NULL
    # without one: anova(m0, m1) checks that two fits share it. Mirrors the
    # Python port's Fit.offset - change together.
    offset = r$offset,
    # What the fit maximised, as the crate labels it ("least_squares",
    # "maximum_likelihood", "reml", "laplace", "agq") with its quadrature
    # node count (1 on "laplace", the count on "agq", 0 elsewhere).
    objective = r$objective,
    objective_nodes = r$objective_nodes,
    # The per-term tables anova(m) prints, built by the crate at fit time:
    # list(error = NULL, ss_type, test, reason, label, kind, statistic,
    # num_df, den_df, den_df_kind, p_value, sum_sq, mean_sq, eta2_partial,
    # row_notes, notes), one entry per term in each column, or
    # list(error = <message>) when the formula gets no table (an interaction
    # without its marginal terms). The likelihood-ratio tables
    # anova(m, test = "LRT") prints are the same lists, NULL unless the fit
    # asked with anova_test = "LRT"; under anova_test = "auto" they stay NULL
    # here and .lrt_tables makes them on the first anova() that needs them.
    # term_test_auto is list(test, reason): what anova(m)'s default
    # test = "auto" gives ("lrt" on a Poisson model with random effects).
    term_table_type3 = r$term_table_type3,
    term_table_type2 = r$term_table_type2,
    term_table_type3_lrt = r$term_table_type3_lrt,
    term_table_type2_lrt = r$term_table_type2_lrt,
    term_test_auto = r$term_test_auto,
    call = call,
    formula = f_str,
    family = fam$object,
    # Port-vocabulary family name ("gamma", not R's "Gamma") - what the
    # methods dispatch on; `family` above is the R-facing object.
    family_name = fam$name,
    # The port-vocabulary link ("inverse_squared", not R's "1/mu^2").
    link = fam$link,
    frame = frame,
    nobs = nrow(frame),
    data_name = data_name,
    # Effective node count: the shim strips ineligible nAGQ>1 to Laplace
    # (with the warning surfaced above), so record what actually ran.
    nAGQ = if (is.null(r$agq_warning)) nAGQ else 1L
  ), class = "fastglmm")
}

# `data` as a data.frame, refusing a list whose columns among `vars` have
# different lengths.
.as_data_frame <- function(data, vars) {
  if (is.data.frame(data)) return(data)
  # as.data.frame() recycles a short list column whose length divides the
  # longest one; model.frame() (glm, lme4) refuses that, and so does this.
  # Same message as run_fit's length check in src/orchestrate.rs - change
  # together.
  if (is.list(data)) {
    len <- vapply(data[intersect(vars, names(data))], NROW, integer(1))
    if (length(unique(len)) > 1L) {
      len <- len[order(names(len), method = "radix")]
      stop(sprintf(
        "columns have mismatched lengths (%s); all columns must have the same length",
        paste(sprintf('"%s": %d', names(len), len), collapse = ", ")
      ), call. = FALSE)
    }
  }
  as.data.frame(data)
}

# The columns `vars` of `data`, every one of them present.
.data_columns <- function(data, vars) {
  missing_cols <- setdiff(vars, names(data))
  if (length(missing_cols)) {
    stop("column(s) not found in data: ", paste(missing_cols, collapse = ", "),
         call. = FALSE)
  }
  # Builds what `[.data.frame` returns, without its dispatch overhead. Only for
  # a plain data.frame: a subclass (tibble, data.table, sf) has its own `[`,
  # and model.frame(fit) keeps whatever class that returns.
  if (identical(oldClass(data), "data.frame")) {
    return(structure(.subset(data, vars), names = vars,
                     row.names = .row_names_info(data, 0L), class = "data.frame"))
  }
  data[vars]
}

# Whether na.omit(frame) would return `frame` unchanged (identical()), so
# the call can be skipped: no missing value anywhere, and every column passes
# through `[` untouched - no attributes, or a plain factor's levels + class.
# `[` drops other attributes, such as a "label".
.na_omit_is_identity <- function(frame) {
  for (x in frame) {
    # anyNA() on a classed vector runs any(is.na(x)), allocating a logical
    # vector; a factor's missing values are its NA codes, so the codes are
    # scanned directly.
    if (anyNA(if (is.factor(x)) unclass(x) else x)) return(FALSE)
    a <- attributes(x)
    if (!is.null(a) && !identical(names(a), c("levels", "class"))) return(FALSE)
  }
  TRUE
}

# `frame` with the evaluated weights= argument `w` parked as "(weights)", or
# unchanged when `w` is NULL; `n` is the row count of the data. With
# `missing_ok` (compare()), an NA entry is kept for the row mask to drop.
.add_weights <- function(frame, w, n, missing_ok = FALSE) {
  if (is.null(w)) return(frame)
  if (!is.numeric(w) || length(w) != n) {
    stop("weights must be a numeric array with one entry per row of data",
         call. = FALSE)
  }
  if (!all(is.finite(if (missing_ok) w[!is.na(w)] else w))) {
    # Mirrors the Python port's wording - change together.
    stop(if (missing_ok) "weights must be finite or missing: no inf entries"
         else "weights must be finite: no NaN, inf, or missing entries", call. = FALSE)
  }
  if (any(w <= 0, na.rm = TRUE)) {
    # The kernel requires strictly positive weights; a zero weight is a row
    # that should not be in the fit at all.
    stop("weights must be positive; drop zero-weight rows from data instead",
         call. = FALSE)
  }
  frame[["(weights)"]] <- as.double(w)
  frame
}

# `frame` with the evaluated offset= argument `o` parked as "(offset)". An
# offset is normally written as an expression over the data
# (offset = log(exposure)), so it is evaluated like weights= and parked in
# the frame: subset= and na.action must drop its entries in lockstep with the
# rows, and reading it off `data` after the filtering would misalign it
# against a subset fit with no error anywhere. `missing_ok` as for weights.
.add_offset <- function(frame, o, n, missing_ok = FALSE) {
  if (is.null(o)) return(frame)
  if (!is.numeric(o) || length(o) != n) {
    stop("offset must be a numeric array with one entry per row of data",
         call. = FALSE)
  }
  if (!all(is.finite(if (missing_ok) o[!is.na(o)] else o))) {
    # Mirrors the Python port's wording - change together.
    stop(if (missing_ok) "offset must be finite or missing: no inf entries"
         else "offset must be finite: no NaN, inf, or missing entries", call. = FALSE)
  }
  frame[["(offset)"]] <- as.double(o)
  frame
}

# The columns `vars` of `frame` marshalled for the kernel: list(numeric =,
# factors =). Factors cross as R factors, levels and all, so the caller's
# declared level order (the treatment base) survives into Rust's
# Column::Factor - the declared-order path.
# A logical predictor or grouping column is a factor, as lme4 builds it
# (model.matrix() and mkReTrms() both go through factor()): levels
# "FALSE"/"TRUE", a dummy named xTRUE, and indicator columns (xFALSE:z,
# xTRUE:z) wherever R's marginality rule codes it without a contrast. The
# response and offset() variables (`numeric_only`) stay 0/1 numerics, as
# glm() reads them.
.marshal <- function(frame, vars, numeric_only) {
  numeric_cols <- list()
  factor_cols <- list()
  for (nm in vars) {
    col <- .subset2(frame, nm)
    if (is.character(col) || (is.logical(col) && !(nm %in% numeric_only))) {
      col <- factor(col) # lexicographic (FALSE < TRUE), as factor() does
    }
    if (is.factor(col)) {
      factor_cols[[nm]] <- col
    } else if (is.numeric(col) || is.logical(col)) {
      numeric_cols[[nm]] <- as.double(col)
    } else {
      stop("column '", nm, "' has unsupported type ", class(col)[1L],
           "; pass numeric, logical, factor, or character columns",
           call. = FALSE)
    }
  }
  list(numeric = numeric_cols, factors = factor_cols)
}

# fastglmm()'s start= as the native fit's warm_start: NULL when it gives
# nothing, so the fit starts cold.
.warm_start <- function(start) {
  warm <- Filter(length, list(beta = as.double(start$beta), theta = as.double(start$theta),
                              dispersion = as.double(start$dispersion)))
  if (length(warm)) warm
}

# The kinds the fit record's notes and `failure` carry that have a class of
# their own, "fastglmm_<kind>"; any other kind is "fastglmm_unknown_note".
# Every class also inherits "fastglmm_diagnostic", so one handler catches the
# whole channel. Mirrors the Python port's _CATEGORIES (glmm-py/glmm/__init__.py)
# - change together.
.NOTE_KINDS <- c(
  "search_limit", "fit_failed", "glm_diverged", "design_unsolvable", "constant_response",
  "too_few_rows", "no_coefficients", "ill_conditioned", "pirls_exhausted",
  "nb_shape_unsettled", "single_level_grouping_dropped", "few_rows_for_random_effects",
  "repeated_grouping", "slope_constant_within_levels", "separated_by_grouping",
  "variance_at_limit", "hessian_se_fallback", "se_unavailable", "gradient_search_fallback",
  "exact_profile_fallback", "precision_limited", "non_integer_response",
  "satterthwaite_curvature", "weight_power_at_bound", "weight_power_search_failures",
  "weight_power_outside_unit", "weight_power_implausible", "robust_with_random_slope",
  "robust_se_unidentified", "robust_df_small"
)

.note_class <- function(kind) {
  c(if (kind %in% .NOTE_KINDS) paste0("fastglmm_", kind) else "fastglmm_unknown_note",
    "fastglmm_diagnostic")
}

# The families nAGQ > 1 covers, as the fallback message names them. Mirrors the
# kernel's AGQ eligibility check (src/orchestrate.rs, the `if nagq > 1` block)
# and the Python port's _AGQ_FAMILIES - change together.
.AGQ_FAMILIES <- "binomial, Poisson, negative-binomial or Gamma"

# Raise one classed warning this port words itself, under the tier and title
# of `kind`; see .warn_entry.
.warn_keep <- function(store, kind, msg, cls) {
  tt <- .glmm_call("warning_kind", list(kind = kind))
  .warn_entry(store, list(tier = tt$tier, kind = kind, title = tt$title, message = msg), cls)
}

# Raise `entry` (its tier, kind, title and message) as a classed warning
# "<Tier>: <title>. <message>" and append those four fields to `store` (an
# environment fastglmm() creates), which becomes `$warnings`.
.warn_entry <- function(store, entry, cls) {
  entry <- entry[c("tier", "kind", "title", "message")]
  store$rows[[length(store$rows) + 1L]] <- entry
  printed <- sprintf("%s%s: %s. %s", toupper(substr(entry$tier, 1L, 1L)),
                     substring(entry$tier, 2L), entry$title, entry$message)
  warning(warningCondition(printed, class = cls, call = NULL))
}

# Every warning this call raised, in raise order, one row each. Zero rows on a
# clean fit. Mirrors the Python port's `Fit.warnings` - change together.
.warnings_frame <- function(store) {
  col <- function(name) vapply(store$rows, `[[`, character(1), name)
  list2DF(list(tier = col("tier"), kind = col("kind"), title = col("title"),
               message = col("message")))
}

# Names of the RE components the optimizer pinned at the boundary, as
# "<term> in <group>", for the singular warning.
#
# Read straight off the kernel's own record of what it pinned. Do NOT
# reconstruct it from varcorr: on a grouping with q >= 2 the pin fixes the
# DIAGONAL of the Cholesky factor while the reported stddev is
# sqrt(lambda_offdiag^2 + lambda_diag^2), which lands at ~1e-9 rather than at 0,
# so a scan for exactly-zero stddevs misses those pins entirely.
#
# character(0) means nothing was pinned - including a model with no variance
# components to pin. `singular` can still be TRUE with `pinned` empty (the
# post-hoc negligible-stddev check is independent of the optimizer's own pin
# decision), so the bare warning text stands regardless - empty `pinned` is
# never evidence that the fit is not singular.
# Mirrors glmm-py/glmm/__init__.py::_pinned_detail - change together.
.pinned_detail <- function(pinned, group_names, group_terms) {
  parts <- character()
  for (g in seq_along(pinned)) {
    terms <- group_terms[[g]]
    grp <- group_names[[g]]
    for (i in which(as.logical(pinned[[g]]))) {
      term <- if (i <= length(terms)) terms[[i]] else sprintf("component %d", i)
      parts <- c(parts, sprintf("%s in %s", term, grp))
    }
  }
  parts
}

.count <- function(n, word) sprintf("%d %s%s", as.integer(n), word, if (n == 1) "" else "s")

# `...` exists only to intercept known lme4 arguments with designed errors
# (spec section 1/section 4) - an unknown argument must never be silently swallowed
# (Decision 5: error, never silently differ).
.check_dots <- function(...) {
  dots <- list(...)
  if (!length(dots)) return(invisible())
  nms <- names(dots) %||% rep("", length(dots))
  for (nm in nms) {
    switch(nm,
      control = stop("control= is not supported: the optimizer (BOBYQA) ",
                     "settings are compiled into the glmm kernel; ",
                     "accepting-and-ignoring them is not acceptable",
                     call. = FALSE),
      verbose = stop("verbose= is not supported: the glmm kernel has no ",
                     "progress reporting hook", call. = FALSE),
      contrasts = stop("contrasts= is not supported: the shared formula ",
                       "parser is treatment-coded with base = first level ",
                       "and offers no hook; relevel() the factor to change ",
                       "the base level", call. = FALSE),
      stop("unused argument", if (nzchar(nm)) paste0(" '", nm, "'") else "",
           " - fastglmm() intercepts rather than swallows unknown arguments",
           call. = FALSE)
    )
  }
  invisible()
}

# nAGQ as an integer, or an error.
.check_nagq <- function(nAGQ) {
  if (!(is.numeric(nAGQ) && length(nAGQ) == 1L && !is.na(nAGQ) &&
        nAGQ == as.integer(nAGQ) && nAGQ >= 1L && nAGQ <= .MAX_NAGQ &&
        nAGQ %% 2L == 1L)) {
    stop(sprintf("nAGQ must be an odd integer in 1..=%d, got %s", .MAX_NAGQ, .arg_repr(nAGQ)),
         call. = FALSE)
  }
  as.integer(nAGQ)
}

# lme4's `REML =`: only the type is checked here. What each value means on
# the model actually fitted (after lowering, so a formula whose every grouping
# was dropped counts as OLS) is decided once, by the shared rule in run_fit
# (src/orchestrate.rs), for both ports.
.check_reml <- function(reml) {
  if (is.null(reml)) return(invisible())
  if (!(is.logical(reml) && length(reml) == 1L && !is.na(reml))) {
    stop(sprintf("REML must be NULL, TRUE or FALSE, got %s", .arg_repr(reml)),
         call. = FALSE)
  }
  invisible()
}

# Anything not caught here falls through to the shared Rust parser, whose own
# message is surfaced verbatim.
.check_formula <- function(formula) {
  # Catches a formula shape the parser has no dedicated message for (it falls
  # through to a generic syntax error), with the same wording the Python port
  # uses (glmm-py/glmm/__init__.py's `_check_call`) - change together.
  if ("." %in% all.vars(formula)) {
    stop("'.' is not supported by the formula parser; list the columns explicitly",
         call. = FALSE)
  }
  if (length(formula) != 3L) {
    # Same text as `materialize`'s no-response error in
    # glmm-rs/src/formula/materialize.rs - change together.
    stop("formula syntax error at position 0: the formula has no response; write it as y ~ x",
         call. = FALSE)
  }
  invisible()
}

# Drops every "(... | ...)" random-effect term from a formula's right-hand
# side, working on the parsed call tree rather than the deparsed string, so
# it survives nested +/- and parenthesized terms the same way the shared
# parser will see them. What is left is the fixed-effects part alone.
# Independent, minimal reimplementation of lme4::nobars: fastglmm must not
# depend on lme4.
.strip_bars <- function(term) {
  if (!is.call(term)) return(term)
  op <- as.character(term[[1L]])
  if (op == "|") return(NULL)
  if (op == "(" && length(term) == 2L) {
    inner <- .strip_bars(term[[2L]])
    if (is.null(inner)) return(NULL)
    term[[2L]] <- inner
    return(term)
  }
  if (op %in% c("+", "-") && length(term) == 3L) {
    l <- .strip_bars(term[[2L]])
    r <- .strip_bars(term[[3L]])
    if (is.null(l)) return(r)
    if (is.null(r)) return(l)
    term[[2L]] <- l
    term[[3L]] <- r
    return(term)
  }
  term
}

# The variables read inside offset() calls anywhere in `term`.
.offset_vars <- function(term) {
  if (!is.call(term)) return(character())
  if (identical(term[[1L]], as.name("offset"))) return(all.vars(term))
  unlist(lapply(as.list(term)[-1L], .offset_vars))
}

# Refuses a formula whose fixed-effects factors would fit a different model
# from what the coefficients say, mirroring the explicit contrasts= error
# above: the shared parser is always treatment-coded, base = first level, and
# has no hook to honor anything else. Character and logical columns go
# through model.matrix() as unordered factors too (R >= 4.0's string default,
# and 0/1 coding for logical), so they are checked the same way. Only the
# UNORDERED slot of options(contrasts) matters here - model.matrix() only
# reaches the ordered slot through an ordered factor, which is refused
# outright regardless of that option. Random-effect grouping and slope
# variables are out of scope on purpose - only fixed-effects factors are
# contrast-coded here.
.check_contrasts <- function(formula, frame) {
  fixed_rhs <- .strip_bars(formula[[3L]])
  if (is.null(fixed_rhs)) return(invisible())
  has_unordered_factor <- FALSE
  for (nm in all.vars(fixed_rhs)) {
    col <- .subset2(frame, nm)
    if (is.null(col)) next
    if (is.ordered(col)) {
      stop("column '", nm, "' is an ordered factor: glmm codes factors ",
           "with treatment contrasts; convert with ",
           "factor(x, ordered = FALSE)", call. = FALSE)
    }
    if (!(is.factor(col) || is.character(col) || is.logical(col))) next
    has_unordered_factor <- TRUE
    if (!is.null(attr(col, "contrasts"))) {
      stop("column '", nm, "' has a contrasts attribute set (contrasts(x) ",
           "<- ...): the shared formula parser is treatment-coded with ",
           "base = first level and offers no hook; remove the attribute or ",
           "relevel() the factor to change the base level", call. = FALSE)
    }
  }
  if (has_unordered_factor &&
      !identical(getOption("contrasts")[[1L]], "contr.treatment")) {
    stop("options(contrasts = ) sets the unordered-factor contrast away from ",
         "the default \"contr.treatment\": the shared formula parser is ",
         "always treatment-coded and offers no hook; reset ",
         "options(contrasts = c(\"contr.treatment\", ...)) or relevel() the ",
         "factor to change the base level", call. = FALSE)
  }
  invisible()
}

# family (object | function | string) -> list(name=, link=, object=) in the
# port vocabulary. A family OBJECT is honored as given - R semantics win -
# which is the documented Gamma() link trap: Gamma() means link "inverse",
# the string "gamma" means the glmm default "log".
.normalize_family <- function(family) {
  if (is.function(family)) family <- family()
  if (inherits(family, "family")) {
    rname <- family$family
    if (grepl("^Negative Binomial", rname)) {
      stop("MASS::negative.binomial(theta) fixes the shape; fastglmm ",
           "estimates it - use family = \"negativebinomial\" (init.theta= ",
           "seeds the shape search)", call. = FALSE)
    }
    name <- switch(rname,
      gaussian = "gaussian",
      binomial = "binomial",
      poisson = "poisson",
      Gamma = "gamma",
      inverse.gaussian = "inversegaussian",
      stop("unsupported family '", rname, "'; expected one of gaussian, ",
           "binomial, poisson, Gamma, inverse.gaussian, \"negativebinomial\"",
           call. = FALSE)
    )
    link <- switch(family$link,
      identity = "identity", logit = "logit", probit = "probit",
      cloglog = "cloglog", log = "log", inverse = "inverse",
      `1/mu^2` = "inverse_squared",
      stop("unsupported link '", family$link, "' for family '", rname, "'",
           call. = FALSE)
    )
    object <- family
  } else if (is.character(family) && length(family) == 1L) {
    name <- switch(tolower(family),
      gaussian = "gaussian", binomial = "binomial", poisson = "poisson",
      gamma = "gamma",
      negativebinomial = , `negative.binomial` = "negativebinomial",
      inversegaussian = , `inverse.gaussian` = "inversegaussian",
      stop("unknown family '", family, "'; expected one of ",
           paste(names(.FAMILIES), collapse = ", "), call. = FALSE)
    )
    link <- .FAMILIES[[name]]$default_link
    object <- switch(name,
      gaussian = stats::gaussian(),
      binomial = stats::binomial(),
      poisson = stats::poisson(),
      gamma = stats::Gamma(link = "log"), # glmm's gamma default, NOT R's
      # No R constructor exists for these; a minimal family-shaped record
      # keeps family(fit) printable.
      structure(list(family = name, link = link), class = "family")
    )
  } else {
    stop("`family` must be a family object, family function, or string",
         call. = FALSE)
  }
  spec <- .FAMILIES[[name]]
  if (!(link %in% spec$links)) {
    stop("family '", name, "' does not support link '", link, "'; expected ",
         "one of ", paste(spec$links, collapse = ", "), call. = FALSE)
  }
  list(name = name, link = link, object = object)
}
