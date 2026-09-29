# fastglmm() - the package's one entry point. The R side does row filtering
# (subset/na.action), family normalization, argument validation, and column
# marshalling; the formula is parsed and lowered by the Rust side
# (glmm::formula - one parser shared with the Python port, spec section 3), so this
# file deliberately contains no model.matrix / terms machinery.

`%||%` <- function(x, y) if (is.null(x)) y else x

# A bad argument value, formatted for an error message the same way the
# Python port's `{value!r}` does: a single string quoted, anything else
# deparsed. Mirrors python/glmm/__init__.py's use of `!r` in its argument
# checks - change together.
.arg_repr <- function(x) {
  if (is.character(x) && length(x) == 1L) sprintf("'%s'", x) else deparse(x)[1]
}

# Family/link table - mirrors python/glmm/__init__.py::_FAMILIES and
# GLMM/src/family.rs; change together. Links are the port vocabulary, not R's
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
# python/glmm/__init__.py::_DISPERSION_FAMILIES; change together.
.DISPERSION_FAMILIES <- c("binomial", "poisson", "gamma", "inversegaussian")

# Mirrors GLMM/src/consts.rs::MAX_NAGQ - change together.
.MAX_NAGQ <- 25L

#' Fit a (generalized) linear or linear mixed model with the glmm Rust kernel
#'
#' One entry point for the whole `glmm` engine, dispatching on `family` the way
#' the kernel itself does: OLS (`gaussian` without random effects), GLM
#' (binomial/poisson/gamma/negative-binomial without random effects), REML LMM
#' (`gaussian` with random effects), and GLMM (Laplace or adaptive
#' Gauss-Hermite) - there is no `lmer`/`glmer` split.
#'
#' The formula is parsed by the same Rust parser the Python port uses. Besides
#' bare column names, it accepts `log(x)`, `sqrt(x)`, `exp(x)`, and `I(x^k)`
#' (`k >= 2`) of a single column; `cbind(successes, failures)` on the
#' left-hand side with `family = binomial`; an `offset()` term; and term
#' removal (`- 1` / `0 +`). Not accepted: `poly()`, arithmetic inside a call,
#' nested calls, and `.` - compute those columns first and pass them by name.
#' `weights=` together with a `cbind()` formula, and `offset=` together with
#' an `offset()` formula term, both error naming "use one". Fixed and random
#' effects support `+`, `:`, `*`, `A/B` nesting, and `(1 + x | g)` random-effect
#' terms with a **full** correlation structure - `(x || g)` and intercept-free
#' RE terms are not fittable by the kernel and raise an error. Contrasts are
#' always treatment coding with the **first factor level** as base; to change
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
#' `"fastglmm_unused_grouping_levels"`, `"fastglmm_re_design_scale_spread"`,
#' `"fastglmm_single_level_grouping_dropped"`, `"fastglmm_hessian_se_fallback"`,
#' `"fastglmm_exact_profile_fallback"`,
#' `"fastglmm_non_integer_response"`, `"fastglmm_rows_dropped_na"`; a note from
#' a newer kernel than this package arrives as `"fastglmm_unknown_note"`),
#' so `withCallingHandlers()` and `suppressWarnings(classes = )` can select
#' them without matching message text.
#'
#' Anything the engine cannot do is an error naming the reason - never a
#' silently different model. That includes `REML = FALSE` (the LMM path is
#' REML-only **by design**), `control=`/`verbose=`, and quasi-likelihood
#' `dispersion=` on binomial/poisson.
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
#'   `list(beta =, theta =)` with `theta` the random-effect Cholesky vector.
#'   Distinct from `init.theta`, which is the negative-binomial shape.
#' @param wald.se Wald standard-error mode: `"hessian"` (default) or `"rx"`.
#' @param dispersion Gamma/inverse-Gaussian dispersion directive: `NULL`
#'   (estimate it, the default: the Pearson moment on a Gamma GLM, maximum
#'   likelihood on a Gamma GLMM, the Pearson moment on inverse-Gaussian),
#'   `"estimate"` (same), or a number to
#'   hold it fixed. Non-`NULL` on binomial/poisson would mean
#'   quasi-likelihood - not implemented, errors.
#' @param init.theta negative-binomial shape seed, named for
#'   `MASS::glm.nb(init.theta=)`. No kernel hook exists yet to seed the shape
#'   search, so any non-`NULL` value is an error (the default cold start is
#'   what runs).
#' @param ... intercepted, never silently swallowed: known lme4 arguments
#'   (`REML`, `control`, `verbose`, `contrasts`) raise errors saying why they
#'   cannot be honored; unknown names error as unused arguments.
#'
#' @return An object of class `"fastglmm"`: fixed effects ([fixef]), Wald
#'   covariance (`vcov()`), variance components on the SD/correlation scale
#'   ([VarCorr]), `converged` and `singular` flags ([isSingular]), a
#'   `diagnostics` list (the solver's own report: `boundary`, `pinned`,
#'   `notes`, plus the three flags above), a `warnings` data.frame (one row
#'   per warning the call raised: `tier`, `kind`, `title`, `message`; match on
#'   `kind`), plus
#'   `print()`, [summary()][summary.fastglmm], [confint()][confint.fastglmm]
#'   (Wald), `nobs()`, [formula()][formula.fastglmm] (returns the formula
#'   **string**), `family()`, and `model.frame()`. Engine-blocked accessors
#'   (`ranef`, `predict`, `fitted`, `residuals`, `coef`, `logLik`/`AIC`,
#'   `terms`) error with the reason.
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
                     ...) {
  .check_dots(...)
  # Captured before `data` is touched: the header's `Data:` line, as lme4
  # deparses it.
  data_name <- paste(deparse(substitute(data), width.cutoff = 500L), collapse = " ")

  if (is.character(formula)) formula <- stats::as.formula(formula)
  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula or a formula string", call. = FALSE)
  }
  f_str <- paste(deparse(formula, width.cutoff = 500L), collapse = " ")
  .check_formula(formula, f_str)

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

  if (!(is.numeric(nAGQ) && length(nAGQ) == 1L && !is.na(nAGQ) &&
        nAGQ == as.integer(nAGQ) && nAGQ >= 1L && nAGQ <= .MAX_NAGQ &&
        nAGQ %% 2L == 1L)) {
    stop(sprintf("nAGQ must be an odd integer in 1..=%d, got %s", .MAX_NAGQ, .arg_repr(nAGQ)),
         call. = FALSE)
  }
  nAGQ <- as.integer(nAGQ)

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
    # mirrors python/glmm/__init__.py's NotImplementedError block - change together.
    stop("quasi-likelihood dispersion on family '", fam$name,
         "' is not yet implemented in the kernel",
         call. = FALSE)
  }
  if (!is.null(init.theta) && fam$name != "negativebinomial") {
    .warn_keep(store, "argument_ignored",
               sprintf("init.theta= is not used for family '%s'.", fam$name), ignored)
    init.theta <- NULL
  }
  if (!is.null(init.theta)) {
    stop("init.theta= (negative-binomial shape seed) has no kernel hook yet; ",
         "only the default cold-start search is supported", call. = FALSE)
  }

  if (!is.null(start)) {
    if (!is.list(start)) {
      stop(sprintf(paste(
        "start must be a list with elements 'beta' and/or 'theta' (theta is the",
        "random-effect Cholesky vector, not the negative-binomial shape - that is",
        "init.theta), got %s"), class(start)[1L]), call. = FALSE)
    }
    unknown <- setdiff(names(start), c("beta", "theta"))
    if (length(unknown)) {
      .warn_keep(store, "argument_ignored", paste0(
        "start accepts only 'beta' and 'theta'; these entries were ignored: ",
        paste(unknown, collapse = ", "), "."), ignored)
    }
  }

  # --- data prep: row filtering (subset, na.action) happens HERE, on the
  # data.frame, before marshalling - data prep, not parsing (spec section 3.1). ---
  if (!is.data.frame(data)) data <- as.data.frame(data)
  vars <- all.vars(formula)
  missing_cols <- setdiff(vars, names(data))
  if (length(missing_cols)) {
    stop("column(s) not found in data: ", paste(missing_cols, collapse = ", "),
         call. = FALSE)
  }
  frame <- data[vars]
  .check_contrasts(formula, frame)

  w <- eval(substitute(weights), data, parent.frame())
  if (!is.null(w)) {
    if (!is.numeric(w) || length(w) != nrow(data) || anyNA(w)) {
      stop("weights must be a numeric array with one entry per row of data",
           call. = FALSE)
    }
    if (!all(is.finite(w))) {
      stop("weights must be finite: no NaN, inf, or missing entries", call. = FALSE)
    }
    if (any(w <= 0)) {
      # The kernel requires strictly positive weights; a zero weight is a row
      # that should not be in the fit at all.
      stop("weights must be positive; drop zero-weight rows from data instead",
           call. = FALSE)
    }
    frame[["(weights)"]] <- as.double(w)
  }

  # An offset is normally written as an expression over the data
  # (offset = log(exposure)), so it is evaluated like weights= and parked in
  # the frame: subset= and na.action must drop its entries in lockstep with
  # the rows, and reading it off `data` after the filtering would misalign it
  # against a subset fit with no error anywhere.
  o <- eval(substitute(offset), data, parent.frame())
  if (!is.null(o)) {
    if (!is.numeric(o) || length(o) != nrow(data) || anyNA(o)) {
      stop("offset must be a numeric array with one entry per row of data",
           call. = FALSE)
    }
    if (!all(is.finite(o))) {
      stop("offset must be finite: no NaN, inf, or missing entries", call. = FALSE)
    }
    frame[["(offset)"]] <- as.double(o)
  }

  s <- eval(substitute(subset), data, parent.frame())
  if (!is.null(s)) frame <- frame[s, , drop = FALSE]

  naf <- if (is.character(na.action)) {
    get(na.action, mode = "function")
  } else {
    na.action
  }
  if (!is.function(naf)) stop("invalid na.action", call. = FALSE)
  n_before_na <- nrow(frame)
  frame <- naf(frame)
  if (anyNA(frame[vars])) {
    stop("missing values remain in the model columns after na.action; ",
         "the kernel cannot fit NA - use na.action = na.omit or complete ",
         "the data", call. = FALSE)
  }
  if (nrow(frame) == 0L) {
    # Mirrors the Python port's fit(), which raises the same text when its own
    # missing-value row drop empties the data - change together
    # (python/glmm/__init__.py, the na_mask block). A subset= or an
    # already-empty data.frame reaching this point instead (n_before_na == 0)
    # has no Python counterpart, so it keeps the plain message.
    if (n_before_na > 0L) {
      stop("every row has a missing value in a column the formula uses; no rows left to fit",
           call. = FALSE)
    }
    stop("no rows left to fit", call. = FALSE)
  }
  if (nrow(frame) < n_before_na) {
    # Mirrors the Python port's fit(), which does its own row-drop bookkeeping
    # since it has no model.frame/na.action to do this for free - change
    # together (python/glmm/__init__.py, the na_mask block).
    .warn_keep(store, "rows_dropped_na", sprintf(
      "Dropped %d of %d row(s): a column the formula uses had a missing value there.",
      n_before_na - nrow(frame), n_before_na),
      c("fastglmm_rows_dropped_na", "fastglmm_diagnostic"))
  }
  w <- frame[["(weights)"]]
  o <- frame[["(offset)"]]

  # --- marshalling: factors cross as (levels, 0-based codes) so the caller's
  # declared level order (the treatment base) survives into Rust's
  # Column::Factor - the declared-order path. ---
  # A logical predictor or grouping column is a factor, as lme4 builds it
  # (model.matrix() and mkReTrms() both go through factor()): levels
  # "FALSE"/"TRUE", a dummy named xTRUE, and indicator columns (xFALSE:z,
  # xTRUE:z) wherever R's marginality rule codes it without a contrast. The
  # response and offset() stay 0/1 numerics, as glm() reads them.
  numeric_only <- c(all.vars(formula[[2L]]), .offset_vars(formula))
  numeric_cols <- list()
  factor_levels <- list()
  factor_codes <- list()
  for (nm in vars) {
    col <- frame[[nm]]
    if (is.character(col) || (is.logical(col) && !(nm %in% numeric_only))) {
      col <- factor(col) # lexicographic (FALSE < TRUE), as factor() does
    }
    if (is.factor(col)) {
      factor_levels[[nm]] <- as.character(levels(col))
      factor_codes[[nm]] <- as.integer(col) - 1L
    } else if (is.numeric(col) || is.logical(col)) {
      numeric_cols[[nm]] <- as.double(col)
    } else {
      stop("column '", nm, "' has unsupported type ", class(col)[1L],
           "; pass numeric, logical, factor, or character columns",
           call. = FALSE)
    }
  }

  r <- fastglmm_fit(
    f_str, numeric_cols, factor_levels, factor_codes,
    fam$name, fam$link, wald.se, nAGQ,
    if (is.null(dispersion)) double() else as.double(dispersion),
    if (is.null(w)) double() else as.double(w),
    if (is.null(o)) double() else as.double(o),
    as.double(start$beta %||% double()),
    as.double(start$theta %||% double())
  )

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
  for (note in r$notes) {
    out <- .note_warning(note, r$names, r$converged)
    if (!is.null(out)) .warn_keep(store, note$kind, out$msg, out$cls)
  }
  if (!isTRUE(r$converged)) {
    out <- .nonconvergence(fam$name, fitted_mixed, r$notes, r$beta, r$aliased, r$deviance, r$y)
    .warn_keep(store, out$kind, out$msg, out$cls)
  }

  p <- length(r$beta)
  beta <- stats::setNames(r$beta, r$names)
  se <- stats::setNames(r$se, r$names)
  aliased <- stats::setNames(r$aliased, r$names)
  # Aliased (rank-deficient) slots print as NA, lme4/lm-style, not NaN.
  beta[aliased] <- NA_real_
  se[aliased] <- NA_real_
  vc <- matrix(r$vcov, p, p, byrow = TRUE, dimnames = list(r$names, r$names))

  structure(list(
    beta = beta,
    se = se,
    vcov = vc,
    varcorr = r$varcorr,
    stddev_se = r$stddev_se,
    aliased = aliased,
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
    #   boundary  "interior" / "at_boundary" / "no_optimum": where the
    #             accepted theta sits. Every theta-carrying route
    #             distinguishes all three (LMM over either kernel, GLMM over
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
    #   notes     list of list(kind=, columns=, pivot=, evals=, final_eval=,
    #             detail=, ratio=); `columns` is
    #             1-based into `names`. Each is raised as a classed warning by
    #             the call above. An absent note means "not detected", never
    #             "checked and clean": the GLMM routes record no pivot, and
    #             the LMM routes flag `IllConditioned` rather than refuse.
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
    deviance = r$deviance,
    # logLik()/AIC()/BIC() inputs. `reml` marks the LMM paths, whose `loglik`
    # is a REML criterion rather than an ML one - see logLik.fastglmm().
    loglik = r$loglik,
    df = as.integer(r$df),
    reml = r$reml,
    re_group_names = r$re_group_names,
    re_group_terms = r$re_group_terms,
    # Labelled conditional modes straight from the kernel - ranef() reshapes
    # these into lme4's data.frame list. Nothing here slices the flat vector:
    # only the kernel knows which RE block layout the data routed to.
    ranef_blocks = r$ranef_blocks,
    fitted = r$fitted,
    # The response as the kernel fitted it (after lowering: a cbind() LHS is
    # the proportion). residuals() reads this rather than re-resolving the
    # response column from the formula, which cbind() makes ambiguous.
    y = r$y,
    # Prior weights the kernel fitted with (NULL when unweighted). From the
    # kernel, not `weights=`: a cbind() LHS lowers to trial-count weights the
    # caller never passed, and Pearson residuals need them.
    weights = r$weights,
    call = match.call(),
    formula = f_str,
    family = fam$object,
    # Port-vocabulary family name ("gamma", not R's "Gamma") - what the
    # methods dispatch on; `family` above is the R-facing object.
    family_name = fam$name,
    frame = frame,
    nobs = nrow(frame),
    data_name = data_name,
    # Effective node count: the shim strips ineligible nAGQ>1 to Laplace
    # (with the warning surfaced above), so record what actually ran.
    nAGQ = if (is.null(r$agq_warning)) nAGQ else 1L
  ), class = "fastglmm")
}

# kind -> c(tier, title), one fixed pair per kind. Mirrors the Python port's
# _WARNING_KINDS (python/glmm/__init__.py) and documentation/warnings.md, which
# holds the message text - change all three together.
.WARNING_KINDS <- list(
  search_limit = c("severe", "Search stopped at its step limit"),
  fit_failed = c("severe", "Fit failed"),
  glm_diverged = c("severe", "Fit diverged"),
  design_unsolvable = c("severe", "Predictors could not be separated"),
  constant_response = c("severe", "Response does not vary"),
  too_few_rows = c("severe", "Too few rows"),
  no_coefficients = c("severe", "Nothing to estimate"),
  pirls_exhausted = c("caution", "Last fitting step did not finish"),
  nb_shape_unsettled = c("caution", "Shape search did not settle"),
  singular = c("caution", "Singular fit"),
  ill_conditioned = c("caution", "Nearly collinear columns"),
  hessian_se_fallback = c("caution", "Simpler standard errors used"),
  exact_profile_fallback = c("caution", "Search retried with a different method"),
  agq_fallback = c("caution", "Adaptive quadrature not used"),
  non_integer_response = c("caution", "Non-integer response"),
  rows_dropped_na = c("caution", "Rows dropped for missing values"),
  argument_ignored = c("note", "Argument ignored"),
  unused_grouping_levels = c("note", "Unused grouping levels"),
  re_design_scale_spread = c("note", "Random-effect predictors on very different scales"),
  single_level_grouping_dropped = c("note", "Random effect dropped (single level)")
)
# A kernel note this wrapper has no entry for keeps its own kind string.
.UNKNOWN_KIND <- c("caution", "Unrecognized solver message")

# The families nAGQ > 1 covers, as the fallback message names them. Mirrors the
# kernel's AGQ eligibility check (src/orchestrate.rs, the `if nagq > 1` block)
# and the Python port's _AGQ_FAMILIES - change together.
.AGQ_FAMILIES <- "binomial, Poisson, negative-binomial or Gamma"

# Raise one classed warning as "<Tier>: <title>. <message>" and append it to
# `store` (an environment fastglmm() creates), which becomes `$warnings`.
.warn_keep <- function(store, kind, msg, cls) {
  tt <- .WARNING_KINDS[[kind]] %||% .UNKNOWN_KIND
  store$rows[[length(store$rows) + 1L]] <-
    list(tier = tt[[1L]], kind = kind, title = tt[[2L]], message = msg)
  printed <- sprintf("%s%s: %s. %s", toupper(substr(tt[[1L]], 1L, 1L)),
                     substring(tt[[1L]], 2L), tt[[2L]], msg)
  warning(warningCondition(printed, class = cls, call = NULL))
}

# Every warning this call raised, in raise order, one row each. Zero rows on a
# clean fit. Mirrors the Python port's `Fit.warnings` - change together.
.warnings_frame <- function(store) {
  col <- function(name) vapply(store$rows, `[[`, character(1), name)
  data.frame(tier = col("tier"), kind = col("kind"), title = col("title"),
             message = col("message"), stringsAsFactors = FALSE)
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
# Mirrors python/glmm/__init__.py::_pinned_detail - change together.
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

# One kernel note as list(msg=, cls=), or NULL to raise nothing. The `kind`
# string, not the English text, is the stable identifier; an unrecognized kind
# comes from a kernel newer than this wrapper (the Rust `Note` enum is
# #[non_exhaustive]) and still warns, under the base class. The Python port
# maps the same kinds to warning categories (glmm/__init__.py) - change
# together.
#
# Classes: "fastglmm_ill_conditioned", "fastglmm_pirls_exhausted",
# "fastglmm_unused_grouping_levels", "fastglmm_re_design_scale_spread",
# "fastglmm_single_level_grouping_dropped", "fastglmm_hessian_se_fallback",
# "fastglmm_exact_profile_fallback",
# "fastglmm_nb_shape_unsettled", "fastglmm_unknown_note" - all inheriting
# "fastglmm_diagnostic", so one handler catches the whole channel. Not every
# note comes from the solver: "unused_grouping_levels",
# "re_design_scale_spread" and "single_level_grouping_dropped" are raised by
# the formula lowering, which is the only layer that sees both the declared
# levels/design and the per-row codes.
.note_warning <- function(note, coef_names, converged) {
  if (identical(note$kind, "ill_conditioned")) {
    # `columns` arrives 1-based from the shim, so it indexes `coef_names`
    # directly. Out of range falls back to the index rather than printing NA,
    # mirroring the Python port's guard.
    named <- paste(
      vapply(note$columns, function(i) {
        if (i >= 1L && i <= length(coef_names)) {
          coef_names[[i]]
        } else {
          sprintf("column %d", i)
        }
      }, character(1)),
      collapse = ", "
    )
    msg <- paste(named, paste(
      "is almost a combination of other columns in the model, so its standard error is",
      "large. Its estimate is still correct, but imprecise. The other columns involved",
      "are not named. Consider dropping or combining predictors that carry the same",
      "information."))
    cls <- c("fastglmm_ill_conditioned", "fastglmm_diagnostic")
  } else if (identical(note$kind, "unused_grouping_levels")) {
    # The kernel packs "<group>: <level>, <level>" (src/orchestrate.rs). Split on
    # the first ": " only: a level label may contain one.
    cut <- regexpr(": ", note$detail, fixed = TRUE)
    group <- if (cut > 0L) substr(note$detail, 1L, cut - 1L) else note$detail
    levels <- if (cut > 0L) substring(note$detail, cut + 2L) else ""
    msg <- sprintf(paste(
      "Grouping factor '%s' has levels with no rows (%s). They stay in the model with",
      "random effects of exactly zero, but they are not counted in the number of groups.",
      "Remove unused categories before fitting."), group, levels)
    cls <- c("fastglmm_unused_grouping_levels", "fastglmm_diagnostic")
  } else if (identical(note$kind, "pirls_exhausted")) {
    # Raised only when the final re-evaluation of a converged fit hit the cap; see
    # the Python port's _note_warning for the other three cases.
    if (!(isTRUE(note$final_eval) && isTRUE(converged))) return(NULL)
    msg <- paste(
      "The final step that computes the reported results ran out of iterations. The",
      "estimates and their standard errors may be less accurate than usual.",
      "Try simplifying the random effects or rescaling the predictors.")
    cls <- c("fastglmm_pirls_exhausted", "fastglmm_diagnostic")
  } else if (identical(note$kind, "nb_shape_unsettled")) {
    msg <- sprintf(paste(
      "The search for the negative binomial shape parameter stopped at its limit of %d",
      "rounds before it settled. The coefficients and standard errors are computed at the",
      "last value it reached, which may not be the best one."), as.integer(note$evals))
    cls <- c("fastglmm_nb_shape_unsettled", "fastglmm_diagnostic")
  } else if (identical(note$kind, "re_design_scale_spread")) {
    msg <- sprintf(paste(
      "The predictors with random slopes for '%s' are on very different scales (ratio",
      "%.3g). The fit is not affected, but the reported random-effect standard deviations",
      "are hard to compare. Rescaling these predictors makes them easier to read."),
      note$detail, note$ratio)
    cls <- c("fastglmm_re_design_scale_spread", "fastglmm_diagnostic")
  } else if (identical(note$kind, "single_level_grouping_dropped")) {
    msg <- sprintf(paste(
      "Grouping factor '%s' has only one level, so no variance between groups can be",
      "estimated from it. Its random effect was dropped; the rest of the model was",
      "fitted without it."), note$detail)
    cls <- c("fastglmm_single_level_grouping_dropped", "fastglmm_diagnostic")
  } else if (identical(note$kind, "hessian_se_fallback")) {
    msg <- paste(
      "The usual standard errors could not be computed, so a simpler method was used.",
      "Its standard errors tend to be too small, so p-values and confidence intervals",
      "may look more precise than they are. Standard errors for the random-effect",
      "standard deviations are not available.")
    cls <- c("fastglmm_hessian_se_fallback", "fastglmm_diagnostic")
  } else if (identical(note$kind, "exact_profile_fallback")) {
    msg <- paste(
      "The default search method for this model did not settle on an answer, so the fit",
      "tried a different search method from the same starting point. The reported",
      "estimates come from whichever method reached the better answer.")
    cls <- c("fastglmm_exact_profile_fallback", "fastglmm_diagnostic")
  } else if (identical(note$kind, "non_integer_response")) {
    # The kernel's Note::NonIntegerResponse (src/fit/mod.rs); `evals` carries the
    # affected row count (src/orchestrate.rs reuses the same NoteInfo slot as
    # nb_shape_unsettled's `rounds`).
    msg <- sprintf(paste(
      "The response is not a whole number in %s. This family expects integer counts",
      "(binomial: the number of successes, y * weights). The fit still ran, but check",
      "whether this is the right family for the data."), .count(note$evals, "row"))
    cls <- c("fastglmm_non_integer_response", "fastglmm_diagnostic")
  } else {
    msg <- sprintf(paste(
      "The solver reported something ('%s') that this installed version does not",
      "recognize. Please report it at https://github.com/pawlenartowicz/glmm/issues."),
      note$kind)
    cls <- c("fastglmm_unknown_note", "fastglmm_diagnostic")
  }
  list(msg = msg, cls = cls)
}

.INNER_STEPS <- "Some of its inner steps ran out of iterations."

# glm_diverged messages by family: separation only means something for a binomial
# response, and the Gamma and inverse-Gaussian fits skip the linear-predictor check
# (src/glm.rs), so theirs cannot be called a divergence to an extreme.
.DIVERGED_BINOMIAL <- paste(
  "The fit did not converge. This usually means a predictor, or a combination of",
  "predictors, predicts the outcome perfectly (separation), so some fitted probabilities",
  "go to 0 or 1. The coefficients are from the last step; standard errors are",
  "not reported. Check the data for separation.")
.DIVERGED_COUNTS <- paste(
  "The fit did not converge. This usually means that some category of a predictor, or",
  "some combination of predictors, has only zero counts, so some fitted counts go to 0.",
  "The coefficients are from the last step; standard errors are not reported. Check for",
  "categories whose counts are all zero.")
.DIVERGED_CONTINUOUS <- paste(
  "The fit did not settle on an answer: the fitting steps stopped before converging. The",
  "coefficients are from the last step; standard errors are not reported. Check predictors",
  "with extreme values; with a link other than log, the log link is usually more stable.")

.count <- function(n, word) sprintf("%d %s%s", as.integer(n), word, if (n == 1) "" else "s")

# The one severe warning of a fit with converged = FALSE. The kernel does not say which
# stopping rule fired, so the port reads the cause off what the fit reports, most specific
# first: no coefficient, too few rows (a one-row response is trivially constant, so this
# comes first), a constant response, then the model. With random effects, a finite deviance
# means the kernel reached an end point (the budget stop, reported at its best point), and
# finite estimated coefficients (aliased slots are NaN by contract) confirm it; anything
# else failed. The deviance is what decides a model with no fixed effects. A GLMM inner
# cap-out during the search is one extra sentence rather than a second warning. Mirrors the
# Python port's _nonconvergence (python/glmm/__init__.py) - change together.
.nonconvergence <- function(family, mixed, notes, beta, aliased, deviance, y) {
  diag_cls <- function(kind) c(paste0("fastglmm_", kind), "fastglmm_diagnostic")
  n_est <- sum(!aliased)
  if (!mixed && n_est == 0L) {
    return(list(kind = "no_coefficients", msg = paste(
      "The model has no coefficients and no random effects, so there is nothing to",
      "estimate. Add an intercept or a predictor."), cls = diag_cls("no_coefficients")))
  }
  if (length(y) <= n_est) {
    return(list(kind = "too_few_rows", msg = sprintf(paste(
      "The model has %s to estimate but only %s, so no estimates were computed. Use more",
      "rows or fewer predictors."), .count(n_est, "coefficient"), .count(length(y), "row")),
      cls = diag_cls("too_few_rows")))
  }
  if (length(y) && isTRUE(all(y == y[[1L]]))) {
    return(list(kind = "constant_response", msg = sprintf(
      "Every value of the response is %s, so there is nothing to estimate. Check the response column.",
      sprintf("%g", y[[1L]])), cls = diag_cls("constant_response")))
  }
  if (!mixed) {
    if (identical(family, "gaussian")) {
      return(list(kind = "design_unsolvable", msg = paste(
        "The predictors could not be separated numerically, so no estimates were",
        "computed. Check for predictors that are copies or near-copies of each other."),
        cls = diag_cls("design_unsolvable")))
    }
    msg <- if (identical(family, "binomial")) {
      .DIVERGED_BINOMIAL
    } else if (family %in% c("poisson", "negativebinomial")) {
      .DIVERGED_COUNTS
    } else {
      .DIVERGED_CONTINUOUS
    }
    return(list(kind = "glm_diverged", msg = msg, cls = diag_cls("glm_diverged")))
  }
  pirls <- vapply(notes, function(n) identical(n$kind, "pirls_exhausted"), logical(1))
  inner <- if (any(pirls)) .INNER_STEPS else character()
  advice <- "Try a simpler random-effects structure or rescale the predictors."
  if (is.finite(deviance) && all(is.finite(beta[!aliased]))) {
    return(list(kind = "search_limit", msg = paste(c(
      "The search for the variance estimates reached its step limit before it settled.",
      inner,
      "The estimates shown are the best point found; they are often close, but this is",
      "not checked. Do not use them until the fit converges.", advice), collapse = " "),
      cls = diag_cls("search_limit")))
  }
  list(kind = "fit_failed", msg = paste(c(
    "The fitting algorithm failed and returned no estimates.", inner, advice),
    collapse = " "), cls = diag_cls("fit_failed"))
}

# `...` exists only to intercept known lme4 arguments with designed errors
# (spec section 1/section 4) - an unknown argument must never be silently swallowed
# (Decision 5: error, never silently differ).
.check_dots <- function(...) {
  dots <- list(...)
  if (!length(dots)) return(invisible())
  nms <- names(dots) %||% rep("", length(dots))
  for (nm in nms) {
    switch(nm,
      REML = {
        # REML = TRUE matches what the engine does, so only FALSE errors.
        if (isFALSE(dots$REML)) {
          stop("REML = FALSE is not supported: the glmm LMM path is REML-only ",
               "by design, a permanent choice, not a gap", call. = FALSE)
        }
      },
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

# Pre-checks for formula shapes the shared Rust parser cannot represent,
# in the order a user hits them (spec section 4 "Parser limits" + engine RE limits).
# Anything not caught here falls through to the parser, whose own message
# (e.g. term removal for `y ~ x - 1`) is surfaced verbatim.
.check_formula <- function(formula, f_str) {
  # Both checks below catch formula shapes the shared Rust parser has no
  # dedicated message for (it falls through to a generic syntax error), with
  # the same wording the Python port uses (python/glmm/__init__.py's `fit`) -
  # change together. Intercept-free random-effect terms ((0+x|g), (-1+x|g))
  # are NOT pre-checked here: the parser already raises
  # RandomInterceptSuppressionUnsupported with a specific message, and that
  # message reaches both ports unchanged.
  if (grepl("||", f_str, fixed = TRUE)) {
    stop("(x || g) double-bar terms are not supported: the glmm kernel always ",
         "fits the full RE correlation structure", call. = FALSE)
  }
  if ("." %in% all.vars(formula)) {
    stop("'.' is not supported by the formula parser; list the columns explicitly",
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
    col <- frame[[nm]]
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
           "estimates it - use family = \"negativebinomial\" (and note ",
           "init.theta seeding has no kernel hook yet)", call. = FALSE)
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
