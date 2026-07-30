# fastglmm() - the package's one entry point (R-port spec section 1, docs/GLMM/plans/
# 2026-07-15-r-port-fast-fitter-spec.md). The R side does row filtering
# (subset/na.action), family normalization, argument validation, and column
# marshalling; the formula is parsed and lowered by the Rust side
# (glmm::formula - one parser shared with the Python port, spec section 3), so this
# file deliberately contains no model.matrix / terms machinery.

`%||%` <- function(x, y) if (is.null(x)) y else x

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
#' The formula is parsed by the same Rust parser the Python port uses, which
#' accepts **bare column names only**: no `I()`, `poly()`, `log(x)`,
#' `cbind()`, `offset()`, no `.`, and no term removal (`- 1` / `0 +`). Compute
#' transformed columns first and pass them by name. Fixed and random effects
#' support `+`, `:`, `*`, `A/B` nesting, and `(1 + x | g)` random-effect terms
#' with a **full** correlation structure - `(x || g)` and intercept-free RE
#' terms are not fittable by the kernel and raise an error. Contrasts are
#' always treatment coding with the **first factor level** as base; to change
#' the base, `relevel()` the factor (a `contrasts` argument is deliberately
#' absent). Character columns are converted to factors with lexicographic
#' level order (as `factor()` does); a factor's declared level order is
#' honored.
#'
#' **`Gamma()` link trap:** R's `Gamma()` family object defaults to
#' `link = "inverse"`, and a family *object* is honored as given - R semantics
#' win. The string form `family = "gamma"` uses the glmm default
#' `link = "log"` instead. The two forms therefore fit different models;
#' choose deliberately.
#'
#' **`nAGQ` fallback (louder than lme4):** `nAGQ > 1` is honored on binomial
#' and Poisson mixed models with a single grouping factor and at most 3 random
#' effects per group. Any other shape **warns and falls back to Laplace**
#' (`nAGQ = 1`) instead of erroring the way `lme4::glmer` does - the fit you
#' get is a Laplace fit, and the warning is the only notice. This mirrors the
#' Python port so the two ports agree.
#'
#' Anything the engine cannot do is an error naming the reason - never a
#' silently different model. That includes `REML = FALSE` (the LMM path is
#' REML-only **by design**), `offset()`, `control=`/`verbose=`,
#' `family = inverse.gaussian()` and `link = "cloglog"` (approved for GLMM
#' 0.1.1, not yet in the kernel), and quasi-likelihood `dispersion=` on
#' binomial/poisson.
#'
#' @param formula an lme4-style model formula (or a string), e.g.
#'   `y ~ t + d + t:d + (1 + t | g)`. Bare column names only - see Details.
#' @param data a `data.frame` (or something coercible) holding every column
#'   the formula names.
#' @param family a family object, family function, or string: one of
#'   `gaussian`, `binomial` (logit/probit), `poisson` (log), `Gamma`
#'   (log/inverse), `"negativebinomial"` (log; the shape `theta` is
#'   estimated). `inverse.gaussian` and `binomial("cloglog")` are approved for
#'   GLMM 0.1.1 and error until the kernel implements them.
#' @param weights optional per-row prior (case) weights - `lme4::glmer`'s
#'   `weights=`. For an aggregated binomial, pass the success **proportion**
#'   as the response and the trial count here (the same model as
#'   `cbind(successes, failures)`, whose syntax the parser does not accept).
#' @param subset optional row filter, evaluated in `data` like `lm`'s
#'   `subset=`. Applied before fitting - row filtering, not parsing.
#' @param na.action how to handle `NA`s in the model columns (default
#'   `getOption("na.action")`, normally [stats::na.omit]). `NA`s must be
#'   resolved before the kernel sees the data; an action that leaves them in
#'   place (`na.pass`) is an error.
#' @param nAGQ adaptive Gauss-Hermite node count: an odd integer in
#'   `1..=25` (`1` = Laplace, the default). More permissive than lme4 in range
#'   but see the fallback note in Details.
#' @param start optional warm start, lme4's name and shape:
#'   `list(beta =, theta =)` with `theta` the random-effect Cholesky vector.
#'   Distinct from `init.theta`, which is the negative-binomial shape.
#' @param wald.se Wald standard-error mode: `"hessian"` (default) or `"rx"`.
#' @param dispersion Gamma dispersion directive: `NULL` (estimate via Pearson,
#'   the default), `"estimate"` (same), or a number to hold it fixed.
#'   Non-`NULL` on binomial/poisson would mean quasi-likelihood - GLMM 0.1.1,
#'   errors today.
#' @param init.theta negative-binomial shape seed, named for
#'   `MASS::glm.nb(init.theta=)`. No kernel hook exists yet to seed the shape
#'   search, so any non-`NULL` value is an error (the default cold start is
#'   what runs).
#' @param ... intercepted, never silently swallowed: known lme4 arguments
#'   (`REML`, `control`, `verbose`, `contrasts`, `offset`) raise errors saying
#'   why they cannot be honored; unknown names error as unused arguments.
#'
#' @return An object of class `"fastglmm"`: fixed effects ([fixef]), Wald
#'   covariance (`vcov()`), variance components on the SD/correlation scale
#'   ([VarCorr]), `converged` and `singular` flags ([isSingular]), plus
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
                     nAGQ = 1L,
                     start = NULL,
                     wald.se = c("hessian", "rx"),
                     dispersion = NULL,
                     init.theta = NULL,
                     ...) {
  .check_dots(...)
  wald.se <- match.arg(wald.se)

  if (is.character(formula)) formula <- stats::as.formula(formula)
  if (!inherits(formula, "formula")) {
    stop("`formula` must be a formula or a formula string", call. = FALSE)
  }
  f_str <- paste(deparse(formula, width.cutoff = 500L), collapse = " ")
  .check_formula(formula, f_str)

  fam <- .normalize_family(family)

  # --- kernel-gap checks: GLMM 0.1.1 approved, not yet implemented (mirrors
  # python/glmm/__init__.py's NotImplementedError block; docs/GLMM/0.1.1/). ---
  if (fam$name == "inversegaussian") {
    stop("family 'inverse.gaussian' requires GLMM 0.1.1; ",
         "not yet implemented in the kernel (docs/GLMM/0.1.1/)", call. = FALSE)
  }
  if (fam$link == "cloglog") {
    stop("link 'cloglog' requires GLMM 0.1.1; ",
         "not yet implemented in the kernel (docs/GLMM/0.1.1/)", call. = FALSE)
  }

  mixed <- grepl("|", f_str, fixed = TRUE)

  # Valid-but-inapplicable options: warn and strip, mirroring the Python port
  # (nothing inapplicable may reach the kernel - its checks are assert!s).
  if (!is.null(dispersion) && !(fam$name %in% .DISPERSION_FAMILIES)) {
    warning("dispersion= is not applicable to family '", fam$name,
            "'; ignored", call. = FALSE)
    dispersion <- NULL
  }
  if (!is.null(dispersion)) {
    ok <- identical(dispersion, "estimate") ||
      (is.numeric(dispersion) && length(dispersion) == 1L && is.finite(dispersion))
    if (!ok) {
      stop("dispersion must be NULL, \"estimate\", or a single number",
           call. = FALSE)
    }
    if (fam$name %in% c("binomial", "poisson") && mixed) {
      warning("quasi-likelihood dispersion on binomial/poisson is GLM-only; ",
              "ignored for a mixed formula", call. = FALSE)
      dispersion <- NULL
    }
  }
  if (identical(dispersion, "estimate") && fam$name == "gamma") {
    # gamma's default (NULL) already computes the Pearson estimate.
    dispersion <- NULL
  }
  if (!is.null(dispersion) && fam$name %in% c("binomial", "poisson")) {
    stop("quasi-likelihood dispersion on family '", fam$name,
         "' requires GLMM 0.1.1; not yet implemented in the kernel ",
         "(docs/GLMM/0.1.1/)", call. = FALSE)
  }
  if (!is.null(init.theta) && fam$name != "negativebinomial") {
    warning("init.theta= applies only to family 'negativebinomial'; ignored",
            call. = FALSE)
    init.theta <- NULL
  }
  if (!is.null(init.theta)) {
    stop("init.theta= (negative-binomial shape seed) has no kernel hook yet; ",
         "only the default cold-start shape search is supported", call. = FALSE)
  }

  if (!(is.numeric(nAGQ) && length(nAGQ) == 1L && !is.na(nAGQ) &&
        nAGQ == as.integer(nAGQ) && nAGQ >= 1L && nAGQ <= .MAX_NAGQ &&
        nAGQ %% 2L == 1L)) {
    stop("nAGQ must be an odd integer in 1..=", .MAX_NAGQ, call. = FALSE)
  }
  nAGQ <- as.integer(nAGQ)

  if (!is.null(start)) {
    if (!is.list(start)) {
      stop("start must be a list with elements 'beta' and/or 'theta' ",
           "(lme4's shape; 'theta' is the RE Cholesky vector, NOT the ",
           "negative-binomial shape - that is init.theta)", call. = FALSE)
    }
    unknown <- setdiff(names(start), c("beta", "theta"))
    if (length(unknown)) {
      warning("start elements ignored: ", paste(unknown, collapse = ", "),
              call. = FALSE)
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

  w <- eval(substitute(weights), data, parent.frame())
  if (!is.null(w)) {
    if (!is.numeric(w) || length(w) != nrow(data) || anyNA(w)) {
      stop("weights must be a numeric vector with one entry per row of data",
           call. = FALSE)
    }
    if (any(w <= 0)) {
      # The kernel requires strictly positive weights; a zero weight is a row
      # that should not be in the fit at all.
      stop("weights must be positive; drop zero-weight rows with subset= ",
           "instead", call. = FALSE)
    }
    frame[["(weights)"]] <- as.double(w)
  }

  s <- eval(substitute(subset), data, parent.frame())
  if (!is.null(s)) frame <- frame[s, , drop = FALSE]

  naf <- if (is.character(na.action)) {
    get(na.action, mode = "function")
  } else {
    na.action
  }
  if (!is.function(naf)) stop("invalid na.action", call. = FALSE)
  frame <- naf(frame)
  if (anyNA(frame[vars])) {
    stop("missing values remain in the model columns after na.action; ",
         "the kernel cannot fit NA - use na.action = na.omit or complete ",
         "the data", call. = FALSE)
  }
  if (nrow(frame) == 0L) stop("no rows left to fit", call. = FALSE)
  w <- frame[["(weights)"]]

  # --- marshalling: factors cross as (levels, 0-based codes) so the caller's
  # declared level order (the treatment base) survives into Rust's
  # Column::Factor - the declared-order path, plan gate 3. ---
  numeric_cols <- list()
  factor_levels <- list()
  factor_codes <- list()
  for (nm in vars) {
    col <- frame[[nm]]
    if (is.character(col)) col <- factor(col) # lexicographic, as factor() does
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
    as.double(start$beta %||% double()),
    as.double(start$theta %||% double())
  )

  if (!is.null(r$agq_warning)) warning(r$agq_warning, call. = FALSE)
  if (r$singular) {
    # lme4's exact text (cross-port agreement, spec section 5), extended with
    # the degenerate components. The Python port emits the same message
    # (glmm/__init__.py) - change together.
    warning(paste(c("boundary (singular) fit: see help('isSingular')",
                    .singular_detail(r$varcorr, r$re_group_names,
                                     r$re_group_terms)),
                  collapse = "; "), call. = FALSE)
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
    converged = r$converged,
    singular = r$singular,
    n_eval = r$n_eval,
    deviance = r$deviance,
    # logLik()/AIC()/BIC() inputs. `reml` marks the LMM paths, whose `loglik`
    # is a REML criterion rather than an ML one - see logLik.fastglmm().
    loglik = r$loglik,
    df = as.integer(r$df),
    reml = r$reml,
    re_group_names = r$re_group_names,
    re_group_terms = r$re_group_terms,
    call = match.call(),
    formula = f_str,
    family = fam$object,
    # Port-vocabulary family name ("gamma", not R's "Gamma") - what the
    # methods dispatch on; `family` above is the R-facing object.
    family_name = fam$name,
    frame = frame,
    nobs = nrow(frame),
    # Effective node count: the shim strips ineligible nAGQ>1 to Laplace
    # (with the warning surfaced above), so record what actually ran.
    nAGQ = if (is.null(r$agq_warning)) nAGQ else 1L
  ), class = "fastglmm")
}

# Names of the exactly-degenerate RE components, for the singular warning:
# "sd(term | group) = 0" per collapsed variance, "corr(a, b | group) = +/-1"
# per degenerate correlation. Exact comparisons are safe because the kernel
# pins boundary components to exact 0 / +/-1 (algorithms-lmm.md "Boundary
# handling"); character(0) when only the relative-tolerance singular check
# fired, which keeps the bare lme4 text.
.singular_detail <- function(varcorr, group_names, group_terms) {
  parts <- character()
  for (g in seq_along(varcorr)) {
    sc <- .stddev_corr(varcorr[[g]])
    terms <- group_terms[[g]]
    grp <- group_names[[g]]
    for (i in which(sc$stddev == 0)) {
      parts <- c(parts, sprintf("sd(%s | %s) = 0", terms[[i]], grp))
    }
    q <- length(sc$stddev)
    if (q > 1L) {
      for (cc in 1:(q - 1L)) {
        for (rr in (cc + 1L):q) {
          if (sc$stddev[cc] > 0 && sc$stddev[rr] > 0 &&
              abs(sc$correlation[rr, cc]) == 1) {
            parts <- c(parts, sprintf("corr(%s, %s | %s) = %+d", terms[[cc]],
                                      terms[[rr]], grp,
                                      as.integer(sc$correlation[rr, cc])))
          }
        }
      }
    }
  }
  parts
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
               "by design, a permanent choice, not a gap ",
               "(docs/GLMM/glmm-vs-lme4-mixedmodels.md)", call. = FALSE)
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
      offset = stop("offset= is not supported: the kernel has no offset ",
                    "field and the formula parser no offset() term ",
                    "(engine spec 2026-07-15-engine-loglik-diagnostics)",
                    call. = FALSE),
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
  if (grepl("||", f_str, fixed = TRUE)) {
    stop("(x || g) double-bar terms are not supported: the glmm kernel ",
         "always fits the full RE correlation structure (a kernel property, ",
         "not a parser gap - see src/spec.rs)", call. = FALSE)
  }
  if (grepl("\\bcbind\\s*\\(", f_str)) {
    stop("cbind(successes, failures) is not accepted by the shared formula ",
         "parser, but the model is reachable: pass the success proportion ",
         "as the response and the trial count as weights= - that is exactly ",
         "lme4's cbind() objective", call. = FALSE)
  }
  if (grepl("\\boffset\\s*\\(", f_str)) {
    stop("offset() is not supported: the kernel has no offset field and the ",
         "formula parser no offset() term", call. = FALSE)
  }
  if ("." %in% all.vars(formula)) {
    stop("'.' is not supported by the shared formula parser; ",
         "list the columns explicitly", call. = FALSE)
  }
  # Intercept-free RE terms: scan each (lhs | rhs) chunk's lhs for 0 / -1.
  re_lhs <- regmatches(f_str, gregexpr("\\(([^|()]*)\\|", f_str))[[1]]
  if (any(grepl("(^|[^[:alnum:]._])0([^[:alnum:]._]|$)|-\\s*1",
                sub("\\|$", "", re_lhs)))) {
    stop("intercept-free random-effect terms ((0 + x | g), (-1 + x | g)) are ",
         "not supported: the RE correlation structure is always full and ",
         "includes the intercept (a kernel property - see src/spec.rs)",
         call. = FALSE)
  }
  # Any function call = a non-bare term (log(x), I(x^2), poly(x, 2), ...).
  ops <- c("~", "+", "*", ":", "/", "|", "(", "-")
  calls <- setdiff(all.names(formula), c(all.vars(formula), ops))
  if (length(calls)) {
    stop("function call(s) in formula are not supported (",
         paste(unique(calls), collapse = ", "), "): the shared parser takes ",
         "bare column names only - compute the column first (e.g. ",
         "data$log_x <- log(data$x)) and pass it by name", call. = FALSE)
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
           "binomial, poisson, Gamma, \"negativebinomial\" ",
           "(inverse.gaussian is GLMM 0.1.1)", call. = FALSE)
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
      stop("unknown family \"", family, "\"; expected one of ",
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
