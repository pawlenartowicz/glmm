# S3 methods for "fastglmm" (R-port spec section 2). Plain S3, deliberately NOT an
# S4 merMod subclass - merMod has slots this engine cannot fill, and a
# half-filled merMod breaks every downstream package that trusts the class.
# Engine-blocked accessors error with the reason (spec section 4) rather than
# returning something silently different from what an lme4 user expects.

# vech-packed (column-major lower-triangular) covariance block -> list of
# per-dimension stddevs + full correlation matrix. Mirrors Rust
# `Fit::stddev_corr` (GLMM/src/fit/mod.rs) - change together. 0-based r/c in
# idx() to keep the formula identical to the Rust side; +1 shifts into R.
.stddev_corr <- function(vech) {
  m <- length(vech)
  q <- as.integer((sqrt(1 + 8 * m) - 1) / 2)
  if (q * (q + 1L) / 2L != m) {
    stop("varcorr block is not a valid vech (length ", m, ")", call. = FALSE)
  }
  idx <- function(r, c) c * q - (c * c - c) / 2 + (r - c) + 1L
  sd <- vapply(0:(q - 1L), function(i) sqrt(vech[idx(i, i)]), 0)
  corr <- diag(1, q)
  if (q > 1L) {
    for (c in 0:(q - 2L)) {
      for (r in (c + 1L):(q - 1L)) {
        rho <- vech[idx(r, c)] / (sd[r + 1L] * sd[c + 1L])
        corr[r + 1L, c + 1L] <- rho
        corr[c + 1L, r + 1L] <- rho
      }
    }
  }
  list(stddev = sd, correlation = corr)
}

# Number of levels each RE grouping has in the fitted frame. A nested
# grouping's composite name ("A:B") counts observed combinations.
.group_sizes <- function(object) {
  vapply(object$re_group_names, function(nm) {
    parts <- strsplit(nm, ":", fixed = TRUE)[[1]]
    if (!all(parts %in% names(object$frame))) return(NA_integer_)
    length(unique(interaction(object$frame[parts], drop = TRUE)))
  }, 0L)
}

# One-line description of what actually ran, for print/summary headers.
.method_line <- function(object) {
  fam <- object$family_name
  mixed <- length(object$varcorr) > 0L
  if (!mixed) {
    if (fam == "gaussian") "Linear model fit by least squares [fastglmm]"
    else "Generalized linear model fit by IRLS [fastglmm]"
  } else if (fam == "gaussian") {
    "Linear mixed model fit by REML [fastglmm]"
  } else if (object$nAGQ > 1L) {
    paste0("Generalized linear mixed model fit by maximum likelihood ",
           "(Adaptive Gauss-Hermite Quadrature, nAGQ = ", object$nAGQ,
           ") [fastglmm]")
  } else {
    paste0("Generalized linear mixed model fit by maximum likelihood ",
           "(Laplace Approximation) [fastglmm]")
  }
}

#' @export
print.fastglmm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  cat(.method_line(x), "\n")
  cat("Formula: ", x$formula, "\n", sep = "")
  cat(" Family: ", x$family$family, " (", x$family$link, ")\n", sep = "")
  if (length(x$varcorr)) {
    cat("Random effects:\n")
    print(VarCorr(x), digits = digits)
    sizes <- .group_sizes(x)
    cat("Number of obs: ", x$nobs, ", groups:  ",
        paste0(x$re_group_names, ", ", sizes, collapse = "; "), "\n", sep = "")
  } else {
    cat("Number of obs:", x$nobs, "\n")
  }
  cat("Fixed effects:\n")
  print(round(x$beta, digits))
  if (!x$converged) cat("Warning: fit did not converge\n")
  if (x$singular) cat("boundary (singular) fit: see help('isSingular')\n")
  invisible(x)
}

#' Summarize a fastglmm fit
#'
#' lme4's blocks in lme4's order: method line, formula, data, the criterion
#' (`REML criterion at convergence` on an LMM, the `AIC BIC logLik deviance
#' df.resid` row on an ML fit), scaled residuals, random effects with
#' variances, the observation/group counts, the Wald-z coefficient table,
#' the correlation of fixed effects, and a footer. The Python port's
#' `Fit.summary()` prints the same blocks - change together.
#'
#' The coefficient table carries Wald z statistics and normal-based p-values
#' for all families: the kernel surfaces no residual df, so there is no t and
#' no Satterthwaite/Kenward-Roger. The REML criterion is printed on LMMs
#' where lme4 prints it; the AIC row is printed only on ML fits, because a
#' REML AIC invites exactly the across-models comparison it is not valid for.
#' `glance()` returns AIC/BIC on request either way.
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return An object of class `"summary.fastglmm"` with `coefficients`
#'   (`Estimate`, `Std. Error`, `z value`, `Pr(>|z|)`), `criterion`,
#'   `scaled_residuals` (`NULL` when the fit did not converge) and
#'   `corr_fixed` (`NULL` when there is one coefficient).
#' @export
summary.fastglmm <- function(object, ...) {
  wald <- .wald(object)
  coefficients <- cbind(
    Estimate = object$beta,
    `Std. Error` = object$se,
    `z value` = wald$z,
    `Pr(>|z|)` = wald$p
  )
  ll <- object$loglik
  criterion <- if (object$reml) {
    c("REML criterion at convergence" = -2 * ll)
  } else {
    ic <- .ml_criteria(object)
    c(AIC = ic$AIC,
      BIC = ic$BIC,
      logLik = ll,
      deviance = ic$deviance,
      df.resid = ic$df.residual)
  }
  scaled <- NULL
  if (length(object$fitted) == object$nobs && object$nobs > 0L) {
    r <- residuals(object, type = "pearson")
    scaled <- stats::quantile(r, c(0, 0.25, 0.5, 0.75, 1), names = FALSE)
    names(scaled) <- c("Min", "1Q", "Median", "3Q", "Max")
  }
  corr_fixed <- if (length(object$beta) >= 2L) stats::cov2cor(object$vcov) else NULL
  structure(list(
    object = object,
    coefficients = coefficients,
    criterion = criterion,
    scaled_residuals = scaled,
    corr_fixed = corr_fixed
  ), class = "summary.fastglmm")
}

#' @export
print.summary.fastglmm <- function(x,
                                   digits = max(3L, getOption("digits") - 3L),
                                   ...) {
  object <- x$object
  cat(.method_line(object), "\n")
  cat("Formula: ", object$formula, "\n", sep = "")
  cat("   Data: ", object$data_name, "\n", sep = "")
  cat(" Family: ", object$family$family, " (", object$family$link, ")\n\n",
      sep = "")
  if (length(x$criterion) == 1L) {
    cat(names(x$criterion), ": ", format(x$criterion, digits = digits + 1L),
        "\n\n", sep = "")
  } else {
    print(x$criterion, digits = digits + 1L)
    cat("\n")
  }
  if (!is.null(x$scaled_residuals)) {
    cat("Scaled residuals:\n")
    print(x$scaled_residuals, digits = digits)
    cat("\n")
  }
  if (length(object$varcorr)) {
    cat("Random effects:\n")
    print(VarCorr(object), digits = digits, variance = TRUE)
    sizes <- .group_sizes(object)
    cat("Number of obs: ", object$nobs, ", groups:  ",
        paste0(object$re_group_names, ", ", sizes, collapse = "; "),
        "\n\n", sep = "")
  } else {
    cat("Number of obs:", object$nobs, "\n\n")
  }
  cat("Fixed effects:\n")
  stats::printCoefmat(x$coefficients, digits = digits, na.print = "NA")
  if (any(object$aliased)) {
    cat("(", sum(object$aliased),
        " coefficient(s) not defined because of singularities)\n", sep = "")
  }
  if (!is.null(x$corr_fixed)) {
    # Full names on both axes (lme4 abbreviates the columns) - the Python
    # port prints the same; change together.
    cat("\nCorrelation of Fixed Effects:\n")
    p <- nrow(x$corr_fixed)
    m <- format(round(x$corr_fixed, 3), nsmall = 3)
    m[upper.tri(m, diag = TRUE)] <- ""
    print(noquote(m[-1L, -p, drop = FALSE]), right = TRUE)
  }
  cat("\n")
  if (object$family_name %in% c("gamma", "negativebinomial", "inversegaussian")) {
    label <- if (object$family_name == "negativebinomial") {
      "Shape (theta)"
    } else {
      "Dispersion (phi, Pearson)"
    }
    cat(label, ": ", format(object$dispersion, digits = digits), "\n", sep = "")
  }
  cat("Optimizer evaluations: ", object$n_eval,
      ", converged: ", object$converged, "\n", sep = "")
  if (object$singular) cat("boundary (singular) fit: see help('isSingular')\n")
  cat("z value / Pr(>|z|) are Wald z on the asymptotic normal; no t or ",
      "residual df is reported.\n", sep = "")
  invisible(x)
}

#' Extract fixed effects
#'
#' Generic + method. The generic is defined here (masking `lme4::fixef` when
#' both are attached is harmless - S3 dispatch finds the same method).
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return Named numeric vector of fixed-effect estimates; aliased
#'   (rank-deficient) coefficients are `NA`, mirroring `lm`/lme4.
#' @export
fixef <- function(object, ...) UseMethod("fixef")

#' @rdname fixef
#' @export
fixef.fastglmm <- function(object, ...) object$beta

#' @export
vcov.fastglmm <- function(object, ...) object$vcov

#' Variance components on the SD/correlation scale
#'
#' Returns one covariance matrix per grouping (declaration order), each with
#' `"stddev"` and `"correlation"` attributes, printed lme4-shaped. This is the
#' load-bearing accessor for parameter-recovery use: `attr(vc$g, "stddev")`
#' are the tau estimates and `attr(vc$g, "correlation")` the rho estimates,
#' straight from the kernel's `varcorr` (validated against lme4's `VarCorr`).
#'
#' For a **gaussian** mixed fit a `Residual` row is printed after the group
#' components, as in lme4, carrying `sigma()` (the REML residual standard
#' deviation the kernel reports as `dispersion`).
#'
#' @param x a [fastglmm] fit.
#' @param ... unused.
#' @return A list of class `"VarCorr.fastglmm"`, one named element per
#'   grouping.
#' @export
VarCorr <- function(x, ...) UseMethod("VarCorr")

#' @rdname VarCorr
#' @export
VarCorr.fastglmm <- function(x, ...) {
  out <- lapply(seq_along(x$varcorr), function(g) {
    sc <- .stddev_corr(x$varcorr[[g]])
    terms <- x$re_group_terms[[g]]
    v <- outer(sc$stddev, sc$stddev) * sc$correlation
    dimnames(v) <- list(terms, terms)
    attr(v, "stddev") <- stats::setNames(sc$stddev, terms)
    attr(v, "correlation") <- sc$correlation
    v
  })
  names(out) <- x$re_group_names
  # "sc" mirrors lme4's residual-sd attribute name; NA for non-gaussian
  # families (phi==1 families have no free residual scale; gamma's
  # sqrt(phi) is a dispersion, reported by sigma(), not a Residual row).
  structure(out, class = "VarCorr.fastglmm",
            family = x$family_name,
            sc = if (x$family_name == "gaussian") sqrt(x$dispersion)
                 else NA_real_)
}

#' @param digits number of significant digits to print.
#' @param variance add a `Variance` column before `Std.Dev.` (lme4's
#'   `print.summary.merMod` shape; the bare `print.fastglmm` header keeps
#'   `Std.Dev.` only, as `lme4::print.merMod` does).
#' @rdname VarCorr
#' @export
print.VarCorr.fastglmm <- function(x,
                                   digits = max(3L,
                                                getOption("digits") - 3L),
                                   variance = FALSE,
                                   ...) {
  cols <- c("Groups", "Name", if (variance) "Variance", "Std.Dev.", "Corr")
  cells <- list()
  for (g in seq_along(x)) {
    sd <- attr(x[[g]], "stddev")
    corr <- attr(x[[g]], "correlation")
    for (i in seq_along(sd)) {
      corr_cells <- if (i > 1L) {
        paste(format(round(corr[i, seq_len(i - 1L)], 2), nsmall = 2), collapse = " ")
      } else {
        ""
      }
      row <- c(if (i == 1L) names(x)[g] else "", names(sd)[i])
      if (variance) row <- c(row, format(sd[i]^2, digits = digits))
      row <- c(row, format(sd[i], digits = digits), corr_cells)
      cells[[length(cells) + 1L]] <- row
    }
  }
  sc <- attr(x, "sc")
  if (!is.null(sc) && is.finite(sc)) {
    row <- c("Residual", "")
    if (variance) row <- c(row, format(sc^2, digits = digits))
    row <- c(row, format(sc, digits = digits), "")
    cells[[length(cells) + 1L]] <- row
  }
  rows <- as.data.frame(do.call(rbind, cells), stringsAsFactors = FALSE)
  names(rows) <- cols
  print(rows, row.names = FALSE, right = FALSE)
  invisible(x)
}

#' Residual scale
#'
#' The residual standard deviation for **gaussian** fits (`sqrt` of the
#' kernel's dispersion: `RSS/(n-p)` for a linear model, matching
#' `summary.lm`'s sigma; the REML residual variance for a mixed model,
#' matching `lme4::sigma`), `sqrt(phi)` for Gamma and inverse-Gaussian, and
#' `1` for binomial/poisson/negative-binomial (the scale is fixed, as in
#' lme4).
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @export
sigma.fastglmm <- function(object, ...) {
  fam <- object$family_name
  if (fam %in% c("gaussian", "gamma", "inversegaussian")) {
    sqrt(object$dispersion)
  } else {
    1
  }
}

#' @export
nobs.fastglmm <- function(object, ...) object$nobs

#' Formula, family, and model frame accessors
#'
#' `formula()` returns the formula **string** as given: the R side never
#' builds a `terms` object (the parser is Rust-side, R-port spec section 3), and
#' synthesizing one would disagree with how the model was actually built -
#' which is also why `terms()` errors.
#'
#' @param x,formula,object a [fastglmm] fit.
#' @param ... unused.
#' @export
formula.fastglmm <- function(x, ...) x$formula

#' @rdname formula.fastglmm
#' @export
family.fastglmm <- function(object, ...) object$family

#' @rdname formula.fastglmm
#' @export
model.frame.fastglmm <- function(formula, ...) formula$frame

#' Wald confidence intervals
#'
#' Normal-approximation intervals off the full Wald covariance matrix
#' (`vcov()`). `method = "profile"` and `method = "boot"` are not
#' available: the engine has no profiling machinery, and bootstrap needs
#' refitting infrastructure not built yet.
#'
#' @param object a [fastglmm] fit.
#' @param parm coefficients to include (names or indices; default all).
#' @param level confidence level.
#' @param method only `"Wald"`.
#' @param ... unused.
#' @return A matrix with one row per coefficient.
#' @export
confint.fastglmm <- function(object, parm, level = 0.95,
                             method = "Wald", ...) {
  if (!identical(method, "Wald")) {
    stop("confint(method = \"", method, "\") is not available: the engine ",
         "has no profiling machinery; only method = \"Wald\" is supported",
         call. = FALSE)
  }
  est <- object$beta
  if (missing(parm)) parm <- names(est)
  q <- stats::qnorm((1 + level) / 2)
  lo <- est[parm] - q * object$se[parm]
  hi <- est[parm] + q * object$se[parm]
  out <- cbind(lo, hi)
  dimnames(out) <- list(names(est[parm]),
                        sprintf("%.1f %%", 100 * c((1 - level) / 2,
                                                   (1 + level) / 2)))
  out
}

#' Is the fit singular?
#'
#' `TRUE` iff the fit converged onto the variance-component boundary - the
#' same condition `lme4::isSingular` reports (the kernel computes it; see
#' `glmm::Fit::singular`).
#'
#' @param x a [fastglmm] fit.
#' @param ... unused.
#' @export
isSingular <- function(x, ...) UseMethod("isSingular")

#' @rdname isSingular
#' @export
isSingular.fastglmm <- function(x, ...) x$singular

# -- Engine-blocked accessors (spec section 4): each is a hard "cannot be done
# honestly today", erroring with the reason and the lifting spec - never a
# silently different answer. ----------------------------------------------

.engine_blocked <- function(what) {
  stop(what, " is not available: it needs a design matrix built from rows the ",
       "fit never saw, and the formula machinery is Rust-side. Fixed effects: ",
       "fixef(). Conditional modes: ranef(). Fitted values: fitted().",
       call. = FALSE)
}

#' Random-effect conditional modes
#'
#' The conditional modes (BLUPs) `b-hat`, in `lme4::ranef`'s shape: a named list
#' of data frames, one per grouping factor in the order the formula declares
#' them, with one row per level and one column per random-effect term.
#'
#' The labels come from the kernel, not from this package: which internal layout
#' a grouping factor lands in is a data-dependent speed decision, so only the
#' layer that made it can say which level a number belongs to. Padded slots of a
#' nested grouping carry no level and are not reported.
#'
#' Two differences from `lme4::ranef` worth knowing. Conditional variances are
#' not computed, so `condVar` is not supported. And a grouping level that owns
#' model width but no rows is reported with its mode shrunk to zero, where lme4
#' has no such row at all; the fit warns about that case
#' (`fastglmm_unused_grouping_levels`).
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return A named `list` of `data.frame`s, empty for a fit with no random
#'   effects or one that did not converge.
#' @export
ranef <- function(object, ...) UseMethod("ranef")

#' @rdname ranef
#' @export
ranef.fastglmm <- function(object, ...) {
  blocks <- object$ranef_blocks
  out <- lapply(blocks, function(b) {
    # `values` arrives row-major (level-major, terms within a level), which is
    # byrow = TRUE for R's column-major matrix().
    m <- matrix(as.double(b$values), nrow = length(b$levels),
                ncol = length(b$terms), byrow = TRUE,
                dimnames = list(b$levels, b$terms))
    as.data.frame(m, stringsAsFactors = FALSE)
  })
  names(out) <- vapply(blocks, function(b) b$group, character(1))
  out
}

#' @export
predict.fastglmm <- function(object, ...) .engine_blocked("predict()")

#' Fitted values
#'
#' The conditional means `mu-hat` per row of the model frame - `lme4::fitted`'s
#' quantity, including the random-effect contribution and any offset, on the
#' response scale.
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return A named numeric vector, named by the model frame's row names. Empty
#'   for a fit that did not converge.
#' @export
fitted.fastglmm <- function(object, ...) {
  v <- as.double(object$fitted)
  if (length(v) == nrow(object$frame)) names(v) <- rownames(object$frame)
  v
}

#' Residuals
#'
#' `type = "response"` is `y - fitted(object)`; `type = "pearson"` divides by
#' `sqrt(phi * V(mu) / w)` with the family's variance function. Deviance and
#' working residuals are not offered: they need per-family deviance formulas
#' nothing else in the package carries, and a wrong default would silently
#' differ from `lme4::residuals(type=)`.
#'
#' @param object a [fastglmm] fit.
#' @param type `"response"` or `"pearson"`.
#' @param ... unused.
#' @export
residuals.fastglmm <- function(object, type = c("response", "pearson"), ...) {
  type <- match.arg(type)
  mu <- as.double(object$fitted)
  if (!length(mu)) {
    stop("residuals are unavailable: the fit did not converge, so fitted() is empty",
         call. = FALSE)
  }
  r <- as.double(object$y) - mu
  if (type == "pearson") r <- r / sqrt(.pearson_scale(object, mu))
  names(r) <- rownames(object$frame)
  r
}

# phi * V(mu) / w for Pearson residuals. Mirrors the kernel's family table
# (src/family.rs) and the Python port's `summary._VARIANCE` - change together.
.pearson_scale <- function(object, mu) {
  fam <- object$family_name
  theta <- object$dispersion
  v <- switch(fam,
    gaussian = rep(1, length(mu)),
    binomial = mu * (1 - mu),
    poisson = mu,
    gamma = mu^2,
    negativebinomial = mu + mu^2 / theta,
    inversegaussian = mu^3
  )
  phi <- if (fam %in% c("gaussian", "gamma", "inversegaussian")) object$dispersion else 1
  w <- object$weights
  if (is.null(w)) w <- rep(1, length(mu))
  phi * v / w
}

#' @export
coef.fastglmm <- function(object, ...) {
  stop("coef() is not implemented: lme4's coef() means fixed + random ",
       "effects per group, and returning fixed effects only would silently ",
       "differ from what an lme4 user expects from the same call. Combine ",
       "fixef() and ranef(), or use fixef() alone.", call. = FALSE)
}

#' Log-likelihood of a fastglmm fit
#'
#' On `stats::logLik`'s scale, so `AIC()` and `BIC()` work directly. This is
#' `glmm::Fit::loglik`, not `$deviance` - the latter is the optimizer criterion
#' and differs from `-2*logLik` by model-dependent constants.
#'
#' For an LMM the value is the **REML criterion**, matching `lme4::logLik` on a
#' REML fit, and the returned object carries `REML = TRUE`. REML criteria are
#' comparable only between models with identical fixed effects; an AIC across
#' LMMs with different fixed effects is invalid. The LMM path is REML-only by
#' design, so there is no ML value to return instead.
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return A `"logLik"` object with `df`, `nobs` and `REML` attributes. `NaN`
#'   wherever the fit failed, matching `$loglik`.
#' @export
logLik.fastglmm <- function(object, ...) {
  val <- object$loglik
  attr(val, "df") <- object$df
  attr(val, "nobs") <- object$nobs
  attr(val, "REML") <- object$reml
  class(val) <- "logLik"
  val
}

# -- broom-shaped accessors. Registered on `generics::tidy`/`generics::glance`
# (not a package-local generic like fixef/VarCorr above): modelsummary,
# texreg, gt and kableExtra call `broom::tidy(fit)`, which IS generics::tidy
# re-exported, so a local generic would never be found. `generics` is the
# import, never `broom` (21 non-base recursive dependencies; nothing here
# calls a broom function). ------------------------------------------------

#' @importFrom generics tidy
#' @export
generics::tidy

#' @importFrom generics glance
#' @export
generics::glance

#' Tidy the fixed effects
#'
#' One row per fixed effect: `term`, `estimate`, `std.error`, `statistic`
#' (Wald z) and `p.value` (normal). Aliased coefficients are `NA` rows.
#'
#' @param x a [fastglmm] fit.
#' @param ... unused.
#' @return A `data.frame`.
#' @export
tidy.fastglmm <- function(x, ...) {
  wald <- .wald(x)
  data.frame(
    term = names(x$beta),
    estimate = unname(x$beta),
    std.error = unname(x$se),
    statistic = unname(wald$z),
    p.value = unname(wald$p),
    stringsAsFactors = FALSE
  )
}

#' One-row model summary
#'
#' `nobs`, `logLik`, `AIC`, `BIC`, `deviance` (`-2 * logLik`, the value the
#' summary's criterion row prints - not `$deviance`, the optimizer criterion),
#' `df.residual` (`nobs - df`) and `REML`. AIC/BIC are returned on REML fits
#' too: computing a number somebody asked for is not printing one they did
#' not, and the `REML` column says what it is. A REML AIC is comparable only
#' across models with identical fixed effects.
#'
#' @param x a [fastglmm] fit.
#' @param ... unused.
#' @return A one-row `data.frame`.
#' @export
glance.fastglmm <- function(x, ...) {
  ic <- .ml_criteria(x)
  data.frame(
    nobs = x$nobs,
    logLik = x$loglik,
    AIC = ic$AIC,
    BIC = ic$BIC,
    deviance = ic$deviance,
    df.residual = ic$df.residual,
    REML = x$reml
  )
}

# Wald z and normal p per fixed effect - summary() and tidy() both print
# these; one site so they cannot drift apart.
.wald <- function(object) {
  z <- object$beta / object$se
  list(z = z, p = 2 * stats::pnorm(-abs(z)))
}

# AIC / BIC / deviance (-2 logLik) / residual df - summary()'s criterion row
# and glance() both report these; one site so they cannot drift apart.
.ml_criteria <- function(object) {
  ll <- object$loglik
  list(
    AIC = -2 * ll + 2 * object$df,
    BIC = -2 * ll + log(object$nobs) * object$df,
    deviance = -2 * ll,
    df.residual = object$nobs - object$df
  )
}

#' @export
terms.fastglmm <- function(x, ...) {
  stop("terms() is not implemented: the formula machinery is Rust-side (one ",
       "parser shared with the Python port, R-port spec section 3), and a ",
       "synthesized R terms object could disagree with how the model was ",
       "actually built. formula() returns the formula string.", call. = FALSE)
}
