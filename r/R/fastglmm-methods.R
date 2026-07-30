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
    cat("Number of obs: ", x$nobs, ", groups: ",
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
#' The coefficient table carries Wald z statistics and normal-based p-values
#' for all families (the kernel surfaces no residual df, so there is no t).
#' There is deliberately **no** `AIC BIC logLik deviance` header line. On the
#' LMM paths `logLik()` is a REML criterion (see [logLik.fastglmm]), so a
#' printed AIC invites exactly the across-models comparison it is not valid for.
#' Call `logLik()`/`AIC()` explicitly instead.
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return An object of class `"summary.fastglmm"` with a `coefficients`
#'   matrix (`Estimate`, `Std. Error`, `z value`, `Pr(>|z|)`).
#' @export
summary.fastglmm <- function(object, ...) {
  z <- object$beta / object$se
  coefficients <- cbind(
    Estimate = object$beta,
    `Std. Error` = object$se,
    `z value` = z,
    `Pr(>|z|)` = 2 * stats::pnorm(-abs(z))
  )
  structure(list(
    object = object,
    coefficients = coefficients
  ), class = "summary.fastglmm")
}

#' @export
print.summary.fastglmm <- function(x,
                                   digits = max(3L, getOption("digits") - 3L),
                                   ...) {
  object <- x$object
  cat(.method_line(object), "\n")
  cat("Formula: ", object$formula, "\n", sep = "")
  cat(" Family: ", object$family$family, " (", object$family$link, ")\n\n",
      sep = "")
  if (length(object$varcorr)) {
    cat("Random effects:\n")
    print(VarCorr(object), digits = digits)
    sizes <- .group_sizes(object)
    cat("Number of obs: ", object$nobs, ", groups: ",
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
  if (object$family_name %in% c("gamma", "negativebinomial")) {
    label <- if (object$family_name == "gamma") {
      "Dispersion (phi, Pearson)"
    } else {
      "Shape (theta)"
    }
    cat(label, ": ", format(object$dispersion, digits = digits), "\n", sep = "")
  }
  cat("Optimizer evaluations: ", object$n_eval,
      ", converged: ", object$converged, "\n", sep = "")
  if (object$singular) cat("boundary (singular) fit: see help('isSingular')\n")
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

#' @export
print.VarCorr.fastglmm <- function(x,
                                   digits = max(3L,
                                                getOption("digits") - 3L),
                                   ...) {
  rows <- data.frame(Groups = character(), Name = character(),
                     Std.Dev. = character(), Corr = character(),
                     stringsAsFactors = FALSE)
  for (g in seq_along(x)) {
    sd <- attr(x[[g]], "stddev")
    corr <- attr(x[[g]], "correlation")
    for (i in seq_along(sd)) {
      corr_cells <- if (i > 1L) {
        paste(format(corr[i, seq_len(i - 1L)], digits = 3), collapse = " ")
      } else {
        ""
      }
      rows <- rbind(rows, data.frame(
        Groups = if (i == 1L) names(x)[g] else "",
        Name = names(sd)[i],
        Std.Dev. = format(sd[i], digits = digits),
        Corr = corr_cells,
        stringsAsFactors = FALSE
      ))
    }
  }
  sc <- attr(x, "sc")
  if (!is.null(sc) && is.finite(sc)) {
    rows <- rbind(rows, data.frame(
      Groups = "Residual", Name = "",
      Std.Dev. = format(sc, digits = digits), Corr = "",
      stringsAsFactors = FALSE
    ))
  }
  print(rows, row.names = FALSE, right = FALSE)
  invisible(x)
}

#' Residual scale
#'
#' The residual standard deviation for **gaussian** fits (`sqrt` of the
#' kernel's dispersion: `RSS/(n-p)` for a linear model, matching
#' `summary.lm`'s sigma; the REML residual variance for a mixed model,
#' matching `lme4::sigma`), `sqrt(phi)` for Gamma, and `1` for
#' binomial/poisson/negative-binomial (the scale is fixed, as in lme4).
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @export
sigma.fastglmm <- function(object, ...) {
  fam <- object$family_name
  if (fam %in% c("gaussian", "gamma")) sqrt(object$dispersion) else 1
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
  stop(what, " is engine-blocked: glmm::Fit does not surface the conditional ",
       "modes / linear predictor yet (engine spec ",
       "2026-07-15-engine-loglik-diagnostics). Fixed effects: fixef().",
       call. = FALSE)
}

#' Random-effect predictions (not available)
#'
#' Errors: conditional modes are computed inside the kernel but not surfaced
#' on `glmm::Fit` yet. The engine spec
#' `2026-07-15-engine-loglik-diagnostics` is what lifts this.
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @export
ranef <- function(object, ...) UseMethod("ranef")

#' @rdname ranef
#' @export
ranef.fastglmm <- function(object, ...) .engine_blocked("ranef()")

#' @export
predict.fastglmm <- function(object, ...) .engine_blocked("predict()")

#' @export
fitted.fastglmm <- function(object, ...) .engine_blocked("fitted()")

#' @export
residuals.fastglmm <- function(object, ...) .engine_blocked("residuals()")

#' @export
coef.fastglmm <- function(object, ...) {
  stop("coef() is not implemented: lme4's coef() means fixed + random ",
       "effects per group, which needs ranef() (engine-blocked); returning ",
       "fixed effects only would silently differ from what an lme4 user ",
       "expects from the same call. Use fixef().", call. = FALSE)
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

#' @export
terms.fastglmm <- function(x, ...) {
  stop("terms() is not implemented: the formula machinery is Rust-side (one ",
       "parser shared with the Python port, R-port spec section 3), and a ",
       "synthesized R terms object could disagree with how the model was ",
       "actually built. formula() returns the formula string.", call. = FALSE)
}
