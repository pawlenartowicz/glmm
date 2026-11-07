# S3 methods for "fastglmm" (R-port spec section 2). Plain S3, deliberately NOT an
# S4 merMod subclass - merMod has slots this engine cannot fill, and a
# half-filled merMod breaks every downstream package that trusts the class.
# Engine-blocked accessors error with the reason (spec section 4) rather than
# returning something silently different from what an lme4 user expects.

# vech-packed (column-major lower-triangular) covariance block -> list of
# per-dimension stddevs + full correlation matrix. Mirrors Rust
# `Fit::stddev_corr` (GLMM/glmm-rs/src/fit/mod.rs) - change together. 0-based r/c in
# idx() to keep the formula identical to the Rust side; +1 shifts into R.
# `group` (1-based, R's own indexing convention) names which varcorr block
# failed; mirrors the Python port's `Fit.stddev_corr` (glmm-py/glmm/__init__.py),
# which reports the same thing 0-based.
.stddev_corr <- function(vech, group = NA_integer_) {
  m <- length(vech)
  q <- as.integer((sqrt(1 + 8 * m) - 1) / 2)
  if (q * (q + 1L) / 2L != m) {
    label <- if (is.na(group)) "varcorr block" else sprintf("varcorr[[%d]]", group)
    stop(sprintf("%s is not a valid vech (length %d)", label, m), call. = FALSE)
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

# One-line description of what actually ran, for print/summary headers: the
# summary record's `method` (glmm-bind/src/summary.rs) with this port's name;
# print() has no summary record - change together.
.method_line <- function(object) {
  fam <- object$family_name
  mixed <- length(object$varcorr) > 0L
  if (!mixed) {
    if (fam == "gaussian") "Linear model fit by least squares [fastglmm]"
    else "Generalized linear model fit by IRLS [fastglmm]"
  } else if (fam == "gaussian") {
    if (object$reml) "Linear mixed model fit by REML [fastglmm]"
    else "Linear mixed model fit by maximum likelihood [fastglmm]"
  } else if (object$nAGQ > 1L) {
    paste0("Generalized linear mixed model fit by maximum likelihood ",
           "(Adaptive Gauss-Hermite Quadrature, nAGQ = ", object$nAGQ,
           ") [fastglmm]")
  } else {
    paste0("Generalized linear mixed model fit by maximum likelihood ",
           "(Laplace Approximation) [fastglmm]")
  }
}

# The title, then on an ML fit of a Gaussian mixed model the ML note, a line
# of its own directly under it and before any table. print() and summary()
# both open with it; the note cannot be switched off. The Python port's
# Summary._notes() prints the same - change together.
.print_title <- function(object) {
  cat(.method_line(object), "\n")
  if (identical(object$family_name, "gaussian") && length(object$varcorr) > 0L &&
      !object$reml) {
    cat("Note: p-values can be somewhat anticonservative.\n")
  }
}

#' @export
print.fastglmm <- function(x, digits = max(3L, getOption("digits") - 3L), ...) {
  .print_title(x)
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
#' (`REML criterion at convergence` on a REML fit, the `AIC BIC logLik deviance
#' df.resid` row on an ML fit), scaled residuals, random effects with
#' variances, the observation/group counts, the coefficient table, the
#' correlation of fixed effects, and a footer. The Python port's
#' `Fit.summary()` prints the same blocks - change together.
#'
#' The coefficient table follows lmerTest, `lm` and `glm`. A Gaussian mixed
#' model gets t tests on lmerTest's Satterthwaite degrees of freedom, with a
#' `df` column, on REML and ML fits. OLS and the Gamma and inverse-Gaussian
#' GLMs with an estimated dispersion get t tests on the residual df `n - p`,
#' printed in the footer (with `summary.lm`'s residual standard error and
#' R-squared lines on OLS). Binomial, Poisson and negative-binomial models, a
#' held dispersion and every GLMM get Wald z tests on the normal, as `glm` and
#' `glmer` do. A Gaussian mixed model fitted with `satterthwaite = FALSE`
#' gets z tests too, and the footer says the df were not computed. The t
#' p-values are the kernel's. The REML criterion is printed on REML fits (a
#' Gaussian mixed model fitted by REML, and an OLS fit made with
#' `REML = TRUE`); the AIC row is printed only on ML fits, because a REML AIC
#' invites exactly the across-models comparison it is not valid for.
#' `glance()` returns AIC/BIC on request either way.
#'
#' A Gaussian mixed model fitted with `REML = FALSE` is titled "Linear mixed
#' model fit by maximum likelihood", and `print()` and `summary()` print one
#' note on a line of its own directly under the title: "Note: p-values can be
#' somewhat anticonservative." It is a printed note, not a warning, and cannot
#' be switched off. REML, OLS, GLM and GLMM fits print no note.
#'
#' A mixed model gets one line under the number of observations with its
#' marginal and conditional R-squared and its adjusted intraclass correlation,
#' as `performance::r2_nakagawa` and `performance::icc` compute them, to
#' three decimals: `Marginal R2: 0.412, conditional R2: 0.705, adjusted ICC:
#' 0.498`. The values are computed when the summary is made. A Poisson or
#' negative binomial GLMM gets instead the line `R2 and ICC need a refit of the
#' null model: call r2(fit) and icc(fit)`, so a summary never refits. The line
#' is left out where the values are `NaN` (a Gamma model with a link other
#' than log, an inverse-Gaussian model, a fit that did not converge) and on a
#' fit saved by a version that did not keep its training data. [r2()] and
#' [icc()] give all four values.
#'
#' On a weighted fit, the lines before `Optimizer evaluations` give the
#' weight power when it was estimated (`Weight power: 0.734 (estimated)`) or
#' fixed at a value other than 1; an estimate outside `[-1, 2]`, the range the
#' simulations behind it searched, adds a line saying so (a line, not a
#' warning: on weights that vary little it happens with nothing wrong); and on
#' a model standard error of an estimated power, a line that the standard
#' errors treat it as known (on an ML fit, also that the simulations
#' estimated it by REML). A robust fit's footnote names CR2 (with its
#' grouping factor) or HC2 and its df.
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return An object of class `"summary.fastglmm"` with `coefficients`
#'   (`Estimate`, `Std. Error`, `df` on a Gaussian mixed model, then
#'   `t value` and `Pr(>|t|)`, or `z value` and `Pr(>|z|)`), `criterion`,
#'   `scaled_residuals` (`NULL` when the fit did not converge),
#'   `corr_fixed` (`NULL` when there is one coefficient) and `effect_sizes`
#'   (the named values `marginal_r2`, `conditional_r2` and `adjusted_icc` of
#'   the R-squared line, `NULL` when it is left out or points to [r2()]) and
#'   `r2_need_refit` (`TRUE` when it points to [r2()] and [icc()]) and
#'   `weight_power_lines` (the weight power's lines, empty when there are
#'   none).
#' @export
summary.fastglmm <- function(object, ...) {
  # A fit saved before .kernel_parts cannot lower its rows again, so it has
  # no R-squared line.
  e <- if (is.null(object$.kernel_parts)) list() else .effect_sizes(object, null_refit = FALSE)
  s <- .summary_record(object, if (length(e)) e, is.null(e))
  coefficients <- cbind(Estimate = s$estimate, `Std. Error` = s$std_error)
  if (!is.null(s$coef_df)) coefficients <- cbind(coefficients, df = s$coef_df)
  coefficients <- cbind(coefficients, s$statistic, s$p_value)
  colnames(coefficients)[ncol(coefficients) - 1:0] <-
    c(paste(s$test, "value"), sprintf("Pr(>|%s|)", s$test))
  rownames(coefficients) <- names(object$beta)
  scaled <- s$scaled_residuals
  if (!is.null(scaled)) names(scaled) <- c("Min", "1Q", "Median", "3Q", "Max")
  corr_fixed <- s$corr_fixed
  if (!is.null(corr_fixed)) dimnames(corr_fixed) <- dimnames(object$vcov)
  effect_sizes <- c(marginal_r2 = s$r2_marginal, conditional_r2 = s$r2_conditional,
                    adjusted_icc = s$icc_adjusted)
  if (length(effect_sizes) != 3L || !all(is.finite(effect_sizes))) effect_sizes <- NULL
  structure(list(
    object = object,
    coefficients = coefficients,
    criterion = unlist(s$criterion),
    scaled_residuals = scaled,
    corr_fixed = corr_fixed,
    effect_sizes = effect_sizes,
    r2_need_refit = s$r2_need_refit,
    residual_df = if (!is.null(s$residual_df)) as.integer(s$residual_df),
    dispersion_label = s$dispersion_label,
    weight_power_lines = as.character(s$weight_power_lines),
    footnote = s$footnote
  ), class = "summary.fastglmm")
}

# The native summary record of a fit, which summary() and tidy() read.
# `%||%` keeps objects saved before a field existed printable.
.summary_record <- function(object, effect_sizes = NULL, need_refit = FALSE) {
  .glmm_call("summary", list(
    fit = list(
      formula = object$formula, family = object$family_name, link = object$link,
      reml = isTRUE(object$reml), nagq = object$nAGQ, nobs = object$nobs, df = object$df,
      loglik = as.double(object$loglik), names = as.character(names(object$beta)),
      beta = unname(as.double(object$beta)), se = unname(as.double(object$se)),
      vcov = as.double(t(object$vcov)), aliased = unname(as.logical(object$aliased)),
      df_method = object$df_method %||% "none",
      coef_df = .optional(object$coef_df), coef_p = .optional(object$coef_p),
      dispersion = as.double(object$dispersion),
      dispersion_held = .optional(object$dispersion_held),
      varcorr = lapply(object$varcorr, as.double),
      re_group_names = as.character(object$re_group_names),
      re_group_terms = lapply(object$re_group_terms, as.character),
      re_term_index = object$re_term_index,
      y = as.double(object$y), fitted = as.double(object$fitted),
      weights = .optional(object$weights),
      weight_power = as.double(object$weight_power %||% 1),
      weight_power_estimated = isTRUE(object$estimated_weight_power),
      n_eval = object$n_eval, converged = isTRUE(object$converged),
      singular = isTRUE(object$singular),
      rss = object$rss %||% NaN, r_squared = object$r_squared %||% NaN,
      adj_r_squared = object$adj_r_squared %||% NaN),
    effect_sizes = effect_sizes, need_refit = need_refit))
}

#' @export
print.summary.fastglmm <- function(x,
                                   digits = max(3L, getOption("digits") - 3L),
                                   ...) {
  object <- x$object
  .print_title(object)
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
        "\n", sep = "")
  } else {
    cat("Number of obs:", object$nobs, "\n")
  }
  # Mirrors Summary._effect_size_lines() (glmm-py/glmm/summary.py) - change
  # together.
  e <- x$effect_sizes
  if (isTRUE(x$r2_need_refit)) {
    cat("R2 and ICC need a refit of the null model: call r2(fit) and icc(fit)\n")
  } else if (!is.null(e)) {
    cat(sprintf("Marginal R2: %.3f, conditional R2: %.3f, adjusted ICC: %.3f\n",
                e[["marginal_r2"]], e[["conditional_r2"]], e[["adjusted_icc"]]))
  }
  cat("\n")
  cat("Fixed effects:\n")
  # The df column is zapped as lme4's print.summary.merMod zaps it.
  if ("df" %in% colnames(x$coefficients)) {
    stats::printCoefmat(x$coefficients, digits = digits, na.print = "NA", zap.ind = 3L)
  } else {
    stats::printCoefmat(x$coefficients, digits = digits, na.print = "NA")
  }
  if (any(object$aliased)) {
    cat("(", sum(object$aliased),
        " coefficient(s) not defined because of singularities)\n", sep = "")
  }
  rdf <- x$residual_df
  if (!is.null(rdf) && identical(object$family_name, "gaussian")) {
    # summary.lm's two lines, printed as print.summary.lm prints them, except
    # that a value that is not finite prints NA. The Python port prints the
    # same - change together.
    num <- function(v, f) if (is.finite(v)) f(v) else "NA"
    sigma <- if (rdf > 0) sqrt(object$rss / rdf) else NaN
    cat("\nResidual standard error: ", num(sigma, function(v) format(signif(v, digits))),
        " on ", rdf, " degrees of freedom\n", sep = "")
    cat("Multiple R-squared: ", num(object$r_squared, function(v) formatC(v, digits = digits)))
    cat(",\tAdjusted R-squared: ",
        num(object$adj_r_squared, function(v) formatC(v, digits = digits)), "\n")
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
  label <- x$dispersion_label
  if (!is.null(label)) {
    cat(label, ": ", format(object$dispersion, digits = digits),
        if (!is.null(rdf)) paste(" on", rdf, "degrees of freedom"), "\n", sep = "")
  }
  # Mirrors Summary.weight_power_lines (glmm-py/glmm/summary.py) - change together.
  if (length(x$weight_power_lines)) cat(x$weight_power_lines, sep = "\n")
  cat("Optimizer evaluations: ", object$n_eval,
      ", converged: ", object$converged, "\n", sep = "")
  if (object$singular) cat("boundary (singular) fit: see help('isSingular')\n")
  footer <- if (identical(.df_method(object), "disabled")) .DF_NOT_COMPUTED else x$footnote
  if (!is.null(footer)) cat(footer, "\n", sep = "")
  # Mirrors Summary.text() (glmm-py/glmm/summary.py) - change together. NROW()
  # keeps objects saved before `warnings` existed printable.
  w <- object$warnings
  if (NROW(w)) {
    cat("\nWarnings:\n")
    cat(sprintf("%s%s: %s. %s\n", toupper(substr(w$tier, 1L, 1L)), substring(w$tier, 2L),
                w$title, w$message), sep = "")
  }
  invisible(x)
}

#' @importFrom nlme fixef
#' @export
nlme::fixef

#' Extract fixed effects
#'
#' Adds a method to `nlme::fixef`, the generic `lme4::fixef` also uses (lme4
#' adds methods to nlme's generic rather than declaring its own), so a
#' [fastglmm] fit and an lme4 `merMod` fit dispatch through the same generic
#' regardless of load order, and `fixef(fit)` works without an explicit
#' `library(nlme)`.
#'
#' @param object a [fastglmm] fit.
#' @param ... unused.
#' @return Named numeric vector of fixed-effect estimates; aliased
#'   (rank-deficient) coefficients are `NA`, mirroring `lm`/lme4.
#' @export
fixef.fastglmm <- function(object, ...) object$beta

#' @export
vcov.fastglmm <- function(object, ...) object$vcov

#' @importFrom nlme VarCorr
#' @export
nlme::VarCorr

#' Variance components on the SD/correlation scale
#'
#' Returns one covariance matrix per grouping (declaration order), each with
#' `"stddev"` and `"correlation"` attributes, printed lme4-shaped. This is the
#' load-bearing accessor for parameter-recovery use: `attr(vc$g, "stddev")`
#' are the tau estimates and `attr(vc$g, "correlation")` the rho estimates,
#' straight from the kernel's `varcorr` (validated against lme4's `VarCorr`).
#'
#' For a **gaussian** mixed fit a `Residual` row is printed after the group
#' components, as in lme4, carrying `sigma()` (the residual standard deviation
#' of the fitted criterion, REML or ML with `REML = FALSE`, that the kernel
#' reports as `dispersion`).
#'
#' Adds a method to `nlme::VarCorr`, the generic `lme4::VarCorr` also uses
#' (lme4 adds methods to nlme's generic rather than declaring its own), so a
#' [fastglmm] fit and an lme4 `merMod` fit dispatch through the same generic
#' regardless of load order, and `VarCorr(fit)` works without an explicit
#' `library(nlme)`.
#'
#' @param x a [fastglmm] fit.
#' @param ... unused.
#' @return A list of class `"VarCorr.fastglmm"`, one named element per
#'   grouping.
#' @rdname VarCorr.fastglmm
#' @export
VarCorr.fastglmm <- function(x, ...) {
  out <- lapply(seq_along(x$varcorr), function(g) {
    sc <- .stddev_corr(x$varcorr[[g]], group = g)
    terms <- x$re_group_terms[[g]]
    v <- outer(sc$stddev, sc$stddev) * sc$correlation
    dimnames(v) <- list(terms, terms)
    attr(v, "stddev") <- stats::setNames(sc$stddev, terms)
    attr(v, "correlation") <- sc$correlation
    # The term of each column (1-based); the print method shows a
    # correlation only within one term.
    attr(v, "term_index") <- x$re_term_index[[g]]
    v
  })
  # A `no_optimum` exit carries an empty `varcorr` next to its grouping names;
  # `names<-` on a zero-length list rejects a non-empty name vector.
  if (length(out)) names(out) <- x$re_group_names
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
#' @rdname VarCorr.fastglmm
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
    term <- attr(x[[g]], "term_index")
    for (i in seq_along(sd)) {
      # As lme4: a correlation only between two columns of one term, whose
      # columns are contiguous; two terms are uncorrelated by the model.
      same <- seq_len(i - 1L)
      if (i <= length(term)) same <- same[term[same] == term[i]]
      corr_cells <- if (length(same)) {
        paste(format(round(corr[i, same], 2), nsmall = 2), collapse = " ")
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
#' `summary.lm`'s sigma; for a mixed model the residual variance of the
#' fitted criterion, REML or ML with `REML = FALSE`, matching `lme4::sigma`),
#' `sqrt(phi)` for Gamma and inverse-Gaussian, and
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
#' `formula()` returns the model formula as given, as a formula object, so
#' `update(m, . ~ . + z)` works. The R side never builds a `terms` object
#' (the parser is Rust-side), and synthesizing one would disagree with how
#' the model was actually built, which is also why `terms()` errors.
#'
#' @param x,formula,object a [fastglmm] fit.
#' @param ... unused.
#' @export
formula.fastglmm <- function(x, ...) {
  # The fit keeps the formula as a string (print, summary, anova and the
  # comparison read it). The global environment is safe because the formula
  # reaches Rust as a string and every variable is read from `data`; any
  # calling frame could hold a fit, which update() would then carry along in
  # the new fit's call.
  stats::as.formula(x$formula, env = globalenv())
}

#' @rdname formula.fastglmm
#' @export
family.fastglmm <- function(object, ...) object$family

#' @rdname formula.fastglmm
#' @export
model.frame.fastglmm <- function(formula, ...) formula$frame

#' Confidence intervals: Wald, or profile likelihood
#'
#' `method = "Wald"` (the default) gives intervals off the Wald standard errors
#' that follow [summary()][summary.fastglmm] coefficient by coefficient: the t
#' quantile on the row's degrees of freedom where the summary tests with t
#' (Satterthwaite df on a Gaussian mixed model, `n - p` on OLS and the Gamma and
#' inverse-Gaussian GLMs, as [stats::confint.lm]), the normal quantile where it
#' tests with z. On a Gaussian mixed model fitted with `satterthwaite = FALSE`
#' the intervals are normal and a message says the df were not computed. The
#' Python port's `Fit.confint()` computes the same intervals through the same
#' kernel function.
#'
#' `method = "profile"` gives profile-likelihood intervals of a Gaussian mixed
#' model, as lme4's `confint(method = "profile")` computes them: one row per
#' random-effect SD and correlation, the residual SD and coefficient, named as
#' lme4 names them (`"sd_(Intercept)|Subject"`, `"cor_Days.(Intercept)|Subject"`,
#' `"sigma"`, then the coefficients). Each parameter is held at points on each
#' side of its estimate while the others are re-optimized, and a bound is where
#' the signed root of the deviance change reaches the normal quantile of
#' `level`. The profiles are of the ML deviance: a REML fit is refitted by ML
#' first, as lme4 does. An SD estimated at 0 has 0 as its lower bound, and a
#' correlation whose profile stays inside the cutoff up to -1 or 1 has that
#' limit. An undetermined bound is `NA`. A GLMM, a model without random
#' effects, and a design with an aliased column are refused with an error. The
#' Python port's `Fit.profile_confint()` computes the same intervals.
#'
#' `method = "boot"` is not available: bootstrap intervals need refitting
#' infrastructure not built yet.
#'
#' @param object a [fastglmm] fit.
#' @param parm the rows to include (default all). With `method = "Wald"`,
#'   coefficient names or indices; with `method = "profile"`, row names, or
#'   lme4's shortcuts `"theta_"` (the SDs, correlations and `"sigma"`) and
#'   `"beta_"` (the coefficients).
#' @param level confidence level.
#' @param method `"Wald"` or `"profile"`.
#' @param ... unused.
#' @return A matrix of lower and upper bounds. With `method = "Wald"`, one row
#'   per coefficient, aliased coefficients `NaN` rows; with
#'   `method = "profile"`, one row per profiled parameter in glmm's order
#'   (per grouping the SDs and correlations column-major, each SD followed by
#'   its correlations, then `sigma`, then the coefficients; lme4 2.0 lists a
#'   block's SDs before its correlations, so match rows to lme4's by name),
#'   `NA` for an undetermined bound.
#' @export
confint.fastglmm <- function(object, parm, level = 0.95,
                             method = c("Wald", "profile"), ...) {
  if (identical(method, c("Wald", "profile"))) method <- "Wald"
  if (!(is.character(method) && length(method) == 1L && method %in% c("Wald", "profile"))) {
    stop("confint(method = ", paste(deparse(method), collapse = ""), ") is not available; ",
         "method = \"Wald\" and method = \"profile\" are supported",
         call. = FALSE)
  }
  if (identical(method, "profile")) {
    return(.profile_confint(object, if (missing(parm)) NULL else as.character(parm), level))
  }
  est <- object$beta
  if (missing(parm)) parm <- names(est)
  # A fit saved before `coef_df` existed has no df: normal intervals, as such a
  # fit's summary() tests with z.
  coef_df <- object$coef_df %||% rep(NaN, length(est))
  ci <- .glmm_call("confint", list(beta = as.double(est), se = as.double(object$se),
                                   coef_df = as.double(coef_df), level = .level(level)))
  out <- cbind(stats::setNames(ci$lower, names(est)),
               stats::setNames(ci$upper, names(est)))[parm, , drop = FALSE]
  colnames(out) <- .ci_colnames(level)
  if (identical(.df_method(object), "disabled")) message(.DF_NOT_COMPUTED)
  out
}

# confint()'s column names, "2.5 %" and "97.5 %" at level 0.95, as stats
# names them.
.ci_colnames <- function(level) {
  a <- (1 - level) / 2
  paste(format(100 * c(a, 1 - a), trim = TRUE, scientific = FALSE, digits = 3), "%")
}

# confint(method = "profile"): the native profile_intervals on the fit's
# training columns, as lme4's bounds matrix; NaN (an undetermined bound)
# becomes NA.
.profile_confint <- function(object, parm, level) {
  r <- .glmm_call("profile_intervals", c(.training_columns(object), list(
    reml = isTRUE(object$reml), weight_power = as.double(object$weight_power %||% 1),
    weight_power_estimated = isTRUE(object$estimated_weight_power),
    level = .level(level), parm = parm)))
  out <- cbind(r$lower, r$upper)
  out[is.nan(out)] <- NA_real_
  dimnames(out) <- list(r$names, .ci_colnames(level))
  out
}

#' Is the fit singular?
#'
#' `TRUE` iff the fit converged onto the variance-component boundary - the
#' same condition `lme4::isSingular` reports (the kernel computes it; see
#' `glmm::Fit::singular`).
#'
#' Two packages, one generic name, and neither one may depend on the other.
#' `isSingular` is lme4's own generic (unlike `fixef`/`ranef`/`VarCorr`
#' above, it has no shared upstream generic such as nlme's to add a method
#' to). fastglmm declares its own `isSingular` generic too, so
#' `isSingular(fit)` on a [fastglmm] fit works even with lme4 not installed
#' at all.
#'
#' Whichever package is attached last wins the bare `isSingular` name in the
#' search path, so both directions still need to work. When lme4's generic
#' is the one left in scope, this method also being registered directly on
#' lme4's generic (by delayed S3 registration - it only activates if and
#' when lme4 happens to be loaded) is what makes a [fastglmm] fit still
#' dispatch correctly. When fastglmm's generic is the one left in scope
#' instead, `isSingular.default` below is what makes a `merMod` fit still
#' dispatch correctly - by forwarding to lme4's generic **only for a
#' `merMod` object, and only when the lme4 namespace is already loaded**,
#' never by loading it: a `merMod` object cannot exist without lme4 already
#' loaded in the first place, so this never needs to load anything, and this
#' package still does not depend on lme4 (not even as a Suggests - the one
#' place lme4 is used live is the hand-run comparison in `tools/`). The
#' `merMod` check also matters on its own: lme4's S4 default method calls
#' `UseMethod("isSingular")`, which lands back here for any object that is
#' neither a `fastglmm` fit nor a `merMod` - forwarding unconditionally
#' would call lme4's generic again and recurse until the C stack fills up,
#' so anything else still falls through to the plain error below.
#'
#' @param x a [fastglmm] fit.
#' @param ... unused.
#' @export
isSingular <- function(x, ...) UseMethod("isSingular")

#' @rdname isSingular
#' @export
#' @exportS3Method lme4::isSingular
isSingular.fastglmm <- function(x, ...) x$singular

#' @rdname isSingular
#' @export
isSingular.default <- function(x, ...) {
  if (isNamespaceLoaded("lme4") && inherits(x, "merMod")) {
    getExportedValue("lme4", "isSingular")(x, ...)
  } else {
    stop("isSingular() does not know how to handle an object of class '",
         class(x)[[1L]], "'. This package's own method handles a fastglmm ",
         "fit directly; any other class needs its own package loaded ",
         "first (a merMod fit needs lme4, for example).", call. = FALSE)
  }
}

# -- Conditional modes, predictions and simulated responses. ---------------

#' @importFrom nlme ranef
#' @export
nlme::ranef

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
#' With `condVar = TRUE` (the default) each data frame carries a `postVar`
#' attribute, as lme4's does: an array of terms x terms x levels holding each
#' level's conditional covariance of its random effects given the data, at the
#' fitted parameters. A fit made with `condvar = FALSE` has none, and some
#' layouts form none: a model with an extra grouping that has a random slope,
#' one over the dense size limit (both fitted by the sparse solver), and a GLMM
#' on the packed layout. There an explicit `condVar = TRUE` is an error, and
#' the default leaves the attribute off.
#'
#' Adds a method to `nlme::ranef`, the generic `lme4::ranef` also uses (lme4
#' adds methods to nlme's generic rather than declaring its own), so a
#' [fastglmm] fit and an lme4 `merMod` fit dispatch through the same generic
#' regardless of load order, and `ranef(fit)` works without an explicit
#' `library(nlme)`.
#'
#' @param object a [fastglmm] fit.
#' @param condVar `TRUE` (the default) to attach the conditional variances as
#'   a `postVar` attribute, `FALSE` for the modes alone.
#' @param ... unused.
#' @return A named `list` of `data.frame`s, empty for a fit with no random
#'   effects or one that did not converge.
#' @export
ranef.fastglmm <- function(object, condVar = TRUE, ...) {
  if (!(is.logical(condVar) && length(condVar) == 1L && !is.na(condVar))) {
    stop(sprintf("condVar must be TRUE or FALSE, got %s", .arg_repr(condVar)), call. = FALSE)
  }
  asked <- !missing(condVar)
  blocks <- object$ranef_blocks
  out <- lapply(blocks, function(b) {
    q <- length(b$terms)
    n <- length(b$levels)
    # `values` arrives row-major (level-major, terms within a level), which is
    # byrow = TRUE for R's column-major matrix().
    m <- matrix(as.double(b$values), nrow = n, ncol = q, byrow = TRUE,
                dimnames = list(b$levels, b$terms))
    re <- as.data.frame(m, stringsAsFactors = FALSE)
    if (condVar) {
      cv <- as.double(b$condvar %||% double())
      if (length(cv)) {
        attr(re, "postVar") <- .post_var(cv, q, n)
      } else if (asked && n) {
        stop("this fit has no conditional variances: it was fitted with condvar = FALSE, ",
             "or its random effects were fitted on a layout that does not form them (an ",
             "extra grouping with a random slope, a model over the dense size limit, or a ",
             "GLMM on the packed layout). Refit with condvar = TRUE, or use ",
             "condVar = FALSE for the modes alone.", call. = FALSE)
      }
    }
    re
  })
  names(out) <- vapply(blocks, function(b) b$group, character(1))
  out
}

# lme4's postVar, terms x terms x levels, from `cv`: per level the vech of its
# q x q block (column-major lower triangle), level after level.
.post_var <- function(cv, q, n) {
  v <- q * (q + 1L) / 2L
  low <- lower.tri(diag(q), diag = TRUE)
  out <- array(0, c(q, q, n))
  for (l in seq_len(n)) {
    m <- matrix(0, q, q)
    # A logical index walks the matrix column-major, as the vech is packed.
    m[low] <- cv[(l - 1L) * v + seq_len(v)]
    m[upper.tri(m)] <- t(m)[upper.tri(m)]
    out[, , l] <- m
  }
  out
}

#' Predictions
#'
#' Predictions on new rows, as `lme4::predict.merMod`: the linear predictor at
#' the fixed effects and, by default, the conditional modes of every grouping
#' (`re.form = NULL`), or the fixed effects alone (`re.form = NA`).
#'
#' `newdata` is lowered as the model frame was: each factor coded with the
#' fit's levels by label (a level of a fixed factor the fit did not see is an
#' error), the fit's column order and transforms, and an `offset()` term
#' evaluated on the new rows. A grouping level the fit did not see gets a
#' random effect of zero for that grouping, with a warning of class
#' `fastglmm_new_grouping_levels`; `allow.new.levels = FALSE` makes it an
#' error, as it is by default in lme4. A `newdata` without the grouping
#' columns predicts with `re.form = NA` only. A row with a missing value is
#' refused. A model fitted with the `offset=` argument predicts its own rows
#' only: give the offset as an `offset()` term in the formula to predict new
#' rows.
#'
#' The standard error is lme4's (`se.fit = TRUE`), with the variance
#' parameters held at their estimates; on a GLMM it uses the fit's
#' fixed-effect covariance (`wald.se`), where lme4 uses the RX one. On the
#' response scale it is the delta method's, and the interval is the inverse
#' link of the link-scale one. A fit with no conditional variances (see
#' [ranef()][ranef.fastglmm]) has a standard error in `re.form = NA` mode
#' only. On a robust fit (`$se_type` `"robust"`) the standard error and
#' interval are model-based, as on the same fit with `se_type = "model"`,
#' with a `"fastglmm_predict_se_model_based"` warning: their formula holds
#' only with the model's covariance.
#'
#' @param object a [fastglmm] fit.
#' @param newdata a data frame of new rows with the columns the formula reads
#'   other than the response; missing for the model frame's rows.
#' @param re.form `NULL` (the default) or `NA`; see above.
#' @param type `"link"` (the default) for the linear predictor, `"response"`
#'   for the mean.
#' @param allow.new.levels whether a grouping level the fit did not see is
#'   accepted (the default) or an error.
#' @param se.fit whether to return the standard error and the interval at
#'   `level`.
#' @param level the coverage of the interval and of the future bounds.
#' @param future whether to return the bounds of a future observation on the
#'   response scale (a non-Gaussian mixed model only): the family's quantiles
#'   mixed over the uncertainty of the linear predictor.
#' @param ... unused.
#' @return A numeric vector named by the rows; with `se.fit` or `future`, a
#'   list: `fit`, then `se.fit`, `lower` and `upper` with `se.fit`, and
#'   `future.lower` and `future.upper` with `future`.
#' @export
predict.fastglmm <- function(object, newdata, re.form = NULL, type = c("link", "response"),
                             allow.new.levels = TRUE, se.fit = FALSE, level = 0.95,
                             future = FALSE, ...) {
  type <- match.arg(type)
  population <- if (is.null(re.form)) {
    FALSE
  } else if (length(re.form) == 1L && !inherits(re.form, "formula") && is.na(re.form)) {
    TRUE
  } else {
    stop("re.form must be NULL or NA: the random effects of every grouping, or none",
         call. = FALSE)
  }
  for (flag in c("allow.new.levels", "se.fit", "future")) {
    v <- get(flag)
    if (!(is.logical(v) && length(v) == 1L && !is.na(v))) {
      stop(sprintf("%s must be TRUE or FALSE, got %s", flag, .arg_repr(v)), call. = FALSE)
    }
  }
  train <- .training_columns(object)
  if (missing(newdata) || is.null(newdata)) {
    has_new <- FALSE
    new <- list()
    rows <- rownames(object$frame)
  } else {
    newdata <- as.data.frame(newdata)
    f <- stats::as.formula(object$formula)
    vars <- intersect(setdiff(all.vars(f), all.vars(f[[2L]])), names(newdata))
    if (anyNA(newdata[vars])) {
      stop("newdata has a missing value in a column the formula uses; drop those rows first",
           call. = FALSE)
    }
    has_new <- TRUE
    new <- .marshal(newdata, vars, .offset_vars(f))
    rows <- rownames(newdata)
  }
  r <- .glmm_call("predict", c(train, list(
    parts = .stored_numbers(object), new_numeric = new$numeric,
    new_factors = new$factors, n_rows = as.double(length(rows)), scale = type,
    population = population, allow_new_levels = allow.new.levels, se = se.fit,
    level = .level(level), future = future,
    # New rows are the caller's own: newdata with a missing value is refused.
    row_index = if (has_new) integer() else .row_index(object), row_base = 1L)))
  for (note in r$notes) {
    if (identical(note$kind, "predict_se_model_based")) {
      # Mirrors the Python port's Fit.predict wording - change together.
      warning(warningCondition(paste(
        "The standard errors and intervals are model-based, not robust: their formula holds",
        "only with the model's covariance."),
        class = c("fastglmm_predict_se_model_based", "fastglmm_diagnostic"), call = NULL))
    }
    if (!identical(note$kind, "new_grouping_levels")) next
    parts <- strsplit(note$detail, ": ", fixed = TRUE)[[1L]]
    name <- object$re_group_names[[as.integer(parts[[1L]]) + 1L]]
    shown <- strsplit(parts[[2L]], ", ", fixed = TRUE)[[1L]]
    n <- length(shown)
    shown <- paste0(paste(shown[seq_len(min(n, 10L))], collapse = ", "),
                    if (n > 10L) sprintf(", and %d more", n - 10L) else "")
    # Mirrors the Python port's Fit.predict wording - change together.
    warning(warningCondition(sprintf(
      "%s of newdata %s a level of %s the fit did not see (rows %s). Their random effect for %s is 0.",
      .count(n, "row"), if (n == 1L) "has" else "have", name, shown, name),
      class = c("fastglmm_new_grouping_levels", "fastglmm_diagnostic"), call = NULL))
  }
  named <- function(v) stats::setNames(as.double(v), rows)
  fit <- named(r$fit)
  if (!se.fit && !future) return(fit)
  out <- list(fit = fit)
  if (se.fit) {
    out$se.fit <- named(r$se)
    out$lower <- named(r$lower)
    out$upper <- named(r$upper)
  }
  if (future) {
    out$future.lower <- named(r$future_lower)
    out$future.upper <- named(r$future_upper)
  }
  out
}

#' Simulated responses
#'
#' `nsim` draws of the response on the model frame's rows, as
#' `lme4::simulate.merMod`: each draw takes new random effects from the fitted
#' covariance, then the family's draw around the conditional mean with the
#' fitted dispersion; a binomial draw is a proportion of the row's trials.
#' The draws come from the crate's own generator, so a seed gives the same
#' draws on every platform.
#'
#' @param object a [fastglmm] fit.
#' @param nsim the number of draws, a whole number of at least 1.
#' @param seed a whole number from 0 to 2^53, or `NULL` (the default) to take
#'   one from R's random number generator (so `set.seed()` fixes it).
#' @param ... unused.
#' @return A data frame of `nobs(object)` rows and columns `sim_1`, ...,
#'   `sim_<nsim>`, with the seed used as its `"seed"` attribute.
#' @export
simulate.fastglmm <- function(object, nsim = 1, seed = NULL, ...) {
  whole <- function(x, max) is.numeric(x) && length(x) == 1L && is.finite(x) &&
    x == round(x) && x >= 0 && x <= max
  if (!(whole(nsim, 2^53) && nsim >= 1)) {
    stop(sprintf("nsim must be a whole number of at least 1, got %s", .arg_repr(nsim)),
         call. = FALSE)
  }
  if (is.null(seed)) {
    seed <- sample.int(.Machine$integer.max, 1L)
  } else if (!whole(seed, 2^53)) {
    stop(sprintf("seed must be NULL or a whole number from 0 to 2^53, got %s", .arg_repr(seed)),
         call. = FALSE)
  }
  draws <- .glmm_call("simulate", c(.training_columns(object), list(
    parts = .stored_numbers(object), nsim = as.double(nsim), seed = as.double(seed),
    row_index = .row_index(object), row_base = 1L)))
  out <- as.data.frame(matrix(draws, nrow = object$nobs, ncol = nsim))
  names(out) <- paste0("sim_", seq_len(nsim))
  rownames(out) <- rownames(object$frame)
  attr(out, "seed") <- seed
  out
}

# Each model frame row's 1-based position in the caller's data, or integer()
# (the frame's own rows, counted from 1) on a fit saved before it was kept.
.row_index <- function(object) as.integer(object$.row_index %||% integer())

# The fit's training data as fastglmm() passed it to the native code: the
# formula, the model frame marshalled again, the family, the weights and the
# offset.
.training_columns <- function(object) {
  if (is.null(object$.kernel_parts)) {
    stop("this fit was made by a fastglmm version that did not keep what predict(), ",
         "simulate(), pairwise(), r2() and icc() read. Refit the model with this version.",
         call. = FALSE)
  }
  f <- stats::as.formula(object$formula)
  frame <- object$frame
  cols <- .marshal(frame, all.vars(f), c(all.vars(f[[2L]]), .offset_vars(f)))
  list(formula = object$formula, numeric = cols$numeric, factors = cols$factors,
       family = object$family_name, link = object$link,
       weights = .optional(frame[["(weights)"]]), offset = .optional(frame[["(offset)"]]))
}

# The fit's numbers as the native predict, simulate and effect_sizes take
# them (`parts`).
.stored_numbers <- function(object) {
  c(list(beta = as.double(object$beta), vcov = as.double(t(object$vcov)),
         model_vcov = .model_vcov_arg(object), varcorr = object$varcorr, dispersion = as.double(object$dispersion),
         weight_power = as.double(object$weight_power %||% 1)),
    object$.kernel_parts)
}

# The native effect_sizes of a fit: list(marginal, conditional, adjusted,
# unadjusted), or NULL on a Poisson or negative binomial GLMM when null_refit
# is FALSE (the values need a refit of the null model).
.effect_sizes <- function(object, null_refit) {
  .glmm_call("effect_sizes", c(.training_columns(object), list(
    parts = .stored_numbers(object), aliased = unname(as.logical(object$aliased)),
    converged = isTRUE(object$converged), objective = object$objective,
    objective_nodes = as.integer(object$objective_nodes), null_refit = null_refit)))
}

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
residuals.fastglmm <- function(object, type = "response", ...) {
  if (!(is.character(type) && length(type) == 1L && type %in% c("response", "pearson"))) {
    stop(sprintf("type must be 'response' or 'pearson', got %s", .arg_repr(type)),
         call. = FALSE)
  }
  mu <- as.double(object$fitted)
  if (!length(mu)) {
    stop("residuals are unavailable: the fit did not converge, so no fitted values are available",
         call. = FALSE)
  }
  r <- if (type == "pearson") {
    .glmm_call("pearson_residuals", list(
      family = object$family_name, y = as.double(object$y), fitted = mu,
      weights = .optional(object$weights), dispersion = as.double(object$dispersion),
      weight_power = as.double(object$weight_power %||% 1)))
  } else {
    as.double(object$y) - mu
  }
  names(r) <- rownames(object$frame)
  r
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
#' On `stats::logLik`'s scale, which [AIC()][AIC.fastglmm] and `BIC()` read.
#' This is `glmm::Fit::loglik`, not `$deviance` - the latter is the optimizer
#' criterion. For an LMM it is lme4's REMLcrit (on an ML fit, lme4's ML
#' deviance) minus a data-independent constant. For a GLMM it differs from
#' `-2*logLik` by a data-only saturated constant on binomial/Poisson fits, and
#' equals `-2*logLik` exactly on Gamma and negative-binomial fits.
#'
#' On a REML fit (a Gaussian mixed model fitted with the default `REML`, or an
#' OLS fit made with `REML = TRUE`) the value is the **REML criterion**,
#' matching `lme4::logLik` on a REML fit (and `logLik(lm, REML = TRUE)` on
#' OLS), and the returned object carries `REML = TRUE`. It stays the REML value
#' because that is the quantity for comparing random-effect structures with the
#' same fixed effects. REML criteria are comparable only between models with
#' identical fixed effects: `AIC()`, `BIC()` and `anova(m0, m1)` refuse REML
#' fits with different fixed effects. To compare fixed effects, use
#' [compare()], which fits both models by ML, or fit both models with
#' `REML = FALSE`; an ML fit returns the ML log-likelihood, with `REML = FALSE`.
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

#' AIC and BIC of fastglmm fits
#'
#' On one fit, the value, as [stats::AIC()] and [stats::BIC()] compute it from
#' [logLik()][logLik.fastglmm]: `-2 * logLik + k * df`, with `k = 2` for AIC
#' and `log(nobs)` for BIC. On a REML fit (a Gaussian mixed model fitted by
#' REML, or an OLS fit made with `REML = TRUE`) the value is built on the REML
#' log-likelihood, as lme4's `AIC()` on a REML fit.
#'
#' On several fits, the `stats` table (`df` and `AIC` or `BIC`), except for the
#' two comparisons that are always wrong, which stop with an error:
#'
#' * a REML fit beside an ML fit: the two log-likelihoods are on different
#'   scales. The error names which fit is which and points to [compare()],
#'   which fits both models the same way. This holds for the fits of
#'   other packages in the call too: an lme4 fit is REML when
#'   `lme4::isREML()` says so, an `lm` or `glm` fit is ML, and any other fit
#'   whose `logLik()` carries a `REML` attribute is read from it.
#' * REML fits with different fixed effects: the REML log-likelihood depends on
#'   the fixed-effects design. The error points to [compare()], which fits
#'   both models by ML, or says to refit both with `REML = FALSE`.
#'
#' The fixed effects are compared as the sets of fixed terms of the stored
#' formulas: term order and the spelling of an interaction do not matter, and
#' `y ~ x` and `y ~ 0 + x` differ. With more than two fits both rules apply to
#' every pair, the fixed-effects rule to every pair of fastglmm REML fits. ML
#' fits compare as in `stats`, whatever their fixed parts.
#'
#' Four uses are not checked: an AIC built from `logLik()` by hand or by a
#' package that does not call `AIC()` (`logLik()` returns the REML value with a
#' `REML = TRUE` attribute); a call whose first argument is not a fastglmm
#' fit, such as `AIC(m_lm, m_reml)`, which `stats` dispatches to its default
#' method; the fixed effects of another package's REML fit, such as
#' `AIC(m_reml, m_lmer)` with different fixed effects (its formula is not
#' compared); and the estimator of another package's fit that is neither an
#' lme4, `lm` nor `glm` fit and whose `logLik()` carries no `REML` attribute,
#' such as an nlme `lme` fit.
#'
#' @param object a [fastglmm] fit.
#' @param ... further fitted models.
#' @param k the penalty per parameter (AIC only), as in [stats::AIC()].
#' @return A number for one fit; for several, a `data.frame` with columns `df`
#'   and `AIC` (or `BIC`), one row per fit.
#' @export
AIC.fastglmm <- function(object, ..., k = 2) {
  .check_ic_fits(list(object, ...), as.list(substitute(list(object, ...)))[-1L], "AIC")
  NextMethod()
}

#' @rdname AIC.fastglmm
#' @export
BIC.fastglmm <- function(object, ...) {
  .check_ic_fits(list(object, ...), as.list(substitute(list(object, ...)))[-1L], "BIC")
  NextMethod()
}

# Stops on the two comparisons AIC()/BIC() refuse. `exprs` are the call's
# unevaluated argument expressions; only an error deparses them, so a
# do.call() with fits passed by value never deparses a whole fit. The stored
# formula is one string (fastglmm()'s f_str), as the Rust check needs.
.check_ic_fits <- function(fits, exprs, what) {
  if (length(fits) < 2L) return(invisible())
  label <- function(i) {
    e <- exprs[[i]]
    if (is.symbol(e) || is.call(e)) {
      s <- deparse1(e)
      if (nchar(s) <= 60L) return(s)
    }
    paste("fit", i)
  }
  reml <- vapply(fits, .fit_is_reml, NA)
  if (any(reml, na.rm = TRUE) && !all(reml, na.rm = TRUE)) {
    stop(sprintf(paste("%s was fitted by REML and %s by ML; comparing their %s needs both",
                       "fitted the same way. Refit %s with REML = FALSE, or use",
                       "compare(data, f0, f1), which picks the estimator itself."),
                 label(which(reml)[1L]), label(which(!reml)[1L]), what,
                 label(which(reml)[1L])), call. = FALSE)
  }
  # The fixed parts are compared on fastglmm REML fits only: another
  # package's formula is not in the form the Rust check reads.
  ours <- which(vapply(fits, inherits, NA, what = "fastglmm") & reml %in% TRUE)
  # "Same fixed part" is an equivalence, so checking each fit against the
  # first checks every pair.
  for (i in ours[-1L]) {
    same <- .glmm_call("same_fixed_part", list(f0 = fits[[ours[1L]]]$formula,
                                               f1 = fits[[i]]$formula))
    if (!same) {
      stop(sprintf(paste0("%s and %s are REML fits with different fixed effects; the ",
                          "REML likelihood depends on the fixed effects, so comparing ",
                          "their %s needs ML fits. Refit both with REML = FALSE, or use ",
                          "compare(data, f0, f1), which fits both models by ML."),
                   label(ours[1L]), label(i), what), call. = FALSE)
    }
  }
  invisible()
}

# TRUE for a REML fit, FALSE for an ML fit, NA when the estimator cannot be
# read. lme4's logLik() carries no REML attribute, so a merMod is read with
# lme4's isREML(). lme4 is not a dependency and is never loaded from here:
# its isREML() is used only when lme4 is already loaded, as it must be for
# stats::AIC() to reach logLik() on a merMod at all. stats' logLik() of an
# lm or glm fit is the ML value.
.fit_is_reml <- function(x) {
  if (inherits(x, "fastglmm")) return(isTRUE(x$reml))
  if (inherits(x, "merMod")) {
    if (!isNamespaceLoaded("lme4")) return(NA)
    return(isTRUE(getExportedValue("lme4", "isREML")(x)))
  }
  if (inherits(x, "lm")) return(FALSE)
  r <- attr(stats::logLik(x), "REML")
  if (isTRUE(r)) TRUE else if (isFALSE(r)) FALSE else NA
}

# -- broom-shaped accessors. Registered on `generics::tidy`/`generics::glance`
# (re-exported the same way as fixef/ranef/VarCorr above, off nlme instead of
# generics): modelsummary, texreg, gt and kableExtra call `broom::tidy(fit)`,
# which IS generics::tidy re-exported, so a local generic would never be
# found. `generics` is the import, never `broom` (21 non-base recursive
# dependencies; nothing here calls a broom function). ---------------------

#' @importFrom generics tidy
#' @export
generics::tidy

#' @importFrom generics glance
#' @export
generics::glance

#' Tidy the fixed effects
#'
#' One row per fixed effect: `term`, `estimate`, `std.error`, `statistic`,
#' `df` (only where [summary()][summary.fastglmm] has a df column: the
#' Satterthwaite df of a Gaussian mixed model) and `p.value`. The statistic
#' and p-value are the summary's: t where it tests with t, Wald z on the
#' normal otherwise. Aliased coefficients are `NA` rows.
#'
#' @param x a [fastglmm] fit.
#' @param ... unused.
#' @return A `data.frame`.
#' @export
tidy.fastglmm <- function(x, ...) {
  s <- .summary_record(x)
  out <- data.frame(
    term = names(x$beta),
    estimate = unname(x$beta),
    std.error = unname(x$se),
    statistic = s$statistic,
    stringsAsFactors = FALSE
  )
  if (!is.null(s$coef_df)) out$df <- s$coef_df
  out$p.value <- s$p_value
  out
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

# The fit's df_method; objects saved before `df_method` existed test with z.
.df_method <- function(object) object$df_method %||% "none"

# The footer of a fit with `satterthwaite = FALSE`; the Python port spells its
# argument `satterthwaite=False`.
.DF_NOT_COMPUTED <- "df not computed (satterthwaite = FALSE)"

# AIC / BIC / deviance (-2 logLik) / residual df, as glance() reports them.
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
       "parser shared with the Python port), and a ",
       "synthesized R terms object could disagree with how the model was ",
       "actually built. formula() returns the model formula as given.", call. = FALSE)
}
