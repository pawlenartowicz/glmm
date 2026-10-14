# anova() and compare(): the per-term table of one fit, and the comparison of
# two models. The crate computes every statistic and p-value (the per-term
# tables at fit time, the comparison in fastglmm_anova_pair and
# fastglmm_compare); this file only lays them out. Mirrors the Python port's
# glmm/anova.py - change together. The per-term heading and note texts are
# also copied in validation/tools/anova_ports/anova_ports.rs, which the anova
# port gate compares with both ports.

#' Tests of fixed terms, and comparisons of two fits
#'
#' With one fit, `anova()` gives one row per fixed term, read from the table the
#' crate built when the model was fitted: nothing is refitted when it is called.
#' Type III by default, built the way lmerTest builds it, so it does not depend
#' on the contrast coding; Type II with `type = "II"`. A Gaussian mixed model
#' gets lmerTest's F with Satterthwaite's denominator df (columns `Sum Sq`,
#' `Mean Sq`, `NumDF`, `DenDF`, `F value`, `Pr(>F)`); OLS and the Gamma and
#' inverse-Gaussian GLMs with an estimated dispersion get the F on the residual
#' df `n - p`; every other fit (binomial, Poisson and negative-binomial GLMs,
#' every GLMM) gets the Wald chi-squared test (columns `Chisq`, `Df`,
#' `Pr(>Chisq)`), but a Poisson model with random effects the likelihood-ratio
#' test, in the same columns (below). There are no random-effect rows: random
#' effects are tested by comparing two models.
#'
#' On a fit with Wald chi-squared rows, `test = "LRT"` gives likelihood-ratio
#' chi-squared rows, in the Wald table's columns: each term's row compares the
#' model with the model refitted under the term's hypothesis (Type III: the
#' reduced design of the hypothesis its Wald row tests, as afex's
#' `mixed(method = "LRT")` on a GLMM; Type II: the model of every term that does
#' not contain the term, with and without it, the rule of car's
#' `Anova(test.statistic = "LR")` on a GLM, applied to GLMMs too). These refits
#' are made when the model is fitted: when the fit asks for them with
#' `fastglmm(..., anova_test = "LRT")`, and on a Poisson model with random
#' effects under the default `anova_test = "auto"`. A term whose columns carry a
#' random slope keeps its Wald row, with a note.
#'
#' The default `test = "auto"` gives the likelihood-ratio rows to a Poisson
#' model with random effects, whatever its size: in simulation the Wald test
#' of a Poisson GLMM was never closer to the nominal level than the
#' likelihood-ratio test, and rejected too often at few clusters. Every other
#' fit gets its Wald chi-squared or F rows. The heading names the test;
#' under `test = "auto"` a chi-squared table's heading also names why `"auto"`
#' chose it (the family, or no random effects).
#'
#' A formula whose interaction
#' lacks one of its marginal terms (`y ~ x + x:f`) gets no table: the fit
#' stands, and `anova()` stops with the reason.
#'
#' With two fits, `anova(m0, m1)` compares them as they are, with no refit and
#' without reading their data: the two fits must have the same rows, the same
#' response, weights, offset, family, link and held dispersion, and one
#' formula must be nested in the other. The fits can be given in either
#' order; the table lists the smaller model first, under its label (`m0` is
#' the first fit given). The test depends on the pair:
#'
#' * two OLS fits: the exact F of `anova.lm`, with its columns `Res.Df`,
#'   `RSS`, `Df`, `Sum of Sq`, `F`, `Pr(>F)`;
#' * two Gamma or inverse-Gaussian GLMs with an estimated dispersion: the Wald
#'   F on `(r, n - p)` (columns `Res.Df`, `Df`, `F`, `Pr(>F)`);
#' * any other pair of ML fits: the likelihood-ratio test, with lme4's columns
#'   `npar`, `AIC`, `BIC`, `logLik`, `Chisq`, `Df`, `Pr(>Chisq)`;
#' * two REML fits with the same fixed part: the likelihood-ratio test on the
#'   REML criterion, with no `AIC` or `BIC` column. A REML fit is compared
#'   with an OLS fit made with `REML = TRUE`.
#'
#' When the random parts differ, the reference is the exact restricted
#' likelihood-ratio test (Crainiceanu and Ruppert 2004) when the larger model
#' is a Gaussian REML fit whose only random term is one unweighted `(1 | g)`,
#' with at least one fixed-effect column, and the smaller is OLS with the same
#' fixed part; otherwise the 50:50
#' mixture of chi-squared distributions when one random dimension is dropped,
#' and chi-squared on the parameter difference, labelled conservative, when
#' more are. The heading names the reference and carries the notes.
#'
#' Two REML fits with different fixed parts, and a REML fit beside an ML fit,
#' are errors: lme4 refits such fits by ML, `anova()` here does not. Use
#' [compare()], which fits both models itself by the right estimator, or refit
#' both with `REML = FALSE`.
#'
#' The layout of these tables is provisional until 1.0.0.
#'
#' @param object a [fastglmm] fit.
#' @param ... a second [fastglmm] fit to compare `object` with; at most one.
#' @param type `"III"` (the default) or `"II"`: the hypotheses of one fit's
#'   per-term table.
#' @param test for one fit, the test of the per-term rows: `"auto"` (the
#'   default) gives the likelihood-ratio rows on a Poisson model with random
#'   effects and the Wald chi-squared (or F) rows on every other fit;
#'   `"Wald"` the Wald chi-squared rows; `"LRT"` the likelihood-ratio rows.
#'   The likelihood-ratio rows are those the fit stored: a fit made with
#'   `fastglmm(..., anova_test = "LRT")` stores them, and so does a Poisson
#'   model with random effects fitted with the default `anova_test = "auto"`
#'   (an error otherwise). On a fit whose rows are F tests (a Gaussian model,
#'   a Gamma or inverse-Gaussian GLM with an estimated dispersion), `"Wald"`
#'   and `"LRT"` are errors.
#' @param nsim for two fits, the number of draws of the exact test's null
#'   distribution: `NULL` (10000) or a whole number from 1 to 1000000.
#' @param seed for two fits, the seed of those draws: `NULL` takes a fixed
#'   default, so the same call gives the same p-value on every run and
#'   [set.seed()] does not change it; or a whole number from 0 to 2^53. R
#'   holds the seed as a double, which is exact up to 2^53, so a larger or
#'   fractional seed is an error.
#' @return A `c("anova", "data.frame")` with a `heading` attribute: one row
#'   per fixed term for one fit; for two fits, one row per fit with the test
#'   on the larger model's row, and the attributes `nsim` and `seed`: the
#'   draws requested and their seed, on every two-fit table, also when the
#'   test made no draw.
#' @seealso [compare()]
#' @examples
#' set.seed(1)
#' g <- factor(rep(1:30, each = 8))
#' x <- rnorm(240)
#' f <- factor(sample(c("a", "b", "c"), 240, replace = TRUE))
#' y <- 1 + 0.5 * x + rnorm(30)[as.integer(g)] + rnorm(240)
#' d <- data.frame(y, x, f, g)
#' m <- fastglmm(y ~ x * f + (1 | g), d)
#' anova(m)
#' anova(fastglmm(y ~ x, d, REML = TRUE), fastglmm(y ~ x + (1 | g), d))
#' n <- rpois(240, exp(0.5 + 0.3 * x + rnorm(30, sd = 0.5)[as.integer(g)]))
#' p <- fastglmm(n ~ x + f + (1 | g), data.frame(n, x, f, g), family = poisson)
#' anova(p)                  # a Poisson GLMM: likelihood-ratio rows by default
#' anova(p, test = "Wald")
#' @export
anova.fastglmm <- function(object, ..., type = c("III", "II"), test = c("auto", "Wald", "LRT"),
                           nsim = NULL, seed = NULL) {
  fits <- c(list(object), .positional(list(...)))
  for (i in seq_along(fits)[-1L]) {
    if (!inherits(fits[[i]], "fastglmm")) {
      stop(sprintf("anova() compares two fastglmm fits; fit %d is a %s.", i,
                   class(fits[[i]])[1L]), call. = FALSE)
    }
  }
  if (length(fits) == 1L) {
    if (!is.null(nsim) || !is.null(seed)) {
      stop("nsim and seed apply to a comparison of two fits, not to one fit's ",
           "per-term table.", call. = FALSE)
    }
    return(.term_table(object, match.arg(type), match.arg(test)))
  }
  if (length(fits) > 2L) {
    stop(sprintf(paste("anova() takes one fit (the per-term table) or two fits",
                       "(a comparison), not %d."), length(fits)), call. = FALSE)
  }
  if (!missing(type)) {
    stop("type applies to one fit's per-term table, not to a comparison of two fits.",
         call. = FALSE)
  }
  if (!missing(test)) {
    stop("test applies to one fit's per-term table, not to a comparison of two fits.",
         call. = FALSE)
  }
  .check_nsim_seed(nsim, seed)
  for (fit in fits) .check_fields(fit, c("objective", "link", "rss"))
  res <- fastglmm_anova_pair(.compare_side(fits[[1L]]), .compare_side(fits[[2L]]),
                             .optional(nsim), .optional(seed))
  if (!is.null(res$refusal)) stop(.pair_refusal(res, fits[[1L]], fits[[2L]]), call. = FALSE)
  .comparison_table(res$comparison, c("m0", "m1"), fits, res$nsim, res$seed, NULL)
}

#' Compare two models fitted to the same data
#'
#' `compare()` takes one data set and two formulas, fits both models on the
#' same rows, and compares them. A row with a missing value in any variable of
#' either formula, in the weights or in the offset is dropped before both fits
#' (where [fastglmm()] refuses a missing weight or offset).
#' `family`, `weights`, `offset` and `nAGQ` are given once and used for both
#' fits. The two formulas can be given in either order, and must be nested:
#' the same response, every fixed term of the smaller formula in the larger
#' one, and every grouping of the smaller one in the larger one with the same
#' or more slopes.
#'
#' There is no `REML` argument: the estimator follows from what differs. On a
#' Gaussian model, both fits are REML fits when only the random part differs
#' (the model without random effects made with `REML = TRUE`) and ML fits when
#' the fixed part differs, alone or with the random part. Every other family
#' is fitted by maximum likelihood. The test is then the one
#' [anova()][anova.fastglmm] gives for the two fits, and `anova(fit0, fit1)`
#' gives the same test and p-value, the exact one included.
#'
#' The name `compare()` and the class `fastglmm_comparison` are provisional
#' until 1.0.0.
#'
#' @param data a `data.frame` (or something coercible) holding every column
#'   either formula names.
#' @param f0,f1 the two model formulas (or strings), as [fastglmm()] takes
#'   them.
#' @param ... must be empty: `compare()` takes two formulas, not three or
#'   more.
#' @param family,weights,offset,nAGQ as in [fastglmm()], for both fits.
#' @param nsim,seed the draws and the seed of the exact restricted
#'   likelihood-ratio test, as in [anova()][anova.fastglmm]: `nsim` a whole
#'   number from 1 to 1000000 (default 10000), `seed` `NULL` for the fixed
#'   default or a whole number from 0 to 2^53 (exact as a double; a larger or
#'   fractional seed is an error).
#' @return A list of class `"fastglmm_comparison"`: `table`, the comparison
#'   table of [anova()][anova.fastglmm] with its rows named `f0` and `f1`
#'   (the smaller model first), and `fit0` and `fit1`, the fits of `f0` and
#'   `f1`. Printing it prints the table. Both fits are made with
#'   `anova_test = "auto"` (see [fastglmm()]): `anova()` of `fit0` or `fit1`
#'   with `test = "LRT"` works only on a Poisson model with random effects.
#'   For the likelihood-ratio rows of another model with chi-squared rows, fit
#'   its formula with `fastglmm(..., anova_test = "LRT")`.
#' @seealso [anova()][anova.fastglmm]
#' @examples
#' set.seed(1)
#' g <- factor(rep(1:30, each = 8))
#' x <- rnorm(240)
#' y <- 1 + 0.5 * x + rnorm(30)[as.integer(g)] + rnorm(240)
#' d <- data.frame(y, x, g)
#' compare(d, y ~ x, y ~ x + (1 | g))
#' @export
compare <- function(data, f0, f1, ..., family = gaussian(), weights = NULL,
                    offset = NULL, nAGQ = 1L, nsim = NULL, seed = NULL) {
  if (inherits(data, "fastglmm") || (!missing(f0) && inherits(f0, "fastglmm")) ||
      (!missing(f1) && inherits(f1, "fastglmm"))) {
    stop("compare() takes data and two formulas, never fits: compare(data, f0, f1). ",
         "anova(m0, m1) compares two fits as they are.", call. = FALSE)
  }
  if (length(.positional(list(...)))) {
    stop("compare() takes two formulas, not three or more.", call. = FALSE)
  }
  data_name <- paste(deparse(substitute(data), width.cutoff = 500L), collapse = " ")
  forms <- lapply(list(f0, f1), function(f) {
    if (is.character(f)) f <- stats::as.formula(f)
    if (!inherits(f, "formula")) {
      stop("f0 and f1 must be formulas or formula strings", call. = FALSE)
    }
    f
  })
  f_strs <- vapply(forms, function(f) paste(deparse(f, width.cutoff = 500L), collapse = " "),
                   character(1))
  for (i in 1:2) .check_formula(forms[[i]], f_strs[[i]])
  fam <- .normalize_family(family)
  if (identical(fam$name, "inversegaussian") && any(grepl("|", f_strs, fixed = TRUE))) {
    stop("family 'inversegaussian' is GLM-only: random-effect terms are not supported",
         call. = FALSE)
  }
  nAGQ <- .check_nagq(nAGQ)
  .check_nsim_seed(nsim, seed)

  # One row mask over the variables of both formulas, the weights and the
  # offset, so both models are fitted to the same rows.
  vars <- unique(unlist(lapply(forms, all.vars)))
  data <- .as_data_frame(data, vars)
  frame <- .data_columns(data, vars)
  for (f in forms) .check_contrasts(f, frame)
  # A missing weight or offset drops its row too, as a missing value in a
  # formula's variable does.
  frame <- .add_weights(frame, eval(substitute(weights), data, parent.frame()), nrow(data),
                        missing_ok = TRUE)
  frame <- .add_offset(frame, eval(substitute(offset), data, parent.frame()), nrow(data),
                       missing_ok = TRUE)
  keep <- stats::complete.cases(frame)
  # Mirrors the Python port's compare() wording - change together.
  where <- "a column either formula uses, the weights or the offset"
  if (!any(keep)) {
    stop(if (nrow(frame)) {
      sprintf("every row has a missing value in %s; no rows left to fit", where)
    } else {
      "no rows left to fit"
    }, call. = FALSE)
  }
  store <- new.env(parent = emptyenv())
  store$rows <- list()
  if (!all(keep)) {
    .warn_keep(store, "rows_dropped_na", sprintf(
      "Dropped %d of %d row(s): %s had a missing value there.",
      sum(!keep), length(keep), where),
      c("fastglmm_rows_dropped_na", "fastglmm_diagnostic"))
    frame <- frame[keep, , drop = FALSE]
  }
  w <- frame[["(weights)"]]
  o <- frame[["(offset)"]]
  numeric_only <- unlist(lapply(forms, function(f) c(all.vars(f[[2L]]), .offset_vars(f))))
  cols <- .marshal(frame, vars, numeric_only)

  res <- fastglmm_compare(f_strs[[1L]], f_strs[[2L]], cols$numeric, cols$levels, cols$codes,
                          fam$name, fam$link, nAGQ, .optional(w), .optional(o),
                          .optional(nsim), .optional(seed))
  call <- match.call()
  # Each fit keeps the columns of its own formula, the weights and the
  # offset, as fastglmm() keeps them; the rows are the shared ones.
  fit <- function(r, i) {
    s <- new.env(parent = emptyenv())
    s$rows <- store$rows
    own <- intersect(c(all.vars(forms[[i]]), "(weights)", "(offset)"), names(frame))
    .fit_object(r, s, fam, nAGQ, NULL, f_strs[[i]], frame[own], call, data_name)
  }
  fit0 <- fit(res$fit0, 1L)
  fit1 <- fit(res$fit1, 2L)
  structure(list(
    table = .comparison_table(res$comparison, c("f0", "f1"), list(fit0, fit1),
                              res$nsim, res$seed, fam$name),
    fit0 = fit0,
    fit1 = fit1
  ), class = "fastglmm_comparison")
}

#' @export
print.fastglmm_comparison <- function(x, ...) {
  print(x$table, ...)
  invisible(x)
}

# The unnamed elements of `dots`; a named one is an error, as fastglmm()
# refuses an unknown argument rather than drop it.
.positional <- function(dots) {
  nm <- names(dots) %||% rep("", length(dots))
  if (any(nzchar(nm))) {
    stop(sprintf("unused argument '%s'", nm[nzchar(nm)][1L]), call. = FALSE)
  }
  dots
}

# A df for a reference line, never in scientific notation and never padded:
# an integer when integral or 1e4 and above, else four significant digits
# (2.5, 12.35); NA and NaN print as NA, an infinite df as Inf or -Inf.
# Mirrors the Python port's fmt_df - change together.
.fmt_df <- function(df) {
  if (is.na(df)) return("NA")
  if (is.infinite(df)) return(if (df > 0) "Inf" else "-Inf")
  if (df == round(df) || abs(df) >= 1e4) return(sprintf("%.0f", df))
  trimws(formatC(df, digits = 4L, format = "fg"))
}

# One notation for a reference distribution, as the Python port writes it. A
# df that is NaN (an aliased term, a row with no denominator df) names none.
.f_reference <- function(num_df, den_df) {
  if (is.nan(num_df) || is.nan(den_df)) "" else
    sprintf("F(%s, %s)", .fmt_df(num_df), .fmt_df(den_df))
}
.chisq_reference <- function(df) if (is.nan(df)) "" else sprintf("\u03c7\u00b2(%s)", .fmt_df(df))

# Why "auto" gave a chi-squared table its test, by the fit's term_test_auto
# reason. Mirrors the Python port's _AUTO_REASONS and
# validation/tools/anova_ports/anova_ports.rs - change together.
.AUTO_REASONS <- c(
  poisson_mixed = "Poisson model with random effects",
  binomial_mixed = "binomial model with random effects",
  negative_binomial_mixed = "negative binomial model with random effects",
  other_mixed = "model with random effects, not binomial, Poisson or negative binomial",
  no_random_effects = "model without random effects")

# The first note of a fit that did not converge. Mirrors the Python port's
# _NOT_CONVERGED and validation/tools/anova_ports/anova_ports.rs - change
# together.
.NOT_CONVERGED <- paste(
  "The fit did not converge, so its estimates are not at a maximum and these tests",
  "are not reliable. Refit until the fit converges, then test again.")

# Stops on a fit made by a fastglmm version without these tests, naming the
# fields it lacks.
.check_fields <- function(fit, fields) {
  absent <- fields[vapply(fields, function(f) is.null(fit[[f]]), NA)]
  if (length(absent)) {
    stop(sprintf(paste("this fit has no %s: it was made by a fastglmm version without these",
                       "tests. Refit the model with this version."),
                 paste(absent, collapse = ", ")), call. = FALSE)
  }
  invisible()
}

# An optional number for the shim: an empty vector when absent.
.optional <- function(x) if (is.null(x)) double() else as.double(x)

# nsim and seed: NULL, or one whole number each. The range of nsim is the
# crate's check (orchestrate::compare_options), with its own message.
.check_nsim_seed <- function(nsim, seed) {
  whole <- function(x) is.numeric(x) && length(x) == 1L && is.finite(x) && x == round(x)
  if (!is.null(nsim) && !whole(nsim)) {
    stop(sprintf("nsim must be NULL or one whole number, got %s", .arg_repr(nsim)),
         call. = FALSE)
  }
  if (!is.null(seed) && !(whole(seed) && seed >= 0 && seed <= 2^53)) {
    stop(sprintf("seed must be NULL or one whole number from 0 to 2^53, got %s",
                 .arg_repr(seed)), call. = FALSE)
  }
  invisible()
}

# One fit's stored per-term result as lmerTest's (F) or car's (Wald or
# likelihood-ratio chi-squared) table, or the stored refusal as an error. A
# Wald table's rows are all F or all chi-squared; an empty table takes its
# kind from df_method. `test` "LRT", and "auto" where the fit's
# term_test_auto is the likelihood-ratio test (a Poisson model with random
# effects), read the likelihood-ratio table the fit stored (a row whose term
# carries a random slope is its Wald row, with a note); "Wald" and "LRT" on F
# rows are the crate's refusal.
.term_table <- function(object, type, test) {
  field <- if (identical(type, "III")) "term_table_type3" else "term_table_type2"
  # Only "auto" reads the fit's auto choice: a fit stored before it still
  # takes an explicit test.
  .check_fields(object, c(field, if (identical(test, "auto")) "term_test_auto"))
  res <- object[[field]]
  if (!is.null(res$error)) stop(res$error, call. = FALSE)
  dm <- .df_method(object)
  f_rows <- if (length(res$kind)) res$kind[[1L]] == "F" else dm != "none"
  if (f_rows && !identical(test, "auto")) {
    stop(fastglmm_term_test_refusal(test, !identical(dm, "residual")), call. = FALSE)
  }
  lrt <- identical(test, "LRT") ||
    (identical(test, "auto") && identical(object$term_test_auto$test, "lrt"))
  if (lrt) {
    res <- object[[paste0(field, "_lrt")]]
    if (is.null(res) && identical(test, "LRT")) {
      stop(paste("this fit stored no likelihood-ratio tables, which are built only when the",
                 'fit asks for them: refit with fastglmm(..., anova_test = "LRT").'),
           call. = FALSE)
    }
    if (is.null(res)) {
      stop(paste('test = "auto" gives likelihood-ratio tests on a Poisson model with random',
                 'effects, and this fit stored none: it was made with anova_test = "Wald".',
                 'Use test = "Wald", or refit with fastglmm(..., anova_test = "auto").'),
           call. = FALSE)
    }
    if (!is.null(res$error)) stop(res$error, call. = FALSE)
  }
  # Satterthwaite's method is named only when some row has its df: on a fit
  # with the pass switched off or failed, no row does.
  satterthwaite <- if (length(res$label)) {
    any(res$den_df_kind == "satterthwaite" & is.finite(res$den_df))
  } else {
    dm == "satterthwaite"
  }
  # The likelihood-ratio heading's second line names the test and why it
  # ran. Mirrors the Python port's heading and
  # validation/tools/anova_ports/anova_ports.rs - change together.
  # An explicit "Wald" keeps the one-line heading.
  title <- if (lrt || (!f_rows && identical(test, "auto"))) {
    why <- if (identical(test, "LRT")) {
      'test = "LRT"'
    } else {
      paste0('test = "auto": ', .AUTO_REASONS[[object$term_test_auto$reason]])
    }
    line <- if (lrt) {
      paste0("Likelihood-ratio chi-squared tests (each term's model refitted without it)",
             if (identical(type, "III")) " on the reduced design")
    } else {
      "Wald chi-squared tests"
    }
    c("Analysis of Deviance Table", paste0(line, "; ", why))
  } else if (!f_rows) {
    "Analysis of Deviance Table (Wald \u03c7\u00b2 tests)"
  } else if (satterthwaite) {
    "Analysis of Variance Table with Satterthwaite's method"
  } else {
    "Analysis of Variance Table"
  }
  # An untested cell is NA, not NaN, so print.anova leaves it blank, as on
  # lmerTest's and car's tables.
  na <- function(v) replace(v, is.nan(v), NA_real_)
  tab <- if (f_rows) {
    data.frame(`Sum Sq` = na(res$sum_sq), `Mean Sq` = na(res$mean_sq),
               NumDF = na(res$num_df), DenDF = na(res$den_df),
               `F value` = na(res$statistic), `Pr(>F)` = na(res$p_value),
               check.names = FALSE)
  } else {
    data.frame(Chisq = na(res$statistic), Df = na(res$num_df),
               `Pr(>Chisq)` = na(res$p_value), check.names = FALSE)
  }
  rownames(tab) <- res$label
  row_notes <- unlist(Map(function(label, notes) {
    if (length(notes)) paste0(label, ": ", notes)
  }, res$label, res$row_notes), use.names = FALSE)
  title[1L] <- paste("Type", res$ss_type, title[1L])
  structure(tab, heading = c(title,
                            if (!isTRUE(object$converged)) .NOT_CONVERGED, res$notes, row_notes),
            class = c("anova", "data.frame"))
}

# What anova(m0, m1) reads of one stored fit, named as fastglmm_anova_pair
# reads it. Never the model frame: a comparison of two fits uses only what
# the fits carry.
.compare_side <- function(fit) {
  list(
    formula = fit$formula,
    family = fit$family_name,
    link = fit$link,
    loglik = as.double(fit$loglik),
    df = as.double(fit$df),
    nobs = as.double(fit$nobs),
    reml = isTRUE(fit$reml),
    objective = fit$objective,
    objective_nodes = as.double(fit$objective_nodes),
    converged = isTRUE(fit$converged),
    varcorr = lapply(fit$varcorr, as.double),
    re_group_names = as.character(fit$re_group_names),
    re_group_terms = lapply(fit$re_group_terms, as.character),
    y = as.double(fit$y),
    weights = as.double(fit$weights),
    offset = as.double(fit$offset),
    dispersion_held = as.double(fit$dispersion_held),
    design_summary = fit$design_summary,
    rss = as.double(fit$rss),
    coef_names = names(fit$beta),
    coef = unname(as.double(fit$beta)),
    vcov = as.double(t(fit$vcov)),
    aliased = unname(as.logical(fit$aliased))
  )
}

# The message of a refusal of anova(m0, m1). The crate's text says what is
# wrong in words common to both ports; for the refusals compare() avoids,
# this port names its own call. The two estimator refusals are worded as
# AIC()'s (.check_ic_fits).
.pair_refusal <- function(res, m0, m1) {
  estimator <- function(fit) if (isTRUE(fit$reml)) "REML" else "ML"
  switch(res$kind,
    mixed_estimators = sprintf(paste(
      "m0 was fitted by %s and m1 by %s. Fit both the same way, or use",
      "compare(data, f0, f1)."), estimator(m0), estimator(m1)),
    reml_different_fixed = paste(
      "m0 and m1 are REML fits with different fixed effects, and REML fits with different",
      "fixed effects cannot be compared by likelihood ratio. Use compare(data, f0, f1),",
      "which fits both models by ML, or refit both with REML = FALSE."),
    paste(c(res$refusal, switch(res$kind,
      different_rows = "compare(data, f0, f1) does that and fits both models in one call.",
      different_y = "compare(data, f0, f1) fits both models to one data set.",
      different_weights = "compare(data, f0, f1) takes the weights once, for both models.",
      different_offset = "compare(data, f0, f1) takes the offset once, for both models.",
      different_family = "compare(data, f0, f1) takes the family once, for both models.",
      NULL)), collapse = " ")
  )
}

# The comparison table from fastglmm_anova_pair's or fastglmm_compare's
# `comparison`: one row per model, the smaller first, named by `labels` (the
# models as given); `fits` are the two fit objects (the OLS pair reads their
# RSS). The heading names the test, the two formulas, the reference (with the
# draws made) and the notes; `nsim` (the draws asked for) and `seed` are
# attributes. `family` is compare()'s, NULL from anova(m0, m1): compare()
# says why it fitted a Gaussian pair by ML.
.comparison_table <- function(cmp, labels, fits, nsim, seed, family) {
  ord <- if (identical(cmp$larger, "m0")) 2:1 else 1:2
  sides <- list(cmp$m0, cmp$m1)[ord]
  labels <- labels[ord]
  num <- function(name) vapply(sides, function(s) as.double(s[[name]]), numeric(1))
  tested <- function(x) c(NA_real_, x)
  lr <- identical(cmp$test, "lr")
  ols <- !lr && identical(cmp$m0$objective, "least_squares")
  cols <- if (lr) {
    c(list(npar = num("npar")),
      if (!any(vapply(sides, function(s) is.null(s$aic), NA))) {
        list(AIC = num("aic"), BIC = num("bic"))
      },
      list(logLik = num("loglik"), Chisq = tested(cmp$statistic),
           Df = tested(if (identical(cmp$reference, "exact")) NA_real_ else cmp$df),
           `Pr(>Chisq)` = tested(cmp$p_value)))
  } else {
    res_df <- num("nobs") - (num("npar") - 1)
    rss <- vapply(fits[ord], function(f) as.double(f$rss), numeric(1))
    c(list(Res.Df = res_df),
      if (ols) list(RSS = rss),
      list(Df = tested(cmp$df)),
      if (ols) list(`Sum of Sq` = tested(rss[[1L]] - rss[[2L]])),
      list(F = tested(cmp$statistic), `Pr(>F)` = tested(cmp$p_value)))
  }
  tab <- as.data.frame(cols, check.names = FALSE, row.names = labels)
  title <- if (lr) {
    switch(cmp$objective,
      reml = "Model comparison: likelihood ratio test on the REML criterion",
      laplace = "Model comparison: likelihood ratio test (Laplace)",
      agq = sprintf("Model comparison: likelihood ratio test (AGQ %d)", cmp$objective_nodes),
      "Model comparison: likelihood ratio test (ML)")
  } else {
    "Model comparison: F test"
  }
  rlrt <- "exact RLRT null distribution (Crainiceanu & Ruppert 2004)"
  reference <- switch(cmp$reference,
    chisq = .chisq_reference(cmp$ref_df),
    chibar = sprintf("50:50 mixture of %s and %s", .chisq_reference(cmp$ref_df),
                     .chisq_reference(cmp$ref_df + 1)),
    exact = if (cmp$draws > 0) {
      sprintf("%s, %.0f draws", rlrt, cmp$draws)
    } else {
      paste0(rlrt, "; no draws needed: the statistic is 0, so p = 1")
    },
    F = .f_reference(cmp$ref_df, cmp$den_df))
  notes <- cmp$notes
  if (identical(family, "gaussian") && lr && !identical(cmp$differs, "random")) {
    notes <- c(notes, "Both models were fitted by ML because their fixed parts differ.")
  }
  models <- paste0(labels, ": ", vapply(sides, `[[`, character(1), "formula"))
  structure(tab, heading = c(title, models, paste0("Reference: ", reference, "."), notes),
            nsim = nsim, seed = seed, class = c("anova", "data.frame"))
}
