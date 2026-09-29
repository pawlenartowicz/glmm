# Deterministic fixtures, mirrored in python/tests/test_warnings.py.
x <- as.numeric(setdiff(-10:10, 0))
# Parenthesized: %% binds tighter than * in R, unlike Python's %.
yy <- 0.3 * x + ((((seq_along(x) - 1L) * 7L) %% 5L) - 2L) * 0.4
FIXED <- data.frame(x = x, y = yy)
g <- factor(sprintf("g%d", (seq_along(x) - 1L) %% 5L))
off <- c(g0 = -3, g1 = -1.5, g2 = 0, g3 = 1.5, g4 = 3)
MIXED <- data.frame(x = x, g = g, y = yy + off[as.character(g)])

printed <- function(row) {
  sprintf("%s%s: %s. %s", toupper(substr(row$tier, 1L, 1L)), substring(row$tier, 2L),
          row$title, row$message)
}

# expect_warning-equivalent that tolerates the fixture's other warnings: keeps
# the first condition of class `cls`, muffles everything.
fit_catching <- function(expr, cls) {
  cond <- NULL
  fit <- withCallingHandlers(expr, warning = function(w) {
    if (is.null(cond) && inherits(w, cls)) cond <<- w
    invokeRestart("muffleWarning")
  })
  list(fit = fit, cond = cond)
}

only <- function(fit, kind) {
  row <- fit$warnings[fit$warnings$kind == kind, , drop = FALSE]
  expect_equal(nrow(row), 1L, info = kind)
  expect_identical(c(row$tier, row$title), fastglmm:::.WARNING_KINDS[[kind]], info = kind)
  row
}

test_that("a clean fit stores a zero-row data.frame", {
  expect_no_warning(fit <- fastglmm(y ~ x + (1 | g), MIXED))
  expect_s3_class(fit$warnings, "data.frame")
  expect_named(fit$warnings, c("tier", "kind", "title", "message"))
  expect_equal(nrow(fit$warnings), 0L)
})

test_that("argument_ignored is stored and printed, one text per case", {
  cases <- list(
    list(args = list(dispersion = 2),
         msg = "dispersion= has no effect for family 'gaussian'."),
    list(args = list(init.theta = 1.5),
         msg = "init.theta= is not used for family 'gaussian'."),
    list(args = list(start = list(beta = c(0, 0), phi = 1, b = 2)),
         msg = "start accepts only 'beta' and 'theta'; these entries were ignored: phi, b.")
  )
  for (case in cases) {
    got <- fit_catching(do.call(fastglmm, c(list(y ~ x, FIXED), case$args)),
                        "fastglmm_argument_ignored")
    row <- only(got$fit, "argument_ignored")
    expect_identical(row$message, case$msg)
    expect_identical(conditionMessage(got$cond), printed(row))
    expect_s3_class(got$cond, "fastglmm_diagnostic")
  }
})

test_that("quasi-likelihood dispersion on a mixed binomial is stored", {
  d <- transform(MIXED, y = as.numeric(y > 0))
  got <- fit_catching(fastglmm(y ~ x + (1 | g), d, family = binomial(),
                               dispersion = "estimate"),
                      "fastglmm_argument_ignored")
  row <- only(got$fit, "argument_ignored")
  expect_identical(row$message, paste(
    "Quasi-likelihood dispersion= is not supported yet for binomial or Poisson models.",
    "The default dispersion of 1 was used."))
  expect_identical(conditionMessage(got$cond), printed(row))
})

test_that("agq_fallback is stored and printed with nAGQ=", {
  # A binomial model with two grouping factors: mixed, but outside what AGQ covers.
  d <- transform(MIXED, y = as.numeric(y > 0),
                 h = factor(sprintf("h%d", (seq_along(x) - 1L) %% 4L)))
  got <- fit_catching(fastglmm(y ~ x + (1 | g) + (1 | h), d, family = binomial(), nAGQ = 3),
                      "fastglmm_agq_fallback")
  row <- only(got$fit, "agq_fallback")
  expect_identical(row$message, paste(
    "nAGQ=3 works only for binomial, Poisson, negative-binomial or Gamma models whose",
    "random effects are in one grouping factor, with at most 3 random effects in it.",
    "This model was fitted without adaptive quadrature."))
  expect_identical(conditionMessage(got$cond), printed(row))
  expect_equal(got$fit$nAGQ, 1L)
  expect_identical(got$fit$warnings$kind, "agq_fallback")
})

test_that("constructed notes store with their tier and title", {
  base <- list(columns = integer(0), pivot = NaN, evals = 0L, final_eval = FALSE,
               detail = "", ratio = NaN)
  cases <- list(
    list(note = modifyList(base, list(kind = "unused_grouping_levels", detail = "g: z, w")),
         cls = "fastglmm_unused_grouping_levels",
         msg = paste("Grouping factor 'g' has levels with no rows (z, w). They stay in the",
                     "model with random effects of exactly zero, but they are not counted in",
                     "the number of groups. Remove unused categories before fitting.")),
    list(note = modifyList(base, list(kind = "re_design_scale_spread", detail = "g",
                                      ratio = 4200)),
         cls = "fastglmm_re_design_scale_spread",
         msg = paste("The predictors with random slopes for 'g' are on very different",
                     "scales (ratio 4.2e+03). The fit is not affected, but the reported",
                     "random-effect standard deviations are hard to compare. Rescaling",
                     "these predictors makes them easier to read.")),
    list(note = modifyList(base, list(kind = "single_level_grouping_dropped", detail = "onegroup")),
         cls = "fastglmm_single_level_grouping_dropped",
         msg = paste("Grouping factor 'onegroup' has only one level, so no variance between",
                     "groups can be estimated from it. Its random effect was dropped; the",
                     "rest of the model was fitted without it.")),
    list(note = modifyList(base, list(kind = "hessian_se_fallback")),
         cls = "fastglmm_hessian_se_fallback",
         msg = paste("The usual standard errors could not be computed, so a simpler method",
                     "was used. Its standard errors tend to be too small, so p-values and",
                     "confidence intervals may look more precise than they are. Standard",
                     "errors for the random-effect standard deviations are not available.")),
    list(note = modifyList(base, list(kind = "exact_profile_fallback")),
         cls = "fastglmm_exact_profile_fallback",
         msg = paste("The default search method for this model did not settle on an answer,",
                     "so the fit tried a different search method from the same starting",
                     "point. The reported estimates come from whichever method reached the",
                     "better answer.")),
    list(note = modifyList(base, list(kind = "pirls_exhausted", final_eval = TRUE)),
         cls = "fastglmm_pirls_exhausted",
         msg = paste("The final step that computes the reported results ran out of",
                     "iterations. The estimates and their standard errors may be",
                     "less accurate than usual. Try simplifying the random effects or",
                     "rescaling the predictors.")),
    list(note = modifyList(base, list(kind = "nb_shape_unsettled", evals = 25L)),
         cls = "fastglmm_nb_shape_unsettled",
         msg = paste("The search for the negative binomial shape parameter stopped at its",
                     "limit of 25 rounds before it settled. The coefficients and standard",
                     "errors are computed at the last value it reached, which may not be",
                     "the best one.")),
    list(note = modifyList(base, list(kind = "from_the_future")),
         cls = "fastglmm_unknown_note",
         msg = paste("The solver reported something ('from_the_future') that this installed",
                     "version does not recognize. Please report it at",
                     "https://github.com/pawlenartowicz/glmm/issues."))
  )
  for (case in cases) {
    out <- fastglmm:::.note_warning(case$note, character(0), TRUE)
    expect_identical(out$msg, case$msg, info = case$note$kind)
    expect_identical(out$cls, c(case$cls, "fastglmm_diagnostic"))
    store <- new.env(parent = emptyenv())
    store$rows <- list()
    cond <- tryCatch(fastglmm:::.warn_keep(store, case$note$kind, out$msg, out$cls),
                     warning = identity)
    row <- fastglmm:::.warnings_frame(store)
    tt <- fastglmm:::.WARNING_KINDS[[case$note$kind]] %||% fastglmm:::.UNKNOWN_KIND
    expect_identical(c(row$tier, row$kind, row$title), c(tt[[1]], case$note$kind, tt[[2]]))
    expect_identical(conditionMessage(cond), printed(row))
  }
})

test_that("singular is stored and printed", {
  # tau0 = 0 data: the RE variance pins to the boundary.
  d <- benchmark_data(seed = 107, family = "binomial", tau0 = 1e-8)
  got <- fit_catching(fastglmm(y ~ t + (1 | g), d, family = binomial()),
                      "fastglmm_singular")
  row <- only(got$fit, "singular")
  expect_identical(row$message, paste0(
    "The random effects are too complex for the data: a variance is estimated at or ",
    "near zero, or a correlation at or near −1 or 1. Consider removing the ",
    "affected random effect. Affected: (Intercept) in g."))
  expect_identical(conditionMessage(got$cond), printed(row))
  expect_identical(got$fit$warnings$kind, "singular")
  expect_true(isSingular(got$fit))
})

test_that("pirls_exhausted is raised only on a converged final evaluation", {
  note <- list(kind = "pirls_exhausted", columns = integer(0), pivot = NaN, evals = 3L,
               final_eval = FALSE, detail = "")
  expect_null(fastglmm:::.note_warning(note, character(0), TRUE))
  expect_null(fastglmm:::.note_warning(note, character(0), FALSE))
  note$final_eval <- TRUE
  expect_null(fastglmm:::.note_warning(note, character(0), FALSE))
  expect_false(is.null(fastglmm:::.note_warning(note, character(0), TRUE)))
})

test_that("unused levels splits on the first separator", {
  note <- list(kind = "unused_grouping_levels", columns = integer(0), pivot = NaN,
               evals = 0L, final_eval = FALSE, detail = "g: a: b, c")
  expect_match(fastglmm:::.note_warning(note, character(0), TRUE)$msg,
               "^Grouping factor 'g' has levels with no rows \\(a: b, c\\)\\.")
})

test_that("a single-level grouping is dropped with a warning, keeping the other grouping", {
  n <- 40L
  d <- data.frame(y = rnorm(n), x = rnorm(n),
                  onegroup = factor(rep("only", n)),
                  h = factor(rep(1:4, 10)))
  got <- fit_catching(fastglmm(y ~ x + (1 | onegroup) + (1 | h), d),
                      "fastglmm_single_level_grouping_dropped")
  row <- only(got$fit, "single_level_grouping_dropped")
  expect_identical(row$message, paste(
    "Grouping factor 'onegroup' has only one level, so no variance between groups can be",
    "estimated from it. Its random effect was dropped; the rest of the model was fitted",
    "without it."))
  expect_identical(names(VarCorr(got$fit)), "h")
})

test_that("every grouping single-level fits as a model without random effects", {
  n <- 20L
  d <- data.frame(y = rnorm(n), x = rnorm(n),
                  g1 = factor(rep("only", n)), g2 = factor(rep("one", n)))
  fit <- withCallingHandlers(
    fastglmm(y ~ x + (1 | g1) + (1 | g2), d),
    warning = function(w) invokeRestart("muffleWarning")
  )
  expect_identical(fit$re_group_names, character(0))
  expect_length(fit$varcorr, 0L)
  expect_true(fit$converged)
})

test_that("suppressed warnings are still stored", {
  fit <- suppressWarnings(fastglmm(y ~ x, FIXED, dispersion = 2))
  expect_identical(fit$warnings$kind, "argument_ignored")
})

test_that("the store is per fit", {
  noisy <- suppressWarnings(fastglmm(y ~ x, FIXED, dispersion = 2))
  clean <- fastglmm(y ~ x, FIXED)
  expect_equal(nrow(noisy$warnings), 1L)
  expect_equal(nrow(clean$warnings), 0L)
})

test_that("nAGQ that changes nothing is an ignored argument", {
  cases <- list(
    list(fit = function() fastglmm(y ~ x + (1 | g), MIXED, nAGQ = 3),
         reason = "a Gaussian model"),
    list(fit = function() fastglmm(y ~ x, data.frame(x = x, y = as.numeric(yy > 0)),
                                   family = binomial(), nAGQ = 3),
         reason = "a model without random effects"))
  for (case in cases) {
    got <- fit_catching(case$fit(), "fastglmm_argument_ignored")
    row <- only(got$fit, "argument_ignored")
    expect_identical(row$message, sprintf(
      "nAGQ=3 has no effect for %s, because nothing is approximated.", case$reason))
    expect_identical(conditionMessage(got$cond), printed(row))
    expect_false(any(got$fit$warnings$kind == "agq_fallback"))
    expect_equal(got$fit$nAGQ, 1L)
  }
})

test_that("argument notes come in the documented order", {
  fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), MIXED, dispersion = 2,
                                   init.theta = 1.5, start = list(phi = 1), nAGQ = 3))
  expect_identical(fit$warnings$kind, rep("argument_ignored", 4L))
  expect_identical(substr(fit$warnings$message, 1L, 10L),
                   c("dispersion", "init.theta", "start acce", "nAGQ=3 has"))
})

INNER <- "Some of its inner steps ran out of iterations."
PIRLS <- list(list(kind = "pirls_exhausted", columns = integer(0), pivot = NaN,
                   evals = 3L, final_eval = FALSE, detail = ""))
ADVICE <- "Try a simpler random-effects structure or rescale the predictors."
SEARCH <- function(inner) paste(c(
  "The search for the variance estimates reached its step limit before it settled.",
  inner, "The estimates shown are the best point found; they are often close, but this is",
  "not checked. Do not use them until the fit converges.", ADVICE), collapse = " ")
FAILED <- function(inner) paste(c(
  "The fitting algorithm failed and returned no estimates.", inner, ADVICE), collapse = " ")
BINOMIAL <- paste(
  "The fit did not converge. This usually means a predictor, or a combination of",
  "predictors, predicts the outcome perfectly (separation), so some fitted probabilities",
  "go to 0 or 1. The coefficients are from the last step; standard errors are",
  "not reported. Check the data for separation.")
COUNTS <- paste(
  "The fit did not converge. This usually means that some category of a predictor, or",
  "some combination of predictors, has only zero counts, so some fitted counts go to 0.",
  "The coefficients are from the last step; standard errors are not reported. Check for",
  "categories whose counts are all zero.")
UNSETTLED <- paste(
  "The fit did not settle on an answer: the fitting steps stopped before converging. The",
  "coefficients are from the last step; standard errors are not reported. Check predictors",
  "with extreme values; with a link other than log, the log link is usually more stable.")
UNSOLVABLE <- paste(
  "The predictors could not be separated numerically, so no estimates were computed.",
  "Check for predictors that are copies or near-copies of each other.")

test_that(".nonconvergence picks one kind, in the documented order", {
  fin <- c(1.5, -2)
  nan <- c(NaN, NaN)
  none <- c(FALSE, FALSE)
  nc <- function(family, mixed, notes = list(), beta = fin, aliased = none,
                 deviance = NaN, y = yy) {
    fastglmm:::.nonconvergence(family, mixed, notes, beta, aliased, deviance, y)
  }
  cases <- list(
    list(nc("gaussian", FALSE, beta = nan), "design_unsolvable", UNSOLVABLE),
    list(nc("binomial", FALSE), "glm_diverged", BINOMIAL),
    list(nc("poisson", FALSE), "glm_diverged", COUNTS),
    list(nc("negativebinomial", FALSE, PIRLS), "glm_diverged", COUNTS),
    list(nc("gamma", FALSE), "glm_diverged", UNSETTLED),
    list(nc("inversegaussian", FALSE), "glm_diverged", UNSETTLED),
    list(nc("gaussian", TRUE, deviance = 12.5), "search_limit", SEARCH(character())),
    list(nc("binomial", TRUE, PIRLS, deviance = 12.5), "search_limit", SEARCH(INNER)),
    list(nc("poisson", TRUE, beta = nan), "fit_failed", FAILED(character())),
    list(nc("poisson", TRUE, PIRLS, beta = nan), "fit_failed", FAILED(INNER)),
    # finite coefficients but no end point: failed, not a search limit
    list(nc("gaussian", TRUE), "fit_failed", FAILED(character())),
    # no fixed effects (y ~ 0 + (1 | g)): the deviance decides
    list(nc("gaussian", TRUE, beta = double(), aliased = logical()), "fit_failed",
         FAILED(character())),
    list(nc("gaussian", TRUE, beta = double(), aliased = logical(), deviance = 12.5),
         "search_limit", SEARCH(character())),
    # an aliased NaN slot is not a failed coefficient
    list(nc("gaussian", TRUE, beta = c(1, NaN), aliased = c(FALSE, TRUE), deviance = 12.5),
         "search_limit", SEARCH(character())),
    list(nc("binomial", FALSE, beta = nan, y = rep(0, 20)), "constant_response",
         "Every value of the response is 0, so there is nothing to estimate. Check the response column."),
    list(nc("poisson", TRUE, PIRLS, beta = nan, y = rep(3, 20)), "constant_response",
         "Every value of the response is 3, so there is nothing to estimate. Check the response column."),
    list(nc("gaussian", FALSE, beta = double(), aliased = logical()), "no_coefficients",
         paste("The model has no coefficients and no random effects, so there is nothing",
               "to estimate. Add an intercept or a predictor.")),
    list(nc("gaussian", FALSE, beta = nan, y = c(1, 2)), "too_few_rows", paste(
      "The model has 2 coefficients to estimate but only 2 rows, so no estimates were",
      "computed. Use more rows or fewer predictors.")),
    list(nc("binomial", TRUE, beta = nan, y = 1), "too_few_rows", paste(
      "The model has 2 coefficients to estimate but only 1 row, so no estimates were",
      "computed. Use more rows or fewer predictors.")),
    list(nc("gaussian", FALSE, beta = c(NaN, 1), aliased = c(TRUE, FALSE), y = 4),
         "too_few_rows", paste(
      "The model has 1 coefficient to estimate but only 1 row, so no estimates were",
      "computed. Use more rows or fewer predictors."))
  )
  for (case in cases) {
    out <- case[[1]]
    expect_identical(out$kind, case[[2]])
    expect_identical(out$msg, case[[3]], info = case[[2]])
    expect_identical(out$cls, c(paste0("fastglmm_", case[[2]]), "fastglmm_diagnostic"))
    store <- new.env(parent = emptyenv())
    store$rows <- list()
    cond <- tryCatch(fastglmm:::.warn_keep(store, out$kind, out$msg, out$cls),
                     warning = identity)
    row <- fastglmm:::.warnings_frame(store)
    expect_identical(row$tier, "severe")
    expect_identical(conditionMessage(cond), printed(row))
  }
})

test_that("a separated GLM stores glm_diverged last", {
  d <- data.frame(x = x, y = as.numeric(x > 0))
  got <- fit_catching(fastglmm(y ~ x, d, family = binomial()), "fastglmm_glm_diverged")
  expect_false(got$fit$converged)
  n <- nrow(got$fit$warnings)
  expect_identical(got$fit$warnings$kind[[n]], "glm_diverged")
  expect_identical(conditionMessage(got$cond), printed(got$fit$warnings[n, ]))
  expect_equal(sum(got$fit$warnings$tier == "severe"), 1L)
})

test_that("an all-zero binomial is a constant response, not separation", {
  d <- data.frame(x = x, y = 0)
  fit <- suppressWarnings(fastglmm(y ~ x, d, family = binomial()))
  expect_false(fit$converged)
  expect_identical(fit$warnings$kind, "constant_response")
})

test_that("too few rows end to end", {
  d <- data.frame(y = c(1, 2, 4), x1 = c(0, 1, 3), x2 = c(1, 0, 2))
  fit <- suppressWarnings(fastglmm(y ~ x1 + x2, d))
  expect_false(fit$converged)
  expect_match(fit$warnings$message[[nrow(fit$warnings)]],
               "^The model has 3 coefficients to estimate but only 3 rows")
})

# "no coefficients end to end" (fastglmm(y ~ 0, FIXED)) is not tested here: R's
# formula lowering refuses a fixed-only formula with no fixed-effect column before
# the kernel sees it (the same refusal the Python port hits), so no_coefficients is
# never reached through fastglmm() itself; the constructed case above (the
# .nonconvergence "no_coefficients" row) is the only coverage for this kind.

test_that("summary ends with the stored warnings", {
  fit <- suppressWarnings(fastglmm(y ~ x, FIXED, dispersion = 2))
  out <- capture.output(print(summary(fit)))
  expect_identical(tail(out, 2L), c(
    "Warnings:",
    "Note: Argument ignored. dispersion= has no effect for family 'gaussian'."))
})

test_that("a clean summary has no Warnings section", {
  out <- capture.output(print(summary(fastglmm(y ~ x, FIXED))))
  expect_false(any(out == "Warnings:"))
})

test_that("summary of an object without $warnings prints no section", {
  fit <- fastglmm(y ~ x, FIXED)
  fit$warnings <- NULL
  expect_no_error(out <- capture.output(print(summary(fit))))
  expect_false(any(out == "Warnings:"))
})

test_that("the kind table matches the Python port", {
  expect_identical(fastglmm:::.WARNING_KINDS, list(
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
  ))
  expect_identical(fastglmm:::.UNKNOWN_KIND, c("caution", "Unrecognized solver message"))
})

# One fixture, four notes across three kinds:
# argument_ignored fires twice (dispersion=/nAGQ=, both inapplicable on a
# Gaussian model, from two different arguments) plus the formula lowering's
# own unused_grouping_levels and re_design_scale_spread, from one
# declared-but-empty group level ("g_mid", between two used ones) and one
# random slope on a much larger scale than the implicit intercept. Five
# groups, eight rows each — same shape as the Rust fixture this mirrors
# (re_design_scale_spread_note_fires_on_mismatched_slope_scale,
# src/fit/common_tests.rs). Same fixture and call in
# python/tests/test_warnings.py - change together.
PARITY <- local({
  n_groups <- 5L
  per_group <- 8L
  x_scale <- 1e4
  z <- numeric(0); y <- numeric(0); g <- character(0)
  for (gi in 0:(n_groups - 1L)) {
    for (j in 0:(per_group - 1L)) {
      jitter <- j - (per_group - 1) / 2
      zv <- x_scale + jitter
      z <- c(z, zv)
      y <- c(y, 1.0 + 0.1 * zv / x_scale + 0.05 * gi)
      g <- c(g, sprintf("g%d", gi))
    }
  }
  data.frame(z = z, y = y,
             g = factor(g, levels = c("g0", "g1", "g2", "g_mid", "g3", "g4")))
})

test_that("port parity sequence adds three more kinds", {
  fit <- suppressWarnings(fastglmm(y ~ z + (1 + z | g), PARITY, dispersion = 2, nAGQ = 3))
  expect_true(fit$converged)
  expect_identical(fit$warnings$tier, c("note", "note", "note", "note"))
  expect_identical(fit$warnings$kind, c(
    "argument_ignored", "argument_ignored",
    "unused_grouping_levels", "re_design_scale_spread"
  ))
  expect_identical(fit$warnings$title, c(
    "Argument ignored", "Argument ignored",
    "Unused grouping levels", "Random-effect predictors on very different scales"
  ))
})

test_that("port parity non_integer_response", {
  # Same fixture and call in python/tests/test_warnings.py - change together.
  # Intercept-only Poisson, mirrors the Rust unit test
  # poisson_non_integer_response_is_noted (src/fit/common_tests.rs).
  d <- data.frame(y = c(1, 2.5, 3, 4))
  expect_warning(fit <- fastglmm(y ~ 1, d, family = "poisson"), "not a whole number")
  expect_true(fit$converged)
  expect_identical(fit$warnings$tier, "caution")
  expect_identical(fit$warnings$kind, "non_integer_response")
  expect_identical(fit$warnings$title, "Non-integer response")
})

# 12-row Gamma-inverse GLMM, 4 clusters of 3 — the exact fixture
# gamma_inverse_adversarial(1e-2) builds in src/fit/glmm_tests.rs, with the
# same literal data in python/tests/test_warnings.py - change together. Rows
# 4..6 (R's 1-based indexing; the second cluster) are scaled by 1e-2,
# spreading the 12 means over about ten decades, which is what makes the
# joint Hessian non-PD.
.HSE_X1 <- c(
  -0.2884326052962697, 0.15675840297260962, -0.29990566105127503, 0.3053648563367106,
  -0.608001199771249, 1.451281361998219, -0.12817918698856592, 0.2873778743945057,
  1.2188666482896826, -1.2520230368147756, -0.01938374443978608, 1.0354827975218956
)
.HSE_Y <- c(
  0.013147681629338537, 0.022571111060908807, 0.009380481267257667,
  13915684.733974772 * 1e-2, 16148934.63843452 * 1e-2, 47449285.58840935 * 1e-2,
  0.03659612272549357, 0.03340213953300986, 0.16526466878896995,
  2288.1399088594685, 5963.477254271042, 19049.155692047607
)

test_that("port parity hessian_se_fallback", {
  d <- data.frame(x = .HSE_X1, y = .HSE_Y,
                   g = factor(sprintf("g%d", rep(0:3, each = 3))))
  fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), d, family = Gamma(link = "inverse")))
  expect_true(fit$converged)
  expect_identical(fit$warnings$tier, "caution")
  expect_identical(fit$warnings$kind, "hessian_se_fallback")
  expect_identical(fit$warnings$title, "Simpler standard errors used")
})
