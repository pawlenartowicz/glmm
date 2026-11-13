# Deterministic fixtures, mirrored in glmm-py/tests/test_warnings.py.
x <- as.numeric(setdiff(-10:10, 0))
# Parenthesized: %% binds tighter than * in R, unlike Python's %.
yy <- 0.3 * x + ((((seq_along(x) - 1L) * 7L) %% 5L) - 2L) * 0.4
FIXED <- data.frame(x = x, y = yy)
g <- factor(sprintf("g%d", (seq_along(x) - 1L) %% 5L))
off <- c(g0 = -3, g1 = -1.5, g2 = 0, g3 = 1.5, g4 = 3)
MIXED <- data.frame(x = x, g = g, y = yy + off[as.character(g)])
# MIXED with groups of rows (i %/% 2) %% 5, so y - 0.3x is not constant within a
# group (in MIXED it is, and the LMM fit rides the theta cap): theta-hat is
# interior and sigma-hat^2 is not tiny.
g_pairs <- factor(sprintf("g%d", ((seq_along(x) - 1L) %/% 2L) %% 5L))
MIXED_PAIRS <- data.frame(x = x, g = g_pairs, y = yy + off[as.character(g_pairs)])

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
  row
}

test_that("a clean fit stores a zero-row data.frame", {
  # MIXED_PAIRS, not MIXED: MIXED rides the theta limit and warns.
  expect_no_warning(fit <- fastglmm(y ~ x + (1 | g), MIXED_PAIRS))
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
         msg = "start accepts only 'beta', 'theta' and 'dispersion'; these entries were ignored: phi, b."),
    list(args = list(start = list(dispersion = 2)),
         msg = "start$dispersion is not used for family 'gaussian'.")
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

test_that("any note kind gets the class fastglmm_<kind>", {
  expect_identical(fastglmm:::.note_class("some_new_kind"),
                   c("fastglmm_some_new_kind", "fastglmm_diagnostic"))
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
    list(fit = function() fastglmm(y ~ x + (1 | g), MIXED_PAIRS, nAGQ = 3),
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
  # MIXED_PAIRS, not MIXED: MIXED rides the theta limit and warns.
  fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), MIXED_PAIRS, dispersion = 2,
                                   init.theta = 1.5, start = list(phi = 1), nAGQ = 3))
  expect_identical(fit$warnings$kind, rep("argument_ignored", 4L))
  expect_identical(substr(fit$warnings$message, 1L, 10L),
                   c("dispersion", "init.theta", "start acce", "nAGQ=3 has"))
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

test_that("a GLM whose first solve fails returns no coefficients", {
  # A 1e-100 weight, the smallest allowed, on the only row that tells x1 from x2
  # makes X'WX singular to rounding: no step is taken, so no coefficient is reported.
  d <- data.frame(x1 = 1, x2 = c(0, 1, 1, 1), y = c(1, 2, 3, 2.5))
  got <- fit_catching(fastglmm(y ~ 0 + x1 + x2, d, family = Gamma(link = "log"),
                               weights = c(1e-100, 1, 1, 1)), "fastglmm_design_unsolvable")
  expect_false(got$fit$converged)
  expect_true(all(is.nan(fixef(got$fit))))
  expect_false(is.null(got$cond))
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
# never reached through fastglmm() itself; the "no_coefficients" row of the failure
# table in glmm-bind/src/notes.rs is the only coverage for this kind.

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

# One fixture: argument_ignored fires twice (dispersion=/nAGQ=, both
# inapplicable on a Gaussian model, from two different arguments). Its declared
# but empty group level ("g_mid", between two used ones) is dropped and raises
# nothing. Five groups, eight rows each, with a random slope on z, whose values sit far from zero. The fit adds two cautions
# of its own: y has no random slope, so the slope's variance comes out at
# zero, and glmm compares standard deviations on its internally standardized
# design, where that is a singular fit (on the raw design z's distance from
# zero hid it); and y is exact in z and g, so the residual variance goes to
# zero and the intercept's theta ends on the upper limit of its search. Same
# fixture and call in glmm-py/tests/test_warnings.py - change together.
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
  expect_identical(fit$warnings$tier, c("note", "note", "caution", "caution"))
  expect_identical(fit$warnings$kind, c(
    "argument_ignored", "argument_ignored", "singular", "variance_at_limit"
  ))
  expect_identical(fit$warnings$title, c(
    "Argument ignored", "Argument ignored", "Singular fit", "Variance at its search limit"
  ))
})

test_that("port parity non_integer_response", {
  # Same fixture and call in glmm-py/tests/test_warnings.py - change together.
  # Intercept-only Poisson, mirrors the Rust unit test
  # poisson_non_integer_response_is_noted (src/fit/common_tests.rs).
  d <- data.frame(y = c(1, 2.5, 3, 4))
  expect_warning(fit <- fastglmm(y ~ 1, d, family = "poisson"), "not a whole number")
  expect_true(fit$converged)
  expect_identical(fit$warnings$tier, "caution")
  expect_identical(fit$warnings$kind, "non_integer_response")
  expect_identical(fit$warnings$title, "Non-integer response")
})

test_that("port parity few_rows_for_random_effects", {
  # A random intercept and slope on 10 subjects, two of them with one row: 20
  # random effects for 18 rows. Same fixture and call in
  # glmm-py/tests/test_warnings.py - change together.
  d <- data.frame(
    y = c(0.26, -0.09, 0.46, -1.73, -0.38, -0.91, -0.86, 0.91, -1.42,
          -0.40, -1.23, -1.62, -1.46, -0.49, 1.31, 0.85, 0.06, 0.36),
    time = c(0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1),
    id = factor(sprintf("s%d", c(1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10)))
  )
  res <- fit_catching(fastglmm(y ~ time + (1 + time | id), d),
                      "fastglmm_few_rows_for_random_effects")
  expect_false(is.null(res$cond))
  row <- only(res$fit, "few_rows_for_random_effects")
  expect_identical(c(row$tier, row$title), c("caution", "Too few rows for the random effects"))
  expect_identical(row$message, paste(
    "The term (1 + time | id) has 20 random effects for 18 rows, so its variance",
    "parameters probably cannot be estimated from these data; lme4 stops on such a model",
    "by default. The estimates are shown; read the variance components with care, or",
    "simplify the random-effects structure."))
  expect_identical(conditionMessage(res$cond), printed(row))
})

# 12 groups of 8 rows; g2 is g under other names, cat is constant within g
# (two categories), and yb is 0 or 1 within every g. Same fixture and calls in
# glmm-py/tests/test_warnings.py - change together.
id_g <- (seq_len(96L) - 1L) %/% 8L
id_x <- as.numeric(((seq_len(96L) - 1L) * 5L) %% 8L)
IDENT <- data.frame(
  x = id_x,
  g = factor(sprintf("g%02d", id_g)),
  g2 = factor(sprintf("h%02d", 11L - id_g)),
  cat = factor(ifelse(id_g < 6L, "A", "B")),
  y = 0.3 * id_x + 0.5 * ((id_g * 7L) %% 5L) + 0.2 * (((seq_len(96L) - 1L) * 3L) %% 7L),
  yb = as.numeric(id_g %% 2L == 0L),
  yv = as.numeric((((seq_len(96L) - 1L) * 5L) %% 3L) == 0L),
  z = 0
)

test_that("port parity repeated_grouping", {
  res <- fit_catching(fastglmm(y ~ x + (1 | g) + (1 | g2), IDENT),
                      "fastglmm_repeated_grouping")
  expect_false(is.null(res$cond))
  expect_s3_class(res$cond, "fastglmm_diagnostic")
  row <- only(res$fit, "repeated_grouping")
  expect_identical(c(row$tier, row$title), c("caution", "Same groups under two names"))
  expect_identical(row$message, paste(
    "Grouping factors 'g' and 'g2' put the rows into exactly the same groups and share a",
    "random effect, so the data cannot tell their variances apart. How the variance is",
    "split between them is arbitrary. Remove one of the two from the model."))
  expect_identical(conditionMessage(res$cond), printed(row))
})

test_that("port parity slope_constant_within_levels", {
  res <- fit_catching(fastglmm(y ~ cat + (1 + cat | g), IDENT),
                      "fastglmm_slope_constant_within_levels")
  expect_false(is.null(res$cond))
  row <- only(res$fit, "slope_constant_within_levels")
  expect_identical(c(row$tier, row$title), c("caution", "Random slope constant within groups"))
  expect_identical(row$message, paste(
    "The random slope on catB does not vary within any group of 'g', so within each group",
    "it cannot be told apart from the random intercept. Its variance and correlations",
    "probably cannot be estimated from these data. Consider removing it from the",
    "random-effects term."))
  expect_identical(conditionMessage(res$cond), printed(row))
})

test_that("port parity separated_by_grouping", {
  res <- fit_catching(fastglmm(yb ~ x + (1 | g), IDENT, family = binomial()),
                      "fastglmm_separated_by_grouping")
  expect_false(is.null(res$cond))
  row <- only(res$fit, "separated_by_grouping")
  expect_identical(c(row$tier, row$title), c("caution", "Response constant within groups"))
  expect_identical(row$message, paste(
    "The response is all 0 or all 1 within every group of 'g', so the likelihood keeps",
    "rising as this grouping's variance grows, and the variance has no finite estimate.",
    "Any reported variance is a stopping point, not an estimate, and the fixed effects",
    "depend on it. Consider dropping this grouping, or a model with a prior on the",
    "variance."))
  expect_identical(conditionMessage(res$cond), printed(row))
})

test_that("port parity one_trial_per_level", {
  # Same fixture and call in glmm-py/tests/test_warnings.py - change together.
  # One 0/1 row per level of id.
  i <- 0:19
  d <- data.frame(x = x, id = sprintf("r%02d", i), yb = as.numeric((7L * i) %% 3L == 0L))
  res <- fit_catching(fastglmm(yb ~ x + (1 | id), d, family = binomial()),
                      "fastglmm_one_trial_per_level")
  expect_false(is.null(res$cond))
  row <- only(res$fit, "one_trial_per_level")
  expect_identical(c(row$tier, row$title), c("caution", "Variance not identified"))
  expect_identical(row$message, paste(
    "Every level of 'id' has one binary trial, so its variance is not identified: it",
    "trades against the fixed effects. Drop this grouping, or aggregate rows."))
  expect_identical(conditionMessage(res$cond), printed(row))
})

test_that("a mixed fit with every fixed column aliased", {
  # Same fixture and call in glmm-py/tests/test_warnings.py - change together. z
  # is all zero, so it is dropped as aliased and the mixed model has no fixed
  # effect left to fit.
  fit <- suppressWarnings(fastglmm(yv ~ 0 + z + (1 | g) + (1 | g2), IDENT,
                                   family = binomial()))
  expect_false(fit$converged)
  expect_identical(fit$warnings$kind, c("repeated_grouping", "no_coefficients"))
  expect_identical(only(fit, "no_coefficients")$message, paste(
    "The model has no fixed-effect coefficients: every fixed-effect column is all zero.",
    "glmm needs at least one fixed effect in a mixed model. Add an intercept or a",
    "predictor."))
})

test_that("port parity variance_at_limit", {
  # Same fixture and call in glmm-py/tests/test_warnings.py - change together. On
  # MIXED, y - 0.3x is constant within each group, so the residual variance goes
  # to zero and theta ends on the upper limit of its search.
  res <- fit_catching(fastglmm(y ~ x + (1 | g), MIXED), "fastglmm_variance_at_limit")
  expect_false(is.null(res$cond))
  expect_s3_class(res$cond, "fastglmm_diagnostic")
  expect_true(res$fit$converged)
  expect_false(res$fit$singular)
  expect_identical(res$fit$diagnostics$boundary, "upper_limit")
  expect_identical(res$fit$warnings$kind, "variance_at_limit")
  row <- only(res$fit, "variance_at_limit")
  expect_identical(c(row$tier, row$title), c("caution", "Variance at its search limit"))
  expect_identical(row$message, paste(
    "A random-effect parameter reached the upper limit of the search (1000 on glmm's",
    "internal scale). On a Gaussian model this usually means the residual variance is",
    "estimated at or near zero; on other families the random-effect variance has no",
    "finite estimate. The estimates are at that limit, not at an optimum. Check for a",
    "response that the predictors and groups explain exactly."))
  expect_identical(conditionMessage(res$cond), printed(row))
})

test_that("port parity dispersion_at_limit", {
  # Same fixture and call in glmm-py/tests/test_warnings.py - change together. A
  # Gamma response within about 1e-8 of its log-link means: the dispersion is
  # near 1e-16, below the smallest value the fit searches.
  i <- 0:119
  gid <- i %/% 10L
  xg <- sin(2.3 * i + 0.7)
  b <- 0.3 * sin(1.9 * (0:11) + 0.4)
  e <- sqrt(2) * sin(1.7 * i + 0.3)
  mu <- exp(0.5 + 0.3 * xg + b[gid + 1L])
  d <- data.frame(x = xg, g = sprintf("g%d", gid), y = mu * (1 + 1e-8 * e))
  res <- fit_catching(fastglmm(y ~ x + (1 | g), d, family = Gamma(link = "log")),
                      "fastglmm_dispersion_at_limit")
  expect_false(is.null(res$cond))
  expect_false(res$fit$converged)
  expect_identical(res$fit$warnings$kind, "dispersion_at_limit")
  row <- only(res$fit, "dispersion_at_limit")
  expect_identical(c(row$tier, row$title), c("severe", "Dispersion at its search limit"))
  expect_identical(row$message, paste(
    "The Gamma dispersion fell below 1e-06, the smallest value the fit searches, so no",
    "estimates were computed. The response varies less than about 0.1% around the",
    "fitted means."))
  expect_identical(conditionMessage(res$cond), printed(row))
})

test_that("a degenerate Gaussian LMM carries the Satterthwaite note", {
  # Same fixture and call in glmm-py/tests/test_warnings.py - change together.
  # Three rows for three coefficients (n <= p) leave a NaN-filled fit with no
  # finite end point, so the Satterthwaite pass is recorded as failed.
  d <- data.frame(y = c(1, 2, 4), x1 = c(0, 1, 3), x2 = c(1, 0, 2),
                  g = factor(c("a", "b", "a")))
  got <- fit_catching(fastglmm(y ~ x1 + x2 + (1 | g), d),
                      "fastglmm_satterthwaite_curvature")
  expect_false(is.null(got$cond))
  expect_match(conditionMessage(got$cond), "could not be computed", fixed = TRUE)
  expect_false(got$fit$converged)
  expect_identical(got$fit$df_method, "failed")
  expect_identical(got$fit$warnings$kind, c("satterthwaite_curvature", "too_few_rows"))
  off <- suppressWarnings(fastglmm(y ~ x1 + x2 + (1 | g), d, satterthwaite = FALSE))
  expect_identical(off$df_method, "disabled")
  expect_identical(off$warnings$kind, "too_few_rows")
})
