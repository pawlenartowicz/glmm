# Spec §4's error table, row by row: everything the engine or the shared
# parser cannot do must be an error naming the reason (Decision 5) — and the
# parser-limit errors must say what to do instead.

err_data <- function() {
  set.seed(3)
  data.frame(y = rnorm(40), x = rnorm(40), s = rnorm(40), f = rnorm(40),
             g = factor(rep(1:8, 5)))
}

test_that("function calls in formulas point at computing the column first", {
  d <- err_data()
  expect_error(fastglmm(y ~ log(x), d), "compute the column first")
  expect_error(fastglmm(y ~ I(x^2), d), "compute the column first")
  expect_error(fastglmm(y ~ poly(x, 2), d), "compute the column first")
  expect_error(fastglmm(y ~ scale(x), d), "compute the column first")
})

test_that("term removal surfaces the parser's own message", {
  d <- err_data()
  expect_error(fastglmm(y ~ x - 1, d), "identifier|removal",
               ignore.case = TRUE)
})

test_that("cbind() points at proportion + weights", {
  expect_error(fastglmm(cbind(s, f) ~ x, err_data(), family = binomial()),
               "proportion.*weights|weights.*proportion")
})

test_that("dot formulas error", {
  expect_error(fastglmm(y ~ ., err_data()), "'\\.' is not supported")
})

test_that("double-bar and intercept-free RE terms name the kernel property", {
  d <- err_data()
  expect_error(fastglmm(y ~ x + (x || g), d), "full RE correlation")
  expect_error(fastglmm(y ~ x + (0 + x | g), d), "always full")
  expect_error(fastglmm(y ~ x + (-1 + x | g), d), "always full")
})

test_that("intercepted lme4 arguments raise designed errors", {
  d <- err_data()
  expect_error(fastglmm(y ~ x + (1 | g), d, REML = FALSE),
               "REML-only by design")
  # REML = TRUE matches what the engine does — no error (warnings, e.g. a
  # boundary fit on this null-effect data, are fine).
  expect_no_error(suppressWarnings(fastglmm(y ~ x + (1 | g), d, REML = TRUE)))
  expect_error(fastglmm(y ~ x, d, control = list()), "compiled into")
  expect_error(fastglmm(y ~ x, d, verbose = TRUE), "verbose")
  expect_error(fastglmm(y ~ x, d, contrasts = list(g = "contr.sum")),
               "relevel")
  expect_error(fastglmm(y ~ x, d, offset = rep(1, 40)), "offset")
  expect_error(fastglmm(y ~ x + offset(x), d), "offset")
  expect_error(fastglmm(y ~ x, d, bogus = 1), "unused argument")
})

test_that("GLMM 0.1.1 families/links error naming the spec", {
  d <- err_data()
  expect_error(fastglmm(y ~ x, d, family = inverse.gaussian()),
               "0\\.1\\.1")
  d$yb <- rbinom(40, 1, 0.5)
  expect_error(fastglmm(yb ~ x, d, family = binomial("cloglog")),
               "0\\.1\\.1")
  expect_error(fastglmm(yb ~ x, d, family = binomial(), dispersion = 2),
               "quasi-likelihood.*0\\.1\\.1")
})

test_that("init.theta has no kernel hook; wrong-family use warns and strips", {
  d <- err_data()
  d$yc <- rpois(40, 2)
  expect_error(fastglmm(yc ~ x, d, family = "negativebinomial",
                        init.theta = 1.5),
               "no kernel hook")
  expect_warning(fit <- fastglmm(y ~ x, d, init.theta = 1.5),
                 "applies only to family 'negativebinomial'")
  expect_true(fit$converged)
})

test_that("MASS::negative.binomial-style fixed-theta family objects error", {
  fam <- structure(list(family = "Negative Binomial(2)", link = "log"),
                   class = "family")
  expect_error(fastglmm(y ~ x, err_data(), family = fam),
               "estimates it")
})

test_that("nAGQ must be an odd integer in 1..=25", {
  d <- err_data()
  expect_error(fastglmm(y ~ x, d, nAGQ = 2), "odd integer")
  expect_error(fastglmm(y ~ x, d, nAGQ = 27), "odd integer")
  expect_error(fastglmm(y ~ x, d, nAGQ = 0), "odd integer")
})

test_that("ineligible-shape nAGQ > 1 warns and falls back to Laplace", {
  # gaussian LMM is AGQ-ineligible: must warn and fit, never panic or error
  # (mirrors the Python port; lme4 would error here — documented divergence).
  d <- err_data()
  # capture_warnings: the null-effect data also (legitimately) warns about a
  # boundary fit, which a single expect_warning would trip over.
  w <- capture_warnings(fit <- fastglmm(y ~ x + (1 | g), d, nAGQ = 3))
  expect_match(w, "Laplace", all = FALSE)
  expect_equal(fit$nAGQ, 1L)
  expect_true(fit$converged)
})

test_that("engine-blocked accessors error with the reason", {
  d <- err_data()
  fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), d)) # boundary fit is fine here
  expect_error(ranef(fit), "engine-blocked")
  expect_error(predict(fit), "engine-blocked")
  expect_error(fitted(fit), "engine-blocked")
  expect_error(residuals(fit), "engine-blocked")
  expect_error(coef(fit), "fixef")
  expect_error(terms(fit), "formula\\(\\) returns")
  expect_error(confint(fit, method = "profile"), "no profiling machinery")
})

test_that("dispersion on a non-dispersion family warns and strips", {
  expect_warning(fit <- fastglmm(y ~ x, err_data(), dispersion = 2),
                 "not applicable")
  expect_true(fit$converged)
})
