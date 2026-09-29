# Data marshalling: the R -> Table trip and the row filtering
# (subset/na.action) that the R side owns.

ols_data <- function() {
  set.seed(7)
  data.frame(
    y = c(1, 2, 2.9, 4.1, 5, 6.2, 6.8, 8.1, 9, 10.2),
    x = 0:9
  )
}

test_that("gaussian OLS fits and names coefficients", {
  fit <- fastglmm(y ~ x, ols_data(), family = gaussian())
  expect_named(fixef(fit), c("(Intercept)", "x"))
  expect_equal(unname(fixef(fit)[["x"]]),
               unname(coef(lm(y ~ x, ols_data()))[["x"]]),
               tolerance = 1e-8)
  expect_true(fit$converged)
})

test_that("a factor's declared level order survives to the base level (gate 3)", {
  set.seed(11)
  d <- data.frame(
    y = rnorm(60),
    f = factor(rep(c("high", "low", "mid"), 20), levels = c("mid", "low", "high"))
  )
  fit <- fastglmm(y ~ f, d, family = gaussian())
  # Base = declared first level ("mid"), NOT the lexicographic first ("high").
  expect_named(fixef(fit), c("(Intercept)", "flow", "fhigh"))
  ref <- coef(lm(y ~ f, d)) # lm honors the same declared order
  expect_equal(unname(fixef(fit)), unname(ref), tolerance = 1e-8)
})

test_that("character columns get lexicographic levels (factor() default)", {
  set.seed(12)
  d <- data.frame(
    y = rnorm(60),
    f = rep(c("b", "c", "a"), 20),
    stringsAsFactors = FALSE
  )
  fit <- fastglmm(y ~ f, d, family = gaussian())
  expect_named(fixef(fit), c("(Intercept)", "fb", "fc"))
})

test_that("subset= filters rows before fitting", {
  d <- ols_data()
  fit <- fastglmm(y ~ x, d, subset = x < 5)
  expect_equal(nobs(fit), 5L)
})

test_that("na.omit drops NA rows; na.pass-style leftovers error", {
  d <- ols_data()
  d$y[3] <- NA
  expect_warning(fit <- fastglmm(y ~ x, d, na.action = na.omit),
                 "Dropped 1 of 10 row")
  expect_equal(nobs(fit), 9L)
  expect_error(fastglmm(y ~ x, d, na.action = na.pass),
               "missing values remain")
  expect_error(fastglmm(y ~ x, d, na.action = na.fail), "missing values")
})

test_that("every row missing in a used column gives the same message as the Python port", {
  # Mirrors python/tests/test_validation.py's
  # test_all_rows_missing_in_a_used_column_is_a_plain_error.
  d <- data.frame(y = c(1, 2, 3), x = c(NA_real_, NA_real_, NA_real_))
  expect_error(
    fastglmm(y ~ x, d),
    "every row has a missing value in a column the formula uses; no rows left to fit"
  )
})

test_that("weights are honored; zero/short/negative weights are clean errors", {
  d <- ols_data()
  w <- rep(c(2, 1), 5)
  fit <- fastglmm(y ~ x, d, weights = w)
  ref <- lm(y ~ x, d, weights = w)
  expect_equal(unname(fixef(fit)), unname(coef(ref)), tolerance = 1e-8)
  expect_error(fastglmm(y ~ x, d, weights = c(1, 2)), "one entry")
  expect_error(fastglmm(y ~ x, d, weights = rep(-1, 10)), "positive")
  expect_error(fastglmm(y ~ x, d, weights = c(0, rep(1, 9))), "positive")
})

test_that("offset= is honored as an expression and stays aligned under subset=", {
  set.seed(21)
  d <- data.frame(x = rnorm(40), exposure = runif(40, 1, 10))
  d$y <- rpois(40, d$exposure * exp(0.3 + 0.5 * d$x))

  fit <- fastglmm(y ~ x, d, family = poisson(), offset = log(exposure))
  ref <- glm(y ~ x, poisson(), d, offset = log(exposure))
  expect_equal(unname(fixef(fit)), unname(coef(ref)), tolerance = 1e-8)

  # The alignment hazard: an offset read off `data` after the row filtering
  # would pair each kept row with the wrong exposure. That fit still converges
  # and still looks reasonable, so only a comparison against the same rows
  # filtered up front catches it.
  keep <- d$x > 0
  sub_fit <- fastglmm(y ~ x, d, family = poisson(), offset = log(exposure),
                      subset = x > 0)
  pre_fit <- fastglmm(y ~ x, d[keep, ], family = poisson(),
                      offset = log(exposure))
  expect_equal(nobs(sub_fit), sum(keep))
  expect_equal(unname(fixef(sub_fit)), unname(fixef(pre_fit)),
               tolerance = 1e-10)

  expect_error(fastglmm(y ~ x, d, family = poisson(), offset = rep(0, 3)),
               "one entry")
})

test_that("missing and unsupported columns are clean errors", {
  expect_error(fastglmm(y ~ z, ols_data()), "not found in data.*z")
  d <- ols_data()
  d$z <- as.Date("2026-01-01") + 0:9
  expect_error(fastglmm(y ~ z, d), "unsupported type")
})

test_that("logical columns are factors with FALSE/TRUE levels, as in lme4", {
  set.seed(31)
  d <- data.frame(g = rep(c(FALSE, TRUE), each = 30), h = factor(rep(1:6, 10)),
                  b = rep(c(TRUE, FALSE), 30), x = rnorm(60))
  d$y <- d$x + 0.5 * d$b + rnorm(6)[d$h] + ifelse(d$g, 0.4, -0.4) + rnorm(60)
  d01 <- transform(d, g = as.integer(g), b = as.integer(b))

  # Grouping: lme4 labels the levels "FALSE"/"TRUE"; the 0/1 coding it
  # replaces fits bit-identically, only the labels change.
  fit <- fastglmm(y ~ x + (1 | g) + (1 | h), d)
  fit01 <- fastglmm(y ~ x + (1 | g) + (1 | h), d01)
  expect_equal(rownames(ranef(fit)$g), c("FALSE", "TRUE"))
  expect_identical(fixef(fit), fixef(fit01))
  expect_identical(unname(unlist(ranef(fit))), unname(unlist(ranef(fit01))))
  expect_identical(as.numeric(logLik(fit)), as.numeric(logLik(fit01)))

  # Fixed effect: the dummy is bTRUE, same numbers as the 0/1 column.
  fit <- fastglmm(y ~ b * x + (1 | h), d)
  fit01 <- fastglmm(y ~ b * x + (1 | h), d01)
  expect_named(fixef(fit), c("(Intercept)", "bTRUE", "x", "bTRUE:x"))
  expect_identical(unname(fixef(fit)), unname(fixef(fit01)))

  # Where R's marginality rule codes the logical by indicators, the design
  # is model.matrix()'s, not a single 0/1 column.
  expect_named(fixef(fastglmm(y ~ x:b + (1 | h), d)),
               colnames(model.matrix(y ~ x:b, d)))
  expect_named(fixef(fastglmm(y ~ 0 + b + (1 | h), d)),
               colnames(model.matrix(y ~ 0 + b, d)))

  # A logical response stays 0/1, as glm() reads it.
  d$yb <- d$y > 0
  fit <- fastglmm(yb ~ x, d, family = binomial())
  ref <- glm(yb ~ x, binomial(), d)
  expect_equal(unname(fixef(fit)), unname(coef(ref)), tolerance = 1e-8)
})

test_that("unused data columns are never marshalled", {
  d <- ols_data()
  d$junk <- replicate(10, list(1)) # unmarshallable, but not in the formula
  expect_silent(fit <- fastglmm(y ~ x, d))
  expect_true(fit$converged)
})

test_that("formula strings are accepted", {
  fit <- fastglmm("y ~ x", ols_data())
  expect_equal(formula(fit), "y ~ x")
})
