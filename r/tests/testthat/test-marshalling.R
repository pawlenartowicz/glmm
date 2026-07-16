# Data marshalling: the R -> Table trip (plan gate 3) and the row filtering
# (subset/na.action) that spec §3.1 assigns to the R side.

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
  fit <- fastglmm(y ~ x, d, na.action = na.omit)
  expect_equal(nobs(fit), 9L)
  expect_error(fastglmm(y ~ x, d, na.action = na.pass),
               "missing values remain")
  expect_error(fastglmm(y ~ x, d, na.action = na.fail), "missing values")
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

test_that("missing and unsupported columns are clean errors", {
  expect_error(fastglmm(y ~ z, ols_data()), "not found in data.*z")
  d <- ols_data()
  d$z <- as.Date("2026-01-01") + 0:9
  expect_error(fastglmm(y ~ z, d), "unsupported type")
})

test_that("logical columns fit as 0/1 numerics", {
  d <- ols_data()
  d$b <- rep(c(TRUE, FALSE), 5)
  fit <- fastglmm(y ~ b, d)
  expect_named(fixef(fit), c("(Intercept)", "b"))
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
