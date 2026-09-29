# Response-domain and dispersion-value validation (GLMM/src/fit/mod.rs).
#
# These checks live in the Rust kernel; fastglmm() does no pre-check of its
# own, so every case here surfaces as a plain R error through the same
# catch_unwind path as a bad weights=/offset=. Mirrored in
# python/tests/test_response_domain.py - change together.

DATA <- data.frame(y = c(1, 2, 3, 4), x = c(0, 1, 2, 3))

test_that("a negative Poisson response is refused", {
  d <- DATA
  d$y[2] <- -1
  expect_error(fastglmm(y ~ x, d, family = "poisson"), "Poisson response")
})

test_that("a negative negative-binomial response is refused", {
  d <- DATA
  d$y[2] <- -1
  expect_error(fastglmm(y ~ x, d, family = "negativebinomial"), "negative-binomial response")
})

test_that("a binomial response above 1 is refused", {
  d <- DATA
  d$y <- c(0, 1, 2, 0)
  expect_error(fastglmm(y ~ x, d, family = "binomial"), "binomial response.*\\[0, 1\\]")
})

test_that("a binomial response below 0 is refused", {
  d <- DATA
  d$y[2] <- -0.1
  expect_error(fastglmm(y ~ x, d, family = "binomial"), "binomial response.*\\[0, 1\\]")
})

test_that("a non-positive Gamma response is refused", {
  d <- DATA
  d$y[2] <- 0
  expect_error(fastglmm(y ~ x, d, family = "gamma"), "Gamma response")
})

test_that("a non-positive inverse-Gaussian response is refused", {
  d <- DATA
  d$y[2] <- -3
  expect_error(fastglmm(y ~ x, d, family = "inversegaussian"), "inverse-Gaussian response")
})

test_that("a negative dispersion is refused", {
  expect_error(fastglmm(y ~ x, DATA, family = "gamma", dispersion = -1), "dispersion")
})

test_that("a zero dispersion is refused", {
  expect_error(fastglmm(y ~ x, DATA, family = "gamma", dispersion = 0), "dispersion")
})

test_that("a non-integer Poisson response still fits, with a warning", {
  # Intercept-only, mirrors the Rust unit test
  # poisson_non_integer_response_is_noted (src/fit/common_tests.rs) exactly.
  d <- data.frame(y = c(1, 2.5, 3, 4))
  expect_warning(f <- fastglmm(y ~ 1, d, family = "poisson"), "not a whole number")
  expect_true(f$converged)
})

test_that("non-integer binomial successes still fit, with a warning", {
  # Intercept-only, mirrors the Rust unit test
  # binomial_non_integer_successes_is_noted (src/fit/common_tests.rs) exactly.
  d <- data.frame(y = c(0.5, 0.5, 0.5, 0.6))
  expect_warning(
    f <- fastglmm(y ~ 1, d, family = "binomial", weights = c(2, 2, 2, 3)),
    "not a whole number"
  )
  expect_true(f$converged)
})
