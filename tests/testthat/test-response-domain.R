# Response-domain and dispersion-value validation (GLMM/src/fit/mod.rs, and
# GLMM/src/orchestrate.rs for the dispersion value).
#
# These checks live in Rust; fastglmm() does no pre-check of its own, so every
# case here surfaces as a plain R error from the native fit. Mirrored in
# python/tests/test_response_domain.py - change together.

DATA <- data.frame(y = c(1, 2, 3, 4), x = c(0, 1, 2, 3))

test_that("a negative Poisson response is refused", {
  d <- DATA
  d$y[2] <- -1
  expect_error(fastglmm(y ~ x, d, family = "poisson"), "Poisson response")
})

test_that("a zero or negative dispersion is refused", {
  for (bad in c(0, -1)) {
    expect_error(fastglmm(y ~ x, DATA, family = "gamma", dispersion = bad),
                 "^dispersion must be a finite number > 0, got ")
  }
})
