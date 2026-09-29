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

test_that("a zero dispersion is refused", {
  expect_error(fastglmm(y ~ x, DATA, family = "gamma", dispersion = 0), "dispersion")
})
