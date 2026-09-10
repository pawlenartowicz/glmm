# Extracted from test-errors.R:96

# setup ------------------------------------------------------------------------
library(testthat)
test_env <- simulate_test_env(package = "fastglmm", path = "..")
attach(test_env, warn.conflicts = FALSE)

# prequel ----------------------------------------------------------------------
err_data <- function() {
  set.seed(3)
  data.frame(y = rnorm(40), x = rnorm(40), s = rnorm(40), f = rnorm(40),
             g = factor(rep(1:8, 5)))
}

# test -------------------------------------------------------------------------
d <- err_data()
d$yb <- rbinom(40, 1, 0.5)
expect_error(fastglmm(yb ~ x, d, family = binomial(), dispersion = 2),
               "quasi-likelihood.*0\\.1\\.1")
