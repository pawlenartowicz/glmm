# Gate 5 (R-port spec §8): the accuracy benchmark's two formulas x two
# families x both nAGQ arms fit end-to-end through this API and produce
# tau0/tau1/rho01 matching a glmer golden. The package is done when this
# passes. Live-golden (glmer refit on the same data) rather than frozen
# numbers so the comparison is exact-data, not generator-version-dependent;
# lme4 is Suggests-only and the whole file is skipped on CRAN.

skip_if_not_installed("lme4")
skip_on_cran()

glmer_quietly <- function(...) {
  suppressWarnings(suppressMessages(lme4::glmer(...)))
}

expect_matches_glmer <- function(fit, ref, tol_beta = 5e-3, tol_tau = 1e-2,
                                 tol_rho = 5e-2) {
  expect_true(fit$converged)
  expect_equal(unname(fixef(fit)), unname(lme4::fixef(ref)),
               tolerance = tol_beta)
  vc <- VarCorr(fit)$g
  vc_ref <- lme4::VarCorr(ref)$g
  expect_equal(unname(attr(vc, "stddev")), unname(attr(vc_ref, "stddev")),
               tolerance = tol_tau)
  if (nrow(vc) > 1L) {
    expect_equal(attr(vc, "correlation")[2, 1],
                 attr(vc_ref, "correlation")[2, 1],
                 tolerance = tol_rho)
  }
}

test_that("binomial random intercept matches glmer, Laplace and nAGQ = 7", {
  d <- benchmark_data(seed = 201, family = "binomial", tau0 = 0.8)
  f <- y ~ t + d + t:d + (1 | g)
  fit1 <- fastglmm(f, d, family = binomial(), nAGQ = 1)
  ref1 <- glmer_quietly(f, d, family = binomial(), nAGQ = 1)
  expect_matches_glmer(fit1, ref1)
  fit7 <- fastglmm(f, d, family = binomial(), nAGQ = 7)
  ref7 <- glmer_quietly(f, d, family = binomial(), nAGQ = 7)
  expect_matches_glmer(fit7, ref7)
})

test_that("poisson random intercept matches glmer, Laplace and nAGQ = 7", {
  d <- benchmark_data(seed = 202, family = "poisson",
                      beta = c(0, 0.5, 0.3, 0.1), tau0 = 0.6)
  f <- y ~ t + d + t:d + (1 | g)
  fit1 <- fastglmm(f, d, family = poisson(), nAGQ = 1)
  ref1 <- glmer_quietly(f, d, family = poisson(), nAGQ = 1)
  expect_matches_glmer(fit1, ref1)
  fit7 <- fastglmm(f, d, family = poisson(), nAGQ = 7)
  ref7 <- glmer_quietly(f, d, family = poisson(), nAGQ = 7)
  expect_matches_glmer(fit7, ref7)
})

test_that("binomial random slope (tau0, tau1, rho01) matches glmer", {
  # m/tau1 chosen so the slope variance is identified (an interior fit, not a
  # boundary one — a singular fit would weaken the tau1/rho comparison).
  d <- benchmark_data(seed = 203, family = "binomial", n_g = 100L, m = 15L,
                      tau0 = 0.8, tau1 = 0.8, rho = 0.3)
  f <- y ~ t + d + t:d + (1 + t | g)
  fit <- fastglmm(f, d, family = binomial())
  ref <- glmer_quietly(f, d, family = binomial())
  expect_matches_glmer(fit, ref)
})

test_that("poisson random slope (tau0, tau1, rho01) matches glmer", {
  d <- benchmark_data(seed = 204, family = "poisson", n_g = 100L,
                      beta = c(0, 0.5, 0.3, 0.1), tau0 = 0.5, tau1 = 0.4,
                      rho = 0.3)
  f <- y ~ t + d + t:d + (1 + t | g)
  fit <- fastglmm(f, d, family = poisson())
  ref <- glmer_quietly(f, d, family = poisson())
  expect_matches_glmer(fit, ref)
})

test_that("the harness's remaining requirements hold: timeable, flagged", {
  # Their design needs a system.time()-able fit and a convergence flag
  # (spec 'Who this is for') — assert both survive the API.
  d <- benchmark_data(seed = 205, family = "binomial")
  elapsed <- system.time(
    fit <- fastglmm(y ~ t + d + t:d + (1 | g), d, family = binomial())
  )[["elapsed"]]
  expect_true(is.finite(elapsed))
  expect_type(fit$converged, "logical")
})
