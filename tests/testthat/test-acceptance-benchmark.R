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

test_that("logLik and AIC match lme4, and the LMM value is lmer's REML criterion", {
  d <- benchmark_data(seed = 206, family = "binomial", tau0 = 0.8)
  f <- y ~ t + d + (1 | g)
  fit <- fastglmm(f, d, family = binomial())
  ref <- glmer_quietly(f, d, family = binomial())
  expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(ref)), tolerance = 1e-5)
  expect_equal(attr(logLik(fit), "df"), attr(logLik(ref), "df"))
  expect_equal(AIC(fit), AIC(ref), tolerance = 1e-4)
  expect_false(attr(logLik(fit), "REML"))

  # The LMM path is REML-only, so its logLik is lmer's REML criterion, not an ML
  # value — comparable only across models with identical fixed effects.
  set.seed(207)
  dl <- data.frame(g = factor(rep(1:40, each = 8)), t = stats::rnorm(320))
  dl$y <- 1 + 0.5 * dl$t + stats::rnorm(40)[as.integer(dl$g)] + stats::rnorm(320)
  fl <- fastglmm(y ~ t + (1 | g), dl)
  rl <- suppressWarnings(suppressMessages(lme4::lmer(y ~ t + (1 | g), dl)))
  expect_equal(as.numeric(logLik(fl)), as.numeric(logLik(rl)), tolerance = 1e-6)
  expect_true(attr(logLik(fl), "REML"))
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
