# The accuracy benchmark's datasets, pinned against this package's own output.
#
# The live comparison against lme4 lives in tools/acceptance-vs-lme4.R and is
# run by hand: lme4 is not a dependency of this package and is not installed on
# the CI runner. What runs here instead is a regression gate. Every number below
# was recorded once from a local build of this package, at full printed
# precision, and compared at CI_REF_REL (helper-pins.R).
#
# What that gates and what it does not: it gates the WRAPPER and the kernel
# together -- a formula lowered to the wrong design, a factor coded the wrong
# way, a nAGQ argument dropped, a variance component reported on the wrong
# scale, or a kernel answer that moved, all show up here. It says nothing about
# whether the number is right in the first place; that claim belongs to the
# cross-engine comparisons, which are local.
#
# The gamma block says at its own site how its numbers were checked.

test_that("binomial random intercept is pinned, Laplace and nAGQ = 7", {
  d <- benchmark_data(seed = 201, family = "binomial", tau0 = 0.8)
  f <- y ~ t + d + t:d + (1 | g)

  fit1 <- fastglmm(f, d, family = binomial(), nAGQ = 1)
  expect_true(fit1$converged)
  expect_equal(unname(fixef(fit1)), c(-0.627395162630681, 0.895926376377332, 0.247561594016142, 0.000413479727165042),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit1)$g, "stddev")),
               c(0.861924739905157), tolerance = CI_REF_REL)

  fit7 <- fastglmm(f, d, family = binomial(), nAGQ = 7)
  expect_true(fit7$converged)
  expect_equal(unname(fixef(fit7)), c(-0.625829018638988, 0.893716432457535, 0.244015519350444, 0.00516534124410457),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit7)$g, "stddev")),
               c(0.882844883191443), tolerance = CI_REF_REL)
})

test_that("poisson random intercept is pinned, Laplace and nAGQ = 7", {
  d <- benchmark_data(seed = 202, family = "poisson",
                      beta = c(0, 0.5, 0.3, 0.1), tau0 = 0.6)
  f <- y ~ t + d + t:d + (1 | g)

  fit1 <- fastglmm(f, d, family = poisson(), nAGQ = 1)
  expect_true(fit1$converged)
  expect_equal(unname(fixef(fit1)), c(-0.203228366780273, 0.725803635018043, 0.291810281351584, -0.198487326683721),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit1)$g, "stddev")),
               c(0.583345352507545), tolerance = CI_REF_REL)

  fit7 <- fastglmm(f, d, family = poisson(), nAGQ = 7)
  expect_true(fit7$converged)
  expect_equal(unname(fixef(fit7)), c(-0.20355106814039, 0.72589349782474, 0.29184692395272, -0.198680382346346),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit7)$g, "stddev")),
               c(0.58547094722065), tolerance = CI_REF_REL)
})

test_that("binomial random slope (tau0, tau1, rho01) is pinned", {
  # m/tau1 chosen so the slope variance is identified (an interior fit, not a
  # boundary one -- a singular fit would weaken the tau1/rho pin).
  d <- benchmark_data(seed = 203, family = "binomial", n_g = 100L, m = 15L,
                      tau0 = 0.8, tau1 = 0.8, rho = 0.3)
  fit <- fastglmm(y ~ t + d + t:d + (1 + t | g), d, family = binomial())
  expect_true(fit$converged)
  vc <- VarCorr(fit)$g
  expect_equal(unname(fixef(fit)), c(-0.792588774256419, 1.07410694961377, 0.694132964295869, -0.129934518298598),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(vc, "stddev")), c(1.04201267969287, 0.871286201276444),
               tolerance = CI_REF_REL)
  expect_equal(attr(vc, "correlation")[2, 1], 0.106174638142076,
               tolerance = CI_REF_REL)
})

test_that("poisson random slope (tau0, tau1, rho01) is pinned", {
  d <- benchmark_data(seed = 204, family = "poisson", n_g = 100L,
                      beta = c(0, 0.5, 0.3, 0.1), tau0 = 0.5, tau1 = 0.4,
                      rho = 0.3)
  fit <- fastglmm(y ~ t + d + t:d + (1 + t | g), d, family = poisson())
  expect_true(fit$converged)
  vc <- VarCorr(fit)$g
  expect_equal(unname(fixef(fit)), c(0.207518881588951, 0.43021623995375, 0.119056414540409, 0.338551938152971),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(vc, "stddev")), c(0.312862312451542, 0.507728342122289),
               tolerance = CI_REF_REL)
  expect_equal(attr(vc, "correlation")[2, 1], 0.349341452882515,
               tolerance = CI_REF_REL)
})

test_that("gamma GLMM is pinned", {
  # Same grouping/time grid as benchmark_data(), built inline (that helper
  # only generates binomial/poisson), with a gamma response on the log link.
  #
  # Re-recorded 2026-09-24, when the mixed-Gamma fit became the maximum of the
  # Laplace likelihood with phi estimated by ML: glmmTMB 1.1.14 on the same data
  # reaches logLik -807.967379848965 (this fit: -807.967379870621), the same
  # SD to 2e-6 and fixed effects to 3e-5 relative. The pin still gates only
  # the wrapper and the kernel; that agreement is the cross-engine check,
  # made by hand.
  set.seed(208)
  n_g <- 60L
  m <- 10L
  g <- factor(rep(seq_len(n_g), each = m))
  t <- rep(seq(0, length.out = m, by = 1 / m), n_g)
  dtrt <- rbinom(n_g * m, 1L, 0.4)
  u0 <- rnorm(n_g, sd = 0.3)
  beta <- c(0.5, 0.3, 0.2, 0.1)
  eta <- beta[1] + beta[2] * t + beta[3] * dtrt + beta[4] * t * dtrt +
    u0[as.integer(g)]
  shape <- 4
  y <- rgamma(n_g * m, shape = shape, rate = shape / exp(eta))
  d <- data.frame(y = y, t = t, d = dtrt, g = g)
  fit <- fastglmm(y ~ t + d + t:d + (1 | g), d, family = Gamma(link = "log"))
  expect_true(fit$converged)
  expect_equal(unname(fixef(fit)), c(0.4885792356687608, 0.2112947287513839, 0.1567059342545184, 0.0491456628104056),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit)$g, "stddev")),
               c(0.258718996469143), tolerance = CI_REF_REL)
  expect_equal(sigma(fit), 0.492147539449984, tolerance = CI_REF_REL)
})

test_that("logLik and AIC are pinned, and the LMM value is the REML criterion", {
  d <- benchmark_data(seed = 206, family = "binomial", tau0 = 0.8)
  fit <- fastglmm(y ~ t + d + (1 | g), d, family = binomial())
  expect_equal(as.numeric(logLik(fit)), -388.97838234408,
               tolerance = CI_REF_REL)
  expect_equal(as.integer(attr(logLik(fit), "df")), 4L)
  expect_equal(AIC(fit), 785.956764688159, tolerance = CI_REF_REL)
  expect_false(attr(logLik(fit), "REML"))

  # The LMM path is REML-only, so its logLik is the REML criterion, not an ML
  # value -- comparable only across models with identical fixed effects.
  set.seed(207)
  dl <- data.frame(g = factor(rep(1:40, each = 8)), t = stats::rnorm(320))
  dl$y <- 1 + 0.5 * dl$t + stats::rnorm(40)[as.integer(dl$g)] + stats::rnorm(320)
  fl <- fastglmm(y ~ t + (1 | g), dl)
  expect_equal(as.numeric(logLik(fl)), -489.587906036441,
               tolerance = CI_REF_REL)
  expect_true(attr(logLik(fl), "REML"))
})

test_that("the harness's remaining requirements hold: timeable, flagged", {
  # A system.time()-able fit and a convergence flag -- assert both survive the
  # API.
  d <- benchmark_data(seed = 205, family = "binomial")
  elapsed <- system.time(
    fit <- fastglmm(y ~ t + d + t:d + (1 | g), d, family = binomial())
  )[["elapsed"]]
  expect_true(is.finite(elapsed))
  expect_type(fit$converged, "logical")
})
