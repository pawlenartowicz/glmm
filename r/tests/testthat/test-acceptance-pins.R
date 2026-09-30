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

  # Re-recorded 2026-09-26, when the no-RE GLM fit that seeds this fit began
  # to stop on R's relative deviance rule: the cold start moved, and the fit
  # lands at logLik -393.542896593029, within 6.3e-13 of the previous point.
  # The fixed effects moved by at most 8e-7 absolute (the last one, 1.6e-4
  # relative on a value near 0.005).
  #
  # Re-recorded 2026-09-28, when the joint search over theta and beta began to
  # step the fixed effects in centred, scaled coordinates: logLik
  # -393.542896593027, 2e-12 higher. The fixed effects moved by at most 5.5e-7
  # absolute (the last one 6.5e-5 relative), the SD by 9e-8 relative.
  fit7 <- fastglmm(f, d, family = binomial(), nAGQ = 7)
  expect_true(fit7$converged)
  expect_equal(unname(fixef(fit7)), c(-0.62582891071147, 0.893716403487087, 0.244015538480626, 0.00516485243902018),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit7)$g, "stddev")),
               c(0.882844802317575), tolerance = CI_REF_REL)
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

  # Re-recorded 2026-09-28, when the joint search over theta and beta began to
  # step the fixed effects in centred, scaled coordinates (logLik
  # -923.041795291571). The fixed effects moved by at most 7.3e-7 absolute,
  # the SD by 2.3e-7 relative.
  fit7 <- fastglmm(f, d, family = poisson(), nAGQ = 7)
  expect_true(fit7$converged)
  expect_equal(unname(fixef(fit7)), c(-0.203551164425076, 0.725894230553309, 0.291846747816096, -0.198680230560223),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit7)$g, "stddev")),
               c(0.585471084582996), tolerance = CI_REF_REL)
})

test_that("binomial random slope (tau0, tau1, rho01) is pinned", {
  # m/tau1 chosen so the slope variance is identified (an interior fit, not a
  # boundary one -- a singular fit would weaken the tau1/rho pin).
  #
  # Re-recorded 2026-09-28, when PIRLS's exact beta-profile step gained a trust
  # region (logLik -923.756172968025): the correlation moved by 6e-8 absolute,
  # the other values by at most 1.1e-8.
  # Re-recorded 2026-09-30, when that step began to compute the curvature of
  # log|A| in beta only on solves where the plain step shows trouble (logLik
  # -923.756172968027): the correlation moved by 6.0e-8 absolute, the other
  # values by at most 6.9e-9.
  d <- benchmark_data(seed = 203, family = "binomial", n_g = 100L, m = 15L,
                      tau0 = 0.8, tau1 = 0.8, rho = 0.3)
  fit <- fastglmm(y ~ t + d + t:d + (1 + t | g), d, family = binomial())
  expect_true(fit$converged)
  vc <- VarCorr(fit)$g
  expect_equal(unname(fixef(fit)), c(-0.792588774256419, 1.074106949613768, 0.694132964295869, -0.129934518298598),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(vc, "stddev")), c(1.042012679692871, 0.871286201276444),
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
  # reaches logLik -807.967379848965 (this fit then: -807.967379870621), the same
  # SD to 2e-6 and fixed effects to 3e-5 relative. The pin still gates only
  # the wrapper and the kernel; that agreement is the cross-engine check,
  # made by hand.
  #
  # Re-recorded 2026-09-26, when PIRLS began to take the Newton step on the
  # log link: this fit now reaches logLik -807.967379832934, 3.8e-8 higher
  # than before and 1.6e-8 above glmmTMB's. The fixed effects moved by up to
  # 1.4e-4 relative, the SD 1.2e-6 and sigma 6.7e-6.
  #
  # Re-recorded 2026-09-28, after the trust region on PIRLS's beta step and the
  # centred, scaled fixed-effect coordinates of the joint search: logLik
  # -807.967379830268, 2.7e-12 higher. The fixed effects moved by at most
  # 1.3e-4 relative (the last one, 6.3e-6 absolute), the SD 3.4e-6 and sigma
  # 7.2e-7.
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
  expect_equal(unname(fixef(fit)), c(0.4885683550922643, 0.2113069197054022, 0.156708904999734, 0.0491449551112931),
               tolerance = CI_REF_REL)
  expect_equal(unname(attr(VarCorr(fit)$g, "stddev")),
               c(0.258718428847095), tolerance = CI_REF_REL)
  expect_equal(sigma(fit), 0.492144615806503, tolerance = CI_REF_REL)
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
