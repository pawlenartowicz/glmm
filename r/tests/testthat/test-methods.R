# Method contracts: VarCorr's vech -> SD/correlation math (the spec's
# load-bearing accessor), vcov/confint consistency, the Gamma() link trap,
# and the accessors' shapes.

test_that("VarCorr returns SD/correlation-scale components for a slope model", {
  d <- benchmark_data(seed = 101, family = "binomial",
                      tau0 = 0.8, tau1 = 0.5, rho = 0.3)
  fit <- fastglmm(y ~ t + d + t:d + (1 + t | g), d, family = binomial())
  vc <- VarCorr(fit)
  expect_named(vc, "g")
  v <- vc$g
  expect_equal(dim(v), c(2L, 2L))
  expect_equal(rownames(v), c("(Intercept)", "t"))
  sd <- attr(v, "stddev")
  corr <- attr(v, "correlation")
  # The attributes must be exactly the sqrt-diagonal / normalized off-diagonal
  # of the covariance block (spec §2: take sqrt(diag), normalize).
  expect_equal(unname(sd), unname(sqrt(diag(v))), tolerance = 1e-12)
  expect_equal(corr[2, 1], v[2, 1] / (sd[[1]] * sd[[2]]), tolerance = 1e-12)
  expect_true(abs(corr[2, 1]) <= 1)
})

test_that("vcov is the full symmetric matrix and se its sqrt-diagonal", {
  d <- benchmark_data(seed = 102, family = "poisson", beta = c(0, 0.5, 0.3, 0.1))
  fit <- fastglmm(y ~ t + d + t:d + (1 | g), d, family = poisson())
  v <- vcov(fit)
  expect_equal(dim(v), c(4L, 4L))
  expect_equal(v, t(v), tolerance = 1e-12)
  expect_equal(unname(sqrt(diag(v))), unname(fit$se), tolerance = 1e-10)
})

test_that("confint is Wald off vcov", {
  d <- benchmark_data(seed = 103, family = "binomial")
  fit <- fastglmm(y ~ t + d + (1 | g), d, family = binomial())
  ci <- confint(fit, level = 0.9)
  q <- qnorm(0.95)
  expect_equal(unname(ci[, 1]), unname(fixef(fit) - q * fit$se),
               tolerance = 1e-12)
  ci_t <- confint(fit, parm = "t")
  expect_equal(rownames(ci_t), "t")
})

test_that("the Gamma() link trap: object honored, string means log", {
  set.seed(21)
  d <- data.frame(x = rnorm(80))
  d$y <- rgamma(80, shape = 4, rate = 4 / exp(0.3 + 0.5 * d$x))
  fit_str <- fastglmm(y ~ x, d, family = "gamma")
  expect_equal(fit_str$family$link, "log")
  fit_obj <- fastglmm(y ~ x, d, family = Gamma()) # R's default: inverse
  expect_equal(fit_obj$family$link, "inverse")
  ref <- glm(y ~ x, data = d, family = Gamma(link = "log"))
  expect_equal(unname(fixef(fit_str)), unname(coef(ref)), tolerance = 1e-5)
  # sigma() for gamma is sqrt(phi), phi the Pearson dispersion.
  expect_equal(sigma(fit_str)^2, summary(ref)$dispersion, tolerance = 1e-5)
})

test_that("sigma is fixed at 1 for binomial/poisson (lme4 parity)", {
  d <- benchmark_data(seed = 104, family = "binomial")
  fit <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  expect_identical(sigma(fit), 1)
})

test_that("gaussian sigma matches lm on a fixed-only fit", {
  set.seed(22)
  d <- data.frame(x = rnorm(100))
  d$y <- 1 + 2 * d$x + rnorm(100, sd = 0.7)
  fit <- fastglmm(y ~ x, d)
  ref <- lm(y ~ x, data = d)
  expect_equal(sigma(fit), sigma(ref), tolerance = 1e-8)
})

test_that("gaussian LMM sigma matches lme4 and VarCorr prints a Residual row", {
  skip_if_not_installed("lme4")
  set.seed(106)
  g <- factor(rep(1:40, each = 8))
  x <- rnorm(320)
  d <- data.frame(y = 1 + 0.5 * x + rnorm(40, sd = 0.9)[as.integer(g)] +
                    rnorm(320, sd = 0.6),
                  x = x, g = g)
  fit <- fastglmm(y ~ x + (1 | g), d)
  ref <- lme4::lmer(y ~ x + (1 | g), data = d, REML = TRUE)
  expect_equal(sigma(fit), sigma(ref), tolerance = 1e-4)
  vc <- VarCorr(fit)
  expect_equal(attr(vc, "sc"), sigma(fit))
  expect_match(paste(capture.output(print(vc)), collapse = "\n"), "Residual")
})

test_that("accessors: nobs, formula string, family, model.frame, isSingular", {
  d <- benchmark_data(seed = 105, family = "binomial")
  fit <- fastglmm(y ~ t + d + (1 | g), d, family = binomial())
  expect_equal(nobs(fit), nrow(d))
  expect_equal(formula(fit), "y ~ t + d + (1 | g)")
  expect_s3_class(family(fit), "family")
  expect_equal(family(fit)$family, "binomial")
  mf <- model.frame(fit)
  expect_setequal(names(mf), c("y", "t", "d", "g"))
  expect_type(isSingular(fit), "logical")
})

test_that("print and summary run and carry the honest header", {
  d <- benchmark_data(seed = 106, family = "binomial")
  fit <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  out <- capture.output(print(fit))
  expect_true(any(grepl("Laplace", out)))
  sout <- capture.output(print(summary(fit)))
  # No faked AIC/logLik header line (spec §2).
  expect_false(any(grepl("AIC", sout)))
  expect_true(any(grepl("Random effects", sout)))
  expect_true(any(grepl("groups: g, 60", sout)))
})

test_that("boundary fits warn with lme4's exact text and flag isSingular", {
  # tau0 = 0 data: the RE variance pins to the boundary.
  d <- benchmark_data(seed = 107, family = "binomial", tau0 = 1e-8)
  expect_warning(fit <- fastglmm(y ~ t + (1 | g), d, family = binomial()),
                 "boundary \\(singular\\) fit: see help\\('isSingular'\\)")
  expect_true(isSingular(fit))
})

test_that("rank-deficient designs mirror lme4's NA coefficients", {
  d <- benchmark_data(seed = 108, family = "binomial")
  d$t2 <- d$t # aliased copy
  fit <- fastglmm(y ~ t + t2 + (1 | g), d, family = binomial())
  expect_true(fit$aliased[["t2"]])
  expect_true(is.na(fixef(fit)[["t2"]]))
  expect_true(fit$converged)
})

test_that("warm start (lme4's start=) is accepted and unknown parts warn", {
  d <- benchmark_data(seed = 109, family = "binomial")
  cold <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  warm <- fastglmm(y ~ t + (1 | g), d, family = binomial(),
                   start = list(beta = unname(fixef(cold)), theta = 0.7))
  expect_equal(unname(fixef(warm)), unname(fixef(cold)), tolerance = 1e-4)
  expect_warning(
    fastglmm(y ~ t + (1 | g), d, family = binomial(),
             start = list(beta = unname(fixef(cold)), theta = 0.7, bogus = 1)),
    "start elements ignored: bogus"
  )
})
