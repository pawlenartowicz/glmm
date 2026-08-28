# Spec §4's error table, row by row: everything the engine or the shared
# parser cannot do must be an error naming the reason (Decision 5) — and the
# parser-limit errors must say what to do instead.

err_data <- function() {
  set.seed(3)
  data.frame(y = rnorm(40), x = rnorm(40), s = rnorm(40), f = rnorm(40),
             g = factor(rep(1:8, 5)))
}

test_that("log(x) and I(x^2) formula terms match the equivalent pre-computed column", {
  d <- err_data()
  d$x <- abs(d$x) + 0.1 # log() needs a positive column

  f1 <- fastglmm(y ~ log(x), d)
  d2 <- transform(d, lx = log(x))
  f2 <- fastglmm(y ~ lx, d2)
  expect_equal(unname(fixef(f1)), unname(fixef(f2)), tolerance = 1e-10)

  f3 <- fastglmm(y ~ I(x^2), d)
  d3 <- transform(d, x2 = x^2)
  f4 <- fastglmm(y ~ x2, d3)
  expect_equal(unname(fixef(f3)), unname(fixef(f4)), tolerance = 1e-10)

  expect_error(fastglmm(y ~ poly(x, 2), d), "formula syntax error")
})

test_that("term removal (- 1) fits without an intercept", {
  d <- err_data()
  fit <- fastglmm(y ~ x - 1, d)
  expect_equal(names(fixef(fit)), c("x"))
  expect_false("(Intercept)" %in% names(fixef(fit)))
})

test_that("cbind() matches the equivalent proportion + weights model, and forbids both together", {
  d <- err_data()
  d$s <- abs(d$s) + 0.5 # cbind() successes/failures must be positive
  d$f <- abs(d$f) + 0.5
  f1 <- fastglmm(cbind(s, f) ~ x, d, family = binomial())
  d2 <- transform(d, p = s / (s + f))
  f2 <- fastglmm(p ~ x, d2, family = binomial(), weights = s + f)
  expect_equal(unname(fixef(f1)), unname(fixef(f2)), tolerance = 1e-10)
  # The trial counts come back as the fit's weights, so Pearson residuals
  # carry the sqrt(trials) factor the hand-weighted fit has.
  expect_equal(unname(f1$weights), d$s + d$f)
  expect_equal(unname(residuals(f1, type = "pearson")),
               unname(residuals(f2, type = "pearson")), tolerance = 1e-10)
  expect_error(
    fastglmm(cbind(s, f) ~ x, d, family = binomial(), weights = s + f),
    "use one"
  )
  d$s[1] <- -2
  expect_error(fastglmm(cbind(s, f) ~ x, d, family = binomial()), "non-negative")
})

test_that("dot formulas error", {
  expect_error(fastglmm(y ~ ., err_data()), "'\\.' is not supported")
})

test_that("double-bar and intercept-free RE terms name the kernel property", {
  d <- err_data()
  expect_error(fastglmm(y ~ x + (x || g), d), "full RE correlation")
  expect_error(fastglmm(y ~ x + (0 + x | g), d), "always full")
  expect_error(fastglmm(y ~ x + (-1 + x | g), d), "always full")
})

test_that("intercepted lme4 arguments raise designed errors", {
  d <- err_data()
  expect_error(fastglmm(y ~ x + (1 | g), d, REML = FALSE),
               "REML-only by design")
  # REML = TRUE matches what the engine does — no error (warnings, e.g. a
  # boundary fit on this null-effect data, are fine).
  expect_no_error(suppressWarnings(fastglmm(y ~ x + (1 | g), d, REML = TRUE)))
  expect_error(fastglmm(y ~ x, d, control = list()), "compiled into")
  expect_error(fastglmm(y ~ x, d, verbose = TRUE), "verbose")
  expect_error(fastglmm(y ~ x, d, contrasts = list(g = "contr.sum")),
               "relevel")
  expect_error(fastglmm(y ~ x, d, bogus = 1), "unused argument")
})

test_that("offset() formula term matches the offset= argument, and forbids both together", {
  d <- err_data()
  f1 <- fastglmm(y ~ x + offset(x), d)
  f2 <- fastglmm(y ~ x, d, offset = x)
  expect_equal(fixef(f1), fixef(f2), tolerance = 1e-10)
  expect_error(
    fastglmm(y ~ x + offset(x), d, offset = x),
    "use one"
  )
})

test_that("quasi-likelihood dispersion on binomial errors naming the spec", {
  d <- err_data()
  d$yb <- rbinom(40, 1, 0.5)
  expect_error(fastglmm(yb ~ x, d, family = binomial(), dispersion = 2),
               "quasi-likelihood.*0\\.1\\.1")
})

test_that("cloglog GLM fits", {
  set.seed(1)
  n <- 300
  x <- rnorm(n)
  mu <- 1 - exp(-exp(0.2 + 0.8 * x))
  d <- data.frame(y = rbinom(n, 1, mu), x = x)
  f <- fastglmm(y ~ x, data = d, family = binomial(link = "cloglog"))
  expect_true(f$converged)
  expect_equal(length(f$beta), 2L)
})

test_that("inverse-Gaussian GLM fits on both links and refuses random effects", {
  set.seed(2)
  n <- 400
  x <- rnorm(n)
  mu <- exp(0.3 + 0.2 * x)
  lam <- 3
  v <- rnorm(n)^2
  x1 <- mu + mu^2 * v / (2 * lam) -
    (mu / (2 * lam)) * sqrt(4 * mu * lam * v + mu^2 * v^2)
  y <- ifelse(runif(n) <= mu / (mu + x1), x1, mu^2 / x1)
  d <- data.frame(y = y, x = x, g = factor(rep(1:20, each = n / 20)))
  f <- fastglmm(y ~ x, data = d, family = inverse.gaussian())
  expect_true(f$converged)
  expect_gt(f$dispersion, 0)
  f2 <- fastglmm(y ~ x, data = d, family = inverse.gaussian(link = "1/mu^2"))
  expect_true(f2$converged)
  expect_error(
    fastglmm(y ~ x + (1 | g), data = d, family = inverse.gaussian()),
    "inverse-Gaussian mixed models"
  )
})

test_that("inverse-Gaussian accepts dispersion = \"estimate\"", {
  set.seed(3)
  d <- data.frame(y = rgamma(200, 4, 2) + 0.1, x = rnorm(200))
  f <- fastglmm(y ~ x, data = d, family = inverse.gaussian(),
                dispersion = "estimate")
  expect_true(f$converged)
})

test_that("init.theta has no kernel hook; wrong-family use warns and strips", {
  d <- err_data()
  d$yc <- rpois(40, 2)
  expect_error(fastglmm(yc ~ x, d, family = "negativebinomial",
                        init.theta = 1.5),
               "no kernel hook")
  expect_warning(fit <- fastglmm(y ~ x, d, init.theta = 1.5),
                 "applies only to family 'negativebinomial'")
  expect_true(fit$converged)
})

test_that("MASS::negative.binomial-style fixed-theta family objects error", {
  fam <- structure(list(family = "Negative Binomial(2)", link = "log"),
                   class = "family")
  expect_error(fastglmm(y ~ x, err_data(), family = fam),
               "estimates it")
})

test_that("nAGQ must be an odd integer in 1..=25", {
  d <- err_data()
  expect_error(fastglmm(y ~ x, d, nAGQ = 2), "odd integer")
  expect_error(fastglmm(y ~ x, d, nAGQ = 27), "odd integer")
  expect_error(fastglmm(y ~ x, d, nAGQ = 0), "odd integer")
})

test_that("ineligible-shape nAGQ > 1 warns and falls back to Laplace", {
  # gaussian LMM is AGQ-ineligible: must warn and fit, never panic or error
  # (mirrors the Python port; lme4 would error here — documented divergence).
  d <- err_data()
  # capture_warnings: the null-effect data also (legitimately) warns about a
  # boundary fit, which a single expect_warning would trip over.
  w <- capture_warnings(fit <- fastglmm(y ~ x + (1 | g), d, nAGQ = 3))
  expect_match(w, "Laplace", all = FALSE)
  expect_equal(fit$nAGQ, 1L)
  expect_true(fit$converged)
})

test_that("unimplemented accessors error with the reason", {
  d <- err_data()
  fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), d)) # boundary fit is fine here
  expect_error(predict(fit), "not available")
  expect_error(coef(fit), "fixef")
  expect_error(terms(fit), "formula\\(\\) returns")
  expect_error(confint(fit, method = "profile"), "no profiling machinery")
})

test_that("dispersion on a non-dispersion family warns and strips", {
  expect_warning(fit <- fastglmm(y ~ x, err_data(), dispersion = 2),
                 "not applicable")
  expect_true(fit$converged)
})
