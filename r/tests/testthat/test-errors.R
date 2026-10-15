# The error table, row by row: everything the engine or the shared
# parser cannot do must be an error naming the reason — and the
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

test_that("cbind() matches the equivalent proportion + weights model", {
  d <- err_data()
  d$s <- abs(d$s) + 0.5 # cbind() successes/failures must be positive
  d$f <- abs(d$f) + 0.5
  # Non-integer successes (s is not a whole number), so both calls raise
  # non_integer_response - the response comparison below still holds.
  expect_warning(f1 <- fastglmm(cbind(s, f) ~ x, d, family = binomial()),
                 "not a whole number")
  d2 <- transform(d, p = s / (s + f))
  expect_warning(f2 <- fastglmm(p ~ x, d2, family = binomial(), weights = s + f),
                 "not a whole number")
  expect_equal(unname(fixef(f1)), unname(fixef(f2)), tolerance = 1e-10)
  # The trial counts come back as the fit's weights, so Pearson residuals
  # carry the sqrt(trials) factor the hand-weighted fit has.
  expect_equal(unname(f1$weights), d$s + d$f)
  expect_equal(unname(residuals(f1, type = "pearson")),
               unname(residuals(f2, type = "pearson")), tolerance = 1e-10)
})

test_that("dot formulas error", {
  expect_error(fastglmm(y ~ ., err_data()), "'\\.' is not supported")
})

test_that("the crossing operator ^ errors with the Python port's message", {
  # R's own parser refuses a formula ending in `+`, so only `^` is tested here.
  d <- transform(err_data(), a = x, b = s, c = f)
  expect_error(fastglmm(y ~ (a + b + c)^2 + (1 | g), d),
               paste("formula syntax error at position 15: the crossing operator '^' is not",
                     "supported: write the terms out (for example (a + b)^2 as a + b + a:b)"),
               fixed = TRUE)
})

test_that("double-bar and intercept-free RE terms name the kernel property", {
  d <- err_data()
  expect_error(fastglmm(y ~ x + (x || g), d), "full RE correlation")
  # Intercept-free RE terms are not pre-checked in R: the shared Rust parser
  # already raises a specific message for this (RandomInterceptSuppressionUnsupported),
  # and it reaches both ports unchanged - see python/tests/test_validation.py's
  # equivalent case.
  expect_error(fastglmm(y ~ x + (0 + x | g), d), "intercept suppression")
  expect_error(fastglmm(y ~ x + (-1 + x | g), d), "intercept suppression")
})

test_that("intercepted lme4 arguments raise designed errors", {
  d <- err_data()
  expect_error(fastglmm(y ~ x, d, control = list()), "compiled into")
  expect_error(fastglmm(y ~ x, d, verbose = TRUE), "verbose")
  expect_error(fastglmm(y ~ x, d, contrasts = list(g = "contr.sum")),
               "relevel")
  expect_error(fastglmm(y ~ x, d, bogus = 1), "unused argument")
})

# 20 groups of 10 with a real between-group variance: the Gaussian, binomial
# and Poisson mixed fits and the Gamma GLM on it raise no warning, so a warning
# in the REML tests below can only come from the argument.
reml_data <- function() {
  set.seed(11)
  g <- factor(rep(seq_len(20L), each = 10L))
  x <- rnorm(200)
  u <- rnorm(20, sd = 0.8)[as.integer(g)]
  data.frame(y = 1 + 0.5 * x + u + rnorm(200), x = x, g = g,
             yb = rbinom(200, 1L, plogis(-0.3 + 0.6 * x + u)),
             yc = rpois(200, exp(0.2 + 0.4 * x + u)),
             yp = rgamma(200, shape = 5, rate = 5 / exp(0.5 + 0.3 * x)),
             g1 = factor(rep("a", 200)))
}

# Every field of the two fits identical, bitwise. Not compared: `call`, and
# the family object, whose closures differ between two `binomial()` calls (its
# family and link are compared); `warnings` too when `same_warnings` is FALSE.
expect_same_fit <- function(f1, f0, same_warnings = TRUE) {
  a <- unclass(f1)
  b <- unclass(f0)
  expect_identical(names(a), names(b))
  skip <- c("call", "family", if (!same_warnings) "warnings")
  for (field in setdiff(names(b), skip)) {
    expect_identical(a[[field]], b[[field]], label = field)
  }
  expect_identical(f1$family[c("family", "link")], f0$family[c("family", "link")])
}

test_that("REML = FALSE on a Gaussian mixed model fits by ML", {
  d <- reml_data()
  f0 <- fastglmm(y ~ x + (1 | g), d)
  expect_no_warning(f1 <- fastglmm(y ~ x + (1 | g), d, REML = FALSE))
  expect_false(f1$reml)
  expect_false(attr(logLik(f1), "REML"))
  # ML shrinks the between-group SD below the REML one on this design.
  sd <- function(f) attr(VarCorr(f)$g, "stddev")
  expect_lt(sd(f1), sd(f0))
})

test_that("REML = FALSE on OLS and on a GLM is the default fit", {
  d <- reml_data()
  expect_no_warning(f1 <- fastglmm(y ~ x, d, REML = FALSE))
  expect_same_fit(f1, fastglmm(y ~ x, d))
  expect_false(f1$reml)
  expect_no_warning(f1 <- fastglmm(yb ~ x, d, family = binomial(), REML = FALSE))
  expect_same_fit(f1, fastglmm(yb ~ x, d, family = binomial()))
  expect_false(f1$reml)
})

test_that("REML = TRUE on a Gaussian mixed model is the default fit", {
  d <- reml_data()
  f0 <- fastglmm(y ~ x + (1 | g), d)
  expect_no_warning(f1 <- fastglmm(y ~ x + (1 | g), d, REML = TRUE))
  expect_true(f1$reml)
  expect_same_fit(f1, f0)
})

test_that("REML on a binomial GLMM: TRUE is ignored with a warning, FALSE silently", {
  d <- reml_data()
  f0 <- fastglmm(yb ~ x + (1 | g), d, family = binomial())
  msg <- paste("REML = TRUE applies only to Gaussian models; it was ignored and the",
               "model was fitted by maximum likelihood.")
  w <- expect_warning(
    f_true <- fastglmm(yb ~ x + (1 | g), d, family = binomial(), REML = TRUE),
    class = "fastglmm_argument_ignored")
  expect_identical(conditionMessage(w), paste("Note: Argument ignored.", msg))
  expect_no_warning(
    f_false <- fastglmm(yb ~ x + (1 | g), d, family = binomial(), REML = FALSE))
  expect_same_fit(f_true, f0, same_warnings = FALSE)
  expect_same_fit(f_false, f0)
  expect_identical(f_true$warnings$kind, "argument_ignored")
  expect_identical(f_true$warnings$message, msg)
  expect_false(f_true$reml)
})

test_that("REML = FALSE on a Poisson GLMM and a Gamma GLM is the default fit", {
  d <- reml_data()
  for (case in list(list(yc ~ x + (1 | g), poisson()),
                    list(yp ~ x, Gamma(link = "log")))) {
    f0 <- fastglmm(case[[1]], d, family = case[[2]])
    expect_no_warning(f1 <- fastglmm(case[[1]], d, family = case[[2]], REML = FALSE))
    expect_same_fit(f1, f0)
    expect_false(f1$reml)
  }
})

test_that("REML = TRUE on OLS reports the REML log-likelihood, as logLik(lm, REML = TRUE)", {
  d <- reml_data()
  f0 <- fastglmm(y ~ x, d)
  expect_no_warning(f1 <- fastglmm(y ~ x, d, REML = TRUE))
  expect_true(f1$reml)
  expect_identical(f1$beta, f0$beta)
  expect_identical(f1$se, f0$se)
  expect_identical(f1$dispersion, f0$dispersion)
  ll <- stats::logLik(stats::lm(y ~ x, d), REML = TRUE)
  expect_equal(as.numeric(logLik(f1)), as.numeric(ll), tolerance = 1e-10)
  expect_equal(attr(logLik(f1), "df"), attr(ll, "df"))
})

test_that("REML is accepted on a formula whose only grouping is dropped", {
  d <- reml_data()
  f0 <- suppressWarnings(fastglmm(y ~ x + (1 | g1), d))
  for (reml in c(TRUE, FALSE)) {
    expect_warning(f <- fastglmm(y ~ x + (1 | g1), d, REML = reml),
                   class = "fastglmm_single_level_grouping_dropped")
    expect_identical(f$warnings$kind, "single_level_grouping_dropped")
    expect_identical(f$reml, reml)
    expect_identical(f$beta, f0$beta)
    # Fitted as OLS, so the OLS row applies: TRUE reports the REML log-likelihood.
    ll <- stats::logLik(stats::lm(y ~ x, d), REML = reml)
    expect_equal(as.numeric(logLik(f)), as.numeric(ll), tolerance = 1e-10)
  }
})

test_that("REML must be one non-NA logical", {
  d <- err_data()
  expect_error(fastglmm(y ~ x + (1 | g), d, REML = NA),
               "REML must be NULL, TRUE or FALSE, got NA", fixed = TRUE)
  expect_error(fastglmm(y ~ x + (1 | g), d, REML = c(TRUE, FALSE)), "TRUE or FALSE")
  # No coercion: a number or a string is not a logical, and an empty one is
  # not "absent".
  for (bad in list(1, "TRUE", logical(0))) {
    expect_error(fastglmm(y ~ x + (1 | g), d, REML = bad), "TRUE or FALSE")
  }
})

test_that("non-treatment contrasts on a fixed-effect factor error instead of fitting silently", {
  set.seed(6)
  n <- 40L
  d <- data.frame(y = rnorm(n), x = rnorm(n),
                  fc = factor(sample(letters[1:3], n, replace = TRUE)),
                  g = factor(rep(1:8, 5)))

  # Baseline: default contrasts fit fine. Only 3 grouping levels, so a
  # boundary/singular warning here is fine, per the pattern above.
  expect_no_error(suppressWarnings(fastglmm(y ~ x + fc + (1 | g), d)))

  # Ordered factor used as a fixed effect errors, wherever it sits in the
  # formula - grouping factors are exempt.
  d_ord <- d
  d_ord$fc <- factor(d_ord$fc, ordered = TRUE)
  expect_error(fastglmm(y ~ x + fc + (1 | g), d_ord),
               "ordered factor.*treatment contrasts")
  # An ordered factor used ONLY as a grouping factor is not checked.
  expect_no_error(suppressWarnings(fastglmm(y ~ x + (1 | fc), d_ord)))

  # A factor with its own contrasts() attribute errors when it is a fixed
  # effect.
  d_attr <- d
  contrasts(d_attr$fc) <- contr.sum(3)
  expect_error(fastglmm(y ~ x + fc + (1 | g), d_attr),
               "contrasts attribute")
  expect_no_error(suppressWarnings(fastglmm(y ~ x + (1 | fc), d_attr)))

  # A non-default options(contrasts = ) errors only when an unordered factor
  # is actually used as a fixed effect.
  old <- options(contrasts = c("contr.sum", "contr.poly"))
  on.exit(options(old), add = TRUE)
  expect_error(fastglmm(y ~ x + fc + (1 | g), d),
               "options\\(contrasts")
  expect_no_error(suppressWarnings(fastglmm(y ~ x + (1 | fc), d)))
  expect_no_error(fastglmm(y ~ x, d)) # no factor used at all
  options(old)
})

test_that("character and logical fixed-effect columns are checked the same way as an unordered factor", {
  set.seed(7)
  n <- 40L
  d <- data.frame(y = rnorm(n), x = rnorm(n),
                   ch = sample(letters[1:3], n, replace = TRUE),
                   lg = sample(c(TRUE, FALSE), n, replace = TRUE),
                   g = factor(rep(1:8, 5)))

  old <- options(contrasts = c("contr.sum", "contr.poly"))
  on.exit(options(old), add = TRUE)
  # Under the default contrasts these fit fine (treatment coding is what
  # the shared parser gives them either way).
  expect_error(fastglmm(y ~ x + ch + (1 | g), d), "options\\(contrasts")
  expect_error(fastglmm(y ~ x + lg + (1 | g), d), "options\\(contrasts")
  options(old)
  expect_no_error(suppressWarnings(fastglmm(y ~ x + ch + (1 | g), d)))
  expect_no_error(suppressWarnings(fastglmm(y ~ x + lg + (1 | g), d)))
})

test_that("a one-level fixed-effect factor errors instead of being dropped silently", {
  n <- 20L
  d <- data.frame(y = rnorm(n), x = rnorm(n), f = rep("z", n), b = rep(TRUE, n))

  # Character, main effect.
  expect_error(fastglmm(y ~ f + x, d),
               "contrasts can be applied only to factors with 2 or more levels")
  # Logical, main effect: factor(rep(TRUE, n)) also has one level.
  expect_error(fastglmm(y ~ b + x, d),
               "contrasts can be applied only to factors with 2 or more levels")
  # The column is named.
  expect_error(fastglmm(y ~ f + x, d), "'f'")
})

test_that("only the unordered slot of options(contrasts) is checked for an unordered factor", {
  set.seed(8)
  n <- 40L
  d <- data.frame(y = rnorm(n), x = rnorm(n),
                   fc = factor(sample(letters[1:3], n, replace = TRUE)),
                   g = factor(rep(1:8, 5)))

  # Changing only the ORDERED slot leaves the unordered slot at its default
  # ("contr.treatment"), so an unordered factor still fits without error.
  old <- options(contrasts = c("contr.treatment", "contr.sum"))
  on.exit(options(old), add = TRUE)
  expect_no_error(suppressWarnings(fastglmm(y ~ x + fc + (1 | g), d)))
  options(old)
})

test_that("inf in weights= or offset= is refused up front, not by the kernel", {
  d <- err_data()
  w <- rep(1, 40)
  w[1] <- Inf
  # The plain up-front message, not the kernel's own ("FitOptions.weights
  # must be finite and > 0") which "weights must be finite" alone would also
  # match as a substring.
  expect_error(fastglmm(y ~ x, d, weights = w), "no NaN, inf, or missing entries")
  o <- rep(0, 40)
  o[1] <- -Inf
  expect_error(fastglmm(y ~ x, d, offset = o), "no NaN, inf, or missing entries")
  # compare() drops a row with a missing weight or offset, so its text names
  # inf alone, as the Python port's does.
  expect_error(compare(d, y ~ 1, y ~ x, weights = w),
               "^weights must be finite or missing: no inf entries$")
  expect_error(compare(d, y ~ 1, y ~ x, offset = o),
               "^offset must be finite or missing: no inf entries$")
})

test_that("quasi-likelihood dispersion on binomial errors as not implemented", {
  d <- err_data()
  d$yb <- rbinom(40, 1, 0.5)
  expect_error(fastglmm(yb ~ x, d, family = binomial(), dispersion = 2),
               "quasi-likelihood.*not yet implemented")
})

test_that("bad dispersion value message matches the Python port", {
  d <- err_data()
  expect_error(
    fastglmm(y ~ x, d, family = Gamma(link = "log"), dispersion = "pearson"),
    "dispersion must be NULL, 'estimate', or a number, got 'pearson'",
    fixed = TRUE
  )
})

test_that("cloglog GLM matches glm on the same data", {
  set.seed(1)
  n <- 300
  x <- rnorm(n)
  mu <- 1 - exp(-exp(0.2 + 0.8 * x))
  d <- data.frame(y = rbinom(n, 1, mu), x = x)
  f <- fastglmm(y ~ x, data = d, family = binomial(link = "cloglog"))
  expect_true(f$converged)
  ref <- glm(y ~ x, data = d, family = binomial(link = "cloglog"))
  expect_equal(unname(fixef(f)), unname(coef(ref)), tolerance = 1e-5,
               info = "cloglog vs glm")
})

test_that("probit GLM fits and matches glm on the same data", {
  set.seed(1)
  n <- 300
  x <- rnorm(n)
  mu <- pnorm(0.2 + 0.8 * x)
  d <- data.frame(y = rbinom(n, 1, mu), x = x)
  f <- fastglmm(y ~ x, data = d, family = binomial(link = "probit"))
  expect_true(f$converged)
  ref <- glm(y ~ x, data = d, family = binomial(link = "probit"))
  expect_equal(unname(fixef(f)), unname(coef(ref)), tolerance = 1e-5,
               info = "probit vs glm")
})

test_that("inverse-Gaussian GLM refuses random effects", {
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
  # Caught by fastglmm()'s own client-side check, matching the Python port
  # (python/tests/test_validation.py's equivalent case) - a raw kernel panic
  # ("inverse-Gaussian mixed models are not implemented") never reaches here.
  expect_error(
    fastglmm(y ~ x + (1 | g), data = d, family = inverse.gaussian()),
    "GLM-only"
  )
})

test_that("inverse-Gaussian GLM matches glm (default link is already 1/mu^2)", {
  # inverse.gaussian()'s default link is "1/mu^2", so f and f2 fit the same
  # model under two spellings; fixef(f) and fixef(f2) are bit-identical.
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
  f2 <- fastglmm(y ~ x, data = d, family = inverse.gaussian(link = "1/mu^2"))
  # glm()'s own IRLS needs a start near the optimum on this data (y was
  # generated on the log-mean scale, not the canonical 1/mu^2 scale, so its
  # default start diverges); starting it at the fit's own beta is a fair
  # check of whether that beta solves the GLM score equations.
  ref <- glm(y ~ x, data = d, family = inverse.gaussian(),
             start = unname(fixef(f)))
  ref2 <- glm(y ~ x, data = d, family = inverse.gaussian(link = "1/mu^2"),
              start = unname(fixef(f2)))
  expect_true(ref$converged, info = "invgauss default link glm ref converged")
  expect_true(ref2$converged, info = "invgauss 1/mu^2 link glm ref converged")
  expect_equal(unname(fixef(f)), unname(coef(ref)), tolerance = 1e-5,
               info = "invgauss default link vs glm")
  expect_equal(unname(fixef(f2)), unname(coef(ref2)), tolerance = 1e-5,
               info = "invgauss 1/mu^2 link vs glm")
})

test_that("inverse-Gaussian accepts dispersion = \"estimate\"", {
  set.seed(3)
  d <- data.frame(y = rgamma(200, 4, 2) + 0.1, x = rnorm(200))
  f <- fastglmm(y ~ x, data = d, family = inverse.gaussian(),
                dispersion = "estimate")
  expect_true(f$converged)
})

test_that("init.theta seeds the negative-binomial shape search", {
  d <- err_data()
  u <- rnorm(8, sd = 0.8)
  d$yc <- rpois(40, exp(0.7 + u[as.integer(d$g)]))
  a <- fastglmm(yc ~ x + (1 | g), d, family = "negativebinomial",
                init.theta = 1.5)
  b <- fastglmm(yc ~ x + (1 | g), d, family = "negativebinomial",
                start = list(dispersion = 1.5))
  expect_identical(a$n_eval, b$n_eval)
  expect_identical(a$deviance, b$deviance)
  expect_true(fastglmm(yc ~ x, d, family = "negativebinomial",
                       init.theta = 1.5)$converged)
  expect_error(fastglmm(yc ~ x + (1 | g), d, family = "negativebinomial",
                        init.theta = 1.5, start = list(dispersion = 1.5)),
               "not both")
  expect_error(fastglmm(yc ~ x + (1 | g), d, family = "negativebinomial",
                        start = list(dispersion = -1)),
               "finite number > 0")
})

test_that("negative binomial GLM fits to convergence and reports theta as dispersion", {
  set.seed(4)
  n <- 600
  x <- rnorm(n)
  true_theta <- 4
  y <- rnbinom(n, size = true_theta, mu = exp(0.5 + 0.3 * x))
  d <- data.frame(y = y, x = x)
  fit <- fastglmm(y ~ x, d, family = "negativebinomial")
  expect_true(fit$converged)
  expect_equal(fit$family_name, "negativebinomial")
  expect_equal(unname(fixef(fit)), c(0.5, 0.3), tolerance = 0.05)
  # dispersion carries the estimated shape theta for this family (measured
  # 4.63 against a true 4 at this seed).
  expect_equal(fit$dispersion, true_theta, tolerance = 0.2)
  expect_equal(fit$df, 3L) # 2 fixed effects + the estimated theta
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
  expect_error(fastglmm(y ~ x, d, nAGQ = 2), "got 2")
})

test_that("unknown family and unsupported link messages match the Python port", {
  d <- err_data()
  expect_error(
    fastglmm(y ~ x, d, family = "logistic"),
    "unknown family 'logistic'; expected one of gaussian, binomial, poisson, gamma, negativebinomial, inversegaussian",
    fixed = TRUE
  )
  expect_error(
    fastglmm(y ~ x, d, family = binomial(link = "identity")),
    "family 'binomial' does not support link 'identity'; expected one of logit, probit, cloglog",
    fixed = TRUE
  )
})

test_that("start must be a list - message matches the Python port's warm_start check", {
  d <- err_data()
  expect_error(
    fastglmm(y ~ x, d, start = c(0, 0)),
    paste(
      "start must be a list with elements 'beta', 'theta' and/or 'dispersion' (theta",
      "is the random-effect Cholesky vector, not the negative-binomial shape - that is",
      "init.theta or dispersion), got numeric"
    ),
    fixed = TRUE
  )
})

test_that("a ragged list is refused, not recycled", {
  # glm() and lmer() refuse it in model.frame(); as.data.frame() would recycle
  # the length-3 column to 6 rows.
  expect_error(
    fastglmm(y ~ x, list(y = c(1.1, 2.3, 2.9, 4.2, 5.1, 5.8), x = c(0, 1, 2))),
    'columns have mismatched lengths ("x": 3, "y": 6)', fixed = TRUE
  )
})

test_that("a wrong-width start names the argument, not a kernel struct", {
  d <- err_data()
  expect_error(fastglmm(y ~ x + (1 | g), d, start = list(beta = 0)),
               "warm-start beta has 1 entries but the model has 2 fixed-effect", fixed = TRUE)
  expect_error(fastglmm(y ~ x + (1 | g), d, start = list(theta = c(1, 0))),
               "warm-start theta has 2 entries but the model has 1 random-effect", fixed = TRUE)
})

test_that("an infinite offset() term names the term, not a kernel struct", {
  d <- err_data()
  d$o <- 0
  d$o[2] <- Inf
  expect_error(fastglmm(y ~ x + offset(o), d), "term 'offset(o)' is not finite at row 1",
               fixed = TRUE)
})

test_that("wald.se must be 'hessian' or 'rx' - mirrors the Python port's wald_se check", {
  d <- err_data()
  expect_error(fastglmm(y ~ x, d, wald.se = "observed"),
               "wald.se must be 'hessian' or 'rx', got 'observed'")
})

test_that("nAGQ > 1 on negative-binomial and Gamma fits quadrature", {
  # NB's theta sits outside the AGQ integral, and Gamma's phi enters it only as
  # a weight on the deviance (the rest of its log-density sits outside), so both
  # families take nAGQ > 1 like binomial/Poisson: no warning, and the fit
  # reports the node count it ran (mirrors the Python port).
  d <- err_data()
  set.seed(4)
  d$cnt <- rpois(nrow(d), exp(0.5 + 0.3 * d$x))
  d$pos <- rgamma(nrow(d), shape = 2, rate = 2 / exp(0.5 + 0.1 * d$x))
  for (spec in list(list(y = "cnt", fam = "negativebinomial"),
                    list(y = "pos", fam = "gamma"))) {
    f <- stats::as.formula(paste(spec$y, "~ x + (1 | g)"))
    w <- capture_warnings(fit <- fastglmm(f, d, family = spec$fam, nAGQ = 7))
    expect_false(any(grepl("nAGQ", w)), info = spec$fam)
    expect_equal(fit$nAGQ, 7L, info = spec$fam)
  }
})

test_that("unimplemented accessors error with the reason", {
  d <- err_data()
  fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), d)) # boundary fit is fine here
  expect_error(predict(fit), "not available")
  expect_error(coef(fit), "fixef")
  expect_error(terms(fit), "formula\\(\\) returns")
  expect_error(confint(fit, method = "profile"), "no profiling machinery")
})
