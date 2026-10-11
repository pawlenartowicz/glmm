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
  # The matrix itself is the covariance its attributes describe.
  sd <- attr(v, "stddev")
  corr <- attr(v, "correlation")
  expect_equal(unname(sd), unname(sqrt(diag(v))), tolerance = 1e-12)
  expect_equal(corr[2, 1], v[2, 1] / (sd[[1]] * sd[[2]]), tolerance = 1e-12)
  # The vech -> SD/correlation math, against hand-computed values. A q = 2
  # block is symmetric in (row, column), so q = 3 is the case that can see a
  # transposed index.
  sc <- fastglmm:::.stddev_corr(c(4, 1, 1.25))
  expect_equal(sc$stddev, c(2, sqrt(1.25)), tolerance = 1e-12)
  expect_equal(sc$correlation[2, 1], 1 / (2 * sqrt(1.25)), tolerance = 1e-12)
  expect_equal(sc$correlation[1, 2], sc$correlation[2, 1])
  # vech is column-major lower-triangular: (v00, v10, v20, v11, v21, v22),
  # here sd = (1, 2, 3) with correlations r10 = 0.5, r20 = -0.25, r21 = 0.1.
  sc3 <- fastglmm:::.stddev_corr(c(1, 1, -0.75, 4, 0.6, 9))
  expect_equal(sc3$stddev, c(1, 2, 3), tolerance = 1e-12)
  expect_equal(sc3$correlation,
               rbind(c(1, 0.5, -0.25), c(0.5, 1, 0.1), c(-0.25, 0.1, 1)),
               tolerance = 1e-12)
  expect_error(fastglmm:::.stddev_corr(c(1, 2)), "is not a valid vech")
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
  expect_equal(unname(fixef(fit_str)), unname(coef(ref)), tolerance = 1e-5,
               info = "gamma log-link vs glm")
  # sigma() for a Gamma GLM is sqrt(phi), phi the Pearson dispersion
  # summary(glm) reports (Sum(w r^2) / (n - p)).
  expect_equal(sigma(fit_str)^2, summary(ref)$dispersion, tolerance = 1e-5)
  # The other half of the trap: fit_obj used the inverse link, so its
  # coefficients must match glm() on the inverse link, not the log one.
  ref_inv <- glm(y ~ x, data = d, family = Gamma(link = "inverse"))
  expect_equal(unname(fixef(fit_obj)), unname(coef(ref_inv)), tolerance = 1e-5,
               info = "gamma inverse-link vs glm")
})

test_that("sigma for inverse-Gaussian is sqrt(phi), phi the Pearson dispersion", {
  set.seed(2)
  n <- 400
  x <- rnorm(n)
  mu <- exp(0.3 + 0.2 * x)
  lam <- 3
  v <- rnorm(n)^2
  x1 <- mu + mu^2 * v / (2 * lam) -
    (mu / (2 * lam)) * sqrt(4 * mu * lam * v + mu^2 * v^2)
  y <- ifelse(runif(n) <= mu / (mu + x1), x1, mu^2 / x1)
  d <- data.frame(y = y, x = x)
  f <- fastglmm(y ~ x, data = d, family = inverse.gaussian())
  expect_equal(sigma(f), sqrt(f$dispersion))
})

test_that("summary prints 'Dispersion (phi, Pearson)' for a Gamma GLM", {
  set.seed(24)
  d <- data.frame(x = rnorm(80))
  d$y <- rgamma(80, shape = 4, rate = 4 / exp(0.3 + 0.5 * d$x))
  fit <- fastglmm(y ~ x, d, family = "gamma")
  out <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_true(grepl("Dispersion (phi, Pearson):", out, fixed = TRUE))
})

test_that("summary prints 'Dispersion (phi, ML)' for a Gamma GLMM", {
  set.seed(25)
  n_g <- 60L
  m <- 10L
  g <- factor(rep(seq_len(n_g), each = m))
  x <- rnorm(n_g * m)
  u0 <- rnorm(n_g, sd = 0.3)
  eta <- 0.5 + 0.2 * x + u0[as.integer(g)]
  y <- rgamma(n_g * m, shape = 4, rate = 4 / exp(eta))
  d <- data.frame(y = y, x = x, g = g)
  fit <- fastglmm(y ~ x + (1 | g), d, family = Gamma(link = "log"))
  out <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_true(grepl("Dispersion (phi, ML):", out, fixed = TRUE))
})

test_that("summary prints 'Dispersion (phi, fixed)' when dispersion is held", {
  set.seed(23)
  d <- data.frame(x = rnorm(80))
  d$y <- rgamma(80, shape = 4, rate = 4 / exp(0.3 + 0.5 * d$x))
  fit <- fastglmm(y ~ x, d, family = "gamma", dispersion = 2.5)
  expect_equal(fit$dispersion, 2.5)
  out <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_true(grepl("Dispersion (phi, fixed): 2.5", out, fixed = TRUE))
})

test_that("sigma is fixed at 1 for binomial/poisson (lme4 agreement)", {
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

test_that("gaussian LMM sigma is pinned and VarCorr prints a Residual row", {
  set.seed(106)
  g <- factor(rep(1:40, each = 8))
  x <- rnorm(320)
  d <- data.frame(y = 1 + 0.5 * x + rnorm(40, sd = 0.9)[as.integer(g)] +
                    rnorm(320, sd = 0.6),
                  x = x, g = g)
  fit <- fastglmm(y ~ x + (1 | g), d)
  # The REML residual standard deviation, pinned against this package's own
  # output. The agreement with lmer on this design is checked by hand in
  # tools/acceptance-vs-lme4.R's family of comparisons, not here: this suite
  # installs no external engine.
  expect_equal(sigma(fit), 0.583493794941051, tolerance = CI_REF_REL)
  vc <- VarCorr(fit)
  expect_equal(attr(vc, "sc"), sigma(fit))
  expect_match(paste(capture.output(print(vc)), collapse = "\n"), "Residual")
})

test_that("VarCorr omits the Residual row on a non-gaussian mixed fit", {
  d <- benchmark_data(seed = 116, family = "binomial")
  fit <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  vc <- VarCorr(fit)
  # phi==1 families carry no free residual scale (see VarCorr.fastglmm), so
  # the sc attribute is NA and print.VarCorr.fastglmm's is.finite(sc) guard
  # must take its false branch.
  expect_true(is.na(attr(vc, "sc")))
  expect_false(grepl("Residual", paste(capture.output(print(vc)), collapse = "\n")))
})

test_that("accessors: nobs, formula string, family, model.frame", {
  d <- benchmark_data(seed = 105, family = "binomial")
  fit <- fastglmm(y ~ t + d + (1 | g), d, family = binomial())
  expect_equal(nobs(fit), nrow(d))
  expect_equal(formula(fit), "y ~ t + d + (1 | g)")
  expect_s3_class(family(fit), "family")
  expect_equal(family(fit)$family, "binomial")
  mf <- model.frame(fit)
  expect_setequal(names(mf), c("y", "t", "d", "g"))
  # isSingular() dispatch is covered separately - it is lme4's own generic
  # (see the fixef/ranef/VarCorr/isSingular masking test below), so it needs
  # library(lme4), not just library(fastglmm).
})

test_that("isSingular(fastglmm fit) works with no other package attached", {
  # The point of fastglmm declaring its own isSingular generic (rather than
  # only registering on lme4's, as fixef/ranef/VarCorr do on nlme's): this
  # must keep working with nothing but fastglmm loaded, lme4 installed or
  # not. Runs in its own Rscript subprocess (see the load-order test below
  # for why), and does not need lme4 installed at all - the case this
  # checks never reaches isSingular.default.
  skip_on_os("windows") # Rscript path/quoting below targets a POSIX shell
  `%||%` <- function(x, y) if (is.null(x)) y else x

  script <- paste(
    "library(fastglmm);",
    "d <- data.frame(y = rnorm(60), x = rnorm(60), g = factor(rep(1:10, 6)));",
    "fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), d));",
    "cat(is.logical(isSingular(fit)))"
  )
  out <- system2("Rscript", c("-e", shQuote(script)), stdout = TRUE, stderr = TRUE)
  expect_equal(attr(out, "status") %||% 0L, 0L, info = paste(out, collapse = "\n"))
  expect_equal(out[length(out)], "TRUE")
})

test_that("isSingular() on a class it cannot handle errors instead of recursing", {
  # isSingular.default forwards only for a merMod object (and only when
  # lme4 is already loaded); anything else, e.g. isSingular(1), must reach
  # the plain error naming the class, not lme4's own S4 default (which
  # calls UseMethod("isSingular") again and would recurse without the
  # merMod check). No lme4 needed for this: the case it forwards to lme4
  # for is checked with a real merMod fit in tools/generic-load-order.R.
  expect_error(isSingular(1), "does not know how to handle.*numeric")
  expect_error(isSingular("x"), "does not know how to handle.*character")
})

# The load-order dispatch checks against a real merMod object (both orders,
# fixef/ranef/VarCorr/isSingular) need lme4 actually installed, which CI
# deliberately does not have - they live in tools/generic-load-order.R,
# run by hand, next to acceptance-vs-lme4.R. isSingular's own no-lme4-at-all
# case is the test above, which does not need lme4 and so stays here.

test_that("summary prints lme4's blocks in lme4's order", {
  d <- benchmark_data(seed = 106, family = "binomial")
  fit <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  out <- capture.output(print(fit))
  expect_true(any(grepl("Laplace", out)))
  expect_true(any(grepl("groups:  g, 60", out, fixed = TRUE)))
  sout <- paste(capture.output(print(summary(fit))), collapse = "\n")
  order <- c(
    "Generalized linear mixed model", "Formula: y ~ t + (1 | g)", "   Data: d",
    "AIC", "BIC", "logLik", "deviance", "df.resid",
    "Scaled residuals:", "Min", "1Q", "Median", "3Q", "Max",
    "Random effects:", "Variance", "Std.Dev.",
    "Number of obs: 600, groups:  g, 60",
    "Fixed effects:", "Pr(>|z|)",
    "Correlation of Fixed Effects:",
    "Optimizer evaluations:"
  )
  pos <- vapply(order, function(s) regexpr(s, sout, fixed = TRUE)[1], 0)
  expect_true(all(pos > 0), info = paste(names(pos)[pos <= 0], collapse = ", "))
  expect_equal(pos, sort(pos))
  # No REML line on an ML fit, and no test footer: z is a GLMM's test.
  expect_false(grepl("REML criterion", sout, fixed = TRUE))
  expect_false(grepl("Satterthwaite", sout, fixed = TRUE))
  expect_false(grepl("Wald z", sout, fixed = TRUE))
})

test_that("a declared, unused grouping level is not counted as a group", {
  # lme4 drops a grouping level with no rows before counting groups (its
  # `ngrps`), so the printed count must match the number of levels that
  # actually appear in the data, not the number of declared factor levels.
  set.seed(118)
  g <- factor(sample(1:5, 40, replace = TRUE), levels = 1:6) # level 6 unused
  d <- data.frame(y = rnorm(40), x = rnorm(40), g = g)
  fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), d))
  expect_equal(nlevels(g), 6L)
  expect_equal(length(unique(g)), 5L)
  out <- capture.output(print(fit))
  expect_true(any(grepl("groups:  g, 5", out, fixed = TRUE)))
  sout <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_true(grepl("groups:  g, 5", sout, fixed = TRUE))
})

test_that("summary on a gaussian LMM prints the REML criterion, not AIC", {
  d <- benchmark_data(seed = 107, family = "poisson")
  d$yg <- d$y + rnorm(nrow(d))
  fit <- fastglmm(yg ~ t + (1 | g), d)
  s <- summary(fit)
  expect_named(s$criterion, "REML criterion at convergence")
  expect_equal(unname(s$criterion), -2 * as.numeric(logLik(fit)))
  sout <- paste(capture.output(print(s)), collapse = "\n")
  expect_true(grepl("REML criterion at convergence:", sout, fixed = TRUE))
  expect_false(grepl("AIC", sout, fixed = TRUE))
  expect_true(grepl("Residual", sout, fixed = TRUE))
  # Scaled residuals are (y - mu) / sigma, quantiles type 7.
  r <- (fit$y - fitted(fit)) / sigma(fit)
  expect_equal(unname(s$scaled_residuals), unname(quantile(r, c(0, .25, .5, .75, 1))))
})

test_that("print and summary on a non-mixed fit skip the random-effects block", {
  set.seed(115)
  d <- data.frame(x = rnorm(50))
  d$y <- 1 + 2 * d$x + rnorm(50, sd = 0.5)
  fit <- fastglmm(y ~ x, d)
  out <- capture.output(print(fit))
  expect_true(any(grepl("Number of obs: 50", out, fixed = TRUE)))
  expect_false(any(grepl("Random effects:", out, fixed = TRUE)))
  expect_false(any(grepl("groups:", out, fixed = TRUE)))

  sout <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_true(grepl("Number of obs: 50", sout, fixed = TRUE))
  expect_false(grepl("Random effects:", sout, fixed = TRUE))
  expect_false(grepl("groups:", sout, fixed = TRUE))
})

test_that("summary carries the correlation of fixed effects off vcov", {
  d <- benchmark_data(seed = 109, family = "binomial")
  fit <- fastglmm(y ~ t + d + (1 | g), d, family = binomial())
  s <- summary(fit)
  expect_equal(s$corr_fixed, cov2cor(vcov(fit)))
})

test_that("VarCorr correlations print at 2 dp", {
  d <- benchmark_data(seed = 101, family = "binomial",
                      tau0 = 0.8, tau1 = 0.5, rho = 0.3)
  fit <- fastglmm(y ~ t + d + t:d + (1 + t | g), d, family = binomial())
  out <- capture.output(print(VarCorr(fit)))
  # The Corr cell is the last token of the slope row; Std.Dev. keeps its
  # own `digits`, so only that cell is held to two decimals.
  slope <- grep("^\\s+t\\s", out, value = TRUE)
  expect_length(slope, 1L)
  corr <- sub(".*\\s", "", trimws(slope))
  expect_match(corr, "^-?[0-9]\\.[0-9]{2}$")
})

test_that("logLik carries df/nobs/REML and feeds AIC/BIC", {
  d <- benchmark_data(seed = 108, family = "binomial")
  fit <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  ll <- logLik(fit)
  expect_s3_class(ll, "logLik")
  expect_true(is.finite(as.numeric(ll)))
  # p fixed effects + one RE variance; binomial estimates no dispersion.
  expect_equal(attr(ll, "df"), 3L)
  expect_equal(attr(ll, "nobs"), nrow(d))
  expect_false(attr(ll, "REML"))
  expect_equal(AIC(fit), -2 * as.numeric(ll) + 2 * attr(ll, "df"))
  expect_equal(BIC(fit), -2 * as.numeric(ll) + log(nrow(d)) * attr(ll, "df"))
})

test_that("ranef and fitted return lme4-shaped values on a Gaussian LMM", {
  # The R side is a pure renderer — the kernel owns the block layout, so what
  # this checks is the reshape and the names. The layout itself is tested
  # crate-side. Gaussian because the LMM paths are what gained conditional
  # modes; benchmark_data only makes binomial/poisson.
  set.seed(2026)
  n_g <- 40L
  m <- 8L
  g <- factor(rep(seq_len(n_g), each = m))
  t <- rnorm(n_g * m)
  u0 <- rnorm(n_g, sd = 1)
  u1 <- rnorm(n_g, sd = 0.7)
  y <- 1 + 2 * t + u0[as.integer(g)] + u1[as.integer(g)] * t +
    rnorm(n_g * m, sd = 0.5)
  d <- data.frame(y = y, t = t, g = g)
  fit <- fastglmm(y ~ t + (1 + t | g), d)
  expect_true(fit$converged)

  re <- ranef(fit)
  expect_named(re, "g")
  expect_s3_class(re$g, "data.frame")
  expect_equal(dim(re$g), c(n_g, 2L))
  expect_equal(colnames(re$g), c("(Intercept)", "t"))
  expect_setequal(rownames(re$g), levels(g))
  expect_true(any(re$g[["t"]] != 0))
  # The data.frame is a reshape of the kernel's flat block, not a second
  # computation — and a transposed reshape would still have the right dim().
  expect_equal(as.numeric(t(as.matrix(re$g))),
               fit$ranef_blocks[[1]]$values, tolerance = 1e-12)

  fv <- fitted(fit)
  expect_length(fv, nrow(d))
  expect_equal(names(fv), rownames(d))
  # Xbeta + Zb: the residual variance is 0.5^2, so fitted must track y far
  # better than the fixed part alone does.
  fixed_only <- fit$beta[[1]] + fit$beta[[2]] * t
  expect_lt(sum((y - fv)^2), sum((y - fixed_only)^2))
})

test_that("diagnostics is additive: the top-level names keep working", {
  d <- benchmark_data(seed = 111, family = "binomial")
  fit <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  expect_named(fit$diagnostics,
               c("converged", "singular", "aliased", "boundary", "pinned",
                 "notes"))
  # One report, two reading paths - the top-level names are unchanged.
  expect_identical(fit$converged, fit$diagnostics$converged)
  expect_identical(fit$singular, fit$diagnostics$singular)
  expect_identical(fit$aliased, fit$diagnostics$aliased)
  expect_true(fit$converged)
  # This fixture has a real group effect, so nothing pins - assert that, not
  # the set of every value the mapper can emit.
  expect_equal(fit$diagnostics$boundary, "interior")
  expect_length(fit$diagnostics$pinned, 0L)
  expect_length(fit$diagnostics$notes, 0L)
})

test_that("a q >= 2 pin is named even though its stddev is not zero", {
  # Mirrors the Python port's test of the same name: on a grouping with
  # q >= 2 the pin fixes the Cholesky DIAGONAL while the reported stddev is
  # sqrt(offdiag^2 + diag^2), so it lands at ~1e-6, not at 0. The resulting
  # 2x2 covariance block is then rank 1 (one Cholesky diagonal pinned to
  # zero), so the correlation is exactly +/-1 by construction - not a
  # rounding artifact. lme4 reports the same exact 1.000 on this design.
  # Reading `pinned` names the component.
  #
  # Design mirrors the Rust lmm::tests::zero_slope_variance_pins_slope_component:
  # 16 clusters x 16 rows, a real fixed slope but zero cluster-varying slope,
  # residual a +/-0.8 period-4 quadrature block against x's +/-1 alternation so
  # every cluster has sum(resid) = 0 and sum(x*resid) = 0 exactly. The REML
  # slope-variance MLE is then 0 while the planted cluster intercepts keep the
  # intercept component interior.
  nc <- 16L
  per <- 16L
  set.seed(5)
  u0 <- 0.6 * rnorm(nc)
  rows <- expand.grid(k = 0:(per - 1L), c = 0:(nc - 1L))
  x <- ifelse(rows$k %% 2L == 0L, 1, -1)
  e <- ifelse((rows$k %/% 2L) %% 2L == 0L, 0.8, -0.8)
  d <- data.frame(y = 0.5 + 0.4 * x + u0[rows$c + 1L] + e, x = x,
                  g = factor(paste0("g", rows$c)))

  fit <- expect_pinned_boundary_fit(
    function() fastglmm(y ~ x + (1 + x | g), d),
    "x in g"
  )
  expect_equal(fit$diagnostics$boundary, "at_boundary")
  # The SLOPE component is the pinned one, aligned with the varcorr block.
  expect_equal(fit$diagnostics$pinned, list(c(FALSE, TRUE)))

  vc <- VarCorr(fit)$g
  sd <- attr(vc, "stddev")
  expect_length(fit$diagnostics$pinned[[1L]], length(sd))
  # The pinned slot is negligible against its sibling but is NOT exactly 0.
  expect_false(sd[[2L]] == 0)
  expect_lt(sd[[2L]] / sd[[1L]], 1e-6)
})

test_that("a pin is named on a sparse-route fit too", {
  # A slope-carrying extra grouping routes the fit to the sparse solver, which
  # assembles its Fit outside the dense mappers. `g` carries no signal, so its
  # single component pins; the residual is centred within each `g` level so
  # not even sampling noise leaves it spurious between-level variance. Mirrors
  # the Python port's test_pinned_detail_is_reported_on_a_sparse_route_fit.
  n <- 240L
  set.seed(7)
  g <- (seq_len(n) - 1L) %% 12L
  h <- (seq_len(n) - 1L) %/% 12L
  x <- runif(n, -1, 1)
  noise <- 0.1 * runif(n, -0.5, 0.5)
  for (lvl in 0:11) {
    sel <- g == lvl
    noise[sel] <- noise[sel] - mean(noise[sel])
  }
  d <- data.frame(
    y = 1 + 0.75 * x + sin(h * 0.37) + cos(h * 0.91) * x + noise,
    x = x, g = factor(paste0("g", g)), h = factor(paste0("h", h))
  )

  fit <- expect_pinned_boundary_fit(
    function() fastglmm(y ~ x + (1 | g) + (1 + x | h), d),
    "\\(Intercept\\) in g"
  )
  expect_equal(fit$diagnostics$pinned, list(TRUE, c(FALSE, FALSE)))
  # One flag per stddev, per block - the alignment .pinned_detail walks.
  for (nm in names(VarCorr(fit))) {
    sd <- attr(VarCorr(fit)[[nm]], "stddev")
    expect_length(fit$diagnostics$pinned[[match(nm, names(VarCorr(fit)))]],
                  length(sd))
  }
})

test_that("an ill-conditioned design warns with its own condition class", {
  # x = [1, a, a+delta] with delta living entirely on rows weighted 1e-11:
  # full-rank raw, so nothing is dropped, and near-singular once weighted - the
  # case only the fit can see (mirrors the Rust
  # diagnostics_ill_conditioned_note_through_fit_cold).
  n <- 60L
  split <- 40L
  a <- ((seq_len(n) - 1L) * 13L) %% 17L - 8
  b <- a + ifelse(seq_len(n) <= split, 0, 1)
  d <- data.frame(
    y = 0.5 + 1.3 * a + 0.477 * b + ((seq_len(n) - 1L) %% 3L - 1),
    a = a, b = b,
    wt = ifelse(seq_len(n) <= split, 1, 1e-11)
  )

  cond <- NULL
  fit <- withCallingHandlers(
    fastglmm(y ~ a + b, d, weights = wt),
    fastglmm_diagnostic = function(cnd) {
      cond <<- cnd
      invokeRestart("muffleWarning")
    }
  )
  # Caught by CLASS, not by message text - that is the point of the class.
  expect_s3_class(cond, "fastglmm_ill_conditioned")
  expect_s3_class(cond, "fastglmm_diagnostic")
  expect_match(conditionMessage(cond), "b is almost a combination of other columns")

  expect_identical(fit$warnings$tier, "caution")
  expect_identical(fit$warnings$kind, "ill_conditioned")
  expect_identical(fit$warnings$title, "Nearly collinear columns")

  expect_true(fit$converged)
  expect_false(any(fit$aliased)) # flagged, not dropped
  expect_length(fit$diagnostics$notes, 1L)
  note <- fit$diagnostics$notes[[1L]]
  expect_equal(note$kind, "ill_conditioned")
  expect_equal(note$columns, 3L) # 1-based into names(): "b"
  expect_equal(names(fit$beta)[note$columns], "b")
  expect_lt(note$pivot, 1e-9)

  # Negative case: the same design at unit weights raises no note at all.
  clean <- fastglmm(y ~ a + b, d)
  expect_length(clean$diagnostics$notes, 0L)
})

test_that("rank-deficient designs mirror lme4's NA coefficients", {
  d <- benchmark_data(seed = 108, family = "binomial")
  d$t2 <- d$t # aliased copy
  fit <- fastglmm(y ~ t + t2 + (1 | g), d, family = binomial())
  expect_true(fit$aliased[["t2"]])
  expect_true(is.na(fixef(fit)[["t2"]]))
  expect_true(fit$converged)
})

test_that("the fit keeps the lowered response and the data name", {
  d <- benchmark_data(seed = 110, family = "binomial")
  fit <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  expect_equal(unname(fit$y), as.double(d$y))
  expect_equal(fit$data_name, "d")
})

test_that("residuals: response and pearson, nothing else", {
  d <- benchmark_data(seed = 111, family = "poisson")
  fit <- fastglmm(y ~ t + (1 | g), d, family = poisson())
  r <- residuals(fit)
  expect_equal(unname(r), unname(fit$y - fitted(fit)))
  rp <- residuals(fit, type = "pearson")
  expect_equal(unname(rp), unname((fit$y - fitted(fit)) / sqrt(fitted(fit))))
  expect_error(residuals(fit, type = "deviance"), "response.*pearson")
})

test_that("pearson residuals stay finite at extreme means", {
  # V = mu^2 (Gamma) and mu^3 (inverse Gaussian) leave the double range here;
  # sqrt(V) does not. Expected: (y - mu) / sqrt(phi * V).
  fit <- structure(list(family_name = "gamma", dispersion = 0.5, weights = NULL,
                        y = c(1.5e200, 1e-200), fitted = c(1e200, 2e-200),
                        frame = data.frame(i = 1:2)),
                   class = "fastglmm")
  expect_equal(unname(residuals(fit, type = "pearson")), c(sqrt(0.5), -sqrt(0.5)))
  fit$family_name <- "inversegaussian"
  fit$dispersion <- 0.25
  fit$y <- c(2e150, 2e-150)
  fit$fitted <- c(1e150, 1e-150)
  expect_equal(unname(residuals(fit, type = "pearson")), c(2e-75, 2e75))
})

test_that("tidy() and glance() dispatch through generics/broom", {
  d <- benchmark_data(seed = 112, family = "binomial")
  fit <- fastglmm(y ~ t + d + (1 | g), d, family = binomial())
  td <- tidy(fit)
  expect_s3_class(td, "data.frame")
  expect_named(td, c("term", "estimate", "std.error", "statistic", "p.value"))
  expect_equal(td$term, names(fixef(fit)))
  expect_equal(td$estimate, unname(fixef(fit)))
  expect_equal(td$statistic, unname(fixef(fit) / fit$se))
  expect_equal(td$p.value, unname(summary(fit)$coefficients[, "Pr(>|z|)"]))
  gl <- glance(fit)
  expect_equal(nrow(gl), 1L)
  expect_named(gl, c("nobs", "logLik", "AIC", "BIC", "deviance", "df.residual", "REML"))
  expect_equal(gl$AIC, AIC(fit))
  expect_equal(gl$BIC, BIC(fit))
  expect_equal(gl$df.residual, nobs(fit) - fit$df)
  expect_false(gl$REML)
  # The whole point of registering on generics: downstream packages call
  # broom::tidy(), which is generics::tidy re-exported.
  skip_if_not_installed("broom")
  expect_equal(broom::tidy(fit), td)
  expect_equal(broom::glance(fit), gl)
})

test_that("glance() returns AIC/BIC on a REML fit and says so", {
  d <- benchmark_data(seed = 113, family = "poisson")
  d$yg <- d$y + rnorm(nrow(d))
  fit <- fastglmm(yg ~ t + (1 | g), d)
  gl <- glance(fit)
  expect_true(gl$REML)
  expect_equal(gl$AIC, -2 * fit$loglik + 2 * fit$df)
  expect_equal(gl$deviance, -2 * fit$loglik)
})

test_that("warm start (lme4's start=) reaches the kernel", {
  # Started at the cold fit's own optimum, the search only has to confirm it:
  # 7 evaluations against the cold search's 11 or 12 (measured 2026-10-07).
  # The margin is what makes this a test of the start arriving. A start merely
  # near the optimum (theta = 0.5 here) takes within one evaluation of the cold
  # search, and that one evaluation flips with round-off: a 1-ulp change in
  # the t column or a reassociated PIRLS sum moves either count by one.
  d <- benchmark_data(seed = 109, family = "binomial")
  cold <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  theta_hat <- sqrt(as.numeric(VarCorr(cold)[[1]]))
  warm <- fastglmm(y ~ t + (1 | g), d, family = binomial(),
                   start = list(beta = unname(fixef(cold)), theta = theta_hat))
  expect_equal(unname(fixef(warm)), unname(fixef(cold)), tolerance = 1e-4)
  expect_lt(warm$n_eval, cold$n_eval - 2L)
})

test_that("the gradient search fields cross the port", {
  # A random-slope LMM searches theta with BFGS; OLS runs no theta search.
  d <- benchmark_data(seed = 115, family = "poisson")
  u0 <- rnorm(nlevels(d$g))
  u1 <- rnorm(nlevels(d$g), sd = 2)
  d$yg <- 1 + d$t + u0[as.integer(d$g)] + u1[as.integer(d$g)] * d$t +
    rnorm(nrow(d), sd = 0.5)
  routed <- fastglmm(yg ~ t + (t | g), d)
  expect_gt(routed$n_grad, 0)
  expect_true(routed$outer_stop %in% c("fdelta", "pgrad", "noise_floor",
                                       "line_search", "max_iter",
                                       "grad_unavailable", "non_finite"))
  expect_true(is.finite(routed$outer_decrement))
  plain <- fastglmm(yg ~ t, d)
  expect_identical(plain$n_grad, 0)
  expect_identical(plain$outer_stop, NA_character_)
  expect_true(is.nan(plain$outer_decrement))
})

test_that("wald.se = \"rx\" is marshalled through to the kernel", {
  # Same beta either way (both are Wald SE methods on the same fit); the SEs
  # themselves must differ, or the option never reached the kernel.
  d <- benchmark_data(seed = 114, family = "binomial")
  fit_hessian <- fastglmm(y ~ t + (1 | g), d, family = binomial())
  fit_rx <- fastglmm(y ~ t + (1 | g), d, family = binomial(), wald.se = "rx")
  expect_true(fit_rx$converged)
  expect_true(all(is.finite(fit_rx$se)))
  expect_equal(unname(fixef(fit_rx)), unname(fixef(fit_hessian)), tolerance = 1e-8)
  expect_false(isTRUE(all.equal(unname(fit_rx$se), unname(fit_hessian$se))))
})

# A Gaussian random-intercept fixture: 40 groups of 8.
lmm_data <- function(seed = 131) {
  set.seed(seed)
  g <- factor(rep(1:40, each = 8))
  x <- rnorm(320)
  data.frame(y = 1 + 0.5 * x + rnorm(40, sd = 0.9)[as.integer(g)] + rnorm(320, sd = 0.6),
             x = x, z = rnorm(320), g = g)
}

test_that("a Gaussian LMM summary has lmerTest's columns with the crate's df and p", {
  fit <- fastglmm(y ~ x + z + (1 | g), lmm_data())
  expect_identical(fit$df_method, "satterthwaite")
  s <- summary(fit)
  expect_identical(colnames(s$coefficients),
                   c("Estimate", "Std. Error", "df", "t value", "Pr(>|t|)"))
  expect_equal(s$coefficients[, "df"], fit$coef_df)
  expect_equal(s$coefficients[, "Pr(>|t|)"], fit$coef_p)
  expect_equal(s$coefficients[, "t value"], fixef(fit) / fit$se)
  expect_true(all(is.finite(fit$coef_df) & fit$coef_df > 0))
  # The crate's p is the t tail on the row's df.
  expect_equal(unname(fit$coef_p),
               unname(2 * pt(-abs(fixef(fit) / fit$se), fit$coef_df)), tolerance = 1e-10)
  out <- capture.output(print(s))
  expect_true(any(grepl("^ +Estimate Std. Error +df t value Pr\\(>\\|t\\|\\)", out)))
  expect_identical(tail(out, 1L),
                   "t tests use Satterthwaite degrees of freedom (finite-difference θ-Hessian).")
})

test_that("tidy() has a df column exactly where summary() has one", {
  lmm <- fastglmm(y ~ x + (1 | g), lmm_data())
  td <- tidy(lmm)
  expect_named(td, c("term", "estimate", "std.error", "statistic", "df", "p.value"))
  cf <- summary(lmm)$coefficients
  expect_equal(td$df, unname(cf[, "df"]))
  expect_equal(td$statistic, unname(cf[, "t value"]))
  expect_equal(td$p.value, unname(cf[, "Pr(>|t|)"]))

  ols <- fastglmm(y ~ x, lmm_data())
  td <- tidy(ols)
  expect_named(td, c("term", "estimate", "std.error", "statistic", "p.value"))
  cf <- summary(ols)$coefficients
  expect_equal(td$statistic, unname(cf[, "t value"]))
  expect_equal(td$p.value, unname(cf[, "Pr(>|t|)"]))
})

test_that("an OLS summary is summary.lm's: t on n - p, sigma and R-squared footer", {
  d <- lmm_data()
  fit <- fastglmm(y ~ x + z, d)
  ref <- lm(y ~ x + z, d)
  expect_identical(fit$df_method, "residual")
  cf <- summary(fit)$coefficients
  expect_identical(colnames(cf), c("Estimate", "Std. Error", "t value", "Pr(>|t|)"))
  expect_equal(unname(cf), unname(coef(summary(ref))), tolerance = 1e-8)
  expect_equal(unname(fit$coef_df), rep(317, 3))
  out <- trimws(capture.output(print(summary(fit))))
  lm_out <- trimws(capture.output(print(summary(ref))))
  for (start in c("Residual standard error:", "Multiple R-squared:")) {
    expect_identical(out[startsWith(out, start)], lm_out[startsWith(lm_out, start)],
                     info = start)
  }
  expect_false(any(grepl("Satterthwaite|Wald z|df not computed", out)))
})

test_that("R-squared matches summary.lm on weighted, no-intercept and offset fits", {
  d <- lmm_data()
  d$w <- rep(c(0.5, 1, 2, 4), length.out = nrow(d))
  d$o <- 0.3 * d$z
  cases <- list(
    weighted = list(fit = fastglmm(y ~ x + z, d, weights = w),
                    ref = lm(y ~ x + z, d, weights = w)),
    no_intercept = list(fit = fastglmm(y ~ 0 + x + z, d), ref = lm(y ~ 0 + x + z, d)),
    offset = list(fit = fastglmm(y ~ x, d, offset = o), ref = lm(y ~ x, d, offset = o)),
    offset_term = list(fit = fastglmm(y ~ x + offset(o), d), ref = lm(y ~ x + offset(o), d))
  )
  for (nm in names(cases)) {
    fit <- cases[[nm]]$fit
    s_ref <- summary(cases[[nm]]$ref)
    expect_equal(fit$r_squared, s_ref$r.squared, tolerance = 1e-10, info = nm)
    expect_equal(fit$adj_r_squared, s_ref$adj.r.squared, tolerance = 1e-10, info = nm)
    expect_equal(fit$rss, sum(stats::weighted.residuals(cases[[nm]]$ref)^2),
                 tolerance = 1e-10, info = nm)
  }
  one <- fastglmm(y ~ 1, d)
  expect_identical(c(one$r_squared, one$adj_r_squared), c(0, 0))
  lmm <- fastglmm(y ~ x + (1 | g), d)
  expect_true(is.nan(lmm$r_squared) && is.nan(lmm$adj_r_squared) && is.nan(lmm$rss))
})

test_that("a Gamma GLM prints t on n - p, with its df in the dispersion line", {
  set.seed(24)
  d <- data.frame(x = rnorm(80))
  d$y <- rgamma(80, shape = 4, rate = 4 / exp(0.3 + 0.5 * d$x))
  fit <- fastglmm(y ~ x, d, family = "gamma")
  ref <- glm(y ~ x, data = d, family = Gamma(link = "log"))
  cf <- summary(fit)$coefficients
  expect_identical(colnames(cf), c("Estimate", "Std. Error", "t value", "Pr(>|t|)"))
  # glm's own fit differs from the kernel's at about 1e-6 relative, which the
  # small p-value amplifies; the df are glm's exactly.
  expect_equal(unname(cf[, "t value"]), unname(coef(summary(ref))[, "t value"]),
               tolerance = 1e-5)
  expect_equal(unname(cf[, "Pr(>|t|)"]), unname(coef(summary(ref))[, "Pr(>|t|)"]),
               tolerance = 1e-3)
  expect_equal(unname(fit$coef_df), rep(df.residual(ref), 2))
  out <- paste(capture.output(print(summary(fit))), collapse = "\n")
  expect_match(out, "Dispersion \\(phi, Pearson\\): [0-9.]+ on 78 degrees of freedom")
  expect_false(grepl("R-squared", out, fixed = TRUE))
  held <- fastglmm(y ~ x, d, family = "gamma", dispersion = 0.25)
  expect_identical(held$df_method, "none")
  expect_identical(colnames(summary(held)$coefficients)[3:4], c("z value", "Pr(>|z|)"))
})

test_that("satterthwaite = FALSE: an LMM tests with z and says so; OLS and Gamma keep t", {
  d <- lmm_data()
  expect_no_warning(fit <- fastglmm(y ~ x + (1 | g), d, satterthwaite = FALSE))
  expect_identical(fit$df_method, "disabled")
  expect_true(all(is.nan(fit$coef_df)) && all(is.nan(fit$coef_p)))
  s <- summary(fit)
  expect_identical(colnames(s$coefficients),
                   c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  expect_equal(s$coefficients[, "Pr(>|z|)"], 2 * pnorm(-abs(fixef(fit) / fit$se)))
  expect_identical(tail(capture.output(print(s)), 1L), "df not computed (satterthwaite = FALSE)")
  expect_message(ci <- confint(fit), "df not computed (satterthwaite = FALSE)", fixed = TRUE)
  expect_equal(ci[, 2], fixef(fit) + qnorm(0.975) * fit$se, tolerance = 1e-12)
  on <- fastglmm(y ~ x + (1 | g), d)
  expect_identical(fixef(fit), fixef(on))

  ols <- fastglmm(y ~ x, d, satterthwaite = FALSE)
  expect_identical(ols$df_method, "residual")
  expect_identical(colnames(summary(ols)$coefficients)[3], "t value")
  set.seed(24)
  dg <- data.frame(x = rnorm(80))
  dg$y <- rgamma(80, shape = 4, rate = 4 / exp(0.3 + 0.5 * dg$x))
  expect_identical(fastglmm(y ~ x, dg, family = "gamma", satterthwaite = FALSE)$df_method,
                   "residual")
  expect_error(fastglmm(y ~ x, d, satterthwaite = NA),
               "satterthwaite must be TRUE or FALSE", fixed = TRUE)
})

test_that("after a failed pass the summary tests with z and the intervals are normal", {
  fit <- fastglmm(y ~ x + (1 | g), lmm_data())
  fit$df_method <- "failed"
  fit$coef_df[] <- NaN
  fit$coef_p[] <- NaN
  cf <- summary(fit)$coefficients
  expect_identical(colnames(cf), c("Estimate", "Std. Error", "z value", "Pr(>|z|)"))
  expect_silent(ci <- confint(fit))
  expect_equal(ci[, 1], fixef(fit) - qnorm(0.975) * fit$se, tolerance = 1e-12)
  out <- capture.output(print(summary(fit)))
  expect_false(any(grepl("Satterthwaite degrees of freedom|df not computed", out)))
})

test_that("confint takes the t quantile on each row's df, as confint.lm does", {
  d <- lmm_data()
  ols <- fastglmm(y ~ x + z, d)
  expect_equal(confint(ols), confint(lm(y ~ x + z, d)), tolerance = 1e-8)
  expect_equal(confint(fastglmm(y ~ x, d, weights = rep(1:2, 160))),
               confint(lm(y ~ x, d, weights = rep(1:2, 160))), tolerance = 1e-8)
  expect_equal(confint(ols, parm = "x", level = 0.9),
               confint(lm(y ~ x + z, d), parm = "x", level = 0.9), tolerance = 1e-8)

  lmm <- fastglmm(y ~ x + z + (1 | g), d)
  a <- c(0.025, 0.975)
  want <- fixef(lmm) + lmm$se * t(vapply(lmm$coef_df, function(v) qt(a, v), a))
  expect_equal(unname(confint(lmm)), unname(want), tolerance = 1e-10)
  expect_identical(colnames(confint(lmm)), c("2.5 %", "97.5 %"))
  expect_error(confint(lmm, level = 1), "level must be a number strictly between 0 and 1",
               fixed = TRUE)
  expect_error(confint(lmm, level = NA), "level must be a number strictly between 0 and 1",
               fixed = TRUE)
})

test_that("an aliased coefficient gives an NA interval", {
  d <- lmm_data()
  d$x2 <- 2 * d$x
  fit <- suppressWarnings(fastglmm(y ~ x + x2, d))
  ci <- confint(fit)
  expect_true(all(is.na(ci["x2", ])))
  expect_true(all(is.finite(ci[c("(Intercept)", "x"), ])))
})
