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
    "Optimizer evaluations:", "Wald z"
  )
  pos <- vapply(order, function(s) regexpr(s, sout, fixed = TRUE)[1], 0)
  expect_true(all(pos > 0), info = paste(names(pos)[pos <= 0], collapse = ", "))
  expect_equal(pos, sort(pos))
  # No REML line on an ML fit.
  expect_false(grepl("REML criterion", sout, fixed = TRUE))
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

test_that("boundary fits warn with lme4's text plus the pinned component", {
  # tau0 = 0 data: the RE variance pins to the boundary.
  d <- benchmark_data(seed = 107, family = "binomial", tau0 = 1e-8)
  expect_warning(fit <- fastglmm(y ~ t + (1 | g), d, family = binomial()),
                 paste0("boundary \\(singular\\) fit: see help\\('isSingular'\\); ",
                        "sd\\(\\(Intercept\\) \\| g\\) pinned at the variance boundary"))
  expect_true(isSingular(fit))
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
  # sqrt(offdiag^2 + diag^2), so it lands at ~1e-10 and the correlation at
  # 1 + 2e-16. Reading `pinned` names it.
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

  expect_warning(fit <- fastglmm(y ~ x + (1 + x | g), d),
                 paste0("boundary \\(singular\\) fit: see help\\('isSingular'\\); ",
                        "sd\\(x \\| g\\) pinned at the variance boundary"))
  expect_true(isSingular(fit))
  expect_equal(fit$diagnostics$boundary, "at_boundary")
  # The SLOPE component is the pinned one, aligned with the varcorr block.
  expect_equal(fit$diagnostics$pinned, list(c(FALSE, TRUE)))

  vc <- VarCorr(fit)$g
  sd <- attr(vc, "stddev")
  expect_length(fit$diagnostics$pinned[[1L]], length(sd))
  # The pinned slot is negligible against its sibling but is NOT exactly 0,
  # and the correlation is NOT exactly +/-1.
  expect_false(sd[[2L]] == 0)
  expect_lt(sd[[2L]] / sd[[1L]], 1e-6)
  expect_false(abs(attr(vc, "correlation")[2L, 1L]) == 1)
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

  expect_warning(fit <- fastglmm(y ~ x + (1 | g) + (1 + x | h), d),
                 "sd\\(\\(Intercept\\) \\| g\\) pinned at the variance boundary")
  expect_true(isSingular(fit))
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
  expect_match(conditionMessage(cond), "b is entangled with one or more other columns")

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

test_that("pirls_exhausted message distinguishes the final re-evaluation", {
  # No known dataset reaches final_eval=TRUE end-to-end, so both message
  # branches are asserted from constructed notes; the Rust-side test
  # pirls_exhausted_payload_survives_flattening pins the payload itself.
  note <- list(kind = "pirls_exhausted", columns = integer(0), pivot = NaN,
               evals = 3L, final_eval = FALSE, detail = "")
  benign <- tryCatch(fastglmm:::.warn_note(note, character(0)),
                     warning = identity)
  expect_s3_class(benign, "fastglmm_pirls_exhausted")
  expect_match(conditionMessage(benign),
               "observation-only and no fitted number is affected")

  note$evals <- 0L
  note$final_eval <- TRUE
  serious <- tryCatch(fastglmm:::.warn_note(note, character(0)),
                      warning = identity)
  expect_s3_class(serious, "fastglmm_pirls_exhausted")
  expect_match(conditionMessage(serious),
               "the reported estimates rest on that truncated solve")
})

test_that("re_design_scale_spread message names the grouping and ratio", {
  # No fixture here drives the note through a real fit (the Rust-side
  # end-to-end test covers that:
  # fit::common_tests::re_design_scale_spread_note_fires_on_mismatched_slope_scale),
  # so the message is asserted from a constructed note.
  note <- list(kind = "re_design_scale_spread", columns = integer(0), pivot = NaN,
               evals = 0L, final_eval = FALSE, detail = "g", ratio = 4200.0)
  cond <- tryCatch(fastglmm:::.warn_note(note, character(0)),
                   warning = identity)
  expect_s3_class(cond, "fastglmm_re_design_scale_spread")
  expect_match(conditionMessage(cond), "grouping 'g'")
  expect_match(conditionMessage(cond), "4.2e\\+03")
  expect_match(conditionMessage(cond), "scales the columns internally")
})

test_that("hessian_se_fallback message", {
  note <- list(kind = "hessian_se_fallback", columns = integer(0), pivot = NaN,
               evals = 0L, final_eval = FALSE, detail = "", ratio = NaN)
  cond <- tryCatch(fastglmm:::.warn_note(note, character(0)),
                   warning = identity)
  expect_s3_class(cond, "fastglmm_hessian_se_fallback")
  expect_match(conditionMessage(cond), "not positive definite")
  expect_match(conditionMessage(cond), "stddev_se is NaN")
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
  # cbind(): y is the proportion the kernel fitted, not either column.
  d$s <- d$y
  d$f <- 1L - d$y + 1L
  fit2 <- fastglmm(cbind(s, f) ~ t + (1 | g), d, family = binomial())
  expect_equal(unname(fit2$y), d$s / (d$s + d$f))
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
