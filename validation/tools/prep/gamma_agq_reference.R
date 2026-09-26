#!/usr/bin/env Rscript
# Exact marginal maximum likelihood of the Gamma/log random-intercept GLMM
# y ~ x1 + (1 | g1) on grid cell gaml_int1_g3000p20_bal_base, by adaptive
# Gauss-Hermite quadrature built here from scratch -- no GLMM package, so the
# reference owes nothing to the engines it checks. GLMMadaptive fails on mixed
# Gamma ("no valid set of coefficients"), and lme4 fits a different objective,
# so this is the reference for fit::glmm_tests::gamma_agq_reaches_the_exact_marginal_ml.
#
# Per cluster j the integrand in u is prod_i dgamma(y_i; shape w_i*a, rate
# w_i*a/mu_i) * dnorm(u; 0, sigma), mu_i = exp(x_i'beta + u), a = 1/phi, w_i
# the row's precision weight (1 unweighted). The adaptive nodes are centred on
# the integrand's mode and scaled by its curvature there (the exact curvature,
# as GLMMadaptive and glmm do). The log-likelihood is maximised over
# (beta, log sigma, log phi) with nlminb; 15, 25 and 51 nodes are run to show
# the 25-node value has settled.
#
#   Rscript validation/tools/prep/gamma_agq_reference.R              # unweighted, stdout only
#   Rscript validation/tools/prep/gamma_agq_reference.R --weights    # precision weights, also
#                                                                     # freezes gamma_agq_reference_wts.json
args <- commandArgs(TRUE)
weighted <- length(args) >= 1 && identical(args[1], "--weights")

file_args <- commandArgs(FALSE)
here <- dirname(normalizePath(sub("--file=", "", grep("--file=", file_args, value = TRUE))))
csv <- if (weighted) "../../grid/data/gaml_int1_g3000p20_bal_base_wts.csv" else
  "../../grid/data/fast/gaml_int1_g3000p20_bal_base.csv"
d <- read.csv(file.path(here, csv))
X <- cbind(1, d$x1)
y <- d$y
g <- as.integer(factor(d$g1))
idx <- split(seq_along(y), g)
# Precision weight per row (1 unweighted -- exact under IEEE754 multiplication,
# so multiplying by it leaves the unweighted arm's bytes untouched).
ww <- if (weighted) d$w else rep(1, length(y))

gh <- function(k) {
  # Golub-Welsch for the physicists' Hermite weight exp(-x^2).
  i <- seq_len(k - 1)
  J <- matrix(0, k, k)
  J[cbind(i, i + 1)] <- sqrt(i / 2)
  J[cbind(i + 1, i)] <- sqrt(i / 2)
  e <- eigen(J, symmetric = TRUE)
  list(x = e$values, w = sqrt(pi) * e$vectors[1, ]^2)
}

loglik <- function(par, nodes) {
  beta <- par[1:2]; sigma <- exp(par[3]); a <- exp(-par[4])
  eta0 <- drop(X %*% beta)
  total <- 0
  for (rows in idx) {
    yy <- y[rows]; e0 <- eta0[rows]; wwi <- ww[rows]
    # log integrand h(u) and its first two derivatives (Gamma/log, row shape
    # w_i*a, rate w_i*a/mu)
    h <- function(u) sum(dgamma(yy, shape = wwi * a, rate = wwi * a / exp(e0 + u), log = TRUE)) +
      dnorm(u, 0, sigma, log = TRUE)
    dh <- function(u) sum(wwi * a * (yy / exp(e0 + u) - 1)) - u / sigma^2
    d2h <- function(u) -sum(wwi * a * yy / exp(e0 + u)) - 1 / sigma^2
    u <- 0
    for (it in 1:50) {
      step <- dh(u) / d2h(u)
      u <- u - step
      if (abs(step) < 1e-12) break
    }
    s <- 1 / sqrt(-d2h(u))
    z <- u + sqrt(2) * s * nodes$x
    lv <- vapply(z, h, 0) + nodes$x^2
    m <- max(lv)
    total <- total + log(sqrt(2) * s) + m + log(sum(nodes$w * exp(lv - m)))
  }
  total
}

start <- if (weighted) {
  c(coef(glm(y ~ x1, data = d, family = Gamma("log"), weights = w)), log(0.5), log(0.5))
} else {
  c(coef(glm(y ~ x1, data = d, family = Gamma("log"))), log(0.5), log(0.5))
}
for (k in c(15, 25, 51)) {
  nodes <- gh(k)
  fit <- nlminb(start, function(p) -loglik(p, nodes),
                control = list(eval.max = 1e4, iter.max = 1e4, rel.tol = 1e-10))
  p <- fit$par
  cat(sprintf("nodes %2d  logLik %.8f  beta %.8f %.8f  sigma_u %.8f  sqrt(phi) %.8f  %s\n",
              k, -fit$objective, p[1], p[2], exp(p[3]), exp(p[4] / 2), fit$message))
  # The 25-node value is what settled (the 15/51 sweep above shows it): freeze
  # it as the weighted arm's own reference, next to the unweighted one's
  # hardcoded constants in fit::glmm_tests::gamma_agq_reaches_the_exact_marginal_ml.
  if (weighted && k == 25) {
    suppressMessages(library(jsonlite))
    ref <- list(
      name = "gaml_int1_g3000p20_bal_base_wts",
      nagq = 25L,
      loglik = -fit$objective,
      beta = I(unname(p[1:2])),
      sigma_u = exp(p[3]),
      phi = exp(p[4])
    )
    out <- file.path(here, "gamma_agq_reference_wts.json")
    write(toJSON(ref, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
    cat(sprintf("wrote %s\n", out))
  }
}
