# Data generators shaped like the Li & Signorelli harness (accuracy-benchmark
# spec §3, docs/GLMM/plans/2026-07-12-accuracy-benchmark-spec.md): numeric
# time grid t, Bernoulli(0.4) treatment d, grouping g; random intercept or
# intercept + t slope. Used by both the unit tests and the gate-5 acceptance
# test against glmer.

benchmark_data <- function(seed, n_g = 60L, m = 10L,
                           beta = c(-0.5, 0.7, 0.4, 0.2),
                           tau0 = 0.8, tau1 = 0, rho = 0,
                           family = c("binomial", "poisson")) {
  family <- match.arg(family)
  set.seed(seed)
  g <- factor(rep(seq_len(n_g), each = m))
  t <- rep(seq(0, length.out = m, by = 1 / m), n_g)
  d <- rbinom(n_g * m, 1L, 0.4)
  if (tau1 > 0) {
    # Correlated (intercept, slope) REs via the Cholesky of the 2x2 cov.
    z0 <- rnorm(n_g)
    z1 <- rnorm(n_g)
    u0 <- tau0 * z0
    u1 <- tau1 * (rho * z0 + sqrt(1 - rho^2) * z1)
  } else {
    u0 <- rnorm(n_g, sd = tau0)
    u1 <- rep(0, n_g)
  }
  eta <- beta[1] + beta[2] * t + beta[3] * d + beta[4] * t * d +
    u0[as.integer(g)] + u1[as.integer(g)] * t
  y <- if (family == "binomial") {
    rbinom(n_g * m, 1L, plogis(eta))
  } else {
    rpois(n_g * m, exp(eta))
  }
  data.frame(y = y, t = t, d = d, g = g)
}
