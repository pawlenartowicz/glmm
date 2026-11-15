#!/usr/bin/env Rscript
# The negative-binomial shape maximum of two fixtures, by a direct profile of
# the likelihood in base R (stats only, no MASS). It prints the values at
# %.17g; the tests in src/fit/glm_tests.rs keep them inline. Nothing is
# written to disk.
#
# 1. validation/data/simulated/sim_nb_tinytheta.csv (gen_nb_tinytheta_data.py
#    in this directory), y ~ x: one positive count in 400 rows. Consumer:
#    nb_glm_tiny_shape_matches_the_base_r_profile (theta-hat and loglik).
# 2. The note fixture, built here: 99999 zeros and one count of 1e9,
#    intercept only. Consumer: nb_glm_on_the_lower_shape_limit_carries_the_note
#    (theta-hat, which lies below glmm's lower search limit).
#
# The profile is ll(theta) = max_beta sum(dnbinom(y, size = theta,
# mu = exp(X beta), log = TRUE)). At each theta, beta comes from Fisher scoring
# (weights mu / (1 + mu / theta), log link) with step halving, to a relative
# change in ll below 1e-14, started from the beta of the previous theta. The
# outer search is optimize(maximum = TRUE, tol = 1e-10) on ln theta over
# [ln 1e-9, ln 1]. optimize stops at sqrt(eps) * |ln theta| + tol / 3, so the
# printed theta is good to about 1e-7 relative; the log-likelihood, flat at its
# maximum, is good to round-off. On fixture 2, mu-hat = mean(y) at every theta, so the
# profile is one-dimensional; its interval is [ln 1e-12, ln 1].
#
# R must be at validation/grid/versions.json's `R` pin, or the run stops
# (jsonlite reads the pin). Run from the crate root:
#   Rscript validation/tools/prep/nb_tiny_shape_profile_reference.R

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", ".."))

local({
  grid_dir <- file.path(suite_dir, "grid")
  source(file.path(grid_dir, "engines", "versions.R"), local = TRUE)
  V <- grid_versions(grid_dir)
  if (!identical(norm_version(getRversion()), norm_version(V[["R"]])))
    stop(sprintf("R version mismatch: grid/versions.json pins %s, running %s",
                 V[["R"]], getRversion()), call. = FALSE)
})
cat(R.version.string, "\n")

nb_ll <- function(y, mu, theta) sum(dnbinom(y, size = theta, mu = mu, log = TRUE))

# Profile log-likelihood at ln theta; `state$beta` carries the warm start.
profile_ll <- function(ln_theta, y, X, state) {
  theta <- exp(ln_theta)
  beta <- state$beta
  ll <- nb_ll(y, exp(drop(X %*% beta)), theta)
  for (it in 1:1000) {
    eta <- drop(X %*% beta)
    mu <- exp(eta)
    w <- mu / (1 + mu / theta)
    z <- eta + (y - mu) / mu
    step <- lm.wfit(X, z, w)$coefficients - beta
    for (h in 0:60) {
      cand <- beta + step / 2^h
      ll_cand <- nb_ll(y, exp(drop(X %*% cand)), theta)
      if (is.finite(ll_cand) && ll_cand >= ll) break
    }
    if (!(is.finite(ll_cand) && ll_cand >= ll)) break
    done <- abs(ll_cand - ll) <= 1e-14 * abs(ll_cand)
    beta <- cand
    ll <- ll_cand
    if (done) break
  }
  state$beta <- beta
  ll
}

fit_profile <- function(label, y, X, interval) {
  state <- new.env()
  state$beta <- c(log(mean(y)), rep(0, ncol(X) - 1))
  o <- optimize(profile_ll, interval, y = y, X = X, state = state,
                maximum = TRUE, tol = 1e-10)
  cat(sprintf("%s theta %.17g\n", label, exp(o$maximum)))
  cat(sprintf("%s loglik %.17g\n", label, o$objective))
}

d <- read.csv(file.path(suite_dir, "data", "simulated", "sim_nb_tinytheta.csv"))
fit_profile("tiny", d$y, cbind(1, d$x), log(c(1e-9, 1)))

y_limit <- c(rep(0, 99999), 1e9)
fit_profile("limit", y_limit, matrix(1, length(y_limit), 1), log(c(1e-12, 1)))
