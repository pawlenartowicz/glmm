#!/usr/bin/env Rscript
# The lme4 optimum of the precise two-slope boundary-pin fixture, for
# `trough_polish_reaches_the_optimum_on_precise_data` (src/fit/core_tests.rs),
# which keeps it inline as PRECISE_TWO_ZERO_SLOPES_LME4_DEVIANCE. It prints the
# value; nothing is written to disk.
#
# Data: validation/data/simulated/sim_poisson_precise_two_zero_slopes.csv
# (written by gen_boundary_pin_data.R). Model: Poisson log,
# y ~ 1 + x1 + x2 + (1 + x1 + x2 | g), glmer at nAGQ = 1 with tolPwrss = 1e-10.
#
# lme4's default fit stops far from the optimum on this data (190 deviance
# units above it, bobyqa out of evaluations), and a tight bobyqa from the
# default start stops 1.7e-6 above it. So the fit is a chain, each step started
# at the last one's (theta, beta): bobyqa with rhoend = 1e-10 from lme4's own
# start, Nelder_Mead twice, bobyqa with rhoend = 1e-10 once more. Every step's
# value is printed; the last is the reference. No step starts from a glmm fit.
#
# The value is on glmm's deviance scale (`Fit::deviance`): -2 logLik plus
# 2 * sum(dpois(y, y, log = TRUE)), the saturated Poisson log-likelihood.
# glmerMod's deviance() is the conditional deviance and is not comparable.
#
# Run from the crate root, with the pinned oracle library on R_LIBS:
#   Rscript validation/tools/prep/boundary_pin_lme4_reference.R

suppressMessages(library(lme4))

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", ".."))
d <- read.csv(file.path(suite_dir, "data", "simulated",
                        "sim_poisson_precise_two_zero_slopes.csv"))
f <- y ~ 1 + x1 + x2 + (1 + x1 + x2 | g)
saturated <- sum(dpois(d$y, d$y, log = TRUE))
dev <- function(m) -2 * as.numeric(logLik(m)) + 2 * saturated
ctl <- function(optimizer, ...) {
  glmerControl(optimizer = optimizer, tolPwrss = 1e-10,
               check.conv.singular = "ignore", optCtrl = list(...))
}
steps <- list(
  list("bobyqa", rhoend = 1e-10, maxfun = 1e5),
  list("Nelder_Mead", maxfun = 1e5),
  list("Nelder_Mead", maxfun = 1e5),
  list("bobyqa", rhoend = 1e-10, maxfun = 1e5)
)

cat("lme4", as.character(packageVersion("lme4")), "on", R.version.string, "\n")
start <- NULL
for (s in steps) {
  control <- do.call(ctl, s)
  m <- if (is.null(start)) {
    glmer(f, d, poisson, nAGQ = 1, control = control)
  } else {
    glmer(f, d, poisson, nAGQ = 1, start = start, control = control)
  }
  cat(sprintf("%-12s deviance %.10f\n", s[[1]], dev(m)))
  start <- list(theta = getME(m, "theta"), fixef = fixef(m))
}
cat(sprintf("PRECISE_TWO_ZERO_SLOPES_LME4_DEVIANCE = %.10f\n", dev(m)))
cat("theta", format(getME(m, "theta"), digits = 10), "\n")
cat("beta ", format(fixef(m), digits = 12), "\n")
