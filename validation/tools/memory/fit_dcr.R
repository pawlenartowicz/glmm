#!/usr/bin/env Rscript
# One arm of the weight-power and CR2 benchmark on one dataset
# of gen_dcr_bench.py, in its own process so dcr_bench.sh reads its wall time
# and peak RSS from /usr/bin/time. Not a validation rung: nothing is compared.
#
#   Rscript fit_dcr.R <arm> <csv> <formula>
#
# arm 0   load every package and read the data, no fit (the baseline)
#     1   fastglmm, weight_power = 1, se_type = "model" (the claimed weights)
#     2   fastglmm, weight_power = "estimate", se_type = "model" (2 - 1: the delta search)
#     3   fastglmm, weight_power = "estimate", se_type = "robust" (3 - 2: CR2 or HC2)
#     3c  arm 3, then one robust contrast_test, whose time is printed (the
#         per-call copy of the robust record into the engine)
#     4   nlme lme (gls without a grouping) with varPower, then clubSandwich
#         CR2 with Satterthwaite df (the published route)
# Prints one status line.

suppressMessages({
  library(fastglmm)
  library(nlme)
  library(clubSandwich)
})

args <- commandArgs(TRUE)
arm <- args[1]
data <- read.csv(args[2])
f <- as.formula(args[3])
if ("g" %in% names(data)) data$g <- factor(data$g)
if (arm == "0") {
  cat(sprintf("arm 0: n=%d\n", nrow(data)))
  quit(save = "no")
}

if (arm %in% c("1", "2", "3", "3c")) {
  power <- if (arm == "1") 1 else "estimate"
  se <- if (arm %in% c("3", "3c")) "robust" else "model"
  m <- fastglmm(f, data = data, weights = data$w, weight_power = power, se_type = se)
  stopifnot(m$converged)
  if (arm == "3c") {
    L <- matrix(0, 1, length(m$beta))
    L[1, 2] <- 1
    t <- system.time(contrast_test(m, L))[["elapsed"]]
    cat(sprintf("contrast_test %.3f s\n", t))
  }
  cat(sprintf("arm %s: n=%d weight_power=%.6f se_type=%s\n", arm, nrow(data), m$weight_power,
              m$se_type))
} else if (arm == "4") {
  fixed <- reformulas::nobars(f)
  bars <- reformulas::findbars(f)
  vf <- varPower(form = ~ w)
  if (length(bars) == 0L) {
    # data in the call itself: clubSandwich re-reads it from there
    m <- eval(bquote(gls(.(fixed), data = .(data), weights = .(vf), method = "REML")))
    cluster <- factor(seq_len(nrow(data)))
  } else {
    random <- as.formula(paste("~", deparse(bars[[1]][[2]]), "| g"))
    m <- lme(fixed, random = random, data = data, weights = vf, method = "REML")
    cluster <- data$g
  }
  ct <- coef_test(m, vcov = "CR2", cluster = cluster, test = "Satterthwaite")
  power <- -2 * unname(coef(m$modelStruct$varStruct, unconstrained = FALSE)[["power"]])
  cat(sprintf("arm 4: n=%d weight_power=%.6f df[2]=%.3f\n", nrow(data), power, ct$df_Satt[2]))
} else {
  stop("unknown arm ", arm)
}
