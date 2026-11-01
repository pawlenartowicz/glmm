# Recipe 9 -- offset (sim_poisson_offset).
#
# A Poisson rate model against a known exposure: each row's expected count
# scales with exposure, so offset = log(exposure) is added to the linear
# predictor with a fixed coefficient of 1 rather than an estimated one. Drop
# the offset and the model instead treats every row as equally exposed,
# which folds the (here, substantial) exposure variation into the fixed
# effect and the random-effect variance instead of explaining it away.
#
# Data: validation/data/simulated/sim_poisson_offset.csv (a fixture generated
# for the validation harness with a log_exposure column already computed --
# offset= takes the log-exposure directly, not the raw exposure). Not one of
# lme4's own datasets, so unlike recipes 1-7 this reads a file path rather
# than library(lme4); data(...).
#
# No manifest rung's goldens/ entry covers this dataset (it is registered at
# manifest rung 28 with a real offset field, but no lme4 reference JSON was
# frozen for it under validation/goldens/) -- this recipe's output is a run,
# not an oracle-pinned result.

suppressPackageStartupMessages(library(fastglmm))

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  dirname(normalizePath(if (length(file_arg)) file_arg else "."))
})

sim_poisson_offset <- read.csv(file.path(script_dir, "..", "..", "..", "validation",
                                          "data", "simulated", "sim_poisson_offset.csv"))
# `cluster` is stored as a plain integer id; the grouping factor in (1 | cluster)
# must be a factor, not a numeric column.
sim_poisson_offset$cluster <- factor(sim_poisson_offset$cluster)

fit_with <- fastglmm(y ~ x + (1 | cluster), sim_poisson_offset,
                      family = poisson(), offset = log_exposure)
fit_without <- fastglmm(y ~ x + (1 | cluster), sim_poisson_offset, family = poisson())

cat("=== with offset = log(exposure) ===\n")
summary(fit_with)
vc_with <- VarCorr(fit_with)
cat("cluster stddev:", attr(vc_with$cluster, "stddev")[1], "\n")

cat("\n=== without the offset (exposure variation folded into the fit) ===\n")
summary(fit_without)
vc_without <- VarCorr(fit_without)
cat("cluster stddev:", attr(vc_without$cluster, "stddev")[1], "\n")

b_with <- fixef(fit_with)
b_without <- fixef(fit_without)
cat(sprintf(
  "\nDropping a real offset does not just bias the intercept: here the slope on x moves from %.4g (with offset) to %.4g (without), and the cluster standard deviation moves from %.4g to %.4g -- the unexplained exposure variation is absorbed by both the fixed and the random-effect side, not cleanly by either alone.\n",
  b_with[["x"]], b_without[["x"]], attr(vc_with$cluster, "stddev")[1], attr(vc_without$cluster, "stddev")[1]
))
cat("\n(no goldens/ entry for sim_poisson_offset -- a run, not an oracle-pinned result)\n")
