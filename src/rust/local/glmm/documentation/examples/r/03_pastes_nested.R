# Recipe 3 -- nested grouping factors (Pastes).
#
# `cask` only means something inside its `batch` -- cask "a" of batch "A" is
# unrelated to cask "a" of batch "B" -- so the model is (1 | batch/cask), R's
# nesting shorthand for "a random intercept per batch, plus a random
# intercept per batch:cask combination". Two variance components come out:
# one for the coarse `batch` grouping (10 levels), one for the finer
# `batch:cask` grouping (30 levels, one per cask actually observed within its
# batch).
#
# Data: lme4's own `Pastes` -- library(lme4); data(Pastes).
#
# Rung: manifest.json rung 4 / goldens/pastes_lmm.json.

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(fastglmm)) # after lme4 (both export fixef/VarCorr/isSingular/ranef; the later library() call wins)

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  dirname(normalizePath(if (length(file_arg)) file_arg else "."))
})
source(file.path(script_dir, "_oracle.R"))

data(Pastes)

fit <- fastglmm(strength ~ (1 | batch/cask), Pastes)

cat("converged:", fit$converged, " singular:", fit$singular, "\n")
summary(fit)

cat("\nloglik (REML crit):", fit$loglik, "\n")

cat("\noracle cross-check vs goldens/pastes_lmm.json (manifest rung 4):\n")
g <- load_golden("pastes_lmm")
est <- g$estimates
check_rel("beta[Intercept]", unname(fixef(fit)[1]), est$beta[1], TOL_BETA_REL)
check_rel("se[Intercept]", unname(fit$se[1]), est$se[1], TOL_SE_REL)

# The golden names the nested grouping "cask:batch"; fastglmm names the same
# grouping "batch:cask" -- same set of factors, order differs. Match on the
# sorted factor set rather than the literal string.
key <- function(name) paste(sort(strsplit(name, ":")[[1]]), collapse = ":")
golden_sd <- sapply(est$varcomp$stddev, `[`, 1)
names(golden_sd) <- sapply(est$varcomp$group, key)
vc <- VarCorr(fit)
for (group_name in fit$re_group_names) {
  sd <- attr(vc[[group_name]], "stddev")[1]
  check_rel(paste0(group_name, " stddev"), sd, golden_sd[[key(group_name)]], TOL_STDDEV_REL)
}
check_abs("loglik (REML crit)", fit$loglik, est$loglik, TOL_LOGLIK_ABS_LMM)
