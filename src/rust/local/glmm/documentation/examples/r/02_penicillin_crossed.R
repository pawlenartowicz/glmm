# Recipe 2 -- crossed grouping factors (Penicillin).
#
# `plate` and `sample` are crossed, not nested: every sample was tested on
# every plate, so neither grouping factor is a subdivision of the other. Two
# independent (1 | g) terms, not a `/` nesting operator.
#
# Data: lme4's own `Penicillin` -- library(lme4); data(Penicillin).
#
# Rung: manifest.json rung 3 / goldens/penicillin_lmm.json.

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(fastglmm)) # after lme4 (both export fixef/VarCorr/isSingular/ranef; the later library() call wins)

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  dirname(normalizePath(if (length(file_arg)) file_arg else "."))
})
source(file.path(script_dir, "_oracle.R"))

data(Penicillin)

fit <- fastglmm(diameter ~ (1 | plate) + (1 | sample), Penicillin)

cat("converged:", fit$converged, " singular:", fit$singular, "\n")
summary(fit)

vc <- VarCorr(fit)
cat("\nplate stddev:", attr(vc$plate, "stddev"), "\n")
cat("sample stddev:", attr(vc$sample, "stddev"), "\n")
cat("loglik (REML crit):", fit$loglik, " reml:", fit$reml, "\n")

cat("\noracle cross-check vs goldens/penicillin_lmm.json (manifest rung 3):\n")
g <- load_golden("penicillin_lmm")
est <- g$estimates
check_rel("beta[Intercept]", unname(fixef(fit)[1]), est$beta[1], TOL_BETA_REL)
check_rel("se[Intercept]", unname(fit$se[1]), est$se[1], TOL_SE_REL)
golden_by_group <- setNames(sapply(est$varcomp$stddev, `[`, 1), est$varcomp$group)
check_rel("plate stddev", attr(vc$plate, "stddev")[1], golden_by_group[["plate"]], TOL_STDDEV_REL)
check_rel("sample stddev", attr(vc$sample, "stddev")[1], golden_by_group[["sample"]], TOL_STDDEV_REL)
check_abs("loglik (REML crit)", fit$loglik, est$loglik, TOL_LOGLIK_ABS_LMM)
