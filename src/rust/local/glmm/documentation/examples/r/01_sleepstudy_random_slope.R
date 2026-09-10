# Recipe 1 -- correlated random slope (sleepstudy).
#
# Data: lme4's own `sleepstudy` -- library(lme4); data(sleepstudy), no file
# path needed.
#
# Rung: manifest.json rung 2 / goldens/sleepstudy_lmm.json.

suppressPackageStartupMessages(library(lme4))     # for the sleepstudy data
suppressPackageStartupMessages(library(fastglmm)) # loaded AFTER lme4 so its
                   # fixef/VarCorr/isSingular/ranef methods are the ones in
                   # scope (both packages export the same generics; the
                   # later library() call wins)

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  dirname(normalizePath(if (length(file_arg)) file_arg else "."))
})
source(file.path(script_dir, "_oracle.R"))

data(sleepstudy)

fit <- fastglmm(Reaction ~ Days + (1 + Days | Subject), sleepstudy)

cat("converged:", fit$converged, " singular:", fit$singular, "\n")
print(fit)
cat("\n--- summary ---\n")
summary(fit)

vc <- VarCorr(fit)
sd_subject <- attr(vc$Subject, "stddev")
corr_subject <- attr(vc$Subject, "correlation")
cat("\nSubject stddevs:", sd_subject, "\n")
cat("Subject corr[1,2]:", corr_subject[1, 2], "\n")
cat("loglik (REML crit):", fit$loglik, " reml:", fit$reml, "\n")

cat("\noracle cross-check vs goldens/sleepstudy_lmm.json (manifest rung 2):\n")
g <- load_golden("sleepstudy_lmm")
est <- g$estimates
check_rel("beta[Intercept]", unname(fixef(fit)[1]), est$beta[1], TOL_BETA_REL)
check_rel("beta[Days]", unname(fixef(fit)[2]), est$beta[2], TOL_BETA_REL)
check_rel("se[Intercept]", unname(fit$se[1]), est$se[1], TOL_SE_REL)
check_rel("se[Days]", unname(fit$se[2]), est$se[2], TOL_SE_REL)
check_rel("Subject sd[Intercept]", sd_subject[1], est$varcomp$stddev[[1]][1], TOL_STDDEV_REL)
check_rel("Subject sd[Days]", sd_subject[2], est$varcomp$stddev[[1]][2], TOL_STDDEV_REL)
check_abs("loglik (REML crit)", fit$loglik, est$loglik, TOL_LOGLIK_ABS_LMM)
