# Recipe 4 -- aggregated binomial via weights= (cbpp).
#
# lme4 writes an aggregated binomial as cbind(incidence, size - incidence) ~
# .... fastglmm's shared formula parser also accepts cbind() directly with
# family = binomial(), but both arguments must be columns -- compute the
# failures column first (failures <- size - incidence; arithmetic inside
# cbind() itself is not accepted) and pass cbind(incidence, failures) ~ ....
# This recipe instead passes the success proportion as the response and the
# trial count as weights= -- exactly lme4's own objective underneath cbind(),
# just spelled differently.
#
# Data: lme4's own `cbpp` -- library(lme4); data(cbpp).
#
# Rung: manifest.json rung 5 / goldens/cbpp_agq_k1.json (nAGQ=1 is the
# Laplace default this recipe fits -- the same model the AGQ study calls its
# k=1 case).

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(fastglmm)) # after lme4 (both export fixef/VarCorr/isSingular/ranef; the later library() call wins)

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  dirname(normalizePath(if (length(file_arg)) file_arg else "."))
})
source(file.path(script_dir, "_oracle.R"))

data(cbpp)
cbpp$prop <- cbpp$incidence / cbpp$size

fit <- fastglmm(prop ~ period + (1 | herd), cbpp,
                 family = binomial(), weights = size)

cat("converged:", fit$converged, " singular:", fit$singular, "\n")
summary(fit)

vc <- VarCorr(fit)
cat("\nherd stddev:", attr(vc$herd, "stddev")[1], "\n")
cat("loglik:", fit$loglik, "\n")

cat("\noracle cross-check vs goldens/cbpp_agq_k1.json (manifest rung 5, nAGQ=1):\n")
g <- load_golden("cbpp_agq_k1")
est <- g$estimates
for (i in seq_along(g$coef_names)) {
  check_rel(sprintf("beta[%s]", g$coef_names[i]), unname(fixef(fit)[i]), est$beta[i], TOL_BETA_REL)
}
for (i in seq_along(g$coef_names)) {
  check_rel(sprintf("se_hessian[%s]", g$coef_names[i]), unname(fit$se[i]), est$se_hessian[i], TOL_SE_HESSIAN_REL)
}
check_rel("herd stddev", attr(vc$herd, "stddev")[1], est$varcomp$stddev[[1]][1], TOL_STDDEV_REL)
check_abs("loglik", fit$loglik, est$loglik, TOL_LOGLIK_ABS_GLMM)
