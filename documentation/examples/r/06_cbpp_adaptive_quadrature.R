# Recipe 6 -- adaptive quadrature (cbpp at nAGQ=7).
#
# Recipe 4's model again, but integrated over the random effect with 7-point
# adaptive Gauss-Hermite quadrature instead of the nAGQ=1 Laplace default.
# Eligible here because the model has a single grouping factor (herd) with
# one random effect per level (q=1) -- AGQ's current cap is q<=3 on a single
# binomial/Poisson grouping factor.
#
# Data: lme4's own `cbpp` -- see recipe 4 for the weights= migration from
# lme4's cbind().
#
# Rung: manifest.json rung 5 at nAGQ=7 / goldens/cbpp_agq_k7.json (the same
# study's k=1 case, goldens/cbpp_agq_k1.json, is recipe 4's fit). The
# cross-check below compares beta, se_hessian and the herd standard deviation
# only -- the crate's own oracle test for this exact golden
# (fit_glmm_binomial_agq_matches_lme4, src/fit/glmm_tests.rs) gates the same
# three quantities and deliberately not log-likelihood. That is not a gap:
# the Laplace-vs-AGQ log-likelihood does not even agree with itself across
# nAGQ in lme4's own output (lme4's glmer reports -92.0 at nAGQ=1 and -50.0
# at nAGQ=7 for this exact fit -- a ~42-unit jump despite beta moving by
# <0.001), so log-likelihood is not the quantity nAGQ eligibility is about
# here; beta, se and the variance component are.

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

fit_laplace <- fastglmm(prop ~ period + (1 | herd), cbpp, family = binomial(), weights = size, nAGQ = 1L)
fit_agq <- fastglmm(prop ~ period + (1 | herd), cbpp, family = binomial(), weights = size, nAGQ = 7L)

cat("=== nAGQ=1 (Laplace) ===\n")
summary(fit_laplace)
cat("loglik:", fit_laplace$loglik, "\n")

cat("\n=== nAGQ=7 (adaptive Gauss-Hermite) ===\n")
summary(fit_agq)
cat("loglik:", fit_agq$loglik, "\n")

cat("\nLaplace -> AGQ(7) movement on this data:\n")
b_lap <- fixef(fit_laplace)
b_agq <- fixef(fit_agq)
for (nm in names(b_agq)) {
  d <- b_agq[[nm]] - b_lap[[nm]]
  rel <- abs(d) / abs(b_lap[[nm]])
  cat(sprintf("  beta[%s]: laplace=%.10g agq7=%.10g delta=%.3g rel=%.3g\n", nm, b_lap[[nm]], b_agq[[nm]], d, rel))
}
cat(sprintf("  loglik: laplace=%.10g agq7=%.10g delta=%.3g\n",
            fit_laplace$loglik, fit_agq$loglik, fit_agq$loglik - fit_laplace$loglik))

cat("\noracle cross-check vs goldens/cbpp_agq_k7.json (manifest rung 5 at nAGQ=7):\n")
cat("(beta, se_hessian, herd stddev only -- see header comment on why loglik is excluded)\n")
g <- load_golden("cbpp_agq_k7")
est <- g$estimates
for (i in seq_along(g$coef_names)) {
  check_rel(sprintf("beta[%s]", g$coef_names[i]), unname(b_agq[i]), est$beta[i], TOL_BETA_REL)
}
for (i in seq_along(g$coef_names)) {
  check_rel(sprintf("se_hessian[%s]", g$coef_names[i]), unname(fit_agq$se[i]), est$se_hessian[i], TOL_SE_HESSIAN_REL)
}
vc <- VarCorr(fit_agq)
check_rel("herd stddev", attr(vc$herd, "stddev")[1], est$varcomp$stddev[[1]][1], TOL_STDDEV_REL)
