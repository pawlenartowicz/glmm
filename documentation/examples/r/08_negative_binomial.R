# Recipe 8 -- negative binomial (sim_nb).
#
# A negative-binomial GLM: `dispersion` on the returned object is theta-hat,
# the NB shape parameter (MASS::glm.nb's theta), not phi -- unlike Gamma,
# where dispersion is the Pearson phi. Overdispersion relative to Poisson is
# 1/theta; a large theta means "close to Poisson", not "a lot of extra
# variance".
#
# Data: validation/data/simulated/sim_nb.csv (a fixture generated for the
# validation harness, since no lme4-bundled dataset exercises negative
# binomial). Not one of lme4's own datasets, so unlike recipes 1-7 this reads
# a file path rather than library(lme4); data(...).
#
# Rung: manifest.json's m3_goldens entry `sim_nb_glm` / goldens/sim_nb_glm.json
# -- a GLM (no random effects), which is the model this recipe fits.

suppressPackageStartupMessages(library(fastglmm))

script_dir <- local({
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  dirname(normalizePath(if (length(file_arg)) file_arg else "."))
})
source(file.path(script_dir, "_oracle.R"))

sim_nb <- read.csv(file.path(script_dir, "..", "..", "..", "validation", "data", "simulated", "sim_nb.csv"))

fit <- fastglmm(y ~ x + grp, sim_nb, family = "negativebinomial")

cat("converged:", fit$converged, " singular:", fit$singular, "\n")
summary(fit)
cat("theta (NB shape):", fit$dispersion, "\n")
cat("loglik:", fit$loglik, "\n")

cat("\noracle cross-check vs goldens/sim_nb_glm.json (GLM, no random effects):\n")
g <- load_golden("sim_nb_glm")
est <- g$estimates
b <- fixef(fit)
for (i in seq_along(g$coef_names)) {
  check_rel(sprintf("beta[%s]", g$coef_names[i]), unname(b[i]), est$beta[i], TOL_BETA_REL)
}
for (i in seq_along(g$coef_names)) {
  check_rel(sprintf("se[%s]", g$coef_names[i]), unname(fit$se[i]), est$se[i], TOL_SE_REL)
}
check_rel("theta", fit$dispersion, est$theta, TOL_BETA_REL)
check_abs("loglik", fit$loglik, est$loglik, TOL_LOGLIK_ABS_GLMM)
