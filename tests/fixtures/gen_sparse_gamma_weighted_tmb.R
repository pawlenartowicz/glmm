#!/usr/bin/env Rscript
# Generator for `fit_sparse_gamma_glmm_weighted_matches_glmmtmb`
# (src/sparse/tests.rs): glmmTMB's Gamma/log fit of the `sim_sparse_gamma` design
# (validation/data/simulated/sim_sparse_gamma.csv, written by
# validation/tools/prep/export_data.R) with precision weights
# w_i = 1 + 0.2 * ((i mod 3) - 1) on the 0-based row index, entered as the
# dispersion offset `dispformula = ~ offset(log(w))`. Prints the test's REF_*
# constants; the test keeps them inline, so nothing is written to disk.
#
# Run from the crate root:
#   Rscript tests/fixtures/gen_sparse_gamma_weighted_tmb.R

suppressMessages(library(glmmTMB))

d <- read.csv(file.path("validation", "data", "simulated", "sim_sparse_gamma.csv"))
d$gp <- factor(d$gp); d$ge <- factor(d$ge)
d$w <- 1 + 0.2 * (((seq_len(nrow(d)) - 1) %% 3) - 1)
f <- glmmTMB(y ~ 1 + x1 + x2 + x3 + x4 + (1|gp) + (1 + x1 + x2 + x3 + x4 | ge),
             family = Gamma("log"), dispformula = ~ offset(log(w)), data = d,
             control = glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4)))

cat(sprintf("// R %s, glmmTMB %s; convergence %d, pdHess %s\n",
            getRversion(), packageVersion("glmmTMB"), f$fit$convergence, f$sdr$pdHess))
fmt <- function(v) paste(sprintf("%.17g", v), collapse = ", ")
cat(sprintf("const REF_BETA: [f64; 5] = [%s];\n", fmt(unname(fixef(f)$cond))))
cat(sprintf("const REF_SE_HESSIAN: [f64; 5] = [%s];\n", fmt(unname(sqrt(diag(vcov(f)$cond))))))
cat(sprintf("const REF_DISPERSION: f64 = %.17g;\n", sigma(f)^2))
cat(sprintf("const REF_LOGLIK: f64 = %.17g;\n", as.numeric(logLik(f))))
