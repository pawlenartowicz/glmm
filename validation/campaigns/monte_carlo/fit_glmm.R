# fastglmm (the deliverable engine) over escalating batches. Arms: Laplace
# (nAGQ=1) and AGQ (nAGQ=11) on ALL 24 cells.
#
# AGQ runs on the slope cells too: the kernel does adaptive quadrature for vector
# random effects (single grouping factor, <=3 REs per group), which glmer cannot
# (`nAGQ > 1` errors on `(1 + time | id)`). Restricting AGQ to the RI cells would
# score glmm's *Laplace* against GLMMadaptive's *adaptive* fit on tau1/rho01 -
# the only published engine doing vector AGQ - which is not a like-for-like
# comparison. On the slope cells GLMMadaptive is glmm_AGQ's comparator; lme4_AGQ
# has no counterpart there (published "-").
suppressMessages(library(fastglmm))
source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))
# ACC_OUT redirects the output dir (the shrinkage sweep writes one dir per arm so
# append_fits does not skip against the frozen baseline's 1000 reps); ACC_SKIP_AGQ
# runs the LA arm only (the sweep is LA-only; the AGQ arm would burn unbudgeted fits).
OUT <- Sys.getenv("ACC_OUT", ""); if (!nzchar(OUT)) OUT <- file.path(ACC_DIR, "results", "glmm")
dir.create(OUT, FALSE, TRUE)
ACC_SKIP_AGQ <- nzchar(Sys.getenv("ACC_SKIP_AGQ", ""))
GLMM_BLOWUP <- 20

fit_rep <- function(cell, nAGQ) function(df, i) {
  est <- NA_EST; good <- FALSE; err <- TRUE; secs <- NA_real_; sing <- NA
  tm <- system.time(fit <- tryCatch(suppressWarnings(
    fastglmm(cell_formula(cell$slope), data = df, family = cell_family(cell$family),
             nAGQ = nAGQ)), error = function(e) e))["elapsed"]
  if (!inherits(fit, "error")) {
    err <- FALSE; secs <- tm; sing <- isTRUE(fit$singular)
    vc <- fastglmm::VarCorr(fit)$id; sd <- attr(vc, "stddev"); cr <- attr(vc, "correlation")
    b <- fit$beta
    est <- c(beta0=b[[1]], beta1=b[[2]], beta2=b[[3]], beta3=b[[4]], tau0=sd[[1]],
             tau1 = if (cell$slope) sd[[2]] else NA_real_,
             rho01 = if (cell$slope) cr[1,2] else NA_real_)
    good <- isTRUE(fit$converged) &&
      all(is.finite(est[c("beta0","beta1","beta2","beta3","tau0")])) &&
      all(abs(est[c("beta0","beta1","beta2","beta3")]) < GLMM_BLOWUP) &&
      est[["tau0"]] < GLMM_BLOWUP && (is.na(est[["tau1"]]) || est[["tau1"]] < GLMM_BLOWUP)
  }
  mk_row(i, est, good, err, secs, sing)
}

for (cell in CELLS) {
  if (length(ACC_ONLY) && !(cell$id %in% ACC_ONLY)) next
  cat(cell$id, "\n"); dl <- load_cell(cell)
  report_arm(cell, "glmm_LA", append_fits(file.path(OUT, paste0(cell$id, "__glmm_LA.csv")),
             dl, ACC_UPTO, fit_rep(cell, 1L)))
  if (!ACC_SKIP_AGQ)
    report_arm(cell, "glmm_AGQ", append_fits(file.path(OUT, paste0(cell$id, "__glmm_AGQ.csv")),
               dl, ACC_UPTO, fit_rep(cell, NAGQ_AGQ)))
}
cat("glmm done (upto", ACC_UPTO, ")\n")
