# lme4 over escalating batches: validation gate (must reproduce the paper's
# published lme4 numbers on the identical data) + Laplace timing denominator.
# Arms: lme4_LA (nAGQ=1, all cells) and lme4_AGQ (nAGQ=11, RI-only - glmer refuses
# AGQ on vector REs). "good" is the paper's exact check (estimation_tool.R:39-44),
# so only their good reps feed the metric.
suppressMessages(library(lme4))
source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))
OUT <- file.path(ACC_DIR, "results", "lme4"); dir.create(OUT, FALSE, TRUE)

fit_rep <- function(cell, nAGQ) function(df, i) {
  est <- NA_EST; good <- FALSE; err <- TRUE; secs <- NA_real_; sing <- NA
  tm <- system.time(m <- try(suppressMessages(suppressWarnings(
    glmer(cell_formula(cell$slope), family = cell_family(cell$family),
          data = df, nAGQ = nAGQ))), silent = TRUE))["elapsed"]
  if (!inherits(m, "try-error")) {
    err <- FALSE; secs <- tm; sing <- isTRUE(lme4::isSingular(m))
    good <- (m@optinfo$conv$opt == 0) &&
      (length(m@optinfo$conv$lme4) == 0) && (length(m@beta) == 4)
    vc <- lme4::VarCorr(m)$id; sd <- attr(vc, "stddev"); cr <- attr(vc, "correlation"); b <- m@beta
    est <- c(beta0=b[1], beta1=b[2], beta2=b[3], beta3=b[4], tau0=sd[[1]],
             tau1 = if (cell$slope) sd[[2]] else NA_real_,
             rho01 = if (cell$slope) cr[1,2] else NA_real_)
  }
  mk_row(i, est, good, err, secs, sing)
}

for (cell in CELLS) {
  if (length(ACC_ONLY) && !(cell$id %in% ACC_ONLY)) next
  cat(cell$id, "\n"); dl <- load_cell(cell)
  report_arm(cell, "lme4_LA", append_fits(file.path(OUT, paste0(cell$id, "__lme4_LA.csv")),
             dl, ACC_UPTO, fit_rep(cell, 1L)))
  if (!cell$slope)
    report_arm(cell, "lme4_AGQ", append_fits(file.path(OUT, paste0(cell$id, "__lme4_AGQ.csv")),
               dl, ACC_UPTO, fit_rep(cell, NAGQ_AGQ)))
}
cat("lme4 done (upto", ACC_UPTO, ")\n")
