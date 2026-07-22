# GLMMadaptive (the "adaptive GLM" reference) over escalating batches: adaptive
# Gauss-Hermite on all 24 cells, one arm, matching the paper's est_GLMMadaptive
# (estimation_tool.R:191). Fit + extraction and the "good" = model$converged
# check mirror their code. Used here as a live timing reference (its published
# accuracy is already the frozen oracle); accuracy is also recorded as a cross-check.
suppressMessages(library(GLMMadaptive))
source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))
OUT <- file.path(ACC_DIR, "results", "glmmadaptive"); dir.create(OUT, FALSE, TRUE)

fixed_f <- y_full ~ time + group + time:group

fit_rep <- function(cell) function(df, i) {
  est <- NA_EST; good <- FALSE; err <- TRUE; secs <- NA_real_; sing <- NA
  rd <- if (cell$slope) (~ 1 + time | id) else (~ 1 | id)
  tm <- system.time(m <- try(suppressMessages(suppressWarnings(
    mixed_model(fixed = fixed_f, random = rd, family = cell_family(cell$family),
                data = df))), silent = TRUE))["elapsed"]
  if (!inherits(m, "try-error")) {
    err <- FALSE; secs <- tm; good <- isTRUE(m$converged)
    cf <- m$coefficients; D <- m$D
    t0 <- sqrt(D["(Intercept)", "(Intercept)"])
    t1 <- if (cell$slope) sqrt(D["time", "time"]) else NA_real_
    r01 <- if (cell$slope) D["(Intercept)","time"]/(t0*t1) else NA_real_
    est <- c(beta0=cf[["(Intercept)"]], beta1=cf[["time"]], beta2=cf[["group"]],
             beta3=cf[["time:group"]], tau0=t0, tau1=t1, rho01=r01)
    # GLMMadaptive has no isSingular; boundary = a variance ~0 or |corr| ~1.
    sing <- (is.finite(t0) && t0 < 1e-3) ||
      (cell$slope && ((is.finite(t1) && t1 < 1e-3) || !is.finite(r01) || abs(r01) > 0.99))
  }
  mk_row(i, est, good, err, secs, sing)
}

for (cell in CELLS) {
  if (length(ACC_ONLY) && !(cell$id %in% ACC_ONLY)) next
  cat(cell$id, "\n"); dl <- load_cell(cell)
  report_arm(cell, "GLMMadaptive", append_fits(file.path(OUT, paste0(cell$id, "__GLMMadaptive.csv")),
             dl, ACC_UPTO, fit_rep(cell)))
}
cat("GLMMadaptive done (upto", ACC_UPTO, ")\n")
