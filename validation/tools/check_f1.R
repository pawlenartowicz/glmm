#!/usr/bin/env Rscript
# The R port against F1 (δ fit + CR2) as the
# earlier simulation study stored it, on the rows of
# validation/goldens/inference/cr2/f1_reps.json (written by goldens_cr2_cells.R:
# the first 20 replications of the 10 f2_cells, two responses each).
#
# THE ORACLE IS SACRED. On a gap, fastglmm is presumed wrong: never relax a
# tolerance or edit the reference.
#
# Per response, model y ~ x + (1 | g) with weights w, coefficient x:
#   1. at the stored δ̂ (weight_power = δ̂, se_type = "robust"): estimate, SE and
#      df within 1e-6 relative of the stored ones;
#   2. by default (the δ fit): δ̂ within 1e-4 of the stored one (golden section
#      and the simulation's Brent end up to 3.9e-5 apart).
# Run at the gate, after installing the package:
#   R CMD INSTALL -l $RLIB glmm-r
#   R_LIBS=$RLIB:$RORACLE Rscript validation/tools/check_f1.R
# Prints the worst gap per quantity with its case and exits 1 if a gate fails.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
suppressPackageStartupMessages({
  library(fastglmm)
  library(jsonlite)
})
ref <- fromJSON(file.path(suite_dir, "goldens", "inference", "cr2", "f1_reps.json"),
                simplifyVector = FALSE)

REL <- 1e-6
DELTA_ABS <- 1e-4

rows <- list()
for (cell in ref$cells) {
  d <- data.frame(g = factor(unlist(cell$g)), x = unlist(cell$x), w = unlist(cell$w))
  for (r in cell$reps) for (sfx in c("_0", "_1")) {
    d$y <- unlist(r[[paste0("y", sfx)]])
    delta <- r[[paste0("delta_hat", sfx)]]
    # The notes (a robust df below 4, say) are not what this checks.
    at <- suppressWarnings(
      fastglmm(y ~ x + (1 | g), d, weights = w, weight_power = delta, se_type = "robust"))
    by_default <- suppressWarnings(fastglmm(y ~ x + (1 | g), d, weights = w))
    stopifnot(at$se_type == "robust", isTRUE(by_default$estimated_weight_power))
    rows[[length(rows) + 1L]] <- data.frame(
      case = sprintf("%s rep %d y%s", cell$cell, r$rep, sfx),
      est = abs(at$beta[["x"]] / r[[paste0("est", sfx)]] - 1),
      se = abs(at$se[["x"]] / r[[paste0("se", sfx)]] - 1),
      df = abs(at$coef_df[[match("x", names(at$beta))]] / r[[paste0("df", sfx)]] - 1),
      delta = abs(by_default$weight_power - delta))
  }
}
gaps <- do.call(rbind, rows)

gates <- c(est = REL, se = REL, df = REL, delta = DELTA_ABS)
failed <- FALSE
cat(sprintf("G8: %d responses (%d cells)\n", nrow(gaps), length(ref$cells)))
for (q in names(gates)) {
  worst <- which.max(gaps[[q]])
  over <- sum(!(gaps[[q]] <= gates[[q]]))
  cat(sprintf("%-5s worst %.3g (%s), limit %g %s, over: %d  %s\n", q, gaps[[q]][worst],
              gaps$case[worst], gates[[q]], if (q == "delta") "abs" else "rel", over,
              if (over) "FAIL" else "pass"))
  if (over) {
    failed <- TRUE
    bad <- gaps[!(gaps[[q]] <= gates[[q]]), c("case", q)]
    print(bad[order(-bad[[q]]), ][seq_len(min(10L, nrow(bad))), ], row.names = FALSE)
  }
}
if (failed) quit(status = 1L, save = "no")
