#!/usr/bin/env Rscript
# The new data of the `predict` inference references -> one CSV per entry,
# ../../data/inference/<entry>_newdata.csv, read through the entry's `newdata`.
# Run from the repository root:
#   Rscript validation/tools/prep/gen_predict_data.R
# Base R only, and no random draws: every value is read from the entry's fit
# CSV, so a re-run reproduces every file byte for byte.
#
# THE ROWS. Each column the entry's formula reads, except the response, takes
# a few values, and the rows are their cross product (expand.grid: the first
# column varies fastest), the columns in the order of the fit CSV:
#   - a grouping factor: its first two levels in the fit data, then the unseen
#     level `new1`. Two grouping columns cross into every pair (Penicillin:
#     plate x sample; Pastes: batch x cask, so a new cask occurs within a seen
#     batch and a seen cask within the new batch);
#   - a fixed factor: all its levels;
#   - a numeric covariate: its first value in the fit data and
#     max + (max - min) / 2, outside the fitted range;
#   - the offset `log_exposure`: its first value and 0.
# The levels of every factor come in the reverse of their first-appearance
# order in the fit data, which is not their sorted order either, so a lowering
# that reads levels by position instead of by label gets a different design.
# `new1` comes after the seen levels.
#
# Numbers are written with sprintf("%.17g") (tools/prep/gen_illcond_data.R's
# rule), so read.csv recovers the doubles of the fit CSV exactly; the script
# reads each file back and stops unless every column round-trips.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", "..")) # nolint
out_dir <- file.path(suite_dir, "data", "inference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Per entry: its fit CSV and the columns it varies, by role. These mirror the
# formulas of the manifest's `predict` entries; change together.
NEWDATA <- list(
  predict_sleepstudy = list(data = "data/empirical/sleepstudy.csv",
                            grouping = "Subject", numeric = "Days"),
  predict_penicillin = list(data = "data/empirical/Penicillin.csv",
                            grouping = c("plate", "sample")),
  predict_pastes = list(data = "data/empirical/Pastes.csv",
                        grouping = c("batch", "cask")),
  predict_glmm_sim_factorial_bern = list(data = "data/simulated/sim_factorial_bern.csv",
                                         grouping = "subj", fixed = c("g", "h"),
                                         numeric = "x"),
  predict_glmm_sim_poisson_offset = list(data = "data/simulated/sim_poisson_offset.csv",
                                         grouping = "cluster", numeric = "x",
                                         offset = "log_exposure"),
  predict_glmm_sim_gamma = list(data = "data/simulated/sim_gamma.csv",
                                grouping = "cluster", fixed = "grp", numeric = "x"),
  predict_glm_sim_factorial_bern = list(data = "data/simulated/sim_factorial_bern.csv",
                                        fixed = c("g", "h"), numeric = "x"),
  predict_ols_sim_factorial = list(data = "data/simulated/sim_factorial.csv",
                                   fixed = c("g", "h"), numeric = "x")
)

newdata_of <- function(spec) {
  fit <- read.csv(file.path(suite_dir, spec$data), stringsAsFactors = FALSE)
  cols <- c(spec$grouping, spec$fixed, spec$numeric, spec$offset)
  missing <- setdiff(cols, names(fit))
  if (length(missing) > 0L) stop(sprintf("%s has no column %s", spec$data, missing[1]))
  values <- list()
  for (col in names(fit)[names(fit) %in% cols]) {
    v <- fit[[col]]
    values[[col]] <- if (col %in% spec$grouping) {
      c(rev(as.character(unique(v)[1:2])), "new1")
    } else if (col %in% spec$fixed) {
      rev(as.character(unique(v)))
    } else if (col %in% spec$numeric) {
      c(v[1], max(v) + (max(v) - min(v)) / 2)
    } else {
      c(v[1], 0)
    }
  }
  expand.grid(values, stringsAsFactors = FALSE, KEEP.OUT.ATTRS = FALSE)
}

for (name in names(NEWDATA)) {
  nd <- newdata_of(NEWDATA[[name]])
  out <- nd
  is_num <- vapply(nd, is.numeric, NA)
  for (col in names(nd)[is_num]) out[[col]] <- sprintf("%.17g", nd[[col]])
  path <- file.path(out_dir, paste0(name, "_newdata.csv"))
  write.csv(out, path, row.names = FALSE, quote = which(!is_num))
  back <- read.csv(path, stringsAsFactors = FALSE)
  for (col in names(nd)) {
    if (!identical(back[[col]], nd[[col]])) {
      stop(sprintf("%s: column %s does not round-trip through read.csv", name, col))
    }
  }
  cat(sprintf("wrote %-32s %3d rows x %d cols\n", basename(path), nrow(nd), ncol(nd)))
}
