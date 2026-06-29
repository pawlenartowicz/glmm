#!/usr/bin/env Rscript
# Export each lme4-origin parity dataset to data/<name>.csv — run ONCE; the CSVs
# are committed and are the neutral input EVERY engine reads (R, Julia, later Rust).
# Exporting from one canonical source (lme4 in R) is what guarantees byte-identical
# input across engines and sidesteps row-order / factor-coding / NA differences
# between the ecosystems' built-in copies. Ordinary parity runs never call this.

suppressMessages({
  library(lme4)
  library(jsonlite)
})

parity_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

manifest <- fromJSON(file.path(parity_dir, "manifest.json"), simplifyDataFrame = FALSE)
data_dir <- file.path(parity_dir, "data")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

for (spec in manifest$datasets) {
  data(list = spec$name, package = "lme4")
  df <- get(spec$name)
  out <- file.path(data_dir, paste0(spec$name, ".csv"))
  # Full dataset, factor labels written verbatim; fit scripts re-coerce the columns
  # named in the manifest `factors` list back to factor / categorical.
  write.csv(df, out, row.names = FALSE)
  cat(sprintf("wrote %-12s  %3d rows x %d cols\n", spec$name, nrow(df), ncol(df)))
}
