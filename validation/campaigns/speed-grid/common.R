#!/usr/bin/env Rscript
# Shared readers and the one agreement band the speed-grid analysis scripts use.
# Sourced by analyze.R and counters.R; nothing else reads it.
#
# JSONL records and manifest cells are keyed on `case_id`, which is this
# campaign's cell-id field. The accuracy grid keys the same two shapes on `cell`,
# so its helpers are not interchangeable with these.

suppressMessages({ library(jsonlite) })

TOL <- list(
  # Fixed effects, relative. The cross-engine agreement band the whole corpus is
  # read at; this campaign uses it only to label a cell's beta agreement, never
  # to gate.
  beta_rel = 1e-3,
  # Coordinates both engines put at zero, below which a relative difference has
  # no answer to give: |x-y| / max(|x|,|y|) reads exactly 1.0 however small the
  # residue gets.
  near_zero_abs = 1e-3
)

# Torn-line tolerant: the runner's kill -9 watchdog can truncate the final line,
# and every complete record before the tear is still a result.
read_jsonl <- function(path) {
  lines <- readLines(path); lines <- lines[nzchar(lines)]
  recs <- list()
  for (ln in lines) {
    rec <- tryCatch(fromJSON(ln, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(rec)) recs[[length(recs) + 1L]] <- rec
  }
  setNames(recs, vapply(recs, `[[`, "", "case_id"))
}

# Manifest cells keyed by case_id. simplifyDataFrame = FALSE keeps `cells` a list
# of per-cell lists, so `cell$family` reads off one cell; the default would
# collapse the array into a data.frame.
manifest_cells <- function(path) {
  m <- fromJSON(path, simplifyDataFrame = FALSE)
  setNames(m$cells, vapply(m$cells, `[[`, "", "case_id"))
}

# Relative difference against the larger magnitude, with the denominator floored
# at 1e-12 so a pair of exact zeros is 0 rather than NaN.
rel_max <- function(x, y, atol = TOL$near_zero_abs) {
  if (length(x) != length(y)) return(NA_real_)
  s <- pmax(abs(x), abs(y), 1e-12)
  max(ifelse(s <= atol, 0, abs(x - y) / s))
}
