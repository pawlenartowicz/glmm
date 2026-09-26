#!/usr/bin/env Rscript
# manifest.json -> grid/data/<cell>.csv. Seeded per cell, so re-running writes
# byte-identical files: the seed lives in the manifest (the committed artifact),
# the CSVs are derived and gitignored -- except data/fast/, which is committed
# because CI reads it.
#
# Cells whose `data` field points OUTSIDE grid/ (../data/empirical/*.csv,
# ../data/simulated/*.csv) are the crate's own test fixtures. They are read in
# place and NEVER written here: data/ is read by include_str! from ten source
# files and by CI on every push, so regenerating one would move numbers nothing
# in this campaign is allowed to move.
#
#   Rscript gen.R              every generated cell
#   Rscript gen.R --fast       only the cells tagged `fast` (into data/fast/)
#   Rscript gen.R <cell> ...   only the named cells
suppressMessages({ library(jsonlite); library(MASS) })
here <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "gen_common.R"))

args <- commandArgs(trailingOnly = TRUE)
only_fast <- "--fast" %in% args
only <- setdiff(args, "--fast")

man <- fromJSON(file.path(here, "manifest.json"), simplifyDataFrame = FALSE)
cells <- man$cells
cells <- Filter(function(c) !is.null(c$seed) && startsWith(c$data, "data/"), cells)
if (only_fast) cells <- Filter(function(c) "fast" %in% c$tags, cells)
if (length(only)) {
  unknown <- setdiff(only, vapply(cells, `[[`, "", "cell"))
  if (length(unknown))
    stop("unknown or non-generated cell(s): ", paste(unknown, collapse = ", "))
  cells <- Filter(function(c) c$cell %in% only, cells)
}

dir.create(file.path(here, "data"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(here, "data", "fast"), showWarnings = FALSE, recursive = TRUE)

n_written <- 0L
for (cell in cells) {
  meta <- sim_cell(cell)
  # `data` already carries the correct relative path (data/<id>.csv or
  # data/fast/<id>.csv) -- gen_manifest.R owns that choice, not this file, so a
  # cell cannot be written to one place and read from another. The seed lives in
  # the cell and is seed_of(cell$cell), so this loop's ORDER is irrelevant to the
  # bytes it writes.
  out <- file.path(here, cell$data)
  dir.create(dirname(out), showWarnings = FALSE, recursive = TRUE)
  write.csv(meta$df, out, row.names = FALSE)
  # The manifest's `factors` and `n_x` are what every engine lowers from; a
  # simulation that produced different ones means gen_common.R and
  # gen_manifest.R have drifted apart.
  # `unlist(list())` is NULL, so an empty manifest `factors` (every GLM cell)
  # needs the explicit as.character() on both sides to compare as character(0).
  stopifnot("factors drifted from the manifest" =
    identical(as.character(unlist(cell$factors)), as.character(meta$factors)))
  # jsonlite parses a whole-number JSON field as integer, while nx_of()'s mixed
  # integer/double arithmetic returns a double for the same value, so both
  # sides need as.integer() to compare as equal rather than as identical types.
  stopifnot("n_x drifted from the manifest" =
    identical(as.integer(cell$n_x), as.integer(meta$n_x)))
  n_written <- n_written + 1L
}
cat(sprintf("wrote %d CSVs under %s\n", n_written, file.path(here, "data")))
