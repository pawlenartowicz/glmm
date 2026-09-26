# Shared cell loop, record shape and JSONL writer for the accuracy grid's R
# engines. lme4.R, glmmtmb.R, glmmadaptive.R and glmm_r.R all source this file,
# so the record every one of them emits is built by one piece of code and the
# four cannot drift apart field by field.
#
# The environment contract, the same five variables every engine of this grid
# reads (the Rust and Julia engines cannot share this code, so the five are
# mirrored there -- change together):
#   GRID_DIR       the grid/ directory; every cell's `data` path is relative to it
#   GRID_MANIFEST  the manifest to read (default <GRID_DIR>/manifest.json)
#   GRID_OUT       the JSONL file to append to (default <GRID_DIR>/results.jsonl)
#   GRID_CELLS     comma-separated cell ids this worker must fit; empty = all
#   GRID_TIMED     "" or "0" = untimed, else the sample count, an integer >= 2

suppressMessages(library(jsonlite))

# Reads the five environment variables and nothing else. No budget field: no
# engine caps its own fits, the run harness's watchdog is the only cell cap and
# it works by killing the process from outside.
grid_env <- function() {
  # Every engine script lives in grid/engines/, so grid/ is the parent of the
  # running script's directory -- the same default the Rust engine takes from
  # its own source location, so an engine can be run by hand with no variables.
  script <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
  default_dir <- normalizePath(file.path(dirname(script[1]), ".."))
  grid_dir <- grid_getenv("GRID_DIR", default_dir)
  cells <- strsplit(grid_getenv("GRID_CELLS", ""), ",")[[1]]
  list(grid_dir = grid_dir,
       manifest_path = grid_getenv("GRID_MANIFEST", file.path(grid_dir, "manifest.json")),
       out_path = grid_getenv("GRID_OUT", file.path(grid_dir, "results.jsonl")),
       cells = if (length(cells[nzchar(cells)]) == 0) NULL else cells[nzchar(cells)],
       timed = grid_timed())
}

# Sys.getenv's own default fires only when the variable is UNSET; a variable
# exported as an empty string (what a shell wrapper does for an unused option)
# comes back as "" and must fall back too.
grid_getenv <- function(name, default) {
  v <- Sys.getenv(name)
  if (nzchar(v)) v else default
}

# Sample count for this run, or NULL when timing is off. Errors rather than
# silently not timing when an engine is run by hand with a malformed value.
grid_timed <- function() {
  v <- trimws(Sys.getenv("GRID_TIMED"))
  if (v == "" || v == "0") return(NULL)
  n <- suppressWarnings(as.integer(v))
  if (is.na(n) || n < 2)
    stop("GRID_TIMED must be 0 or an integer >= 2 (got '", v,
         "'); N=2 keeps 1 sample after the warm-up discard", call. = FALSE)
  n
}

# BLAS threads to 1, in-process. A belt-and-braces mirror of what the run
# harness exports before launching a worker: six engines running at once would
# otherwise each take the whole machine. Set after the BLAS is loaded this has
# no effect on some builds, which is why it is a mirror and not the primary.
grid_threads <- function() {
  Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1")
  invisible(NULL)
}

# The cells this worker still has to fit, in GRID_CELLS's own order (manifest
# order when GRID_CELLS is empty), narrowed by what is already in the output
# file. Fit order has to be the runner's: its watchdog blames a timeout on the
# first cell of GRID_CELLS still missing from the output, so an engine walking
# a different order fits the wrong cell first and the blame lands on the wrong
# one (mirrors run.sh's `next_missing` -- change together). The resume scan is
# what lets a killed worker be relaunched on the same output without refitting
# or duplicating: a kill -9 can truncate the final line, so a line that does
# not parse is skipped rather than fatal.
grid_cells <- function(env) {
  manifest <- jsonlite::fromJSON(env$manifest_path, simplifyDataFrame = FALSE)
  done <- character(0)
  if (file.exists(env$out_path)) {
    lines <- readLines(env$out_path, warn = FALSE)
    done <- unlist(lapply(lines[nzchar(lines)], function(l)
      tryCatch(jsonlite::fromJSON(l)$cell, error = function(e) NULL)))
  }
  by_id <- stats::setNames(manifest$cells,
                            vapply(manifest$cells, function(c) c[["cell"]], character(1)))
  ids <- if (is.null(env$cells)) names(by_id) else env$cells
  unknown <- setdiff(ids, names(by_id))
  if (length(unknown))
    stop("unknown cell id(s) not in manifest: ", paste(unknown, collapse = ", "), call. = FALSE)
  Filter(function(c) !(c[["cell"]] %in% done), by_id[ids])
}

# One cell's data frame. The manifest's `data` field is relative to grid/, so a
# generated cell (data/<id>.csv) and a crate fixture (../data/simulated/<name>.csv)
# are read by the same line and the manifest is the only place that decides which.
#
# Factor coercion is not cosmetic: the CSV round trip loses it, and numeric-looking
# levels come back as integers. The sorted-level default fixes the treatment
# contrast base at the first sorted level, which is what the manifest's
# `truth.coef_names` and the other engines' coding are aligned to.
#
# `prop` on a trials cell (manifest `weights`) is the proportion response that
# the Julia and glmm formula dialects put on the left-hand side. The R dialect
# does not: `r_formula` carries the trial counts in its own cbind() response, so
# none of the R engines built on this helper reads the column. It is part of
# this helper's output all the same, because the engines whose formulas DO need
# it build their own data frames and synthesize it themselves -- keeping the
# expression here means the one definition of `prop` stays visible next to the
# `weights` field it comes from.
#
# EVERY manifest key is read with [[ ]] and never with $, here and in each
# engine script. `$` on a list partial-matches, and the manifest has two keys
# where that silently returns the wrong one: no cell carries both `weights`
# (trial counts) and `weights_col` (prior weights), so on a prior-weight cell
# `cell$weights` returns the prior-weight column name and this function would
# divide the response by it.
grid_read_cell <- function(env, cell) {
  df <- read.csv(file.path(env$grid_dir, cell[["data"]]), stringsAsFactors = FALSE)
  for (f in unlist(cell[["factors"]])) df[[f]] <- factor(df[[f]])
  trials <- cell[["weights"]]
  if (!is.null(trials)) df$prop <- df[[cell[["response"]]]] / df[[trials]]
  df
}

# Append mode, so a relaunched worker adds to what the resume scan just read.
grid_open_out <- function(env) {
  dir.create(dirname(env$out_path), showWarnings = FALSE, recursive = TRUE)
  file(env$out_path, open = "a")
}

# One place the record is seeded, so no engine can silently drop a field: an
# engine that never reaches a field emits null rather than an absent key, and
# the comparator never has to guess whether a missing key means "no value" or
# "this engine forgot".
#
# se_rx and se_hessian are the two keys that are legitimately ABSENT where an
# engine has none, so they are NOT seeded here; a script that has them assigns
# them, and a script that does not leaves them out.
grid_record <- function(cell, engine, engine_version) {
  list(cell = cell[["cell"]], engine = engine, engine_version = engine_version,
       converged = FALSE, singular = FALSE, status = "engine-fail",
       message = NULL, coef_names = I(character(0)), beta = I(numeric(0)),
       varcomp = list(), sigma = NULL, nb_theta = NULL,
       loglik = NULL, deviance = NULL, n_eval = NULL,
       wall_seconds = NULL, fits_per_sample = 1L)
}

# What a caught error leaves behind. Grid corners are expected to break engines;
# a failure is a data point, not a reason to lose the rest of the run.
grid_fail <- function(rec, msg) {
  rec$status <- "engine-fail"
  rec$converged <- FALSE
  rec$message <- msg
  rec
}

# Timing protocol, mirrored in every engine of this grid: GRID_TIMED unset means
# one untimed call; otherwise it IS the sample count, an integer >= 2, the first
# sample discarded, median of the rest. The untimed branch takes NO timestamp --
# an untimed run reports wall_seconds = NULL, so a clock read there would be a
# value nothing consumes.
grid_time <- function(env, f) {
  if (is.null(env$timed)) {
    return(list(value = f(), wall_seconds = NULL, fits_per_sample = 1L))
  }
  walls <- numeric(env$timed)
  v <- NULL
  for (i in seq_len(env$timed)) {
    t0 <- Sys.time()
    v <- f()
    walls[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  list(value = v, wall_seconds = stats::median(walls[-1]), fits_per_sample = 1L)
}

# loglik and deviance, written TOGETHER and nowhere else, so the two can never
# disagree on one engine and agree on another. deviance is -2*loglik before any
# deviance-alignment correction.
#
# `rec[["x"]] <- list(NULL)` and not `rec$x <- NULL`: the second DELETES the
# element, which would drop the two keys from the record exactly on the
# non-finite fits that most need them present.
grid_set_loglik <- function(rec, ll) {
  ok <- is.numeric(ll) && length(ll) == 1L && is.finite(ll)
  rec["loglik"] <- if (ok) as.numeric(ll) else list(NULL)
  rec["deviance"] <- if (ok) -2 * as.numeric(ll) else list(NULL)
  rec
}

# THE JSONL WRITER. All four keyword arguments are load-bearing:
#   auto_unbox     scalars are scalars, not length-1 arrays
#   digits = I(17) every double comes back bit-identical -- see below
#   na = "null"    an NA scalar writes `null`
#   null = "null"  an R NULL element writes `null`, NOT jsonlite's DEFAULT `{}`.
#
# digits = I(n) is jsonlite's SIGNIFICANT-digit form, and 17 is the number of
# significant decimal digits that round-trips every IEEE-754 double. jsonlite's
# `digits = NA` reads as "maximum precision" but writes 15, which silently
# rounds: at NA, 0.1 + 0.2 writes `0.3` and pi writes `3.14159265358979`, and
# neither reads back equal to the double that went in. These records are an
# oracle and are compared at the 1e-12 level, so the writer must not be the
# thing that sets the resolution.
#
# `null = "null"` is the argument that is easy to omit and impossible to see in
# a jq has() check: with the default, `"sigma": {}` satisfies has("sigma") while
# every downstream is.null() test reads it as a non-null empty list, and the
# deviance-alignment guard that asks `is.null(ll) || !is.finite(ll)` silently
# stops working.
#
# Flushed per line because the run harness's watchdog watches the file's mtime
# to tell a slow fit from a hung one.
grid_write <- function(con, rec) {
  writeLines(jsonlite::toJSON(rec, auto_unbox = TRUE, digits = I(17),
                              na = "null", null = "null"), con)
  flush(con)
}
