#!/usr/bin/env Rscript
# TIMING summary of the accuracy grid -- pure REPORTING. The pass/fail gate
# lives in compare.R; the accuracy views live in summarize_accuracy.R (shares
# runs.R's run-discovery vocabulary -- change together).
#
# Reads TIMED runs only (run_meta.json's `timed` is non-null). Prints
# provenance per engine FIRST, always -- seconds do not transfer across
# machines (only ratios do, and only weakly), so this block exists to stop
# two boxes' rows from being silently read as one comparison. If the runs it
# found span more than one machine, it prints their provenance and REFUSES
# the seconds table rather than guess which one to keep.
#
# Then: median seconds per fit (the grid's timing protocol already reduces
# each cell to one number -- N samples, first discarded, median of the rest --
# so there is no further reduction to do here), and glmm's speedup factor
# against every oracle that has a timed run. An unlocked-clock run is never
# dropped -- its numbers are still shown, with a loud warning naming it, both
# in the provenance block and right above the table.
#
# NEVER FAILS THE BUILD: the whole body runs inside one tryCatch that always
# ends in quit(status = 0).

suppressMessages(library(jsonlite))

opt_glmm_run <- NULL
parse_args <- function() {
  for (a in commandArgs(TRUE)) {
    if (startsWith(a, "--glmm-run=")) opt_glmm_run <<- sub("^--glmm-run=", "", a)
    else stop(sprintf("unknown argument `%s`. Usage: summarize_timing.R [--glmm-run=DIR]", a),
              call. = FALSE)
  }
}

safe <- function(expr, what) tryCatch(expr, error = function(e) {
  cat(sprintf("  (%s unavailable: %s)\n", what, conditionMessage(e))); NULL })

# runs.R answers "the newest UNTIMED subset:full run" (the accuracy
# reference) and deliberately excludes timed ones. Timing needs the
# opposite question -- the newest TIMED run for one engine directory -- which
# is not a rewrite of newest_oracle_run/newest_glmm_run, just a different one
# built the same way (full+scratch search, newest date first) and reusing
# their `read_run`.
newest_timed_run <- function(grid_dir, dir_name) {
  root <- file.path(grid_dir, "runs", dir_name)
  dirs <- c(list.dirs(root, recursive = FALSE),
            list.dirs(file.path(root, "scratch"), recursive = FALSE))
  dirs <- Filter(function(d) file.exists(file.path(d, "run_meta.json")) &&
                             file.exists(file.path(d, "results.jsonl")), dirs)
  ok <- list()
  for (d in dirs) {
    m <- jsonlite::fromJSON(file.path(d, "run_meta.json"), simplifyVector = TRUE)
    if (is.null(m$timed) || is.na(m$timed)) next
    ok[[length(ok) + 1L]] <- list(dir = d, date = m$date)
  }
  if (!length(ok)) return(NULL)
  ok <- ok[order(vapply(ok, `[[`, "", "date"), decreasing = TRUE)]
  read_run(ok[[1]]$dir)
}

LABEL <- c(glmm = "glmm", lme4 = "lme4", glmmTMB = "glmmTMB",
           GLMMadaptive = "MA", MixedModels = "mmjl")

fmt_t <- function(x) if (is.null(x) || is.na(x)) "-" else sprintf("%.6f", x)
fmt_x <- function(other, mine) if (is.na(other) || is.na(mine) || mine == 0) "-" else sprintf("%.1fx", other / mine)

# A timed run is UNLOCKED when run.sh's own no_turbo probe (recorded verbatim
# in run_meta.json) read anything but 1, or the run was explicitly launched
# with --timed-unlocked (timed_unlocked: true). Either way its seconds are
# powersave noise, not measurements, and every place that shows them says so.
is_unlocked <- function(m) isTRUE(m$timed_unlocked) || !identical(as.character(m$no_turbo), "1")

main <- function() {
  glmm_run <- safe(if (!is.null(opt_glmm_run)) read_run(opt_glmm_run)
                    else newest_timed_run(GRID, "glmm"), "glmm")
  # One column per ORACLE that actually has a timed run -- whichever of the
  # four are present -- never a fixed engine list and never a port
  # (glmm_python/glmm_r) column: a port's time is the same kernel plus FFI
  # tax, not a speed comparison this table is answering.
  oracle_runs <- setNames(lapply(ORACLES, function(o)
    safe(newest_timed_run(GRID, ENGINE_DIR[[o]]), o)), ORACLES)
  runs <- c(list(glmm = glmm_run), oracle_runs)
  # "Timed run" means run_meta.json$timed is set -- a run object that slipped
  # through with timed == NULL (e.g. a bad --glmm-run= override) is dropped
  # here rather than trusted.
  runs <- Filter(function(r) !is.null(r) && !is.null(r$meta$timed) && !is.na(r$meta$timed), runs)

  cat("== timing provenance ==\n")
  if (!length(runs)) {
    cat("  no timed run present\n")
    return(invisible(NULL))
  }
  for (e in names(runs)) {
    m <- runs[[e]]$meta
    cat(sprintf("  %-6s %-30s no_turbo=%-2s pin=%-8s load=%-16s %s  %s  timed=%s\n",
                LABEL[[e]], m$machine, m$no_turbo, m$pin_cores, m$loadavg_start,
                substr(m$glmm_git_rev, 1, 8), m$date, m$timed))
    if (is_unlocked(m)) {
      cat(sprintf("  WARNING: clock NOT locked on %s (no_turbo=%s) -- its numbers below are noise, not measurements\n",
                  LABEL[[e]], m$no_turbo))
    }
  }
  cat("\n")

  machines <- unique(vapply(runs, function(r) r$meta$machine, ""))
  if (length(machines) > 1) {
    cat(sprintf("REFUSING the seconds table: %d different machines in this set of runs (%s).\n",
                length(machines), paste(machines, collapse = " | ")))
    cat("Seconds do not transfer across machines (only ratios do, and only weakly).\n")
    return(invisible(NULL))
  }

  if (!("glmm" %in% names(runs))) {
    cat("no glmm timed run -- nothing to report a speedup against\n")
    return(invisible(NULL))
  }

  cat("NOTE: glmm's wall_seconds times the fit only (model-matrix lowering excluded);\n",
      "the R/Julia engines time the whole fit call, lowering included.\n", sep = "")

  cells_all <- grid_cells_by_id(file.path(GRID, "manifest.json"))
  cids <- intersect(names(cells_all), names(runs[["glmm"]]$recs))
  if (!length(cids)) {
    cat("\nno cell in the glmm timed run is a known manifest cell\n")
    return(invisible(NULL))
  }
  ord <- cids[order(vapply(cells_all[cids], `[[`, "", "family"), cids)]

  # Every engine with a timed run gets a column -- unlocked ones included,
  # per the warning already printed above. `runs`'s own key order (glmm, then
  # ORACLES order) is what both this list and the header below follow.
  timing_engines <- names(runs)
  speedup_vs <- setdiff(timing_engines, "glmm")
  unlocked_here <- Filter(function(e) is_unlocked(runs[[e]]$meta), timing_engines)
  if (length(unlocked_here)) {
    for (e in unlocked_here)
      cat(sprintf("WARNING: %s's seconds below are UNLOCKED-CLOCK numbers -- treat as noise, not measurements.\n",
                  LABEL[[e]]))
  }

  cat("\n== median seconds per fit (fast subset) ==\n")
  header <- sprintf("%-34s %-17s", "cell", "family")
  for (e in timing_engines) header <- paste0(header, sprintf(" %9s", LABEL[[e]]))
  for (e in speedup_vs) header <- paste0(header, sprintf(" %10s", paste0("vs_", LABEL[[e]])))
  cat(header, "\n")

  for (cid in ord) {
    cl <- cells_all[[cid]]
    # fits_per_sample is always 1 in the grid; dividing by it keeps
    # this column honestly "seconds per fit" even if that ever changes.
    secs <- setNames(vapply(timing_engines, function(e) {
      r <- runs[[e]]$recs[[cid]]
      if (is.null(r) || is.null(r$wall_seconds)) NA_real_ else r$wall_seconds / r$fits_per_sample
    }, 0), timing_engines)
    row <- sprintf("%-34s %-17s", cid, cl$family)
    for (e in timing_engines) row <- paste0(row, sprintf(" %9s", fmt_t(secs[[e]])))
    for (e in speedup_vs) row <- paste0(row, sprintf(" %10s", fmt_x(secs[[e]], secs[["glmm"]])))
    cat(row, "\n")
  }
}

# GRID is this script's own directory, resolved from `--file=` exactly as
# compare.R resolves it, so a throwaway grid tree that symlinks the scripts
# and holds its own runs/ is reported on as itself. The whole body runs
# inside one tryCatch so a missing runs.R, a bad flag or any unexpected error
# still exits 0 -- a report script must never fail a build.
tryCatch({
  GRID <- normalizePath(dirname(sub(
    "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
  source(file.path(GRID, "runs.R"))
  parse_args()
  main()
}, error = function(e) {
  cat("ERROR (report continues, exit stays 0): ", conditionMessage(e), "\n", sep = "")
})
quit(status = 0)
