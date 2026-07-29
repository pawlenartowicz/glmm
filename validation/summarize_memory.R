#!/usr/bin/env Rscript
# MEMORY summary of the memory.sh legs -- pure REPORTING, no pass/fail gate (there
# is no accuracy tolerance for a peak-RSS number). Shares the read/format helper
# CONVENTION of summarize_timing.R/summarize_accuracy.R/summarize_parallel.R (each
# duplicates its own read_engine/short_name helpers rather than factoring them into
# a shared file -- this one does the same, change together only in spirit, not by
# extracting a common module).
#
# Input is different from the other three summarizers: they read one JSON per
# dataset from results/<engine>_{empirical,simulated}/; this one reads
# results/memory/<leg>.tsv (engine, dataset, rung, n, levels, peak_rss_kb) --
# validation/memory/memory.sh's output, one row per (engine, dataset) it fit in a
# fresh process under `/usr/bin/time -f '%M'`.
#
#   Rscript summarize_memory.R [leg]
#     Defaults to after-alloc (the final 0.1.3 leg: all three glmm engines
#     present, oracles backfilled across all 13 large models). Pass a leg name
#     to render an earlier one (e.g. after-phase1).
#
# One view: CROSS-ENGINE, LARGE MODELS ONLY (the 43 manifest rungs are dropped --
# their peak RSS is dominated by runtime start-up, which is exactly what this view
# is built to exclude, so they'd only show noise). Per dataset x engine, values
# are (peak_rss - engine's load-only baseline) in MB, i.e. fit cost with each
# runtime's own start-up cost subtracted out -- read from
# results/memory/baselines.tsv (memory.sh's `baselines` leg; this script fails
# with a clear message if that file is missing). lme4/MixedModels come from
# results/memory/oracles.tsv (collected once by memory.sh, never per leg); a "-"
# cell means that engine has no measurement there (for mmjl: large_8/9/10 are its
# own documented fit failures), not a zero.
#
# No clock lock needed (memory.sh's header) -- unlike summarize_timing.R/
# summarize_parallel.R, this file has nothing to caveat there.

suite_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

read_leg <- function(leg) {
  path <- file.path(suite_dir, "results", "memory", paste0(leg, ".tsv"))
  if (!file.exists(path)) return(NULL)
  read.delim(path, stringsAsFactors = FALSE)
}

read_oracles <- function() {
  path <- file.path(suite_dir, "results", "memory", "oracles.tsv")
  if (!file.exists(path)) return(NULL)
  read.delim(path, stringsAsFactors = FALSE)
}

# engine -> baseline_kb (memory.sh's `baselines` leg -- one load-only process
# per engine). Required: without it there is no way to separate fit cost from
# runtime start-up, so fail loudly rather than silently falling back to raw
# peak RSS.
read_baselines <- function() {
  path <- file.path(suite_dir, "results", "memory", "baselines.tsv")
  if (!file.exists(path)) {
    stop(sprintf(
      "results/memory/baselines.tsv not found -- run `memory.sh baselines` first (see memory.sh header)"))
  }
  df <- read.delim(path, stringsAsFactors = FALSE)
  setNames(df$baseline_kb, df$engine)
}

is_large_rung <- function(rung) grepl("^L", rung)

short_name <- function(n) {
  n <- sub("^sim_", "s_", n)
  n <- gsub("binomial", "binom", n, fixed = TRUE)
  n <- gsub("crossed", "cross", n, fixed = TRUE)
  n <- gsub("balanced", "balan", n, fixed = TRUE)
  n
}

fmt_mb <- function(kb) if (is.null(kb) || is.na(kb)) "-" else sprintf("%.1f", kb / 1024)

# One row per dataset x engine -> nested lookup engine -> dataset -> peak_rss_kb,
# the same engine-keyed shape read_engine() builds in the other three summarizers
# (there from JSON fields, here from TSV rows).
kb_by_engine <- function(df) {
  if (is.null(df)) return(list())
  out <- list()
  for (e in unique(df$engine)) {
    sub <- df[df$engine == e, ]
    out[[e]] <- setNames(sub$peak_rss_kb, sub$dataset)
  }
  out
}

# Dataset metadata (rung/n/levels) is engine-independent (same CSV every time),
# so one row per dataset suffices regardless of how many engines measured it.
meta_of <- function(df) {
  if (is.null(df)) return(NULL)
  m <- df[!duplicated(df$dataset), c("dataset", "rung", "n", "levels")]
  rownames(m) <- m$dataset
  m
}

args <- commandArgs(TRUE)
leg_name <- if (length(args) >= 1) args[1] else "after-alloc"

leg <- read_leg(leg_name)
oracles <- read_oracles()

if (is.null(leg)) {
  stop(sprintf("results/memory/%s.tsv not found -- run memory.sh first", leg_name))
}

ENGINE_LABEL <- c(rust = "glmm", py = "glmm_python", glmm_r = "glmm_r",
                   lme4 = "lme4", mixedmodels = "mmjl")
# memory.sh's TSV engine column ("py") vs baselines.tsv's engine column
# ("glmm_python") -- the two legs were named independently (baselines.tsv's
# names mirror memory.sh's --engines flag values elsewhere in the file); this
# maps a TSV engine key to its baselines.tsv row.
BASELINE_KEY <- c(rust = "rust", py = "glmm_python", glmm_r = "glmm_r",
                   lme4 = "lme4", mixedmodels = "mixedmodels")

meta <- meta_of(leg)
if (!is.null(oracles)) {
  om <- meta_of(oracles)
  meta <- rbind(meta, om[setdiff(rownames(om), rownames(meta)), , drop = FALSE])
}
kb_all <- modifyList(kb_by_engine(oracles), kb_by_engine(leg)) # leg wins on any engine name clash
baselines <- read_baselines()

# net_kb: peak_rss - engine's load-only baseline. NA propagates through (no
# reading, no fit, or no baseline row for that engine) rather than being
# silently treated as zero.
net_kb <- function(e, ds) {
  raw <- kb_all[[e]][ds]
  # Single-bracket: `[[` on an absent name throws instead of yielding NA, which
  # would make the is.na guard below dead code and abort the whole render for an
  # engine that has measurement rows but no baselines.tsv row.
  base <- baselines[BASELINE_KEY[[e]]]
  if (is.null(raw) || is.na(raw) || is.null(base) || is.na(base)) NA_real_ else raw - base
}

# LARGE MODELS ONLY -- see header note.
order_names <- rownames(meta)[is_large_rung(meta$rung)]
order_names <- order_names[order(suppressWarnings(as.numeric(sub("^L", "", meta[order_names, "rung"]))))]

COLS <- Filter(function(e) e %in% names(kb_all), c("rust", "py", "glmm_r", "lme4", "mixedmodels"))

cat("== memory (peak RSS minus engine baseline, MB) --", leg_name, "leg", if (!is.null(oracles)) "+ oracles.tsv" else "", "-- large models only ==\n")
cat("Values are peak RSS with each engine's load-only start-up baseline\n",
    "subtracted (results/memory/baselines.tsv, from `memory.sh baselines`) --\n",
    "this reads as fit cost, not runtime footprint. A \"-\" cell means no\n",
    "measurement for that engine (mmjl's large_8/9/10 are its own fit failures).\n\n")
name_w <- max(nchar(order_names)) + 1L
lead <- function(name, rung, n, levels) sprintf(paste0("%-", name_w, "s %-4s %10s %8s"), name, rung, n, levels)
header <- paste0(lead("dataset", "rung", "n", "levels"))
for (e in COLS) header <- paste0(header, sprintf(" %11s", ENGINE_LABEL[[e]]))
cat(header, "\n")
cat(strrep("-", nchar(header)), "\n")
for (ds in order_names) {
  m <- meta[ds, ]
  row <- lead(short_name(ds), m$rung, m$n, m$levels)
  for (e in COLS) row <- paste0(row, sprintf(" %11s", fmt_mb(net_kb(e, ds))))
  cat(row, "\n")
}
