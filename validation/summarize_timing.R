#!/usr/bin/env Rscript
# TIMING summary of the validation results -- pure REPORTING, the pass/fail gate lives 
# in compare.R; the accuracy views live in summarize_accuracy.R (shares the read/format 
# helpers below -- change together).
#
# Per dataset x engine, the median PER-FIT time (rx / hessian split), plus the
# glmm speedup factor vs each oracle.
#
# Timing is indicative only unless the machine was locked when fit.* ran (README).

suppressMessages(library(jsonlite))

suite_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

read_engine <- function(dir_name) {
  files <- unlist(lapply(c("empirical", "simulated"), function(s)
    list.files(file.path(suite_dir, "results", paste0(dir_name, "_", s)),
               pattern = "\\.json$", full.names = TRUE)))
  if (length(files) == 0) return(NULL)
  res <- lapply(files, fromJSON, simplifyVector = TRUE,
                simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
  if (!length(res)) return(NULL)
  setNames(res, vapply(res, `[[`, "", "dataset"))
}

# The design's "visually separated" rule: rungs print in two passes, empirical
# first, then simulated -- the set is which lme4_empirical carries.
empirical_names <- vapply(
  list.files(file.path(suite_dir, "results", "lme4_empirical"), pattern = "\\.json$"),
  function(f) sub("\\.json$", "", f), "")

# Per-method PER-FIT time. The JSON stores the median per SAMPLE, and a sample is
# `fits_per_sample` fits (batched rungs time 10 fits to beat R's timer floor) --
# normalize here so t_rx/t_hess are always seconds per fit. GLMM rungs split
# rx/hessian (the FD-Hessian / numDeriv SE is the main cost); gaussian/legacy
# carry a single median, shown in the rx slot.
#
# CONSTRUCTION-INCLUSIVE by preference: the `_full` fields (glmm only) time
# `lower + fit`, the span lme4 / MixedModels / the Python port all measure
# (formula+data -> model -> fit). Preferring them here puts glmm on the SAME axis
# as the other three engines, so the vs_lme4 / vs_mmjl / vs_py ratios below are
# same-to-same rather than flattering glmm by the lowering it alone hoisted out.
# The fit-only `_median` fields (retained in the JSON for the solver-isolation
# analyses) are the fallback for any engine without a `_full` variant (all of
# lme4 / mmjl / py, whose single time already includes construction).
time_of <- function(r) {
  t <- r$timing
  rx <- if (!is.null(t$fit_seconds_median_rx_full)) t$fit_seconds_median_rx_full
        else if (!is.null(t$fit_seconds_median_full)) t$fit_seconds_median_full
        else if (!is.null(t$fit_seconds_median_rx)) t$fit_seconds_median_rx
        else if (!is.null(t$fit_seconds_median)) t$fit_seconds_median
        else t$fit_seconds_min
  hess <- if (!is.null(t$fit_seconds_median_hessian_full)) t$fit_seconds_median_hessian_full
          else if (!is.null(t$fit_seconds_median_hessian)) t$fit_seconds_median_hessian
          else NA_real_
  fps <- if (!is.null(t$fits_per_sample)) t$fits_per_sample else 1L
  c(rx = rx / fps, hess = hess / fps)
}

fmt_t <- function(x) if (is.null(x) || is.na(x)) "-" else sprintf("%.6f", x)

short_name <- function(n) {
  n <- sub("^sim_", "s_", n)
  n <- gsub("binomial", "binom", n, fixed = TRUE)
  n <- gsub("crossed", "cross", n, fixed = TRUE)
  n <- gsub("balanced", "balan", n, fixed = TRUE)
  n
}

ENGINE_DIRS <- c(lme4 = "lme4", mixedmodels = "mixedmodels", glmm = "glmm",
                 glmm_python = "glmm_python", glmm_r = "glmm_r")
# AGQ pass (VALIDATION_AGQ, engines/lme4.R + engines/glmm.rs): a sibling results
# tree, absent unless that opt-in pass has been run. Read separately from
# ENGINE_DIRS because these are extra ROWS for datasets already in the table, not
# extra engine columns -- and because compare.R must never see them (design 6).
AGQ_DIRS <- c(lme4 = "lme4_agq", glmm = "glmm_agq")
agq <- Filter(Negate(is.null), lapply(AGQ_DIRS, read_engine))
agq <- lapply(agq, function(lst) setNames(lst, short_name(names(lst))))
# Quadrature order per dataset, for the row label -- the manifest's `agq` field is
# what the pass itself read, so the label cannot drift from what was fitted.
AGQ_K <- local({
  ds <- fromJSON(file.path(suite_dir, "manifest.json"),
                 simplifyDataFrame = FALSE)$datasets
  marked <- Filter(function(s) !is.null(s$agq), ds)
  setNames(lapply(marked, `[[`, "agq"), short_name(vapply(marked, `[[`, "", "name")))
})
ENGINE_LABEL <- c(lme4 = "lme4", mixedmodels = "mmjl", glmm = "glmm",
                  glmm_python = "py", glmm_r = "r")
data <- Filter(Negate(is.null), lapply(ENGINE_DIRS, read_engine))
data <- lapply(data, function(lst) setNames(lst, short_name(names(lst))))
present <- names(data)
ref <- data[["lme4"]]
if (is.null(ref)) stop("no lme4 reference present -- run engines/lme4.R first")
order_names <- names(ref)[order(vapply(ref, `[[`, 0L, "rung"))]

TIMING_COLS <- c("glmm", "glmm_python", "glmm_r", "lme4", "mixedmodels")
timing_engines <- Filter(function(e) e %in% present, TIMING_COLS)

# glmm speedup vs lme4/mmjl: how many times faster glmm is (other_time / glmm_time).
fmt_x <- function(other, mine) {
  if (is.na(other) || is.na(mine) || mine == 0) "-" else sprintf("%.1fx", other / mine)
}
# py_gap/r_gap read in the same direction as vs_lme4/vs_mmjl (other/glmm), but each
# port is the same kernel, so it is the port tax (conversion + FFI), not a speedup.
SPEEDUP_VS <- Filter(function(e) e %in% present,
                     c("lme4", "mixedmodels", "glmm_python", "glmm_r"))
# A port column is a tax, not a speedup, so it is headed <lang>_gap rather than vs_<lang>.
PORT_ENGINES <- c("glmm_python", "glmm_r")
speedup_label <- function(e) {
  if (e %in% PORT_ENGINES) paste0(ENGINE_LABEL[[e]], "_gap") else paste0("vs_", ENGINE_LABEL[[e]])
}

# ---- provenance: which box, and was its clock locked ----------------------------
# `run.sh --timings` writes results/run_meta_<engine>.json under ITS engine names,
# which differ from this script's result-dir names -- RUN_META_NAME bridges the two
# and must change together with run.sh's ENGINES loop. Seconds do not transfer
# across machines (only ratios do, and only weakly), so this block exists to stop
# two boxes' rows from being silently read as one comparison.
RUN_META_NAME <- c(glmm = "rust", glmm_python = "py", glmm_r = "glmm_r",
                   lme4 = "lme4", mixedmodels = "jl")
read_run_meta <- function(e) {
  p <- file.path(suite_dir, "results", paste0("run_meta_", RUN_META_NAME[[e]], ".json"))
  if (!file.exists(p)) return(NULL)
  fromJSON(p, simplifyVector = TRUE)
}
metas <- Filter(Negate(is.null),
                setNames(lapply(timing_engines, read_run_meta), timing_engines))
label_of <- function(es) paste(vapply(es, function(e) ENGINE_LABEL[[e]], ""), collapse = ", ")

cat("== timing provenance ==\n")
if (!length(metas)) {
  cat("  no results/run_meta_*.json -- these timings either predate the run_meta\n",
      "  convention or did not come from `run.sh --timings`. Provenance unknown:\n",
      "  do not compare them against timings from any other run.\n", sep = "")
} else {
  for (e in names(metas)) {
    m <- metas[[e]]
    cat(sprintf("  %-6s %-30s no_turbo=%-2s pin=%-13s %s  %s\n", ENGINE_LABEL[[e]],
                m$machine, m$no_turbo, m$pin, substr(m$glmm_git_rev, 1, 8), m$started))
  }
  unlabelled <- setdiff(timing_engines, names(metas))
  if (length(unlabelled))
    cat(sprintf("  WARNING: no run_meta for %s -- provenance unknown, its seconds are uncomparable.\n",
                label_of(unlabelled)))
  boxes <- unique(vapply(metas, function(m) m$machine, ""))
  if (length(boxes) > 1)
    cat(sprintf("  WARNING: %d DIFFERENT MACHINES in one table (%s).\n%s",
                length(boxes), paste(boxes, collapse = " | "),
                "  Seconds AND the vs_/gap columns below are NOT comparable across them.\n"))
  loose <- names(Filter(function(m) !identical(as.character(m$no_turbo), "1"), metas))
  if (length(loose))
    cat(sprintf("  WARNING: clock NOT locked for %s -- powersave noise, not measurements (run bench-l).\n",
                label_of(loose)))
}
cat("\n")

cat("== timing (median seconds per fit) ==\n")
name_w <- max(nchar(order_names)) + 1L
lead_row <- function(name, rung, metric) sprintf(paste0("%-", name_w, "s %-4s %-3s"), name, rung, metric)
header <- lead_row("dataset", "rung", "")
for (e in timing_engines) header <- paste0(header, sprintf(" %9s", ENGINE_LABEL[[e]]))
if ("glmm" %in% present) for (e in SPEEDUP_VS) header <- paste0(header, sprintf(" %7s", speedup_label(e)))
cat(header, "\n")
cat(strrep("-", nchar(header)), "\n")
for (group in c("empirical", "simulated")) {
  cat(sprintf("== %s ==\n", group))
  names_in_group <- if (group == "empirical") empirical_names else setdiff(order_names, empirical_names)
  for (name in intersect(order_names, names_in_group)) {
    a <- ref[[name]]
    tms <- setNames(lapply(timing_engines, function(e) {
      b <- data[[e]][[name]]
      if (is.null(b)) c(rx = NA_real_, hess = NA_real_) else time_of(b)
    }), timing_engines)
    rx_row <- lead_row(name, a$rung, "rx")
    for (tm in tms) rx_row <- paste0(rx_row, sprintf(" %9s", fmt_t(tm["rx"])))
    if ("glmm" %in% present) for (e in SPEEDUP_VS) rx_row <- paste0(rx_row, sprintf(" %7s", fmt_x(tms[[e]]["rx"], tms[["glmm"]]["rx"])))
    cat(rx_row, "\n")
    if (any(!is.na(vapply(tms, `[[`, 0, "hess")))) {
      h_row <- lead_row("", "", "h")
      for (tm in tms) h_row <- paste0(h_row, sprintf(" %9s", fmt_t(tm["hess"])))
      if ("glmm" %in% present) for (e in SPEEDUP_VS) h_row <- paste0(h_row, sprintf(" %7s", fmt_x(tms[[e]]["hess"], tms[["glmm"]]["hess"])))
      cat(h_row, "\n")
    }
    # AGQ row, only for datasets the opt-in pass covered. The rx slot is the right
    # one to read: glmm's AGQ pass records the same rx/hessian split, and rx is the
    # arm without the FD-Hessian on top, so it is the closest thing to "time to fit".
    if (length(agq) && !is.null(agq[["glmm"]][[name]])) {
      a_tms <- setNames(lapply(timing_engines, function(e) {
        b <- if (e %in% names(agq)) agq[[e]][[name]] else NULL
        if (is.null(b)) c(rx = NA_real_, hess = NA_real_) else time_of(b)
      }), timing_engines)
      a_row <- lead_row("", "", sprintf("a%d", AGQ_K[[name]]))
      for (tm in a_tms) a_row <- paste0(a_row, sprintf(" %9s", fmt_t(tm["rx"])))
      if ("glmm" %in% present) for (e in SPEEDUP_VS) a_row <- paste0(a_row, sprintf(" %7s", fmt_x(a_tms[[e]]["rx"], a_tms[["glmm"]]["rx"])))
      cat(a_row, "\n")
    }
  }
}
cat("\nrx/h = time to fit + produce that SE (Hessian is the cost);",
    "gaussian/legacy single time shown under rx (no h row).\n",
    "vs_lme4/vs_mmjl = glmm speedup factor (other engine's time / glmm's time).\n",
    "py_gap = Python port time / glmm time (same kernel; the port tax of dict scan,\n",
    "  float() conversion, and the FFI copy). See engines/glmm_python.py.\n",
    "r_gap = R port time / glmm time (same kernel through the fastglmm extendr\n",
    "  wrapper; the port tax of the R<->Rust copy). See engines/glmm_r.R.\n",
    "aK = the same fit at nAGQ=K instead of Laplace (opt-in VALIDATION_AGQ pass;\n",
    "  absent unless it was run). NOT a controlled comparison: glmm fits these with\n",
    "  parallel_inner ON -- shipped config vs shipped config, threads included.\n",
    "  A blank lme4 cell is glmer refusing nAGQ>1 on a vector RE, not a missing run.\n")
