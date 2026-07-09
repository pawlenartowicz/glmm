#!/usr/bin/env Rscript
# Human-readable TIMING summary of the parity results -- pure REPORTING, the
# pass/fail gate lives in compare.R; the accuracy views live in
# summarize_accuracy.R (shares the read/format helpers below -- change together).
#
# Per dataset x engine, the median PER-FIT time (rx / hessian split), plus the
# glmm speedup factor vs each oracle.
#
# Timing is indicative only unless the machine was locked when fit.* ran (README).

suppressMessages(library(jsonlite))

parity_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

read_engine <- function(dir_name) {
  dir <- file.path(parity_dir, "results", dir_name)
  if (!dir.exists(dir)) return(NULL)
  files <- list.files(dir, pattern = "\\.json$", full.names = TRUE)
  res <- lapply(files, fromJSON, simplifyVector = TRUE,
                simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
  if (!length(res)) return(NULL)
  setNames(res, vapply(res, `[[`, "", "dataset"))
}

# Per-method PER-FIT time. The JSON stores the median per SAMPLE, and a sample is
# `fits_per_sample` fits (batched rungs time 10 fits to beat R's timer floor) --
# normalize here so t_rx/t_hess are always seconds per fit. GLMM rungs split
# rx/hessian (the FD-Hessian / numDeriv SE is the main cost); gaussian/legacy
# carry a single median, shown in the rx slot.
time_of <- function(r) {
  t <- r$timing
  rx <- if (!is.null(t$fit_seconds_median_rx)) t$fit_seconds_median_rx
        else if (!is.null(t$fit_seconds_median)) t$fit_seconds_median
        else t$fit_seconds_min
  hess <- if (!is.null(t$fit_seconds_median_hessian)) t$fit_seconds_median_hessian else NA_real_
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

ENGINE_DIRS <- c(lme4 = "lme4", mixedmodels = "mixedmodels", glmm = "glmm")
ENGINE_LABEL <- c(lme4 = "lme4", mixedmodels = "mmjl", glmm = "glmm")
data <- Filter(Negate(is.null), lapply(ENGINE_DIRS, read_engine))
data <- lapply(data, function(lst) setNames(lst, short_name(names(lst))))
present <- names(data)
ref <- data[["lme4"]]
if (is.null(ref)) stop("no lme4 reference present -- run oracle/fit.R first")
order_names <- names(ref)[order(vapply(ref, `[[`, 0L, "rung"))]

TIMING_COLS <- c("glmm", "lme4", "mixedmodels")
timing_engines <- Filter(function(e) e %in% present, TIMING_COLS)

# glmm speedup vs lme4/mmjl: how many times faster glmm is (other_time / glmm_time).
fmt_x <- function(other, mine) {
  if (is.na(other) || is.na(mine) || mine == 0) "-" else sprintf("%.1fx", other / mine)
}
SPEEDUP_VS <- Filter(function(e) e %in% present, c("lme4", "mixedmodels"))

cat("== timing (median seconds per fit) ==\n")
name_w <- max(nchar(order_names)) + 1L
lead_row <- function(name, rung, metric) sprintf(paste0("%-", name_w, "s %-4s %-3s"), name, rung, metric)
header <- lead_row("dataset", "rung", "")
for (e in timing_engines) header <- paste0(header, sprintf(" %9s", ENGINE_LABEL[[e]]))
if ("glmm" %in% present) for (e in SPEEDUP_VS) header <- paste0(header, sprintf(" %9s", paste0("vs_", ENGINE_LABEL[[e]])))
cat(header, "\n")
cat(strrep("-", nchar(header)), "\n")
for (name in order_names) {
  a <- ref[[name]]
  tms <- setNames(lapply(timing_engines, function(e) {
    b <- data[[e]][[name]]
    if (is.null(b)) c(rx = NA_real_, hess = NA_real_) else time_of(b)
  }), timing_engines)
  rx_row <- lead_row(name, a$rung, "rx")
  for (tm in tms) rx_row <- paste0(rx_row, sprintf(" %9s", fmt_t(tm["rx"])))
  if ("glmm" %in% present) for (e in SPEEDUP_VS) rx_row <- paste0(rx_row, sprintf(" %9s", fmt_x(tms[[e]]["rx"], tms[["glmm"]]["rx"])))
  cat(rx_row, "\n")
  if (any(!is.na(vapply(tms, `[[`, 0, "hess")))) {
    h_row <- lead_row("", "", "h")
    for (tm in tms) h_row <- paste0(h_row, sprintf(" %9s", fmt_t(tm["hess"])))
    if ("glmm" %in% present) for (e in SPEEDUP_VS) h_row <- paste0(h_row, sprintf(" %9s", fmt_x(tms[[e]]["hess"], tms[["glmm"]]["hess"])))
    cat(h_row, "\n")
  }
}
cat("\nrx/h = time to fit + produce that SE (Hessian is the cost);",
    "gaussian/legacy single time shown under rx (no h row).\n",
    "vs_lme4/vs_mmjl = glmm speedup factor (other engine's time / glmm's time).\n")
