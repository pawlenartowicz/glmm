#!/usr/bin/env Rscript
# PARALLEL-PASS summary: glmm vs glmm-parallel ONLY -- the speedup glmm's own
# `parallel` feature buys over glmm's own serial path. Deliberately not a parity
# view: no other engine appears here, and none of these numbers is comparable to
# lme4/mmjl timings (different core sets). Cross-engine timing lives in
# summarize_timing.R; the pass/fail gate in compare.R.
#
# Legs (results/glmm_parallel/run_meta.json records the exact configuration):
#   serial   = results/glmm_{empirical,simulated}   feature off, pinned to ONE core (taskset -c 1)
#   parallel = results/glmm_parallel                --features parallel, P-core set + RAYON_NUM_THREADS
# Speedup = serial_time / parallel_time, i.e. "measured speedup against single core".
# Parallel results are bit-identical to serial by design -- any estimate mismatch
# is a bug, not noise (compare.R tolerances do not apply here).
#
# Timing is meaningless unless the machine was locked for BOTH legs (README).
#
# Read/format helpers mirror summarize_timing.R -- change together.

suppressMessages(library(jsonlite))

parity_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

read_engine <- function(dir_name) {
  dir <- file.path(parity_dir, "results", dir_name)
  if (!dir.exists(dir)) return(NULL)
  files <- list.files(dir, pattern = "\\.json$", full.names = TRUE)
  files <- files[basename(files) != "run_meta.json"]
  res <- lapply(files, fromJSON, simplifyVector = TRUE,
                simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
  if (!length(res)) return(NULL)
  setNames(res, vapply(res, `[[`, "", "dataset"))
}

# Serial glmm results are split empirical/simulated (parity/README.md) -- merge
# both dirs, same shape as compare.R's read_engine merge. glmm_parallel (below)
# stays a single directory and keeps using read_engine() above.
read_glmm_serial <- function() {
  files <- unlist(lapply(c("empirical", "simulated"), function(s)
    list.files(file.path(parity_dir, "results", paste0("glmm_", s)),
               pattern = "\\.json$", full.names = TRUE)))
  files <- files[basename(files) != "run_meta.json"]
  if (length(files) == 0) return(NULL)
  res <- lapply(files, fromJSON, simplifyVector = TRUE,
                simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
  setNames(res, vapply(res, `[[`, "", "dataset"))
}

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
fmt_x <- function(serial, par) {
  if (is.na(serial) || is.na(par) || par == 0) "-" else sprintf("%.2fx", serial / par)
}

short_name <- function(n) {
  n <- sub("^sim_", "s_", n)
  n <- gsub("binomial", "binom", n, fixed = TRUE)
  n <- gsub("crossed", "cross", n, fixed = TRUE)
  n <- gsub("balanced", "balan", n, fixed = TRUE)
  n
}

serial <- read_glmm_serial()
par    <- read_engine("glmm_parallel")
if (is.null(serial)) stop("no serial glmm results -- run oracle/fit.rs (feature off, taskset -c 1) first")
if (is.null(par))    stop("no glmm_parallel results -- run the parallel leg (see run_meta.json) first")
serial <- setNames(serial, short_name(names(serial)))
par    <- setNames(par,    short_name(names(par)))

meta_path <- file.path(parity_dir, "results", "glmm_parallel", "run_meta.json")
if (file.exists(meta_path)) {
  m <- fromJSON(meta_path)
  cat(sprintf("parallel leg: %s, RAYON_NUM_THREADS=%s | serial leg: %s | %s\n\n",
              m$parallel_leg$pinning, m$parallel_leg$rayon_num_threads,
              m$serial_leg$pinning, m$clock))
}

# Only datasets present in BOTH legs; ordered by rung (from the parallel leg).
names_both <- intersect(names(par), names(serial))
names_both <- names_both[order(vapply(par[names_both], `[[`, 0L, "rung"))]

cat("== parallel pass: glmm-parallel speedup vs single-core serial glmm ==\n")
name_w <- max(nchar(names_both)) + 1L
lead_row <- function(name, rung, metric) sprintf(paste0("%-", name_w, "s %-4s %-3s"), name, rung, metric)
header <- paste0(lead_row("dataset", "rung", ""),
                 sprintf(" %10s %10s %8s", "serial", "parallel", "speedup"))
cat(header, "\n")
cat(strrep("-", nchar(header)), "\n")
for (name in names_both) {
  ts <- time_of(serial[[name]])
  tp <- time_of(par[[name]])
  rung <- par[[name]]$rung
  cat(lead_row(name, rung, "rx"),
      sprintf(" %10s %10s %8s", fmt_t(ts["rx"]), fmt_t(tp["rx"]), fmt_x(ts["rx"], tp["rx"])), "\n")
  if (!is.na(ts["hess"]) || !is.na(tp["hess"])) {
    cat(lead_row("", "", "h"),
        sprintf(" %10s %10s %8s", fmt_t(ts["hess"]), fmt_t(tp["hess"]), fmt_x(ts["hess"], tp["hess"])), "\n")
    # FD-grid-isolated: the Hessian grid is the parallel surface on Laplace
    # fits (rx runs no parallel code at nagq=1), so h - rx per leg isolates it.
    gs <- ts["hess"] - ts["rx"]; gp <- tp["hess"] - tp["rx"]
    if (!is.na(gs) && !is.na(gp) && gs > 0 && gp > 0) {
      cat(lead_row("", "", "fd"),
          sprintf(" %10s %10s %8s", fmt_t(gs), fmt_t(gp), fmt_x(gs, gp)), "\n")
    }
  }
}
cat("\nrx = fit only (no parallel code at nagq=1 -- rx deltas are noise floor);",
    "h = fit + FD-Hessian SE;\nfd = h - rx, the FD-Hessian grid alone (the parallel surface).",
    "speedup = serial/parallel.\n")

agq_path <- file.path(parity_dir, "results", "glmm_parallel", "agq_timings.csv")
if (file.exists(agq_path)) {
  agq <- read.csv(agq_path, stringsAsFactors = FALSE)
  cat("\n== AGQ (nagq > 1): cluster loop, per-fit seconds ==\n")
  agq$per_fit <- agq$min_batch_seconds / agq$batch
  header2 <- sprintf("%-13s %-5s %12s %12s %12s %10s %10s",
                     "dataset", "nagq", "node_outer", "clust_outer", "parallel", "par_spdup", "ser_delta")
  cat(header2, "\n")
  cat(strrep("-", nchar(header2)), "\n")
  for (ds in unique(agq$dataset)) for (k in unique(agq$nagq[agq$dataset == ds])) {
    g <- agq[agq$dataset == ds & agq$nagq == k, ]
    t_no <- g$per_fit[g$leg == "node_outer_1c"]
    t_co <- g$per_fit[g$leg == "cluster_outer_1c"]
    t_p  <- g$per_fit[g$leg == "parallel_6c"]
    cat(sprintf("%-13s %-5d %12s %12s %12s %10s %+9.1f%%\n",
                ds, k, fmt_t(t_no), fmt_t(t_co), fmt_t(t_p),
                fmt_x(t_no, t_p), 100 * (t_co - t_no) / t_no))
  }
  cat("\npar_spdup = node_outer / parallel (speedup vs single-core serial).\n",
      "ser_delta = serial cluster-outer vs node-outer on one core",
      "(+ = the restructure costs serial time; the (1|INDEX) 1-row-per-cluster",
      "shape is the known pathological case).\n")
}
