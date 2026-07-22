#!/usr/bin/env Rscript
# ACCURACY summary of the validation results -- pure REPORTING, the pass/fail gate 
# (with the per-quantity tolerances) lives in compare.R; the timing view lives 
# in summarize_timing.R (shares the read/format helpers below -- change together). 
# 
# Two views:
#   1. ACCURACY vs lme4: per dataset x engine, max relative diff on beta and on
#      each SE method (Rx and Hessian separately).
#   2. SE BY METHOD ("5 tests" / most-honest, gap 1.1): per dataset, the SE
#      estimates laid out so like method meets like. GLMM rungs have FIVE -- Rx
#      from glmm/lme4/MixedModels and Hessian from glmm/lme4 (MixedModels has no
#      Hessian variant); LMM/gaussian rungs have a single profiled SE per engine.

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

# Max relative difference over two aligned vectors; NA on absent/mismatched input.
rel_max <- function(x, y) {
  if (is.null(x) || is.null(y) || length(x) != length(y)) return(NA_real_)
  max(abs(x - y) / pmax(abs(x), abs(y), 1e-12))
}

# se_rx-slot vector (gaussian rungs carry a single profiled `se` in that role).
se_rx_of <- function(r) {
  if (is.null(r)) NULL
  else if (r$family == "gaussian") r$estimates$se else r$estimates$se_rx
}

fmt <- function(d) if (is.null(d) || is.na(d)) "-" else sprintf("%.1e", d)

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
if (is.null(ref)) stop("no lme4 reference present -- run engines/lme4.R first")
order_names <- names(ref)[order(vapply(ref, `[[`, 0L, "rung"))]

# ── view 1: accuracy vs lme4 ─────────────────────────────────────────────────
cat("== accuracy vs lme4 (max relative diff; gate in compare.R) ==\n")
cat(sprintf("%-16s %-4s %-12s %9s %9s %9s\n",
            "dataset", "rung", "engine", "beta", "SE(Rx)", "SE(Hess)"))
cat(strrep("-", 64), "\n")
for (group in c("empirical", "simulated")) {
  cat(sprintf("== %s ==\n", group))
  names_in_group <- if (group == "empirical") empirical_names else setdiff(order_names, empirical_names)
  for (name in intersect(order_names, names_in_group)) {
    a <- ref[[name]]
    for (e in setdiff(present, "lme4")) {
      b <- data[[e]][[name]]
      if (is.null(b)) next
      cat(sprintf("%-16s %-4d %-12s %9s %9s %9s\n",
                  name, a$rung, ENGINE_LABEL[[e]],
                  fmt(rel_max(a$estimates$beta, b$estimates$beta)),
                  fmt(rel_max(se_rx_of(a), se_rx_of(b))),
                  fmt(rel_max(a$estimates$se_hessian, b$estimates$se_hessian))))
    }
  }
}
cat("\nSE(Rx) = conditional on theta-hat (gaussian rungs: the single profiled SE);",
    "SE(Hess) = theta-beta coupled, glmm/lme4 only (n/a elsewhere).\n")

# ── view 2: SE by method (the 5 estimates) ───────────────────────────────────
se_cell <- function(v, i) {
  if (is.null(v) || i > length(v) || is.na(v[i])) "-" else sprintf("%.6f", v[i])
}
se_table <- function(title, rows, cols, row_label = "coef") {
  cat(sprintf("  %s\n", title))
  cat(sprintf("    %-14s %s\n", row_label,
              paste(sprintf("%12s", names(cols)), collapse = " ")))
  for (i in seq_along(rows))
    cat(sprintf("    %-14s %s\n", rows[i],
                paste(vapply(cols, function(v) sprintf("%12s", se_cell(v, i)), ""),
                      collapse = " ")))
}
cat("\n== SE by method ==\n")
for (group in c("empirical", "simulated")) {
  cat(sprintf("\n== %s ==\n", group))
  names_in_group <- if (group == "empirical") empirical_names else setdiff(order_names, empirical_names)
  for (name in intersect(order_names, names_in_group)) {
    a <- ref[[name]]
    g <- data[["glmm"]][[name]]
    mm <- data[["mixedmodels"]][[name]]
    coefs <- a$coef_names
    cat(sprintf("\n%s (rung %d, %s)\n", name, a$rung, a$family))
    if (a$family == "gaussian") {
      se_table("profiled SE", coefs,
               list(glmm = g$estimates$se, lme4 = a$estimates$se,
                    mmjl = mm$estimates$se))
    } else {
      se_table("Rx SE (conditional on theta-hat)", coefs,
               list(glmm = g$estimates$se_rx, lme4 = a$estimates$se_rx,
                    mmjl = mm$estimates$se_rx))
      se_table("Hessian SE (theta-beta coupled; glmm/lme4 only)", coefs,
               list(glmm = g$estimates$se_hessian, lme4 = a$estimates$se_hessian))
    }
    # Pairwise Rx max-rel diffs: the two oracles disagree ~2e-4 on their own Rx, and
    # glmm sits on the MixedModels value -- so glmm-mm << glmm-lme4 ~= lme4-mm.
    gx <- se_rx_of(g); lx <- se_rx_of(a); mx <- se_rx_of(mm)
    cat("  Rx agreement (max rel)\n")
    cat(sprintf("    %-14s %12s\n", "glmm-lme4", fmt(rel_max(gx, lx))))
    cat(sprintf("    %-14s %12s\n", "glmm-mmjl", fmt(rel_max(gx, mx))))
    cat(sprintf("    %-14s %12s\n", "lme4-mmjl", fmt(rel_max(lx, mx))))
    # RE-stddev SE (Hessian theta block; GLMM only). Per grouping, glmm vs lme4.
    l_sdse <- unlist(lapply(a$estimates$varcomp, `[[`, "stddev_se"))
    if (!is.null(l_sdse)) {
      g_sdse <- unlist(lapply(g$estimates$varcomp, `[[`, "stddev_se"))
      grps <- vapply(a$estimates$varcomp, `[[`, "", "group")
      se_table("RE-stddev SE (joint-Hessian theta block)", grps,
               list(glmm = g_sdse, lme4 = l_sdse), row_label = "group")
    }
  }
}
