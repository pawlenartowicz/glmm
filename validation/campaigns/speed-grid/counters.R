#!/usr/bin/env Rscript
# Counter aggregation over the two counter passes. Usage:
#   Rscript counters.R <laplace.jsonl> <agq.jsonl> <out.csv> [baseline.csv]
# Laplace/LMM pass: campaigns/speed-grid, no nagq key, counters 1-3
# (stage1_evals, stage2_evals, stage1_shrink_evals, stage2_shrink_evals,
# pirls_hist). AGQ pass: campaigns/estimate-grid restricted to its 33 nagq
# cells, counter 4 (agq_evals, agq_node_evals). family/structure/n_theta/p
# are not in the JSONL records -- both passes' manifests carry them per
# case_id, so we join on that (same convention as analyze.R's `cells` join).
# The optional 4th argument writes the speed baseline: one row per
# subfamily (family_structure), median wall_seconds and n_eval, from
# status=="ok" Laplace cells only. A capped, non-converged fit (18000+
# evals) is not a fit that finished normally, so it is not a comparable
# timing unit -- excluding it is not about which direction it moves the
# median: on this grid dropping the two capped cells actually RAISED
# gaussian_q8q2x's median wall time (18.67s -> 26.45s), because the capped
# cell happened to run faster than that subfamily's normal cells.
suppressMessages({ library(jsonlite) })

args <- commandArgs(TRUE)
if (length(args) < 3) stop("usage: counters.R laplace.jsonl agq.jsonl out.csv [baseline.csv]")
suite_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
# read_jsonl / manifest_cells live in tol.R (shared with the analyze.R scripts).
source(file.path(suite_dir, "..", "..", "tol.R"))

laplace <- read_jsonl(args[1]); agq <- read_jsonl(args[2])

laplace_cells <- manifest_cells(file.path(suite_dir, "manifest.json"))
agq_cells <- manifest_cells(file.path(suite_dir, "..", "estimate-grid", "manifest.json"))

hist_zero <- function(x) if (is.null(x)) integer(0) else x

# ---- per-record row, shared by both passes --------------------------------
build_row <- function(cid, rec, cell, pass_name) {
  hist <- hist_zero(rec$pirls_hist)
  n_iters <- length(hist)
  idx <- seq_len(n_iters) - 1L    # pirls_hist[i] = evals that took i iterations, 0-indexed
  iters_total <- sum(idx * hist)
  evals <- sum(hist)
  iters_expanded <- rep(idx, hist)   # one entry per PIRLS evaluation, its iteration count
  p50 <- if (evals > 0) as.numeric(quantile(iters_expanded, 0.5, type = 1)) else NA_real_
  p90 <- if (evals > 0) as.numeric(quantile(iters_expanded, 0.9, type = 1)) else NA_real_
  iters_max <- if (evals > 0) max(iters_expanded) else NA_real_
  data.frame(
    case_id = cid, pass = pass_name,
    family = cell$family, structure = cell$structure, n_theta = cell$n_theta,
    p = cell$n_x + 1L,
    status = rec$status, converged = isTRUE(rec$converged), n_eval = rec$n_eval,
    stage1_evals = rec$stage1_evals, stage2_evals = rec$stage2_evals,
    stage1_shrink_evals = rec$stage1_shrink_evals,
    stage2_shrink_evals = rec$stage2_shrink_evals,
    pirls_evals = evals, pirls_iters_total = iters_total,
    pirls_iters_p50 = p50, pirls_iters_p90 = p90, pirls_iters_max = iters_max,
    agq_evals = if (is.null(rec$agq_evals)) 0L else rec$agq_evals,
    agq_node_evals = if (is.null(rec$agq_node_evals)) 0L else rec$agq_node_evals,
    singular = if (is.null(rec$singular)) NA else rec$singular,
    wall_seconds = if (is.null(rec$wall_seconds)) NA_real_ else rec$wall_seconds,
    stringsAsFactors = FALSE)
}

rows <- list()
for (cid in names(laplace_cells))
  if (!is.null(laplace[[cid]]))
    rows[[length(rows) + 1L]] <- build_row(cid, laplace[[cid]], laplace_cells[[cid]], "laplace")
for (cid in names(agq_cells))
  if (!is.null(agq[[cid]]))
    rows[[length(rows) + 1L]] <- build_row(cid, agq[[cid]], agq_cells[[cid]], "agq")
res <- do.call(rbind, rows)

out_cols <- c("case_id", "pass", "family", "structure", "n_theta", "p", "status",
              "converged", "n_eval", "stage1_evals", "stage2_evals",
              "stage1_shrink_evals", "stage2_shrink_evals", "pirls_evals",
              "pirls_iters_total", "pirls_iters_p50", "pirls_iters_p90",
              "pirls_iters_max", "agq_evals", "agq_node_evals")
write.csv(res[, out_cols], args[3], row.names = FALSE)

# ---- speed baseline (locked Laplace pass, ok cells only) -------------------
if (length(args) >= 4) {
  base_ok <- res[res$pass == "laplace" & res$status == "ok", ]
  base_ok$subfamily <- paste(base_ok$family, base_ok$structure, sep = "_")
  baseline <- do.call(rbind, lapply(split(base_ok, base_ok$subfamily), function(s) data.frame(
    subfamily = s$subfamily[1],
    median_wall_seconds = median(s$wall_seconds),
    median_n_eval = median(s$n_eval),
    n_cells = nrow(s))))
  baseline <- baseline[order(baseline$subfamily), ]
  write.csv(baseline, args[4], row.names = FALSE)
}

# ---- console aggregates -----------------------------------------------------
lap <- res[res$pass == "laplace", ]
lap_ok <- lap[lap$status == "ok" & lap$n_eval > 0, ]
# On the dense route stage1_evals + stage2_evals == n_eval. On the sparse
# GLMM two-stage route, stage1_evals + stage2_evals == n_eval + 1 (the d1
# warm-start eval is counted in both stage totals but only once in
# n_eval), so stage1_evals / n_eval can exceed the dense-route range by a
# hair on a sparse cell. This grid has zero sparse-routed cells, so it
# does not show up here -- noted for whoever reruns this on a grid that does.
lap_ok$stage_ratio <- lap_ok$stage1_evals / lap_ok$n_eval

cat("== (1) stage1_evals / n_eval ==\n")
# Gaussian (LMM) cells take the single-stage route -- everything is
# Stage::Two by design, so stage1_evals is always 0 for them. Pooling
# gaussian in with GLMM here would measure the family mix (how many cells
# are gaussian), not the optimizer's stage-1/stage-2 split. So report the
# gaussian count separately and restrict the ratio to GLMM cells only.
n_gaussian <- sum(lap_ok$family == "gaussian")
n_glmm <- sum(lap_ok$family != "gaussian")
cat(sprintf("scope: %d/%d cells are gaussian (single-stage LMM route, no stage 1 by design); ",
            n_gaussian, nrow(lap_ok)))
cat(sprintf("%d/%d cells are GLMM (binomial/poisson, two-stage route)\n", n_glmm, nrow(lap_ok)))
glmm_ok <- lap_ok[lap_ok$family != "gaussian", ]
n_glmm_nonzero <- sum(glmm_ok$stage1_evals > 0)
cat(sprintf("of the %d GLMM cells, %d have stage1_evals > 0\n", nrow(glmm_ok), n_glmm_nonzero))
cat(sprintf("GLMM cells only: median=%.4f range=[%.4f, %.4f] n=%d\n",
            median(glmm_ok$stage_ratio), min(glmm_ok$stage_ratio), max(glmm_ok$stage_ratio),
            nrow(glmm_ok)))
glmm_ok$n_theta_bin <- cut(glmm_ok$n_theta, c(0, 2, 5, 11, 20, 40),
                          labels = c("1-2", "3-5", "6-11", "12-20", "21-40"))
by_bin <- split(glmm_ok, glmm_ok$n_theta_bin)
for (b in names(by_bin)) {
  s <- by_bin[[b]]
  if (nrow(s) == 0) next
  cat(sprintf("  n_theta %-6s: median=%.4f range=[%.4f, %.4f] n=%d\n",
              b, median(s$stage_ratio), min(s$stage_ratio), max(s$stage_ratio), nrow(s)))
}

cat("\n== (2) stage2_shrink_evals / stage2_evals, by singular ==\n")
lap_s2 <- lap[lap$status == "ok" & lap$stage2_evals > 0, ]
lap_s2$shrink_ratio <- lap_s2$stage2_shrink_evals / lap_s2$stage2_evals
for (sg in c(FALSE, TRUE)) {
  # which(...) is NA-safe: lap_s2$singular == sg alone would keep NA-vs-sg
  # comparisons as NA and turn them into phantom all-NA rows in the subset.
  s <- lap_s2[which(lap_s2$singular == sg), ]
  if (nrow(s) == 0) { cat(sprintf("  singular=%s: no cells\n", sg)); next }
  cat(sprintf("  singular=%-5s: median=%.4f range=[%.4f, %.4f] n=%d\n",
              sg, median(s$shrink_ratio), min(s$shrink_ratio), max(s$shrink_ratio), nrow(s)))
}

cat("\n== (3) PIRLS iterations per evaluation ==\n")
lap_pirls <- lap[lap$status == "ok" & lap$pirls_evals > 0, ]
if (nrow(lap_pirls) == 0) {
  cat("  no PIRLS evaluations in the Laplace/LMM pass (LMM cells have no PIRLS loop)\n")
} else {
  cat("  pooled per-eval-iters median by family:\n")
  for (f in unique(lap_pirls$family)) {
    s <- lap_pirls[lap_pirls$family == f, ]
    cat(sprintf("    %s: median_iters_per_eval=%.2f n_cells=%d\n",
                f, median(s$pirls_iters_total / s$pirls_evals), nrow(s)))
  }
}

cat("\n== (4) AGQ pass: agq_evals, agq_node_evals, nodes/eval per cell ==\n")
agq_rows <- res[res$pass == "agq", ]
for (i in seq_len(nrow(agq_rows))) {
  r <- agq_rows[i, ]
  npe <- if (r$agq_evals > 0) r$agq_node_evals / r$agq_evals else NA_real_
  cat(sprintf("  %-40s agq_evals=%-5d agq_node_evals=%-6d nodes/eval=%.2f\n",
              r$case_id, r$agq_evals, r$agq_node_evals, npe))
}

cat("\n== failed / timed-out cells per pass ==\n")
for (pn in c("laplace", "agq")) {
  s <- res[res$pass == pn, ]
  bad <- sum(s$status != "ok")
  cat(sprintf("  %s: %d of %d cells not status=='ok'\n", pn, bad, nrow(s)))
}
