#!/usr/bin/env Rscript
# ACCURACY summary of the accuracy grid -- pure REPORTING. The pass/fail gate
# (with the per-quantity tolerances) lives in compare.R; the timing view lives
# in summarize_timing.R (shares runs.R's run-discovery and comparison
# vocabulary -- change together).
#
# Three views, always printed:
#   1. accuracy vs EACH oracle: per cell x oracle, the max relative diff on
#      beta, on SE by method (Rx and Hessian separately) and on stddev. glmm
#      sits on the left of every comparison: no oracle is "the" reference here.
#   2. SE by method: per cell, the SE estimates laid out so like method meets
#      like -- Rx SE from the engines that compute it conditional on
#      theta-hat (glmm, lme4, MixedModels), Hessian SE from the engines that
#      keep the theta-beta coupling (glmm, lme4, glmmTMB, GLMMadaptive). A
#      column is printed only for the engines that have that method on that
#      cell, and coefficients are reordered into glmm's own order by name
#      before display (never positionally).
#   3. truth error per family, generated cells only: err_glmm and every
#      oracle's err_oracle, the best oracle's paired difference and its
#      2xSEM, and n. No verdict column -- the verdict is compare.R's.
#
# NEVER FAILS THE BUILD. Run discovery is wrapped, and the whole body below
# runs inside one tryCatch that always ends in quit(status = 0) -- a report
# that could not read a single run still exits clean and says why.

suppressMessages(library(jsonlite))

opt_glmm_run <- NULL
opt_fast <- FALSE
parse_args <- function() {
  for (a in commandArgs(TRUE)) {
    if (startsWith(a, "--glmm-run=")) opt_glmm_run <<- sub("^--glmm-run=", "", a)
    else if (identical(a, "--fast")) opt_fast <<- TRUE
    else stop(sprintf("unknown argument `%s`. Usage: summarize_accuracy.R [--glmm-run=DIR] [--fast]", a),
              call. = FALSE)
  }
}

# A REPORT NEVER FAILS A BUILD. newest_oracle_run() stop()s on a version
# mismatch because compare.R must treat that as an error; here the same
# condition is news, not a verdict -- print it and carry on with whatever
# runs were readable.
safe <- function(expr, what) tryCatch(expr, error = function(e) {
  cat(sprintf("  (%s unavailable: %s)\n", what, conditionMessage(e))); NULL })

fmt <- function(d) if (is.null(d) || is.na(d)) "-" else sprintf("%.1e", d)

se_cell <- function(v, i) if (is.null(v) || i > length(v) || is.na(v[i])) "-" else sprintf("%.6f", v[i])
se_table <- function(title, coefs, cols) {
  cat(sprintf("    %s\n", title))
  cat(sprintf("      %-14s %s\n", "coef", paste(sprintf("%12s", names(cols)), collapse = " ")))
  for (i in seq_along(coefs))
    cat(sprintf("      %-14s %s\n", coefs[i],
                paste(vapply(cols, function(v) sprintf("%12s", se_cell(v, i)), ""), collapse = " ")))
}

# The report's OWN truth-error floor, deliberately separate from tol.R's
# TOL$truth_floor: that constant is filled by the pilot and tol_for() stops on
# an NA one, but this script must run BEFORE the pilot exists. Same rule
# compare.R's --dev-floor mode uses to measure the pinned constant in the
# first place, over the same coordinate set (near_zero_truth_coords, runs.R):
# ceil1(2 x max nonzero |truth| under 0.1 on the deliberate near-zero
# coordinates the `nearzero`/`boundary` regimes generate).
truth_floor_probe <- function(cells) {
  small <- near_zero_truth_coords(cells)
  nz <- small[small > 0]
  if (length(nz)) ceil1(2 * max(nz)) else 0
}

ORACLE_LABEL <- c(lme4 = "lme4", glmmTMB = "glmmTMB", GLMMadaptive = "MA", MixedModels = "mmjl")

# Reorder rec[[field]] into base_names' coefficient order by NAME (coef_perm,
# from runs.R), never positionally -- lme4 orders InstEval's `dept`
# numerically while MixedModels/glmm order it lexicographically. NULL when the
# field is absent or the two coefficient sets cannot be matched.
aligned_field <- function(rec, base_names, field) {
  if (is.null(rec) || is.null(rec[[field]])) return(NULL)
  perm <- coef_perm(base_names, rec$coef_names)
  if (is.null(perm)) return(NULL)
  as.numeric(rec[[field]])[perm]
}

main <- function() {
  glmm_run <- if (!is.null(opt_glmm_run)) newest_glmm_run(GRID, "glmm", opt_glmm_run)
              else safe(newest_glmm_run(GRID, "glmm", NULL), "glmm run")

  # manifest_cells is the WHOLE manifest, kept separate from the (possibly
  # --fast-filtered) cells_all used for display: a run's completeness is
  # judged against every cell whose `oracles` list names that engine, not
  # against the current selection -- a --fast report still reads a full-grid
  # reference, and a reference judged complete against 20 cells would not be
  # one. Same rule compare.R's `required` follows.
  manifest_cells <- grid_cells_by_id(file.path(GRID, "manifest.json"))
  cells_all <- if (opt_fast) Filter(function(cl) "fast" %in% cl[["tags"]], manifest_cells) else manifest_cells

  cids <- if (is.null(glmm_run)) character(0) else intersect(names(cells_all), names(glmm_run$recs))
  ord <- if (!length(cids)) cids
         else cids[order(vapply(cells_all[cids], `[[`, "", "family"), cids)]

  V <- safe(grid_versions(GRID), "versions.json")
  runs <- setNames(lapply(ORACLES, function(o) {
    if (is.null(V)) return(NULL)
    required <- names(Filter(function(cl) o %in% cl[["oracles"]], manifest_cells))
    safe(newest_oracle_run(GRID, o, V[[o]], required), o)
  }), ORACLES)

  # ── view 1: accuracy vs each oracle ────────────────────────────────────────
  cat("== view 1: accuracy vs each oracle (max relative diff; the gate is in compare.R) ==\n")
  if (is.null(glmm_run)) {
    cat("  no glmm run present\n")
  } else if (!length(ord)) {
    cat("  no cell in this selection has a glmm record\n")
  } else {
    cat(sprintf("%-34s %-17s %-8s %-12s %9s %9s %9s %9s\n",
                "cell", "family", "arm", "engine", "beta", "SE(rx)", "SE(hess)", "stddev"))
    for (cid in ord) {
      cl <- cells_all[[cid]]; g <- glmm_run$recs[[cid]]
      oracles_here <- intersect(ORACLES, cl[["oracles"]])
      first <- TRUE
      for (o in oracles_here) {
        r <- runs[[o]]$recs[[cid]]
        orec <- if (is.null(r)) list() else setNames(list(r), o)
        diff_if_has <- function(q) if (has_quantity(g, q)) nearest_oracle_diff(g, orec, cl, q)$diff else NA_real_
        d_beta <- diff_if_has("beta")
        d_rx   <- diff_if_has("se_rx")
        d_h    <- diff_if_has("se_hessian")
        d_sd   <- diff_if_has("stddev")
        cat(sprintf("%-34s %-17s %-8s %-12s %9s %9s %9s %9s\n",
                    if (first) cid else "", if (first) cl[["family"]] else "",
                    if (first) cell_arm(cl) else "", ORACLE_LABEL[[o]],
                    fmt(d_beta), fmt(d_rx), fmt(d_h), fmt(d_sd)))
        first <- FALSE
      }
    }
  }

  # ── view 2: SE by method ────────────────────────────────────────────────────
  cat("\n== view 2: SE by method ==\n")
  if (is.null(glmm_run)) {
    cat("  no glmm run present\n")
  } else if (!length(ord)) {
    cat("  no cell in this selection has a glmm record\n")
  } else {
    for (cid in ord) {
      cl <- cells_all[[cid]]; g <- glmm_run$recs[[cid]]
      cat(sprintf("\n%s (%s/%s, %s)\n", cid, cl[["family"]], cl[["link"]], cell_arm(cl)))
      base <- g$coef_names

      rx_cols <- list()
      for (e in c("glmm", "lme4", "MixedModels")) {
        rec <- if (identical(e, "glmm")) g else runs[[e]]$recs[[cid]]
        v <- aligned_field(rec, base, "se_rx")
        if (!is.null(v)) rx_cols[[if (identical(e, "glmm")) "glmm" else ORACLE_LABEL[[e]]]] <- v
      }
      if (length(rx_cols)) se_table("Rx SE (conditional on theta-hat)", base, rx_cols)

      # Hessian SE is legitimately absent on a gaussian cell (single profiled
      # SE, no theta-beta coupling to report) -- no engine ever has it there,
      # so the table is silently omitted rather than printed as a gap.
      h_cols <- list()
      for (e in c("glmm", "lme4", "glmmTMB", "GLMMadaptive")) {
        rec <- if (identical(e, "glmm")) g else runs[[e]]$recs[[cid]]
        v <- aligned_field(rec, base, "se_hessian")
        if (!is.null(v)) h_cols[[if (identical(e, "glmm")) "glmm" else ORACLE_LABEL[[e]]]] <- v
      }
      if (length(h_cols)) se_table("Hessian SE (theta-beta coupled)", base, h_cols)
    }
  }

  # ── view 3: truth error per family ─────────────────────────────────────────
  cat("\n== view 3: truth error per family (generated cells only) ==\n")
  if (all(vapply(ORACLES, function(o) is.null(runs[[o]]), TRUE))) {
    cat("  no oracle run present\n")
  } else if (is.null(glmm_run)) {
    cat("  no glmm run present\n")
  } else {
    gen <- Filter(function(cl) !is.null(cl[["truth"]]) && !is.null(cl[["seed"]]) &&
                                !is.null(glmm_run$recs[[cl[["cell"]]]]), cells_all)
    if (!length(gen)) {
      cat("  no generated cell in this selection has a glmm record\n")
    } else {
      floor_ <- truth_floor_probe(cells_all)
      orec_by_cell <- setNames(lapply(names(cells_all), function(cid)
        setNames(lapply(ORACLES, function(o) runs[[o]]$recs[[cid]]), ORACLES)),
        names(cells_all))
      keys <- sort(unique(vapply(gen, function(cl) paste(cl[["family"]], cell_arm(cl)), "")))
      cat(sprintf("%-17s %-8s %-8s %4s %10s %10s %10s %10s %10s %-12s %s\n",
                  "family", "arm", "quantity", "n", "err_glmm", "err_lme4",
                  "err_glmmTMB", "err_MA", "err_mmjl", "best", "paired_2sem"))
      for (key in keys) {
        parts <- strsplit(key, " ")[[1]]; family <- parts[1]; arm <- parts[2]
        grp <- Filter(function(cl) identical(paste(cl[["family"]], cell_arm(cl)), key), gen)
        for (q in c("beta", "stddev", "corr")) {
          pt <- paired_truth_table(grp, glmm_run$recs, orec_by_cell,
                                   truth_extract[[q]], truth_vector[[q]], floor_)
          if (!length(pt$rows)) next
          best_o <- names(pt$rows)[which.min(vapply(pt$rows, `[[`, 0, "err_oracle"))]
          best <- pt$rows[[best_o]]
          err_of <- function(o) if (is.null(pt$rows[[o]])) "-" else sprintf("%.4f", pt$rows[[o]]$err_oracle)
          cat(sprintf("%-17s %-8s %-8s %4d %10.4f %10s %10s %10s %10s %-12s %.4f\n",
                      family, arm, q, best$n, best$err_glmm,
                      err_of("lme4"), err_of("glmmTMB"), err_of("GLMMadaptive"), err_of("MixedModels"),
                      ORACLE_LABEL[[best_o]], best$paired_2sem))
        }
      }
    }
  }
}

# Everything below runs inside ONE tryCatch so a missing runs.R, a bad flag or
# any unexpected error still exits 0 -- a report script must never fail a
# build. GRID is this script's own directory, resolved from `--file=` exactly
# as compare.R resolves it, so a throwaway grid tree that symlinks the scripts
# and holds its own runs/ is reported on as itself.
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
