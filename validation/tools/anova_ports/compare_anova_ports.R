#!/usr/bin/env Rscript
# The anova port gate's comparator: the cells anova_ports.rs, anova_ports.py
# and anova_ports.R wrote, compared three ways (Rust against Python, Rust
# against R, Python against R) cell by cell.
#
#   Rscript compare_anova_ports.R <rust.jsonl> <python.jsonl> <r.jsonl>
#
# The three front ends call the same kernel, so the band is the grid's port
# band, TOL$port_rel with port_rel_max (grid/tol.R, the metric compare.R
# --ports-only applies): round-off, diagnostic at 0, never widened. A value is
# a number or one of the strings "NaN", "Inf", "-Inf"; two non-finite cells
# agree only when they are the same of the three, and a number against a
# non-finite value does not. The gate fails on:
#   - a front end whose calls are not exactly EXPECTED_CALLS (a call missing
#     from all three, or one all three added, fails too);
#   - a refused call (one `error` cell) outside EXPECTED_REFUSED, and a call in
#     EXPECTED_REFUSED that a front end no longer refuses;
#   - a cell one front end wrote and another did not (a missing cell);
#   - a cell written twice by one front end;
#   - a value that is neither a number nor "NaN", "Inf", "-Inf";
#   - a value past the band;
#   - a table whose column names, or row names, differ in name or order between
#     any two front ends (both ports give R's column names, and the Rust front
#     end lays its tables out the same way).
#
# Not compared here: the exact test's `nsim` and `seed` and the heading's model
# order; the per-port tests pin them (python/tests/test_anova.py,
# r/tests/testthat/test-anova.R).
#
# Base R and jsonlite only: the CI port-gate job installs no statistics
# package.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("usage: Rscript compare_anova_ports.R <rust.jsonl> <python.jsonl> <r.jsonl>",
       call. = FALSE)
}
here <- dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "..", "..", "grid", "tol.R"))
band <- TOL$port_rel

# The call list of anova_ports.{rs,py,R}, pinned here so that a call all three
# front ends stop making (a manifest `case` renamed, an entry's `kind` moved, a
# table dropped) fails instead of shrinking the gate. Change together with the
# front ends.
PAIR_NAMES <- c(
  "anova_random_pair_sleepstudy", "anova_random_pair_penicillin",
  "anova_random_pair_pastes", "anova_ml_pair_sleepstudy_fixed",
  "anova_ml_pair_sim_factorial", "anova_ml_pair_sleepstudy_both",
  "anova_ols_pair_machines", "anova_ols_pair_wls_basic", "anova_glm_pair_cbpp",
  "anova_glm_pair_grouseticks", "anova_glm_pair_glm_gamma", "anova_glmm_pair_cbpp",
  "anova_glmm_vs_glm_cbpp", "anova_glmm_vs_glm_sim_binomial_slope1",
  "anova_glmm_pair_grouseticks", "anova_glmm_pair_sim_binomial_slope1",
  "anova_glmm_vs_glm_sim_nb", "anova_gamma_glmm_vs_glm_sim_gamma")
TERM_LABELS <- c("lmer_sim_factorial", "lm_sim_factorial", "glmer_sim_factorial_bern",
                 "glm_sim_factorial_bern")
EXPECTED_CALLS <- c(
  paste0(c("compare:", "anova:"), rep(PAIR_NAMES, each = 2L)),
  paste0("terms:", rep(TERM_LABELS, each = 2L), c(":III", ":II")),
  "compare:missing_x2_wls_basic", "compare:missing_x2_wls_basic:nobs",
  "compare:rlrt_dyestuff", "anova:rlrt_dyestuff")
stopifnot(length(EXPECTED_CALLS) == 48L, !anyDuplicated(EXPECTED_CALLS))

# The non-finite cells (NaN, Inf, -Inf) each front end writes over EXPECTED_CALLS:
# row 0's test columns of every compare/anova pair table, the Df column of the
# rlrt_dyestuff tables, and the refusal `error` cells. Pinned so that a finite
# value turning non-finite in all three front ends fails, as the pairwise
# comparison alone cannot see it. Changes together with the call list above.
EXPECTED_NONFINITE <- 120L

# The calls every front end refuses: the formula frontend does not take
# `Reaction ~ 1 + (Days | Subject)` (a random slope whose variable is not a
# fixed term). The same refusal as REFUSED_PAIRS in tests/inference_oracle.rs:
# change the two lists together.
EXPECTED_REFUSED <- c("compare:anova_ml_pair_sleepstudy_fixed",
                      "anova:anova_ml_pair_sleepstudy_fixed")

read_cells <- function(path) {
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  if (!length(lines)) stop(sprintf("%s: no cells", path), call. = FALSE)
  recs <- lapply(lines, jsonlite::fromJSON)
  field <- function(k, f) vapply(recs, function(r) f(r[[k]]), f(recs[[1]][[k]]))
  # A malformed value reads as NA (not NaN), flagged below.
  value <- vapply(recs, function(r) {
    v <- r$value
    if (is.numeric(v) && length(v) == 1L) as.double(v) else
      if (is.character(v) && length(v) == 1L && v %in% c("NaN", "Inf", "-Inf")) as.double(v) else
        NA_real_
  }, 0)
  d <- data.frame(call = field("call", as.character), row = field("row", as.character),
                  col = field("col", as.character), row_index = field("row_index", as.integer),
                  col_index = field("col_index", as.integer), value = value,
                  stringsAsFactors = FALSE)
  d$key <- paste(d$call, d$row, d$col, sep = " | ")
  d
}

fronts <- list(Rust = read_cells(args[[1]]), Python = read_cells(args[[2]]),
               R = read_cells(args[[3]]))
any_fail <- FALSE
fail <- function(fmt, ...) {
  cat(sprintf(paste0("FAIL: ", fmt, "\n"), ...))
  any_fail <<- TRUE
}

for (name in names(fronts)) {
  d <- fronts[[name]]
  calls_made <- unique(d$call)
  for (k in setdiff(EXPECTED_CALLS, calls_made)) fail("%s made no call %s", name, k)
  for (k in setdiff(calls_made, EXPECTED_CALLS)) fail("%s made the call %s, not in EXPECTED_CALLS", name, k)
  refused <- unique(d$call[d$col == "error"])
  for (k in setdiff(refused, EXPECTED_REFUSED)) {
    fail("%s refused %s, not in EXPECTED_REFUSED (see its stderr for the reason)", name, k)
  }
  for (k in setdiff(EXPECTED_REFUSED, refused)) {
    fail(paste("%s no longer refuses %s: if it now fits, take it out of EXPECTED_REFUSED",
               "together with REFUSED_PAIRS in tests/inference_oracle.rs"), name, k)
  }
  for (k in d$key[is.na(d$value) & !is.nan(d$value)]) {
    fail("%s wrote the cell [%s] with a value that is not a number, \"NaN\", \"Inf\" or \"-Inf\"",
         name, k)
  }
  n_nf <- sum(is.nan(d$value) | is.infinite(d$value))
  if (n_nf != EXPECTED_NONFINITE) {
    fail("%s wrote %d non-finite cells, expected %d (EXPECTED_NONFINITE)", name, n_nf,
         EXPECTED_NONFINITE)
  }
  dup <- unique(d$key[duplicated(d$key)])
  for (k in dup) fail("%s wrote the cell [%s] more than once", name, k)
}

all_keys <- unique(unlist(lapply(fronts, `[[`, "key")))
for (name in names(fronts)) {
  for (k in setdiff(all_keys, fronts[[name]]$key)) fail("%s has no cell [%s]", name, k)
}

# The same column names and row names, in the same order, per table, between
# every two front ends.
ordered_names <- function(d, call, what, index) {
  s <- d[d$call == call, ]
  unique(s[[what]][order(s[[index]])])
}
pairs <- list(c("Rust", "Python"), c("Rust", "R"), c("Python", "R"))
shared_calls <- Reduce(intersect, lapply(fronts, function(d) unique(d$call)))
for (call in shared_calls) {
  for (what in c("col", "row")) {
    index <- paste0(what, "_index")
    for (p in pairs) {
      a <- ordered_names(fronts[[p[1]]], call, what, index)
      b <- ordered_names(fronts[[p[2]]], call, what, index)
      if (!identical(a, b)) {
        fail("%s: the %s names differ: %s [%s], %s [%s]", call, what,
             p[1], paste(a, collapse = ", "), p[2], paste(b, collapse = ", "))
      }
    }
  }
}

# Values, cell by cell; per call the largest relative difference of each
# comparison.
common <- Reduce(intersect, lapply(fronts, `[[`, "key"))
vals <- lapply(fronts, function(d) d$value[match(common, d$key)])
calls <- fronts$Rust$call[match(common, fronts$Rust$key)]
diff_of <- function(a, b) {
  vapply(seq_along(a), function(i) {
    x <- a[i]
    y <- b[i]
    if (is.finite(x) && is.finite(y)) port_rel_max(x, y) else
      if ((is.nan(x) && is.nan(y)) || (is.infinite(x) && identical(x, y))) 0 else Inf
  }, 0)
}
diffs <- lapply(pairs, function(p) diff_of(vals[[p[1]]], vals[[p[2]]]))
names(diffs) <- vapply(pairs, paste, "", collapse = "-")

for (p in names(diffs)) {
  bad <- which(diffs[[p]] > band)
  for (i in bad) {
    sides <- strsplit(p, "-", fixed = TRUE)[[1]]
    fail("%s, cell [%s]: %s %s, %s %s (rel %.3g, band %.0e)", p, common[i],
         sides[1], format(vals[[sides[1]]][i], digits = 17),
         sides[2], format(vals[[sides[2]]][i], digits = 17), diffs[[p]][i], band)
  }
}

cat(sprintf("\n=== anova port gate: Rust vs Python vs R (band %.0e) ===\n", band))
cat(sprintf("%-52s %5s  %-10s %-10s %-10s\n", "call", "cells", "Rust-Py", "Rust-R", "Py-R"))
fmt_d <- function(d) {
  if (!is.finite(d)) "FAIL(nf)" else if (d > band) sprintf("FAIL %.1e", d) else sprintf("%.1e", d)
}
for (call in unique(calls)) {
  at <- calls == call
  cat(sprintf("%-52s %5d  %-10s %-10s %-10s\n", call, sum(at),
              fmt_d(max(diffs[[1]][at])), fmt_d(max(diffs[[2]][at])),
              fmt_d(max(diffs[[3]][at]))))
}
cat(sprintf("expected refusals (each front end checked above): %s\n",
            paste(EXPECTED_REFUSED, collapse = ", ")))
cat(sprintf("compared: %d cells in %d calls (expected %d); cells per front end: Rust %d, Python %d, R %d\n",
            length(common), length(unique(calls)), length(EXPECTED_CALLS), nrow(fronts$Rust),
            nrow(fronts$Python), nrow(fronts$R)))
if (any_fail) {
  cat("anova port gate: FAIL\n")
  quit(status = 1L)
}
cat("anova port gate: PASS\n")
