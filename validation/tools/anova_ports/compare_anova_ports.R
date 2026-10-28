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
# non-finite value does not. A text cell (a `text` in place of the `value`:
# a per-term table's heading with its notes, and the fit's term_test_auto
# tags) agrees only when the texts are identical. The gate fails on:
#   - a front end whose calls are not exactly EXPECTED_CALLS (a call missing
#     from all three, or one all three added, fails too);
#   - a refused call (one `error` cell) outside EXPECTED_REFUSED, and a call in
#     EXPECTED_REFUSED that a front end no longer refuses;
#   - a cell one front end wrote and another did not (a missing cell);
#   - a cell written twice by one front end;
#   - a value that is neither a number nor "NaN", "Inf", "-Inf";
#   - a per-term table (a `terms:` call not refused) without its three text
#     cells: [heading | text], [term_test_auto | test], [term_test_auto |
#     reason];
#   - a value past the band, or two texts that differ;
#   - a table whose column names, or row names, differ in name or order between
#     any two front ends (both ports give R's column names, and the Rust front
#     end lays its tables out the same way).
#
# Not compared here: the exact test's `nsim` and `seed`, a comparison's
# heading (its model order among others), and the refusal messages, whose
# differences between the ports are allowed: the message for `test = "LRT"`
# on a fit that stored no likelihood-ratio table, and for `test = "auto"` on
# a Poisson GLMM fitted with `anova_test = "Wald"`, differ only in each
# port's call syntax; `test` with two fits and a bad `test` value follow
# each port's messages for `type` (two fits: an R error, a Python
# `TypeError`; a bad value: R's `match.arg`, a Python `ValueError`). The
# per-port tests pin them
# (python/tests/test_anova.py, r/tests/testthat/test-anova.R).
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
  # test = "LRT" on fits made with anova_test = "LRT": the likelihood-ratio
  # tables of a binomial GLMM, a binomial GLM and a binomial GLMM with trials.
  paste0("terms:", rep(c("glmer_sim_factorial_bern", "glm_sim_factorial_bern", "glmer_cbpp"),
                       each = 2L), c(":III:LRT", ":II:LRT")),
  # test = "LRT" on a binomial GLMM whose random slope is a fixed term: the
  # refit without that term keeps the random slope.
  "terms:glmer_sim_binomial_slope1:III:LRT",
  # anova(m) at test = "auto" on a Poisson GLMM: the likelihood-ratio table;
  # at an explicit "Wald": the Wald table.
  "terms:glmer_poisson_grouseticks:III:auto", "terms:glmer_poisson_grouseticks:III:Wald",
  # test = "auto" on a negative binomial and a Gamma GLMM: Wald, each with its
  # reason in the heading (the TERM_LABELS tables carry the binomial GLMM's
  # and the GLM's).
  "terms:glmer_nb_sim_nb:III:auto", "terms:glmer_gamma_sim_gamma:III:auto",
  "compare:missing_x2_wls_basic", "compare:missing_x2_wls_basic:nobs",
  "compare:rlrt_dyestuff", "anova:rlrt_dyestuff",
  # The post-fit calls: predict with standard errors on new rows and simulate
  # (2 draws, seed 1) of a Gaussian LMM and a binomial GLMM; r2, icc and pairwise
  # (Holm and Tukey) of the Gaussian one.
  paste0(c("predict:", "simulate:"),
         rep(c("lmer_sim_factorial", "glmer_sim_factorial_bern"), each = 2L)),
  "r2:lmer_sim_factorial", "icc:lmer_sim_factorial",
  "pairwise:lmer_sim_factorial:g:holm", "pairwise:lmer_sim_factorial:g:tukey")
stopifnot(length(EXPECTED_CALLS) == 67L, !anyDuplicated(EXPECTED_CALLS))

# The non-finite cells (NaN, Inf, -Inf) each front end writes over EXPECTED_CALLS:
# row 0's test columns of every compare/anova pair table and the Df column of
# the rlrt_dyestuff tables. Pinned so that a finite value turning non-finite in
# all three front ends fails, as the pairwise comparison alone cannot see it.
# Changes together with the call list above.
EXPECTED_NONFINITE <- 124L

# The calls every front end refuses. Empty: every call fits.
EXPECTED_REFUSED <- character(0)

read_cells <- function(path) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
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
  text <- vapply(recs, function(r) {
    t <- r$text
    if (is.character(t) && length(t) == 1L) enc2utf8(t) else NA_character_
  }, "")
  d <- data.frame(call = field("call", as.character), row = field("row", as.character),
                  col = field("col", as.character), row_index = field("row_index", as.integer),
                  col_index = field("col_index", as.integer), value = value, text = text,
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
    fail("%s no longer refuses %s: if it now fits, take it out of EXPECTED_REFUSED", name, k)
  }
  for (k in setdiff(unique(d$call[startsWith(d$call, "terms:")]), refused)) {
    for (cell in c("heading | text", "term_test_auto | test", "term_test_auto | reason")) {
      if (!paste(k, cell, sep = " | ") %in% d$key[!is.na(d$text)]) {
        fail("%s wrote no text cell [%s | %s]", name, k, cell)
      }
    }
  }
  for (k in d$key[is.na(d$value) & !is.nan(d$value) & is.na(d$text)]) {
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
# comparison (0 for two identical texts, Inf for two that differ).
common <- Reduce(intersect, lapply(fronts, `[[`, "key"))
vals <- lapply(fronts, function(d) d$value[match(common, d$key)])
texts <- lapply(fronts, function(d) d$text[match(common, d$key)])
calls <- fronts$Rust$call[match(common, fronts$Rust$key)]
is_text <- Reduce(`|`, lapply(texts, Negate(is.na)))
diff_of <- function(p) {
  a <- vals[[p[1]]]
  b <- vals[[p[2]]]
  vapply(seq_along(a), function(i) {
    if (is_text[i]) return(if (identical(texts[[p[1]]][i], texts[[p[2]]][i])) 0 else Inf)
    x <- a[i]
    y <- b[i]
    if (is.finite(x) && is.finite(y)) port_rel_max(x, y) else
      if ((is.nan(x) && is.nan(y)) || (is.infinite(x) && identical(x, y))) 0 else Inf
  }, 0)
}
diffs <- lapply(pairs, diff_of)
names(diffs) <- vapply(pairs, paste, "", collapse = "-")

shown <- function(side, i) {
  if (is_text[i]) encodeString(texts[[side]][i], quote = '"') else
    format(vals[[side]][i], digits = 17)
}
for (p in names(diffs)) {
  bad <- which(diffs[[p]] > band)
  for (i in bad) {
    sides <- strsplit(p, "-", fixed = TRUE)[[1]]
    if (is_text[i]) {
      fail("%s, cell [%s]: the texts differ: %s %s, %s %s", p, common[i],
           sides[1], shown(sides[1], i), sides[2], shown(sides[2], i))
    } else {
      fail("%s, cell [%s]: %s %s, %s %s (rel %.3g, band %.0e)", p, common[i],
           sides[1], shown(sides[1], i), sides[2], shown(sides[2], i), diffs[[p]][i], band)
    }
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
            if (length(EXPECTED_REFUSED)) paste(EXPECTED_REFUSED, collapse = ", ") else "none"))
cat(sprintf(paste("compared: %d cells (%d of them text) in %d calls (expected %d);",
                  "cells per front end: Rust %d, Python %d, R %d\n"),
            length(common), sum(is_text), length(unique(calls)), length(EXPECTED_CALLS),
            nrow(fronts$Rust), nrow(fronts$Python), nrow(fronts$R)))
if (any_fail) {
  cat("anova port gate: FAIL\n")
  quit(status = 1L)
}
cat("anova port gate: PASS\n")
