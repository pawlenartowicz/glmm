# Helpers shared by the boundary-probe corpus generators (gen_lmm.R, gen_q3.R,
# gen_r1.R, gen_precision.R), and by tools/prep/gen_boundary_pin_data.R, which
# builds the two committed in-crate boundary-pin fixtures: a change here that
# moves their bytes moves those fixtures. Each generator sources this
# file, builds its datasets in one fixed loop order, writes one CSV per
# dataset into the output directory and one grid manifest
# (`{"cells": [...]}`) that the speed campaign's `grid_fit` driver reads with
# GRID_MANIFEST / GRID_DATA_DIR.
#
# Reproducibility: each generator calls set.seed(<its base seed>) once, draws
# one seed per dataset from that stream, and re-seeds before simulating each
# dataset, so a dataset's bytes depend only on its own seed (recorded in the
# manifest's `seed`). Re-running a generator into an empty directory gives
# byte-identical files.

suppressPackageStartupMessages(library(jsonlite))

out_dir_arg <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 1) stop("usage: Rscript <generator>.R <output-dir>")
  dir.create(args[1], recursive = TRUE, showWarnings = FALSE)
  normalizePath(args[1])
}

# Row layout shared by every shape: G levels of the primary grouping g, `per`
# rows each. The crossed grouping h has G levels too and is assigned so that
# each g level sees `per` consecutive h levels (wrapping), and each h level
# holds `per` rows: fully crossed when G == per, balanced otherwise.
layout_rows <- function(G, per) {
  gi <- rep(seq_len(G), each = per)
  j <- rep(seq_len(per) - 1L, times = G)
  hi <- ((gi - 1L + j) %% G) + 1L
  list(n = G * per, gi = gi, hi = hi, j = j)
}

level_labels <- function(prefix, idx, width) sprintf(paste0("%s%0", width, "d"), prefix, idx)

# One random-effect draw per level: independent normal effects (truth
# correlation 0); an SD of 0 gives exact zeros.
draw_re <- function(n_levels, sd) rnorm(n_levels, 0, sd)

simulate_response <- function(eta, family, size = NULL) {
  n <- length(eta)
  switch(family,
    gaussian = eta + rnorm(n),
    binomial = if (is.null(size)) rbinom(n, 1, plogis(eta)) else rbinom(n, size, plogis(eta)),
    poisson = rpois(n, exp(eta)),
    stop("family ", family)
  )
}

write_dataset <- function(df, out_dir, name) {
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE, quote = FALSE)
}

# One manifest cell. `rhs` is the formula's right-hand side without the
# intercept (e.g. "x1 + (1 | g)"); `aggregated` marks a binomial fit on
# (incidence, size) counts, lowered through the `weights` field. `max_fun`
# is 1000 per θ coordinate, the LMM search budget; grid_fit reads it only to
# label a non-converged record `maxeval`.
make_cell <- function(case_id, corpus, family, factors, rhs, n_theta, n_obs, seed,
                      truth_sd, pattern, reml = NULL, nagq = NULL, data = NULL,
                      aggregated = FALSE) {
  if (aggregated) {
    r_lhs <- "cbind(incidence, size - incidence)"
    jl_lhs <- "prop"
  } else {
    r_lhs <- "y"
    jl_lhs <- "y"
  }
  cell <- list(
    case_id = case_id, corpus = corpus, family = family,
    factors = as.list(factors),
    r_formula = paste0(r_lhs, " ~ 1 + ", rhs),
    jl_formula = paste0("@formula(", jl_lhs, " ~ 1 + ", rhs, ")"),
    n_theta = n_theta, n_obs = n_obs, max_fun = 1000L * n_theta, seed = seed,
    pattern = pattern, truth_sd = as.list(truth_sd)
  )
  if (!is.null(reml)) cell$reml <- reml
  if (!is.null(nagq)) cell$nagq <- nagq
  if (!is.null(data)) cell$data <- data
  if (aggregated) cell$weights <- "size"
  cell
}

write_manifest <- function(cells, path, generated_by) {
  m <- list(
    schema = "grid_fit manifest: one cell per fit; `data` names the CSV when several cells share one",
    generated_by = generated_by,
    cells = cells
  )
  writeLines(toJSON(m, auto_unbox = TRUE, digits = NA, pretty = TRUE), path)
  cat(sprintf("%s: %d cells -> %s\n", generated_by, length(cells), path))
}

dataset_seeds <- function(base, n) {
  set.seed(base)
  sample.int(.Machine$integer.max, n)
}
