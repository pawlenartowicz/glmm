# q3 corpus of the boundary-probe measurement: one grouping with an intercept
# and two random slopes, y ~ x1 + x2 + (1 + x1 + x2 | g1), Gaussian (REML),
# Bernoulli binomial and Poisson, sizes S/M/L (G = 20, 50, 200 with 10 rows per
# level), 500 datasets per family and size: 4,500 fits. Replicate r takes the
# truth pattern (r - 1) %% 3: every SD 0.5; the x2 slope SD 0; both slope SDs
# 0 (167, 167 and 166 datasets per family and size). Fixed effects: intercept
# 1 (Gaussian), 0 (binomial), 0.5 (Poisson); x1 0.5; x2 -0.3. Gaussian
# residual SD 1; random effects independent (truth correlation 0).
#
#   R_LIBS=<lib with jsonlite> Rscript gen_q3.R <output-dir>
#
# writes <output-dir>/<dataset>.csv and <output-dir>/manifest_q3.json.

local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  source(file.path(dirname(normalizePath(f)), "gen_common.R"))
})

BASE_SEED <- 20261030L
FAMILIES <- c(gaussian = "gau", binomial = "bin", poisson = "poi")
INTERCEPT <- c(gaussian = 1, binomial = 0, poisson = 0.5)
GS <- c(S = 20L, M = 50L, L = 200L)
PER <- 10L
REPS <- 500L
SD <- 0.5
PATTERNS <- c("all", "one0", "two0")
ZEROED <- list(all = integer(0), one0 = 3L, two0 = c(2L, 3L))
RHS <- "x1 + x2 + (1 + x1 + x2 | g1)"

out_dir <- out_dir_arg()
seeds <- dataset_seeds(BASE_SEED, length(FAMILIES) * length(GS) * REPS)
cells <- vector("list", length(seeds))
k <- 0L
for (family in names(FAMILIES)) {
  for (size in names(GS)) {
    G <- GS[[size]]
    for (r in seq_len(REPS)) {
      k <- k + 1L
      pattern <- PATTERNS[(r - 1L) %% 3L + 1L]
      sds <- rep(SD, 3)
      sds[ZEROED[[pattern]]] <- 0
      set.seed(seeds[k])
      lay <- layout_rows(G, PER)
      x1 <- rnorm(lay$n)
      x2 <- rnorm(lay$n)
      b0 <- draw_re(G, sds[1])[lay$gi]
      b1 <- draw_re(G, sds[2])[lay$gi]
      b2 <- draw_re(G, sds[3])[lay$gi]
      eta <- INTERCEPT[[family]] + b0 + (0.5 + b1) * x1 + (-0.3 + b2) * x2
      y <- simulate_response(eta, family)
      df <- data.frame(y = y, x1 = x1, x2 = x2, g1 = level_labels("g", lay$gi, 3))
      name <- sprintf("q3_%s_%s_%s_r%03d", FAMILIES[[family]], size, pattern, r)
      write_dataset(df, out_dir, name)
      cells[[k]] <- make_cell(
        case_id = name, corpus = "q3", family = family, factors = "g1", rhs = RHS,
        n_theta = 6L, n_obs = lay$n, seed = seeds[k], truth_sd = sds, pattern = pattern,
        reml = if (family == "gaussian") TRUE else NULL
      )
    }
  }
}
write_manifest(cells, file.path(out_dir, "manifest_q3.json"), "gen_q3.R")
