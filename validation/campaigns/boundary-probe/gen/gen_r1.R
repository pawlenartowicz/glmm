# R1 corpus of the boundary-probe measurement: random-effect blocks of at most
# two terms, binomial (Bernoulli) and Poisson, sizes S/M/L (G = 20, 50, 200
# with 10 rows per level), the varied SD s in {0, 0.05, 0.2}:
#   int    y ~ x + (1 | g)               s is the intercept SD
#   slope  y ~ x + (1 + x | g)           s is the slope SD, intercept SD 0.5
#   cross  y ~ x + (1 | g) + (1 | h)     s is the crossed SD, g SD 0.5
# 700 datasets per family, size and s; replicate r takes the shape
# (r - 1) %% 3 (234, 233 and 233 datasets): 12,600 Laplace fits (corpus
# `r1`). Every single-grouping dataset (int, slope) is fitted again at
# nagq = 5 (corpus `r1_agq5`, 8,406 fits on the same CSVs). Fixed effects:
# intercept 0 (binomial) or 0.5 (Poisson), x 0.5; random effects independent.
#
#   R_LIBS=<lib with jsonlite> Rscript gen_r1.R <output-dir>
#
# writes <output-dir>/<dataset>.csv, <output-dir>/manifest_r1.json and
# <output-dir>/manifest_r1_agq5.json.

local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  source(file.path(dirname(normalizePath(f)), "gen_common.R"))
})

BASE_SEED <- 20261031L
FAMILIES <- c(binomial = "bin", poisson = "poi")
INTERCEPT <- c(binomial = 0, poisson = 0.5)
GS <- c(S = 20L, M = 50L, L = 200L)
PER <- 10L
SDS <- c("0" = 0, "005" = 0.05, "02" = 0.2)
REPS <- 700L
SD <- 0.5
SHAPES <- list(
  int = list(rhs = "x + (1 | g)", n_theta = 1L, factors = "g"),
  slope = list(rhs = "x + (1 + x | g)", n_theta = 3L, factors = "g"),
  cross = list(rhs = "x + (1 | g) + (1 | h)", n_theta = 2L, factors = c("g", "h"))
)
NAGQ <- 5L

out_dir <- out_dir_arg()
seeds <- dataset_seeds(BASE_SEED, length(FAMILIES) * length(GS) * length(SDS) * REPS)
cells <- vector("list", length(seeds))
agq_cells <- list()
k <- 0L
for (family in names(FAMILIES)) {
  for (size in names(GS)) {
    G <- GS[[size]]
    for (s_name in names(SDS)) {
      s <- SDS[[s_name]]
      for (r in seq_len(REPS)) {
        k <- k + 1L
        shape_name <- names(SHAPES)[(r - 1L) %% 3L + 1L]
        shape <- SHAPES[[shape_name]]
        set.seed(seeds[k])
        lay <- layout_rows(G, PER)
        x <- rnorm(lay$n)
        eta <- INTERCEPT[[family]] + 0.5 * x
        sds <- switch(shape_name,
          int = {
            eta <- eta + draw_re(G, s)[lay$gi]
            s
          },
          slope = {
            eta <- eta + draw_re(G, SD)[lay$gi] + draw_re(G, s)[lay$gi] * x
            c(SD, s)
          },
          cross = {
            eta <- eta + draw_re(G, SD)[lay$gi] + draw_re(G, s)[lay$hi]
            c(SD, s)
          }
        )
        y <- simulate_response(eta, family)
        df <- data.frame(y = y, x = x, g = level_labels("g", lay$gi, 3))
        if (shape_name == "cross") df$h <- level_labels("h", lay$hi, 3)
        name <- sprintf("r1_%s_%s_%s_sd%s_r%03d", FAMILIES[[family]], shape_name, size, s_name, r)
        write_dataset(df, out_dir, name)
        cells[[k]] <- make_cell(
          case_id = name, corpus = "r1", family = family, factors = shape$factors,
          rhs = shape$rhs, n_theta = shape$n_theta, n_obs = lay$n, seed = seeds[k],
          truth_sd = sds, pattern = paste0("sd", s_name)
        )
        if (shape_name != "cross") {
          agq_cells[[length(agq_cells) + 1L]] <- make_cell(
            case_id = paste0(name, "_agq", NAGQ), corpus = "r1_agq5", family = family,
            factors = shape$factors, rhs = shape$rhs, n_theta = shape$n_theta,
            n_obs = lay$n, seed = seeds[k], truth_sd = sds, pattern = paste0("sd", s_name),
            nagq = NAGQ, data = name
          )
        }
      }
    }
  }
}
write_manifest(cells, file.path(out_dir, "manifest_r1.json"), "gen_r1.R")
write_manifest(agq_cells, file.path(out_dir, "manifest_r1_agq5.json"), "gen_r1.R")
