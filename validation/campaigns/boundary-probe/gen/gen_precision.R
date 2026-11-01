# Precision corpus of the boundary-probe measurement: data precise enough
# that a pin of a small but real variance costs deviance units. Poisson with
# log-means e^2, e^6, e^10, e^14, e^16 (intercept 2 ... 16, x 0.1) and
# aggregated binomial with 10, 1e3, 1e5 trials per row (intercept 0, x 0.3);
# one random-effect SD at 0 or 9e-5, the others 0.5, in three shapes:
#   cross   y ~ x + (1 | g) + (1 | h)      the h SD is the small one
#   nested  y ~ x + (1 | g) + (1 | gs)     gs nested in g; the gs SD is small
#   slope   y ~ x + (1 + x | g)            the slope SD is small
# G = 20 levels of g with 10 rows each (h: 20 levels, 10 rows each; gs: 5
# levels per g, 2 rows each); 50 replicates per regime, SD and shape: 2,400
# Laplace fits. Random effects independent.
#
#   R_LIBS=<lib with jsonlite> Rscript gen_precision.R <output-dir>
#
# writes <output-dir>/<dataset>.csv and <output-dir>/manifest_precision.json.

local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  source(file.path(dirname(normalizePath(f)), "gen_common.R"))
})

BASE_SEED <- 20261101L
REGIMES <- list(
  pois_e2 = list(family = "poisson", intercept = 2, slope = 0.1),
  pois_e6 = list(family = "poisson", intercept = 6, slope = 0.1),
  pois_e10 = list(family = "poisson", intercept = 10, slope = 0.1),
  pois_e14 = list(family = "poisson", intercept = 14, slope = 0.1),
  pois_e16 = list(family = "poisson", intercept = 16, slope = 0.1),
  bin_n10 = list(family = "binomial", intercept = 0, slope = 0.3, size = 10),
  bin_n1e3 = list(family = "binomial", intercept = 0, slope = 0.3, size = 1e3),
  bin_n1e5 = list(family = "binomial", intercept = 0, slope = 0.3, size = 1e5)
)
SMALL <- c("0" = 0, "9em5" = 9e-5)
SHAPES <- list(
  cross = list(rhs = "x + (1 | g) + (1 | h)", n_theta = 2L, factors = c("g", "h")),
  nested = list(rhs = "x + (1 | g) + (1 | gs)", n_theta = 2L, factors = c("g", "gs")),
  slope = list(rhs = "x + (1 + x | g)", n_theta = 3L, factors = "g")
)
G <- 20L
PER <- 10L
SUB <- 5L
REPS <- 50L
SD <- 0.5

out_dir <- out_dir_arg()
seeds <- dataset_seeds(BASE_SEED, length(REGIMES) * length(SMALL) * length(SHAPES) * REPS)
cells <- vector("list", length(seeds))
k <- 0L
for (regime_name in names(REGIMES)) {
  regime <- REGIMES[[regime_name]]
  aggregated <- !is.null(regime$size)
  for (small_name in names(SMALL)) {
    small <- SMALL[[small_name]]
    for (shape_name in names(SHAPES)) {
      shape <- SHAPES[[shape_name]]
      for (r in seq_len(REPS)) {
        k <- k + 1L
        set.seed(seeds[k])
        lay <- layout_rows(G, PER)
        si <- (lay$gi - 1L) * SUB + lay$j %/% (PER %/% SUB) + 1L
        x <- rnorm(lay$n)
        eta <- regime$intercept + regime$slope * x + draw_re(G, SD)[lay$gi]
        eta <- eta + switch(shape_name,
          cross = draw_re(G, small)[lay$hi],
          nested = draw_re(G * SUB, small)[si],
          slope = draw_re(G, small)[lay$gi] * x
        )
        df <- data.frame(x = x, g = level_labels("g", lay$gi, 2))
        if (aggregated) {
          df$incidence <- simulate_response(eta, "binomial", size = regime$size)
          df$size <- regime$size
        } else {
          df$y <- simulate_response(eta, "poisson")
        }
        if (shape_name == "cross") df$h <- level_labels("h", lay$hi, 2)
        if (shape_name == "nested") df$gs <- level_labels("s", si, 3)
        name <- sprintf("prec_%s_%s_sd%s_r%02d", regime_name, shape_name, small_name, r)
        write_dataset(df, out_dir, name)
        cells[[k]] <- make_cell(
          case_id = name, corpus = "precision", family = regime$family,
          factors = shape$factors, rhs = shape$rhs, n_theta = shape$n_theta,
          n_obs = lay$n, seed = seeds[k], truth_sd = c(SD, small),
          pattern = paste0("sd", small_name), aggregated = aggregated
        )
      }
    }
  }
}
write_manifest(cells, file.path(out_dir, "manifest_precision.json"), "gen_precision.R")
