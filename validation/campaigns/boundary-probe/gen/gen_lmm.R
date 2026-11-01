# Gaussian corpus of the boundary-probe measurement: four random-effect shapes
# (one grouping with two slopes; crossed intercepts, which route dense; a
# slope on the primary grouping plus a crossed intercept; a slope on the extra
# grouping, which routes sparse), G in {10, 30, 100} with 10 rows per level,
# three truth patterns (every SD 0.5; one SD 0; two SDs 0), 100 replicates per
# cell, every dataset fitted by REML and by ML: 7,200 fits on 3,600 datasets.
# Fixed effects 1, 0.5 (x1), -0.3 (x2); residual SD 1; random effects
# independent (truth correlation 0).
#
#   R_LIBS=<lib with jsonlite> Rscript gen_lmm.R <output-dir>
#
# writes <output-dir>/<dataset>.csv and <output-dir>/manifest_lmm.json.

local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  source(file.path(dirname(normalizePath(f)), "gen_common.R"))
})

BASE_SEED <- 20261029L
GS <- c(10L, 30L, 100L)
PER <- 10L
REPS <- 100L
SD <- 0.5
BETA <- c(1, 0.5, -0.3)

# Each random-effect term: grouping and covariate ("1" = intercept). `one0`
# and `two0` name the terms whose truth SD is 0 under those patterns.
SHAPES <- list(
  q3 = list(
    rhs = "x1 + x2 + (1 + x1 + x2 | g)", n_theta = 6L,
    terms = list(c("g", "1"), c("g", "x1"), c("g", "x2")), one0 = 3L, two0 = c(2L, 3L)
  ),
  cross = list(
    rhs = "x1 + (1 | g) + (1 | h)", n_theta = 2L,
    terms = list(c("g", "1"), c("h", "1")), one0 = 2L, two0 = c(1L, 2L)
  ),
  slopecross = list(
    rhs = "x1 + (1 + x1 | g) + (1 | h)", n_theta = 4L,
    terms = list(c("g", "1"), c("g", "x1"), c("h", "1")), one0 = 2L, two0 = c(2L, 3L)
  ),
  slopeextra = list(
    rhs = "x1 + (1 | g) + (1 + x1 | h)", n_theta = 4L,
    terms = list(c("g", "1"), c("h", "1"), c("h", "x1")), one0 = 3L, two0 = c(1L, 3L)
  )
)
PATTERNS <- c("all", "one0", "two0")

out_dir <- out_dir_arg()
n_datasets <- length(SHAPES) * length(GS) * length(PATTERNS) * REPS
seeds <- dataset_seeds(BASE_SEED, n_datasets)
cells <- vector("list", 2L * n_datasets)
k <- 0L
for (shape_name in names(SHAPES)) {
  shape <- SHAPES[[shape_name]]
  uses_x2 <- grepl("x2", shape$rhs)
  uses_h <- any(vapply(shape$terms, function(t) t[1] == "h", logical(1)))
  for (G in GS) {
    for (pattern in PATTERNS) {
      sds <- rep(SD, length(shape$terms))
      if (pattern != "all") sds[shape[[pattern]]] <- 0
      for (r in seq_len(REPS)) {
        k <- k + 1L
        seed <- seeds[k]
        set.seed(seed)
        lay <- layout_rows(G, PER)
        x1 <- rnorm(lay$n)
        x2 <- if (uses_x2) rnorm(lay$n) else numeric(lay$n)
        eta <- BETA[1] + BETA[2] * x1 + BETA[3] * x2
        for (t in seq_along(shape$terms)) {
          term <- shape$terms[[t]]
          idx <- if (term[1] == "g") lay$gi else lay$hi
          b <- draw_re(G, sds[t])[idx]
          cov <- switch(term[2], "1" = 1, x1 = x1, x2 = x2)
          eta <- eta + b * cov
        }
        y <- simulate_response(eta, "gaussian")
        df <- data.frame(y = y, x1 = x1)
        if (uses_x2) df$x2 <- x2
        df$g <- level_labels("g", lay$gi, 3)
        if (uses_h) df$h <- level_labels("h", lay$hi, 3)
        name <- sprintf("lmm_%s_g%d_%s_r%03d", shape_name, G, pattern, r)
        write_dataset(df, out_dir, name)
        factors <- if (uses_h) c("g", "h") else "g"
        for (reml in c(TRUE, FALSE)) {
          cells[[2L * k - as.integer(reml)]] <- make_cell(
            case_id = paste0(name, if (reml) "_reml" else "_ml"), corpus = "lmm",
            family = "gaussian", factors = factors, rhs = shape$rhs,
            n_theta = shape$n_theta, n_obs = lay$n, seed = seed, truth_sd = sds,
            pattern = pattern, reml = reml, data = name
          )
        }
      }
    }
  }
}
write_manifest(cells, file.path(out_dir, "manifest_lmm.json"), "gen_lmm.R")
