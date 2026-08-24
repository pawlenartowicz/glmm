#!/usr/bin/env Rscript
# Export each lme4-origin validation dataset to data/empirical/<name>.csv (sim rungs
# below go to data/simulated/) — run ONCE; the CSVs are committed and are the
# neutral input EVERY engine reads (R, Julia, later Rust).
# Exporting from one canonical source (lme4 in R) is what guarantees byte-identical
# input across engines and sidesteps row-order / factor-coding / NA differences
# between the ecosystems' built-in copies. Ordinary validation runs never call this.

suppressMessages({
  library(lme4)
  library(nlme)   # Machines / Oats (rungs 10-11) are bundled here, not lme4
  library(jsonlite)
  library(MASS)   # rnegbin for the simulated NB golden dataset; mvrnorm for sims below
})

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

manifest <- fromJSON(file.path(suite_dir, "manifest.json"), simplifyDataFrame = FALSE)
data_dir_of <- function(spec)
  file.path(suite_dir, "data",
            if (identical(spec$source, "sim")) "simulated" else "empirical")
dir.create(file.path(suite_dir, "data", "empirical"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(suite_dir, "data", "simulated"), showWarnings = FALSE, recursive = TRUE)

for (spec in manifest$datasets) {
  if (!identical(spec$source, "lme4") && !identical(spec$source, "nlme")) next  # sim rungs generated below
  if (!is.null(spec$data)) next  # re-linked rung (cbpp_probit): reuses another rung's CSV, nothing to export
  data(list = spec$name, package = spec$source)
  df <- as.data.frame(get(spec$name))  # strips nlme's groupedData/formula/units attrs

  # Every factor column -> plain unordered factor before writing labels. write.csv
  # only ever emits character labels (level order/orderedness never reaches the CSV
  # bytes -- verified directly), so this is a no-op for the 9 pre-existing lme4
  # exports; it only matters in intent for Machines$Worker/Oats$Block/VerbAgg$resp/
  # cake$temperature, the first ordered-factor columns this harness exports, so a
  # downstream reader never has to depend on R's ordered-factor convention.
  for (col in names(df)) if (is.factor(df[[col]])) df[[col]] <- factor(as.character(df[[col]]))

  # VerbAgg: response as an explicit 0/1 int column, not the raw r2 factor -- keeps
  # the binary outcome a neutral numeric column, consistent with how the other
  # binomial data in this corpus (cbpp's incidence/size) is committed.
  if (identical(spec$name, "VerbAgg")) df$y <- as.integer(df$r2 == "Y")

  # Arabidopsis: nutrient/rack are small-integer-coded factors in the source (plain
  # int columns), not R factors -- coerce here so they round-trip through the CSV
  # as labels the manifest's `factors` re-coercion step can pick up downstream.
  if (identical(spec$name, "Arabidopsis")) {
    df$nutrient <- factor(df$nutrient)
    df$rack     <- factor(df$rack)
  }

  out <- file.path(data_dir_of(spec), paste0(spec$name, ".csv"))
  write.csv(df, out, row.names = FALSE)
  cat(sprintf("wrote %-12s  %3d rows x %d cols\n", spec$name, nrow(df), ncol(df)))
}

# --- Simulated Gamma / NB datasets for the M3 family goldens ------------------
# Gamma and NB have no lme4-bundled dataset, so the frozen reference is "R's fit on
# THIS committed CSV". A fixed seed makes the data reproducible; the CSV (not the
# seed) is the artifact every consumer reads. A scalar random intercept over 24
# clusters with sd 0.6 keeps the GLMM references non-singular (Task R required a
# bump from the design's 20x10/sd0.5 sketch -- glmer(Gamma) went singular there).
set.seed(20260630)
make_clustered <- function(n_clust = 24, per = 12) {
  cl  <- factor(rep(seq_len(n_clust), each = per))
  x   <- rnorm(n_clust * per)
  grp <- factor(sample(c("a", "b"), n_clust * per, replace = TRUE))
  u   <- rnorm(n_clust, sd = 0.6)[cl]
  list(cl = cl, x = x, grp = grp,
       eta = 0.3 + 0.6 * x + 0.4 * (grp == "b") + u)
}

g    <- make_clustered()
mu_g <- exp(g$eta)                                   # Gamma log-link mean
y_g  <- rgamma(length(mu_g), shape = 2, scale = mu_g / 2)   # E[y]=mu, shape=2
write.csv(data.frame(cluster = g$cl, x = g$x, grp = g$grp, y = y_g),
          file.path(suite_dir, "data", "simulated", "sim_gamma.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_gamma", length(y_g), 4L))

nb    <- make_clustered()
mu_nb <- exp(nb$eta)
y_nb  <- MASS::rnegbin(length(mu_nb), mu = mu_nb, theta = 1.5)
write.csv(data.frame(cluster = nb$cl, x = nb$x, grp = nb$grp, y = y_nb),
          file.path(suite_dir, "data", "simulated", "sim_nb.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_nb", length(y_nb), 4L))

# --- Simulated crossed random-slope dataset for the #3 VarCorr synthetic -------
# g1 carries a random intercept + slope on x (q=2); g2 is a crossed intercept.
# Reference = R's lme4 fit on THIS committed CSV (fixed seed -> reproducible).
set.seed(20260701)
make_slope <- function(n1 = 20, n2 = 8, per = 6) {
  n   <- n1 * per
  g1  <- factor(rep(seq_len(n1), each = per))
  g2  <- factor(sample(seq_len(n2), n, replace = TRUE))
  x   <- rnorm(n)
  u0  <- rnorm(n1, sd = 1.2)[g1]
  u1  <- rnorm(n1, sd = 0.7)[g1]
  v0  <- rnorm(n2, sd = 0.9)[g2]
  y   <- 1.0 + 0.5 * x + u0 + u1 * x + v0 + rnorm(n, sd = 0.8)
  data.frame(y = y, x = x, g1 = g1, g2 = g2)
}
d_slope <- make_slope()
write.csv(d_slope, file.path(suite_dir, "data", "simulated", "sim_slope.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_slope", nrow(d_slope), 4L))

# --- Simulated near-collinear fixed design for the #4 rank-deficiency golden ----
# x3 = x1 + x2 + tiny jitter -> R drops the last collinear column (aliased) and
# fits the reduced model. Reference = R's glm(gaussian) on THIS committed CSV.
# Jitter sd 1e-13 (NOT 1e-9): glm.fit's QR drop tolerance is min(1e-7,
# epsilon/1000) = 1e-11 (control$epsilon defaults to 1e-8), tighter than lm's
# 1e-7. At sd 1e-9 x3's relative residual (~7e-10) exceeds 1e-11 so glm KEEPS the
# column (huge inflated coefs, no drop) -- not a valid column-drop oracle. sd 1e-13
# puts the relative residual ~7e-14, well under 1e-11, so glm reliably aliases x3.
set.seed(20260702)
make_collinear <- function(n = 80) {
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  x3 <- x1 + x2 + rnorm(n, sd = 1e-13) # near-exact linear dependence (see note above)
  y  <- 1.0 + 0.7 * x1 - 0.4 * x2 + rnorm(n, sd = 0.5)
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3)
}
d_col <- make_collinear()
write.csv(d_col, file.path(suite_dir, "data", "simulated", "sim_collinear.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_collinear", nrow(d_col), 4L))

# --- Simulated wide-crossed LMM (7 extra intercept factors, over-cap) --------
# 8 grouping factors total: g1 (primary) + c1..c7 (7 crossed extras).
# n_extras=7 > MAX_EXTRA_GROUPINGS=6 → over-envelope-by-count ⇒ sparse-Z path.
# Reference = lme4 REML fit on THIS committed CSV (fixed seed → reproducible).
# sd_g1=1.0, sd_c{1,6}=0.8, sd_c{2,4}=0.7, sd_c{3,5}=0.6, sd_c7=0.5 chosen
# to give clearly distinct variances; residual sd=0.6 keeps VCs well above 0.
set.seed(20260703)
make_wide_crossed <- function(n = 420, n_g1 = 12, n_c = 8) {
  g1 <- factor(sample(seq_len(n_g1), n, replace = TRUE))
  c1 <- factor(sample(seq_len(n_c),  n, replace = TRUE))
  c2 <- factor(sample(seq_len(n_c),  n, replace = TRUE))
  c3 <- factor(sample(seq_len(n_c),  n, replace = TRUE))
  c4 <- factor(sample(seq_len(n_c),  n, replace = TRUE))
  c5 <- factor(sample(seq_len(n_c),  n, replace = TRUE))
  c6 <- factor(sample(seq_len(n_c),  n, replace = TRUE))
  c7 <- factor(sample(seq_len(n_c),  n, replace = TRUE))
  x    <- rnorm(n)
  u_g1 <- rnorm(n_g1, sd = 1.0)[g1]
  u_c1 <- rnorm(n_c,  sd = 0.8)[c1]
  u_c2 <- rnorm(n_c,  sd = 0.7)[c2]
  u_c3 <- rnorm(n_c,  sd = 0.6)[c3]
  u_c4 <- rnorm(n_c,  sd = 0.7)[c4]
  u_c5 <- rnorm(n_c,  sd = 0.6)[c5]
  u_c6 <- rnorm(n_c,  sd = 0.8)[c6]
  u_c7 <- rnorm(n_c,  sd = 0.5)[c7]
  y    <- 1.5 + 0.8 * x + u_g1 + u_c1 + u_c2 + u_c3 + u_c4 + u_c5 + u_c6 + u_c7 +
          rnorm(n, sd = 0.6)
  data.frame(y = y, x = x, g1 = g1, c1 = c1, c2 = c2, c3 = c3,
             c4 = c4, c5 = c5, c6 = c6, c7 = c7)
}
d_wc <- make_wide_crossed()
write.csv(d_wc, file.path(suite_dir, "data", "simulated", "sim_wide_crossed.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_wide_crossed", nrow(d_wc), ncol(d_wc)))

# --- Simulated over-WIDTH random-slope LMM (q_g = 5, over-cap by WIDTH) -------
# ge carries intercept + 4 slopes → q_g = 5 > MAX_EXTRA_Q = 4, so the design is
# over-envelope by a single grouping's WIDTH (not by count) and routes to the
# sparse-Z path. This is the one over-cap axis with NO dense (NoZ) twin — NoZ
# physically cannot run q_g>4 — so the sparse fit is gated ONLY against lme4 here,
# and this golden is the sole oracle for the extras LEVEL-MAJOR multi-slope Z
# column layout at q_g=5. Reference = lme4 REML fit on THIS committed CSV.
# True ge covariance = compound-symmetry corr 0.25 (PD; min eig 0.75) on distinct
# diagonal sds (1.0,0.9,0.7,0.5,0.3) — distinct so a column permutation in Z would
# shift per-term variances and fail the golden; corr 0.25 (not 0) keeps the
# unstructured 5×5 fit in the interior so lme4 does not report singular.
set.seed(20260704)
make_wide_slopes <- function(n = 1200, n_gp = 20, n_ge = 40) {
  gp <- factor(sample(seq_len(n_gp), n, replace = TRUE))
  ge <- factor(sample(seq_len(n_ge), n, replace = TRUE))
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); x4 <- rnorm(n)
  sd_ge <- c(1.0, 0.9, 0.7, 0.5, 0.3)
  corr  <- matrix(0.25, 5, 5); diag(corr) <- 1.0
  Sigma <- diag(sd_ge) %*% corr %*% diag(sd_ge)
  b_ge  <- MASS::mvrnorm(n_ge, mu = rep(0, 5), Sigma = Sigma)  # n_ge x 5
  u_gp  <- rnorm(n_gp, sd = 0.8)[gp]
  re_ge <- b_ge[ge, 1] + b_ge[ge, 2] * x1 + b_ge[ge, 3] * x2 +
           b_ge[ge, 4] * x3 + b_ge[ge, 5] * x4
  y <- 1.5 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 - 0.2 * x4 +
       u_gp + re_ge + rnorm(n, sd = 0.6)
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, gp = gp, ge = ge)
}
d_ws <- make_wide_slopes()
write.csv(d_ws, file.path(suite_dir, "data", "simulated", "sim_wide_slopes.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_wide_slopes", nrow(d_ws), ncol(d_ws)))

# --- Simulated crossed slope-EXTRA LMM (rung 7: in-envelope sparse-routed class) --
# Both crossed groupings carry intercept + slope on x. Two slope terms are needed
# on purpose: the formula frontend hoists the FIRST slope-carrying term to the
# primary grouping, so `(1|g1) + (1+x|g2)` would lower to a q_p=2 primary + an
# intercept-only extra (NoZ-routed). With slopes on both, g1 is primary and g2
# stays a slope-carrying extra (q_g=2) — the exact in-envelope class the d2
# crossover routing sends to the sparse kernel, previously absent from the corpus.
# g1 has more levels than g2 so lme4's VarCorr order (descending levels) matches
# glmm's [primary | extra] block order positionally in compare.R.
# Reference = lme4/MixedModels fits on THIS committed CSV (fixed seed).
set.seed(20260705)
make_slope_extra <- function(n1 = 24, n2 = 16, per = 15) {
  n  <- n1 * per
  g1 <- factor(rep(seq_len(n1), each = per))
  g2 <- factor(sample(seq_len(n2), n, replace = TRUE))
  x  <- rnorm(n)
  S1 <- diag(c(1.1, 0.6)) %*% matrix(c(1, 0.3, 0.3, 1), 2) %*% diag(c(1.1, 0.6))
  S2 <- diag(c(0.9, 0.5)) %*% matrix(c(1, 0.2, 0.2, 1), 2) %*% diag(c(0.9, 0.5))
  b1 <- MASS::mvrnorm(n1, mu = c(0, 0), Sigma = S1)
  b2 <- MASS::mvrnorm(n2, mu = c(0, 0), Sigma = S2)
  y  <- 1.0 + 0.5 * x + b1[g1, 1] + b1[g1, 2] * x +
        b2[g2, 1] + b2[g2, 2] * x + rnorm(n, sd = 0.7)
  data.frame(y = y, x = x, g1 = g1, g2 = g2)
}
d_se <- make_slope_extra()
write.csv(d_se, file.path(suite_dir, "data", "simulated", "sim_slope_extra.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_slope_extra", nrow(d_se), ncol(d_se)))

# --- Step-2 sparse non-Gaussian validation datasets (rungs 8-9 + two goldens) -----
# One dataset per newly-wired over-envelope family arm (step-2 spec §4). Each
# design trips a REAL envelope cap so classify_design routes to Solver::Sparse
# (the non-Gaussian router has no slope-extra clause — only `over` reaches
# Sparse): over-COUNT = 7 crossed intercept extras (> MAX_EXTRA_GROUPINGS=6,
# the sim_wide_crossed template), over-WIDTH = one q_g=5 slope-block extra
# (> MAX_EXTRA_Q=4, the sim_wide_slopes template). Binomial + Poisson are
# curated 3-way rungs 8-9 (both reference engines fit them, Laplace); Gamma +
# NB go to the goldens track (neither oracle wires them 3-way).

# Rung 8: sim_sparse_binomial — over-count, cbpp SHAPE (aggregated binomial:
# `incidence` successes of `size` trials per row; fit.jl's binomial branch
# needs exactly this shape plus the manifest `weights` field). RE sds sized so
# the logit-scale signal survives aggregation without saturating fits.
set.seed(20260706)
make_sparse_binomial <- function(n = 240, n_g1 = 12, n_c = 8) {
  g1 <- factor(sample(seq_len(n_g1), n, replace = TRUE))
  cs <- lapply(1:7, function(k) factor(sample(seq_len(n_c), n, replace = TRUE)))
  x    <- rnorm(n)
  size <- sample(5:20, n, replace = TRUE)
  sd_c <- c(0.50, 0.45, 0.40, 0.50, 0.40, 0.45, 0.35)
  eta  <- 0.2 + 0.5 * x + rnorm(n_g1, sd = 0.8)[g1]
  for (k in 1:7) eta <- eta + rnorm(n_c, sd = sd_c[k])[cs[[k]]]
  incidence <- rbinom(n, size, plogis(eta))
  d <- data.frame(incidence = incidence, size = size, x = x, g1 = g1)
  for (k in 1:7) d[[paste0("c", k)]] <- cs[[k]]
  d
}
d_sb <- make_sparse_binomial()
write.csv(d_sb, file.path(suite_dir, "data", "simulated", "sim_sparse_binomial.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_sparse_binomial", nrow(d_sb), ncol(d_sb)))

# Rung 9: sim_sparse_poisson — over-count, Bernoulli-free count response.
set.seed(20260707)
make_sparse_poisson <- function(n = 420, n_g1 = 12, n_c = 8) {
  g1 <- factor(sample(seq_len(n_g1), n, replace = TRUE))
  cs <- lapply(1:7, function(k) factor(sample(seq_len(n_c), n, replace = TRUE)))
  x  <- rnorm(n)
  sd_c <- c(0.45, 0.40, 0.35, 0.45, 0.35, 0.40, 0.30)
  eta  <- 0.3 + 0.5 * x + rnorm(n_g1, sd = 0.7)[g1]
  for (k in 1:7) eta <- eta + rnorm(n_c, sd = sd_c[k])[cs[[k]]]
  y <- rpois(n, exp(eta))
  d <- data.frame(y = y, x = x, g1 = g1)
  for (k in 1:7) d[[paste0("c", k)]] <- cs[[k]]
  d
}
d_sp <- make_sparse_poisson()
write.csv(d_sp, file.path(suite_dir, "data", "simulated", "sim_sparse_poisson.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_sparse_poisson", nrow(d_sp), ncol(d_sp)))

# Golden: sim_sparse_gamma — over-WIDTH (q_g = 5 slope-block extra), gamma/log.
# Reuses the sim_wide_slopes recipe (CS corr 0.25 on distinct diagonal sds so a
# Z column permutation shifts per-term variances and fails the golden; corr>0
# keeps the unstructured 5x5 interior/non-singular), scaled down for the log
# link so mu stays in a sane range. shape=2 => E[y]=mu.
set.seed(20260708)
make_sparse_gamma <- function(n = 1200, n_gp = 20, n_ge = 40) {
  gp <- factor(sample(seq_len(n_gp), n, replace = TRUE))
  ge <- factor(sample(seq_len(n_ge), n, replace = TRUE))
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); x4 <- rnorm(n)
  sd_ge <- c(0.70, 0.50, 0.40, 0.30, 0.20)
  corr  <- matrix(0.25, 5, 5); diag(corr) <- 1.0
  Sigma <- diag(sd_ge) %*% corr %*% diag(sd_ge)
  b_ge  <- MASS::mvrnorm(n_ge, mu = rep(0, 5), Sigma = Sigma)
  u_gp  <- rnorm(n_gp, sd = 0.5)[gp]
  eta <- 0.5 + 0.6 * x1 - 0.4 * x2 + 0.3 * x3 - 0.2 * x4 + u_gp +
         b_ge[ge, 1] + b_ge[ge, 2] * x1 + b_ge[ge, 3] * x2 +
         b_ge[ge, 4] * x3 + b_ge[ge, 5] * x4
  mu <- exp(eta)
  y  <- rgamma(n, shape = 2, scale = mu / 2)
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, gp = gp, ge = ge)
}
d_sg <- make_sparse_gamma()
write.csv(d_sg, file.path(suite_dir, "data", "simulated", "sim_sparse_gamma.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_sparse_gamma", nrow(d_sg), ncol(d_sg)))

# Golden: sim_sparse_nb — over-count, negative-binomial counts (theta = 1.5,
# the sim_nb convention).
set.seed(20260709)
make_sparse_nb <- function(n = 420, n_g1 = 12, n_c = 8) {
  g1 <- factor(sample(seq_len(n_g1), n, replace = TRUE))
  cs <- lapply(1:7, function(k) factor(sample(seq_len(n_c), n, replace = TRUE)))
  x  <- rnorm(n)
  sd_c <- c(0.45, 0.40, 0.35, 0.45, 0.35, 0.40, 0.30)
  eta  <- 0.3 + 0.5 * x + rnorm(n_g1, sd = 0.7)[g1]
  for (k in 1:7) eta <- eta + rnorm(n_c, sd = sd_c[k])[cs[[k]]]
  y <- MASS::rnegbin(n, mu = exp(eta), theta = 1.5)
  d <- data.frame(y = y, x = x, g1 = g1)
  for (k in 1:7) d[[paste0("c", k)]] <- cs[[k]]
  d
}
d_sn <- make_sparse_nb()
write.csv(d_sn, file.path(suite_dir, "data", "simulated", "sim_sparse_nb.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_sparse_nb", nrow(d_sn), ncol(d_sn)))

# --- Rung 15: sim_three_level -- pure 3-level nesting depth (g1/g2), gaussian ----
# 10 "schools" (g1), 4 "classes" (g2) nested within each -- g2 labels 1..4 are
# REUSED across schools on purpose: nesting notation (1|g1/g2) disambiguates via
# the g1:g2 interaction, exactly like Oats' Block/Variety. Distinct sd_g1/sd_g2
# so a Z-column-order bug shifts per-level variance and fails the golden.
set.seed(20260710)
make_three_level <- function(n_g1 = 10, n_g2 = 4, per = 10) {
  n   <- n_g1 * n_g2 * per
  g1  <- factor(rep(seq_len(n_g1), each = n_g2 * per))
  g2  <- factor(rep(rep(seq_len(n_g2), each = per), times = n_g1))
  x   <- rnorm(n)
  ig  <- interaction(g1, g2, drop = TRUE)
  u1  <- rnorm(n_g1, sd = 1.0)[g1]
  u2  <- rnorm(nlevels(ig), sd = 0.7)[as.integer(ig)]
  y   <- 1.5 + 0.8 * x + u1 + u2 + rnorm(n, sd = 0.6)
  data.frame(y = y, x = x, g1 = g1, g2 = g2)
}
d_tl <- make_three_level()
write.csv(d_tl, file.path(suite_dir, "data", "simulated", "sim_three_level.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_three_level", nrow(d_tl), ncol(d_tl)))

# --- Rung 16: sim_max_q_slope -- single grouping at EXACTLY MAX_PRIMARY_Q=8 ------
# (intercept + 7 slopes) on one grouping g1. 60 groups x 25 obs (1500 rows) --
# 60 groups is well above the 36 free covariance parameters an unstructured 8x8
# needs, so the reference fit stays non-singular. Distinct diagonal sds
# (1.0..0.3) and a shared corr=0.2 (PD; keeps the fit off the boundary without
# masking a column-permutation bug, same rationale as sim_wide_slopes).
set.seed(20260711)
make_max_q_slope <- function(n_g1 = 60, per = 25) {
  n  <- n_g1 * per
  g1 <- factor(rep(seq_len(n_g1), each = per))
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rnorm(n); x4 <- rnorm(n)
  x5 <- rnorm(n); x6 <- rnorm(n); x7 <- rnorm(n)
  sd_b  <- c(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
  corr  <- matrix(0.2, 8, 8); diag(corr) <- 1.0
  Sigma <- diag(sd_b) %*% corr %*% diag(sd_b)
  b     <- MASS::mvrnorm(n_g1, mu = rep(0, 8), Sigma = Sigma)
  re    <- b[g1, 1] + b[g1, 2] * x1 + b[g1, 3] * x2 + b[g1, 4] * x3 +
           b[g1, 5] * x4 + b[g1, 6] * x5 + b[g1, 7] * x6 + b[g1, 8] * x7
  y <- 1.5 + 0.8 * x1 - 0.5 * x2 + 0.3 * x3 - 0.2 * x4 +
       0.4 * x5 - 0.3 * x6 + 0.2 * x7 + re + rnorm(n, sd = 0.5)
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5, x6 = x6, x7 = x7,
             g1 = g1)
}
d_mq <- make_max_q_slope()
write.csv(d_mq, file.path(suite_dir, "data", "simulated", "sim_max_q_slope.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_max_q_slope", nrow(d_mq), ncol(d_mq)))

# --- Rung 17: sim_crossed_at_cap -- EXACTLY 6 crossed extras = MAX_EXTRA_GROUPINGS,
# poisson -- the boundary just below the over-cap(7) sim_sparse_poisson (which is
# sparse-routed); 6 extras is the widest the dense/NoZ path can still take.
set.seed(20260712)
make_crossed_at_cap <- function(n = 420, n_g1 = 12, n_c = 8) {
  g1 <- factor(sample(seq_len(n_g1), n, replace = TRUE))
  cs <- lapply(1:6, function(k) factor(sample(seq_len(n_c), n, replace = TRUE)))
  x  <- rnorm(n)
  sd_c <- c(0.45, 0.40, 0.35, 0.45, 0.35, 0.40)
  eta  <- 0.3 + 0.5 * x + rnorm(n_g1, sd = 0.6)[g1]
  for (k in 1:6) eta <- eta + rnorm(n_c, sd = sd_c[k])[cs[[k]]]
  y <- rpois(n, exp(eta))
  d <- data.frame(y = y, x = x, g1 = g1)
  for (k in 1:6) d[[paste0("c", k)]] <- cs[[k]]
  d
}
d_cc <- make_crossed_at_cap()
write.csv(d_cc, file.path(suite_dir, "data", "simulated", "sim_crossed_at_cap.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_crossed_at_cap", nrow(d_cc), ncol(d_cc)))

# --- Rung 18: sim_binomial_slope_crossed -- 2 crossed groupings, EACH with a
# random slope (q=2 both) -- current binomial cases (cbpp, sim_sparse_binomial)
# are intercept-only. Aggregated binomial shape (incidence/size), cbpp/
# sim_sparse_binomial convention.
set.seed(20260713)
make_binomial_slope_crossed <- function(n = 300, n_g1 = 15, n_g2 = 12) {
  g1   <- factor(sample(seq_len(n_g1), n, replace = TRUE))
  g2   <- factor(sample(seq_len(n_g2), n, replace = TRUE))
  x    <- rnorm(n)
  size <- sample(5:20, n, replace = TRUE)
  S1 <- diag(c(0.8, 0.5)) %*% matrix(c(1, 0.2, 0.2, 1), 2) %*% diag(c(0.8, 0.5))
  S2 <- diag(c(0.6, 0.4)) %*% matrix(c(1, 0.15, 0.15, 1), 2) %*% diag(c(0.6, 0.4))
  b1 <- MASS::mvrnorm(n_g1, mu = c(0, 0), Sigma = S1)
  b2 <- MASS::mvrnorm(n_g2, mu = c(0, 0), Sigma = S2)
  eta <- 0.2 + 0.5 * x + b1[g1, 1] + b1[g1, 2] * x + b2[g2, 1] + b2[g2, 2] * x
  incidence <- rbinom(n, size, plogis(eta))
  data.frame(incidence = incidence, size = size, x = x, g1 = g1, g2 = g2)
}
d_bsc <- make_binomial_slope_crossed()
write.csv(d_bsc, file.path(suite_dir, "data", "simulated", "sim_binomial_slope_crossed.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_binomial_slope_crossed", nrow(d_bsc), ncol(d_bsc)))

# --- Rung 19: sim_poisson_nested -- 3-level nesting depth in the Poisson/Laplace
# kernel; today's Poisson coverage (grouseticks, sim_sparse_poisson) is
# crossed-only, never nested. Same g1/g2 nesting shape as sim_three_level,
# smaller RE sds to keep the log-link mean in a sane range.
set.seed(20260714)
make_poisson_nested <- function(n_g1 = 10, n_g2 = 4, per = 10) {
  n  <- n_g1 * n_g2 * per
  g1 <- factor(rep(seq_len(n_g1), each = n_g2 * per))
  g2 <- factor(rep(rep(seq_len(n_g2), each = per), times = n_g1))
  x  <- rnorm(n)
  ig <- interaction(g1, g2, drop = TRUE)
  u1 <- rnorm(n_g1, sd = 0.5)[g1]
  u2 <- rnorm(nlevels(ig), sd = 0.3)[as.integer(ig)]
  eta <- 0.5 + 0.3 * x + u1 + u2
  y  <- rpois(n, exp(eta))
  data.frame(y = y, x = x, g1 = g1, g2 = g2)
}
d_pn <- make_poisson_nested()
write.csv(d_pn, file.path(suite_dir, "data", "simulated", "sim_poisson_nested.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_poisson_nested", nrow(d_pn), ncol(d_pn)))

# --- Rung 20: sim_unbalanced_nested -- 3-level nesting with heavily unbalanced
# group sizes (per-g1 counts spanning ~2 to ~200) -- numerical stability under
# imbalance, vs sim_three_level's balanced design.
set.seed(20260715)
make_unbalanced_nested <- function(n_g1 = 10, n_g2 = 3) {
  sizes <- round(exp(seq(log(2), log(200), length.out = n_g1)))
  g1  <- factor(rep(seq_len(n_g1), times = sizes))
  n   <- length(g1)
  g2  <- factor(unlist(lapply(sizes, function(s) sample(seq_len(n_g2), s, replace = TRUE))))
  x   <- rnorm(n)
  ig  <- interaction(g1, g2, drop = TRUE)
  u1  <- rnorm(n_g1, sd = 1.0)[g1]
  u2  <- rnorm(nlevels(ig), sd = 0.6)[as.integer(ig)]
  y   <- 1.0 + 0.6 * x + u1 + u2 + rnorm(n, sd = 0.5)
  data.frame(y = y, x = x, g1 = g1, g2 = g2)
}
d_un <- make_unbalanced_nested()
write.csv(d_un, file.path(suite_dir, "data", "simulated", "sim_unbalanced_nested.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_unbalanced_nested", nrow(d_un), ncol(d_un)))

# --- Rung 21: sim_nested_crossed_mix -- one nested factor (g1/g2) AND one
# independent crossed factor (c1) together -- a design combo not present in any
# existing dataset (all prior nested/crossed cases are one or the other alone).
set.seed(20260716)
make_nested_crossed_mix <- function(n_g1 = 8, n_g2 = 3, per = 12, n_c1 = 6) {
  n   <- n_g1 * n_g2 * per
  g1  <- factor(rep(seq_len(n_g1), each = n_g2 * per))
  g2  <- factor(rep(rep(seq_len(n_g2), each = per), times = n_g1))
  c1  <- factor(sample(seq_len(n_c1), n, replace = TRUE))
  x   <- rnorm(n)
  ig  <- interaction(g1, g2, drop = TRUE)
  u1  <- rnorm(n_g1, sd = 1.0)[g1]
  u2  <- rnorm(nlevels(ig), sd = 0.6)[as.integer(ig)]
  uc  <- rnorm(n_c1, sd = 0.8)[c1]
  y   <- 1.2 + 0.6 * x + u1 + u2 + uc + rnorm(n, sd = 0.5)
  data.frame(y = y, x = x, g1 = g1, g2 = g2, c1 = c1)
}
d_ncm <- make_nested_crossed_mix()
write.csv(d_ncm, file.path(suite_dir, "data", "simulated", "sim_nested_crossed_mix.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_nested_crossed_mix", nrow(d_ncm), ncol(d_ncm)))

# --- NB theta-bracket-edge GLM datasets (coverage-gaps G2) --------------------
# Two sim_nb siblings whose glm.nb theta-hat sits near an end of glmm's theta
# search bracket [1e-3, 1e4] (fit.rs NB_THETA_LO/HI). Parameters were chosen by
# a pre-freeze glm.nb sweep: at these (theta, n, b0) MASS::glm.nb converges with
# ZERO warnings (no theta.ml iteration/alternation limit), so the reference is
# trustworthy -- more extreme settings put theta.ml itself at its limits.
# Low edge: theta_true = 0.005 -> theta-hat ~4.1e-3 (heavy overdispersion).
set.seed(20260717)
make_nb_edge <- function(n, b0, theta) {
  x   <- rnorm(n)
  grp <- factor(sample(c("a", "b"), n, replace = TRUE))
  mu  <- exp(b0 + 0.6 * x + 0.4 * (grp == "b"))
  data.frame(x = x, grp = grp, y = MASS::rnegbin(n, mu = mu, theta = theta))
}
d_nl <- make_nb_edge(n = 400, b0 = 1.0, theta = 0.005)
write.csv(d_nl, file.path(suite_dir, "data", "simulated", "sim_nb_lowtheta.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_nb_lowtheta", nrow(d_nl), ncol(d_nl)))

# High edge: near-Poisson, theta_true = 800 -> theta-hat ~5.3e2 (the profile
# is nearly flat up there, so theta-hat scatters across cells; information
# about theta comes from count size x n, hence n = 4000). Two constraints cap
# how close to NB_THETA_HI the frozen reference can sit: (a) the zero-warning
# check was done on the CSV ROUND-TRIP, not the in-memory data -- write.csv's
# 15-sig-digit format perturbs the flat profile enough to flip marginal cells
# into glm.nb's alternation/iteration limits, and cells with theta-hat > ~1e3
# all warned; (b) glmm's IRLS mu=1 cold start diverges for log-link counts
# with ybar over ~25 (b0 >= ~2.8 here), so larger-count cells (which carry
# more theta information) cannot be gated in-crate at all.
set.seed(20260724)
d_nh <- make_nb_edge(n = 4000, b0 = 2.0, theta = 800)
write.csv(d_nh, file.path(suite_dir, "data", "simulated", "sim_nb_hightheta.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_nb_hightheta", nrow(d_nh), ncol(d_nh)))

# --- NB unbalanced-nested GLMM golden dataset (coverage-gaps G2) --------------
# sim_nb's GLMM sibling on an UNBALANCED NESTED design (per-g1 sizes 8..120 on
# an exp ladder, the sim_unbalanced_nested precedent; g2 labels reused across
# parents so (1|g1/g2) disambiguates). theta = 1.5, the sim_nb convention.
# glmer.nb on this seed: 0 conv messages, non-singular (pre-freeze check).
set.seed(20260719)
make_nb_nested <- function(n_g1 = 12, n_g2 = 3, lo = 8, hi = 120) {
  sizes <- round(exp(seq(log(lo), log(hi), length.out = n_g1)))
  g1  <- factor(rep(seq_len(n_g1), times = sizes))
  n   <- length(g1)
  g2  <- factor(unlist(lapply(sizes, function(s) sample(seq_len(n_g2), s, replace = TRUE))))
  x   <- rnorm(n)
  ig  <- interaction(g1, g2, drop = TRUE)
  u1  <- rnorm(n_g1, sd = 0.6)[g1]
  u2  <- rnorm(nlevels(ig), sd = 0.4)[as.integer(ig)]
  eta <- 0.8 + 0.5 * x + u1 + u2
  data.frame(y = MASS::rnegbin(n, mu = exp(eta), theta = 1.5),
             x = x, g1 = g1, g2 = g2)
}
d_nn <- make_nb_nested()
write.csv(d_nn, file.path(suite_dir, "data", "simulated", "sim_nb_nested.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_nb_nested", nrow(d_nn), ncol(d_nn)))

# --- Collinear fixed column on a MIXED design (coverage-gaps G5) --------------
# sim_collinear's LMM sibling: x3 = x1 + x2 + jitter sd 1e-13 (same rationale --
# see the sim_collinear note on drop tolerances) plus a scalar random intercept
# over 12 clusters. lmer's rankMatrix check drops the LAST dependent column
# (x3), fits the reduced model non-singular, and fixef simply omits the dropped
# name -- the golden records lme4's choice via coef_names.
set.seed(20260720)
make_collinear_lmm <- function(n_g = 12, per = 15) {
  n  <- n_g * per
  g  <- factor(rep(seq_len(n_g), each = per))
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  x3 <- x1 + x2 + rnorm(n, sd = 1e-13)
  u  <- rnorm(n_g, sd = 0.9)[g]
  y  <- 1.0 + 0.7 * x1 - 0.4 * x2 + u + rnorm(n, sd = 0.5)
  data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, g = g)
}
d_cl <- make_collinear_lmm()
write.csv(d_cl, file.path(suite_dir, "data", "simulated", "sim_collinear_lmm.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_collinear_lmm", nrow(d_cl), ncol(d_cl)))

# --- High-mean Poisson GLM dataset (bug-fixes B3) ------------------------------
# Regression dataset for the IRLS log-link cold start: from the old mu = 1 seed
# (eta = 0) any count data with ybar over ~25-30 made the first WLS step
# overshoot and IRLS run away (beta -> ~9e304); R converges via its
# mustart = y + 0.1 initialize, which glmm now mirrors. ybar ~ 90 here puts the
# data far past the old divergence threshold while keeping glm() warning-free.
set.seed(20260725)
make_poisson_highmean <- function(n = 300, b0 = 4.3) {
  x   <- rnorm(n)
  grp <- factor(sample(c("a", "b"), n, replace = TRUE))
  mu  <- exp(b0 + 0.3 * x + 0.2 * (grp == "b"))
  data.frame(x = x, grp = grp, y = rpois(n, mu))
}
d_ph <- make_poisson_highmean()
write.csv(d_ph, file.path(suite_dir, "data", "simulated", "sim_poisson_highmean.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_poisson_highmean", nrow(d_ph), ncol(d_ph)))

# --- Rungs 25-27: vector-RE AGQ datasets (full-AGQ spec Part 5) ---------------
# Single grouping factor with a vector random effect (q=2 intercept+slope, q=3
# intercept+2 slopes) -- the exact shape the vector AGQ kernel (agq_deviance_vec)
# admits. Sized so AGQ matters: small clusters / sparse information, where the
# Laplace approximation is visibly biased. Confirmed at freeze (glmer Laplace vs
# GLMMadaptive nAGQ=11): RE-sd shifts 5%/36% (binomial q=2), 2%/9% (poisson
# q=2), 5%/13%/14% (binomial q=3); both oracles converge cleanly, non-singular.
# Two tiers per dataset: Laplace-anchor rung in manifest.datasets (vs glmer,
# machine-tight) + AGQ k=7/11 goldens in m3_goldens (vs GLMMadaptive).

# Rung 25: sim_binomial_slope1 -- Bernoulli rows (NOT aggregated: per-row trials
# of 1 are the low-information regime that maximizes Laplace bias), q=2.
set.seed(20260726)
make_binomial_slope1 <- function(n_g = 60, per = 8) {
  n <- n_g * per
  g <- factor(rep(seq_len(n_g), each = per))
  x <- rnorm(n)
  S <- diag(c(1.0, 0.6)) %*% matrix(c(1, 0.3, 0.3, 1), 2) %*% diag(c(1.0, 0.6))
  b <- MASS::mvrnorm(n_g, mu = c(0, 0), Sigma = S)
  eta <- 0.3 + 0.6 * x + b[g, 1] + b[g, 2] * x
  data.frame(y = rbinom(n, 1, plogis(eta)), x = x, g = g)
}
d_bs1 <- make_binomial_slope1()
write.csv(d_bs1, file.path(suite_dir, "data", "simulated", "sim_binomial_slope1.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_binomial_slope1", nrow(d_bs1), ncol(d_bs1)))

# Rung 26: sim_poisson_slope1 -- sparse counts (b0 = -1.2 puts the marginal mean
# ~0.5/row; Poisson-Laplace bias vanishes at high counts, so low counts are the
# regime worth testing), q=2.
set.seed(20260730)
make_poisson_slope1 <- function(n_g = 60, per = 4) {
  n <- n_g * per
  g <- factor(rep(seq_len(n_g), each = per))
  x <- rnorm(n)
  S <- diag(c(1.0, 0.6)) %*% matrix(c(1, -0.2, -0.2, 1), 2) %*% diag(c(1.0, 0.6))
  b <- MASS::mvrnorm(n_g, mu = c(0, 0), Sigma = S)
  eta <- -1.2 + 0.4 * x + b[g, 1] + b[g, 2] * x
  data.frame(y = rpois(n, exp(eta)), x = x, g = g)
}
d_ps1 <- make_poisson_slope1()
write.csv(d_ps1, file.path(suite_dir, "data", "simulated", "sim_poisson_slope1.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_poisson_slope1", nrow(d_ps1), ncol(d_ps1)))

# Rung 27: sim_binomial_slope2 -- Bernoulli, q=3 (intercept + 2 slopes): pins the
# temporary q_p<=3 cap surface and the kernel's dimensional generality. Larger
# n_g/per than q=2: 6 varcomp parameters need the extra information for a clean
# non-singular optimum on BOTH oracles (smaller draws left glmer with a
# degenerate Hessian at freeze).
set.seed(20260728)
make_binomial_slope2 <- function(n_g = 100, per = 12) {
  n <- n_g * per
  g <- factor(rep(seq_len(n_g), each = per))
  x1 <- rnorm(n); x2 <- rnorm(n)
  sds <- c(1.0, 0.7, 0.6)
  R <- matrix(c(1, 0.3, 0.1, 0.3, 1, 0.2, 0.1, 0.2, 1), 3)
  S <- diag(sds) %*% R %*% diag(sds)
  b <- MASS::mvrnorm(n_g, mu = c(0, 0, 0), Sigma = S)
  # b0 = 0.5 (not ~0): at a near-zero realized intercept the beta[0] direction is
  # shallow enough that glmer's optimizer stops ~1e-4 absolute short of the
  # MM/glmm optimum, and the tiny |beta0| denominator inflates that to a spurious
  # >1e-3 RELATIVE beta gap at the compare.R gate. A clearly nonzero intercept
  # keeps the relative comparison honest.
  eta <- 0.5 + 0.5 * x1 - 0.4 * x2 + b[g, 1] + b[g, 2] * x1 + b[g, 3] * x2
  data.frame(y = rbinom(n, 1, plogis(eta)), x1 = x1, x2 = x2, g = g)
}
d_bs2 <- make_binomial_slope2()
write.csv(d_bs2, file.path(suite_dir, "data", "simulated", "sim_binomial_slope2.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_binomial_slope2", nrow(d_bs2), ncol(d_bs2)))

# --- Rung 28: sim_poisson_offset -- Poisson GLMM with a known non-trivial
# exposure offset (spec section 2, the offset= oracle rung). log_exposure is a
# per-row LOG exposure (not a fixed effect column): glm(offset=)/glmer(offset=)
# add it to the linear predictor with an implicit coefficient of 1, the classic
# rate-model use ("y is a count over a varying exposure"). Own isolated seed
# block appended at the end -- does not disturb any earlier dataset's draws.
# per = 80 (not the sketch's 20): at per = 20 the slope's se_hessian gap vs
# lme4 sits at 2-2.5e-3 REGARDLESS of seed (checked 7 seeds) -- confirmed by a
# zero-offset control (same design, log_exposure forced to 0) that collapses
# the gap to ~1e-5, so the wide, non-trivial offset column genuinely raises
# the single-step-FD-vs-numDeriv noise floor here (the same method-floor
# effect documented for TOL$stddev_se_rel), not a data-generation fluke. Also
# not fixable by picking a lucky seed at per=20 -- the gap is seed-independent
# to within a factor of 2. per=80 (n=2400) pushes the same gap down to
# 6-8e-4 (checked 4 seeds), comfortably under se_hessian_rel=1e-3 with margin,
# without touching the tolerance itself.
set.seed(20260801)
make_poisson_offset <- function(n_clust = 30, per = 80) {
  n   <- n_clust * per
  cl  <- rep(seq_len(n_clust), each = per)
  x   <- rnorm(n)
  exposure <- runif(n, 0.5, 5)
  log_exposure <- log(exposure)
  b_cluster <- rnorm(n_clust, sd = 0.5)[cl]
  eta <- 0.3 + 0.5 * x + b_cluster + log_exposure
  y   <- rpois(n, exp(eta))
  data.frame(cluster = cl, x = x, log_exposure = log_exposure, y = y)
}
d_po <- make_poisson_offset()
write.csv(d_po, file.path(suite_dir, "data", "simulated", "sim_poisson_offset.csv"), row.names = FALSE)
cat(sprintf("wrote %-12s  %3d rows x %d cols\n", "sim_poisson_offset", nrow(d_po), ncol(d_po)))
