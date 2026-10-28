#!/usr/bin/env Rscript
# The ill-conditioned GLM designs of the `conditioning` references ->
# ../../data/inference/glm_band_<design>_<family>.csv, one file per design and
# family: designs a and d for logit, probit, poisson and gamma (log link), designs
# b and c for logit, probit and poisson (14 files). goldens_glm_band.py fits them.
#
# RUN ONCE; THE CSVs ARE COMMITTED. The CSV, not the seed, is the artifact: a
# re-run must reproduce every file byte for byte. Run from the repository root:
#   Rscript validation/tools/prep/gen_glm_band_data.R
# Base R (stats) only. RNG: R 4.5.3's default kinds (Mersenne-Twister, Inversion,
# Rejection). Values are written with sprintf("%.17g"), as gen_illcond_data.R
# does, and every column is checked to read back identical: at 15 digits
# a + 2^-k * b equals `a` for k near 50, and design a would turn into design d.
#
# THE DESIGNS. 200 rows, i = 1..200. Every file has the columns y, w, a, where
# w = 1 + ((i - 1) mod 3) are the prior weights an entry may use and
# a ~ N(0, 1). The block is rows 1 to 20. Per family, set.seed(<seed>), then in
# this order: a = rnorm(200); b = rnorm(200); the outcome y0 of every row,
# drawn at eta = 0.5 + 0.5a (logit and probit rbinom(200, 1, mu), poisson
# rpois(200, mu), gamma rgamma(200, shape = 2, scale = mu / 2)); then, except on
# gamma, u = rnorm(9) and d on the block in nine +/- pairs of rows with equal w:
# in each run of six block rows, rows r, r + 1, r + 2 carry the next three u and
# rows r + 3, r + 4, r + 5 carry their negatives; rows 19 (w = 1) and 20
# (w = 2) are left without a partner and carry d = 0. d is drawn last, so
# designs a and d do not depend on it. Seeds: logit 20261101, probit 20261102, poisson
# 20261103, gamma 20261104.
#   a  y = y0; columns x2_k<k> = a + 2^-k * b, k = 3, 7, 10, 13, 17, 20, 23,
#      27, 30, 33, 37, 40, 43, 47, 50 (one delta per decade).
#   b  x2 = a + d on the block, a elsewhere; columns off_<e> = -s on the block,
#      0 elsewhere, e = 2..15, with s such that the Fisher weight at eta = -s is
#      10^-e (logit mu(1 - mu); probit dnorm^2 / (pnorm (1 - pnorm)), solved by
#      uniroot; poisson mu, so s = e log(10)). Off the block y = y0. On the
#      block y = 1, so sum(d * y) = 0 and sum(w * d * y) = 0 there and nothing
#      separates along d. (Exactly zero for d; the fits read x2 - a, which
#      differs from d by rounding, so their sums are about 1e-16.) With a nonzero sum the score along d would ask
#      sum(w * d * mu) = sum(w * d * y) on the block, and the MLE would lift the
#      block's mu back toward 1 on the rows with the largest |d|, so the
#      weights at the MLE would not follow the push. On probit the block
#      weights at the MLE fall more slowly than the push, and tau reaches only
#      about 1e-9 to 1e-8 at the largest push, because the probit score of a
#      y = 1 row at eta = -s is about s rather than about 1, so the block pulls
#      the intercept up and lifts its own eta. Were d
#      constant on the block, the offset would lie in the span of (1, a, x2),
#      and the MLE would absorb the push into the coefficients and move no
#      weight.
#   c  as b, with block outcomes that separate along d: logit and probit
#      y = 1[d > 0] (one member of each pair, 0 on rows 19 and 20); poisson
#      y = 0 with d replaced by |d|. No finite MLE.
#   d  y = y0; x2 = a, an exact duplicate of a.
# The script stops unless each offset is outside the column span of
# (1, a, x2) (which also fails when d is constant on the block),
# sum(d * y) = 0 and sum(w * d * y) = 0 on design b's block (checked pair by
# pair, so exactly), no x2_k column equals a on every row, and design c's
# block outcomes separate along x2 - a as written.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", "..")) # nolint
out_dir <- file.path(suite_dir, "data", "inference")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

N <- 200L
BLOCK <- 1:20
K <- c(3, 7, 10, 13, 17, 20, 23, 27, 30, 33, 37, 40, 43, 47, 50)
E <- 2:15
SEEDS <- c(logit = 20261101, probit = 20261102, poisson = 20261103, gamma = 20261104)

# s with Fisher weight 10^-e at eta = -s. The logit root is
# mu = 2x / (1 + sqrt(1 - 4x)), the form of (1 - sqrt(1 - 4x)) / 2 that does
# not cancel at small x.
push <- function(family, e) {
  x <- 10^-e
  switch(family,
    logit = {
      mu <- 2 * x / (1 + sqrt(1 - 4 * x))
      log1p(-mu) - log(mu)
    },
    probit = uniroot(function(s) {
      2 * dnorm(-s, log = TRUE) - pnorm(-s, log.p = TRUE) - pnorm(s, log.p = TRUE) + e * log(10)
    }, c(0, 40), tol = 1e-15)$root,
    poisson = e * log(10))
}

draw_y <- function(family, eta) {
  switch(family,
    logit = rbinom(N, 1, plogis(eta)),
    probit = rbinom(N, 1, pnorm(eta)),
    poisson = rpois(N, exp(eta)),
    gamma = rgamma(N, shape = 2, scale = exp(eta) / 2))
}

emit <- function(name, df) {
  out <- df
  for (nm in names(df)) out[[nm]] <- sprintf("%.17g", df[[nm]])
  path <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(out, path, row.names = FALSE, quote = FALSE)
  back <- read.csv(path)
  for (nm in names(df)) {
    if (!identical(as.numeric(back[[nm]]), as.numeric(df[[nm]]))) {
      stop(sprintf("%s: column %s does not round-trip through read.csv", name, nm))
    }
  }
  cat(sprintf("wrote %-24s  %4d rows x %d cols  (round-trip exact)\n",
              name, nrow(df), ncol(df)))
}

offset_columns <- function(family, df, x2) {
  for (e in E) {
    off <- numeric(N)
    off[BLOCK] <- -push(family, e)
    if (qr(cbind(1, df$a, x2, off))$rank != 4L) {
      stop(sprintf("%s: off_%d lies in the column span of (1, a, x2)", family, e))
    }
    df[[sprintf("off_%d", e)]] <- off
  }
  df
}

for (family in names(SEEDS)) {
  set.seed(SEEDS[[family]])
  a <- rnorm(N)
  b <- rnorm(N)
  y0 <- draw_y(family, 0.5 + 0.5 * a)
  w <- 1 + ((seq_len(N) - 1) %% 3)
  base <- data.frame(y = y0, w = w, a = a)

  da <- base
  for (k in K) {
    x2 <- a + 2^-k * b
    if (all(x2 == a)) stop(sprintf("%s: x2_k%d equals a on every row", family, k))
    da[[sprintf("x2_k%d", k)]] <- x2
  }
  emit(sprintf("glm_band_a_%s", family), da)
  emit(sprintf("glm_band_d_%s", family), cbind(base, x2 = a))
  if (family == "gamma") next

  u <- rnorm(9)
  d <- numeric(length(BLOCK))
  first <- c(1:3, 7:9, 13:15)
  d[first] <- u
  d[first + 3] <- -u

  db <- base
  db$y[BLOCK] <- 1L
  for (wt in list(rep(1, length(BLOCK)), w[BLOCK])) {
    t <- wt * d * db$y[BLOCK]
    if (any(t[first] + t[first + 3] != 0) || any(t[-c(first, first + 3)] != 0)) {
      stop(sprintf("%s: sum(d * y) or sum(w * d * y) is not 0 on the block", family))
    }
  }
  x2 <- a
  x2[BLOCK] <- a[BLOCK] + d
  db$x2 <- x2
  emit(sprintf("glm_band_b_%s", family), offset_columns(family, db, x2))

  dc <- base
  dd <- if (family == "poisson") abs(d) else d
  dc$y[BLOCK] <- if (family == "poisson") 0L else as.integer(d > 0)
  x2 <- a
  x2[BLOCK] <- a[BLOCK] + dd
  dc$x2 <- x2
  # Read off the column the fits see, not d: separation must hold for x2 - a
  # after rounding.
  v <- x2[BLOCK] - a[BLOCK]
  yc <- dc$y[BLOCK]
  separates <- if (family == "poisson") all(yc == 0 & v >= 0) && any(v > 0) else
    all(v[yc == 1] > 0) && all(v[yc == 0] <= 0) && any(v != 0)
  if (!separates) stop(sprintf("%s: design c's block outcomes do not separate along x2 - a", family))
  emit(sprintf("glm_band_c_%s", family), offset_columns(family, dc, x2))
}
