#!/usr/bin/env Rscript
# Simulated datasets for the WEIGHTS parity suite -> data_simulated/<rung>.csv.
# Run ONCE (or via run.sh --prep); the CSVs are committed and are the neutral
# input every engine reads. Fixed seeds; the CSV (not the seed) is the artifact.
#
# Weights are a plain `w` column. Non-integer weights are lognormal (mean ~1);
# the aggregated-binomial rungs instead carry integer trial counts
# (incidence/size, the main harness's standard lowering). Gaussian core rungs
# draw heteroskedastic errors (sd ~ 1/sqrt(w)) so the weights are the TRUE
# precision weights and the reference SEs are well-behaved; the pathological
# WLS rungs keep homoskedastic errors -- there the stress is the weight
# pattern itself, and 1/sqrt(1e-12)-scaled responses would add a second,
# unattributable pathology on top.

suppressMessages(library(MASS))  # mvrnorm (slope-block RE draws), rnegbin

weights_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
out_dir <- file.path(weights_dir, "data_simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

emit <- function(name, df) {
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("wrote %-20s  %4d rows x %d cols\n", name, nrow(df), ncol(df)))
}

lognormal_w <- function(n) rlnorm(n, meanlog = 0, sdlog = 0.5)

# --- W1 wls_basic: gaussian fixed-only, lognormal non-integer weights ---------
set.seed(20260720)
n <- 200
w <- lognormal_w(n)
x1 <- rnorm(n); x2 <- rnorm(n)
y <- 1.0 + 0.5 * x1 - 0.3 * x2 + rnorm(n, sd = 0.8 / sqrt(w))
emit("wls_basic", data.frame(y = y, x1 = x1, x2 = x2, w = w))

# --- W2 glm_binomial_agg: binomial GLM, integer trial counts ------------------
set.seed(20260721)
n <- 150
x <- rnorm(n)
size <- sample(5:20, n, replace = TRUE)
incidence <- rbinom(n, size, plogis(0.3 + 0.6 * x))
emit("glm_binomial_agg", data.frame(incidence = incidence, size = size, x = x))

# --- W3 glm_poisson: Poisson GLM (log), lognormal weights ---------------------
set.seed(20260722)
n <- 200
w <- lognormal_w(n)
x <- rnorm(n)
grp <- factor(sample(c("a", "b"), n, replace = TRUE))
y <- rpois(n, exp(0.4 + 0.5 * x + 0.3 * (grp == "b")))
emit("glm_poisson", data.frame(y = y, x = x, grp = grp, w = w))

# --- W4 glm_gamma: Gamma GLM (log), lognormal weights -------------------------
set.seed(20260723)
n <- 200
w <- lognormal_w(n)
x <- rnorm(n)
grp <- factor(sample(c("a", "b"), n, replace = TRUE))
mu <- exp(0.5 + 0.6 * x + 0.4 * (grp == "b"))
y <- rgamma(n, shape = 2, scale = mu / 2)   # E[y] = mu
emit("glm_gamma", data.frame(y = y, x = x, grp = grp, w = w))

# --- W5 glm_nb: NB GLM (log), lognormal weights -------------------------------
# n = 300: theta-hat is the noisiest quantity here; more rows keep the two
# engines' alternating theta loops on the same optimum.
set.seed(20260724)
n <- 300
w <- lognormal_w(n)
x <- rnorm(n)
grp <- factor(sample(c("a", "b"), n, replace = TRUE))
y <- MASS::rnegbin(n, mu = exp(0.5 + 0.5 * x + 0.3 * (grp == "b")), theta = 1.5)
emit("glm_nb", data.frame(y = y, x = x, grp = grp, w = w))

# --- W6 lmm_intercept: (1|g), lognormal weights -------------------------------
set.seed(20260725)
n_g <- 30; per <- 10; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
w <- lognormal_w(n)
x <- rnorm(n)
y <- 1.0 + 0.6 * x + rnorm(n_g, sd = 0.9)[g] + rnorm(n, sd = 0.7 / sqrt(w))
emit("lmm_intercept", data.frame(y = y, x = x, g = g, w = w))

# --- W7 lmm_slope: (1 + x | g), lognormal weights -----------------------------
set.seed(20260726)
n_g <- 24; per <- 12; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
w <- lognormal_w(n)
x <- rnorm(n)
S <- diag(c(1.1, 0.6)) %*% matrix(c(1, 0.3, 0.3, 1), 2) %*% diag(c(1.1, 0.6))
b <- MASS::mvrnorm(n_g, mu = c(0, 0), Sigma = S)
y <- 1.0 + 0.5 * x + b[g, 1] + b[g, 2] * x + rnorm(n, sd = 0.7 / sqrt(w))
emit("lmm_slope", data.frame(y = y, x = x, g = g, w = w))

# --- W8 lmm_crossed: two crossed slope-carrying groupings (sparse path) -------
# Slopes on BOTH groupings on purpose (sim_slope_extra's rationale): a
# slope-carrying crossed extra is what routes the design to the sparse kernel;
# g1 has more levels than g2 so lme4's VarCorr order (descending levels)
# matches glmm's [primary | extra] block order positionally in compare.R.
set.seed(20260727)
n1 <- 24; n2 <- 16; per <- 15; n <- n1 * per
g1 <- factor(rep(seq_len(n1), each = per))
g2 <- factor(sample(seq_len(n2), n, replace = TRUE))
w <- lognormal_w(n)
x <- rnorm(n)
S1 <- diag(c(1.1, 0.6)) %*% matrix(c(1, 0.3, 0.3, 1), 2) %*% diag(c(1.1, 0.6))
S2 <- diag(c(0.9, 0.5)) %*% matrix(c(1, 0.2, 0.2, 1), 2) %*% diag(c(0.9, 0.5))
b1 <- MASS::mvrnorm(n1, mu = c(0, 0), Sigma = S1)
b2 <- MASS::mvrnorm(n2, mu = c(0, 0), Sigma = S2)
y <- 1.0 + 0.5 * x + b1[g1, 1] + b1[g1, 2] * x + b2[g2, 1] + b2[g2, 2] * x +
     rnorm(n, sd = 0.7 / sqrt(w))
emit("lmm_crossed", data.frame(y = y, x = x, g1 = g1, g2 = g2, w = w))

# --- W9 glmm_poisson: Poisson GLMM (1|g), dense path, lognormal weights -------
set.seed(20260728)
n_g <- 15; per <- 20; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
w <- lognormal_w(n)
x <- rnorm(n)
y <- rpois(n, exp(0.4 + 0.4 * x + rnorm(n_g, sd = 0.5)[g]))
emit("glmm_poisson", data.frame(y = y, x = x, g = g, w = w))

# --- W10 glmm_binomial: aggregated binomial GLMM, 7 crossed extras (sparse) ---
# sim_sparse_binomial's over-count recipe (7 intercept extras > cap of 6 routes
# the non-Gaussian design to Solver::Sparse); trial counts are the weights.
set.seed(20260729)
n <- 240; n_g1 <- 12; n_c <- 8
g1 <- factor(sample(seq_len(n_g1), n, replace = TRUE))
cs <- lapply(1:7, function(k) factor(sample(seq_len(n_c), n, replace = TRUE)))
x <- rnorm(n)
size <- sample(5:20, n, replace = TRUE)
sd_c <- c(0.50, 0.45, 0.40, 0.50, 0.40, 0.45, 0.35)
eta <- 0.2 + 0.5 * x + rnorm(n_g1, sd = 0.8)[g1]
for (k in 1:7) eta <- eta + rnorm(n_c, sd = sd_c[k])[cs[[k]]]
incidence <- rbinom(n, size, plogis(eta))
d <- data.frame(incidence = incidence, size = size, x = x, g1 = g1)
for (k in 1:7) d[[paste0("c", k)]] <- cs[[k]]
emit("glmm_binomial", d)

# --- P1 path_extreme_range: WLS, weights spanning 1e-6 .. 1e6 -----------------
# Stresses sqrt(w)-scaled Gram assembly / digit loss. Homoskedastic errors
# (header note): the weight range IS the pathology under test.
set.seed(20260730)
n <- 200
w <- 10^runif(n, -6, 6)
x <- rnorm(n)
y <- 1.0 + 0.5 * x + rnorm(n, sd = 0.8)
emit("path_extreme_range", data.frame(y = y, x = x, w = w))

# --- P2 path_near_zero: WLS, a block of rows with w ~ 1e-12 -------------------
# Effective row deletion; fit.R's dropped-rows gate checks beta with the block
# kept vs removed. The block's y is drawn from a DIFFERENT slope so silently
# leaking those rows into the fit cannot cancel out.
set.seed(20260731)
n <- 200; n_zero <- 40
w <- lognormal_w(n)
zero <- seq_len(n_zero)                      # first 40 rows: near-zero block
w[zero] <- 1e-12 * runif(n_zero, 0.5, 1.5)
x <- rnorm(n)
y <- 1.0 + 0.5 * x + rnorm(n, sd = 0.8)
y[zero] <- 5.0 - 2.0 * x[zero] + rnorm(n_zero, sd = 0.8)
emit("path_near_zero", data.frame(y = y, x = x, w = w))

# --- P3 path_huge_int: LMM intercept, integer weights in the thousands --------
# Stresses the deviance/loglik constant accumulation (-sum log w, weighted
# saturated terms) at aggregation scale.
set.seed(20260732)
n_g <- 20; per <- 10; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
w <- sample(1000:5000, n, replace = TRUE)
x <- rnorm(n)
y <- 1.0 + 0.6 * x + rnorm(n_g, sd = 0.9)[g] + rnorm(n, sd = 25 / sqrt(w))
emit("path_huge_int", data.frame(y = y, x = x, g = g, w = w))

# --- P4 path_dominant: WLS, one observation carries 99% of sum(w) -------------
# Leverage concentration: w[1]/sum(w) = 0.99 exactly.
set.seed(20260733)
n <- 100
w <- rep(1.0, n)
w[1] <- 99 * (n - 1)
x <- rnorm(n)
y <- 1.0 + 0.5 * x + rnorm(n, sd = 0.8)
emit("path_dominant", data.frame(y = y, x = x, w = w))

# --- U1 all_ones: LMM slope, w identically 1 (identity gate in fit.rs) --------
set.seed(20260734)
n_g <- 20; per <- 10; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
x <- rnorm(n)
S <- diag(c(1.0, 0.6)) %*% matrix(c(1, 0.25, 0.25, 1), 2) %*% diag(c(1.0, 0.6))
b <- MASS::mvrnorm(n_g, mu = c(0, 0), Sigma = S)
y <- 1.0 + 0.5 * x + b[g, 1] + b[g, 2] * x + rnorm(n, sd = 0.7)
emit("all_ones", data.frame(y = y, x = x, g = g, w = rep(1L, n)))
