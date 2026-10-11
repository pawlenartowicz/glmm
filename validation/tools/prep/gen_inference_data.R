#!/usr/bin/env Rscript
# The shared input data of the inference references (the `data` CSVs of the
# manifest's `inference_references` entries): Dyestuff2 from lme4's bundled copy
# -> ../../data/empirical/Dyestuff2.csv, and seven simulated designs ->
# ../../data/simulated/<name>.csv, the same directories every rung uses.
#
# RUN ONCE; THE CSVs ARE COMMITTED. The CSV, not the seed, is the artifact: the
# inference references are computed on these exact bytes, so a re-run must
# reproduce every file byte for byte, and a design is never changed in place.
# Run from the repository root:
#   Rscript validation/tools/prep/gen_inference_data.R
# Base R (stats) only, plus lme4's bundled Dyestuff2 (read with data(), lme4 is
# not attached). Every file is written with write.csv(..., row.names = FALSE).
# RNG: R 4.5.3's default kinds, RNGkind() = Mersenne-Twister, Inversion,
# Rejection (the last is sample()'s); a profile that changes them changes the CSVs.
# Correlated random effects are drawn as cbind(z1, z2) %*% chol(Sigma), with
# z1 = rnorm(K) drawn before z2 = rnorm(K), K the number of groups.
#
# Dyestuff2 has no rung and no `datasets` row, so export_data.R (which owns
# Dyestuff.csv and rewrites every CSV it owns on each run) never reaches it.
# It is the singular case: its REML random-intercept variance is exactly 0.
#
# THE DESIGNS. Draws are made in the order listed in each block.
#
# sim_factorial (set.seed(20261008)), columns subj, g, h, x, y; 120 rows.
#   18 subjects s01..s18 with row counts 4,5,5,6,6,6,6,7,7,7,7,7,7,8,8,8,8,8.
#   Cells g in {a, b, c} x h in {p, q} with fixed counts a.p 30, a.q 12,
#   b.p 18, b.q 24, c.p 10, c.q 26, assigned to rows by one sample() of the
#   120 cell labels. Then x ~ N(0, 1) per row. Mean
#     10 + 1.0[g=b] - 0.5[g=c] + 0.8[h=q] + 0.5x + 0.6[b:q] - 0.4[c:q]
#        + 0.3[g=b]x - 0.2[g=c]x.
#   Then random (1 + x | subj): standard deviations 1.0 and 0.4, correlation
#   0.3. Then the residual, standard deviation 1.0.
#
# sim_factorial_bern (set.seed(20261009)), columns subj, g, h, x, y; 120 rows.
#   The subj, g, h, x of sim_factorial (reused in the same run, not redrawn).
#     eta = -0.3 + 0.6[g=b] - 0.4[g=c] + 0.5[h=q] + 0.4x + 0.5[b:q] - 0.3[c:q]
#           + 0.3[g=b]x - 0.2[g=c]x + b0 + b1 x,
#   random (b0, b1) per subj with standard deviations 0.8 and 0.3, correlation
#   0.3; then y = rbinom(120, 1, plogis(eta)).
#
# sim_factorial_empty (no draw): sim_factorial without its g = c, h = q rows
#   (94 rows).
#
# sim_rlrt_few_unbal (set.seed(20261010)), columns g, x, z, y; 32 rows.
#   Groups g1..g4 of sizes 3, 5, 9, 15. x ~ N(0, 1) per row; z ~ N(0, 1) per
#   group, constant within it; b ~ N(0, 0.7^2) per group; eps ~ N(0, 1);
#   y = 1 + 0.5x + 0.5z + b_g + eps.
#
# sim_rlrt_large_k (set.seed(20261011)), columns g, x, y; 3000 rows.
#   Groups g0001..g1000 with 3 rows each. x ~ N(0, 1); b ~ N(0, 0.5^2) per
#   group; eps ~ N(0, 1); y = 1 + 0.5x + b_g + eps.
#
# sim_corr_boundary (set.seed(20261012)), columns g, x, y; 60 rows.
#   Groups g01..g10 with 6 rows each. x ~ U(-1, 1) per row; b0 ~ N(0, 0.8^2)
#   per group; b1 = 0.5 b0 (true correlation exactly 1); eps ~ N(0, 1);
#   y = 2 + 1.0x + b0 + b1 x + eps.
#
# sim_near_singular (set.seed(20261040)), columns g, x, y; 60 rows.
#   Groups g01..g12 with 5 rows each. x ~ N(0, 1) per row; b ~ N(0, 0.15^2)
#   per group; eps ~ N(0, 1); y = 1 + 0.5x + b_g + eps.
#
# SEED CHOICE for the last two designs, made in a scratch R session (R 4.5.3,
# lme4 2.0.6, fastglmm 0.5.0) before this script's first run, on the design
# code below. A seed is accepted when both lme4 and fastglmm report what the
# design exists to exercise; seeds are tried from the first value upward, at
# most 20.
#   sim_corr_boundary, from 20261012: accept when lmer(y ~ x + (x | g),
#     REML = TRUE) has getME(m, "theta")[3] == 0 with the other two entries
#     > 0, and fastglmm on the same formula has isSingular(m) TRUE, both
#     standard deviations > 0 and |corr| >= 1 - 1e-6 in VarCorr(m).
#     20261012 accepted (first tried; no seed rejected): lme4 theta =
#     (0.402055, 0.51779, 0); fastglmm singular, standard deviations
#     (0.400404, 0.515662), correlation 1.0000000000.
#   sim_near_singular, from 20261040: accept when lmer(y ~ x + (1 | g),
#     REML = TRUE) has isSingular(m) FALSE and 0.01 < getME(m, "theta") < 0.2,
#     and fastglmm on the same formula has isSingular(m) FALSE with the
#     random-intercept standard deviation over the residual one in the same
#     interval.
#     20261040 accepted (first tried; no seed rejected): lme4 not singular,
#     theta 0.196384; fastglmm not singular, ratio 0.196384.
#
# SANITY READS on the written CSVs (read off R 4.5.3 with lme4 2.0.6 and
# lmerTest 3.2.1; observations, not gates):
#   sim_factorial: lmer(y ~ g * h + x + (1 + x | subj), REML = TRUE) is not
#     singular (theta 1.16895, -0.113481, 0.236442), no warning.
#   sim_factorial_empty: the same lmerTest fit drops one column ("fixed-effect
#     model matrix is rank deficient"), and anova(m, type = 3) prints
#     "Missing cells for: gc:hq. Interpret type III hypotheses with care."
#   sim_factorial_bern: glmer(y ~ g * h + x + (1 | subj), binomial) converges
#     without a warning or a convergence message (theta 1.0179).
#   Dyestuff2: lmer(Yield ~ 1 + (1 | Batch)) has theta == 0 (boundary
#     (singular) fit).
#   sim_rlrt_few_unbal: lmer(y ~ 1 + x + z + (1 | g)) fits without a warning
#     (theta 1.33814).
#   sim_rlrt_large_k: lmer(y ~ 1 + x + (1 | g)) fits without a warning
#     (theta 0.462482).
#   sim_corr_boundary and sim_near_singular, read back from the CSVs: the seed
#     checks above give the same values as on the in-memory draws.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", "..")) # nolint
emp_dir <- file.path(suite_dir, "data", "empirical")
sim_dir <- file.path(suite_dir, "data", "simulated")
dir.create(emp_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(sim_dir, showWarnings = FALSE, recursive = TRUE)

write_csv <- function(df, dir, name) {
  write.csv(df, file.path(dir, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("wrote %-20s  %4d rows x %d cols\n", name, nrow(df), ncol(df)))
}

# --- Dyestuff2 ---------------------------------------------------------------
data(list = "Dyestuff2", package = "lme4")
dy2 <- as.data.frame(Dyestuff2)
for (col in names(dy2)) if (is.factor(dy2[[col]])) dy2[[col]] <- factor(as.character(dy2[[col]]))
write_csv(dy2, emp_dir, "Dyestuff2")

# --- sim_factorial, sim_factorial_bern, sim_factorial_empty ------------------
# Two correlated random effects per group: cbind(z1, z2) %*% chol(Sigma).
re2 <- function(k, sd1, sd2, rho) {
  Sigma <- matrix(c(sd1^2, rho * sd1 * sd2, rho * sd1 * sd2, sd2^2), 2, 2)
  z1 <- rnorm(k)
  z2 <- rnorm(k)
  cbind(z1, z2) %*% chol(Sigma)
}

set.seed(20261008)
subj_ids <- sprintf("s%02d", 1:18)
subj <- rep(subj_ids, times = c(4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8))
cell <- sample(rep(c("a.p", "a.q", "b.p", "b.q", "c.p", "c.q"),
                   times = c(30, 12, 18, 24, 10, 26)))
g <- substr(cell, 1, 1)
h <- substr(cell, 3, 3)
x <- rnorm(120)
gb <- g == "b"; gc <- g == "c"; hq <- h == "q"
si <- match(subj, subj_ids)
mu <- 10 + 1.0 * gb - 0.5 * gc + 0.8 * hq + 0.5 * x + 0.6 * (gb & hq) - 0.4 * (gc & hq) +
  0.3 * gb * x - 0.2 * gc * x
b <- re2(18, 1.0, 0.4, 0.3)
y <- mu + b[si, 1] + b[si, 2] * x + rnorm(120, 0, 1.0)
sim_factorial <- data.frame(subj = subj, g = g, h = h, x = x, y = y)
write_csv(sim_factorial, sim_dir, "sim_factorial")

set.seed(20261009)
b <- re2(18, 0.8, 0.3, 0.3)
eta <- -0.3 + 0.6 * gb - 0.4 * gc + 0.5 * hq + 0.4 * x + 0.5 * (gb & hq) - 0.3 * (gc & hq) +
  0.3 * gb * x - 0.2 * gc * x + b[si, 1] + b[si, 2] * x
write_csv(data.frame(subj = subj, g = g, h = h, x = x, y = rbinom(120, 1, plogis(eta))),
          sim_dir, "sim_factorial_bern")

write_csv(sim_factorial[!(sim_factorial$g == "c" & sim_factorial$h == "q"), ],
          sim_dir, "sim_factorial_empty")

# --- sim_rlrt_few_unbal --------------------------------------------------------
set.seed(20261010)
g_ids <- sprintf("g%d", 1:4)
g <- rep(g_ids, times = c(3, 5, 9, 15))
gi <- match(g, g_ids)
x <- rnorm(32)
z <- rnorm(4)[gi]
b <- rnorm(4, 0, 0.7)
y <- 1 + 0.5 * x + 0.5 * z + b[gi] + rnorm(32, 0, 1)
write_csv(data.frame(g = g, x = x, z = z, y = y), sim_dir, "sim_rlrt_few_unbal")

# --- sim_rlrt_large_k ----------------------------------------------------------
set.seed(20261011)
g_ids <- sprintf("g%04d", 1:1000)
g <- rep(g_ids, each = 3)
gi <- match(g, g_ids)
x <- rnorm(3000)
b <- rnorm(1000, 0, 0.5)
y <- 1 + 0.5 * x + b[gi] + rnorm(3000, 0, 1)
write_csv(data.frame(g = g, x = x, y = y), sim_dir, "sim_rlrt_large_k")

# --- sim_corr_boundary ---------------------------------------------------------
set.seed(20261012)
g_ids <- sprintf("g%02d", 1:10)
g <- rep(g_ids, each = 6)
gi <- match(g, g_ids)
x <- runif(60, -1, 1)
b0 <- rnorm(10, 0, 0.8)
b1 <- 0.5 * b0
y <- 2 + 1.0 * x + b0[gi] + b1[gi] * x + rnorm(60, 0, 1)
write_csv(data.frame(g = g, x = x, y = y), sim_dir, "sim_corr_boundary")

# --- sim_near_singular ---------------------------------------------------------
set.seed(20261040)
g_ids <- sprintf("g%02d", 1:12)
g <- rep(g_ids, each = 5)
gi <- match(g, g_ids)
x <- rnorm(60)
b <- rnorm(12, 0, 0.15)
y <- 1 + 0.5 * x + b[gi] + rnorm(60, 0, 1)
write_csv(data.frame(g = g, x = x, y = y), sim_dir, "sim_near_singular")
