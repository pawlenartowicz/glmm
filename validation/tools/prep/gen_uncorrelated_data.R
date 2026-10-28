#!/usr/bin/env Rscript
# The data of the uncorrelated random-effect goldens (the `m3_goldens` rows
# with `||`, `(0 + x | g)` or several terms on one grouping): two simulated
# designs -> ../../data/simulated/<name>.csv, and two empirical CSVs that add
# columns to committed ones -> ../../data/empirical/<name>.csv.
#
# RUN ONCE; THE CSVs ARE COMMITTED. The CSV, not the seed, is the artifact: the
# goldens are computed on these exact bytes, so a re-run must reproduce every
# file byte for byte, and a design is never changed in place.
# Run from the repository root:
#   Rscript validation/tools/prep/gen_uncorrelated_data.R
# Base R (stats) only. The simulated files are written with
# write.csv(..., row.names = FALSE).
# RNG: R 4.5.3's default kinds, RNGkind() = Mersenne-Twister, Inversion,
# Rejection; a profile that changes them changes the CSVs.
# Correlated random effects are drawn as cbind(z1, z2) %*% chol(Sigma), with
# z1 = rnorm(K) drawn before z2 = rnorm(K), K the number of groups (the rule of
# gen_inference_data.R).
#
# THE DESIGNS. Draws are made in the order listed in each block.
#
# sim_uncorr_rho (set.seed(20261020)), columns g, x, y; 240 rows.
#   Groups g01..g30 with 8 rows each. x ~ N(0, 1) per row; random (b0, b1)
#   per group with standard deviations 1.0 and 0.5, correlation 0.8;
#   eps ~ N(0, 0.5^2); y = 1 + 0.5x + b0 + b1 x + eps. The case where the
#   diagonal fit differs from the correlated one.
#
# sim_uncorr_boundary (set.seed(20261024)), columns g, x, y; 200 rows.
#   Groups g01..g20 with 10 rows each. x ~ N(0, 1) per row; b0 ~ N(0, 1) per
#   group; no slope variance; eps ~ N(0, 1); y = 1 + 0.5x + b0 + eps. The case
#   where a variance of the diagonal fit sits on the boundary.
#
# sleepstudy_d2 (no draw): sleepstudy.csv plus D2 = (Days - 4.5)^2, a second
#   slope for a grouping with two terms that `||` cannot write.
#
# Machines_dummy (no draw): Machines.csv plus 0/1 columns MachineB and
#   MachineC (Machine == "B", Machine == "C"), so lme4 and glmmTMB fit a
#   diagonal block over every column of the factor through numeric columns.
#
# The two empirical files are built from the committed CSVs' text, the new columns
# appended to each line, so the existing columns keep their bytes.
#
# SEED CHOICE, made in a scratch R session (R 4.5.3, lme4 2.0.6, reformulas
# 0.4.4) before this script's first run, on the design code below. Seeds are
# tried from the first value upward, at most 20.
#   sim_uncorr_rho, from 20261020: accept when lmer(y ~ x + (x | g),
#     REML = TRUE) has isSingular(m) FALSE and |corr| >= 0.5 in VarCorr(m),
#     and lmer(y ~ x + (x || g), REML = TRUE) has isSingular(m) FALSE.
#     20261020 accepted (first tried; no seed rejected): correlated fit theta
#     (1.70569, 0.683999, 0.548339), correlation 0.780234; diagonal fit theta
#     (1.69662, 0.877146).
#   sim_uncorr_boundary, from 20261021: accept when lmer(y ~ x + (x || g),
#     REML = TRUE) has getME(m, "theta") = (> 0, 0).
#     20261024 accepted; rejected 20261021 (theta (0.562849, 3.75163e-08)),
#     20261022 (theta (1.62344, 0.245037)), 20261023 (theta (0.673646,
#     2.90791e-05)). Accepted theta (1.15397, 0).

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", "..")) # nolint
emp_dir <- file.path(suite_dir, "data", "empirical")
sim_dir <- file.path(suite_dir, "data", "simulated")

write_csv <- function(df, dir, name) {
  write.csv(df, file.path(dir, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("wrote %-20s  %4d rows x %d cols\n", name, nrow(df), ncol(df)))
}

# Writes `from`'s lines with the columns of `cols` (a named list) appended.
# The values are integers or exact binary fractions, so as.character() is
# their exact decimal form.
append_columns <- function(from, name, cols) {
  lines <- readLines(file.path(emp_dir, paste0(from, ".csv")))
  stopifnot(all(lengths(cols) == length(lines) - 1L))
  header <- paste0(lines[1], paste0(",\"", names(cols), "\"", collapse = ""))
  body <- do.call(paste, c(list(lines[-1]), lapply(cols, as.character), sep = ","))
  writeLines(c(header, body), file.path(emp_dir, paste0(name, ".csv")))
  cat(sprintf("wrote %-20s  %4d rows, added %s\n", name, length(body),
              paste(names(cols), collapse = ", ")))
}

# --- sim_uncorr_rho ------------------------------------------------------------
re2 <- function(k, sd1, sd2, rho) {
  Sigma <- matrix(c(sd1^2, rho * sd1 * sd2, rho * sd1 * sd2, sd2^2), 2, 2)
  z1 <- rnorm(k)
  z2 <- rnorm(k)
  cbind(z1, z2) %*% chol(Sigma)
}

set.seed(20261020)
g_ids <- sprintf("g%02d", 1:30)
g <- rep(g_ids, each = 8)
gi <- match(g, g_ids)
x <- rnorm(240)
b <- re2(30, 1.0, 0.5, 0.8)
y <- 1 + 0.5 * x + b[gi, 1] + b[gi, 2] * x + rnorm(240, 0, 0.5)
write_csv(data.frame(g = g, x = x, y = y), sim_dir, "sim_uncorr_rho")

# --- sim_uncorr_boundary -------------------------------------------------------
set.seed(20261024)
g_ids <- sprintf("g%02d", 1:20)
g <- rep(g_ids, each = 10)
gi <- match(g, g_ids)
x <- rnorm(200)
b0 <- rnorm(20, 0, 1)
y <- 1 + 0.5 * x + b0[gi] + rnorm(200, 0, 1)
write_csv(data.frame(g = g, x = x, y = y), sim_dir, "sim_uncorr_boundary")

# --- sleepstudy_d2 -------------------------------------------------------------
days <- read.csv(file.path(emp_dir, "sleepstudy.csv"))$Days
append_columns("sleepstudy", "sleepstudy_d2", list(D2 = (days - 4.5)^2))

# --- Machines_dummy ------------------------------------------------------------
machine <- read.csv(file.path(emp_dir, "Machines.csv"), stringsAsFactors = FALSE)$Machine
stopifnot(setequal(machine, c("A", "B", "C")))
append_columns("Machines", "Machines_dummy",
               list(MachineB = as.integer(machine == "B"), MachineC = as.integer(machine == "C")))
