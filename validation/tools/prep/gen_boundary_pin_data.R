#!/usr/bin/env Rscript
# The simulated dataset of the in-crate boundary-pin tests
# (src/fit/core_tests.rs) -> ../../data/simulated/<name>.csv, built with the
# boundary-probe campaign's helpers (campaigns/boundary-probe/gen/gen_common.R):
#
#   sim_poisson_precise_two_zero_slopes  Poisson log,
#     y ~ 1 + x1 + x2 + (1 + x1 + x2 | g), 20 levels g001..g020 x 10 rows,
#     x1, x2 ~ N(0, 1); intercept 14, x1 0.1, x2 -0.1; intercept SD 0.5, both
#     slope SDs 0. Columns y, x1, x2, g. Dataset 143 of a seed search over 300
#     datasets of this design (base seed 202610314, dataset seed 1617775066),
#     chosen on 2026-10-30 because, with the trough polish of that day (theta not
#     scaled), the re-pin of a polished endpoint lowered it by more than 0.5
#     deviance units both on aarch64 macOS (-0.642, the largest there) and on
#     x86_64 Linux (-0.614). Only dataset 202 also moved by more than 0.5 on
#     both; dataset 229 moved more on x86_64 alone. Its search pins both
#     slopes about 1.2 deviance units above the optimum, which
#     boundary_pin_lme4_reference.R computes with lme4.
#
# Written with write.csv(..., row.names = FALSE, quote = FALSE). The CSV, not
# the seed, is the artifact: a re-run must reproduce it byte for byte.
# Verified 2026-10-30 on R 4.5.2 (Mac arm64), default RNGkind
# (Mersenne-Twister, Inversion, Rejection).
#
#   Rscript validation/tools/prep/gen_boundary_pin_data.R [<output-dir>]
#
# The output directory defaults to validation/data/simulated.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", ".."))
source(file.path(suite_dir, "campaigns", "boundary-probe", "gen", "gen_common.R"))
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else file.path(suite_dir, "data", "simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# The seed of dataset k of n, as the campaign generators draw it.
nth_seed <- function(base, n, k) dataset_seeds(base, n)[k]

emit <- function(name, df) {
  write_dataset(df, out_dir, name)
  cat(sprintf("wrote %-36s %4d rows x %d cols\n", name, nrow(df), ncol(df)))
}

# --- sim_poisson_precise_two_zero_slopes (dataset 143 of 300) ----------------
seed <- nth_seed(202610314L, 300L, 143L)
stopifnot(seed == 1617775066L)
set.seed(seed)
lay <- layout_rows(20L, 10L)
x1 <- rnorm(lay$n)
x2 <- rnorm(lay$n)
b0 <- draw_re(20L, 0.5)[lay$gi]
eta <- 14 + b0 + 0.1 * x1 - 0.1 * x2
y <- simulate_response(eta, "poisson")
emit("sim_poisson_precise_two_zero_slopes",
     data.frame(y = y, x1 = x1, x2 = x2, g = level_labels("g", lay$gi, 3)))
