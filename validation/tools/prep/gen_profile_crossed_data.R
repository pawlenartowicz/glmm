#!/usr/bin/env Rscript
# The data of the in-crate profile test on a crossed model whose slope column
# sits far from zero (src/inference/profile_tests.rs) ->
# ../../data/simulated/profile_crossed_uncentered.csv.
#
#   profile_crossed_uncentered (set.seed(203)), columns Date, image, trial,
#     prob, val, aro, sig; 735 rows. The 23 x 32 crossing of Date and image,
#     trial = image mod 8 + 1, 735 of the 736 cells drawn without replacement.
#     prob ~ U(2, 5), val ~ N(3, 1), aro ~ N(4, 1) per row; per grouping f a
#     random intercept and a random slope on prob, independent normals, with
#     SDs (0.9, 0.05) for Date, (0.25, 0.13) for image and (0.3, 0.35) for
#     trial, intercepts drawn before slopes; eps ~ N(0, 1.27^2);
#     sig = 2 - 0.5 prob - 0.6 val + 0.9 aro + the three random effects + eps.
#     The test fits sig ~ prob + val + aro + (prob | Date) + (prob | image) +
#     (prob | trial) by ML on these columns and on prob, val, aro centered,
#     and compares the profile intervals of the image slope SD.
#
# Written with write.csv(..., row.names = FALSE, quote = FALSE). The CSV, not
# the seed, is the artifact: a re-run must reproduce it byte for byte.
# Verified 2026-11-15 on R 4.5.3 (x86_64 Linux), default RNGkind
# (Mersenne-Twister, Inversion, Rejection). Base R only.
#
#   Rscript validation/tools/prep/gen_profile_crossed_data.R [<output-dir>]
#
# The output directory defaults to validation/data/simulated.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..", ".."))
args <- commandArgs(trailingOnly = TRUE)
out_dir <- if (length(args) >= 1) args[1] else file.path(suite_dir, "data", "simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

set.seed(203)
D <- expand.grid(Date = factor(1:23), image = factor(1:32))
D$trial <- factor(as.integer(D$image) %% 8 + 1)
D <- D[sample(nrow(D), 735), ]
D$prob <- runif(735, 2, 5); D$val <- rnorm(735, 3, 1); D$aro <- rnorm(735, 4, 1)
re <- function(f, s0, s1) { u0 <- rnorm(nlevels(f), 0, s0); u1 <- rnorm(nlevels(f), 0, s1); u0[f] + u1[f] * D$prob }
D$sig <- 2 + -0.5 * D$prob - 0.6 * D$val + 0.9 * D$aro + re(D$Date, 0.9, 0.05) + re(D$image, 0.25, 0.13) + re(D$trial, 0.3, 0.35) + rnorm(735, 0, 1.27)

write.csv(D, file.path(out_dir, "profile_crossed_uncentered.csv"), row.names = FALSE, quote = FALSE)
