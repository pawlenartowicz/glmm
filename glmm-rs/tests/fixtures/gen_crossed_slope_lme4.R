#!/usr/bin/env Rscript
# Generator for the lme4 reference of `crossed_slope_fit_matches_lme4_golden`
# (src/lmm/tests.rs): the REML fit of `y ~ x1 + (1 + x1 | pid) + (1 + x1 | eid)`
# on the n = 96 dataset of `crossed_slope_golden_dataset` (8 primary clusters x 6
# crossed levels x 2 reps). It prints the test's G_* constants; the test keeps
# them inline, so nothing is written to disk.
#
# The data are rebuilt here, not read from a file: `lcg` and `dataset` below mirror
# `src/fit/common_tests.rs::lcg` and `src/lmm/tests.rs::crossed_slope_golden_dataset`
# operation for operation (change together), so lme4 fits the same doubles the
# test builds. The 64-bit LCG state is held as four 16-bit limbs, so every
# intermediate stays below 2^53 and the stream is exact in f64.
#
# Optional argument: a CSV written by `dump_crossed_slope_golden_csv`
# (src/lmm/tests.rs); the script then stops unless the rebuilt data equal it
# double for double.
#
# Run from the crate root:  Rscript tests/fixtures/gen_crossed_slope_lme4.R [csv]

suppressMessages(library(lme4))

limbs <- function(v) {  # v < 2^53, exact in f64
  out <- numeric(4)
  for (k in 1:4) { out[k] <- v %% 65536; v <- (v - out[k]) / 65536 }
  out
}
# 6364136223846793005 = 0x5851F42D4C957F2D, 1442695040888963407 = 0x14057B7EF767814F
LCG_A <- c(0x7F2D, 0x4C95, 0xF42D, 0x5851)
LCG_C <- c(0x814F, 0xF767, 0x7B7E, 0x1405)

# state <- state * A + C (mod 2^64); value ((state >> 11) / 2^53) * 2 - 1.
lcg_env <- function(seed) {
  st <- limbs(seed)
  function() {
    r <- numeric(4)
    for (i in 1:4) for (j in 1:(5 - i)) r[i + j - 1] <- r[i + j - 1] + st[i] * LCG_A[j]
    r <- r + LCG_C
    carry <- 0
    for (k in 1:4) {
      t <- r[k] + carry
      st[k] <<- t %% 65536
      carry <- (t - st[k]) / 65536
    }
    v <- st[4] * 2^37 + st[3] * 2^21 + st[2] * 2^5 + (st[1] - st[1] %% 2048) / 2048
    (v / 2^53) * 2 - 1
  }
}

dataset <- function() {
  n_prim <- 8; n_cross <- 6; n <- 96
  lcg <- lcg_env(20260629)
  u0p <- vapply(seq_len(n_prim), function(i) 0.7 * lcg(), 0)
  u1p <- vapply(seq_len(n_prim), function(i) 0.5 * lcg(), 0)
  u0e <- vapply(seq_len(n_cross), function(i) 0.6 * lcg(), 0)
  u1e <- vapply(seq_len(n_cross), function(i) 0.4 * lcg(), 0)
  x1 <- y <- numeric(n); pid <- eid <- integer(n)
  for (i in 0:(n - 1)) {
    pp <- i %% n_prim + 1
    ee <- (i %/% n_prim) %% n_cross + 1
    pid[i + 1] <- pp - 1; eid[i + 1] <- ee - 1
    x <- lcg()
    x1[i + 1] <- x
    y[i + 1] <- 1.0 + 0.8 * x + u0p[pp] + u1p[pp] * x + u0e[ee] + u1e[ee] * x + 0.5 * lcg()
  }
  data.frame(x1, y, pid, eid)
}

d <- dataset()
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 1) {
  ref <- read.csv(args[1])
  stopifnot(identical(d$x1, ref$x1), identical(d$y, ref$y),
            identical(as.numeric(d$pid), as.numeric(ref$pid)),
            identical(as.numeric(d$eid), as.numeric(ref$eid)))
  cat(sprintf("data identical to %s\n", args[1]))
}

d$pid <- factor(d$pid); d$eid <- factor(d$eid)
# optimizer = "bobyqa" (minqa) is the reference's own setting. lmer's default
# (nloptwrap) stops 2e-9 higher in the REML criterion on this design, which moves
# the variance components by up to 6e-5 relative (measured with lme4 2.0-6).
m <- lmer(y ~ x1 + (1 + x1 | pid) + (1 + x1 | eid), data = d, REML = TRUE,
          control = lmerControl(optimizer = "bobyqa"))
vc <- VarCorr(m)
b <- unname(fixef(m))
msgs <- m@optinfo$conv$lme4$messages
cat(sprintf("// R %s, lme4 %s, REML, optimizer bobyqa; convergence messages: %s\n",
            getRversion(), packageVersion("lme4"),
            if (length(msgs) == 0) "none" else paste(msgs, collapse = "; ")))
out <- c(G_BETA0 = b[1], G_BETA1 = b[2], G_SIGMA2 = sigma(m)^2,
         G_PID_V0 = vc$pid[1, 1], G_PID_V1 = vc$pid[2, 2], G_PID_COV = vc$pid[1, 2],
         G_EID_V0 = vc$eid[1, 1], G_EID_V1 = vc$eid[2, 2], G_EID_COV = vc$eid[1, 2])
for (k in names(out)) cat(sprintf("const %s: f64 = %.17g;\n", k, out[[k]]))
