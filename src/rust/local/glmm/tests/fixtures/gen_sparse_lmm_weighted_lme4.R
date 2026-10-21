#!/usr/bin/env Rscript
# Generator for `fit_sparse_lmm_weighted_matches_lme4` (src/sparse/tests.rs): the
# n = 400 weighted Gaussian design `y ~ x1 + x2 + (1 | g1) + (1 + x2 | g2)`,
# weights w_i = 1 + (i mod 3) on the 0-based row index, and lme4's REML fit of it.
# Prints the test's inline CSV (write.table's 15 significant digits, the bytes the
# test parses) and its REF_* constants. The test keeps both inline, so nothing is
# written to disk.
#
# The reference is fitted on the full-precision `d`, not on the 15-digit CSV the
# test reads.
#
# Run from the crate root:
#   Rscript tests/fixtures/gen_sparse_lmm_weighted_lme4.R

suppressMessages(library(lme4))

set.seed(2026)
n <- 400; n_g1 <- 20; n_g2 <- 15
g1 <- rep(0:(n_g1 - 1), each = n / n_g1)
g2 <- (seq_len(n) - 1) %% n_g2
x1 <- rnorm(n); x2 <- rnorm(n)
b1 <- rnorm(n_g1, sd = 1.5)
b2_int <- rnorm(n_g2, sd = 1.0); b2_slope <- rnorm(n_g2, sd = 0.8)
y <- 2 + 0.7 * x1 + 1.2 * x2 + b1[g1 + 1] + b2_int[g2 + 1] +
     b2_slope[g2 + 1] * x2 + rnorm(n, sd = 1.0)
w <- 1 + (seq_len(n) - 1) %% 3
d <- data.frame(y, x1, x2, g1, g2)

cat("// CSV (y,x1,x2,g1,g2)\n")
write.table(d, stdout(), sep = ",", row.names = FALSE, col.names = FALSE)

f <- lmer(y ~ x1 + x2 + (1 | g1) + (1 + x2 | g2), data = d,
          weights = w, REML = TRUE)
co <- summary(f)$coefficients
vc <- as.data.frame(VarCorr(f))
sd_of <- function(grp, v1, v2 = NA) {
  r <- vc$grp == grp & vc$var1 == v1 & (if (is.na(v2)) is.na(vc$var2) else vc$var2 %in% v2)
  vc$sdcor[r]
}
msgs <- f@optinfo$conv$lme4$messages
cat(sprintf("// R %s, lme4 %s, REML; convergence messages: %s\n",
            getRversion(), packageVersion("lme4"),
            if (length(msgs) == 0) "none" else paste(msgs, collapse = "; ")))
out <- c(REF_B0 = co[1, 1], REF_B1 = co[2, 1], REF_B2 = co[3, 1],
         REF_SE0 = co[1, 2], REF_SE1 = co[2, 2], REF_SE2 = co[3, 2],
         REF_SD_G1 = sd_of("g1", "(Intercept)"),
         REF_SD_G2_INT = sd_of("g2", "(Intercept)"),
         REF_SD_G2_SLOPE = sd_of("g2", "x2"),
         REF_CORR_G2 = sd_of("g2", "(Intercept)", "x2"),
         REF_REMLCRIT = REMLcrit(f))
for (k in names(out)) cat(sprintf("const %s: f64 = %.17g;\n", k, out[[k]]))
