#!/usr/bin/env Rscript
# The scale-variation GLM designs, emitted as committed CSVs ->
# ../data/simulated/<name>.csv, the same directory every other simulated rung
# uses. Run standalone with `Rscript validation/prep/gen_scale_data.R`, or as one
# of the prep scripts `run.sh --prep` invokes.
#
# WHY THESE EXIST. The GLM/IRLS divergence guard used to bound |beta|, an
# absolute bound on a quantity that is not a property of the model: multiplying a
# predictor column by 1000 divides its coefficient by 1000 and changes nothing
# else -- same fitted values, same deviance, same conditioning, same iteration
# count. The guard therefore accepted or rejected the same model depending on the
# caller's units. It now bounds max|eta|, which does not move under rescaling.
# These designs are the reference side of that claim:
#
#   sim_scale_logit      y ~ x, y ~ x_small, y ~ x_big   ONE logistic fit in
#                        three unit systems (x_small = x/1000, x_big = x*1000).
#                        stats::glm has no coefficient cap and fits all three
#                        identically; the three goldens pin that glmm does too.
#   sim_scale_sep        y = 1[x > 0]                    complete separation.
#                        Both engines must REJECT. glm.fit rejects by running out
#                        of iterations, not by bounding beta, and returns large
#                        coefficients with two warnings; the claim gated here is
#                        the flag, not the number -- both engines stop at an
#                        arbitrary point on a path to infinity.
#   sim_scale_gamma_inv  y ~ x, Gamma(inverse), mu ~ 0.01 so eta = 1/mu ~ 100.
#                        A legitimate fit whose eta is far above the threshold,
#                        which is why the guard skips this family/link pair.
#
# ROUND-TRIP EXACTNESS. Values are written with sprintf("%.17g"), not write.csv's
# 15 significant digits, and every column is read back and compared with
# identical(). The Rust tests read these same CSVs with include_str!, so the two
# engines must see the same doubles. (Note that R's as.numeric is not correctly
# rounded on near-halfway decimal strings, so a CSV cannot pin doubles across
# languages to the last bit; these rungs are gated at tol.R's 1e-3 beta band, far
# above that, so it does not bite here.)
#
# No RNG: a 16-bit LCG, the same generator gen_illcond_data.R uses, keeps every
# intermediate below 2^53 so the stream is exact in f64 and re-running over the
# committed CSVs is byte-identical.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..")) # nolint
out_dir <- file.path(suite_dir, "data", "simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Mirrors gen_illcond_data.R::stream -- s <- (75s + 74) mod 65537, value
# s/65537 - 0.5, uniform on (-0.5, 0.5).
stream <- function(k) {
  s <- 1
  out <- numeric(k)
  for (i in seq_len(k)) {
    s <- (75 * s + 74) %% 65537
    out[i] <- s / 65537 - 0.5
  }
  out
}

emit <- function(name, df) {
  out <- df
  for (nm in names(df)) out[[nm]] <- sprintf("%.17g", df[[nm]])
  path <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(out, path, row.names = FALSE, quote = FALSE)
  back <- read.csv(path)
  for (nm in names(df)) {
    # A 0/1-only column round-trips through read.csv as integer, not double;
    # coerce before comparing -- the value round-trips exactly, only R's
    # storage class differs, and it is the value this check is for.
    if (!identical(as.numeric(back[[nm]]), df[[nm]])) {
      stop(sprintf("%s: column %s does not round-trip through read.csv", name, nm))
    }
  }
  cat(sprintf("wrote %-22s  %4d rows x %d cols  (round-trip exact)\n",
              name, nrow(df), ncol(df)))
}

n <- 200

# --- sim_scale_logit: one model, three unit systems --------------------------
# x spans +/-2 so the slope is well determined; y is a deterministic Bernoulli
# draw at p = plogis(-0.3 + 2x), which gives overlap in both directions (no
# separation). x_small and x_big are the SAME column in different units.
scale_logit <- function() {
  u <- stream(2 * n)
  x <- 4 * u[seq_len(n)]
  p <- 1 / (1 + exp(-(-0.3 + 2 * x)))
  y <- as.numeric((u[n + seq_len(n)] + 0.5) < p)
  data.frame(y = y, x = x, x_small = x / 1000, x_big = x * 1000)
}

# --- sim_scale_sep: complete separation --------------------------------------
scale_sep <- function() {
  x <- 4 * stream(n)
  data.frame(y = as.numeric(x > 0), x = x)
}

# --- sim_scale_gamma_inv: honest large eta -----------------------------------
# eta = 100 - 20x exactly, so mu ~ 0.01 and eta ~ 100 -- more than three times
# the divergence threshold, and entirely legitimate.
scale_gamma_inv <- function() {
  u <- stream(2 * n)
  x <- u[seq_len(n)]
  eta <- 100 - 20 * x
  y <- (1 + 0.1 * u[n + seq_len(n)]) / eta
  data.frame(y = y, x = x)
}

emit("sim_scale_logit", scale_logit())
emit("sim_scale_sep", scale_sep())
emit("sim_scale_gamma_inv", scale_gamma_inv())
