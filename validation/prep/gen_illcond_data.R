#!/usr/bin/env Rscript
# The two ill-conditioned-but-computable LMM designs the crate fits and flags,
# emitted as committed CSVs -> ../data/simulated/<name>.csv, the same directory
# every other rung uses. Run standalone with
# `Rscript validation/prep/gen_illcond_data.R`, or as the fourth of the prep
# scripts `run.sh --prep` invokes (alongside export_data.R for rungs 1-28,
# gen_weights_data.R for 29-43 and gen_large_theta_data.R for 44-45).
#
# WHY THESE EXIST. Both designs are built in Rust inside src/fit/lmm_tests.rs and
# both used to be discarded -- one NaN-filled, the other silently reduced by a
# column -- by a rank guard whose statistic measured column SCALE rather than
# collinearity. They are fitted now, so for the first time there is a reference
# to fit them against: lme4 fits both full designs. The reference values frozen
# in those tests come from lmer() on the CSVs this script writes.
#
#   sim_dynrange_lmm       y ~ 1 + u + w + (1|g)      no collinearity anywhere;
#                          `u` is a CLUSTER-LEVEL column at scale 3e-7. Every
#                          per-column pivot ratio is O(1) -- the columns are
#                          mutually distinguishable to full precision -- and the
#                          old min/max L-diagonal statistic threw the whole fit
#                          away purely over a choice of units.
#   sim_entangled_pair_lmm y ~ 1 + t + v + z + (1|g)  v = t*(1 + 3e-6*(-1)^i),
#                          near-collinear with t but four orders clear of
#                          ALIAS_EPS, so nothing is redundant and no column may
#                          be dropped. t and v are not separately identified to
#                          any useful precision and BOTH engines say so, each
#                          reporting |beta| ~ 3.8e7 with a standard error of the
#                          same size.
#
# BIT-EXACTNESS IS THE POINT, and it is why this script does not use write.csv.
# The oracle is only an oracle if lme4 fits the SAME doubles the Rust test
# builds. Two things secure that:
#
#   1. The generator arithmetic below mirrors the Rust builders operation for
#      operation. The 16-bit LCG (s <- (75s + 74) mod 65537, value s/65537 - 0.5)
#      keeps every intermediate below 2^53, so the stream is exact in f64; the
#      within-cluster mean is accumulated in a plain double loop rather than with
#      sum(), which accumulates in long double on x86 and would round
#      differently. Verified 2026-08-01: every one of the 9000 doubles across the
#      two designs is bit-identical to the Rust builders' output.
#   2. Values are written with sprintf("%.17g"), not write.csv's 15 significant
#      digits, so read.csv recovers the identical doubles. Verified by
#      round-tripping every column through read.csv and comparing with
#      identical(). At 15 digits it does not round-trip, and on a design whose
#      smallest pivot ratio is ~9e-12 that loss is not academic.
#
# Re-running over the committed CSVs is byte-identical -- no seeds, no RNG, just
# the LCG -- which is what makes it safe on the `--prep` path, since `--prep`
# implies `--oracles` and any byte change would invalidate the frozen references.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), "..")) # nolint
out_dir <- file.path(suite_dir, "data", "simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# src/fit/lmm_tests.rs::gap_a_stream. Local to these designs -- the crate's other
# fixtures use a different, 64-bit generator.
stream <- function(k) {
  s <- 1
  out <- numeric(k)
  for (i in seq_len(k)) {
    s <- (75 * s + 74) %% 65537
    out[i] <- s / 65537 - 0.5
  }
  out
}

# Full precision, one column at a time, so the CSV round-trips exactly. `g` stays
# an integer label; it is read as a factor by every consumer.
emit <- function(name, df) {
  out <- df
  for (nm in names(df)) {
    if (nm != "g") out[[nm]] <- sprintf("%.17g", df[[nm]])
  }
  path <- file.path(out_dir, paste0(name, ".csv"))
  write.csv(out, path, row.names = FALSE, quote = FALSE)
  back <- read.csv(path)
  for (nm in names(df)) {
    if (nm != "g" && !identical(back[[nm]], df[[nm]])) {
      stop(sprintf("%s: column %s does not round-trip through read.csv", name, nm))
    }
  }
  cat(sprintf("wrote %-24s  %4d rows x %d cols  (round-trip exact)\n",
              name, nrow(df), ncol(df)))
}

# --- sim_dynrange_lmm: pure dynamic range, nothing collinear -----------------
# Mirrors src/fit/lmm_tests.rs::lmm_pure_dynamic_range_design_fits_in_full.
# `u` enters the response at unit scale (beta_u = 2 on u/c), so the SIGNAL is
# well-conditioned even though the COLUMN is tiny -- the estimate must land near
# 2/c = 6.67e6 with a standard error of its own order, and that huge SE is the
# correct report on a column whose entries are 3e-7, not a defect.
dynrange <- function() {
  jn <- 25; m <- 40; cc <- 3e-7; s_scale <- 20; tau <- 16; sigma <- 1
  n <- jn * m
  g <- stream(jn)
  h <- stream(n)
  y <- numeric(n); u <- numeric(n); w <- numeric(n); gid <- integer(n)
  for (j in 0:(jn - 1)) {
    u_j <- cc * ((j + 1) / jn)
    b_j <- tau * g[j + 1]
    for (i in 0:(m - 1)) {
      r <- j * m + i
      w_i <- s_scale * (i / m - (m - 1) / (2 * m))
      y[r + 1] <- 1 + 2 * (u_j / cc) + 0.5 * w_i + b_j + sigma * h[r + 1]
      u[r + 1] <- u_j
      w[r + 1] <- w_i
      gid[r + 1] <- j
    }
  }
  data.frame(y = y, u = u, w = w, g = gid)
}

# --- sim_entangled_pair_lmm: t and v distinguishable but badly entangled -----
# Mirrors src/fit/lmm_tests.rs::build_gap_a_salvage_design(with_exact_alias =
# FALSE). The exactly-dependent fifth column that builder can add (s = 1 + z) is
# NOT emitted: it is dropped by the alias gate before the solver runs, so the fit
# it produces is this four-column fit and lme4 would need a separate spec to say
# anything different.
#
# The pair sits in the WITHIN-cluster block deliberately. V^-1 downdates the
# cluster-level block with per-cluster outer products, and on a near-collinear
# CLUSTER-LEVEL pair that downdate cancels the pivot away entirely, so such a
# design is not a witness for this branch at all.
entangled_pair <- function() {
  jn <- 25; m <- 40; s_scale <- 20; tau <- 16; sigma <- 1
  d <- 3e-6; rho <- 1e-5
  n <- jn * m
  s_small <- s_scale * rho
  g <- stream(jn)
  # One stream, split: h[1..n] shapes `t`, h[(n+1)..2n] is the residual noise.
  # Reusing the same slice for both would make the noise collinear with `t` and
  # drive sigma-hat^2 to zero.
  h <- stream(2 * n)
  y <- numeric(n); tt <- numeric(n); v <- numeric(n); z <- numeric(n)
  gid <- integer(n)
  for (j in 0:(jn - 1)) {
    b_j <- tau * g[j + 1]
    # Plain double accumulation, matching Rust's sequential Iterator::sum. R's
    # sum() uses a long double accumulator on x86 and lands elsewhere.
    acc <- 0
    for (k in (j * m):(j * m + m - 1)) acc <- acc + h[k + 1]
    t_bar <- acc / m
    for (i in 0:(m - 1)) {
      r <- j * m + i
      # z: the ramp. t: a DIFFERENT within-cluster mean-zero pattern, so t and z
      # are not collinear with each other -- only t and v are.
      z_i <- s_scale * (i / m - (m - 1) / (2 * m))
      t_i <- s_small * (h[r + 1] - t_bar)
      v_i <- t_i * (1 + d * if (i %% 2 == 0) 1 else -1)
      y[r + 1] <- 1 + (1 / s_small) * t_i + 0.5 * z_i + b_j + sigma * h[n + r + 1]
      tt[r + 1] <- t_i
      v[r + 1] <- v_i
      z[r + 1] <- z_i
      gid[r + 1] <- j
    }
  }
  data.frame(y = y, t = tt, v = v, z = z, g = gid)
}

emit("sim_dynrange_lmm", dynrange())
emit("sim_entangled_pair_lmm", entangled_pair())
