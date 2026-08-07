#!/usr/bin/env Rscript
# Simulated datasets for the large-theta-hat coverage tier (R1-R3)
# -> ../data/simulated/<rung>.csv, the same directory every other rung uses.
# Run standalone with `Rscript validation/prep/gen_large_theta_data.R`, or as the
# third of the five prep scripts `run.sh --prep` invokes (alongside
# export_data.R for rungs 1-28, gen_weights_data.R for 29-43, gen_illcond_data.R
# and gen_scale_data.R).
#
# THE STAGED CSVs ARE AUTHORITATIVE AND FROZEN, like the reference JSONs
# (RULE 0 shape): the CSV (not the seed) is the artifact. DO NOT re-run the
# R1-R3 blocks in place, including via `--prep`. Measured 2026-08-06 on this
# box (R 4.5.3, glibc 2.42, x86_64): a re-run reproduces R4 byte-identically
# but writes R1-R3 with ~1-ULP differences in scattered rnorm/rpois draws
# (78/1800, 24/600, 4/160 rows), self-consistent across reruns -- so the
# staged R1-R3 content came from a libm environment this machine no longer
# provides, and which one was not recorded. Frozen lme4 references exist for
# R1/R2, so regenerating the CSVs silently invalidates them. If regeneration
# is ever unavoidable, regenerate CSVs and references together in one step
# and record the generating R and libm versions here.
#
# Why a third prep script rather than more blocks in export_data.R: every rung
# that file emits is already backed by a frozen reference that cannot be
# regenerated, so appending to it puts the whole 1-28 corpus one editing mistake
# away from a rewritten CSV. gen_weights_data.R already set the precedent of a
# self-contained per-tier script (rungs 29-43); this follows it.
#
# EVERY sd_g below is a TUNED value, not the target theta-hat. The spec's
# targets are on the FITTED theta-hat (glmer nAGQ = 1, tolPwrss = 1e-13 -- the
# control validation/engines/lme4.R pins), and the Laplace approximation is
# biased LOW in exactly this regime: it is what these rungs exist to exercise.
# Measured at freeze, one continuous (x) + one binary (z) fixed effect throughout:
#
#   R1 Bernoulli 300x6:  sd_g 5.54 -> theta-hat 4.51  (Laplace bias -18%)
#   R2 Poisson   100x6:  sd_g 2.70 -> theta-hat 2.97  (bias -1%; the normal
#                        approximation to the Poisson mode is far better than
#                        to the Bernoulli one, hence the much smaller gap)
#   R3 binomial   20x8:  sd_g 0    -> theta-hat exactly 0 on BOTH engines

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
out_dir <- file.path(suite_dir, "data", "simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

emit <- function(name, df) {
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("wrote %-22s  %4d rows x %d cols\n", name, nrow(df), ncol(df)))
}

# --- R1 sim_binomial_bigsd: Bernoulli, scalar (1|g), theta-hat ~ 4.5 ----------
# 300 groups x 6 Bernoulli rows: MANY SMALL clusters is the half of the shape
# that matters as much as the large sd. It is the standard hard cell for Laplace
# (little per-cluster information, so the mode's normal approximation is at its
# worst) and it is the shape lme4 issue #289 / toenail (294 patients,
# theta-hat 4.708) has. At this sd most clusters are all-0 or all-1, which is
# the point: the FD Hessian's truncation error grows as theta-hat^2 and nothing
# else in the corpus sits above 1.34.
# b0 = 0.5 (not ~0) on sim_binomial_slope2's recorded rationale: a near-zero
# realized intercept turns a small absolute beta[0] gap into a spurious >1e-3
# RELATIVE gap at compare.R's beta gate. The fitted intercept here lands at
# 0.83, comfortably clear of that.
# Also fit at nAGQ = 7 and 11 as separate goldens (spec section 2, R1): Bernoulli
# keeps those clear of lme4's nAGQ > 1 logLik offset, which is zero for
# per-row trials and nonzero for the aggregated form.
set.seed(20260901)
n_g <- 300; per <- 6; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
x <- rnorm(n)
z <- rbinom(n, 1, 0.5)
b <- rnorm(n_g, sd = 5.54)
eta <- 0.5 + 0.8 * x - 0.6 * z + b[g]
emit("sim_binomial_bigsd",
     data.frame(y = rbinom(n, 1, plogis(eta)), x = x, z = z, g = g))

# --- R2 sim_poisson_bigsd: Poisson-log, scalar (1|g), theta-hat ~ 3 -----------
# The count-family counterpart of R1, Laplace only. Same fixed-effect shape
# (one continuous + one binary) so the two rungs differ in family and nothing
# else that matters. 100 groups x 6 rows.
# b0 = -0.8 balances the two things a log link at sd 2.7 trades off: a higher
# intercept pushes the top group's counts into the thousands (the +3sd group
# already peaks near 2900 here), a lower one drives the all-zero-group fraction
# up. ~57% zero rows at this setting -- the low-count regime where the
# Poisson-Laplace mode solve is actually stressed, matching sim_poisson_slope1's
# recorded rationale, while keeping the largest count finite and unremarkable.
set.seed(20260911)
n_g <- 100; per <- 6; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
x <- rnorm(n)
z <- rbinom(n, 1, 0.5)
b <- rnorm(n_g, sd = 2.70)
eta <- -0.8 + 0.5 * x - 0.4 * z + b[g]
emit("sim_poisson_bigsd", data.frame(y = rpois(n, exp(eta)), x = x, z = z, g = g))

# --- R3 sim_binomial_zerosd: aggregated binomial, (1|g) with NO variance ------
# The other end of the same axis: the grouping factor's effect is EXACTLY zero
# (no b[g] term at all), so both engines' optimizers land on the theta boundary.
# Aggregated (incidence/size) rather than Bernoulli, the cbpp/sim_sparse_binomial
# shape: it keeps R3 a different response shape from R1 for the price of nothing,
# since no AGQ golden is asked for here.
#
# THE SEED IS LOAD-BEARING, and not for the usual reproducibility reason.
# compare.R's rel_max floors its denominator at 1e-12, so an oracle that lands
# on a TINY NONZERO stddev against glmm's exact 0.0 makes the stddev gate read
# exactly 1.0 and fail -- a harness property, not a defect in the rung. At
# theta_true = 0 the boundary is only the sample maximum about half the time;
# a swept freeze check over five seeds at this size found glmer landing on
# exact 0.0 for three of them and on 1.2e-2 / 1.2e-1 for the other two, and the
# Bernoulli variant of the same design additionally produced 1.5e-8 and 3.9e-8
# (isSingular = TRUE, i.e. flagged singular yet NOT exactly zero -- precisely
# the rel_max = 1.0 trap). 20260901 is a seed where BOTH glmer and glmm return
# bit-exact 0.0. Re-seeding this block without re-running that check is how the
# freeze gets wasted.
#
# 20 groups x 8 rows, size 5..20 trials: cbpp's scale. Deliberately modest --
# power to reject theta = 0 grows with both the group count and the trials per
# group, so a larger design would make the boundary landing less reliable, not
# more informative.
set.seed(20260901)
n_g <- 20; per <- 8; n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
x <- rnorm(n)
z <- rbinom(n, 1, 0.5)
size <- sample(5:20, n, replace = TRUE)
eta <- 0.3 + 0.5 * x - 0.4 * z          # no b[g]: theta_true = 0 exactly
emit("sim_binomial_zerosd",
     data.frame(incidence = rbinom(n, size, plogis(eta)), size = size,
                x = x, z = z, g = g))

# ---------------------------------------------------------------------------
# R4: sim_sparse_binomial_bigsd -- rung 46, the SPARSE arm of the large-theta-hat
# regime. Rungs 44-45 above are both dense-routed scalar (1|g), so the sparse
# solver's behaviour at large theta-hat was untested rather than merely
# uncalibrated. This is the cross of two existing shapes: the tuned-sd machinery
# of R1/R2 with the 1-primary-plus-7-crossed-extras skeleton that
# prep/export_data.R uses for sim_sparse_poisson.
#
# WHY BERNOULLI, NOT POISSON. A Poisson design was tried first and abandoned.
# On a log link, the dynamic range of the fitted mean across groups is
# exp(+-3*theta-hat) -- at theta-hat in [3,4.5] that is e^+-11, so the largest
# counts run into the tens of thousands. Two independent problems live in that
# regime and neither is a tuning knob: (1) the sparse PIRLS cold start (u = 0)
# needed one halving more than PIRLS_MAX_HALVINGS to walk back to the mode on
# the design that reaches theta-hat(g1) ~ 3.6 (fixed separately, see
# PIRLS_MAX_HALVINGS's doc comment in src/glmm/mod.rs); (2) independently of
# that fix, the deviance-sum rounding noise at counts in the tens of thousands
# swamps the FD Hessian's step, so the standard-error agreement with the
# reference degrades with the counts' magnitude, not with any solver setting --
# a 3e-6 nudge to the converged fit moves se(beta[0]) by under 2e-5 relative at
# small counts and by ~1.9e-3 at max y ~ 27000. Shrinking the counts to fix the
# second problem costs the theta-hat target: the only free knob is the
# intercept, and pushing it low enough to keep counts small also drives the
# response toward >90% zero rows and the fitted theta-hat outside the 3.0-4.5
# band. A Bernoulli response removes the mechanism at its root instead of
# trading it off: mu is bounded in (0,1), so the IRLS weight mu(1-mu) <= 1/4
# regardless of theta-hat, the cold PIRLS start stays modest, and the deviance
# summands cannot grow without bound. Sparse-large-theta-hat Poisson coverage
# is therefore deliberately not attempted here, not a gap left for later --
# the conditioning problem is a property of the log-link large-count regime,
# not of this crate's solver.
#
# WHY SEVEN CROSSED EXTRAS AND NOT A RANDOM SLOPE. Both route the fit to the
# sparse solver, but only the all-scalar shape keeps engines/lme4.R emitting
# stddev_se (it is emitted for binomial/poisson with all-scalar RE blocks), and
# stddev_se is half of what this rung exists to gate.
#
# WHY THE EXTRAS STAY SMALL. The sparse FD Hessian builds its theta step
# per-component; one large component alongside seven small ones exercises both
# sides of that rule in a single fit.
#
# WHY 300 GROUPS OF 12 (3600 ROWS), NOT A SMALLER SKELETON. A first attempt
# reused rung 44's 120-groups-of-6 shape with sd_c = 0.4; at a theta-hat(g1)
# high enough to be interesting almost every Bernoulli row saturates (y is
# determined by which side of 0 the linear predictor falls, with little
# per-row information left), and four of the seven crossed components pinned
# at zero. 300 groups of 12 with sd_c = 0.5 keeps every component clear of the
# boundary; do not shrink the row count back.
#
# sd_g1 below is a TUNED value, not the target theta-hat -- same rule as R1/R2:
# the target is on the FITTED theta-hat under glmer nAGQ = 1, tolPwrss = 1e-13
# (the control validation/engines/lme4.R pins), and Laplace is biased low here.
#   sd_g1 4.00 -> theta-hat(g1) 3.9076; sd_c 0.5 -> theta-hat(c1..c7) all clear
#   of the boundary (0.251 .. 3.908 across the eight fitted stddevs). glmer
#   converges with no warnings and isSingular is FALSE.
# b0 = 0.5, not near zero, on sim_binomial_slope2's recorded rationale -- a
# near-zero realized intercept turns a small absolute beta[0] gap into a
# spurious relative one at compare.R's beta gate.
set.seed(20260921)
n_g1 <- 300; per <- 12; n <- n_g1 * per
n_c <- 8
g1 <- factor(rep(seq_len(n_g1), each = per))
cs <- lapply(seq_len(7), function(k) factor(sample.int(n_c, n, replace = TRUE)))
x <- rnorm(n)
z <- rbinom(n, 1, 0.5)
b1 <- rnorm(n_g1, sd = 4.00)
bc <- lapply(seq_len(7), function(k) rnorm(n_c, sd = 0.5))
eta <- 0.5 + 0.5 * x - 0.4 * z + b1[g1]
for (k in seq_len(7)) eta <- eta + bc[[k]][cs[[k]]]
df <- data.frame(y = rbinom(n, 1, plogis(eta)), x = x, z = z, g1 = g1)
for (k in seq_len(7)) df[[paste0("c", k)]] <- cs[[k]]
emit("sim_sparse_binomial_bigsd", df)
