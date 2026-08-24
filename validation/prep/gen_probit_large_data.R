#!/usr/bin/env Rscript
# Simulated dataset for the LARGE-PROBIT coverage rung
# -> ../data/simulated/sim_probit_large.csv, the same directory every other rung
# uses. Run standalone with `Rscript validation/prep/gen_probit_large_data.R`, or
# as the sixth of the prep scripts `run.sh --prep` invokes.
#
# THE STAGED CSV IS AUTHORITATIVE AND FROZEN, like the reference JSONs:
# the CSV, not the seed, is the artifact. Regenerating it silently
# invalidates the committed lme4/MixedModels references, so if regeneration is
# ever unavoidable, regenerate CSV and references together in one step and record
# the generating R version here.
#   Generated 2026-08-23 on R 4.5.3, glibc 2.42, x86_64.
#
# WHY THIS RUNG EXISTS. Before it, the whole corpus carried exactly one probit
# fit — cbpp_probit, 56 rows at ~3 ms — and none of the speed, estimate or
# monte-carlo campaigns carried any. That is enough to gate the probit link's
# CORRECTNESS and nothing else: at 56 rows a vectorized family kernel is pure
# measurement noise, so a probit performance regression, or an accuracy
# regression that only appears once the row loop is long enough for its rounding
# to accumulate, was invisible. This rung makes both visible.
#
# WHY A SEPARATE PREP SCRIPT. Every rung export_data.R / gen_weights_data.R /
# gen_large_theta_data.R emits is already backed by a frozen reference that
# cannot be regenerated, so appending to any of them puts those corpora one
# editing mistake away from a rewritten CSV. gen_weights_data.R set the
# precedent of a self-contained per-tier script and gen_large_theta_data.R
# followed it; so does this.
#
# SIZE: 100 groups x 96 rows = 9,600 rows, p = 5. Three things set it.
#   - It has to be long enough that the per-row transcendental work dominates a
#     PIRLS iteration rather than the O(k^3) block solve. At 100 clusters and a
#     scalar RE the solve is trivial, so essentially the whole eval is the row
#     pass -- which is the thing being timed. 9,600 rows is 170x cbpp_probit.
#   - THE 10,000-ROW CEILING IS HARD, and it is lme4's, not a taste call.
#     glmerControl's `check.conv.nobsmax` defaults to 10000, above which lme4
#     stops computing the optimizer Hessian: `m@optinfo$derivs` comes back NULL
#     and `vcov(use.hessian = TRUE)` -- which engines/lme4.R calls for every
#     rung -- hard-errors with "Hessian is unavailable". Measured 2026-08-23 on
#     this design: derivs present at n = 2,000 and 5,000, absent at 10,000 and
#     20,000. A bigger rung would therefore need a per-rung lme4 control
#     override, i.e. a different reference model spec for one rung, which the
#     oracle rules do not allow. So the rung is sized under the ceiling and the
#     row count is spent on rows per cluster rather than on more clusters.
#   - 96 rows per cluster (not 6, the large-theta-hat tier's shape) puts each
#     conditional mode in the well-determined regime, where Laplace is accurate
#     and the FD-Hessian SEs agree with the references tightly. This rung is not
#     a hard-cell rung; rungs 44-46 already own that axis. Here the design is
#     deliberately easy so that a gate miss means a kernel defect and not the
#     approximation's own bias.
# The CSV lands at ~610 kB, between VerbAgg's 450 kB and InstEval's 2.3 MB.
#
# SHAPE: Bernoulli response, scalar (1 | g), p = 5 (intercept + three continuous
# + one binary). Bernoulli rather than aggregated because the probit arm of the
# family kernel is exercised identically either way while the Bernoulli form
# keeps the prior-weight path out of the picture — weighted binomial is already
# gated at cbpp, cbpp_probit and the whole 29-43 tier, and mixing the two axes
# in one rung would make a failure ambiguous. p = 5 rather than the corpus's
# usual two predictors so the WLS solve sees a non-degenerate X while staying
# far from any conditioning question.
#
# COEFFICIENTS: chosen so eta stays inside roughly +-3.5, i.e. mu inside
# (2e-4, 1-2e-4). A probit link saturates fast — |eta| > 7 already clamps mu at
# family::PROB_EPS — and a saturated design would test the clamp rather than the
# erfc kernel. sd_g = 0.7 puts the fitted theta-hat near 0.66 with 96 rows per
# cluster to pin it, clear of the boundary and clear of the large-theta-hat
# regime rungs 44-46 cover.
# b0 = 0.3, not ~0, on sim_binomial_slope2's recorded rationale: a near-zero
# realized intercept turns a small absolute beta[0] gap into a spurious >1e-3
# RELATIVE gap at compare.R's beta gate.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
out_dir <- file.path(suite_dir, "data", "simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

emit <- function(name, df) {
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("wrote %-22s  %6d rows x %d cols\n", name, nrow(df), ncol(df)))
}

set.seed(20260823)
n_g <- 100
per <- 96
n <- n_g * per
g <- factor(rep(seq_len(n_g), each = per))
x1 <- rnorm(n)
x2 <- rnorm(n)
x3 <- rnorm(n)
z <- rbinom(n, 1, 0.5)
b <- rnorm(n_g, sd = 0.7)
eta <- 0.3 + 0.5 * x1 - 0.4 * x2 + 0.25 * x3 - 0.6 * z + b[g]
emit("sim_probit_large",
     data.frame(y = rbinom(n, 1, pnorm(eta)), x1 = x1, x2 = x2, x3 = x3,
                z = z, g = g))
