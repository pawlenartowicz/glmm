#!/usr/bin/env Rscript
# Simulated dataset for the INVERSE-GAUSSIAN GLM rung
# -> ../data/simulated/sim_igauss.csv
# Run standalone with `Rscript validation/prep/gen_igauss_data.R`, or as the
# seventh of the prep scripts `run.sh --prep` invokes.
#
# THE STAGED CSV IS AUTHORITATIVE AND FROZEN, like the reference JSONs.
# Regenerating it silently invalidates the committed references; if regeneration
# is ever unavoidable, regenerate CSV and references together in one step and
# record the generating R version here.
#   Generated 2026-08-26 on R 4.5.3, glibc 2.42, x86_64.
#
# WHY A NEW FIXTURE. Gamma and binomial data are link-agnostic within their own
# family, so sim_gamma and cbpp could each carry a second link cell. Nothing in
# the corpus was simulated with V(mu) = mu^3, and no existing y column satisfies
# the inverse-Gaussian model's shape, so the two IG cells need their own data.
#
# SIZE AND SHAPE: 2,000 rows, p = 3 (intercept + one continuous + one two-level
# factor), one dataset carrying BOTH link cells. Realized on the frozen CSV:
# mu in [1.2496, 1.8742], so eta = 1/mu^2 in [0.2847, 0.6404] -- comfortably
# inside the InverseSquared link's eta > 0 domain with no row near the
# MU_FLOOR clamp, so the rung gates the kernel's arithmetic and not its
# boundary projection. dispersion 0.3: large enough that phi-hat is
# identifiable, small enough that the sampler does not produce y within
# rounding distance of 0 (the family's domain edge, where dev_resid's 1/y
# divides).
#
# SAMPLER: statmod::rinvgauss is not an existing R dependency of this harness
# (README.md's setup list is lme4 + jsonlite + GLMMadaptive; no DESCRIPTION/renv
# under validation/ names statmod), so this script does not add it. It samples
# with the base-R Wald transform instead (Michael, Schucany & Haas 1976), which
# needs only `stats`.
#
# COEFFICIENTS: eta = log(mu) = 0.4 + 0.05*x + 0.05*I(grp=="b"), x ~ N(0,1),
# realizing mu in [1.2496, 1.8742] on the frozen CSV -- narrower than the
# (0.5, 3) band this design was aimed at, with margin on both sides rather
# than spanning it. dispersion phi = 0.3 -> Wald lambda = 1/phi = 10/3, the
# convention R's inverse.gaussian() family uses (V(mu) = phi * mu^3). A
# steeper slope (tried at 0.10/0.15) makes the y tail long enough that
# glm(family = inverse.gaussian(link = "1/mu^2"))'s default IRLS start
# (mustart = y) overshoots into eta <= 0 on some seeds ("no valid set of
# coefficients... please supply starting values"); the shallower slope keeps
# that fit converging under R's own default start, which is what Step 2 checks
# and what goldens_agq.R relies on.

rinvgauss_base <- function(n, mu, lambda) {
  # Michael, Schucany & Haas (1976) transform: y0 = chi-square(1), then the
  # smaller root, accepted with probability mu/(mu+x1).
  v <- rnorm(n)^2
  x1 <- mu + mu^2 * v / (2 * lambda) -
    (mu / (2 * lambda)) * sqrt(4 * mu * lambda * v + mu^2 * v^2)
  ifelse(runif(n) <= mu / (mu + x1), x1, mu^2 / x1)
}

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
out_dir <- file.path(suite_dir, "data", "simulated")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

emit <- function(name, df) {
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
  cat(sprintf("wrote %-22s  %6d rows x %d cols\n", name, nrow(df), ncol(df)))
}

set.seed(20260826)
n <- 2000
x <- rnorm(n)
grp <- factor(rep(c("a", "b"), length.out = n))
eta <- 0.4 + 0.05 * x + 0.05 * (grp == "b")
mu <- exp(eta)
phi <- 0.3
lambda <- 1 / phi
y <- rinvgauss_base(n, mu, lambda)
emit("sim_igauss", data.frame(y = y, x = x, grp = grp))
