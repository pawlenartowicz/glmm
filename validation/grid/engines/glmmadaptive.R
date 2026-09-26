#!/usr/bin/env Rscript
# GLMMadaptive reference fits over the accuracy grid: one JSONL record per
# cell, appended to $GRID_OUT. Resume-safe -- a cell already in the output is
# skipped, so a killed worker can simply be relaunched. Per-cell errors are
# caught and recorded as engine-fail; a grid corner that breaks GLMMadaptive is
# a data point, not a reason to lose the other cells.
#
# THE ORACLE IS SACRED. These records are a frozen reference glmm is later held
# to. Nothing here is ever edited to make a downstream engine agree, and no fit
# is truncated: there is no evaluation cap anywhere in this file, only the
# convergence controls below. The run harness's watchdog is the only cell cap
# and it works by killing the process from outside.
#
# Cells whose `oracles` list does not name GLMMadaptive are not fitted and get
# NO record. `oracles_of` (gen_manifest.R) hands this script only non-gaussian,
# non-inverse-Gaussian cells with exactly one grouping factor and no
# `weights_col` -- so every case a fixed-only path, an inverse-Gaussian family
# or a per-row prior weight would need is one this script never has to guard.

here <- normalizePath(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "versions.R"))
source(file.path(here, "common.R"))

env <- grid_env()
# Thread limits BEFORE library(GLMMadaptive). A threaded BLAS reads
# OMP_NUM_THREADS when its shared object is loaded, so setting the variables
# after the load is a no-op on those builds: six engine workers would then
# each take the whole machine. GLMMadaptive pulls in nlme/Matrix, which is
# what reaches the BLAS.
grid_threads()
suppressMessages({
  library(GLMMadaptive)
  library(jsonlite)
})

V <- grid_versions(env$grid_dir)
# Records "0.9.7" -- the dot form packageVersion() returns -- while
# versions.json pins the CRAN spelling "0.9-7"; norm_version is what makes the
# two compare equal in assert_pkg_version.
ENGINE_VERSION <- assert_pkg_version("GLMMadaptive", V[["GLMMadaptive"]])

# Tightened controls (speed-grid campaign precedent): mixed_model's DEFAULTS
# under-converge on low-information cells. update_GH_every = 1 re-adapts the
# quadrature grid every iteration, the like-for-like convention with glmm,
# which re-adapts at every deviance evaluation. No eval cap: mixed_model has no
# such knob, and iter_EM/iter_qN_outer are convergence controls, not a budget.
MA_CTRL <- list(iter_EM = 300, iter_qN_outer = 60,
                tol1 = 1e-8, tol2 = 1e-10, tol3 = 1e-12, update_GH_every = 1)

# The manifest's family + link pair -> a GLMMadaptive family object. No
# gaussian (oracles_of never hands this script a gaussian cell) and no
# inverse-Gaussian (GLMMadaptive has none); the switch's default case stops
# loudly rather than guessing if either ever arrives.
fam_obj <- function(family, link) {
  switch(family,
    binomial = stats::binomial(link = link),
    poisson  = stats::poisson(link = link),
    gamma    = GLMMadaptive::Gamma.fam(),
    negativebinomial = GLMMadaptive::negative.binomial(),
    stop("unsupported family for GLMMadaptive: ", family))
}

# The fit as a closure, so the timing loop can call it more than once.
#
# Aggregated-binomial cells (cell[["weights"]] == "size") use
# `ma_fixed = cbind(incidence, size - incidence) ~ ...` -- GLMMadaptive's
# actual binomial-trials form. Its `weights=` argument is a per-CLUSTER
# replicate multiplier, NOT a per-row trial count like glm/glmer's, so it
# cannot carry the aggregated-binomial convention; cbind carries it directly.
# gen_manifest.R already writes ma_fixed in that form, so nothing is parsed
# here. Prior-weight cells (`weights_col`) never reach this script at all
# (oracles_of excludes them), so there is no weights argument anywhere below.
#
# No `offset =` argument: the `offset(...)` term is already inside `ma_fixed`.
#
# Wrapped in with_warnings (below), the same construction lme4.R uses: a
# convergence or numerical-quality warning from mixed_model() would otherwise
# go to the console where nothing records it, so it is captured here and
# routed into `message` by record_glmmadaptive.
make_fit <- function(cell, df) {
  ff <- stats::as.formula(cell[["ma_fixed"]])
  rf <- stats::as.formula(cell[["ma_random"]])
  nagq <- if (is.null(cell[["nagq"]])) 1L else as.integer(cell[["nagq"]])
  fam <- fam_obj(cell[["family"]], cell[["link"]])
  function() with_warnings(
    GLMMadaptive::mixed_model(fixed = ff, random = rf, data = df,
                              family = fam, nAGQ = nagq, control = MA_CTRL))
}

# Runs `expr` and returns its value together with every warning it raised,
# instead of letting the warnings go to the console where nothing records
# them.
with_warnings <- function(expr) {
  w <- character(0)
  v <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = v, warnings = w)
}

# `m$D` -> stddev + correlation (`cov2cor`), the RE covariance on the absolute
# linear-predictor scale. `group` is parsed off `ma_random` (a `~ terms | g`
# string) rather than off the fit: the `trimws` is load-bearing -- the raw
# substring after the last `|` carries a leading space (`" g1"`), and
# `compare.R` joins `varcomp` by group name, so an untrimmed name would miss on
# every GLMMadaptive record.
record_glmmadaptive <- function(cell, m, fit_warnings, rec) {
  rec$converged <- isTRUE(m$converged) && is.finite(as.numeric(stats::logLik(m)))
  cf <- GLMMadaptive::fixef(m)
  rec$coef_names <- I(names(cf))
  rec$beta <- I(unname(cf))
  # The joint observed information over every parameter -- keeps the beta-theta
  # coupling, the like-for-like partner of lme4's use.hessian = TRUE and
  # glmmTMB's vcov(m)$cond. GLMMadaptive has no Rx-conditional counterpart, so
  # se_rx is never assigned.
  rec$se_hessian <- I(unname(sqrt(diag(stats::vcov(m, parm = "fixed-effects")))))

  D <- m$D
  sd <- sqrt(diag(D))
  rec$varcomp <- list(list(
    group = trimws(sub("^.*\\|", "", cell[["ma_random"]])),
    terms = I(colnames(D)),
    stddev = I(unname(sd)),
    corr = unname(stats::cov2cor(D))))
  rec$singular <- FALSE  # GLMMadaptive reports no singular-fit diagnostic

  if (cell[["family"]] == "gamma") {
    # Gamma.fam() (GLMMadaptive source, read 2026-09-22): the log-density uses
    # shape = exp(phis) with y ~ dgamma(shape, scale = mu/shape), so
    # Var(y) = mu^2/shape -- the same 1/shape dispersion convention as R's own
    # glm Gamma family. sigma is that dispersion's square root, the schema's
    # convention across engines: sigma = sqrt(1/shape) = exp(-phis/2).
    rec$sigma <- exp(-m$phis[1] / 2)
  }
  if (cell[["family"]] == "negativebinomial") {
    # negative.binomial() (GLMMadaptive source, read 2026-09-22) exponentiates
    # phis before using it as the NB2 shape (Var(y) = mu + mu^2/theta) --
    # confirmed against a live fit on nb_int1_g3000p20_bal_base 2026-09-22:
    # exp(phis) sits near the cell's true theta (1.5), phis itself does not.
    rec$nb_theta <- exp(m$phis[1])
  }

  if (length(fit_warnings) > 0) rec$message <- paste(fit_warnings, collapse = "; ")
  rec <- grid_set_loglik(rec, as.numeric(stats::logLik(m)))
  rec$status <- if (isTRUE(rec$converged)) "ok" else "engine-fail"
  rec
}

fit_cell <- function(env, cell) {
  rec <- grid_record(cell, "GLMMadaptive", ENGINE_VERSION)
  df <- grid_read_cell(env, cell)
  # Reading and typing the data frame stays outside the timed region: the
  # other engines are also holding typed data when their own timer starts.
  timed <- grid_time(env, make_fit(cell, df))
  if (!is.null(timed$wall_seconds)) rec$wall_seconds <- timed$wall_seconds
  rec$fits_per_sample <- timed$fits_per_sample
  fit <- timed$value   # with_warnings() result: list(value = <model>, warnings = <chr>)
  record_glmmadaptive(cell, fit$value, fit$warnings, rec)
}

con <- grid_open_out(env)
for (cell in grid_cells(env)) {
  if (!("GLMMadaptive" %in% unlist(cell[["oracles"]]))) next
  rec <- tryCatch(fit_cell(env, cell),
                  error = function(e) grid_fail(
                    grid_record(cell, "GLMMadaptive", ENGINE_VERSION), conditionMessage(e)))
  grid_write(con, rec)
  cat(sprintf("GLMMadaptive  %-42s  %s\n", cell[["cell"]], rec$status))
}
close(con)
