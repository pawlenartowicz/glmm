#!/usr/bin/env Rscript
# glmmTMB reference fits over the accuracy grid: one JSONL record per cell,
# appended to $GRID_OUT. Resume-safe -- a cell already in the output is skipped,
# so a killed worker can simply be relaunched. Per-cell errors are caught and
# recorded as engine-fail; a grid corner that breaks glmmTMB is a data point,
# not a reason to lose the other cells.
#
# THE ORACLE IS SACRED. These records are a frozen reference glmm is later held
# to. Nothing here is ever edited to make a downstream engine agree, and no fit
# is truncated: there is no evaluation cap anywhere in this file. The run
# harness's watchdog is the only cell cap and it works by killing the process
# from outside.
#
# Cells whose `oracles` list does not name glmmTMB are not fitted and get NO
# record: a placeholder would pollute the reference with a row the comparator
# would have to learn to ignore.
#
# A GLM cell (no random effect) is fitted by glmmTMB itself, giving it the same
# formula with no `(... | ...)` term: a supported fixed-effects fit. VarCorr's
# `cond` component is then empty on that cell, so `varcomp` comes out `list()`
# and `singular` comes out FALSE with no separate branch.

here <- normalizePath(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "versions.R"))
source(file.path(here, "common.R"))

env <- grid_env()
# Thread limits BEFORE library(glmmTMB). A threaded BLAS reads OMP_NUM_THREADS
# when its shared object is loaded, so setting the variables after the load is
# a no-op on those builds: six engine workers would then each take the whole
# machine. glmmTMB pulls in TMB and Matrix, which is what reaches the BLAS.
grid_threads()
suppressMessages({
  library(glmmTMB)
  library(jsonlite)
})

V <- grid_versions(env$grid_dir)
# Records "1.1.14" -- the string packageVersion() returns, which happens to be
# the same spelling versions.json pins it with.
ENGINE_VERSION <- assert_pkg_version("glmmTMB", V[["glmmTMB"]])

# The manifest's family + link pair -> a glmmTMB family object. glmmTMB 1.1.14
# has no inverse-Gaussian family (confirmed against the installed package's
# exports), which is why `oracles_of` never hands this script an
# inversegaussian cell; the switch's default case stops loudly rather than
# guessing if one ever does.
fam_obj <- function(family, link) {
  switch(family,
    gaussian = stats::gaussian(),
    binomial = stats::binomial(link = link),
    poisson  = stats::poisson(link = link),
    gamma    = stats::Gamma(link = link),
    negativebinomial = glmmTMB::nbinom2(link = link),
    stop("unsupported family for glmmTMB: ", family))
}

# TRUE for a cell with no random effect -- still fitted by glmmTMB itself
# (see file header), but REML is forced off for it below and its SE fields
# follow a different split than a mixed cell's.
is_glm <- function(cell) identical(cell[["structure"]], "glm")

# The fit as a closure, so the timing loop can call it more than once.
#
# Weights, the cases the manifest distinguishes:
#   trials cell (`weights`)            the counts are already inside
#                                       r_formula's cbind(y, size - y)
#                                       response -- NO weights argument, which
#                                       would apply them a second time.
#   prior-weight cell (`weights_col`),  a dispersion offset and NO weights
#   gaussian or gamma                   argument. The grid draws these cells with
#                                       precision weights (row i has dispersion
#                                       phi / w_i, gen_common.R), and glmmTMB's
#                                       `weights` instead multiplies each row's
#                                       log-density by w_i, a different
#                                       likelihood. The offset is on glmmTMB's
#                                       dispersion scale: log(sigma) on gaussian,
#                                       so -0.5 * log(w); log(shape) on gamma, so
#                                       +log(w).
#   prior-weight cell, other family     weights = that column. On poisson and NB
#                                       there is no dispersion to put the weight
#                                       on, and the two meanings coincide.
# No `offset =` argument on any cell: the `offset(...)` term is already in
# r_formula, and passing both would apply it twice.
#
# REML is meaningful only on a cell with a random effect to restrict; a GLM
# cell's `reml` flag (true whenever its family is gaussian, per gen_manifest.R)
# is ignored here rather than passed through.
#
# Wrapped in with_warnings (below): glmmTMB can warn DURING THE FIT ITSELF --
# "non-positive-definite Hessian matrix", "extreme or very small eigenvalues
# detected", "failed to invert Hessian from numDeriv::jacobian(), falling back
# to internal vcov estimate" -- without touching `m$fit$convergence`, so the
# warning text is the only evidence the returned model is not fully trustworthy.
#
# glmmTMB's default start is beta = 0. On an inverse link that is eta = 0 and
# mu = Inf, and the fit stops with "negative log-likelihood is NaN at starting
# parameter values". Only then is the fit retried from the fixed-effects GLM's
# beta (as validation/tools/goldens_agq.R does), so a cell that starts from the
# default is unaffected by this branch. The retry is recorded in `message`.
make_fit <- function(cell, df) {
  fm <- stats::as.formula(cell[["r_formula"]])
  wcol <- cell[["weights_col"]]
  fam <- fam_obj(cell[["family"]], cell[["link"]])
  reml <- !is_glm(cell) && isTRUE(cell[["reml"]])
  wv <- if (is.null(wcol)) NULL else df[[wcol]]
  disp <- NULL
  if (!is.null(wcol) && cell[["family"]] %in% c("gaussian", "gamma")) {
    scale <- if (cell[["family"]] == "gaussian") "-0.5" else "1"
    disp <- stats::as.formula(sprintf("~ offset(%s * log(%s))", scale, wcol))
  }
  w <- if (is.null(disp)) wv else NULL
  fit_from <- function(start = NULL) {
    if (is.null(disp))
      glmmTMB::glmmTMB(fm, data = df, family = fam, weights = w, REML = reml,
                       start = start)
    else
      glmmTMB::glmmTMB(fm, data = df, family = fam, dispformula = disp,
                       REML = reml, start = start)
  }
  function() with_warnings(tryCatch(fit_from(), error = function(e) {
    msg <- conditionMessage(e)
    if (!grepl("NaN at starting parameter values", msg, fixed = TRUE)) stop(e)
    b0 <- stats::coef(stats::glm(reformulas::nobars(fm), data = df, family = fam,
                                 weights = wv))
    warning("default start failed (", msg, "); refitted from the fixed-effects GLM beta")
    fit_from(list(beta = unname(b0)))
  }))
}

# Runs `expr` and returns its value together with every warning it raised,
# instead of letting the warnings go to the console where nothing records
# them. glmmTMB's fit and vcov calls both signal a warning and still RETURN A
# VALUE when something about the fit is unreliable, so the warning is the only
# evidence the number is not what it claims to be.
with_warnings <- function(expr) {
  w <- character(0)
  v <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = v, warnings = w)
}

# `se`, the single Rx-and-Hessian-indistinguishable SE from `vcov(m)$cond`:
# glmmTMB's vcov is the conditional block of the inverse joint TMB Hessian over
# every parameter, so it keeps the beta-theta coupling and is the like-for-like
# partner of lme4's `use.hessian = TRUE`.
record_glmmtmb <- function(cell, m, fit_warnings, rec) {
  notes <- fit_warnings
  # m$fit$convergence == 0 alone is not enough: glmmTMB can warn about a
  # non-positive-definite Hessian or extreme eigenvalues (see make_fit) without
  # touching it. `pdHess` is the diagnostic those two warnings are raised from,
  # so it is the second required condition.
  rec$converged <- isTRUE(m$fit$convergence == 0) && isTRUE(m$sdr$pdHess)
  # `rec["n_eval"] <- list(NULL)`, not `rec$n_eval <- NULL`: the second DELETES
  # the key, the same trap grid_set_loglik documents.
  rec["n_eval"] <- if ("function" %in% names(m$fit$evaluations))
    list(as.integer(m$fit$evaluations[["function"]])) else list(NULL)

  cf <- glmmTMB::fixef(m)$cond
  rec$coef_names <- I(names(cf))
  rec$beta <- I(unname(cf))

  vc <- tryCatch(with_warnings(stats::vcov(m)$cond),
                 error = function(e) list(value = NULL, warnings = conditionMessage(e)))
  notes <- c(notes, vc$warnings)
  # "failed to invert Hessian ... falling back to internal vcov estimate" can
  # be raised at FIT time (inside glmmTMB's own finalizeTMB, before the model
  # is even returned) or, defensively, at this vcov call -- checked across
  # both sources together. When it fires, `pdHess` can still read TRUE (the
  # fallback is attempted only after the primary Hessian is judged usable) but
  # the matrix vcov() hands back is an internal fallback estimate, not the
  # requested inverse-Hessian SE. The field(s) it would have filled are left
  # ABSENT rather than record a substituted number as if it were the real one.
  fallback <- is.null(vc$value) ||
    any(grepl("falling back to internal vcov estimate", notes, fixed = TRUE))
  # The field split is NOT the same on a GLM cell as on a mixed one:
  #   gaussian (GLM or mixed)      se_rx only -- no Rx-vs-Hessian question.
  #   non-gaussian, mixed          se_hessian only, per the general convention.
  #   non-gaussian, GLM (no RE)    BOTH se_rx and se_hessian carry the one SE --
  #                                a fixed-effects fit has no theta to condition
  #                                on, so the two methods cannot differ, and this
  #                                is where lme4.R and the Rust engine put that
  #                                single value.
  if (!fallback) {
    se <- unname(sqrt(diag(vc$value)))
    if (cell[["family"]] == "gaussian") {
      rec$se_rx <- I(se)
    } else if (is_glm(cell)) {
      rec$se_rx <- I(se)
      rec$se_hessian <- I(se)
    } else {
      rec$se_hessian <- I(se)
    }
  }

  varc <- glmmTMB::VarCorr(m)$cond
  rec$varcomp <- lapply(names(varc), function(g) {
    block <- varc[[g]]
    sd <- attr(block, "stddev")
    corr <- attr(block, "correlation")
    list(group = g, terms = I(names(sd)), stddev = I(unname(sd)),
         corr = unname(corr))
  })
  # Empty on a GLM cell (varc has no names to iterate), which also makes `any`
  # over zero blocks come out FALSE with no separate branch.
  rec$singular <- any(unlist(lapply(varc, function(b) any(attr(b, "stddev") < 1e-8))))

  # sigma is one convention across the grid: residual SD on gaussian,
  # sqrt(dispersion) on gamma, null elsewhere. Explicitly null on nbinom2,
  # where sigma() returns the NB SHAPE, not a dispersion -- letting it through
  # would make sigma == nb_theta here while glmm reports sigma == null on the
  # same family. nb_theta is the one field that carries the NB shape.
  if (cell[["family"]] %in% c("gaussian", "gamma")) rec$sigma <- stats::sigma(m)
  if (cell[["family"]] == "negativebinomial") rec$nb_theta <- stats::sigma(m)

  # Every warning collected above (fit-time and vcov-time) goes into `message`,
  # whether or not it flipped `converged` or `se_rx`/`se_hessian` -- a warning
  # on an otherwise-ok fit is still worth seeing when triaging the run.
  if (length(notes) > 0) rec$message <- paste(notes, collapse = "; ")
  rec <- grid_set_loglik(rec, as.numeric(stats::logLik(m)))
  rec$status <- if (isTRUE(rec$converged)) "ok" else "engine-fail"
  rec
}

fit_cell <- function(env, cell) {
  rec <- grid_record(cell, "glmmTMB", ENGINE_VERSION)
  df <- grid_read_cell(env, cell)
  # Reading and typing the data frame stays outside the timed region: the
  # other engines are also holding typed data when their own timer starts.
  timed <- grid_time(env, make_fit(cell, df))
  if (!is.null(timed$wall_seconds)) rec$wall_seconds <- timed$wall_seconds
  rec$fits_per_sample <- timed$fits_per_sample
  fit <- timed$value   # with_warnings() result: list(value = <model>, warnings = <chr>)
  record_glmmtmb(cell, fit$value, fit$warnings, rec)
}

con <- grid_open_out(env)
for (cell in grid_cells(env)) {
  if (!("glmmTMB" %in% unlist(cell[["oracles"]]))) next
  rec <- tryCatch(fit_cell(env, cell),
                  error = function(e) grid_fail(
                    grid_record(cell, "glmmTMB", ENGINE_VERSION), conditionMessage(e)))
  grid_write(con, rec)
  cat(sprintf("glmmTMB  %-42s  %s\n", cell[["cell"]], rec$status))
}
close(con)
