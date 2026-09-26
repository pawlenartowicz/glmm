#!/usr/bin/env Rscript
# lme4 reference fits over the accuracy grid: one JSONL record per cell,
# appended to $GRID_OUT. Resume-safe -- a cell already in the output is skipped,
# so a killed worker can simply be relaunched. Per-cell errors are caught and
# recorded as engine-fail; a grid corner that breaks lme4 is a data point, not a
# reason to lose the other 700 cells.
#
# THE ORACLE IS SACRED. These records are a frozen reference glmm is later held
# to. Nothing here is ever edited to make a downstream engine agree, and no fit
# is truncated: there is no evaluation cap anywhere in this file, because a
# truncated oracle fit is not an oracle. The run harness's watchdog is the only
# cell cap and it works by killing the process from outside.
#
# Cells whose `oracles` list does not name lme4 are not fitted and get NO
# record: a placeholder would pollute the reference with a row the comparator
# would have to learn to ignore.

here <- normalizePath(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "common.R"))
source(file.path(here, "versions.R"))

env <- grid_env()
# Thread limits BEFORE library(lme4). A threaded BLAS reads OMP_NUM_THREADS
# when its shared object is loaded, so setting the variables after the load is
# a no-op on those builds: six engine workers would then each take the whole
# machine. lme4 pulls in Matrix, which is what reaches the BLAS.
grid_threads()
suppressMessages({
  library(lme4)
  library(jsonlite)
})

V <- grid_versions(env$grid_dir)
# Records "2.0.6" -- the string packageVersion() returns -- while versions.json
# spells the pin "2.0-6". norm_version is what makes the two compare equal, and
# the RECORDED string is always the one packageVersion() gave.
ENGINE_VERSION <- assert_pkg_version("lme4", V[["lme4"]])

# PIRLS tolerance, 1e-13 against a default of 1e-7. glmer's deviance is
# assembled from working weights one PIRLS iteration behind the mode, so at the
# default its objective sits ~5.6e-4 above the true Laplace deviance on cbpp and
# vcov(use.hessian = TRUE) carries ~1% spurious theta-beta curvature. Value
# picked by a per-model sweep against tight-tolerance finite differences: cbpp
# is converged by 1e-10 but a sparse Poisson model keeps a 1.3% Hessian-SE
# residual until 1e-12, and grouseticks' vcov blips 1.5% on one coefficient at
# exactly 1e-12 while being clean at 1e-10, 1e-11 and 1e-13. 1e-13 is the value
# all of them agree at; 1e-16 aborts in step-halving, so stay above lme4's
# numeric floor. This is a solver-precision setting, not a model change.
TOLPWRSS <- 1e-13

# glmer computes the optimizer's finite-difference derivatives only when BOTH
# `nobs < check.conv.nobsmax` (default 10000) and `npar < check.conv.nparmax`
# (default 50), and vcov(use.hessian = TRUE) needs them -- so above 10000 rows
# lme4 reports no Hessian SE at all and the grid's se_hessian column goes empty
# on exactly the cells it matters most for. Asking for the derivatives directly
# restores it. The nparmax half never fires on this grid: the widest cell,
# binb_q8q2x_g3000p20_bal_base, has 40 covariance parameters and 8 fixed
# effects, 48 together, under the default 50 on either way of counting.
#
# calc.derivs rather than a raised check.conv.nobsmax, although both restore the
# derivatives: check.conv.nobsmax also governs whether checkConv RUNS lme4's
# gradient and Hessian convergence tests, so raising it would change which large
# cells this oracle calls converged. calc.derivs changes what is computed and
# nothing about the verdict.
#
# MEASURED 2026-09-23 on an UNLOCKED machine (no_turbo = 0) and pinned to one
# core, so read the walls as sizing estimates, not benchmarks. Two 30000-row
# binomial cells, each fitted twice through this same script with only this
# control flipped:
#   binb_int1_g30000p20_bal_base     3.51 s -> 3.69 s
#   binb_cross6_g30000p20_bal_base  55.39 s -> 64.69 s
# identical coefficients and identical n_eval in both pairs (127 and 631), no new
# convergence message, and se_hessian present only in the second arm.
#
# The derivative cost grows with the square of the parameter count, but the
# widest corner is out of reach for a different reason: on the same box
# binb_q8_g30000p20_bal_base (36 covariance parameters, 30000 rows) does not
# finish at all and is recorded as `timeout` -- "watchdog killed the engine after
# 600s" -- so that corner is the run harness's watchdog problem with or without
# this setting.
CALC_DERIVS <- TRUE

# The manifest's family + link pair -> an R family object.
#
# Two traps live here and nowhere else. R's Gamma() defaults to the INVERSE
# link, so the link is always passed explicitly rather than defaulted. And the
# manifest spells the inverse-Gaussian second link `inverse_squared` where R
# spells it `1/mu^2`; this is the one place that mapping is applied.
fam_obj <- function(family, link) {
  switch(family,
    gaussian = stats::gaussian(link = link),
    binomial = stats::binomial(link = link),
    poisson  = stats::poisson(link = link),
    gamma    = stats::Gamma(link = link),
    inversegaussian = stats::inverse.gaussian(
      link = if (identical(link, "inverse_squared")) "1/mu^2" else link),
    stop("unsupported family: ", family))
}

# stddev, VERSION-INVARIANTLY. lme4 parametrizes the RE covariance as
# Sigma = sigma^2 * Lambda Lambda', and `theta` is Lambda's lower triangle in
# COLUMN-MAJOR order. What VarCorr() reports from that changed between major
# versions: lme4 2.0-6 (2026-07-16) records that for GLMMs with an estimated
# scale parameter the variances and standard deviations had been "incorrectly,
# scaled by the estimated dispersion parameters" and are now printed unscaled.
# So VarCorr's Gamma stddev is theta*sigma under 1.1.x and theta under 2.0.x.
#
# Reading theta directly and applying sigma ONLY on the gaussian path makes this
# record identical under both major versions:
#   gaussian            sigma is the residual sd; the RE sd on the
#                       linear-predictor scale IS sigma*theta, and no version
#                       ever disagreed about that.
#   binomial/poisson/NB sigma == 1 by construction, so theta IS the sd.
#   gamma               theta unscaled -- the 2.0 convention.
# `sigma` is recorded separately in its own field either way, so nothing is lost.
#
# theta -> per-grouping block: for a q-wide grouping the next q(q+1)/2 entries
# are Lambda's column-major lower triangle; Sigma_rel = Lambda Lambda',
# stddev = sqrt(diag), corr = Sigma_rel / (sd sd').
lme4_varcomp <- function(m, scale_by_sigma) {
  th   <- lme4::getME(m, "theta")
  cnms <- lme4::getME(m, "cnms")
  s    <- if (scale_by_sigma) stats::sigma(m) else 1
  off  <- 0L
  lapply(seq_along(cnms), function(i) {
    terms <- cnms[[i]]
    q <- length(terms)
    n <- q * (q + 1L) / 2L
    v <- th[off + seq_len(n)]
    off <<- off + n
    L <- matrix(0, q, q)
    L[lower.tri(L, diag = TRUE)] <- v   # column-major lower triangle == lme4's theta order
    S  <- (L %*% t(L)) * s^2
    sd <- sqrt(diag(S))
    # I() keeps a length-1 vector a JSON array under auto_unbox, so the
    # comparator indexes every grouping the same way whatever its width.
    list(group = names(cnms)[i], terms = I(terms), stddev = I(unname(sd)),
         corr = unname(if (q == 1L) matrix(1, 1, 1) else S / outer(sd, sd)))
  })
}

# TRUE for a cell with no random effect. Those are fitted by stats::lm,
# stats::glm or MASS::glm.nb; the record still says engine "lme4", and `message`
# carries the routing so it is visible in the JSONL without a second engine name
# for the comparator to track.
is_glm <- function(cell) identical(cell[["structure"]], "glm")

glm_routing <- function(cell) {
  if (cell[["family"]] == "gaussian") "stats::lm"
  else if (cell[["family"]] == "negativebinomial") "MASS::glm.nb"
  else "stats::glm"
}

# The fit as a closure, so the timing loop can call it more than once.
#
# Weights and offsets, the three cases the manifest distinguishes:
#   trials cell (`weights`)      the counts are already inside r_formula's
#                                cbind(y, size - y) response -- NO weights
#                                argument, which would apply them a second time.
#   prior-weight cell (`weights_col`)  weights = that column, on every family.
#   offset cell (`offset_col`)   nothing to do: r_formula already carries the
#                                offset(...) term.
make_fit <- function(cell, df) {
  fm <- stats::as.formula(cell[["r_formula"]])
  w <- if (is.null(cell[["weights_col"]])) NULL else df[[cell[["weights_col"]]]]
  nagq <- if (is.null(cell[["nagq"]])) 1L else as.integer(cell[["nagq"]])
  if (is_glm(cell)) {
    return(switch(cell[["family"]],
      gaussian = function() stats::lm(fm, data = df, weights = w),
      negativebinomial = function() MASS::glm.nb(fm, data = df, weights = w),
      function() stats::glm(fm, data = df, weights = w,
                            family = fam_obj(cell[["family"]], cell[["link"]]))))
  }
  if (cell[["family"]] == "gaussian") {
    # `reml` is meaningful only here; a GLMM has no restricted likelihood.
    return(function() lme4::lmer(fm, data = df, weights = w,
                                 REML = isTRUE(cell[["reml"]])))
  }
  if (cell[["family"]] == "negativebinomial") {
    # glmer.nb estimates the shape by profiling around a Poisson glmer start;
    # the named arguments below reach that inner glmer through its `...`.
    return(function() lme4::glmer.nb(fm, data = df, weights = w, nAGQ = nagq,
                                     control = lme4::glmerControl(tolPwrss = TOLPWRSS,
                                                                  calc.derivs = CALC_DERIVS)))
  }
  function() lme4::glmer(fm, data = df, weights = w, nAGQ = nagq,
                         family = fam_obj(cell[["family"]], cell[["link"]]),
                         control = lme4::glmerControl(tolPwrss = TOLPWRSS,
                                                      calc.derivs = CALC_DERIVS))
}

# A fixed-effects fit has ONE standard error: there is no theta to condition on,
# so the Rx and Hessian methods cannot differ. varcomp is empty for the same
# reason -- there is no random effect to normalise.
#
# That single SE is written to BOTH se_rx and se_hessian on a non-gaussian cell,
# and to se_rx alone on a gaussian one, because that is where every other engine
# of this grid puts it: they split the two slots by FAMILY, not by whether the
# cell has a random effect, so a non-gaussian GLM cell fills both and a gaussian
# one fills se_rx. Emitting a different field set here would leave the
# comparator with one side of a pair on those cells.
record_glm <- function(cell, m, rec) {
  rec$message <- glm_routing(cell)
  # lm reports no iteration count; glm and glm.nb both count IRLS iterations.
  rec$converged <- if (is.null(m$converged)) TRUE else isTRUE(m$converged)
  if (!is.null(m$iter)) rec$n_eval <- as.integer(m$iter)
  rec$coef_names <- I(names(stats::coef(m)))
  rec$beta <- I(unname(stats::coef(m)))
  rec$se_rx <- se_of(stats::vcov(m))
  if (cell[["family"]] != "gaussian") rec$se_hessian <- rec$se_rx
  rec$varcomp <- list()
  # sigma() is the residual sd on lm and sqrt(dispersion) on glm, which is the
  # one convention the schema's `sigma` carries across families. Every other
  # family's scale is fixed at 1 and reports null; the negative-binomial shape
  # is not a scale and belongs in nb_theta alone.
  if (cell[["family"]] %in% c("gaussian", "gamma")) rec$sigma <- stats::sigma(m)
  if (cell[["family"]] == "negativebinomial") rec$nb_theta <- unname(m$theta)
  rec <- grid_set_loglik(rec, as.numeric(stats::logLik(m)))
  rec$status <- if (isTRUE(rec$converged)) "ok" else "engine-fail"
  rec
}

# Runs `expr` and returns its value together with every warning it raised,
# instead of letting the warnings go to the console where nothing records them.
# lme4's vcov signals a warning and still RETURNS A VALUE when it substitutes
# one estimator for another, so the warning is the only evidence that the number
# is not the one that was asked for.
with_warnings <- function(expr) {
  w <- character(0)
  v <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = v, warnings = w)
}

se_of <- function(V) I(unname(sqrt(diag(as.matrix(V)))))

record_mixed <- function(cell, m, rec) {
  # `converged` collapses every lme4 message to one bit, but the messages are
  # not one thing: a singular-fit note, a max-gradient warning and a Hessian
  # warning all land in the same FALSE. Triage of a non-converged cell in a run
  # of hundreds cannot proceed without knowing which one fired, so the text is
  # carried through verbatim.
  msgs <- as.character(unlist(m@optinfo$conv$lme4$messages))
  rec$converged <- length(msgs) == 0
  notes <- msgs
  rec$singular <- lme4::isSingular(m)
  # On a negative-binomial cell this is the FINAL inner glmer's evaluation count,
  # which is the only one the fitted object keeps: glmer.nb's outer profiling over
  # the shape refits that glmer several times and leaves no count of its own.
  rec$n_eval <- as.integer(m@optinfo$feval)
  rec$coef_names <- I(names(lme4::fixef(m)))
  rec$beta <- I(unname(lme4::fixef(m)))
  if (cell[["family"]] == "gaussian") {
    # The LMM SE is profiled and exact -- one method, no theta-beta coupling
    # question, so there is no se_hessian to record.
    rec$se_rx <- se_of(stats::vcov(m))
    rec$sigma <- stats::sigma(m)
  } else {
    # A GLMM vcov has two methods that genuinely differ under the Laplace
    # approximation:
    #   se_hessian  finite-difference Hessian of the joint deviance over
    #               (theta, beta) -- keeps the coupling; glmer's own default.
    #   se_rx       Schur complement conditional on theta-hat -- drops it.
    # Both are recorded so the comparator can hold like method against like.
    #
    # vcov(use.hessian = TRUE) does NOT fail loudly when the finite-difference
    # Hessian is unusable: when that matrix is not positive definite or holds
    # NAs, lme4 warns and RETURNS THE RX ESTIMATE instead. Recording that would
    # put an Rx number in the se_hessian slot and make the comparator read a
    # method substitution as agreement between the two methods. So any warning
    # at all from this call disqualifies the value -- for use.hessian = TRUE the
    # only warnings lme4 can raise are that fallback and a failed conversion
    # that returns an NA matrix, and neither is the Hessian SE. The slot is left
    # absent and the text goes to `message`; an outright error is handled the
    # same way.
    hess <- tryCatch(with_warnings(stats::vcov(m, use.hessian = TRUE)),
                     error = function(e) list(value = NULL,
                                              warnings = conditionMessage(e)))
    if (length(hess$warnings) == 0) rec$se_hessian <- se_of(hess$value)
    else notes <- c(notes, hess$warnings)
    # The Rx call has one EXPECTED warning: lme4 advises switching to the
    # Hessian when the two estimates differ by more than 1e-4, which is exactly
    # the quantity being recorded here. That one is dropped; anything else the
    # call says is news and is carried into `message`.
    rx <- with_warnings(stats::vcov(m, use.hessian = FALSE))
    rec$se_rx <- se_of(rx$value)
    notes <- c(notes, rx$warnings[!grepl("differ by >", rx$warnings, fixed = TRUE)])
    if (cell[["family"]] == "gamma") rec$sigma <- stats::sigma(m)
    if (cell[["family"]] == "negativebinomial")
      rec$nb_theta <- unname(lme4::getME(m, "glmer.nb.theta"))
  }
  if (length(notes) > 0) rec$message <- paste(notes, collapse = "; ")
  rec <- grid_set_loglik(rec, as.numeric(stats::logLik(m)))
  # Built after every scalar field: it is the one field derived from a shape the
  # fit may not have filled in, so a surprise here costs only this field.
  rec$varcomp <- lme4_varcomp(m, scale_by_sigma = cell[["family"]] == "gaussian")
  # Only the fit's own convergence decides this. A vcov note is a note about one
  # recorded field, not a failed fit.
  rec$status <- if (isTRUE(rec$converged)) "ok" else "engine-fail"
  rec
}

fit_cell <- function(env, cell) {
  rec <- grid_record(cell, "lme4", ENGINE_VERSION)
  df <- grid_read_cell(env, cell)
  # Reading and typing the data frame stays outside the timed region: the other
  # engines are also holding typed data when their own timer starts.
  timed <- grid_time(env, make_fit(cell, df))
  if (!is.null(timed$wall_seconds)) rec$wall_seconds <- timed$wall_seconds
  rec$fits_per_sample <- timed$fits_per_sample
  if (is_glm(cell)) record_glm(cell, timed$value, rec)
  else record_mixed(cell, timed$value, rec)
}

con <- grid_open_out(env)
for (cell in grid_cells(env)) {
  if (!("lme4" %in% unlist(cell[["oracles"]]))) next
  rec <- tryCatch(fit_cell(env, cell),
                  error = function(e) grid_fail(
                    grid_record(cell, "lme4", ENGINE_VERSION), conditionMessage(e)))
  grid_write(con, rec)
  cat(sprintf("lme4  %-42s  %s\n", cell[["cell"]], rec$status))
}
close(con)
