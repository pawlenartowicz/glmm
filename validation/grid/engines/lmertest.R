#!/usr/bin/env Rscript
# lmerTest reference fits over the accuracy grid's Gaussian LMM cells: one JSONL
# record per cell, appended to $GRID_OUT, carrying lmerTest's per-coefficient
# Satterthwaite degrees of freedom as `coef_df`. compare.R's gate 5 holds glmm's
# `coef_df` to these. Resume-safe and fail-soft the way lme4.R is: a cell
# already in the output is skipped, and a per-cell error is recorded as
# engine-fail rather than ending the run.
#
# THE ORACLE IS SACRED. These records are a frozen reference glmm is later held
# to. Nothing here is ever edited to make a downstream engine agree, and no fit
# is truncated: there is no evaluation cap anywhere in this file, because a
# truncated oracle fit is not an oracle. The run harness's watchdog is the only
# cell cap and it works by killing the process from outside.
#
# Cells whose `oracles` list does not name lmerTest are not fitted and get NO
# record. gen_manifest.R names it on every Gaussian cell with a random effect.
#
# The fit is lme4's (lmerTest::lmer calls lme4::lmer and then derives the
# variance-parameter Hessian and the Jacobian of vcov(beta) from the same
# deviance function), so this engine records only what lme4.R does not:
# the df, and what gate 5 needs to pair them. The fit's other numbers stay with
# lme4.R, which is the one oracle gates 1 to 3 read for them.

here <- normalizePath(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "common.R"))
source(file.path(here, "versions.R"))

env <- grid_env()
# Thread limits BEFORE library(lmerTest), for the reason lme4.R gives: a
# threaded BLAS reads OMP_NUM_THREADS when its shared object is loaded.
grid_threads()
suppressMessages({
  library(lmerTest)
  library(jsonlite)
})

V <- grid_versions(env$grid_dir)
# "3.2.1" recorded, "3.2-1" pinned: norm_version compares the two spellings, and
# the recorded string is the one packageVersion() gave (versions.R).
ENGINE_VERSION <- assert_pkg_version("lmerTest", V[["lmerTest"]])

# Every warning `expr` raises, kept instead of printed: lmerTest signals a
# variance-parameter Hessian with a negative or near-zero eigenvalue as a
# warning and still returns a fit, so the warning is the only record of it.
with_warnings <- function(expr) {
  w <- character(0)
  v <- withCallingHandlers(expr, warning = function(cond) {
    w <<- c(w, conditionMessage(cond))
    invokeRestart("muffleWarning")
  })
  list(value = v, warnings = w)
}

# The fit as a closure, as lme4.R::make_fit's Gaussian branch builds it: the
# prior-weight column when the cell has one, REML from the cell. A trials or
# offset cell is never Gaussian, so neither case arises here.
make_fit <- function(cell, df) {
  fm <- stats::as.formula(cell[["r_formula"]])
  # `.w`, not `w`: lm/glm/lmer/glmmTMB look `weights` up in `data` first, so a
  # local named like a data column (sim_dynrange_lmm has a predictor `w`) would
  # silently fit with that column as the prior weights.
  .w <- if (is.null(cell[["weights_col"]])) NULL else df[[cell[["weights_col"]]]]
  reml <- isTRUE(cell[["reml"]])
  function() with_warnings(lmerTest::lmer(fm, data = df, weights = .w, REML = reml))
}

# The singular-fit note is the one lme4 convergence message that does not make
# the fit unconverged here (lme4.R counts it as unconverged). A boundary fit is
# where glmm's df and lmerTest's are read against each other most carefully, so
# gate 5 has to see it; `singular` carries the fact instead.
SINGULAR_NOTE <- "boundary (singular) fit"

record_fit <- function(m, warns, rec, s = with_warnings(summary(m))) {
  msgs <- as.character(unlist(m@optinfo$conv$lme4$messages))
  # The optimizer's own return code and warnings, read as lme4.R reads them:
  # lme4 skips its gradient checks above check.conv.nobsmax rows, so these are
  # the only record of an optimizer that stopped at its budget.
  opt_code <- m@optinfo$conv$opt
  opt_warn <- as.character(unlist(m@optinfo$warnings))
  opt_failed <- (!is.null(opt_code) && !identical(as.numeric(opt_code), 0)) ||
                length(opt_warn) > 0
  conv_msgs <- msgs[!grepl(SINGULAR_NOTE, msgs, fixed = TRUE)]
  rec$converged <- length(conv_msgs) == 0 && !opt_failed
  rec$singular <- lme4::isSingular(m)
  notes <- c(msgs, opt_warn, warns)
  if (!is.null(opt_code) && !identical(as.numeric(opt_code), 0) && !length(opt_warn))
    notes <- c(notes, sprintf("optimizer return code %s", format(opt_code)))
  # Every column of the formula's design, a column lme4 dropped as aliased
  # included (beta and df NA there, written null), as glmm records it: gate 5
  # pairs the two by name and drops a coordinate null on both sides.
  fe <- lme4::fixef(m, add.dropped = TRUE)
  rec$coef_names <- I(names(fe))
  rec$beta <- I(unname(fe))
  # summary() computes the Satterthwaite df per retained coefficient, placed
  # here by name; pipeline mode passes the one it timed. Its warnings are
  # lmerTest's too and go to `message` with the fit's.
  notes <- c(notes, s$warnings)
  co <- s$value$coefficients
  rec["coef_df"] <- if ("df" %in% colnames(co)) {
    df <- setNames(rep(NA_real_, length(fe)), names(fe))
    df[rownames(co)] <- co[, "df"]
    list(I(unname(df)))
  } else list(NULL)
  if (is.null(rec$coef_df)) notes <- c(notes, "summary() reported no df column")
  notes <- unique(notes)
  if (length(notes) > 0) rec$message <- paste(notes, collapse = "; ")
  rec$status <- if (isTRUE(rec$converged)) "ok" else "engine-fail"
  rec
}

# Pipeline mode times the fit, summary() and anova() (Type III, Satterthwaite
# DenDF, lmerTest's defaults) together, and the record is read from what the
# timed call returned.
fit_cell <- function(env, cell) {
  rec <- grid_record(cell, "lmerTest", ENGINE_VERSION)
  if (env$pipeline) grid_check_pipeline_cell(cell)
  df <- grid_read_cell(env, cell)
  fit <- make_fit(cell, df)
  timed <- grid_time(env, if (!env$pipeline) fit else function() {
    f <- fit()
    list(fit = f, summary = with_warnings(summary(f$value)),
         anova = with_warnings(anova(f$value)))
  })
  if (!is.null(timed$wall_seconds)) rec$wall_seconds <- timed$wall_seconds
  rec$fits_per_sample <- timed$fits_per_sample
  if (!env$pipeline) return(record_fit(timed$value$value, timed$value$warnings, rec))
  v <- timed$value
  a <- v$anova$value
  rec$term_ddf <- as.list(setNames(a[["DenDF"]], rownames(a)))
  record_fit(v$fit$value, c(v$fit$warnings, v$anova$warnings), rec, v$summary)
}

con <- grid_open_out(env)
for (cell in grid_cells(env)) {
  # Pipeline mode lets every cell through to fit_cell, which fails one that is
  # not Gaussian mixed rather than leave it without a record.
  if (!env$pipeline && !("lmerTest" %in% unlist(cell[["oracles"]]))) next
  rec <- tryCatch(fit_cell(env, cell), error = function(e)
    grid_fail(grid_record(cell, "lmerTest", ENGINE_VERSION), conditionMessage(e)))
  grid_write(con, rec)
  cat(sprintf("lmerTest  %-42s  %s\n", cell[["cell"]], rec$status))
}
close(con)
