# dev_align.R -- per-engine x per-family deviance convention alignment for the
# accuracy grid. Shared convention: dev = -2 * (loglik + addend), where `loglik`
# is what the engine reported and `addend` is a CLOSED-FORM constant with its
# source cited, on the logLik scale.
#
# THE RULE THAT SHAPES THIS FILE: every entry is a closed-form constant with its
# source cited. A constant FITTED to close a gap is forbidden, and on Gamma it is
# forbidden twice over -- a fitted Gamma constant would hide exactly the deviance
# mismatch the gate exists to show. Where an engine/family pair has no confirmed
# closed form, its entry stays "unconfirmed" and aligned_dev returns NA with a
# `why`, which compare.R prints as a LOUD exclusion. NA is an honest answer; a
# made-up constant is not.
#
# EVERY ENTRY IS AN `align()` RECORD with four fields:
#   addend     "none", or a closed-form function(rec, df, cell) -> numeric(1) on
#              the loglik scale, with its derivation and source cited
#              immediately above.
#   status     "confirmed" | "derived" | "unconfirmed".
#   confirmed  the date the measurement behind a "confirmed" status was taken;
#              NA on the other two.
#   note       free text: what was measured, or why it is derived.
#
# THE THREE STATUSES, and the rule imposed on them. FITTING a constant to close a
# gap is forbidden, and so is CONFIRMING a constant on Gamma cells -- a constant
# fitted to close a Gamma gap would hide the deviance mismatch the gate exists to
# show. Neither rule forbids the deviance gate from RUNNING on Gamma; on the
# contrary, that gate is why the grid exists.
#
#   "confirmed"  the per-engine MECHANISM -- how this engine reports logLik --
#                was measured in the pilot on GLM cells and on gaussian /
#                binomial / poisson mixed cells, and agreed inside the loglik
#                band. `confirmed` carries the date. NEVER measured on Gamma.
#   "derived"    the SAME mechanism, applied to a family the pilot may not
#                confirm on (gamma, negativebinomial, inversegaussian). Nothing
#                was fitted: the addend is the identical closed form, or the
#                identical "none", that the confirmed families use. The deviance
#                gate therefore RUNS on Gamma cells, which is the point -- and if
#                the derivation is wrong the symptom is a loud |Delta| > dev_big
#                convention-mismatch FAIL, never a silent pass.
#   "unconfirmed" the pilot could not establish the mechanism at all.
#                aligned_dev returns NA + why, compare.R excludes the engine
#                loudly on those cells. An honest gap, not a guess.
#
# A "derived" entry is never edited because a Gamma cell failed. That is the one
# move this file exists to prevent.
align <- function(addend, status, confirmed = NA_character_, note = "") {
  stopifnot(status %in% c("confirmed", "derived", "unconfirmed"))
  list(addend = addend, status = status, confirmed = confirmed, note = note)
}

# Locate this file's own directory regardless of the caller's cwd (compare.R
# sources it by an absolute script_dir path; a standalone `Rscript -e
# 'source("dev_align.R")'` run has cwd == this directory instead). Frames are
# walked innermost-first so a nested source() still resolves to the file actually
# being read, and the basename test keeps a nested source() of some OTHER file
# from answering for this one.
#
# RESOLVED ONCE, DURING LOAD, and cached. `source()` records the path it is
# reading in the sourcing frame's `ofile`, and those frames exist only while the
# source() call is still on the stack. The one caller, grid_read_data, runs long
# afterwards, so an uncached walk would always miss and fall back to `--file=`
# (which names the SOURCER, not this file) or to the working directory. The call
# just below this definition is what makes the walk fire while it can still
# answer.
dev_align_dir_cache <- new.env(parent = emptyenv())
dev_align_dir <- function() {
  if (!is.null(dev_align_dir_cache$dir)) return(dev_align_dir_cache$dir)
  dir <- NULL
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (is.character(of) && length(of) == 1L && file.exists(of) &&
        identical(basename(of), "dev_align.R")) {
      dir <- dirname(normalizePath(of))
      break
    }
  }
  if (is.null(dir)) {
    arg <- grep("--file=", commandArgs(FALSE), value = TRUE)
    dir <- if (length(arg) == 1L) dirname(normalizePath(sub("--file=", "", arg)))
           else getwd()
  }
  dev_align_dir_cache$dir <- dir
  dir
}
invisible(dev_align_dir())

# The cell's data frame, read once per cell and kept. `cell[["data"]]` is an explicit
# path relative to this directory, so there is no name-to-directory convention to
# guess. Memoised because a constant that needs the response vector is evaluated
# once per ENGINE per cell, and re-reading a 30000-row CSV six times over is the
# difference between a comparator that runs in seconds and one that does not.
grid_data_cache <- new.env(parent = emptyenv())
grid_read_data <- function(cell) {
  key <- cell[["cell"]]
  if (!is.null(grid_data_cache[[key]])) return(grid_data_cache[[key]])
  df <- read.csv(file.path(dev_align_dir(), cell[["data"]]), stringsAsFactors = FALSE)
  grid_data_cache[[key]] <- df
  df
}

# --- closed forms -----------------------------------------------------------
# Response extraction. Two conventions appear in the grid: an aggregated
# `cbind(y, n - y) ~ ...` trials cell, which carries a trial-count column, and a
# bare response column otherwise. Bernoulli needs no special case -- with n = 1
# and y in {0, 1} the saturated-binomial sum below is identically 0 on every
# row (lchoose(1, y) = 0, and the y*log(p) and (n-y)*log(1-p) terms each hit
# their own skip), which is the well-known fact that a Bernoulli fit is always
# saturated. Plugging n = 1 into the same formula is correct, not an
# approximation.
# `cell[["response"]]` is the RESPONSE COLUMN NAME, always present in the manifest.
# Reading it -- rather than hardcoding "y"/"incidence" -- is what makes these
# constants work on the empirical cells, whose responses are `r2` (VerbAgg),
# `TICKS` (grouseticks), `angle` (cake) and so on.
binomial_response <- function(cell, df) {
  y <- df[[cell[["response"]]]]
  if (!is.null(cell[["weights"]])) return(list(y = y, n = df[[cell[["weights"]]]]))
  list(y = y, n = rep(1, nrow(df)))
}

# Saturated-model loglik, in the verified closed form of 2026-08-24. That
# derivation computes sat_* = -2 * logLik(saturated), an AIC-style constant; this
# returns the loglik itself (that value / -2), which is what aligned_dev ADDS to
# lme4's reported nAGQ>1 loglik. Adding the un-halved, unnegated sat_* instead
# overshoots by 2x; that is the failure mode the cbpp and grouseticks nAGQ=1
# goldens were kept to catch.
# Signature (rec, df, cell), like every ALIGN addend. Returns NA_real_ -- never
# stop()s -- on a family with no verified closed form, so aligned_dev's
# "NA + why" contract holds all the way down. The manifest's oracle lists already
# keep lme4 off gamma/NB AGQ cells, so this arm should be unreachable; returning
# NA rather than throwing means a manifest change cannot turn compare.R into a
# crash.
saturated_loglik <- function(rec, df, cell) {
  if (identical(cell[["family"]], "binomial")) {
    r <- binomial_response(cell, df)
    y <- r$y; n <- r$n; p <- y / n
    sat <- -2 * sum(lchoose(n, y) + ifelse(y > 0, y * log(p), 0) +
                      ifelse(y < n, (n - y) * log(1 - p), 0))
    return(-sat / 2)
  }
  if (identical(cell[["family"]], "poisson")) {
    y <- df[[cell[["response"]]]]
    sat <- -2 * sum(ifelse(y > 0, y * log(y), 0) - y - lfactorial(y))
    return(-sat / 2)
  }
  NA_real_
}

# Binomial-trials normalising constant, sum(lchoose(n_i, y_i)). An engine that
# fits the aggregated form as a PROPORTION with prior weights drops this term
# from its loglik while one that fits cbind(successes, failures) keeps it; the
# two objectives differ by exactly this data-only constant and by nothing else,
# so they share an argmin and every parameter estimate. Zero on a Bernoulli cell
# (lchoose(1, y) = 0), so it is safe to apply to the whole binomial family.
# No ALIGN entry reads it yet; it lives here so the closed form has one place to
# live, for an engine whose pilot measurement shows it drops the term.
lchoose_binomial <- function(rec, df, cell) {
  r <- binomial_response(cell, df)
  sum(lchoose(r$n, r$y))
}

# Gaussian REML constant, df_reml * (1 + log(2*pi)) / 2 with df_reml = n - p, on
# the loglik scale. An engine reporting a profiled REML DEVIANCE rather than the
# REML criterion is short by exactly this. lme4's logLik on a REML fit already
# carries the term, which is why lme4's gaussian entry is "none"; the constant
# exists for an engine whose pilot measurement shows it does not.
# Signature is (rec, df, cell) like every other ALIGN entry -- NOT a closure over
# a variable from aligned_dev's frame. `p` comes from the RECORD's own
# coef_names, so a rank-deficient fit that dropped an aliased column is counted
# as the engine actually counted it.
reml_gaussian_const <- function(rec, df, cell) {
  df_reml <- nrow(df) - length(rec$coef_names)
  df_reml * (1 + log(2 * pi)) / 2
}

# --- prior-weight gaussian cells: excluded, not aligned ---------------------
# A gaussian cell with a `weights_col` is two different likelihoods, not one
# likelihood reported two ways. R's weighted normal log-likelihood (stats::lm's
# logLik, and lme4's) counts the N rows and carries a 0.5*sum(log w) term,
#   0.5*sum(log w) - N/2 * (log(2*pi*D/N) + 1),      D = sum(w * resid^2),
# while GLM.jl's Normal log-likelihood -- what MixedModels.jl reports on a
# fixed-effects gaussian cell -- puts n = sum(w) in place of N and has no
# sum(log w) term at all,
#   -sum(w)/2 * (log(2*pi*D/sum(w)) + 1).
# The two differ by a SCALE on the log-likelihood, so no additive constant
# aligns them and nothing closed-form can be written here. These records are
# excluded from the deviance gate instead, which is what returning NA does.
#
# MEASURED 2026-09-22 on wls_basic (N = 200, sum(w) = 223.285466,
# sum(log w) = -2.973678, unlocked box): MixedModels reports -258.98477623
# where stats::lm and glmm report -244.47659409. The two formulas above
# reproduce that 14.50818213 gap to eight decimals and the coefficients agree on
# all four engines, so it is a reporting scale and not a different fit. lme4 and
# glmmTMB are oracles on every prior-weight gaussian cell in the manifest and
# both report stats::lm's scale there (glmmTMB through the dispersion offset in
# engines/glmmtmb.R), so gate 1 keeps a reference on each one.
#
# The exclusion is written here rather than left to the gate's own outlier
# test, which drops the oracle furthest from the MEDIAN of the candidates and so
# depends on how many oracles converged on the cell.
#
# `n_theta == 0` is the fixed-effects test, not `structure == "glm"`: the
# committed-fixture cells carry no `structure` field.
mixedmodels_gaussian <- function(rec, df, cell) {
  if (cell[["n_theta"]] == 0 && !is.null(cell[["weights_col"]])) return(NA_real_)
  0
}

# --- engines that maximise a different objective: excluded, not aligned ------
# glmm's Laplace log-likelihood builds log|A| from the OBSERVED curvature of the
# integrand at the mode, and on Gamma it maximises over the dispersion on
# fixed-effects and mixed cells alike, with each prior weight dividing its row's
# dispersion. That is glmmTMB's objective (on prior-weight cells through the
# dispersion offset in engines/glmmtmb.R), and three reference engines report
# a different one on some cells. A different objective has no alignment
# constant, so those records return NA and are excluded from gate 1 loudly.
# Gate 2 reads the same exclusions through objective_differs below, per
# quantity.
#
#   * lme4's glmer and glmer.nb build log|A| from the expected (Fisher) weight,
#     which differs from the observed one on every non-canonical link: probit
#     and cloglog binomial, and negative binomial's log. On a mixed Gamma cell
#     glmer also plugs in D/Σw for phi instead of maximising over it (pwrss/n
#     is only what sigma() reports downstream).
#   * On a fixed-effects Gamma cell lme4's record is stats::glm, whose logLik
#     evaluates the density at the plug-in dispersion deviance/n, not at the
#     maximum over it. MEASURED 2026-09-23 on gaml_glm_g3000 (3000 rows, log
#     link): stats::glm reports -3906.4959533778 at its plug-in 0.5546683, and
#     maximising the same log-likelihood over the shape gives -3901.0360811 at
#     shape 1.953085, which is glmmTMB's reported -3901.0360810 to 5e-08.
#   * MixedModels.jl's GLMM Laplace uses the Fisher weight as glmer's does, so
#     its mixed probit and cloglog cells are excluded the same way.
#
# glmmTMB's Gamma entry excludes no cell: glmmTMB maximises over the
# dispersion on fixed-effects cells as glmm does, and on prior-weight cells the
# dispersion offset in engines/glmmtmb.R gives it glmm's precision-weight
# likelihood.
fisher_laplace <- function(cell) {
  cell[["n_theta"]] > 0 &&
    (cell[["family"]] %in% c("negativebinomial") ||
     (identical(cell[["family"]], "binomial") &&
      isTRUE(cell[["link"]] %in% c("probit", "cloglog"))))
}
lme4_objective <- function(rec, df, cell) {
  if (identical(cell[["family"]], "gamma")) return(NA_real_)
  if (fisher_laplace(cell)) return(NA_real_)
  0
}
mixedmodels_binomial <- function(rec, df, cell) {
  if (fisher_laplace(cell)) return(NA_real_)
  0
}

# --- the table --------------------------------------------------------------
# Rows are engines, columns families. An `agq` key applies where the cell's
# nagq > 1 and is consulted BEFORE the family key.
#
# WHAT MAY MOVE AN ENTRY, and what may not. An entry becomes "confirmed" only on
# a measurement of that engine's reporting mechanism taken on GLM cells or on
# gaussian / binomial / poisson mixed cells, and the date of that measurement
# goes in the entry. A gamma / negativebinomial / inversegaussian entry is
# "derived" from the confirmed mechanism of the same engine and carries no date,
# because nothing is ever measured on those families. An entry with no
# measurement behind it is "unconfirmed", with the reason in its note. No entry
# is ever edited because a Gamma cell failed the deviance gate.
ALIGN <- list(
  glmm = list(
    # Fit::loglik is documented as "deviance with its dropped data-only
    # constants restored, on the logLik() scale (R/lme4)" -- the LMM path
    # reports -REMLcrit/2 and the GLMM path the marginal Laplace/AGQ
    # log-likelihood. So "none" on every family, and it is EVIDENCED: the three
    # confirmed entries below are glmm against stats::lm / stats::glm on the
    # fixed-effects cells and against lme4's logLik(merMod) on the mixed ones.
    gaussian         = align("none", "confirmed", "2026-09-22",
      "6 gaussian cells (stats::lm on the fixed-effects cell, REML mixed cells to 30000 rows, sleepstudy, a prior-weight cell): max |diff| 1.90e-06 against a 2e-06 band, worst on lmm_q4sx2_g30000p5_bal_base"),
    binomial         = align("none", "confirmed", "2026-09-22",
      "14 binomial cells -- Bernoulli and aggregated, logit/probit/cloglog, 60 to 30000 rows, cbpp, and one nAGQ=7 cell: max |diff| 4.18e-07 against a 1e-03 band"),
    poisson          = align("none", "confirmed", "2026-09-22",
      "4 poisson cells (stats::glm, mixed, offset, prior weights): max |diff| 3.8e-09 against a 1e-03 band"),
    # One reporting path serves every family: the family chooses which
    # likelihood terms are summed, never what the crate reports about the sum.
    gamma            = align("none", "derived",
      note = "same reporting mechanism as the confirmed gaussian, binomial and poisson entries for this engine: Fit::loglik has one reporting path and the family enters only the likelihood terms it sums"),
    negativebinomial = align("none", "derived",
      note = "same reporting mechanism as the confirmed gaussian, binomial and poisson entries for this engine: Fit::loglik has one reporting path and the family enters only the likelihood terms it sums"),
    inversegaussian  = align("none", "derived",
      note = "same reporting mechanism as the confirmed gaussian, binomial and poisson entries for this engine: Fit::loglik has one reporting path and the family enters only the likelihood terms it sums")),
  lme4 = list(
    # lme4 is the convention anchor: logLik(merMod) is the scale every other
    # entry is measured against. The fixed-effects cells are fitted by
    # stats::lm / stats::glm / MASS::glm.nb (the record's `message` says which),
    # so there logLik IS stats::logLik by construction; on the mixed cells the
    # evidence is that three independent engines land on the same number.
    gaussian         = align("none", "confirmed", "2026-09-22",
      "stats::lm on the fixed-effects cells; on 6 mixed gaussian cells glmm, glmmTMB and MixedModels each sit within 1.90e-06 of logLik(merMod)"),
    binomial         = align(lme4_objective, "confirmed", "2026-09-22",
      "stats::glm on the fixed-effects cells; on 11 to 14 binomial cells glmm, glmmTMB and MixedModels each sit within 4.18e-07 of logLik(merMod). Mixed probit and cloglog cells return NA and are excluded since 2026-09-24, when glmm's Laplace moved to the observed curvature (see lme4_objective above)"),
    poisson          = align("none", "confirmed", "2026-09-22",
      "stats::glm on the fixed-effects cells; on 3 poisson cells glmm, glmmTMB and MixedModels each sit within 4.3e-09 of logLik(merMod)"),
    # MASS::glm.nb, stats::glm and glmer all reach logLik the same way; the
    # family changes the density, not the reporting.
    gamma            = align(lme4_objective, "derived",
      note = "same reporting mechanism as the confirmed gaussian, binomial and poisson entries for this engine, but every gamma record returns NA and is excluded: stats::glm's plug-in dispersion on the fixed-effects cells, and glmer's D/Σw plug-in dispersion (with the Fisher-weight Laplace on the log link) on the mixed ones, are not glmm's objective (see lme4_objective above)"),
    negativebinomial = align(lme4_objective, "derived",
      note = "same reporting mechanism as the confirmed gaussian, binomial and poisson entries for this engine: R's logLik is one generic over lm, glm and merMod and the family enters only the density it sums. Mixed cells return NA and are excluded: glmer.nb's Fisher-weight Laplace is not glmm's objective (see lme4_objective above)"),
    inversegaussian  = align("none", "derived",
      note = "same reporting mechanism as the confirmed gaussian, binomial and poisson entries for this engine: R's logLik is one generic over lm, glm and merMod and the family enters only the density it sums"),
    # lme4's nAGQ>1 logLik omits the saturated term. Verified 2026-08-24 against
    # the frozen cbpp and grouseticks goldens: adding it closes an 84-unit /
    # 931-unit raw deviance gap to under 1 unit. Carried over CONFIRMED, with
    # that date, because the measurement is on file. The manifest's oracle lists
    # keep lme4 off gamma and NB AGQ cells, where saturated_loglik has no closed
    # form.
    agq = align(saturated_loglik, "confirmed", "2026-08-24",
                "closed-form saturated logLik; verified against the frozen cbpp and grouseticks nAGQ goldens, and again 2026-09-22 on pois_int1_g3000p20_bal_base_agq7, where lme4's reported logLik plus the term lands 5.2e-08 from GLMMadaptive's")),
  glmmTMB = list(
    # On prior-weight cells glmmTMB fits the dispersion offset
    # (engines/glmmtmb.R), which reports lme4's and stats::lm's logLik.
    gaussian         = align("none", "confirmed", "2026-09-22",
      "5 gaussian cells without prior weights (stats::lm, REML mixed to 30000 rows, sleepstudy, sim_slope_lmm): max |diff| 1.12e-06 against a 2e-06 band. 10 prior-weight gaussian cells with the dispersion offset (wls_basic, path_*, lmm_intercept, lmm_slope, lmm_crossed, lmm_*_wts), glmmTMB run of 2026-09-27: max |diff| 2.19e-08 against lme4's logLik"),
    # The probit and cloglog cells are a different OPTIMUM, not a different
    # report, so they say nothing about the mechanism: glmmTMB's logLik there is
    # 0.13 to 1.55 ABOVE lme4's while its beta is up to 1.3e-01 from lme4's and
    # glmm's beta is 1.7e-05 from it. Reporting that gap is gate 1's job.
    binomial         = align("none", "confirmed", "2026-09-22",
      "9 logit cells -- Bernoulli and aggregated, fixed-effects and mixed, 60 to 30000 rows, cbpp: max |diff| 2.2e-07 against a 1e-03 band"),
    poisson          = align("none", "confirmed", "2026-09-22",
      "4 poisson cells (stats::glm, mixed, offset, prior weights): max |diff| 4.3e-09 against a 1e-03 band"),
    gamma            = align("none", "derived",
      note = "same reporting mechanism as the confirmed binomial and poisson entries for this engine: glmmTMB reports the same marginal Laplace log-likelihood whatever the family, and the family enters only the density TMB integrates. Since 2026-09-24 no gamma cell is excluded: glmm maximises over the dispersion on fixed-effects cells as glmmTMB does, and on prior-weight cells the dispersion offset in engines/glmmtmb.R gives glmm's precision-weight likelihood"),
    negativebinomial = align("none", "derived",
      note = "same reporting mechanism as the confirmed binomial and poisson entries for this engine: glmmTMB reports the same marginal Laplace log-likelihood whatever the family, and the family enters only the density TMB integrates")),
    # no inversegaussian key: the manifest never sends glmmTMB such a cell
  GLMMadaptive = list(
    binomial         = align("none", "confirmed", "2026-09-22",
      "binb_int1_g3000p20_bal_base_agq7, the one binomial cell this engine is sent that another engine also fits on the AGQ arm: 1.5e-09 from lme4's saturated-term-aligned nAGQ=7 logLik"),
    poisson          = align("none", "confirmed", "2026-09-22",
      "pois_int1_g3000p20_bal_base_agq7: 5.2e-08 from lme4's saturated-term-aligned nAGQ=7 logLik, which also re-confirms that term on a Poisson cell"),
    gamma            = align("none", "derived",
      note = "same reporting mechanism as the confirmed binomial and poisson entries for this engine: GLMMadaptive reports the adaptive-quadrature marginal log-likelihood whatever the family, and the family enters only the density it quadratures"),
    negativebinomial = align("none", "derived",
      note = "same reporting mechanism as the confirmed binomial and poisson entries for this engine: GLMMadaptive reports the adaptive-quadrature marginal log-likelihood whatever the family, and the family enters only the density it quadratures")),
    # no gaussian / inversegaussian keys: the manifest never sends either
  MixedModels = list(
    # Prior-weight FIXED-EFFECTS gaussian cells are excluded rather than
    # aligned -- see mixedmodels_gaussian above. The prior-weight MIXED cell is
    # not: MixedModels reaches GLM.jl only without a random effect.
    gaussian         = align(mixedmodels_gaussian, "confirmed", "2026-09-22",
      "6 gaussian cells including the prior-weight mixed cell: max |diff| 1.86e-06 against a 2e-06 band. Prior-weight fixed-effects cells return NA and are excluded"),
    binomial         = align(mixedmodels_binomial, "confirmed", "2026-09-22",
      "11 binomial cells -- Bernoulli and aggregated, logit and probit, fixed-effects and mixed, 60 to 30000 rows, cbpp: max |diff| 3.8e-07 against a 1e-03 band. Mixed probit and cloglog cells return NA and are excluded since 2026-09-24 (see lme4_objective above)"),
    poisson          = align("none", "confirmed", "2026-09-22",
      "4 poisson cells (GLM, mixed, offset, prior weights): max |diff| 3.4e-09 against a 1e-03 band"))
)
# The two ports ARE glmm. Aliasing rather than copying keeps them from drifting.
ALIGN$glmm_python <- ALIGN$glmm
ALIGN$glmm_r <- ALIGN$glmm

# --- the resolver -----------------------------------------------------------
# The ALIGN entry for this record's engine on this cell, or NULL when the engine
# has no row or no entry for the family.
align_entry <- function(rec, cell) {
  tbl <- ALIGN[[rec$engine]]
  if (is.null(tbl)) return(NULL)
  nagq <- if (is.null(cell[["nagq"]])) 1L else as.integer(cell[["nagq"]])
  if (nagq > 1L && !is.null(tbl$agq)) tbl$agq else tbl[[cell[["family"]]]]
}

# NEVER stop()s. Every failure mode returns NA_real_ with a `why` attribute, and
# compare.R prints it as a LOUD exclusion. A comparator that can crash on one
# unexpected cell cannot be run overnight.
aligned_dev <- function(rec, cell) {
  fail <- function(why) { out <- NA_real_; attr(out, "why") <- why; out }
  ll <- rec$loglik
  if (is.null(ll) || !is.numeric(ll) || length(ll) != 1L || !is.finite(ll)) {
    return(fail(sprintf("%s reported no finite loglik", rec$engine)))
  }
  if (is.null(ALIGN[[rec$engine]])) {
    return(fail(sprintf("no alignment row for engine %s", rec$engine)))
  }
  entry <- align_entry(rec, cell)
  if (is.null(entry)) {
    return(fail(sprintf("%s/%s: no alignment entry (this engine does not fit this family)",
                        rec$engine, cell[["family"]])))
  }
  if (identical(entry$status, "unconfirmed")) {
    return(fail(sprintf("%s/%s: alignment mechanism not yet confirmed (pilot)",
                        rec$engine, cell[["family"]])))
  }
  addend <- if (identical(entry$addend, "none")) 0
            else entry$addend(rec, grid_read_data(cell), cell)
  if (!is.numeric(addend) || length(addend) != 1L || !is.finite(addend)) {
    # An addend that returns NA does so BY DESIGN -- an engine, family or
    # weighting where the two reports have no closed form between them -- so the
    # exclusion says that rather than describing the NA as a broken constant.
    why <- if (is.numeric(addend) && length(addend) == 1L && is.na(addend))
             "no closed-form alignment constant for this cell"
           else "alignment constant is not a finite number"
    return(fail(sprintf("%s/%s: %s", rec$engine, cell[["family"]], why)))
  }
  -2 * (ll + addend)
}

# TRUE when this record is one of the exclusions above: its entry is a closed
# form that returns NA on this cell by design (lme4_objective,
# mixedmodels_binomial, mixedmodels_gaussian, or
# saturated_loglik on a family it has no closed form for). This is narrower than
# is.na(aligned_dev(...)): a record with no finite loglik, or an engine with no
# entry for the family, is not on a different objective. compare.R's gate 2
# reads it, per quantity.
objective_differs <- function(rec, cell) {
  entry <- align_entry(rec, cell)
  if (is.null(entry) || !is.function(entry$addend)) return(FALSE)
  is.na(entry$addend(rec, grid_read_data(cell), cell))
}

# Which entries are asserted rather than measured -- printed by compare.R's
# header so a run always says what its deviance scale rests on.
align_status <- function() {
  do.call(rbind, lapply(names(ALIGN), function(e)
    do.call(rbind, lapply(names(ALIGN[[e]]), function(f) data.frame(
      engine = e, family = f, status = ALIGN[[e]][[f]]$status,
      confirmed = ALIGN[[e]][[f]]$confirmed, stringsAsFactors = FALSE)))))
}

# --- GLMMadaptive's vector-AGQ rule, evaluated at glmm's point ---------------
# With q >= 2 random effects per group, nAGQ > 1 is a tensor-product
# Gauss-Hermite grid, and a product grid is not rotation-invariant: the value
# depends on which square root of the mode's posterior covariance places it.
# glmm (src/glmm/agq.rs, agq_deviance_vec) places the nodes in u-space
# (b = Lambda u) at u_hat + sqrt(2) L_A^{-T} z, with A = Lambda' Z'WZ Lambda + I =
# L_A L_A'. GLMMadaptive 0.9-7 (GHfun) places them in b-space at
# b_hat + sqrt(2) R^{-1} z, with R = chol(H_b) upper and H_b = Z'WZ + D^{-1}. Both
# are valid adaptive rules for the same integral, but at one point they differ by
# 2.6e-4 (binb_q2s) and 7.4e-5 (pois_q2s), above dev_eps. There is no closed-form
# constant between them, so gate 1 compares like for like instead: GLMMadaptive's
# rule at glmm's point against GLMMadaptive's rule at GLMMadaptive's point (its
# reported deviance).
#
# Written in u-space so it needs no D^{-1}. GLMMadaptive's node matrix in u-space
# is M = L_A^{-T} Q, where F = Lambda L_A^{-T} = U Q' is the RQ factorization of F
# (U upper triangular, Q orthogonal), because Lambda M = U is then the unique
# upper-triangular root of H_b^{-1}. Nothing here divides by an entry of Lambda
# or D. Off the boundary the value depends only on D = Lambda Lambda', so Lambda
# is rebuilt as chol(D) from the record's stddev and corr (written round-trip
# exact); glmm's own Lambda is never needed.
# On a singular D the rule has no limit -- its value depends on the direction of
# approach -- so there is nothing to evaluate and the function returns NA.
#
# GLMMadaptive's H_b is a central difference of the score, the OBSERVED
# information, so the negative-binomial and Gamma weights below are the observed
# ones; on the canonical logit and log links the two coincide.
#
# Families: binomial/logit, poisson/log, negativebinomial/log, gamma/log.
# Anything else returns NA with a `why`, never a guess.

# Cells where gate 1 scores GLMMadaptive's comparison under its own rule: AGQ
# with more than one random effect per group. GLMMadaptive fits exactly one
# grouping factor, so n_theta > 1 is q >= 2. The two rules coincide at q = 1.
ga_rule_cell <- function(cell) {
  !is.null(cell[["nagq"]]) && cell[["nagq"]] > 1 && cell[["n_theta"]] > 1
}

# Gauss-Hermite nodes and weights for weight exp(-x^2), by Golub-Welsch.
gh_nodes <- function(k) {
  J <- matrix(0, k, k)
  off <- sqrt(seq_len(k - 1L) / 2)
  J[cbind(seq_len(k - 1L), 2:k)] <- off
  J[cbind(2:k, seq_len(k - 1L))] <- off
  e <- eigen(J, symmetric = TRUE)
  list(x = e$values, w = sqrt(pi) * e$vectors[1, ]^2)
}

# Householder RQ of a square F: F = U Q' with U upper triangular. Rows are
# reduced from the bottom (LAPACK dgerqf order); a zero row takes the identity
# reflector, as LAPACK does. Column signs of Q do not matter: the product grid is
# symmetric in every coordinate.
rq_q <- function(F) {
  q <- nrow(F); Q <- diag(q); U <- F
  if (q < 2L) return(Q)
  for (i in q:2) {
    x <- U[i, 1:i]; nx <- sqrt(sum(x^2))
    if (nx == 0) next
    v <- x; v[i] <- v[i] + (if (x[i] >= 0) 1 else -1) * nx
    H <- diag(i) - 2 * tcrossprod(v) / sum(v^2)
    U[, 1:i] <- U[, 1:i] %*% H
    Q[, 1:i] <- Q[, 1:i] %*% H
  }
  Q
}

# Per-row log density, score d/d eta and observed weight -d^2/d eta^2, or NULL
# for a family/link with no evaluator.
ga_rule_family <- function(cell, df, rec) {
  fam <- cell[["family"]]; link <- cell[["link"]]
  if (identical(fam, "binomial") && identical(link, "logit")) {
    r <- binomial_response(cell, df); y <- r$y; n <- r$n
    return(list(
      ll = function(eta) y * eta - n * (pmax(eta, 0) + log1p(exp(-abs(eta)))) + lchoose(n, y),
      score = function(eta) y - n * plogis(eta),
      w = function(eta) { p <- plogis(eta); n * p * (1 - p) }))
  }
  y <- df[[cell[["response"]]]]
  if (identical(fam, "poisson") && identical(link, "log")) {
    return(list(ll = function(eta) y * eta - exp(eta) - lgamma(y + 1),
                score = function(eta) y - exp(eta),
                w = function(eta) exp(eta)))
  }
  if (identical(fam, "negativebinomial") && identical(link, "log")) {
    th <- as.numeric(rec$nb_theta)
    return(list(
      ll = function(eta) lgamma(y + th) - lgamma(th) - lgamma(y + 1) + th * log(th) +
        y * eta - (th + y) * log(exp(eta) + th),
      score = function(eta) { mu <- exp(eta); th * (y - mu) / (mu + th) },
      w = function(eta) { mu <- exp(eta); th * mu * (th + y) / (mu + th)^2 }))
  }
  # GLMMadaptive's Gamma.fam(): dgamma(y, shape = nu, scale = mu / nu), with the
  # dispersion 1/nu. Every record's `sigma` is that dispersion's square root, so
  # nu = 1 / sigma^2.
  if (identical(fam, "gamma") && identical(link, "log")) {
    nu <- 1 / as.numeric(rec$sigma)^2
    return(list(ll = function(eta) stats::dgamma(y, shape = nu, scale = exp(eta) / nu, log = TRUE),
                score = function(eta) nu * (y * exp(-eta) - 1),
                w = function(eta) nu * y * exp(-eta)))
  }
  NULL
}

# -2 x GLMMadaptive's nAGQ log-likelihood at the point `rec` reports, on the
# cell's data, or NA with a `why`. `rec` is any engine's record: glmm's for the
# gate, GLMMadaptive's own for a self-check.
ga_rule_dev <- function(rec, cell) {
  fail <- function(why) { out <- NA_real_; attr(out, "why") <- why; out }
  df <- grid_read_data(cell)
  fm <- ga_rule_family(cell, df, rec)
  if (is.null(fm)) return(fail(sprintf("no GLMMadaptive-rule evaluator for %s/%s",
                                       cell[["family"]], cell[["link"]])))
  mf <- stats::model.frame(stats::as.formula(cell[["ma_fixed"]]), df)
  X <- stats::model.matrix(stats::as.formula(cell[["ma_fixed"]]), mf)
  off0 <- stats::model.offset(mf); if (is.null(off0)) off0 <- 0
  rparts <- strsplit(cell[["ma_random"]], "|", fixed = TRUE)[[1]]
  Z <- stats::model.matrix(stats::as.formula(rparts[1]), df)
  gid <- as.integer(factor(df[[trimws(rparts[2])]]))
  G <- max(gid); q <- ncol(Z)
  beta <- as.numeric(rec$beta)[match(colnames(X), rec$coef_names)]
  if (anyNA(beta) || length(beta) != ncol(X)) return(fail("coefficient names do not match ma_fixed"))
  vc <- rec$varcomp
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  if (length(vc) != 1L || !identical(as.character(unlist(vc[[1]]$terms)), colnames(Z)))
    return(fail("random-effect terms do not match ma_random"))
  sd <- stddevs_of(rec)
  R <- diag(q); R[upper.tri(R)] <- corrs_of(rec); R[lower.tri(R)] <- t(R)[lower.tri(R)]
  Lam <- tryCatch(t(chol(diag(sd, q) %*% R %*% diag(sd, q))), error = function(e) NULL)
  if (is.null(Lam)) return(fail("singular random-effect covariance: GLMMadaptive's rule has no limit there"))

  # Per-group u-space mode by Newton; every family above has a positive weight,
  # so A is positive definite at every step.
  eta0 <- off0 + drop(X %*% beta); ZL <- Z %*% Lam
  pairs <- expand.grid(a = seq_len(q), b = seq_len(q))
  A_of <- function(w) {
    S <- vapply(seq_len(nrow(pairs)), function(k)
      rowsum(ZL[, pairs$a[k]] * ZL[, pairs$b[k]] * w, gid, reorder = TRUE)[, 1], numeric(G))
    lapply(seq_len(G), function(c) matrix(S[c, ], q, q) + diag(q))
  }
  U <- matrix(0, G, q); done <- FALSE
  for (it in 1:100) {
    eta <- eta0 + rowSums(ZL * U[gid, , drop = FALSE])
    grad <- rowsum(ZL * fm$score(eta), gid, reorder = TRUE) - U
    A <- A_of(fm$w(eta))
    step <- matrix(vapply(seq_len(G), function(c) solve(A[[c]], grad[c, ]), numeric(q)),
                   G, q, byrow = TRUE)
    U <- U + step
    if (max(abs(step)) < 1e-12) { done <- TRUE; break }
  }
  if (!done) return(fail("GLMMadaptive-rule evaluator: mode search did not converge"))
  A <- A_of(fm$w(eta0 + rowSums(ZL * U[gid, , drop = FALSE])))

  gh <- gh_nodes(as.integer(cell[["nagq"]]))
  z <- as.matrix(expand.grid(rep(list(gh$x), q)))
  lw <- log(apply(as.matrix(expand.grid(rep(list(gh$w), q))), 1, prod)) + rowSums(z^2)
  Ms <- vector("list", G); ldM <- numeric(G)
  for (c in seq_len(G)) {
    LA <- t(chol(A[[c]]))
    M <- backsolve(t(LA), diag(q))            # L_A^{-T}
    Ms[[c]] <- M %*% rq_q(Lam %*% M)
    ldM[c] <- -sum(log(diag(LA)))
  }
  vals <- matrix(0, G, nrow(z))
  for (j in seq_len(nrow(z))) {
    Uj <- U + sqrt(2) * matrix(vapply(Ms, function(M) drop(M %*% z[j, ]), numeric(q)),
                               G, q, byrow = TRUE)
    eta <- eta0 + rowSums(ZL * Uj[gid, , drop = FALSE])
    vals[, j] <- rowsum(fm$ll(eta), gid, reorder = TRUE)[, 1] - 0.5 * rowSums(Uj^2) -
      (q / 2) * log(2 * pi) + lw[j]
  }
  m <- apply(vals, 1, max)
  -2 * sum((q / 2) * log(2) + ldM + m + log(rowSums(exp(vals - m))))
}
