#!/usr/bin/env Rscript
# Weights-tier reference fits -> validation/goldens/weights/<name>.json.
#
# THE ORACLE IS SACRED. These JSONs are the frozen references the weights tier
# (the `datasets` rows of manifest.json with `tier: "weights"`, rungs 29-43) is
# held to by the oracle tier (tests/validation_oracle.rs::weights_corpus). On any
# glmm disagreement, glmm is presumed wrong: never relax a tolerance or edit a
# reference. A reference is regenerated only when its model spec (formula,
# family, link) is proven wrong, with a recorded reason.
#
# Fixed-only rows are fit by lm / glm / MASS::glm.nb, mixed rows by lmer / glmer
# at nAGQ = 1. Each JSON keeps the harness's own schema (`dataset`, no `kind`, no
# `r_formula`: the model lives in the manifest row), which weights_corpus adapts.
# `engine` / `engine_version` stamp the fitter and its version, checked against
# grid/versions.json by tests/golden_provenance.rs: `stats::lm` and `stats::glm`
# carry R's own version (`R-<x.y.z>`), `MASS::glm.nb` MASS's, and a mixed row
# `lme4` with lme4's.
#
# A BARE RUN REFITS AND OVERWRITES ALL 15 COMMITTED REFERENCES, so it is a
# deliberate act, never a routine step. One row:
#   VALIDATION_ONLY=wls_basic Rscript validation/tools/goldens_weights.R
# (a comma-separated list names several). lme4, MASS and numDeriv must be at
# their grid/versions.json pins and R at its `R` pin, or the run stops before
# fitting anything.
# Timing is off unless VALIDATION_TIMINGS is set (see timing_runs below).

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

# The packages that write numbers into these references, against
# grid/versions.json; a mismatch stops the run.
local({
  grid_dir <- file.path(suite_dir, "grid")
  source(file.path(grid_dir, "engines", "versions.R"), local = TRUE)
  V <- grid_versions(grid_dir)
  for (pkg in c("lme4", "MASS", "numDeriv")) assert_pkg_version(pkg, V[[pkg]])
  # stats ships with R: the fixed-only rows' stamp is R's version.
  if (!identical(norm_version(getRversion()), norm_version(V[["R"]])))
    stop(sprintf("R version mismatch: grid/versions.json pins %s, running %s",
                 V[["R"]], getRversion()), call. = FALSE)
})

suppressMessages({
  library(lme4)
  library(jsonlite)
  library(numDeriv)
})

manifest <- fromJSON(file.path(suite_dir, "manifest.json"), simplifyDataFrame = FALSE)
data_dir_of <- function(spec)
  file.path(suite_dir, "data",
            if (identical(spec$source, "sim")) "simulated" else "empirical")
out_dir <- file.path(suite_dir, "goldens", "weights")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Timing is OPT-IN. VALIDATION_TIMINGS unset / "" / "0" means do not time
# (`timing` is written null); otherwise it IS the sample count, an integer >= 2,
# first pass discarded, median of the rest. A malformed value is an error rather
# than a silent skip. Each JSON records its own n_runs, so files timed under any
# count stay self-describing.
timing_runs <- function() {
  v <- trimws(Sys.getenv("VALIDATION_TIMINGS"))
  if (v == "" || v == "0") return(NULL)
  n <- suppressWarnings(as.integer(v))
  if (is.na(n) || n < 2)
    stop("VALIDATION_TIMINGS must be 0 or an integer >= 2 (got '", v,
         "'); N=2 keeps 1 sample after the warm-up discard")
  n
}
N_RUNS <- timing_runs()  # NULL on an untimed run

read_dataset <- function(spec) {
  # `data` field: CSV to read when it differs from the rung name. Absent = the
  # name itself.
  src_name <- if (is.null(spec$data)) spec$name else spec$data
  df <- read.csv(file.path(data_dir_of(spec), paste0(src_name, ".csv")),
                 stringsAsFactors = FALSE)
  # Re-establish factor typing for grouping + categorical fixed-effect columns
  # (the CSV round-trip loses it). Coercion order is the sorted-level default, so
  # the treatment-contrast base is the first sorted level.
  for (f in unlist(spec$factors)) df[[f]] <- factor(df[[f]])
  df
}

# TRUE when the rung has a random term. Fixed-only rungs are fit via
# lm/glm/MASS::glm.nb below and skip every lme4-specific step (glmerControl,
# vcov(use.hessian=), VarCorr, isSingular).
is_mixed <- function(spec) grepl("|", spec$r_formula, fixed = TRUE)
# Prior weights, when the rung declares a `weights_col`. NULL = unit weights. The
# aggregated-binomial `weights` field is separate: there the trial counts enter
# through the cbind() response, not a weights= argument.
weights_of <- function(spec, df)
  if (is.null(spec$weights_col)) NULL else df[[spec$weights_col]]
# Known per-row additive linear-predictor offset (the manifest `offset` key) --
# NULL = no offset. Mirrors weights_of's plain named-column lookup.
offset_of <- function(spec, df)
  if (is.null(spec$offset)) NULL else df[[spec$offset]]

fit_call <- function(spec, df) {
  fm <- as.formula(spec$r_formula)
  w <- weights_of(spec, df)
  o <- offset_of(spec, df)
  if (!is_mixed(spec)) {
    return(switch(spec$family,
      gaussian = function() lm(fm, data = df, weights = w, offset = o),
      binomial = function() glm(fm, family = binomial(), data = df, weights = w, offset = o),
      poisson  = function() glm(fm, family = poisson(), data = df, weights = w, offset = o),
      gamma    = function() glm(fm, family = Gamma(link = "log"), data = df, weights = w, offset = o),
      # glm.nb() has no offset= parameter (MASS forwards ... to glm.control,
      # which errors on it, even NULL) -- an offset would need an offset()
      # term inside the formula instead. No manifest rung is both negbin and
      # offset-bearing, so this stays unsupported rather than guessed at.
      negbin   = function() MASS::glm.nb(fm, data = df, weights = w),
      stop("unsupported fixed-only family: ", spec$family)))
  }
  if (spec$family == "gaussian") {
    function() lmer(fm, data = df, REML = isTRUE(spec$reml), weights = w, offset = o)
  } else {
    # `link` field: non-canonical link override. Absent = the family's canonical
    # link.
    fam <- switch(spec$family,
      binomial = binomial(link = if (is.null(spec$link)) "logit" else spec$link),
      poisson  = poisson(),
      gamma    = Gamma(link = if (is.null(spec$link)) "log" else spec$link),
      stop("unsupported family: ", spec$family))
    # nAGQ = 1 (Laplace) PINNED, not defaulted: glmm's GLMM kernel is
    # glmer-faithful at nAGQ = 1, so the reference must be Laplace to compare
    # like with like.
    # tolPwrss = 1e-13 (default 1e-7), recorded in the result JSON: glmer's ldL2
    # is assembled from pp$Xwts -- working weights one PIRLS iteration behind the
    # mode -- so at the default tolPwrss its devfun sits ~5.6e-4 above the true
    # Laplace deviance (cbpp) and vcov(use.hessian=TRUE)/logLik carry ~1% spurious
    # theta/theta-beta curvature. 1e-13 is the value a per-rung sweep (2026-07-04)
    # found all four GLMM rungs of the main corpus agree at (1e-16 aborts in
    # step-halving; stay above lme4's numeric floor). A solver-precision setting,
    # not a spec change: the model (formula/family/link/nAGQ) is untouched.
    function() glmer(fm, data = df, family = fam, weights = w, offset = o,
                     nAGQ = 1, control = glmerControl(tolPwrss = 1e-13))
  }
}
# Mirrors fit_call's glmerControl -- change together. Recorded per result JSON so
# frozen references are self-describing (like n_runs).
TOLPWRSS <- 1e-13

# Times `batch` fits per sample so sub-resolution fits stay above the timer floor
# (Sys.time() resolution is coarse against a fit of tens of microseconds).
# fit_seconds_median is then the median time for `fits_per_sample` fits; divide
# by it for the per-fit estimate.
time_fit <- function(make_fit, batch = 1L) {
  times <- numeric(N_RUNS)
  for (i in seq_len(N_RUNS)) {
    t0 <- Sys.time()
    for (b in seq_len(batch)) invisible(make_fit())
    times[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  list(fit_seconds_median = median(times[-1]), n_runs = N_RUNS,
       warmup_discarded = 1L, fits_per_sample = batch)
}

# Timing for one dataset. Gaussian and fixed-only rungs: one median. GLMM rungs:
# SPLIT by SE method -- vcov(use.hessian=TRUE) runs numDeriv on the deviance (the
# main time consumer), vcov(use.hessian=FALSE) is the cheap RX block.
time_one <- function(spec, make_fit) {
  # Untimed run: no loops at all. NA rather than NULL because jsonlite writes an
  # empty object for a NULL list element and `null` for NA under na = "null".
  if (is.null(N_RUNS)) return(NA)
  batch <- if (is.null(spec$timing_batch)) 1L else as.integer(spec$timing_batch)
  if (!is_mixed(spec)) return(time_fit(make_fit, batch))
  if (spec$family == "gaussian") return(time_fit(make_fit, batch))
  t_rx <- time_fit(function() suppressWarnings(vcov(make_fit(), use.hessian = FALSE)), batch)
  t_h  <- time_fit(function() vcov(make_fit(), use.hessian = TRUE), batch)
  list(fit_seconds_median_rx = t_rx$fit_seconds_median,
       fit_seconds_median_hessian = t_h$fit_seconds_median,
       n_runs = N_RUNS, warmup_discarded = 1L, fits_per_sample = batch)
}

# SE of each RE standard deviation for a GLMM, from the joint (theta, beta) Hessian
# of glmer's Laplace deviance -- the merDeriv-style estimator glmm's WaldSe::Hessian
# path also uses. glmer's `devFunOnly` deviance is -2*logL(theta, beta) at nAGQ=1,
# so observed info = H/2 and cov = 2*H^{-1}; the sqrt of the theta-block diagonal is
# the SE of theta. For an intercept-only (scalar, q=1) grouping and a GLMM (residual
# scale sigma == 1) the RE stddev EQUALS its theta, so that SE is the stddev SE
# directly (getME theta order == VarCorr(m) grouping order == the varcomp_of order,
# so the vector aligns positionally). Returns one SE per grouping, in VarCorr order.
theta_hessian_stddev_se <- function(m) {
  th <- getME(m, "theta")
  pars <- c(th, fixef(m))
  devfun <- update(m, devFunOnly = TRUE)
  H <- numDeriv::hessian(devfun, pars)
  cov <- 2 * solve(H)   # deviance = -2logL => info = H/2 => cov = info^{-1} = 2 H^{-1}
  sqrt(diag(cov)[seq_along(th)])
}

# VarCorr -> common representation: per random term, the RE term names, their
# standard deviations, and the correlation matrix between them (lme4's
# `stddev`/`correlation` attributes, absolute scale). `sd_se`, when supplied (GLMM
# rungs), is a per-grouping stddev-SE vector in VarCorr order (see
# `theta_hessian_stddev_se`) attached as `stddev_se`.
varcomp_of <- function(m, sd_se = NULL) {
  vc <- VarCorr(m)
  nm <- names(vc)
  lapply(seq_along(nm), function(i) {
    g <- nm[i]
    block <- vc[[g]]
    sd <- attr(block, "stddev")
    corr <- attr(block, "correlation")
    # I() keeps length-1 vectors (single-term groupings) as JSON arrays under
    # auto_unbox, so positional comparison is uniform across rungs.
    entry <- list(group = g, terms = I(names(sd)),
                  stddev = I(unname(sd)), corr = unname(corr))
    if (!is.null(sd_se)) entry$stddev_se <- I(unname(sd_se[i]))
    entry
  })
}

# Emit one result JSON + console line (shared tail of fit_one and fit_one_fixed).
write_result <- function(spec, res) {
  out <- file.path(out_dir, paste0(spec$name, ".json"))
  # digits = NA: full double precision -- this is an oracle, not a display.
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  # `$` on the NA an untimed run leaves behind would error ("invalid for atomic
  # vectors"), so test for the list first; the console line then just omits the time.
  t_disp <- if (!is.list(res$timing)) NULL
            else if (!is.null(res$timing$fit_seconds_median)) res$timing$fit_seconds_median
            else res$timing$fit_seconds_median_hessian  # GLMM: show the heavier method
  cat(sprintf("%-12s  %-18s  rung %d  converged=%s singular=%s%s\n",
              res$engine, spec$name, spec$rung, res$converged, res$singular,
              if (is.null(t_disp)) "" else sprintf("  fit_median=%.4gs", t_disp)))
}

# Fixed-only rungs: lm / glm / MASS::glm.nb reference fits, same result schema
# minus the mixed-model-only fields (varcomp empty, no tolPwrss). Non-gaussian SE
# lands in `se_rx`: the single GLM SE has no Rx-vs-Hessian method split.
fit_one_fixed <- function(spec, df, make_fit, m) {
  # Dropped-rows gate: rows with near-zero weight must be effective row
  # deletion -- the weighted fit's beta has to match the same fit with those rows
  # REMOVED. beta only, by construction: the near-zero rows contribute nothing to
  # the RSS but still count in the residual df, so sigma and the SEs differ
  # between the two fits.
  if (identical(spec$gate, "dropped_rows")) {
    keep <- weights_of(spec, df) > 1e-6
    dfk  <- df[keep, ]
    mk   <- lm(as.formula(spec$r_formula), data = dfk, weights = weights_of(spec, dfk))
    d <- max(abs(coef(m) - coef(mk)) / pmax(abs(coef(m)), 1e-12))
    if (d > 1e-8) stop(sprintf(
      "%s: dropped-rows gate FAILED -- beta with w~1e-12 rows kept vs removed differs by %.3e",
      spec$name, d))
    cat(sprintf("%-18s  dropped-rows gate ok (beta rel diff %.1e)\n", spec$name, d))
  }

  est <- list(beta = I(unname(coef(m))))
  se <- unname(sqrt(diag(vcov(m))))
  if (spec$family == "gaussian") est$se <- I(se) else est$se_rx <- I(se)
  est$loglik <- as.numeric(logLik(m))
  est$varcomp <- list()
  if (spec$family == "gaussian") est$sigma <- sigma(m)

  engine <- if (spec$family == "gaussian") "stats::lm"
            else if (spec$family == "negbin") "MASS::glm.nb" else "stats::glm"
  res <- list(
    dataset = spec$name, engine = engine,
    engine_version = if (spec$family == "negbin") as.character(packageVersion("MASS"))
                     else paste0("R-", getRversion()),
    family = spec$family,
    reml = if (is.null(spec$reml)) NA else isTRUE(spec$reml),
    rung = spec$rung,
    converged = if (is.null(m$converged)) TRUE else isTRUE(m$converged),
    singular = FALSE,
    optimizer = if (spec$family == "gaussian") "lm"
                else if (spec$family == "negbin") "glm.nb" else "glm.fit",
    n_eval = if (is.null(m$iter)) 1L else as.integer(m$iter),
    coef_names = I(names(coef(m))),
    estimates = est,
    timing = time_one(spec, make_fit)
  )
  write_result(spec, res)
}

fit_one <- function(spec) {
  df <- read_dataset(spec)
  make_fit <- fit_call(spec, df)
  m <- make_fit()
  if (!is_mixed(spec)) return(fit_one_fixed(spec, df, make_fit, m))

  conv_msgs <- m@optinfo$conv$lme4$messages
  est <- list(beta = I(unname(fixef(m))))
  if (spec$family == "gaussian") {
    # LMM SE is profiled/exact -- one method, no theta-beta coupling question.
    est$se <- I(unname(sqrt(diag(as.matrix(vcov(m))))))
  } else {
    # GLMM vcov has two methods that genuinely differ (Laplace approximation).
    # se_hessian: FD-Hessian of the joint Laplace deviance over (theta, beta) --
    #   keeps the theta-beta coupling; this is glmer's default (use.hessian=TRUE).
    # se_rx: Schur complement conditional on theta-hat -- drops the coupling.
    # Both are recorded so each is compared with the matching glmm method.
    # use.hessian=FALSE warns that the two differ by >1e-4 -- that divergence is
    # exactly what is recorded, so the advisory is silenced.
    est$se_hessian <- I(unname(sqrt(diag(as.matrix(vcov(m, use.hessian = TRUE))))))
    est$se_rx <- suppressWarnings(
      I(unname(sqrt(diag(as.matrix(vcov(m, use.hessian = FALSE)))))))
  }
  est$loglik <- as.numeric(logLik(m))
  # stddev_se (GLMM only, and only when every grouping is scalar q=1 -- the only
  # shape the theta == stddev identity holds for). Dispersion families (gamma)
  # are EXCLUDED: glmer scales VarCorr stddev by sigma-hat, so theta != stddev and
  # the theta-scale SE would be the wrong quantity.
  sd_se <- NULL
  if (spec$family %in% c("binomial", "poisson") &&
      all(lengths(lapply(VarCorr(m), attr, "stddev")) == 1)) {
    sd_se <- theta_hessian_stddev_se(m)
  }
  est$varcomp <- varcomp_of(m, sd_se)
  if (spec$family == "gaussian") est$sigma <- sigma(m)

  res <- list(
    dataset = spec$name, engine = "lme4",
    engine_version = as.character(packageVersion("lme4")),
    family = spec$family,
    reml = if (is.null(spec$reml)) NA else isTRUE(spec$reml),
    rung = spec$rung,
    converged = is.null(conv_msgs) || length(conv_msgs) == 0,
    singular = isSingular(m),
    optimizer = paste(unlist(m@optinfo$optimizer), collapse = "+"),
    n_eval = as.integer(m@optinfo$feval),
    tolPwrss = if (spec$family == "gaussian") NULL else TOLPWRSS,
    coef_names = I(names(fixef(m))),
    estimates = est,
    timing = time_one(spec, make_fit)
  )

  write_result(spec, res)
}

# The weights tier only. VALIDATION_ONLY=<name>[,<name>...] fits only the named
# rows, so one reference can be (re)generated without rewriting the others; a
# name that is not a weights-tier row stops the run.
specs <- Filter(function(s) identical(s$tier, "weights"), manifest$datasets)
only <- Sys.getenv("VALIDATION_ONLY")
if (nzchar(only)) {
  keep <- strsplit(only, ",")[[1]]
  unknown <- setdiff(keep, vapply(specs, `[[`, "", "name"))
  if (length(unknown) > 0)
    stop("VALIDATION_ONLY names no weights-tier row: ", paste(unknown, collapse = ", "))
  specs <- Filter(function(s) s$name %in% keep, specs)
}
for (spec in specs) fit_one(spec)
