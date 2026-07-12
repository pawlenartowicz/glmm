#!/usr/bin/env Rscript
# lme4 reference fits over the parity datasets -> results/lme4_{empirical,simulated}/<dataset>.json.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference glmm is later held to.
# On any glmm disagreement, glmm is presumed wrong. A reference is regenerated ONLY
# if the model SPEC (formula/family/link) is proven wrong, with a recorded reason.
# Never relax a tolerance or edit a result to make a downstream engine pass.

suppressMessages({
  library(lme4)
  library(jsonlite)
  library(numDeriv)
})

# PARITY_SUITE_DIR: suite-directory override (e.g. parity/weights/, set by that
# suite's run.sh) -- manifest.json, data_{empirical,simulated}/ and results/ are
# all resolved under it. Unset = this script's own parity/ dir (main harness,
# byte-identical behavior).
suite <- Sys.getenv("PARITY_SUITE_DIR")
parity_dir <- if (nzchar(suite)) normalizePath(suite) else
  normalizePath(file.path(dirname(sub(
    "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

manifest <- fromJSON(file.path(parity_dir, "manifest.json"), simplifyDataFrame = FALSE)
data_dir_of <- function(spec)
  file.path(parity_dir,
            if (identical(spec$source, "sim")) "data_simulated" else "data_empirical")
out_dir_of <- function(spec)
  file.path(parity_dir, "results",
            paste0("lme4_", if (identical(spec$source, "sim")) "simulated" else "empirical"))
dir.create(file.path(parity_dir, "results", "lme4_empirical"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(parity_dir, "results", "lme4_simulated"), showWarnings = FALSE, recursive = TRUE)

# Timing loop; first pass discarded, median of the rest reported. 10 runs (was
# 100): multi-second GLMM rungs made 100 repeats cost ~an hour per engine; each
# result JSON records its own n_runs, so old 100-run files stay self-describing.
N_RUNS <- 10

read_dataset <- function(spec) {
  # `data` field: CSV to read when it differs from the rung name -- lets a
  # re-linked rung (cbpp_probit) reuse an already-committed dataset byte-for-byte
  # instead of duplicating it. Absent = the name itself, the original behavior.
  src_name <- if (is.null(spec$data)) spec$name else spec$data
  df <- read.csv(file.path(data_dir_of(spec), paste0(src_name, ".csv")),
                 stringsAsFactors = FALSE)
  # Re-establish factor typing for grouping + categorical fixed-effect columns
  # (CSV round-trip loses it; numeric-looking levels like cbpp `period` come back
  # as integers). Coercion order is the sorted-level default => treatment-contrast
  # base = first sorted level, matched against Julia's DummyCoding in compare.R.
  for (f in unlist(spec$factors)) df[[f]] <- factor(df[[f]])
  df
}

# TRUE when the rung has a random term -- fixed-only rungs (weights suite) are
# fit via lm/glm/MASS::glm.nb below, and skip every lme4-specific step
# (glmerControl, vcov(use.hessian=), VarCorr, isSingular).
is_mixed <- function(spec) grepl("|", spec$r_formula, fixed = TRUE)
# Prior weights, when the rung declares a `weights_col` (weights suite). NULL =
# unit weights, the pre-existing behavior for every main-harness rung. The
# aggregated-binomial `weights` field is separate: there the trial counts enter
# through the cbind() response, not a weights= argument.
weights_of <- function(spec, df)
  if (is.null(spec$weights_col)) NULL else df[[spec$weights_col]]

fit_call <- function(spec, df) {
  fm <- as.formula(spec$r_formula)
  w <- weights_of(spec, df)
  if (!is_mixed(spec)) {
    return(switch(spec$family,
      gaussian = function() lm(fm, data = df, weights = w),
      binomial = function() glm(fm, family = binomial(), data = df, weights = w),
      poisson  = function() glm(fm, family = poisson(), data = df, weights = w),
      gamma    = function() glm(fm, family = Gamma(link = "log"), data = df, weights = w),
      negbin   = function() MASS::glm.nb(fm, data = df, weights = w),
      stop("unsupported fixed-only family: ", spec$family)))
  }
  if (spec$family == "gaussian") {
    function() lmer(fm, data = df, REML = isTRUE(spec$reml), weights = w)
  } else {
    # `link` field: non-canonical link override (cbpp_probit). Absent = the
    # family's canonical link, the pre-existing behavior for every other rung.
    fam <- switch(spec$family,
      binomial = binomial(link = if (is.null(spec$link)) "logit" else spec$link),
      poisson  = poisson(),
      gamma    = Gamma(link = if (is.null(spec$link)) "log" else spec$link),
      stop("unsupported family: ", spec$family))
    # nAGQ = 1 (Laplace) PINNED, not defaulted: glmm's GLMM kernel is glmer-faithful
    # nAGQ=1 (src/glmm/mod.rs), so this CURATED cross-engine sweep must be Laplace to
    # compare like-to-like with it. AGQ (nAGQ > 1, single scalar RE only) is MORE
    # accurate but a different estimator -- on cbpp it shifts beta ~5e-4 and the Hessian
    # SE ~1.2% (converged by nAGQ=10). It lives in the SEPARATE goldens track
    # (oracle/fit_m3_goldens.R: lme4 at nAGQ=1/7/11), where glmm's AGQ joins once its M3
    # kernel lands -- not here (design 6: the 6-rung oracle is not expanded).
    # tolPwrss = 1e-13 (default 1e-7), recorded in the result JSON: glmer's ldL2
    # is assembled from pp$Xwts -- working weights one PIRLS iteration behind the
    # mode -- so at the default tolPwrss its devfun sits ~5.6e-4 above the true
    # Laplace deviance (cbpp) and vcov(use.hessian=TRUE)/logLik carry ~1% spurious
    # theta/theta-beta curvature. Value picked by a per-rung sweep (2026-07-04,
    # measured against glmm's tight-tol FD, all four GLMM rungs): cbpp is converged
    # by 1e-10 (docs/GLMM/2026-07-04-glmm-hessian-curvature-diagnosis.md,
    # Resolution) but sim_sparse_poisson keeps a 1.3% se_hessian residual until
    # 1e-12, and grouseticks' vcov blips 1.5% on cHEIGHT at EXACTLY 1e-12 (fine at
    # 1e-10/1e-11/1e-13) -- 1e-13 is the value all four rungs agree at, flat vs
    # 1e-12 where 1e-12 is itself clean (1e-16 aborts in step-halving; stay above
    # lme4's numeric floor). This is a documented solver-precision setting, not a
    # spec change -- the model (formula/family/link/nAGQ) is untouched.
    function() glmer(fm, data = df, family = fam, weights = w, nAGQ = 1,
                     control = glmerControl(tolPwrss = 1e-13))
  }
}
# Mirrors fit_call's glmerControl -- change together. Recorded per result JSON so
# frozen references are self-describing (like n_runs).
TOLPWRSS <- 1e-13

# Times `batch` fits per sample so sub-resolution fits stay above the timer floor
# (Sys.time() resolution is coarse vs a ~60us Dyestuff fit -- a single-fit delta is
# all noise). fit_seconds_median is then the median time for `fits_per_sample` fits;
# divide by it for the per-fit estimate. batch=1 (the default) is the old behavior.
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

# Timing for one dataset. Gaussian rungs: a single profiled SE -> one median. GLMM
# rungs: SPLIT by SE method -- vcov(use.hessian=TRUE) runs numDeriv on the deviance
# (the main time consumer), vcov(use.hessian=FALSE) is the cheap RX block. Time the
# full fit+vcov for each (the glmer fit underlies both, so the gap is the SE-method
# cost), mirroring glmm's per-method Rx/Hessian fit timings.
time_one <- function(spec, make_fit) {
  batch <- if (is.null(spec$timing_batch)) 1L else as.integer(spec$timing_batch)
  # Fixed-only rungs: no Rx/Hessian method split (that is a glmer vcov choice),
  # a single fit timing regardless of family.
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
# directly -- the only reachable GLMM case here (verified: getME theta order ==
# VarCorr(m) grouping order == the varcomp_of order, so the vector aligns
# positionally). Returns one SE per grouping, in VarCorr order.
theta_hessian_stddev_se <- function(m) {
  th <- getME(m, "theta")
  pars <- c(th, fixef(m))
  devfun <- update(m, devFunOnly = TRUE)
  H <- numDeriv::hessian(devfun, pars)
  cov <- 2 * solve(H)   # deviance = -2logL => info = H/2 => cov = info^{-1} = 2 H^{-1}
  sqrt(diag(cov)[seq_along(th)])
}

# VarCorr -> common representation: per grouping factor, the RE term names, their
# standard deviations, and the correlation matrix between them. lme4 reports these
# directly via the `stddev`/`correlation` attributes (absolute scale). `sd_se`, when
# supplied (GLMM rungs), is a per-grouping stddev-SE vector in VarCorr order (see
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
    # auto_unbox, so positional comparison in compare.R is uniform across rungs.
    entry <- list(group = g, terms = I(names(sd)),
                  stddev = I(unname(sd)), corr = unname(corr))
    if (!is.null(sd_se)) entry$stddev_se <- I(unname(sd_se[i]))
    entry
  })
}

# Emit one result JSON + console line (shared tail of both fit_one branches).
write_result <- function(spec, res) {
  out <- file.path(out_dir_of(spec), paste0(spec$name, ".json"))
  # digits = NA: full double precision -- this is an oracle, not a display.
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  t_disp <- if (!is.null(res$timing$fit_seconds_median)) res$timing$fit_seconds_median
            else res$timing$fit_seconds_median_hessian  # GLMM: show the heavier method
  cat(sprintf("lme4  %-12s  rung %d  converged=%s singular=%s  fit_median=%.4gs\n",
              spec$name, spec$rung, res$converged, res$singular, t_disp))
}

# Fixed-only rungs (weights suite): lm / glm / MASS::glm.nb reference fits, same
# result schema minus the mixed-model-only fields (varcomp empty, no tolPwrss).
# Non-gaussian SE lands in `se_rx` -- the single GLM SE has no Rx-vs-Hessian
# method split, and compare.R reads non-gaussian SEs from that slot.
fit_one_fixed <- function(spec, df, make_fit, m) {
  # Dropped-rows gate (weights suite P2): rows with near-zero weight must be
  # effective row deletion -- the weighted fit's beta has to match the same fit
  # with those rows REMOVED. beta only, by construction: the near-zero rows
  # contribute nothing to the RSS but still count in the residual df, so sigma
  # and the SEs differ between the two fits.
  if (identical(spec$gate, "dropped_rows")) {
    keep <- weights_of(spec, df) > 1e-6
    dfk  <- df[keep, ]
    mk   <- lm(as.formula(spec$r_formula), data = dfk, weights = weights_of(spec, dfk))
    d <- max(abs(coef(m) - coef(mk)) / pmax(abs(coef(m)), 1e-12))
    if (d > 1e-8) stop(sprintf(
      "%s: dropped-rows gate FAILED -- beta with w~1e-12 rows kept vs removed differs by %.3e",
      spec$name, d))
    cat(sprintf("lme4  %-12s  dropped-rows gate ok (beta rel diff %.1e)\n", spec$name, d))
  }

  est <- list(beta = I(unname(coef(m))))
  se <- unname(sqrt(diag(vcov(m))))
  if (spec$family == "gaussian") est$se <- I(se) else est$se_rx <- I(se)
  est$loglik <- as.numeric(logLik(m))
  est$varcomp <- list()
  if (spec$family == "gaussian") est$sigma <- sigma(m)

  res <- list(
    dataset = spec$name, engine = "lme4",
    engine_version = as.character(packageVersion("lme4")),
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
    # Emit BOTH so the parity check compares like method to like: lme4_rx vs
    # MixedModels (RX-only) vs glmm_rx, and lme4_hessian vs glmm_hessian.
    # use.hessian=FALSE warns that the two differ by >1e-4 -- that divergence is
    # exactly what we are recording, so silence the advisory.
    est$se_hessian <- I(unname(sqrt(diag(as.matrix(vcov(m, use.hessian = TRUE))))))
    est$se_rx <- suppressWarnings(
      I(unname(sqrt(diag(as.matrix(vcov(m, use.hessian = FALSE)))))))
  }
  est$loglik <- as.numeric(logLik(m))
  # stddev_se (GLMM only, and only when every grouping is scalar q=1 -- the reachable
  # GLMM case and the only shape the theta==stddev identity holds for). glmm's
  # WaldSe::Hessian path exposes the matching number; gated in compare.R.
  # Dispersion families (gamma) are EXCLUDED: glmer scales VarCorr stddev by
  # sigma-hat (0.757 on sim_gamma), so theta != stddev and the theta-scale SE below
  # is the wrong quantity -- emitting it would gate unlike against unlike (glmm's
  # stddev_se is on the stddev scale). compare.R shows sd_se n/a when one side
  # omits it; a delta-method Jacobian would be needed to make this like-for-like.
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
    coef_names = I(names(fixef(m))),  # contrast-coding assertion vs Julia
    estimates = est,
    timing = time_one(spec, make_fit)
  )

  write_result(spec, res)
}

# PARITY_ONLY=<name>[,<name>...]: fit only the named datasets. This is how a NEW
# rung gets its reference generated without rewriting the frozen results of the
# existing ones (the oracle is sacred — corpus growth must not touch old files).
only <- Sys.getenv("PARITY_ONLY")
specs <- manifest$datasets
if (nzchar(only)) {
  keep <- strsplit(only, ",")[[1]]
  specs <- Filter(function(s) s$name %in% keep, specs)
}
for (spec in specs) fit_one(spec)
