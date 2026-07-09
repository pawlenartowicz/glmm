#!/usr/bin/env Rscript
# lme4 reference fits over the parity datasets -> results/lme4/<dataset>.json.
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

parity_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

manifest <- fromJSON(file.path(parity_dir, "manifest.json"), simplifyDataFrame = FALSE)
out_dir <- file.path(parity_dir, "results", "lme4")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# Timing loop; first pass discarded, median of the rest reported. 10 runs (was
# 100): multi-second GLMM rungs made 100 repeats cost ~an hour per engine; each
# result JSON records its own n_runs, so old 100-run files stay self-describing.
N_RUNS <- 10

read_dataset <- function(spec) {
  df <- read.csv(file.path(parity_dir, "data", paste0(spec$name, ".csv")),
                 stringsAsFactors = FALSE)
  # Re-establish factor typing for grouping + categorical fixed-effect columns
  # (CSV round-trip loses it; numeric-looking levels like cbpp `period` come back
  # as integers). Coercion order is the sorted-level default => treatment-contrast
  # base = first sorted level, matched against Julia's DummyCoding in compare.R.
  for (f in unlist(spec$factors)) df[[f]] <- factor(df[[f]])
  df
}

fit_call <- function(spec, df) {
  fm <- as.formula(spec$r_formula)
  if (spec$family == "gaussian") {
    function() lmer(fm, data = df, REML = isTRUE(spec$reml))
  } else {
    fam <- switch(spec$family, binomial = binomial(), poisson = poisson(),
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
    function() glmer(fm, data = df, family = fam, nAGQ = 1,
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

fit_one <- function(spec) {
  df <- read_dataset(spec)
  make_fit <- fit_call(spec, df)
  m <- make_fit()

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
  sd_se <- NULL
  if (spec$family != "gaussian" && all(lengths(lapply(VarCorr(m), attr, "stddev")) == 1)) {
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
    tolPwrss = if (spec$family == "gaussian") NULL else TOLPWRSS,
    coef_names = I(names(fixef(m))),  # contrast-coding assertion vs Julia
    estimates = est,
    timing = time_one(spec, make_fit)
  )

  out <- file.path(out_dir, paste0(spec$name, ".json"))
  # digits = NA: full double precision -- this is an oracle, not a display.
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  t_disp <- if (!is.null(res$timing$fit_seconds_median)) res$timing$fit_seconds_median
            else res$timing$fit_seconds_median_hessian  # GLMM: show the heavier method
  cat(sprintf("lme4  %-12s  rung %d  converged=%s singular=%s  fit_median=%.4gs\n",
              spec$name, spec$rung, res$converged, res$singular, t_disp))
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
