#!/usr/bin/env Rscript
# Family/link/AGQ reference fits -> validation/goldens/<name>.json.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the in-crate goldens
# (hardcoded constants in src/*.rs tests) are validated against. On any glmm
# disagreement, glmm is presumed wrong -- never relax a tolerance or edit a reference.
#
# These JSONs are the in-crate goldens tier's own references and are written
# nowhere else: lme4/MASS for the scalar rungs (the in-crate GLMM SE is gated
# against lme4 alone; no Julia/Rust cross-check), GLMMadaptive for the vector-RE
# AGQ rungs (the `oracle` field below -- glmer refuses nAGQ>1 for vector REs).
# Run once to freeze: Rscript validation/tools/goldens_agq.R
#
# A BARE RUN REFITS AND OVERWRITES EVERY GOLDEN. Use VALIDATION_ONLY (bottom of this
# file) for anything short of a deliberate full re-freeze. That matters more now that
# the manifest can carry a spec whose reference does NOT exist yet: `pending_reference`
# (an OPTIONAL m3_goldens field -- no spec carries it while every reference is frozen,
# which is the steady state) marks a golden that is registered so this script
# can generate it, but is not frozen and is excluded from the in-crate Tier 2 corpus
# (tests/validation_oracle.rs::is_pending). Generating one is
#   VALIDATION_ONLY=<name> Rscript validation/tools/goldens_agq.R
# and the `pending_reference` field is then dropped from the manifest entry in the
# same change -- Tier 2 fails while the flag and the JSON coexist. This script itself
# ignores the field: it is a registry annotation, not a fit setting, and it is not
# copied into the reference JSON.

suppressMessages({
  library(lme4)
  library(MASS)         # glm.nb
  library(jsonlite)
})
# GLMMadaptive (vector-RE AGQ oracle: specs with oracle="GLMMadaptive"; glmer refuses
# nAGQ>1 for vector REs -- see validation/README.md) is attached at FIRST USE, inside
# fit_one_glmmadaptive, not here: a top-level library() call would make the
# whole script unloadable on a machine without the package -- including a
# VALIDATION_ONLY run naming only glmer/glm/lmer specs, which never touch it. Those
# are the majority (44 of 50 goldens) and are exactly what a single-golden
# regeneration is. Attaching inside the function keeps `mixed_model`/`packageVersion`
# resolving unqualified as before, so the vector-AGQ path is unchanged.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

manifest <- fromJSON(file.path(suite_dir, "manifest.json"), simplifyDataFrame = FALSE)
out_dir  <- file.path(suite_dir, "goldens")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fam_obj <- function(family, link) switch(family,
  gaussian = gaussian(link = link),
  poisson  = poisson(link = link),
  binomial = binomial(link = link),
  gamma    = Gamma(link = link),
  # "inverse_squared" is the port's spelling of R's "1/mu^2" -- the same mapping
  # .normalize_family applies on the R port side; keep the two spellings consistent.
  inversegaussian = inverse.gaussian(link = if (link == "inverse_squared") "1/mu^2" else link),
  stop("fam_obj: unsupported family ", family))

# VarCorr -> per grouping factor: term names, stddevs, correlation matrix. Mirrors
# the oracle scripts' varcomp writers so the JSON schema matches the curated GLMM references.
varcomp_of <- function(m) {
  vc <- VarCorr(m)
  lapply(names(vc), function(g) {
    block <- vc[[g]]
    sd <- attr(block, "stddev")
    corr <- attr(block, "correlation")
    list(group = g, terms = I(names(sd)),
         stddev = I(unname(sd)), corr = unname(corr))
  })
}

# m3_goldens specs carry a bare `data` name (no `source` field like the curated
# manifest.datasets rungs), so the empirical/simulated split is read off the
# `sim_` prefix convention directly (mirrors Step 2's split-by-filename).
data_dir_of_name <- function(name)
  file.path(suite_dir, "data", if (startsWith(name, "sim_")) "simulated" else "empirical")

read_dataset <- function(spec) {
  df <- read.csv(file.path(data_dir_of_name(spec$data), paste0(spec$data, ".csv")),
                 stringsAsFactors = FALSE)
  for (f in unlist(spec$factors)) df[[f]] <- factor(df[[f]])
  df
}

# GLMMadaptive reference fit (vector-RE AGQ rungs, spec oracle="GLMMadaptive"):
# mixed_model(fixed, random, nAGQ=k) -- k quadrature points PER RE DIMENSION
# (product grid, k^q nodes/cluster), the same convention as glmm's nagq. The
# manifest spells the fixed/random split out (`ma_fixed`/`ma_random`) rather
# than parsing it off r_formula; r_formula stays as the equivalent glmer form.
# Frozen quantities: beta, Hessian SEs (vcov(parm="fixed-effects") -- observed
# information of the AGQ log-likelihood, the like-for-like partner of glmm's
# WaldSe::Hessian beta block), and varcomp from m$D (stddev + correlation) in
# the shared golden schema. NO deviance/logLik: GLMMadaptive's logLik carries
# different additive constants than glmer's devfun convention, so the deviance
# scale is owned by the in-crate k-convergence invariants,
# not this oracle.
fit_one_glmmadaptive <- function(spec, df) {
  # Attached here rather than at the top of the file -- see the header note.
  suppressMessages(library(GLMMadaptive))
  nagq <- as.integer(spec$nagq)
  # Tightened controls, recorded per JSON (the lme4.R tolPwrss=1e-13 precedent):
  # mixed_model's DEFAULTS under-converge on the low-information rungs -- on
  # sim_binomial_slope1 (k=7) the default EM/qN stop leaves logLik 4e-3 below
  # the true optimum and beta ~3e-3/6e-3 off (measured at freeze, 2026-07-13);
  # glmm sits at the better optimum. update_GH_every=1 re-adapts the quadrature
  # grid every iteration -- the like-for-like convention (glmm re-adapts at
  # every deviance eval). Verified stable: a further tightening step moves
  # logLik < 1e-4 on every rung.
  ctrl <- list(iter_EM = 300, iter_qN_outer = 60,
               tol1 = 1e-8, tol2 = 1e-10, tol3 = 1e-12, update_GH_every = 1)
  m <- mixed_model(fixed = as.formula(spec$ma_fixed),
                   random = as.formula(spec$ma_random),
                   data = df, family = fam_obj(spec$family, spec$link),
                   nAGQ = nagq, control = ctrl)
  se <- sqrt(diag(vcov(m, parm = "fixed-effects")))
  est <- list(
    beta = I(unname(fixef(m))),
    se_hessian = I(unname(se)),
    varcomp = list(list(
      group  = sub("^.*\\|\\s*", "", spec$ma_random),
      terms  = I(colnames(m$D)),
      stddev = I(unname(sqrt(diag(m$D)))),
      corr   = unname(cov2cor(m$D))))
  )
  res <- list(
    name = spec$name, engine = "GLMMadaptive",
    engine_version = as.character(packageVersion("GLMMadaptive")),
    kind = spec$kind, data = spec$data,
    family = spec$family, link = spec$link,
    nagq = nagq,
    control = ctrl,   # written into the golden JSON as-is, self-describing
    r_formula = spec$r_formula,
    converged = isTRUE(m$converged), singular = FALSE,
    coef_names = I(names(fixef(m))),
    estimates = est
  )
  out <- file.path(out_dir, paste0(spec$name, ".json"))
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  cat(sprintf("m3  %-20s  %-4s  %-9s nAGQ=%-2d converged=%s (GLMMadaptive)\n",
              spec$name, spec$kind, spec$family, nagq, res$converged))
}

# glmmTMB reference fit (spec oracle="glmmTMB"): the exact Laplace objective by
# automatic differentiation -- the observed curvature in log|A| on every link and,
# on Gamma, the dispersion a free parameter of the likelihood. That is the
# objective glmm minimizes on the non-canonical links (probit, cloglog, NB/log,
# Gamma/log) and on mixed Gamma, where lme4's glmer minimizes a different one
# (Fisher weights in log|A|; phi left out of PIRLS and profiled as D/n), so these
# goldens are the like-for-like reference there and the lme4 goldens of the same
# rungs stay frozen beside them. Frozen quantities: beta, se_hessian (vcov()$cond,
# the conditional block of the inverse joint Hessian, the like-for-like partner of
# glmm's WaldSe::Hessian), loglik, varcomp, and the dispersion glmm reports on the
# family (Gamma: sigma()^2 = phi; NB: sigma() = theta). No se_rx (glmmTMB has
# none). Converged = convergence code 0 AND a positive-definite Hessian, the grid
# engine's own rule (validation/grid/engines/glmmtmb.R).
fit_one_glmmtmb <- function(spec, df) {
  suppressMessages(library(glmmTMB))
  fam <- switch(spec$family,
    binomial = stats::binomial(link = spec$link),
    poisson  = stats::poisson(link = spec$link),
    gamma    = stats::Gamma(link = spec$link),
    negbin   = glmmTMB::nbinom2(link = spec$link),
    stop("fit_one_glmmtmb: unsupported family ", spec$family))
  fm <- as.formula(spec$r_formula)
  # glmmTMB's default start is beta = 0; on an inverse link (sim_gamma_inv_glmm)
  # that is eta = 0, mu = Inf, and the fit stops at "negative log-likelihood is
  # NaN at starting parameter values". Only then is the fit retried from the
  # fixed-effects GLM's beta, so every golden that converges from the default
  # start is unaffected by this branch.
  m <- tryCatch(glmmTMB::glmmTMB(fm, data = df, family = fam), error = function(e) {
    b0 <- coef(glm(lme4::nobars(fm), data = df, family = fam))
    glmmTMB::glmmTMB(fm, data = df, family = fam, start = list(beta = unname(b0)))
  })
  cf <- glmmTMB::fixef(m)$cond
  varc <- glmmTMB::VarCorr(m)$cond
  est <- list(
    beta = I(unname(cf)),
    se_hessian = I(unname(sqrt(diag(stats::vcov(m)$cond)))),
    loglik = as.numeric(stats::logLik(m)),
    varcomp = lapply(names(varc), function(g) {
      block <- varc[[g]]
      list(group = g, terms = I(names(attr(block, "stddev"))),
           stddev = I(unname(attr(block, "stddev"))),
           corr = unname(attr(block, "correlation")))
    })
  )
  if (spec$family == "gamma") est$dispersion <- stats::sigma(m)^2
  if (spec$family == "negbin") est$theta <- stats::sigma(m)
  converged <- isTRUE(m$fit$convergence == 0) && isTRUE(m$sdr$pdHess)
  singular <- any(unlist(lapply(varc, function(b) any(attr(b, "stddev") < 1e-8))))
  res <- list(
    name = spec$name, engine = "glmmTMB::glmmTMB",
    engine_version = as.character(packageVersion("glmmTMB")),
    kind = spec$kind, data = spec$data,
    family = spec$family, link = spec$link,
    nagq = 1L,
    r_formula = spec$r_formula,
    converged = converged, singular = singular,
    coef_names = I(names(cf)),
    estimates = est
  )
  out <- file.path(out_dir, paste0(spec$name, ".json"))
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  cat(sprintf("m3  %-20s  %-4s  %-9s converged=%s singular=%s (glmmTMB)\n",
              spec$name, spec$kind, spec$family, converged, singular))
}

# Gamma GLM at the maximum-likelihood dispersion, glmm's Gamma GLM convention:
# stats::glm's beta and deviance; phi-hat = 1/a, a the root of
# log(a) - digamma(a) = D / (2 * sum(w)) (MASS::gamma.shape's equation at unit
# weights, solved here to machine precision rather than at gamma.shape's
# default eps); se = summary(m, dispersion = phi-hat); loglik the maximised
# sum(w * dgamma) at phi-hat. stats::glm's own golden of the same spec keeps
# the Pearson dispersion and D/sum(w) plug-in logLik it reports.
fit_one_glm_ml <- function(spec, df) {
  stopifnot(spec$family == "gamma", spec$kind == "glm")
  m <- glm(as.formula(spec$r_formula), data = df, family = fam_obj(spec$family, spec$link))
  w <- m$prior.weights
  mu <- fitted(m)
  cc <- deviance(m) / (2 * sum(w))
  a <- exp(stats::uniroot(function(t) t - digamma(exp(t)) - cc, c(-30, 30),
                          tol = 1e-15)$root)
  phi <- 1 / a
  est <- list(
    beta = I(unname(coef(m))),
    se = I(unname(coef(summary(m, dispersion = phi))[, 2])),
    loglik = sum(w * stats::dgamma(m$y, shape = a, rate = a / mu, log = TRUE)),
    dispersion = phi
  )
  res <- list(
    name = spec$name, engine = "stats::glm+ML-dispersion",
    engine_version = paste0("R-", getRversion()),
    kind = spec$kind, data = spec$data,
    family = spec$family, link = spec$link,
    nagq = 1L,
    r_formula = spec$r_formula,
    converged = isTRUE(m$converged), singular = FALSE,
    coef_names = I(names(coef(m))),
    estimates = est
  )
  out <- file.path(out_dir, paste0(spec$name, ".json"))
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  cat(sprintf("m3  %-20s  %-4s  %-9s converged=%s (ML dispersion)\n",
              spec$name, spec$kind, spec$family, isTRUE(m$converged)))
}

fit_one <- function(spec) {
  df <- read_dataset(spec)
  if (identical(spec$oracle, "GLMMadaptive")) return(fit_one_glmmadaptive(spec, df))
  if (identical(spec$oracle, "glmmTMB")) return(fit_one_glmmtmb(spec, df))
  if (identical(spec$oracle, "glm_ml")) return(fit_one_glm_ml(spec, df))
  fm <- as.formula(spec$r_formula)
  nagq <- if (is.null(spec$nagq)) 1L else as.integer(spec$nagq)

  est <- list()
  # `engine` records the function that actually produced the fit, not the package
  # that owns the branch: only 23 of these 32 goldens come from lme4. Six kind=glm
  # rungs are stats::glm and three are MASS::glm.nb, and a doc line claiming "matches
  # lme4" over a MASS-generated fixture is a silent convention mismatch (the review's
  # Axis 2). Metadata only -- nothing downstream reads this field.
  if (spec$kind == "glm") {
    m <- if (spec$family == "negbin") MASS::glm.nb(fm, data = df)
         else glm(fm, data = df, family = fam_obj(spec$family, spec$link))
    engine <- if (spec$family == "negbin") "MASS::glm.nb" else "stats::glm"
    coef_names <- names(coef(m))
    est$beta <- I(unname(coef(m)))
    # glm() SE already carry the dispersion scaling: Gamma SE are sqrt(phi)-scaled,
    # binomial/poisson use phi=1. This is exactly the convention the in-crate fit
    # reproduces, so se compares directly.
    est$se <- I(unname(sqrt(diag(vcov(m)))))
    est$loglik <- as.numeric(logLik(m))
    converged <- isTRUE(m$converged)
    singular  <- FALSE
  } else if (spec$kind == "lmm") {
    # Gaussian LMM via lmer (glmer has no gaussian family). Schema mirrors the
    # curated LMM goldens (se + varcomp + sigma), NOT the glmer se_hessian/se_rx
    # form -- the #3 VarCorr test reads beta + varcomp only.
    reml <- if (is.null(spec$reml)) TRUE else isTRUE(spec$reml)
    m <- lme4::lmer(fm, data = df, REML = reml)
    engine <- "lme4::lmer"
    coef_names <- names(fixef(m))
    est$beta <- I(unname(fixef(m)))
    est$se <- I(unname(sqrt(diag(as.matrix(vcov(m))))))
    est$loglik <- as.numeric(logLik(m))
    est$varcomp <- varcomp_of(m)
    est$sigma <- sigma(m)
    conv_msgs <- m@optinfo$conv$lme4$messages
    converged <- is.null(conv_msgs) || length(conv_msgs) == 0
    singular  <- isSingular(m)
  } else {
    # Optional per-spec tolPwrss (manifest `tolPwrss`): the curated oracle's
    # 1e-13 tightening (glmer's default 1e-7 leaves a
    # lagged-ldL2 SE artifact). Only specs that set it get it; the pre-existing
    # goldens stay frozen at the default they were generated with.
    ctrl <- if (!is.null(spec$tolPwrss)) glmerControl(tolPwrss = spec$tolPwrss)
            else glmerControl()
    # glmer.nb takes the same control object (it forwards ... to glmer): without
    # one, a spec's tolPwrss would be silently ignored on the negbin rungs while
    # applying everywhere else -- the kind of split that makes a golden's
    # provenance unreadable from its own file.
    m <- if (spec$family == "negbin")
           lme4::glmer.nb(fm, data = df, nAGQ = nagq, control = ctrl)
         else glmer(fm, data = df, family = fam_obj(spec$family, spec$link),
                    nAGQ = nagq, control = ctrl)
    engine <- if (spec$family == "negbin") "lme4::glmer.nb" else "lme4::glmer"
    coef_names <- names(fixef(m))
    est$beta <- I(unname(fixef(m)))
    # Two GLMM SE methods: se_hessian keeps the theta-beta
    # coupling (glmer default, use.hessian=TRUE); se_rx is the Schur complement
    # conditional on theta-hat. Emit both; glmm is gated against the matching one.
    est$se_hessian <- I(unname(sqrt(diag(as.matrix(vcov(m, use.hessian = TRUE))))))
    est$se_rx <- suppressWarnings(
      I(unname(sqrt(diag(as.matrix(vcov(m, use.hessian = FALSE)))))))
    est$loglik <- as.numeric(logLik(m))
    est$varcomp <- varcomp_of(m)
    conv_msgs <- m@optinfo$conv$lme4$messages
    converged <- is.null(conv_msgs) || length(conv_msgs) == 0
    singular  <- isSingular(m)
  }

  # Gamma dispersion phi: GLM reports the Pearson moment estimator directly
  # (summary()$dispersion). For the GLMM, emit the same Pearson form computed by
  # hand -- whether glmer couples phi into the fit is the design-3 open question the
  # in-crate test resolves; emitting Pearson + the lme4 sigma() lets the test pick.
  # Gamma and inverse-Gaussian both profile phi out of the likelihood the same
  # way, so both get the Pearson dispersion (and, for a glmm kind, sigma()) --
  # inversegaussian currently only registers glm cells, but the branch is
  # written for both kinds like Gamma's, not special-cased to glm alone.
  if (spec$family %in% c("gamma", "inversegaussian")) {
    if (spec$kind == "glm") {
      est$dispersion <- summary(m)$dispersion
    } else {
      pr <- residuals(m, type = "pearson")
      est$dispersion <- sum(pr^2) / (nobs(m) - length(fixef(m)))
      est$sigma <- sigma(m)
    }
  }
  if (spec$family == "negbin") {
    est$theta <- if (spec$kind == "glm") m$theta else getME(m, "glmer.nb.theta")
  }

  # Version of the package owning `engine`. stats ships with R, so its version is
  # R's own; the rest carry their package version.
  engine_pkg <- sub("::.*$", "", engine)
  engine_version <- if (engine_pkg == "stats") paste0("R-", getRversion())
                    else as.character(packageVersion(engine_pkg))

  res <- list(
    name = spec$name, engine = engine,
    engine_version = engine_version,
    kind = spec$kind, data = spec$data,
    family = spec$family, link = spec$link,
    nagq = nagq,
    tolPwrss = spec$tolPwrss,  # NULL (dropped) unless the spec sets it

    r_formula = spec$r_formula,
    converged = converged, singular = singular,
    coef_names = I(coef_names),
    estimates = est
  )
  out <- file.path(out_dir, paste0(spec$name, ".json"))
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  cat(sprintf("m3  %-20s  %-4s  %-9s converged=%s singular=%s\n",
              spec$name, spec$kind, spec$family, converged, singular))
}

# VALIDATION_ONLY=<name>[,<name>...]: fit only the named goldens —
# lets a NEW golden get its reference generated without rewriting the frozen
# results of the existing ones (the oracle is sacred).
only <- Sys.getenv("VALIDATION_ONLY")
specs <- manifest$m3_goldens
if (nzchar(only)) {
  keep <- strsplit(only, ",")[[1]]
  specs <- Filter(function(s) s$name %in% keep, specs)
}
for (spec in specs) {
  tryCatch(fit_one(spec),
           error = function(e) cat(sprintf("m3  %-20s  ERROR: %s\n",
                                            spec$name, conditionMessage(e))))
}
