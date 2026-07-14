#!/usr/bin/env Rscript
# M3 family/link/AGQ reference fits -> parity/goldens/<name>.json.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the in-crate M3 goldens
# (hardcoded constants in src/*.rs tests) are validated against. On any glmm
# disagreement, glmm is presumed wrong -- never relax a tolerance or edit a reference.
#
# Deliberately SEPARATE from oracle/fit.R + the results/<engine>_{empirical,simulated}/
# tree: compare.R discovers references by globbing results/lme4_{empirical,simulated}/*.json,
# so dropping new-family fits
# there would pull them into the curated cross-engine sweep and expand the 6-rung
# oracle -- which design 6 forbids ("the curated 6-rung oracle is NOT expanded").
# Writing to goldens/ keeps that sweep byte-for-byte untouched. lme4/MASS for the
# scalar rungs (the in-crate GLMM SE is gated against lme4 alone, gap 1.1; no
# Julia/Rust cross-check); GLMMadaptive for the vector-RE AGQ rungs (oracle field
# below -- glmer refuses nAGQ>1 for vector REs, full-AGQ spec locked decision 6).
# Run once to freeze (NOT part of run.sh): Rscript parity/oracle/fit_m3_goldens.R

suppressMessages({
  library(lme4)
  library(MASS)         # glm.nb
  library(GLMMadaptive) # vector-RE AGQ oracle (specs with oracle="GLMMadaptive";
                        # glmer refuses nAGQ>1 for vector REs). See parity/README.md.
  library(jsonlite)
})

parity_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

manifest <- fromJSON(file.path(parity_dir, "manifest.json"), simplifyDataFrame = FALSE)
out_dir  <- file.path(parity_dir, "goldens")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fam_obj <- function(family, link) switch(family,
  gaussian = gaussian(link = link),
  poisson  = poisson(link = link),
  binomial = binomial(link = link),
  gamma    = Gamma(link = link),
  stop("fam_obj: unsupported family ", family))

# VarCorr -> per grouping factor: term names, stddevs, correlation matrix. Mirrors
# oracle/fit.R::varcomp_of so the JSON schema matches the curated GLMM references.
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
  file.path(parity_dir, if (startsWith(name, "sim_")) "data_simulated" else "data_empirical")

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
# scale is owned by the in-crate k-convergence invariants (spec Part 4 layer 1),
# not this oracle.
fit_one_glmmadaptive <- function(spec, df) {
  nagq <- as.integer(spec$nagq)
  # Tightened controls, recorded per JSON (the fit.R tolPwrss=1e-13 precedent):
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
    control = ctrl,   # self-describing, like fit.R's tolPwrss field
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

fit_one <- function(spec) {
  df <- read_dataset(spec)
  if (identical(spec$oracle, "GLMMadaptive")) return(fit_one_glmmadaptive(spec, df))
  fm <- as.formula(spec$r_formula)
  nagq <- if (is.null(spec$nagq)) 1L else as.integer(spec$nagq)

  est <- list()
  if (spec$kind == "glm") {
    m <- if (spec$family == "negbin") MASS::glm.nb(fm, data = df)
         else glm(fm, data = df, family = fam_obj(spec$family, spec$link))
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
    # 1e-13 tightening (see oracle/fit.R -- glmer's default 1e-7 leaves a
    # lagged-ldL2 SE artifact). Only specs that set it get it; the pre-existing
    # goldens stay frozen at the default they were generated with.
    ctrl <- if (!is.null(spec$tolPwrss)) glmerControl(tolPwrss = spec$tolPwrss)
            else glmerControl()
    m <- if (spec$family == "negbin") lme4::glmer.nb(fm, data = df, nAGQ = nagq)
         else glmer(fm, data = df, family = fam_obj(spec$family, spec$link),
                    nAGQ = nagq, control = ctrl)
    coef_names <- names(fixef(m))
    est$beta <- I(unname(fixef(m)))
    # Two GLMM SE methods (see oracle/fit.R): se_hessian keeps the theta-beta
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
  if (spec$family == "gamma") {
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

  res <- list(
    name = spec$name, engine = "lme4",
    engine_version = as.character(packageVersion("lme4")),
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

# PARITY_ONLY=<name>[,<name>...]: fit only the named goldens (mirrors fit.R) —
# lets a NEW golden get its reference generated without rewriting the frozen
# results of the existing ones (the oracle is sacred).
only <- Sys.getenv("PARITY_ONLY")
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
