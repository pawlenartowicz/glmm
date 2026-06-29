#!/usr/bin/env Rscript
# Cross-engine agreement check. Takes lme4 as the reference and compares every other
# present engine's result JSON against it, per dataset, on beta / SE / varcomp stddev
# / loglik. Surfaces the "two reference engines must agree" cross-check (lme4 vs
# MixedModels.jl) NOW; when results/glmm/ lands, glmm joins the same comparison.
#
# Tolerances are PER-QUANTITY (gap doc 5, 1.1) -- one global threshold can't serve
# point estimates that agree to ~1e-4 and SEs that legitimately differ by percent.
# Starting points, tuned against the first reference run; recorded here, never
# relaxed to make an engine pass (oracle is sacred).
#
# GLMM SE is EXEMPT from the lme4-vs-MixedModels gate: lme4 keeps the theta-beta
# coupling, MixedModels.jl drops it (~3% smaller) -- a documented method difference
# (gap 1.1). The cross-check RECORDS that gap without failing on it. glmm's default
# keeps the coupling, so glmm's GLMM SE IS gated -- against lme4 only.

suppressMessages(library(jsonlite))

parity_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

TOL <- list(
  beta_rel   = 1e-3,   # fixed effects: relative
  stddev_rel = 1e-3,   # varcomp std-devs: relative
  loglik_abs = 1e-4,   # shared logLik scale: absolute
  se_rel     = 1e-3    # LMM SE: tight (all engines compute it identically)
)

read_engine <- function(engine) {
  dir <- file.path(parity_dir, "results", engine)
  if (!dir.exists(dir)) return(list())
  files <- list.files(dir, pattern = "\\.json$", full.names = TRUE)
  res <- lapply(files, fromJSON, simplifyVector = TRUE,
                simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
  setNames(res, vapply(res, `[[`, "", "dataset"))
}

# Max relative difference over two aligned numeric vectors; NA on length mismatch
# so it shows up as a hard failure rather than a silently-recycled false pass.
rel_max <- function(x, y) {
  if (length(x) != length(y)) return(NA_real_)
  max(abs(x - y) / pmax(abs(x), abs(y), 1e-12))
}
stddevs <- function(r) unlist(lapply(r$estimates$varcomp, `[[`, "stddev"))

mark <- function(diff, tol) {
  if (is.na(diff)) return("FAIL(len)")
  if (diff <= tol) "ok" else "FAIL"
}

lme4 <- read_engine("lme4")
if (length(lme4) == 0) stop("no lme4 results -- run oracle/fit.R first")
others <- Filter(function(e) length(read_engine(e)) > 0, c("mixedmodels", "glmm"))
if (length(others) == 0) {
  cat("only lme4 results present -- nothing to compare yet ",
      "(run oracle/fit.jl for the second engine)\n")
  quit(status = 0)
}

any_fail <- FALSE
for (engine in others) {
  eng <- read_engine(engine)
  cat(sprintf("\n=== lme4  vs  %s ===\n", engine))
  cat(sprintf("%-12s %-5s  %-10s %-10s %-10s %-10s  %s\n",
              "dataset", "rung", "beta", "se", "stddev", "loglik", "coef"))
  for (name in names(lme4)) {
    a <- lme4[[name]]; b <- eng[[name]]
    if (is.null(b)) next
    gaussian <- a$family == "gaussian"
    # GLMM SE: gated for glmm (keeps coupling, vs lme4); recorded-only for
    # MixedModels.jl (drops coupling -- known method gap).
    se_gated <- gaussian || engine == "glmm"

    d_beta <- rel_max(a$estimates$beta, b$estimates$beta)
    d_se   <- rel_max(a$estimates$se,   b$estimates$se)
    d_sd   <- rel_max(stddevs(a), stddevs(b))
    d_ll   <- abs(a$estimates$loglik - b$estimates$loglik)
    coef_ok <- identical(a$coef_names, b$coef_names)

    m_beta <- mark(d_beta, TOL$beta_rel)
    m_se   <- if (se_gated) mark(d_se, TOL$se_rel) else "rec"
    m_sd   <- mark(d_sd, TOL$stddev_rel)
    m_ll   <- mark(d_ll, TOL$loglik_abs)

    failed <- any(c(m_beta, m_se, m_sd, m_ll) %in% c("FAIL", "FAIL(len)")) || !coef_ok
    any_fail <- any_fail || failed
    cat(sprintf("%-12s %-5d  %.1e/%-4s %.1e/%-4s %.1e/%-4s %.1e/%-4s  %s\n",
                name, a$rung,
                d_beta, m_beta, d_se, m_se, d_sd, m_sd, d_ll, m_ll,
                if (coef_ok) "ok" else "MISMATCH"))
  }
}

cat(sprintf("\n%s\n", if (any_fail) "RESULT: disagreements found -- investigate (flag, do not relax tolerance)"
                       else "RESULT: all gated quantities agree within tolerance"))
quit(status = if (any_fail) 1 else 0)
