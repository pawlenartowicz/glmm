#!/usr/bin/env Rscript
# Cross-engine agreement check. Takes lme4 as the reference and compares every other
# present engine's result JSON against it, per dataset, on beta / SE / varcomp stddev
# / loglik. Surfaces the "two reference engines must agree" cross-check (lme4 vs
# MixedModels.jl) NOW; results/glmm_{empirical,simulated}/ joins the same comparison.
#
# Tolerances are PER-QUANTITY (gap doc 5, 1.1) -- one global threshold can't serve
# point estimates that agree to ~1e-4 and SEs that legitimately differ by percent.
# Starting points, tuned against the first reference run; recorded here, never
# relaxed to make an engine pass (oracle is sacred).
#
# Each engine's results live split across two dirs, `<engine>_empirical/` and
# `<engine>_simulated/` (parity/README.md) -- read_engine() merges both.
#
# GLMM SE is split by METHOD so the comparison is like-for-like (gap 1.1). The
# Laplace SE has two variants that genuinely differ: se_hessian keeps the theta-beta
# coupling (lme4 use.hessian=TRUE default; glmm WaldSe::Hessian), se_rx drops it
# (lme4 use.hessian=FALSE; MixedModels' only vcov; glmm WaldSe::Rx). Comparing across
# methods is what produced the spurious ~1-1.5% "gap". So here:
#   se_rx      -- all three engines compute it; gated tightly (lme4_rx vs MM ~ 4e-4).
#   se_rx:mm   -- the SAME se_rx but vs the SECOND oracle (MixedModels), not lme4. The
#                 two references disagree by ~2e-4 on their own Rx, and glmm sits on the
#                 MixedModels value (glmm == MM ~2e-8 on cbpp) while lme4 is the outlier
#                 -- so checking only vs lme4 hides glmm's exact agreement with an
#                 independent engine. n/a in the mixedmodels row (mm-vs-mm).
#   se_hessian -- only lme4 and glmm compute it; gated like se_rx (the references are
#                 artifact-free since 2026-07-04 -- see TOL); MixedModels has none, shown n/a.
# Gaussian rungs have a single profiled `se` (no method choice), compared in the se_rx slot.

suppressMessages(library(jsonlite))

# PARITY_SUITE_DIR: suite-directory override (mirrors fit.R) -- results/ resolves
# under it; unset = this script's own parity/ dir. tol.R is always the SHARED one
# next to this script (suites reuse the main tolerances by design).
script_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
suite <- Sys.getenv("PARITY_SUITE_DIR")
parity_dir <- if (nzchar(suite)) normalizePath(suite) else script_dir

source(file.path(script_dir, "tol.R"))

# lme4 labels categorical contrasts "period2"; MixedModels "period: 2", and
# interactions "a & b" where lme4 writes "a:b". Same base, same levels, same order
# (the beta columns line up positionally -- that is what the beta gate verifies).
# Normalize the cosmetic formatting so the coef-name assertion checks coding, not
# label style.
norm_coef <- function(x) gsub("[:& ]", "", x)

read_engine <- function(engine) {
  files <- unlist(lapply(c("empirical", "simulated"), function(s)
    list.files(file.path(parity_dir, "results", paste0(engine, "_", s)),
               pattern = "\\.json$", full.names = TRUE)))
  if (length(files) == 0) return(list())
  res <- lapply(files, fromJSON, simplifyVector = TRUE,
                simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
  setNames(res, vapply(res, `[[`, "", "dataset"))
}

stddevs <- function(r) unlist(lapply(r$estimates$varcomp, `[[`, "stddev"))
# Per-grouping SE of the RE stddev (GLMM Hessian only); NULL when absent (LMM rungs,
# MixedModels which has no Hessian variant) so rel_max reports it n/a rather than 0.
stddev_ses <- function(r) {
  v <- lapply(r$estimates$varcomp, `[[`, "stddev_se")
  if (all(vapply(v, is.null, logical(1)))) NULL else unlist(v)
}
# se_rx-slot vector: gaussian rungs carry a single profiled `se`, GLMM rungs `se_rx`.
se_rx_of <- function(r) if (r$family == "gaussian") r$estimates$se else r$estimates$se_rx

mark <- function(diff, tol) {
  if (is.na(diff)) return("FAIL(len)")
  if (diff <= tol) "ok" else "FAIL"
}

lme4 <- read_engine("lme4")
if (length(lme4) == 0) stop("no lme4 results -- run oracle/fit.R first")
others <- Filter(function(e) length(read_engine(e)) > 0,
                 c("mixedmodels", "glmm"))
mixedmodels <- read_engine("mixedmodels")  # second oracle, for the se_rx:mm cross-check
if (length(others) == 0) {
  cat("only lme4 results present -- nothing to compare yet ",
      "(run oracle/fit.jl for the second engine)\n")
  quit(status = 0)
}

# One comparison cell: "<reldiff>/<mark>", or just the mark when there is no number
# to show (n/a -- engine lacks that method; FAIL(len) -- length mismatch).
cell <- function(d, m) sprintf("%-10s", if (is.na(d)) m else sprintf("%.0e/%s", d, m))

any_fail <- FALSE
for (engine in others) {
  eng <- read_engine(engine)
  cat(sprintf("\n=== lme4  vs  %s ===\n", engine))
  cat(sprintf("%-12s %-5s  %-10s %-10s %-10s %-10s %-10s %-10s %-10s  %s\n",
              "dataset", "rung", "beta", "se_rx", "se_rx:mm", "se_hess",
              "stddev", "sd_se", "loglik", "coef"))
  for (name in names(lme4)) {
    a <- lme4[[name]]; b <- eng[[name]]
    if (is.null(b)) next
    gaussian <- a$family == "gaussian"

    d_beta <- rel_max(a$estimates$beta, b$estimates$beta)
    # Fixed-only rungs (weights suite) carry an empty varcomp on both sides --
    # n/a, not a comparison (rel_max over zero-length vectors would warn -Inf).
    d_sd   <- if (is.null(stddevs(a)) && is.null(stddevs(b))) NA_real_
              else rel_max(stddevs(a), stddevs(b))
    # loglik: glmm's Fit exposes none yet -- n/a (ungated) when the engine omits it,
    # rather than erroring on `value - NULL`. Gated only where the engine reports it.
    d_ll   <- if (is.null(b$estimates$loglik)) NA_real_
              else abs(a$estimates$loglik - b$estimates$loglik)
    coef_ok <- identical(norm_coef(a$coef_names), norm_coef(b$coef_names))

    # SE by method (gap 1.1). Gaussian: single profiled `se`, shown in the rx slot.
    # GLMM: se_rx is method-matched across all engines (gated tight); se_hessian
    # exists only where both sides compute it (lme4 & glmm) -- n/a when the engine
    # lacks it (MixedModels), gated at se_hessian_rel when present.
    if (gaussian) {
      d_se_rx <- rel_max(a$estimates$se, b$estimates$se)
      m_se_rx <- mark(d_se_rx, TOL$se_rel)
      d_se_h  <- NA_real_; m_se_h <- "n/a"
    } else {
      d_se_rx <- rel_max(a$estimates$se_rx, b$estimates$se_rx)
      m_se_rx <- mark(d_se_rx, TOL$se_rel)
      if (!is.null(b$estimates$se_hessian)) {
        d_se_h <- rel_max(a$estimates$se_hessian, b$estimates$se_hessian)
        m_se_h <- mark(d_se_h, TOL$se_hessian_rel)
      } else { d_se_h <- NA_real_; m_se_h <- "n/a" }
    }

    # se_rx vs the SECOND oracle (MixedModels), not lme4: records glmm's agreement
    # with the other reference. n/a for the mixedmodels row (mm-vs-mm) and where mm
    # lacks the dataset. Gated like se_rx -- a glmm that disagreed with BOTH oracles
    # is a real flag, not a tolerance to relax.
    bmm <- mixedmodels[[name]]
    if (engine != "mixedmodels" && !is.null(bmm)) {
      d_se_mm <- rel_max(se_rx_of(b), se_rx_of(bmm))
      m_se_mm <- mark(d_se_mm, TOL$se_rel)
    } else { d_se_mm <- NA_real_; m_se_mm <- "n/a" }

    # stddev_se: GLMM RE-stddev SE, only where both engines report it (lme4 & glmm
    # Hessian). n/a for gaussian rungs and MixedModels (no Hessian variant).
    sd_se_a <- stddev_ses(a); sd_se_b <- stddev_ses(b)
    if (!is.null(sd_se_a) && !is.null(sd_se_b)) {
      d_sd_se <- rel_max(sd_se_a, sd_se_b)
      m_sd_se <- mark(d_sd_se, TOL$stddev_se_rel)
    } else { d_sd_se <- NA_real_; m_sd_se <- "n/a" }

    m_beta <- mark(d_beta, TOL$beta_rel)
    m_sd   <- if (is.null(stddevs(a)) && is.null(stddevs(b))) "n/a"
              else mark(d_sd, TOL$stddev_rel)
    m_ll   <- if (is.na(d_ll)) "n/a"
              else mark(d_ll, if (gaussian) TOL$loglik_abs_lmm else TOL$loglik_abs_glmm)

    failed <- any(c(m_beta, m_se_rx, m_se_mm, m_se_h, m_sd, m_sd_se, m_ll) %in%
                  c("FAIL", "FAIL(len)")) || !coef_ok
    any_fail <- any_fail || failed
    cat(sprintf("%-12s %-5d  %s %s %s %s %s %s %s  %s\n",
                name, a$rung,
                cell(d_beta, m_beta), cell(d_se_rx, m_se_rx), cell(d_se_mm, m_se_mm),
                cell(d_se_h, m_se_h), cell(d_sd, m_sd), cell(d_sd_se, m_sd_se),
                cell(d_ll, m_ll),
                if (coef_ok) "ok" else "MISMATCH"))
  }
}

cat(sprintf("\n%s\n", if (any_fail) "RESULT: disagreements found -- investigate (flag, do not relax tolerance)"
                       else "RESULT: all gated quantities agree within tolerance"))
quit(status = if (any_fail) 1 else 0)
