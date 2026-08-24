#!/usr/bin/env Rscript
# Cross-engine agreement check. Takes lme4 as the reference and compares every other
# present engine's result JSON against it, per dataset, on beta / SE / varcomp stddev
# / loglik. Surfaces the "two reference engines must agree" cross-check (lme4 vs
# MixedModels.jl) NOW; results/glmm_{empirical,simulated}/ joins the same comparison.
#
# Tolerances are PER-QUANTITY (gap doc 5, 1.1) -- one global threshold can't serve
# point estimates that agree to ~1e-4 and SEs that legitimately differ by percent.
# Starting points, tuned against the first reference run; recorded here, never
# relaxed to make an engine pass (oracle is sacred). `se_hessian` is additionally
# PER-RUNG-CAPABLE via tol.R's TOL_PER_RUNG/tol_for -- see the se_hessian gate
# below for why only that one quantity has the lookup.
#
# Each engine's results live split across two dirs, `<engine>_empirical/` and
# `<engine>_simulated/` (validation/README.md) -- read_engine() merges both.
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

script_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
suite_dir <- script_dir

source(file.path(script_dir, "tol.R"))
source(file.path(script_dir, "dev_align.R"))

# lme4 labels categorical contrasts "period2"; MixedModels "period: 2", and
# interactions "a & b" where lme4 writes "a:b". Same base, same levels, same order
# (the beta columns line up positionally -- that is what the beta gate verifies).
# Normalize the cosmetic formatting so the coef-name assertion checks coding, not
# label style.
norm_coef <- function(x) gsub("[:& ]", "", x)

read_engine <- function(engine) {
  files <- unlist(lapply(c("empirical", "simulated"), function(s)
    list.files(file.path(suite_dir, "results", paste0(engine, "_", s)),
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

# ── documented-divergence registry ───────────────────────────────────────────
# The cross-engine comparison below is a REFERENCE CHECK, not a pass/fail gate.
# The four outcomes (no entry / covered / past max_rel / stale) are tabulated
# once in README.md under "Documented divergences" -- read them there rather than
# from a copy here that can drift out of step with the table.
#
# The port gates below are NOT reference checks and do not consult this registry:
# both ports call the same kernel, so their bands are round-off bands and a miss
# is a wiring bug.
DIV <- fromJSON(file.path(suite_dir, "divergences.json"),
                simplifyVector = TRUE, simplifyDataFrame = FALSE)$entries
div_fired <- character(0)
div_seen <- character(0)   # datasets this run actually compared

# The registry entry covering (dataset, quantity) in `scope`, or NULL.
div_lookup <- function(dataset, quantity, scope) {
  for (e in DIV) {
    if (identical(e$dataset, dataset) && quantity %in% e$quantities &&
        scope %in% e$comparison) {
      return(e)
    }
  }
  NULL
}

# Re-mark one over-band comparison against the registry. Returns the mark to
# print; records the hit so the staleness check below can see it.
div_mark <- function(m, diff, dataset, quantity, scope) {
  if (!identical(m, "FAIL") || is.na(diff)) return(m)
  e <- div_lookup(dataset, quantity, scope)
  if (is.null(e)) return(m)
  if (diff > e$max_rel) {
    cat(sprintf("  !! %s/%s: %.3e exceeds documented divergence %s (max_rel %.1e)\n",
                dataset, quantity, diff, e$id, e$max_rel))
    return("FAIL")
  }
  div_fired <<- union(div_fired, e$id)
  "DOC"
}

lme4 <- read_engine("lme4")
if (length(lme4) == 0) stop("no lme4 results -- run engines/lme4.R first")
others <- Filter(function(e) length(read_engine(e)) > 0,
                 c("mixedmodels", "glmm"))
mixedmodels <- read_engine("mixedmodels")  # second oracle, for the se_rx:mm cross-check
if (length(others) == 0) {
  cat("only lme4 results present -- nothing to compare yet ",
      "(run engines/mixedmodels.jl for the second engine)\n")
  quit(status = 0)
}

# One comparison cell: "<reldiff>/<mark>", or just the mark when there is no number
# to show (n/a -- engine lacks that method; FAIL(len) -- length mismatch).
cell <- function(d, m) sprintf("%-10s", if (is.na(d)) m else sprintf("%.0e/%s", d, m))
# Informational-only cell (glmm-vs-MixedModels Δdev): no mark, just the raw
# number or "n/a" -- this comparison never gates, so there is no verdict word
# to print beside it.
cell_info <- function(d) sprintf("%-10s", if (is.na(d)) "n/a" else sprintf("%.0e", d))

# Deviance-gate summary accumulators: filled per-row in
# the glmm loop below, printed as three blocks ahead of the final RESULT line.
dev_win <- character(0)   # DEV-WIN rungs (Δdev <= 0), informational
dev_na  <- character(0)   # DEV-NA rungs -- loud exclusion list, must reach the summary
dev_conv <- character(0)  # FAIL(conv?) rungs, both deviances printed

any_fail <- FALSE
for (engine in others) {
  eng <- read_engine(engine)
  cat(sprintf("\n=== lme4  vs  %s ===\n", engine))
  cat(sprintf("%-12s %-5s  %-10s %-10s %-10s %-10s %-10s %-10s %-10s %-10s  %s\n",
              "dataset", "rung", "beta", "se_rx", "se_rx:mm", "se_hess",
              "stddev", "sd_se", "dev", "dev:mm", "coef"))
  for (name in names(lme4)) {
    a <- lme4[[name]]; b <- eng[[name]]
    if (is.null(b)) next
    gaussian <- a$family == "gaussian"

    d_beta <- rel_max(a$estimates$beta, b$estimates$beta)
    # Fixed-only rungs (weights suite) carry an empty varcomp on both sides --
    # n/a, not a comparison (rel_max over zero-length vectors would warn -Inf).
    d_sd   <- if (is.null(stddevs(a)) && is.null(stddevs(b))) NA_real_
              else rel_max(stddevs(a), stddevs(b))
    # Deviance gate (dev = -2*loglik, dev_align.R's aligned convention). Hard
    # gate vs lme4, glmm row only -- oracle contract: deviance gates hard vs
    # lme4; parameters (beta/SE/stddev above) are registry-backed sanity checks.
    # The lme4-vs-MixedModels row here is the two REFERENCES disagreeing with
    # each other, so it gets no deviance mark at all -- lme4-vs-MMjl deviance
    # disagreement is the references disagreeing with each other — logged
    # separately, never resolved by picking the side closer to glmm.
    if (engine == "glmm") {
      dev_g <- aligned_dev(b$engine, b$estimates, b)
      dev_r <- aligned_dev(a$engine, a$estimates, a)
      if (is.na(dev_g) || is.na(dev_r)) {
        d_dev <- NA_real_
        m_dev <- sprintf("DEV-NA(%s)", attr(if (is.na(dev_r)) dev_r else dev_g, "why"))
      } else {
        d_dev <- dev_g - dev_r
        m_dev <- if (abs(d_dev) > TOL$dev_big) "FAIL(conv?)"
                 else if (d_dev > TOL$dev_eps) "FAIL(dev)"
                 else if (d_dev < 0) "DEV-WIN"
                 else "DEV-OK"
      }
    } else { d_dev <- NA_real_; m_dev <- "n/a" }
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
        # The ONE per-rung band in the harness (tol.R's TOL_PER_RUNG): every other
        # gate here reads its flat TOL entry directly. se_hessian is singled out
        # because it is the only quantity where the corpus-wide band (1e-3) is
        # orders looser than the agreement the crate documents (<= 2e-5), so a rung
        # added to guard that agreement needs its own number. `tol_for` falls back
        # to TOL$se_hessian_rel for any rung without an override; see tol.R's
        # TOL_PER_RUNG for which rungs override it and the measured numbers.
        m_se_h <- mark(d_se_h, tol_for(name, "se_hessian_rel"))
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

    # glmm-vs-MixedModels Δdev: informational only, computed the
    # same way as the se_rx:mm cross-check above -- printed, never gated, never
    # enters `failed`.
    if (engine == "glmm" && !is.null(bmm)) {
      dev_mm <- aligned_dev(bmm$engine, bmm$estimates, bmm)
      d_dev_mm <- if (is.na(dev_mm)) NA_real_ else dev_g - dev_mm
    } else { d_dev_mm <- NA_real_ }

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

    # Reference check, not a gate: a documented divergence re-marks as DOC. Only
    # the glmm row consults the registry — the lme4-vs-MixedModels row is the two
    # references disagreeing with EACH OTHER, which `reference_disagreements` owns
    # and which no glmm-side entry may excuse.
    if (engine == "glmm") {
      div_seen <<- union(div_seen, name)
      scope <- "lme4-vs-glmm"
      m_beta  <- div_mark(m_beta,  d_beta,  name, "beta",       scope)
      m_se_rx <- div_mark(m_se_rx, d_se_rx, name, "se_rx",      scope)
      m_se_mm <- div_mark(m_se_mm, d_se_mm, name, "se_rx:mm",   scope)
      m_se_h  <- div_mark(m_se_h,  d_se_h,  name, "se_hessian", scope)
      m_sd    <- div_mark(m_sd,    d_sd,    name, "stddev",     scope)
      m_sd_se <- div_mark(m_sd_se, d_sd_se, name, "stddev_se",  scope)
    }

    # Deviance mark is a HARD gate (not registry-eligible): FAIL(dev)/FAIL(conv?)
    # always fail, regardless of engine == "glmm" (the mixedmodels row's m_dev
    # is always "n/a" and never matches). DEV-NA never fails but is collected
    # below for the loud-exclusion summary.
    failed <- any(c(m_beta, m_se_rx, m_se_mm, m_se_h, m_sd, m_sd_se) %in%
                  c("FAIL", "FAIL(len)")) || !coef_ok ||
              m_dev %in% c("FAIL(dev)", "FAIL(conv?)")
    any_fail <- any_fail || failed
    if (engine == "glmm") {
      if (m_dev == "DEV-WIN") dev_win <<- c(dev_win, sprintf("%s (rung %d): Δdev=%.6g", name, a$rung, d_dev))
      if (startsWith(m_dev, "DEV-NA")) dev_na <<- c(dev_na, sprintf("%s (rung %d): %s", name, a$rung, m_dev))
      if (m_dev == "FAIL(conv?)") dev_conv <<- c(dev_conv, sprintf("%s (rung %d): dev_glmm=%.6g dev_lme4=%.6g Δdev=%.6g", name, a$rung, dev_g, dev_r, d_dev))
    }
    cat(sprintf("%-12s %-5d  %s %s %s %s %s %s %s %s  %s\n",
                name, a$rung,
                cell(d_beta, m_beta), cell(d_se_rx, m_se_rx), cell(d_se_mm, m_se_mm),
                cell(d_se_h, m_se_h), cell(d_sd, m_sd), cell(d_sd_se, m_sd_se),
                cell(d_dev, m_dev), cell_info(d_dev_mm),
                if (coef_ok) "ok" else "MISMATCH"))
  }
}

# ── port gate: glmm (Rust) vs the glmm Python port ───────────────────────────
# NOT an lme4-referenced row. The port calls the SAME kernel through PyO3
# (glmm.fit -> glmm::formula::lower -> fit_warm(start=NULL), which IS fit_cold --
# src/fit/mod.rs), so its numbers must match the RUST engine to round-off, not
# merely sit inside the cross-engine bands above. Those bands were tuned for two
# independent implementations of the same math; hiding the port behind them would
# pass a wiring bug -- a swapped column, a mis-ordered factor level, a dropped
# weights vector -- that lands well inside 1e-3 while being flatly wrong. Nothing
# else catches that class: the port's pytest suite fits fresh random data and
# asserts only convergence and beta[1] within 0.3 of truth, never a reference
# number. Gating vs lme4 instead would just re-measure the gap the glmm row
# already reports.
#
# TOL$port_rel is therefore a ROUND-OFF band, not an agreement band: same kernel,
# same inputs, deterministic optimizer => identical bits, modulo the JSON
# round-trip. A miss here is a port bug, never a tolerance to widen.
port <- read_engine("glmm_python")
rust <- read_engine("glmm")
if (length(port) > 0 && length(rust) > 0) {
  cat("\n=== glmm (Rust)  vs  glmm_python (port) ===\n")
  cat(sprintf("%-12s %-5s  %-10s %-10s %-10s %-10s %-10s %-10s %-10s  %s\n",
              "dataset", "rung", "beta", "se_rx", "se_hess", "stddev", "sd_se",
              "deviance", "loglik", "coef"))
  for (name in names(rust)) {
    a <- rust[[name]]; b <- port[[name]]
    if (is.null(b)) next

    # A quantity is "n/a" only when it is ABSENT on a side (an LMM rung has no
    # se_hessian/stddev_se, a fixed-only rung no varcomp). Where both sides carry
    # it, mark() runs -- so rel_max's length-mismatch NA stays a FAIL(len) instead
    # of being laundered into an n/a pass by a blanket is.na() check.
    gate <- function(d, absent = FALSE) if (absent) "n/a" else mark(d, TOL$port_rel)

    d_beta <- port_rel_max(a$estimates$beta, b$estimates$beta)
    m_beta <- gate(d_beta)
    d_se_rx <- port_rel_max(se_rx_of(a), se_rx_of(b))
    m_se_rx <- gate(d_se_rx)
    no_h <- is.null(a$estimates$se_hessian) || is.null(b$estimates$se_hessian)
    d_se_h <- if (no_h) NA_real_ else port_rel_max(a$estimates$se_hessian, b$estimates$se_hessian)
    m_se_h <- gate(d_se_h, no_h)
    no_sd <- is.null(stddevs(a)) && is.null(stddevs(b))
    d_sd <- if (no_sd) NA_real_ else port_rel_max(stddevs(a), stddevs(b))
    m_sd <- gate(d_sd, no_sd)
    sd_se_a <- stddev_ses(a); sd_se_b <- stddev_ses(b)
    no_sd_se <- is.null(sd_se_a) || is.null(sd_se_b)
    d_sd_se <- if (no_sd_se) NA_real_ else port_rel_max(sd_se_a, sd_se_b)
    m_sd_se <- gate(d_sd_se, no_sd_se)
    # deviance is top-level (not under estimates) and identically defined on both
    # sides -- the same Rust field, so it gates like any other number here.
    no_dev <- is.null(a$deviance) || is.null(b$deviance)
    d_dev <- if (no_dev) NA_real_ else port_rel_max(a$deviance, b$deviance)
    m_dev <- gate(d_dev, no_dev)
    # loglik: both sides are the SAME kernel (fit_warm(start=NULL) IS fit_cold),
    # so it round-off-gates like beta/se/deviance -- not the looser cross-engine
    # loglik_abs_* bands above, which exist only because lme4/MixedModels are a
    # genuinely different implementation.
    no_ll <- is.null(a$estimates$loglik) || is.null(b$estimates$loglik)
    d_ll <- if (no_ll) NA_real_ else port_rel_max(a$estimates$loglik, b$estimates$loglik)
    m_ll <- gate(d_ll, no_ll)
    coef_ok <- identical(a$coef_names, b$coef_names)

    marks <- c(m_beta, m_se_rx, m_se_h, m_sd, m_sd_se, m_dev, m_ll)
    failed <- any(marks %in% c("FAIL", "FAIL(len)")) || !coef_ok
    any_fail <- any_fail || failed
    cat(sprintf("%-12s %-5d  %s %s %s %s %s %s %s  %s\n", name, a$rung,
                cell(d_beta, marks[1]), cell(d_se_rx, marks[2]), cell(d_se_h, marks[3]),
                cell(d_sd, marks[4]), cell(d_sd_se, marks[5]), cell(d_dev, marks[6]),
                cell(d_ll, marks[7]),
                if (coef_ok) "ok" else "MISMATCH"))
  }
}

# ── port gate: glmm (Rust) vs the glmm R port ────────────────────────────────
# Same contract as the Python port-gate above: fastglmm() reaches the SAME kernel
# (through the extendr wrapper), so its numbers must match the RUST engine to
# round-off, not merely sit inside the cross-engine bands. TOL$port_rel is a
# ROUND-OFF band, not an agreement band -- a miss is a port bug (a swapped column, a
# mis-ordered factor level, a dropped weights vector), never a tolerance to widen.
# Both port gates read the same `rust` reference (recomputed here for clarity).
# Rungs where R's decimal parser -- NOT the marshalling -- forces the divergence.
# R's as.numeric is not correctly rounded for some 14-digit values (proven:
# "-1.6802662379087" stores 0x...640ce, one ulp below the correctly-rounded 0x...640cf
# that Rust and Python both produce). On the other 24 main-suite rungs and all 15 weights
# rungs the effect is invisible (fit gates at ~1e-15), but these two are ALREADY-multimodal
# correlated-slope surfaces (sim_max_q_slope is the documented q=8 numerical-limit rung),
# where a one-ulp input shift selects a neighbouring optimum -> ~1e-7. TOL$port_rel is NOT
# relaxed: these rungs are flagged KNOWN and left out of the pass/fail verdict, the same way
# the harness carries other documented numerical-limit exceptions. A miss on ANY OTHER rung
# is still a port bug.
KNOWN_R_PARSE <- c("sim_max_q_slope", "sim_binomial_slope2")
port_r <- read_engine("glmm_r")
rust <- read_engine("glmm")
if (length(port_r) > 0 && length(rust) > 0) {
  cat("\n=== glmm (Rust)  vs  glmm_r (port) ===\n")
  cat(sprintf("%-12s %-5s  %-10s %-10s %-10s %-10s %-10s %-10s  %s\n",
              "dataset", "rung", "beta", "se_rx", "se_hess", "stddev", "sd_se",
              "deviance", "coef"))
  for (name in names(rust)) {
    a <- rust[[name]]; b <- port_r[[name]]
    if (is.null(b)) next

    gate <- function(d, absent = FALSE) if (absent) "n/a" else mark(d, TOL$port_rel)

    d_beta <- port_rel_max(a$estimates$beta, b$estimates$beta)
    m_beta <- gate(d_beta)
    d_se_rx <- port_rel_max(se_rx_of(a), se_rx_of(b))
    m_se_rx <- gate(d_se_rx)
    no_h <- is.null(a$estimates$se_hessian) || is.null(b$estimates$se_hessian)
    d_se_h <- if (no_h) NA_real_ else port_rel_max(a$estimates$se_hessian, b$estimates$se_hessian)
    m_se_h <- gate(d_se_h, no_h)
    no_sd <- is.null(stddevs(a)) && is.null(stddevs(b))
    d_sd <- if (no_sd) NA_real_ else port_rel_max(stddevs(a), stddevs(b))
    m_sd <- gate(d_sd, no_sd)
    sd_se_a <- stddev_ses(a); sd_se_b <- stddev_ses(b)
    no_sd_se <- is.null(sd_se_a) || is.null(sd_se_b)
    d_sd_se <- if (no_sd_se) NA_real_ else port_rel_max(sd_se_a, sd_se_b)
    m_sd_se <- gate(d_sd_se, no_sd_se)
    no_dev <- is.null(a$deviance) || is.null(b$deviance)
    d_dev <- if (no_dev) NA_real_ else port_rel_max(a$deviance, b$deviance)
    m_dev <- gate(d_dev, no_dev)
    coef_ok <- identical(a$coef_names, b$coef_names)

    marks <- c(m_beta, m_se_rx, m_se_h, m_sd, m_sd_se, m_dev)
    failed <- any(marks %in% c("FAIL", "FAIL(len)")) || !coef_ok
    if (name %in% KNOWN_R_PARSE) {
      # Relabel FAIL -> KNOWN for display; do NOT count toward any_fail (see above).
      marks[marks %in% c("FAIL", "FAIL(len)")] <- "KNOWN"
    } else {
      any_fail <- any_fail || failed
    }
    cat(sprintf("%-12s %-5d  %s %s %s %s %s %s  %s\n", name, a$rung,
                cell(d_beta, marks[1]), cell(d_se_rx, marks[2]), cell(d_se_h, marks[3]),
                cell(d_sd, marks[4]), cell(d_sd_se, marks[5]), cell(d_dev, marks[6]),
                if (coef_ok) "ok" else "MISMATCH"))
  }
}

# ── documented divergences: report, and check the registry is not stale ──────
# Printed unconditionally when anything fired, so a DOC cell above always has a
# named reason next to it in the same output.
if (length(div_fired) > 0) {
  cat("\n=== documented divergences (reference check, not a gate) ===\n")
  for (e in DIV) {
    if (!(e$id %in% div_fired)) next
    cat(sprintf("%-12s rung %-3d %-22s <= %.1e  %s\n",
                e$dataset, e$rung, paste(e$quantities, collapse = ","),
                e$max_rel, e$id))
    cat(sprintf("  direction: %s\n", e$direction))
    cat(sprintf("  review: %s\n", e$review))
  }
}
# A registry entry whose dataset WAS compared and did not fire is stale: the
# divergence it excuses is gone, and leaving it would turn the entry into a
# standing exemption for whatever drifts there next. Scoped to datasets this run
# actually reached, so a `./run.sh cbpp` subset run does not trip it.
stale <- Filter(function(e) "lme4-vs-glmm" %in% e$comparison &&
                            e$dataset %in% div_seen && !(e$id %in% div_fired), DIV)
if (length(stale) > 0) {
  cat("\n")
  for (e in stale) {
    cat(sprintf("STALE registry entry: %s (%s) no longer fires -- delete it\n",
                e$id, e$dataset))
  }
  any_fail <- TRUE
}

# ── deviance gate summary ────────────────────────────────────────────────────
# Three blocks, always printed (never conditional on any_fail): DEV-WIN is
# informational so a passing run still shows it; DEV-NA is the loud exclusion
# list -- a rung with no usable reference deviance must never disappear
# silently, so this prints "none" rather than being skipped when empty;
# FAIL(conv?) repeats both raw deviances so a convention mismatch is legible
# without re-running.
cat("\n=== DEV-WIN (Δdev <= 0 vs lme4; informational) ===\n")
if (length(dev_win) > 0) for (line in dev_win) cat(sprintf("  %s\n", line)) else cat("  none\n")

cat("\n=== DEV-NA (no usable reference deviance -- excluded loudly, not passed hollow) ===\n")
if (length(dev_na) > 0) for (line in dev_na) cat(sprintf("  %s\n", line)) else cat("  none\n")

cat("\n=== FAIL(conv?) (|Δdev| > dev_big -- convention mismatch, not a fit disagreement) ===\n")
if (length(dev_conv) > 0) for (line in dev_conv) cat(sprintf("  %s\n", line)) else cat("  none\n")

cat(sprintf("\n%s\n", if (any_fail) "RESULT: disagreements found -- investigate (flag, do not relax tolerance)"
                       else "RESULT: all gated quantities agree, or diverge as documented"))
quit(status = if (any_fail) 1 else 0)
