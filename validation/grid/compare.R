#!/usr/bin/env Rscript
# The accuracy grid's four gates, over one glmm run and the pinned oracle runs.
#
#   1. deviance vs the BEST oracle on the cell's estimator arm (hard)
#   2. parameters vs the NEAREST oracle, backed by divergences.json
#   3. truth error per family and arm (hard)
#   4. the two port gates, glmm vs glmm_python and glmm vs glmm_r (round-off)
#
# No engine is "the" reference. Gate 1 takes the best converged oracle on the
# cell's arm, gate 2 passes a quantity that is in band against AT LEAST ONE
# oracle, and gate 3 scores every engine against the generating truth. That is
# what keeps one oracle's defect in one family from reddening the whole grid
# while still making the defect visible -- it shows up in gate 3, where it is a
# statement about that oracle rather than about glmm. On the few cells where
# every oracle's OWN deviance round-off is above dev_eps (manifest `dev_ref`),
# gate 1 uses the frozen high-precision value in dev_ref.json instead. On an AGQ
# cell with q >= 2 random effects scored against GLMMadaptive, the two engines'
# quadrature grids differ (dev_align.R, ga_rule_dev), so gate 1 compares glmm's
# point under GLMMadaptive's rule with GLMMadaptive's own deviance.
#
# Oracle-versus-oracle disagreement is never resolved by picking the oracle
# closest to glmm. It is reported and written up outside the gates.
#
# EXIT CODES. 0 every gate passed (or diverged as documented), 1 at least one
# gate FAILED, 2 a precondition error: no matching oracle run, an oracle run
# whose engine_version does not match versions.json, an unmeasured tol.R
# constant, an unreadable run directory. 2 is deliberately not 1 -- "the gate
# says no" and "the gate could not run" are different statements. Every one of
# those conditions reaches 2 by being CAUGHT, not by propagating: the whole body
# runs inside one tryCatch, so a stop() from newest_oracle_run or tol_for prints
# its message and exits 2 rather than an R traceback and a status of 1.

suppressMessages(library(jsonlite))

# The grid directory is this script's own, NOT the resolved location of the
# sibling scripts: runs/, manifest.json and divergences.json are read from here,
# so a throwaway grid tree that symlinks the scripts and holds its own runs/ is
# gated as itself.
script_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

# ── arguments ────────────────────────────────────────────────────────────────
opt_glmm_run <- NULL
opt_fast <- FALSE
opt_ports <- FALSE
opt_devfloor <- FALSE
# Parsed inside the tryCatch at the bottom, not here, so a mistyped flag is a
# precondition error (exit 2) like every other "the gate could not run" case
# rather than a bare R error and a status of 1.
parse_args <- function() {
  for (a in commandArgs(TRUE)) {
    if (startsWith(a, "--glmm-run=")) opt_glmm_run <<- sub("^--glmm-run=", "", a)
    else if (identical(a, "--fast")) opt_fast <<- TRUE
    else if (identical(a, "--ports-only")) opt_ports <<- TRUE
    else if (identical(a, "--dev-floor")) opt_devfloor <<- TRUE
    else stop(sprintf(paste0("unknown argument `%s`. Usage: compare.R [--glmm-run=DIR] ",
                             "[--fast] [--ports-only] [--dev-floor]"), a), call. = FALSE)
  }
  # Rejected rather than ignored: a flag that does nothing reads as a cell
  # restriction that was honoured.
  if (opt_fast && opt_ports) {
    stop("--fast has no effect with --ports-only: the port gate compares whatever ",
         "cells both runs hold", call. = FALSE)
  }
}

# ── printers ─────────────────────────────────────────────────────────────────
# One comparison cell: "<reldiff>/<mark>", or just the mark when there is no
# number to show (n/a -- nobody reports that quantity; FAIL(len) -- the two
# vectors could not be aligned).
cell <- function(d, m, w = 10) sprintf(sprintf("%%-%ds", w),
                                       if (is.na(d)) m else sprintf("%.0e/%s", d, m))

# ── documented-divergence registry ───────────────────────────────────────────
# Gate 2 is a reference check, not a pass/fail gate. The four outcomes (no entry
# / covered / past max_rel / stale) are written out once in the "//" string of
# divergences.json -- read them there rather than from a copy here that can drift
# out of step.
#
# The deviance gate has NO registry escape and the port gates do not consult the
# registry either: both ports call the same kernel, so their bands are round-off
# bands and a miss is a wiring bug.
DIV <- fromJSON(file.path(script_dir, "divergences.json"),
                simplifyVector = TRUE, simplifyDataFrame = FALSE)$entries
div_fired <- character(0)
# "<cell> <quantity>" pairs this run actually COMPARED -- not merely reached. A
# cell whose quantity came out `n/a` or `no-ref` was never put to the registry, so
# an entry on it cannot be called stale on the strength of this run.
div_seen <- character(0)

# The registry entry covering (cell, quantity) in `scope`, or NULL.
div_lookup <- function(cell_id, quantity, scope) {
  for (e in DIV) {
    if (identical(e$cell, cell_id) && quantity %in% e$quantities &&
        scope %in% e$comparison) {
      return(e)
    }
  }
  NULL
}

# Re-mark one over-band comparison against the registry. Returns the mark to
# print; records the hit so the staleness check below can see it.
div_mark <- function(m, diff, cell_id, quantity, scope) {
  if (!identical(m, "FAIL") || is.na(diff)) return(m)
  e <- div_lookup(cell_id, quantity, scope)
  if (is.null(e)) return(m)
  # Recorded as fired BEFORE the max_rel test. An entry whose divergence grew
  # past its ceiling still fired -- the stale check's advice is "delete it", and
  # printing that beside a FAIL saying the same divergence outgrew the entry
  # would tell the reader to throw away the one record of what it used to be.
  # Over max_rel the entry needs its number re-measured, not deleting.
  div_fired <<- union(div_fired, e$id)
  if (diff > e$max_rel) {
    cat(sprintf("  !! %s/%s: %.3e exceeds documented divergence %s (max_rel %.1e)\n",
                cell_id, quantity, diff, e$id, e$max_rel))
    return("FAIL")
  }
  "DOC"
}

# ── frozen high-precision deviance references ───────────────────────────────
# A handful of cells carry a manifest `dev_ref` (gen_manifest.R's DEV_REF):
# their data make every reference engine's OWN f64 deviance round off by more
# than dev_eps, so no oracle can gate them. Gate 1 falls back to the
# high-precision value pinned here instead of best_oracle_dev's oracle
# comparison; dev_verdict's dev_eps/dev_big and its sign rule are unchanged.
# Unlike divergences.json this is not a registry with a staleness check --
# there is no band to grow past, only a fixed reference value with its own
# provenance in dev_ref.json.
DEV_REF <- fromJSON(file.path(script_dir, "dev_ref.json"),
                    simplifyVector = TRUE, simplifyDataFrame = FALSE)$entries

# best_oracle_dev()-shaped result for a `dev_ref` cell, so the caller needs no
# separate code path for printing or for dev_verdict.
dev_ref_dev <- function(cell_id) {
  for (e in DEV_REF) if (identical(e$cell, cell_id)) {
    return(list(best = as.numeric(e$value), who = "frozen-ref",
               excluded = "dev_ref.json", gap = NA_real_, why = NA_character_))
  }
  stop(sprintf("cell %s carries manifest `dev_ref` but has no entry in dev_ref.json",
              cell_id), call. = FALSE)
}

# Gate 1's glmm deviance against the reference `b` picked. On a ga_rule_cell
# (dev_align.R) where that reference is GLMMadaptive, it is -2 x GLMMadaptive's
# AGQ log-likelihood at glmm's reported point, carrying a `rule` attribute so every
# line that prints the comparison names the rule; everywhere else it is
# aligned_dev. A glmm record that is not a converged fit keeps aligned_dev: it has
# no point to evaluate, and gate 2 already fails it by name.
gate1_glmm_dev <- function(g, cl, b) {
  if (!isTRUE(g$converged) || !identical(b$who, "GLMMadaptive") || !ga_rule_cell(cl))
    return(aligned_dev(g, cl))
  d <- ga_rule_dev(g, cl)
  attr(d, "rule") <- "GLMMadaptive's AGQ rule"
  d
}

# The quantities gate 2 compares, in print order. `has_quantity` (runs.R) decides
# which of them a record actually carries.
GATE2_QUANTITIES <- c("beta", "se_rx", "se_hessian", "stddev", "corr")

# Whether oracle record `r` is left out of gate 2's reference set for quantity
# `q` on this cell. The records gate 1 drops for a different objective
# (dev_align.R's objective_differs -- lme4 on Gamma and on mixed
# probit/cloglog/NB cells, MixedModels.jl on mixed probit/cloglog and on
# prior-weight fixed-effects gaussian cells) are the candidates; the lists live
# only in dev_align.R.
#
# On a mixed cell a different objective puts every parameter at a different
# optimum, so the record is out for every quantity. On a fixed-effects cell the
# objectives differ only in how the dispersion enters: stats::glm reports its
# Gamma logLik at a plug-in dispersion, and GLM.jl reports the normal
# log-likelihood on a sum(w) scale on a prior-weight cell. None of that moves beta
# -- every engine solves the same weighted score equations for it (measured
# 2026-09-26 on gaml_glm_g3000, glm_gamma, wls_basic and the path_* cells: at
# most 1.3e-05 apart). The standard errors do move with the dispersion
# estimate, and only lme4's record, which is stats::lm / stats::glm there, uses
# glmm's dispersion: lm's residual variance, and summary(glm)'s Pearson moment
# on Gamma. MixedModels.jl reports SEs 5e-02 to 0.99 away on the prior-weight
# fixed-effects gaussian cells, so it stays out.
gate2_excluded <- function(r, cl, q) {
  if (!objective_differs(r, cl)) return(FALSE)
  if (cl[["n_theta"]] > 0) return(TRUE)
  if (identical(q, "beta")) return(FALSE)
  !identical(r$engine, "lme4")
}

# The numbers a record holds for one gate-2 quantity.
quantity_values <- function(r, q) switch(q,
  beta = r$beta, se_rx = r$se_rx, se_hessian = r$se_hessian,
  stddev = stddevs_of(r), corr = corrs_of(r),
  stop("no such quantity ", q))

# A record that carries the quantity but not one finite value of it -- glmmTMB
# writes its Hessian SEs as nulls when it cannot invert the Hessian -- reports
# nothing to compare against.
reports_finite <- function(r, q) {
  has_quantity(r, q) && any(is.finite(as.numeric(unlist(quantity_values(r, q)))))
}

# se_rx has no oracle in glmm's convention on these cells: glmm's Rx is the
# Schur complement of the OBSERVED information, and the only oracles that
# report se_rx at all (lme4, MixedModels.jl) use the EXPECTED information at
# their own optimum -- the same non-canonical split
# `fisher_laplace` (dev_align.R) already names for the deviance gate, plus
# mixed Gamma, which that helper does not cover because lme4_objective excludes
# Gamma by family alone rather than through fisher_laplace. glmm's own Rx stays
# covered by the in-crate identity test
# (rx_se_is_the_penalized_deviance_curvature_in_beta), not by this gate.
se_rx_no_convention_oracle <- function(cl) {
  fisher_laplace(cl) || (cl[["n_theta"]] > 0 && identical(cl[["family"]], "gamma"))
}

any_fail <- FALSE

# ── gate 4: the port gates ───────────────────────────────────────────────────
# NOT an oracle-referenced comparison. Each port calls the SAME kernel -- the
# Python one through PyO3, the R one through extendr -- with the same lowering
# and a deterministic optimizer, so its numbers must match the Rust engine to
# round-off, not merely sit inside the cross-engine bands the oracle gates use.
# Those bands were tuned for two independent implementations of the same math;
# hiding a port behind them would pass a wiring bug -- a swapped column, a
# mis-ordered factor level, a dropped weights vector -- that lands well inside
# 1e-3 while being flatly wrong. Nothing else catches that class: the port test
# suites fit fresh random data and assert convergence and a coefficient near
# truth, never a reference number.
#
# TOL$port_rel is therefore a ROUND-OFF band, not an agreement band: same kernel,
# same inputs, deterministic optimizer => identical bits, modulo the JSON
# round-trip. A miss here is a port bug, never a tolerance to widen.
#
# `known` is the exception class for cells where R's decimal PARSER, not the
# marshalling, forces the divergence: R's as.numeric is not correctly rounded for
# some 14-digit values (proven: "-1.6802662379087" stores one ulp below the
# correctly-rounded double that Rust and Python both produce). On an
# already-multimodal correlated-slope surface a one-ulp input shift selects a
# neighbouring optimum. TOL$port_rel is NOT relaxed for those cells: they are
# flagged KNOWN and left out of the verdict, and a miss on ANY OTHER cell is
# still a port bug. The list starts EMPTY and its members are identified by
# measurement on this grid, never assumed.
KNOWN_R_PARSE <- character(0)

# A port gate with no port run FAILS. It does not print a note and move on: a
# gate that skips itself is not a gate that passed, and this is the check a CI
# job calls, where "no run directory" and "the ports agree" must never produce
# the same green.
port_gate <- function(grid_dir, rust, engine, known) {
  pr <- newest_glmm_run(grid_dir, engine, NULL)
  if (is.null(pr)) {
    cat(sprintf(paste0("FAIL: no %s run under runs/%s/ (scratch included), so its port ",
                       "gate did not run. Run that port over the same cells first.\n"),
                engine, engine))
    any_fail <<- TRUE
    return(invisible(NULL))
  }
  cat(sprintf("\n=== glmm (Rust)  vs  %s (port) ===\n", engine))
  cat(sprintf("%-34s  %-10s %-10s %-10s %-10s %-10s %-10s  %s\n",
              "cell", "beta", "se_rx", "se_hess", "stddev", "deviance", "loglik", "coef"))
  compared <- 0L
  for (name in names(rust$recs)) {
    a <- rust$recs[[name]]; b <- pr$recs[[name]]
    if (is.null(b)) next
    compared <- compared + 1L
    band <- tol_for(name, "port_rel")
    gate <- function(d, absent = FALSE) if (absent) "n/a" else mark(d, band)

    d_beta <- port_rel_max(a$beta, b$beta)
    m_beta <- gate(d_beta)
    # A refused fit (both sides agree there is no estimate) reports its SE
    # slot as a same-length vector of NA, not an absent field -- caught here
    # so it reads as "nothing to compare" like the absent-field case above it,
    # not as a length mismatch (`rel_max`'s own both-NA collapse already
    # drops it to NA_real_, which `mark()` would otherwise print FAIL(len)).
    no_rx <- is.null(a$se_rx) || is.null(b$se_rx) ||
      (all(is.na(a$se_rx)) && all(is.na(b$se_rx)))
    d_se_rx <- if (no_rx) NA_real_ else port_rel_max(a$se_rx, b$se_rx)
    m_se_rx <- gate(d_se_rx, no_rx)
    no_h <- is.null(a$se_hessian) || is.null(b$se_hessian) ||
      (all(is.na(a$se_hessian)) && all(is.na(b$se_hessian)))
    d_se_h <- if (no_h) NA_real_ else port_rel_max(a$se_hessian, b$se_hessian)
    m_se_h <- gate(d_se_h, no_h)
    no_sd <- length(stddevs_of(a)) == 0 && length(stddevs_of(b)) == 0
    d_sd <- if (no_sd) NA_real_ else port_rel_max(stddevs_of(a), stddevs_of(b))
    m_sd <- gate(d_sd, no_sd)
    no_dev <- is.null(a$deviance) || is.null(b$deviance)
    d_dev <- if (no_dev) NA_real_ else port_rel_max(a$deviance, b$deviance)
    m_dev <- gate(d_dev, no_dev)
    # loglik: both sides are the SAME kernel, so it round-off-gates like
    # beta/se/deviance -- not the looser cross-engine loglik bands, which exist
    # only because the oracles are a genuinely different implementation.
    no_ll <- is.null(a$loglik) || is.null(b$loglik)
    d_ll <- if (no_ll) NA_real_ else port_rel_max(a$loglik, b$loglik)
    m_ll <- gate(d_ll, no_ll)
    coef_ok <- identical(a$coef_names, b$coef_names)

    marks <- c(m_beta, m_se_rx, m_se_h, m_sd, m_dev, m_ll)
    failed <- any(marks %in% c("FAIL", "FAIL(len)")) || !coef_ok
    if (name %in% known) {
      # Relabel FAIL -> KNOWN for display; do NOT count toward any_fail.
      marks[marks %in% c("FAIL", "FAIL(len)")] <- "KNOWN"
    } else {
      any_fail <<- any_fail || failed
    }
    cat(sprintf("%-34s  %s %s %s %s %s %s  %s\n", name,
                cell(d_beta, marks[1]), cell(d_se_rx, marks[2]), cell(d_se_h, marks[3]),
                cell(d_sd, marks[4]), cell(d_dev, marks[5]), cell(d_ll, marks[6]),
                if (coef_ok) "ok" else "MISMATCH"))
  }
  # Every port run says how much of the glmm run it actually covered. A port run
  # from a different cell selection shares no cell with this one, prints an empty
  # table and would otherwise read as a clean pass -- the same "a gate that
  # skipped itself is not a gate that passed" rule as the missing-run case above.
  # A PARTIAL overlap is legitimate (the ports are run on the fast subset), so it
  # is reported, not failed; only zero overlap fails.
  cat(sprintf("compared: %d of %d cells in the glmm run\n", compared, length(rust$recs)))
  if (compared == 0L) {
    cat(sprintf(paste0("FAIL: the %s run shares no cell with the glmm run, so its port ",
                       "gate compared nothing. Run that port over the same cells.\n"),
                engine))
    any_fail <<- TRUE
  }
}

# ── gate 3 printing ──────────────────────────────────────────────────────────
# One table per (family, arm, quantity): every oracle's err_oracle, n,
# paired_mean and paired_2sem beside glmm's err_glmm, so an oracle that is itself
# off is visible rather than merely not chosen. paired_2sem is the run's own
# measurement of the quantity the band was pinned from; it is printed, never
# gated, and it is how a later re-pin gets its number.
truth_table_block <- function(label, pt, family, gated) {
  cat(sprintf("\n--- %s ---\n", label))
  if (pt$n_undefined > 0) {
    cat(sprintf("  %d cell(s) excluded: correlation undefined on a pinned block\n",
                pt$n_undefined))
  }
  if (!length(pt$rows)) {
    # An empty table is benign only when there was nothing to score: a group of
    # scalar-random-effect cells has no correlation, and no verdict is owed. When
    # cells DID carry a truth vector and not one pair survived, the gate could not
    # run on cells it was supposed to cover, and that fails -- a truth comparison
    # that could not be made is not a truth comparison that passed.
    if (pt$n_scoreable <= 0L) {
      cat("  no cell in this group carries a scoreable truth vector for this quantity -- nothing to score\n")
      return(invisible(NULL))
    }
    # `--dev-floor` runs this table with gated = FALSE and carries no verdict, so
    # there the same state is a note about what could not be measured, not a
    # failure label the mode is in no position to hand out.
    cat(sprintf(paste0("  %s %d cell(s) carry a scoreable truth vector for this ",
                       "quantity, but on none of them did glmm and an oracle on this arm ",
                       "both converge, so the family was never scored\n"),
                if (gated) "FAIL(truth):" else "not scored:", pt$n_scoreable))
    if (gated) any_fail <<- TRUE
    return(invisible(NULL))
  }
  cat(sprintf("  %-14s %5s  %-11s %-11s %-12s %-12s\n",
              "oracle", "n", "err_glmm", "err_oracle", "paired_mean", "paired_2sem"))
  for (r in pt$rows) {
    cat(sprintf("  %-14s %5d  %-11.4e %-11.4e %-12.3e %-12.3e\n",
                r$oracle, r$n, r$err_glmm, r$err_oracle, r$paired_mean, r$paired_2sem))
  }
  if (!gated) {
    cat("  reported only\n")
    return(invisible(NULL))
  }
  best <- pt$rows[[which.min(vapply(pt$rows, `[[`, 0, "err_oracle"))]]
  band <- truth_band(family)
  fail <- best$err_glmm > best$err_oracle + band
  cat(sprintf("  VERDICT: best=%s  err_glmm %.4e vs err_best %.4e + band %.1e  -> %s\n",
              best$oracle, best$err_glmm, best$err_oracle, band,
              if (fail) "FAIL(truth)" else "ok"))
  if (fail) any_fail <<- TRUE
}

# ── --dev-floor: MEASUREMENT, not a gate ─────────────────────────────────────
# Prints the tables the pilot needs to pin dev_eps, dev_big, truth_floor and the
# per-family truth bands, and carries no verdict. It deliberately lives here
# rather than in a script of its own so the numbers are measured by exactly the
# code that will later gate on them -- a separate measurer can drift from the
# gate, and then the band no longer describes the thing being gated.
#
# It must tolerate dev_eps / dev_big / truth_floor being NA: it is what measures
# them. So it reads none of the three, and uses a local E_BIG_PROBE for the
# outlier exclusion -- wide enough that nothing real is excluded while the real
# constant is still unknown.
E_BIG_PROBE <- 50

dev_floor_report <- function(cells, glmm_recs, orec_by_cell) {
  cat("\n=== --dev-floor: MEASUREMENT, not a gate ===\n")
  cat(sprintf(paste0("Outlier exclusion inside this mode uses a local probe of %g ",
                     "deviance units, not TOL$dev_big, which is what this mode ",
                     "measures.\n"), E_BIG_PROBE))

  # The truth-error tables need a floor, and the floor is one of the things this
  # mode measures, so it is measured FIRST and the tables below are computed
  # against that measurement rather than against a number picked here. Same rule
  # as item 5 prints: ceil1(2 x max nonzero |truth| under 0.1 on the deliberate
  # near-zero cells).
  small <- near_zero_truth_coords(cells)
  nz <- small[small > 0]
  floor_probe <- if (length(nz)) ceil1(2 * max(nz)) else 0
  cat(sprintf(paste0("Truth errors below use a floor probe of %g, measured from this ",
                     "run's own near-zero coordinates (item 5), not TOL$truth_floor.\n"),
              floor_probe))

  # 1. Delta dev per cell, glmm vs best oracle, sorted by |Delta| desc, with a
  #    `benign` flag: beta, the SEs and stddev all inside their bands against at
  #    least one oracle. ONLY benign cells may set a stopping-rule noise floor --
  #    a cell whose parameters disagree is a different fit, not a noise sample.
  rows <- list()
  for (cid in names(cells)) {
    cl <- cells[[cid]]; g <- glmm_recs[[cid]]
    if (is.null(g)) next
    orec <- orec_by_cell[[cid]]
    b <- best_oracle_dev(cl, orec, E_BIG_PROBE)
    dg <- gate1_glmm_dev(g, cl, b)
    delta <- if (is.na(dg) || is.na(b$best)) NA_real_ else as.numeric(dg) - b$best
    benign <- TRUE
    for (q in c("beta", "se_rx", "se_hessian", "stddev")) {
      if (!has_quantity(g, q)) next
      d <- nearest_oracle_diff(g, orec, cl, q)$diff
      if (is.na(d) || d > band_for(cl, q)) benign <- FALSE
    }
    rows[[cid]] <- list(cell = cid, family = cl[["family"]], arm = cell_arm(cl),
                        delta = delta, benign = benign, who = b$who)
  }
  ord <- order(vapply(rows, function(r) if (is.na(r$delta)) -Inf else abs(r$delta), 0),
               decreasing = TRUE)
  cat("\n--- 1. per-cell Delta dev (glmm - best oracle), |Delta| descending ---\n")
  cat(sprintf("%-34s %-17s %-8s %-14s %-8s %s\n",
              "cell", "family", "arm", "delta_dev", "benign", "best"))
  for (r in rows[ord]) {
    cat(sprintf("%-34s %-17s %-8s %-14s %-8s %s\n", r$cell, r$family, r$arm,
                if (is.na(r$delta)) "n/a" else sprintf("%.6g", r$delta),
                if (r$benign) "yes" else "no",
                if (is.na(r$who)) "-" else r$who))
  }

  # 2. per-family max benign |Delta dev|, and the suggested dev_eps.
  cat("\n--- 2. max benign |Delta dev| per family, suggested dev_eps = ceil1(10 x max) ---\n")
  cat(sprintf("%-17s %5s %-14s %s\n", "family", "n", "max_benign", "suggested_dev_eps"))
  for (fam in sort(unique(vapply(rows, `[[`, "", "family")))) {
    v <- vapply(Filter(function(r) identical(r$family, fam) && r$benign &&
                                   !is.na(r$delta), rows), function(r) abs(r$delta), 0)
    cat(sprintf("%-17s %5d %-14s %s\n", fam, length(v),
                if (!length(v)) "n/a" else sprintf("%.6g", max(v)),
                if (!length(v)) "n/a" else sprintf("%.6g", ceil1(10 * max(v)))))
  }

  # 3. dev_big must sit far above the benign spread and far below the smallest
  #    REAL convention constant. The two edges it has to fit between: the largest
  #    pairwise oracle-vs-oracle aligned-deviance gap per family (below it), and
  #    the nAGQ saturated deficit the AGQ cells carry (above it).
  cat("\n--- 3. largest pairwise oracle-vs-oracle |Delta dev| per family ---\n")
  gaps <- list()
  for (cid in names(cells)) {
    cl <- cells[[cid]]
    dv <- c()
    for (o in names(orec_by_cell[[cid]])) {
      r <- orec_by_cell[[cid]][[o]]
      if (!isTRUE(r$converged) || !identical(engine_arm(o, cl), cell_arm(cl))) next
      d <- aligned_dev(r, cl)
      if (!is.na(d)) dv <- c(dv, as.numeric(d))
    }
    if (length(dv) >= 2L)
      gaps[[cid]] <- list(family = cl[["family"]], gap = max(dv) - min(dv))
  }
  cat(sprintf("%-17s %5s %s\n", "family", "n", "max_pairwise_gap"))
  for (fam in sort(unique(vapply(cells, `[[`, "", "family")))) {
    v <- vapply(Filter(function(x) identical(x$family, fam), gaps), `[[`, 0, "gap")
    cat(sprintf("%-17s %5d %s\n", fam, length(v),
                if (!length(v)) "n/a" else sprintf("%.6g", max(v))))
  }
  cat("\n--- 3b. nAGQ saturated deficit on the AGQ cells (a REAL convention constant) ---\n")
  agq <- Filter(function(cl) identical(cell_arm(cl), "agq"), cells)
  if (!length(agq)) {
    cat("  no AGQ cell in this selection\n")
  } else {
    for (cl in agq) {
      g <- glmm_recs[[cl[["cell"]]]]
      s <- saturated_loglik(g, grid_read_data(cl), cl)
      cat(sprintf("  %-34s %-17s deficit = %s\n", cl[["cell"]], cl[["family"]],
                  if (is.na(s)) "no closed form" else sprintf("%.6g", abs(2 * s))))
    }
  }

  # 4. the truth-band measurement: err_e for glmm and every oracle, the paired
  #    differences, their mean, their SEM and 2 x SEM (the band), and n.
  cat("\n--- 4. truth error per family and arm (the paired_2sem column IS the band) ---\n")
  truth_gate(cells, glmm_recs, orec_by_cell, gated = FALSE, floor_ = floor_probe)

  # 5. the |truth| coordinates the truth_floor has to clear: the deliberate
  #    near-zero ones the `nearzero` and `boundary` regimes generate.
  cat("\n--- 5. |truth| coordinates below 0.1 on the nearzero and boundary cells ---\n")
  if (!length(small)) {
    cat("  no near-zero coordinate on any nearzero or boundary cell in this selection\n")
  } else {
    cat(sprintf("  %d coordinates, spanning %.6g .. %.6g; %d exactly zero\n",
                length(small), min(small), max(small), sum(small == 0)))
    cat(sprintf("  suggested truth_floor = ceil1(2 x max nonzero) = %s\n",
                if (!length(nz)) "n/a (every coordinate is exactly zero)"
                else sprintf("%.6g", ceil1(2 * max(nz)))))
  }
}

# ── gate 3 driver ────────────────────────────────────────────────────────────
# Runs only over cells with a truth AND a seed -- the generated ones. Empirical
# and committed-fixture cells have no recorded truth and are excluded; the count
# is printed on its own line, separately from the TRUTH-MISMATCH list, because
# "no truth recorded" and "truth recorded but not comparable" are different
# statements.
truth_gate <- function(cells, glmm_recs, orec_by_cell, gated = TRUE, floor_ = NULL) {
  gen <- Filter(function(cl) !is.null(cl[["truth"]]) && !is.null(cl[["seed"]]), cells)
  cat(sprintf("\ncells excluded from the truth gate (no recorded truth): %d of %d\n",
              length(cells) - length(gen), length(cells)))
  if (!length(gen)) return(invisible(NULL))
  # truth_floor is grid-wide; the cell id only routes the read through tol_for's
  # unmeasured-constant guard. --dev-floor passes its own probe instead, because
  # it is the mode that measures the constant.
  if (is.null(floor_)) floor_ <- tol_for(gen[[1]][["cell"]], "truth_floor")
  mismatches <- character(0)
  keys <- unique(vapply(gen, function(cl) paste(cl[["family"]], cell_arm(cl)), ""))
  for (key in sort(keys)) {
    grp <- Filter(function(cl) identical(paste(cl[["family"]], cell_arm(cl)), key), gen)
    family <- grp[[1]][["family"]]
    for (q in c("beta", "stddev", "corr")) {
      pt <- paired_truth_table(grp, glmm_recs, orec_by_cell,
                               truth_extract[[q]], truth_vector[[q]], floor_)
      mismatches <- c(mismatches, pt$mismatches)
      truth_table_block(sprintf("%s / %s", key, q), pt, family, gated)
    }
  }

  # sigma / dispersion / nb_theta: REPORTED, never gated. Gate 3 scores beta,
  # random-effect SDs and correlations only. This table is here because the
  # dispersion is the parameter the grid was built to watch and recording it
  # costs one more column.
  cat("\n--- dispersion-scale parameters vs truth (reported, not gated) ---\n")
  cat(sprintf("%-17s %-10s %-14s %5s  %s\n", "family", "quantity", "engine", "n", "mean_rel_err"))
  for (fam in sort(unique(vapply(gen, `[[`, "", "family")))) {
    grp <- Filter(function(cl) identical(cl[["family"]], fam), gen)
    for (q in c("sigma", "nb_theta")) {
      for (eng in c("glmm", ORACLES)) {
        v <- c()
        for (cl in grp) {
          r <- if (identical(eng, "glmm")) glmm_recs[[cl[["cell"]]]]
               else orec_by_cell[[cl[["cell"]]]][[eng]]
          if (is.null(r) || !isTRUE(r$converged)) next
          p <- truth_scale_pairs(cl, r)[[q]]
          if (is.null(p)) next
          v <- c(v, abs(p[1] - p[2]) / abs(p[2]))
        }
        if (!length(v)) next
        cat(sprintf("%-17s %-10s %-14s %5d  %.4e\n", fam, q, eng, length(v), mean(v)))
      }
    }
  }

  # A truth comparison that could not be made is not a truth comparison that
  # passed, so every length mismatch FAILS the gate. A coefficient-ordering bug,
  # a dropped aliased column or a varcomp flattened in a different group order
  # all land here.
  cat("\n=== TRUTH-MISMATCH (estimate and truth vectors could not be aligned) ===\n")
  if (!length(mismatches)) {
    cat("  none\n")
  } else {
    for (line in mismatches) cat(sprintf("  %s\n", line))
    if (gated) any_fail <<- TRUE
  }
}

# ── loading ──────────────────────────────────────────────────────────────────
# The first two lines of output name the run every number below comes from --
# its directory, its subset and label -- and the third says what the deviance
# scale rests on. No run is ever gated or reported without saying which one it
# was, so this is the first thing every mode does.
load_glmm_run <- function(grid_dir) {
  run <- newest_glmm_run(grid_dir, "glmm", opt_glmm_run)
  if (is.null(run)) {
    stop("no glmm run under runs/glmm/ (scratch included) has both run_meta.json ",
         "and results.jsonl", call. = FALSE)
  }
  st <- align_status()
  cat(sprintf("  align:   %d confirmed, %d derived, %d unconfirmed of %d deviance entries\n",
              sum(st$status == "confirmed"), sum(st$status == "derived"),
              sum(st$status == "unconfirmed"), nrow(st)))
  run
}

# Everything the gates and --dev-floor both read, loaded once here so the two
# modes cannot drift apart: --dev-floor measures the constants the gates will
# later read, and a measurement taken on a different cell selection or a
# different reference run does not describe the thing being gated.
load_inputs <- function(grid_dir) {
  glmm_run <- load_glmm_run(grid_dir)
  # A run that covered less than the selection would narrow every gate below it
  # while still printing a pass, so the two have to be asked for together: only
  # `--fast` may be gated against a `fast` run. Any other partial run is a
  # precondition error, because the cells it never fitted are exactly the ones a
  # gate cannot speak for.
  subset <- glmm_run$meta[["subset"]]
  if (!identical(subset, "full") && !opt_fast) {
    stop(sprintf(paste0("the glmm run's subset is \"%s\", not \"full\", and --fast was not ",
                        "given: gating the whole selection against a partial run would ",
                        "narrow every gate silently. Pass --fast (for a `fast` run) or gate ",
                        "a full run."), subset), call. = FALSE)
  }
  cells_all <- grid_cells_by_id(file.path(grid_dir, "manifest.json"))
  cells <- if (opt_fast) Filter(function(cl) "fast" %in% cl[["tags"]], cells_all) else cells_all
  if (!length(cells)) stop("no cell selected -- the manifest tags no cell `fast`", call. = FALSE)
  have <- intersect(names(cells), names(glmm_run$recs))
  cat(sprintf("  cells:   %d selected, %d with a glmm record, %d without\n",
              length(cells), length(have), length(cells) - length(have)))
  # The selected cells the run holds no record for travel on rather than being
  # dropped: a gate that quietly compares fewer cells than it was asked to is not
  # a gate that passed, so they are failed by name in the glmm-fail block.
  no_record <- setdiff(names(cells), have)
  cells <- cells[have]

  V <- grid_versions(grid_dir)
  runs <- list()
  for (o in ORACLES) {
    # What a complete run of this oracle must contain, taken from the WHOLE
    # manifest and not from the current selection: a `--fast` comparison still
    # reads a full-grid reference, and a reference judged complete against 20
    # cells would not be one.
    required <- names(Filter(function(cl) o %in% cl[["oracles"]], cells_all))
    r <- newest_oracle_run(grid_dir, o, V[[o]], required)
    if (is.null(r)) {
      stop(sprintf(paste0("no run under runs/%s/ is an accuracy reference for %s: it ",
                          "needs subset \"full\", no timing, and a record for all %d ",
                          "cells whose manifest `oracles` name it. Run the full grid ",
                          "for that oracle first."),
                   ENGINE_DIR[[o]], o, length(required)), call. = FALSE)
    }
    runs[[o]] <- r
    cat(sprintf("%-14s %s  (version %s, %d cells required)\n",
                paste0(o, ":"), r$dir, r$meta$engine_version, length(required)))
  }
  # Built once per cell and reused by gates 1, 2 and 3: an oracle that did not fit
  # a cell is simply ABSENT from its list rather than present with a NULL that
  # every consumer has to guard.
  orec_by_cell <- setNames(lapply(names(cells), function(cid)
    Filter(Negate(is.null),
           setNames(lapply(ORACLES, function(o) runs[[o]]$recs[[cid]]), ORACLES))),
    names(cells))
  list(grid_dir = grid_dir, glmm_run = glmm_run, cells = cells,
       orec_by_cell = orec_by_cell, no_record = no_record)
}

# ── the main flow ────────────────────────────────────────────────────────────
run_gates <- function() {
  grid_dir <- script_dir

  # --ports-only short-circuits past versions.json, past every oracle lookup and
  # past every pilot constant: it is the gate a CI job calls, so it has to run on
  # a machine with no statistics package installed at all.
  if (opt_ports) {
    glmm_run <- load_glmm_run(grid_dir)
    port_gate(grid_dir, glmm_run, "glmm_python", character(0))
    port_gate(grid_dir, glmm_run, "glmm_r", KNOWN_R_PARSE)
    return(invisible(NULL))
  }

  inp <- load_inputs(grid_dir)
  glmm_run <- inp$glmm_run; cells <- inp$cells; orec_by_cell <- inp$orec_by_cell

  # Deviance-gate summary accumulators, printed as three blocks below.
  dev_win <- character(0)   # DEV-WIN cells (Delta <= 0), informational
  dev_na <- character(0)    # DEV-NA cells -- the loud exclusion list
  dev_conv <- character(0)  # FAIL(conv?) cells, both deviances printed
  no_ref <- character(0)    # gate-2 cells with a quantity that had no reference
  se_rx_excluded <- character(0)  # cells where se_rx is n/a (Rx convention)
  no_mle <- character(0)    # no-MLE cells, and whether glmm refused them
  # Cells where glmm produced no fit to compare -- a timeout, an engine-fail, or
  # no record at all. These FAIL: "glmm did not fit this cell" is a result about
  # glmm, and the one thing it must not do is read as a pass.
  glmm_fail <- vapply(inp$no_record,
                      function(cid) sprintf("%s: glmm produced no record", cid), "")
  if (length(glmm_fail)) any_fail <<- TRUE

  cat("\n=== gates 1 and 2: per cell ===\n")
  cat(sprintf("%-34s %-17s %-8s %-18s %-10s %-20s %-10s %-10s %-10s %-14s %s\n",
              "cell", "family", "arm", "dev", "beta", "se_rx", "se_hess",
              "stddev", "corr", "best", "excluded / note"))
  for (cid in names(cells)) {
    cl <- cells[[cid]]; g <- glmm_run$recs[[cid]]; orec <- orec_by_cell[[cid]]

    # A cell whose data admit no maximum-likelihood estimate (manifest `no_mle`,
    # set by gen_manifest.R) has no optimum for gates 1 and 2 to compare, so it is
    # gated the other way round: the only correct result is glmm's own refusal,
    # the record engines/common.rs writes for a fit that returned unconverged.
    # A converged glmm fit FAILS here, and so do a panic, a timeout or a declined
    # launch, since none of them is that refusal. The oracle records are printed,
    # not consulted: GLM.jl reports "converged" on separated data once the
    # deviance stops moving, at a finite point on the path to infinity.
    if (!is.null(cl[["no_mle"]])) {
      refused <- !isTRUE(g$converged) && identical(g$status, "engine-fail") &&
                 startsWith(if (is.null(g$message)) "" else g$message, "not converged:")
      m <- if (refused) "no-MLE" else "FAIL(no-MLE)"
      if (!refused) any_fail <<- TRUE
      ostat <- vapply(names(orec), function(o)
        sprintf("%s=%s", o, if (isTRUE(orec[[o]]$converged)) "converged" else orec[[o]]$status), "")
      no_mle <- c(no_mle, sprintf("%s: %s, glmm status=%s message=%s; oracles %s; %s",
                                  cid, if (refused) "glmm refused as required"
                                       else "glmm did NOT refuse",
                                  if (is.null(g$status)) "-" else g$status,
                                  if (is.null(g$message)) "-" else g$message,
                                  paste(ostat, collapse = ", "), cl[["no_mle"]]))
      cat(sprintf("%-34s %-17s %-8s %-18s see the no-MLE block\n",
                  cid, cl[["family"]], cell_arm(cl), m))
      next
    }

    # Gate 1. eps and big are read through tol_for, never off TOL directly: that
    # is where the unmeasured-constant guard lives. A `dev_ref` cell has no
    # usable oracle deviance (dev_ref.json's `method` field says why), so `b`
    # comes from the frozen high-precision value instead of the cell's oracles.
    eps <- tol_for(cid, "dev_eps"); big <- tol_for(cid, "dev_big")
    b <- if (!is.null(cl[["dev_ref"]])) dev_ref_dev(cid) else best_oracle_dev(cl, orec, big)
    dg <- gate1_glmm_dev(g, cl, b)
    rule <- attr(dg, "rule")
    who_dev <- if (is.null(rule)) b$who else sprintf("%s under %s", b$who, rule)
    delta <- if (is.na(dg) || is.na(b$best)) NA_real_ else as.numeric(dg) - b$best
    m_dev <- dev_verdict(delta, eps, big)
    why_dev <- if (!identical(m_dev, "DEV-NA")) NA_character_
               else if (is.na(dg)) attr(dg, "why") else b$why

    # Gate 2. Only CONVERGED oracle records are a reference; an oracle that ran
    # and did not converge has no estimate to be near. Of those, a record on a
    # different objective is left out per quantity (gate2_excluded above). A
    # quantity with no converged reference left, or whose references carry no
    # finite value of it, is `no-ref`: a loud exclusion like DEV-NA, never a
    # FAIL. Charging it to glmm as FAIL(len) would read as "glmm disagrees" when
    # the true statement is "nothing to compare against". `n/a` stays the
    # narrower case -- a usable oracle exists, but nobody on either side reports
    # that particular quantity.
    orec_conv <- Filter(function(r) isTRUE(r$converged), orec)
    diffs <- setNames(rep(NA_real_, length(GATE2_QUANTITIES)), GATE2_QUANTITIES)
    marks <- setNames(rep("n/a", length(GATE2_QUANTITIES)), GATE2_QUANTITIES)
    no_ref_q <- character(0)
    # A glmm record that is not a converged fit has nothing to compare, so no
    # quantity of it is put to an oracle and the cell is failed by name below.
    # Reading the numbers of such a record instead says the wrong thing twice
    # over: a timeout or an engine-fail carries an empty beta, which prints `n/a`
    # on every column and passes the cell in silence, or a same-length all-null
    # beta, which prints FAIL(len) and reads as "glmm disagrees".
    if (!isTRUE(g$converged)) {
      marks[] <- "glmm-fail"
      glmm_fail <- c(glmm_fail, sprintf("%s: status=%s message=%s", cid,
                                        if (is.null(g$status)) "-" else g$status,
                                        if (is.null(g$message)) "-" else g$message))
      any_fail <<- TRUE
    } else {
      for (q in GATE2_QUANTITIES) {
        # se_rx has no oracle in glmm's Rx convention on a mixed probit,
        # cloglog, NB or Gamma cell (se_rx_no_convention_oracle above) --
        # excluded outright, not merely left without a reference, so it is
        # reported here rather than falling into `no-ref`/`n/a` below.
        # Canonical-link cells are unaffected.
        if (identical(q, "se_rx") && se_rx_no_convention_oracle(cl)) {
          marks[[q]] <- "n/a (Rx convention)"
          se_rx_excluded <- c(se_rx_excluded, cid)
          next
        }
        orec_ref <- Filter(function(r) !gate2_excluded(r, cl, q), orec_conv)
        if (!length(orec_ref)) {
          marks[[q]] <- "no-ref"
          no_ref_q <- c(no_ref_q, sprintf("%s (no converged oracle on glmm's objective)", q))
          next
        }
        if (!has_quantity(g, q)) { marks[[q]] <- "n/a"; next }
        if (!any(vapply(orec_ref, has_quantity, TRUE, q))) { marks[[q]] <- "n/a"; next }
        if (!any(vapply(orec_ref, reports_finite, TRUE, q))) {
          marks[[q]] <- "no-ref"
          no_ref_q <- c(no_ref_q, sprintf("%s (no oracle reports a finite value)", q))
          next
        }
        nd <- nearest_oracle_diff(g, orec_ref, cl, q)
        diffs[[q]] <- nd$diff
        # Recorded as compared only HERE, on the branch that reaches the registry:
        # the stale check may only call an entry dead on a quantity this run put to
        # it, never on one that came out `n/a` or `no-ref`.
        div_seen <<- union(div_seen, paste(cid, q))
        marks[[q]] <- div_mark(mark(nd$diff, band_for(cl, q)), nd$diff, cid, q,
                               "glmm-vs-oracle")
      }
    }
    if (length(no_ref_q))
      no_ref <- c(no_ref, sprintf("%s: %d oracle record(s), %d converged; %s",
                                  cid, length(orec), length(orec_conv),
                                  paste(no_ref_q, collapse = ", ")))

    # The deviance mark is a HARD gate with no registry escape; DEV-NA never
    # fails but is collected below for the loud-exclusion summary.
    failed <- any(marks %in% c("FAIL", "FAIL(len)")) ||
              startsWith(m_dev, "FAIL")
    any_fail <<- any_fail || failed
    if (identical(m_dev, "DEV-WIN"))
      dev_win <- c(dev_win, sprintf("%s: Delta=%.6g vs %s", cid, delta, who_dev))
    if (identical(m_dev, "DEV-NA"))
      dev_na <- c(dev_na, sprintf("%s: %s", cid, why_dev))
    if (identical(m_dev, "FAIL(conv?)"))
      dev_conv <- c(dev_conv, sprintf("%s: dev_glmm=%.6g dev_%s=%.6g Delta=%.6g",
                                      cid, as.numeric(dg), who_dev, b$best, delta))

    cat(sprintf("%-34s %-17s %-8s %s %s %s %s %s %s %-14s %s\n",
                cid, cl[["family"]], cell_arm(cl),
                cell(delta, m_dev, 18),
                cell(diffs[["beta"]], marks[["beta"]]),
                cell(diffs[["se_rx"]], marks[["se_rx"]], 20),
                cell(diffs[["se_hessian"]], marks[["se_hessian"]]),
                cell(diffs[["stddev"]], marks[["stddev"]]),
                cell(diffs[["corr"]], marks[["corr"]]),
                if (is.na(b$who)) "-" else b$who,
                # The same column carries both things the best-oracle selection
                # has to say, and it can only ever say one of them: exclusion
                # needs three candidates, the gap is printed at exactly two. The
                # quadrature rule gate 1 used, when it is not glmm's own, goes
                # after it.
                paste(c(if (length(b$excluded)) paste(b$excluded, collapse = ",")
                        else if (!is.na(b$gap)) sprintf("2-oracle gap=%.3g", b$gap)
                        else if (is.null(rule)) "-",
                        if (!is.null(rule)) sprintf("dev: %s at glmm's point", rule)),
                      collapse = "; ")))
  }

  # Three blocks, always printed, never conditional on any_fail: DEV-WIN is
  # informational so a passing run still shows it; DEV-NA is the loud exclusion
  # list -- a cell with no usable reference deviance must never disappear
  # silently, so it prints "none" rather than being skipped when empty;
  # FAIL(conv?) repeats both raw deviances so a convention mismatch is legible
  # without re-running.
  cat("\n=== DEV-WIN (Delta dev <= 0 vs the best oracle; informational) ===\n")
  if (length(dev_win)) for (line in dev_win) cat(sprintf("  %s\n", line)) else cat("  none\n")

  cat("\n=== DEV-NA (no usable reference deviance -- excluded loudly, not passed hollow) ===\n")
  if (length(dev_na)) for (line in dev_na) cat(sprintf("  %s\n", line)) else cat("  none\n")

  cat("\n=== FAIL(conv?) (|Delta dev| > dev_big -- convention mismatch, not a fit disagreement) ===\n")
  if (length(dev_conv)) for (line in dev_conv) cat(sprintf("  %s\n", line)) else cat("  none\n")

  cat(sprintf(paste0("\n=== no-ref (%d cell(s): gate 2 had no reference for the quantities ",
                     "named -- no converged oracle on glmm's objective, or none that ",
                     "reports a finite value) ===\n"), length(no_ref)))
  if (length(no_ref)) for (line in no_ref) cat(sprintf("  %s\n", line)) else cat("  none\n")

  cat(sprintf(paste0("\n=== se_rx: not gated on %d mixed probit/cloglog/negativebinomial/",
                     "gamma cell(s) -- printed n/a (Rx convention). glmm's Rx is the Schur ",
                     "complement of the OBSERVED information; the ",
                     "only oracles that report se_rx (lme4, MixedModels.jl) use the EXPECTED ",
                     "information at their own optimum. Canonical-link cells stay gated. ===\n"),
              length(se_rx_excluded)))

  # The mirror image of no-ref, and unlike it a FAILURE: there the reference is
  # missing, here glmm's own fit is. Printed even when empty, for the same reason
  # DEV-NA is -- a cell nothing was compared on must never disappear quietly.
  cat(sprintf("\n=== glmm-fail (%d cell(s): glmm produced no fit, so nothing was compared) ===\n",
              length(glmm_fail)))
  if (length(glmm_fail)) for (line in glmm_fail) cat(sprintf("  %s\n", line)) else cat("  none\n")

  # Printed even when empty: a cell gated on its refusal must never read as one
  # that was compared, nor disappear.
  cat(sprintf(paste0("\n=== no-MLE (%d cell(s): the data admit no maximum-likelihood ",
                     "estimate, so glmm must refuse; a converged glmm fit fails) ===\n"),
              length(no_mle)))
  if (length(no_mle)) for (line in no_mle) cat(sprintf("  %s\n", line)) else cat("  none\n")

  # Printed unconditionally when anything fired, so a DOC cell above always has a
  # named reason next to it in the same output.
  if (length(div_fired)) {
    cat("\n=== documented divergences (reference check, not a gate) ===\n")
    for (e in DIV) {
      if (!(e$id %in% div_fired)) next
      cat(sprintf("%-34s %-22s <= %.1e  %s\n", e$cell,
                  paste(e$quantities, collapse = ","), e$max_rel, e$id))
      cat(sprintf("  direction: %s\n", e$direction))
      cat(sprintf("  summary: %s\n", e$summary))
    }
  }
  # A registry entry whose (cell, quantity) WAS compared and did not fire is
  # stale: the divergence it excuses is gone, and leaving it would turn the entry
  # into a standing exemption for whatever drifts there next. Scoped to what this
  # run actually compared, so neither a subset run nor a quantity that came out
  # `n/a` trips it.
  stale <- Filter(function(e) "glmm-vs-oracle" %in% e$comparison &&
                              any(paste(e$cell, e$quantities) %in% div_seen) &&
                              !(e$id %in% div_fired), DIV)
  if (length(stale)) {
    cat("\n")
    for (e in stale) {
      cat(sprintf("STALE registry entry: %s (%s) no longer fires -- delete it\n",
                  e$id, e$cell))
    }
    any_fail <<- TRUE
  }

  cat("\n=== gate 3: truth error ===\n")
  truth_gate(cells, glmm_run$recs, orec_by_cell)

  port_gate(grid_dir, glmm_run, "glmm_python", character(0))
  port_gate(grid_dir, glmm_run, "glmm_r", KNOWN_R_PARSE)
}

# --dev-floor reads exactly the inputs the gates read, through the same loader,
# but runs no gate and sets no verdict.
run_dev_floor <- function() {
  inp <- load_inputs(script_dir)
  dev_floor_report(inp$cells, inp$glmm_run$recs, inp$orec_by_cell)
}

status <- tryCatch({
  # Sourced INSIDE the tryCatch: a load-time failure here -- an unmeasured
  # tol.R constant, a TOL_PER_CELL key that names no cell -- is a "the gate could
  # not run" condition, which the exit contract puts at 2, and outside the
  # tryCatch it would propagate as a bare R error and a status of 1. runs.R brings
  # tol.R, dev_align.R and engines/versions.R with it, so the comparison
  # vocabulary arrives in one piece and no consumer can load half of it.
  source(file.path(script_dir, "runs.R"))
  parse_args()
  if (opt_devfloor) {
    run_dev_floor()
  } else {
    run_gates()
    cat(sprintf("\n%s\n",
                if (any_fail) "RESULT: disagreements found -- investigate (flag, do not relax tolerance)"
                else "RESULT: all gated quantities agree, or diverge as documented"))
  }
  if (any_fail) 1L else 0L
}, error = function(e) {
  cat("PRECONDITION: ", conditionMessage(e), "\n", sep = "")
  2L
})
quit(status = status)
