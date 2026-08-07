#!/usr/bin/env Rscript
# Diligent-grid analysis (full-AGQ vector-RE spec Part 6: docs/GLMM/plans/
# 2026-07-12-full-agq-vector-re-spec.md). Joins glmm vs the oracle per cell over
# the 510-cell diligent manifest and answers: does glmm match the reference
# engines' ANSWERS -- estimates, variance components, Hessian SEs -- across the
# whole grid, and where AGQ applies is it right at matched nodes?
#
#   Rscript analyze.R [glmm.jsonl] [oracle.jsonl]
#   (defaults: results/glmm.jsonl, results/lme4.jsonl)
#
# Three cell categories, each with its own oracle + tolerance band (tol.R):
#   * laplace  (477 cells, no `nagq`)  -- oracle glmer/lmer Laplace; main bands
#                                         (beta_rel, stddev_rel, se_hessian_rel).
#   * agq_int1 (15 cells)              -- oracle glmer(nAGQ=7); agq_* bands.
#   * agq_q2s  (18 cells; 17 joinable) -- oracle GLMMadaptive(nAGQ=7); agq_*
#                                         bands, NO deviance (GLMMadaptive's
#                                         logLik carries different additive
#                                         constants than glmer's devfun -- spec
#                                         Part 6). The 6 bina_q2s_* aggregated-
#                                         binomial cells join like the rest since
#                                         the AGQ x weights gate was lifted
#                                         2026-07-14 (1 GLMMadaptive oracle-timeout
#                                         leaves 17 joinable).
#
# Deviance is a cross-check (aligned -2logL) only on lme4/lmer rows, matching
# analyze_grid.R's alignment (gaussian REML offset; GLMM identity). It is not a
# hard gate -- beta/stddev/se/corr are the validation gates.
suppressMessages({ library(jsonlite) })

suite_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(suite_dir, "..", "..", "tol.R"))
args <- commandArgs(TRUE)
glmm_path   <- if (length(args) >= 1) args[1] else
  file.path(suite_dir, "results", "glmm.jsonl")
oracle_path <- if (length(args) >= 2) args[2] else
  file.path(suite_dir, "results", "lme4.jsonl")
out_dir <- file.path(suite_dir, "reports")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

glmm <- read_jsonl(glmm_path); oracle <- read_jsonl(oracle_path)
manifest <- fromJSON(file.path(suite_dir, "manifest.json"),
                     simplifyDataFrame = FALSE)
cells <- setNames(manifest$cells, vapply(manifest$cells, `[[`, "", "case_id"))

# ---- helpers -------------------------------------------------------------------
# read_jsonl/stddevs_of/corrs_of live in tol.R (shared with verify_boundary37.R).
# Aligned -2logL on the oracle's recorded -2*logLik(m) scale (GLMMadaptive rows
# have deviance=null -> NA). Gaussian: glmm carries the REML criterion minus the
# df*(1+log 2pi) offset (validated in-crate, Task 2); add it back. GLMM: glmm is
# on the UNIT-DEVIANCE scale (D = 2(logL_sat - logL), saturated constants
# dropped -- pinned vs MixedModels' objective(), analyze_grid.R), while
# speed-grid/fit.R records glmer's -2*logLik, which KEEPS them: aligned = D - 2 logL_sat.
# logL_sat: binary binomial 0 (identity), poisson sum dpois(y,y,log),
# aggregated binomial sum dbinom(k,n,k/n,log). Pinned on this run: unaligned,
# pois/bina cells sat at a constant ~0.6 relative offset while binb cells agreed
# to ~1e-9; with the correction all three families align to ~1e-9.
sat_loglik <- function(cell) {
  df <- read.csv(file.path(suite_dir, "..", "speed-grid", "data",  # campaign-local, prep.R output
                           paste0(cell$case_id, ".csv")))
  if (cell$family == "poisson") sum(dpois(df$y, df$y, log = TRUE))
  else if (!is.null(df$size))                          # aggregated binomial
    sum(dbinom(df$incidence, df$size, df$incidence / df$size, log = TRUE))
  else 0                                               # binary binomial
}
glmm_dev_aligned <- function(cell, dev) {
  if (is.null(dev) || !is.finite(dev)) return(NA_real_)
  if (cell$family == "gaussian") {
    p <- cell$n_x + 1
    dev + (cell$n_obs - p) * (1 + log(2 * pi))
  } else dev - 2 * sat_loglik(cell)
}

# AGQ x weights: the gate that made aggregated-binomial AGQ cells oracle-only was
# lifted 2026-07-14 (docs/GLMM/plans/2026-07-14-agq-prior-weights-spec.md -- prior
# weights now thread through both AGQ kernels). All 12 bina_ AGQ cells (6 int1 +
# 6 q2s) fit and join like any other cell; there is no named coverage boundary
# anymore, so any AGQ engine-fail is once again a real flag.

rows <- list()
for (cid in names(cells)) {
  cell <- cells[[cid]]
  g <- glmm[[cid]]; o <- oracle[[cid]]
  is_agq  <- !is.null(cell$nagq)
  is_q2s  <- is_agq && identical(cell$structure, "q2s")
  category <- if (!is_agq) "laplace" else if (is_q2s) "agq_q2s" else "agq_int1"
  o_engine <- if (!is.null(o)) o$engine else NA_character_
  # deviance comparable only against glmer/lmer (GLMMadaptive constants differ)
  dev_comparable <- !is.na(o_engine) && o_engine != "GLMMadaptive"

  # tolerance bands per category
  if (is_agq) {
    tb <- TOL$agq_beta_rel; ts <- TOL$agq_stddev_rel
    tse <- TOL$agq_se_hessian_rel; tc <- TOL$agq_corr_abs
  } else {
    tb <- TOL$beta_rel; ts <- TOL$stddev_rel
    tse <- TOL$se_hessian_rel; tc <- TOL$agq_corr_abs  # corr: same 4e-3 abs band as
                                       # AGQ (tol.R has no Laplace-specific corr band)
  }

  g_ok <- !is.null(g) && identical(g$status, "ok")
  o_ok <- !is.null(o) && identical(o$status, "ok")
  # glmm watchdog timeouts (30 s per-fit budget) are a COMPUTE-BUDGET boundary,
  # not an accuracy verdict -- named separately (run 2026-07-14: 13 cells, the
  # giant-LMM slow tail, cross8/q5q2/q6q2x3/q8q2x at g30000 plus one g300 q8q2x).
  g_timeout <- !is.null(g) && identical(g$status, "timeout")
  # Oracle "engine-fail" triage (run 2026-07-14): speed-grid/fit.R marks ANY lme4
  # convergence message as engine-fail, but two distinct things hide under it.
  #   * beta EMPTY  -- a real R error: every one is lme4's identifiability
  #     refusal ("number of observations <= number of random effects") on
  #     p5 high-q LMM cells (q5..q8 with 5 obs/group). glmm fits them; the
  #     oracle cannot -- a named oracle-side coverage limit, not a glmm result.
  #   * beta PRESENT -- lmer/glmer converged with a warning and returned
  #     estimates -- historically all lumped as "oracle-singular", but the
  #     warning isn't always lme4's singular-fit warning (boundary-fits
  #     follow-up spec, Part B: this file's predecessor scored this off the
  #     status proxy alone, never reading lme4's actual `singular` field).
  o_estimates <- !is.null(o) && !o_ok && length(o$beta) > 0 &&
                !identical(o$status, "timeout")
  # Oracle watchdog timeouts (120 s budget): 2 giant q8q2x LMM cells + 1
  # GLMMadaptive q2s cell on this run -- oracle-side compute limit, no answer.
  o_timeout <- !is.null(o) && identical(o$status, "timeout")
  # Real singular flags (Fit::singular / lme4 isSingular, both exported --
  # speed-grid/fit.rs's `rec["singular"]`, speed-grid/fit.R's `isSingular(m)`).
  # oracle-singular is now
  # gated on THIS, not the status proxy: a convergence-warning cell whose
  # real flag says non-singular joins and gates like any other cell instead
  # of being silently parked ungated.
  sing_glmm   <- !is.null(g) && isTRUE(g$singular)
  sing_oracle <- !is.null(o) && isTRUE(o$singular)
  o_singular  <- o_estimates && sing_oracle

  d_beta <- d_sd <- d_corr <- d_se <- d_dev <- NA_real_
  beta_len_mismatch <- sd_len_mismatch <- corr_len_mismatch <- FALSE
  if (g_ok && (o_ok || o_estimates)) {
    gb <- as.numeric(g$beta); ob <- as.numeric(o$beta)
    beta_len_mismatch <- length(gb) != length(ob)
    d_beta <- rel_max(gb, ob)
    sg <- stddevs_of(g); so <- stddevs_of(o)
    sd_len_mismatch <- length(sg) != length(so)
    d_sd   <- if (length(sg) && length(so)) rel_max(sg, so) else NA_real_
    cg <- corrs_of(g); co <- corrs_of(o)
    corr_len_mismatch <- length(cg) != length(co)
    d_corr <- if (length(cg) && length(co)) max(abs(cg - co)) else NA_real_
    d_se   <- if (!is.null(g$se) && !is.null(o$se) && length(g$se))
                rel_max(as.numeric(g$se), as.numeric(o$se)) else NA_real_
    if (dev_comparable) {
      dg <- glmm_dev_aligned(cell, g$deviance)
      dm <- if (is.null(o$deviance)) NA_real_ else as.numeric(o$deviance)
      if (is.finite(dg) && is.finite(dm)) d_dev <- abs(dg - dm) / (1 + abs(dm))
    }
  }

  # pass/fail on the joinable gates (deviance is a cross-check, not a gate).
  # beta/stddev length mismatches are real disagreements (rel_max's NA there
  # means "not comparable", never "matches") -- treat as failures, not passes.
  gate_fail <- (!is.na(d_beta) && d_beta > tb) || beta_len_mismatch ||
               (!is.na(d_sd)   && d_sd   > ts) || sd_len_mismatch ||
               (!is.na(d_corr) && d_corr > tc) || corr_len_mismatch ||
               (!is.na(d_se)   && d_se   > tse)

  status <- if (g_timeout)                       "glmm-timeout"
            else if (!g_ok && !o_ok)             "both-fail"
            else if (!g_ok)                      "glmm-fail"
            else if (o_singular)                 "oracle-singular"
            else if (o_timeout)                  "oracle-timeout"
            else if (!o_ok && !o_estimates)      "oracle-error"
            else if (gate_fail)                  "mismatch"
            else                                 "ok"

  rows[[length(rows) + 1L]] <- data.frame(
    case_id = cid, category = category, family = cell$family,
    structure = cell$structure, n_theta = cell$n_theta, n_obs = cell$n_obs,
    balance = cell$balance, regime = cell$regime,
    nagq = if (is_agq) cell$nagq else 1L,
    oracle_engine = o_engine, status = status,
    glmm_status = if (is.null(g)) "absent" else g$status,
    oracle_status = if (is.null(o)) "absent" else o$status,
    sing_glmm = sing_glmm, sing_oracle = sing_oracle,
    sing_agree = sing_glmm == sing_oracle,
    d_beta = d_beta, d_stddev = d_sd, d_corr = d_corr, d_se = d_se,
    d_dev_rel = d_dev,
    tol_beta = tb, tol_stddev = ts, tol_corr = tc, tol_se = tse,
    wall_glmm = if (g_ok) g$wall_seconds else NA_real_,
    wall_oracle = if (o_ok) o$wall_seconds else NA_real_,
    stringsAsFactors = FALSE)
}
res <- do.call(rbind, rows)

# Boundary-fits follow-up (docs/GLMM/plans/2026-07-14-boundary-fits-followup-
# spec.md, Part A): the 37 oracle-error cells (lme4 identifiability refusal,
# obs<=REs) verified against a third oracle, MixedModels.jl
# (verify_boundary37.R -> boundary37_verdicts.csv). 36/37 confirmed correct
# (34 direct agreement + 2 same-optimum-flat-direction, REML criterion aligns
# to <0.01 despite a gated parameter gap -- both engines land on a degenerate
# boundary, same argument as the q2s ADJUDICATED nearzero class); reclassified
# oracle-error -> ok with oracle_engine = MixedModels. One real disagreement
# (lmm_q8_g3000p5_bal_lowsnr, REML criterion gap +2.03: glmm converged to a
# measurably worse optimum) stays a named validation failure, not hand-waved.
b37_path <- file.path(out_dir, "boundary37_verdicts.csv")
if (file.exists(b37_path)) {
  b37 <- read.csv(b37_path, stringsAsFactors = FALSE)
  # Verdict strings are defined in verify_boundary37.R -- change both together.
  confirmed <- b37$case_id[b37$verdict %in%
    c("confirmed (MM agrees)", "boundary (same optimum, flat direction)")]
  flagged   <- b37$case_id[b37$verdict == "mismatch (real disagreement)"]
  i <- match(confirmed, res$case_id)
  res$status[i] <- "ok"; res$oracle_engine[i] <- "MixedModels"
  i <- match(flagged, res$case_id)
  res$status[i] <- "mismatch"; res$oracle_engine[i] <- "MixedModels"
}

write.csv(res, file.path(out_dir, "status_map.csv"), row.names = FALSE)

# ---- named failure list (manifest order, campaign discipline) ------------------
# Validation failures = mismatches + real glmm/both engine failures. Everything else
# with a non-ok status is a NAMED boundary section, listed after the failures:
# glmm-timeout (compute budget), oracle-error (lme4 identifiability refusal), and
# oracle-singular (oracle boundary fit -- gaps recorded, not gated).
fail_status <- c("mismatch", "glmm-fail", "both-fail")
ids_of <- function(statuses) Filter(function(cid)
  res$status[match(cid, res$case_id)] %in% statuses, names(cells))
# Per-cell attribution notes for the q2s AGQ mismatches (run 2026-07-14), two
# evidence classes:
#   * the two large-beta bal_base cells -- third-engine evidence, reproducible
#     via adjudicate.R (change together): glmer-Laplace on the same
#     data reproduces glmm's beta AND theta to 4 decimals (p20/p100 clusters,
#     Laplace ~ AGQ there), so glmm is at the right optimum and GLMMadaptive
#     under-converged -- on the p100 cell its Hessian is non-PD at its
#     "optimum" (vcov = NaN, se recorded null). Oracle-side, not glmm error.
#   * the three nearzero cells -- boundary argument, no refit needed: beta/SE/
#     intercept-stddev agree to <= 9e-4; only the ~0.008-0.1-scale slope
#     stddev (degenerate component, corr pinned at +-1) exceeds the relative
#     band, which is meaningless at that scale.
ADJUDICATED <- c(
  pois_q2s_g3000p20_bal_base  = "ADJUDICATED oracle-side (adjudicate.R): glmer-Laplace == glmm to 4 dp; GLMMadaptive under-converged",
  pois_q2s_g3000p100_bal_base = "ADJUDICATED oracle-side (adjudicate.R): glmer-Laplace == glmm to 4 dp; GLMMadaptive non-PD Hessian (se=null)",
  pois_q2s_g300p20_bal_nearzero  = "BOUNDARY near-zero theta: beta/SE agree; only the ~0.008-scale degenerate slope stddev and its corr (pinned at -1 vs -0.85) exceed the bands",
  binb_q2s_g3000p20_bal_nearzero = "BOUNDARY near-zero theta: beta/SE agree; only the ~0.1-scale degenerate slope stddev (corr pinned at +1, within band) exceeds the relative band",
  bina_q2s_g3000p20_bal_nearzero = "BOUNDARY near-zero theta: beta (2.7e-5) / SE (8.9e-4) / corr (1.3e-3, within band) agree; only the ~0.02-scale degenerate slope stddev exceeds the relative band",
  # boundary-fits follow-up spec Part A (verify_boundary37.R): the one real
  # disagreement out of 37 lme4-identifiability-refusal cells verified
  # against MixedModels.jl -- REML criterion gap +2.03, glmm at a measurably
  # worse optimum than MM on this cell (not a flat-direction/representation
  # difference like the other 36).
  lmm_q8_g3000p5_bal_lowsnr = "REAL DISAGREEMENT vs MixedModels.jl (verify_boundary37.R): REML criterion gap +2.03 -- glmm converged to a worse optimum, not a boundary/flat-direction artifact",
  # ── mismatch-diagnosis pass (classify_mismatch.R -> reports/mismatch_classes.csv;
  # verify_boundary37.R in VERIFY_OBJECTIVE mode -> reports/objective_verdicts.csv)
  #
  # Two pieces of evidence closed the cells below, and neither is a comparison
  # of the two engines' numbers -- which is the point, because on a
  # disagreement that comparison cannot say who is right.
  #
  #   1. THE OBJECTIVE AT BOTH ANSWERS. lme4's own REML criterion evaluated at
  #      glmm's answer and at lme4's. Lower wins; the engine that produced the
  #      criterion gets no home advantage from it. On every gaussian cell here
  #      glmm scores at or below lme4, so none of them is glmm landing wrong.
  #      Each verdict was checked by running lme4's own answer through the
  #      identical path first: it reproduces lme4's recorded deviance to <=4e-5,
  #      so the theta reconstruction the scoring depends on is sound.
  #   2. WHAT LME4 SAID. fit.R now records m@optinfo$conv$lme4$messages
  #      verbatim. Two cells turn out to carry lme4's own max|grad| convergence
  #      FAILURE rather than its singular-fit note -- lme4 reporting that it did
  #      not converge, on cells previously read as glmm disagreeing with it.
  #
  # The remaining 38 oracle-singular cells all carry the singular-fit note and
  # nothing else, and 25 of them break no band at all.
  lmm_q4sx2_g300p20_bal_lowsnr = "GLMM CONFIRMED, oracle stopped short (verify_boundary37.R objective mode): lme4's own REML criterion is 1529.835 at glmm's answer vs 1537.213 at lme4's -- glmm 7.38 LOWER; lme4 restarted from glmm's theta stays at 1529.835. lme4's g1 intercept SD pinned to 0 is the worse optimum, not the reference",
  binb_nest2s_g300p20_bal_base = "ORACLE UNDER-CONVERGED (fit.R messages): lme4 reports max|grad| = 0.00269 against its own 0.002 tol -- it did not converge. Deviance agrees to 1.0e-8 rel; the one breaching coefficient is the 0.0320 intercept, 4.8e-5 apart in absolute terms against its own SE of 0.223",
  lmm_cross8_g3000p20_single_base = "ORACLE UNDER-CONVERGED (fit.R messages): lme4 reports max|grad| = 0.0179 against its own 0.002 tol. Objective mode agrees -- glmm scores 3.3e-4 lower on lme4's own REML criterion",
  lmm_cross4_g3000p5_bal_nearzero  = "SAME OPTIMUM (objective mode): glmm 2.5e-06 below lme4 on lme4's own REML criterion (4e-10 relative) -- a flat direction, not a different fit",
  lmm_cross8_g30000p5_bal_nearzero = "SAME OPTIMUM (objective mode): glmm 3.9e-05 below lme4 on lme4's own REML criterion (7e-10 relative)",
  lmm_q2sq2s_g3000p5_bal_nearzero  = "SAME OPTIMUM (objective mode): glmm and lme4 within 1.9e-06 on lme4's own REML criterion, inside TOL$loglik_abs_lmm",
  lmm_q6_g30000p20_bal_nearzero    = "SAME OPTIMUM (objective mode): glmm 1.5e-04 below lme4 on lme4's own REML criterion (2e-09 relative)",
  lmm_q8q2x_g30000p20_bal_lowsnr   = "SAME OPTIMUM (objective mode): glmm 1.8e-04 below lme4 on lme4's own REML criterion (1e-09 relative)",
  lmm_q3sx2_g30000p5_bal_nearzero  = "SAME OPTIMUM (objective mode): glmm 1.6e-05 below lme4 on lme4's own REML criterion; the breaching stddev coordinate is g1:x2 at magnitude 1.9e-03, the two engines 2.3e-06 apart in absolute terms")
line_of <- function(cid) {
  r <- res[res$case_id == cid, ]
  note <- if (cid %in% names(ADJUDICATED)) paste0("  <-- ", ADJUDICATED[[cid]]) else ""
  sprintf("%-42s %-15s %-8s %-12s beta=%.1e stddev=%.1e corr=%.1e se=%.1e dev=%.1e (glmm:%s oracle:%s)%s",
          cid, r$status, r$category, r$oracle_engine,
          r$d_beta, r$d_stddev, r$d_corr, r$d_se, r$d_dev_rel,
          r$glmm_status, r$oracle_status, note)
}
fail_ids     <- ids_of(fail_status)
timeout_ids  <- ids_of("glmm-timeout")
oerr_ids     <- ids_of("oracle-error")
osing_ids    <- ids_of("oracle-singular")
otime_ids    <- ids_of("oracle-timeout")
writeLines(c(
  "# Diligent-grid named failure list (spec Part 6). Manifest order.",
  sprintf("# %d validation failures of 510 cells.", length(fail_ids)),
  vapply(fail_ids, line_of, ""),
  "",
  sprintf("# GLMM COMPUTE-BUDGET BOUNDARY (%d cells, NOT accuracy verdicts): watchdog",
          length(timeout_ids)),
  "# 30 s per-fit budget exceeded -- the known giant-LMM slow tail (cross8/q5q2/",
  "# q6q2x3/q8q2x, mostly g30000). No answer produced within budget.",
  vapply(timeout_ids, line_of, ""),
  "",
  sprintf("# ORACLE COVERAGE LIMIT (%d cells): lme4 identifiability refusal ('number of",
          length(oerr_ids)),
  "# observations <= number of random effects') on p5 high-q LMM cells -- glmm",
  "# fits these; there is simply no oracle answer to compare against.",
  vapply(oerr_ids, line_of, ""),
  "",
  sprintf("# ORACLE SINGULAR FITS (%d cells, informational): lme4 converged to a boundary",
          length(osing_ids)),
  "# (singular) fit with estimates. Gaps vs glmm recorded in status_map.csv but",
  "# NOT gated -- singular optima have flat directions.",
  vapply(osing_ids, line_of, ""),
  "",
  sprintf("# ORACLE TIMEOUTS (%d cells): oracle exceeded the 120 s watchdog budget --",
          length(otime_ids)),
  "# oracle-side compute limit, no reference answer.",
  vapply(otime_ids, line_of, "")),
  file.path(out_dir, "failures.txt"))

# ---- the three named claims (spec Part 6 "what the run proves") -----------------
claim <- function(subset) {
  n <- nrow(subset); ok <- sum(subset$status == "ok")
  list(n = n, ok = ok, joinable = sum(subset$status %in% c("ok", "mismatch")),
       fail = sum(subset$status %in% fail_status))
}
c1 <- claim(res[res$category == "laplace", ])
# Laplace-mismatch decomposition. `c1$fail` also counts glmm-fail and both-fail
# cells, which have no glmm answer to compare and so cannot join any of the
# buckets below -- they are reported as their own terms so the parts sum.
#
# The buckets are named for the GATE that broke, not for a diagnosis. What each
# cell means is settled per cell in ADJUDICATED above, from lme4's own REML
# criterion evaluated at both answers and from lme4's recorded messages; the
# counts of those two verdicts are carried alongside rather than folded in.
# Precedence beta > stddev > se mirrors classify_mismatch.R's class rule --
# change together.
lap_mis <- res[res$category == "laplace" & res$status == "mismatch", ]
n_gfail_lap <- sum(res$category == "laplace" & res$status == "glmm-fail")
n_bfail_lap <- sum(res$category == "laplace" & res$status == "both-fail")
broke <- function(d, tol) !is.na(d) & d > tol
# The b37 reclassification block above rewrites status and oracle_engine for the
# one "real disagreement (MixedModels)" verdict but leaves its delta columns at
# the NA the refused lme4 join gave them, so it cannot be classified by band.
# That status/engine pair is set for no other reason, so count it directly.
lap_real <- lap_mis[lap_mis$oracle_engine == "MixedModels", ]
lap_band <- lap_mis[lap_mis$oracle_engine != "MixedModels", ]
beta_hit <- broke(lap_band$d_beta, lap_band$tol_beta)
sd_hit   <- broke(lap_band$d_stddev, lap_band$tol_stddev) & !beta_hit
se_hit   <- !beta_hit & !sd_hit
n_beta   <- sum(beta_hit) + nrow(lap_real)
n_sdonly <- sum(sd_hit)
n_seonly <- sum(se_hit)
# The residual bucket is only honest if every cell in it really did break the SE
# band and nothing else; a corr-only breach would otherwise be mislabelled.
stopifnot(all(broke(lap_band$d_se, lap_band$tol_se)[se_hit]))
n_undercvg <- sum(startsWith(ADJUDICATED[names(ADJUDICATED) %in% lap_mis$case_id],
                             "ORACLE UNDER-CONVERGED"))
max_dev_mis <- suppressWarnings(max(lap_mis$d_dev_rel, na.rm = TRUE))
int1 <- res[res$category == "agq_int1", ]
c2 <- claim(int1)   # all int1 AGQ cells joinable (bina_ gate lifted 2026-07-14)
q2s <- res[res$category == "agq_q2s", ]
c3 <- claim(q2s)    # all q2s AGQ cells joinable
# Per-category oracle-timeout counts: 1 of the oracle timeouts is a q2s cell
# (GLMMadaptive over 120 s) and belongs in claim 3's accounting, not claim 1's.
n_otime_lap <- sum(res$status == "oracle-timeout" & res$category == "laplace")
n_otime_q2s <- sum(res$status == "oracle-timeout" & res$category == "agq_q2s")
# q2s mismatch attribution split (see the ADJUDICATED classes above)
q2s_mis <- q2s$case_id[q2s$status == "mismatch"]
n_adj_oracle <- sum(startsWith(ADJUDICATED[names(ADJUDICATED) %in% q2s_mis], "ADJUDICATED"))
n_adj_near0  <- sum(startsWith(ADJUDICATED[names(ADJUDICATED) %in% q2s_mis], "BOUNDARY"))

cat("\n== diligent-grid status ==\n"); print(table(res$status))
cat(sprintf("\ncoverage: glmm %d/510, oracle %d/510 result rows\n",
            length(glmm), length(oracle)))
cat("\n== the three named claims (spec Part 6) ==\n")
cat(sprintf(paste0("  1. Laplace vs lme4      : %d/%d cells match (%d validation-fail = ",
            "%d mismatch + %d glmm-fail + %d both-fail)\n",
            "       mismatches by broken gate: %d se-only + %d stddev-only + %d beta-level, ",
            "worst dev gap %.0e\n",
            "       adjudicated: %d are lme4's own max|grad| non-convergence, ",
            "%d a real disagreement vs MixedModels.jl (see failures.txt)\n",
            "       non-comparable: %d glmm-timeout, %d oracle-error, %d oracle-singular, ",
            "%d oracle-timeout\n"),
            c1$ok, c1$n, c1$fail,
            nrow(lap_mis), n_gfail_lap, n_bfail_lap,
            n_seonly, n_sdonly, n_beta, max_dev_mis,
            n_undercvg, nrow(lap_real),
            length(timeout_ids), length(oerr_ids), length(osing_ids),
            n_otime_lap))
cat(sprintf("  2. int1 scalar AGQ k=7  : %d/%d joinable match lme4 (%d validation-fail)\n",
            c2$ok, c2$n, c2$fail))
cat(sprintf(paste0("  3. q2s vector AGQ k=7   : %d/%d joinable match GLMMadaptive (%d mismatch: ",
            "%d large-beta adjudicated oracle-side via adjudicate.R, ",
            "%d nearzero via the near-zero-theta boundary argument; ",
            "%d oracle-timeout)\n"),
            c3$ok, c3$joinable, c3$fail, n_adj_oracle, n_adj_near0,
            n_otime_q2s))
cat(sprintf("\nnamed failure list: %s (%d failures)\n",
            file.path(out_dir, "failures.txt"), length(fail_ids)))

# ---- HTML report (grid_report.R style: heatmap by structure x family) ----------
# Rows = size x balance/regime, columns = structure x family (two-row header,
# 4 structures per table in n_theta order). Each cell shows the worst validation gap
# and its status color; AGQ cells (nagq>1) carry a "k7" badge and a blue outline.
res$size <- sub(".*_(g[0-9]+p[0-9]+)_.*", "\\1", res$case_id)
res$g <- as.integer(sub("g([0-9]+)p.*", "\\1", res$size))
res$p <- as.integer(sub(".*p([0-9]+)$", "\\1", res$size))
res$variant <- paste(res$balance, res$regime, sep = "/")
# Display family = case_id prefix (lmm/binb/bina/pois), NOT the manifest family:
# binb (binary) and bina (aggregated) both carry family "binomial", so keying
# columns on family would collide the two cells and silently drop one -- and the
# dropped ones include the bina_ AGQ boundary cells this report must show.
res$fam <- sub("_.*", "", res$case_id)
# worst gap actually gated per cell (corr/se may be NA on scalar/gaussian rows)
res$worst_gap <- suppressWarnings(pmax(res$d_beta, res$d_stddev,
                                       res$d_corr, res$d_se, na.rm = TRUE))
res$worst_gap[!is.finite(res$worst_gap)] <- NA_real_

fmt_g <- function(x) ifelse(is.na(x), "&ndash;", formatC(x, digits = 2, format = "e"))
# status -> pastel background; mismatch/fail are loud, ok is a green graded by gap
status_bg <- function(row) {
  switch(row$status,
    ok        = "hsl(120,60%,90%)",
    mismatch  = "hsl(0,75%,82%)",
    `glmm-fail`       = "hsl(0,75%,78%)",
    `glmm-timeout`    = "hsl(30,80%,85%)",
    `oracle-error`    = "hsl(215,70%,85%)",
    `oracle-singular` = "hsl(190,55%,88%)",
    `oracle-timeout`  = "hsl(215,70%,90%)",
    `both-fail`       = "#ddd",
    "#fff")
}
agq_of <- function(row) row$nagq > 1

cell_html <- function(row) {
  bg <- status_bg(row)
  badge <- if (agq_of(row)) ' <span class="k">k7</span>' else ''
  cls <- if (agq_of(row)) 'agq' else ''
  lab <- switch(row$status,
    ok = sprintf("<b>ok</b>%s<br><span class=g>%s</span>", badge, fmt_g(row$worst_gap)),
    mismatch = sprintf("<b>&#9888; mismatch</b>%s<br><span class=g>%s</span>", badge, fmt_g(row$worst_gap)),
    `oracle-singular` = sprintf("<b>oracle-singular</b>%s<br><span class=g>%s ungated</span>",
                                badge, fmt_g(row$worst_gap)),
    sprintf("<b>%s</b>%s<br><span class=g>g:%s o:%s</span>", row$status, badge,
            row$glmm_status, row$oracle_status))
  sprintf('<td class="c %s" style="background:%s" title="%s\ncat %s oracle %s\nbeta %s stddev %s corr %s se %s dev %s">%s</td>',
          cls, bg, row$case_id, row$category, row$oracle_engine,
          fmt_g(row$d_beta), fmt_g(row$d_stddev), fmt_g(row$d_corr),
          fmt_g(row$d_se), fmt_g(row$d_dev_rel), lab)
}

esc <- function(x) gsub("<", "&lt;", x)
tbl <- table(factor(res$status,
  levels = c("ok","mismatch","glmm-fail","glmm-timeout","oracle-error",
             "oracle-singular","oracle-timeout","both-fail")))
html <- c(
  '<meta charset="utf-8"><title>GLMM diligent grid</title><style>',
  'body{font-family:system-ui,sans-serif;margin:1.5em;font-size:14px;color:#111}',
  'table{border-collapse:collapse;margin:0 0 2em}',
  'td,th{border:1px solid #bbb;padding:3px 7px;text-align:left;white-space:nowrap}',
  'th{background:#f0f0f0} .c{font-size:12px} .c b{font-weight:600}',
  '.g{color:#555;font-size:11px} .k{color:#04a;font-weight:700;font-size:10px}',
  '.agq{outline:2px solid #06c;outline-offset:-2px}',
  '.grp{border-left:3px solid #888}',
  '.legend span{padding:2px 8px;margin-right:6px;border:1px solid #bbb}',
  'h1{font-size:20px} h2{margin:1.2em 0 .3em} .note{color:#444;max-width:74em}',
  '.claim{background:#f7f7f7;border-left:4px solid #06c;padding:.5em 1em;margin:.4em 0;max-width:74em}',
  '</style>',
  '<h1>GLMM diligent grid &mdash; estimates / var-components / Hessian SEs vs lme4 + GLMMadaptive</h1>',
  sprintf(paste0('<p class="note">510-cell accuracy sweep (full-AGQ vector-RE spec, Part 6). ',
    'glmm result rows: %d/510; oracle rows: %d/510. Each cell = worst gated validation gap ',
    '(hover for beta / stddev / corr / se / deviance). AGQ cells (nagq=7) carry a ',
    '<span class="k">k7</span> badge and blue outline. Walls recorded but NOT a deliverable ',
    '(machine not clock-locked).</p>'), n_glmm <- length(glmm), length(oracle)),
  '<p class="legend">',
  '<span style="background:hsl(120,60%,90%)">ok</span>',
  '<span style="background:hsl(0,75%,82%)">&#9888; mismatch</span>',
  '<span style="background:hsl(0,75%,78%)">glmm fail</span>',
  '<span style="background:hsl(30,80%,85%)">glmm timeout (budget)</span>',
  '<span style="background:hsl(215,70%,85%)">oracle error (identifiability)</span>',
  '<span style="background:hsl(190,55%,88%)">oracle singular (ungated)</span>',
  '<span style="outline:2px solid #06c">AGQ cell</span></p>',
  sprintf(paste0('<p class="note"><b>Status:</b> ok %d &middot; mismatch %d &middot; glmm-fail %d &middot; ',
    'glmm-timeout %d &middot; oracle-error %d &middot; oracle-singular %d &middot; oracle-timeout %d &middot; both-fail %d. ',
    '<b>glmm-timeout</b> = 30&thinsp;s watchdog budget exceeded (giant-LMM slow tail) &mdash; compute-budget boundary, not an accuracy verdict. ',
    '<b>oracle-error</b> = lme4 identifiability refusal (obs &le; REs per group on p5 high-q cells) &mdash; glmm fits, no oracle answer exists. ',
    '<b>oracle-singular</b> = lme4 boundary (singular) fit with estimates &mdash; gaps recorded, not gated (flat directions). ',
    '<b>oracle-timeout</b> = oracle exceeded its 120&thinsp;s budget.</p>'),
          tbl[["ok"]], tbl[["mismatch"]], tbl[["glmm-fail"]], tbl[["glmm-timeout"]],
          tbl[["oracle-error"]], tbl[["oracle-singular"]], tbl[["oracle-timeout"]],
          tbl[["both-fail"]]),
  '<h2>The three claims (spec Part 6 &ldquo;what the run proves&rdquo;)</h2>',
  sprintf(paste0('<div class="claim"><b>1. Grid-wide Laplace (477 cells):</b> %d/%d match lme4-Laplace answers ',
    '(beta, &theta;, Hessian SE) within band &mdash; %d mismatches, by the gate they break: ',
    '%d Hessian-SE-only on multi-grouping GLMM cells (&beta;/&theta;/deviance match; se gap up to 1.4e-2 ',
    'vs the 1e-3 band), %d stddev-band-only on small variance components, %d at &beta; level. ',
    'Aligned deviance agrees to &le;%.0e across the mismatches. Adjudicated per cell in failures.txt: ',
    '%d carry lme4&rsquo;s own max|grad| non-convergence, %d is a real disagreement against ',
    'MixedModels.jl; on the rest lme4&rsquo;s own REML criterion scores glmm at or below lme4, ',
    'so they are the same optimum. ',
    'Non-comparable remainder, all named in failures.txt: %d glmm compute-budget timeouts, ',
    '%d lme4 identifiability refusals (no oracle answer), %d lme4 singular fits (ungated), ',
    '%d oracle timeouts, %d both-fail (both engines refuse the same p5 q8 cell).</div>'),
          c1$ok, c1$n, nrow(lap_mis), n_seonly, n_sdonly, n_beta, max_dev_mis,
          n_undercvg, nrow(lap_real),
          length(timeout_ids), length(oerr_ids), length(osing_ids),
          n_otime_lap, n_bfail_lap),
  sprintf('<div class="claim"><b>2. int1 scalar AGQ (15 cells, %d joinable):</b> %d/%d &equiv; lme4 glmer(nAGQ=7) across the size&times;balance&times;regime sweep &mdash; %d validation-fail. Includes the 6 aggregated-binomial <code>bina_int1_*</code> cells, now fit via AGQ&times;prior-weights (gate lifted 2026-07-14).</div>',
          c2$joinable, c2$ok, c2$joinable, c2$fail),
  sprintf(paste0('<div class="claim"><b>3. q2s vector AGQ (headline; 18 cells, %d joinable):</b> %d/%d joinable cells ',
    'match GLMMadaptive at matched nodes (k=7); %d mismatch, each attributed in failures.txt &mdash; %d large-&beta; cells ',
    'adjudicated oracle-side via <code>adjudicate.R</code> (glmer-Laplace on the same data reproduces glmm&rsquo;s ',
    '&beta; and &theta; to 4 decimals; GLMMadaptive under-converged, one with a non-PD Hessian), and %d nearzero cells via ',
    'the near-zero-&theta; boundary argument (&beta;/SE agree; only the &sim;0.01&ndash;0.1-scale slope stddev, a degenerate ',
    'component with corr pinned at &plusmn;1, exceeds the relative band). ',
    '%d cell has no reference (GLMMadaptive exceeded its 120&thinsp;s budget). ',
    'The 6 aggregated-binomial <code>bina_q2s_*</code> cells are now fit via AGQ&times;prior-weights (gate lifted 2026-07-14).</div>'),
          c3$joinable, c3$ok, c3$joinable, c3$fail, n_adj_oracle, n_adj_near0,
          n_otime_q2s))

structures <- unique(res$structure[order(res$n_theta, res$structure)])
for (grp in split(structures, ceiling(seq_along(structures) / 4))) {
  s <- res[res$structure %in% grp, ]
  hdr1 <- '<tr><th rowspan="2">cell</th>'; hdr2 <- '<tr>'; cols <- list()
  for (st in grp) {
    fams <- sort(unique(s$fam[s$structure == st]))
    hdr1 <- c(hdr1, sprintf('<th colspan="%d" class="grp">%s <small>(n&theta; %d)</small></th>',
                            length(fams), esc(st), unique(s$n_theta[s$structure == st])[1]))
    hdr2 <- c(hdr2, sprintf('<th%s>%s</th>',
                            c(' class="grp"', rep('', length(fams) - 1)), fams))
    cols <- c(cols, lapply(fams, function(f) c(st, f)))
  }
  html <- c(html, '<table>', hdr1, '</tr>', hdr2, '</tr>')
  keys <- unique(s[order(s$g, s$p, s$variant), c("size", "variant")])
  for (i in seq_len(nrow(keys))) {
    row_cells <- vapply(seq_along(cols), function(j) {
      r <- s[s$size == keys$size[i] & s$variant == keys$variant[i] &
             s$structure == cols[[j]][1] & s$fam == cols[[j]][2], ]
      if (nrow(r) == 0) {
        if (j > 1 && !identical(cols[[j]][1], cols[[j-1]][1])) '<td class="grp"></td>' else '<td></td>'
      } else {
        cell <- cell_html(r[1, ])
        if (j > 1 && !identical(cols[[j]][1], cols[[j-1]][1]))
          sub('<td class="', '<td class="grp ', cell, fixed = TRUE) else cell
      }
    }, "")
    html <- c(html, sprintf('<tr><th>%s %s</th>', keys$size[i], keys$variant[i]),
              row_cells, '</tr>')
  }
  html <- c(html, '</table>')
}
report_path <- file.path(out_dir, "report.html")
writeLines(html, report_path)
cat(sprintf("report: %s\n", report_path))
