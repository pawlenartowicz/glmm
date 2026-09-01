#!/usr/bin/env Rscript
# Study-A analysis over the campaign JSONLs. Usage:
#   Rscript analyze.R <glmm.jsonl> <mixedmodels.jsonl> [lme4.jsonl]
# Correctness gate: compare.R's beta tolerance (tol.R) against the MixedModels
# reference (lme4 where present breaks glmm-vs-MM ties). Deviance scales are
# aligned to lme4's -2*logLik convention: LMM glmm + df*(1+log(2pi)) (validated
# in-crate, Task 2); GLMM glmm - 2*logL_saturated(y) (validated below against
# any cell where both engines converged -- the offset must be constant per cell
# family; hard-stop if not).
suppressMessages({ library(jsonlite) })

args <- commandArgs(TRUE)
if (length(args) < 2) stop("usage: analyze.R glmm.jsonl mm.jsonl [lme4.jsonl]")
suite_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(suite_dir, "..", "..", "tol.R"))
# Durable summaries (committed) live under reports/, not results/ (gitignored).
out_dir <- file.path(suite_dir, "reports")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# read_jsonl lives in tol.R (sourced above), torn-line tolerant.
glmm <- read_jsonl(args[1]); mm <- read_jsonl(args[2])

# JIT guard: fit.jl warm-up-fits each cell and records compile_seconds
# for the timed fit as proof it ran hot. Any nonzero (>1 ms) compile time in a
# timed fit means JIT leaked into wall_seconds — HARD STOP, the walls are not
# quotable. Records without the field (watchdog timeout stubs, engine-fail
# stubs, pre-2026-07-11 passes) are skipped, but a whole pass without the
# field gets a loud warning: its walls predate the warm-up fix.
mm_compile <- vapply(mm, function(r)
  if (is.null(r$compile_seconds)) NA_real_ else r$compile_seconds, 0)
if (any(mm_compile > 1e-3, na.rm = TRUE))
  stop("JIT leaked into ", sum(mm_compile > 1e-3, na.rm = TRUE),
       " MixedModels timed fits (compile_seconds > 1 ms): ",
       paste(head(names(mm)[which(mm_compile > 1e-3)], 5), collapse = ", "),
       " -- walls from this pass are invalid")
if (all(is.na(mm_compile)))
  warning("MixedModels pass has no compile_seconds field -- pre-warm-up data, ",
          "walls include JIT and must not be quoted", call. = FALSE)
anchor <- if (length(args) >= 3 && file.exists(args[3])) read_jsonl(args[3]) else list()
cells <- manifest_cells(file.path(suite_dir, "manifest.json"))

# ---- deviance alignment to MixedModels' objective() scale ----------------------
# glmm's deviance conventions, pinned empirically on grid cells: poisson,
# binary binomial AND aggregated binomial are ALL on MixedModels'
# unit-deviance scale -- identity, no constant. Aggregated binomial was
# re-pinned 2026-07-11: the 2026-07-09 pin had it on the choose-free scale
# (correction = twice the aggregated saturated kernel), but the prior-weights
# landing of 2026-07-10 moved the weighted-binomial deviance onto the
# unit-deviance scale; on the 2026-07-11 Study-A run identity agrees with
# MixedModels to 1.7e-9 worst-case across all 36 agreed aggregated cells while
# the old kernel correction was off by ~1e1 relative on every one (the
# align-check hard stop below is what caught it). Gaussian: REML offset
# df*(1+log(2pi)) validated in-crate against lme4 (Task 2).
glmm_dev_aligned <- function(cell, dev) {
  # timeout/engine-fail records carry deviance:null -> NULL here; is.finite(NULL)
  # is logical(0) and would abort the whole analysis
  if (is.null(dev) || !is.finite(dev)) return(NA_real_)
  if (cell$family == "gaussian") {
    p <- cell$n_x + 1          # intercept + covariates (treatment-free designs)
    df_reml <- cell$n_obs - p
    dev + df_reml * (1 + log(2 * pi))
  } else dev                   # all GLMM families: identity
}

# validation of the GLMM alignment: on converged glmm+MM pairs the aligned gap
# must be small (same surface, two optimizers -- 1e-2 relative band, looser
# than tier-0's 1e-3 because this is a false-stop guard, not a validation gate).
# A systematic family-wide offset means the alignment above is wrong:
# HARD STOP with the measured offsets.
align_check <- new.env(); align_check$bad <- 0L

rows <- list(); profiles <- list()
for (cid in names(cells)) {
  cell <- cells[[cid]]
  g <- glmm[[cid]]; m <- mm[[cid]]
  if (is.null(g) || is.null(m)) next
  d_beta <- if (g$status == "ok" && m$status == "ok")
    rel_max(g$beta, m$beta) else NA_real_
  agree <- !is.na(d_beta) && d_beta <= TOL$beta_rel
  status <- if (g$status != "ok") g$status
            else if (m$status != "ok") "ok"          # glmm fine, rival failed
            else if (agree) "ok" else "mismatch"
  dg <- glmm_dev_aligned(cell, g$deviance)
  dm <- if (is.null(m$deviance)) NA_real_ else m$deviance
  if (cell$family != "gaussian" && isTRUE(agree) &&
      is.finite(dg) && is.finite(dm) && abs(dg - dm) > 1e-2 * (1 + abs(dm)))
    align_check$bad <- align_check$bad + 1L
  rows[[length(rows) + 1L]] <- data.frame(
    case_id = cid, family = cell$family, structure = cell$structure,
    n_theta = cell$n_theta, n_obs = cell$n_obs, balance = cell$balance,
    regime = cell$regime, status = status,
    glmm_status = g$status, mm_status = m$status,
    n_eval_glmm = g$n_eval, n_eval_mm = m$n_eval,
    eval_ratio = ifelse(g$status == "ok" & m$status == "ok",
                        g$n_eval / m$n_eval, NA_real_),
    dev_glmm = dg, dev_mm = dm,
    wall_glmm = g$wall_seconds, wall_mm = m$wall_seconds,
    wall_source = if (is.null(g$wall_source)) NA_character_ else g$wall_source,
    per_eval_glmm = g$wall_seconds / max(g$n_eval, 1),
    per_eval_mm = m$wall_seconds / max(m$n_eval, 1))
}
res <- do.call(rbind, rows)
if (align_check$bad > nrow(res) * 0.02)
  stop("GLMM deviance alignment failed on ", align_check$bad,
       " agreed cells -- saturated-constant derivation is wrong; investigate before trusting profiles")

write.csv(res, file.path(out_dir, "status_map.csv"), row.names = FALSE)

# ---- eval-ratio quantiles ------------------------------------------------------
qtab <- function(split_col) {
  ok <- res[!is.na(res$eval_ratio), ]
  do.call(rbind, lapply(split(ok, ok[[split_col]]), function(s) data.frame(
    group = s[[split_col]][1], n = nrow(s),
    median = median(s$eval_ratio), p90 = quantile(s$eval_ratio, 0.9))))
}
res$n_theta_bin <- cut(res$n_theta, c(0, 2, 5, 11, 20, 40),
                       labels = c("1-2", "3-5", "6-11", "12-20", "21-40"))
ratio_out <- rbind(cbind(axis = "n_theta_bin", qtab("n_theta_bin")),
                   cbind(axis = "family",      qtab("family")),
                   cbind(axis = "structure",   qtab("structure")),
                   cbind(axis = "regime",      qtab("regime")))
write.csv(ratio_out, file.path(out_dir, "eval_ratio.csv"), row.names = FALSE)

# ---- More-Wild data profiles ---------------------------------------------------
# solved(tau): engine deviance <= best + tau * (1 + |best|); budget axis in
# k*(n_theta+1) evals (More & Wild 2009, SIAM J. Optim. 20(1)); k runs to the
# pre-registered eval cap (k = 500).
prof <- list()
for (tau in c(1e-3, 1e-5)) {
  ok <- res[is.finite(res$dev_glmm) & is.finite(res$dev_mm), ]
  best <- pmin(ok$dev_glmm, ok$dev_mm)
  for (k in 1:500) {
    budget <- k * (ok$n_theta + 1)
    prof[[length(prof) + 1L]] <- data.frame(
      tau = tau, k = k,
      glmm = mean(ok$dev_glmm <= best + tau * (1 + abs(best)) & ok$n_eval_glmm <= budget),
      mixedmodels = mean(ok$dev_mm <= best + tau * (1 + abs(best)) & ok$n_eval_mm <= budget))
  }
}
write.csv(do.call(rbind, prof), file.path(out_dir, "data_profiles.csv"), row.names = FALSE)

# ---- lme4 to-do: mismatches + rival engine-fails + seeded 20-cell audit --------
# Manifest order (not case_id sort or sample order) -- the watchdog's lme4
# timeout-attribution keys off this file's line order.
# Rival engine-fail cells (glmm ok, MM crashes -- 2026-07-11: all nest2s
# PosDefException) get status "ok" above with glmm's answer unverified; anchor
# them all so "converges where MM can't" also means "to the right answer".
# MM *timeouts* are excluded: expected-heavy 30000-row cells, lme4 would
# mostly burn its 120 s budget the same way.
set.seed(20260709)
audit <- sample(res$case_id[res$status == "ok"], min(20, sum(res$status == "ok")))
rival_fail <- res$case_id[res$glmm_status == "ok" & res$mm_status == "engine-fail"]
todo_set <- union(union(res$case_id[res$status == "mismatch"], audit), rival_fail)
todo <- Filter(function(cid) cid %in% todo_set, names(cells))
writeLines(todo, file.path(out_dir, "lme4_todo.txt"))

# ---- console summary -----------------------------------------------------------
cat("\n== status map ==\n"); print(table(res$status))
cat("\n== glmm/MM eval-ratio quantiles ==\n"); print(ratio_out, row.names = FALSE)
cat(sprintf("\nlme4 to-do: %d cells (%d mismatch + %d audit) -> %s\n",
            length(todo), sum(res$status == "mismatch"), length(audit),
            file.path(out_dir, "lme4_todo.txt")))
cat("\nNOTE: wall/per-eval columns are meaningful ONLY for clock-locked passes",
    "(check results/run_meta_*.json no_turbo==1).\n")
