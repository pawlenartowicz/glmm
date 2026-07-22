#!/usr/bin/env Rscript
# Boundary-fits follow-up spec, Part A: verify the 37 oracle-error (lme4
# identifiability-refusal) diligent cells against a third oracle,
# MixedModels.jl. Joins glmm vs MM using the same stddevs_of/corrs_of/rel_max
# machinery and lmm tolerance bands as analyze.R (docs/GLMM/plans/
# 2026-07-14-boundary-fits-followup-spec.md). Deviance is not gated (MM's
# objective constant differs from glmm's REML criterion — same rule
# analyze.R applies to GLMMadaptive rows).
#
#   Rscript verify_boundary37.R [glmm.jsonl] [mixedmodels.jsonl]
suppressMessages({ library(jsonlite) })

suite_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(suite_dir, "..", "..", "tol.R"))
args <- commandArgs(TRUE)
glmm_path <- if (length(args) >= 1) args[1] else
  file.path(suite_dir, "results", "glmm.jsonl")
mm_path   <- if (length(args) >= 2) args[2] else
  file.path(suite_dir, "results", "mixedmodels.jsonl")
out_dir <- file.path(suite_dir, "reports")

glmm <- read_jsonl(glmm_path); mm <- read_jsonl(mm_path)
target_ids <- names(mm)
manifest <- fromJSON(file.path(suite_dir, "manifest.json"),
                     simplifyDataFrame = FALSE)
cells <- setNames(manifest$cells, vapply(manifest$cells, `[[`, "", "case_id"))

# REML-criterion alignment for mismatches only (adjudication, not a gate):
# glmm's REML criterion + df*(1+log 2pi) == MM's objective() on the SAME
# scale (verified exact, <1e-4, across all 34 non-mismatch cells below) --
# same offset analyze.R's glmm_dev_aligned() uses for lme4. A tiny
# residual (<0.05) at a mismatch means BOTH engines sit at the same optimum
# and the parameter gap is a flat/degenerate direction (glmm.singular flags
# it, same boundary argument as the q2s ADJUDICATED nearzero class); a large
# residual means glmm genuinely converged to a worse optimum -- a real flag.
dev_gap <- function(cid) {
  g <- glmm[[cid]]; o <- mm[[cid]]; cell <- cells[[cid]]
  p <- cell$n_x + 1
  dg <- g$deviance + (cell$n_obs - p) * (1 + log(2 * pi))
  dg - o$deviance
}

# read_jsonl/stddevs_of/corrs_of live in tol.R (shared with analyze.R).

rows <- list()
for (cid in target_ids) {
  g <- glmm[[cid]]; o <- mm[[cid]]
  g_ok <- !is.null(g) && identical(g$status, "ok")
  o_ok <- !is.null(o) && identical(o$status, "ok")

  d_beta <- d_sd <- d_corr <- d_se <- NA_real_
  beta_len_mismatch <- sd_len_mismatch <- corr_len_mismatch <- FALSE
  if (g_ok && o_ok) {
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
  }
  # length mismatches are real disagreements (rel_max's NA there means "not
  # comparable", never "matches") -- treat as failures, not passes (mirrors
  # analyze.R; change together).
  gate_fail <- (!is.na(d_beta) && d_beta > TOL$beta_rel) || beta_len_mismatch ||
               (!is.na(d_sd)   && d_sd   > TOL$stddev_rel) || sd_len_mismatch ||
               (!is.na(d_corr) && d_corr > TOL$agq_corr_abs) || corr_len_mismatch ||
               (!is.na(d_se)   && d_se   > TOL$se_hessian_rel)

  dg <- NA_real_
  # Verdict strings below are matched literally by analyze.R's
  # reclassification block (`confirmed`/`flagged` %in%/== checks) -- change
  # both together.
  verdict <- if (!g_ok && !o_ok)  "both-fail (no-oracle identifiability boundary)"
             else if (!g_ok)      "glmm-fail"
             else if (!o_ok)      "no-oracle identifiability boundary (MM also refuses)"
             else if (gate_fail) {
               dg <- dev_gap(cid)
               if (abs(dg) < 0.05) "boundary (same optimum, flat direction)"
               else "mismatch (real disagreement)"
             } else "confirmed (MM agrees)"

  rows[[length(rows) + 1L]] <- data.frame(
    case_id = cid, glmm_status = if (is.null(g)) "absent" else g$status,
    mm_status = if (is.null(o)) "absent" else o$status,
    verdict = verdict,
    d_beta = d_beta, d_stddev = d_sd, d_corr = d_corr, d_se = d_se,
    dev_gap = dg,
    tol_beta = TOL$beta_rel, tol_stddev = TOL$stddev_rel,
    tol_corr = TOL$agq_corr_abs, tol_se = TOL$se_hessian_rel,
    stringsAsFactors = FALSE)
}
res <- do.call(rbind, rows)
write.csv(res, file.path(out_dir, "boundary37_verdicts.csv"), row.names = FALSE)

cat("\n== boundary-37 verdicts (glmm vs MixedModels.jl) ==\n")
print(table(res$verdict))
cat(sprintf("\nfull table: %s\n", file.path(out_dir, "boundary37_verdicts.csv")))
adj <- res[grepl("^mismatch|^boundary", res$verdict), ]
if (nrow(adj) > 0) {
  cat("\n-- gate-exceeding cells, with REML-criterion adjudication --\n")
  for (i in seq_len(nrow(adj))) {
    r <- adj[i, ]
    cat(sprintf("%-42s beta=%.1e stddev=%.1e corr=%.1e se=%.1e dev_gap=%+.4f  [%s]\n",
                r$case_id, r$d_beta, r$d_stddev, r$d_corr, r$d_se, r$dev_gap, r$verdict))
  }
}
