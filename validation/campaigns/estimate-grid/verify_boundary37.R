#!/usr/bin/env Rscript
# Verifies the 37 oracle-error (lme4 identifiability-refusal) diligent cells
# against a third oracle, MixedModels.jl. Joins glmm vs MM using the same
# stddevs_of/corrs_of/rel_max machinery and lmm tolerance bands as analyze.R.
# Deviance is not gated (MM's objective constant differs from glmm's REML
# criterion — same rule analyze.R applies to GLMMadaptive rows).
#
#   Rscript verify_boundary37.R [glmm.jsonl] [mixedmodels.jsonl]
#
# ── objective mode ────────────────────────────────────────────────────────
# Set VERIFY_OBJECTIVE to a file of case_ids and the script ALSO runs the
# whose-optimum-is-lower protocol on those cells, writing
# reports/objective_verdicts.csv. That protocol answers a different question
# from the join above -- not "do two engines' numbers agree" but "which of two
# candidate answers scores better on one engine's own objective" -- and the
# second question is the only one that decides a disagreement. See the
# `-- objective mode --` block at the bottom.
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

# ── objective mode ─────────────────────────────────────────────────────────
# The join above compares two engines' ANSWERS. When they disagree that cannot
# say which is right, because both are just numbers. This block asks the
# question that can: it takes lme4's own REML criterion as the referee and
# evaluates it at BOTH candidate answers. Whichever scores lower is the better
# fit of the model lme4 itself is optimizing -- a verdict, not a resemblance.
#
# HOW A CANDIDATE IS SCORED WITHOUT ITS RESIDUAL SD. lmer(devFunOnly = TRUE) is
# a function of theta, the covariance factor RELATIVE to the residual SD, and
# neither engine records that SD -- only the absolute variance components. So a
# candidate fixes theta's DIRECTION but not its scale: every recorded answer
# names a ray {theta_shape / s : s > 0} through theta-space rather than a point.
# The criterion is minimized along that ray. The minimum is a lower bound on the
# candidate's own score, which is all a "which is lower" verdict needs, and it
# costs one 1-D optimization instead of a refit.
#
# THE SELF-CHECK THAT MAKES THIS TRUSTWORTHY. lme4's OWN recorded answer runs
# through the identical path. Its ray passes through its own optimum, so the
# minimum along it must reproduce the deviance lme4 recorded. `lme4_roundtrip`
# is that residual: if it is not ~0, the theta reconstruction is wrong and every
# other number in the row is meaningless. Read it before reading the verdict.
#
# GAUSSIAN CELLS ONLY. glmer's devfun takes (theta, beta) jointly and its
# profiling story is different; nothing here applies to it unchanged.
objective_file <- Sys.getenv("VERIFY_OBJECTIVE", "")
if (nzchar(objective_file)) {
  suppressMessages(library(lme4))
  lme4_path <- if (length(args) >= 3) args[3] else
    file.path(suite_dir, "results", "lme4.jsonl")
  lme4_rec <- read_jsonl(lme4_path)
  obj_ids <- readLines(objective_file); obj_ids <- obj_ids[nzchar(obj_ids)]

  # Lower-triangular Cholesky factor that tolerates a SEMI-definite Sigma.
  # base::chol() errors on one, and a boundary fit -- the whole population this
  # block exists for -- is exactly where a variance component sits at zero and
  # Sigma loses rank. Cholesky-Banachiewicz with the pivot clamped at zero
  # returns the factor lme4's own theta carries there.
  chol_lower_psd <- function(S) {
    n <- nrow(S); L <- matrix(0, n, n)
    for (j in seq_len(n)) {
      d <- S[j, j] - sum(L[j, seq_len(j - 1)]^2)
      L[j, j] <- if (d > 0) sqrt(d) else 0
      if (j < n) for (i in (j + 1):n) {
        L[i, j] <- if (L[j, j] > 0)
          (S[i, j] - sum(L[i, seq_len(j - 1)] * L[j, seq_len(j - 1)])) / L[j, j] else 0
      }
    }
    L
  }

  # A record's varcomp -> theta shape, in the term order lme4 itself uses.
  # `cnms` comes off the fitted model rather than the manifest formula because
  # lme4 reorders random-effect terms internally (by number of levels); building
  # theta in formula order would silently score a different model.
  theta_shape_of <- function(rec, cnms) {
    vc <- rec$varcomp
    if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
    unlist(lapply(names(cnms), function(gn) {
      g <- Filter(function(e) identical(e$group, gn), vc)[[1]]
      terms <- as.character(unlist(g$terms))
      sd <- as.numeric(unlist(g$stddev))
      cr <- g$corr; if (is.list(cr)) cr <- cr[[1]]
      cr <- matrix(as.numeric(as.matrix(cr)), length(sd), length(sd))
      # lme4 writes NA correlations against a zero-SD row; they multiply a zero
      # variance either way, so zero is the only finite value that changes nothing.
      cr[!is.finite(cr)] <- 0; diag(cr) <- 1
      k <- match(cnms[[gn]], terms)
      sd <- sd[k]; cr <- cr[k, k, drop = FALSE]
      S <- diag(sd, nrow = length(sd)) %*% cr %*% diag(sd, nrow = length(sd))
      L <- chol_lower_psd(S)
      L[lower.tri(L, diag = TRUE)]   # column-major lower triangle == lme4's theta
    }), use.names = FALSE)
  }

  rows <- list()
  for (cid in obj_ids) {
    cell <- cells[[cid]]
    if (!identical(cell$family, "gaussian")) next
    df <- read.csv(file.path(suite_dir, "..", "speed-grid", "data",
                             paste0(cid, ".csv")))
    for (f in unlist(cell$factors)) df[[f]] <- factor(df[[f]])
    devfun <- lmer(as.formula(cell$r_formula), data = df, REML = isTRUE(cell$reml),
                   devFunOnly = TRUE)
    m_free <- lmer(as.formula(cell$r_formula), data = df, REML = isTRUE(cell$reml),
                   control = lmerControl(optCtrl = list(maxeval = cell$max_fun)))
    cnms <- getME(m_free, "cnms")

    # Best score along a candidate's ray. The bracket spans four decades either
    # side of 1; a residual SD outside that on a standardized simulation grid
    # would be a broken fit, not a scale this needs to reach.
    ray_min <- function(shape) {
      f <- function(ls) devfun(shape / exp(ls))
      o <- optimize(f, interval = c(-9, 9), tol = 1e-10)
      c(obj = o$objective, s = exp(o$minimum))
    }
    g_ray <- ray_min(theta_shape_of(glmm[[cid]], cnms))
    o_ray <- ray_min(theta_shape_of(lme4_rec[[cid]], cnms))

    # lme4 restarted FROM glmm's answer. If lme4's own optimizer, handed glmm's
    # point, walks back to lme4's recorded answer, lme4 prefers it; if it stays,
    # lme4's recorded answer was a stopping point rather than its optimum.
    seeded <- tryCatch({
      m <- lmer(as.formula(cell$r_formula), data = df, REML = isTRUE(cell$reml),
                start = list(theta = unname(g_ray["s"] ^ -1 *
                                            theta_shape_of(glmm[[cid]], cnms))),
                control = lmerControl(optCtrl = list(maxeval = cell$max_fun)))
      as.numeric(-2 * logLik(m))
    }, error = function(e) NA_real_)

    tol_obj <- TOL$loglik_abs_lmm
    verdict <- if (abs(g_ray["obj"] - o_ray["obj"]) <= tol_obj)
                 "same optimum"
               else if (g_ray["obj"] < o_ray["obj"])
                 "glmm better on lme4's own criterion"
               else "lme4 better (glmm at a worse optimum)"

    rows[[length(rows) + 1L]] <- data.frame(
      case_id = cid,
      obj_at_glmm = unname(g_ray["obj"]), obj_at_lme4 = unname(o_ray["obj"]),
      obj_gap = unname(g_ray["obj"] - o_ray["obj"]),
      # The band is ABSOLUTE (TOL$loglik_abs_lmm), which is the right convention
      # for a criterion but reads oddly on a 5000-scale one: a gap of 2e-6 there
      # is a decision on the 10th significant figure. Reported so a "better"
      # verdict can be seen for what it is.
      obj_gap_rel = unname((g_ray["obj"] - o_ray["obj"]) / abs(o_ray["obj"])),
      obj_seeded_from_glmm = seeded,
      obj_lme4_free = as.numeric(-2 * logLik(m_free)),
      lme4_roundtrip = unname(o_ray["obj"]) - as.numeric(lme4_rec[[cid]]$deviance),
      verdict = verdict, stringsAsFactors = FALSE)
  }
  ores <- do.call(rbind, rows)
  write.csv(ores, file.path(out_dir, "objective_verdicts.csv"), row.names = FALSE)
  cat("\n== objective verdicts (lme4's REML criterion at both candidates) ==\n")
  for (i in seq_len(nrow(ores))) {
    r <- ores[i, ]
    cat(sprintf("%-42s glmm %.5f  lme4 %.5f  gap %+.2e  seeded %.5f  [roundtrip %+.1e]  %s\n",
                r$case_id, r$obj_at_glmm, r$obj_at_lme4, r$obj_gap,
                r$obj_seeded_from_glmm, r$lme4_roundtrip, r$verdict))
  }
  cat(sprintf("\ntable: %s\n", file.path(out_dir, "objective_verdicts.csv")))
}
