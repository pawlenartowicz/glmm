#!/usr/bin/env Rscript
# Estimate-grid mismatch classification.
# Reads what is already on disk -- reports/status_map.csv plus results/*.jsonl --
# and writes reports/mismatch_classes.csv: one row per unresolved cell with the
# class it falls in, which bands it broke, the aligned deviance gap, and the
# ABSOLUTE magnitude of the stddev coordinate that broke the stddev band.
#
# No fits. This produces a classification table, not a verdict on which engine
# is right -- that judgment is made elsewhere, from this table's output.
#
#   Rscript classify_mismatch.R [glmm.jsonl] [lme4.jsonl] [mixedmodels.jsonl]
#
# If results/lme4_msg.jsonl exists it is joined in too, for lme4's verbatim
# convergence messages. Those matter because `converged` is one bit
# covering several distinct warnings: a cell parked for lme4's singular-fit note
# and a cell parked because lme4's own gradient check failed are the same FALSE
# in the original records, and only the second is evidence about the ORACLE's
# fit rather than about the boundary.
#
# Scope = the cells analyze.R leaves unresolved: status `mismatch` (gate broken)
# and status `oracle-singular` (parked at analyze.R's status branch BEFORE the
# gate runs, so their deltas are computed and written but never judged).
suppressMessages({ library(jsonlite) })

suite_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(suite_dir, "..", "..", "tol.R"))
args <- commandArgs(TRUE)
glmm_path <- if (length(args) >= 1) args[1] else
  file.path(suite_dir, "results", "glmm.jsonl")
lme4_path <- if (length(args) >= 2) args[2] else
  file.path(suite_dir, "results", "lme4.jsonl")
mm_path   <- if (length(args) >= 3) args[3] else
  file.path(suite_dir, "results", "mixedmodels.jsonl")
out_dir <- file.path(suite_dir, "reports")

# ── the two thresholds this step needs that no existing constant supplies ──────
#
# DEV_REL_TOL -- "the aligned deviance agrees" for the class-(a) check. analyze.R
# computes d_dev_rel = |dev_glmm_aligned - dev_oracle| / (1 + |dev_oracle|) and
# deliberately does NOT gate on it, so the corpus has no band for it; tol.R's
# loglik_abs_* are absolute, on a different scale. MEASURED over the 60 cells in
# scope: every cell whose deviance is comparable sits at 4.55e-8 or below except
# lmm_q4sx2_g300p20_bal_lowsnr at 4.80e-3, five orders of magnitude out. 1e-6 is
# the round decade above the measured worst of the agreeing group, and the one
# outlier misses it by 4800x -- the two populations do not come close to
# touching, so nothing here is sensitive to where between them the line sits.
DEV_REL_TOL <- 1e-6
# NEARZERO_MAG -- class (c)'s boundary: below what absolute stddev does a
# relative comparison stop carrying information? The spec leaves this open on
# purpose ("a decision, not a given"), so this is a PROVISIONAL cut and the
# sd_worst_mag column is reported next to every class so it can be redrawn
# without re-deriving anything. 1e-2 is one decade above TOL$near_zero_abs, the
# absolute floor rel_max already exempts: coordinates between the two are ones
# rel_max still asks a relative question about but whose own magnitude is within
# a decade of the floor where that question was ruled meaningless.
NEARZERO_MAG <- 1e-2

status_map <- read.csv(file.path(out_dir, "status_map.csv"), stringsAsFactors = FALSE)
glmm <- read_jsonl(glmm_path)
lme4 <- read_jsonl(lme4_path)
mm   <- read_jsonl(mm_path)
msg_path <- file.path(suite_dir, "results", "lme4_msg.jsonl")
lme4_msg <- if (file.exists(msg_path)) read_jsonl(msg_path) else list()
manifest <- fromJSON(file.path(suite_dir, "manifest.json"), simplifyDataFrame = FALSE)
cells <- setNames(manifest$cells, vapply(manifest$cells, `[[`, "", "case_id"))

targets <- status_map$case_id[status_map$status %in% c("mismatch", "oracle-singular")]

# The MixedModels-adjudicated cells carry NO deltas in status_map.csv: analyze.R's
# reclassification block rewrites their `status` and `oracle_engine` from
# boundary37_verdicts.csv but leaves the delta columns at the NA they got from the
# lme4 join, which refused those cells outright. Read the deltas back from the
# verdicts file, or every such cell classifies as "no band broken" on missing data.
b37 <- read.csv(file.path(out_dir, "boundary37_verdicts.csv"), stringsAsFactors = FALSE)
for (i in which(status_map$oracle_engine == "MixedModels")) {
  v <- b37[match(status_map$case_id[i], b37$case_id), ]
  if (is.na(v$case_id[1])) next
  for (k in c("d_beta", "d_stddev", "d_corr", "d_se",
              "tol_beta", "tol_stddev", "tol_corr", "tol_se")) {
    status_map[[k]][i] <- v[[k]]
  }
}

# stddevs_of (tol.R) flattened WITH its coordinate labels. The ordering here must
# stay identical to stddevs_of's -- groups sorted by name, terms in emitted order
# -- or the labels name the wrong coordinate. Change together.
stddev_labels_of <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(character(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  vc <- vc[order(vapply(vc, function(g) g$group, ""))]
  unlist(lapply(vc, function(g)
    paste0(g$group, ":", unlist(g$terms))))
}

# Which oracle record a cell joins against. analyze.R reclassified the 37
# lme4-identifiability cells to MixedModels via boundary37_verdicts.csv, so a
# cell's oracle_engine column -- not the file it was read from -- names its
# reference.
oracle_of <- function(cid, engine) {
  if (identical(engine, "MixedModels")) mm[[cid]] else lme4[[cid]]
}

rows <- list()
for (cid in targets) {
  sm <- status_map[status_map$case_id == cid, ]
  cell <- cells[[cid]]
  g <- glmm[[cid]]; o <- oracle_of(cid, sm$oracle_engine)

  # ── which bands broke ───────────────────────────────────────────────────────
  # Recomputed from the recorded deltas rather than re-derived: analyze.R already
  # wrote both the delta and the band it was judged against into status_map.csv.
  broke <- character(0)
  if (!is.na(sm$d_beta)   && sm$d_beta   > sm$tol_beta)   broke <- c(broke, "beta")
  if (!is.na(sm$d_stddev) && sm$d_stddev > sm$tol_stddev) broke <- c(broke, "stddev")
  if (!is.na(sm$d_corr)   && sm$d_corr   > sm$tol_corr)   broke <- c(broke, "corr")
  if (!is.na(sm$d_se)     && sm$d_se     > sm$tol_se)     broke <- c(broke, "se")

  dev_ok <- if (is.na(sm$d_dev_rel)) NA else sm$d_dev_rel <= DEV_REL_TOL
  # MixedModels rows have no d_dev_rel (analyze.R computes it only against
  # lme4/lmer). verify_boundary37.R recorded an ABSOLUTE REML-criterion gap for
  # them instead -- a different scale, so it gets its own column rather than
  # being folded into d_dev_rel.
  dev_gap_abs <- if (identical(sm$oracle_engine, "MixedModels"))
    b37$dev_gap[match(cid, b37$case_id)] else NA_real_

  # ── the stddev coordinate that broke the band ───────────────────────────────
  # rel_max reports one number over the whole vector; the class turns on WHICH
  # coordinate produced it and how big that coordinate is in absolute terms.
  sd_worst_lab <- NA_character_
  sd_glmm <- sd_oracle <- sd_worst_mag <- sd_worst_rel <- NA_real_
  n_exempt <- NA_integer_   # coordinates rel_max skipped under TOL$near_zero_abs
  pinned_side <- NA_character_
  if (!is.null(g) && !is.null(o)) {
    sg <- stddevs_of(g); so <- stddevs_of(o)
    if (length(sg) && length(sg) == length(so)) {
      lab <- stddev_labels_of(g)
      mag <- pmax(abs(sg), abs(so), 1e-12)
      rel <- abs(sg - so) / mag
      exempt <- mag <= TOL$near_zero_abs
      n_exempt <- sum(exempt)
      rel[exempt] <- 0
      j <- which.max(rel)
      if (rel[j] > 0) {
        sd_worst_lab <- lab[j]; sd_glmm <- sg[j]; sd_oracle <- so[j]
        sd_worst_mag <- mag[j]; sd_worst_rel <- rel[j]
        # A coordinate one engine pins to an exact zero: the relative metric can
        # only ever return 1.0 there, whatever the other engine reports.
        if (sg[j] == 0) pinned_side <- "glmm"
        else if (so[j] == 0) pinned_side <- "oracle"
      }
    }
  }

  # ── what lme4 actually said ─────────────────────────────────────────────────
  # Three outcomes worth telling apart, and the third is the one that changes a
  # verdict: `max-grad` means lme4's OWN convergence check failed on that cell,
  # so a gap against it is evidence about the oracle's fit, not about glmm's.
  msgs <- if (!is.null(lme4_msg[[cid]])) as.character(unlist(lme4_msg[[cid]]$messages))
          else character(0)
  msg_class <- if (is.null(lme4_msg[[cid]]) || is.null(lme4_msg[[cid]]$messages)) "n/a"
               else if (length(msgs) == 0) "none"
               else if (any(grepl("singular", msgs))) "singular-fit"
               else if (any(grepl("max\\|grad\\|", msgs))) "max-grad"
               else "other"
  msg_text <- if (length(msgs)) sub("\n.*", "", msgs[1]) else ""

  # ── class ───────────────────────────────────────────────────────────────────
  # Assigned by which bands broke, most-consequential first: a beta disagreement
  # is not explainable by a reporting convention, so it takes precedence over a
  # stddev breach on the same cell.
  cls <-
    if (length(broke) == 0) "(ok) within all bands"
    else if ("beta" %in% broke) "(d) beta-level disagreement"
    else if ("stddev" %in% broke) {
      if (!is.na(pinned_side) && !is.na(sd_worst_mag) &&
          sd_worst_mag <= TOL$near_zero_abs)      "(b0) both engines at zero"
      else if (!is.na(pinned_side))               "(b) pinned component"
      else if (!is.na(sd_worst_mag) &&
               sd_worst_mag < NEARZERO_MAG)       "(c) near-zero theta"
      else                                        "(e) unexplained"
    }
    else if (identical(broke, "se")) "(a) SE-only"
    else "(e) unexplained"

  rows[[length(rows) + 1L]] <- data.frame(
    case_id = cid, status = sm$status, class = cls,
    bands_broken = if (length(broke)) paste(broke, collapse = "+") else "",
    category = sm$category, family = sm$family, structure = sm$structure,
    oracle_engine = sm$oracle_engine,
    sing_glmm = sm$sing_glmm, sing_oracle = sm$sing_oracle,
    d_beta = sm$d_beta, d_stddev = sm$d_stddev, d_corr = sm$d_corr,
    d_se = sm$d_se, d_dev_rel = sm$d_dev_rel, dev_ok = dev_ok,
    dev_gap_abs = dev_gap_abs,
    sd_worst_coord = sd_worst_lab, sd_worst_rel = sd_worst_rel,
    sd_worst_mag = sd_worst_mag, sd_glmm = sd_glmm, sd_oracle = sd_oracle,
    sd_pinned_by = pinned_side, n_stddev_exempt = n_exempt,
    lme4_msg_class = msg_class, lme4_message = msg_text,
    tol_beta = sm$tol_beta, tol_stddev = sm$tol_stddev,
    tol_corr = sm$tol_corr, tol_se = sm$tol_se,
    stringsAsFactors = FALSE)
}
res <- do.call(rbind, rows)
res <- res[order(res$class, res$case_id), ]
write.csv(res, file.path(out_dir, "mismatch_classes.csv"), row.names = FALSE)

cat(sprintf("\n== estimate-grid mismatch classes (%d cells: %d mismatch + %d oracle-singular) ==\n",
            nrow(res), sum(res$status == "mismatch"),
            sum(res$status == "oracle-singular")))
print(table(res$class, res$status))
cat("\n-- class x what lme4 said --\n")
print(table(res$class, res$lme4_msg_class))
cat(sprintf("\nDEV_REL_TOL = %.0e   NEARZERO_MAG = %.0e (provisional -- see sd_worst_mag)\n",
            DEV_REL_TOL, NEARZERO_MAG))

cat("\n-- per cell --\n")
for (cl in sort(unique(res$class))) {
  cat(sprintf("\n%s\n", cl))
  s <- res[res$class == cl, ]
  for (i in seq_len(nrow(s))) {
    r <- s[i, ]
    cat(sprintf("  %-42s %-15s broke=%-15s beta=%.1e sd=%.1e corr=%.1e se=%.1e dev=%.1e%s\n",
                r$case_id, r$status, if (nzchar(r$bands_broken)) r$bands_broken else "-",
                r$d_beta, r$d_stddev, r$d_corr, r$d_se, r$d_dev_rel,
                if (is.na(r$sd_worst_mag)) "" else
                  sprintf("\n      worst sd coord %s: glmm %.6g vs oracle %.6g (mag %.2e, rel %.2e%s)",
                          r$sd_worst_coord, r$sd_glmm, r$sd_oracle, r$sd_worst_mag,
                          r$sd_worst_rel,
                          if (is.na(r$sd_pinned_by)) "" else
                            paste0(", pinned to 0 by ", r$sd_pinned_by))))
  }
}
cat(sprintf("\ntable: %s\n", file.path(out_dir, "mismatch_classes.csv")))
