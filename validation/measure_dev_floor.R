#!/usr/bin/env Rscript
# Corpus Δdev measurement. For every curated rung with both
# a glmm result and an lme4 result, computes d = aligned_dev(glmm) - aligned_dev(lme4)
# (and, informationally, vs MixedModels.jl where present) and prints/writes the
# table that tol.R's dev_eps/dev_big are pinned from.
# Rerunnable: same inputs (results/ from a fresh `./run.sh
# --oracles`) -> same results/dev_floor.csv.
#
# NOTE on aligned_dev's nagq>1 correction: it never fires here. The curated
# cross-engine sweep this script reads (results/<engine>_{empirical,simulated}/)
# is pinned at nAGQ=1 for every rung (manifest.json's "//agq_field" note --
# the AGQ opt-in pass writes to the SEPARATE results/{lme4,glmm}_agq_* dirs,
# which compare.R's and this script's glob never sees). So `meta$nagq` is
# always absent/1 here and passing the raw result object as `meta` (it carries
# $family, which IS needed if the branch ever did fire) is sufficient -- no
# need to resolve meta$data/r_formula/load_meta_data for this corpus.

suppressMessages(library(jsonlite))

script_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
suite_dir <- script_dir

source(file.path(script_dir, "tol.R"))
source(file.path(script_dir, "dev_align.R"))

# Same layout/merge convention as compare.R::read_engine -- results split
# across `<engine>_empirical/` and `<engine>_simulated/`, keyed by dataset name.
read_engine <- function(engine) {
  files <- unlist(lapply(c("empirical", "simulated"), function(s)
    list.files(file.path(suite_dir, "results", paste0(engine, "_", s)),
               pattern = "\\.json$", full.names = TRUE)))
  if (length(files) == 0) return(list())
  res <- lapply(files, fromJSON, simplifyVector = TRUE,
                simplifyDataFrame = FALSE, simplifyMatrix = TRUE)
  setNames(res, vapply(res, `[[`, "", "dataset"))
}

# Same small extraction helpers compare.R uses for the parameter gates (copied
# rather than sourced -- compare.R is a standalone script, not a library: it
# runs top-level comparisons and quit()s when read).
stddevs   <- function(r) unlist(lapply(r$estimates$varcomp, `[[`, "stddev"))
se_rx_of  <- function(r) if (r$family == "gaussian") r$estimates$se else r$estimates$se_rx
mark      <- function(diff, tol) {
  if (is.na(diff)) return("FAIL(len)")
  if (diff <= tol) "ok" else "FAIL"
}

# A rung is "benign" (its parameters sit in-band) when beta/SE/stddev all pass
# their ordinary TOL bands vs lme4: known-good fits are what define the
# stopping-rule noise floor, so only they may set it.
# Deliberately does NOT consult divergences.json: this is a measurement run
# characterizing dev's own noise floor, not the gated reference check.
is_benign <- function(a, b) {
  gaussian <- identical(a$family, "gaussian")
  d_beta <- rel_max(a$estimates$beta, b$estimates$beta)
  d_sd   <- if (is.null(stddevs(a)) && is.null(stddevs(b))) NA_real_
            else rel_max(stddevs(a), stddevs(b))
  if (gaussian) {
    d_se <- rel_max(a$estimates$se, b$estimates$se)
  } else {
    d_se <- rel_max(a$estimates$se_rx, b$estimates$se_rx)
  }
  m_beta <- mark(d_beta, TOL$beta_rel)
  m_sd   <- if (is.na(d_sd)) "n/a" else mark(d_sd, TOL$stddev_rel)
  m_se   <- mark(d_se, TOL$se_rel)
  !any(c(m_beta, m_sd, m_se) %in% c("FAIL", "FAIL(len)"))
}

# Sparse-routed rungs, enumerated by hand (manifest.json carries no
# route field, and a "sparse" substring in the dataset name is not the route --
# e.g. sim_sparse_binomial_bigsd (46) IS sparse but sim_sparse_gamma (24) is
# also sparse despite the family-first name; sim_slope_extra (7) has no
# "sparse" in its name at all despite being sparse-routed; sim_crossed_at_cap (17)
# sits exactly at MAX_EXTRA_GROUPINGS and is dense -- sparse needs strictly more).
SPARSE_RUNGS <- c(7, 8, 9, 18, 24, 38, 46)

lme4 <- read_engine("lme4")
mm   <- read_engine("mixedmodels")
glmm <- read_engine("glmm")
if (length(glmm) == 0) stop("no glmm results -- run ./run.sh --oracles first")
if (length(lme4) == 0) stop("no lme4 results -- run ./run.sh --oracles first")

rows <- list()
for (name in names(glmm)) {
  g <- glmm[[name]]
  a <- lme4[[name]]
  if (is.null(a)) {
    cat(sprintf("SKIP %-28s why=no lme4 result for this rung\n", name))
    next
  }
  b <- mm[[name]]  # may be NULL -- informational only

  dev_g <- aligned_dev(g$engine, g$estimates, g)
  dev_a <- aligned_dev(a$engine, a$estimates, a)
  if (is.na(dev_g) || is.na(dev_a)) {
    why <- if (is.na(dev_g)) attr(dev_g, "why") else attr(dev_a, "why")
    if (is.null(why)) why <- "aligned_dev returned NA with no why attribute"
    cat(sprintf("SKIP %-28s why=%s\n", name, why))
    next
  }
  d_lme4 <- dev_g - dev_a

  d_mm <- NA_real_
  if (!is.null(b)) {
    dev_b <- aligned_dev(b$engine, b$estimates, b)
    if (!is.na(dev_b)) d_mm <- dev_g - dev_b
  }

  route <- if (a$rung %in% SPARSE_RUNGS) "sparse" else "dense"
  benign <- is_benign(a, g)

  if (d_lme4 <= 0) {
    cat(sprintf("DEV-WIN %-28s rung=%-3d d_vs_lme4=%.6g (glmm deviance <= lme4's)\n",
                name, a$rung, d_lme4))
  }

  rows[[length(rows) + 1L]] <- list(
    rung = a$rung, dataset = name, family = a$family, route = route,
    dev_glmm = dev_g, dev_lme4 = dev_a, d_vs_lme4 = d_lme4, d_vs_mm = d_mm,
    benign = benign
  )
}

df <- do.call(rbind.data.frame, c(rows, stringsAsFactors = FALSE))
df <- df[order(-abs(df$d_vs_lme4)), ]

out_path <- file.path(suite_dir, "results", "dev_floor.csv")
write.csv(df, out_path, row.names = FALSE)

cat("\n=== Δdev vs lme4 (and, informationally, vs MixedModels.jl), sorted by |Δdev| desc ===\n")
cat(sprintf("%-28s %-5s %-8s %-8s %14s %14s %10s %7s\n",
            "dataset", "rung", "family", "route", "dev_glmm", "dev_lme4",
            "d_vs_lme4", "d_vs_mm"))
for (i in seq_len(nrow(df))) {
  r <- df[i, ]
  mm_str <- if (is.na(r$d_vs_mm)) "n/a" else sprintf("%.6g", r$d_vs_mm)
  cat(sprintf("%-28s %-5d %-8s %-8s %14.6f %14.6f %10.6g %7s%s\n",
              r$dataset, r$rung, r$family, r$route, r$dev_glmm, r$dev_lme4,
              r$d_vs_lme4, mm_str, if (r$benign) "" else "  (non-benign)"))
}

cat("\n=== per-family max benign |Δdev| (vs lme4; benign = beta/se/stddev all in-band) ===\n")
for (fam in sort(unique(df$family))) {
  sub <- df[df$family == fam & df$benign, ]
  if (nrow(sub) == 0) {
    cat(sprintf("%-10s no benign rungs\n", fam))
  } else {
    cat(sprintf("%-10s max |Δdev| = %.6g  (n benign = %d / %d)\n",
                fam, max(abs(sub$d_vs_lme4)), nrow(sub), sum(df$family == fam)))
  }
}

cat(sprintf("\nWrote %s (%d rungs)\n", out_path, nrow(df)))
