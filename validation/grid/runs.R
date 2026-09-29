# runs.R -- run discovery and the comparison metrics for the accuracy grid.
#
# DEFINES FUNCTIONS AND NOTHING ELSE. compare.R is a script: it runs the
# comparisons at top level and quit()s, so the summaries cannot source it. The
# pieces all three need -- finding the right run directories, the estimator-arm
# split, the per-quantity bands, the deviance and truth metrics -- live here
# instead, in one copy. Sourcing this file is free; it prints nothing and
# compares nothing.
#
# The three files below are sourced HERE rather than by each consumer, so a
# consumer that wants the grid's comparison vocabulary takes one source() line
# and cannot end up with half of it. rel_max, port_rel_max, read_jsonl,
# stddevs_of, corrs_of, tol_for come from tol.R; aligned_dev, align_status and
# ga_rule_dev from dev_align.R; grid_versions and norm_version from
# engines/versions.R.
# None of them is redefined here -- one definition each, repo-wide.

# This file's own directory, resolved WHILE THE source() THAT READS IT IS STILL
# ON THE STACK: `ofile` lives in the sourcing frame and is gone afterwards.
# Frames are walked innermost-first so a nested source() still resolves to the
# file being read, and the basename test keeps a nested source() of some other
# file from answering for this one. A symlinked grid tree resolves to the real
# directory here, which is right: this only decides where the SIBLING SCRIPTS
# are read from. Where `runs/` is read from is the caller's `grid_dir`.
runs_script_dir <- local({
  dir <- NULL
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (is.character(of) && length(of) == 1L && file.exists(of) &&
        identical(basename(of), "runs.R")) {
      dir <- dirname(normalizePath(of))
      break
    }
  }
  if (is.null(dir)) {
    arg <- grep("--file=", commandArgs(FALSE), value = TRUE)
    dir <- if (length(arg) == 1L) dirname(normalizePath(sub("--file=", "", arg)))
           else getwd()
  }
  dir
})
source(file.path(runs_script_dir, "tol.R"))
source(file.path(runs_script_dir, "dev_align.R"))
source(file.path(runs_script_dir, "engines", "versions.R"))

# ── run discovery ────────────────────────────────────────────────────────────
# The reference for each oracle is the NEWEST run under runs/<oracle>/ whose
# run_meta.json has subset == "full" and whose engine_version matches
# versions.json. A version mismatch is an ERROR, not a fallback: a reference
# fitted by an unpinned oracle is not the reference. A `fast` or timed oracle run
# is never an accuracy reference, which is what the subset test and the `timed`
# test below enforce.
ORACLES <- c("lme4", "glmmTMB", "GLMMadaptive", "MixedModels")
ENGINE_DIR <- c(lme4 = "lme4", glmmTMB = "glmmtmb",
                GLMMadaptive = "glmmadaptive", MixedModels = "mixedmodels")

# Both files are checked before either is parsed: jsonlite treats a missing path
# as a JSON string and reports a lexical error pointing at the path, which tells
# a caller who mistyped --glmm-run= nothing about what went wrong.
read_run <- function(dir) {
  for (f in c("run_meta.json", "results.jsonl")) {
    if (!file.exists(file.path(dir, f)))
      stop(sprintf("%s is not a run directory: no %s in it", dir, f), call. = FALSE)
  }
  meta <- jsonlite::fromJSON(file.path(dir, "run_meta.json"), simplifyVector = TRUE)
  list(dir = dir, meta = meta, recs = read_jsonl(file.path(dir, "results.jsonl")))
}

# `required_cells` is every manifest cell whose `oracles` list names this engine.
# A run qualifies only if its results.jsonl holds a record for all of them, of
# ANY status: run_meta.json's `subset` records what the invocation was ASKED to
# do, and it is written before the run can be killed, time out or die halfway, so
# it is intent, not evidence. The record set is the only evidence the run
# finished, and an accuracy reference missing a third of its cells would silently
# shrink every gate that reads it. An engine-fail or timeout record counts:
# "this engine could not fit this cell" is an answer, a missing line is not.
newest_oracle_run <- function(grid_dir, oracle, want_version, required_cells) {
  root <- file.path(grid_dir, "runs", ENGINE_DIR[[oracle]])
  dirs <- list.dirs(root, recursive = FALSE)
  ok <- list()
  for (d in dirs) {
    mp <- file.path(d, "run_meta.json")
    if (!file.exists(mp) || !file.exists(file.path(d, "results.jsonl"))) next
    m <- jsonlite::fromJSON(mp, simplifyVector = TRUE)
    # subset == "full" ONLY. run.sh records "fast" for a --fast run and "cells"
    # for a cell-restricted one, so a smoke, probe or pilot run is filtered out
    # here by construction rather than by someone remembering to delete it.
    if (!identical(m$subset, "full")) next
    if (!is.null(m$timed) && !is.na(m$timed)) next
    ok[[length(ok) + 1L]] <- list(dir = d, date = m$date, version = m$engine_version)
  }
  if (!length(ok)) return(NULL)
  ok <- ok[order(vapply(ok, `[[`, "", "date"), decreasing = TRUE)]
  for (cand in ok) {
    run <- read_run(cand$dir)
    missing <- setdiff(required_cells, names(run$recs))
    if (length(missing)) {
      cat(sprintf(paste0("SKIP %s run %s: run_meta says subset \"full\" but ",
                         "results.jsonl holds no record for %d of the %d cells whose ",
                         "manifest `oracles` name it (first missing: %s).\n"),
                  oracle, basename(cand$dir), length(missing), length(required_cells),
                  missing[[1]]))
      next
    }
    # The version is checked on the first COMPLETE run, and a mismatch there is
    # an error rather than a reason to keep walking back: a reference fitted by
    # an unpinned oracle is not the reference, and silently reaching past it to
    # an older one would gate against a version nobody asked for.
    if (!identical(norm_version(cand$version), norm_version(want_version))) {
      stop(sprintf(paste0("%s: newest complete full run %s was fitted by version %s, ",
                          "but grid/versions.json pins %s. A version mismatch is an ",
                          "error, not a fallback -- rerun that oracle on the full grid ",
                          "into a NEW run directory, never edit an old one."),
                   oracle, basename(cand$dir), cand$version, want_version),
           call. = FALSE)
    }
    return(run)
  }
  NULL
}

# The glmm (or port) run to gate or report on. `subset: "full"` runs are
# PREFERRED, newest first; anything else is a fallback used only when no full run
# exists. Without that preference, a `--fast` port-gate run made minutes after
# the full baseline would be "the newest run under runs/glmm/" and the campaign's
# final adjudication would silently gate 60 cells against four full oracle
# references. Whatever it picks, it PRINTS -- directory, subset and label -- so
# the choice is never invisible.
newest_glmm_run <- function(grid_dir, engine = "glmm", override = NULL) {
  say <- function(r) {
    cat(sprintf("%s run:  %s\n", engine, r$dir))
    cat(sprintf("  subset:  %s   label: %s   cells: %s\n",
                r$meta$subset, r$meta$label, r$meta$cells))
    r
  }
  if (!is.null(override)) return(say(read_run(override)))
  root <- file.path(grid_dir, "runs", engine)
  dirs <- c(list.dirs(root, recursive = FALSE),
            list.dirs(file.path(root, "scratch"), recursive = FALSE))
  dirs <- Filter(function(d) file.exists(file.path(d, "run_meta.json")) &&
                             file.exists(file.path(d, "results.jsonl")), dirs)
  if (!length(dirs)) return(NULL)
  metas <- lapply(dirs, function(d)
    jsonlite::fromJSON(file.path(d, "run_meta.json"), simplifyVector = TRUE))
  full <- vapply(metas, function(m) identical(m$subset, "full"), TRUE)
  dates <- vapply(metas, `[[`, "", "date")
  # order(): full runs first, then newest date first within each group.
  ord <- order(!full, dates, decreasing = c(FALSE, TRUE), method = "radix")
  if (!any(full)) {
    cat(sprintf("NOTE: no %s run with subset \"full\" -- falling back to the newest partial run.\n",
                engine))
  }
  say(read_run(dirs[ord][1]))
}

# ── coefficient alignment ────────────────────────────────────────────────────
# lme4 labels categorical contrasts "period2"; MixedModels "period: 2", and
# interactions "a & b" where lme4 writes "a:b". Same base, same levels. Normalize
# the cosmetic formatting so the coefficient match checks coding, not label style.
norm_coef <- function(x) gsub("[:& ]", "", x)

# Permutation that reorders `b_names` into `a_names`'s order (after norm_coef),
# or NULL when the two coefficient sets can't be matched unambiguously --
# different name sets, a name repeated after normalization, or a missing name.
# NULL means "do not compare": engines are free to order factor levels
# differently (lme4's numeric `dept` vs MixedModels/glmm's lexicographic one on
# InstEval), so a caller must never fall back to positional order.
coef_perm <- function(a_names, b_names) {
  na <- norm_coef(a_names); nb <- norm_coef(b_names)
  if (length(na) != length(nb) || anyDuplicated(na) || anyDuplicated(nb) ||
      !setequal(na, nb)) {
    return(NULL)
  }
  match(na, nb)
}

# rel_max over a coefficient-indexed vector pair, reordering `y` into `x`'s
# coefficient order first via coef_perm. NA_real_ (rendered FAIL(len)/n/a by the
# mark()/cell() machinery, same as any other non-comparable case) when alignment
# fails or the lengths don't match their name vectors -- never a positional
# number over mismatched coefficients.
rel_max_by_coef <- function(x, a_names, y, b_names) {
  if (length(x) != length(a_names) || length(y) != length(b_names)) return(NA_real_)
  perm <- coef_perm(a_names, b_names)
  if (is.null(perm)) return(NA_real_)
  rel_max(x, y[perm])
}

mark <- function(diff, tol) {
  if (is.na(diff)) return("FAIL(len)")
  if (diff <= tol) "ok" else "FAIL"
}

# Correlations are compared by the MAX ABSOLUTE difference, not by rel_max.
# Their band was measured as an absolute number (agq_corr_abs = 4e-3 against
# GLMMadaptive on the vector-RE AGQ rungs), and a relative test against an
# absolute band silently tightens as the correlation shrinks: two engines 2e-3
# apart pass at rho = 0.8 and fail at rho = 0.05, having disagreed by the same
# amount both times. Coordinates whose own magnitude (the larger of the two
# sides) is at or below `floor_` score 0, the same near-zero rule rel_max
# applies, because a correlation of zero in both engines carries no signal to
# compare. NA on a length mismatch, so it surfaces as a hard failure rather than
# a silently-recycled false pass.
corr_abs_max <- function(x, y, floor_) {
  if (length(x) != length(y)) return(NA_real_)
  if (!length(x)) return(NA_real_)
  # An UNDEFINED coordinate, not a missing one. An engine that pins a whole
  # random-effect block to zero has no correlation to report for it -- the ratio
  # is 0/0 -- and records the off-diagonals as null, which arrives here as NA.
  # There is no comparison to make, so this says so and the caller renders it the
  # way it renders any other absent quantity.
  if (anyNA(x) || anyNA(y)) return(NA_real_)
  keep <- pmax(abs(x), abs(y)) > floor_
  if (!any(keep)) return(0)
  max(abs(x[keep] - y[keep]))
}

# ── estimator arms ───────────────────────────────────────────────────────────
# Laplace and AGQ minimise DIFFERENT objectives, so a deviance comparison that
# mixes them is meaningless. The cell's arm is what glmm was asked to do; an
# engine's arm is what that engine actually ran on this cell.
cell_arm <- function(cell) if (is.null(cell[["nagq"]]) || cell[["nagq"]] <= 1) "laplace" else "agq"
engine_arm <- function(engine, cell) switch(engine,
  glmm = , glmm_python = , glmm_r = cell_arm(cell),
  lme4 = cell_arm(cell),            # honors nAGQ
  GLMMadaptive = cell_arm(cell),    # runs at the cell's nagq (1 on a Laplace cell)
  glmmTMB = "laplace",              # Laplace only
  MixedModels = "laplace",          # Laplace only
  stop("no arm rule for engine ", engine))

# The band for one quantity on one cell, ARM-DEPENDENT. The agq_* bands were
# calibrated against GLMMadaptive on the vector-RE AGQ rungs and are the wrong
# floor for a Laplace cell, which would silently borrow a band it never earned.
# se_rx has no AGQ variant because none was ever calibrated. The correlation band
# is ABSOLUTE -- a relative question has no answer once both sides are near zero
# -- and a Laplace cell uses near_zero_abs, the general floor.
# compare.R and summarize_accuracy.R both call this, so the two can never
# disagree about which band a cell is read against.
band_for <- function(cell, quantity) {
  agq <- identical(cell_arm(cell), "agq")
  key <- switch(quantity,
    beta       = if (agq) "agq_beta_rel" else "beta_rel",
    se_rx      = "se_rel",
    se_hessian = if (agq) "agq_se_hessian_rel" else "se_hessian_rel",
    stddev     = if (agq) "agq_stddev_rel" else "stddev_rel",
    corr       = if (agq) "agq_corr_abs" else "near_zero_abs",
    stop("no band for quantity ", quantity))
  tol_for(cell[["cell"]], key)
}

# ── gate 1: deviance vs the BEST oracle ──────────────────────────────────────
# `oracle_recs` is the per-cell list of oracle records, built once by the caller
# and reused by gates 1 and 2, with the oracles that did not fit the cell simply
# ABSENT rather than present as a NULL every consumer has to guard.
# best = min aligned deviance over the oracles that (a) fit this cell, (b)
# converged, (c) ran on the CELL'S arm, and (d) were not excluded as an outlier.
#
# OUTLIER EXCLUSION, from THREE candidates up: an oracle whose own aligned
# deviance is more than e_big from the other oracles is excluded from `best` for
# that cell and the exclusion printed. Implemented against the MEDIAN of the
# candidate set, which is not itself dragged by the outlier.
#
# EXACTLY TWO CANDIDATES EXCLUDE NOBODY. The median of two is their midpoint, so
# each sits half the gap from it and any disagreement wider than 2 x e_big throws
# out BOTH -- the cell goes DEV-NA and the gate falls silent exactly where the
# two engines disagree most. Measured on the pilot: every gamma, cloglog and
# negative-binomial cell has only lme4 and glmmTMB, and the Gamma cells are where
# those two disagree, which is the disagreement gate 1 exists to show. So with
# two candidates `best` is simply the smaller of the pair, `gap` carries their
# distance for the caller to print, and the normal verdict follows: glmm level
# with the better one passes, glmm level with the worse one fails on Delta, and a
# glmm deviance e_big from both is still the convention-mismatch failure.
# With a single candidate there is nothing to be an outlier FROM either.
#
# A COLLAPSED GAUSSIAN FIT IS NOT A CANDIDATE, and that is decided before the
# two-versus-three branch above, because the outlier test cannot reach it: with
# two candidates it excludes nobody, and a residual standard deviation driven to
# zero takes the likelihood to infinity, so the collapsed fit holds the lowest
# deviance on the cell and gate 1 would ask glmm to match a fit that describes
# nothing. The test is the near-zero floor on the reported sigma, and it is
# GAUSSIAN ONLY: on Gamma the same record slot carries a dispersion, where a
# small value is an ordinary tight fit rather than a degenerate one.
best_oracle_dev <- function(cell, oracle_recs, e_big) {
  cand <- list()
  excluded <- character(0)
  near_zero <- tol_for(cell[["cell"]], "near_zero_abs")
  for (o in names(oracle_recs)) {
    r <- oracle_recs[[o]]
    if (is.null(r) || !isTRUE(r$converged)) next
    if (!identical(engine_arm(o, cell), cell_arm(cell))) next
    if (identical(cell[["family"]], "gaussian") && !is.null(r$sigma) &&
        isTRUE(as.numeric(r$sigma) < near_zero)) {
      excluded <- c(excluded, sprintf("%s(sigma=%.2g collapsed)", o, as.numeric(r$sigma)))
      next
    }
    d <- aligned_dev(r, cell)
    if (is.na(d)) next
    cand[[o]] <- as.numeric(d)
  }
  if (!length(cand)) return(list(best = NA_real_, who = NA_character_,
                                 excluded = excluded, gap = NA_real_,
                                 why = "no oracle on this cell's arm has a usable aligned deviance"))
  gap <- NA_real_
  if (length(cand) == 2L) {
    gap <- abs(diff(unlist(cand)))
  } else if (length(cand) >= 3L) {
    med <- stats::median(unlist(cand))
    outliers <- names(cand)[abs(unlist(cand) - med) > e_big]
    excluded <- c(excluded, outliers)
    cand <- cand[setdiff(names(cand), outliers)]
  }
  if (!length(cand)) return(list(best = NA_real_, who = NA_character_,
                                 excluded = excluded, gap = gap,
                                 why = "every oracle on this arm was excluded as an outlier"))
  who <- names(cand)[which.min(unlist(cand))]
  list(best = min(unlist(cand)), who = who, excluded = excluded, gap = gap,
       why = NA_character_)
}

# The verdict for one cell. eps and big are read through tol_for(cell[["cell"]], ...)
# at the call site -- NEVER `TOL$dev_eps` directly. tol_for is where the NA guard
# lives, and a gate that bypassed it would compare against NA (which is FALSE
# everywhere) and pass every cell silently. That is the one failure mode a
# tolerance table must not have, and gate 1 is the gate the two constants exist
# for, so it is the last place to take a shortcut.
dev_verdict <- function(delta, eps, big) {
  if (is.na(delta)) return("DEV-NA")
  if (abs(delta) > big) return("FAIL(conv?)")   # convention mismatch, not a fit result
  if (delta > eps) return("FAIL(dev)")
  if (delta <= 0) return("DEV-WIN")             # equal or better optimum of the shared objective
  "DEV-OK"
}

# ── gate 2: parameters vs the NEAREST oracle ─────────────────────────────────
# A quantity is ABSENT from a record, not merely small: an LMM cell has no
# se_hessian, a fixed-effects-only cell no varcomp. The test has to be explicit
# because rel_max treats TWO EMPTY vectors as a length MATCH rather than an
# absence -- a GLM cell's varcomp is `[]` on every engine, so both sides are
# numeric(0) and max() over the empty comparison is -Inf, not NA. Checking here
# also keeps rel_max's length-mismatch NA a real FAIL(len) instead of laundering
# it into an n/a pass with a blanket is.na().
has_quantity <- function(rec, quantity) switch(quantity,
  beta       = length(rec$beta) > 0,
  se_rx      = !is.null(rec$se_rx) && length(rec$se_rx) > 0,
  se_hessian = !is.null(rec$se_hessian) && length(rec$se_hessian) > 0,
  stddev     = length(stddevs_of(rec)) > 0,
  # A correlation that is UNDEFINED counts as absent. An engine that pins a whole
  # random-effect block to zero records that block's off-diagonals as null (the
  # ratio is 0/0), so the quantity does not exist for that engine on that cell --
  # the same state a scalar random effect is in, and it prints the same `n/a`.
  # The block's standard deviations are real numbers and still compare.
  corr       = length(corrs_of(rec)) > 0 && !anyNA(corrs_of(rec)),
  stop("no such quantity ", quantity))

# glmm passes a quantity if it is within the tol.R band of AT LEAST ONE oracle
# that fit the cell. "At least one" is the whole shape of this gate, and it is
# why a defect in one oracle does not redden every cell of a family here: glmm
# agrees with another engine, so the quantity passes, and it is gate 3 that makes
# the defect visible instead.
# Returns the SMALLEST difference over the oracles and which oracle gave it;
# NA_real_ when no oracle offered a comparable vector at all.
nearest_oracle_diff <- function(g, oracle_recs, cell, quantity) {
  best <- NA_real_; who <- NA_character_
  for (o in names(oracle_recs)) {
    r <- oracle_recs[[o]]
    if (is.null(r) || !isTRUE(r$converged)) next
    d <- switch(quantity,
      beta       = rel_max_by_coef(g$beta, g$coef_names, r$beta, r$coef_names),
      se_rx      = if (is.null(g$se_rx) || is.null(r$se_rx)) NA_real_
                   else rel_max_by_coef(g$se_rx, g$coef_names, r$se_rx, r$coef_names),
      se_hessian = if (is.null(g$se_hessian) || is.null(r$se_hessian)) NA_real_
                   else rel_max_by_coef(g$se_hessian, g$coef_names, r$se_hessian, r$coef_names),
      # stddevs_of and corrs_of flatten POSITIONALLY once the groups are in
      # canonical name order, so the two records must agree on what those
      # positions are: the same grouping factors, each with the same terms. When
      # they do not, the two vectors can still be the same length while meaning
      # different things, and a number computed over them is not a comparison.
      stddev     = if (!identical(varcomp_keys(g), varcomp_keys(r))) NA_real_
                   else rel_max(stddevs_of(g), stddevs_of(r)),
      # ABSOLUTE, and the band doubles as the near-zero floor -- see corr_abs_max.
      corr       = if (!identical(varcomp_keys(g), varcomp_keys(r))) NA_real_
                   else corr_abs_max(corrs_of(g), corrs_of(r), band_for(cell, "corr")),
      stop("no such quantity ", quantity))
    if (is.na(d)) next
    if (is.na(best) || d < best) { best <- d; who <- o }
  }
  list(diff = best, who = who)
}

# ── gate 3: truth error per family ───────────────────────────────────────────
# Over the GENERATED cells of one family and one estimator arm, the per-cell
# error is
#     |estimate - truth| / |truth|        when |truth| >= truth_floor
#     |estimate - truth|                  otherwise
# computed separately for beta, for random-effect SDs and for correlations.
#
# THE PAIRING IS THE WHOLE GATE, so it is built into the statistic and not only
# into the band. For EACH oracle, err_glmm and err_oracle are means over the SAME
# set of cells -- those where glmm AND that oracle both converged and both
# produced a comparable vector. The best oracle is the one with the lowest
# err_oracle ON ITS OWN PAIRED SET, and the band is 2 x SEM of the per-cell
# difference on that same set. Taking each engine's mean over its own converged
# subset instead would compare a glmm mean over 80 cells against an oracle mean
# over 55, and the pairing argument -- that the band is about the glmm-versus-
# oracle gap, not about the spread between a 60-row and a 30000-row cell -- would
# not hold for the quantity actually compared.
truth_err <- function(est, truth, floor_) {
  if (!length(est)) return(NA_real_)
  mean(ifelse(abs(truth) < floor_, abs(est - truth), abs(est - truth) / abs(truth)))
}

# One (family, arm, quantity) row per oracle. `extract(rec, cell)` returns the
# estimate vector aligned to the truth vector, or NULL when they cannot be
# aligned -- see the TRUTH-MISMATCH rule in compare.R.
paired_truth_table <- function(cells, glmm_recs, oracle_recs_by_cell, extract,
                               truth_vec, floor_) {
  rows <- list(); mismatches <- character(0)
  # Cells where the quantity is UNDEFINED rather than wrong: an engine that pins
  # a random-effect block to zero reports that block's correlations as null, and
  # a correlation that does not exist cannot be scored against a true one. These
  # leave the group, counted and printed -- unlike a length mismatch, which is a
  # wiring fault and fails.
  undefined <- character(0)
  # How many cells of this group carry a truth vector for this quantity at all.
  # An empty table means two different things -- "a scalar random effect has no
  # correlation to score" and "every pair on these cells failed to converge" --
  # and only the caller can tell them apart, so the count travels with the rows.
  n_with_truth <- sum(vapply(cells, function(cl) length(truth_vec(cl)) > 0, TRUE))
  for (o in ORACLES) {
    d_glmm <- c(); d_orac <- c(); ids <- character(0)
    for (cl in cells) {
      # One estimator arm only, the rule gate 1 uses: a Laplace oracle scored
      # into an AGQ group would be compared on a different objective's optimum.
      if (!identical(engine_arm(o, cl), cell_arm(cl))) next
      g <- glmm_recs[[cl[["cell"]]]]; r <- oracle_recs_by_cell[[cl[["cell"]]]][[o]]
      if (is.null(g) || is.null(r) || !isTRUE(g$converged) || !isTRUE(r$converged)) next
      tv <- truth_vec(cl)
      eg <- extract(g, cl); er <- extract(r, cl)
      # A quantity this cell's truth does not carry -- a scalar random effect has
      # no correlation -- is nothing to compare, not a mismatch. It is only
      # silent while BOTH sides are empty; an engine that reported coordinates
      # the truth does not have falls through to the length test below.
      if (!length(tv) && !length(eg) && !length(er)) next
      if ((!is.null(eg) && anyNA(eg)) || (!is.null(er) && anyNA(er))) {
        undefined <- union(undefined, cl[["cell"]])
        next
      }
      # A LENGTH MISMATCH IS A FAILURE, NOT AN EXCLUSION. A coefficient-ordering
      # bug, a dropped aliased column or a varcomp flattened in a different group
      # order all show up here, and dropping the cell from the mean would hide a
      # systematic wiring fault behind a slightly smaller n.
      if (is.null(eg) || length(eg) != length(tv)) {
        mismatches <- c(mismatches, sprintf("%s glmm len=%s truth=%d", cl[["cell"]],
                                            if (is.null(eg)) "NULL" else length(eg), length(tv)))
        next
      }
      if (is.null(er) || length(er) != length(tv)) {
        mismatches <- c(mismatches, sprintf("%s %s len=%s truth=%d", cl[["cell"]], o,
                                            if (is.null(er)) "NULL" else length(er), length(tv)))
        next
      }
      d_glmm <- c(d_glmm, truth_err(eg, tv, floor_))
      d_orac <- c(d_orac, truth_err(er, tv, floor_))
      ids <- c(ids, cl[["cell"]])
    }
    if (!length(ids)) next
    diff <- d_glmm - d_orac
    rows[[o]] <- list(oracle = o, n = length(ids),
                      err_glmm = mean(d_glmm), err_oracle = mean(d_orac),
                      paired_mean = mean(diff),
                      paired_2sem = 2 * stats::sd(diff) / sqrt(length(diff)))
  }
  # `n_scoreable` is what an empty table has to be judged against: cells that
  # carry a truth vector MINUS those whose estimate is undefined. Without the
  # subtraction a group made entirely of pinned blocks would read as "cells to
  # score, none scored" and fail for a reason that is not a fault.
  list(rows = rows, mismatches = mismatches, n_with_truth = n_with_truth,
       n_undefined = length(undefined),
       n_scoreable = n_with_truth - length(undefined))
}

# The per-family truth band, read the way tol_for reads a tolerance: an
# unmeasured band STOPS rather than being compared against. NA compares false, so
# gating against it would pass every family silently.
truth_band <- function(family) {
  band <- TOL_TRUTH_BAND[[family]]
  if (is.null(band))
    stop(sprintf("truth_band: no TOL_TRUTH_BAND entry for family `%s`", family),
         call. = FALSE)
  if (is.na(band))
    stop(sprintf(paste0("truth_band: TOL_TRUTH_BAND$%s has not been measured yet. ",
                        "Run compare.R --dev-floor and write the number into ",
                        "grid/tol.R with its measurement."), family),
         call. = FALSE)
  band
}

# Estimate vectors aligned to the truth vector, one extractor per gated quantity.
# beta is reordered into the truth's coefficient order BY NAME and returns NULL
# when the two name sets cannot be matched -- never positional. stddev and corr
# use the same group-name flattening on both sides that stddevs_of/corrs_of
# impose, so the truth is read in the order the records are.
truth_extract <- list(
  beta = function(rec, cell) {
    if (length(rec$beta) != length(rec$coef_names)) return(NULL)
    perm <- coef_perm(cell[["truth"]][["coef_names"]], rec$coef_names)
    if (is.null(perm)) return(NULL)
    as.numeric(rec$beta)[perm]
  },
  stddev = function(rec, cell) stddevs_of(rec),
  corr = function(rec, cell) corrs_of(rec))

truth_vector <- list(
  beta = function(cell) as.numeric(cell[["truth"]][["beta"]]),
  stddev = function(cell) stddevs_of(list(varcomp = cell[["truth"]][["varcomp"]])),
  corr = function(cell) corrs_of(list(varcomp = cell[["truth"]][["varcomp"]])))

# Ceil-to-one-significant-figure, the house rule every measured band in tol.R
# follows.
ceil1 <- function(x) {
  if (!is.finite(x) || x <= 0) return(x)
  e <- floor(log10(x)); ceiling(x / 10^e) * 10^e
}

# The deliberate near-zero truth coordinates the grid generates: every |truth|
# coordinate below 0.1 on the `nearzero` and `boundary` cells, across beta, the
# random-effect standard deviations and the correlations. TOL$truth_floor is sized
# from this set, and both the measurement that pins it and the report's own probe
# read it here so the floor a report prints is the floor the gate uses.
near_zero_truth_coords <- function(cells) {
  small <- c()
  for (cl in cells) {
    if (!identical(cl[["regime"]], "nearzero") &&
        !identical(cl[["regime"]], "boundary")) next
    if (is.null(cl[["truth"]])) next
    v <- abs(c(truth_vector$beta(cl), truth_vector$stddev(cl), truth_vector$corr(cl)))
    small <- c(small, v[v < 0.1])
  }
  small
}

# The one dispersion-scale number a cell has a true value for. `sigma` and
# `dispersion` are mutually exclusive in the manifest (a gaussian/LMM cell has
# the residual SD, a Gamma cell the dispersion φ). Every engine's record `sigma`
# slot holds an SD, √φ on Gamma, so a dispersion truth is compared as √φ;
# `nb_theta` is its own field on both sides.
truth_scale_pairs <- function(cell, rec) {
  tru <- cell[["truth"]]
  out <- list()
  s <- if (!is.null(tru[["sigma"]])) tru[["sigma"]]
       else if (!is.null(tru[["dispersion"]])) sqrt(tru[["dispersion"]])
  if (!is.null(s) && !is.null(rec$sigma)) out$sigma <- c(as.numeric(rec$sigma), as.numeric(s))
  if (!is.null(tru[["nb_theta"]]) && !is.null(rec$nb_theta))
    out$nb_theta <- c(as.numeric(rec$nb_theta), as.numeric(tru[["nb_theta"]]))
  out
}
