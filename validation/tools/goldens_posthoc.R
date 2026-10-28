#!/usr/bin/env Rscript
# Pairwise comparison references -> validation/goldens/inference/posthoc/<name>.json,
# one per `kind = "posthoc"` entry of the manifest's `inference_references` block.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's pairwise
# comparisons are validated against. On any glmm disagreement, glmm is presumed
# wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN REFITS AND OVERWRITES EVERY REFERENCE IT OWNS (every `posthoc` entry).
# Use VALIDATION_ONLY for anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_posthoc.R
#
# Each entry has one member in the anova member shape (see goldens_anova.R),
# fitted by fit_member, and names `factor`, the compared factor, and `by`, a
# factor whose levels split the comparisons into families, or null. The script
# runs emmeans(m, pairwise ~ factor | by, adjust = "none",
# lmer.df = "satterthwaite"): pbkrtest is installed, so without lmer.df emmeans
# would give an lmer fit Kenward-Roger df. Every numeric covariate sits at its
# mean (emmeans' default reference grid). The Tukey p-values are the same
# contrasts summarised with adjust = "tukey", so the two p columns share rows.
#
# The JSON has `emmeans` (rows: `level`, `by`, `estimate`, `SE`, `df`) and
# `contrasts` (rows: `contrast`, `by`, `estimate`, `SE`, `df`, `statistic`,
# `statistic_name`, `p_none`, `p_tukey`), on the link scale. `by` is the row's
# by level, null without a by; a contrast's family is all rows of its by level.
# On a z row (glm, glmer) `df` is infinite and written as null. On log and logit
# links `response` holds the contrast rows again with type = "response": `ratio`
# (an odds ratio or a rate ratio) and its delta-method `SE` in place of
# `estimate` and `SE`, with emmeans' "a / b" contrast labels; the statistic and
# p-values are emmeans' own for that summary (it tests on the link scale, so they
# match `contrasts` to round-off).
#
# An entry may also name `confint_adjust`, a list of emmeans adjustments (e.g.
# ["none", "bonferroni", "tukey"]). The JSON then holds `confint`, the rows of
# confint(<contrasts>, adjust = <each>, level = 0.95) for each adjustment in
# turn, in the listed order (rows: `adjust`, `contrast`, `by`, `estimate`, `SE`,
# `df`, `lower`, `upper`, on the link scale), and on log and logit links
# `confint_response`, the same with type = "response": `ratio`, its `SE`, and
# the bounds back-transformed by exp. A Bonferroni or Tukey family is all rows of
# one by level, as for the p-values; the run stops unless emmeans reports the
# adjustment it was asked for. Without the field neither block is written.
#
# A contrast label is emmeans' own: on levels that read as numbers it prefixes
# the factor name ("period1 - period2"). Any warning, from the fit or from
# emmeans, stops the run, and so does any message but emmeans' note that the
# factor is involved in an interaction. Every number written is finite, else the
# run stops; only `df` on a z row and `by` without a by factor are null.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
EXTRA_PKGS <- assert_extra_versions(c("emmeans", "pbkrtest"))

GENERATOR <- "goldens_posthoc.R"
out_dir <- file.path(suite_dir, "goldens", "inference", "posthoc")

# The by level of each row of summary `s`, or NA (null in the JSON) without a by.
by_levels <- function(s, by) {
  if (is.null(by)) return(rep(NA_character_, nrow(s)))
  as.character(s[[by]])
}

# Infinite df (asymptotic rows) becomes NA, written as null.
finite_df <- function(df) ifelse(is.finite(df), df, NA_real_)

# The contrast rows of `none` and `tukey`, two summaries of one contrast set,
# with the estimate column named `est_key` in the JSON.
contrast_rows <- function(name, none, tukey, by, est_key) {
  stat_col <- intersect(c("t.ratio", "z.ratio"), names(none))
  if (length(stat_col) != 1L || !identical(none$contrast, tukey$contrast)) {
    stop(sprintf("%s: unexpected contrast summary columns %s", name,
                 paste(names(none), collapse = ", ")), call. = FALSE)
  }
  est <- none[[attr(none, "estName")]]
  bys <- by_levels(none, by)
  lapply(seq_len(nrow(none)), function(i) {
    row <- list(contrast = as.character(none$contrast[i]), by = bys[i], estimate = est[i],
                SE = none$SE[i], df = finite_df(none$df[i]), statistic = none[[stat_col]][i],
                statistic_name = substr(stat_col, 1L, 1L), p_none = none$p.value[i],
                p_tukey = tukey$p.value[i])
    names(row)[names(row) == "estimate"] <- est_key
    row
  })
}

# The 95% interval rows of contrast set `con` for each adjustment in `adjusts`,
# with the estimate column named `est_key` in the JSON. emmeans names the bounds
# lower.CL/upper.CL on t rows and asymp.LCL/asymp.UCL on z rows; both become
# `lower`/`upper`.
confint_rows <- function(name, con, adjusts, by, est_key, type) {
  unlist(lapply(adjusts, function(a) {
    s <- confint(con, adjust = a, level = 0.95, type = type)
    mesg <- attr(s, "mesg")
    ok <- if (a == "none") !any(grepl("^Conf-level adjustment", mesg)) else
      any(grepl(sprintf("^Conf-level adjustment: %s method", a), mesg))
    if (!ok) {
      stop(sprintf("%s: confint did not apply adjust = \"%s\": %s", name, a,
                   paste(mesg, collapse = " | ")), call. = FALSE)
    }
    cl <- attr(s, "clNames")
    est <- s[[attr(s, "estName")]]
    bys <- by_levels(s, by)
    lapply(seq_len(nrow(s)), function(i) {
      row <- list(adjust = a, contrast = as.character(s$contrast[i]), by = bys[i],
                  estimate = est[i], SE = s$SE[i], df = finite_df(s$df[i]),
                  lower = s[[cl[1]]][i], upper = s[[cl[2]]][i])
      names(row)[names(row) == "estimate"] <- est_key
      row
    })
  }), recursive = FALSE)
}

posthoc_reference <- function(entry) {
  data <- read_entry_data(entry)
  mems <- entry$members
  if (length(mems) != 1L) stop(sprintf("%s: needs one member", entry$name), call. = FALSE)
  m <- fit_member(mems[[1]], data)
  spec <- if (is.null(entry$by)) {
    reformulate(entry$factor, response = "pairwise")
  } else {
    reformulate(sprintf("%s | %s", entry$factor, entry$by), response = "pairwise")
  }
  res <- emmeans::emmeans(m, spec, adjust = "none", lmer.df = "satterthwaite")
  emm <- summary(res$emmeans)
  bys <- by_levels(emm, entry$by)
  out <- list(
    emmeans = lapply(seq_len(nrow(emm)), function(i) {
      list(level = as.character(emm[[entry$factor]][i]), by = bys[i],
           estimate = emm[[attr(emm, "estName")]][i], SE = emm$SE[i],
           df = finite_df(emm$df[i]))
    }),
    contrasts = contrast_rows(entry$name, summary(res$contrasts),
                              summary(res$contrasts, adjust = "tukey"), entry$by, "estimate")
  )
  if (mems[[1]]$link %in% c("log", "logit")) {
    out$response <- contrast_rows(entry$name, summary(res$contrasts, type = "response"),
                                  summary(res$contrasts, type = "response", adjust = "tukey"),
                                  entry$by, "ratio")
  }
  adjusts <- unlist(entry$confint_adjust)
  if (!is.null(adjusts)) {
    out$confint <- confint_rows(entry$name, res$contrasts, adjusts, entry$by, "estimate",
                                "link")
    if (mems[[1]]$link %in% c("log", "logit")) {
      out$confint_response <- confint_rows(entry$name, res$contrasts, adjusts, entry$by,
                                           "ratio", "response")
    }
  }
  out
}

# Stops unless every value of every row of `out` is finite, leaving out only
# `df` on z rows (all rows of one fit share the statistic) and `by` when the
# entry has no by factor.
check_posthoc <- function(out, entry) {
  nullable <- c(if (identical(out$contrasts[[1]]$statistic_name, "z")) "df",
                if (is.null(entry$by)) "by")
  for (block in names(out)) {
    for (i in seq_along(out[[block]])) {
      row <- out[[block]][[i]]
      check_finite(row[setdiff(names(row), nullable)], entry$name, sprintf("/%s/%d", block, i))
    }
  }
  invisible(out)
}

# emmeans 2.0.4's note when the compared factor is in an interaction term.
INTERACTION_NOTE <- "NOTE: Results may be misleading due to involvement in interactions\n"

for (entry in select_entries("posthoc")) {
  cat(sprintf("%s ...\n", entry$name))
  obj <- withCallingHandlers(posthoc_reference(entry),
    warning = function(w) {
      stop(sprintf("%s: %s", entry$name, conditionMessage(w)), call. = FALSE)
    },
    message = function(mm) {
      if (!identical(conditionMessage(mm), INTERACTION_NOTE)) {
        stop(sprintf("%s: %s", entry$name, trimws(conditionMessage(mm))), call. = FALSE)
      }
      invokeRestart("muffleMessage")
    })
  check_posthoc(obj, entry)
  write_reference(c(list(provenance = provenance(GENERATOR, extra = EXTRA_PKGS)), obj),
                  file.path(out_dir, paste0(entry$name, ".json")))
}
