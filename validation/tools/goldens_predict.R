#!/usr/bin/env Rscript
# Prediction references -> validation/goldens/inference/predict/<name>.json,
# one per `kind = "predict"` entry of the manifest's `inference_references` block.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's
# predictions are validated against. On any glmm disagreement, glmm is presumed
# wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN REFITS AND OVERWRITES EVERY REFERENCE IT OWNS (every `predict` entry).
# Use VALIDATION_ONLY for anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_predict.R
#
# An entry has `members` in the anova member shape (tools/goldens_anova.R's
# header): one stats::lm or stats::glm fit, or an lme4 fit first and its
# glmmTMB refit second. Its `newdata` is a CSV path relative to validation/
# (written by tools/prep/gen_predict_data.R), read with the entry's `factors`
# applied as to the fit data; it has no response column.
#
# The JSON carries, from the first member:
#   - `coef` (lm, glm) or `fixef`, as `names` and `values`; on a mixed fit
#     `ranef`, per grouping of ranef(): `grouping`, `levels`, `terms` and
#     `values` (levels x terms);
#   - `predict`, one value per newdata row: `population` (re.form = NA) and,
#     on a mixed fit, `conditional` (re.form = NULL with allow.new.levels =
#     TRUE, so an unseen level contributes zero), each with `link` and
#     `response` (on an lm both are predict.lm's one scale);
#   - `unseen_rows`, the 0-based rows with a level no grouping of the fit has
#     seen, and `default_errors`, per such row its `row` and the `error` that
#     lme4's predict() raises on it alone under its default allow.new.levels =
#     FALSE. Every other row predicts under the default, else the run stops.
#     Both are empty on lm and glm;
#   - on a mixed fit, `se_fit_lme4` and `se_fit_glmmtmb` from predict(se.fit =
#     TRUE) of each member: per mode (`population`, `conditional`, as in
#     `predict`) and scale (`link`, `response`), the `fit` and `se_fit`
#     vectors, or the `error` message where the engine refuses that call.
#     On a glmer fit every lme4 call warns that its SE is an approximation
#     (lme4 2.0.6 warns on every se.fit call of a model that is not an LMM);
#     that one message, matched in full, is allowed there only and recorded
#     as `warning` beside that call's `fit` and `se_fit`;
#   - on a mixed fit, `condvar`, per grouping of ranef(m, condVar = TRUE):
#     `grouping`, `levels`, `terms`, `postvar` (its postVar array, terms x
#     terms x levels) and `condsd`, the rows of that grouping in
#     as.data.frame(ranef(m, condVar = TRUE)), each with `level`, `term` and
#     `condsd`.
# Every number written is finite, else the run stops. A warning from any fit
# or call stops the run.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
EXTRA_PKGS <- assert_extra_versions("glmmTMB")

GENERATOR <- "goldens_predict.R"
out_dir <- file.path(suite_dir, "goldens", "inference", "predict")

finite <- function(x, what) {
  if (!is.numeric(x) || !all(is.finite(x))) stop(sprintf("%s is not finite", what), call. = FALSE)
  unname(x)
}

# f(), or list(error = <message>) when it raises an error. A warning stops the
# run: its handler sits in the outer tryCatch, so the error it raises is not
# caught as a refusal (in one tryCatch the error handler would wrap the
# warning handler and catch it). f() returns the engine's raw value and the
# caller checks it afterwards, for the same reason.
value_or_error <- function(what, f) {
  tryCatch(tryCatch(f(), error = function(e) list(error = conditionMessage(e))),
           warning = function(w) stop(sprintf("%s: %s", what, conditionMessage(w)), call. = FALSE))
}

named <- function(x, what) list(names = I(names(x)), values = I(finite(x, what)))

# Both scales of predict() at random-effect form `re`.
both_scales <- function(m, nd, re, what) {
  list(link = finite(predict(m, nd, re.form = re, allow.new.levels = TRUE, type = "link"),
                     paste(what, "link")),
       response = finite(predict(m, nd, re.form = re, allow.new.levels = TRUE,
                                 type = "response"), paste(what, "response")))
}

GLMER_SE_WARNING <- paste("se.fit computation uses an approximation to estimate the",
                          "sampling distribution of the parameters")

# predict(se.fit = TRUE) per mode and scale, the same call on lme4 and glmmTMB.
# `allowed`, when set, is the one warning message a call may raise; it is
# recorded in the call's `warning` field. Any other warning stops the run.
se_fit_of <- function(m, nd, what, allowed = NULL) {
  modes <- list(population = NA, conditional = NULL)
  out <- list()
  for (mode in names(modes)) {
    for (type in c("link", "response")) {
      at <- sprintf("%s %s %s", what, mode, type)
      warned <- NULL
      p <- value_or_error(at, function() {
        withCallingHandlers(
          predict(m, nd, re.form = modes[[mode]], allow.new.levels = TRUE, type = type,
                  se.fit = TRUE),
          warning = function(w) {
            if (!is.null(allowed) && identical(conditionMessage(w), allowed)) {
              warned <<- conditionMessage(w)
              invokeRestart("muffleWarning")
            }
          })
      })
      out[[mode]][[type]] <- if (!is.null(p$error)) p else
        c(list(fit = finite(p$fit, at), se_fit = finite(p$se.fit, at)),
          if (!is.null(warned)) list(warning = warned))
    }
  }
  out
}

# 0-based newdata rows whose level of some grouping of `m` is not a level of
# that grouping in the fit. Each grouping's name in getME(m, "flist") is the
# expression lme4 built its factor from (`cask:batch` for `batch/cask`), so it
# is evaluated on the newdata the same way.
unseen_rows <- function(m, nd) {
  flist <- lme4::getME(m, "flist")
  seen <- rep(TRUE, nrow(nd))
  for (key in names(flist)) {
    g <- as.character(eval(str2lang(key), nd))
    seen <- seen & g %in% levels(flist[[key]])
  }
  which(!seen) - 1L
}

ranef_blocks <- function(rr) {
  lapply(names(rr), function(g) {
    list(grouping = g, levels = I(rownames(rr[[g]])), terms = I(names(rr[[g]])),
         values = finite(unname(as.matrix(rr[[g]])), paste("ranef", g)))
  })
}

condvar_blocks <- function(rr, what) {
  df <- as.data.frame(rr)
  lapply(names(rr), function(g) {
    rows <- df[df$grpvar == g, ]
    list(grouping = g, levels = I(rownames(rr[[g]])), terms = I(names(rr[[g]])),
         postvar = finite(attr(rr[[g]], "postVar"), paste(what, g, "postVar")),
         condsd = data.frame(level = as.character(rows$grp), term = as.character(rows$term),
                             condsd = finite(rows$condsd, paste(what, g, "condsd"))))
  })
}

mixed_reference <- function(entry, data, nd) {
  funs <- vapply(entry$members, `[[`, "", "fun")
  if (length(funs) != 2L || !funs[1] %in% c("lme4::lmer", "lme4::glmer") ||
      funs[2] != "glmmTMB::glmmTMB") {
    stop(sprintf("%s: members are %s", entry$name, paste(funs, collapse = ", ")), call. = FALSE)
  }
  m <- fit_member(entry$members[[1]], data)
  tmb <- fit_member(entry$members[[2]], data)
  rr <- lme4::ranef(m, condVar = TRUE)
  unseen <- unseen_rows(m, nd)
  default_errors <- list()
  for (i in seq_len(nrow(nd))) {
    at <- sprintf("%s row %d", entry$name, i - 1L)
    res <- value_or_error(at, function() predict(m, nd[i, , drop = FALSE]))
    if (is.list(res) != ((i - 1L) %in% unseen)) {
      stop(sprintf("%s: lme4's default %s, but the row is %s", at,
                   if (is.list(res)) "refuses it" else "predicts it",
                   if ((i - 1L) %in% unseen) "unseen" else "seen"), call. = FALSE)
    }
    if (is.list(res)) default_errors[[length(default_errors) + 1L]] <- list(row = i - 1L,
                                                                            error = res$error)
  }
  list(fixef = named(lme4::fixef(m), paste(entry$name, "fixef")),
       ranef = ranef_blocks(rr),
       predict = list(population = both_scales(m, nd, NA, paste(entry$name, "population")),
                      conditional = both_scales(m, nd, NULL, paste(entry$name, "conditional"))),
       unseen_rows = I(unseen),
       default_errors = default_errors,
       se_fit_lme4 = se_fit_of(m, nd, paste(entry$name, "lme4"),
                               allowed = if (funs[1] == "lme4::glmer") GLMER_SE_WARNING),
       se_fit_glmmtmb = se_fit_of(tmb, nd, paste(entry$name, "glmmTMB")),
       condvar = condvar_blocks(rr, entry$name))
}

fixed_reference <- function(entry, data, nd) {
  mem <- entry$members[[1]]
  if (length(entry$members) != 1L || !mem$fun %in% c("stats::lm", "stats::glm")) {
    stop(sprintf("%s: a fixed-effect entry needs one stats::lm or stats::glm member",
                 entry$name), call. = FALSE)
  }
  m <- fit_member(mem, data)
  link <- if (mem$fun == "stats::lm") predict(m, nd) else predict(m, nd, type = "link")
  list(coef = named(coef(m), paste(entry$name, "coef")),
       predict = list(population = list(
         link = finite(link, paste(entry$name, "link")),
         response = finite(predict(m, nd, type = "response"), paste(entry$name, "response")))),
       unseen_rows = I(integer()),
       default_errors = list())
}

for (entry in select_entries("predict")) {
  cat(sprintf("%s ...\n", entry$name))
  data <- read_entry_data(entry)
  nd <- read_entry_data(modifyList(entry, list(data = entry$newdata)))
  build <- if (length(entry$members) == 1L) fixed_reference else mixed_reference
  ref <- withCallingHandlers(build(entry, data, nd), warning = function(w) {
    stop(sprintf("%s: %s", entry$name, conditionMessage(w)), call. = FALSE)
  })
  obj <- c(list(provenance = provenance(GENERATOR, extra = EXTRA_PKGS)), ref)
  write_reference(obj, file.path(out_dir, paste0(entry$name, ".json")))
}
