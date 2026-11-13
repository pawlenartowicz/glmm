#!/usr/bin/env Rscript
# The R front end of the input port gate: every input of inputs.json made
# through the R port, each written to stdout as one JSON line: the input's
# id, status ("accepted" or "refused"), message (the refusal text, "" when
# accepted) and warnings (the text of every warning the call raised, in
# order). Mirrors input_ports.py input for input: change together. Each
# builder below passes its input in this port's own idiom (R integers, an
# empty column name, subset= with an NA); the formula, family and table come
# from inputs.json.
#
# Run from the repository root:
#   Rscript validation/tools/input_ports/input_ports.R > input_r.jsonl
suppressMessages({ library(fastglmm); library(jsonlite) })
here <- dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
SPEC <- jsonlite::fromJSON(file.path(here, "inputs.json"), simplifyDataFrame = FALSE)
DATA <- as.data.frame(SPEC$data)
SUBSET <- SPEC$subset

fit_input <- function(row, data = DATA, ...) {
  fastglmm(row$formula, data, family = row$family, ...)
}

# weights=, offset= and subset= are evaluated by fastglmm() in `data`, then in
# its caller's frame, so they are written into each call rather than passed
# through fit_input()'s `...`.
BUILD <- list(
  nagq_numpy_integer = function(row) fit_input(row, nAGQ = 5L),
  init_theta_numpy_integer = function(row) fit_input(row, init.theta = 3L),
  dispersion_numpy_integer = function(row) fit_input(row, dispersion = 2L),
  column_name_not_a_string = function(row) {
    data <- cbind(DATA, DATA["x"])
    names(data)[ncol(data)] <- ""
    fit_input(row, data)
  },
  subset_with_na = function(row) {
    fastglmm(row$formula, DATA, family = row$family, subset = SUBSET)
  },
  weights_with_na = function(row) {
    fastglmm(row$formula, DATA, family = row$family, weights = replace(rep(1, nrow(DATA)), 4, NA))
  },
  offset_with_na = function(row) {
    fastglmm(row$formula, DATA, family = row$family, offset = replace(rep(0, nrow(DATA)), 4, NA))
  },
  interaction_slope = function(row) fit_input(row))

ids <- vapply(SPEC$inputs, `[[`, "", "id")
if (!setequal(names(BUILD), ids)) {
  stop("builders and inputs.json differ: ",
       paste(sort(union(setdiff(names(BUILD), ids), setdiff(ids, names(BUILD)))),
             collapse = ", "), call. = FALSE)
}

for (row in SPEC$inputs) {
  status <- "accepted"
  msg <- ""
  texts <- character(0)
  # Outside the tryCatch, so a warning raised before a refusal is kept.
  withCallingHandlers(
    tryCatch(BUILD[[row$id]](row), error = function(e) {
      status <<- "refused"
      msg <<- conditionMessage(e)
    }),
    warning = function(w) {
      texts <<- c(texts, conditionMessage(w))
      invokeRestart("muffleWarning")
    })
  rec <- list(id = row$id, status = status, message = enc2utf8(msg),
              warnings = as.list(enc2utf8(texts)))
  writeLines(enc2utf8(as.character(jsonlite::toJSON(rec, auto_unbox = TRUE))),
             useBytes = TRUE)
}
