#!/usr/bin/env Rscript
# The input port gate's comparator: the lines input_ports.py and
# input_ports.R wrote, compared input by input.
#
#   Rscript compare_input_ports.R <python.jsonl> <r.jsonl>
#
# The expected inputs are the ids of inputs.json beside this file. The gate
# fails on:
#   - an id that is not in inputs.json;
#   - an id of inputs.json missing from a file;
#   - an id written twice by one front end;
#   - a different status (accepted or refused);
#   - a different message (the refusal text);
#   - a different warnings list: its length, or any text, in order.
# It prints every difference with both sides, then the number of inputs
# compared, and fails when that number is not the number of ids in
# inputs.json. Only the two ports' answers are compared, not an expected
# outcome: a change that moves both ports the same way passes here.
#
# Base R and jsonlite only: the CI port-gate job installs no statistics
# package.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: Rscript compare_input_ports.R <python.jsonl> <r.jsonl>", call. = FALSE)
}
here <- dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
spec <- jsonlite::fromJSON(file.path(here, "inputs.json"), simplifyDataFrame = FALSE)
expected <- vapply(spec$inputs, `[[`, "", "id")

diffs <- character(0)
note <- function(...) diffs <<- c(diffs, sprintf(...))

read_records <- function(path, side) {
  lines <- readLines(path, warn = FALSE, encoding = "UTF-8")
  recs <- lapply(lines[nzchar(lines)], jsonlite::fromJSON)
  ids <- vapply(recs, `[[`, "", "id")
  for (id in setdiff(ids, expected)) note("%s: %s wrote an input not in inputs.json", id, side)
  for (id in unique(ids[duplicated(ids)])) note("%s: %s wrote it twice", id, side)
  for (id in setdiff(expected, ids)) note("%s: missing from %s", id, side)
  names(recs) <- ids
  recs[ids %in% expected & !(ids %in% ids[duplicated(ids)])]
}

py <- read_records(args[[1]], "Python")
r <- read_records(args[[2]], "R")
both <- intersect(intersect(expected, names(py)), names(r))
for (id in both) {
  a <- py[[id]]
  b <- r[[id]]
  if (!identical(a$status, b$status)) {
    note("%s: status differs: Python %s, R %s", id, a$status, b$status)
  }
  if (!identical(a$message, b$message)) {
    note("%s: message differs:\n  Python: %s\n  R:      %s", id, a$message, b$message)
  }
  wa <- as.character(unlist(a$warnings))
  wb <- as.character(unlist(b$warnings))
  if (!identical(wa, wb)) {
    note("%s: warnings differ:\n  Python: [%s]\n  R:      [%s]", id,
         paste(wa, collapse = " | "), paste(wb, collapse = " | "))
  }
}

if (length(diffs)) writeLines(diffs)
cat(sprintf("compared %d inputs, %d differences\n", length(both), length(diffs)))
if (length(diffs) > 0L || length(both) != length(expected) || length(both) == 0L) {
  quit(status = 1L)
}
