#!/usr/bin/env Rscript
# HC2 references -> validation/goldens/inference/cr2/<name>.json, one per
# `inference_references` entry with `generator = "goldens_hc2.R"`.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's HC2 SE and
# Bell-McCaffrey df are validated against (gate G4 of spec 2-dcr). On any glmm
# disagreement, glmm is presumed wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN OVERWRITES EVERY REFERENCE IT OWNS. Use VALIDATION_ONLY for anything
# short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_hc2.R
#
# Per entry: stats::lm on the entry's formula with prior weights `weights`, then
# clubSandwich's vcovCR(type = "CR2", target = 1 / w, inverse_var = FALSE) with one
# cluster per row (the working model diag(1/w) of a fit with weights w; clubSandwich's
# default target on lm is the identity) and coef_test(test = "Satterthwaite") on it.
# The JSON carries coefficients, beta, se, df and the HC2 vcov. Before writing, the run checks the SE and df against
# tools/cr2/efficiency_lib.R's cr2() with one-row clusters (working model
# diag(1/w)) and stops on a relative gap above 1e-10.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
extra <- assert_extra_versions("clubSandwich")
source(file.path(suite_dir, "tools", "cr2", "efficiency_lib.R"))

GENERATOR <- "goldens_hc2.R"

hc2_reference <- function(entry) {
  data <- read_entry_data(entry)
  data$.w <- data[[entry$weights]]
  m <- stats::lm(as.formula(entry$r_formula), data = data, weights = .w)
  n <- nrow(data)
  vc <- clubSandwich::vcovCR(m, cluster = seq_len(n), type = "CR2", target = 1 / data$.w,
                             inverse_var = FALSE)
  ct <- clubSandwich::coef_test(m, vcov = vc, test = "Satterthwaite")
  X <- stats::model.matrix(m)
  ours <- lapply(seq_len(ncol(X)), function(j)
    cr2(X, data$y, unname(coef(m)), 0, 1 / data$.w, as.list(seq_len(n)), j))
  gap <- max(abs(vapply(ours, `[[`, 0, "se") / ct$SE - 1), abs(vapply(ours, `[[`, 0, "df") / ct$df_Satt - 1))
  cat(sprintf("%s: cr2() against clubSandwich, largest relative gap %.2e\n", entry$name, gap))
  if (gap > 1e-10) stop(entry$name, ": cr2() and clubSandwich disagree", call. = FALSE)
  check_finite(list(provenance = provenance(GENERATOR, extra = extra),
                    coefficients = colnames(X), beta = unname(coef(m)), se = ct$SE,
                    df = ct$df_Satt, vcov = unname(as.matrix(vc))), entry$name)
}

for (entry in Filter(function(e) identical(e$generator, GENERATOR), select_entries("cr2"))) {
  write_reference(hc2_reference(entry),
                  file.path(suite_dir, "goldens", "inference", "cr2", paste0(entry$name, ".json")))
  cat(sprintf("wrote %s\n", entry$name))
}
