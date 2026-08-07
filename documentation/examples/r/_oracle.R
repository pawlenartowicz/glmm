# Shared helper for the worked-examples recipes: load a frozen golden JSON
# from validation/goldens/ and print a pass/fail comparison against a fitted
# value. Not part of the fastglmm package -- a doc-build convenience so every
# recipe's printed comparison numbers come from the golden file, never typed
# by hand.
#
# Tolerance constants mirror validation/tol.R (kept as constants here rather
# than sourced, since tol.R defines a much larger list tied to the harness's
# own loader).

.oracle_dir <- function() {
  # Resolve relative to this file's own location so the recipe scripts can be
  # run from any working directory.
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
  this_file <- if (length(file_arg)) file_arg else sys.frame(1)$ofile
  file.path(dirname(normalizePath(this_file)), "..", "..", "..", "validation", "goldens")
}

load_golden <- function(name) {
  jsonlite::fromJSON(file.path(.oracle_dir(), paste0(name, ".json")))
}

# validation/tol.R
TOL_BETA_REL <- 1e-3
TOL_SE_REL <- 1e-3
TOL_SE_HESSIAN_REL <- 1e-3
TOL_STDDEV_REL <- 1e-3
TOL_LOGLIK_ABS_LMM <- 2e-6
TOL_LOGLIK_ABS_GLMM <- 1e-3

check_rel <- function(label, got, ref, tol) {
  err <- if (ref != 0) abs(got - ref) / abs(ref) else abs(got - ref)
  status <- if (err <= tol) "PASS" else "FAIL"
  cat(sprintf("  %-28s got=%-18.10g ref=%-18.10g rel_err=%.3g  tol=%.3g  [%s]\n",
              label, got, ref, err, tol, status))
  invisible(err <= tol)
}

check_abs <- function(label, got, ref, tol) {
  err <- abs(got - ref)
  status <- if (err <= tol) "PASS" else "FAIL"
  cat(sprintf("  %-28s got=%-18.10g ref=%-18.10g abs_err=%.3g  tol=%.3g  [%s]\n",
              label, got, ref, err, tol, status))
  invisible(err <= tol)
}
