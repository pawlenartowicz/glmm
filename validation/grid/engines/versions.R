# Pinned-oracle version assertions. versions.json is the single source of truth;
# every R engine script and install_oracles.R read the pin through here, so a
# bump is one file edit and cannot drift between consumers.
#
# TWO SPELLINGS OF ONE VERSION. CRAN writes lme4 "2.0-6" and GLMMadaptive
# "0.9-7"; packageVersion() renders both with a dot ("2.0.6", "0.9.7"), because
# R's numeric_version class accepts either separator and prints the dot form.
# versions.json keeps the CRAN/spec spelling, `norm_version` maps both sides to
# the dot form for comparison, and EVERY RECORDED version -- the JSONL
# `engine_version`, run_meta's `engine_version`, the run-directory name -- is the
# string packageVersion() actually returned. Nothing rewrites it back.

grid_versions <- function(grid_dir) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("grid_versions: jsonlite is required to read versions.json", call. = FALSE)
  }
  # jsonlite's simplifyVector collapses a flat JSON object of scalars to a
  # list, not a named character vector, so unlist() does the collapsing here.
  # A NAMED CHARACTER VECTOR is what every caller of grid_versions() indexes
  # with `V[["lme4"]]`.
  unlist(jsonlite::fromJSON(file.path(grid_dir, "versions.json"), simplifyVector = TRUE))
}

norm_version <- function(v) gsub("-", ".", as.character(v), fixed = TRUE)

# Stops with BOTH versions printed -- a silent fallback to whatever is
# installed would put an unpinned oracle's numbers in a committed run directory.
# `source` names where `want` came from, so a call made by hand with a literal
# version cannot print a sentence claiming versions.json holds that literal.
assert_pkg_version <- function(pkg, want, source = "grid/versions.json") {
  have <- tryCatch(as.character(utils::packageVersion(pkg)),
                   error = function(e) NA_character_)
  if (is.na(have)) {
    stop(sprintf("%s is not installed; %s pins %s. Run grid/install_oracles.R.",
                 pkg, source, want), call. = FALSE)
  }
  if (!identical(norm_version(have), norm_version(want))) {
    stop(sprintf("%s version mismatch: %s pins %s, installed is %s. Run grid/install_oracles.R.",
                 pkg, source, want, have), call. = FALSE)
  }
  invisible(have)
}
