#!/usr/bin/env Rscript
# One command to bring a fresh machine to grid/versions.json's pinned R oracle set.
#
# SIDE EFFECT, read before running: this installs into the FIRST entry of
# .libPaths() (the user library) and REPLACES whatever version of lme4 / glmmTMB /
# GLMMadaptive is there. Nothing else on the machine is consulted about that.
#
# remotes::install_version(pkg, version) is used for all three even where the pin
# happens to be CRAN's current version: it pins exactly, resolving against the
# current version first and src/contrib/Archive/ afterwards, so this script keeps
# reaching the same set after the next CRAN release.
#
# The R and Julia pins in versions.json are NOT asserted here -- the language
# runtime is not the estimator; run.sh records them in run_meta.json.
# The Julia package pin lives in grid/Project.toml + grid/Manifest.toml and is
# reached with `julia --project=<grid> -e 'using Pkg; Pkg.instantiate()'`.

REPO <- "https://cloud.r-project.org"
lib <- .libPaths()[1]
here <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))

for (boot in c("jsonlite", "remotes")) {
  if (!requireNamespace(boot, quietly = TRUE)) {
    cat(sprintf("installing bootstrap package %s into %s\n", boot, lib))
    utils::install.packages(boot, lib = lib, repos = REPO)
  }
}

source(file.path(here, "engines", "versions.R"))
V <- grid_versions(here)

# lme4 2.0-6's DESCRIPTION pins `LinkingTo: Rcpp (>= 1.1.1-1.1)`; R's version
# comparison treats "-" and "." as equal separators, so this is the same floor
# as "1.1.1.1.1". Rcpp itself is not one of the pinned oracles in versions.json
# -- it is a build tool for lme4 -- so it is brought current rather than pinned.
RCPP_FLOOR <- "1.1.1.1.1"
have_rcpp <- tryCatch(utils::packageVersion("Rcpp"), error = function(e) NA)
if (is.na(have_rcpp) || have_rcpp < RCPP_FLOOR) {
  cat(sprintf("Rcpp %s < %s required by lme4 2.0-6, installing current Rcpp into %s\n",
              if (is.na(have_rcpp)) "(not installed)" else as.character(have_rcpp),
              RCPP_FLOOR, lib))
  utils::install.packages("Rcpp", lib = lib, repos = REPO)
}

PKGS <- c("lme4", "glmmTMB", "GLMMadaptive")
for (pkg in PKGS) {
  want <- V[[pkg]]
  have <- tryCatch(as.character(utils::packageVersion(pkg)),
                   error = function(e) NA_character_)
  if (!is.na(have) && identical(norm_version(have), norm_version(want))) {
    cat(sprintf("%-13s %s already installed\n", pkg, have))
    next
  }
  cat(sprintf("%-13s installing %s (had %s) into %s\n", pkg, want,
              if (is.na(have)) "nothing" else have, lib))
  remotes::install_version(pkg, version = want, lib = lib,
                           repos = REPO, upgrade = "never")
}

# Assert rather than trust the installer: install_version can succeed and still
# leave an older copy earlier on .libPaths().
for (pkg in PKGS) {
  cat(sprintf("%-13s pinned %-8s installed %s  OK\n", pkg, V[[pkg]],
              assert_pkg_version(pkg, V[[pkg]])))
}
cat(sprintf("R runtime: pinned %s, running %s (recorded, not asserted)\n",
            V[["R"]], as.character(getRversion())))
