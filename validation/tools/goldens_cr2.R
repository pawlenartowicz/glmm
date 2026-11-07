#!/usr/bin/env Rscript
# Dense CR2 references at a given working model -> validation/goldens/inference/cr2/<name>.json,
# one per `inference_references` entry with `generator = "goldens_cr2.R"`.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's CR2 kernel
# is validated against (gate G2 of spec 2-dcr). On any glmm disagreement, glmm is
# presumed wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN OVERWRITES EVERY REFERENCE IT OWNS. Use VALIDATION_ONLY for anything
# short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_cr2.R
#
# Per entry: the data CSV (tools/prep/gen_cr2_data.py) gives X = [1, x1, x2] and,
# per cluster g, Phi_k = diag(lam) + U U' with U the u columns. beta is the GLS
# estimate under Phi; the CR2 SE and Pustejovsky-Tipton df of each coefficient come
# from tools/cr2/efficiency_lib.R's cr2() (`oracle = "cr2"`, random intercept) or
# tools/cr2/f2_slope_cr2_mc.R's cr2_phi() (`oracle = "cr2_phi"`), both with the
# §3.2 cut. The JSON carries coefficients, beta, se and df.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
source(file.path(suite_dir, "tools", "cr2", "efficiency_lib.R"))
# cr2_phi() alone: the rest of f2_slope_cr2_mc.R needs scripts outside the repo.
local({
  ex <- parse(file.path(suite_dir, "tools", "cr2", "f2_slope_cr2_mc.R"), encoding = "UTF-8")
  defines <- vapply(ex, function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
                      identical(e[[2]], as.name("cr2_phi")), NA)
  stopifnot(sum(defines) == 1L)
  eval(ex[[which(defines)]], envir = globalenv())
})

GENERATOR <- "goldens_cr2.R"

# The working model of `data` and its GLS fit.
working_model <- function(data) {
  X <- cbind(1, data$x1, data$x2)
  idx <- split(seq_len(nrow(data)), data$g)
  U <- as.matrix(data[grep("^u[0-9]+$", names(data))])
  Phi <- lapply(idx, function(i) diag(data$lam[i], length(i)) + tcrossprod(U[i, , drop = FALSE]))
  Wg <- lapply(Phi, solve)
  H <- Reduce(`+`, Map(function(i, w) crossprod(X[i, , drop = FALSE], w %*% X[i, , drop = FALSE]), idx, Wg))
  g <- Reduce(`+`, Map(function(i, w) crossprod(X[i, , drop = FALSE], w %*% data$y[i]), idx, Wg))
  list(X = X, idx = idx, U = U, Phi = Phi, beta = drop(solve(H, g)))
}

cr2_reference <- function(entry) {
  data <- read_entry_data(entry)
  wm <- working_model(data)
  one <- function(j) switch(entry$oracle,
    cr2 = {
      stopifnot(ncol(wm$U) == 1L, all(wm$U == wm$U[1]))
      cr2(wm$X, data$y, wm$beta, wm$U[1]^2, data$lam, wm$idx, j)
    },
    cr2_phi = cr2_phi(wm$X, data$y, wm$beta, wm$Phi, wm$idx, j),
    stop("unknown oracle ", entry$oracle))
  res <- lapply(seq_len(ncol(wm$X)), one)
  ref <- list(provenance = provenance(GENERATOR), oracle = entry$oracle,
              coefficients = c("(Intercept)", "x1", "x2"), beta = wm$beta,
              se = vapply(res, `[[`, 0, "se"), df = vapply(res, `[[`, 0, "df"))
  check_finite(ref, entry$name)
}

for (entry in Filter(function(e) identical(e$generator, GENERATOR), select_entries("cr2"))) {
  write_reference(cr2_reference(entry),
                  file.path(suite_dir, "goldens", "inference", "cr2", paste0(entry$name, ".json")))
  cat(sprintf("wrote %s\n", entry$name))
}
