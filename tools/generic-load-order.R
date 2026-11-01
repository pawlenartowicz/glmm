# Cross-package generic dispatch, checked against a real lme4 fit.
#
# RUN BY HAND, LOCALLY. This is not part of R CMD check and not part of CI:
# it needs lme4, which is not a dependency of this package (not even a
# Suggests - the only live use of lme4 is here and in acceptance-vs-lme4.R,
# both hand-run) and is deliberately not installed on the CI runner.
# .Rbuildignore keeps this file out of the built tarball.
#
# From the r/ directory:
#
#   Rscript -e 'library(fastglmm); library(testthat); \
#               testthat::test_file("tools/generic-load-order.R")'
#
# fixef/ranef/VarCorr register methods on nlme's generics, which lme4
# re-exports unchanged, so there is only ever one such generic in scope and
# no load order to test for them. isSingular is lme4's own generic, and
# fastglmm declares one of its own too (fastglmm-methods.R), so both load
# orders need checking on both a fastglmm fit and a merMod fit - the two
# directions this script exists for. isSingular working with fastglmm
# loaded and no other package attached at all (no lme4 installed) is
# covered in tests/testthat/test-methods.R, which does not need this file.
#
# Each order runs in its own Rscript subprocess, not by detaching and
# reattaching packages within this session: fastglmm carries a compiled
# DLL, and unloading/reloading one mid-session is not guaranteed to leave
# the process in the same state as never having loaded it.

run_order <- function(first, second) {
  script <- sprintf(paste(
    "library(%s); library(%s);",
    "d <- data.frame(y = rnorm(60), x = rnorm(60), g = factor(rep(1:10, 6)));",
    "fit <- suppressWarnings(fastglmm(y ~ x + (1 | g), d));",
    "mfit <- lmer(y ~ x + (1 | g), data = d);",
    "cat(is.numeric(fixef(fit)), is.numeric(fixef(mfit)),",
    "inherits(VarCorr(fit), 'VarCorr.fastglmm'), !is.null(VarCorr(mfit)),",
    "identical(names(ranef(fit)), 'g'), is.logical(isSingular(fit)),",
    "is.logical(isSingular(mfit)), sep = ' ')"
  ), first, second)
  out <- system2("Rscript", c("-e", shQuote(script)), stdout = TRUE, stderr = TRUE)
  status <- attr(out, "status")
  list(status = if (is.null(status)) 0L else status, out = out)
}

test_that("fixef/ranef/VarCorr/isSingular dispatch correctly in both load orders", {
  # Order A: fastglmm attached first, lme4 second - lme4's generic is the
  # one left in scope, so a fastglmm fit's isSingular() needs the delayed
  # S3 registration; a merMod fit dispatches through lme4's own S4 method
  # as always.
  a <- run_order("fastglmm", "lme4")
  expect_equal(a$status, 0L, info = paste(a$out, collapse = "\n"))
  expect_equal(a$out[length(a$out)], "TRUE TRUE TRUE TRUE TRUE TRUE TRUE")

  # Order B: lme4 attached first, fastglmm second - fastglmm's generic is
  # the one left in scope, so a merMod fit's isSingular() needs
  # isSingular.default to forward to lme4's generic (already loaded, since
  # a merMod object cannot exist otherwise); a fastglmm fit dispatches
  # through fastglmm's own generic as always.
  b <- run_order("lme4", "fastglmm")
  expect_equal(b$status, 0L, info = paste(b$out, collapse = "\n"))
  expect_equal(b$out[length(b$out)], "TRUE TRUE TRUE TRUE TRUE TRUE TRUE")
})
