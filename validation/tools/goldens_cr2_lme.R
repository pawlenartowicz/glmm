#!/usr/bin/env Rscript
# clubSandwich CR2 on nlme fits -> validation/goldens/inference/cr2/<name>.json, one per
# `inference_references` entry with `generator = "goldens_cr2_lme.R"`.
#
# THE ORACLE IS SACRED. These JSONs are the independent check of the crate's robust
# fits (gate G5 of spec 2-dcr). On any glmm disagreement, glmm is presumed wrong:
# never relax a tolerance or edit a reference.
#
# A BARE RUN OVERWRITES EVERY REFERENCE IT OWNS. Use VALIDATION_ONLY for anything
# short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_cr2_lme.R
#
# Per entry: a REML fit, nlme::lme when r_formula has a random term (random intercept
# only) and nlme::gls when it has none, with Var(e_i) = sigma^2 w_i^(-delta) for the
# prior weight w = the entry's `weights`. `weight_power = "fixed"` fits delta = 1
# (varFixed on 1 / w); "estimated" fits varPower(form = ~ w), whose power t gives
# delta = -2 t, the value a glmm fit takes as `weight_power = Fixed(delta)` (the
# shared delta of G5). Then clubSandwich's coef_test(vcov = "CR2", test =
# "Satterthwaite"), clustered by the group, one cluster per row on a gls fit. The
# JSON carries coefficients, beta, se, df, delta, sigma2 and, on lme, tau2. Before
# writing, the run checks SE and df against tools/cr2/efficiency_lib.R's cr2() at the
# fitted working model and stops on a relative gap above 1e-8.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
extra <- assert_extra_versions(c("nlme", "clubSandwich"))
source(file.path(suite_dir, "tools", "cr2", "efficiency_lib.R"))

GENERATOR <- "goldens_cr2_lme.R"

lme_reference <- function(entry) {
  data <- read_entry_data(entry)
  data$.w <- data[[entry$weights]]
  data$.iw <- 1 / data$.w  # a plain column: clubSandwich fails on varFixed(~ I(1 / .w)) in gls
  f <- as.formula(entry$r_formula)
  fixed <- reformulas::nobars(f)
  bars <- reformulas::findbars(f)
  vf <- switch(entry$weight_power,
               fixed = nlme::varFixed(~ .iw),
               estimated = nlme::varPower(form = ~ .w),
               stop("unknown weight_power ", entry$weight_power))
  n <- nrow(data)
  if (length(bars) == 0L) {
    # data in the call itself: clubSandwich re-reads it from there
    m <- eval(bquote(nlme::gls(.(fixed), data = .(data), weights = .(vf), method = "REML",
                               control = nlme::glsControl(tolerance = 1e-12, msTol = 1e-12,
                                                          maxIter = 500, msMaxIter = 500))))
    cluster <- factor(seq_len(n))
    idx <- as.list(seq_len(n))
    tau2 <- 0
  } else {
    stopifnot(length(bars) == 1L, identical(deparse(bars[[1]][[2]]), "1"))
    g <- deparse(bars[[1]][[3]])
    m <- nlme::lme(fixed, random = as.formula(paste("~ 1 |", g)), data = data, weights = vf,
                   method = "REML",
                   control = nlme::lmeControl(tolerance = 1e-12, msTol = 1e-12, maxIter = 500,
                                              msMaxIter = 500, niterEM = 100))
    cluster <- data[[g]]
    idx <- split(seq_len(n), data[[g]])
    tau2 <- as.numeric(nlme::getVarCov(m)[1, 1])
  }
  delta <- if (entry$weight_power == "fixed") 1 else
    -2 * unname(coef(m$modelStruct$varStruct, unconstrained = FALSE)[["power"]])
  sigma2 <- m$sigma^2
  ct <- clubSandwich::coef_test(m, vcov = "CR2", cluster = cluster, test = "Satterthwaite")
  X <- stats::model.matrix(fixed, data)
  beta <- unname(if (inherits(m, "gls")) stats::coef(m) else nlme::fixef(m))
  ours <- lapply(seq_len(ncol(X)), function(j)
    cr2(X, data$y, beta, tau2, sigma2 * data$.w^(-delta), idx, j))
  gap <- max(abs(vapply(ours, `[[`, 0, "se") / ct$SE - 1), abs(vapply(ours, `[[`, 0, "df") / ct$df_Satt - 1))
  cat(sprintf("%s: cr2() against clubSandwich, largest relative gap %.2e\n", entry$name, gap))
  if (gap > 1e-8) stop(entry$name, ": cr2() and clubSandwich disagree", call. = FALSE)
  ref <- list(provenance = provenance(GENERATOR, extra = extra), coefficients = colnames(X),
              beta = beta, se = ct$SE, df = ct$df_Satt, delta = delta, sigma2 = sigma2)
  if (length(bars) > 0L) ref$tau2 <- tau2
  check_finite(ref, entry$name)
}

for (entry in Filter(function(e) identical(e$generator, GENERATOR), select_entries("cr2"))) {
  write_reference(lme_reference(entry),
                  file.path(suite_dir, "goldens", "inference", "cr2", paste0(entry$name, ".json")))
  cat(sprintf("wrote %s\n", entry$name))
}
