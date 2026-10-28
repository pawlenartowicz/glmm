#!/usr/bin/env Rscript
# Effect-size references -> validation/goldens/inference/effectsize/<name>.json,
# one per `kind = "effectsize"` entry of the manifest's `inference_references`
# block.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's R²,
# ICC and partial eta² are validated against. On any glmm disagreement, glmm is
# presumed wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN REFITS AND OVERWRITES EVERY EFFECTSIZE REFERENCE. Use
# VALIDATION_ONLY for anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_effectsize.R
#
# Each entry has one member, fitted with fit_member. What the JSON holds follows
# the member's `fun`:
#   - lmer and glmer: `r2` (`marginal`, `conditional`) from
#     performance::r2_nakagawa(approximation = "lognormal") and `icc`
#     (`adjusted`, `unadjusted`) from performance::icc;
#   - lmer, lm and glm: `eta2`, one row per fixed term of the Type III table in
#     the table's order, with `term`, the `F`, `df1` and `df2` handed to
#     effectsize::F_to_eta2, and its `eta2_partial`. On lmer the table is
#     lmerTest's anova(type = 3) with Satterthwaite df. On lm it is
#     car::Anova(type = 3) and on glm car::Anova(type = 3, test.statistic =
#     "Wald"), both on the member refitted with contr.sum on the entry's
#     `factors` (fit_member_sum), since car's Type III depends on the coding;
#     the glm row's F is the Wald chi-square over its df, and df2 is
#     df.residual() of the fit. car's `(Intercept)` row is left out, as the
#     lmerTest table has none.
# Any R warning stops the run, so no value R flagged is frozen.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
EXTRA_PKGS <- assert_extra_versions(c("performance", "effectsize", "insight"))

GENERATOR <- "goldens_effectsize.R"
out_dir <- file.path(suite_dir, "goldens", "inference", "effectsize")

# The Type III F table of fit `m` (member `mem`) as a data frame of `term`, `F`,
# `df1`, `df2`, or NULL for a fit with no F table (glmer).
f_table <- function(entry, mem, m, data) {
  switch(mem$fun,
    "lme4::lmer" = {
      a <- anova(lmerTest::as_lmerModLmerTest(m), type = 3)
      data.frame(term = rownames(a), F = a[["F value"]], df1 = a[["NumDF"]],
                 df2 = a[["DenDF"]])
    },
    "stats::lm" = {
      a <- car::Anova(fit_member_sum(mem, data, unlist(entry$factors)), type = 3)
      a <- a[!rownames(a) %in% c("(Intercept)", "Residuals"), ]
      data.frame(term = rownames(a), F = a[["F value"]], df1 = a[["Df"]],
                 df2 = df.residual(m))
    },
    "stats::glm" = {
      a <- car::Anova(fit_member_sum(mem, data, unlist(entry$factors)), type = 3,
                      test.statistic = "Wald")
      a <- a[rownames(a) != "(Intercept)", ]
      data.frame(term = rownames(a), F = a[["Chisq"]] / a[["Df"]], df1 = a[["Df"]],
                 df2 = df.residual(m))
    },
    "lme4::glmer" = NULL,
    stop(sprintf("%s: unsupported fun %s", entry$name, mem$fun), call. = FALSE))
}

effectsize_reference <- function(entry) {
  if (length(entry$members) != 1L) {
    stop(sprintf("%s: an effectsize entry has one member", entry$name), call. = FALSE)
  }
  data <- read_entry_data(entry)
  mem <- entry$members[[1]]
  m <- fit_member(mem, data)
  out <- list()
  if (methods::is(m, "merMod")) {
    r2 <- performance::r2_nakagawa(m, approximation = "lognormal")
    icc <- performance::icc(m)
    out$r2 <- list(marginal = unname(r2$R2_marginal), conditional = unname(r2$R2_conditional))
    out$icc <- list(adjusted = icc$ICC_adjusted, unadjusted = icc$ICC_unadjusted)
  }
  tab <- f_table(entry, mem, m, data)
  if (!is.null(tab)) {
    es <- effectsize::F_to_eta2(tab$F, tab$df1, tab$df2)
    out$eta2 <- lapply(seq_len(nrow(tab)), function(i) {
      list(term = tab$term[i], F = tab$F[i], df1 = tab$df1[i], df2 = tab$df2[i],
           eta2_partial = es$Eta2_partial[i])
    })
  }
  if (length(unlist(out)) == 0L) stop(sprintf("%s: no value", entry$name), call. = FALSE)
  check_finite(out, entry$name)
}

for (entry in select_entries("effectsize")) {
  cat(sprintf("%s ...\n", entry$name))
  obj <- withCallingHandlers(
    c(list(provenance = provenance(GENERATOR, extra = EXTRA_PKGS)),
      effectsize_reference(entry)),
    warning = function(w) {
      stop(sprintf("%s: %s", entry$name, conditionMessage(w)), call. = FALSE)
    })
  write_reference(obj, file.path(out_dir, paste0(entry$name, ".json")))
}
