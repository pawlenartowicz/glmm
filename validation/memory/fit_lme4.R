#!/usr/bin/env Rscript
# lme4 oracle runner for the large synthetic memory-measurement models
# (models.json). NOT a validation rung -- see models.json's header; this is
# not compared to anything, only measured for peak RSS. lme4/MixedModels do
# not change between memory legs, so they are collected once into
# results/memory/oracles.tsv and reused in every leg comparison rather than
# refit per leg.
#
#   Rscript fit_lme4.R <csv> <formula> <factors_csv> [family]
#
# formula: the same R-style string as models.json's "formula" field (lme4's
# glmer()/lmer()/glm()/lm() all accept it verbatim -- no cbind/probit/dot-
# sanitizing quirks arise on these rows, unlike the curated manifest rungs'
# lme4.R). factors_csv: comma-separated grouping column names, coerced to
# factor() (empty for the two no-RE rows below).
#
# family (optional, default "binomial"): the original 4 oracle rows (1/4/6/9)
# were all binomial-with-RE, so every existing 3-arg call (memory.sh's
# run_oracles) keeps working unchanged. The oracle backfill added three more
# row shapes this script must also route correctly, picked by whether the
# formula has a "|" RE term AND family:
#   RE + binomial   (rows 2/3/5/7/8/10)  -> glmer(family = binomial())
#   RE + gaussian    (row 11)             -> lmer() (identity link, no family)
#   no RE + binomial (row 12)             -> glm(family = binomial())
#   no RE + gaussian (row 13)             -> lm()
suppressMessages(library(lme4))

args <- commandArgs(TRUE)
csv <- args[1]
formula_str <- args[2]
factors <- strsplit(args[3], ",", fixed = TRUE)[[1]]
family_str <- if (length(args) >= 4) args[4] else "binomial"

df <- read.csv(csv, stringsAsFactors = FALSE)
for (f in factors) df[[f]] <- factor(df[[f]])

has_re <- grepl("|", formula_str, fixed = TRUE)
gaussian <- family_str == "gaussian"
m <- if (has_re && !gaussian) {
  glmer(as.formula(formula_str), data = df, family = binomial())
} else if (has_re && gaussian) {
  lmer(as.formula(formula_str), data = df)
} else if (!has_re && !gaussian) {
  glm(as.formula(formula_str), data = df, family = binomial())
} else {
  lm(as.formula(formula_str), data = df)
}
cat(sprintf("n=%d deviance=%.6f\n", nrow(df), deviance(m)))
