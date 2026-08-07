#!/usr/bin/env Rscript
# R-port (fastglmm) runner for the large synthetic memory-measurement models
# (models.json). NOT a validation rung -- see models.json's header. Fits an
# arbitrary generated CSV + formula through the installed fastglmm package
# (the same extendr wrapper engines/glmm_r.R drives for the curated manifest rungs),
# so memory.sh can wrap this in `/usr/bin/time -f '%M'` one process per
# (engine, model).
#
#   Rscript fit_r.R <csv> <formula> <family> <link> <factors_csv> <nagq>
#
# factors_csv: comma-separated grouping column names, coerced to factor()
# (lexicographic level order, matching read.csv's stringsAsFactors=FALSE
# default plus an explicit factor() call). Prints one status line -- stdout is
# not compared to anything.

suppressMessages(library(fastglmm))

args <- commandArgs(TRUE)
csv <- args[1]
formula_str <- args[2]
family_str <- args[3]
link <- args[4]
factors <- strsplit(args[5], ",", fixed = TRUE)[[1]]
nagq <- as.integer(args[6])

df <- read.csv(csv, stringsAsFactors = FALSE)
for (f in factors) df[[f]] <- factor(df[[f]])

fam <- if (family_str == "gaussian") {
  gaussian()
} else if (nzchar(link)) {
  get(family_str)(link = link)
} else {
  get(family_str)()
}

m <- fastglmm(as.formula(formula_str), data = df, family = fam, nAGQ = nagq)
cat(sprintf("n=%d deviance=%.6f\n", nrow(df), m$deviance))
