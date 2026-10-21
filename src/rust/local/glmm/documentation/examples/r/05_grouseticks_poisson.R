# Recipe 5 -- Poisson GLMM (grouseticks).
#
# Tick counts per chick, with a per-brood random intercept: chicks in the
# same brood share unmeasured causes of infestation (nest site, parent
# condition) that a fixed effect alone cannot capture.
#
# Data: lme4's own `grouseticks` -- library(lme4); data(grouseticks).
#
# No manifest rung behind this exact formula: rung 6 (grouseticks) fits the
# centered cHEIGHT against all three crossed grouping factors (BROOD, INDEX,
# LOCATION) together, a different model from this recipe's single
# (1 | BROOD) on raw HEIGHT -- dropping two grouping factors and swapping the
# height variable changes what each variance component absorbs, so the
# golden numbers for rung 6 are not a valid comparison target for this fit.
# This recipe's output is a run, not an oracle-pinned result.

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(fastglmm)) # after lme4 (both export fixef/VarCorr/isSingular/ranef; the later library() call wins)

data(grouseticks)

fit <- fastglmm(TICKS ~ YEAR + HEIGHT + (1 | BROOD), grouseticks,
                 family = poisson())

cat("converged:", fit$converged, " singular:", fit$singular, "\n")
summary(fit)

vc <- VarCorr(fit)
cat("\nBROOD stddev:", attr(vc$BROOD, "stddev")[1], "\n")
cat("loglik:", fit$loglik, "\n")
cat("\n(no manifest rung matches this formula -- a run, not an oracle-pinned result)\n")
