# Recipe 7 -- factors and interactions (cake), and changing the base level.
#
# recipe*temp desugars to recipe + temp + recipe:temp: a main effect per
# recipe, a slope on temp, and a per-recipe deviation from that slope.
# Treatment contrasts code `recipe` against its first level ("A", cake's
# factor level order as shipped). relevel() changes which level that is,
# without touching the data -- there is deliberately no contrasts= argument.
#
# Data: lme4's own `cake` -- library(lme4); data(cake).
#
# No manifest rung behind this formula -- `cake` does not appear in
# validation/goldens/ at all (manifest rung 13 names it, but no lme4
# reference JSON was frozen there). This recipe's output is a run, not an
# oracle-pinned result.

suppressPackageStartupMessages(library(lme4))
suppressPackageStartupMessages(library(fastglmm)) # after lme4 (both export fixef/VarCorr/isSingular/ranef; the later library() call wins)

data(cake)

fit_a <- fastglmm(angle ~ recipe * temp + (1 | recipe:replicate), cake)

cat("=== base = A (cake's factor level order as shipped) ===\n")
summary(fit_a)

cake_b <- cake
cake_b$recipe <- relevel(cake_b$recipe, ref = "B")
fit_b <- fastglmm(angle ~ recipe * temp + (1 | recipe:replicate), cake_b)

cat("\n=== base = B (relevel(cake$recipe, ref = \"B\")) ===\n")
summary(fit_b)

cat(sprintf(
  "\nSame fit, different parameterization: fitted values and loglik agree (loglik A=%.10g, loglik B=%.10g, delta=%.3g); only which contrasts are directly readable off fixef() changes.\n",
  fit_a$loglik, fit_b$loglik, fit_b$loglik - fit_a$loglik
))
cat("\n(cake carries no goldens/ entry -- a run, not an oracle-pinned result)\n")
