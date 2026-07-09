#!/usr/bin/env Rscript
# Generator for glmm_hessian_vcov.json -- the n=96 / 12-cluster `y ~ x1 + (1|grp)`
# glmer fixture behind `fd_hessian_cov_matches_glmer_use_hessian_true` (and the
# pipeline test that cites its band). Reads the committed JSON's data block
# (x / y / cluster_ids are the fixture's identity and never change), refits, and
# rewrites the derived fields: theta, beta, vcov_hessian, vcov_rx.
#
# tolPwrss = 1e-13, matching parity/oracle/fit.R (change together): at glmer's
# default 1e-7 the ldL2 term uses working weights one PIRLS iteration behind the
# mode, putting a ~1% spurious theta/theta-beta curvature into
# vcov(use.hessian=TRUE) -- the artifact the fixture carried until 2026-07-04
# (docs/GLMM/2026-07-04-glmm-hessian-curvature-diagnosis.md, Resolution).
# vcov_rx (use.hessian=FALSE) never reads that curvature and only moves at the
# theta-hat level (~1e-7).
#
# Run from the crate root:  Rscript tests/fixtures/gen_glmm_hessian_vcov.R

suppressMessages({ library(lme4); library(jsonlite) })

path <- file.path("tests", "fixtures", "glmm_hessian_vcov.json")
fx <- fromJSON(path, simplifyVector = TRUE, simplifyMatrix = TRUE)

df <- data.frame(
  y = as.numeric(fx$y),
  x1 = fx$x[, 2],                      # x column 1 is the intercept
  grp = factor(fx$cluster_ids)
)

m <- glmer(y ~ x1 + (1 | grp), data = df, family = binomial(), nAGQ = 1,
           control = glmerControl(tolPwrss = 1e-13))

fx$theta <- as.numeric(getME(m, "theta"))
fx$beta <- as.numeric(fixef(m))
fx$vcov_hessian <- unname(as.matrix(vcov(m, use.hessian = TRUE)))
fx$vcov_rx <- unname(suppressWarnings(as.matrix(vcov(m, use.hessian = FALSE))))

# digits = NA: full double precision -- an oracle fixture, not a display.
write(toJSON(fx, auto_unbox = TRUE, pretty = TRUE, digits = NA), path)
cat(sprintf("rewrote %s: theta %.10f, beta %s\n", path, fx$theta,
            paste(sprintf("%.10f", fx$beta), collapse = " ")))
