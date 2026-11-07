#!/usr/bin/env Rscript
# Generator for glmm_hessian_vcov.json -- the n=96 / 12-cluster `y ~ x1 + (1|grp)`
# glmer fixture behind `joint_hessian_cov_matches_glmer_use_hessian_true` (and the
# pipeline test that cites its band). Rebuilds the data block (x / y / cluster_ids)
# from its seed, stops unless it equals the committed one (the data are the
# fixture's identity and never change), refits, and rewrites the derived fields:
# theta, beta, vcov_hessian, vcov_rx.
#
# The data recipe is set.seed(2137), 12 clusters x 8 rows,
# eta = -0.4 + 0.5 * x1 + u (u normal with variance 0.3, one per cluster), y ~ Bernoulli(plogis(eta)).
# x1 is stored as jsonlite writes it at digits = 12, so it is rounded the same way
# here before the comparison and the fit.
#
# tolPwrss = 1e-13, matching validation/tools/goldens_agq.R (change together): at glmer's
# default 1e-7 the ldL2 term uses working weights one PIRLS iteration behind the
# mode, putting a ~1% spurious theta/theta-beta curvature into
# vcov(use.hessian=TRUE) -- tolPwrss=1e-13 avoids that distortion.
# vcov_rx (use.hessian=FALSE) never reads that curvature and only moves at the
# theta-hat level (~1e-7).
#
# Run from the crate root:  Rscript tests/fixtures/gen_glmm_hessian_vcov.R

suppressMessages({ library(lme4); library(jsonlite) })

path <- file.path("tests", "fixtures", "glmm_hessian_vcov.json")
fx <- fromJSON(path, simplifyVector = TRUE, simplifyMatrix = TRUE)

set.seed(2137)
n_grp <- 12; per <- 8; n <- n_grp * per
grp <- rep(seq_len(n_grp), each = per)
x1  <- rnorm(n)
eta <- -0.4 + 0.5 * x1 + rnorm(n_grp, sd = sqrt(0.3))[grp]
y   <- rbinom(n, 1, plogis(eta))
x1  <- fromJSON(toJSON(x1, digits = 12))
stopifnot(fx$n == n, identical(fx$x[, 1], rep(1, n)), identical(fx$x[, 2], x1),
          identical(as.numeric(fx$y), as.numeric(y)),
          identical(as.numeric(fx$cluster_ids), as.numeric(grp - 1L)))

df <- data.frame(y = y, x1 = x1, grp = factor(grp))

m <- glmer(y ~ x1 + (1 | grp), data = df, family = binomial(), nAGQ = 1,
           control = glmerControl(tolPwrss = 1e-13))

fx$theta <- as.numeric(getME(m, "theta"))
fx$beta <- as.numeric(fixef(m))
fx$vcov_hessian <- unname(as.matrix(vcov(m, use.hessian = TRUE)))
fx$vcov_rx <- unname(suppressWarnings(as.matrix(vcov(m, use.hessian = FALSE))))

# digits = NA: jsonlite's maximum, 15 significant digits (not a full 17-digit
# round trip), so each value carries up to 5e-15 relative rounding.
write(toJSON(fx, auto_unbox = TRUE, pretty = TRUE, digits = NA), path)
cat(sprintf("rewrote %s: theta %.10f, beta %s\n", path, fx$theta,
            paste(sprintf("%.10f", fx$beta), collapse = " ")))
