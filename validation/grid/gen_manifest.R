#!/usr/bin/env Rscript
# Generates grid/manifest.json -- THE reproducibility artifact of the grid. Data
# is derived (gen.R), the manifest is the record. Deliberately non-full crossing:
#
#   CORE     every family/link/form x every structure it can identify, at ONE
#            reference point (n_obs 3000, per_group 20, bal, base, Laplace).
#   VARIANTS one axis at a time away from the core -- size, balance, regime,
#            nagq -- on a fixed set of representative structures per FAMILY
#            (five families, not per link), named in REPRESENTATIVE.
#   EXTRAS   GLM (no random effect) cells per family, offset cells (poisson,
#            negative binomial), prior-weight cells outside the binomial-trials
#            case (gaussian, poisson, gamma).
#   READ-IN  every empirical dataset and every committed simulated fixture under
#            the crate's data/ tree, read in place, never regenerated.
#
# Uncorrelated slopes have no structure code here.
#
# THE ENUMERATION ORDER IS COSMETIC: each cell's seed is seed_of(cell$cell), a
# hash of its own id, so inserting or removing a cell anywhere leaves every
# other cell's data byte-identical. Trimming REPRESENTATIVE after the pilot has
# already measured is what makes that property required rather than nice.
#
# No cell carries an evaluation cap. A truncated oracle fit is not an oracle;
# every engine runs with its own default optimizer settings and the only cap
# anywhere is run.sh's per-cell watchdog, which records a timeout rather than a
# converged-looking truncated fit.
suppressMessages({ library(jsonlite); library(MASS) })
here <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "gen_common.R"))

GLM_N_X <- 3L            # numeric predictors on a no-random-effect cell
AGQ_K   <- 7L            # quadrature points on the AGQ variant cells
VALIDATION <- normalizePath(file.path(here, ".."))

# ---- truth for the committed fixtures ----------------------------------------
# Transcribed from the prep scripts that wrote the CSVs. Only fixtures whose
# generator draws from an EXPLICIT parameter vector get an entry; everything
# else goes in FIXTURE_NO_TRUTH with a reason, because a truth reverse-
# engineered from the data is not truth.
#
# KEYED ON THE CELL ID, which for a datasets rung is its `name` and for a
# goldens-derived cell is the GOLDEN's name -- `sim_nb_glmm`, not `sim_nb`
# (`sim_nb` is a golden DATA STEM, there is no datasets rung by that name).
# Getting that wrong is silent: the entry simply never matches.
#
# THE TABLE IS COMPLETE, NOT ILLUSTRATIVE. Every read-in simulated cell appears
# in exactly one of the two lists below; the self-checks at the end assert it,
# and the generator stops rather than emitting a fixture cell whose truth nobody
# decided.
#
# Nested groupings are named parent:child (`g1:g2`), the crate's own convention.
# lme4 displays the same grouping child:parent; that is a display difference,
# not a different grouping.

# Varcomp block: `corr` is either one off-diagonal value (compound symmetry) or
# a full correlation matrix.
vcb <- function(group, terms, stddev, corr = 0) {
  q <- length(stddev)
  m <- if (is.matrix(corr)) corr else { mm <- matrix(corr, q, q); diag(mm) <- 1; mm }
  list(group = group, terms = I(terms), stddev = I(stddev), corr = m)
}
tr <- function(coef_names, beta, varcomp = list(), sigma = NULL,
               dispersion = NULL, nb_theta = NULL)
  list(coef_names = I(coef_names), beta = I(beta), varcomp = varcomp,
       sigma = sigma, dispersion = dispersion, nb_theta = nb_theta)

IC <- "(Intercept)"

FIXTURE_TRUTH <- list(
  # prep/export_data.R, make_clustered() + the Gamma block: eta = 0.3 + 0.6*x +
  # 0.4*(grp=="b") + u, u ~ N(0, 0.6^2) over 24 clusters; y ~ rgamma(shape = 2,
  # scale = mu/2), so phi = 1/shape = 0.5.
  sim_gamma = tr(c(IC, "x", "grpb"), c(0.3, 0.6, 0.4),
                 list(vcb("cluster", IC, 0.6)), dispersion = 0.5),
  # prep/export_data.R, the SAME make_clustered() draw; y ~ MASS::rnegbin(theta =
  # 1.5). The cell is the golden `sim_nb_glmm` (data stem sim_nb).
  sim_nb_glmm = tr(c(IC, "x", "grpb"), c(0.3, 0.6, 0.4),
                   list(vcb("cluster", IC, 0.6)), nb_theta = 1.5),
  # Same data, fitted fixed-only: the RE variance is part of the residual, so the
  # GLM cell's truth carries the betas and theta but no varcomp.
  sim_nb_glm = tr(c(IC, "x", "grpb"), c(0.3, 0.6, 0.4), nb_theta = 1.5),
  # prep/export_data.R, make_slope(): y = 1.0 + 0.5*x + u0 + u1*x + v0 + e,
  # sd(u0) = 1.2, sd(u1) = 0.7 drawn INDEPENDENTLY (so the true correlation is
  # exactly 0, not the 0.2 the generated cells use), sd(v0) = 0.9, residual 0.8.
  # Cell = the golden `sim_slope_lmm` (data stem sim_slope).
  sim_slope_lmm = tr(c(IC, "x"), c(1.0, 0.5),
                     list(vcb("g1", c(IC, "x"), c(1.2, 0.7), 0),
                          vcb("g2", IC, 0.9)),
                     sigma = 0.8),
  # prep/export_data.R, make_slope_extra(): y = 1.0 + 0.5*x + b1 + b2, both
  # groupings intercept+slope, S1 = sds (1.1, 0.6) corr 0.3, S2 = sds (0.9, 0.5)
  # corr 0.2, residual 0.7.
  sim_slope_extra = tr(c(IC, "x"), c(1.0, 0.5),
                       list(vcb("g1", c(IC, "x"), c(1.1, 0.6), 0.3),
                            vcb("g2", c(IC, "x"), c(0.9, 0.5), 0.2)),
                       sigma = 0.7),
  # prep/export_data.R, make_sparse_binomial(): eta = 0.2 + 0.5*x + u_g1 +
  # sum_k u_ck, sd(g1) = 0.8, sd(c1..c7) = (0.50, 0.45, 0.40, 0.50, 0.40, 0.45,
  # 0.35); incidence ~ rbinom(size, plogis(eta)).
  sim_sparse_binomial = tr(c(IC, "x"), c(0.2, 0.5),
    c(list(vcb("g1", IC, 0.8)),
      mapply(function(nm, s) vcb(nm, IC, s), paste0("c", 1:7),
             c(0.50, 0.45, 0.40, 0.50, 0.40, 0.45, 0.35), SIMPLIFY = FALSE,
             USE.NAMES = FALSE))),
  # prep/export_data.R, make_sparse_poisson(): eta = 0.3 + 0.5*x + u_g1 +
  # sum_k u_ck, sd(g1) = 0.7, sd(c1..c7) = (0.45, 0.40, 0.35, 0.45, 0.35, 0.40,
  # 0.30); y ~ rpois(exp(eta)).
  sim_sparse_poisson = tr(c(IC, "x"), c(0.3, 0.5),
    c(list(vcb("g1", IC, 0.7)),
      mapply(function(nm, s) vcb(nm, IC, s), paste0("c", 1:7),
             c(0.45, 0.40, 0.35, 0.45, 0.35, 0.40, 0.30), SIMPLIFY = FALSE,
             USE.NAMES = FALSE))),
  # prep/export_data.R, make_sparse_gamma(): eta = 0.5 + 0.6*x1 - 0.4*x2 +
  # 0.3*x3 - 0.2*x4 + u_gp + b_ge, sd(gp) = 0.5, ge block sds
  # (0.70, 0.50, 0.40, 0.30, 0.20) with compound-symmetry corr 0.25;
  # y ~ rgamma(shape = 2, scale = mu/2), phi = 0.5.
  sim_sparse_gamma = tr(c(IC, "x1", "x2", "x3", "x4"),
                        c(0.5, 0.6, -0.4, 0.3, -0.2),
                        list(vcb("gp", IC, 0.5),
                             vcb("ge", c(IC, "x1", "x2", "x3", "x4"),
                                 c(0.70, 0.50, 0.40, 0.30, 0.20), 0.25)),
                        dispersion = 0.5),
  # prep/export_data.R, make_sparse_nb(): the make_sparse_poisson recipe with
  # y ~ MASS::rnegbin(theta = 1.5). Cell = the golden `sim_sparse_nb`.
  sim_sparse_nb = tr(c(IC, "x"), c(0.3, 0.5),
    c(list(vcb("g1", IC, 0.7)),
      mapply(function(nm, s) vcb(nm, IC, s), paste0("c", 1:7),
             c(0.45, 0.40, 0.35, 0.45, 0.35, 0.40, 0.30), SIMPLIFY = FALSE,
             USE.NAMES = FALSE)),
    nb_theta = 1.5),
  # prep/export_data.R, make_three_level(): y = 1.5 + 0.8*x + u1 + u2 + e,
  # sd(g1) = 1.0, sd(g1:g2) = 0.7, residual 0.6.
  sim_three_level = tr(c(IC, "x"), c(1.5, 0.8),
                       list(vcb("g1", IC, 1.0), vcb("g1:g2", IC, 0.7)),
                       sigma = 0.6),
  # prep/export_data.R, make_max_q_slope(): y = 1.5 + 0.8*x1 - 0.5*x2 + 0.3*x3 -
  # 0.2*x4 + 0.4*x5 - 0.3*x6 + 0.2*x7 + b[g1], an 8-wide block with sds
  # (1.0 .. 0.3) and compound-symmetry corr 0.2, residual 0.5.
  sim_max_q_slope = tr(c(IC, paste0("x", 1:7)),
                       c(1.5, 0.8, -0.5, 0.3, -0.2, 0.4, -0.3, 0.2),
                       list(vcb("g1", c(IC, paste0("x", 1:7)),
                                c(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3), 0.2)),
                       sigma = 0.5),
  # prep/export_data.R, make_crossed_at_cap(): eta = 0.3 + 0.5*x + u_g1 +
  # sum_k u_ck, sd(g1) = 0.6, sd(c1..c6) = (0.45, 0.40, 0.35, 0.45, 0.35, 0.40).
  sim_crossed_at_cap = tr(c(IC, "x"), c(0.3, 0.5),
    c(list(vcb("g1", IC, 0.6)),
      mapply(function(nm, s) vcb(nm, IC, s), paste0("c", 1:6),
             c(0.45, 0.40, 0.35, 0.45, 0.35, 0.40), SIMPLIFY = FALSE,
             USE.NAMES = FALSE))),
  # prep/export_data.R, make_binomial_slope_crossed(): eta = 0.2 + 0.5*x + b1 +
  # b2, S1 = sds (0.8, 0.5) corr 0.2, S2 = sds (0.6, 0.4) corr 0.15.
  sim_binomial_slope_crossed = tr(c(IC, "x"), c(0.2, 0.5),
    list(vcb("g1", c(IC, "x"), c(0.8, 0.5), 0.2),
         vcb("g2", c(IC, "x"), c(0.6, 0.4), 0.15))),
  # prep/export_data.R, make_poisson_nested(): eta = 0.5 + 0.3*x + u1 + u2,
  # sd(g1) = 0.5, sd(g1:g2) = 0.3.
  sim_poisson_nested = tr(c(IC, "x"), c(0.5, 0.3),
                          list(vcb("g1", IC, 0.5), vcb("g1:g2", IC, 0.3))),
  # prep/export_data.R, make_unbalanced_nested(): y = 1.0 + 0.6*x + u1 + u2 + e,
  # sd(g1) = 1.0, sd(g1:g2) = 0.6, residual 0.5.
  sim_unbalanced_nested = tr(c(IC, "x"), c(1.0, 0.6),
                             list(vcb("g1", IC, 1.0), vcb("g1:g2", IC, 0.6)),
                             sigma = 0.5),
  # prep/export_data.R, make_nested_crossed_mix(): y = 1.2 + 0.6*x + u1 + u2 +
  # uc + e, sd(g1) = 1.0, sd(g1:g2) = 0.6, sd(c1) = 0.8, residual 0.5.
  sim_nested_crossed_mix = tr(c(IC, "x"), c(1.2, 0.6),
                              list(vcb("g1", IC, 1.0), vcb("g1:g2", IC, 0.6),
                                   vcb("c1", IC, 0.8)),
                              sigma = 0.5),
  # prep/export_data.R, make_binomial_slope1(): eta = 0.3 + 0.6*x + b[g],
  # block sds (1.0, 0.6) corr 0.3; Bernoulli response.
  sim_binomial_slope1 = tr(c(IC, "x"), c(0.3, 0.6),
                           list(vcb("g", c(IC, "x"), c(1.0, 0.6), 0.3))),
  # prep/export_data.R, make_poisson_slope1(): eta = -1.2 + 0.4*x + b[g],
  # block sds (1.0, 0.6) corr -0.2.
  sim_poisson_slope1 = tr(c(IC, "x"), c(-1.2, 0.4),
                          list(vcb("g", c(IC, "x"), c(1.0, 0.6), -0.2))),
  # prep/export_data.R, make_binomial_slope2(): eta = 0.5 + 0.5*x1 - 0.4*x2 +
  # b[g], block sds (1.0, 0.7, 0.6) with correlations (0.3, 0.1, 0.2).
  sim_binomial_slope2 = tr(c(IC, "x1", "x2"), c(0.5, 0.5, -0.4),
    list(vcb("g", c(IC, "x1", "x2"), c(1.0, 0.7, 0.6),
             matrix(c(1, 0.3, 0.1, 0.3, 1, 0.2, 0.1, 0.2, 1), 3, 3)))),
  # prep/export_data.R, make_poisson_offset(): eta = 0.3 + 0.5*x + u_cluster +
  # log_exposure, sd(cluster) = 0.5. The offset enters with an implicit
  # coefficient of 1 and is not part of beta.
  sim_poisson_offset = tr(c(IC, "x"), c(0.3, 0.5),
                          list(vcb("cluster", IC, 0.5))),
  # prep/export_data.R, make_nb_edge(n = 4000, b0 = 2.0, theta = 800):
  # mu = exp(2.0 + 0.6*x + 0.4*(grp=="b")).
  sim_nb_hightheta_glm = tr(c(IC, "x", "grpb"), c(2.0, 0.6, 0.4), nb_theta = 800),
  # prep/export_data.R, make_nb_edge(n = 400, b0 = 1.0, theta = 0.005).
  sim_nb_lowtheta_glm = tr(c(IC, "x", "grpb"), c(1.0, 0.6, 0.4), nb_theta = 0.005),
  # prep/export_data.R, make_nb_nested(): eta = 0.8 + 0.5*x + u1 + u2,
  # sd(g1) = 0.6, sd(g1:g2) = 0.4, theta = 1.5.
  sim_nb_nested_glmm = tr(c(IC, "x"), c(0.8, 0.5),
                          list(vcb("g1", IC, 0.6), vcb("g1:g2", IC, 0.4)),
                          nb_theta = 1.5),
  # prep/export_data.R, make_poisson_highmean(): mu = exp(4.3 + 0.3*x +
  # 0.2*(grp=="b")).
  sim_poisson_highmean_glm = tr(c(IC, "x", "grpb"), c(4.3, 0.3, 0.2)),
  # prep/export_data.R, make_wide_crossed(): y = 1.5 + 0.8*x + u_g1 + sum_k u_ck
  # + e, sd(g1) = 1.0, sd(c1..c7) = (0.8, 0.7, 0.6, 0.7, 0.6, 0.8, 0.5),
  # residual 0.6.
  sim_wide_crossed_lmm = tr(c(IC, "x"), c(1.5, 0.8),
    c(list(vcb("g1", IC, 1.0)),
      mapply(function(nm, s) vcb(nm, IC, s), paste0("c", 1:7),
             c(0.8, 0.7, 0.6, 0.7, 0.6, 0.8, 0.5), SIMPLIFY = FALSE,
             USE.NAMES = FALSE)),
    sigma = 0.6),
  # prep/export_data.R, make_wide_slopes(): y = 1.5 + 0.8*x1 - 0.5*x2 + 0.3*x3 -
  # 0.2*x4 + u_gp + b_ge + e, sd(gp) = 0.8, ge block sds
  # (1.0, 0.9, 0.7, 0.5, 0.3) with compound-symmetry corr 0.25, residual 0.6.
  sim_wide_slopes_lmm = tr(c(IC, "x1", "x2", "x3", "x4"),
                           c(1.5, 0.8, -0.5, 0.3, -0.2),
                           list(vcb("gp", IC, 0.8),
                                vcb("ge", c(IC, "x1", "x2", "x3", "x4"),
                                    c(1.0, 0.9, 0.7, 0.5, 0.3), 0.25)),
                           sigma = 0.6),
  # prep/export_data.R, make_cloglog_nested_crossed(): eta = -0.8 + 0.5*x + u1 +
  # u2 + uc, sd(g1) = 0.5, sd(g1:g2) = 0.3, sd(c1) = 0.4; p = 1 - exp(-exp(eta)).
  sim_cloglog_nested_crossed = tr(c(IC, "x"), c(-0.8, 0.5),
                                  list(vcb("g1", IC, 0.5), vcb("g1:g2", IC, 0.3),
                                       vcb("c1", IC, 0.4))),

  # tools/prep/gen_weights_data.R, W1: y = 1.0 + 0.5*x1 - 0.3*x2 + e with
  # sd(e) = 0.8/sqrt(w), so 0.8 IS the weighted residual sd.
  wls_basic = tr(c(IC, "x1", "x2"), c(1.0, 0.5, -0.3), sigma = 0.8),
  # tools/prep/gen_weights_data.R, W2: incidence ~ rbinom(size, plogis(0.3 + 0.6*x)).
  glm_binomial_agg = tr(c(IC, "x"), c(0.3, 0.6)),
  # tools/prep/gen_weights_data.R, W3: y ~ rpois(exp(0.4 + 0.5*x + 0.3*(grp=="b"))).
  # The `w` column is a prior weight, not part of the data-generating model.
  glm_poisson = tr(c(IC, "x", "grpb"), c(0.4, 0.5, 0.3)),
  # tools/prep/gen_weights_data.R, W4: mu = exp(0.5 + 0.6*x + 0.4*(grp=="b")),
  # y ~ rgamma(shape = 2, scale = mu/2), phi = 0.5.
  glm_gamma = tr(c(IC, "x", "grpb"), c(0.5, 0.6, 0.4), dispersion = 0.5),
  # tools/prep/gen_weights_data.R, W5: y ~ rnegbin(exp(0.5 + 0.5*x + 0.3*(grp=="b")),
  # theta = 1.5).
  glm_nb = tr(c(IC, "x", "grpb"), c(0.5, 0.5, 0.3), nb_theta = 1.5),
  # tools/prep/gen_weights_data.R, W6: y = 1.0 + 0.6*x + u[g] + e, sd(g) = 0.9,
  # sd(e) = 0.7/sqrt(w).
  lmm_intercept = tr(c(IC, "x"), c(1.0, 0.6), list(vcb("g", IC, 0.9)), sigma = 0.7),
  # tools/prep/gen_weights_data.R, W7: y = 1.0 + 0.5*x + b[g] + e, block sds
  # (1.1, 0.6) corr 0.3, sd(e) = 0.7/sqrt(w).
  lmm_slope = tr(c(IC, "x"), c(1.0, 0.5),
                 list(vcb("g", c(IC, "x"), c(1.1, 0.6), 0.3)), sigma = 0.7),
  # tools/prep/gen_weights_data.R, W8: y = 1.0 + 0.5*x + b1 + b2 + e, S1 sds
  # (1.1, 0.6) corr 0.3, S2 sds (0.9, 0.5) corr 0.2, sd(e) = 0.7/sqrt(w).
  lmm_crossed = tr(c(IC, "x"), c(1.0, 0.5),
                   list(vcb("g1", c(IC, "x"), c(1.1, 0.6), 0.3),
                        vcb("g2", c(IC, "x"), c(0.9, 0.5), 0.2)),
                   sigma = 0.7),
  # tools/prep/gen_weights_data.R, W9: y ~ rpois(exp(0.4 + 0.4*x + u[g])),
  # sd(g) = 0.5.
  glmm_poisson = tr(c(IC, "x"), c(0.4, 0.4), list(vcb("g", IC, 0.5))),
  # tools/prep/gen_weights_data.R, W10: the make_sparse_binomial recipe --
  # eta = 0.2 + 0.5*x + u_g1 + sum_k u_ck, sd(g1) = 0.8,
  # sd(c1..c7) = (0.50, 0.45, 0.40, 0.50, 0.40, 0.45, 0.35).
  glmm_binomial = tr(c(IC, "x"), c(0.2, 0.5),
    c(list(vcb("g1", IC, 0.8)),
      mapply(function(nm, s) vcb(nm, IC, s), paste0("c", 1:7),
             c(0.50, 0.45, 0.40, 0.50, 0.40, 0.45, 0.35), SIMPLIFY = FALSE,
             USE.NAMES = FALSE))),
  # tools/prep/gen_weights_data.R, P1: y = 1.0 + 0.5*x + e with sd(e) = 0.8
  # HOMOSKEDASTIC while the fit is weighted by a 1e-6..1e6 `w` column, so the
  # weighted model has no true residual sd and sigma is left null on purpose.
  path_extreme_range = tr(c(IC, "x"), c(1.0, 0.5)),
  # tools/prep/gen_weights_data.R, P3: y = 1.0 + 0.6*x + u[g] + e, sd(g) = 0.9,
  # sd(e) = 25/sqrt(w), so 25 IS the weighted residual sd.
  path_huge_int = tr(c(IC, "x"), c(1.0, 0.6), list(vcb("g", IC, 0.9)), sigma = 25),
  # tools/prep/gen_weights_data.R, P4: y = 1.0 + 0.5*x + e, sd(e) = 0.8 homoskedastic
  # while one row carries 99% of sum(w) -- same reason as P1 for a null sigma.
  path_dominant = tr(c(IC, "x"), c(1.0, 0.5)),
  # tools/prep/gen_weights_data.R, U1: y = 1.0 + 0.5*x + b[g] + e, block sds
  # (1.0, 0.6) corr 0.25, residual 0.7, and w is identically 1 -- the name is
  # about the weight column, not about the design, so the parameter vector is
  # an ordinary one.
  all_ones = tr(c(IC, "x"), c(1.0, 0.5),
                list(vcb("g", c(IC, "x"), c(1.0, 0.6), 0.25)), sigma = 0.7),

  # tools/prep/gen_large_theta_data.R, R1: eta = 0.5 + 0.8*x - 0.6*z + b[g],
  # sd(g) = 5.54 (a tuned generating sd; the FITTED theta-hat is about 4.5,
  # because Laplace is biased low in this regime).
  sim_binomial_bigsd = tr(c(IC, "x", "z"), c(0.5, 0.8, -0.6),
                          list(vcb("g", IC, 5.54))),
  # tools/prep/gen_large_theta_data.R, R2: eta = -0.8 + 0.5*x - 0.4*z + b[g],
  # sd(g) = 2.70 (tuned; fitted theta-hat about 3).
  sim_poisson_bigsd = tr(c(IC, "x", "z"), c(-0.8, 0.5, -0.4),
                         list(vcb("g", IC, 2.70))),
  # tools/prep/gen_large_theta_data.R, R3: eta = 0.3 + 0.5*x - 0.4*z with NO b[g] term
  # at all -- the grouping's true sd is EXACTLY zero, which is the fixture's
  # whole point.
  sim_binomial_zerosd = tr(c(IC, "x", "z"), c(0.3, 0.5, -0.4),
                           list(vcb("g", IC, 0))),
  # tools/prep/gen_large_theta_data.R, R4: eta = 0.5 + 0.5*x - 0.4*z + b1[g1] +
  # sum_k bc_k, sd(g1) = 4.00 (tuned; fitted theta-hat 3.91), sd(c1..c7) = 0.5.
  sim_sparse_binomial_bigsd = tr(c(IC, "x", "z"), c(0.5, 0.5, -0.4),
    c(list(vcb("g1", IC, 4.00)),
      mapply(function(nm, s) vcb(nm, IC, s), paste0("c", 1:7), rep(0.5, 7),
             SIMPLIFY = FALSE, USE.NAMES = FALSE))),

  # tools/prep/gen_probit_large_data.R: eta = 0.3 + 0.5*x1 - 0.4*x2 + 0.25*x3 -
  # 0.6*z + b[g], sd(g) = 0.7; y ~ rbinom(1, pnorm(eta)).
  sim_probit_large = tr(c(IC, "x1", "x2", "x3", "z"),
                        c(0.3, 0.5, -0.4, 0.25, -0.6),
                        list(vcb("g", IC, 0.7))),

  # tools/prep/gen_igauss_data.R: eta = log(mu) = 0.4 + 0.05*x + 0.05*(grp=="b"),
  # dispersion phi = 0.3 (R's inverse.gaussian convention, V(mu) = phi*mu^3).
  # Only the LOG-link cell can claim this vector; the 1/mu^2 cell refits the
  # same data on another link and is in FIXTURE_NO_TRUTH.
  sim_igauss_glm = tr(c(IC, "x", "grpb"), c(0.4, 0.05, 0.05), dispersion = 0.3),

  # tools/prep/gen_scale_data.R, scale_logit(): p = plogis(-0.3 + 2*x), y a
  # deterministic Bernoulli draw off the same LCG stream.
  sim_scale_logit_glm = tr(c(IC, "x"), c(-0.3, 2.0)),
  # tools/prep/gen_scale_data.R, scale_gamma_inv(): eta = 1/mu = 100 - 20*x EXACTLY,
  # and y = (1 + 0.1*u)/eta is a deterministic +/-10% perturbation rather than a
  # Gamma draw, so the dispersion has no generating value and stays null.
  sim_scale_gamma_inv_glm = tr(c(IC, "x"), c(100, -20))
)

FIXTURE_NO_TRUTH <- list(
  # Named reason per cell. These are not "unknown"; each is a fixture built
  # around a numerical property rather than around a parameter vector.
  sim_collinear_glm = "near-collinear design (x3 = x1 + x2 + 1e-13 jitter); the aliased column has no true beta",
  sim_collinear_lmm = "near-collinear design (x3 = x1 + x2 + 1e-13 jitter) on a mixed model; the aliased column has no true beta",
  sim_dynrange_lmm = "tools/prep/gen_illcond_data.R draws the random effect and the residual from a deterministic 16-bit LCG uniform, not from normals, so the variance parameters have no generating value",
  sim_entangled_pair_lmm = "tools/prep/gen_illcond_data.R: the same LCG uniform draws as sim_dynrange_lmm, and t and v are entangled to 3e-6 so neither slope is separately identified",
  sim_igauss_inv_sq_glm = "the same data as sim_igauss_glm refitted under 1/mu^2; the generating vector is on the log scale and is not this fit's truth",
  sim_scale_sep_glm = "complete separation (y = 1[x > 0]); the maximum-likelihood beta is infinite, so there is no parameter vector",
  path_near_zero = "tools/prep/gen_weights_data.R draws the near-zero-weight block from a different slope (y = 5.0 - 2.0*x) than the rest, so the fixture has no single parameter vector"
)

# Cells whose data admit NO maximum-likelihood estimate, with the reason. On
# these the only correct result is a refusal, so the cell carries `no_mle` and
# compare.R gates the other way round: glmm's refusal passes, and a converged
# glmm fit FAILS. Membership is a property of the data, proven when the entry is
# added, never a way to excuse a fit glmm got wrong; a cell that has an optimum
# does not belong here however badly an engine does on it.
NO_MLE <- list(
  # max x over y = 0 is -0.0177 and min x over y = 1 is 0.0033, so the
  # Bernoulli deviance is positive at every finite beta and tends to 0 along
  # beta = c * (-m, 1), m in the gap, as c -> infinity. Albert, A. & Anderson,
  # J. A. (1984), On the existence of maximum likelihood estimates in logistic
  # regression models, Biometrika 71, 1-10.
  sim_scale_sep_glm = "complete separation (y = 1[x > 0]): the deviance has no minimum at any finite beta, so the only correct result is a refusal"
)

# Cells whose reference engines cannot gate gate 1's deviance because their OWN
# round-off on this data is larger than dev_eps, with the reason. compare.R
# reads validation/grid/dev_ref.json for these instead of the oracle records --
# a frozen high-precision value, not a registry escape: gate 1 keeps the normal
# dev_eps and the normal sign rule, just against a different reference.
# Membership is a property of the data (proven when the entry is added), never
# a way to excuse a fit glmm got wrong.
DEV_REF <- list(
  sim_entangled_pair_lmm = "t and v are entangled to 3e-6 (tools/prep/gen_illcond_data.R), so lme4, glmmTMB and glmm each converge to a slightly different theta and their own reported REML criterion carries round-off of 1e-4 to 7e-4 against the true optimum -- above dev_eps even though all three theta agree with each other to 1e-10 on the true objective"
)

FIXTURE_SKIPPED <- list(
  # data/simulated/ or data/empirical/ CSV stems that become no cell at all,
  # with the reason. The self-checks below assert this list plus the cells
  # covers every CSV under both directories, so a fixture cannot vanish
  # quietly.
  #
  # EXPECTED TO HOLD EXACTLY ONE ENTRY. Every simulated fixture -- including
  # sim_binomial_zerosd and sim_entangled_pair_lmm -- becomes a cell. A second
  # entry here would mean dropping a committed fixture from the grid; add the
  # cell instead.
  InstEval = "not one of the eleven empirical datasets the grid covers; 73421 rows x a crossed d/s/dept design would take a large share of the one-night oracle budget"
)

# ---- per-cell oracles --------------------------------------------------------
# Number of grouping factors and the width of the first random-effect block.
# Generated cells read both off STRUCTURES; a read-in cell has no structure
# code, so they are parsed out of its r_formula's `(terms | grouping)` parts.
# A `/` inside a grouping is two groupings ((1 | g1/g2)); a `:` or `&` is one.
RE_TERM_RE <- "\\(([^()|]*)\\|([^()]*)\\)"
re_terms_of <- function(r_formula) {
  parts <- regmatches(r_formula, gregexpr(RE_TERM_RE, r_formula))[[1]]
  lapply(parts, function(p) {
    inner <- substr(p, 2, nchar(p) - 1)
    sides <- strsplit(inner, "|", fixed = TRUE)[[1]]
    list(terms = trimws(strsplit(trimws(sides[1]), "+", fixed = TRUE)[[1]]),
         groups = trimws(strsplit(trimws(sides[2]), "/", fixed = TRUE)[[1]]))
  })
}
re_shape <- function(cell) {
  if (identical(cell$structure, "glm")) return(list(n_groups = 0L, q1 = 0L))
  if (!is.null(cell$structure)) {
    st <- STRUCTURES[[cell$structure]]
    return(list(n_groups = length(st$q), q1 = st$q[1]))
  }
  rt <- re_terms_of(cell$r_formula)
  list(n_groups = sum(vapply(rt, function(t) length(t$groups), 0L)),
       q1 = length(rt[[1]]$terms))
}
# n_theta counts FORMULA terms, so a factor-valued random slope (Machines'
# (1 + Machine | Worker)) counts as one term rather than as its contrast
# columns; on the generated cells the two agree exactly.
n_theta_readin <- function(r_formula) {
  rt <- re_terms_of(r_formula)
  sum(vapply(rt, function(t) {
    q <- length(t$terms)
    length(t$groups) * q * (q + 1) / 2
  }, 0))
}

# Which oracles apply to a cell, mechanised so the engine scripts never have to
# decide for themselves. An engine is listed here ONLY if it can actually fit
# the cell: run.sh hands each engine exactly the cells its name appears on, the
# engines assert it, and a missing record is therefore a failure rather than an
# expected absence.
oracles_of <- function(cell) {
  o <- character(0)
  agq <- !is.null(cell$nagq) && cell$nagq > 1
  shape <- re_shape(cell)
  n_groups <- shape$n_groups
  q1 <- shape$q1

  # lme4: every family (NB through glmer.nb), and GLM cells through stats::glm /
  # stats::lm / MASS::glm.nb inside the same script. Two AGQ exclusions:
  #   - glmer refuses nAGQ > 1 for a VECTOR RE, so a q2s AGQ cell has no lme4 arm;
  #   - on an AGQ cell lme4's logLik omits the saturated term, and the correction
  #     for it is a closed form that exists only for binomial and poisson.
  #     Keeping lme4 on a gamma or NB AGQ cell would put a record in the
  #     reference that no alignment can place on the shared deviance scale.
  lme4_ok <- !(agq && q1 > 1L) &&
             !(agq && !(cell$family %in% c("binomial", "poisson")))
  if (lme4_ok) o <- c(o, "lme4")

  # glmmTMB: every cell EXCEPT inverse-Gaussian. Checked against the pinned
  # glmmTMB: its family list has no inverse-Gaussian entry, so the two
  # inverse-Gaussian GLM fixture cells would be an error, not a missing number.
  # Laplace only; REML = TRUE on gaussian cells; fits GLM cells with no RE term.
  if (cell$family != "inversegaussian") o <- c(o, "glmmTMB")

  # MixedModels.jl: gaussian, binomial logit and probit, poisson. No cloglog, no
  # Gamma, no NB, no inverse-Gaussian, no AGQ arm -- so it is dropped on an AGQ
  # cell rather than contributing a Laplace number to an AGQ comparison.
  mm_ok <- (cell$family == "gaussian") ||
           (cell$family == "binomial" && cell$link %in% c("logit", "probit")) ||
           (cell$family == "poisson")
  if (mm_ok && !agq) o <- c(o, "MixedModels")

  # GLMMadaptive: the AGQ oracle, and only that. It takes an AGQ cell with
  # EXACTLY ONE grouping factor, at the cell's nagq, which makes it the oracle
  # for the vector-RE AGQ cells lme4 refuses. Five exclusions:
  #   - LAPLACE cells. Driven to nAGQ = 1 mixed_model does not land on a
  #     reliable optimum: measured on a Bernoulli pilot cell it returned a
  #     random-effect sd of 4.6e-16 and a log-likelihood 58 units worse than
  #     the same fit at its own default quadrature, while reporting
  #     convergence. A degenerate optimum reported as converged is worse than
  #     no record, so the Laplace comparison is left to the other three engines.
  #   - no fixed-only path, so GLM cells are out;
  #   - no inverse-Gaussian family;
  #   - mixed_model's `weights=` is a per-CLUSTER replicate multiplier, NOT
  #     per-row prior weights, so a `weights_col` cell cannot be fitted here on
  #     the same convention as every other engine. Excluded rather than fitted
  #     on a different objective.
  #   - a non-log Gamma link: GLMMadaptive::Gamma.fam() is log-link only, so
  #     an inverse-link cell would be fitted as a different model.
  ma_ok <- !is.null(cell$nagq) &&
           cell$family != "gaussian" && cell$family != "inversegaussian" &&
           !(cell$family == "gamma" && cell$link != "log") &&
           n_groups == 1L && is.null(cell$weights_col)
  if (ma_ok) o <- c(o, "GLMMadaptive")
  o
}

# GLMMadaptive's two formula halves: the fixed part is r_formula with the RE
# terms stripped (keeping a cbind(...) response, which is its binomial-trials
# form) and keeping any offset(...) term, which is how the offset reaches it.
ma_of <- function(cell) {
  fixed <- trimws(gsub(paste0("\\s*\\+\\s*", RE_TERM_RE), "", cell$r_formula))
  rt <- re_terms_of(cell$r_formula)[[1]]
  list(fixed = fixed,
       random = sprintf("~ %s | %s", paste(rt$terms, collapse = " + "),
                        paste(rt$groups, collapse = "/")))
}

# ---- cell assembly -----------------------------------------------------------
FIELD_ORDER <- c("cell", "family", "link", "form", "structure", "n_theta",
                 "n_obs", "per_group", "balance", "regime", "seed", "data",
                 "factors", "n_x", "response", "r_formula", "jl_formula",
                 "glmm_formula", "ma_fixed", "ma_random", "weights",
                 "weights_col", "offset_col", "nagq", "reml", "oracles",
                 "tags", "truth", "truth_why", "no_mle", "dev_ref")

cells <- list()
emit_cell <- function(c0) {
  c0$oracles <- I(oracles_of(c0))
  if ("GLMMadaptive" %in% c0$oracles) {
    ma <- ma_of(c0)
    c0$ma_fixed <- ma$fixed
    c0$ma_random <- ma$random
  }
  c0$tags <- I(character(0))
  stopifnot("a cell carries a field outside the schema" =
    all(names(c0) %in% FIELD_ORDER))
  cells[[length(cells) + 1L]] <<- c0[intersect(FIELD_ORDER, names(c0))]
  invisible(NULL)
}

add_generated <- function(arm, structure, n_obs, per, balance, regime,
                          nagq = NULL, offset_col = NULL, weights_col = NULL) {
  st <- if (identical(structure, "glm")) NULL else STRUCTURES[[structure]]
  c0 <- list(family = arm$family, link = arm$link)
  if (!is.na(arm$form)) c0$form <- arm$form
  c0$structure <- structure
  c0$n_theta <- if (is.null(st)) 0 else n_theta_of(st)
  c0$n_obs <- n_obs
  if (!is.null(st)) {
    c0$per_group <- per
    c0$balance <- balance
    c0$regime <- regime
  }
  c0$n_x <- if (is.null(st)) GLM_N_X else nx_of(st)
  if (!is.null(nagq)) c0$nagq <- nagq
  if (!is.null(offset_col)) c0$offset_col <- offset_col
  if (!is.null(weights_col)) c0$weights_col <- weights_col
  c0$cell <- cell_id_of(c0, arm$tag)
  c0$seed <- seed_of(c0$cell)
  c0$data <- sprintf("data/%s.csv", c0$cell)
  c0$factors <- I(if (is.null(st)) character(0) else group_names(st))
  c0$response <- if (identical(c0$form, "trials")) "incidence" else "y"
  if (identical(c0$form, "trials")) c0$weights <- "size"
  f <- formulas_of(c0)
  c0$r_formula <- f$r
  c0$jl_formula <- f$jl
  c0$glmm_formula <- f$glmm
  c0$reml <- identical(c0$family, "gaussian")
  c0$truth <- truth_of(c0)
  emit_cell(c0)
}

# ---- the crossing ------------------------------------------------------------
# 1. Core: every arm x every structure it can identify, at the core point.
for (arm in ARMS) for (sname in names(STRUCTURES)) {
  if (!feasible(STRUCTURES[[sname]], CORE$n_obs, CORE$per_group)) next
  add_generated(arm, sname, CORE$n_obs, CORE$per_group, CORE$balance, CORE$regime)
}

# 2. Size variants: one axis off the core, canonical arm per family.
for (fam in names(REPRESENTATIVE)) {
  arm <- ARMS[[CANONICAL_ARM[[fam]]]]
  for (sname in REPRESENTATIVE[[fam]]) for (sz in SIZES) {
    if (sz[1] == CORE$n_obs && sz[2] == CORE$per_group) next
    if (!feasible(STRUCTURES[[sname]], sz[1], sz[2])) next
    add_generated(arm, sname, sz[1], sz[2], CORE$balance, CORE$regime)
  }
}

# 3. Balance variants.
for (fam in names(REPRESENTATIVE)) {
  arm <- ARMS[[CANONICAL_ARM[[fam]]]]
  for (sname in REPRESENTATIVE[[fam]]) for (b in BALANCES) {
    if (!feasible(STRUCTURES[[sname]], CORE$n_obs, CORE$per_group)) next
    add_generated(arm, sname, CORE$n_obs, CORE$per_group, b, CORE$regime)
  }
}

# 4. Regime variants. highcorr is skipped on a scalar block: there is no
# correlation to raise.
for (fam in names(REPRESENTATIVE)) {
  arm <- ARMS[[CANONICAL_ARM[[fam]]]]
  for (sname in REPRESENTATIVE[[fam]]) {
    if (!feasible(STRUCTURES[[sname]], CORE$n_obs, CORE$per_group)) next
    for (r in c(BASE_REGIMES, EXTRA_REGIME[[fam]])) {
      if (r == "highcorr" && max(STRUCTURES[[sname]]$q) < 2) next
      add_generated(arm, sname, CORE$n_obs, CORE$per_group, CORE$balance, r)
    }
  }
}

# 5. AGQ variants, non-gaussian families only. Bernoulli / count forms only:
# glmm refuses `weights` with nAGQ > 1 at its model-shape gate, so a trials AGQ
# cell would be a guaranteed glmm engine-fail and would measure nothing.
for (fam in setdiff(names(REPRESENTATIVE), "gaussian")) {
  arm <- ARMS[[CANONICAL_ARM[[fam]]]]
  if (identical(arm$form, "trials")) next
  for (sname in c("int1", "q2s"))
    add_generated(arm, sname, CORE$n_obs, CORE$per_group, CORE$balance,
                  CORE$regime, nagq = AGQ_K)
}

# 6. GLM (no random effect) cells, one per arm.
for (arm in ARMS)
  add_generated(arm, "glm", CORE$n_obs, NULL, NULL, NULL)

# 7. Offset cells: a known additive term on the linear-predictor scale.
for (fam in c("poisson", "negativebinomial")) {
  arm <- ARMS[[CANONICAL_ARM[[fam]]]]
  for (sname in c("int1", "nest2"))
    add_generated(arm, sname, CORE$n_obs, CORE$per_group, CORE$balance,
                  CORE$regime, offset_col = "log_exposure")
}

# 8. Prior-weight cells. Never on a binomial-trials cell -- there `weights`
# already carries the trial count and the two fields are mutually exclusive.
for (fam in c("gaussian", "poisson", "gamma")) {
  arm <- ARMS[[CANONICAL_ARM[[fam]]]]
  for (sname in c("int1", "q2s"))
    add_generated(arm, sname, CORE$n_obs, CORE$per_group, CORE$balance,
                  CORE$regime, weights_col = "w")
}

# ---- read-in cells -----------------------------------------------------------
old <- fromJSON(file.path(VALIDATION, "manifest.json"), simplifyDataFrame = FALSE)

FAMILY_RENAME <- c(negbin = "negativebinomial")
CANONICAL_LINK <- c(gaussian = "identity", binomial = "logit", poisson = "log",
                    gamma = "log", negativebinomial = "log",
                    inversegaussian = "log")

response_of_r <- function(r) {
  lhs <- trimws(strsplit(r, "~", fixed = TRUE)[[1]][1])
  if (startsWith(lhs, "cbind("))
    trimws(strsplit(sub("^cbind\\(", "", lhs), ",", fixed = TRUE)[[1]][1])
  else lhs
}
# A Julia-dialect formula for a fixture that has none: same right-hand side,
# with an aggregated-binomial cbind(...) response rewritten to the `prop`
# column every engine synthesizes. None of the fixtures that need this carries
# a `:` grouping, so no `&` rewrite is required going this way.
jl_from_r <- function(r) {
  sides <- strsplit(r, " ~ ", fixed = TRUE)[[1]]
  sprintf("@formula(%s ~ %s)",
          if (startsWith(trimws(sides[1]), "cbind(")) "prop" else trimws(sides[1]),
          paste(sides[-1], collapse = " ~ "))
}
# Keep the offset in the fixed part, before the first random-effect term, the
# same order formulas_of emits it in.
add_offset_term <- function(f, col) {
  term <- sprintf("offset(%s)", col)
  if (grepl(" + (1", f, fixed = TRUE))
    sub(" + (1", sprintf(" + %s + (1", term), f, fixed = TRUE)
  else sub(")$", sprintf(" + %s)", term), f)
}

add_readin <- function(entry, subdir, name = entry[["name"]]) {
  fam <- entry[["family"]]
  if (fam %in% names(FAMILY_RENAME)) fam <- unname(FAMILY_RENAME[fam])
  stem <- if (!is.null(entry[["data"]])) entry[["data"]] else entry[["name"]]
  path <- file.path(VALIDATION, "data", subdir, paste0(stem, ".csv"))
  r_formula <- entry[["r_formula"]]
  jl_formula <- if (!is.null(entry[["jl_formula"]])) entry[["jl_formula"]] else jl_from_r(r_formula)

  c0 <- list(cell = name, family = fam,
             link = if (!is.null(entry[["link"]])) entry[["link"]] else CANONICAL_LINK[[fam]])
  if (!grepl(RE_TERM_RE, r_formula)) c0$structure <- "glm"
  c0$n_theta <- if (identical(c0$structure, "glm")) 0 else n_theta_readin(r_formula)
  c0$n_obs <- nrow(read.csv(path))
  c0$data <- sprintf("../data/%s/%s.csv", subdir, stem)
  c0$factors <- I(as.character(unlist(entry[["factors"]])))
  c0$response <- response_of_r(r_formula)
  # glmm_formula is taken BEFORE the offset term goes in: glmm and the two
  # ports read the offset out of offset_col through the fit options, so their
  # formula must never carry it.
  c0$glmm_formula <- glmm_formula_of(jl_formula)
  if (!is.null(entry[["offset"]])) {
    c0$offset_col <- entry[["offset"]]
    r_formula <- add_offset_term(r_formula, entry[["offset"]])
    jl_formula <- add_offset_term(jl_formula, entry[["offset"]])
  }
  c0$r_formula <- r_formula
  c0$jl_formula <- jl_formula
  if (!is.null(entry[["weights"]])) c0$weights <- entry[["weights"]]
  if (!is.null(entry[["weights_col"]])) c0$weights_col <- entry[["weights_col"]]
  c0$reml <- identical(fam, "gaussian")
  if (identical(subdir, "empirical")) {
    c0$truth_why <- "empirical data; no generating parameter vector"
  } else if (!is.null(FIXTURE_TRUTH[[name]])) {
    c0$truth <- FIXTURE_TRUTH[[name]]
  } else {
    c0$truth_why <- FIXTURE_NO_TRUTH[[name]]
  }
  if (!is.null(NO_MLE[[name]])) c0$no_mle <- NO_MLE[[name]]
  if (!is.null(DEV_REF[[name]])) c0$dev_ref <- DEV_REF[[name]]
  emit_cell(c0)
}

# 9. Empirical cells: every non-sim entry of the crate's own manifest except the
# one named in FIXTURE_SKIPPED. Family, link, reml, factors, weights and the two
# formulas are copied, not re-derived.
for (entry in old$datasets) {
  if (identical(entry[["source"]], "sim")) next
  if (!is.null(FIXTURE_SKIPPED[[entry[["name"]]]])) next
  add_readin(entry, "empirical")
}

# 10a. Committed simulated fixtures that are curated rungs.
for (entry in old$datasets) {
  if (!identical(entry[["source"]], "sim")) next
  add_readin(entry, "simulated")
}

# 10b. The goldens that read a committed CSV no rung covers -- that is what
# keeps a fixture from vanishing just because it is a golden rather than a rung.
# The two inverse-Gaussian GLM cells arrive here.
GOLDEN_CELLS <- c(
  "sim_collinear_glm", "sim_collinear_lmm", "sim_dynrange_lmm",
  "sim_igauss_glm", "sim_igauss_inv_sq_glm", "sim_nb_glm", "sim_nb_glmm",
  "sim_nb_hightheta_glm", "sim_nb_lowtheta_glm", "sim_nb_nested_glmm",
  "sim_poisson_highmean_glm", "sim_scale_gamma_inv_glm", "sim_scale_logit_glm",
  "sim_scale_sep_glm", "sim_slope_lmm", "sim_sparse_nb",
  "sim_wide_crossed_lmm", "sim_wide_slopes_lmm")
golden_by_name <- setNames(old$m3_goldens, vapply(old$m3_goldens, `[[`, "", "name"))
for (nm in GOLDEN_CELLS) {
  entry <- golden_by_name[[nm]]
  stopifnot("a named golden is not in the crate's manifest" = !is.null(entry))
  add_readin(entry, "simulated")
}

# 10c. The two committed CSVs neither list reaches. Both become cells: a
# zero-variance-component fixture and an entangled-pair LMM are exactly the
# shapes the grid exists to keep. The model comes from the header of the
# generator that wrote each CSV.
add_readin(list(name = "sim_binomial_zerosd", family = "binomial",
                link = "logit", factors = list("g"), weights = "size",
                r_formula = "cbind(incidence, size - incidence) ~ 1 + x + z + (1 | g)"),
           "simulated")
add_readin(list(name = "sim_entangled_pair_lmm", family = "gaussian",
                factors = list("g"),
                r_formula = "y ~ 1 + t + v + z + (1 | g)"),
           "simulated")

# ---- the pilot and fast sets -------------------------------------------------
# Both are built by SELECTION from the cells already enumerated above -- never by
# sampling -- so the assertions below prove the span rather than hope for it.
# Each helper returns the FIRST matching cell id in enumeration order
# (deterministic) or "" when the shape does not exist; `need` turns a missing
# REQUIRED shape into a loud stop rather than a silently short set.
pick <- function(pred) {
  hit <- Filter(pred, cells)
  if (!length(hit)) "" else hit[[1]]$cell
}
need <- function(what, id) {
  if (!nzchar(id)) stop(sprintf("a %s cell is required and the crossing produced none", what))
  id
}
same_form <- function(c, a) if (is.na(a$form)) is.null(c$form) else identical(c$form, a$form)
at_core <- function(c)
  identical(c$n_obs, CORE$n_obs) && identical(c$per_group, CORE$per_group) &&
  identical(c$balance, CORE$balance) && identical(c$regime, CORE$regime) &&
  is.null(c$nagq) && is.null(c$offset_col) && is.null(c$weights_col)
# The small end of the size axis: a generated mixed cell of at most 300 rows,
# otherwise at the core point on every axis. `per_group` is the field that is
# absent on a GLM cell and on every read-in cell, so testing it first is what
# keeps those out.
at_small <- function(c)
  !is.null(c$per_group) && c$n_obs <= 300L &&
  identical(c$balance, "bal") && identical(c$regime, "base") &&
  is.null(c$nagq) && is.null(c$offset_col) && is.null(c$weights_col)

# The scalar cell of every family/link/form arm at the core point: 11 ids that
# between them reach every family, every link, every form and every oracle. Both
# selected sets open on it, so it is one function rather than two copies.
core_arm_ids <- function()
  vapply(ARMS, function(a) need(sprintf("core %s/%s", a$family, a$link),
    pick(function(c) at_core(c) && identical(c$structure, "int1") &&
      identical(c$family, a$family) && identical(c$link, a$link) && same_form(c, a))), "")

PILOT_IDS <- unique(c(
  # (1) one core cell per family/link/form arm.
  core_arm_ids(),
  # (2) one GLM cell per family -> the closed-form deviance constants are
  #     confirmed on GLM cells first. SIX families, not five: the
  #     inverse-Gaussian GLM fixture cells are the only inversegaussian cells in
  #     the grid, and the self-checks assert the pilot spans every family present.
  vapply(c("gaussian", "binomial", "poisson", "gamma", "negativebinomial"), function(f)
    need(paste(f, "GLM"), pick(function(c) identical(c$structure, "glm") &&
      identical(c$family, f))), ""),
  need("inverse-Gaussian GLM", pick(function(c) identical(c$family, "inversegaussian"))),
  # (3) the size corners on binomial/logit bernoulli int1 -> the per-cell wall
  #     ladder and lme4's se_hessian above 10000 rows. The core point (3000,20)
  #     is one of them, so the ladder has no hole.
  vapply(list(c(60L, 5L), c(300L, 20L), c(3000L, 20L), c(3000L, 100L), c(30000L, 20L)),
    function(sz) need(sprintf("binomial int1 g%dp%d", sz[1], sz[2]),
      pick(function(c) identical(c$structure, "int1") && identical(c$family, "binomial") &&
        identical(c$link, "logit") && identical(c$form, "bernoulli") &&
        identical(c$n_obs, sz[1]) && identical(c$per_group, sz[2]) &&
        identical(c$balance, "bal") && identical(c$regime, "base") && is.null(c$nagq))), ""),
  # (4) the two WIDE 30000-row cells -> the RAM watch. A size variant only
  #     exists for a structure in REPRESENTATIVE, which is why that list carries
  #     q4sx2 (gaussian) and q8 (binomial); `need` fails loudly if a later trim
  #     removes them.
  need("wide gaussian 30000-row", pick(function(c) identical(c$family, "gaussian") &&
    identical(c$n_obs, 30000L) && !is.null(c$structure) &&
    !identical(c$structure, "glm") && STRUCTURES[[c$structure]]$q[1] >= 4L)),
  need("wide binomial 30000-row", pick(function(c) identical(c$family, "binomial") &&
    identical(c$n_obs, 30000L) && !is.null(c$structure) &&
    !identical(c$structure, "glm") && STRUCTURES[[c$structure]]$q[1] >= 4L)),
  # (5) one scalar AGQ and one vector AGQ cell -> lme4's nAGQ>1 saturated
  #     correction and GLMMadaptive's vector-RE arm, the one lme4 refuses.
  need("scalar AGQ", pick(function(c) !is.null(c$nagq) && identical(c$structure, "int1"))),
  need("vector AGQ", pick(function(c) !is.null(c$nagq) && identical(c$structure, "q2s"))),
  # (6) nearzero and boundary on binomial int1 -> the truth floor.
  need("binomial nearzero", pick(function(c) identical(c$structure, "int1") &&
    identical(c$family, "binomial") && identical(c$regime, "nearzero"))),
  need("binomial boundary", pick(function(c) identical(c$structure, "int1") &&
    identical(c$family, "binomial") && identical(c$regime, "boundary"))),
  # (7) one offset cell and one prior-weight cell -> those two wiring paths are
  #     exercised before the overnight run, not during it.
  need("offset", pick(function(c) !is.null(c$offset_col))),
  need("prior-weight", pick(function(c) !is.null(c$weights_col))),
  # (8) the read-in paths: two empirical cells and two committed fixtures, one of
  #     each with truth and one without.
  "cbpp", "sleepstudy", "sim_gamma", "sim_slope_lmm"
))
stopifnot("pilot set is not ~30 cells" =
  length(PILOT_IDS) >= 26L && length(PILOT_IDS) <= 38L)
stopifnot("PILOT_IDS names a cell that does not exist" =
  all(PILOT_IDS %in% vapply(cells, `[[`, "", "cell")))
for (i in seq_along(cells))
  if (cells[[i]]$cell %in% PILOT_IDS) cells[[i]]$tags <- I("pilot")

# ---- the fast set ------------------------------------------------------------
# About 85 cells spanning family x structure x size, the subset CI can afford to
# fit on every push and the only subset a timed run is allowed to use. Every
# family and every structure class appears at least once, every empirical cell is
# in, and a serial glmm pass over the set stays under a minute -- which is met by
# choosing the GENERATED members, the only axis open here. There is deliberately
# NO row-count ceiling: VerbAgg has 7584 rows and is an empirical cell, so it is
# in regardless.
EMPIRICAL_IDS <- vapply(Filter(function(c)
  startsWith(c$data, "../data/empirical/"), cells), `[[`, "", "cell")

# Committed fixtures whose pathology is worth re-fitting on every push. Each one
# is here for a shape the generated grid does not reach: the prior-weight
# convention on four families, the crossed-grouping level cap, the widest single
# slope block in the corpus, both ends of the negative-binomial theta range, an
# exactly-zero variance component, and the only inverse-Gaussian data in the
# grid -- without that last one `fast` would miss a whole family.
FAST_FIXTURES <- c(
  "wls_basic", "glm_gamma", "glm_nb", "glmm_poisson",
  "sim_crossed_at_cap", "sim_max_q_slope",
  "sim_nb_hightheta_glm", "sim_nb_lowtheta_glm",
  "sim_binomial_zerosd", "sim_igauss_glm")

FAST_IDS <- unique(c(
  # (1) every family/link/form arm at the core point.
  core_arm_ids(),
  # (2) each structure class, at the smallest size that identifies it, on the
  #     canonical arm of every family. Size variants exist only for the codes in
  #     REPRESENTATIVE, and the enumeration walks structure-major then size, so
  #     the first small hit is the cheapest cell of that class. Per family rather
  #     than once overall, because that is what makes the set span family x
  #     structure and it costs nothing: none of these cells exceeds 300 rows.
  unlist(lapply(names(REPRESENTATIVE), function(fam) {
    codes <- REPRESENTATIVE[[fam]]
    vapply(STRUCTURE_CLASSES, function(cls) {
      of_class <- codes[vapply(codes, st_class, "") == cls]
      need(sprintf("small %s %s", fam, cls),
        pick(function(c) at_small(c) && identical(c$family, fam) &&
          !is.null(c$structure) && c$structure %in% of_class))
    }, "")
  })),
  # (3) four size points on one arm, all at the small end -> the size axis is a
  #     ladder in `fast` too, without paying for a 30000-row cell.
  vapply(list(c(60L, 5L), c(300L, 5L), c(300L, 20L), c(3000L, 20L)),
    function(sz) need(sprintf("binomial int1 g%dp%d", sz[1], sz[2]),
      pick(function(c) identical(c$structure, "int1") && identical(c$family, "binomial") &&
        identical(c$link, "logit") && identical(c$form, "bernoulli") &&
        identical(c$n_obs, sz[1]) && identical(c$per_group, sz[2]) &&
        identical(c$balance, "bal") && identical(c$regime, "base") && is.null(c$nagq))), ""),
  # (4) every empirical cell.
  EMPIRICAL_IDS,
  # (5) the committed fixtures above.
  FAST_FIXTURES,
  # (6) the two degenerate-variance regimes on one arm -> the truth floor and the
  #     boundary of the parameter space stay under test.
  vapply(c("boundary", "nearzero"), function(r) need(paste("binomial", r),
    pick(function(c) identical(c$structure, "int1") && identical(c$family, "binomial") &&
      identical(c$regime, r))), "")
))
# (7) the nAGQ = AGQ_K copy of every fast cell adaptive quadrature can fit, so a
# timed run times each such cell under both Laplace and AGQ. The shape rule
# mirrors the kernel's nagq > 1 gate (src/fit/common.rs, assert_model_shape):
# binomial / Poisson / NB / Gamma, one grouping factor, at most 3 random effects
# per group -- change together. A generated copy is a generated cell of its own
# (own id, seed and CSV), built exactly as the section-5 AGQ cells are, and
# reuses one of them when it already exists. A read-in copy reads the same CSV
# under the id <name>_agq<K>.
agq_ok <- function(c) {
  s <- re_shape(c)
  is.null(c$nagq) && s$n_groups == 1L && s$q1 <= 3L &&
    c$family %in% c("binomial", "poisson", "negativebinomial", "gamma")
}
for (c in Filter(function(c) c$cell %in% FAST_IDS && agq_ok(c), cells)) {
  if (!is.null(c$seed)) {
    arm <- Find(function(a) identical(a$family, c$family) &&
                  identical(a$link, c$link) && same_form(c, a), ARMS)
    id <- cell_id_of(modifyList(c, list(nagq = AGQ_K)), arm$tag)
    if (!(id %in% vapply(cells, `[[`, "", "cell")))
      add_generated(arm, c$structure, c$n_obs, c$per_group, c$balance, c$regime,
                    nagq = AGQ_K, offset_col = c$offset_col, weights_col = c$weights_col)
  } else {
    id <- sprintf("%s_agq%d", c$cell, AGQ_K)
    c0 <- c
    c0[c("tags", "oracles", "ma_fixed", "ma_random")] <- NULL
    c0$cell <- id
    c0$nagq <- AGQ_K
    emit_cell(c0)
  }
  FAST_IDS <- c(FAST_IDS, id)
}
stopifnot("FAST_IDS names a cell that does not exist" =
  all(FAST_IDS %in% vapply(cells, `[[`, "", "cell")))
# A `fast` generated cell's CSV lives in data/fast/, which is committed because
# CI reads it. A `fast` read-in cell keeps pointing at the crate's own fixture
# tree -- nothing is copied.
for (i in seq_along(cells))
  if (cells[[i]]$cell %in% FAST_IDS) {
    cells[[i]]$tags <- I(c(cells[[i]]$tags, "fast"))
    if (!is.null(cells[[i]]$seed))
      cells[[i]]$data <- sprintf("data/fast/%s.csv", cells[[i]]$cell)
  }

# ---- self-checks -------------------------------------------------------------
ids <- vapply(cells, `[[`, "", "cell")
stopifnot("duplicate cell id" = !anyDuplicated(ids))
stopifnot("a cell has no n_obs" = all(vapply(cells, function(c) is.numeric(c$n_obs), TRUE)))
stopifnot("a cell has no response column" =
  all(vapply(cells, function(c) is.character(c$response) && nzchar(c$response), TRUE)))
stopifnot("a cell has no glmm_formula" =
  all(vapply(cells, function(c) is.character(c$glmm_formula), TRUE)))
stopifnot("a glmm_formula carries an offset() term -- it would be applied twice" =
  !any(vapply(cells, function(c) grepl("offset(", c$glmm_formula, fixed = TRUE), TRUE)))
stopifnot("an offset cell has no offset( ) term in r_formula" =
  all(vapply(cells, function(c) is.null(c$offset_col) ||
    grepl(sprintf("offset(%s)", c$offset_col), c$r_formula, fixed = TRUE), TRUE)))
stopifnot("a cell has neither truth nor truth_why" =
  all(vapply(cells, function(c) !is.null(c[["truth"]]) || !is.null(c[["truth_why"]]), TRUE)))
stopifnot("a cell has no oracle at all" =
  all(vapply(cells, function(c) length(c$oracles) > 0, TRUE)))
stopifnot("weights and weights_col are mutually exclusive" =
  all(vapply(cells, function(c) is.null(c[["weights"]]) || is.null(c[["weights_col"]]), TRUE)))
stopifnot("no cell carries a max_fun field" =
  all(vapply(cells, function(c) is.null(c$max_fun), TRUE)))

# Every read-in simulated cell is in exactly one of the two truth lists, and
# every committed CSV is either read by a cell or named in FIXTURE_SKIPPED.
# Without this, a fixture drops out of the grid silently and the only symptom is
# a smaller cell count nobody was counting.
# A read-in AGQ copy carries its parent's truth, so only the parents are checked.
fixture_cells <- Filter(function(c) is.null(c$seed) && is.null(c$nagq) &&
                          startsWith(c$data, "../data/simulated/"), cells)
fx_ids <- vapply(fixture_cells, `[[`, "", "cell")
in_truth <- fx_ids %in% names(FIXTURE_TRUTH)
in_none  <- fx_ids %in% names(FIXTURE_NO_TRUTH)
stopifnot("a fixture cell is in neither FIXTURE_TRUTH nor FIXTURE_NO_TRUTH" = all(in_truth | in_none))
stopifnot("a fixture cell is in BOTH truth lists" = !any(in_truth & in_none))
stopifnot("FIXTURE_TRUTH/FIXTURE_NO_TRUTH name a cell that does not exist" =
  all(c(names(FIXTURE_TRUTH), names(FIXTURE_NO_TRUTH)) %in% ids))
# A cell with no MLE has no parameter vector to be the truth of, and NO_MLE is
# only ever applied to read-in fixtures, so a stray name would vanish silently.
stopifnot("NO_MLE names a cell that is not a no-truth fixture" =
  all(names(NO_MLE) %in% intersect(fx_ids, names(FIXTURE_NO_TRUTH))))
# Same shape of check for DEV_REF: a cell with a frozen deviance reference is
# by construction a no-truth fixture too -- its data property is what breaks
# the oracle deviances, not what the fixture is FOR -- so a stray name here
# would otherwise vanish silently the same way.
stopifnot("DEV_REF names a cell that is not a no-truth fixture" =
  all(names(DEV_REF) %in% intersect(fx_ids, names(FIXTURE_NO_TRUTH))))
for (sub in c("simulated", "empirical")) {
  stems <- sub("\\.csv$", "", list.files(file.path(here, "..", "data", sub), pattern = "\\.csv$"))
  used  <- unique(sub("\\.csv$", "", basename(vapply(cells, `[[`, "", "data"))))
  missing <- setdiff(stems, c(used, names(FIXTURE_SKIPPED)))
  if (length(missing))
    stop(sprintf("data/%s CSVs reach no cell and are not in FIXTURE_SKIPPED: %s",
                 sub, paste(missing, collapse = ", ")))
}

# Every family, and every oracle, is reachable from the pilot set.
pilot <- Filter(function(c) "pilot" %in% c$tags, cells)
stopifnot("pilot misses a family" =
  setequal(unique(vapply(pilot, `[[`, "", "family")),
           unique(vapply(cells, `[[`, "", "family"))))
stopifnot("pilot misses an oracle" =
  setequal(unique(unlist(lapply(pilot, `[[`, "oracles"))),
           c("lme4", "glmmTMB", "MixedModels", "GLMMadaptive")))

# The fast set: size, the empirical members the subset is required to carry, and
# the two spans it is chosen for.
fast <- Filter(function(c) "fast" %in% c$tags, cells)
stopifnot("fast set is not ~85 cells" = { n <- length(fast); n >= 50 && n <= 90 })
stopifnot("fast is missing an empirical cell" =
  all(EMPIRICAL_IDS %in% vapply(fast, `[[`, "", "cell")))
stopifnot("fast misses a family" =
  setequal(unique(vapply(fast, `[[`, "", "family")),
           unique(vapply(cells, `[[`, "", "family"))))
stopifnot("fast misses a structure class" =
  setequal(STRUCTURE_CLASSES,
           unique(vapply(Filter(function(c) !is.null(c$structure) &&
                                  c$structure %in% names(STRUCTURES), fast),
                         function(c) st_class(c$structure), ""))))
# gen.R writes a cell wherever its own `data` field points, so the tag and the
# directory have to agree in both directions or a cell is fit on a CSV nobody
# regenerated.
stopifnot("a fast generated cell does not read data/fast/" =
  all(vapply(fast, function(c) is.null(c$seed) ||
    identical(c$data, sprintf("data/fast/%s.csv", c$cell)), TRUE)))
stopifnot("a cell outside the fast set reads data/fast/" =
  all(vapply(cells, function(c) "fast" %in% c$tags ||
    !startsWith(c$data, "data/fast/"), TRUE)))

# ---- write -------------------------------------------------------------------
manifest <- list(schema = "glmm-accuracy-grid/1",
                 generated_by = "grid/gen_manifest.R",
                 generated = format(Sys.Date()),
                 cells = cells)
# digits = NA because truth is an oracle, not a display; na/null = "null"
# because jsonlite's default encodes an R NULL as {}, and every downstream
# is.null() test would then see a non-null empty list.
write(toJSON(manifest, auto_unbox = TRUE, pretty = TRUE, digits = NA,
             na = "null", null = "null"),
      file.path(here, "manifest.json"))

cat(sprintf("cells: %d (pilot %d, fast %d)\n", length(cells), length(pilot), length(fast)))
cat("by family:\n"); print(table(vapply(cells, `[[`, "", "family")))
cat("by origin:\n"); print(table(vapply(cells, function(c)
  if (!is.null(c$seed)) "generated" else "read-in", "")))
if (length(cells) < 600 || length(cells) > 1000)
  stop(sprintf("cell count %d is far outside the 700-900 target -- a crossing bug, not a trim",
               length(cells)))
if (length(cells) < 700 || length(cells) > 900)
  cat(sprintf("NOTE: %d cells, target 700-900. Add or remove ONE structure code from\n  REPRESENTATIVE in gen_common.R and re-run; each one is worth about 30 cells.\n",
              length(cells)))
