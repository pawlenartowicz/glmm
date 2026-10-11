#!/usr/bin/env Rscript
# lmerTest Satterthwaite references -> validation/goldens/inference/lmertest/<name>.json,
# one per `inference_references` entry of kind "lmertest".
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's
# Satterthwaite df, t, p, intervals and contest layer are validated against. On
# any glmm disagreement, glmm is presumed wrong: never relax a tolerance or edit
# a reference.
#
# A BARE RUN OVERWRITES EVERY lmertest REFERENCE. Use VALIDATION_ONLY for
# anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_lmertest.R
# (tools/inference_common.R: select_entries, pending_reference).
#
# THE ENTRIES. Fields beyond the shared schema: `reml` (true, or false for an
# `_ml` twin fitted with REML = FALSE), `weights` (the CSV column of prior
# weights, passed by value, or null), `boundary` ("none", "singular",
# "correlation" or "near_singular") and `proposal_dependent`: false on the
# summary entries `lmertest_<data>[_ml]` (the fit, df and contest layer), true on
# the per-term entries `lmertest_anova_<data>[_ml]` (lmerTest's Type III and
# Type II tables, which depend on its hypothesis matrices).
#
# WHAT ONE JSON HOLDS. The fit is lmerTest::lmer on the entry's data (factors as
# the entry lists them, contr.treatment by name), and its devfun is built as
# lmerTest::lmer builds it (the fit's arguments plus devFunOnly = TRUE and
# check.rankX = "silent.drop.cols"); the script stops unless that devfun
# reproduces the fit's vcov_varpar and Jac_list bit for bit. jsonlite drops the
# names of vectors and matrices, so every name is a string array beside its
# values, and no number written is non-finite (write_reference writes Inf and NA
# both as null; the script stops instead).
#   theta, theta_names       lme4's getME(m, "theta"), equal to its "par" here
#   cnms, group_names, group_levels
#                            one entry per random-effect term in lme4's order:
#                            its column names, grouping factor, level count
#   sigma, beta, coef_names  sigma(m), m@beta, the retained columns
#   cols_dropped             names of attr(model.matrix(m), "col.dropped")
#   vcov_beta                m@vcov_beta, rows and columns coef_names
#   reml_criterion | deviance  REMLcrit(m) on a REML fit, deviance(m) on ML
#   estimate, std_error, df, t_value, p_value
#                            the columns of summary(m)$coefficients
#   level, lower, upper      contest1D(m, e_i, confint = TRUE, level = level)
#   varpar_names             theta_names then "sigma": the coordinates of
#                            vcov_varpar, Jac_list (one p x p matrix per
#                            coordinate) and h
#   h                        boundary entries only: lmerTest's numDeriv Hessian
#                            of devfun_vp at getVarPar(m), numDeriv's defaults
#   fd_settings, df_second   lmerTest's own FD band: df recomputed by contest1D
#                            on each e_i after vcov_varpar and Jac_list are
#                            recomputed at each of two numDeriv settings;
#                            df_second is p x 2, one column per setting
#   conditions               lme4's and lmerTest's messages and warnings, in
#                            order, each with the stage that raised it
#                            (stage "devfun": raised by this script's own
#                            devfun rebuild, not by lmerTest)
# REML entries also carry the contest inputs: contest_md (p >= 2: the identity L,
# a seeded 2-row L and a rank-deficient 3-row L, row 3 = row 1 + row 2, each with
# rhs = 0). Of the eigenvalues d of L vcov_beta L', contestMD retains the q above
# sqrt(.Machine$double.eps) * d[1]; nu_m has one value per retained direction
# (q = 1: contestMD's contest1D route, nu_m its df and no gap). With q >= 2, an L
# whose minimum relative gap between retained eigenvalues, min |diff(d)| / d[1],
# is below 1e-4 is not frozen: it records only L, that gap and the reason. Then
# contest_1d (one seeded l); the sleepstudy entry carries fstat_ddf_cases, inputs
# and outputs of lmerTest:::get_Fstat_ddf(nu, tol = 1e-8).
#
# A PER-TERM (proposal_dependent) JSON holds r_formula, reml, boundary, then
#   twin                     the summary entry of the same data, formula, reml,
#                            weights and factors, whose JSON holds this fit's
#                            theta-hat and groupings (checked against the fit);
#                            an entry with no twin holds the fit fields above,
#                            theta through h, in their place
#   assign, assign_names, term_labels
#                            attr(model.matrix(m), "assign") over the retained
#                            columns (assign_names), and the term labels its
#                            values 1, 2, ... index (0 is the intercept)
#   type3, type2             anova(m, type = 3) and anova(m, type = 2): `terms`
#                            (the table's rows), sum_sq, mean_sq, NumDF, DenDF,
#                            F, p_value per term; `hypotheses`, one {term, L,
#                            row_names, col_names} per term, L lmerTest's
#                            attr(, "hypotheses") over the retained columns;
#                            DenDF_second (terms x 2), contestMD on each L after
#                            vcov_varpar and Jac_list are recomputed at each
#                            fd_settings setting, as df_second above
#   fd_settings, conditions  as above (anova's own at stage "anova")
#   basis_pair               lmertest_anova_sim_factorial only: contestMD on the
#                            Type III L of `g` and on M %*% L, M a fixed
#                            invertible 2 x 2, each recorded as a contest_md case

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
options(contrasts = c("contr.treatment", "contr.poly"))

GENERATOR <- "goldens_lmertest.R"
EIG_TOL <- 1e-8          # as_lmerModLT's eigenvalue cut for vcov_varpar
FD_DEFAULT <- list(hessian_d = 0.1, jacobian_d = 1e-4, r = 4)   # numDeriv's defaults
FD_SECOND <- list(list(hessian_d = 0.05, jacobian_d = 5e-5, r = 4),
                  list(hessian_d = 0.2, jacobian_d = 2e-4, r = 4))
CONFINT_LEVEL <- 0.95
MIN_REL_GAP <- 1e-4
SEED_L <- 1L
SEED_LVEC <- 2L
NEAR_SINGULAR_MAX <- 0.2
BASIS_PAIR <- list(entry = "lmertest_anova_sim_factorial", term = "g",
                   M = matrix(c(1, 0.3, -0.7, 2), 2))
TWIN_THETA_TOL <- 1e-13  # the twin JSON's theta carries 15 significant digits

# Evaluates expr and records its messages and warnings in order, without
# muffling them.
with_conditions <- function(expr, stage) {
  conds <- list()
  value <- withCallingHandlers(expr,
    message = function(c) {
      conds[[length(conds) + 1L]] <<- list(stage = stage, type = "message",
                                           text = sub("\n$", "", conditionMessage(c)))
    },
    warning = function(c) {
      conds[[length(conds) + 1L]] <<- list(stage = stage, type = "warning",
                                           text = conditionMessage(c))
    })
  list(value = value, conditions = conds)
}

fit_args <- function(e, d) {
  if (!is.logical(e$reml) || length(e$reml) != 1L) {
    stop(sprintf("%s: `reml` must be true or false", e$name), call. = FALSE)
  }
  args <- list(formula = stats::as.formula(e$r_formula), data = d, REML = e$reml)
  if (!is.null(e$weights)) {
    if (!e$weights %in% names(d)) {
      stop(sprintf("%s: weights column %s is not in %s", e$name, e$weights, e$data),
           call. = FALSE)
    }
    args$weights <- d[[e$weights]]
  }
  args
}

# vcov_varpar and Jac_list at one numDeriv setting, formed as as_lmerModLT forms them.
fd_inputs <- function(varpar, devfun, reml, p, s) {
  h <- numDeriv::hessian(func = lmerTest:::devfun_vp, x = varpar,
                         method.args = list(d = s$hessian_d, r = s$r),
                         devfun = devfun, reml = reml)
  eig_h <- eigen(h, symmetric = TRUE)
  pos <- eig_h$values > EIG_TOL
  h_inv <- with(eig_h, {
    vectors[, pos, drop = FALSE] %*% diag(1 / values[pos], nrow = sum(pos)) %*%
      t(vectors[, pos, drop = FALSE])
  })
  jac <- numDeriv::jacobian(func = lmerTest:::get_covbeta, x = varpar,
                            method.args = list(d = s$jacobian_d, r = s$r), devfun = devfun)
  list(h = h, vcov_varpar = 2 * h_inv,
       Jac_list = lapply(seq_len(ncol(jac)), function(i) array(jac[, i], dim = rep(p, 2))))
}

unit <- function(p, i) replace(numeric(p), i, 1)

# contestMD(m, L) with rhs = 0, plus the q per-direction nu_m and the retained
# eigenvalues, recomputed with contestMD's own code.
md_case <- function(m, L, label, row_names, coef_names, seed = NULL) {
  out <- list(label = label)
  if (!is.null(seed)) out$seed <- seed
  out <- c(out, list(L = L, row_names = I(row_names), col_names = I(coef_names), rhs = 0))
  VL <- L %*% m@vcov_beta %*% t(L)
  eig <- eigen(VL)
  d <- eig$values
  tol <- max(sqrt(.Machine$double.eps) * d[1], 0)
  q <- sum(d > tol)
  if (q < 1L) stop(sprintf("contest_md %s: no eigenvalue above tol", label), call. = FALSE)
  if (q >= 2L) {
    gap <- min(abs(diff(d[1:q]))) / d[1]
    out$min_rel_gap <- gap
    if (gap < MIN_REL_GAP) {
      out$frozen <- FALSE
      out$reason <- sprintf("minimum relative eigen-gap %.3g is below %g", gap, MIN_REL_GAP)
      return(out)
    }
  }
  tab <- lmerTest::contestMD(m, L)
  PtL <- crossprod(eig$vectors, L)[1:q, ]
  if (q == 1L) {
    # contestMD's one-direction route: contest1D on PtL, whose df is the DenDF.
    nu_m <- lmerTest::contest1D(m, PtL)$df
    ddf <- nu_m
  } else {
    grad_PLcov <- lapply(1:q, function(i) {
      vapply(m@Jac_list, function(J) lmerTest:::qform(PtL[i, ], J), numeric(1L))
    })
    nu_m <- vapply(1:q, function(i) {
      2 * d[i]^2 / lmerTest:::qform(grad_PLcov[[i]], m@vcov_varpar)
    }, numeric(1L))
    ddf <- lmerTest:::get_Fstat_ddf(nu_m, tol = 1e-8)
  }
  if (!identical(ddf, tab$DenDF)) {
    stop(sprintf("contest_md %s: the recomputed nu_m do not give contestMD's DenDF", label),
         call. = FALSE)
  }
  c(out, list(frozen = TRUE, F = tab[["F value"]], NumDF = tab$NumDF, DenDF = tab$DenDF,
              p_value = tab[["Pr(>F)"]], sum_sq = tab[["Sum Sq"]], mean_sq = tab[["Mean Sq"]],
              eigenvalues = I(d[1:q]), nu_m = I(nu_m)))
}

contest_inputs <- function(m, coef_names) {
  p <- length(m@beta)
  out <- list()
  if (p >= 2L) {
    set.seed(SEED_L)
    L2 <- matrix(rnorm(2L * p), nrow = 2L)
    rows2 <- c("seeded_1", "seeded_2")
    out$contest_md <- list(
      md_case(m, diag(p), "identity", coef_names, coef_names),
      md_case(m, L2, "seeded", rows2, coef_names, seed = SEED_L),
      md_case(m, rbind(L2, L2[1, ] + L2[2, ]), "rank_deficient",
              c(rows2, "seeded_1+seeded_2"), coef_names, seed = SEED_L))
  }
  set.seed(SEED_LVEC)
  l <- rnorm(p)
  t1 <- lmerTest::contest1D(m, l)
  out$contest_1d <- list(seed = SEED_LVEC, l = I(l), estimate = t1$Estimate,
                         std_error = t1[["Std. Error"]], df = t1$df,
                         t_value = t1[["t value"]], p_value = t1[["Pr(>|t|)"]])
  out
}

# lmerTest:::get_Fstat_ddf(nu, tol = 1e-8), one case per branch.
fstat_ddf_cases <- function() {
  cases <- list(one = c(7.25), equal_within_tol = c(6, 6.000000004),
                equal_le_2 = c(1.5, 1.5), any_le_2 = c(1.5, 9.75),
                general = c(4.5, 11.25, 37.125))
  unname(lapply(names(cases), function(k) {
    list(case = k, nu = I(cases[[k]]), tol = 1e-8,
         ddf = lmerTest:::get_Fstat_ddf(cases[[k]], tol = 1e-8))
  }))
}

assert_finite <- function(x, path = "") {
  if (is.list(x)) {
    nm <- names(x)
    for (k in seq_along(x)) {
      assert_finite(x[[k]], paste0(path, "/", if (!is.null(nm) && nzchar(nm[k])) nm[k] else k))
    }
  } else if (is.numeric(x) || is.logical(x) || is.character(x)) {
    bad <- if (is.numeric(x)) !is.finite(x) else is.na(x)
    if (any(bad)) stop(sprintf("non-finite or NA value at %s", path), call. = FALSE)
  }
  invisible(NULL)
}

# The fit of entry e, its devfun and the checks of the summary contract.
fit_entry <- function(e) {
  d <- read_entry_data(e)
  args <- fit_args(e, d)
  fit <- with_conditions(do.call(lmerTest::lmer, args), "fit")
  m <- fit$value
  if (!inherits(m, "lmerModLmerTest")) {
    stop(sprintf("%s: lmerTest::lmer returned a %s", e$name, class(m)[1]), call. = FALSE)
  }
  dev <- with_conditions(do.call(lme4::lmer, c(args, list(
    devFunOnly = TRUE, control = lme4::lmerControl(check.rankX = "silent.drop.cols")))),
    "devfun")
  devfun <- dev$value
  theta <- getME(m, "theta")
  if (!identical(unname(getME(m, "par")), unname(theta))) {
    stop(sprintf("%s: getME(m, \"par\") is not theta", e$name), call. = FALSE)
  }
  if (!identical(m@vcov_beta, as.matrix(vcov(m)))) {
    stop(sprintf("%s: m@vcov_beta is not as.matrix(vcov(m))", e$name), call. = FALSE)
  }
  p <- length(m@beta)
  coef_names <- colnames(m@vcov_beta)
  cf <- summary(m)$coefficients
  if (!identical(rownames(cf), coef_names) || !identical(names(fixef(m)), coef_names)) {
    stop(sprintf("%s: coefficient names disagree", e$name), call. = FALSE)
  }
  varpar <- lmerTest:::getVarPar(m)
  fd0 <- fd_inputs(varpar, devfun, e$reml, p, FD_DEFAULT)
  if (!identical(fd0$vcov_varpar, m@vcov_varpar) || !identical(fd0$Jac_list, m@Jac_list)) {
    stop(sprintf("%s: the rebuilt devfun does not reproduce vcov_varpar and Jac_list",
                 e$name), call. = FALSE)
  }
  list(m = m, devfun = devfun, theta = theta, p = p, coef_names = coef_names, cf = cf,
       varpar = varpar, fd0 = fd0, conditions = c(fit$conditions, dev$conditions))
}

# Copies of the fit with vcov_varpar and Jac_list recomputed at each FD_SECOND setting.
second_fits <- function(e, f) {
  lapply(FD_SECOND, function(s) {
    fd <- fd_inputs(f$varpar, f$devfun, e$reml, f$p, s)
    m2 <- f$m
    m2@vcov_varpar <- fd$vcov_varpar
    m2@Jac_list <- fd$Jac_list
    m2
  })
}

# The fit fields, theta through h, in the key order of the summary JSONs.
fit_fields <- function(e, f) {
  m <- f$m
  p <- f$p
  cf <- f$cf
  ci <- lapply(seq_len(p), function(i) {
    lmerTest::contest1D(m, unit(p, i), confint = TRUE, level = CONFINT_LEVEL)
  })
  cn <- getME(m, "cnms")
  flist <- getME(m, "flist")
  theta_names <- names(f$theta)
  out <- list(
    theta = I(unname(f$theta)), theta_names = I(theta_names),
    cnms = unname(lapply(cn, I)), group_names = I(names(cn)),
    group_levels = I(vapply(names(cn), function(g) nlevels(flist[[g]]), 1L, USE.NAMES = FALSE)),
    sigma = sigma(m), beta = I(unname(m@beta)), coef_names = I(f$coef_names),
    cols_dropped = I(as.character(names(attr(model.matrix(m), "col.dropped")))),
    vcov_beta = unname(m@vcov_beta))
  if (e$reml) out$reml_criterion <- unname(REMLcrit(m)) else out$deviance <- deviance(m)
  out <- c(out, list(
    estimate = I(unname(cf[, "Estimate"])), std_error = I(unname(cf[, "Std. Error"])),
    df = I(unname(cf[, "df"])), t_value = I(unname(cf[, "t value"])),
    p_value = I(unname(cf[, "Pr(>|t|)"])),
    level = CONFINT_LEVEL,
    lower = I(vapply(ci, `[[`, 0, "lower")), upper = I(vapply(ci, `[[`, 0, "upper")),
    varpar_names = I(c(theta_names, "sigma")),
    vcov_varpar = unname(m@vcov_varpar), Jac_list = m@Jac_list))
  if (!identical(e$boundary, "none")) out$h <- f$fd0$h
  out
}

reference_of <- function(e) {
  f <- fit_entry(e)
  m <- f$m
  p <- f$p
  df_second <- vapply(second_fits(e, f), function(m2) {
    vapply(seq_len(p), function(i) lmerTest::contest1D(m2, unit(p, i))$df, numeric(1L))
  }, numeric(p))
  df_second <- matrix(df_second, nrow = p)
  out <- c(list(r_formula = e$r_formula, reml = e$reml, boundary = e$boundary),
           fit_fields(e, f))
  out$fd_settings <- FD_SECOND
  out$df_second <- df_second
  out$conditions <- f$conditions
  if (e$reml) {
    out <- c(out, contest_inputs(m, f$coef_names))
    if (identical(e$data, "data/empirical/sleepstudy.csv")) out$fstat_ddf_cases <- fstat_ddf_cases()
    if (identical(e$boundary, "near_singular")) {
      lw <- getME(m, "lower")
      diag_theta <- f$theta[lw == 0]
      if (!all(diag_theta > 0 & diag_theta < NEAR_SINGULAR_MAX)) {
        stop(sprintf("%s: near-singular entry has diagonal theta %s, not in (0, %g)",
                     e$name, paste(format(diag_theta), collapse = " "), NEAR_SINGULAR_MAX),
             call. = FALSE)
      }
    }
  }
  out$provenance <- provenance(GENERATOR)
  assert_finite(out, e$name)
  out
}

# The summary entry (proposal_dependent: false) fitting the same model as e, or NULL.
twin_of <- function(e) {
  refs <- jsonlite::fromJSON(file.path(suite_dir, "manifest.json"),
                             simplifyDataFrame = FALSE)$inference_references
  same <- Filter(function(t) {
    identical(t$kind, "lmertest") && identical(t$proposal_dependent, FALSE) &&
      identical(t$data, e$data) && identical(t$r_formula, e$r_formula) &&
      identical(t$reml, e$reml) && identical(t$weights, e$weights) &&
      identical(t$factors, e$factors)
  }, refs)
  if (length(same) > 1L) stop(sprintf("%s: more than one twin", e$name), call. = FALSE)
  if (length(same) == 0L) NULL else same[[1L]]$name
}

# anova(m, type = type) with its hypotheses, and each term's DenDF recomputed by
# contestMD on its L at the second FD settings.
anova_table <- function(e, m, ms, type) {
  a <- with_conditions(anova(m, type = type), "anova")
  tab <- a$value
  Ls <- attr(tab, "hypotheses")
  if (!identical(rownames(tab), names(Ls))) {
    stop(sprintf("%s: type %d rows are not its hypotheses", e$name, type), call. = FALSE)
  }
  for (k in seq_along(Ls)) {
    if (!identical(lmerTest::contestMD(m, Ls[[k]])$DenDF, tab$DenDF[k])) {
      stop(sprintf("%s: type %d, %s: contestMD on its L is not the table's DenDF",
                   e$name, type, names(Ls)[k]), call. = FALSE)
    }
  }
  den2 <- vapply(Ls, function(L) {
    vapply(ms, function(m2) lmerTest::contestMD(m2, L)$DenDF, numeric(1L))
  }, numeric(length(ms)))
  list(value = list(
    terms = I(rownames(tab)), sum_sq = I(tab[["Sum Sq"]]), mean_sq = I(tab[["Mean Sq"]]),
    NumDF = I(tab$NumDF), DenDF = I(tab$DenDF), F = I(tab[["F value"]]),
    p_value = I(tab[["Pr(>F)"]]),
    hypotheses = unname(lapply(names(Ls), function(k) {
      list(term = k, L = unname(Ls[[k]]), row_names = I(rownames(Ls[[k]])),
           col_names = I(colnames(Ls[[k]])))
    })),
    DenDF_second = t(matrix(unname(den2), nrow = length(ms)))),
    conditions = a$conditions)
}

# The per-term (proposal_dependent: true) reference of entry e.
anova_reference_of <- function(e) {
  f <- fit_entry(e)
  m <- f$m
  out <- list(r_formula = e$r_formula, reml = e$reml, boundary = e$boundary)
  twin <- twin_of(e)
  if (is.null(twin)) {
    out <- c(out, fit_fields(e, f))
  } else {
    tj <- jsonlite::fromJSON(file.path(suite_dir, "goldens", "inference", "lmertest",
                                       paste0(twin, ".json")))
    if (!isTRUE(all.equal(tj$theta, unname(f$theta), tolerance = TWIN_THETA_TOL))) {
      stop(sprintf("%s: theta differs from its twin %s", e$name, twin), call. = FALSE)
    }
    out$twin <- twin
  }
  X <- model.matrix(m)
  if (!identical(colnames(X), f$coef_names)) {
    stop(sprintf("%s: model.matrix columns are not the coefficients", e$name), call. = FALSE)
  }
  out$assign <- I(attr(X, "assign"))
  out$assign_names <- I(colnames(X))
  out$term_labels <- I(attr(terms(m), "term.labels"))
  ms <- second_fits(e, f)
  t3 <- anova_table(e, m, ms, 3L)
  t2 <- anova_table(e, m, ms, 2L)
  out$type3 <- t3$value
  out$type2 <- t2$value
  out$fd_settings <- FD_SECOND
  out$conditions <- c(f$conditions, t3$conditions, t2$conditions)
  if (identical(e$name, BASIS_PAIR$entry)) {
    k <- match(BASIS_PAIR$term, t3$value$terms)
    h3 <- t3$value$hypotheses[[k]]
    out$basis_pair <- list(
      term = BASIS_PAIR$term, M = BASIS_PAIR$M,
      lmertest = md_case(m, h3$L, "lmertest_type3", h3$row_names, h3$col_names),
      mixed = md_case(m, BASIS_PAIR$M %*% h3$L, "mixed", c("mixed_1", "mixed_2"),
                      h3$col_names))
    if (!isTRUE(out$basis_pair$lmertest$frozen) || !isTRUE(out$basis_pair$mixed$frozen) ||
        !identical(out$basis_pair$lmertest$DenDF, unclass(t3$value$DenDF)[k])) {
      stop("basis pair: a case is not frozen or its DenDF is not the table's", call. = FALSE)
    }
  }
  out$provenance <- provenance(GENERATOR)
  assert_finite(out, e$name)
  out
}

for (e in select_entries("lmertest")) {
  cat(sprintf("%s\n", e$name))
  ref <- if (identical(e$proposal_dependent, FALSE)) {
    reference_of(e)
  } else if (identical(e$proposal_dependent, TRUE)) {
    anova_reference_of(e)
  } else {
    stop(sprintf("%s: `proposal_dependent` must be true or false", e$name), call. = FALSE)
  }
  write_reference(ref,
                  file.path(suite_dir, "goldens", "inference", "lmertest", paste0(e$name, ".json")))
}
