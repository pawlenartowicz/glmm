#!/usr/bin/env Rscript
# Model comparison references -> validation/goldens/inference/anova/<name>.json,
# one per `kind = "anova"` entry of the manifest's `inference_references` block.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's model
# comparisons are validated against. On any glmm disagreement, glmm is presumed
# wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN REFITS AND OVERWRITES EVERY REFERENCE IT OWNS (every `anova` entry).
# Use VALIDATION_ONLY for anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_anova.R
#
# Each entry names its `case`; CASES below maps a case to the function that
# builds its reference, so a case added later is one more function and one more
# CASES element, and leaves the references of the other cases unchanged. The
# fits come from the entry's `members` (on a pair, the smaller model first):
# `fun`, the R function; `formula`; `family` and `link`; `REML` (lmer and
# glmmTMB, else null); `nAGQ` (glmer and glmer.nb, else null); `weights`, a
# column of the entry's data or null; `tolPwrss` on every glmer and glmer.nb
# fit, passed as glmerControl(tolPwrss = ...).
#
# Per fit the JSON carries `logLik`, `npar` (attr(logLik, "df")), `nobs` and
# `deviance` (REMLcrit() on a REML fit, else deviance()), and on a fit that is
# not REML also `AIC` and `BIC`. The entry's `reference` says which p a gate
# reads: `chisq` (p_chisq), `mixture` (p_mixture, the 50:50 mixture of
# chi-square(k) and chi-square(k + 1), k = mixture_k = the npar difference
# minus 1; the entry's `mixture_k` is the expected k, and any other npar
# difference stops the run), `envelope` (`envelope: true`; the p reference is
# the exact RLRT of the `rlrt` entry named by the manifest's `rlrt_entry`), `f`
# (anova.lm's F) or `wald_f` (the Wald F of another entry; the LRT here is
# recorded only). `rlrt_entry` is not limited to envelope entries: every ranova
# entry whose row drops the last random effect carries it, and on the one mixture
# entry among them (anova_ranova_lmm_intercept, weighted) it names the `rlrt`
# entry whose `loglik_lm_reml` is the row's reduced logLik, while the gated p
# stays p_mixture.
# Names are written as separate string arrays (jsonlite drops dimnames); no
# value is Inf, NaN or NA, and a cell R reports as NA (the first row of anova.lm
# and anova.glm, ranova's `<none>` row) is left out.
#
# The per-term cases (car_glmm, car_glm, car_lm, linear_hypothesis, nonmarginal)
# have no `reference`; their `members` hold one fit, except nonmarginal's (the
# lmer fit, then its lm twin). The car cases refit the member with contr.sum on
# the entry's `contr_sum` columns and write, per fit (`treatment`, `contr_sum`),
# `coef_names`, `coef` and `vcov`, then three tables of car::Anova: `type2` and
# `type3` on the treatment fit and `type3_sum` on the contr.sum refit, each with
# `fit`, `gated` (false on the treatment Type III, which is recorded only),
# `test` ("Chisq", Wald, or "F" on the lm) and `rows` in car's order. A row has
# `term`, `statistic`, `df` (and `res_df` on the lm), `p` and its hypothesis
# matrix `L` (rows over every coefficient, with `row_names` and `col_names`).
# A Type III table's `(Intercept)` row is recorded only: the gated rows are the
# entry's `term_labels`. The script rebuilds each L as car does (Type II:
# car:::relatives and car:::ConjComp in car's vcov metric; Type III: identity
# rows on the term's columns) and stops unless car::linearHypothesis(m, L)
# gives the row's statistic to 1e-10 relative. linear_hypothesis writes the fit's
# coefficients, the L of the entry's `tested` columns, and
# car::linearHypothesis(test = "F")'s F, num_df, den_df and p. nonmarginal
# records four calls on the fit and its twin (lmerTest's Type III and Type II,
# car's Type II Wald on the lmer fit, car's Type II on the lm): each has the
# table `rows` (R's column names as keys, NA cells left out) or the `error`
# message, and `conditions`, the warning and message texts the call raised.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
local({
  grid_dir <- file.path(suite_dir, "grid")
  source(file.path(grid_dir, "engines", "versions.R"), local = TRUE)
  assert_pkg_version("glmmTMB", grid_versions(grid_dir)[["glmmTMB"]])
})

GENERATOR <- "goldens_anova.R"
out_dir <- file.path(suite_dir, "goldens", "inference", "anova")

# Stops unless every number in `x` (a nested list) is finite, so no Inf, NaN
# or NA reaches a JSON.
check_finite <- function(x, name, path = "") {
  if (is.list(x)) {
    keys <- names(x)
    for (i in seq_along(x)) {
      key <- if (is.null(keys) || !nzchar(keys[i])) as.character(i) else keys[i]
      check_finite(x[[i]], name, paste0(path, "/", key))
    }
  } else if (anyNA(x) || (is.numeric(x) && !all(is.finite(x)))) {
    stop(sprintf("%s: %s is not finite", name, path), call. = FALSE)
  }
  invisible(x)
}

# The family call of a member, in the grid's family spelling.
family_call <- function(family, link) {
  fun <- switch(family, gaussian = "gaussian", binomial = "binomial", poisson = "poisson",
                gamma = "Gamma", stop("family_call: unsupported family ", family))
  str2lang(sprintf('%s(link = "%s")', fun, link))
}

# The call that fits member `mem` on `data`, and the environment to evaluate it
# in (holding `data`, so update() inside ranova finds it). Built as a call, not
# by do.call, so the model's stored call names the weights column.
member_call <- function(mem, data) {
  env <- new.env(parent = globalenv())
  env$data <- data
  f <- as.formula(mem$formula, env = env)
  w <- if (is.null(mem$weights)) NULL else as.name(mem$weights)
  ctrl <- if (is.null(mem$tolPwrss)) NULL else
    bquote(lme4::glmerControl(tolPwrss = .(mem$tolPwrss)))
  cl <- switch(mem$fun,
    "lme4::lmer" = bquote(lme4::lmer(.(f), data = data, REML = .(mem$REML), weights = .(w))),
    "stats::lm" = bquote(stats::lm(.(f), data = data, weights = .(w))),
    "stats::glm" = bquote(stats::glm(.(f), family = .(family_call(mem$family, mem$link)),
                                     data = data, weights = .(w))),
    "lme4::glmer" = bquote(lme4::glmer(.(f), family = .(family_call(mem$family, mem$link)),
                                       data = data, weights = .(w), nAGQ = .(mem$nAGQ),
                                       control = .(ctrl))),
    "MASS::glm.nb" = {
      if (!identical(mem$link, "log")) stop("member_call: glm.nb is fitted with its log link")
      bquote(MASS::glm.nb(.(f), data = data, weights = .(w)))
    },
    "lme4::glmer.nb" = {
      if (!identical(mem$link, "log")) stop("member_call: glmer.nb is fitted with its log link")
      bquote(lme4::glmer.nb(.(f), data = data, weights = .(w), nAGQ = .(mem$nAGQ),
                            control = .(ctrl)))
    },
    "glmmTMB::glmmTMB" = bquote(glmmTMB::glmmTMB(.(f),
                                                 family = .(family_call(mem$family, mem$link)),
                                                 data = data, weights = .(w),
                                                 REML = .(mem$REML))),
    stop("member_call: unsupported fun ", mem$fun))
  list(call = cl, env = env)
}

fit_member <- function(mem, data) {
  mc <- member_call(mem, data)
  eval(mc$call, mc$env)
}

member_summary <- function(m, mem) {
  ll <- logLik(m)
  reml <- isTRUE(mem$REML)
  out <- list(
    logLik = as.numeric(ll),
    npar = as.integer(attr(ll, "df")),
    nobs = as.integer(nobs(m)),
    deviance = if (reml) unname(lme4::REMLcrit(m)) else unname(deviance(m))
  )
  if (!reml) {
    out$AIC <- AIC(m)
    out$BIC <- BIC(m)
  }
  out
}

# The `reference` fields of one comparison with statistic `stat` and npar
# difference `dnpar`.
reference_fields <- function(entry, stat, dnpar) {
  switch(entry$reference,
    mixture = {
      k <- dnpar - 1L
      if (!identical(as.integer(k), as.integer(entry$mixture_k))) {
        stop(sprintf("%s: the npar difference %d gives mixture k %d, the entry expects %d",
                     entry$name, dnpar, k, as.integer(entry$mixture_k)), call. = FALSE)
      }
      list(mixture_k = as.integer(k),
           p_mixture = 0.5 * pchisq(stat, k, lower.tail = FALSE) +
             0.5 * pchisq(stat, k + 1L, lower.tail = FALSE))
    },
    envelope = list(envelope = TRUE),
    chisq = , f = , wald_f = list(),
    stop(sprintf("%s: unknown reference %s", entry$name, entry$reference), call. = FALSE))
}

# Two fits through lme4's anova() (sorted by npar, so the second row is the
# larger model): `refit` only on two REML lmer fits. lme4 refuses a glmer fit
# with nAGQ > 1 against a glm fit (its aggregated-binomial logLik shifts there);
# that pair is taken only on Bernoulli data, where the shift is zero, and gets
# anova.merMod's own arithmetic on the two logLik: 2 * pmax(0, diff), the npar
# difference and pchisq.
lme4_pair <- function(entry, refit = TRUE) {
  data <- read_entry_data(entry)
  mems <- entry$members
  m0 <- fit_member(mems[[1]], data)
  m1 <- fit_member(mems[[2]], data)
  s <- list(member_summary(m0, mems[[1]]), member_summary(m1, mems[[2]]))
  dnpar <- s[[2]]$npar - s[[1]]$npar
  if (!methods::is(m0, "merMod") && isTRUE(mems[[2]]$nAGQ > 1)) {
    if (!all(model.response(model.frame(m0)) %in% c(0, 1))) {
      stop(sprintf("%s: nAGQ > 1 against a glm needs a Bernoulli response", entry$name),
           call. = FALSE)
    }
    stat <- 2 * max(0, s[[2]]$logLik - s[[1]]$logLik)
    return(c(list(case = entry$case, reference = entry$reference, members = s,
                  statistic = stat, df = as.integer(dnpar),
                  p_chisq = pchisq(stat, dnpar, lower.tail = FALSE)),
             reference_fields(entry, stat, dnpar)))
  }
  a <- if (methods::is(m1, "merMod")) anova(m1, m0, refit = refit) else anova(m0, m1)
  if (!identical(rownames(a), c("m0", "m1"))) {
    stop(sprintf("%s: anova rows are %s, not m0, m1", entry$name,
                 paste(rownames(a), collapse = ", ")), call. = FALSE)
  }
  if (a[2, "Df"] != dnpar) {
    stop(sprintf("%s: anova Df %s differs from the npar difference %d", entry$name,
                 a[2, "Df"], dnpar), call. = FALSE)
  }
  stat <- a[2, "Chisq"]
  c(list(case = entry$case, reference = entry$reference, members = s,
         statistic = stat, df = as.integer(a[2, "Df"]), p_chisq = a[2, "Pr(>Chisq)"]),
    reference_fields(entry, stat, dnpar))
}

# The larger fit's coefficients, covariance and the tested column names.
larger_coefs <- function(m0, m1) {
  b <- coef(m1)
  list(coef_names = I(names(b)), coef = unname(b), vcov = unname(vcov(m1)),
       tested = I(setdiff(names(b), names(coef(m0)))))
}

# The likelihood ratio of two fits from their logLik.
loglik_lrt <- function(s) {
  stat <- 2 * (s[[2]]$logLik - s[[1]]$logLik)
  dnpar <- s[[2]]$npar - s[[1]]$npar
  list(statistic = stat, df = as.integer(dnpar),
       p_chisq = pchisq(stat, dnpar, lower.tail = FALSE))
}

case_random_pair <- function(entry) lme4_pair(entry, refit = FALSE)
case_ml_pair <- function(entry) lme4_pair(entry)
case_glmm_pair <- function(entry) lme4_pair(entry)
case_glmm_vs_glm <- function(entry) lme4_pair(entry)

case_ols_pair <- function(entry) {
  data <- read_entry_data(entry)
  mems <- entry$members
  m0 <- fit_member(mems[[1]], data)
  m1 <- fit_member(mems[[2]], data)
  s <- list(member_summary(m0, mems[[1]]), member_summary(m1, mems[[2]]))
  a <- anova(m0, m1)
  c(list(case = entry$case, reference = entry$reference, members = s),
    loglik_lrt(s),
    list(anova_lm = list(res_df = as.integer(a[["Res.Df"]]), rss = a[["RSS"]],
                         df = as.integer(a[2, "Df"]), sum_sq = a[2, "Sum of Sq"],
                         F = a[2, "F"], p = a[2, "Pr(>F)"])),
    larger_coefs(m0, m1),
    reference_fields(entry, NA_real_, NA_integer_))
}

case_glm_pair <- function(entry) {
  data <- read_entry_data(entry)
  mems <- entry$members
  m0 <- fit_member(mems[[1]], data)
  m1 <- fit_member(mems[[2]], data)
  s <- list(member_summary(m0, mems[[1]]), member_summary(m1, mems[[2]]))
  a <- anova(m0, m1, test = "Chisq")
  c(list(case = entry$case, reference = entry$reference, members = s),
    loglik_lrt(s),
    list(anova_glm = list(resid_df = as.integer(a[["Resid. Df"]]),
                          resid_dev = a[["Resid. Dev"]], df = as.integer(a[2, "Df"]),
                          deviance = a[2, "Deviance"], p = a[2, "Pr(>Chi)"])),
    larger_coefs(m0, m1),
    reference_fields(entry, NA_real_, NA_integer_))
}

# One lmerTest::ranova table on the REML fit of the entry's one member.
case_ranova <- function(entry) {
  data <- read_entry_data(entry)
  mem <- entry$members[[1]]
  m <- fit_member(mem, data)
  r <- lmerTest::ranova(m, reduce.terms = TRUE)
  if (!identical(rownames(r)[1], "<none>")) {
    stop(sprintf("%s: ranova's first row is %s", entry$name, rownames(r)[1]), call. = FALSE)
  }
  forms <- attr(r, "formulae")
  full_npar <- as.integer(r[1, "npar"])
  rows <- lapply(rownames(r)[-1], function(label) {
    f <- forms[[label]]
    if (is.null(f)) stop(sprintf("%s: ranova gives no formula for row %s", entry$name, label))
    npar <- as.integer(r[label, "npar"])
    if (r[label, "Df"] != full_npar - npar) {
      stop(sprintf("%s: row %s Df differs from the npar difference", entry$name, label),
           call. = FALSE)
    }
    c(list(label = label, reduced_formula = paste(deparse(f, width.cutoff = 500L), collapse = ""),
           logLik = r[label, "logLik"], npar = npar, LRT = r[label, "LRT"],
           Df = as.integer(r[label, "Df"]), p_chisq = r[label, "Pr(>Chisq)"]),
      reference_fields(entry, r[label, "LRT"], full_npar - npar))
  })
  list(case = entry$case, reference = entry$reference,
       full = list(formula = mem$formula, logLik = r[1, "logLik"], npar = full_npar),
       rows = rows)
}

# The Gamma GLMM against its GLM through glmmTMB (one likelihood
# for both; members 1 and 2) and through lme4 (glm and glmer; members 3 and 4),
# with lme4's -devfun/2 at theta = 0 and the GLM's beta-hat. The statistic and
# the mixture p are glmmTMB's.
case_gamma_glmm_vs_glm <- function(entry) {
  data <- read_entry_data(entry)
  mems <- entry$members
  funs <- vapply(mems, `[[`, "", "fun")
  if (!identical(funs, c("glmmTMB::glmmTMB", "glmmTMB::glmmTMB", "stats::glm", "lme4::glmer"))) {
    stop(sprintf("%s: members are %s", entry$name, paste(funs, collapse = ", ")), call. = FALSE)
  }
  fits <- lapply(mems, fit_member, data = data)
  s <- Map(member_summary, fits, mems)
  dc <- member_call(mems[[4]], data)
  dc$call$devFunOnly <- TRUE
  devfun <- eval(dc$call, dc$env)
  n_theta <- length(lme4::getME(fits[[4]], "theta"))
  neg_devfun_half <- -devfun(c(rep(0, n_theta), coef(fits[[3]]))) / 2
  stat <- max(0, 2 * (s[[2]]$logLik - s[[1]]$logLik))
  dnpar <- s[[2]]$npar - s[[1]]$npar
  c(list(case = entry$case, reference = entry$reference, members = s,
         neg_devfun_half = neg_devfun_half,
         offset = s[[3]]$logLik - neg_devfun_half,
         statistic = stat, df = as.integer(dnpar),
         p_chisq = pchisq(stat, dnpar, lower.tail = FALSE)),
    reference_fields(entry, stat, dnpar))
}

# Fit-free: chi-square upper tails and the 50:50 mixture on the entry's grid.
case_mixture_points <- function(entry) {
  stat <- as.numeric(unlist(entry$t))
  k <- as.integer(unlist(entry$k))
  q <- function(kk) pchisq(stat, kk, lower.tail = FALSE)
  list(case = entry$case, t = stat, k = k,
       q_chisq = t(vapply(k, q, stat)),
       p_mixture = t(vapply(k, function(kk) 0.5 * q(kk) + 0.5 * q(kk + 1L), stat)))
}

# Per-term references (the cases car_glmm, car_glm, car_lm, linear_hypothesis,
# nonmarginal).

# Member `mem` refitted with contr.sum on the factor columns `cols`: the call of
# fit_member plus `contrasts`.
fit_member_sum <- function(mem, data, cols) {
  mc <- member_call(mem, data)
  mc$call$contrasts <- setNames(rep(list("contr.sum"), length(cols)), cols)
  eval(mc$call, mc$env)
}

# The fixed coefficients of a fit (no aliased column allowed), their covariance
# and names.
fit_coefs <- function(m, name) {
  b <- if (methods::is(m, "merMod")) lme4::fixef(m) else coef(m)
  if (anyNA(b)) stop(sprintf("%s: the fit has aliased coefficients", name), call. = FALSE)
  list(coef_names = I(names(b)), coef = unname(b), vcov = unname(as.matrix(vcov(m))))
}

# The covariance car::Anova's Type II hands to ConjComp: Anova.merMod's default
# on a glmer fit, Anova.glm's on a glm (Wald), Anova_II_lm's on an lm.
car_metric <- function(m) {
  if (methods::is(m, "merMod")) {
    car:::getVcov(vcov(m, complete = FALSE), m)
  } else if (inherits(m, "glm")) {
    car:::getVcov(vcov(m, complete = TRUE), m)
  } else {
    vcov(m, complete = FALSE)
  }
}

# Each row's hypothesis matrix as car builds it on a fit with no aliased column:
# Type II through car:::relatives and car:::ConjComp in car's metric
# (Anova_II_mer, Anova_II_default, Anova_II_lm), Type III as identity rows on the
# term's columns (Anova_III_mer, Anova_III_default, Anova_III_lm), its first row
# the intercept. Row names are the term's coefficient names.
car_hypotheses <- function(m, type) {
  b <- if (methods::is(m, "merMod")) lme4::fixef(m) else coef(m)
  cn <- names(b)
  if (!identical(cn[1], "(Intercept)")) stop("car_hypotheses: the fit has no intercept")
  I.p <- diag(length(b))
  assign <- attr(model.matrix(m), "assign")
  labels <- attr(terms(m), "term.labels")
  hyp <- function(term, subs, L) {
    if (nrow(L) != length(subs)) stop(sprintf("car_hypotheses: term %s has %d rows", term, nrow(L)))
    list(term = term, L = L, row_names = I(cn[subs]), col_names = I(cn))
  }
  if (type == 3) {
    terms3 <- c("(Intercept)", labels)
    return(lapply(seq_along(terms3), function(i) {
      subs <- which(assign == i - 1L)
      hyp(terms3[i], subs, I.p[subs, , drop = FALSE])
    }))
  }
  fac <- attr(terms(m), "factors")
  metric <- car_metric(m)
  lapply(seq_along(labels), function(i) {
    subs_term <- which(assign == i)
    subs_rel <- NULL
    for (r in car:::relatives(labels[i], labels, fac)) subs_rel <- c(subs_rel, which(assign == r))
    h1 <- I.p[subs_rel, , drop = FALSE]
    h2 <- I.p[c(subs_rel, subs_term), , drop = FALSE]
    L <- if (nrow(h1) == 0) h2 else t(car:::ConjComp(t(h1), t(h2), metric))
    hyp(labels[i], subs_term, L[!apply(L, 1, function(x) all(x == 0)), , drop = FALSE])
  })
}

# One car::Anova table `a` of fit `m` with its rebuilt hypotheses: the rows in
# car's order, each checked against car::linearHypothesis(m, L).
car_table <- function(name, m, a, type, fit, gated) {
  lm_fit <- !inherits(m, "glm") && !methods::is(m, "merMod")
  test <- if (lm_fit) "F" else "Chisq"
  stat_col <- if (lm_fit) "F value" else "Chisq"
  p_col <- if (lm_fit) "Pr(>F)" else "Pr(>Chisq)"
  hyps <- car_hypotheses(m, type)
  terms_a <- setdiff(rownames(a), "Residuals")
  if (!identical(terms_a, vapply(hyps, `[[`, "", "term"))) {
    stop(sprintf("%s: Type %d rows are %s", name, type, paste(terms_a, collapse = ", ")),
         call. = FALSE)
  }
  rows <- lapply(hyps, function(h) {
    stat <- a[h$term, stat_col]
    lh <- car::linearHypothesis(m, h$L)
    if (abs(lh[[test]][2] - stat) > 1e-10 * abs(stat)) {
      stop(sprintf("%s: Type %d %s: linearHypothesis gives %.17g, the table %.17g", name, type,
                   h$term, lh[[test]][2], stat), call. = FALSE)
    }
    if (a[h$term, "Df"] != nrow(h$L)) {
      stop(sprintf("%s: Type %d %s: Df %s, L has %d rows", name, type, h$term,
                   a[h$term, "Df"], nrow(h$L)), call. = FALSE)
    }
    c(list(term = h$term, statistic = stat, df = as.integer(a[h$term, "Df"])),
      if (lm_fit) list(res_df = as.integer(a["Residuals", "Df"])),
      list(p = a[h$term, p_col], L = h$L, row_names = h$row_names, col_names = h$col_names))
  })
  list(fit = fit, gated = gated, test = test, rows = rows)
}

# The car cases: the treatment fit's Type II and Type III tables and the
# contr.sum refit's Type III table, through `anova_of(m, type)`.
car_case <- function(entry, anova_of) {
  data <- read_entry_data(entry)
  mem <- entry$members[[1]]
  cols <- unlist(entry$contr_sum)
  m <- fit_member(mem, data)
  ms <- fit_member_sum(mem, data, cols)
  list(case = entry$case,
       term_labels = I(attr(terms(m), "term.labels")),
       treatment = fit_coefs(m, entry$name),
       contr_sum = c(list(contrasts = I(cols)), fit_coefs(ms, entry$name)),
       type2 = car_table(entry$name, m, anova_of(m, 2), 2, "treatment", TRUE),
       type3 = car_table(entry$name, m, anova_of(m, 3), 3, "treatment", FALSE),
       type3_sum = car_table(entry$name, ms, anova_of(ms, 3), 3, "contr_sum", TRUE))
}

# glmer: Anova.merMod's default test.statistic, "Chisq" (Wald).
case_car_glmm <- function(entry) car_case(entry, function(m, type) car::Anova(m, type = type))
# glm: Anova.glm defaults to the LR test, so the Wald test is named.
case_car_glm <- function(entry) {
  car_case(entry, function(m, type) car::Anova(m, type = type, test.statistic = "Wald"))
}
# lm: F tests.
case_car_lm <- function(entry) car_case(entry, function(m, type) car::Anova(m, type = type))

# car::linearHypothesis(test = "F") on the entry's `tested` columns, checked
# against the Wald F from the fit's coefficients and covariance.
case_linear_hypothesis <- function(entry) {
  data <- read_entry_data(entry)
  m <- fit_member(entry$members[[1]], data)
  fc <- fit_coefs(m, entry$name)
  cn <- as.character(fc$coef_names)
  tested <- unlist(entry$tested)
  if (!all(tested %in% cn)) stop(sprintf("%s: tested columns not in the fit", entry$name))
  L <- diag(length(cn))[match(tested, cn), , drop = FALSE]
  lh <- car::linearHypothesis(m, L, test = "F")
  d <- L %*% fc$coef
  f <- drop(t(d) %*% solve(L %*% fc$vcov %*% t(L), d)) / nrow(L)
  if (abs(f - lh$F[2]) > 1e-10 * abs(f)) {
    stop(sprintf("%s: linearHypothesis F %.17g, Wald F %.17g", entry$name, lh$F[2], f),
         call. = FALSE)
  }
  c(list(case = entry$case), fc,
    list(L = L, row_names = I(tested), col_names = I(cn), F = lh$F[2],
         num_df = as.integer(lh$Df[2]), den_df = as.integer(lh$Res.Df[2]),
         p = lh[2, "Pr(>F)"]))
}

# One call's record: its table rows (R's column names as keys, NA cells left
# out) or its error message, and the warning and message texts it raised.
record_call <- function(call_fun) {
  conds <- character()
  keep <- function(c) conds <<- c(conds, conditionMessage(c))
  res <- tryCatch(withCallingHandlers(call_fun(),
                                      warning = function(w) {
                                        keep(w)
                                        invokeRestart("muffleWarning")
                                      },
                                      message = function(mm) {
                                        keep(mm)
                                        invokeRestart("muffleMessage")
                                      }),
                  error = function(e) e)
  if (inherits(res, "error")) return(list(error = conditionMessage(res), conditions = I(conds)))
  rows <- lapply(rownames(res), function(rn) {
    vals <- lapply(names(res), function(col) res[rn, col])
    names(vals) <- names(res)
    c(list(term = rn), Filter(function(v) !is.na(v), vals))
  })
  list(rows = rows, conditions = I(conds))
}

# A non-marginal formula: lmerTest's Type III and Type II and car's Type II on the
# lmer fit (member 1), car's Type II on its lm twin (member 2).
case_nonmarginal <- function(entry) {
  data <- read_entry_data(entry)
  mems <- entry$members
  m <- fit_member(mems[[1]], data)
  mt <- lmerTest::as_lmerModLmerTest(m)
  twin <- fit_member(mems[[2]], data)
  list(case = entry$case,
       term_labels = I(attr(terms(m), "term.labels")),
       coef_names = I(names(lme4::fixef(m))),
       lmertest_type3 = record_call(function() anova(mt, type = 3)),
       lmertest_type2 = record_call(function() anova(mt, type = 2)),
       car_type2_lmer = record_call(function() car::Anova(m, type = 2)),
       car_type2_lm = record_call(function() car::Anova(twin, type = 2)))
}

CASES <- list(
  random_pair = case_random_pair,
  ml_pair = case_ml_pair,
  ols_pair = case_ols_pair,
  glm_pair = case_glm_pair,
  glmm_pair = case_glmm_pair,
  glmm_vs_glm = case_glmm_vs_glm,
  gamma_glmm_vs_glm = case_gamma_glmm_vs_glm,
  ranova = case_ranova,
  mixture_points = case_mixture_points,
  car_glmm = case_car_glmm,
  car_glm = case_car_glm,
  car_lm = case_car_lm,
  linear_hypothesis = case_linear_hypothesis,
  nonmarginal = case_nonmarginal
)

for (entry in select_entries("anova")) {
  cat(sprintf("%s ...\n", entry$name))
  build <- CASES[[entry$case]]
  if (is.null(build)) stop(sprintf("%s: unknown case %s", entry$name, entry$case), call. = FALSE)
  obj <- c(list(provenance = provenance(GENERATOR, extra = "glmmTMB")),
           check_finite(build(entry), entry$name))
  write_reference(obj, file.path(out_dir, paste0(entry$name, ".json")))
}
