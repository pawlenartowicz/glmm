#!/usr/bin/env Rscript
# Profile-likelihood confidence interval references ->
# validation/goldens/inference/profile/<name>.json, one per `kind = "profile"`
# entry of the manifest's `inference_references` block.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's
# profile-likelihood intervals are validated against. On any glmm disagreement,
# glmm is presumed wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN REFITS AND OVERWRITES EVERY PROFILE REFERENCE. Use VALIDATION_ONLY
# for anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_profile.R
#
# Each entry has one member, an lme4::lmer fit `m` made with fit_member. Every
# interval is read off the ML deviance: ml = refitML(m) on a REML member, m
# itself on an ML member. The intervals are lme4's confint(m, method =
# "profile", level = <entry level>, signames = FALSE), except where the entry's
# optional `beta_method` is "offset_refit" (absent means "lme4_profile"):
#   - "lme4_profile": every row from that confint call;
#   - "offset_refit": the variance rows (SDs, correlations and sigma) from
#     confint with parm = "theta_", and each coefficient's row computed here by
#     its own profile: for coefficient j held at b, lme4::lmer(REML = FALSE)
#     refits the model with column j of the fixed design dropped and b * x_j
#     as the offset (same prior weights, same random part), and
#     zeta(b) = sign(b - beta_j) * sqrt(deviance(refit) - deviance(ml)).
#     Each bound is the uniroot of zeta(b) = -/+ qnorm((1 + level) / 2),
#     bracketed outward from beta_j by doublings of the Wald interval's
#     half-width (ml's vcov). The refit with coefficient j held at beta_j must
#     reproduce deviance(ml) within 1e-6 (the anchor), zeta must rise strictly
#     along a grid from the estimate to the bracket's end, a refit below
#     deviance(ml) by more than 1e-6 stops the run, and so does any message or
#     warning of a refit.
#     The route exists because lme4 2.0.6's fixed-effect profile on a
#     prior-weighted LMM warns (non-monotonic profile, bad spline fit) and
#     comes out about 20 times narrower than the Wald interval; its variance
#     rows on the same fit do not warn. Such a JSON's provenance carries a
#     `note` that says so.
# `signames = FALSE` gives the row names `sd_<term>|<group>`,
# `cor_<row term>.<col term>|<group>`, `sigma` and the coefficient names (the
# names lme4 documents under the deprecated `oldNames = FALSE`). The JSON holds:
#   - `level`, the entry's;
#   - `deviance`, deviance(ml): -2 x the ML log-likelihood the profiles start
#     from;
#   - `rows`, the variance rows in confint's order, then the coefficients in
#     fixef order (confint's own order), each with `parameter` (the row name),
#     `kind` ("sd", "cor", "sigma" or "beta"), `group` (the grouping factor;
#     null on sigma and beta), `term` (the SD's column; the correlation's column
#     term, as.data.frame(VarCorr())'s var1; the coefficient name; null on
#     sigma), `term2` (the correlation's row term, var2; null on every other
#     kind), `estimate` (ml's: the SD or correlation of
#     as.data.frame(VarCorr(ml)), sigma(ml), fixef(ml)), `lower`, `upper` and
#     `method` ("lme4_profile" or "offset_refit", the route of that row).
# The names built from VarCorr(ml) and fixef(ml) must equal the interval row
# names as a set, else the run stops.
#
# Any R warning stops the run, so no value R flagged is frozen. confint's one
# message, "Computing profile confidence intervals ...", is muffled; any other
# message from confint stops the run. The member fit and its ML refit of an
# entry may print only that entry's messages in FIT_MESSAGES; any other message
# of theirs stops the run.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()

GENERATOR <- "goldens_profile.R"
out_dir <- file.path(suite_dir, "goldens", "inference", "profile")
CONFINT_MESSAGE <- "Computing profile confidence intervals ...\n"
BETA_METHODS <- c("lme4_profile", "offset_refit")
OFFSET_NOTE <- paste(
  "beta rows by ML offset refits, not lme4's profile: lme4 2.0.6's fixed-effect profile on a",
  "prior-weighted LMM warns (non-monotonic profile) and comes out about 20x narrower than Wald")
# Two optimizations of the same ML deviance agree to this much. The held refit
# at the estimate must reproduce deviance(ml) within it (the anchor: measured
# 4.2e-9 and 8.8e-9 on profile_lmm_slope_weighted's two coefficients, while a
# held model built wrong, e.g. without the prior weights, moves the deviance by
# whole units), and a held refit may sit below deviance(ml) by this much before
# the run stops; smaller drops read as zeta = 0. At a bound, where zeta is about
# 2, a deviance error of 1e-6 moves zeta by 2.6e-7.
DEVIANCE_TOL <- 1e-6
# Doublings of the Wald half-width tried per side before a bound is given up.
MAX_DOUBLINGS <- 10L
# Points of the monotonicity grid between the estimate and the bracket's end.
GRID_POINTS <- 8L
# The messages a member fit (fit_member) or its ML refit (refitML) may print,
# by entry name; an entry not named here may print none.
FIT_MESSAGES <- list(
  # The REML fit of Dyestuff2 puts the Batch SD at 0, the singular fit the
  # entry exists for.
  profile_dyestuff2 = "boundary (singular) fit: see help('isSingular')\n")

# `expr` (a member fit or its ML refit of entry `name`) with its messages
# checked: one of the entry's FIT_MESSAGES prints, any other stops the run.
checked_fit <- function(name, expr) {
  withCallingHandlers(expr, message = function(msg) {
    if (!conditionMessage(msg) %in% FIT_MESSAGES[[name]]) {
      stop(sprintf("%s: fit message: %s", name, conditionMessage(msg)), call. = FALSE)
    }
  })
}

# confint(method = "profile") of `m` over `parm` (all parameters when NULL),
# with its progress message muffled and any other message turned into an error.
profile_confint <- function(name, m, level, parm = NULL) {
  withCallingHandlers(
    if (is.null(parm)) {
      confint(m, method = "profile", level = level, signames = FALSE)
    } else {
      confint(m, method = "profile", parm = parm, level = level, signames = FALSE)
    },
    message = function(msg) {
      if (identical(conditionMessage(msg), CONFINT_MESSAGE)) {
        invokeRestart("muffleMessage")
      }
      stop(sprintf("%s: confint message: %s", name, conditionMessage(msg)), call. = FALSE)
    })
}

# One row per parameter of the ML fit `ml`, named as confint names it: a list of
# `parameter`, `kind`, `group`, `term`, `term2` and `estimate`. NA marks a null.
parameter_rows <- function(ml) {
  vc <- as.data.frame(VarCorr(ml))
  rows <- lapply(seq_len(nrow(vc)), function(i) {
    r <- vc[i, ]
    if (identical(r$grp, "Residual")) {
      list(parameter = "sigma", kind = "sigma", group = NA_character_, term = NA_character_,
           term2 = NA_character_, estimate = sigma(ml))
    } else if (is.na(r$var2)) {
      list(parameter = sprintf("sd_%s|%s", r$var1, r$grp), kind = "sd", group = r$grp,
           term = r$var1, term2 = NA_character_, estimate = r$sdcor)
    } else {
      list(parameter = sprintf("cor_%s.%s|%s", r$var2, r$var1, r$grp), kind = "cor",
           group = r$grp, term = r$var1, term2 = r$var2, estimate = r$sdcor)
    }
  })
  beta <- fixef(ml)
  c(rows, lapply(names(beta), function(b) {
    list(parameter = b, kind = "beta", group = NA_character_, term = b,
         term2 = NA_character_, estimate = unname(beta[[b]]))
  }))
}

# The function of b giving the ML deviance of member `mem` on `data` with
# coefficient `j` of the ML fit `ml` held at b: member_call's lmer call with
# column j of ml's fixed design dropped, b * x_j as the offset, REML = FALSE,
# the member's prior weights and ml's random part. Any message or warning of a
# refit stops the run.
held_deviance_fn <- function(name, mem, data, ml, j) {
  X <- getME(ml, "X")
  if (nrow(X) != nrow(data)) {
    stop(sprintf("%s: the fixed design has %d rows, the data %d", name, nrow(X), nrow(data)),
         call. = FALSE)
  }
  for (col in c("held_X", "held_offset")) {
    if (col %in% names(data)) stop(sprintf("%s: data has a column %s", name, col), call. = FALSE)
  }
  data$held_X <- X[, -j, drop = FALSE]
  mc <- member_call(mem, data)
  random <- formula(ml, random.only = TRUE)
  f <- if (ncol(X) > 1L) update(random, . ~ 0 + held_X + .) else update(random, . ~ 0 + .)
  environment(f) <- mc$env
  mc$call[[2]] <- f
  mc$call$REML <- FALSE
  mc$call$offset <- quote(held_offset)
  function(b) {
    mc$env$data$held_offset <- b * X[, j]
    fit <- withCallingHandlers(eval(mc$call, mc$env),
      message = function(msg) {
        stop(sprintf("%s: held refit (%s = %.17g) message: %s", name, colnames(X)[j], b,
                     conditionMessage(msg)), call. = FALSE)
      },
      warning = function(wn) {
        stop(sprintf("%s: held refit (%s = %.17g) warning: %s", name, colnames(X)[j], b,
                     conditionMessage(wn)), call. = FALSE)
      })
    deviance(fit)
  }
}

# The interval of coefficient `j` of the ML fit `ml` by held refits (see the
# header): c(lower, upper).
offset_interval <- function(name, mem, data, ml, j, level) {
  coef_name <- names(fixef(ml))[j]
  b_hat <- unname(fixef(ml)[j])
  se <- sqrt(diag(as.matrix(vcov(ml)))[[j]])
  z <- qnorm((1 + level) / 2)
  dev0 <- deviance(ml)
  held_deviance <- held_deviance_fn(name, mem, data, ml, j)
  anchor <- held_deviance(b_hat) - dev0
  if (abs(anchor) > DEVIANCE_TOL) {
    stop(sprintf("%s: the refit with %s held at its estimate misses the ML deviance by %.3g",
                 name, coef_name, anchor), call. = FALSE)
  }
  zeta <- function(b) {
    d <- held_deviance(b) - dev0
    if (d < -DEVIANCE_TOL) {
      stop(sprintf("%s: the refit with %s held at %.17g has a deviance %.3g below the ML fit's",
                   name, coef_name, b, -d), call. = FALSE)
    }
    sign(b - b_hat) * sqrt(max(0, d))
  }
  bound <- function(s) {
    inner <- b_hat
    outer <- NA_real_
    for (k in seq_len(MAX_DOUBLINGS)) {
      cand <- b_hat + s * z * se * 2^(k - 1L)
      if (s * zeta(cand) > z) {
        outer <- cand
        break
      }
      inner <- cand
    }
    if (is.na(outer)) {
      stop(sprintf("%s: %s side %+d does not bracket zeta = %g within %d doublings", name,
                   coef_name, s, s * z, MAX_DOUBLINGS), call. = FALSE)
    }
    grid <- b_hat + (outer - b_hat) * seq_len(GRID_POINTS) / GRID_POINTS
    along <- s * vapply(grid, zeta, 0)
    if (any(diff(c(0, along)) <= 0)) {
      stop(sprintf("%s: zeta of %s is not monotonic on side %+d: %s", name, coef_name, s,
                   paste(signif(s * along, 6), collapse = ", ")), call. = FALSE)
    }
    uniroot(function(b) zeta(b) - s * z, sort(c(inner, outer)), tol = 1e-10 * se)$root
  }
  c(bound(-1), bound(1))
}

profile_reference <- function(entry) {
  if (length(entry$members) != 1L) {
    stop(sprintf("%s: a profile entry has one member", entry$name), call. = FALSE)
  }
  mem <- entry$members[[1]]
  if (!identical(mem$fun, "lme4::lmer")) {
    stop(sprintf("%s: a profile member is an lme4::lmer fit, not %s", entry$name, mem$fun),
         call. = FALSE)
  }
  beta_method <- if (is.null(entry$beta_method)) "lme4_profile" else entry$beta_method
  if (!beta_method %in% BETA_METHODS) {
    stop(sprintf("%s: beta_method %s is not one of %s", entry$name, beta_method,
                 paste(BETA_METHODS, collapse = ", ")), call. = FALSE)
  }
  data <- read_entry_data(entry)
  m <- checked_fit(entry$name, fit_member(mem, data))
  ml <- if (isREML(m)) checked_fit(entry$name, refitML(m)) else m
  rows <- parameter_rows(ml)
  built <- vapply(rows, `[[`, "", "parameter")
  beta_names <- names(fixef(ml))
  if (identical(beta_method, "lme4_profile")) {
    ci <- profile_confint(entry$name, m, entry$level)
    method <- setNames(rep("lme4_profile", nrow(ci)), rownames(ci))
  } else {
    ci_var <- profile_confint(entry$name, m, entry$level, parm = "theta_")
    if (any(rownames(ci_var) %in% beta_names)) {
      stop(sprintf("%s: confint(parm = \"theta_\") returned coefficient rows", entry$name),
           call. = FALSE)
    }
    ci_beta <- t(vapply(seq_along(beta_names), function(j) {
      offset_interval(entry$name, mem, data, ml, j, entry$level)
    }, c(0, 0)))
    rownames(ci_beta) <- beta_names
    ci <- rbind(ci_var, ci_beta)
    method <- setNames(c(rep("lme4_profile", nrow(ci_var)), rep("offset_refit", nrow(ci_beta))),
                       rownames(ci))
  }
  if (!setequal(built, rownames(ci)) || anyDuplicated(built) > 0L ||
      anyDuplicated(rownames(ci)) > 0L) {
    stop(sprintf("%s: parameter names %s do not match the interval rows %s", entry$name,
                 paste(built, collapse = ", "), paste(rownames(ci), collapse = ", ")),
         call. = FALSE)
  }
  by_name <- setNames(rows, built)
  out_rows <- lapply(rownames(ci), function(p) {
    r <- by_name[[p]]
    r$lower <- unname(ci[p, 1])
    r$upper <- unname(ci[p, 2])
    r$method <- unname(method[[p]])
    check_finite(r[c("estimate", "lower", "upper")], entry$name, paste0("/rows/", p))
    r
  })
  out <- list(level = entry$level, deviance = deviance(ml), rows = out_rows)
  check_finite(out[c("level", "deviance")], entry$name)
  list(out = out, note = if (identical(beta_method, "offset_refit")) OFFSET_NOTE else NULL)
}

for (entry in select_entries("profile")) {
  cat(sprintf("%s ...\n", entry$name))
  obj <- withCallingHandlers({
    ref <- profile_reference(entry)
    prov <- provenance(GENERATOR)
    if (!is.null(ref$note)) prov$note <- ref$note
    c(list(provenance = prov), ref$out)
  }, warning = function(w) {
    stop(sprintf("%s: %s", entry$name, conditionMessage(w)), call. = FALSE)
  })
  write_reference(obj, file.path(out_dir, paste0(entry$name, ".json")))
}
