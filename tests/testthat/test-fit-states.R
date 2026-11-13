# Every fit state the kernel reaches (converged, singular, failed, aliased),
# on every model shape where it exists, through every public call on a fit.
# A call must return or raise the package's own error, never a base R error;
# on a fit with no estimates (every coefficient NA) every estimate it returns
# is NA; and every per-grouping output comes in the order the formula declares
# the groupings.
#
# The data table is deterministic (no RNG) and is built the same way in
# glmm-rs/tests/fit_states.rs and glmm-py/tests/test_fit_states.py: change the
# three together. R's CI fails on any warning, so every fit and call runs
# inside suppressWarnings() and the states are read from the fit's fields.

fs_n <- 96L

# The 39 fits: every (state, family, shape) that exists. Shapes: A `y ~ f + x`;
# B `+ (1 | g)`; C `+ (1 | a) + (1 | b)`; D `+ (1 + x | g)`.
fs_rows <- function() {
  rows <- list()
  push <- function(state, binomial, shapes) {
    for (shape in shapes) {
      rows[[length(rows) + 1L]] <<- list(state = state, binomial = binomial, shape = shape)
    }
  }
  for (binomial in c(FALSE, TRUE)) {
    push("converged", binomial, c("A", "B", "C", "D"))
    push("singular", binomial, c("B", "C", "D"))
    push("too_few_rows", binomial, c("A", "B", "C", "D"))
    push("aliased_fixed", binomial, c("A", "B", "C", "D"))
  }
  push("constant_response", FALSE, c("A", "B", "C", "D"))
  push("separated", TRUE, c("A", "B", "C", "D"))
  push("aliased_slope", FALSE, "D")
  rows
}

fs_label <- function(row) {
  sprintf("%s/%s/%s", row$state, if (row$binomial) "binomial" else "gaussian", row$shape)
}

fs_formula <- function(row) {
  re <- switch(row$shape, A = "", B = " + (1 | g)", C = " + (1 | a) + (1 | b)",
               D = " + (1 + x | g)")
  switch(row$state,
    aliased_fixed = paste0("y ~ f + x + x2", re),
    aliased_slope = "y ~ f + x + x2 + (1 + x2 | g)",
    paste0("y ~ f + x", re))
}

# The row's formula without `x` in the fixed part, for the two-fit calls.
fs_reduced_formula <- function(row) sub(" + x", "", fs_formula(row), fixed = TRUE)

# The declared groupings and the random effects each one carries.
fs_groupings <- function(row) {
  if (row$state == "aliased_slope" || row$shape == "D") return(c(g = 2L))
  switch(row$shape, A = integer(), B = c(g = 1L), C = c(a = 1L, b = 1L))
}

# The failure each failed row's one severe warning names.
fs_failure <- function(row) {
  if (row$state == "constant_response" && row$shape != "A") return("constant_response")
  switch(row$state, separated = "glm_diverged", too_few_rows = "too_few_rows", NULL)
}

fs_family <- function(row) if (row$binomial) binomial() else gaussian()

# The table `row` is fitted on: all 96 rows, or rows 0, 13, 26 and 39 of them
# (0-based) for a too-few-rows fit (n = p = 4).
fs_data <- function(row) {
  i <- 0:(fs_n - 1L)
  x <- sin(1.7 * i + 0.3)
  z <- 0.8 * sin(2.3 * i + 0.7)
  g <- (i %/% 3L) %% 6L
  a <- i %% 4L
  b <- (i %/% 4L) %% 12L
  shape <- if (row$state == "aliased_slope") "D" else row$shape
  y <- if (row$state == "constant_response") {
    rep(3, fs_n)
  } else if (row$state == "separated") {
    as.numeric(x > 0)
  } else if (row$state == "singular" && row$binomial) {
    as.numeric((i %/% 16L) %% 2L == 0L)
  } else if (row$state == "singular" && shape == "C") {
    # Exact two-way centring on the balanced crossed design.
    z - ave(z, a) - ave(z, b) + mean(z)
  } else if (row$state == "singular") {
    z - ave(z, g)
  } else {
    u <- c(-1.0, 0.6, 0.2, -0.4, 1.1, -0.5)
    re <- switch(shape,
      A = 0,
      B = u[g + 1L],
      C = c(-0.8, 0.5, 0.9, -0.6)[a + 1L] + 0.7 * sin(b + 1),
      D = u[g + 1L] + c(0.5, -0.4, 0.3, -0.6, 0.2, 0.4)[g + 1L] * x)
    eta <- 0.5 * x + c(0, 0.3, -0.4)[i %% 3L + 1L] + re
    if (row$binomial) {
      t <- 0.618034 * i + 0.1
      as.numeric(t - floor(t) < 1 / (1 + exp(-2 * eta)))
    } else {
      1 + eta + z
    }
  }
  d <- data.frame(y = y, f = factor(c("p", "q", "r")[i %% 3L + 1L]),
                  g = factor(paste0("g", g)), a = factor(paste0("a", a)),
                  b = factor(paste0("b", b)), x = x, x2 = 2 * x)
  if (row$state == "too_few_rows") d <- d[c(1L, 14L, 27L, 40L), ]
  d
}

# One fitted row and what its calls need: the reduced fit of the two-fit
# calls, or the error that refused it.
fs_context <- function(row) {
  d <- fs_data(row)
  fit <- suppressWarnings(fastglmm(fs_formula(row), d, family = fs_family(row)))
  reduced <- tryCatch(
    suppressWarnings(fastglmm(fs_reduced_formula(row), d, family = fs_family(row))),
    error = identity)
  list(row = row, label = fs_label(row), data = d, fit = fit, reduced = reduced)
}

fs_contexts <- lapply(fs_rows(), fs_context)

fs_no_estimates <- function(fit) all(is.na(fit$beta))

# TRUE when `e` is an error the package raised: a native refusal (class
# fastglmm_error_*), a stop(call. = FALSE), which has no call, or a stop()
# inside a package function. A base R error such as `subscript out of
# bounds` carries a call to a function outside the package.
fs_is_refusal <- function(e) {
  if (any(startsWith(class(e), "fastglmm_error_"))) return(TRUE)
  call <- conditionCall(e)
  if (is.null(call)) return(TRUE)
  fn <- call[[1L]]
  is.name(fn) && exists(as.character(fn), envir = asNamespace("fastglmm"), inherits = FALSE)
}

# The columns `cols` of a table that has them, as one numeric vector.
fs_columns <- function(tab, cols) unlist(tab[intersect(cols, names(tab))], use.names = FALSE)

fs_test_columns <- c("Sum Sq", "Mean Sq", "DenDF", "F value", "Eta2_partial", "Pr(>F)",
                     "Chisq", "Pr(>Chisq)")

# A field of the fit, read as it is.
fs_field <- function(name, estimate, pick = function(v) unlist(v, use.names = FALSE)) {
  list(estimate = estimate, run = function(ctx) pick(ctx$fit[[name]]))
}

# The estimate columns of a stored per-term table (NULL, or list(error =) when
# the formula gets none).
fs_term_table <- function(v) {
  unlist(v[c("statistic", "den_df", "p_value", "sum_sq", "mean_sq", "eta2_partial")],
         use.names = FALSE)
}

# The estimate columns of a model comparison table, from the row of the fit
# under test (row `own`) and the test cells on the second row.
fs_comparison <- function(tab, own) {
  c(fs_columns(tab[own, ], c("AIC", "BIC", "logLik", "RSS")),
    fs_columns(tab[2L, ], c("Chisq", "Pr(>Chisq)", "F", "Pr(>F)", "Sum of Sq")))
}

# Every public call on a fit: whether its numbers are estimates (NA on a fit
# with no estimates), and the call with its fixed arguments, returning those
# numbers. A name without "(" is a method of class "fastglmm" or a field of
# the fit, the list "every public method and field..." checks; a name with
# "(" is another call that reads a fit. A method and a field of one name
# share an entry that reads both.
fs_calls <- list(
  AIC = list(estimate = TRUE, run = function(ctx) AIC(ctx$fit)),
  anova = list(estimate = TRUE, run = function(ctx) fs_columns(anova(ctx$fit), fs_test_columns)),
  BIC = list(estimate = TRUE, run = function(ctx) BIC(ctx$fit)),
  coef = list(estimate = TRUE, run = function(ctx) coef(ctx$fit)),
  confint = list(estimate = TRUE, run = function(ctx) as.vector(confint(ctx$fit))),
  deviance = list(estimate = TRUE, run = function(ctx) c(deviance(ctx$fit), ctx$fit$deviance)),
  family = list(estimate = FALSE, run = function(ctx) list(family(ctx$fit), ctx$fit$family)),
  fitted = list(estimate = TRUE, run = function(ctx) c(fitted(ctx$fit), ctx$fit$fitted)),
  fixef = list(estimate = TRUE, run = function(ctx) fixef(ctx$fit)),
  formula = list(estimate = FALSE, run = function(ctx) list(formula(ctx$fit), ctx$fit$formula)),
  glance = list(estimate = TRUE, run = function(ctx) {
    fs_columns(glance(ctx$fit), c("logLik", "AIC", "BIC", "deviance", "df.residual"))
  }),
  isSingular = list(estimate = FALSE, run = function(ctx) isSingular(ctx$fit)),
  logLik = list(estimate = TRUE, run = function(ctx) as.numeric(logLik(ctx$fit))),
  model.frame = list(estimate = FALSE, run = function(ctx) model.frame(ctx$fit)),
  nobs = list(estimate = FALSE, run = function(ctx) c(nobs(ctx$fit), ctx$fit$nobs)),
  predict = list(estimate = TRUE, run = function(ctx) predict(ctx$fit)),
  print = list(estimate = FALSE, run = function(ctx) capture.output(print(ctx$fit))),
  ranef = list(estimate = TRUE, run = function(ctx) {
    unlist(lapply(ranef(ctx$fit), function(b) c(unlist(b), attr(b, "postVar"))),
           use.names = FALSE)
  }),
  residuals = list(estimate = TRUE, run = function(ctx) residuals(ctx$fit)),
  # A binomial fit's sigma() is the family's fixed 1, not an estimate.
  sigma = list(estimate = TRUE, run = function(ctx) {
    s <- sigma(ctx$fit)
    if (ctx$row$binomial) numeric() else s
  }),
  simulate = list(estimate = TRUE, run = function(ctx) {
    unlist(simulate(ctx$fit, nsim = 2, seed = 1), use.names = FALSE)
  }),
  summary = list(estimate = TRUE, run = function(ctx) {
    s <- summary(ctx$fit)
    c(s$coefficients, s$criterion)
  }),
  terms = list(estimate = FALSE, run = function(ctx) terms(ctx$fit)),
  tidy = list(estimate = TRUE, run = function(ctx) {
    fs_columns(tidy(ctx$fit), c("estimate", "std.error", "statistic", "df", "p.value"))
  }),
  VarCorr = list(estimate = TRUE, run = function(ctx) {
    unlist(lapply(VarCorr(ctx$fit), function(v) c(v, attr(v, "stddev"))), use.names = FALSE)
  }),
  vcov = list(estimate = TRUE, run = function(ctx) c(vcov(ctx$fit), ctx$fit$vcov)),

  beta = fs_field("beta", TRUE),
  se = fs_field("se", TRUE),
  varcorr = fs_field("varcorr", TRUE),
  stddev_se = fs_field("stddev_se", TRUE),
  aliased = fs_field("aliased", FALSE),
  coef_df = fs_field("coef_df", TRUE),
  coef_p = fs_field("coef_p", TRUE),
  df_method = fs_field("df_method", FALSE),
  # `retained` is a list of column positions, not an estimate.
  satterthwaite_record = fs_field("satterthwaite_record", TRUE, function(v) {
    unlist(v[setdiff(names(v), "retained")], use.names = FALSE)
  }),
  rss = fs_field("rss", TRUE),
  r_squared = fs_field("r_squared", TRUE),
  adj_r_squared = fs_field("adj_r_squared", TRUE),
  dispersion = fs_field("dispersion", TRUE),
  dispersion_held = fs_field("dispersion_held", FALSE),
  converged = fs_field("converged", FALSE),
  singular = fs_field("singular", FALSE),
  diagnostics = fs_field("diagnostics", FALSE),
  warnings = fs_field("warnings", FALSE),
  n_eval = fs_field("n_eval", FALSE),
  n_grad = fs_field("n_grad", FALSE),
  outer_stop = fs_field("outer_stop", FALSE),
  # The gradient search's own stopping measure, a diagnostic of the search.
  outer_decrement = fs_field("outer_decrement", FALSE),
  loglik = fs_field("loglik", TRUE),
  df = fs_field("df", FALSE),
  reml = fs_field("reml", FALSE),
  re_group_names = fs_field("re_group_names", FALSE),
  re_group_terms = fs_field("re_group_terms", FALSE),
  re_term_index = fs_field("re_term_index", FALSE),
  ranef_blocks = fs_field("ranef_blocks", TRUE, function(v) {
    unlist(lapply(v, function(b) c(b$values, b$condvar)), use.names = FALSE)
  }),
  y = fs_field("y", FALSE),
  weights = fs_field("weights", FALSE),
  weight_power = fs_field("weight_power", TRUE),
  estimated_weight_power = fs_field("estimated_weight_power", FALSE),
  se_type = fs_field("se_type", FALSE),
  model_vcov = fs_field("model_vcov", TRUE),
  robust = fs_field("robust", TRUE),
  design_summary = fs_field("design_summary", FALSE),
  offset = fs_field("offset", FALSE),
  objective = fs_field("objective", FALSE),
  objective_nodes = fs_field("objective_nodes", FALSE),
  term_table_type3 = fs_field("term_table_type3", TRUE, fs_term_table),
  term_table_type2 = fs_field("term_table_type2", TRUE, fs_term_table),
  term_table_type3_lrt = fs_field("term_table_type3_lrt", TRUE, fs_term_table),
  term_table_type2_lrt = fs_field("term_table_type2_lrt", TRUE, fs_term_table),
  term_test_auto = fs_field("term_test_auto", FALSE),
  call = fs_field("call", FALSE),
  family_name = fs_field("family_name", FALSE),
  link = fs_field("link", FALSE),
  frame = fs_field("frame", FALSE),
  data_name = fs_field("data_name", FALSE),
  nAGQ = fs_field("nAGQ", FALSE),

  `confint(method = "profile")` = list(estimate = TRUE, run = function(ctx) {
    as.vector(confint(ctx$fit, method = "profile"))
  }),
  `residuals(type = "pearson")` = list(estimate = TRUE, run = function(ctx) {
    residuals(ctx$fit, type = "pearson")
  }),
  `predict(se.fit = TRUE, type = "link")` = list(estimate = TRUE, run = function(ctx) {
    unlist(predict(ctx$fit, newdata = ctx$data, se.fit = TRUE, type = "link"), use.names = FALSE)
  }),
  `predict(se.fit = TRUE, type = "response")` = list(estimate = TRUE, run = function(ctx) {
    unlist(predict(ctx$fit, newdata = ctx$data, se.fit = TRUE, type = "response"),
           use.names = FALSE)
  }),
  `print(summary())` = list(estimate = FALSE, run = function(ctx) {
    capture.output(print(summary(ctx$fit)))
  }),
  `print(VarCorr())` = list(estimate = FALSE, run = function(ctx) {
    capture.output(print(VarCorr(ctx$fit)))
  }),
  # Only this fit's own numbers and the test: the reduced fit is another fit,
  # which can have estimates where this one has none.
  `anova(fit0, fit)` = list(estimate = TRUE, run = function(ctx) {
    if (inherits(ctx$reduced, "error")) stop(ctx$reduced)
    fs_comparison(anova(ctx$reduced, ctx$fit), "m1")
  }),
  `r2()` = list(estimate = TRUE, run = function(ctx) r2(ctx$fit)),
  `icc()` = list(estimate = TRUE, run = function(ctx) icc(ctx$fit)),
  # An infinite df marks a z row, a label of the test, not an estimate.
  `pairwise()` = list(estimate = TRUE, run = function(ctx) {
    tab <- pairwise(ctx$fit, "f")
    df <- tab$df[!is.infinite(tab$df)]
    c(fs_columns(tab, c("estimate", "SE", "t.ratio", "z.ratio", "p.value")), df)
  }),
  `confint(pairwise())` = list(estimate = TRUE, run = function(ctx) {
    fs_columns(confint(pairwise(ctx$fit, "f")), c("estimate", "lower.CL", "upper.CL"))
  }),
  `contrast_test()` = list(estimate = TRUE, run = function(ctx) {
    p <- length(ctx$fit$beta)
    fs_columns(contrast_test(ctx$fit, diag(p)[1L, , drop = FALSE]), fs_test_columns)
  }),
  `compare()` = list(estimate = TRUE, run = function(ctx) {
    cmp <- compare(ctx$data, fs_reduced_formula(ctx$row), fs_formula(ctx$row),
                   family = fs_family(ctx$row))
    fs_comparison(cmp$table, "f1")
  })
)

# Methods and fields of a fit no row calls, with the reason.
fs_skip <- character()

# Runs `call` on `ctx`: list(value =) when it returns, list(error =) when it
# raises.
fs_run <- function(call, ctx) {
  tryCatch(list(value = suppressWarnings(call$run(ctx))), error = function(e) list(error = e))
}

# Passes when `x` is empty; a failure lists every entry of `x`.
fs_expect_none <- function(x) {
  expect(length(x) == 0L, paste(c(sprintf("%s:", deparse(substitute(x))), x), collapse = "\n"))
}

fs_summary_lines <- function(fit) capture.output(print(suppressWarnings(summary(fit))))

test_that("every fit state reaches its state", {
  failures <- character()
  for (ctx in fs_contexts) {
    fit <- ctx$fit
    kinds <- vapply(fit$diagnostics$notes, function(n) n$kind, character(1))
    x2_aliased <- "x2" %in% names(fit$aliased) && fit$aliased[["x2"]] && is.na(fit$beta[["x2"]])
    ok <- switch(ctx$row$state,
      converged = fit$converged && !fit$singular,
      singular = fit$converged && fit$singular,
      constant_response = if (ctx$row$shape == "A") {
        fit$converged
      } else {
        !fit$converged && fs_no_estimates(fit)
      },
      too_few_rows = !fit$converged && fs_no_estimates(fit),
      separated = !fit$converged && "separated" %in% kinds,
      aliased_fixed = ,
      aliased_slope = fit$converged && x2_aliased)
    if (!ok) {
      failures <- c(failures, sprintf("%s: converged %s, singular %s, beta %s, notes %s",
                                      ctx$label, fit$converged, fit$singular,
                                      paste(fit$beta, collapse = " "),
                                      paste(kinds, collapse = " ")))
    }
  }
  expect_identical(length(fs_contexts), 39L)
  fs_expect_none(failures)
})

test_that("every call on every fit returns or raises the package's error", {
  failures <- character()
  ran <- 0L
  pw_returned <- character()
  fail <- function(ctx, what) failures <<- c(failures, paste0(ctx$label, ": ", what))
  for (ctx in fs_contexts) {
    fit <- ctx$fit
    for (name in names(fs_calls)) {
      ran <- ran + 1L
      out <- fs_run(fs_calls[[name]], ctx)
      if (!is.null(out$error)) {
        if (!fs_is_refusal(out$error)) {
          fail(ctx, sprintf("%s: not the package's error: %s (call %s)", name,
                            conditionMessage(out$error), deparse1(conditionCall(out$error))))
        }
        next
      }
      values <- out$value
      if (fs_calls[[name]]$estimate && fs_no_estimates(fit) && any(!is.na(values))) {
        fail(ctx, sprintf("%s: a number is not NA on a fit with no estimates: %s", name,
                          paste(values, collapse = " ")))
      }
      if (name == "simulate" && fit$converged && !all(is.finite(values))) {
        fail(ctx, sprintf("simulate: a draw is not finite on a converged fit: %s",
                          paste(values, collapse = " ")))
      }
    }

    # The summary's title and groups line agree with the fit's groupings.
    lines <- fs_summary_lines(fit)
    mixed <- length(fit$re_group_names) > 0L
    if (grepl("mixed model", lines[1L], fixed = TRUE) != mixed) {
      fail(ctx, sprintf("summary title: %s", lines[1L]))
    }
    groups <- fs_groupings(ctx$row)
    counts <- vapply(names(groups), function(g) length(unique(ctx$data[[g]])), integer(1))
    want <- if (length(groups)) {
      sprintf("Number of obs: %d, groups:  %s", nrow(ctx$data),
              paste(names(groups), counts, sep = ", ", collapse = "; "))
    } else {
      sprintf("Number of obs: %d", nrow(ctx$data))
    }
    # cat() ends the line of a fit without groupings with a space.
    got <- sub(" $", "", grep("^Number of obs", lines, value = TRUE))
    if (!identical(got, want)) fail(ctx, sprintf("summary groups line: %s", paste(got, collapse = " | ")))
    if (ctx$label == "too_few_rows/gaussian/A" && !any(grepl("on NA degrees of freedom", lines))) {
      fail(ctx, "summary footer has no 'on NA degrees of freedom'")
    }

    # A failed fit holds exactly one severe warning, of its failure's kind.
    failure <- fs_failure(ctx$row)
    if (!is.null(failure)) {
      severe <- fit$warnings$kind[fit$warnings$tier == "severe"]
      if (!identical(severe, failure)) {
        fail(ctx, sprintf("severe warnings: %s", paste(severe, collapse = " ")))
      }
    }

    # pairwise() and its intervals of a fit that did not converge open with the note.
    if (!fit$converged) {
      tab <- tryCatch(suppressWarnings(pairwise(fit, "f")), error = function(e) NULL)
      if (!is.null(tab)) {
        pw_returned <- c(pw_returned, ctx$label)
        if (!identical(attr(tab, "footer")[1L], fastglmm:::.NOT_CONVERGED)) {
          fail(ctx, sprintf("pairwise note: %s", attr(tab, "footer")[1L]))
        }
        ci_note <- attr(suppressWarnings(confint(tab)), "footer")[1L]
        if (!identical(ci_note, paste(
          "The fit did not converge, so its estimates are not at a maximum and these intervals",
          "are not reliable. Change the model until it converges (a simpler random part often",
          "helps), then compute them again."))) {
          fail(ctx, sprintf("pairwise confint note: %s", ci_note))
        }
      }
    }
  }
  expect_identical(ran, 39L * length(fs_calls))
  expect_identical(length(pw_returned), 15L)
  expect_true("separated/binomial/A" %in% pw_returned)
  fs_expect_none(failures)
})

test_that("per-grouping outputs follow the declaration order", {
  failures <- character()
  checked <- 0L
  for (ctx in fs_contexts) {
    groups <- fs_groupings(ctx$row)
    if (!length(groups)) next
    checked <- checked + 1L
    fit <- ctx$fit
    declared <- names(groups)
    counts <- vapply(declared, function(g) length(unique(ctx$data[[g]])), integer(1))
    bad <- function(what, got) {
      failures <<- c(failures, sprintf("%s: %s: %s", ctx$label, what, paste(got, collapse = " ")))
    }
    out <- tryCatch(suppressWarnings({
      if (!identical(fit$re_group_names, declared)) bad("re_group_names", fit$re_group_names)
      vc <- VarCorr(fit)
      if (!identical(names(vc), declared)) bad("names(VarCorr)", names(vc))
      re <- ranef(fit)
      if (length(re) && !identical(names(re), declared)) bad("names(ranef)", names(re))
      if (length(fit$varcorr) != length(fit$re_group_names)) bad("length(varcorr)", length(fit$varcorr))
      for (k in seq_along(declared)) {
        q <- length(fit$re_group_terms[[k]])
        if (q != groups[[k]]) bad(sprintf("re_group_terms[[%d]]", k), fit$re_group_terms[[k]])
        if (length(fit$varcorr[[k]]) != q * (q + 1L) / 2L) {
          bad(sprintf("varcorr[[%d]] length", k), length(fit$varcorr[[k]]))
        }
        if (length(re) && !identical(dim(re[[k]]), c(counts[[k]], q))) {
          bad(sprintf("ranef[[%d]] dim", k), dim(re[[k]]))
        }
      }
      prof <- tryCatch(confint(fit, method = "profile"), error = function(e) NULL)
      if (!is.null(prof)) {
        grp <- sub(".*\\|", "", grep("|", rownames(prof), fixed = TRUE, value = TRUE))
        if (!identical(unique(grp), declared) || is.unsorted(match(grp, declared))) {
          bad("confint(method = \"profile\") rows", rownames(prof))
        }
      }
      NULL
    }), error = identity)
    if (!is.null(out)) bad("raised", conditionMessage(out))
  }
  expect_identical(checked, 31L)
  fs_expect_none(failures)
})

test_that("every public method and field of a fit is called or skipped", {
  methods <- unique(attr(utils::methods(class = "fastglmm"), "info")$generic)
  fields <- unique(unlist(lapply(fs_contexts, function(ctx) names(ctx$fit))))
  public <- union(methods, grep("^\\.", fields, value = TRUE, invert = TRUE))
  listed <- c(grep("(", names(fs_calls), fixed = TRUE, value = TRUE, invert = TRUE),
              names(fs_skip))
  expect_gte(length(methods), 26L)
  not_called <- setdiff(public, listed)
  not_public <- setdiff(listed, public)
  fs_expect_none(not_called)
  fs_expect_none(not_public)
})

test_that("a fit refused before fitting raises the package's error", {
  empty <- data.frame(y = numeric(), x = numeric(), g = factor())
  e <- tryCatch(fastglmm(y ~ x + (1 | g), empty), error = identity)
  expect_true(inherits(e, "error") && fs_is_refusal(e))
  expect_identical(conditionMessage(e), "Data has no rows (n = 0); a fit needs at least one row")
  d <- fs_data(list(state = "converged", binomial = FALSE, shape = "B"))
  e <- tryCatch(fastglmm(~ x, d), error = identity)
  expect_true(inherits(e, "error") && fs_is_refusal(e))
  expect_identical(conditionMessage(e),
                   "formula syntax error: the formula has no response; write it as y ~ x")
})

test_that("predicting the training rows finds every row estimable", {
  failures <- character()
  predicted <- 0L
  aliased_predicted <- 0L
  for (ctx in fs_contexts) {
    lost <- FALSE
    ok <- tryCatch({
      withCallingHandlers(
        predict(ctx$fit, ctx$data, re.form = NA),
        warning = function(w) {
          if (inherits(w, "fastglmm_rows_not_estimable")) lost <<- TRUE
          invokeRestart("muffleWarning")
        }
      )
      TRUE
    }, error = function(e) FALSE)
    if (ok) {
      predicted <- predicted + 1L
      if (startsWith(ctx$row$state, "aliased")) aliased_predicted <- aliased_predicted + 1L
    }
    if (lost) failures <- c(failures, paste0(ctx$label, ": a training row is not estimable"))
  }
  expect_identical(length(fs_contexts), 39L)
  expect_identical(predicted, 39L)
  expect_identical(aliased_predicted, 9L)
  fs_expect_none(failures)
})

test_that("no fit is at the negative binomial shape limit", {
  failures <- character()
  for (ctx in fs_contexts) {
    kinds <- vapply(ctx$fit$diagnostics$notes, function(n) n$kind, character(1))
    if ("nb_shape_at_limit" %in% kinds) failures <- c(failures, ctx$label)
  }
  expect_identical(length(fs_contexts), 39L)
  fs_expect_none(failures)
})

test_that("one_trial_per_level marks the one-row-per-level fits", {
  # The too-few-rows rows 0, 13, 26 and 39 fall in four different levels of g,
  # of a and of b, so every level of a binomial too-few-rows fit holds one 0/1
  # row. No other fit carries the note.
  failures <- character()
  for (ctx in fs_contexts) {
    notes <- Filter(function(n) n$kind == "one_trial_per_level", ctx$fit$diagnostics$notes)
    got <- vapply(notes, function(n) n$detail, character(1))
    want <- character()
    if (ctx$row$state == "too_few_rows" && ctx$row$binomial) {
      want <- switch(ctx$row$shape, A = character(), B = "g", C = c("a", "b"), D = "g")
    }
    if (!identical(got, want)) {
      failures <- c(failures, sprintf("%s: %s", ctx$label, paste(got, collapse = " ")))
    }
  }
  expect_identical(length(fs_contexts), 39L)
  fs_expect_none(failures)
})

test_that("no fit is at the gamma dispersion limit", {
  # No row is Gamma, so the note on any of them is raised on the wrong family.
  failures <- character()
  for (ctx in fs_contexts) {
    kinds <- vapply(ctx$fit$diagnostics$notes, function(n) n$kind, character(1))
    if ("dispersion_at_limit" %in% kinds) failures <- c(failures, ctx$label)
  }
  expect_identical(length(fs_contexts), 39L)
  fs_expect_none(failures)
})

test_that("summary counts the observed levels of an unbalanced nested grouping", {
  # A has 4 levels holding 2, 4, 1 and 3 levels of B: 10 A:B combinations,
  # which the kernel pads to 4 x 4 slots. lme4 prints 10.
  counts <- c(2L, 4L, 1L, 3L)
  a <- rep(paste0("a", seq_along(counts) - 1L), counts * 3L)
  b <- unlist(lapply(counts, function(k) rep(paste0("b", seq_len(k) - 1L), each = 3L)))
  i <- seq_along(a) - 1L
  d <- data.frame(y = sin(1.7 * i) + 0.1 * i, x = cos(2.3 * i), A = a, B = b)
  m <- suppressWarnings(fastglmm(y ~ x + (1 | A/B), d))
  text <- paste(capture.output(print(summary(m))), collapse = "\n")
  expect_true(grepl("A:B, 10", text, fixed = TRUE))
})
