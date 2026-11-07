# Robust fits (se_type): CR2 and HC2 against clubSandwich, the default, the
# robust notes and refusals. Mirrors glmm-py/tests/test_robust.py - change
# together. The references live in the repository's validation/ tree; a source
# tarball ships none, so glmm-r/tools/cran-tarball.sh copies the files read
# here into tests/testthat/fixtures/. Its file list mirrors the names used in
# this file - change together.

fixtures <- test_path("fixtures")
validation <- test_path("..", "..", "..", "validation")

golden <- function(name) {
  f <- paste0(name, ".json")
  jsonlite::fromJSON(if (dir.exists(fixtures)) {
    file.path(fixtures, f)
  } else {
    file.path(validation, "goldens", "inference", "cr2", f)
  })
}

cr2_data <- function(name) {
  f <- paste0(name, ".csv")
  d <- utils::read.csv(if (dir.exists(fixtures)) {
    file.path(fixtures, f)
  } else {
    file.path(validation, "data", "inference", f)
  })
  if (!is.null(d$g)) d$g <- factor(paste0("g", d$g))
  d
}

skip_without_references <- function() {
  skip_if_not(dir.exists(fixtures) || dir.exists(validation),
              "neither tests/testthat/fixtures/ nor the validation/ tree is here")
  skip_if_not_installed("jsonlite")
}

quiet_fit <- function(formula, data, ...) {
  suppressWarnings(fastglmm(formula, data, weights = w, ...))
}

RI <- y ~ x1 + x2 + (1 | g)
RS <- y ~ x1 + x2 + (1 + x1 | g)
WLS <- y ~ x1 + x2

rel <- function(got, want) max(abs(unname(got) / want - 1))

test_that("CR2 matches clubSandwich on an nlme fit (G5)", {
  # One cell, for the wiring; the kernel's tests cover every cell.
  skip_without_references()
  g <- golden("cr2_lme_ri_estimated_s10")
  m <- quiet_fit(RI, cr2_data("cr2_ri_s10"), weight_power = g$delta,
                 se_type = "robust", REML = TRUE)
  expect_identical(m$se_type, "robust")
  expect_lt(rel(m$se, g$se), 1e-7)
  expect_lt(rel(m$coef_df, g$df), 1e-7)
})

test_that("claimed weights with a robust SE are HC2 (G4)", {
  skip_without_references()
  g <- golden("hc2_n20")
  m <- quiet_fit(WLS, cr2_data("hc2_n20"), weight_power = 1, se_type = "robust")
  expect_identical(c(m$se_type, m$df_method), c("robust", "bell_mccaffrey"))
  expect_lt(rel(m$se, g$se), 1e-10)
  expect_lt(rel(m$coef_df, g$df), 1e-10)
  expect_lt(max(abs(unname(vcov(m)) - g$vcov)), 1e-10 * max(abs(g$vcov)))
})

test_that("the default se_type is robust on a REML weight-power fit without a random slope", {
  skip_without_references()
  # One robust and one model row, for the wiring; the kernel's tests hold the
  # whole decision table.
  rows <- list(
    list("cr2_ri_s10", RI, list(), "robust", "pustejovsky_tipton"),
    list("cr2_ri_s10", RI, list(weight_power = "estimate", REML = FALSE), "model",
         "satterthwaite"))
  for (row in rows) {
    m <- do.call(quiet_fit, c(list(row[[2]], cr2_data(row[[1]])), row[[3]]))
    expect_identical(c(m$se_type, m$df_method), c(row[[4]], row[[5]]))
    expect_identical(is.null(m$robust), row[[4]] == "model")
    expect_identical(is.null(m$model_vcov), row[[4]] == "model")
  }
  m <- quiet_fit(WLS, cr2_data("hc2_n20"))
  expect_false(m$weight_power == 1)
  expect_true(all(is.finite(m$coef_p)))
  r <- quiet_fit(RI, cr2_data("cr2_ri_s10"), weight_power = 1, se_type = "robust")
  expect_identical(dim(r$model_vcov), c(3L, 3L))
  expect_length(r$robust$er, 3L * 10L)
})

test_that("the robust notes warn with their class and text", {
  skip_without_references()
  df_small <- "fastglmm_robust_df_small"
  expect_warning(suppressWarnings(fastglmm(RS, cr2_data("cr2_rs_s10"), weights = w,
                                           weight_power = 1, se_type = "robust"),
                                  classes = df_small),
                 "asked for on a model with a random slope",
                 class = "fastglmm_robust_with_random_slope")
  d <- cr2_data("cr2_ri_s10")
  expect_warning(fastglmm(RI, d, weights = w, weight_power = 1, se_type = "robust"),
                 "robust degrees of freedom of x2 \\(2\\.[0-9]+\\) are below 4",
                 class = "fastglmm_robust_df_small")
  d$one <- as.numeric(d$g == "g1")
  expect_warning(m <- fastglmm(y ~ x1 + one + (1 | g), d, weights = w, weight_power = 1,
                               se_type = "robust"),
                 "one: one group alone carries",
                 class = "fastglmm_robust_se_unidentified")
  expect_true(is.na(m$se[["one"]]) && is.na(m$coef_df[["one"]]))
})

test_that("the weight-power search notes warn with their class and text", {
  # Narrow weights whose noise follows w^-40: the profile peaks past the search
  # range [-30, 30].
  set.seed(4)
  n <- 200
  x <- runif(n, -1, 1)
  w <- 1 + 0.05 * runif(n)
  d <- data.frame(x = x, y = 1 + 0.5 * x + rnorm(n) * w^-40, w = w)
  expect_warning(fastglmm(y ~ x, d, weights = w),
                 "The fitted weight power, .* is not a maximum",
                 class = "fastglmm_weight_power_at_bound")
  # Weights near 1e200 on noise that follows r^-1: the search range narrows to
  # keep w^delta within 1e+-250, so no fit in the search fails.
  n <- 60
  x <- runif(n, -1, 1)
  r <- 1e4^runif(n)
  r[1:2] <- c(1, 1e4)
  d <- data.frame(x = x, y = 1 + 0.5 * x + rnorm(n) * r^-0.5, w = 1e200 * r)
  failures <- 0
  m <- withCallingHandlers(fastglmm(y ~ x, d, weights = w),
                           fastglmm_weight_power_search_failures = function(w) failures <<- failures + 1,
                           warning = function(w) invokeRestart("muffleWarning"))
  expect_equal(failures, 0)
  expect_true(m$converged && all(is.finite(m$se)))
})

test_that("an estimated weight power far from the weights warns by level", {
  # Weights spread 20 exactly, noise sd r^(-power / 2): 5 lands outside [-1, 2]
  # with the variances 20^4 off the weights (Caution), 1.5 inside it (Note).
  set.seed(1)
  n <- 200
  x <- runif(n, -1, 1)
  r <- 20^runif(n)
  r[1:2] <- c(1, 20)
  e <- rnorm(n)
  d <- data.frame(x = x, y5 = 1 + 0.5 * x + e * r^-2.5, y15 = 1 + 0.5 * x + e * r^-0.75, r = r)
  expect_warning(m <- fastglmm(y5 ~ x, d, weights = r),
                 "Caution: Implausible weight power. The fitted weight power, ",
                 class = "fastglmm_weight_power_implausible", fixed = TRUE)
  row <- m$warnings[startsWith(m$warnings$kind, "weight_power"), ]
  expect_identical(c(row$kind, row$tier), c("weight_power_implausible", "caution"))
  expect_identical(row$message, sprintf(paste(
    "The fitted weight power, %.4g, is outside [-1, 2], the range the simulations behind",
    "it covered, and the fitted variances differ from the weights by up to %.4g times",
    "across the data. Check the weights, or fix the power with weight_power = 1 (the",
    "weights as given)."), m$weight_power, 20^abs(m$weight_power - 1)))
  expect_warning(m <- fastglmm(y15 ~ x, d, weights = r),
                 "Note: Weight power outside [0, 1]. The fitted weight power, ",
                 class = "fastglmm_weight_power_outside_unit", fixed = TRUE)
  row <- m$warnings[startsWith(m$warnings$kind, "weight_power"), ]
  expect_identical(c(row$kind, row$tier), c("weight_power_outside_unit", "note"))
  expect_true(m$weight_power > 1 && m$weight_power <= 2)
  expect_identical(row$message, sprintf(paste(
    "The fitted weight power, %.4g, is outside [0, 1] (0: the weights carry no",
    "information on the variance; 1: the weights as given). With few rows or a weak fit",
    "(low R\u00b2) this happens by chance and is not by itself a problem."), m$weight_power))
})

test_that("the robust refusals are errors that name the way out", {
  skip_without_references()
  d <- cr2_data("cr2_ri_s10")
  d$h <- factor(rep_len(c("h0", "h1", "h2", "h3"), nrow(d)))
  d$count <- round(abs(d$y) * 3)
  cases <- list(
    list(y ~ x1 + (1 | g) + (1 | h), "gaussian", list()),
    list(count ~ x1, "poisson", list()),
    list(RI, "gaussian", list(REML = FALSE)),
    list(WLS, "gaussian", list(weight_power = "estimate", REML = FALSE)))
  for (case in cases) {
    expect_error(do.call(fastglmm, c(list(case[[1]], d, family = case[[2]], weights = d$w,
                                          se_type = "robust"), case[[3]])),
                 'use se_type = "model"', fixed = TRUE, class = "simpleError")
  }
})

test_that("a fit saved before the robust fields reads power 1 and the model SE", {
  skip_without_references()
  fit <- quiet_fit(RI, cr2_data("cr2_ri_s10"), weight_power = 1)
  old <- fit
  old$weight_power <- NULL
  old$estimated_weight_power <- NULL
  old$se_type <- NULL
  old$model_vcov <- NULL
  old$robust <- NULL
  expect_identical(capture.output(print(summary(old))), capture.output(print(summary(fit))))
  L <- diag(3)[-1, ]
  expect_identical(contrast_test(old, L)$`F value`, contrast_test(fit, L)$`F value`)
})

test_that("summary prints the weight power and the SE footnote", {
  # Mirrors test_summary_prints_the_weight_power_and_the_se_footnote
  # (glmm-py/tests/test_robust.py) - change together.
  skip_without_references()
  lines <- function(name, formula, ...) {
    capture.output(print(summary(quiet_fit(formula, cr2_data(name), ...))))
  }
  known <- "Standard errors treat the estimated weight power as known"
  cr2 <- lines("cr2_ri_s10", RI)
  expect_true(any(grepl("^Weight power: -?[0-9.]+ \\(estimated\\)$", cr2)))
  expect_true(any(grepl("Standard errors: CR2, clustered by g; t tests use Pustejovsky-Tipton",
                        cr2, fixed = TRUE)))
  expect_false(any(grepl(known, cr2, fixed = TRUE)))
  hc2 <- lines("hc2_n20", WLS)
  expect_true("Standard errors: HC2; t tests use Bell-McCaffrey degrees of freedom." %in% hc2)
  slope <- lines("cr2_rs_s10", RS)
  expect_true(paste0(known, ".") %in% slope)
  ml <- lines("cr2_ri_s10", RI, weight_power = "estimate", REML = FALSE)
  expect_true(paste0(known, "; the simulations behind it estimated it by REML.") %in% ml)
  expect_true("Weight power: 0.5" %in% lines("cr2_ri_s10", RI, weight_power = 0.5))
  expect_false(any(grepl("Weight power", lines("cr2_ri_s10", RI, weight_power = 1))))
})

test_that("contrast_test() and pairwise() read the robust pieces", {
  # D12: a one-row test is the robust t squared on the robust df (on a unit
  # vector the fit's own se and coef_df), with the Satterthwaite pass off too;
  # a comparison is the robust contrast of its L; the same after saveRDS. A
  # joint test is the model fit's F, with a warning.
  skip_without_references()
  d <- cr2_data("cr2_ri_s10")
  d$f <- factor(paste0("f", (seq_len(nrow(d)) - 1L) %% 3L))
  form <- y ~ x1 + x2 + f + (1 | g)
  robust <- quiet_fit(form, d, weight_power = 1, se_type = "robust")
  off <- quiet_fit(form, d, weight_power = 1, se_type = "robust", satterthwaite = FALSE)
  model <- quiet_fit(form, d, weight_power = 1)
  path <- tempfile(fileext = ".rds")
  saveRDS(robust, path)
  e <- diag(length(robust$beta))
  dimnames(e) <- list(names(robust$beta), names(robust$beta))
  for (m in list(robust, off, readRDS(path))) {
    expect_no_warning(t <- contrast_test(m, e["x1", ]))
    expect_equal(t$`F value`, unname((m$beta[["x1"]] / m$se[["x1"]])^2), tolerance = 1e-12)
    expect_equal(t$DenDF, unname(m$coef_df[["x1"]]), tolerance = 1e-12)
    expect_warning(t <- contrast_test(m, e["x2", ]), "robust degrees of freedom are below 4",
                   class = "fastglmm_robust_df_small")
    expect_lt(m$coef_df[["x2"]], 4)
    expect_equal(t$DenDF, unname(m$coef_df[["x2"]]), tolerance = 1e-12)
    pw <- suppressWarnings(pairwise(m, "f", adjust = "none"))
    ls <- list(-e["ff1", ], -e["ff2", ], e["ff1", ] - e["ff2", ])
    for (i in seq_along(ls)) {
      t <- suppressWarnings(contrast_test(m, ls[[i]]))
      expect_equal(pw$t.ratio[[i]]^2, t$`F value`, tolerance = 1e-12)
      expect_equal(pw$df[[i]], t$DenDF, tolerance = 1e-12)
    }
  }
  expect_warning(joint <- contrast_test(robust, e[c("x1", "x2"), ]),
                 "model-based standard errors", class = "fastglmm_model_based_joint_test")
  want <- contrast_test(model, e[c("x1", "x2"), ])
  expect_equal(joint$`F value`, want$`F value`, tolerance = 1e-12)
  expect_equal(joint$DenDF, want$DenDF, tolerance = 1e-12)
  # anova() heading: the one-df rows on the robust df, f's row model-based;
  # on WLS every row is one df.
  wls <- quiet_fit(WLS, cr2_data("hc2_n20"), weight_power = 1, se_type = "robust")
  headings <- vapply(list(robust, wls), function(m) {
    attr(suppressWarnings(anova(m)), "heading")[1L]
  }, "")
  expect_identical(headings, c(
    "Type III Analysis of Variance Table with Pustejovsky-Tipton's and Satterthwaite's methods",
    "Type III Analysis of Variance Table with Bell-McCaffrey's method"))
})

test_that("robust comparisons and predictions warn", {
  # A comparison one group alone identifies is NaN, with the rows named; a
  # predict SE on a robust fit is the model fit's, with a warning when asked for.
  skip_without_references()
  d <- cr2_data("cr2_ri_s10")
  d$f <- factor(ifelse(d$g == "g1", "f1", paste0("f", 2L * ((seq_len(nrow(d)) - 1L) %% 2L))))
  form <- y ~ x1 + f + (1 | g)
  m <- quiet_fit(form, d, weight_power = 1, se_type = "robust")
  model <- quiet_fit(form, d, weight_power = 1)
  expect_warning(pw <- pairwise(m, "f", adjust = "none"), "^f0 - f1, f1 - f2: One group alone",
                 class = "fastglmm_robust_se_unidentified")
  expect_true(all(is.nan(pw$SE[c(1, 3)])) && is.finite(pw$SE[[2]]))
  expect_warning(got <- predict(m, se.fit = TRUE), "model-based, not robust",
                 class = "fastglmm_predict_se_model_based")
  expect_equal(got$se.fit, predict(model, se.fit = TRUE)$se.fit, tolerance = 1e-12)
  expect_no_warning(predict(m))
})

# The WLS pair and the random-intercept pair, smaller formula first.
delta_pairs <- list(
  list("hc2_n100", y ~ x1, y ~ x1 + x2),
  list("cr2_ri_s10", y ~ x1 + (1 | g), y ~ x1 + x2 + (1 | g)))

test_that("anova() of two ML weight-power fits counts each power in npar", {
  skip_without_references()
  for (p in delta_pairs) {
    d <- cr2_data(p[[1]])
    m0 <- quiet_fit(p[[2]], d, weight_power = "estimate", REML = FALSE)
    m1 <- quiet_fit(p[[3]], d, weight_power = "estimate", REML = FALSE)
    expect_true(m0$estimated_weight_power && m1$estimated_weight_power)
    t <- anova(m0, m1)
    re <- if (is.null(d$g)) 0 else 1
    expect_equal(t$npar, c(length(m0$beta), length(m1$beta)) + 2 + re)
    expect_equal(t$npar, c(m0$df, m1$df))
    expect_equal(t$Df[2], 1)
  }
})

test_that("anova() refuses pairs whose weight powers differ", {
  skip_without_references()
  for (p in delta_pairs) {
    d <- cr2_data(p[[1]])
    expect_error(anova(quiet_fit(p[[2]], d, weight_power = "estimate"),
                       quiet_fit(p[[3]], d, weight_power = "estimate")),
                 'Pass REML = FALSE and weight_power = "estimate" to both', fixed = TRUE)
    delta <- quiet_fit(p[[2]], d, weight_power = "estimate", REML = FALSE)
    claimed <- quiet_fit(p[[3]], d, weight_power = 1, REML = FALSE)
    expect_error(anova(delta, claimed), "estimated its weight power.*takes weight_power once")
    half <- quiet_fit(p[[2]], d, weight_power = 0.5, REML = FALSE)
    expect_error(anova(half, claimed),
                 "weight_power = 0.5 and the second with weight_power = 1", fixed = TRUE)
  }
  d <- cr2_data("cr2_ri_s10")
  expect_error(anova(quiet_fit(y ~ x1, d, weight_power = "estimate"),
                     quiet_fit(y ~ x1 + (1 | g), d, weight_power = "estimate")),
               "weight_power = 1 and compare again.*15\\.5%")
})

test_that("compare() fits the weight power by ML on both sides", {
  skip_without_references()
  for (p in delta_pairs) {
    d <- cr2_data(p[[1]])
    c <- suppressWarnings(compare(d, p[[2]], p[[3]], weights = w))
    for (f in list(c$fit0, c$fit1)) {
      expect_true(f$estimated_weight_power && !f$reml)
      expect_identical(f$se_type, "model")
    }
    expect_equal(c$table$npar, c(c$fit0$df, c$fit1$df))
    claimed <- suppressWarnings(compare(d, p[[2]], p[[3]], weights = w, weight_power = 1))
    expect_false(claimed$fit0$estimated_weight_power || claimed$fit1$estimated_weight_power)
  }
  d <- cr2_data("cr2_ri_s10")
  expect_error(suppressWarnings(compare(d, y ~ x1 + (1 | g), y ~ x1, weights = w)),
               "weight_power = 1 and compare again")
})

test_that("profile intervals refuse a weight power other than 1", {
  skip_without_references()
  d <- cr2_data("cr2_ri_s10")
  for (power in list("estimate", 0.5)) {
    m <- quiet_fit(RI, d, weight_power = power)
    expect_error(confint(m, method = "profile"), "weight.power")
  }
})

test_that("an explicit test on a robust WLS fit names the residual df", {
  skip_without_references()
  m <- quiet_fit(WLS, cr2_data("hc2_n100"))
  expect_identical(m$df_method, "bell_mccaffrey")
  expect_error(anova(m, test = "Wald"), "F tests on the residual df")
})
