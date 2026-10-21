# anova(m) reads the per-term table stored at fit time; anova(m0, m1)
# compares two stored fits with no refit; compare(data, f0, f1) fits both
# models on the same rows. The crate computes every statistic and p-value;
# these tests check what the R side builds from them.

# 40 groups of 8, a random intercept and a random slope on x, and a
# three-level factor f.
anova_data <- function(seed = 311) {
  set.seed(seed)
  g <- factor(rep(1:40, each = 8))
  x <- rnorm(320)
  f <- factor(sample(c("a", "b", "c"), 320, replace = TRUE))
  u0 <- rnorm(40, sd = 0.9)[as.integer(g)]
  u1 <- rnorm(40, sd = 0.4)[as.integer(g)]
  data.frame(y = 1 + 0.5 * x + c(0, 0.4, -0.3)[as.integer(f)] + u0 + u1 * x +
               rnorm(320, sd = 0.6),
             x = x, z = rnorm(320), f = f, g = g)
}

# A table's columns alone: no row names, no heading.
columns <- function(tab) unname(lapply(tab, identity))

test_that("anova(m) is the stored Type III table, an anova data frame with no wrapper", {
  d <- anova_data()
  m <- fastglmm(y ~ x * f + (1 | g), d)
  tab <- anova(m)
  expect_identical(class(tab), c("anova", "data.frame"))
  expect_identical(names(tab),
                   c("Sum Sq", "Mean Sq", "NumDF", "DenDF", "F value", "Pr(>F)"))
  expect_identical(rownames(tab), c("x", "f", "x:f"))
  expect_identical(attr(tab, "heading")[1L],
                   "Type III Analysis of Variance Table with Satterthwaite's method")
  stored <- m$term_table_type3
  expect_identical(tab[["F value"]], stored$statistic)
  expect_identical(tab$NumDF, stored$num_df)
  expect_identical(tab$DenDF, stored$den_df)
  expect_identical(tab[["Pr(>F)"]], stored$p_value)
  expect_identical(tab[["Sum Sq"]], stored$sum_sq)
  # Read from the fit, never recomputed.
  m$term_table_type3$statistic[1L] <- 99
  expect_identical(anova(m)[["F value"]][1L], 99)
  # Type II on request, from its own stored table.
  tab2 <- anova(m, type = "II")
  expect_identical(attr(tab2, "heading")[1L],
                   "Type II Analysis of Variance Table with Satterthwaite's method")
  expect_identical(tab2[["F value"]], m$term_table_type2$statistic)
  expect_error(anova(m, type = "I"), "'arg' should be one of")
  # No Satterthwaite claim when no row has its df.
  m_off <- fastglmm(y ~ x * f + (1 | g), d, satterthwaite = FALSE)
  expect_identical(attr(anova(m_off), "heading")[1L], "Type III Analysis of Variance Table")
  # A fit that did not converge says so first.
  m$converged <- FALSE
  expect_identical(attr(anova(m), "heading")[2L], paste(
    "The fit did not converge, so its estimates are not at a maximum and these tests",
    "are not reliable. Change the model until it converges (a simpler random part often",
    "helps), then test again."))
  # A fit saved by a version without these tests is refused by name.
  m$term_table_type3 <- NULL
  expect_error(anova(m), "this fit has no term_table_type3", fixed = TRUE)
  m2 <- fastglmm(y ~ x + (1 | g), d)
  m2$objective <- NULL
  expect_error(anova(m2, fastglmm(y ~ x + (1 + x | g), d)), "this fit has no objective",
               fixed = TRUE)
})

test_that("anova(m) has the Wald chi-squared columns on a GLMM and the residual df on OLS", {
  d <- benchmark_data(seed = 321, family = "binomial")
  m <- fastglmm(y ~ t * d + (1 | g), d, family = binomial())
  tab <- anova(m)
  expect_identical(class(tab), c("anova", "data.frame"))
  expect_identical(names(tab), c("Chisq", "Df", "Pr(>Chisq)"))
  expect_identical(rownames(tab), c("t", "d", "t:d"))
  expect_identical(attr(tab, "heading"), c(
    "Type III Analysis of Deviance Table",
    'Wald chi-squared tests; test = "auto": binomial model with random effects'))
  expect_identical(attr(anova(m, test = "Wald"), "heading"),
                   "Type III Analysis of Deviance Table (Wald chi-squared tests)")
  expect_identical(tab$Chisq, m$term_table_type3$statistic)

  o <- fastglmm(y ~ x + f, anova_data())
  tab_o <- anova(o)
  expect_identical(attr(tab_o, "heading")[1L], "Type III Analysis of Variance Table")
  expect_identical(tab_o$DenDF, rep(320 - 4, 2))
})

test_that("anova(m) stops with the stored refusal, and the fit itself stands", {
  d <- anova_data()
  m <- fastglmm(y ~ x + x:f, d)
  expect_true(m$converged)
  expect_type(m$term_table_type3$error, "character")
  expect_error(anova(m), m$term_table_type3$error, fixed = TRUE)
  expect_error(anova(m, type = "II"), m$term_table_type2$error, fixed = TRUE)
  # A table that failed inside the crate is an error too, with its own text.
  m$term_table_type3 <- list(error = "internal error while building the per-term table")
  expect_error(anova(m), "internal error while building the per-term table", fixed = TRUE)
})

test_that("an aliased term's row note names its coefficient, never a column index", {
  # x2 copies x, so its column is aliased and its row is not tested.
  d <- anova_data()
  d$x2 <- d$x
  tab <- anova(fastglmm(y ~ f + x + x2, d))
  expect_true(is.na(tab["x2", "F value"]))
  expect_false(is.nan(tab["x2", "F value"]))
  expect_true("x2: The coefficient x2 is aliased, so the hypothesis was not tested." %in%
                attr(tab, "heading"))
})

test_that("anova(m0, m1) compares two REML fits: no AIC or BIC column, the smaller first", {
  d <- anova_data()
  small <- fastglmm(y ~ x + (1 | g), d)
  big <- fastglmm(y ~ x + (1 + x | g), d)
  tab <- anova(small, big)
  expect_identical(class(tab), c("anova", "data.frame"))
  expect_identical(names(tab), c("npar", "logLik", "Chisq", "Df", "Pr(>Chisq)"))
  expect_identical(rownames(tab), c("m0", "m1"))
  h <- attr(tab, "heading")
  expect_identical(h[1:3], c("Model comparison: likelihood ratio test on the REML criterion",
                             paste("m0:", small$formula), paste("m1:", big$formula)))
  expect_true("Reference: 50:50 mixture of \u03c7\u00b2(1) and \u03c7\u00b2(2)." %in% h)
  expect_equal(tab$npar, c(small$df, big$df))
  expect_identical(tab$logLik, c(small$loglik, big$loglik))
  expect_true(is.na(tab$Chisq[1L]) && tab$Chisq[2L] > 0)
  expect_identical(attr(tab, "nsim"), 10000)
  expect_identical(attr(tab, "seed"), 20261008)
  # Either order: the rows keep the labels of the fits as given.
  back <- anova(big, small)
  expect_identical(rownames(back), c("m1", "m0"))
  expect_identical(columns(back), columns(tab))
  # The heading lists the models in the table's order, the smaller first.
  expect_identical(attr(back, "heading")[2:3],
                   c(paste("m1:", small$formula), paste("m0:", big$formula)))
})

test_that("anova(m0, m1) on ML fits has lme4's columns and the Laplace label on a GLMM", {
  d <- anova_data()
  m0 <- fastglmm(y ~ x + (1 | g), d, REML = FALSE)
  m1 <- fastglmm(y ~ x + f + (1 | g), d, REML = FALSE)
  tab <- anova(m0, m1)
  expect_identical(names(tab), c("npar", "AIC", "BIC", "logLik", "Chisq", "Df", "Pr(>Chisq)"))
  expect_identical(attr(tab, "heading")[1L], "Model comparison: likelihood ratio test (ML)")
  expect_equal(tab$AIC, c(AIC(m0), AIC(m1)))
  expect_equal(tab$BIC, c(BIC(m0), BIC(m1)))
  expect_identical(tab$Df[2L], 2)
  expect_true(any(attr(tab, "heading") == "Reference: \u03c7\u00b2(2)."))

  b <- benchmark_data(seed = 322, family = "binomial")
  g0 <- fastglmm(y ~ t + d + (1 | g), b, family = binomial())
  g1 <- fastglmm(y ~ t * d + (1 | g), b, family = binomial())
  expect_identical(attr(anova(g0, g1), "heading")[1L],
                   "Model comparison: likelihood ratio test (Laplace)")
})

test_that("two OLS fits get anova.lm's F and columns", {
  d <- anova_data()
  tab <- anova(fastglmm(y ~ x, d), fastglmm(y ~ x + z + f, d))
  ref <- anova(lm(y ~ x, d), lm(y ~ x + z + f, d))
  expect_identical(names(tab), names(ref))
  expect_equal(tab$Res.Df, ref$Res.Df)
  expect_equal(tab$RSS, ref$RSS, tolerance = 1e-10)
  expect_equal(tab$Df, ref$Df)
  expect_equal(tab[["Sum of Sq"]], ref[["Sum of Sq"]], tolerance = 1e-8)
  expect_equal(tab$F, ref$F, tolerance = 1e-8)
  expect_equal(tab[["Pr(>F)"]], ref[["Pr(>F)"]], tolerance = 1e-8)
  expect_true("Reference: F(3, 315)." %in% attr(tab, "heading"))
  # A residual df of 10000 or more is written out, never as 1e+04.
  set.seed(326)
  big <- data.frame(x = rnorm(10010), z = rnorm(10010))
  big$y <- big$x + rnorm(10010)
  h <- attr(anova(fastglmm(y ~ x, big), fastglmm(y ~ x + z, big)), "heading")
  expect_true("Reference: F(1, 10007)." %in% h)
})

test_that("two Gamma GLMs with an estimated dispersion get the Wald F and car's columns", {
  d <- anova_data()
  set.seed(327)
  d$yy <- rgamma(320, shape = 3, rate = 3 / exp(0.2 + 0.4 * d$x))
  tab <- anova(fastglmm(yy ~ x, d, family = Gamma(link = "log")),
               fastglmm(yy ~ x + f, d, family = Gamma(link = "log")))
  expect_identical(names(tab), c("Res.Df", "Df", "F", "Pr(>F)"))
  expect_identical(tab$Res.Df, c(318, 316))
  h <- attr(tab, "heading")
  expect_identical(h[1L], "Model comparison: F test")
  expect_true("Reference: F(2, 316)." %in% h)
})

test_that("a df in a reference line has no padding and no scientific notation", {
  df <- c(2.5, 0.5, 9.99995, 999.95, 5.00004, 12.3456, 3, 10007, 12345.6)
  expect_identical(vapply(df, .fmt_df, ""),
                   c("2.5", "0.5", "10", "1000", "5", "12.35", "3", "10007", "12346"))
  # Not finite, as R prints it (the Python port's fmt_df agrees).
  expect_identical(vapply(c(Inf, -Inf, NaN, NA), .fmt_df, ""), c("Inf", "-Inf", "NA", "NA"))
})

test_that("broom::tidy reads the term table and the comparison table", {
  skip_if_not_installed("broom")
  d <- anova_data()
  m <- fastglmm(y ~ x * f + (1 | g), d)
  # broom names the columns it does not know (NumDF, DenDF) in a warning.
  td <- suppressWarnings(broom::tidy(anova(m)))
  expect_identical(td$term, c("x", "f", "x:f"))
  expect_identical(td$statistic, anova(m)[["F value"]])
  expect_identical(td$p.value, anova(m)[["Pr(>F)"]])
  m0 <- fastglmm(y ~ x + (1 | g), d, REML = FALSE)
  m1 <- fastglmm(y ~ x + f + (1 | g), d, REML = FALSE)
  tab <- anova(m0, m1)
  tc <- broom::tidy(tab)
  expect_identical(tc$term, c("m0", "m1"))
  expect_identical(tc$statistic, tab$Chisq)
  expect_identical(tc$p.value, tab[["Pr(>Chisq)"]])
})

test_that("compare() fits both formulas and returns the table and the two fits", {
  d <- anova_data()
  cmp <- compare(d, y ~ x + (1 | g), y ~ x + f + (1 | g))
  expect_named(cmp, c("table", "fit0", "fit1"))
  expect_s3_class(cmp$fit0, "fastglmm")
  expect_identical(class(cmp$table), c("anova", "data.frame"))
  expect_identical(rownames(cmp$table), c("f0", "f1"))
  # The fixed part differs: both fitted by ML, as fastglmm() fits them.
  expect_false(cmp$fit0$reml || cmp$fit1$reml)
  m0 <- fastglmm(y ~ x + (1 | g), d, REML = FALSE)
  m1 <- fastglmm(y ~ x + f + (1 | g), d, REML = FALSE)
  expect_identical(cmp$fit1$beta, m1$beta)
  expect_identical(columns(cmp$table), columns(anova(m0, m1)))
  # Each fit keeps the columns of its own formula only, as fastglmm() does.
  expect_identical(model.frame(cmp$fit0), model.frame(m0))
  expect_identical(model.frame(cmp$fit1), model.frame(m1))
  # The bigger formula first: fit0 is still the first formula's fit.
  back <- compare(d, y ~ x + f + (1 | g), y ~ x + (1 | g))
  expect_identical(rownames(back$table), c("f1", "f0"))
  expect_identical(back$fit0$formula, "y ~ x + f + (1 | g)")
  expect_output(print(cmp), "f1: y ~ x + f + (1 | g)", fixed = TRUE)
  # compare() says why a Gaussian pair was fitted by ML; anova() does not.
  ml_note <- "Both models were fitted by ML because their fixed parts differ."
  expect_true(ml_note %in% attr(cmp$table, "heading"))
  expect_false(ml_note %in% attr(anova(cmp$fit0, cmp$fit1), "heading"))
  # Only on a Gaussian likelihood-ratio pair: not on an F pair, not on
  # another family (the random-only pair is in the exact-RLRT test).
  expect_false(ml_note %in% attr(compare(d, y ~ x, y ~ x + z)$table, "heading"))
  b <- benchmark_data(seed = 322, family = "binomial")
  expect_false(ml_note %in% attr(compare(b, y ~ t + (1 | g), y ~ t + d + (1 | g),
                                         family = binomial())$table, "heading"))
})

test_that("the exact RLRT p-value is the same from anova(m0, m1) and compare(), on every run", {
  d <- anova_data()
  o <- fastglmm(y ~ x, d, REML = TRUE)
  l <- fastglmm(y ~ x + (1 | g), d)
  set.seed(1)
  a <- anova(o, l)
  set.seed(2)
  b <- anova(o, l)
  cmp <- compare(d, y ~ x, y ~ x + (1 | g))
  expect_identical(a, b)
  expect_identical(a[["Pr(>Chisq)"]], cmp$table[["Pr(>Chisq)"]])
  expect_true(is.na(a$Df[2L]))
  expect_true(any(grepl("^Reference: exact RLRT .*, 10000 draws[.]$",
                        attr(a, "heading"))))
  # Only the random part differs: both are REML fits, with no ML note.
  expect_false("Both models were fitted by ML because their fixed parts differ." %in%
                 attr(cmp$table, "heading"))
  # nsim and seed are arguments, bounded and stated in the error.
  a2 <- anova(o, l, nsim = 2000, seed = 7)
  expect_identical(attr(a2, "nsim"), 2000)
  expect_identical(attr(a2, "seed"), 7)
  expect_identical(a2[["Pr(>Chisq)"]],
                   compare(d, y ~ x, y ~ x + (1 | g), nsim = 2000, seed = 7)$table[["Pr(>Chisq)"]])
  expect_error(anova(o, l, nsim = 0), "nsim must be a whole number from 1 to 1000000, got 0",
               fixed = TRUE)
  expect_error(compare(d, y ~ x, y ~ x + (1 | g), nsim = 1e6 + 1),
               "nsim must be a whole number from 1 to 1000000", fixed = TRUE)
  expect_error(anova(o, l, nsim = 2.5), "nsim must be NULL or one whole number", fixed = TRUE)
  expect_error(anova(o, l, seed = -1), "seed must be NULL or one whole number", fixed = TRUE)
  # A seed is a double in R, exact up to 2^53: a larger or fractional one is refused.
  expect_identical(attr(anova(o, l, nsim = 100, seed = 2^53), "seed"), 2^53)
  expect_error(anova(o, l, seed = 2^53 + 2),
               "seed must be NULL or one whole number from 0 to 2^53", fixed = TRUE)
  expect_error(compare(d, y ~ x, y ~ x + (1 | g), seed = 7.5),
               "seed must be NULL or one whole number from 0 to 2^53", fixed = TRUE)
})

test_that("an exact-RLRT row at a zero statistic prints no draw count", {
  d <- anova_data()
  # Residuals with every group mean at zero: the variance of g is estimated at 0.
  e <- rnorm(320, sd = 0.6)
  d$y <- 1 + 0.5 * d$x + e - ave(e, d$g)
  o <- fastglmm(y ~ x, d, REML = TRUE)
  l <- suppressWarnings(fastglmm(y ~ x + (1 | g), d))
  tab <- anova(o, l)
  expect_identical(tab[["Pr(>Chisq)"]][2L], 1)
  h <- attr(tab, "heading")
  expect_true(any(startsWith(h, "Reference: exact RLRT")))
  expect_false(any(grepl("[0-9] draws", h)))
  expect_true(any(grepl("no draws needed", h, fixed = TRUE)))
})

test_that("anova() and compare() take two models, never three", {
  d <- anova_data()
  m0 <- fastglmm(y ~ x, d)
  m1 <- fastglmm(y ~ x + z, d)
  m2 <- fastglmm(y ~ x + z + f, d)
  expect_error(anova(m0, m1, m2),
               "anova() takes one fit (the per-term table) or two fits (a comparison), not 3.",
               fixed = TRUE)
  expect_error(compare(d, y ~ x, y ~ x + z, y ~ x + z + f),
               "compare() takes two formulas, not three or more.", fixed = TRUE)
  # compare() takes data and formulas, never fits.
  expect_error(compare(m0, m1), "anova(m0, m1)", fixed = TRUE)
  expect_error(compare(d, m0, m1), "anova(m0, m1)", fixed = TRUE)
  # Arguments that do not apply are errors, not silently dropped.
  expect_error(anova(m0, type = "II", nsim = 10), "nsim and seed", fixed = TRUE)
  expect_error(anova(m0, m1, type = "II"), "type", fixed = TRUE)
  expect_error(anova(m0, m1, refit = FALSE), "unused argument 'refit'", fixed = TRUE)
})

test_that("every refusal of anova(m0, m1) says why, and names compare() where it helps", {
  d <- anova_data()
  reml0 <- fastglmm(y ~ x + (1 | g), d)
  reml1 <- fastglmm(y ~ x + z + (1 | g), d)
  ml1 <- fastglmm(y ~ x + z + (1 | g), d, REML = FALSE)
  o0 <- fastglmm(y ~ x, d)

  expect_error(anova(reml0, reml1), paste(
    "the two models are REML fits with different fixed effects; the REML likelihood",
    "depends on the fixed effects, so a comparison needs ML fits. Fit both models by",
    "maximum likelihood (ML). Pass REML = FALSE to both fits, or use compare(data, f0,",
    "f1), which fits both by ML."), fixed = TRUE)
  expect_error(anova(reml0, ml1), paste(
    "the first model was fitted by REML and the second by ML; a comparison needs both",
    "fitted the same way. Fit both models by maximum likelihood (ML), or both by REML",
    "when their fixed parts are the same. Or use compare(data, f0, f1), which picks the",
    "estimator itself."), fixed = TRUE)
  expect_error(anova(ml1, reml0), "the first model was fitted by ML and the second by REML;",
               fixed = TRUE)

  expect_error(anova(o0, fastglmm(y ~ x + z, d[-1, ])), paste(
    "the first model has 320 rows and the second has 319; a comparison needs both fits on",
    "the same rows. Fit both models to the same rows, dropping the rows with a missing",
    "value in any variable of either formula first. Or use compare(data, f0, f1), which",
    "drops those rows once for both fits."), fixed = TRUE)
  d2 <- d
  d2$y[1L] <- d2$y[1L] + 1
  expect_error(anova(o0, fastglmm(y ~ x + z, d2)), paste(
    "the two models have different responses y (the same row count, different values);",
    "a comparison needs both fits on the same data. Fit both models to the same data.",
    "Or use compare(data, f0, f1), which fits both to the same data."), fixed = TRUE)
  expect_error(anova(o0, fastglmm(y ~ x + z, d, weights = rep(2, 320))), paste(
    "the two models were fitted with different weights; a comparison needs the same",
    "weights in both. Fit both models with the same weights. Or use compare(data, f0,",
    "f1), which takes the weights once."), fixed = TRUE)
  expect_error(anova(o0, fastglmm(y ~ x + z, d, offset = d$z)), paste(
    "the two models were fitted with different offsets; a comparison needs the same",
    "offset in both. Fit both models with the same offset. Or use compare(data, f0,",
    "f1), which takes the offset once."), fixed = TRUE)
  b <- benchmark_data(seed = 323, family = "binomial")
  expect_error(anova(fastglmm(y ~ t, b, family = binomial()),
                     fastglmm(y ~ t + d, b, family = poisson())), paste(
    "the first model was fitted as binomial with logit link and the second as poisson",
    "with log link; a comparison needs the same family and link in both. Fit both models",
    "with the same family and link. Or use compare(data, f0, f1), which takes the family",
    "and link once."), fixed = TRUE)

  # The refusals that need no pointer carry the crate's text unchanged.
  bad <- reml1
  bad$converged <- FALSE
  expect_error(anova(reml0, bad),
               "the second model did not converge, so its log-likelihood is", fixed = TRUE)
  set.seed(324)
  gd <- data.frame(x = rnorm(200))
  gd$y <- rgamma(200, shape = 3, rate = 3 / exp(0.2 + 0.4 * gd$x))
  gd$z <- rnorm(200)
  expect_error(anova(fastglmm(y ~ x, gd, family = "gamma", dispersion = 0.5),
                     fastglmm(y ~ x + z, gd, family = "gamma")),
               paste("the first model holds the dispersion at 0.5 and the second does not",
                     "hold the dispersion"),
               fixed = TRUE)
  d$y2 <- d$y
  expect_error(anova(o0, fastglmm(y2 ~ x + z, d)),
               "the two formulas have different responses (y and y2)", fixed = TRUE)
  expect_error(anova(fastglmm(y ~ x + (1 | g), d, REML = FALSE),
                     fastglmm(y ~ z + (1 | g), d, REML = FALSE)),
               "neither formula is nested in the other", fixed = TRUE)
  # The crate's text as it is, with nothing appended.
  expect_identical(tryCatch(anova(o0, o0), error = conditionMessage),
                   paste("the two models have the same fixed and random parts, so there is",
                         "nothing to test. Compare two models that differ by at least one",
                         "parameter."))
  broken <- o0
  broken$formula <- "y ~ (x"
  expect_error(anova(broken, fastglmm(y ~ x + z, d)), "the first model's formula does not parse",
               fixed = TRUE)
})

test_that("anova() on fits restored with readRDS gives the same tables", {
  d <- anova_data()
  small <- fastglmm(y ~ x, d, REML = TRUE)
  big <- fastglmm(y ~ x * f + (1 | g), d)
  path <- tempfile(fileext = ".rds")
  saveRDS(list(small, big), path)
  back <- readRDS(path)
  expect_identical(anova(back[[2L]]), anova(big))
  expect_identical(anova(back[[2L]], type = "II"), anova(big, type = "II"))
  l <- fastglmm(y ~ x + (1 | g), d)
  saveRDS(l, path)
  expect_identical(anova(small, readRDS(path)), anova(small, l))
})

test_that("anova(m0, m1) does not read the model frame", {
  d <- anova_data()
  m0 <- fastglmm(y ~ x + (1 | g), d, REML = FALSE)
  m1 <- fastglmm(y ~ x + f + (1 | g), d, REML = FALSE)
  tab <- anova(m0, m1)
  m0$frame$y <- 0
  m1$frame <- NULL
  expect_identical(anova(m0, m1), tab)
})

test_that("compare() drops a row missing in either formula from both fits", {
  d <- anova_data()
  d$z[c(3L, 7L)] <- NA
  expect_warning(
    cmp <- compare(d, y ~ x + (1 | g), y ~ x + z + (1 | g)),
    paste("Dropped 2 of 320 row(s): a column either formula uses, the weights or the offset",
          "had a missing value there."),
    fixed = TRUE, class = "fastglmm_rows_dropped_na")
  expect_identical(cmp$fit0$nobs, 318L)
  expect_identical(cmp$fit1$nobs, 318L)
  expect_identical(cmp$fit0$y, cmp$fit1$y)
  # The same as two fits on the complete rows.
  full <- d[complete.cases(d), ]
  expect_identical(cmp$fit0$beta, fastglmm(y ~ x + (1 | g), full, REML = FALSE)$beta)
  # A missing weight or offset drops its row from both fits too.
  w <- rep(1, 320)
  w[5L] <- NA
  expect_warning(cmp_w <- compare(d, y ~ x + (1 | g), y ~ x + z + (1 | g), weights = w),
                 class = "fastglmm_rows_dropped_na")
  expect_identical(cmp_w$fit0$nobs, 317L)
  expect_identical(cmp_w$fit1$nobs, 317L)
  expect_identical(names(model.frame(cmp_w$fit0)), c("y", "x", "g", "(weights)"))
  d$z <- NA_real_
  expect_error(compare(d, y ~ x + (1 | g), y ~ x + z + (1 | g)),
               paste("every row has a missing value in a column either formula uses, the",
                     "weights or the offset; no rows left to fit"), fixed = TRUE)
})

test_that("the fit stores the offset, the objective and both per-term results", {
  d <- anova_data()
  m <- fastglmm(y ~ x + (1 | g), d, offset = d$z)
  expect_identical(m$offset, d$z)
  expect_identical(m$objective, "reml")
  expect_identical(m$objective_nodes, 0L)
  expect_null(fastglmm(y ~ x, d)$offset)
  expect_identical(m$term_table_type3$ss_type, "III")
  expect_identical(m$term_table_type2$ss_type, "II")
  b <- benchmark_data(seed = 325, family = "binomial")
  g5 <- fastglmm(y ~ t + (1 | g), b, family = binomial(), nAGQ = 5)
  expect_identical(c(g5$objective, as.character(g5$objective_nodes)), c("agq", "5"))
})

# anova(m, test): "auto" and "Wald" read the stored Wald tables, "LRT" the
# likelihood-ratio tables a fit stores when fastglmm(..., anova_test = "LRT")
# asks for them.
lrt_fits <- function() {
  d <- benchmark_data(seed = 321, family = "binomial")
  list(wald = fastglmm(y ~ t * d + (1 | g), d, family = binomial()),
       lrt = fastglmm(y ~ t * d + (1 | g), d, family = binomial(), anova_test = "LRT"))
}

test_that("test picks the stored table: auto and Wald the Wald table, LRT the likelihood-ratio one", {
  fits <- lrt_fits()
  m <- fits$wald
  m_lrt <- fits$lrt
  # The default stores no likelihood-ratio table, and asking moves nothing
  # of the fit.
  expect_null(m$term_table_type3_lrt)
  expect_null(m$term_table_type2_lrt)
  expect_identical(m$term_test_auto, list(test = "chisq", reason = "binomial_mixed"))
  expect_identical(m_lrt$beta, m$beta)
  expect_identical(m_lrt$vcov, m$vcov)
  expect_identical(m_lrt$loglik, m$loglik)
  for (type in c("III", "II")) {
    wald <- anova(m, type = type, test = "Wald")
    auto <- anova(m, type = type)
    expect_identical(anova(m, type = type, test = "auto"), auto)
    expect_identical(anova(m_lrt, type = type), auto)
    expect_identical(anova(m_lrt, type = type, test = "Wald"), wald)
    # auto gives the Wald numbers under a heading that names auto's reason.
    expect_identical(columns(auto), columns(wald))
    expect_identical(rownames(auto), rownames(wald))
    expect_identical(attr(auto, "heading"), c(
      sprintf("Type %s Analysis of Deviance Table", type),
      'Wald chi-squared tests; test = "auto": binomial model with random effects'))
    expect_identical(attr(wald, "heading"),
                     sprintf("Type %s Analysis of Deviance Table (Wald chi-squared tests)", type))
    lrt <- anova(m_lrt, type = type, test = "LRT")
    stored <- m_lrt[[if (type == "III") "term_table_type3_lrt" else "term_table_type2_lrt"]]
    expect_identical(class(lrt), c("anova", "data.frame"))
    expect_identical(names(lrt), c("Chisq", "Df", "Pr(>Chisq)"))
    expect_identical(rownames(lrt), c("t", "d", "t:d"))
    expect_identical(stored$kind, rep("lrt", 3L))
    expect_identical(lrt$Chisq, stored$statistic)
    expect_identical(lrt$Df, stored$num_df)
    expect_identical(lrt[["Pr(>Chisq)"]], stored$p_value)
    expect_false(isTRUE(all.equal(lrt$Chisq, wald$Chisq)))
  }
  expect_error(anova(m, test = "lrt"), "'arg' should be one of")
  d <- benchmark_data(seed = 321, family = "binomial")
  expect_error(fastglmm(y ~ t + (1 | g), d, family = binomial(), anova_test = "lrt"),
               "'arg' should be one of")
})

test_that("test comes after type, nsim and seed stay, and test applies to one fit only", {
  args <- formals(getS3method("anova", "fastglmm"))
  expect_identical(names(args), c("object", "...", "type", "test", "nsim", "seed"))
  expect_identical(eval(args$test), c("auto", "Wald", "LRT"))
  expect_identical(eval(formals(fastglmm)$anova_test), c("auto", "Wald", "LRT"))
  d <- anova_data()
  m0 <- fastglmm(y ~ x + (1 | g), d)
  m1 <- fastglmm(y ~ x + (1 + x | g), d)
  two <- "test applies to one fit's per-term table, not to a comparison of two fits."
  expect_error(anova(m0, m1, test = "Wald"), two, fixed = TRUE)
  expect_error(anova(m0, m1, test = "auto"), two, fixed = TRUE)
  expect_error(anova(m0, test = "auto", nsim = 10), "nsim and seed", fixed = TRUE)
})

test_that("LRT on a fit that stored no likelihood-ratio table names the fit argument", {
  m <- lrt_fits()$wald
  msg <- paste("this fit stored no likelihood-ratio tables, which are built only when the fit",
               'asks for them: refit with fastglmm(..., anova_test = "LRT").')
  expect_error(anova(m, test = "LRT"), msg, fixed = TRUE)
  expect_error(anova(m, type = "II", test = "LRT"), msg, fixed = TRUE)
})

test_that("the heading names the likelihood-ratio test and why; Wald keeps its heading", {
  m_lrt <- lrt_fits()$lrt
  expect_identical(attr(anova(m_lrt, test = "LRT"), "heading"), c(
    "Type III Analysis of Deviance Table",
    paste("Likelihood-ratio chi-squared tests (each term's model refitted without it)",
          'on the reduced design; test = "LRT"')))
  expect_identical(attr(anova(m_lrt, type = "II", test = "LRT"), "heading"), c(
    "Type II Analysis of Deviance Table",
    paste("Likelihood-ratio chi-squared tests (each term's model refitted without it);",
          'test = "LRT"')))
  expect_identical(attr(anova(m_lrt, test = "Wald"), "heading"),
                   "Type III Analysis of Deviance Table (Wald chi-squared tests)")
  # A fit that did not converge says so first, then the table's own notes.
  m_lrt$converged <- FALSE
  expect_identical(attr(anova(m_lrt, test = "LRT"), "heading")[3L], paste(
    "The fit did not converge, so its estimates are not at a maximum and these tests",
    "are not reliable. Change the model until it converges (a simpler random part often",
    "helps), then test again."))
  # A stored refusal is an error with its own text.
  m_lrt$term_table_type2_lrt <- list(error = "internal error while building the per-term table")
  expect_error(anova(m_lrt, type = "II", test = "LRT"),
               "internal error while building the per-term table", fixed = TRUE)
})

test_that("Wald or LRT on a fit with F rows is an error naming its F test; the fit stands", {
  d <- anova_data()
  lmm <- fastglmm(y ~ x * f + (1 | g), d, anova_test = "LRT")
  expect_true(lmm$converged)
  satt <- "per-term tests on a Gaussian mixed model are F tests; "
  advice <- "rows are not available, so leave the test at its default."
  expect_error(anova(lmm, test = "Wald"), paste0(satt, "Wald chi-squared ", advice),
               fixed = TRUE)
  expect_error(anova(lmm, type = "II", test = "LRT"),
               paste0(satt, "likelihood-ratio ", advice), fixed = TRUE)
  # The same text the crate stored when the fit asked.
  expect_identical(lmm$term_table_type3_lrt$error, paste0(satt, "likelihood-ratio ", advice))
  expect_identical(anova(lmm, test = "auto"), anova(fastglmm(y ~ x * f + (1 | g), d)))
  ols <- fastglmm(y ~ x + f, d)
  expect_error(anova(ols, test = "LRT"), paste0(
    "per-term tests on this fit are F tests on the residual df; ",
    "likelihood-ratio ", advice), fixed = TRUE)
})

# A Poisson GLMM, whose "auto" is the likelihood-ratio test, fitted with each
# anova_test.
pois_fits <- function() {
  d <- benchmark_data(seed = 323, family = "poisson")
  lapply(c(auto = "auto", Wald = "Wald", LRT = "LRT"), function(a) {
    fastglmm(y ~ t * d + (1 | g), d, family = poisson(), anova_test = a)
  })
}

test_that("auto gives a Poisson GLMM the likelihood-ratio table, under its own heading", {
  fits <- pois_fits()
  reason <- 'test = "auto": Poisson model with random effects'
  lr_line <- "Likelihood-ratio chi-squared tests (each term's model refitted without it)"
  for (m in fits) {
    expect_identical(m$term_test_auto, list(test = "lrt", reason = "poisson_mixed"))
  }
  expect_identical(fits$auto$term_table_type3_lrt$reason, "poisson_mixed")
  expect_null(fits$Wald$term_table_type3_lrt)
  for (type in c("III", "II")) {
    design <- if (type == "III") " on the reduced design" else ""
    for (m in fits[c("auto", "LRT")]) {
      auto <- anova(m, type = type)
      lrt <- anova(m, type = type, test = "LRT")
      expect_identical(anova(m, type = type, test = "auto"), auto)
      expect_identical(attr(auto, "heading"), c(
        sprintf("Type %s Analysis of Deviance Table", type),
        paste0(lr_line, design, "; ", reason)))
      expect_identical(attr(lrt, "heading")[2L], paste0(lr_line, design, '; test = "LRT"'))
      expect_identical(columns(auto), columns(lrt))
      expect_identical(rownames(auto), c("t", "d", "t:d"))
    }
    # The Wald table is the same whatever the fit stored.
    wald <- anova(fits$Wald, type = type, test = "Wald")
    expect_identical(attr(wald, "heading"),
                     sprintf("Type %s Analysis of Deviance Table (Wald chi-squared tests)", type))
    expect_identical(anova(fits$auto, type = type, test = "Wald"), wald)
    expect_identical(anova(fits$LRT, type = type, test = "Wald"), wald)
  }
})

test_that("auto on a Poisson GLMM fitted with anova_test = \"Wald\" says so", {
  m <- pois_fits()$Wald
  msg <- paste('test = "auto" gives likelihood-ratio tests on a Poisson model with random',
               'effects, and this fit stored none: it was made with anova_test = "Wald".',
               'Use test = "Wald", or refit with fastglmm(..., anova_test = "auto").')
  expect_error(anova(m), msg, fixed = TRUE)
  expect_error(anova(m, type = "II"), msg, fixed = TRUE)
  # compare() makes its two fits as anova_test = "Wald" makes them.
  d <- benchmark_data(seed = 323, family = "poisson")
  cmp <- compare(d, y ~ t + d + (1 | g), y ~ t * d + (1 | g), family = poisson())
  expect_error(anova(cmp$fit1), msg, fixed = TRUE)
  expect_error(anova(cmp$fit1, type = "II"), msg, fixed = TRUE)
  for (type in c("III", "II")) {
    expect_identical(anova(cmp$fit1, type = type, test = "Wald"),
                     anova(m, type = type, test = "Wald"))
  }
  glm <- fastglmm(y ~ t * d, d, family = poisson())
  expect_identical(glm$term_test_auto, list(test = "chisq", reason = "no_random_effects"))
  expect_null(glm$term_table_type3_lrt)
  expect_identical(attr(anova(glm), "heading"), c(
    "Type III Analysis of Deviance Table",
    'Wald chi-squared tests; test = "auto": model without random effects'))
})

test_that("auto names the Wald test and the reason on every other chi-squared fit", {
  d <- benchmark_data(seed = 323, family = "poisson")
  d$pos <- exp(0.3 * d$t + 0.2 * d$d + rnorm(nrow(d), sd = 0.3))
  other <- "Gamma model with random effects"
  cases <- list(
    list(fastglmm(y ~ t * d + (1 | g), d, family = "negativebinomial"),
         "negative_binomial_mixed", "negative binomial model with random effects"),
    list(fastglmm(pos ~ t * d + (1 | g), d, family = Gamma(link = "log")), "other_mixed", other))
  for (k in cases) {
    m <- k[[1L]]
    expect_identical(m$term_test_auto, list(test = "chisq", reason = k[[2L]]))
    expect_null(m$term_table_type3_lrt)
    for (type in c("III", "II")) {
      auto <- anova(m, type = type)
      wald <- anova(m, type = type, test = "Wald")
      expect_identical(attr(auto, "heading"), c(
        sprintf("Type %s Analysis of Deviance Table", type),
        paste0('Wald chi-squared tests; test = "auto": ', k[[3L]])))
      expect_identical(columns(auto), columns(wald))
    }
  }
  # A fit made before the auto choice was stored refuses only "auto".
  old <- cases[[1L]][[1L]]
  old$term_test_auto <- NULL
  expect_identical(anova(old, test = "Wald"), anova(cases[[1L]][[1L]], test = "Wald"))
  expect_error(anova(old), "this fit has no term_test_auto", fixed = TRUE)
})

# contrast_test(m, L): one joint test of a fit's coefficients, from what the
# fit stores.

contrast_data <- function() {
  d <- anova_data()
  set.seed(312)
  d$n <- rpois(nrow(d), exp(0.2 + 0.3 * d$x + 0.3 * as.integer(d$g) %% 3))
  d
}

unit_rows <- function(m, cols) {
  L <- matrix(0, length(cols), length(fixef(m)))
  L[cbind(seq_along(cols), match(cols, names(fixef(m))))] <- 1
  L
}

test_that("contrast_test() of a term's unit rows is its anova row, bit for bit", {
  # No interaction, so the Type III hypothesis of each term is its unit rows:
  # the row read back from the stored fields is the kernel's own row.
  d <- contrast_data()
  cases <- list(
    list(fastglmm(y ~ x + f + (1 | g), d), "auto",
         "Analysis of Variance Table with Satterthwaite's method"),
    list(fastglmm(y ~ x + f, d), "auto", "Analysis of Variance Table"),
    list(fastglmm(n ~ x + f + (1 | g), d, family = poisson()), "Wald",
         "Analysis of Deviance Table (Wald chi-squared tests)"))
  for (case in cases) {
    m <- case[[1L]]
    tab <- contrast_test(m, unit_rows(m, c("fb", "fc")))
    term <- anova(m, test = case[[2L]])
    expect_identical(attr(tab, "heading"), case[[3L]])
    expect_identical(rownames(tab), "contrast")
    expect_identical(names(tab), names(term))
    expect_identical(columns(tab), columns(term["f", ]))
    # An L given as a vector is a one-row L.
    expect_identical(columns(contrast_test(m, unit_rows(m, "x")[1, ])), columns(term["x", ]))
  }
  expect_identical(class(tab), c("anova", "data.frame"))
  # The record survives saveRDS().
  m <- cases[[1L]][[1L]]
  path <- tempfile(fileext = ".rds")
  saveRDS(m, path)
  expect_identical(contrast_test(readRDS(path), unit_rows(m, c("fb", "fc"))),
                   contrast_test(m, unit_rows(m, c("fb", "fc"))))
  unlink(path)
})

test_that("contrast_test() gives an LMM with no Satterthwaite pass the NA F row", {
  d <- contrast_data()
  m <- fastglmm(y ~ x + f + (1 | g), d, satterthwaite = FALSE)
  tab <- contrast_test(m, unit_rows(m, "x"))
  expect_identical(attr(tab, "heading"), c("Analysis of Variance Table", paste(
    "contrast: No denominator df (Satterthwaite degrees of freedom were not computed for",
    "this fit), so the hypothesis was not tested.")))
  expect_true(all(is.na(unlist(tab))))
})

test_that("contrast_test() on an aliased coefficient of a Satterthwaite LMM", {
  d <- contrast_data()
  d$x2 <- 2 * d$x
  a <- fastglmm(y ~ x + x2 + f + (1 | g), d)
  expect_identical(a$df_method, "satterthwaite")
  expect_identical(unname(a$aliased), c(FALSE, FALSE, TRUE, FALSE, FALSE))
  expect_identical(a$satterthwaite_record$retained, c(1L, 2L, 4L, 5L))
  expect_identical(columns(contrast_test(a, unit_rows(a, c("fb", "fc")))),
                   columns(anova(a)["f", ]))
  expect_identical(attr(contrast_test(a, unit_rows(a, c("x", "x2"))), "heading")[2L],
                   "contrast: The coefficient x2 is aliased, so the hypothesis was not tested.")
})

test_that("contrast_test() runs on a fit stored by 0.6.1 unless it is a Satterthwaite LMM", {
  d <- contrast_data()
  m <- fastglmm(y ~ x + f, d)
  old <- m
  old$satterthwaite_record <- NULL
  expect_identical(contrast_test(old, unit_rows(m, c("fb", "fc"))),
                   contrast_test(m, unit_rows(m, c("fb", "fc"))))
  m <- fastglmm(y ~ x + f + (1 | g), d)
  old <- m
  old$satterthwaite_record <- NULL
  expect_error(contrast_test(old, unit_rows(m, "x")), "this fit has no satterthwaite_record",
               fixed = TRUE)
})

test_that("contrast_test() refuses an L of the wrong shape or not finite", {
  d <- contrast_data()
  m <- fastglmm(y ~ x + f + (1 | g), d)
  p <- length(fixef(m))
  expect_error(contrast_test(m, numeric(p + 1L)), sprintf("L must have %d columns", p),
               fixed = TRUE)
  expect_error(contrast_test(m, matrix(0, 2, p - 1L)), sprintf("L must have %d columns", p),
               fixed = TRUE)
  expect_error(contrast_test(m, matrix(0, 0, p)), "L must have at least one row", fixed = TRUE)
  expect_error(contrast_test(m, array(0, c(1, 1, p))), "L must be a numeric matrix or vector",
               fixed = TRUE)
  expect_error(contrast_test(m, letters[seq_len(p)]), "L must be a numeric matrix or vector",
               fixed = TRUE)
  expect_error(contrast_test(m, rep(NA_real_, p)), "L must be finite numbers", fixed = TRUE)
  expect_error(contrast_test(m, rep(Inf, p)), "L must be finite numbers, got inf", fixed = TRUE)
  expect_error(contrast_test(list(), numeric(p)), "contrast_test() takes a fastglmm fit",
               fixed = TRUE)
})
