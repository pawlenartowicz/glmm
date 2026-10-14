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
    "are not reliable. Refit until the fit converges, then test again."))
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
  expect_identical(attr(tab, "heading")[1L],
                   "Type III Analysis of Deviance Table (Wald \u03c7\u00b2 tests)")
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
    "m0 and m1 are REML fits with different fixed effects, and REML fits with different",
    "fixed effects cannot be compared by likelihood ratio. Use compare(data, f0, f1),",
    "which fits both models by ML, or refit both with REML = FALSE."), fixed = TRUE)
  expect_error(anova(reml0, ml1), paste(
    "m0 was fitted by REML and m1 by ML. Fit both the same way, or use",
    "compare(data, f0, f1)."), fixed = TRUE)
  expect_error(anova(ml1, reml0), "m0 was fitted by ML and m1 by REML.", fixed = TRUE)

  expect_error(anova(o0, fastglmm(y ~ x + z, d[-1, ])), paste(
    "m0 has 320 rows and m1 has 319; a comparison needs both fits on the same rows.",
    "Fit both models to the same rows, dropping the rows with a missing value in any",
    "variable of either formula first. compare(data, f0, f1) does that and fits both",
    "models in one call."), fixed = TRUE)
  d2 <- d
  d2$y[1L] <- d2$y[1L] + 1
  expect_error(anova(o0, fastglmm(y ~ x + z, d2)), paste(
    "m0 and m1 have different responses y (the same row count, different values); a",
    "comparison needs both fits on the same data. Fit both models to the same data.",
    "compare(data, f0, f1) fits both models to one data set."), fixed = TRUE)
  expect_error(anova(o0, fastglmm(y ~ x + z, d, weights = rep(2, 320))), paste(
    "m0 and m1 were fitted with different weights; a comparison needs the same weights",
    "in both. Refit both models with the same weights. compare(data, f0, f1) takes the",
    "weights once, for both models."), fixed = TRUE)
  expect_error(anova(o0, fastglmm(y ~ x + z, d, offset = d$z)), paste(
    "m0 and m1 were fitted with different offsets; a comparison needs the same offset",
    "in both. Refit both models with the same offset. compare(data, f0, f1) takes the",
    "offset once, for both models."), fixed = TRUE)
  b <- benchmark_data(seed = 323, family = "binomial")
  expect_error(anova(fastglmm(y ~ t, b, family = binomial()),
                     fastglmm(y ~ t + d, b, family = poisson())), paste(
    "m0 was fitted as binomial with logit link and m1 as poisson with log link; a",
    "comparison needs the same family and link in both. Refit both models with the",
    "same family and link. compare(data, f0, f1) takes the family once, for both",
    "models."), fixed = TRUE)

  # The refusals that need no pointer carry the crate's text unchanged.
  bad <- reml1
  bad$converged <- FALSE
  expect_error(anova(reml0, bad), "m1 did not converge, so its log-likelihood is",
               fixed = TRUE)
  set.seed(324)
  gd <- data.frame(x = rnorm(200))
  gd$y <- rgamma(200, shape = 3, rate = 3 / exp(0.2 + 0.4 * gd$x))
  gd$z <- rnorm(200)
  expect_error(anova(fastglmm(y ~ x, gd, family = "gamma", dispersion = 0.5),
                     fastglmm(y ~ x + z, gd, family = "gamma")),
               "m0 holds the dispersion at 0.5 and m1 does not hold the dispersion",
               fixed = TRUE)
  d$y2 <- d$y
  expect_error(anova(o0, fastglmm(y2 ~ x + z, d)),
               "the two formulas have different responses (y and y2)", fixed = TRUE)
  expect_error(anova(fastglmm(y ~ x + (1 | g), d, REML = FALSE),
                     fastglmm(y ~ z + (1 | g), d, REML = FALSE)),
               "neither formula is nested in the other", fixed = TRUE)
  # The crate's text as it is, with nothing appended.
  expect_identical(tryCatch(anova(o0, o0), error = conditionMessage),
                   paste("m0 and m1 have the same fixed and random parts, so there is nothing",
                         "to test. Compare two models that differ by at least one parameter."))
  broken <- o0
  broken$formula <- "y ~ (x"
  expect_error(anova(broken, fastglmm(y ~ x + z, d)), "m0's formula does not parse",
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
