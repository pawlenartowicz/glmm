# Data marshalling: the R -> Table trip and the row filtering
# (subset/na.action) that the R side owns.

ols_data <- function() {
  set.seed(7)
  data.frame(
    y = c(1, 2, 2.9, 4.1, 5, 6.2, 6.8, 8.1, 9, 10.2),
    x = 0:9
  )
}

test_that("a factor's declared level order survives to the base level", {
  set.seed(11)
  d <- data.frame(
    y = rnorm(60),
    f = factor(rep(c("high", "low", "mid"), 20), levels = c("mid", "low", "high"))
  )
  fit <- fastglmm(y ~ f, d, family = gaussian())
  # Base = declared first level ("mid"), NOT the lexicographic first ("high").
  expect_named(fixef(fit), c("(Intercept)", "flow", "fhigh"))
  ref <- coef(lm(y ~ f, d)) # lm honors the same declared order
  expect_equal(unname(fixef(fit)), unname(ref), tolerance = 1e-8)
})

test_that("character columns get lexicographic levels (factor() default)", {
  set.seed(12)
  d <- data.frame(
    y = rnorm(60),
    f = rep(c("b", "c", "a"), 20),
    stringsAsFactors = FALSE
  )
  fit <- fastglmm(y ~ f, d, family = gaussian())
  expect_named(fixef(fit), c("(Intercept)", "fb", "fc"))
})

test_that("relabelled and unused factor levels give the same fit", {
  # Declaring the used levels in reverse order and adding two unused ones
  # changes the base level and the coding, not the model.
  i <- 0:39
  u_g <- c(-0.6, 0.4, 0.1, -0.3, 0.7, -0.2, 0.5, -0.4)
  x <- sin(1.7 * i + 0.3)
  f <- c("p", "q", "r")[i %% 3 + 1]
  g <- paste0("g", i %/% 5)
  y <- 2 + 0.5 * x + 0.3 * (f == "q") - 0.4 * (f == "r") + u_g[i %/% 5 + 1] +
    0.4 * sin(2.3 * i + 0.7)
  levels_of <- function(used) {
    r <- rev(used)
    k <- length(r) %/% 2
    c("zz", r[seq_len(k)], "yy", r[-seq_len(k)])
  }
  d_chr <- data.frame(y = y, x = x, f = f, g = g, stringsAsFactors = FALSE)
  d_fac <- data.frame(
    y = y, x = x,
    f = factor(f, levels = levels_of(c("p", "q", "r"))),
    g = factor(g, levels = levels_of(paste0("g", 0:7)))
  )
  ref <- fastglmm(y ~ x + f + (1 | g), d_chr)
  fit <- fastglmm(y ~ x + f + (1 | g), d_fac)
  expect_equal(unname(fitted(fit)), unname(fitted(ref)), tolerance = 1e-5)
  expect_equal(as.numeric(logLik(fit)), as.numeric(logLik(ref)), tolerance = 1e-5)
  sd_g <- function(m) attr(VarCorr(m)$g, "stddev")
  expect_equal(sd_g(fit), sd_g(ref), tolerance = 1e-5)
  expect_identical(names(fixef(fit))[1:4], c("(Intercept)", "x", "fq", "fp"))
})

test_that("na.omit drops NA rows; na.pass-style leftovers error", {
  d <- ols_data()
  d$y[3] <- NA
  expect_warning(fit <- fastglmm(y ~ x, d, na.action = na.omit),
                 "Dropped 1 of 10 row")
  expect_equal(nobs(fit), 9L)
  expect_identical(fit$warnings$tier, "caution")
  expect_identical(fit$warnings$kind, "rows_dropped_na")
  expect_identical(fit$warnings$title, "Rows dropped for missing values")
  expect_identical(fit$warnings$message,
    "Dropped 1 of 10 row(s): a column the formula uses had a missing value there.")
  expect_error(fastglmm(y ~ x, d, na.action = na.pass),
               "missing values remain")
  expect_error(fastglmm(y ~ x, d, na.action = na.fail), "missing values")
})

test_that("every row missing in a used column gives the same message as the Python port", {
  # Mirrors glmm-py/tests/test_validation.py's
  # test_all_rows_missing_in_a_used_column_is_a_plain_error.
  d <- data.frame(y = c(1, 2, 3), x = c(NA_real_, NA_real_, NA_real_))
  expect_error(
    fastglmm(y ~ x, d),
    "every row has a missing value in a column the formula uses; no rows left to fit"
  )
})

test_that("weights are honored; zero/short/negative weights are clean errors", {
  d <- ols_data()
  w <- rep(c(2, 1), 5)
  fit <- fastglmm(y ~ x, d, weights = w)
  ref <- lm(y ~ x, d, weights = w)
  expect_equal(unname(fixef(fit)), unname(coef(ref)), tolerance = 1e-8)
  expect_error(fastglmm(y ~ x, d, weights = c(1, 2)), "one entry")
  expect_error(fastglmm(y ~ x, d, weights = rep(-1, 10)), "positive")
  expect_error(fastglmm(y ~ x, d, weights = c(0, rep(1, 9))), "positive")
})

test_that("an NA or NaN in weights= or offset= is refused as not finite", {
  d <- ols_data()
  for (bad in c(NA, NaN)) {
    v <- rep(1, nrow(d))
    v[3] <- bad
    expect_error(fastglmm(y ~ x, d, weights = v),
                 "weights must be finite: no NaN, inf, or missing entries", fixed = TRUE)
    expect_error(fastglmm(y ~ x, d, offset = v),
                 "offset must be finite: no NaN, inf, or missing entries", fixed = TRUE)
  }
})

test_that("offset= is honored as an expression and stays aligned under subset=", {
  set.seed(21)
  d <- data.frame(x = rnorm(40), exposure = runif(40, 1, 10))
  d$y <- rpois(40, d$exposure * exp(0.3 + 0.5 * d$x))

  fit <- fastglmm(y ~ x, d, family = poisson(), offset = log(exposure))
  ref <- glm(y ~ x, poisson(), d, offset = log(exposure))
  expect_equal(unname(fixef(fit)), unname(coef(ref)), tolerance = 1e-8)

  # The alignment hazard: an offset read off `data` after the row filtering
  # would pair each kept row with the wrong exposure. That fit still converges
  # and still looks reasonable, so only a comparison against the same rows
  # filtered up front catches it.
  keep <- d$x > 0
  sub_fit <- fastglmm(y ~ x, d, family = poisson(), offset = log(exposure),
                      subset = x > 0)
  pre_fit <- fastglmm(y ~ x, d[keep, ], family = poisson(),
                      offset = log(exposure))
  expect_equal(nobs(sub_fit), sum(keep))
  expect_equal(unname(fixef(sub_fit)), unname(fixef(pre_fit)),
               tolerance = 1e-10)

  expect_error(fastglmm(y ~ x, d, family = poisson(), offset = rep(0, 3)),
               "one entry")
})

test_that("missing and unsupported columns are clean errors", {
  expect_error(fastglmm(y ~ z, ols_data()), "not found in data.*z")
  d <- ols_data()
  d$z <- as.Date("2026-01-01") + 0:9
  expect_error(fastglmm(y ~ z, d), "unsupported type")
})

test_that("logical columns are factors with FALSE/TRUE levels, as in lme4", {
  set.seed(31)
  d <- data.frame(g = rep(c(FALSE, TRUE), each = 30), h = factor(rep(1:6, 10)),
                  b = rep(c(TRUE, FALSE), 30), x = rnorm(60))
  d$y <- d$x + 0.5 * d$b + rnorm(6)[d$h] + ifelse(d$g, 0.4, -0.4) + rnorm(60)
  d01 <- transform(d, g = as.integer(g), b = as.integer(b))

  # Grouping: lme4 labels the levels "FALSE"/"TRUE"; the 0/1 coding it
  # replaces fits bit-identically, only the labels change.
  fit <- fastglmm(y ~ x + (1 | g) + (1 | h), d)
  fit01 <- fastglmm(y ~ x + (1 | g) + (1 | h), d01)
  expect_equal(rownames(ranef(fit)$g), c("FALSE", "TRUE"))
  expect_identical(fixef(fit), fixef(fit01))
  expect_identical(unname(unlist(ranef(fit))), unname(unlist(ranef(fit01))))
  expect_identical(as.numeric(logLik(fit)), as.numeric(logLik(fit01)))

  # Fixed effect: the dummy is bTRUE, same numbers as the 0/1 column.
  fit <- fastglmm(y ~ b * x + (1 | h), d)
  fit01 <- fastglmm(y ~ b * x + (1 | h), d01)
  expect_named(fixef(fit), c("(Intercept)", "bTRUE", "x", "bTRUE:x"))
  expect_identical(unname(fixef(fit)), unname(fixef(fit01)))

  # Where R's marginality rule codes the logical by indicators, the design
  # is model.matrix()'s, not a single 0/1 column.
  expect_named(fixef(fastglmm(y ~ x:b + (1 | h), d)),
               colnames(model.matrix(y ~ x:b, d)))
  expect_named(fixef(fastglmm(y ~ 0 + b + (1 | h), d)),
               colnames(model.matrix(y ~ 0 + b, d)))

  # A logical response stays 0/1, as glm() reads it.
  d$yb <- d$y > 0
  fit <- fastglmm(yb ~ x, d, family = binomial())
  ref <- glm(yb ~ x, binomial(), d)
  expect_equal(unname(fixef(fit)), unname(coef(ref)), tolerance = 1e-8)
})

test_that("unused data columns are never marshalled", {
  d <- ols_data()
  d$junk <- replicate(10, list(1)) # unmarshallable, but not in the formula
  expect_silent(fit <- fastglmm(y ~ x, d))
  expect_true(fit$converged)
})

test_that("formula strings are accepted", {
  fit <- fastglmm("y ~ x", ols_data())
  expect_s3_class(formula(fit), "formula")
  expect_equal(deparse(formula(fit)), "y ~ x")
})

frame_data <- function() {
  set.seed(31)
  n <- 40
  data.frame(y = rnorm(n), x = rnorm(n), g = factor(rep(letters[1:8], each = 5)))
}

test_that("model.frame(fit) is the formula's columns after subset= and na.action", {
  d <- frame_data()
  n <- nrow(d)
  vars <- c("y", "x", "g")
  with_na <- d
  with_na$x[3] <- NA
  char_rows <- d
  rownames(char_rows) <- paste0("r", seq_len(n))
  labelled <- d
  attr(labelled$x, "label") <- "the x"
  factor_attr <- d
  attr(factor_attr$g, "note") <- 1
  named_col <- d
  names(named_col$x) <- paste0("n", seq_len(n))
  df_attr <- d
  attr(df_attr, "note") <- "not carried into the frame"
  # `ref` is the data.frame the expected frame is cut from, when `data` is not one.
  cases <- list(
    plain = list(data = d),
    character_row_names = list(data = char_rows),
    permuted_row_names = list(data = d[c(5:n, 1:4), ]),
    # na.omit rebuilds every column with `[`, which drops a plain attribute.
    label_attribute = list(data = labelled),
    factor_with_extra_attribute = list(data = factor_attr),
    named_column = list(data = named_col),
    data_frame_attribute = list(data = df_attr),
    list_input = list(data = as.list(d), ref = d),
    missing_value = list(data = with_na),
    na_exclude = list(data = with_na, na.action = stats::na.exclude),
    na_exclude_complete = list(data = d, na.action = stats::na.exclude),
    subset = list(data = d, subset = seq(1, n, 2))
  )
  for (nm in names(cases)) {
    cs <- cases[[nm]]
    naf <- if (is.null(cs$na.action)) stats::na.omit else cs$na.action
    fit <- suppressWarnings(
      fastglmm(y ~ x + (1 | g), cs$data, na.action = naf, subset = cs$subset)
    )
    expected <- (if (is.null(cs$ref)) cs$data else cs$ref)[vars]
    if (!is.null(cs$subset)) expected <- expected[cs$subset, , drop = FALSE]
    expect_true(identical(model.frame(fit), naf(expected)), label = nm)
  }
})

test_that("a tibble keeps its class in model.frame(fit)", {
  skip_if_not_installed("tibble")
  d <- frame_data()
  tb <- tibble::as_tibble(d)
  fit <- fastglmm(y ~ x + (1 | g), tb)
  expect_identical(class(model.frame(fit)), class(tb))
  expect_identical(fit$beta, fastglmm(y ~ x + (1 | g), d)$beta)
})

test_that("a duplicate column name the formula uses is refused, an unused one is not", {
  d <- frame_data()
  dup_x <- cbind(d, x = rev(d$x))
  msg <- "data has duplicate column name(s): x"
  expect_error(fastglmm(y ~ x + (1 | g), dup_x), msg, fixed = TRUE)
  expect_error(compare(dup_x, y ~ x, y ~ x + (1 | g)), msg, fixed = TRUE)
  dup_z <- cbind(d, z = d$x, z = d$y)
  m <- fastglmm(y ~ x + (1 | g), dup_z)
  expect_identical(names(fixef(m)), c("(Intercept)", "x"))
  expect_error(predict(fastglmm(y ~ x + (1 | g), d), newdata = dup_x), msg, fixed = TRUE)
})

test_that("an NA in a logical subset= drops that row without a warning, as lm does", {
  set.seed(5)
  d <- data.frame(y = rnorm(29), x = rnorm(29))
  s <- c(NA, NA, NA, rep(TRUE, 26))
  expect_no_warning(fit <- fastglmm(y ~ x, d, subset = s))
  expect_equal(nobs(fit), 26)
  expect_identical(fit$.row_index, 4:29)
})

test_that("a data.table keeps its class in model.frame(fit)", {
  skip_if_not_installed("data.table")
  d <- frame_data()
  dt <- data.table::as.data.table(d)
  fit <- fastglmm(y ~ x + (1 | g), dt)
  expect_identical(class(model.frame(fit)), class(dt))
  expect_identical(fit$beta, fastglmm(y ~ x + (1 | g), d)$beta)
})

test_that("dotted column names fit", {
  set.seed(1)
  school.id <- factor(rep(1:10, each = 8)); age.years <- rnorm(80)
  d <- data.frame(y = 1 + 0.5 * age.years + rnorm(10)[school.id] + rnorm(80), age.years, school.id)
  m <- fastglmm(y ~ age.years + (1 | school.id), data = d)
  expect_true("age.years" %in% names(fixef(m)))  # coef() stops by design
})

test_that("columns named like a fit's index fields fit", {
  set.seed(2)
  g <- factor(rep(1:10, each = 8)); x <- rnorm(80)
  d <- data.frame(retained = rbinom(80, 1, plogis(0.5 * x + rnorm(10)[g])),
                  re_term_index = x, null_cluster = g)
  m <- fastglmm(retained ~ re_term_index + (1 | null_cluster), d, family = binomial)
  expect_named(fixef(m), c("(Intercept)", "re_term_index"))
})

test_that("a failed mixed fit has full-length NaN random-effect fields", {
  # n = p = 3 leaves no finite end point: one grouping, two levels, q = 1.
  # Mirrors test_fit_wiring.py's
  # test_failed_mixed_fit_has_full_length_nan_random_effect_fields - change
  # together.
  d <- data.frame(y = c(1, 2, 4), x1 = c(0, 1, 3), x2 = c(1, 0, 2),
                  g = c("a", "b", "a"))
  fit <- suppressWarnings(fastglmm(y ~ x1 + x2 + (1 | g), d))
  expect_false(fit$converged)
  expect_length(fit$varcorr, 1L)
  expect_length(fit$varcorr[[1L]], 1L)
  expect_length(fit$stddev_se, 1L)
  expect_length(fit$.kernel_parts$ranef, 2L)
  expect_length(fitted(fit), 3L)
  for (v in list(fit$varcorr[[1L]], fit$stddev_se, fit$.kernel_parts$ranef, fitted(fit))) {
    expect_true(all(is.nan(v)))
  }
})
