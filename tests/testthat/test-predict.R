# predict(), simulate() and ranef(condVar = TRUE): what crosses the port
# boundary and how it is worded. The numbers themselves are checked in the
# crate (RULE 6).

predict_data <- function() {
  set.seed(20261026)
  k <- 12L
  per <- 10L
  g <- rep(seq_len(k), each = per)
  h <- rep(seq_len(per), times = k)
  x <- rnorm(k * per)
  o <- 0.2 * rnorm(k * per)
  u <- rnorm(k, sd = 0.8)[g]
  v <- rnorm(per, sd = 0.5)[h]
  s <- rnorm(per, sd = 0.4)[h]
  data.frame(
    y = 1 + 0.5 * x + o + u + 0.5 * rnorm(k * per),
    yb = rbinom(k * per, 1L, plogis(-0.2 + 0.8 * x + u)),
    yc = 1 + 0.5 * x + u + v + 0.5 * rnorm(k * per),
    ys = 1 + 0.5 * x + u + s * x + 0.5 * rnorm(k * per),
    x = x, o = o,
    g = sprintf("g%02d", g), h = sprintf("h%d", h)
  )
}

quiet_fit <- function(formula, data, ...) suppressWarnings(fastglmm(formula, data, ...))

test_that("predict on the model frame at the response scale is fitted()", {
  d <- predict_data()
  m <- quiet_fit(y ~ x + (1 | g), d)
  expect_equal(predict(m, type = "response"), fitted(m), tolerance = 1e-10)
  b <- quiet_fit(yb ~ x + (1 | g), d, family = binomial())
  expect_equal(predict(b, d, type = "response"), unname(fitted(b)), tolerance = 1e-10,
               ignore_attr = TRUE)
})

test_that("the default scale is the link", {
  d <- predict_data()
  b <- quiet_fit(yb ~ x + (1 | g), d, family = binomial())
  expect_equal(plogis(predict(b)), fitted(b), tolerance = 1e-10)
  expect_identical(predict(b), predict(b, type = "link"))
})

test_that("re.form = NA is X beta plus the offset", {
  d <- predict_data()
  m <- quiet_fit(y ~ x + offset(o) + (1 | g), d)
  want <- drop(cbind(1, d$x) %*% fixef(m)) + d$o
  expect_equal(unname(predict(m, d, re.form = NA)), want, tolerance = 1e-12)
  expect_equal(unname(predict(m, d[c("x", "o")], re.form = NA)), want, tolerance = 1e-12)
  expect_equal(unname(predict(m, re.form = NA)), want, tolerance = 1e-12)
})

test_that("se.fit returns the fit, its se and the interval", {
  d <- predict_data()
  m <- quiet_fit(y ~ x + (1 | g), d)
  out <- predict(m, d, se.fit = TRUE)
  expect_named(out, c("fit", "se.fit", "lower", "upper"))
  expect_identical(out$fit, predict(m, d))
  expect_true(all(out$se.fit > 0))
  expect_true(all(out$lower < out$fit & out$fit < out$upper))
})

test_that("an unseen grouping level warns with its class and gets the population prediction", {
  d <- predict_data()
  m <- quiet_fit(y ~ x + (1 | g), d)
  new <- d[c(1, 2, 16), ]
  new$g <- c("g01", "zz", "zz")
  w <- tryCatch(predict(m, new), warning = function(w) w)
  expect_s3_class(w, "fastglmm_new_grouping_levels")
  expect_s3_class(w, "fastglmm_diagnostic")
  expect_match(conditionMessage(w), "\\bg\\b")
  expect_match(conditionMessage(w), "rows 2, 3")
  got <- suppressWarnings(predict(m, new))
  pop <- predict(m, new, re.form = NA)
  expect_identical(unname(got[2:3]), unname(pop[2:3]))
  expect_false(got[[1]] == pop[[1]])
  expect_error(predict(m, new, allow.new.levels = FALSE), "did not see")
})

test_that("bad arguments are refused", {
  d <- predict_data()
  m <- quiet_fit(y ~ x + (1 | g), d)
  expect_error(predict(m, d, type = "probability"), "should be one of")
  expect_error(predict(m, d, re.form = ~ (1 | g)), "re.form must be NULL or NA")
  expect_error(predict(m, d, level = 1.5), "level must be in")
  expect_error(predict(m, d[c("x", "y")]), "grouping columns")
  expect_error(predict(m, d, future = TRUE), "future applies only to a non-Gaussian mixed model")
})

test_that("simulate is seeded and has one row per fitted row", {
  d <- predict_data()
  d$y[4] <- NA
  m <- quiet_fit(y ~ x + (1 | g), d)
  a <- simulate(m, nsim = 3, seed = 7)
  expect_s3_class(a, "data.frame")
  expect_identical(dim(a), c(nobs(m), 3L))
  expect_identical(nobs(m), nrow(d) - 1L)
  expect_named(a, c("sim_1", "sim_2", "sim_3"))
  expect_identical(a, simulate(m, nsim = 3, seed = 7))
  expect_false(identical(a, simulate(m, nsim = 3, seed = 8)))
  set.seed(1)
  b <- simulate(m)
  set.seed(1)
  expect_identical(b, simulate(m))
  expect_error(simulate(m, nsim = 0), "nsim")
})

test_that("ranef(condVar = TRUE) carries a postVar of terms x terms x levels", {
  d <- predict_data()
  m <- quiet_fit(ys ~ x + (1 + x | g), d)
  re <- ranef(m, condVar = TRUE)$g
  pv <- attr(re, "postVar")
  expect_identical(dim(pv), c(2L, 2L, nrow(re)))
  expect_identical(pv[1, 2, ], pv[2, 1, ])
  expect_identical(ranef(m)$g, re)
  expect_null(attr(ranef(m, condVar = FALSE)$g, "postVar"))
})

test_that("a fit without conditional variances raises a clear error", {
  d <- predict_data()
  # An extra grouping with a random slope routes to the sparse solver, which
  # forms no conditional variances.
  m <- quiet_fit(ys ~ x + (1 | g) + (1 + x | h), d)
  expect_error(ranef(m, condVar = TRUE), "no conditional variances")
  expect_null(attr(ranef(m)$g, "postVar"))
  expect_length(predict(m), nrow(d))
  expect_error(predict(m, se.fit = TRUE), "no conditional variances")
})

test_that("condvar = FALSE skips the conditional variances", {
  d <- predict_data()
  on <- quiet_fit(ys ~ x + (1 + x | g), d)
  off <- quiet_fit(ys ~ x + (1 + x | g), d, condvar = FALSE)
  expect_null(attr(ranef(off)$g, "postVar"))
  expect_identical(unclass(ranef(off)$g), unclass(ranef(on, condVar = FALSE)$g))
  expect_error(ranef(off, condVar = TRUE), "condvar = FALSE")
  expect_error(predict(off, se.fit = TRUE), "condvar off")
})

test_that("a crossed fit keeps its stored covariances through saveRDS", {
  d <- predict_data()
  m <- quiet_fit(yc ~ x + (1 | g) + (1 | h), d)
  path <- tempfile(fileext = ".rds")
  saveRDS(m, path)
  back <- readRDS(path)
  expect_identical(predict(back, se.fit = TRUE), predict(m, se.fit = TRUE))
})
