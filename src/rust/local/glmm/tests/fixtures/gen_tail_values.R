#!/usr/bin/env Rscript
# Generator for tail_values.json: R's p-value tails and quantiles, frozen once as
# the reference for `src/inference/tails_tests.rs` (the t, F and chi-squared upper
# tails and the t and normal quantile wrappers). Each point is written with its
# arguments and R's value:
#   t_two_sided  2 * pt(-abs(t), df)                  (two-sided t p-value)
#   f_upper      pf(f, num_df, den_df, lower.tail = FALSE)
#   chisq_upper  pchisq(x, df, lower.tail = FALSE)
#   qt           qt(p, df)
#   qnorm        qnorm(p)
# The values come from base R's `stats` only; jsonlite only writes the file.
# The points include the deep tail pchisq(107.2, 1, lower.tail = FALSE), about
# 4e-25, which a `1 - CDF` tail returns as 0.
#
# Run from the crate root:  Rscript tests/fixtures/gen_tail_values.R

suppressMessages(library(jsonlite))

path <- file.path("tests", "fixtures", "tail_values.json")

t_pts <- data.frame(
  t  = c(0, 1.5, 2.0, -3.2, 40, 2.5),
  df = c(1, 1, 2.5, 30, 30, 1e6)
)
t_pts$p <- 2 * pt(-abs(t_pts$t), t_pts$df)

f_pts <- data.frame(
  f      = c(2.5, 3.1, 0.7, 50),
  num_df = c(1, 4, 12, 4),
  den_df = c(3, 17.3, 2.1, 17.3)
)
f_pts$p <- pf(f_pts$f, f_pts$num_df, f_pts$den_df, lower.tail = FALSE)

chisq_pts <- data.frame(
  x  = c(107.2, 3.84, 0.5, 30, 200),
  df = c(1, 1, 2, 5.5, 10)
)
chisq_pts$p <- pchisq(chisq_pts$x, chisq_pts$df, lower.tail = FALSE)

qt_pts <- data.frame(
  p  = c(0.975, 0.975, 0.025, 0.995),
  df = c(1, 2.5, 30, 1e6)
)
qt_pts$q <- qt(qt_pts$p, qt_pts$df)

qnorm_pts <- data.frame(p = c(0.975, 0.995, 0.5, 1e-10))
qnorm_pts$q <- qnorm(qnorm_pts$p)

fx <- list(
  provenance = list(
    r_version = R.version.string,
    generator = "tests/fixtures/gen_tail_values.R"
  ),
  t_two_sided = t_pts,
  f_upper = f_pts,
  chisq_upper = chisq_pts,
  qt = qt_pts,
  qnorm = qnorm_pts
)

# digits = NA: jsonlite's maximum, 15 significant digits, so each value carries at
# most 5e-15 relative rounding, far inside the tests' 1e-12 band.
write(toJSON(fx, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), path)
cat(sprintf("wrote %s (%s)\n", path, R.version.string))
