#!/usr/bin/env Rscript
# lme4 reference fits over the parity datasets -> results/lme4/<dataset>.json.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference glmm is later held to.
# On any glmm disagreement, glmm is presumed wrong. A reference is regenerated ONLY
# if the model SPEC (formula/family/link) is proven wrong, with a recorded reason.
# Never relax a tolerance or edit a result to make a downstream engine pass.

suppressMessages({
  library(lme4)
  library(jsonlite)
})

parity_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))

manifest <- fromJSON(file.path(parity_dir, "manifest.json"), simplifyDataFrame = FALSE)
out_dir <- file.path(parity_dir, "results", "lme4")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

N_RUNS <- 10  # timing loop; first pass discarded, min of the rest reported

read_dataset <- function(spec) {
  df <- read.csv(file.path(parity_dir, "data", paste0(spec$name, ".csv")),
                 stringsAsFactors = FALSE)
  # Re-establish factor typing for grouping + categorical fixed-effect columns
  # (CSV round-trip loses it; numeric-looking levels like cbpp `period` come back
  # as integers). Coercion order is the sorted-level default => treatment-contrast
  # base = first sorted level, matched against Julia's DummyCoding in compare.R.
  for (f in unlist(spec$factors)) df[[f]] <- factor(df[[f]])
  df
}

fit_call <- function(spec, df) {
  fm <- as.formula(spec$r_formula)
  if (spec$family == "gaussian") {
    function() lmer(fm, data = df, REML = isTRUE(spec$reml))
  } else {
    fam <- switch(spec$family, binomial = binomial(), poisson = poisson(),
                  stop("unsupported family: ", spec$family))
    function() glmer(fm, data = df, family = fam)
  }
}

time_fit <- function(make_fit) {
  times <- numeric(N_RUNS)
  for (i in seq_len(N_RUNS)) {
    t0 <- Sys.time()
    invisible(make_fit())
    times[i] <- as.numeric(Sys.time() - t0, units = "secs")
  }
  list(fit_seconds_min = min(times[-1]), n_runs = N_RUNS, warmup_discarded = 1L)
}

# VarCorr -> common representation: per grouping factor, the RE term names, their
# standard deviations, and the correlation matrix between them. lme4 reports these
# directly via the `stddev`/`correlation` attributes (absolute scale).
varcomp_of <- function(m) {
  vc <- VarCorr(m)
  lapply(names(vc), function(g) {
    block <- vc[[g]]
    sd <- attr(block, "stddev")
    corr <- attr(block, "correlation")
    # I() keeps length-1 vectors (single-term groupings) as JSON arrays under
    # auto_unbox, so positional comparison in compare.R is uniform across rungs.
    list(group = g, terms = I(names(sd)),
         stddev = I(unname(sd)), corr = unname(corr))
  })
}

fit_one <- function(spec) {
  df <- read_dataset(spec)
  make_fit <- fit_call(spec, df)
  m <- make_fit()

  conv_msgs <- m@optinfo$conv$lme4$messages
  est <- list(
    beta   = I(unname(fixef(m))),
    se     = I(unname(sqrt(diag(as.matrix(vcov(m)))))),
    loglik = as.numeric(logLik(m)),
    varcomp = varcomp_of(m)
  )
  if (spec$family == "gaussian") est$sigma <- sigma(m)

  res <- list(
    dataset = spec$name, engine = "lme4",
    engine_version = as.character(packageVersion("lme4")),
    family = spec$family,
    reml = if (is.null(spec$reml)) NA else isTRUE(spec$reml),
    rung = spec$rung,
    converged = is.null(conv_msgs) || length(conv_msgs) == 0,
    singular = isSingular(m),
    coef_names = I(names(fixef(m))),  # contrast-coding assertion vs Julia
    estimates = est,
    timing = time_fit(make_fit)
  )

  out <- file.path(out_dir, paste0(spec$name, ".json"))
  # digits = NA: full double precision -- this is an oracle, not a display.
  write(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  cat(sprintf("lme4  %-12s  rung %d  converged=%s singular=%s  fit_min=%.4gs\n",
              spec$name, spec$rung, res$converged, res$singular,
              res$timing$fit_seconds_min))
}

for (spec in manifest$datasets) fit_one(spec)
