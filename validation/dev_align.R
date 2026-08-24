# dev_align.R -- per-family x per-method deviance convention alignment.
# Shared convention: dev = -2 * loglik as each engine reports it, corrected
# per the inventory below. Sourced by compare.R and measure_dev_floor.R.
# mirrors tests/oracle_support/mod.rs's dev_align module -- change together

# Inventory -- every case cites its source; "none" is an explicit statement that
# the engines' loglik conventions already agree there (evidenced by the loglik
# band passing on the frozen goldens):
#   *             default          none -- loglik band (tol.R loglik_abs_lmm/glmm) green on frozen goldens
#   lme4          nagq>1           add verified closed-form saturated logLik correction (closed-form.R, verified 2026-08-24)
#   GLMMadaptive  vector-RE AGQ    no loglik in golden -> NA (excluded loudly) -- goldens/sim_{binomial,poisson}_slope*_agq_k{7,11}.json field audit 2026-08-24
#   glmm          gaussian REML    none at loglik level -- speed-grid analyze.R:73-75 adds df_reml*(1+log(2*pi)) to glmm's *internal dev*; reported loglik already matches lme4's REML criterion (loglik band green)

# Locate this file's own directory regardless of the caller's cwd (compare.R
# sources it by an absolute script_dir path; a standalone `Rscript -e
# 'source("dev_align.R")'` run has cwd == this directory instead) -- same
# frame-walk trick as tol.R's tol_suite_dir, without that function's
# manifest.json dependency (dev_align.R only ever needs its own directory).
dev_align_dir <- function() {
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (is.character(of) && length(of) == 1L && file.exists(of) &&
        identical(basename(of), "dev_align.R")) {
      return(dirname(normalizePath(of)))
    }
  }
  arg <- grep("--file=", commandArgs(FALSE), value = TRUE)
  if (length(arg) == 1L) return(dirname(normalizePath(sub("--file=", "", arg))))
  getwd()
}

# Goldens carry a bare dataset NAME (meta$data == "cbpp", "sim_binomial_slope1",
# ...), not the data itself -- same split-by-`sim_` convention as
# engines/goldens_agq.R::data_dir_of_name (empirical/ for named real datasets,
# simulated/ for the sim_* corpus), read back from validation/data/*.csv (the
# CSVs engines/lme4.R and goldens_agq.R fit from).
load_meta_data <- function(meta) {
  sub_dir <- if (startsWith(meta$data, "sim_")) "simulated" else "empirical"
  path <- file.path(dev_align_dir(), "data", sub_dir, paste0(meta$data, ".csv"))
  read.csv(path, stringsAsFactors = FALSE)
}

# Response vector(s) off meta$r_formula. Two conventions appear in the corpus:
#   cbind(y, n - y) ~ ...   -- aggregated binomial counts (cbpp: incidence/size)
#   y ~ ...                 -- bare response column, y is 0/1 Bernoulli (n = 1
#                              per row) for binomial, a count for poisson
# Bernoulli data does not need a special case below: with n = 1 and y in
# {0, 1}, the closed-form saturated-binomial sum is identically 0 on every
# row (lchoose(1, y) = 0; the y*log(p) and (n-y)*log(1-p) terms each hit their
# own y==0/y==n skip), which is the well-known fact that a Bernoulli fit is
# always saturated -- so plugging n = 1 into the same formula is correct, not
# an approximation.
binomial_response <- function(df, r_formula) {
  cb <- regmatches(r_formula, regexec(
    "cbind\\(\\s*(\\w+)\\s*,\\s*(\\w+)\\s*-\\s*\\1\\s*\\)", r_formula))[[1]]
  if (length(cb) == 3) {
    return(list(y = df[[cb[2]]], n = df[[cb[3]]]))
  }
  resp <- trimws(strsplit(r_formula, "~", fixed = TRUE)[[1]][1])
  if (!resp %in% names(df)) {
    stop("binomial_response: cannot find response column `", resp,
         "` (or a cbind(y, n-y) form) in r_formula `", r_formula, "`")
  }
  list(y = df[[resp]], n = rep(1, nrow(df)))
}

poisson_response <- function(df, r_formula) {
  resp <- trimws(strsplit(r_formula, "~", fixed = TRUE)[[1]][1])
  if (!resp %in% names(df)) {
    stop("poisson_response: cannot find response column `", resp,
         "` in r_formula `", r_formula, "`")
  }
  df[[resp]]
}

# Closed form of the saturated-model loglik, ported verbatim (sign and form)
# from the verified closed-form script (closed-form.R, 2026-08-24). That script
# computes `sat_binom`/`sat_pois` = -2 * logLik(saturated) (an AIC-style
# constant); this returns the loglik itself (that value / -2), which is what
# aligned_dev adds directly to lme4's reported nAGQ>1 loglik -- verified
# against the frozen cbpp and grouseticks nAGQ=1 goldens (closes an
# 84-unit / 931-unit raw deviance gap to <1 unit). Adding the un-halved,
# unnegated `sat_*` constant instead overshoots by 2x; that is the failure
# mode those two goldens are here to catch.
saturated_loglik_deficit <- function(meta) {
  if (!meta$family %in% c("binomial", "poisson")) {
    stop("no verified saturated correction for family ", meta$family)
  }
  df <- load_meta_data(meta)
  if (identical(meta$family, "binomial")) {
    r <- binomial_response(df, meta$r_formula)
    y <- r$y; n <- r$n; p <- y / n
    sat_binom <- -2 * sum(lchoose(n, y) + ifelse(y > 0, y * log(p), 0) +
                           ifelse(y < n, (n - y) * log(1 - p), 0))
    return(-sat_binom / 2)
  }
  # poisson
  y <- poisson_response(df, meta$r_formula)
  sat_pois <- -2 * sum(ifelse(y > 0, y * log(y), 0) - y - lfactorial(y))
  -sat_pois / 2
}

# engine, as read straight off a golden JSON's `engine` field, is NOT a bare
# "lme4" -- goldens carry the qualified R call
# ("lme4::glmer", "lme4::glmer.nb", "lme4::lmer"; verified 2026-08-24 across
# every goldens/*.json). Matched with a prefix test rather than identical() so
# the nAGQ>1 correction actually fires instead of silently never matching.
is_lme4 <- function(engine) startsWith(engine, "lme4")

aligned_dev <- function(engine, estimates, meta) {
  ll <- estimates$loglik
  if (is.null(ll)) {
    out <- NA_real_
    attr(out, "why") <- sprintf("%s golden carries no loglik", engine)
    return(out)
  }
  nagq <- if (!is.null(meta$nagq)) meta$nagq else 1L
  if (is_lme4(engine) && nagq > 1L) {
    ll <- ll + saturated_loglik_deficit(meta)   # sign verified in Step 4
  }
  -2 * ll
}
