#!/usr/bin/env Rscript
# R-port side of the cross-language validation suite -- fits the validation datasets with
# the `fastglmm` R package (the extendr wrapper over the same Rust `glmm` kernel) and
# writes `results/glmm_r_{empirical,simulated}/<ds>.json` in the common schema
# (validation/README.md).
#
# Why a fifth engine. `fastglmm()` calls the SAME kernel as `glmm.rs` and the Python
# port `glmm_python.py`: the formula is lowered through `glmm::formula` and fit through
# the one cold-start path, so the estimates must match the Rust `glmm` row to round-off,
# and `compare.R` gates them there (TOL$port_rel) rather than against lme4 -- a port
# bug (a swapped column, a mis-ordered factor level, a dropped weights vector) shows
# up as a Rust-vs-R disagreement, which the package's own testthat suite cannot see:
# it fits fresh data and asserts only convergence, never a reference number.
#
# This file is a deliberate line-for-line port of `glmm_python.py` (read_csv_path,
# build_data, formula_of, build_fit_data, varcomp, median_secs, fit_one) -- a divergence
# between the two ports' rewrite logic must be visible as a diff between near-identical
# functions, not hidden behind an independently-derived R version. It is NOT named
# `lme4.R` (that is already the lme4 oracle).
#
# Manifest-driven, same reason `glmm_python.py`/`glmm.rs` are: the manifest carries both
# the main rungs and the weights tier. Run via `run.sh` (ENGINES has "glmm_r") or
# `Rscript glmm_r.R`; paths are anchored at this file, so the cwd does not matter.

suppressMessages({
  library(fastglmm)
  library(jsonlite)
})

`%||%` <- function(x, y) if (is.null(x)) y else x

# Correctly-rounded string->double parse via glibc strtod. R_strtod (behind
# as.numeric/read.csv/scan) is NOT correctly rounded: a near-halfway decimal string
# can parse 1 ULP below the double Python's float() and Rust's str::parse produce.
# Two known cases in sim_sparse_gamma.csv ("0.639264896501405", "0.51258705226397",
# confirmed by exact-fraction comparison) fed rung 24's flat optimization surface a
# 2-ULP input difference that BOBYQA amplified to ~1e-3 in beta, failing the
# round-off port gate (TOL$port_rel). Every numeric column the kernel sees must go
# through this shim -- do not "simplify" back to as.numeric. (data.table::fread
# mis-rounds the same strings; glibc strtod is verified against the Python bits.)
Rcpp::cppFunction('
NumericVector strtod_parse(CharacterVector s) {
  int n = s.size();
  NumericVector out(n);
  for (int i = 0; i < n; i++) out[i] = std::strtod(CHAR(STRING_ELT(s, i)), nullptr);
  return out;
}', includes = "#include <cstdlib>")

# The installed package's version -- the R package is pinned lockstep with the crate
# (DESCRIPTION Version), so this is glmm.rs's CARGO_PKG_VERSION by construction.
VERSION <- as.character(packageVersion("fastglmm"))

# Timing is OPT-IN and its sample count lives in run.sh, not here, so the engines'
# medians are comparable without renormalizing and no mirrored constant has to be
# kept in step across the five.
#
# THE contract, mirrored in glmm.rs / lme4.R / mixedmodels.jl / glmm_python.py
# `timing_runs` -- change together: VALIDATION_TIMINGS unset / "" / "0" means do not
# time (`timing` is written null); otherwise it IS the sample count, an integer >= 2,
# first (cold) pass discarded, MEDIAN of the rest. run.sh validates the value; this
# errors rather than silently skipping timing when run by hand with a malformed one.
timing_runs <- function() {
  v <- trimws(Sys.getenv("VALIDATION_TIMINGS"))
  if (v == "" || v == "0") return(NULL)
  n <- suppressWarnings(as.integer(v))
  if (is.na(n) || n < 2)
    stop("VALIDATION_TIMINGS must be 0 or an integer >= 2 (got '", v,
         "'); N=2 keeps 1 sample after the warm-up discard")
  n
}
N_RUNS <- timing_runs()  # NULL on an untimed run

# Suite directory (manifest + data + results root). Mirrors glmm_python.py's SUITE.
HERE <- normalizePath(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
SUITE <- dirname(HERE)

# NaN/Inf -> JSON null (mirrors common.rs::num / glmm_python.py's num): a non-converged
# fit leaves NaN-filled estimates, and the comparators cannot read a `NaN` literal. For
# ARRAY fields, wrap with nums() (a list keeps its length, non-finite -> null element);
# for a SCALAR field use num_scalar() (NA, written as null via toJSON's na="null").
num_scalar <- function(x) if (is.numeric(x) && length(x) == 1L && is.finite(x)) x else NA_real_
nums <- function(xs) lapply(as.numeric(xs), function(x) if (is.finite(x)) x else NULL)

read_csv_path <- function(path) {
  # Read a validation CSV (unquoted header + rows, `,`-split) -- mirrors
  # common.rs::read_csv_path / glmm_python.py, deliberately including its naivety:
  # the corpus carries no embedded commas.
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  unq <- function(s) trimws(gsub('^"|"$', "", trimws(s)))
  header <- unq(strsplit(lines[1], ",", fixed = TRUE)[[1]])
  rows <- lapply(lines[-1], function(ln) unq(strsplit(ln, ",", fixed = TRUE)[[1]]))
  list(header = header, rows = rows)
}

is_float <- function(s) !is.na(suppressWarnings(as.numeric(s)))

build_data <- function(header, rows, factors) {
  # Columns typed the way common.rs::build_table / glmm_python.py::build_data type
  # them: manifest `factors` are categorical, as is any column that fails to parse as
  # numeric anywhere. Dots in R-origin headers (Arabidopsis' total.fruits) become
  # underscores to match jl_formula's sanitized names.
  #
  # Factor levels are forced to BYTE order (sort method="radix" = C locale = Rust's
  # str ordering), NOT R's default factor() which sorts by the session LC_COLLATE.
  # The Python port passes string columns that the kernel byte-sorts (Column::
  # factor_from_labels); to feed the SAME codes -- and gate at round-off against the
  # Rust row -- fastglmm must receive factors already in that order (it honors a
  # factor's declared level order). This is invisible to a fixed-effect factor's coef
  # names when the two orders happen to agree, but a GROUPING factor whose byte and
  # locale orders differ (VerbAgg's `item`) would otherwise permute the group order,
  # perturb FP summation, and diverge on the flatter optimization surfaces.
  out <- list()
  for (j in seq_along(header)) {
    name <- header[j]
    values <- vapply(rows, function(r) r[[j]], "")
    is_factor <- name %in% factors || any(!vapply(values, is_float, logical(1)))
    out[[gsub(".", "_", name, fixed = TRUE)]] <-
      if (is_factor) factor(values, levels = sort(unique(values), method = "radix"))
      else strtod_parse(values)
  }
  data.frame(out, stringsAsFactors = FALSE, check.names = FALSE)
}

formula_of <- function(spec) {
  # The manifest entry's formula, lowered to what the crate's parser takes -- mirrors
  # glmm_python.py::formula_of step for step: jl_formula is the source (guaranteed cbind-free),
  # `@formula(...)` unwrapped; Julia's `&` grouping operator becomes the parser's `:`;
  # the explicit `1` intercept is stripped (the parser treats it as implicit). Rungs
  # without a jl_formula (gamma, the weights suite) fall back to r_formula, whose
  # aggregated-binomial `cbind(...)` response is rewritten to the synthesized `prop`.
  jl <- spec$jl_formula
  if (!is.null(jl)) {
    if (!(startsWith(jl, "@formula(") && endsWith(jl, ")"))) {
      stop("jl_formula not in @formula(...) shape: ", jl)
    }
    f <- substr(jl, nchar("@formula(") + 1L, nchar(jl) - 1L)
  } else {
    r <- spec$r_formula
    if (is.null(r)) stop("manifest entry missing both jl_formula and r_formula")
    parts <- strsplit(r, "~", fixed = TRUE)[[1]]
    resp <- parts[1]
    f <- if (length(parts) > 1L && startsWith(trimws(resp), "cbind("))
      paste0("prop ~", paste(parts[-1], collapse = "~")) else r
  }
  sub(" ~ 1 + ", " ~ ", gsub(" & ", ":", f, fixed = TRUE), fixed = TRUE)
}

build_fit_data <- function(spec, header, rows, factors) {
  # `list(data, weights)` for one manifest entry -- the R twin of
  # common.rs::lower_dataset_generic plus glmm.rs's `weights_col` branch, ported
  # from glmm_python.py::build_fit_data. An aggregated-binomial rung (manifest `weights`)
  # synthesizes `prop = incidence/<weights_col>` so jl_formula's `prop ~ ...` response
  # resolves, and passes the cluster sizes as prior weights, one per aggregate row.
  # `weights_col` (the weights suite) is plain per-row weights off a named column. The
  # two are mutually exclusive per rung by design -- asserted, as in glmm.rs.
  data <- build_data(header, rows, factors)
  column <- function(name) strtod_parse(vapply(rows, function(r) r[[match(name, header)]], ""))

  # Manifest `offset`: a plain named-column lookup, no synthesis -- the counterpart of
  # glmm_python.py::offset_of. Copied to the FIXED name `validation_off` rather than
  # referenced under its manifest name, so do_fit() can name it as a literal symbol for
  # fastglmm's eval-in-data (the same reason `validation_wts` exists).
  oc <- spec[["offset"]]
  if (!is.null(oc)) {
    if (!(oc %in% header)) stop("offset ", oc, " not in CSV header")
    data[["validation_off"]] <- column(oc)
  }

  # `[[` (exact), not `$`: R's `$` partial-matches, so `spec$weights` would return the
  # `weights_col` value on a weights-suite rung (weights absent) and trip the assert.
  w_name <- spec[["weights"]]
  if (!is.null(w_name)) {
    if (!is.null(spec[["weights_col"]])) stop("weights_col and weights are mutually exclusive")
    sizes <- column(w_name)
    data[["prop"]] <- column("incidence") / sizes
    return(list(data = data, weights = sizes))
  }
  wc <- spec[["weights_col"]]
  if (!is.null(wc)) {
    if (!(wc %in% header)) stop("weights_col ", wc, " not in CSV header")
    return(list(data = data, weights = column(wc)))
  }
  list(data = data, weights = NULL)
}

build_family <- function(spec) {
  # Manifest `family` (+ optional `link`) -> the family ARGUMENT fastglmm() takes.
  # fastglmm() has no separate `link=` kwarg (unlike glmm.fit): the link rides on the
  # family object. Gamma is ALWAYS Gamma(link="log" unless overridden) -- never bare
  # Gamma(), whose R default "inverse" would silently fit a different model than every
  # other engine (the "Gamma() link trap", fastglmm.R). negbin must be the STRING form:
  # a MASS::negative.binomial(theta) object fixes the shape, which fastglmm() rejects
  # (it estimates the shape).
  link <- spec$link
  switch(spec$family,
    gaussian = stats::gaussian(),
    binomial = stats::binomial(link = link %||% "logit"),
    poisson  = stats::poisson(),
    gamma    = stats::Gamma(link = link %||% "log"),
    negbin   = "negativebinomial",
    stop("unsupported family: ", spec$family)
  )
}

do_fit <- function(data, formula, family, wald_se) {
  # One fastglmm() call. weights and offset ride as data columns (`validation_wts`,
  # `validation_off`) referenced by name so fastglmm's eval-in-data resolves them
  # deterministically inside the timing closures, never via parent.frame() -- which is
  # why the four arms are spelled out rather than assembled with do.call(). No manifest
  # rung carries both today; the arm exists so one would not silently drop its offset.
  # suppressWarnings: the expected singular-boundary / nAGQ-fallback notices are captured
  # on the fit object (singular, converged), not needed on stderr for a batch run.
  has_w <- "validation_wts" %in% names(data)
  has_o <- "validation_off" %in% names(data)
  suppressWarnings(
    if (has_w && has_o)
      fastglmm(formula, data, family, weights = validation_wts,
               offset = validation_off, wald.se = wald_se)
    else if (has_w)
      fastglmm(formula, data, family, weights = validation_wts, wald.se = wald_se)
    else if (has_o)
      fastglmm(formula, data, family, offset = validation_off, wald.se = wald_se)
    else
      fastglmm(formula, data, family, wald.se = wald_se)
  )
}

median_secs <- function(batch, call) {
  # Median seconds over N_RUNS samples, warm-up (first) discarded. Each sample times
  # `batch` fits (the manifest `timing_batch` every engine reads); summarize_timing.R
  # divides by fits_per_sample. Mirrors glmm_python.py::median_secs.
  #
  # Timer is Sys.time(), NOT proc.time(): proc.time()'s elapsed field ticks at 1 ms on
  # Linux, so every sub-ms fit (the small intercept-only/3-level rungs) floored to
  # exactly 0.001 s and reported a bogus 5-14x port "tax". Sys.time() resolves to ~2 us
  # here (measured), which is the true high-resolution-wall-clock analogue of Python's
  # perf_counter -- proc.time was the wrong analogue for it.
  samples <- numeric(N_RUNS)
  for (i in seq_len(N_RUNS)) {
    t0 <- as.numeric(Sys.time())
    for (j in seq_len(batch)) call()
    samples[i] <- as.numeric(Sys.time()) - t0
  }
  stats::median(samples[-1])
}

group_names_match <- function(a, b) {
  # Order-invariant on `:`-joined components -- mirrors glmm_python.py/glmm.rs::group_names_match:
  # lme4 names a nested inner group `child:parent`, the formula frontend `parent:child`.
  identical(sort(strsplit(a, ":", fixed = TRUE)[[1]]),
            sort(strsplit(b, ":", fixed = TRUE)[[1]]))
}

varcomp <- function(fit, ref_order, include_se) {
  # Variance components in the common schema, one entry per grouping factor, built from
  # the package's exported VarCorr() (per-group stddev + correlation) -- what
  # glmm_python.py::varcomp assembles by hand from Fit.stddev_corr. stddev_se (GLMM Hessian
  # only) is read from fit$stddev_se exactly as glmm_python.py reads f.stddev_se: a flat
  # vech-layout vector over ALL theta coordinates, so groupings are walked in
  # DECLARATION order with a cumulative offset advanced by q*(q+1)/2, and an entry is
  # emitted only for scalar (q=1) groupings -- the only shape the theta==stddev identity
  # holds for, same as lme4's own gating. Reindexed to `ref_order` because compare.R
  # aligns varcomp POSITIONALLY, not by name.
  gnames <- fit$re_group_names
  gterms <- fit$re_group_terms
  vc <- if (length(gnames)) VarCorr(fit) else list()

  natural <- vector("list", length(gnames))
  theta_offset <- 0L
  for (i in seq_along(gnames)) {
    stddev <- as.numeric(attr(vc[[i]], "stddev"))
    corr <- attr(vc[[i]], "correlation")
    q <- length(stddev)
    entry <- list(
      group = gnames[i],
      terms = as.list(gterms[[i]]),
      stddev = nums(stddev),
      corr = lapply(seq_len(nrow(corr)), function(r) nums(corr[r, ]))
    )
    if (include_se && q == 1L) entry$stddev_se <- nums(fit$stddev_se[theta_offset + 1L])
    theta_offset <- theta_offset + q * (q + 1L) / 2L
    natural[[i]] <- entry
  }

  lapply(ref_order, function(name) {
    idx <- Find(function(i) group_names_match(gnames[i], name), seq_along(gnames))
    if (is.null(idx)) stop("reference group ", name, " not found in fit's re_groups")
    natural[[idx]]
  })
}

fit_one <- function(spec) {
  # Fit one manifest entry end-to-end (load -> fit(+SE split) -> time -> reindex varcomp
  # to the reference's grouping order -> write). Mirrors glmm_python.py::fit_one.
  ds <- spec$name
  family_str <- spec$family
  gaussian <- family_str == "gaussian"
  factors <- spec$factors %||% character(0)
  source <- if (identical(spec$source, "sim")) "simulated" else "empirical"

  # `data` field: CSV to read when it differs from the rung name -- a re-linked rung
  # (cbpp_probit) reuses the committed dataset byte-for-byte.
  data_name <- spec$data %||% ds
  csv <- read_csv_path(file.path(SUITE, "data", source, paste0(data_name, ".csv")))
  fd <- build_fit_data(spec, csv$header, csv$rows, factors)
  data <- fd$data
  if (!is.null(fd$weights)) data[["validation_wts"]] <- fd$weights
  formula <- formula_of(spec)
  family <- build_family(spec)
  timing_batch <- spec$timing_batch %||% 1L

  # Reference grouping order (compare.R aligns varcomp positionally, not by name) --
  # read off the already-frozen lme4 result rather than re-deriving lme4's convention.
  reference <- fromJSON(file.path(SUITE, "results", paste0("lme4_", source),
                                  paste0(ds, ".json")),
                        simplifyVector = TRUE, simplifyDataFrame = FALSE,
                        simplifyMatrix = TRUE)
  ref_order <- vapply(reference$estimates$varcomp, function(e) e$group, "")

  fh_fit <- do_fit(data, formula, family, "hessian")
  fixed_only <- length(fh_fit$re_group_names) == 0L

  if (gaussian || fixed_only) {
    # One SE, no method choice: a gaussian rung has a single profiled `se`, a fixed-only
    # GLM (weights suite) has no theta. Emitted in the slot glmm_python.py uses for each -- `se`
    # for gaussian, `se_rx` for fixed-only -- so compare.R's se_rx_of() lines them up.
    # NA, not NULL, on an untimed run: jsonlite writes an empty object for a NULL
    # list element and `null` for NA under na = "null".
    timing <- if (is.null(N_RUNS)) NA else list(
      fit_seconds_median = median_secs(timing_batch,
        function() do_fit(data, formula, family, "hessian")),
      n_runs = N_RUNS, warmup_discarded = 1L, fits_per_sample = timing_batch)
    estimates <- list(beta = nums(fh_fit$beta), varcomp = varcomp(fh_fit, ref_order, FALSE))
    estimates[[if (gaussian) "se" else "se_rx"]] <- nums(fh_fit$se)
    converged <- fh_fit$converged; n_eval <- fh_fit$n_eval; deviance <- fh_fit$deviance
  } else {
    # GLMM SE has two genuinely different Laplace variants -- emit both so compare.R
    # checks like to like: se_hessian keeps the theta-beta coupling (glmm default), se_rx
    # is conditional on theta-hat. beta/tau is wald_se-independent.
    fr_fit <- do_fit(data, formula, family, "rx")
    # Split timing by SE method -- the FD-Hessian is the main time consumer, Rx is one
    # closed-form Schur solve. Same PIRLS fit underlies both.
    timing <- if (is.null(N_RUNS)) NA else list(
      fit_seconds_median_rx = median_secs(timing_batch,
        function() do_fit(data, formula, family, "rx")),
      fit_seconds_median_hessian = median_secs(timing_batch,
        function() do_fit(data, formula, family, "hessian")),
      n_runs = N_RUNS, warmup_discarded = 1L, fits_per_sample = timing_batch)
    estimates <- list(
      beta = nums(fh_fit$beta),
      se_hessian = nums(fh_fit$se),
      se_rx = nums(fr_fit$se),
      # stddev_se from the Hessian fit's theta block.
      varcomp = varcomp(fh_fit, ref_order, TRUE))
    converged <- fh_fit$converged && fr_fit$converged
    n_eval <- fh_fit$n_eval + fr_fit$n_eval
    deviance <- fh_fit$deviance
  }

  res <- list(
    dataset = ds,
    engine = "glmm_r",
    engine_version = paste0(VERSION, "-local"),
    family = family_str,
    reml = if (gaussian) (spec$reml %||% FALSE) else NA,
    rung = spec$rung,
    converged = converged,
    singular = fh_fit$singular,
    optimizer = "bobyqa",
    n_eval = n_eval,
    deviance = num_scalar(deviance),
    coef_names = as.list(names(fh_fit$beta)),
    estimates = estimates,
    timing = timing)
  write_result(ds, source, res)
}

write_result <- function(ds, source, res) {
  out <- file.path(SUITE, "results", paste0("glmm_r_", source), paste0(ds, ".json"))
  # digits = NA: full-precision doubles. The port gate (TOL$port_rel = 1e-12) compares
  # glmm_r against the Rust glmm row; jsonlite's default 4-digit rounding would fail it.
  writeLines(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), out)
  # `$` on the NA an untimed run leaves behind would error ("invalid for atomic
  # vectors"), so test for the list first; the console line then omits the time.
  t <- if (is.list(res$timing)) res$timing$fit_seconds_median %||% res$timing$fit_seconds_median_rx
       else NULL
  cat(sprintf("glmm_r   %-12s  rung %s  converged=%s%s\n",
              ds, res$rung, res$converged,
              if (is.null(t)) "" else sprintf("  fit_median=%.4fs", t)))
}

main <- function() {
  for (source in c("empirical", "simulated")) {
    dir.create(file.path(SUITE, "results", paste0("glmm_r_", source)),
               showWarnings = FALSE, recursive = TRUE)
  }
  manifest <- fromJSON(file.path(SUITE, "manifest.json"),
                       simplifyVector = TRUE, simplifyDataFrame = FALSE,
                       simplifyMatrix = TRUE)
  # VALIDATION_ONLY=<name>[,<name>...]: fit only the named datasets (mirrors the other
  # engines) -- reruns a single rung without repaying the full-corpus timing cost.
  only <- Sys.getenv("VALIDATION_ONLY")
  want <- if (!nzchar(only)) function(ds) TRUE else {
    names <- strsplit(only, ",", fixed = TRUE)[[1]]
    function(ds) ds %in% names
  }
  for (spec in manifest$datasets) {
    if (!want(spec$name)) next
    fit_one(spec)
  }
}

main()
