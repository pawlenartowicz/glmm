#!/usr/bin/env Rscript
# The grid's R port engine: the SAME kernel as grid/engines/glmm.rs reached
# through the fastglmm package (extendr), so its numbers must match the Rust
# engine to ROUND-OFF. Mirrors grid/engines/glmm_python.py function for
# function -- change together. It is NOT an oracle.
suppressMessages({ library(fastglmm); library(jsonlite) })
here <- normalizePath(dirname(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
source(file.path(here, "common.R"))
# The package version exactly as it reports it -- the same spelling glmm.rs
# writes ("0.4.0", no suffix); a dirty tree's provenance lives in
# run_meta.json's glmm_git_rev, not here.
VERSION <- as.character(packageVersion("fastglmm"))
env <- grid_env()
grid_threads()

`%||%` <- function(x, y) if (is.null(x)) y else x

# Correctly-rounded string->double parse via glibc strtod. R_strtod (behind
# as.numeric/read.csv/scan) is NOT correctly rounded: a near-halfway decimal
# string can parse 1 ULP off Python's float() and Rust's str::parse. Every
# numeric column the kernel sees must go through this shim -- do not
# "simplify" back to as.numeric, or a flat optimization surface can amplify a
# 1-2 ULP input difference past the round-off port gate (1e-12).
Rcpp::cppFunction('
NumericVector strtod_parse(CharacterVector s) {
  int n = s.size();
  NumericVector out(n);
  for (int i = 0; i < n; i++) out[i] = std::strtod(CHAR(STRING_ELT(s, i)), nullptr);
  return out;
}', includes = "#include <cstdlib>")

read_csv_path <- function(path) {
  # Read a grid CSV (unquoted header + rows, `,`-split) -- mirrors
  # common.rs::read_csv_path / glmm_python.py::read_csv_path, deliberately
  # including its naivety: the grid corpus carries no embedded commas.
  lines <- readLines(path, warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  unq <- function(s) trimws(gsub('^"|"$', "", trimws(s)))
  header <- unq(strsplit(lines[1], ",", fixed = TRUE)[[1]])
  rows <- lapply(lines[-1], function(ln) unq(strsplit(ln, ",", fixed = TRUE)[[1]]))
  list(header = header, rows = rows)
}

is_float <- function(s) !is.na(suppressWarnings(as.numeric(s)))

build_data <- function(header, rows, factors) {
  # Columns typed the way common.rs::build_table / glmm_python.py::build_data
  # type them: manifest `factors` are categorical, as is any column that
  # fails to parse as numeric anywhere. Dots in R-origin headers
  # (Arabidopsis' total.fruits) become underscores to match glmm_formula's
  # sanitized names.
  #
  # Factor levels are forced to BYTE order (sort method="radix" = C locale =
  # Rust's str ordering), NOT R's default factor() which sorts by the
  # session LC_COLLATE. The Python port passes string columns that the
  # kernel byte-sorts (Column::factor_from_labels); to feed the SAME codes
  # -- and gate at round-off against the Rust row -- fastglmm must receive
  # factors already in that order (it honors a factor's declared level
  # order). This is invisible to a fixed-effect factor's coef names when the
  # two orders happen to agree, but a GROUPING factor whose byte and locale
  # orders differ would otherwise permute the group order, perturb FP
  # summation, and diverge on the flatter optimization surfaces.
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

build_family <- function(cell) {
  # Manifest family (+ optional link) -> the family ARGUMENT fastglmm() takes.
  # fastglmm() has no separate link= kwarg (unlike glmm.fit): the link rides
  # on the family object. Gamma is ALWAYS Gamma(link="log" unless overridden)
  # -- never bare Gamma(), whose R default "inverse" would silently fit a
  # different model than every other engine (the "Gamma() link trap",
  # fastglmm.R). negativebinomial must be the STRING form: a
  # MASS::negative.binomial(theta) object fixes the shape, which fastglmm()
  # rejects (it estimates the shape).
  #
  # The resolved link is computed into its own variable and handed to the
  # family constructor as a BARE symbol. gaussian()/binomial()/Gamma()/
  # inverse.gaussian() resolve link= through substitute()/deparse() first,
  # falling back to evaluating it only once the deparsed text does not name a
  # known link: a compound expression passed directly (e.g. `link %||%
  # "log"`) deparses across R's line-width limit once it is long enough, and
  # a multi-line deparse makes that internal dispatch's `if (linktemp %in%
  # okLinks)` error with "the condition has length > 1". A bare symbol always
  # deparses to one word on one line, so the fallback path evaluates it
  # normally -- confirmed against inverse.gaussian() directly, which crashed
  # exactly this way on the inline form.
  link <- cell[["link"]]
  fam <- cell[["family"]]
  if (identical(fam, "negativebinomial")) return("negativebinomial")
  resolved <- switch(fam,
    gaussian = "identity",
    binomial = link %||% "logit",
    poisson  = "log",
    gamma    = link %||% "log",
    inversegaussian = if (identical(link, "inverse_squared")) "1/mu^2" else (link %||% "log"),
    stop("unsupported family: ", fam)
  )
  switch(fam,
    gaussian = stats::gaussian(link = resolved),
    binomial = stats::binomial(link = resolved),
    poisson  = stats::poisson(link = resolved),
    gamma    = stats::Gamma(link = resolved),
    inversegaussian = stats::inverse.gaussian(link = resolved)
  )
}

do_fit <- function(data, formula, family, wald_se, nagq) {
  # One fastglmm() call. weights and offset ride as DATA COLUMNS
  # (validation_wts, validation_off) referenced by name so fastglmm's
  # eval-in-data resolves them deterministically inside the fit call rather
  # than through parent.frame() -- the reason the four arms are spelled out
  # instead of assembled with do.call(). No grid cell carries both a
  # weights/weights_col field and an offset_col at once by construction, but
  # the arm exists so one would not silently drop the other's column.
  # suppressWarnings: the expected singular-boundary / nAGQ-fallback notices
  # are captured on the fit object (singular, converged), not needed on
  # stderr for a run of ~800 cells.
  has_w <- "validation_wts" %in% names(data)
  has_o <- "validation_off" %in% names(data)
  suppressWarnings(
    if (has_w && has_o)
      fastglmm(formula, data, family, weights = validation_wts,
               offset = validation_off, wald.se = wald_se, nAGQ = nagq)
    else if (has_w)
      fastglmm(formula, data, family, weights = validation_wts,
               wald.se = wald_se, nAGQ = nagq)
    else if (has_o)
      fastglmm(formula, data, family, offset = validation_off,
               wald.se = wald_se, nAGQ = nagq)
    else
      fastglmm(formula, data, family, wald.se = wald_se, nAGQ = nagq)
  )
}

# NaN/Inf -> JSON null (mirrors common.rs::num / glmm_python.py::num). For
# ARRAY fields, wrap with nums() (a list keeps its length, non-finite ->
# null element); for a SCALAR field use num_scalar() (NA, written as null via
# toJSON's na = "null").
num_scalar <- function(x) if (is.numeric(x) && length(x) == 1L && is.finite(x)) x else NA_real_
nums <- function(xs) lapply(as.numeric(xs), function(x) if (is.finite(x)) x else NULL)

varcomp <- function(fit) {
  # Variance components in the grid schema, one entry per grouping factor,
  # from VarCorr(fit) -- the R twin of the Python port's assembly off
  # Fit.stddev_corr. Empty for a fixed-only cell and empty when the fit left
  # varcorr unfilled: VarCorr() already returns length 0 in that case
  # (fastglmm-methods.R iterates seq_along(x$varcorr)). By group NAME; no
  # ref_order reindex.
  vc <- VarCorr(fit)
  gnames <- names(vc)
  lapply(seq_along(vc), function(i) {
    sd <- attr(vc[[i]], "stddev")
    corr <- attr(vc[[i]], "correlation")
    list(
      group = gnames[i],
      terms = as.list(names(sd)),
      stddev = nums(sd),
      corr = lapply(seq_len(nrow(corr)), function(r) nums(corr[r, ]))
    )
  })
}

not_converged_text <- function(fh) {
  # The boundary state, which variance components were pinned there, and any
  # solver notes -- mirrors common's common.rs::not_converged_text. Keeps a
  # non-convergence diagnosable in a run of ~800 cells, where the
  # alternative is a record that only says "engine-fail".
  d <- fh$diagnostics
  paste0("not converged: boundary=", d$boundary,
         ", pinned=", jsonlite::toJSON(d$pinned, auto_unbox = TRUE),
         ", notes=", jsonlite::toJSON(d$notes, auto_unbox = TRUE))
}

fit_one <- function(cell) {
  rec <- grid_record(cell, "glmm_r", VERSION)
  tryCatch({
    # `data` is relative to grid/, so a generated cell and a crate fixture
    # are read by the same line. Built with read_csv_path/build_data (not
    # common.R's grid_read_cell): grid_read_cell's factor() call sorts by
    # the session locale and its data comes through read.csv's own (not
    # correctly rounded) numeric parser, neither of which the round-off
    # gate against glmm.rs can tolerate.
    csv <- read_csv_path(file.path(env$grid_dir, cell[["data"]]))
    df <- build_data(csv$header, csv$rows, unlist(cell[["factors"]]) %||% character(0))
    column <- function(name) strtod_parse(vapply(csv$rows,
      function(r) r[[match(name, csv$header)]], ""))

    gaussian <- identical(cell[["family"]], "gaussian")

    # Trial counts vs prior weights -- the two manifest fields, mutually
    # exclusive by construction (gen_manifest.R asserts it). `[[`, not `$`:
    # `$` partial-matches, so `cell$weights` would resolve to `weights_col`
    # on a prior-weight cell (weights absent) and silently double-apply it.
    w_name <- cell[["weights"]]
    if (!is.null(w_name)) {
      sizes <- column(w_name)
      df$prop <- column(cell[["response"]]) / sizes
      df$validation_wts <- sizes
    } else {
      wc <- cell[["weights_col"]]
      if (!is.null(wc)) df$validation_wts <- column(wc)
    }
    # THE OFFSET, applied exactly once: glmm_formula carries no offset(...) term.
    oc <- cell[["offset_col"]]
    if (!is.null(oc)) df$validation_off <- column(oc)

    fml <- as.formula(cell[["glmm_formula"]])
    fam <- build_family(cell)
    k   <- if (is.null(cell[["nagq"]])) 1L else as.integer(cell[["nagq"]])

    # Reading and typing the data frame stays outside the timed region: the
    # other engines of this grid are already holding a typed data frame when
    # their own timer starts. Everything from the CSV read on is inside this
    # tryCatch, mirroring common.rs's catch_unwind scope -- a grid corner that
    # breaks the port anywhere in setup (a bad formula, an unsupported
    # family/link pair) is one failed cell, not a run-ending crash.
    timed <- grid_time(env, function() do_fit(df, fml, fam, "hessian", k))
    fh <- timed$value
    if (!is.null(timed$wall_seconds)) rec$wall_seconds <- timed$wall_seconds
    rec$fits_per_sample <- timed$fits_per_sample
    if (gaussian) {
      rec$se_rx <- nums(fh$se)
    } else {
      fr <- do_fit(df, fml, fam, "rx", k)
      rec$se_hessian <- nums(fh$se); rec$se_rx <- nums(fr$se)
    }
    rec$coef_names <- as.list(names(fh$beta))
    rec$beta <- nums(fh$beta)
    rec$varcomp <- varcomp(fh)  # by group NAME; no ref_order
    rec <- grid_set_loglik(rec, fh$loglik)
    # Assign only on the branch that applies; grid_record() already seeded
    # sigma/nb_theta at NULL, and `rec$sigma <- NULL` on the other branch
    # would DELETE the key (R list `$<-` semantics), not merely keep it null.
    if (cell[["family"]] %in% c("gaussian", "gamma")) rec$sigma <- num_scalar(sqrt(fh$dispersion))
    if (identical(cell[["family"]], "negativebinomial")) rec$nb_theta <- num_scalar(fh$dispersion)
    rec$n_eval <- fh$n_eval
    rec$converged <- isTRUE(fh$converged)
    rec$singular <- isTRUE(fh$singular)
    rec$status <- if (isTRUE(fh$converged)) "ok" else "engine-fail"
    if (!isTRUE(fh$converged)) rec$message <- not_converged_text(fh)
  }, error = function(e) rec <<- grid_fail(rec, conditionMessage(e)))
  rec
}

con <- grid_open_out(env)
for (cell in grid_cells(env)) {
  rec <- fit_one(cell)
  grid_write(con, rec)
  cat(sprintf("glmm_r  %-42s  %s\n", cell[["cell"]], rec$status))
}
close(con)
