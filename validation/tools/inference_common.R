# Shared setup for the inference reference generators (the scripts that write
# validation/goldens/inference/<kind>/<name>.json for the entries of the
# manifest's `inference_references` block). Sourced, never run on its own.
#
# THE ORACLE IS SACRED. The JSONs your generator writes are the frozen reference
# the crate's inference tests are validated against. On any glmm disagreement,
# glmm is presumed wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN OVERWRITES EVERY REFERENCE OF ITS KIND. Use VALIDATION_ONLY for
# anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/<generator>.R
# runs only the named entries (a name that is not an entry of the generator's
# kind stops the run). An entry carrying `pending_reference` is registered but
# not frozen: a bare run skips it and prints its name, and it is generated only
# when named in VALIDATION_ONLY; the `pending_reference` field is then dropped
# from the manifest entry in the same change.
#
# THE GENERATOR CONTRACT. Every generator that sources this file:
#   1. computes suite_dir (validation/) from the `--file=` of commandArgs(FALSE),
#      as tools/goldens_agq.R does, and sources this file from there;
#   2. calls inference_setup() at top level, before any other work (it may
#      re-execute the script and exit; see the function);
#   3. calls assert_inference_versions(), then assert_extra_versions(<pkgs>) on
#      any further package it uses, keeping the names it returns;
#   4. loops over select_entries(<kind>), one reference per entry, reading an
#      entry's `data` with read_entry_data(entry) (which applies its `factors`)
#      and fitting an entry's `members` with fit_member (or fit_member_sum, the
#      same fit with contr.sum on named factors), both built on member_call and
#      family_call;
#   5. attaches provenance(<its own file name>, extra = <those names>) to each
#      result, as `provenance`;
#   6. stops on a number that is not finite (check_finite) before it writes the
#      result with write_reference(obj, file.path(suite_dir, "goldens",
#      "inference", <kind>, paste0(<name>, ".json"))), which sets `name`.

INFERENCE_THREAD_VARS <- c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                           "MKL_NUM_THREADS", "FLEXIBLAS_NUM_THREADS")
INFERENCE_PKGS <- c("lmerTest", "RLRsim", "car", "numDeriv", "MASS", "jsonlite")
INFERENCE_KINDS <- c("lmertest", "rlrt", "anova", "predict", "posthoc", "effectsize")

# This file's own directory (validation/tools), recorded while source() runs it,
# by the same sys.frame(i)$ofile walk as grid/runs.R::runs_script_dir. The
# functions below locate validation/ from it, never from the working directory
# or `--file=`, so they also work for a caller outside validation/.
INFERENCE_COMMON_DIR <- local({
  dir <- NULL
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (is.character(of) && length(of) == 1L && file.exists(of) &&
        identical(basename(of), "inference_common.R")) {
      dir <- dirname(normalizePath(of))
      break
    }
  }
  if (is.null(dir)) {
    stop("inference_common.R: cannot locate its own directory; load it with ",
         "source(\"<path>/validation/tools/inference_common.R\")", call. = FALSE)
  }
  dir
})

# Threads to 1, then attach lme4 and the pinned packages.
#
# Setting the four variables with Sys.setenv() here would come too late:
# OpenBLAS and OpenMP read them when the library loads, which under FlexiBLAS is
# at R start-up (grid/engines/common.R::grid_threads says the same of its
# in-process mirror). So a process started without all four at "1" re-executes
# the calling script under Rscript with all four set (VALIDATION_ONLY and the
# script's arguments pass through) and exits with the child's status. Only the
# script and its arguments are forwarded, not Rscript's own options (such as
# --vanilla): the child starts as a plain `Rscript <script>` would.
inference_setup <- function() {
  if (!all(Sys.getenv(INFERENCE_THREAD_VARS) == "1")) {
    if (nzchar(Sys.getenv("GLMM_INFERENCE_REEXEC"))) {
      stop("inference_setup: re-executed with ", paste(INFERENCE_THREAD_VARS, collapse = ", "),
           " set to 1, but they still read ",
           paste(sprintf("%s=\"%s\"", INFERENCE_THREAD_VARS, Sys.getenv(INFERENCE_THREAD_VARS)),
                 collapse = ", "), call. = FALSE)
    }
    file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
    if (length(file_arg) != 1L) {
      stop("inference_setup: OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and ",
           "FLEXIBLAS_NUM_THREADS must all be 1, and without a script file (`--file=`) ",
           "this process cannot re-execute itself with them set. Set all four to 1 on ",
           "the command line, e.g. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ",
           "MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1 Rscript -e '...'", call. = FALSE)
    }
    status <- system2(file.path(R.home("bin"), "Rscript"),
                      shQuote(c(sub("^--file=", "", file_arg), commandArgs(TRUE))),
                      env = c(paste0(INFERENCE_THREAD_VARS, "=1"), "GLMM_INFERENCE_REEXEC=1"))
    quit(status = status, save = "no")
  }
  for (pkg in c("lme4", INFERENCE_PKGS)) {
    suppressPackageStartupMessages(library(pkg, character.only = TRUE))
  }
  invisible(NULL)
}

# lme4, the six pins and afex against grid/versions.json; a mismatch stops the
# run. afex is asserted here but neither attached by inference_setup() nor in
# provenance()'s default list: only goldens_anova.R's likelihood-ratio entries
# use it, and they name it in their own provenance.
assert_inference_versions <- function() {
  grid_dir <- file.path(dirname(INFERENCE_COMMON_DIR), "grid")
  source(file.path(grid_dir, "engines", "versions.R"), local = TRUE)
  V <- grid_versions(grid_dir)
  for (pkg in c("lme4", INFERENCE_PKGS, "afex")) assert_pkg_version(pkg, V[[pkg]])
  invisible(NULL)
}

# The packages `pkgs` a generator uses beyond the ones above, against
# grid/versions.json; a mismatch stops the run. Returns `pkgs`, the list that
# generator hands to provenance(extra = ...).
assert_extra_versions <- function(pkgs) {
  grid_dir <- file.path(dirname(INFERENCE_COMMON_DIR), "grid")
  source(file.path(grid_dir, "engines", "versions.R"), local = TRUE)
  V <- grid_versions(grid_dir)
  for (pkg in pkgs) assert_pkg_version(pkg, V[[pkg]])
  pkgs
}

# Stops unless every number in `x` (a nested list) is finite and no leaf holds
# an NA, so no Inf, NaN or NA reaches a JSON. `name` and `path` label the leaf.
check_finite <- function(x, name, path = "") {
  if (is.list(x)) {
    keys <- names(x)
    for (i in seq_along(x)) {
      key <- if (is.null(keys) || !nzchar(keys[i])) as.character(i) else keys[i]
      check_finite(x[[i]], name, paste0(path, "/", key))
    }
  } else if (anyNA(x) || (is.numeric(x) && !all(is.finite(x)))) {
    stop(sprintf("%s: %s is not finite", name, path), call. = FALSE)
  }
  invisible(x)
}

# tools/goldens_glm_band.py has a Python copy of provenance, write_reference and
# select_entries; change it together with these three.
#
# What produced a reference, in a fixed key order. No date and no host name, so a
# re-run on the same machine is byte-identical. `extra` names packages beyond
# lme4 and the six pins that the generator also uses (e.g. "glmmTMB").
provenance <- function(generator, extra = character()) {
  pkgs <- c("lme4", INFERENCE_PKGS, extra)
  list(
    r_version = as.character(getRversion()),
    packages = setNames(lapply(pkgs, function(p) as.character(utils::packageVersion(p))), pkgs),
    rng_kind = RNGkind(),
    blas = extSoftVersion()[["BLAS"]],
    lapack = La_library(),
    threads = as.list(Sys.getenv(INFERENCE_THREAD_VARS)),
    generator = generator
  )
}

# Sets `name` to the file stem, then writes the JSON as tools/goldens_agq.R does.
write_reference <- function(obj, path) {
  obj$name <- tools::file_path_sans_ext(basename(path))
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  write(jsonlite::toJSON(obj, auto_unbox = TRUE, pretty = TRUE, digits = NA, na = "null"), path)
}

# The data frame of an entry: its `data` CSV (a path relative to validation/),
# read as tools/goldens_agq.R::read_dataset reads a rung's CSV, with each column
# named in the entry's optional `factors` converted by factor() in the same way.
# Stops on a fit-free entry (`data` null) and on a `factors` name that is not a
# column of the CSV.
read_entry_data <- function(entry) {
  if (is.null(entry$data)) {
    stop(sprintf("read_entry_data: entry %s has no `data` (a fit-free entry)", entry$name),
         call. = FALSE)
  }
  df <- read.csv(file.path(dirname(INFERENCE_COMMON_DIR), entry$data),
                 stringsAsFactors = FALSE)
  factors <- unlist(entry$factors)
  missing <- setdiff(factors, names(df))
  if (length(missing) > 0L) {
    stop(sprintf("read_entry_data: entry %s names factors that are not columns of %s: %s",
                 entry$name, entry$data, paste(missing, collapse = ", ")), call. = FALSE)
  }
  for (f in factors) df[[f]] <- factor(df[[f]])
  df
}

# The `inference_references` entries of `kind` this run generates: the names in
# VALIDATION_ONLY when it is set (any name that is not an entry of `kind` stops
# the run), else every entry without `pending_reference` (each skipped one is
# printed).
select_entries <- function(kind) {
  if (!kind %in% INFERENCE_KINDS) {
    stop(sprintf("select_entries: unknown kind \"%s\" (one of %s)", kind,
                 paste(INFERENCE_KINDS, collapse = ", ")), call. = FALSE)
  }
  manifest <- jsonlite::fromJSON(file.path(dirname(INFERENCE_COMMON_DIR), "manifest.json"),
                                 simplifyDataFrame = FALSE)
  if (!"inference_references" %in% names(manifest)) {
    stop("select_entries: manifest.json has no `inference_references` block", call. = FALSE)
  }
  entries <- Filter(function(e) identical(e$kind, kind), manifest$inference_references)
  names_of <- vapply(entries, `[[`, "", "name")
  only <- Sys.getenv("VALIDATION_ONLY")
  if (nzchar(only)) {
    keep <- strsplit(only, ",", fixed = TRUE)[[1]]
    unknown <- setdiff(keep, names_of)
    if (length(unknown) > 0L) {
      stop(sprintf("select_entries: VALIDATION_ONLY names no %s entry: %s", kind,
                   paste(unknown, collapse = ", ")), call. = FALSE)
    }
    return(entries[names_of %in% keep])
  }
  pending <- vapply(entries, function(e) !is.null(e$pending_reference), NA)
  for (e in entries[pending]) {
    cat(sprintf("skip %s (pending_reference: %s)\n", e$name, e$pending_reference))
  }
  entries[!pending]
}

# The family call of a member, in the grid's family spelling.
family_call <- function(family, link) {
  fun <- switch(family, gaussian = "gaussian", binomial = "binomial", poisson = "poisson",
                gamma = "Gamma", stop("family_call: unsupported family ", family))
  str2lang(sprintf('%s(link = "%s")', fun, link))
}

# The call that fits member `mem` on `data`, and the environment to evaluate it
# in (holding `data`, so update() inside ranova finds it). Built as a call, not
# by do.call, so the model's stored call names the weights column.
member_call <- function(mem, data) {
  env <- new.env(parent = globalenv())
  env$data <- data
  f <- as.formula(mem$formula, env = env)
  w <- if (is.null(mem$weights)) NULL else as.name(mem$weights)
  ctrl <- if (is.null(mem$tolPwrss)) NULL else
    bquote(lme4::glmerControl(tolPwrss = .(mem$tolPwrss)))
  cl <- switch(mem$fun,
    "lme4::lmer" = bquote(lme4::lmer(.(f), data = data, REML = .(mem$REML), weights = .(w))),
    "stats::lm" = bquote(stats::lm(.(f), data = data, weights = .(w))),
    "stats::glm" = bquote(stats::glm(.(f), family = .(family_call(mem$family, mem$link)),
                                     data = data, weights = .(w))),
    "lme4::glmer" = bquote(lme4::glmer(.(f), family = .(family_call(mem$family, mem$link)),
                                       data = data, weights = .(w), nAGQ = .(mem$nAGQ),
                                       control = .(ctrl))),
    "MASS::glm.nb" = {
      if (!identical(mem$link, "log")) stop("member_call: glm.nb is fitted with its log link")
      bquote(MASS::glm.nb(.(f), data = data, weights = .(w)))
    },
    "lme4::glmer.nb" = {
      if (!identical(mem$link, "log")) stop("member_call: glmer.nb is fitted with its log link")
      bquote(lme4::glmer.nb(.(f), data = data, weights = .(w), nAGQ = .(mem$nAGQ),
                            control = .(ctrl)))
    },
    "glmmTMB::glmmTMB" = bquote(glmmTMB::glmmTMB(.(f),
                                                 family = .(family_call(mem$family, mem$link)),
                                                 data = data, weights = .(w),
                                                 REML = .(mem$REML))),
    stop("member_call: unsupported fun ", mem$fun))
  list(call = cl, env = env)
}

fit_member <- function(mem, data) {
  mc <- member_call(mem, data)
  eval(mc$call, mc$env)
}

# Member `mem` refitted with contr.sum on the factor columns `cols`: the call of
# fit_member plus `contrasts`.
fit_member_sum <- function(mem, data, cols) {
  mc <- member_call(mem, data)
  mc$call$contrasts <- setNames(rep(list("contr.sum"), length(cols)), cols)
  eval(mc$call, mc$env)
}
