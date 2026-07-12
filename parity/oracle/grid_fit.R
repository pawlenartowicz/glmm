#!/usr/bin/env Rscript
# lme4 lazy-anchor runner for the optimizer-grid campaign: fits ONLY the cells
# named in GRID_TODO (mismatch cells + audit sample from analyze_grid.R) or
# GRID_ONLY. One JSONL line per cell, resume-safe. Eval cap: the manifest's
# pre-registered max_fun, via optCtrl$maxeval for lmer's default nloptwrap
# (gaussian) or optCtrl$maxfun for glmer's default bobyqa/Nelder_Mead pair
# (binomial/poisson) -- the two optimizer families use different control names.
suppressMessages({ library(lme4); library(jsonlite) })

parity_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
manifest <- fromJSON(Sys.getenv("GRID_MANIFEST",
  file.path(parity_dir, "manifest_grid.json")), simplifyDataFrame = FALSE)
out_path <- Sys.getenv("GRID_OUT",
  file.path(parity_dir, "results", "grid", "lme4_anchor.jsonl"))
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
tag <- Sys.getenv("GRID_CONFIG_TAG", "")

todo_file <- Sys.getenv("GRID_TODO", "")
todo <- if (nzchar(todo_file)) readLines(todo_file) else
  strsplit(Sys.getenv("GRID_ONLY", ""), ",")[[1]]
todo <- todo[nzchar(todo)]
if (length(todo) == 0) stop("no cells: set GRID_TODO or GRID_ONLY")

done <- character(0)
if (file.exists(out_path)) {
  lines <- readLines(out_path)
  done <- unlist(lapply(lines[nzchar(lines)], function(l)  # kill -9 can truncate the last line
    tryCatch(fromJSON(l)$case_id, error = function(e) NULL)))
}

cells <- Filter(function(c) c$case_id %in% todo && !(c$case_id %in% done),
                manifest$cells)
con <- file(out_path, open = "a")
for (cell in cells) {
  base <- list(case_id = cell$case_id, seed = cell$seed, engine = "lme4",
               config_tag = tag)
  t0 <- Sys.time()
  rec <- tryCatch({
    df <- read.csv(file.path(parity_dir, "data_simulated", "grid",
                             paste0(cell$case_id, ".csv")))
    for (f in unlist(cell$factors)) df[[f]] <- factor(df[[f]])
    fm <- as.formula(cell$r_formula)
    m <- if (cell$family == "gaussian") {
      # REML iff the manifest cell says so -- mirrors fit.R's `isTRUE(spec$reml)`
      # and grid_fit.jl's `get(cell, :reml, false) === true`.
      lmer(fm, data = df, REML = isTRUE(cell$reml),
           control = lmerControl(optCtrl = list(maxeval = cell$max_fun)))
    } else {
      fam <- switch(cell$family, binomial = binomial(), poisson = poisson())
      # glmer's default optimizer pair (bobyqa, Nelder_Mead; both minqa-based)
      # honors "maxfun", not "maxeval" -- passing maxeval here is a silent no-op
      # (warns "unused control arguments ignored", cap never applied). lmer's
      # default nloptwrap honors maxeval, hence the split by family.
      glmer(fm, data = df, family = fam, nAGQ = 1,
            control = glmerControl(tolPwrss = 1e-13,
                                   optCtrl = list(maxfun = cell$max_fun)))
    }
    conv <- is.null(m@optinfo$conv$lme4$messages) ||
            length(m@optinfo$conv$lme4$messages) == 0
    feval <- as.integer(m@optinfo$feval)
    status <- if (feval >= cell$max_fun) "maxeval" else if (conv) "ok" else "engine-fail"
    c(base, list(
      optimizer = paste(unlist(m@optinfo$optimizer), collapse = "+"),
      n_eval = feval, converged = conv, singular = isSingular(m),
      deviance = as.numeric(-2 * logLik(m)),
      beta = I(unname(fixef(m))),
      se = I(unname(sqrt(diag(as.matrix(vcov(m)))))),
      status = status))
  }, error = function(e) c(base, list(
      optimizer = "", n_eval = 0L, converged = FALSE, singular = FALSE,
      deviance = NULL, beta = I(numeric(0)), se = I(numeric(0)),
      status = "engine-fail")))
  rec$wall_seconds <- as.numeric(Sys.time() - t0, units = "secs")
  writeLines(toJSON(rec, auto_unbox = TRUE, digits = NA, null = "null"), con)
  flush(con)
}
close(con)
