#!/usr/bin/env Rscript
# lme4/GLMMadaptive oracle runner for the optimizer-grid campaign. Two manifests
# share this file (env-var-driven, mirrors fit.rs's GRID_MANIFEST/GRID_OUT):
#
# - manifest.json (default): lazy anchor -- fits ONLY the cells named in
#   GRID_TODO (mismatch cells + audit sample from analyze.R) or GRID_ONLY.
# - estimate-grid's manifest.json: three engine branches selected off
#   the manifest's per-cell `nagq`/`structure` -- lmer/glmer Laplace for the 477
#   AGQ-ineligible cells (as the anchor above), glmer(nAGQ=k) for the 15 `int1`
#   scalar-AGQ cells, GLMMadaptive::mixed_model(nAGQ=k) for the 18 `q2s`
#   vector-AGQ cells (glmer refuses nAGQ>1 for vector REs).
#   Every branch also records varcomp (theta hat), the one new diligent quantity.
#
# One JSONL line per cell, resume-safe. Eval cap: the manifest's pre-registered
# max_fun, via optCtrl$maxeval for lmer's default nloptwrap (gaussian) or
# optCtrl$maxfun for glmer's default bobyqa/Nelder_Mead pair (binomial/poisson)
# -- the two optimizer families use different control names. GLMMadaptive has no
# equivalent eval cap knob; its own iteration controls are used instead (below).
suppressMessages({ library(lme4); library(jsonlite); library(GLMMadaptive) })

here_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
manifest <- fromJSON(Sys.getenv("GRID_MANIFEST",
  file.path(here_dir, "manifest.json")), simplifyDataFrame = FALSE)
out_path <- Sys.getenv("GRID_OUT",
  file.path(here_dir, "results", "lme4_anchor.jsonl"))
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

# theta hat: VarCorr -> per grouping factor, term names, stddevs, correlation
# matrix (diligent run, spec Part 6). Mirrors engines/lme4.R::varcomp_of and
# engines/goldens_agq.R::varcomp_of (the sd_se-less form -- the theta-Hessian
# SE goldens_agq.R attaches is a per-rung numDeriv Hessian, too expensive to
# run per grid cell here) so the schema matches the glmm/GLMMadaptive
# `varcomp` field the analysis join reads.
varcomp_of <- function(m) {
  vc <- VarCorr(m)
  lapply(names(vc), function(g) {
    block <- vc[[g]]
    sd <- attr(block, "stddev")
    corr <- attr(block, "correlation")
    list(group = g, terms = I(names(sd)),
         stddev = I(unname(sd)), corr = unname(corr))
  })
}

# GLMMadaptive tightened controls (goldens_agq.R precedent): the package's
# defaults under-converge on low-information rungs; update_GH_every=1 re-adapts
# the quadrature grid every iteration, the like-for-like convention with glmm.
MA_CTRL <- list(iter_EM = 300, iter_qN_outer = 60,
                tol1 = 1e-8, tol2 = 1e-10, tol3 = 1e-12, update_GH_every = 1)

con <- file(out_path, open = "a")
for (cell in cells) {
  is_agq <- !is.null(cell$nagq)
  is_vec_agq <- is_agq && identical(cell$structure, "q2s")
  engine <- if (is_vec_agq) "GLMMadaptive" else "lme4"
  base <- list(case_id = cell$case_id, seed = cell$seed, engine = engine,
               config_tag = tag)
  t0 <- Sys.time()
  rec <- tryCatch({
    df <- read.csv(file.path(here_dir, "data",  # campaign-local, prep.R output
                             paste0(cell$case_id, ".csv")))
    for (f in unlist(cell$factors)) df[[f]] <- factor(df[[f]])
    if (is_vec_agq) {
      # GLMMadaptive vector-AGQ branch (18 q2s cells). Aggregated-binomial
      # cells (cell$weights == "size") use `ma_fixed = cbind(incidence, size -
      # incidence) ~ ...` -- GLMMadaptive's actual binomial-trials form
      # (verified against a live fit); its `weights=` arg is a per-CLUSTER
      # replicate multiplier, NOT a per-row trial count like glm/glmer's, so
      # it cannot carry the aggregated-binomial convention -- no weights arg
      # needed here, cbind carries it directly.
      #
      # KNOWN COVERAGE GAP: glmm's own aggregated-binomial lowering always
      # routes through `FitOptions.weights` (the prior-weights/trial-count
      # convention, `common.rs::lower_dataset_generic`'s `prop`
      # synthesis), and `nagq>1 + weights` is a locked-refused combination on
      # the glmm side (`FitOptions.weights with nAGQ > 1 is not supported`,
      # src/fit/mod.rs assert_model_shape, full-AGQ spec Part 2). So these 6
      # `bina_q2s_*` cells fit HERE (a legitimate oracle-side AGQ reference,
      # confirmed above) but glmm engine-fails on the matching cell -- only
      # 12/18 q2s cells (and all 15 int1) are glmm-vs-oracle joinable under
      # AGQ; the 6 bina_q2s_* GLMMadaptive rows are oracle-only until/unless
      # glmm's weights+AGQ gate is revisited (out of this task's scope).
      fam <- switch(cell$family, binomial = binomial(), poisson = poisson())
      m <- mixed_model(fixed = as.formula(cell$ma_fixed),
                       random = as.formula(cell$ma_random),
                       data = df, family = fam,
                       nAGQ = as.integer(cell$nagq), control = MA_CTRL)
      c(base, list(
        optimizer = "GLMMadaptive", n_eval = NA_integer_,
        converged = isTRUE(m$converged), singular = FALSE,
        deviance = NA_real_,  # not comparable to glmer's devfun scale (spec Part 6)
        beta = I(unname(fixef(m))),
        se = I(unname(sqrt(diag(vcov(m, parm = "fixed-effects"))))),
        varcomp = list(list(
          group  = sub("^.*\\|\\s*", "", cell$ma_random),
          terms  = I(colnames(m$D)),
          stddev = I(unname(sqrt(diag(m$D)))),
          corr   = unname(cov2cor(m$D)))),
        status = if (isTRUE(m$converged)) "ok" else "engine-fail"))
    } else if (cell$family == "gaussian") {
      # REML iff the manifest cell says so -- mirrors engines/lme4.R's `isTRUE(spec$reml)`
      # and fit.jl's `get(cell, :reml, false) === true`.
      m <- lmer(as.formula(cell$r_formula), data = df, REML = isTRUE(cell$reml),
           control = lmerControl(optCtrl = list(maxeval = cell$max_fun)))
      msgs <- as.character(unlist(m@optinfo$conv$lme4$messages))
      conv <- length(msgs) == 0
      feval <- as.integer(m@optinfo$feval)
      status <- if (feval >= cell$max_fun) "maxeval" else if (conv) "ok" else "engine-fail"
      c(base, list(
        optimizer = paste(unlist(m@optinfo$optimizer), collapse = "+"),
        n_eval = feval, converged = conv, singular = isSingular(m),
        # `converged` collapses every lme4 message to one bit, but the messages
        # are not one thing: the singular-fit note, a max-gradient warning and a
        # Hessian warning all land in the same FALSE. Downstream triage of a
        # non-converged cell cannot proceed without knowing which one fired, so
        # the text is carried through verbatim. Mirrored in the glmer branch
        # below -- change together.
        messages = I(msgs),
        deviance = as.numeric(-2 * logLik(m)),
        beta = I(unname(fixef(m))),
        se = I(unname(sqrt(diag(as.matrix(vcov(m)))))),
        varcomp = varcomp_of(m),
        status = status))
    } else {
      # glmer branch: Laplace (nAGQ=1, 477 ineligible cells) or scalar AGQ
      # (nAGQ=cell$nagq, the 15 int1 cells) -- glmer's default optimizer pair
      # (bobyqa, Nelder_Mead; both minqa-based) honors "maxfun", not "maxeval"
      # (passing maxeval here is a silent no-op -- warns "unused control
      # arguments ignored", cap never applied). lmer's default nloptwrap
      # honors maxeval, hence the split by family (above).
      fam <- switch(cell$family, binomial = binomial(), poisson = poisson())
      m <- glmer(as.formula(cell$r_formula), data = df, family = fam,
            nAGQ = if (is_agq) as.integer(cell$nagq) else 1L,
            # GRID_TOLPWRSS: PIRLS tolerance override for the seed-extension
            # passes (estimate-grid seedext_gen.R study); default keeps the
            # 1e-13 the frozen campaign results were fit at.
            control = glmerControl(
              tolPwrss = as.numeric(Sys.getenv("GRID_TOLPWRSS", "1e-13")),
              optCtrl = list(maxfun = cell$max_fun)))
      msgs <- as.character(unlist(m@optinfo$conv$lme4$messages))
      conv <- length(msgs) == 0
      feval <- as.integer(m@optinfo$feval)
      status <- if (feval >= cell$max_fun) "maxeval" else if (conv) "ok" else "engine-fail"
      c(base, list(
        optimizer = paste(unlist(m@optinfo$optimizer), collapse = "+"),
        n_eval = feval, converged = conv, singular = isSingular(m),
        messages = I(msgs),   # mirrors the lmer branch above -- change together
        deviance = as.numeric(-2 * logLik(m)),
        beta = I(unname(fixef(m))),
        se = I(unname(sqrt(diag(as.matrix(vcov(m)))))),
        varcomp = varcomp_of(m),
        status = status))
    }
  }, error = function(e) c(base, list(
      optimizer = "", n_eval = 0L, converged = FALSE, singular = FALSE,
      deviance = NULL, beta = I(numeric(0)), se = I(numeric(0)),
      varcomp = list(), status = "engine-fail")))
  rec$wall_seconds <- as.numeric(Sys.time() - t0, units = "secs")
  writeLines(toJSON(rec, auto_unbox = TRUE, digits = NA, na = "null"), con)
  flush(con)
}
close(con)
