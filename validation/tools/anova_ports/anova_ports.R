#!/usr/bin/env Rscript
# The R front end of the anova port gate: the call list of anova_ports.rs,
# made through the R port (anova.fastglmm, compare), each table written to
# stdout as one JSONL row per cell. Mirrors anova_ports.rs and anova_ports.py
# call for call: change together. The call list and the cell format are
# documented in anova_ports.rs.
#
# Run from the repository root:
#   Rscript validation/tools/anova_ports/anova_ports.R > anova_r.jsonl
suppressMessages({ library(fastglmm); library(jsonlite) })
here <- dirname(normalizePath(sub("--file=", "",
  grep("--file=", commandArgs(FALSE), value = TRUE))))
validation_dir <- normalizePath(file.path(here, "..", ".."))

PAIR_CASES <- c("random_pair", "ml_pair", "ols_pair", "glm_pair", "glmm_pair",
                "glmm_vs_glm", "gamma_glmm_vs_glm")

# The models whose per-term tables are compared, by label.
TERM_MODELS <- list(
  lmer_sim_factorial = list(data = "data/simulated/sim_factorial.csv",
       factors = c("subj", "g", "h"), formula = "y ~ g * h + x + (1 + x | subj)",
       family = "gaussian", link = "identity"),
  lm_sim_factorial = list(data = "data/simulated/sim_factorial.csv",
       factors = c("subj", "g", "h"), formula = "y ~ g * h + x",
       family = "gaussian", link = "identity"),
  glmer_sim_factorial_bern = list(data = "data/simulated/sim_factorial_bern.csv",
       factors = c("subj", "g", "h"), formula = "y ~ g * h + x + (1 | subj)",
       family = "binomial", link = "logit"),
  glm_sim_factorial_bern = list(data = "data/simulated/sim_factorial_bern.csv",
       factors = c("subj", "g", "h"), formula = "y ~ g * h + x",
       family = "binomial", link = "logit"),
  glmer_sim_binomial_slope1 = list(data = "data/simulated/sim_binomial_slope1.csv",
       factors = "g", formula = "y ~ x + (1 + x | g)", family = "binomial", link = "logit"),
  glmer_cbpp = list(data = "data/empirical/cbpp.csv", factors = c("herd", "period"),
       formula = "cbind(incidence, size - incidence) ~ period + (1 | herd)",
       family = "binomial", link = "logit"),
  glmer_poisson_grouseticks = list(data = "data/empirical/grouseticks.csv",
       factors = c("YEAR", "BROOD"), formula = "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)",
       family = "poisson", link = "log"),
  glmer_nb_sim_nb = list(data = "data/simulated/sim_nb.csv", factors = c("cluster", "grp"),
       formula = "y ~ 1 + x + grp + (1 | cluster)", family = "negativebinomial",
       link = "log"),
  glmer_gamma_sim_gamma = list(data = "data/simulated/sim_gamma.csv",
       factors = c("cluster", "grp"), formula = "y ~ 1 + x + grp + (1 | cluster)",
       family = "gamma", link = "log"))

# One fit's per-term calls, as anova_ports.rs's TERM_CALLS. The fit is made
# with anova_test = "LRT" for test = "LRT" and at the default "auto"
# otherwise; a call is named terms:<label>:<type>, with :<test> appended when
# named.
term_call <- function(model, types, test, named) {
  list(model = model, types = types, test = test, named = named)
}
TERM_CALLS <- list(
  term_call("lmer_sim_factorial", c("III", "II"), "auto", FALSE),
  term_call("lm_sim_factorial", c("III", "II"), "auto", FALSE),
  term_call("glmer_sim_factorial_bern", c("III", "II"), "auto", FALSE),
  term_call("glm_sim_factorial_bern", c("III", "II"), "auto", FALSE),
  term_call("glmer_sim_factorial_bern", c("III", "II"), "LRT", TRUE),
  term_call("glm_sim_factorial_bern", c("III", "II"), "LRT", TRUE),
  term_call("glmer_cbpp", c("III", "II"), "LRT", TRUE),
  term_call("glmer_sim_binomial_slope1", "III", "LRT", TRUE),
  term_call("glmer_poisson_grouseticks", "III", "auto", TRUE),
  term_call("glmer_poisson_grouseticks", "III", "Wald", TRUE),
  term_call("glmer_nb_sim_nb", "III", "auto", TRUE),
  term_call("glmer_gamma_sim_gamma", "III", "auto", TRUE))

MISSING_ROW <- 5L  # 1-based row of wls_basic.csv whose x2 is made missing

# Correctly-rounded string to double parse through the C library's strtod, as
# grid/engines/glmm_r.R parses: R's own as.numeric and read.csv are not
# correctly rounded, and a one-ulp input difference can move a fit past the
# port band.
Rcpp::cppFunction('
NumericVector strtod_parse(CharacterVector s) {
  int n = s.size();
  NumericVector out(n);
  for (int i = 0; i < n; i++) out[i] = std::strtod(CHAR(STRING_ELT(s, i)), nullptr);
  return out;
}', includes = "#include <cstdlib>")

# The columns of one CSV, typed as the grid's engines type them: a column
# named in `factors`, or one with a cell that is not a number, is a factor
# with its distinct labels in byte order (radix sort) as levels.
read_data <- function(path, factors) {
  lines <- readLines(file.path(validation_dir, path), warn = FALSE)
  lines <- lines[nzchar(trimws(lines))]
  split <- function(s) trimws(gsub('^"|"$', "", trimws(strsplit(s, ",", fixed = TRUE)[[1]])))
  header <- split(lines[1])
  rows <- lapply(lines[-1], split)
  out <- list()
  for (j in seq_along(header)) {
    cells <- vapply(rows, function(r) r[[j]], "")
    numeric <- !anyNA(suppressWarnings(as.numeric(cells)))
    out[[header[j]]] <- if (header[j] %in% factors || !numeric) {
      factor(cells, levels = sort(unique(cells), method = "radix"))
    } else {
      strtod_parse(cells)
    }
  }
  data.frame(out, check.names = FALSE)
}

# `cbind(s, t - s) ~ ...` as `cbind(s, failures) ~ ...` with the column
# `failures = t - s` added to `data`; any other formula as it is.
aggregated <- function(formula, data) {
  if (!startsWith(formula, "cbind(")) return(list(formula = formula, data = data))
  rest <- substring(formula, nchar("cbind(") + 1L)
  close <- regexpr(")", rest, fixed = TRUE)
  inner <- substring(rest, 1L, close - 1L)
  tail <- substring(rest, close + 1L)
  comma <- regexpr(",", inner, fixed = TRUE)
  succ <- trimws(substring(inner, 1L, comma - 1L))
  fail <- substring(inner, comma + 1L)
  minus <- regexpr("-", fail, fixed = TRUE)
  if (minus < 0L) return(list(formula = formula, data = data))
  total <- trimws(substring(fail, 1L, minus - 1L))
  sub_col <- trimws(substring(fail, minus + 1L))
  data$failures <- data[[total]] - data[[sub_col]]
  list(formula = sprintf("cbind(%s, failures)%s", succ, tail), data = data)
}

# The family argument fastglmm() takes, as grid/engines/glmm_r.R builds it.
family_arg <- function(family, link) {
  switch(family,
    gaussian = stats::gaussian(),
    binomial = stats::binomial(link = link),
    poisson = stats::poisson(),
    gamma = stats::Gamma(link = link),
    negativebinomial = "negativebinomial",
    stop("unsupported family: ", family))
}

# A number as it is; a non-finite one as "NaN", "Inf" or "-Inf". R's NA (the
# R port's empty cell) is written "NaN", as the other two front ends write it.
cell_value <- function(v) {
  if (is.finite(v)) v else if (is.na(v)) "NaN" else if (v > 0) "Inf" else "-Inf"
}

# One JSONL row per cell: `rows` in order, `cols` a named list of columns.
emit <- function(call, rows, cols) {
  for (i in seq_along(rows)) {
    for (j in seq_along(cols)) {
      v <- as.double(cols[[j]][[i]])
      cell <- list(call = call, row = rows[[i]], col = names(cols)[[j]],
                   row_index = i - 1L, col_index = j - 1L,
                   value = cell_value(v))
      writeLines(jsonlite::toJSON(cell, auto_unbox = TRUE, digits = I(17)))
    }
  }
}

# One text cell (a heading, a tag) at `at`, c(row_index, col_index), written
# as UTF-8 whatever the locale.
emit_text <- function(call, row, col, at, text) {
  cell <- list(call = call, row = row, col = col, row_index = at[[1]],
               col_index = at[[2]], text = enc2utf8(text))
  writeLines(enc2utf8(as.character(jsonlite::toJSON(cell, auto_unbox = TRUE))),
             useBytes = TRUE)
}

refused <- function(call, why) {
  message(sprintf("%s: refused: %s", call, conditionMessage(why)))
  emit(call, "", list(error = NA_real_))
}

# A port table (an "anova" data frame) as it is: its row names and columns.
port_table <- function(call, tab) {
  emit(call, rownames(tab), as.list(tab))
}

# One fit, the weights (a column of `data` or NULL) riding as the data column
# validation_wts, referenced by name as the grid's R engine does.
fit <- function(data, m, weights, anova_test = "auto") {
  fam <- family_arg(m$family, m$link)
  f <- stats::as.formula(m$formula)
  if (is.null(weights)) {
    fastglmm(f, data, fam, nAGQ = m$nagq, REML = m$reml, anova_test = anova_test)
  } else {
    data$validation_wts <- weights
    fastglmm(f, data, fam, weights = validation_wts, nAGQ = m$nagq, REML = m$reml,
             anova_test = anova_test)
  }
}

anova_call <- function(call, data, m, weights) {
  tryCatch(port_table(call, anova(fit(data, m[[1]], weights), fit(data, m[[2]], weights))),
           error = function(e) refused(call, e))
}

compare_call <- function(call, data, m, weights, nobs = FALSE) {
  fam <- family_arg(m[[1]]$family, m[[1]]$link)
  k <- max(m[[1]]$nagq, m[[2]]$nagq)
  f0 <- stats::as.formula(m[[1]]$formula)
  f1 <- stats::as.formula(m[[2]]$formula)
  cmp <- tryCatch({
    if (is.null(weights)) {
      compare(data, f0, f1, family = fam, nAGQ = k)
    } else {
      data$validation_wts <- weights
      compare(data, f0, f1, family = fam, weights = validation_wts, nAGQ = k)
    }
  }, error = function(e) e)
  if (inherits(cmp, "error")) return(refused(call, cmp))
  port_table(call, cmp$table)
  if (nobs) {
    emit(paste0(call, ":nobs"), c("fit0", "fit1"),
         list(nobs = c(cmp$fit0$nobs, cmp$fit1$nobs)))
  }
}

member <- function(m, data) {
  a <- aggregated(m$formula, data)
  list(m = list(formula = a$formula, family = m$family, link = m$link,
                reml = m$REML, nagq = if (is.null(m$nAGQ)) 1L else as.integer(m$nAGQ)),
       data = a$data)
}

pairs <- function() {
  manifest <- jsonlite::fromJSON(file.path(validation_dir, "manifest.json"),
                                 simplifyDataFrame = FALSE)
  for (e in manifest$inference_references) {
    if (!identical(e$kind, "anova") || !isTRUE(e$case %in% PAIR_CASES)) next
    data <- read_data(e$data, unlist(e$factors))
    a <- member(e$members[[1]], data)
    b <- member(e$members[[2]], a$data)
    m <- list(a$m, b$m)
    w <- e$members[[1]]$weights
    weights <- if (is.null(w)) NULL else b$data[[w]]
    compare_call(paste0("compare:", e$name), b$data, m, weights)
    anova_call(paste0("anova:", e$name), b$data, m, weights)
  }
}

# anova(m, type, test) per TERM_CALLS, each table followed by its text cells:
# the heading (with the notes), and the fit's term_test_auto.
term_calls <- function() {
  for (cl in TERM_CALLS) {
    t <- TERM_MODELS[[cl$model]]
    a <- aggregated(t$formula, read_data(t$data, t$factors))
    m <- list(formula = a$formula, family = t$family, link = t$link, reml = NULL, nagq = 1L)
    calls <- sprintf("terms:%s:%s%s", cl$model, cl$types,
                     if (cl$named) paste0(":", cl$test) else "")
    f <- tryCatch(fit(a$data, m, NULL, if (identical(cl$test, "LRT")) "LRT" else "auto"),
                  error = function(e) e)
    for (i in seq_along(calls)) {
      call <- calls[[i]]
      if (inherits(f, "error")) {
        refused(call, f)
        next
      }
      tab <- tryCatch(anova(f, type = cl$types[[i]], test = cl$test), error = function(e) e)
      if (inherits(tab, "error")) {
        refused(call, tab)
        next
      }
      port_table(call, tab)
      n <- nrow(tab)
      k <- ncol(tab)
      emit_text(call, "heading", "text", c(n, k), paste(attr(tab, "heading"), collapse = "\n"))
      emit_text(call, "term_test_auto", "test", c(n + 1L, k + 1L), f$term_test_auto$test)
      emit_text(call, "term_test_auto", "reason", c(n + 1L, k + 2L), f$term_test_auto$reason)
    }
  }
}

missing_value_call <- function() {
  data <- read_data("data/simulated/wls_basic.csv", character(0))
  data$x2[MISSING_ROW] <- NA_real_
  m <- lapply(c("y ~ x1", "y ~ x1 + x2"), function(f) {
    list(formula = f, family = "gaussian", link = "identity", reml = NULL, nagq = 1L)
  })
  compare_call("compare:missing_x2_wls_basic", data, m, data$w, nobs = TRUE)
}

rlrt_calls <- function() {
  data <- read_data("data/empirical/Dyestuff.csv", "Batch")
  m <- lapply(c("Yield ~ 1", "Yield ~ 1 + (1 | Batch)"), function(f) {
    list(formula = f, family = "gaussian", link = "identity", reml = TRUE, nagq = 1L)
  })
  compare_call("compare:rlrt_dyestuff", data, m, NULL)
  anova_call("anova:rlrt_dyestuff", data, m, NULL)
}

pairs()
term_calls()
missing_value_call()
rlrt_calls()
