#!/usr/bin/env Rscript
# The weights-of-trust cells, regenerated -> validation/goldens/inference/cr2/f1_reps.json
# (the `f1_reps` entry of the manifest's `inference_references` block) and
# validation/data/inference/dcr_cells/<cell>.csv (its `designs`).
#
# THE ORACLE IS SACRED. f1_reps.json carries the F1 (δ fit + CR2) estimate, SE,
# df and δ̂ that formula_search_mc.R stored in formula_search_reps_R1000.csv, next
# to the rows they were fitted on. On any glmm disagreement, glmm is presumed
# wrong: never relax a tolerance or edit a reference.
#
# A BARE RUN OVERWRITES f1_reps.json AND EVERY dcr_cells CSV. Needs the stored
# replications (1.6 GB, outside the repo; streamed through awk, never read whole):
#   WOT_REPS_CSV=<path>/formula_search_reps_R1000.csv Rscript validation/tools/goldens_cr2_cells.R
#
# formula_search_mc.R and f2_extensions_mc.R cannot be sourced (each reads its path
# from `--file=` and runs its checks and simulation at top level), so their cell and
# stream blocks and run_cell()'s y draw are copied below; make_design() and the noise
# curves are parsed out of tools/cr2/wls_delta_mc.R; the rest comes from
# tools/cr2/efficiency_lib.R. The manifest entry's `design_sources` records the sha256
# of each source as copied.
# Per replication r of a cell, y_0 is run_cell()'s draw from the cell's main stream
# (run_methods() draws nothing; the bootstrap has its own stream) and
# y_1 = y_0 + beta1 * x. The rows are checked against the CSV before anything is
# written: beta1 to 1e-12 relative, and an lme4 REML fit with the claimed weights
# against the stored `today` estimate (method 1, the same model) to 1e-3 of its SE.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()
library(parallel)  # nextRNGStream(), as formula_search_mc.R attaches it
source(file.path(suite_dir, "tools", "cr2", "efficiency_lib.R"))

GENERATOR <- "goldens_cr2_cells.R"
entries <- Filter(function(e) identical(e$generator, GENERATOR), select_entries("cr2"))
reps_csv <- Sys.getenv("WOT_REPS_CSV")
if (length(entries) > 0L && !file.exists(reps_csv)) {
  stop("set WOT_REPS_CSV to formula_search_reps_R1000.csv", call. = FALSE)
}

# ---- copied from formula_search_mc.R: cells and streams ----------------------------
fc <- followup_cells()
procs_d <- list(increasing     = function(w) w^(1 / 4),
                step           = function(w) ifelse(w < 0.1, sqrt(10), 1),
                floor_nearzero = function(w) sqrt(1 + 10^(-1.5) / w))
cells_d <- list()
for (des in c("10x10", "50x10")) for (dr in 1:5) {
  b <- fc$cells[[which(fc$cell_id == sprintf("B/%s/d%d/wide/honest", des, dr))]]
  for (p in names(procs_d)) {
    cl <- b; cl$part <- "D"; cl$process <- p; cl$noise_sd <- procs_d[[p]](b$d$w)
    if (p == "floor_nearzero") {
      nz <- seq(20, nrow(b$d), by = 20)
      cl$d$w[nz] <- 1e-8; cl$noise_sd[nz] <- 3
    }
    cells_d[[length(cells_d) + 1]] <- cl
  }
}
s <- fc$s_check; seeds_d <- vector("list", length(cells_d))
for (k in seq_along(cells_d)) { s <- nextRNGStream(s); seeds_d[[k]] <- s }
procs_e <- procs_b[c("precision", "honest", "partial", "floor_mid")]
cells_e <- list()
for (G_e in c(10, 50)) {
  s <- nextRNGStream(s); assign(".Random.seed", s, envir = .GlobalEnv)
  g_e <- factor(rep(seq_len(G_e), each = 10))
  x_e <- rnorm(10 * G_e); w_e <- exp(runif(10 * G_e, log(0.05), 0)); xg_e <- rnorm(G_e)[as.integer(g_e)]
  for (pat in c("missing_slope", "group_level")) for (p in names(procs_e))
    cells_e[[length(cells_e) + 1]] <- list(
      part = "E", design = sprintf("%dx10", G_e), draw = 1, pattern = pat, process = p,
      d = data.frame(g = g_e, x = if (pat == "group_level") xg_e else x_e, w = w_e),
      noise_sd = procs_e[[p]](w_e), att = FALSE, slope_sd = if (pat == "missing_slope") 0.5 else 0)
}
seeds_e <- vector("list", length(cells_e))
for (k in seq_along(cells_e)) { s <- nextRNGStream(s); seeds_e[[k]] <- s }
in_ab <- vapply(fc$cells, `[[`, "", "part") %in% c("A", "B")
cells <- c(fc$cells[in_ab], cells_d, cells_e)
seeds <- c(fc$seeds[in_ab], seeds_d, seeds_e)
cell_id <- vapply(cells, function(k) paste(k$part, k$design, paste0("d", k$draw), k$pattern,
                                           k$process, sep = "/"), "")
f2_cells <- c(outer(c("A/%s/d1/continuous/precision", "A/%s/d1/junk/junk", "B/%s/d1/wide/honest",
                      "B/%s/d1/wide/bump", "D/%s/d1/wide/step"), c("10x10", "50x10"), sprintf))
# ---- end of the copy ----------------------------------------------------------------

# ---- copied from f2_extensions_mc.R, which continues the same stream `s`: the slope
# grid (S); its group-level x cells (G) come after it and are not needed --------------
slope_vars <- c(0, 0.01, 0.025, 0.05, 0.1, 0.25, 1)
procs_s <- procs_b[c("precision", "honest", "floor_mid")]
cells_s <- list()
for (G_s in c(10, 50)) for (dr in 1:3) {
  s <- nextRNGStream(s); assign(".Random.seed", s, envir = .GlobalEnv)
  g_s <- factor(rep(seq_len(G_s), each = 10))
  x_s <- rnorm(10 * G_s); w_s <- exp(runif(10 * G_s, log(0.05), 0))
  sl <- c(lapply(slope_vars, sqrt), list(rep(c(0, sqrt(0.5)), length.out = G_s)))
  names(sl) <- c(sprintf("slope%g", slope_vars), "slope_het")
  for (pat in names(sl)) for (p in names(procs_s))
    cells_s[[length(cells_s) + 1]] <- list(
      part = "S", design = sprintf("%dx10", G_s), draw = dr, pattern = pat, process = p,
      d = data.frame(g = g_s, x = x_s, w = w_s), noise_sd = procs_s[[p]](w_s), att = FALSE,
      slope_sd = sl[[pat]])
}
# ---- end of the copy ----------------------------------------------------------------
cells <- c(cells, cells_s)
cell_id <- c(cell_id, vapply(cells_s, function(k) paste(k$part, k$design, paste0("d", k$draw),
                                                        k$pattern, k$process, sep = "/"), ""))

# The WLS cells of wls_delta_mc.R (tools/cr2/): make_design() and the noise variance
# curves, read out of it (the script runs its simulation at top level). Ids are
# WLS/n<n>/<pattern>/<curve>; make_design() seeds Mersenne-Twister itself.
wls <- local({
  ex <- parse(file.path(suite_dir, "tools", "cr2", "wls_delta_mc.R"), encoding = "UTF-8")
  env <- new.env()
  for (name in c("make_design", "curves")) {
    hit <- vapply(ex, function(e) is.call(e) && identical(e[[1]], as.name("<-")) &&
                    identical(e[[2]], as.name(name)), NA)
    stopifnot(sum(hit) == 1L)
    eval(ex[[which(hit)]], envir = env)
  }
  env
})
wls_cell <- function(id) {
  f <- strsplit(id, "/", fixed = TRUE)[[1]]
  n <- as.integer(sub("^n", "", f[2]))
  kind <- RNGkind("Mersenne-Twister")
  d <- wls$make_design(f[3], G = n / 10)
  RNGkind(kind[1])
  list(d = d, noise_sd = sqrt(wls$curves[[f[4]]](d$w)))
}

# The oracle SE and beta1 of run_cell(), then its main-stream draws of replications
# 1..R (sl = 0 on every f2 cell, so no slope draw).
draw_reps <- function(k, R) {
  cell <- cells[[k]]; d <- cell$d; n <- nrow(d); G <- nlevels(d$g)
  stopifnot((cell$slope_sd %||% 0) == 0)
  Xo <- oracle_x(cell); j <- ncol(Xo)
  Mo <- solve(crossprod(Xo, solve(true_v(cell), Xo)))
  beta1 <- 2 * sqrt(Mo[j, j])
  assign(".Random.seed", seeds[[k]], envir = .GlobalEnv)
  y0 <- lapply(seq_len(R), function(r) rnorm(G)[as.integer(d$g)] + cell$noise_sd * rnorm(n))
  list(beta1 = beta1, x1 = Xo[, j], y0 = y0)
}

# The stored rows of `methods` for `ids`, replications 1..R. The first nine columns
# hold no free text, so awk's comma split is safe on them.
stored_rows <- function(ids, methods, R) {
  header <- names(read.csv(reps_csv, nrows = 1L, check.names = FALSE))
  cmd <- sprintf("awk -F, '(%s) && $9 <= %d' %s",
                 paste(sprintf("$10 == \"\\\"%s\\\"\"", methods), collapse = " || "), R,
                 shQuote(reps_csv))
  rows <- read.csv(pipe(cmd), header = FALSE, col.names = header, stringsAsFactors = FALSE)
  rows[rows$cell %in% ids, ]
}

f1_reference <- function(entry) {
  R <- as.integer(entry$reps)
  ids <- unlist(entry$cells)
  stopifnot(setequal(ids, f2_cells))
  st <- stored_rows(ids, c("delta_cr2", "today"), R)
  out <- lapply(ids, function(id) {
    k <- which(cell_id == id); d <- cells[[k]]$d
    dr <- draw_reps(k, R)
    s_f1 <- st[st$cell == id & st$method == "delta_cr2", ]
    s_today <- st[st$cell == id & st$method == "today", ]
    stopifnot(nrow(s_f1) == R, nrow(s_today) == R, all(sort(s_f1$rep) == seq_len(R)))
    s_f1 <- s_f1[order(s_f1$rep), ]; s_today <- s_today[order(s_today$rep), ]
    if (any(abs(s_f1$beta1 / dr$beta1 - 1) > 1e-12)) stop(id, ": beta1 is not the stored one")
    reps <- lapply(seq_len(R), function(r) {
      y <- list(dr$y0[[r]], dr$y0[[r]] + dr$beta1 * dr$x1)
      for (a in 1:2) {
        sfx <- c("_0", "_1")[a]
        m <- lme4::lmer(yy ~ x + (1 | g), data = transform(d, yy = y[[a]]), weights = w, REML = TRUE)
        gap <- abs(lme4::fixef(m)[["x"]] - s_today[[paste0("est", sfx)]][r])
        if (gap > 1e-3 * s_today[[paste0("se", sfx)]][r]) {
          stop(sprintf("%s rep %d%s: the regenerated y does not give the stored estimate", id, r, sfx))
        }
      }
      f1 <- s_f1[r, ]
      list(rep = r, y_0 = y[[1]], y_1 = y[[2]],
           est_0 = f1$est_0, se_0 = f1$se_0, df_0 = f1$df_0, delta_hat_0 = f1$delta_hat_0,
           est_1 = f1$est_1, se_1 = f1$se_1, df_1 = f1$df_1, delta_hat_1 = f1$delta_hat_1)
    })
    list(cell = id, g = as.integer(d$g), x = d$x, w = d$w, beta1 = dr$beta1, reps = reps)
  })
  check_finite(out, entry$name)
  list(provenance = provenance(GENERATOR), source = basename(reps_csv),
       source_md5 = unname(tools::md5sum(reps_csv)), method = "delta_cr2", cells = out)
}

# One CSV per design: g, x, w, the row noise SD and the row's group slope SD (0 without
# a slope); a WLS design has no g and no slope column.
write_designs <- function(entry) {
  dir <- file.path(suite_dir, "data", "inference", "dcr_cells")
  dir.create(dir, showWarnings = FALSE, recursive = TRUE)
  for (id in unlist(entry$designs)) {
    if (startsWith(id, "WLS/")) {
      cell <- wls_cell(id)
      d <- data.frame(x = cell$d$x, w = cell$d$w, noise_sd = cell$noise_sd)
    } else {
      cell <- cells[[which(cell_id == id)]]
      g <- as.integer(cell$d$g)
      d <- data.frame(g = g, x = cell$d$x, w = cell$d$w, noise_sd = cell$noise_sd,
                      slope_sd = rep_len(cell$slope_sd %||% 0, max(g))[g])
    }
    write.csv(d, file.path(dir, paste0(gsub("/", "_", id), ".csv")), row.names = FALSE)
  }
}

for (entry in entries) {
  ref <- f1_reference(entry)
  write_reference(ref, file.path(suite_dir, "goldens", "inference", "cr2", paste0(entry$name, ".json")))
  write_designs(entry)
  cat(sprintf("wrote %s (%d cells x %d reps) and %d designs\n", entry$name, length(ref$cells),
              as.integer(entry$reps), length(entry$designs)))
}
