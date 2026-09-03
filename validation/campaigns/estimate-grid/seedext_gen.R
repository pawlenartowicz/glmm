#!/usr/bin/env Rscript
# Seed-extension study generator (lme4 tolPwrss / finite-difference-Hessian
# question, lme4 issue #998): takes the 106 glmer-vs-lme4 cells of the
# estimate-grid and adds 9 fresh data seeds each (suffix _e1.._e9, seed =
# base + 100000*k, the b1-replicate offset scheme). Writes
# manifest_seedext.json + seedext_todo.txt here and the CSVs into
# ../speed-grid/data/ (where both fit drivers look). The frozen campaign
# manifest/results are untouched.
#
# For the two b1-flagged glmer cells (binb_q2s/pois_q2s g3000p20_bal_base)
# _e1.._e4 land on the same seeds as their existing _s2.._s5 replicates --
# same data under a different name, harmless.
#
# Simulation code is NOT duplicated: prep.R's top-level assignments
# (STRUCTURES, sd_ladder, extra_levels, sim_cell, ...) are parsed and
# evaluated below; its side-effecting top-level calls and loops are skipped.
suppressMessages({ library(jsonlite); library(MASS) })

here_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
speed_dir <- normalizePath(file.path(here_dir, "..", "speed-grid"))
data_dir <- file.path(speed_dir, "data")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

# pull sim_cell + friends out of prep.R: evaluate only `<-` assignments
for (e in as.list(parse(file.path(speed_dir, "prep.R")))) {
  if (is.call(e) && identical(e[[1]], as.name("<-"))) eval(e)
}
stopifnot(is.function(sim_cell), is.list(STRUCTURES))

# cell selection: the glmer-vs-lme4 rows of the frozen status map
status <- read.csv(file.path(here_dir, "reports", "status_map.csv"),
                   stringsAsFactors = FALSE)
# q2s+nAGQ cells route to GLMMadaptive in fit.R (vector AGQ; glmer refuses),
# so they carry no glmer tolPwrss signal -- excluded. Drops one cell
# (binb_q2s_g3000p20_bal_rare, the oracle-timeout row).
sel_ids <- status$case_id[status$family != "gaussian" &
                          status$oracle_engine == "lme4" &
                          !(status$structure == "q2s" & !is.na(status$nagq) &
                            status$nagq > 1)]
manifest <- fromJSON(file.path(here_dir, "manifest.json"),
                     simplifyDataFrame = FALSE)
cells <- Filter(function(c) c$case_id %in% sel_ids, manifest$cells)
stopifnot(length(cells) == length(sel_ids))

# sim_cell branches on the pre-remap family tag; recover it from the case_id
# prefix (manifest carries the remapped "binomial"/"poisson")
sim_family <- c(binb = "binomial_bin", bina = "binomial_agg", pois = "poisson")

ext_cells <- list()
for (cell in cells) {
  tag <- sub("_.*", "", cell$case_id)
  for (k in 1:9) {
    c2 <- cell
    c2$case_id <- sprintf("%s_e%d", cell$case_id, k)
    c2$seed <- cell$seed + 100000L * k
    # I(): keep length-1 re_q/factors as JSON arrays through the round-trip
    # (fit.rs expects arrays; auto_unbox would collapse ["g1"] to "g1")
    c2$re_q <- I(unlist(cell$re_q))
    c2$factors <- I(unlist(cell$factors))
    sim_spec <- c2
    sim_spec$family <- sim_family[[tag]]
    meta <- sim_cell(sim_spec)
    write.csv(meta$df, file.path(data_dir, paste0(c2$case_id, ".csv")),
              row.names = FALSE)
    ext_cells[[length(ext_cells) + 1L]] <- c2
  }
}

ext <- list(schema = "glmm-grid-manifest/1",
            generated_by = "campaigns/estimate-grid/seedext_gen.R",
            cells = ext_cells)
write(toJSON(ext, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null"),
      file.path(here_dir, "manifest_seedext.json"))
writeLines(vapply(ext_cells, `[[`, "", "case_id"),
           file.path(here_dir, "seedext_todo.txt"))
cat(sprintf("seedext cells: %d (from %d base cells)  csvs in %s\n",
            length(ext_cells), length(cells), data_dir))
