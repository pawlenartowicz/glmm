# Shared pieces for the accuracy-benchmark fit drivers.
# True parameters and the 24-cell catalog come from manifest.json (single source
# of truth). Response is y_full (the beta=(0.1,0.3,-0.2,0.1) main design).
# Fits run in escalating batches: ACC_UPTO caps the cumulative rep count; each
# driver appends only the reps not yet done (resumable, no refits).

ACC_DIR <- dirname(sub("^--file=", "",
                       grep("^--file=", commandArgs(FALSE), value = TRUE)[1]))
if (is.na(ACC_DIR) || !nzchar(ACC_DIR)) ACC_DIR <- getwd()
EXT_DATA <- file.path(ACC_DIR, "external", "data")

.manifest <- jsonlite::fromJSON(file.path(ACC_DIR, "manifest.json"),
                                simplifyDataFrame = FALSE)
CELLS <- .manifest$cells
TRUE_PARAMS <- unlist(.manifest$true_params)
NAGQ_AGQ <- as.integer(.manifest$nagq_agq)
ACC_UPTO <- { v <- suppressWarnings(as.integer(Sys.getenv("ACC_UPTO", "1000"))); if (is.na(v)) 1000L else v }
ACC_ONLY <- strsplit(Sys.getenv("ACC_ONLY", ""), ",")[[1]]

cell_formula <- function(slope) {
  rd <- if (slope) "(1 + time | id)" else "(1 | id)"
  stats::as.formula(paste("y_full ~ time + group + time:group +", rd))
}
cell_family <- function(family) if (family == "Bernoulli") stats::binomial() else stats::poisson()

# Per-rep record with a uniform schema across engines. err = fit threw / gave no
# estimate; good = converged and not blown up (err implies !good). NA params on
# RI-only cells for tau1/rho01.
mk_row <- function(i, est, good, err, secs, singular = NA) {
  data.frame(rep = i, beta0 = est[["beta0"]], beta1 = est[["beta1"]],
             beta2 = est[["beta2"]], beta3 = est[["beta3"]], tau0 = est[["tau0"]],
             tau1 = est[["tau1"]], rho01 = est[["rho01"]],
             good = good, singular = singular, err = err, secs = secs)
}

# A fit is singular when it sits on the boundary of the RE parameter space: a
# variance driven to ~0 or a correlation pinned to +/-1 (or undefined because a
# variance collapsed). Mirrors lme4::isSingular. Used to make the metric filter
# apples-to-apples with the paper, which drops boundary fits via lme4's warnings.
# Preferred source is the engine's own flag (the `singular` column); this derives
# the identical result from the estimates when that column is absent.
VAR_TOL <- 1e-3; COR_TOL <- 0.99
derive_singular <- function(d, slope) {
  s <- is.finite(d$tau0) & d$tau0 < VAR_TOL
  if (slope) s <- s | (is.finite(d$tau1) & d$tau1 < VAR_TOL) |
    !is.finite(d$rho01) | (is.finite(d$rho01) & abs(d$rho01) > COR_TOL)
  s
}
NA_EST <- c(beta0=NA_real_, beta1=NA_real_, beta2=NA_real_, beta3=NA_real_,
            tau0=NA_real_, tau1=NA_real_, rho01=NA_real_)

# Fit reps (done+1)..upto for one cell/arm and append to out_csv. fit_rep(df, i)
# returns a one-row data.frame (via mk_row). Returns counts for the batch report.
append_fits <- function(out_csv, data_ls, upto, fit_rep) {
  had <- file.exists(out_csv)
  done <- if (had) nrow(read.csv(out_csv)) else 0L
  upto <- min(upto, length(data_ls))
  if (done >= upto) return(c(done = done, new = 0, good_new = 0, err_new = 0))
  rows <- lapply((done + 1L):upto, function(i) fit_rep(data_ls[[i]], i))
  df <- do.call(rbind, rows)
  write.table(df, out_csv, sep = ",", row.names = FALSE,
              col.names = !had, append = had, qmethod = "double")
  c(done = upto, new = nrow(df), good_new = sum(df$good), err_new = sum(df$err))
}

# Read a cell's 1000-sim list once per driver invocation.
load_cell <- function(cell) {
  e <- new.env(); load(file.path(EXT_DATA, paste0(cell$id, ".RData")), envir = e)
  e$simulate_data_ls
}

report_arm <- function(cell, arm, r) {
  if (r[["new"]] == 0) { cat(sprintf("  %-9s skip (have %d)\n", arm, r[["done"]])); return() }
  msg <- sprintf("  %-9s +%d -> %d reps, %d good", arm, r[["new"]], r[["done"]], r[["good_new"]])
  if (r[["err_new"]] > 0) msg <- paste0(msg, sprintf("  !! %d ERRORED", r[["err_new"]]))
  cat(msg, "\n")
}
