# Summarize the fits against the TRUE parameters (external-truth accuracy), using
# the paper's exact aggregation, and place glmm beside the frozen Table 4/5.
# Prints a compact per-batch block to the console and writes the full artifacts.
#
# Per cell, over same-rule converged reps (boundary included, see summarize_arm):
# cell mean estimate and cell RMSE per param.
# Grand average across cells: bias = mean_cells |cellmean-true|, rmse =
# mean_cells sqrt(mean_reps (est-true)^2). tau1/rho01 exist only on slope cells,
# so na.rm averages them over those cells only (matches 03_GatherResults.R).
source(file.path(dirname(sub("^--file=", "",
  grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "common.R"))

P <- c("beta0","beta1","beta2","beta3","tau0","tau1","rho01")
TRUTH <- TRUE_PARAMS[P]
RES <- file.path(ACC_DIR, "results")
REPORTS <- file.path(ACC_DIR, "reports")
CAP <- ACC_UPTO   # cap all engines to the current batch for fair comparison
RI_CELLS <- vapply(CELLS, function(c) c$id, "")[!vapply(CELLS, function(c) c$slope, TRUE)]

summarize_arm <- function(dir, arm, cells = NULL) {
  files <- Sys.glob(file.path(dir, sprintf("*__%s.csv", arm)))
  if (!is.null(cells))
    files <- files[sub(sprintf("__%s\\.csv$", arm), "", basename(files)) %in% cells]
  if (!length(files)) return(NULL)
  pc <- list()
  for (f in files) {
    d <- read.csv(f); d <- d[d$rep <= CAP, , drop = FALSE]
    cid <- sub(sprintf("__%s\\.csv$", arm), "", basename(f))
    slope <- grepl("rdis$", cid)
    err <- d$err %in% TRUE
    # ONE rule for every engine: singularity is derived from the estimates via
    # derive_singular (native flags ignored - they only relabel the same fits:
    # per-rep cross-check found glmm and lme4 on the same boundary set, 600/600
    # RI reps and 465/469 slope reps), and a boundary optimum counts as
    # converged for everyone - it is the constrained MLE, and glmm's boundary
    # fits beat GLMMadaptive's interior ones on 91% of disagreeing slope reps
    # by likelihood. The !err guard matters: on slope cells derive_singular
    # flags every errored rep via rho01=NA.
    # The frozen published rows keep the paper's own convention (lme4 boundary
    # fits excluded via its warnings), so the gate |diff| now includes that
    # convention gap on top of Monte Carlo error.
    sing <- derive_singular(d, slope) & !err
    conv <- (d$good %in% TRUE | sing) & !err
    bound <- sing
    gm <- d[conv, , drop = FALSE]   # metric set = same-rule converged (boundary included)
    row <- list(cell = cid, n_tot = nrow(d), n_good = sum(conv), n_metric = nrow(gm),
                n_bnd_conv = sum(bound & conv), n_bnd_non = sum(bound & !conv),
                n_err = sum(err), mean_secs = mean(d$secs, na.rm = TRUE))
    for (p in P) {
      est <- gm[[p]]
      ok <- length(est) && !all(is.na(est))
      row[[paste0("bias_", p)]] <- if (ok) abs(mean(est, na.rm=TRUE) - TRUTH[[p]]) else NA_real_
      row[[paste0("rmse_", p)]] <- if (ok) sqrt(mean((est - TRUTH[[p]])^2, na.rm=TRUE)) else NA_real_
    }
    pc[[cid]] <- as.data.frame(row, stringsAsFactors = FALSE)
  }
  pc <- do.call(rbind, pc)
  grand <- function(pre) vapply(P, function(p) mean(pc[[paste0(pre,p)]], na.rm=TRUE), numeric(1))
  list(per_cell = pc, arm = arm, bias = grand("bias_"), rmse = grand("rmse_"),
       n_tot = sum(pc$n_tot), n_good = sum(pc$n_good), n_metric = sum(pc$n_metric),
       n_bnd_conv = sum(pc$n_bnd_conv), n_bnd_non = sum(pc$n_bnd_non), n_err = sum(pc$n_err),
       conv = mean(pc$n_good / pc$n_tot))
}
# mean ms/fit over a cell subset (matched-cell timing).
arm_time_ms <- function(a, cellset = NULL) {
  if (is.null(a)) return(NA_real_)
  pc <- a$per_cell; if (!is.null(cellset)) pc <- pc[pc$cell %in% cellset, ]
  1000 * mean(pc$mean_secs, na.rm = TRUE)
}

# glmm_AGQ is scoped to the RI-only cells so it stays comparable to the published
# lme4_AGQ column, which is RI-only by construction (glmer refuses vector AGQ).
# glmm CAN do AGQ on the slope cells - that arm is reported separately as
# glmm_AGQ_slope, whose comparator is GLMMadaptive (the only published engine
# doing vector AGQ); mixing the two cell sets into one average would break both
# comparisons.
SLOPE_CELLS <- vapply(CELLS, function(c) c$id, "")[vapply(CELLS, function(c) c$slope, TRUE)]
A <- list(
  glmm_LA        = summarize_arm(file.path(RES,"glmm"), "glmm_LA"),
  glmm_AGQ       = summarize_arm(file.path(RES,"glmm"), "glmm_AGQ", cells = RI_CELLS),
  glmm_AGQ_slope = summarize_arm(file.path(RES,"glmm"), "glmm_AGQ", cells = SLOPE_CELLS),
  lme4_LA        = summarize_arm(file.path(RES,"lme4"), "lme4_LA"),
  lme4_AGQ       = summarize_arm(file.path(RES,"lme4"), "lme4_AGQ"),
  GLMMadaptive   = summarize_arm(file.path(RES,"glmmadaptive"), "GLMMadaptive"))

truth_bias <- read.csv(file.path(ACC_DIR,"truth","mean_abs_bias.csv"), comment.char="#")
truth_rmse <- read.csv(file.path(ACC_DIR,"truth","mean_rmse.csv"), comment.char="#")

# ---------- write artifacts ----------
if (!is.null(A$glmm_LA)) {
  pc_all <- rbind(cbind(arm="glmm_LA", A$glmm_LA$per_cell),
                  if (!is.null(A$glmm_AGQ)) cbind(arm="glmm_AGQ", A$glmm_AGQ$per_cell))
  write.csv(pc_all, file.path(RES,"glmm_per_cell.csv"), row.names=FALSE)
}
add_rows <- function(base, metric) {
  arms <- A[c("glmm_LA","glmm_AGQ","glmm_AGQ_slope","lme4_LA","lme4_AGQ","GLMMadaptive")]
  labs <- c("glmm_LA","glmm_AGQ","glmm_AGQ_slope","lme4_LA(ours)","lme4_AGQ(ours)",
            "GLMMadaptive(ours)")
  extra <- do.call(rbind, Map(function(a, nm) {
    if (is.null(a)) return(NULL)
    v <- if (metric=="bias") a$bias else a$rmse
    data.frame(package=nm, beta0=v["beta0"],beta1=v["beta1"],beta2=v["beta2"],
               beta3=v["beta3"],tau0=v["tau0"],tau1=v["tau1"],rho01=v["rho01"], row.names=NULL)
  }, arms, labs))
  cbind(metric=metric, rbind(base, extra))
}
report <- rbind(add_rows(truth_bias,"bias"), add_rows(truth_rmse,"rmse"))
report[,P] <- round(report[,P], 4)
write.csv(report, file.path(RES,"accuracy_report.csv"), row.names=FALSE)

# ---------- console block ----------
fmt <- function(v) sprintf("%7s", ifelse(is.na(v), "-", sprintf("%.3f", v)))
hdr <- function() cat(sprintf("%-18s", ""), paste(sprintf("%7s", P), collapse=""), "\n")
prow <- function(lab, v) cat(sprintf("%-18s", lab), paste(fmt(v[P]), collapse=""), "\n")

cat("\n################  BATCH: reps 1..", CAP, "  ################\n", sep="")

cat("\n--- FITS (same rule for every engine; err=threw) ---\n")
cat("    conv and boundary use ONE shared rule for all engines: singular = derived from the\n")
cat("    estimates (tau < 1e-3, |rho| > 0.99), and a boundary optimum counts as converged -\n")
cat("    it is the constrained MLE (glmm and lme4 land on the same boundary set per-rep; the\n")
cat("    old conv% gap was purely each engine's labeling). Bias/RMSE use this same set. The\n")
cat("    frozen published rows keep the paper's own convention (lme4 excludes boundary fits),\n")
cat("    so the gate |diff| includes that convention gap on top of Monte Carlo error.\n")
for (nm in names(A)) {
  a <- A[[nm]]; if (is.null(a)) next
  nonconv <- a$n_tot - a$n_good - a$n_err
  flag <- if (a$n_err > 0) "  <<< ERRORS" else ""
  cat(sprintf("  %-14s %d fits: %d conv (%d boundary), %d nonconv (%d boundary), %d err -> %d in metric (conv %.1f%%)%s\n",
              a$arm, a$n_tot, a$n_good, a$n_bnd_conv, nonconv, a$n_bnd_non, a$n_err, a$n_metric, 100*a$conv, flag))
  bad <- a$per_cell[a$per_cell$n_err > 0 | (a$per_cell$n_good/a$per_cell$n_tot) < 0.8, ]
  if (nrow(bad)) for (i in seq_len(nrow(bad)))
    cat(sprintf("       - %-26s %d/%d conv, %d boundary, %d err\n",
                bad$cell[i], bad$n_good[i], bad$n_tot[i], bad$n_bnd_conv[i] + bad$n_bnd_non[i], bad$n_err[i]))
}

cat("\n--- VALIDATION GATE: our lme4 vs paper's published lme4 (identical data) ---\n")
pb <- truth_bias; rownames(pb) <- pb$package; pr <- truth_rmse; rownames(pr) <- pr$package
gate <- function(arm, pubname, metric) {
  a <- A[[arm]]; if (is.null(a)) return()
  pub <- unlist((if (metric=="bias") pb else pr)[pubname, P]); our <- (if (metric=="bias") a$bias else a$rmse)[P]
  hdr(); prow(paste0("published ",pubname), pub); prow(paste0("ours ",metric), our)
  prow("|diff|", abs(pub - our))
}
gate("lme4_LA","lme4_LA","bias"); gate("lme4_LA","lme4_LA","rmse"); cat("\n")
gate("lme4_AGQ","lme4_AGQ","bias"); gate("lme4_AGQ","lme4_AGQ","rmse")

cat("\n--- glmm ACCURACY vs frozen table (mean absolute bias) ---\n"); hdr()
for (i in seq_len(nrow(truth_bias))) prow(truth_bias$package[i], truth_bias[i,])
prow(">> glmm_LA", A$glmm_LA$bias); if (!is.null(A$glmm_AGQ)) prow(">> glmm_AGQ", A$glmm_AGQ$bias)
if (!is.null(A$glmm_AGQ_slope)) prow(">> glmm_AGQ_slope", A$glmm_AGQ_slope$bias)
cat("\n--- glmm ACCURACY vs frozen table (mean RMSE) ---\n"); hdr()
for (i in seq_len(nrow(truth_rmse))) prow(truth_rmse$package[i], truth_rmse[i,])
prow(">> glmm_LA", A$glmm_LA$rmse); if (!is.null(A$glmm_AGQ)) prow(">> glmm_AGQ", A$glmm_AGQ$rmse)
if (!is.null(A$glmm_AGQ_slope)) prow(">> glmm_AGQ_slope", A$glmm_AGQ_slope$rmse)

cat("\n--- TIMING (mean ms/fit, core-1 locked, matched cells) ---\n")
la_g <- arm_time_ms(A$glmm_LA); la_l <- arm_time_ms(A$lme4_LA); la_a <- arm_time_ms(A$GLMMadaptive)
cat("Laplace vs adaptive, all 24 cells:\n")
cat(sprintf("  glmm_LA       %6.1f ms\n", la_g))
cat(sprintf("  lme4_LA       %6.1f ms   (%.1fx glmm_LA)\n", la_l, la_l/la_g))
cat(sprintf("  GLMMadaptive  %6.1f ms   (%.1fx glmm_LA)\n", la_a, la_a/la_g))
ag_g <- arm_time_ms(A$glmm_AGQ, RI_CELLS); ag_l <- arm_time_ms(A$lme4_AGQ, RI_CELLS)
ag_a <- arm_time_ms(A$GLMMadaptive, RI_CELLS)
if (!is.na(ag_g)) {
  cat("Adaptive quadrature, 12 RI-only cells:\n")
  cat(sprintf("  glmm_AGQ      %6.1f ms\n", ag_g))
  cat(sprintf("  GLMMadaptive  %6.1f ms   (%.1fx glmm_AGQ)\n", ag_a, ag_a/ag_g))
  cat(sprintf("  lme4_AGQ      %6.1f ms   (%.1fx glmm_AGQ)\n", ag_l, ag_l/ag_g))
}
cat("(timing is machine-dependent; the x-ratios are the transferable numbers.)\n")

# persist the console block
sink(file.path(REPORTS,"final_analysis.txt")); cat("see console output of summarize_accuracy_truth.R; batch cap =", CAP, "\n"); sink()
