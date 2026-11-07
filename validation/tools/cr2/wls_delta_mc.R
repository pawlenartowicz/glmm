# WLS Monte Carlo: does the plain δ fit hold size without random effects, and how do
# robust SEs on the given and on the fitted weights compare?
#   delta   lm with weights w^δ̂, δ̂ maximizes the REML log-likelihood of the
#           weighted lm on [−1, 2] (optimize, tol 1e-4; logLik.lm keeps ½Σlog ω,
#           so profiles over δ are comparable); model-based SE, t with n − p df
#           (δ̂ treated as known). This is nlme::gls(weights = varPower(form = ~ w))
#           up to the parameterization δ = −2·power.
#   today   lm with the user's weights w (δ = 1); model-based SE, t with n − p df.
#   unw     lm without weights (δ = 0); model-based SE, t with n − p df.
#   today_hc2  fit of today; HC2 SE, Bell–McCaffrey df.
#   today_hc3  fit of today; HC3 SE, t with n − p df.
#   delta_hc2  fit of delta; HC2 SE, Bell–McCaffrey df (δ̂ treated as known).
#   oracle  GLS with the true variances; exact SE; normal quantile.
# HC2 with Bell–McCaffrey df is CR2 with one-row clusters and Pustejovsky–Tipton df
# (clubSandwich, inverse_var = TRUE); checked against clubSandwich at the start.
# Model: y = βx + ε, Var(εᵢ) = v(wᵢ), no random effect. Cells: the 16 cells of the
# design-only screen's LM design, built at n rows (n/10 blocks of 10 rows, the
# grouping dropped; junk adds 5 rows with w = 1e-8 per block, so 1.5n rows), design
# seed 20261039 (Mersenne-Twister) at every n. At n = 100 these are the screen's
# designs, so the results line up with its exact sizes at the pseudo-true δ.
# Effect sizes: β = 0 and β = 2 × the oracle's SE, on the same ε draw.
# Seeds: data from seed 20261001 + n, L'Ecuyer-CMRG, one stream per cell.
# Run: Rscript wls_delta_mc.R [R [n ...]]  (defaults 1000; n = 20 50 100 500).
# 4 cores (mclapply over cells).
# Outputs: results/wls_delta_{reps,summary}_R<R>.csv and results/wls_delta_R<R>.txt.

library(parallel)
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
out <- file.path(here, "results")
args <- commandArgs(TRUE)
R <- if (length(args)) as.integer(args[1]) else 1000L
ns <- if (length(args) > 1) as.integer(args[-1]) else c(20L, 50L, 100L, 500L)
stopifnot(all(ns %% 10 == 0))
t_start <- proc.time()[3]

# ---- designs and noise curves (the screen's; noise variance per row) --------
make_design <- function(pattern, seed = 20261039, G = 10, m = 10) {
  set.seed(seed)
  n <- G * m
  g <- rep(seq_len(G), each = m)
  x <- rnorm(n)
  w <- switch(pattern,
    continuous = exp(runif(n, log(0.05), 0)),
    near01     = sample(rep(c(0.05, 1), length.out = n)),
    wide       = exp(runif(n, log(0.001), 0)),
    junk       = NULL)
  if (pattern == "junk") {  # 10 rows w = 1 plus 5 rows w = 1e-8 per group
    g <- sort(c(rep(seq_len(G), each = m), rep(seq_len(G), times = m / 2)))
    x <- rnorm(length(g))
    w <- ave(seq_along(g), g, FUN = function(i) c(rep(1, m), rep(1e-8, m / 2)))
  }
  data.frame(x = x, w = w)
}
curves <- list(
  precision  = function(w) 1 / w,
  honest     = function(w) rep(1, length(w)),
  partial    = function(w) w^(-1 / 2),
  floor      = function(w) 0.5 + 0.5 / w,
  floor_mid  = function(w) 1 + 10^(-1.5) / w,
  linear     = function(w) 2 - w,
  bump       = function(w) 1 + 9 * exp(-(log10(w) + 1.5)^2 / 0.5),
  step       = function(w) ifelse(w < 0.1, 10, 1),
  increasing = function(w) w^(0.5),
  junk       = function(w) ifelse(w == 1, 1, 9))
cells <- data.frame(
  pattern = c(rep("continuous", 5), rep("near01", 3), rep("wide", 7), "junk"),
  curve = c("precision", "honest", "partial", "floor", "linear",
            "precision", "honest", "floor",
            "precision", "honest", "partial", "floor_mid", "bump", "step", "increasing",
            "junk"))
cells$id <- paste(cells$pattern, cells$curve, sep = "/")
cells <- do.call(rbind, lapply(ns, function(n) cbind(n = n, cells)))
RNGkind("Mersenne-Twister")
designs <- Map(function(p, n) make_design(p, G = n / 10), cells$pattern, cells$n)

# ---- fits ------------------------------------------------------------------
reml_ll <- function(X, y, om) stats:::logLik.lm(lm.wfit(X, y, om), REML = TRUE)[1]
wls_fit <- function(X, y, om) {
  f <- lm.wfit(X, y, om)
  n <- nrow(X); p <- f$rank
  s2 <- sum(f$weights * f$residuals^2) / (n - p)
  c(beta = f$coefficients[[2]], se = sqrt(s2 * chol2inv(qr.R(f$qr))[2, 2]), df = n - p)
}
delta_fit <- function(X, y, w) {
  dh <- optimize(function(dl) reml_ll(X, y, w^dl), c(-1, 2), maximum = TRUE, tol = 1e-4)$maximum
  c(delta = dh, wls_fit(X, y, w^dh))
}
# HC2 / HC3 SE of coefficient 2 and the Bell–McCaffrey df, after whitening by √ω
# (X̃ = √ω X, ẽ = √ω e, A = (X̃ᵀX̃)⁻¹, hᵢ the leverages, aᵢ = x̃ᵢᵀA e₂):
#   Var_HC2 = Σ aᵢ² ẽᵢ² / (1 − hᵢ),  Var_HC3 = Σ aᵢ² ẽᵢ² / (1 − hᵢ)²;
#   df = (tr Q)² / tr Q², Q = D(I − H̃)D, D = diag(aᵢ / √(1 − hᵢ)) (Bell & McCaffrey
#   2002; Imbens & Kolesár 2016). With U = DX̃: tr Q = Σ aᵢ², and
#   tr Q² = Σ dᵢ⁴ − 2 tr(A UᵀD²U) + tr((AUᵀU)²), so no n × n matrix is formed.
hc_fit <- function(X, y, om) {
  f <- lm.wfit(X, y, om)
  Xt <- X * sqrt(om); et <- f$residuals * sqrt(om)
  A <- chol2inv(qr.R(f$qr))
  h <- rowSums((Xt %*% A) * Xt)
  a <- drop(Xt %*% A[, 2])
  d <- a / sqrt(1 - h)
  U <- Xt * d
  AUU <- A %*% crossprod(U)
  trQ2 <- sum(d^4) - 2 * sum(diag(A %*% crossprod(U, U * d^2))) + sum(AUU * t(AUU))
  c(beta = f$coefficients[[2]], se_hc2 = sqrt(sum(a^2 * et^2 / (1 - h))), df_bm = sum(a^2)^2 / trQ2,
    se_hc3 = sqrt(sum(a^2 * et^2 / (1 - h)^2)), df = nrow(X) - f$rank)
}
oracle <- function(X, y, v) {
  f <- lm.wfit(X, y, 1 / v)
  c(beta = f$coefficients[[2]], se = sqrt(solve(crossprod(X, X / v))[2, 2]))
}

# ---- checks ----------------------------------------------------------------
chk <- local({
  d <- designs[[1]]; set.seed(1); y <- rnorm(nrow(d)); X <- cbind(1, d$x)
  l <- lm(y ~ x, data = cbind(d, y = y), weights = w^0.7)
  c(ll_diff = abs(reml_ll(X, y, d$w^0.7) - logLik(l, REML = TRUE)[1]),
    se_diff = abs(delta_fit(X, y, d$w)[["se"]] -
                  summary(lm(y ~ x, data = cbind(d, y = y),
                             weights = w^delta_fit(X, y, d$w)[["delta"]]))$coefficients[2, 2]),
    hc = {
      ct <- clubSandwich::coef_test(l, vcov = "CR2", cluster = seq_len(nrow(d)), test = "Satterthwaite",
                                    inverse_var = TRUE)
      hf <- hc_fit(X, y, d$w^0.7)
      c(hc2_diff = abs(hf[["se_hc2"]] - ct$SE[2]), bm_df_diff = abs(hf[["df_bm"]] - ct$df_Satt[2]),
        hc3_diff = abs(hf[["se_hc3"]] - sqrt(sandwich::vcovHC(l, type = "HC3")[2, 2])))
    })
})

# ---- simulation ------------------------------------------------------------
RNGkind("L'Ecuyer-CMRG")
seeds <- vector("list", nrow(cells))
for (n in ns) {
  set.seed(20261001 + n)
  s <- .Random.seed
  for (k in which(cells$n == n)) { s <- nextRNGStream(s); seeds[[k]] <- s }
}

run_cell <- function(k) {
  assign(".Random.seed", seeds[[k]], envir = .GlobalEnv)
  d <- designs[[k]]; X <- cbind(1, d$x); n <- nrow(X)
  v <- curves[[cells$curve[k]]](d$w)
  se_or <- sqrt(solve(crossprod(X, X / v))[2, 2])
  b1 <- 2 * se_or
  res <- vector("list", R)
  for (r in seq_len(R)) {
    e <- rnorm(n, sd = sqrt(v))
    one <- function(b) {
      y <- b * d$x + e
      df <- tryCatch(delta_fit(X, y, d$w), error = function(err) c(delta = NA, beta = NA, se = NA, df = NA))
      or <- oracle(X, y, v)
      td <- wls_fit(X, y, d$w)
      uw <- wls_fit(X, y, rep(1, n))
      th <- hc_fit(X, y, d$w)
      dh <- if (is.na(df[["delta"]])) c(beta = NA, se_hc2 = NA, df_bm = NA) else hc_fit(X, y, d$w^df[["delta"]])
      data.frame(n = cells$n[k], cell = cells$id[k], rep = r, beta_true = b,
                 delta_hat = df[["delta"]], beta_delta = df[["beta"]], se_delta = df[["se"]],
                 df_delta = df[["df"]], beta_today = td[["beta"]], se_today = td[["se"]],
                 df_today = td[["df"]], beta_unw = uw[["beta"]], se_unw = uw[["se"]],
                 df_unw = uw[["df"]],
                 beta_today_hc2 = th[["beta"]], se_today_hc2 = th[["se_hc2"]], df_today_hc2 = th[["df_bm"]],
                 beta_today_hc3 = th[["beta"]], se_today_hc3 = th[["se_hc3"]], df_today_hc3 = th[["df"]],
                 beta_delta_hc2 = dh[["beta"]], se_delta_hc2 = dh[["se_hc2"]], df_delta_hc2 = dh[["df_bm"]],
                 beta_oracle = or[["beta"]], se_oracle = or[["se"]])
    }
    res[[r]] <- rbind(one(0), one(b1))
  }
  do.call(rbind, res)
}
t0 <- proc.time()[3]
reps <- do.call(rbind, mclapply(seq_len(nrow(cells)), run_cell, mc.cores = 4, mc.preschedule = FALSE))
t_sim <- proc.time()[3] - t0

meths <- c("delta", "today", "unw", "today_hc2", "today_hc3", "delta_hc2")
for (m in meths) {
  q <- qt(0.975, reps[[paste0("df_", m)]]); se <- reps[[paste0("se_", m)]]; b <- reps[[paste0("beta_", m)]]
  reps[[paste0("rej_", m)]] <- abs(b) / se > q
  reps[[paste0("width_", m)]] <- 2 * q * se
  reps[[paste0("cover_", m)]] <- abs(b - reps$beta_true) / se <= q
}
reps$rej_oracle <- abs(reps$beta_oracle) / reps$se_oracle > qnorm(0.975)
reps$width_oracle <- 2 * qnorm(0.975) * reps$se_oracle

summ <- do.call(rbind, lapply(split(reps, factor(paste(reps$n, reps$cell),
                                                  levels = paste(cells$n, cells$id))), function(z) {
  z0 <- z[z$beta_true == 0, ]; z1 <- z[z$beta_true != 0, ]
  per <- lapply(meths, function(m) {
    o <- data.frame(mean(z0[[paste0("rej_", m)]], na.rm = TRUE),
                    mean(z0[[paste0("width_", m)]], na.rm = TRUE) / mean(z0$width_oracle),
                    sd(z0[[paste0("beta_", m)]], na.rm = TRUE) / z0$se_oracle[1],
                    mean(z1[[paste0("rej_", m)]], na.rm = TRUE),
                    mean(z1[[paste0("cover_", m)]], na.rm = TRUE))
    setNames(o, paste0(c("typeI_", "width_ratio_", "sd_ratio_", "power_", "coverage_"), m))
  })
  data.frame(n = z$n[1], cell = z$cell[1], do.call(cbind, per),
             typeI_oracle = mean(z0$rej_oracle),
             power_oracle = mean(z1$rej_oracle),
             delta_mean = mean(z0$delta_hat, na.rm = TRUE),
             delta_sd = sd(z0$delta_hat, na.rm = TRUE),
             at_bound = mean(pmin(abs(z0$delta_hat + 1), abs(z0$delta_hat - 2)) < 0.01, na.rm = TRUE),
             errors = sum(is.na(z$beta_delta)))
}))
rownames(summ) <- NULL

write.csv(reps, file.path(out, sprintf("wls_delta_reps_R%d.csv", R)), row.names = FALSE)
write.csv(summ, file.path(out, sprintf("wls_delta_summary_R%d.csv", R)), row.names = FALSE)
mcse <- sqrt(0.05 * 0.95 / R)
sink(file.path(out, sprintf("wls_delta_R%d.txt", R)))
cat(R.version.string, "\n")
cat(sprintf("R = %d per cell  n = %s  data seed 20261001 + n (L'Ecuyer-CMRG, one stream per cell); design seed 20261039\n",
            R, paste(ns, collapse = ", ")))
cat(sprintf("MC SE of a 0.05 rate = %.4f   0.05 + 2 MC SE = %.4f\n", mcse, 0.05 + 2 * mcse))
cat(sprintf("simulation time %.1f s (4 cores); total %.1f s\n\n", t_sim, proc.time()[3] - t_start))
cat("== checks ==\n")
cat(sprintf("|REML logLik, lm.wfit route − lm| = %.2e   |δ-fit SE − summary.lm SE| = %.2e\n",
            chk[["ll_diff"]], chk[["se_diff"]]))
cat(sprintf("|HC2 SE − clubSandwich CR2| = %.2e   |BM df − clubSandwich df| = %.2e   |HC3 SE − sandwich HC3| = %.2e\n\n",
            chk[["hc.hc2_diff"]], chk[["hc.bm_df_diff"]], chk[["hc.hc3_diff"]]))
cat("== per cell (Type I and width at β = 0; power and coverage at β = 2 × oracle SE) ==\n")
print(format(summ, digits = 3), row.names = FALSE)
for (n in ns) {
  z <- summ[summ$n == n, ]
  cat(sprintf("\n== over cells, n = %d ==\n", n))
  for (m in meths) {
    ti <- z[[paste0("typeI_", m)]]; wr <- z[[paste0("width_ratio_", m)]]
    cat(sprintf("%-5s Type I mean %.4f  worst %.3f (%s)  cells > 0.075: %d  cells > 0.05 + 2 MC SE: %d  width mean %.3f worst %.3f  power mean %.3f\n",
                m, mean(ti), max(ti), z$cell[which.max(ti)], sum(ti > 0.075), sum(ti > 0.05 + 2 * mcse),
                mean(wr), max(wr), mean(z[[paste0("power_", m)]])))
  }
  cat(sprintf("oracle Type I mean %.4f  power mean %.3f\n", mean(z$typeI_oracle), mean(z$power_oracle)))
  cat(sprintf("errors: %d\n", sum(z$errors)))
}
sink()
cat(readLines(file.path(out, sprintf("wls_delta_R%d.txt", R))), sep = "\n")
