# Uses the scale-free cut of efficiency_lib.R's bmh(); see validation/manifest.json `copied_tools`.
# One more arm for f2_extensions_mc.R, on the same data, 10-group cells only:
#   F1S slope_cr2  δ profiled on y ~ x + (1 + x | g); CR2 SE and Pustejovsky–Tipton df with
#                  that fit as the working model. Gap: F1's df come from a working model with
#                  no slope, so they do not fall when slopes are real.
# Cells: the 10 × 10 cells of Parts A, B, D, E and S (175). Measures as in f2_extensions_mc.R.
# This file takes the cells, seeds and helpers from f2_extensions_mc.R (every line before its
# `# The S and G cells run first`), so the two cannot drift. cr2_phi() is cr2() of
# efficiency_lib.R with Φ_k given.
# Run: Rscript f2_slope_cr2_mc.R [R] [cores] [cache_dir]
here <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))))
src <- readLines(file.path(here, "f2_extensions_mc.R"), encoding = "UTF-8")
eval(parse(text = src[seq_len(grep("^# The S and G cells run first", src) - 1)], encoding = "UTF-8"))

cr2_phi <- function(X, y, beta, Phi, idx, j = 2) {
  Wg <- lapply(Phi, solve)
  M <- solve(Reduce(`+`, Map(function(i, w) crossprod(X[i, , drop = FALSE], w %*% X[i, , drop = FALSE]), idx, Wg)))
  e <- y - X %*% beta
  cc <- replace(numeric(ncol(X)), j, 1)
  v <- matrix(0, ncol(X), length(idx)); U <- vector("list", length(idx))
  for (k in seq_along(idx)) {
    i <- idx[[k]]; Xk <- X[i, , drop = FALSE]
    D <- chol(Phi[[k]]); C <- Phi[[k]] - Xk %*% M %*% t(Xk)
    B <- D %*% C %*% t(D)
    A <- t(D) %*% bmh(B, D, C) %*% D
    E <- M %*% t(Xk) %*% Wg[[k]] %*% A
    v[, k] <- E %*% e[i]
    U[[k]] <- drop(crossprod(E, cc))
  }
  V <- tcrossprod(v)
  uPu <- mapply(function(u, P) drop(crossprod(u, P %*% u)), U, Phi)
  UX <- t(mapply(function(u, i) drop(crossprod(u, X[i, , drop = FALSE])), U, idx))
  Q <- diag(uPu, length(idx)) - UX %*% M %*% t(UX)
  list(se = sqrt(V[j, j]), df = sum(diag(Q))^2 / sum(Q^2))
}

one <- function(d, y, idx) {
  d$y <- y; X <- cbind(1, d$x)
  within_x <- any(tapply(d$x, d$g, function(v) diff(range(v))) > 0)
  form <- if (within_x) y ~ x + (1 + x | g) else y ~ x + (1 | g)
  tryCatch({
    ms <- profile_fit(d, function(p) d$w^p, lim_d, form)
    vc <- ms$varcorr[[1]]; s2 <- sigma(ms)^2
    Gm <- if (within_x) matrix(vc[c(1, 2, 2, 3)], 2) else diag(c(vc[1], 0))
    Phi <- lapply(idx, function(i) {
      Z <- cbind(1, d$x[i]); Z %*% Gm %*% t(Z) + diag(s2 * d$w[i]^(-ms$p_hat), length(i))
    })
    c2 <- cr2_phi(X, y, unname(ms$beta), Phi, idx)
    c(est = ms$beta[["x"]], se = c2$se, df = c2$df, sing = isSingular(ms))
  }, error = function(e) c(est = NA, se = NA, df = NA, sing = NA))
}

run_c <- function(k) {
  f <- file.path(cache, sprintf("s_%03d.rds", k))
  if (!is.na(cache) && file.exists(f)) return(readRDS(f))
  assign(".Random.seed", seeds[[k]], envir = .GlobalEnv)
  cell <- cells[[k]]; d <- cell$d; n <- nrow(d); G <- nlevels(d$g)
  idx <- split(seq_len(n), d$g); gi <- as.integer(d$g)
  Xo <- oracle_x(cell); j <- ncol(Xo)
  sl <- rep_len(cell$slope_sd %||% 0, G)
  Vt <- true_v(cell)
  if (any(sl > 0)) Vt <- Vt + outer(sl[gi] * d$x, sl[gi] * d$x) * outer(gi, gi, "==")
  Mo <- solve(crossprod(Xo, solve(Vt, Xo))); orc <- sqrt(Mo[j, j]); beta1 <- 2 * orc
  m <- matrix(NA_real_, R, 8)
  for (r in seq_len(R)) {
    y0 <- rnorm(G)[gi] + cell$noise_sd * rnorm(n)
    if (any(sl > 0)) y0 <- y0 + (sl * rnorm(G))[gi] * d$x
    m[r, ] <- c(one(d, y0, idx), one(d, y0 + beta1 * Xo[, j], idx))
  }
  ok <- !is.na(m[, 2]); ok1 <- !is.na(m[, 6])
  z <- qt(0.975, m[, 3])
  o <- data.frame(cell = cell_id[k], part = cell$part, draw = cell$draw, pattern = cell$pattern,
                  process = cell$process, n_ok = sum(ok), type1 = mean((abs(m[, 1] / m[, 2]) > z)[ok]),
                  power = mean((abs(m[, 5] / m[, 6]) > qt(0.975, m[, 7]))[ok1]),
                  rho = var(m[ok, 1]) / mean(m[ok, 2]^2), width = mean((z * m[, 2])[ok]) / (qnorm(0.975) * orc),
                  df = exp(mean(log(m[ok, 3]))), singular = mean(m[ok, 4]))
  if (!is.na(cache)) { tmp <- paste0(f, ".tmp"); saveRDS(o, tmp); file.rename(tmp, f) }
  o
}
todo <- which(grepl("^(A|B|D|E|S)/10x10/", cell_id))
res <- mclapply(todo, function(k) tryCatch(run_c(k), error = function(e) e), mc.cores = cores, mc.preschedule = FALSE)
bad <- vapply(res, inherits, NA, "error")
if (any(bad)) {
  for (i in which(bad)) cat("cell failed:", cell_id[todo[i]], conditionMessage(res[[i]]), "\n")
  stop("cells failed")
}
out_s <- do.call(rbind, res)
write.csv(out_s, file.path(out, sprintf("f2_slope_cr2_summary_R%d.csv", R)), row.names = FALSE)
cat(sprintf("elapsed %.1f min, %d cells\n", as.numeric(Sys.time() - t_start, units = "mins"), length(todo)))
