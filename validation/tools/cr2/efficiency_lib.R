# Copy of weights_of_trust/checks/efficiency_lib.R with the cut of spec 2-dcr §3.2
# (bmh() below); see validation/manifest.json `copied_tools`.
# Shared helpers for efficiency_mc.R, efficiency_followup_mc.R and
# formula_search_mc.R: the fastglmm fit, the B^(-1/2) of CR2 with its cut, the CR2 SE
# with Pustejovsky–Tipton df, the REML profile fit over one weight parameter,
# and the follow-up run's noise processes, cells and streams. Sourcing this file
# draws no random numbers. The calling script loads fastglmm. efficiency_mc.R
# defines its own make_cells() after sourcing, which replaces the one here.

# fastglmm evaluates `weights` in `data` first, so the argument must not share
# a name with a data column (here `w`).
fit <- function(d, wts, satt = TRUE, form = y ~ x + (1 | g)) {
  fastglmm(form, d, weights = wts, satterthwaite = satt)
}

# B^(-1/2) for B = D C Dᵀ (D upper, Φ = DᵀD), on the eigenvectors of B's largest
# eigenvalues, as many as T = D⁻ᵀ C D⁻¹ has above 1e-12; the rest set to 0. T has
# no units and its eigenvalues lie in [0, 1], so the cut is scale-free.
bmh <- function(B, D, C) {
  Di <- backsolve(D, diag(nrow(D)))
  rk <- sum(eigen(crossprod(Di, C %*% Di), symmetric = TRUE, only.values = TRUE)$values > 1e-12)
  e <- eigen(B, symmetric = TRUE)
  v <- e$vectors[, seq_len(rk), drop = FALSE]
  v %*% (e$values[seq_len(rk)]^(-1 / 2) * t(v))
}

# CR2 SE and Pustejovsky–Tipton df for the coefficient in column j of X, working
# model Φ = tau2·ZZᵀ + diag(s2v), clusters given by the row index list `idx`.
# Also returns the full CR2 vcov, |GLS β̂ under Φ − beta| for column j (~0 when
# beta is the fit's own estimate), tr(Q) (the expected CR2 variance under Φ),
# M = (XᵀΦ⁻¹X)⁻¹ and the per-cluster matrices E_k = M X_kᵀ Φ_k⁻¹ A_k (p × n_k),
# so that the CR2 vcov of residuals e is Σ_k E_k e_k e_kᵀ E_kᵀ.
cr2 <- function(X, y, beta, tau2, s2v, idx, j = 2) {
  Phi <- lapply(idx, function(i) tau2 + diag(s2v[i], length(i)))
  Wg <- lapply(Phi, solve)
  M <- solve(Reduce(`+`, Map(function(i, w) crossprod(X[i, , drop = FALSE], w %*% X[i, , drop = FALSE]), idx, Wg)))
  b_gls <- M %*% Reduce(`+`, Map(function(i, w) crossprod(X[i, , drop = FALSE], w %*% y[i]), idx, Wg))
  e <- y - X %*% beta
  cc <- replace(numeric(ncol(X)), j, 1)
  v <- matrix(0, ncol(X), length(idx)); U <- E_k <- vector("list", length(idx))
  for (k in seq_along(idx)) {
    i <- idx[[k]]; Xk <- X[i, , drop = FALSE]
    D <- chol(Phi[[k]]); C <- Phi[[k]] - Xk %*% M %*% t(Xk)
    B <- D %*% C %*% t(D)
    A <- t(D) %*% bmh(B, D, C) %*% D
    E <- M %*% t(Xk) %*% Wg[[k]] %*% A        # p × n_k
    v[, k] <- E %*% e[i]
    U[[k]] <- drop(crossprod(E, cc))          # u_k = A W X M c
    E_k[[k]] <- E
  }
  V <- tcrossprod(v)
  uPu <- mapply(function(u, P) drop(crossprod(u, P %*% u)), U, Phi)
  UX <- t(mapply(function(u, i) drop(crossprod(u, X[i, , drop = FALSE])), U, idx))
  Q <- diag(uPu, length(idx)) - UX %*% M %*% t(UX)
  list(se = sqrt(V[j, j]), df = sum(diag(Q))^2 / sum(Q^2), vcov = V,
       gls_diff = abs(b_gls[j] - beta[j]), trQ = sum(diag(Q)), M = M, E = E_k)
}

# REML profile over one weight parameter p: maximize logLik(fit(d, wfun(p))) on
# `interval` (optimize, tol 1e-4), then refit with wfun(p̂) and Satterthwaite df.
# Returns that fit with $p_hat and $prof_nonconv (non-converged profile fits).
profile_fit <- function(d, wfun, interval, form = y ~ x + (1 | g)) {
  nc <- 0L
  prof <- function(p) {
    m <- fit(d, wfun(p), FALSE, form)
    if (!isTRUE(m$converged)) nc <<- nc + 1L
    logLik(m)[1]
  }
  ph <- optimize(prof, interval, maximum = TRUE, tol = 1e-4)$maximum
  m <- fit(d, wfun(ph), form = form); m$p_hat <- ph; m$prof_nonconv <- nc; m
}

# ---- the follow-up run's designs (efficiency_followup_mc.R) -------------------
# Noise SD per row.
procs_a <- list(precision = function(w) 1 / sqrt(w),
                honest    = function(w) rep(1, length(w)),
                partial   = function(w) w^(-1 / 4),
                floor     = function(w) sqrt(0.5 + 0.5 / w),
                linear    = function(w) sqrt(2 - w))
procs_b <- list(precision = function(w) 1 / sqrt(w),
                honest    = function(w) rep(1, length(w)),
                partial   = function(w) w^(-1 / 4),
                floor_mid = function(w) sqrt(1 + 10^(-1.5) / w),
                bump      = function(w) sqrt(1 + 9 * exp(-(log10(w) + 1.5)^2 / 0.5)))
procs_c <- procs_a[c("precision", "honest")]
make_cells <- function(G, draw) {
  design <- sprintf("%dx10", G)
  g <- factor(rep(seq_len(G), each = 10))
  x <- rnorm(10 * G)
  w_pat <- list(continuous = exp(runif(10 * G, log(0.05), 0)),
                near01 = sample(rep(c(0.05, 1), 5 * G)))
  junk <- rbind(data.frame(g = g, x = rnorm(10 * G), w = 1),
                data.frame(g = factor(rep(seq_len(G), times = 5)), x = rnorm(5 * G), w = 1e-8))
  w_wide <- exp(runif(10 * G, log(0.001), 0))
  cell <- function(part, pattern, process, d, noise_sd)
    list(part = part, design = design, draw = draw, pattern = pattern, process = process,
         d = d, noise_sd = noise_sd, att = part == "C")
  cells <- list()
  for (pat in names(w_pat)) for (p in names(procs_a)) {
    d <- data.frame(g = g, x = x, w = w_pat[[pat]])
    cells[[length(cells) + 1]] <- cell("A", pat, p, d, procs_a[[p]](d$w))
  }
  cells[[length(cells) + 1]] <- cell("A", "junk", "junk", junk, ifelse(junk$w == 1, 1, 3))
  d <- data.frame(g = g, x = x, w = w_wide)
  for (p in names(procs_b)) cells[[length(cells) + 1]] <- cell("B", "wide", p, d, procs_b[[p]](d$w))
  d <- data.frame(g = g, x = x, w = w_pat$continuous)
  for (p in names(procs_c)) cells[[length(cells) + 1]] <- cell("C", "continuous", p, d, procs_c[[p]](d$w))
  cells
}

# The follow-up run's cells and streams: seed 20261031, L'Ecuyer-CMRG; one
# stream per design draw, then one per cell, then one for the CR2 checks.
# Returns the cells (sorted by part), their ids, their seeds and the check stream.
followup_cells <- function() {
  RNGkind("L'Ecuyer-CMRG")
  set.seed(20261031)
  s <- .Random.seed
  cells <- list()
  for (G in c(10, 50)) for (dr in 1:5) {
    s <- nextRNGStream(s); assign(".Random.seed", s, envir = .GlobalEnv)
    cells <- c(cells, make_cells(G, dr))
  }
  cells <- cells[order(vapply(cells, `[[`, "", "part"), seq_along(cells))]
  cell_id <- vapply(cells, function(k) paste(k$part, k$design, paste0("d", k$draw), k$pattern,
                                             k$process, sep = "/"), "")
  seeds <- vector("list", length(cells))
  for (k in seq_along(cells)) { s <- nextRNGStream(s); seeds[[k]] <- s }
  list(cells = cells, cell_id = cell_id, seeds = seeds, s_check = nextRNGStream(s))
}

# True V and the oracle's design: (1, x), or (1, w, w·x) in Part C.
true_v <- function(cell) outer(as.integer(cell$d$g), as.integer(cell$d$g), "==") + diag(cell$noise_sd^2)
oracle_x <- function(cell) with(cell$d, if (cell$att) cbind(1, w, w * x) else cbind(1, x))
