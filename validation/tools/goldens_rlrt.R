#!/usr/bin/env Rscript
# RLRsim exact RLRT references -> validation/goldens/inference/rlrt/<name>.json,
# one per `kind = "rlrt"` entry of the manifest's `inference_references` block.
#
# THE ORACLE IS SACRED. These JSONs are the frozen reference the crate's exact
# RLRT is validated against. On any glmm disagreement, glmm is presumed wrong:
# never relax a tolerance or edit a reference.
#
# A BARE RUN REFITS AND OVERWRITES EVERY RLRT REFERENCE. Use VALIDATION_ONLY for
# anything short of a deliberate full re-freeze:
#   VALIDATION_ONLY=<name>[,<name>...] Rscript validation/tools/goldens_rlrt.R
#
# Per entry (seed, nsim, paired_draws and weights_col come from the entry): the
# REML fit of the one-component LMM, its fixed-effects-only reduction, RLRsim's
# exactRLRT at `nsim` draws (observed statistic, p-value, the null stored sorted
# as the count of zero draws plus the positive draws), the inputs RLRsim's
# RLRTSim builds but does not return (normalized eigenvalues `mu`, their maximum,
# the lambda grid), and, when `paired_draws > 0`, the first `paired_draws` draws'
# inputs (ChiK, Chi1) and outputs (res, lambdaind). RLRsimCpp returns only the
# last draw's inputs, so the per-draw inputs come from an R replay of its
# rchisq calls, checked bitwise against that last draw. An entry with
# `weights_col` gets no RLRsim run (RLRsim ignores prior weights): only n, p and
# the two REML log-likelihoods.

suite_dir <- normalizePath(file.path(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))), ".."))
source(file.path(suite_dir, "tools", "inference_common.R"))
inference_setup()
assert_inference_versions()

out_dir <- file.path(suite_dir, "goldens", "inference", "rlrt")

# Stops unless every value of `x` is finite, so no Inf, NaN or NA reaches a JSON.
finite <- function(x, what, name) {
  if (!all(is.finite(unlist(x)))) stop(sprintf("%s: %s is not finite", name, what), call. = FALSE)
  x
}

rlrt_reference <- function(entry) {
  name <- entry$name
  data <- read_entry_data(entry)
  f <- as.formula(entry$r_formula)
  f0 <- reformulas::nobars(f)

  if (!is.null(entry$weights_col)) {
    wts <- data[[entry$weights_col]]
    m <- lme4::lmer(f, data = data, REML = TRUE, weights = wts)
    red <- lm(f0, data = data, weights = wts)
    X <- lme4::getME(m, "X")
    return(list(
      provenance = provenance("goldens_rlrt.R"),
      n = nrow(X),
      p = ncol(X),
      loglik_reml = finite(as.numeric(logLik(m)), "loglik_reml", name),
      loglik_lm_reml = finite(as.numeric(logLik(red, REML = TRUE)), "loglik_lm_reml", name)
    ))
  }

  seed <- as.integer(entry$seed)
  nsim <- as.integer(entry$nsim)
  paired_draws <- as.integer(entry$paired_draws)
  m <- lme4::lmer(f, data = data, REML = TRUE)
  red <- lm(f0, data = data)

  ex <- RLRsim::exactRLRT(m, seed = seed, nsim = nsim)
  if (is.null(ex$sample)) stop(sprintf("%s: exactRLRT returned no null sample", name), call. = FALSE)
  null <- as.vector(ex$sample)

  # RLRTSim's and exactRLRT's own expressions (lambda0 = 0 branch of make.lambdagrid).
  d <- RLRsim:::extract.lmerModDesign(m)
  Vr <- d$Vr
  if (all(Vr == 0)) diag(Vr) <- 1
  mu <- svd(chol(cov2cor(Vr)) %*% t(qr.resid(qr(d$X), d$Z)), nu = 0, nv = 0)$d^2
  max_mu <- max(mu)
  mu <- mu / max_mu
  grid <- c(0, exp(seq(-10, 8, length = 199)))
  n <- nrow(d$X)
  p <- ncol(d$X)
  K <- min(n, ncol(d$Z))

  null_pos <- sort(null[null > 0])
  null_n_zero <- sum(null == 0)
  if (null_n_zero + length(null_pos) != nsim) {
    stop(sprintf("%s: the null has negative or missing draws", name), call. = FALSE)
  }

  out <- list(
    provenance = provenance("goldens_rlrt.R"),
    n = n, p = p, K = K,
    mu = finite(mu, "mu", name),
    max_mu = finite(max_mu, "max_mu", name),
    grid = grid,
    seed = seed,
    nsim = nsim,
    loglik_reml = finite(as.numeric(logLik(m)), "loglik_reml", name),
    loglik_lm_reml = finite(as.numeric(logLik(red, REML = TRUE)), "loglik_lm_reml", name),
    statistic = finite(unname(ex$statistic), "statistic", name),
    p_value = finite(ex$p.value, "p_value", name),
    null_n_zero = null_n_zero,
    null_pos = finite(null_pos, "null_pos", name)
  )

  if (paired_draws > 0L) {
    set.seed(seed)
    direct <- RLRsim:::RLRsimCpp(p = as.integer(p), k = as.integer(K), n = as.integer(n),
                                 nsim = as.integer(nsim), g = 200L, q = 0L,
                                 mu = as.double(mu), lambda = as.double(grid),
                                 lambda0 = 0, xi = as.double(mu), REML = TRUE)
    if (!identical(direct$res, null)) {
      stop(sprintf("%s: RLRsimCpp's res differs from exactRLRT's sample", name), call. = FALSE)
    }
    if (!identical(attr(ex$sample, "lambda"), grid[direct$lambdaind])) {
      stop(sprintf("%s: RLRsimCpp's lambdaind does not index the sample's lambda", name),
           call. = FALSE)
    }
    set.seed(seed)
    chik <- numeric(paired_draws)
    chi1 <- matrix(0, nrow = paired_draws, ncol = K)
    for (b in seq_len(nsim)) {
      ck <- rchisq(1, n - p - K)
      c1 <- rchisq(K, 1)
      if (b <= paired_draws) {
        chik[b] <- ck
        chi1[b, ] <- c1
      }
    }
    if (!identical(ck, direct$ChiK) || !identical(c1, as.vector(direct$Chi1))) {
      stop(sprintf("%s: the replayed last draw differs from RLRsimCpp's", name), call. = FALSE)
    }
    lambdaind <- direct$lambdaind[seq_len(paired_draws)]
    if (!all(lambdaind == round(lambdaind))) {
      stop(sprintf("%s: lambdaind is not integer", name), call. = FALSE)
    }
    out$paired <- list(
      ChiK = finite(chik, "paired ChiK", name),
      Chi1 = finite(chi1, "paired Chi1", name),
      res = finite(direct$res[seq_len(paired_draws)], "paired res", name),
      lambdaind = as.integer(lambdaind)
    )
  }

  # Balanced intercept-only design: the one-way ANOVA F test and the atom P(F <= 1).
  groups <- m@flist[[1]]
  if (p == 1L && length(unique(table(groups))) == 1L) {
    y <- model.response(model.frame(m))
    F_obs <- anova(lm(y ~ groups))[["F value"]][1]
    out$F_obs <- finite(F_obs, "F_obs", name)
    out$F_p <- finite(pf(F_obs, K - 1, n - K, lower.tail = FALSE), "F_p", name)
    out$F_atom <- finite(pf(1, K - 1, n - K), "F_atom", name)
  }
  out
}

for (entry in select_entries("rlrt")) {
  cat(sprintf("%s ...\n", entry$name))
  write_reference(rlrt_reference(entry), file.path(out_dir, paste0(entry$name, ".json")))
}
