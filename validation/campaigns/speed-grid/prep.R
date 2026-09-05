#!/usr/bin/env Rscript
# Optimizer-grid campaign generator (design: docs/GLMM 2026-07-09 grid campaign):
# writes manifest.json (committed — THE reproducibility artifact; CSVs are
# derived) and data/<case_id>.csv (gitignored). Deliberately
# non-full crossing: core = structure x size x {balanced} x {baseline}; balance
# and regime variants are added for a deterministic ~30% stratified subset.
# One data seed per cell; b1-flagged cells get 5 seed replicates (suffix _s1.._s5).
#
# Nesting is realized with globally-unique inner labels + plain intercept terms
# ((1|g1)+(1|g2), g2 labels unique within g1) rather than the `/` operator --
# semantically identical, and it sidesteps engine-specific nesting syntax
# (the formula frontend has no `/`; the tier-0 corpus uses the same unique-label trick).
suppressMessages({ library(jsonlite); library(MASS) })

here_dir <- normalizePath(dirname(sub(
  "--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))))
grid_dir <- file.path(here_dir, "data")
dir.create(grid_dir, showWarnings = FALSE, recursive = TRUE)

SEED_BASE <- 811000L  # grid seed namespace; per-cell seed = SEED_BASE + cell index

# ---- structure catalog -------------------------------------------------------
# Each structure: RE terms as a list of q values (q = 1 intercept-only, q >= 2
# intercept + q-1 slopes on x1..x_{q-1}); `nested` = TRUE chains every later
# term inside the previous one (unique labels), an integer m chains terms 2..m
# and leaves the rest crossed. n_theta = sum q(q+1)/2. The ladder targets
# {1,2,3,5,8,12,18,27,40} are realized via slope/factor counts (design "Axes");
# neighbors fill the gaps so aggregate curves have support between targets.
STRUCTURES <- list(
  int1    = list(q = c(1),                      nested = FALSE, glmm = TRUE),   # nt 1
  int2x   = list(q = c(1, 1),                   nested = FALSE, glmm = TRUE),   # nt 2
  nest2   = list(q = c(1, 1),                   nested = TRUE,  glmm = TRUE),   # nt 2
  q2s     = list(q = c(2),                      nested = FALSE, glmm = TRUE),   # nt 3
  nest3   = list(q = c(1, 1, 1),                nested = TRUE,  glmm = FALSE),  # nt 3
  nestmix = list(q = c(1, 1, 1),                nested = 2L,    glmm = FALSE),  # nt 3 (g2 in g1, g3 crossed)
  cross4  = list(q = c(1, 1, 1, 1),             nested = FALSE, glmm = TRUE),   # nt 4
  nest2s  = list(q = c(2, 1),                   nested = TRUE,  glmm = TRUE),   # nt 4 (slope primary, nested inner)
  q2sx2   = list(q = c(2, 1, 1),                nested = FALSE, glmm = FALSE),  # nt 5
  q3s     = list(q = c(3),                      nested = FALSE, glmm = FALSE),  # nt 6
  cross6  = list(q = c(1, 1, 1, 1, 1, 1),       nested = FALSE, glmm = TRUE),   # nt 6 (many-crossed)
  q2sq2s  = list(q = c(2, 2),                   nested = FALSE, glmm = FALSE),  # nt 6 (two slope blocks)
  cross8  = list(q = c(1, 1, 1, 1, 1, 1, 1, 1), nested = FALSE, glmm = FALSE),  # nt 8
  q3sx2   = list(q = c(3, 1, 1),                nested = FALSE, glmm = FALSE),  # nt 8
  q4      = list(q = c(4),                      nested = FALSE, glmm = FALSE),  # nt 10
  q4sx2   = list(q = c(4, 1, 1),                nested = FALSE, glmm = FALSE),  # nt 12
  q5q2    = list(q = c(5, 2),                   nested = FALSE, glmm = FALSE),  # nt 18
  q6      = list(q = c(6),                      nested = FALSE, glmm = FALSE),  # nt 21
  q6q2x3  = list(q = c(6, 2, 1, 1, 1),          nested = FALSE, glmm = FALSE),  # nt 27
  q8      = list(q = c(8),                      nested = FALSE, glmm = FALSE),  # nt 36 (sim_max_q_slope's class)
  q8q2x   = list(q = c(8, 2, 1),                nested = FALSE, glmm = FALSE)   # nt 40
)
n_theta_of <- function(st) sum(vapply(st$q, function(q) q * (q + 1) / 2, 0))

# ---- size grid (n_obs x obs-per-primary-group, feasible combos only) ---------
SIZES <- list(
  c(300, 5), c(300, 20),
  c(3000, 5), c(3000, 20), c(3000, 100),
  c(30000, 5), c(30000, 20), c(30000, 100)
)  # (300,100) -> 3 groups: never feasible, excluded up front

# GLMM arms: subset of structures/sizes (design: ~100-150 GLMM cells)
GLMM_SIZES <- list(c(300, 20), c(3000, 20), c(3000, 100))
GLMM_ARMS  <- c("binomial_bin", "binomial_agg", "poisson")

BALANCES <- c("skew", "single")            # variants; "bal" is the core
REGIMES  <- c("nearzero", "highcorr", "lowsnr")  # variants; "base" is the core
                                            # + per-family extras (rare / lowmu)

# ---- feasibility -------------------------------------------------------------
# Primary grouping must have enough levels to identify its q x q block, extras
# need >= 6 levels. Extras get min(30, max(6, G %/% 2)) levels (8-level tier-0
# convention scaled up so 30k-obs cells don't saturate).
feasible <- function(st, n_obs, per) {
  G <- n_obs %/% per
  q1 <- st$q[1]
  G >= max(q1 * (q1 + 1) / 2 + 2, 2 * q1 + 2) && G >= 6
}
extra_levels <- function(G) min(30L, max(6L, G %/% 2L))

# ---- simulation --------------------------------------------------------------
# Baseline sd ladder per term (descending, like the tier-0 sim recipes), corr 0.2
# inside slope blocks; residual sd 0.6 (gaussian). Regimes perturb exactly one
# thing each (design "parameter regime" axis).
sd_ladder <- function(q, k) 0.9^(seq_len(q) - 1) * c(1.0, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5)[k]

sim_cell <- function(cell) {
  set.seed(cell$seed)
  st <- STRUCTURES[[cell$structure]]
  n <- cell$n_obs; per <- cell$per_group; G <- n %/% per
  nx <- max(1L, max(st$q) - 1L)   # covariates: enough for the widest slope block
  X <- matrix(rnorm(n * nx), n, nx, dimnames = list(NULL, paste0("x", seq_len(nx))))

  # primary factor assignment per balance level
  g1 <- switch(cell$balance,
    bal    = rep(seq_len(G), length.out = n),
    skew   = {  # 20% of groups carry 80% of observations
      heavy <- seq_len(max(1L, round(0.2 * G)))
      p <- ifelse(seq_len(G) %in% heavy, 4 / length(heavy), 1 / (G - length(heavy)))
      sample(seq_len(G), n, replace = TRUE, prob = p / sum(p))
    },
    single = {  # ~40% of rows in size-1-2 groups appended after the regulars
      n_reg <- round(0.6 * n); reg <- rep(seq_len(G), length.out = n_reg)
      n_sing <- n - n_reg
      sing <- G + rep(seq_len(ceiling(n_sing / 2)), each = 2)[seq_len(n_sing)]
      c(reg, sing)
    })
  g1 <- factor(g1)
  nl1 <- nlevels(g1)

  corr_val <- if (cell$regime == "highcorr") 0.9 else 0.2
  eta <- 0.5
  betas <- rep(c(0.8, -0.5, 0.3, -0.2, 0.4, -0.3, 0.2), length.out = nx)
  if (cell$regime == "lowsnr") betas <- betas * 0.25
  if (cell$family == "binomial_bin" || cell$family == "binomial_agg") {
    eta <- if (cell$regime == "rare") qlogis(0.02) else 0.2
  }
  if (cell$family == "poisson" || cell$family == "negbin")
    eta <- if (cell$regime == "lowmu") log(0.5) else 0.4
  if (cell$family == "gamma") eta <- 0.4
  for (j in seq_len(nx)) eta <- eta + betas[j] * X[, j]

  df <- data.frame(X)
  fac_names <- character(0)
  parent <- g1
  # nesting chain reach: TRUE = every term, integer m = terms 2..m, FALSE = none
  nested_upto <- if (isTRUE(st$nested)) length(st$q)
                 else if (is.numeric(st$nested)) as.integer(st$nested) else 1L
  for (k in seq_along(st$q)) {
    q <- st$q[k]
    if (k == 1) {
      f <- g1
    } else if (k <= nested_upto) {
      # nested chain: term k splits each level of term k-1 into 3 (unique labels)
      f <- factor(paste0(as.integer(parent), "_", sample(1:3, n, replace = TRUE)))
      parent <- f
    } else {
      f <- factor(sample(seq_len(extra_levels(nl1)), n, replace = TRUE))
    }
    nm <- if (k == 1) "g1" else paste0("g", k)
    df[[nm]] <- f
    fac_names <- c(fac_names, nm)
    nl <- nlevels(f)
    sds <- sd_ladder(q, k)
    if (cell$family != "gaussian") sds <- sds * 0.6      # keep link-scale sane
    if (cell$regime == "nearzero" && k == 1) sds[q] <- 0.02
    Sigma <- diag(sds, q) %*% (matrix(corr_val, q, q) + diag(1 - corr_val, q)) %*% diag(sds, q)
    b <- mvrnorm(nl, rep(0, q), Sigma); b <- matrix(b, nl, q)
    eta <- eta + b[as.integer(f), 1]
    if (q >= 2) for (d in 2:q) eta <- eta + b[as.integer(f), d] * X[, d - 1]
  }

  if (cell$family == "gaussian") {
    resid_sd <- if (cell$regime == "lowsnr") 3.0 else 0.6
    df$y <- eta + rnorm(n, sd = resid_sd)
  } else if (cell$family == "binomial_bin") {
    df$y <- rbinom(n, 1, plogis(eta))
  } else if (cell$family == "binomial_agg") {
    df$size <- sample(5:20, n, replace = TRUE)
    df$incidence <- rbinom(n, df$size, plogis(eta))
  } else if (cell$family == "poisson") {
    df$y <- rpois(n, exp(eta))
  } else if (cell$family == "negbin") {
    df$y <- MASS::rnegbin(n, mu = exp(eta), theta = 1.5)   # sim_nb convention (export_data.R)
  } else if (cell$family == "gamma") {
    df$y <- rgamma(n, shape = 2, scale = exp(eta) / 2)     # E[y] = mu, shape 2, as sim_gamma
  }
  list(df = df, factors = fac_names, n_x = nx)
}

# ---- formula emission ---------------------------------------------------------
formulas_of <- function(cell, meta) {
  st <- STRUCTURES[[cell$structure]]
  re <- vapply(seq_along(st$q), function(k) {
    q <- st$q[k]; nm <- if (k == 1) "g1" else paste0("g", k)
    if (q == 1) sprintf("(1 | %s)", nm)
    else sprintf("(1 + %s | %s)", paste(paste0("x", seq_len(q - 1)), collapse = " + "), nm)
  }, "")
  fx <- paste(paste0("x", seq_len(meta$n_x)), collapse = " + ")
  resp <- if (cell$family == "binomial_agg") "prop" else "y"
  rhs <- paste(c("1", fx, re), collapse = " + ")
  r_resp <- if (cell$family == "binomial_agg") "cbind(incidence, size - incidence)" else "y"
  list(r  = sprintf("%s ~ %s", r_resp, rhs),
       jl = sprintf("@formula(%s ~ %s)", resp, rhs))
}

# ---- cell enumeration ---------------------------------------------------------
cells <- list(); idx <- 0L
add_cell <- function(structure, family, n_obs, per, balance, regime, b1 = FALSE) {
  idx <<- idx + 1L
  st <- STRUCTURES[[structure]]
  fam_tag <- c(gaussian = "lmm", binomial_bin = "binb", binomial_agg = "bina",
               poisson = "pois", negbin = "nb", gamma = "gam")[family]
  cells[[length(cells) + 1L]] <<- list(
    case_id = sprintf("%s_%s_g%dp%d_%s_%s", fam_tag, structure, n_obs, per, balance, regime),
    family = family, structure = structure, n_theta = n_theta_of(st),
    n_obs = n_obs, per_group = per, balance = balance, regime = regime,
    seed = SEED_BASE + idx, max_fun = as.integer(500 * (n_theta_of(st) + 1)),
    b1 = b1)
}

# core LMM crossing
core_i <- 0L
for (sname in names(STRUCTURES)) for (sz in SIZES) {
  if (!feasible(STRUCTURES[[sname]], sz[1], sz[2])) next
  core_i <- core_i + 1L
  add_cell(sname, "gaussian", sz[1], sz[2], "bal", "base")
  # stratified ~30%: every 10-cell stripe's first 3 core cells spawn variants
  if (core_i %% 10L < 3L) {
    for (b in BALANCES) add_cell(sname, "gaussian", sz[1], sz[2], b, "base")
    for (r in REGIMES) {
      if (r == "highcorr" && max(STRUCTURES[[sname]]$q) < 2) next
      add_cell(sname, "gaussian", sz[1], sz[2], "bal", r)
    }
  }
}
# GLMM arms
glmm_i <- 0L
for (sname in names(STRUCTURES)) {
  if (!isTRUE(STRUCTURES[[sname]]$glmm)) next
  for (sz in GLMM_SIZES) for (fam in GLMM_ARMS) {
    if (!feasible(STRUCTURES[[sname]], sz[1], sz[2])) next
    glmm_i <- glmm_i + 1L
    add_cell(sname, fam, sz[1], sz[2], "bal", "base")
    if (glmm_i %% 10L < 3L) {
      add_cell(sname, fam, sz[1], sz[2], "skew", "base")
      extra <- if (fam == "poisson") "lowmu" else "rare"
      add_cell(sname, fam, sz[1], sz[2], "bal", extra)
      add_cell(sname, fam, sz[1], sz[2], "bal", "nearzero")
    }
  }
}
# NB and Gamma arms (P4 campaign prep, 2026-09-06): same structures, sizes and
# variant stripe as the GLMM arms above. Appended AFTER every earlier cell on
# purpose: seed = SEED_BASE + cell index, so inserting these into the loop above
# would reseed the whole grid and void the recorded baselines keyed on those
# cells (W0 counters, p1_per_cell). Gamma has no per-family extra regime.
P4_ARMS <- c("negbin", "gamma")
p4_i <- 0L
for (sname in names(STRUCTURES)) {
  if (!isTRUE(STRUCTURES[[sname]]$glmm)) next
  for (sz in GLMM_SIZES) for (fam in P4_ARMS) {
    if (!feasible(STRUCTURES[[sname]], sz[1], sz[2])) next
    p4_i <- p4_i + 1L
    add_cell(sname, fam, sz[1], sz[2], "bal", "base")
    if (p4_i %% 10L < 3L) {
      add_cell(sname, fam, sz[1], sz[2], "skew", "base")
      if (fam == "negbin") add_cell(sname, fam, sz[1], sz[2], "bal", "lowmu")
      add_cell(sname, fam, sz[1], sz[2], "bal", "nearzero")
    }
  }
}
# B1 subgrid: ~30 LMM cells spanning the ladder (5 seeds each), plus the GLMM
# satellite. Marked on EXISTING cells: every structure at (3000, 20) plus a
# ladder-spanning subset at (300, 20), balanced/baseline only. IDs that turn
# out infeasible simply never match (the flag loop is a no-op for them).
b1_ids <- sprintf("lmm_%s_g3000p20_bal_base", names(STRUCTURES))
b1_ids <- c(b1_ids, sprintf("lmm_%s_g300p20_bal_base",
  c("int1", "q2s", "q3s", "q4", "q5q2", "q6", "q8", "q8q2x")))
b1_ids <- c(b1_ids, "binb_q2s_g3000p20_bal_base", "pois_q2s_g3000p20_bal_base",
            "bina_int2x_g3000p20_bal_base", "pois_cross6_g3000p20_bal_base",
            "binb_nest2_g3000p20_bal_base", "pois_int1_g3000p20_bal_base")
for (i in seq_along(cells))
  if (cells[[i]]$case_id %in% b1_ids) cells[[i]]$b1 <- TRUE

# ---- generate data + finalize manifest ----------------------------------------
manifest <- list(schema = "glmm-grid-manifest/1",
                 generated_by = "campaigns/speed-grid/prep.R",
                 b1_seed_suffixes = paste0("s", 1:5),
                 cells = list())
for (cell in cells) {
  meta <- sim_cell(cell)
  write.csv(meta$df, file.path(grid_dir, paste0(cell$case_id, ".csv")), row.names = FALSE)
  if (isTRUE(cell$b1)) {
    file.copy(file.path(grid_dir, paste0(cell$case_id, ".csv")),
              file.path(grid_dir, paste0(cell$case_id, "_s1.csv")), overwrite = TRUE)
    for (s in 2:5) {
      c2 <- cell; c2$seed <- cell$seed + 100000L * (s - 1L)
      m2 <- sim_cell(c2)
      write.csv(m2$df, file.path(grid_dir, paste0(cell$case_id, "_s", s, ".csv")),
                row.names = FALSE)
    }
  }
  f <- formulas_of(cell, meta)
  entry <- cell
  entry$re_q <- I(STRUCTURES[[cell$structure]]$q)
  entry$factors <- I(meta$factors); entry$n_x <- meta$n_x
  entry$r_formula <- f$r; entry$jl_formula <- f$jl
  entry$reml <- if (cell$family == "gaussian") TRUE else NULL
  entry$weights <- if (cell$family == "binomial_agg") "size" else NULL
  entry$family <- c(gaussian = "gaussian", binomial_bin = "binomial",
                    binomial_agg = "binomial", poisson = "poisson",
                    negbin = "negbin", gamma = "gamma")[[cell$family]]
  manifest$cells[[length(manifest$cells) + 1L]] <- entry
}
write(toJSON(manifest, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null"),
      file.path(here_dir, "manifest.json"))
cat(sprintf("cells: %d (b1-flagged: %d)  csvs in %s\n",
            length(manifest$cells),
            sum(vapply(manifest$cells, function(c) isTRUE(c$b1), TRUE)), grid_dir))
