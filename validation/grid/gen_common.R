#!/usr/bin/env Rscript
# Shared cell-space definitions for the accuracy grid: the structure catalog,
# the family/link/form arms, the size / balance / regime axes, the truth rules
# and the data simulator. Sourced by gen_manifest.R (which writes manifest.json)
# and by gen.R (which writes the CSVs). Truth and data therefore come out of ONE
# set of functions -- truth_of and sim_cell both read re_sds / re_corr /
# eta0_of / betas_of, so the recorded parameter vector cannot drift from the
# drawn response.
#
# Nesting is realized with globally-unique inner labels + plain intercept terms
# ((1|g1)+(1|g2), g2 labels unique within g1) rather than the `/` operator --
# semantically identical, and it sidesteps engine-specific nesting syntax
# (the crate's formula frontend has no `/`).
suppressMessages(library(MASS))

# ---- structure catalog -------------------------------------------------------
# Each structure: RE terms as a list of q values (q = 1 intercept-only, q >= 2
# intercept + q-1 slopes on x1..x_{q-1}); `nested` = TRUE chains every later
# term inside the previous one (unique labels), an integer m chains terms 2..m
# and leaves the rest crossed. n_theta = sum q(q+1)/2. The ladder targets
# {1,2,3,5,8,12,18,27,40} are realized via slope/factor counts; neighbors fill
# the gaps so aggregate curves have support between targets.
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

# The four structure CLASSES the grid spans. A lone RE term is `scalar` when it
# is intercept-only and `vector` otherwise; with several terms, a chain inside
# another term makes it `nested`, a slope block anywhere makes it `vector`, and
# plain intercepts on independent groupings make it `crossed`. Nesting is tested
# first because nest2s (a slope block with a nested inner term) is a nested
# design, not a second vector-slope one.
st_class <- function(sname) {
  st <- STRUCTURES[[sname]]
  if (length(st$q) == 1L) return(if (st$q[1] == 1L) "scalar" else "vector")
  if (!identical(st$nested, FALSE)) return("nested")
  if (max(st$q) >= 2L) "vector" else "crossed"
}
STRUCTURE_CLASSES <- c("scalar", "nested", "crossed", "vector")

# Extra (crossed, non-nested) groupings get min(30, max(6, G %/% 2)) levels --
# the 8-level convention of the in-crate corpus scaled up so 30k-obs cells do
# not saturate.
extra_levels <- function(G) min(30L, max(6L, G %/% 2L))

# Per-cell seed: a pure function of the cell id, so adding or removing a cell
# never moves another cell's data. Multiplying each code point by its squared
# position makes the hash order-sensitive; the modulus is 2^31 - 1, R's
# set.seed range.
seed_of <- function(id) {
  cp <- utf8ToInt(id)
  as.integer(sum(as.numeric(cp) * seq_along(cp)^2) %% 2147483647)
}

# 11 family/link/form arms. Binomial appears in BOTH the Bernoulli and the
# trials>1 (weights: size) form for each of its three links; the two Gamma links
# and the NB log link are separate arms.
ARMS <- list(
  list(family = "gaussian",         link = "identity", form = NA, tag = "lmm"),
  list(family = "binomial",         link = "logit",    form = "bernoulli", tag = "binb"),
  list(family = "binomial",         link = "logit",    form = "trials",    tag = "bina"),
  list(family = "binomial",         link = "probit",   form = "bernoulli", tag = "prbb"),
  list(family = "binomial",         link = "probit",   form = "trials",    tag = "prba"),
  list(family = "binomial",         link = "cloglog",  form = "bernoulli", tag = "cllb"),
  list(family = "binomial",         link = "cloglog",  form = "trials",    tag = "clla"),
  list(family = "poisson",          link = "log",      form = NA, tag = "pois"),
  list(family = "gamma",            link = "log",      form = NA, tag = "gaml"),
  list(family = "gamma",            link = "inverse",  form = NA, tag = "gami"),
  list(family = "negativebinomial", link = "log",      form = NA, tag = "nb")
)

# The variant arms: ONE canonical link/form per family (the representative set
# is per family, five families, not per link).
CANONICAL_ARM <- list(gaussian = 1L, binomial = 2L, poisson = 8L, gamma = 9L,
                      negativebinomial = 11L)

# Representative structures per family, varied one axis at a time from the core.
# Each list covers at least one scalar (int1), one nested (nest2/nest2s/nest3),
# one crossed (cross4/cross6/int2x) and one vector-slope (q2s/q2sx2/q4sx2)
# structure. THIS IS THE TRIM KNOB: shortening these lists is how the full
# oracle wall is brought inside one night at six workers -- but the four
# structure classes and the WIDE code below are not tradeable against wall time.
#
# THE MEASUREMENT THAT SIZES THESE LISTS, taken 2026-09-23 on an UNLOCKED clock
# (no_turbo = 0), so it is a sizing estimate good to tens of percent and never a
# benchmark. Per-cell walls on a binomial scalar cell, whole invocation including
# process startup: 2.6 s at 60 rows, 2.7 s at 3000, 5.6 s at 30000 for lme4 and
# glmmTMB; 31-34 s for MixedModels, whose Julia load dominates and is paid once
# per worker rather than once per cell; 3.9 s per cell for GLMMadaptive, whose
# whole share is 8 AGQ cells. The wide corners are measured, not interpolated:
# cross6 at 30000x20 costs 65 s in lme4, and q8 at 30000 rows with per_group >=
# 20 does not finish inside run.sh's 600 s watchdog at all.
# The four oracles run one after another, each at --jobs 6, so a night's budget
# is their sum: 0.9 h on the size ladder alone, 2.1 h once every 30000-row cell
# is charged a width surcharge and the three q8 x 30000 cells the full 600 s,
# against a 10 h night. Nothing is trimmed; all five lists are at full length.
#
# THE WIDE CODE. The RAM watch has to happen on 30000-row WIDE cells, and a size
# variant only exists for a structure that is in this list, so a list with no
# q >= 4 code has no wide 30000-row cell at all and the watch measures nothing.
# gaussian carries q4sx2 (q1 = 4, n_theta = 12); binomial carries q8 (q1 = 8,
# n_theta = 36, the corpus's widest single block) for exactly this reason. The
# other three families do not need one -- the RAM ceiling is set by the widest
# design any worker holds, and lme4/glmmTMB on a gaussian or binomial
# 30000 x 100 cell is that worst case.
REPRESENTATIVE <- list(
  gaussian         = c("int1", "q2s", "nest2", "nest3",  "cross4", "cross6", "q4sx2"),
  binomial         = c("int1", "q2s", "nest2", "nest2s", "cross4", "cross6", "q8"),
  poisson          = c("int1", "q2s", "nest2", "nest2s", "cross4", "cross6", "q2sx2"),
  gamma            = c("int1", "q2s", "nest2", "nest2s", "cross4", "int2x",  "q2sx2"),
  negativebinomial = c("int1", "q2s", "nest2", "nest2s", "cross4", "int2x",  "q2sx2")
)
# Both asserted at load, because the trim is a hand edit:
stopifnot("a REPRESENTATIVE list lost its wide code" =
  all(vapply(REPRESENTATIVE[c("gaussian", "binomial")],
             function(v) any(vapply(STRUCTURES[v], function(st) st$q[1] >= 4L, TRUE)), TRUE)))
stopifnot("a REPRESENTATIVE list lost a structure class" =
  all(vapply(REPRESENTATIVE,
             function(v) all(STRUCTURE_CLASSES %in% vapply(v, st_class, "")), TRUE)))

# The core reference point: every family/link/form x every structure it can
# identify sits here, and every variant moves exactly one axis away from it.
CORE <- list(n_obs = 3000L, per_group = 20L, balance = "bal", regime = "base")

# n_obs x per_group. 60 is the small-n row. Combinations that cannot be
# identified are dropped by `feasible` below -- but three of the twelve nominal
# pairs can NEVER pass it for ANY structure, because they leave fewer than 6
# primary groups no matter what: (60,20) -> G = 3, (60,100) -> G = 0,
# (300,100) -> G = 3. Listing a pair that is dead by arithmetic would make the
# axis look wider than it is, so they are excluded here and the reason is this
# comment, not a silent drop inside `feasible`. Nine live pairs remain.
SIZES <- list(c(60L, 5L),
              c(300L, 5L), c(300L, 20L),
              c(3000L, 5L), c(3000L, 20L), c(3000L, 100L),
              c(30000L, 5L), c(30000L, 20L), c(30000L, 100L))

BALANCES <- c("skew", "single")            # variants; "bal" is the core
BASE_REGIMES <- c("nearzero", "lowsnr", "highcorr", "boundary")
EXTRA_REGIME <- list(gaussian = character(0), binomial = "rare",
                     poisson = "lowmu", gamma = character(0),
                     negativebinomial = "lowmu")

# ---- feasibility / the 60-row identifiability drop rule ----------------------
# Three conditions, all of them about IDENTIFIABILITY, not taste, and all three
# reachable -- a clause that cannot fail is not a condition:
#   1. The primary grouping must have enough levels to identify its q1 x q1
#      covariance block, and at least 6 levels.
#   2. Each observation must carry at most one covariance parameter per
#      ~10 rows: n_obs >= 10 * n_theta.
#   3. A nested chain must have room for its own levels: a nested term splits
#      each parent level into 3, so a chain of depth d needs G * 3^(d-1) levels
#      to be distinguishable at all, and each of them at least 2 observations --
#      n_obs >= 2 * G * 3^(d-1).
# NOTE what is deliberately NOT a condition: an extra crossed grouping's level
# count. extra_levels(G) = min(30, max(6, G %/% 2)) has a floor of 6 by
# construction, so "extras need >= 6 levels" can never fail and is not written.
#
# Condition 2 is what makes the 60-row row drop the wide structures: at
# n_obs = 60 only structures with n_theta <= 6 survive (int1, int2x, nest2, q2s,
# nest3, nestmix, cross4, nest2s, q2sx2, q3s, cross6, q2sq2s), and the
# q4/q5q2/q6/q8 family -- 10 to 40 covariance parameters -- never appears there.
# Fitting 40 parameters to 60 rows is not a hard cell, it is an unidentified
# one, and an oracle disagreement there would measure nothing.
feasible <- function(st, n_obs, per) {
  G  <- n_obs %/% per
  q1 <- st$q[1]
  nested_depth <- if (isTRUE(st$nested)) length(st$q)
                  else if (is.numeric(st$nested)) as.integer(st$nested) else 1L
  G >= max(q1 * (q1 + 1L) / 2L + 2L, 2L * q1 + 2L) &&
    G >= 6L &&
    n_obs >= 10L * n_theta_of(st) &&
    n_obs >= 2L * G * 3L^(nested_depth - 1L)
}

# ---- truth rules -------------------------------------------------------------
nx_of <- function(st) max(1L, max(st$q) - 1L)

# Baseline sd ladder per RE term (descending), then the per-regime overrides.
# `k` is the grouping index (1 = primary), `q` its width.
sd_ladder <- function(q, k) 0.9^(seq_len(q) - 1) * c(1.0, 0.8, 0.7, 0.6, 0.6, 0.5, 0.5, 0.5)[k]

re_sds <- function(cell, k, q) {
  sds <- sd_ladder(q, k)
  if (cell$family != "gaussian") sds <- sds * 0.6   # keep link-scale sane
  if (identical(cell$regime, "nearzero") && k == 1L) sds[q] <- 0.02
  # `boundary`: the true RE sd is EXACTLY zero on the primary grouping, so the
  # boundary of the parameter space is a class of its own rather than an
  # accident of `nearzero`.
  if (identical(cell$regime, "boundary") && k == 1L) sds <- rep(0, q)
  sds
}
re_corr <- function(cell) if (identical(cell$regime, "highcorr")) 0.9 else 0.2

# Intercept on the linear-predictor scale.
# gamma/inverse gets eta0 = 20: the inverse link needs eta > 0 for every row, and
# the fixed part plus the random effects reach about +/-11 in the widest cell
# (7 slopes and an 8-wide RE block at 5 sigma). 20 keeps every drawn eta clear of
# zero with margin. draw_response asserts min(eta) > 1e-3 and stops on a violation
# rather than handing on a design whose mu is negative. Same spirit as the
# corpus's own large-eta Gamma-inverse fixture (tools/prep/gen_scale_data.R).
eta0_of <- function(cell) {
  if (identical(cell$family, "gamma"))
    return(if (identical(cell$link, "inverse")) 20.0 else 0.4)
  if (identical(cell$family, "binomial"))
    return(if (identical(cell$regime, "rare")) stats::qlogis(0.02) else 0.2)
  if (cell$family %in% c("poisson", "negativebinomial"))
    return(if (identical(cell$regime, "lowmu")) log(0.5) else 0.4)
  0.5   # gaussian
}

betas_of <- function(cell, nx) {
  b <- rep(c(0.8, -0.5, 0.3, -0.2, 0.4, -0.3, 0.2), length.out = nx)
  if (identical(cell$regime, "lowsnr")) b <- b * 0.25
  # gamma/inverse rides on eta0 = 20; halving the slopes keeps eta positive.
  if (identical(cell$family, "gamma") && identical(cell$link, "inverse")) b <- b * 0.5
  b
}

RESID_SD <- function(cell) if (identical(cell$regime, "lowsnr")) 3.0 else 0.6
GAMMA_SHAPE <- 2       # y ~ rgamma(shape = 2, scale = mu / 2): E[y] = mu
NB_THETA    <- 1.5     # the sim_nb convention (prep/export_data.R)

group_names <- function(st) c("g1", if (length(st$q) > 1L) paste0("g", 2:length(st$q)))
term_names  <- function(q) c("(Intercept)", if (q > 1L) paste0("x", seq_len(q - 1L)))

truth_of <- function(cell) {
  if (identical(cell$structure, "glm")) {
    nx <- cell$n_x
    return(list(
      coef_names = I(c("(Intercept)", paste0("x", seq_len(nx)))),
      beta = I(c(eta0_of(cell), betas_of(cell, nx))),
      varcomp = list(),
      sigma = if (cell$family == "gaussian") RESID_SD(cell) else NULL,
      dispersion = if (cell$family == "gamma") 1 / GAMMA_SHAPE else NULL,
      nb_theta = if (cell$family == "negativebinomial") NB_THETA else NULL))
  }
  st <- STRUCTURES[[cell$structure]]
  nx <- nx_of(st)
  rho <- re_corr(cell)
  vc <- lapply(seq_along(st$q), function(k) {
    q <- st$q[k]
    sds <- re_sds(cell, k, q)
    corr <- matrix(rho, q, q); diag(corr) <- 1
    list(group = group_names(st)[k], terms = I(term_names(q)),
         stddev = I(sds), corr = corr)
  })
  list(
    coef_names = I(c("(Intercept)", paste0("x", seq_len(nx)))),
    beta = I(c(eta0_of(cell), betas_of(cell, nx))),
    varcomp = vc,
    sigma = if (cell$family == "gaussian") RESID_SD(cell) else NULL,
    dispersion = if (cell$family == "gamma") 1 / GAMMA_SHAPE else NULL,
    nb_theta = if (cell$family == "negativebinomial") NB_THETA else NULL)
}

# ---- simulation --------------------------------------------------------------
# The offset column, the prior-weight column and the link-aware response draw,
# shared by the GLM branch and the mixed branch of sim_cell so the two cannot
# drift apart. `eta` arrives carrying the fixed part and (mixed branch) the
# random effects; the offset is folded in here, BEFORE the response, so the
# recorded beta stays the truth.
draw_response <- function(cell, df, eta, n, factors, n_x) {
  if (!is.null(cell$offset_col)) {
    df[[cell$offset_col]] <- log(runif(n, 0.5, 2))
    eta <- eta + df[[cell$offset_col]]
  }
  # Prior (precision) weights: one column of small integer precision weights. The
  # response draw below has to match the likelihood every engine fits with
  # `weights = w`, which is Var[y_i] = sigma^2 / w_i -- so a gaussian residual
  # gets sd RESID_SD / sqrt(w) and a Gamma draw gets shape GAMMA_SHAPE * w,
  # which scales its dispersion 1/shape by 1/w. Drawing homoskedastic errors
  # under a weighted fit would make the recorded truth contradict the
  # likelihood. Same convention as the crate's weights corpus,
  # validation/tools/prep/gen_weights_data.R. A Poisson prior weight does not change
  # the draw, and the recorded truth stays the UNWEIGHTED sigma / dispersion --
  # that is the quantity the weighted fit estimates.
  w <- rep(1, n)
  if (!is.null(cell$weights_col)) {
    df[[cell$weights_col]] <- sample(1:5, n, replace = TRUE)
    w <- df[[cell$weights_col]]
  }

  linkinv <- switch(cell$link,
    identity = function(e) e,
    logit    = stats::plogis,
    probit   = stats::pnorm,
    cloglog  = function(e) -expm1(-exp(e)),
    log      = exp,
    inverse  = function(e) 1 / e,
    stop("no linkinv for link ", cell$link))
  if (identical(cell$link, "inverse")) {
    stopifnot("gamma/inverse cell drew a non-positive eta" = min(eta) > 1e-3)
  }
  mu <- linkinv(eta)
  # The response COLUMN NAME is the manifest's `response` field, and every
  # engine and every deviance closed form reads it from there. On a generated
  # cell it is "y", or "incidence" on a trials cell.
  if (cell$family == "gaussian") {
    df$y <- eta + rnorm(n, sd = RESID_SD(cell) / sqrt(w))
  } else if (cell$family == "binomial" && identical(cell$form, "trials")) {
    df$size <- sample(5:20, n, replace = TRUE)
    df$incidence <- rbinom(n, df$size, mu)
  } else if (cell$family == "binomial") {
    df$y <- rbinom(n, 1, mu)
  } else if (cell$family == "poisson") {
    df$y <- rpois(n, mu)
  } else if (cell$family == "negativebinomial") {
    df$y <- MASS::rnegbin(n, mu = mu, theta = NB_THETA)
  } else if (cell$family == "gamma") {
    df$y <- rgamma(n, shape = GAMMA_SHAPE * w, scale = mu / (GAMMA_SHAPE * w))
  } else {
    stop("no response draw for family ", cell$family)
  }
  list(df = df, factors = factors, n_x = n_x)
}

sim_cell <- function(cell) {
  # GLM (no random effect) cells: n_obs rows, n_x numeric predictors, no
  # grouping factor and therefore no per_group/balance/regime. Taken FIRST,
  # because everything below this point reads cell$per_group. truth_of has the
  # matching branch; the two must agree, which is why both read
  # eta0_of/betas_of.
  if (identical(cell$structure, "glm")) {
    set.seed(seed_of(cell$cell))
    n  <- cell$n_obs
    nx <- cell$n_x
    X <- matrix(rnorm(n * nx), n, nx,
                dimnames = list(NULL, paste0("x", seq_len(nx))))
    df <- data.frame(X)
    eta <- eta0_of(cell)
    b <- betas_of(cell, nx)
    for (j in seq_len(nx)) eta <- eta + b[j] * X[, j]
    return(draw_response(cell, df, eta, n, factors = character(0), n_x = nx))
  }

  set.seed(seed_of(cell$cell))
  st <- STRUCTURES[[cell$structure]]
  n <- cell$n_obs; per <- cell$per_group; G <- n %/% per
  nx <- nx_of(st)   # covariates: enough for the widest slope block
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

  corr_val <- re_corr(cell)
  eta <- eta0_of(cell)
  betas <- betas_of(cell, nx)
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
    nm <- group_names(st)[k]
    df[[nm]] <- f
    fac_names <- c(fac_names, nm)
    nl <- nlevels(f)
    sds <- re_sds(cell, k, q)
    if (any(sds > 0)) {
      Sigma <- diag(sds, q) %*% (matrix(corr_val, q, q) + diag(1 - corr_val, q)) %*% diag(sds, q)
      b <- matrix(mvrnorm(nl, rep(0, q), Sigma), nl, q)
    } else {
      # `boundary` zeroes the block; mvrnorm on a singular zero covariance is
      # not guaranteed to return exact zeros, and exact zeros are the point.
      b <- matrix(0, nl, q)
    }
    eta <- eta + b[as.integer(f), 1]
    if (q >= 2) for (d in 2:q) eta <- eta + b[as.integer(f), d] * X[, d - 1]
  }

  draw_response(cell, df, eta, n, factors = fac_names, n_x = nx)
}

# ---- formula emission ---------------------------------------------------------
# Three dialects. `glmm_formula` differs from `jl_formula` in exactly one way:
# it never carries an offset(...) term, because glmm and the two ports take the
# offset through the fit options while lme4 / glmmTMB / GLMMadaptive /
# MixedModels.jl read it out of the formula. Building it here rather than
# stripping the term downstream is what makes "applied exactly once" mechanical.
formulas_of <- function(cell) {
  st <- if (identical(cell$structure, "glm")) NULL else STRUCTURES[[cell$structure]]
  re <- if (is.null(st)) character(0) else vapply(seq_along(st$q), function(k) {
    q <- st$q[k]; nm <- group_names(st)[k]
    if (q == 1) sprintf("(1 | %s)", nm)
    else sprintf("(1 + %s | %s)", paste(paste0("x", seq_len(q - 1)), collapse = " + "), nm)
  }, "")
  nx <- cell$n_x
  fx <- paste(paste0("x", seq_len(nx)), collapse = " + ")
  off <- if (!is.null(cell$offset_col)) sprintf("offset(%s)", cell$offset_col) else NULL
  # Two right-hand sides: WITH the offset term and WITHOUT.
  rhs_off <- paste(c("1", fx, off, re), collapse = " + ")
  rhs_no  <- paste(c("1", fx, re), collapse = " + ")
  trials <- identical(cell$form, "trials")
  r_resp <- if (trials)
    sprintf("cbind(%s, %s - %s)", cell$response, cell[["weights"]], cell$response)
    else cell$response
  jl_resp <- if (trials) "prop" else cell$response
  list(
    r    = sprintf("%s ~ %s", r_resp, rhs_off),
    jl   = sprintf("@formula(%s ~ %s)", jl_resp, rhs_off),
    # The crate's parser treats the intercept as always implicit and has no term
    # for a literal `1`, so the "1 + " is stripped here rather than in four
    # engine scripts. `&` never appears -- formulas_of does not emit it.
    glmm = strip_glmm_intercept(sprintf("%s ~ %s", jl_resp, rhs_no)))
}

# The crate's formula dialect, from a Julia-style one: the "@formula(...)"
# wrapper unwrapped, the explicit leading intercept term dropped, and Julia's
# crossed-grouping `&` written as the parser's `:`. Same three rewrites the
# in-crate engine harness applies to a manifest jl_formula.
glmm_formula_of <- function(jl) {
  inner <- sub("^@formula\\(", "", sub("\\)$", "", jl))
  strip_glmm_intercept(gsub(" & ", ":", inner, fixed = TRUE))
}
strip_glmm_intercept <- function(f) sub(" ~ 1 + ", " ~ ", f, fixed = TRUE)

# `tag` is an ARMS attribute (`lmm`, `binb`, `bina`, `prbb`, `prba`, `cllb`,
# `clla`, `pois`, `gaml`, `gami`, `nb`) and is passed IN rather than read off
# the cell: it is not a manifest field, so a cell object round-tripped through
# manifest.json would not carry it and this function would silently produce
# "NA_int1_...". Only gen_manifest.R calls it, and only while it still has the
# arm. Cells that read a committed CSV keep the fixture's own name instead and
# never come through here.
cell_id_of <- function(cell, tag) {
  suffix <- paste0(
    if (!is.null(cell$nagq))        sprintf("_agq%d", cell$nagq) else "",
    if (!is.null(cell$offset_col))  "_off" else "",
    if (!is.null(cell$weights_col)) "_wts" else "")
  if (identical(cell$structure, "glm"))
    return(sprintf("%s_glm_g%d%s", tag, cell$n_obs, suffix))
  sprintf("%s_%s_g%dp%d_%s_%s%s", tag, cell$structure, cell$n_obs,
          cell$per_group, cell$balance, cell$regime, suffix)
}
