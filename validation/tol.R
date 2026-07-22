TOL <- list(
  beta_rel        = 1e-3,   # fixed effects: relative
  stddev_rel      = 1e-3,   # varcomp std-devs: relative
  loglik_abs_lmm  = 2e-6,   # LMM REML criterion: near-exact across engines (~1e-9 typical).
                            #   Measured worst 1.23e-6 on sim_max_q_slope (q=8, 36 theta params,
                            #   the corpus's largest LMM covariance; ~6e-10 relative — MixedModels
                            #   sits 5e-7 from lme4 on the same rung), so 2e-6 = measured worst +
                            #   margin, same convention as se_hessian_rel below.
  loglik_abs_glmm = 1e-3,   # GLMM Laplace logLik: two optimizers land ~3e-6 relative
                            #   apart on the same surface (beta/varcomp confirm same fit)
  se_rel          = 1e-3,   # LMM SE + method-matched GLMM RX: tight (same method, all engines)
  se_hessian_rel  = 1e-3,   # GLMM Hessian pair (lme4 vs glmm), same band as se_rel. History:
                            #   this sat at 3e-2 while the frozen oracle carried lme4's lagged-
                            #   ldL2 tolPwrss artifact (~1.3%: glmer's Xwts run one PIRLS
                            #   iteration behind the mode; docs/GLMM/2026-07-04-glmm-hessian-
                            #   curvature-diagnosis.md, Resolution). The 2026-07-04 references
                            #   are artifact-free (lme4.R tolPwrss=1e-13, recorded per JSON) and
                            #   glmm's FD runs PIRLS at its tight FD-only tol, so the engines
                            #   agree to worst 2e-5 (grouseticks); 1e-3 = measured-worst + the
                            #   same ~margin se_rel carries over ITS measured worst.
  stddev_se_rel   = 3e-3,   # GLMM RE-stddev SE (lme4 numDeriv vs glmm single-step FD, both on
                            #   the joint (theta,beta) Hessian theta block). Same artifact
                            #   history as se_hessian_rel (was 3e-2). Against the artifact-free
                            #   oracle the worst gap is 8e-4 (sim_sparse_poisson) -- the
                            #   single-step-FD vs numDeriv-Richardson method floor on the theta
                            #   block, noisier than the beta block, hence the wider band.

  # glmm (Rust) vs the glmm Python port -- a ROUND-OFF band, not an agreement band.
  # The port drives the same kernel through PyO3 (fit_warm(start=NULL) IS fit_cold),
  # with the same lowering and a deterministic optimizer, so every gated quantity is
  # bit-identical and only the JSON round-trip could perturb it (it does not: Rust
  # and Python both emit shortest-round-trip f64, jsonlite parses back exact). Any
  # nonzero value here means the port fed the kernel something different -- a wiring
  # bug -- so this is the one band that is diagnostic at 0 and is NEVER widened.
  # Measured worst across the 26-rung corpus at freeze (2026-07-16): exactly 0.
  port_rel        = 1e-12,

  # Vector-RE AGQ rungs vs GLMMadaptive (goldens/sim_*_slope*_agq_k*.json; the
  # in-crate gates in src/fit/glmm_tests.rs use these numbers -- change
  # together). Calibrated empirically at freeze (2026-07-13) against goldens
  # frozen with TIGHTENED mixed_model controls (see goldens_agq.R -- the
  # GLMMadaptive DEFAULTS under-converge by ~4e-3 logLik on the low-info rungs,
  # the same artifact class as lme4's default tolPwrss). Measured matched-k
  # worsts: k=11 agrees to <= 7.1e-4 on EVERY quantity; every k=7 worst below
  # comes from the sparsest rung (sim_poisson_slope1, 4 obs/cluster) where each
  # engine still carries its own quadrature truncation error -- GLMMadaptive's
  # own k=7 se[0] sits 1.2e-2 from its own k=25 limit while glmm's k=7 is
  # within 1.1e-3 of that limit, so the matched-k=7 gap is oracle-side
  # k-truncation, not implementation disagreement. Bands = measured worst +
  # the usual ~2x margin.
  agq_beta_rel       = 3e-3,  # worst 1.4e-3 (poisson k=7); binomial rungs <= 6.7e-5
  agq_stddev_rel     = 4e-3,  # worst 1.7e-3 (poisson k=7)
  agq_corr_abs       = 4e-3,  # ABSOLUTE (correlations near 0 break relative); worst 1.6e-3
  agq_se_hessian_rel = 2e-2   # worst 1.3e-2 (poisson k=7 intercept, the MA k-truncation
                              #   case above); next-worst 2.6e-3, k=11 worst 7.1e-4
)

# Max relative difference over two aligned numeric vectors; NA on length mismatch
# so it shows up as a hard failure rather than a silently-recycled false pass.
rel_max <- function(x, y) {
  if (length(x) != length(y)) return(NA_real_)
  max(abs(x - y) / pmax(abs(x), abs(y), 1e-12))
}

# Shared join helpers (analyze_diligent.R + verify_boundary37.R -- change together).

# Torn-line tolerant (kill -9 watchdog can truncate the final line) -- same
# discipline as analyze_grid.R's reader.
read_jsonl <- function(path) {
  lines <- readLines(path); lines <- lines[nzchar(lines)]
  recs <- list()
  for (ln in lines) {
    rec <- tryCatch(fromJSON(ln, simplifyVector = TRUE), error = function(e) NULL)
    if (!is.null(rec)) recs[[length(recs) + 1L]] <- rec
  }
  setNames(recs, vapply(recs, `[[`, "", "case_id"))
}

# varcomp -> stddev vector and off-diagonal correlations, flattened across
# grouping factors in NAME order (deterministic join key; glmm records g1,g2,...,
# oracle records the same factor names). Positional alignment within a group is
# the term order both engines emit ((Intercept), slope, ...).
stddevs_of <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(numeric(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  vc <- vc[order(vapply(vc, function(g) g$group, ""))]
  unlist(lapply(vc, function(g) as.numeric(unlist(g$stddev))))
}
# Off-diagonal correlations (upper triangle) flattened across groups, name-order.
# Scalar groups ([[1]]) contribute nothing.
corrs_of <- function(rec) {
  vc <- rec$varcomp
  if (is.null(vc) || length(vc) == 0) return(numeric(0))
  if (is.data.frame(vc)) vc <- lapply(seq_len(nrow(vc)), function(i) as.list(vc[i, ]))
  vc <- vc[order(vapply(vc, function(g) g$group, ""))]
  unlist(lapply(vc, function(g) {
    m <- g$corr
    if (is.null(m)) return(numeric(0))
    # After the data.frame round-trip above, corr is a length-1 LIST wrapping
    # the k x k matrix; as.matrix on that gives a 1x1 list-matrix and the
    # nrow<2 guard silently dropped every correlation (dead corr gate, caught
    # in review 2026-07-14). Unwrap first.
    if (is.list(m)) m <- m[[1]]
    m <- as.matrix(m); if (nrow(m) < 2) return(numeric(0))
    m[upper.tri(m)]
  }))
}
