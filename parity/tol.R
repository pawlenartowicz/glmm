TOL <- list(
  beta_rel        = 1e-3,   # fixed effects: relative
  stddev_rel      = 1e-3,   # varcomp std-devs: relative
  loglik_abs_lmm  = 1e-6,   # LMM REML criterion: near-exact across engines (~1e-9 seen)
  loglik_abs_glmm = 1e-3,   # GLMM Laplace logLik: two optimizers land ~3e-6 relative
                            #   apart on the same surface (beta/varcomp confirm same fit)
  se_rel          = 1e-3,   # LMM SE + method-matched GLMM RX: tight (same method, all engines)
  se_hessian_rel  = 1e-3,   # GLMM Hessian pair (lme4 vs glmm), same band as se_rel. History:
                            #   this sat at 3e-2 while the frozen oracle carried lme4's lagged-
                            #   ldL2 tolPwrss artifact (~1.3%: glmer's Xwts run one PIRLS
                            #   iteration behind the mode; docs/GLMM/2026-07-04-glmm-hessian-
                            #   curvature-diagnosis.md, Resolution). The 2026-07-04 references
                            #   are artifact-free (fit.R tolPwrss=1e-13, recorded per JSON) and
                            #   glmm's FD runs PIRLS at its tight FD-only tol, so the engines
                            #   agree to worst 2e-5 (grouseticks); 1e-3 = measured-worst + the
                            #   same ~margin se_rel carries over ITS measured worst.
  stddev_se_rel   = 3e-3    # GLMM RE-stddev SE (lme4 numDeriv vs glmm single-step FD, both on
                            #   the joint (theta,beta) Hessian theta block). Same artifact
                            #   history as se_hessian_rel (was 3e-2). Against the artifact-free
                            #   oracle the worst gap is 8e-4 (sim_sparse_poisson) -- the
                            #   single-step-FD vs numDeriv-Richardson method floor on the theta
                            #   block, noisier than the beta block, hence the wider band.
)

# Max relative difference over two aligned numeric vectors; NA on length mismatch
# so it shows up as a hard failure rather than a silently-recycled false pass.
rel_max <- function(x, y) {
  if (length(x) != length(y)) return(NA_real_)
  max(abs(x - y) / pmax(abs(x), abs(y), 1e-12))
}
