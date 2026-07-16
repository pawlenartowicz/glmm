use bobyqa::{Bobyqa, Config};
use faer::dyn_stack::MemBuffer;
use faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch;
use faer::sparse::linalg::cholesky::{
    factorize_symbolic_cholesky, CholeskySymbolicParams, SymbolicCholesky,
};
use faer::sparse::linalg::SupernodalThreshold;
use faer::sparse::{SparseColMat, Triplet};
use faer::{Mat, MatRef, Par, Side, Spec};

use crate::lmm::{LmmGroupings, GLMM_RHO_END, RHO_BEGIN, THETA_TRUTH_FLOOR};

use super::BETA_BOX;

/// All GLMM solver scratch — allocated once per (spec, max_n) shape.
pub struct GlmmWorkspace {
    /// reused RE structure (estimator-agnostic)
    pub groupings: LmmGroupings,
    /// Outcome family/link selecting the PIRLS IRLS math. `Binomial{Logit}` runs
    /// the verbatim fused-SIMD path; other families take the general branch.
    pub family: crate::Family,
    /// NB dispersion θ̂ fixed for this fit by the marginal-θ outer loop — read by
    /// the PIRLS/AGQ variance/deviance only when `family` is `NegativeBinomial`.
    /// Defaulted to `f64::NAN` at construction; the `fit::fit_glmm` adapter sets it
    /// per fit (NB passes the current θ iterate, every other family leaves it NaN).
    pub nb_theta: f64,
    /// adaptive GH node count; 1 = Laplace. >1 only fires on the single-grouping-factor
    /// binomial/Poisson AGQ paths — scalar intercept (`agq::agq_deviance`) or vector RE
    /// with `q_p ∈ 2..=3` (`agq::agq_deviance_vec`); ignored otherwise.
    pub nagq: u8,
    /// AGQ per-cluster scratch; unused on the Laplace path. Shape-dependent:
    /// scalar (`q_p==1`) is `4·n_primary` (center loglik | node u_cj | per-node
    /// loglik | running log-sum); vector (`q_p∈2..=3`) is `2·n_primary + k^q·(q+1)`
    /// (center loglik | running log-sum | product-grid node table). See the
    /// sizing at construction.
    pub agq_scratch: Vec<f64>,
    /// FitOptions::parallel_inner, copied per fit by the fit.rs adapter (the
    /// nb_theta pattern). Runtime gate for the parallel kernels in `parallel`
    /// builds: when false — or in any serial build — the per-fit
    /// ClusterRowIndex is never built, agq_deviance runs the original
    /// node-outer loop, and the FD-Hessian grids stay serial; a batch caller
    /// pays nothing.
    pub parallel_inner: bool,
    /// Per-cluster row CSR for the cluster-outer AGQ loop — rayon's
    /// work-splitting substrate. Built once per fit in fit_glmm (cluster_ids
    /// is fit-fixed) iff the `parallel` feature is compiled in AND
    /// `nagq > 1 && parallel_inner`; None otherwise. Serial builds are always
    /// node-outer: cluster-outer serially regresses many-tiny-cluster shapes
    /// (see the build site in fit_glmm).
    pub(crate) cluster_rows: Option<super::agq::ClusterRowIndex>,
    /// total RE columns (groupings.k_total)
    pub k: usize,
    /// fixed-effect predictors
    pub p: usize,
    /// count of variance-component (θ) parameters (groupings.n_theta())
    pub n_theta: usize,
    /// max_n × k dense RE design (built per (spec, N) by `build_z`)
    pub z: Mat<f64>,
    /// max_n × k = ZΛ (rebuilt per BOBYQA eval)
    pub m: Mat<f64>,
    /// Joint (θ,β) BOBYQA solver, dimension `n_theta + p`.
    pub solver: Bobyqa, // sized n_theta + p
    /// Joint solver's live iterate: `[θ (n_theta) | β (p)]`.
    pub params: Vec<f64>, // [θ | β]
    /// Joint solver box lower bounds, length `n_theta + p`.
    pub lower: Vec<f64>,
    /// Joint solver box upper bounds, length `n_theta + p`.
    pub upper: Vec<f64>,
    /// θ-only BOBYQA solver for the two-stage optimizer's stage 1: sized
    /// `n_theta`, configured with the same `rho_begin`/`GLMM_RHO_END` schedule
    /// as `solver` and the `sparse_lmm_seed` mid-model `npt` rule
    /// (`ceil(1.5·n_theta) + 1` at `n_theta ≥ 3`, else `2·n_theta + 1`) — not
    /// the joint solver's `npt`, which differs. See `fit_glmm`.
    pub solver_stage1: Bobyqa,
    /// θ-only candidate/incumbent buffer for stage 1, length `n_theta`; seeded
    /// from `params`'s θ prefix at construction.
    pub params_stage1: Vec<f64>,
    /// Two-stage optimizer flag, `true` by default: a θ-only stage-1 BOBYQA
    /// warm-starts the joint (θ,β) stage-2 solve, cutting the outer evaluation
    /// count (~2× on grouseticks). Set to `false` for the single joint (θ,β)
    /// solve — retained as an A/B escape hatch (the `force_dense_schur`
    /// precedent), not a supported alternative; both paths converge to the same
    /// optimum within oracle tolerances.
    pub two_stage: bool,
    // PIRLS scratch (sized max_n / k):
    /// PIRLS linear predictor η, length max_n.
    pub eta: Vec<f64>,
    /// PIRLS fitted mean μ, length max_n.
    pub prob: Vec<f64>,
    /// PIRLS working weights W, length max_n.
    pub w: Vec<f64>,
    /// Σ_j x·β, hoisted out of the PIRLS iteration (β fixed within a solve)
    pub eta_fixed: Vec<f64>,
    /// n×q_p row-major mᵢ = Λ_p'·zᵢ (blocked path) — filled once per PIRLS solve
    pub m_buf: Vec<f64>,
    /// n×(q_p−1) row-major f64 copy of x[:, slope_cols] — filled once per fit
    pub z_buf: Vec<f64>,
    /// (Mu)ᵢ per iteration via GEMV; overwritten in place by the IRLS residual W·Mu + (y−p) before the RHS GEMV
    pub mu: Vec<f64>,
    /// Per-row prior weights `wᵢ` (`FitOptions::weights`; all-1 when absent —
    /// zero behavioral change). Semantics mirror the sparse twin
    /// (`SparseGlmmWorkspace::prior_w`): `wᵢ·W̃ᵢ` on the working weight,
    /// `wᵢ·devᵢ` on the deviance, `wᵢ·ρᵢ` on the score; everything downstream
    /// (A/RHS scatter, β border, Schur, FD Hessian) reads `w`/ρ and inherits it.
    pub(crate) prior_w: Vec<f64>,
    /// True iff `prior_w` was filled from `FitOptions::weights`. Gates the
    /// fused-SIMD logit fast path in `pirls.rs` (the fused kernel cannot take
    /// per-row weights; weighted logit runs the general scalar branch).
    pub(crate) weighted: bool,
    /// Current RE-mode iterate û, length k.
    pub u: Vec<f64>,
    /// previous accepted PIRLS iterate, step-halving backtrack buffer (len k.max(1))
    pub u_prev: Vec<f64>,
    /// within-fit û warm-start incumbent; RESET to 0 each fit_glmm — never carried across fits
    pub u_seed: Vec<f64>,
    /// k × k  M'WM + I. `dense_schur_fill` (se.rs) re-factors THIS field after a
    /// converged Fixed-mode PIRLS solve, so `pirls_solve` must leave it holding
    /// the raw symmetric A — never the in-place Cholesky factor.
    pub a: Mat<f64>,
    /// Copy-then-factor target for `a`'s Cholesky (k×k): `pirls_solve` copies
    /// `a`'s lower triangle in here (mirroring `.llt(Side::Lower)`'s internal
    /// copy) and runs `cholesky_in_place` on THIS buffer, leaving `a` itself
    /// untouched for `dense_schur_fill` to re-read.
    pub a_chol: Mat<f64>,
    /// Scratch for `a_chol`'s in-place `cholesky_in_place` (k×k, θ-independent
    /// size) — avoids the per-PIRLS-iteration `.llt(Side::Lower)` allocation on
    /// the dense `pirls_solve` hot path.
    pub a_llt_mem: MemBuffer,
    /// max_n × k = W∘M scratch for the dense-Gram GEMM (rebuilt per PIRLS iteration)
    pub wm: Mat<f64>,
    /// max_n × p = W∘X scratch for the X'WX GEMM (rebuilt per PIRLS iteration,
    /// all three pirls variants and the three se.rs schur-fill twins)
    pub wx: Mat<f64>,
    /// length k
    pub a_rhs: Vec<f64>,
    /// s · q_p² packed per-cluster q_p×q_p blocks (no-extras path; Σ wᵢmᵢmᵢ'+I then Crout L)
    pub a_blocks: Vec<f64>,
    // structured (block-diagonal core + Schur) path scratch — see
    // `pirls_solve_blocked_extras`. Sized to the worst-case grid shape; left
    // FACTORED (core L's + Schur L) after a converged structured PIRLS so
    // `structured_schur_fill` reuses them. `q_core = primary_q + nested_per_parent`,
    // `e = k_crossed`, `s = n_primary`.
    /// s · q_core² packed per-cluster core blocks (D_f+I then Crout L)
    pub core_blocks: Vec<f64>,
    /// s · q_core · e core↔crossed coupling C_f (row-major per cluster)
    pub coupling: Vec<f64>,
    /// e × e Schur S = (E+I) − Σ_f C_f'A_f⁻¹C_f (row-major; Crout L in place)
    pub schur_blk: Vec<f64>,
    /// q_p × q_p primary Λ_p scratch (row-major)
    pub lam: Vec<f64>,
    // Packed M = ZΛ nonzeros for the STRUCTURED path — filled once per deviance
    // eval by `build_packed_m` (replaces `apply_lambda` there), then read by the
    // structured PIRLS passes and `structured_schur_fill` so they never touch the
    // dense faer `m`. `q_core = primary_q + nested_per_parent`, `G_cap =
    // MAX_EXTRA_GROUPINGS`. Sized once at construction — no per-solve alloc.
    /// max_n · q_core row-major; [i·q_core+local] = M[(i, core_col(f,local))]
    pub m_core_buf: Vec<f64>,
    /// max_n · G_cap row-major; nonzero M value (z·θ) per crossed grouping
    pub cross_val: Vec<f64>,
    /// max_n · G_cap row-major; its crossed-block-local index b (0..e)
    pub cross_col: Vec<u32>,
    /// max_n; #crossed nonzeros for row i (≤ G ≤ MAX_EXTRA_GROUPINGS < 256)
    pub n_cross: Vec<u8>,
    // Per-cluster CSR of C_f's nonzero crossed columns (cluster f's slice is
    // coup_cols[coup_ptr[f]..coup_ptr[f+1]], sorted + deduped). Rebuilt on
    // pinning-mask transitions (see `coup_mask`) from cross_col/n_cross;
    // structured_factor's Schur build walks it instead of all e columns.
    /// ≤ max_n · G_cap entries before dedup
    pub coup_cols: Vec<u32>,
    /// n_primary + 1 offsets
    pub coup_ptr: Vec<u32>,
    /// θ-pinning mask (bit g = crossed grouping g has θ == 0.0) the current
    /// coup_cols/coup_ptr CSR was built for; `None` ⇒ not built this fit. The
    /// CSR pattern depends on the design AND this mask (build_packed_m drops
    /// pinned groupings), so the structured deviance rebuilds only on mask
    /// transitions. Reset to None at each fit_glmm start (mirrors u_seed).
    pub coup_mask: Option<u32>,
    /// Cached sparse factor of the crossed Schur `S`. `Some` only on the
    /// structured crossed path with `e > 0`; built per fit by `StructuredSchur::new`
    /// after `build_z`. `None` ⇒ the dense/blocked/e=0 paths, which never touch it.
    pub(crate) structured_schur: Option<StructuredSchur>,
    /// Test-only: force the dense `glmm_block_chol` Schur factor instead of the
    /// cached sparse one, so the both-paths cross-check runs both at one θ. Always
    /// `false` in production (the sparse factor is the only path).
    pub(crate) force_dense_schur: bool,
    // inference scratch:
    /// p × p
    pub xtwx: Mat<f64>,
    /// p × k
    pub xtwm: Mat<f64>,
    /// k × p  = A⁻¹ M'WX
    pub ainv_mtwx: Mat<f64>,
    /// p × p  X'WX − X'WM A⁻¹ M'WX
    pub schur: Mat<f64>,
    /// Scratch for `schur`'s in-place `cholesky_in_place` (p×p) — avoids the
    /// per-PIRLS-iteration `.llt(Side::Lower)` allocation on the `BetaStep::Profile`
    /// β-Schur border step (dense/blocked/structured PIRLS variants alike).
    pub schur_llt_mem: MemBuffer,
    /// length p (copied from params[n_theta..])
    pub betas: Vec<f64>,
    // β-profiling (`BetaStep`) scratch — see `pirls::BetaStep`. All length p.
    /// len p: Profile-mode δβ RHS/solution scratch; also the Fixed-mode β-input transient (deviance.rs copies params[n_theta..] here — NOT `betas`, which is the reported output)
    pub beta_rhs: Vec<f64>,
    /// len p: Profile-mode β backtrack buffer (last-accepted β; the halving twin of `u_prev`). Untouched in Fixed mode.
    pub beta_prev: Vec<f64>,
    /// len p: stage-1 profiled-β in/out buffer
    pub beta_prof: Vec<f64>,
    /// len p: stage-1 incumbent β snapshot (mirrors u_seed)
    pub beta_seed: Vec<f64>,
    /// length p
    pub var_diag: Vec<f64>,
    /// p×p Cov(β̂) — `var_diag` is its diagonal, and both are filled together at
    /// the same target indices (NaN elsewhere). Workspace-owned, not returned on
    /// `GlmmFit`, so filling it costs no per-fit allocation and the `Rx` warm
    /// path keeps its zero-alloc gate. Sourced from the full matrix each SE arm
    /// already forms: `Rx` from the Schur forward-solve columns, `Hessian` from
    /// `fd_hessian_cov`'s β block. Mapped to `Fit::vcov` by `fit/glmm.rs`.
    pub vcov: Mat<f64>,
    /// p×p scratch holding column `j` of `L⁻¹` at each target `j` — the `Rx`
    /// arm's per-target forward solves, kept so their pairwise dots can fill
    /// `vcov`'s off-diagonals instead of only `‖·‖²` on its diagonal.
    pub vcov_cols: Mat<f64>,
    /// length p
    pub t_sq: Vec<f64>,
    // SE of each θ coordinate = sqrt of the θ-block diagonal of the joint (θ,β)
    // FD-Hessian covariance (length n_theta). Filled ONLY on the `WaldSe::Hessian`
    // GLMM path from the θ block `fd_hessian_cov` already inverts (it otherwise
    // discards it); NaN under `WaldSe::Rx`, on the Hessian RX fallback, and on a
    // non-converged fit. For a SCALAR grouping (q=1, dispersion≡1) the RE stddev
    // equals its θ, so this is that stddev's SE directly (identity delta map); the
    // only reachable GLMM groupings are scalar (intercept-only).
    /// length n_theta
    pub theta_se: Vec<f64>,
    /// length p; Var(β̂)_jj forward-solve scratch (per-target)
    pub fwd_solve: Vec<f64>,
    // joint Wald scratch (reuse lme::joint_wald_chi_sq):
    /// Inverse of the joint Wald K matrix, p×p (see `lme::joint_wald_chi_sq`).
    pub joint_k_inv: Mat<f64>,
    /// Cholesky factor of the joint Wald Σ_t, p×p.
    pub joint_sigma_t_chol: Mat<f64>,
    /// Joint Wald right-hand side, length p.
    pub joint_rhs: Vec<f64>,
    // FD-Hessian SE scratch (`fd_hessian_cov`), allocated once so the per-fit
    // hessian path reuses them. `m = n_theta + p = params.len()`.
    /// m × m joint-deviance Hessian
    pub hess_scratch: Mat<f64>,
    /// length m; converged γ̂ snapshot restored each return
    pub fd_saved: Vec<f64>,
    /// length m; per-coordinate FD step h_k
    pub fd_steps: Vec<f64>,
    /// When true, `laplace_deviance_at` seeds PIRLS from `u_seed` (the fitted mode
    /// û(γ̂)) instead of û = 0. Set ONLY by `fd_hessian_cov`, and only for its
    /// perturbed evals: a FIXED shared seed (every eval from the same constant
    /// u_seed) keeps each f(γ) a function of γ alone, so FD order-independence
    /// holds — a *chained* seed would not. Reset on every `fd_hessian_cov` exit so
    /// non-FD callers keep their cold, order-free û = 0 start.
    pub warm_seed_active: bool,
    /// PIRLS exit-tol override read by `laplace_deviance_at` and forwarded to every
    /// PIRLS variant. `Some(PIRLS_TOL_REL_FD)` ONLY while `fd_hessian_cov` runs (set
    /// on entry, reset on every exit — the `warm_seed_active` discipline), so the FD
    /// second differences see a deviance smooth to ~1e-8 instead of the canonical
    /// 1e-6 exit noise. `None` everywhere else: the fit/BOBYQA path never pays the
    /// extra inner iterations and stays bit-identical.
    pub pirls_tol_override: Option<f64>,
}

impl GlmmWorkspace {
    /// Build the GLMM workspace for a Glm+cluster spec. `slope_cols` are the
    /// x_full indices of the primary slopes (`spec.cluster_slope_design_cols`).
    pub fn for_cluster_spec(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
        nagq: u8,
    ) -> Self {
        let groupings = LmmGroupings::from_cluster_spec(cluster, max_n, slope_cols);
        Self::from_groupings(groupings, cluster.family, p, max_n, nagq)
    }

    /// Build the workspace from an already-constructed `LmmGroupings` (+ family).
    /// The extracted tail of `for_cluster_spec` — everything past the groupings
    /// build depends only on `(groupings, family, p, max_n, nagq)`, never on the
    /// `ModelSpec` or `slope_cols` themselves. Split out so a per-thread FD-Hessian
    /// worker can reconstruct a fresh, identically-sized workspace from a live one's
    /// cloned groupings without re-threading the spec (`fd_worker_ws`).
    pub(crate) fn from_groupings(
        groupings: LmmGroupings,
        family: crate::Family,
        p: usize,
        max_n: usize,
        nagq: u8,
    ) -> Self {
        let k = groupings.k_total;
        let n_theta = groupings.n_theta();
        let q = groupings.primary_q;
        let n_primary = groupings.n_primary;
        // Structured-path block sizes: core width q_core = q_p + nested children,
        // crossed width e. Buffers stay 1-sized minima when the shape has no
        // extras (the no-extras blocked path never touches them).
        let q_core = q + groupings.nested_per_parent;
        let e_crossed = groupings.k_crossed();

        // Bounds: θ part from blind_theta_and_bounds; β part = [−BETA_BOX, BETA_BOX].
        let (theta0, mut lower, mut upper) = groupings.blind_theta_and_bounds();
        let mut params = theta0;
        params.extend(std::iter::repeat_n(0.0, p)); // β cold default; overwritten at fit
        lower.extend(std::iter::repeat_n(-BETA_BOX, p));
        upper.extend(std::iter::repeat_n(BETA_BOX, p));

        // ρ_begin ≤ RHO_BEGIN and ≤ 0.1·min diagonal θ₀ (mirror for_cluster_spec_ext)
        // so the cold blind start is not projected onto a bound. The start is now the
        // structure-only blind θ₀ (M3.5), so each diagonal entry is THETA0.
        let blind_theta = vec![crate::lmm::THETA0; n_theta];
        let min_diag = groupings
            .diagonal_theta()
            .iter()
            .map(|&i| blind_theta[i].max(THETA_TRUTH_FLOOR))
            .fold(f64::INFINITY, f64::min);
        // Hoisted so the joint solver below and `solver_stage1` (the θ-only stage-1
        // solver) share the exact same computed θ-portion rho_begin — a pure
        // extraction, not a new derivation.
        let rho_begin = (0.1 * min_diag).min(RHO_BEGIN);
        // MIRRORS the joint config in `sparse::glmm::fit_glmm_sparse` — both feed
        // through the shared `apply_campaign_overrides` tail.
        let mut config = Config {
            rho_begin,
            rho_end: GLMM_RHO_END,
            ..Config::new(n_theta + p)
        };
        crate::lmm::apply_campaign_overrides(&mut config, n_theta + p);
        // Stage-1 θ-only BOBYQA config: same rho_begin/rho_end schedule as the
        // joint solver above, but `npt` mirrors `sparse_lmm_seed`'s mid-model
        // rule (lmm.rs), NOT the joint solver's — the two are sized for
        // different-dimension searches and this crate's precedent for a
        // θ-only search is `sparse_lmm_seed`. MIRRORS `config1` in
        // `fit_glmm_sparse` (sparse.rs) — change together. Both feed through
        // the shared `apply_campaign_overrides` tail.
        let npt_stage1 = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        let mut config_stage1 = Config {
            rho_begin,
            rho_end: GLMM_RHO_END,
            npt: npt_stage1,
            ..Config::new(n_theta)
        };
        crate::lmm::apply_campaign_overrides(&mut config_stage1, n_theta);
        // θ-only incumbent buffer, seeded from the θ-prefix of the joint start.
        let params_stage1 = params[..n_theta].to_vec();

        GlmmWorkspace {
            groupings,
            family,
            nb_theta: f64::NAN,
            nagq,
            // Scalar AGQ (`q_p==1`, `agq::agq_deviance`) uses `4·s` slots
            // (ctr|ucj|acc|sum). The vector kernel (`q_p∈2..=3`,
            // `agq::agq_deviance_vec`) instead needs `2·s` (ctr|sum) plus a
            // per-eval product-grid node table of `k^q·(q+1)` f64 (the q z-vector
            // components + the summed Liu–Pierce reweight per node), `k=nagq`
            // fixed per workspace. `nagq=1` shapes never reach the vector kernel,
            // so their `k^q=1` table is a harmless `q+1` slots.
            agq_scratch: if q >= 2 {
                let kq = (nagq as usize).pow(q as u32);
                vec![0.0; (2 * n_primary + kq * (q + 1)).max(1)]
            } else {
                vec![0.0; (4 * n_primary).max(1)]
            },
            parallel_inner: false,
            cluster_rows: None,
            k,
            p,
            n_theta,
            z: Mat::zeros(max_n, k.max(1)),
            m: Mat::zeros(max_n, k.max(1)),
            solver: Bobyqa::new(n_theta + p, config)
                .expect("BOBYQA config constants are valid by construction"),
            params,
            lower,
            upper,
            solver_stage1: Bobyqa::new(n_theta, config_stage1)
                .expect("BOBYQA config constants are valid by construction"),
            params_stage1,
            // Dimension-gated: for small (n_theta, p) shapes stage 1 roughly doubles
            // the PIRLS-solve count (see the field doc) while BOBYQA reaches the same
            // stage-2 optimum blind almost as fast, so it isn't worth its own cost —
            // BUT for wider joint dims the un-warm-started single-stage search can cost
            // much MORE than stage 1 saves (a first cut at this threshold, gated only
            // on named datasets rather than a corpus sweep, missed this and shipped a
            // regression — see below).
            //
            // Threshold below is from a locked-machine (`bench-l`, `taskset -c 1`)
            // timing sweep of every rung that reaches this constructor (non-Gaussian
            // dense NoZ; `parity/` datasets with a sparse/LMM path are unaffected by
            // this field and were confirmed identical across both sweep arms) — NOT
            // just the plan's named reach set, which turned out to include one false
            // positive. Protocol: per arm (forced skip vs forced keep), two independent
            // `parity_fit` invocations, each itself the median of 9 timed samples
            // after a discarded warmup (see parity/oracle/fit.rs); invocations agreed
            // within ~2% everywhere, and the table shows the keep-arm/skip-arm medians
            // (Poisson rows re-measured after the dense-PIRLS weight-loop revert that
            // restored the pre-helper per-row math; the logit rows — cbpp, VerbAgg —
            // run the fused-SIMD arm that revert never touched):
            //
            //   dataset             (n_theta,p)  skip-vs-keep fit_median   verdict
            //   cbpp                (1,4)        0.0036 vs 0.0041s (-14%)  SKIP wins
            //   sim_poisson_nested  (2,2)        0.0085 vs 0.0099s (-14%)  SKIP wins
            //   grouseticks         (3,4)        0.4053 vs 0.2168s (+87%)  KEEP wins
            //   Arabidopsis         (2,6)        0.0772 vs 0.0276s (+180%) KEEP wins
            //   VerbAgg             (2,7)        2.1575 vs 1.0082s (+114%) KEEP wins
            //   sim_crossed_at_cap  (7,2)        0.1736 vs 0.1263s (+37%)  KEEP wins
            //
            // Arabidopsis (n_theta=2, p=6) is the false positive: an earlier version of
            // this threshold (`n_theta <= 2 && p <= 6`) put it in the skip set purely
            // because it matched the two true winners on n_theta, without checking p
            // against a real measurement — it is in fact the single biggest regression
            // in the corpus (skip is 2.8x SLOWER), because the un-warm-started 8-dim
            // joint BOBYQA search costs far more than the skipped stage-1 pass saves.
            // (Correctness was NOT at risk either way — beta/SE/varcomp agreed to
            // ~1e-6 relative between skip and keep on Arabidopsis; this is purely a
            // performance threshold, re-derive it if the corpus changes.)
            //
            // The two true winners both have p ≤ 4; every loser (including the false
            // positive) has p ≥ 6 — a wide, data-supported margin — so `p ≤ 4` replaces
            // the earlier `p ≤ 6`. `n_theta ≤ 2` is unchanged (grouseticks at n_theta=3
            // is the nearest loser on that axis and was never miscategorized).
            two_stage: !(n_theta <= 2 && p <= 4),
            eta: vec![0.0; max_n],
            prob: vec![0.0; max_n],
            w: vec![0.0; max_n],
            eta_fixed: vec![0.0; max_n],
            m_buf: vec![0.0; max_n * q],
            z_buf: vec![0.0; max_n * (q - 1)],
            mu: vec![0.0; max_n],
            prior_w: vec![1.0; max_n],
            weighted: false,
            u: vec![0.0; k.max(1)],
            u_prev: vec![0.0; k.max(1)],
            u_seed: vec![0.0; k.max(1)],
            a: Mat::zeros(k.max(1), k.max(1)),
            a_chol: Mat::zeros(k.max(1), k.max(1)),
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                k.max(1),
                Par::Seq,
                Spec::default(),
            )),
            wm: Mat::zeros(max_n, k.max(1)),
            wx: Mat::zeros(max_n, p),
            a_rhs: vec![0.0; k.max(1)],
            a_blocks: vec![0.0; (q * q * n_primary).max(1)],
            core_blocks: vec![0.0; (q_core * q_core * n_primary).max(1)],
            coupling: vec![0.0; (q_core * n_primary * e_crossed).max(1)],
            schur_blk: vec![0.0; (e_crossed * e_crossed).max(1)],
            lam: vec![0.0; q * q],
            // Packed-M buffers (structured path). `q_core = q + nested_per_parent`,
            // `G_cap = MAX_EXTRA_GROUPINGS`. `.max(1)` keeps a valid (never-read)
            // allocation on the no-extras shapes that route elsewhere.
            m_core_buf: vec![0.0; (max_n * q_core).max(1)],
            cross_val: vec![0.0; (max_n * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
            cross_col: vec![0u32; (max_n * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
            n_cross: vec![0u8; max_n.max(1)],
            coup_cols: vec![0u32; (max_n * crate::lmm::MAX_EXTRA_GROUPINGS).max(1)],
            coup_ptr: vec![0u32; n_primary + 1],
            coup_mask: None,
            structured_schur: None,
            force_dense_schur: false,
            xtwx: Mat::zeros(p, p),
            xtwm: Mat::zeros(p, k.max(1)),
            ainv_mtwx: Mat::zeros(k.max(1), p),
            schur: Mat::zeros(p, p),
            schur_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                p,
                Par::Seq,
                Spec::default(),
            )),
            betas: vec![0.0; p],
            beta_rhs: vec![0.0; p],
            beta_prev: vec![0.0; p],
            beta_prof: vec![0.0; p],
            beta_seed: vec![0.0; p],
            var_diag: vec![0.0; p],
            vcov: Mat::zeros(p, p),
            vcov_cols: Mat::zeros(p, p),
            t_sq: vec![0.0; p],
            theta_se: vec![f64::NAN; n_theta],
            fwd_solve: vec![0.0; p],
            joint_k_inv: Mat::zeros(p, p),
            joint_sigma_t_chol: Mat::zeros(p, p),
            joint_rhs: vec![0.0; p],
            hess_scratch: Mat::zeros((n_theta + p).max(1), (n_theta + p).max(1)),
            fd_saved: vec![0.0; n_theta + p],
            fd_steps: vec![0.0; n_theta + p],
            warm_seed_active: false,
            pirls_tol_override: None,
        }
    }
}

/// Fresh workspace for one FD-Hessian worker thread: an independently-sized clone
/// of `src` carrying everything an FD deviance eval reads or mutates, so a rayon
/// grid thread never aliases the live workspace. Bit-identity rests on `fd_eval`
/// restoring `params` from `fd_saved` and seeding û from the frozen `u_seed` every
/// eval — each grid cell is a pure function of `(fd_saved, fd_steps, u_seed,
/// design)`, identical whichever workspace computes it.
///
/// Construction reuses `from_groupings` (fresh scratch, correctly sized) on a
/// CLONE of `src`'s groupings, then copies the load-bearing state the fresh
/// constructor zeroes: the built design (`z`, `z_buf`), the structured crossed
/// factor (rebuilt as its own per-thread scratch — see `StructuredSchur::
/// clone_scratch`), and the FD seed state `fd_hessian_cov` set before the grid.
/// A missed field is a silent aliasing bug; the knob-on-vs-off bit-identity test
/// in `glmm/tests.rs` is the enforcement.
///
/// `coup_mask` is deliberately left `None` (the fresh constructor's value): the
/// worker rebuilds its own coupling CSR on the first structured eval, matching the
/// serial path's per-fit rebuild. `nb_theta`/`force_dense_schur` are copied because
/// the deviance reads them; `cluster_rows` stays `None` — on the AGQ path the
/// node-outer fallback it triggers is bit-identical to the cluster-outer loop.
#[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
pub(crate) fn fd_worker_ws(src: &GlmmWorkspace, n: usize) -> GlmmWorkspace {
    let mut w =
        GlmmWorkspace::from_groupings(src.groupings.clone(), src.family, src.p, n, src.nagq);
    // Built design (build_z / fill_z_f64 output — constant across the FD grid).
    // src may be sized for max_n >= n (reusable-workspace surface); only the first
    // n rows are live, and fill_z_f64 fills row-major, so a prefix slice is correct.
    w.z = src.z.clone();
    let len = w.z_buf.len();
    w.z_buf.copy_from_slice(&src.z_buf[..len]);
    w.prior_w[..n].copy_from_slice(&src.prior_w[..n]);
    w.weighted = src.weighted;
    // Crossed-Schur factor: fresh per-thread scratch over the same symbolic pattern.
    w.structured_schur = src.structured_schur.as_ref().map(|ss| ss.clone_scratch());
    w.nb_theta = src.nb_theta;
    w.force_dense_schur = src.force_dense_schur;
    // FD seed state (fd_hessian_cov sets these before the grid).
    w.params.copy_from_slice(&src.params);
    w.fd_saved.copy_from_slice(&src.fd_saved);
    w.fd_steps.copy_from_slice(&src.fd_steps);
    w.u_seed.copy_from_slice(&src.u_seed);
    w.warm_seed_active = src.warm_seed_active;
    w.pirls_tol_override = src.pirls_tol_override;
    w
}

// Kernel — written as borrow-split FREE fns so the BOBYQA closure in fit_glmm can call
// them on destructured workspace fields without re-borrowing the whole workspace.

/// In-place lower Crout Cholesky of a `q×q` block stored row-major in `blk`
/// (lower triangle read; on return the lower triangle holds L). Returns false on
/// a non-positive pivot — the module's failure surface. q ≤ MAX_PRIMARY_Q (8).
/// Thin wrapper over the shared kernel in `crate::linalg::block_chol`.
pub(crate) fn glmm_block_chol(blk: &mut [f64], q: usize) -> bool {
    crate::linalg::block_chol(blk, q)
}

/// Solve `L Lᵀ x = b` in place (`b` overwritten with `x`) for the `q×q` lower
/// factor `l` produced by `glmm_block_chol` (row-major, diagonal = L pivots).
/// Forward `L y = b` then back `Lᵀ x = y`.
pub(crate) fn glmm_block_solve(l: &[f64], q: usize, b: &mut [f64]) {
    for r in 0..q {
        let mut v = b[r];
        for c in 0..r {
            v -= l[r * q + c] * b[c];
        }
        b[r] = v / l[r * q + r];
    }
    for r in (0..q).rev() {
        let mut v = b[r];
        for c in (r + 1)..q {
            v -= l[c * q + r] * b[c];
        }
        b[r] = v / l[r * q + r];
    }
}

/// Panel variant of `glmm_block_solve`: solve `L Lᵀ X = B` in place for a
/// row-major `q×nc` RHS panel (`panel[r·nc..(r+1)·nc]` = row r) against the same
/// row-major factor. Identical substitution with the column loop hoisted inside:
/// each factor entry is read once per row op and the inner loop runs over the
/// contiguous row slice (vectorizable axpy) instead of re-walking the factor
/// once per RHS column.
pub(crate) fn glmm_block_solve_panel(l: &[f64], q: usize, panel: &mut [f64], nc: usize) {
    for r in 0..q {
        let (done, rest) = panel.split_at_mut(r * nc);
        let row_r = &mut rest[..nc];
        for c in 0..r {
            let lrc = l[r * q + c];
            for (x, &y) in row_r.iter_mut().zip(&done[c * nc..(c + 1) * nc]) {
                *x -= lrc * y;
            }
        }
        let d = l[r * q + r];
        for x in row_r.iter_mut() {
            *x /= d;
        }
    }
    for r in (0..q).rev() {
        let (head, rest) = panel.split_at_mut((r + 1) * nc);
        let row_r = &mut head[r * nc..];
        for c in (r + 1)..q {
            let lcr = l[c * q + r];
            for (x, &y) in row_r.iter_mut().zip(&rest[(c - r - 1) * nc..(c - r) * nc]) {
                *x -= lcr * y;
            }
        }
        let d = l[r * q + r];
        for x in row_r.iter_mut() {
            *x /= d;
        }
    }
}

/// Cached sparse factor of the `e`-wide crossed Schur complement `S`.
/// `S`'s sparsity pattern is fixed by the crossed incidence (θ-independent), so the
/// symbolic factor is built ONCE per fit here and numeric-refactored every PIRLS
/// iteration / BOBYQA eval into `l_values`. Mirrors `SparseLmmWorkspace`
/// (`sparse.rs`), one factor narrower — only the crossed tail, not the whole system.
/// `None` for nested-only shapes (`e = 0`, no Schur).
pub(crate) struct StructuredSchur {
    /// Symbolic Cholesky of `S`'s pattern (AMD; simplicial or supernodal by
    /// faer's AUTO heuristic — `logdet_llt` handles both). Reused every refactor.
    pub(crate) symbolic: SymbolicCholesky<usize>,
    /// `S`'s value container in the fixed CSC pattern (lower tri + full diagonal).
    /// Values overwritten per PIRLS iteration by a gather from the dense `schur_blk`;
    /// pattern never changes. `parts_mut()` gives (symbolic, values) for the gather.
    pub(crate) axx: SparseColMat<usize, f64>,
    /// L-factor value buffer, length `symbolic.len_val()`. Overwritten per refactor.
    pub(crate) l_values: Vec<f64>,
    /// Numeric-factor scratch, sized once from `factorize_numeric_llt_scratch`.
    pub(crate) fac_mem: MemBuffer,
    /// Solve scratch, sized once from `solve_in_place_scratch(1, …)` — the Schur
    /// back-solve (PIRLS and each SE column) is always a single RHS column.
    pub(crate) solve_mem: MemBuffer,
    /// Downdate panels for `structured_factor`'s per-cluster `S −= C_f'A_f⁻¹C_f`
    /// (the LMM sparse-tail kernels A–D port): `c_panel` the gathered nonzero
    /// coupling columns (row-major `qc×e_f`), `y_panel` its `A_f⁻¹`-solved copy,
    /// `dd_temp` the `C_f'·Y` product (col-major `e_f×e_f`, lower). Sized once
    /// to `max_f e_f` off the FULL θ-independent incidence (`cols_of` in `new` —
    /// a superset of every θ-masked `coup_cols` CSR the fit visits), overwritten
    /// per cluster. Used only at `qc > 1`: at `qc == 1` `structured_factor`
    /// routes to the scalar walk and `new` sizes these to 0 (change together).
    /// The panel is NOT a win at qc=1 — the downdate is rank-1, so the staging
    /// (gather + `dd_temp` + a second scatter pass) doubles memory traffic for
    /// identical arithmetic and measured a +4–7% per-eval loss on the cross6
    /// GLMM cells (2026-07-14 drift investigation). It stays for qc>1, the only
    /// case that has real qc×e_f batched work to amortize the staging.
    pub(crate) c_panel: Vec<f64>,
    pub(crate) y_panel: Vec<f64>,
    pub(crate) dd_temp: Vec<f64>,
}

impl StructuredSchur {
    pub(crate) fn new(
        g: &LmmGroupings,
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        n: usize,
    ) -> Option<StructuredSchur> {
        let e = g.k_crossed();
        if e == 0 {
            return None;
        }
        let s = g.n_primary;
        let k_family = (g.primary_q + g.nested_per_parent) * s;
        // Per-cluster set of crossed block-local columns each cluster touches — the
        // FULL incidence over all crossed groupings (NOT θ-filtered; the pattern must
        // be a superset for every θ the optimizer visits).
        let mut cols_of: Vec<Vec<u32>> = vec![Vec::new(); s];
        for cf in g.crossed.iter() {
            let off = g.extra_offsets[cf.decl];
            let ids = &extra_ids[cf.decl];
            for i in 0..n {
                let f = cluster_ids[i] as usize;
                let b = off + ids[i] as usize * cf.q - k_family;
                cols_of[f].push(b as u32);
            }
        }
        for v in cols_of.iter_mut() {
            v.sort_unstable();
            v.dedup();
        }
        // Pattern triplets: full diagonal + Σ_f (coup_cols[f] × coup_cols[f]) lower tri.
        // Dedup by a visited-set of (a,b) so try_new_from_triplets sees each once.
        let mut seen = std::collections::HashSet::<(usize, usize)>::new();
        let mut trips = Vec::<Triplet<usize, usize, f64>>::new();
        for b in 0..e {
            trips.push(Triplet::new(b, b, 0.0));
            seen.insert((b, b));
        }
        for cols in &cols_of {
            for &bb in cols {
                for &aa in cols {
                    let (a, b) = (aa as usize, bb as usize);
                    if a >= b && seen.insert((a, b)) {
                        trips.push(Triplet::new(a, b, 0.0));
                    }
                }
            }
        }
        let axx = SparseColMat::<usize, f64>::try_new_from_triplets(e, e, &trips)
            .expect("Schur pattern triplets well-formed");
        let symbolic = factorize_symbolic_cholesky(
            axx.symbolic(),
            Side::Lower,
            Default::default(), // AMD fill-reducing ordering
            CholeskySymbolicParams {
                // AUTO: simplicial or supernodal per pattern; `logdet_llt`
                // handles both arms. Mirrors `clone_scratch` — change together.
                supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
                ..Default::default()
            },
        )
        .expect("Schur symbolic factorization");
        let l_values = vec![0.0f64; symbolic.len_val()];
        let fac_mem = MemBuffer::new(
            symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default()),
        );
        let solve_mem = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
        let qc = g.primary_q + g.nested_per_parent;
        let max_ef = cols_of.iter().map(|v| v.len()).max().unwrap_or(0);
        // At qc == 1 `structured_factor` routes the downdate to its scalar arm
        // (the panel staging is a +4–7% per-eval loss there; 2026-07-14 drift
        // investigation), so the panel buffers are never touched — size them to 0.
        // Mirrors the `qc != 1` filter in `structured_factor` — change together:
        // widening the route without resizing slices zero-length buffers and
        // panics. `clone_scratch` follows automatically (it mirrors these lengths).
        let panel_ef = if qc == 1 { 0 } else { max_ef };
        Some(StructuredSchur {
            symbolic,
            axx,
            l_values,
            fac_mem,
            solve_mem,
            c_panel: vec![0.0f64; qc * panel_ef],
            y_panel: vec![0.0f64; qc * panel_ef],
            dd_temp: vec![0.0f64; panel_ef * panel_ef],
        })
    }

    /// Per-thread clone for an FD-Hessian worker: shares nothing mutable with
    /// `self`. The symbolic pattern (`axx`) is copied and RE-factorized here rather
    /// than cloned — `SymbolicCholesky` is not `Clone`, but `factorize_symbolic_
    /// cholesky` on the same pattern with the same (deterministic AMD) ordering
    /// reproduces it bit-for-bit, so the per-eval numeric refactor lands on the
    /// identical elimination tree as `self`'s. `l_values`/scratch are fresh (their
    /// contents are overwritten every refactor). Mirrors `new`'s tail exactly.
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    pub(crate) fn clone_scratch(&self) -> StructuredSchur {
        let axx = self.axx.clone();
        let symbolic = factorize_symbolic_cholesky(
            axx.symbolic(),
            Side::Lower,
            Default::default(), // AMD fill-reducing ordering (deterministic)
            CholeskySymbolicParams {
                // Mirrors `new` — change together (same pattern + same params
                // ⇒ same supernodal/simplicial decision on every worker).
                supernodal_flop_ratio_threshold: SupernodalThreshold::AUTO,
                ..Default::default()
            },
        )
        .expect("Schur symbolic factorization");
        let l_values = vec![0.0f64; symbolic.len_val()];
        let fac_mem = MemBuffer::new(
            symbolic.factorize_numeric_llt_scratch::<f64>(Par::Seq, Spec::default()),
        );
        let solve_mem = MemBuffer::new(symbolic.solve_in_place_scratch::<f64>(1, Par::Seq));
        StructuredSchur {
            symbolic,
            axx,
            l_values,
            fac_mem,
            solve_mem,
            c_panel: vec![0.0f64; self.c_panel.len()],
            y_panel: vec![0.0f64; self.y_panel.len()],
            dd_temp: vec![0.0f64; self.dd_temp.len()],
        }
    }
}

/// Build the dense RE design Z (`max_n × k`, level-major) for one dataset.
///
/// Layout mirrors `LmmGroupings`'s RE-column convention (`from_cluster_spec`):
/// the primary block is `q_p · n_primary` wide, level-major within each
/// component (`[intercept 0..S | slope_0 … | slope_{q-2}]` → at level `lvl`,
/// component `c`, the column is `lvl·q_p + c`), then each extra grouping's
/// indicator columns at its ABSOLUTE `extra_offsets[e]` (already includes the
/// primary block width — do not add it again). `slope_cols` index `x`.
///
/// Builds the GLMM design-`Z` (`ws.z`) for the dense-extras path; the
/// block-diagonal / structured fits reconstruct `mᵢ` per row instead.
pub fn build_z(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) {
    let g = &ws.groupings;
    let q = g.primary_q;
    for c in 0..ws.k {
        for i in 0..n {
            ws.z[(i, c)] = 0.0;
        }
    }
    for i in 0..n {
        let lvl = cluster_ids[i] as usize;
        let base = lvl * q;
        ws.z[(i, base)] = 1.0; // intercept
                               // Slope cols come from the workspace's own `primary_slope_cols` (set once at
                               // construction), mirroring `fill_z_f64` — not a per-call param. q_p−1 == #slopes.
        for d in 0..q - 1 {
            ws.z[(i, base + 1 + d)] = x[(i, g.primary_slope_cols[d])];
        }
    }
    for (e, ids) in extra_ids.iter().enumerate() {
        let off = g.extra_offsets[e]; // ABSOLUTE — do not add primary width again
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            ws.z[(i, off + ids[i] as usize)] = 1.0;
        }
    }
}

/// Per-fit f64 widening of the primary-slope columns: `z_buf[i·(q−1)+d] =
/// x[i, slope_cols[d]]`. θ/β change per BOBYQA eval but `x` is fixed per fit,
/// so this hoists every f32 MatRef load out of the per-solve M fill — the fill
/// becomes a pure contiguous-f64 product. f32→f64 widening is value-exact, so
/// bit-identity is preserved. No-op at q_p = 1 (no slope columns).
pub(crate) fn fill_z_f64(g: &LmmGroupings, x: MatRef<f64>, z_buf: &mut [f64], n: usize) {
    let q = g.primary_q;
    for i in 0..n {
        for d in 0..q - 1 {
            z_buf[i * (q - 1) + d] = x[(i, g.primary_slope_cols[d])];
        }
    }
}

/// Form M = ZΛ in place, with Λ block-diagonal: a shared lower-triangular
/// primary block `Λ_p` (q_p×q_p) repeated per primary level, then one scalar
/// θ_e per extra grouping. `Λ_p` is the column-major vech θ prefix expanded by
/// `lmm::primary_lambda` (row-major lower-tri storage, so `lam[r*q + c]` is its
/// (r,c) entry). Each extra grouping's columns scale by its θ scalar.
pub(crate) fn apply_lambda(
    groupings: &LmmGroupings,
    params: &[f64],
    z: MatRef<f64>,
    m: &mut Mat<f64>,
    lam: &mut [f64],
    n: usize,
) {
    let q = groupings.primary_q;
    let s = groupings.n_primary;
    crate::lmm::primary_lambda(&params[..groupings.n_theta()], q, lam);
    for lvl in 0..s {
        let base = lvl * q;
        for i in 0..n {
            for c in 0..q {
                let mut acc = 0.0;
                for r in c..q {
                    acc += z[(i, base + r)] * lam[r * q + c];
                }
                m[(i, base + c)] = acc;
            }
        }
    }
    let base_theta = q * (q + 1) / 2;
    // The GLMM structured path carries intercept-only extras (q_g == 1), so each
    // extra owns a single scalar θ at `base_theta + e` (== its `vech_start`).
    // Slope-carrying extras (q_g > 1) never reach here: `classify_design` routes
    // any extra-slopes shape to `Solver::Sparse` for every family.
    debug_assert!(!groupings.extra_slopes_any);
    // Each extra grouping owns a CONTIGUOUS column block at its ABSOLUTE
    // `extra_offsets[e]`, scaled by its own scalar θ. Span the block by the
    // grouping's OWN width — NOT the gap to the next declaration's offset:
    // `extra_offsets` is non-monotonic (a nested grouping always sits at the low
    // `prim_width` slot, so a crossed-before-nested declaration makes offsets
    // decrease), so a "scale up to the next offset" loop empties one block and
    // over-scales another. A nested grouping spans `n_primary · nested_per_parent`
    // child columns; a crossed grouping spans its stored level count.
    for (e, &off) in groupings.extra_offsets.iter().enumerate() {
        let theta_e = params[base_theta + e];
        let width = if groupings.nested.map(|nf| nf.vech_start) == Some(base_theta + e) {
            s * groupings.nested_per_parent
        } else {
            groupings
                .crossed
                .iter()
                .find(|cf| cf.vech_start == base_theta + e)
                .map(|cf| cf.n_levels)
                .expect("an extra grouping is either nested or crossed")
        };
        for col in off..off + width {
            for i in 0..n {
                m[(i, col)] = z[(i, col)] * theta_e;
            }
        }
    }
}

/// Pack the STRUCTURED-path nonzeros of `M = ZΛ` into the workspace's packed
/// buffers, once per deviance eval — the structured analogue of `apply_lambda`,
/// which it replaces on this path (`apply_lambda` writes the full dense `n×k`
/// every eval; this writes only the `q_core` core + ≤`G` crossed nonzeros each
/// row reads). `m_core_buf[i·q_core+local]` = the `Λ`-scaled core value
/// `M[(i, core_col(f,local))]` for row `i`'s primary cluster `f = cluster_ids[i]`
/// (primary `local<q`: `Σ_{r≥local} z_r·lam[r·q+local]`, mirroring `apply_lambda`'s
/// core write and the blocked-path fill; nested `local≥q`: the nested indicator
/// scaled by its θ). For each crossed grouping with `θ≠0`, the row's single active
/// level contributes one nonzero: `cross_val = z·θ`, `cross_col = b` (the crossed
/// block-local index, `0..e`), with `n_cross[i]` the count (`≤ G`). A θ-pinned
/// (θ=0) grouping is skipped, mirroring `apply_lambda`'s `z·θ=0` ⇒ no nonzero. The
/// crossed-column scan over `z` is O(n·e) but runs ONCE per eval; the per-PIRLS-
/// iteration passes then read O(n·G). Reads `z` (the dense design `build_z` left),
/// `lam` (filled here via `primary_lambda`), `params`, and `cluster_ids`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_packed_m(
    g: &LmmGroupings,
    params: &[f64],
    z: MatRef<f64>,
    lam: &mut [f64],
    cluster_ids: &[u32],
    m_core_buf: &mut [f64],
    cross_val: &mut [f64],
    cross_col: &mut [u32],
    n_cross: &mut [u8],
    n: usize,
) {
    let q = g.primary_q;
    let s = g.n_primary;
    let np = g.nested_per_parent;
    let qc = q + np;
    let prim_width = q * s;
    let k_family = qc * s;
    let base_theta = q * (q + 1) / 2;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    // Intercept-only extras on the GLMM structured path (see `apply_lambda`;
    // `classify_design` routes extra-slopes shapes to Sparse for every family).
    debug_assert!(!g.extra_slopes_any);
    crate::lmm::primary_lambda(&params[..g.n_theta()], q, lam);
    let theta_nested = g.nested.map(|nf| params[nf.vech_start]).unwrap_or(0.0);
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        // Core primary block: same `Σ_{r≥c} z_r·lam[r·q+c]` reduction `apply_lambda`
        // writes at M[(i, f·q+c)], packed to the local layout the passes read.
        for c in 0..q {
            let mut acc = 0.0;
            for r in c..q {
                acc += z[(i, f * q + r)] * lam[r * q + c];
            }
            m_core_buf[i * qc + c] = acc;
        }
        // Core nested children of parent f: the indicator scaled by θ_nested (one of
        // the np slots is 1, the rest 0 — kept as written zeros so the passes sum a
        // contiguous q_core slice).
        for j in 0..np {
            let col = prim_width + f * np + j;
            m_core_buf[i * qc + q + j] = z[(i, col)] * theta_nested;
        }
        // Crossed: one nonzero per crossed grouping (its single active level), θ-pinned
        // groupings skipped.
        let mut cnt = 0usize;
        for cf in &g.crossed {
            let theta = params[cf.vech_start];
            if theta == 0.0 {
                continue;
            }
            // q_g==1 here (see debug_assert above), so vech_start − base_theta is
            // this factor's declaration index into extra_offsets.
            let off = g.extra_offsets[cf.vech_start - base_theta];
            for col in off..off + cf.n_levels {
                let zv = z[(i, col)];
                if zv != 0.0 {
                    cross_col[i * g_cap + cnt] = (col - k_family) as u32;
                    cross_val[i * g_cap + cnt] = zv * theta;
                    cnt += 1;
                    break;
                }
            }
        }
        n_cross[i] = cnt as u8;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_schur_new_builds_symbolic_for_grouseticks() {
        // Shared grouseticks 3-crossed fixture (INDEX primary, [BROOD, LOCATION]
        // crossed); this test needs only the groupings + ids, not X/y.
        let (model, ids, _x, _y, n, _p) = crate::glmm::tests::grouseticks_3crossed_fixture();
        let g = LmmGroupings::from_cluster_spec(&model, n, &[]);
        let cluster_ids = ids.primary;
        let extra_ids = ids.extra;
        let ss = StructuredSchur::new(&g, &cluster_ids, &extra_ids, n).expect("e = 181 > 0 ⇒ Some");
        assert_eq!(ss.axx.ncols(), g.k_crossed());
        assert_eq!(ss.axx.ncols(), 181);
        // Symbolic factor allocated a non-empty L; diagonal is fully present.
        assert!(
            ss.symbolic.len_val() >= ss.axx.ncols(),
            "at least the e diagonal entries"
        );
        assert_eq!(ss.l_values.len(), ss.symbolic.len_val());
        // Fill-in is far below dense: dense would be e·(e+1)/2 = 16471 lower entries.
        assert!(
            ss.symbolic.len_val() < 16471,
            "sparse factor must have less fill than the dense lower triangle"
        );
    }
}
