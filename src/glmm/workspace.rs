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
    /// Outcome family/link selecting the PIRLS IRLS math — the arm
    /// `simd_transcendental::family_pass` dispatches to each iteration.
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
    /// max_n × k dense RE design (built per (spec, N) by `build_z`). Allocated
    /// only on the dense fallback route (extras present and the core oversized,
    /// `!structured_extras_eligible()`) — 0×0 on the blocked AND structured
    /// routes: the structured route packs its nonzeros straight from the ids
    /// (`build_packed_m`) and never materializes the dense design.
    pub z: Mat<f64>,
    /// max_n × k = ZΛ (rebuilt per BOBYQA eval). Allocated only on the dense
    /// fallback route (extras present and the core oversized,
    /// `!structured_extras_eligible()`) — 0×0 on the blocked and structured
    /// routes, which never read it (`deviance.rs:194`).
    pub m: Mat<f64>,
    /// Joint (θ,β) BOBYQA solver, dimension `n_theta + p`.
    pub solver: Bobyqa, // sized n_theta + p
    /// Joint solver's live iterate: `[θ (n_theta) | β (p)]`. STABLE READ-BACK
    /// CONVENTION: after `fit_glmm` returns with `converged == true`, this
    /// holds the pinned optimum — `params[..n_theta]` is θ̂ (boundary
    /// components zeroed) and `params[n_theta..n_theta + p]` is β̂ (stage 2 is
    /// a joint [θ | β] solve, and `betas` is copied from this suffix) — so a
    /// caller may read it back as the warm start for a subsequent fit of
    /// related data. On a non-converged fit the content is an arbitrary
    /// iterate — do not read it.
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
    /// True iff `prior_w` was filled from `FitOptions::weights`. Selects between
    /// the two logit arms of `simd_transcendental::family_pass`: the fused
    /// `Σ log1pexp` deviance identity holds only for unweighted Bernoulli rows.
    pub(crate) weighted: bool,
    /// Current RE-mode iterate û, length k.
    pub u: Vec<f64>,
    /// previous accepted PIRLS iterate, step-halving backtrack buffer (len k.max(1))
    pub u_prev: Vec<f64>,
    /// within-fit û warm-start incumbent; RESET to 0 each fit_glmm — never carried across fits
    pub u_seed: Vec<f64>,
    /// k × k  M'WM + I. `dense_schur_fill` (se.rs) re-factors THIS field after a
    /// converged Fixed-mode PIRLS solve, so `pirls_solve` must leave it holding
    /// the raw symmetric A — never the in-place Cholesky factor. Allocated
    /// only on the dense fallback route — 0×0 on the blocked and structured
    /// routes, which never read it (`deviance.rs:194`).
    pub a: Mat<f64>,
    /// Copy-then-factor target for `a`'s Cholesky (k×k): `pirls_solve` copies
    /// `a`'s lower triangle in here (mirroring `.llt(Side::Lower)`'s internal
    /// copy) and runs `cholesky_in_place` on THIS buffer, leaving `a` itself
    /// untouched for `dense_schur_fill` to re-read. Allocated only on the
    /// dense fallback route — 0×0 on the blocked and structured routes, which
    /// never read it (`deviance.rs:194`).
    pub a_chol: Mat<f64>,
    /// Scratch for `a_chol`'s in-place `cholesky_in_place` (k×k, θ-independent
    /// size) — avoids the per-PIRLS-iteration `.llt(Side::Lower)` allocation on
    /// the dense `pirls_solve` hot path. Allocated only on the dense fallback
    /// route — 0×0 on the blocked and structured routes, which never read it
    /// (`deviance.rs:194`).
    pub a_llt_mem: MemBuffer,
    /// max_n × k = W∘M scratch for the dense-Gram GEMM (rebuilt per PIRLS
    /// iteration). Allocated only on the dense fallback route — 0×0 on the
    /// blocked and structured routes, which never read it (`deviance.rs:194`).
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
    // joint Wald scratch (reuse lmm::joint_wald_chi_sq):
    /// Inverse of the joint Wald K matrix, p×p (see `lmm::joint_wald_chi_sq`).
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
    /// û(γ̂)) instead of û = 0. Set ONLY by `fd_hessian_cov`, for **every** one of
    /// its evals including the central f0, and reset on every `fd_hessian_cov` exit
    /// so non-FD callers keep their cold, order-free û = 0 start. Same fixed-seed
    /// FD-derivative invariant as `fd_hessian_cov` in se.rs — see there for the
    /// derivation and for why f0 is inside the warm set too.
    pub warm_seed_active: bool,
    /// PIRLS exit-tol override read by `laplace_deviance_at` and forwarded to every
    /// PIRLS variant. `Some(pirls_tol_fd(family))` ONLY while `fd_hessian_cov` runs
    /// (set on entry, reset on every exit — the `warm_seed_active` discipline), so
    /// the FD second differences see a deviance converged at least as far as the
    /// fit's own exit. `None` everywhere else: the fit/BOBYQA path never pays the
    /// extra inner iterations and stays bit-identical.
    pub pirls_tol_override: Option<f64>,
    /// Per-row linear-predictor offset (`FitOptions::offset`), read by every
    /// `eta_fixed` refresh (`pirls::refresh_eta_fixed` and its two blocked-path
    /// inline twins). `None` ⇒ no offset, byte-identical to the pre-offset code.
    pub(crate) offset: Option<Vec<f64>>,
    /// Count of fit-path (`pirls_tol_override.is_none()`) PIRLS solves that ran
    /// the full `PIRLS_MAX_ITERS` cap without converging — observation-only,
    /// bookkeeping read back by `FitDiagnostics`/`Note::PirlsExhausted`, never by
    /// any numeric path. Reset to 0 at the top of every `fit_glmm` (mirrors
    /// `u_seed`) so a `loop_advanced` reuse never carries a prior draw's count.
    pub(crate) pirls_exhausted: u32,
    /// Whether the FINAL re-evaluation at the pinned γ̂ itself exhausted the
    /// PIRLS cap. Reset to `false` at the top of every `fit_glmm`.
    pub(crate) final_pirls_exhausted: bool,
    /// Observation-only optimizer counters for the fit in progress — the stage
    /// split, the shrink phase, the PIRLS histogram and the AGQ node cost.
    /// Never read by any numeric path. Reset to `new()` at the top of every
    /// `fit_glmm` (mirrors `pirls_exhausted`) so a `loop_advanced` reuse never
    /// carries a prior draw's counts.
    pub(crate) counters: crate::counters::EvalCounters,
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
        // The dense GLMM kernel builds intercept-only extra groupings exclusively
        // (`build_z` emits no slope columns for extras), so a slope-carrying extra
        // would fit a REDUCED model and report it as a normal success. Callers must
        // route such a design to the sparse solver — `classify_design`'s
        // `slope_extras` clause does. Checked here, once per workspace build, rather
        // than in `apply_lambda`/`build_packed_m`, whose per-eval `debug_assert`s
        // stay debug-only because they sit in the hot loop.
        assert!(
            !groupings.extra_slopes_any,
            "dense GLMM kernel cannot fit a slope-carrying extra grouping — route it to the sparse solver"
        );
        let k = groupings.k_total;
        let n_theta = groupings.n_theta();
        let q = groupings.primary_q;
        let n_primary = groupings.n_primary;
        // Structured-path block sizes: core width q_core = q_p + nested children,
        // crossed width e. Buffers stay 1-sized minima when the shape has no
        // extras (the no-extras blocked path never touches them).
        let q_core = q + groupings.nested_per_parent;
        let e_crossed = groupings.k_crossed();

        // Which of `laplace_deviance`'s three routes this shape takes is fixed by `groupings`
        // alone (deviance.rs:194) — decided here so the n×k buffers exist only on the route
        // that reads them. Sized 0×0 (not `.max(1)`) on the routes that don't: a missed read
        // must panic on the bounds check, not silently return a zero.
        let has_extras = !groupings.extra_offsets.is_empty();
        let needs_dense = has_extras && !groupings.structured_extras_eligible();

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
        let mut config = Config::new(n_theta + p);
        config.rho_begin = rho_begin;
        config.rho_end = GLMM_RHO_END;
        crate::lmm::apply_campaign_overrides(&mut config, n_theta + p);
        // Stage-1 θ-only BOBYQA config: same rho_begin/rho_end schedule as the
        // joint solver above, but `npt` mirrors `sparse_lmm_seed`'s mid-model
        // rule (`src/lmm/mod.rs`), NOT the joint solver's — the two are sized for
        // different-dimension searches and this crate's precedent for a
        // θ-only search is `sparse_lmm_seed`. MIRRORS `config1` in
        // `fit_glmm_sparse` (sparse.rs) — change together. Both feed through
        // the shared `apply_campaign_overrides` tail.
        let npt_stage1 = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        let mut config_stage1 = Config::new(n_theta);
        config_stage1.rho_begin = rho_begin;
        config_stage1.rho_end = GLMM_RHO_END;
        config_stage1.npt = npt_stage1;
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
            z: if needs_dense {
                Mat::zeros(max_n, k.max(1))
            } else {
                Mat::zeros(0, 0)
            },
            m: if needs_dense {
                Mat::zeros(max_n, k.max(1))
            } else {
                Mat::zeros(0, 0)
            },
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
            // dense NoZ; `validation/` datasets with a sparse/LMM path are unaffected by
            // this field and were confirmed identical across both sweep arms) — the
            // full corpus, not a hand-picked dataset list: an earlier hand-picked list
            // missed one loser (Arabidopsis, see below). Protocol: per arm (forced skip
            // vs forced keep), two independent
            // `validation_fit` invocations, each itself the median of 9 timed samples
            // after a discarded warmup (see validation/engines/glmm.rs); invocations agreed
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
            a: if needs_dense {
                Mat::zeros(k.max(1), k.max(1))
            } else {
                Mat::zeros(0, 0)
            },
            a_chol: if needs_dense {
                Mat::zeros(k.max(1), k.max(1))
            } else {
                Mat::zeros(0, 0)
            },
            a_llt_mem: MemBuffer::new(cholesky_in_place_scratch::<f64>(
                if needs_dense { k.max(1) } else { 1 },
                Par::Seq,
                Spec::default(),
            )),
            wm: if needs_dense {
                Mat::zeros(max_n, k.max(1))
            } else {
                Mat::zeros(0, 0)
            },
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
            offset: None,
            pirls_exhausted: 0,
            final_pirls_exhausted: false,
            counters: crate::counters::EvalCounters::new(),
        }
    }

    /// Test-only: materialize the six dense buffers (`z`, `m`, `wm`, `a`, `a_chol`,
    /// `a_llt_mem`) at full size regardless of the route `from_groupings` picked.
    /// Some tests deliberately drive the dense kernel (`apply_lambda`/`pirls_solve`)
    /// directly against a workspace built for a blocked or structured shape, to
    /// assert the fast path agrees with the dense one — that assertion needs the
    /// dense buffers to exist even though production never allocates them off the
    /// dense route. `max_n` is read back from `eta` (always allocated at full size,
    /// on every route).
    #[cfg(test)]
    pub(crate) fn ensure_dense_buffers(&mut self) {
        let max_n = self.eta.len();
        let k = self.k.max(1);
        self.z = Mat::zeros(max_n, k);
        self.m = Mat::zeros(max_n, k);
        self.wm = Mat::zeros(max_n, k);
        self.a = Mat::zeros(k, k);
        self.a_chol = Mat::zeros(k, k);
        self.a_llt_mem = MemBuffer::new(cholesky_in_place_scratch::<f64>(
            k,
            Par::Seq,
            Spec::default(),
        ));
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
    w.offset = src.offset.clone();
    w
}

// Kernel — written as borrow-split FREE fns so the BOBYQA closure in fit_glmm can call
// them on destructured workspace fields without re-borrowing the whole workspace.

/// In-place lower Crout Cholesky of a `q×q` block stored row-major in `blk`
/// (lower triangle read; on return the lower triangle holds L). Returns false on
/// a non-positive pivot — the module's failure surface. q ≤ MAX_PRIMARY_Q (8).
/// Thin wrapper over the shared kernel in `crate::linalg::block_chol`.
pub(crate) fn glmm_block_chol<T: crate::scalar::Scalar>(blk: &mut [T], q: usize) -> bool {
    crate::linalg::block_chol(blk, q)
}

/// Solve `L Lᵀ x = b` in place (`b` overwritten with `x`) for the `q×q` lower
/// factor `l` produced by `glmm_block_chol` (row-major, diagonal = L pivots).
/// Forward `L y = b` then back `Lᵀ x = y`.
pub(crate) fn glmm_block_solve<T: crate::scalar::Scalar>(l: &[T], q: usize, b: &mut [T]) {
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
pub(crate) fn glmm_block_solve_panel<T: crate::scalar::Scalar>(
    l: &[T],
    q: usize,
    panel: &mut [T],
    nc: usize,
) {
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
/// Builds the GLMM design-`Z` (`ws.z`) for the dense-fallback path only; the
/// block-diagonal (no-extras) and structured fits both reconstruct `mᵢ` per
/// row from the ids instead and never read `ws.z` (0×0 there, so this returns
/// immediately).
pub fn build_z(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    cluster_ids: &[u32],
    extra_ids: &[Vec<u32>],
    n: usize,
) {
    // Sized 0×0 by `from_groupings` on the no-extras blocked route (nothing ever
    // reads it there) — the constructor is the single place that decides this,
    // so a route change there is all this early return needs to track.
    if ws.z.ncols() == 0 {
        return;
    }
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
            // Read AS A RANDOM-EFFECT column, so it takes the internal scale
            // (`LmmGroupings::set_slope_scales`); the same x column keeps its raw
            // value everywhere the fixed-effect design reads it. Mirrored by
            // `fill_z_f64` and by the Rx M row in `se::blocked_schur_fill` —
            // change together.
            ws.z[(i, base + 1 + d)] = x[(i, g.primary_slope_cols[d])] / g.primary_slope_scales[d];
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

/// Per-fit hoist of the primary-slope Z columns: `z_buf[i·(q−1)+d] =
/// x[i, slope_cols[d]] / s_d`. θ/β change per BOBYQA eval but `x` and the scales
/// are fixed per fit, so this lifts the MatRef load and the scale division out of
/// the per-solve M fill — the fill becomes a pure contiguous-f64 product. `s_d`
/// is the RE column's internal scale (`LmmGroupings::set_slope_scales`); mirrored
/// by `build_z` and by the Rx M row in `se::blocked_schur_fill` — change
/// together. No-op at q_p = 1 (no slope columns).
pub(crate) fn fill_z_f64(g: &LmmGroupings, x: MatRef<f64>, z_buf: &mut [f64], n: usize) {
    let q = g.primary_q;
    for i in 0..n {
        for d in 0..q - 1 {
            z_buf[i * (q - 1) + d] = x[(i, g.primary_slope_cols[d])] / g.primary_slope_scales[d];
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
/// `M[(i, core_col(f,local))]` for row `i`'s primary cluster (primary `local<q`:
/// `Σ_{r≥local} z_r·lam[r·q+local]`, mirroring `apply_lambda`'s core write and
/// the blocked-path fill; nested `local≥q`: the nested indicator scaled by its
/// θ). For each crossed grouping with `θ≠0`, the row's single active level
/// contributes one nonzero: `cross_val = z·θ`, `cross_col = b` (the crossed
/// block-local index, `0..e`), with `n_cross[i]` the count (`≤ G`). A θ-pinned
/// (θ=0) grouping is skipped, mirroring `apply_lambda`'s `z·θ=0` ⇒ no nonzero.
/// Packs M's nonzeros from the grouping ids and `z_buf`; the dense `z` is
/// never materialized on this route (it is 0×0 there — see
/// `GlmmWorkspace::z`'s doc). Every value it used to read
/// back out of `z` is reconstructed straight from what `build_z` would have
/// written there: the primary core from `z_buf` (the pre-widened slope buffer
/// `fill_z_f64` fills, same source `build_z` widens from `x`), the nested
/// indicator and the crossed level from `extra_ids` (the same slice `build_z`
/// takes) directly — no scan needed for either. Reads `z_buf`, `extra_ids`,
/// `cluster_ids` (only for the nested global→local id conversion — see below),
/// `lam` (filled here via `primary_lambda`), and `params`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_packed_m(
    g: &LmmGroupings,
    params: &[f64],
    z_buf: &[f64],
    extra_ids: &[Vec<u32>],
    lam: &mut [f64],
    cluster_ids: &[u32],
    m_core_buf: &mut [f64],
    cross_val: &mut [f64],
    cross_col: &mut [u32],
    n_cross: &mut [u8],
    n: usize,
) {
    let q = g.primary_q;
    let np = g.nested_per_parent;
    let qc = q + np;
    let k_family = qc * g.n_primary;
    let base_theta = q * (q + 1) / 2;
    let g_cap = crate::lmm::MAX_EXTRA_GROUPINGS;
    // Intercept-only extras on the GLMM structured path (see `apply_lambda`;
    // `classify_design` routes extra-slopes shapes to Sparse for every family).
    debug_assert!(!g.extra_slopes_any);
    crate::lmm::primary_lambda(&params[..g.n_theta()], q, lam);
    let theta_nested = g.nested.map(|nf| params[nf.vech_start]).unwrap_or(0.0);
    // Declaration index into `extra_offsets`/`extra_ids` for the nested factor;
    // `None` when there is no nested grouping (`np == 0`, the loop below never
    // reads it then).
    let nested_decl = g.nested.map(|nf| nf.vech_start - base_theta);
    for i in 0..n {
        let f = cluster_ids[i] as usize;
        // Core primary block: the identical `Σ_{r≥c} z_r·lam[r·q+c]` reduction
        // that `glmm/pirls/blocked.rs`'s `pirls_solve_blocked` per-solve M fill runs
        // (whose own comment records it as bit-identical to the z-sourced form) — z_r is
        // 1.0 at r==0 (the intercept `build_z` always writes) or the
        // pre-widened slope value `z_buf[i·(q−1)+(r−1)]` otherwise (the same
        // `x` column `build_z` would have widened into the row's `z` block).
        for c in 0..q {
            let mut acc = 0.0;
            for r in c..q {
                let zr = if r == 0 {
                    1.0
                } else {
                    z_buf[i * (q - 1) + (r - 1)]
                };
                acc += zr * lam[r * q + c];
            }
            m_core_buf[i * qc + c] = acc;
        }
        // Core nested children of parent f: `extra_ids` stores the nested id
        // GLOBAL (dense over all parents — see `GroupIds`'s doc), while the
        // packed core slots are LOCAL to this row's own parent block, so the
        // global id needs `f`'s own `f·np` prefix subtracted back off before
        // it can be compared against the local `j`. `build_z` wrote a 1.0
        // indicator at the row's own (global) nested id and 0.0 at every other
        // slot, so the packed value is θ_nested at that one local `j` and
        // `0.0 · theta_nested` everywhere else — kept as the multiply, not a
        // bare `0.0` literal, so a transient negative θ (an FD-Hessian
        // perturbation step, or an off-optimum BOBYQA trial — θ is only
        // guaranteed ≥ 0 at the pinned post-fit point, not at every eval)
        // still lands on the same `-0.0` the z-sourced form would have.
        if np > 0 {
            let global_id = extra_ids[nested_decl.expect("np > 0 ⇒ g.nested is Some")][i] as usize;
            // Before the ids-based rewrite, a malformed id panicked on the z
            // bounds check in every build profile; these two asserts keep at
            // least the debug-profile tripwire now that `z` is never read here.
            debug_assert!(
                global_id >= f * np,
                "row {i}: nested global id {global_id} underflows parent {f}'s block (f*np={})",
                f * np
            );
            let local_id = global_id - f * np;
            debug_assert!(
                local_id < np,
                "row {i}: nested local id {local_id} out of range (np={np})"
            );
            for j in 0..np {
                m_core_buf[i * qc + q + j] = if j == local_id {
                    theta_nested
                } else {
                    0.0 * theta_nested
                };
            }
        }
        // Crossed: one nonzero per crossed grouping (its single active level), θ-pinned
        // groupings skipped. `build_z` wrote that level's one 1.0 at column
        // `off + extra_ids[e][i]`, so the former linear scan for the first
        // nonzero collapses to a direct index — no scan, no `z` read.
        let mut cnt = 0usize;
        for cf in &g.crossed {
            let theta = params[cf.vech_start];
            if theta == 0.0 {
                continue;
            }
            // q_g==1 here (see debug_assert above), so vech_start − base_theta is
            // this factor's declaration index into extra_offsets/extra_ids.
            let e = cf.vech_start - base_theta;
            // Same tripwire as the nested branch above: before the ids-based
            // rewrite a malformed id panicked on the z bounds check in every
            // build profile; this keeps at least the debug-profile check.
            debug_assert!(
                (extra_ids[e][i] as usize) < cf.n_levels,
                "row {i}: crossed id {} out of range (n_levels={})",
                extra_ids[e][i],
                cf.n_levels
            );
            let off = g.extra_offsets[e];
            let col = off + extra_ids[e][i] as usize;
            cross_col[i * g_cap + cnt] = (col - k_family) as u32;
            cross_val[i * g_cap + cnt] = theta; // z·θ where z ≡ 1.0 exactly
            cnt += 1;
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
