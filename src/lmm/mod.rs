//! General-machine LMM solver core — family-blocked profiled-REML deviance
//! (primary + nested children eliminated family-by-family, crossed factors in
//! a dense tail Cholesky with [X y]) + BOBYQA θ-search over one diagonal θ
//! component per grouping. A degenerate single-intercept `ClusterSpec`
//! collapses to the per-cluster shrink-downdate arithmetic (up to FP
//! reassociation), so the q=1 validation corpus re-proves on this machine.
//!
//! Engine-resident: ALL Gaussian mixed (LMM) specs dispatch here through the
//! unified fit core — the single-random-intercept shape is an `LmmDense`/BOBYQA
//! case like any other, from every tier (stable and loop). A scalar-Brent
//! kernel for this shape was retired: its measured 3× win on that shape was
//! an allocation artifact — it reused preallocated scratch while
//! the old dispatch rebuilt a workspace per call, and handing BOBYQA the true θ
//! changed BOBYQA's runtime by under 6%, so the θ search was never the cost.
//! The reusable `FitWorkspace` captures that win for every shape. Brent is also
//! strictly worse at high cluster counts (its `O(n_clusters·P)` per-evaluation
//! downdate has no counterpart in `reml_deviance`'s balanced-collapse path), and
//! the two agree on β̂ to ~1e-9, so there was no accuracy axis to trade either.
//!
//! Hot-loop invariants (carried over from the retired `lme.rs`):
//!  * Bounded allocations on the warm path (twin test in `lmm::tests`): all
//!    scratch and the BOBYQA solver live in `LmmWorkspace`, allocated once
//!    per (p, max_clusters) shape; the only per-call allocations are faer
//!    `llt` internals — the same acceptance the shipped path carries.
//!  * Inference is squared statistics (`t_sq = β̂²/Var(β̂)`); never sqrt the
//!    SE, never call a CDF on the per-fit path.
//!  * `f64::INFINITY` is the deviance failure surface.
//!
//! BOBYQA is Powell, M.J.D. (2009), *The BOBYQA algorithm for bound constrained
//! optimization without derivatives*, Cambridge report DAMTP 2009/NA06.

use bobyqa::{Bobyqa, Config, RestartConfig, Status};
use faer::reborrow::ReborrowMut;
use faer::{Mat, MatMut, MatRef};
use std::sync::OnceLock;

use crate::scalar::Scalar;
use crate::FLOAT_NEAR_ZERO;

mod kernel;
#[cfg(test)]
mod tests;

pub(crate) use kernel::precompute_balanced_collapse;
pub use kernel::{reml_deviance, LmmSuffStats};
pub(crate) use kernel::{reml_gradient, reml_hessian, LmmDualScratch, LmmHyperScratch};

/// θ start — DIAGONAL vech entries only; off-diagonals cold-start at 0
/// (unit diagonal, the lme4/MixedModels.jl default — the
/// `blind_theta_and_bounds` shape). Cold start per fit; no warm-start
/// across sims (would re-import cross-grid-point path dependence).
pub const THETA0: f64 = 1.0;
/// Per-component θ upper box — mirrors the retired scalar Brent kernel's reach (θ ≤ 1e3).
pub const THETA_HI: f64 = 1e3;
/// Initial trust radius. Must be ≤ 1.0: PRIMA start-projection silently moves
/// an x₀ within rho_begin of a bound; 0.5 keeps θ₀ = 1.0 strictly clear of
/// the 0 lower bound. Box width 1e3 ≥ 2·rho_begin is the crate's up-front
/// validity requirement.
pub const RHO_BEGIN: f64 = 0.5;
/// Final trust radius = θ̂ target accuracy. 1e-6 measured equivalent to 1e-8
/// on every validation check under the amended abs floors (stat 1e-4 /
/// β̂ 1e-5), at 15.1–15.7 vs 19.5–20.7 evals/fit — a ~25% eval cut for free.
pub const RHO_END: f64 = 1e-6;
/// GLMM-path final trust radius (θ AND β jointly, not θ-only like `RHO_END`).
/// Re-measured 2026-08-06 (full `cargo test --release` across all five
/// feature configs plus a 20-rung validation sweep with tightening-only noise
/// floors): 3e-6 fully green; 1e-5 fails 5 tests, the one oracle-band
/// casualty being `fit_glmm_poisson_agq_matches_lme4` — grouseticks AGQ k=7
/// β[0] at 1.9331e-3 vs its 1e-3 lme4 band (1.93×, and ~9× the rung's
/// 1.03e-4 optimizer noise floor, so a real accuracy loss, not path wander);
/// the other four are Rust-vs-Rust bit-exact pins. The 2026-07-03 sweep's
/// 1e-5 casualty (`two_stage_matches_single_stage_on_grouseticks`) now
/// passes at 1e-5; that sweep also saw 3e-5 and 1e-4 add further oracle
/// failures. 3e-6 is the boundary-adjacent candidate — do NOT loosen past
/// it. Original timing rationale (vs 1e-6): 0.265–0.269s vs 0.288s on
/// `grouseticks` (n=403, 7 free params), a ~7–8% wall-time cut for free.
pub const GLMM_RHO_END: f64 = 3e-6;
/// Truth-start floor: a `Some(θ₀)` start is clamped to max(θ₀, this) so a
/// zero/near-zero true θ never starts the search on the boundary itself.
/// Keep ≥ 10·RHO_END: the future scaled schedule derives
/// rho_begin = 0.1·θ₀, and the crate requires rho_end ≤ rho_begin.
pub const THETA_TRUTH_FLOOR: f64 = 0.01;
/// Pin threshold: a Converged diagonal component ≤ this is deterministically
/// pinned at exactly 0 and counted converged. 1e-4 aligns the class boundary
/// with the shipped τ̂≈0 detection (the retired `lme.rs` pinned boundary_hit=1
/// fits at θ = 1e-4).
///
/// **Tested on the INTERNAL θ** — the scaled coordinate the solver minimizes over
/// ([`LmmGroupings::set_slope_scales`]), not on θ in the design's own units.
/// Internal θ is what the optimizer's own stopping geometry lives on, and it is
/// the only version of the test that does not change its verdict when a
/// random-slope column is re-expressed in different units. The cost is that a
/// badly scaled design can be flagged differently from lme4's `isSingular`,
/// which applies the same 1e-4 to user-scale θ.
pub const PIN_THETA: f64 = 1e-4;
/// Ill-conditioning DETECTION floor for the dense-LMM route, on the
/// scale-invariant per-column pivot ratio of X'V⁻¹X at θ̂
/// ([`crate::ols::min_pivot_ratio`]). Below it the fixed-effect coefficients are
/// barely identified and the diagnostics channel says so — it is **not** a
/// refuse threshold, and this route refuses no design on conditioning grounds
/// (see the guard site for the two measured reasons).
///
/// Calibrated 2026-07-31 against a 1-ULP perturbation sweep of `y`. `1e-12` sits
/// about two decades above where the statistic stops tracking conditioning on
/// this route, which is the range where it still ranks designs. It is
/// conservative: β̂'s measured movement here is 9.1e-8 at pivot 9.7e-13, far
/// steadier than the `1e-15 / pivot` law the OLS and GLM routes obey.
///
/// No SOLVER path reads it: the kernel records the raw pivot and the comparison
/// happens once, in `LmmResultView::diagnostics`, which is what fills
/// `FitDiagnostics::ill_conditioned`.
pub const PIVOT_MIN: f64 = 1e-12;

/// BOBYQA config for an n_theta-dimensional θ-search. `Config::new` supplies
/// the PRIMA defaults (npt = 2n+1, max_fun = 500·n) — at n = 1 exactly
/// npt = 3 / max_fun = 500. Test-only (reached only via the test-gated
/// [`LmmWorkspace::with_groupings`]); the live path inlines its own config.
#[cfg(test)]
pub fn bobyqa_config(n_theta: usize) -> Config {
    let mut config = Config::new(n_theta);
    config.rho_begin = RHO_BEGIN;
    config.rho_end = RHO_END;
    apply_campaign_overrides(&mut config, n_theta);
    config
}

/// Dev-only env hooks for sweeping BOBYQA's npt/max_fun without recompiling
/// (npt and max_fun share this seam).
/// `LMM_NPT_FORMULA=<mult>n<add>` overrides npt at EVERY BOBYQA config site:
/// dense LMM (`for_cluster_spec_ext`), the blind seed (`bobyqa_config`), the
/// sparse θ seed (`sparse_lmm_seed`), the sparse GLMM stage-1 + joint configs
/// (`sparse.rs`), and the GLMM joint + stage-1 solvers (`glmm/workspace.rs`)
/// — each evaluated against that solver's own dimension (joint: n_theta + p).
/// `LMM_MAX_FUN_FORMULA` does the same for max_fun (unclamped). Formula-shaped
/// values only: a flat constant would violate `n+2 ≤ npt ≤ (n+1)(n+2)/2` at
/// small n once n changes between call sites, so flat inputs parse to None
/// and the shipped value stays instead of silently producing an illegal npt.
/// Read once per process (OnceLock) — sweeps set them per run via env var.
pub(crate) fn eval_formula(formula: &str, n: usize) -> Option<usize> {
    let (mult, add) = formula.split_once('n')?;
    let mult: f64 = mult.parse().ok()?;
    let add: usize = add.parse().ok()?;
    Some((mult * n as f64).ceil() as usize + add)
}

pub(crate) fn npt_from_formula(formula: &str, n: usize) -> Option<usize> {
    eval_formula(formula, n).map(|v| v.clamp(n + 2, (n + 1) * (n + 2) / 2))
}

fn env_formula(var: &'static str, cell: &'static OnceLock<Option<String>>) -> Option<String> {
    cell.get_or_init(|| std::env::var(var).ok().filter(|s| !s.is_empty()))
        .clone()
}

pub(crate) fn npt_override(n: usize) -> Option<usize> {
    static V: OnceLock<Option<String>> = OnceLock::new();
    npt_from_formula(&env_formula("LMM_NPT_FORMULA", &V)?, n)
}

pub(crate) fn max_fun_override(n: usize) -> Option<usize> {
    static V: OnceLock<Option<String>> = OnceLock::new();
    eval_formula(&env_formula("LMM_MAX_FUN_FORMULA", &V)?, n)
}

/// Shared tail for all BOBYQA config sites: the dev-only campaign env hooks
/// (no-op when unset) and the shipped restart schedule (always on).
///
/// The restart re-solves from the incumbent θ with a discarded interpolation
/// model once a cycle has spent an eighth of the evaluation budget that
/// remained when it started. Only fits that cross that cap can move — with
/// `cycle_budget_frac` non-zero, reaching `rho_end` no longer restarts on its
/// own, so a solve converging inside its cap returns exactly what no-restart
/// returns, evaluation count included. Note the trigger is the per-cycle cap,
/// not `max_fun` exhaustion: a fit can restart with most of `max_fun` unspent.
/// BOBYQA keeps the best point across cycles, so a restart never returns worse
/// than stopping would have.
///
/// The four fields are written out even though they are `RestartConfig::new()`'s
/// own defaults: the schedule is adopted by value, so a later bobyqa release
/// changing its defaults cannot silently move this fit path.
pub(crate) fn apply_campaign_overrides(config: &mut Config, n: usize) {
    if let Some(npt) = npt_override(n) {
        config.npt = npt;
    }
    if let Some(mf) = max_fun_override(n) {
        config.max_fun = mf.max(config.npt + 1);
    }
    let mut restart = RestartConfig::new();
    restart.cycle_budget_frac = 0.125;
    restart.max_restarts = 1;
    restart.improve_rel_tol = 1e-6;
    // Off: the eval cap dominates it in every measured case, and it cannot see
    // a cycle that crawls without reducing rho.
    restart.stall_reductions = 0;
    config.restart = Some(restart);
}

fn two_stage_enabled() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("LMM_TWO_STAGE").is_ok_and(|v| v == "1"))
}

/// Experimental two-stage warm restart, gated behind `LMM_TWO_STAGE=1`. Stage 1:
/// cheapest legal model (npt = n+2), loose rho_end 1e-3 — reach the basin.
/// Stage 2: fresh solver (BOBYQA cannot grow npt mid-run — the inverse-KKT
/// update assumes a constant set size), npt = 2n+1, shipped RHO_END, rho_begin
/// shrunk to the local scale (0.1·min diagonal θ₁, clamped to
/// [10·RHO_END, RHO_BEGIN]). Returns a merged Outcome: stage-2 status/point,
/// summed n_eval. LMM_STAGE_PROBE=1 prints per-stage evals + the θ₁→θ̂ distance,
/// for judging whether a third stage would pay for itself. Dev seam only —
/// allocates two solvers per fit, which the shipped path never does.
///
/// Deliberately left uncounted (`EvalCounters` stays empty on this path): it
/// is not the shipped path, and `LMM_STAGE_PROBE=1` above already prints its
/// per-stage evals.
fn two_stage_minimize(
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch,
    theta: &mut [f64],
    lower: &[f64],
    upper: &[f64],
) -> bobyqa::Outcome {
    let n = theta.len();
    let c1 = {
        let mut c = Config::new(n);
        c.npt = n + 2;
        c.rho_begin = RHO_BEGIN;
        c.rho_end = 1e-3;
        c
    };
    let mut s1 = Bobyqa::new(n, c1).expect("stage-1 config valid");
    let out1 = s1.minimize(|xs| reml_deviance(xs, suff, fit), theta, lower, upper);
    let theta1 = theta.to_vec();

    let min_diag = suff
        .groupings
        .diagonal_theta()
        .iter()
        .map(|&i| theta[i])
        .fold(f64::INFINITY, f64::min);
    let rho_begin2 = (0.1 * min_diag).clamp(10.0 * RHO_END, RHO_BEGIN);
    let c2 = {
        let mut c = Config::new(n);
        c.npt = 2 * n + 1;
        c.rho_begin = rho_begin2;
        c.rho_end = RHO_END;
        c
    };
    let mut s2 = Bobyqa::new(n, c2).expect("stage-2 config valid");
    let out2 = s2.minimize(|xs| reml_deviance(xs, suff, fit), theta, lower, upper);

    if std::env::var("LMM_STAGE_PROBE").is_ok_and(|v| v == "1") {
        let dist = theta1
            .iter()
            .zip(theta.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        eprintln!(
            "stage_evals={},{} stage1_dist={dist:.6e}",
            out1.n_eval, out2.n_eval
        );
    }
    bobyqa::Outcome {
        n_eval: out1.n_eval + out2.n_eval,
        ..out2
    }
}

/// Topology-only BOBYQA solver + blind θ₀ + per-component boxes for the sparse-Z
/// path (`sparse::fit_mle_sparse`), byte-identical to what
/// `LmmWorkspace::for_cluster_spec_ext` seeds — but WITHOUT the dense O(K)
/// suff-stats / fit scratch the sparse path exists to avoid. The θ schedule
/// (scaled `rho_begin`, mid `npt`) and the blind seed/bounds are TOPOLOGY-ONLY
/// (functions of `n_theta`/`diagonal_theta`, not of K), so an in-envelope design
/// fit through here matches the NoZ path to machine precision (the superset
/// property).
///
/// MIRRORS the config/seed in `LmmWorkspace::for_cluster_spec_ext` — change
/// together (both feed through the shared `apply_campaign_overrides` tail).
pub(crate) fn sparse_lmm_seed(groupings: &LmmGroupings) -> (Bobyqa, Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_theta = groupings.n_theta();
    let blind_theta = vec![THETA0; n_theta];
    let rho_begin = (0.1
        * groupings
            .diagonal_theta()
            .iter()
            .map(|&i| blind_theta[i])
            .fold(f64::INFINITY, f64::min))
    .min(RHO_BEGIN);
    let npt = if n_theta >= 3 {
        (3 * n_theta).div_ceil(2) + 1
    } else {
        2 * n_theta + 1
    };
    let mut config = Config::new(n_theta);
    config.rho_begin = rho_begin;
    config.rho_end = RHO_END;
    config.npt = npt;
    apply_campaign_overrides(&mut config, n_theta);
    let (theta, lower, upper) = groupings.blind_theta_and_bounds();
    let solver =
        Bobyqa::new(n_theta, config).expect("BOBYQA config constants are valid by construction");
    (solver, theta, lower, upper)
}

/// Capacity ceilings — single-sourced in `crate::consts` and re-exported here
/// as `crate::lmm::MAX_*` so callers can validate a spec against them before
/// a fit is ever built.
pub use crate::consts::{MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q, MAX_PRIMARY_Q};

// ---------------------------------------------------------------------------
// LmmGroupings — grouping-structure metadata shared by suff stats + deviance.
// ---------------------------------------------------------------------------

/// A crossed extra grouping factor's θ-layout descriptor. `vech_start` is the
/// θ index where its `vech(Λ_g)` block begins (`q·(q+1)/2` entries, column-major
/// lower-tri); `q = 1 + #slopes` is its RE width; `n_levels` is its crossed level
/// count. For `q == 1` (intercept only) `vech_start` is the old scalar θ index and
/// the block is a single variance — byte-identical to the pre-slope layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossedFactor {
    pub vech_start: usize,
    pub q: usize,
    pub n_levels: usize,
    /// Declaration index among the extra groupings — indexes `extra_offsets` /
    /// `extra_q` / `extra_slope_cols`.
    pub decl: usize,
}

/// A nested extra grouping factor's θ-layout descriptor (child RE columns sit in
/// the family block). `vech_start`/`q` as in [`CrossedFactor`]; the level count is
/// `n_primary · nested_per_parent`, derived from the primary, so it is not stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NestedFactor {
    pub vech_start: usize,
    pub q: usize,
    /// Declaration index among the extra groupings.
    pub decl: usize,
}

/// Grouping-structure metadata the suff stats and deviance share.
///
/// RE column order is the ELIMINATION order — `[primary 0..S | nested
/// children (parent-contiguous: child id = parent·n_per + within) | crossed
/// factors last]` — decoupled from θ order, which stays `[primary, extras in
/// declaration order]` (matching data-gen draw order and the truth-start
/// vector).
#[derive(Clone)]
pub struct LmmGroupings {
    /// Primary level capacity at the sized max_n.
    pub n_primary: usize,
    /// Children per parent; 0 = no nested extra (family width 1).
    pub nested_per_parent: usize,
    /// The nested extra's θ-layout descriptor, if any.
    pub nested: Option<NestedFactor>,
    /// Crossed extras in declaration order.
    pub crossed: Vec<CrossedFactor>,
    /// RE-column offset of extra g (declaration order) — where its globalized
    /// level ids land in `s`/`counts`.
    pub extra_offsets: Vec<usize>,
    /// Total RE columns K.
    pub k_total: usize,
    /// Primary RE block width `q_p = 1 + #slopes` (1 = intercept only).
    pub primary_q: usize,
    /// `[X y]` row indices of the slope covariates (their x_full design columns),
    /// one per slope in declaration order, used to recover the per-level `q_p×q_p`
    /// Gram from `s`. Empty iff `primary_q == 1`.
    pub primary_slope_cols: Vec<usize>,
    /// θ-indices of the diagonal variance components (pinnable), in
    /// `boundary_rate_per_component` order: the `q_p` primary vech diagonals
    /// (column-major — the diagonal of column d sits at offset `Σ_{j<d}(q_p−j)`)
    /// then the extra-grouping scalars. q_p=1 ⇒ `[0, extras…]`; q_p=2 ⇒
    /// `[0, 2, extras…]`; off-diagonal vech entries excluded. Computed once per
    /// workspace by `compute_diagonal_theta` (the vech-diagonal walk lives there,
    /// single-sourced) so the per-fit pin loop borrows instead of reallocating.
    pub diagonal_theta: Vec<usize>,
    /// Solver-path gate: true iff any extra grouping carries a random slope
    /// (`q_g > 1`). The scalar tail in `reml_deviance` stays byte-identical when
    /// false; the blocked per-factor `Λ_g` tail is taken when true.
    pub extra_slopes_any: bool,
    /// Per-extra-grouping RE width `q_g = 1 + #slopes`, DECLARATION order (parallel
    /// to `extra_offsets`). Sizes each grouping's RE-column block (`level·q_g + d`)
    /// and its `vech(Λ_g)`. All `1` for intercept-only extras (pre-slope shape).
    pub extra_q: Vec<usize>,
    /// Per-extra-grouping `[X y]` row indices of that grouping's slope covariates
    /// (resolved x columns), DECLARATION order — the crossed/nested analogue of
    /// `primary_slope_cols`. `extra_slope_cols[e]` has `extra_q[e] − 1` entries.
    /// Empty (or empty inner) for intercept-only extras. Used by `add_rows_multi`
    /// (covariate-weighted scatter) and the blocked tail Gram recovery.
    pub extra_slope_cols: Vec<Vec<usize>>,
    /// Internal scale `s_d` of each primary random-slope design column, parallel
    /// to `primary_slope_cols`. Every site that reads that column as a
    /// RANDOM-EFFECT covariate divides by it; the fixed-effect design keeps the
    /// raw column. See [`LmmGroupings::set_slope_scales`] for the identity and
    /// [`rms_column_scale`] for the statistic. All `1.0` until `set_slope_scales`
    /// runs, and `1.0` divides exactly, so an unset grouping takes the unscaled
    /// arithmetic bit-for-bit.
    pub primary_slope_scales: Vec<f64>,
    /// Per-extra-grouping twin of `primary_slope_scales`, parallel to
    /// `extra_slope_cols` (declaration order).
    pub extra_slope_scales: Vec<Vec<f64>>,
}

/// Weighted root-mean-square of design column `col`: `√(Σᵢ wᵢ xᵢ² / Σᵢ wᵢ)`.
///
/// This is the internal scale a random-effect design column is divided by. It is
/// the weighted second moment about ZERO, not a centered sd, for one reason: it
/// returns EXACTLY `1.0` on a constant-1 column under any weights, because
/// `Σw/Σw` is `1.0` in IEEE-754 and `√1 = 1`. That exactness is what makes an
/// implicit intercept subcolumn's factor exact and keeps an intercept-only
/// design bit-identical to the unscaled path. A centered sd returns `0` there
/// and would need a special case.
///
/// A degenerate column — zero total weight, zero weighted second moment, or a
/// non-finite one — returns `1.0`: the scaling map has to stay invertible, and a
/// column with no spread has no conditioning to fix.
pub fn rms_column_scale(x: MatRef<'_, f64>, col: usize, weights: Option<&[f64]>) -> f64 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..x.nrows() {
        let w = weights.map_or(1.0, |w| w[i]);
        let v = x[(i, col)];
        num += w * v * v;
        den += w;
    }
    if den <= 0.0 {
        return 1.0;
    }
    let s = (num / den).sqrt();
    if s.is_finite() && s > 0.0 {
        s
    } else {
        1.0
    }
}

/// The vech-diagonal θ-index walk, single source of truth for
/// `LmmGroupings::diagonal_theta`. Each factor (the primary, then every extra in
/// declaration order) contributes its `q` diagonal entries (column-major: the
/// diagonal of column d sits at offset `Σ_{j<d}(q−j)` within the factor's vech
/// block), and the next factor's block starts `q(q+1)/2` θ-slots later. With
/// every extra `q == 1` this is `[primary diagonals, base+0, base+1, …]` — the
/// pre-slope layout.
fn compute_diagonal_theta(primary_q: usize, extra_qs: &[usize]) -> Vec<usize> {
    let mut idx = Vec::with_capacity(primary_q + extra_qs.len());
    let mut start = 0usize;
    for &q in std::iter::once(&primary_q).chain(extra_qs.iter()) {
        let mut off = start;
        for d in 0..q {
            idx.push(off);
            off += q - d; // advance past column d's vech block (length q−d)
        }
        start += q * (q + 1) / 2; // next factor's vech block
    }
    idx
}

impl LmmGroupings {
    /// Single q=1 grouping shape.
    pub fn single(max_clusters: usize) -> Self {
        LmmGroupings {
            n_primary: max_clusters,
            nested_per_parent: 0,
            nested: None,
            crossed: vec![],
            extra_offsets: vec![],
            k_total: max_clusters,
            primary_q: 1,
            primary_slope_cols: vec![],
            diagonal_theta: compute_diagonal_theta(1, &[]), // [0]
            extra_slopes_any: false,
            extra_q: vec![],
            extra_slope_cols: vec![],
            primary_slope_scales: vec![],
            extra_slope_scales: vec![],
        }
    }

    /// Structure for a (validated) ClusterSpec at workspace size max_n.
    /// validate() guarantees ≤ 1 nested entry and crossed ⇒ FixedClusters.
    /// `slope_cols` are the x_full column indices for the primary slopes
    /// (`spec.cluster_slope_design_cols` as usize); pass `&[]` for callers without slopes.
    pub fn from_cluster_spec(
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
    ) -> Self {
        // Intercept-only extras (or layout-only callers): no extra-slope x-cols.
        Self::from_cluster_spec_ext(cluster, max_n, slope_cols, &[])
    }

    /// As [`from_cluster_spec`], plus the resolved `[X y]` column indices of each
    /// extra grouping's slope covariates (`extra_slope_cols[e]`, declaration order;
    /// pass `&[]` for intercept-only extras). The RE *widths* still come from the
    /// `ModelSpec` (`1 + gs.slopes.len()`); these only supply the x-columns the
    /// suff-stats weight by. `extra_slope_cols` shorter than the extra count, or
    /// an empty inner vec, means that grouping is intercept-only.
    pub fn from_cluster_spec_ext(
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
        extra_slope_cols: &[Vec<usize>],
    ) -> Self {
        use crate::GroupingRelation;
        // RE-path constructor: only built for mixed models, so `re` is present.
        let re = cluster
            .re
            .as_ref()
            .expect("LmmGroupings::from_cluster_spec_ext requires re: Some (mixed model)");
        // Sole cluster-count formula in the crate: it rounds up under `FixedSize`,
        // which keeps a partial trailing parent's ids in range (production N is an
        // atom multiple; tests may not be).
        let n_primary = re.sizing.n_clusters_at(max_n);
        let q_p = 1 + slope_cols.len();
        // Width-general layout: the primary block is `q_p · n_primary` wide
        // ([intercept 0..S | slope_0 S..2S | … | slope_{q-2}]), and the
        // nested children + crossed tail follow exactly as before, shifted up by
        // the (q_p−1)·n_primary slope columns. q_p=1 ⇒ `prim_width == n_primary`,
        // so every offset (and k_total) is byte-identical to the pre-slope path.
        // OWNING site for the RE-column layout: `add_rows_multi`'s zx_slope/s fills
        // and `primary_gram`/`reml_deviance`'s reads use the same `d·n_primary + f`
        // (slope) / `prim_width + f·np + c` (nested-child) convention — change together.
        // `glmm::pirls_solve_blocked_extras` and `glmm::structured_schur_fill` also
        // read the `prim_width + f·np + c` nested-child convention to gather each
        // primary cluster's core-block columns — change together.
        let prim_width = q_p * n_primary;
        let n_extras = re.extra_groupings.len();
        // Over-envelope-by-count designs are legal — they route to the
        // sparse-Z path (`fit::classify_design`), which builds its own cap-free
        // structures. The old `n_extras <= MAX_EXTRA_GROUPINGS` guard was a
        // NoZ-envelope check; routing (not this constructor) now enforces it, so
        // dense code is never handed an over-envelope design.
        // θ order is declaration order: each extra owns a vech(Λ_g) block of
        // q_g(q_g+1)/2 slots starting at `vech_start`, packed after the primary
        // vech. For all q_g == 1, `vech_start == base_theta + g` — the pre-slope
        // scalar layout, bit-identical.
        let base_theta = q_p * (q_p + 1) / 2;
        let extra_qs: Vec<usize> = re
            .extra_groupings
            .iter()
            .map(|gs| 1 + gs.slopes.len())
            .collect();
        let mut vech_starts = vec![0usize; n_extras];
        let mut cursor = base_theta;
        for (g, &q_g) in extra_qs.iter().enumerate() {
            vech_starts[g] = cursor;
            cursor += q_g * (q_g + 1) / 2;
        }
        let mut nested_per_parent = 0usize;
        let mut nested = None;
        let mut extra_offsets = vec![0usize; n_extras];
        for (g, gs) in re.extra_groupings.iter().enumerate() {
            if let GroupingRelation::NestedWithin { n_per_parent } = gs.relation {
                // `.max(1)` clamp mirrored by `fit::core`'s capacity pin — change together.
                nested_per_parent = (n_per_parent).max(1) as usize;
                nested = Some(NestedFactor {
                    vech_start: vech_starts[g],
                    q: extra_qs[g],
                    decl: g,
                });
                extra_offsets[g] = prim_width; // nested children begin after the primary block
            }
        }
        // Nested children are q_nested RE columns each (q_nested == 1 ⇒ the
        // pre-slope single-indicator width).
        let q_nested = nested.map(|nf| nf.q).unwrap_or(0);
        // Nested block width mirrored by `fit::core`'s capacity pin — change together.
        let mut off = prim_width + n_primary * nested_per_parent * q_nested;
        let mut crossed = Vec::new();
        for (g, gs) in re.extra_groupings.iter().enumerate() {
            if let GroupingRelation::Crossed { n_clusters } = gs.relation {
                // Crossed level count + `.max(1)` mirrored by `fit::core`'s capacity
                // pin — change together.
                let k = (n_clusters).max(1) as usize;
                let q_g = extra_qs[g];
                crossed.push(CrossedFactor {
                    vech_start: vech_starts[g],
                    q: q_g,
                    n_levels: k,
                    decl: g,
                });
                extra_offsets[g] = off;
                off += k * q_g; // q_g RE columns per crossed level
            }
        }
        let extra_slopes_any = extra_qs.iter().any(|&q| q > 1);
        // Per-grouping slope x-cols, declaration order, padded to the extra count
        // (intercept-only groupings get an empty inner vec). A provided inner vec
        // must match `extra_qs[g] − 1` (one x-col per slope).
        let extra_slope_cols: Vec<Vec<usize>> = (0..n_extras)
            .map(|g| {
                let v = extra_slope_cols.get(g).cloned().unwrap_or_default();
                debug_assert!(v.is_empty() || v.len() == extra_qs[g] - 1);
                v
            })
            .collect();
        LmmGroupings {
            n_primary,
            nested_per_parent,
            nested,
            crossed,
            extra_offsets,
            k_total: off,
            primary_q: q_p,
            primary_slope_cols: slope_cols.to_vec(),
            diagonal_theta: compute_diagonal_theta(q_p, &extra_qs),
            extra_slopes_any,
            extra_q: extra_qs,
            primary_slope_scales: vec![1.0; slope_cols.len()],
            extra_slope_scales: extra_slope_cols
                .iter()
                .map(|v| vec![1.0; v.len()])
                .collect(),
            extra_slope_cols,
        }
    }

    /// Recompute the internal RE design-column scales from this fit's design and
    /// prior weights. Every route calls this once, after the grouping structure
    /// is built and before anything reads a design column as a random-effect
    /// covariate.
    ///
    /// The identity being installed: for a `q`-column RE block with per-column
    /// scales `s₁…s_q`, replacing the block's design `Z` by `Z̃ = Z·diag(1/sᵢ)`
    /// and its relative Cholesky factor `Λ` by `Λ̃[i][j] = sᵢ·Λ[i][j]` leaves
    /// `Z̃Λ̃ = ZΛ` — the same model, the same likelihood, the same REML criterion.
    /// Only the path BOBYQA takes through that criterion changes: θ̃ = s·θ puts
    /// every component on a comparable scale, so the solver's single trust radius
    /// can resolve them all (Bates et al. 2015 §3 for the `Λ_θ` parametrization
    /// this rides on). The covariance block back-maps as
    /// `D[i][j] = D̃[i][j]/(sᵢ·s_j)`, and a conditional mode as `b = b̃/sᵢ`;
    /// neither direction puts a Jacobian into the criterion.
    ///
    /// The implicit intercept subcolumn is exactly `1.0` and is not stored, so an
    /// intercept-only grouping is untouched by construction.
    pub fn set_slope_scales(&mut self, x: MatRef<'_, f64>, weights: Option<&[f64]>) {
        for d in 0..self.primary_slope_cols.len() {
            self.primary_slope_scales[d] = rms_column_scale(x, self.primary_slope_cols[d], weights);
        }
        for e in 0..self.extra_slope_cols.len() {
            for d in 0..self.extra_slope_cols[e].len() {
                self.extra_slope_scales[e][d] =
                    rms_column_scale(x, self.extra_slope_cols[e][d], weights);
            }
        }
    }

    /// Λ-row scale factor for row `r` of block `b` (`b == 0` the primary, `b == e+1`
    /// extra grouping `e`). Row 0 is the intercept subcolumn — exactly `1.0`.
    pub fn block_row_scale(&self, b: usize, r: usize) -> f64 {
        if r == 0 {
            return 1.0;
        }
        if b == 0 {
            self.primary_slope_scales[r - 1]
        } else {
            self.extra_slope_scales[b - 1][r - 1]
        }
    }

    /// Per-θ-index Λ-row scale, in θ order (primary vech, then each extra's vech in
    /// declaration order — the layout `n_theta`/`vech_start` assign). The solver's
    /// internal θ̃ relates to θ in the user's design units by `θ̃ = s·θ`
    /// entry-wise, so this vector is the forward map for a user warm start and the
    /// divisor for anything read back off θ̂ (see [`Self::set_slope_scales`]).
    pub fn theta_row_scales(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n_theta()];
        self.fill_theta_row_scales(&mut out);
        out
    }

    /// [`Self::theta_row_scales`] into caller-owned storage, so a per-draw caller
    /// can read the scales without a heap block. `out.len()` must be `n_theta()`.
    pub fn fill_theta_row_scales(&self, out: &mut [f64]) {
        debug_assert_eq!(out.len(), self.n_theta());
        let mut i = 0;
        for (b, &q) in std::iter::once(&self.primary_q)
            .chain(self.extra_q.iter())
            .enumerate()
        {
            // Column-major vech: column c contributes rows c..q.
            for c in 0..q {
                for r in c..q {
                    out[i] = self.block_row_scale(b, r);
                    i += 1;
                }
            }
        }
    }

    /// True iff any random-effect design column carries a scale other than exactly
    /// `1.0` — i.e. iff this fit's arithmetic differs from the unscaled path at all.
    pub fn any_slope_scaled(&self) -> bool {
        self.primary_slope_scales.iter().any(|&s| s != 1.0)
            || self
                .extra_slope_scales
                .iter()
                .any(|v| v.iter().any(|&s| s != 1.0))
    }

    /// Primary vech (`q_p(q_p+1)/2`) + a `vech(Λ_g)` block per extra grouping
    /// (`q_g(q_g+1)/2` each). With every `q_g == 1` this is
    /// `primary vech + #extras` — the pre-slope shape.
    pub fn n_theta(&self) -> usize {
        let prim = self.primary_q * (self.primary_q + 1) / 2;
        let nested = self.nested.map(|nf| nf.q * (nf.q + 1) / 2).unwrap_or(0);
        let crossed: usize = self.crossed.iter().map(|cf| cf.q * (cf.q + 1) / 2).sum();
        prim + nested + crossed
    }
    /// Columns eliminated family-by-family: the `q_p` primary RE cols per level
    /// plus nested children (`q_nested` cols each). (`k_crossed = k_total −
    /// k_family` is the dense tail.) `q_nested == 1` ⇒ the pre-slope width.
    pub fn k_family(&self) -> usize {
        let q_nested = self.nested.map(|nf| nf.q).unwrap_or(0);
        self.n_primary * self.primary_q + self.n_primary * self.nested_per_parent * q_nested
    }
    /// Width of the dense crossed-family tail: `k_total - k_family()`.
    pub fn k_crossed(&self) -> usize {
        self.k_total - self.k_family()
    }
    /// True iff the structured block+Schur GLMM PIRLS path
    /// (`glmm::pirls_solve_blocked_extras`) applies: extra groupings are present
    /// (an empty-extras shape routes to the no-extras *blocked* path instead) and
    /// the per-primary core-block width `q_core = primary_q + nested_per_parent`
    /// fits the `MAX_PRIMARY_Q` stack scratch the per-block Crout solve uses.
    /// Extra groupings here are intercept-only because `classify_design` routes
    /// any extra-slopes shape to `Solver::Sparse` for every family, so no
    /// slopes-on-extras check is needed. A non-eligible extras shape
    /// (oversized core) falls through to the dense `glmm::pirls_solve`.
    pub fn structured_extras_eligible(&self) -> bool {
        !self.extra_offsets.is_empty() && self.primary_q + self.nested_per_parent <= MAX_PRIMARY_Q
    }
    /// Borrow of the cached diagonal θ-index map (computed once per workspace by
    /// `compute_diagonal_theta`). Zero-alloc: the per-fit pin loop and the
    /// rho-schedule fold read this slice without reallocating. See the
    /// `diagonal_theta` field doc for the column-major vech layout.
    pub fn diagonal_theta(&self) -> &[usize] {
        &self.diagonal_theta
    }

    /// Blind θ₀ and per-component boxes. Diagonal vech entries (the q_p primary
    /// variances + extra scalars) start at THETA0 with box [0, HI]; off-diagonal
    /// vech entries start at 0 with the signed box [−HI, HI]. q_p=1 ⇒ the
    /// all-diagonal shape (every entry diagonal: θ₀ = [THETA0;n], box [0, HI]).
    pub fn blind_theta_and_bounds(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n_theta();
        let mut theta = vec![THETA0; n];
        let mut lower = vec![0.0; n];
        let upper = vec![THETA_HI; n];
        let diag = self.diagonal_theta();
        // Off-diagonal vech entries (primary AND every extra factor's Λ_g) get a
        // blind start of 0 and a signed box; diagonals keep θ₀ = THETA0, box
        // [0, HI]. q_g=1 extras are all-diagonal, so they are untouched.
        for i in 0..n {
            if !diag.contains(&i) {
                theta[i] = 0.0; // off-diagonal blind start
                lower[i] = -THETA_HI; // signed box
            }
        }
        (theta, lower, upper)
    }
}

// ---------------------------------------------------------------------------
// LmmFitScratch
// ---------------------------------------------------------------------------

/// Per-fit scratch, allocated once per (p, max_clusters). `T` is the kernel
/// scalar (`f64` today; the dual-number type when the derivative work lands);
/// fields that stay `f64` say why at their own doc.
pub struct LmmFitScratch<T = f64> {
    /// Row-major w×w family block (w = q_p + n_per) — assembled and
    /// Crout-factored in place per family; rows contiguous for the Crout.
    pub fam_a: Vec<T>,
    /// Stacked forward-solved family couplings, t_dim × W_tot column-major
    /// (W_tot = n_primary·w): column f·w+r is family f's L_f⁻¹B_f row r,
    /// contiguous. Filled and solved per family, consumed by ONE triangular
    /// GEMM downdate after the family loop — the per-family tail re-traversals
    /// are gone.
    pub bt: Vec<T>,
    /// (k_crossed+m)² tail [[H, B_x],[B_xᵀ, C]] over [crossed | X y],
    /// column-major lower triangle (entry (i,j) at j·t_dim+i); GEMM-downdated
    /// once per eval, then factored out-of-place into `tail_l` below.
    pub tail: Vec<T>,
    /// Out-of-place `chol_lower` output for `tail` (`t_dim²`, same layout):
    /// `tail` must survive UNFACTORED for `recover_ranef_family`, which
    /// re-factors it at θ̂ (see that function's doc for the contract).
    pub tail_l: Vec<T>,
    /// λ per local crossed column (θ of the owning factor), refreshed per eval.
    pub lam_x: Vec<T>,
    /// q_p×q_p primary-slope scratch (row-major), refreshed per eval/family: the
    /// lower-tri Λ_p and the per-level Gram G_f. Empty on the q_p=1 path. Kept in
    /// scratch so the deviance hot loop stays zero-alloc (the warm-path invariant).
    pub prim_lam: Vec<T>,
    pub prim_gram: Vec<T>,
    /// Balanced-collapse Grams: pair-major r ≤ r′ blocks, w(w+1)/2 of
    /// them, each a FULL t_dim×t_dim column-major G_rr′ = Σ_f raw_r(f)·raw_r′(f)ᵀ
    /// over the active balanced prefix — θ-independent, refreshed once per fit
    /// by `precompute_balanced_collapse`. Empty on the slope path (collapse
    /// never applies there). Stays `f64`: θ-independent.
    pub fam_gram: Vec<f64>,
    /// `w·t_dim` staging buffer for `precompute_balanced_collapse`'s per-family
    /// raw rows — its own field so that θ-independent precompute stops
    /// borrowing the now-`T` `bt` (which is per-eval scratch).
    pub collapse_stage: Vec<f64>,
    /// t_dim² combine scratch for the collapse downdate (lower triangle used);
    /// its first w slots double as the A⁻¹ forward-solve temp.
    pub comb: Vec<T>,
    /// w×w row-major A(θ)⁻¹, rebuilt per eval on the collapse path.
    pub a_inv: Vec<T>,
    /// Active balanced families (prefix length). 0 = collapse off → the
    /// per-family loop runs (the fallback and the pre-F behaviour).
    pub collapse_n_active: usize,
    /// m×m trailing block of the tail factor — identical semantics to the
    /// augmented [X y] factor; every recovery step reads only this.
    pub factor: Mat<f64>,
    pub betas: Vec<f64>,
    pub var_diag: Vec<f64>,
    pub t_sq: Vec<f64>,
    pub u: Vec<f64>,
    /// Spherical conditional modes `û` at θ̂ over the full RE-column set
    /// (elimination order, length `k_total`), written once per fit by
    /// [`recover_ranef`]. Its OWN buffer, not `u`: `u`'s head is the
    /// standard-error forward-solve scratch, which runs first and would be
    /// overwritten.
    pub ranef_u: Vec<f64>,
    /// Whether `ranef_u` holds a usable recovery for the current fit. False
    /// before any recovery has run, and after one whose re-factorization failed
    /// (which can only happen at a θ̂ the deviance itself already factored, so it
    /// is a numerical-edge guard, not an expected path).
    pub ranef_ok: bool,
    /// [`recover_ranef`]'s family-path solve buffers: crossed-tail û_x
    /// (k_crossed) and the per-family back-substitution rhs (w). Scratch
    /// fields so the recovery pass keeps the warm path allocation-free; every
    /// entry is overwritten before it is read, so no per-fit reset is needed.
    pub ranef_ux: Vec<f64>,
    pub ranef_rhs: Vec<f64>,
    pub sigma_sq: T,
    /// p×p X'V⁻¹X rebuild (L_XX·L_XXᵀ) + the shared joint-Wald scratch
    /// (mirrors the lme workspace triple the promoted helper expects).
    pub joint_xtvix: Mat<f64>,
    pub joint_k_inv: Mat<f64>,
    pub joint_sigma_t_chol: Mat<f64>,
    pub joint_rhs: Vec<f64>,
    // --- crossed/nested-slopes blocked path (`reml_deviance_blocked`) ---
    // All empty unless `extra_slopes_any`; sized once here so the blocked warm
    // path stays zero-alloc. `k = k_total`, `dim = k + m`.
    /// k×k materialized block-diagonal relative-covariance factor Λ (column-major
    /// lower-tri), refreshed per θ-eval.
    pub blocked_lam: Vec<f64>,
    /// k×k raw RE design Gram ZᵀZ (column-major), refreshed per θ-eval.
    pub blocked_g: Vec<f64>,
    /// k×k scratch holding Λᵀ·ZᵀZ between the two Λ-applications.
    pub blocked_tmp: Vec<f64>,
    /// dim×dim penalized augmented matrix [[ΛᵀZᵀZΛ+I, ΛᵀZᵀ[Xy]],[·, [Xy]ᵀ[Xy]]]
    /// (column-major lower-tri), faer-llt'd per θ-eval.
    pub blocked_p: Vec<f64>,
}

impl<T: Scalar> LmmFitScratch<T> {
    pub fn new(p: usize, max_clusters: usize) -> Self {
        Self::with_groupings(p, &LmmGroupings::single(max_clusters))
    }

    pub fn with_groupings(p: usize, g: &LmmGroupings) -> Self {
        let m = p + 1;
        let w = g.primary_q + g.nested_per_parent; // q_p primary cols + nested children
        let t_dim = g.k_crossed() + m;
        // prim_lam / prim_gram hold the q_p×q_p primary Λ/Gram on the slope path
        // AND on the blocked crossed/nested-slopes path (which unpacks the primary
        // Λ even when q_p == 1), so size them whenever either path can run.
        let q2 = if g.primary_q > 1 || g.extra_slopes_any {
            g.primary_q * g.primary_q
        } else {
            0
        };
        // Collapse scratch only on the intercept-primary path; slope w would
        // mis-size it and the path never collapses.
        let npairs = if g.primary_q == 1 { w * (w + 1) / 2 } else { 0 };
        // Blocked crossed/nested-slopes scratch only when that path is taken.
        let blocked_kk = if g.extra_slopes_any {
            g.k_total * g.k_total
        } else {
            0
        };
        let blocked_dim = if g.extra_slopes_any { g.k_total + m } else { 0 };
        LmmFitScratch {
            fam_a: vec![T::ZERO; w * w],
            bt: vec![T::ZERO; g.n_primary * w * t_dim],
            tail: vec![T::ZERO; t_dim * t_dim],
            tail_l: vec![T::ZERO; t_dim * t_dim],
            lam_x: vec![T::ZERO; g.k_crossed()],
            prim_lam: vec![T::ZERO; q2],
            prim_gram: vec![T::ZERO; q2],
            fam_gram: vec![0.0; npairs * t_dim * t_dim],
            collapse_stage: vec![0.0; w * t_dim],
            // max(t_dim², w): the first w slots double as the A⁻¹ forward-solve
            // temp, and deep nesting can push w past t_dim² (tiny p, large n_per).
            comb: vec![
                T::ZERO;
                if npairs > 0 {
                    (t_dim * t_dim).max(w)
                } else {
                    0
                }
            ],
            a_inv: vec![T::ZERO; if npairs > 0 { w * w } else { 0 }],
            collapse_n_active: 0,
            factor: Mat::zeros(m, m),
            betas: vec![0.0; p],
            var_diag: vec![0.0; p],
            t_sq: vec![0.0; p],
            u: vec![0.0; p],
            ranef_u: vec![0.0; g.k_total],
            ranef_ok: false,
            ranef_ux: vec![0.0; g.k_crossed()],
            ranef_rhs: vec![0.0; w],
            sigma_sq: T::from_f64(f64::NAN),
            joint_xtvix: Mat::zeros(p, p),
            joint_k_inv: Mat::zeros(p, p),
            joint_sigma_t_chol: Mat::zeros(p, p),
            joint_rhs: vec![0.0; p],
            blocked_lam: vec![0.0; blocked_kk],
            blocked_g: vec![0.0; blocked_kk],
            blocked_tmp: vec![0.0; blocked_kk],
            blocked_p: vec![0.0; blocked_dim * blocked_dim],
        }
    }
}

// ---------------------------------------------------------------------------
// LmmWorkspace — everything a fit needs, allocated once per problem shape.
// ---------------------------------------------------------------------------

/// Per-problem-shape scratch: sufficient stats, deviance/PIRLS buffers, and
/// θ-solver state, allocated once and reused across fits of the same shape.
pub struct LmmWorkspace {
    /// Accumulated per-RE-column sufficient statistics for the current data.
    pub suff: LmmSuffStats,
    /// Deviance/PIRLS scratch buffers sized to the same problem shape.
    pub fit: LmmFitScratch,
    /// BOBYQA solver state — `Bobyqa::new` is the crate's only allocation
    /// site; `minimize` is zero-alloc on the warm path.
    pub solver: Bobyqa,
    /// θ in/out buffer for `minimize`; holds θ̂ (post-pin) after `fit_lmm`.
    pub theta: Vec<f64>,
    /// Per-component box bounds. Diagonal entries: [0, THETA_HI].
    pub lower: Vec<f64>,
    /// Upper box bound, always THETA_HI regardless of diagonal/off-diagonal
    /// (see `lower` for the entry that carries the diagonal/off-diagonal split).
    pub upper: Vec<f64>,
}

impl LmmWorkspace {
    /// Allocates a workspace for a single-intercept grouping at `max_clusters`.
    /// Test-only since the retirement of the raw `LmmWorkspace` loop surface —
    /// the live core path builds through [`Self::for_cluster_spec_ext`].
    #[cfg(test)]
    pub fn new(p: usize, max_clusters: usize) -> Self {
        Self::with_groupings(p, LmmGroupings::single(max_clusters))
    }

    /// Workspace for a validated non-degenerate ClusterSpec at max_n. Carries
    /// the spec-derived truth start and a scaled BOBYQA schedule.
    /// `slope_cols` are the x_full column indices for the primary slopes
    /// (`spec.cluster_slope_design_cols` as usize); pass `&[]` for callers without slopes.
    /// Test-only convenience over [`Self::for_cluster_spec_ext`] (the live path)
    /// since the raw `LmmWorkspace` loop surface was retired.
    #[cfg(test)]
    pub fn for_cluster_spec(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
    ) -> Self {
        Self::for_cluster_spec_ext(p, cluster, max_n, slope_cols, &[])
    }

    /// As [`for_cluster_spec`], plus each extra grouping's resolved slope x-columns
    /// (`extra_slope_cols`, declaration order; `&[]` for intercept-only extras) —
    /// the crossed/nested-slopes entry. Both `glmm::fit` (standalone) and
    /// `glmm::mcpower` bind through here.
    pub fn for_cluster_spec_ext(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
        extra_slope_cols: &[Vec<usize>],
    ) -> Self {
        let groupings =
            LmmGroupings::from_cluster_spec_ext(cluster, max_n, slope_cols, extra_slope_cols);
        let n_theta = groupings.n_theta();
        // MIRRORED by `sparse_lmm_seed` (the sparse-Z path's topology-only solver
        // seed) — change the schedule (rho_begin / npt / bounds) together (both
        // feed through the shared `apply_campaign_overrides` tail).
        // Scaled schedule: rho_begin = 0.1·min θ₀ — the eval count is
        // dominated by rho shrinkage, not travel distance. The start is now the cold
        // blind θ₀ (ModelSpec is structure-only), so every diagonal entry is
        // THETA0 and the fold collapses to it. Fold over DIAGONAL entries only — a
        // signed off-diagonal λ_{d,j} near 0 must not drive the start radius.
        let blind_theta = vec![THETA0; n_theta];
        let rho_begin = (0.1
            * groupings
                .diagonal_theta()
                .iter()
                .map(|&i| blind_theta[i])
                .fold(f64::INFINITY, f64::min))
        .min(RHO_BEGIN);
        // npt: ⌈1.5n⌉+1 from n_theta = 3 up, Powell's 2n+1 below: the mid model
        // wins on every measured dim ≥ 3
        // (n=3 lmm_slope 1.06x / crossed_nested 1.05x, n=6 multislope 1.10x —
        // mostly smaller kernel inner dims, evals/fit flat), while at n=2 the
        // range collapses to n+2, which loses (lmm_nested 0.88x, evals 21.8→26.6).
        // GLMM keeps 2n+1 — its sweep was mixed-to-negative (glmm.rs). LMM-only.
        let npt = if n_theta >= 3 {
            (3 * n_theta).div_ceil(2) + 1
        } else {
            2 * n_theta + 1
        };
        let mut config = Config::new(n_theta);
        config.rho_begin = rho_begin;
        config.rho_end = RHO_END;
        config.npt = npt;
        apply_campaign_overrides(&mut config, n_theta);
        let fit = LmmFitScratch::with_groupings(p, &groupings);
        let (theta, lower, upper) = groupings.blind_theta_and_bounds();
        LmmWorkspace {
            suff: LmmSuffStats::with_groupings(p, groupings),
            fit,
            solver: Bobyqa::new(n_theta, config)
                .expect("BOBYQA config constants are valid by construction"),
            theta,
            lower,
            upper,
        }
    }

    /// Allocates the suff-stats accumulator, fit scratch, and BOBYQA solver
    /// state for the given grouping shape, once per problem shape. Test-only
    /// (reached only via [`Self::new`]) since the raw loop surface was retired.
    #[cfg(test)]
    pub fn with_groupings(p: usize, groupings: LmmGroupings) -> Self {
        let n_theta = groupings.n_theta();
        let fit = LmmFitScratch::with_groupings(p, &groupings);
        let (theta, lower, upper) = groupings.blind_theta_and_bounds();
        LmmWorkspace {
            suff: LmmSuffStats::with_groupings(p, groupings),
            fit,
            // The constants are valid by construction for the crate's checks
            // (npt default within bounds; box width 1e3 ≥ 2·RHO_BEGIN), so a
            // failure here is an engine bug, not a runtime branch.
            solver: Bobyqa::new(n_theta, bobyqa_config(n_theta))
                .expect("BOBYQA config constants are valid by construction"),
            theta,
            lower,
            upper,
        }
    }
}

// ---------------------------------------------------------------------------
// Primary slope block helpers (q_p×q_p) — free fns; q_p is tiny.
// ---------------------------------------------------------------------------

/// Unpack the primary q×q lower-triangular Λ from the column-major vech θ prefix
/// into `lam` (row-major, len q·q; upper triangle zeroed). `pub(crate)` — the
/// introspection surface reuses it to reconstruct the RE covariance D = ΛΛ′.
/// Caller owns `lam` so the deviance hot loop stays zero-alloc.
pub fn primary_lambda<T: Scalar>(theta: &[T], q: usize, lam: &mut [T]) {
    for v in lam[..q * q].iter_mut() {
        *v = T::ZERO;
    }
    let mut t = 0;
    for c in 0..q {
        for r in c..q {
            lam[r * q + c] = theta[t];
            t += 1;
        }
    }
}

/// Per-level primary Gram G_f (q×q, row-major) recovered from suff stats into
/// `gram`, no new accumulator: G[0][0]=n_f; G[0][a]=G[a][0]=Σ x_{a-1} over f;
/// G[a][b]=Σ x_{a-1} x_{b-1} over f. The slope covariates are [X y] rows, so
/// every entry sits in `s`. Component d's RE col at level f is `d·n_primary + f`
/// (mirrors `from_cluster_spec`'s RE-column layout — change together).
fn primary_gram<T: Scalar>(
    suff: &LmmSuffStats,
    g: &LmmGroupings,
    f: usize,
    q: usize,
    gram: &mut [T],
) {
    let n_prim = g.n_primary;
    for v in gram[..q * q].iter_mut() {
        *v = T::ZERO;
    }
    gram[0] = T::from_f64(suff.counts[f]); // G[0][0]
    for a in 1..q {
        // `s`'s COLUMN carries the RE column's own internal scale (divided in at
        // accumulation), but its ROW is the raw `[X y]` entry — so a Gram between
        // two RE columns picks up only one of the two divisions here and needs the
        // row side applied explicitly. Intercept rows have scale 1 by construction.
        let s_a = g.primary_slope_scales[a - 1];
        let sa = T::from_f64(suff.s[(g.primary_slope_cols[a - 1], f)] / s_a); // Σ z_{a-1} over f
        gram[a] = sa;
        gram[a * q] = sa;
        for b in 1..=a {
            // Σ z_{a-1} z_{b-1} over f — slope_{a-1}'s subcol against slope_{b-1}'s level.
            let v = T::from_f64(suff.s[(g.primary_slope_cols[a - 1], b * n_prim + f)] / s_a);
            gram[a * q + b] = v;
            gram[b * q + a] = v;
        }
    }
}

/// A_f = I_q + Λ′ G Λ into the lower triangle of the row-major `fam_a` block
/// (`stride` = family width w; what Crout reads). Λ lower-tri row-major,
/// G symmetric row-major. (Λ′G)[r][e] = Σ_{d≥r} Λ[d][r] G[d][e]; A[r][c] =
/// δ_{rc} + Σ_{e≥c} (Λ′G)[r][e] Λ[e][c]. The Λ′G row is hoisted into a
/// stack scratch per r (it's c-independent; recomputing it inside the c
/// loop made this O(q⁴)) — same d/e summation order, so bit-identical.
/// Measured ~0 wall-clock change even at q=8 (sim_max_q_slope): the
/// per-eval cost lives elsewhere; kept for the strictly-lower op count.
fn assemble_primary_a<T: Scalar>(fam_a: &mut [T], stride: usize, lam: &[T], gram: &[T], q: usize) {
    let mut m_r = [T::ZERO; MAX_PRIMARY_Q];
    for r in 0..q {
        for (e, m_re) in m_r.iter_mut().enumerate().take(q) {
            let mut acc = T::ZERO;
            for d in r..q {
                acc += lam[d * q + r] * gram[d * q + e];
            }
            *m_re = acc;
        }
        for c in 0..=r {
            let mut s = T::ZERO;
            for e in c..q {
                s += m_r[e] * lam[e * q + c];
            }
            fam_a[r * stride + c] = if r == c { T::ONE + s } else { s };
        }
    }
}

/// Family `f`'s `w×w` block `A_f = I + Λ_f′G_fΛ_f` into the lower triangle of
/// the row-major `fam_a` (`w = q_p + n_per`): the primary `q_p×q_p` block, then
/// the nested children's diagonals and their coupling to the primary.
///
/// Lifted out of [`reml_deviance`]'s family loop so [`recover_ranef`] rebuilds
/// the same `L_f` the deviance factored rather than restating the assembly —
/// the deviance keeps only the LAST family's factor, and recovery needs all of
/// them. Pure extraction: same operations in the same order, so the deviance is
/// bit-identical to the pre-extraction path.
#[allow(clippy::too_many_arguments)] // one call site each; the alternative is a struct of borrows
fn assemble_fam_a<T: Scalar>(
    fam_a: &mut [T],
    prim_gram: &mut [T],
    prim_lam: &[T],
    suff: &LmmSuffStats,
    f: usize,
    w: usize,
    th_p: T,
    th_n: T,
    slope: bool,
) {
    let g = &suff.groupings;
    let np = g.nested_per_parent;
    if slope {
        let q = g.primary_q;
        primary_gram(suff, g, f, q, prim_gram);
        assemble_primary_a(fam_a, w, prim_lam, prim_gram, q); // I + Λ′GΛ
                                                              // Composed nested children (rows/cols q..q+np). Scalar child λ = θ_n;
                                                              // child–child off-diagonals are 0 (children never
                                                              // co-occur). The primary↔child off-diagonal A[(q+c, e)] folds the raw
                                                              // cross-Gram (intercept = counts[child]; slope d = s[(slope_col_d,
                                                              // child_re_col)]) through Λ_p, mirroring how the scalar path reads counts for the
                                                              // intercept↔child term. n_primary = primary level count (slope RE
                                                              // stride); np = children per parent (nested width) — kept distinct.
        for c in 0..np {
            // Nested child RE col = prim_width + f·np + c (prim_width = q_p·n_primary).
            let gcol = g.n_primary * g.primary_q + f * np + c;
            let n_c = T::from_f64(suff.counts[gcol]);
            for c2 in 0..np {
                fam_a[(q + c) * w + (q + c2)] = T::ZERO;
            }
            fam_a[(q + c) * w + (q + c)] = T::ONE + th_n * th_n * n_c;
            // Primary↔child: A[(q+c, e)] = θ_n · Σ_{d≥e} Λ_p[d,e] · Graw_d,
            // Graw_0 = n_c (intercept), Graw_d = Σ_{i∈child} x_{slope_{d-1}}.
            for e in 0..q {
                let mut acc = T::ZERO;
                for d in e..q {
                    // `s`-ROW read of a slope covariate ⇒ apply the internal scale
                    // (the column here is the child's intercept, scale 1); see
                    // `primary_gram` for why the row side is not already divided.
                    let graw_d = if d == 0 {
                        n_c
                    } else {
                        T::from_f64(
                            suff.s[(g.primary_slope_cols[d - 1], gcol)]
                                / g.primary_slope_scales[d - 1],
                        )
                    };
                    acc += prim_lam[d * q + e] * graw_d;
                }
                fam_a[(q + c) * w + e] = th_n * acc;
            }
        }
    } else {
        // parent–child counts = child row counts (a child's rows all lie
        // inside its parent).
        let n_f = T::from_f64(suff.counts[f]);
        fam_a[0] = T::ONE + th_p * th_p * n_f;
        for c in 0..np {
            let gcol = g.n_primary + f * np + c;
            let n_c = T::from_f64(suff.counts[gcol]);
            for c2 in 0..np {
                fam_a[(1 + c) * w + (1 + c2)] = T::ZERO;
            }
            fam_a[(1 + c) * w] = th_p * th_n * n_c;
            fam_a[(1 + c) * w + (1 + c)] = T::ONE + th_n * th_n * n_c;
        }
    }
}

/// Family `f`'s coupling `B_f` to the `[crossed | X y]` tail, written as `w`
/// contiguous `t_dim`-long columns (`bt_fam[r·t_dim + t]`) — the same slice
/// [`reml_deviance`] writes at `fit.bt`'s columns `f·w .. f·w+w`. Extracted
/// alongside [`assemble_fam_a`], for the same reason and with the same
/// bit-identity claim.
#[allow(clippy::too_many_arguments)] // as `assemble_fam_a`
fn assemble_fam_b<T: Scalar>(
    bt_fam: &mut [T],
    lam_x: &[T],
    prim_lam: &[T],
    suff: &LmmSuffStats,
    f: usize,
    t_dim: usize,
    kx: usize,
    slope: bool,
    th_p: T,
    th_n: T,
) {
    let g = &suff.groupings;
    let m = suff.m;
    let np = g.nested_per_parent;
    if slope {
        // Primary rows folded through Λ_p; nested-child rows scaled by θ_n
        // (built at the shifted child offset). n_prim is the primary level
        // count (slope RE stride: slope d-1's col at level f = d·n_prim+f);
        // np is the nested width — kept distinct.
        let q = g.primary_q;
        let n_prim = g.n_primary;
        // Primary rows ↔ crossed tail: intercept (d=0) reads zx[(f,b)];
        // slope d reads zx_slope[(d·n_prim+f, b)]; both folded through Λ_p,
        // scaled by the crossed λ_b. Column-b slices hoisted (unit-stride).
        for b in 0..kx {
            let lam_b = lam_x[b];
            let zxb = suff.zx.col(b).try_as_col_major().unwrap().as_slice();
            let zxsb = suff.zx_slope.col(b).try_as_col_major().unwrap().as_slice();
            for r in 0..q {
                let mut brb = T::ZERO;
                for d in r..q {
                    let zeta = T::from_f64(if d == 0 { zxb[f] } else { zxsb[d * n_prim + f] });
                    brb += prim_lam[d * q + r] * zeta;
                }
                bt_fam[r * t_dim + b] = lam_b * brb;
            }
        }
        // Primary rows ↔ [X y] tail: Z_f′[Xy] row d at col j is s[(j, d·n_prim+f)]
        // (intercept d=0 at col f), folded through Λ_p. Level-f s-columns
        // hoisted once per family (unit-stride faer columns).
        let mut s_cols: [&[f64]; MAX_PRIMARY_Q] = [&[]; MAX_PRIMARY_Q];
        for (d, sc) in s_cols.iter_mut().enumerate().take(q) {
            *sc = suff
                .s
                .col(d * n_prim + f)
                .try_as_col_major()
                .unwrap()
                .as_slice();
        }
        for r in 0..q {
            let bcol = &mut bt_fam[r * t_dim + kx..r * t_dim + kx + m];
            for j in 0..m {
                let mut brj = T::ZERO;
                #[allow(clippy::needless_range_loop)]
                for d in r..q {
                    brj += prim_lam[d * q + r] * T::from_f64(s_cols[d][j]);
                }
                bcol[j] = brj;
            }
        }
        // Nested-child rows (q..q+np) — built at the shifted child RE col.
        for c in 0..np {
            let gcol = n_prim * q + f * np + c; // prim_width + f·np + c
            let off = (q + c) * t_dim;
            for b in 0..kx {
                bt_fam[off + b] = th_n * lam_x[b] * T::from_f64(suff.zx[(gcol, b)]);
            }
            let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
            let bcol = &mut bt_fam[off + kx..off + kx + m];
            for j in 0..m {
                bcol[j] = th_n * T::from_f64(scol[j]);
            }
        }
    } else {
        let s_f = suff.s.col(f).try_as_col_major().unwrap().as_slice();
        for b in 0..kx {
            bt_fam[b] = th_p * lam_x[b] * T::from_f64(suff.zx[(f, b)]);
        }
        {
            let bcol = &mut bt_fam[kx..kx + m];
            for j in 0..m {
                bcol[j] = th_p * T::from_f64(s_f[j]);
            }
        }
        for c in 0..np {
            let gcol = g.n_primary + f * np + c;
            let off = (1 + c) * t_dim;
            for b in 0..kx {
                bt_fam[off + b] = th_n * lam_x[b] * T::from_f64(suff.zx[(gcol, b)]);
            }
            let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
            let bcol = &mut bt_fam[off + kx..off + kx + m];
            for j in 0..m {
                bcol[j] = th_n * T::from_f64(scol[j]);
            }
        }
    }
}

/// Forward-solve `L_f⁻¹B_f` in place on one family's `w` tail-coupling columns
/// — axpy over contiguous `t_dim`-slices; per element the k-order subtractions
/// and the final divide are unchanged from the old row-sweep (solved `k<r`
/// values are final in both orders). Extracted alongside [`assemble_fam_b`].
/// The divide is deliberately NOT hoisted into a reciprocal the way the sparse
/// path's `schur_phase_b` does it — that is a ≤1-ulp change, and this path's
/// values are pinned.
fn fam_forward_solve<T: Scalar>(bt_fam: &mut [T], t_dim: usize, w: usize, fam_a: &[T]) {
    for r in 0..w {
        let (done, rest) = bt_fam.split_at_mut(r * t_dim);
        let col_r = &mut rest[..t_dim];
        for k in 0..r {
            let l_rk = fam_a[r * w + k];
            let col_k = &done[k * t_dim..(k + 1) * t_dim];
            for t in 0..t_dim {
                col_r[t] -= l_rk * col_k[t];
            }
        }
        let l_rr = fam_a[r * w + r];
        #[allow(clippy::needless_range_loop)]
        for t in 0..t_dim {
            col_r[t] /= l_rr;
        }
    }
}

// ---------------------------------------------------------------------------
// Conditional-mode recovery — once at θ̂, after β̂.
// ---------------------------------------------------------------------------

/// Recover the spherical conditional modes `û` at θ̂ into `fit.ranef_u`, setting
/// `fit.ranef_ok`.
///
/// **Why this is a separate pass at all.** The profiled REML criterion is a
/// determinant-and-quadratic-form identity computable straight off the factors,
/// so no evaluation ever forms `u` — never forming it is what makes each
/// evaluation cost O(clusters) instead of O(rows), and this does not touch that.
/// The modes are recovered afterwards by back-substitution: with `[u; β]`
/// solving the penalized least-squares system, eliminating β gives
///
/// ```text
/// u = L_ZZ⁻ᵀ (U_y − U_X β̂),   U = L_ZZ⁻¹ Λ′Z′[X y]
/// ```
///
/// and every input survives at convergence. What each evaluation DISCARDS is the
/// per-family `L_f` and the tail factor, so this pass rebuilds them — one extra
/// deviance evaluation's work against the 50–300 the θ-search already spent.
///
/// Caller contract: run AFTER the pin evaluation at θ̂ and after the β̂ backsolve
/// and the standard-error block. `betas` must hold β̂; `fit.tail` (general path)
/// or `fit.blocked_p` (crossed/nested-slopes path) must hold the state that
/// evaluation left. Nothing on the fit path is read back afterwards, so this
/// moves no reported estimate.
pub(crate) fn recover_ranef(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch) {
    fit.ranef_ok = false;
    let k = suff.groupings.k_total;
    if k == 0 || suff.n_rows == 0 {
        return;
    }
    fit.ranef_u[..k].fill(0.0);
    fit.ranef_ok = if suff.groupings.extra_slopes_any {
        recover_ranef_blocked(theta, suff, fit)
    } else {
        recover_ranef_family(theta, suff, fit)
    };
}

/// Recovery on the crossed/nested-slopes blocked path. `fit.blocked_p` still
/// holds the UNFACTORED penalized augmented matrix at θ̂ (faer's `llt` builds a
/// fresh factor rather than working in place), so one re-factorization hands
/// back both halves at once: `L_ZZ` is its leading `k×k` block and `U` its
/// `[X y]` rows, `U[a][j] = L[(k+j), a]`.
fn recover_ranef_blocked(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> bool {
    let _ = theta; // Λ is reconstructed by the caller; only the factor is read here
    let g = &suff.groupings;
    let m = suff.m;
    let p = m - 1;
    let k = g.k_total;
    let dim = k + m;
    let pref = faer::MatRef::from_column_major_slice(&fit.blocked_p[..dim * dim], dim, dim);
    let Ok(chol) = pref.llt(faer::Side::Lower) else {
        return false;
    };
    let l = chol.L();
    // rhs = U_y − U_X β̂, then one upper-triangular back-substitution against
    // L_ZZᵀ over the whole RE-column set.
    let u = &mut fit.ranef_u;
    for a in 0..k {
        let mut acc = l[(k + p, a)];
        for j in 0..p {
            acc -= l[(k + j, a)] * fit.betas[j];
        }
        u[a] = acc;
    }
    for a in (0..k).rev() {
        let mut acc = u[a];
        for i in (a + 1)..k {
            acc -= l[(i, a)] * u[i];
        }
        let laa = l[(a, a)];
        if !(laa.is_finite() && laa > 0.0) {
            return false;
        }
        u[a] = acc / laa;
    }
    true
}

/// Recovery on the family-blocked path (general dense AND balanced-collapse).
///
/// `L_ZZ` is `[[L_A, 0], [B_c, L_c]]` over `[families | crossed]`: block-diagonal
/// per family, then the crossed tail. `fit.tail` holds the DOWNDATED tail
/// `T − L21·L21ᵀ` that the evaluation factored, so re-factoring it recovers both
/// `L_c` and the crossed rows of `U`. The families are rebuilt one at a time —
/// the evaluation keeps only the last one's `L_f` — which is also why this arm
/// serves the collapse path unchanged: collapse replaces the family LOOP with
/// one representative `A(θ)` and a θ-independent Gram combine, but the per-family
/// `A_f` it stands in for is exactly what [`assemble_fam_a`] rebuilds, and the
/// tail it leaves behind is the same downdated tail.
fn recover_ranef_family(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> bool {
    let g = &suff.groupings;
    let m = suff.m;
    let p = m - 1;
    let kf = g.k_family();
    let kx = g.k_crossed();
    let t_dim = kx + m;
    let np = g.nested_per_parent;
    let q_p = g.primary_q;
    let w = q_p + np;
    let n_prim = g.n_primary;
    let th_p = theta[0];
    let th_n = g.nested.map(|nf| theta[nf.vech_start]).unwrap_or(0.0);
    let slope = q_p > 1;

    // Refill the θ-derived scratch rather than trusting whichever arm ran last:
    // the collapse arm never touches `prim_lam`, and both are cheap.
    if slope {
        primary_lambda(theta, q_p, &mut fit.prim_lam);
    }
    {
        let mut b = 0usize;
        for cf in &g.crossed {
            for _ in 0..cf.n_levels {
                fit.lam_x[b] = theta[cf.vech_start];
                b += 1;
            }
        }
    }

    // --- crossed tail: û_x is final on its own, since L_ZZᵀ is block-UPPER and
    // the crossed block is its last one. ---
    let u_x = &mut fit.ranef_ux;
    if kx > 0 {
        let tail_ref =
            faer::MatRef::from_column_major_slice(&fit.tail[..t_dim * t_dim], t_dim, t_dim);
        let Ok(chol) = tail_ref.llt(faer::Side::Lower) else {
            return false;
        };
        let l = chol.L();
        for b in 0..kx {
            let mut acc = l[(kx + p, b)];
            for j in 0..p {
                acc -= l[(kx + j, b)] * fit.betas[j];
            }
            u_x[b] = acc;
        }
        for b in (0..kx).rev() {
            let mut acc = u_x[b];
            for i in (b + 1)..kx {
                acc -= l[(i, b)] * u_x[i];
            }
            let lbb = l[(b, b)];
            if !(lbb.is_finite() && lbb > 0.0) {
                return false;
            }
            u_x[b] = acc / lbb;
        }
        for (b, &v) in u_x.iter().enumerate() {
            fit.ranef_u[kf + b] = v;
        }
    }

    // --- families: L_Aᵀ û_fam = (U_y − U_X β̂) − B_cᵀ û_x, family by family. ---
    let rhs = &mut fit.ranef_rhs;
    for f in 0..n_prim {
        assemble_fam_a(
            &mut fit.fam_a,
            &mut fit.prim_gram,
            &fit.prim_lam,
            suff,
            f,
            w,
            th_p,
            th_n,
            slope,
        );
        if !crate::linalg::block_chol(&mut fit.fam_a[..w * w], w) {
            return false;
        }
        // One family's columns of `bt` are scratch here — the fit is over and
        // nothing reads the stacked couplings again. Family 0's slot serves
        // every family, so this stays allocation-free at any cluster count.
        let bt_fam = &mut fit.bt[..w * t_dim];
        assemble_fam_b(
            bt_fam,
            &fit.lam_x,
            &fit.prim_lam,
            suff,
            f,
            t_dim,
            kx,
            slope,
            th_p,
            th_n,
        );
        fam_forward_solve(bt_fam, t_dim, w, &fit.fam_a);
        for r in 0..w {
            let col = &bt_fam[r * t_dim..(r + 1) * t_dim];
            let mut acc = col[kx + p];
            for j in 0..p {
                acc -= col[kx + j] * fit.betas[j];
            }
            for (b, &ux) in u_x.iter().enumerate() {
                acc -= col[b] * ux;
            }
            rhs[r] = acc;
        }
        for r in (0..w).rev() {
            let mut acc = rhs[r];
            for (i, &solved) in rhs.iter().enumerate().take(w).skip(r + 1) {
                acc -= fit.fam_a[i * w + r] * solved;
            }
            rhs[r] = acc / fit.fam_a[r * w + r];
        }
        // Scatter into the RE-column layout `from_cluster_spec_ext` defines:
        // primary component d at `d·n_primary + f`, nested child c at
        // `prim_width + f·n_per + c` — change together.
        for (r, &v) in rhs.iter().enumerate().take(q_p) {
            fit.ranef_u[r * n_prim + f] = v;
        }
        for c in 0..np {
            fit.ranef_u[q_p * n_prim + f * np + c] = rhs[q_p + c];
        }
    }
    true
}

// ---------------------------------------------------------------------------
// fit_lmm — BOBYQA θ-search + once-at-θ̂ recovery.
// ---------------------------------------------------------------------------

/// One general-path fit summary. θ̂ (post-pin) stays in `ws.theta`; β̂/Var/t²
/// land in `ws.fit`'s target slots — no per-fit allocation.
pub struct LmmFit {
    /// Residual variance estimate σ̂² at θ̂. NaN only when there is no
    /// endpoint to report at all (`ModelDegenerate` / rank-deficient) —
    /// finite on a `MaxFunReached` cap-out too (the plateau policy: a
    /// `MaxFunReached` cap-out reports its finite endpoint with
    /// `converged == false` rather than NaN-filling).
    pub sigma_sq: f64,
    /// Whether the optimizer reached an interior minimum or a pinned
    /// boundary (both count); false on a `MaxFunReached` cap-out (the
    /// endpoint is still reported, honestly, as non-converged) and on
    /// optimizer/numerical failure.
    pub converged: bool,
    /// Coding inherited from the retired `lme.rs`: 0 = interior min, 1 = pinned at a variance
    /// boundary (counted converged), 2 = no accepted optimum — either a
    /// `MaxFunReached` cap-out (finite endpoint still reported below) or an
    /// optimizer/numerical failure (NaN-filled).
    pub boundary_hit: u8,
    /// Objective evaluations consumed (diagnostics only).
    pub n_eval: usize,
    /// Observation-only evaluation counters for this fit. Gated because
    /// `LmmFit` is re-exported `pub` under `loop_advanced`: with `counters`
    /// off, that tier's surface must be unchanged.
    #[cfg(feature = "counters")]
    pub counters: crate::counters::EvalCounters,
    /// Joint Wald-χ² over the target set (the shared `joint_wald_chi_sq` helper). Under
    /// H₀: β_T = 0, asymptotically χ²(k). NaN on optimizer/numerical failure
    /// or an empty target set; finite on a `MaxFunReached` endpoint.
    pub joint_t_sq: f64,
    /// Bit k set iff diagonal variance component k (in `diagonal_theta()`
    /// order) pinned at 0. 0 unless `converged` — a `MaxFunReached` endpoint
    /// is reported as a point, not an accepted boundary, so it never sets
    /// this mask even though its near-zero diagonals are still numerically
    /// pinned for FP stability. u64 mask: over-envelope sparse designs can
    /// exceed the 32-component NoZ ceiling (up to 64).
    pub pinned_components: u64,
    /// Minimized profiled REML deviance at the accepted (or capped) θ̂ (the
    /// `reml_deviance` value after the pin re-eval). NaN only on
    /// optimizer/numerical failure — finite on a `MaxFunReached` endpoint.
    pub deviance: f64,
    /// Scale-invariant per-column pivot ratio of X'V⁻¹X at θ̂
    /// ([`crate::ols::min_pivot_ratio`]), with `pivot_col` the column attaining
    /// it. **Detection only** — no branch in this route reads it, and none may
    /// start to; see the measurement recorded at the computation site. Below
    /// [`PIVOT_MIN`] the coefficients are barely identified and the diagnostics
    /// channel says so. NaN when the deviance re-eval left no trustworthy
    /// factor to measure.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
}

/// Fit by BOBYQA minimisation of the REML profiled deviance over the box-
/// bounded θ, with β̂ / σ̂² / Var(β̂_target) recovered once at θ̂.
///
/// Caller contract: `ws.suff` holds the accumulated rows (reset + add_rows
/// per dataset); `target_indices` index design columns.
///
/// Compute the joint Wald-χ² statistic
///
/// ```text
/// W = β̂_T' [(K⁻¹)_TT]⁻¹ β̂_T / σ̂²
/// ```
///
/// where `K = X' V(θ̂)⁻¹ X` (or `X'X` on the τ̂≈0 path; the formula is the
/// same — the caller passes whichever `xtvix` was just used) and `T` is the
/// configured target subset. Under H₀: β_T = 0, `W ~ χ²(k)` asymptotically.
///
/// **Algorithm (bounded-alloc; all explicit storage from `LmmFitScratch` —
/// the two faer `llt` calls below allocate internally):**
///   1. Re-Cholesky `xtvix` (idempotent — same factorisation already cached in
///      `xtvix_factor`; faer doesn't expose a `LltRef` reconstructed from a
///      pre-computed L without recomputation, and the cost is O(p³/3) which
///      is negligible at p ≤ 20).
///   2. `solve_in_place(I_p)` against the new Cholesky → `joint_k_inv` becomes
///      `K⁻¹`. The function refills the identity internally before solving, so
///      the caller need not pre-initialize `joint_k_inv`.
///   3. Gather `(K⁻¹)_TT` into the top-left k×k block of `joint_sigma_t_chol`.
///   4. In-place Cholesky on that k×k block.
///   5. Solve `Σ_T x = β̂_T` via the new k×k Cholesky factor (`solve_in_place`
///      on a length-`k` column view of `joint_rhs`).
///   6. `W_raw = β̂_T · x`.
///   7. Divide by `σ̂²` and return; on any numerical failure return `NaN`.
///
/// Returns `NaN` if `k == 0`, `σ̂² ≤ 0`, or any of the Cholesky steps fail.
pub(crate) fn joint_wald_chi_sq(
    xtvix: MatRef<'_, f64>,
    betas: &[f64],
    sigma_sq: f64,
    target_indices: &[u32],
    mut joint_k_inv: MatMut<'_, f64>,
    mut joint_sigma_t_chol: MatMut<'_, f64>,
    joint_rhs: &mut [f64],
) -> f64 {
    use faer::linalg::solvers::Solve;
    use faer::reborrow::Reborrow;

    let p = betas.len();
    let k = target_indices.len();
    if k == 0 || p == 0 || !(sigma_sq.is_finite() && sigma_sq > FLOAT_NEAR_ZERO) {
        return f64::NAN;
    }

    // Step 1: re-Cholesky K = X'V⁻¹X (lower).
    let chol = match xtvix.llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::NAN,
    };

    // Step 2: K⁻¹ via `solve_in_place(I_p)`. Refill identity here so the
    // helper is safe to call repeatedly without depending on `LmmSuffStats::reset`
    // to have just run (cost: p² writes, p ≤ 20).
    for j in 0..p {
        for i in 0..p {
            joint_k_inv[(i, j)] = if i == j { 1.0 } else { 0.0 };
        }
    }
    chol.solve_in_place(joint_k_inv.rb_mut());

    // Step 3: gather Σ_T = (K⁻¹)_TT into the top-left k×k of `joint_sigma_t_chol`.
    for (a, &ti) in target_indices.iter().enumerate() {
        let ti = ti as usize;
        if ti >= p {
            return f64::NAN;
        }
        for (b, &tj) in target_indices.iter().enumerate() {
            let tj = tj as usize;
            if tj >= p {
                return f64::NAN;
            }
            joint_sigma_t_chol[(a, b)] = joint_k_inv[(ti, tj)];
        }
    }

    // Step 4: in-place Cholesky on the k×k block. We pass the full p×p view's
    // top-left k×k submatrix to faer.
    let sigma_t_view = joint_sigma_t_chol.rb().submatrix(0, 0, k, k);
    let chol_t = match sigma_t_view.llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::NAN,
    };

    // Step 5: copy β̂_T into the first k slots of joint_rhs and solve in
    // place via a k×1 column view over that prefix (the same
    // `from_column_major_slice_mut` idiom as the β solve) — no owned Mat,
    // no copy-out.
    for (a, &ti) in target_indices.iter().enumerate() {
        joint_rhs[a] = betas[ti as usize];
    }
    {
        let mut rhs = MatMut::from_column_major_slice_mut(&mut joint_rhs[..k], k, 1usize);
        chol_t.solve_in_place(rhs.rb_mut());
    }

    // Step 6: W_raw = β̂_T · x.
    let mut w_raw = 0.0_f64;
    for (a, &ti) in target_indices.iter().enumerate() {
        w_raw += betas[ti as usize] * joint_rhs[a];
    }
    if !w_raw.is_finite() {
        return f64::NAN;
    }

    // Step 7: divide by σ̂².
    w_raw / sigma_sq
}

/// `theta_start`: `None` → blind start (diagonals THETA0, off-diagonals 0,
/// the default for arbitrary provided bytes); `Some(θ₀)` → per-component spec-derived truth
/// start, `[primary, extras in declaration order]` (Y is always synthetic, so
/// true θ_g = τ_g/σ is known), with diagonal components clamped to
/// THETA_TRUTH_FLOOR and off-diagonals passed through verbatim. A
/// per-scenario constant — determinism and chunk merging are unaffected. The
/// DGP-derived hint is a deliberate, recorded exception to the
/// generation↔estimation split.
pub fn fit_lmm(
    ws: &mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
) -> LmmFit {
    fit_lmm_impl(ws, target_indices, theta_start, two_stage_enabled())
}

/// Test-visible wrapper: runs [`fit_lmm`]'s body with the two-stage warm
/// restart forced on, bypassing the `LMM_TWO_STAGE` env read — the unit test
/// asserting stage parity never touches the process environment (env
/// mutation races the parallel test runner).
#[cfg(test)]
pub(crate) fn fit_lmm_two_stage(
    ws: &mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
) -> LmmFit {
    fit_lmm_impl(ws, target_indices, theta_start, true)
}

fn fit_lmm_impl(
    ws: &mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    two_stage: bool,
) -> LmmFit {
    let LmmWorkspace {
        suff,
        fit,
        solver,
        theta,
        lower,
        upper,
    } = ws;
    let p = suff.m - 1;

    // Arm the balanced collapse for this dataset's counts (cheap —
    // O(n_primary·w²·t_dim²) once per fit; sets collapse_n_active = 0 on any
    // unbalanced/slope shape, which keeps the per-family loop).
    precompute_balanced_collapse(suff, fit);

    // Cold start per fit (no warm-start across sims — would re-import
    // cross-grid-point path dependence). A Some-start is clamped to the
    // floor on its diagonal coordinates only (off-diagonals pass through
    // verbatim); under the fixed RHO_BEGIN, PRIMA's start-projection may still
    // move a small start to rho_begin off the 0 bound — benign and
    // deterministic. The scaled schedule (rho_begin = 0.1·θ₀) that makes
    // small starts pay off is activation at workspace-construction time: rho
    // lives in the solver's construction-time Config and θ₀ is per-scenario,
    // so it belongs where the workspace is built per workload.
    match theta_start {
        Some(ts) => {
            debug_assert_eq!(ts.len(), theta.len());
            // A caller's θ is in the design's own units; the solver works on the
            // internally scaled θ̃ = s·θ (`LmmGroupings::set_slope_scales`), so the
            // warm start takes the forward map before anything else touches it.
            // THETA_TRUTH_FLOOR then floors the INTERNAL diagonals — the same scale
            // `PIN_THETA` tests, so the two absolute thresholds stay on one axis.
            let s = suff.groupings.theta_row_scales();
            for ((t, &v), &sc) in theta.iter_mut().zip(ts).zip(s.iter()) {
                *t = v * sc;
            }
            for &i in suff.groupings.diagonal_theta() {
                theta[i] = theta[i].max(THETA_TRUTH_FLOOR);
            }
        }
        None => {
            // Blind start: diagonals THETA0, off-diagonals 0 (the
            // `blind_theta_and_bounds` shape, zero-alloc). The former
            // all-THETA0 start put off-diagonal vech entries at 1.0 — off the
            // lme4/MixedModels unit-diagonal convention — and on the wide-slope
            // grid stratum that start funnels BOBYQA into a second-best optimum
            // in 8/9 cells (regression goldens at validation/goldens/optima/ pin
            // the correct optimum). Mirrors the sparse GLMM joint seed, which
            // fixed the same trap earlier (`fit_glmm_sparse`'s θ cold start,
            // sparse.rs).
            for t in theta.iter_mut() {
                *t = 0.0;
            }
            for &i in suff.groupings.diagonal_theta() {
                theta[i] = THETA0;
            }
        }
    }
    let mut counters = crate::counters::EvalCounters::new();
    let out = if two_stage {
        two_stage_minimize(suff, fit, theta, lower, upper)
    } else {
        solver.minimize(
            |xs| {
                let d = reml_deviance(xs, suff, fit);
                counters.record_eval(crate::counters::Stage::Two, d);
                d
            },
            theta,
            lower,
            upper,
        )
    };

    // Status mapping (the plateau policy): a `MaxFunReached` cap-out reports
    // its finite endpoint with `converged == false` rather than NaN-filling.
    // Converged ⇒ candidate fit. MaxFunReached ⇒ still runs the same
    // pin + rank-guard + recovery below (an honest finite endpoint), but
    // `converged` stays false and `boundary_hit` stays 2 rather than
    // migrating into the accepted-boundary (1) code. ModelDegenerate has no
    // endpoint worth reporting ⇒ NaN-fill (boundary_hit == 2). TargetReached
    // unreachable (f_target stays -inf); InvalidArgs would be an engine bug —
    // the workspace fixes shapes and bounds.
    debug_assert!(out.status != Status::InvalidArgs);
    let converged = matches!(out.status, Status::Converged);
    let has_endpoint = matches!(out.status, Status::Converged | Status::MaxFunReached);

    // Per-component deterministic pin: every DIAGONAL variance component ≤
    // PIN_THETA collapses to exactly 0 — FP-stable across platforms. Applied
    // to any reported endpoint (converged or capped), but `pinned`/
    // `pinned_components` (⇒ boundary_hit == 1, "accepted boundary") only
    // latch when the fit actually converged — a capped endpoint is reported
    // as a point, not accepted onto the boundary. Off-diagonal vech entries
    // (signed slope covariances) are never pinned: a corr → ±1 boundary
    // presents as the *diagonal* λ_{dd} → 0 under the Cholesky
    // parameterization, so pinning the diagonal is the whole policy.
    let diag = suff.groupings.diagonal_theta();
    let mut pinned = false;
    let mut pinned_components = 0u64;
    if has_endpoint {
        for (k, &ti) in diag.iter().enumerate() {
            if theta[ti] <= PIN_THETA {
                theta[ti] = 0.0;
                if converged {
                    pinned = true;
                    if k < u64::BITS as usize {
                        pinned_components |= 1u64 << k;
                    }
                }
            }
        }
    }

    // Pin eval at θ̂ — refreshes factor/σ̂² at the accepted-or-capped point
    // (the shipped path's "pin Cholesky at θ̂" step).
    let dev = if has_endpoint {
        reml_deviance(theta, suff, fit)
    } else {
        f64::INFINITY
    };

    // Ill-conditioning DETECTION on the leading p×p block of the augmented
    // factor, i.e. on X'V⁻¹X at θ̂. This route does not refuse at any pivot
    // value, and nothing below may start to. Two measured reasons
    // (2026-07-31, a 1-ULP perturbation sweep down past total loss of β̂):
    //
    //   * The standard errors stay truthful all the way down. They are stable
    //     under the perturbation to ~1e-5 relative at rungs where β̂ moves by
    //     100%, and they never understate the actual error — at the bottom they
    //     over-cover by four orders. A caller reading the SE can always tell the
    //     estimate is worthless, so refusing destroys information for no gain.
    //   * On this route the pivot statistic itself floors out: below a certain
    //     conditioning it stops tracking and wanders in [4.6e-16, 8.2e-15] while
    //     β̂ keeps degrading. No refuse threshold is constructible from it.
    //
    // `dev.is_finite()` still gates the measurement: a ModelDegenerate exit, or
    // a θ̂ whose deviance re-eval is non-finite, leaves no trustworthy factor.
    // Those two remain the only NaN-fill conditions, and they are what they
    // always were — no honest endpoint at all.
    let (pivot, pivot_col) = if dev.is_finite() {
        crate::ols::min_pivot_ratio(fit.factor.as_ref(), p)
    } else {
        (f64::NAN, 0)
    };
    if !has_endpoint || !dev.is_finite() {
        fit.ranef_ok = false;
        for v in fit.betas.iter_mut() {
            *v = f64::NAN;
        }
        for &t in target_indices {
            fit.var_diag[t as usize] = f64::NAN;
            fit.t_sq[t as usize] = f64::NAN;
        }
        return LmmFit {
            sigma_sq: f64::NAN,
            converged: false,
            boundary_hit: 2,
            n_eval: out.n_eval,
            #[cfg(feature = "counters")]
            counters,
            joint_t_sq: f64::NAN,
            pinned_components: 0,
            deviance: f64::NAN,
            pivot,
            pivot_col: pivot_col as u32,
        };
    }

    // β̂: backward solve L_XXᵀ β̂ = l_yX, where l_yX[j] = factor[(p, j)] (the
    // y-row of the augmented factor) — the once-at-θ̂ backsolve.
    for j in (0..p).rev() {
        let mut acc = fit.factor[(p, j)];
        for k in (j + 1)..p {
            acc -= fit.factor[(k, j)] * fit.betas[k];
        }
        fit.betas[j] = acc / fit.factor[(j, j)];
    }

    // Var(β̂_j) = σ̂²·‖L_XX⁻¹e_j‖² per target; t² = β̂²/Var — the retired `lme.rs`'s
    // step-7 forward-solve recipe on this factor.
    let sigma_sq = fit.sigma_sq;
    for &tj in target_indices {
        let tj = tj as usize;
        for v in fit.u[..p].iter_mut() {
            *v = 0.0;
        }
        for i in 0..p {
            let b_i = if i == tj { 1.0 } else { 0.0 };
            let mut acc = b_i;
            for k in 0..i {
                acc -= fit.factor[(i, k)] * fit.u[k];
            }
            fit.u[i] = acc / fit.factor[(i, i)];
        }
        let norm_sq: f64 = fit.u[..p].iter().map(|v| v * v).sum();
        let vd = sigma_sq * norm_sq;
        fit.var_diag[tj] = vd;
        fit.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
            (fit.betas[tj] * fit.betas[tj]) / vd
        } else {
            f64::NAN
        };
    }

    // Joint Wald-χ² over the target set — the shared `joint_wald_chi_sq` helper
    // (`pub(crate)`). It re-Choleskys X'V⁻¹X internally, so hand it the product
    // the augmented factor already encodes: X'V⁻¹X = L_XX·L_XXᵀ (leading p×p
    // of fit.factor; the y row is index p).
    let joint_t_sq = if target_indices.is_empty() {
        f64::NAN
    } else {
        for j in 0..p {
            for i in 0..p {
                let mut acc = 0.0;
                for k in 0..=i.min(j) {
                    acc += fit.factor[(i, k)] * fit.factor[(j, k)];
                }
                fit.joint_xtvix[(i, j)] = acc;
            }
        }
        joint_wald_chi_sq(
            fit.joint_xtvix.as_ref(),
            &fit.betas,
            sigma_sq,
            target_indices,
            fit.joint_k_inv.as_mut(),
            fit.joint_sigma_t_chol.as_mut(),
            &mut fit.joint_rhs,
        )
    };

    // Conditional modes, last: it reads β̂ and reuses `fit.tail`/`fit.bt`/
    // `fit.fam_a` as scratch, so it must come after every step that reads the
    // factor state the θ̂ evaluation left.
    recover_ranef(theta, suff, fit);

    LmmFit {
        sigma_sq,
        converged,
        boundary_hit: if converged { u8::from(pinned) } else { 2 },
        n_eval: out.n_eval,
        #[cfg(feature = "counters")]
        counters,
        joint_t_sq,
        pinned_components,
        deviance: dev,
        pivot,
        pivot_col: pivot_col as u32,
    }
}
