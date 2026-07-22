//! General-machine LMM solver core — family-blocked profiled-REML deviance
//! (primary + nested children eliminated family-by-family, crossed factors in
//! a dense tail Cholesky with [X y]) + BOBYQA θ-search over one diagonal θ
//! component per grouping. A degenerate single-intercept `ClusterSpec`
//! collapses to the per-cluster shrink-downdate arithmetic (up to FP
//! reassociation), so the q=1 validation corpus re-proves on this machine.
//!
//! Engine-resident: ALL Gaussian mixed (LMM) specs dispatch here through the
//! unified fit core — the single-random-intercept shape is an `LmmDense`/BOBYQA
//! case like any other, from every tier (stable and loop). The scalar-Brent
//! `lme_fit` in `lme.rs` is NOT reachable from `fit_on`. Its measured 3× win on
//! that shape was an allocation artifact — it reused preallocated scratch while
//! the old dispatch rebuilt a workspace per call, and handing BOBYQA the true θ
//! changed BOBYQA's runtime by under 6%, so the θ search was never the cost.
//! The reusable `FitWorkspace` captures that win for every shape. Brent is also
//! strictly worse at high cluster counts (its `O(n_clusters·P)` per-evaluation
//! downdate has no counterpart in `reml_deviance`'s balanced-collapse path), and
//! the two agree on β̂ to ~1e-9, so there was no accuracy axis to trade either.
//!
//! Hot-loop invariants (mirror `lme.rs`):
//!  * Bounded allocations on the warm path (twin test in `lmm::tests`): all
//!    scratch and the BOBYQA solver live in `LmmWorkspace`, allocated once
//!    per (p, max_clusters) shape; the only per-call allocations are faer
//!    `llt` internals — the same acceptance the shipped path carries.
//!  * Inference is squared statistics (`t_sq = β̂²/Var(β̂)`); never sqrt the
//!    SE, never call a CDF on the per-fit path.
//!  * `f64::INFINITY` is the deviance failure surface.
//!
//! **NR** = Press, Teukolsky, Vetterling & Flannery (2007), *Numerical Recipes:
//! The Art of Scientific Computing*, 3rd ed., Cambridge University Press.
//! BOBYQA is Powell, M.J.D. (2009), *The BOBYQA algorithm for bound constrained
//! optimization without derivatives*, Cambridge report DAMTP 2009/NA06.

use crate::ols::chol_rank_deficient;
use bobyqa::{Bobyqa, Config, Status};
use faer::{Mat, MatRef};
use std::sync::OnceLock;

/// θ start — DIAGONAL vech entries only; off-diagonals cold-start at 0
/// (unit diagonal, the lme4/MixedModels.jl default — the
/// `blind_theta_and_bounds` shape). Cold start per fit; no warm-start
/// across sims (would re-import cross-grid-point path dependence).
pub const THETA0: f64 = 1.0;
/// Per-component θ upper box — mirrors the shipped Brent reach (θ ≤ 1e3).
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
/// Re-swept {1e-6, 3e-6, 1e-5, 3e-5, 1e-4} against the full `cargo test
/// --release` suite (2026-07-03, 181 tests — the calibrating set widened by
/// the roadmap-step-2 sparse non-Gaussian solvers: both-paths cross-checks,
/// over-envelope smokes, and the sim_sparse_nb golden): 1e-6 and 3e-6 pass
/// 181/181; **1e-5 now fails** (`two_stage_matches_single_stage_on_grouseticks`
/// — it passed the pre-step-2 159-test sweep, so the boundary moved in); 3e-5
/// adds `fit_glmm_poisson_agq_matches_lme4` (β[0] off ~3e-3, past the crate's
/// beta_rel=1e-3 oracle floor) and 1e-4 adds
/// `fit_glmm_poisson_grouseticks_matches_lme4`. 3e-6 is retained: still fully
/// green, but it is now the boundary-adjacent candidate (the old "one step
/// back from 1e-5" margin is gone) — do NOT loosen past it. Original timing
/// rationale (vs 1e-6): 0.265–0.269s vs 0.288s on `grouseticks` (n=403,
/// 7 free params), a ~7–8% wall-time cut for free.
pub const GLMM_RHO_END: f64 = 3e-6;
/// Truth-start floor: a `Some(θ₀)` start is clamped to max(θ₀, this) so a
/// zero/near-zero true θ never starts the search on the boundary itself.
/// Keep ≥ 10·RHO_END: the future scaled schedule derives
/// rho_begin = 0.1·θ₀, and the crate requires rho_end ≤ rho_begin.
pub const THETA_TRUTH_FLOOR: f64 = 0.01;
/// Pin threshold: a Converged diagonal component ≤ this is deterministically
/// pinned at exactly 0 and counted converged. 1e-4 aligns the class boundary
/// with the shipped τ̂≈0 detection (`lme.rs` pins boundary_hit=1 fits at
/// θ = 1e-4).
pub const PIN_THETA: f64 = 1e-4;
/// Rank guard on the p×p block of the factor — mirrors `lme.rs` EPS_RANK.
pub const EPS_RANK: f64 = 1e-8;

/// BOBYQA config for an n_theta-dimensional θ-search. `Config::new` supplies
/// the PRIMA defaults (npt = 2n+1, max_fun = 500·n) — at n = 1 exactly
/// npt = 3 / max_fun = 500. Test-only (reached only via the test-gated
/// [`LmmWorkspace::with_groupings`]); the live path inlines its own config.
#[cfg(test)]
pub fn bobyqa_config(n_theta: usize) -> Config {
    let mut config = Config {
        rho_begin: RHO_BEGIN,
        rho_end: RHO_END,
        ..Config::new(n_theta)
    };
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

/// Shared tail for all BOBYQA config sites — campaign env hooks, no-op when unset.
pub(crate) fn apply_campaign_overrides(config: &mut Config, n: usize) {
    if let Some(npt) = npt_override(n) {
        config.npt = npt;
    }
    if let Some(mf) = max_fun_override(n) {
        config.max_fun = mf.max(config.npt + 1);
    }
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
fn two_stage_minimize(
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch,
    theta: &mut [f64],
    lower: &[f64],
    upper: &[f64],
) -> bobyqa::Outcome {
    let n = theta.len();
    let c1 = Config {
        npt: n + 2,
        rho_begin: RHO_BEGIN,
        rho_end: 1e-3,
        ..Config::new(n)
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
    let c2 = Config {
        npt: 2 * n + 1,
        rho_begin: rho_begin2,
        rho_end: RHO_END,
        ..Config::new(n)
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
    let mut config = Config {
        rho_begin,
        rho_end: RHO_END,
        npt,
        ..Config::new(n_theta)
    };
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
            extra_slope_cols,
        }
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
        // [0, HI]. q_g=1 extras are all-diagonal, so they are untouched — the
        // pre-slope behaviour (the loop used to stop at the primary vech).
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
// LmmSuffStats — augmented per-RE-column sufficient statistics.
// ---------------------------------------------------------------------------

/// w_i = [x_i ; y_i] (length m = p+1): `c` = Σ w wᵀ (lower triangle),
/// `s[:, a]` = Σ_{i in RE column a} w_i, `counts[a]` = n_a, over the full
/// RE-column set (`[primary | nested children | crossed]`, elimination order).
/// `zx` holds cross-counts only when crossed factors exist (crossed ⇒ Regime A
/// ⇒ K moderate); nested-only designs derive parent↔child coupling from
/// `counts` + the id/n_per parent map, so Regime-B-nested memory stays O(K·m).
///
/// Layout invariant: `s` stays per-RE-column-addressable with `counts`
/// alongside — the balanced-design collapse slots in at exactly this
/// granularity later; don't fold columns at accumulation time.
pub struct LmmSuffStats {
    /// Augmented width m = p + 1 (y in the last slot).
    pub m: usize,
    /// Rows accumulated since the last reset.
    pub n_rows: usize,
    /// Highest primary cluster id + 1 seen since the last reset.
    pub n_clusters: usize,
    /// Grouping-structure metadata (RE column layout, crossed/nested shape)
    /// shared with the deviance kernel.
    pub groupings: LmmGroupings,
    /// m×m Σ w wᵀ (lower triangle; upper never read).
    pub c: Mat<f64>,
    /// m × k_total per-RE-column Σ w.
    pub s: Mat<f64>,
    /// Per-RE-column Gram value Σ z_int²·wᵢ over rows in that RE column
    /// (z_int = 1, so unit weights reduce to the raw row count n_a). NOT a row
    /// count once weighted — every consumer reads it purely as a Gram diagonal
    /// entry; `df` for the fit comes from `n_rows`, never from `counts`.
    pub counts: Vec<f64>,
    /// Crossed cross-counts: zx[(a, b)] = #rows where RE column `a` and
    /// crossed column `k_family + b` co-occur. 0×0 when no crossed factors;
    /// nested↔primary coupling is derived from `counts` + the id/n_per parent
    /// map instead. Same-factor crossed pairs never co-occur (level-disjoint),
    /// so those entries stay 0 and the Ω assembly can read unconditionally.
    pub zx: Mat<f64>,
    /// Slope-weighted twin of `zx` (the slope-composition): for a primary slope RE
    /// column `scol = (d+1)·n_primary + f`, `zx_slope[(scol, b)] = Σ_{i ∈ f ∩
    /// crossed_b} x_{slope_d}` — the covariate-weighted co-occurrence the
    /// slope↔crossed coupling in `fam_b` reads (plain `zx` is unweighted, fit for
    /// the intercept row only). Same shape as `zx` (`k_total × k_crossed`); only
    /// the slope-RE-col rows are filled. 0×0 when no crossed factor; left all-zero
    /// when `primary_q == 1` (no slopes).
    pub zx_slope: Mat<f64>,
    /// Per-row widened [X y] (len m) — filled once per row so the c-triangle
    /// and s scatter read contiguous f64 instead of re-indexing the f32 data
    /// plane per (i, j). Scratch, not a statistic: reset leaves it alone.
    pub w_buf: Vec<f64>,
}

impl LmmSuffStats {
    /// Accumulator for a single-intercept grouping at `max_clusters` clusters
    /// (`k_total = max_clusters`, no crossed/nested columns).
    pub fn new(p: usize, max_clusters: usize) -> Self {
        Self::with_groupings(p, LmmGroupings::single(max_clusters))
    }

    /// Accumulator sized for an arbitrary `LmmGroupings` layout, including
    /// nested and crossed RE columns.
    pub fn with_groupings(p: usize, groupings: LmmGroupings) -> Self {
        let m = p + 1;
        let k = groupings.k_total;
        let kx = groupings.k_crossed();
        LmmSuffStats {
            m,
            n_rows: 0,
            n_clusters: 0,
            c: Mat::zeros(m, m),
            s: Mat::zeros(m, k),
            counts: vec![0.0; k],
            zx: Mat::zeros(if kx > 0 { k } else { 0 }, kx),
            zx_slope: Mat::zeros(if kx > 0 { k } else { 0 }, kx),
            w_buf: vec![0.0; m],
            groupings,
        }
    }

    /// Reset to "no rows seen", reusing storage.
    pub fn reset(&mut self) {
        let m = self.m;
        for j in 0..m {
            for i in 0..m {
                self.c[(i, j)] = 0.0;
            }
        }
        for a in 0..self.counts.len() {
            for j in 0..m {
                self.s[(j, a)] = 0.0;
            }
            self.counts[a] = 0.0;
        }
        let (zr, zc) = (self.zx.nrows(), self.zx.ncols());
        for j in 0..zc {
            for i in 0..zr {
                self.zx[(i, j)] = 0.0;
                self.zx_slope[(i, j)] = 0.0;
            }
        }
        self.n_rows = 0;
        self.n_clusters = 0;
    }

    /// Primary-only convenience — the primary-only shape.
    pub fn add_rows(&mut self, x: MatRef<'_, f64>, y: &[f64], cluster_ids: &[u32]) {
        self.add_rows_multi(x, y, cluster_ids, &[], None);
    }

    /// Accumulate a block of rows for every grouping. `extra_ids[g]` holds
    /// extra grouping g's GLOBALIZED level ids (workspace layout — crossed
    /// 0..I, nested parent·n_per+within), declaration order; this routine maps
    /// them onto the elimination-order column offsets. `weights[i]` (prior/case
    /// weight, unstable `loop_advanced` surface) is `wᵢ`; `None` is unit weight.
    /// Per-row rule for folding prior weights into the unit-weight suff-stats
    /// accumulator: every row is conceptually √wᵢ-scaled before hitting the math,
    /// so `wi.sqrt()` (`zw`) multiplies `w_buf` once (propagating one `zw` into
    /// every `[X y]` and slope-`z` read) and every bare intercept-`z=1.0` site
    /// takes one more explicit `zw` (or `wi` where both intercept sides already
    /// collapsed to a single literal — `zw·zw = wi`).
    pub fn add_rows_multi(
        &mut self,
        x: MatRef<'_, f64>,
        y: &[f64],
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        weights: Option<&[f64]>,
    ) {
        debug_assert_eq!(x.nrows(), y.len());
        debug_assert_eq!(x.nrows(), cluster_ids.len());
        debug_assert_eq!(extra_ids.len(), self.groupings.extra_offsets.len());
        debug_assert!(weights.is_none_or(|w| w.len() == x.nrows()));
        let p = self.m - 1;
        debug_assert_eq!(x.ncols(), p);
        let kf = self.groupings.k_family();
        let n_g = 1 + extra_ids.len();
        let mut gid = [0usize; 1 + MAX_EXTRA_GROUPINGS];
        for row in 0..x.nrows() {
            let wi = weights.map_or(1.0, |w| w[row]);
            let zw = wi.sqrt();
            gid[0] = cluster_ids[row] as usize;
            for (e, ids) in extra_ids.iter().enumerate() {
                // Intercept RE column of this level's q_g-wide block. q_g==1 ⇒ the
                // pre-slope `offset + id` (byte-identical); slope cols follow at
                // `gid + 1 .. gid + q_g`.
                gid[1 + e] =
                    self.groupings.extra_offsets[e] + ids[row] as usize * self.groupings.extra_q[e];
            }
            debug_assert!(gid[..n_g].iter().all(|&a| a < self.counts.len()));
            for &a in &gid[..n_g] {
                // Σ z_int²·wᵢ = Σ wᵢ over rows in this RE column (z_int = 1).
                self.counts[a] += wi;
            }
            // Load this row's [X y] into w_buf, then fold in one `zw` per side:
            // downstream reads of `w_buf` (the c Gram, slope-z reads) each carry
            // exactly one `zw`, so a product of two reads carries `zw² = wᵢ`.
            for j in 0..p {
                self.w_buf[j] = x[(row, j)];
            }
            self.w_buf[p] = y[row];
            for wj in &mut self.w_buf[..self.m] {
                *wj *= zw;
            }
            for &a in &gid[..n_g] {
                let scol = self
                    .s
                    .col_mut(a)
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut();
                // Intercept z = 1 becomes `zw`; `w_buf` already carries one `zw`,
                // so the product carries `zw² = wᵢ` — total wᵢ·[X y] per row.
                #[allow(clippy::needless_range_loop)]
                for j in 0..self.m {
                    scol[j] += zw * self.w_buf[j];
                }
            }
            for j in 0..self.m {
                let wj = self.w_buf[j];
                let ccol = self
                    .c
                    .col_mut(j)
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut();
                #[allow(clippy::needless_range_loop)]
                for i in j..self.m {
                    ccol[i] += self.w_buf[i] * wj;
                }
            }
            if self.groupings.k_crossed() > 0 && !self.groupings.extra_slopes_any {
                let slope = self.groupings.primary_q > 1;
                let n_prim = self.groupings.n_primary;
                for bi in 0..n_g {
                    let b = gid[bi];
                    if b >= kf {
                        let bl = b - kf;
                        #[allow(clippy::needless_range_loop)]
                        for ai in 0..n_g {
                            if ai != bi {
                                // Both sides intercept (z=1), collapsed to one
                                // literal — the weighted product is zw·zw = wᵢ.
                                self.zx[(gid[ai], bl)] += wi;
                            }
                        }
                        // Slope-weighted twin for the slope↔crossed coupling.
                        // The intercept row is `zx`'s gid[0]; each slope
                        // component d's RE col at this row's primary level gid[0]
                        // is (d+1)·n_primary + gid[0]. Reuses this crossed col `bl`
                        // — no re-derivation of crossed memberships. x widens
                        // f32→f64. Only the primary's crossed co-occurrence
                        // matters: a slope lives on the primary grouping, so the
                        // weight is x_{slope}; nested/other-crossed groupings carry
                        // no slope, so they contribute nothing here.
                        if slope {
                            for (d, &sc) in self.groupings.primary_slope_cols.iter().enumerate() {
                                let z = self.w_buf[sc]; // already carries one zw
                                                        // scol mirrors from_cluster_spec's RE-column layout — change together.
                                let scol = (d + 1) * n_prim + gid[0];
                                // b's side is the crossed intercept (z=1 → zw);
                                // total zw·zw = wᵢ, matching Σ wᵢ·x_slope·1.
                                self.zx_slope[(scol, bl)] += z * zw;
                            }
                        }
                    }
                }
            } else if self.groupings.k_crossed() > 0 {
                // Blocked crossed/nested-slopes path: fill `zx` with the FULL
                // covariate-weighted co-occurrence zx[(a_col, b_local)] = Σ z_a·z_b
                // over rows, for every (RE col a, crossed col b) on DISTINCT
                // groupings. z is 1 for an intercept component, x_{slope} for a slope
                // component. This subsumes the scalar path's counts + zx_slope; the
                // blocked tail reads all cross-factor coupling from here, the per-
                // level diagonal blocks from `s`/`counts`. zx_slope stays unused.
                let g = &self.groupings;
                let n_prim = g.n_primary;
                for bi in 0..n_g {
                    let b = gid[bi];
                    if b < kf {
                        continue; // only crossed columns own a `b_local`
                    }
                    let bl = b - kf;
                    let q_b = if bi == 0 {
                        g.primary_q
                    } else {
                        g.extra_q[bi - 1]
                    };
                    for db in 0..q_b {
                        let z_b = if db == 0 {
                            zw // intercept z=1 → zw (one weight side)
                        } else if bi == 0 {
                            self.w_buf[g.primary_slope_cols[db - 1]]
                        } else {
                            self.w_buf[g.extra_slope_cols[bi - 1][db - 1]]
                        };
                        let b_local = bl + db;
                        for ai in 0..n_g {
                            if ai == bi {
                                continue;
                            }
                            let q_a = if ai == 0 {
                                g.primary_q
                            } else {
                                g.extra_q[ai - 1]
                            };
                            for da in 0..q_a {
                                let (a_col, z_a) = if da == 0 {
                                    (gid[ai], zw) // intercept z=1 → zw
                                } else if ai == 0 {
                                    (
                                        da * n_prim + gid[0],
                                        self.w_buf[g.primary_slope_cols[da - 1]],
                                    )
                                } else {
                                    (gid[ai] + da, self.w_buf[g.extra_slope_cols[ai - 1][da - 1]])
                                };
                                self.zx[(a_col, b_local)] += z_a * z_b;
                            }
                        }
                    }
                }
            }
            // Primary slopes: each slope k's RE column at level gid[0] (offset
            // (k+1)·n_primary + gid[0]) accumulates z = x_{slope_k} weighted sums
            // into `s`; the intercept subcol (gid[0]) is already filled with z=1
            // above. counts is NOT incremented for slope subcols (the Gram reads
            // `s`, not counts). z and the [X y] weights widen f32→f64.
            if self.groupings.primary_q > 1 {
                let n_prim = self.groupings.n_primary;
                for (k, &sc) in self.groupings.primary_slope_cols.iter().enumerate() {
                    let z = self.w_buf[sc];
                    let scol = (k + 1) * n_prim + gid[0];
                    let scol_mut = self
                        .s
                        .col_mut(scol)
                        .try_as_col_major_mut()
                        .unwrap()
                        .as_slice_mut();
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..self.m {
                        scol_mut[j] += z * self.w_buf[j];
                    }
                }
            }
            // Extra-grouping slopes: slope d of grouping e accumulates z = x_{slope_d}
            // weighted [X y] into its RE column `gid[1+e] + 1 + d` (the q_g-wide level
            // block is [intercept | slope_0 | …]). The intercept subcol gid[1+e] is
            // already filled with z=1 by the `s` scatter above; counts is NOT
            // incremented for slope subcols (the Gram reads `s`). Same covariate-
            // weighted recipe as the primary; intercept-only extras scatter nothing.
            if self.groupings.extra_slopes_any {
                for e in 0..self.groupings.extra_slope_cols.len() {
                    let gintercept = gid[1 + e];
                    let n_d = self.groupings.extra_slope_cols[e].len();
                    for d in 0..n_d {
                        let sc = self.groupings.extra_slope_cols[e][d];
                        let z = self.w_buf[sc];
                        let scol = gintercept + 1 + d;
                        let scol_mut = self
                            .s
                            .col_mut(scol)
                            .try_as_col_major_mut()
                            .unwrap()
                            .as_slice_mut();
                        #[allow(clippy::needless_range_loop)]
                        for j in 0..self.m {
                            scol_mut[j] += z * self.w_buf[j];
                        }
                    }
                }
            }
            if gid[0] + 1 > self.n_clusters {
                self.n_clusters = gid[0] + 1;
            }
        }
        self.n_rows += x.nrows();
    }
}

// ---------------------------------------------------------------------------
// LmmFitScratch — per-fit scratch, allocated once per (p, max_clusters).
// ---------------------------------------------------------------------------

pub struct LmmFitScratch {
    /// Row-major w×w family block (w = q_p + n_per) — assembled and
    /// Crout-factored in place per family; rows contiguous for the Crout.
    pub fam_a: Vec<f64>,
    /// Stacked forward-solved family couplings, t_dim × W_tot column-major
    /// (W_tot = n_primary·w): column f·w+r is family f's L_f⁻¹B_f row r,
    /// contiguous. Filled and solved per family, consumed by ONE triangular
    /// GEMM downdate after the family loop — the per-family tail re-traversals
    /// are gone.
    pub bt: Vec<f64>,
    /// (k_crossed+m)² tail [[H, B_x],[B_xᵀ, C]] over [crossed | X y],
    /// column-major lower triangle (entry (i,j) at j·t_dim+i); GEMM-downdated
    /// once per eval, then one dense faer llt over a MatRef view.
    pub tail: Vec<f64>,
    /// λ per local crossed column (θ of the owning factor), refreshed per eval.
    pub lam_x: Vec<f64>,
    /// q_p×q_p primary-slope scratch (row-major), refreshed per eval/family: the
    /// lower-tri Λ_p and the per-level Gram G_f. Empty on the q_p=1 path. Kept in
    /// scratch so the deviance hot loop stays zero-alloc (the warm-path invariant).
    pub prim_lam: Vec<f64>,
    pub prim_gram: Vec<f64>,
    /// Balanced-collapse Grams: pair-major r ≤ r′ blocks, w(w+1)/2 of
    /// them, each a FULL t_dim×t_dim column-major G_rr′ = Σ_f raw_r(f)·raw_r′(f)ᵀ
    /// over the active balanced prefix — θ-independent, refreshed once per fit
    /// by `precompute_balanced_collapse`. Empty on the slope path (collapse
    /// never applies there).
    pub fam_gram: Vec<f64>,
    /// t_dim² combine scratch for the collapse downdate (lower triangle used);
    /// its first w slots double as the A⁻¹ forward-solve temp.
    pub comb: Vec<f64>,
    /// w×w row-major A(θ)⁻¹, rebuilt per eval on the collapse path.
    pub a_inv: Vec<f64>,
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
    pub sigma_sq: f64,
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

impl LmmFitScratch {
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
            fam_a: vec![0.0; w * w],
            bt: vec![0.0; g.n_primary * w * t_dim],
            tail: vec![0.0; t_dim * t_dim],
            lam_x: vec![0.0; g.k_crossed()],
            prim_lam: vec![0.0; q2],
            prim_gram: vec![0.0; q2],
            fam_gram: vec![0.0; npairs * t_dim * t_dim],
            // max(t_dim², w): the first w slots double as the A⁻¹ forward-solve
            // temp, and deep nesting can push w past t_dim² (tiny p, large n_per).
            comb: vec![
                0.0;
                if npairs > 0 {
                    (t_dim * t_dim).max(w)
                } else {
                    0
                }
            ],
            a_inv: vec![0.0; if npairs > 0 { w * w } else { 0 }],
            collapse_n_active: 0,
            factor: Mat::zeros(m, m),
            betas: vec![0.0; p],
            var_diag: vec![0.0; p],
            t_sq: vec![0.0; p],
            u: vec![0.0; p],
            sigma_sq: f64::NAN,
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
        let mut config = Config {
            rho_begin,
            rho_end: RHO_END,
            npt,
            ..Config::new(n_theta)
        };
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
pub fn primary_lambda(theta: &[f64], q: usize, lam: &mut [f64]) {
    for v in lam[..q * q].iter_mut() {
        *v = 0.0;
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
fn primary_gram(suff: &LmmSuffStats, g: &LmmGroupings, f: usize, q: usize, gram: &mut [f64]) {
    let n_prim = g.n_primary;
    for v in gram[..q * q].iter_mut() {
        *v = 0.0;
    }
    gram[0] = suff.counts[f]; // G[0][0]
    for a in 1..q {
        let sa = suff.s[(g.primary_slope_cols[a - 1], f)]; // Σ x_{a-1} over f
        gram[a] = sa;
        gram[a * q] = sa;
        for b in 1..=a {
            // Σ x_{a-1} x_{b-1} over f — slope_{a-1}'s subcol against slope_{b-1}'s level.
            let v = suff.s[(g.primary_slope_cols[a - 1], b * n_prim + f)];
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
fn assemble_primary_a(fam_a: &mut [f64], stride: usize, lam: &[f64], gram: &[f64], q: usize) {
    let mut m_r = [0.0_f64; MAX_PRIMARY_Q];
    for r in 0..q {
        for (e, m_re) in m_r.iter_mut().enumerate().take(q) {
            let mut acc = 0.0;
            for d in r..q {
                acc += lam[d * q + r] * gram[d * q + e];
            }
            *m_re = acc;
        }
        for c in 0..=r {
            let mut s = 0.0;
            for e in c..q {
                s += m_r[e] * lam[e * q + c];
            }
            fam_a[r * stride + c] = if r == c { 1.0 + s } else { s };
        }
    }
}

/// Balanced-collapse precompute: detect a balanced active prefix and
/// accumulate the θ-independent cross-Grams G_rr′ from the suff stats. Returns
/// false (and arms the fallback loop) when the design is unbalanced, has a
/// slope primary, or is empty. Balance = counts[f] equal over an active prefix
/// and zero after, per child slot c equal across active families (the
/// grid-atom-snapped layout; non-prefix actives are conservatively rejected).
/// `fit.bt` is per-eval scratch, free here — its first w·t_dim slots stage each
/// family's raw rows.
pub(crate) fn precompute_balanced_collapse(suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> bool {
    let g = &suff.groupings;
    fit.collapse_n_active = 0;
    if g.primary_q != 1 || g.n_primary == 0 || suff.n_rows == 0 {
        return false;
    }
    let np = g.nested_per_parent;
    let w = 1 + np;
    let kx = g.k_crossed();
    let m = suff.m;
    let t_dim = kx + m;
    let n0 = suff.counts[0];
    if n0 == 0.0 {
        return false;
    }
    let mut n_active = 1;
    while n_active < g.n_primary && suff.counts[n_active] == n0 {
        n_active += 1;
    }
    if suff.counts[n_active..g.n_primary].iter().any(|&c| c != 0.0) {
        return false; // hole or non-prefix layout — fall back
    }
    for c in 0..np {
        let c0 = suff.counts[g.n_primary + c]; // family 0, child slot c
        for f in 0..g.n_primary {
            let cc = suff.counts[g.n_primary + f * np + c];
            if (f < n_active && cc != c0) || (f >= n_active && cc != 0.0) {
                return false;
            }
        }
    }
    // Grams over the active prefix (inactive families are all-zero rows and
    // would contribute nothing anyway).
    let blk = t_dim * t_dim;
    let npairs = w * (w + 1) / 2;
    fit.fam_gram[..npairs * blk].fill(0.0);
    for f in 0..n_active {
        for r in 0..w {
            let gcol = if r == 0 {
                f
            } else {
                g.n_primary + f * np + (r - 1)
            };
            let dst = &mut fit.bt[r * t_dim..(r + 1) * t_dim];
            for (b, slot) in dst[..kx].iter_mut().enumerate() {
                *slot = suff.zx[(gcol, b)];
            }
            let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
            dst[kx..kx + m].copy_from_slice(scol);
        }
        let (bt, gram) = (&fit.bt, &mut fit.fam_gram);
        let mut pidx = 0;
        for r in 0..w {
            for rp in r..w {
                let gblk = &mut gram[pidx * blk..(pidx + 1) * blk];
                for j in 0..t_dim {
                    let vj = bt[rp * t_dim + j];
                    if vj != 0.0 {
                        for i in 0..t_dim {
                            gblk[j * t_dim + i] += bt[r * t_dim + i] * vj;
                        }
                    }
                }
                pidx += 1;
            }
        }
    }
    fit.collapse_n_active = n_active;
    true
}

// ---------------------------------------------------------------------------
// reml_deviance — the blocked-Cholesky objective.
// ---------------------------------------------------------------------------

/// Crossed/nested random-slopes REML deviance — the gated `extra_slopes_any`
/// path. Builds the full penalized augmented matrix
/// `P = [[ΛᵀZᵀZΛ + I, ΛᵀZᵀ[Xy]], [·, [Xy]ᵀ[Xy]]]` over `[all RE cols | X y]`
/// and takes ONE dense Cholesky. The crossed dimension is bounded (crossed forces
/// a FixedClusters primary), so `k = k_total` is independent of N. The block-
/// diagonal `Λ` carries each grouping's `q_g×q_g` relative-covariance factor; the
/// raw RE Gram `ZᵀZ` is recovered from the suff stats (per-level diagonal blocks
/// from `s`/`counts`, cross-factor blocks from the weighted `zx`). Same deviance
/// normalization as [`reml_deviance`] (`log|L_ZZ|² + log|L_XX|² + (N−P)·log σ̂²`),
/// so it reduces to the scalar value (to FP reassociation) when every `q_g == 1`.
///
/// Zero-alloc warm path: every buffer lives in `LmmFitScratch` (`blocked_*`),
/// sized once when `extra_slopes_any`. Returns INFINITY on any Cholesky failure.
fn reml_deviance_blocked(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> f64 {
    let g = &suff.groupings;
    let m = suff.m;
    let p = m - 1;
    let k = g.k_total;
    let dim = k + m;
    let n_prim = g.n_primary;
    let q_p = g.primary_q;
    let np = g.nested_per_parent;
    let prim_width = q_p * n_prim;
    let kf = g.k_family();

    // --- Λ (k×k, column-major lower-tri): block-diagonal per grouping/level ---
    fit.blocked_lam[..k * k].fill(0.0);
    // Primary: Λ_p on each level f's SCATTERED component columns {d·n_prim + f}.
    primary_lambda(theta, q_p, &mut fit.prim_lam);
    for f in 0..n_prim {
        for dc in 0..q_p {
            for dr in dc..q_p {
                let row = dr * n_prim + f;
                let col = dc * n_prim + f;
                fit.blocked_lam[col * k + row] = fit.prim_lam[dr * q_p + dc];
            }
        }
    }
    // Nested children: each child level's q_n×q_n Λ_n on its CONTIGUOUS block
    // [ic .. ic+q_n] — mirrors the crossed block below; the only difference is the
    // block stride (a child id is `f·np + c`, scattered after the primary block).
    // q_n==1 ⇒ a single scalar θ_n on the diagonal, byte-identical to the prior
    // intercept-only layout.
    if let Some(nf) = g.nested {
        let q_n = nf.q;
        let mut lam_n = [0.0f64; MAX_EXTRA_Q * MAX_EXTRA_Q];
        primary_lambda(&theta[nf.vech_start..], q_n, &mut lam_n);
        for f in 0..n_prim {
            for c in 0..np {
                let ic = prim_width + (f * np + c) * q_n;
                for dc in 0..q_n {
                    for dr in dc..q_n {
                        fit.blocked_lam[(ic + dc) * k + (ic + dr)] = lam_n[dr * q_n + dc];
                    }
                }
            }
        }
    }
    // Crossed: Λ_g on each level's CONTIGUOUS block [ic .. ic+q_g].
    let mut lam_g = [0.0f64; MAX_EXTRA_Q * MAX_EXTRA_Q];
    for cf in &g.crossed {
        let q = cf.q;
        primary_lambda(&theta[cf.vech_start..], q, &mut lam_g);
        let off = g.extra_offsets[cf.decl];
        for c in 0..cf.n_levels {
            let ic = off + c * q;
            for dc in 0..q {
                for dr in dc..q {
                    fit.blocked_lam[(ic + dc) * k + (ic + dr)] = lam_g[dr * q + dc];
                }
            }
        }
    }

    // --- raw RE design Gram G = ZᵀZ (k×k, full symmetric, column-major) ---
    // Step B: cross-factor coupling from the weighted `zx` (a = any RE col, b =
    // crossed col). Same-factor entries are 0 in `zx`; overwritten by step C.
    fit.blocked_g[..k * k].fill(0.0);
    for b in kf..k {
        let bl = b - kf;
        for a in 0..k {
            let v = suff.zx[(a, bl)];
            fit.blocked_g[b * k + a] = v;
            fit.blocked_g[a * k + b] = v;
        }
    }
    // Step C: per-level diagonal blocks from `s`/`counts`.
    // Primary family blocks G_f (component-major scatter).
    for f in 0..n_prim {
        primary_gram(suff, g, f, q_p, &mut fit.prim_gram);
        for dr in 0..q_p {
            for dc in 0..q_p {
                fit.blocked_g[(dc * n_prim + f) * k + (dr * n_prim + f)] =
                    fit.prim_gram[dr * q_p + dc];
            }
        }
    }
    // Nested children: per-child q_n×q_n diagonal Gram block + the primary↔child
    // cross-Gram. The diagonal block is a level's covariate-weighted scatter from
    // `s`/`counts`, identical in form to a crossed level's block (below). The cross
    // block is the within-family coupling — a nested child shares its rows with its
    // parent, so this q_p×q_n Σ_{child} z^{prim}·z^{child} is NOT in `zx` (which is
    // crossed-only). Entry (prim da, child dc): z=1 for an intercept component,
    // x_slope for a slope component; the four cases pick the matching `s`/`counts`
    // scatter. q_n==1 collapses to the prior n_c diagonal + the dc==0 cross column.
    if let Some(nf) = g.nested {
        let q_n = nf.q;
        let nscols = &g.extra_slope_cols[nf.decl];
        for f in 0..n_prim {
            for c in 0..np {
                let ic = prim_width + (f * np + c) * q_n;
                let n_c = suff.counts[ic];
                for dr in 0..q_n {
                    for dc in 0..q_n {
                        let v = if dr == 0 && dc == 0 {
                            n_c
                        } else if dr == 0 {
                            suff.s[(nscols[dc - 1], ic)] // Σ x_{dc-1}
                        } else if dc == 0 {
                            suff.s[(nscols[dr - 1], ic)] // Σ x_{dr-1}
                        } else {
                            suff.s[(nscols[dr - 1], ic + dc)] // Σ x_{dr-1} x_{dc-1}
                        };
                        fit.blocked_g[(ic + dc) * k + (ic + dr)] = v;
                    }
                }
                for da in 0..q_p {
                    let prow = da * n_prim + f;
                    for dc in 0..q_n {
                        let ccol = ic + dc;
                        let v = if da == 0 && dc == 0 {
                            n_c
                        } else if dc == 0 {
                            suff.s[(g.primary_slope_cols[da - 1], ic)] // Σ x^p_{da-1}
                        } else if da == 0 {
                            suff.s[(nscols[dc - 1], ic)] // Σ x^n_{dc-1}
                        } else {
                            suff.s[(g.primary_slope_cols[da - 1], ic + dc)] // Σ x^p_{da-1} x^n_{dc-1}
                        };
                        fit.blocked_g[ccol * k + prow] = v;
                        fit.blocked_g[prow * k + ccol] = v;
                    }
                }
            }
        }
    }
    // Crossed diagonal blocks G_gc (intercept n_c, slope rows covariate-weighted).
    for cf in &g.crossed {
        let q = cf.q;
        let off = g.extra_offsets[cf.decl];
        let scols = &g.extra_slope_cols[cf.decl];
        for c in 0..cf.n_levels {
            let ic = off + c * q;
            let n_c = suff.counts[ic];
            for dr in 0..q {
                for dc in 0..q {
                    let v = if dr == 0 && dc == 0 {
                        n_c
                    } else if dr == 0 {
                        suff.s[(scols[dc - 1], ic)] // Σ x_{dc-1}
                    } else if dc == 0 {
                        suff.s[(scols[dr - 1], ic)] // Σ x_{dr-1}
                    } else {
                        suff.s[(scols[dr - 1], ic + dc)] // Σ x_{dr-1} x_{dc-1}
                    };
                    fit.blocked_g[(ic + dc) * k + (ic + dr)] = v;
                }
            }
        }
    }

    // --- penalized augmented matrix P (dim×dim, column-major lower-tri) ---
    // P_zz = Λᵀ G Λ + I via two block-diagonal-aware contractions (tmp = ΛᵀG).
    fit.blocked_tmp[..k * k].fill(0.0);
    for bp in 0..k {
        for a in 0..k {
            let mut acc = 0.0;
            for ap in 0..k {
                let l = fit.blocked_lam[a * k + ap]; // Λ[ap][a]
                if l != 0.0 {
                    acc += l * fit.blocked_g[bp * k + ap]; // G[ap][bp]
                }
            }
            fit.blocked_tmp[bp * k + a] = acc;
        }
    }
    for b in 0..k {
        for a in b..k {
            let mut acc = 0.0;
            for bp in 0..k {
                let l = fit.blocked_lam[b * k + bp]; // Λ[bp][b]
                if l != 0.0 {
                    acc += fit.blocked_tmp[bp * k + a] * l;
                }
            }
            if a == b {
                acc += 1.0; // + I
            }
            fit.blocked_p[b * dim + a] = acc;
        }
    }
    // P_zx = Λᵀ Zᵀ[Xy]: row (k+j), col a = Σ_{a'} Λ[a'][a]·s[(j, a')]. (Zᵀ[Xy] = s.)
    for a in 0..k {
        for j in 0..m {
            let mut acc = 0.0;
            for ap in 0..k {
                let l = fit.blocked_lam[a * k + ap];
                if l != 0.0 {
                    acc += l * suff.s[(j, ap)];
                }
            }
            fit.blocked_p[a * dim + (k + j)] = acc;
        }
    }
    // [Xy]ᵀ[Xy] block = suff.c (lower-tri).
    for j in 0..m {
        for i in j..m {
            fit.blocked_p[(k + j) * dim + (k + i)] = suff.c[(i, j)];
        }
    }

    // --- one dense Cholesky; read the deviance off the factor ---
    let pref = faer::MatRef::from_column_major_slice(&fit.blocked_p[..dim * dim], dim, dim);
    let chol = match pref.llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::INFINITY,
    };
    let l = chol.L();
    let mut log_lzz_half = 0.0_f64;
    for i in 0..k {
        let lii = l[(i, i)];
        if !(lii.is_finite() && lii > 0.0) {
            return f64::INFINITY;
        }
        log_lzz_half += lii.ln();
    }
    let mut log_lxx_sq = 0.0_f64;
    for j in 0..p {
        let ljj = l[(k + j, k + j)];
        if !(ljj.is_finite() && ljj > 0.0) {
            return f64::INFINITY;
        }
        log_lxx_sq += ljj.ln();
    }
    log_lxx_sq *= 2.0;
    // Trailing m×m → fit.factor (recovery reads only this — augmented [X y] semantics).
    for j in 0..m {
        for i in 0..m {
            fit.factor[(i, j)] = if i >= j { l[(k + i, k + j)] } else { 0.0 };
        }
    }
    let lyy = fit.factor[(p, p)];
    let r_sq = lyy * lyy;
    let df = (suff.n_rows - p) as f64;
    let sigma_sq = r_sq / df;
    if !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        return f64::INFINITY;
    }
    fit.sigma_sq = sigma_sq;
    2.0 * log_lzz_half + log_lxx_sq + df * sigma_sq.ln()
}

/// REML profiled deviance at θ via the family-blocked augmented Cholesky.
///
/// Ω_θ over [primary | nested children | crossed | X y]. The leading block is
/// block-diagonal per FAMILY (a primary level + its nested children — nested
/// children never co-occur across parents), so it is eliminated family-by-
/// family: factor the (1+n_per)² A_f, forward-solve its coupling to the
/// [crossed | X y] tail — cost linear in cluster count. The per-family tail
/// downdates are stacked into ONE triangular GEMM after the family loop
/// (Tail −= Bt·Bt′ over the solved couplings in `bt`; result-moving vs the
/// old sequential per-family subtraction).
/// Crossed factors couple everything (the dense Z_a′Z_b coupling, sanctioned
/// dense within the stated regime), so they stay in the tail with [X y]: one
/// dense (k_crossed+m) faer llt per evaluation. With no extras this is the
/// per-cluster shrink downdate up to FP reassociation, and with no crossed
/// factors the tail is just the m×m [X y] block.
///
/// Balanced collapse (intercept-only primary): when the per-fit precompute
/// (`precompute_balanced_collapse`) finds a balanced active prefix — grid
/// atom-snapping guarantees one at production N — the family loop is replaced
/// by ONE Crout of the common A(θ), log|L_ZZ|'s family part by
/// n_active·log|L|, and the stacked-GEMM downdate by a θ-independent Gram
/// combine Σ_{r,r′} A⁻¹[r,r′]·scale_r·scale_r′·G_rr′, column-scaled by
/// diag(λ_x | 1). Reassociation-level result movement vs the loop; unbalanced
/// counts and the slope path (data-dependent A_f) keep the loop.
///
/// The deviance reads OFF THE FACTORS — log|L_ZZ|² from the family pivots +
/// the crossed tail diagonal, log|L_XX|², r² = L[p,p]² from the trailing m×m
/// block — no β backsolve per evaluation. Normalization matches
/// `lme.rs::profiled_deviance` exactly:
///   dev(θ) = log|V| + log|X'V⁻¹X| + (N−P)·log(σ̂²),
/// so general-vs-shipped deviance values agree to FP error, not up to a
/// constant. Returns INFINITY on any Cholesky failure / non-positive σ̂².
///
/// θ is vech-packed per grouping — [primary, extras in declaration order]. The
/// primary block is width-general: `Λ_p` is the column-major vech θ prefix
/// (`q_p(q_p+1)/2` entries), and the per-level Gram `G_f` is recovered from `s`
/// with no new accumulator.
///
/// The composition: the q_p primary block coexists with the intercept-only
/// crossed/nested extra tail in one family-blocked elimination. The family block
/// is `q_p + nested_per_parent` wide; the new primary-slope↔nested-child
/// off-diagonal falls out of `s` (free), and the primary-slope↔crossed-factor
/// coupling reads the slope-weighted `zx_slope` twin (each slope row d at level f
/// is `zx_slope[(d·n_primary+f, b)]`, vs the intercept's unweighted `zx[(f, b)]`).
/// The extra-grouping scalars keep q_g = 1.
pub fn reml_deviance(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> f64 {
    let g = &suff.groupings;
    debug_assert_eq!(theta.len(), g.n_theta());
    let m = suff.m;
    let p = m - 1;
    if suff.n_rows <= p || p == 0 {
        return f64::INFINITY;
    }
    // Crossed/nested random slopes route to the dense blocked path; the scalar
    // tail below stays byte-identical for every current (q_g==1) contract.
    if g.extra_slopes_any {
        return reml_deviance_blocked(theta, suff, fit);
    }
    let kf = g.k_family();
    let kx = g.k_crossed();
    let t_dim = kx + m;
    let np = g.nested_per_parent;
    let w = g.primary_q + np; // width-general family width: q_p primary cols + nested children
    let th_p = theta[0];
    let th_n = g.nested.map(|nf| theta[nf.vech_start]).unwrap_or(0.0);

    // Width-general primary factor (q_p ≥ 2 ⇒ slope path; q_p == 1 ⇒ scalar,
    // kept byte-identical). The slope path may now carry a crossed/nested tail
    // (the slope-composition). Λ_p is the vech-packed θ prefix, refreshed into
    // scratch (`fit.prim_lam`) so the hot loop stays zero-alloc.
    let slope = g.primary_q > 1;
    if slope {
        primary_lambda(theta, g.primary_q, &mut fit.prim_lam);
    }

    // λ per local crossed column. This scalar tail handles intercept-only extras
    // (q_g == 1, `vech_start` is the scalar θ index); slopes-on-extras (q_g > 1)
    // route to the blocked path before reaching here.
    debug_assert!(!g.extra_slopes_any);
    {
        let mut b = 0usize;
        for cf in &g.crossed {
            for _ in 0..cf.n_levels {
                fit.lam_x[b] = theta[cf.vech_start];
                b += 1;
            }
        }
    }

    // --- tail init: [[H, ·],[B_x, C]] (lower triangle, column-major) ---
    fit.tail[..t_dim * t_dim].fill(0.0);
    for b in 0..kx {
        let lam = fit.lam_x[b];
        let gcol = kf + b;
        // Cross-factor coupling (row b in earlier columns a < b); same-factor
        // zx entries are structurally 0.
        let zxb = suff.zx.col(b).try_as_col_major().unwrap().as_slice();
        for a in 0..b {
            fit.tail[a * t_dim + b] = lam * fit.lam_x[a] * zxb[kf + a];
        }
        let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
        let tcol = &mut fit.tail[b * t_dim..(b + 1) * t_dim];
        tcol[b] = 1.0 + lam * lam * suff.counts[gcol];
        for j in 0..m {
            tcol[kx + j] = lam * scol[j];
        }
    }
    for j in 0..m {
        let ccol = suff.c.col(j).try_as_col_major().unwrap().as_slice();
        let tcol = &mut fit.tail[(kx + j) * t_dim..(kx + j + 1) * t_dim];
        tcol[kx + j..kx + m].copy_from_slice(&ccol[j..m]);
    }

    // --- family elimination ---
    let collapse = !slope && fit.collapse_n_active > 0;
    let mut log_lzz_half = 0.0_f64; // hoisted — single binding both arms write
    if collapse {
        let n_active = fit.collapse_n_active;
        // One representative A from the balanced prefix (family 0) — the
        // legacy q=1 fill verbatim.
        let n_f = suff.counts[0];
        fit.fam_a[0] = 1.0 + th_p * th_p * n_f;
        for c in 0..np {
            let n_c = suff.counts[g.n_primary + c];
            for c2 in 0..np {
                fit.fam_a[(1 + c) * w + (1 + c2)] = 0.0;
            }
            fit.fam_a[(1 + c) * w] = th_p * th_n * n_c;
            fit.fam_a[(1 + c) * w + (1 + c)] = 1.0 + th_n * th_n * n_c;
        }
        // Crout — the legacy in-place loop, one factor for all families.
        let mut log_l_half = 0.0_f64;
        for j in 0..w {
            let mut d = fit.fam_a[j * w + j];
            for k in 0..j {
                let v = fit.fam_a[j * w + k];
                d -= v * v;
            }
            if !(d.is_finite() && d > 0.0) {
                return f64::INFINITY;
            }
            let l = d.sqrt();
            fit.fam_a[j * w + j] = l;
            log_l_half += l.ln();
            for i in (j + 1)..w {
                let mut v = fit.fam_a[i * w + j];
                for k in 0..j {
                    v -= fit.fam_a[i * w + k] * fit.fam_a[j * w + k];
                }
                fit.fam_a[i * w + j] = v / l;
            }
        }
        log_lzz_half = (n_active as f64) * log_l_half;
        // A⁻¹ = L⁻ᵀL⁻¹ column by column (w ≤ 1+n_per — hand-rolled). comb's
        // first w slots are the forward-solve temp; comb is refilled below.
        for r in 0..w {
            for i in 0..w {
                let mut acc = if i == r { 1.0 } else { 0.0 };
                for k in 0..i {
                    acc -= fit.fam_a[i * w + k] * fit.comb[k];
                }
                fit.comb[i] = acc / fit.fam_a[i * w + i];
            }
            for i in (0..w).rev() {
                let mut acc = fit.comb[i];
                for k in (i + 1)..w {
                    acc -= fit.fam_a[k * w + i] * fit.a_inv[k * w + r];
                }
                fit.a_inv[i * w + r] = acc / fit.fam_a[i * w + i];
            }
        }
        // Combine: comb(lower) = Σ_{r≤r′} scale_r·scale_r′·A⁻¹[r,r′]·(G + [r≠r′]Gᵀ).
        let t2 = t_dim * t_dim;
        fit.comb[..t2].fill(0.0);
        let (comb, gram) = (&mut fit.comb, &fit.fam_gram);
        let mut pidx = 0;
        for r in 0..w {
            let sr = if r == 0 { th_p } else { th_n };
            for rp in r..w {
                let srp = if rp == 0 { th_p } else { th_n };
                let coeff = sr * srp * fit.a_inv[r * w + rp];
                let gblk = &gram[pidx * t2..(pidx + 1) * t2];
                if coeff != 0.0 {
                    if r == rp {
                        for j in 0..t_dim {
                            for i in j..t_dim {
                                comb[j * t_dim + i] += coeff * gblk[j * t_dim + i];
                            }
                        }
                    } else {
                        for j in 0..t_dim {
                            for i in j..t_dim {
                                comb[j * t_dim + i] +=
                                    coeff * (gblk[j * t_dim + i] + gblk[i * t_dim + j]);
                            }
                        }
                    }
                }
                pidx += 1;
            }
        }
        // Tail −= D·comb·D, D = diag(λ_x | 1_m) — column scaling folded here.
        for j in 0..t_dim {
            let dj = if j < kx { fit.lam_x[j] } else { 1.0 };
            for i in j..t_dim {
                let di = if i < kx { fit.lam_x[i] } else { 1.0 };
                fit.tail[j * t_dim + i] -= di * dj * fit.comb[j * t_dim + i];
            }
        }
    } else {
        for f in 0..g.n_primary {
            // A_f (w×w lower): the primary q_p×q_p block A_p = I + Λ′GΛ, then (on the
            // intercept-only scalar path) nested-child diags + parent–child counts. The
            // slope branch additionally carries the composed nested children;
            // the scalar/q_p=1 `else` stays byte-identical (q_p=1 parity).
            if slope {
                let q = g.primary_q;
                primary_gram(suff, g, f, q, &mut fit.prim_gram);
                // Disjoint field borrows keep this zero-alloc and borrow-checked.
                assemble_primary_a(&mut fit.fam_a, w, &fit.prim_lam, &fit.prim_gram, q); // I + Λ′GΛ
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
                    let n_c = suff.counts[gcol];
                    for c2 in 0..np {
                        fit.fam_a[(q + c) * w + (q + c2)] = 0.0;
                    }
                    fit.fam_a[(q + c) * w + (q + c)] = 1.0 + th_n * th_n * n_c;
                    // Primary↔child: A[(q+c, e)] = θ_n · Σ_{d≥e} Λ_p[d,e] · Graw_d,
                    // Graw_0 = n_c (intercept), Graw_d = Σ_{i∈child} x_{slope_{d-1}}.
                    for e in 0..q {
                        let mut acc = 0.0;
                        for d in e..q {
                            let graw_d = if d == 0 {
                                n_c
                            } else {
                                suff.s[(g.primary_slope_cols[d - 1], gcol)]
                            };
                            acc += fit.prim_lam[d * q + e] * graw_d;
                        }
                        fit.fam_a[(q + c) * w + e] = th_n * acc;
                    }
                }
            } else {
                // parent–child counts = child row counts (a child's rows all lie
                // inside its parent).
                let n_f = suff.counts[f];
                fit.fam_a[0] = 1.0 + th_p * th_p * n_f;
                for c in 0..np {
                    let gcol = g.n_primary + f * np + c;
                    let n_c = suff.counts[gcol];
                    for c2 in 0..np {
                        fit.fam_a[(1 + c) * w + (1 + c2)] = 0.0;
                    }
                    fit.fam_a[(1 + c) * w] = th_p * th_n * n_c;
                    fit.fam_a[(1 + c) * w + (1 + c)] = 1.0 + th_n * th_n * n_c;
                }
            }
            // In-place Crout Cholesky over the row-major w×w block, w ≤ 1+n_per,
            // via the shared kernel in `crate::linalg::block_chol` (zero-alloc;
            // false on a non-positive pivot, mapped to +INFINITY here — the
            // module's failure surface). Chains are ≤ w links — not chain-sick.
            // Pivots are multiplied into a per-family product (≤ w ≈ 9 terms,
            // each ≥ 1 since A_f's diagonal is 1.0 + θ²·n) and logged once
            // after the loop instead of once per pivot — log(∏l) = Σln(l),
            // same value, ~w× fewer .ln() calls. Must NOT accumulate this
            // product across the outer `f` loop: with ~60 families and θ up
            // to THETA_HI (1e3) during BOBYQA exploration, a global product
            // over ~480 terms can overflow f64 to +Infinity where the
            // per-family scoping (bounded product, reset each family) stays
            // finite.
            if !crate::linalg::block_chol(&mut fit.fam_a[..w * w], w) {
                return f64::INFINITY;
            }
            let mut fam_prod = 1.0_f64;
            for j in 0..w {
                fam_prod *= fit.fam_a[j * w + j];
            }
            log_lzz_half += fam_prod.ln();
            // B_f (rows = Bt columns f·w..f·w+w, each contiguous): cols [crossed | X y].
            let fb = f * w;
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
                    let lam_b = fit.lam_x[b];
                    let zxb = suff.zx.col(b).try_as_col_major().unwrap().as_slice();
                    let zxsb = suff.zx_slope.col(b).try_as_col_major().unwrap().as_slice();
                    for r in 0..q {
                        let mut brb = 0.0;
                        for d in r..q {
                            let zeta = if d == 0 { zxb[f] } else { zxsb[d * n_prim + f] };
                            brb += fit.prim_lam[d * q + r] * zeta;
                        }
                        fit.bt[(fb + r) * t_dim + b] = lam_b * brb;
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
                    let bcol = &mut fit.bt[(fb + r) * t_dim + kx..(fb + r) * t_dim + kx + m];
                    for j in 0..m {
                        let mut brj = 0.0;
                        #[allow(clippy::needless_range_loop)]
                        for d in r..q {
                            brj += fit.prim_lam[d * q + r] * s_cols[d][j];
                        }
                        bcol[j] = brj;
                    }
                }
                // Nested-child rows (q..q+np) — built at the shifted child RE col.
                for c in 0..np {
                    let gcol = n_prim * q + f * np + c; // prim_width + f·np + c
                    let off = (fb + q + c) * t_dim;
                    for b in 0..kx {
                        fit.bt[off + b] = th_n * fit.lam_x[b] * suff.zx[(gcol, b)];
                    }
                    let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
                    let bcol = &mut fit.bt[off + kx..off + kx + m];
                    for j in 0..m {
                        bcol[j] = th_n * scol[j];
                    }
                }
            } else {
                let s_f = suff.s.col(f).try_as_col_major().unwrap().as_slice();
                let b0 = fb * t_dim;
                for b in 0..kx {
                    fit.bt[b0 + b] = th_p * fit.lam_x[b] * suff.zx[(f, b)];
                }
                {
                    let bcol = &mut fit.bt[b0 + kx..b0 + kx + m];
                    for j in 0..m {
                        bcol[j] = th_p * s_f[j];
                    }
                }
                for c in 0..np {
                    let gcol = g.n_primary + f * np + c;
                    let off = (fb + 1 + c) * t_dim;
                    for b in 0..kx {
                        fit.bt[off + b] = th_n * fit.lam_x[b] * suff.zx[(gcol, b)];
                    }
                    let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
                    let bcol = &mut fit.bt[off + kx..off + kx + m];
                    for j in 0..m {
                        bcol[j] = th_n * scol[j];
                    }
                }
            }
            // Forward-solve L_f⁻¹ B_f in place on this family's Bt columns — axpy
            // over contiguous t_dim-slices; per element the k-order subtractions
            // and the final divide are unchanged from the old row-sweep (solved
            // k<r values are final in both orders).
            for r in 0..w {
                let (done, rest) = fit.bt.split_at_mut((fb + r) * t_dim);
                let col_r = &mut rest[..t_dim];
                for k in 0..r {
                    let l_rk = fit.fam_a[r * w + k];
                    let col_k = &done[(fb + k) * t_dim..(fb + k + 1) * t_dim];
                    for t in 0..t_dim {
                        col_r[t] -= l_rk * col_k[t];
                    }
                }
                let l_rr = fit.fam_a[r * w + r];
                #[allow(clippy::needless_range_loop)]
                for t in 0..t_dim {
                    col_r[t] /= l_rr;
                }
            }
        }

        // --- one stacked downdate: Tail −= Σ_f B_f′B_f = Bt·Bt′ (lower) ---
        // The n_primary per-family rank-w tail re-traversals collapse into ONE
        // triangular GEMM through faer's blocked multi-accumulator FMA kernels
        // (Par::Seq — per-fit parallelism is the outer rayon loop). RESULT-MOVING:
        // GEMM accumulation order replaces the per-family sequential subtraction;
        // sanctioned, verified against the brute-force oracle + validation bands which
        // are orders wider than the reorder's last-ulp footprint.
        let w_tot = g.n_primary * w;
        {
            let bt = faer::MatRef::from_column_major_slice(&fit.bt[..t_dim * w_tot], t_dim, w_tot);
            let tail = faer::MatMut::from_column_major_slice_mut(
                &mut fit.tail[..t_dim * t_dim],
                t_dim,
                t_dim,
            );
            faer::linalg::matmul::triangular::matmul(
                tail,
                faer::linalg::matmul::triangular::BlockStructure::TriangularLower,
                faer::Accum::Add,
                bt,
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                bt.transpose(),
                faer::linalg::matmul::triangular::BlockStructure::Rectangular,
                -1.0,
                faer::Par::Seq,
            );
        }
    }

    // --- dense tail factorization (faer llt on a MatRef view of the tail
    // scratch — same call/FP exposure as before) ---
    let tail_ref = faer::MatRef::from_column_major_slice(&fit.tail[..t_dim * t_dim], t_dim, t_dim);
    let chol = match tail_ref.llt(faer::Side::Lower) {
        Ok(c) => c,
        Err(_) => return f64::INFINITY,
    };
    let l = chol.L();
    for b in 0..kx {
        let lbb = l[(b, b)];
        if !(lbb.is_finite() && lbb > 0.0) {
            return f64::INFINITY;
        }
        log_lzz_half += lbb.ln();
    }
    let log_lzz_sq = 2.0 * log_lzz_half;
    // Trailing m×m → fit.factor (augmented [X y] semantics; recovery reads only this).
    for j in 0..m {
        let lcol = l.col(kx + j).try_as_col_major().unwrap().as_slice();
        for i in 0..m {
            fit.factor[(i, j)] = if i >= j { lcol[kx + i] } else { 0.0 };
        }
    }

    let mut log_lxx_sq = 0.0_f64;
    for j in 0..p {
        let ljj = fit.factor[(j, j)];
        if !(ljj.is_finite() && ljj > 0.0) {
            return f64::INFINITY;
        }
        log_lxx_sq += ljj.ln();
    }
    log_lxx_sq *= 2.0;

    let lyy = fit.factor[(p, p)];
    let r_sq = lyy * lyy;
    let df = (suff.n_rows - p) as f64;
    let sigma_sq = r_sq / df;
    if !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        return f64::INFINITY;
    }
    fit.sigma_sq = sigma_sq;

    log_lzz_sq + log_lxx_sq + df * sigma_sq.ln()
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
    /// Shipped `lme.rs` coding: 0 = interior min, 1 = pinned at a variance
    /// boundary (counted converged), 2 = no accepted optimum — either a
    /// `MaxFunReached` cap-out (finite endpoint still reported below) or an
    /// optimizer/numerical failure (NaN-filled).
    pub boundary_hit: u8,
    /// Objective evaluations consumed (diagnostics only).
    pub n_eval: usize,
    /// Joint Wald-χ² over the target set (the shared `lme.rs` helper). Under
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
}

/// Fit by BOBYQA minimisation of the REML profiled deviance over the box-
/// bounded θ, with β̂ / σ̂² / Var(β̂_target) recovered once at θ̂.
///
/// Caller contract: `ws.suff` holds the accumulated rows (reset + add_rows
/// per dataset); `target_indices` index design columns.
///
/// `theta_start`: `None` → blind start (diagonals THETA0, off-diagonals 0,
/// the default for arbitrary provided bytes); `Some(θ₀)` → per-component spec-derived truth
/// start, `[primary, extras in declaration order]` (Y is always synthetic, so
/// true θ_g = τ_g/σ is known), each component clamped to THETA_TRUTH_FLOOR. A
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
    // floor; under the fixed RHO_BEGIN, PRIMA's start-projection may still
    // move a small start to rho_begin off the 0 bound — benign and
    // deterministic. The scaled schedule (rho_begin = 0.1·θ₀) that makes
    // small starts pay off is activation at workspace-construction time: rho
    // lives in the solver's construction-time Config and θ₀ is per-scenario,
    // so it belongs where the workspace is built per workload.
    match theta_start {
        Some(ts) => {
            debug_assert_eq!(ts.len(), theta.len());
            for (t, &v) in theta.iter_mut().zip(ts) {
                *t = v.max(THETA_TRUTH_FLOOR);
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
    let out = if two_stage {
        two_stage_minimize(suff, fit, theta, lower, upper)
    } else {
        solver.minimize(|xs| reml_deviance(xs, suff, fit), theta, lower, upper)
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
                    pinned_components |= 1u64 << k;
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

    // Rank guard on the p×p block — mirrors lme.rs's EPS_RANK min/max-diag
    // test on the pinning factor.
    let degenerate = !dev.is_finite() || chol_rank_deficient(fit.factor.as_ref(), p, EPS_RANK);
    if !has_endpoint || degenerate {
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
            joint_t_sq: f64::NAN,
            pinned_components: 0,
            deviance: f64::NAN,
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

    // Var(β̂_j) = σ̂²·‖L_XX⁻¹e_j‖² per target; t² = β̂²/Var — the lme.rs
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

    // Joint Wald-χ² over the target set — the shared lme.rs helper (promoted
    // pub(crate)). It re-Choleskys X'V⁻¹X internally, so hand it the product
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
        crate::lme::joint_wald_chi_sq(
            fit.joint_xtvix.as_ref(),
            &fit.betas,
            sigma_sq,
            target_indices,
            fit.joint_k_inv.as_mut(),
            fit.joint_sigma_t_chol.as_mut(),
            &mut fit.joint_rhs,
        )
    };

    LmmFit {
        sigma_sq,
        converged,
        boundary_hit: if converged { u8::from(pinned) } else { 2 },
        n_eval: out.n_eval,
        joint_t_sq,
        pinned_components,
        deviance: dev,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lme::{profiled_deviance, LmeSuffStats};
    use crate::test_support::{
        build_lme_scratch, extra_level_of_row, intercept_only_spec, model_atom, TestWs,
    };
    use crate::{Family, Grouping, GroupingRelation, ModelSpec, ReStructure, Sizing};

    /// Grammar: `<mult>n<add>`, e.g. `2n1` = 2·n+1, `1.5n1` = ⌈1.5·n⌉+1, `1n2` =
    /// n+2. Result clamped to BOBYQA's legal `[n+2, (n+1)(n+2)/2]`: flat constants
    /// (and small-n underflow) violate the bounds, so the hook clamps rather than
    /// panic deep in `Bobyqa::new`.
    #[test]
    fn npt_formula_parses_and_clamps() {
        assert_eq!(npt_from_formula("2n1", 36), Some(73));
        assert_eq!(npt_from_formula("1.5n1", 36), Some(55)); // ⌈54⌉+1
        assert_eq!(npt_from_formula("1n2", 36), Some(38));
        assert_eq!(npt_from_formula("1.5n1", 2), Some(4)); // ⌈3⌉+1 = 4 = n+2 ✓
        assert_eq!(npt_from_formula("3n0", 2), Some(6)); // 6 = (n+1)(n+2)/2 cap
        assert_eq!(npt_from_formula("1n0", 3), Some(5)); // clamped up to n+2
        assert_eq!(npt_from_formula("500n500", 8), Some(45)); // max_fun grammar reuses
                                                              // the parser; the CLAMP is
                                                              // npt-specific — see Step 3
        assert_eq!(npt_from_formula("garbage", 8), None);
        assert_eq!(npt_from_formula("73", 8), None); // flat constants rejected
    }

    #[test]
    fn formula_eval_unclamped() {
        assert_eq!(eval_formula("500n500", 8), Some(4500));
        assert_eq!(eval_formula("2n1", 36), Some(73));
        assert_eq!(eval_formula("n2", 8), None); // mult is mandatory: write 1n2
    }

    /// Deterministic pseudo-data (NR LCG), uniform in (−1, 1).
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }

    /// n=48, p=3 (intercept + x1 + x2), 6 clusters,
    /// y = 0.5 + 0.4·x1 − 0.2·x2 + u_c + 0.8·e.
    fn hand_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 42u64;
        let u_c: Vec<f64> = (0..n_clusters).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 3);
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % n_clusters;
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            let x2 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            x[(i, 2)] = x2;
            y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + u_c[c] + 0.8 * lcg(&mut st);
        }
        (x, y, ids)
    }

    /// Populate a fresh `TestWs`'s lme suff-stats from a dataset and
    /// return it (helper shared by the deviance + fit parity tests).
    fn shipped_workspace(x: &Mat<f64>, y: &[f64], ids: &[u32], n_clusters: u32) -> TestWs {
        let mut ws = TestWs::new(x.nrows(), x.ncols(), n_clusters as usize);
        ws.reset_lme_suff_stats();
        let mut suff = LmeSuffStats {
            xtx: ws.lme_xtx.as_mut(),
            xty: &mut ws.lme_xty,
            yty: &mut ws.lme_yty,
            sum_xc: ws.lme_sum_xc.as_mut(),
            sum_yc: &mut ws.lme_sum_yc,
            cluster_sizes: &mut ws.lme_cluster_sizes,
            n_clusters_seen: &mut ws.lme_n_clusters_seen,
            panel_x: &mut ws.panel_x,
            panel_y: &mut ws.panel_y,
        };
        suff.add_rows(x.as_ref(), y, ids);
        ws
    }

    /// Same quantity, two factorizations — both return
    /// log|V| + log|X'V⁻¹X| + (N−P)·log σ̂², so agreement is FP-level
    /// (≤ 1e-9 rel), not up-to-a-constant. THE formulation proof, held on
    /// every θ probed.
    #[test]
    fn deviance_matches_shipped_across_theta() {
        let (x, y, ids) = hand_dataset();
        let mut ws = shipped_workspace(&x, &y, &ids, 6);
        let mut scratch = build_lme_scratch(&mut ws, 48, 6);

        let mut suff = LmmSuffStats::new(3, 6);
        suff.add_rows(x.as_ref(), &y, &ids);
        let mut fit = LmmFitScratch::new(3, 6);
        let mut fit_c = LmmFitScratch::new(3, 6);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));

        for &theta in &[0.0, 1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0] {
            let dev_ship = profiled_deviance(theta, &mut scratch);
            let dev_gen = reml_deviance(&[theta], &suff, &mut fit);
            assert!(dev_ship.is_finite() && dev_gen.is_finite(), "θ={theta}");
            let tol = 1e-9 * dev_ship.abs().max(1.0);
            assert!(
                (dev_ship - dev_gen).abs() <= tol,
                "θ={theta}: shipped {dev_ship} vs general {dev_gen}"
            );
            // Collapse arm — reassociation band vs the general loop incl. θ=0.
            let dev_c = reml_deviance(&[theta], &suff, &mut fit_c);
            let band = 1e-9 * dev_gen.abs().max(1.0);
            assert!(
                (dev_c - dev_gen).abs() <= band,
                "θ={theta}: collapse {dev_c} vs general {dev_gen}"
            );
        }
    }

    /// All scratch is overwritten per call — re-evaluating a θ after an
    /// intervening different-θ call reproduces bit-identical deviance and σ̂²
    /// (mirrors lme.rs's stale-state test).
    #[test]
    fn reml_deviance_overwrites_state() {
        let (x, y, ids) = hand_dataset();
        let mut suff = LmmSuffStats::new(3, 6);
        suff.add_rows(x.as_ref(), &y, &ids);
        let mut fit = LmmFitScratch::new(3, 6);

        let dev_a = reml_deviance(&[1.0], &suff, &mut fit);
        let sig_a = fit.sigma_sq;
        let _ = reml_deviance(&[2.0], &suff, &mut fit);
        let dev_b = reml_deviance(&[1.0], &suff, &mut fit);
        let sig_b = fit.sigma_sq;
        assert_eq!(dev_a, dev_b, "deviance(θ=1) must be reproducible");
        assert_eq!(sig_a, sig_b, "σ̂²(θ=1) must be reproducible");
    }

    /// Correctness prerequisite for workspace reuse across simulation draws:
    /// `suff.reset()` followed by a refill on a DIFFERENT dataset (same shape, different `y`)
    /// must reproduce a freshly-constructed workspace's fit bit-for-bit. Same
    /// buffers + same code path ⇒ identical float reassociation, so the
    /// assertion is exact `==`, not a tolerance band. If this fails, `reset()`
    /// (src/lmm.rs) leaves some `LmmSuffStats` field stale across datasets.
    #[test]
    fn reused_workspace_refill_matches_fresh() {
        let (x, y_a, ids) = hand_dataset();
        // B: same shape/ids as A, deterministically different y (constant shift
        // + a fixed rescale) — not randomized, so the comparison stays exact.
        let y_b: Vec<f64> = y_a.iter().map(|&v| 1.7 - 0.3 * v).collect();
        let targets: Vec<u32> = vec![1, 2];

        // Fresh workspace, fit B directly.
        let mut ws_fresh = LmmWorkspace::new(3, 6);
        ws_fresh.suff.reset();
        ws_fresh.suff.add_rows(x.as_ref(), &y_b, &ids);
        let fit_fresh = fit_lmm(&mut ws_fresh, &targets, None);

        // Reused workspace: fit A first, reset, refill with B, fit again.
        let mut ws_reused = LmmWorkspace::new(3, 6);
        ws_reused.suff.reset();
        ws_reused.suff.add_rows(x.as_ref(), &y_a, &ids);
        let _ = fit_lmm(&mut ws_reused, &targets, None);
        ws_reused.suff.reset();
        ws_reused.suff.add_rows(x.as_ref(), &y_b, &ids);
        let fit_reused = fit_lmm(&mut ws_reused, &targets, None);

        assert_eq!(
            fit_fresh.deviance, fit_reused.deviance,
            "deviance must be bit-identical after reset+refill on new data"
        );
        assert_eq!(
            ws_fresh.fit.betas, ws_reused.fit.betas,
            "betas must be bit-identical after reset+refill on new data"
        );
        assert_eq!(
            ws_fresh.fit.var_diag, ws_reused.fit.var_diag,
            "var_diag must be bit-identical after reset+refill on new data"
        );
    }

    /// Exercises the plateau policy: a `MaxFunReached` cap-out must still
    /// report the honest finite endpoint
    /// (β̂/σ̂²/SE/deviance), with `converged = false` and `boundary_hit == 2`
    /// (not the accepted-boundary 1). Forces the cap by swapping in a solver
    /// whose `max_fun` is the legal minimum (`npt + 1`) — one eval past the
    /// initial model build, nowhere near this dataset's optimum — bypassing
    /// `LMM_MAX_FUN_FORMULA` entirely so the test carries no process-env race.
    #[test]
    fn maxfun_cap_reports_honest_endpoint() {
        let (x, y, ids) = hand_dataset();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws = LmmWorkspace::new(3, 6);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let n_theta = ws.theta.len();
        let npt = 2 * n_theta + 1; // n_theta == 1 here: PRIMA's minimum npt
        let config = Config {
            npt,
            max_fun: npt + 1,
            ..Config::new(n_theta)
        };
        ws.solver = Bobyqa::new(n_theta, config).expect("legal minimal config");

        let fit = fit_lmm(&mut ws, &targets, None);

        assert!(!fit.converged, "capped fit must not report converged");
        assert_eq!(
            fit.boundary_hit, 2,
            "capped fit must not migrate into the accepted-boundary code"
        );
        assert_eq!(
            fit.pinned_components, 0,
            "a capped endpoint is a point, not an accepted boundary"
        );
        assert!(
            fit.deviance.is_finite(),
            "plateau policy: capped endpoint must report a finite deviance"
        );
        assert!(
            fit.sigma_sq.is_finite(),
            "plateau policy: capped endpoint must report a finite sigma_sq"
        );
        assert!(
            fit.joint_t_sq.is_finite(),
            "plateau policy: capped endpoint must report a finite joint_t_sq"
        );
        for &tj in &targets {
            assert!(
                ws.fit.betas[tj as usize].is_finite(),
                "plateau policy: capped endpoint must not NaN-fill beta"
            );
        }
        assert!(fit.n_eval <= npt + 1, "n_eval must reflect the forced cap");

        // Pinned values are the deterministic truncated-BOBYQA endpoint (hand_dataset,
        // max_fun = npt+1) — a regression lock, not an external oracle: any solver-path
        // change that moves the honest cap-out endpoint should fail this test.
        let rel = |got: f64, want: f64| (got - want).abs() / want.abs().max(1e-12);
        assert!(
            rel(fit.deviance, -62.08988134487164) < 1e-6,
            "deviance = {}",
            fit.deviance
        );
        assert!(
            rel(fit.sigma_sq, 0.16043347869402982) < 1e-6,
            "sigma_sq = {}",
            fit.sigma_sq
        );
        assert!(
            rel(fit.joint_t_sq, 14.568949550460516) < 1e-6,
            "joint_t_sq = {}",
            fit.joint_t_sq
        );
        let want_betas = [
            0.4691004480864937,
            0.26391548909385104,
            -0.33307894295165125,
        ];
        for (j, &wb) in want_betas.iter().enumerate() {
            assert!(
                rel(ws.fit.betas[j], wb) < 1e-6,
                "betas[{j}] = {}, want {}",
                ws.fit.betas[j],
                wb
            );
        }
    }

    /// End-to-end q=1 parity on the hand dataset: the general machine vs the
    /// shipped `lme_fit` on the same suff-stats bytes, at the amended
    /// tolerances (rel 1e-4, abs floors β̂ 1e-5 / stat 1e-4 — the measured Brent
    /// θ̂-placement-noise floor).
    #[test]
    fn fit_matches_shipped_lme_fit_on_hand_dataset() {
        let (x, y, ids) = hand_dataset();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws_ship = shipped_workspace(&x, &y, &ids, 6);
        let scratch = build_lme_scratch(&mut ws_ship, 48, 6);
        let ship = crate::lme::lme_fit(x.as_ref(), &y, &ids, &targets, None, scratch);
        assert!(ship.converged);

        let mut ws = LmmWorkspace::new(3, 6);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm(&mut ws, &targets, None);
        assert!(fit.converged);
        assert!(fit.boundary_hit <= 1);

        for j in 0..3 {
            let (a, b) = (ship.betas[j], ws.fit.betas[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
                "β[{j}]: {a} vs {b}"
            );
        }
        for &tj in &targets {
            let a = ship.t_sq[tj as usize].sqrt();
            let b = ws.fit.t_sq[tj as usize].sqrt();
            let d = (a - b).abs();
            assert!(
                d <= 1e-4 || d <= 1e-4 * a.abs().max(b.abs()),
                "stat[{tj}]: {a} vs {b}"
            );
        }
        let (a, b) = (ship.joint_t_sq, fit.joint_t_sq);
        let d = (a - b).abs();
        assert!(
            d <= 1e-4 || d <= 1e-4 * a.abs().max(b.abs()),
            "joint: {a} vs {b}"
        );
    }

    /// Deterministic pin: y carries NO between-cluster signal by construction —
    /// residuals alternate ±0.8 within each cluster with equal counts, so every
    /// cluster's residual sum is exactly 0 and the REML deviance is minimized at
    /// θ = 0. The fit must pin (boundary_hit == 1), write θ̂ = exactly 0.0, and
    /// count as converged: zero variance is a legitimate boundary optimum, not
    /// a failure to fit.
    #[test]
    fn zero_between_cluster_variance_pins_at_exactly_zero() {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 7u64;
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            ids[i] = (i % n_clusters) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            // i/n_clusters cycles 0..8 within each cluster: 4 even, 4 odd ⇒
            // the ±0.8 residuals cancel exactly per cluster.
            let e = if (i / n_clusters) % 2 == 0 { 0.8 } else { -0.8 };
            y[i] = 0.5 + 0.4 * x1 + e;
        }
        let mut ws = LmmWorkspace::new(2, n_clusters);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        assert_eq!(fit.boundary_hit, 1);
        assert_eq!(ws.theta[0], 0.0, "pin must be exact 0.0, not merely small");
        assert!(ws.fit.betas[1].is_finite());
    }

    /// Rank deficiency fails cleanly: x2 = 0.1·x1 (the scaled-duplicate fixture —
    /// exact duplicates can slip through faer's llt grey zone) must produce a
    /// non-converged, NaN-filled fit with boundary_hit == 2.
    #[test]
    fn rank_deficient_design_fails_cleanly() {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 11u64;
        let mut x = Mat::<f64>::zeros(n, 3);
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            ids[i] = (i % n_clusters) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            x[(i, 2)] = 0.1 * x1; // 0.1-scaled duplicate → guaranteed non-convergence
            y[i] = 0.5 + 0.4 * x1 + 0.8 * lcg(&mut st);
        }
        let mut ws = LmmWorkspace::new(3, n_clusters);
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let fit = fit_lmm(&mut ws, &[1, 2], None);
        assert!(!fit.converged);
        assert_eq!(fit.boundary_hit, 2);
        assert!(ws.fit.betas.iter().all(|b| b.is_nan()));
        assert!(ws.fit.t_sq[1].is_nan() && ws.fit.t_sq[2].is_nan());
    }

    /// A truth-started fit (`theta_start: Some`) reaches the same answer as the
    /// blind fit on the same bytes — and Some(0.0) exercises the
    /// THETA_TRUTH_FLOOR clamp rather than starting on the 0 boundary.
    /// Bands are the amended floors: two BOBYQA runs from different
    /// starts each place θ̂ within the rho_end band of the same minimum.
    #[test]
    fn theta_start_some_matches_blind_fit() {
        let (x, y, ids) = hand_dataset();
        let targets: Vec<u32> = vec![1, 2];

        let mut ws_blind = LmmWorkspace::new(3, 6);
        ws_blind.suff.add_rows(x.as_ref(), &y, &ids);
        let blind = fit_lmm(&mut ws_blind, &targets, None);
        assert!(blind.converged);

        for start in [[0.0], [0.6]] {
            let mut ws = LmmWorkspace::new(3, 6);
            ws.suff.add_rows(x.as_ref(), &y, &ids);
            let fit = fit_lmm(&mut ws, &targets, Some(&start));
            assert!(fit.converged, "start {start:?}");
            for j in 0..3 {
                let (a, b) = (ws_blind.fit.betas[j], ws.fit.betas[j]);
                let d = (a - b).abs();
                assert!(
                    d <= 1e-5 || d <= 1e-4 * a.abs().max(b.abs()),
                    "start {start:?} β[{j}]: blind {a} vs started {b}"
                );
            }
        }
    }

    /// Bounded-allocation warm-path twin of lme.rs's
    /// `lme_fit_warm_path_bounded_alloc`. Marked #[ignore] because dhat measures
    /// process-wide allocations and concurrent tests contaminate the count:
    ///   cargo test -p glmm --features alloc-tests lmm_fit_warm_path_bounded_alloc -- --ignored --test-threads=1
    ///
    /// BOUND locks the measured warm-path block count. LmmWorkspace itself is
    /// allocation-free across fits (Bobyqa::new is the only solver allocation,
    /// done once). On the faer kernel the per-call blocks are `llt` internals —
    /// ~2 per deviance evaluation (15.1–15.7 evals/fit at rho_end 1e-6, the
    /// measured mean), the same acceptance the shipped path's 26
    /// blocks/call carry; if a future faer version changes its Cholesky
    /// internals, update the bound — do not relax it. A hand-rolled
    /// owned-kernel replacement for faer's `llt` was tried and rejected: its
    /// wasm `f64::ln` took a different ULP path than the native build (the
    /// factorization itself was fine), which broke cross-platform
    /// bit-equality. The faer bound stays the locked steady state.
    #[cfg(feature = "alloc-tests")]
    #[test]
    #[ignore]
    fn lmm_fit_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        const BOUND: u64 = 4800; // Measured 4600 (this machine) — ~46 blocks/fit of faer `llt` internals on the family-blocked q=1 path (one m×m tail llt per eval). `fit_lmm` no longer allocates per fit (the diagonal_theta index map is cached once on LmmGroupings), so this count is purely faer's Cholesky internals — faer-version/machine specific. q=1 deviance is byte-identical to the hand-rolled augmented-factor deviance (held by the lmm_parity corpus + golden_rng), so the eval trajectory is unchanged; the count differs from the prior 3804 only because faer's blocked llt allocates more per eval than the hand-rolled augmented factor. If faer changes its Cholesky internals, update — do not relax.

        let (x, y, ids) = hand_dataset();
        let targets: Vec<u32> = vec![1, 2];
        let mut ws = LmmWorkspace::new(3, 6);

        // Warmup drives one-time setup outside the profiler window.
        ws.suff.reset();
        ws.suff.add_rows(x.as_ref(), &y, &ids);
        let _ = fit_lmm(&mut ws, &targets, None);

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.suff.reset();
            ws.suff.add_rows(x.as_ref(), &y, &ids);
            let _ = fit_lmm(&mut ws, &targets, None);
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND,
            "fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND
        );
    }

    // -----------------------------------------------------------------------
    // Multi-grouping: layout-true datasets, suff-stats, family-blocked
    // deviance vs a brute-force n×n oracle, and end-to-end fits.
    // -----------------------------------------------------------------------

    /// Layout-true multi-grouping dataset: primary S=6, crossed I=4, nested
    /// np=2 (optional), p=3, n = n_blocks·atom rows. Ids come from the
    /// contract layout helpers — the same functions the workspace uses.
    #[allow(clippy::type_complexity)]
    fn multi_dataset(
        with_nested: bool,
        n_blocks: usize,
    ) -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<Vec<u32>>, ModelSpec) {
        let mut cluster = intercept_only_spec(Sizing::FixedClusters { n_clusters: 6 });
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            slopes: vec![],
        });
        if with_nested {
            cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
                relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                slopes: vec![],
            });
        }
        let n = n_blocks * model_atom(&cluster);
        let mut st = 99u64;
        let u_p: Vec<f64> = (0..6).map(|_| 0.5 * lcg(&mut st)).collect();
        let u_x: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
        let u_n: Vec<f64> = (0..12).map(|_| 0.3 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 3);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let n_extras = cluster.re.as_ref().unwrap().extra_groupings.len();
        let mut eids: Vec<Vec<u32>> = vec![vec![0u32; n]; n_extras];
        for i in 0..n {
            pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
            #[allow(clippy::needless_range_loop)]
            for g in 0..n_extras {
                eids[g][i] = extra_level_of_row(&cluster, g, i) as u32;
            }
            let x1 = lcg(&mut st);
            let x2 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            x[(i, 2)] = x2;
            y[i] = 0.5 + 0.4 * x1 - 0.2 * x2
                + u_p[pid[i] as usize]
                + u_x[eids[0][i] as usize]
                + if with_nested {
                    u_n[eids[1][i] as usize]
                } else {
                    0.0
                }
                + 0.8 * lcg(&mut st);
        }
        (x, y, pid, eids, cluster)
    }

    /// `diagonal_theta` / `n_theta` / `k_family` at q_p ∈ {1, 2, 3} — locks
    /// the column-major vech ordering. q_p=1 must reproduce the intercept-only
    /// baseline values; q_p>1 tests the standalone slope branch (no extras, k_crossed=0).
    #[test]
    fn groupings_vech_layout() {
        let sizing = Sizing::FixedClusters { n_clusters: 4 };
        let base = intercept_only_spec(sizing.clone());

        // q_p = 1 (intercept-only): shape must be unchanged.
        let g1 = LmmGroupings::from_cluster_spec(&base, 40, &[]);
        assert_eq!(g1.n_theta(), 1);
        assert_eq!(g1.k_family(), 4); // 4 clusters × 1
        assert_eq!(g1.diagonal_theta(), &[0][..]);

        // q_p = 2 (1 slope): vech([σ_00, σ_10, σ_11]) length 3; diagonals at 0, 2.
        let mut spec2 = base.clone();
        spec2.re.as_mut().unwrap().slopes.push(1);
        let g2 = LmmGroupings::from_cluster_spec(&spec2, 40, &[1]);
        assert_eq!(g2.primary_q, 2);
        assert_eq!(g2.n_theta(), 3); // 2·3/2 = 3
        assert_eq!(g2.k_family(), 8); // 4 clusters × 2
        assert_eq!(g2.k_total, 8);
        assert_eq!(g2.diagonal_theta(), &[0, 2][..]); // off-diagonal vech[1]=1 excluded

        // q_p = 3 (2 slopes): vech([σ_00, σ_10, σ_11, σ_20, σ_21, σ_22]) length 6; diagonals at 0, 3, 5.
        let mut spec3 = base.clone();
        spec3.re.as_mut().unwrap().slopes.push(1);
        spec3.re.as_mut().unwrap().slopes.push(2);
        let g3 = LmmGroupings::from_cluster_spec(&spec3, 40, &[1, 2]);
        assert_eq!(g3.primary_q, 3);
        assert_eq!(g3.n_theta(), 6); // 3·4/2 = 6
        assert_eq!(g3.k_family(), 12); // 4 clusters × 3
        assert_eq!(g3.k_total, 12);
        assert_eq!(g3.diagonal_theta(), &[0, 3, 5][..]);
    }

    /// Suff-stats bookkeeping on a hand-checkable block: counts per RE column,
    /// per-column sums, crossed cross-counts.
    #[test]
    fn suff_stats_multi_accumulators() {
        let (x, y, pid, eids, cluster) = multi_dataset(true, 1); // one atom block, n=48
        let g = LmmGroupings::from_cluster_spec(&cluster, 48, &[]);
        assert_eq!(g.n_primary, 6);
        assert_eq!(g.nested_per_parent, 2);
        assert_eq!(g.k_family(), 18); // 6 + 6·2
        assert_eq!(g.k_total, 22); // + 4 crossed
        assert_eq!(g.n_theta(), 3);
        let mut suff = LmmSuffStats::with_groupings(3, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        // One full-factorial block: every primary level has 8 rows, every
        // child 4, every crossed level 12.
        for f in 0..6 {
            assert_eq!(suff.counts[f], 8.0);
        }
        for c in 6..18 {
            assert_eq!(suff.counts[c], 4.0);
        }
        for b in 18..22 {
            assert_eq!(suff.counts[b], 12.0);
        }
        // Crossed co-occurrence: each (primary, crossed) pair shares exactly
        // 2 rows in a full factorial of 6·4·2.
        assert_eq!(suff.zx[(0, 0)], 2.0);
        assert_eq!(suff.zx[(5, 3)], 2.0);
        // Same-factor crossed pairs never co-occur.
        assert_eq!(suff.zx[(18, 1)], 0.0);
        // Intercept column sum = row count per level.
        assert!((suff.s[(0, 0)] - 8.0).abs() < 1e-12);
    }

    /// Textbook REML deviance on the explicit n×n V — the oracle for the
    /// family-blocked elimination. dev = ln|V| + ln|X'V⁻¹X| + (N−P)·ln σ̂²,
    /// σ̂² = (y'V⁻¹y − β̂'X'V⁻¹y)/(N−P).  `groups[g]` = grouping g's global
    /// level ids (primary first); `theta[g]` the matching component.
    fn brute_force_deviance(theta: &[f64], x: &Mat<f64>, y: &[f64], groups: &[&[u32]]) -> f64 {
        use faer::linalg::solvers::Solve;
        let n = x.nrows();
        let p = x.ncols();
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] = 1.0;
        }
        for (g, ids) in groups.iter().enumerate() {
            let t2 = theta[g] * theta[g];
            for i in 0..n {
                for j in 0..n {
                    if ids[i] == ids[j] {
                        v[(i, j)] += t2;
                    }
                }
            }
        }
        let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
        let mut log_det_v = 0.0;
        for i in 0..n {
            log_det_v += vc.L()[(i, i)].ln();
        }
        let log_det_v = 2.0 * log_det_v;
        let mut vix = (*x).clone();
        vc.solve_in_place(vix.as_mut());
        let mut viy = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            viy[(i, 0)] = y[i];
        }
        vc.solve_in_place(viy.as_mut());
        let mut xtvix = Mat::<f64>::zeros(p, p);
        let mut xtviy = vec![0.0; p];
        for a in 0..p {
            for b in 0..p {
                let mut acc = 0.0;
                for i in 0..n {
                    acc += x[(i, a)] * vix[(i, b)];
                }
                xtvix[(a, b)] = acc;
            }
            let mut acc = 0.0;
            for i in 0..n {
                acc += x[(i, a)] * viy[(i, 0)];
            }
            xtviy[a] = acc;
        }
        let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
        let mut log_det_k = 0.0;
        for a in 0..p {
            log_det_k += kc.L()[(a, a)].ln();
        }
        let log_det_k = 2.0 * log_det_k;
        let mut beta = Mat::<f64>::zeros(p, 1);
        for a in 0..p {
            beta[(a, 0)] = xtviy[a];
        }
        kc.solve_in_place(beta.as_mut());
        let mut ytviy = 0.0;
        for i in 0..n {
            ytviy += y[i] * viy[(i, 0)];
        }
        let mut bxy = 0.0;
        for a in 0..p {
            bxy += beta[(a, 0)] * xtviy[a];
        }
        let df = (n - p) as f64;
        let sigma_sq = (ytviy - bxy) / df;
        log_det_v + log_det_k + df * sigma_sq.ln()
    }

    fn assert_deviance_matches_oracle(with_nested: bool, thetas: &[Vec<f64>]) {
        let (x, y, pid, eids, cluster) = multi_dataset(with_nested, 2);
        let n = x.nrows();
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut suff = LmmSuffStats::with_groupings(3, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut fit = LmmFitScratch::with_groupings(3, &gref);
        let mut fit_c = LmmFitScratch::with_groupings(3, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));
        // Oracle wants global ids per grouping.
        let mut groups: Vec<&[u32]> = vec![&pid];
        for e in &eids {
            groups.push(e);
        }
        for th in thetas {
            let dev = reml_deviance(th, &suff, &mut fit);
            let oracle = brute_force_deviance(th, &x, &y, &groups);
            assert!(dev.is_finite(), "θ={th:?}");
            let tol = 1e-8 * oracle.abs().max(1.0);
            assert!(
                (dev - oracle).abs() <= tol,
                "θ={th:?}: family-blocked {dev} vs oracle {oracle}"
            );
            // Collapse arm: same θ through the balanced path — reassociation
            // band vs the loop, oracle band absolute.
            let dev_c = reml_deviance(th, &suff, &mut fit_c);
            let band = 1e-9 * dev.abs().max(1.0);
            assert!(
                (dev_c - dev).abs() <= band,
                "θ={th:?}: collapse {dev_c} vs loop {dev}"
            );
            assert!(
                (dev_c - oracle).abs() <= tol,
                "θ={th:?}: collapse vs oracle"
            );
        }
    }

    #[test]
    fn crossed_deviance_matches_brute_force() {
        assert_deviance_matches_oracle(
            false,
            &[
                vec![0.5, 0.3],
                vec![1.0, 1.0],
                vec![2.0, 0.1],
                vec![0.0, 0.7],
                vec![1e-3, 1e-3],
            ],
        );
    }

    #[test]
    fn crossed_plus_nested_deviance_matches_brute_force() {
        assert_deviance_matches_oracle(
            true,
            &[
                vec![0.5, 0.3, 0.2],
                vec![1.0, 1.0, 1.0],
                vec![0.0, 0.5, 0.9],
                vec![2.0, 0.05, 0.4],
            ],
        );
    }

    /// Unbalanced counts must take the legacy loop byte-for-byte: a failed
    /// precompute leaves collapse_n_active = 0 and the eval path untouched.
    #[test]
    fn unbalanced_counts_fall_back_byte_identical() {
        let (x, y, pid, eids, cluster) = multi_dataset(true, 2);
        let n = x.nrows() - 1; // truncate one row — last cluster short
        let g = LmmGroupings::from_cluster_spec(&cluster, x.nrows(), &[]);
        let mut suff = LmmSuffStats::with_groupings(3, g);
        let eids_t: Vec<Vec<u32>> = eids.iter().map(|e| e[..n].to_vec()).collect();
        suff.add_rows_multi(x.as_ref().subrows(0, n), &y[..n], &pid[..n], &eids_t, None);
        let gref = LmmGroupings::from_cluster_spec(&cluster, x.nrows(), &[]);
        let mut fit_a = LmmFitScratch::with_groupings(3, &gref);
        let mut fit_b = LmmFitScratch::with_groupings(3, &gref);
        assert!(!precompute_balanced_collapse(&suff, &mut fit_b));
        for th in [[0.5, 0.3, 0.2], [1.0, 1.0, 1.0], [0.0, 0.5, 0.9]] {
            let a = reml_deviance(&th, &suff, &mut fit_a);
            let b = reml_deviance(&th, &suff, &mut fit_b);
            assert_eq!(a.to_bits(), b.to_bits(), "θ={th:?}");
        }
    }

    /// Off-grid N under `FixedSize`: row 17 of 18 sits in cluster 4, so five
    /// primary levels exist and the nested block must cover all five parents.
    #[test]
    fn fixed_size_off_grid_n_keeps_the_partial_trailing_cluster() {
        let mut cluster = intercept_only_spec(Sizing::FixedSize { cluster_size: 4 });
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            slopes: vec![],
        });
        let n = 18;
        let sizing = &cluster.re.as_ref().unwrap().sizing;
        assert_eq!(sizing.cluster_of_row(n - 1), 4);
        assert_eq!(sizing.n_clusters_at(n), 5);
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        assert_eq!(g.n_primary, 5);
        assert_eq!(g.n_primary * g.nested_per_parent, 10);
        assert_eq!(g.k_total, 15);
    }

    /// Nested-only in Regime B — the path with NO crossed tail (zx is 0×0)
    /// and parents that grow with N.
    #[test]
    fn nested_regime_b_deviance_matches_brute_force() {
        let mut cluster = intercept_only_spec(Sizing::FixedSize { cluster_size: 8 });
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            slopes: vec![],
        });
        let n = 4 * model_atom(&cluster); // 64
        let mut st = 7u64;
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut cid = vec![0u32; n];
        let u_p: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
        let u_c: Vec<f64> = (0..16).map(|_| 0.3 * lcg(&mut st)).collect();
        for i in 0..n {
            pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
            cid[i] = extra_level_of_row(&cluster, 0, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] =
                0.5 + 0.4 * x1 + u_p[pid[i] as usize] + u_c[cid[i] as usize] + 0.8 * lcg(&mut st);
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut suff = LmmSuffStats::with_groupings(2, g);
        let eids = vec![cid.clone()];
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));
        for th in [[0.6, 0.4], [1.0, 1.0], [0.2, 0.0], [0.0, 0.0]] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &cid]);
            let tol = 1e-8 * oracle.abs().max(1.0);
            assert!((dev - oracle).abs() <= tol, "θ={th:?}: {dev} vs {oracle}");
            // Collapse arm — reassociation band vs the loop incl. the θ=0 edge.
            let dev_c = reml_deviance(&th, &suff, &mut fit_c);
            let band = 1e-9 * dev.abs().max(1.0);
            assert!(
                (dev_c - dev).abs() <= band,
                "θ={th:?}: collapse {dev_c} vs {dev}"
            );
        }
    }

    /// Balanced-collapse applicability: balanced intercept designs precompute,
    /// slope groupings and unbalanced counts fall back.
    #[test]
    fn balanced_collapse_applicability() {
        // Balanced: the regime-B nested dataset (atom-multiple by construction).
        let mut cluster = intercept_only_spec(Sizing::FixedSize { cluster_size: 8 });
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
            slopes: vec![],
        });
        let n = 4 * model_atom(&cluster); // 64
        let max_n = 2 * n; // workspace sized for a larger grid top — active PREFIX
        let mut st = 7u64;
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut cid = vec![0u32; n];
        for i in 0..n {
            pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
            cid[i] = extra_level_of_row(&cluster, 0, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5 + 0.4 * x1 + 0.8 * lcg(&mut st);
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, max_n, &[]);
        let n_primary = g.n_primary;
        let mut suff = LmmSuffStats::with_groupings(2, g);
        let eids = vec![cid.clone()];
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        let gref = LmmGroupings::from_cluster_spec(&cluster, max_n, &[]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit));
        assert_eq!(fit.collapse_n_active, n / 8);
        assert!(fit.collapse_n_active < n_primary); // genuinely a prefix

        // Unbalanced: drop the last row — the trailing cluster is short.
        let mut suff_u =
            LmmSuffStats::with_groupings(2, LmmGroupings::from_cluster_spec(&cluster, max_n, &[]));
        let eids_u = vec![cid[..n - 1].to_vec()];
        suff_u.add_rows_multi(
            x.as_ref().subrows(0, n - 1),
            &y[..n - 1],
            &pid[..n - 1],
            &eids_u,
            None,
        );
        assert!(!precompute_balanced_collapse(&suff_u, &mut fit));
        assert_eq!(fit.collapse_n_active, 0);

        // Slope path: never applicable — populated, balanced data, so the
        // rejection is the q_p guard, not the empty-suff early-out (balanced
        // slope counts would otherwise pass the count checks).
        let (xs, ys, ids_s) = slope_dataset();
        let gs = slope_groupings();
        let mut suff_s = LmmSuffStats::with_groupings(2, slope_groupings());
        suff_s.add_rows_multi(xs.as_ref(), &ys, &ids_s, &[], None);
        let mut fit_s = LmmFitScratch::with_groupings(2, &gs);
        assert!(!precompute_balanced_collapse(&suff_s, &mut fit_s));
    }

    /// Balanced collapse with prior weights: constant w ≡ 2 preserves the
    /// per-cluster `counts` equality (`counts[f] = 2·n_f`, still exactly equal
    /// across the balanced prefix), so the collapse must STILL trigger — and
    /// the collapse-taken weighted fit must reproduce the unweighted one's
    /// β/SE/tau2 (θ̃ = √c·θ maps the weighted profiled deviance onto the
    /// unweighted one; θ̂² scales by 1/c, σ̂² by c, tau2 = θ²σ̂² invariant).
    /// Both fits take the collapse branch (asserted below), so agreement is a
    /// numeric check of the collapse kernel consuming weighted Grams, not just
    /// of the accumulator.
    #[test]
    fn balanced_collapse_weighted_fit_invariant() {
        let (x, y, ids) = hand_dataset(); // balanced: 6 clusters × 8 rows
        let n = x.nrows();
        let targets: Vec<u32> = vec![1, 2];
        let w = vec![2.0f64; n];

        let mut ws_w = LmmWorkspace::new(3, 6);
        ws_w.suff
            .add_rows_multi(x.as_ref(), &y, &ids, &[], Some(&w));
        assert!(
            precompute_balanced_collapse(&ws_w.suff, &mut ws_w.fit),
            "constant weights keep exact per-cluster counts equality"
        );
        assert_eq!(ws_w.fit.collapse_n_active, 6);
        let fit_w = fit_lmm(&mut ws_w, &targets, None);
        assert!(fit_w.converged);

        let mut ws_u = LmmWorkspace::new(3, 6);
        ws_u.suff.add_rows(x.as_ref(), &y, &ids);
        let fit_u = fit_lmm(&mut ws_u, &targets, None);
        assert!(fit_u.converged);

        // Two independent BOBYQA runs agree to the rho_end floor, not machine
        // precision — same 1e-6 relative band as the fit.rs invariance tests.
        for j in 0..3 {
            let (a, b) = (ws_u.fit.betas[j], ws_w.fit.betas[j]);
            assert!(
                (a - b).abs() / a.abs() < 1e-6,
                "β[{j}] unweighted {a} vs w≡2 {b}"
            );
        }
        for &tj in &targets {
            let (a, b) = (
                ws_u.fit.var_diag[tj as usize].sqrt(),
                ws_w.fit.var_diag[tj as usize].sqrt(),
            );
            assert!(
                (a - b).abs() / a < 1e-6,
                "se[{tj}] unweighted {a} vs w≡2 {b}"
            );
        }
        let (tu, tw) = (
            ws_u.theta[0] * ws_u.theta[0] * fit_u.sigma_sq,
            ws_w.theta[0] * ws_w.theta[0] * fit_w.sigma_sq,
        );
        assert!(
            (tu - tw).abs() / tu < 1e-6,
            "tau2 unweighted {tu} vs w≡2 {tw}"
        );
    }

    /// Two crossed factors — the dense cross-factor coupling block.
    #[test]
    fn two_crossed_factors_deviance_matches_brute_force() {
        let mut cluster = intercept_only_spec(Sizing::FixedClusters { n_clusters: 3 });
        for k in [4u32, 2u32] {
            cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
                relation: GroupingRelation::Crossed { n_clusters: k },
                slopes: vec![],
            });
        }
        let n = 2 * model_atom(&cluster); // 48
        let mut st = 21u64;
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut e0 = vec![0u32; n];
        let mut e1 = vec![0u32; n];
        let u_p: Vec<f64> = (0..3).map(|_| 0.5 * lcg(&mut st)).collect();
        let u_a: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
        let u_b: Vec<f64> = (0..2).map(|_| 0.3 * lcg(&mut st)).collect();
        for i in 0..n {
            pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
            e0[i] = extra_level_of_row(&cluster, 0, i) as u32;
            e1[i] = extra_level_of_row(&cluster, 1, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5
                + 0.4 * x1
                + u_p[pid[i] as usize]
                + u_a[e0[i] as usize]
                + u_b[e1[i] as usize]
                + 0.8 * lcg(&mut st);
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut suff = LmmSuffStats::with_groupings(2, g);
        let eids = vec![e0.clone(), e1.clone()];
        suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        let mut fit_c = LmmFitScratch::with_groupings(2, &gref);
        assert!(precompute_balanced_collapse(&suff, &mut fit_c));
        for th in [[0.5, 0.4, 0.3], [1.0, 1.0, 1.0], [0.3, 0.0, 0.8]] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_deviance(&th, &x, &y, &[&pid, &e0, &e1]);
            let tol = 1e-8 * oracle.abs().max(1.0);
            assert!((dev - oracle).abs() <= tol, "θ={th:?}: {dev} vs {oracle}");
            // Collapse arm — reassociation band vs the loop.
            let dev_c = reml_deviance(&th, &suff, &mut fit_c);
            let band = 1e-9 * dev.abs().max(1.0);
            assert!(
                (dev_c - dev).abs() <= band,
                "θ={th:?}: collapse {dev_c} vs {dev}"
            );
        }
    }

    /// Per-component pin: items carry NO between-level signal by construction
    /// (each item sees every subject equally, and the ±0.8 residual pattern is
    /// block-constant so item means cancel exactly), while subjects carry a
    /// real u_p. The crossed component must pin at exactly 0 (boundary_hit
    /// == 1) with the primary component interior.
    #[test]
    fn zero_crossed_variance_pins_only_that_component() {
        let s_cl = 4usize;
        let i_cl = 3usize;
        let mut cluster = intercept_only_spec(Sizing::FixedClusters {
            n_clusters: s_cl as u32,
        });
        cluster.re.as_mut().unwrap().extra_groupings.push(Grouping {
            relation: GroupingRelation::Crossed {
                n_clusters: i_cl as u32,
            },
            slopes: vec![],
        });
        let n = 4 * model_atom(&cluster); // 48: 4 blocks ⇒ ±0.8 cancels per item
        let mut st = 5u64;
        let u_p: Vec<f64> = (0..s_cl).map(|_| 0.8 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut eid = vec![0u32; n];
        for i in 0..n {
            pid[i] = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i) as u32;
            eid[i] = extra_level_of_row(&cluster, 0, i) as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            let e = if (i / model_atom(&cluster)) % 2 == 0 {
                0.8
            } else {
                -0.8
            };
            y[i] = 0.5 + 0.4 * x1 + u_p[pid[i] as usize] + e;
        }
        let mut ws = LmmWorkspace::for_cluster_spec(2, &cluster, n, &[]);
        let eids = vec![eid];
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        assert_eq!(fit.boundary_hit, 1);
        assert_eq!(ws.theta[1], 0.0, "crossed component must pin at exact 0.0");
        assert!(
            ws.theta[0] > PIN_THETA,
            "primary component must stay interior"
        );
        assert!(fit.joint_t_sq.is_finite());
    }

    /// End-to-end crossed+nested fit recovers the generating β within wide
    /// sanity bands and produces finite Wald machinery — the L1 smoke for the
    /// full multi-grouping pipeline (the statistical gates live in L3).
    #[test]
    fn crossed_nested_fit_recovers_betas() {
        let (x, y, pid, eids, cluster) = multi_dataset(true, 4); // n = 192
        let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        let fit = fit_lmm(&mut ws, &[1, 2], None);
        assert!(fit.converged);
        assert!((ws.fit.betas[1] - 0.4).abs() < 0.15);
        assert!((ws.fit.betas[2] + 0.2).abs() < 0.15);
        // Deterministic regression lock (lcg-seeded multi_dataset) alongside the
        // planted-value recovers-check above, which documents intent.
        assert!(
            (ws.fit.betas[1] - 0.40829926961384383).abs() / 0.40829926961384383_f64.abs() < 1e-6
        );
        assert!(
            (ws.fit.betas[2] - -0.2916210839321183).abs() / 0.2916210839321183_f64.abs() < 1e-6
        );
        assert!(ws.fit.t_sq[1].is_finite() && ws.fit.t_sq[2].is_finite());
        assert!(fit.joint_t_sq.is_finite() && fit.joint_t_sq > 0.0);
        assert_eq!(ws.theta.len(), 3);
    }

    /// General-path twin of lmm_fit_warm_path_bounded_alloc: crossed+nested
    /// workspace. Per-call blocks are the tail-llt faer internals (the family
    /// loop is hand-rolled, zero-alloc) — the same acceptance class as q=1.
    /// Warm-started from a cold prime fit's fitted θ (the loop tier's production
    /// pattern), matching the few-eval regime the production path runs.
    #[cfg(feature = "alloc-tests")]
    #[test]
    #[ignore]
    fn lmm_fit_general_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        const BOUND_GENERAL: u64 = 8400; // Measured 8000 (this machine) — ~80 blocks/fit truth-started (scaled rho + spec-derived start; the few-eval regime the production path runs). Per-eval faer `llt` internals only: the family loop is hand-rolled zero-alloc and the cached diagonal_theta map removed the per-fit Vec, so this count is faer-version/machine specific. If faer changes its Cholesky internals, update — do not relax.

        let (x, y, pid, eids, cluster) = multi_dataset(true, 2);
        let targets: Vec<u32> = vec![1, 2];
        let mut ws = LmmWorkspace::for_cluster_spec(3, &cluster, x.nrows(), &[]);

        ws.suff.reset();
        ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
        // prime cold, then warm-start subsequent refits from the previous fit's fitted θ
        // (the loop tier's production pattern; replaces the deleted spec truth-start).
        let _ = fit_lmm(&mut ws, &targets, None);
        let warm = ws.theta.clone();

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.suff.reset();
            ws.suff.add_rows_multi(x.as_ref(), &y, &pid, &eids, None);
            let _ = fit_lmm(&mut ws, &targets, Some(&warm));
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND_GENERAL,
            "general fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND_GENERAL
        );
    }

    /// Crossed-slopes twin of the bounded-alloc gate: the blocked path's only
    /// per-eval heap traffic is the faer `llt` internals (everything else lives in
    /// `LmmFitScratch.blocked_*`, sized once). Same acceptance class as the other
    /// general fits; faer-version/machine specific — if faer changes its Cholesky
    /// internals, update the bound, do not relax.
    #[cfg(feature = "alloc-tests")]
    #[test]
    #[ignore]
    fn lmm_fit_crossed_slope_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        // Measured ~46100 (this machine, faer 0.x): ~460 blocks/fit = the dim≈31
        // tail `llt` internals × the ~50–90 BOBYQA evals of a 6-θ fit. ALL faer-
        // internal — `reml_deviance_blocked` itself is zero-alloc (every buffer is
        // in `blocked_*` scratch; only a stack `lam_g`). faer-version/machine
        // specific; if faer changes its Cholesky internals, update — do not relax.
        const BOUND: u64 = 55000;

        let (x, y, pid, eid) = crossed_slope_golden_dataset();
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 8 },
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 6 },
                    slopes: vec![1],
                }],
            }),
        };
        let mut ws = LmmWorkspace::for_cluster_spec_ext(2, &cluster, x.nrows(), &[1], &[vec![1]]);
        ws.suff.reset();
        ws.suff
            .add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&eid), None);
        // prime cold, then warm-start subsequent refits from the previous fit's fitted θ
        // (the loop tier's production pattern; replaces the deleted spec truth-start).
        let _ = fit_lmm(&mut ws, &[1], None);
        let warm = ws.theta.clone();

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.suff.reset();
            ws.suff
                .add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&eid), None);
            let _ = fit_lmm(&mut ws, &[1], Some(&warm));
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND,
            "crossed-slope fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND
        );
    }

    // -----------------------------------------------------------------------
    // Standalone primary slopes: q_p×q_p primary block, oracle deviance,
    // diagonal-only pin. Data lives on the engine's f32 plane (mirrors the scalar
    // oracle convention); the brute force widens the identical bytes to f64, so
    // the 1e-8 match is exact, not modulo an f32↔f64 roundtrip.
    // -----------------------------------------------------------------------

    /// n=64, p=2 (intercept + x1), 8 clusters, y carries u₀ + u₁·x1.
    fn slope_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
        let (n, nc) = (64usize, 8usize);
        let mut st = 71u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % nc;
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5 + 0.4 * x1 + u0[c] + u1[c] * x1 + 0.8 * lcg(&mut st);
        }
        (x, y, ids)
    }

    /// n=96, p=3 (intercept + x1 + x2), 8 clusters, y carries u₀ + u₁·x1 + u₂·x2.
    fn multislope_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
        let (n, nc) = (96usize, 8usize);
        let mut st = 91u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
        let u2: Vec<f64> = (0..nc).map(|_| 0.25 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 3);
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % nc;
            ids[i] = c as u32;
            let (x1, x2) = (lcg(&mut st), lcg(&mut st));
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            x[(i, 2)] = x2;
            y[i] = 0.5 + 0.4 * x1 + 0.2 * x2 + u0[c] + u1[c] * x1 + u2[c] * x2 + 0.8 * lcg(&mut st);
        }
        (x, y, ids)
    }

    /// Textbook REML deviance with a q×q D over the slope columns of Z_p.
    /// `theta` is the column-major vech of Λ (q×q lower-tri); D_rel = ΛΛ′
    /// (σ-relative); V = I + Z·D_rel·Z′ with Z_i = [1, x[i, slope_cols]]. The
    /// f32 data is widened to f64 so the oracle reads the same bytes the suff
    /// stats accumulated.
    fn brute_force_slope_deviance(
        theta: &[f64],
        x: &Mat<f64>,
        y: &[f64],
        ids: &[u32],
        slope_cols: &[usize],
        q: usize,
    ) -> f64 {
        use faer::linalg::solvers::Solve;
        let (n, p) = (x.nrows(), x.ncols());
        // Λ (q×q lower-tri) from column-major vech, then D = ΛΛ′.
        let mut lam = vec![0.0f64; q * q];
        let mut t = 0;
        for c in 0..q {
            for r in c..q {
                lam[r * q + c] = theta[t];
                t += 1;
            }
        }
        let mut d = vec![0.0f64; q * q];
        for i in 0..q {
            for j in 0..q {
                let mut s = 0.0;
                for k in 0..q {
                    s += lam[i * q + k] * lam[j * q + k];
                }
                d[i * q + j] = s;
            }
        }
        let zrow = |i: usize| -> Vec<f64> {
            let mut z = vec![1.0];
            for &sc in slope_cols {
                z.push(x[(i, sc)]);
            }
            z
        };
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] += 1.0;
        }
        for i in 0..n {
            let zi = zrow(i);
            for j in 0..n {
                if ids[i] == ids[j] {
                    let zj = zrow(j);
                    let mut acc = 0.0;
                    for a in 0..q {
                        for b in 0..q {
                            acc += zi[a] * d[a * q + b] * zj[b];
                        }
                    }
                    v[(i, j)] += acc;
                }
            }
        }
        // REML profile (unchanged from the scalar oracle): ldv + ldk + df·ln s².
        let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldv = 0.0;
        for i in 0..n {
            ldv += vc.L()[(i, i)].ln();
        }
        let ldv = 2.0 * ldv;
        let mut vix = (*x).clone();
        vc.solve_in_place(vix.as_mut());
        let mut viy = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            viy[(i, 0)] = y[i];
        }
        vc.solve_in_place(viy.as_mut());
        let mut xtvix = Mat::<f64>::zeros(p, p);
        let mut xtviy = vec![0.0; p];
        for aa in 0..p {
            for bb in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += x[(i, aa)] * vix[(i, bb)];
                }
                xtvix[(aa, bb)] = s;
            }
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, aa)] * viy[(i, 0)];
            }
            xtviy[aa] = s;
        }
        let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldk = 0.0;
        for aa in 0..p {
            ldk += kc.L()[(aa, aa)].ln();
        }
        let ldk = 2.0 * ldk;
        let mut beta = Mat::<f64>::zeros(p, 1);
        for aa in 0..p {
            beta[(aa, 0)] = xtviy[aa];
        }
        kc.solve_in_place(beta.as_mut());
        let mut ytviy = 0.0;
        for i in 0..n {
            ytviy += y[i] * viy[(i, 0)];
        }
        let mut bxy = 0.0;
        for aa in 0..p {
            bxy += beta[(aa, 0)] * xtviy[aa];
        }
        let df = (n - p) as f64;
        let s2 = (ytviy - bxy) / df;
        ldv + ldk + df * s2.ln()
    }

    fn slope_groupings() -> LmmGroupings {
        // 8 primary clusters, one slope on x_full col 1; no extras.
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 8 },
                slopes: vec![0],
                extra_groupings: vec![],
            }),
        };
        LmmGroupings::from_cluster_spec(&cluster, 64, &[1])
    }

    fn multislope_groupings() -> LmmGroupings {
        // 8 primary clusters, two slopes on x_full cols 1,2; no extras.
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 8 },
                slopes: vec![0, 1],
                extra_groupings: vec![],
            }),
        };
        LmmGroupings::from_cluster_spec(&cluster, 96, &[1, 2])
    }

    #[test]
    fn slope_deviance_matches_brute_force() {
        let (x, y, ids) = slope_dataset();
        let mut suff = LmmSuffStats::with_groupings(2, slope_groupings());
        suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let mut fit = LmmFitScratch::with_groupings(2, &slope_groupings());
        // θ = vech(Λ), q=2: [λ₀₀, λ₁₀, λ₁₁].
        for th in [
            vec![1.0, 0.0, 1.0],
            vec![0.5, 0.2, 0.4],
            vec![2.0, -0.5, 0.7],
            vec![1e-3, 1e-3, 1e-3],
            // θ at THETA_HI (BOBYQA's box upper bound): the per-family Crout
            // pivot product must stay finite here — a product accumulated
            // across all families instead of reset per family would overflow.
            vec![THETA_HI, 0.0, THETA_HI],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_slope_deviance(&th, &x, &y, &ids, &[1], 2);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    #[test]
    fn multislope_deviance_matches_brute_force() {
        let (x, y, ids) = multislope_dataset();
        let mut suff = LmmSuffStats::with_groupings(3, multislope_groupings());
        suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let mut fit = LmmFitScratch::with_groupings(3, &multislope_groupings());
        // θ = vech(Λ), q=3: [λ₀₀, λ₁₀, λ₂₀, λ₁₁, λ₂₁, λ₂₂].
        for th in [
            vec![1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            vec![0.6, 0.2, -0.1, 0.4, 0.15, 0.3],
            vec![1.5, -0.4, 0.3, 0.7, -0.2, 0.5],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_slope_deviance(&th, &x, &y, &ids, &[1, 2], 3);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// End-to-end single-slope fit recovers the planted structure within BOBYQA
    /// bands and pins nothing on a well-identified design.
    #[test]
    fn slope_fit_converges_interior() {
        let (x, y, ids) = slope_dataset();
        let mut ws = LmmWorkspace::with_groupings(2, slope_groupings());
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        // Planted [intercept 0.5, slope 0.4]; small (n=64, 8 clusters) REML draw
        // recovers ≈[0.46, 0.20] — directionally correct, finite-sample attenuated.
        // Pin sign + a band tight enough to catch a sign flip, a collapse to 0, or a
        // blow-up (mere `is_finite` passed any of those).
        assert!(
            (0.2..0.8).contains(&ws.fit.betas[0]),
            "intercept {}",
            ws.fit.betas[0]
        );
        assert!(
            (0.05..0.6).contains(&ws.fit.betas[1]),
            "slope {}",
            ws.fit.betas[1]
        );
        // Deterministic regression lock alongside the bands above.
        assert!(
            (ws.fit.betas[0] - 0.46265883331118085).abs() / 0.46265883331118085_f64.abs() < 1e-6
        );
        assert!(
            (ws.fit.betas[1] - 0.20152611939449563).abs() / 0.20152611939449563_f64.abs() < 1e-6
        );
        assert_eq!(fit.pinned_components & !0b11, 0); // only 2 components exist
    }

    /// End-to-end two-slope fit: 3 components (intercept + 2 slopes), interior.
    #[test]
    fn multislope_fit_converges_interior() {
        let (x, y, ids) = multislope_dataset();
        let mut ws = LmmWorkspace::with_groupings(3, multislope_groupings());
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let fit = fit_lmm(&mut ws, &[1, 2], None);
        assert!(fit.converged);
        // Planted [0.5, 0.4, 0.2]; recovered ≈[0.51, 0.64, 0.28]. Both slopes positive
        // with β̂₁ > β̂₂ (planted ordering preserved) — pin that, so a β₁/β₂ swap or a
        // scale collapse fails where the old `is_finite` pair passed.
        assert!(
            (0.2..0.9).contains(&ws.fit.betas[0]),
            "intercept {}",
            ws.fit.betas[0]
        );
        assert!(
            (0.2..1.1).contains(&ws.fit.betas[1]),
            "slope x1 {}",
            ws.fit.betas[1]
        );
        assert!(
            (0.0..0.7).contains(&ws.fit.betas[2]),
            "slope x2 {}",
            ws.fit.betas[2]
        );
        assert!(
            ws.fit.betas[1] > ws.fit.betas[2],
            "x1 slope must exceed x2 slope"
        );
        // Deterministic regression lock alongside the bands above.
        assert!((ws.fit.betas[0] - 0.5129839426148501).abs() / 0.5129839426148501_f64.abs() < 1e-6);
        assert!((ws.fit.betas[1] - 0.6442611282130077).abs() / 0.6442611282130077_f64.abs() < 1e-6);
        assert!(
            (ws.fit.betas[2] - 0.28355377896623535).abs() / 0.28355377896623535_f64.abs() < 1e-6
        );
        assert_eq!(fit.pinned_components & !0b111, 0); // only 3 components exist
    }

    /// The experimental two-stage warm restart must reach the same
    /// optimum as single-stage on a well-behaved rung — stage 1 (npt = n+2,
    /// rho_end 1e-3, measured correctness-safe on the validation corpus) finds the
    /// basin, stage 2 (npt = 2n+1, shipped rho_end) refines from stage 1's point.
    /// Uses the multislope fixture (n_theta = 6) so the shipped mid-npt formula
    /// (`n_theta >= 3`) is the one exercised by the single-stage comparator.
    #[test]
    fn two_stage_matches_single_stage_optimum() {
        let (x, y, ids) = multislope_dataset();
        let mut ws1 = LmmWorkspace::with_groupings(3, multislope_groupings());
        ws1.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let targets = [1u32, 2];
        let f1 = fit_lmm(&mut ws1, &targets, None);

        let (x, y, ids) = multislope_dataset();
        let mut ws2 = LmmWorkspace::with_groupings(3, multislope_groupings());
        ws2.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let f2 = fit_lmm_two_stage(&mut ws2, &targets, None);

        assert!(f2.converged);
        assert!(
            (f1.deviance - f2.deviance).abs() < 1e-6,
            "two-stage must land on the same optimum: {} vs {}",
            f1.deviance,
            f2.deviance
        );
        assert!(f2.n_eval > 0);
    }

    /// Slope-variance collapse pins the SLOPE component (bit 1), not the
    /// intercept. x1 is a within-cluster antithetic ±1 pattern that carries a
    /// real fixed slope but ZERO cluster-varying slope, and the residual is a
    /// ±0.8 period-4 quadrature block (+,+,−,− against x1's +,−,+,−) so every
    /// cluster has Σ resid = 0 AND Σ x1·resid = 0 exactly — the REML
    /// slope-variance MLE is 0, so λ₁₁ pins (bit 1) while the planted u₀ keeps
    /// λ₀₀ interior. (The original lockstep ±0.8 pattern made resid ≡ 0.8·x1 —
    /// collinear with the slope covariate, so σ̂²→0 once large θ₀ absorbed the
    /// exactly-identified cluster means, the deviance ran unbounded to the θ₀
    /// box bound, and the λ₁₁ pin rode FP noise on the degenerate surface; the
    /// quadrature pattern keeps σ̂² positive and θ̂₀ genuinely interior.) Large
    /// balanced design (16 clusters × 16 rows) so finite-sample REML does not
    /// overfit a spurious slope RE the way a small noisy draw does.
    #[test]
    fn zero_slope_variance_pins_slope_component() {
        let (nc, per) = (16usize, 16usize);
        let n = nc * per;
        let mut st = 5u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        #[allow(clippy::needless_range_loop)]
        for c in 0..nc {
            for k in 0..per {
                let i = c * per + k;
                ids[i] = c as u32;
                // x1: identical antithetic pattern in every cluster (±1
                // alternating) — no between-cluster slope signal.
                let x1 = if k % 2 == 0 { 1.0 } else { -1.0 };
                // residual: ±0.8 period-4 quadrature against x1, so per cluster
                // Σ x1·resid = 0 AND Σ resid = 0 (no slope/intercept RE pull
                // from the noise; only the planted u₀ moves intercepts).
                let e = if (k / 2) % 2 == 0 { 0.8 } else { -0.8 };
                x[(i, 0)] = 1.0;
                x[(i, 1)] = x1;
                y[i] = 0.5 + 0.4 * x1 + u0[c] + e;
            }
        }
        let mut ws = LmmWorkspace::with_groupings(
            2,
            LmmGroupings::from_cluster_spec(
                &ModelSpec {
                    family: Family::Gaussian,
                    re: Some(ReStructure {
                        sizing: Sizing::FixedClusters {
                            n_clusters: nc as u32,
                        },
                        slopes: vec![0],
                        extra_groupings: vec![],
                    }),
                },
                n,
                &[1],
            ),
        );
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged);
        assert!(
            ws.theta[2] == 0.0,
            "slope λ₁₁ must pin to exactly 0, got {:e}",
            ws.theta[2]
        );
        assert!(fit.pinned_components & 0b10 != 0, "slope component bit set");
        assert!(
            ws.theta[0] > PIN_THETA,
            "intercept component must stay interior"
        );
        assert!(
            ws.theta[0] < THETA_HI,
            "intercept component must be off the box bound"
        );
    }

    // -----------------------------------------------------------------------
    // Composition: primary slope (1 + x1 | g) co-existing with an
    // intercept-only crossed (1 | item) / nested (1 | g:sub) extra. The
    // family-blocked deviance must match a brute-force V = I + Z_p D_p Z_p′ +
    // τ_e² Z_e Z_e′. Data on the f32 plane (the suff-stats input convention);
    // the oracle widens the identical bytes, so the 1e-8 match is exact.
    // -----------------------------------------------------------------------

    /// n=80, p=2 (intercept + x1), 8 primary clusters crossed with 5 items;
    /// y carries u₀ + u₁·x1 (primary) + v (item intercept).
    fn composed_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let (n, nc, ni) = (80usize, 8usize, 5usize);
        let mut st = 41u64;
        let u0: Vec<f64> = (0..nc).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..nc).map(|_| 0.3 * lcg(&mut st)).collect();
        let v: Vec<f64> = (0..ni).map(|_| 0.4 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let (mut pid, mut iid) = (vec![0u32; n], vec![0u32; n]);
        for i in 0..n {
            let (c, it) = (i % nc, i % ni);
            pid[i] = c as u32;
            iid[i] = it as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5 + 0.4 * x1 + u0[c] + u1[c] * x1 + v[it] + 0.8 * lcg(&mut st);
        }
        (x, y, pid, iid)
    }

    /// primary (1 + x1 | g), crossed (1 | item); slope on x_full col 1.
    fn composed_groupings() -> LmmGroupings {
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 8 },
                slopes: vec![0],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 5 },
                    slopes: vec![],
                }],
            }),
        };
        LmmGroupings::from_cluster_spec(&cluster, 80, &[1])
    }

    // --- θ-layout generalization (scalar → vech ranges) ---

    /// One intercept-only primary + one crossed grouping of RE width `q_g`
    /// (intercept + `q_g−1` slopes), expressed through the slope machinery — the
    /// θ-layout fixture. Slope columns are placeholders (layout reads only
    /// `slopes.len()`).
    fn groupings_primary1_crossed_qg(q_g: usize) -> LmmGroupings {
        let slopes: Vec<crate::ColumnId> = (0..q_g - 1).map(|k| (k + 1) as u32).collect();
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 8 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 5 },
                    slopes,
                }],
            }),
        };
        LmmGroupings::from_cluster_spec(&cluster, 80, &[])
    }

    #[test]
    fn extra_qg1_theta_layout_matches_scalar() {
        // Intercept-only crossed factor through the slope machinery = the old
        // scalar layout: one primary scalar + one extra scalar.
        let g = groupings_primary1_crossed_qg(1);
        assert_eq!(g.n_theta(), 1 + 1);
        assert_eq!(g.crossed[0].vech_start, 1);
        assert_eq!(g.crossed[0].q, 1);
        assert!(!g.extra_slopes_any);
    }

    #[test]
    fn extra_qg2_theta_packs_vech3() {
        let g = groupings_primary1_crossed_qg(2);
        assert_eq!(g.crossed[0].q, 2);
        assert_eq!(g.n_theta(), 1 + 3); // primary scalar + vech(2×2)=3
        assert!(g.extra_slopes_any);
        // The extra block's two diagonal θ indices are vech_start (=1) and
        // vech_start + 2 (=3) under the column-major lower-tri convention.
        let diag = &g.diagonal_theta;
        assert!(diag.contains(&1) && diag.contains(&3));
        // Off-diagonal λ₁₀ at index 2 is NOT a diagonal (signed box).
        assert!(!diag.contains(&2));
    }

    // --- Extra-slope sufficient statistics ---

    /// Brute-force the `s` columns for a crossed factor carrying a slope: the
    /// intercept subcol is Σ_{rows∈level} [X y]; the slope subcol is Σ x_slope·[X y].
    #[test]
    fn extra_crossed_slope_s_columns_match_bruteforce() {
        let n = 6usize;
        let p = 3; // [1, x1, x2]
        let xd = [
            (0.5, -0.2),
            (-0.3, 0.7),
            (0.9, 0.1),
            (-0.6, -0.4),
            (0.2, 0.8),
            (0.4, -0.5),
        ];
        let cluster_ids = [0u32, 1, 0, 1, 0, 1];
        let crossed_ids = [0u32, 1, 2, 0, 1, 2];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x = Mat::<f64>::zeros(n, p);
        for i in 0..n {
            x[(i, 0)] = 1.0;
            x[(i, 1)] = xd[i].0;
            x[(i, 2)] = xd[i].1;
        }
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 2 },
                slopes: vec![],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 3 },
                    slopes: vec![1],
                }],
            }),
        };
        // crossed slope on x_full col 1.
        let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[], &[vec![1]]);
        // crossed block: q_g=2, offset = prim_width = 2 (n_primary=2, q_p=1).
        assert_eq!(g.extra_offsets[0], 2);
        assert_eq!(g.extra_q[0], 2);
        let mut suff = LmmSuffStats::with_groupings(p, g);
        suff.add_rows_multi(x.as_ref(), &y, &cluster_ids, &[crossed_ids.to_vec()], None);
        let m = p + 1;
        for c in 0..3usize {
            let icol = 2 + c * 2;
            let scol = icol + 1;
            let mut s_int = vec![0.0; m];
            let mut s_slope = vec![0.0; m];
            for i in 0..n {
                if crossed_ids[i] as usize == c {
                    let w = [x[(i, 0)], x[(i, 1)], x[(i, 2)], y[i]];
                    let x1 = x[(i, 1)];
                    for j in 0..m {
                        s_int[j] += w[j];
                        s_slope[j] += x1 * w[j];
                    }
                }
            }
            for j in 0..m {
                assert!(
                    (suff.s[(j, icol)] - s_int[j]).abs() < 1e-12,
                    "intercept col level {c} row {j}: got {} want {}",
                    suff.s[(j, icol)],
                    s_int[j]
                );
                assert!(
                    (suff.s[(j, scol)] - s_slope[j]).abs() < 1e-12,
                    "slope col level {c} row {j}: got {} want {}",
                    suff.s[(j, scol)],
                    s_slope[j]
                );
            }
            // counts only on the intercept subcol.
            let n_c = crossed_ids.iter().filter(|&&l| l as usize == c).count() as f64;
            assert_eq!(suff.counts[icol], n_c);
            assert_eq!(suff.counts[scol], 0.0);
        }
    }

    /// REML deviance on the explicit n×n V for the composed model: the 2×2
    /// primary slope block (D_p = ΛΛ′ over [1, x1]) PLUS the extra-grouping
    /// intercept block (θ_e² when the extra ids match). The f32 data is widened
    /// to f64 so the oracle reads the same bytes the suff stats accumulated.
    /// `eid` is the extra grouping's level id per row (item, or nested child).
    /// θ = [primary vech λ₀₀, λ₁₀, λ₁₁ ; extra scalar θ_e].
    fn brute_force_composed_deviance(
        theta: &[f64],
        x: &Mat<f64>,
        y: &[f64],
        pid: &[u32],
        eid: &[u32],
    ) -> f64 {
        let n = x.nrows();
        let (a, b, c) = (theta[0], theta[1], theta[2]);
        // D_p = ΛΛ′, Λ = [[a,0],[b,c]] (column-major vech).
        let (d00, d01, d11) = (a * a, a * b, b * b + c * c);
        let te2 = theta[3] * theta[3];
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] += 1.0;
        }
        for i in 0..n {
            for j in 0..n {
                if pid[i] == pid[j] {
                    let (zi1, zj1) = (x[(i, 1)], x[(j, 1)]);
                    v[(i, j)] += d00 + d01 * (zi1 + zj1) + d11 * zi1 * zj1;
                }
                if eid[i] == eid[j] {
                    v[(i, j)] += te2;
                }
            }
        }
        reml_profile_from_v(&v, x, y)
    }

    /// REML profiled deviance from an explicit n×n marginal V (in residual-σ²
    /// units): `log|V| + log|XᵀV⁻¹X| + (N−P)·log σ̂²`. The shared V→deviance back
    /// end for every brute-force oracle (composed, crossed-slope, …).
    fn reml_profile_from_v(v: &Mat<f64>, x: &Mat<f64>, y: &[f64]) -> f64 {
        use faer::linalg::solvers::Solve;
        let (n, p) = (x.nrows(), x.ncols());
        let vc = v.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldv = 0.0;
        for i in 0..n {
            ldv += vc.L()[(i, i)].ln();
        }
        let ldv = 2.0 * ldv;
        let mut vix = (*x).clone();
        vc.solve_in_place(vix.as_mut());
        let mut viy = Mat::<f64>::zeros(n, 1);
        for i in 0..n {
            viy[(i, 0)] = y[i];
        }
        vc.solve_in_place(viy.as_mut());
        let mut xtvix = Mat::<f64>::zeros(p, p);
        let mut xtviy = vec![0.0; p];
        for aa in 0..p {
            for bb in 0..p {
                let mut s = 0.0;
                for i in 0..n {
                    s += x[(i, aa)] * vix[(i, bb)];
                }
                xtvix[(aa, bb)] = s;
            }
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, aa)] * viy[(i, 0)];
            }
            xtviy[aa] = s;
        }
        let kc = xtvix.as_ref().llt(faer::Side::Lower).unwrap();
        let mut ldk = 0.0;
        for aa in 0..p {
            ldk += kc.L()[(aa, aa)].ln();
        }
        let ldk = 2.0 * ldk;
        let mut beta = Mat::<f64>::zeros(p, 1);
        for aa in 0..p {
            beta[(aa, 0)] = xtviy[aa];
        }
        kc.solve_in_place(beta.as_mut());
        let mut ytviy = 0.0;
        for i in 0..n {
            ytviy += y[i] * viy[(i, 0)];
        }
        let mut bxy = 0.0;
        for aa in 0..p {
            bxy += beta[(aa, 0)] * xtviy[aa];
        }
        let df = (n - p) as f64;
        let s2 = (ytviy - bxy) / df;
        ldv + ldk + df * s2.ln()
    }

    /// Brute-force REML deviance for a CROSSED-SLOPE model
    /// `y ~ x1 + (1+x1 | primary) + (1+x1 | crossed)`: V = I + Z_p D_p Z_pᵀ +
    /// Z_e D_e Z_eᵀ, each D a 2×2 from its vech θ over [1, x1]. θ =
    /// [primary vech (3) ; crossed vech (3)].
    fn brute_force_crossed_slope_deviance(
        theta: &[f64],
        x: &Mat<f64>,
        y: &[f64],
        pid: &[u32],
        eid: &[u32],
    ) -> f64 {
        let n = x.nrows();
        let (ap, bp, cp) = (theta[0], theta[1], theta[2]);
        let (dp00, dp01, dp11) = (ap * ap, ap * bp, bp * bp + cp * cp);
        let (ae, be, ce) = (theta[3], theta[4], theta[5]);
        let (de00, de01, de11) = (ae * ae, ae * be, be * be + ce * ce);
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] += 1.0;
        }
        for i in 0..n {
            for j in 0..n {
                let (zi, zj) = (x[(i, 1)], x[(j, 1)]);
                if pid[i] == pid[j] {
                    v[(i, j)] += dp00 + dp01 * (zi + zj) + dp11 * zi * zj;
                }
                if eid[i] == eid[j] {
                    v[(i, j)] += de00 + de01 * (zi + zj) + de11 * zi * zj;
                }
            }
        }
        reml_profile_from_v(&v, x, y)
    }

    /// Slope + crossed: the composed deviance matches the brute-force oracle to
    /// 1e-8 — the slope-composition gate. zx_slope carries the slope↔crossed
    /// coupling; the primary 2×2 block and the item intercept block are coupled
    /// through the shared family-blocked tail.
    #[test]
    fn composed_deviance_matches_brute_force() {
        let (x, y, pid, iid) = composed_dataset();
        let mut suff = LmmSuffStats::with_groupings(2, composed_groupings());
        suff.add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&iid), None); // item ids as the single extra grouping
        let mut fit = LmmFitScratch::with_groupings(2, &composed_groupings());
        // θ = [λ₀₀, λ₁₀, λ₁₁, θ_c].
        for th in [
            vec![1.0, 0.0, 1.0, 0.5],
            vec![0.6, 0.2, 0.4, 0.3],
            vec![1.5, -0.4, 0.7, 0.8],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_composed_deviance(&th, &x, &y, &pid, &iid);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// CROSSED SLOPES (the headline lme4-agreement case): `y ~ x1 + (1+x1 | primary)
    /// + (1+x1 | item)` — both grouping factors carry a random slope on x1, so the
    /// gated blocked path runs. Deviance must match the explicit-V oracle to 1e-7
    /// across θ, including the primary-slope↔crossed-slope coupling (the x1²
    /// weighted co-occurrence) the blocked `zx` fill captures.
    #[test]
    fn crossed_slope_deviance_matches_brute_force() {
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 5 },
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 4 },
                    slopes: vec![1],
                }],
            }),
        };
        let n = 60; // atom = 5·4 = 20 ⇒ 3 balanced blocks
        let mut st = 91u64;
        let u0p: Vec<f64> = (0..5).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1p: Vec<f64> = (0..5).map(|_| 0.3 * lcg(&mut st)).collect();
        let u0e: Vec<f64> = (0..4).map(|_| 0.4 * lcg(&mut st)).collect();
        let u1e: Vec<f64> = (0..4).map(|_| 0.3 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut eid = vec![0u32; n];
        for i in 0..n {
            let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
            let item = extra_level_of_row(&cluster, 0, i) as usize;
            pid[i] = par as u32;
            eid[i] = item as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5
                + 0.4 * x1
                + u0p[par]
                + u1p[par] * x1
                + u0e[item]
                + u1e[item] * x1
                + 0.8 * lcg(&mut st);
        }
        let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
        assert!(g.extra_slopes_any, "must route to the blocked path");
        let mut suff = LmmSuffStats::with_groupings(2, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &[eid.clone()], None);
        let gref = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        // θ = [primary vech (λ₀₀,λ₁₀,λ₁₁) ; crossed vech (λ₀₀,λ₁₀,λ₁₁)].
        for th in [
            vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![0.7, 0.2, 0.5, 0.6, 0.1, 0.4],
            vec![1.3, -0.3, 0.6, 0.9, -0.2, 0.5],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_crossed_slope_deviance(&th, &x, &y, &pid, &eid);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-7 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// NESTED SLOPES (the nested-slope defect): `y ~ x1 + (1+x1 | grp) + (1+x1 | class)`,
    /// class nested in grp — both grouping factors carry a random slope on x1, so
    /// the gated blocked path runs with a nested factor of q_n = 2. Before the
    /// fix the blocked path assembled the nested children intercept-only (scalar
    /// θ_n), diverging to NaN. The marginal V is grouping-agnostic (Σ_g Z_g D_g Z_gᵀ
    /// over rows sharing a level id), so the crossed-slope oracle is reused with the
    /// GLOBAL nested child id as the extra level. Matches the explicit-V oracle to
    /// 1e-7 across θ.
    #[test]
    fn nested_slope_deviance_matches_brute_force() {
        let n_per_parent = 3u32;
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 5 },
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::NestedWithin { n_per_parent },
                    slopes: vec![1],
                }],
            }),
        };
        let n = 60; // atom = primary 5 · nested 3 = 15 ⇒ 4 balanced blocks
        let n_child = 5 * n_per_parent as usize;
        let mut st = 137u64;
        let u0p: Vec<f64> = (0..5).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1p: Vec<f64> = (0..5).map(|_| 0.3 * lcg(&mut st)).collect();
        let u0e: Vec<f64> = (0..n_child).map(|_| 0.4 * lcg(&mut st)).collect();
        let u1e: Vec<f64> = (0..n_child).map(|_| 0.3 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut eid = vec![0u32; n];
        for i in 0..n {
            let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
            let child = extra_level_of_row(&cluster, 0, i); // GLOBAL child id
            pid[i] = par as u32;
            eid[i] = child as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5
                + 0.4 * x1
                + u0p[par]
                + u1p[par] * x1
                + u0e[child]
                + u1e[child] * x1
                + 0.8 * lcg(&mut st);
        }
        let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
        assert!(g.extra_slopes_any, "must route to the blocked path");
        assert!(g.nested.is_some(), "must carry a nested factor");
        let mut suff = LmmSuffStats::with_groupings(2, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &[eid.clone()], None);
        let gref = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[1], &[vec![1]]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        // θ = [primary vech (λ₀₀,λ₁₀,λ₁₁) ; nested vech (λ₀₀,λ₁₀,λ₁₁)].
        for th in [
            vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![0.7, 0.2, 0.5, 0.6, 0.1, 0.4],
            vec![1.3, -0.3, 0.6, 0.9, -0.2, 0.5],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_crossed_slope_deviance(&th, &x, &y, &pid, &eid);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-7 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// End-to-end NESTED-SLOPE fit: the original nested-slope symptom was BOBYQA diverging
    /// to NaN (`converged = false`) on every seed because the blocked objective was
    /// mis-assembled. With the correct objective the full θ-search must converge to
    /// a finite interior fit. Asserts `converged`, no numerical failure
    /// (`boundary_hit != 2`), finite θ̂/σ̂², and β̂ recovered near the planted
    /// [0.5, 0.4].
    #[test]
    fn nested_slope_fit_converges() {
        let n_per_parent = 3u32;
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 5 },
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::NestedWithin { n_per_parent },
                    slopes: vec![1],
                }],
            }),
        };
        let n = 120; // atom = 5·3 = 15 ⇒ 8 balanced blocks
        let n_child = 5 * n_per_parent as usize;
        let mut st = 137u64;
        let u0p: Vec<f64> = (0..5).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1p: Vec<f64> = (0..5).map(|_| 0.3 * lcg(&mut st)).collect();
        let u0e: Vec<f64> = (0..n_child).map(|_| 0.4 * lcg(&mut st)).collect();
        let u1e: Vec<f64> = (0..n_child).map(|_| 0.3 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut eid = vec![0u32; n];
        for i in 0..n {
            let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
            let child = extra_level_of_row(&cluster, 0, i);
            pid[i] = par as u32;
            eid[i] = child as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5
                + 0.4 * x1
                + u0p[par]
                + u1p[par] * x1
                + u0e[child]
                + u1e[child] * x1
                + 0.8 * lcg(&mut st);
        }
        let mut ws = LmmWorkspace::for_cluster_spec_ext(2, &cluster, n, &[1], &[vec![1]]);
        ws.suff.reset();
        ws.suff
            .add_rows_multi(x.as_ref(), &y, &pid, &[eid.clone()], None);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged, "nested-slope fit must converge");
        assert_ne!(fit.boundary_hit, 2, "must not be a numerical (NaN) failure");
        assert!(
            fit.sigma_sq.is_finite() && fit.sigma_sq > 0.0,
            "σ̂² {}",
            fit.sigma_sq
        );
        assert!(ws.theta.iter().all(|t| t.is_finite()), "θ̂ {:?}", ws.theta);
        assert!(
            (0.2..0.8).contains(&ws.fit.betas[0]),
            "intercept {}",
            ws.fit.betas[0]
        );
        assert!(
            (0.1..0.7).contains(&ws.fit.betas[1]),
            "slope {}",
            ws.fit.betas[1]
        );
        // Deterministic regression lock (seed 137) alongside the wide recovers-check above.
        assert!((ws.fit.betas[0] - 0.6209080774915476).abs() / 0.6209080774915476_f64.abs() < 1e-6);
        assert!((ws.fit.betas[1] - 0.257915422474595).abs() / 0.257915422474595_f64.abs() < 1e-6);
    }

    /// General brute-force REML deviance: V = I + Σ_g Z_g D_g Z_gᵀ where each
    /// factor `(ids, vech)` contributes a 2×2 D over [1, x1] (an intercept-only
    /// factor passes `[θ, 0, 0]`). Used for the multi-crossed-factor oracle.
    fn brute_force_slopes_deviance(x: &Mat<f64>, y: &[f64], factors: &[(&[u32], [f64; 3])]) -> f64 {
        let n = x.nrows();
        let mut v = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            v[(i, i)] += 1.0;
        }
        for &(ids, vech) in factors {
            let (a, b, c) = (vech[0], vech[1], vech[2]);
            let (d00, d01, d11) = (a * a, a * b, b * b + c * c);
            for i in 0..n {
                for j in 0..n {
                    if ids[i] == ids[j] {
                        let (zi, zj) = (x[(i, 1)], x[(j, 1)]);
                        v[(i, j)] += d00 + d01 * (zi + zj) + d11 * zi * zj;
                    }
                }
            }
        }
        reml_profile_from_v(&v, x, y)
    }

    /// TWO crossed factors with slopes:
    /// `y ~ x1 + (1 | primary) + (1+x1 | c1) + (1+x1 | c2)`. Exercises the
    /// crossed↔crossed slope coupling (c1's slope column against c2's, the x1²
    /// weighted co-occurrence between two distinct crossed factors) — the part
    /// neither the composed nor single-crossed test reaches. Matches the
    /// explicit-V oracle to 1e-7.
    #[test]
    fn two_crossed_slopes_deviance_matches_brute_force() {
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 3 },
                slopes: vec![], // primary intercept-only
                extra_groupings: vec![
                    Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 3 },
                        slopes: vec![1],
                    },
                    Grouping {
                        relation: GroupingRelation::Crossed { n_clusters: 3 },
                        slopes: vec![1],
                    },
                ],
            }),
        };
        let n = 54; // atom = 3·3·3 = 27 ⇒ 2 blocks
        let mut st = 73u64;
        let up: Vec<f64> = (0..3).map(|_| 0.45 * lcg(&mut st)).collect();
        let u0a: Vec<f64> = (0..3).map(|_| 0.4 * lcg(&mut st)).collect();
        let u1a: Vec<f64> = (0..3).map(|_| 0.3 * lcg(&mut st)).collect();
        let u0b: Vec<f64> = (0..3).map(|_| 0.35 * lcg(&mut st)).collect();
        let u1b: Vec<f64> = (0..3).map(|_| 0.28 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut c1 = vec![0u32; n];
        let mut c2 = vec![0u32; n];
        for i in 0..n {
            let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
            let a = extra_level_of_row(&cluster, 0, i) as usize;
            let b = extra_level_of_row(&cluster, 1, i) as usize;
            pid[i] = par as u32;
            c1[i] = a as u32;
            c2[i] = b as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5
                + 0.4 * x1
                + up[par]
                + u0a[a]
                + u1a[a] * x1
                + u0b[b]
                + u1b[b] * x1
                + 0.8 * lcg(&mut st);
        }
        let g = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[], &[vec![1], vec![1]]);
        assert!(g.extra_slopes_any);
        let mut suff = LmmSuffStats::with_groupings(2, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &[c1.clone(), c2.clone()], None);
        let gref = LmmGroupings::from_cluster_spec_ext(&cluster, n, &[], &[vec![1], vec![1]]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        // θ = [primary scalar ; c1 vech (3) ; c2 vech (3)].
        for th in [
            vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            vec![0.6, 0.7, 0.2, 0.4, 0.6, -0.1, 0.35],
            vec![0.8, 1.2, -0.3, 0.5, 0.9, 0.25, 0.45],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_slopes_deviance(
                &x,
                &y,
                &[
                    (&pid, [th[0], 0.0, 0.0]),
                    (&c1, [th[1], th[2], th[3]]),
                    (&c2, [th[4], th[5], th[6]]),
                ],
            );
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-7 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// Deterministic crossed-slope dataset for the lme4 golden: 8 primary
    /// clusters × 6 crossed levels × 2 reps (n=96),
    /// `y = 1.0 + 0.8·x1 + u0p + u1p·x1 + u0e + u1e·x1 + ε`. The Rust generator is
    /// the source of truth; `dump_crossed_slope_golden_csv` writes it for the R
    /// `lme4::lmer` reference whose fit is frozen in `GOLDEN_LME4_*`.
    fn crossed_slope_golden_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let (n_prim, n_cross, n) = (8usize, 6usize, 96usize);
        let mut st = 20260629u64;
        let u0p: Vec<f64> = (0..n_prim).map(|_| 0.7 * lcg(&mut st)).collect();
        let u1p: Vec<f64> = (0..n_prim).map(|_| 0.5 * lcg(&mut st)).collect();
        let u0e: Vec<f64> = (0..n_cross).map(|_| 0.6 * lcg(&mut st)).collect();
        let u1e: Vec<f64> = (0..n_cross).map(|_| 0.4 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut eid = vec![0u32; n];
        for i in 0..n {
            let pp = i % n_prim; // FixedClusters primary: i % n_clusters
            let ee = (i / n_prim) % n_cross; // crossed: (i / n_prim) % n_cross
            pid[i] = pp as u32;
            eid[i] = ee as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 1.0
                + 0.8 * x1
                + u0p[pp]
                + u1p[pp] * x1
                + u0e[ee]
                + u1e[ee] * x1
                + 0.5 * lcg(&mut st);
        }
        (x, y, pid, eid)
    }

    /// Run once (`cargo test -p ... dump_crossed_slope_golden_csv -- --ignored`)
    /// to regenerate the CSV the R reference reads. Not a normal test.
    #[test]
    #[ignore]
    fn dump_crossed_slope_golden_csv() {
        let (x, y, pid, eid) = crossed_slope_golden_dataset();
        let mut s = String::from("x1,y,pid,eid\n");
        for i in 0..y.len() {
            s.push_str(&format!("{},{},{},{}\n", x[(i, 1)], y[i], pid[i], eid[i]));
        }
        std::fs::write("/tmp/crossed_slope_golden.csv", s).unwrap();
    }

    /// L3 golden: `glmm`'s crossed-slope fit must reproduce `lme4::lmer`'s REML fit
    /// of `y ~ x1 + (1+x1|pid) + (1+x1|eid)` on the committed dataset — fixed
    /// effects, residual σ², and both 2×2 RE covariances. Frozen from
    /// `/tmp/golden_fit.R` (lme4 1.1, bobyqa). Recovered D_g = σ̂²·Λ_gΛ_gᵀ from θ̂.
    #[test]
    fn crossed_slope_fit_matches_lme4_golden() {
        // lme4 golden (REML, bobyqa).
        const G_BETA0: f64 = 1.0582083262;
        const G_BETA1: f64 = 0.6334043248;
        const G_SIGMA2: f64 = 0.0921249591;
        const G_PID_V0: f64 = 0.1406815355; // var(intercept)
        const G_PID_V1: f64 = 0.1237856496; // var(x1)
        const G_PID_COV: f64 = 0.0127473486;
        const G_EID_V0: f64 = 0.1828301299;
        const G_EID_V1: f64 = 0.0396985129;
        const G_EID_COV: f64 = -0.0456611171;

        let (x, y, pid, eid) = crossed_slope_golden_dataset();
        let n = y.len();
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 8 },
                slopes: vec![1],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::Crossed { n_clusters: 6 },
                    slopes: vec![1],
                }],
            }),
        };
        let mut ws = LmmWorkspace::for_cluster_spec_ext(2, &cluster, n, &[1], &[vec![1]]);
        ws.suff.reset();
        ws.suff
            .add_rows_multi(x.as_ref(), &y, &pid, std::slice::from_ref(&eid), None);
        let fit = fit_lmm(&mut ws, &[1], None);
        assert!(fit.converged, "golden fit must converge");
        let s2 = fit.sigma_sq;

        // Fixed effects + residual variance.
        assert!(
            (ws.fit.betas[0] - G_BETA0).abs() < 1e-4,
            "β0 {} vs {G_BETA0}",
            ws.fit.betas[0]
        );
        assert!(
            (ws.fit.betas[1] - G_BETA1).abs() < 1e-4,
            "β1 {} vs {G_BETA1}",
            ws.fit.betas[1]
        );
        assert!(
            (s2 - G_SIGMA2).abs() <= 1e-3 * G_SIGMA2,
            "σ² {s2} vs {G_SIGMA2}"
        );

        // D_g = σ̂²·Λ_gΛ_gᵀ from θ̂ (primary vech θ[0..3], crossed vech θ[3..6]).
        let dblock = |t: &[f64]| {
            let (a, b, c) = (t[0], t[1], t[2]);
            (s2 * a * a, s2 * (b * b + c * c), s2 * a * b) // (v0, v1, cov)
        };
        let (pv0, pv1, pcov) = dblock(&ws.theta[0..3]);
        let (ev0, ev1, ecov) = dblock(&ws.theta[3..6]);
        let close = |got: f64, want: f64, name: &str| {
            assert!(
                (got - want).abs() <= 2e-3 * want.abs().max(1e-3),
                "{name}: {got} vs {want}"
            );
        };
        close(pv0, G_PID_V0, "pid var0");
        close(pv1, G_PID_V1, "pid var1");
        close(pcov, G_PID_COV, "pid cov");
        close(ev0, G_EID_V0, "eid var0");
        close(ev1, G_EID_V1, "eid var1");
        close(ecov, G_EID_COV, "eid cov");
    }

    /// Slope + NESTED: `(1 + x1 | g) + (1 | g:sub)` — the composed deviance with
    /// a nested child tail (vs the crossed tail above). Exercises the
    /// primary-slope↔child off-diagonal (read from `s`) and the shifted nested
    /// offset `q_p·n_primary + f·np + c`. The nested child ids are globalized
    /// (parent·np + within) — the workspace layout the contract helpers produce.
    #[test]
    fn composed_nested_deviance_matches_brute_force() {
        // 8 primary clusters × 2 children each, fixed-size 8 ⇒ 64 rows / 4 blocks.
        let cluster = ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedSize { cluster_size: 8 },
                slopes: vec![0],
                extra_groupings: vec![Grouping {
                    relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                    slopes: vec![],
                }],
            }),
        };
        let n = 4 * model_atom(&cluster); // 64
        let mut st = 47u64;
        let u0: Vec<f64> = (0..8).map(|_| 0.5 * lcg(&mut st)).collect();
        let u1: Vec<f64> = (0..8).map(|_| 0.3 * lcg(&mut st)).collect();
        let u_c: Vec<f64> = (0..16).map(|_| 0.35 * lcg(&mut st)).collect(); // 8 parents × 2 children
        let mut x = Mat::<f64>::zeros(n, 2);
        let mut y = vec![0.0f64; n];
        let mut pid = vec![0u32; n];
        let mut cid = vec![0u32; n]; // globalized child id (parent·np + within)
        for i in 0..n {
            let par = cluster.re.as_ref().unwrap().sizing.cluster_of_row(i);
            let child = extra_level_of_row(&cluster, 0, i); // already globalized par·np + within
            pid[i] = par as u32;
            cid[i] = child as u32;
            let x1 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            y[i] = 0.5 + 0.4 * x1 + u0[par] + u1[par] * x1 + u_c[child] + 0.8 * lcg(&mut st);
        }
        let g = LmmGroupings::from_cluster_spec(&cluster, n, &[1]);
        let mut suff = LmmSuffStats::with_groupings(2, g);
        suff.add_rows_multi(x.as_ref(), &y, &pid, &[cid.clone()], None);
        let gref = LmmGroupings::from_cluster_spec(&cluster, n, &[1]);
        let mut fit = LmmFitScratch::with_groupings(2, &gref);
        // The brute-force oracle is V-shape-agnostic: the nested child block adds
        // θ_n² when the (globalized) child ids match — same form as the crossed.
        for th in [
            vec![1.0, 0.0, 1.0, 0.5],
            vec![0.7, 0.25, 0.5, 0.4],
            vec![1.3, -0.3, 0.6, 0.2],
        ] {
            let dev = reml_deviance(&th, &suff, &mut fit);
            let oracle = brute_force_composed_deviance(&th, &x, &y, &pid, &cid);
            assert!(dev.is_finite(), "θ={th:?}");
            assert!(
                (dev - oracle).abs() <= 1e-8 * oracle.abs().max(1.0),
                "θ={th:?}: {dev} vs {oracle}"
            );
        }
    }

    /// Bounded-allocation twin — the standalone slope workspace
    /// allocates only faer `llt` internals on the warm `fit_lmm` loop, the same
    /// acceptance class as the q=1 / general twins.
    ///   cargo test -p glmm --features alloc-tests lmm_fit_slope_warm_path_bounded_alloc -- --ignored --test-threads=1
    #[cfg(feature = "alloc-tests")]
    #[test]
    #[ignore]
    fn lmm_fit_slope_warm_path_bounded_alloc() {
        const N_CALLS: usize = 100;
        const BOUND_SLOPE: u64 = 12000; // Measured 11400 (this machine) — ~114 blocks/fit of faer `llt` internals (one m×m tail llt per eval × ~54 evals on the blind 3-D q_p=2 surface; the family loop + primary Λ/Gram are zero-alloc scratch, and the cached diagonal_theta map removed the per-fit Vec). Higher total than q=1's 4600 only via the larger blind eval count, not a richer per-eval alloc — faer-version/machine specific. If faer's Cholesky internals change, update — do not relax.

        let (x, y, ids) = slope_dataset();
        let targets: Vec<u32> = vec![1];
        let mut ws = LmmWorkspace::with_groupings(2, slope_groupings());

        ws.suff.reset();
        ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
        let _ = fit_lmm(&mut ws, &targets, None);

        let profiler = dhat::Profiler::builder().testing().build();
        for _ in 0..N_CALLS {
            ws.suff.reset();
            ws.suff.add_rows_multi(x.as_ref(), &y, &ids, &[], None);
            let _ = fit_lmm(&mut ws, &targets, None);
        }
        let stats = dhat::HeapStats::get();
        drop(profiler);
        assert!(
            stats.total_blocks <= BOUND_SLOPE,
            "slope fit_lmm allocated {} blocks across {} warm-path calls (BOUND = {})",
            stats.total_blocks,
            N_CALLS,
            BOUND_SLOPE
        );
    }
}
