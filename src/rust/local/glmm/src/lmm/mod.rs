//! General-machine LMM solver core: family-blocked profiled REML or ML deviance
//! (primary + nested children eliminated family-by-family, crossed factors in
//! a dense tail Cholesky with [X y]) + BOBYQA θ-search over one diagonal θ
//! component per grouping. A degenerate single-intercept `ClusterSpec`
//! collapses to the per-cluster shrink-downdate arithmetic (up to FP
//! reassociation), so the q=1 validation corpus re-proves on this machine.
//!
//! Engine-resident: ALL Gaussian mixed (LMM) specs dispatch here through the
//! unified fit core — the single-random-intercept shape is a `FitKind::Lmm`/BOBYQA
//! case like any other, from every tier (stable and loop). A scalar closed-form
//! solve for this shape would only save the per-call workspace allocation that
//! the reusable `FitWorkspace` (allocated once per (p, max_clusters) shape)
//! already avoids for every shape, so BOBYQA's θ-search itself costs under 6%
//! of runtime here. BOBYQA also scales better at high cluster counts than a
//! scalar per-cluster downdate (`O(n_clusters·P)` per evaluation, with no
//! counterpart in `reml_deviance`'s balanced-collapse path), and matches a
//! scalar solve's β̂ to ~1e-9, so there is no accuracy axis to trade either.
//!
//! Hot-loop invariants:
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
use faer::{Mat, MatMut, MatRef};
use std::sync::OnceLock;

use crate::scalar::Scalar;
use crate::standardize::ColumnScaling;

mod kernel;
pub(crate) mod pg_grad;
mod re_map;
pub(crate) mod satterthwaite;
#[cfg(test)]
mod tests;

pub(crate) use kernel::{
    group_regression_shape, precompute_balanced_collapse, precompute_group_regressions, DesignQr,
    GRAM_PIVOT_MIN,
};
pub use kernel::{reml_deviance, LmmSuffStats};
use re_map::{pack_vech, refactor, PivotFloor};

/// θ start — DIAGONAL vech entries only; off-diagonals cold-start at 0
/// (unit diagonal, the lme4/MixedModels.jl default — the
/// `blind_theta_and_bounds` shape). Cold start per fit; no warm-start
/// across sims (would re-import cross-grid-point path dependence).
pub const THETA0: f64 = 1.0;
/// Per-component θ upper box (θ ≤ 1e3).
pub const THETA_HI: f64 = 1e3;
/// Initial trust radius of the LMM θ search from the blind start (diagonals
/// `THETA0`). Must be ≤ 1.0: PRIMA's start projection moves an x₀ that lies
/// within rho_begin of a bound, and the box is [−THETA_HI, THETA_HI] on every
/// coordinate (`blind_theta_and_bounds`), so θ₀ = 1.0 stays clear of both
/// bounds at any rho_begin ≤ 1.0. Box width 2e3 ≥ 2·rho_begin is the crate's
/// up-front validity requirement.
///
/// Measured 2026-10-02 (aarch64, evaluation counts) on a stratified sample of
/// the Gaussian cells of the accuracy grid (46 cells, one cold fit each),
/// against a start radius of 0.1: the evaluation total is 0.88, 0.81, 0.83,
/// 0.95 and 0.78 of it at n_θ 1, 2–3, 4–8, 9–17 and 18+, and 0.66 to 0.88 on
/// the four widest LMM cells (n_θ 21 to 40, four start nudges each), with no
/// deviance worse by more than 1e-6 and no convergence or singular flag
/// changed. 0.2 gains less, 1.0 gains less at n_θ 18+. A single cell's count
/// moves by up to ±15% under a start nudge of 1e-9, so only totals over many
/// cells carry the choice.
pub const RHO_BEGIN: f64 = 0.5;
/// Initial trust radius of a θ-only search that starts from a caller's θ₀
/// (`fit_warm`, the `loop_advanced` truth and carry starts): the LMM search and
/// GLMM stage 1 after the start guard kept the caller's θ₀. Such a start sits
/// near the optimum, where a wide first model only re-explores. On aarch64, on
/// parametric replicates of the grid cells started at the generating θ or at
/// the previous replicate's estimate, 0.05 takes 4 to 6% more LMM and 1 to 2%
/// more stage-1 evaluations than 0.1, and 0.2 takes 4% more on stage 1 (0.98 to
/// 1.00 of 0.1 on the LMM); on the loop-tier benchmark shapes, 0.03 to 0.2 move
/// the warm total by −1% to +4%.
pub const RHO_BEGIN_WARM: f64 = 0.1;
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
/// Initial trust radius of a GLMM search from the blind θ₀: the θ-only stage-1
/// search (`PqlThenJoint` and `ExactProfile`) and the joint search on the
/// `Joint` route (which includes nAGQ > 1). Measured 2026-10-02 (aarch64,
/// evaluation counts) against 0.1 on a stratified sample of the accuracy grid
/// (one cold fit per cell): the exact-profile evaluation total is 0.97, 1.01,
/// 0.96, 0.88 and 0.71 of it at n_θ 1, 2–3, 4–8, 9–17 and 18+, the PQL stage-1
/// total 1.03, 1.07, 0.84 and 0.86 at n_θ 2–3, 4–8, 9–17 and 18+, the `Joint`
/// route 0.98. 0.5 does as well on the widest exact-profile cells but loses at
/// n_θ 9–17. No fit ends at a worse deviance.
pub const GLMM_RHO_BEGIN: f64 = 0.2;
/// Initial trust radius of the joint `[θ | β̃ | ln φ]` search when it polishes
/// a stage-1 optimum (`PqlThenJoint`, and the `ExactProfile` fallback onto it).
/// The start is already close, so a wide first model only re-explores: on
/// aarch64, on the 58 `PqlThenJoint` cells of the same sample the polish takes
/// 0.83 of the evaluations it takes at 0.1, and a radius of 0.2 takes 1.53
/// times those at 0.03. Not used when stage 1 ends near a singular block: see
/// [`POLISH_NEAR_SINGULAR`].
pub const GLMM_RHO_POLISH: f64 = 0.03;
/// A stage-1 optimum with a diagonal |θ̃| below this (the search's own
/// coordinates) hands the joint polish a block near the singular boundary.
/// From such a start the 0.03 polish can stop on the flat boundary surface
/// short of the optimum: measured on the singular gamma-inverse grid cells,
/// one start nudge in eight ended 2.3e-3 deviance units above every other
/// fit, which a polish at `RHO_BEGIN_WARM` reaches. So that polish starts at
/// `RHO_BEGIN_WARM` instead. On the non-singular cells the rule never fires.
pub const POLISH_NEAR_SINGULAR: f64 = 1e-2;
/// Truth-start floor: a `Some(θ₀)` start is clamped to max(θ₀, this) so a
/// zero/near-zero true θ never starts the search on the boundary itself.
/// It sits above `PIN_THETA`, so a floored start never begins at a pinned
/// component.
pub const THETA_TRUTH_FLOOR: f64 = 0.01;
/// Pin threshold: a Converged diagonal component ≤ this is deterministically
/// pinned at exactly 0 and counted converged. 1e-4 aligns the class boundary
/// with the shipped τ̂≈0 detection, which pins `boundary_hit=1` fits at
/// θ = 1e-4.
///
/// **Tested on the INTERNAL θ**, the coordinate the solver minimizes over on
/// the fitted design (`crate::standardize`), not on θ in the design's own units.
/// Internal θ is what the optimizer's own stopping geometry lives on, and it is
/// the only version of the test that does not change its verdict when a
/// random-slope column is re-expressed in different units. The cost is that a
/// badly scaled design can be flagged differently from lme4's `isSingular`,
/// which applies the same 1e-4 to user-scale θ.
pub const PIN_THETA: f64 = 1e-4;
/// Ill-conditioning DETECTION floor for the LMM and GLMM routes, on the
/// scale-invariant per-column pivot ratio ([`crate::ols::min_pivot_ratio`]) of
/// X'V⁻¹X at θ̂ on the LMM, and of the β information its SE arm factors on the
/// GLMM. Below it the fixed-effect coefficients are
/// barely identified and the diagnostics channel says so — it is **not** a
/// refuse threshold, and neither route refuses a design on conditioning grounds
/// (see the LMM guard site for the two measured reasons).
///
/// Calibrated 2026-07-31 against a 1-ULP perturbation sweep of `y`. `1e-12` sits
/// about two decades above where the statistic stops tracking conditioning on
/// this route, which is the range where it still ranks designs. It is
/// conservative: β̂'s measured movement here is 9.1e-8 at pivot 9.7e-13, far
/// steadier than the `1e-15 / pivot` law the OLS and GLM routes obey.
///
/// That sweep is on a single-grouping design, where the dense and the sparse
/// kernel report the same pivot and the same movement. The floor does not track
/// the random-effect structure: with one crossed grouping added, β̂'s movement at
/// pivot 9.7e-11 — above the floor, so unflagged — is 4.3e-5 on the dense kernel
/// and 1.1e-4 on the sparse one, against 9.4e-10 without it.
///
/// The GLMM route was swept the same way on 2026-10-03 (worst of 16 ±1-ULP
/// sign patterns on `y`; a count or a success proportion re-rounded by one ULP
/// is still a valid response), on `x = [1, a, a + δ·b]`, `δ = 2⁻ᵏ`, Poisson-log
/// and binomial logit and probit with 10 trials, under one random intercept,
/// with a crossed one added, and with a crossed random slope. The pivot is the
/// one that route records: the Rx Schur complement, or the β block of the joint
/// Hessian, which agree to within 21% on every design swept. Where β is
/// profiled out exactly (the first two shapes), β̂ moves by 2.8e-8 to 2.6e-6 at
/// pivot 9e-13 to 1e-12, by at most 3.1e-6 down to pivot 1.4e-14, and by 1.6e-5
/// to 1.4e-4 at pivot 3.4e-15: `1e-12` sits about two decades above that knee,
/// conservative as on the LMM. On the joint `[θ | β]` search (the crossed
/// slope, packed layout) the search's own stopping width hides the
/// conditioning: β̂ moves by 9e-6 to 3.2e-4 at every pivot from 0.6 down to
/// 5e-15, so there the floor is the profiled routes' value, not one measured on
/// it.
///
/// No SOLVER path reads it: the kernels record the raw pivot and the comparison
/// happens once per route, in `LmmResultView::diagnostics` and
/// `GlmmResultView::diagnostics`, which is what fills
/// `FitDiagnostics::ill_conditioned`.
pub const PIVOT_MIN: f64 = 1e-12;

/// BOBYQA config for an n_theta-dimensional θ-search. `Config::new` supplies
/// the PRIMA default npt = 2n+1 (3 at n = 1); `apply_campaign_overrides` sets
/// max_fun = 1000·n. Test-only (reached only via the test-gated
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
/// LMM (`for_cluster_spec_ext`), the blind seed (`bobyqa_config`), and the GLMM
/// joint + stage-1 solvers (`glmm/workspace.rs`) — each evaluated against that
/// solver's own dimension (joint: n_theta + p).
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
/// The evaluation budget is `1000·n`, twice PRIMA's `500·n` default. The budget
/// sets the restart trigger together with `cycle_budget_frac`: the first cycle
/// restarts after 1/8 of the budget, so 1000·n at 1/8 fires at 750 evaluations
/// for n = 6 (500·n at 1/8: 375), and the second cycle then runs to `rho_end`.
/// Measured 2026-09-11 on the boundary-pin corpora (q3 4,500 fits, R1 12,600,
/// two AGQ cells of 3,000): a fit stopping before the trigger is bit-identical;
/// on q3, 17 fits improve by more than 1e-6 (largest −3.5e-4), 8 worsen
/// (largest +6.6e-5), the one fit that hit the 500·n cap converges at 796
/// evaluations, total evaluations 0.985×, KKT distribution unchanged; R1 and
/// the AGQ cells move no fit by more than 3e-6. No corpus fit reaches 1000·n.
/// A fit that never converges spends twice the evaluations before reporting it.
///
/// The four fields are written out even though they are `RestartConfig::new()`'s
/// own defaults: the schedule is adopted by value, so a later bobyqa release
/// changing its defaults cannot silently move this fit path.
pub(crate) fn apply_campaign_overrides(config: &mut Config, n: usize) {
    config.max_fun = 1000 * n;
    // Off: at the full `npt = (n+1)(n+2)/2` (every one-dimensional search) PRIMA's
    // RESCUE test is decided by rounding alone and spends evaluations; Powell's
    // original factor does not. On the accuracy grid's one-θ cells: 15% fewer
    // evaluations, deviance equal to round-off. Other `npt` values are unaffected.
    config.prima_parity = false;
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
/// count. For `q == 1` (intercept only) `vech_start` is the scalar θ index and
/// the block is a single variance.
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
    /// Gram from `s_raw`. Empty iff `primary_q == 1`.
    pub primary_slope_cols: Vec<usize>,
    /// θ-indices of the diagonal variance components (pinnable), in
    /// `boundary_rate_per_component` order: the `q_p` primary vech diagonals
    /// (column-major — the diagonal of column d sits at offset `Σ_{j<d}(q_p−j)`)
    /// then the extra-grouping scalars. q_p=1 ⇒ `[0, extras…]`; q_p=2 ⇒
    /// `[0, 2, extras…]`; off-diagonal vech entries excluded. Computed once per
    /// workspace by `compute_diagonal_theta` (the vech-diagonal walk lives there,
    /// single-sourced) so the per-fit pin loop borrows instead of reallocating.
    pub diagonal_theta: Vec<usize>,
    /// Parallel to `diagonal_theta`: for diagonal `k` at column `d` of a
    /// `q`-wide vech block, `q − d` — the diagonal's own vech run, diagonal
    /// entry first, followed by its `q − d − 1` off-diagonal entries below it
    /// in the same column, contiguous at `diagonal_theta[k] + 1 ..
    /// diagonal_theta[k] + diagonal_run_len[k]` (column-major vech packs a
    /// column's rows contiguously). `1` when the diagonal has no off-diagonal
    /// below it. Computed once per workspace by `compute_diagonal_run_len`,
    /// the companion walk to `compute_diagonal_theta`.
    pub diagonal_run_len: Vec<usize>,
    /// Solver-path gate: true iff any extra grouping carries a random slope
    /// (`q_g > 1`). The scalar tail in `reml_deviance` stays byte-identical when
    /// false; the blocked per-factor `Λ_g` tail is taken when true.
    pub extra_slopes_any: bool,
    /// Per-extra-grouping RE width `q_g = 1 + #slopes`, DECLARATION order (parallel
    /// to `extra_offsets`). Sizes each grouping's RE-column block (`level·q_g + d`)
    /// and its `vech(Λ_g)`. All `1` for intercept-only extras.
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
/// every extra `q == 1` this is `[primary diagonals, base+0, base+1, …]`.
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

/// Companion walk to `compute_diagonal_theta`, single source of truth for
/// `LmmGroupings::diagonal_run_len`: for each diagonal, `q − d` where `d` is
/// its column within its `q`-wide vech block. Must walk the same
/// `(primary_q, extra_qs)` factor/column loop as `compute_diagonal_theta` —
/// the two are kept as separate loops rather than one returning both, so a
/// future edit to the walk itself has to be made in two obviously-named
/// functions instead of silently drifting between a tuple and a caller
/// working out which half is which.
fn compute_diagonal_run_len(primary_q: usize, extra_qs: &[usize]) -> Vec<usize> {
    let mut run_len = Vec::with_capacity(primary_q + extra_qs.len());
    for &q in std::iter::once(&primary_q).chain(extra_qs.iter()) {
        for d in 0..q {
            run_len.push(q - d);
        }
    }
    run_len
}

impl LmmGroupings {
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

    /// As [`Self::from_cluster_spec`], plus the resolved `[X y]` column indices of each
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
        // nested children + crossed tail follow, shifted up by the
        // (q_p−1)·n_primary slope columns. q_p=1 ⇒ `prim_width == n_primary`.
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
        // structures. Routing enforces the `n_extras <= MAX_EXTRA_GROUPINGS`
        // NoZ-envelope check (not this constructor), so dense code is never
        // handed an over-envelope design.
        // θ order is declaration order: each extra owns a vech(Λ_g) block of
        // q_g(q_g+1)/2 slots starting at `vech_start`, packed after the primary
        // vech. For all q_g == 1, `vech_start == base_theta + g`.
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
        // Nested children are q_nested RE columns each (q_nested == 1 ⇒ a
        // single indicator column).
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
            diagonal_run_len: compute_diagonal_run_len(q_p, &extra_qs),
            extra_slopes_any,
            extra_q: extra_qs,
            extra_slope_cols,
        }
    }

    /// Primary vech (`q_p(q_p+1)/2`) + a `vech(Λ_g)` block per extra grouping
    /// (`q_g(q_g+1)/2` each). With every `q_g == 1` this is
    /// `primary vech + #extras`.
    pub fn n_theta(&self) -> usize {
        let prim = self.primary_q * (self.primary_q + 1) / 2;
        let nested = self.nested.map(|nf| nf.q * (nf.q + 1) / 2).unwrap_or(0);
        let crossed: usize = self.crossed.iter().map(|cf| cf.q * (cf.q + 1) / 2).sum();
        prim + nested + crossed
    }
    /// Columns eliminated family-by-family: the `q_p` primary RE cols per level
    /// plus nested children (`q_nested` cols each). (`k_crossed = k_total −
    /// k_family` is the dense tail.)
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
    /// (oversized core) falls through to the packed `glmm::pirls_solve_packed`.
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
    /// variances + extra scalars) start at THETA0; off-diagonal vech entries
    /// start at 0. Every entry, diagonal included, gets the signed box
    /// [−HI, HI]. q_p=1 ⇒ the all-diagonal shape (θ₀ = `[THETA0;n]`).
    ///
    /// The diagonals are deliberately NOT boxed at 0. Σ = ΛΛᵀ is unchanged when
    /// a whole column of Λ is negated, so the deviance is even in each block
    /// column; with Λ_jj ≥ 0 enforced, the two sign halves of column j meet
    /// only on the face Λ_jj = 0, and a search that reaches that face from
    /// the wrong half (the entries below with the sign that does not lead to
    /// the optimum) sees the deviance rise in every in-box direction and
    /// stops there, a deviance unit or more above the optimum — the exit is
    /// first-order stationary on the box, so no local check tells it apart.
    /// With the bound gone the face is an interior point of a smooth
    /// function and the search walks through it into the other half on its
    /// own. MixedModels.jl searches the same unbounded box. Every pin site
    /// calls `fix_column_signs` on the endpoint, so a negative diagonal never
    /// leaves the crate.
    ///
    /// The box shapes BOBYQA's interpolation set and trust-region steps on
    /// every fit, boundary or not.
    pub fn blind_theta_and_bounds(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n_theta();
        let mut theta = vec![THETA0; n];
        let lower = vec![-THETA_HI; n];
        let upper = vec![THETA_HI; n];
        let diag = self.diagonal_theta();
        // Off-diagonal vech entries (primary AND every extra factor's Λ_g) get a
        // blind start of 0; diagonals keep θ₀ = THETA0. q_g=1 extras are
        // all-diagonal, so they are untouched.
        for (i, t) in theta.iter_mut().enumerate() {
            if !diag.contains(&i) {
                *t = 0.0; // off-diagonal blind start
            }
        }
        (theta, lower, upper)
    }
}

// ---------------------------------------------------------------------------
// LmmFitScratch
// ---------------------------------------------------------------------------

/// The criterion an LMM kernel minimizes over θ: the profiled REML deviance
/// or the profiled ML deviance. Set when the workspace is built and frozen
/// with it, so one workspace never fits under two criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Criterion {
    Reml,
    Ml,
}

impl Criterion {
    /// The residual degrees of freedom `ν` the criterion's σ̂² divides by and
    /// its log-likelihood constant counts: `n − p` under REML, `n` under ML.
    /// `n ≤ p` gives 0 under REML (no fit there).
    pub(crate) fn nu(self, n: usize, p: usize) -> usize {
        match self {
            Criterion::Reml => n.saturating_sub(p),
            Criterion::Ml => n,
        }
    }
}

/// Per-fit scratch, allocated once per (p, max_clusters). All `f64`: the LMM
/// deviance is evaluated at `f64` only (its θ-gradient on the per-group
/// regression route is the closed form in [`pg_grad`]); the dual-number
/// derivatives of the GLMM path live in `src/glmm/`.
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
    /// once per eval, then factored out-of-place into `tail_l` below.
    pub tail: Vec<f64>,
    /// Out-of-place `chol_lower` output for `tail` (`t_dim²`, same layout):
    /// `tail` must survive UNFACTORED for `recover_ranef_family`, which
    /// re-factors it at θ̂ (see that function's doc for the contract).
    pub tail_l: Vec<f64>,
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
    /// `w·t_dim` staging buffer for `precompute_balanced_collapse`'s per-family
    /// raw rows, its own field so the θ-independent precompute does not borrow
    /// the per-eval `bt`.
    pub collapse_stage: Vec<f64>,
    /// t_dim² combine scratch for the collapse downdate (lower triangle used);
    /// its first w slots double as the A⁻¹ forward-solve temp.
    pub comb: Vec<f64>,
    /// w×w row-major A(θ)⁻¹, rebuilt per eval on the collapse path.
    pub a_inv: Vec<f64>,
    /// Active balanced families (prefix length). 0 = collapse off → the
    /// per-family loop runs (the fallback and the pre-F behaviour).
    pub collapse_n_active: usize,
    /// Per-group regression route (`precompute_group_regressions`): for each
    /// active group, `B_g = G_g⁻¹` as a row-major `q×q` block (lower triangle
    /// read), `q_p²` per group. Empty unless the shape can take the route
    /// (random slopes on the primary grouping, nothing else). θ-independent,
    /// refreshed once per fit.
    pub pg_b: Vec<f64>,
    /// Per active group, `A_g = G_g⁻¹ Z_g′[X̃ y]` as `q` contiguous `m`-long
    /// rows (the layout [`fam_forward_solve`] reads). Same lifetime as `pg_b`.
    pub pg_a: Vec<f64>,
    /// `m×m` column-major lower triangle of `P₀ = Σ_g [X̃ y]_g′(I − H_g)[X̃ y]_g`,
    /// the within-group residual cross-products (`H_g` the hat matrix of
    /// `Z_g`): the θ-free part of the tail. Same lifetime as `pg_b`.
    pub pg_p0: Vec<f64>,
    /// `Σ_g ln|L_g|` over the active groups, `G_g = L_g L_g′`: half of the
    /// θ-free `Σ_g ln|G_g|` term of `ln|V|`.
    pub pg_log_g_half: f64,
    /// Groups the route iterates (non-empty levels). 0 = route off: the
    /// family loop runs.
    pub pg_n_active: usize,
    /// `q×q` row-major `D = ΛΛ′`, rebuilt once per evaluation on the route.
    pub pg_d: Vec<f64>,
    /// Per active group, `L_g = chol(D + B_g)` (row-major lower, `q²`), kept by
    /// the value pass for the θ-gradient ([`pg_grad::pg_gradient`]). Same
    /// sizing as `pg_b`.
    pub pg_l: Vec<f64>,
    /// The gradient's stacked-solve block, `p × q·PG_GRAD_BLOCK` column-major
    /// (`N_gᵀ` for one block of groups), and its `m_g = M_g e` entries.
    pub pg_nt: Vec<f64>,
    /// See `pg_nt`.
    pub pg_mv: Vec<f64>,
    /// `β̂` off the border factor, `p` entries, for the gradient.
    pub pg_beta: Vec<f64>,
    /// m×m trailing block of the tail factor — identical semantics to the
    /// augmented [X y] factor; every recovery step reads only this.
    pub factor: Mat<f64>,
    /// Spherical conditional modes `û` at θ̂ over the full RE-column set
    /// (elimination order, length `k_total`), written once per fit by
    /// [`recover_ranef`]. Its OWN buffer, not [`LmmRecovery::u`]: that one is
    /// the standard-error forward-solve scratch, which runs first and would be
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
    /// σ̂² at the last evaluation's θ, by the criterion: `r²/(n − p)` under
    /// REML, `r²/n` under ML (`r² = L[p,p]²` off the augmented factor).
    pub sigma_sq: f64,
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
    /// (column-major lower-tri). Read again by `recover_ranef_blocked` after the
    /// fit, so it must stay unfactored — the Cholesky factor goes in `blocked_l`.
    pub blocked_p: Vec<f64>,
    /// dim×dim Cholesky factor of `blocked_p`, filled per θ-eval by a copy +
    /// in-place factor at `Par::Seq` (see `reml_deviance_blocked` in kernel.rs).
    pub blocked_l: Vec<f64>,
    /// The criterion the deviance evaluates. `Criterion::Reml` from
    /// [`Self::with_groupings`]; the fit core sets it at build.
    pub(crate) criterion: Criterion,
}

impl LmmFitScratch {
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
        // Per-group regression scratch only on the shape that route admits.
        let pg_groups = if group_regression_shape(g) {
            g.n_primary
        } else {
            0
        };
        let pg_q = if pg_groups > 0 { g.primary_q } else { 0 };
        LmmFitScratch {
            fam_a: vec![0.0; w * w],
            bt: vec![0.0; g.n_primary * w * t_dim],
            tail: vec![0.0; t_dim * t_dim],
            tail_l: vec![0.0; t_dim * t_dim],
            lam_x: vec![0.0; g.k_crossed()],
            prim_lam: vec![0.0; q2],
            prim_gram: vec![0.0; q2],
            fam_gram: vec![0.0; npairs * t_dim * t_dim],
            collapse_stage: vec![0.0; w * t_dim],
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
            pg_b: vec![0.0; pg_groups * pg_q * pg_q],
            pg_a: vec![0.0; pg_groups * pg_q * m],
            pg_p0: vec![0.0; if pg_groups > 0 { m * m } else { 0 }],
            pg_log_g_half: 0.0,
            pg_n_active: 0,
            pg_d: vec![0.0; pg_q * pg_q],
            pg_l: vec![0.0; pg_groups * pg_q * pg_q],
            pg_nt: vec![0.0; p * pg_q * pg_groups.min(pg_grad::PG_GRAD_BLOCK)],
            pg_mv: vec![0.0; pg_q * pg_groups.min(pg_grad::PG_GRAD_BLOCK)],
            pg_beta: vec![0.0; if pg_groups > 0 { p } else { 0 }],
            factor: Mat::zeros(m, m),
            ranef_u: vec![0.0; g.k_total],
            ranef_ok: false,
            ranef_ux: vec![0.0; g.k_crossed()],
            ranef_rhs: vec![0.0; w],
            sigma_sq: f64::NAN,
            blocked_lam: vec![0.0; blocked_kk],
            blocked_g: vec![0.0; blocked_kk],
            blocked_tmp: vec![0.0; blocked_kk],
            blocked_p: vec![0.0; blocked_dim * blocked_dim],
            blocked_l: vec![0.0; blocked_dim * blocked_dim],
            criterion: Criterion::Reml,
        }
    }
}

// ---------------------------------------------------------------------------
// LmmRecovery / LmmKernel — the once-at-θ̂ buffers, and the objective behind them.
// ---------------------------------------------------------------------------

/// The `p`-sized buffers the once-at-θ̂ recovery block fills: β̂, its per-target
/// variance and squared Wald statistic, and the joint-Wald scratch. Kernel-
/// independent — both objectives leave the same augmented `[X y]` factor, and
/// every step here reads only that — so they live beside [`LmmKernel`] rather
/// than inside it.
pub struct LmmRecovery {
    /// Fixed-effect estimates β̂, predictor-indexed.
    pub betas: Vec<f64>,
    /// The same estimates on the orthogonalized design `X̃ = X·U⁻¹` the kernels
    /// accumulate (`DesignQr`): `β̃ = U·β̂`. The conditional-mode recovery reads
    /// these, since its Grams are on `X̃`.
    pub beta_tilde: Vec<f64>,
    /// Per-target `Var(β̂_j)`; non-target slots are never written.
    pub var_diag: Vec<f64>,
    /// Per-target `t² = β̂_j²/Var(β̂_j)`; non-target slots are never written.
    pub t_sq: Vec<f64>,
    /// `L_XX⁻¹e_j` forward-solve scratch for `var_diag` (first `p` slots).
    pub u: Vec<f64>,
    /// p×p; column `j` holds `u` as solved for target `j` — the columns
    /// [`joint_wald_chi_sq`] reads.
    pub joint_cols: Mat<f64>,
    /// p×p and length-p scratch of [`joint_wald_chi_sq`].
    pub joint_qr: Mat<f64>,
    pub joint_z: Vec<f64>,
}

impl LmmRecovery {
    pub fn new(p: usize) -> Self {
        LmmRecovery {
            betas: vec![0.0; p],
            beta_tilde: vec![0.0; p],
            var_diag: vec![0.0; p],
            t_sq: vec![0.0; p],
            u: vec![0.0; p],
            joint_cols: Mat::zeros(p, p),
            joint_qr: Mat::zeros(p, p),
            joint_z: vec![0.0; p],
        }
    }
}

/// The profiled objective (REML or ML) and the augmented factor it leaves:
/// dense sufficient statistics with the family-blocked Cholesky, or the
/// sparse-Z Gram/Schur workspace. [`fit_lmm`] reads both through the same five
/// calls.
pub(crate) enum LmmKernel {
    Dense {
        suff: Box<LmmSuffStats>,
        fit: Box<LmmFitScratch>,
    },
    /// `ws` is built per call by `fit::lmm::accumulate_lmm_rows` (the Gram
    /// scatter is the constructor) and is `None` before the first accumulate;
    /// `g` is the grouping structure it is built from, carrying the RE column
    /// scales of the current design.
    Sparse {
        g: Box<LmmGroupings>,
        ws: Option<Box<crate::sparse::SparseLmmWorkspace>>,
    },
}

impl LmmKernel {
    /// Profiled deviance at `theta` under the kernel's criterion (REML or ML),
    /// leaving the augmented `[X y]` factor [`Self::factor`] reads.
    /// `f64::INFINITY` is the failure surface.
    pub(crate) fn deviance(&mut self, theta: &[f64]) -> f64 {
        match self {
            LmmKernel::Dense { suff, fit } => reml_deviance(theta, suff, fit),
            LmmKernel::Sparse { ws, .. } => {
                crate::sparse::sparse_reml_deviance(theta, Self::sparse_ws_mut(ws))
            }
        }
    }

    /// Keep back what the NEXT evaluation would otherwise throw away, so
    /// [`Self::recover_ranef`] can back-substitute the conditional modes off
    /// it. The dense kernel needs nothing: it re-factors from `fit.tail` /
    /// `fit.blocked_p`, which every evaluation leaves behind.
    pub(crate) fn prepare_recovery(&mut self) {
        match self {
            LmmKernel::Dense { .. } => {}
            LmmKernel::Sparse { ws, .. } => Self::sparse_ws_mut(ws).arm_recovery(),
        }
    }

    /// The augmented `m×m` `[X y]` factor the last evaluation left (lower
    /// triangular, y row at index `p`): β̂, `Var(β̂)` and the pivot ratio all
    /// come off it.
    pub(crate) fn factor(&self) -> MatRef<'_, f64> {
        match self {
            LmmKernel::Dense { fit, .. } => fit.factor.as_ref(),
            LmmKernel::Sparse { ws, .. } => Self::sparse_ws(ws).factor.as_ref(),
        }
    }

    /// The fixed-effect factor `U` the kernel accumulated `X` through.
    pub(crate) fn design_qr(&self) -> &DesignQr {
        match self {
            LmmKernel::Dense { suff, .. } => &suff.design_qr,
            LmmKernel::Sparse { ws, .. } => &Self::sparse_ws(ws).design_qr,
        }
    }

    /// Map the last evaluation's `[X̃ y]` factor to the `[X y]` one in place
    /// ([`DesignQr::map_factor`]).
    pub(crate) fn map_factor_to_design(&mut self) {
        match self {
            LmmKernel::Dense { suff, fit } => suff.design_qr.map_factor(fit.factor.as_mut()),
            LmmKernel::Sparse { ws, .. } => {
                let ws = Self::sparse_ws_mut(ws);
                ws.design_qr.map_factor(ws.factor.as_mut());
            }
        }
    }

    /// σ̂² at the last evaluation's θ.
    pub(crate) fn sigma_sq(&self) -> f64 {
        match self {
            LmmKernel::Dense { fit, .. } => fit.sigma_sq,
            LmmKernel::Sparse { ws, .. } => Self::sparse_ws(ws).sigma_sq,
        }
    }

    /// Spherical conditional modes `û` at θ̂ from the state the last evaluation
    /// left; `betas` must hold β̂. Returns whether the recovery succeeded.
    pub(crate) fn recover_ranef(&mut self, theta: &[f64], betas: &[f64]) -> bool {
        match self {
            LmmKernel::Dense { suff, fit } => {
                recover_ranef(theta, suff, fit, betas);
                fit.ranef_ok
            }
            LmmKernel::Sparse { ws, .. } => {
                let ws = Self::sparse_ws_mut(ws);
                let u = crate::sparse::sparse_recover_u(ws, betas);
                // Armed for this one back-substitution: later evaluations at
                // other θ (the Satterthwaite pass's) copy nothing into it.
                ws.rec = None;
                match u {
                    Some(u) => {
                        ws.ranef_u = u;
                        true
                    }
                    None => {
                        ws.ranef_u.clear();
                        false
                    }
                }
            }
        }
    }

    /// Drop any `û` a previous fit on this workspace left, so [`Self::ranef_u`]
    /// reports none for a fit that reached no endpoint.
    pub(crate) fn invalidate_recovery(&mut self) {
        match self {
            LmmKernel::Dense { fit, .. } => fit.ranef_ok = false,
            LmmKernel::Sparse { ws, .. } => Self::sparse_ws_mut(ws).ranef_u.clear(),
        }
    }

    /// `û` over the full RE-column set, or empty when no recovery succeeded for
    /// the current fit.
    pub(crate) fn ranef_u(&self) -> &[f64] {
        match self {
            LmmKernel::Dense { fit, .. } => {
                if fit.ranef_ok {
                    &fit.ranef_u
                } else {
                    &[]
                }
            }
            LmmKernel::Sparse { ws, .. } => &Self::sparse_ws(ws).ranef_u,
        }
    }

    /// Grouping structure, carrying the RE column scales of the current design.
    pub(crate) fn groupings(&self) -> &LmmGroupings {
        match self {
            LmmKernel::Dense { suff, .. } => &suff.groupings,
            LmmKernel::Sparse { g, .. } => g,
        }
    }

    /// Rows behind the accumulated Grams. Equal to the caller's `n` except on
    /// the degenerate dense `p == 0` shape, where nothing was accumulated.
    pub(crate) fn n_rows(&self) -> usize {
        match self {
            LmmKernel::Dense { suff, .. } => suff.n_rows,
            LmmKernel::Sparse { ws, .. } => Self::sparse_ws(ws).n,
        }
    }

    /// The criterion the deviance evaluates, frozen at the workspace build.
    pub(crate) fn criterion(&self) -> Criterion {
        match self {
            LmmKernel::Dense { fit, .. } => fit.criterion,
            LmmKernel::Sparse { ws, .. } => Self::sparse_ws(ws).criterion,
        }
    }

    fn sparse_ws(
        ws: &Option<Box<crate::sparse::SparseLmmWorkspace>>,
    ) -> &crate::sparse::SparseLmmWorkspace {
        ws.as_deref().expect(SPARSE_NOT_ACCUMULATED)
    }

    fn sparse_ws_mut(
        ws: &mut Option<Box<crate::sparse::SparseLmmWorkspace>>,
    ) -> &mut crate::sparse::SparseLmmWorkspace {
        ws.as_deref_mut().expect(SPARSE_NOT_ACCUMULATED)
    }
}

/// The sparse kernel's Gram scatter IS its constructor, so every call below the
/// accumulate step has a workspace by construction; reaching this is a wiring
/// bug, not a runtime condition.
const SPARSE_NOT_ACCUMULATED: &str = "sparse LMM kernel used before accumulate_lmm_rows";

// ---------------------------------------------------------------------------
// LmmWorkspace — everything a fit needs, allocated once per problem shape.
// ---------------------------------------------------------------------------

/// Per-problem-shape scratch: the profiled objective and its buffers, the
/// once-at-θ̂ recovery buffers, and θ-solver state, allocated once and reused
/// across fits of the same shape.
pub struct LmmWorkspace {
    /// The objective and the augmented factor it leaves.
    pub(crate) kernel: LmmKernel,
    /// β̂ and the standard-error/Wald buffers, filled once at θ̂.
    pub recovery: LmmRecovery,
    /// BOBYQA solver state for the blind start (`RHO_BEGIN`). `Bobyqa::new` is
    /// the crate's only allocation site; `minimize` is zero-alloc on the warm
    /// path.
    pub solver: Bobyqa,
    /// Solver for a caller-supplied θ₀: `config` at `RHO_BEGIN_WARM`. Built on
    /// the first such fit, so a workspace that only fits cold never pays for
    /// it and a warm loop allocates it once.
    pub(crate) solver_warm: Option<Bobyqa>,
    /// The blind-start solver's config, kept to build `solver_warm`.
    pub(crate) config: Config,
    /// BFGS search buffers for the per-group route (sized on first use, reset
    /// at the start of every search: nothing carries between fits).
    pub(crate) bfgs: crate::optim::bfgs::BfgsState,
    /// θ in/out buffer for `minimize`; holds θ̂ (post-pin) after `fit_lmm`.
    pub theta: Vec<f64>,
    /// Lower box bound, −THETA_HI on every entry (`blind_theta_and_bounds`).
    pub lower: Vec<f64>,
    /// Upper box bound, THETA_HI on every entry.
    pub upper: Vec<f64>,
    /// `family::weight_scale` of the rows `fit::lmm::accumulate_lmm_rows` last
    /// accumulated (`1.0` unweighted): the kernel runs on ŵ = w/s, so its σ̂²
    /// is the raw-scale σ̂² divided by s.
    pub(crate) weight_scale: f64,
    /// The scaled prior weights ŵ = w/s that `fit::lmm::accumulate_lmm_rows`
    /// hands the kernel (√ŵ once the sparse kernel has taken them). Empty
    /// until the first weighted draw, then refilled in place, so a weighted
    /// loop allocates it once.
    pub(crate) w_scaled: Vec<f64>,
    /// The loop tier's Satterthwaite pass buffers
    /// ([`satterthwaite::pass_targets`]), built on its first call on the
    /// workspace. The stable path's pass builds its own.
    pub(crate) pass_scratch: Option<Box<satterthwaite::PassScratch>>,
}

impl LmmWorkspace {
    /// Allocates a workspace for a single-intercept grouping at `max_clusters`.
    /// Test-only since the retirement of the raw `LmmWorkspace` loop surface —
    /// the live core path builds through [`Self::for_cluster_spec_ext`].
    #[cfg(test)]
    pub fn new(p: usize, max_clusters: usize) -> Self {
        let spec = crate::test_support::intercept_only_spec(crate::Sizing::FixedClusters {
            n_clusters: max_clusters as u32,
        });
        Self::with_groupings(p, LmmGroupings::from_cluster_spec(&spec, max_clusters, &[]))
    }

    /// Workspace for a validated non-degenerate ClusterSpec at max_n. Carries
    /// the blind θ start and its BOBYQA schedule (`RHO_BEGIN`, `RHO_END`).
    /// `slope_cols` are the x_full column indices for the primary slopes
    /// (`spec.cluster_slope_design_cols` as usize); pass `&[]` for callers without slopes.
    /// Test-only convenience over [`Self::for_cluster_spec_ext`] (the live path);
    /// no production caller constructs a raw `LmmWorkspace` directly.
    #[cfg(test)]
    pub fn for_cluster_spec(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
    ) -> Self {
        Self::for_cluster_spec_ext(p, cluster, max_n, slope_cols, &[], false)
    }

    /// As [`for_cluster_spec`], plus each extra grouping's resolved slope x-columns
    /// (`extra_slope_cols`, declaration order; `&[]` for intercept-only extras) —
    /// the crossed/nested-slopes entry. Both `glmm::fit` (standalone) and
    /// `glmm::mcpower` bind through here.
    ///
    /// `sparse` picks the kernel: it is `fit::classify_design(spec) ==
    /// Solver::Sparse` on the live path, and the sparse kernel is a superset of
    /// the dense one, so a caller may pass `true` on an in-envelope design to
    /// cross-check the two at the same θ schedule.
    pub fn for_cluster_spec_ext(
        p: usize,
        cluster: &crate::ModelSpec,
        max_n: usize,
        slope_cols: &[usize],
        extra_slope_cols: &[Vec<usize>],
        sparse: bool,
    ) -> Self {
        let groupings =
            LmmGroupings::from_cluster_spec_ext(cluster, max_n, slope_cols, extra_slope_cols);
        let n_theta = groupings.n_theta();
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
        config.rho_begin = RHO_BEGIN;
        config.rho_end = RHO_END;
        config.npt = npt;
        apply_campaign_overrides(&mut config, n_theta);
        let (theta, lower, upper) = groupings.blind_theta_and_bounds();
        // The sparse kernel carries no dense O(K) suff-stats / fit scratch —
        // that is what it exists to avoid — and its Gram scatter is its
        // constructor, so it is empty until the first accumulate.
        let kernel = if sparse {
            LmmKernel::Sparse {
                g: Box::new(groupings),
                ws: None,
            }
        } else {
            let fit = Box::new(LmmFitScratch::with_groupings(p, &groupings));
            LmmKernel::Dense {
                suff: Box::new(LmmSuffStats::with_groupings(p, groupings)),
                fit,
            }
        };
        LmmWorkspace {
            kernel,
            recovery: LmmRecovery::new(p),
            solver: Bobyqa::new(n_theta, config)
                .expect("BOBYQA config constants are valid by construction"),
            solver_warm: None,
            bfgs: Default::default(),
            config,
            theta,
            lower,
            upper,
            weight_scale: 1.0,
            w_scaled: Vec::new(),
            pass_scratch: None,
        }
    }

    /// Allocates the suff-stats accumulator, fit scratch, and BOBYQA solver
    /// state for the given grouping shape, once per problem shape. Test-only;
    /// production code reaches this only via [`Self::new`].
    #[cfg(test)]
    pub fn with_groupings(p: usize, groupings: LmmGroupings) -> Self {
        let n_theta = groupings.n_theta();
        let fit = Box::new(LmmFitScratch::with_groupings(p, &groupings));
        let (theta, lower, upper) = groupings.blind_theta_and_bounds();
        LmmWorkspace {
            kernel: LmmKernel::Dense {
                suff: Box::new(LmmSuffStats::with_groupings(p, groupings)),
                fit,
            },
            recovery: LmmRecovery::new(p),
            // The constants are valid by construction for the crate's checks
            // (npt default within bounds; box width 1e3 ≥ 2·RHO_BEGIN), so a
            // failure here is an engine bug, not a runtime branch.
            solver: Bobyqa::new(n_theta, bobyqa_config(n_theta))
                .expect("BOBYQA config constants are valid by construction"),
            solver_warm: None,
            bfgs: Default::default(),
            config: bobyqa_config(n_theta),
            theta,
            lower,
            upper,
            weight_scale: 1.0,
            w_scaled: Vec::new(),
            pass_scratch: None,
        }
    }

    /// The dense kernel's accumulated sufficient statistics. Panics on a sparse
    /// workspace, which has none.
    #[cfg(test)]
    pub fn suff_mut(&mut self) -> &mut LmmSuffStats {
        match &mut self.kernel {
            LmmKernel::Dense { suff, .. } => suff,
            LmmKernel::Sparse { .. } => panic!("sparse LMM workspace has no suff stats"),
        }
    }

    /// The dense kernel's deviance scratch. Panics on a sparse workspace, which
    /// has none.
    #[cfg(test)]
    pub fn fit_mut(&mut self) -> &mut LmmFitScratch {
        match &mut self.kernel {
            LmmKernel::Dense { fit, .. } => fit,
            LmmKernel::Sparse { .. } => panic!("sparse LMM workspace has no fit scratch"),
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

/// Rewrite every pinned RE block's Λ into the canonical Σ-preserving form and
/// report whether any θ entry moved. Run AFTER a pin loop has snapped diagonals
/// to exactly 0, and re-run the pin test on the result.
///
/// At a pinned diagonal `Λ_jj = 0` the entries BELOW it in column `j` are not
/// identified by Σ: rotating that column leaves `Σ = ΛΛ′` unchanged, and with
/// it the REML and Laplace deviances, β̂, `varcorr` and `ranef` — so BOBYQA can
/// stop anywhere on that flat circle. Re-factoring Σ with a
/// semidefinite-tolerant lower Cholesky picks the one representative whose
/// redundant column is exactly zero: the mass below a pinned diagonal folds
/// into the trailing diagonals, where it is a variance component the caller
/// can read. Σ is preserved by construction, so under those objectives this
/// changes the coordinate, not the fit.
///
/// The vector AGQ deviance is NOT a function of Σ alone: its product
/// Gauss–Hermite grid sits in the `u = Λ⁻¹b` coordinates, so the same Σ under
/// a rotated Λ integrates to a different value. `glmm::fit_glmm` therefore
/// canonicalizes only a copy on its AGQ route, for the pin flags, and keeps the
/// search's own Λ.
///
/// `Diagnostics::pinned` depends on it: without this fold, a pin flag could
/// sit on a column with a live off-diagonal, a direction in which Σ still
/// varies. With it, a pin flags a rank-deficient direction of Σ (zero variance
/// conditional on the earlier components of the block), which is a zero
/// variance component only when the rest of its row is zero too.
///
/// `theta` is the kernel's θ, on the fitted design. Row scaling commutes with
/// this factorization (`S·chol(Σ)` is lower triangular with the same zero
/// pattern and squares to `SΣS`), so a column's scale does not change what is
/// folded.
pub(crate) fn canonicalize_pinned_blocks(g: &LmmGroupings, theta: &mut [f64]) -> bool {
    let mut changed = false;
    let mut start = 0usize;
    // Same factor walk as `compute_diagonal_theta` / `fix_column_signs`
    // (primary, then each extra in declaration order) — change together.
    for &q in std::iter::once(&g.primary_q).chain(g.extra_q.iter()) {
        let len = q * (q + 1) / 2;
        // q == 1 has nothing below a diagonal to fold. Past MAX_PRIMARY_Q the
        // stack scratch below would not hold the block; only the sparse route
        // builds those, so such a block keeps whatever representative the
        // optimizer stopped at.
        if (2..=MAX_PRIMARY_Q).contains(&q) && canonicalize_block(&mut theta[start..start + len], q)
        {
            changed = true;
        }
        start += len;
    }
    changed
}

/// Sign canonicalization of a search endpoint: every block column whose
/// diagonal is negative is negated whole — the diagonal and every entry
/// below it in that column. The search box lets a diagonal go negative
/// (`blind_theta_and_bounds` carries the reason); this restores the
/// non-negative-diagonal Cholesky convention every reader of θ assumes.
/// Σ = ΛΛᵀ is unchanged BIT for bit — each entry of Σ is a sum of products
/// `Λ_rk·Λ_ck` over columns k, and `(−a)·(−b)` is the same double as `a·b` —
/// so the deviance is unchanged too and nothing needs re-evaluating. Runs at
/// every pin site on the raw endpoint, before the pin test, the canonical
/// block and the SE pass. Returns whether any column was negated.
///
/// Same block walk as [`canonicalize_pinned_blocks`] and [`fix_mode_signs`] —
/// change together. It
/// needs no scratch, so unlike that one it has no block-width limit and
/// covers the sparse route's wide blocks too.
pub(crate) fn fix_column_signs(g: &LmmGroupings, theta: &mut [f64]) -> bool {
    let mut changed = false;
    let mut start = 0usize;
    for &q in std::iter::once(&g.primary_q).chain(g.extra_q.iter()) {
        // Column-major vech: column d of a q-wide block is the run of q − d
        // entries starting at its diagonal.
        let mut off = start;
        for d in 0..q {
            if theta[off] < 0.0 {
                for v in theta[off..off + (q - d)].iter_mut() {
                    *v = -*v;
                }
                changed = true;
            }
            off += q - d;
        }
        start += q * (q + 1) / 2;
    }
    changed
}

/// Companion of [`fix_column_signs`] for a GLMM's conditional modes. Call it on
/// the same θ just BEFORE that one: it reads the diagonal signs that one clears.
/// b = Λu, so negating column d of a block's Λ keeps b only if column d's u
/// entries, at every level of that factor, are negated too. A mode solve
/// warm-started from the result then starts next to the mode at the sign-fixed
/// θ instead of at its mirror image — which on a link with more than one PIRLS
/// basin (Gamma-inverse, `glmm::se`) can be a different mode.
/// `primary_slope_major` picks the primary RE-column layout: the sparse path's
/// `d·n_primary + level` (true) or the dense GLMM's `level·q_p + d` (false);
/// extra groupings are level-major, `extra_offsets[decl] + level·q + d`, on
/// both. Same block walk as [`fix_column_signs`] — change together.
pub(crate) fn fix_mode_signs(
    g: &LmmGroupings,
    theta: &[f64],
    u: &mut [f64],
    primary_slope_major: bool,
) {
    let q_p = g.primary_q;
    let mut off = 0usize;
    for d in 0..q_p {
        if theta[off] < 0.0 {
            for lvl in 0..g.n_primary {
                let c = if primary_slope_major {
                    d * g.n_primary + lvl
                } else {
                    lvl * q_p + d
                };
                u[c] = -u[c];
            }
        }
        off += q_p - d;
    }
    let nested = g.nested.map(|nf| {
        (
            nf.vech_start,
            nf.q,
            nf.decl,
            g.n_primary * g.nested_per_parent,
        )
    });
    let crossed = g
        .crossed
        .iter()
        .map(|cf| (cf.vech_start, cf.q, cf.decl, cf.n_levels));
    for (vech_start, q, decl, n_levels) in nested.into_iter().chain(crossed) {
        let base = g.extra_offsets[decl];
        let mut off = vech_start;
        for d in 0..q {
            if theta[off] < 0.0 {
                for lvl in 0..n_levels {
                    let c = base + lvl * q + d;
                    u[c] = -u[c];
                }
            }
            off += q - d;
        }
    }
}

/// One block of [`canonicalize_pinned_blocks`]: `vech` is that block's
/// column-major lower-triangular vech slice and `2 ≤ q ≤ MAX_PRIMARY_Q` its
/// width. Returns whether any entry moved.
fn canonicalize_block(vech: &mut [f64], q: usize) -> bool {
    // With no diagonal at the pin threshold Λ is already the unique Cholesky
    // factor of its own Σ and the factorization below would only reproduce it
    // up to round-off. Returning early instead leaves every non-singular fit's
    // Λ bit-for-bit as the optimizer returned it.
    let mut off = 0usize;
    let mut any_pinned = false;
    for d in 0..q {
        any_pinned |= vech[off] <= PIN_THETA;
        off += q - d;
    }
    if !any_pinned {
        return false;
    }

    // Semidefinite factor of ΛΛᵀ with the `q·eps·tr Σ` pivot floor: a pinned
    // column stays exactly zero (`PivotFloor::Trace` carries the floor's
    // argument).
    let mut lam = [0.0_f64; MAX_PRIMARY_Q * MAX_PRIMARY_Q];
    primary_lambda(vech, q, &mut lam); // Λ lower-tri, row-major
    let mut scratch = [0.0_f64; 2 * MAX_PRIMARY_Q * MAX_PRIMARY_Q];
    refactor(q, &mut lam, &mut scratch, PivotFloor::Trace);
    let mut packed = [0.0_f64; MAX_PRIMARY_Q * (MAX_PRIMARY_Q + 1) / 2];
    let packed = &mut packed[..vech.len()];
    pack_vech(&lam, q, packed);
    let changed = vech
        .iter()
        .zip(packed.iter())
        .any(|(a, b)| a.to_bits() != b.to_bits());
    vech.copy_from_slice(packed);
    changed
}

/// Per-level primary Gram G_f (q×q, row-major) recovered from suff stats into
/// `gram`, no new accumulator: G[0][0]=n_f; G[0][a]=G[a][0]=Σ x_{a-1} over f;
/// G[a][b]=Σ x_{a-1} x_{b-1} over f. The slope covariates are raw-`x` rows, so
/// every entry sits in `s_raw`. Component d's RE col at level f is `d·n_primary + f`
/// (mirrors `from_cluster_spec`'s RE-column layout — change together).
fn primary_gram(suff: &LmmSuffStats, g: &LmmGroupings, f: usize, q: usize, gram: &mut [f64]) {
    let n_prim = g.n_primary;
    for v in gram[..q * q].iter_mut() {
        *v = 0.0;
    }
    gram[0] = suff.counts[f]; // G[0][0]
    for a in 1..q {
        let sa = suff.s_raw[(g.primary_slope_cols[a - 1], f)]; // Σ z_{a-1} over f
        gram[a] = sa;
        gram[a * q] = sa;
        for b in 1..=a {
            // Σ z_{a-1} z_{b-1} over f — slope_{a-1}'s subcol against slope_{b-1}'s level.
            let v = suff.s_raw[(g.primary_slope_cols[a - 1], b * n_prim + f)];
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
    let mut m_r = [0.0; MAX_PRIMARY_Q];
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

/// Family `f`'s `w×w` block `A_f = I + Λ_f′G_fΛ_f` into the lower triangle of
/// the row-major `fam_a` (`w = q_p + n_per`): the primary `q_p×q_p` block, then
/// the nested children's diagonals and their coupling to the primary.
///
/// Shared by [`reml_deviance`]'s family loop and [`recover_ranef`], so recovery
/// rebuilds the same `L_f` the deviance factored rather than restating the
/// assembly — the deviance keeps only the LAST family's factor, and recovery
/// needs all of them.
#[allow(clippy::too_many_arguments)] // one call site each; the alternative is a struct of borrows
fn assemble_fam_a(
    fam_a: &mut [f64],
    prim_gram: &mut [f64],
    prim_lam: &[f64],
    suff: &LmmSuffStats,
    f: usize,
    w: usize,
    th_p: f64,
    th_n: f64,
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
                                                              // cross-Gram (intercept = counts[child]; slope d = s_raw[(slope_col_d,
                                                              // child_re_col)]) through Λ_p, mirroring how the scalar path reads counts for the
                                                              // intercept↔child term. n_primary = primary level count (slope RE
                                                              // stride); np = children per parent (nested width) — kept distinct.
        for c in 0..np {
            // Nested child RE col = prim_width + f·np + c (prim_width = q_p·n_primary).
            let gcol = g.n_primary * g.primary_q + f * np + c;
            let n_c = suff.counts[gcol];
            for c2 in 0..np {
                fam_a[(q + c) * w + (q + c2)] = 0.0;
            }
            fam_a[(q + c) * w + (q + c)] = 1.0 + th_n * th_n * n_c;
            // Primary↔child: A[(q+c, e)] = θ_n · Σ_{d≥e} Λ_p[d,e] · Graw_d,
            // Graw_0 = n_c (intercept), Graw_d = Σ_{i∈child} x_{slope_{d-1}}.
            for e in 0..q {
                let mut acc = 0.0;
                for d in e..q {
                    let graw_d = if d == 0 {
                        n_c
                    } else {
                        suff.s_raw[(g.primary_slope_cols[d - 1], gcol)]
                    };
                    acc += prim_lam[d * q + e] * graw_d;
                }
                fam_a[(q + c) * w + e] = th_n * acc;
            }
        }
    } else {
        // parent–child counts = child row counts (a child's rows all lie
        // inside its parent).
        let n_f = suff.counts[f];
        fam_a[0] = 1.0 + th_p * th_p * n_f;
        for c in 0..np {
            let gcol = g.n_primary + f * np + c;
            let n_c = suff.counts[gcol];
            for c2 in 0..np {
                fam_a[(1 + c) * w + (1 + c2)] = 0.0;
            }
            fam_a[(1 + c) * w] = th_p * th_n * n_c;
            fam_a[(1 + c) * w + (1 + c)] = 1.0 + th_n * th_n * n_c;
        }
    }
}

/// Family `f`'s coupling `B_f` to the `[crossed | X y]` tail, written as `w`
/// contiguous `t_dim`-long columns (`bt_fam[r·t_dim + t]`) — the same slice
/// [`reml_deviance`] writes at `fit.bt`'s columns `f·w .. f·w+w`. Shared for
/// the same reason as [`assemble_fam_a`].
#[allow(clippy::too_many_arguments)] // as `assemble_fam_a`
fn assemble_fam_b(
    bt_fam: &mut [f64],
    lam_x: &[f64],
    prim_lam: &[f64],
    suff: &LmmSuffStats,
    f: usize,
    t_dim: usize,
    kx: usize,
    slope: bool,
    th_p: f64,
    th_n: f64,
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
                let mut brb = 0.0;
                for d in r..q {
                    let zeta = if d == 0 { zxb[f] } else { zxsb[d * n_prim + f] };
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
                let mut brj = 0.0;
                #[allow(clippy::needless_range_loop)]
                for d in r..q {
                    brj += prim_lam[d * q + r] * s_cols[d][j];
                }
                bcol[j] = brj;
            }
        }
        // Nested-child rows (q..q+np) — built at the shifted child RE col.
        for c in 0..np {
            let gcol = n_prim * q + f * np + c; // prim_width + f·np + c
            let off = (q + c) * t_dim;
            for b in 0..kx {
                bt_fam[off + b] = th_n * lam_x[b] * suff.zx[(gcol, b)];
            }
            let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
            let bcol = &mut bt_fam[off + kx..off + kx + m];
            for j in 0..m {
                bcol[j] = th_n * scol[j];
            }
        }
    } else {
        let s_f = suff.s.col(f).try_as_col_major().unwrap().as_slice();
        for b in 0..kx {
            bt_fam[b] = th_p * lam_x[b] * suff.zx[(f, b)];
        }
        {
            let bcol = &mut bt_fam[kx..kx + m];
            for j in 0..m {
                bcol[j] = th_p * s_f[j];
            }
        }
        for c in 0..np {
            let gcol = g.n_primary + f * np + c;
            let off = (1 + c) * t_dim;
            for b in 0..kx {
                bt_fam[off + b] = th_n * lam_x[b] * suff.zx[(gcol, b)];
            }
            let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
            let bcol = &mut bt_fam[off + kx..off + kx + m];
            for j in 0..m {
                bcol[j] = th_n * scol[j];
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
fn fam_forward_solve(bt_fam: &mut [f64], t_dim: usize, w: usize, fam_a: &[f64]) {
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
/// **Why this is a separate pass at all.** The profiled criterion (REML or ML)
/// is a determinant-and-quadratic-form identity computable straight off the factors,
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
pub(crate) fn recover_ranef(
    theta: &[f64],
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch,
    betas: &[f64],
) {
    fit.ranef_ok = false;
    let k = suff.groupings.k_total;
    if k == 0 || suff.n_rows == 0 {
        return;
    }
    fit.ranef_u[..k].fill(0.0);
    fit.ranef_ok = if suff.groupings.extra_slopes_any {
        recover_ranef_blocked(theta, suff, fit, betas)
    } else {
        recover_ranef_family(theta, suff, fit, betas)
    };
}

/// Recovery on the crossed/nested-slopes blocked path. `fit.blocked_p` still
/// holds the UNFACTORED penalized augmented matrix at θ̂ (`SeqLlt` builds a
/// fresh factor rather than working in place), so one re-factorization hands
/// back both halves at once: `L_ZZ` is its leading `k×k` block and `U` its
/// `[X y]` rows, `U[a][j] = L[(k+j), a]`.
fn recover_ranef_blocked(
    theta: &[f64],
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch,
    betas: &[f64],
) -> bool {
    let _ = theta; // Λ is reconstructed by the caller; only the factor is read here
    let g = &suff.groupings;
    let m = suff.m;
    let p = m - 1;
    let k = g.k_total;
    let dim = k + m;
    let pref = faer::MatRef::from_column_major_slice(&fit.blocked_p[..dim * dim], dim, dim);
    let Some(chol) = crate::linalg::SeqLlt::new(pref) else {
        return false;
    };
    let l = chol.L();
    // rhs = U_y − U_X β̂, then one upper-triangular back-substitution against
    // L_ZZᵀ over the whole RE-column set.
    let u = &mut fit.ranef_u;
    for a in 0..k {
        let mut acc = l[(k + p, a)];
        for j in 0..p {
            acc -= l[(k + j, a)] * betas[j];
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
fn recover_ranef_family(
    theta: &[f64],
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch,
    betas: &[f64],
) -> bool {
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
        let Some(chol) = crate::linalg::SeqLlt::new(tail_ref) else {
            return false;
        };
        let l = chol.L();
        for b in 0..kx {
            let mut acc = l[(kx + p, b)];
            for j in 0..p {
                acc -= l[(kx + j, b)] * betas[j];
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
                acc -= col[kx + j] * betas[j];
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
    /// Coding: 0 = interior min, 1 = pinned at a variance
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
    /// Minimized profiled deviance of the kernel's criterion (REML or ML) at
    /// the accepted (or capped) θ̂ (the `reml_deviance` value after the pin
    /// re-eval), on the standardized design. NaN only on
    /// optimizer/numerical failure — finite on a `MaxFunReached` endpoint.
    pub deviance: f64,
    /// Scale-invariant per-column pivot ratio of X'V⁻¹X at θ̂
    /// (`min_pivot_ratio`), with `pivot_col` the column attaining
    /// it. **Detection only** — no branch in this route reads it, and none may
    /// start to; see the measurement recorded at the computation site. Below
    /// `PIVOT_MIN` the coefficients are barely identified and the diagnostics
    /// channel says so. NaN when the deviance re-eval left no trustworthy
    /// factor to measure.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
    /// The gradient search (the per-group route's BFGS) did not converge and
    /// handed its incumbent to BOBYQA — see
    /// [`crate::Note::GradientSearchFallback`].
    pub gradient_search_fallback: bool,
    /// How the gradient search ended, `None` when the fit searched with BOBYQA
    /// alone: the driver's stop, its gradient count and the endpoint decrement
    /// `½·gᵀHg`. Kept when BOBYQA finished the fit after it.
    pub(crate) grad_report: Option<(crate::optim::bfgs::BfgsStop, usize, f64)>,
}

/// Joint Wald-χ² statistic over the target set `T`,
///
/// ```text
/// W = β̂_T' Σ_T⁻¹ β̂_T / σ̂²
/// ```
///
/// with `Σ_T` the targets' block of `Var(β̂)/σ̂²`, read from a square root of
/// it: column `t` of `cols` (for each target `t`; the other columns are not
/// read) holds `c_t` with `Var(β̂_a, β̂_b) = σ̂²·c_aᵀc_b`, so `Σ_T = C_TᵀC_T`.
/// The fit routes hold these columns already, the per-target forward solves
/// `L⁻¹bₜ` their variances are the norms of. Under H₀: β_T = 0,
/// `W ~ χ²(k)` asymptotically.
///
/// `W = ‖R⁻ᵀβ̂_T‖²/σ̂²` with `C_T = Q·R` by modified Gram–Schmidt, which yields
/// the same `R` as Householder QR in floating point (Björck & Paige 1992,
/// *SIAM J. Matrix Anal. Appl.* 13(1)), so `W` carries `cond(C_T)·ε`. Forming
/// `Σ_T` and factoring it, or factoring the information re-formed from its
/// factor and inverting that, would square it: measured on `[1, x, x + δz]`
/// at cond(X) ≈ 2e6, 2e-4 relative on a two-target `W` against 3e-11 here.
///
/// `qr` (`p×p`) and `z` (length `p`) are scratch. `NaN` when `k == 0`,
/// `σ̂² ≤ 0`, a target is out of range, or `C_T` is rank deficient.
pub(crate) fn joint_wald_chi_sq(
    cols: MatRef<'_, f64>,
    betas: &[f64],
    sigma_sq: f64,
    target_indices: &[u32],
    mut qr: MatMut<'_, f64>,
    z: &mut [f64],
) -> f64 {
    let p = betas.len();
    let k = target_indices.len();
    if k == 0 || p == 0 || !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        return f64::NAN;
    }
    let r = cols.nrows();
    for (a, &t) in target_indices.iter().enumerate() {
        let t = t as usize;
        if t >= p {
            return f64::NAN;
        }
        for i in 0..r {
            qr[(i, a)] = cols[(i, t)];
        }
        z[a] = betas[t];
    }
    // Step j takes column j's norm `R_jj` and removes its direction from the
    // later columns (`R_jl`), and forward-substitutes `Rᵀz = β̂_T` along:
    // `z[l]` holds `β̂_l − Σ_{j<l} R_jl·z_j` until step `l` divides it by `R_ll`.
    let mut w = 0.0_f64;
    for j in 0..k {
        let rjj = (0..r).map(|i| qr[(i, j)] * qr[(i, j)]).sum::<f64>().sqrt();
        if !(rjj.is_finite() && rjj > 0.0) {
            return f64::NAN;
        }
        for i in 0..r {
            qr[(i, j)] /= rjj;
        }
        let zj = z[j] / rjj;
        w += zj * zj;
        for l in (j + 1)..k {
            let rjl: f64 = (0..r).map(|i| qr[(i, j)] * qr[(i, l)]).sum();
            for i in 0..r {
                let qij = qr[(i, j)];
                qr[(i, l)] -= rjl * qij;
            }
            z[l] -= rjl * zj;
        }
    }
    if !w.is_finite() {
        return f64::NAN;
    }
    w / sigma_sq
}

/// The profiled deviance under the kernel's criterion (REML or ML) as the
/// optimizer layer's objective
/// ([`crate::optim::Objective`]): every evaluation is counted, and the finite
/// ones too (`fit_lmm`'s `converged` needs two). The profiled criterion has
/// no inner solve, so the optimizer's resolution is not used.
///
/// On the per-group regression route it is also a
/// [`crate::optim::GradObjective`]: the gradient of the same criterion comes
/// off the state the last evaluation left ([`pg_grad::pg_gradient`]).
struct LmmObjective<'a> {
    kernel: &'a mut LmmKernel,
    counters: &'a mut crate::counters::EvalCounters,
    finite_evals: &'a mut usize,
    /// Whether the last evaluation was finite (the gradient needs its state).
    last_finite: bool,
}

impl crate::optim::Objective for LmmObjective<'_> {
    fn value(&mut self, x: &[f64], _resolution: f64) -> Option<f64> {
        let d = self.kernel.deviance(x);
        self.last_finite = d.is_finite();
        if self.last_finite {
            *self.finite_evals += 1;
        }
        self.counters.record_eval(crate::counters::Stage::Two, d);
        Some(d)
    }
}

impl crate::optim::GradObjective for LmmObjective<'_> {
    fn gradient(&mut self, g: &mut [f64]) -> bool {
        match &mut *self.kernel {
            LmmKernel::Dense { suff, fit } if self.last_finite && fit.pg_n_active > 0 => {
                pg_grad::pg_gradient(suff, fit, g)
            }
            _ => false,
        }
    }
}

/// The θ search of [`fit_lmm`], from `theta` (the start, already clamped)
/// within `[lower, upper]`, leaving the endpoint in `theta`. The route's rule
/// ([`lmm_outer_optimizer`]) picks the optimizer. BOBYQA searches from a warm
/// start with `solver_warm` (`RHO_BEGIN_WARM`) and from the blind start with
/// `solver` (`RHO_BEGIN`). BFGS's endpoint is accepted when it passes the
/// certificate; otherwise the fallback ladder runs:
///
/// - a search that stopped on its convergence tests but failed the
///   certificate is polished by BOBYQA from its endpoint, at
///   `RHO_BEGIN_WARM` (part of the route, no `Note`);
/// - any other stop hands the incumbent to BOBYQA with its unchanged budget,
///   and the second value returned is `true`
///   ([`crate::Note::GradientSearchFallback`]);
/// - a start with no finite value runs exactly the search the BOBYQA route
///   would have run.
///
/// The returned `Outcome` is the finishing optimizer's; its `n_eval` counts
/// every value evaluation of both. `hint`, when its dimension matches, starts
/// the BFGS search from its inverse Hessian (it is never written). The
/// gradient evaluations reuse the value
/// evaluations' state and are not counted. The third value is `true` when a
/// certified BFGS endpoint was accepted: its convergence rests on the
/// gradient, so it needs no second finite value (a warm start at the optimum
/// certifies after one value and one gradient). The fourth value is the BFGS
/// search's own stop, gradient count and endpoint decrement `½·gᵀHg`, kept
/// when BOBYQA finished the fit after it; `None` on the BOBYQA route.
#[allow(clippy::too_many_arguments)]
pub(crate) fn search_theta(
    kernel: &mut LmmKernel,
    solver: &mut Bobyqa,
    solver_warm: &mut Option<Bobyqa>,
    config: &Config,
    bfgs: &mut crate::optim::bfgs::BfgsState,
    theta: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    warm: bool,
    hint: Option<&crate::optim::CurvatureHint>,
    counters: &mut crate::counters::EvalCounters,
    finite_evals: &mut usize,
) -> (
    bobyqa::Outcome,
    bool,
    bool,
    Option<(crate::optim::bfgs::BfgsStop, usize, f64)>,
) {
    use crate::optim::bfgs::{bfgs_minimize, BfgsConfig, BfgsStop};
    use crate::optim::{bobyqa_minimize, OuterOptimizer};
    let n = theta.len();
    let route = lmm_outer_optimizer(kernel, n);
    let mut obj = LmmObjective {
        kernel,
        counters,
        finite_evals,
        last_finite: false,
    };
    match route {
        OuterOptimizer::Bobyqa => {
            let s = if warm {
                warm_bobyqa(solver_warm, config, n)
            } else {
                solver
            };
            (
                bobyqa_minimize(s, &mut obj, theta, lower, upper),
                false,
                false,
                None,
            )
        }
        OuterOptimizer::Bfgs => {
            let h0 = hint.filter(|h| h.dim() == n).map(|h| h.inv_hessian());
            let bo = bfgs_minimize(&mut obj, theta, lower, upper, &BfgsConfig::LMM, bfgs, h0);
            let report = Some((bo.stop, bo.n_grad, bo.decrement));
            if bo.certified(theta, obj.kernel.groupings().diagonal_theta()) {
                let out = bobyqa::Outcome {
                    f: bo.f,
                    n_eval: bo.n_val,
                    status: Status::Converged,
                };
                return (out, false, true, report);
            }
            let polish = bo.converged();
            let s = if bo.stop == BfgsStop::NonFinite && !warm {
                solver
            } else {
                warm_bobyqa(solver_warm, config, n)
            };
            let mut out = bobyqa_minimize(s, &mut obj, theta, lower, upper);
            out.n_eval += bo.n_val;
            (out, !polish, false, report)
        }
    }
}

/// The curvature hint at the θ̂ the last [`fit_lmm`] on `ws` left in
/// `ws.theta`, for the per-group route's gradient search; `None` off that
/// route (no gradient) or when the finite-difference Hessian there is not
/// positive definite. It re-evaluates the deviance `n_θ + 1` times, so the
/// kernel's state no longer describes θ̂ afterwards. The hint is the
/// curvature of the kernel's criterion. A hint taken on a fit of the other
/// criterion is accepted by [`search_theta`] and only seeds the BFGS inverse
/// Hessian: the search and its gradient are the kernel's criterion's; the
/// seeded inverse Hessian enters only as a start (and the first decrement).
pub(crate) fn lmm_curvature_hint(ws: &mut LmmWorkspace) -> Option<crate::optim::CurvatureHint> {
    use crate::optim::{GradObjective, Objective};
    let LmmWorkspace {
        kernel,
        theta,
        lower,
        upper,
        ..
    } = ws;
    if lmm_outer_optimizer(kernel, theta.len()) != crate::optim::OuterOptimizer::Bfgs {
        return None;
    }
    let mut counters = crate::counters::EvalCounters::new();
    let mut finite = 0usize;
    let mut obj = LmmObjective {
        kernel,
        counters: &mut counters,
        finite_evals: &mut finite,
        last_finite: false,
    };
    crate::optim::CurvatureHint::from_gradients(theta, lower, upper, |x, g| {
        obj.value(x, 0.0);
        obj.gradient(g)
    })
}

/// The `RHO_BEGIN_WARM` solver, built on first use.
fn warm_bobyqa<'a>(slot: &'a mut Option<Bobyqa>, config: &Config, n: usize) -> &'a mut Bobyqa {
    slot.get_or_insert_with(|| {
        let mut c = *config;
        c.rho_begin = RHO_BEGIN_WARM;
        Bobyqa::new(n, c).expect("warm BOBYQA config is valid by construction")
    })
}

/// Whether the per-group θ-gradient ([`pg_grad::pg_gradient`]) is armed on
/// `kernel` for the accumulated dataset: the dense kernel with the per-group
/// regression route on. The condition [`LmmObjective::gradient`] and
/// [`lmm_outer_optimizer`] test (the route admits `q_p ≥ 2` only, so its
/// `n_θ ≥ 3` holds wherever this does); the Satterthwaite pass reads it to
/// choose its derivative route.
pub(crate) fn per_group_gradient_armed(kernel: &LmmKernel) -> bool {
    matches!(kernel, LmmKernel::Dense { fit, .. } if fit.pg_n_active > 0)
}

/// The outer optimizer a fit's θ search runs: BFGS where the per-group
/// regression route is armed for this dataset, which admits random slopes on
/// the primary grouping only (`q_p ≥ 2`, so `n_θ ≥ 3`), and BOBYQA everywhere
/// else. A function of the shape and the data's level
/// Grams, not of the start or the criterion: cold and warm fits, REML and
/// ML, route alike, and on BFGS the search minimizes the kernel's criterion
/// with that criterion's gradient.
///
/// Measured on the route's cells (2026-10-07, pawel, locked P-core, search
/// wall against BOBYQA from the same start): 3 θ cold geo 0.42 (none slower),
/// warm 0.48; 36 θ 0.04 cold, 0.05 warm.
fn lmm_outer_optimizer(kernel: &LmmKernel, n_theta: usize) -> crate::optim::OuterOptimizer {
    match kernel {
        LmmKernel::Dense { fit, .. } if fit.pg_n_active > 0 && n_theta >= 3 => {
            crate::optim::OuterOptimizer::Bfgs
        }
        _ => crate::optim::OuterOptimizer::Bobyqa,
    }
}

/// Fit by minimising the profiled deviance of the kernel's criterion (REML or
/// ML) over the box-bounded θ, by BOBYQA or, on the per-group regression route,
/// BFGS, with β̂ / σ̂² / Var(β̂_target) recovered once at θ̂.
///
/// Caller contract: `ws.suff` holds the accumulated rows (reset + add_rows
/// per dataset); `target_indices` index design columns.
///
/// `theta_start`: `None` → blind start (diagonals THETA0, off-diagonals 0,
/// the default for arbitrary provided bytes); `Some(θ₀)` → per-component spec-derived truth
/// start, `[primary, extras in declaration order]` (Y is always synthetic, so
/// true θ_g = τ_g/σ is known), with diagonal components clamped to
/// THETA_TRUTH_FLOOR and off-diagonals passed through verbatim. A
/// per-scenario constant — determinism and chunk merging are unaffected. The
/// DGP-derived hint is a deliberate, recorded exception to the
/// generation↔estimation split.
///
/// `map` is the column standardization the rows were accumulated under:
/// β̂, `var_diag`, `t_sq` and the joint Wald are reported through it
/// (`ColumnScaling::IDENTITY` for a design fitted as given).
///
/// `hint`: a caller's [`crate::optim::CurvatureHint`] for the gradient search
/// (`loop_advanced::fit_on_hinted`); read only, and ignored on the BOBYQA
/// route and when its dimension does not match.
pub fn fit_lmm(
    ws: &mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    hint: Option<&crate::optim::CurvatureHint>,
    map: &ColumnScaling,
) -> LmmFit {
    let LmmWorkspace {
        kernel,
        recovery,
        solver,
        solver_warm,
        config,
        bfgs,
        theta,
        lower,
        upper,
        ..
    } = ws;
    // p is the width the recovery buffers were allocated at.
    let p = recovery.betas.len();

    // Arm the balanced collapse for this dataset's counts (cheap —
    // O(n_primary·w²·t_dim²) once per fit; sets collapse_n_active = 0 on any
    // unbalanced/slope shape, which keeps the per-family loop), and the
    // per-group regression route for a random-slopes primary (same cost
    // order; off on any other shape or an ill-conditioned group). The sparse
    // kernel has neither.
    if let LmmKernel::Dense { suff, fit } = &mut *kernel {
        precompute_balanced_collapse(suff, fit);
        precompute_group_regressions(suff, fit);
    }

    // A Some-start is clamped to the floor on its diagonal coordinates only
    // (off-diagonals pass through verbatim) and searched by `solver_warm`, at
    // `RHO_BEGIN_WARM`; the blind start by `solver`, at `RHO_BEGIN`
    // (`search_theta`).
    match theta_start {
        Some(ts) => {
            debug_assert_eq!(ts.len(), theta.len());
            // `ts` is already on the fitted design (`fit::common::start_to_design`),
            // so THETA_TRUTH_FLOOR floors the same θ `PIN_THETA` tests and the two
            // absolute thresholds stay on one axis.
            for (t, &v) in theta.iter_mut().zip(ts) {
                *t = v;
            }
            for &i in kernel.groupings().diagonal_theta() {
                theta[i] = theta[i].max(THETA_TRUTH_FLOOR);
            }
        }
        None => {
            // Blind start: diagonals THETA0, off-diagonals 0 (the
            // `blind_theta_and_bounds` shape, zero-alloc). The former
            // all-THETA0 start put off-diagonal vech entries at 1.0 — off the
            // lme4/MixedModels unit-diagonal convention — and on the wide-slope
            // grid stratum that start funnels BOBYQA into a second-best optimum
            // in 8/9 cells (regression goldens at
            // validation/campaigns/speed-campaign/optima/ pin
            // the correct optimum). Mirrors the GLMM joint seed
            // (`glmm::fit_glmm`'s θ cold start), which carries the same rule.
            for t in theta.iter_mut() {
                *t = 0.0;
            }
            for &i in kernel.groupings().diagonal_theta() {
                theta[i] = THETA0;
            }
        }
    }
    let mut counters = crate::counters::EvalCounters::new();
    // mirrors the stage-1 read in glmm/mod.rs — change together.
    let mut finite_evals = 0usize;
    let (out, gradient_search_fallback, certified, grad_report) = search_theta(
        kernel,
        solver,
        solver_warm,
        config,
        bfgs,
        theta,
        lower,
        upper,
        theta_start.is_some(),
        hint,
        &mut counters,
        &mut finite_evals,
    );

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
    let converged = matches!(out.status, Status::Converged) && (finite_evals >= 2 || certified);
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
    //
    // Sign canonicalization first (`fix_column_signs`): the box lets a
    // diagonal end negative, so a block column whose diagonal did is negated
    // whole. Σ and the deviance are unchanged, and the pin test then sees
    // non-negative diagonals. Mirror the GLMM pin site — change together.
    if has_endpoint {
        fix_column_signs(kernel.groupings(), theta);
    }
    let diag = kernel.groupings().diagonal_theta();
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
        // Σ-preserving canonical Λ, then the pin test again on the new
        // diagonals: the fold can retire a component that was pinned only as a
        // degenerate coordinate, and can expose a trailing diagonal that pins.
        // The flags are rebuilt from scratch rather than OR'd. Nothing moves
        // unless a diagonal pinned, so an interior fit stays bit-identical, and
        // the pinned re-eval below runs on the canonicalized θ. The GLMM pin
        // site departs from this on its AGQ route, and undoes a pin that
        // raises the deviance; the reasons are written there.
        if canonicalize_pinned_blocks(kernel.groupings(), theta) {
            pinned = false;
            pinned_components = 0;
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
    }

    // Pin eval at θ̂ — refreshes factor/σ̂² at the accepted-or-capped point
    // (the shipped path's "pin Cholesky at θ̂" step). Armed first: this is the
    // evaluation the conditional-mode recovery rides on, and arming keeps the
    // per-family factors it needs for this call alone.
    let dev = if has_endpoint {
        kernel.prepare_recovery();
        kernel.deviance(theta)
    } else {
        f64::INFINITY
    };

    // The kernel factored `[X̃ y]`, X̃ = X·U⁻¹ (`DesignQr`). β̃ comes off that
    // factor first, by the backward solve L̃_XXᵀ β̃ = l_yX̃ (the y row of the
    // augmented factor is index p): the conditional-mode recovery works on X̃.
    // Then the factor maps to `[X y]`, which is what the pivot, Var(β̂) and the
    // joint Wald below read.
    if dev.is_finite() {
        let factor = kernel.factor();
        for j in (0..p).rev() {
            let mut acc = factor[(p, j)];
            for k in (j + 1)..p {
                acc -= factor[(k, j)] * recovery.beta_tilde[k];
            }
            recovery.beta_tilde[j] = acc / factor[(j, j)];
        }
        kernel.map_factor_to_design();
    }

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
        crate::ols::min_pivot_ratio(kernel.factor(), p)
    } else {
        (f64::NAN, 0)
    };
    if !has_endpoint || !dev.is_finite() {
        kernel.invalidate_recovery();
        for v in recovery.betas.iter_mut() {
            *v = f64::NAN;
        }
        for &t in target_indices {
            recovery.var_diag[t as usize] = f64::NAN;
            recovery.t_sq[t as usize] = f64::NAN;
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
            gradient_search_fallback,
            grad_report,
        };
    }

    // β̂ = U⁻¹β̃ — two triangular solves (L̃_XXᵀ above, U here) rather than
    // one against the mapped factor Uᵀ·L̃_XX, which would round the product.
    recovery.betas.copy_from_slice(&recovery.beta_tilde);
    kernel.design_qr().map_beta(&mut recovery.betas);
    // β̂ on the caller's columns: β = B·β̂(standardized), `map` the column
    // standardization the rows were accumulated under. `beta_tilde` stays on
    // the kernel's design for the conditional-mode recovery below.
    map.beta_to_caller(&mut recovery.betas);
    let factor = kernel.factor();

    // Var(β̂ₜ) = σ̂²·‖L_XX⁻¹bₜ‖² per target, bₜ row t of the column map B (eₜ
    // when the design is fitted as given); t² = β̂²/Var, via a forward solve on
    // this factor.
    let sigma_sq = kernel.sigma_sq();
    for &tj in target_indices {
        let tj = tj as usize;
        for v in recovery.u[..p].iter_mut() {
            *v = 0.0;
        }
        for i in 0..p {
            let b_i = map.rhs(tj, i);
            let mut acc = b_i;
            for k in 0..i {
                acc -= factor[(i, k)] * recovery.u[k];
            }
            recovery.u[i] = acc / factor[(i, i)];
        }
        let norm_sq: f64 = recovery.u[..p].iter().map(|v| v * v).sum();
        for i in 0..p {
            recovery.joint_cols[(i, tj)] = recovery.u[i];
        }
        let vd = sigma_sq * norm_sq;
        recovery.var_diag[tj] = vd;
        recovery.t_sq[tj] = if vd.is_finite() && vd > 0.0 {
            (recovery.betas[tj] * recovery.betas[tj]) / vd
        } else {
            f64::NAN
        };
    }

    // Joint Wald-χ² over the target set, from the forward-solve columns kept
    // above (`Var(β̂_a, β̂_b) = σ̂²·c_aᵀc_b`).
    let joint_t_sq = joint_wald_chi_sq(
        recovery.joint_cols.as_ref(),
        &recovery.betas,
        sigma_sq,
        target_indices,
        recovery.joint_qr.as_mut(),
        &mut recovery.joint_z,
    );

    // Conditional modes, last: it reads β̂ and reuses the evaluation's own
    // scratch, so it must come after every step that reads the factor state the
    // θ̂ evaluation left.
    kernel.recover_ranef(theta, &recovery.beta_tilde);

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
        gradient_search_fallback,
        grad_report,
    }
}

/// [`fit_lmm`] without a curvature hint, for the tests.
#[cfg(test)]
pub(crate) fn fit_lmm_unhinted(
    ws: &mut LmmWorkspace,
    target_indices: &[u32],
    theta_start: Option<&[f64]>,
    map: &ColumnScaling,
) -> LmmFit {
    fit_lmm(ws, target_indices, theta_start, None, map)
}
