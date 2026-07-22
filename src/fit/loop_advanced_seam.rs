//! Dev-only "loop_advanced" seam — UNSTABLE, NOT semver-covered, re-exported
//! only through `crate::loop_advanced` (gated by the `loop_advanced` Cargo
//! feature, off by default). Two independent seams share this file because
//! both are Gaussian-LMM-only dev surfaces layered on the same dispatch
//! helpers (`common::spec_sized_from_ids`/`assert_group_ids`,
//! `lmm::{accumulate_lmm_rows, fit_lmm_into}`):
//!
//! - the **adjudication seam**: lets a caller check its own θ-search or refit
//!   logic against the exact profiled-REML closure `fit` minimizes, by
//!   exposing that closure evaluated at a caller-fixed θ or minimized under a
//!   caller-configured BOBYQA schedule;
//! - **caller-owned LMM workspace reuse**: lets a caller (MCPower's hot loop)
//!   allocate the per-shape LMM workspace once and refit many same-shape
//!   datasets against it.
//!
//! The shipped `fit_cold`/`fit_warm` path is untouched by either.

#[cfg(feature = "loop_advanced")]
use crate::lmm::{LmmFitScratch, LmmSuffStats, LmmWorkspace};
#[cfg(feature = "loop_advanced")]
use crate::{Family, GroupIds, ModelSpec};
// Only the test-gated `build_lmm_workspace`/`refit_lmm` pair below still uses these.
#[cfg(all(test, feature = "loop_advanced"))]
use crate::StartValues;

#[cfg(feature = "loop_advanced")]
use super::common::{assert_group_ids, spec_sized_from_ids};
#[cfg(feature = "loop_advanced")]
use super::lmm::accumulate_lmm_rows;
#[cfg(all(test, feature = "loop_advanced"))]
use super::lmm::fit_lmm_into;
#[cfg(feature = "loop_advanced")]
use super::{classify_design, Solver};
#[cfg(all(test, feature = "loop_advanced"))]
use super::{Fit, FitOptions};

// ---------------------------------------------------------------------------
// Dev-only adjudication seam (loop_advanced) — lets a caller adjudicate its
// own θ-search against the reference objective.
// NOT semver-covered. Gaussian LMM only: exposes the exact profiled-REML
// closure `fit` minimizes, (a) evaluated at a caller-fixed θ and (b) minimized
// under a caller-configured schedule. The shipped path is untouched.
// ---------------------------------------------------------------------------

/// The θ ↦ profiled-REML-deviance closure the dev seam hands out.
#[cfg(feature = "loop_advanced")]
type LmmObjective<'a> = dyn FnMut(&[f64]) -> f64 + 'a;

/// Per-eval trace hook for [`lmm_sweep_fit`]: `(k, θ, f)` per objective call.
#[cfg(feature = "loop_advanced")]
pub type LmmTrace<'a> = dyn FnMut(usize, &[f64], f64) + 'a;

/// θ-independent, design-bound state for the LMM sweep seam. [`build_lmm_seam_ws`]
/// builds this ONCE from x/y/ids; [`lmm_sweep_fit_on`] re-solves it at any number
/// of θ₀ without re-accumulating the design (the redundant rebuild a two-stage
/// warm-restart run previously paid per stage). Deliberately holds no raw
/// x/y/ids — the type shape is the reuse guard: a caller has no field through
/// which to smuggle different data into a second sweep on the same `LmmSeamWs`.
// `SparseLmmWorkspace` is `pub(crate)` (sparse.rs, untouched here) — an
// internal implementation detail, not a type dev-seam callers construct or
// name field-by-field. `GlmmWorkspace::cluster_rows`/`structured_schur` in
// glmm/workspace.rs wrap `pub(crate)` internals the identical way (also
// reachable through loop_advanced) and are left as bare `private_interfaces`
// warnings there; silenced here instead of promoting the type to `pub` for
// one enum variant.
#[cfg(feature = "loop_advanced")]
#[allow(private_interfaces)]
pub enum LmmSeamWs {
    /// Dense (`Solver::NoZ`) route: accumulated suff-stats plus the Cholesky
    /// scratch buffers `reml_deviance` factors into, already armed for the
    /// balanced-collapse fast path if the design qualifies.
    Dense {
        /// Accumulated per-cluster sufficient statistics for the design.
        suff: Box<LmmSuffStats>,
        /// Cholesky/collapse scratch buffers `reml_deviance` factors into.
        fit: Box<LmmFitScratch>,
    },
    /// Sparse (`Solver::Sparse`) route: one symbolic-factor workspace.
    Sparse {
        /// Symbolic-factor workspace for the sparse REML objective.
        ws: Box<crate::sparse::SparseLmmWorkspace>,
    },
}

/// Single O(N) build pass for the sweep seam: marshals the LMM inputs exactly
/// as `fit_mle` does — sized spec, slope columns, workspace, suff-stats/
/// symbolic-factor accumulation, dense balanced-collapse arming — and returns
/// the θ-independent workspace plus its groupings. Gaussian LMM only (see the
/// assert). `precompute_balanced_collapse` runs once here, not per sweep: it
/// is a pure function of `suff` (full overwrite of `fit`'s collapse buffers,
/// not an incremental accumulate — see its body), so it stays valid across
/// any number of later [`lmm_sweep_fit_on`] calls on the same `suff`.
#[cfg(feature = "loop_advanced")]
pub fn build_lmm_seam_ws(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
) -> (LmmSeamWs, crate::lmm::LmmGroupings) {
    assert!(
        matches!(model.family, Family::Gaussian) && model.re.is_some(),
        "dev objective seam covers Gaussian LMM only"
    );
    assert_group_ids(model.re.as_ref().unwrap(), ids, n);
    let sized = spec_sized_from_ids(model, ids);
    let re = sized.re.as_ref().unwrap();
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();
    match classify_design(&sized, 1) {
        Solver::NoZ => {
            let mut ws =
                LmmWorkspace::for_cluster_spec_ext(p, &sized, n, &slope_cols, &extra_slope_cols);
            accumulate_lmm_rows(&mut ws, x, y, n, p, &ids.primary, &ids.extra, None);
            let LmmWorkspace { suff, mut fit, .. } = ws;
            crate::lmm::precompute_balanced_collapse(&suff, &mut fit);
            let g = suff.groupings.clone();
            (
                LmmSeamWs::Dense {
                    suff: Box::new(suff),
                    fit: Box::new(fit),
                },
                g,
            )
        }
        Solver::Sparse => {
            let g = crate::lmm::LmmGroupings::from_cluster_spec_ext(
                &sized,
                n,
                &slope_cols,
                &extra_slope_cols,
            );
            let xm = faer::MatRef::from_row_major_slice(x, n, p);
            let ws = crate::sparse::SparseLmmWorkspace::new(
                &g,
                xm,
                &ids.primary,
                &ids.extra,
                y,
                n,
                p,
                None,
            );
            (LmmSeamWs::Sparse { ws: Box::new(ws) }, g)
        }
    }
}

/// Marshal the LMM inputs exactly as `fit_mle` does and hand the
/// ready-to-evaluate workspace to `f`. Thin adapter over [`build_lmm_seam_ws`]:
/// reconstructs the same `obj` closure `f` used to see directly, so
/// [`lmm_objective_at`] keeps its one-shot build-then-evaluate behavior
/// unchanged.
#[cfg(feature = "loop_advanced")]
fn with_lmm_objective<R>(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    f: impl FnOnce(&mut LmmObjective<'_>, &crate::lmm::LmmGroupings) -> R,
) -> R {
    let (mut ws, g) = build_lmm_seam_ws(x, y, n, p, model, ids);
    match &mut ws {
        LmmSeamWs::Dense { suff, fit } => {
            let mut obj = |theta: &[f64]| crate::lmm::reml_deviance(theta, suff, fit);
            f(&mut obj, &g)
        }
        LmmSeamWs::Sparse { ws } => {
            let mut obj = |theta: &[f64]| crate::sparse::sparse_reml_deviance(theta, ws);
            f(&mut obj, &g)
        }
    }
}

/// Profiled REML deviance of the LMM objective at a fixed θ (glmm's own vech
/// layout: primary column-major lower triangle, then extras in declaration
/// order). Raw optimizer scale — the same value `Fit::deviance` reports for an
/// unweighted fit.
#[cfg(feature = "loop_advanced")]
pub fn lmm_objective_at(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    theta: &[f64],
) -> f64 {
    with_lmm_objective(x, y, n, p, model, ids, |obj, g| {
        assert_eq!(
            theta.len(),
            g.n_theta(),
            "theta length must match the model"
        );
        obj(theta)
    })
}

/// Outcome of [`lmm_sweep_fit`]: the accepted point and objective, plus the
/// eval count and raw convergence bit (no pinning, no β recovery).
#[cfg(feature = "loop_advanced")]
pub struct LmmSweepOutcome {
    /// Profiled REML deviance at `theta`.
    pub deviance: f64,
    /// θ at the accepted point, in the same vech layout as [`lmm_objective_at`].
    pub theta: Vec<f64>,
    /// Number of objective evaluations the solver used.
    pub n_eval: usize,
    /// Whether the solver reported convergence (vs. hitting `max_fun`/other stop).
    pub converged: bool,
}

/// The θ-search body shared by [`lmm_sweep_fit`] (via [`lmm_sweep_fit_on`]) and
/// `with_lmm_objective`'s one-shot sibling: minimizes `obj` under a caller-
/// configured BOBYQA schedule. `theta0` is used VERBATIM (`None` → the shipped
/// blind start) — unlike `fit`'s warm start, which clamps every component to
/// `THETA_TRUTH_FLOOR` and so cannot express a negative off-diagonal start.
/// npt and rho_begin are derived exactly as the shipped sites derive them (mid
/// npt ⌈1.5n⌉+1 from n ≥ 3, rho_begin = min(0.1·min diag θ₀, RHO_BEGIN) floored
/// at 10·rho_end), so `(theta0 = None, rho_end = RHO_END, max_fun = None)`
/// replays a shipped grid fit trajectory-identically; `trace` then observes
/// every (k, θ, f) evaluation without any hook in the shipped path. A fresh
/// `Bobyqa` is allocated per call — correct even when `obj` closes over a
/// [`LmmSeamWs`] reused across calls, since npt/rho_begin (and thus the
/// solver's interpolation set) legitimately differ per θ₀/schedule; only the
/// design-bound `suff`/`fit`/sparse `ws` behind `obj` are shared.
#[cfg(feature = "loop_advanced")]
#[allow(clippy::too_many_arguments)] // dev seam, marshals the fit_mle surface + schedule
fn lmm_sweep_search(
    obj: &mut LmmObjective<'_>,
    g: &crate::lmm::LmmGroupings,
    theta0: Option<&[f64]>,
    rho_end: f64,
    max_fun: Option<usize>,
    mut trace: Option<&mut LmmTrace<'_>>,
) -> LmmSweepOutcome {
    use bobyqa::{Bobyqa, Config, Status};
    let n_theta = g.n_theta();
    let (blind, lower, upper) = g.blind_theta_and_bounds();
    let mut theta = match theta0 {
        Some(t) => {
            assert_eq!(t.len(), n_theta, "theta0 length must match the model");
            t.to_vec()
        }
        // Mirror `fit_lmm`'s cold arm exactly (replay fidelity depends on
        // this): the blind seed — diagonals THETA0, off-diagonals 0. An
        // all-THETA0 seed (diagonals AND off-diagonals) mis-scales Λ on wide
        // vech blocks and BOBYQA stalls in that basin instead of reaching the
        // optimum, so the shipped LMM paths seed off-diagonals at 0 instead.
        None => blind,
    };
    let min_diag = g
        .diagonal_theta()
        .iter()
        .map(|&i| theta[i])
        .fold(f64::INFINITY, f64::min);
    let rho_begin = (0.1 * min_diag)
        .min(crate::lmm::RHO_BEGIN)
        .max(10.0 * rho_end);
    let npt = if n_theta >= 3 {
        (3 * n_theta).div_ceil(2) + 1
    } else {
        2 * n_theta + 1
    };
    let mut config = Config {
        rho_begin,
        rho_end,
        npt,
        ..Config::new(n_theta)
    };
    crate::lmm::apply_campaign_overrides(&mut config, n_theta);
    if let Some(mf) = max_fun {
        config.max_fun = mf;
    }
    let mut solver = Bobyqa::new(n_theta, config).expect("dev sweep config valid");
    let mut k = 0usize;
    let out = solver.minimize(
        |xs| {
            let v = obj(xs);
            k += 1;
            if let Some(t) = trace.as_mut() {
                t(k, xs, v);
            }
            v
        },
        &mut theta,
        &lower,
        &upper,
    );
    LmmSweepOutcome {
        deviance: obj(&theta),
        theta,
        n_eval: out.n_eval,
        converged: matches!(out.status, Status::Converged),
    }
}

/// Minimize the LMM objective held by `ws` (built once by
/// [`build_lmm_seam_ws`]) under a caller-configured BOBYQA schedule — the
/// warm-restart seam: call this any number of times on the same `ws` at
/// different θ₀/schedules without re-accumulating the design. See
/// [`lmm_sweep_search`] for the schedule/replay contract; `ws` and `g` are
/// exactly the pair `build_lmm_seam_ws` returns.
#[cfg(feature = "loop_advanced")]
#[allow(clippy::too_many_arguments)] // dev seam, marshals the fit_mle surface + schedule
pub fn lmm_sweep_fit_on(
    ws: &mut LmmSeamWs,
    g: &crate::lmm::LmmGroupings,
    theta0: Option<&[f64]>,
    rho_end: f64,
    max_fun: Option<usize>,
    trace: Option<&mut LmmTrace<'_>>,
) -> LmmSweepOutcome {
    match ws {
        LmmSeamWs::Dense { suff, fit } => {
            let mut obj = |theta: &[f64]| crate::lmm::reml_deviance(theta, suff, fit);
            lmm_sweep_search(&mut obj, g, theta0, rho_end, max_fun, trace)
        }
        LmmSeamWs::Sparse { ws } => {
            let mut obj = |theta: &[f64]| crate::sparse::sparse_reml_deviance(theta, ws);
            lmm_sweep_search(&mut obj, g, theta0, rho_end, max_fun, trace)
        }
    }
}

/// One-shot build-then-minimize: [`build_lmm_seam_ws`] followed by a single
/// [`lmm_sweep_fit_on`] call. Use [`build_lmm_seam_ws`] directly when sweeping
/// the same design at multiple θ₀ — this rebuilds the workspace every call.
#[cfg(feature = "loop_advanced")]
#[allow(clippy::too_many_arguments)] // dev seam, marshals the fit_mle surface + schedule
pub fn lmm_sweep_fit(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    model: &ModelSpec,
    ids: &GroupIds,
    theta0: Option<&[f64]>,
    rho_end: f64,
    max_fun: Option<usize>,
    trace: Option<&mut LmmTrace<'_>>,
) -> LmmSweepOutcome {
    let (mut ws, g) = build_lmm_seam_ws(x, y, n, p, model, ids);
    lmm_sweep_fit_on(&mut ws, &g, theta0, rho_end, max_fun, trace)
}

// ---------------------------------------------------------------------------
// Dev-only caller-owned LMM workspace reuse (loop_advanced) — MCPower pays the
// per-shape allocation once across ~1000 same-shape fits (power simulation:
// X/ids fixed or varying, y re-simulated every draw).
// ---------------------------------------------------------------------------

/// Caller-owned "build once" entry for the [`refit_lmm`] reuse path: allocates
/// the per-shape LMM workspace (suff-stats accumulator, fit scratch, BOBYQA
/// solver state) exactly as `fit_mle` does internally, but hands it back to
/// the caller instead of consuming it inline. Pair with [`refit_lmm`]: call
/// this once per model SHAPE, then `refit_lmm` once per dataset of that shape.
///
/// `model` is used AS GIVEN — unlike [`fit_cold`]/[`fit_warm`], this does
/// **not** derive level counts from a [`GroupIds`] (no `spec_sized_from_ids`
/// step). Pass a spec whose RE level counts already match the real data (a
/// shape-reuse caller already knows these — that is the shape being reused);
/// an under-sized spec (e.g. a placeholder `n_clusters`) sizes the workspace
/// too small and [`refit_lmm`]'s accumulation silently indexes out of the
/// allocated range in a release build (the bounds check is `debug_assert`-only,
/// mirroring `add_rows_multi`'s own guard).
///
/// # Panics
///
/// If `model.re` is `None` (fixed-only design) — mirrors `fit_mle`'s own
/// mixed-model requirement.
// Not on the public loop surface — `build_workspace`/`fit_on` is; kept only for
// the `refit_lmm_matches_fresh_fit_cold` equivalence test.
#[cfg(all(test, feature = "loop_advanced"))]
pub fn build_lmm_workspace(p: usize, model: &ModelSpec, n: usize) -> LmmWorkspace {
    let re = model
        .re
        .as_ref()
        .expect("build_lmm_workspace requires a mixed model (re: Some)");
    // slope_cols/extra_slope_cols derivation mirrors fit_mle's (fit.rs) verbatim.
    let slope_cols: Vec<usize> = re.slopes.iter().map(|&c| c as usize).collect();
    let extra_slope_cols: Vec<Vec<usize>> = re
        .extra_groupings
        .iter()
        .map(|g| g.slopes.iter().map(|&c| c as usize).collect())
        .collect();
    LmmWorkspace::for_cluster_spec_ext(p, model, n, &slope_cols, &extra_slope_cols)
}

/// "Different `y`, same shape" per-call refit on a caller-owned `ws` (built
/// once by [`build_lmm_workspace`]): re-accumulates row-level sufficient
/// statistics and re-solves θ without repaying the per-shape workspace
/// allocation. Reusable across any number of calls of the SAME shape as `ws`
/// was built for (same `p`, `model`, `n`, cluster/grouping structure) — `x`/
/// `ids` may vary or stay fixed between calls, since accumulation re-runs on
/// every call regardless (a new `y` must be read either way; that O(N) cost is
/// irreducible). Allocation-free after the first `build_lmm_workspace` call:
/// the per-call `x_mat` row-major→column-major convert and the returned
/// `Fit`'s O(p) result `Vec`s still allocate (as at every fit entry point in
/// this module) — only the workspace's own buffers (suff-stats, fit scratch,
/// solver state) are reused.
// Not on the public loop surface — `build_workspace`/`fit_on` is; kept only for
// the equivalence test.
#[cfg(all(test, feature = "loop_advanced"))]
#[allow(clippy::too_many_arguments)] // marshals the kernel's (ws, x, y, n, p, ids, opts, start) surface
pub fn refit_lmm(
    ws: &mut LmmWorkspace,
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    ids: &GroupIds,
    opts: &FitOptions,
    start: Option<&StartValues>,
) -> Fit {
    // Identity-link offset as the exact y-shift before accumulation — mirrors
    // fit_mle; change together.
    let y_shifted: Vec<f64>;
    let y_eff: &[f64] = match &opts.offset {
        Some(o) => {
            y_shifted = y.iter().zip(o).map(|(&yi, &oi)| yi - oi).collect();
            &y_shifted
        }
        None => y,
    };
    accumulate_lmm_rows(
        ws,
        x,
        y_eff,
        n,
        p,
        &ids.primary,
        &ids.extra,
        opts.weights.as_deref(),
    );
    fit_lmm_into(ws, n, p, opts, start)
}
