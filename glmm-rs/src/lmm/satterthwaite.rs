//! Satterthwaite degrees of freedom for the fixed effects of a Gaussian LMM:
//! one derivative pass at θ̂, with no refit.
//!
//! lmerTest's df of a contrast `l` is `2v² / (gᵀAg)` over the variance
//! parameters `(θ, σ)`, with `v = lᵀCl`, `C = Var(β̂)`, `g_k = lᵀ(∂C/∂φ_k)l`
//! and `A = 2H⁻¹`, `H` the Hessian of the deviance (REML or ML, the criterion
//! the fit minimized) with σ not profiled (Satterthwaite, 1946; Kuznetsova,
//! Brockhoff and Christensen, 2017). The crate minimizes the deviance `d_p(θ)`
//! with the residual variance `τ = σ²` profiled out, and the two are related
//! exactly: eliminating `τ` from `H` by its Schur complement gives the
//! profiled Hessian `H_p = ∇²d_p`, and `τ`'s own term is `v²/ν` in closed form
//! (`g_τ = v/τ̂`, `H_ττ = ν/τ̂²` at τ̂). So
//!
//! `df = v² / (g_pᵀ H_p⁺ g_p + v²/ν)`, `g_p = ∇_θ (lᵀ C_p(θ) l)`,
//!
//! with `C_p(θ) = σ̂²(θ)·(X'V⁻¹X)⁻¹` the profiled covariance the fit reports
//! and `ν` the divisor of `σ̂²`: `N − p` under REML and `N` under ML, read from
//! the kernel's criterion ([`LmmKernel::criterion`]), never from the fit's
//! options. The pass differentiates whatever criterion the kernel carries. The
//! residual-only record of OLS and the dispersion GLMs
//! ([`SatterthwaiteState::residual`]) keeps `ν = n − p` on either criterion.
//! `0 < df ≤ ν` whenever `H_p⁺` is PSD, and a fit with no free variance
//! parameter gets `ν`. The equivalent lmerTest form, which the state stores,
//! has coordinates `(θ_free, τ)`, `A_eff = 2·blockdiag(H_p⁺, τ̂²/ν)` and
//! `Jac_eff = [∂C_p/∂θ_k …, C/τ̂]`.
//!
//! **The exact pass.** On either kernel, with a crossed tail of at most
//! `exact::EXACT_TAIL_MAX` columns, `H_p` and `∂C_p/∂θ_k` are the closed
//! forms of `exact` (`satterthwaite_exact.rs`), from one deviance evaluation
//! at θ̃ and one walk over the families and the crossed tail, with no step:
//! they depend on no step factor and scale exactly with a rescale.
//!
//! **The value route.** On a wider crossed tail, and as the in-crate
//! reference the exact pass is checked against, `H_p` comes from central
//! finite differences of the
//! same `f64` profiled deviance the θ search minimized
//! ([`LmmKernel::deviance`]), and `∂C_p/∂θ_k` from central differences of `C_p`
//! at the stencil's axis points, on either kernel. The stencil is the axis
//! pairs `θ̃ ± h_k e_k` plus, per pair of coordinates, the two points
//! `θ̃ ± (h_i e_i + h_j e_j)` (numDeriv's `genD` pairing without its Richardson
//! ladder): `n² + n + 1` evaluations for `n` free coordinates. The pass runs on
//! the kernel's θ̃ (the relative Cholesky factor on the standardized design);
//! `gᵀH⁻¹g` is invariant to that smooth reparameterization at a stationary θ̂.
//! Each free coordinate's step is the step factor times the largest `|θ̃|` of
//! its covariance block, with no absolute floor, so a uniform rescale of θ̃
//! (the kernel's rescale retry) scales every step with it and leaves the df
//! unchanged.
//!
//! **Free coordinates.** The deviance and `C_p` depend on Λ only through
//! `ΛΛᵀ`, so both are even in each Λ column. At a column that is exactly zero
//! in θ̂ (a pinned boundary, canonicalized by `fit_lmm`) the gradient and every
//! mixed second derivative with that column vanish, so dropping its
//! coordinates leaves `gᵀH⁺g` unchanged; its own curvature block can be
//! singular, which is the second reason to drop it. The rest of `H_p` is scaled
//! to a unit diagonal, `D^−½·H_p·D^−½` with `D = |diag(H_p)|` (1 where a
//! diagonal entry is zero or not finite), and goes through
//! [`psd_pseudo_inverse`], which drops directions at or below `1e-8·λ_max` and
//! counts the negative ones; the result is mapped back by `D^−½`. lmerTest cuts
//! its unscaled Hessian instead; the two differ only where a coordinate's
//! scale alone puts an eigenvalue below the threshold (measured: the ports'
//! fixture `_PARITY_*` in `glmm-py/tests/test_warnings.py`, and no Gaussian
//! mixed cell of the fast grid, whose lmerTest df comparison is unchanged).
//!
//! The pass reads θ̂ from [`LmmWorkspace::theta`] and never writes it. It
//! overwrites the factor and `σ̂²` the kernel holds, so it runs after the fit
//! has read them; it never maps the factor in place
//! ([`LmmKernel::map_factor_to_design`]), forming `C` from the unmapped factor
//! instead.

use faer::{Mat, MatRef};

use super::{
    forward_solve, precompute_balanced_collapse, precompute_group_regressions, LmmGroupings,
    LmmKernel, LmmWorkspace,
};
use crate::inference::evd::psd_pseudo_inverse;
use crate::standardize::{ColumnScaling, Units};

#[path = "satterthwaite_exact.rs"]
mod exact;

/// The finite-difference step factor: `h_k = ETA·m_b` on the kernel's θ̃,
/// `m_b` the largest `|θ̃|` of the coordinate's block ([`steps_into`]). Set by
/// the step sweep (`satterthwaite_tests.rs::step_sweep_is_flat_at_eta`, its
/// table there): over the steps 3e-5, 1e-4, 3e-4 and 1e-3 the df is flattest
/// between 3e-4 and 1e-3 (largest change 1.9e-5), while the deviance's
/// round-off from the random effects moves it by up to 6.6e-5 between 1e-4
/// and 3e-4 and 1.3e-3 between 3e-5 and 1e-4. The largest change at 3e-4 is
/// round-off on `sim_nested_crossed_mix`.
const ETA: f64 = 3e-4;

/// The pseudo-inverse threshold on `H_p` scaled to a unit diagonal
/// ([`assemble_into`]), relative to its largest eigenvalue.
const REL_TOL: f64 = 1e-8;

/// How `H_p` and `∂C/∂θ` were differentiated (a record the tests read).
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DerivRoute {
    /// The closed forms of `exact`.
    Exact,
    /// Central finite differences of the deviance value and of `C_p`.
    Value,
}

/// What the Satterthwaite pass leaves for the df and the contrasts. Every
/// `p × p` matrix is over the fit's retained columns in the caller's order
/// (all `p` columns unless [`SatterthwaiteState::retained`] says otherwise),
/// with the column standardization and the fixed-effect orthogonalization
/// mapped out.
///
/// The free θ coordinates are in the order the groupings were declared.
#[derive(Clone, Debug)]
pub(crate) struct SatterthwaiteState {
    /// `A_eff = 2·blockdiag(H_p⁺, τ̂²/ν)` over `(θ_free, τ)`, `n_free + 1`
    /// square.
    pub(crate) a_eff: Mat<f64>,
    /// The θ entries of `Jac_eff`: `∂C/∂θ̃_k`, one per free coordinate. The τ
    /// entry, `C/τ̂`, is formed on use.
    pub(crate) jac_theta: Vec<Mat<f64>>,
    /// `C = Var(β̂)` over every column, whatever the fit's target columns.
    pub(crate) c: Mat<f64>,
    /// τ̂ = σ̂², on the scale the fit reports it (prior weights' scale undone).
    pub(crate) tau: f64,
    /// The divisor of σ̂², from the kernel's criterion: `N − p` (retained
    /// columns) on a REML fit, `N` on an ML fit. The residual-only record of
    /// OLS and the dispersion GLMs carries `n − p` whatever the criterion.
    pub(crate) nu: f64,
    /// Directions of `H_p` dropped by the pseudo-inverse as null, counted on
    /// `H_p` scaled to a unit diagonal ([`assemble_into`]).
    pub(crate) dropped: usize,
    /// Directions below `−REL_TOL·λ_max`, not inverted, counted on the same
    /// scaled `H_p`.
    pub(crate) negative: usize,
    /// The derivative route the pass took, recorded for the tests only: the
    /// pass chooses the route itself (`collect_into`, from the kernel), and no
    /// non-test code reads it.
    #[cfg(test)]
    pub(crate) route: DerivRoute,
    /// Which record this is; one status, so no two of failed, switched off
    /// and not run can hold at once.
    status: Status,
    /// On a rank-deficient fit, the caller columns the state's `p × p`
    /// matrices cover, ascending (the state lives in the reduced column
    /// space); `None` when every column was retained.
    pub(crate) retained: Option<Vec<usize>>,
}

/// Which record a [`SatterthwaiteState`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Status {
    /// A pass that succeeded, or a residual-only record.
    Live,
    /// The pass failed: a deviance, `H_p`, `C` or τ̂ was not finite (or the
    /// eigendecomposition of `H_p` failed). Every df is then NaN.
    Failed,
    /// The pass was switched off (`FitOptions::satterthwaite = false`): the
    /// record holds no numbers (ν is NaN, so every df is NaN).
    SwitchedOff,
    /// No pass ran on this Gaussian LMM fit (the loop tier's
    /// `FitView::into_fit`): the record holds no numbers, as a switched-off
    /// one.
    NotRun,
}

/// What the pass collected at θ̂; [`assemble`] is a pure function of it. The
/// exact pass fills `hess` and `jac` (and `f0`, `c0`, `tau`, `nu`); the value
/// route fills the stencil's values.
#[derive(Clone, Debug)]
pub(crate) struct Collected {
    /// θ̃ indices of the free coordinates.
    pub(crate) free: Vec<usize>,
    /// Whether the exact pass filled `hess` and `jac`.
    pub(crate) exact: bool,
    /// The exact `H_p` over the free coordinates.
    pub(crate) hess: Mat<f64>,
    /// The exact `∂C/∂θ̃_k`, one per free coordinate.
    pub(crate) jac: Vec<Mat<f64>>,
    /// The step `h_k` of each free coordinate (value route).
    pub(crate) h: Vec<f64>,
    /// The deviance at θ̂.
    pub(crate) f0: f64,
    /// `[f(θ̂ + h_k e_k), f(θ̂ − h_k e_k)]` per free coordinate.
    pub(crate) f_axis: Vec<[f64; 2]>,
    /// `[f(θ̂ + h_i e_i + h_j e_j), f(θ̂ − h_i e_i − h_j e_j)]` per pair `i < j`
    /// of free coordinates, `i` major.
    pub(crate) f_mixed: Vec<[f64; 2]>,
    /// `C` at θ̂.
    pub(crate) c0: Mat<f64>,
    /// `[C(θ̂ + h_k e_k), C(θ̂ − h_k e_k)]` per free coordinate.
    pub(crate) c_axis: Vec<[Mat<f64>; 2]>,
    /// τ̂ (as [`SatterthwaiteState::tau`]).
    pub(crate) tau: f64,
    /// ν (as [`SatterthwaiteState::nu`]).
    pub(crate) nu: f64,
}

impl SatterthwaiteState {
    /// The record of a pass switched off by `FitOptions::satterthwaite`.
    pub(crate) fn switched_off() -> Self {
        SatterthwaiteState {
            a_eff: Mat::zeros(0, 0),
            jac_theta: Vec::new(),
            c: Mat::zeros(0, 0),
            tau: f64::NAN,
            nu: f64::NAN,
            dropped: 0,
            negative: 0,
            #[cfg(test)]
            route: DerivRoute::Value,
            status: Status::SwitchedOff,
            retained: None,
        }
    }

    /// The record of a Gaussian LMM fit no pass ran on (the loop tier's
    /// `FitView::into_fit`, whether or not the loop tier's `satterthwaite`
    /// later runs on the workspace: it writes only its `df_out`, never the
    /// `Fit`): no numbers, and no heap allocation, as [`Self::switched_off`].
    pub(crate) fn not_run() -> Self {
        SatterthwaiteState {
            status: Status::NotRun,
            ..Self::switched_off()
        }
    }

    /// The record of a Gaussian LMM fit with no finite endpoint, where no
    /// pass can run: failed, with no numbers.
    pub(crate) fn failed_record() -> Self {
        SatterthwaiteState {
            status: Status::Failed,
            ..Self::switched_off()
        }
    }

    /// The residual-only record of OLS and of a Gamma or inverse Gaussian
    /// GLM with φ estimated: the full `C` and `ν = n − p`, no variance
    /// parameters (`a_eff` 0 × 0, no `jac_theta`), so every df is ν. On a
    /// fit with no estimates (every entry of `beta` NaN, or no column) ν is
    /// NaN, never the `n − p` of a fit with `n ≤ p` or the `n` of one with no
    /// columns: there is no estimate to test.
    pub(crate) fn residual(c: Mat<f64>, nu: f64, beta: &[f64]) -> Self {
        SatterthwaiteState {
            c,
            nu: if beta.iter().all(|b| b.is_nan()) {
                f64::NAN
            } else {
                nu
            },
            status: Status::Live,
            ..Self::switched_off()
        }
    }

    /// A live record from its numbers, as a port stored them
    /// (`orchestrate::SatterthwaiteRecord`). The dropped and negative counts
    /// (read only by the fit's curvature note and the tests) and the
    /// test-only route are set to their defaults: no contrast reads them.
    #[cfg(feature = "orchestrate")]
    pub(crate) fn from_record(
        a_eff: Mat<f64>,
        jac_theta: Vec<Mat<f64>>,
        c: Mat<f64>,
        tau: f64,
        nu: f64,
        retained: Option<Vec<usize>>,
    ) -> Self {
        SatterthwaiteState {
            a_eff,
            jac_theta,
            c,
            tau,
            nu,
            retained,
            status: Status::Live,
            ..Self::switched_off()
        }
    }

    /// Whether this is a [`Self::residual`] record: a live state with no
    /// variance parameters (a Gaussian LMM's `a_eff` always has τ's row).
    pub(crate) fn residual_only(&self) -> bool {
        self.status == Status::Live && self.a_eff.nrows() == 0
    }

    /// Whether the pass failed: every df is NaN.
    pub(crate) fn is_failed(&self) -> bool {
        self.status == Status::Failed
    }

    /// Whether the pass was switched off by `FitOptions::satterthwaite`.
    pub(crate) fn is_disabled(&self) -> bool {
        self.status == Status::SwitchedOff
    }

    /// Whether this is the [`Self::not_run`] record.
    pub(crate) fn is_not_run(&self) -> bool {
        self.status == Status::NotRun
    }

    /// The Satterthwaite df of the contrast `l` (over the state's columns: the
    /// fit's retained columns, in caller order):
    /// `v²/(g_pᵀH_p⁺g_p + v²/ν)`, capped at ν; ν exactly when no coordinate is
    /// free; NaN after a failed pass. `g` is the buffer of `g_p`.
    fn df_into(&self, l: &[f64], g: &mut Vec<f64>) -> f64 {
        if self.is_failed() {
            return f64::NAN;
        }
        let n = self.jac_theta.len();
        if n == 0 {
            return self.nu;
        }
        let v = qform(l, self.c.as_ref());
        g.clear();
        g.extend(self.jac_theta.iter().map(|j| qform(l, j.as_ref())));
        // A_eff's θ block is 2·H_p⁺ exactly, so halving it is exact too.
        let ghg = 0.5 * qform(g, self.a_eff.as_ref().submatrix(0, 0, n, n));
        let df = v * v / (ghg + v * v / self.nu);
        // `v²/(v²/ν)` need not round back to ν; a NaN stays NaN.
        if df > self.nu {
            self.nu
        } else {
            df
        }
    }

    /// [`Self::df_into`] of every unit vector `e_i`.
    pub(crate) fn coef_df(&self) -> Vec<f64> {
        let p = self.c.nrows();
        let mut e = vec![0.0; p];
        let mut g = Vec::new();
        (0..p)
            .map(|i| {
                e[i] = 1.0;
                let df = self.df_into(&e, &mut g);
                e[i] = 0.0;
                df
            })
            .collect()
    }
}

/// The pass at the θ̂ the last fit left in `ws.theta`; `map` and `units` are
/// the column standardization and the units the fit's rows were accumulated
/// under.
pub(crate) fn pass(
    ws: &mut LmmWorkspace,
    map: &ColumnScaling,
    units: &Units,
) -> SatterthwaiteState {
    let nu = residual_nu(ws);
    let LmmWorkspace {
        kernel,
        recovery,
        theta,
        ..
    } = ws;
    assemble(&collect(
        kernel,
        theta,
        &recovery.beta_tilde,
        units,
        map,
        nu,
        ETA,
        true,
    ))
}

/// The loop tier's [`pass`]: the same evaluations and arithmetic, so the same
/// df bit for bit, on the buffers of [`LmmWorkspace::pass_scratch`]. They are
/// built on the first call, sized for every θ coordinate free, so each later
/// call on the workspace allocates only the pseudo-inverse's blocks (and, with
/// a crossed tail, the exact pass's tail-inverse scratch), whatever the draw's
/// free coordinates. Writes the df of each `targets` column into
/// `df_out` and leaves its other entries alone, and the df of each pair's
/// contrast `e_p − e_n` into `contrast_df_out` (NaN for a pair with an index
/// at least the column count, or with its two indices equal, which is no
/// contrast); returns `false`, writing
/// nothing, when the pass failed.
pub(crate) fn pass_targets(
    ws: &mut LmmWorkspace,
    map: &ColumnScaling,
    units: &Units,
    targets: &[u32],
    df_out: &mut [f64],
    contrasts: &[(u32, u32)],
    contrast_df_out: &mut [f64],
) -> bool {
    let nu = residual_nu(ws);
    let LmmWorkspace {
        kernel,
        recovery,
        theta,
        pass_scratch,
        ..
    } = ws;
    let s = pass_scratch
        .get_or_insert_with(|| Box::new(PassScratch::new(recovery.beta_tilde.len(), theta.len())));
    // The scratch is sized once, for the workspace's fixed `p` and `n_θ`.
    debug_assert_eq!(
        s.e.len(),
        recovery.beta_tilde.len(),
        "pass scratch: p changed"
    );
    collect_into(
        kernel,
        theta,
        &recovery.beta_tilde,
        units,
        map,
        nu,
        ETA,
        true,
        &mut s.col,
        &mut s.stencil,
    );
    assemble_into(
        &s.col,
        &mut s.state,
        &mut s.hess,
        &mut s.spare_jac,
        &mut s.diag_sqrt,
    );
    if s.state.is_failed() {
        return false;
    }
    for &t in targets {
        let t = t as usize;
        s.e[t] = 1.0;
        df_out[t] = s.state.df_into(&s.e, &mut s.g);
        s.e[t] = 0.0;
    }
    let p = s.e.len();
    for (&(pc, nc), out) in contrasts.iter().zip(contrast_df_out.iter_mut()) {
        let (pc, nc) = (pc as usize, nc as usize);
        if pc >= p || nc >= p || pc == nc {
            *out = f64::NAN;
            continue;
        }
        s.e[pc] = 1.0;
        s.e[nc] = -1.0;
        *out = s.state.df_into(&s.e, &mut s.g);
        s.e[pc] = 0.0;
        s.e[nc] = 0.0;
    }
    true
}

/// The buffers of [`pass_targets`], one set per workspace: the collected
/// values, the stencil's and the assembly's buffers and the state, each sized
/// for `p` columns and every one of the `n_θ` coordinates free.
pub(crate) struct PassScratch {
    col: Collected,
    stencil: StencilScratch,
    state: SatterthwaiteState,
    /// `H_p`, and `√diag(H_p)`, as [`assemble_into`] leaves them.
    hess: Mat<f64>,
    diag_sqrt: Vec<f64>,
    /// The `∂C/∂θ̃_k` matrices `state.jac_theta` does not hold this call.
    spare_jac: Vec<Mat<f64>>,
    /// `g_p` of [`SatterthwaiteState::df_into`], and the contrast it takes:
    /// the unit vector `e_t` or a pair's `e_p − e_n`, all zero between uses.
    g: Vec<f64>,
    e: Vec<f64>,
}

impl PassScratch {
    fn new(p: usize, n_theta: usize) -> Self {
        PassScratch {
            col: Collected::with_capacity(p, n_theta),
            stencil: StencilScratch::new(p, n_theta),
            state: SatterthwaiteState {
                a_eff: Mat::zeros(n_theta + 1, n_theta + 1),
                jac_theta: Vec::with_capacity(n_theta),
                c: Mat::zeros(p, p),
                ..SatterthwaiteState::switched_off()
            },
            hess: Mat::zeros(n_theta, n_theta),
            diag_sqrt: Vec::with_capacity(n_theta),
            spare_jac: (0..n_theta).map(|_| Mat::zeros(p, p)).collect(),
            g: Vec::with_capacity(n_theta),
            e: vec![0.0; p],
        }
    }
}

/// The pass's buffers besides the values it collects, sized for `p`
/// columns and `n_θ` coordinates.
struct StencilScratch {
    /// `M = B·U⁻¹` (see [`stencil`]), and one row of it.
    m: Mat<f64>,
    row: Vec<f64>,
    /// `W` of [`c_at_into`].
    w: Mat<f64>,
    /// The evaluation point.
    point: Vec<f64>,
    /// The `C` pairs [`Collected::c_axis`] and the `∂C/∂θ̃_k`
    /// [`Collected::jac`] do not hold this call.
    spare_c: Vec<[Mat<f64>; 2]>,
    spare_jac: Vec<Mat<f64>>,
    /// The exact pass's buffers.
    exact: exact::ExactScratch,
    /// What [`shift_response_into`] replaced, and its shifted Gram row.
    saved: SavedResponse,
    gram_row: Vec<f64>,
}

impl StencilScratch {
    fn new(p: usize, n_theta: usize) -> Self {
        StencilScratch {
            m: Mat::zeros(p, p),
            row: vec![0.0; p],
            w: Mat::zeros(p, p),
            point: Vec::with_capacity(n_theta),
            spare_c: (0..n_theta)
                .map(|_| [Mat::zeros(p, p), Mat::zeros(p, p)])
                .collect(),
            spare_jac: (0..n_theta).map(|_| Mat::zeros(p, p)).collect(),
            exact: exact::ExactScratch::default(),
            saved: SavedResponse::default(),
            gram_row: Vec::new(),
        }
    }
}

impl Collected {
    /// Empty values with room for `n_θ` free coordinates and `p × p` `C`.
    fn with_capacity(p: usize, n_theta: usize) -> Self {
        Collected {
            free: Vec::with_capacity(n_theta),
            exact: false,
            hess: Mat::zeros(n_theta, n_theta),
            jac: Vec::with_capacity(n_theta),
            h: Vec::with_capacity(n_theta),
            f0: f64::NAN,
            f_axis: Vec::with_capacity(n_theta),
            f_mixed: Vec::with_capacity(n_theta * n_theta.saturating_sub(1) / 2),
            c0: Mat::zeros(p, p),
            c_axis: Vec::with_capacity(n_theta),
            tau: f64::NAN,
            nu: f64::NAN,
        }
    }
}

/// Sets `v`'s length to `n`, moving its surplus to `spare` and taking what it
/// lacks from there (`make` builds one only when `spare` runs out), so buffers
/// built once serve every `n` up to their count without allocating.
fn pooled_resize<T>(v: &mut Vec<T>, spare: &mut Vec<T>, n: usize, mut make: impl FnMut() -> T) {
    if v.len() > n {
        spare.extend(v.drain(n..));
    }
    while v.len() < n {
        v.push(spare.pop().unwrap_or_else(&mut make));
    }
}

/// [`pass`] at a caller's kernel θ̃ instead of `ws.theta`.
#[cfg(test)]
pub(crate) fn pass_at(
    ws: &mut LmmWorkspace,
    theta: &[f64],
    map: &ColumnScaling,
    units: &Units,
) -> SatterthwaiteState {
    let nu = residual_nu(ws);
    assemble(&collect(
        &mut ws.kernel,
        theta,
        &ws.recovery.beta_tilde,
        units,
        map,
        nu,
        ETA,
        true,
    ))
}

/// [`pass_at`] on the value route, whatever the kernel: the reference the
/// exact pass is checked against.
#[cfg(all(test, feature = "formula"))]
pub(crate) fn pass_value_at(
    ws: &mut LmmWorkspace,
    theta: &[f64],
    map: &ColumnScaling,
    units: &Units,
) -> SatterthwaiteState {
    let nu = residual_nu(ws);
    assemble(&collect(
        &mut ws.kernel,
        theta,
        &ws.recovery.beta_tilde,
        units,
        map,
        nu,
        ETA,
        false,
    ))
}

/// ν, the divisor of σ̂², by the criterion the kernel evaluates
/// ([`LmmKernel::criterion`]): `N − p` under REML, `N` under ML (`p` the
/// retained columns).
fn residual_nu(ws: &LmmWorkspace) -> f64 {
    let kernel = &ws.kernel;
    kernel
        .criterion()
        .nu(kernel.n_rows(), ws.recovery.betas.len()) as f64
}

/// The θ̃ indices of every Λ column that is not exactly zero in `theta`, in
/// θ̃ order (each block's `vech` is column-major, so a column is a contiguous
/// run from its diagonal), less the entries the grouping's terms hold at zero:
/// the fit never moved those, so they are not parameters of its criterion.
pub(crate) fn free_coordinates(g: &LmmGroupings, theta: &[f64]) -> Vec<usize> {
    let mut free = Vec::new();
    free_coordinates_into(g, theta, &mut free);
    free
}

/// [`free_coordinates`] into `free`.
fn free_coordinates_into(g: &LmmGroupings, theta: &[f64], free: &mut Vec<usize>) {
    free.clear();
    let structural = g.structural_theta();
    for (&start, &len) in g.diagonal_theta().iter().zip(&g.diagonal_run_len) {
        let column = start..start + len;
        if theta[column.clone()].iter().any(|&t| t != 0.0) {
            // A free diagonal's run still holds the structural entries below
            // it (the covariances with a later term).
            free.extend(column.filter(|k| structural.binary_search(k).is_err()));
        }
    }
}

/// The pass's values at θ̂: the exact pass where it covers the kernel and
/// `exact` allows it ([`exact_into`]), the value route otherwise, on the
/// response shifted by `X̃·beta_tilde` ([`shift_response_into`]) with the
/// kernel's statistics put back afterwards. `beta_tilde` is the fit's β̂ on the
/// kernel's design and in its units. `units` are the rows' units: τ̂ is
/// reported in the caller's (`Units::dispersion_scale`), and `C` takes `k²`
/// (`Units::var_scale`; the weight scale cancels between the kernel's σ̂² and
/// its `(X̃'V⁻¹X̃)⁻¹`). `eta` is the value route's step factor ([`ETA`] in
/// the pass).
#[allow(clippy::too_many_arguments)] // the pass's inputs, split off the workspace
fn collect(
    kernel: &mut LmmKernel,
    theta: &[f64],
    beta_tilde: &[f64],
    units: &Units,
    map: &ColumnScaling,
    nu: f64,
    eta: f64,
    exact: bool,
) -> Collected {
    let (p, n_theta) = (beta_tilde.len(), theta.len());
    let mut col = Collected::with_capacity(p, n_theta);
    collect_into(
        kernel,
        theta,
        beta_tilde,
        units,
        map,
        nu,
        eta,
        exact,
        &mut col,
        &mut StencilScratch::new(p, n_theta),
    );
    col
}

/// [`collect`] into `col`, on the buffers `s`; every value of `col` is
/// rewritten.
#[allow(clippy::too_many_arguments)] // the pass's inputs, split off the workspace
fn collect_into(
    kernel: &mut LmmKernel,
    theta: &[f64],
    beta_tilde: &[f64],
    units: &Units,
    map: &ColumnScaling,
    nu: f64,
    eta: f64,
    exact: bool,
    col: &mut Collected,
    s: &mut StencilScratch,
) {
    if exact && exact::covers(kernel) {
        exact_into(kernel, theta, units, map, nu, col, s);
        return;
    }
    shift_response_into(kernel, beta_tilde, &mut s.saved, &mut s.gram_row);
    stencil(kernel, theta, units, map, nu, eta, col, s);
    restore_response(kernel, &s.saved);
}

/// `M = B·U⁻¹` into `m` (`p × p`), with `row` a length-`p` scratch: row `t`
/// maps the kernel's coefficients to the caller's coefficient `t`, so
/// `β_t = M_t·β̃` and `C = σ̂²·M(L̃L̃ᵀ)⁻¹Mᵀ`, `L̃` the unmapped fixed-effect
/// factor. `U` is the kernel's [`LmmKernel::design_qr`], fixed by the
/// accumulation, so `M` does not depend on θ.
pub(crate) fn caller_map_into(
    kernel: &LmmKernel,
    map: &ColumnScaling,
    row: &mut [f64],
    m: &mut Mat<f64>,
) {
    for t in 0..m.nrows() {
        for (i, r) in row.iter_mut().enumerate() {
            *r = map.rhs(t, i);
        }
        kernel.design_qr().solve_row(row);
        for (i, &r) in row.iter().enumerate() {
            m[(t, i)] = r;
        }
    }
}

/// The exact pass: the deviance, `C` and τ̂ at θ̂, then `H_p`
/// ([`exact::exact_terms`]) and `∂C/∂θ̃_k = k²·Wᵀ(σ̂²·NᵀD_kN − (a_k/ν)·I)W`
/// with `W = L̃⁻¹Mᵀ` as [`c_at_into`] leaves it (the kernel's
/// `∂C̃/∂θ̃_k = L̃⁻ᵀ(σ̂²·NᵀD_kN − (a_k/ν)·I)L̃⁻¹`, mapped as `C` is). A deviance
/// or a family factor that fails leaves NaN for [`assemble_into`] to fail on.
#[allow(clippy::too_many_arguments)] // as `stencil`
fn exact_into(
    kernel: &mut LmmKernel,
    theta: &[f64],
    units: &Units,
    map: &ColumnScaling,
    nu: f64,
    col: &mut Collected,
    s: &mut StencilScratch,
) {
    let p = s.m.nrows();
    let k2 = units.var_scale();
    free_coordinates_into(kernel.groupings(), theta, &mut col.free);
    let n = col.free.len();
    caller_map_into(kernel, map, &mut s.row, &mut s.m);
    col.exact = true;
    col.h.clear();
    col.f_axis.clear();
    col.f_mixed.clear();
    pooled_resize(&mut col.c_axis, &mut s.spare_c, 0, || {
        [Mat::zeros(p, p), Mat::zeros(p, p)]
    });
    pooled_resize(&mut col.jac, &mut s.spare_jac, n, || Mat::zeros(p, p));
    col.f0 = kernel.deviance(theta);
    c_at_into(kernel, &s.m, &mut s.w, &mut col.c0, k2);
    let s2 = kernel.sigma_sq();
    col.tau = s2 * units.dispersion_scale();
    col.nu = nu;
    if !col.f0.is_finite() || n == 0 {
        col.hess.resize_with(n, n, |_, _| f64::NAN);
        return;
    }
    if !exact::exact_terms(kernel, theta, &col.free, nu, &mut s.exact, &mut col.hess) {
        return;
    }
    let x = &mut s.exact;
    for (k, jac) in col.jac.iter_mut().enumerate() {
        jac.resize_with(p, p, |_, _| 0.0);
        let ndn = &x.sums.ndn[k * p * p..(k + 1) * p * p];
        let ak = x.sums.a[k] / nu;
        // prod = (σ̂²·NᵀD_kN − (a_k/ν)·I)·W, then jac = k²·Wᵀ·prod.
        for i in 0..p {
            for j in 0..p {
                x.prod[i * p + j] = (0..p)
                    .map(|u| (s2 * ndn[i * p + u] - if u == i { ak } else { 0.0 }) * s.w[(u, j)])
                    .sum();
            }
        }
        for a in 0..p {
            for b in a..p {
                let v = k2 * (0..p).map(|i| s.w[(i, a)] * x.prod[i * p + b]).sum::<f64>();
                jac[(a, b)] = v;
                jac[(b, a)] = v;
            }
        }
    }
}

/// The value route's evaluations, on whatever response the kernel holds, at
/// the step factor `eta`.
#[allow(clippy::too_many_arguments)] // as `collect_into`
fn stencil(
    kernel: &mut LmmKernel,
    theta: &[f64],
    units: &Units,
    map: &ColumnScaling,
    nu: f64,
    eta: f64,
    col: &mut Collected,
    s: &mut StencilScratch,
) {
    let p = s.m.nrows();
    let k2 = units.var_scale();
    free_coordinates_into(kernel.groupings(), theta, &mut col.free);
    let n = col.free.len();
    caller_map_into(kernel, map, &mut s.row, &mut s.m);
    col.exact = false;
    pooled_resize(&mut col.jac, &mut s.spare_jac, 0, || Mat::zeros(p, p));

    s.point.clear();
    s.point.extend_from_slice(theta);
    col.f0 = kernel.deviance(&s.point);
    c_at_into(kernel, &s.m, &mut s.w, &mut col.c0, k2);
    col.tau = kernel.sigma_sq() * units.dispersion_scale();
    col.nu = nu;
    col.f_mixed.clear();

    steps_into(kernel.groupings(), theta, &col.free, eta, &mut col.h);
    col.f_axis.clear();
    pooled_resize(&mut col.c_axis, &mut s.spare_c, n, || {
        [Mat::zeros(p, p), Mat::zeros(p, p)]
    });
    for (a, (&k, &hk)) in col.free.iter().zip(&col.h).enumerate() {
        let mut f = [0.0; 2];
        for (side, step) in [hk, -hk].into_iter().enumerate() {
            s.point[k] = theta[k] + step;
            f[side] = kernel.deviance(&s.point);
            s.point[k] = theta[k];
            c_at_into(kernel, &s.m, &mut s.w, &mut col.c_axis[a][side], k2);
        }
        col.f_axis.push(f);
    }

    for i in 0..n {
        for j in i + 1..n {
            let (ki, kj) = (col.free[i], col.free[j]);
            let mut f = [0.0; 2];
            for (side, sign) in [1.0, -1.0].into_iter().enumerate() {
                s.point[ki] = theta[ki] + sign * col.h[i];
                s.point[kj] = theta[kj] + sign * col.h[j];
                f[side] = kernel.deviance(&s.point);
                s.point[ki] = theta[ki];
                s.point[kj] = theta[kj];
            }
            col.f_mixed.push(f);
        }
    }
}

/// The steps of the free coordinates `free` (ascending) into `h`: `eta·m_b`,
/// with `m_b` the largest `|θ̃|` over the free coordinates of the coordinate's
/// `vech(Λ_g)` block (the primary grouping's, then each extra grouping's in
/// declaration order). A rescale of θ̃ by `2^−k` scales every step with it, so
/// it leaves the df unchanged; a free coordinate's diagonal run is nonzero, so
/// `m_b > 0`.
fn steps_into(g: &LmmGroupings, theta: &[f64], free: &[usize], eta: f64, h: &mut Vec<f64>) {
    h.clear();
    let mut rest = free;
    for b in 0..g.n_blocks() {
        let end = g.vech_start(b + 1);
        let (block, tail) = rest.split_at(rest.partition_point(|&k| k < end));
        let m_b = block.iter().fold(0.0, |m: f64, &k| m.max(theta[k].abs()));
        h.extend(block.iter().map(|_| eta * m_b));
        rest = tail;
    }
}

/// The kernel's statistics that hold the response, as they were before
/// [`shift_response_into`]: the `y` row of the `[X̃ y]` Gram (and its `y`
/// column, which the sparse kernel stores too), the `y` entry of each
/// random-effect column's `Z'[X̃ y]`, and the `y` column of the QR kernel's
/// triangles while it is on.
#[derive(Default)]
struct SavedResponse {
    xy: Vec<f64>,
    yx: Vec<f64>,
    zy: Vec<f64>,
    qr_y: Vec<f64>,
}

/// [`shift_response_into`] on fresh buffers.
#[cfg(all(test, feature = "formula"))]
fn shift_response(kernel: &mut LmmKernel, b: &[f64]) -> SavedResponse {
    let mut saved = SavedResponse::default();
    shift_response_into(kernel, b, &mut saved, &mut Vec::new());
    saved
}

/// Replace the response `y` by `y − X̃b` in the kernel's sufficient statistics
/// (and re-arm the dense kernel's precomputes on them), saving what was there
/// into `saved`; `row` is the buffer of the shifted Gram row. Every statistic
/// is linear or quadratic in the row `[x̃ y]`, so the shift is exact algebra on
/// them: `Z'y ← Z'y − Z'X̃b`, `X̃'y ← X̃'y − X̃'X̃b`, `y'y ← y'y − bᵀ(X̃'y + X̃'y_new)`.
///
/// The profiled deviance and `C` do not depend on `b` (β is profiled out), but
/// their round-off does: each evaluation reads `r²` off a downdate of `y'y`,
/// which carries an absolute error of order `ε·y'y`, and the stencil's second
/// differences divide it by `h²`. With `b` the fit's β̂ the shifted `y'y` is the
/// residual's, not the raw response's. The shift's own rounding is fixed (it
/// does not depend on θ), so it moves the deviance smoothly, by a relative
/// `ε·y'y/r²` at most. Both kernels' exact-fit floors are `dim·ε·y'y` on the
/// statistics, so they drop with the shift as well: a stencil point near an
/// exact fit can come back finite where the unshifted statistics would call
/// it exact.
fn shift_response_into(
    kernel: &mut LmmKernel,
    b: &[f64],
    saved: &mut SavedResponse,
    row: &mut Vec<f64>,
) {
    let p = b.len();
    let (xy, yx, zy) = (&mut saved.xy, &mut saved.yx, &mut saved.zy);
    xy.clear();
    yx.clear();
    zy.clear();
    saved.qr_y.clear();
    match kernel {
        LmmKernel::Dense { suff, fit } => {
            debug_assert_eq!(p, suff.m - 1, "b has the kernel's p entries");
            // `c` holds its lower triangle only.
            let c = &suff.c;
            xy.extend((0..=p).map(|j| c[(p, j)]));
            shifted_gram_row(|i, j| c[(i.max(j), i.min(j))], b, row);
            for (j, &v) in row.iter().enumerate() {
                suff.c[(p, j)] = v;
            }
            zy.extend((0..suff.s.ncols()).map(|a| suff.s[(p, a)]));
            for a in 0..suff.s.ncols() {
                let shift: f64 = (0..p).map(|j| b[j] * suff.s[(j, a)]).sum();
                suff.s[(p, a)] -= shift;
            }
            if let Some(qr) = fit.qr.as_deref_mut().filter(|q| q.on) {
                qr.shift_response(b, &mut saved.qr_y);
            }
            precompute_balanced_collapse(suff, fit);
            precompute_group_regressions(suff, fit);
        }
        LmmKernel::Sparse { ws, .. } => {
            let ws = LmmKernel::sparse_ws_mut(ws);
            debug_assert_eq!(p, ws.p, "b has the kernel's p entries");
            let m = p + 1;
            let cxy = &ws.cxy;
            xy.extend((0..=p).map(|j| cxy[(p, j)]));
            yx.extend((0..=p).map(|i| cxy[(i, p)]));
            shifted_gram_row(|i, j| cxy[(i.max(j), i.min(j))], b, row);
            for (j, &v) in row.iter().enumerate() {
                ws.cxy[(p, j)] = v;
                ws.cxy[(j, p)] = v;
            }
            zy.extend(ws.ztxy.chunks(m).map(|r| r[p]));
            for r in ws.ztxy.chunks_mut(m) {
                let shift: f64 = (0..p).map(|j| b[j] * r[j]).sum();
                r[p] -= shift;
            }
        }
    }
}

/// The `y` row of the Gram `g` (read symmetric, `y` at index `b.len()`) after
/// `y ← y − X̃b`, into `row`.
fn shifted_gram_row(g: impl Fn(usize, usize) -> f64, b: &[f64], row: &mut Vec<f64>) {
    let p = b.len();
    row.clear();
    row.reserve(p + 1);
    row.extend((0..p).map(|j| g(p, j) - (0..p).map(|k| b[k] * g(k, j)).sum::<f64>()));
    let yy = g(p, p) - (0..p).map(|j| b[j] * (g(p, j) + row[j])).sum::<f64>();
    row.push(yy);
}

/// Undo [`shift_response_into`] bit for bit, the dense kernel's precomputes
/// included, so the pass leaves the statistics a later evaluation reads (the
/// curvature hint's, a refit's) as the fit left them.
fn restore_response(kernel: &mut LmmKernel, saved: &SavedResponse) {
    match kernel {
        LmmKernel::Dense { suff, fit } => {
            let p = suff.m - 1;
            for (j, &v) in saved.xy.iter().enumerate() {
                suff.c[(p, j)] = v;
            }
            for (a, &v) in saved.zy.iter().enumerate() {
                suff.s[(p, a)] = v;
            }
            if let Some(qr) = fit.qr.as_deref_mut().filter(|q| q.on) {
                qr.restore_response(&saved.qr_y);
            }
            precompute_balanced_collapse(suff, fit);
            precompute_group_regressions(suff, fit);
        }
        LmmKernel::Sparse { ws, .. } => {
            let ws = LmmKernel::sparse_ws_mut(ws);
            let p = ws.p;
            for (j, &v) in saved.xy.iter().enumerate() {
                ws.cxy[(p, j)] = v;
            }
            for (i, &v) in saved.yx.iter().enumerate() {
                ws.cxy[(i, p)] = v;
            }
            for (r, &v) in ws.ztxy.chunks_mut(ws.m).zip(&saved.zy) {
                r[p] = v;
            }
        }
    }
}

/// `C = k2·σ̂²·WᵀW`, `W = L̃⁻¹Mᵀ` (into `w`), from the factor and σ̂² the
/// kernel's last evaluation left, into `c` (`p × p`, as `w`); exactly
/// symmetric. `k2` is `Units::var_scale`, so `C` is in the caller's units.
fn c_at_into(kernel: &LmmKernel, m: &Mat<f64>, w: &mut Mat<f64>, c: &mut Mat<f64>, k2: f64) {
    let p = m.nrows();
    let (l, s2) = (kernel.factor(), kernel.sigma_sq() * k2);
    for t in 0..p {
        let col = w.col_as_slice_mut(t);
        for (i, v) in col.iter_mut().enumerate() {
            *v = m[(t, i)];
        }
        forward_solve(l, p, col);
    }
    for a in 0..p {
        for b in a..p {
            let v = s2 * (0..p).map(|i| w[(i, a)] * w[(i, b)]).sum::<f64>();
            c[(a, b)] = v;
            c[(b, a)] = v;
        }
    }
}

/// `xᵀAx`.
fn qform(x: &[f64], a: MatRef<'_, f64>) -> f64 {
    let mut acc = 0.0;
    for (i, &xi) in x.iter().enumerate() {
        for (j, &xj) in x.iter().enumerate() {
            acc += xi * a[(i, j)] * xj;
        }
    }
    acc
}

/// `H_p`, its pseudo-inverse, `A_eff` and the θ entries of `Jac_eff` from the
/// collected values.
pub(crate) fn assemble(col: &Collected) -> SatterthwaiteState {
    let mut st = SatterthwaiteState::switched_off();
    assemble_into(
        col,
        &mut st,
        &mut Mat::zeros(0, 0),
        &mut Vec::new(),
        &mut Vec::new(),
    );
    st
}

/// [`assemble`] into `st`, every field rewritten, with `hess` the buffer of
/// `H_p` (left scaled to its unit diagonal), `spare_jac` the pool
/// `st.jac_theta` resizes through and `diag_sqrt` the buffer of `√diag(H_p)`.
///
/// The pseudo-inverse cuts on `D^−½·H_p·D^−½`, `D` the diagonal of `H_p`, and
/// its result is mapped back by `D^−½`, so a coordinate's units alone drop no
/// direction. When a diagonal entry is not positive and finite, `D` is the
/// identity: the cut on `H_p` itself.
fn assemble_into(
    col: &Collected,
    st: &mut SatterthwaiteState,
    hess: &mut Mat<f64>,
    spare_jac: &mut Vec<Mat<f64>>,
    diag_sqrt: &mut Vec<f64>,
) {
    let n = col.free.len();
    let finite = |m: &Mat<f64>| m.col_iter().all(|c| c.iter().all(|v| v.is_finite()));
    let mut failed = !(col.f0.is_finite() && col.tau.is_finite() && finite(&col.c0))
        || col
            .f_axis
            .iter()
            .chain(&col.f_mixed)
            .flatten()
            .any(|v| !v.is_finite())
        || col.c_axis.iter().flatten().any(|c| !finite(c))
        || (col.exact && (!finite(&col.hess) || col.jac.iter().any(|j| !finite(j))));

    hessian_into(col, hess);
    let (pr, pc) = (col.c0.nrows(), col.c0.ncols());
    if col.exact {
        pooled_resize(&mut st.jac_theta, spare_jac, col.jac.len(), || {
            Mat::zeros(pr, pc)
        });
        for (jac, src) in st.jac_theta.iter_mut().zip(&col.jac) {
            jac.resize_with(pr, pc, |_, _| 0.0);
            jac.as_mut().copy_from(src.as_ref());
        }
    } else {
        let n_jac = col.c_axis.len().min(col.h.len());
        pooled_resize(&mut st.jac_theta, spare_jac, n_jac, || Mat::zeros(pr, pc));
        for (jac, ([cp, cm], &hk)) in st.jac_theta.iter_mut().zip(col.c_axis.iter().zip(&col.h)) {
            jac.resize_with(cp.nrows(), cp.ncols(), |_, _| 0.0);
            for b in 0..cp.ncols() {
                for a in 0..cp.nrows() {
                    jac[(a, b)] = (cp[(a, b)] - cm[(a, b)]) / (2.0 * hk);
                }
            }
        }
    }

    let a_eff = &mut st.a_eff;
    a_eff.resize_with(n + 1, n + 1, |_, _| f64::NAN);
    for j in 0..=n {
        for i in 0..=n {
            a_eff[(i, j)] = f64::NAN;
        }
    }
    let (mut dropped, mut negative) = (0, 0);
    if !failed {
        diag_sqrt.clear();
        // `√|H_ii|`, so a negative curvature keeps its sign and is counted as
        // negative; a zero or non-finite diagonal entry is left unscaled.
        diag_sqrt.extend((0..n).map(|i| {
            let d = hess[(i, i)].abs().sqrt();
            if d.is_finite() && d > 0.0 {
                d
            } else {
                1.0
            }
        }));
        for j in 0..n {
            for i in 0..n {
                hess[(i, j)] /= diag_sqrt[i] * diag_sqrt[j];
            }
        }
        match psd_pseudo_inverse(hess.as_ref(), REL_TOL) {
            Ok(pinv) => {
                for j in 0..=n {
                    for i in 0..=n {
                        a_eff[(i, j)] = 0.0;
                    }
                }
                for j in 0..n {
                    for i in 0..n {
                        a_eff[(i, j)] = 2.0 * pinv.inv[(i, j)] / (diag_sqrt[i] * diag_sqrt[j]);
                    }
                }
                a_eff[(n, n)] = 2.0 * col.tau * col.tau / col.nu;
                (dropped, negative) = (pinv.dropped, pinv.negative);
            }
            Err(_) => failed = true,
        }
    }

    st.c.resize_with(pr, pc, |_, _| 0.0);
    st.c.as_mut().copy_from(col.c0.as_ref());
    st.tau = col.tau;
    st.nu = col.nu;
    st.dropped = dropped;
    st.negative = negative;
    #[cfg(test)]
    {
        st.route = if col.exact {
            DerivRoute::Exact
        } else {
            DerivRoute::Value
        };
    }
    st.status = if failed { Status::Failed } else { Status::Live };
    st.retained = None;
}

/// [`hessian_into`] on a fresh matrix.
#[cfg(all(test, feature = "formula"))]
fn hessian(col: &Collected) -> Mat<f64> {
    let mut hess = Mat::zeros(0, 0);
    hessian_into(col, &mut hess);
    hess
}

/// `H_p` over the free coordinates from the collected values, exactly
/// symmetric, into `hess`.
fn hessian_into(col: &Collected, hess: &mut Mat<f64>) {
    let n = col.free.len();
    hess.resize_with(n, n, |_, _| 0.0);
    if col.exact {
        hess.as_mut().copy_from(col.hess.as_ref());
        return;
    }
    // Central differences: the axis pairs on the diagonal, and per pair
    // (f(+i+j) + f(−i−j) − f(±i) − f(±j) + 2f₀)/(2hᵢhⱼ) off it, written to
    // both triangles so H_p is exactly symmetric.
    for j in 0..n {
        for i in 0..n {
            hess[(i, j)] = 0.0;
        }
    }
    for (k, ([fp, fm], &hk)) in col.f_axis.iter().zip(&col.h).enumerate() {
        hess[(k, k)] = (fp - 2.0 * col.f0 + fm) / (hk * hk);
    }
    let mut pair = col.f_mixed.iter();
    for i in 0..n {
        for j in i + 1..n {
            let [fpp, fmm] = pair.next().expect("one mixed pair per i < j");
            let axis = col.f_axis[i][0] + col.f_axis[i][1] + col.f_axis[j][0] + col.f_axis[j][1];
            let v = (fpp + fmm - axis + 2.0 * col.f0) / (2.0 * col.h[i] * col.h[j]);
            hess[(i, j)] = v;
            hess[(j, i)] = v;
        }
    }
}

#[cfg(test)]
#[path = "satterthwaite_tests.rs"]
pub(crate) mod tests;
