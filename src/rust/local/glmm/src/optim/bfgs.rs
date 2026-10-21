//! Projected BFGS on a box, for routes with an analytic gradient.
//!
//! The search keeps an `n×n` inverse-Hessian approximation `H` and, at the
//! incumbent `x` with gradient `g`, steps along `d = −H_FF g_F` over the free
//! set `F` (a coordinate on a bound with the gradient pushing outward is held
//! for the step). Each step backtracks (Armijo, halving) on value-only
//! evaluations of the projected point; the accepted trial is the last value
//! the objective saw, so its gradient comes off that evaluation's state
//! ([`GradObjective`]). `H` starts as the identity, is rescaled by Shanno's
//! `sᵀy/yᵀy` at the first update, and resets to the identity on a step that
//! is not a descent direction. The first step after a (re)start is capped at
//! [`BfgsConfig::first_step_cap`] in the max norm, so the identity's
//! unit-free step cannot leave the region the start describes.
//!
//! An objective that knows some coordinates' curvatures at the start
//! ([`GradObjective::start_curvature`]) seeds those diagonal entries of `H`
//! with their inverses instead of 1. A seeded entry keeps its value through
//! Shanno's rescaling and every restart, and the first-step cap reads and
//! scales the unseeded coordinates only, since a seeded coordinate's step is
//! already in its own units. Shanno's rescaling gives each contiguous run of
//! unseeded coordinates its own `sᵀy/yᵀy`, read over that run alone: seeds
//! that split the point into blocks (the GLMM route's β̃ between θ and ψ)
//! leave blocks whose curvatures differ by orders of magnitude, and one ratio
//! over both would set one block's scale from the other's. Without a seed the
//! one run is every coordinate, the plain Shanno scale. A first step across a
//! concave stretch of the unseeded coordinates makes no update, so the next
//! step starts again from the seeded start under the first-step cap.
//!
//! Stops:
//!
//! - `Pgrad`: the projected-gradient max norm `max|P(x − g) − x|` is below
//!   [`BfgsConfig::pgrad_tol`];
//! - `Fdelta`: after the second iteration, the accepted decrease is below
//!   `max(fdelta_abs, FDELTA_ULPS·ε·|f|)` ([`BfgsConfig::fdelta_abs`]). The
//!   absolute part is in deviance units: a relative stop (`1e-11·(1 + |f|)`)
//!   grows with n, because the deviance is O(n), and ended a 300k-row GLMM
//!   4.7e-4 above its optimum (2026-10-07 measurements). The ulp part
//!   keeps the stop reachable where the absolute part is below the
//!   deviance's own rounding (at 1e-9, |f| above about 3e5 makes `16·ε·|f|`
//!   exceed it; at |f| = 1e7 it is 3.6e-8, still three orders below
//!   `dev_eps`);
//! - `NoiseFloor`: no trial could be accepted, every trial's predicted
//!   decrease was below that threshold, and the decrement is below the
//!   certificate's bound (a warm start at the optimum stops here after one
//!   value and one gradient);
//! - `LineSearch`: no accepted trial in [`BfgsConfig::max_backtracks`]
//!   halvings; `MaxIter`: the iteration cap; `GradUnavailable` and
//!   `NonFinite`: the objective failed at the start or at an accepted point.
//!
//! [`BfgsOutcome::certified`] is the acceptance test every route applies to
//! the endpoint; an endpoint that fails it is polished by BOBYQA.
//!
//! No allocation per iteration: every buffer is [`BfgsState`], sized on the
//! first search of a dimension and reset (not carried) at the start of every
//! search.

use super::GradObjective;

/// Relative floor of the `Fdelta` stop, in units of `ε·|f|`: two deviances
/// that differ by fewer ulps than this are not a measured decrease.
pub(crate) const FDELTA_ULPS: f64 = 16.0;

/// The certificate's bound on the BFGS decrement `½·gᵀHg` (free set, deviance
/// units). Measured endpoint maximum on the LMM route: 1.7e-9 cold.
pub(crate) const CERT_DECREMENT: f64 = 1e-8;

/// A diagonal θ with `|θ| <` this fails the certificate: near a variance
/// collapse the decrement can understate the remaining gain (measured 60x on a
/// GLMM boundary cell), so such an endpoint is polished. Shared with the GLMM
/// route.
pub(crate) const CERT_NEAR_ZERO: f64 = 1e-3;

/// Per-route search settings.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BfgsConfig {
    /// Iteration cap; reaching it hands the incumbent to BOBYQA.
    pub max_iter: usize,
    /// Armijo halvings per line search before the search gives up.
    pub max_backtracks: usize,
    /// Armijo sufficient-decrease constant.
    pub armijo_c: f64,
    /// `Pgrad` stop threshold on the projected-gradient max norm.
    pub pgrad_tol: f64,
    /// Max-norm cap of the first step after a (re)start of `H`.
    pub first_step_cap: f64,
    /// Absolute part of the `Fdelta` and `NoiseFloor` threshold, in deviance
    /// units (see the module doc).
    pub fdelta_abs: f64,
}

impl BfgsConfig {
    /// The LMM per-group route's settings (measured 2026-10-06/07 on 44 cold
    /// and 50 warm cells: every stop `Fdelta`, no cap reached).
    pub(crate) const LMM: Self = Self {
        max_iter: 1000,
        max_backtracks: 30,
        armijo_c: 1e-4,
        pgrad_tol: 1e-6,
        first_step_cap: 0.5,
        fdelta_abs: 1e-9,
    };

    /// The GLMM Laplace route's settings, over `[θ | β̃]` and a free Gamma
    /// `ln φ`: the measured prototype's (308 grid and 32 stress cells,
    /// 2026-10-06/07), whose largest iteration count was about 90 (q8 cells).
    ///
    /// `fdelta_abs` is 1e-10, not the LMM's 1e-9. Where a variance column
    /// collapses toward zero the deviance is nearly flat along a curved
    /// valley, and the search descends it in waves: the accepted decrease
    /// dips for a few iterations and grows again. At 1e-9 a dip ended a
    /// Gamma-inverse q6 fit 2.4e-5 above its optimum (cold, decreases down to
    /// 8.9e-10) and 9.6e-6 above (truth start, predicted decrease below the
    /// floor), with the decrement at 9e-10 and 3e-10; the joint polish then
    /// took 13078 and 9927 evaluations. At 1e-10 both reach the optimum
    /// (2026-10-09, `gami_q6_g3000p20_bal_base`).
    pub(crate) const GLMM: Self = Self {
        max_iter: 200,
        max_backtracks: 30,
        armijo_c: 1e-4,
        pgrad_tol: 1e-6,
        first_step_cap: 0.5,
        fdelta_abs: 1e-10,
    };
}

/// Why a search stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BfgsStop {
    /// The accepted decrease fell below the `Fdelta` threshold.
    Fdelta,
    /// The projected gradient fell below `pgrad_tol`.
    Pgrad,
    /// No trial was accepted within `max_backtracks` halvings.
    LineSearch,
    /// The iteration cap was reached.
    MaxIter,
    /// The gradient was unavailable or non-finite at an accepted point.
    GradUnavailable,
    /// The value at the start point was not finite.
    NonFinite,
    /// No trial could be accepted, but every trial's predicted decrease was
    /// below the `Fdelta` threshold and the decrement is at most
    /// [`CERT_DECREMENT`]: the incumbent sits at the deviance's noise floor
    /// (a warm start at the optimum ends here).
    NoiseFloor,
}

/// What a search reports. `x` (the caller's buffer) holds the incumbent: the
/// last accepted point, or the start when nothing was accepted.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BfgsOutcome {
    /// Objective at the incumbent (`+∞` on `NonFinite`).
    pub f: f64,
    /// Value evaluations, the start and every line-search trial.
    pub n_val: usize,
    /// Gradient evaluations.
    pub n_grad: usize,
    /// Iterations started.
    pub iters: usize,
    /// Why it stopped.
    pub stop: BfgsStop,
    /// `½·gᵀHg` over the free set at the incumbent (deviance units), the
    /// quasi-Newton estimate of the decrease left.
    pub decrement: f64,
}

impl BfgsOutcome {
    /// Whether the search stopped on a convergence test (`Fdelta`, `Pgrad`
    /// or `NoiseFloor`) rather than a failure.
    pub(crate) fn converged(&self) -> bool {
        matches!(
            self.stop,
            BfgsStop::Fdelta | BfgsStop::Pgrad | BfgsStop::NoiseFloor
        )
    }

    /// The endpoint certificate: the search stopped on a convergence test
    /// ([`Self::converged`], each reached with a successful gradient), the
    /// decrement is at most [`CERT_DECREMENT`], and no diagonal coordinate
    /// (`diagonal` indexes `x`) ended with `|x| <` [`CERT_NEAR_ZERO`].
    pub(crate) fn certified(&self, x: &[f64], diagonal: &[usize]) -> bool {
        self.converged()
            && self.decrement <= CERT_DECREMENT
            && diagonal.iter().all(|&i| x[i].abs() >= CERT_NEAR_ZERO)
    }
}

/// Workspace-owned search buffers. Sized on the first search of a dimension,
/// then reused; every search starts from the identity, so nothing carries
/// from one fit to the next.
#[derive(Debug, Default)]
pub(crate) struct BfgsState {
    n: usize,
    /// Row-major `n×n` inverse-Hessian approximation.
    h: Vec<f64>,
    g: Vec<f64>,
    g_new: Vec<f64>,
    d: Vec<f64>,
    s: Vec<f64>,
    y: Vec<f64>,
    hy: Vec<f64>,
    xt: Vec<f64>,
    free: Vec<bool>,
    /// The start's inverse curvature per coordinate, from
    /// [`GradObjective::start_curvature`]: positive on a seeded coordinate,
    /// 0 on every other one.
    seed: Vec<f64>,
}

impl BfgsState {
    /// The last search's `n×n` inverse-Hessian approximation (row-major), as
    /// it ended: the start (the identity, with its seeded entries) until a
    /// first update.
    pub(crate) fn inv_hessian(&self) -> &[f64] {
        &self.h
    }

    /// Sizes the buffers for `n` coordinates (allocating only when `n`
    /// changes), clears the seeds and sets `H = I`.
    fn reset(&mut self, n: usize) {
        if self.n != n {
            self.n = n;
            self.h = vec![0.0; n * n];
            self.g = vec![0.0; n];
            self.g_new = vec![0.0; n];
            self.d = vec![0.0; n];
            self.s = vec![0.0; n];
            self.y = vec![0.0; n];
            self.hy = vec![0.0; n];
            self.xt = vec![0.0; n];
            self.free = vec![true; n];
            self.seed = vec![0.0; n];
        }
        self.seed.fill(0.0);
        self.set_identity(1.0);
    }

    fn set_identity(&mut self, scale: f64) {
        let n = self.n;
        self.h.fill(0.0);
        for i in 0..n {
            self.h[i * n + i] = scale;
        }
    }

    /// `H = I` with each seeded diagonal entry at its own value.
    fn set_start(&mut self) {
        self.set_identity(1.0);
        let n = self.n;
        for i in (0..n).filter(|&i| self.seed[i] > 0.0) {
            self.h[i * n + i] = self.seed[i];
        }
    }

    /// Shanno's rescaling at the first update: `H` diagonal, each seeded entry
    /// at its own value, each contiguous run of unseeded coordinates at its
    /// own `sᵀy/yᵀy` over the run (see the module doc). A run's pair carries
    /// no positive curvature when its step or gradient change is zero (every
    /// coordinate of it held on a bound) or the step crossed a concave
    /// stretch in it; its ratio is then no scale, and it takes `fallback`
    /// (see the caller).
    fn set_shanno(&mut self, fallback: f64) {
        let n = self.n;
        self.h.fill(0.0);
        let mut i = 0;
        while i < n {
            if self.seed[i] > 0.0 {
                self.h[i * n + i] = self.seed[i];
                i += 1;
                continue;
            }
            let end = (i..n).find(|&j| self.seed[j] > 0.0).unwrap_or(n);
            let sy: f64 = (i..end).map(|j| self.s[j] * self.y[j]).sum();
            let yy: f64 = (i..end).map(|j| self.y[j] * self.y[j]).sum();
            let scale = if sy > 0.0 && yy > 0.0 {
                sy / yy
            } else {
                fallback
            };
            for j in i..end {
                self.h[j * n + j] = scale;
            }
            i = end;
        }
    }

    /// The free set at `x` for the gradient in `self.g`.
    fn mark_free(&mut self, x: &[f64], lo: &[f64], hi: &[f64]) {
        for (i, f) in self.free.iter_mut().enumerate() {
            let g = self.g[i];
            *f = !((x[i] <= lo[i] && g > 0.0) || (x[i] >= hi[i] && g < 0.0));
        }
    }

    /// `d = −H_FF g_F` (0 off the free set); returns `dᵀg`.
    #[allow(clippy::needless_range_loop)] // dense small-matrix loops index several arrays
    fn direction(&mut self) -> f64 {
        let n = self.n;
        let mut slope = 0.0;
        for i in 0..n {
            let mut acc = 0.0;
            if self.free[i] {
                let row = &self.h[i * n..(i + 1) * n];
                for j in 0..n {
                    if self.free[j] {
                        acc -= row[j] * self.g[j];
                    }
                }
            }
            self.d[i] = acc;
            slope += acc * self.g[i];
        }
        slope
    }

    /// `½·g_Fᵀ H_FF g_F`.
    fn decrement(&self) -> f64 {
        let n = self.n;
        let mut acc = 0.0;
        for i in (0..n).filter(|&i| self.free[i]) {
            let row = &self.h[i * n..(i + 1) * n];
            for j in (0..n).filter(|&j| self.free[j]) {
                acc += self.g[i] * row[j] * self.g[j];
            }
        }
        0.5 * acc
    }

    /// The BFGS inverse update with `s`, `y` (`ρ = 1/sᵀy`):
    /// `H ← H − ρ(H y sᵀ + s yᵀ H) + (ρ² yᵀHy + ρ) s sᵀ`.
    fn update(&mut self, sy: f64) {
        let n = self.n;
        let rho = 1.0 / sy;
        for i in 0..n {
            let row = &self.h[i * n..(i + 1) * n];
            self.hy[i] = row.iter().zip(&self.y).map(|(h, y)| h * y).sum();
        }
        let yhy: f64 = self.y.iter().zip(&self.hy).map(|(y, hy)| y * hy).sum();
        let ss = rho * rho * yhy + rho;
        for i in 0..n {
            for j in 0..n {
                self.h[i * n + j] += -rho * (self.hy[i] * self.s[j] + self.s[i] * self.hy[j])
                    + ss * self.s[i] * self.s[j];
            }
        }
    }
}

fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Projected BFGS from `x` within `[lo, hi]`; on return `x` holds the
/// incumbent. Evaluates at full tolerance (`resolution` 0) throughout.
/// `h0`, a row-major `n×n` inverse Hessian (a caller's
/// [`CurvatureHint`](super::CurvatureHint)), replaces the identity start: no
/// first-step cap and no Shanno rescaling then, since the scale is known.
/// Without `h0` the objective's [`GradObjective::start_curvature`] seeds
/// diagonal entries (see the module doc).
#[allow(clippy::needless_range_loop)] // dense small-matrix loops index several arrays
pub(crate) fn bfgs_minimize(
    obj: &mut impl GradObjective,
    x: &mut [f64],
    lo: &[f64],
    hi: &[f64],
    cfg: &BfgsConfig,
    st: &mut BfgsState,
    h0: Option<&[f64]>,
) -> BfgsOutcome {
    let n = x.len();
    debug_assert!(lo.len() == n && hi.len() == n);
    st.reset(n);
    let hinted = match h0 {
        Some(h) if h.len() == n * n => {
            st.h.copy_from_slice(h);
            true
        }
        _ => false,
    };
    let mut out = BfgsOutcome {
        f: f64::INFINITY,
        n_val: 1,
        n_grad: 0,
        iters: 0,
        stop: BfgsStop::MaxIter,
        decrement: f64::INFINITY,
    };
    let mut fk = match obj.value(x, 0.0) {
        Some(v) if v.is_finite() => v,
        _ => {
            out.stop = BfgsStop::NonFinite;
            return out;
        }
    };
    out.f = fk;
    out.n_grad = 1;
    if !obj.gradient(&mut st.g) || !all_finite(&st.g) {
        out.stop = BfgsStop::GradUnavailable;
        return out;
    }
    let pg0 = (0..n)
        .map(|i| ((x[i] - st.g[i]).clamp(lo[i], hi[i]) - x[i]).abs())
        .fold(0.0f64, f64::max);
    if !hinted {
        obj.start_curvature(&mut st.seed);
        for v in st.seed.iter_mut() {
            if !(v.is_finite() && *v > 0.0) {
                *v = 0.0;
            }
        }
    }
    let any_seeded = st.seed.iter().any(|&v| v > 0.0);
    if any_seeded {
        st.set_start();
    }
    if pg0 < cfg.pgrad_tol {
        out.stop = BfgsStop::Pgrad;
        st.mark_free(x, lo, hi);
        out.decrement = st.decrement();
        return out;
    }
    let mut first = !hinted;
    while out.iters < cfg.max_iter {
        out.iters += 1;
        st.mark_free(x, lo, hi);
        let mut slope = st.direction();
        if slope >= 0.0 || !slope.is_finite() {
            // Not a descent direction: restart from steepest descent.
            st.set_start();
            first = true;
            slope = st.direction();
            if slope >= 0.0 || !slope.is_finite() {
                // The projected gradient vanishes on the free set.
                out.stop = BfgsStop::Pgrad;
                break;
            }
        }
        if first {
            let dn = (0..n)
                .filter(|&i| st.seed[i] == 0.0)
                .fold(0.0f64, |m, i| m.max(st.d[i].abs()));
            if dn > cfg.first_step_cap {
                let sc = cfg.first_step_cap / dn;
                if !any_seeded {
                    st.d.iter_mut().for_each(|v| *v *= sc);
                    slope *= sc;
                } else {
                    // `H` is diagonal after a (re)start, so the partly scaled
                    // step is still a descent direction.
                    for i in (0..n).filter(|&i| st.seed[i] == 0.0) {
                        st.d[i] *= sc;
                    }
                    slope = (0..n).map(|i| st.d[i] * st.g[i]).sum();
                }
            }
        }
        // A trial whose predicted decrease is below the deviance's noise
        // cannot be told from the incumbent; the search stops at the floor
        // rather than halving further.
        let floor = cfg.fdelta_abs.max(FDELTA_ULPS * f64::EPSILON * fk.abs());
        let mut alpha = 1.0;
        let mut accepted = None;
        for _ in 0..cfg.max_backtracks {
            if alpha * -slope < floor {
                break;
            }
            for i in 0..n {
                st.xt[i] = (x[i] + alpha * st.d[i]).clamp(lo[i], hi[i]);
            }
            out.n_val += 1;
            if let Some(ft) = obj.value(&st.xt, 0.0).filter(|v| v.is_finite()) {
                let pred: f64 = (0..n).map(|i| (st.xt[i] - x[i]) * st.g[i]).sum();
                if ft <= fk + cfg.armijo_c * pred.min(alpha * slope).min(0.0) {
                    accepted = Some(ft);
                    break;
                }
            }
            alpha *= 0.5;
        }
        let Some(ft) = accepted else {
            st.mark_free(x, lo, hi);
            out.stop = if alpha * -slope < floor && st.decrement() <= CERT_DECREMENT {
                BfgsStop::NoiseFloor
            } else {
                BfgsStop::LineSearch
            };
            break;
        };
        // The accepted trial was the last value call: its gradient is next.
        out.n_grad += 1;
        let grad_ok = obj.gradient(&mut st.g_new) && all_finite(&st.g_new);
        for i in 0..n {
            st.s[i] = st.xt[i] - x[i];
            st.y[i] = st.g_new[i] - st.g[i];
        }
        x.copy_from_slice(&st.xt);
        let decrease = fk - ft;
        fk = ft;
        out.f = fk;
        if !grad_ok {
            out.stop = BfgsStop::GradUnavailable;
            break;
        }
        std::mem::swap(&mut st.g, &mut st.g_new);
        let sy: f64 = st.s.iter().zip(&st.y).map(|(s, y)| s * y).sum();
        let sn = st.s.iter().map(|v| v * v).sum::<f64>().sqrt();
        let yy: f64 = st.y.iter().map(|v| v * v).sum();
        if sy > 1e-12 * sn * yy.sqrt() {
            let mut update = true;
            if first {
                // Shanno's scale reads the unseeded coordinates only. Read
                // over all of them, a stiff seeded coordinate (a Gamma `ψ`,
                // curvature of order n, or β̃ at about 2·Σw) sets the scale
                // of the rest too: a Gamma-inverse random-slope fit then
                // ended certified in a local minimum 0.029 above the optimum
                // (`gamma_inverse_gradient_route_matches_bobyqa_in_any_units`,
                // y·2⁻⁵, 2026-10-09). So a run whose own pair carries no
                // positive curvature takes the ratio over every unseeded
                // coordinate. When that pair moved but carries no positive
                // curvature (the step crossed a concave stretch), the full
                // ratio would be the seeded coordinates' and shrink the rest
                // to their units, so the update is skipped and the next step
                // starts again from the start matrix under the first-step
                // cap: on 144 held-out fits this took a Gamma-inverse random
                // slope from 67 to 28 evaluations (2026-10-21). Without a seed
                // the unseeded pair is the full pair, positive by the check
                // above, so the skip never fires. Only when the unseeded pair
                // did not move (every unseeded coordinate held on a bound, or
                // none left) does the full `sᵀy/yᵀy` set the scale.
                let unseeded = (0..n).filter(|&i| st.seed[i] == 0.0);
                let sy_u: f64 = unseeded.clone().map(|i| st.s[i] * st.y[i]).sum();
                let yy_u: f64 = unseeded.map(|i| st.y[i] * st.y[i]).sum();
                if yy_u > 0.0 && sy_u <= 0.0 {
                    update = false;
                } else {
                    let fallback = if yy_u > 0.0 { sy_u / yy_u } else { sy / yy };
                    st.set_shanno(fallback);
                    first = false;
                }
            }
            if update {
                st.update(sy);
            }
        }
        let pg = (0..n)
            .map(|i| ((x[i] - st.g[i]).clamp(lo[i], hi[i]) - x[i]).abs())
            .fold(0.0f64, f64::max);
        if pg < cfg.pgrad_tol {
            out.stop = BfgsStop::Pgrad;
            break;
        }
        let floor = cfg.fdelta_abs.max(FDELTA_ULPS * f64::EPSILON * fk.abs());
        if out.iters > 2 && decrease.abs() < floor {
            out.stop = BfgsStop::Fdelta;
            break;
        }
    }
    if !matches!(out.stop, BfgsStop::GradUnavailable | BfgsStop::NonFinite) {
        st.mark_free(x, lo, hi);
        out.decrement = st.decrement();
    }
    out
}
