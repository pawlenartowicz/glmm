//! The outer optimizers behind every θ search, and the one contract they share.
//!
//! A fit path hands its objective to this layer through [`Objective`] (values
//! only) or [`GradObjective`] (values plus the gradient at the last value
//! point), and the layer runs the optimizer the route's rule picked
//! ([`OuterOptimizer`]). Two members today:
//!
//! - **BOBYQA** (`bobyqa` crate), derivative-free, the floor every route falls
//!   back to. [`bobyqa_minimize`] runs it on an [`Objective`] and hands the
//!   objective BOBYQA's trust radius ρ as its `resolution`, so an inner solve
//!   can tie its tolerance to it. Bit for bit the same search as calling
//!   `Bobyqa::minimize` with the objective's values (`minimize_with_radius`'s
//!   documented guarantee).
//! - **Projected BFGS** ([`bfgs::bfgs_minimize`]), for routes with an analytic
//!   gradient. It evaluates at full tolerance (`resolution` 0) and takes the
//!   gradient from the state the last value evaluation left, so an accepted
//!   point costs one value plus the gradient's own work, never a second inner
//!   solve.
//!
//! A further optimizer adds an [`OuterOptimizer`] member, a rule arm on the
//! route that uses it, and a diagnostics value; the fit paths keep calling this
//! layer. A 1-D method (a bracket search for a single θ) needs only
//! [`Objective`], and must report its bracket width as `resolution` so an
//! inner solve can be as loose as BOBYQA's early evaluations are: a 1-D
//! search at full tolerance was measured slower end to end than BOBYQA on
//! 1-θ GLMMs despite fewer evaluations (2026-10-07).
//!
//! Every search here is a pure function of its inputs: the state a search
//! keeps ([`bfgs::BfgsState`]) is workspace-owned scratch, reset at the start
//! of every search, so a refit on a reused workspace reproduces a fresh one
//! bit for bit. The one curvature input a caller can give, a
//! [`CurvatureHint`], is an explicit argument the fit only reads: a fit is a
//! pure function of (data, start, hint), in any call order.

pub(crate) mod bfgs;
#[cfg(test)]
mod tests;

use bobyqa::{Bobyqa, Outcome};

/// The value objective every outer optimizer calls.
pub(crate) trait Objective {
    /// The objective at `x`. `resolution` is the optimizer's current step
    /// resolution (BOBYQA's trust radius ρ, a bracket width for a 1-D method),
    /// so an inner solve can tie its tolerance to it; 0 asks for full
    /// tolerance. `None` or a non-finite value is a failed trial.
    fn value(&mut self, x: &[f64], resolution: f64) -> Option<f64>;
}

/// What a gradient search may call in addition to [`Objective::value`].
pub(crate) trait GradObjective: Objective {
    /// The gradient at the point of the LAST [`Objective::value`] call, read
    /// off the state that call left (no second inner solve). The driver calls
    /// it only right after a full-tolerance `value` call at the point it
    /// accepts, and only when that value was finite. `false` when the gradient
    /// is unavailable there, which ends the search (the route hands over to
    /// BOBYQA).
    fn gradient(&mut self, g: &mut [f64]) -> bool;

    /// The coordinates whose curvature the objective knows at the start: it
    /// writes each one's inverse curvature into its entry of `inv_curv`
    /// (length n, all 0 on entry), and the driver seeds those diagonal entries
    /// of its start inverse Hessian with them (see [`bfgs::bfgs_minimize`]).
    /// An entry left at 0, or not finite and positive, is not seeded. Asked
    /// once, right after the start's value and gradient, and only when no
    /// caller's inverse Hessian is given. The default seeds nothing: the
    /// search starts from the identity.
    fn start_curvature(&self, _inv_curv: &mut [f64]) {}
}

/// A frozen curvature estimate for the gradient search, taken once from a
/// pilot fit and passed with every draw of a simulation loop
/// (`loop_advanced`: `curvature_hint`, then `fit_on_hinted`).
///
/// It holds an inverse Hessian of the deviance at the pilot's optimum, in the
/// coordinates the gradient search runs in:
///
/// - LMM: the inverse of a finite-difference Hessian at θ̂ (forward
///   differences of the analytic gradient, symmetrized), over θ;
/// - GLMM (binomial, Poisson and Gamma Laplace): the pilot search's own
///   final BFGS inverse-Hessian estimate, over `[θ | β̃]` with β whitened
///   (`BetaScale`), and `[θ | β̃ | ln φ]` on a free Gamma dispersion, so it
///   costs nothing to take.
///
/// The gradient search starts from it instead of the identity, so a draw near
/// the pilot's optimum needs fewer iterations: on 20-draw loops from the truth
/// warm start (2026-10-07), BFGS search time against BOBYQA fell from 0.50 to
/// 0.25 (geometric mean over 10 LMM cells, 3 to 36 θ) and from 0.37 to 0.23
/// with the pilot's BFGS estimate on 8 GLMM cells. The fit never writes it,
/// so results do not depend on call order; a hint whose dimension does not
/// match the fit's search is rejected. A GLMM hint also carries the response
/// units its search ran in, and a fit whose first search runs in other units
/// ignores it.
#[derive(Clone, Debug)]
pub struct CurvatureHint {
    n: usize,
    inv_hessian: Vec<f64>,
    /// The response units `c` of the GLMM search the hint was taken in (the
    /// kernel fits `y/c` on Gamma's inverse link,
    /// `fit::glmm::fit_glmm_in_units`); 1 everywhere else.
    pub(crate) units: f64,
}

impl CurvatureHint {
    /// The dimension of the search the hint was taken in: n_θ for an LMM;
    /// n_θ + p for a GLMM, + 1 on a Gamma fit whose dispersion is free.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Row-major `n×n` inverse Hessian.
    pub fn inv_hessian(&self) -> &[f64] {
        &self.inv_hessian
    }

    /// The response units the hint was taken in: 1 everywhere except a Gamma
    /// inverse-link GLMM.
    pub fn units(&self) -> f64 {
        self.units
    }

    /// A hint rebuilt from its parts, so it can cross a process or WASM
    /// boundary as plain numbers: `from_parts(h.dim(), h.inv_hessian(),
    /// h.units())` gives back `h` bit for bit. `inv_hessian` is stored as
    /// given, not re-symmetrized, since a hint from the LMM finite-difference
    /// path need not be bit-symmetric. `None` on a length other than `n·n`, a
    /// value that is not finite, `units` not finite and positive, or a matrix
    /// that is not positive definite.
    pub fn from_parts(n: usize, inv_hessian: &[f64], units: f64) -> Option<Self> {
        if inv_hessian.len() != n * n
            || inv_hessian.iter().any(|v| !v.is_finite())
            || !(units.is_finite() && units > 0.0)
        {
            return None;
        }
        spd_inverse(n, inv_hessian)?;
        Some(Self {
            n,
            inv_hessian: inv_hessian.to_vec(),
            units,
        })
    }

    /// A hint from an `n×n` row-major inverse Hessian already in hand (the
    /// GLMM search's final BFGS estimate, in that search's `units`),
    /// symmetrized; `None` unless [`Self::from_parts`] accepts it.
    pub(crate) fn from_inverse(n: usize, inv_hessian: &[f64], units: f64) -> Option<Self> {
        if inv_hessian.len() != n * n {
            return None;
        }
        let mut h = inv_hessian.to_vec();
        for i in 0..n {
            for j in 0..i {
                let a = 0.5 * (h[i * n + j] + h[j * n + i]);
                h[i * n + j] = a;
                h[j * n + i] = a;
            }
        }
        Self::from_parts(n, &h, units)
    }

    /// The hint at `x` within `[lo, hi]` from `grad` (the gradient at a point,
    /// `false` when unavailable): `n + 1` gradients, forward steps of
    /// `1e-4·max(1, |x_j|)` (backward at the upper bound). `None` when a
    /// gradient fails or the symmetrized Hessian is not positive definite.
    pub(crate) fn from_gradients(
        x: &[f64],
        lo: &[f64],
        hi: &[f64],
        mut grad: impl FnMut(&[f64], &mut [f64]) -> bool,
    ) -> Option<Self> {
        let n = x.len();
        let mut g0 = vec![0.0; n];
        if !grad(x, &mut g0) || g0.iter().any(|v| !v.is_finite()) {
            return None;
        }
        let mut hm = vec![0.0; n * n];
        let mut xt = x.to_vec();
        let mut gt = vec![0.0; n];
        for j in 0..n {
            let mut h = 1e-4 * x[j].abs().max(1.0);
            if x[j] + h > hi[j] {
                h = -h;
            }
            if x[j] + h < lo[j] {
                return None;
            }
            xt[j] = x[j] + h;
            let ok = grad(&xt, &mut gt);
            xt[j] = x[j];
            if !ok || gt.iter().any(|v| !v.is_finite()) {
                return None;
            }
            for i in 0..n {
                hm[i * n + j] = (gt[i] - g0[i]) / h;
            }
        }
        for i in 0..n {
            for j in 0..i {
                let a = 0.5 * (hm[i * n + j] + hm[j * n + i]);
                hm[i * n + j] = a;
                hm[j * n + i] = a;
            }
        }
        spd_inverse(n, &hm).map(|inv_hessian| Self {
            n,
            inv_hessian,
            units: 1.0,
        })
    }
}

/// Inverse of a symmetric `n×n` matrix (row-major) through its Cholesky
/// factor; `None` when a pivot is not clearly positive.
fn spd_inverse(n: usize, a: &[f64]) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        if d.is_nan() || d <= 1e-14 * a[j * n + j].abs().max(1e-300) {
            return None;
        }
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = v / ljj;
        }
    }
    // inv = L⁻ᵀ L⁻¹, one column at a time.
    let mut inv = vec![0.0; n * n];
    let mut col = vec![0.0; n];
    for c in 0..n {
        for i in 0..n {
            let mut v = if i == c { 1.0 } else { 0.0 };
            for k in 0..i {
                v -= l[i * n + k] * col[k];
            }
            col[i] = v / l[i * n + i];
        }
        for i in (0..n).rev() {
            let mut v = col[i];
            for k in (i + 1)..n {
                v -= l[k * n + i] * inv[k * n + c];
            }
            inv[i * n + c] = v / l[i * n + i];
        }
    }
    Some(inv)
}

/// A plain value closure `f(x, resolution)` as an [`Objective`] that never
/// fails a trial: every value reaches the optimizer as it is, a non-finite
/// one included, so [`bobyqa_minimize`] on it runs exactly the search
/// `Bobyqa::minimize` runs on the closure's values. The GLMM routes, whose
/// objectives are closures over a destructured workspace, go through it.
pub(crate) struct ValueFn<F>(pub F);

impl<F: FnMut(&[f64], f64) -> f64> Objective for ValueFn<F> {
    fn value(&mut self, x: &[f64], resolution: f64) -> Option<f64> {
        Some((self.0)(x, resolution))
    }
}

/// Which outer optimizer a fit's θ search ran, chosen by the route's rule.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OuterOptimizer {
    /// Derivative-free BOBYQA (the default, and every fallback).
    Bobyqa,
    /// Projected BFGS on an analytic gradient ([`bfgs::bfgs_minimize`]).
    Bfgs,
}

/// The optimizer a route's θ search runs on a cold fit and on a warm one,
/// fixed per workspace from the shape alone (the GLMM gradient route). A
/// fit reads the member for its start kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoutePlan {
    /// A fit from the blind start.
    pub cold: OuterOptimizer,
    /// A fit from a caller's start (the start guard runs).
    pub warm: OuterOptimizer,
}

impl RoutePlan {
    /// BOBYQA on both start kinds: every shape outside a gradient rule.
    pub(crate) const BOBYQA: Self = Self {
        cold: OuterOptimizer::Bobyqa,
        warm: OuterOptimizer::Bobyqa,
    };
}

/// Runs `solver` on `obj` from `x` within `[lower, upper]`, handing the
/// objective the trust radius ρ as its resolution. A `None` from the
/// objective reaches BOBYQA as `+∞`, its failed-trial value; a non-finite
/// value passes through as it is, so an objective that always answers
/// `Some(d)` gets exactly the search `Bobyqa::minimize` would run on `d`.
pub(crate) fn bobyqa_minimize(
    solver: &mut Bobyqa,
    obj: &mut impl Objective,
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
) -> Outcome {
    solver.minimize_with_radius(
        |xs, radius| obj.value(xs, radius.rho).unwrap_or(f64::INFINITY),
        x,
        lower,
        upper,
    )
}
