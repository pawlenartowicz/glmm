//! The objectives of the profile-likelihood intervals of a Gaussian LMM: the
//! ML deviance with one parameter held, on lme4's scale, over a workspace an
//! ML fit left at θ̂.
//!
//! **Variance parameters.** The coordinates are, per random-effect block in θ
//! order, the column-major `vech` entries that are not structural zeros: an SD
//! on the diagonal, a correlation below it (lme4's θ order within a term); then
//! the residual SD σ. All are in the caller's units. A vector `v` maps to the
//! kernel's θ̃ through the relative covariance: with `s` the prior-weight scale
//! the kernel runs under ([`LmmWorkspace::weight_scale`]), the kernel's
//! residual variance is `σ_int² = σ²/s`, and each block's `Σ = D·R·D` divided
//! by `σ_int²` is factored by [`semidef_lower_chol`] (trace floor). A factor
//! that does not reproduce that matrix to `1e-8·tr` means `R` is not positive
//! semidefinite, and the point is infeasible. The factor, packed as the
//! caller's θ, goes onto the fitted design through
//! [`LmmGroupings::theta_caller_to_design`], structural entries zeroed. With
//! `d_p = kernel.deviance(θ̃)` (ML, σ profiled), `σ̂² = kernel.sigma_sq()` and
//! `N` the kernel's rows, the ML deviance at the given σ is
//! `d_p + N·(ln(σ_int²/σ̂²) + σ̂²/σ_int² − 1)`, from
//! `ln|L|² + N·ln σ² + r²/σ²` with `r² = N·σ̂²`.
//!
//! **Coefficients.** At any θ̃ the penalized residual sum of squares is
//! `r² + (β − β̂)ᵀA(β − β̂)`, so with `β_j = b` and the other coefficients free
//! its minimum is `r² + (b − β̂_j)²/(A⁻¹)_jj`. Both come off the unmapped
//! factor the evaluation leaves, as the Satterthwaite pass reads `C`: row `j`
//! of `M = B·U⁻¹` ([`caller_map_into`]) gives `β̂_j = M_j·β̃` (β̃ by the
//! backward solve on `X̃`) and `Var_j = σ̂²·‖L̃⁻¹M_jᵀ‖²`. The ML deviance
//! profiled over σ and `β_{−j}` is then `d_p + N·ln(1 + (b − β̂_j)²/(N·Var_j))`.
//! A coefficient needs no refit with its column dropped, which the kernel could
//! not run on an intercept-only model (it refuses `p == 0`).
//!
//! **Held minimizations.** Both run BOBYQA ([`bobyqa_minimize`]) with the
//! workspace's own schedule from its warm radius ([`warm_bobyqa`]), on as many
//! coordinates as the fit's free θ, so the workspace's warm solver serves both.
//! A coefficient's search moves θ̃ in the fit's box, structural entries held
//! at zero. A variance parameter's search moves the other coordinates in
//! scaled units `u = v/c`: `c` is the estimate for a positive SD and for σ,
//! `σ̂` for an SD estimated at 0, and 1 for a correlation. An SD's box is
//! `u ∈ [0, THETA_HI·max(1, σ̂_int/(s·c))]`: in caller units at least the fit's
//! own box for that row, `THETA_HI·σ̂_int/s` (`σ̂_int` the kernel's residual SD,
//! the scale `Fit::varcorr` turns θ̃ into SDs by, `s` the standardization
//! scale of the row's column, 1 on an intercept), so a small SD the fit did not
//! pin can still grow while another parameter is held, and never narrower than
//! `[0, THETA_HI]` in `u`, so the estimate is inside it. σ's box is
//! `[1/THETA_HI, THETA_HI]` in scaled units and a correlation's `[−1, 1]`.
//!
//! Every evaluation overwrites the factor and `σ̂²` the kernel holds; the
//! workspace's θ̂ (`LmmWorkspace::theta`) is never written.

use bobyqa::Status;
use faer::Mat;

use super::re_map::{pack_vech, semidef_lower_chol, PivotFloor};
use super::satterthwaite::caller_map_into;
use super::{warm_bobyqa, Criterion, LmmGroupings, LmmKernel, LmmWorkspace, THETA_HI};
use crate::fit::varcorr_block;
use crate::optim::{bobyqa_minimize, ValueFn};
use crate::standardize::ColumnScaling;

/// One lme4-scale variance coordinate. `block` is the grouping in kernel slot
/// order (see `Perm`): the kernel's primary first, then its extras; `row` and
/// `col` index the block's columns in θ's layout (a block without an intercept
/// keeps its idle column 0); `theta_index` is the entry's slot in θ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum VarCoord {
    /// The SD of column `row`.
    Sd {
        block: usize,
        row: usize,
        theta_index: usize,
    },
    /// The correlation of columns `row > col`.
    Cor {
        block: usize,
        row: usize,
        col: usize,
        theta_index: usize,
    },
    /// The residual SD σ.
    Sigma,
}

/// The coordinates and where they sit in θ.
struct Layout {
    coords: Vec<VarCoord>,
    /// Per θ entry, its coordinate in `coords`; `None` at a structural entry.
    slot: Vec<Option<usize>>,
    /// Block widths `q`, θ order.
    blocks: Vec<usize>,
    /// `s` of `σ_int² = σ²/s`.
    weight_scale: f64,
}

/// The buffers of one map from `v` to θ̃: a block's matrix and its factor
/// (`q_max²` each) and the random-effect maps' scratch.
struct CholScratch {
    sig: Vec<f64>,
    lam: Vec<f64>,
    remap: Vec<f64>,
}

impl CholScratch {
    fn new(lay: &Layout, g: &LmmGroupings) -> Self {
        let q = lay.blocks.iter().copied().max().unwrap_or(1);
        CholScratch {
            sig: vec![0.0; q * q],
            lam: vec![0.0; q * q],
            remap: vec![0.0; g.re_map_scratch_len()],
        }
    }
}

/// Offset of entry `(r, c)`, `r ≥ c`, in a `q`-wide block's column-major vech.
fn vech_at(q: usize, r: usize, c: usize) -> usize {
    c * q - (c * c - c) / 2 + (r - c)
}

/// The profile objectives over a workspace an ML fit just left at θ̂.
pub(crate) struct ProfileKernel<'a> {
    ws: &'a mut LmmWorkspace,
    map: &'a ColumnScaling,
    lay: Layout,
    /// The lme4-scale estimate, the units `c` of the scaled search and its box.
    est: Vec<f64>,
    unit: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// θ̂, the fit's box and its structural entries.
    theta_hat: Vec<f64>,
    theta_lower: Vec<f64>,
    theta_upper: Vec<f64>,
    structural: Vec<usize>,
    /// The ML deviance at θ̂ (kernel scale), and the kernel's rows `N`.
    dev_hat: f64,
    n: f64,
    /// `M = B·U⁻¹` (`p × p`).
    m: Mat<f64>,
    /// θ̃ of the point being evaluated, the variance search's full point in
    /// caller and in scaled units, β̃ and `L̃⁻¹M_jᵀ`.
    theta_buf: Vec<f64>,
    v: Vec<f64>,
    u: Vec<f64>,
    beta: Vec<f64>,
    w: Vec<f64>,
    chol: CholScratch,
}

impl<'a> ProfileKernel<'a> {
    /// Over the workspace an ML fit just left at θ̂; `map` is the column
    /// standardization its rows were accumulated under. Evaluates the deviance
    /// at θ̂ once.
    pub(crate) fn new(ws: &'a mut LmmWorkspace, map: &'a ColumnScaling) -> ProfileKernel<'a> {
        debug_assert_eq!(
            ws.kernel.criterion(),
            Criterion::Ml,
            "the profile reads the ML deviance"
        );
        let lay = {
            let g = ws.kernel.groupings();
            let blocks: Vec<usize> = std::iter::once(g.primary_q)
                .chain(g.extra_q.iter().copied())
                .collect();
            let structural = g.structural_theta();
            let mut coords = Vec::new();
            let mut slot = vec![None; g.n_theta()];
            let mut t = 0usize;
            for (block, &q) in blocks.iter().enumerate() {
                for col in 0..q {
                    for row in col..q {
                        if structural.binary_search(&t).is_err() {
                            slot[t] = Some(coords.len());
                            coords.push(if row == col {
                                VarCoord::Sd {
                                    block,
                                    row,
                                    theta_index: t,
                                }
                            } else {
                                VarCoord::Cor {
                                    block,
                                    row,
                                    col,
                                    theta_index: t,
                                }
                            });
                        }
                        t += 1;
                    }
                }
            }
            coords.push(VarCoord::Sigma);
            Layout {
                coords,
                slot,
                blocks,
                weight_scale: ws.weight_scale,
            }
        };

        let theta_hat = ws.theta.clone();
        let dev_hat = ws.kernel.deviance(&theta_hat);
        let sigma_int_sq = ws.kernel.sigma_sq();
        let n = ws.kernel.n_rows() as f64;
        let p = ws.recovery.betas.len();
        let mut m = Mat::zeros(p, p);
        caller_map_into(&ws.kernel, map, &mut vec![0.0; p], &mut m);

        // The estimate off θ̂ as `Fit::varcorr` and `Fit::stddev_corr` read it.
        let g = ws.kernel.groupings();
        let nc = lay.coords.len();
        let mut est = vec![0.0; nc];
        let mut cur = 0usize;
        for (b, &q) in lay.blocks.iter().enumerate() {
            let len = q * (q + 1) / 2;
            let vc = varcorr_block(&theta_hat[cur..cur + len], q, sigma_int_sq, g, b, map);
            for col in 0..q {
                for row in col..q {
                    let Some(i) = lay.slot[cur + vech_at(q, row, col)] else {
                        continue;
                    };
                    let sd = |r: usize| vc[vech_at(q, r, r)].sqrt();
                    est[i] = if row == col {
                        sd(row)
                    } else if sd(row) > 0.0 && sd(col) > 0.0 {
                        (vc[vech_at(q, row, col)] / (sd(row) * sd(col))).clamp(-1.0, 1.0)
                    } else {
                        // Not identified next to a zero SD; any value is the same Σ.
                        0.0
                    };
                }
            }
            cur += len;
        }
        let sigma_hat = (sigma_int_sq * ws.weight_scale).sqrt();
        est[nc - 1] = sigma_hat;

        // The residual scale `Fit::varcorr` turns θ̃ into caller SDs by (the
        // kernel's σ̂_int, without the weight scale).
        let sigma_unit = sigma_int_sq.sqrt();
        let (mut unit, mut lower, mut upper) = (vec![1.0; nc], vec![0.0; nc], vec![0.0; nc]);
        for (i, c) in lay.coords.iter().enumerate() {
            (unit[i], lower[i], upper[i]) = match *c {
                VarCoord::Sd { block, row, .. } => {
                    // The caller's row divides by its column's scale (1 on the
                    // intercept), so the fit's box reaches `THETA_HI·σ̂_int/s`.
                    // Never narrower than `[0, THETA_HI]` in `u`, so the start
                    // stays inside.
                    let s = g.block_std(block, row, map).1;
                    let c = if est[i] > 0.0 { est[i] } else { sigma_hat };
                    (c, 0.0, THETA_HI * (sigma_unit / (s * c)).max(1.0))
                }
                VarCoord::Cor { .. } => (1.0, -1.0, 1.0),
                VarCoord::Sigma => (sigma_hat, 1.0 / THETA_HI, THETA_HI),
            };
        }

        let chol = CholScratch::new(&lay, g);
        let structural = g.structural_theta().to_vec();
        ProfileKernel {
            theta_buf: vec![0.0; theta_hat.len()],
            v: vec![0.0; nc],
            u: vec![0.0; nc],
            beta: vec![0.0; p],
            w: vec![0.0; p],
            theta_lower: ws.lower.clone(),
            theta_upper: ws.upper.clone(),
            structural,
            chol,
            est,
            unit,
            lower,
            upper,
            theta_hat,
            dev_hat,
            n,
            m,
            lay,
            ws,
            map,
        }
    }

    /// lme4-scale coordinates: blocks in θ order, structural entries left out,
    /// then σ.
    pub(crate) fn coords(&self) -> &[VarCoord] {
        &self.lay.coords
    }

    /// The lme4-scale vector (caller units) at the fit's θ̂ and σ̂. A
    /// correlation next to a zero SD, which no Σ identifies, reads 0.
    pub(crate) fn estimate(&self) -> Vec<f64> {
        self.est.clone()
    }

    /// [`theta_into`] on the kernel's parts.
    #[cfg(test)]
    pub(crate) fn theta_of(&self, v: &[f64], theta: &mut [f64]) -> bool {
        let g = self.ws.kernel.groupings();
        let mut chol = CholScratch::new(&self.lay, g);
        theta_into(g, self.map, &self.lay, v, theta, &mut chol)
    }

    /// [`var_dev`] on the kernel's parts.
    #[cfg(test)]
    pub(crate) fn var_deviance(&mut self, v: &[f64]) -> f64 {
        var_dev(
            &mut self.ws.kernel,
            self.map,
            &self.lay,
            v,
            &mut self.theta_buf,
            &mut self.chol,
        )
    }

    /// [`beta_dev`] on the kernel's parts.
    #[cfg(test)]
    pub(crate) fn beta_deviance(&mut self, theta: &[f64], j: usize, b: f64) -> f64 {
        beta_dev(
            &mut self.ws.kernel,
            &self.m,
            self.n,
            &mut self.beta,
            &mut self.w,
            theta,
            j,
            b,
        )
    }

    /// Minimum of the ML deviance at `v` ([`var_dev`]) with coordinate `k`
    /// held at `value`. `start` is a full lme4-scale vector inside the
    /// search's box (entry `k` is ignored); it returns the argmin, with
    /// `value` at `k`. Returns `(deviance, converged)`.
    pub(crate) fn hold_var(&mut self, k: usize, value: f64, start: &mut [f64]) -> (f64, bool) {
        let nc = self.lay.coords.len();
        debug_assert_eq!(start.len(), nc);
        for ((u, &s), &c) in self.u.iter_mut().zip(start.iter()).zip(&self.unit) {
            *u = s / c;
        }
        self.u[k] = 0.0;
        let config = self.ws.config;
        let n_free = self.ws.kernel.groupings().n_free_theta();
        debug_assert_eq!(n_free + 1, nc, "the held search has the free θ's dimension");
        let mut slot = self.ws.solver_warm.take();
        let Self {
            ws,
            map,
            lay,
            unit,
            lower,
            upper,
            theta_buf,
            v,
            u,
            chol,
            ..
        } = self;
        let out = bobyqa_minimize(
            warm_bobyqa(&mut slot, &config, n_free),
            &mut ValueFn(|x: &[f64], _: f64| {
                for ((vi, &xi), &c) in v.iter_mut().zip(x).zip(unit.iter()) {
                    *vi = xi * c;
                }
                v[k] = value;
                var_dev(&mut ws.kernel, map, lay, v, theta_buf, chol)
            }),
            u,
            lower,
            upper,
            &[k],
        );
        ws.solver_warm = slot;
        for ((s, &ui), &c) in start.iter_mut().zip(u.iter()).zip(unit.iter()) {
            *s = ui * c;
        }
        start[k] = value;
        (out.f, out.status == Status::Converged && out.f.is_finite())
    }

    /// Minimum of [`beta_dev`] over θ̃ in the fit's box with `β_j = b`.
    /// `start` is a full θ̃ (structural entries ignored); it returns the
    /// argmin. Returns `(deviance, converged)`.
    pub(crate) fn hold_beta(&mut self, j: usize, b: f64, start: &mut [f64]) -> (f64, bool) {
        let config = self.ws.config;
        let n_free = self.ws.kernel.groupings().n_free_theta();
        let mut slot = self.ws.solver_warm.take();
        let Self {
            ws,
            m,
            n,
            beta,
            w,
            theta_lower,
            theta_upper,
            structural,
            ..
        } = self;
        let out = bobyqa_minimize(
            warm_bobyqa(&mut slot, &config, n_free),
            &mut ValueFn(|x: &[f64], _: f64| beta_dev(&mut ws.kernel, m, *n, beta, w, x, j, b)),
            start,
            theta_lower,
            theta_upper,
            structural,
        );
        ws.solver_warm = slot;
        (out.f, out.status == Status::Converged && out.f.is_finite())
    }

    /// θ̂, the θ̃ the fit left.
    pub(crate) fn theta_hat(&self) -> &[f64] {
        &self.theta_hat
    }

    /// The ML deviance at θ̂ (kernel scale), the base every profile is
    /// measured from.
    pub(crate) fn kernel_deviance_at_hat(&self) -> f64 {
        self.dev_hat
    }
}

/// θ̃ of `v` on the fitted design, into `theta` (θ's full layout); false
/// when a correlation block is not PSD, or `v` holds a negative or
/// non-finite SD, a correlation outside `[−1, 1]` or a σ that is not
/// positive.
fn theta_into(
    g: &LmmGroupings,
    map: &ColumnScaling,
    lay: &Layout,
    v: &[f64],
    theta: &mut [f64],
    chol: &mut CholScratch,
) -> bool {
    let sigma = v[lay.coords.len() - 1];
    if !(sigma.is_finite() && sigma > 0.0) {
        return false;
    }
    let sigma_int_sq = sigma * sigma / lay.weight_scale;
    let mut cur = 0usize;
    for &q in &lay.blocks {
        let read = |r: usize, c: usize| lay.slot[cur + vech_at(q, r, c)].map(|i| v[i]);
        let sd = |r: usize| read(r, r).unwrap_or(0.0);
        let (sig, lam) = (&mut chol.sig[..q * q], &mut chol.lam[..q * q]);
        for r in 0..q {
            let sd_r = sd(r);
            if !(sd_r.is_finite() && sd_r >= 0.0) {
                return false;
            }
            for c in 0..r {
                let rho = read(r, c).unwrap_or(0.0);
                if !(-1.0..=1.0).contains(&rho) {
                    return false;
                }
                sig[r * q + c] = sd_r * sd(c) * rho / sigma_int_sq;
            }
            sig[r * q + r] = sd_r * sd_r / sigma_int_sq;
        }
        semidef_lower_chol(sig, q, lam, PivotFloor::Trace);
        let trace: f64 = (0..q).map(|r| sig[r * q + r]).sum();
        let reproduced = (0..q).all(|r| {
            (0..=r).all(|c| {
                let s: f64 = (0..=c).map(|k| lam[r * q + k] * lam[c * q + k]).sum();
                (s - sig[r * q + c]).abs() <= 1e-8 * trace
            })
        });
        if !reproduced {
            return false;
        }
        let len = q * (q + 1) / 2;
        pack_vech(lam, q, &mut theta[cur..cur + len]);
        cur += len;
    }
    g.theta_caller_to_design(theta, &mut chol.remap, map);
    g.zero_structural(theta);
    true
}

/// ML deviance (kernel scale) at `v`, σ taken from `v`; `+∞` when `v` is
/// infeasible ([`theta_into`]) or the evaluation fails.
fn var_dev(
    kernel: &mut LmmKernel,
    map: &ColumnScaling,
    lay: &Layout,
    v: &[f64],
    theta: &mut [f64],
    chol: &mut CholScratch,
) -> f64 {
    if !theta_into(kernel.groupings(), map, lay, v, theta, chol) {
        return f64::INFINITY;
    }
    let dev = kernel.deviance(theta);
    let sigma = v[lay.coords.len() - 1];
    // `ln(σ_int²/σ̂²) + σ̂²/σ_int² − 1 = x − ln(1 + x)`, `x = σ̂²/σ_int² − 1`.
    let x = kernel.sigma_sq() * lay.weight_scale / (sigma * sigma) - 1.0;
    let d = dev + kernel.n_rows() as f64 * (x - x.ln_1p());
    if d.is_finite() {
        d
    } else {
        f64::INFINITY
    }
}

/// ML deviance at θ̃ `theta` profiled over σ and every coefficient but
/// `j`, which is held at `b` (caller units); `+∞` on failure. `beta` and `w`
/// are length-`p` scratch.
#[allow(clippy::too_many_arguments)] // the kernel, M and the two buffers, split off the profile
fn beta_dev(
    kernel: &mut LmmKernel,
    m: &Mat<f64>,
    n: f64,
    beta: &mut [f64],
    w: &mut [f64],
    theta: &[f64],
    j: usize,
    b: f64,
) -> f64 {
    let dev = kernel.deviance(theta);
    if !dev.is_finite() {
        return f64::INFINITY;
    }
    let p = beta.len();
    let f = kernel.factor();
    // β̃ by the backward solve `L̃_XXᵀβ̃ = l_yX̃` (the y row is index p).
    for i in (0..p).rev() {
        let mut acc = f[(p, i)];
        for k in (i + 1)..p {
            acc -= f[(k, i)] * beta[k];
        }
        beta[i] = acc / f[(i, i)];
    }
    let beta_j: f64 = (0..p).map(|i| m[(j, i)] * beta[i]).sum();
    // `w = L̃⁻¹M_jᵀ`.
    for i in 0..p {
        let mut acc = m[(j, i)];
        for k in 0..i {
            acc -= f[(i, k)] * w[k];
        }
        w[i] = acc / f[(i, i)];
    }
    let var_j = kernel.sigma_sq() * w.iter().map(|x| x * x).sum::<f64>();
    let d = b - beta_j;
    let out = dev + n * (d * d / (n * var_j)).ln_1p();
    if out.is_finite() {
        out
    } else {
        f64::INFINITY
    }
}
