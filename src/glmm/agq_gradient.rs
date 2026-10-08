//! Exact `f64` gradient of the scalar-intercept AGQ deviance in
//! `γ = [λ | β]`, from the state an AGQ value evaluation leaves: the converged
//! per-cluster modes `û_c` (`pirls.u`) and the fixed part of the predictor
//! `η_fix = Xβ + offset` (`pirls.eta_fixed`). Its one caller is the SE step's
//! gradient-difference Hessian (`se::agq_gradient_difference_hessian`), which
//! differences this gradient instead of carrying hyper-dual lanes per row.
//!
//! Per cluster `c`, with `ℓ_c(u) = Σ_i ℓ_i(η_fix,i + λu) − u²/2`, `ℓ_i` the
//! row log-likelihood, `û` the mode, `A = −ℓ_c''(û)`, `σ = A^{−1/2}`, the
//! nodes `u_j = û + √2·σ·z_j` and `ω_j = w_j·e^{z_j²}` (Gauss–Hermite `z_j`,
//! `w_j`; Liu & Pierce 1994, *Biometrika* 81(3)):
//!
//! ```text
//!   D_c = −2·[ln σ + ln Σ_j ω_j·e^{ℓ_c(u_j)} − ln √π]
//! ```
//!
//! `D_c` depends on `γ` directly through `ℓ_c` and through `û` and `σ`:
//!
//! ```text
//!   dD_c = −2·[d ln σ + Σ_j π_j·(∂_γ ℓ_c(u_j) + ℓ_c'(u_j)·(dû + √2·z_j·σ·d ln σ))]
//!   dû    = ∂_γ ℓ_c'(û) / A                      (implicit function, ℓ_c'(û) = 0)
//!   dA    = −(∂_γ ℓ_c''(û) + ℓ_c'''(û)·dû),   d ln σ = −½·dA / A
//! ```
//!
//! with `π_j` the normalized node terms. The `û` and `σ` terms do not cancel
//! at a finite number of nodes: the exact integral does not depend on where the
//! rule is centered, the quadrature does. The derivation holds for any link
//! whose AGQ scale is the observed curvature, which is how
//! `pirls::evaluate_at_mode` builds `A` on a non-canonical link, so probit
//! takes the same assembly with its own row terms.
//!
//! Covered ([`covers`]): binomial logit and probit, Poisson log; one scalar
//! random intercept (`q_p = 1`), no extra groupings, so no dispersion
//! coordinate. Every other AGQ shape keeps the hyper-dual rung.
//!
//! Measured 2026-10-07 (pawel scratch `mmjl-rethink/b7/`, `probit/`, `agqse/`):
//! the gradient matches central differences of the AGQ deviance to their noise
//! floor on the grid's AGQ cells and on clusters of 2, 3 and 5 rows, and costs
//! about half a value evaluation.

use faer::MatRef;

use super::agq::ClusterRowIndex;
use super::pirls::{pirls_solve_blocked, BetaStep};
use super::workspace::{GlmmLayout, GlmmWorkspace};
use crate::spec::{BinomialLink, Family, PoissonLink};

#[derive(Clone, Copy)]
enum RowFamily {
    Logit,
    PoissonLog,
    Probit,
}

fn row_family(f: Family) -> Option<RowFamily> {
    match f {
        Family::Binomial {
            link: BinomialLink::Logit,
        } => Some(RowFamily::Logit),
        Family::Binomial {
            link: BinomialLink::Probit,
        } => Some(RowFamily::Probit),
        Family::Poisson {
            link: PoissonLink::Log,
        } => Some(RowFamily::PoissonLog),
        _ => None,
    }
}

/// Whether this gradient covers the workspace's AGQ shape: `nagq > 1`, the
/// blocked layout with one scalar random intercept and no extra groupings, and
/// a family and link [`row_family`] knows.
pub(crate) fn covers(ws: &GlmmWorkspace) -> bool {
    ws.nagq > 1
        && ws.layout == GlmmLayout::Blocked
        && ws.groupings.primary_q == 1
        && ws.groupings.extra_offsets.is_empty()
        && row_family(ws.family).is_some()
}

/// `(ℓ_i, ℓ_i', ℓ_i'', ℓ_i''')` in `η` for prior weight `w`, up to a
/// `γ`-free constant that cancels in `π_j` and in `dD`.
#[inline]
fn row_terms(f: RowFamily, y: f64, w: f64, eta: f64) -> (f64, f64, f64, f64) {
    match f {
        RowFamily::Logit => {
            // ℓ = w·[yη − ln(1 + e^η)]
            let mu = 1.0 / (1.0 + (-eta).exp());
            let l1pe = if eta > 0.0 {
                eta + (-eta).exp().ln_1p()
            } else {
                eta.exp().ln_1p()
            };
            let v = mu * (1.0 - mu);
            (
                w * (y * eta - l1pe),
                w * (y - mu),
                -w * v,
                -w * v * (1.0 - 2.0 * mu),
            )
        }
        RowFamily::PoissonLog => {
            let mu = eta.exp();
            (w * (y * eta - mu), w * (y - mu), -w * mu, -w * mu)
        }
        RowFamily::Probit => {
            // ℓ = w·[y·ln Φ(η) + (1 − y)·ln Φ(−η)]. `family::probit_side(x)`
            // gives (ln Φ(x), λ(x), λ'(x), λ''(x)) with λ = φ/Φ, tail-safe, so
            // ℓ' = w·[y·λ(η) − (1 − y)·λ(−η)], ℓ'' = w·[y·λ'(η) + (1 − y)·λ'(−η)],
            // ℓ''' = w·[y·λ''(η) − (1 − y)·λ''(−η)]. A Bernoulli row reads one side.
            let (mut l, mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0, 0.0);
            if y > 0.0 {
                let (lp, a, a1, a2) = crate::family::probit_side(eta);
                l += y * lp;
                s1 += y * a;
                s2 += y * a1;
                s3 += y * a2;
            }
            if y < 1.0 {
                let (lq, b, b1, b2) = crate::family::probit_side(-eta);
                let yc = 1.0 - y;
                l += yc * lq;
                s1 -= yc * b;
                s2 += yc * b1;
                s3 -= yc * b2;
            }
            (w * l, w * s1, w * s2, w * s3)
        }
    }
}

/// What the gradient-difference Hessian keeps between SE calls: its own
/// per-cluster row index (never `ws.cluster_rows`, whose presence switches the
/// AGQ value kernel's loop order), rebuilt in place on every call because a
/// reused workspace may see new `cluster_ids`, and [`assemble`]'s buffers.
pub(crate) struct AgqGd {
    pub(crate) idx: ClusterRowIndex,
    pub(crate) sc: AgqGradScratch,
}

impl AgqGd {
    pub(crate) fn for_shape(p: usize, nagq: u8) -> Self {
        Self {
            idx: ClusterRowIndex::default(),
            sc: AgqGradScratch::new(p, nagq),
        }
    }

    pub(crate) fn matches_shape(&self, p: usize, nagq: u8) -> bool {
        self.sc.p == p && self.sc.k == nagq as usize
    }
}

/// Per-call buffers of [`assemble`], sized once for `p` fixed effects and `k`
/// nodes, so the gradient-difference Hessian's `2m` calls allocate nothing.
pub(crate) struct AgqGradScratch {
    p: usize,
    k: usize,
    /// `Σ_i ℓ_i''·x_i` and `Σ_i ℓ_i'''·x_i` at `û`, length `p`.
    x2: Vec<f64>,
    x3: Vec<f64>,
    /// `dû` and `d ln σ` per coordinate, length `1 + p`.
    du: Vec<f64>,
    dls: Vec<f64>,
    /// `Σ_i ℓ_i'(u_j)·x_i` per node, `k × p` row-major.
    xt: Vec<f64>,
    /// Per node: `ℓ_c'(u_j)` without the prior, the log node term, `u_j`.
    tj: Vec<f64>,
    lj: Vec<f64>,
    uj: Vec<f64>,
    /// `ln w_j + z_j²` per node.
    ln_omega: Vec<f64>,
}

impl AgqGradScratch {
    pub(crate) fn new(p: usize, nagq: u8) -> Self {
        let k = nagq as usize;
        let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
        let nodes = &crate::consts::GH_NODES[blk..blk + k];
        let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
        Self {
            p,
            k,
            x2: vec![0.0; p],
            x3: vec![0.0; p],
            du: vec![0.0; p + 1],
            dls: vec![0.0; p + 1],
            xt: vec![0.0; k * p],
            tj: vec![0.0; k],
            lj: vec![0.0; k],
            uj: vec![0.0; k],
            ln_omega: (0..k).map(|j| wts[j].ln() + nodes[j] * nodes[j]).collect(),
        }
    }
}

/// The AGQ deviance gradient into `grad[..1 + p]` at `params = [λ | β]`, from
/// the modes and `η_fix` the last AGQ value evaluation at `params` left in
/// `ws.pirls`. `x` is the fixed-effect design that evaluation read (the search
/// design under `BetaScale`). Precondition: [`covers`] holds and `sc` was built
/// for this `p` and `ws.nagq`.
///
/// `false`, with `grad` unspecified, when a row's predictor at the mode or at a
/// node reaches the link's clamp (`family::clamp_eta_bounds`; only Poisson-log
/// has one here). Past it the value kernel's deviance stops moving while these
/// row terms do not, so the two would describe different objectives.
// Index loops: each walks several per-node or per-coordinate arrays in step.
#[allow(clippy::needless_range_loop)]
pub(crate) fn assemble(
    ws: &GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    idx: &ClusterRowIndex,
    params: &[f64],
    sc: &mut AgqGradScratch,
    grad: &mut [f64],
) -> bool {
    let fam = row_family(ws.family).expect("assemble: covers() holds");
    let (eta_lo, eta_hi) = crate::family::clamp_eta_bounds(ws.family);
    let mut clamped = false;
    let (p, k) = (sc.p, sc.k);
    debug_assert_eq!(p, ws.p);
    debug_assert_eq!(k, ws.nagq as usize);
    let m = p + 1;
    let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
    let nodes = &crate::consts::GH_NODES[blk..blk + k];
    let sq2 = std::f64::consts::SQRT_2;
    let lam = params[0];
    let ef = &ws.pirls.eta_fixed;
    let u = &ws.pirls.u;
    let pw = &ws.prior_w;
    grad[..m].fill(0.0);
    for c in 0..ws.groupings.n_primary {
        let rows = idx.cluster_rows(c);
        let uh = u[c];
        // Sums at the mode: Σℓ', Σℓ'', Σℓ''' and their x-weighted forms.
        let (mut s1, mut s2, mut s3) = (0.0, 0.0, 0.0);
        sc.x2.fill(0.0);
        sc.x3.fill(0.0);
        for &i in rows {
            let i = i as usize;
            let eta = ef[i] + lam * uh;
            clamped |= eta <= eta_lo || eta >= eta_hi;
            let (_, a, b, d) = row_terms(fam, y[i], pw[i], eta);
            s1 += a;
            s2 += b;
            s3 += d;
            for kk in 0..p {
                let xv = x[(i, kk)];
                sc.x2[kk] += b * xv;
                sc.x3[kk] += d * xv;
            }
        }
        let a_c = 1.0 - lam * lam * s2;
        let sig = 1.0 / a_c.sqrt();
        let luuu = lam * lam * lam * s3;
        for g in 0..m {
            // ∂_γ ℓ_c'(û) and ∂_γ ℓ_c''(û); coordinate 0 is λ.
            let (dlu, dluu) = if g == 0 {
                (s1 + lam * uh * s2, 2.0 * lam * s2 + lam * lam * uh * s3)
            } else {
                (lam * sc.x2[g - 1], lam * lam * sc.x3[g - 1])
            };
            sc.du[g] = dlu / a_c;
            let da = -(dluu + luuu * sc.du[g]);
            sc.dls[g] = -0.5 * da / a_c;
        }
        // Node terms.
        let mut lmax = f64::NEG_INFINITY;
        for j in 0..k {
            let u_j = uh + sq2 * sig * nodes[j];
            sc.uj[j] = u_j;
            let mut l = -0.5 * u_j * u_j;
            let mut t = 0.0;
            let xt = &mut sc.xt[j * p..(j + 1) * p];
            xt.fill(0.0);
            for &i in rows {
                let i = i as usize;
                let eta = ef[i] + lam * u_j;
                clamped |= eta <= eta_lo || eta >= eta_hi;
                let (li, si, _, _) = row_terms(fam, y[i], pw[i], eta);
                l += li;
                t += si;
                for kk in 0..p {
                    xt[kk] += si * x[(i, kk)];
                }
            }
            sc.tj[j] = t;
            sc.lj[j] = sc.ln_omega[j] + l;
            lmax = lmax.max(sc.lj[j]);
        }
        let mut z = 0.0;
        for j in 0..k {
            z += (sc.lj[j] - lmax).exp();
        }
        let lse = lmax + z.ln();
        for g in 0..m {
            let mut acc = sc.dls[g];
            for j in 0..k {
                let pj = (sc.lj[j] - lse).exp();
                let dl_g = if g == 0 {
                    sc.tj[j] * sc.uj[j]
                } else {
                    sc.xt[j * p + g - 1]
                };
                let lu = lam * sc.tj[j] - sc.uj[j];
                acc += pj * (dl_g + lu * (sc.du[g] + sq2 * nodes[j] * sig * sc.dls[g]));
            }
            grad[g] += -2.0 * acc;
        }
    }
    !clamped
}

/// The conditional modes `û_c` and `η_fix` at `ws.params`, into `ws.pirls`, by
/// the same blocked PIRLS solve the AGQ value evaluation runs first
/// (`agq::agq_deviance`), seeded the way `deviance::laplace_deviance_at` seeds
/// (from `ws.u_seed` under `warm_seed_active`, else from zero), at
/// `ws.fd.pirls_tol_override`. Everything [`assemble`] reads, without the node
/// sum the value itself needs: the gradient-difference Hessian never reads a
/// value at its perturbed points, and on probit that sum was 60% of each
/// point's cost. The mode is bit-identical to the one the full evaluation
/// leaves. `false` when PIRLS does not converge.
pub(crate) fn mode_at(
    ws: &mut GlmmWorkspace,
    x: MatRef<f64>,
    y: &[f64],
    cluster_ids: &[u32],
    n: usize,
) -> bool {
    let kk = ws.k.max(1);
    if ws.fd.warm_seed_active {
        ws.pirls.u[..kk].copy_from_slice(&ws.u_seed[..kk]);
    } else {
        ws.pirls.u[..kk].fill(0.0);
    }
    let (n_theta, p) = (ws.n_theta, ws.p);
    let (family, nb_theta, weighted) = (ws.family, ws.nb_theta, ws.weighted);
    let tol = ws.fd.pirls_tol_override;
    let GlmmWorkspace {
        groupings,
        params,
        beta_rhs,
        pirls,
        z_buf,
        prior_w,
        wx,
        offset,
        ..
    } = ws;
    beta_rhs[..p].copy_from_slice(&params[n_theta..n_theta + p]);
    crate::lmm::primary_lambda(&params[..n_theta], groupings.primary_q, &mut pirls.lam);
    // Throwaway counters: SE evaluations stay out of every reported count.
    let mut counters = crate::counters::EvalCounters::new();
    let (_, _, _, conv) = pirls_solve_blocked(
        family,
        nb_theta,
        groupings,
        cluster_ids,
        x,
        y,
        &prior_w[..n],
        weighted,
        &mut beta_rhs[..p],
        BetaStep::Fixed,
        pirls,
        z_buf,
        None,
        wx,
        offset.as_deref(),
        tol,
        n,
        &mut counters,
    );
    conv
}
