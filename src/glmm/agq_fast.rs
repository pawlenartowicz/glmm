//! `f64` node sums of the AGQ kernels (`agq::agq_deviance`,
//! `agq::agq_deviance_vec`) for the log links and logit, and of the scalar
//! kernel for probit.
//!
//! The generic node loop evaluates every row's unit deviance at every node
//! from scratch (`family::dev_resid_eta`: clamp, inverse link, μ-form deviance
//! with its `log1p`) and subtracts the center loglik `ℓ_c(ũ_c)` afterwards.
//! Here each row's node deviance is written as its change from the row's
//! center `(η₀, μ₀)`, the PIRLS mode that `ℓ_c(ũ_c)` is read at, as a
//! function of the node shift `δ = η − η₀`. With `w` the prior weight and
//! `d` the unit deviance, the row adds `−½w·(d(η₀+δ) − d(η₀))` to
//! `ℓ_c(u) − ℓ_c(ũ_c)`, and
//!
//! ```text
//!   Poisson-log  ½Δd = μ₀·expm1(δ) − y·δ
//!   Gamma-log    ½Δd = δ + (y/μ₀)·expm1(−δ)
//!   logit        ½Δd = log1p(p₀·expm1(δ)) − y·δ,            p₀ = σ(η₀),  η₀ ≤ 0
//!                     = log1p(q₀·expm1(−δ)) + (1−y)·δ,       q₀ = σ(−η₀), η₀ > 0
//!   NB           ½Δd = (y+θ)·log1p(b₀·expm1(δ)) − y·δ,       b₀ = μ₀/(μ₀+θ), μ₀ ≤ θ
//!                     = (y+θ)·log1p(c₀·expm1(−δ)) + θ·δ,      c₀ = θ/(μ₀+θ),  μ₀ > θ
//! ```
//!
//! (Poisson `d = 2[y ln(y/μ) − (y−μ)]`, Gamma `2[−ln(y/μ) + (y−μ)/μ]`,
//! binomial `2[y ln(y/μ) + (1−y) ln((1−y)/(1−μ))]` with
//! `ln(1+e^{η₀+δ}) − ln(1+e^{η₀}) = log1p(σ(η₀)·expm1(δ))`, NB
//! `2[y ln(y/μ) − (y+θ) ln((y+θ)/(μ+θ))]` with
//! `ln((μ+θ)/(μ₀+θ)) = log1p(b₀·expm1(δ))`; McCullagh–Nelder 1989 §2.3.) The
//! logit and NB pairs pick the side whose coefficient is at most ½, so the
//! `log1p` argument stays above −½ and keeps its relative precision. So every
//! row term is `a·expm1(s·δ) + h·δ` (`Lin`: Poisson, Gamma-log) or
//! `a·log1p(p·expm1(s·δ)) + h·δ` (`Log1p`: logit, NB), with `(a, p, s, h)`
//! computed once per call. No `y·ln y` constant appears and nothing cancels
//! against the center: the term is small where the deviance change is small.
//!
//! A row whose node predictors could reach the link's clamp (`family::clamp_eta`)
//! or the log-link tail (`family::in_tail`, NB and Gamma-log), or whose two
//! parts are large enough that their cancellation loses absolute precision
//! (`TERM_MAG_MAX`), is a slow row:
//! its constants are zero (`s = 0` marks it, so its fast term is exactly 0)
//! and it is added back per node from `family::dev_resid_eta` minus its center
//! `family::dev_resid_at`, as the generic loop computes it. The guard bounds
//! the node shift over the whole grid: `|δ| ≤ ‖g‖₁·max|z_j|`, where
//! `δ(z) = g·z` is linear in the standardized node (scalar `g = √2·λ·σ_c`;
//! vector `g = √2·L_c⁻¹Λ'm_i`, `m_i = [1, z_i]`).
//!
//! Cluster compression (scalar route, `Lin` only). With a scalar intercept the
//! node shift `δ = λ·√2·σ_c·z_j` is the same for every row of cluster `c`, and
//! `s` is the family's constant (`+1` Poisson, `−1` Gamma-log), so the cluster's
//! row terms sum to `expm1(s·δ)·Σa_i + δ·Σh_i`: two per-cluster sums replace a
//! pass over the rows per node. A cluster takes it when its largest center
//! `|η₀|` plus the node-shift bound stays within `agq::COMPRESS_ETA_MAX` (so
//! every row is a fast row) and both sums are finite (`agq::compress_ok`);
//! any other cluster keeps the row terms.
//!
//! Probit (scalar route only, [`probit_sums`]). Probit's `ln Φ` has no
//! `expm1` form, so its row term is the loglik change itself,
//! `w·[y·(ln Φ(η) − ln Φ(η₀)) + (1−y)·(ln Φ(−η) − ln Φ(−η₀))]`, with every
//! `ln Φ` from the SIMD `simd_transcendental::ln_phi_fill` (one branch-free
//! `erfc` and one `ln` per value, relative-accurate on both sides) in place of
//! the generic loop's per-row `link_inv`, `erfc` and μ-form `log1p`. The two
//! forms are the same function: the generic deviance takes `ln μ`, `ln(1−μ)`
//! from η on its tail (`family::in_tail`) and from the stored μ elsewhere. The
//! center values `ln Φ(±η₀)` are computed once per call; the saturated
//! `y·ln y` parts cancel in the difference and never appear. A row whose node
//! predictors could pass `LN_PHI_X_MAX` (37, inside `erfc`'s certified range)
//! is a slow row, added back the generic way as above.

use super::agq::{compress_ok, ClusterRowIndex};
use crate::family::{ETA_MAX, LOG_TAIL_ETA};
use crate::simd_transcendental::{ln_phi_fill, LN_PHI_X_MAX};
use crate::spec::{BinomialLink, Family, GammaLink};
use std::f64::consts::SQRT_2;

/// Largest node shift `|δ|` a fast row takes: `expm1(±δ)` stays far from
/// overflow. Every log-link row hits its `η` limit first.
const DELTA_MAX: f64 = 600.0;

/// Which of the two row-term shapes a family takes (module docs).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// `a·expm1(s·δ) + h·δ`: Poisson, Gamma-log.
    Lin,
    /// `a·log1p(p·expm1(s·δ)) + h·δ`: binomial logit, NB.
    Log1p,
}

/// The fast-path kind of `family`, `None` for the families and links that
/// keep the generic node loop.
pub(crate) fn kind(family: Family) -> Option<Kind> {
    match family {
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
        } => Some(Kind::Lin),
        Family::Binomial {
            link: BinomialLink::Logit,
        }
        | Family::NegativeBinomial { .. } => Some(Kind::Log1p),
        _ => None,
    }
}

/// `|η|` a fast row's node predictors must stay strictly inside: the clamp
/// for Poisson, the tail switch for NB and Gamma-log, none for logit (no
/// clamp; its tail form is the same function the row term evaluates).
fn eta_limit(family: Family) -> f64 {
    match family {
        Family::Poisson { .. } => ETA_MAX,
        Family::Binomial { .. } => f64::INFINITY,
        _ => LOG_TAIL_ETA,
    }
}

/// Whether a row with center `eta0` and node-shift bound `b` is a fast row.
/// The `1e-12` widening covers the rounding between the bound and the node
/// predictors the loop forms. False on a NaN.
#[inline]
fn is_fast(family: Family, eta0: f64, b: f64) -> bool {
    let reach = (eta0.abs() + b) * (1.0 + 1e-12);
    reach < eta_limit(family) && b <= DELTA_MAX
}

/// Largest bound on a fast row's two parts, `|a|·max|f(s·δ)| + |h|·max|δ|`
/// over the node range (`f = expm1` for `Lin`, `log1p(p·expm1(·))`, bounded
/// by `|δ|`, for `Log1p`). The two parts nearly cancel where the row's
/// deviance change is small, so the term's absolute rounding error is about
/// `eps` times this bound: `2^20·eps ≈ 2.3e-10` per row. A row past it is a
/// slow row, whose μ-form deviance keeps its relative precision. On a log
/// link the parts grow like `√μ₀` (`b ∝ 1/√(Σμ)`): about `7.5·√μ₀` for a
/// one-row Poisson cluster at k = 7, so counts above about `2e10`.
const TERM_MAG_MAX: f64 = 1_048_576.0;

/// The fast row constants of a row with center `(eta0, mu0)`, prior weight
/// `w`, node-shift bound `b` and its `g = mag_factor(kind, b)`, or `None`
/// for a slow row: one whose node
/// range reaches the clamp or the tail ([`is_fast`]) or whose parts exceed
/// [`TERM_MAG_MAX`]. `None` on a NaN.
#[inline]
#[allow(clippy::too_many_arguments)]
fn fast_consts(
    family: Family,
    nb_theta: f64,
    y: f64,
    w: f64,
    eta0: f64,
    mu0: f64,
    b: f64,
    g: f64,
) -> Option<[f64; 4]> {
    if !is_fast(family, eta0, b) {
        return None;
    }
    let k = row_consts(family, nb_theta, y, w, eta0, mu0);
    (k[0].abs() * g + k[3].abs() * b <= TERM_MAG_MAX).then_some(k)
}

/// The bound on `|f(s·δ)|` for `|δ| ≤ b` that [`fast_consts`] scales `|a|` by:
/// `expm1(b)` for `Lin`, `b` for `Log1p` (`p ≤ ½`). Computed once per bound
/// (per cluster on the scalar route), not per row.
#[inline]
fn mag_factor(kind: Kind, b: f64) -> f64 {
    if kind == Kind::Lin {
        b.exp_m1()
    } else {
        b
    }
}

/// `(a, p, s, h)` of a fast row at its center `(eta0, mu0)`, prior weight `w`
/// (module docs; the row adds `−(term)` to the loglik difference).
#[inline]
pub(crate) fn row_consts(
    family: Family,
    nb_theta: f64,
    y: f64,
    w: f64,
    eta0: f64,
    mu0: f64,
) -> [f64; 4] {
    match family {
        Family::Poisson { .. } => [w * mu0, 0.0, 1.0, -(w * y)],
        Family::Gamma { .. } => [w * (y / mu0), 0.0, -1.0, w],
        Family::Binomial { .. } => {
            let e = (-eta0.abs()).exp();
            let pm = e / (1.0 + e); // σ(−|η₀|) ≤ ½
            if eta0 <= 0.0 {
                [w, pm, 1.0, -(w * y)]
            } else {
                [w, pm, -1.0, w * (1.0 - y)]
            }
        }
        Family::NegativeBinomial { .. } => {
            let den = mu0 + nb_theta;
            let a = w * (y + nb_theta);
            if mu0 <= nb_theta {
                [a, mu0 / den, 1.0, -(w * y)]
            } else {
                [a, nb_theta / den, -1.0, w * nb_theta]
            }
        }
        _ => unreachable!("agq_fast::kind admits only Poisson, Gamma-log, logit and NB"),
    }
}

/// One fast row's term at node shift `d` (module docs). The kernels inline
/// this form per tile; this scalar copy is the tests' reference.
#[cfg(test)]
#[inline(always)]
pub(crate) fn term<const LOG1P: bool>(a: f64, p: f64, s: f64, h: f64, d: f64) -> f64 {
    let e = (s * d).exp_m1();
    if LOG1P {
        a * (p * e).ln_1p() + h * d
    } else {
        a * e + h * d
    }
}

/// A slow row's `½w·(d(η) − d(η₀))` the generic way.
#[inline]
#[allow(clippy::too_many_arguments)]
fn slow_term(
    family: Family,
    nb_theta: f64,
    y: f64,
    w: f64,
    eta_node: f64,
    eta0: f64,
    mu0: f64,
) -> f64 {
    0.5 * w
        * (crate::family::dev_resid_eta(family, nb_theta, y, eta_node)
            - crate::family::dev_resid_at(family, nb_theta, y, eta0, mu0))
}

fn max_abs_node(nodes: &[f64]) -> f64 {
    nodes.iter().fold(0.0f64, |m, z| m.max(z.abs()))
}

/// Split the row-constant buffer: `a | p | s | h` (`m` each), then `extra`.
fn split_rows(buf: &mut Vec<f64>, m: usize, extra: usize) -> [&mut [f64]; 5] {
    let need = 4 * m + extra;
    if buf.len() < need {
        buf.resize(need, 0.0);
    }
    let (a, rest) = buf[..need].split_at_mut(m);
    let (p, rest) = rest.split_at_mut(m);
    let (sg, rest) = rest.split_at_mut(m);
    let (h, rest) = rest.split_at_mut(m);
    [a, p, sg, h, rest]
}

/// One node of a compressed cluster: `ℓ_c(u_cj) − ℓ_c(ũ_c)` from the cluster's
/// sums `sa = Σa_i`, `sh = Σh_i` of its `Lin` row constants, `sg` the family's
/// `s` (module docs). The prior part is `−h·(ũ_c + ½h)` with `h = √2·σ_c·z_j`,
/// as the row-term arms form it. Shared by both arms of [`scalar_sums`], so a
/// compressed cluster's sum is the same bits on either.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compressed_node(sa: f64, sh: f64, sg: f64, u_c: f64, a_block: f64, lambda: f64, zj: f64) -> f64 {
    let hc = SQRT_2 * (1.0 / a_block) * zj;
    let d = lambda * hc;
    -(hc * (u_c + 0.5 * hc)) - (sa * (sg * d).exp_m1() + sh * d)
}

/// Elements of one SIMD tile: bounded stack scratch, never an `n × k^q`
/// buffer.
const TILE: usize = 128;

/// `e[t] ← expm1(e[t])` over a tile through the SIMD `exp` kernel, as
/// `exp(x) − 1`: an absolute error of an ulp of 1, which the row term scales
/// by `a`, below the rounding the center loglik already carries.
#[inline]
fn expm1_tile(e: &mut [f64]) {
    crate::simd_transcendental::exp_fill(e);
    for x in e.iter_mut() {
        *x -= 1.0;
    }
}

/// `r[t] ← log1p(r[t])` over a tile (`r > −1`): `ln(u)·r/(u − 1)` with
/// `u = 1 + r` rounded and `ln u` from the SIMD `ln_pos_fill`, which is
/// `log1p(r)` to the precision of `ln u` (Goldberg 1991, "What every computer
/// scientist should know about floating-point arithmetic", Theorem 4; the NB
/// fold of `simd_transcendental::family_pass` takes it the same way).
#[inline]
fn log1p_tile(r: &mut [f64], l: &mut [f64]) {
    for (lt, &rt) in l.iter_mut().zip(r.iter()) {
        *lt = 1.0 + rt;
    }
    crate::simd_transcendental::ln_pos_fill(l);
    for (rt, &lt) in r.iter_mut().zip(l.iter()) {
        let u = 1.0 + *rt;
        if u != 1.0 {
            *rt = lt * *rt / (u - 1.0);
        }
    }
}

/// Scalar-intercept node sums: fills `sum[c] = Σ_j exp(ln w_j + z_j² +
/// ℓ_c(u_cj) − ℓ_c(ũ_c))`, the quantity both arms of `agq_deviance`'s node
/// loop leave in `sum`. `eta`/`prob` are the PIRLS mode the center term was
/// read at; `dl` and `acc` are `s`-long scratch. With a cluster index the
/// sums run cluster-outer (rayon under `parallel`), without one node-outer,
/// bit-identical to each other as the generic arms are.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scalar_sums(
    kind: Kind,
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    cluster_ids: &[u32],
    n: usize,
    eta: &[f64],
    prob: &[f64],
    eta_fixed: &[f64],
    u: &[f64],
    a_blocks: &[f64],
    lambda: f64,
    nodes: &[f64],
    wts: &[f64],
    rows_buf: &mut Vec<f64>,
    slow: &mut Vec<u32>,
    dl: &mut [f64],
    acc: &mut [f64],
    sum: &mut [f64],
    cluster_rows: Option<&ClusterRowIndex>,
) {
    let s = sum.len();
    let k = nodes.len();
    let mut tbl = [0.0f64; 2 * crate::consts::MAX_NAGQ as usize];
    for (j, (&zj, &wj)) in nodes.iter().zip(wts).enumerate() {
        tbl[2 * j] = zj;
        tbl[2 * j + 1] = wj.ln() + zj * zj;
    }
    let zmax = max_abs_node(nodes);
    // Per-cluster bound on |δ| = |λ|·√2·σ_c·|z_j| over the nodes.
    // `acc[c]` holds the bound's magnitude factor for the row pass
    // (`mag_factor`); the node-outer arm reuses it as its accumulator after.
    for c in 0..s {
        dl[c] = lambda.abs() * SQRT_2 * (1.0 / a_blocks[c]) * zmax;
        acc[c] = mag_factor(kind, dl[c]);
    }
    // Per-cluster extras: slow-row count | compressed sums Σa, Σh | max |η₀|,
    // then the compressed flag in place of the max.
    let [ra, rp, rs, rh, extra] = split_rows(rows_buf, n, 4 * s);
    let (nslow, extra) = extra.split_at_mut(s);
    let (csa, extra) = extra.split_at_mut(s);
    let (csh, cflag) = extra.split_at_mut(s);
    nslow.fill(0.0);
    let lin = kind == Kind::Lin;
    if lin {
        csa.fill(0.0);
        csh.fill(0.0);
        cflag.fill(0.0);
    }
    slow.clear();
    for i in 0..n {
        let c = cluster_ids[i] as usize;
        let fc = fast_consts(
            family, nb_theta, y[i], prior_w[i], eta[i], prob[i], dl[c], acc[c],
        );
        if let Some([a, p, sg, h]) = fc {
            (ra[i], rp[i], rs[i], rh[i]) = (a, p, sg, h);
            if lin {
                csa[c] += a;
                csh[c] += h;
            }
        } else {
            (ra[i], rp[i], rs[i], rh[i]) = (0.0, 0.0, 0.0, 0.0);
            slow.push(i as u32);
            nslow[c] += 1.0;
        }
        if lin {
            let e = eta[i].abs();
            if e > cflag[c] {
                cflag[c] = e;
            }
        }
    }
    // Compressed clusters (module docs): `cflag[c] = 1`. Their rows' constants
    // are zeroed for the node-outer arm, whose tiles then add exact zeros.
    let mut any_rows = !lin;
    if lin {
        for c in 0..s {
            let ok = nslow[c] == 0.0 && compress_ok(cflag[c], dl[c], csa[c], csh[c]);
            cflag[c] = if ok { 1.0 } else { 0.0 };
            any_rows |= !ok;
        }
        if cluster_rows.is_none() && any_rows {
            for i in 0..n {
                if cflag[cluster_ids[i] as usize] != 0.0 {
                    (ra[i], rp[i], rs[i], rh[i]) = (0.0, 0.0, 0.0, 0.0);
                }
            }
        }
    }
    let sg_lin = if matches!(family, Family::Gamma { .. }) {
        -1.0
    } else {
        1.0
    };
    let (ra, rp, rs, rh, nslow) = (&*ra, &*rp, &*rs, &*rh, &*nslow);
    let (csa, csh, cflag) = (&*csa, &*csh, &*cflag);
    let slow_add = |i: usize, c: usize, zj: f64| {
        let u_cj = u[c] + SQRT_2 * (1.0 / a_blocks[c]) * zj;
        slow_term(
            family,
            nb_theta,
            y[i],
            prior_w[i],
            eta_fixed[i] + lambda * u_cj,
            eta[i],
            prob[i],
        )
    };
    if let Some(idx) = cluster_rows {
        // Cluster-outer: every operand and every per-accumulator order of
        // the node-outer loop below (the SIMD kernels are elementwise, head
        // and tail bit-identical), so the two arms agree bit for bit.
        let per_cluster = |c: usize, sum_c: &mut f64| {
            if lin && cflag[c] != 0.0 {
                let mut acc_sum = 0.0;
                for j in 0..k {
                    let a_c = compressed_node(
                        csa[c],
                        csh[c],
                        sg_lin,
                        u[c],
                        a_blocks[c],
                        lambda,
                        nodes[j],
                    );
                    acc_sum += (tbl[2 * j + 1] + a_c).exp();
                }
                *sum_c = acc_sum;
                return;
            }
            let rows = idx.cluster_rows(c);
            let m = rows.len();
            let g = (TILE / m.max(1)).clamp(1, k.max(1));
            let mut ebuf = [0.0f64; TILE];
            let mut lbuf = [0.0f64; TILE];
            let mut d_g = [0.0f64; TILE];
            let mut acc_g = [0.0f64; TILE];
            let mut acc_sum = 0.0;
            let mut j0 = 0;
            while j0 < k {
                let j1 = (j0 + g).min(k);
                let ng = j1 - j0;
                for gi in 0..ng {
                    let hc = SQRT_2 * (1.0 / a_blocks[c]) * nodes[j0 + gi];
                    d_g[gi] = lambda * hc;
                    acc_g[gi] = -(hc * (u[c] + 0.5 * hc));
                }
                let chunk = TILE / ng;
                let mut p0 = 0;
                while p0 < m {
                    let p1 = (p0 + chunk).min(m);
                    let mc = p1 - p0;
                    let len = ng * mc;
                    for gi in 0..ng {
                        for r in 0..mc {
                            ebuf[gi * mc + r] = rs[rows[p0 + r] as usize] * d_g[gi];
                        }
                    }
                    expm1_tile(&mut ebuf[..len]);
                    if kind == Kind::Log1p {
                        for gi in 0..ng {
                            for r in 0..mc {
                                ebuf[gi * mc + r] *= rp[rows[p0 + r] as usize];
                            }
                        }
                        log1p_tile(&mut ebuf[..len], &mut lbuf[..len]);
                    }
                    for gi in 0..ng {
                        let mut a_c = acc_g[gi];
                        for r in 0..mc {
                            let i = rows[p0 + r] as usize;
                            a_c -= ra[i] * ebuf[gi * mc + r] + rh[i] * d_g[gi];
                        }
                        acc_g[gi] = a_c;
                    }
                    p0 = p1;
                }
                for gi in 0..ng {
                    let mut acc_c = acc_g[gi];
                    if nslow[c] > 0.0 {
                        for &i in rows {
                            let i = i as usize;
                            if rs[i] == 0.0 {
                                acc_c -= slow_add(i, c, nodes[j0 + gi]);
                            }
                        }
                    }
                    acc_sum += (tbl[2 * (j0 + gi) + 1] + acc_c).exp();
                }
                j0 = j1;
            }
            *sum_c = acc_sum;
        };
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            sum[..s]
                .par_iter_mut()
                .enumerate()
                .for_each(|(c, sc)| per_cluster(c, sc));
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        for (c, sc) in sum[..s].iter_mut().enumerate() {
            per_cluster(c, sc);
        }
        return;
    }
    let mut ebuf = [0.0f64; TILE];
    let mut lbuf = [0.0f64; TILE];
    for (j, &zj) in nodes.iter().enumerate() {
        for c in 0..s {
            let hc = SQRT_2 * (1.0 / a_blocks[c]) * zj;
            dl[c] = lambda * hc;
            acc[c] = if lin && cflag[c] != 0.0 {
                compressed_node(csa[c], csh[c], sg_lin, u[c], a_blocks[c], lambda, zj)
            } else {
                -(hc * (u[c] + 0.5 * hc))
            };
        }
        let mut i0 = if any_rows { 0 } else { n };
        while i0 < n {
            let i1 = (i0 + TILE).min(n);
            let len = i1 - i0;
            let e = &mut ebuf[..len];
            for (t, et) in e.iter_mut().enumerate() {
                let i = i0 + t;
                *et = rs[i] * dl[cluster_ids[i] as usize];
            }
            expm1_tile(e);
            if kind == Kind::Log1p {
                for (t, et) in e.iter_mut().enumerate() {
                    *et *= rp[i0 + t];
                }
                log1p_tile(e, &mut lbuf[..len]);
            }
            for (t, &et) in e.iter().enumerate() {
                let i = i0 + t;
                let c = cluster_ids[i] as usize;
                acc[c] -= ra[i] * et + rh[i] * dl[c];
            }
            i0 = i1;
        }
        for &i in slow.iter() {
            let i = i as usize;
            let c = cluster_ids[i] as usize;
            acc[c] -= slow_add(i, c, zj);
        }
        let ln_wj = tbl[2 * j + 1];
        for c in 0..s {
            sum[c] += (ln_wj + acc[c]).exp();
        }
    }
}

/// Whether `family` takes [`probit_sums`]: binomial probit, scalar intercept.
/// The vector kernel keeps the generic node loop for it.
pub(crate) fn is_probit(family: Family) -> bool {
    matches!(
        family,
        Family::Binomial {
            link: BinomialLink::Probit
        }
    )
}

/// Whether a probit row with center `eta0` and node-shift bound `b` is a fast
/// row: its node predictors stay inside [`LN_PHI_X_MAX`], the range
/// `ln_phi_fill` is certified on. The `1e-12` widening covers the rounding
/// between the bound and the node predictors the loop forms. False on a NaN.
#[inline]
fn probit_is_fast(eta0: f64, b: f64) -> bool {
    (eta0.abs() + b) * (1.0 + 1e-12) < LN_PHI_X_MAX
}

/// Split the probit row buffer: `x0 | sg | ta | tb | c1 | c2` (`n` each).
fn split_probit_rows(buf: &mut Vec<f64>, n: usize) -> [&mut [f64]; 6] {
    let need = 6 * n;
    if buf.len() < need {
        buf.resize(need, 0.0);
    }
    let (x0, rest) = buf[..need].split_at_mut(n);
    let (sg, rest) = rest.split_at_mut(n);
    let (ta, rest) = rest.split_at_mut(n);
    let (tb, rest) = rest.split_at_mut(n);
    let (c1, c2) = rest.split_at_mut(n);
    [x0, sg, ta, tb, c1, c2]
}

/// [`scalar_sums`] for binomial probit (module docs, "Probit"): fills the same
/// `sum[c] = Σ_j exp(ln w_j + z_j² + ℓ_c(u_cj) − ℓ_c(ũ_c))`. Each fast row
/// adds its loglik change from the center,
///
/// ```text
///   w·[y·(ln Φ(η) − ln Φ(η₀)) + (1−y)·(ln Φ(−η) − ln Φ(−η₀))],   η = η₀ + δ,
/// ```
///
/// written on the row's side `sg = ±1` as `ta·(L(x) − L(x₀)) + tb·(L(−x) −
/// L(−x₀))` with `x = sg·η`, `L = ln Φ` from `simd_transcendental::ln_phi_fill`,
/// `ta`/`tb` the side's weights (a Bernoulli row has `tb = 0`, one `ln Φ` per
/// node), and the center values `L(±x₀)` computed once per call. The second
/// side runs only when some row has `0 < y < 1` (aggregated binomial), for
/// every row then, so both arms below form the same expression per row. A
/// slow row ([`probit_is_fast`] false) has `sg = ta = tb = 0` and `x₀ = 0`, so
/// its fast term is exactly 0, and is added back per node through
/// `family::dev_resid_eta`, as the generic loop computes it. At the node
/// `z = 0` the shift is exactly 0, so every fast row adds exactly 0 there.
/// Same arms, same per-accumulator order and the same `parallel` structure as
/// [`scalar_sums`]: cluster-outer and node-outer agree bit for bit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn probit_sums(
    y: &[f64],
    prior_w: &[f64],
    cluster_ids: &[u32],
    n: usize,
    eta: &[f64],
    prob: &[f64],
    eta_fixed: &[f64],
    u: &[f64],
    a_blocks: &[f64],
    lambda: f64,
    nodes: &[f64],
    wts: &[f64],
    rows_buf: &mut Vec<f64>,
    slow: &mut Vec<u32>,
    dl: &mut [f64],
    acc: &mut [f64],
    sum: &mut [f64],
    cluster_rows: Option<&ClusterRowIndex>,
) {
    let family = Family::Binomial {
        link: BinomialLink::Probit,
    };
    let s = sum.len();
    let k = nodes.len();
    let mut tbl = [0.0f64; 2 * crate::consts::MAX_NAGQ as usize];
    for (j, (&zj, &wj)) in nodes.iter().zip(wts).enumerate() {
        tbl[2 * j] = zj;
        tbl[2 * j + 1] = wj.ln() + zj * zj;
    }
    let zmax = max_abs_node(nodes);
    for c in 0..s {
        dl[c] = lambda.abs() * SQRT_2 * (1.0 / a_blocks[c]) * zmax;
    }
    let [rx0, rsg, rta, rtb, rc1, rc2] = split_probit_rows(rows_buf, n);
    slow.clear();
    let mut both = false;
    for i in 0..n {
        let c = cluster_ids[i] as usize;
        if probit_is_fast(eta[i], dl[c]) {
            let (yi, w) = (y[i], prior_w[i]);
            let sg = if yi >= 0.5 { 1.0 } else { -1.0 };
            let (ta, tb) = if sg > 0.0 {
                (w * yi, w * (1.0 - yi))
            } else {
                (w * (1.0 - yi), w * yi)
            };
            (rx0[i], rsg[i], rta[i], rtb[i]) = (sg * eta[i], sg, ta, tb);
            both |= tb != 0.0;
        } else {
            (rx0[i], rsg[i], rta[i], rtb[i]) = (0.0, 0.0, 0.0, 0.0);
            slow.push(i as u32);
        }
    }
    // Center values L(x₀), and L(−x₀) when the second side runs, in tiles
    // (`ln_phi_fill` is elementwise, so the tiling does not change a value).
    let mut nqb = [0.0f64; TILE];
    let mut i0 = 0;
    while i0 < n {
        let i1 = (i0 + TILE).min(n);
        rc1[i0..i1].copy_from_slice(&rx0[i0..i1]);
        ln_phi_fill(&mut rc1[i0..i1], &mut nqb);
        if both {
            for t in i0..i1 {
                rc2[t] = -rx0[t];
            }
            ln_phi_fill(&mut rc2[i0..i1], &mut nqb);
        }
        i0 = i1;
    }
    let (rx0, rsg, rta, rtb, rc1, rc2) = (&*rx0, &*rsg, &*rta, &*rtb, &*rc1, &*rc2);
    let slow_ro: &[u32] = slow;
    let slow_add = |i: usize, c: usize, zj: f64| {
        let u_cj = u[c] + SQRT_2 * (1.0 / a_blocks[c]) * zj;
        slow_term(
            family,
            0.0,
            y[i],
            prior_w[i],
            eta_fixed[i] + lambda * u_cj,
            eta[i],
            prob[i],
        )
    };
    // One row's loglik change at node shift `d`, from its tile values
    // `lb = L(x)` and `ln = L(−x)` (the latter read only when `both`).
    let row_term = |i: usize, lb: f64, ln: f64| {
        if both {
            rta[i] * (lb - rc1[i]) + rtb[i] * (ln - rc2[i])
        } else {
            rta[i] * (lb - rc1[i])
        }
    };
    if let Some(idx) = cluster_rows {
        let per_cluster = |c: usize, sum_c: &mut f64| {
            let rows = idx.cluster_rows(c);
            let m = rows.len();
            let g = (TILE / m.max(1)).clamp(1, k.max(1));
            let mut xb = [0.0f64; TILE];
            let mut xn = [0.0f64; TILE];
            let mut nq = [0.0f64; TILE];
            let mut d_g = [0.0f64; TILE];
            let mut acc_g = [0.0f64; TILE];
            let has_slow = rows.iter().any(|&i| rsg[i as usize] == 0.0);
            let mut acc_sum = 0.0;
            let mut j0 = 0;
            while j0 < k {
                let j1 = (j0 + g).min(k);
                let ng = j1 - j0;
                for gi in 0..ng {
                    let hc = SQRT_2 * (1.0 / a_blocks[c]) * nodes[j0 + gi];
                    d_g[gi] = lambda * hc;
                    acc_g[gi] = -(hc * (u[c] + 0.5 * hc));
                }
                let chunk = TILE / ng;
                let mut p0 = 0;
                while p0 < m {
                    let p1 = (p0 + chunk).min(m);
                    let mc = p1 - p0;
                    let len = ng * mc;
                    for gi in 0..ng {
                        for r in 0..mc {
                            let i = rows[p0 + r] as usize;
                            xb[gi * mc + r] = rx0[i] + rsg[i] * d_g[gi];
                        }
                    }
                    if both {
                        for t in 0..len {
                            xn[t] = -xb[t];
                        }
                        ln_phi_fill(&mut xn[..len], &mut nq);
                    }
                    ln_phi_fill(&mut xb[..len], &mut nq);
                    for (gi, ag) in acc_g[..ng].iter_mut().enumerate() {
                        let mut a_c = *ag;
                        for r in 0..mc {
                            let i = rows[p0 + r] as usize;
                            let t = gi * mc + r;
                            a_c += row_term(i, xb[t], xn[t]);
                        }
                        *ag = a_c;
                    }
                    p0 = p1;
                }
                for gi in 0..ng {
                    let mut acc_c = acc_g[gi];
                    if has_slow {
                        for &i in rows {
                            let i = i as usize;
                            if rsg[i] == 0.0 {
                                acc_c -= slow_add(i, c, nodes[j0 + gi]);
                            }
                        }
                    }
                    acc_sum += (tbl[2 * (j0 + gi) + 1] + acc_c).exp();
                }
                j0 = j1;
            }
            *sum_c = acc_sum;
        };
        #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
        {
            use rayon::prelude::*;
            sum[..s]
                .par_iter_mut()
                .enumerate()
                .for_each(|(c, sc)| per_cluster(c, sc));
        }
        #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
        for (c, sc) in sum[..s].iter_mut().enumerate() {
            per_cluster(c, sc);
        }
        return;
    }
    let mut xb = [0.0f64; TILE];
    let mut xn = [0.0f64; TILE];
    let mut nq = [0.0f64; TILE];
    sum[..s].fill(0.0);
    for (j, &zj) in nodes.iter().enumerate() {
        for c in 0..s {
            let hc = SQRT_2 * (1.0 / a_blocks[c]) * zj;
            dl[c] = lambda * hc;
            acc[c] = -(hc * (u[c] + 0.5 * hc));
        }
        let mut i0 = 0;
        while i0 < n {
            let i1 = (i0 + TILE).min(n);
            let len = i1 - i0;
            for (t, xt) in xb[..len].iter_mut().enumerate() {
                let i = i0 + t;
                *xt = rx0[i] + rsg[i] * dl[cluster_ids[i] as usize];
            }
            if both {
                for t in 0..len {
                    xn[t] = -xb[t];
                }
                ln_phi_fill(&mut xn[..len], &mut nq);
            }
            ln_phi_fill(&mut xb[..len], &mut nq);
            for t in 0..len {
                let i = i0 + t;
                acc[cluster_ids[i] as usize] += row_term(i, xb[t], xn[t]);
            }
            i0 = i1;
        }
        for &i in slow_ro.iter() {
            let i = i as usize;
            let c = cluster_ids[i] as usize;
            acc[c] -= slow_add(i, c, zj);
        }
        let ln_wj = tbl[2 * j + 1];
        for c in 0..s {
            sum[c] += (ln_wj + acc[c]).exp();
        }
    }
}

/// Vector-RE node sums (`q ∈ 2..=3`): fills `sum[c]` as `agq_deviance_vec`'s
/// per-cluster closure does. `node_tbl` is its `k^q·(q+1)` product-grid
/// table (z-vector, then the summed Liu–Pierce reweight). Row constants are
/// laid out in the CSR order of `idx`, so a cluster's rows are contiguous;
/// its node shifts are evaluated a tile at a time (several nodes per tile on
/// a small cluster).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vector_sums(
    kind: Kind,
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: &[f64],
    z_buf: &[f64],
    q: usize,
    n: usize,
    eta: &[f64],
    prob: &[f64],
    eta_fixed: &[f64],
    u: &[f64],
    a_blocks: &[f64],
    lam: &[f64],
    nodes: &[f64],
    node_tbl: &[f64],
    rows_buf: &mut Vec<f64>,
    sum: &mut [f64],
    idx: &ClusterRowIndex,
) {
    let s = sum.len();
    let qz = q - 1;
    let kq = node_tbl.len() / (q + 1);
    let zmax = max_abs_node(nodes);
    let all_rows = idx.rows();
    let [ra, rp, rs, rh, rest] = split_rows(rows_buf, n, qz * n + s);
    let (zc, nslow) = rest.split_at_mut(qz * n);
    nslow.fill(0.0);
    for c in 0..s {
        let lblk = &a_blocks[c * q * q..(c + 1) * q * q];
        for pos in idx.cluster_range(c) {
            let i = all_rows[pos] as usize;
            let mut m = [1.0f64; 3];
            for d in 0..qz {
                let zd = z_buf[i * qz + d];
                zc[pos * qz + d] = zd;
                m[d + 1] = zd;
            }
            // g = √2·L_c⁻¹·Λ'm: v = Λ'm (Λ lower row-major), then L x = v.
            let mut x = [0.0f64; 3];
            for cc in 0..q {
                let mut v = 0.0;
                for r in cc..q {
                    v += lam[r * q + cc] * m[r];
                }
                for t in 0..cc {
                    v -= lblk[cc * q + t] * x[t];
                }
                x[cc] = v / lblk[cc * q + cc];
            }
            let b = SQRT_2 * zmax * x[..q].iter().fold(0.0, |acc, v| acc + v.abs());
            let fc = fast_consts(
                family,
                nb_theta,
                y[i],
                prior_w[i],
                eta[i],
                prob[i],
                b,
                mag_factor(kind, b),
            );
            if let Some([a, p, sg, h]) = fc {
                (ra[pos], rp[pos], rs[pos], rh[pos]) = (a, p, sg, h);
            } else {
                (ra[pos], rp[pos], rs[pos], rh[pos]) = (0.0, 0.0, 0.0, 0.0);
                nslow[c] += 1.0;
            }
        }
    }
    let (ra, rp, rs, rh, zc, nslow) = (&*ra, &*rp, &*rs, &*rh, &*zc, &*nslow);
    let per_cluster = |c: usize, sum_c: &mut f64| {
        let range = idx.cluster_range(c);
        let (r0, m) = (range.start, range.len());
        let lblk = &a_blocks[c * q * q..(c + 1) * q * q];
        let ubase = c * q;
        // δ at CSR position `pos` for node shift `dv` (= Λ_p·h).
        let delta = |pos: usize, dv: &[f64; 3]| {
            let mut d = dv[0];
            for dd in 0..qz {
                d += zc[pos * qz + dd] * dv[dd + 1];
            }
            d
        };
        let g = (TILE / m.max(1)).clamp(1, TILE);
        let mut ebuf = [0.0f64; TILE];
        let mut lbuf = [0.0f64; TILE];
        let mut dbuf = [0.0f64; TILE];
        let mut hh_g = [[0.0f64; 3]; TILE];
        let mut dv_g = [[0.0f64; 3]; TILE];
        let mut acc_g = [0.0f64; TILE];
        let mut acc_sum = 0.0;
        let mut t0 = 0;
        while t0 < kq {
            let t1 = (t0 + g).min(kq);
            let ng = t1 - t0;
            for gi in 0..ng {
                let nbase = (t0 + gi) * (q + 1);
                let znode = &node_tbl[nbase..nbase + q];
                // h = √2·L_cᵀ⁻¹·z_t (the generic loop's back-solve, then its √2).
                let hh = &mut hh_g[gi];
                for r in (0..q).rev() {
                    let mut v = znode[r];
                    for cc in (r + 1)..q {
                        v -= lblk[cc * q + r] * hh[cc];
                    }
                    hh[r] = v / lblk[r * q + r];
                }
                let mut acc_c = 0.0;
                for r in 0..q {
                    hh[r] *= SQRT_2;
                    // −½(‖ũ+h‖² − ‖ũ‖²) = −h·(ũ + ½h)
                    acc_c -= hh[r] * (u[ubase + r] + 0.5 * hh[r]);
                }
                acc_g[gi] = acc_c;
                for r in 0..q {
                    let mut vv = 0.0;
                    for cc in 0..=r {
                        vv += lam[r * q + cc] * hh[cc];
                    }
                    dv_g[gi][r] = vv;
                }
            }
            // Row chunks: the whole cluster when several nodes share a tile.
            let chunk = TILE / ng;
            let mut p0 = 0;
            while p0 < m {
                let p1 = (p0 + chunk).min(m);
                let mc = p1 - p0;
                let len = ng * mc;
                for gi in 0..ng {
                    for r in 0..mc {
                        let pos = r0 + p0 + r;
                        let d = delta(pos, &dv_g[gi]);
                        dbuf[gi * mc + r] = d;
                        ebuf[gi * mc + r] = rs[pos] * d;
                    }
                }
                expm1_tile(&mut ebuf[..len]);
                if kind == Kind::Log1p {
                    for gi in 0..ng {
                        for r in 0..mc {
                            ebuf[gi * mc + r] *= rp[r0 + p0 + r];
                        }
                    }
                    log1p_tile(&mut ebuf[..len], &mut lbuf[..len]);
                }
                for (gi, a_g) in acc_g[..ng].iter_mut().enumerate() {
                    let mut a_c = *a_g;
                    for r in 0..mc {
                        let pos = r0 + p0 + r;
                        let t = gi * mc + r;
                        a_c -= ra[pos] * ebuf[t] + rh[pos] * dbuf[t];
                    }
                    *a_g = a_c;
                }
                p0 = p1;
            }
            for gi in 0..ng {
                let mut acc_c = acc_g[gi];
                if nslow[c] > 0.0 {
                    // The generic node predictor η = η_fix + z_i·Λ_p(ũ + h).
                    let hh = &hh_g[gi];
                    let mut v_cj = [0.0f64; 3];
                    for r in 0..q {
                        let mut vv = 0.0;
                        for cc in 0..=r {
                            vv += lam[r * q + cc] * (u[ubase + cc] + hh[cc]);
                        }
                        v_cj[r] = vv;
                    }
                    for pos in range.clone() {
                        if rs[pos] != 0.0 {
                            continue;
                        }
                        let i = all_rows[pos] as usize;
                        let mut eta_i = eta_fixed[i] + v_cj[0];
                        for d in 0..qz {
                            eta_i += z_buf[i * qz + d] * v_cj[d + 1];
                        }
                        acc_c -=
                            slow_term(family, nb_theta, y[i], prior_w[i], eta_i, eta[i], prob[i]);
                    }
                }
                acc_sum += (node_tbl[(t0 + gi) * (q + 1) + q] + acc_c).exp();
            }
            t0 = t1;
        }
        *sum_c = acc_sum;
    };
    #[cfg(all(feature = "parallel", not(target_arch = "wasm32")))]
    {
        use rayon::prelude::*;
        sum[..s]
            .par_iter_mut()
            .enumerate()
            .for_each(|(c, sc)| per_cluster(c, sc));
    }
    #[cfg(not(all(feature = "parallel", not(target_arch = "wasm32"))))]
    for (c, sc) in sum[..s].iter_mut().enumerate() {
        per_cluster(c, sc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::family::{dev_resid_eta, link_inv};
    use crate::spec::{NegBinomialLink, PoissonLink};

    /// The row term against the generic `½w·(dev_resid_eta(η₀+δ) − dev_resid_eta(η₀))`
    /// over a grid of y, η₀ and δ: zero counts, binomial proportions, the logit
    /// tail (`in_tail` past μ = 1 − 1e-4), and log-link rows up to the fast
    /// guard's edge (Poisson's clamp, the NB/Gamma-log tail switch). Within
    /// 1e-12 of the larger deviance, floored at 1.
    #[test]
    fn row_term_matches_dev_resid_eta() {
        let w = 1.7;
        let deltas: [f64; 11] = [
            -6.0, -2.5, -1.0, -1e-3, -1e-9, 0.0, 1e-7, 0.01, 0.7, 3.0, 6.0,
        ];
        let cases: Vec<(Family, f64, Vec<f64>, Vec<f64>)> = vec![
            (
                Family::Poisson {
                    link: PoissonLink::Log,
                },
                0.0,
                vec![0.0, 1.0, 2.0, 7.0, 40.0, 1e4],
                vec![-30.0, -5.0, -0.5, 0.0, 0.3, 2.0, 9.0, 690.0],
            ),
            (
                Family::Gamma {
                    link: GammaLink::Log,
                },
                0.0,
                vec![1e-3, 0.4, 1.0, 3.0, 250.0],
                vec![-220.0, -20.0, -1.0, 0.0, 0.8, 5.0, 220.0],
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Logit,
                },
                0.0,
                vec![0.0, 0.2, 0.5, 1.0],
                vec![-40.0, -9.5, -2.0, -0.1, 0.0, 0.1, 2.0, 9.5, 40.0],
            ),
            (
                Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
                0.7,
                vec![0.0, 1.0, 3.0, 50.0, 1e4],
                vec![-30.0, -3.0, -0.2, 0.0, 0.5, 4.0, 12.0, 220.0],
            ),
            (
                Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
                25.0,
                vec![0.0, 2.0, 30.0],
                vec![-5.0, 0.0, 3.2, 3.3, 8.0],
            ),
        ];
        let mut worst = 0.0f64;
        for (family, theta, ys, etas) in cases {
            let k = kind(family).unwrap();
            for &y in &ys {
                for &e0 in &etas {
                    let mu0 = link_inv(family, e0);
                    let [a, p, s, h] = row_consts(family, theta, y, w, e0, mu0);
                    for &d in &deltas {
                        if !is_fast(family, e0, d.abs()) {
                            continue;
                        }
                        let fast = match k {
                            Kind::Lin => term::<false>(a, p, s, h, d),
                            Kind::Log1p => term::<true>(a, p, s, h, d),
                        };
                        let d1 = dev_resid_eta(family, theta, y, e0 + d);
                        let d0 = dev_resid_eta(family, theta, y, e0);
                        let slow = 0.5 * w * (d1 - d0);
                        let scale = (0.5 * w * d1.abs().max(d0.abs())).max(1.0);
                        let err = (fast - slow).abs() / scale;
                        worst = worst.max(err);
                        assert!(
                            err <= 1e-12,
                            "{family:?} θ={theta} y={y} η₀={e0} δ={d}: fast {fast} generic {slow} (rel {err:e})"
                        );
                    }
                }
            }
        }
        eprintln!("row_term_matches_dev_resid_eta: worst scaled error {worst:e}");
    }

    /// The guard sends a row whose node range reaches the clamp or the
    /// log-link tail to the generic path, and keeps logit rows whatever η₀.
    #[test]
    fn guard_stops_short_of_clamp_and_tail() {
        let pois = Family::Poisson {
            link: PoissonLink::Log,
        };
        let gam = Family::Gamma {
            link: GammaLink::Log,
        };
        let logit = Family::Binomial {
            link: BinomialLink::Logit,
        };
        assert!(is_fast(pois, 690.0, 9.0));
        assert!(!is_fast(pois, 690.0, 10.0));
        assert!(!is_fast(pois, -695.0, 5.0));
        assert!(is_fast(gam, 220.0, 9.0));
        assert!(!is_fast(gam, -225.0, 5.0));
        assert!(is_fast(logit, 500.0, 50.0));
        assert!(!is_fast(logit, 0.0, 700.0));
        assert!(!is_fast(pois, f64::NAN, 1.0));
    }

    /// The magnitude guard: a Poisson row at η₀ = 205 (a count near e^205,
    /// whose parts `μ₀·expm1(δ)` and `y·δ` cancel to O(1) from O(√μ₀)) is a
    /// slow row; the same row at η₀ = 2 stays fast; a NaN center is slow.
    #[test]
    fn magnitude_guard_sends_huge_counts_to_the_generic_path() {
        let pois = Family::Poisson {
            link: PoissonLink::Log,
        };
        let big = 205f64.exp();
        let b = 3.0 / big.sqrt();
        assert!(
            fast_consts(pois, 0.0, big, 1.0, 205.0, big, b, mag_factor(Kind::Lin, b)).is_none()
        );
        let small = 2f64.exp();
        assert!(fast_consts(pois, 0.0, 7.0, 1.0, 2.0, small, 1.0, 1f64.exp_m1()).is_some());
        assert!(fast_consts(pois, 0.0, 7.0, 1.0, f64::NAN, small, 1.0, 1f64.exp_m1()).is_none());
    }

    /// `probit_sums` against the generic node loop (`dev_resid_eta` per row
    /// and node, minus the center `dev_resid_at`) on clusters of 1, 2, 3 and 5
    /// rows, Bernoulli and aggregated binomial (weights up to 1e4), at
    /// θ = 0, near 0 and large, with slow rows past `LN_PHI_X_MAX` in the mix;
    /// and the cluster-outer arm against the node-outer arm, to the bit.
    #[test]
    fn probit_sums_match_the_generic_node_loop() {
        let family = Family::Binomial {
            link: BinomialLink::Probit,
        };
        let sizes = [2usize, 3, 5, 1, 2, 3, 5, 5, 3, 2, 140];
        let s = sizes.len();
        let mut cluster_ids = Vec::new();
        for (c, &m) in sizes.iter().enumerate() {
            cluster_ids.extend(std::iter::repeat_n(c as u32, m));
        }
        let n = cluster_ids.len();
        // A deterministic spread of values (no RNG dependency in the crate).
        let frac = |i: usize, a: f64| ((i as f64 * a).sin() * 43758.5453).fract().abs();
        let k = 7usize;
        let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
        let nodes = &crate::consts::GH_NODES[blk..blk + k];
        let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
        let idx = ClusterRowIndex::build(&cluster_ids, s);
        let mut worst = 0.0f64;
        for aggregated in [false, true] {
            for &lambda in &[0.0, 1e-6, 0.8, 2.5] {
                let mut y = vec![0.0; n];
                let mut w = vec![1.0; n];
                let mut ef = vec![0.0; n];
                for (i, e) in ef.iter_mut().enumerate() {
                    *e = -3.0 + 6.0 * frac(i, 12.9898);
                }
                // Slow rows: centers whose node range passes 37, on both sides.
                ef[3] = 36.9;
                ef[7] = -36.95;
                ef[n - 1] = 40.0;
                for i in 0..n {
                    if aggregated {
                        // Proportions near each row's own fitted value, as at
                        // a mode, so the node sums stay finite at 1e4 trials.
                        let trials: f64 = [1.0, 3.0, 20.0, 1e4][i % 4];
                        w[i] = trials;
                        let c = cluster_ids[i] as usize;
                        let e0 = ef[i] + lambda * (-1.5 + 3.0 * frac(c + 7, 3.17));
                        let p0 = link_inv(family, e0);
                        let jitter = 0.3 * (frac(i, 78.233) - 0.5);
                        y[i] = ((p0 + jitter * (p0 * (1.0 - p0) / trials).sqrt()).clamp(0.0, 1.0)
                            * trials)
                            .round()
                            / trials;
                    } else {
                        y[i] = if frac(i, 78.233) < 0.4 { 1.0 } else { 0.0 };
                    }
                }
                let u: Vec<f64> = (0..s).map(|c| -1.5 + 3.0 * frac(c + 7, 3.17)).collect();
                let a_blocks: Vec<f64> = (0..s).map(|c| 0.6 + 2.0 * frac(c + 11, 5.3)).collect();
                let eta: Vec<f64> = (0..n)
                    .map(|i| ef[i] + lambda * u[cluster_ids[i] as usize])
                    .collect();
                let prob: Vec<f64> = eta.iter().map(|&e| link_inv(family, e)).collect();
                // Generic reference, as agq_deviance's node loop computes it.
                let mut want = vec![0.0; s];
                let mut mag = vec![0.0f64; s];
                for c in 0..s {
                    let mut ctr = -0.5 * u[c] * u[c];
                    for i in 0..n {
                        if cluster_ids[i] as usize == c {
                            ctr -= 0.5
                                * w[i]
                                * crate::family::dev_resid_at(family, 0.0, y[i], eta[i], prob[i]);
                        }
                    }
                    for (&zj, &wj) in nodes.iter().zip(wts) {
                        let u_cj = u[c] + SQRT_2 * (1.0 / a_blocks[c]) * zj;
                        let mut a = -0.5 * u_cj * u_cj;
                        for i in 0..n {
                            if cluster_ids[i] as usize == c {
                                a -= 0.5
                                    * w[i]
                                    * dev_resid_eta(family, 0.0, y[i], ef[i] + lambda * u_cj);
                            }
                        }
                        mag[c] = mag[c].max(a.abs() + ctr.abs());
                        want[c] += (wj.ln() + zj * zj + a - ctr).exp();
                    }
                }
                let run = |rows: Option<&ClusterRowIndex>| {
                    let (mut rb, mut sl) = (Vec::new(), Vec::new());
                    let (mut dl, mut acc, mut sum) = (vec![0.0; s], vec![0.0; s], vec![0.0; s]);
                    probit_sums(
                        &y,
                        &w,
                        &cluster_ids,
                        n,
                        &eta,
                        &prob,
                        &ef,
                        &u,
                        &a_blocks,
                        lambda,
                        nodes,
                        wts,
                        &mut rb,
                        &mut sl,
                        &mut dl,
                        &mut acc,
                        &mut sum,
                        rows,
                    );
                    (sum, sl)
                };
                let (got_c, slow) = run(Some(&idx));
                let (got_n, _) = run(None);
                assert!(slow.contains(&(n as u32 - 1)));
                if lambda >= 0.8 {
                    assert!(slow.contains(&3) && slow.contains(&7));
                }
                for c in 0..s {
                    assert_eq!(
                        got_c[c].to_bits(),
                        got_n[c].to_bits(),
                        "arms differ: cluster {c}, λ={lambda}, aggregated={aggregated}"
                    );
                    // The generic reference subtracts the center from the
                    // node loglik, so it carries eps times their magnitude.
                    assert!(want[c].is_finite(), "reference overflowed: cluster {c}");
                    let rel = (got_c[c] - want[c]).abs() / want[c].abs() / (1.0 + mag[c]);
                    worst = worst.max(rel);
                    assert!(
                        rel <= 1e-13,
                        "cluster {c} (size {}), λ={lambda}, aggregated={aggregated}: fast {} generic {} (rel {rel:e})",
                        sizes[c],
                        got_c[c],
                        want[c]
                    );
                }
            }
        }
        eprintln!(
            "probit_sums_match_the_generic_node_loop: worst error per unit magnitude {worst:e}"
        );
    }

    /// The probit guard: a center at 36 with a node reach of 0.9 is fast,
    /// with 1.1 it is slow, both sides; a NaN center is slow.
    #[test]
    fn probit_guard_stops_inside_the_certified_range() {
        assert!(probit_is_fast(36.0, 0.9));
        assert!(!probit_is_fast(36.0, 1.1));
        assert!(probit_is_fast(-36.0, 0.9));
        assert!(!probit_is_fast(-36.0, 1.1));
        assert!(!probit_is_fast(f64::NAN, 0.1));
    }
}
