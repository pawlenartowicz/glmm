//! CR2 and HC2: the cluster-robust covariance of a Gaussian fit's β̂ and the
//! Pustejovsky–Tipton df of a contrast (spec 2-dcr §3.2).
//!
//! Cluster k has the working model Φ_k = diag(λ) + U Uᵀ, H_k = X_kᵀΦ_k⁻¹X_k,
//! M = (Σ_k H_k)⁻¹ and C_k = Φ_k − X_k M X_kᵀ. A_k solves A C_k A = Φ_k on the
//! directions where T_k = D⁻ᵀ C_k D⁻¹ (Φ_k = DᵀD) has an eigenvalue above
//! [`CUT`], and is 0 on the rest (Pustejovsky and Tipton 2018, eq. 7–8);
//! E_k = M X_kᵀ Φ_k⁻¹ A_k. A cluster stores E_k r_k, F_k = E_k X_k and
//! E_kΦ_kE_kᵀ = M H_k M + F_k M F_kᵀ; the CR2 variance and the df of any
//! contrast come from those alone.
//!
//! Outside the small, well-conditioned clusters of [`DENSE`] no n_k × n_k
//! matrix is formed. With R_s = Φ + s²Φ⁻¹ and
//! G_s = M⁻¹ − Xᵀ R_s⁻¹ X, A = I + (2/π) ∫₀^∞ R_s⁻¹ X G_s⁻¹ Xᵀ R_s⁻¹ ds, taken
//! as a fixed sum over N rational poles (Hale, Higham and Trefethen 2008,
//! method 3) with N set from a bound on the spectrum at relative tolerance
//! [`POLE_TOL`], and no stop test. A pole costs one weighted cross-product of
//! [X, r, U] and small solves. Each column of [X, r] is split once into a part
//! Λ⁻¹-orthogonal to U plus U B, which keeps those cross-products free of
//! cancellation under strong random effects. A cluster that alone identifies a
//! direction of β (T_k has an eigenvalue at the cut: leverage 1) takes a
//! deflated sum and stores those null directions. A one-row cluster has a
//! scalar A_k and takes its closed form, which is HC2 when U is empty.
//!
//! Port of `validation/tools/cr2/centered.py` (`split`, `base_T`, `node_T`,
//! `quad_cluster`), `lib.py` (`nodes`, `N_for`, `spectrum`) and
//! `rank_check.py` (`reduce_u`).

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::{factor, solve as lu_solve};
use faer::linalg::matmul::matmul;
use faer::traits::{math_utils::one, ComplexField};
use faer::{c64, Accum, Conj, Mat, MatRef, Par};

use crate::inference::contest::qform;
use crate::inference::evd::sym_eigen;
use crate::linalg::SeqLlt;

/// The cut on T_k's eigenvalues, which have no units and lie in [0, 1].
const CUT: f64 = 1e-12;
/// df below which a robust test has no assurance (spec 2-dcr §6).
pub(crate) const DF_SMALL: f64 = 4.0;
/// A cluster of at most `DENSE.0` rows whose spectrum bounds have
/// zmax / zmin ≤ `DENSE.1` takes [`dense_cluster`], any other the pole sum
/// ([`pole_cluster`]). Dense is 2.4 to 4.8 times as fast up to 12 rows and
/// 2.8 times as slow at 20; its error grows with zmax / zmin (9e-14 at 2e5,
/// 6e-11 at 4e8), reaching 1e-12 near 4e6 (DCR-T7, `docs/0.9.0/dcr-timing.md`).
const DENSE: (usize, f64) = (12, 1e6);
/// Relative tolerance of the rational approximation of z^(−1/2), which sets N.
// The spec's 1e-10 left G6 at 1.15e-10; 1e-13 meets G6, rank and leverage at their measured 4e-13, at ~28% more poles.
const POLE_TOL: f64 = 1e-13;

/// One cluster at the final fit, in the kernel's coordinates:
/// Φ_k = diag(`lambda`) + `u` `u`ᵀ.
pub(crate) struct ClusterBlock<'a> {
    /// X_k, n_k × p.
    pub(crate) x: MatRef<'a, f64>,
    /// The marginal residuals r_k = y_k − X_k β̂.
    pub(crate) r: &'a [f64],
    /// The diagonal of Φ_k's residual part.
    pub(crate) lambda: &'a [f64],
    /// n_k × q, of any column rank (q = 0 for none).
    pub(crate) u: MatRef<'a, f64>,
}

/// What a fit keeps of CR2: per cluster E_k r_k, F_k and E_kΦ_kE_kᵀ.
pub(crate) struct Cr2Pieces {
    /// E_k r_k, column k for cluster k (p × G).
    pub(crate) er: Mat<f64>,
    /// (F_k, E_kΦ_kE_kᵀ) of each cluster of more than one row, in cluster order.
    pub(crate) whole: Vec<(Mat<f64>, Mat<f64>)>,
    /// One-row clusters in cluster order, a column each: E_k (p × 1), so
    /// F_k = e_k x_kᵀ and E_kΦ_kE_kᵀ = φ_k e_k e_kᵀ, with x_k in `row_x` and
    /// the scalar Φ_k in `row_phi`. O(p) per row.
    pub(crate) row_e: Mat<f64>,
    pub(crate) row_x: Mat<f64>,
    pub(crate) row_phi: Vec<f64>,
    /// (k, n): H_(−k) n = 0 with nᵀM⁻¹n = 1, so a contrast with cᵀn ≠ 0 has no
    /// information outside cluster k.
    pub(crate) null: Vec<(usize, Vec<f64>)>,
}

/// The CR2 pieces of `blocks` at M = `m` = (Σ_k X_kᵀΦ_k⁻¹X_k)⁻¹, or `None`
/// when a λ is not finite and > 0 (an exact fit has σ̂² = 0), `m` is not
/// finite and positive definite, or the spectrum leaves f64's range.
pub(crate) fn cr2_pieces(blocks: &[ClusterBlock<'_>], m: MatRef<'_, f64>) -> Option<Cr2Pieces> {
    cr2_pieces_with(blocks, m, DENSE)
}

/// [`cr2_pieces`] with `dense` in place of [`DENSE`].
pub(crate) fn cr2_pieces_with(
    blocks: &[ClusterBlock<'_>],
    m: MatRef<'_, f64>,
    dense: (usize, f64),
) -> Option<Cr2Pieces> {
    let (lo, hi) = blocks
        .iter()
        .flat_map(|b| b.lambda)
        .fold((f64::INFINITY, 0.0f64), |(a, b), &l| (a.min(l), b.max(l)));
    if !(lo > 0.0 && hi.is_finite()) {
        return None;
    }
    // A_k is the same for Φ_k/s, so the pieces run on Φ_k/s and M/s, s = 4^k
    // (exact) near the middle of λ's range, and E Φ Eᵀ, φ and n are scaled
    // back: the pole bounds then do not depend on the scale of y.
    let r = 2f64.powi(((lo.log2() + hi.log2()) / 4.0).round() as i32);
    let s = r * r;
    let lam: Vec<Vec<f64>> = blocks
        .iter()
        .map(|b| b.lambda.iter().map(|l| l / s).collect())
        .collect();
    let u: Vec<Mat<f64>> = blocks
        .iter()
        .map(|b| Mat::from_fn(b.u.nrows(), b.u.ncols(), |i, j| b.u[(i, j)] / r))
        .collect();
    let scaled: Vec<ClusterBlock<'_>> = blocks
        .iter()
        .zip(&lam)
        .zip(&u)
        .map(|((b, lambda), u)| ClusterBlock {
            x: b.x,
            r: b.r,
            lambda,
            u: u.as_ref(),
        })
        .collect();
    let ms = Mat::from_fn(m.nrows(), m.ncols(), |i, j| m[(i, j)] / s);
    let mut pc = scaled_pieces(&scaled, ms.as_ref(), dense)?;
    for (_, epe) in &mut pc.whole {
        *epe *= faer::Scale(s);
    }
    pc.row_phi.iter_mut().for_each(|v| *v *= s);
    for (_, nd) in &mut pc.null {
        nd.iter_mut().for_each(|v| *v *= r);
    }
    Some(pc)
}

/// [`cr2_pieces`] on Φ_k and M of order 1.
fn scaled_pieces(
    blocks: &[ClusterBlock<'_>],
    m: MatRef<'_, f64>,
    (dense_rows, dense_ratio): (usize, f64),
) -> Option<Cr2Pieces> {
    let p = m.nrows();
    let g = blocks.len();
    if !(0..p).all(|j| m.col(j).iter().all(|v| v.is_finite())) {
        return None;
    }
    let lm = SeqLlt::new(m)?.L().to_owned();
    // Clusters of more than one row: U reduced to full column rank, and H_k.
    let multi: Vec<Option<(Mat<f64>, Mat<f64>)>> = blocks
        .iter()
        .map(|b| {
            if b.r.len() == 1 {
                return Some(None);
            }
            let u = reduce_u(b.lambda, b.u)?;
            let h = sym(phi_cross(b.lambda, u.as_ref(), b.x).as_ref());
            Some(Some((u, h)))
        })
        .collect::<Option<_>>()?;
    let add_h = |acc: &mut Mat<f64>, k: usize| match &multi[k] {
        Some((_, h)) => add_scaled(acc, h, 1.0),
        None => {
            let (x, phi) = one_row(&blocks[k]);
            for j in 0..p {
                for i in 0..p {
                    acc[(i, j)] += x[i] * x[j] / phi;
                }
            }
        }
    };
    // H_(−k) as a prefix plus a suffix sum, so it has no subtraction in it.
    let mut suffix = Vec::new();
    let mut run = Mat::zeros(p, p);
    for k in (0..g).rev() {
        if multi[k].is_some() {
            suffix.push(run.clone());
        }
        add_h(&mut run, k);
    }
    let mut prefix = Mat::<f64>::zeros(p, p);
    let mut er = Vec::with_capacity(p * g);
    let mut whole = Vec::new();
    let (mut row_e, mut row_x, mut row_phi) = (Vec::new(), Vec::new(), Vec::new());
    let mut null = Vec::new();
    for (k, b) in blocks.iter().enumerate() {
        match &multi[k] {
            Some((u, h)) => {
                let mut hmk = suffix.pop().expect("one suffix per cluster");
                add_scaled(&mut hmk, &prefix, 1.0);
                let sp = spectrum(b.lambda, u, &hmk, lm.as_ref())?;
                let c = if b.r.len() <= dense_rows && sp.zmax / sp.zmin <= dense_ratio {
                    dense_cluster(b, u, sp, m)?
                } else {
                    pole_cluster(b, (u, h, &hmk), sp, m)?
                };
                er.extend(c.er);
                whole.push((c.f, c.epe));
                null.extend(
                    (0..c.vnull.ncols()).map(|j| (k, c.vnull.col(j).iter().copied().collect())),
                );
            }
            None => {
                let (x, phi) = one_row(b);
                let mx: Vec<f64> = (0..p)
                    .map(|i| (0..p).map(|j| m[(i, j)] * x[j]).sum())
                    .collect();
                let xmx: f64 = x.iter().zip(&mx).map(|(a, b)| a * b).sum();
                // A_k = T_k^(−1/2) with the scalar T_k = 1 − x_kᵀ M x_k / φ_k.
                let t = 1.0 - xmx / phi;
                let a = if t > CUT {
                    t.sqrt().recip()
                } else {
                    null.push((k, mx.iter().map(|v| v / xmx.sqrt()).collect()));
                    0.0
                };
                let e: Vec<f64> = mx.iter().map(|v| v * a / phi).collect();
                er.extend(e.iter().map(|v| v * b.r[0]));
                row_e.extend(e);
                row_x.extend(x);
                row_phi.push(phi);
            }
        }
        add_h(&mut prefix, k);
    }
    let cols = |v: &[f64]| MatRef::from_column_major_slice(v, p, v.len() / p.max(1)).to_owned();
    Some(Cr2Pieces {
        er: cols(&er),
        whole,
        row_e: cols(&row_e),
        row_x: cols(&row_x),
        row_phi,
        null,
    })
}

impl Cr2Pieces {
    /// The pieces on `p` columns, row and column `kept[j]` taking this record's
    /// `j` and the rest zero: the record of a fit that dropped the other
    /// columns. Row `x_k` is a design row, scattered the same way.
    pub(crate) fn widen(self, kept: &[usize], p: usize) -> Cr2Pieces {
        let rows = |a: Mat<f64>| {
            let mut out = Mat::zeros(p, a.ncols());
            for (r, &o) in kept.iter().enumerate() {
                for k in 0..a.ncols() {
                    out[(o, k)] = a[(r, k)];
                }
            }
            out
        };
        let square = |a: Mat<f64>| {
            let mut out = Mat::zeros(p, p);
            for (r, &o) in kept.iter().enumerate() {
                for (s, &q) in kept.iter().enumerate() {
                    out[(o, q)] = a[(r, s)];
                }
            }
            out
        };
        Cr2Pieces {
            er: rows(self.er),
            whole: self
                .whole
                .into_iter()
                .map(|(f, epe)| (square(f), square(epe)))
                .collect(),
            row_e: rows(self.row_e),
            row_x: rows(self.row_x),
            row_phi: self.row_phi,
            null: self
                .null
                .into_iter()
                .map(|(k, n)| {
                    let mut out = vec![0.0; p];
                    for (r, &o) in kept.iter().enumerate() {
                        out[o] = n[r];
                    }
                    (k, out)
                })
                .collect(),
        }
    }

    /// The CR2 covariance of β̂, Σ_k E_k r_k r_kᵀ E_kᵀ.
    pub(crate) fn vcov(&self) -> Mat<f64> {
        mul(self.er.as_ref(), self.er.transpose())
    }

    /// cᵀ [`Self::vcov`] c, Σ_k (cᵀE_k r_k)².
    pub(crate) fn var(&self, c: &[f64]) -> f64 {
        (0..self.er.ncols())
            .map(|k| {
                let a: f64 = c.iter().enumerate().map(|(i, v)| v * self.er[(i, k)]).sum();
                a * a
            })
            .sum()
    }

    /// The Pustejovsky–Tipton df of the contrast `c`, (tr Q)² / tr(Q²), at the
    /// fit's M = `m`.
    pub(crate) fn df(&self, m: MatRef<'_, f64>, c: &[f64]) -> f64 {
        // Q is quadratic in c and the df free of its scale: c taken (exactly,
        // by a power of 2) to cᵀMc near 1 keeps (tr Q)² in range whatever the
        // scale of y.
        let cmc = qform(c, m);
        let k = if cmc > 0.0 && cmc.is_finite() {
            (cmc.log2() / 2.0).round() as i32
        } else {
            0
        };
        let c: Vec<f64> = c.iter().map(|v| v * 2f64.powi(-k)).collect();
        let (t, t2) = self.q_traces(m, &c);
        t * t / t2
    }

    /// (tr Q, tr Q²) for Q = diag(u_kᵀΦ_k u_k) − U M Uᵀ, U's rows f_kᵀ = cᵀF_k,
    /// by the low-rank form: tr(U M Uᵀ U M Uᵀ) = tr(M W M W), W = Σ_k f_k f_kᵀ.
    fn q_traces(&self, m: MatRef<'_, f64>, c: &[f64]) -> (f64, f64) {
        let p = c.len();
        let mut w = Mat::<f64>::zeros(p, p);
        let (mut sq, mut sq2, mut sfmf, mut sqfmf) = (0.0, 0.0, 0.0, 0.0);
        let mut add = |f: &[f64], q: f64| {
            let fmf = qform(f, m);
            sq += q;
            sq2 += q * q;
            sfmf += fmf;
            sqfmf += q * fmf;
            for j in 0..p {
                for i in 0..p {
                    w[(i, j)] += f[i] * f[j];
                }
            }
        };
        let mut f = vec![0.0; p];
        for (fk, epe) in &self.whole {
            for (j, fj) in f.iter_mut().enumerate() {
                *fj = (0..p).map(|i| fk[(i, j)] * c[i]).sum();
            }
            add(&f, qform(c, epe.as_ref()));
        }
        for (k, phi) in self.row_phi.iter().enumerate() {
            let a: f64 = (0..p).map(|i| self.row_e[(i, k)] * c[i]).sum();
            for (j, fj) in f.iter_mut().enumerate() {
                *fj = self.row_x[(j, k)] * a;
            }
            add(&f, phi * a * a);
        }
        let mw = mul(m, w.as_ref());
        let trmwmw: f64 = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| mw[(i, j)] * mw[(j, i)])
            .sum();
        (sq - sfmf, sq2 - 2.0 * sqfmf + trmwmw)
    }

    /// Whether one cluster alone identifies the contrast `c`: cᵀn ≠ 0 for a
    /// null direction n, read as (cᵀn)² > [`CUT`] · cᵀMc, the share of cᵀMc
    /// along n (at most 1 by Cauchy–Schwarz, and free of β's coordinates).
    pub(crate) fn identified_by_one(&self, m: MatRef<'_, f64>, c: &[f64]) -> bool {
        let cmc = qform(c, m);
        self.null.iter().any(|(_, n)| {
            let cn: f64 = c.iter().zip(n).map(|(a, b)| a * b).sum();
            cn * cn > CUT * cmc
        })
    }
}

/// A multi-row cluster's pieces, and its null directions as columns.
struct Pole {
    er: Vec<f64>,
    f: Mat<f64>,
    epe: Mat<f64>,
    vnull: Mat<f64>,
}

/// The bounds of the spectrum the poles cover, and the null directions.
struct Spectrum {
    vnull: Mat<f64>,
    zmin: f64,
    zmax: f64,
}

/// `lib.py::spectrum`: T_k's eigenvalues that differ from 1 are those of
/// Lᵀ H_(−k) L (M = L Lᵀ), a p×p problem with no cancellation; the ones at or
/// below the cut give the null directions L w. `None` when a bound or their
/// ratio is not finite and > 0.
fn spectrum(lam: &[f64], u: &Mat<f64>, hmk: &Mat<f64>, lm: MatRef<'_, f64>) -> Option<Spectrum> {
    let p = lm.nrows();
    let s = sym(mul(mul(lm.transpose(), hmk.as_ref()).as_ref(), lm).as_ref());
    let eig = sym_eigen(s.as_ref(), true).ok()?;
    let w = eig.vectors.expect("vectors asked for");
    let dead: Vec<usize> = (0..p).filter(|&i| eig.values[i] <= CUT).collect();
    let wd = Mat::from_fn(p, dead.len(), |i, j| w[(i, dead[j])]);
    let tmin = eig
        .values
        .iter()
        .filter(|&&t| t > CUT)
        .fold(1.0f64, |a, &t| a.min(t));
    let lo = lam.iter().fold(f64::INFINITY, |a, &l| a.min(l));
    let uu = if u.ncols() > 0 {
        let e = sym_eigen(mul(u.transpose(), u.as_ref()).as_ref(), false).ok()?;
        e.values[u.ncols() - 1]
    } else {
        0.0
    };
    let hi = lam.iter().fold(0.0f64, |a, &l| a.max(l)) + uu;
    let (zmin, zmax) = (lo * lo * tmin, hi * hi);
    (zmin > 0.0 && (zmax / zmin).is_finite()).then(|| Spectrum {
        vnull: mul(lm, wd.as_ref()),
        zmin,
        zmax,
    })
}

/// `centered.py::quad_cluster`, with the deflated branch of
/// `lib.py::quad_cluster_deflated`.
/// `(u, h, hmk)` are the reduced U, H_k and H_(−k), `sp` the cluster's
/// [`spectrum`].
fn pole_cluster(
    b: &ClusterBlock<'_>,
    (u, h, hmk): (&Mat<f64>, &Mat<f64>, &Mat<f64>),
    sp: Spectrum,
    m: MatRef<'_, f64>,
) -> Option<Pole> {
    let (n, p) = (b.r.len(), b.x.ncols());
    let lam = b.lambda;
    let (s, w) = nodes(sp.zmin, sp.zmax, n_poles(sp.zmax / sp.zmin));
    let d = sp.vnull.ncols();
    let mut lx = h.clone();
    let mut htil = h.clone();
    let lr = if d == 0 {
        let y = Mat::from_fn(n, p + 1, |i, j| if j < p { b.x[(i, j)] } else { b.r[i] });
        let (yt, bm, sm) = split(lam, u.as_ref(), y.as_ref());
        let t0 = base_t(lam, &yt, bm.as_ref(), sm.as_ref());
        let mut lr: Vec<f64> = (0..p).map(|i| t0[(i, p)]).collect();
        for (&sj, &wj) in s.iter().zip(&w) {
            let (tr, ti) = node_t(lam, sj, u, &yt, bm.as_ref());
            let gs = Mat::from_fn(p, p, |i, j| hmk[(i, j)] + (h[(i, j)] - tr[(i, j)]));
            let sol = solve(gs.as_ref(), tr.as_ref().subrows(0, p));
            let pm = Mat::from_fn(p, p, |i, j| ti[(i, j)] / sj);
            accumulate(&mut lx, &mut lr, &mul(pm.as_ref(), sol.as_ref()), wj);
        }
        lr
    } else {
        // With c = Φ⁻¹X V (C c = 0): shift by s0 on those directions, take the
        // same sum with y = Φ⁻¹c as d more columns, and remove the known term,
        // A = Ã − s0^(−1/2) c (cᵀΦ⁻¹c)⁻¹ cᵀ.
        let pl = p + d;
        let cc = phi_solve(lam, u.as_ref(), mul(b.x, sp.vnull.as_ref()).as_ref());
        let yv = phi_solve(lam, u.as_ref(), cc.as_ref());
        let sc = sym(mul(cc.transpose(), yv.as_ref()).as_ref());
        let s0 = sp.zmax;
        let y = Mat::from_fn(n, pl + 1, |i, j| match j {
            j if j < p => b.x[(i, j)],
            j if j < pl => yv[(i, j - p)],
            _ => b.r[i],
        });
        let (yt, bm, sm) = split(lam, u.as_ref(), y.as_ref());
        let t0 = base_t(lam, &yt, bm.as_ref(), sm.as_ref());
        let mut lr: Vec<f64> = (0..p).map(|i| t0[(i, pl)]).collect();
        for (&sj, &wj) in s.iter().zip(&w) {
            let (tr, ti) = node_t(lam, sj, u, &yt, bm.as_ref());
            let k = Mat::from_fn(pl, pl, |i, j| match (i < p, j < p) {
                (true, true) => hmk[(i, j)] + (h[(i, j)] - tr[(i, j)]),
                (false, false) => -tr[(i, j)] - sc[(i - p, j - p)] / s0,
                _ => -tr[(i, j)],
            });
            let rhs = Mat::from_fn(pl, p + 1, |i, j| tr[(i, if j < p { j } else { pl })]);
            let pm = Mat::from_fn(p, pl, |i, j| ti[(i, j)] / sj);
            let sol = solve(k.as_ref(), rhs.as_ref());
            accumulate(&mut lx, &mut lr, &mul(pm.as_ref(), sol.as_ref()), wj);
        }
        let pc = mul(phi_solve(lam, u.as_ref(), b.x).transpose(), cc.as_ref());
        let pcsi = mul(
            pc.as_ref(),
            solve(sc.as_ref(), Mat::<f64>::identity(d, d).as_ref()).as_ref(),
        );
        let ccr: Vec<f64> = (0..d)
            .map(|j| cc.col(j).iter().zip(b.r).map(|(a, b)| a * b).sum())
            .collect();
        let f0 = s0.sqrt().recip();
        add_scaled(
            &mut lx,
            &mul(pcsi.as_ref(), mul(cc.transpose(), b.x).as_ref()),
            -f0,
        );
        for (i, v) in lr.iter_mut().enumerate() {
            *v -= f0 * (0..d).map(|j| pcsi[(i, j)] * ccr[j]).sum::<f64>();
        }
        add_scaled(&mut htil, &mul(pcsi.as_ref(), pc.transpose()), -1.0);
        lr
    };
    let f = mul(m, lx.as_ref());
    let mut epe = mul(mul(m, htil.as_ref()).as_ref(), m);
    add_scaled(
        &mut epe,
        &mul(mul(f.as_ref(), m).as_ref(), f.transpose()),
        1.0,
    );
    Some(Pole {
        er: (0..p)
            .map(|i| (0..p).map(|j| m[(i, j)] * lr[j]).sum())
            .collect(),
        f,
        epe,
        vnull: sp.vnull,
    })
}

/// `lib.py::dense_cluster`, O(n_k³): Φ_k = L Lᵀ, B = Lᵀ C_k L and
/// A_k = L B^(−1/2) Lᵀ on B's n_k − d largest eigenvalues, d the null
/// directions in `sp`. `u` as [`pole_cluster`]; `None` when Φ_k or B has no
/// factorization.
fn dense_cluster(
    b: &ClusterBlock<'_>,
    u: &Mat<f64>,
    sp: Spectrum,
    m: MatRef<'_, f64>,
) -> Option<Pole> {
    let (n, p) = (b.r.len(), b.x.ncols());
    let uut = mul(u.as_ref(), u.transpose());
    let phi = Mat::from_fn(n, n, |i, j| {
        uut[(i, j)] + if i == j { b.lambda[i] } else { 0.0 }
    });
    let llt = SeqLlt::new(phi.as_ref())?;
    let l = llt.L().to_owned();
    let xmx = mul(mul(b.x, m).as_ref(), b.x.transpose());
    let cm = Mat::from_fn(n, n, |i, j| phi[(i, j)] - xmx[(i, j)]);
    let eb = sym_eigen(
        sym(mul(mul(l.transpose(), cm.as_ref()).as_ref(), l.as_ref()).as_ref()).as_ref(),
        true,
    )
    .ok()?;
    let v = eb.vectors.expect("vectors asked for");
    let keep = sp.vnull.ncols()..n;
    let bmh = Mat::from_fn(n, n, |i, j| {
        keep.clone()
            .map(|k| v[(i, k)] * v[(j, k)] / eb.values[k].sqrt())
            .sum()
    });
    let a = mul(mul(l.as_ref(), bmh.as_ref()).as_ref(), l.transpose());
    let mut px = b.x.to_owned();
    llt.solve_in_place(px.as_mut());
    let e = mul(mul(m, px.transpose()).as_ref(), a.as_ref());
    Some(Pole {
        er: (0..p)
            .map(|i| (0..n).map(|j| e[(i, j)] * b.r[j]).sum())
            .collect(),
        f: mul(e.as_ref(), b.x),
        epe: mul(mul(e.as_ref(), phi.as_ref()).as_ref(), e.transpose()),
        vnull: sp.vnull,
    })
}

/// `lx += w · cor[:, :p]`, `lr += w · cor[:, p]`.
fn accumulate(lx: &mut Mat<f64>, lr: &mut [f64], cor: &Mat<f64>, w: f64) {
    let p = lr.len();
    for j in 0..p {
        for i in 0..p {
            lx[(i, j)] += w * cor[(i, j)];
        }
    }
    for (i, v) in lr.iter_mut().enumerate() {
        *v += w * cor[(i, p)];
    }
}

/// A one-row cluster's x_k and scalar Φ_k = λ + ‖u‖².
fn one_row(b: &ClusterBlock<'_>) -> (Vec<f64>, f64) {
    let x = b.x.row(0).iter().copied().collect();
    let uu: f64 = b.u.row(0).iter().map(|v| v * v).sum();
    (x, b.lambda[0] + uu)
}

/// `rank_check.py::reduce_u`: U V over the eigenvectors of S = UᵀΛ⁻¹U with an
/// eigenvalue above ε · max(1, λmax(S)). A dropped direction with eigenvalue 0
/// has U v = 0, so U Uᵀ does not change. `None` when S is not finite.
fn reduce_u(lam: &[f64], u: MatRef<'_, f64>) -> Option<Mat<f64>> {
    let q = u.ncols();
    if q == 0 {
        return Some(Mat::zeros(lam.len(), 0));
    }
    let ul = Mat::from_fn(lam.len(), q, |i, j| u[(i, j)] / lam[i]);
    let s = sym(mul(u.transpose(), ul.as_ref()).as_ref());
    let eig = sym_eigen(s.as_ref(), true).ok()?;
    let v = eig.vectors.expect("vectors asked for");
    let cut = f64::EPSILON * eig.values[q - 1].max(1.0);
    let keep: Vec<usize> = (0..q).filter(|&i| eig.values[i] > cut).collect();
    Some(mul(
        u,
        Mat::from_fn(q, keep.len(), |i, j| v[(i, keep[j])]).as_ref(),
    ))
}

/// `centered.py::split`: Y = Ỹ + U B with ỸᵀΛ⁻¹U = 0, in two passes so the
/// orthogonality holds to rounding; returns (Ỹ, B, S = UᵀΛ⁻¹U).
fn split(lam: &[f64], u: MatRef<'_, f64>, y: MatRef<'_, f64>) -> (Mat<f64>, Mat<f64>, Mat<f64>) {
    let (n, q, k) = (lam.len(), u.ncols(), y.ncols());
    if q == 0 {
        return (y.to_owned(), Mat::zeros(0, k), Mat::zeros(0, 0));
    }
    let ul = Mat::from_fn(n, q, |i, j| u[(i, j)] / lam[i]);
    let s = mul(ul.transpose(), u);
    let b = solve(s.as_ref(), mul(ul.transpose(), y).as_ref());
    let ub = mul(u, b.as_ref());
    let yt = Mat::from_fn(n, k, |i, j| y[(i, j)] - ub[(i, j)]);
    let db = solve(s.as_ref(), mul(ul.transpose(), yt.as_ref()).as_ref());
    let udb = mul(u, db.as_ref());
    (
        Mat::from_fn(n, k, |i, j| yt[(i, j)] - udb[(i, j)]),
        Mat::from_fn(q, k, |i, j| b[(i, j)] + db[(i, j)]),
        s,
    )
}

/// `centered.py::base_T`: YᵀΦ⁻¹Y = ỸᵀΛ⁻¹Ỹ + Bᵀ (I + S)⁻¹ S B.
fn base_t(lam: &[f64], yt: &Mat<f64>, b: MatRef<'_, f64>, s: MatRef<'_, f64>) -> Mat<f64> {
    let ytl = Mat::from_fn(yt.nrows(), yt.ncols(), |i, j| yt[(i, j)] / lam[i]);
    let mut t = mul(yt.transpose(), ytl.as_ref());
    let q = s.nrows();
    if q > 0 {
        let sf = solve(eye_plus(s).as_ref(), s);
        add_scaled(
            &mut t,
            &mul(mul(b.transpose(), sf.as_ref()).as_ref(), b),
            1.0,
        );
    }
    t
}

/// YᵀΦ⁻¹Y through the split.
fn phi_cross(lam: &[f64], u: MatRef<'_, f64>, y: MatRef<'_, f64>) -> Mat<f64> {
    let (yt, b, s) = split(lam, u, y);
    base_t(lam, &yt, b.as_ref(), s.as_ref())
}

/// Φ⁻¹Z = Λ⁻¹Z̃ + Λ⁻¹U (I + S)⁻¹ B through the split.
fn phi_solve(lam: &[f64], u: MatRef<'_, f64>, z: MatRef<'_, f64>) -> Mat<f64> {
    let (zt, b, s) = split(lam, u, z);
    let ub = if u.ncols() > 0 {
        mul(u, solve(eye_plus(s.as_ref()).as_ref(), b.as_ref()).as_ref())
    } else {
        Mat::zeros(lam.len(), z.ncols())
    };
    Mat::from_fn(lam.len(), z.ncols(), |i, j| {
        (zt[(i, j)] + ub[(i, j)]) / lam[i]
    })
}

/// `centered.py::node_T`: Yᵀ(Φ − i s)⁻¹Y as (real, imaginary) from one
/// weighted cross-product of [Ỹ, U]. With Ω = (Λ − i s)⁻¹, K = I + UᵀΩU and
/// Tᵤ = K⁻¹UᵀΩỸ: ỸᵀΩỸ − ỸᵀΩU Tᵤ + Tᵤᵀ B + Bᵀ Tᵤ + Bᵀ K⁻¹UᵀΩU B.
fn node_t(
    lam: &[f64],
    s: f64,
    u: &Mat<f64>,
    yt: &Mat<f64>,
    b: MatRef<'_, f64>,
) -> (Mat<f64>, Mat<f64>) {
    let (m, q) = (yt.ncols(), u.ncols());
    let (wre, wim): (Vec<f64>, Vec<f64>) = lam
        .iter()
        .map(|&l| {
            let den = l * l + s * s;
            (l / den, s / den)
        })
        .unzip();
    let cols: Vec<&[f64]> = (0..m)
        .map(|j| yt.col_as_slice(j))
        .chain((0..q).map(|j| u.col_as_slice(j)))
        .collect();
    let (gr, gi) = grams(&cols, &wre, &wim);
    if q == 0 {
        return (gr, gi);
    }
    let gc = Mat::from_fn(m + q, m + q, |i, j| c64::new(gr[(i, j)], gi[(i, j)]));
    let gtt = gc.as_ref().submatrix(0, 0, m, m);
    let gtu = gc.as_ref().submatrix(0, m, m, q);
    let guu = gc.as_ref().submatrix(m, m, q, q);
    let k = Mat::from_fn(q, q, |i, j| {
        guu[(i, j)] + c64::new(f64::from(u8::from(i == j)), 0.0)
    });
    let tu = solve(k.as_ref(), gtu.transpose());
    let tuu = solve(k.as_ref(), guu);
    let bc = Mat::from_fn(q, m, |i, j| c64::new(b[(i, j)], 0.0));
    let a1 = mul(gtu, tu.as_ref());
    let a2 = mul(tu.transpose(), bc.as_ref());
    let a3 = mul(bc.transpose(), mul(tuu.as_ref(), bc.as_ref()).as_ref());
    let t = Mat::from_fn(m, m, |i, j| {
        gtt[(i, j)] - a1[(i, j)] + a2[(i, j)] + a2[(j, i)] + a3[(i, j)]
    });
    (
        Mat::from_fn(m, m, |i, j| t[(i, j)].re),
        Mat::from_fn(m, m, |i, j| t[(i, j)].im),
    )
}

/// (Σ_r w_re,r y_r y_rᵀ, Σ_r w_im,r y_r y_rᵀ) over the rows y_r of `cols`.
fn grams(cols: &[&[f64]], wre: &[f64], wim: &[f64]) -> (Mat<f64>, Mat<f64>) {
    let k = cols.len();
    let mut gr = Mat::zeros(k, k);
    let mut gi = Mat::zeros(k, k);
    for (j, cj) in cols.iter().enumerate() {
        for (i, ci) in cols.iter().enumerate().skip(j) {
            let (mut a, mut b) = (0.0, 0.0);
            for (((&yi, &yj), &wr), &wi) in ci.iter().zip(*cj).zip(wre).zip(wim) {
                let v = yi * yj;
                a += v * wr;
                b += v * wi;
            }
            gr[(i, j)] = a;
            gr[(j, i)] = a;
            gi[(i, j)] = b;
            gi[(j, i)] = b;
        }
    }
    (gr, gi)
}

// ---- the rational approximation of z^(−1/2) on [zmin, zmax] (Zolotarev;
// Hale, Higham and Trefethen 2008, method 3), after `lib.py` ----

/// K(k) from the complementary modulus `kc`, by the AGM.
fn agm_k(kc: f64) -> f64 {
    let (mut a, mut b) = (1.0f64, kc);
    for _ in 0..64 {
        if (a - b).abs() <= 1e-15 * a {
            break;
        }
        (a, b) = (0.5 * (a + b), (a * b).sqrt());
    }
    std::f64::consts::PI / (2.0 * a)
}

/// (sc(u), dn(u)) for 0 < u ≤ K/2 from the complementary modulus `kc`
/// (Bulirsch's recurrence; it takes kc itself, so kc² = 1e-36 loses nothing).
fn sc_dn(u: f64, kc: f64) -> (f64, f64) {
    let (mut a, mut emc) = (1.0f64, kc * kc);
    let (mut em, mut en) = (Vec::new(), Vec::new());
    let mut c = 1.0;
    for _ in 0..40 {
        em.push(a);
        emc = emc.sqrt();
        en.push(emc);
        c = 0.5 * (a + emc);
        if (a - emc).abs() <= 1e-9 * a {
            break;
        }
        emc *= a;
        a = c;
    }
    let (sn, cn) = (u * c).sin_cos();
    a = cn / sn;
    c *= a;
    let mut dn = 1.0;
    for (&b, &e) in em.iter().zip(&en).rev() {
        a *= c;
        c *= dn;
        dn = (e + a) / (b + a);
        a = c / b;
    }
    (c.recip(), dn)
}

/// Nodes s_j and weights w_j with z^(−1/2) ≈ Σ_j w_j / (z + s_j²) on
/// [zmin, zmax]: (2/π) ∫₀^∞ ds / (z + s²) at s = √zmin sc(u | k′) by the
/// midpoint rule in u, the upper half mirrored in log s about √(zmin zmax).
fn nodes(zmin: f64, zmax: f64, n: usize) -> (Vec<f64>, Vec<f64>) {
    let kc = (zmin / zmax).sqrt();
    let h = agm_k(kc) / n as f64;
    let pref = 2.0 * h * zmin.sqrt() / std::f64::consts::PI;
    let (mut s, mut w) = (vec![0.0; n], vec![0.0; n]);
    for (j, (sj, wj)) in s
        .iter_mut()
        .zip(w.iter_mut())
        .enumerate()
        .take(n.div_ceil(2))
    {
        let (sc, dn) = sc_dn((j as f64 + 0.5) * h, kc);
        *sj = zmin.sqrt() * sc;
        *wj = pref * dn * (1.0 + sc * sc);
    }
    let cc = (zmin * zmax).sqrt();
    for j in 0..n / 2 {
        let i = n - 1 - j;
        s[i] = cc / s[j];
        w[i] = w[j] * cc / (s[j] * s[j]);
    }
    (s, w)
}

/// Poles for relative error [`POLE_TOL`] on a spectrum of width `ratio` (HHT eq. 4.7,
/// with `lib.py`'s constant).
fn n_poles(ratio: f64) -> usize {
    let pi2 = std::f64::consts::PI * std::f64::consts::PI;
    ((4.0 / POLE_TOL).ln() * (ratio.ln() + 2.78) / (2.0 * pi2)).ceil() as usize + 1
}

// ---- small dense helpers, at `Par::Seq` (see `linalg::SeqLlt` for why) ----

fn sym(a: MatRef<'_, f64>) -> Mat<f64> {
    Mat::from_fn(a.nrows(), a.ncols(), |i, j| 0.5 * (a[(i, j)] + a[(j, i)]))
}

fn eye_plus(s: MatRef<'_, f64>) -> Mat<f64> {
    Mat::from_fn(s.nrows(), s.ncols(), |i, j| {
        s[(i, j)] + f64::from(u8::from(i == j))
    })
}

/// `acc += s · b`.
fn add_scaled(acc: &mut Mat<f64>, b: &Mat<f64>, s: f64) {
    for j in 0..acc.ncols() {
        for i in 0..acc.nrows() {
            acc[(i, j)] += s * b[(i, j)];
        }
    }
}

fn mul<T: ComplexField>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let mut c = Mat::<T>::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, one::<T>(), Par::Seq);
    c
}

/// A⁻¹B by LU with partial pivoting.
fn solve<T: ComplexField>(a: MatRef<'_, T>, b: MatRef<'_, T>) -> Mat<T> {
    let n = a.nrows();
    let mut lu = a.to_owned();
    let mut x = b.to_owned();
    let (mut perm, mut perm_inv) = (vec![0usize; n], vec![0usize; n]);
    let req = factor::lu_in_place_scratch::<usize, T>(n, n, Par::Seq, Default::default()).or(
        lu_solve::solve_in_place_scratch::<usize, T>(n, x.ncols(), Par::Seq),
    );
    let mut buf = MemBuffer::new(req);
    let stack = MemStack::new(&mut buf);
    let (_, perm) = factor::lu_in_place(
        lu.as_mut(),
        &mut perm,
        &mut perm_inv,
        Par::Seq,
        stack,
        Default::default(),
    );
    lu_solve::solve_in_place_with_conj(
        lu.as_ref(),
        lu.as_ref(),
        perm,
        Conj::No,
        x.as_mut(),
        Par::Seq,
        stack,
    );
    x
}

#[cfg(test)]
#[path = "cr2_tests.rs"]
mod tests;
