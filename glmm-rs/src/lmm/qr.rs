//! The square-root (QR) form of the dense family kernel's profiled deviance,
//! for fits whose Gram form loses the residual in rounding.
//!
//! The Gram kernel ([`super::reml_deviance`]) reads `r²` and the `X̃′V⁻¹X̃`
//! pivots off differences of accumulated cross-products, so each carries an
//! absolute error of order `ε·y′Wy` (or `ε·C_jj`). When the residuals are tiny
//! against `y′Wy` (weights spanning many decades) that error swamps them.
//! Here the same criterion is read off orthogonal factorizations of the
//! weighted rows stacked on the penalty rows (Bates & DebRoy 2004),
//!
//! ```text
//!   [ √Ŵ·Z·Λ   √Ŵ·X̃   √Ŵ·y ]
//!   [   I        0       0  ]
//! ```
//!
//! so `r` itself comes out of rotations, known to about `ε·‖√Ŵy‖`, and the
//! pivots are products of rotations, not differences (Golub & Van Loan 2013
//! §5.1–5.3).
//!
//! **Once per accumulation** (every rescale round refills it), each primary
//! level's rows `√ŵᵢ·[z_f,i | x̃ᵢ | yᵢ]` are Givens-merged one at a time into
//! an upper triangle of width `d = w + m` (`w` the family width,
//! `m = p + 1`). Only its first `w` rows,
//! `[R_ZZ,f | R_Z·,f]`, are kept per level; what each row leaves after the
//! first `w` columns are zeroed is θ-free and goes straight into one shared
//! `m × m` triangle `R_within` (a merge into the level's own lower block and
//! then into `R_within` would give the same Gram).
//!
//! **Per θ**, each level's `w` rows `[R_ZZ,f·Λ_f | R_Z·,f]` are merged into
//! the penalty rows `[I_w | 0]`. The `w × w` triangle `S_f` they leave has
//! `S_fᵀS_f = I + Λ_fᵀZ_fᵀŴZ_fΛ_f`, so `2·Σ ln S_f,jj` is the level's share of
//! `ln|I + ΛᵀZᵀŴZΛ|` (`ln|L_ZZ|²`); the `m`-wide rows left over are merged into the tail `T`, which
//! starts as `R_within`. After every level `T` is the `[X̃ y]` factor:
//! `ln|X̃′V⁻¹X̃| = 2·Σ_{j<p} ln T_jj` and `r = T_pp`.
//!
//! Every triangle is stored packed by rows: row `j` of a `rows × d` upper
//! trapezoid holds columns `j..d`, starting at `j·(2d − j + 1)/2`.
//!
//! Bates, D. M. & DebRoy, S. (2004). Linear mixed models and penalized least
//! squares. *Journal of Multivariate Analysis* 91(1), 1–17.
//! Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.
//! Johns Hopkins.

use faer::MatRef;

use super::{primary_lambda, slope_value, Criterion, LmmFitScratch, LmmGroupings, LmmSuffStats};

/// The compressed rows of one accumulation and the per-evaluation scratch,
/// sized once by [`QrKernel::new`]; nothing allocates per evaluation.
pub(crate) struct QrKernel {
    /// The deviance dispatches here while set (`LmmKernel::deviance`).
    pub(crate) on: bool,
    /// Per primary level, the first `w` rows of its `d`-wide triangle,
    /// packed: `[R_ZZ,f | R_Z·,f]`, [`Self::level_len`] numbers each.
    rz: Vec<f64>,
    /// `R_within`, `m × m`, packed.
    r_within: Vec<f64>,
    /// Per-evaluation: one level's `S_f` with its `m` right columns (`w`
    /// rows of a `d`-wide triangle, packed), the tail `T` (`m × m`, packed),
    /// and one row of width `d`.
    s: Vec<f64>,
    t: Vec<f64>,
    row: Vec<f64>,
    w: usize,
    m: usize,
}

/// Offset of row `j` in a packed upper trapezoid of width `d`.
#[inline]
fn row_start(j: usize, d: usize) -> usize {
    j * (2 * d - j + 1) / 2
}

/// Givens-merge `row` (length `d`) into the first `rows` rows of the packed
/// upper trapezoid `tri`: on return `row[..rows]` is zero and the rest of it is
/// what the rotations left. Each diagonal stays `≥ 0` (it becomes a `hypot`).
fn merge_row(tri: &mut [f64], d: usize, rows: usize, row: &mut [f64]) {
    let mut off = 0;
    for j in 0..rows {
        let b = row[j];
        if b != 0.0 {
            let r = &mut tri[off..off + d - j];
            let a = r[0];
            let rho = a.hypot(b);
            let (c, s) = (a / rho, b / rho);
            r[0] = rho;
            row[j] = 0.0;
            for (x, y) in r[1..].iter_mut().zip(&mut row[j + 1..d]) {
                let (xv, yv) = (*x, *y);
                *x = c * xv + s * yv;
                *y = c * yv - s * xv;
            }
        }
        off += d - j;
    }
}

impl QrKernel {
    pub(crate) fn new(g: &LmmGroupings, p: usize) -> Self {
        let w = g.primary_q + g.nested_per_parent;
        let m = p + 1;
        let lvl = row_start(w, w + m);
        let tri = row_start(m, m);
        QrKernel {
            on: false,
            rz: vec![0.0; g.n_primary * lvl],
            r_within: vec![0.0; tri],
            s: vec![0.0; lvl],
            t: vec![0.0; tri],
            row: vec![0.0; w + m],
            w,
            m,
        }
    }

    /// `y ← y − X̃b` on the stored triangles, the old `y` column saved into
    /// `saved`. Each triangle is `Qᵀ` times its rows `[z x̃ y]`, so its `y`
    /// column is `Qᵀy` and moves by its `X̃` columns times `b`, exactly as the
    /// rows would.
    pub(crate) fn shift_response(&mut self, b: &[f64], saved: &mut Vec<f64>) {
        saved.clear();
        self.y_entries(|x, k0, y| {
            saved.push(*y);
            *y -= x.iter().zip(&b[k0..]).map(|(a, c)| a * c).sum::<f64>();
        });
    }

    /// Undo [`Self::shift_response`] bit for bit.
    pub(crate) fn restore_response(&mut self, saved: &[f64]) {
        let mut old = saved.iter();
        self.y_entries(|_, _, y| *y = *old.next().expect("one saved entry per y entry"));
    }

    /// The stored `y` column entry by entry, each with the `X̃` part of its
    /// row and the index of that part's first column: every level's `w` rows,
    /// then `R_within`'s `m` rows.
    fn y_entries(&mut self, mut each: impl FnMut(&[f64], usize, &mut f64)) {
        let (w, m) = (self.w, self.m);
        let (p, d) = (m - 1, w + m);
        let lvl = self.level_len();
        for level in self.rz.chunks_exact_mut(lvl) {
            for j in 0..w {
                let r = &mut level[row_start(j, d)..row_start(j + 1, d)];
                let (x, y) = r[w - j..].split_at_mut(p);
                each(x, 0, &mut y[0]);
            }
        }
        for j in 0..m {
            let r = &mut self.r_within[row_start(j, m)..row_start(j + 1, m)];
            let (x, y) = r.split_at_mut(p - j);
            each(x, j, &mut y[0]);
        }
    }

    /// The stored `y` column, in [`Self::y_entries`] order.
    #[cfg(test)]
    pub(crate) fn y_column(&mut self) -> Vec<f64> {
        let mut v = Vec::new();
        self.y_entries(|_, _, y| v.push(*y));
        v
    }

    /// Numbers kept per level: `w(w+1)/2 + w·m`.
    fn level_len(&self) -> usize {
        row_start(self.w, self.w + self.m)
    }

    /// Compress the rows the Gram statistics were accumulated from. `x̃` goes
    /// through the `U` of `suff.design_qr`, so both kernels share one `U`
    /// (`det U = 1`, no correction) and the factor maps back the same way; slope
    /// values are read from the raw rows as the Gram reads them (`slope_value`).
    /// `weights` is ŵ, `None` unit weight.
    #[allow(clippy::too_many_arguments)] // the accumulator's own row surface
    pub(crate) fn fill(
        &mut self,
        suff: &LmmSuffStats,
        x: MatRef<'_, f64>,
        z: MatRef<'_, f64>,
        y: &[f64],
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        weights: Option<&[f64]>,
    ) {
        let g = &suff.groupings;
        let (w, m) = (self.w, self.m);
        let (p, d, q, np) = (m - 1, w + m, g.primary_q, g.nested_per_parent);
        let lvl = self.level_len();
        let nested = g.nested.map(|nf| &extra_ids[nf.decl][..]);
        self.rz.fill(0.0);
        self.r_within.fill(0.0);
        let row = &mut self.row;
        for i in 0..y.len() {
            let sw = weights.map_or(1.0, |w| w[i].sqrt());
            let f = cluster_ids[i] as usize;
            row[..w].fill(0.0);
            row[0] = sw;
            for (k, &sc) in g.primary_slope_cols.iter().enumerate() {
                row[1 + k] = sw * slope_value(x, z, p, i, sc);
            }
            if let Some(ids) = nested {
                // Child id = parent·n_per + within (`LmmGroupings`).
                let c = ids[i] as usize - f * np;
                debug_assert!(c < np);
                row[q + c] = sw;
            }
            for j in 0..p {
                row[w + j] = x[(i, j)];
            }
            suff.design_qr.solve_row(&mut row[w..w + p]);
            for v in &mut row[w..w + p] {
                *v *= sw;
            }
            row[d - 1] = sw * y[i];
            merge_row(&mut self.rz[f * lvl..(f + 1) * lvl], d, w, row);
            merge_row(&mut self.r_within, m, m, &mut row[w..]);
        }
    }

    /// Level `f`'s `S_f` and its right columns into `self.s`, at the relative
    /// factor `Λ_f = diag(Λ_p, θ_n·I)` (`prim_lam` the row-major `Λ_p` when
    /// `q_p > 1`, else `th_p`); with `tail`, the leftover rows are merged into
    /// `self.t`. Returns `ln|S_f|`.
    fn reduce_level(
        &mut self,
        f: usize,
        q: usize,
        prim_lam: &[f64],
        th_p: f64,
        th_n: f64,
        tail: bool,
    ) -> f64 {
        let (w, m) = (self.w, self.m);
        let d = w + m;
        let lvl = self.level_len();
        let lam = |k: usize, c: usize| if q == 1 { th_p } else { prim_lam[k * q + c] };
        let rz = &self.rz[f * lvl..(f + 1) * lvl];
        self.s.fill(0.0);
        for j in 0..w {
            self.s[row_start(j, d)] = 1.0;
        }
        for r in 0..w {
            // Row r of [R_ZZ,f·Λ_f | R_Z·,f]; `rr[k − r]` is R[r][k].
            let rr = &rz[row_start(r, d)..row_start(r + 1, d)];
            let row = &mut self.row;
            for (c, v) in row[..w].iter_mut().enumerate() {
                *v = if c < q {
                    // Λ_p is lower triangular: k runs from max(r, c) to q − 1.
                    (r.max(c)..q).map(|k| rr[k - r] * lam(k, c)).sum()
                } else if c >= r {
                    rr[c - r] * th_n
                } else {
                    0.0
                };
            }
            row[w..].copy_from_slice(&rr[w - r..]);
            merge_row(&mut self.s, d, w, row);
            if tail {
                merge_row(&mut self.t, m, m, &mut row[w..]);
            }
        }
        (0..w).map(|j| self.s[row_start(j, d)].ln()).sum()
    }

    /// Level `f`'s `S_ff = Z_fᵀV_f⁻¹Z_f` (`w × w`) into `s` and `T_f =
    /// Z_fᵀV_f⁻¹[X̃ r]` (`w × m`) into `t`, row-major, at `Λ_f` as
    /// [`Self::reduce_level`] reads it, for the exact Satterthwaite pass; `r =
    /// y − X̃β̃` with `ev = [−β̃; 1]`. With `R = R_ZZ,f` and `R_y = R_Z·,f`
    /// (`Z_fᵀŴZ_f = RᵀR`, `Z_fᵀŴ[X̃ y] = RᵀR_y`, every other row of the level
    /// orthogonal to `Z_f`) and `R̃` the triangle with `R̃ᵀR̃ = I + RΛ_fΛ_fᵀRᵀ`,
    /// left by merging the rows of `(RΛ_f)ᵀ` into `I_w`,
    /// `Z_fᵀV_f⁻¹Z_f = Rᵀ(I + RΛ_fΛ_fᵀRᵀ)⁻¹R = YᵀY` with `Y = R̃⁻ᵀR`, and `T_f =
    /// Yᵀ·R̃⁻ᵀ[R_y,X̃ R_y·ev]`: rotations, triangular solves and sums of
    /// products, with no subtraction of Gram terms (Golub & Van Loan 2013
    /// §5.1–5.3). The residual column is contracted with `ev` first. `yz`
    /// (`w × (w + m)`) is scratch. `false` on a non-positive pivot of `R̃`.
    #[allow(clippy::too_many_arguments)] // `reduce_level`'s inputs and the outputs
    pub(crate) fn family_inputs(
        &mut self,
        f: usize,
        q: usize,
        prim_lam: &[f64],
        th_p: f64,
        th_n: f64,
        ev: &[f64],
        yz: &mut [f64],
        s: &mut [f64],
        t: &mut [f64],
    ) -> bool {
        let (w, m) = (self.w, self.m);
        let (p, d) = (m - 1, w + m);
        let lvl = self.level_len();
        let lam = |k: usize, c: usize| if q == 1 { th_p } else { prim_lam[k * q + c] };
        let rz = &self.rz[f * lvl..(f + 1) * lvl];
        // A = R·Λ_f (w × w at stride w), as `reduce_level` forms its rows.
        for r in 0..w {
            let rr = &rz[row_start(r, d)..row_start(r + 1, d)];
            for c in 0..w {
                yz[r * w + c] = if c < q {
                    (r.max(c)..q).map(|k| rr[k - r] * lam(k, c)).sum()
                } else if c >= r {
                    rr[c - r] * th_n
                } else {
                    0.0
                };
            }
        }
        // R̃, packed with width w, from the columns of A merged into I_w.
        let tri = &mut self.s[..row_start(w, w)];
        tri.fill(0.0);
        for j in 0..w {
            tri[row_start(j, w)] = 1.0;
        }
        for i in 0..w {
            let row = &mut self.row[..w];
            for (k, v) in row.iter_mut().enumerate() {
                *v = yz[k * w + i];
            }
            merge_row(tri, w, w, row);
        }
        // [Y | Z] = R̃⁻ᵀ[R | R_y,X̃ R_y·ev] by forward substitution on R̃ᵀ.
        for i in 0..w {
            let rr = &rz[row_start(i, d)..row_start(i + 1, d)];
            let o = i * d;
            for c in 0..w {
                yz[o + c] = if c >= i { rr[c - i] } else { 0.0 };
            }
            yz[o + w..o + w + p].copy_from_slice(&rr[w - i..w - i + p]);
            yz[o + w + p] = (0..m).map(|j| rr[w - i + j] * ev[j]).sum();
            for k in 0..i {
                let rki = tri[row_start(k, w) + i - k];
                for c in 0..d {
                    yz[o + c] -= rki * yz[k * d + c];
                }
            }
            let rii = tri[row_start(i, w)];
            if !(rii.is_finite() && rii > 0.0) {
                return false;
            }
            for v in &mut yz[o..o + d] {
                *v /= rii;
            }
        }
        for a in 0..w {
            for b in 0..w {
                s[a * w + b] = (0..w).map(|i| yz[i * d + a] * yz[i * d + b]).sum();
            }
            for j in 0..m {
                t[a * m + j] = (0..w).map(|i| yz[i * d + a] * yz[i * d + w + j]).sum();
            }
        }
        true
    }
}

/// The profiled deviance at `theta` under `fit.criterion` on the QR kernel
/// `fit.qr` holds, the same criterion [`super::reml_deviance`] returns. Writes
/// `fit.factor = Tᵀ` (diagonal `≥ 0` by construction, so `TᵀT` is the Gram's
/// `[X̃ y]` block and every reader of the factor runs unchanged) and
/// `fit.sigma_sq = r²/ν`. `+∞` when `n ≤ p`, `p = 0`, a `T_jj` is zero or not
/// finite, or `r = 0`. There is no exact-fit floor here: `r` is computed, not
/// left over from a subtraction.
pub(crate) fn qr_deviance(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch) -> f64 {
    let g = &suff.groupings;
    let m = suff.m;
    let p = m - 1;
    if suff.n_rows <= p || p == 0 {
        return f64::INFINITY;
    }
    let LmmFitScratch {
        qr,
        prim_lam,
        factor,
        sigma_sq,
        criterion,
        ..
    } = fit;
    let qr = qr
        .as_deref_mut()
        .expect("qr_deviance: the QR kernel is built");
    let q = g.primary_q;
    if q > 1 {
        primary_lambda(theta, q, prim_lam);
    }
    let th_n = g.nested.map_or(0.0, |nf| theta[nf.vech_start]);
    qr.t.copy_from_slice(&qr.r_within);
    let mut log_lzz = 0.0;
    for f in 0..g.n_primary {
        log_lzz += qr.reduce_level(f, q, prim_lam, theta[0], th_n, true);
    }
    let t = &qr.t;
    let mut log_lxx = 0.0;
    for j in 0..p {
        let tjj = t[row_start(j, m)];
        if !(tjj.is_finite() && tjj > 0.0) {
            return f64::INFINITY;
        }
        log_lxx += tjj.ln();
    }
    let r = t[row_start(p, m)];
    let nu = criterion.nu(suff.n_rows, p) as f64;
    let s2 = r * r / nu;
    if !(r.is_finite() && r > 0.0 && s2.is_finite() && s2 > 0.0) {
        return f64::INFINITY;
    }
    for i in 0..m {
        let ti = &t[row_start(i, m)..row_start(i + 1, m)];
        for j in 0..m {
            factor[(j, i)] = if j >= i { ti[j - i] } else { 0.0 };
        }
    }
    *sigma_sq = s2;
    match criterion {
        Criterion::Reml => 2.0 * log_lzz + 2.0 * log_lxx + nu * s2.ln(),
        Criterion::Ml => 2.0 * log_lzz + nu * s2.ln(),
    }
}

/// The spherical conditional modes `û` at θ̂ into `fit.ranef_u`, setting
/// `fit.ranef_ok`; `betas` holds β̃ (on `X̃`). Per level the rotated block is
/// rebuilt at θ̂ (nothing is kept between evaluations): its top rows
/// `[S_f | K_f]` satisfy `S_fᵀK_f = Λ_fᵀZ_fᵀŴ[X̃ y]`, so `K_f` is the Gram
/// recovery's `L_f⁻¹B_f` and `û_f` solves `S_f·û_f = K_f,y − K_f,X̃·β̃` by back
/// substitution. The layout is `recover_ranef_family`'s: primary component d
/// at `d·n_primary + f`, child c at `q_p·n_primary + f·n_per + c`.
pub(crate) fn qr_recover_ranef(
    theta: &[f64],
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch,
    betas: &[f64],
) {
    let g = &suff.groupings;
    fit.ranef_ok = false;
    let k = g.k_total;
    if k == 0 || suff.n_rows == 0 {
        return;
    }
    let LmmFitScratch {
        qr,
        prim_lam,
        ranef_u,
        ranef_rhs: u,
        ranef_ok,
        ..
    } = fit;
    let qr = qr
        .as_deref_mut()
        .expect("qr_recover_ranef: the QR kernel is built");
    let (q, np, n_prim) = (g.primary_q, g.nested_per_parent, g.n_primary);
    let (w, m) = (qr.w, qr.m);
    let (p, d) = (m - 1, w + m);
    if q > 1 {
        primary_lambda(theta, q, prim_lam);
    }
    let th_n = g.nested.map_or(0.0, |nf| theta[nf.vech_start]);
    ranef_u[..k].fill(0.0);
    for f in 0..n_prim {
        qr.reduce_level(f, q, prim_lam, theta[0], th_n, false);
        let s = &qr.s;
        for r in (0..w).rev() {
            let sr = &s[row_start(r, d)..row_start(r + 1, d)];
            // `sr[c − r]` is row r's column c: S_f for c < w, K_f after.
            let kx = &sr[w - r..];
            let mut acc = kx[p] - (0..p).map(|j| kx[j] * betas[j]).sum::<f64>();
            for i in r + 1..w {
                acc -= sr[i - r] * u[i];
            }
            let srr = sr[0];
            if !(srr.is_finite() && srr > 0.0) {
                return;
            }
            u[r] = acc / srr;
        }
        for (r, &v) in u.iter().enumerate().take(q) {
            ranef_u[r * n_prim + f] = v;
        }
        for c in 0..np {
            ranef_u[q * n_prim + f * np + c] = u[q + c];
        }
    }
    *ranef_ok = true;
}
