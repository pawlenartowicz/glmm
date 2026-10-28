//! Random-effect maps between the kernel's coordinates and the caller's.
//!
//! Coordinates, in one place. The kernel holds `Λ̃`, the relative factor on the
//! fitted design `x̃` (`crate::standardize`). The caller's effects are
//! `b = N·b̃` with `N = D·C`: `C` has a unit diagonal and `C₀ᵣ = −mᵣ/sᵣ`;
//! `D = diag(1, 1/sᵣ)`. A slope column the standardization only scales (a 0/1
//! or constant column; `mᵣ = 0`) enters `D` alone. The caller's `Λ` is the
//! lower-triangular factor of `N·Λ̃·Λ̃ᵀ·Nᵀ`. When no slope of the block is
//! centered, `N` is diagonal, `N·Λ̃` is already lower triangular and no
//! factorization runs.

use faer::MatRef;

use super::{primary_lambda, LmmGroupings};
use crate::standardize::ColumnScaling;

/// Pivot floor of [`semidef_lower_chol`]. A pivot at or below it is a removed
/// rank, not a direction, and its column of `L` stays exactly zero.
#[derive(Clone, Copy)]
pub(crate) enum PivotFloor {
    /// `q·eps·tr Σ` on every column: round-off in forming Σ and in the
    /// factorization is `O(q·eps·‖Σ‖₂)` and `‖Σ‖₂ ≤ tr Σ` for PSD Σ. The floor
    /// that canonicalizes a pinned search endpoint.
    Trace,
    /// `q·eps·Σ_cc` on column `c`, keeping at most `max_rank` pivots: the
    /// coordinate maps' floor. A pivot `Σ_cc − Σₖ L_ck²` is a difference of
    /// numbers no larger than `Σ_cc`, so its rounding is `O(q·eps·Σ_cc)`. The
    /// trace floor would cut a real direction whose diagonal is small against
    /// the trace: after centering, `tr Σ` grows like `(m/s)²` while the slope's
    /// conditional variance shrinks like `(s/m)²`. The cap covers the other
    /// side: a dependent column's rounding residual of a few `eps·Σ_cc` can sit
    /// above the floor, and accepting it divides the column below by
    /// `√residual`. So when more than `max_rank` pivots pass, the surplus ones
    /// with the smallest `L_cc²/Σ_cc` (a residual is `O(eps)`, a real direction
    /// is not) are set to zero and the factorization is redone. The result is
    /// kept only if it still reproduces `Σ` (to `1e-8·tr Σ`); otherwise the
    /// uncapped factor stands.
    Column { max_rank: usize },
}

/// Lower semidefinite Cholesky `Σ = L·Lᵀ` of a PSD `Σ` given by its lower
/// triangle `sig[r·q + c]` (c ≤ r), into `l` (row-major, zeros above the
/// diagonal), under `floor`. A NaN pivot is treated as a removed rank.
pub(crate) fn semidef_lower_chol(sig: &[f64], q: usize, l: &mut [f64], floor: PivotFloor) {
    let max_rank = match floor {
        PivotFloor::Trace => q,
        PivotFloor::Column { max_rank } => max_rank,
    };
    let accepted = chol_pass(sig, q, l, floor, 0);
    if accepted <= max_rank {
        return;
    }
    // Columns past the 64th are never forced (the `pinned` mask stops there too).
    let mut forced = 0u64;
    for _ in max_rank..accepted {
        let ratio = |c: usize| l[c * q + c] * l[c * q + c] / sig[c * q + c];
        let pick = (0..q.min(64))
            .filter(|&c| l[c * q + c] > 0.0 && forced & (1 << c) == 0)
            .min_by(|&a, &b| ratio(a).total_cmp(&ratio(b)));
        if let Some(c) = pick {
            forced |= 1 << c;
        }
    }
    chol_pass(sig, q, l, floor, forced);
    let trace: f64 = (0..q).map(|r| sig[r * q + r]).sum();
    let preserved = (0..q).all(|r| {
        (0..=r).all(|c| {
            let v: f64 = (0..=c).map(|k| l[r * q + k] * l[c * q + k]).sum();
            (v - sig[r * q + c]).abs() <= 1e-8 * trace
        })
    });
    if !preserved {
        chol_pass(sig, q, l, floor, 0);
    }
}

/// One factorization pass of [`semidef_lower_chol`] that also leaves out the
/// columns whose bit is set in `forced`. Returns the number of pivots it
/// accepted.
fn chol_pass(sig: &[f64], q: usize, l: &mut [f64], floor: PivotFloor, forced: u64) -> usize {
    let eps_q = q as f64 * f64::EPSILON;
    let trace_tol = match floor {
        PivotFloor::Trace => eps_q * (0..q).map(|r| sig[r * q + r]).sum::<f64>(),
        PivotFloor::Column { .. } => 0.0,
    };
    l[..q * q].fill(0.0);
    let mut accepted = 0;
    for c in 0..q {
        let tol = match floor {
            PivotFloor::Trace => trace_tol,
            PivotFloor::Column { .. } => eps_q * sig[c * q + c],
        };
        let mut d = sig[c * q + c];
        for k in 0..c {
            d -= l[c * q + k] * l[c * q + k];
        }
        if (c < 64 && forced & (1 << c) != 0) || d <= tol || d.is_nan() {
            continue;
        }
        accepted += 1;
        let lcc = d.sqrt();
        l[c * q + c] = lcc;
        for r in (c + 1)..q {
            let mut s = sig[r * q + c];
            for k in 0..c {
                s -= l[r * q + k] * l[c * q + k];
            }
            l[r * q + c] = s / lcc;
        }
    }
    accepted
}

/// `lam` (row-major `q×q`, lower triangular or not) → the lower semidefinite
/// factor of `Σ = lam·lamᵀ` under `floor`, in place. `scratch.len() ≥ 2q²`.
pub(crate) fn refactor(q: usize, lam: &mut [f64], scratch: &mut [f64], floor: PivotFloor) {
    let (sig, l) = scratch.split_at_mut(q * q);
    for r in 0..q {
        for c in 0..=r {
            let mut s = 0.0;
            for k in 0..q {
                s += lam[r * q + k] * lam[c * q + k];
            }
            sig[r * q + c] = s;
        }
    }
    semidef_lower_chol(sig, q, l, floor);
    lam[..q * q].copy_from_slice(&l[..q * q]);
}

/// Row `r ≥ 1` of the row-major `q×k` `mat` divided by `by(r)`.
fn div_rows(mat: &mut [f64], q: usize, k: usize, by: impl Fn(usize) -> f64) {
    for r in 1..q {
        let d = by(r);
        for v in mat[r * k..(r + 1) * k].iter_mut() {
            *v /= d;
        }
    }
}

impl LmmGroupings {
    /// `(mᵣ/sᵣ, sᵣ)` of row `r` of block `b` (`b == 0` primary) under the fit's
    /// column standardization `cs`, read off the row's design column; row 0,
    /// the intercept, is left as it is.
    fn block_std(&self, b: usize, r: usize, cs: &ColumnScaling) -> (f64, f64) {
        if r == 0 {
            return (0.0, 1.0);
        }
        let c = if b == 0 {
            self.primary_slope_cols[r - 1]
        } else {
            self.extra_slope_cols[b - 1][r - 1]
        };
        (cs.shift(c), cs.scale(c))
    }

    fn block_q(&self, b: usize) -> usize {
        if b == 0 {
            self.primary_q
        } else {
            self.extra_q[b - 1]
        }
    }

    fn n_blocks(&self) -> usize {
        1 + self.extra_q.len()
    }

    pub(crate) fn block_has_shift(&self, b: usize, cs: &ColumnScaling) -> bool {
        (1..self.block_q(b)).any(|r| self.block_std(b, r, cs).0 != 0.0)
    }

    pub(crate) fn any_block_shift(&self, cs: &ColumnScaling) -> bool {
        (0..self.n_blocks()).any(|b| self.block_has_shift(b, cs))
    }

    pub(crate) fn any_column_std(&self, cs: &ColumnScaling) -> bool {
        (0..self.n_blocks()).any(|b| {
            (1..self.block_q(b)).any(|r| {
                let (sh, s) = self.block_std(b, r, cs);
                sh != 0.0 || s != 1.0
            })
        })
    }

    pub(crate) fn re_map_scratch_len(&self) -> usize {
        let q = (0..self.n_blocks())
            .map(|b| self.block_q(b))
            .max()
            .unwrap_or(1);
        3 * q * q
    }

    /// `mat ← C·mat` (`sign = −1`) or `C⁻¹·mat` (`sign = 1`): row 0 +=
    /// `sign·(mᵣ/sᵣ)·row r`.
    fn shift_rows(
        &self,
        b: usize,
        q: usize,
        k: usize,
        mat: &mut [f64],
        sign: f64,
        cs: &ColumnScaling,
    ) {
        for r in 1..q {
            let sh = sign * self.block_std(b, r, cs).0;
            for c in 0..k {
                mat[c] += sh * mat[r * k + c];
            }
        }
    }

    /// `mat ← D·mat`: row r ÷ sᵣ.
    fn scale_rows(&self, b: usize, q: usize, k: usize, mat: &mut [f64], cs: &ColumnScaling) {
        div_rows(mat, q, k, |r| self.block_std(b, r, cs).1);
    }

    /// `mat ← N·mat` for a `q×k` row-major block in the kernel's
    /// coordinates (Λ̃ for varcorr and ranef). No factorization: the product is
    /// what the covariance and the conditional modes need, and with a centered
    /// slope it is not lower triangular.
    pub(crate) fn kernel_rows_to_caller(
        &self,
        b: usize,
        q: usize,
        k: usize,
        mat: &mut [f64],
        cs: &ColumnScaling,
    ) {
        self.shift_rows(b, q, k, mat, -1.0, cs);
        self.scale_rows(b, q, k, mat, cs);
    }
}

impl LmmGroupings {
    /// Kernel Λ̃ of block `b` (row-major lower-triangular, as `primary_lambda`
    /// writes it) → the caller's Λ, in place. Steps: if a slope is centered,
    /// Λ = lower factor of C·Λ̃·Λ̃ᵀ·Cᵀ; then Λ ← D·Λ. D is applied
    /// after the factorization (a diagonal scaling commutes with it), so the
    /// column's units stay out of the rank cap's check, and the factor keeps
    /// Λ̃'s rank: as many exact-zero diagonals as Λ̃ has
    /// ([`PivotFloor::Column`]). A non-finite Λ̃ gives a non-finite Λ.
    /// `scratch.len() ≥ 2q²`.
    pub(crate) fn lambda_kernel_to_caller(
        &self,
        b: usize,
        q: usize,
        lam: &mut [f64],
        scratch: &mut [f64],
        cs: &ColumnScaling,
    ) {
        if self.block_has_shift(b, cs) {
            if (0..q).any(|r| (0..=r).any(|c| lam[r * q + c].is_nan())) {
                lam[..q * q].fill(f64::NAN);
                return;
            }
            let max_rank = q - (0..q).filter(|&r| lam[r * q + r] == 0.0).count();
            self.shift_rows(b, q, q, lam, -1.0, cs);
            refactor(q, lam, scratch, PivotFloor::Column { max_rank });
        }
        self.scale_rows(b, q, q, lam, cs);
    }

    /// The caller's Λ of block `b` → the kernel's Λ̃ on the fitted design.
    /// `N⁻¹ = C⁻¹·D⁻¹`: row r × sᵣ, then row 0 += Σᵣ (mᵣ/sᵣ)·row r, refactor.
    pub(crate) fn lambda_caller_to_design(
        &self,
        b: usize,
        q: usize,
        lam: &mut [f64],
        scratch: &mut [f64],
        cs: &ColumnScaling,
    ) {
        for r in 1..q {
            let s = self.block_std(b, r, cs).1;
            for v in lam[r * q..(r + 1) * q].iter_mut() {
                *v *= s;
            }
        }
        if self.block_has_shift(b, cs) {
            let max_rank = q - (0..q).filter(|&r| lam[r * q + r] == 0.0).count();
            self.shift_rows(b, q, q, lam, 1.0, cs);
            refactor(q, lam, scratch, PivotFloor::Column { max_rank });
        }
    }

    /// Whole kernel θ (slot order) → caller θ, per block through
    /// [`Self::lambda_kernel_to_caller`]. `scratch.len() ≥ re_map_scratch_len()`.
    pub(crate) fn theta_kernel_to_caller(
        &self,
        theta: &[f64],
        out: &mut [f64],
        scratch: &mut [f64],
        cs: &ColumnScaling,
    ) {
        let mut cur = 0usize;
        for b in 0..self.n_blocks() {
            let q = self.block_q(b);
            let len = q * (q + 1) / 2;
            if q == 1 {
                out[cur] = theta[cur];
            } else {
                let (lam, rest) = scratch.split_at_mut(q * q);
                primary_lambda(&theta[cur..], q, lam);
                self.lambda_kernel_to_caller(b, q, lam, rest, cs);
                pack_vech(lam, q, &mut out[cur..cur + len]);
            }
            cur += len;
        }
    }

    /// [`Self::theta_kernel_to_caller`] into `out`, and the `pinned` bits in
    /// the caller's coordinates when they can differ from the kernel's: some
    /// block has a centered slope. With a diagonal `N` on every block the zero
    /// pattern is the kernel's, and `None` says its own mask stands.
    pub(crate) fn theta_to_caller(
        &self,
        theta: &[f64],
        converged: bool,
        out: &mut [f64],
        scratch: &mut [f64],
        cs: &ColumnScaling,
    ) -> Option<u64> {
        self.theta_kernel_to_caller(theta, out, scratch, cs);
        self.any_block_shift(cs)
            .then(|| self.pinned_mask(out, converged))
    }

    /// [`Self::theta_to_caller`], allocating.
    pub(crate) fn theta_to_caller_vec(
        &self,
        theta: &[f64],
        converged: bool,
        cs: &ColumnScaling,
    ) -> (Vec<f64>, Option<u64>) {
        let mut out = vec![0.0; theta.len()];
        let mut scratch = vec![0.0; self.re_map_scratch_len()];
        let pins = self.theta_to_caller(theta, converged, &mut out, &mut scratch, cs);
        (out, pins)
    }

    /// Caller θ → the kernel's θ̃ on the fitted design, in place: the inverse
    /// of [`Self::theta_kernel_to_caller`], for a warm start.
    pub(crate) fn theta_caller_to_design(
        &self,
        theta: &mut [f64],
        scratch: &mut [f64],
        cs: &ColumnScaling,
    ) {
        if !self.any_column_std(cs) {
            return;
        }
        let mut cur = 0usize;
        for b in 0..self.n_blocks() {
            let q = self.block_q(b);
            let len = q * (q + 1) / 2;
            if q > 1 {
                let (lam, rest) = scratch.split_at_mut(q * q);
                primary_lambda(&theta[cur..], q, lam);
                self.lambda_caller_to_design(b, q, lam, rest, cs);
                pack_vech(lam, q, &mut theta[cur..cur + len]);
            }
            cur += len;
        }
    }

    /// `pinned` bits in the caller's coordinates: diagonal k is pinned when its
    /// caller θ entry is exactly 0.0 (the maps above leave exact zeros).
    fn pinned_mask(&self, theta_caller: &[f64], converged: bool) -> u64 {
        if !converged {
            return 0;
        }
        let mut mask = 0u64;
        for (k, &ti) in self.diagonal_theta().iter().enumerate() {
            if theta_caller[ti] == 0.0 && k < u64::BITS as usize {
                mask |= 1u64 << k;
            }
        }
        mask
    }

    /// Per block, per reported row: `‖row r of Λₖ‖`, the kernel-space
    /// standard deviation over σ. `singular`'s negligible-component check
    /// compares these (commensurate across intercept and slopes, centering
    /// included). A hidden column 0 is left out: its zero is the model's, not
    /// a component that went to zero.
    pub(crate) fn kernel_stddevs(&self, theta_kernel: &[f64]) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(self.n_blocks());
        let mut cur = 0usize;
        for b in 0..self.n_blocks() {
            let q = self.block_q(b);
            let mut lam = vec![0.0; q * q];
            primary_lambda(&theta_kernel[cur..], q, &mut lam);
            out.push(
                (usize::from(self.hidden[b])..q)
                    .map(|r| (0..=r).map(|c| lam[r * q + c].powi(2)).sum::<f64>().sqrt())
                    .collect(),
            );
            cur += q * (q + 1) / 2;
        }
        out
    }

    /// `v`, laid out like θ (each block's column-major vech, primary then each
    /// extra), without the entries of each hidden block's column 0, which are
    /// the first `q` of its vech: what is left of the block is the vech of its
    /// reported columns, the layout of `Fit::varcorr`. In place. An empty `v`
    /// (nothing of that kind reported) stays empty.
    pub(crate) fn drop_hidden(&self, v: &mut Vec<f64>) {
        if v.is_empty() || !self.hidden.contains(&true) {
            return;
        }
        let (mut cur, mut kept) = (0usize, 0usize);
        for b in 0..self.n_blocks() {
            let q = self.block_q(b);
            let skip = if self.hidden[b] { q } else { 0 };
            for i in cur + skip..cur + q * (q + 1) / 2 {
                v[kept] = v[i];
                kept += 1;
            }
            cur += q * (q + 1) / 2;
        }
        v.truncate(kept);
    }
}

impl LmmGroupings {
    /// SEs of the caller's random-effect standard deviations, from `theta` the
    /// kernel θ̃ and `cov` its covariance (`n_theta` square). Laid out like θ:
    /// row k's SE at its diagonal vech slot, NaN at the off-diagonal slots.
    /// Row k's SD is `sd_k = √Σ_kk = ‖row k of M‖`, `M = N·Λ̃`, and its SE is
    /// the delta method's, `Var(sd_k) = J·Cov·Jᵀ`. `M` is linear in θ̃: slot
    /// `a` is Λ̃'s entry `(r, c)`, `∂M/∂θ̃_a` is `N`'s column r placed in
    /// column c, so `J_a = M_kc·N_kr/sd_k`. One path for every block, centered
    /// or not, with no factorization. At `sd_k = 0` the SE is NaN, unless row k
    /// of `M` reads a single slot (a scalar block, the intercept of a block
    /// with no centered slope): `sd_k = |N_kr·θ̃_a|` has the slope `|N_kr|`
    /// from either side, and the SE is `|N_kr|·√Cov_aa` there as everywhere.
    pub(crate) fn stddev_se_kernel_to_caller(
        &self,
        theta: &[f64],
        cov: MatRef<'_, f64>,
        out: &mut [f64],
        cs: &ColumnScaling,
    ) {
        let mut cur = 0usize;
        for b in 0..self.n_blocks() {
            let q = self.block_q(b);
            let len = q * (q + 1) / 2;
            let cov = cov.submatrix(cur, cur, len, len);
            let mut m = vec![0.0; q * q];
            primary_lambda(&theta[cur..], q, &mut m);
            self.kernel_rows_to_caller(b, q, q, &mut m, cs);
            let mut n = vec![0.0; q * q];
            for r in 0..q {
                n[r * q + r] = 1.0;
            }
            self.kernel_rows_to_caller(b, q, q, &mut n, cs);
            // Vech slot a ↔ Λ̃ entry (r, c), column-major: column c, rows c..q.
            let pos: Vec<(usize, usize)> =
                (0..q).flat_map(|c| (c..q).map(move |r| (r, c))).collect();
            let mut jac = vec![0.0; len];
            out[cur..cur + len].fill(f64::NAN);
            for k in 0..q {
                let sd = (0..q).map(|c| m[k * q + c].powi(2)).sum::<f64>().sqrt();
                let slot = cur + k * q - k * k.saturating_sub(1) / 2;
                if sd == 0.0 {
                    let mut reads = pos
                        .iter()
                        .enumerate()
                        .filter(|(_, &(r, _))| n[k * q + r] != 0.0);
                    if let (Some((a, &(r, _))), None) = (reads.next(), reads.next()) {
                        out[slot] = n[k * q + r].abs() * sd_of(cov[(a, a)]);
                    }
                    continue;
                }
                for (j, &(r, c)) in jac.iter_mut().zip(&pos) {
                    *j = m[k * q + c] * n[k * q + r] / sd;
                }
                let mut v = 0.0;
                for (a, &ja) in jac.iter().enumerate().filter(|(_, &j)| j != 0.0) {
                    for (c, &jc) in jac.iter().enumerate().filter(|(_, &j)| j != 0.0) {
                        v += ja * cov[(a, c)] * jc;
                    }
                }
                out[slot] = sd_of(v);
            }
            cur += len;
        }
    }
}

/// `√v` of a variance, a rounding-negative one clamped to 0. A NaN (no θ
/// block was reported) stays NaN, which `f64::max(v, 0.0)` would turn into 0.
fn sd_of(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else {
        v.sqrt()
    }
}

/// Row-major lower-triangular `lam` → column-major vech (column c, rows c..q).
pub(crate) fn pack_vech(lam: &[f64], q: usize, out: &mut [f64]) {
    let mut t = 0;
    for c in 0..q {
        for r in c..q {
            out[t] = lam[r * q + c];
            t += 1;
        }
    }
}

#[cfg(test)]
#[path = "re_map_tests.rs"]
mod tests;
