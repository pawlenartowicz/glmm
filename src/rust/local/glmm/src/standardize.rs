//! Internal standardization of the fixed-effect design.
//!
//! Every fit runs on `x̃ᵢⱼ = (xᵢⱼ − mⱼ)/sⱼ` and reports in the caller's units.
//! With `η = Σⱼ x̃ᵢⱼ β̃ⱼ = Σⱼ xᵢⱼ βⱼ`, the caller's coefficients are `β = Bβ̃`,
//! where `B` is the identity except `Bⱼⱼ = 1/sⱼ` and, on the intercept row,
//! `B₀ⱼ = −mⱼ/sⱼ`. The transform is a preconditioner: any invertible choice
//! fits the same model, and nothing downstream assumes `x̃` has mean 0 or
//! standard deviation 1.
//!
//! Per column, with prior weights `wᵢ` (1 when absent):
//! `mⱼ = Σwᵢxᵢⱼ/Σwᵢ` when the design has an intercept, else `0`;
//! `sⱼ = √(Σwᵢ(xᵢⱼ − mⱼ)²/Σwᵢ)`. Left as they are (`mⱼ = 0`, `sⱼ = 1`): the
//! intercept (the first column whose entries are all exactly `1.0`), every
//! column whose entries are all exactly `0.0` or `1.0`, every constant column
//! of a design with an intercept, and every column whose `sⱼ` is zero or not
//! finite. One exception: a 0/1 or constant column that a random-effect block
//! reads as a slope is scaled but not centered (`mⱼ = 0`, so `sⱼ` is its root
//! mean square about zero and its zeros stay exact zeros). The random-effect
//! search then runs on comparable coordinates for every slope: its relative
//! factor's row for that slope is in units of the slope's root mean square.
//!
//! Centering is exact only where an intercept absorbs the constant `−mⱼβⱼ` it
//! adds: in `X` that is the intercept column; in a random-effect block that
//! uses column `j` as a slope it is the block's own random intercept, and only
//! when the slope is in the same unstructured term as that intercept. A slope
//! in any other term (`(x || g)`, or a block without a random intercept) is
//! therefore scaled about zero and not centered, as a 0/1 slope is; a column
//! that is such a slope in one grouping is not centered in any.
//!
//! A random-effect-only column (`Design::re_columns`, slope index `p + j`) is
//! never centered and always divided by its prior-weighted root mean square
//! about zero, the rule a slope column of `X` gets on a design fitted as
//! given. Dividing a column of `Z` is an exact reparametrization of every
//! block, whatever its terms, and leaves `X` and `β` untouched.

use faer::MatMut;

/// One draw's column statistics and the fixed-effect map `β = Bβ̃` they
/// define. Every route reads it by reference; [`ColumnScaling::IDENTITY`] is
/// the map of a design fitted as given, on which every method is exact (no
/// arithmetic on the values).
#[derive(Clone, Debug)]
pub(crate) struct ColumnScaling {
    center: Vec<f64>,
    scale: Vec<f64>,
    shift: Vec<f64>,
    /// [`Self::compute`]'s per-column flags (`FLAG_*`), sized `p` with the rest
    /// so a draw allocates nothing.
    flags: Vec<u8>,
    /// Columns a random-effect block reads as a slope ([`Self::with_re_slopes`]),
    /// fixed at build.
    re_slope: Vec<bool>,
    /// The slope columns that must not be centered ([`Self::with_re_slopes`]),
    /// fixed at build.
    re_uncentered: Vec<bool>,
    /// `sⱼ` of random-effect-only column `j` (slope index `p + j`),
    /// [`Self::compute_re_only`]'s; empty when the design has none.
    re_only_scale: Vec<f64>,
    intercept: Option<usize>,
    identity: bool,
}

/// Every entry is exactly `0.0` or `1.0`.
const FLAG_BINARY: u8 = 1;
/// Every entry is exactly `1.0`.
const FLAG_ONES: u8 = 2;
/// Every entry equals the first row's.
const FLAG_CONSTANT: u8 = 4;

impl ColumnScaling {
    /// The identity map, of any width.
    pub(crate) const IDENTITY: &'static ColumnScaling = &ColumnScaling {
        center: Vec::new(),
        scale: Vec::new(),
        shift: Vec::new(),
        flags: Vec::new(),
        re_slope: Vec::new(),
        re_uncentered: Vec::new(),
        re_only_scale: Vec::new(),
        intercept: None,
        identity: true,
    };

    pub(crate) fn identity(p: usize) -> Self {
        ColumnScaling {
            center: vec![0.0; p],
            scale: vec![1.0; p],
            shift: vec![0.0; p],
            flags: vec![0; p],
            re_slope: vec![false; p],
            re_uncentered: vec![false; p],
            re_only_scale: Vec::new(),
            intercept: None,
            identity: true,
        }
    }

    /// The identity map of width `p`, with `slopes`, the columns the
    /// random-effect blocks read as slopes, and `uncentered`, the slope
    /// columns that are in no unstructured term with their grouping's random
    /// intercept, marked for [`Self::compute`]. A column in both lists (a
    /// slope in two groupings, one of which vetoes) is not centered.
    pub(crate) fn with_re_slopes(
        p: usize,
        slopes: impl IntoIterator<Item = usize>,
        uncentered: impl IntoIterator<Item = usize>,
    ) -> Self {
        let mut cs = Self::identity(p);
        for c in slopes {
            cs.re_slope[c] = true;
        }
        for c in uncentered {
            cs.re_slope[c] = true;
            cs.re_uncentered[c] = true;
        }
        cs
    }

    pub(crate) fn set_identity(&mut self) {
        self.center.fill(0.0);
        self.scale.fill(1.0);
        self.shift.fill(0.0);
        self.intercept = None;
        self.identity = true;
    }

    /// This draw's statistics. `x` is row-major `n×p`; `weights` are the prior
    /// weights (binomial trial counts as they are). With `standardize` false
    /// every column is left as it is except the random-effect slopes, which are
    /// scaled as a 0/1 slope column is (the design fitted as given,
    /// `FitWorkspace::set_prestandardized`).
    ///
    /// The flags first, one column at a time: each test stops at the first row
    /// that fails it. Then two passes over the rows, each accumulating every
    /// column in ascending row order: `Σwᵢxᵢⱼ` first, then `Σwᵢ(xᵢⱼ − mⱼ)²`.
    /// Without prior weights the passes add `xᵢⱼ` and `(xᵢⱼ − mⱼ)²` and take
    /// `Σwᵢ = n`: the same values, since `1·v` is exact and `n < 2⁵³`.
    /// `center` and `shift` hold the two sums until the last loop turns them
    /// into `mⱼ`, `sⱼ`.
    pub(crate) fn compute(
        &mut self,
        x: &[f64],
        n: usize,
        p: usize,
        weights: Option<&[f64]>,
        standardize: bool,
    ) {
        debug_assert_eq!(self.scale.len(), p);
        self.set_identity();
        if n == 0 || p == 0 || !(standardize || self.re_slope.contains(&true)) {
            return;
        }
        let x = &x[..n * p];
        let weights = weights.map(|w| &w[..n]);
        for (j, flag) in self.flags.iter_mut().enumerate() {
            let col = || x[j..].iter().step_by(p);
            let first = x[j];
            *flag = if col().all(|&v| v == 1.0) {
                FLAG_BINARY | FLAG_ONES | FLAG_CONSTANT
            } else {
                // `v·(v − 1)` is exactly 0 for `v` = ±0 and 1 and for no other
                // double: a tiny `v` gives `−v` exactly, and near 1 `v − 1` is
                // exact and at least 2⁻⁵³ in size; NaN and ±∞ fail. It is one
                // comparison per row without a branch. `v == 0.0 || v == 1.0`
                // compiles to two branches, which mispredict on random 0/1 data.
                let binary = col().all(|&v| v * (v - 1.0) == 0.0);
                let constant = col().all(|&v| v == first);
                (u8::from(binary) * FLAG_BINARY) | (u8::from(constant) * FLAG_CONSTANT)
            };
        }
        let (center, shift) = (&mut self.center[..p], &mut self.shift[..p]);
        let sw = match weights {
            None => {
                for row in x.chunks_exact(p) {
                    for (c, &v) in center.iter_mut().zip(row) {
                        *c += v;
                    }
                }
                n as f64
            }
            Some(w) => {
                let mut sw = 0.0;
                for (row, &wi) in x.chunks_exact(p).zip(w) {
                    sw += wi;
                    for (c, &v) in center.iter_mut().zip(row) {
                        *c += wi * v;
                    }
                }
                sw
            }
        };
        self.intercept = self.flags.iter().position(|&f| f & FLAG_ONES != 0);
        for (j, c) in center.iter_mut().enumerate() {
            // Intercept, 0/1 columns and any further all-ones column: as they
            // are. A constant column next to an intercept has no spread to
            // scale; its computed `sⱼ` would be the round-off of `mⱼ` alone.
            // Either kind read as a random-effect slope is scaled about zero,
            // and so is a slope that must not be centered.
            let kept = !standardize
                || self.flags[j] & FLAG_BINARY != 0
                || (self.intercept.is_some() && self.flags[j] & FLAG_CONSTANT != 0)
                || self.re_uncentered[j];
            let scaled = !kept || (self.re_slope[j] && Some(j) != self.intercept);
            self.flags[j] = u8::from(scaled);
            *c = if kept || self.intercept.is_none() {
                0.0
            } else {
                *c / sw
            };
        }
        match weights {
            None => {
                for row in x.chunks_exact(p) {
                    for ((s, &m), &v) in shift.iter_mut().zip(&*center).zip(row) {
                        let d = v - m;
                        *s += d * d;
                    }
                }
            }
            Some(w) => {
                for (row, &wi) in x.chunks_exact(p).zip(w) {
                    for ((s, &m), &v) in shift.iter_mut().zip(&*center).zip(row) {
                        let d = v - m;
                        *s += wi * d * d;
                    }
                }
            }
        }
        for j in 0..p {
            let (m, ss) = (self.center[j], self.shift[j]);
            self.center[j] = 0.0;
            self.shift[j] = 0.0;
            if self.flags[j] == 0 {
                continue;
            }
            let s = (ss / sw).sqrt();
            if !(s > 0.0 && s.is_finite() && m.is_finite()) {
                continue; // degenerate: as it is
            }
            self.center[j] = m;
            self.scale[j] = s;
            self.shift[j] = m / s;
            self.identity = false;
        }
    }

    /// This draw's statistics of the random-effect-only columns, row-major
    /// `n×s` `z`, after [`Self::compute`] (which resets the map): each column
    /// gets `sⱼ = √(Σwᵢzᵢⱼ²/Σwᵢ)` and no center, accumulated in ascending row
    /// order as `compute` accumulates a column it does not center. A column
    /// whose `sⱼ` is zero or not finite is left as it is.
    pub(crate) fn compute_re_only(
        &mut self,
        z: &[f64],
        n: usize,
        s: usize,
        weights: Option<&[f64]>,
    ) {
        self.re_only_scale.clear();
        self.re_only_scale.resize(s, 0.0);
        if s == 0 {
            return;
        }
        let ss = &mut self.re_only_scale;
        let mut sw = 0.0;
        for (i, row) in z[..n * s].chunks_exact(s).enumerate() {
            let wi = weights.map_or(1.0, |w| w[i]);
            sw += wi;
            for (a, &v) in ss.iter_mut().zip(row) {
                *a += if weights.is_some() { wi * v * v } else { v * v };
            }
        }
        for a in ss.iter_mut() {
            let sj = (*a / sw).sqrt();
            *a = if sj > 0.0 && sj.is_finite() {
                self.identity = false;
                sj
            } else {
                1.0
            };
        }
    }

    pub(crate) fn is_identity(&self) -> bool {
        self.identity
    }
    // `intercept` and `center` are read by the tests only: the fit reads the
    // statistics through `standardize_column`, `shift`, `scale` and the β maps
    // below.
    #[cfg(test)]
    pub(crate) fn intercept(&self) -> Option<usize> {
        self.intercept
    }
    #[cfg(test)]
    pub(crate) fn center(&self, j: usize) -> f64 {
        self.center[j]
    }
    /// `sⱼ` of slope index `j` (a column of `X` below `p`, a random-effect-only
    /// column from `p`); 1 on a column left as it is (and on
    /// [`Self::IDENTITY`]).
    pub(crate) fn scale(&self, j: usize) -> f64 {
        if self.identity {
            1.0
        } else if let Some(s) = self.re_only(j) {
            s
        } else {
            self.scale[j]
        }
    }
    /// `mⱼ/sⱼ` of slope index `j`, as [`Self::scale`]; 0 on a column left as
    /// it is, on a random-effect-only column (and on [`Self::IDENTITY`]).
    pub(crate) fn shift(&self, j: usize) -> f64 {
        if self.identity || j >= self.scale.len() {
            0.0
        } else {
            self.shift[j]
        }
    }

    /// The scale of random-effect-only column `j − p`, `None` below `p`.
    fn re_only(&self, j: usize) -> Option<f64> {
        let p = self.scale.len();
        (j >= p).then(|| self.re_only_scale[j - p])
    }

    /// Column `j` of `x̃` in place, `v ← (v − mⱼ)/sⱼ`, `j` a slope index as in
    /// [`Self::scale`]; a column left as it is is not touched.
    pub(crate) fn standardize_column(&self, j: usize, col: &mut [f64]) {
        if self.identity {
            return;
        }
        let (m, s) = match self.re_only(j) {
            Some(s) => (0.0, s),
            None => (self.center[j], self.scale[j]),
        };
        if m == 0.0 && s == 1.0 {
            return;
        }
        for v in col {
            *v = (*v - m) / s;
        }
    }

    /// Entry `i` of row `t` of `B`. `Var(βₜ) = scale·‖L⁻¹bₜ‖²` with `bₜ` this row
    /// and `LLᵀ` the internal information: `Var(β) = B·Ṽ·Bᵀ`.
    #[inline]
    pub(crate) fn rhs(&self, t: usize, i: usize) -> f64 {
        if self.identity {
            return if i == t { 1.0 } else { 0.0 };
        }
        if Some(t) == self.intercept {
            if i == t {
                1.0
            } else {
                -self.shift[i]
            }
        } else if i == t {
            1.0 / self.scale[t]
        } else {
            0.0
        }
    }

    /// `βₜ = (Bβ̃)ₜ` for one coefficient.
    pub(crate) fn beta_coef(&self, t: usize, beta_tilde: &[f64]) -> f64 {
        if self.identity {
            return beta_tilde[t];
        }
        if Some(t) == self.intercept {
            let mut acc = beta_tilde[t];
            for (i, &b) in beta_tilde.iter().enumerate() {
                if i != t {
                    acc -= self.shift[i] * b;
                }
            }
            acc
        } else {
            beta_tilde[t] / self.scale[t]
        }
    }

    /// `β̃ → β = Bβ̃` in place: `β₀ = β̃₀ − Σⱼ(mⱼ/sⱼ)β̃ⱼ`, `βⱼ = β̃ⱼ/sⱼ`.
    pub(crate) fn beta_to_caller(&self, beta: &mut [f64]) {
        if self.identity {
            return;
        }
        if let Some(c) = self.intercept {
            beta[c] = self.beta_coef(c, beta);
        }
        for (j, b) in beta.iter_mut().enumerate() {
            if Some(j) != self.intercept {
                *b /= self.scale[j];
            }
        }
    }

    /// `β → β̃ = B⁻¹β` in place: `β̃ⱼ = sⱼβⱼ`, `β̃₀ = β₀ + Σⱼ mⱼβⱼ`.
    pub(crate) fn beta_to_internal(&self, beta: &mut [f64]) {
        if self.identity {
            return;
        }
        if let Some(c) = self.intercept {
            let mut acc = beta[c];
            for (i, &b) in beta.iter().enumerate() {
                if i != c {
                    acc += self.center[i] * b;
                }
            }
            beta[c] = acc;
        }
        for (j, b) in beta.iter_mut().enumerate() {
            if Some(j) != self.intercept {
                *b *= self.scale[j];
            }
        }
    }

    /// `r → r·B⁻¹` in place, `r` a row against the columns of `x̃`: since
    /// `x̃ = X·B`, the same row against the caller's `X`. `rₜ ← sₜrₜ + mₜr₀`
    /// off the intercept, which is unchanged.
    pub(crate) fn row_to_caller(&self, row: &mut [f64]) {
        if self.identity {
            return;
        }
        let r0 = self.intercept.map_or(0.0, |c| row[c]);
        for (j, v) in row.iter_mut().enumerate() {
            if Some(j) != self.intercept {
                *v = self.scale[j] * *v + self.center[j] * r0;
            }
        }
    }

    /// `M ← B·M·Bᵀ` in place, `O(p²)`. With `B = D·C` (`C` the unit-diagonal
    /// shift on the intercept row, `D = diag(1/sⱼ)`): first `C·M·Cᵀ`, then `D`
    /// on both sides.
    pub(crate) fn congruence(&self, mut m: MatMut<'_, f64>) {
        if self.identity {
            return;
        }
        let p = m.nrows();
        if let Some(c) = self.intercept {
            for a in 0..p {
                let mut acc = m[(a, c)];
                for i in 0..p {
                    if i != c {
                        acc -= self.shift[i] * m[(a, i)];
                    }
                }
                m[(a, c)] = acc;
            }
            for b in 0..p {
                let mut acc = m[(c, b)];
                for i in 0..p {
                    if i != c {
                        acc -= self.shift[i] * m[(i, b)];
                    }
                }
                m[(c, b)] = acc;
            }
        }
        for j in 0..p {
            if Some(j) == self.intercept {
                continue;
            }
            let s = self.scale[j];
            for a in 0..p {
                m[(a, j)] /= s;
            }
            for b in 0..p {
                m[(j, b)] /= s;
            }
        }
    }

    /// `2·Σⱼ ln sⱼ`: `x̃ = X·B` gives `ln|X̃ᵀV⁻¹X̃| = ln|XᵀV⁻¹X| − 2·Σⱼ ln sⱼ`
    /// (a shift has determinant 1), so the caller's REML deviance is the
    /// internal one plus this.
    pub(crate) fn reml_deviance_shift(&self) -> f64 {
        if self.identity {
            return 0.0;
        }
        2.0 * self.scale.iter().map(|s| s.ln()).sum::<f64>()
    }
}

#[cfg(test)]
#[path = "standardize_tests.rs"]
mod tests;
