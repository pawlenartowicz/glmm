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
//!
//! **The response, the prior weights and the offset** ([`Units`]). On the
//! families with a dispersion (Gaussian, Gamma, inverse Gaussian) the prior
//! weights are precision weights: row `i` has variance `φ·V(μᵢ)/wᵢ`, so
//! `w → w/s` with `φ → φ/s` is an exact reparametrization. Every route fits
//! on `ŵ = w/s`, `s` the power of two nearest the geometric mean of `w`
//! (`family::weight_scale`), and the dispersion it reports is `s` times the
//! internal one; β, the SEs, the covariances, θ, τ² and the log-likelihood
//! do not take `s`. On binomial, Poisson and negative binomial the weights
//! are trials or exposure, part of the model, and stay as given.
//!
//! The response takes a power-of-two scale `c` on the same three families,
//! the Gaussian response is first centred on its weighted least-squares fit
//! `x̃β̃₀`, and on a log link with an intercept the offset is centred at its
//! weighted mean `m`. Every route fits on the internal values and maps back
//! once, where the column map is applied. With `β̃ = k·(β̃_int + δ̃)` on x̃'s
//! columns (the derivations are on [`Units`]):
//!
//! | | Gaussian | Gamma, IG, log link | Gamma, inverse link | IG, 1/μ² link |
//! |---|---|---|---|---|
//! | internal y | `(y − o − x̃β̃₀)/c` | `y/c` | `y/c` | `y/c` |
//! | internal offset | none (in y) | `o − m` | `c·o` | `c²·o` |
//! | `k`, `δ̃` | `c`, `β̃₀/c` | `1`, `ln c − m` on the intercept | `1/c`, 0 | `1/c²`, 0 |
//! | `Var(β̂)` | `·c²` | unchanged | `/c²` | `/c⁴` |
//! | τ², varcorr | `·c²` (θ unchanged) | unchanged | `/c²` (θ `/c`) | no mixed model |
//! | dispersion `φ = d·φ_int` | `d = s·c²` | Gamma `d = s`, IG `d = s/c` | `d = s` | `d = s/c` |
//! | fitted μ | from the caller's β | `·c` | `·c` | `·c` |
//! | log-likelihood | `− ν·ln c` | `− n·ln c` | `− n·ln c` | `− n·ln c` |
//!
//! ν is `n` on ML and `n − p` on REML; a deviance takes `−2` times the
//! log-likelihood's shift. Poisson and negative binomial take only `m`
//! (`δ̃ = −m` on the intercept); binomial takes nothing. A log link without
//! an intercept gets `c = 1` and `m = 0`.

use faer::MatMut;

use crate::Family;

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
            let mut s = (ss / sw).sqrt();
            let mut m = m;
            // Past |x| ≈ 1.3e154 the sums overflow (or below ≈ 1e-154 the
            // squares underflow): the two passes again on the column divided
            // by its largest |xᵢⱼ|, so `mⱼ` and `sⱼ` are those of a column
            // in [-1, 1], scaled back. A centered column has `mⱼ ≠ 0` unless
            // its mean is exactly 0, where centering is a no-op either way.
            if !(s > 0.0 && s.is_finite() && m.is_finite()) {
                (m, s) = rescaled_spread(x, p, j, weights, sw, m != 0.0);
            }
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

/// One draw's units of the response, the prior weights and the offset (the
/// module doc's last part), and their map back to the caller's units. Built
/// once per workspace with its buffers at `n_max`, so a draw allocates
/// nothing; [`Units::IDENTITY`] is the map of a fit run on the data as given.
///
/// The response scale `c` on Gamma and inverse Gaussian: the fit runs on
/// `y/c`, and the caller's β is `k·(β̃_int + δ̃)` on x̃'s columns, its
/// `Var(β̂)` `k²` times the internal one and its dispersion `d·φ_int`. Each
/// entry follows from the density:
/// - Both are scale families: a row's density in `y` is its density in `y/c`
///   over `c`, so the log-likelihood takes `−n·ln c` and a deviance `+2n·ln c`.
/// - Gamma's φ does not depend on y's units (`d = s`); the inverse-Gaussian
///   unit deviance `(y − μ)²/(μ²y)` scales as `1/c`, so `φ_int = φ·c/s`
///   (`d = s/c`). A held φ goes in as `v/d`.
/// - Log link: `y/c` moves η by `−ln c`, which the intercept absorbs
///   (`δ̃ = ln c` on it, `k = 1`). Without an intercept nothing absorbs it,
///   and `c = 1`.
/// - Log link with an intercept and an offset, on every family: the offset
///   is centred at its weighted mean `m` (`ŵ` on Gamma and inverse
///   Gaussian, the trials or exposure on Poisson and negative binomial). The
///   fit runs on `o − m`, which moves η by `−m`; the intercept takes it
///   (`δ̃ = ln c − m`). μ, the deviance and every variance do not move.
/// - Inverse link: η = 1/μ scales as `c`, and with it β, the random effects,
///   θ and the offset (`k = 1/c`); on 1/μ², as `c²` (`k = 1/c²`).
/// - The fitted means take `c`; θ and τ² on the log link, and the Wald `t²`
///   on every link, do not.
///
/// On Gaussian the fit runs on `(y − o − x̃β̃₀)/c`, `β̃₀` the weighted
/// least-squares fit of `y − o` on x̃ and `c` the power of two nearest the
/// weighted root mean square of that residual. `y = Xβ + Zb + e` scales as a
/// whole, so `k = c`, `δ̃ = β̃₀/c` (the shift is exact: the residual, which
/// the LMM criterion and the OLS fit read, does not depend on it), and with
/// `β` the random effects, σ² and every variance component (σ²'s `d = s·c²`;
/// θ, relative to σ, does not move). The log-likelihood takes `−ν·ln c` and
/// the LMM deviance `+2ν·ln c`, ν = n on ML and n − p on REML, whose
/// criterion is the density of the n − p error contrasts. The offset is
/// folded into the response, so no Gaussian route gets one.
#[derive(Clone, Debug)]
pub(crate) struct Units {
    /// Which map applies, from the family and link; fixed at build.
    kind: UnitsKind,
    /// Weight scale `s`: `family::weight_scale` of the prior weights on a
    /// dispersion family, 1 otherwise.
    s: f64,
    /// Response scale `c`: a power of two on Gaussian, Gamma and inverse
    /// Gaussian, 1 otherwise.
    c: f64,
    /// Offset centre `m` on a log link with an intercept and an offset, 0
    /// otherwise.
    m: f64,
    /// The first column of the caller's x whose entries are all exactly 1.0
    /// (the rule `ColumnScaling` uses), found on every family but binomial.
    intercept: Option<usize>,
    /// `δ̃` on x̃'s columns (`p` long on every family but binomial, empty
    /// there): `β̃₀/c` on Gaussian, `ln c − m` on the intercept of a log link,
    /// zeros otherwise.
    shift: Vec<f64>,
    /// `B·δ̃`, the same shift on the caller's columns.
    shift_caller: Vec<f64>,
    /// `ŵ = w/s`, the internal response and the offset on the internal
    /// scale, `n_max` long where the family can change them (empty
    /// otherwise); their first `n` entries are read only while the matching
    /// flag is set.
    w: Vec<f64>,
    y: Vec<f64>,
    offset: Vec<f64>,
    w_changed: bool,
    y_changed: bool,
    offset_changed: bool,
    n: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UnitsKind {
    /// Binomial: weights, response and offset as given.
    Counts,
    /// Poisson and negative binomial, on the log link: weights and response
    /// as given, the offset centred.
    CountsLog,
    Gaussian,
    GammaLog,
    GammaInverse,
    InverseGaussianLog,
    InverseGaussianInverseSquared,
}

impl Units {
    /// The map of a fit run on the data as given.
    pub(crate) const IDENTITY: &'static Units = &Units {
        kind: UnitsKind::Counts,
        s: 1.0,
        c: 1.0,
        m: 0.0,
        intercept: None,
        shift: Vec::new(),
        shift_caller: Vec::new(),
        w: Vec::new(),
        y: Vec::new(),
        offset: Vec::new(),
        w_changed: false,
        y_changed: false,
        offset_changed: false,
        n: 0,
    };

    /// The identity map of a workspace of `family` for up to `n_max` rows
    /// and `p` columns.
    pub(crate) fn new(family: Family, n_max: usize, p: usize) -> Units {
        use crate::{GammaLink, InverseGaussianLink};
        let kind = match family {
            Family::Gaussian => UnitsKind::Gaussian,
            Family::Gamma {
                link: GammaLink::Log,
            } => UnitsKind::GammaLog,
            Family::Gamma {
                link: GammaLink::Inverse,
            } => UnitsKind::GammaInverse,
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            } => UnitsKind::InverseGaussianLog,
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            } => UnitsKind::InverseGaussianInverseSquared,
            Family::Poisson { .. } | Family::NegativeBinomial { .. } => UnitsKind::CountsLog,
            Family::Binomial { .. } => UnitsKind::Counts,
        };
        let buf = |on: bool, len: usize| if on { vec![0.0; len] } else { Vec::new() };
        let dispersion = !matches!(kind, UnitsKind::Counts | UnitsKind::CountsLog);
        let shifted = kind != UnitsKind::Counts;
        let offset = !matches!(kind, UnitsKind::Counts | UnitsKind::Gaussian);
        Units {
            kind,
            s: 1.0,
            c: 1.0,
            m: 0.0,
            intercept: None,
            shift: buf(shifted, p),
            shift_caller: buf(shifted, p),
            w: buf(dispersion, n_max),
            y: buf(dispersion, n_max),
            offset: buf(offset, n_max),
            w_changed: false,
            y_changed: false,
            offset_changed: false,
            n: 0,
        }
    }

    /// This draw's weight scale from the caller's prior weights (`None`: unit
    /// weights) on rows `0..n`. `s` is a power of two, so `ŵ` is `w` exactly
    /// rescaled; at `s = 1` the caller's weights are used as given.
    pub(crate) fn compute(&mut self, weights: Option<&[f64]>, n: usize) {
        self.s = 1.0;
        self.w_changed = false;
        let counts = matches!(self.kind, UnitsKind::Counts | UnitsKind::CountsLog);
        let Some(w) = weights.filter(|_| !counts) else {
            return;
        };
        self.s = crate::family::weight_scale(Some(w), n);
        if self.s != 1.0 {
            for (dst, &wi) in self.w[..n].iter_mut().zip(w) {
                *dst = wi / self.s;
            }
            self.w_changed = true;
            self.n = n;
        }
    }

    /// This draw's response scale, after [`Self::compute`] on the same rows:
    /// `c = 2^round(Σŵᵢ·log₂ yᵢ / Σŵᵢ)` on Gamma and inverse Gaussian (`c = 1`
    /// on a log link without an intercept), the exponent clamped to
    /// `[−1022, 1023]` so `c` and `1/c` are normal numbers; `c = 1` where the
    /// mean is not finite. On a log link with an intercept and an offset, the
    /// offset centre `m = Σwᵢoᵢ / Σwᵢ` (the weights the routes fit on; `m = 0`
    /// where it is not finite). `x` is the caller's row-major `n×p` design and
    /// `weights` the caller's. On Gaussian it finds the intercept only;
    /// [`Self::compute_gaussian`] does the rest.
    pub(crate) fn compute_response(
        &mut self,
        x: &[f64],
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        n: usize,
        p: usize,
    ) {
        self.c = 1.0;
        self.m = 0.0;
        self.intercept = None;
        self.shift.fill(0.0);
        self.shift_caller.fill(0.0);
        self.y_changed = false;
        self.offset_changed = false;
        if self.kind == UnitsKind::Counts || n == 0 {
            return;
        }
        self.intercept = (0..p).find(|&j| x[j..n * p].iter().step_by(p).all(|&v| v == 1.0));
        let log = match self.kind {
            UnitsKind::GammaLog | UnitsKind::InverseGaussianLog | UnitsKind::CountsLog => true,
            UnitsKind::GammaInverse | UnitsKind::InverseGaussianInverseSquared => false,
            UnitsKind::Counts | UnitsKind::Gaussian => return,
        };
        if log && self.intercept.is_none() {
            return;
        }
        if self.kind != UnitsKind::CountsLog {
            let (mut swl, mut sw) = (0.0, 0.0);
            let w = self.weights(weights);
            for (i, &yi) in y[..n].iter().enumerate() {
                let wi = w.map_or(1.0, |w| w[i]);
                swl += wi * yi.log2();
                sw += wi;
            }
            let e = (swl / sw).round();
            if e.is_finite() && e != 0.0 {
                self.c = 2.0_f64.powi(e.clamp(-1022.0, 1023.0) as i32);
                for (dst, &v) in self.y[..n].iter_mut().zip(y) {
                    *dst = v / self.c;
                }
                self.y_changed = true;
                self.n = n;
                if let Some(o) = offset.filter(|_| !log) {
                    let f = if self.kind == UnitsKind::GammaInverse {
                        self.c
                    } else {
                        self.c * self.c
                    };
                    for (dst, &v) in self.offset[..n].iter_mut().zip(o) {
                        *dst = v * f;
                    }
                    self.offset_changed = true;
                }
            }
        }
        let Some(intercept) = self.intercept.filter(|_| log) else {
            return;
        };
        if let Some(o) = offset {
            let (mut swo, mut sw) = (0.0, 0.0);
            let w = self.weights(weights);
            for (i, &oi) in o[..n].iter().enumerate() {
                let wi = w.map_or(1.0, |w| w[i]);
                swo += wi * oi;
                sw += wi;
            }
            let m = swo / sw;
            if m.is_finite() && m != 0.0 {
                self.m = m;
                for (dst, &v) in self.offset[..n].iter_mut().zip(o) {
                    *dst = v - m;
                }
                self.offset_changed = true;
                self.n = n;
            }
        }
        // A column map leaves the intercept as it is (`B·eᵢ = eᵢ`).
        self.shift[intercept] = self.c.ln() - self.m;
        self.shift_caller[intercept] = self.shift[intercept];
    }

    /// The Gaussian response, after [`Self::compute_response`] on the same
    /// rows: the residual `r = y − o − x̃·β̃₀` (`β̃₀` the weighted
    /// least-squares fit of `y − o` on x̃ by `shift`'s factor, 0 without one)
    /// and `c = 2^round(log₂ ‖r‖)`, `‖r‖` its weighted root mean square, the
    /// exponent clamped as in [`Self::compute_response`] and `c = 1` where
    /// `‖r‖` is zero or not finite. The internal response is `r/c`. `x_mat` is
    /// x̃ (column-major, `n` rows) and `map` its column map; `weights` are
    /// the caller's.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compute_gaussian(
        &mut self,
        x_mat: faer::MatRef<'_, f64>,
        y: &[f64],
        weights: Option<&[f64]>,
        offset: Option<&[f64]>,
        shift: &mut crate::fit::ShiftBuf,
        n: usize,
        map: &ColumnScaling,
    ) {
        debug_assert_eq!(self.kind, UnitsKind::Gaussian);
        let p = self.shift.len();
        let w = if self.w_changed {
            Some(&self.w[..n])
        } else {
            weights
        };
        let r = &mut self.y[..n];
        let v = |i: usize| y[i] - offset.map_or(0.0, |o| o[i]);
        // `ŵ(y − o)` for the fit's right-hand side; the residual pass below
        // starts from `y − o`.
        match w {
            Some(w) => r.iter_mut().enumerate().for_each(|(i, d)| *d = w[i] * v(i)),
            None => r.iter_mut().enumerate().for_each(|(i, d)| *d = v(i)),
        }
        let beta0 = shift.solve(x_mat, r);
        // The intercept's term first: `y − o` and `β̃₀` on the intercept are
        // near each other on a response far from zero, so their difference is
        // exact (Sterbenz) and the smaller terms keep their digits. Column by
        // column on tiles of rows that stay in L1, so each row sees that
        // order and x̃ is read once.
        let intercept = self.intercept;
        let order = intercept
            .into_iter()
            .chain((0..p).filter(move |&j| Some(j) != intercept));
        const TILE: usize = 512;
        let mut ss = 0.0;
        for (t, rt) in r.chunks_mut(TILE).enumerate() {
            let i0 = t * TILE;
            if w.is_some() {
                rt.iter_mut().enumerate().for_each(|(i, d)| *d = v(i0 + i));
            }
            if let Some(b) = beta0 {
                for j in order.clone() {
                    let col = x_mat.col(j).try_as_col_major().expect("x̃ is column-major");
                    let bj = b[j];
                    rt.iter_mut()
                        .zip(&col.as_slice()[i0..])
                        .for_each(|(d, &x)| *d -= x * bj);
                }
            }
            ss = match w {
                Some(w) => rt
                    .iter()
                    .zip(&w[i0..])
                    .fold(ss, |a, (&v, &wi)| a + wi * v * v),
                None => rt.iter().fold(ss, |a, &v| a + v * v),
            };
        }
        self.y_changed = true;
        self.n = n;
        let sw = w.map_or(n as f64, |w| w.iter().sum());
        let mut rms = (ss / sw).sqrt();
        // Past |r| ≈ 1.3e154 the sum overflows, or below ≈ 1e-154 the squares
        // underflow: the spread of `r` divided by its largest |rᵢ|, as
        // [`ColumnScaling::compute`] falls back.
        if !(rms > 0.0 && rms.is_finite()) {
            rms = rescaled_spread(r, 1, 0, w, sw, false).1;
        }
        let e = rms.log2().round();
        if e.is_finite() && e != 0.0 {
            self.c = 2.0_f64.powi(e.clamp(-1022.0, 1023.0) as i32);
            r.iter_mut().for_each(|v| *v /= self.c);
        }
        if let Some(b) = beta0 {
            for (d, &v) in self.shift.iter_mut().zip(b) {
                *d = v / self.c;
            }
            self.shift_caller.copy_from_slice(&self.shift);
            map.beta_to_caller(&mut self.shift_caller);
        }
    }

    /// The weight scale `s`.
    pub(crate) fn s(&self) -> f64 {
        self.s
    }

    /// The response scale `c`: the routes fit `y/c`.
    pub(crate) fn c(&self) -> f64 {
        self.c
    }

    /// The offset centre `m`.
    pub(crate) fn m(&self) -> f64 {
        self.m
    }

    /// The intercept column [`Self::compute_response`] found (every family
    /// but binomial).
    pub(crate) fn intercept(&self) -> Option<usize> {
        self.intercept
    }

    /// The weights the routes fit on: `ŵ`, or `raw` (the caller's weights,
    /// the ones [`Self::compute`] read) where they did not change.
    pub(crate) fn weights<'a>(&'a self, raw: Option<&'a [f64]>) -> Option<&'a [f64]> {
        if self.w_changed {
            Some(&self.w[..self.n])
        } else {
            raw
        }
    }

    /// The response the routes fit on: `y/c`, `(y − o − x̃β̃₀)/c` on Gaussian,
    /// or `raw` where it did not change.
    pub(crate) fn y<'a>(&'a self, raw: &'a [f64]) -> &'a [f64] {
        if self.y_changed {
            &self.y[..self.n]
        } else {
            raw
        }
    }

    /// The offset the routes fit on: `o − m` on a log link, `c·o` on the
    /// inverse link, `c²·o` on 1/μ², or `raw` where it did not change.
    pub(crate) fn offset<'a>(&'a self, raw: Option<&'a [f64]>) -> Option<&'a [f64]> {
        if self.offset_changed {
            Some(&self.offset[..self.n])
        } else {
            raw
        }
    }

    /// `k`: the caller's β is `k·(β̃_int + δ̃)`.
    pub(crate) fn beta_scale(&self) -> f64 {
        match self.kind {
            UnitsKind::Gaussian => self.c,
            UnitsKind::GammaInverse => 1.0 / self.c,
            UnitsKind::InverseGaussianInverseSquared => (1.0 / self.c) * (1.0 / self.c),
            _ => 1.0,
        }
    }

    /// `k²`: the caller's `Var(β̂)` over the internal one.
    pub(crate) fn var_scale(&self) -> f64 {
        self.beta_scale() * self.beta_scale()
    }

    /// `d`: the caller's dispersion over the internal one.
    pub(crate) fn dispersion_scale(&self) -> f64 {
        match self.kind {
            UnitsKind::Gaussian => self.s * self.var_scale(),
            UnitsKind::InverseGaussianLog | UnitsKind::InverseGaussianInverseSquared => {
                self.s / self.c
            }
            _ => self.s,
        }
    }

    /// A held dispersion `v` (the caller's) on the internal scale.
    pub(crate) fn held_dispersion_in(&self, v: f64) -> f64 {
        v / self.dispersion_scale()
    }

    /// The factor from the GLM kernel's variance at φ = 1 on the internal
    /// data to the caller's `Var(β̂)/φ̂`: `Var(β̂) = k²·φ_int·(X̃ᵀŴX̃)⁻¹` with
    /// `φ_int = φ/d`.
    pub(crate) fn glm_var_scale(&self) -> f64 {
        self.var_scale() / self.dispersion_scale()
    }

    /// What the log-likelihood of the internal data takes to be the caller's:
    /// `−n·ln c`.
    pub(crate) fn loglik_shift(&self, n: usize) -> f64 {
        -(n as f64) * self.c.ln()
    }

    /// `δ̃` on caller column `t` (`B·δ̃`).
    pub(crate) fn shift_caller(&self, t: usize) -> f64 {
        self.shift_caller.get(t).copied().unwrap_or(0.0)
    }

    /// `β̃_int → k·(β̃_int + δ̃)`, on x̃'s columns.
    pub(crate) fn beta_from_internal(&self, beta: &mut [f64]) {
        for (b, &d) in beta.iter_mut().zip(&self.shift) {
            if d != 0.0 {
                *b += d;
            }
        }
        let k = self.beta_scale();
        if k != 1.0 {
            beta.iter_mut().for_each(|b| *b *= k);
        }
    }

    /// The caller's β from the internal β̃ (`B·k·(β̃_int + δ̃)`).
    pub(crate) fn beta_to_caller(&self, map: &ColumnScaling, beta: &mut [f64]) {
        self.beta_from_internal(beta);
        map.beta_to_caller(beta);
    }
}

/// Column `j`'s `(mⱼ, sⱼ)` from the column divided by its largest `|xᵢⱼ|`
/// (`mⱼ = 0` unless `centered`), the fallback of [`ColumnScaling::compute`]
/// where the direct sums are not finite or the spread underflows. `(NaN, NaN)`
/// for an all-zero or non-finite column.
fn rescaled_spread(
    x: &[f64],
    p: usize,
    j: usize,
    weights: Option<&[f64]>,
    sw: f64,
    centered: bool,
) -> (f64, f64) {
    let col = || x[j..].iter().step_by(p);
    let big = col().fold(0.0f64, |m, &v| m.max(v.abs()));
    if !(big > 0.0 && big.is_finite()) {
        return (f64::NAN, f64::NAN);
    }
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let mean = if centered {
        col()
            .enumerate()
            .map(|(i, &v)| w(i) * (v / big))
            .sum::<f64>()
            / sw
    } else {
        0.0
    };
    let ss: f64 = col()
        .enumerate()
        .map(|(i, &v)| {
            let d = v / big - mean;
            w(i) * d * d
        })
        .sum();
    (mean * big, (ss / sw).sqrt() * big)
}

#[cfg(test)]
#[path = "standardize_tests.rs"]
mod tests;
