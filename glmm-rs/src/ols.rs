//! OLS sufficient-statistics fit + t² inference.
//!
//! Hot-loop invariant: never call sqrt on the SE side, never call abs on β,
//! never call Student-t CDF. All thresholds come from the caller's
//! squared critical values. Per-coefficient inference outputs are
//! `t_sq_j = β̂_j² / var_diag_j`, where `var_diag_j = σ̂² · (X'X)⁻¹_jj`,
//! computed via `L u = e_j` → `‖u‖²` on the Cholesky factor of `X'X`.
//!
//! Scratch lives in the caller's workspace and is passed through `OlsScratch<'_>`.
//! `fit_suff_stats_t_sq` writes through the scratch and returns
//! `OlsFitView<'_>` borrowing the same storage. The warm path still
//! allocates per fit: `linalg::SeqLlt` allocates its Cholesky factor on every
//! call, before the factor is copied into `fit_factor`.

use faer::reborrow::{IntoConst, Reborrow, ReborrowMut};
use faer::{MatMut, MatRef};

use crate::standardize::{ColumnScaling, Units};

/// Rows per widened f64 panel in the suff-stats GEMM accumulate
/// (`OlsSuffStats::add_rows`). Bounds the f64
/// working set to PANEL_ROWS·P so the GEMM stays cache-resident instead of
/// streaming a full n×p copy.
/// Tuned 2026-06-12 over {128, 256, 512}, clock-locked, off-mode fits/s:
/// ols_large_n 3686 / 3802 / 3831, ols_wide 20792 / 20746 / 20657 — flat
/// within ~1–3% and opposite preferences, so 256 keeps the balanced middle.
pub const PANEL_ROWS: usize = 256;

/// Caller-owned scratch passed into `fit_suff_stats_t_sq`. Built at the call site
/// by reborrowing fields of the caller's workspace directly — this is the form that
/// composes with NLL split-borrowing (a `&mut self` helper method would
/// borrow the whole workspace and conflict with the shared `x_full`/`y_full`
/// reads in the same loop iteration).
pub struct OlsScratch<'w> {
    /// Output buffer for `β̂ = (X'X)⁻¹X'y`, length ≥ `p`, x-matrix column order.
    pub fit_betas: &'w mut [f64],
    /// Output buffer for `σ̂² · (X'X)⁻¹_jj` per requested target, length ≥ `t`.
    pub fit_var_diag: &'w mut [f64],
    /// Output buffer for `t_sq_j = β̂_j² / var_diag_j` per requested target, length ≥ `t`.
    pub fit_t_sq: &'w mut [f64],
    /// Forward-solve scratch for `L u = e_j`, length ≥ `p`; reused per target.
    pub fit_u_scratch: &'w mut [f64],
    /// `P × P` buffer that receives the lower-triangular Cholesky factor `L` of `X'X`.
    pub fit_factor: MatMut<'w, f64>,
    /// `P × 1` buffer used as the rhs for the `L L' β = X'y` solve, then overwritten with `β̂`.
    pub fit_rhs: MatMut<'w, f64>,
}

/// Borrowed view into the scratch produced by `fit_suff_stats_t_sq`.
/// Lifetime ties back to the workspace that owns
/// the storage.
pub struct OlsFitView<'a> {
    /// `β̂ = (X'X)⁻¹X'y`, length `p`, x-matrix column order, on the design the
    /// statistics were accumulated from (β̃ when the fit ran through a column
    /// map, see `fit_suff_stats_t_sq_mapped`).
    pub betas: &'a [f64],
    /// Length `t` — only the first `n_targets` entries are populated. In the
    /// caller's units under a column map, like `t_sq`.
    pub var_diag: &'a [f64],
    /// Length `t`.
    pub t_sq: &'a [f64],
    /// `P × P` factor: the lower-triangular Cholesky factor `L` of `X'X`
    /// (`L · L' = X'X`), which is what `fit_suff_stats_t_sq` writes and
    /// posthoc consumes. Contents are valid only when `converged == true`. On the
    /// `n <= p || p == 0` early-return path the factor is not written, so it
    /// holds either zeros (first call after workspace construction) or stale
    /// data from a previous fit. Posthoc gates on `converged` so this is safe
    /// in practice; new consumers must do the same.
    pub factor: MatRef<'a, f64>,
    /// Residual variance `σ̂² = RSS / df_resid`. `NaN` on every non-converged / rank-deficient return.
    pub sigma_sq: f64,
    /// Residual degrees of freedom, `n − p`.
    pub df_resid: u32,
    /// `false` on the `n <= p || p == 0` early-return path; all other fields are then unreliable.
    pub converged: bool,
    /// `‖y − X β̂‖²`, summed from the rows when the kernel was given them
    /// (the stable route), else the closed form `y′y − β̂′X′y` from the
    /// running sums (the streaming `fit_suff_stats_t_sq`). `NaN` on every
    /// non-converged / rank-deficient return.
    pub rss: f64,
    /// `Σ (yᵢ − ȳ)²`, `ȳ = sum_y / n`, summed from the rows when the kernel
    /// was given them (the stable route), else the one-pass closed form
    /// `y′y − (Σy)²/n` from the running sums (the streaming
    /// `fit_suff_stats_t_sq`), which cancels when `|ȳ|` is large next to the
    /// spread of `y`. `NaN` on every non-converged / rank-deficient return.
    pub sst: f64,
    /// Scale-invariant per-column pivot ratio of the (possibly weighted) Gram
    /// this fit came from (`min_pivot_ratio`), with `pivot_col` the column
    /// attaining it. **Detection only** — nothing here reads it to accept or
    /// reject a design, and nothing may start to; the reasoning is recorded at
    /// the computation site. Below `PIVOT_MIN` the coefficients are barely
    /// identified and the diagnostics channel says so. NaN on every
    /// non-converged return, where no factor was formed.
    pub pivot: f64,
    /// Column attaining `pivot`. Meaningless when `pivot` is NaN.
    pub pivot_col: u32,
}

/// Caller-owned scratch for the sufficient-statistics OLS path, parallel in
/// shape to `OlsScratch`. Built at the call site by reborrowing fields of
/// the caller's workspace so NLL split-borrowing keeps it composable with the design-
/// matrix reads inside the same loop iteration.
pub struct OlsSuffStats<'w> {
    /// `P × P` accumulator for `X'X` (only the lower triangle is meaningful).
    pub xtx: MatMut<'w, f64>,
    /// Length `P` accumulator for `X'y`.
    pub xty: &'w mut [f64],
    /// Scalar `y'y`.
    pub yty: &'w mut f64,
    /// Σ yᵢ — drives `SST = Σyᵢ² − (Σyᵢ)²/n` at the batch site.
    pub sum_y: &'w mut f64,
    /// Total rows added so far.
    pub n_rows: &'w mut usize,
    /// Panel-widening scratch, len ≥ `min(block rows, PANEL_ROWS) · p` —
    /// workspace `panel_x` at the hot sites.
    pub panel_x: &'w mut [f64],
    /// y twin, len ≥ `min(block rows, PANEL_ROWS)` — workspace `panel_y`.
    pub panel_y: &'w mut [f64],
}

impl<'w> OlsSuffStats<'w> {
    /// Accumulate a contiguous block of rows. Writes only into the lower
    /// triangle of `xtx` (i ≥ j in `xtx[(i, j)]`); the Cholesky path reads
    /// `Side::Lower` only, so storing the upper triangle would be wasted work.
    ///
    /// Panel-GEMM: each ≤PANEL_ROWS slice of the block is repacked densely
    /// (column-major) into `panel_x`/`panel_y` once, then X'X/X'y accumulate through
    /// faer GEMM (`Accum::Add`, `Par::Seq` — per-fit parallelism is the outer
    /// rayon loop) on x86_64, column dot products elsewhere (`Self::add_panel`).
    /// GEMM's blocked accumulation keeps the FP-add chain off the
    /// critical path, unlike a per-row rank-1 triangle update, which serializes
    /// one add per row (mirrors glm.rs's X'WX conversion). `yty`/`sum_y` stay
    /// scalar in row order.
    ///
    /// Caller's responsibility: `x_block.nrows() == y_block.len()` and the
    /// column count matches the workspace's predictor count.
    pub fn add_rows(&mut self, x_block: MatRef<'_, f64>, y_block: &[f64]) {
        debug_assert_eq!(x_block.nrows(), y_block.len());
        let p = self.xty.len();
        debug_assert_eq!(x_block.ncols(), p);
        let m = x_block.nrows();
        debug_assert!(self.panel_x.len() >= PANEL_ROWS.min(m) * p);
        debug_assert!(self.panel_y.len() >= PANEL_ROWS.min(m));

        let mut off = 0;
        while off < m {
            let rows = (m - off).min(PANEL_ROWS);
            // Re-pack the panel densely, column-major, leading dim = rows.
            for j in 0..p {
                let col = &mut self.panel_x[j * rows..(j + 1) * rows];
                for (i, v) in col.iter_mut().enumerate() {
                    *v = x_block[(off + i, j)];
                }
            }
            for i in 0..rows {
                let y_row = y_block[off + i];
                self.panel_y[i] = y_row;
                *self.yty += y_row * y_row;
                *self.sum_y += y_row;
            }
            self.add_panel(rows);
            off += rows;
        }
        *self.n_rows += m;
    }

    /// `X'X += Xₚ'Xₚ` (lower triangle) and `X'y += Xₚ'yₚ` for the packed panel
    /// of `rows` rows, through faer GEMM.
    ///
    /// x86_64 only: there faer runs its own GEMM kernels, which do not
    /// allocate. On every other target (aarch64, wasm32) faer hands a product
    /// past 16³ to the `gemm` crate, whose `gemm_basic_generic` (gemm-common
    /// 0.19) heap-allocates a packing buffer and a flag vector on every call,
    /// so the warm path would allocate twice per panel.
    #[cfg(target_arch = "x86_64")]
    fn add_panel(&mut self, rows: usize) {
        use faer::linalg::matmul::matmul;
        use faer::linalg::matmul::triangular::{self, BlockStructure};
        use faer::{Accum, Par};
        let p = self.xty.len();
        let xp = MatRef::from_column_major_slice(&self.panel_x[..rows * p], rows, p);
        triangular::matmul(
            self.xtx.rb_mut(),
            BlockStructure::TriangularLower,
            Accum::Add,
            xp.transpose(),
            BlockStructure::Rectangular,
            xp,
            BlockStructure::Rectangular,
            1.0,
            Par::Seq,
        );
        matmul(
            MatMut::from_column_major_slice_mut(&mut *self.xty, p, 1),
            Accum::Add,
            xp.transpose(),
            MatRef::from_column_major_slice(&self.panel_y[..rows], rows, 1),
            1.0,
            Par::Seq,
        );
    }

    /// The x86_64 [`Self::add_panel`] off x86_64: one column dot product per
    /// lower-triangle entry and per `X'y` entry, allocation-free on any target.
    #[cfg(not(target_arch = "x86_64"))]
    fn add_panel(&mut self, rows: usize) {
        let p = self.xty.len();
        let col = |j: usize| &self.panel_x[j * rows..(j + 1) * rows];
        let yp = &self.panel_y[..rows];
        for j in 0..p {
            for i in j..p {
                self.xtx[(i, j)] += panel_dot(col(i), col(j));
            }
            self.xty[j] += panel_dot(col(j), yp);
        }
    }
}

/// `Σ aᵢbᵢ` on eight accumulators, reduced pairwise. Eight, not four: on
/// the entangled design of `ols_reml_reports_the_closed_form_reml_loglik`
/// (columns `a` and `a + 2⁻²⁰·b`, whose Gram entry needs about 59 bits),
/// four accumulators move the REML log-likelihood 1.3e-10 relative off the
/// closed form; eight reach the 2.4e-11 a compensated (Dot2) sum gives, the
/// same as faer's GEMM there (measured 2026-10-16, aarch64-apple-darwin).
#[cfg(not(target_arch = "x86_64"))]
fn panel_dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = [0.0f64; 8];
    let (ac, bc) = (a.chunks_exact(8), b[..a.len()].chunks_exact(8));
    let tail: f64 = ac
        .remainder()
        .iter()
        .zip(bc.remainder())
        .map(|(x, y)| x * y)
        .sum();
    for (x, y) in ac.zip(bc) {
        for k in 0..8 {
            s[k] += x[k] * y[k];
        }
    }
    (((s[0] + s[1]) + (s[2] + s[3])) + ((s[4] + s[5]) + (s[6] + s[7]))) + tail
}

// ---------------------------------------------------------------------------
// Shared triangular-solve helper
// ---------------------------------------------------------------------------

/// Solve a triangular system `T·u = b` into `scratch` and return `‖u‖²`.
///
/// - `upper=false`: lower-triangular factor; row access `factor[(i, k)]`.
/// - `upper=true`:  upper-triangular factor; column access `factor[(k, i)]`.
///
/// Returns `f64::NAN` immediately on a zero diagonal element. Only exact zero:
/// the factor carries the units of the matrix it factors (`X'WX` those of the
/// IRLS weight, 1/μ on inverse-Gaussian/log), so an absolute floor on its
/// diagonal turns small units into a refusal, and a non-positive-definite
/// matrix never reaches here (its Cholesky already fails).
/// The `b` closure supplies the right-hand side: `b(i)` is called once per row.
///
/// This de-duplicates the forward-substitution + norm-squared accumulation
/// shared by `fit_suff_stats_t_sq` and `ols_contrast_t_sq` (both Cholesky
/// path, lower) and by `glm::glm_irls_fit`'s per-target `Var(β̂_j)` step,
/// which reuses it directly: `Var(β̂_j) = ((X'WX)⁻¹)_jj = ‖L⁻¹ e_j‖²`, the
/// same identity as `fit_suff_stats_t_sq`'s `(X'X)⁻¹_jj` case above.
#[inline]
pub(crate) fn triangular_solve_norm_sq(
    factor: MatRef<'_, f64>,
    b: impl Fn(usize) -> f64,
    scratch: &mut [f64],
    p: usize,
    upper: bool,
) -> f64 {
    for v in &mut scratch[..p] {
        *v = 0.0;
    }
    for i in 0..p {
        let mut acc = b(i);
        for k in 0..i {
            acc -= if upper {
                factor[(k, i)]
            } else {
                factor[(i, k)]
            } * scratch[k];
        }
        let diag = factor[(i, i)];
        if diag == 0.0 {
            return f64::NAN;
        }
        scratch[i] = acc / diag;
    }
    let mut norm_sq = 0.0;
    for &v in &scratch[..p] {
        norm_sq += v * v;
    }
    norm_sq
}

/// NaN-fill the three scratch output slices (`betas[..p]`, `var_diag[..t]`,
/// `t_sq[..t]`) on the rank-deficient / early-return paths so callers can
/// detect non-convergence from NaN outputs alone. Successful paths overwrite
/// every populated slot.
/// Shared by the OLS fit and the GLM IRLS preamble.
#[inline]
pub(crate) fn nan_fill_ols_scratch(
    betas: &mut [f64],
    var_diag: &mut [f64],
    t_sq: &mut [f64],
    p: usize,
    t: usize,
) {
    betas[..p].fill(f64::NAN);
    var_diag[..t].fill(f64::NAN);
    t_sq[..t].fill(f64::NAN);
}

/// Build the canonical non-converged `OlsFitView`: all-NaN numerics,
/// `converged: false`, slices/factor passed through as the NaN-filled scratch.
/// `df_resid` is taken as an argument because the `n <= p` early-return sites
/// compute it with a saturating sub while the post-rank-check sites use an
/// unchecked `n - p` (where `n > p` already holds) — each caller keeps owning
/// its own expression so no panic path is reintroduced.
#[inline]
fn nonconverged_view<'a>(
    betas: &'a [f64],
    var_diag: &'a [f64],
    t_sq: &'a [f64],
    factor: MatRef<'a, f64>,
    df_resid: u32,
) -> OlsFitView<'a> {
    OlsFitView {
        betas,
        var_diag,
        t_sq,
        factor,
        sigma_sq: f64::NAN,
        df_resid,
        converged: false,
        rss: f64::NAN,
        sst: f64::NAN,
        pivot: f64::NAN,
        pivot_col: 0,
    }
}

/// Relative pivot floor for rank-deficiency detection. A column is
/// aliased when its Cholesky Schur pivot drops below `ALIAS_EPS · G_dd` (its OWN
/// Gram diagonal) — i.e. its residual norm, after projecting onto the retained
/// earlier columns, falls below `sqrt(ALIAS_EPS)` of its own original norm.
/// Per-column ⇒ scale-invariant (a small-magnitude independent column is not
/// wrongly dropped just because another column is large). `1e-14 = (1e-7)²`
/// mirrors lme4/R's `dqrdc2`, which drops at `1e-7` relative to the column norm;
/// the pivot is a squared quantity, so the eps is squared. (Distinct from OLS's
/// `1e-12` guard, which is on the L-diagonal = √pivot scale — a different basis.)
pub(crate) const ALIAS_EPS: f64 = 1e-14;

/// Order-preserving rank reveal: which columns of `X` are linearly dependent on
/// EARLIER columns, from the lower triangle of the Gram `G = XᵀX` (`p×p`).
/// Left-to-right (natural order, NO pivoting-by-norm) so the dropped set matches
/// lme4/R's `dqrdc2`: the later column of a collinear group is aliased, the
/// earlier retained. A column's Schur pivot (`G_dd − Σ_{j<d, retained} L_dj²`)
/// below `eps · G_dd` — its OWN Gram diagonal, so the test is scale-invariant and
/// matches dqrdc2's per-column-norm tolerance — marks it aliased; its `L` column
/// stays zero, so it contributes nothing to later pivots. Returns a length-`p`
/// mask (`true` = aliased/drop). `p` tiny ⇒ the `O(p³)` factor is negligible.
pub(crate) fn aliased_columns(gram: MatRef<'_, f64>, p: usize, eps: f64) -> Vec<bool> {
    aliased_factor(gram, p, eps).0
}

/// [`aliased_columns`]' mask with the factor it is decided on: `l` is the
/// row-major `p × p` lower triangle, with row `d` of an aliased column `d`
/// holding its entries against the retained columns before it and its own
/// column left zero.
pub(crate) fn aliased_factor(gram: MatRef<'_, f64>, p: usize, eps: f64) -> (Vec<bool>, Vec<f64>) {
    let mut l = vec![0.0f64; p * p]; // row-major lower-tri factor of the retained block
    let mut aliased = vec![false; p];
    for d in 0..p {
        let g_dd = gram[(d, d)];
        let mut piv = g_dd;
        for j in 0..d {
            if !aliased[j] {
                piv -= l[d * p + j] * l[d * p + j];
            }
        }
        if piv <= eps * g_dd {
            aliased[d] = true; // L column d left zero → invisible to later pivots
            continue;
        }
        let ljj = piv.sqrt();
        l[d * p + d] = ljj;
        for i in (d + 1)..p {
            let mut s = gram[(i, d)]; // lower triangle: i ≥ d
            for j in 0..d {
                if !aliased[j] {
                    s -= l[i * p + j] * l[d * p + j];
                }
            }
            l[i * p + d] = s / ljj;
        }
    }
    (aliased, l)
}

/// Ill-conditioning DETECTION floor for the OLS and GLM routes, on the
/// **weighted** Gram `X'WX` (`x_eff = √w·x`, and the converged IRLS `X'WX` for
/// GLM). Below it the coefficients are barely identified and the diagnostics
/// channel says so; neither route refuses a design over it.
///
/// Calibrated 2026-07-31 against a 1-ULP perturbation sweep. On these two routes
/// the worst relative movement of β̂ under a ±1-ULP re-rounding of `y` tracks
/// `1e-15 / pivot` closely — at pivot 7.0e-14 the law predicts 1.4e-2 and 1.3e-2
/// was measured — so `1e-12` is where β̂ has about three significant digits left.
///
/// The weighted-vs-raw distinction is load-bearing: [`aliased_columns`] tests
/// the RAW `x`, and a design that is full-rank unweighted can be near-singular
/// once weighted, so this measurement must not inherit that verdict.
///
/// No SOLVER path reads it: the kernels record the raw pivot and the comparison
/// happens once per route, in `OlsFitView::diagnostics` / `GlmFitView::diagnostics`,
/// which is what fills `FitDiagnostics::ill_conditioned`.
pub(crate) const PIVOT_MIN: f64 = 1e-12;

/// Scale-invariant rank statistic of a Cholesky factor `L` (`p × p` leading
/// block, lower-triangular): the minimum over columns of the Schur pivot divided
/// by that column's OWN Gram diagonal, plus the column that attains it.
///
/// This is the statistic [`aliased_columns`] already tests per column, read
/// straight off the factor instead of re-derived from the Gram. For an
/// unpivoted Cholesky in natural column order the two quantities are literally
/// entries of `L`:
///
/// > pivot_d = L_dd²  and  G_dd = Σ_{k ≤ d} L_dk²
///
/// (the second is row `d` of `L·Lᵀ` on its diagonal), so the ratio needs no
/// scratch matrix and no re-factorization — `O(p²)` and allocation-free, which
/// is why the guards can run it on every fit where `aliased_columns`'s
/// Gram form would allocate per fit.
///
/// Scale-invariant because each column is compared against its own norm:
/// rescaling one column of `X` scales pivot and diagonal together and leaves the
/// ratio fixed — unlike a min/max diagonal ratio, which conflates
/// near-collinearity with the design's column scaling.
///
/// A non-finite pivot or a non-positive reconstructed diagonal is arithmetic
/// exhaustion rather than a ratio, and returns `(0.0, d)` — the hardest possible
/// failure, attributed to the column that produced it. `p == 0` returns
/// `(f64::INFINITY, 0)`; the callers all reject an empty design earlier.
pub(crate) fn min_pivot_ratio(factor: MatRef<'_, f64>, p: usize) -> (f64, usize) {
    let mut min_ratio = f64::INFINITY;
    let mut min_col = 0usize;
    for d in 0..p {
        let mut g_dd = 0.0;
        for k in 0..=d {
            let l_dk = factor[(d, k)];
            g_dd += l_dk * l_dk;
        }
        let l_dd = factor[(d, d)];
        let pivot = l_dd * l_dd;
        if !g_dd.is_finite() || g_dd <= 0.0 || !pivot.is_finite() {
            return (0.0, d);
        }
        let ratio = pivot / g_dd;
        if ratio < min_ratio {
            min_ratio = ratio;
            min_col = d;
        }
    }
    (min_ratio, min_col)
}

/// The production OLS fit: Cholesky on accumulated sufficient statistics.
/// Starts from `(X'X, X'y, y'y, n_rows)` instead of a raw `(X, y)` pair —
/// this is the cross-N reuse path: each successive call sees a strictly larger
/// `n_rows` without re-doing the O(N·p²) reduction from scratch.
///
/// Inputs:
/// - `xtx_lower`: lower triangle of `X'X` (read-only). Upper triangle is
///   ignored — `Cholesky(Side::Lower)` only reads `i ≥ j` entries.
/// - `xty`: length-`P` vector of `X'y`.
/// - `yty`: scalar `y'y`.
/// - `n_rows`: number of rows accumulated into `xtx`/`xty`/`yty`.
/// - `target_indices`: per-coefficient indices into β̂ to test.
/// - `xtx_work`: `P × P` scratch buffer — `xtx_lower` is copied here because
///   faer's Cholesky reads the input matrix.
/// - `scratch`: caller-owned scratch.
#[expect(
    clippy::too_many_arguments,
    reason = "sufficient-statistics kernel; each arg is a distinct precomputed input"
)]
pub fn fit_suff_stats_t_sq<'a>(
    xtx_lower: MatRef<'_, f64>,
    xty: &[f64],
    yty: f64,
    sum_y: f64,
    n_rows: usize,
    target_indices: &[u32],
    xtx_work: MatMut<'_, f64>,
    scratch: OlsScratch<'a>,
) -> OlsFitView<'a> {
    fit_suff_stats_t_sq_mapped(
        xtx_lower,
        xty,
        yty,
        sum_y,
        n_rows,
        target_indices,
        xtx_work,
        scratch,
        ColumnScaling::IDENTITY,
        Units::IDENTITY,
        None,
        None,
    )
}

/// [`fit_suff_stats_t_sq`] on a design fitted through a column map `B`
/// (`x̃ = X·B`, `β = B·β̃`) and on a response in `units` (the caller's β is
/// `B·k·(β̃ + δ̃)`). The returned `betas` are β̃, on the design the
/// statistics were accumulated from and in the internal units, and so are
/// `sigma_sq` and `rss`; `var_diag` and `t_sq` are the caller's:
/// `Var(βₜ) = k²·σ̂²·‖L⁻¹bₜ‖²` with `bₜ` row `t` of `B`, `t² = βₜ²/Var(βₜ)`.
///
/// `qr`: the statistics were accumulated on `X·U⁻¹` (`lmm::DesignQr`), whose
/// Gram is well conditioned where `X`'s is not. The factor and β come back
/// mapped to `X` — `L = Uᵀ·L̃`, `β = U⁻¹·β̃` — so the view reads as if `X`
/// itself had been factored, with `X`'s digits kept.
///
/// `rows`: `(x, y)`, the `n × p` design and length-`n` response the
/// statistics were accumulated from, before any `U⁻¹` transform. With
/// `Some`, the RSS is summed from their residuals at β after the `qr`
/// mapping; with `None` it is the closed form from the running sums.
#[expect(
    clippy::too_many_arguments,
    reason = "sufficient-statistics kernel; each arg is a distinct precomputed input"
)]
pub(crate) fn fit_suff_stats_t_sq_mapped<'a>(
    xtx_lower: MatRef<'_, f64>,
    xty: &[f64],
    yty: f64,
    sum_y: f64,
    n_rows: usize,
    target_indices: &[u32],
    mut xtx_work: MatMut<'_, f64>,
    scratch: OlsScratch<'a>,
    map: &ColumnScaling,
    units: &Units,
    qr: Option<&crate::lmm::DesignQr>,
    rows: Option<(MatRef<'_, f64>, &[f64])>,
) -> OlsFitView<'a> {
    let p = xty.len();
    let t = target_indices.len();
    let n = n_rows;

    debug_assert_eq!(xtx_lower.nrows(), p);
    debug_assert_eq!(xtx_lower.ncols(), p);
    debug_assert_eq!(xtx_work.nrows(), p);
    debug_assert_eq!(xtx_work.ncols(), p);

    let OlsScratch {
        fit_betas,
        fit_var_diag,
        fit_t_sq,
        fit_u_scratch,
        mut fit_factor,
        mut fit_rhs,
    } = scratch;

    debug_assert!(p <= fit_betas.len(), "scratch sized for fewer predictors");
    debug_assert!(t <= fit_var_diag.len());
    debug_assert!(p <= fit_rhs.nrows(), "fit_rhs must hold at least p rows");

    // NaN-fill on the rank-deficient / early-return paths so callers can
    // detect non-convergence from NaN outputs alone. Successful paths
    // overwrite every populated slot.
    nan_fill_ols_scratch(fit_betas, fit_var_diag, fit_t_sq, p, t);

    if n <= p || p == 0 {
        return nonconverged_view(
            &fit_betas[..p],
            &fit_var_diag[..t],
            &fit_t_sq[..t],
            fit_factor.into_const(),
            n.saturating_sub(p) as u32,
        );
    }

    // Copy the lower triangle of X'X into the working buffer (faer's Cholesky
    // only reads `Side::Lower`, but it inspects all entries it reads — write
    // them all). Upper triangle is irrelevant; we leave it untouched.
    for j in 0..p {
        for i in j..p {
            xtx_work[(i, j)] = xtx_lower[(i, j)];
        }
    }

    let chol = match crate::linalg::SeqLlt::new(xtx_work.rb()) {
        Some(c) => c,
        None => {
            return nonconverged_view(
                &fit_betas[..p],
                &fit_var_diag[..t],
                &fit_t_sq[..t],
                fit_factor.into_const(),
                (n - p) as u32,
            );
        }
    };

    // Materialize L (owned Mat; strict upper triangle zeroed) and MEASURE its
    // conditioning. `Llt::new(...)` has already rejected strictly non-PD inputs;
    // what remains is the near-singular grey zone it accepts, and this route
    // does not refuse any of it.
    //
    // A `min|L_ii| / max|L_ii|` guard is wrong twice over: it never fires on
    // the designs it would exist for (measured 4.8e-8 on a weighted-collinear
    // design, four orders above a 1e-12 threshold), and refusing is the wrong
    // response anyway. Measured 2026-07-31 on a 1-ULP
    // perturbation sweep down past total loss of β̂: the standard errors stay
    // stable to 1.3e-13 relative and never understate the actual error — at the
    // bottom of the sweep the route reports β̂ = 7.11 for a true 0.5 with an SE
    // of 6.1e5. The SE column already tells the caller the estimate is
    // worthless, so refusing would destroy the coefficients for no gain.
    //
    // The Gram behind `l` is the WEIGHTED one, and that is the point: the
    // pre-dispatch alias gate tests the raw `x` and cannot see a design that
    // only goes singular under `W`.
    let l = chol.L();

    // Copy L into the caller-owned `fit_factor` so the returned view borrows
    // workspace storage rather than the locally-owned `l` Mat.
    for j in 0..p {
        for i in 0..p {
            fit_factor[(i, j)] = if i >= j { l[(i, j)] } else { 0.0 };
        }
    }

    // β̂ via two triangular solves: L · z = X'y; L' · β̂ = z. Use the top
    // `p` rows of `fit_rhs` as the rhs buffer.
    for j in 0..p {
        fit_rhs[(j, 0)] = xty[j];
    }
    chol.solve_in_place(fit_rhs.rb_mut().subrows_mut(0, p));
    for j in 0..p {
        fit_betas[j] = fit_rhs[(j, 0)];
    }

    // β̃'X̃'y = β'X'y, so the closed form needs no mapping; β and the factor do.
    let bty: f64 = (0..p).map(|j| fit_betas[j] * xty[j]).sum();
    if let Some(qr) = qr {
        qr.map_beta(&mut fit_betas[..p]);
        qr.map_factor(fit_factor.rb_mut().submatrix_mut(0, 0, p, p));
    }
    // RSS = ‖y − Xβ̂‖² and SST = Σ(yᵢ − ȳ)², summed from the rows when given:
    // the closed forms y'y − β̂'X'y and y'y − (Σy)²/n cancel when y is far
    // from 0 next to its residuals or its spread.
    let (rss, sst) = match rows {
        Some((x, y)) => {
            let ybar = sum_y / n as f64;
            (0..n).fold((0.0, 0.0), |(rss, sst), i| {
                let r = y[i] - (0..p).map(|j| x[(i, j)] * fit_betas[j]).sum::<f64>();
                let d = y[i] - ybar;
                (rss + r * r, sst + d * d)
            })
        }
        None => (yty - bty, yty - (sum_y * sum_y) / n as f64),
    };
    let (pivot, pivot_col) = min_pivot_ratio(fit_factor.rb(), p);
    let df_resid = (n - p) as u32;
    let sigma_sq = rss / df_resid as f64;

    // For each target: forward-solve L · u = bₜ, row t of the column map B
    // (eₜ on an unmapped design; read L[i, k] directly — no transposed access —
    // since L is already lower-triangular). Then var_diag = σ̂² · ‖u‖² because
    // B(X̃'X̃)⁻¹Bᵀ = (L⁻¹Bᵀ)ᵀ(L⁻¹Bᵀ) and its diagonal entries are ‖L⁻¹bₜ‖².
    for (out_idx, &tj) in target_indices.iter().enumerate() {
        let tj = tj as usize;
        if tj >= p {
            continue;
        }
        let norm_sq = triangular_solve_norm_sq(
            fit_factor.rb(),
            |i| map.rhs(tj, i),
            fit_u_scratch,
            p,
            false, // lower-triangular L: row access factor[(i, k)]
        );
        let mut vd = units.var_scale() * (sigma_sq * norm_sq);
        // A variance that underflows or overflows is not representable: NaN,
        // not a false 0 or inf. Mirrors `fit::common::vcov_from_chol`.
        if sigma_sq > 0.0 && sigma_sq.is_finite() && !(vd > 0.0 && vd.is_finite()) {
            vd = f64::NAN;
        }
        fit_var_diag[out_idx] = vd;
        if vd > 0.0 && vd.is_finite() {
            let beta_j =
                units.beta_scale() * (map.beta_coef(tj, &fit_betas[..p]) + units.shift_caller(tj));
            fit_t_sq[out_idx] = (beta_j * beta_j) / vd;
        } else {
            fit_t_sq[out_idx] = f64::NAN;
        }
    }

    OlsFitView {
        betas: &fit_betas[..p],
        var_diag: &fit_var_diag[..t],
        t_sq: &fit_t_sq[..t],
        factor: fit_factor.into_const(),
        sigma_sq,
        df_resid,
        converged: true,
        rss,
        sst,
        pivot,
        pivot_col: pivot_col as u32,
    }
}

// ---------------------------------------------------------------------------
// Contrast t² — pairwise contrast β_p − β_n via Cholesky forward solve
// ---------------------------------------------------------------------------

/// Compute the Wald t² statistic for the pairwise contrast `β_p − β_n`.
///
/// Uses the lower-triangular Cholesky factor `L` of `X'X` (stored in the
/// `factor` field of `OlsFitView`) to compute:
///
/// ```text
/// c = e_p − e_n          (contrast vector, length p)
/// L · u = c               (forward solve)
/// var(β_p − β_n) = σ̂² · ‖u‖²
/// t² = (β̂_p − β̂_n)² / var
/// ```
///
/// Returns `NaN` when the fit did not converge, when an index is ≥ p or the
/// two are equal (a zero contrast has no test), when `scratch` is shorter
/// than p, or when the variance is not finite and positive.
///
/// Valid only on a view fitted without a column map (`fit_suff_stats_t_sq`):
/// it reads `betas` and `factor` as the caller's own coefficients and Gram.
///
/// `scratch` must be length ≥ `p`; it is overwritten and not read on output.
pub fn ols_contrast_t_sq(fit: &OlsFitView<'_>, p_col: u32, n_col: u32, scratch: &mut [f64]) -> f64 {
    if !fit.converged {
        return f64::NAN;
    }
    let p = fit.betas.len();
    let var = fit.sigma_sq
        * pair_contrast_norm_sq(
            fit.factor,
            ColumnScaling::IDENTITY,
            p,
            p_col,
            n_col,
            scratch,
        );
    // NaN here also covers the index and scratch checks, before `betas` is read.
    if var.is_nan() {
        return f64::NAN;
    }
    contrast_t_sq_of(fit.betas[p_col as usize] - fit.betas[n_col as usize], var)
}

/// ‖L⁻¹·Bᵀ(e_p − e_n)‖² over the leading p × p block of `factor`; NaN when an
/// index is ≥ p, the indices are equal, or `scratch` is shorter than p.
///
/// `factor` is the lower-triangular `L` of the internal information `LLᵀ` and
/// `B` the column map, so `Var(β_p − β_n) = scale · ‖L⁻¹·Bᵀ(e_p − e_n)‖²` with
/// `scale` the fit's residual variance (1 on a GLM's φ = 1 scale). The forward
/// solve's right-hand side is `map.rhs(p_col, i) − map.rhs(n_col, i)`, which
/// under [`ColumnScaling::IDENTITY`] is exactly 1, −1 and 0.
pub(crate) fn pair_contrast_norm_sq(
    factor: MatRef<'_, f64>,
    map: &ColumnScaling,
    p: usize,
    p_col: u32,
    n_col: u32,
    scratch: &mut [f64],
) -> f64 {
    let pc = p_col as usize;
    let nc = n_col as usize;
    if pc >= p || nc >= p || pc == nc || scratch.len() < p {
        return f64::NAN;
    }
    triangular_solve_norm_sq(
        factor,
        |i| map.rhs(pc, i) - map.rhs(nc, i),
        scratch,
        p,
        false, // lower-triangular L: row access factor[(i, k)]
    )
}

/// diff²/var, NaN unless var is finite and positive.
#[inline]
pub(crate) fn contrast_t_sq_of(diff: f64, var: f64) -> f64 {
    if var > 0.0 && var.is_finite() {
        (diff * diff) / var
    } else {
        f64::NAN
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "ols_tests.rs"]
mod tests;
