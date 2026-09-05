//! Deviance kernel: the suff-stats accumulator (`LmmSuffStats`) and `reml_deviance`/`reml_deviance_blocked`/`precompute_balanced_collapse`/the REML dual gradient and Hessian.

use super::*;
use crate::dual::{Dual, HyperDual};
use crate::glmm::{unpack_hessian, DerivStatus};

// ---------------------------------------------------------------------------
// LmmSuffStats — augmented per-RE-column sufficient statistics.
// ---------------------------------------------------------------------------

/// w_i = [x_i ; y_i] (length m = p+1): `c` = Σ w wᵀ (lower triangle),
/// `s[:, a]` = Σ_{i in RE column a} w_i, `counts[a]` = n_a, over the full
/// RE-column set (`[primary | nested children | crossed]`, elimination order).
/// `zx` holds cross-counts only when crossed factors exist (crossed ⇒ Regime A
/// ⇒ K moderate); nested-only designs derive parent↔child coupling from
/// `counts` + the id/n_per parent map, so Regime-B-nested memory stays O(K·m).
///
/// Layout invariant: `s` stays per-RE-column-addressable with `counts`
/// alongside — the balanced-design collapse slots in at exactly this
/// granularity later; don't fold columns at accumulation time.
pub struct LmmSuffStats {
    /// Augmented width m = p + 1 (y in the last slot).
    pub m: usize,
    /// Rows accumulated since the last reset.
    pub n_rows: usize,
    /// Highest primary cluster id + 1 seen since the last reset.
    pub n_clusters: usize,
    /// Grouping-structure metadata (RE column layout, crossed/nested shape)
    /// shared with the deviance kernel.
    pub groupings: LmmGroupings,
    /// m×m Σ w wᵀ (lower triangle; upper never read).
    pub c: Mat<f64>,
    /// m × k_total per-RE-column Σ w.
    pub s: Mat<f64>,
    /// Per-RE-column Gram value Σ z_int²·wᵢ over rows in that RE column
    /// (z_int = 1, so unit weights reduce to the raw row count n_a). NOT a row
    /// count once weighted — every consumer reads it purely as a Gram diagonal
    /// entry; `df` for the fit comes from `n_rows`, never from `counts`.
    pub counts: Vec<f64>,
    /// Crossed cross-counts: zx[(a, b)] = #rows where RE column `a` and
    /// crossed column `k_family + b` co-occur. 0×0 when no crossed factors;
    /// nested↔primary coupling is derived from `counts` + the id/n_per parent
    /// map instead. Same-factor crossed pairs never co-occur (level-disjoint),
    /// so those entries stay 0 and the Ω assembly can read unconditionally.
    pub zx: Mat<f64>,
    /// Slope-weighted twin of `zx` (the slope-composition): for a primary slope RE
    /// column `scol = (d+1)·n_primary + f`, `zx_slope[(scol, b)] = Σ_{i ∈ f ∩
    /// crossed_b} x_{slope_d}` — the covariate-weighted co-occurrence the
    /// slope↔crossed coupling in `fam_b` reads (plain `zx` is unweighted, fit for
    /// the intercept row only). Same shape as `zx` (`k_total × k_crossed`); only
    /// the slope-RE-col rows are filled. 0×0 when no crossed factor; left all-zero
    /// when `primary_q == 1` (no slopes).
    pub zx_slope: Mat<f64>,
    /// Per-row widened [X y] (len m) — filled once per row so the c-triangle
    /// and s scatter read contiguous f64 instead of re-indexing the f32 data
    /// plane per (i, j). Scratch, not a statistic: reset leaves it alone.
    pub w_buf: Vec<f64>,
}

impl LmmSuffStats {
    /// Accumulator for a single-intercept grouping at `max_clusters` clusters
    /// (`k_total = max_clusters`, no crossed/nested columns).
    pub fn new(p: usize, max_clusters: usize) -> Self {
        Self::with_groupings(p, LmmGroupings::single(max_clusters))
    }

    /// Accumulator sized for an arbitrary `LmmGroupings` layout, including
    /// nested and crossed RE columns.
    pub fn with_groupings(p: usize, groupings: LmmGroupings) -> Self {
        let m = p + 1;
        let k = groupings.k_total;
        let kx = groupings.k_crossed();
        LmmSuffStats {
            m,
            n_rows: 0,
            n_clusters: 0,
            c: Mat::zeros(m, m),
            s: Mat::zeros(m, k),
            counts: vec![0.0; k],
            zx: Mat::zeros(if kx > 0 { k } else { 0 }, kx),
            zx_slope: Mat::zeros(if kx > 0 { k } else { 0 }, kx),
            w_buf: vec![0.0; m],
            groupings,
        }
    }

    /// Reset to "no rows seen", reusing storage.
    pub fn reset(&mut self) {
        let m = self.m;
        for j in 0..m {
            for i in 0..m {
                self.c[(i, j)] = 0.0;
            }
        }
        for a in 0..self.counts.len() {
            for j in 0..m {
                self.s[(j, a)] = 0.0;
            }
            self.counts[a] = 0.0;
        }
        let (zr, zc) = (self.zx.nrows(), self.zx.ncols());
        for j in 0..zc {
            for i in 0..zr {
                self.zx[(i, j)] = 0.0;
                self.zx_slope[(i, j)] = 0.0;
            }
        }
        self.n_rows = 0;
        self.n_clusters = 0;
    }

    /// Primary-only convenience — the primary-only shape.
    pub fn add_rows(&mut self, x: MatRef<'_, f64>, y: &[f64], cluster_ids: &[u32]) {
        self.add_rows_multi(x, y, cluster_ids, &[], None);
    }

    /// Accumulate a block of rows for every grouping. `extra_ids[g]` holds
    /// extra grouping g's GLOBALIZED level ids (workspace layout — crossed
    /// 0..I, nested parent·n_per+within), declaration order; this routine maps
    /// them onto the elimination-order column offsets. `weights[i]` (prior/case
    /// weight, unstable `loop_advanced` surface) is `wᵢ`; `None` is unit weight.
    /// Per-row rule for folding prior weights into the unit-weight suff-stats
    /// accumulator: every row is conceptually √wᵢ-scaled before hitting the math,
    /// so `wi.sqrt()` (`zw`) multiplies `w_buf` once (propagating one `zw` into
    /// every `[X y]` and slope-`z` read) and every bare intercept-`z=1.0` site
    /// takes one more explicit `zw` (or `wi` where both intercept sides already
    /// collapsed to a single literal — `zw·zw = wi`).
    pub fn add_rows_multi(
        &mut self,
        x: MatRef<'_, f64>,
        y: &[f64],
        cluster_ids: &[u32],
        extra_ids: &[Vec<u32>],
        weights: Option<&[f64]>,
    ) {
        debug_assert_eq!(x.nrows(), y.len());
        debug_assert_eq!(x.nrows(), cluster_ids.len());
        debug_assert_eq!(extra_ids.len(), self.groupings.extra_offsets.len());
        debug_assert!(weights.is_none_or(|w| w.len() == x.nrows()));
        let p = self.m - 1;
        debug_assert_eq!(x.ncols(), p);
        let kf = self.groupings.k_family();
        let n_g = 1 + extra_ids.len();
        let mut gid = [0usize; 1 + MAX_EXTRA_GROUPINGS];
        for row in 0..x.nrows() {
            let wi = weights.map_or(1.0, |w| w[row]);
            let zw = wi.sqrt();
            gid[0] = cluster_ids[row] as usize;
            for (e, ids) in extra_ids.iter().enumerate() {
                // Intercept RE column of this level's q_g-wide block. q_g==1 ⇒ the
                // pre-slope `offset + id` (byte-identical); slope cols follow at
                // `gid + 1 .. gid + q_g`.
                gid[1 + e] =
                    self.groupings.extra_offsets[e] + ids[row] as usize * self.groupings.extra_q[e];
            }
            debug_assert!(gid[..n_g].iter().all(|&a| a < self.counts.len()));
            for &a in &gid[..n_g] {
                // Σ z_int²·wᵢ = Σ wᵢ over rows in this RE column (z_int = 1).
                self.counts[a] += wi;
            }
            // Load this row's [X y] into w_buf, then fold in one `zw` per side:
            // downstream reads of `w_buf` (the c Gram, slope-z reads) each carry
            // exactly one `zw`, so a product of two reads carries `zw² = wᵢ`.
            for j in 0..p {
                self.w_buf[j] = x[(row, j)];
            }
            self.w_buf[p] = y[row];
            for wj in &mut self.w_buf[..self.m] {
                *wj *= zw;
            }
            for &a in &gid[..n_g] {
                let scol = self
                    .s
                    .col_mut(a)
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut();
                // Intercept z = 1 becomes `zw`; `w_buf` already carries one `zw`,
                // so the product carries `zw² = wᵢ` — total wᵢ·[X y] per row.
                #[allow(clippy::needless_range_loop)]
                for j in 0..self.m {
                    scol[j] += zw * self.w_buf[j];
                }
            }
            for j in 0..self.m {
                let wj = self.w_buf[j];
                let ccol = self
                    .c
                    .col_mut(j)
                    .try_as_col_major_mut()
                    .unwrap()
                    .as_slice_mut();
                #[allow(clippy::needless_range_loop)]
                for i in j..self.m {
                    ccol[i] += self.w_buf[i] * wj;
                }
            }
            if self.groupings.k_crossed() > 0 && !self.groupings.extra_slopes_any {
                let slope = self.groupings.primary_q > 1;
                let n_prim = self.groupings.n_primary;
                for bi in 0..n_g {
                    let b = gid[bi];
                    if b >= kf {
                        let bl = b - kf;
                        #[allow(clippy::needless_range_loop)]
                        for ai in 0..n_g {
                            if ai != bi {
                                // Both sides intercept (z=1), collapsed to one
                                // literal — the weighted product is zw·zw = wᵢ.
                                self.zx[(gid[ai], bl)] += wi;
                            }
                        }
                        // Slope-weighted twin for the slope↔crossed coupling.
                        // The intercept row is `zx`'s gid[0]; each slope
                        // component d's RE col at this row's primary level gid[0]
                        // is (d+1)·n_primary + gid[0]. Reuses this crossed col `bl`
                        // — no re-derivation of crossed memberships. x widens
                        // f32→f64. Only the primary's crossed co-occurrence
                        // matters: a slope lives on the primary grouping, so the
                        // weight is x_{slope}; nested/other-crossed groupings carry
                        // no slope, so they contribute nothing here.
                        if slope {
                            for (d, &sc) in self.groupings.primary_slope_cols.iter().enumerate() {
                                // Slope covariate read AS A RANDOM-EFFECT column, so it
                                // takes the internal scale (`LmmGroupings::set_slope_scales`);
                                // the same x column keeps its raw value in the `c`/`s`
                                // fixed-effect scatters above. Already carries one zw.
                                let z = self.w_buf[sc] / self.groupings.primary_slope_scales[d];
                                // scol mirrors from_cluster_spec's RE-column layout — change together.
                                let scol = (d + 1) * n_prim + gid[0];
                                // b's side is the crossed intercept (z=1 → zw);
                                // total zw·zw = wᵢ, matching Σ wᵢ·x_slope·1.
                                self.zx_slope[(scol, bl)] += z * zw;
                            }
                        }
                    }
                }
            } else if self.groupings.k_crossed() > 0 {
                // Blocked crossed/nested-slopes path: fill `zx` with the FULL
                // covariate-weighted co-occurrence zx[(a_col, b_local)] = Σ z_a·z_b
                // over rows, for every (RE col a, crossed col b) on DISTINCT
                // groupings. z is 1 for an intercept component, x_{slope} for a slope
                // component. This subsumes the scalar path's counts + zx_slope; the
                // blocked tail reads all cross-factor coupling from here, the per-
                // level diagonal blocks from `s`/`counts`. zx_slope stays unused.
                let g = &self.groupings;
                let n_prim = g.n_primary;
                for bi in 0..n_g {
                    let b = gid[bi];
                    if b < kf {
                        continue; // only crossed columns own a `b_local`
                    }
                    let bl = b - kf;
                    let q_b = if bi == 0 {
                        g.primary_q
                    } else {
                        g.extra_q[bi - 1]
                    };
                    for db in 0..q_b {
                        // Every slope read here is a Z entry, so it takes the
                        // internal scale; the intercept's is exactly 1 by construction.
                        let z_b = if db == 0 {
                            zw // intercept z=1 → zw (one weight side)
                        } else if bi == 0 {
                            self.w_buf[g.primary_slope_cols[db - 1]]
                                / g.primary_slope_scales[db - 1]
                        } else {
                            self.w_buf[g.extra_slope_cols[bi - 1][db - 1]]
                                / g.extra_slope_scales[bi - 1][db - 1]
                        };
                        let b_local = bl + db;
                        for ai in 0..n_g {
                            if ai == bi {
                                continue;
                            }
                            let q_a = if ai == 0 {
                                g.primary_q
                            } else {
                                g.extra_q[ai - 1]
                            };
                            for da in 0..q_a {
                                let (a_col, z_a) = if da == 0 {
                                    (gid[ai], zw) // intercept z=1 → zw
                                } else if ai == 0 {
                                    (
                                        da * n_prim + gid[0],
                                        self.w_buf[g.primary_slope_cols[da - 1]]
                                            / g.primary_slope_scales[da - 1],
                                    )
                                } else {
                                    (
                                        gid[ai] + da,
                                        self.w_buf[g.extra_slope_cols[ai - 1][da - 1]]
                                            / g.extra_slope_scales[ai - 1][da - 1],
                                    )
                                };
                                self.zx[(a_col, b_local)] += z_a * z_b;
                            }
                        }
                    }
                }
            }
            // Primary slopes: each slope k's RE column at level gid[0] (offset
            // (k+1)·n_primary + gid[0]) accumulates z = x_{slope_k} weighted sums
            // into `s`; the intercept subcol (gid[0]) is already filled with z=1
            // above. counts is NOT incremented for slope subcols (the Gram reads
            // `s`, not counts). z and the [X y] weights widen f32→f64.
            if self.groupings.primary_q > 1 {
                let n_prim = self.groupings.n_primary;
                for (k, &sc) in self.groupings.primary_slope_cols.iter().enumerate() {
                    // Z entry ⇒ internal scale (see the zx_slope fill above).
                    let z = self.w_buf[sc] / self.groupings.primary_slope_scales[k];
                    let scol = (k + 1) * n_prim + gid[0];
                    let scol_mut = self
                        .s
                        .col_mut(scol)
                        .try_as_col_major_mut()
                        .unwrap()
                        .as_slice_mut();
                    #[allow(clippy::needless_range_loop)]
                    for j in 0..self.m {
                        scol_mut[j] += z * self.w_buf[j];
                    }
                }
            }
            // Extra-grouping slopes: slope d of grouping e accumulates z = x_{slope_d}
            // weighted [X y] into its RE column `gid[1+e] + 1 + d` (the q_g-wide level
            // block is [intercept | slope_0 | …]). The intercept subcol gid[1+e] is
            // already filled with z=1 by the `s` scatter above; counts is NOT
            // incremented for slope subcols (the Gram reads `s`). Same covariate-
            // weighted recipe as the primary; intercept-only extras scatter nothing.
            if self.groupings.extra_slopes_any {
                for e in 0..self.groupings.extra_slope_cols.len() {
                    let gintercept = gid[1 + e];
                    let n_d = self.groupings.extra_slope_cols[e].len();
                    for d in 0..n_d {
                        let sc = self.groupings.extra_slope_cols[e][d];
                        // Z entry ⇒ internal scale (see the zx_slope fill above).
                        let z = self.w_buf[sc] / self.groupings.extra_slope_scales[e][d];
                        let scol = gintercept + 1 + d;
                        let scol_mut = self
                            .s
                            .col_mut(scol)
                            .try_as_col_major_mut()
                            .unwrap()
                            .as_slice_mut();
                        #[allow(clippy::needless_range_loop)]
                        for j in 0..self.m {
                            scol_mut[j] += z * self.w_buf[j];
                        }
                    }
                }
            }
            if gid[0] + 1 > self.n_clusters {
                self.n_clusters = gid[0] + 1;
            }
        }
        self.n_rows += x.nrows();
    }
}

/// Balanced-collapse precompute: detect a balanced active prefix and
/// accumulate the θ-independent cross-Grams G_rr′ from the suff stats. Returns
/// false (and arms the fallback loop) when the design is unbalanced, has a
/// slope primary, or is empty. Balance = counts[f] equal over an active prefix
/// and zero after, per child slot c equal across active families (the
/// grid-atom-snapped layout; non-prefix actives are conservatively rejected).
/// `fit.collapse_stage` is this precompute's own staging buffer — its first
/// w·t_dim slots stage each family's raw rows.
///
/// Generic over the scratch's scalar: every field it writes is `f64` or
/// `usize` at any `T` (`fam_gram`, `collapse_stage`, `collapse_n_active`), and
/// the Grams it accumulates come from the `f64` sufficient statistics, so a
/// dual scratch arms exactly the same collapse the `f64` scratch does.
pub(crate) fn precompute_balanced_collapse<T: Scalar>(
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch<T>,
) -> bool {
    let g = &suff.groupings;
    fit.collapse_n_active = 0;
    if g.primary_q != 1 || g.n_primary == 0 || suff.n_rows == 0 {
        return false;
    }
    let np = g.nested_per_parent;
    let w = 1 + np;
    let kx = g.k_crossed();
    let m = suff.m;
    let t_dim = kx + m;
    let n0 = suff.counts[0];
    if n0 == 0.0 {
        return false;
    }
    let mut n_active = 1;
    while n_active < g.n_primary && suff.counts[n_active] == n0 {
        n_active += 1;
    }
    if suff.counts[n_active..g.n_primary].iter().any(|&c| c != 0.0) {
        return false; // hole or non-prefix layout — fall back
    }
    for c in 0..np {
        let c0 = suff.counts[g.n_primary + c]; // family 0, child slot c
        for f in 0..g.n_primary {
            let cc = suff.counts[g.n_primary + f * np + c];
            if (f < n_active && cc != c0) || (f >= n_active && cc != 0.0) {
                return false;
            }
        }
    }
    // Grams over the active prefix (inactive families are all-zero rows and
    // would contribute nothing anyway).
    let blk = t_dim * t_dim;
    let npairs = w * (w + 1) / 2;
    fit.fam_gram[..npairs * blk].fill(0.0);
    for f in 0..n_active {
        for r in 0..w {
            let gcol = if r == 0 {
                f
            } else {
                g.n_primary + f * np + (r - 1)
            };
            let dst = &mut fit.collapse_stage[r * t_dim..(r + 1) * t_dim];
            for (b, slot) in dst[..kx].iter_mut().enumerate() {
                *slot = suff.zx[(gcol, b)];
            }
            let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
            dst[kx..kx + m].copy_from_slice(scol);
        }
        let (bt, gram) = (&fit.collapse_stage, &mut fit.fam_gram);
        let mut pidx = 0;
        for r in 0..w {
            for rp in r..w {
                let gblk = &mut gram[pidx * blk..(pidx + 1) * blk];
                for j in 0..t_dim {
                    let vj = bt[rp * t_dim + j];
                    if vj != 0.0 {
                        for i in 0..t_dim {
                            gblk[j * t_dim + i] += bt[r * t_dim + i] * vj;
                        }
                    }
                }
                pidx += 1;
            }
        }
    }
    fit.collapse_n_active = n_active;
    true
}

// ---------------------------------------------------------------------------
// reml_deviance — the blocked-Cholesky objective.
// ---------------------------------------------------------------------------

/// Crossed/nested random-slopes REML deviance — the gated `extra_slopes_any`
/// path. Builds the full penalized augmented matrix
/// `P = [[ΛᵀZᵀZΛ + I, ΛᵀZᵀ[Xy]], [·, [Xy]ᵀ[Xy]]]` over `[all RE cols | X y]`
/// and takes ONE dense Cholesky. The crossed dimension is bounded (crossed forces
/// a FixedClusters primary), so `k = k_total` is independent of N. The block-
/// diagonal `Λ` carries each grouping's `q_g×q_g` relative-covariance factor; the
/// raw RE Gram `ZᵀZ` is recovered from the suff stats (per-level diagonal blocks
/// from `s`/`counts`, cross-factor blocks from the weighted `zx`). Same deviance
/// normalization as [`reml_deviance`] (`log|L_ZZ|² + log|L_XX|² + (N−P)·log σ̂²`),
/// so it reduces to the scalar value (to FP reassociation) when every `q_g == 1`.
///
/// Zero-alloc warm path: every buffer lives in `LmmFitScratch` (`blocked_*`),
/// sized once when `extra_slopes_any`. Returns INFINITY on any Cholesky failure.
fn reml_deviance_blocked(theta: &[f64], suff: &LmmSuffStats, fit: &mut LmmFitScratch<f64>) -> f64 {
    let g = &suff.groupings;
    let m = suff.m;
    let p = m - 1;
    let k = g.k_total;
    let dim = k + m;
    let n_prim = g.n_primary;
    let q_p = g.primary_q;
    let np = g.nested_per_parent;
    let prim_width = q_p * n_prim;
    let kf = g.k_family();

    // --- Λ (k×k, column-major lower-tri): block-diagonal per grouping/level ---
    fit.blocked_lam[..k * k].fill(0.0);
    // Primary: Λ_p on each level f's SCATTERED component columns {d·n_prim + f}.
    primary_lambda(theta, q_p, &mut fit.prim_lam);
    for f in 0..n_prim {
        for dc in 0..q_p {
            for dr in dc..q_p {
                let row = dr * n_prim + f;
                let col = dc * n_prim + f;
                fit.blocked_lam[col * k + row] = fit.prim_lam[dr * q_p + dc];
            }
        }
    }
    // Nested children: each child level's q_n×q_n Λ_n on its CONTIGUOUS block
    // [ic .. ic+q_n] — mirrors the crossed block below; the only difference is the
    // block stride (a child id is `f·np + c`, scattered after the primary block).
    // q_n==1 ⇒ a single scalar θ_n on the diagonal, byte-identical to the prior
    // intercept-only layout.
    if let Some(nf) = g.nested {
        let q_n = nf.q;
        let mut lam_n = [0.0f64; MAX_EXTRA_Q * MAX_EXTRA_Q];
        primary_lambda(&theta[nf.vech_start..], q_n, &mut lam_n);
        for f in 0..n_prim {
            for c in 0..np {
                let ic = prim_width + (f * np + c) * q_n;
                for dc in 0..q_n {
                    for dr in dc..q_n {
                        fit.blocked_lam[(ic + dc) * k + (ic + dr)] = lam_n[dr * q_n + dc];
                    }
                }
            }
        }
    }
    // Crossed: Λ_g on each level's CONTIGUOUS block [ic .. ic+q_g].
    let mut lam_g = [0.0f64; MAX_EXTRA_Q * MAX_EXTRA_Q];
    for cf in &g.crossed {
        let q = cf.q;
        primary_lambda(&theta[cf.vech_start..], q, &mut lam_g);
        let off = g.extra_offsets[cf.decl];
        for c in 0..cf.n_levels {
            let ic = off + c * q;
            for dc in 0..q {
                for dr in dc..q {
                    fit.blocked_lam[(ic + dc) * k + (ic + dr)] = lam_g[dr * q + dc];
                }
            }
        }
    }

    // --- raw RE design Gram G = ZᵀZ (k×k, full symmetric, column-major) ---
    // Step B: cross-factor coupling from the weighted `zx` (a = any RE col, b =
    // crossed col). Same-factor entries are 0 in `zx`; overwritten by step C.
    fit.blocked_g[..k * k].fill(0.0);
    for b in kf..k {
        let bl = b - kf;
        for a in 0..k {
            let v = suff.zx[(a, bl)];
            fit.blocked_g[b * k + a] = v;
            fit.blocked_g[a * k + b] = v;
        }
    }
    // Step C: per-level diagonal blocks from `s`/`counts`.
    // Primary family blocks G_f (component-major scatter).
    for f in 0..n_prim {
        primary_gram(suff, g, f, q_p, &mut fit.prim_gram);
        for dr in 0..q_p {
            for dc in 0..q_p {
                fit.blocked_g[(dc * n_prim + f) * k + (dr * n_prim + f)] =
                    fit.prim_gram[dr * q_p + dc];
            }
        }
    }
    // Nested children: per-child q_n×q_n diagonal Gram block + the primary↔child
    // cross-Gram. The diagonal block is a level's covariate-weighted scatter from
    // `s`/`counts`, identical in form to a crossed level's block (below). The cross
    // block is the within-family coupling — a nested child shares its rows with its
    // parent, so this q_p×q_n Σ_{child} z^{prim}·z^{child} is NOT in `zx` (which is
    // crossed-only). Entry (prim da, child dc): z=1 for an intercept component,
    // x_slope for a slope component; the four cases pick the matching `s`/`counts`
    // scatter. q_n==1 collapses to the prior n_c diagonal + the dc==0 cross column.
    if let Some(nf) = g.nested {
        let q_n = nf.q;
        let nscols = &g.extra_slope_cols[nf.decl];
        // `s`-ROW reads of a slope covariate below take the internal scale of the
        // row's own column; the `s`-COLUMN already carries its own (`primary_gram`).
        let nssc = &g.extra_slope_scales[nf.decl];
        for f in 0..n_prim {
            for c in 0..np {
                let ic = prim_width + (f * np + c) * q_n;
                let n_c = suff.counts[ic];
                for dr in 0..q_n {
                    for dc in 0..q_n {
                        let v = if dr == 0 && dc == 0 {
                            n_c
                        } else if dr == 0 {
                            suff.s[(nscols[dc - 1], ic)] / nssc[dc - 1] // Σ z_{dc-1}
                        } else if dc == 0 {
                            suff.s[(nscols[dr - 1], ic)] / nssc[dr - 1] // Σ z_{dr-1}
                        } else {
                            suff.s[(nscols[dr - 1], ic + dc)] / nssc[dr - 1] // Σ z_{dr-1} z_{dc-1}
                        };
                        fit.blocked_g[(ic + dc) * k + (ic + dr)] = v;
                    }
                }
                for da in 0..q_p {
                    let prow = da * n_prim + f;
                    for dc in 0..q_n {
                        let ccol = ic + dc;
                        let v = if da == 0 && dc == 0 {
                            n_c
                        } else if dc == 0 {
                            suff.s[(g.primary_slope_cols[da - 1], ic)]
                                / g.primary_slope_scales[da - 1] // Σ z^p_{da-1}
                        } else if da == 0 {
                            suff.s[(nscols[dc - 1], ic)] / nssc[dc - 1] // Σ z^n_{dc-1}
                        } else {
                            suff.s[(g.primary_slope_cols[da - 1], ic + dc)]
                                / g.primary_slope_scales[da - 1] // Σ z^p_{da-1} z^n_{dc-1}
                        };
                        fit.blocked_g[ccol * k + prow] = v;
                        fit.blocked_g[prow * k + ccol] = v;
                    }
                }
            }
        }
    }
    // Crossed diagonal blocks G_gc (intercept n_c, slope rows covariate-weighted).
    for cf in &g.crossed {
        let q = cf.q;
        let off = g.extra_offsets[cf.decl];
        let scols = &g.extra_slope_cols[cf.decl];
        // `s`-ROW side of the internal scale, as in the nested block above.
        let ssc = &g.extra_slope_scales[cf.decl];
        for c in 0..cf.n_levels {
            let ic = off + c * q;
            let n_c = suff.counts[ic];
            for dr in 0..q {
                for dc in 0..q {
                    let v = if dr == 0 && dc == 0 {
                        n_c
                    } else if dr == 0 {
                        suff.s[(scols[dc - 1], ic)] / ssc[dc - 1] // Σ z_{dc-1}
                    } else if dc == 0 {
                        suff.s[(scols[dr - 1], ic)] / ssc[dr - 1] // Σ z_{dr-1}
                    } else {
                        suff.s[(scols[dr - 1], ic + dc)] / ssc[dr - 1] // Σ z_{dr-1} z_{dc-1}
                    };
                    fit.blocked_g[(ic + dc) * k + (ic + dr)] = v;
                }
            }
        }
    }

    // --- penalized augmented matrix P (dim×dim, column-major lower-tri) ---
    // P_zz = Λᵀ G Λ + I via two block-diagonal-aware contractions (tmp = ΛᵀG).
    fit.blocked_tmp[..k * k].fill(0.0);
    for bp in 0..k {
        for a in 0..k {
            let mut acc = 0.0;
            for ap in 0..k {
                let l = fit.blocked_lam[a * k + ap]; // Λ[ap][a]
                if l != 0.0 {
                    acc += l * fit.blocked_g[bp * k + ap]; // G[ap][bp]
                }
            }
            fit.blocked_tmp[bp * k + a] = acc;
        }
    }
    for b in 0..k {
        for a in b..k {
            let mut acc = 0.0;
            for bp in 0..k {
                let l = fit.blocked_lam[b * k + bp]; // Λ[bp][b]
                if l != 0.0 {
                    acc += fit.blocked_tmp[bp * k + a] * l;
                }
            }
            if a == b {
                acc += 1.0; // + I
            }
            fit.blocked_p[b * dim + a] = acc;
        }
    }
    // P_zx = Λᵀ Zᵀ[Xy]: row (k+j), col a = Σ_{a'} Λ[a'][a]·s[(j, a')]. (Zᵀ[Xy] = s.)
    for a in 0..k {
        for j in 0..m {
            let mut acc = 0.0;
            for ap in 0..k {
                let l = fit.blocked_lam[a * k + ap];
                if l != 0.0 {
                    acc += l * suff.s[(j, ap)];
                }
            }
            fit.blocked_p[a * dim + (k + j)] = acc;
        }
    }
    // [Xy]ᵀ[Xy] block = suff.c (lower-tri).
    for j in 0..m {
        for i in j..m {
            fit.blocked_p[(k + j) * dim + (k + i)] = suff.c[(i, j)];
        }
    }

    // --- one dense Cholesky; read the deviance off the factor ---
    // Factor a separate `blocked_l` buffer at `Par::Seq`, never `blocked_p` itself
    // via `MatRef::llt` — `.llt` reads `faer::get_global_parallelism()`, which is
    // `Par::rayon(0)` whenever faer's default `rayon` feature is on, so above
    // dim 64 it would spawn and use the global rayon pool on every θ-eval even
    // without the `parallel` feature or `FitOptions::parallel_inner` (RULE 1).
    // `blocked_p` itself must stay unfactored: `recover_ranef_blocked` re-reads
    // it after the fit. Mirrors `<f64 as Scalar>::chol_lower` in scalar.rs.
    if !<f64 as Scalar>::chol_lower(&fit.blocked_p, dim, &mut fit.blocked_l) {
        return f64::INFINITY;
    }
    let l = faer::MatRef::from_column_major_slice(&fit.blocked_l[..dim * dim], dim, dim);
    let mut log_lzz_half = 0.0_f64;
    for i in 0..k {
        let lii = l[(i, i)];
        if !(lii.is_finite() && lii > 0.0) {
            return f64::INFINITY;
        }
        log_lzz_half += lii.ln();
    }
    let mut log_lxx_sq = 0.0_f64;
    for j in 0..p {
        let ljj = l[(k + j, k + j)];
        if !(ljj.is_finite() && ljj > 0.0) {
            return f64::INFINITY;
        }
        log_lxx_sq += ljj.ln();
    }
    log_lxx_sq *= 2.0;
    // Trailing m×m → fit.factor (recovery reads only this — augmented [X y] semantics).
    for j in 0..m {
        for i in 0..m {
            fit.factor[(i, j)] = if i >= j { l[(k + i, k + j)] } else { 0.0 };
        }
    }
    let lyy = fit.factor[(p, p)];
    let r_sq = lyy * lyy;
    let df = (suff.n_rows - p) as f64;
    let sigma_sq = r_sq / df;
    if !(sigma_sq.is_finite() && sigma_sq > 0.0) {
        return f64::INFINITY;
    }
    fit.sigma_sq = sigma_sq;
    2.0 * log_lzz_half + log_lxx_sq + df * sigma_sq.ln()
}

/// REML profiled deviance at θ via the family-blocked augmented Cholesky.
///
/// Ω_θ over [primary | nested children | crossed | X y]. The leading block is
/// block-diagonal per FAMILY (a primary level + its nested children — nested
/// children never co-occur across parents), so it is eliminated family-by-
/// family: factor the (1+n_per)² A_f, forward-solve its coupling to the
/// [crossed | X y] tail — cost linear in cluster count. The per-family tail
/// downdates are stacked into ONE triangular GEMM after the family loop
/// (Tail −= Bt·Bt′ over the solved couplings in `bt`; result-moving vs the
/// old sequential per-family subtraction).
/// Crossed factors couple everything (the dense Z_a′Z_b coupling, sanctioned
/// dense within the stated regime), so they stay in the tail with [X y]: one
/// dense (k_crossed+m) `Scalar::chol_lower`, out-of-place into `tail_l`, per
/// evaluation. With no extras this is the
/// per-cluster shrink downdate up to FP reassociation, and with no crossed
/// factors the tail is just the m×m [X y] block.
///
/// Balanced collapse (intercept-only primary): when the per-fit precompute
/// (`precompute_balanced_collapse`) finds a balanced active prefix — grid
/// atom-snapping guarantees one at production N — the family loop is replaced
/// by ONE Crout of the common A(θ), log|L_ZZ|'s family part by
/// n_active·log|L|, and the stacked-GEMM downdate by a θ-independent Gram
/// combine Σ_{r,r′} A⁻¹[r,r′]·scale_r·scale_r′·G_rr′, column-scaled by
/// diag(λ_x | 1). Reassociation-level result movement vs the loop; unbalanced
/// counts and the slope path (data-dependent A_f) keep the loop.
///
/// The deviance reads OFF THE FACTORS — log|L_ZZ|² from the family pivots +
/// the crossed tail diagonal, log|L_XX|², r² = L[p,p]² from the trailing m×m
/// block — no β backsolve per evaluation. Normalization convention:
///   dev(θ) = log|V| + log|X'V⁻¹X| + (N−P)·log(σ̂²),
/// so the collapse and general paths agree to FP error, not up to a
/// constant. Returns INFINITY on any Cholesky failure / non-positive σ̂².
///
/// θ is vech-packed per grouping — [primary, extras in declaration order]. The
/// primary block is width-general: `Λ_p` is the column-major vech θ prefix
/// (`q_p(q_p+1)/2` entries), and the per-level Gram `G_f` is recovered from `s`
/// with no new accumulator.
///
/// The composition: the q_p primary block coexists with the intercept-only
/// crossed/nested extra tail in one family-blocked elimination. The family block
/// is `q_p + nested_per_parent` wide; the new primary-slope↔nested-child
/// off-diagonal falls out of `s` (free), and the primary-slope↔crossed-factor
/// coupling reads the slope-weighted `zx_slope` twin (each slope row d at level f
/// is `zx_slope[(d·n_primary+f, b)]`, vs the intercept's unweighted `zx[(f, b)]`).
/// The extra-grouping scalars keep q_g = 1.
///
/// # Panics
///
/// If `suff.groupings.extra_slopes_any` and `T` is not `f64`: crossed/nested
/// random slopes route to the `f64`-only `reml_deviance_blocked` tail.
pub fn reml_deviance<T: Scalar + 'static>(
    theta: &[T],
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch<T>,
) -> T {
    let g = &suff.groupings;
    debug_assert_eq!(theta.len(), g.n_theta());
    let m = suff.m;
    let p = m - 1;
    if suff.n_rows <= p || p == 0 {
        return T::from_f64(f64::INFINITY);
    }
    // Crossed/nested random slopes route to the dense blocked path; the scalar
    // tail below stays byte-identical for every current (q_g==1) contract.
    // `reml_deviance_blocked` is `f64`-only — the crossed/nested-slopes tail
    // keeps a hand-written f64 route; a generic route is future work, so a
    // non-`f64` T must not reach it.
    if g.extra_slopes_any {
        assert!(
            T::IS_F64,
            "reml_deviance_blocked (crossed/nested slopes) is f64-only"
        );
        // Zero-alloc theta staging: `MAX_THETA` already bounds every fixed
        // θ-unpack scratch in this module (`lam_n`/`lam_g` above).
        let mut theta_f64 = [0.0f64; crate::consts::MAX_THETA];
        for (dst, src) in theta_f64.iter_mut().zip(theta.iter()) {
            *dst = src.value();
        }
        // `T::IS_F64` guarantees `T` and `f64` are the same type here; `Any` is
        // the safe way to recover that fact since the type system does not
        // track it structurally. No `unsafe`, no reinterpret cast.
        let fit64: &mut LmmFitScratch<f64> = (fit as &mut dyn std::any::Any)
            .downcast_mut()
            .expect("T::IS_F64 asserted above");
        return T::from_f64(reml_deviance_blocked(
            &theta_f64[..theta.len()],
            suff,
            fit64,
        ));
    }
    let kf = g.k_family();
    let kx = g.k_crossed();
    let t_dim = kx + m;
    let np = g.nested_per_parent;
    let w = g.primary_q + np; // width-general family width: q_p primary cols + nested children
    let th_p = theta[0];
    let th_n = g.nested.map(|nf| theta[nf.vech_start]).unwrap_or(T::ZERO);

    // Width-general primary factor (q_p ≥ 2 ⇒ slope path; q_p == 1 ⇒ scalar,
    // kept byte-identical). The slope path may now carry a crossed/nested tail
    // (the slope-composition). Λ_p is the vech-packed θ prefix, refreshed into
    // scratch (`fit.prim_lam`) so the hot loop stays zero-alloc.
    let slope = g.primary_q > 1;
    if slope {
        primary_lambda(theta, g.primary_q, &mut fit.prim_lam);
    }

    // λ per local crossed column. This scalar tail handles intercept-only extras
    // (q_g == 1, `vech_start` is the scalar θ index); slopes-on-extras (q_g > 1)
    // route to the blocked path before reaching here.
    debug_assert!(!g.extra_slopes_any);
    {
        let mut b = 0usize;
        for cf in &g.crossed {
            for _ in 0..cf.n_levels {
                fit.lam_x[b] = theta[cf.vech_start];
                b += 1;
            }
        }
    }

    // --- tail init: [[H, ·],[B_x, C]] (lower triangle, column-major) ---
    fit.tail[..t_dim * t_dim].fill(T::ZERO);
    for b in 0..kx {
        let lam = fit.lam_x[b];
        let gcol = kf + b;
        // Cross-factor coupling (row b in earlier columns a < b); same-factor
        // zx entries are structurally 0.
        let zxb = suff.zx.col(b).try_as_col_major().unwrap().as_slice();
        for a in 0..b {
            fit.tail[a * t_dim + b] = lam * fit.lam_x[a] * T::from_f64(zxb[kf + a]);
        }
        let scol = suff.s.col(gcol).try_as_col_major().unwrap().as_slice();
        let tcol = &mut fit.tail[b * t_dim..(b + 1) * t_dim];
        tcol[b] = T::ONE + lam * lam * T::from_f64(suff.counts[gcol]);
        for j in 0..m {
            tcol[kx + j] = lam * T::from_f64(scol[j]);
        }
    }
    for j in 0..m {
        let ccol = suff.c.col(j).try_as_col_major().unwrap().as_slice();
        let tcol = &mut fit.tail[(kx + j) * t_dim..(kx + j + 1) * t_dim];
        for (dst, &src) in tcol[kx + j..kx + m].iter_mut().zip(&ccol[j..m]) {
            *dst = T::from_f64(src);
        }
    }

    // --- family elimination ---
    // `collapse_n_active` is written only by `precompute_balanced_collapse`,
    // which every entry point arms once per call — the `f64` fit through
    // `fit_lmm_impl`, the dual gradient and Hessian through `reml_gradient` /
    // `reml_hessian`. A scratch that was never armed has it at 0 and takes the
    // per-family loop below: the same objective mathematically, but not
    // bit-equal to the collapse route (the collapse reassociates the family sum
    // into a θ-independent Gram combine).
    let collapse = !slope && fit.collapse_n_active > 0;
    let mut log_lzz_half = T::ZERO; // hoisted — single binding both arms write
    if collapse {
        let n_active = fit.collapse_n_active;
        // One representative A from the balanced prefix (family 0) — the
        // legacy q=1 fill verbatim.
        let n_f = T::from_f64(suff.counts[0]);
        fit.fam_a[0] = T::ONE + th_p * th_p * n_f;
        for c in 0..np {
            let n_c = T::from_f64(suff.counts[g.n_primary + c]);
            for c2 in 0..np {
                fit.fam_a[(1 + c) * w + (1 + c2)] = T::ZERO;
            }
            fit.fam_a[(1 + c) * w] = th_p * th_n * n_c;
            fit.fam_a[(1 + c) * w + (1 + c)] = T::ONE + th_n * th_n * n_c;
        }
        // Crout — the legacy in-place loop, one factor for all families.
        let mut log_l_half = T::ZERO;
        for j in 0..w {
            let mut d = fit.fam_a[j * w + j];
            for k in 0..j {
                let v = fit.fam_a[j * w + k];
                d -= v * v;
            }
            if !(d.value().is_finite() && d.value() > 0.0) {
                return T::from_f64(f64::INFINITY);
            }
            let l = d.sqrt();
            fit.fam_a[j * w + j] = l;
            log_l_half += l.ln();
            for i in (j + 1)..w {
                let mut v = fit.fam_a[i * w + j];
                for k in 0..j {
                    v -= fit.fam_a[i * w + k] * fit.fam_a[j * w + k];
                }
                fit.fam_a[i * w + j] = v / l;
            }
        }
        log_lzz_half = T::from_f64(n_active as f64) * log_l_half;
        // A⁻¹ = L⁻ᵀL⁻¹ column by column (w ≤ 1+n_per — hand-rolled). comb's
        // first w slots are the forward-solve temp; comb is refilled below.
        for r in 0..w {
            for i in 0..w {
                let mut acc = if i == r { T::ONE } else { T::ZERO };
                for k in 0..i {
                    acc -= fit.fam_a[i * w + k] * fit.comb[k];
                }
                fit.comb[i] = acc / fit.fam_a[i * w + i];
            }
            for i in (0..w).rev() {
                let mut acc = fit.comb[i];
                for k in (i + 1)..w {
                    acc -= fit.fam_a[k * w + i] * fit.a_inv[k * w + r];
                }
                fit.a_inv[i * w + r] = acc / fit.fam_a[i * w + i];
            }
        }
        // Combine: comb(lower) = Σ_{r≤r′} scale_r·scale_r′·A⁻¹[r,r′]·(G + [r≠r′]Gᵀ).
        let t2 = t_dim * t_dim;
        fit.comb[..t2].fill(T::ZERO);
        let (comb, gram) = (&mut fit.comb, &fit.fam_gram);
        let mut pidx = 0;
        for r in 0..w {
            let sr = if r == 0 { th_p } else { th_n };
            for rp in r..w {
                let srp = if rp == 0 { th_p } else { th_n };
                let coeff = sr * srp * fit.a_inv[r * w + rp];
                let gblk = &gram[pidx * t2..(pidx + 1) * t2];
                if coeff.value() != 0.0 {
                    if r == rp {
                        for j in 0..t_dim {
                            for i in j..t_dim {
                                comb[j * t_dim + i] += coeff * T::from_f64(gblk[j * t_dim + i]);
                            }
                        }
                    } else {
                        for j in 0..t_dim {
                            for i in j..t_dim {
                                comb[j * t_dim + i] +=
                                    coeff * T::from_f64(gblk[j * t_dim + i] + gblk[i * t_dim + j]);
                            }
                        }
                    }
                }
                pidx += 1;
            }
        }
        // Tail −= D·comb·D, D = diag(λ_x | 1_m) — column scaling folded here.
        for j in 0..t_dim {
            let dj = if j < kx { fit.lam_x[j] } else { T::ONE };
            for i in j..t_dim {
                let di = if i < kx { fit.lam_x[i] } else { T::ONE };
                fit.tail[j * t_dim + i] -= di * dj * fit.comb[j * t_dim + i];
            }
        }
    } else {
        for f in 0..g.n_primary {
            // A_f (w×w lower): the primary q_p×q_p block A_p = I + Λ′GΛ, then (on the
            // intercept-only scalar path) nested-child diags + parent–child counts. The
            // slope branch additionally carries the composed nested children;
            // the scalar/q_p=1 `else` stays byte-identical (q_p=1 parity).
            // Disjoint field borrows keep this zero-alloc and borrow-checked.
            assemble_fam_a(
                &mut fit.fam_a,
                &mut fit.prim_gram,
                &fit.prim_lam,
                suff,
                f,
                w,
                th_p,
                th_n,
                slope,
            );
            // In-place Crout Cholesky over the row-major w×w block, w ≤ 1+n_per,
            // via the shared kernel in `crate::linalg::block_chol` (zero-alloc;
            // false on a non-positive pivot, mapped to +INFINITY here — the
            // module's failure surface). Chains are ≤ w links — not chain-sick.
            // Pivots are multiplied into a per-family product (≤ w ≈ 9 terms,
            // each ≥ 1 since A_f's diagonal is 1.0 + θ²·n) and logged once
            // after the loop instead of once per pivot — log(∏l) = Σln(l),
            // same value, ~w× fewer .ln() calls. Must NOT accumulate this
            // product across the outer `f` loop: with ~60 families and θ up
            // to THETA_HI (1e3) during BOBYQA exploration, a global product
            // over ~480 terms can overflow f64 to +Infinity where the
            // per-family scoping (bounded product, reset each family) stays
            // finite.
            if !crate::linalg::block_chol(&mut fit.fam_a[..w * w], w) {
                return T::from_f64(f64::INFINITY);
            }
            let mut fam_prod = T::ONE;
            for j in 0..w {
                fam_prod *= fit.fam_a[j * w + j];
            }
            log_lzz_half += fam_prod.ln();
            // B_f (rows = Bt columns f·w..f·w+w, each contiguous): cols [crossed | X y],
            // then the forward-solve L_f⁻¹B_f in place on that same slice.
            let fb = f * w;
            let bt_fam = &mut fit.bt[fb * t_dim..(fb + w) * t_dim];
            assemble_fam_b(
                bt_fam,
                &fit.lam_x,
                &fit.prim_lam,
                suff,
                f,
                t_dim,
                kx,
                slope,
                th_p,
                th_n,
            );
            fam_forward_solve(bt_fam, t_dim, w, &fit.fam_a);
        }

        // --- one stacked downdate: Tail −= Σ_f B_f′B_f = Bt·Bt′ (lower) ---
        // The n_primary per-family rank-w tail re-traversals collapse into ONE
        // triangular GEMM through faer's blocked multi-accumulator FMA kernels
        // (Par::Seq — per-fit parallelism is the outer rayon loop). RESULT-MOVING:
        // GEMM accumulation order replaces the per-family sequential subtraction;
        // sanctioned, verified against the brute-force oracle + validation bands which
        // are orders wider than the reorder's last-ulp footprint.
        let w_tot = g.n_primary * w;
        // Disjoint field borrows: `bt` and `tail` are read/written by the
        // decomposition, `syrk_scratch` is its lane workspace.
        let LmmFitScratch {
            bt,
            tail,
            syrk_scratch,
            ..
        } = &mut *fit;
        T::syrk_lower_sub(
            &bt[..t_dim * w_tot],
            t_dim,
            w_tot,
            &mut tail[..t_dim * t_dim],
            syrk_scratch,
        );
    }

    // Out-of-place: `fit.tail` must survive holding the UNFACTORED matrix for
    // `recover_ranef_family`, which re-factors it at θ̂.
    if !T::chol_lower(
        &fit.tail[..t_dim * t_dim],
        t_dim,
        &mut fit.tail_l[..t_dim * t_dim],
    ) {
        return T::from_f64(f64::INFINITY);
    }
    for b in 0..kx {
        let lbb = fit.tail_l[b * t_dim + b];
        if !(lbb.value().is_finite() && lbb.value() > 0.0) {
            return T::from_f64(f64::INFINITY);
        }
        log_lzz_half += lbb.ln();
    }
    let log_lzz_sq = T::from_f64(2.0) * log_lzz_half;
    // Trailing m×m → fit.factor (augmented [X y] semantics; recovery reads only
    // this). Read off `tail_l`, not `fit.factor` — the copy is exact, so reading
    // either gives the same f64, but this is what lets `fit.factor` stay
    // `Mat<f64>` and receive `.value()`.
    for j in 0..m {
        for i in 0..m {
            fit.factor[(i, j)] = if i >= j {
                fit.tail_l[(kx + j) * t_dim + (kx + i)].value()
            } else {
                0.0
            };
        }
    }

    let mut log_lxx_sq = T::ZERO;
    for j in 0..p {
        let ljj = fit.tail_l[(kx + j) * t_dim + (kx + j)];
        if !(ljj.value().is_finite() && ljj.value() > 0.0) {
            return T::from_f64(f64::INFINITY);
        }
        log_lxx_sq += ljj.ln();
    }
    log_lxx_sq *= T::from_f64(2.0);

    let lyy = fit.tail_l[(kx + p) * t_dim + (kx + p)];
    let r_sq = lyy * lyy;
    let df = T::from_f64((suff.n_rows - p) as f64);
    let sigma_sq = r_sq / df;
    if !(sigma_sq.value().is_finite() && sigma_sq.value() > 0.0) {
        return T::from_f64(f64::INFINITY);
    }
    fit.sigma_sq = sigma_sq;

    log_lzz_sq + log_lxx_sq + df * sigma_sq.ln()
}

// ---------------------------------------------------------------------------
// REML dual gradient and Hessian — thin seed-call-read wrappers over
// `reml_deviance::<T>`. No PIRLS here: the REML criterion is a closed-form
// evaluation of the suff stats, so one dual call gives exact derivatives —
// unlike the GLMM side (`glmm::derivative`), there is no mode solve and no
// non-canonical refinement loop.
// ---------------------------------------------------------------------------

/// Dual-arithmetic twin of `LmmFitScratch`, sized to hold `n_theta` θ-lanes
/// for [`reml_gradient`]. `N` covers `n_theta` ALONE — β is profiled out of
/// the REML criterion, unlike the GLMM side's `N` (`n_theta + p`; see
/// `glmm::derivative`'s own module doc on where its dual buffers live and
/// how they are sized).
/// Same instantiated `{4, 5, 6, 8, 12}` lane set as the GLMM dual scratch's
/// `MAX_DUAL_N`/`GlmmDualScratch` (`glmm::derivative` — change together) —
/// 12 is the top rung; the gradient chunks above it (`⌈n_theta / 12⌉` passes
/// of `Dual<12>`), so no `n_theta` is out of reach.
///
/// Held by `LmmWorkspace` (`dual_scratch`), built on the first derivative
/// request and reused: the derivative diagnostics run once per converged fit,
/// so a per-fit local would reallocate the whole buffer list on every warm
/// refit — `bt` alone is `n_primary·w·t_dim` dual numbers.
pub(crate) enum LmmDualScratch {
    D4(LmmFitScratch<Dual<4>>),
    D5(LmmFitScratch<Dual<5>>),
    D6(LmmFitScratch<Dual<6>>),
    D8(LmmFitScratch<Dual<8>>),
    D12(LmmFitScratch<Dual<12>>),
}

impl LmmDualScratch {
    /// Allocate the smallest instantiated `N` covering `n_theta`, built by
    /// the existing generic `LmmFitScratch::with_groupings` constructor — no
    /// new buffer list. Always `Some` — above the top rung the gradient
    /// chunks, so the top rung covers every `n_theta`. The `Option` stays
    /// because [`LmmHyperScratch::for_groupings`], which mirrors this
    /// constructor, still refuses above 12.
    pub(crate) fn for_groupings(n_theta: usize, p: usize, g: &LmmGroupings) -> Option<Self> {
        Some(match n_theta {
            0..=4 => LmmDualScratch::D4(LmmFitScratch::with_groupings(p, g)),
            5 => LmmDualScratch::D5(LmmFitScratch::with_groupings(p, g)),
            6 => LmmDualScratch::D6(LmmFitScratch::with_groupings(p, g)),
            7..=8 => LmmDualScratch::D8(LmmFitScratch::with_groupings(p, g)),
            _ => LmmDualScratch::D12(LmmFitScratch::with_groupings(p, g)),
        })
    }
}

/// Hessian twin of [`LmmDualScratch`] — same lane set, `HyperDual<N, H>`
/// buffers, but capped: `None` iff `n_theta > 12`. A Hessian cannot chunk the
/// way the gradient does, because a cross-chunk second-derivative block needs
/// both coordinates' first-order lanes live in the same pass.
///
/// The variants' size gap (clippy's `large_enum_variant`) comes entirely from
/// `LmmFitScratch::sigma_sq: T`, the one field that is a bare `T` rather than
/// a `Vec<T>`/`Mat<f64>` — every other field is heap-backed and does not grow
/// with `H`. Reusing `LmmFitScratch` unmodified (no new buffer list) is the
/// point, so the allow stays rather than boxing a variant, which would only
/// move the same bytes to the heap for a value this type is always accessed
/// through `&mut`, never moved or copied.
#[allow(clippy::large_enum_variant)]
pub(crate) enum LmmHyperScratch {
    H4(LmmFitScratch<HyperDual<4, 10>>),
    H5(LmmFitScratch<HyperDual<5, 15>>),
    H6(LmmFitScratch<HyperDual<6, 21>>),
    H8(LmmFitScratch<HyperDual<8, 36>>),
    H12(LmmFitScratch<HyperDual<12, 78>>),
}

impl LmmHyperScratch {
    /// See [`LmmDualScratch::for_groupings`] — same ladder, `HyperDual`
    /// buffers, and `None` above the top rung rather than that constructor's
    /// chunking.
    pub(crate) fn for_groupings(n_theta: usize, p: usize, g: &LmmGroupings) -> Option<Self> {
        Some(match n_theta {
            0..=4 => LmmHyperScratch::H4(LmmFitScratch::with_groupings(p, g)),
            5 => LmmHyperScratch::H5(LmmFitScratch::with_groupings(p, g)),
            6 => LmmHyperScratch::H6(LmmFitScratch::with_groupings(p, g)),
            7..=8 => LmmHyperScratch::H8(LmmFitScratch::with_groupings(p, g)),
            9..=12 => LmmHyperScratch::H12(LmmFitScratch::with_groupings(p, g)),
            _ => return None,
        })
    }
}

/// θ-lane seeding, the `reml_deviance::<Dual<N>>` call, and the lane read —
/// the whole seed-call-read body [`reml_gradient`]'s per-`N` match arms hand a
/// typed scratch to. `Dual<N>`'s fields are public (`v`, `d`), so the unit-lane
/// seed is a plain struct literal — no `Seed` trait needed the way the GLMM
/// side has one (that trait also serves a multi-call refinement loop this
/// function has no equivalent of).
///
/// `theta.len()` may exceed `N`: the body then runs `⌈n_theta / N⌉` passes,
/// each seeding its own chunk of θ and writing that chunk's slice of `grad`.
/// One pass whenever `n_theta <= N`, which is every shape the ladder covers
/// exactly.
fn run_reml_gradient<const N: usize>(
    theta: &[f64],
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch<Dual<N>>,
    grad: &mut [f64],
) -> DerivStatus {
    let n_theta = theta.len();
    let zero = Dual::<N> {
        v: 0.0,
        d: [0.0; N],
    };
    let mut last = f64::NAN;
    // `⌈n_theta / N⌉` passes of at most `N` seeded coordinates: a pass gives
    // coordinate `offset + j` the lane `j`, and every other coordinate rides
    // along in `theta_full` as a plain value with zero lanes. Lanes are
    // independent in forward mode, so the concatenated slices are bit for bit
    // what one `Dual<n_theta>` pass would give. `MAX_THETA` is the crate's own
    // θ-length ceiling and already sizes every fixed θ-unpack buffer in this
    // module.
    let mut theta_full = [zero; crate::consts::MAX_THETA];
    for (j, &tj) in theta.iter().enumerate() {
        theta_full[j] = Dual { v: tj, d: [0.0; N] };
    }
    for offset in (0..n_theta).step_by(N) {
        let width = N.min(n_theta - offset);
        for t in theta_full[..n_theta].iter_mut() {
            t.d = [0.0; N];
        }
        for j in 0..width {
            theta_full[offset + j].d[j] = 1.0; // unit lane j — θ is linear in itself
        }
        let dev = reml_deviance(&theta_full[..n_theta], suff, fit);
        // The kernel's only failure signal is a bare +∞ (see reml_deviance's own
        // Cholesky/non-positive-σ̂² early returns) — `Dual::from_f64` zeroes the
        // lanes there, so checking the value alone is sufficient.
        if !dev.value().is_finite() {
            return DerivStatus::NotConverged;
        }
        grad[offset..offset + width].copy_from_slice(&dev.d[..width]);
        last = dev.value();
    }
    DerivStatus::Ok(last)
}

/// Gradient of the family-blocked REML deviance with respect to θ. `grad` has
/// `theta.len()` entries. `scratch` is sized by the caller ([`LmmDualScratch::for_groupings`]).
///
/// `Unsupported`: the crossed/nested-slopes path
/// (`suff.groupings.extra_slopes_any`, which routes `reml_deviance` to the
/// `f64`-only `reml_deviance_blocked` and would otherwise panic on a non-`f64`
/// `T` — checked here so that panic is never reached). A `theta.len()` above
/// the top rung is NOT refused: the gradient chunks, `⌈n_θ / 12⌉` passes of
/// `13` lane-evaluations each, so any `scratch` variant serves any
/// `theta.len()`.
/// `NotConverged`: the evaluated deviance is non-finite.
pub(crate) fn reml_gradient(
    theta: &[f64],
    suff: &LmmSuffStats,
    scratch: &mut LmmDualScratch,
    grad: &mut [f64],
) -> DerivStatus {
    debug_assert_eq!(theta.len(), suff.groupings.n_theta());
    if suff.groupings.extra_slopes_any {
        return DerivStatus::Unsupported;
    }
    // The collapse is a pure function of `suff` (a full overwrite of the
    // collapse buffers, not an incremental accumulate), so arming it once per
    // call here is correct however many times the caller reuses the scratch.
    // Without this the dual objective runs the per-family loop and the whole
    // `t_dim × w_tot` syrk on designs where the `f64` objective takes one `w×w`
    // Crout and a θ-independent Gram combine.
    match scratch {
        LmmDualScratch::D4(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_gradient::<4>(theta, suff, fit, grad)
        }
        LmmDualScratch::D5(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_gradient::<5>(theta, suff, fit, grad)
        }
        LmmDualScratch::D6(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_gradient::<6>(theta, suff, fit, grad)
        }
        LmmDualScratch::D8(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_gradient::<8>(theta, suff, fit, grad)
        }
        LmmDualScratch::D12(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_gradient::<12>(theta, suff, fit, grad)
        }
    }
}

/// Hessian twin of [`run_reml_gradient`] — same seed-call-read shape, one
/// `reml_deviance::<HyperDual<N, H>>` call gives exact first AND second
/// derivatives (no refinement loop needed here either).
fn run_reml_hessian<const N: usize, const H: usize>(
    theta: &[f64],
    suff: &LmmSuffStats,
    fit: &mut LmmFitScratch<HyperDual<N, H>>,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    let n_theta = theta.len();
    if n_theta > N {
        return DerivStatus::Unsupported;
    }
    let zero = HyperDual::<N, H> {
        v: 0.0,
        d: [0.0; N],
        h: [0.0; H],
    };
    let mut theta_t = [zero; N];
    for (j, &tj) in theta.iter().enumerate() {
        let mut d = [0.0; N];
        d[j] = 1.0;
        // A coordinate is linear in itself, so its own second derivative is
        // zero — `h` stays all-zero (mirrors `dual::HyperDual`'s own `Seed`
        // convention on the GLMM side).
        theta_t[j] = HyperDual {
            v: tj,
            d,
            h: [0.0; H],
        };
    }
    let dev = reml_deviance(&theta_t[..n_theta], suff, fit);
    if !dev.value().is_finite() {
        return DerivStatus::NotConverged;
    }
    grad[..n_theta].copy_from_slice(&dev.d[..n_theta]);
    let hlen = n_theta * (n_theta + 1) / 2;
    unpack_hessian(hess, &dev.h[..hlen], n_theta);
    DerivStatus::Ok(dev.value())
}

/// Gradient and exact Hessian of the same criterion [`reml_gradient`]
/// differentiates. `hess` is an `n_theta × n_theta` `Mat<f64>`, both triangles
/// filled (the packed hyper-dual lower triangle mirrored) — same output-shape
/// rule as `glmm::derivative::laplace_hessian`. `Unsupported` for
/// `extra_slopes_any` and for `n_theta > MAX` (12, the top instantiated
/// `HyperDual` rung) — unlike `reml_gradient`, a Hessian cannot chunk above
/// that rung, because a cross-chunk second-derivative block needs both
/// coordinates' first-order lanes in one pass. `NotConverged` on a
/// non-finite deviance.
pub(crate) fn reml_hessian(
    theta: &[f64],
    suff: &LmmSuffStats,
    scratch: &mut LmmHyperScratch,
    grad: &mut [f64],
    hess: &mut Mat<f64>,
) -> DerivStatus {
    debug_assert_eq!(theta.len(), suff.groupings.n_theta());
    if suff.groupings.extra_slopes_any {
        return DerivStatus::Unsupported;
    }
    // The collapse is a pure function of `suff` (a full overwrite of the
    // collapse buffers, not an incremental accumulate), so arming it once per
    // call here is correct however many times the caller reuses the scratch.
    // Without this the dual objective runs the per-family loop and the whole
    // `t_dim × w_tot` syrk on designs where the `f64` objective takes one `w×w`
    // Crout and a θ-independent Gram combine.
    match scratch {
        LmmHyperScratch::H4(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_hessian::<4, 10>(theta, suff, fit, grad, hess)
        }
        LmmHyperScratch::H5(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_hessian::<5, 15>(theta, suff, fit, grad, hess)
        }
        LmmHyperScratch::H6(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_hessian::<6, 21>(theta, suff, fit, grad, hess)
        }
        LmmHyperScratch::H8(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_hessian::<8, 36>(theta, suff, fit, grad, hess)
        }
        LmmHyperScratch::H12(fit) => {
            precompute_balanced_collapse(suff, fit);
            run_reml_hessian::<12, 78>(theta, suff, fit, grad, hess)
        }
    }
}
