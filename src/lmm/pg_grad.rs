//! The θ-gradient of the profiled REML deviance on the per-group regression
//! route, read off the state the last [`reml_deviance`](super::reml_deviance)
//! evaluation left.
//!
//! On the route (see `kernel::precompute_group_regressions`) the tail is
//! `T = P₀ + Σ_g A_g′K_g⁻¹A_g` with `K_g = D + B_g`, `D = ΛΛ′`, `B_g = G_g⁻¹`,
//! and with `ν = n − p`, `r² = e′Te`, `e = [−β̂; 1]` the deviance is
//!
//! ```text
//!   f = Σ_g ln|G_g| + Σ_g ln|K_g| + ln|T_xx| + ν ln(r²/ν)    (+ θ-free constants)
//! ```
//!
//! Differentiating through `D` (the only θ-dependent piece) gives
//!
//! ```text
//!   ∂f/∂D = Σ_g L_g⁻ᵀ (I − N_g N_g′ − (ν/r²) m_g m_g′) L_g⁻¹,
//!     K_g = L_g L_g′,   M_g = L_g⁻¹ A_g  (the value pass's `bt` block),
//!     N_g = M_{g,x} L_xx⁻ᵀ,   m_g = M_g e,
//!   ∂f/∂Λ = 2 (∂f/∂D) Λ,   ∂f/∂θ = the lower triangle of ∂f/∂Λ in vech order.
//! ```
//!
//! The three terms are `∂ ln|K_g|`, `∂ ln|T_xx|` and `∂ ν ln r²`: `T` depends
//! on `D` only through `A_g′K_g⁻¹A_g`, whose derivative is `−M_g′ (∂K_g) M_g`
//! after the `L_g` whitening, and `T_xx⁻¹`, `r⁻²` contract it with `N_g` and
//! `m_g` (`L_xx` is the leading `p×p` block of the tail factor and
//! `r² = l_yy²`). Verified against central finite differences of the
//! deviance in `tests.rs` (`pg_gradient_matches_finite_differences`).
//!
//! Cost: one `p×p` triangular solve on `p × Σq` columns (stacked, processed
//! [`PG_GRAD_BLOCK`] groups at a time so its scratch is bounded) plus `O(q³)`
//! per group in a kernel monomorphized over `q`; `L_g` comes from the value
//! pass (`pg_l`). Measured 2026-10-07 (pawel P-core): value plus gradient
//! 1.35x a value at `q = 2`, 1.43x at `q = 4`, 1.99x at `q = 8`.

use super::{LmmFitScratch, LmmSuffStats, MAX_PRIMARY_Q};

/// Groups per stacked triangular solve. Bounds the gradient's scratch
/// (`pg_nt`, `p·q·PG_GRAD_BLOCK` f64) independently of the group count; the
/// blocks run in group order and every column is solved alone, so the result
/// is the same for any block size.
pub(crate) const PG_GRAD_BLOCK: usize = 512;

/// Writes `∂f/∂θ` at the θ of the last evaluation into `grad`. Requires that
/// evaluation to have been on the route and finite (`pg_n_active > 0`, the
/// tail factor and `bt` current); returns `false` when a pivot of the border
/// factor is not positive or a value is not finite.
#[allow(clippy::needless_range_loop)] // dense small-matrix loops index several arrays
pub(crate) fn pg_gradient(suff: &LmmSuffStats, fit: &mut LmmFitScratch, grad: &mut [f64]) -> bool {
    debug_assert!(fit.pg_n_active > 0);
    let q = suff.groupings.primary_q;
    debug_assert_eq!(grad.len(), q * (q + 1) / 2);
    let m = suff.m;
    let p = m - 1;
    let nu = (suff.n_rows - p) as f64;
    // β̂ off the border factor: L_xx′ β = l_yx (column-major, read by columns).
    for j in (0..p).rev() {
        let col = fit.factor.col_as_slice(j);
        let mut acc = col[p];
        for k in (j + 1)..p {
            acc -= col[k] * fit.pg_beta[k];
        }
        if col[j].is_nan() || col[j] <= 0.0 {
            return false;
        }
        fit.pg_beta[j] = acc / col[j];
    }
    let lyy = fit.factor.col_as_slice(p)[p];
    if lyy.is_nan() || lyy <= 0.0 {
        return false;
    }
    let c = nu / (lyy * lyy);
    let mut acc = [0.0f64; MAX_PRIMARY_Q * MAX_PRIMARY_Q];
    let n_active = fit.pg_n_active;
    let mut g0 = 0;
    while g0 < n_active {
        let g1 = (g0 + PG_GRAD_BLOCK).min(n_active);
        let cols = (g1 - g0) * q;
        // Stage N_gᵀ's right-hand sides (the x rows of the block's bt columns)
        // and m_g = M_g e.
        for w in 0..cols {
            let col = &fit.bt[(g0 * q + w) * m..(g0 * q + w + 1) * m];
            fit.pg_nt[w * p..(w + 1) * p].copy_from_slice(&col[..p]);
            let mut mv = col[p];
            for (cj, bj) in col[..p].iter().zip(&fit.pg_beta) {
                mv -= cj * bj;
            }
            fit.pg_mv[w] = mv;
        }
        if p > 0 {
            faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                fit.factor.as_ref().submatrix(0, 0, p, p),
                faer::MatMut::from_column_major_slice_mut(&mut fit.pg_nt[..p * cols], p, cols),
                faer::Par::Seq,
            );
        }
        match q {
            1 => groups::<1>(fit, g0..g1, p, c, &mut acc),
            2 => groups::<2>(fit, g0..g1, p, c, &mut acc),
            3 => groups::<3>(fit, g0..g1, p, c, &mut acc),
            4 => groups::<4>(fit, g0..g1, p, c, &mut acc),
            5 => groups::<5>(fit, g0..g1, p, c, &mut acc),
            6 => groups::<6>(fit, g0..g1, p, c, &mut acc),
            7 => groups::<7>(fit, g0..g1, p, c, &mut acc),
            8 => groups::<8>(fit, g0..g1, p, c, &mut acc),
            _ => unreachable!("the route admits q ≤ MAX_PRIMARY_Q = 8"),
        }
        g0 = g1;
    }
    // ∂f/∂Λ = 2 F Λ with F = ∂f/∂D (symmetric; `acc` holds its lower
    // triangle, row-major at stride q), vech order: column-major lower.
    let fsym = |r: usize, k: usize| {
        if k <= r {
            acc[r * q + k]
        } else {
            acc[k * q + r]
        }
    };
    let lam = &fit.prim_lam;
    let mut t = 0;
    for cc in 0..q {
        for r in cc..q {
            let mut v = 0.0;
            for k in 0..q {
                v += fsym(r, k) * lam[k * q + cc];
            }
            grad[t] = 2.0 * v;
            t += 1;
        }
    }
    grad.iter().all(|v| v.is_finite())
}

/// Adds `Σ_g L_g⁻ᵀ H_g L_g⁻¹` over `range` (groups in order) into the lower
/// triangle of `acc` (row-major, stride `Q`), `H_g = I − N_g N_g′ − c m_g m_g′`
/// with `N_g`, `m_g` the block-local columns staged in `pg_nt` / `pg_mv`.
#[allow(clippy::needless_range_loop)] // dense small-matrix loops index several arrays
#[inline(always)]
fn groups<const Q: usize>(
    fit: &LmmFitScratch,
    range: std::ops::Range<usize>,
    p: usize,
    c: f64,
    acc: &mut [f64; MAX_PRIMARY_Q * MAX_PRIMARY_Q],
) {
    let qq = Q * Q;
    let g0 = range.start;
    for gi in range {
        let lg = &fit.pg_l[gi * qq..(gi + 1) * qq];
        // L_g⁻¹ (lower), one reciprocal per pivot.
        let mut li = [[0.0f64; Q]; Q];
        for r in 0..Q {
            let rd = 1.0 / lg[r * Q + r];
            li[r][r] = rd;
            for cc in 0..r {
                let mut s = 0.0;
                for k in cc..r {
                    s += lg[r * Q + k] * li[k][cc];
                }
                li[r][cc] = -rd * s;
            }
        }
        let base = (gi - g0) * Q;
        let mut h = [[0.0f64; Q]; Q];
        for r in 0..Q {
            let nr = &fit.pg_nt[(base + r) * p..(base + r + 1) * p];
            for s in 0..=r {
                let ns = &fit.pg_nt[(base + s) * p..(base + s + 1) * p];
                let mut v = if r == s { 1.0 } else { 0.0 };
                for (a, b) in nr.iter().zip(ns) {
                    v -= a * b;
                }
                v -= c * fit.pg_mv[base + r] * fit.pg_mv[base + s];
                h[r][s] = v;
                h[s][r] = v;
            }
        }
        // T = H L⁻¹, then F_g = L⁻ᵀ T (lower half).
        let mut t = [[0.0f64; Q]; Q];
        for r in 0..Q {
            for cc in 0..Q {
                let mut s = 0.0;
                for k in cc..Q {
                    s += h[r][k] * li[k][cc];
                }
                t[r][cc] = s;
            }
        }
        for a in 0..Q {
            for cc in 0..=a {
                let mut s = 0.0;
                for r in a..Q {
                    s += li[r][a] * t[r][cc];
                }
                acc[a * Q + cc] += s;
            }
        }
    }
}
