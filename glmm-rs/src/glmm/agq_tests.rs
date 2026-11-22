use super::*;
use crate::family::dev_resid_eta;
use crate::spec::{GammaLink, PoissonLink};

fn unif(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// The compressed per-cluster node sum against the row-by-row kernel
/// (`Σ_j exp(ln_wj + ℓ_c(u_cj) − ℓ_c(ũ_c))` against `Σ_j w_j + Σ_j w_j·expm1(x_j)`,
/// with `A_c − 1 = 1/σ² − 1`, and `ℓ_c` summed row by row through
/// `dev_resid_eta`, the node predictor rebuilt as `η_fix + λ·u_cj` as the AGQ loop
/// does), on random clusters: 1 to 60 rows, mode η in [−3, 3] from a fixed part plus
/// an offset, zero and positive weights, Poisson counts with many zeros, positive
/// Gamma responses, either sign of λ, k = 3, 7, 25. Compared on `ln Σ`, the
/// quantity that enters the deviance, scaled by `max(1, |a|, |b|)`.
#[test]
fn compressed_cluster_sum_matches_row_kernel() {
    let mut st = 20261005u64;
    for family in [
        Family::Poisson {
            link: PoissonLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
    ] {
        assert!(compressible(family));
        let gamma = matches!(family, Family::Gamma { .. });
        let mut worst = 0.0f64;
        for k in [3usize, 7, 25] {
            let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
            let nodes = &crate::consts::GH_NODES[blk..blk + k];
            let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
            let ln_wj: Vec<f64> = nodes
                .iter()
                .zip(wts)
                .map(|(&z, &w)| w.ln() + z * z)
                .collect();
            for _ in 0..400 {
                let m = 1 + (unif(&mut st) * 60.0) as usize;
                let lambda =
                    (0.05 + 1.5 * unif(&mut st)) * if unif(&mut st) < 0.2 { -1.0 } else { 1.0 };
                let u_c = 3.0 * (unif(&mut st) - 0.5);
                let sigma = 0.05 + 0.9 * unif(&mut st);
                let (mut sa, mut sb) = (0.0f64, 0.0f64);
                let mut ctr = -0.5 * u_c * u_c;
                let mut rows = Vec::with_capacity(m);
                for _ in 0..m {
                    let offset = 2.0 * (unif(&mut st) - 0.5);
                    let eta0_target = 6.0 * (unif(&mut st) - 0.5);
                    let eta_fix = (eta0_target - offset - lambda * u_c) + offset;
                    let eta0 = eta_fix + lambda * u_c;
                    let mu0 = eta0.exp();
                    let a = if unif(&mut st) < 0.15 {
                        0.0
                    } else {
                        0.2 + 3.0 * unif(&mut st)
                    };
                    let y = if gamma {
                        mu0 * (0.2 + 2.0 * unif(&mut st))
                    } else if unif(&mut st) < 0.3 {
                        0.0
                    } else {
                        (mu0 * 2.0 * unif(&mut st)).round()
                    };
                    if gamma {
                        sa += a;
                        sb += a * y / mu0;
                    } else {
                        sa += a * y;
                        sb += a * mu0;
                    }
                    ctr -= 0.5 * a * dev_resid_eta(family, 0.0, y, eta0);
                    rows.push((eta_fix, y, a));
                }
                let mut row_sum = 0.0f64;
                for (j, &zj) in nodes.iter().enumerate() {
                    let u_cj = u_c + std::f64::consts::SQRT_2 * sigma * zj;
                    let mut acc = -0.5 * u_cj * u_cj;
                    for &(eta_fix, y, a) in &rows {
                        acc -= 0.5 * a * dev_resid_eta(family, 0.0, y, eta_fix + lambda * u_cj);
                    }
                    row_sum += (ln_wj[j] + acc - ctr).exp();
                }
                let excess = 1.0 / (sigma * sigma) - 1.0;
                let comp =
                    compressed_cluster_sum(gamma, sa, sb, u_c, sigma, excess, lambda, nodes, wts);
                let wsum: f64 = wts.iter().sum();
                let (a, b) = (row_sum.ln(), wsum.ln() + (comp / wsum).ln_1p());
                let err = (a - b).abs() / 1f64.max(a.abs()).max(b.abs());
                assert!(
                    err <= 1e-12,
                    "{family:?} k={k} m={m}: row {a} vs compressed {b} (scaled {err:e})"
                );
                worst = worst.max(err);
            }
        }
        eprintln!("{family:?}: worst scaled error {worst:e}");
    }
}

/// The generic route's compression gate on a Poisson-log cluster whose one
/// large row is past `agq_fast`'s `TERM_MAG_MAX` (μ₀ = y = 1e12 at a node-shift
/// bound of 1e-5: parts near 2e7 against 2^20) and on the same row below it
/// (μ₀ = y = 1e9: parts near 2e4), beside a cluster of small counts, at `f64`
/// and at a dual type. Both centers sit far inside the η window, so only the
/// row guard in `compression_gather` refuses the first case; `fast_consts`
/// at the same row is the `f64` route's guard, asserted alongside.
#[test]
fn compression_gate_refuses_a_row_past_the_magnitude_bound() {
    fn gate<T: Scalar>(mu_big: f64) -> [bool; 2] {
        let family = Family::Poisson {
            link: PoissonLink::Log,
        };
        let ids = [0u32, 0, 1, 1];
        let mu0 = [mu_big, 2.0, 1.5, 3.0];
        let y = [mu_big, 1.0, 2.0, 3.0];
        let w = [1.0; 4];
        let eta: Vec<T> = mu0.iter().map(|&m| T::from_f64(m.ln())).collect();
        let prob: Vec<T> = mu0.iter().map(|&m| T::from_f64(m)).collect();
        let b = 1e-5;
        let (mut sa, mut sb, mut reach) = ([T::ZERO; 2], [T::ZERO; 2], [T::ZERO; 2]);
        compression_gather(
            family,
            0.0,
            &y,
            &w,
            &ids,
            &eta,
            &prob,
            &|_| b,
            &mut sa,
            &mut sb,
            &mut reach,
        );
        [0, 1].map(|c| compress_ok(reach[c].value(), b, sa[c].value(), sb[c].value()))
    }
    let pois = Family::Poisson {
        link: PoissonLink::Log,
    };
    let b = 1e-5;
    let g = super::super::agq_fast::mag_factor(super::super::agq_fast::Kind::Lin, b);
    for (mu, past) in [(1e12, true), (1e9, false)] {
        let f64_guard = super::super::agq_fast::fast_consts(pois, 0.0, mu, 1.0, mu.ln(), mu, b, g);
        assert_eq!(f64_guard.is_none(), past, "μ₀ = {mu:e}: f64 row guard");
        let want = [!past, true];
        assert_eq!(gate::<f64>(mu), want, "μ₀ = {mu:e}: f64 gate");
        assert_eq!(
            gate::<crate::dual::Dual<2>>(mu),
            want,
            "μ₀ = {mu:e}: dual gate"
        );
    }
}

/// `vec_prior`'s `−h·(ũ − ½·B h)` against the exponent's first form
/// `‖z‖² − ½‖ũ + h‖² + ½‖ũ‖²`, with `h = √2·L⁻ᵀz` from
/// `linalg::block_chol_unit_shift` of `B`, at q = 2 and 3 over every node of
/// a 3-point grid. At `B` and `ũ` of order 1 the first form is evaluated as
/// written. At order 1e-100 it rounds to the cancellation of its O(1) parts,
/// so the reference is its first-order term `zᵀB₀z − √2·ũ₀·z` times the
/// scale, exact to O(1e-100) relative.
#[test]
fn vec_prior_matches_the_z_form() {
    let mut st = 20261119u64;
    for q in [2usize, 3] {
        for scale in [1.0, 1e-100] {
            // B₀ = MMᵀ (lower triangle stored), ũ₀ in [−1, 1].
            let mm: Vec<f64> = (0..q * q).map(|_| 2.0 * unif(&mut st) - 1.0).collect();
            let mut b0 = vec![0.0; q * q];
            for r in 0..q {
                for c in 0..=r {
                    b0[r * q + c] = (0..q).map(|k| mm[r * q + k] * mm[c * q + k]).sum();
                }
            }
            let u0: Vec<f64> = (0..q).map(|_| 2.0 * unif(&mut st) - 1.0).collect();
            let b: Vec<f64> = b0.iter().map(|v| v * scale).collect();
            let u: Vec<f64> = u0.iter().map(|v| v * scale).collect();
            let mut l = b.clone();
            crate::linalg::block_chol_unit_shift(&mut l, q).expect("A = I + B is SPD");
            let pts = [-1.224744871391589, 0.0, 1.224744871391589];
            for t in 0..pts.len().pow(q as u32) {
                let z: Vec<f64> = (0..q).map(|d| pts[t / 3usize.pow(d as u32) % 3]).collect();
                let mut h = [0.0f64; 3];
                for r in (0..q).rev() {
                    let mut v = z[r];
                    for cc in (r + 1)..q {
                        v -= l[cc * q + r] * h[cc];
                    }
                    h[r] = v / l[r * q + r];
                }
                for hr in &mut h[..q] {
                    *hr *= std::f64::consts::SQRT_2;
                }
                let got = vec_prior(&h, &u, &b, q);
                let want = if scale == 1.0 {
                    (0..q)
                        .map(|r| z[r] * z[r] - 0.5 * (u[r] + h[r]).powi(2) + 0.5 * u[r] * u[r])
                        .sum::<f64>()
                } else {
                    let mut zbz = 0.0;
                    for r in 0..q {
                        for c in 0..q {
                            let (hi, lo) = if c > r { (c, r) } else { (r, c) };
                            zbz += z[r] * b0[hi * q + lo] * z[c];
                        }
                    }
                    let uz: f64 = (0..q).map(|r| u0[r] * z[r]).sum();
                    scale * (zbz - std::f64::consts::SQRT_2 * uz)
                };
                let err = (got - want).abs() / (scale + want.abs());
                assert!(
                    err <= 1e-13,
                    "q = {q}, scale {scale:e}, z = {z:?}: vec_prior {got:e} vs {want:e}"
                );
            }
        }
    }
}

/// The compressed route's eligibility: inside the η window and finite sums only;
/// at or past the window, or on a non-finite sum, the cluster keeps the row kernel.
#[test]
fn compress_ok_window_and_non_finite() {
    assert!(compress_ok(150.0, 50.0, 1.0, 1.0));
    assert!(!compress_ok(150.0, 50.5, 1.0, 1.0));
    assert!(!compress_ok(f64::NAN, 0.0, 1.0, 1.0));
    assert!(!compress_ok(1.0, 0.0, f64::INFINITY, 1.0));
    assert!(!compress_ok(1.0, 0.0, 1.0, f64::NAN));
    assert!(!compressible(Family::Gamma {
        link: GammaLink::Inverse
    }));
    assert!(!compressible(Family::NegativeBinomial {
        link: crate::spec::NegBinomialLink::Log
    }));
}
