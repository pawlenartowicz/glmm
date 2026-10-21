use super::*;
use crate::standardize::ColumnScaling;
use crate::{Family, ModelSpec, ReStructure, Sizing};

/// Groupings for one block (1 + x1 + x2 | g), slope columns 1 and 2 of a p=3 X.
fn one_block(q_slopes: &[usize]) -> LmmGroupings {
    let spec = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 4 },
            slopes: q_slopes.iter().map(|&c| c as u32).collect(),
            extra_groupings: vec![],
        }),
    };
    LmmGroupings::from_cluster_spec_ext(&spec, 16, q_slopes, &[])
}

/// A scaling with intercept in column 0 and the given (m, s) on columns 1..
fn scaling(cols: &[(f64, f64)]) -> ColumnScaling {
    // Build through `compute` on a design whose weighted mean / sd are the
    // requested ones: two rows per column at m ± s.
    let p = 1 + cols.len();
    let n = 2;
    let mut x = vec![1.0; n * p];
    for (k, &(m, s)) in cols.iter().enumerate() {
        x[k + 1] = m - s;
        x[p + k + 1] = m + s;
    }
    let mut cs = ColumnScaling::identity(p);
    cs.compute(&x, n, p, None, true);
    cs
}

fn vech_to_sigma(theta: &[f64], q: usize) -> Vec<f64> {
    let mut lam = vec![0.0; q * q];
    primary_lambda(theta, q, &mut lam);
    let mut sig = vec![0.0; q * q];
    for r in 0..q {
        for c in 0..q {
            sig[r * q + c] = (0..q).map(|k| lam[r * q + k] * lam[c * q + k]).sum();
        }
    }
    sig
}

#[test]
fn theta_round_trip_is_the_identity_full_rank() {
    let g = one_block(&[1, 2]);
    let cs = scaling(&[(1000.0, 3.0), (-2.0, 0.01)]);
    let caller = [0.9, 0.2, -0.1, 0.5, 0.05, 0.3]; // column-major vech, q = 3
    let mut scratch = vec![0.0; g.re_map_scratch_len()];
    let mut t = caller;
    g.theta_caller_to_design(&mut t, &mut scratch, &cs);
    let mut back = [0.0; 6];
    g.theta_kernel_to_caller(&t, &mut back, &mut scratch, &cs);
    let (a, b) = (vech_to_sigma(&caller, 3), vech_to_sigma(&back, 3));
    for k in 0..9 {
        assert!(
            (a[k] - b[k]).abs() <= 1e-10 * (1.0 + a[k].abs()),
            "Σ[{k}] {} vs {}",
            a[k],
            b[k]
        );
    }
    for k in 0..6 {
        assert!(
            (back[k] - caller[k]).abs() <= 1e-9 * (1.0 + caller[k].abs()),
            "θ[{k}]"
        );
    }
}

#[test]
fn scale_only_block_needs_no_factorization() {
    let g = one_block(&[1]);
    // No intercept in X: nothing centered, N diagonal.
    let p = 2;
    let x = [2.0, 4.0, 3.0, 5.0];
    let mut cs = ColumnScaling::identity(p);
    cs.compute(&x, 2, p, None, true);
    assert!(!g.block_has_shift(0, &cs));
    let internal = [0.7, 0.3, 0.2];
    let mut caller = [0.0; 3];
    let mut scratch = vec![0.0; g.re_map_scratch_len()];
    g.theta_kernel_to_caller(&internal, &mut caller, &mut scratch, &cs);
    let s1 = cs.scale(1);
    assert_eq!(caller, [0.7, 0.3 / s1, 0.2 / s1]);
}

/// A full-rank block maps to a full-rank caller factor however far a slope's
/// mean sits from zero against its scale. With `r = m/s`, centering makes
/// `Σ₀₀ = Σ̃₀₀ − 2r·Σ̃₀₁ + r²·Σ̃₁₁` while the slope's conditional variance is
/// `det Σ̃/Σ₀₀`: tiny against `tr Σ`, not against its own `Σ₁₁`. Cutting it to
/// an exact zero would report a false `pinned` bit and a zero slope θ. A pivot
/// floor of `q·eps·tr Σ` did that to Σ̃ = I from m/s ≈ 7000 and to internal
/// SDs 0.1 with ρ̃ = 0.9 from m/s ≈ 4500. Both stddevs keep a finite,
/// positive SE there.
#[test]
fn full_rank_block_stays_full_rank_at_large_shifts() {
    let l11 = (0.01f64 - 0.0081).sqrt();
    let cases = [
        ([1.0, 0.0, 1.0], 7.0e3),
        ([1.0, 0.0, 1.0], 1.0e5),
        ([0.1, 0.09, l11], 1.73e3),
        ([0.1, 0.09, l11], 1.0e4),
    ];
    let cov = faer::Mat::from_fn(3, 3, |a, b| if a == b { 0.01 } else { 0.002 });
    let g = one_block(&[1]);
    let mut scratch = vec![0.0; g.re_map_scratch_len()];
    let mut misses = vec![];
    for (internal, r) in cases {
        let cs = scaling(&[(r, 1.0)]);
        let mut caller = [0.0; 3];
        g.theta_kernel_to_caller(&internal, &mut caller, &mut scratch, &cs);
        let sig_t = vech_to_sigma(&internal, 2);
        let sig00 = sig_t[0] - 2.0 * r * sig_t[1] + r * r * sig_t[3];
        let want = ((sig_t[0] * sig_t[3] - sig_t[1] * sig_t[1]) / sig00).sqrt();
        let mut out = [0.0; 3];
        g.stddev_se_kernel_to_caller(&internal, cov.as_ref(), &mut out, &cs);
        // The pivot is a difference of numbers of size Σ₁₁, so it carries a
        // relative error of about eps·Σ₁₁/pivot: 2e-6 at m/s = 1e5 with Σ̃ = I.
        if (caller[2] - want).abs() > 1e-4 * want
            || g.pinned_mask(&caller, true) != 0
            || ![out[0], out[2]].iter().all(|v| v.is_finite() && *v > 0.0)
        {
            misses.push((r, want, caller, out));
        }
    }
    assert!(
        misses.is_empty(),
        "(m/s, slope diagonal wanted, θ, SE): {misses:?}"
    );
}

/// Row k's reported stddev `√Σ_kk`, read off the caller θ that
/// `theta_kernel_to_caller` maps θ̃ to (its rows are a factor of `Σ`).
fn caller_stddevs(g: &LmmGroupings, theta: &[f64], q: usize, cs: &ColumnScaling) -> Vec<f64> {
    let mut caller = vec![0.0; theta.len()];
    g.theta_kernel_to_caller(
        theta,
        &mut caller,
        &mut vec![0.0; g.re_map_scratch_len()],
        cs,
    );
    let sig = vech_to_sigma(&caller, q);
    (0..q).map(|k| sig[k * q + k].sqrt()).collect()
}

/// The stddev SEs equal `J·Cov·Jᵀ` with `J` the central differences of the
/// reported stddevs in θ̃, at each row's diagonal vech slot, NaN off it: on a
/// centered block and on one whose 0/1 slopes are only scaled.
#[test]
fn stddev_se_matches_finite_differences_of_the_reported_stddevs() {
    let g = one_block(&[1, 2]);
    let theta = [0.9, 0.2, -0.1, 0.5, 0.05, 0.3];
    let cov = faer::Mat::from_fn(6, 6, |a, b| {
        if a == b {
            0.01 * (1.0 + a as f64)
        } else {
            0.002
        }
    });
    let x = [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let mut cs01 = ColumnScaling::with_re_slopes(3, [1, 2]);
    cs01.compute(&x, 3, 3, None, true);
    assert!(!g.block_has_shift(0, &cs01));
    for cs in [scaling(&[(12.0, 3.0), (-2.0, 0.5)]), cs01] {
        let mut got = [0.0; 6];
        g.stddev_se_kernel_to_caller(&theta, cov.as_ref(), &mut got, &cs);
        let mut jac = [[0.0f64; 6]; 3];
        for a in 0..6 {
            let h = 1e-6;
            let (mut tp, mut tm) = (theta, theta);
            tp[a] += h;
            tm[a] -= h;
            let (fp, fm) = (
                caller_stddevs(&g, &tp, 3, &cs),
                caller_stddevs(&g, &tm, 3, &cs),
            );
            for k in 0..3 {
                jac[k][a] = (fp[k] - fm[k]) / (2.0 * h);
            }
        }
        // Diagonal vech slots of rows 0, 1, 2: (0,0) (1,1) (2,2).
        for (k, slot) in [(0, 0), (1, 3), (2, 5)] {
            let v: f64 = (0..6)
                .flat_map(|a| (0..6).map(move |c| (a, c)))
                .map(|(a, c)| jac[k][a] * cov[(a, c)] * jac[k][c])
                .sum();
            assert!(
                (got[slot] - v.sqrt()).abs() <= 1e-6 * v.sqrt(),
                "shift {}: sd[{k}]: {} vs {}",
                g.block_has_shift(0, &cs),
                got[slot],
                v.sqrt()
            );
        }
        assert!(
            [got[1], got[2], got[4]].iter().all(|v| v.is_nan()),
            "{got:?}"
        );
    }
}

/// A row that reads one θ̃ slot (a scalar block, the intercept of a block with
/// no centered slope) keeps that slot's SE `√Cov_aa`, bit for bit, at zero
/// too: `|θ̃_a|` has the slope 1 from either side. A row that reads several
/// slots has no derivative at a zero stddev and reports NaN.
#[test]
fn single_slot_rows_keep_the_theta_se() {
    let cov = faer::Mat::from_fn(3, 3, |a, b| {
        if a == b {
            0.01 * (1.0 + a as f64)
        } else {
            0.002
        }
    });
    let scalar = one_block(&[]);
    for t in [0.7, -0.7, 0.0] {
        let mut out = [0.0];
        scalar.stddev_se_kernel_to_caller(
            &[t],
            cov.as_ref(),
            &mut out,
            &ColumnScaling::identity(1),
        );
        assert_eq!(out[0].to_bits(), cov[(0, 0)].sqrt().to_bits(), "θ̃ = {t}");
    }
    let g = one_block(&[1]);
    let cs = ColumnScaling::identity(2);
    for (theta, slope_nan) in [
        ([0.9, 0.2, 0.5], false),
        ([0.0, 0.2, 0.5], false),
        ([0.9, 0.0, 0.0], true),
    ] {
        let mut out = [0.0; 3];
        g.stddev_se_kernel_to_caller(&theta, cov.as_ref(), &mut out, &cs);
        assert_eq!(out[0].to_bits(), cov[(0, 0)].sqrt().to_bits(), "{theta:?}");
        assert_eq!(out[2].is_nan(), slope_nan, "{theta:?}: {out:?}");
    }
}

/// A rank-deficient kernel block maps to a caller factor with exactly as many
/// zero diagonals and pinned bits, preserving Σ = N·Σ̃·Nᵀ, at small and large
/// shifts and a non-unit scale (the pivot floor alone does not absorb the
/// re-factorization's rounding: the rank cap does) and when the block is not
/// canonical (the vector-AGQ search keeps the entries below a pinned diagonal).
#[test]
fn pinned_block_keeps_exact_zeros_at_small_shifts() {
    let mut st = 11u64;
    let mut u = move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut misses = vec![];
    for q in [2usize, 3] {
        let slopes: Vec<usize> = (1..q).collect();
        for (m, s) in [
            (0.01, 1.0),
            (0.05, 1.0),
            (0.1, 1.0),
            (0.3, 1.0),
            (3.0, 1.0),
            (50.0, 2.0),
        ] {
            let ms = m / s;
            let g = one_block(&slopes);
            let cs = scaling(&vec![(m, s); q - 1]);
            let mut scratch = vec![0.0; g.re_map_scratch_len()];
            let len = q * (q + 1) / 2;
            for draw in 0..400 {
                let pin = draw % q;
                let canonical = (draw / q) % 2 == 0;
                let mut lam = vec![0.0; q * q];
                for r in 0..q {
                    for c in 0..=r {
                        lam[r * q + c] = u();
                    }
                    lam[r * q + r] = lam[r * q + r].abs() + 0.1;
                }
                lam[pin * q + pin] = 0.0;
                if canonical {
                    for r in pin..q {
                        lam[r * q + pin] = 0.0;
                    }
                }
                let mut theta = vec![0.0; len];
                pack_vech(&lam, q, &mut theta);
                let mut caller = vec![0.0; len];
                g.theta_kernel_to_caller(&theta, &mut caller, &mut scratch, &cs);
                let zeros = g
                    .diagonal_theta()
                    .iter()
                    .filter(|&&k| caller[k] == 0.0)
                    .count();
                if zeros != 1 || g.pinned_mask(&caller, true).count_ones() != 1 {
                    misses.push((q, ms, draw, canonical, caller.clone()));
                }
                // Σ is preserved: N·Σ̃·Nᵀ against the caller factor. Not to
                // round-off of Σ itself: a small real pivot (a conditional
                // variance near zero) amplifies the rounding of the entries
                // below it, measured up to 5.5e-12 relative here; a wrong
                // pivot dropped or kept would be O(1).
                let (sig_t, sig_c) = (vech_to_sigma(&theta, q), vech_to_sigma(&caller, q));
                let mut nm = vec![0.0; q * q];
                nm[0] = 1.0;
                for r in 1..q {
                    nm[r * q + r] = 1.0 / s;
                    nm[r] = -ms;
                }
                for r in 0..q {
                    for c in 0..q {
                        let want: f64 = (0..q)
                            .flat_map(|i| (0..q).map(move |j| (i, j)))
                            .map(|(i, j)| nm[r * q + i] * sig_t[i * q + j] * nm[c * q + j])
                            .sum();
                        assert!(
                            (sig_c[r * q + c] - want).abs() <= 1e-10 * (1.0 + want.abs()),
                            "q={q} m/s={ms} draw {draw}: Σ[{r}][{c}] {} vs {want}",
                            sig_c[r * q + c]
                        );
                    }
                }
            }
        }
    }
    assert!(
        misses.is_empty(),
        "{} draws without exactly one zero diagonal, first {:?}",
        misses.len(),
        misses.first()
    );
}

/// A NaN θ covariance (the `Rx` arm and the Hessian's RX fallback report no θ
/// block) maps to NaN SEs on every block, centered or not: never to a zero SE.
#[test]
fn nan_theta_cov_gives_nan_ses() {
    let cov = faer::Mat::from_fn(3, 3, |_, _| f64::NAN);
    for shift in [false, true] {
        let g = one_block(&[1]);
        let cs = if shift {
            scaling(&[(5.0, 2.0)])
        } else {
            ColumnScaling::identity(2)
        };
        let mut out = [0.0; 3];
        g.stddev_se_kernel_to_caller(&[0.9, 0.2, 0.5], cov.as_ref(), &mut out, &cs);
        assert!(out.iter().all(|v| v.is_nan()), "shift {shift}: {out:?}");
    }
}

/// A non-finite kernel θ (a failed draw) maps to a non-finite caller θ on a
/// centered block, as it does on every other block: never to a zero that a
/// warm-start carry would read as a pinned component.
#[test]
fn nan_kernel_theta_stays_nan_on_a_centered_block() {
    let g = one_block(&[1]);
    let cs = scaling(&[(5.0, 2.0)]);
    let mut scratch = vec![0.0; g.re_map_scratch_len()];
    let mut out = [0.0; 3];
    g.theta_kernel_to_caller(&[f64::NAN, 0.3, 0.4], &mut out, &mut scratch, &cs);
    assert!(out.iter().all(|v| v.is_nan()), "{out:?}");
}
