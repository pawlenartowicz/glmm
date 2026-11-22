use super::*;
use crate::family::{dev_resid_eta, link_inv};
use crate::spec::{NegBinomialLink, PoissonLink};

/// The module docs' row term `a·expm1(s·δ) + h·δ` (`LOG1P`: `a·ln(1 + p·expm1(s·δ)) + h·δ`),
/// the reference `row_consts` is checked against. The shipped tiles inline their
/// own per-tile form; their values are checked against the generic node loop by
/// `glmm::tests::agq_fast_route_value_matches_the_generic_dual_route`.
fn term<const LOG1P: bool>(a: f64, p: f64, s: f64, h: f64, d: f64) -> f64 {
    let e = (s * d).exp_m1();
    if LOG1P {
        a * (p * e).ln_1p() + h * d
    } else {
        a * e + h * d
    }
}

/// The row term against the generic `½w·(dev_resid_eta(η₀+δ) − dev_resid_eta(η₀))`
/// over a grid of y, η₀ and δ: zero counts, binomial proportions, the logit
/// tail (`in_tail` past μ = 1 − 1e-4), and log-link rows up to the fast
/// guard's edge (Poisson's clamp, the NB/Gamma-log tail switch). Within
/// 1e-12 of the larger deviance, floored at 1. This checks `row_consts`
/// (production) and the identity; the tiles themselves are checked at the
/// value level by `glmm::tests::agq_fast_route_value_matches_the_generic_dual_route`.
#[test]
fn row_term_matches_dev_resid_eta() {
    let w = 1.7;
    let deltas: [f64; 11] = [
        -6.0, -2.5, -1.0, -1e-3, -1e-9, 0.0, 1e-7, 0.01, 0.7, 3.0, 6.0,
    ];
    let cases: Vec<(Family, f64, Vec<f64>, Vec<f64>)> = vec![
        (
            Family::Poisson {
                link: PoissonLink::Log,
            },
            0.0,
            vec![0.0, 1.0, 2.0, 7.0, 40.0, 1e4],
            vec![-30.0, -5.0, -0.5, 0.0, 0.3, 2.0, 9.0, 690.0],
        ),
        (
            Family::Gamma {
                link: GammaLink::Log,
            },
            0.0,
            vec![1e-3, 0.4, 1.0, 3.0, 250.0],
            vec![-220.0, -20.0, -1.0, 0.0, 0.8, 5.0, 220.0],
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            0.0,
            vec![0.0, 0.2, 0.5, 1.0],
            vec![-40.0, -9.5, -2.0, -0.1, 0.0, 0.1, 2.0, 9.5, 40.0],
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            0.7,
            vec![0.0, 1.0, 3.0, 50.0, 1e4],
            vec![-30.0, -3.0, -0.2, 0.0, 0.5, 4.0, 12.0, 220.0],
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            25.0,
            vec![0.0, 2.0, 30.0],
            vec![-5.0, 0.0, 3.2, 3.3, 8.0],
        ),
    ];
    let mut worst = 0.0f64;
    for (family, theta, ys, etas) in cases {
        let k = kind(family).unwrap();
        for &y in &ys {
            for &e0 in &etas {
                let mu0 = link_inv(family, e0);
                let [a, p, s, h] = row_consts(family, theta, y, w, e0, mu0);
                for &d in &deltas {
                    if !is_fast(family, e0, d.abs()) {
                        continue;
                    }
                    let fast = match k {
                        Kind::Lin => term::<false>(a, p, s, h, d),
                        Kind::Log1p => term::<true>(a, p, s, h, d),
                    };
                    let d1 = dev_resid_eta(family, theta, y, e0 + d);
                    let d0 = dev_resid_eta(family, theta, y, e0);
                    let slow = 0.5 * w * (d1 - d0);
                    let scale = (0.5 * w * d1.abs().max(d0.abs())).max(1.0);
                    let err = (fast - slow).abs() / scale;
                    worst = worst.max(err);
                    assert!(
                        err <= 1e-12,
                        "{family:?} θ={theta} y={y} η₀={e0} δ={d}: fast {fast} generic {slow} (rel {err:e})"
                    );
                }
            }
        }
    }
    eprintln!("row_term_matches_dev_resid_eta: worst scaled error {worst:e}");
}

/// The guard sends a row whose node range reaches the clamp or the
/// log-link tail to the generic path, and keeps logit rows whatever η₀.
#[test]
fn guard_stops_short_of_clamp_and_tail() {
    let pois = Family::Poisson {
        link: PoissonLink::Log,
    };
    let gam = Family::Gamma {
        link: GammaLink::Log,
    };
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    assert!(is_fast(pois, 690.0, 9.0));
    assert!(!is_fast(pois, 690.0, 10.0));
    assert!(!is_fast(pois, -695.0, 5.0));
    assert!(is_fast(gam, 220.0, 9.0));
    assert!(!is_fast(gam, -225.0, 5.0));
    assert!(is_fast(logit, 500.0, 50.0));
    assert!(!is_fast(logit, 0.0, 700.0));
    assert!(!is_fast(pois, f64::NAN, 1.0));
}

/// The magnitude guard: a Poisson row at η₀ = 205 (a count near e^205,
/// whose parts `μ₀·expm1(δ)` and `y·δ` cancel to O(1) from O(√μ₀)) is a
/// slow row; the same row at η₀ = 2 stays fast; a NaN center is slow.
#[test]
fn magnitude_guard_sends_huge_counts_to_the_generic_path() {
    let pois = Family::Poisson {
        link: PoissonLink::Log,
    };
    let big = 205f64.exp();
    let b = 3.0 / big.sqrt();
    assert!(fast_consts(pois, 0.0, big, 1.0, 205.0, big, b, mag_factor(Kind::Lin, b)).is_none());
    let small = 2f64.exp();
    assert!(fast_consts(pois, 0.0, 7.0, 1.0, 2.0, small, 1.0, 1f64.exp_m1()).is_some());
    assert!(fast_consts(pois, 0.0, 7.0, 1.0, f64::NAN, small, 1.0, 1f64.exp_m1()).is_none());
}

/// `probit_sums` against the generic node loop (`dev_resid_eta` per row
/// and node, minus the center `dev_resid_at`, with the exponent in its first
/// form `z_j² + ℓ_c(u_cj) − ℓ_c(ũ_c)` and `A_c − 1 = a_c² − 1`, in the node term
/// `w_j·expm1(x_cj)` of `agq::agq_deviance`) on clusters of 1, 2, 3 and 5
/// rows, Bernoulli and aggregated binomial (weights up to 1e4), at
/// θ = 0, near 0 and large, with slow rows past `LN_PHI_X_MAX` in the mix;
/// and the cluster-outer arm against the node-outer arm, to the bit.
#[test]
fn probit_sums_match_the_generic_node_loop() {
    let family = Family::Binomial {
        link: BinomialLink::Probit,
    };
    let sizes = [2usize, 3, 5, 1, 2, 3, 5, 5, 3, 2, 140];
    let s = sizes.len();
    let mut cluster_ids = Vec::new();
    for (c, &m) in sizes.iter().enumerate() {
        cluster_ids.extend(std::iter::repeat_n(c as u32, m));
    }
    let n = cluster_ids.len();
    // A deterministic spread of values (no RNG dependency in the crate).
    let frac = |i: usize, a: f64| ((i as f64 * a).sin() * 43758.5453).fract().abs();
    let k = 7usize;
    let blk = crate::consts::GH_OFFSETS[(k - 1) / 2];
    let nodes = &crate::consts::GH_NODES[blk..blk + k];
    let wts = &crate::consts::GH_WEIGHTS[blk..blk + k];
    let idx = ClusterRowIndex::build(&cluster_ids, s);
    let mut worst = 0.0f64;
    for aggregated in [false, true] {
        for &lambda in &[0.0, 1e-6, 0.8, 2.5] {
            let mut y = vec![0.0; n];
            let mut w = vec![1.0; n];
            let mut ef = vec![0.0; n];
            for (i, e) in ef.iter_mut().enumerate() {
                *e = -3.0 + 6.0 * frac(i, 12.9898);
            }
            // Slow rows: centers whose node range passes 37, on both sides.
            ef[3] = 36.9;
            ef[7] = -36.95;
            ef[n - 1] = 40.0;
            for i in 0..n {
                if aggregated {
                    // Proportions near each row's own fitted value, as at
                    // a mode, so the node sums stay finite at 1e4 trials.
                    let trials: f64 = [1.0, 3.0, 20.0, 1e4][i % 4];
                    w[i] = trials;
                    let c = cluster_ids[i] as usize;
                    let e0 = ef[i] + lambda * (-1.5 + 3.0 * frac(c + 7, 3.17));
                    let p0 = link_inv(family, e0);
                    let jitter = 0.3 * (frac(i, 78.233) - 0.5);
                    y[i] = ((p0 + jitter * (p0 * (1.0 - p0) / trials).sqrt()).clamp(0.0, 1.0)
                        * trials)
                        .round()
                        / trials;
                } else {
                    y[i] = if frac(i, 78.233) < 0.4 { 1.0 } else { 0.0 };
                }
            }
            let u: Vec<f64> = (0..s).map(|c| -1.5 + 3.0 * frac(c + 7, 3.17)).collect();
            let a_blocks: Vec<f64> = (0..s).map(|c| 0.6 + 2.0 * frac(c + 11, 5.3)).collect();
            let a_excess: Vec<f64> = a_blocks.iter().map(|&a| a * a - 1.0).collect();
            let eta: Vec<f64> = (0..n)
                .map(|i| ef[i] + lambda * u[cluster_ids[i] as usize])
                .collect();
            let prob: Vec<f64> = eta.iter().map(|&e| link_inv(family, e)).collect();
            // Generic reference, as agq_deviance's node loop computes it.
            let mut want = vec![0.0; s];
            let mut mag = vec![0.0f64; s];
            for c in 0..s {
                let mut ctr = 0.0;
                for i in 0..n {
                    if cluster_ids[i] as usize == c {
                        ctr -= 0.5
                            * w[i]
                            * crate::family::dev_resid_at(family, 0.0, y[i], eta[i], prob[i]);
                    }
                }
                for (&zj, &wj) in nodes.iter().zip(wts) {
                    let h = SQRT_2 * (1.0 / a_blocks[c]) * zj;
                    let u_cj = u[c] + h;
                    let mut a = 0.0;
                    for i in 0..n {
                        if cluster_ids[i] as usize == c {
                            a -= 0.5
                                * w[i]
                                * dev_resid_eta(family, 0.0, y[i], ef[i] + lambda * u_cj);
                        }
                    }
                    // The magnitude of the two loglik values the generic loop
                    // subtracts, priors included.
                    mag[c] =
                        mag[c].max((a - 0.5 * u_cj * u_cj).abs() + (ctr - 0.5 * u[c] * u[c]).abs());
                    // The exponent in its first form, z² + ℓ_c(u_cj) − ℓ_c(ũ_c),
                    // so the code's `−h·(ũ − ½·(A − 1)·h)` is tested against it.
                    let x = zj * zj + (a - 0.5 * u_cj * u_cj) - (ctr - 0.5 * u[c] * u[c]);
                    want[c] += wj * x.exp_m1();
                }
            }
            let run = |rows: Option<&ClusterRowIndex>| {
                let (mut rb, mut sl) = (Vec::new(), Vec::new());
                let (mut dl, mut acc, mut sum) = (vec![0.0; s], vec![0.0; s], vec![0.0; s]);
                probit_sums(
                    &y,
                    &w,
                    &cluster_ids,
                    n,
                    &eta,
                    &prob,
                    &ef,
                    &u,
                    &a_blocks,
                    &a_excess,
                    lambda,
                    nodes,
                    wts,
                    &mut rb,
                    &mut sl,
                    &mut dl,
                    &mut acc,
                    &mut sum,
                    rows,
                );
                (sum, sl)
            };
            let (got_c, slow) = run(Some(&idx));
            let (got_n, _) = run(None);
            assert!(slow.contains(&(n as u32 - 1)));
            if lambda >= 0.8 {
                assert!(slow.contains(&3) && slow.contains(&7));
            }
            for c in 0..s {
                assert_eq!(
                    got_c[c].to_bits(),
                    got_n[c].to_bits(),
                    "arms differ: cluster {c}, λ={lambda}, aggregated={aggregated}"
                );
                // Compared on the bracket `ln_1p(sum / Σ w_j)` that enters the
                // deviance. The generic reference subtracts the center from
                // the node loglik, so it carries eps times their magnitude.
                assert!(want[c].is_finite(), "reference overflowed: cluster {c}");
                let wsum: f64 = wts.iter().sum();
                let rel =
                    ((got_c[c] / wsum).ln_1p() - (want[c] / wsum).ln_1p()).abs() / (1.0 + mag[c]);
                worst = worst.max(rel);
                assert!(
                    rel <= 1e-13,
                    "cluster {c} (size {}), λ={lambda}, aggregated={aggregated}: fast {} generic {} (rel {rel:e})",
                    sizes[c],
                    got_c[c],
                    want[c]
                );
            }
        }
    }
    eprintln!("probit_sums_match_the_generic_node_loop: worst error per unit magnitude {worst:e}");
}

/// The probit guard: a center at 36 with a node reach of 0.9 is fast,
/// with 1.1 it is slow, both sides; a NaN center is slow.
#[test]
fn probit_guard_stops_inside_the_certified_range() {
    assert!(probit_is_fast(36.0, 0.9));
    assert!(!probit_is_fast(36.0, 1.1));
    assert!(probit_is_fast(-36.0, 0.9));
    assert!(!probit_is_fast(-36.0, 1.1));
    assert!(!probit_is_fast(f64::NAN, 0.1));
}
