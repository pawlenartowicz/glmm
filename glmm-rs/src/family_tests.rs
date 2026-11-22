use super::*;
use crate::{BinomialLink, Family, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink};

/// The unit deviance keeps its digits where `y` is near `μ` at a large
/// scale: counts near 1e10, and the near-equal proportions or means a
/// huge prior weight or a tiny dispersion multiplies. References are
/// 50-digit evaluations (mpmath) of the same expressions at these exact
/// `f64` inputs. A log taken as a quotient or as a difference of two logs
/// misses each by 5e-6 to 7e-5 relative, which, times the row's scale,
/// sits above the PIRLS exit band; so does an NB form whose two terms are
/// each near `y·(y − μ)/μ` (4e-7 relative here). The NB arm of
/// `simd_transcendental::family_pass`, the one the `f64` fits run, must
/// agree with `dev_resid` at its own μ to the same band.
#[test]
fn unit_deviance_is_accurate_near_y_equal_mu() {
    let nb = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let cases = [
        (
            Family::Poisson {
                link: PoissonLink::Log,
            },
            10_000_070_000.0,
            1e10,
            f64::NAN,
            0.489_998_856_670_668_3,
            1e-9,
        ),
        (
            nb,
            11_000_000_000.0,
            1e10,
            50.0,
            0.468_982_017_294_786_7,
            1e-9,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            0.3 + 2f64.powi(-20),
            0.3,
            f64::NAN,
            4.330_924_528_896_284e-12,
            1e-9,
        ),
        (
            Family::Gamma {
                link: GammaLink::Log,
            },
            0.7 + 2f64.powi(-20),
            0.7,
            f64::NAN,
            1.856_109_950_438_723e-12,
            1e-9,
        ),
    ];
    for (family, y, mu, nb_theta, reference, band) in cases {
        let d: f64 = dev_resid(family, nb_theta, y, mu);
        let rel = (d - reference).abs() / reference;
        assert!(
            rel < band,
            "{family:?}: {d} vs 50-digit {reference} (rel {rel:.1e})"
        );
    }
    let (y, theta) = (vec![11_000_000_000.0; 9], 50.0);
    let mut eta = vec![1e10f64.ln(); 9];
    let (mut prob, mut w) = (vec![0.0; 9], vec![0.0; 9]);
    let (simd, _) = crate::simd_transcendental::family_pass(
        nb,
        theta,
        &mut eta,
        &y,
        &[],
        false,
        0.0,
        &mut prob,
        &mut w,
        &mut [],
    );
    let scalar: f64 = (0..9).map(|i| dev_resid(nb, theta, y[i], prob[i])).sum();
    let rel = (simd - scalar).abs() / scalar;
    assert!(
        rel < 1e-9,
        "NB family_pass {simd} vs dev_resid {scalar} (rel {rel:.1e})"
    );
}

/// Past cloglog's e^{−t} underflow (η > ln 745) and probit's Φ underflow
/// (|η| > 38) the tail weight is exactly 0; the working residual must still
/// be finite, or z = η + r poisons X'Wz.
#[test]
fn tail_working_residual_is_finite_where_the_weight_underflows() {
    let cloglog = Family::Binomial {
        link: BinomialLink::Cloglog,
    };
    let probit = Family::Binomial {
        link: BinomialLink::Probit,
    };
    for (family, eta) in [
        (cloglog, 7.0),
        (cloglog, 15.0),
        (probit, 40.0),
        (probit, -40.0),
    ] {
        for y in [0.0, 1.0] {
            let (_, _, r) = irls_weight_and_resid(family, f64::NAN, y, eta);
            assert!(r.is_finite(), "{family:?} η = {eta} y = {y}: r = {r}");
        }
    }
}

/// `observed_weight`'s `dr/dη` per link against a central difference of
/// `r(η) = (dμ/dη)/V(μ(η))`, read back through `w_obs = w − (y−μ)·dr/dη`
/// at `y − μ = 1` (so `dr/dη = w − w_obs`). Step 1e-5, band 1e-7: the
/// truncation error is O(h²) ≈ 1e-10 on these smooth links, the
/// cancellation error ≈ 1e-11, so the band is ~1000× the expected error.
#[test]
fn observed_weight_matches_fd_of_score_factor() {
    let fams = [
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        },
        // Constant-`r` links: the derivative is exactly 0.
        Family::Gamma {
            link: GammaLink::Inverse,
        },
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        },
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        Family::Poisson {
            link: PoissonLink::Log,
        },
    ];
    let nb_theta = 2.5;
    let r = |f: Family, eta: f64| mu_eta(f, eta) / variance(f, nb_theta, link_inv(f, eta));
    for f in fams {
        // Positive η only: Gamma-inverse and IG-inverse-squared need η > 0.
        for eta in [0.3_f64, 0.9, 1.7] {
            let h = 1e-5;
            let fd = (r(f, eta + h) - r(f, eta - h)) / (2.0 * h);
            let mu = link_inv(f, eta);
            let w = 1.0;
            let got = w - observed_weight(f, nb_theta, mu + 1.0, 1.0, eta, mu, w);
            assert!(
                (got - fd).abs() < 1e-7,
                "{f:?} eta={eta}: dr/deta {got} vs fd {fd}"
            );
        }
    }
}

/// The ten `(family, link)` cells `weight_eta_deriv` and its table cover,
/// paired with the η domain each needs (Gamma-inverse and IG-inverse-squared
/// need `η > 0`).
fn weight_eta_deriv_cells() -> Vec<(Family, &'static [f64])> {
    const GEN: &[f64] = &[-1.3, 0.2, 1.7];
    // 0.5 rather than 0.3: at h = 1e-5 the central difference's O(h²) truncation
    // term tracks the third derivative of w(η), which is steep enough near
    // η = 0.3 on the Gamma-inverse link (w = η⁻²) to exceed the 1e-7 band;
    // IG-inverse-squared (w = η^(-3/2)/4) is milder and rides along.
    const POS: &[f64] = &[0.5, 0.9, 1.7];
    vec![
        (Family::Gaussian, GEN),
        (
            Family::Poisson {
                link: PoissonLink::Log,
            },
            GEN,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            GEN,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            GEN,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Cloglog,
            },
            GEN,
        ),
        (
            Family::Gamma {
                link: GammaLink::Log,
            },
            GEN,
        ),
        (
            Family::Gamma {
                link: GammaLink::Inverse,
            },
            POS,
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            GEN,
        ),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
            GEN,
        ),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            },
            POS,
        ),
    ]
}

/// `weight_eta_deriv` against the `Dual<1>` derivative of the Fisher weight
/// `irls_weight_and_resid` returns. The two must agree to round-off: the
/// exact β-profile's pass A calls the closed form off the tail and the
/// `Dual<1>` line on it, and the border's curvature differentiates the
/// closed form again.
#[test]
fn weight_eta_deriv_matches_dual1_of_irls_weight() {
    use crate::dual::Dual;
    let nb_theta = 2.5;
    for (f, etas) in weight_eta_deriv_cells() {
        for &eta in etas {
            // The 2.5 row is implied by the 1.0 row while every arm is
            // `w·g(η, μ)`; it guards a future arm that is not.
            for &prior_w in &[1.0_f64, 2.5_f64] {
                let e = Dual::<1> { v: eta, d: [1.0] };
                let (mu_d, w_d, _) = irls_weight_and_resid(f, nb_theta, 1.0, e);
                let want = prior_w * w_d.d[0];
                let got = weight_eta_deriv(f, nb_theta, eta, mu_d.v, prior_w * w_d.v);
                assert!(
                    (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "{f:?} eta={eta} prior_w={prior_w}: got {got} want {want}"
                );
            }
        }
    }
}

/// `mu_eta_eta` against the `Dual<1>` derivative of `mu_eta` at the same η,
/// for every family/link, over a grid that includes each link's
/// [`clamp_eta`] bounds and ordinary interior points. Band fixed from the
/// first run: worst observed relative error 2.1e-16, round-off on both
/// sides.
#[test]
fn mu_eta_eta_matches_dual1_of_mu_eta() {
    use crate::dual::Dual;
    let cells: Vec<(Family, Vec<f64>)> = vec![
        (Family::Gaussian, vec![-5.0, 0.0, 3.0]),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            vec![-27.7, -3.0, 0.0, 1.7, 27.7],
        ),
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            vec![-7.03, -1.0, 0.0, 1.0, 7.03],
        ),
        (
            Family::Binomial {
                link: BinomialLink::Cloglog,
            },
            vec![-ETA_MAX, -27.6, 0.0, 3.32, ETA_MAX.ln()],
        ),
        (
            Family::Poisson {
                link: PoissonLink::Log,
            },
            vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
        ),
        (
            Family::Gamma {
                link: GammaLink::Log,
            },
            vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
        ),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
            vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
        ),
        (
            Family::Gamma {
                link: GammaLink::Inverse,
            },
            vec![MU_FLOOR, 0.5, 5.0, ETA_MAX],
        ),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            },
            vec![MU_FLOOR, 0.5, 5.0, ETA_MAX],
        ),
    ];
    for (f, etas) in cells {
        for eta in etas {
            let e = Dual::<1> { v: eta, d: [1.0] };
            let dual_lane = mu_eta(f, e).d[0];
            let closed = mu_eta_eta(f, eta);
            let band = 1e-12 * closed.abs().max(1.0);
            assert!(
                (dual_lane - closed).abs() <= band,
                "{f:?} eta={eta}: dual {dual_lane} vs closed {closed}"
            );
        }
    }
}

/// Every family with a tail ([`in_tail`]), at tail points on each
/// reachable side: the tail row must really be in the tail, its unit
/// deviance finite, and every per-row quantity the exact derivative the
/// kernels treat it as — held against the `Dual<1>` lane of the quantity
/// one order down, through the same entry points the kernels call.
/// `row_score = −½·∂dev/∂η`, `observed_weight = −∂row_score/∂η`,
/// `weight_eta_deriv = ∂w/∂η` of the `irls_weight_and_resid` weight, and
/// `observed_weight_eta_deriv = ∂observed_weight/∂η`. The stored μ each
/// quantity is handed is the one [`link_inv`] builds, so probit and
/// cloglog see their `PROB_EPS`-bounded μ, as in a fit. Band fixed from
/// the first run with an order of magnitude of headroom.
#[test]
fn tail_forms_match_dual1() {
    use crate::dual::Dual;
    let nb_theta = 2.5;
    let probit = Family::Binomial {
        link: BinomialLink::Probit,
    };
    let cloglog = Family::Binomial {
        link: BinomialLink::Cloglog,
    };
    let logit = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let nb = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let ig = Family::InverseGaussian {
        link: InverseGaussianLink::Log,
    };
    // (family, y values, tail η points)
    let cells: &[(Family, &[f64], &[f64])] = &[
        (
            probit,
            &[0.0, 0.3, 1.0],
            &[-7.1, -12.0, -40.0, -300.0, 3.8, 5.0, 6.0, 7.1, 12.0, 40.0],
        ),
        (
            cloglog,
            &[0.0, 0.3, 1.0],
            &[-27.7, -60.0, -600.0, 2.3, 3.4, 6.0, 30.0, 650.0],
        ),
        (
            logit,
            &[0.0, 0.3, 1.0],
            &[-27.7, -60.0, -600.0, 9.3, 27.7, 60.0, 600.0],
        ),
        (nb, &[0.0, 3.0], &[-231.0, -600.0, 231.0, 600.0]),
        (gamma, &[0.5, 3.0], &[-231.0, -600.0, 231.0, 600.0]),
        (ig, &[0.5, 3.0], &[-231.0, 231.0, 600.0]),
    ];
    let close = |got: f64, want: f64| (got - want).abs() <= 1e-11 * want.abs().max(1.0);
    for &(f, ys, etas) in cells {
        for &eta in etas {
            let mu = link_inv(f, eta);
            assert!(in_tail(f, eta, mu), "{f:?} eta={eta}: not a tail point");
            let e = Dual::<1> { v: eta, d: [1.0] };
            let mu_d = link_inv(f, e);
            for &y in ys {
                let dev = dev_resid_at(f, nb_theta, y, e, mu_d);
                assert!(dev.v.is_finite(), "{f:?} eta={eta} y={y}: dev {}", dev.v);
                let score = row_score(f, nb_theta, y, 1.0, e, mu_d);
                assert!(
                    close(score.v, -0.5 * dev.d[0]),
                    "{f:?} eta={eta} y={y}: score {} vs −½·dev' {}",
                    score.v,
                    -0.5 * dev.d[0]
                );
                let (_, w_d, _) = irls_weight_and_resid(f, nb_theta, y, e);
                let w_eta = weight_eta_deriv(f, nb_theta, eta, mu, w_d.v);
                assert!(
                    close(w_eta, w_d.d[0]),
                    "{f:?} eta={eta} y={y}: w' {w_eta} vs dual {}",
                    w_d.d[0]
                );
                let wo_d = observed_weight(f, nb_theta, y, 1.0, e, mu_d, w_d);
                assert!(
                    close(wo_d.v, -score.d[0]),
                    "{f:?} eta={eta} y={y}: w_obs {} vs −score' {}",
                    wo_d.v,
                    -score.d[0]
                );
                let wo_eta = observed_weight_eta_deriv(f, nb_theta, y, 1.0, eta, mu, w_eta);
                assert!(
                    close(wo_eta, wo_d.d[0]),
                    "{f:?} eta={eta} y={y}: w_obs' {wo_eta} vs dual {}",
                    wo_d.d[0]
                );
            }
        }
    }
    // Closed form on cloglog's upper side with y = 0: the deviance is
    // `2e^η`, so the observed weight is `e^η`. The μ form at the stored
    // `1 − PROB_EPS` would divide by `V` at the bound and go strongly
    // negative, making the Laplace `A` non-PD.
    let eta = 3.32;
    let mu = link_inv(cloglog, eta);
    assert!(in_tail(cloglog, eta, mu));
    let fisher = irls_weight_and_resid(cloglog, f64::NAN, 0.0, eta).1;
    let wo = observed_weight(cloglog, f64::NAN, 0.0, 1.0, eta, mu, fisher);
    let t = eta.exp();
    assert!(
        (wo - t).abs() <= 1e-12 * t,
        "cloglog tail y=0: observed weight {wo} vs e^η {t}"
    );
}

/// The deviance keeps growing into the tail: on each family with a μ bound
/// or a tail, walking η further out against the data (a y > 0 row toward
/// μ → 0, a y < 1 or y < μ row toward the upper side) strictly raises
/// `dev_resid_at`, finite at every step, with no constant stretch past
/// `MU_FLOOR` or `PROB_EPS`. And at each switch from the μ form to the
/// tail form the two agree to round-off (to the μ form's 1e-12 at the
/// binomial upper switch).
#[test]
fn tail_deviance_grows_and_meets_the_mu_form() {
    let nb_theta = 2.5;
    let rising = |f: Family, y: f64, etas: &[f64]| {
        let d: Vec<f64> = etas
            .iter()
            .map(|&e| {
                let e = clamp_eta(f, e);
                dev_resid_at(f, nb_theta, y, e, link_inv(f, e))
            })
            .collect();
        for k in 1..d.len() {
            assert!(
                d[k].is_finite() && d[k] > d[k - 1],
                "{f:?} y={y}: deviance {:?} at η {:?} is not rising",
                d,
                etas
            );
        }
    };
    for link in [
        BinomialLink::Logit,
        BinomialLink::Probit,
        BinomialLink::Cloglog,
    ] {
        let f = Family::Binomial { link };
        rising(f, 1.0, &[-5.0, -10.0, -30.0, -60.0, -100.0, -600.0]);
        rising(f, 0.0, &[3.0, 5.0, 10.0, 30.0, 100.0, 600.0]);
    }
    for f in [
        Family::Poisson {
            link: PoissonLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
    ] {
        rising(
            f,
            2.0,
            &[-10.0, -23.1, -30.0, -100.0, -229.0, -231.0, -500.0],
        );
    }
    // IG's deviance `(y/μ − 1)²/y` passes f64's range near η = −354, where
    // the true value does too.
    rising(
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        },
        2.0,
        &[-10.0, -23.1, -30.0, -100.0, -229.0, -231.0, -350.0],
    );
    // Upper side of the log links: NB and Gamma keep rising; IG's
    // deviance tends to its true bound 1/y as μ → ∞.
    for f in [
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
    ] {
        rising(f, 2.0, &[10.0, 100.0, 229.0, 231.0, 500.0, 699.0]);
    }
    // Continuity at the switch, on the side where μ is exact.
    let meet = |f: Family, y: f64, eta_in: f64, eta_out: f64| {
        let (mi, mo) = (link_inv(f, eta_in), link_inv(f, eta_out));
        assert!(!in_tail(f, eta_in, mi) && in_tail(f, eta_out, mo));
        let (di, dout) = (
            dev_resid_at(f, nb_theta, y, eta_in, mi),
            dev_resid_at(f, nb_theta, y, eta_out, mo),
        );
        // Both points sit 1e-9 apart in η; the deviance slope there is
        // O(|η|), so the gap is the slope's share plus round-off.
        assert!(
            (di - dout).abs() <= 1e-6 * di.abs().max(1.0),
            "{f:?} y={y}: μ form {di} vs tail form {dout} across the switch"
        );
    };
    let lo_probit = -7.034_483_825_117_6;
    meet(
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        1.0,
        lo_probit + 1e-9,
        lo_probit - 1e-9,
    );
    let lo_logit = (PROB_EPS / (1.0 - PROB_EPS)).ln();
    meet(
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        1.0,
        lo_logit + 1e-9,
        lo_logit - 1e-9,
    );
    for f in [
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        },
    ] {
        meet(f, 2.0, -LOG_TAIL_ETA + 1e-9, -LOG_TAIL_ETA - 1e-9);
        meet(f, 2.0, LOG_TAIL_ETA - 1e-9, LOG_TAIL_ETA + 1e-9);
    }
    // The binomial upper switch, `1 − μ = UPPER_TAIL_Q`, where the μ
    // form's `1 − μ` still carries 12 digits.
    for (link, eta) in [
        (BinomialLink::Logit, 9.210_240_366_975_85),
        (BinomialLink::Probit, 3.719_016_485_455_68),
        (BinomialLink::Cloglog, 2.220_326_806_367_846),
    ] {
        meet(Family::Binomial { link }, 0.0, eta - 1e-8, eta + 1e-8);
    }
}

/// A y = 0 row with `1 − μ` near 1e-10, inside the `(PROB_EPS,
/// UPPER_TAIL_Q)` band, keeps every digit of its deviance and score on
/// each binomial link: through `row_score` and `dev_resid_eta` (PIRLS and
/// AGQ) and through the family pass's deviance and `w·(z − η)` (GLM IRLS
/// and the PIRLS refresh). References are 50-digit values (mpmath) of
/// `−2·ln(1 − μ)` and `−(−ln(1 − μ))'`. A μ form taking `1 − μ` from the
/// stored μ misses the probit and cloglog scores here by about 1e-7
/// relative and the logit deviance by 5e-9.
#[test]
fn binomial_upper_rows_keep_their_digits() {
    let cases = [
        (
            BinomialLink::Logit,
            22.5,
            45.000_000_000_338_38,
            -0.999_999_999_830_810_2,
        ),
        (
            BinomialLink::Probit,
            6.3,
            45.256_529_260_900_02,
            -6.451_586_678_649_745,
        ),
        (
            BinomialLink::Cloglog,
            3.1,
            44.395_902_562_883_27,
            -22.197_951_281_441_64,
        ),
    ];
    let rel = |got: f64, want: f64| (got - want).abs() / want.abs();
    for (link, eta, dev_ref, score_ref) in cases {
        let f = Family::Binomial { link };
        let dev = dev_resid_eta(f, f64::NAN, 0.0, eta);
        let score = row_score(f, f64::NAN, 0.0, 1.0, eta, link_inv(f, eta));
        // Weighted, so logit takes the general arm rather than the fused
        // unweighted kernel, whose deviance never forms `1 − μ`.
        let (mut e, mut prob, mut w, mut z) =
            (vec![eta; 5], vec![0.0; 5], vec![0.0; 5], vec![0.0; 5]);
        let (pass_dev, _) = crate::simd_transcendental::family_pass(
            f,
            f64::NAN,
            &mut e,
            &[0.0; 5],
            &[1.0; 5],
            true,
            0.0,
            &mut prob,
            &mut w,
            &mut z,
        );
        for (what, got, want) in [
            ("dev_resid_eta", dev, dev_ref),
            ("row_score", score, score_ref),
            ("family_pass deviance", pass_dev / 5.0, dev_ref),
            ("family_pass w·(z − η)", w[0] * (z[0] - e[0]), score_ref),
        ] {
            assert!(
                rel(got, want) < 1e-13,
                "{link:?} η = {eta}: {what} {got} vs 50-digit {want} (rel {:.1e})",
                rel(got, want)
            );
        }
    }
}

#[test]
fn poisson_log_canonical_quantities() {
    let f = Family::Poisson {
        link: PoissonLink::Log,
    };
    let eta = 0.5_f64;
    let mu = link_inv(f, eta);
    assert!((mu - eta.exp()).abs() < 1e-12); // g⁻¹ = exp
    assert!((variance(f, f64::NAN, mu) - mu).abs() < 1e-12); // V(μ)=μ
                                                             // canonical: w = V(μ) = μ; working resid = (y−μ)/μ
    let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 3.0, eta);
    assert!((m - mu).abs() < 1e-12 && (w - mu).abs() < 1e-12);
    assert!((r - (3.0 - mu) / mu).abs() < 1e-12);
}

#[test]
fn gamma_log_noncanonical_weight() {
    let f = Family::Gamma {
        link: GammaLink::Log,
    };
    let eta = 0.2_f64;
    let mu = eta.exp();
    // log link on Gamma is non-canonical: dμ/dη=μ, V=μ² → w=μ²/μ²=1
    let (_m, w, r) = irls_weight_and_resid(f, f64::NAN, 1.0, eta);
    assert!((w - 1.0).abs() < 1e-12, "w={w}");
    assert!((r - (1.0 - mu) / mu).abs() < 1e-12); // (y−μ)·dη/dμ = (y−μ)/μ
}

#[test]
fn gamma_inverse_residual_sign() {
    let f = Family::Gamma {
        link: GammaLink::Inverse,
    };
    let eta = 0.5_f64; // η>0 required; μ=1/η=2
    let mu = 1.0 / eta;
    // dμ/dη=−μ², V=μ² → w=(μ²)²/μ²=μ²; resid=(y−μ)·dη/dμ=−(y−μ)/μ²
    let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 3.0, eta);
    assert!((m - mu).abs() < 1e-12 && (w - mu * mu).abs() < 1e-12);
    assert!((r - (-(3.0 - mu) / (mu * mu))).abs() < 1e-12, "r={r}");
}

#[test]
fn nb_variance_uses_theta() {
    let f = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    // V(μ) = μ + μ²/θ, with θ̂ threaded explicitly.
    let mu = 3.0;
    assert!((variance(f, 2.0, mu) - (mu + mu * mu / 2.0)).abs() < 1e-12);
}

/// The identity `Fit::loglik()` relies on: `−½·Σwᵢdᵢ + saturated_loglik`
/// must equal the exact `Σwᵢ·log f(yᵢ; μᵢ)` at ANY μ (not just the fitted
/// one), per family — checked against directly-written log-densities.
#[test]
fn saturated_loglik_restores_exact_densities() {
    let lg = crate::simd_transcendental::ln_gamma;
    let y = [0.0, 1.0, 3.0, 7.0];
    let mu = [0.5, 1.2, 2.5, 6.0];
    let w = [1.0, 2.0, 1.0, 3.0];

    let fp = Family::Poisson {
        link: PoissonLink::Log,
    };
    let dev: f64 = (0..4)
        .map(|i| w[i] * dev_resid(fp, f64::NAN, y[i], mu[i]))
        .sum();
    let direct: f64 = (0..4)
        .map(|i| w[i] * (y[i] * mu[i].ln() - mu[i] - lg(y[i] + 1.0)))
        .sum();
    let restored = -0.5 * dev + saturated_loglik(fp, f64::NAN, &y, Some(&w));
    assert!(
        (restored - direct).abs() < 1e-10,
        "poisson {restored} vs {direct}"
    );

    let th = 1.7;
    let fnb = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let dev: f64 = (0..4).map(|i| w[i] * dev_resid(fnb, th, y[i], mu[i])).sum();
    let direct: f64 = (0..4)
        .map(|i| {
            w[i] * (lg(y[i] + th) - lg(th) - lg(y[i] + 1.0)
                + th * (th / (th + mu[i])).ln()
                + y[i] * (mu[i] / (th + mu[i])).ln())
        })
        .sum();
    let restored = -0.5 * dev + saturated_loglik(fnb, th, &y, Some(&w));
    assert!(
        (restored - direct).abs() < 1e-10,
        "nb {restored} vs {direct}"
    );

    // Aggregated binomial: y is a success PROPORTION, w the trial count m.
    let fb = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let yb = [0.0, 0.5, 2.0 / 3.0, 1.0];
    let m = [2.0, 4.0, 3.0, 5.0];
    let mub = [0.3, 0.55, 0.6, 0.8];
    let dev: f64 = (0..4)
        .map(|i| m[i] * dev_resid(fb, f64::NAN, yb[i], mub[i]))
        .sum();
    let direct: f64 = (0..4)
        .map(|i| {
            let s = m[i] * yb[i];
            lg(m[i] + 1.0) - lg(s + 1.0) - lg(m[i] - s + 1.0)
                + s * mub[i].ln()
                + (m[i] - s) * (1.0 - mub[i]).ln()
        })
        .sum();
    let restored = -0.5 * dev + saturated_loglik(fb, f64::NAN, &yb, Some(&m));
    assert!(
        (restored - direct).abs() < 1e-10,
        "binomial {restored} vs {direct}"
    );
}

/// The Poisson row `y·ln y − y − lnΓ(y+1)` and the aggregated binomial row
/// `ln C(m,s) + s·ln y + (m−s)·ln(1−y)`, against 60-digit values (mpmath
/// `loggamma`, at the f64 inputs) out to counts near 1e15. Their terms are
/// of order `y·ln y` while the value is of order `ln y`, so formed term by
/// term they keep about 1e-5 absolute at `y = 1e10`; the band is 1e-12.
#[test]
fn saturated_loglik_is_accurate_at_any_count() {
    let fp = Family::Poisson {
        link: PoissonLink::Log,
    };
    let pois = [
        (1.0, -1.0),
        (2.5, -1.410_246_772_661_686_6),
        (7.0, -1.903_790_317_678_221_2),
        (1e3, -4.372_899_506_026_297),
        (1e7, -8.977_986_367_017_166),
        (1e10, -12.431_863_998_183_234),
        (1e15, -18.188_326_730_660_015),
    ];
    for (y, want) in pois {
        let got = saturated_loglik(fp, f64::NAN, &[y], None);
        assert!(
            (got - want).abs() <= 1e-12,
            "poisson y = {y:e}: {got} vs {want}"
        );
    }
    let fb = Family::Binomial {
        link: BinomialLink::Logit,
    };
    // (trials m, proportion y, 60-digit row value)
    let binom = [
        (2.0, 0.5, -std::f64::consts::LN_2),
        (3.0, 2.0 / 3.0, -0.810_930_216_216_328_8),
        (2.5, 0.4, -0.766_238_435_648_986),
        (1e3, 0.375, -3.647_671_953_727_958),
        (1e10, 0.25, -11.594_875_781_425_177),
        (2f64.powi(40), 2f64.powi(-40), -0.999_999_999_999_545_3),
        (1e15, 0.5, -17.495_179_550_100_07),
    ];
    for (m, y, want) in binom {
        let got = saturated_loglik(fb, f64::NAN, &[y], Some(&[m]));
        assert!(
            (got - want).abs() <= 1e-12,
            "binomial m = {m:e}, y = {y}: {got} vs {want}"
        );
    }
}

#[test]
fn probit_mu_eta_is_normal_pdf() {
    let f = Family::Binomial {
        link: BinomialLink::Probit,
    };
    // g⁻¹=Φ (via the high-precision `phi_hp`, ~1e-15), dμ/dη=φ(η) exact pdf.
    assert!((link_inv(f, 0.0) - 0.5).abs() < 1e-13);
    assert!((mu_eta(f, 0.0) - (1.0 / (2.0 * std::f64::consts::PI).sqrt())).abs() < 1e-12);
}

#[test]
fn cloglog_link_and_derivative() {
    let f = Family::Binomial {
        link: BinomialLink::Cloglog,
    };
    // μ = 1 − exp(−exp η); at η=0 that is 1 − e⁻¹.
    let mu0 = link_inv(f, 0.0);
    assert!((mu0 - (1.0 - (-1.0f64).exp())).abs() < 1e-15, "μ(0)={mu0}");
    // dμ/dη = exp(η − exp η); max at η=0, value e⁻¹.
    let d0 = mu_eta(f, 0.0);
    assert!((d0 - (-1.0f64).exp()).abs() < 1e-15, "dμ/dη(0)={d0}");
    // Asymmetric: μ approaches 1 much faster than 0.
    assert!(link_inv(f, 2.0) > 0.999);
    assert!(link_inv(f, -2.0) < 0.13);
    // General Fisher weight (dμ/dη)²/V(μ), V = μ(1−μ).
    let eta = 0.4_f64;
    let mu = link_inv(f, eta);
    let dm = mu_eta(f, eta);
    let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 1.0, eta);
    assert!((m - mu).abs() < 1e-15);
    assert!((w - dm * dm / (mu * (1.0 - mu))).abs() < 1e-12, "w={w}");
    assert!((r - (1.0 - mu) / dm).abs() < 1e-12, "r={r}");
}

#[test]
fn cloglog_eta_is_clamped_at_eta_max() {
    let f = Family::Binomial {
        link: BinomialLink::Cloglog,
    };
    assert_eq!(clamp_eta(f, 1e6), ETA_MAX);
    assert_eq!(clamp_eta_bounds(f), (-ETA_MAX, ETA_MAX));
    // At the bound `e^η` is finite, μ sits on its `PROB_EPS` bound, and
    // the tail deviance `2e^η` of a y = 0 row is finite.
    let mu = link_inv(f, 1e6);
    assert_eq!(mu, 1.0 - PROB_EPS);
    assert!(mu_eta(f, 1e6).is_finite());
    let d = dev_resid_at(f, f64::NAN, 0.0, ETA_MAX, mu);
    assert!(d.is_finite() && d > 1e300, "d={d}");
    // Logit and probit take no η clamp.
    for link in [BinomialLink::Logit, BinomialLink::Probit] {
        assert_eq!(clamp_eta(Family::Binomial { link }, 1e6), 1e6);
    }
}

#[test]
fn inverse_gaussian_inverse_squared_quantities() {
    let f = Family::InverseGaussian {
        link: InverseGaussianLink::InverseSquared,
    };
    let eta = 0.25_f64; // η>0 required; μ = η^(−1/2) = 2
    let mu = link_inv(f, eta);
    assert!((mu - 2.0).abs() < 1e-12, "μ={mu}");
    // V(μ) = μ³
    assert!((variance(f, f64::NAN, mu) - 8.0).abs() < 1e-12);
    // dμ/dη = −½·η^(−3/2) = −μ³/2
    let dm = mu_eta(f, eta);
    assert!((dm - (-4.0)).abs() < 1e-12, "dμ/dη={dm}");
    // General branch: w = (dμ/dη)²/V = μ³/4; resid = (y−μ)/(dμ/dη)
    let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 3.0, eta);
    assert!((m - mu).abs() < 1e-12 && (w - 2.0).abs() < 1e-12, "w={w}");
    assert!((r - (3.0 - 2.0) / -4.0).abs() < 1e-12, "r={r}");
    // η ≤ 0 is outside the OPEN domain, like Gamma-inverse.
    assert!(eta_infeasible(f, 1.0, 0.0));
    assert!(eta_infeasible(f, 1.0, -1.0));
    assert!(!eta_infeasible(f, 1.0, 1e-3));
}

#[test]
fn inverse_gaussian_log_quantities() {
    let f = Family::InverseGaussian {
        link: InverseGaussianLink::Log,
    };
    let eta = 0.7_f64;
    let mu = eta.exp();
    assert!((link_inv(f, eta) - mu).abs() < 1e-12);
    // dμ/dη = μ, V = μ³ → w = μ²/μ³ = 1/μ; resid = (y−μ)/μ
    let (_m, w, r) = irls_weight_and_resid(f, f64::NAN, 4.0, eta);
    assert!((w - 1.0 / mu).abs() < 1e-12, "w={w}");
    assert!((r - (4.0 - mu) / mu).abs() < 1e-12, "r={r}");
    // The log link evaluates η on ±ETA_MAX; past it the row's deviance
    // would stop moving, so a trial there is refused.
    assert!(!eta_infeasible(f, 1.0, -50.0));
    assert!(eta_infeasible(f, 1.0, -ETA_MAX - 1.0));
    assert!(eta_infeasible(f, 0.0, ETA_MAX + 1.0));
    assert!(!eta_infeasible(f, 1.0, f64::NAN));
    // A y = 0 row past the lower bound is at its deviance's own limit.
    assert!(!eta_infeasible(f, 0.0, -ETA_MAX - 1.0));
}

#[test]
fn inverse_gaussian_saturated_loglik_is_nan() {
    // Like Gamma: the objective substitutes `inv_gaussian_aic`, which already
    // carries the profiled dispersion, so there is no saturated constant to
    // restore. A caller reaching this arm is a bug, surfaced as NaN.
    let f = Family::InverseGaussian {
        link: InverseGaussianLink::Log,
    };
    assert!(saturated_loglik(f, f64::NAN, &[1.0, 2.0], None).is_nan());
}

/// Links and interior η points for the observed-weight derivative checks
/// (positive η for the inverse links, whose domain is η > 0). Covers every
/// link, so a link wrongly classified by `exact_curvature_differs` in
/// either direction is caught: wrongly true reaches the `unreachable!`,
/// wrongly false returns the Fisher `dw/dη` and misses the `Dual<1>` lane.
fn observed_weight_cells() -> Vec<(Family, Vec<f64>)> {
    vec![
        (Family::Gaussian, vec![-1.0, 0.3]),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            },
            vec![0.4, 1.5],
        ),
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            vec![-1.3, 0.0, 0.4, 2.1],
        ),
        (
            Family::Binomial {
                link: BinomialLink::Cloglog,
            },
            vec![-2.0, -0.3, 0.5, 1.2],
        ),
        (
            Family::Gamma {
                link: GammaLink::Log,
            },
            vec![-1.0, 0.2, 2.3],
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            vec![-1.0, 0.2, 2.3],
        ),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
            vec![-0.5, 0.2, 1.1],
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            vec![-1.0, 0.3],
        ),
        (
            Family::Poisson {
                link: PoissonLink::Log,
            },
            vec![-1.0, 0.3],
        ),
        (
            Family::Gamma {
                link: GammaLink::Inverse,
            },
            vec![0.4, 1.5],
        ),
    ]
}

/// `observed_weight_eta_deriv` (`dW_obs/dη`, the third η-derivative of the
/// row's log-likelihood) against the `Dual<1>` lane of `observed_weight`
/// itself, with the Fisher weight and μ carried as duals of the same η.
/// Band 1e-11 relative: both sides are closed forms at `f64`.
#[test]
fn observed_weight_eta_deriv_matches_dual1_of_observed_weight() {
    use crate::dual::Dual;
    let nb_theta = 2.5;
    for (f, etas) in observed_weight_cells() {
        for &eta in &etas {
            for &(y, prior_w) in &[(0.3_f64, 1.0_f64), (1.0, 2.5), (4.0, 0.7)] {
                // Binomial rows need y in [0, 1].
                let y = if matches!(f, Family::Binomial { .. }) {
                    y.min(1.0) * 0.8
                } else {
                    y
                };
                let e = Dual::<1> { v: eta, d: [1.0] };
                let (mu_d, w_raw, _) = irls_weight_and_resid(f, nb_theta, y, e);
                let w_d = Dual::<1>::from_f64(prior_w) * w_raw;
                let want = observed_weight(f, nb_theta, y, prior_w, e, mu_d, w_d).d[0];
                let w_eta = weight_eta_deriv(f, nb_theta, eta, mu_d.v, w_d.v);
                let got = observed_weight_eta_deriv(f, nb_theta, y, prior_w, eta, mu_d.v, w_eta);
                assert!(
                    (got - want).abs() <= 1e-11 * want.abs().max(1.0),
                    "{f:?} eta={eta} y={y} pw={prior_w}: got {got} want {want}"
                );
            }
        }
    }
}

/// `gamma_dispersion_term(ln φ) + D/φ` is the whole `−2·Σᵢ log f(yᵢ; shape
/// aᵢ, mean μᵢ)`, `aᵢ = wᵢ/φ`, normalising terms included, at every φ:
/// PIRLS on the prior weights `wᵢ/φ` returns `D/φ`, and this term is the
/// rest. The reference is the Gamma log-density written out term by term,
/// `aᵢ·ln aᵢ − aᵢ·ln μᵢ + (aᵢ−1)·ln yᵢ − aᵢ·yᵢ/μᵢ − lnΓ(aᵢ)`, precision
/// weights (row `i` has variance `φ·V(μᵢ)/wᵢ`).
#[test]
fn gamma_dispersion_term_completes_the_gamma_log_density() {
    let y = [0.7_f64, 1.4, 2.2, 3.1, 0.2];
    let mu = [0.9_f64, 1.2, 2.6, 2.8, 0.5];
    let w = [1.0_f64, 2.0, 0.5, 1.5, 1.0];
    let n = y.len();
    let family = Family::Gamma {
        link: GammaLink::Log,
    };
    let sum_ln_y: f64 = y.iter().map(|v| v.ln()).sum();
    let dev: f64 = (0..n)
        .map(|i| w[i] * dev_resid(family, f64::NAN, y[i], mu[i]))
        .sum();
    for &phi in &[0.05_f64, 0.3, 1.0, 2.5] {
        let want = -2.0
            * (0..n)
                .map(|i| {
                    let a = w[i] / phi;
                    let lg = crate::simd_transcendental::ln_gamma(a);
                    a * a.ln() - a * mu[i].ln() + (a - 1.0) * y[i].ln() - a * y[i] / mu[i] - lg
                })
                .sum::<f64>();
        let got = gamma_dispersion_term(phi.ln(), Some(&w), n, sum_ln_y) + dev / phi;
        assert!(
            (got - want).abs() <= 1e-12 * want.abs().max(1.0),
            "φ = {phi}: {got} vs {want}"
        );
    }
    // Unweighted (`weights = None`) must equal every row sharing `w = 1`.
    let ones = [1.0_f64; 5];
    for &phi in &[0.05_f64, 0.3, 1.0, 2.5] {
        let via_none = gamma_dispersion_term(phi.ln(), None, n, sum_ln_y);
        let via_unit_weights = gamma_dispersion_term(phi.ln(), Some(&ones), n, sum_ln_y);
        assert!(
            (via_none - via_unit_weights).abs() <= 1e-12 * via_unit_weights.abs().max(1.0),
            "φ = {phi}: None {via_none} vs unit weights {via_unit_weights}"
        );
    }
}

/// `gamma_dispersion_term` per row, `2a − 2a·ln a + 2·lnΓ(a)` at `a = 1/φ`,
/// against 60-digit values (mpmath `loggamma`) from the switch to Stirling's
/// series at a = 20 out to the a ≈ 1e16 of a near-exact fit. Band 1e-14
/// relative; the direct form misses it from about a = 50 (1e-10 off at 1e6,
/// no correct digit at 1e16).
#[test]
fn gamma_dispersion_term_row_is_accurate_at_small_phi() {
    let refs = [
        (20.0_f64, -1.149_522_567_760_651_7),
        (50.0, -2.070_812_650_124_833_8),
        (1e3, -5.069_711_545_911_68),
        (1e6, -11.977_633_324_888_262),
        (1e10, -21.187_973_863_514_445),
        (1e16, -35.003_484_421_495_385),
        (1e30, -67.239_675_723_412_02),
    ];
    for (a, want) in refs {
        let got = gamma_dispersion_term(-a.ln(), None, 1, 0.0);
        assert!(
            (got - want).abs() <= 1e-14 * want.abs(),
            "a = {a:e}: {got} vs {want}"
        );
    }
}

/// The closed-form first and second `ln φ` derivatives of
/// `gamma_dispersion_term` against central differences of the term itself,
/// unweighted and with row-varying weights. Bands: 1e-6 and 1e-4 relative,
/// the O(h²) truncation of a step of 1e-4.
#[test]
fn gamma_dispersion_term_derivatives_match_central_differences() {
    let n = 4;
    let sum_ln_y = 1.3_f64;
    let w = [0.5_f64, 1.0, 2.5, 4.0];
    let h = 1e-4;
    for weights in [None, Some(&w[..])] {
        let g = |t: f64| gamma_dispersion_term(t, weights, n, sum_ln_y);
        for &psi in &[-3.0_f64, -1.0, 0.0, 0.9] {
            let d1 = (g(psi + h) - g(psi - h)) / (2.0 * h);
            let d2 = (g(psi + h) - 2.0 * g(psi) + g(psi - h)) / (h * h);
            let got1 = gamma_dispersion_term_d1(psi, weights, n);
            let got2 = gamma_dispersion_term_d2(psi, weights, n);
            assert!(
                (got1 - d1).abs() <= 1e-6 * d1.abs().max(1.0),
                "weighted={} ψ = {psi}: d1 {got1} vs {d1}",
                weights.is_some()
            );
            assert!(
                (got2 - d2).abs() <= 1e-4 * d2.abs().max(1.0),
                "weighted={} ψ = {psi}: d2 {got2} vs {d2}",
                weights.is_some()
            );
        }
    }
}

/// `gamma_ml_dispersion` solves the Gamma ML shape equation
/// `ln a − ψ(a) = D/(2n)`, `a = 1/φ`, from the near-Poisson end (large a)
/// to the heavy-tailed one (small a), and a zero deviance gives φ = 0.
#[test]
fn gamma_ml_dispersion_solves_the_shape_equation() {
    let n = 288;
    for &c in &[1e-8_f64, 1e-4, 0.05, 0.4418, 1.0, 5.0, 40.0] {
        let phi = gamma_ml_dispersion(2.0 * c * n as f64, None, n);
        let a = 1.0 / phi;
        // The residual is itself computed with the cancelling difference,
        // so its own rounding, a few ulp of ln a, is allowed on top.
        let resid = a.ln() - crate::dual::digamma(a) - c;
        let tol = 1e-12 * c + 16.0 * f64::EPSILON * a.ln().abs().max(1.0);
        assert!(
            phi > 0.0 && resid.abs() <= tol,
            "c = {c}: φ = {phi}, residual {resid}"
        );
        // Near the Poisson end the root is φ ≈ 2c (a ≈ 1/(2c)).
        if c < 1e-3 {
            assert!(
                (phi / (2.0 * c) - 1.0).abs() < 2.0 * c,
                "c = {c}: φ = {phi}"
            );
        }
    }
    assert_eq!(gamma_ml_dispersion(0.0, None, n), 0.0);
    // A deviance that rounds below zero on data the mean model reproduces
    // exactly is still an exact fit, not a NaN.
    assert_eq!(gamma_ml_dispersion(-1.4e-15, None, n), 0.0);
    assert!(gamma_ml_dispersion(f64::NAN, None, n).is_nan());
    // MASS::gamma.shape on sim_gamma's log-link GLM: dev 254.45332674125353
    // over 288 rows gives 1/alpha = 0.78574357084515345.
    let phi = gamma_ml_dispersion(254.45332674125353, None, 288);
    assert!((phi - 0.7857435708451534).abs() < 1e-14, "φ = {phi}");
}

/// The weighted arm solves the per-row equation `Σᵢ wᵢ·(ln aᵢ − ψ₀(aᵢ)) =
/// D/2`, `aᵢ = wᵢ/φ`: check the residual of that equation directly at the
/// returned root, on weights that vary row to row (not all-equal, so the
/// unweighted closed form does not apply), and that `Σŵ = n` unit weights
/// reproduce the unweighted root.
#[test]
fn gamma_ml_dispersion_weighted_solves_the_per_row_equation() {
    let w = [0.4_f64, 0.8, 1.3, 2.5, 3.0];
    let n = w.len();
    for &dev in &[0.5_f64, 3.0, 12.0, 40.0] {
        let phi = gamma_ml_dispersion(dev, Some(&w), n);
        assert!(phi > 0.0 && phi.is_finite(), "dev = {dev}: φ = {phi}");
        let resid: f64 = w
            .iter()
            .map(|&wi| {
                let ai = wi / phi;
                wi * (ai.ln() - crate::dual::digamma(ai))
            })
            .sum::<f64>()
            - dev / 2.0;
        assert!(
            resid.abs() <= 1e-9 * dev.max(1.0),
            "dev = {dev}: φ = {phi}, residual {resid}"
        );
    }
    let ones = [1.0_f64; 5];
    let via_ones = gamma_ml_dispersion(12.0, Some(&ones), 5);
    let via_none = gamma_ml_dispersion(12.0, None, 5);
    assert!(
        (via_ones - via_none).abs() <= 1e-12 * via_none.abs().max(1.0),
        "unit weights {via_ones} vs None {via_none}"
    );
}

/// Unweighted matches R's `inverse.gaussian()$aic` formula exactly; the
/// weighted half is the precision closed form (row `i` has variance
/// `φ·V(μᵢ)/wᵢ`, so `φ̂ = D/n`, `aic = n·(ln(2πφ̂)+1) + 3Σln yᵢ − Σln wᵢ +
/// 2`), a different formula from R's `Σwᵢ`/`Σwᵢln yᵢ` case-weight one.
#[test]
fn inv_gaussian_aic_matches_precision_formula() {
    // R: aic = n*(log(dev/n*2*pi)+1) + 3*sum(log(y)) + 2
    let y = [1.5_f64, 2.0, 0.75, 3.25];
    let n = y.len();
    let dev = 0.42_f64;
    let got = inv_gaussian_aic(&y, dev, n, None, None);
    let disp = dev / n as f64;
    let want = n as f64 * ((2.0 * std::f64::consts::PI * disp).ln() + 1.0)
        + 3.0 * y.iter().map(|v| v.ln()).sum::<f64>()
        + 2.0;
    assert!((got - want).abs() < 1e-12, "got {got} want {want}");
    // Precision weights: φ̂ = D/n (n, not Σw), and an extra −Σln wᵢ term
    // (`−2ℓ = D/φ + n·ln(2πφ) + 3·Σln yᵢ − Σln wᵢ`).
    let w = [1.0_f64, 2.0, 0.5, 1.5];
    let gotw = inv_gaussian_aic(&y, dev, n, Some(&w), None);
    let wantw = n as f64 * ((2.0 * std::f64::consts::PI * disp).ln() + 1.0)
        + 3.0 * y.iter().map(|v| v.ln()).sum::<f64>()
        - w.iter().map(|wi| wi.ln()).sum::<f64>()
        + 2.0;
    assert!((gotw - wantw).abs() < 1e-12, "got {gotw} want {wantw}");
    // Unit weights must reproduce the unweighted value exactly (ln 1 = 0).
    let ones = [1.0_f64; 4];
    let via_ones = inv_gaussian_aic(&y, dev, n, Some(&ones), None);
    assert!((via_ones - got).abs() < 1e-12);
}

/// A held φ (`held_phi = Some(v)`) does NOT collapse `D/φ` to `n` — that
/// identity only holds at the profiled `φ̂ = D/n`. At an arbitrary `v` the
/// formula keeps `D/v` literal: `−2ℓ = D/v + n·ln(2πv) + 3Σln yᵢ − Σln wᵢ`.
/// Cross-checked against R's closed form (`sum(statmod::dinvgauss(y, mu,
/// dispersion = v))` reduces to the same expression at fixed `v`, since
/// `dinvgauss`'s `1/dispersion` scaling is this family's `1/φ`).
#[test]
fn inv_gaussian_aic_matches_precision_formula_at_held_phi() {
    let y = [1.5_f64, 2.0, 0.75, 3.25];
    let n = y.len();
    let dev = 0.42_f64;
    let v = 0.3_f64; // deliberately NOT dev/n (0.105), so D/φ != n.
    let got = inv_gaussian_aic(&y, dev, n, None, Some(v));
    let want = dev / v
        + n as f64 * (2.0 * std::f64::consts::PI * v).ln()
        + 3.0 * y.iter().map(|v| v.ln()).sum::<f64>()
        + 2.0;
    assert!((got - want).abs() < 1e-12, "got {got} want {want}");
    // Precision weights carry the same −Σln wᵢ term as the profiled case.
    let w = [1.0_f64, 2.0, 0.5, 1.5];
    let gotw = inv_gaussian_aic(&y, dev, n, Some(&w), Some(v));
    let wantw = want - w.iter().map(|wi| wi.ln()).sum::<f64>();
    assert!((gotw - wantw).abs() < 1e-12, "got {gotw} want {wantw}");
    // A held φ EQUAL to the profiled value must reproduce the profiled AIC.
    let disp = dev / n as f64;
    let via_held = inv_gaussian_aic(&y, dev, n, None, Some(disp));
    let via_profiled = inv_gaussian_aic(&y, dev, n, None, None);
    assert!(
        (via_held - via_profiled).abs() < 1e-12,
        "held-at-profiled {via_held} vs profiled {via_profiled}"
    );
}

#[cfg(test)]
mod probit_rows_tests {
    use super::*;

    const PROBIT_F: Family = Family::Binomial {
        link: BinomialLink::Probit,
    };

    /// η over `[−9, 9]` (both tails and the whole off-tail range), the μ the
    /// family pass stores for it, and its Fisher weight.
    fn rows() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut eta = Vec::new();
        let mut e = -9.0;
        while e <= 9.0 {
            eta.push(e);
            e += 0.000_731;
        }
        let mu: Vec<f64> = eta.iter().map(|&e| link_inv(PROBIT_F, e)).collect();
        let w: Vec<f64> = eta
            .iter()
            .map(|&e| irls_weight_and_resid(PROBIT_F, f64::NAN, 0.0, e).1)
            .collect();
        (eta, mu, w)
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-300)
    }

    /// `ProbitRows`' score, observed weight and `dW_obs/dη` against
    /// `row_score`, `observed_weight` and `observed_weight_eta_deriv` (libm
    /// `exp` in `mu_eta`) on every row of the sweep, for y on both sides and
    /// in between and a non-unit prior weight. The bounds are 4x the maxima
    /// measured 2026-10-07 on x86 (score 4.9e-16, observed weight 1.3e-14, its
    /// η-derivative 3.4e-13): an `exp` that differs in its last bit moves each by
    /// a few ULP of its own terms. `dW_obs/dη` crosses 0, so it
    /// is bounded relative to `max(1, |value|)`. Tail rows go to the generic
    /// functions and must be equal.
    #[test]
    fn probit_rows_match_the_generic_row_functions() {
        let (eta, mu, w) = rows();
        let (mut s_rel, mut wo_rel, mut we_abs) = (0.0f64, 0.0f64, 0.0f64);
        let mut tails = 0;
        for y in [0.0, 1.0, 0.3] {
            for pw in [1.0, 3.7] {
                let mut r = ProbitRows::new(PROBIT_F, &eta, &mu).unwrap();
                for i in 0..eta.len() {
                    let (e, m) = (eta[i], mu[i]);
                    let sa = r.score(i, y, pw);
                    let sb = row_score(PROBIT_F, f64::NAN, y, pw, e, m);
                    let wa = r.observed_weight(i, y, pw, pw * w[i]);
                    let wb = observed_weight(PROBIT_F, f64::NAN, y, pw, e, m, pw * w[i]);
                    let ea = r.observed_weight_eta_deriv(i, y, pw);
                    let eb = observed_weight_eta_deriv(PROBIT_F, f64::NAN, y, pw, e, m, 0.0);
                    if in_tail(PROBIT_F, e, m) {
                        tails += 1;
                        assert_eq!(sa.to_bits(), sb.to_bits(), "tail score at η = {e}");
                        assert_eq!(wa.to_bits(), wb.to_bits(), "tail w_obs at η = {e}");
                        assert_eq!(ea.to_bits(), eb.to_bits(), "tail w_obs' at η = {e}");
                        continue;
                    }
                    if sb != 0.0 {
                        s_rel = s_rel.max(rel(sa, sb));
                    }
                    wo_rel = wo_rel.max(rel(wa, wb));
                    we_abs = we_abs.max((ea - eb).abs() / eb.abs().max(1.0));
                }
            }
        }
        eprintln!("ProbitRows vs generic: score rel {s_rel:.3e}, w_obs rel {wo_rel:.3e}, w_obs' {we_abs:.3e}, tail rows {tails}");
        assert!(tails > 0);
        assert!(s_rel <= S_REL_BOUND, "score rel {s_rel:e}");
        assert!(wo_rel <= WO_REL_BOUND, "w_obs rel {wo_rel:e}");
        assert!(we_abs <= WE_BOUND, "w_obs' {we_abs:e}");
    }

    const S_REL_BOUND: f64 = 2e-15;
    const WO_REL_BOUND: f64 = 5e-14;
    const WE_BOUND: f64 = 1.5e-12;

    /// A row's value does not depend on how the rows are walked: the tile is
    /// refilled whenever a row leaves it, and `exp_fill`'s scalar tail is
    /// bit-identical to its SIMD head, so a reversed walk (a refill per row
    /// at a different tile offset) gives the same bits as the forward one.
    #[test]
    fn probit_rows_are_independent_of_walk_order() {
        let (eta, mu, w) = rows();
        let n = eta.len();
        let mut fwd = ProbitRows::new(PROBIT_F, &eta, &mu).unwrap();
        let mut rev = ProbitRows::new(PROBIT_F, &eta, &mu).unwrap();
        let a: Vec<[f64; 3]> = (0..n)
            .map(|i| {
                [
                    fwd.score(i, 0.3, 2.0),
                    fwd.observed_weight(i, 0.3, 2.0, w[i]),
                    fwd.observed_weight_eta_deriv(i, 0.3, 2.0),
                ]
            })
            .collect();
        for i in (0..n).rev() {
            let b = [
                rev.score(i, 0.3, 2.0),
                rev.observed_weight(i, 0.3, 2.0, w[i]),
                rev.observed_weight_eta_deriv(i, 0.3, 2.0),
            ];
            for c in 0..3 {
                assert_eq!(a[i][c].to_bits(), b[c].to_bits(), "row {i} quantity {c}");
            }
        }
    }
}
