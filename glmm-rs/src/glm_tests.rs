use super::*;
use crate::test_support::TestWs;
use faer::Mat;

/// Build a `GlmScratch` borrowing every IRLS field of `ws`. Used by tests
/// to avoid duplicating the inline struct literal at each call site.
fn glm_scratch(ws: &mut TestWs) -> GlmScratch<'_> {
    GlmScratch {
        irls_eta: &mut ws.irls_eta,
        irls_p: &mut ws.irls_p,
        irls_w: &mut ws.irls_w,
        irls_z: &mut ws.irls_z,
        irls_betas: &mut ws.irls_betas,
        irls_betas_new: &mut ws.irls_betas_new,
        irls_var_diag: &mut ws.irls_var_diag,
        irls_t_sq: &mut ws.irls_t_sq,
        irls_u_scratch: &mut ws.irls_u_scratch,
        irls_xtwx: ws.irls_xtwx.as_mut(),
        irls_xtwz: &mut ws.irls_xtwz,
        irls_l: ws.irls_l.as_mut(),
        irls_wx: &mut ws.irls_wx,
        irls_broken: &mut ws.irls_broken,
    }
}

/// All-0 and all-1 Bernoulli y short-circuit (n_iter 0, not converged)
/// with z², deviance and null deviance NaN.
#[test]
fn glm_z_sq_and_deviances_nan_on_non_converged() {
    let n = 100;
    let p = 2;
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = (i as f64) / (n as f64) - 0.5;
    }
    for y_val in [0.0f64, 1.0] {
        let y = vec![y_val; n];
        let mut ws = TestWs::new(n, p, 0);
        let targets: Vec<u32> = vec![0, 1];
        let fit = glm_irls_fit(
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            f64::NAN,
            x.as_ref(),
            &y,
            &targets,
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(!fit.converged, "y={y_val}");
        assert_eq!(fit.n_iter, 0, "y={y_val}");
        for &t in fit.t_sq.iter() {
            assert!(
                t.is_nan(),
                "y={y_val}: z² must be NaN on non-converged fit, got {t}"
            );
        }
        assert!(
            fit.deviance.is_nan(),
            "y={y_val}: deviance must be NaN on non-converged"
        );
        assert!(
            fit.deviance_null.is_nan(),
            "y={y_val}: deviance_null must be NaN on the all-0/all-1 short circuit"
        );
    }
}

#[test]
fn glm_separation_marks_non_converged() {
    // Build a fully-separated dataset: y = (x > 0). Logistic regression
    // diverges (β → ∞, so η → ∞); the divergence cap or the separation stop should fire.
    let n = 200;
    let p = 2;
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        x[(i, 0)] = 1.0;
        let xi = (i as f64 - n as f64 / 2.0) / 10.0; // wide span
        x[(i, 1)] = xi;
        y[i] = if xi > 0.0 { 1.0 } else { 0.0 };
    }
    let mut ws = TestWs::new(n, p, 0);
    let targets: Vec<u32> = vec![1];
    let fit = glm_irls_fit(
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        f64::NAN,
        x.as_ref(),
        &y,
        &targets,
        None,
        None,
        None,
        glm_scratch(&mut ws),
    );
    assert!(
        !fit.converged,
        "fully separated data must report non-converged"
    );
    assert!(fit.separated, "separation must be reported as such");
    for (k, (&t, &v)) in fit.t_sq.iter().zip(fit.var_diag.iter()).enumerate() {
        assert!(
            t.is_nan() && v.is_nan(),
            "target {k}: t_sq {t}, var_diag {v}"
        );
    }
    assert!(fit.deviance.is_nan(), "deviance {}", fit.deviance);
    assert!(
        fit.deviance_null.is_nan(),
        "deviance_null {}",
        fit.deviance_null
    );
    // β is left at the last iterate, not NaN-filled.
    assert!(fit.betas.iter().all(|b| b.is_finite()), "{:?}", fit.betas);
}

/// Separation that crosses the bound: `[1, x, x + d]` with `d = ±1` on a
/// block of ten rows whose outcomes separate along `d` (`y = 1` where
/// `d = 1`), pushed to the edge by an offset of −12, and `d = 0` elsewhere.
/// As the block's weights vanish `X'WX` turns singular, the solve on the
/// QR factor throws β₁ and β₂ out to ∓3e34, and η then absorbs every later
/// step: the step reads 0 and the deviance repeats, so both stops fire on
/// a fit that is running off. Their round-off bound is far above
/// `SEPARATION_STEP`, so the step cannot show a move and η decides, as on
/// a fit that runs out its budget.
#[test]
fn glm_separation_past_the_round_off_bound_is_not_converged() {
    let (n, m) = (60, 10);
    let mut st = 1u64;
    let mut x = Mat::<f64>::zeros(n, 3);
    let (mut y, mut offset) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let x1 = 2.0 * crate::fit::common_tests::lcg(&mut st);
        let u = 0.5 * (crate::fit::common_tests::lcg(&mut st) + 1.0);
        let d = match (i < m, i % 2) {
            (true, 0) => 1.0,
            (true, _) => -1.0,
            _ => 0.0,
        };
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1;
        x[(i, 2)] = x1 + d;
        y[i] = if i < m {
            f64::from(u8::from(d > 0.0))
        } else {
            f64::from(u8::from(u < 1.0 / (1.0 + (-0.3 * x1).exp())))
        };
        if i < m {
            offset[i] = -12.0;
        }
    }
    let mut ws = TestWs::new(n, 3, 0);
    let fit = glm_irls_fit(
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        f64::NAN,
        x.as_ref(),
        &y,
        &[0, 1, 2],
        None,
        None,
        Some(&offset),
        glm_scratch(&mut ws),
    );
    assert!(
        !fit.converged && fit.separated,
        "converged {}, separated {}, β {:?}",
        fit.converged,
        fit.separated,
        fit.betas
    );
}

/// Quasi-separation on the cloglog upper tail: one category whose rows are
/// all 1. There 1 − μ = exp(−e^η) vanishes so fast that the deviance rule
/// fires at η ≈ 3.3 while the category's η still moves about e^{−η} ≈ 0.04
/// per step, well short of the divergence guard. The fit has no finite
/// maximum and must say so (`SEPARATION_STEP`).
#[test]
fn glm_cloglog_upper_tail_separation_marks_non_converged() {
    let (m, p) = (20usize, 3usize);
    let n = 3 * m;
    let mut x = Mat::<f64>::zeros(n, p);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let g = i / m;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = f64::from(u8::from(g == 1));
        x[(i, 2)] = f64::from(u8::from(g == 2));
        y[i] = match g {
            0 => f64::from(u8::from(i % 3 == 0)),
            1 => f64::from(u8::from(i % 3 != 0)),
            _ => 1.0,
        };
    }
    let mut ws = TestWs::new(n, p, 0);
    let fit = glm_irls_fit(
        crate::Family::Binomial {
            link: crate::BinomialLink::Cloglog,
        },
        f64::NAN,
        x.as_ref(),
        &y,
        &[],
        None,
        None,
        None,
        glm_scratch(&mut ws),
    );
    assert!(
        !fit.converged && fit.separated,
        "converged {}, separated {}",
        fit.converged,
        fit.separated
    );
    assert!(fit.betas.iter().all(|b| b.is_finite()), "{:?}", fit.betas);
}

/// A cloglog fit started where a y = 0 row sits at η ≈ 20.2 (a leverage row
/// at x = 10, start β = (0.2, 2, 0)). There the Fisher weight underflows
/// while the score is −e^η, so the working response overflows and the
/// Fisher step is not finite; the Newton step's weight on that row is e^η.
/// A finite maximum exists. Reference: R 4.5 `optim(c(0, 0, 0), D,
/// gradient, method = "BFGS", reltol = 1e-16)`, restarted at its own
/// optimum until it stops moving, on D(β) = −2Σ[y·ln(−expm1(−t)) −
/// (1 − y)·t], t = exp(Xβ): D = 40.8525618518509, gradient norm 1e-6.
#[test]
fn glm_cloglog_start_with_a_y0_row_deep_in_the_upper_tail_converges() {
    const REF_BETA: [f64; 3] = [0.063854992759243, -0.0109381919145387, 0.187123718391584];
    const REF_DEV: f64 = 40.8525618518509;
    #[rustfmt::skip]
    let y_head: [f64; 30] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ];
    // 30 rows on x ∈ [−2, 2], then a y = 0 row at x = 10 and a y = 1 row at
    // x = −10.
    let n = 32;
    let mut x = Mat::<f64>::zeros(n, 3);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let (xi, x2, yi) = match i {
            30 => (10.0, 0.0, 0.0),
            31 => (-10.0, 0.0, 1.0),
            _ => (
                -2.0 + 4.0 * i as f64 / 29.0,
                (1.7 * i as f64).cos(),
                y_head[i],
            ),
        };
        x[(i, 0)] = 1.0;
        x[(i, 1)] = xi;
        x[(i, 2)] = x2;
        y[i] = yi;
    }
    let mut ws = TestWs::new(n, 3, 0);
    let fit = glm_irls_fit(
        crate::Family::Binomial {
            link: crate::BinomialLink::Cloglog,
        },
        f64::NAN,
        x.as_ref(),
        &y,
        &[],
        Some(&[0.2, 2.0, 0.0]),
        None,
        None,
        glm_scratch(&mut ws),
    );
    assert!(fit.converged, "must converge; β = {:?}", fit.betas);
    assert!(
        (fit.deviance - REF_DEV).abs() / REF_DEV < 1e-12,
        "deviance {} vs optim {REF_DEV}",
        fit.deviance
    );
    for (j, &r) in REF_BETA.iter().enumerate() {
        let rel = (fit.betas[j] - r).abs() / r.abs();
        assert!(rel < 1e-6, "β[{j}] = {} vs optim {r}", fit.betas[j]);
    }
}

/// `beta_start` seeds β and η directly instead of the family's cold-start
/// rule, so warm-starting at an already-converged optimum should settle
/// in far fewer IRLS iterations than a cold start and land on the same β.
#[test]
fn glm_irls_fit_warm_start_reaches_same_optimum_in_fewer_iterations() {
    let n = 200;
    let p = 2;
    let targets: Vec<u32> = vec![0, 1];
    let (x, y) = scaled_logit_design(n, 1.0);

    let mut ws_cold = TestWs::new(n, p, 0);
    let cold = glm_irls_fit(
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        f64::NAN,
        x.as_ref(),
        &y,
        &targets,
        None,
        None,
        None,
        glm_scratch(&mut ws_cold),
    );
    assert!(cold.converged, "cold fit must converge");
    let cold_betas = cold.betas.to_vec();
    let cold_iter = cold.n_iter;

    let mut ws_warm = TestWs::new(n, p, 0);
    let warm = glm_irls_fit(
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        f64::NAN,
        x.as_ref(),
        &y,
        &targets,
        Some(&cold_betas),
        None,
        None,
        glm_scratch(&mut ws_warm),
    );
    assert!(warm.converged, "warm fit must converge");
    assert!(
        warm.n_iter < cold_iter,
        "warm-starting at the converged optimum must take fewer IRLS \
         iterations than the cold start: warm={} cold={cold_iter}",
        warm.n_iter
    );
    for (w, c) in warm.betas.iter().zip(cold_betas.iter()) {
        assert!(
            (w - c).abs() < 1e-9,
            "warm start must land on the same optimum as the cold fit: {w} vs {c}"
        );
    }
}

/// Well-conditioned logistic design whose slope column is multiplied by
/// `scale`. The response is generated from the UNSCALED column, so the three
/// scalings are literally the same model in three unit systems. The LCG is
/// local to this builder — the fit path itself stays RNG-free.
fn scaled_logit_design(n: usize, scale: f64) -> (Mat<f64>, Vec<f64>) {
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    let mut s: u64 = 12345;
    for i in 0..n {
        let xu = ((i as f64) / (n as f64) - 0.5) * 4.0;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = xu * scale;
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (s >> 11) as f64 / ((1u64 << 53) as f64);
        let p = 1.0 / (1.0 + (0.3 - 2.0 * xu).exp());
        y[i] = f64::from(u8::from(u < p));
    }
    (x, y)
}

/// Same idea for a Poisson log-link design: μ = exp(0.5 + 0.8·x), counts
/// spread deterministically around it.
fn scaled_poisson_design(n: usize, scale: f64) -> (Mat<f64>, Vec<f64>) {
    let mut x = Mat::<f64>::zeros(n, 2);
    let mut y = vec![0.0f64; n];
    for i in 0..n {
        let xu = ((i as f64) / (n as f64) - 0.5) * 4.0;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = xu * scale;
        let mu = (0.5 + 0.8 * xu).exp();
        let jitter = 0.7 + 0.2 * ((i % 4) as f64);
        y[i] = (mu * jitter).round();
    }
    (x, y)
}

/// η = Xβ does not move when a predictor column is rescaled — the
/// compensating change in β̂ is exact — so a guard that bounds η gives the
/// same accept/reject decision in every unit system. A guard bounding
/// `|β_j| > 30` instead would reject the x/1000 fit with NaN variances while
/// accepting the identical x and x·1000 fits — scale-dependent, unlike the
/// η-based guard here. On the Poisson row such a cap returns
/// `converged: false` with β̂ = (0.4915, 803.09) and NaN standard errors.
#[test]
fn glm_guard_is_scale_invariant() {
    let n = 200;
    let p = 2;
    let targets: Vec<u32> = vec![0, 1];
    type Design = fn(usize, f64) -> (Mat<f64>, Vec<f64>);
    let cases: [(&str, crate::Family, Design, &[f64]); 2] = [
        (
            "logit",
            crate::Family::Binomial {
                link: crate::BinomialLink::Logit,
            },
            scaled_logit_design,
            &[1e-3, 1e3],
        ),
        (
            "poisson",
            crate::Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            scaled_poisson_design,
            &[1e-3],
        ),
    ];
    for (name, family, design, scales) in cases {
        let fit_at = |scale: f64| {
            let (x, y) = design(n, scale);
            let mut ws = TestWs::new(n, p, 0);
            let f = glm_irls_fit(
                family,
                f64::NAN,
                x.as_ref(),
                &y,
                &targets,
                None,
                None,
                None,
                glm_scratch(&mut ws),
            );
            (
                f.converged,
                f.betas.to_vec(),
                f.deviance,
                f.var_diag.to_vec(),
            )
        };
        let (c1, b1, d1, v1) = fit_at(1.0);
        assert!(c1, "{name}: x must converge");
        assert!(v1.iter().all(|q| q.is_finite()), "{name}: {v1:?}");
        for &s in scales {
            let (c, b, d, v) = fit_at(s);
            assert!(
                c,
                "{name}: the same model in every unit system must converge, not at x*{s}"
            );
            // Tolerance floor is ~1e-12 relative, not 0: `xu * s` is a
            // rounded double, so the designs are not literally the same
            // numbers even though they are the same model.
            assert!(
                (b[0] - b1[0]).abs() <= 1e-9 * b1[0].abs().max(1.0),
                "{name} x*{s}: intercept must not depend on the slope column's units: {} vs {}",
                b[0],
                b1[0]
            );
            let rescaled = b[1] * s;
            assert!(
                (rescaled - b1[1]).abs() <= 1e-9 * b1[1].abs().max(1.0),
                "{name} x*{s}: slope must scale exactly with the unit change: {rescaled} vs {}",
                b1[1]
            );
            assert!(
                (d - d1).abs() <= 1e-9 * d1.abs(),
                "{name} x*{s}: deviance must not depend on units: {d} vs {d1}"
            );
            assert!(
                v.iter().all(|q| q.is_finite()),
                "{name} x*{s}: variances must be finite in every unit system: {v:?}"
            );
            // Column scaling D = diag(1, s) turns X'WX into D(X'WX)D, so
            // the covariance (X'WX)⁻¹ picks up D⁻¹ on both sides: the
            // intercept variance is exactly unchanged and the slope
            // variance is divided by s². A `|β_j| > 30`-style guard that
            // let a scale-dependent variance through would fail this, not
            // just the finiteness check above.
            assert!(
                (v[0] - v1[0]).abs() <= 1e-9 * v1[0].abs(),
                "{name} x*{s}: intercept variance must not depend on the slope column's units: {} vs {}",
                v[0],
                v1[0]
            );
            let rescaled = v[1] * s * s;
            assert!(
                (rescaled - v1[1]).abs() <= 1e-9 * v1[1].abs(),
                "{name} x*{s}: slope variance must scale by s²: {rescaled} vs {}",
                v1[1]
            );
        }
    }
}

/// A GLM whose `X'WX` factor fails solves its step on `X·U⁻¹`, `U` from the
/// QR of `√W·X`, and reaches the maximum. On `entangled_dyadic_designs` at
/// `δ = 2⁻²⁷` (pivot ratio about 5e-17) the Gram is singular to rounding,
/// and a solve from it ends every arm not converged with β₂ off by 16% to
/// 100%. The `[1, a, b]` twin spans the same space, so the entangled
/// coefficient is the twin's divided by `δ`, its variance by `δ²`. Arms:
/// logit; probit, whose Newton step keeps the raw Gram; Poisson with an
/// offset and prior weights, whose right-hand side must stay `z − o` and
/// then `z − η`. Logit and Poisson also exercise the round-off stop on the
/// size of the terms: η = Xβ sums terms near 1e8 (β₂ ≈ 3e6, |a| ≤ 50) that
/// cancel to order 1, so η carries about 1e-8 of rounding and the deviance
/// moves by more than the relative rule allows from one pass to the next.
/// Measured against the twin (relative): logit deviance 1.5e-10, β₂ 1.0e-9,
/// `var_diag₂` 1.0e-9; probit 5.6e-10, 9.3e-6 (6e-5 of its SE, the stop
/// rule's slack on a linearly converging link) and 2.1e-6; Poisson 6.4e-11,
/// 1.4e-9 and 5.9e-7.
#[test]
fn glm_steps_through_the_design_qr_when_the_gram_factor_fails() {
    let n = 400;
    let (xv, xrv, delta) = crate::fit::common_tests::entangled_dyadic_designs(n, 27);
    let col_major = |v: &[f64]| Mat::<f64>::from_fn(n, 3, |i, j| v[3 * i + j]);
    let (x, xr) = (col_major(&xv), col_major(&xrv));
    let offset: Vec<f64> = (0..n).map(|i| 0.1 * ((i % 7) as f64 - 3.0)).collect();
    let w: Vec<f64> = (0..n).map(|i| 0.5 + 0.25 * (i % 5) as f64).collect();
    // (family, and the relative tolerances on β₂, the deviance excess and var₂).
    let arms = [
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            1e-8,
            1e-9,
            1e-8,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            1e-4,
            1e-8,
            1e-5,
        ),
        (
            Family::Poisson {
                link: crate::PoissonLink::Log,
            },
            1e-8,
            1e-9,
            1e-5,
        ),
    ];
    let targets = [0u32, 1, 2];
    for (family, beta_tol, dev_tol, var_tol) in arms {
        let poisson = matches!(family, Family::Poisson { .. });
        let (off, pw) = if poisson {
            (Some(&offset[..]), Some(&w[..]))
        } else {
            (None, None)
        };
        let mut st = 7u64;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let eta = 0.2 + 0.02 * xrv[3 * i + 1] + 0.02 * xrv[3 * i + 2];
                let u = 0.5 * (crate::fit::common_tests::lcg(&mut st) + 1.0);
                if poisson {
                    // Inverse-CDF Poisson draw.
                    let mu = (eta + offset[i]).exp();
                    let (mut k, mut pk) = (0.0, (-mu).exp());
                    let mut cdf = pk;
                    while u > cdf {
                        k += 1.0;
                        pk *= mu / k;
                        cdf += pk;
                    }
                    k
                } else {
                    f64::from(u8::from(u < 1.0 / (1.0 + (-eta).exp())))
                }
            })
            .collect();
        let fit = |x: MatRef<'_, f64>| {
            let mut ws = TestWs::new(n, 3, 0);
            let v = glm_irls_fit(
                family,
                f64::NAN,
                x,
                &y,
                &targets,
                None,
                pw,
                off,
                glm_scratch(&mut ws),
            );
            (v.converged, v.pivot, v.deviance, v.betas[2], v.var_diag[2])
        };
        let (conv, pivot, dev, beta, var) = fit(x.as_ref());
        let (conv_r, _, dev_r, beta_r, var_r) = fit(xr.as_ref());
        assert!(conv_r, "{family:?}: the twin converges");
        let want_beta = beta_r / delta;
        assert!(
            (beta - want_beta).abs() <= beta_tol * want_beta.abs(),
            "{family:?}: β₂ {beta} against {want_beta}"
        );
        assert!(conv, "{family:?}: converges");
        assert!(
            pivot < crate::lmm::GRAM_PIVOT_MIN,
            "{family:?}: pivot {pivot:e}"
        );
        // One-sided: both fits share one optimum, so a lower deviance is a
        // closer approach to it; a broken QR step can only land above.
        assert!(
            dev - dev_r <= dev_tol * dev_r,
            "{family:?}: deviance {dev} against {dev_r}"
        );
        let want_var = var_r / (delta * delta);
        assert!(
            (var - want_var).abs() <= var_tol * want_var,
            "{family:?}: var₂ {var} against {want_var}"
        );
    }
}

/// `[1, x, e, e]` (`n` rows, `e` the indicator of the last row) when
/// `dup`, else `[1, x, e]`, with a response for `family`: the Givens pass
/// over `√W·X` leaves `R_33` exactly zero, since no rotation reaches the
/// duplicate once the first copy has absorbed the last row.
fn breakdown_case(family: Family, dup: bool) -> (Mat<f64>, Vec<f64>) {
    let n = 50;
    let p = if dup { 4 } else { 3 };
    let x = Mat::<f64>::from_fn(n, p, |i, j| match j {
        0 => 1.0,
        1 => i as f64 / n as f64 - 0.5,
        _ => f64::from(u8::from(i == n - 1)),
    });
    let y = (0..n)
        .map(|i| match family {
            Family::Gamma { .. } => 1.0 + 0.25 * ((3 * i) % 7) as f64,
            _ if i == n - 1 => 3.0,
            _ => ((7 * i) % 5) as f64,
        })
        .collect();
    (x, y)
}

/// A breakdown at the last solve (`R_33 = 0` on `[1, x, e, e]`) keeps the
/// fit: converged, with the deviance of the `[1, x, e]` fit (same column
/// space) and the SEs withheld. The duplicate takes a zero step, so from a
/// cold start it stays at 0. Gamma/log also takes Newton steps, which skip
/// a pass with a broken column (from the raw Gram the duplicate would move
/// to −0.17). Measured against `[1, x, e]` (relative deviance): Poisson 0,
/// Gamma 2.1e-16.
#[test]
fn glm_breakdown_keeps_the_deviance_and_withholds_the_ses() {
    let arms = [
        Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        Family::Gamma {
            link: crate::GammaLink::Log,
        },
    ];
    for family in arms {
        let (x, y) = breakdown_case(family, true);
        let mut ws = TestWs::new(50, 4, 0);
        let v = glm_irls_fit(
            family,
            f64::NAN,
            x.as_ref(),
            &y,
            &[0, 1, 2, 3],
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        let (xr, _) = breakdown_case(family, false);
        let mut ws_r = TestWs::new(50, 3, 0);
        let r = glm_irls_fit(
            family,
            f64::NAN,
            xr.as_ref(),
            &y,
            &[0, 1, 2],
            None,
            None,
            None,
            glm_scratch(&mut ws_r),
        );
        assert!(r.converged, "{family:?}: the [1, x, e] fit converges");
        assert!(v.converged && v.se_unavailable, "{family:?}");
        assert!(!v.separated, "{family:?}");
        assert_eq!((v.pivot, v.pivot_col), (0.0, 3), "{family:?}");
        assert!(v.var_diag.iter().all(|d| d.is_nan()), "{family:?}");
        assert!(v.t_sq.iter().all(|t| t.is_nan()), "{family:?}");
        assert!(v.betas.iter().all(|b| b.is_finite()), "{family:?}");
        assert_eq!(v.betas[3], 0.0, "{family:?}");
        let rel = (v.deviance - r.deviance).abs() / r.deviance;
        assert!(
            rel <= 1e-14,
            "{family:?}: deviance {} against {}",
            v.deviance,
            r.deviance
        );
    }
}

/// From a warm start the broken-down column stays at its start value: the
/// loop tier's call shape, a draw seeded from the truth.
#[test]
fn glm_breakdown_holds_the_column_at_its_warm_start() {
    let family = Family::Poisson {
        link: crate::PoissonLink::Log,
    };
    let (x, y) = breakdown_case(family, true);
    let mut ws = TestWs::new(50, 4, 0);
    let v = glm_irls_fit(
        family,
        f64::NAN,
        x.as_ref(),
        &y,
        &[0, 1, 2, 3],
        Some(&[0.5, 0.1, 0.3, 0.25]),
        None,
        None,
        glm_scratch(&mut ws),
    );
    assert!(v.converged && v.se_unavailable);
    assert_eq!(v.betas[3], 0.25);
}

/// A column zero under the weights is not a breakdown: the Gram has a zero
/// pivot there and the fit ends not converged, with no deviance. The logit
/// arm's alternating 0/1 response keeps it off the all-0/all-1 short circuit.
#[test]
fn glm_zero_weighted_column_stays_not_converged() {
    let arms = [
        Family::Poisson {
            link: crate::PoissonLink::Log,
        },
        Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
    ];
    for family in arms {
        let (mut x, mut y) = breakdown_case(family, false);
        for i in 0..50 {
            x[(i, 2)] = 0.0;
        }
        if matches!(family, Family::Binomial { .. }) {
            y = (0..50).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
        }
        let mut ws = TestWs::new(50, 3, 0);
        let v = glm_irls_fit(
            family,
            f64::NAN,
            x.as_ref(),
            &y,
            &[0, 1, 2],
            None,
            None,
            None,
            glm_scratch(&mut ws),
        );
        assert!(!v.converged && !v.se_unavailable, "{family:?}");
        assert!(v.deviance.is_nan(), "{family:?}");
    }
}

// -----------------------------------------------------------------
// GLM deviance_null golden value (external oracle: R glm()$null.deviance)
// -----------------------------------------------------------------

#[test]
fn glm_deviance_null_golden_value() {
    // null deviance = -2*(y_sum*ln(y_bar) + (n-y_sum)*ln(1-y_bar))
    // y_sum=40, n=100, y_bar=0.4
    // → expected = -2*(40*ln(0.4) + 60*ln(0.6)) ≈ 134.6023334
    // R: glm(c(rep(1,40),rep(0,60)) ~ 1, family=binomial)$null.deviance = 134.6023
    //
    // y pattern: every 5th group, first 2 are 1 and last 3 are 0.
    // This gives exactly 40 ones spread throughout, so there is no separation
    // between y and the linearly-increasing x predictor.
    let n = 100;
    let p = 2;
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = (i as f64) / (n as f64) - 0.5; // arbitrary predictor
    }
    // y[i] = 1 for i % 5 < 2, else 0 → exactly 40 ones, not separated from x.
    let mut y = vec![0.0f64; n];
    for (i, v) in y.iter_mut().enumerate() {
        if i % 5 < 2 {
            *v = 1.0;
        }
    }

    let mut ws = TestWs::new(n, p, 0);
    let targets: Vec<u32> = vec![1];
    let fit = glm_irls_fit(
        crate::Family::Binomial {
            link: crate::BinomialLink::Logit,
        },
        f64::NAN,
        x.as_ref(),
        &y,
        &targets,
        None,
        None,
        None,
        glm_scratch(&mut ws),
    );
    assert!(fit.converged, "must converge: non-separated y pattern");

    // External oracle: R glm(y ~ x, family=binomial)$null.deviance with y_sum=40/n=100
    // (R: 134.6023); null deviance = -2*(40*ln(0.4) + 60*ln(0.6)) depends only on y_bar.
    let expected = -2.0 * (40.0 * 0.4f64.ln() + 60.0 * 0.6f64.ln());
    let rel = (fit.deviance_null - expected).abs() / expected;
    assert!(
        rel <= 1e-12,
        "deviance_null = {}, expected {expected} (rel {rel:e})",
        fit.deviance_null
    );
}

/// Weighted Gamma(log) null/residual deviance vs R glm(weights=).
/// Convention: null mean is the WEIGHTED mean μ̂₀ = Σwᵢyᵢ/Σwᵢ and both
/// deviances accumulate Σwᵢdᵢ. Same fixture as `fit/glm_tests.rs`'s
/// `fit_glm_gamma_weighted_matches_r` (β/SE/φ gated there). The Gamma
/// deviance does not change when y is rescaled, so the same references hold
/// at y × 2⁻⁴⁰, where ȳ ≈ 2e-12 sits below `family::MU_FLOOR`.
#[test]
fn glm_weighted_deviance_null_golden_value() {
    // R 4.5.3 oracle (set.seed(42), n = 40):
    //   x1 <- round(rnorm(n), 4); w <- sample(1:4, n, replace = TRUE)
    //   eta <- 0.4 + 0.8 * x1
    //   yg <- round(rgamma(n, shape = 2, scale = exp(eta) / 2), 6)
    //   fg <- glm(yg ~ x1, family = Gamma("log"), weights = w)
    //   print(fg$null.deviance, digits = 15); print(fg$deviance, digits = 15)
    let x1: [f64; 40] = [
        1.371, -0.5647, 0.3631, 0.6329, 0.4043, -0.1061, 1.5115, -0.0947, 2.0184, -0.0627, 1.3049,
        2.2866, -1.3889, -0.2788, -0.1333, 0.636, -0.2843, -2.6565, -2.4405, 1.3201, -0.3066,
        -1.7813, -0.1719, 1.2147, 1.8952, -0.4305, -0.2573, -1.7632, 0.4601, -0.64, 0.4555, 0.7048,
        1.0351, -0.6089, 0.505, -1.717, -0.7845, -0.8509, -2.4142, 0.0361,
    ];
    let w: [f64; 40] = [
        4.0, 1.0, 2.0, 1.0, 1.0, 4.0, 4.0, 1.0, 3.0, 3.0, 1.0, 4.0, 1.0, 4.0, 4.0, 2.0, 1.0, 4.0,
        2.0, 2.0, 2.0, 4.0, 1.0, 2.0, 1.0, 2.0, 4.0, 3.0, 4.0, 1.0, 4.0, 1.0, 4.0, 3.0, 2.0, 2.0,
        3.0, 1.0, 1.0, 2.0,
    ];
    let y: [f64; 40] = [
        2.421196, 0.850101, 1.188318, 0.917668, 1.895064, 2.717167, 4.391082, 0.266883, 1.853922,
        1.838375, 5.959549, 19.008523, 0.121882, 1.544704, 1.422566, 0.758422, 1.264496, 0.147806,
        0.06751, 2.907132, 0.3538, 0.223494, 0.297625, 5.273375, 12.534684, 0.514577, 1.473477,
        0.485665, 0.962023, 1.043896, 1.771311, 1.926229, 7.592099, 1.298714, 0.675125, 0.201756,
        1.814679, 1.104297, 0.434436, 0.470596,
    ];
    const REF_DEV_NULL: f64 = 140.09428224081;
    const REF_DEV: f64 = 39.1211374203115;
    let n = 40;
    let p = 2;
    let mut x = Mat::<f64>::zeros(n, p);
    for i in 0..n {
        x[(i, 0)] = 1.0;
        x[(i, 1)] = x1[i];
    }
    let targets: Vec<u32> = vec![0, 1];
    for scale in [1.0, 2f64.powi(-40)] {
        let ys: Vec<f64> = y.iter().map(|v| v * scale).collect();
        let mut ws = TestWs::new(n, p, 0);
        let fit = glm_irls_fit(
            crate::Family::Gamma {
                link: crate::GammaLink::Log,
            },
            f64::NAN,
            x.as_ref(),
            &ys,
            &targets,
            None,
            Some(&w),
            None,
            glm_scratch(&mut ws),
        );
        assert!(
            fit.converged,
            "weighted gamma GLM must converge (y × {scale})"
        );
        assert!(
            (fit.deviance_null - REF_DEV_NULL).abs() / REF_DEV_NULL < 1e-8,
            "deviance_null = {} vs R {REF_DEV_NULL} (y × {scale})",
            fit.deviance_null
        );
        assert!(
            (fit.deviance - REF_DEV).abs() / REF_DEV < 1e-6,
            "deviance = {} vs R {REF_DEV} (y × {scale})",
            fit.deviance
        );
    }
}

/// `glm_contrast_t_sq` on a hand-built view: `l` is the factor of
/// `X'WX = [[4, 6], [6, 14]]`, whose inverse is `[[0.7, −0.3], [−0.3, 0.2]]`,
/// so c = e₁ − e₀ has `cᵀ(X'WX)⁻¹c = 1.5` and t² = (2.2 − 0.7)²/1.5 = 1.5.
/// The swapped pair is exact (the forward solve of −c is the negation of
/// that of c); `se_unavailable`, a non-converged fit, a zero contrast, an
/// index out of range and a short scratch each give NaN.
#[test]
fn glm_contrast_t_sq_hand_value() {
    let mut l = Mat::<f64>::zeros(2, 2);
    l[(0, 0)] = 2.0;
    l[(1, 0)] = 3.0;
    l[(1, 1)] = 5.0_f64.sqrt();
    let betas = [0.7_f64, 2.2];
    let var_diag = [0.0_f64; 2];
    let t_sq = [0.0_f64; 2];
    let mu = [0.0_f64; 1];
    let view = GlmFitView {
        betas: &betas,
        var_diag: &var_diag,
        t_sq: &t_sq,
        l: l.as_ref(),
        n_iter: 3,
        converged: true,
        separated: false,
        deviance: 0.0,
        deviance_null: 0.0,
        mu: &mu,
        pivot: 1.0,
        pivot_col: 0,
        se_unavailable: false,
        broken: &[],
        stop: None,
    };
    let mut s = [0.0_f64; 2];
    assert!((glm_contrast_t_sq(&view, 1, 0, &mut s) - 1.5).abs() < 1e-12);
    assert_eq!(
        glm_contrast_t_sq(&view, 1, 0, &mut s).to_bits(),
        glm_contrast_t_sq(&view, 0, 1, &mut s).to_bits()
    );
    assert!(glm_contrast_t_sq(
        &GlmFitView {
            se_unavailable: true,
            ..view
        },
        1,
        0,
        &mut s
    )
    .is_nan());
    assert!(glm_contrast_t_sq(
        &GlmFitView {
            converged: false,
            ..view
        },
        1,
        0,
        &mut s
    )
    .is_nan());
    assert!(glm_contrast_t_sq(&view, 1, 1, &mut s).is_nan());
    assert!(glm_contrast_t_sq(&view, 2, 0, &mut s).is_nan());
    assert!(glm_contrast_t_sq(&view, 1, 0, &mut [0.0; 1]).is_nan());
}
