use super::*;

// Reference: system libm (std f64). Accuracy against an MPFR oracle
// (primitives ≤1 ULP) is verified offline; in-repo we re-assert SIMD == scalar-libm
// accuracy (exp/log1p ≤1 ULP, composed p ≤2 ULP) — the regression net for the coeffs.
fn ulp(a: f64, b: f64) -> i128 {
    let o = |x: f64| {
        let b = x.to_bits() as i64;
        (if b < 0 { i64::MIN.wrapping_sub(b) } else { b }) as i128
    };
    (o(a) - o(b)).abs()
}
fn libm_fused(eta: f64) -> (f64, f64) {
    if eta >= 0.0 {
        let z = (-eta).exp();
        (1.0 / (1.0 + z), eta + z.ln_1p())
    } else {
        let z = eta.exp();
        (z / (1.0 + z), z.ln_1p())
    }
}

#[test]
fn simd_kernel_within_1ulp_of_libm() {
    // Dense grid straddling the η=0 seam, widened to the certified ±700
    // domain (the exp/log1p reduction's clamp bound) — the deep tail is
    // reachable during the first three IRLS iterations, before the
    // divergence guard in `glm.rs` engages. One extra point past the clamp
    // (1e3) checks the saturated regime.
    let n = 20_002usize;
    let mut eta: Vec<f64> = (0..n)
        .map(|k| -700.0 + 1400.0 * k as f64 / n as f64)
        .collect();
    eta.push(1.0e3);
    let n = eta.len(); // 20_003 -> not a multiple of any lane count, exercises the SIMD tail too

    // Per-element accuracy of the kernel formula. `scalar_fused` is bit-identical
    // op-for-op to the SIMD path, so this guards the coefficients directly. The
    // primitives are ≤1 ULP of the true value (proved offline vs an MPFR oracle);
    // in-repo we bound the kernel-vs-libm difference (both ≤1 ULP of truth → ≤2 apart).
    let (mut pmax, mut lpmax, mut wmax) = (0i128, 0i128, 0i128);
    for &e in &eta {
        let (p, w, lp) = scalar_fused::<{ FUSED_DEFAULT }>(e);
        let (libp, liblp) = libm_fused(e);
        pmax = pmax.max(ulp(p, libp));
        lpmax = lpmax.max(ulp(lp, liblp));
        assert!(w >= crate::glm::WEIGHT_CLAMP && w.is_finite());
        // `w` is only clamp/finite-checked above, which a wrong-but-in-range
        // weight formula would still pass — recompute p(1-p) from the libm
        // reference p, floored the same way, and ULP-check against it.
        // `w = p(1-p)`'s derivative `1-2p` amplifies the p gap at interior
        // η (see the unfused twin below): measured 1 ULP against glibc,
        // 6 ULP against wasm32's libm (η ≈ 1.68, where the kernel and libm
        // p sit 1 ULP either side of the true value). Each target is pinned
        // at its own measurement + 1.
        let w_ref = (libp * (1.0 - libp)).max(crate::glm::WEIGHT_CLAMP);
        wmax = wmax.max(ulp(w, w_ref));
    }
    assert!(pmax <= 2, "sigmoid p drifted {pmax} ULP from libm");
    assert!(lpmax <= 2, "log1pexp drifted {lpmax} ULP from libm");
    let w_bound = if cfg!(target_arch = "wasm32") { 7 } else { 2 };
    assert!(
        wmax <= w_bound,
        "IRLS weight w drifted {wmax} ULP from libm p(1-p)"
    );

    // End-to-end SIMD dispatch path: p filled within the same band, and the
    // lane-reduced Σlog1pexp tracks the scalar Σ (the reorder moves only last bits).
    let mut p = vec![0.0; n];
    let mut w = vec![0.0; n];
    let lp_sum = pw_and_log1pexp_sum(&eta, &mut p, &mut w);
    let mut p_simd_max = 0i128;
    let mut ref_sum = 0.0;
    for i in 0..n {
        let (libp, liblp) = libm_fused(eta[i]);
        p_simd_max = p_simd_max.max(ulp(p[i], libp));
        ref_sum += liblp;
    }
    assert!(
        p_simd_max <= 2,
        "SIMD-path p drifted {p_simd_max} ULP from libm"
    );
    assert!(
        (lp_sum - ref_sum).abs() <= 1e-9 * ref_sum.abs().max(1.0),
        "Σlog1pexp drift {lp_sum} vs {ref_sum}"
    );
}

#[test]
fn unfused_kernel_within_3ulp_of_libm() {
    // The wasm32 arithmetic (plain mul/add, no fma), instantiated on native.
    // Cody-Waite reduction is fma-free-safe by design; only the Horner steps
    // double-round. Measured 2026-06-11 on the 20,003-pt grid: p 2 ULP,
    // log1pexp 2 ULP, SIMD p 2 ULP — bound pinned at measured+1 = 3. If it
    // ever exceeds that, a reduction step has become fma-dependent and the
    // policy is wrong — investigate before shipping.
    let n = 20_003usize;
    let eta: Vec<f64> = (0..n).map(|k| -40.0 + 80.0 * k as f64 / n as f64).collect();
    let (mut pmax, mut lpmax, mut wmax) = (0i128, 0i128, 0i128);
    for &e in &eta {
        let (p, w, lp) = scalar_fused::<false>(e);
        let (libp, liblp) = libm_fused(e);
        pmax = pmax.max(ulp(p, libp));
        lpmax = lpmax.max(ulp(lp, liblp));
        assert!(w >= crate::glm::WEIGHT_CLAMP && w.is_finite());
        // Same libm-formula ULP check as the fused kernel test, against the
        // unfused (plain mul/add) weight. `w = p(1-p)`'s derivative `1-2p`
        // amplifies whatever ULP gap the unfused `p` already carries (up to
        // 3 ULP here) at interior η, well away from either tail. Measured
        // 29 ULP; pinned at measured+1.
        let w_ref = (libp * (1.0 - libp)).max(crate::glm::WEIGHT_CLAMP);
        wmax = wmax.max(ulp(w, w_ref));
    }
    assert!(pmax <= 3, "unfused sigmoid p drifted {pmax} ULP from libm");
    assert!(lpmax <= 3, "unfused log1pexp drifted {lpmax} ULP from libm");
    assert!(
        wmax <= 30,
        "unfused IRLS weight w drifted {wmax} ULP from libm p(1-p)"
    );

    // End-to-end dispatch of the unfused op on native SIMD lanes.
    let mut p = vec![0.0; n];
    let mut w = vec![0.0; n];
    let lp_sum = pulp::Arch::new().dispatch(PwLog1pexpOp::<false> {
        eta: &eta,
        p: &mut p,
        w: &mut w,
    });
    let mut ref_sum = 0.0;
    let mut p_simd_max = 0i128;
    for i in 0..n {
        let (libp, liblp) = libm_fused(eta[i]);
        p_simd_max = p_simd_max.max(ulp(p[i], libp));
        ref_sum += liblp;
    }
    assert!(
        p_simd_max <= 3,
        "unfused SIMD p drifted {p_simd_max} ULP from libm"
    );
    assert!((lp_sum - ref_sum).abs() <= 1e-9 * ref_sum.abs().max(1.0));
}

#[test]
fn exp_fill_within_1ulp_of_libm_full_domain() {
    // Full certified domain [−700, 700] — positive arguments exercise the
    // same Cody-Waite reduction (sign-agnostic) and the 2^k build up to
    // k+1023 = 2033. Measured 2026-06-11 over this 20,003-pt grid: scalar
    // and SIMD both ≤1 ULP of libm — pinned at measured (1).
    let n = 20_003usize;
    let xs: Vec<f64> = (0..n)
        .map(|k| -700.0 + 1400.0 * k as f64 / n as f64)
        .collect();
    let mut emax = 0i128;
    for &x in &xs {
        emax = emax.max(ulp(exp_clamped(x), x.exp()));
    }
    assert!(emax <= 1, "exp_clamped drifted {emax} ULP from libm");
    let mut buf = xs.clone();
    exp_fill(&mut buf);
    let mut smax = 0i128;
    for i in 0..n {
        smax = smax.max(ulp(buf[i], xs[i].exp()));
    }
    assert!(smax <= 1, "exp_fill drifted {smax} ULP from libm");
    // Clamp bound itself, not just "still finite": a wrong clamp constant
    // would still pass a finite/positive check.
    let mut edge = vec![-1.0e9, 1.0e9];
    exp_fill(&mut edge);
    assert_eq!(edge[0].to_bits(), exp_clamped(EXP_ARG_FLOOR).to_bits());
    assert_eq!(edge[1].to_bits(), exp_clamped(EXP_ARG_CEIL).to_bits());
}

#[test]
fn phi_hp_full_double_precision() {
    // Reference Φ values (R `pnorm`, 16 figs). The probit FD-Hessian SE needs
    // Φ to ~machine precision — A&S 7.1.26's ~7.5e-8 is far too coarse here.
    let cases = [
        (-6.0, 9.865_876_450_376_968e-10),
        (-3.0, 1.349_898_031_630_095e-3),
        (-1.959_963_984_540_054, 0.025),
        (-1.0, 0.158_655_253_931_457_05),
        (0.0, 0.5),
        (0.5, 0.691_462_461_274_013_1),
        (1.0, 0.841_344_746_068_542_9),
        (1.959_963_984_540_054, 0.975),
        (3.0, 0.998_650_101_968_369_9),
        (6.0, 0.999_999_999_013_412_3),
    ];
    for (z, want) in cases {
        let got = phi_hp(z);
        assert!(
            (got - want).abs() <= 1e-14 * want.abs().max(1e-12),
            "phi_hp({z}) = {got}, want {want}"
        );
    }
    // Tail symmetry Φ(−z) = 1 − Φ(z) to full precision.
    for &z in &[0.3, 1.7, 4.2, 8.0] {
        assert!((phi_hp(-z) - (1.0 - phi_hp(z))).abs() <= 1e-15);
    }
}

#[test]
fn ln_gamma_full_double_precision() {
    // Reference lnΓ values (R `lgamma`, 16 figs), spanning the range the Gamma
    // dispersion 1/φ visits (small, ~1, integer factorials, large).
    let cases = [
        (0.5, 0.572_364_942_924_700_1), // ln(√π)
        (1.0, 0.0),
        (1.5, -0.120_782_237_635_245_22),
        (1.9, -0.038_984_275_923_082_73),
        (2.0, 0.0),
        (5.0, 3.178_053_830_347_945_6), // ln(4!) = ln 24
        (10.0, 12.801_827_480_081_469), // ln(9!)
        (100.0, 359.134_205_369_575_4),
    ];
    for (x, want) in cases {
        let got = ln_gamma(x);
        assert!(
            (got - want).abs() <= 1e-13 * want.abs().max(1.0),
            "ln_gamma({x}) = {got}, want {want}"
        );
    }
}

/// `lnΓ(x) = −ln x − γx + O(x²)`, so below 1e-12 `lnΓ(x) + ln x` is under
/// 1e-12. Without the small-x branch the series divides by zero below
/// about 1.1e-16 and returns +inf.
#[test]
fn ln_gamma_is_finite_for_tiny_x() {
    for x in [1e-20, 1e-100, 1e-300] {
        let got = ln_gamma(x);
        assert!(got.is_finite(), "ln_gamma({x}) = {got}");
        assert!((got + x.ln()).abs() <= 1e-12, "ln_gamma({x}) = {got}");
    }
}

/// The NB row terms take `lnΓ(θ)` down to the shape's lower search
/// limit, 1e-6. References: mpmath `loggamma` at 50 digits, rounded to
/// the nearest double (validation/tools/prep/ln_gamma_tiny_reference.py).
#[test]
fn ln_gamma_is_accurate_at_tiny_shapes() {
    let cases = [
        (1e-6, 13.815509980749432),
        (1e-5, 11.512919692895826),
        (1e-4, 9.210282658633963),
    ];
    let errs: Vec<f64> = cases
        .iter()
        .map(|&(x, want)| (ln_gamma(x) - want).abs() / want)
        .collect();
    assert!(
        errs.iter().all(|&e| e <= 1e-14),
        "relative errors at 1e-6, 1e-5, 1e-4: {errs:?}"
    );
}

#[test]
fn erfc_blend_accuracy_and_head_tail_identity() {
    // The probit arm's Φ must stay near machine precision (the FD-Hessian SE
    // differentiates it twice), so the branch-free blend is measured against
    // `erfc_cody` — the branching reference `phi_hp` itself calls. The grid
    // covers all three |x| regions and both seams (0.46875, 4.0); the
    // certified domain stops at √700, so 26 is the top.
    //
    // Measured 2026-08-23 on this 20,003-pt grid: 4 ULP, pinned at
    // measured+1. The blend's rationals are Cody's op-for-op, so the whole
    // gap is the `exp(−⌊16y⌋²/16²)·exp(−δ)` factor: two of this module's
    // owned `exp`s (≤1 ULP each) where Cody calls libm (≤0.5 ULP each),
    // composed through a product. That is the floor for this construction —
    // a tighter bound needs a sharper owned `exp`, not a better blend. 4 ULP
    // on Φ is 9e-16 relative, inside `phi_hp`'s own documented ~1e-15.
    let n = 20_003usize;
    let xs: Vec<f64> = (0..n).map(|k| -26.0 + 52.0 * k as f64 / n as f64).collect();
    let mut smax = 0i128;
    for &x in &xs {
        smax = smax.max(ulp(scalar_erfc_blend::<{ FUSED_DEFAULT }>(x), erfc_cody(x)));
    }
    assert!(smax <= 5, "scalar erfc blend drifted {smax} ULP from Cody");
    // SIMD head vs its scalar twin: bit-identical, the
    // `simd_fused`/`scalar_fused` discipline.
    let mut buf = xs.clone();
    erfc_fill(&mut buf);
    for i in 0..n {
        assert_eq!(
            buf[i].to_bits(),
            scalar_erfc_blend::<{ FUSED_DEFAULT }>(xs[i]).to_bits(),
            "erfc SIMD head diverged from its scalar twin at x={}",
            xs[i]
        );
    }
    // Composed through the probit link, over the whole range where μ is not
    // already clamped to `family::PROB_EPS` (|η| ≲ 7.03). Measured 4 ULP.
    let mut zmax = 0i128;
    for k in 0..n {
        let z = -8.0 + 16.0 * k as f64 / n as f64;
        let (mu, _) = scalar_probit::<{ FUSED_DEFAULT }>(z);
        zmax = zmax.max(ulp(mu, phi_hp(z)));
    }
    assert!(zmax <= 5, "probit μ drifted {zmax} ULP from phi_hp");
}

/// The cloglog arm's μ keeps its relative precision at small μ: on both
/// lanes (the whole slice, SIMD head and tail, and each row alone, the
/// scalar tail) it is within 4 ULP relative of `family::link_inv`'s
/// `−expm1(−e^η)` at η from −30 to −5, μ from `PROB_EPS` to 6.7e-3.
#[test]
fn cloglog_mu_keeps_its_relative_precision_at_small_mu() {
    use crate::spec::{BinomialLink, Family};
    let family = Family::Binomial {
        link: BinomialLink::Cloglog,
    };
    let n = 1_003usize;
    let eta: Vec<f64> = (0..n).map(|k| -30.0 + 25.0 * k as f64 / n as f64).collect();
    let y: Vec<f64> = (0..n).map(|k| (k % 2) as f64).collect();
    let pw = vec![1.0; n];
    let pass = |e: &mut [f64], yy: &[f64], w8: &[f64], p: &mut [f64]| {
        let (mut w, mut z) = (vec![0.0; e.len()], vec![0.0; e.len()]);
        family_pass(family, f64::NAN, e, yy, w8, true, 0.0, p, &mut w, &mut z);
    };
    let mut whole = vec![0.0; n];
    pass(&mut eta.clone(), &y, &pw, &mut whole);
    let mut one = vec![0.0; n];
    for i in 0..n {
        pass(&mut [eta[i]], &y[i..=i], &pw[i..=i], &mut one[i..=i]);
    }
    for i in 0..n {
        let want: f64 = crate::family::link_inv(family, eta[i]);
        for (got, lane) in [(whole[i], "whole slice"), (one[i], "one row")] {
            let rel = (got - want).abs() / want;
            assert!(
                rel <= 4.0 * f64::EPSILON,
                "{lane} at eta {}: {got:e} against {want:e} (rel {rel:e})",
                eta[i]
            );
        }
    }
}

#[test]
fn family_pass_simd_head_matches_scalar_tail() {
    // The `simd_fused`/`scalar_fused` discipline, applied per family arm: a
    // whole-slice call (SIMD head + tail) must agree TO THE BIT with the same
    // rows run one at a time (which is pure tail, every lane width).
    use crate::spec::{
        BinomialLink, Family, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink,
    };
    let n = 1_003usize;
    let eta: Vec<f64> = (0..n).map(|k| -4.0 + 8.0 * k as f64 / n as f64).collect();
    let pw: Vec<f64> = (0..n).map(|k| 1.0 + (k % 5) as f64).collect();
    let cases: [(Family, f64); 8] = [
        (
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            f64::NAN,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Cloglog,
            },
            f64::NAN,
        ),
        (
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            f64::NAN,
        ),
        (
            Family::Poisson {
                link: PoissonLink::Log,
            },
            f64::NAN,
        ),
        (
            Family::Gamma {
                link: GammaLink::Log,
            },
            f64::NAN,
        ),
        (
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            1.7,
        ),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
            f64::NAN,
        ),
        (
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            },
            f64::NAN,
        ),
    ];
    for (family, nb_theta) in cases {
        let y: Vec<f64> = match family {
            Family::Binomial { .. } => (0..n).map(|k| (k % 3) as f64 / 2.0).collect(),
            Family::Gamma { .. } | Family::InverseGaussian { .. } => {
                (0..n).map(|k| 0.5 + (k % 7) as f64).collect()
            }
            _ => (0..n).map(|k| (k % 9) as f64).collect(),
        };
        let run =
            |lo: usize, hi: usize, e: &mut [f64], p: &mut [f64], w: &mut [f64], z: &mut [f64]| {
                family_pass(
                    family,
                    nb_theta,
                    &mut e[lo..hi],
                    &y[lo..hi],
                    &pw[lo..hi],
                    true,
                    0.0,
                    &mut p[lo..hi],
                    &mut w[lo..hi],
                    &mut z[lo..hi],
                )
            };
        let (mut e1, mut p1, mut w1, mut z1) =
            (eta.clone(), vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        run(0, n, &mut e1, &mut p1, &mut w1, &mut z1);
        let (mut e2, mut p2, mut w2, mut z2) =
            (eta.clone(), vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            run(i, i + 1, &mut e2, &mut p2, &mut w2, &mut z2);
        }
        for i in 0..n {
            for (a, b, what) in [
                (e1[i], e2[i], "eta"),
                (p1[i], p2[i], "mu"),
                (w1[i], w2[i], "w"),
                (z1[i], z2[i], "z"),
            ] {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{family:?} {what} head/tail split at row {i}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
fn family_pass_gamma_inverse_flags_infeasible_eta() {
    use crate::spec::{Family, GammaLink};
    let f = Family::Gamma {
        link: GammaLink::Inverse,
    };
    let n = 37usize;
    let y = vec![1.5; n];
    let ok: Vec<f64> = (0..n).map(|k| 0.1 + 0.05 * k as f64).collect();
    let run = |eta: &[f64]| {
        let (mut e, mut p, mut w, mut z) = (eta.to_vec(), vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        family_pass(
            f,
            f64::NAN,
            &mut e,
            &y,
            &[],
            false,
            0.0,
            &mut p,
            &mut w,
            &mut z,
        )
        .1
    };
    assert!(!run(&ok));
    // One negative η anywhere in the head, and one in the sub-lane tail.
    let mut head_bad = ok.clone();
    head_bad[2] = -0.5;
    assert!(run(&head_bad));
    let mut tail_bad = ok.clone();
    tail_bad[n - 1] = -0.5;
    assert!(run(&tail_bad));
}

#[test]
fn exp_fill_infinity_saturates_the_domain_clamp() {
    // ±inf sits past [-700, 700] on either side, ordered normally against
    // the clamp bounds (no NaN ambiguity), so the SIMD head's `min_f64s`/
    // `max_f64s` and the scalar tail's `f64::clamp` agree: both floor/ceil
    // it exactly like any other out-of-range finite input.
    let mut short = vec![f64::NEG_INFINITY, f64::INFINITY]; // scalar tail only
    exp_fill(&mut short);
    assert_eq!(short[0].to_bits(), exp_clamped(EXP_ARG_FLOOR).to_bits());
    assert_eq!(short[1].to_bits(), exp_clamped(EXP_ARG_CEIL).to_bits());

    let mut long = vec![0.0; 16]; // forces indices 0,1 through the SIMD head
    long[0] = f64::NEG_INFINITY;
    long[1] = f64::INFINITY;
    for (i, x) in long.iter_mut().enumerate().skip(2) {
        *x = -4.0 + i as f64;
    }
    exp_fill(&mut long);
    assert_eq!(long[0].to_bits(), exp_clamped(EXP_ARG_FLOOR).to_bits());
    assert_eq!(long[1].to_bits(), exp_clamped(EXP_ARG_CEIL).to_bits());
}

/// A NaN comes out of `exp_fill` as NaN or as exp(−700), never as any
/// other value. Which one depends on where it lands and on the backend:
/// the scalar tail's `f64::clamp` keeps NaN; a SIMD lane clamps with the
/// backend's `min`/`max`, which drop the NaN on x86_64 (AVX2/AVX-512) and
/// on pulp's scalar backend (lane width 1: plain wasm32, x86_64 without
/// AVX2) but propagate it on wasm32 `+simd128`.
#[test]
fn exp_fill_nan_comes_out_nan_or_floored() {
    let floor = exp_clamped(EXP_ARG_FLOOR);
    let mut short = vec![f64::NAN];
    exp_fill(&mut short);
    let mut long = vec![0.0; 16];
    long[0] = f64::NAN;
    for (i, x) in long.iter_mut().enumerate().skip(1) {
        *x = -4.0 + i as f64;
    }
    exp_fill(&mut long);
    for v in [short[0], long[0]] {
        assert!(v.is_nan() || v.to_bits() == floor.to_bits(), "{v}");
    }
}

/// `ln_pos_fill` against libm over the whole positive normal range
/// (log-spaced from `f64::MIN_POSITIVE` to `1e300`, plus values next to 1,
/// the NB deviance's ratio near the mode): within 2 ULP everywhere, and the
/// SIMD head equals its scalar tail element by element.
#[test]
fn ln_pos_fill_matches_libm_within_2ulp() {
    let mut xs: Vec<f64> = (0..20_000)
        .map(|k| 10f64.powf(-307.0 + 607.0 * k as f64 / 19_999.0))
        .collect();
    for k in 1..2000 {
        xs.push(1.0 + (k as f64 - 1000.0) * 1e-6);
        xs.push(1.0 + (k as f64 - 1000.0) * 1e-14);
    }
    xs.push(f64::MIN_POSITIVE);
    xs.push(1.0);
    xs.push(2.0);
    xs.push(std::f64::consts::SQRT_2);
    let mut got = xs.clone();
    ln_pos_fill(&mut got);
    for (&x, &g) in xs.iter().zip(&got) {
        let r = x.ln();
        let ulp = if r == 0.0 {
            f64::MIN_POSITIVE
        } else {
            (f64::from_bits(r.abs().to_bits() + 1) - r.abs()).abs()
        };
        assert!((g - r).abs() <= 2.0 * ulp, "ln({x}) = {g}, libm {r}");
        assert_eq!(
            g.to_bits(),
            scalar_ln_pos::<{ FUSED_DEFAULT }>(x).to_bits(),
            "x={x}"
        );
    }
}

/// `ln_pos_fill` outside its domain: no NaN or infinity comes out on any
/// backend. The scalar tail floors a NaN to `MIN_POSITIVE` and reads `+∞`
/// as `2¹⁰²⁴`; a SIMD lane gives a finite value whose bits depend on the
/// backend's `max` (see the function's doc), so only finiteness is pinned
/// there.
#[test]
fn ln_pos_fill_non_finite_input_comes_out_finite() {
    let mut tail_only = vec![f64::NAN];
    ln_pos_fill(&mut tail_only);
    assert_eq!(tail_only[0].to_bits(), f64::MIN_POSITIVE.ln().to_bits());
    let mut tail_inf = vec![f64::INFINITY];
    ln_pos_fill(&mut tail_inf);
    assert!((tail_inf[0] - 1024.0 * std::f64::consts::LN_2).abs() < 1e-12);

    let mut head = vec![0.5; 16];
    head[0] = f64::NAN;
    head[1] = f64::INFINITY;
    ln_pos_fill(&mut head);
    assert!(head.iter().all(|v| v.is_finite()), "{head:?}");
}

/// The NB arm of `family_pass` reads its `ln`s through `ln_pos_fill`,
/// which maps a NaN to a finite value; a NaN μ must still give the NaN
/// deviance `family::dev_resid` gives. A NaN η reaches μ as NaN on the
/// scalar tail (`f64::clamp` keeps NaN), which a 1-row call and the last
/// row of a 17-row call are on the lane-width 2, 4 and 8 backends. Pulp's
/// scalar backend (lane width 1) has no tail and floors the NaN η, giving a
/// finite μ; the deviance must then be finite too.
#[test]
fn family_pass_nb_nan_mu_gives_nan_deviance() {
    let fam = Family::NegativeBinomial {
        link: crate::spec::NegBinomialLink::Log,
    };
    for len in [1usize, 17] {
        let mut eta = vec![0.3; len];
        eta[len - 1] = f64::NAN;
        let y = vec![2.0; len];
        let (mut prob, mut w) = (vec![0.0; len], vec![0.0; len]);
        let (dev, _) = family_pass(
            fam,
            5.0,
            &mut eta,
            &y,
            &[],
            false,
            0.0,
            &mut prob,
            &mut w,
            &mut [],
        );
        assert_eq!(
            dev.is_nan(),
            prob[len - 1].is_nan(),
            "len {len}: deviance {dev}, μ {}",
            prob[len - 1]
        );
    }
}

/// At θ = 1e-6 and η = 700 the ratio θ/(μ+θ) is about 1e-310, subnormal,
/// and `ln_pos_fill` floors it to `f64::MIN_POSITIVE`; a fold that kept
/// that log would be 0.76% off. With y = 0 each row's deviance is
/// `2θ·ln((μ+θ)/θ)`. 17 rows put rows on the SIMD head and the scalar
/// tail on lane widths 2, 4 and 8.
#[test]
fn family_pass_nb_tiny_shape_at_eta_max_is_exact() {
    let fam = Family::NegativeBinomial {
        link: crate::spec::NegBinomialLink::Log,
    };
    let th = 1e-6;
    let len = 17;
    let mut eta = vec![crate::family::ETA_MAX; len];
    let y = vec![0.0; len];
    let (mut prob, mut w) = (vec![0.0; len], vec![0.0; len]);
    let (dev, _) = family_pass(
        fam,
        th,
        &mut eta,
        &y,
        &[],
        false,
        0.0,
        &mut prob,
        &mut w,
        &mut [],
    );
    let want = len as f64 * 2.0 * th * ((700f64.exp() + th).ln() - th.ln());
    assert!(
        ((dev - want) / want).abs() <= 1e-12,
        "deviance {dev}, want {want}"
    );
}

/// A NaN η on a link that clamps η is flagged as not evaluable, wherever
/// it lands: row 0 of a 17-row call is a SIMD lane on every backend, row
/// 16 and a 1-row call are the scalar tail on lane widths 2, 4 and 8. The
/// SIMD clamp drops the NaN on x86 and on pulp's scalar backend, so without
/// the flag that row would give a finite μ and deviance at the lower clamp
/// bound.
#[test]
fn family_pass_flags_nan_eta_on_clamped_links() {
    use crate::spec::{NegBinomialLink, PoissonLink};
    let fams = [
        Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        Family::Poisson {
            link: PoissonLink::Log,
        },
        Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Log,
        },
        Family::Gamma {
            link: GammaLink::Inverse,
        },
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        },
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        },
    ];
    let mut checked = 0;
    for fam in fams {
        for (len, at) in [(1usize, 0usize), (17, 0), (17, 16)] {
            let mut eta = vec![0.3; len];
            eta[at] = f64::NAN;
            let y = vec![1.0; len];
            let (mut prob, mut w) = (vec![0.0; len], vec![0.0; len]);
            let (dev, infeasible) = family_pass(
                fam,
                5.0,
                &mut eta,
                &y,
                &[],
                false,
                0.0,
                &mut prob,
                &mut w,
                &mut [],
            );
            assert!(infeasible, "{fam:?} len {len}, NaN at {at}: deviance {dev}");
            checked += 1;
        }
    }
    assert_eq!(checked, 21);
}

/// A NaN η makes `Σ log1pexp` NaN wherever it lands: the scalar tail (a
/// 1-row call, the last row of a 17-row call on lane widths 2, 4 and 8)
/// and a SIMD lane (row 0 of the 17-row call). So the unweighted-logit
/// deviance `2·(Σ log1pexp − Σ y·η)` is NaN through this sum as well as
/// through `Σ y·η`. `+∞` gives `+∞`.
#[test]
fn pw_and_log1pexp_sum_nan_eta_gives_nan_sum() {
    assert!(scalar_log1pexp(f64::NAN).is_nan());
    for (len, at) in [(1usize, 0usize), (17, 0), (17, 16)] {
        let mut eta = vec![0.3; len];
        eta[at] = f64::NAN;
        let (mut p, mut w) = (vec![0.0; len], vec![0.0; len]);
        let sum = pw_and_log1pexp_sum(&eta, &mut p, &mut w);
        assert!(sum.is_nan(), "len {len}, NaN at {at}: sum {sum}");
    }

    let eta = [f64::INFINITY];
    let mut p = [0.0];
    let mut w = [0.0];
    let sum = pw_and_log1pexp_sum(&eta, &mut p, &mut w);
    assert_eq!(p[0], 1.0);
    assert_eq!(w[0], crate::glm::WEIGHT_CLAMP);
    assert!(sum.is_infinite() && sum > 0.0);
}

/// Separated rows at `|η|` up to 3e296: `Σ log1pexp(η)` and `Σ y·η` are
/// both near 5e297 and agree to their last bit, so the fused
/// `2·(Σ log1pexp − Σ y·η)` keeps only rounding noise. The exact deviance
/// is `2·ln 2`, all from the `η = 0, y = 0` row, and both the SIMD route
/// and its generic mirror must return it.
#[test]
fn unweighted_logit_deviance_survives_cancelling_sums() {
    let fam = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let n = 61;
    let eta0: Vec<f64> = (0..n).map(|i| 1e295 * (i as f64 - 30.0)).collect();
    let y: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i > 30))).collect();
    let yeta: f64 = eta0.iter().zip(&y).map(|(e, yi)| e * yi).sum();
    let want = 2.0 * std::f64::consts::LN_2;
    let (mut prob, mut w) = (vec![0.0; n], vec![0.0; n]);
    let mut eta = eta0.clone();
    let (simd, _) = family_pass(
        fam,
        f64::NAN,
        &mut eta,
        &y,
        &[],
        false,
        yeta,
        &mut prob,
        &mut w,
        &mut [],
    );
    let mut eta = eta0;
    let (generic, _) = crate::scalar::generic_family_pass(
        fam,
        f64::NAN,
        &mut eta,
        &y,
        &[],
        false,
        yeta,
        &mut prob,
        &mut w,
        &mut [],
    );
    for (route, dev) in [("simd", simd), ("generic", generic)] {
        assert!(
            (dev - want).abs() < 1e-12,
            "{route}: deviance {dev:e}, want {want}"
        );
    }
}

#[cfg(test)]
mod ln_phi_tests {
    use super::*;

    /// `ln_phi_fill` against `family::probit_side`'s `ln Φ` (Cody `erfc`, the
    /// asymptotic form past `x = −4√2`, `ln_1p(−Φ(−x))` on `x ≥ 0`) over the
    /// whole served range `[−37, 37]`, both `erfc` seams (`|x|/√2` = 0.46875
    /// and 4) and `probit_side`'s own seam at `−4√2`, as relative error; and
    /// the SIMD head against the scalar tail (one-element calls are all tail)
    /// to the bit.
    #[test]
    fn ln_phi_fill_matches_probit_side() {
        let n = 200_001usize;
        let mut xs: Vec<f64> = (0..n)
            .map(|k| -LN_PHI_X_MAX + 2.0 * LN_PHI_X_MAX * k as f64 / (n - 1) as f64)
            .collect();
        let r2 = std::f64::consts::SQRT_2;
        for c in [0.46875 * r2, 4.0 * r2, 0.0, 1e-300, 1e-17, 1e-8, 36.999] {
            for d in [-1e-12, 0.0, 1e-12] {
                xs.push(c + d);
                xs.push(-(c + d));
            }
        }
        let mut buf = xs.clone();
        let mut nq = vec![0.0; buf.len()];
        ln_phi_fill(&mut buf, &mut nq);
        let mut worst = 0.0f64;
        let mut worst_x = 0.0;
        for (&x, &got) in xs.iter().zip(&buf) {
            let want = crate::family::probit_side(x).0;
            let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
            if rel > worst {
                worst = rel;
                worst_x = x;
            }
        }
        eprintln!("ln_phi_fill: worst relative error {worst:e} at x = {worst_x}");
        assert!(
            worst <= LN_PHI_REL_BOUND,
            "ln_phi_fill off by {worst:e} at x = {worst_x}"
        );
        for (k, &x) in xs.iter().enumerate() {
            let mut one = [x];
            let mut nq1 = [0.0];
            ln_phi_fill(&mut one, &mut nq1);
            assert_eq!(
                one[0].to_bits(),
                buf[k].to_bits(),
                "head and tail differ at x = {x}"
            );
        }
    }
}

#[cfg(test)]
mod binomial_fold_tests {
    use crate::spec::{BinomialLink, Family};

    /// The binomial deviance fold against `family::dev_resid_at` row by row
    /// (probit, cloglog and logit; η over both tails; y = 0, 1 and a
    /// proportion), its weighted sum against the scalar sum, and a NaN μ. The
    /// per-row bounds are about 4x the maxima measured 2026-10-07 on x86
    /// (absolute 1.4e-14; relative 2.1e-13 on rows with d > 1e-6): a row's
    /// deviance runs to 0 at y = μ, where its error is the `ln`'s ULP times a
    /// term of order |y − μ|, not of d.
    #[test]
    fn binomial_dev_fold_matches_dev_resid_at() {
        let (mut worst_abs, mut worst_big) = (0.0f64, 0.0f64);
        for link in [
            BinomialLink::Probit,
            BinomialLink::Cloglog,
            BinomialLink::Logit,
        ] {
            let fam = Family::Binomial { link };
            let mut eta = Vec::new();
            let mut e = -12.0;
            while e <= 12.0 {
                eta.push(crate::family::clamp_eta(fam, e));
                e += 0.000_917;
            }
            let prob: Vec<f64> = eta
                .iter()
                .map(|&e| crate::family::link_inv(fam, e))
                .collect();
            let n = eta.len();
            for yv in [0.0, 1.0, 0.3, 0.999] {
                let y = vec![yv; n];
                let pw: Vec<f64> = (0..n).map(|i| 1.0 + (i % 7) as f64 * 0.25).collect();
                for i in 0..n {
                    let a = super::binomial_dev_fold(
                        fam,
                        &eta[i..i + 1],
                        &y[i..i + 1],
                        &[],
                        &prob[i..i + 1],
                    );
                    let b = crate::family::dev_resid_at(fam, f64::NAN, yv, eta[i], prob[i]);
                    if crate::family::in_tail(fam, eta[i], prob[i]) {
                        assert_eq!(a.to_bits(), b.to_bits(), "{link:?} tail row η = {}", eta[i]);
                    }
                    worst_abs = worst_abs.max((a - b).abs());
                    if b.abs() > 1e-6 {
                        worst_big = worst_big.max((a - b).abs() / b.abs());
                    }
                }
                let sum = super::binomial_dev_fold(fam, &eta, &y, &pw, &prob);
                let want: f64 = (0..n)
                    .map(|i| {
                        pw[i] * crate::family::dev_resid_at(fam, f64::NAN, yv, eta[i], prob[i])
                    })
                    .sum();
                assert!(
                    (sum - want).abs() <= 1e-12 * want.abs().max(1.0),
                    "{link:?} y={yv}: {sum} vs {want}"
                );
            }
        }
        eprintln!(
            "binomial fold vs dev_resid_at: abs {worst_abs:.3e}, rel(d > 1e-6) {worst_big:.3e}"
        );
        assert!(worst_abs <= 6e-14, "fold abs {worst_abs:e}");
        assert!(worst_big <= 1e-12, "fold rel {worst_big:e}");
        let fam = Family::Binomial {
            link: BinomialLink::Probit,
        };
        let d = super::binomial_dev_fold(fam, &[0.0, 0.1], &[1.0, 0.0], &[], &[0.5, f64::NAN]);
        assert!(d.is_nan());
    }
}
