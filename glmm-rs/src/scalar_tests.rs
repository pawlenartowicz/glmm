/// `generic_family_pass::<f64>` (the trait default's logic, extracted so a
/// test can reach it — `impl Scalar for f64` overrides the default itself
/// with the SIMD kernel) must agree with `simd_transcendental::family_pass`
/// on four families (logit, probit, Poisson-log, Gamma-log), unweighted, since
/// both compute the same math on the same η/y/prior_w.
///
/// Unweighted logit and probit go through different owned kernels by
/// design (`scalar_sigmoid_owned`/`scalar_erfc_blend` in the SIMD path vs
/// `sigmoid_stable`/`phi_hp` here), so they are banded, citing the same
/// gap the existing `erfc_blend_accuracy_and_head_tail_identity` pins (≤5
/// ULP on Φ). Poisson-log and Gamma-log call the identical scalar chain
/// (`link_inv`/`variance`/`dev_resid`) in both routes, but the SIMD path's
/// `exp` is a vector `exp` (via `pulp`) and the generic default's `exp`
/// binds to `f64::exp` — measured 2026-08-31 on this grid: 1 ULP apart on
/// `μ` at one of eight points (`η=-0.1`), so both families are banded on
/// an absolute tolerance rather than asserted `==`.
#[test]
fn generic_default_agrees_with_simd_family_pass() {
    use crate::spec::{BinomialLink, Family, GammaLink, PoissonLink};

    type PassResult = (f64, bool, Vec<f64>, Vec<f64>, Vec<f64>);

    fn run(family: Family, eta0: &[f64], y: &[f64], weighted: bool) -> (PassResult, PassResult) {
        let n = eta0.len();
        let prior_w: Vec<f64> = vec![];
        let yeta = if weighted {
            0.0
        } else {
            eta0.iter().zip(y).map(|(&e, &yy)| yy * e).sum()
        };

        let mut eta_a = eta0.to_vec();
        let mut prob_a = vec![0.0; n];
        let mut w_a = vec![0.0; n];
        let mut z_a = vec![0.0; n];
        let (dev_a, inf_a) = super::generic_family_pass(
            family,
            f64::NAN,
            &mut eta_a,
            y,
            &prior_w,
            weighted,
            yeta,
            &mut prob_a,
            &mut w_a,
            &mut z_a,
        );

        let mut eta_b = eta0.to_vec();
        let mut prob_b = vec![0.0; n];
        let mut w_b = vec![0.0; n];
        let mut z_b = vec![0.0; n];
        let (dev_b, inf_b) = crate::simd_transcendental::family_pass(
            family,
            f64::NAN,
            &mut eta_b,
            y,
            &prior_w,
            weighted,
            yeta,
            &mut prob_b,
            &mut w_b,
            &mut z_b,
        );

        (
            (dev_a, inf_a, eta_a, prob_a, w_a),
            (dev_b, inf_b, eta_b, prob_b, w_b),
        )
    }

    let etas: Vec<f64> = vec![-3.0, -1.0, -0.1, 0.0, 0.2, 1.0, 3.0, 8.0];
    let ys: Vec<f64> = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0];

    // Unweighted Bernoulli logit — fused fast path on both sides, but
    // through different owned sigmoid/log1pexp kernels: banded.
    let (a, b) = run(
        Family::Binomial {
            link: BinomialLink::Logit,
        },
        &etas,
        &ys,
        false,
    );
    assert!(
        (a.0 - b.0).abs() < 1e-9,
        "logit deviance drifted: {a:?} vs {b:?}"
    );
    for (pa, pb) in a.3.iter().zip(&b.3) {
        assert!((pa - pb).abs() < 1e-12, "logit prob drifted");
    }

    // Probit — general branch on both sides, but `probit_cdf` binds to
    // `phi_hp` here vs `scalar_erfc_blend` in the SIMD kernel: banded at
    // the same ≤5 ULP gap `erfc_blend_accuracy_and_head_tail_identity`
    // pins on Φ itself, which loosens to a small absolute band once it
    // propagates through the weight and deviance folds.
    let (a, b) = run(
        Family::Binomial {
            link: BinomialLink::Probit,
        },
        &etas,
        &ys,
        false,
    );
    assert!(
        (a.0 - b.0).abs() < 1e-9,
        "probit deviance drifted: {a:?} vs {b:?}"
    );
    for (pa, pb) in a.3.iter().zip(&b.3) {
        assert!((pa - pb).abs() < 1e-12, "probit prob drifted");
    }

    // Poisson-log — same scalar chain on both routes; ~1 ULP from the
    // SIMD exp — banded.
    let (a, b) = run(
        Family::Poisson {
            link: PoissonLink::Log,
        },
        &etas,
        &[0.0, 1.0, 2.0, 0.0, 3.0, 1.0, 5.0, 4.0],
        false,
    );
    assert!((a.0 - b.0).abs() < 1e-12, "poisson-log deviance drifted");
    for (pa, pb) in a.3.iter().zip(&b.3) {
        assert!(
            (pa - pb).abs() <= 2.0 * f64::EPSILON * pa.abs().max(1.0),
            "poisson-log mu drifted beyond 2 ULP"
        );
    }

    // Gamma-log — same scalar chain on both routes; ~1 ULP from the
    // SIMD exp — banded.
    let (a, b) = run(
        Family::Gamma {
            link: GammaLink::Log,
        },
        &etas,
        &[0.5, 1.2, 2.0, 0.3, 4.0, 1.0, 6.0, 2.5],
        false,
    );
    assert!((a.0 - b.0).abs() < 1e-12, "gamma-log deviance drifted");
    for (pa, pb) in a.3.iter().zip(&b.3) {
        assert!(
            (pa - pb).abs() <= 2.0 * f64::EPSILON * pa.abs().max(1.0),
            "gamma-log mu drifted beyond 2 ULP"
        );
    }
}
