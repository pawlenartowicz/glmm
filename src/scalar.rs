//! The scalar type the blocked fit kernel is generic over.
//!
//! `f64` is the production instantiation and is bit-identical to the
//! pre-generic kernel. A forward dual number implements the same trait.
//!
//! Control flow never depends on the derivative part: every branch in the
//! kernel — convergence bands, step-halving, Cholesky pivot signs, sparsity
//! skips — compares [`Scalar::value`], so the iterate path a fit takes is the
//! same at every `T`.

use crate::spec::{BinomialLink, Family};

/// Sealing supertrait: private, so `Scalar` cannot be implemented outside the
/// crate and gaining a method is not a semver break.
mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}
    impl<const N: usize> Sealed for crate::dual::Dual<N> {}
    impl<const N: usize, const H: usize> Sealed for crate::dual::HyperDual<N, H> {}
}

/// One scalar of the fit kernel: the arithmetic, the elementary functions and
/// the batched family pass the likelihood needs.
#[doc(hidden)]
pub trait Scalar:
    sealed::Sealed
    + Copy
    + Send
    + Sync
    + std::fmt::Debug
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// True only for the `f64` implementation. Read by the one kernel block
    /// that is still `f64`-only (the PIRLS β-profiling border), which asserts
    /// on it rather than silently dropping derivatives.
    const IS_F64: bool;

    /// Lift a constant. The derivative part, where one exists, is zero.
    fn from_f64(v: f64) -> Self;
    /// The value part. `f64`'s implementation is the identity.
    fn value(self) -> f64;

    /// `√x`.
    fn sqrt(self) -> Self;
    /// `x.max(other)`, selected on the value part.
    fn max_f64(self, other: f64) -> Self;
    /// `x.clamp(lo, hi)`, selected on the value part.
    fn clamp_f64(self, lo: f64, hi: f64) -> Self;

    /// `exp(x)`.
    fn exp(self) -> Self;
    /// `exp(x) − 1`, accurate near zero. The cloglog inverse link is written
    /// `−expm1(−e^η)` to keep small μ's relative precision; f' = f'' = exp.
    fn exp_m1(self) -> Self;
    /// `ln(x)`, `x > 0`.
    fn ln(self) -> Self;
    /// `ln(1 + x)`, `x > −1`, accurate near zero.
    fn ln_1p(self) -> Self;
    /// `log(1 + exp(x))`, overflow-safe on both tails.
    fn log1pexp(self) -> Self;
    /// Logistic `1/(1+exp(−x))`, overflow-safe on both tails.
    fn sigmoid(self) -> Self;
    /// Standard normal CDF `Φ(x)` — the probit inverse link.
    fn probit_cdf(self) -> Self;
    /// `lnΓ(x)`, `x > 0`.
    fn ln_gamma(self) -> Self;

    /// The batched per-family η-pass: reads the raw `eta` in place and leaves
    /// it holding `family::clamp_eta`'s projection, fills `prob` with μ and `w`
    /// with the floored IRLS weight, fills `z` with the working response when
    /// non-empty. Returns `(Σ wᵢdᵢ, any raw η the link does not evaluate)`
    /// (`family::eta_infeasible`).
    /// `y`, `prior_w` and `nb_theta` are data and stay `f64`.
    ///
    /// The default body is the scalar statement in `crate::family`, row by row.
    /// `f64` overrides it with the SIMD kernel; see the module docs on
    /// `crate::simd_transcendental`.
    #[allow(clippy::too_many_arguments)]
    fn family_pass(
        family: Family,
        nb_theta: f64,
        eta: &mut [Self],
        y: &[f64],
        prior_w: &[f64],
        weighted: bool,
        yeta: Self,
        prob: &mut [Self],
        w: &mut [Self],
        z: &mut [Self],
    ) -> (Self, bool) {
        generic_family_pass(
            family, nb_theta, eta, y, prior_w, weighted, yeta, prob, w, z,
        )
    }

    /// The slice as `f64`s: `Some` only for the `f64` implementation, `None`
    /// for the dual types. Lets a generic kernel take an `f64`-only route
    /// (the AGQ node sums, `glmm::agq_fast`) without a trait bound of its own.
    #[inline(always)]
    fn as_f64s(s: &[Self]) -> Option<&[f64]> {
        let _ = s;
        None
    }
    /// Mutable twin of [`Scalar::as_f64s`].
    #[inline(always)]
    fn as_f64s_mut(s: &mut [Self]) -> Option<&mut [f64]> {
        let _ = s;
        None
    }
}

/// The default body of [`Scalar::family_pass`], the row-by-row scalar
/// statement mirroring `simd_transcendental::family_pass` arm for arm — kept
/// as a free function (rather than inlined into the trait default) so a test
/// can call it directly at `T = f64` and compare it against the SIMD kernel
/// that `impl Scalar for f64` uses instead.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generic_family_pass<T: Scalar>(
    family: Family,
    nb_theta: f64,
    eta: &mut [T],
    y: &[f64],
    prior_w: &[f64],
    weighted: bool,
    yeta: T,
    prob: &mut [T],
    w: &mut [T],
    z: &mut [T],
) -> (T, bool) {
    let n = eta.len();
    let clamp = crate::glm::WEIGHT_CLAMP;
    // Unweighted Bernoulli logit keeps the fused 2·(Σ log1pexp(η) − Σ y·η)
    // identity, mirroring the f64 route's fast path (which reaches it through
    // the SIMD kernel instead).
    if matches!(
        family,
        Family::Binomial {
            link: BinomialLink::Logit
        }
    ) && !weighted
    {
        let mut lp = T::ZERO;
        for i in 0..n {
            let e = eta[i];
            let p = e.sigmoid();
            prob[i] = p;
            w[i] = (p * (T::ONE - p)).max_f64(clamp);
            lp += e.log1pexp();
            if !z.is_empty() {
                z[i] = e + (T::from_f64(y[i]) - p) / w[i];
            }
        }
        return (logit_fused_deviance(lp, yeta, eta, y), false);
    }
    let mut dev = T::ZERO;
    let mut infeasible = false;
    for i in 0..n {
        infeasible |= crate::family::eta_infeasible(family, y[i], eta[i]);
        eta[i] = crate::family::clamp_eta(family, eta[i]);
        let (mu, w_raw, r) = crate::family::irls_weight_and_resid(family, nb_theta, y[i], eta[i]);
        let pw = if prior_w.is_empty() { 1.0 } else { prior_w[i] };
        prob[i] = mu;
        w[i] = (T::from_f64(pw) * w_raw).max_f64(clamp);
        if !z.is_empty() {
            z[i] = eta[i] + r;
        }
        dev += T::from_f64(pw) * crate::family::dev_resid_at(family, nb_theta, y[i], eta[i], mu);
    }
    (dev, infeasible)
}

/// Largest `Σ log1pexp(ηᵢ) / Σ (log1pexp(ηᵢ) − yᵢηᵢ)` at which
/// [`logit_fused_deviance`] keeps the fused value: 2²⁶, so the subtraction
/// cancels at most half of the 53 significand bits. Ordinary fits sit orders
/// of magnitude below it and keep the fused value bit for bit.
const LOGIT_FUSED_CANCEL_MAX: f64 = 67_108_864.0;

/// The unweighted Bernoulli-logit deviance `2·Σ (log1pexp(ηᵢ) − yᵢηᵢ)` from
/// the fused sums `lp = Σ log1pexp(ηᵢ)` and `yeta = Σ yᵢηᵢ`, shared by the
/// SIMD route (`simd_transcendental::family_pass`) and its generic mirror.
///
/// The two sums are each about `Σ|ηᵢ|`, so their difference carries an
/// absolute rounding error of about `ε·lp`, not `ε·D`. Where every row sits
/// far out on the side its `y` predicts (a separated fit at large `|η|`), the
/// row terms are tiny next to `|η|` and that error is the whole result, of
/// either sign: at `|η|` near 1e297 it reads as a deviance of −1e283. Past
/// [`LOGIT_FUSED_CANCEL_MAX`] the deviance is summed row by row instead, as
/// `(1 − y)·log1pexp(η) + y·log1pexp(−η)`, which equals
/// `log1pexp(η) − yη` because `log1pexp(η) − η = log1pexp(−η)` and has two
/// non-negative terms, so nothing cancels. A non-finite sum stays the fused
/// value, so a NaN or infinite η still shows in the deviance.
pub(crate) fn logit_fused_deviance<T: Scalar>(lp: T, yeta: T, eta: &[T], y: &[f64]) -> T {
    let half = lp - yeta;
    let (lp_v, yeta_v) = (lp.value(), yeta.value());
    if lp_v.is_finite() && yeta_v.is_finite() && lp_v > LOGIT_FUSED_CANCEL_MAX * half.value() {
        let mut rows = T::ZERO;
        for (&e, &yi) in eta.iter().zip(y) {
            rows += T::from_f64(1.0 - yi) * e.log1pexp() + T::from_f64(yi) * (-e).log1pexp();
        }
        return T::from_f64(2.0) * rows;
    }
    T::from_f64(2.0) * half
}

impl Scalar for f64 {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const IS_F64: bool = true;

    #[inline(always)]
    fn from_f64(v: f64) -> f64 {
        v
    }
    #[inline(always)]
    fn value(self) -> f64 {
        self
    }

    #[inline(always)]
    fn sqrt(self) -> f64 {
        f64::sqrt(self)
    }
    #[inline(always)]
    fn max_f64(self, other: f64) -> f64 {
        f64::max(self, other)
    }
    #[inline(always)]
    fn clamp_f64(self, lo: f64, hi: f64) -> f64 {
        f64::clamp(self, lo, hi)
    }

    #[inline(always)]
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    #[inline(always)]
    fn exp_m1(self) -> f64 {
        f64::exp_m1(self)
    }
    #[inline(always)]
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    #[inline(always)]
    fn ln_1p(self) -> f64 {
        f64::ln_1p(self)
    }
    #[inline(always)]
    fn log1pexp(self) -> f64 {
        crate::simd_transcendental::scalar_log1pexp(self)
    }
    #[inline(always)]
    fn sigmoid(self) -> f64 {
        crate::glm::sigmoid_stable(self)
    }
    #[inline(always)]
    fn probit_cdf(self) -> f64 {
        crate::simd_transcendental::phi_hp(self)
    }
    #[inline(always)]
    fn ln_gamma(self) -> f64 {
        crate::simd_transcendental::ln_gamma(self)
    }

    #[inline(always)]
    fn family_pass(
        family: Family,
        nb_theta: f64,
        eta: &mut [f64],
        y: &[f64],
        prior_w: &[f64],
        weighted: bool,
        yeta: f64,
        prob: &mut [f64],
        w: &mut [f64],
        z: &mut [f64],
    ) -> (f64, bool) {
        crate::simd_transcendental::family_pass(
            family, nb_theta, eta, y, prior_w, weighted, yeta, prob, w, z,
        )
    }

    #[inline(always)]
    fn as_f64s(s: &[f64]) -> Option<&[f64]> {
        Some(s)
    }
    #[inline(always)]
    fn as_f64s_mut(s: &mut [f64]) -> Option<&mut [f64]> {
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::Scalar;
    /// The `f64` implementation must be the identity on every method that has
    /// a plain std twin, and must bind to the SAME function the kernel called
    /// before the trait existed — that identity is what makes `T = f64`
    /// bit-identical, so it is asserted with `==`, not a tolerance.
    #[test]
    fn f64_impl_is_the_identity_binding() {
        for &v in &[-3.5_f64, -1e-9, 0.0, 0.5, 1.0, 40.0, 700.0] {
            assert_eq!(Scalar::exp(v), v.exp());
            assert_eq!(Scalar::exp_m1(v), v.exp_m1());
            assert_eq!(Scalar::sigmoid(v), crate::glm::sigmoid_stable(v));
            assert_eq!(Scalar::probit_cdf(v), crate::simd_transcendental::phi_hp(v));
            assert_eq!(Scalar::value(v), v);
            assert_eq!(<f64 as Scalar>::from_f64(v), v);
            assert_eq!(
                Scalar::log1pexp(v),
                crate::simd_transcendental::scalar_log1pexp(v)
            );
        }
        for &v in &[1e-8_f64, 0.5, 1.0, 12.0] {
            assert_eq!(Scalar::ln(v), v.ln());
            assert_eq!(Scalar::sqrt(v), v.sqrt());
            assert_eq!(Scalar::ln_gamma(v), crate::simd_transcendental::ln_gamma(v));
        }
    }

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

        fn run(
            family: Family,
            eta0: &[f64],
            y: &[f64],
            weighted: bool,
        ) -> (PassResult, PassResult) {
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
}
