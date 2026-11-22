//! The scalar type the blocked fit kernel is generic over.
//!
//! `f64` is the production instantiation. A forward dual number implements
//! the same trait.
//!
//! Control flow never depends on the derivative part: every branch in the
//! kernel — convergence bands, step-halving, Cholesky pivot signs, sparsity
//! skips — compares [`Scalar::value`], so the iterate path a fit takes is the
//! same at every `T`.

use crate::spec::{BinomialLink, Family};

/// Sealing supertrait: private, so `Scalar` cannot be implemented outside the
/// crate.
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
    /// True only for `f64`. Branches on it skip `f64`-only work (the pinned
    /// crossed-θ skip, the β-profile border); see each reader.
    const IS_F64: bool;

    /// Lift a constant. The derivative part, where one exists, is zero.
    fn from_f64(v: f64) -> Self;
    /// The value part. `f64`'s implementation is the identity.
    fn value(self) -> f64;
    /// `c·x` for a constant `c`: the value and every derivative lane times `c`.
    /// Equals `from_f64(c) * x` except that a lane stays finite where the value
    /// is not (`0·∞` is NaN in the full product) and a zero lane may differ
    /// in sign.
    fn scale(self, c: f64) -> Self;

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
    // Unweighted 0/1 Bernoulli logit (`!weighted`) keeps the fused
    // 2·(Σ log1pexp(η) − Σ y·η) identity, mirroring the f64 route's fast path
    // (which reaches it through the SIMD kernel instead).
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
    fn scale(self, c: f64) -> f64 {
        self * c
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
#[path = "scalar_tests.rs"]
mod tests;
