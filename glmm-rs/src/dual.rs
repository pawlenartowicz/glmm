//! Crate-own forward-mode dual number types.
//!
//! [`Dual`] carries a value and `N` first-derivative lanes; [`HyperDual`] carries
//! a value, `N` first-derivative lanes and the packed second-derivative block.
//! On both types the value part is the number
//! `impl Scalar for f64` computes at the same iterate, up to round-off: the
//! AGQ node sums take another formula at `f64` (`glmm::agq_fast`'s row terms,
//! where the dual types run the generic node loop), and agree with it to
//! round-off. So control flow that branches on the value takes the same path
//! at every scalar type unless a round-off difference crosses a branch edge.

/// `ψ(x) = dlnΓ/dx` for `x > 0`.
///
/// Abramowitz & Stegun 6.3.5 (recurrence `ψ(x) = ψ(x+1) − 1/x`) applied until
/// `x ≥ 14`, then 6.3.18's asymptotic series
/// `ψ(x) ≈ ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸) − 1/(132x¹⁰)`.
/// At `x ≥ 14` the first dropped term is ≈4e-16 absolute — inside the
/// closed-form test's 1e-14 band with ~25× margin. (At the often-quoted
/// `x ≥ 6` threshold it is ~1e-11 and the test fails.) That accuracy is what
/// the Gamma-GLMM derivative needs.
///
/// `family::ln_minus_digamma` keeps its own copy of this series on purpose
/// (switch point 20, not 14): it needs `ln x − ψ(x)` in the form free of
/// cancellation.
pub(crate) fn digamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 14.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    acc + x.ln()
        - 0.5 * r
        - r2 * (1.0 / 12.0
            - r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0 - r2 * (1.0 / 240.0 - r2 / 132.0))))
}

/// `ψ′(x)`, the trigamma function, for `x > 0`.
///
/// Abramowitz & Stegun 6.4.6 (recurrence `ψ′(x) = ψ′(x+1) + 1/x²`) to `x ≥ 14`,
/// then 6.4.12's asymptotic series
/// `ψ′(x) ≈ 1/x + 1/(2x²) + 1/(6x³) − 1/(30x⁵) + 1/(42x⁷) − 1/(30x⁹)`.
/// First dropped term ≈2e-14 absolute at `x ≥ 14` — ~5× margin at the
/// closed-form test's tightest point (`x = 2`, where the band's floor of 1
/// binds).
pub(crate) fn trigamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 14.0 {
        acc += 1.0 / (x * x);
        x += 1.0;
    }
    let r = 1.0 / x;
    let r2 = r * r;
    acc + r * (1.0 + 0.5 * r + r2 * (1.0 / 6.0 - r2 * (1.0 / 30.0 - r2 * (1.0 / 42.0 - r2 / 30.0))))
}

/// Forward dual number: a value and `N` first-derivative lanes.
///
/// Lane `j` carries `∂·/∂p_j` for the `j`-th entry of the parameter vector the
/// evaluation was seeded on. Unused lanes (when the model's `m = n_theta + p`
/// is smaller than `N`) stay zero and cost arithmetic, never correctness.
#[derive(Clone, Copy, Debug)]
pub struct Dual<const N: usize> {
    /// The value part — what the `f64` kernel computes at the same iterate, up
    /// to the generic `family_pass` default's rounding against the `f64` SIMD
    /// twin (the buffers and row passes are identical on both routes). Control
    /// flow branches only on this field.
    pub v: f64,
    /// First-derivative lanes, `d[j] = ∂v/∂p_j`.
    pub d: [f64; N],
}

impl<const N: usize> std::ops::Add for Dual<N> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = self.d[j] + rhs.d[j];
        }
        Dual {
            v: self.v + rhs.v,
            d,
        }
    }
}

impl<const N: usize> std::ops::Sub for Dual<N> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = self.d[j] - rhs.d[j];
        }
        Dual {
            v: self.v - rhs.v,
            d,
        }
    }
}

impl<const N: usize> std::ops::Neg for Dual<N> {
    type Output = Self;
    fn neg(self) -> Self {
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = -self.d[j];
        }
        Dual { v: -self.v, d }
    }
}

impl<const N: usize> std::ops::Mul for Dual<N> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = self.v * rhs.d[j] + self.d[j] * rhs.v;
        }
        Dual {
            v: self.v * rhs.v,
            d,
        }
    }
}

impl<const N: usize> std::ops::Div for Dual<N> {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let q = self.v / rhs.v;
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = (self.d[j] - q * rhs.d[j]) / rhs.v;
        }
        Dual { v: q, d }
    }
}

impl<const N: usize> std::ops::AddAssign for Dual<N> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> std::ops::SubAssign for Dual<N> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> std::ops::MulAssign for Dual<N> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> std::ops::DivAssign for Dual<N> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Chain rule for a unary function: `value = f(a.v)`, `lane j = fp · a.d[j]`.
/// `fp` is `f'(a.v)`, computed by the caller in plain f64 arithmetic — never
/// by differentiating the f64 twin's internals.
#[inline]
fn chain1<const N: usize>(a: Dual<N>, value: f64, fp: f64) -> Dual<N> {
    let mut d = [0.0f64; N];
    #[allow(clippy::needless_range_loop)]
    for j in 0..N {
        d[j] = fp * a.d[j];
    }
    Dual { v: value, d }
}

impl<const N: usize> crate::scalar::Scalar for Dual<N> {
    const ZERO: Self = Dual {
        v: 0.0,
        d: [0.0; N],
    };
    const ONE: Self = Dual {
        v: 1.0,
        d: [0.0; N],
    };
    const IS_F64: bool = false;

    #[inline]
    fn from_f64(v: f64) -> Self {
        Dual { v, d: [0.0; N] }
    }
    #[inline]
    fn value(self) -> f64 {
        self.v
    }
    #[inline]
    fn scale(self, c: f64) -> Self {
        let mut d = self.d;
        for dj in &mut d {
            *dj *= c;
        }
        Dual { v: self.v * c, d }
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = f64::sqrt(self.v);
        chain1(self, s, 0.5 / s)
    }
    #[inline]
    fn max_f64(self, other: f64) -> Self {
        // A clamp that bites makes the result a constant in the parameters, so
        // every derivative lane is zero. That is the derivative of the clamped
        // function, not an approximation of it.
        if self.v >= other {
            self
        } else {
            Self::from_f64(other)
        }
    }
    #[inline]
    fn clamp_f64(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            Self::from_f64(lo)
        } else if self.v > hi {
            Self::from_f64(hi)
        } else {
            self
        }
    }

    #[inline]
    fn exp(self) -> Self {
        let e = f64::exp(self.v);
        chain1(self, e, e)
    }
    #[inline]
    fn exp_m1(self) -> Self {
        // The `−1` is a constant: the derivative is `exp`, not `exp_m1`.
        chain1(self, f64::exp_m1(self.v), f64::exp(self.v))
    }
    #[inline]
    fn ln(self) -> Self {
        chain1(self, f64::ln(self.v), 1.0 / self.v)
    }
    #[inline]
    fn ln_1p(self) -> Self {
        chain1(self, f64::ln_1p(self.v), 1.0 / (1.0 + self.v))
    }
    #[inline]
    fn log1pexp(self) -> Self {
        let value = crate::simd_transcendental::scalar_log1pexp(self.v);
        let s = crate::glm::sigmoid_stable(self.v);
        chain1(self, value, s)
    }
    #[inline]
    fn sigmoid(self) -> Self {
        let s = crate::glm::sigmoid_stable(self.v);
        chain1(self, s, s * (1.0 - s))
    }
    #[inline]
    fn probit_cdf(self) -> Self {
        let value = crate::simd_transcendental::phi_hp(self.v);
        let phi = crate::family::FRAC_1_SQRT_2PI * f64::exp(-0.5 * self.v * self.v);
        chain1(self, value, phi)
    }
}

// `check` calling `assert_send_sync::<Dual<4>>()` is the compile-time
// assertion: typeck runs on every item body regardless of whether it is ever
// called, so a `Dual<4>: Send + Sync` failure would error here even though
// nothing invokes `check`. Neither function needs a caller, so both are
// legitimately unreachable — `#[allow(dead_code)]` accordingly (the crate's
// blanket `not(feature = "loop_advanced")` dead-code allow does not cover a
// `loop_advanced` build, where this would otherwise warn).
const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn check() {
        assert_send_sync::<Dual<4>>();
    }
};

/// Forward hyper-dual number: a value, `N` first-derivative lanes and the
/// packed lower triangle of the `N×N` second-derivative block.
///
/// `h[idx(i, j)]` with `i >= j` carries `∂²·/∂p_i∂p_j`; `idx(i, j) = i*(i+1)/2 + j`
/// and the block has `N*(N+1)/2` entries. Packed rather than full `N×N`: the
/// Hessian of a real function is symmetric, so the strict upper triangle is
/// redundant arithmetic on every operation — the saving is the arithmetic, not
/// the bytes.
///
/// `H` is a second const parameter carrying `N*(N+1)/2` — stable Rust cannot
/// write `[f64; N * (N + 1) / 2]` in a struct field, so the packed length is
/// passed alongside `N` and checked by a `const _` assertion in the impl
/// block below. Only the consistent pairs `(4,10)`, `(5,15)`, `(6,21)`,
/// `(8,36)`, `(12,78)` are instantiated.
#[derive(Clone, Copy, Debug)]
pub struct HyperDual<const N: usize, const H: usize> {
    /// The value part.
    pub v: f64,
    /// First-derivative lanes, `d[j] = ∂v/∂p_j`.
    pub d: [f64; N],
    /// Packed lower triangle of the second-derivative block, length `N*(N+1)/2`.
    pub h: [f64; H],
}

impl<const N: usize, const H: usize> HyperDual<N, H> {
    /// A mismatched `(N, H)` pair is caught here rather than by a packed-index
    /// out-of-bounds panic somewhere inside the arithmetic.
    const _CHECK_PACKED_LEN: () = assert!(H == N * (N + 1) / 2);
}

/// Forces `HyperDual::<N, H>::_CHECK_PACKED_LEN` to be evaluated for this
/// `(N, H)` instantiation. An associated const on a generic impl is only
/// checked when referenced, so every operator and `chain2` calls this once —
/// a mismatched pair is then a compile error, not a packed-index panic.
#[allow(clippy::let_unit_value)]
#[inline]
fn assert_packed_len<const N: usize, const H: usize>() {
    let _ = HyperDual::<N, H>::_CHECK_PACKED_LEN;
}

impl<const N: usize, const H: usize> std::ops::Add for HyperDual<N, H> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        assert_packed_len::<N, H>();
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = self.d[j] + rhs.d[j];
        }
        let mut h = [0.0f64; H];
        #[allow(clippy::needless_range_loop)]
        for k in 0..H {
            h[k] = self.h[k] + rhs.h[k];
        }
        HyperDual {
            v: self.v + rhs.v,
            d,
            h,
        }
    }
}

impl<const N: usize, const H: usize> std::ops::Sub for HyperDual<N, H> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        assert_packed_len::<N, H>();
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = self.d[j] - rhs.d[j];
        }
        let mut h = [0.0f64; H];
        #[allow(clippy::needless_range_loop)]
        for k in 0..H {
            h[k] = self.h[k] - rhs.h[k];
        }
        HyperDual {
            v: self.v - rhs.v,
            d,
            h,
        }
    }
}

impl<const N: usize, const H: usize> std::ops::Neg for HyperDual<N, H> {
    type Output = Self;
    fn neg(self) -> Self {
        assert_packed_len::<N, H>();
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = -self.d[j];
        }
        let mut h = [0.0f64; H];
        #[allow(clippy::needless_range_loop)]
        for k in 0..H {
            h[k] = -self.h[k];
        }
        HyperDual { v: -self.v, d, h }
    }
}

impl<const N: usize, const H: usize> std::ops::Mul for HyperDual<N, H> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        assert_packed_len::<N, H>();
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = self.v * rhs.d[j] + self.d[j] * rhs.v;
        }
        let mut h = [0.0f64; H];
        for i in 0..N {
            for j in 0..=i {
                let ij = i * (i + 1) / 2 + j;
                h[ij] = self.v * rhs.h[ij]
                    + self.d[i] * rhs.d[j]
                    + self.d[j] * rhs.d[i]
                    + self.h[ij] * rhs.v;
            }
        }
        HyperDual {
            v: self.v * rhs.v,
            d,
            h,
        }
    }
}

impl<const N: usize, const H: usize> std::ops::Div for HyperDual<N, H> {
    type Output = Self;
    // Direct quotient rule in one pass over the packed `h` triangle,
    // mirroring `Dual::div`: from `a = f·b`, `f_j = (a_j − f·b_j)/b.v` and
    // `f_ij = (a_ij − f·b_ij − f_i·b_j − f_j·b_i)/b.v`. One pass, where
    // `a * recip(b)` would take two full second-order passes; the two differ
    // only in rounding.
    fn div(self, rhs: Self) -> Self {
        assert_packed_len::<N, H>();
        let q = self.v / rhs.v;
        let mut d = [0.0f64; N];
        #[allow(clippy::needless_range_loop)]
        for j in 0..N {
            d[j] = (self.d[j] - q * rhs.d[j]) / rhs.v;
        }
        let mut h = [0.0f64; H];
        for i in 0..N {
            for j in 0..=i {
                let ij = i * (i + 1) / 2 + j;
                h[ij] = (self.h[ij] - q * rhs.h[ij] - d[i] * rhs.d[j] - d[j] * rhs.d[i]) / rhs.v;
            }
        }
        HyperDual { v: q, d, h }
    }
}

impl<const N: usize, const H: usize> std::ops::AddAssign for HyperDual<N, H> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize, const H: usize> std::ops::SubAssign for HyperDual<N, H> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize, const H: usize> std::ops::MulAssign for HyperDual<N, H> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize, const H: usize> std::ops::DivAssign for HyperDual<N, H> {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

/// Chain rule for a unary function to second order:
///   value      = f(a.v)
///   lane j     = f'·a.d[j]
///   packed ij  = f'·a.h[ij] + f''·a.d[i]·a.d[j]
/// `fp` and `fpp` are `f'(a.v)` and `f''(a.v)` in plain f64 arithmetic.
fn chain2<const N: usize, const H: usize>(
    a: HyperDual<N, H>,
    value: f64,
    fp: f64,
    fpp: f64,
) -> HyperDual<N, H> {
    assert_packed_len::<N, H>();
    let mut d = [0.0f64; N];
    let mut h = [0.0f64; H];
    #[allow(clippy::needless_range_loop)]
    for j in 0..N {
        d[j] = fp * a.d[j];
    }
    for i in 0..N {
        for j in 0..=i {
            let ij = i * (i + 1) / 2 + j;
            h[ij] = fp * a.h[ij] + fpp * a.d[i] * a.d[j];
        }
    }
    HyperDual { v: value, d, h }
}

// See the twin block above `Dual`'s `impl Scalar` for why `#[allow(dead_code)]`
// is correct here rather than a missing caller.
const _: () = {
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}
    #[allow(dead_code)]
    fn check() {
        assert_send_sync::<HyperDual<4, 10>>();
    }
};

impl<const N: usize, const H: usize> crate::scalar::Scalar for HyperDual<N, H> {
    const ZERO: Self = HyperDual {
        v: 0.0,
        d: [0.0; N],
        h: [0.0; H],
    };
    const ONE: Self = HyperDual {
        v: 1.0,
        d: [0.0; N],
        h: [0.0; H],
    };
    const IS_F64: bool = false;

    #[inline]
    fn from_f64(v: f64) -> Self {
        HyperDual {
            v,
            d: [0.0; N],
            h: [0.0; H],
        }
    }
    #[inline]
    fn value(self) -> f64 {
        self.v
    }
    #[inline]
    fn scale(self, c: f64) -> Self {
        let (mut d, mut h) = (self.d, self.h);
        for dj in &mut d {
            *dj *= c;
        }
        for hk in &mut h {
            *hk *= c;
        }
        HyperDual {
            v: self.v * c,
            d,
            h,
        }
    }

    #[inline]
    fn sqrt(self) -> Self {
        let s = f64::sqrt(self.v);
        chain2(self, s, 0.5 / s, -0.25 / (s * self.v))
    }
    #[inline]
    fn max_f64(self, other: f64) -> Self {
        // A clamp that bites makes the result a constant in the parameters, so
        // every derivative lane — first AND second order — is zero. That is
        // the derivative of the clamped function, not an approximation of it.
        if self.v >= other {
            self
        } else {
            Self::from_f64(other)
        }
    }
    #[inline]
    fn clamp_f64(self, lo: f64, hi: f64) -> Self {
        if self.v < lo {
            Self::from_f64(lo)
        } else if self.v > hi {
            Self::from_f64(hi)
        } else {
            self
        }
    }

    #[inline]
    fn exp(self) -> Self {
        let e = f64::exp(self.v);
        chain2(self, e, e, e)
    }
    #[inline]
    fn exp_m1(self) -> Self {
        // The `−1` is a constant: both derivatives are `exp`, not `exp_m1`.
        let e = f64::exp(self.v);
        chain2(self, f64::exp_m1(self.v), e, e)
    }
    #[inline]
    fn ln(self) -> Self {
        chain2(
            self,
            f64::ln(self.v),
            1.0 / self.v,
            -1.0 / (self.v * self.v),
        )
    }
    #[inline]
    fn ln_1p(self) -> Self {
        let d = 1.0 / (1.0 + self.v);
        chain2(self, f64::ln_1p(self.v), d, -d * d)
    }
    #[inline]
    fn log1pexp(self) -> Self {
        let value = crate::simd_transcendental::scalar_log1pexp(self.v);
        let s = crate::glm::sigmoid_stable(self.v);
        chain2(self, value, s, s * (1.0 - s))
    }
    #[inline]
    fn sigmoid(self) -> Self {
        let s = crate::glm::sigmoid_stable(self.v);
        chain2(self, s, s * (1.0 - s), s * (1.0 - s) * (1.0 - 2.0 * s))
    }
    #[inline]
    fn probit_cdf(self) -> Self {
        let value = crate::simd_transcendental::phi_hp(self.v);
        let phi = crate::family::FRAC_1_SQRT_2PI * f64::exp(-0.5 * self.v * self.v);
        chain2(self, value, phi, -self.v * phi)
    }
}

#[cfg(test)]
#[path = "dual_tests.rs"]
mod tests;
