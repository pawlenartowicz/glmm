//! Crate-own forward-mode dual number types.
//!
//! [`Dual`] carries a value and `N` first-derivative lanes; [`HyperDual`] carries
//! a value, `N` first-derivative lanes and the packed second-derivative block.
//! On both types the value part is always the
//! `f64` binding — the same number W1's `impl Scalar for f64` would compute at
//! the same iterate — so control flow that branches on the value takes the
//! same path at every scalar type.

/// `ψ(x) = dlnΓ/dx` for `x > 0`.
///
/// Abramowitz & Stegun 6.3.5 (recurrence `ψ(x) = ψ(x+1) − 1/x`) applied until
/// `x ≥ 14`, then 6.3.18's asymptotic series
/// `ψ(x) ≈ ln x − 1/(2x) − 1/(12x²) + 1/(120x⁴) − 1/(252x⁶) + 1/(240x⁸) − 1/(132x¹⁰)`.
/// At `x ≥ 14` the first dropped term is ≈4e-16 absolute — inside the
/// closed-form test's 1e-14 band with ~25× margin. (At the often-quoted
/// `x ≥ 6` threshold it is ~1e-11 and the test fails.) That accuracy is what
/// the Gamma-GLMM derivative needs.
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
    fn abs(self) -> Self {
        let fp = if self.v == 0.0 { 0.0 } else { self.v.signum() };
        chain1(self, f64::abs(self.v), fp)
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
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
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
    #[inline]
    fn ln_gamma(self) -> Self {
        let value = crate::simd_transcendental::ln_gamma(self.v);
        chain1(self, value, digamma(self.v))
    }

    fn chol_lower(a: &[Self], dim: usize, l_out: &mut [Self]) -> bool {
        crate::scalar::chol_lower_generic(a, dim, l_out)
    }

    // Write `bt = B0 + Σ_k ε_k B_k`, where `B0` is the value lanes of `bt` and
    // `B_k` is derivative lane `k`. Since `ε_k ε_l = 0`,
    //   bt · btᵀ = B0 B0ᵀ + Σ_k ε_k (B_k B0ᵀ + B0 B_kᵀ).
    // The value part of the result is exactly `B0 B0ᵀ`, so it goes through the
    // same faer call `f64` uses (`tri_lower_sub_gemm(_, _, B0, B0, _)`) and
    // keeps `f64`'s rounding. Each derivative lane is the sum of two
    // rectangular gemms, `B_k B0ᵀ` and `B0 B_kᵀ`, added in that order — the
    // generic triple loop instead accumulates value and derivative terms
    // together inside one `k`-sum, so the two routes reorder floating-point
    // addition relative to each other. That reordering only ever shows up in
    // the last ulp (checked by `dual_syrk_faer_lanes_match_generic_loop`
    // below), never in a validated answer. `HyperDual` keeps the generic
    // loop: unpacking its packed second-derivative block into rectangular
    // buffers would cost more than the loop it replaces.
    fn syrk_lower_sub(bt: &[Self], t_dim: usize, w_tot: usize, tail: &mut [Self]) {
        // One allocation per call, on the gradient path (once per fit), not
        // the fit's inner loop — accepted here, unlike the alloc-free kernel
        // paths this crate otherwise holds to.
        let mut b0 = vec![0.0_f64; t_dim * w_tot];
        let mut bk = vec![0.0_f64; t_dim * w_tot];
        let mut scratch = vec![0.0_f64; t_dim * t_dim];

        for (k, dst) in b0.iter_mut().enumerate() {
            *dst = bt[k].v;
        }
        for j in 0..t_dim {
            for i in j..t_dim {
                scratch[j * t_dim + i] = tail[j * t_dim + i].v;
            }
        }
        crate::scalar::tri_lower_sub_gemm(&mut scratch, t_dim, &b0, &b0, w_tot);
        for j in 0..t_dim {
            for i in j..t_dim {
                tail[j * t_dim + i].v = scratch[j * t_dim + i];
            }
        }

        for lane in 0..N {
            for (k, dst) in bk.iter_mut().enumerate() {
                *dst = bt[k].d[lane];
            }
            for j in 0..t_dim {
                for i in j..t_dim {
                    scratch[j * t_dim + i] = tail[j * t_dim + i].d[lane];
                }
            }
            crate::scalar::tri_lower_sub_gemm(&mut scratch, t_dim, &bk, &b0, w_tot);
            crate::scalar::tri_lower_sub_gemm(&mut scratch, t_dim, &b0, &bk, w_tot);
            for j in 0..t_dim {
                for i in j..t_dim {
                    tail[j * t_dim + i].d[lane] = scratch[j * t_dim + i];
                }
            }
        }
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
/// block below. Only the consistent pairs `(4,10)`, `(8,36)`, `(12,78)` are
/// instantiated.
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
    // `f_ij = (a_ij − f·b_ij − f_i·b_j − f_j·b_i)/b.v`. Replaces an earlier
    // `a * recip(b)` form (two full second-order passes); results differ
    // from it only in rounding.
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

/// `1/b` as a hyper-dual: `f(x) = 1/x`, `f' = −1/x²`, `f'' = 2/x³`, fed through
/// the general unary chain rule below. No non-test caller since `Div` moved to
/// a direct quotient rule; kept as the hand-checked reference the recip test
/// pins.
#[allow(dead_code)]
fn recip<const N: usize, const H: usize>(b: HyperDual<N, H>) -> HyperDual<N, H> {
    let r = 1.0 / b.v;
    chain2(b, r, -r * r, 2.0 * r * r * r)
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
    fn abs(self) -> Self {
        let fp = if self.v == 0.0 { 0.0 } else { self.v.signum() };
        chain2(self, f64::abs(self.v), fp, 0.0)
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
    fn mul_add(self, a: Self, b: Self) -> Self {
        self * a + b
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
    #[inline]
    fn ln_gamma(self) -> Self {
        let value = crate::simd_transcendental::ln_gamma(self.v);
        chain2(self, value, digamma(self.v), trigamma(self.v))
    }

    fn chol_lower(a: &[Self], dim: usize, l_out: &mut [Self]) -> bool {
        crate::scalar::chol_lower_generic(a, dim, l_out)
    }

    fn syrk_lower_sub(bt: &[Self], t_dim: usize, w_tot: usize, tail: &mut [Self]) {
        crate::scalar::syrk_lower_sub_generic(bt, t_dim, w_tot, tail)
    }
}

#[cfg(test)]
mod tests {
    use super::{digamma, trigamma, Dual, HyperDual};
    use crate::scalar::{chol_lower_generic, syrk_lower_sub_generic, Scalar};

    /// Closed forms, not a reimplementation: ψ(1) = −γ, ψ(½) = −γ − 2ln2,
    /// ψ(2) = 1 − γ (A&S 6.3.2–6.3.4); ψ′(1) = π²/6, ψ′(½) = π²/2,
    /// ψ′(2) = π²/6 − 1 (A&S 6.4.3–6.4.5). Every point here goes through the
    /// recurrence to `x ≥ 14`; `digamma_is_the_derivative_of_ln_gamma`'s
    /// `x = 30` reaches the series directly.
    #[test]
    fn digamma_trigamma_match_closed_forms() {
        const EULER: f64 = 0.577_215_664_901_532_9;
        let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        for (x, want) in [
            (1.0, -EULER),
            (0.5, -EULER - 2.0 * std::f64::consts::LN_2),
            (2.0, 1.0 - EULER),
            (10.0, 2.251_752_589_066_721),
        ] {
            assert!(
                (digamma(x) - want).abs() <= 1e-14 * want.abs().max(1.0),
                "psi({x})"
            );
        }
        for (x, want) in [
            (1.0, pi2_6),
            (0.5, 3.0 * pi2_6),
            (2.0, pi2_6 - 1.0),
            (10.0, 0.105_166_335_681_685_74),
        ] {
            assert!(
                (trigamma(x) - want).abs() <= 1e-13 * want.abs().max(1.0),
                "psi'({x})"
            );
        }
    }

    /// ψ is the derivative of the crate's own `ln_gamma` — central FD against
    /// the shipped Lanczos series, so the two agree as a pair and not just
    /// against a table. Band is the FD floor at h = 1e-5, not the series'.
    #[test]
    fn digamma_is_the_derivative_of_ln_gamma() {
        use crate::simd_transcendental::ln_gamma;
        for &x in &[0.3_f64, 1.0, 2.5, 7.0, 30.0] {
            let h = 1e-5 * x;
            let fd = (ln_gamma(x + h) - ln_gamma(x - h)) / (2.0 * h);
            assert!(
                (digamma(x) - fd).abs() <= 1e-8 * fd.abs().max(1.0),
                "x = {x}"
            );
        }
    }

    /// f(a, b) = (a·b + a) / (b − a) at (a, b) = (2, 5), lanes seeded on a and b.
    /// ∂f/∂a = ((b + 1)(b − a) + (ab + a)) / (b − a)² = (6·3 + 12)/9 = 10/3
    /// ∂f/∂b = (a(b − a) − (ab + a)) / (b − a)² = (6 − 12)/9 = −2/3
    #[test]
    fn dual_arithmetic_matches_hand_partials() {
        let a = Dual::<2> {
            v: 2.0,
            d: [1.0, 0.0],
        };
        let b = Dual::<2> {
            v: 5.0,
            d: [0.0, 1.0],
        };
        let f = (a * b + a) / (b - a);
        assert!((f.v - 4.0).abs() < 1e-15);
        assert!((f.d[0] - 10.0 / 3.0).abs() < 1e-14);
        assert!((f.d[1] + 2.0 / 3.0).abs() < 1e-14);
    }

    /// `AddAssign`/`SubAssign`/`MulAssign`/`DivAssign` are written as
    /// `*self = *self <op> rhs`; check each agrees with its non-assigning
    /// twin on the same operands.
    #[test]
    fn assign_ops_match_non_assign_twins() {
        let a = Dual::<2> {
            v: 2.0,
            d: [1.0, 0.3],
        };
        let b = Dual::<2> {
            v: 5.0,
            d: [0.7, 1.0],
        };

        let mut add = a;
        add += b;
        assert_eq!(add.v, (a + b).v);
        assert_eq!(add.d, (a + b).d);

        let mut sub = a;
        sub -= b;
        assert_eq!(sub.v, (a - b).v);
        assert_eq!(sub.d, (a - b).d);

        let mut mul = a;
        mul *= b;
        assert_eq!(mul.v, (a * b).v);
        assert_eq!(mul.d, (a * b).d);

        let mut div = a;
        div /= b;
        assert_eq!(div.v, (a / b).v);
        assert_eq!(div.d, (a / b).d);
    }

    /// A constant lifted with `from_f64` has all-zero derivative lanes.
    #[test]
    fn from_f64_gives_all_zero_lanes() {
        fn from_f64<const N: usize>(v: f64) -> Dual<N> {
            Dual { v, d: [0.0; N] }
        }
        let x: Dual<3> = from_f64(7.0);
        assert_eq!(x.v, 7.0);
        assert_eq!(x.d, [0.0, 0.0, 0.0]);
    }

    /// Every `Scalar` method's derivative lane against a central difference of the
    /// same method at f64 — the twin the value part binds to, so a mismatch is a
    /// chain-rule error and not a difference between two implementations of the
    /// same function. h = 1e-6 relative puts the FD truncation and round-off floor
    /// near 1e-10, so the band is 1e-8. No point below 0.25: `h` is absolute
    /// (1e-6) there, and the central-FD truncation `(h²/6)·f'''` for `ln`, `sqrt`
    /// and `ln_gamma` at x ≤ 1e-3 exceeds the bands by orders.
    // `fd(|t| Scalar::method(t), x)` reads `Scalar::method` as a plain
    // function reference and clippy flags each as a redundant closure, so
    // the lint is silenced here rather than rewriting every closure.
    #[allow(clippy::redundant_closure)]
    #[test]
    fn dual_chain_rules_match_central_fd_of_the_f64_twin() {
        fn fd(f: impl Fn(f64) -> f64, x: f64) -> f64 {
            let h = 1e-6 * x.abs().max(1.0);
            (f(x + h) - f(x - h)) / (2.0 * h)
        }
        let pts_all = [-3.5_f64, -0.7, 0.0, 0.4, 1.0, 4.0];
        let pts_pos = [0.25_f64, 1.0, 3.0, 12.0];

        for &x in &pts_all {
            let a = Dual::<1> { v: x, d: [1.0] };
            assert!(
                (Scalar::exp(a).d[0] - fd(|t| Scalar::exp(t), x)).abs()
                    <= 1e-8 * fd(|t| Scalar::exp(t), x).abs().max(1.0)
            );
            assert!(
                (Scalar::exp_m1(a).d[0] - fd(|t| Scalar::exp_m1(t), x)).abs()
                    <= 1e-8 * fd(|t| Scalar::exp_m1(t), x).abs().max(1.0)
            );
            assert!((Scalar::sigmoid(a).d[0] - fd(|t| Scalar::sigmoid(t), x)).abs() <= 1e-8);
            assert!((Scalar::log1pexp(a).d[0] - fd(|t| Scalar::log1pexp(t), x)).abs() <= 1e-8);
            assert!((Scalar::probit_cdf(a).d[0] - fd(|t| Scalar::probit_cdf(t), x)).abs() <= 1e-8);
            // Value part is the f64 binding, exactly — assert with `==`.
            assert_eq!(Scalar::exp(a).v, Scalar::exp(x));
            assert_eq!(Scalar::exp_m1(a).v, Scalar::exp_m1(x));
            assert_eq!(Scalar::sigmoid(a).v, Scalar::sigmoid(x));
            assert_eq!(Scalar::log1pexp(a).v, Scalar::log1pexp(x));
            assert_eq!(Scalar::probit_cdf(a).v, Scalar::probit_cdf(x));
        }
        for &x in &pts_pos {
            let a = Dual::<1> { v: x, d: [1.0] };
            assert!((Scalar::ln(a).d[0] - fd(|t| Scalar::ln(t), x)).abs() <= 1e-8 * (1.0 / x));
            assert!((Scalar::sqrt(a).d[0] - fd(|t| Scalar::sqrt(t), x)).abs() <= 1e-8);
            assert!((Scalar::ln_gamma(a).d[0] - fd(|t| Scalar::ln_gamma(t), x)).abs() <= 1e-7);
            assert_eq!(Scalar::ln(a).v, Scalar::ln(x));
            assert_eq!(Scalar::sqrt(a).v, Scalar::sqrt(x));
            assert_eq!(Scalar::ln_gamma(a).v, Scalar::ln_gamma(x));
        }
    }

    /// `clamp_f64` and `max_f64` zero every derivative lane when the bound
    /// bites, and pass every lane through unchanged when it does not. The
    /// value part matches `f64::clamp`/`f64::max` in both arms.
    #[test]
    fn clamp_and_max_zero_lanes_only_when_the_bound_bites() {
        let a = Dual::<2> {
            v: 5.0,
            d: [1.0, -2.0],
        };

        // max_f64: bound does not bite (self.v >= other) — lanes pass through.
        let m = Scalar::max_f64(a, 1.0);
        assert_eq!(m.v, f64::max(5.0, 1.0));
        assert_eq!(m.d, [1.0, -2.0]);

        // max_f64: bound bites (self.v < other) — lanes zero.
        let m = Scalar::max_f64(a, 9.0);
        assert_eq!(m.v, f64::max(5.0, 9.0));
        assert_eq!(m.d, [0.0, 0.0]);

        // clamp_f64: inside [lo, hi] — lanes pass through.
        let c = Scalar::clamp_f64(a, 0.0, 10.0);
        assert_eq!(c.v, f64::clamp(5.0, 0.0, 10.0));
        assert_eq!(c.d, [1.0, -2.0]);

        // clamp_f64: below lo — lanes zero.
        let c = Scalar::clamp_f64(a, 6.0, 10.0);
        assert_eq!(c.v, f64::clamp(5.0, 6.0, 10.0));
        assert_eq!(c.d, [0.0, 0.0]);

        // clamp_f64: above hi — lanes zero.
        let c = Scalar::clamp_f64(a, 0.0, 4.0);
        assert_eq!(c.v, f64::clamp(5.0, 0.0, 4.0));
        assert_eq!(c.d, [0.0, 0.0]);
    }

    /// `chol_lower_generic` at `T = f64`: factor a small SPD matrix and check
    /// `L·Lᵀ` reassembles the input. Column-major indexing (`j*dim + i`) is
    /// exercised independently of any dual arithmetic here.
    #[test]
    fn chol_lower_generic_f64_reassembles_the_input() {
        // SPD, column-major, dim = 3.
        let dim = 3;
        let a: [f64; 9] = [
            4.0, 12.0, -16.0, //
            12.0, 37.0, -43.0, //
            -16.0, -43.0, 98.0,
        ];
        let mut l = [0.0f64; 9];
        assert!(chol_lower_generic(&a, dim, &mut l));

        for i in 0..dim {
            for j in 0..dim {
                let mut s = 0.0;
                for k in 0..dim {
                    // l is lower-triangular; l[k*dim + r] is L[r, k].
                    s += l[k * dim + i] * l[k * dim + j];
                }
                assert!(
                    (s - a[j * dim + i]).abs() <= 1e-12 * a[j * dim + i].abs().max(1.0),
                    "({i},{j}): {s} vs {}",
                    a[j * dim + i]
                );
            }
        }
    }

    /// The same SPD matrix as `Dual<2>`, lanes seeded on entries `a[0,0]` and
    /// `a[2,2]` (the diagonal, so a small perturbation stays SPD). Each lane
    /// of `chol_lower_generic`'s output is checked against a central FD of
    /// the `f64` factorization perturbed on the same entry.
    #[test]
    fn chol_lower_generic_dual_lanes_match_central_fd() {
        let dim = 3;
        let base: [f64; 9] = [
            4.0, 12.0, -16.0, //
            12.0, 37.0, -43.0, //
            -16.0, -43.0, 98.0,
        ];

        // f64 factorization at a perturbed (0,0) or (2,2) entry.
        let factor_f64 = |a: &[f64; 9]| -> [f64; 9] {
            let mut l = [0.0f64; 9];
            assert!(chol_lower_generic(a, dim, &mut l));
            l
        };

        let mut a_dual = [Dual::<2> {
            v: 0.0,
            d: [0.0, 0.0],
        }; 9];
        for (idx, &v) in base.iter().enumerate() {
            a_dual[idx] = Dual { v, d: [0.0, 0.0] };
        }
        a_dual[0].d[0] = 1.0; // lane 0 seeded on a[0,0]
        a_dual[8].d[1] = 1.0; // lane 1 seeded on a[2,2]

        let mut l_dual = [Dual::<2> {
            v: 0.0,
            d: [0.0, 0.0],
        }; 9];
        assert!(chol_lower_generic(&a_dual, dim, &mut l_dual));

        for (lane, entry) in [(0usize, 0usize), (1usize, 8usize)] {
            let h = 1e-6;
            let mut a_plus = base;
            let mut a_minus = base;
            a_plus[entry] += h;
            a_minus[entry] -= h;
            let l_plus = factor_f64(&a_plus);
            let l_minus = factor_f64(&a_minus);
            for k in 0..9 {
                let fd = (l_plus[k] - l_minus[k]) / (2.0 * h);
                assert!(
                    (l_dual[k].d[lane] - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                    "lane {lane}, entry {k}: {} vs fd {fd}",
                    l_dual[k].d[lane]
                );
            }
        }
    }

    /// `syrk_lower_sub_generic` at `T = f64` against a hand-written triple
    /// loop on a `4×3` `bt`, and the strict upper triangle of `tail` is left
    /// untouched (sentinel values survive).
    #[test]
    fn syrk_lower_sub_generic_matches_hand_triple_loop() {
        let t_dim = 4;
        let w_tot = 3;
        // bt, column-major t_dim x w_tot.
        let bt: [f64; 12] = [
            1.0, 2.0, 3.0, 4.0, // column 0
            0.5, -1.0, 2.0, 1.5, // column 1
            -0.5, 0.25, 1.0, -2.0, // column 2
        ];

        const SENTINEL: f64 = -999.0;
        let mut tail = [SENTINEL; 16]; // 4x4, column-major

        syrk_lower_sub_generic(&bt, t_dim, w_tot, &mut tail);

        // Hand triple loop, same formula, independent implementation.
        let mut want = [SENTINEL; 16];
        for j in 0..t_dim {
            for i in j..t_dim {
                let mut s = 0.0;
                for k in 0..w_tot {
                    s += bt[k * t_dim + i] * bt[k * t_dim + j];
                }
                want[j * t_dim + i] = SENTINEL - s;
            }
        }

        for j in 0..t_dim {
            for i in 0..t_dim {
                let idx = j * t_dim + i;
                if i >= j {
                    assert!(
                        (tail[idx] - want[idx]).abs() <= 1e-12,
                        "({i},{j}): {} vs {}",
                        tail[idx],
                        want[idx]
                    );
                } else {
                    // Strict upper triangle: untouched sentinel.
                    assert_eq!(tail[idx], SENTINEL, "({i},{j}) was touched");
                }
            }
        }
    }

    /// Small deterministic LCG, `x_{n+1} = (1103515245 x_n + 12345) mod 2^31`,
    /// mapped to `[-1, 1]` — enough spread to exercise every lane without
    /// pulling in an RNG crate.
    fn lcg_next(state: &mut u64) -> f64 {
        *state = (state.wrapping_mul(1_103_515_245).wrapping_add(12_345)) % (1u64 << 31);
        (*state as f64 / (1u64 << 31) as f64) * 2.0 - 1.0
    }

    /// `<Dual<4> as Scalar>::syrk_lower_sub`'s faer-per-lane path against
    /// `syrk_lower_sub_generic`'s triple loop, on a `7×11` `bt` with all four
    /// lanes nonzero. Checks the value lane and every derivative lane agree
    /// on the lower triangle (the reordering the override introduces is a
    /// last-ulp effect, not a correctness gap) and that the strict upper
    /// triangle of `tail` is untouched.
    #[test]
    fn dual_syrk_faer_lanes_match_generic_loop() {
        for (t_dim, w_tot) in [(7usize, 11usize), (1usize, 1usize)] {
            let mut state = 12_345_u64;
            let mut bt = vec![
                Dual::<4> {
                    v: 0.0,
                    d: [0.0; 4]
                };
                t_dim * w_tot
            ];
            for e in bt.iter_mut() {
                e.v = lcg_next(&mut state);
                for lane in 0..4 {
                    e.d[lane] = lcg_next(&mut state);
                }
            }
            const SENTINEL: f64 = -777.0;
            let seed = Dual::<4> {
                v: SENTINEL,
                d: [SENTINEL; 4],
            };
            let mut tail_faer = vec![seed; t_dim * t_dim];
            for j in 0..t_dim {
                for i in j..t_dim {
                    let mut v = Dual::<4> {
                        v: lcg_next(&mut state),
                        d: [0.0; 4],
                    };
                    for lane in 0..4 {
                        v.d[lane] = lcg_next(&mut state);
                    }
                    tail_faer[j * t_dim + i] = v;
                }
            }
            let mut tail_generic = tail_faer.clone();

            <Dual<4> as Scalar>::syrk_lower_sub(&bt, t_dim, w_tot, &mut tail_faer);
            syrk_lower_sub_generic(&bt, t_dim, w_tot, &mut tail_generic);

            for j in 0..t_dim {
                for i in 0..t_dim {
                    let idx = j * t_dim + i;
                    if i >= j {
                        let a = tail_faer[idx];
                        let b = tail_generic[idx];
                        assert!(
                            (a.v - b.v).abs() <= 1e-12 * a.v.abs().max(1.0),
                            "t_dim={t_dim} w_tot={w_tot} ({i},{j}) value: {} vs {}",
                            a.v,
                            b.v
                        );
                        for lane in 0..4 {
                            assert!(
                                (a.d[lane] - b.d[lane]).abs() <= 1e-12 * a.d[lane].abs().max(1.0),
                                "t_dim={t_dim} w_tot={w_tot} ({i},{j}) lane {lane}: {} vs {}",
                                a.d[lane],
                                b.d[lane]
                            );
                        }
                    } else {
                        assert_eq!(
                            tail_faer[idx].v, SENTINEL,
                            "t_dim={t_dim} w_tot={w_tot} ({i},{j}) value was touched"
                        );
                        assert_eq!(
                            tail_faer[idx].d, [SENTINEL; 4],
                            "t_dim={t_dim} w_tot={w_tot} ({i},{j}) derivative was touched"
                        );
                    }
                }
            }
        }
    }

    /// f(a, b) = a²·b at (a, b) = (3, 2), lanes on a and b.
    /// ∂f/∂a = 2ab = 12, ∂f/∂b = a² = 9,
    /// ∂²f/∂a² = 2b = 4, ∂²f/∂a∂b = 2a = 6, ∂²f/∂b² = 0.
    #[test]
    fn hyperdual_arithmetic_matches_hand_second_partials() {
        let a = HyperDual::<2, 3> {
            v: 3.0,
            d: [1.0, 0.0],
            h: [0.0; 3],
        };
        let b = HyperDual::<2, 3> {
            v: 2.0,
            d: [0.0, 1.0],
            h: [0.0; 3],
        };
        let f = a * a * b;
        assert!((f.v - 18.0).abs() < 1e-14);
        assert!((f.d[0] - 12.0).abs() < 1e-14 && (f.d[1] - 9.0).abs() < 1e-14);
        // packed: idx(0,0)=0, idx(1,0)=1, idx(1,1)=2
        assert!((f.h[0] - 4.0).abs() < 1e-14);
        assert!((f.h[1] - 6.0).abs() < 1e-14);
        assert!(f.h[2].abs() < 1e-14);
    }

    /// `recip`, `f = 1/x` at `x = 4`: value `0.25`, first `−1/16`, second `2/64`.
    #[test]
    fn hyperdual_recip_matches_hand_second_partials() {
        let x = super::HyperDual::<1, 1> {
            v: 4.0,
            d: [1.0],
            h: [0.0],
        };
        let r = super::recip(x);
        assert!((r.v - 0.25).abs() < 1e-14);
        assert!((r.d[0] - (-1.0 / 16.0)).abs() < 1e-14);
        assert!((r.h[0] - 2.0 / 64.0).abs() < 1e-14);
    }

    /// `a / b * b == a` to `1e-13` in every lane (value, first, second),
    /// checking division round-trips.
    #[test]
    fn hyperdual_division_round_trips() {
        let a = HyperDual::<2, 3> {
            v: 3.0,
            d: [1.0, 0.5],
            h: [0.2, -0.3, 0.1],
        };
        let b = HyperDual::<2, 3> {
            v: 2.0,
            d: [0.3, 1.0],
            h: [-0.1, 0.4, 0.2],
        };
        let round_trip = a / b * b;
        assert!((round_trip.v - a.v).abs() < 1e-13);
        for j in 0..2 {
            assert!((round_trip.d[j] - a.d[j]).abs() < 1e-13, "lane {j}");
        }
        for k in 0..3 {
            assert!((round_trip.h[k] - a.h[k]).abs() < 1e-13, "packed {k}");
        }
    }

    /// f'' from HyperDual against a central difference of f' from Dual, at the same
    /// points the chain-rule FD test above uses. Differencing the analytic first
    /// derivative, not the value, so the FD floor is ~1e-10 rather than ~1e-5.
    // `Scalar::$m(t)` inside `fd1`'s closure reads `Scalar::method` as a plain
    // function reference and clippy flags it as a redundant closure —
    // silenced here rather than rewriting the closure.
    #[allow(clippy::redundant_closure)]
    #[test]
    fn hyperdual_second_derivatives_match_fd_of_the_dual_first() {
        fn fd1(f: impl Fn(f64) -> f64, x: f64) -> f64 {
            let h = 1e-6 * x.abs().max(1.0);
            (f(x + h) - f(x - h)) / (2.0 * h)
        }
        macro_rules! check {
            ($m:ident, $x:expr, $tol:expr) => {{
                let x: f64 = $x;
                let a = HyperDual::<1, 1> {
                    v: x,
                    d: [1.0],
                    h: [0.0],
                };
                let want = fd1(|t| Scalar::$m(Dual::<1> { v: t, d: [1.0] }).d[0], x);
                assert!(
                    (Scalar::$m(a).h[0] - want).abs() <= $tol * want.abs().max(1.0),
                    concat!(stringify!($m), " at {}"),
                    x
                );
                assert_eq!(Scalar::$m(a).v, Scalar::$m(x));
                assert_eq!(
                    Scalar::$m(a).d[0],
                    Scalar::$m(Dual::<1> { v: x, d: [1.0] }).d[0]
                );
            }};
        }
        for &x in &[-3.5_f64, -0.7, 0.4, 1.0, 4.0] {
            check!(exp, x, 1e-7);
            check!(exp_m1, x, 1e-7);
            check!(sigmoid, x, 1e-7);
            check!(log1pexp, x, 1e-7);
            check!(probit_cdf, x, 1e-7);
        }
        for &x in &[0.25_f64, 1.0, 3.0, 12.0] {
            check!(ln, x, 1e-7);
            check!(sqrt, x, 1e-7);
            check!(ln_gamma, x, 1e-6);
        }
    }

    /// `clamp_f64` and `max_f64` zero every derivative lane, first AND second
    /// order, when the bound bites — and pass every lane through unchanged
    /// when it does not.
    #[test]
    fn hyperdual_clamp_and_max_zero_both_lane_arrays_only_when_the_bound_bites() {
        let a = HyperDual::<2, 3> {
            v: 5.0,
            d: [1.0, -2.0],
            h: [0.5, -0.25, 0.1],
        };

        // max_f64: bound does not bite (self.v >= other) — lanes pass through.
        let m = Scalar::max_f64(a, 1.0);
        assert_eq!(m.v, f64::max(5.0, 1.0));
        assert_eq!(m.d, [1.0, -2.0]);
        assert_eq!(m.h, [0.5, -0.25, 0.1]);

        // max_f64: bound bites (self.v < other) — both lane arrays zero.
        let m = Scalar::max_f64(a, 9.0);
        assert_eq!(m.v, f64::max(5.0, 9.0));
        assert_eq!(m.d, [0.0, 0.0]);
        assert_eq!(m.h, [0.0, 0.0, 0.0]);

        // clamp_f64: inside [lo, hi] — lanes pass through.
        let c = Scalar::clamp_f64(a, 0.0, 10.0);
        assert_eq!(c.v, f64::clamp(5.0, 0.0, 10.0));
        assert_eq!(c.d, [1.0, -2.0]);
        assert_eq!(c.h, [0.5, -0.25, 0.1]);

        // clamp_f64: below lo — both lane arrays zero.
        let c = Scalar::clamp_f64(a, 6.0, 10.0);
        assert_eq!(c.v, f64::clamp(5.0, 6.0, 10.0));
        assert_eq!(c.d, [0.0, 0.0]);
        assert_eq!(c.h, [0.0, 0.0, 0.0]);

        // clamp_f64: above hi — both lane arrays zero.
        let c = Scalar::clamp_f64(a, 0.0, 4.0);
        assert_eq!(c.v, f64::clamp(5.0, 0.0, 4.0));
        assert_eq!(c.d, [0.0, 0.0]);
        assert_eq!(c.h, [0.0, 0.0, 0.0]);
    }

    /// `chol_lower_generic` at `HyperDual<2, 3>`: same SPD matrix and lane
    /// seeding as `chol_lower_generic_dual_lanes_match_central_fd`, but the
    /// second-order lanes are checked against a central FD of the `Dual`
    /// factorization's first-order lanes (mirroring the FD-of-the-first-
    /// derivative technique above, applied to the batched primitive instead
    /// of an elementary function).
    #[test]
    fn chol_lower_generic_hyperdual_second_lanes_match_fd_of_dual_first() {
        let dim = 3;
        let base: [f64; 9] = [
            4.0, 12.0, -16.0, //
            12.0, 37.0, -43.0, //
            -16.0, -43.0, 98.0,
        ];

        // Dual<1> factorization, lane seeded on one entry — reused at a
        // perturbed entry value to build the central FD of the first lane.
        let dual_lane = |entry: usize, perturb: f64| -> [f64; 9] {
            let mut a_dual = [Dual::<1> { v: 0.0, d: [0.0] }; 9];
            for (idx, &v) in base.iter().enumerate() {
                a_dual[idx] = Dual { v, d: [0.0] };
            }
            a_dual[entry].v += perturb;
            a_dual[entry].d[0] = 1.0;
            let mut l_dual = [Dual::<1> { v: 0.0, d: [0.0] }; 9];
            assert!(chol_lower_generic(&a_dual, dim, &mut l_dual));
            let mut out = [0.0f64; 9];
            for k in 0..9 {
                out[k] = l_dual[k].d[0];
            }
            out
        };

        let mut a_hd = [HyperDual::<2, 3> {
            v: 0.0,
            d: [0.0, 0.0],
            h: [0.0; 3],
        }; 9];
        for (idx, &v) in base.iter().enumerate() {
            a_hd[idx] = HyperDual {
                v,
                d: [0.0, 0.0],
                h: [0.0; 3],
            };
        }
        a_hd[0].d[0] = 1.0; // lane 0 seeded on a[0,0]
        a_hd[8].d[1] = 1.0; // lane 1 seeded on a[2,2]

        let mut l_hd = [HyperDual::<2, 3> {
            v: 0.0,
            d: [0.0, 0.0],
            h: [0.0; 3],
        }; 9];
        assert!(chol_lower_generic(&a_hd, dim, &mut l_hd));

        // packed idx(0,0)=0 is ∂²/∂a[0,0]² and idx(1,1)=2 is ∂²/∂a[2,2]² — the
        // two pure second partials reachable by perturbing one diagonal entry
        // at a time and staying SPD; idx(1,0) would need a joint perturbation
        // of both entries at once to FD directly.
        let h = 1e-6;
        for (packed_idx, entry, label) in
            [(0usize, 0usize, "idx(0,0)"), (2usize, 8usize, "idx(1,1)")]
        {
            let l_plus = dual_lane(entry, h);
            let l_minus = dual_lane(entry, -h);
            for k in 0..9 {
                let fd = (l_plus[k] - l_minus[k]) / (2.0 * h);
                assert!(
                    (l_hd[k].h[packed_idx] - fd).abs() <= 1e-5 * fd.abs().max(1.0),
                    "packed {label}, entry {k}: {} vs fd {fd}",
                    l_hd[k].h[packed_idx]
                );
            }
        }
    }
}
