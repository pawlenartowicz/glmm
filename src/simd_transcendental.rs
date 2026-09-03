//! Restricted-domain SIMD transcendentals for the GLM/GLMM fit path — a single
//! compiled-in, full-precision `exp`/`log1p` that vectorizes across the PIRLS
//! row loop (the scalar libm `exp`/`ln_1p` are extern calls the compiler cannot
//! vectorize; profiling put them at ~51% of the no-extras GLMM fit).
//!
//! Two primitives, on exactly the domains the fit path needs — the stable
//! sigmoid (`glm::sigmoid_stable`) and `log1pexp` — namely `exp` of a non-positive
//! argument and `log1p` of `z ∈ (0,1]`:
//!   - `exp` on `[−700, 700]` (clamped): Cody-Waite reduce `x = k·ln2 + r`,
//!     degree-11 minimax `exp(r)` on `[-ln2/2, ln2/2]`, scale by `2^k`. The
//!     reduction is sign-agnostic, so the same kernel serves the fit path's
//!     non-positive arguments and the generation-side full-domain entries
//!     (`exp_clamped`/`exp_fill`).
//!   - `log1p` on `(0,1]`: fdlibm form `f - (hfsq - s·(hfsq + R))`, `s = f/(2+f)`,
//!     `R = w·h(w)`, `w = s²`, degree-9 minimax `h`. Keeps the dominant `f - ½f²`
//!     exact → tail-safe (never forms `1+z`) and ≤1 ULP.
//!
//! Coefficients were derived offline by Remez minimax against an MPFR-class
//! oracle; the `≤1 ULP` accuracy is re-asserted in-repo by the L1 guard test
//! below (vs system libm, the same bar the scalar path meets). The composed
//! sigmoid `p = 1/(1+z)` inherits one extra division → 2 ULP, identical to the
//! scalar `sigmoid_stable` it replaces.
//!
//! **Not** bit-identical to the previous libm helper (different polynomial + a
//! lane-wise deviance reduction): the GLMM goldens move by ≤ a few ULP and are
//! re-frozen. Determinism is within-platform per engine version; SIMD lane width
//! is runtime-dispatched (AVX2→4, AVX-512→8, wasm128→2), so a horizontal
//! reduction sums a µarch-dependent number of partials — cross-µarch results
//! agree only to the last bits, by design (not gated on byte-identity).
//!
//! `2^k` is built with an FP magic-add + `u64` transmute/mask/mul rather than the
//! textbook `(k+1023)<<52`: pulp 0.22's `Simd` trait exposes no integer
//! shift/convert. Valid for `k ≥ -1023` (i.e. `x ⪆ -709`), always true on the
//! fit path where `η` is bounded by `ETA_DIVERGENCE_CAP` — a bound on η
//! directly, so this is exact rather than approximate.
//!
//! **fma policy.** wasm simd128 has no FMA instruction, so a guaranteed-fused
//! `mul_add` lowers to the soft-float compiler-builtins libcall there (measured
//! 9–41× on the GLM/GLMM wasm rows). Every fused site therefore goes through
//! `fmadd`/`fmadd_scalar`, keyed by a `const FUSED: bool` generic: native
//! instantiates `FUSED = true` (hardware fma, byte-identical to the pre-policy
//! code), wasm32 instantiates `FUSED = false` (plain mul/add, vectorizable
//! simd128). Native↔wasm byte-equality is deliberately dropped — the wasm bench
//! gate is tier-2 |Δk| instead. **All future fused ops in owned kernels go
//! through `fmadd`/`fmadd_scalar` — never raw `mul_add`.**
//! The unfused path is safe by construction: Cody-Waite's whole point is that
//! `kf·LN2HI` is exactly representable (≤11-bit integer × 33-bit mantissa), so
//! losing the fuse there costs nothing; the magic-add round-to-nearest tolerates
//! the separately-rounded `x·LOG2E + RND_MAGIC` (a ±1 shift in `k` at ties keeps
//! `r` inside the poly domain); only the Horner steps double-round, ~1–2 extra
//! ULP composed (pinned by `unfused_kernel_within_3ulp_of_libm`).

use crate::spec::{BinomialLink, Family, GammaLink, InverseGaussianLink};
use pulp::Simd;

// exp(r) Horner coefficients (ascending), exp on (-∞,0], degree 11.
const EXP_C: [f64; 12] = [
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3ff0000000000000),
    f64::from_bits(0x3fe0000000000010),
    f64::from_bits(0x3fc55555555554a2),
    f64::from_bits(0x3fa555555554f370),
    f64::from_bits(0x3f81111111130dd6),
    f64::from_bits(0x3f56c16c1878111c),
    f64::from_bits(0x3f2a01a0110572b2),
    f64::from_bits(0x3efa01992d0fe736),
    f64::from_bits(0x3ec71df4520aaeeb),
    f64::from_bits(0x3e928b311c7eb84f),
    f64::from_bits(0x3e5ad661c903688b),
];
// log1p h(w) Horner coefficients (ascending), fdlibm form, degree 9. h(0)=2/3.
const LOG1P_H: [f64; 10] = [
    f64::from_bits(0x3fe5555555555555),
    f64::from_bits(0x3fd999999999a455),
    f64::from_bits(0x3fd24924923cd3a0),
    f64::from_bits(0x3fcc71c727660721),
    f64::from_bits(0x3fc745cefc3caf8b),
    f64::from_bits(0x3fc3b18cab0fef6e),
    f64::from_bits(0x3fc10ab0536ce75b),
    f64::from_bits(0x3fbebaa07b021d58),
    f64::from_bits(0x3fb67ff2751e342c),
    f64::from_bits(0x3fc4b8585fced69a),
];
const EXP_DEG: usize = EXP_C.len() - 1; // 11
const LOG1P_DEG: usize = LOG1P_H.len() - 1; // 9

const LN2HI: f64 = f64::from_bits(0x3fe62e42fee00000); // ln2, low mantissa zeroed (Cody-Waite)
const LN2LO: f64 = f64::from_bits(0x3dea39ef35793c76); // ln2 - LN2HI
const LOG2E: f64 = f64::from_bits(0x3ff71547652b82fe); // 1/ln2
                                                       // round-to-nearest-int: (y + RND_MAGIC) - RND_MAGIC for |y| < 2^51.
const RND_MAGIC: f64 = 1.5 * (1u64 << 52) as f64;
// low 52 bits of (kf + BIAS_MAGIC) equal (k + 1023).
const BIAS_MAGIC: f64 = (1u64 << 52) as f64 + 1023.0;
const MANT_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;
const SHIFT52: u64 = 1u64 << 52;
// exp(-|η|) underflows below ~e⁻⁷⁴⁵; clamp the argument so the FP-magic 2^k build
// stays in-domain (k ≥ -1022). For |η| > 700, z ≈ 0 and the saturated p/w/lp match
// the old libm path to rounding — this only guards the (off-fit-path) extreme tail,
// where the libm code saturated sigmoid → {0,1} anyway.
const EXP_ARG_FLOOR: f64 = -700.0;
// Upper clamp twin for the full-domain entries (`exp_clamped`/`exp_fill`): the
// Cody-Waite reduction and the EXP_C polynomial are sign-agnostic (r ∈
// [−ln2/2, ln2/2] either way) and the FP-magic 2^k build holds for k+1023 ≤ 2046
// (x ≤ ~709) — 700 keeps symmetric headroom. exp(700) ≈ 1e304, finite; the
// generation-side caller (hsk lognormal multiplier) saturates astronomically
// rarely there, where libm would be approaching f64::MAX/inf anyway.
const EXP_ARG_CEIL: f64 = 700.0;

// fma policy: see the module header ("fma policy"). `FUSED` is a const generic
// rather than a bare `cfg!` check so the native test suite can instantiate and
// ULP-guard the wasm (unfused) arithmetic on native hardware; production
// entries pick `FUSED_DEFAULT` for the compile target.
pub(crate) const FUSED_DEFAULT: bool = cfg!(not(target_arch = "wasm32"));

#[inline(always)]
fn fmadd<S: Simd, const FUSED: bool>(simd: S, a: S::f64s, b: S::f64s, c: S::f64s) -> S::f64s {
    if FUSED {
        simd.mul_add_f64s(a, b, c)
    } else {
        simd.add_f64s(simd.mul_f64s(a, b), c)
    }
}

#[inline(always)]
fn fmadd_scalar<const FUSED: bool>(a: f64, b: f64, c: f64) -> f64 {
    if FUSED {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

#[inline(always)]
fn simd_exp_reduced<S: Simd, const FUSED: bool>(simd: S, x: S::f64s) -> S::f64s {
    let t = fmadd::<S, FUSED>(simd, x, simd.splat_f64s(LOG2E), simd.splat_f64s(RND_MAGIC));
    let kf = simd.sub_f64s(t, simd.splat_f64s(RND_MAGIC));
    let neg_kf = simd.neg_f64s(kf);
    let hi = fmadd::<S, FUSED>(simd, neg_kf, simd.splat_f64s(LN2HI), x); // x - kf·ln2hi
    let r = fmadd::<S, FUSED>(simd, neg_kf, simd.splat_f64s(LN2LO), hi); // - kf·ln2lo
    let mut acc = simd.splat_f64s(EXP_C[EXP_DEG]);
    let mut j = EXP_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd::<S, FUSED>(simd, acc, r, simd.splat_f64s(EXP_C[j]));
    }
    let e = simd.add_f64s(kf, simd.splat_f64s(BIAS_MAGIC));
    let m = simd.and_u64s(simd.transmute_u64s_f64s(e), simd.splat_u64s(MANT_MASK));
    let pow2 = simd.transmute_f64s_u64s(simd.mul_u64s(m, simd.splat_u64s(SHIFT52)));
    simd.mul_f64s(acc, pow2)
}

#[inline(always)]
fn simd_log1p_unit<S: Simd, const FUSED: bool>(simd: S, z: S::f64s) -> S::f64s {
    let f = z;
    let hfsq = simd.mul_f64s(simd.splat_f64s(0.5), simd.mul_f64s(f, f));
    let s = simd.div_f64s(f, simd.add_f64s(simd.splat_f64s(2.0), f));
    let w = simd.mul_f64s(s, s);
    let mut acc = simd.splat_f64s(LOG1P_H[LOG1P_DEG]);
    let mut j = LOG1P_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd::<S, FUSED>(simd, acc, w, simd.splat_f64s(LOG1P_H[j]));
    }
    let rr = simd.mul_f64s(w, acc);
    let inner = simd.mul_f64s(s, simd.add_f64s(hfsq, rr));
    simd.sub_f64s(f, simd.sub_f64s(hfsq, inner))
}

// All fit-path helpers share `z = exp(-|η|)` (clamped) and the `η ≥ 0` sign mask.
#[inline(always)]
fn simd_z_mask<S: Simd, const FUSED: bool>(simd: S, eta: S::f64s) -> (S::f64s, S::m64s) {
    let neg_abs = simd.max_f64s(
        simd.neg_f64s(simd.abs_f64s(eta)),
        simd.splat_f64s(EXP_ARG_FLOOR),
    );
    let z = simd_exp_reduced::<S, FUSED>(simd, neg_abs);
    let mask = simd.greater_than_or_equal_f64s(eta, simd.splat_f64s(0.0));
    (z, mask)
}

/// Fused `(p, w, log1pexp(η))` for the GLM IRLS / GLMM PIRLS row-pass — one
/// `exp` and one `log1p` of the same argument shared across all three.
/// `p = sigmoid(η)` (branchless `glm::sigmoid_stable`),
/// `w = max(p(1-p), WEIGHT_CLAMP)`, `lp = log(1 + exp(η))` (stable in both tails).
#[inline(always)]
fn simd_fused<S: Simd, const FUSED: bool>(simd: S, eta: S::f64s) -> (S::f64s, S::f64s, S::f64s) {
    let one = simd.splat_f64s(1.0);
    let (z, mask) = simd_z_mask::<S, FUSED>(simd, eta);
    let l = simd_log1p_unit::<S, FUSED>(simd, z);
    let opz = simd.add_f64s(one, z);
    let p = simd.select_f64s(mask, simd.div_f64s(one, opz), simd.div_f64s(z, opz));
    let lp = simd.select_f64s(mask, simd.add_f64s(eta, l), l);
    let w = simd.max_f64s(
        simd.mul_f64s(p, simd.sub_f64s(one, p)),
        simd.splat_f64s(crate::glm::WEIGHT_CLAMP),
    );
    (p, w, lp)
}

// Scalar mirror of `simd_fused`, bit-identical op-for-op (same coeffs, FMA via
// `f64::mul_add`), used for the sub-lane tail so the whole row range runs one
// kernel regardless of where the SIMD chunk boundary falls.
#[inline]
fn scalar_exp_reduced<const FUSED: bool>(x: f64) -> f64 {
    let kf = fmadd_scalar::<FUSED>(x, LOG2E, RND_MAGIC) - RND_MAGIC;
    let hi = fmadd_scalar::<FUSED>(-kf, LN2HI, x);
    let r = fmadd_scalar::<FUSED>(-kf, LN2LO, hi);
    let mut acc = EXP_C[EXP_DEG];
    let mut j = EXP_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd_scalar::<FUSED>(acc, r, EXP_C[j]);
    }
    let m = (kf + BIAS_MAGIC).to_bits() & MANT_MASK;
    acc * f64::from_bits(m.wrapping_mul(SHIFT52))
}
#[inline]
fn scalar_log1p_unit<const FUSED: bool>(z: f64) -> f64 {
    let f = z;
    let hfsq = 0.5 * (f * f);
    let s = f / (2.0 + f);
    let w = s * s;
    let mut acc = LOG1P_H[LOG1P_DEG];
    let mut j = LOG1P_DEG;
    while j > 0 {
        j -= 1;
        acc = fmadd_scalar::<FUSED>(acc, w, LOG1P_H[j]);
    }
    let rr = w * acc;
    f - (hfsq - s * (hfsq + rr))
}
#[inline]
fn scalar_z<const FUSED: bool>(eta: f64) -> f64 {
    scalar_exp_reduced::<FUSED>((-eta.abs()).max(EXP_ARG_FLOOR))
}
#[inline]
fn scalar_fused<const FUSED: bool>(eta: f64) -> (f64, f64, f64) {
    let z = scalar_z::<FUSED>(eta);
    let lp = scalar_log1pexp_from_z::<FUSED>(z, eta);
    let p = if eta >= 0.0 {
        1.0 / (1.0 + z)
    } else {
        z / (1.0 + z)
    };
    let w = (p * (1.0 - p)).max(crate::glm::WEIGHT_CLAMP);
    (p, w, lp)
}

// `log(1 + eᵉᵗᵃ)` core, taking the caller's `z = e^{−|η|}` so `scalar_fused`
// reuses the `z` it already computed for μ/W; the eta-only wrapper below
// serves `scalar_log1pexp` (the `Scalar::log1pexp` binding).
#[inline]
fn scalar_log1pexp_from_z<const FUSED: bool>(z: f64, eta: f64) -> f64 {
    let l = scalar_log1p_unit::<FUSED>(z);
    if eta >= 0.0 {
        eta + l
    } else {
        l
    }
}

#[inline]
fn scalar_log1pexp_generic<const FUSED: bool>(eta: f64) -> f64 {
    scalar_log1pexp_from_z::<FUSED>(scalar_z::<FUSED>(eta), eta)
}

/// `log(1 + eᵉᵗᵃ)`, overflow-safe on both tails: the `z = e^{−|η|}` reduction
/// plus `log1p(z)`, with the `η ≥ 0` branch adding `η` back. Same two kernels
/// `scalar_fused` uses — extracted so `Scalar::log1pexp` binds to them without
/// computing the μ/W companions.
#[inline]
pub(crate) fn scalar_log1pexp(eta: f64) -> f64 {
    scalar_log1pexp_generic::<{ FUSED_DEFAULT }>(eta)
}

struct PwLog1pexpOp<'a, const FUSED: bool> {
    eta: &'a [f64],
    p: &'a mut [f64],
    w: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for PwLog1pexpOp<'_, FUSED> {
    type Output = f64;
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f64 {
        let (eh, et) = S::as_simd_f64s(self.eta);
        let (ph, pt) = S::as_mut_simd_f64s(self.p);
        let (wh, wt) = S::as_mut_simd_f64s(self.w);
        let mut dsum = simd.splat_f64s(0.0);
        for i in 0..eh.len() {
            let (p, w, lp) = simd_fused::<S, FUSED>(simd, eh[i]);
            ph[i] = p;
            wh[i] = w;
            dsum = simd.add_f64s(dsum, lp);
        }
        let mut acc = simd.reduce_sum_f64s(dsum);
        for i in 0..et.len() {
            let (p, w, lp) = scalar_fused::<FUSED>(et[i]);
            pt[i] = p;
            wt[i] = w;
            acc += lp;
        }
        acc
    }
}

/// Fill `p[i] = sigmoid(η[i])` and `w[i] = max(p(1-p), WEIGHT_CLAMP)` for the
/// whole slice via the SIMD kernel; return `Σ log1pexp(η[i])` (lane-wise SIMD
/// reduction + scalar tail). The PIRLS deviance is `2·(Σ log1pexp − Σ y·η)`,
/// with the `Σ y·η` half accumulated by the caller's scalar η-pass.
///
/// Folding that half in here — one lane-wise `Σ (log1pexp(ηᵢ) − yᵢηᵢ)` instead of
/// two sums — was built and measured, and does NOT pay. Measured 2026-08-23,
/// locked clock, one pinned P-core, min of 5 warm fits, on a 52515-row single-
/// intercept Bernoulli logit GLMM: 4.1% slower per BOBYQA evaluation and 4.8%
/// slower on the whole fit. The scalar add it removes from the caller's η-pass
/// is worth less than the fourth input stream it adds to this loop, which is the
/// tightest row kernel in the crate. It also perturbed the deviance enough to
/// cost three extra outer evaluations. Do not re-fold without a fresh
/// measurement.
pub(crate) fn pw_and_log1pexp_sum(eta: &[f64], p: &mut [f64], w: &mut [f64]) -> f64 {
    debug_assert_eq!(eta.len(), p.len());
    debug_assert_eq!(eta.len(), w.len());
    pulp::Arch::new().dispatch(PwLog1pexpOp::<{ FUSED_DEFAULT }> { eta, p, w })
}

struct SigmoidInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for SigmoidInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let one = simd.splat_f64s(1.0);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for x in head.iter_mut() {
            let (z, mask) = simd_z_mask::<S, FUSED>(simd, *x);
            let opz = simd.add_f64s(one, z);
            *x = simd.select_f64s(mask, simd.div_f64s(one, opz), simd.div_f64s(z, opz));
        }
        for x in tail.iter_mut() {
            let z = scalar_z::<FUSED>(*x);
            *x = if *x >= 0.0 {
                1.0 / (1.0 + z)
            } else {
                z / (1.0 + z)
            };
        }
    }
}

/// In-place `buf[i] = sigmoid(buf[i])` for the generation-side binary-outcome
/// draw — SIMD head + bit-identical scalar tail, p only (the fit path's fused
/// (p, w, Σlog1pexp) variant is `pw_and_log1pexp_sum`).
#[inline]
pub fn sigmoid_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(SigmoidInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

/// Owned scalar `exp` on (−∞, 0], platform-default fma policy, argument clamped
/// to the kernel's 2^k domain. The generation-side replacement for libm `.exp()`
/// at non-positive arguments (erfc's `exp(−x²)`); ≤1 ULP of libm.
#[inline]
pub fn exp_nonpos(x: f64) -> f64 {
    scalar_exp_reduced::<{ FUSED_DEFAULT }>(x.max(EXP_ARG_FLOOR))
}

/// Owned scalar `exp` on the full certified domain `[−700, 700]` (two-sided
/// clamp); ≤1 ULP of libm. Scalar tail twin of `exp_fill`.
pub(crate) fn exp_clamped(x: f64) -> f64 {
    scalar_exp_reduced::<{ FUSED_DEFAULT }>(x.clamp(EXP_ARG_FLOOR, EXP_ARG_CEIL))
}

struct ExpInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for ExpInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let lo = simd.splat_f64s(EXP_ARG_FLOOR);
        let hi = simd.splat_f64s(EXP_ARG_CEIL);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for x in head.iter_mut() {
            *x = simd_exp_reduced::<S, FUSED>(simd, simd.min_f64s(simd.max_f64s(*x, lo), hi));
        }
        for x in tail.iter_mut() {
            *x = exp_clamped(*x);
        }
    }
}

/// In-place `buf[i] = exp(buf[i])`, argument clamped to `[−700, 700]` — the
/// generation-side column pass for the heteroskedasticity lognormal multiplier.
/// SIMD head + bit-identical scalar tail (`exp_clamped`).
#[inline]
pub fn exp_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(ExpInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

// ln(u) on the censored-Φ domain: bit-trick range reduction reusing LOG1P_H —
// ln(u) = k·ln2 + log1p(m−1) with the fdlibm √2-normalization m ∈ [√2/2, √2)
// (k = round(log2 u)): a plain m ∈ [1,2) reduction cancels catastrophically as
// u → 1 (k=−1 against log1p(2u−1) → ln2, measured ~5000 ULP); √2-normalizing
// gives k=0 there and feeds log1p the small argument directly. m comes from
// forcing the exponent field to 1023 (AND mantissa, OR bits(1.0)), then a
// halving select for the m ≥ √2 half. LOG1P_H stays valid: its w = s² domain
// only shrinks (|s| ≤ 0.172 → w ≤ 0.0295 ⊂ [0, 1/9]). pulp 0.22 exposes no
// integer shift or u64→f64 convert, so k is recovered with an 11-step
// compare-ladder kf = −Σ_{j=1..11} [u < √2·2^−j] (exactly consistent with the
// m ≥ √2 select — the thresholds are exact 2^−j scalings of the same √2). The
// two-sided clamp makes the ladder total on the skewed-marginal call site:
// u ≤ 2^−11 → ln(2^−11) = −7.625, past the −EXP_CAP censor either way (the
// censored-Exp(1) cap ≈ 6.96), so the capped output matches libm's exactly;
// u = 1.0 (Φ saturates only at |z| ≳ 8.3, outside the generated ±6-SD latent
// range) → ln(1 − 2^−53) ≈ −1.1e-16 instead of −0.0.
const LN_U_FLOOR: f64 = 4.8828125e-4; // 2^-11
const LN_U_CEIL: f64 = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF); // 1 − 2^-53
const ONE_BITS: u64 = 0x3FF0_0000_0000_0000;

#[inline]
fn scalar_ln_unit<const FUSED: bool>(u: f64) -> f64 {
    let u = u.clamp(LN_U_FLOOR, LN_U_CEIL);
    let mut kf = 0.0f64;
    let mut th = std::f64::consts::SQRT_2 * 0.5;
    for _ in 0..11 {
        if u < th {
            kf -= 1.0;
        }
        th *= 0.5;
    }
    let m = f64::from_bits((u.to_bits() & MANT_MASK) | ONE_BITS);
    let m = if m < std::f64::consts::SQRT_2 {
        m
    } else {
        0.5 * m
    };
    let l = scalar_log1p_unit::<FUSED>(m - 1.0);
    fmadd_scalar::<FUSED>(kf, LN2HI, fmadd_scalar::<FUSED>(kf, LN2LO, l))
}

struct LnInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for LnInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let one = simd.splat_f64s(1.0);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for v in head.iter_mut() {
            let u = simd.min_f64s(
                simd.max_f64s(*v, simd.splat_f64s(LN_U_FLOOR)),
                simd.splat_f64s(LN_U_CEIL),
            );
            let mut kf = simd.splat_f64s(0.0);
            let mut th = std::f64::consts::SQRT_2 * 0.5;
            for _ in 0..11 {
                let mask = simd.less_than_f64s(u, simd.splat_f64s(th));
                kf = simd.select_f64s(mask, simd.sub_f64s(kf, one), kf);
                th *= 0.5;
            }
            let m = simd.transmute_f64s_u64s(simd.or_u64s(
                simd.and_u64s(simd.transmute_u64s_f64s(u), simd.splat_u64s(MANT_MASK)),
                simd.splat_u64s(ONE_BITS),
            ));
            let lt = simd.less_than_f64s(m, simd.splat_f64s(std::f64::consts::SQRT_2));
            let m = simd.select_f64s(lt, m, simd.mul_f64s(simd.splat_f64s(0.5), m));
            let l = simd_log1p_unit::<S, FUSED>(simd, simd.sub_f64s(m, one));
            let inner = fmadd::<S, FUSED>(simd, kf, simd.splat_f64s(LN2LO), l);
            *v = fmadd::<S, FUSED>(simd, kf, simd.splat_f64s(LN2HI), inner);
        }
        for v in tail.iter_mut() {
            *v = scalar_ln_unit::<FUSED>(*v);
        }
    }
}

/// Owned scalar `ln` on the censored-Φ domain (clamped to `[2^−11, 1−2^−53]`) —
/// the skewed-marginal replacement for libm `.ln()`. Scalar twin of `ln_fill`,
/// bit-identical per element.
#[inline]
pub fn ln_owned(u: f64) -> f64 {
    scalar_ln_unit::<{ FUSED_DEFAULT }>(u)
}

/// In-place `buf[i] = ln(buf[i])` on the censored-Φ domain — SIMD head +
/// bit-identical scalar tail (`ln_owned`).
#[inline]
pub fn ln_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(LnInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

// A&S 7.1.26 constants — shared by the SIMD Φ kernel below and its scalar
// mirror `scalar_phi`; change together.
const ERF_A1: f64 = 0.254829592;
const ERF_A2: f64 = -0.284496736;
const ERF_A3: f64 = 1.421413741;
const ERF_A4: f64 = -1.453152027;
const ERF_A5: f64 = 1.061405429;
const ERF_P: f64 = 0.3275911;

struct PhiInplaceOp<'a, const FUSED: bool> {
    buf: &'a mut [f64],
}
impl<const FUSED: bool> pulp::WithSimd for PhiInplaceOp<'_, FUSED> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        let one = simd.splat_f64s(1.0);
        let half = simd.splat_f64s(0.5);
        let c = simd.splat_f64s(std::f64::consts::FRAC_1_SQRT_2);
        let (head, tail) = S::as_mut_simd_f64s(self.buf);
        for v in head.iter_mut() {
            let x = simd.mul_f64s(simd.neg_f64s(*v), c); // x = (−z)·(1/√2)
            let neg = simd.less_than_f64s(x, simd.splat_f64s(0.0));
            let ax = simd.abs_f64s(x);
            let t = simd.div_f64s(
                one,
                simd.add_f64s(one, simd.mul_f64s(simd.splat_f64s(ERF_P), ax)),
            );
            // (((((A5·t + A4)·t) + A3)·t + A2)·t + A1)·t — plain mul/add, as scalar.
            let mut poly = simd.add_f64s(
                simd.mul_f64s(simd.splat_f64s(ERF_A5), t),
                simd.splat_f64s(ERF_A4),
            );
            poly = simd.add_f64s(simd.mul_f64s(poly, t), simd.splat_f64s(ERF_A3));
            poly = simd.add_f64s(simd.mul_f64s(poly, t), simd.splat_f64s(ERF_A2));
            poly = simd.add_f64s(simd.mul_f64s(poly, t), simd.splat_f64s(ERF_A1));
            poly = simd.mul_f64s(poly, t);
            let e = simd_exp_reduced::<S, FUSED>(
                simd,
                simd.max_f64s(
                    simd.mul_f64s(simd.neg_f64s(ax), ax),
                    simd.splat_f64s(EXP_ARG_FLOOR),
                ),
            );
            let y = simd.sub_f64s(one, simd.mul_f64s(poly, e));
            let erf = simd.select_f64s(neg, simd.neg_f64s(y), y);
            *v = simd.mul_f64s(half, simd.sub_f64s(one, erf));
        }
        for v in tail.iter_mut() {
            *v = scalar_phi(*v);
        }
    }
}

/// Scalar mirror of the SIMD Φ head above; bit-identical op-for-op (same A&S
/// 7.1.26 constants, same owned exp_nonpos) — change together. Kept
/// self-contained here (not shared through a common dependency) so
/// `phi_fill`'s tail loop stays a plain function call with no extra crate edge.
#[inline]
pub(crate) fn scalar_phi(z: f64) -> f64 {
    let x = -z * std::f64::consts::FRAC_1_SQRT_2;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + ERF_P * ax);
    let poly = (((((ERF_A5 * t + ERF_A4) * t) + ERF_A3) * t + ERF_A2) * t + ERF_A1) * t;
    let y = 1.0 - poly * exp_nonpos(-ax * ax);
    0.5 * (1.0 - sign * y)
}

/// High-precision Φ(z) = ½·erfc(−z/√2), accurate to ~1e-15 (full double).
/// Used **only** by the probit link in `family.rs`. Deliberately separate from
/// `scalar_phi`/`phi_fill`: that data-gen path is bit-pinned to
/// the A&S-7.1.26 form (~7.5e-8) and must not change, but the probit
/// Hessian SE differentiates Φ twice and amplifies its error, so the fit
/// needs a Φ good to machine precision. `erfc` is W. J. Cody's rational
/// Chebyshev approximation (CALERF, Netlib SPECFUN), the same algorithm libm
/// uses; coefficients verbatim from the reference.
#[inline]
pub(crate) fn phi_hp(z: f64) -> f64 {
    0.5 * erfc_cody(-z * std::f64::consts::FRAC_1_SQRT_2)
}

// W. J. Cody CALERF constants — shared by the branching scalar reference
// `erfc_cody` below and the branch-free SIMD blend `simd_erfc`/`scalar_erfc_blend`
// in the family-kernel section; change together.
#[allow(clippy::excessive_precision)]
const ERFC_A: [f64; 5] = [
    3.16112374387056560e00,
    1.13864154151050156e02,
    3.77485237685302021e02,
    3.20937758913846947e03,
    1.85777706184603153e-1,
];
#[allow(clippy::excessive_precision)]
const ERFC_B: [f64; 4] = [
    2.36012909523441209e01,
    2.44024637934444173e02,
    1.28261652607737228e03,
    2.84423683343917062e03,
];
#[allow(clippy::excessive_precision)]
const ERFC_C: [f64; 9] = [
    5.64188496988670089e-1,
    8.88314979438837594e00,
    6.61191906371416295e01,
    2.98635138197400131e02,
    8.81952221241769090e02,
    1.71204761263407058e03,
    2.05107837782607147e03,
    1.23033935479799725e03,
    2.15311535474403846e-8,
];
#[allow(clippy::excessive_precision)]
const ERFC_D: [f64; 8] = [
    1.57449261107098347e01,
    1.17693950891312499e02,
    5.37181101862009858e02,
    1.62138957456669019e03,
    3.29079923573345963e03,
    4.36261909014324716e03,
    3.43936767414372164e03,
    1.23033935480374942e03,
];
#[allow(clippy::excessive_precision)]
const ERFC_P: [f64; 6] = [
    3.05326634961232344e-1,
    3.60344899949804439e-1,
    1.25781726111229246e-1,
    1.60837851487422766e-2,
    6.58749161529837803e-4,
    1.63153871373020978e-2,
];
#[allow(clippy::excessive_precision)]
const ERFC_Q: [f64; 5] = [
    2.56852019228982242e00,
    1.87295284992346047e00,
    5.27905102951428412e-1,
    6.05183413124413191e-2,
    2.33520497626869185e-3,
];
#[allow(clippy::excessive_precision)]
const ERFC_SQRPI: f64 = 5.6418958354775628695e-1;
const ERFC_THRESH: f64 = 0.46875;
const ERFC_SIXTEN: f64 = 16.0;
const ERFC_XSMALL: f64 = 1.11e-16;
const ERFC_XBIG: f64 = 26.543;

/// `erfc(x)` to ~1e-15 — W. J. Cody, *Rational Chebyshev approximation for the
/// error function* (Math. Comp. 1969), CALERF jint=1. Three regions in |x|:
/// rational `erf` for |x|≤0.46875, two `erfc` rationals for the tails, with the
/// `exp(−⌊16y⌋²/16)·exp(−δ)` split that preserves precision in the exponential.
fn erfc_cody(x: f64) -> f64 {
    const A: [f64; 5] = ERFC_A;
    const B: [f64; 4] = ERFC_B;
    const C: [f64; 9] = ERFC_C;
    const D: [f64; 8] = ERFC_D;
    const P: [f64; 6] = ERFC_P;
    const Q: [f64; 5] = ERFC_Q;
    const SQRPI: f64 = ERFC_SQRPI;
    const THRESH: f64 = ERFC_THRESH;
    const SIXTEN: f64 = ERFC_SIXTEN;
    const XSMALL: f64 = ERFC_XSMALL;
    const XBIG: f64 = ERFC_XBIG;

    let y = x.abs();
    if y <= THRESH {
        // erf region; erfc = 1 − erf, and erf carries x's sign (odd).
        let ysq = if y > XSMALL { y * y } else { 0.0 };
        let mut xnum = A[4] * ysq;
        let mut xden = ysq;
        for i in 0..3 {
            xnum = (xnum + A[i]) * ysq;
            xden = (xden + B[i]) * ysq;
        }
        return 1.0 - x * (xnum + A[3]) / (xden + B[3]);
    }
    let mut result = if y <= 4.0 {
        let mut xnum = C[8] * y;
        let mut xden = y;
        for i in 0..7 {
            xnum = (xnum + C[i]) * y;
            xden = (xden + D[i]) * y;
        }
        (xnum + C[7]) / (xden + D[7])
    } else if y >= XBIG {
        0.0
    } else {
        let ysq = 1.0 / (y * y);
        let mut xnum = P[5] * ysq;
        let mut xden = ysq;
        for i in 0..4 {
            xnum = (xnum + P[i]) * ysq;
            xden = (xden + Q[i]) * ysq;
        }
        (SQRPI - ysq * (xnum + P[4]) / (xden + Q[4])) / y
    };
    if y < XBIG {
        // Reintroduce the exp(−y²) factor with the ⌊16y⌋/16 split for precision.
        let ysq = (y * SIXTEN).trunc() / SIXTEN;
        let del = (y - ysq) * (y + ysq);
        result *= (-ysq * ysq).exp() * (-del).exp();
    }
    // erfc(−|x|) = 2 − erfc(|x|).
    if x < 0.0 {
        2.0 - result
    } else {
        result
    }
}

/// `ln Γ(x)` for `x > 0`, accurate to ~1e-15 (full double). Lanczos
/// approximation, `g = 7`, 9-coefficient series (Lanczos 1964; coefficients the
/// widely-used Godfrey/Boost set) — relative error < 2e-16 on `x ∈ (0, ∞)`.
/// Used by the Gamma-GLMM Laplace objective (`family::gamma_aic`, lme4's
/// `Gamma()$aic`): the dispersion enters the deviance only through `lnΓ(1/φ)`, so
/// matching `glmer` needs `lnΓ` to machine precision. No `digamma`/`trigamma` —
/// lme4 profiles the dispersion as `D/Σw` rather than solving the ML score.
#[allow(clippy::excessive_precision)]
pub(crate) fn ln_gamma(x: f64) -> f64 {
    // Lanczos g=7 coefficients (c[0] is the series constant a₀).
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    const G: f64 = 7.0;
    const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_74; // ½·ln(2π)
                                                       // Reflection is unnecessary here (callers pass x>0), so use the direct series.
    let x = x - 1.0;
    let mut a = C[0];
    let t = x + G + 0.5;
    for (i, &c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    LN_SQRT_2PI + (x + 0.5) * t.ln() - t + a.ln()
}

/// In-place `buf[i] = Φ(buf[i])` — SIMD mirror of `distributions::phi`
/// (bit-identical per element; the scalar tail calls the self-contained
/// `scalar_phi` twin above).
#[inline]
pub fn phi_fill(buf: &mut [f64]) {
    pulp::Arch::new().dispatch(PhiInplaceOp::<{ FUSED_DEFAULT }> { buf });
}

// ---------------------------------------------------------------------------
// Shared family kernel — one batched η → (μ, W, z) pass per (family, link)
// ---------------------------------------------------------------------------
//
// `family_pass` below replaces the per-row scalar loop that every non-canonical
// family used to run at the four IRLS/PIRLS assembly sites (`glm::glm_irls_fit`
// and the three `glmm::pirls::pirls_solve*` variants). One dispatch on `family`
// picks a vectorized arm instead of one `match` + two libm calls per row; the
// log-link arms additionally compute `exp(η)` ONCE and reuse it for both μ and
// dμ/dη, where the scalar `family::link_inv` + `family::mu_eta` pair computed it
// twice.
//
// The arms are held to `family.rs`'s formulas — `family::irls_weight_and_resid`
// stays the scalar statement of the IRLS triple (μ, W_raw, working residual),
// including its canonical-link shortcut `W_raw = V(μ)` for logit and Poisson-log.
// Where an arm's μ moves against that scalar reference it is from replacing a
// libm `exp` with this module's owned kernel: ≤2 ULP on the log-link arms, and
// up to the 5 ULP `erfc_blend_accuracy_and_head_tail_identity` pins on the probit
// arm, whose blend composes two owned `exp`s through a product.

/// Branch-free SIMD `erfc(x)` — Cody CALERF (see [`erfc_cody`]) with all three
/// `|x|` regions evaluated and blended by mask, because one lane vector
/// straddles region boundaries. Roughly 3× the scalar arithmetic, all of it
/// cheap polynomial work, against a libm call the compiler cannot vectorize at
/// all.
///
/// Certified for `|x| ≤ √700 ≈ 26.45`: past that the reintroduced
/// `exp(−⌊16y⌋²/16)` factor hits [`EXP_ARG_FLOOR`] and saturates. `erfc` is
/// below 1e-305 there, 293 decades past where the probit μ has already clamped
/// to `family::PROB_EPS` (1e-12), so the fit path never reads that region.
///
/// Scalar twin: [`scalar_erfc_blend`], op-for-op — change together.
#[inline(always)]
fn simd_erfc<S: Simd, const FUSED: bool>(simd: S, x: S::f64s) -> S::f64s {
    let one = simd.splat_f64s(1.0);
    let y = simd.abs_f64s(x);

    // Region 1, |x| ≤ 0.46875: the `erf` rational, erfc = 1 − erf (erf odd in x).
    let ysq = simd.select_f64s(
        simd.less_than_f64s(simd.splat_f64s(ERFC_XSMALL), y),
        simd.mul_f64s(y, y),
        simd.splat_f64s(0.0),
    );
    let mut xnum = simd.mul_f64s(simd.splat_f64s(ERFC_A[4]), ysq);
    let mut xden = ysq;
    for i in 0..3 {
        xnum = simd.mul_f64s(simd.add_f64s(xnum, simd.splat_f64s(ERFC_A[i])), ysq);
        xden = simd.mul_f64s(simd.add_f64s(xden, simd.splat_f64s(ERFC_B[i])), ysq);
    }
    let r1 = simd.sub_f64s(
        one,
        simd.div_f64s(
            simd.mul_f64s(x, simd.add_f64s(xnum, simd.splat_f64s(ERFC_A[3]))),
            simd.add_f64s(xden, simd.splat_f64s(ERFC_B[3])),
        ),
    );

    // Region 2, 0.46875 < |x| ≤ 4.
    let mut xnum = simd.mul_f64s(simd.splat_f64s(ERFC_C[8]), y);
    let mut xden = y;
    for i in 0..7 {
        xnum = simd.mul_f64s(simd.add_f64s(xnum, simd.splat_f64s(ERFC_C[i])), y);
        xden = simd.mul_f64s(simd.add_f64s(xden, simd.splat_f64s(ERFC_D[i])), y);
    }
    let r2 = simd.div_f64s(
        simd.add_f64s(xnum, simd.splat_f64s(ERFC_C[7])),
        simd.add_f64s(xden, simd.splat_f64s(ERFC_D[7])),
    );

    // Region 3, |x| > 4. Evaluated on `max(y, 4)` so the 1/y² the discarded
    // small-|x| lanes would form stays finite (y = 0 is a live input here).
    let yb = simd.max_f64s(y, simd.splat_f64s(4.0));
    let iy2 = simd.div_f64s(one, simd.mul_f64s(yb, yb));
    let mut xnum = simd.mul_f64s(simd.splat_f64s(ERFC_P[5]), iy2);
    let mut xden = iy2;
    for i in 0..4 {
        xnum = simd.mul_f64s(simd.add_f64s(xnum, simd.splat_f64s(ERFC_P[i])), iy2);
        xden = simd.mul_f64s(simd.add_f64s(xden, simd.splat_f64s(ERFC_Q[i])), iy2);
    }
    let r3 = simd.div_f64s(
        simd.sub_f64s(
            simd.splat_f64s(ERFC_SQRPI),
            simd.div_f64s(
                simd.mul_f64s(iy2, simd.add_f64s(xnum, simd.splat_f64s(ERFC_P[4]))),
                simd.add_f64s(xden, simd.splat_f64s(ERFC_Q[4])),
            ),
        ),
        yb,
    );

    // Both tail rationals carry the exp(−y²) factor, split as
    // exp(−⌊16y⌋²/16²)·exp(−δ) to keep the exponential's precision.
    // `⌊16y⌋` via round-to-nearest + a downward correction: pulp 0.22 exposes no
    // `trunc`, and 16y ≤ 425 here is far inside the magic-add's |v| < 2^51 range.
    let t16 = simd.mul_f64s(y, simd.splat_f64s(ERFC_SIXTEN));
    let rnd = simd.sub_f64s(
        simd.add_f64s(t16, simd.splat_f64s(RND_MAGIC)),
        simd.splat_f64s(RND_MAGIC),
    );
    let trunc = simd.select_f64s(simd.less_than_f64s(t16, rnd), simd.sub_f64s(rnd, one), rnd);
    let yq = simd.div_f64s(trunc, simd.splat_f64s(ERFC_SIXTEN));
    let del = simd.mul_f64s(simd.sub_f64s(y, yq), simd.add_f64s(y, yq));
    let e1 = simd_exp_reduced::<S, FUSED>(
        simd,
        simd.max_f64s(
            simd.neg_f64s(simd.mul_f64s(yq, yq)),
            simd.splat_f64s(EXP_ARG_FLOOR),
        ),
    );
    let e2 = simd_exp_reduced::<S, FUSED>(
        simd,
        simd.max_f64s(simd.neg_f64s(del), simd.splat_f64s(EXP_ARG_FLOOR)),
    );
    let tail = simd.mul_f64s(
        simd.select_f64s(simd.less_than_f64s(simd.splat_f64s(4.0), y), r3, r2),
        simd.mul_f64s(e1, e2),
    );
    let tail = simd.select_f64s(
        simd.greater_than_or_equal_f64s(y, simd.splat_f64s(ERFC_XBIG)),
        simd.splat_f64s(0.0),
        tail,
    );
    // erfc(−|x|) = 2 − erfc(|x|) — the two tail rationals are in |x| only. The
    // region-1 branch is exempt: it already carries x's sign through `erf`'s
    // oddness, which is why Cody returns from that region before the reflection.
    let tail = simd.select_f64s(
        simd.less_than_f64s(x, simd.splat_f64s(0.0)),
        simd.sub_f64s(simd.splat_f64s(2.0), tail),
        tail,
    );

    simd.select_f64s(
        simd.less_than_f64s(simd.splat_f64s(ERFC_THRESH), y),
        tail,
        r1,
    )
}

/// Scalar twin of [`simd_erfc`], bit-identical op-for-op (same blend, same owned
/// `exp`) — the sub-lane tail of every probit row range. Deliberately NOT
/// [`erfc_cody`]: that one branches per region and calls libm `exp`, so it
/// differs from the blend in the last bits (the 5 ULP band
/// `erfc_blend_accuracy_and_head_tail_identity` pins, on a measured 4).
#[inline]
fn scalar_erfc_blend<const FUSED: bool>(x: f64) -> f64 {
    let y = x.abs();

    let ysq = if ERFC_XSMALL < y { y * y } else { 0.0 };
    let mut xnum = ERFC_A[4] * ysq;
    let mut xden = ysq;
    for i in 0..3 {
        xnum = (xnum + ERFC_A[i]) * ysq;
        xden = (xden + ERFC_B[i]) * ysq;
    }
    let r1 = 1.0 - x * (xnum + ERFC_A[3]) / (xden + ERFC_B[3]);

    let mut xnum = ERFC_C[8] * y;
    let mut xden = y;
    for i in 0..7 {
        xnum = (xnum + ERFC_C[i]) * y;
        xden = (xden + ERFC_D[i]) * y;
    }
    let r2 = (xnum + ERFC_C[7]) / (xden + ERFC_D[7]);

    let yb = y.max(4.0);
    let iy2 = 1.0 / (yb * yb);
    let mut xnum = ERFC_P[5] * iy2;
    let mut xden = iy2;
    for i in 0..4 {
        xnum = (xnum + ERFC_P[i]) * iy2;
        xden = (xden + ERFC_Q[i]) * iy2;
    }
    let r3 = (ERFC_SQRPI - iy2 * (xnum + ERFC_P[4]) / (xden + ERFC_Q[4])) / yb;

    let t16 = y * ERFC_SIXTEN;
    let rnd = (t16 + RND_MAGIC) - RND_MAGIC;
    let trunc = if t16 < rnd { rnd - 1.0 } else { rnd };
    let yq = trunc / ERFC_SIXTEN;
    let del = (y - yq) * (y + yq);
    let e1 = scalar_exp_reduced::<FUSED>((-(yq * yq)).max(EXP_ARG_FLOOR));
    let e2 = scalar_exp_reduced::<FUSED>((-del).max(EXP_ARG_FLOOR));
    let tail = if 4.0 < y { r3 } else { r2 } * (e1 * e2);
    let tail = if y >= ERFC_XBIG { 0.0 } else { tail };
    let tail = if x < 0.0 { 2.0 - tail } else { tail };

    if ERFC_THRESH < y {
        tail
    } else {
        r1
    }
}

// Probit μ = Φ(η) = ½·erfc(−η/√2) and dμ/dη = φ(η) = exp(−η²/2)/√(2π), sharing
// this module's owned `exp` for both. `family::link_inv`'s scalar `phi_hp` is the
// same identity on the branching `erfc_cody`; the two agree to within the 5 ULP
// `erfc_blend_accuracy_and_head_tail_identity` pins, on a measured 4.
#[inline(always)]
fn simd_probit<S: Simd, const FUSED: bool>(simd: S, eta: S::f64s) -> (S::f64s, S::f64s) {
    let mu = simd.mul_f64s(
        simd.splat_f64s(0.5),
        simd_erfc::<S, FUSED>(
            simd,
            simd.mul_f64s(
                simd.neg_f64s(eta),
                simd.splat_f64s(std::f64::consts::FRAC_1_SQRT_2),
            ),
        ),
    );
    let dmu = simd.mul_f64s(
        simd.splat_f64s(crate::family::FRAC_1_SQRT_2PI),
        simd_exp_reduced::<S, FUSED>(
            simd,
            simd.max_f64s(
                simd.mul_f64s(simd.splat_f64s(-0.5), simd.mul_f64s(eta, eta)),
                simd.splat_f64s(EXP_ARG_FLOOR),
            ),
        ),
    );
    (mu, dmu)
}

#[inline]
fn scalar_probit<const FUSED: bool>(eta: f64) -> (f64, f64) {
    let mu = 0.5 * scalar_erfc_blend::<FUSED>(-eta * std::f64::consts::FRAC_1_SQRT_2);
    let dmu = crate::family::FRAC_1_SQRT_2PI
        * scalar_exp_reduced::<FUSED>((-0.5 * (eta * eta)).max(EXP_ARG_FLOOR));
    (mu, dmu)
}

/// Sigmoid on the owned `exp`, without the fused `(w, log1pexp)` companions —
/// the weighted-logit arm needs μ alone (its deviance goes through
/// `family::dev_resid`, not the `Σ log1pexp − Σ y·η` identity the unweighted
/// fast path uses).
#[inline(always)]
fn simd_sigmoid<S: Simd, const FUSED: bool>(simd: S, eta: S::f64s) -> S::f64s {
    let one = simd.splat_f64s(1.0);
    let (z, mask) = simd_z_mask::<S, FUSED>(simd, eta);
    let opz = simd.add_f64s(one, z);
    simd.select_f64s(mask, simd.div_f64s(one, opz), simd.div_f64s(z, opz))
}

#[inline]
fn scalar_sigmoid_owned<const FUSED: bool>(eta: f64) -> f64 {
    let z = scalar_z::<FUSED>(eta);
    if eta >= 0.0 {
        1.0 / (1.0 + z)
    } else {
        z / (1.0 + z)
    }
}

/// The batched per-family η-pass. Reads the RAW `eta` in place and leaves it
/// holding `family::clamp_eta`'s projection; fills `prob` with μ and `w` with the
/// floored IRLS working weight `(wᵢ·W_raw).max(WEIGHT_CLAMP)`; fills `z` with the
/// working response `η + r` when non-empty (the GLM IRLS site needs it, the three
/// PIRLS sites do not). Returns `(Σ wᵢ·dᵢ, any-η-outside-the-link's-open-domain)`.
///
/// - `prior_w` empty ⇒ unit prior weights.
/// - `weighted` selects between the two logit forms that exist today: unweighted
///   Bernoulli logit keeps the fused `2·(Σ log1pexp(η) − Σ y·η)` deviance (hence
///   `yeta`, which the caller's η-pass accumulates), weighted binomial goes
///   through `family::dev_resid` because that identity does not hold for
///   aggregated proportions.
#[allow(clippy::too_many_arguments)] // marshals (family, nb_theta, eta, y, prior_w, weighted, yeta, prob, w, z)
pub(crate) fn family_pass(
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
    let n = eta.len();
    debug_assert_eq!(prob.len(), n);
    debug_assert_eq!(w.len(), n);
    debug_assert_eq!(y.len(), n);
    // `z` and `prior_w` are empty when not needed; any other length would put
    // their SIMD head/tail split at a different row than η's and mis-index.
    debug_assert!(z.is_empty() || z.len() == n);
    debug_assert!(prior_w.is_empty() || prior_w.len() == n);

    // Unweighted Bernoulli logit: the pre-existing fused kernel, untouched, so
    // this route stays byte-identical to the fast path it replaces.
    if matches!(
        family,
        Family::Binomial {
            link: BinomialLink::Logit
        }
    ) && !weighted
    {
        let lp_sum = pw_and_log1pexp_sum(eta, prob, w);
        if !z.is_empty() {
            for i in 0..n {
                z[i] = eta[i] + (y[i] - prob[i]) / w[i];
            }
        }
        return (2.0 * (lp_sum - yeta), false);
    }

    // Gaussian never routes here (the OLS/LMM paths own it); keep the scalar
    // statement rather than a SIMD arm that no fit can reach.
    if matches!(family, Family::Gaussian) {
        let mut dev = 0.0;
        for i in 0..n {
            let (mu, w_raw, r) =
                crate::family::irls_weight_and_resid(family, nb_theta, y[i], eta[i]);
            let pw = if prior_w.is_empty() { 1.0 } else { prior_w[i] };
            prob[i] = mu;
            w[i] = (pw * w_raw).max(crate::glm::WEIGHT_CLAMP);
            if !z.is_empty() {
                z[i] = eta[i] + r;
            }
            dev += pw * crate::family::dev_resid(family, nb_theta, y[i], mu);
        }
        return (dev, false);
    }

    let infeasible = pulp::Arch::new().dispatch(FamilyMuWOp::<{ FUSED_DEFAULT }> {
        family,
        nb_theta,
        eta,
        y,
        prior_w,
        prob,
        w,
        z,
    });
    // Deviance fold off the filled μ. Kept scalar and in `family::dev_resid`'s
    // exact form: its `ln`s are outside this module's restricted-domain `ln`
    // (arguments run over `y/μ` on the whole positive line), and holding the
    // deviance arithmetic fixed keeps this change's movement confined to μ.
    let mut dev = 0.0;
    for i in 0..n {
        let pw = if prior_w.is_empty() { 1.0 } else { prior_w[i] };
        dev += pw * crate::family::dev_resid(family, nb_theta, y[i], prob[i]);
    }
    (dev, infeasible)
}

/// In-place `buf[i] = erfc(buf[i])` through the blend kernel — SIMD head +
/// [`scalar_erfc_blend`] tail. Test-only: the fit path reaches the blend through
/// the probit arm of [`family_pass`], where μ is already clamped to
/// `family::PROB_EPS` and so cannot expose the tail the ULP guard measures.
#[cfg(test)]
fn erfc_fill(buf: &mut [f64]) {
    struct Op<'a, const FUSED: bool> {
        buf: &'a mut [f64],
    }
    impl<const FUSED: bool> pulp::WithSimd for Op<'_, FUSED> {
        type Output = ();
        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) {
            let (head, tail) = S::as_mut_simd_f64s(self.buf);
            for v in head.iter_mut() {
                *v = simd_erfc::<S, FUSED>(simd, *v);
            }
            for v in tail.iter_mut() {
                *v = scalar_erfc_blend::<FUSED>(*v);
            }
        }
    }
    pulp::Arch::new().dispatch(Op::<{ FUSED_DEFAULT }> { buf });
}

struct FamilyMuWOp<'a, const FUSED: bool> {
    family: Family,
    nb_theta: f64,
    eta: &'a mut [f64],
    y: &'a [f64],
    prior_w: &'a [f64],
    prob: &'a mut [f64],
    w: &'a mut [f64],
    z: &'a mut [f64],
}

impl<const FUSED: bool> pulp::WithSimd for FamilyMuWOp<'_, FUSED> {
    type Output = bool; // any raw η outside the link's open domain
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> bool {
        let (eh, et) = S::as_mut_simd_f64s(self.eta);
        let (ph, pt) = S::as_mut_simd_f64s(self.prob);
        let (wh, wt) = S::as_mut_simd_f64s(self.w);
        let (zh, zt) = S::as_mut_simd_f64s(self.z);
        let (yh, yt) = S::as_simd_f64s(self.y);
        let (gh, gt) = S::as_simd_f64s(self.prior_w);
        let nb_theta = self.nb_theta;

        let zero = simd.splat_f64s(0.0);
        let one = simd.splat_f64s(1.0);
        let clampv = simd.splat_f64s(crate::glm::WEIGHT_CLAMP);
        let mut bad = zero; // lane-wise count of domain-infeasible rows
        let mut bad_tail = false;

        // Each arm supplies a lane function `η_raw → (η_clamped, μ, W_raw, r)`;
        // the assembly below is common — the prior-weight multiply, the
        // `WEIGHT_CLAMP` floor, and the working response `z = η + r` (skipped
        // when `z` is empty, i.e. at the three PIRLS sites). The SIMD head and
        // the sub-lane scalar tail run the same expressions op-for-op.
        // The lane bodies are expanded TEXTUALLY, not passed as closures. A closure
        // here is not a style choice: the AVX intrinsics pulp calls are
        // `#[target_feature]` functions, which inline only into a caller that
        // carries the same feature. A closure boundary blocks that, and every
        // `_mm256_mul_pd`/`_mm256_fmadd_pd` degrades into a real call — measured
        // 2026-08-23 on the toy28 probit fit at 23.9 s against 2.05 s for the
        // scalar path this replaces, with the intrinsics showing up as their own
        // symbols in `perf report`. Keep the bodies inline.
        macro_rules! run_arm {
            (|$e:ident, $yi:ident| $simd_body:block, |$es:ident, $ys:ident| $scalar_body:block) => {{
                for i in 0..eh.len() {
                    let $e = eh[i];
                    let $yi = yh[i];
                    let (ec, mu, w_raw, r) = $simd_body;
                    eh[i] = ec;
                    ph[i] = mu;
                    let pw = if gh.is_empty() { one } else { gh[i] };
                    wh[i] = simd.max_f64s(simd.mul_f64s(pw, w_raw), clampv);
                    if let Some(slot) = zh.get_mut(i) {
                        *slot = simd.add_f64s(ec, r);
                    }
                }
                for i in 0..et.len() {
                    let $es = et[i];
                    let $ys = yt[i];
                    let (ec, mu, w_raw, r) = $scalar_body;
                    et[i] = ec;
                    pt[i] = mu;
                    let pw = if gt.is_empty() { 1.0 } else { gt[i] };
                    wt[i] = (pw * w_raw).max(crate::glm::WEIGHT_CLAMP);
                    if let Some(slot) = zt.get_mut(i) {
                        *slot = ec + r;
                    }
                }
            }};
        }

        match self.family {
            // μ = Φ(η), dμ/dη = φ(η); general Fisher weight (dμ/dη)²/V(μ).
            Family::Binomial {
                link: BinomialLink::Probit,
            } => {
                let lo = simd.splat_f64s(crate::family::PROB_EPS);
                let hi = simd.splat_f64s(1.0 - crate::family::PROB_EPS);
                run_arm!(
                    |e, yi| {
                        // clamp_eta is the identity for both binomial links.
                        let (mu_raw, dmu) = simd_probit::<S, FUSED>(simd, e);
                        let mu = simd.min_f64s(simd.max_f64s(mu_raw, lo), hi);
                        let v = simd.mul_f64s(mu, simd.sub_f64s(one, mu));
                        let w_raw = simd.div_f64s(simd.mul_f64s(dmu, dmu), v);
                        let r = simd.div_f64s(simd.sub_f64s(yi, mu), dmu);
                        (e, mu, w_raw, r)
                    },
                    |e, yi| {
                        let (mu_raw, dmu) = scalar_probit::<FUSED>(e);
                        let mu =
                            mu_raw.clamp(crate::family::PROB_EPS, 1.0 - crate::family::PROB_EPS);
                        let v = mu * (1.0 - mu);
                        (e, mu, dmu * dmu / v, (yi - mu) / dmu)
                    }
                );
            }
            // μ = 1 − exp(−exp η), dμ/dη = exp(η)·exp(−exp η); general Fisher
            // weight (dμ/dη)²/V(μ). Two `exp`s, both reused: `t = exp η` is the
            // derivative's first factor and `s = exp(−t)` is 1−μ. η carries a real
            // upper clamp here (ln ETA_MAX) — without it exp(exp η) overflows.
            // μ is `1 − s`, not `−expm1(−t)` as in `family::link_inv`: no owned
            // SIMD expm1 exists, so μ below ~1e-8 differs from the scalar
            // statement in relative terms (same class of gap as the owned `exp`,
            // see this file's preamble).
            Family::Binomial {
                link: BinomialLink::Cloglog,
            } => {
                let elo = simd.splat_f64s(-crate::family::ETA_MAX);
                let ehi = simd.splat_f64s(crate::family::ETA_MAX.ln());
                let lo = simd.splat_f64s(crate::family::PROB_EPS);
                let hi = simd.splat_f64s(1.0 - crate::family::PROB_EPS);
                run_arm!(
                    |e, yi| {
                        let ec = simd.min_f64s(simd.max_f64s(e, elo), ehi);
                        let t = simd_exp_reduced::<S, FUSED>(simd, ec);
                        let s = simd_exp_reduced::<S, FUSED>(simd, simd.neg_f64s(t));
                        let mu = simd.min_f64s(simd.max_f64s(simd.sub_f64s(one, s), lo), hi);
                        let dmu = simd.mul_f64s(t, s);
                        let v = simd.mul_f64s(mu, simd.sub_f64s(one, mu));
                        let w_raw = simd.div_f64s(simd.mul_f64s(dmu, dmu), v);
                        (ec, mu, w_raw, simd.div_f64s(simd.sub_f64s(yi, mu), dmu))
                    },
                    |e, yi| {
                        let ec = e.clamp(-crate::family::ETA_MAX, crate::family::ETA_MAX.ln());
                        let t = scalar_exp_reduced::<FUSED>(ec);
                        let s = scalar_exp_reduced::<FUSED>(-t);
                        let mu =
                            (1.0 - s).clamp(crate::family::PROB_EPS, 1.0 - crate::family::PROB_EPS);
                        let dmu = t * s;
                        let v = mu * (1.0 - mu);
                        (ec, mu, dmu * dmu / v, (yi - mu) / dmu)
                    }
                );
            }
            // Weighted binomial logit — the canonical shortcut W_raw = V(μ),
            // matching `family::irls_weight_and_resid`. Unweighted Bernoulli
            // logit never reaches here (`family_pass` routes it to the fused
            // `pw_and_log1pexp_sum` kernel).
            Family::Binomial {
                link: BinomialLink::Logit,
            } => {
                let lo = simd.splat_f64s(crate::family::PROB_EPS);
                let hi = simd.splat_f64s(1.0 - crate::family::PROB_EPS);
                run_arm!(
                    |e, yi| {
                        let mu_raw = simd_sigmoid::<S, FUSED>(simd, e);
                        let mu = simd.min_f64s(simd.max_f64s(mu_raw, lo), hi);
                        let v = simd.mul_f64s(mu, simd.sub_f64s(one, mu));
                        (e, mu, v, simd.div_f64s(simd.sub_f64s(yi, mu), v))
                    },
                    |e, yi| {
                        let mu_raw = scalar_sigmoid_owned::<FUSED>(e);
                        let mu =
                            mu_raw.clamp(crate::family::PROB_EPS, 1.0 - crate::family::PROB_EPS);
                        let v = mu * (1.0 - mu);
                        (e, mu, v, (yi - mu) / v)
                    }
                );
            }
            // Canonical Poisson-log: one `exp`, and W_raw = V(μ) = μ.
            Family::Poisson { .. } => {
                let elo = simd.splat_f64s(-crate::family::ETA_MAX);
                let ehi = simd.splat_f64s(crate::family::ETA_MAX);
                let mfl = simd.splat_f64s(crate::family::MU_FLOOR);
                run_arm!(
                    |e, yi| {
                        let ec = simd.min_f64s(simd.max_f64s(e, elo), ehi);
                        let mu = simd.max_f64s(simd_exp_reduced::<S, FUSED>(simd, ec), mfl);
                        (ec, mu, mu, simd.div_f64s(simd.sub_f64s(yi, mu), mu))
                    },
                    |e, yi| {
                        let ec = e.clamp(-crate::family::ETA_MAX, crate::family::ETA_MAX);
                        let mu = scalar_exp_reduced::<FUSED>(ec).max(crate::family::MU_FLOOR);
                        (ec, mu, mu, (yi - mu) / mu)
                    }
                );
            }
            // Non-canonical log links: `exp(η)` computed once and reused for both
            // μ and dμ/dη — the duplicate `link_inv`/`mu_eta` call this kernel
            // exists to remove. V(μ) is μ² (Gamma) or μ + μ²/θ (NB).
            Family::Gamma {
                link: GammaLink::Log,
            }
            | Family::NegativeBinomial { .. } => {
                let is_nb = matches!(self.family, Family::NegativeBinomial { .. });
                let elo = simd.splat_f64s(-crate::family::ETA_MAX);
                let ehi = simd.splat_f64s(crate::family::ETA_MAX);
                let mfl = simd.splat_f64s(crate::family::MU_FLOOR);
                let th = simd.splat_f64s(nb_theta);
                run_arm!(
                    |e, yi| {
                        let ec = simd.min_f64s(simd.max_f64s(e, elo), ehi);
                        let ex = simd_exp_reduced::<S, FUSED>(simd, ec);
                        let mu = simd.max_f64s(ex, mfl);
                        let msq = simd.mul_f64s(mu, mu);
                        let v = if is_nb {
                            simd.add_f64s(mu, simd.div_f64s(msq, th))
                        } else {
                            msq
                        };
                        let w_raw = simd.div_f64s(simd.mul_f64s(ex, ex), v);
                        (ec, mu, w_raw, simd.div_f64s(simd.sub_f64s(yi, mu), ex))
                    },
                    |e, yi| {
                        let ec = e.clamp(-crate::family::ETA_MAX, crate::family::ETA_MAX);
                        let ex = scalar_exp_reduced::<FUSED>(ec);
                        let mu = ex.max(crate::family::MU_FLOOR);
                        let v = if is_nb {
                            mu + mu * mu / nb_theta
                        } else {
                            mu * mu
                        };
                        (ec, mu, ex * ex / v, (yi - mu) / ex)
                    }
                );
            }
            // Gamma inverse link: no transcendental, but one of the two live
            // `eta_infeasible` cases (μ = 1/η needs η > 0). The infeasible rows are
            // counted lane-wise and reduced once, rather than OR'd per row.
            Family::Gamma {
                link: GammaLink::Inverse,
            } => {
                let elo = simd.splat_f64s(crate::family::MU_FLOOR);
                let ehi = simd.splat_f64s(crate::family::ETA_MAX);
                for i in 0..eh.len() {
                    let raw = eh[i];
                    // `family::eta_infeasible` is `raw <= 0.0`, which is FALSE on
                    // NaN; `less_than_or_equal` keeps that, where negating a
                    // `0 < raw` test would flip it. Counted as 1.0/0.0 per lane and
                    // reduced once — pulp 0.22 exposes no mask reduction.
                    bad = simd.add_f64s(
                        bad,
                        simd.select_f64s(simd.less_than_or_equal_f64s(raw, zero), one, zero),
                    );
                    let ec = simd.min_f64s(simd.max_f64s(raw, elo), ehi);
                    let mu_raw = simd.div_f64s(one, ec);
                    let mu = simd.max_f64s(mu_raw, elo);
                    let dmu = simd.neg_f64s(simd.mul_f64s(mu_raw, mu_raw));
                    let v = simd.mul_f64s(mu, mu);
                    let w_raw = simd.div_f64s(simd.mul_f64s(dmu, dmu), v);
                    eh[i] = ec;
                    ph[i] = mu;
                    let pw = if gh.is_empty() { one } else { gh[i] };
                    wh[i] = simd.max_f64s(simd.mul_f64s(pw, w_raw), clampv);
                    if let Some(slot) = zh.get_mut(i) {
                        *slot = simd.add_f64s(ec, simd.div_f64s(simd.sub_f64s(yh[i], mu), dmu));
                    }
                }
                for i in 0..et.len() {
                    let raw = et[i];
                    bad_tail |= raw <= 0.0;
                    let ec = raw.clamp(crate::family::MU_FLOOR, crate::family::ETA_MAX);
                    let mu_raw = 1.0 / ec;
                    let mu = mu_raw.max(crate::family::MU_FLOOR);
                    let dmu = -(mu_raw * mu_raw);
                    let v = mu * mu;
                    et[i] = ec;
                    pt[i] = mu;
                    let pw = if gt.is_empty() { 1.0 } else { gt[i] };
                    wt[i] = (pw * (dmu * dmu / v)).max(crate::glm::WEIGHT_CLAMP);
                    if let Some(slot) = zt.get_mut(i) {
                        *slot = ec + (yt[i] - mu) / dmu;
                    }
                }
            }
            // IG log link: `exp(η)` once, reused for μ and dμ/dη. V(μ) = μ³, so
            // the general Fisher weight is exp(η)²/μ³.
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            } => {
                let elo = simd.splat_f64s(-crate::family::ETA_MAX);
                let ehi = simd.splat_f64s(crate::family::ETA_MAX);
                let mfl = simd.splat_f64s(crate::family::MU_FLOOR);
                run_arm!(
                    |e, yi| {
                        let ec = simd.min_f64s(simd.max_f64s(e, elo), ehi);
                        let ex = simd_exp_reduced::<S, FUSED>(simd, ec);
                        let mu = simd.max_f64s(ex, mfl);
                        let v = simd.mul_f64s(simd.mul_f64s(mu, mu), mu);
                        let w_raw = simd.div_f64s(simd.mul_f64s(ex, ex), v);
                        (ec, mu, w_raw, simd.div_f64s(simd.sub_f64s(yi, mu), ex))
                    },
                    |e, yi| {
                        let ec = e.clamp(-crate::family::ETA_MAX, crate::family::ETA_MAX);
                        let ex = scalar_exp_reduced::<FUSED>(ec);
                        let mu = ex.max(crate::family::MU_FLOOR);
                        let v = mu * mu * mu;
                        (ec, mu, ex * ex / v, (yi - mu) / ex)
                    }
                );
            }
            // IG 1/μ² link: μ = η^(−1/2), dμ/dη = −μ³/2, V = μ³ ⇒ W_raw = μ³/4.
            // The second live `eta_infeasible` case (η > 0 required), counted
            // lane-wise exactly as the Gamma-inverse arm above does. `sqrt` is
            // IEEE-exact on both paths, so head and tail agree to the bit. Spelled
            // out by hand rather than through `run_arm!`, which has no channel to
            // carry the `bad`/`bad_tail` accumulation out of its lane closures —
            // same reason the Gamma-inverse arm above bypasses the macro.
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            } => {
                let elo = simd.splat_f64s(crate::family::MU_FLOOR);
                let ehi = simd.splat_f64s(crate::family::ETA_MAX);
                let half = simd.splat_f64s(0.5);
                for i in 0..eh.len() {
                    let raw = eh[i];
                    bad = simd.add_f64s(
                        bad,
                        simd.select_f64s(simd.less_than_or_equal_f64s(raw, zero), one, zero),
                    );
                    let ec = simd.min_f64s(simd.max_f64s(raw, elo), ehi);
                    let mu = simd.max_f64s(simd.div_f64s(one, simd.sqrt_f64s(ec)), elo);
                    let mu3 = simd.mul_f64s(simd.mul_f64s(mu, mu), mu);
                    let dmu = simd.neg_f64s(simd.mul_f64s(half, mu3));
                    let w_raw = simd.div_f64s(simd.mul_f64s(dmu, dmu), mu3);
                    eh[i] = ec;
                    ph[i] = mu;
                    let pw = if gh.is_empty() { one } else { gh[i] };
                    wh[i] = simd.max_f64s(simd.mul_f64s(pw, w_raw), clampv);
                    if let Some(slot) = zh.get_mut(i) {
                        *slot = simd.add_f64s(ec, simd.div_f64s(simd.sub_f64s(yh[i], mu), dmu));
                    }
                }
                for i in 0..et.len() {
                    let raw = et[i];
                    bad_tail |= raw <= 0.0;
                    let ec = raw.clamp(crate::family::MU_FLOOR, crate::family::ETA_MAX);
                    let mu = (1.0 / ec.sqrt()).max(crate::family::MU_FLOOR);
                    let mu3 = mu * mu * mu;
                    let dmu = -0.5 * mu3;
                    let w_raw = dmu * dmu / mu3;
                    et[i] = ec;
                    pt[i] = mu;
                    let pw = if gt.is_empty() { 1.0 } else { gt[i] };
                    wt[i] = (pw * w_raw).max(crate::glm::WEIGHT_CLAMP);
                    if let Some(slot) = zt.get_mut(i) {
                        *slot = ec + (yt[i] - mu) / dmu;
                    }
                }
            }
            Family::Gaussian => unreachable!("Gaussian is handled before dispatch"),
        }
        bad_tail || simd.reduce_sum_f64s(bad) > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference: system libm (std f64). Accuracy was established offline against
    // an MPFR oracle (primitives ≤1 ULP); in-repo we re-assert SIMD == scalar-libm
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
        // dense grid straddling the η=0 seam and into both tails
        let n = 20_003usize; // not a multiple of any lane count -> exercises the SIMD tail too
        let eta: Vec<f64> = (0..n).map(|k| -40.0 + 80.0 * k as f64 / n as f64).collect();

        // Per-element accuracy of the kernel formula. `scalar_fused` is bit-identical
        // op-for-op to the SIMD path, so this guards the coefficients directly. The
        // primitives are ≤1 ULP of the true value (proved offline vs an MPFR oracle);
        // in-repo we bound the kernel-vs-libm difference (both ≤1 ULP of truth → ≤2 apart).
        let (mut pmax, mut lpmax) = (0i128, 0i128);
        for &e in &eta {
            let (p, w, lp) = scalar_fused::<{ FUSED_DEFAULT }>(e);
            let (libp, liblp) = libm_fused(e);
            pmax = pmax.max(ulp(p, libp));
            lpmax = lpmax.max(ulp(lp, liblp));
            assert!(w >= crate::glm::WEIGHT_CLAMP && w.is_finite());
        }
        assert!(pmax <= 2, "sigmoid p drifted {pmax} ULP from libm");
        assert!(lpmax <= 2, "log1pexp drifted {lpmax} ULP from libm");

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
        let (mut pmax, mut lpmax) = (0i128, 0i128);
        for &e in &eta {
            let (p, w, lp) = scalar_fused::<false>(e);
            let (libp, liblp) = libm_fused(e);
            pmax = pmax.max(ulp(p, libp));
            lpmax = lpmax.max(ulp(lp, liblp));
            assert!(w >= crate::glm::WEIGHT_CLAMP && w.is_finite());
        }
        assert!(pmax <= 3, "unfused sigmoid p drifted {pmax} ULP from libm");
        assert!(lpmax <= 3, "unfused log1pexp drifted {lpmax} ULP from libm");

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
    fn sigmoid_fill_within_2ulp_of_libm() {
        // Same grid discipline as the fused kernel test; in-place column op.
        let n = 20_003usize;
        let eta: Vec<f64> = (0..n).map(|k| -40.0 + 80.0 * k as f64 / n as f64).collect();
        let mut buf = eta.clone();
        sigmoid_fill(&mut buf);
        let mut pmax = 0i128;
        for i in 0..n {
            let (libp, _) = libm_fused(eta[i]);
            pmax = pmax.max(ulp(buf[i], libp));
            assert!(buf[i].is_finite() && (0.0..=1.0).contains(&buf[i]));
        }
        assert!(pmax <= 2, "sigmoid_fill drifted {pmax} ULP from libm");
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
        // Clamp behaviour at the edges stays finite.
        let mut edge = vec![-1.0e9, 1.0e9];
        exp_fill(&mut edge);
        assert!(edge[0] > 0.0 && edge[1].is_finite());
    }

    #[test]
    fn ln_fill_within_2ulp_of_libm() {
        // The skewed-marginal live domain [e^−EXP_CAP, 1) ≈ [9.5e-4, 1) — the
        // compare-ladder covers k ∈ {−11…−1} and LOG1P_H runs on m−1 ∈ [0,1).
        // Measured 2026-06-11 over this 20,003-pt grid: scalar and SIMD both
        // ≤2 ULP of libm ln — pinned at measured.
        let n = 20_003usize;
        let lo = 9.5e-4f64;
        let us: Vec<f64> = (0..n)
            .map(|k| lo + (1.0 - lo) * k as f64 / n as f64)
            .collect();
        let mut smax = 0i128;
        for &u in &us {
            smax = smax.max(ulp(ln_owned(u), u.ln()));
        }
        assert!(smax <= 2, "ln_owned drifted {smax} ULP from libm");
        let mut buf = us.clone();
        ln_fill(&mut buf);
        let mut vmax = 0i128;
        for i in 0..n {
            vmax = vmax.max(ulp(buf[i], us[i].ln()));
        }
        assert!(vmax <= 2, "ln_fill drifted {vmax} ULP from libm");
        // Clamp totality: below-domain input lands past the −EXP_CAP censor;
        // u = 1.0 maps to ln(1 − 2^−53), not −0.0.
        assert!(ln_owned(1.0e-9) < -6.96);
        assert!(ln_owned(0.0) < -6.96);
        assert!(ln_owned(1.0) < 0.0 && ln_owned(1.0) > -3.0e-16);
    }

    #[test]
    fn phi_fill_bit_identical_to_scalar_phi() {
        // phi_fill mirrors scalar_phi op-for-op (A&S poly in plain
        // mul/add — fma-policy-neutral — plus the shared owned exp), so head and
        // tail must agree to the bit, the simd_fused/scalar_fused discipline.
        let n = 20_003usize;
        let z: Vec<f64> = (0..n).map(|k| -9.0 + 18.0 * k as f64 / n as f64).collect();
        let mut buf = z.clone();
        phi_fill(&mut buf);
        for i in 0..n {
            assert_eq!(
                buf[i].to_bits(),
                scalar_phi(z[i]).to_bits(),
                "phi_fill diverged from scalar phi at z={}",
                z[i]
            );
        }
    }

    #[test]
    fn phi_hp_full_double_precision() {
        // Reference Φ values (R `pnorm`, 16 figs). The probit FD-Hessian SE needs
        // Φ to ~machine precision — scalar_phi's ~7.5e-8 is far too coarse here.
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
        // SIMD head vs its scalar twin: bit-identical, the `phi_fill` discipline.
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
            let run = |lo: usize,
                       hi: usize,
                       e: &mut [f64],
                       p: &mut [f64],
                       w: &mut [f64],
                       z: &mut [f64]| {
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
            let (mut e, mut p, mut w, mut z) =
                (eta.to_vec(), vec![0.0; n], vec![0.0; n], vec![0.0; n]);
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
    fn weight_clamped_and_finite() {
        let eta: Vec<f64> = vec![-50.0, -10.0, -1e-9, 0.0, 1e-9, 10.0, 50.0, 1e3];
        let mut p = vec![0.0; eta.len()];
        let mut w = vec![0.0; eta.len()];
        pw_and_log1pexp_sum(&eta, &mut p, &mut w);
        for i in 0..eta.len() {
            assert!(p[i].is_finite() && (0.0..=1.0).contains(&p[i]));
            assert!(w[i] >= crate::glm::WEIGHT_CLAMP && w[i].is_finite());
        }
    }
}
