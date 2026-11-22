use super::{digamma, trigamma, Dual, HyperDual};
use crate::scalar::Scalar;

/// Closed forms, not a reimplementation: ψ(1) = −γ, ψ(½) = −γ − 2ln2,
/// ψ(2) = 1 − γ (A&S 6.3.2–6.3.4); ψ′(1) = π²/6, ψ′(½) = π²/2,
/// ψ′(2) = π²/6 − 1 (A&S 6.4.3–6.4.5). Every point here goes through the
/// recurrence to `x ≥ 14`; the `x = 30` point of
/// `digamma_is_the_derivative_of_ln_gamma` reaches the series directly.
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

/// ψ = (lnΓ)′: `digamma` against a central difference of the crate's
/// `ln_gamma`, which shares no code with it. `h = 1e-6·max(x, 1)` puts the
/// FD floor near 1e-10; the band is 1e-7. `x = 30` skips the recurrence.
#[test]
fn digamma_is_the_derivative_of_ln_gamma() {
    let ln_gamma = crate::simd_transcendental::ln_gamma;
    for &x in &[0.25_f64, 0.3, 1.0, 3.0, 12.0, 30.0] {
        let h = 1e-6 * x.max(1.0);
        let fd = (ln_gamma(x + h) - ln_gamma(x - h)) / (2.0 * h);
        assert!((digamma(x) - fd).abs() <= 1e-7, "x = {x}");
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

/// Every `Scalar` method's derivative lane against a central difference of the
/// same method at f64 — the twin the value part binds to, so a mismatch is a
/// chain-rule error and not a difference between two implementations of the
/// same function. h = 1e-6 relative puts the FD truncation and round-off floor
/// near 1e-10, so the band is 1e-8. No point below 0.25: `h` is absolute
/// (1e-6) there, and the central-FD truncation `(h²/6)·f'''` for `ln` and
/// `sqrt` at x ≤ 1e-3 exceeds the bands by orders.
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
    let pts_pos = [0.25_f64, 0.3, 1.0, 3.0, 12.0, 30.0];

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
        assert!(
            (Scalar::sigmoid(a).d[0] - fd(|t| Scalar::sigmoid(t), x)).abs()
                <= 1e-8 * fd(|t| Scalar::sigmoid(t), x).abs().max(1.0)
        );
        assert!(
            (Scalar::log1pexp(a).d[0] - fd(|t| Scalar::log1pexp(t), x)).abs()
                <= 1e-8 * fd(|t| Scalar::log1pexp(t), x).abs().max(1.0)
        );
        assert!(
            (Scalar::probit_cdf(a).d[0] - fd(|t| Scalar::probit_cdf(t), x)).abs()
                <= 1e-8 * fd(|t| Scalar::probit_cdf(t), x).abs().max(1.0)
        );
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
        assert_eq!(Scalar::ln(a).v, Scalar::ln(x));
        assert_eq!(Scalar::sqrt(a).v, Scalar::sqrt(x));
    }
    // `ln_1p` (the probit tail rows, family.rs) needs x > −1.
    for &x in &[-0.7_f64, 0.0, 0.4, 1.0, 4.0] {
        let a = Dual::<1> { v: x, d: [1.0] };
        let want = fd(|t| Scalar::ln_1p(t), x);
        assert!((Scalar::ln_1p(a).d[0] - want).abs() <= 1e-8 * want.abs().max(1.0));
        assert_eq!(Scalar::ln_1p(a).v, Scalar::ln_1p(x));
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

/// f'' from HyperDual against a central difference of f' from Dual, at points like
/// those the chain-rule FD test above uses. Differencing the analytic first
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
    }
    for &x in &[-0.7_f64, 0.4, 1.0, 4.0] {
        check!(ln_1p, x, 1e-7);
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
