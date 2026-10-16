//! The p-value tails and quantile wrappers against R's `pt`, `pf`, `pchisq`,
//! `qt` and `qnorm`, frozen in `tests/fixtures/tail_values.json` by
//! `tests/fixtures/gen_tail_values.R`, plus the NaN, `≤ 0`, infinite and
//! endpoint rules, which the fixture does not cover.

use super::{chisq_upper_p, f_upper_p, normal_quantile, t_quantile, t_two_sided_p};
use serde::Deserialize;

/// Relative band against the frozen R values. The fixture stores 15
/// significant digits (at most 5e-15 relative rounding).
const REL: f64 = 1e-12;

#[derive(Deserialize)]
struct TPoint {
    t: f64,
    df: f64,
    p: f64,
}

#[derive(Deserialize)]
struct FPoint {
    f: f64,
    num_df: f64,
    den_df: f64,
    p: f64,
}

#[derive(Deserialize)]
struct ChisqPoint {
    x: f64,
    df: f64,
    p: f64,
}

#[derive(Deserialize)]
struct QtPoint {
    p: f64,
    df: f64,
    q: f64,
}

#[derive(Deserialize)]
struct QnormPoint {
    p: f64,
    q: f64,
}

#[derive(Deserialize)]
struct TailFixture {
    t_two_sided: Vec<TPoint>,
    f_upper: Vec<FPoint>,
    chisq_upper: Vec<ChisqPoint>,
    qt: Vec<QtPoint>,
    qnorm: Vec<QnormPoint>,
}

fn load_tail_fixture() -> TailFixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/tail_values.json"
    );
    let s = std::fs::read_to_string(path).expect("read tail fixture");
    serde_json::from_str(&s).expect("parse tail fixture")
}

fn assert_rel(got: f64, want: f64, what: &str) {
    let err = (got - want).abs();
    assert!(
        err <= REL * want.abs(),
        "{what}: got {got:e}, R {want:e}, relative error {:e}",
        err / want.abs()
    );
}

#[test]
fn tails_match_r() {
    let fx = load_tail_fixture();
    for pt in &fx.t_two_sided {
        let what = format!("t_two_sided_p({}, {})", pt.t, pt.df);
        assert_rel(t_two_sided_p(pt.t, pt.df), pt.p, &what);
    }
    for pt in &fx.f_upper {
        let what = format!("f_upper_p({}, {}, {})", pt.f, pt.num_df, pt.den_df);
        assert_rel(f_upper_p(pt.f, pt.num_df, pt.den_df), pt.p, &what);
    }
    for pt in &fx.chisq_upper {
        let what = format!("chisq_upper_p({}, {})", pt.x, pt.df);
        assert_rel(chisq_upper_p(pt.x, pt.df), pt.p, &what);
    }
}

#[test]
fn quantiles_match_r() {
    let fx = load_tail_fixture();
    for pt in &fx.qt {
        let what = format!("t_quantile({}, {})", pt.p, pt.df);
        assert_rel(t_quantile(pt.p, pt.df), pt.q, &what);
    }
    for pt in &fx.qnorm {
        let what = format!("normal_quantile({})", pt.p);
        assert_rel(normal_quantile(pt.p), pt.q, &what);
    }
}

/// A NaN statistic, and a NaN, zero, negative or infinite df, give NaN from
/// every tail (each df slot of the F tail separately).
#[test]
fn nan_statistic_and_bad_df_give_nan() {
    for stat in [f64::NAN, 2.0] {
        for df in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
            assert!(t_two_sided_p(stat, df).is_nan(), "t({stat}, {df})");
            assert!(chisq_upper_p(stat, df).is_nan(), "chisq({stat}, {df})");
            assert!(f_upper_p(stat, df, 3.0).is_nan(), "f({stat}, {df}, 3)");
            assert!(f_upper_p(stat, 3.0, df).is_nan(), "f({stat}, 3, {df})");
        }
    }
    assert!(t_two_sided_p(f64::NAN, 3.0).is_nan());
    assert!(chisq_upper_p(f64::NAN, 3.0).is_nan());
    assert!(f_upper_p(f64::NAN, 3.0, 3.0).is_nan());
}

/// `t = 0`, `f ≤ 0` and `x ≤ 0` give 1; an infinite statistic gives 0
/// (`−∞` for F and χ² falls under the `≤ 0` rule).
#[test]
fn boundary_statistics() {
    assert_eq!(t_two_sided_p(0.0, 3.0), 1.0);
    assert_eq!(t_two_sided_p(f64::INFINITY, 3.0), 0.0);
    assert_eq!(t_two_sided_p(f64::NEG_INFINITY, 3.0), 0.0);
    for s in [0.0, -1.0, f64::NEG_INFINITY] {
        assert_eq!(f_upper_p(s, 2.0, 3.0), 1.0, "f = {s}");
        assert_eq!(chisq_upper_p(s, 2.0), 1.0, "x = {s}");
    }
    assert_eq!(f_upper_p(f64::INFINITY, 2.0, 3.0), 0.0);
    assert_eq!(chisq_upper_p(f64::INFINITY, 2.0), 0.0);
}

/// A NaN `p` or one outside `[0, 1]` gives NaN from both wrappers, the df
/// rules give NaN from `t_quantile`, and `p = 0`, `p = 1` give `−∞`, `+∞`.
#[test]
fn quantile_domain_and_endpoints() {
    for p in [f64::NAN, -0.1, 1.1] {
        assert!(t_quantile(p, 3.0).is_nan(), "qt({p}, 3)");
        assert!(normal_quantile(p).is_nan(), "qnorm({p})");
    }
    for df in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
        assert!(t_quantile(0.975, df).is_nan(), "qt(0.975, {df})");
    }
    assert_eq!(t_quantile(0.0, 3.0), f64::NEG_INFINITY);
    assert_eq!(t_quantile(1.0, 3.0), f64::INFINITY);
    assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
    assert_eq!(normal_quantile(1.0), f64::INFINITY);
}
