mod common;

use common::{field, fit_args, rec, text};
use glmm_bind::{call, registry, Port, Record, Value};

/// `call` dispatches on the first entry of a name, so a second one would be
/// unreachable.
#[test]
fn registry_names_are_unique() {
    let mut names: Vec<_> = registry().iter().map(|f| f.name).collect();
    names.sort_unstable();
    let n = names.len();
    names.dedup();
    assert_eq!(names.len(), n, "{names:?}");
}

/// An OLS fit of y on x over six rows, with a zero weight at row 3.
fn zero_weight_fit(row_index: Option<Vec<i64>>, row_base: i64) -> Record {
    let numeric = rec(vec![
        ("y", Value::Vec(vec![1.0, 2.5, 2.0, 4.0, 5.5, 5.0])),
        ("x", Value::Vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
    ]);
    let mut r = fit_args("y ~ x", numeric, Record::new(), row_base);
    r.push("weights", Value::Vec(vec![1.0, 1.0, 1.0, 0.0, 1.0, 1.0]));
    if let Some(ri) = row_index {
        r.push("row_index", Value::IntVec(ri));
    }
    r
}

#[test]
fn fit_error_carries_the_callers_row() {
    for (row_index, base, shown) in [
        (Some(vec![10, 11, 12, 13, 14, 15]), 1, 13),
        (None, 1, 4),
        (None, 0, 3),
    ] {
        let e = call(Port::Python, "fit", zero_weight_fit(row_index, base)).unwrap_err();
        assert_eq!(e.code, "fit_error");
        assert_eq!(e.row, Some(shown), "{}", e.message);
        assert!(e.message.contains(&format!("row {shown}")), "{}", e.message);
    }
}

fn ols(y: Vec<f64>, formula: &str) -> Value {
    let numeric = rec(vec![
        ("y", Value::Vec(y)),
        (
            "x",
            Value::Vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        ),
        (
            "z",
            Value::Vec(vec![0.3, -1.0, 0.8, 0.1, -0.4, 1.2, 0.0, 0.5]),
        ),
    ]);
    call(
        Port::Python,
        "fit",
        fit_args(formula, numeric, Record::new(), 0),
    )
    .unwrap()
}

/// One side of `compare_fits` from a fit record, as a port's stored fit keeps it.
fn side(fit: &Value, formula: &str) -> Value {
    let keep = [
        "loglik",
        "df",
        "nobs",
        "reml",
        "objective",
        "objective_nodes",
        "converged",
        "varcorr",
        "re_group_names",
        "re_group_terms",
        "y",
        "weights",
        "offset",
        "design_summary",
        "rss",
        "vcov",
        "aliased",
    ];
    let mut r = rec(vec![
        ("formula", text(formula)),
        ("family", text("gaussian")),
        ("link", text("identity")),
        ("dispersion_held", Value::Null),
        ("coef_names", field(fit, "names").clone()),
        ("coef", field(fit, "beta").clone()),
    ]);
    for k in keep {
        r.push(k, field(fit, k).clone());
    }
    Value::Record(r)
}

fn compare_fits(port: Port, m0: Value, m1: Value) -> Result<Value, glmm_bind::BindError> {
    call(port, "compare_fits", rec(vec![("m0", m0), ("m1", m1)]))
}

#[test]
fn a_compare_refusal_points_to_each_ports_compare() {
    let y = vec![1.0, 2.5, 2.0, 4.0, 5.5, 5.0, 7.5, 7.0];
    let (f0, f1) = ("y ~ x", "y ~ x + z");
    let small = ols(y.clone(), f0);
    let large = ols(y.clone(), f1);
    let c = compare_fits(Port::Python, side(&small, f0), side(&large, f1)).unwrap();
    assert!(matches!(field(&c, "test"), Value::Str(t) if t == "f"));
    assert!(matches!(field(&c, "larger"), Value::I64(1)));
    let p = field(&c, "p_value");
    assert!(
        matches!(p, Value::F64(v) if (0.0..=1.0).contains(v)),
        "{p:?}"
    );

    let other_y: Vec<f64> = y.iter().map(|v| v + 1.0).collect();
    let moved = ols(other_y, f1);
    for (port, call) in [
        (Port::Python, "glmm.compare(data, f0, f1)"),
        (Port::R, "compare(data, f0, f1)"),
    ] {
        let e = compare_fits(port, side(&small, f0), side(&moved, f1)).unwrap_err();
        assert_eq!(e.code, "compare_error");
        let tail = format!(" Or use {call}, which fits both to the same data.");
        assert!(e.message.ends_with(&tail), "{}", e.message);
    }
}

#[test]
fn nsim_out_of_range_gets_the_range_message_whatever_its_sign() {
    for (nsim, shown) in [(-1.0, "-1"), (0.0, "0"), (2e16, "20000000000000000")] {
        let args = rec(vec![
            ("m0", Value::Record(Record::new())),
            ("m1", Value::Record(Record::new())),
            ("nsim", Value::F64(nsim)),
        ]);
        let e = call(Port::Python, "compare_fits", args).unwrap_err();
        assert_eq!(
            e.message,
            format!("nsim must be a whole number from 1 to 1000000, got {shown}")
        );
    }
}
