//! Data and record helpers shared by the integration tests.

#![allow(dead_code)]

use glmm_bind::{Record, Value};

/// sleepstudy as `(reaction, days, subject)`, Subject's levels in first-seen order.
pub fn sleepstudy() -> (Vec<f64>, Vec<f64>, Vec<String>) {
    let csv = include_str!("../../../validation/data/empirical/sleepstudy.csv");
    let (mut reaction, mut days, mut subject) = (Vec::new(), Vec::new(), Vec::new());
    for line in csv.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        reaction.push(f[0].parse().unwrap());
        days.push(f[1].parse().unwrap());
        subject.push(f[2].trim_matches('"').to_string());
    }
    (reaction, days, subject)
}

/// `(levels, codes)` of `labels`, levels in first-seen order.
pub fn factor(labels: &[String]) -> (Vec<String>, Vec<u32>) {
    let mut levels: Vec<String> = Vec::new();
    let codes = labels
        .iter()
        .map(|l| match levels.iter().position(|v| v == l) {
            Some(i) => i as u32,
            None => {
                levels.push(l.clone());
                (levels.len() - 1) as u32
            }
        })
        .collect();
    (levels, codes)
}

pub fn rec(fields: Vec<(&str, Value)>) -> Record {
    let mut r = Record::new();
    for (k, v) in fields {
        r.push(k, v);
    }
    r
}

pub fn text(s: &str) -> Value {
    Value::Str(s.to_string())
}

/// The arguments every `fit` call in these tests shares, the data aside.
pub fn fit_args(formula: &str, numeric: Record, factors: Record, row_base: i64) -> Record {
    rec(vec![
        ("formula", text(formula)),
        ("numeric", Value::Record(numeric)),
        ("factors", Value::Record(factors)),
        ("family", text("gaussian")),
        ("link", text("identity")),
        ("wald_se", text("hessian")),
        ("nagq", Value::I64(1)),
        ("satterthwaite", Value::Bool(true)),
        ("condvar", Value::Bool(true)),
        ("row_base", Value::I64(row_base)),
    ])
}

pub fn field<'a>(v: &'a Value, name: &str) -> &'a Value {
    match v {
        Value::Record(r) => r.get(name).unwrap_or_else(|| panic!("no field {name:?}")),
        other => panic!("not a record: {other:?}"),
    }
}

pub fn f64s(v: &Value) -> &[f64] {
    match v {
        Value::Vec(x) => x,
        other => panic!("not a Vec: {other:?}"),
    }
}
