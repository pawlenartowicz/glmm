mod common;

use std::collections::HashMap;

use common::{f64s, factor, field, fit_args, rec, sleepstudy, text};
use glmm_bind::{call, Record, Value};

#[test]
fn unknown_function_is_invalid_argument() {
    let e = call("nope", Record::new()).unwrap_err();
    assert_eq!(e.code, "invalid_argument");
    assert!(
        e.message.contains("unknown function 'nope'"),
        "{}",
        e.message
    );
}

#[test]
fn unexpected_and_missing_arguments_are_named() {
    let r = rec(vec![("formula", text("y ~ x")), ("bogus", Value::Null)]);
    let e = call("fit", r).unwrap_err();
    assert_eq!(e.code, "invalid_argument");
    assert!(
        e.message.contains("unexpected argument 'bogus'"),
        "{}",
        e.message
    );
    let e = call("fit", Record::new()).unwrap_err();
    assert!(
        e.message.contains("missing required argument"),
        "{}",
        e.message
    );
}

#[test]
fn a_duplicate_argument_is_refused() {
    let r = rec(vec![("formula", text("y ~ x")), ("formula", text("y ~ 1"))]);
    let e = call("fit", r).unwrap_err();
    assert!(e.message.contains("'formula' twice"), "{}", e.message);
}

/// Subject's levels in an order no sort gives, so a level order kept to the
/// engine shows in the result.
fn shuffled_subject() -> (Vec<String>, Vec<u32>, Vec<String>) {
    let (_, _, subject) = sleepstudy();
    let (mut levels, codes) = factor(&subject);
    let original = levels.clone();
    levels.reverse();
    let n = levels.len() as u32;
    let codes = codes.into_iter().map(|c| n - 1 - c).collect();
    (levels, codes, original)
}

#[test]
fn fit_through_call_matches_run_fit() {
    let (reaction, days, _) = sleepstudy();
    let (levels, codes, _) = shuffled_subject();
    let formula = "Reaction ~ Days + (Days | Subject)";

    let numeric = rec(vec![
        ("Reaction", Value::Vec(reaction.clone())),
        ("Days", Value::Vec(days.clone())),
    ]);
    let factors = rec(vec![(
        "Subject",
        Value::Factor {
            levels: levels.clone(),
            codes: codes.iter().map(|&c| c as i32).collect(),
        },
    )]);
    let out = call("fit", fit_args(formula, numeric, factors, 0)).unwrap();

    let direct = glmm::orchestrate::run_fit(
        formula,
        HashMap::from([
            ("Reaction".to_string(), reaction),
            ("Days".to_string(), days),
        ]),
        HashMap::from([("Subject".to_string(), (levels.clone(), codes))]),
        "gaussian",
        "identity",
        "hessian",
        1,
        None,
        None,
        None,
        None,
        true,
        true,
        None,
        "auto",
        None,
        None,
    )
    .unwrap();
    assert_eq!(f64s(field(&out, "beta")), direct.beta.as_slice());
    assert_eq!(f64s(field(&out, "se")), direct.se.as_slice());
    assert!(matches!(field(&out, "loglik"), Value::F64(v) if *v == direct.loglik));

    // The caller's level order reaches the engine (the reference level survives).
    let block = match field(&out, "ranef_blocks") {
        Value::List(b) => &b[0],
        other => panic!("{other:?}"),
    };
    assert!(matches!(field(block, "levels"), Value::StrVec(l) if *l == levels));
}

#[test]
fn a_missing_factor_code_is_refused() {
    let numeric = rec(vec![("y", Value::Vec(vec![1.0, 2.0, 3.0, 4.0]))]);
    let factors = rec(vec![(
        "g",
        Value::Factor {
            levels: vec!["a".into(), "b".into()],
            codes: vec![0, 1, -1, 0],
        },
    )]);
    let e = call("fit", fit_args("y ~ g", numeric, factors, 0)).unwrap_err();
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("factor column \"g\""), "{}", e.message);
}

/// `weight_power` and `se_type` reach the engine through `fit` and
/// `compare`: a value outside them is refused naming the argument, a fixed
/// or estimated power comes back as the record's `weight_power`, and the
/// engine's refusal of `se_type = "robust"` (on an ML mixed fit) is a
/// `fit_error`.
#[test]
fn weight_power_and_se_type_reach_fit_and_compare() {
    let (reaction, days, _) = sleepstudy();
    let n = reaction.len();
    let numeric = || {
        rec(vec![
            ("Reaction", Value::Vec(reaction.clone())),
            ("Days", Value::Vec(days.clone())),
        ])
    };
    let weights = Value::Vec((0..n).map(|i| 1.0 + (i % 4) as f64).collect());
    let fit = |extra: Vec<(&str, Value)>| {
        let mut r = fit_args("Reaction ~ Days", numeric(), Record::new(), 0);
        r.push("weights", weights.clone());
        for (k, v) in extra {
            r.push(k, v);
        }
        call("fit", r)
    };
    let e = fit(vec![("weight_power", text("Estimate"))]).unwrap_err();
    assert_eq!(e.code, "invalid_argument");
    assert!(
        e.message
            .starts_with(r#"weight_power must be a finite number or "estimate""#),
        "{}",
        e.message
    );
    let e = fit(vec![("se_type", text("CR2"))]).unwrap_err();
    assert!(e
        .message
        .starts_with(r#"se_type must be "model" or "robust""#));

    let out = fit(vec![("weight_power", Value::F64(0.5))]).unwrap();
    assert!(matches!(field(&out, "weight_power"), Value::F64(v) if *v == 0.5));
    assert_eq!(field(&out, "se_type"), &text("model"));
    let out = fit(vec![("weight_power", text("estimate"))]).unwrap();
    assert!(matches!(field(&out, "weight_power"), Value::F64(v) if v.is_finite() && *v != 1.0));

    let (levels, codes) = factor(&sleepstudy().2);
    let subject = rec(vec![(
        "Subject",
        Value::Factor {
            levels,
            codes: codes.iter().map(|&c| c as i32).collect(),
        },
    )]);
    let compare = rec(vec![
        ("f0", text("Reaction ~ 1 + (1 | Subject)")),
        ("f1", text("Reaction ~ Days + (1 | Subject)")),
        ("numeric", Value::Record(numeric())),
        ("factors", Value::Record(subject)),
        ("family", text("gaussian")),
        ("link", text("identity")),
        ("nagq", Value::I64(1)),
        ("weights", weights.clone()),
        ("se_type", text("robust")),
        ("row_base", Value::I64(0)),
    ]);
    let e = call("compare", compare).unwrap_err();
    assert_eq!(e.code, "fit_error");
    assert!(e.message.contains("se_type"), "{}", e.message);
}

/// A robust fit's record: `se_type` "robust", `df_method` "bell_mccaffrey"
/// without a grouping factor and "pustejovsky_tipton" with one, the model
/// covariance, and the pieces with the null directions' clusters as an
/// index; a model fit carries neither. The default weighted call is robust.
#[test]
fn a_robust_fit_record_carries_its_pieces() {
    let (reaction, days, subject) = sleepstudy();
    let n = reaction.len();
    let numeric = || {
        rec(vec![
            ("Reaction", Value::Vec(reaction.clone())),
            ("Days", Value::Vec(days.clone())),
        ])
    };
    let (levels, codes) = factor(&subject);
    let factors = || {
        rec(vec![(
            "Subject",
            Value::Factor {
                levels: levels.clone(),
                codes: codes.iter().map(|&c| c as i32).collect(),
            },
        )])
    };
    let weights = Value::Vec((0..n).map(|i| 1.0 + (i % 4) as f64).collect());
    let fit = |formula: &str, extra: Vec<(&str, Value)>| {
        let mut r = fit_args(formula, numeric(), factors(), 0);
        r.push("weights", weights.clone());
        for (k, v) in extra {
            r.push(k, v);
        }
        call("fit", r).unwrap()
    };
    let p = 2;
    for (formula, method, extra) in [
        (
            "Reaction ~ Days",
            "bell_mccaffrey",
            vec![
                ("weight_power", Value::F64(1.0)),
                ("se_type", text("robust")),
            ],
        ),
        ("Reaction ~ Days", "bell_mccaffrey", vec![]),
        (
            "Reaction ~ Days + (1 | Subject)",
            "pustejovsky_tipton",
            vec![],
        ),
    ] {
        let out = fit(formula, extra);
        assert_eq!(field(&out, "se_type"), &text("robust"), "{formula}");
        assert_eq!(field(&out, "df_method"), &text(method), "{formula}");
        assert!(f64s(field(&out, "coef_df")).iter().all(|d| d.is_finite()));
        assert!(f64s(field(&out, "coef_p")).iter().all(|d| d.is_finite()));
        match field(&out, "model_vcov") {
            Value::List(rows) => assert_eq!(rows.len(), p),
            other => panic!("{other:?}"),
        }
        let robust = field(&out, "robust");
        assert!(matches!(field(robust, "null_cluster"), Value::Index(_)));
        let er = f64s(field(robust, "er"));
        let clusters = if method == "bell_mccaffrey" {
            n
        } else {
            levels.len()
        };
        assert_eq!(er.len(), p * clusters, "{formula}");
    }
    let model = fit("Reaction ~ Days", vec![("weight_power", Value::F64(1.0))]);
    assert_eq!(field(&model, "df_method"), &text("residual"));
    assert_eq!(field(&model, "robust"), &Value::Null);
    assert_eq!(field(&model, "model_vcov"), &Value::List(vec![]));
}
