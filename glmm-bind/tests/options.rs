//! Every option is parsed and checked in bind: the accepted forms reach the
//! fit, and a refusal names the argument as the calling port spells it.

mod common;

use common::{f64s, factor, field, fit_args, rec, sleepstudy, text};
use glmm_bind::{call, BindError, Port, Record, Value};

const BOTH: [Port; 2] = [Port::Python, Port::R];

fn numeric(y: Vec<f64>) -> Record {
    let (_, days, _) = sleepstudy();
    rec(vec![("y", Value::Vec(y)), ("Days", Value::Vec(days))])
}

fn subject() -> Record {
    let (levels, codes) = factor(&sleepstudy().2);
    rec(vec![(
        "Subject",
        Value::Factor {
            levels,
            codes: codes.iter().map(|&c| c as i32).collect(),
        },
    )])
}

fn reaction() -> Vec<f64> {
    sleepstudy().0
}

fn counts() -> Vec<f64> {
    reaction().iter().map(|r| (r / 25.0).round()).collect()
}

fn binary() -> Vec<f64> {
    reaction()
        .iter()
        .map(|&r| f64::from(u8::from(r > 300.0)))
        .collect()
}

/// `fit` of `formula` on `y`, Days and Subject, family `family` with its
/// default link, the other options `extra` (set over `fit_args`' values).
fn fit(
    port: Port,
    formula: &str,
    y: Vec<f64>,
    family: &str,
    extra: Vec<(&str, Value)>,
) -> Result<Value, BindError> {
    let mut r = fit_args(formula, numeric(y), subject(), 0);
    r.set("family", text(family));
    r.set("link", Value::Null);
    for (k, v) in extra {
        r.set(k, v);
    }
    call(port, "fit", r)
}

fn compare(port: Port, f0: &str, f1: &str, extra: Vec<(&str, Value)>) -> Result<Value, BindError> {
    let mut r = rec(vec![
        ("f0", text(f0)),
        ("f1", text(f1)),
        ("numeric", Value::Record(numeric(reaction()))),
        ("factors", Value::Record(subject())),
        ("family", text("gaussian")),
        ("nagq", Value::I64(1)),
        ("row_base", Value::I64(0)),
    ]);
    for (k, v) in extra {
        r.set(k, v);
    }
    call(port, "compare", r)
}

fn refused(r: Result<Value, BindError>) -> String {
    let e = r.unwrap_err();
    assert_eq!(e.code, "invalid_argument", "{}", e.message);
    e.message
}

#[test]
fn family_names_ignore_case_and_take_rs_dotted_names() {
    for (family, y, name, link) in [
        ("Gaussian", reaction(), "gaussian", "identity"),
        ("negative.binomial", counts(), "negativebinomial", "log"),
        ("inverse.gaussian", reaction(), "inversegaussian", "log"),
    ] {
        let out = fit(Port::Python, "y ~ Days", y, family, vec![]).unwrap();
        assert_eq!(field(&out, "family"), &text(name));
        assert_eq!(field(&out, "link"), &text(link));
    }
    let e = refused(fit(Port::R, "y ~ Days", reaction(), "gausian", vec![]));
    assert_eq!(
        e,
        "unknown family 'gausian'; expected one of gaussian, binomial, poisson, gamma, \
         negativebinomial, inversegaussian"
    );
    let e = refused(fit(
        Port::Python,
        "y ~ Days",
        counts(),
        "Poisson",
        vec![("link", text("logit"))],
    ));
    assert_eq!(
        e,
        "family 'poisson' does not support link 'logit'; expected one of log"
    );
    let e = refused(fit(
        Port::Python,
        "y ~ Days",
        reaction(),
        "Gaussian",
        vec![("family", Value::I64(1))],
    ));
    assert_eq!(e, "family must be a string, got 1");
    let ig = "family 'inversegaussian' is GLM-only: random-effect terms are not supported";
    let e = refused(fit(
        Port::Python,
        "y ~ Days + (1 | Subject)",
        reaction(),
        "inversegaussian",
        vec![],
    ));
    assert_eq!(e, ig);

    let out = compare(
        Port::R,
        "y ~ 1 + (1 | Subject)",
        "y ~ Days + (1 | Subject)",
        vec![("family", text("GAUSSIAN"))],
    )
    .unwrap();
    assert_eq!(field(field(&out, "fit1"), "family"), &text("gaussian"));
    assert_eq!(field(field(&out, "fit1"), "link"), &text("identity"));
    let e = refused(compare(
        Port::R,
        "y ~ 1",
        "y ~ Days + (1 | Subject)",
        vec![("family", text("inverse.gaussian"))],
    ));
    assert_eq!(e, ig);
}

#[test]
fn nagq_is_an_odd_whole_number_up_to_max_nagq() {
    for (nagq, used) in [
        (Value::I64(1), 1),
        (Value::I64(3), 3),
        (Value::F64(25.0), 25),
    ] {
        let out = fit(
            Port::R,
            "y ~ Days + (1 | Subject)",
            binary(),
            "binomial",
            vec![("nagq", nagq)],
        )
        .unwrap();
        assert_eq!(field(&out, "nagq"), &Value::I64(used));
    }
    for (port, nagq, shown) in [
        (
            Port::Python,
            Value::I64(2),
            "nagq must be an odd integer in 1..=25, got 2",
        ),
        (
            Port::R,
            Value::F64(27.0),
            "nAGQ must be an odd integer in 1..=25, got 27",
        ),
        (
            Port::Python,
            text("3"),
            r#"nagq must be an odd integer in 1..=25, got the string "3""#,
        ),
    ] {
        let e = refused(fit(
            port,
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![("nagq", nagq)],
        ));
        assert_eq!(e, shown);
    }
}

#[test]
fn a_bool_in_a_refusal_is_spelled_by_port() {
    for (port, shown) in [
        (
            Port::Python,
            "nagq must be an odd integer in 1..=25, got True",
        ),
        (Port::R, "nAGQ must be an odd integer in 1..=25, got TRUE"),
    ] {
        let e = refused(fit(
            port,
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![("nagq", Value::BoolVec(vec![true]))],
        ));
        assert_eq!(e, shown);
    }
    let e = refused(fit(
        Port::R,
        "y ~ Days",
        reaction(),
        "gamma",
        vec![("dispersion", Value::Bool(false))],
    ));
    assert_eq!(
        e,
        "dispersion must be NULL, 'estimate', or a number, got FALSE"
    );
}

#[test]
fn reml_satterthwaite_and_condvar_are_bools() {
    let mixed = "y ~ Days + (1 | Subject)";
    let lmm = |port, extra| fit(port, mixed, reaction(), "gaussian", extra);
    for (v, reml) in [
        (Value::Null, true),
        (Value::Bool(true), true),
        (Value::BoolVec(vec![false]), false),
    ] {
        let out = lmm(Port::R, vec![("reml", v)]).unwrap();
        assert_eq!(field(&out, "reml"), &Value::Bool(reml));
    }
    let df = |out: &Value| f64s(field(out, "coef_df")).to_vec();
    let with = lmm(Port::Python, vec![]).unwrap();
    let without = lmm(
        Port::R,
        vec![("satterthwaite", Value::BoolVec(vec![false]))],
    )
    .unwrap();
    assert_ne!(df(&with), df(&without));
    let no_condvar = lmm(Port::Python, vec![("condvar", Value::Bool(false))]).unwrap();
    assert_ne!(field(&with, "condvar"), field(&no_condvar, "condvar"));

    for (port, null, t, f) in [
        (Port::Python, "None", "True", "False"),
        (Port::R, "NULL", "TRUE", "FALSE"),
    ] {
        let reml = if port == Port::R { "REML" } else { "reml" };
        let e = refused(lmm(port, vec![("reml", Value::I64(1))]));
        assert_eq!(e, format!("{reml} must be {null}, {t} or {f}, got 1"));
        for name in ["satterthwaite", "condvar"] {
            let e = refused(lmm(port, vec![(name, Value::I64(1))]));
            assert_eq!(e, format!("{name} must be {t} or {f}, got 1"));
        }
    }
}

#[test]
fn wald_se_is_hessian_or_rx() {
    let glmm = |port, v| {
        fit(
            port,
            "y ~ Days + (1 | Subject)",
            binary(),
            "binomial",
            vec![("wald_se", v)],
        )
    };
    glmm(Port::Python, text("rx")).unwrap();
    glmm(Port::R, Value::StrVec(vec!["hessian".into()])).unwrap();
    for (port, name) in [(Port::Python, "wald_se"), (Port::R, "wald.se")] {
        let e = refused(glmm(port, text("Hessian")));
        assert_eq!(
            e,
            format!(r#"{name} must be 'hessian' or 'rx', got the string "Hessian""#)
        );
    }
}

#[test]
fn anova_test_is_auto_wald_or_lrt() {
    let ols = |v| {
        fit(
            Port::Python,
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![("anova_test", v)],
        )
    };
    let out = ols(text("LRT")).unwrap();
    assert_ne!(field(&out, "term_table_type3_lrt"), &Value::Null);
    let out = ols(text("Wald")).unwrap();
    assert_eq!(field(&out, "term_table_type3_lrt"), &Value::Null);
    assert_eq!(
        refused(ols(text("lrt"))),
        r#"anova_test must be "auto", "Wald" or "LRT", got "lrt""#
    );
}

#[test]
fn dispersion_is_estimate_or_a_number() {
    let gamma = |port, v| {
        fit(
            port,
            "y ~ Days",
            reaction(),
            "gamma",
            vec![("dispersion", v)],
        )
    };
    let out = gamma(Port::Python, text("estimate")).unwrap();
    assert_eq!(field(&out, "dispersion_held"), &Value::Null);
    let out = gamma(Port::R, Value::Vec(vec![2.0])).unwrap();
    assert_eq!(field(&out, "dispersion_held"), &Value::F64(2.0));
    assert_eq!(field(&out, "dispersion"), &Value::F64(2.0));
    for (port, null) in [(Port::Python, "None"), (Port::R, "NULL")] {
        let e = refused(gamma(port, text("x")));
        assert_eq!(
            e,
            format!(r#"dispersion must be {null}, 'estimate', or a number, got the string "x""#)
        );
    }
    let e = refused(fit(
        Port::Python,
        "y ~ Days",
        binary(),
        "binomial",
        vec![("dispersion", Value::F64(2.0))],
    ));
    assert_eq!(
        e,
        "quasi-likelihood dispersion on family 'binomial' is not yet implemented in the kernel"
    );
    // A family without a dispersion drops the argument before its form is read.
    let out = fit(
        Port::Python,
        "y ~ Days",
        reaction(),
        "gaussian",
        vec![("dispersion", text("x"))],
    )
    .unwrap();
    assert_eq!(field(&out, "dispersion_held"), &Value::Null);
}

#[test]
fn warm_start_is_a_record_and_init_theta_is_given_once() {
    let start = rec(vec![
        ("beta", Value::Vec(vec![250.0, 10.0])),
        ("theta", Value::Vec(vec![])),
    ]);
    let out = fit(
        Port::Python,
        "y ~ Days + (1 | Subject)",
        reaction(),
        "gaussian",
        vec![("warm_start", Value::Record(start))],
    )
    .unwrap();
    let ws = field(&out, "warm_start");
    assert_eq!(f64s(field(ws, "beta")), [250.0, 10.0]);
    assert_eq!(field(ws, "dispersion"), &Value::Null);
    for (port, shown) in [
        (
            Port::Python,
            "warm_start must be a dict with keys 'beta', 'theta' and/or 'dispersion' (theta is \
             the random-effect Cholesky vector, not the negative-binomial shape - that is \
             init_theta or dispersion), got a numeric vector",
        ),
        (
            Port::R,
            "start must be a list with elements 'beta', 'theta' and/or 'dispersion' (theta is \
             the random-effect Cholesky vector, not the negative-binomial shape - that is \
             init.theta or dispersion), got a numeric vector",
        ),
    ] {
        let e = refused(fit(
            port,
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![("warm_start", Value::Vec(vec![1.0, 2.0]))],
        ));
        assert_eq!(e, shown);
    }
    let e = refused(fit(
        Port::Python,
        "y ~ Days",
        reaction(),
        "gaussian",
        vec![("warm_start", Value::Record(rec(vec![("beta", text("x"))])))],
    ));
    assert_eq!(
        e,
        r#"warm_start entry 'beta' must be a numeric vector, got the string "x""#
    );

    let nb = |port, extra| fit(port, "y ~ Days", counts(), "negativebinomial", extra);
    let seed = rec(vec![("dispersion", Value::F64(3.0))]);
    for (port, shown) in [
        (
            Port::Python,
            r#"give the negative-binomial shape seed once: init_theta= or warm_start["dispersion"], not both"#,
        ),
        (
            Port::R,
            "give the negative-binomial shape seed once: init.theta= or start$dispersion, not both",
        ),
    ] {
        let e = refused(nb(
            port,
            vec![
                ("init_theta", Value::F64(2.0)),
                ("warm_start", Value::Record(seed.clone())),
            ],
        ));
        assert_eq!(e, shown);
    }
    let out = nb(Port::R, vec![("init_theta", Value::F64(2.0))]).unwrap();
    assert_eq!(
        field(field(&out, "warm_start"), "dispersion"),
        &Value::F64(2.0)
    );
    let e = refused(nb(Port::R, vec![("init_theta", text("x"))]));
    assert_eq!(
        e,
        r#"start$dispersion (or init.theta) must be one number, got the string "x""#
    );
    let e = nb(Port::Python, vec![("init_theta", Value::F64(f64::NAN))]).unwrap_err();
    assert_eq!(
        e.message,
        "warm-start dispersion must be a finite number > 0, got NaN"
    );
}

#[test]
fn nsim_and_seed_take_each_ports_range() {
    let fixed = ("y ~ 1 + (1 | Subject)", "y ~ Days + (1 | Subject)");
    let out = compare(
        Port::Python,
        fixed.0,
        fixed.1,
        vec![("seed", Value::U64(u64::MAX))],
    )
    .unwrap();
    assert_eq!(field(&out, "seed"), &Value::U64(u64::MAX));
    let out = compare(
        Port::R,
        fixed.0,
        fixed.1,
        vec![("seed", Value::F64(2f64.powi(53)))],
    )
    .unwrap();
    assert_eq!(field(&out, "seed"), &Value::U64(1 << 53));
    let e = refused(compare(
        Port::R,
        fixed.0,
        fixed.1,
        vec![("seed", Value::U64((1 << 53) + 1))],
    ));
    assert_eq!(
        e,
        "seed must be NULL or one whole number from 0 to 2^53, got 9007199254740993"
    );
    let e = refused(compare(
        Port::Python,
        fixed.0,
        fixed.1,
        vec![("seed", Value::I64(-1))],
    ));
    assert_eq!(
        e,
        "seed must be None or a whole number from 0 to 2**64 - 1, got -1"
    );

    let compare_fits = |port, extra: Vec<(&str, Value)>| {
        let mut r = rec(vec![
            ("m0", Value::Record(Record::new())),
            ("m1", Value::Record(Record::new())),
        ]);
        for (k, v) in extra {
            r.set(k, v);
        }
        call(port, "compare_fits", r)
    };
    let e = refused(compare_fits(Port::R, vec![("seed", Value::U64(u64::MAX))]));
    assert!(e.starts_with("seed must be NULL or one whole"), "{e}");
    let range = "nsim must be a whole number from 1 to 1000000, got 0";
    assert_eq!(
        refused(compare_fits(Port::Python, vec![("nsim", Value::I64(0))])),
        range
    );

    let simulate = |port, nsim, seed| {
        let r = rec(vec![
            ("formula", text("y ~ 1")),
            ("numeric", Value::Record(Record::new())),
            ("factors", Value::Record(Record::new())),
            ("family", text("gaussian")),
            ("link", text("identity")),
            ("parts", Value::Record(Record::new())),
            ("nsim", nsim),
            ("seed", seed),
            ("row_base", Value::I64(0)),
        ]);
        call(port, "simulate", r)
    };
    assert_eq!(
        refused(simulate(Port::R, Value::F64(0.0), Value::F64(1.0))),
        range
    );
    let e = refused(simulate(Port::R, Value::F64(1.0), Value::F64(-1.0)));
    assert_eq!(
        e,
        "seed must be NULL or one whole number from 0 to 2^53, got -1"
    );
}

/// The fitted values of `predict` on the training rows, or its refusal.
fn predict(port: Port, extra: Vec<(&str, Value)>) -> Result<Value, BindError> {
    let parts = fit(
        Port::Python,
        "y ~ Days + (1 | Subject)",
        reaction(),
        "gaussian",
        vec![],
    )
    .unwrap();
    let Value::Record(parts) = parts else {
        unreachable!()
    };
    let mut r = rec(vec![
        ("formula", text("y ~ Days + (1 | Subject)")),
        ("numeric", Value::Record(numeric(reaction()))),
        ("factors", Value::Record(subject())),
        ("family", text("gaussian")),
        ("link", text("identity")),
        ("parts", Value::Record(parts)),
        ("n_rows", Value::I64(180)),
        ("scale", text("response")),
        ("allow_new_levels", Value::Bool(false)),
        ("se", Value::Bool(false)),
        ("level", Value::F64(0.95)),
        ("future", Value::Bool(false)),
        ("row_base", Value::I64(0)),
    ]);
    for (k, v) in extra {
        r.set(k, v);
    }
    call(port, "predict", r)
}

#[test]
fn predict_options_are_checked_in_bind() {
    let conditional = predict(Port::Python, vec![]).unwrap();
    let population = predict(Port::R, vec![("re_form", text("NA"))]).unwrap();
    assert_ne!(field(&conditional, "fit"), field(&population, "fit"));
    for (port, shown) in [
        (
            Port::Python,
            r#"re_form must be None (the random effects of every grouping) or "NA" (the fixed effects only), got the string "na""#,
        ),
        (
            Port::R,
            r#"re.form must be NULL (the random effects of every grouping) or NA (the fixed effects only), got the string "na""#,
        ),
    ] {
        assert_eq!(refused(predict(port, vec![("re_form", text("na"))])), shown);
    }
    for (port, name, arg, t, f) in [
        (
            Port::Python,
            "allow_new_levels",
            "allow_new_levels",
            "True",
            "False",
        ),
        (
            Port::R,
            "allow_new_levels",
            "allow.new.levels",
            "TRUE",
            "FALSE",
        ),
        (Port::Python, "se", "se_fit", "True", "False"),
        (Port::R, "se", "se.fit", "TRUE", "FALSE"),
        (Port::R, "future", "future", "TRUE", "FALSE"),
    ] {
        let e = refused(predict(port, vec![(name, Value::I64(1))]));
        assert_eq!(e, format!("{arg} must be {t} or {f}, got 1"));
    }
    assert_eq!(
        refused(predict(Port::Python, vec![("scale", text("probability"))])),
        r#"type must be "link" or "response", got "probability""#
    );
    assert_eq!(
        refused(predict(Port::R, vec![("level", Value::F64(1.0))])),
        "level must be in (0, 1), got 1"
    );
}

#[test]
fn a_python_none_gets_the_options_own_text() {
    let lmm = |extra| {
        fit(
            Port::Python,
            "y ~ Days + (1 | Subject)",
            reaction(),
            "gaussian",
            extra,
        )
    };
    assert_eq!(
        refused(lmm(vec![("satterthwaite", Value::Null)])),
        "satterthwaite must be True or False, got None"
    );
    assert_eq!(
        refused(lmm(vec![("nagq", Value::Null)])),
        "nagq must be an odd integer in 1..=25, got None"
    );
    assert_eq!(
        refused(lmm(vec![("anova_test", Value::Null)])),
        r#"anova_test must be "auto", "Wald" or "LRT", got None"#
    );
    assert_eq!(
        refused(predict(Port::Python, vec![("se", Value::Null)])),
        "se_fit must be True or False, got None"
    );
    assert_eq!(
        refused(predict(Port::Python, vec![("scale", Value::Null)])),
        r#"type must be "link" or "response", got None"#
    );
}

#[test]
fn an_r_na_gets_the_options_own_text() {
    let lmm = |extra| {
        fit(
            Port::R,
            "y ~ Days + (1 | Subject)",
            reaction(),
            "gaussian",
            extra,
        )
    };
    assert_eq!(
        refused(lmm(vec![("satterthwaite", Value::Na)])),
        "satterthwaite must be TRUE or FALSE, got NA"
    );
    assert_eq!(
        refused(lmm(vec![("reml", Value::Na)])),
        "REML must be NULL, TRUE or FALSE, got NA"
    );
    assert_eq!(
        refused(lmm(vec![("anova_test", Value::Na)])),
        r#"anova_test must be "auto", "Wald" or "LRT", got NA"#
    );
    assert_eq!(
        refused(predict(Port::R, vec![("se", Value::Na)])),
        "se.fit must be TRUE or FALSE, got NA"
    );
}

#[test]
fn pairwise_options_are_checked_in_bind() {
    let pairwise = |port, extra: Vec<(&str, Value)>| {
        let mut r = rec(vec![
            ("formula", text("y ~ 1")),
            ("numeric", Value::Record(Record::new())),
            ("factors", Value::Record(Record::new())),
            ("family", text("gaussian")),
            ("link", text("identity")),
            ("df_method", text("residual")),
            ("beta", Value::Vec(vec![])),
            ("vcov", Value::Vec(vec![])),
            ("aliased", Value::BoolVec(vec![])),
            ("nobs", Value::I64(0)),
            ("names", Value::StrVec(vec![])),
            ("factor", text("g")),
            ("adjust", text("holm")),
            ("converged", Value::Bool(true)),
        ]);
        for (k, v) in extra {
            r.set(k, v);
        }
        call(port, "pairwise", r)
    };
    let e = refused(pairwise(Port::Python, vec![("factor", Value::I64(1))]));
    assert_eq!(e, "factor must be one variable name, got 1");
    for (port, null) in [(Port::Python, "None"), (Port::R, "NULL")] {
        let e = refused(pairwise(port, vec![("by", Value::I64(2))]));
        assert_eq!(e, format!("by must be {null} or one variable name, got 2"));
    }
    let e = refused(pairwise(Port::R, vec![("at", Value::Vec(vec![1.0]))]));
    assert_eq!(e, "at gives the values of a numeric by, so it needs by");
    let e = refused(pairwise(
        Port::R,
        vec![("by", text("x")), ("at", Value::Vec(vec![1.0, f64::NAN]))],
    ));
    assert_eq!(e, "at must be one or more finite numbers, got NaN");
    let e = refused(pairwise(Port::Python, vec![("adjust", text("sidak"))]));
    assert_eq!(
        e,
        r#"adjust must be "holm", "bonferroni", "tukey" or "none", got "sidak""#
    );
}

#[test]
fn a_one_sided_formula_gets_the_parser_text() {
    let want = "formula syntax error: the formula has no response; write it as y ~ x";
    for port in BOTH {
        assert_eq!(
            refused(fit(port, "~ Days", reaction(), "gaussian", vec![])),
            want
        );
        assert_eq!(refused(compare(port, "~ 1", "~ Days", vec![])), want);
        assert_eq!(refused(compare(port, "y ~ Days", "~ Days", vec![])), want);
        assert_eq!(refused(compare(port, "Days", "y ~ Days", vec![])), want);
    }
}

#[test]
fn empty_data_gets_the_no_rows_text() {
    let no_rows = "Data has no rows (n = 0); a fit needs at least one row";
    let empty = rec(vec![("y", Value::Vec(vec![])), ("x", Value::Vec(vec![]))]);
    let g = rec(vec![(
        "g",
        Value::Factor {
            levels: vec!["a".into()],
            codes: vec![],
        },
    )]);
    for (formula, factors) in [("y ~ x", Record::new()), ("y ~ g", g)] {
        let r = fit_args(formula, empty.clone(), factors, 0);
        let e = call(Port::Python, "fit", r).unwrap_err();
        assert_eq!((e.code, e.message.as_str()), ("fit_error", no_rows));
    }
    let mut r = rec(vec![
        ("f0", text("y ~ 1")),
        ("f1", text("y ~ x")),
        ("numeric", Value::Record(empty)),
        ("factors", Value::Record(Record::new())),
        ("family", text("gaussian")),
        ("nagq", Value::I64(1)),
        ("row_base", Value::I64(1)),
    ]);
    r.set("link", Value::Null);
    let e = call(Port::R, "compare", r).unwrap_err();
    assert_eq!((e.code, e.message.as_str()), ("fit_error", no_rows));
}

/// `(kind, tier, title, message)` of each of a fit record's `argument_notes`.
fn argument_notes(out: &Value) -> Vec<(String, String, String, String)> {
    let Value::List(notes) = field(out, "argument_notes") else {
        panic!("argument_notes is not a list")
    };
    let s = |n: &Value, k: &str| match field(n, k) {
        Value::Str(s) => s.clone(),
        other => panic!("{k} is not a string: {other:?}"),
    };
    notes
        .iter()
        .map(|n| (s(n, "kind"), s(n, "tier"), s(n, "title"), s(n, "message")))
        .collect()
}

#[test]
fn ignored_arguments_come_back_as_notes() {
    let mixed = "y ~ Days + (1 | Subject)";
    let (levels, codes) = factor(
        &sleepstudy()
            .1
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>(),
    );
    let mut two_groupings = subject();
    two_groupings.push(
        "D",
        Value::Factor {
            levels,
            codes: codes.iter().map(|&c| c as i32).collect(),
        },
    );
    let start = |fields: Vec<(&str, Value)>| Value::Record(rec(fields));
    let ignored = ("argument_ignored", "note", "Argument ignored");
    // (formula, y, family, options, kind/tier/title, Python message, R message)
    let cases = [
        (
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![("dispersion", Value::F64(2.0))],
            ignored,
            "dispersion= has no effect for family 'gaussian'.",
            "dispersion= has no effect for family 'gaussian'.",
        ),
        (
            mixed,
            binary(),
            "binomial",
            vec![("dispersion", Value::F64(2.0))],
            ignored,
            "Quasi-likelihood dispersion= is not supported yet for binomial or Poisson models. \
             The default dispersion of 1 was used.",
            "Quasi-likelihood dispersion= is not supported yet for binomial or Poisson models. \
             The default dispersion of 1 was used.",
        ),
        (
            "y ~ Days",
            counts(),
            "poisson",
            vec![("init_theta", Value::F64(2.0))],
            ignored,
            "init_theta= is not used for family 'poisson'.",
            "init.theta= is not used for family 'poisson'.",
        ),
        (
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![(
                "warm_start",
                start(vec![("beta", Value::Vec(vec![])), ("phi", Value::F64(1.0))]),
            )],
            ignored,
            "warm_start accepts only 'beta', 'theta' and 'dispersion'; these entries were \
             ignored: phi.",
            "start accepts only 'beta', 'theta' and 'dispersion'; these entries were ignored: \
             phi.",
        ),
        (
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![("warm_start", start(vec![("dispersion", Value::F64(2.0))]))],
            ignored,
            r#"warm_start["dispersion"] is not used for family 'gaussian'."#,
            "start$dispersion is not used for family 'gaussian'.",
        ),
        (
            "y ~ Days",
            reaction(),
            "gaussian",
            vec![("weight_power", Value::F64(0.5))],
            ignored,
            "weight_power = 0.5 has no effect without weights.",
            "weight_power = 0.5 has no effect without weights.",
        ),
        (
            mixed,
            reaction(),
            "gaussian",
            vec![("nagq", Value::I64(3))],
            ignored,
            "nagq=3 has no effect for a Gaussian model, because nothing is approximated.",
            "nAGQ=3 has no effect for a Gaussian model, because nothing is approximated.",
        ),
        (
            "y ~ Days",
            binary(),
            "binomial",
            vec![("nagq", Value::I64(3))],
            ignored,
            "nagq=3 has no effect for a model without random effects, because nothing is \
             approximated.",
            "nAGQ=3 has no effect for a model without random effects, because nothing is \
             approximated.",
        ),
        (
            "y ~ Days + (1 | Subject) + (1 | D)",
            binary(),
            "binomial",
            vec![
                ("nagq", Value::I64(3)),
                ("factors", Value::Record(two_groupings)),
            ],
            ("agq_fallback", "caution", "Adaptive quadrature not used"),
            "nagq=3 works only for binomial, Poisson, negative-binomial or Gamma models whose \
             random effects are in one grouping factor, with at most 3 random effects in it. \
             This model was fitted without adaptive quadrature.",
            "nAGQ=3 works only for binomial, Poisson, negative-binomial or Gamma models whose \
             random effects are in one grouping factor, with at most 3 random effects in it. \
             This model was fitted without adaptive quadrature.",
        ),
        (
            "y ~ Days",
            counts(),
            "poisson",
            vec![("reml", Value::Bool(true))],
            ignored,
            "reml=True applies only to Gaussian models; it was ignored and the model was fitted \
             by maximum likelihood.",
            "REML = TRUE applies only to Gaussian models; it was ignored and the model was \
             fitted by maximum likelihood.",
        ),
    ];
    for (formula, y, family, extra, (kind, tier, title), python, r) in cases {
        for (port, message) in [(Port::Python, python), (Port::R, r)] {
            let out = fit(port, formula, y.clone(), family, extra.clone()).unwrap();
            assert_eq!(
                argument_notes(&out),
                [(kind.into(), tier.into(), title.into(), message.into())],
                "{formula} {family} {extra:?}"
            );
        }
    }
    let clean = fit(Port::R, mixed, reaction(), "gaussian", vec![]).unwrap();
    assert!(argument_notes(&clean).is_empty());
    let Value::Record(r) = clean else {
        unreachable!()
    };
    assert!(r.get("agq_warning").is_none() && r.get("reml_warning").is_none());
}

#[test]
fn an_unseen_level_names_the_grouping_and_the_ports_argument() {
    let (levels, _) = factor(&sleepstudy().2);
    let new = |port: Port| {
        let base = if port == Port::R { 1 } else { 0 };
        predict(
            port,
            vec![
                (
                    "new_numeric",
                    Value::Record(rec(vec![("Days", Value::Vec(vec![1.0, 2.0]))])),
                ),
                (
                    "new_factors",
                    Value::Record(rec(vec![(
                        "Subject",
                        Value::Factor {
                            levels: vec![levels[0].clone(), "unseen".into()],
                            codes: vec![0, 1],
                        },
                    )])),
                ),
                ("n_rows", Value::I64(2)),
                ("row_base", Value::I64(base)),
            ],
        )
        .unwrap_err()
    };
    for (port, row, arg) in [
        (Port::R, 2, "allow.new.levels = TRUE"),
        (Port::Python, 1, "allow_new_levels=True"),
    ] {
        let e = new(port);
        assert_eq!((e.code, e.row), ("fit_error", Some(row)));
        assert_eq!(
            e.message,
            format!(
                "grouping \"Subject\" has a level the fit did not see at row {row}; set {arg} \
                 to predict it with a zero random effect"
            )
        );
    }
}

#[test]
fn a_warm_start_dispersion_is_kept_only_where_the_kernel_reads_it() {
    let mixed = "y ~ Days + (1 | Subject)";
    let start = Value::Record(rec(vec![("dispersion", Value::F64(2.0))]));
    for (formula, y, family, kept) in [
        ("y ~ Days", binary(), "binomial", false),
        ("y ~ Days", counts(), "poisson", false),
        ("y ~ Days", reaction(), "inversegaussian", false),
        ("y ~ Days", reaction(), "gamma", false),
        ("y ~ Days", counts(), "negativebinomial", true),
        (mixed, counts(), "negativebinomial", true),
        (mixed, reaction(), "gamma", true),
    ] {
        let out = fit(
            Port::R,
            formula,
            y,
            family,
            vec![("warm_start", start.clone())],
        )
        .unwrap();
        let seed = field(field(&out, "warm_start"), "dispersion");
        let notes: Vec<String> = argument_notes(&out).into_iter().map(|n| n.3).collect();
        if kept {
            assert_eq!(seed, &Value::F64(2.0), "{formula} {family}");
            assert!(notes.is_empty(), "{formula} {family}: {notes:?}");
        } else {
            assert_eq!(seed, &Value::Null, "{formula} {family}");
            assert_eq!(
                notes,
                [format!(
                    "start$dispersion is not used for family '{family}'."
                )],
            );
        }
    }
}

#[test]
fn an_unseen_level_is_found_row_by_row_then_grouping_by_grouping() {
    let formula = "y ~ Days + (1 | Subject) + (1 | D)";
    let days = sleepstudy().1;
    let (d_levels, d_codes) = factor(&days.iter().map(|d| d.to_string()).collect::<Vec<_>>());
    let mut factors = subject();
    factors.push(
        "D",
        Value::Factor {
            levels: d_levels.clone(),
            codes: d_codes.iter().map(|&c| c as i32).collect(),
        },
    );
    let parts = fit(
        Port::Python,
        formula,
        reaction(),
        "gaussian",
        vec![("factors", Value::Record(factors.clone()))],
    )
    .unwrap();
    let (s_levels, _) = factor(&sleepstudy().2);
    // Rows of (Subject, D), each `None` a level the fit did not see.
    let refusal = |port: Port, rows: &[(Option<usize>, Option<usize>)]| {
        // The codes of `levels` plus one unseen level, `None` the unseen one.
        let column = |levels: &[String], codes: Vec<Option<usize>>| {
            let mut levels = levels.to_vec();
            levels.push("unseen".into());
            let unseen = (levels.len() - 1) as i32;
            Value::Factor {
                levels,
                codes: codes
                    .into_iter()
                    .map(|c| c.map_or(unseen, |i| i as i32))
                    .collect(),
            }
        };
        let r = rec(vec![
            ("formula", text(formula)),
            ("numeric", Value::Record(numeric(reaction()))),
            ("factors", Value::Record(factors.clone())),
            ("family", text("gaussian")),
            ("link", text("identity")),
            ("parts", parts.clone()),
            (
                "new_numeric",
                Value::Record(rec(vec![("Days", Value::Vec(vec![1.0; rows.len()]))])),
            ),
            (
                "new_factors",
                Value::Record(rec(vec![
                    (
                        "Subject",
                        column(&s_levels, rows.iter().map(|r| r.0).collect()),
                    ),
                    ("D", column(&d_levels, rows.iter().map(|r| r.1).collect())),
                ])),
            ),
            ("n_rows", Value::I64(rows.len() as i64)),
            ("scale", text("response")),
            ("allow_new_levels", Value::Bool(false)),
            ("se", Value::Bool(false)),
            ("level", Value::F64(0.95)),
            ("future", Value::Bool(false)),
            ("row_base", Value::I64(if port == Port::R { 1 } else { 0 })),
        ]);
        let e = call(port, "predict", r).unwrap_err();
        assert_eq!(e.code, "fit_error");
        (e.message, e.row)
    };
    let shown = |g: &str, row: usize, arg: &str| {
        format!(
            "grouping \"{g}\" has a level the fit did not see at row {row}; set {arg} to \
             predict it with a zero random effect"
        )
    };
    // D's unseen level comes a row before Subject's: the earlier row wins.
    let rows = [(Some(0), Some(0)), (Some(0), None), (None, Some(0))];
    assert_eq!(
        refusal(Port::R, &rows),
        (shown("D", 2, "allow.new.levels = TRUE"), Some(2))
    );
    assert_eq!(
        refusal(Port::Python, &rows),
        (shown("D", 1, "allow_new_levels=True"), Some(1))
    );
    // Both unseen in one row: the grouping the formula declares first wins.
    let rows = [(Some(1), Some(1)), (Some(1), Some(2)), (None, None)];
    assert_eq!(
        refusal(Port::R, &rows),
        (shown("Subject", 3, "allow.new.levels = TRUE"), Some(3))
    );
}
