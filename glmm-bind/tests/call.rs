mod common;

use std::collections::HashMap;

use common::{f64s, factor, field, fit_args, rec, sleepstudy, text};
use glmm::formula::anova::TermTest;
use glmm::{Family, FitOptions};
use glmm_bind::{call, Port, Record, Value};

#[test]
fn unknown_function_is_invalid_argument() {
    let e = call(Port::Python, "nope", Record::new()).unwrap_err();
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
    let e = call(Port::Python, "fit", r).unwrap_err();
    assert_eq!(e.code, "invalid_argument");
    assert!(
        e.message.contains("unexpected argument 'bogus'"),
        "{}",
        e.message
    );
    let e = call(Port::Python, "fit", Record::new()).unwrap_err();
    assert!(
        e.message.contains("missing required argument"),
        "{}",
        e.message
    );
}

#[test]
fn a_duplicate_argument_is_refused() {
    let r = rec(vec![("formula", text("y ~ x")), ("formula", text("y ~ 1"))]);
    let e = call(Port::Python, "fit", r).unwrap_err();
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
    let out = call(Port::Python, "fit", fit_args(formula, numeric, factors, 0)).unwrap();

    let mut opts = FitOptions::default();
    opts.condvar = true;
    let direct = glmm::orchestrate::run_fit(
        formula,
        HashMap::from([
            ("Reaction".to_string(), reaction),
            ("Days".to_string(), days),
        ]),
        HashMap::from([("Subject".to_string(), (levels.clone(), codes))]),
        Family::Gaussian,
        opts,
        None,
        None,
        None,
        TermTest::Auto,
    )
    .unwrap();
    assert_eq!(f64s(field(&out, "beta")), direct.beta.as_slice());
    assert_eq!(f64s(field(&out, "se")), direct.se.as_slice());
    assert!(matches!(field(&out, "loglik"), Value::F64(v) if *v == direct.loglik));
    let sd_se = f64s(field(&out, "stddev_se"));
    assert_eq!(sd_se.len(), 3, "one vech of q = 2");
    assert!(sd_se.iter().all(|v| v.is_nan()));

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
    let e = call(Port::Python, "fit", fit_args("y ~ g", numeric, factors, 0)).unwrap_err();
    assert_eq!(e.code, "invalid_argument");
    assert!(e.message.contains("factor column \"g\""), "{}", e.message);
}

/// R sends a scalar as a length-1 vector; an error names its element, as it
/// does for Python's plain scalar.
#[test]
fn a_length_one_vector_is_described_by_its_element() {
    let (reaction, days, _) = sleepstudy();
    let fit = |k: &str, v: Value| {
        let numeric = rec(vec![
            ("Reaction", Value::Vec(reaction.clone())),
            ("Days", Value::Vec(days.clone())),
        ]);
        let mut r = fit_args("Reaction ~ Days", numeric, Record::new(), 0);
        r.push(k, v);
        call(Port::Python, "fit", r).unwrap_err().message
    };
    let e = fit("weight_power", Value::StrVec(vec!["est".into()]));
    assert!(
        e.starts_with(
            r#"weight_power must be a finite number or "estimate", got the string "est""#
        ),
        "{e}"
    );
    let e = fit("weight_power", Value::Vec(vec![f64::NAN]));
    assert!(e.ends_with("got NaN"), "{e}");
    let e = fit("weight_power", Value::Vec(vec![1.0, 2.0]));
    assert!(e.ends_with("got a numeric vector"), "{e}");
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
        call(Port::Python, "fit", r)
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
    let e = call(Port::Python, "compare", compare).unwrap_err();
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
        call(Port::Python, "fit", r).unwrap()
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
        assert_eq!(f64s(field(robust, "t")).len(), p * p, "{formula}");
    }
    let model = fit("Reaction ~ Days", vec![("weight_power", Value::F64(1.0))]);
    assert_eq!(field(&model, "df_method"), &text("residual"));
    assert_eq!(field(&model, "robust"), &Value::Null);
    assert_eq!(field(&model, "model_vcov"), &Value::List(vec![]));
}

#[test]
fn pairwise_and_its_intervals_carry_their_notes() {
    let numeric = rec(vec![(
        "y",
        Value::Vec(vec![1.0, 1.4, 0.8, 2.1, 2.5, 1.9, 3.2, 2.7, 3.6]),
    )]);
    let factors = rec(vec![(
        "f",
        Value::Factor {
            levels: vec!["a".into(), "b".into(), "c".into()],
            codes: vec![0, 0, 0, 1, 1, 1, 2, 2, 2],
        },
    )]);
    let out = call(
        Port::Python,
        "fit",
        fit_args("y ~ f", numeric.clone(), factors.clone(), 0),
    )
    .unwrap();
    let pairwise = |adjust: &str| {
        let mut r = rec(vec![
            ("formula", text("y ~ f")),
            ("numeric", Value::Record(numeric.clone())),
            ("factors", Value::Record(factors.clone())),
            ("family", text("gaussian")),
            ("link", text("identity")),
            ("factor", text("f")),
            ("adjust", text(adjust)),
            ("converged", Value::Bool(false)),
        ]);
        for k in ["df_method", "beta", "vcov", "aliased", "nobs", "names"] {
            r.push(k, field(&out, k).clone());
        }
        call(Port::Python, "pairwise", r).unwrap()
    };
    let intervals = |table: Value| {
        call(
            Port::R,
            "pairwise_confint",
            rec(vec![("table", table), ("level", Value::F64(0.9))]),
        )
        .unwrap()
    };
    let notes = |v: &Value| match field(v, "notes") {
        Value::StrVec(n) => n.clone(),
        other => panic!("{other:?}"),
    };
    let holm = pairwise("holm");
    assert_eq!(
        notes(&holm),
        [
            "The fit did not converge, so its estimates are not at a maximum and these tests are \
             not reliable. Change the model until it converges (a simpler random part often \
             helps), then test again.",
            "P value adjustment: holm method for 3 tests",
        ]
    );
    assert_eq!(
        notes(&intervals(holm)),
        [
            "The fit did not converge, so its estimates are not at a maximum and these \
             intervals are not reliable. Change the model until it converges (a simpler random \
             part often helps), then compute them again.",
            "Confidence level used: 0.9",
            "Holm has no simultaneous interval; Bonferroni intervals shown.",
        ]
    );
    let tukey = notes(&intervals(pairwise("tukey")));
    assert_eq!(
        tukey[2],
        "Conf-level adjustment: tukey method for comparing a family of 3 estimates"
    );
}

#[test]
fn pairwise_notes_count_each_by_groups_family() {
    // y ~ f * h with the cell f = c, h = v empty: in h = v only a - b is
    // estimable, so that group has one test where h = u has three.
    let codes_f = vec![0, 0, 1, 1, 2, 2, 0, 0, 1, 1];
    let codes_h = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1];
    let numeric = rec(vec![(
        "y",
        Value::Vec(vec![1.0, 1.4, 2.1, 2.5, 3.2, 2.7, 1.8, 2.2, 3.1, 2.6]),
    )]);
    let factors = rec(vec![
        (
            "f",
            Value::Factor {
                levels: vec!["a".into(), "b".into(), "c".into()],
                codes: codes_f,
            },
        ),
        (
            "h",
            Value::Factor {
                levels: vec!["u".into(), "v".into()],
                codes: codes_h,
            },
        ),
    ]);
    let out = call(
        Port::Python,
        "fit",
        fit_args("y ~ f * h", numeric.clone(), factors.clone(), 0),
    )
    .unwrap();
    let pairwise = |adjust: &str| {
        let mut r = rec(vec![
            ("formula", text("y ~ f * h")),
            ("numeric", Value::Record(numeric.clone())),
            ("factors", Value::Record(factors.clone())),
            ("family", text("gaussian")),
            ("link", text("identity")),
            ("factor", text("f")),
            ("by", text("h")),
            ("adjust", text(adjust)),
            ("converged", Value::Bool(true)),
        ]);
        for k in ["df_method", "beta", "vcov", "aliased", "nobs", "names"] {
            r.push(k, field(&out, k).clone());
        }
        call(Port::Python, "pairwise", r).unwrap()
    };
    let notes = |v: &Value| match field(v, "notes") {
        Value::StrVec(n) => n.clone(),
        other => panic!("{other:?}"),
    };
    let intervals = |table: Value| {
        notes(
            &call(
                Port::R,
                "pairwise_confint",
                rec(vec![("table", table), ("level", Value::F64(0.95))]),
            )
            .unwrap(),
        )
    };
    let aliased = [
        "a - c, h = v: The coefficient fc:hv is aliased, so the hypothesis was not tested.",
        "b - c, h = v: The coefficient fc:hv is aliased, so the hypothesis was not tested.",
    ];
    let holm = pairwise("holm");
    let p = f64s(field(&holm, "p_value"));
    assert!(p[..4].iter().all(|v| !v.is_nan()) && p[4..].iter().all(|v| v.is_nan()));
    assert_eq!(
        notes(&holm),
        [
            &["P value adjustment: holm method for 1 or 3 tests"][..],
            &aliased
        ]
        .concat()
    );
    let tukey = pairwise("tukey");
    assert_eq!(
        notes(&tukey),
        [
            &["P value adjustment: tukey method for varying family sizes"][..],
            &aliased
        ]
        .concat()
    );
    assert_eq!(
        intervals(tukey),
        [
            &[
                "Confidence level used: 0.95",
                "Conf-level adjustment: tukey method for varying family sizes"
            ][..],
            &aliased
        ]
        .concat()
    );
    assert_eq!(
        intervals(pairwise("bonferroni")),
        [
            &[
                "Confidence level used: 0.95",
                "Conf-level adjustment: bonferroni method for 1 or 3 estimates"
            ][..],
            &aliased
        ]
        .concat()
    );
}

/// R counts the design column from 1, as it counts rows; `row_index` renames
/// the row only.
#[test]
fn a_design_refusal_counts_its_column_from_row_base() {
    let data = |x: Vec<f64>, z: Vec<f64>| {
        let numeric = rec(vec![
            (
                "y",
                Value::Vec(vec![1.0, 2.5, 2.0, 4.0, 5.5, 5.0, 6.5, 8.0]),
            ),
            ("x", Value::Vec(x)),
            ("z", Value::Vec(z)),
        ]);
        let factors = rec(vec![(
            "g",
            Value::Factor {
                levels: vec!["a".into(), "b".into(), "c".into(), "d".into()],
                codes: vec![0, 0, 1, 1, 2, 2, 3, 3],
            },
        )]);
        (numeric, factors)
    };
    let plain = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut wide = plain.clone();
    wide[3] = 2e100;

    let (numeric, factors) = data(wide.clone(), plain.clone());
    let e = call(Port::R, "fit", fit_args("y ~ x", numeric, factors, 1)).unwrap_err();
    assert_eq!(e.code, "fit_error");
    assert_eq!(e.row, Some(4));
    assert_eq!(
        e.message,
        "x must be finite with |x| <= 1e100 (row 4, column 2, value 2e100; glmm::limits::MAX_DATA_ABS)"
    );

    let (numeric, factors) = data(plain.clone(), wide.clone());
    let e = call(
        Port::R,
        "fit",
        fit_args("y ~ x + (0 + z | g)", numeric, factors, 1),
    )
    .unwrap_err();
    assert_eq!(
        e.message,
        "re_columns must be finite with |re_columns| <= 1e100 (row 4, column 1, value 2e100; glmm::limits::MAX_DATA_ABS)"
    );

    let (numeric, factors) = data(wide, plain);
    let mut args = fit_args("y ~ x", numeric, factors, 1);
    args.push("row_index", Value::IntVec((100..108).collect()));
    let e = call(Port::R, "fit", args).unwrap_err();
    assert_eq!(e.row, Some(103));
    assert!(e.message.contains("(row 103, column 2,"), "{}", e.message);
}

/// A kernel panic is a crate bug, so it reaches the port as `internal`, never
/// as a refused argument. A stored fit whose `aliased` is one entry shorter
/// than `beta` panics on an index inside the contrast test.
#[test]
fn a_kernel_panic_is_an_internal_error() {
    let r = rec(vec![
        ("df_method", text("residual")),
        ("beta", Value::Vec(vec![1.0, 2.0])),
        ("vcov", Value::Vec(vec![1.0, 0.0, 0.0, 1.0])),
        ("aliased", Value::BoolVec(vec![false])),
        ("nobs", Value::I64(10)),
        ("dispersion", Value::F64(1.0)),
        ("names", Value::StrVec(vec!["a".into(), "b".into()])),
        ("l", Value::Vec(vec![0.0, 1.0])),
        ("rows", Value::I64(1)),
    ]);
    let e = call(Port::Python, "contrast_test", r).unwrap_err();
    assert_eq!(e.code, "internal", "{}", e.message);
}

/// The `singular_warning` of a fit record: the entry with its Affected list
/// on a singular converged fit, `Null` on a fit that is not singular and on
/// one that did not converge.
#[test]
fn the_singular_warning_comes_with_a_singular_converged_fit() {
    // y has no part that varies with a or b, so both intercept variances pin.
    let n = 96;
    let z: Vec<f64> = (0..n).map(|i| 0.8 * (2.3 * i as f64 + 0.7).sin()).collect();
    let a: Vec<usize> = (0..n).map(|i| i % 4).collect();
    let b: Vec<usize> = (0..n).map(|i| (i / 4) % 12).collect();
    let mean_by = |g: &[usize], k: usize| -> Vec<f64> {
        let (mut sum, mut cnt) = (vec![0.0; k], vec![0.0; k]);
        for (v, &l) in z.iter().zip(g) {
            sum[l] += v;
            cnt[l] += 1.0;
        }
        g.iter().map(|&l| sum[l] / cnt[l]).collect()
    };
    let (ma, mb) = (mean_by(&a, 4), mean_by(&b, 12));
    let mean = z.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = (0..n).map(|i| z[i] - ma[i] - mb[i] + mean).collect();
    let factor_of = |g: &[usize], k: usize, p: &str| Value::Factor {
        levels: (0..k).map(|l| format!("{p}{l}")).collect(),
        codes: g.iter().map(|&l| l as i32).collect(),
    };
    let factors = rec(vec![
        ("a", factor_of(&a, 4, "a")),
        ("b", factor_of(&b, 12, "b")),
    ]);
    let fit = |y: Vec<f64>, formula: &str| {
        let numeric = rec(vec![("y", Value::Vec(y))]);
        call(
            Port::Python,
            "fit",
            fit_args(formula, numeric, factors.clone(), 0),
        )
        .unwrap()
    };
    let singular = fit(y.clone(), "y ~ 1 + (1 | a) + (1 | b)");
    assert_eq!(field(&singular, "converged"), &Value::Bool(true));
    let entry = field(&singular, "singular_warning");
    assert_eq!(field(entry, "kind"), &text("singular"));
    assert_eq!(field(entry, "tier"), &text("caution"));
    assert_eq!(field(entry, "title"), &text("Singular fit"));
    assert_eq!(
        field(entry, "message"),
        &text(
            "The random effects are too complex for the data: a variance is estimated at or \
             near zero, or a correlation at or near \u{2212}1 or 1. Consider removing the \
             affected random effect. Affected: (Intercept) in a, (Intercept) in b."
        )
    );

    let planted: Vec<f64> = (0..n)
        .map(|i| y[i] + [-1.0, 0.6, 0.2, -0.4][a[i]])
        .collect();
    let plain = fit(planted, "y ~ 1 + (1 | a)");
    assert_eq!(field(&plain, "converged"), &Value::Bool(true));
    assert_eq!(field(&plain, "singular"), &Value::Bool(false));
    assert_eq!(field(&plain, "singular_warning"), &Value::Null);

    let failed = fit(vec![3.0; n], "y ~ 1 + (1 | a)");
    assert_eq!(field(&failed, "converged"), &Value::Bool(false));
    assert_eq!(field(&failed, "singular_warning"), &Value::Null);
}

// The anova port gate sends every call through `call`, as both ports do, so
// a glmm-bind bug shared by the ports (a dropped option, two record fields
// swapped) passes that gate. The tests below compare each function it calls
// with the direct crate call, field by field, each option at a value other
// than the gate's so that a dropped one moves a number.

type NumericColumns = HashMap<String, Vec<f64>>;
type FactorColumns = HashMap<String, (Vec<String>, Vec<u32>)>;

/// A data set of `validation/data` with its family and link.
struct Data {
    numeric: NumericColumns,
    factors: FactorColumns,
    family: &'static str,
    link: &'static str,
}

impl Data {
    /// `csv` with the columns named in `factors` as factors, levels in
    /// first-seen order.
    fn load(csv: &str, factors: &[&str], family: &'static str, link: &'static str) -> Self {
        let mut lines = csv.lines();
        let names: Vec<&str> = lines
            .next()
            .unwrap()
            .split(',')
            .map(|s| s.trim_matches('"'))
            .collect();
        let rows: Vec<Vec<&str>> = lines.map(|l| l.split(',').collect()).collect();
        let (mut numeric, mut factor_columns) = (HashMap::new(), HashMap::new());
        for (j, name) in names.into_iter().enumerate() {
            let cells = rows.iter().map(|r| r[j].trim_matches('"'));
            if factors.contains(&name) {
                let labels: Vec<String> = cells.map(String::from).collect();
                factor_columns.insert(name.to_string(), factor(&labels));
            } else {
                numeric.insert(
                    name.to_string(),
                    cells.map(|c| c.parse().unwrap()).collect(),
                );
            }
        }
        Data {
            numeric,
            factors: factor_columns,
            family,
            link,
        }
    }

    fn kernel_family(&self) -> Family {
        glmm::orchestrate::family_from_str(self.family, self.link).unwrap()
    }

    /// `numeric` and `factors` as a port passes them.
    fn columns(&self) -> (Record, Record) {
        let mut numeric = Record::new();
        for (k, v) in &self.numeric {
            numeric.push(k, Value::Vec(v.clone()));
        }
        let mut factors = Record::new();
        for (k, v) in &self.factors {
            factors.push(k, factor_value(v));
        }
        (numeric, factors)
    }

    /// `formula`, the columns, `family` and `link`, as a port passes a
    /// stored fit's training data.
    fn model_data(&self, formula: &str) -> Vec<(&'static str, Value)> {
        let (numeric, factors) = self.columns();
        vec![
            ("formula", text(formula)),
            ("numeric", Value::Record(numeric)),
            ("factors", Value::Record(factors)),
            ("family", text(self.family)),
            ("link", text(self.link)),
        ]
    }

    /// `formula` fitted through `call`, as the anova gate's `fit` passes it,
    /// and by `run_fit`.
    fn fit(
        &self,
        formula: &str,
        nagq: u8,
        reml: Option<bool>,
        anova_test: TermTest,
    ) -> (Value, glmm::orchestrate::FitResult) {
        let (numeric, factors) = self.columns();
        let mut r = fit_args(formula, numeric, factors, 0);
        r.set("family", text(self.family));
        r.set("link", text(self.link));
        let test = match anova_test {
            TermTest::Auto => "auto",
            TermTest::Lrt => "LRT",
            _ => "Wald",
        };
        r.set("anova_test", text(test));
        r.set("nagq", Value::I64(nagq.into()));
        r.set("reml", reml.map_or(Value::Null, Value::Bool));
        let out = call(Port::Python, "fit", r).unwrap();
        let mut opts = FitOptions::default();
        opts.condvar = true;
        opts.nagq = nagq;
        opts.reml = reml;
        let direct = glmm::orchestrate::run_fit(
            formula,
            self.numeric.clone(),
            self.factors.clone(),
            self.kernel_family(),
            opts,
            None,
            None,
            None,
            anova_test,
        )
        .unwrap();
        assert!(direct.converged, "{formula}");
        assert_eq!(
            f64s(field(&out, "beta")),
            direct.beta.as_slice(),
            "{formula}"
        );
        (out, direct)
    }
}

fn factor_value((levels, codes): &(Vec<String>, Vec<u32>)) -> Value {
    Value::Factor {
        levels: levels.clone(),
        codes: codes.iter().map(|&c| c as i32).collect(),
    }
}

/// A number field, a count included.
fn num(v: &Value) -> f64 {
    match v {
        Value::F64(x) => *x,
        Value::I64(n) => *n as f64,
        Value::U64(n) => *n as f64,
        other => panic!("not a number: {other:?}"),
    }
}

/// `got` is `want` exactly, NaN matching NaN.
fn same(got: &Value, want: f64, what: &str) {
    let g = num(got);
    assert!(
        g == want || (g.is_nan() && want.is_nan()),
        "{what}: {g} != {want}"
    );
}

fn same_vec(got: &Value, want: &[f64], what: &str) {
    let g = f64s(got);
    assert_eq!(g.len(), want.len(), "{what}");
    for (i, (a, b)) in g.iter().zip(want).enumerate() {
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "{what}[{i}]: {a} != {b}"
        );
    }
}

fn strings(v: &Value) -> &[String] {
    match v {
        Value::StrVec(s) => s,
        other => panic!("not a string vector: {other:?}"),
    }
}

/// A comparison record against the crate's comparison of the same two fits,
/// `rss` each side's residual sum of squares.
fn assert_comparison(
    out: &Value,
    c: &glmm::formula::anova::Comparison,
    rss: [f64; 2],
    nsim: usize,
    seed: u64,
    at: &str,
) {
    use glmm::formula::anova::{ComparisonTest, Side};
    let Value::List(sides) = field(out, "sides") else {
        panic!("{at}: no sides")
    };
    assert_eq!(sides.len(), 2, "{at}");
    for (i, (s, m)) in sides.iter().zip([&c.m0, &c.m1]).enumerate() {
        let at = format!("{at}, side {i}");
        assert_eq!(field(s, "formula"), &text(&m.formula_label), "{at}");
        same(field(s, "npar"), m.npar as f64, &at);
        same(field(s, "nobs"), m.nobs as f64, &at);
        // `npar` counts the coefficients and one scale parameter.
        same(field(s, "res_df"), (m.nobs + 1 - m.npar) as f64, &at);
        same(field(s, "loglik"), m.loglik, &at);
        assert_eq!(
            field(s, "aic"),
            &m.aic.map_or(Value::Null, Value::F64),
            "{at}"
        );
        assert_eq!(
            field(s, "bic"),
            &m.bic.map_or(Value::Null, Value::F64),
            "{at}"
        );
        same(field(s, "rss"), rss[i], &at);
    }
    let (test, statistic, df, p) = match &c.test {
        ComparisonTest::Lr(t) => ("lr", t.statistic, t.df as f64, t.p_value),
        ComparisonTest::Linear(t) => ("f", t.statistic, t.num_df, t.p_value),
        other => panic!("{at}: {other:?}"),
    };
    let large = match c.larger {
        Side::First => 0,
        Side::Second => 1,
    };
    assert_eq!(field(out, "test"), &text(test), "{at}");
    assert_eq!(field(out, "larger"), &Value::I64(large), "{at}");
    same(field(out, "statistic"), statistic, at);
    same(field(out, "df"), df, at);
    same(field(out, "p_value"), p, at);
    same(field(out, "nsim"), nsim as f64, at);
    same(field(out, "seed"), seed as f64, at);
    let ols = matches!(field(out, "ols"), Value::Bool(true));
    let sum_sq = if ols {
        rss[1 - large as usize] - rss[large as usize]
    } else {
        f64::NAN
    };
    same(field(out, "sum_sq"), sum_sq, at);
}

const SLEEPSTUDY: &str = include_str!("../../validation/data/empirical/sleepstudy.csv");
const DYESTUFF: &str = include_str!("../../validation/data/empirical/Dyestuff.csv");
const FACTORIAL: &str = include_str!("../../validation/data/simulated/sim_factorial.csv");
const FACTORIAL_BERN: &str = include_str!("../../validation/data/simulated/sim_factorial_bern.csv");

/// The anova gate's Gaussian LMM and binomial GLMM on `sim_factorial`.
fn factorial_models() -> [(Data, &'static str); 2] {
    let f = ["subj", "g", "h"];
    [
        (
            Data::load(FACTORIAL, &f, "gaussian", "identity"),
            "y ~ g * h + x + (1 + x | subj)",
        ),
        (
            Data::load(FACTORIAL_BERN, &f, "binomial", "logit"),
            "y ~ g * h + x + (1 | subj)",
        ),
    ]
}

/// `compare` and `compare_fits` (on two fits made through `call`) against
/// `run_compare`: an exact restricted LR test at `nsim` and `seed` other
/// than the defaults (a dropped one moves the p-value), an ML pair whose
/// fits `fit` makes with `reml = false`, an OLS F test, and a binomial pair
/// at `nagq = 5`.
#[test]
fn compare_and_compare_fits_match_run_compare() {
    let sleep = || Data::load(SLEEPSTUDY, &["Subject"], "gaussian", "identity");
    let cases = [
        (
            Data::load(DYESTUFF, &["Batch"], "gaussian", "identity"),
            ["Yield ~ 1", "Yield ~ 1 + (1 | Batch)"],
            1,
            Some(true),
        ),
        (
            sleep(),
            [
                "Reaction ~ 1 + (Days | Subject)",
                "Reaction ~ Days + (Days | Subject)",
            ],
            1,
            Some(false),
        ),
        (sleep(), ["Reaction ~ 1", "Reaction ~ Days"], 1, Some(false)),
        (
            Data::load(FACTORIAL_BERN, &["subj", "g", "h"], "binomial", "logit"),
            ["y ~ x + (1 | subj)", "y ~ g + x + (1 | subj)"],
            5,
            None,
        ),
    ];
    let (nsim, seed) = (2000, 7);
    let mut kinds = Vec::new();
    for (data, [f0, f1], nagq, reml) in cases {
        let mut opts = FitOptions::default();
        opts.condvar = true;
        opts.nagq = nagq;
        let direct = glmm::orchestrate::run_compare(
            f0,
            f1,
            data.numeric.clone(),
            data.factors.clone(),
            data.kernel_family(),
            opts,
            None,
            None,
            Some(nsim),
            Some(seed),
        )
        .unwrap();
        let rss = [direct.fit0.rss, direct.fit1.rss];

        let mut args = rec(vec![("f0", text(f0)), ("f1", text(f1))]);
        for (k, v) in data.model_data(f0).into_iter().skip(1) {
            args.push(k, v);
        }
        args.push("nagq", Value::I64(nagq.into()));
        args.push("nsim", Value::I64(nsim as i64));
        args.push("seed", Value::I64(seed as i64));
        args.push("row_base", Value::I64(0));
        let out = call(Port::Python, "compare", args).unwrap();
        assert_comparison(&out, &direct.comparison, rss, nsim, seed, f1);
        for (side, fit) in [("fit0", &direct.fit0), ("fit1", &direct.fit1)] {
            let r = field(&out, side);
            assert_eq!(f64s(field(r, "beta")), fit.beta.as_slice(), "{f1} {side}");
            same(field(r, "loglik"), fit.loglik, side);
            same(field(r, "nobs"), fit.nobs as f64, side);
        }

        let stored = |f: &str| {
            let Value::Record(mut r) = data.fit(f, nagq, reml, TermTest::Wald).0 else {
                unreachable!()
            };
            r.set("formula", text(f));
            Value::Record(r)
        };
        let args = rec(vec![
            ("m0", stored(f0)),
            ("m1", stored(f1)),
            ("nsim", Value::I64(nsim as i64)),
            ("seed", Value::I64(seed as i64)),
        ]);
        let out = call(Port::Python, "compare_fits", args).unwrap();
        assert_comparison(&out, &direct.comparison, rss, nsim, seed, f1);
        kinds.push(match field(&out, "reference_kind") {
            Value::Str(k) => k.clone(),
            other => panic!("{other:?}"),
        });
    }
    assert_eq!(kinds, ["exact", "chisq", "f", "chisq"]);
}

/// `predict` against `orchestrate::predict` on the anova gate's new rows (one
/// a `subj` the fit did not see), at `level = 0.8`: the Gaussian LMM without
/// its random effects (`re_form = NA`), the binomial GLMM on the response
/// scale with the intervals of a future observation (`future`).
#[test]
fn predict_through_call_matches_predict() {
    let subj = ["s01", "s02", "s99", "s18"].map(String::from);
    let g = ["a", "b", "c", "c"].map(String::from);
    let h = ["p", "q", "p", "q"].map(String::from);
    let x = vec![0.5, -1.0, 0.0, 1.25];
    let new_factors: FactorColumns = HashMap::from([
        ("subj".to_string(), factor(&subj)),
        ("g".to_string(), factor(&g)),
        ("h".to_string(), factor(&h)),
    ]);
    let level = 0.8;
    for ((data, formula), gaussian) in factorial_models().into_iter().zip([true, false]) {
        let (out, f) = data.fit(formula, 1, None, TermTest::Auto);
        let (scale, population, future) = if gaussian {
            (glmm::Scale::Link, true, false)
        } else {
            (glmm::Scale::Response, false, true)
        };
        let mut args = rec(data.model_data(formula));
        let mut nf = Record::new();
        for (k, v) in &new_factors {
            nf.push(k, factor_value(v));
        }
        for (k, v) in [
            ("parts", out.clone()),
            (
                "new_numeric",
                Value::Record(rec(vec![("x", Value::Vec(x.clone()))])),
            ),
            ("new_factors", Value::Record(nf)),
            ("n_rows", Value::I64(4)),
            ("scale", text(if gaussian { "link" } else { "response" })),
            ("re_form", if population { text("NA") } else { Value::Null }),
            ("allow_new_levels", Value::Bool(true)),
            ("se", Value::Bool(true)),
            ("level", Value::F64(level)),
            ("future", Value::Bool(future)),
            ("row_base", Value::I64(0)),
        ] {
            args.push(k, v);
        }
        let p = call(Port::Python, "predict", args).unwrap();

        let parts = glmm::orchestrate::StoredParts {
            beta: f.beta.clone(),
            vcov: f.vcov.clone(),
            model_vcov: f.model_vcov.clone(),
            varcorr: f.varcorr.clone(),
            ranef: f.ranef.clone(),
            ranef_levels: f.ranef_levels.clone(),
            condvar: f.condvar.clone(),
            condvar_x: f.condvar_x.clone(),
            condvar_cross: f.condvar_cross.clone(),
            dispersion: f.dispersion,
            weight_power: f.weight_power,
        };
        let d = glmm::orchestrate::predict(
            formula,
            data.numeric.clone(),
            data.factors.clone(),
            data.kernel_family(),
            None,
            None,
            &parts,
            Some((
                HashMap::from([("x".to_string(), x.clone())]),
                new_factors.clone(),
                4,
            )),
            scale,
            population,
            true,
            "allow_new_levels=True",
            true,
            level,
            future,
        )
        .unwrap();
        for (k, want) in [
            ("fit", &d.fit),
            ("se", &d.se),
            ("lower", &d.lower),
            ("upper", &d.upper),
            ("future_lower", &d.future_lower),
            ("future_upper", &d.future_upper),
        ] {
            same_vec(field(&p, k), want, &format!("{formula} {k}"));
        }
        assert_eq!(f64s(field(&p, "lower")).len(), 4, "{formula}");
        assert_eq!(
            f64s(field(&p, "future_lower")).len(),
            if future { 4 } else { 0 }
        );
        let Value::List(notes) = field(&p, "notes") else {
            panic!("{formula}: notes")
        };
        let kinds: Vec<&Value> = notes.iter().map(|n| field(n, "kind")).collect();
        let want: Vec<Value> = d.notes.iter().map(|n| text(n.kind)).collect();
        assert_eq!(kinds, want.iter().collect::<Vec<_>>(), "{formula}");
    }
}

/// `pairwise` against `orchestrate::pairwise` and `pairwise_columns` on the
/// anova gate's Gaussian LMM, with `by = "h"` and `adjust = "tukey"`.
#[test]
fn pairwise_through_call_matches_pairwise() {
    let [(data, formula), _] = factorial_models();
    let (out, f) = data.fit(formula, 1, None, TermTest::Auto);
    let mut args = rec(data.model_data(formula));
    for (k, v) in [
        ("df_method", field(&out, "df_method").clone()),
        ("beta", field(&out, "beta").clone()),
        ("vcov", field(&out, "vcov").clone()),
        ("model_vcov", field(&out, "model_vcov").clone()),
        ("robust", field(&out, "robust").clone()),
        ("aliased", field(&out, "aliased").clone()),
        ("nobs", field(&out, "nobs").clone()),
        ("record", field(&out, "satterthwaite_record").clone()),
        ("names", field(&out, "names").clone()),
        ("factor", text("g")),
        ("by", text("h")),
        ("adjust", text("tukey")),
        ("converged", Value::Bool(true)),
    ] {
        args.push(k, v);
    }
    let pw = call(Port::Python, "pairwise", args).unwrap();

    let d = glmm::orchestrate::pairwise(
        formula,
        data.numeric.clone(),
        data.factors.clone(),
        data.kernel_family(),
        f.df_method,
        &f.beta,
        &f.vcov,
        &f.model_vcov,
        &f.aliased,
        f.nobs,
        f.satterthwaite_record.clone(),
        f.robust.clone(),
        "g",
        Some("h"),
        None,
        glmm::formula::Adjust::Tukey,
    )
    .unwrap();
    let c = glmm::orchestrate::pairwise_columns(&d, &f.names);
    assert_eq!(c.contrast.len(), 6, "three contrasts in each level of h");
    assert_eq!(strings(field(&pw, "contrast")), c.contrast.as_slice());
    assert_eq!(strings(field(&pw, "by_level")), c.by_level.as_slice());
    assert_eq!(field(&pw, "adjust"), &text(c.adjust));
    same(field(&pw, "levels"), c.levels as f64, "levels");
    for (k, want) in [
        ("estimate", &c.estimate),
        ("se", &c.se),
        ("df", &c.df),
        ("statistic", &c.statistic),
        ("p_value", &c.p_value),
        ("p_adjusted", &c.p_adjusted),
        ("ratio", &c.ratio),
        ("ratio_se", &c.ratio_se),
    ] {
        same_vec(field(&pw, k), want, k);
    }
}

/// `effect_sizes` (the gate's `r2` and `icc`) against
/// `orchestrate::effect_sizes`, on the Gaussian LMM and on the binomial
/// GLMM fitted at `nagq = 5`, whose `objective_nodes` the call reads back.
#[test]
fn effect_sizes_through_call_match_effect_sizes() {
    for ((data, formula), nagq) in factorial_models().into_iter().zip([1, 5]) {
        let (out, f) = data.fit(formula, nagq, None, TermTest::Auto);
        let mut args = rec(data.model_data(formula));
        for (k, v) in [
            ("parts", out.clone()),
            ("aliased", field(&out, "aliased").clone()),
            ("converged", field(&out, "converged").clone()),
            ("objective", field(&out, "objective").clone()),
            ("objective_nodes", field(&out, "objective_nodes").clone()),
            ("null_refit", Value::Bool(true)),
        ] {
            args.push(k, v);
        }
        let e = call(Port::Python, "effect_sizes", args).unwrap();

        let parts = glmm::orchestrate::StoredParts {
            beta: f.beta.clone(),
            vcov: f.vcov.clone(),
            model_vcov: f.model_vcov.clone(),
            varcorr: f.varcorr.clone(),
            ranef: f.ranef.clone(),
            ranef_levels: f.ranef_levels.clone(),
            condvar: f.condvar.clone(),
            condvar_x: f.condvar_x.clone(),
            condvar_cross: f.condvar_cross.clone(),
            dispersion: f.dispersion,
            weight_power: f.weight_power,
        };
        let (r2, icc) = glmm::orchestrate::effect_sizes(
            formula,
            data.numeric.clone(),
            data.factors.clone(),
            data.kernel_family(),
            None,
            None,
            &parts,
            &f.aliased,
            f.converged,
            f.objective,
            true,
        )
        .unwrap()
        .unwrap();
        for (k, want) in [
            ("marginal", r2.marginal),
            ("conditional", r2.conditional),
            ("adjusted", icc.adjusted),
            ("unadjusted", icc.unadjusted),
        ] {
            assert!(want.is_finite(), "{formula} {k}");
            same(field(&e, k), want, &format!("{formula} {k}"));
        }
    }
}

/// The per-term tables of a fit record (the anova gate's term cells) against
/// `orchestrate::term_columns` of `run_fit`'s tables, with `term_test_auto`;
/// and `term_table_text`'s notes, which lead with the table's notes and its
/// rows' notes. The Gaussian LMM at `"auto"` (F rows), the binomial GLMM at
/// `"LRT"` (Wald and likelihood-ratio tables).
#[test]
fn term_tables_match_term_columns() {
    let mut checked = 0;
    for ((data, formula), test) in factorial_models()
        .into_iter()
        .zip([TermTest::Auto, TermTest::Lrt])
    {
        let (out, f) = data.fit(formula, 1, None, test);
        let (auto_test, auto_reason) = glmm::orchestrate::choice_tags(&f.term_test_auto);
        let auto = field(&out, "term_test_auto");
        assert_eq!(field(auto, "test"), &text(auto_test), "{formula}");
        assert_eq!(field(auto, "reason"), &text(auto_reason), "{formula}");
        let lrt = matches!(test, TermTest::Lrt);
        for (name, table, lrt_table) in [
            ("term_table_type3", Some(&f.term_table_type3), false),
            ("term_table_type2", Some(&f.term_table_type2), false),
            (
                "term_table_type3_lrt",
                f.term_table_type3_lrt.as_ref(),
                true,
            ),
            (
                "term_table_type2_lrt",
                f.term_table_type2_lrt.as_ref(),
                true,
            ),
        ] {
            let at = format!("{formula} {name}");
            let stored = field(&out, name);
            let Some(table) = table else {
                assert!(!lrt && stored.is_null(), "{at}");
                continue;
            };
            let c = glmm::orchestrate::term_columns(table.as_ref().unwrap());
            assert_eq!(field(stored, "ss_type"), &text(c.ss_type), "{at}");
            assert_eq!(field(stored, "test"), &text(c.test), "{at}");
            assert_eq!(field(stored, "reason"), &text(c.reason), "{at}");
            assert_eq!(strings(field(stored, "label")), c.label.as_slice(), "{at}");
            for (k, want) in [
                ("statistic", &c.statistic),
                ("num_df", &c.num_df),
                ("den_df", &c.den_df),
                ("p_value", &c.p_value),
                ("sum_sq", &c.sum_sq),
                ("mean_sq", &c.mean_sq),
                ("eta2_partial", &c.eta2_partial),
            ] {
                same_vec(field(stored, k), want, &format!("{at} {k}"));
            }
            let words = call(
                Port::Python,
                "term_table_text",
                rec(vec![
                    ("table", stored.clone()),
                    ("test", text(if lrt_table { "LRT" } else { "auto" })),
                    ("df_method", field(&out, "df_method").clone()),
                    ("converged", Value::Bool(true)),
                    ("term_test_auto", auto.clone()),
                ]),
            )
            .unwrap();
            let lead: Vec<String> =
                c.notes
                    .iter()
                    .cloned()
                    .chain(c.label.iter().zip(&c.row_notes).flat_map(|(label, notes)| {
                        notes.iter().map(move |n| format!("{label}: {n}"))
                    }))
                    .collect();
            let notes = strings(field(&words, "notes"));
            assert!(notes.starts_with(&lead), "{at}: {notes:?} vs {lead:?}");
            assert_eq!(strings(field(&words, "references")).len(), c.label.len());
            checked += 1;
        }
    }
    assert_eq!(checked, 6, "two Gaussian tables, four binomial");
}
