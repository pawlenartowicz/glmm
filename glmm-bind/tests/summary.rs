mod common;

use common::{f64s, field, fit_args, rec, text};
use glmm_bind::{call, Port, Record, Value};

/// A stored two-coefficient OLS-shaped fit, as a port passes it: every field,
/// a z table unless a test sets `df_method`.
fn fit(extra: Vec<(&str, Value)>) -> Record {
    let mut r = rec(vec![
        ("formula", text("y ~ x")),
        ("family", text("gaussian")),
        ("link", text("identity")),
        ("reml", Value::Bool(false)),
        ("nagq", Value::I64(1)),
        ("nobs", Value::I64(4)),
        ("df", Value::I64(3)),
        ("loglik", Value::F64(-10.0)),
        (
            "names",
            Value::StrVec(vec!["(Intercept)".into(), "x".into()]),
        ),
        ("beta", Value::Vec(vec![2.0, -0.5])),
        ("se", Value::Vec(vec![1.0, 0.25])),
        ("vcov", Value::Vec(vec![1.0, 0.1, 0.1, 0.0625])),
        ("aliased", Value::BoolVec(vec![false, false])),
        ("dispersion", Value::F64(1.0)),
        ("varcorr", Value::List(vec![])),
        ("y", Value::Vec(vec![1.0, 2.0, 3.0, 4.0])),
        ("fitted", Value::Vec(vec![1.5, 1.5, 3.5, 3.5])),
        ("n_eval", Value::I64(0)),
        ("converged", Value::Bool(true)),
        ("singular", Value::Bool(false)),
        ("df_method", text("none")),
        ("coef_df", Value::Vec(vec![f64::NAN; 2])),
        ("coef_p", Value::Vec(vec![f64::NAN; 2])),
        ("rss", Value::F64(f64::NAN)),
        ("r_squared", Value::F64(f64::NAN)),
        ("adj_r_squared", Value::F64(f64::NAN)),
        ("warnings", Value::List(vec![])),
    ]);
    for (k, v) in extra {
        r.set(k, v);
    }
    r
}

fn summary(fit: Record) -> Value {
    call(
        Port::Python,
        "summary",
        rec(vec![("fit", Value::Record(fit))]),
    )
    .unwrap()
}

fn s(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        other => panic!("not a Str: {other:?}"),
    }
}

#[test]
fn an_ml_fit_has_lme4s_criterion_row_and_a_reml_fit_its_criterion() {
    let ml = summary(fit(vec![]));
    let Value::Record(c) = field(&ml, "criterion") else {
        panic!("criterion is not a record")
    };
    assert_eq!(
        c.keys().collect::<Vec<_>>(),
        ["AIC", "BIC", "logLik", "deviance", "df.resid"]
    );
    assert_eq!(c.get("AIC"), Some(&Value::F64(26.0)));
    assert_eq!(c.get("BIC"), Some(&Value::F64(20.0 + 3.0 * 4f64.ln())));
    assert_eq!(c.get("df.resid"), Some(&Value::I64(1)));

    let reml = summary(fit(vec![("reml", Value::Bool(true))]));
    let Value::Record(c) = field(&reml, "criterion") else {
        panic!("criterion is not a record")
    };
    assert_eq!(
        c.iter().collect::<Vec<_>>(),
        [("REML criterion at convergence", &Value::F64(20.0))]
    );
}

/// A mixed fit that did not converge has no variance blocks but keeps its
/// groupings: it keeps the mixed title and its groups, and its df.resid is
/// NaN because df is not counted.
#[test]
fn a_failed_mixed_fit_keeps_its_title_and_groups_and_has_no_df_resid() {
    let failed = |family: &str, reml: bool| {
        summary(fit(vec![
            ("family", text(family)),
            ("reml", Value::Bool(reml)),
            ("converged", Value::Bool(false)),
            ("df", Value::I64(0)),
            ("loglik", Value::F64(f64::NAN)),
            ("beta", Value::Vec(vec![f64::NAN; 2])),
            ("se", Value::Vec(vec![f64::NAN; 2])),
            ("vcov", Value::Vec(vec![f64::NAN; 4])),
            ("dispersion", Value::F64(f64::NAN)),
            ("fitted", Value::Vec(vec![])),
            ("varcorr", Value::List(vec![])),
            (
                "re_groups",
                Value::List(vec![Value::List(vec![
                    text("g"),
                    Value::StrVec(vec!["(Intercept)".into()]),
                ])]),
            ),
            ("ranef_levels", Value::IntVec(vec![6])),
        ]))
    };
    let b = failed("binomial", false);
    assert_eq!(
        s(field(&b, "method")),
        "Generalized linear mixed model fit by maximum likelihood (Laplace Approximation) [glmm]"
    );
    assert_eq!(
        field(field(&b, "groups"), "levels"),
        &Value::IntVec(vec![6])
    );
    assert_eq!(field(&b, "random_effects"), &Value::List(vec![]));
    let Value::Record(c) = field(&b, "criterion") else {
        panic!("criterion is not a record")
    };
    assert!(matches!(c.get("df.resid"), Some(Value::F64(v)) if v.is_nan()));
    let g = failed("gaussian", true);
    assert_eq!(
        s(field(&g, "method")),
        "Linear mixed model fit by REML [glmm]"
    );
}

#[test]
fn the_test_follows_the_df_method() {
    // No df: Wald z on the normal, z = 2 gives p = 0.0455.
    let z = summary(fit(vec![]));
    assert_eq!(s(field(&z, "test")), "z");
    assert!((f64s(field(&z, "p_value"))[0] - 0.045_500_263_896_358_42).abs() < 1e-15);
    assert!(field(&z, "coef_df").is_null() && field(&z, "residual_df").is_null());

    // Satterthwaite: t with the fit's df and p, and the footnote.
    let sat = summary(fit(vec![
        ("df_method", text("satterthwaite")),
        ("coef_df", Value::Vec(vec![15.0, 16.25])),
        ("coef_p", Value::Vec(vec![0.0632, 0.0526])),
    ]));
    assert_eq!(s(field(&sat, "test")), "t");
    assert_eq!(f64s(field(&sat, "coef_df")), [15.0, 16.25]);
    assert_eq!(f64s(field(&sat, "p_value")), [0.0632, 0.0526]);
    assert!(s(field(&sat, "footnote")).starts_with("t tests use Satterthwaite"));

    // Residual df: t, no df column, n - p in the footer.
    let ols = summary(fit(vec![
        ("df_method", text("residual")),
        ("coef_p", Value::Vec(vec![0.18, 0.057])),
    ]));
    assert_eq!(s(field(&ols, "test")), "t");
    assert!(field(&ols, "coef_df").is_null());
    assert_eq!(field(&ols, "residual_df"), &Value::I64(2));

    // Robust df: t with the per-row df column, as Satterthwaite's.
    for m in ["pustejovsky_tipton", "bell_mccaffrey"] {
        let robust = summary(fit(vec![
            ("df_method", text(m)),
            ("coef_df", Value::Vec(vec![5.5, 3.25])),
            ("coef_p", Value::Vec(vec![0.09, 0.12])),
        ]));
        assert_eq!(s(field(&robust, "test")), "t", "{m}");
        assert_eq!(f64s(field(&robust, "coef_df")), [5.5, 3.25], "{m}");
        assert_eq!(f64s(field(&robust, "p_value")), [0.09, 0.12], "{m}");
        // An OLS fit keeps summary.lm's footer; a mixed one never had it.
        let rdf = field(&robust, "residual_df");
        if m == "bell_mccaffrey" {
            assert_eq!(rdf, &Value::I64(2), "{m}");
        } else {
            assert!(rdf.is_null(), "{m}");
        }
    }

    // Switched off or failed: z, and no footnote (each port words its own).
    for m in ["disabled", "failed"] {
        let off = summary(fit(vec![("df_method", text(m))]));
        assert_eq!(s(field(&off, "test")), "z", "{m}");
        assert!(field(&off, "footnote").is_null(), "{m}");
    }
}

#[test]
fn scaled_residuals_need_the_fitted_values() {
    let with = summary(fit(vec![("dispersion", Value::F64(0.25))]));
    // (y - fitted) / 0.5 = [-1, 1, -1, 1].
    assert_eq!(
        f64s(field(&with, "scaled_residuals")),
        [-1.0, -1.0, 0.0, 1.0, 1.0]
    );
    let without = summary(fit(vec![("fitted", Value::Vec(vec![]))]));
    assert!(field(&without, "scaled_residuals").is_null());
}

#[test]
fn an_aliased_row_is_nan_and_counted() {
    let r = summary(fit(vec![("aliased", Value::BoolVec(vec![false, true]))]));
    for name in ["estimate", "std_error", "statistic", "p_value"] {
        let v = f64s(field(&r, name));
        assert!(v[0].is_finite() && v[1].is_nan(), "{name}: {v:?}");
    }
    assert_eq!(field(&r, "n_aliased"), &Value::I64(1));
}

#[test]
fn a_fit_stored_by_0_8_0_without_the_optional_fields_summarizes_as_a_z_table() {
    let optional = [
        "df_method",
        "coef_df",
        "coef_p",
        "rss",
        "r_squared",
        "adj_r_squared",
        "warnings",
    ];
    let mut old = Record::new();
    for (k, v) in fit(vec![]).into_fields() {
        if !optional.contains(&k.as_str()) {
            old.push(&k, v);
        }
    }
    let r = summary(old);
    assert_eq!(s(field(&r, "test")), "z");
    assert!(field(&r, "dispersion_label").is_null());
    assert!(matches!(field(&r, "rss"), Value::F64(x) if x.is_nan()));
    assert_eq!(field(&r, "warnings"), &Value::List(vec![]));
}

#[test]
fn a_mixed_fit_has_its_random_effects_groups_and_effect_sizes() {
    let mixed = fit(vec![
        (
            "varcorr",
            Value::List(vec![Value::Vec(vec![4.0, 1.0, 1.25])]),
        ),
        (
            "re_groups",
            Value::List(vec![Value::List(vec![
                text("Subject"),
                Value::StrVec(vec!["(Intercept)".into(), "Days".into()]),
            ])]),
        ),
        ("re_term_index", Value::List(vec![Value::Index(vec![0, 0])])),
        ("ranef_levels", Value::IntVec(vec![18])),
        ("reml", Value::Bool(true)),
    ]);
    let sizes = rec(vec![
        ("marginal", Value::F64(0.4)),
        ("conditional", Value::F64(0.7)),
        ("adjusted", Value::F64(0.5)),
        ("unadjusted", Value::F64(0.3)),
    ]);
    let r = call(
        Port::Python,
        "summary",
        rec(vec![
            ("fit", Value::Record(mixed)),
            ("effect_sizes", Value::Record(sizes)),
        ]),
    )
    .unwrap();
    assert_eq!(
        s(field(&r, "method")),
        "Linear mixed model fit by REML [glmm]"
    );
    assert_eq!(field(&r, "residual_variance"), &Value::F64(1.0));
    let Value::List(re) = field(&r, "random_effects") else {
        panic!("random_effects is not a list")
    };
    assert_eq!(s(field(&re[0], "group")), "Subject");
    assert_eq!(f64s(field(&re[0], "stddev")), [2.0, 1.25f64.sqrt()]);
    let Value::Mat { data, .. } = field(&re[0], "corr") else {
        panic!("corr is not a matrix")
    };
    assert_eq!(data[1], 1.0 / (2.0 * 1.25f64.sqrt()));
    assert_eq!(
        field(field(&r, "groups"), "levels"),
        &Value::IntVec(vec![18])
    );
    assert_eq!(field(&r, "r2_conditional"), &Value::F64(0.7));
    assert_eq!(field(&r, "r2_need_refit"), &Value::Bool(false));
}

#[test]
fn the_title_suffix_and_ml_note_come_by_port() {
    let mixed = |extra: Vec<(&str, Value)>| {
        let mut r = fit(vec![
            ("varcorr", Value::List(vec![Value::Vec(vec![4.0])])),
            (
                "re_groups",
                Value::List(vec![Value::List(vec![
                    text("g"),
                    Value::StrVec(vec!["(Intercept)".into()]),
                ])]),
            ),
            ("ranef_levels", Value::IntVec(vec![2])),
        ]);
        for (k, v) in extra {
            r.set(k, v);
        }
        r
    };
    let note = text("Note: p-values can be somewhat anticonservative.");
    for (port, title) in [
        (
            Port::Python,
            "Linear mixed model fit by maximum likelihood [glmm]",
        ),
        (
            Port::R,
            "Linear mixed model fit by maximum likelihood [fastglmm]",
        ),
    ] {
        let r = call(
            port,
            "summary",
            rec(vec![("fit", Value::Record(mixed(vec![])))]),
        )
        .unwrap();
        assert_eq!(s(field(&r, "method")), title);
        assert_eq!(field(&r, "ml_note"), &note);
    }
    let reml = summary(mixed(vec![("reml", Value::Bool(true))]));
    assert_eq!(field(&reml, "ml_note"), &Value::Null);
    let failed = summary(mixed(vec![(
        "varcorr",
        Value::List(vec![Value::Vec(vec![f64::NAN])]),
    )]));
    assert_eq!(field(&failed, "ml_note"), &Value::Null);
    assert_eq!(field(&summary(fit(vec![])), "ml_note"), &Value::Null);
}

#[test]
fn pearson_residuals_stay_finite_at_extreme_means() {
    // V = mu^2 (Gamma) leaves the double range here; sqrt(V) does not.
    let r = call(
        Port::Python,
        "pearson_residuals",
        rec(vec![
            ("family", text("gamma")),
            ("y", Value::Vec(vec![1.5e200, 1e-200])),
            ("fitted", Value::Vec(vec![1e200, 2e-200])),
            ("dispersion", Value::F64(0.5)),
        ]),
    )
    .unwrap();
    let want = 0.5f64.sqrt();
    let got = f64s(&r);
    assert!(
        (got[0] - want).abs() < 1e-15 && (got[1] + want).abs() < 1e-15,
        "{got:?}"
    );
}

/// `Σ wᵢ^v dᵢ`, `dᵢ` the family's unit deviance, against the hand-worked
/// Poisson sum `2(0 − (0 − 1)) + 0 + 2(3 ln 3 − 2)` and a Gaussian RSS.
#[test]
fn deviance_residual_sum_matches_the_poisson_formula() {
    let sum = |family: &str, y: Vec<f64>, mu: Vec<f64>, extra: Vec<(&str, Value)>| {
        let mut r = rec(vec![
            ("family", text(family)),
            ("y", Value::Vec(y)),
            ("fitted", Value::Vec(mu)),
            ("dispersion", Value::F64(1.0)),
        ]);
        for (k, v) in extra {
            r.push(k, v);
        }
        match call(Port::Python, "deviance_residual_sum", r).unwrap() {
            Value::F64(v) => v,
            other => panic!("{other:?}"),
        }
    };
    let pois = |extra| sum("poisson", vec![0.0, 1.0, 3.0], vec![1.0; 3], extra);
    let close = |got: f64, want: f64| assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    close(pois(vec![]), 4.591673732008658);
    close(
        pois(vec![("weights", Value::Vec(vec![2.0, 1.0, 1.0]))]),
        6.591673732008658,
    );
    close(
        pois(vec![
            ("weights", Value::Vec(vec![4.0, 1.0, 1.0])),
            ("weight_power", Value::F64(0.5)),
        ]),
        6.591673732008658,
    );
    close(sum("gaussian", vec![1.0, 2.0], vec![0.0, 0.0], vec![]), 5.0);
}

/// The Pearson residuals, alone and in the summary's scaled-residual block,
/// read the weights raised to the fit's `weight_power`; a record without it
/// (a fit saved before it) reads power 1.
#[test]
fn pearson_residuals_read_the_weight_power() {
    let w = vec![1.0, 3.0, 5.0, 7.0];
    let wv: Vec<f64> = w.iter().map(|v: &f64| v.powf(0.5)).collect();
    let pearson = |weights: &[f64], power: Option<f64>| {
        let mut r = rec(vec![
            ("family", text("gaussian")),
            ("y", Value::Vec(vec![1.0, 2.0, 3.0, 4.0])),
            ("fitted", Value::Vec(vec![1.5, 1.5, 3.5, 3.5])),
            ("weights", Value::Vec(weights.to_vec())),
            ("dispersion", Value::F64(2.0)),
        ]);
        if let Some(v) = power {
            r.push("weight_power", Value::F64(v));
        }
        f64s(&call(Port::Python, "pearson_residuals", r).unwrap()).to_vec()
    };
    assert_eq!(pearson(&w, Some(0.5)), pearson(&wv, None));
    assert_ne!(pearson(&w, None), pearson(&wv, None));
    let scaled = |weights: &[f64], power: Option<f64>| {
        let mut extra = vec![("weights", Value::Vec(weights.to_vec()))];
        if let Some(v) = power {
            extra.push(("weight_power", Value::F64(v)));
        }
        f64s(field(&summary(fit(extra)), "scaled_residuals")).to_vec()
    };
    assert_eq!(scaled(&w, Some(0.5)), scaled(&wv, None));
}

fn lines(r: &Value) -> Vec<&str> {
    match field(r, "weight_power_lines") {
        Value::StrVec(v) => v.iter().map(String::as_str).collect(),
        other => panic!("not a StrVec: {other:?}"),
    }
}

const KNOWN: &str = "Standard errors treat the estimated weight power as known.";

/// The power prints when estimated or other than 1; an estimate gets the
/// out-of-range line outside [-1, 2] and, under a model SE, the footnote that
/// the SE treats it as known (on an ML fit, that the simulations estimated
/// it by REML). A robust fit's footnote names CR2 (with its cluster) or HC2
/// and its df.
#[test]
fn the_weight_power_lines_and_the_se_footnote_follow_the_fit() {
    let power = |v: f64, estimated: bool, extra: Vec<(&str, Value)>| {
        let mut e = vec![
            ("weight_power", Value::F64(v)),
            ("weight_power_estimated", Value::Bool(estimated)),
        ];
        e.extend(extra);
        summary(fit(e))
    };
    // A 0.8.0 record (no power) and a claimed fit print no line.
    assert!(lines(&summary(fit(vec![]))).is_empty());
    assert!(lines(&power(1.0, false, vec![])).is_empty());
    assert_eq!(lines(&power(0.5, false, vec![])), ["Weight power: 0.5"]);
    let sat = vec![
        ("df_method", text("satterthwaite")),
        ("coef_df", Value::Vec(vec![15.0, 16.25])),
        ("coef_p", Value::Vec(vec![0.0632, 0.0526])),
    ];
    let reml = power(
        0.734_23,
        true,
        [sat.clone(), vec![("reml", Value::Bool(true))]].concat(),
    );
    assert_eq!(lines(&reml), ["Weight power: 0.7342 (estimated)", KNOWN]);
    assert!(s(field(&reml, "footnote")).starts_with("t tests use Satterthwaite"));
    let ml = power(0.734_23, true, sat);
    assert_eq!(
        lines(&ml)[1],
        "Standard errors treat the estimated weight power as known; the simulations behind it \
         estimated it by REML."
    );

    let robust = |method: &str, v: f64| {
        power(
            v,
            true,
            vec![
                ("reml", Value::Bool(true)),
                ("df_method", text(method)),
                ("coef_df", Value::Vec(vec![5.5, 3.25])),
                ("coef_p", Value::Vec(vec![0.09, 0.12])),
                (
                    "re_groups",
                    Value::List(vec![Value::List(vec![
                        text("Subject"),
                        Value::StrVec(vec!["(Intercept)".into()]),
                    ])]),
                ),
            ],
        )
    };
    let cr2 = robust("pustejovsky_tipton", 2.0);
    assert_eq!(lines(&cr2), ["Weight power: 2 (estimated)"]);
    assert_eq!(
        s(field(&cr2, "footnote")),
        "Standard errors: CR2, clustered by Subject; t tests use Pustejovsky-Tipton degrees of \
         freedom."
    );
    let hc2 = robust("bell_mccaffrey", -3.5);
    assert_eq!(
        lines(&hc2),
        [
            "Weight power: -3.5 (estimated)",
            "The estimated weight power is outside [-1, 2], the range the simulations behind it \
             covered."
        ]
    );
    assert_eq!(
        s(field(&hc2, "footnote")),
        "Standard errors: HC2; t tests use Bell-McCaffrey degrees of freedom."
    );
}

/// A seeded uniform on [0, 1) and a standard normal (Box-Muller).
struct Draws(u64);

impl Draws {
    fn unif(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (1.0 - self.unif(), self.unif());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Review focus 1: on narrow weights (n = 100, max w / min w = 1.2) the
/// estimate lands outside [-1, 2] often with nothing wrong. The default fit
/// succeeds with the outside-[0, 1] note only (factor 1.2^|δ̂ − 1| below 10,
/// so not the implausible one) and its summary prints the out-of-range line.
#[test]
fn narrow_weights_print_the_out_of_range_line_with_the_mild_note() {
    let n = 100;
    let mut d = Draws(1);
    let x: Vec<f64> = (0..n).map(|_| d.normal()).collect();
    let y: Vec<f64> = x.iter().map(|x| 1.0 + 0.5 * x + d.normal()).collect();
    let mut w: Vec<f64> = (0..n).map(|_| 1.0 + 0.2 * d.unif()).collect();
    (w[0], w[1]) = (1.0, 1.2);
    let mut args = fit_args(
        "y ~ x",
        rec(vec![("y", Value::Vec(y)), ("x", Value::Vec(x))]),
        Record::new(),
        0,
    );
    args.push("weights", Value::Vec(w));
    let out = call(Port::Python, "fit", args).unwrap();
    let Value::F64(delta) = *field(&out, "weight_power") else {
        panic!("weight_power is not a number")
    };
    assert!(!(-1.0..=2.0).contains(&delta), "{delta}");
    assert_eq!(field(&out, "weight_power_estimated"), &Value::Bool(true));
    let Value::List(notes) = field(&out, "notes") else {
        panic!("notes is not a list")
    };
    let kinds: Vec<&str> = notes.iter().map(|n| s(field(n, "kind"))).collect();
    assert_eq!(kinds, ["weight_power_outside_unit"]);
    assert_eq!(field(&notes[0], "delta"), &Value::F64(delta));

    // The record a port passes: the fit's own fields plus what the port keeps.
    let mut stored = rec(vec![
        ("formula", text("y ~ x")),
        ("family", text("gaussian")),
        ("link", text("identity")),
        ("nagq", Value::I64(1)),
    ]);
    for k in [
        "reml",
        "nobs",
        "df",
        "loglik",
        "names",
        "beta",
        "se",
        "vcov",
        "aliased",
        "dispersion",
        "varcorr",
        "y",
        "fitted",
        "weights",
        "weight_power",
        "weight_power_estimated",
        "n_eval",
        "converged",
        "singular",
        "df_method",
        "coef_df",
        "coef_p",
    ] {
        stored.push(k, field(&out, k).clone());
    }
    let r = summary(stored);
    let l = lines(&r);
    assert_eq!(l.len(), 2, "{l:?}");
    assert!(l[0].ends_with(" (estimated)") && l[1].contains("outside [-1, 2]"));
    assert!(s(field(&r, "footnote")).starts_with("Standard errors: HC2"));
}
