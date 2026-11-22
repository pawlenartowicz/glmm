//! The Rust front end of the anova port gate: one fixed list of model
//! comparisons, per-term tables and post-fit calls, each table written to stdout as one JSONL
//! row per cell. `anova_ports.py` and `anova_ports.R` make the same calls
//! through the Python and R ports, and `compare_anova_ports.R` compares the
//! three files cell by cell at the grid's port band. The three front ends
//! mirror each other call for call: change together.
//!
//! The call list:
//! - every two-model comparison among `validation/manifest.json`'s
//!   `kind = "anova"` entries (the cases in [`PAIR_CASES`], the first two
//!   `members`, the smaller model first), through `compare(data, f0, f1)` and
//!   through `anova(m0, m1)` on the two members fitted one by one, each with
//!   its own `REML` and `nAGQ`;
//! - the per-term tables of [`TERM_CALLS`], on the models of [`TERM_MODELS`]:
//!   the Type III and Type II tables of `y ~ g * h + x` with and without a
//!   random intercept, gaussian on `sim_factorial` and binomial on
//!   `sim_factorial_bern`, at the default test `"auto"` (the lmer F layout
//!   with Satterthwaite's DenDF, the OLS F layout on the residual df, and the
//!   Wald chi-squared layout of a GLMM and of a GLM); the likelihood-ratio
//!   tables (`test = "LRT"` on a fit made with `anova_test = "LRT"`), Type III
//!   and II, of both binomial models and of a binomial GLMM with trials
//!   (`cbpp`); and Type III tables of a Poisson GLMM at `"auto"` (its
//!   likelihood-ratio table) and at an explicit `"Wald"`, and of a negative
//!   binomial and a Gamma GLMM at `"auto"` (Wald), so that every reason
//!   `"auto"` gives a chi-squared table is in some heading; and the Type III
//!   likelihood-ratio table of a binomial GLMM whose random slope is a fixed
//!   term, whose refit without that term keeps the random slope;
//! - one `compare` on `wls_basic` with a missing value in `x2`, which only the
//!   larger formula uses, with each fit's `nobs` as two more cells;
//! - one exact restricted likelihood-ratio test on Dyestuff, at the default
//!   draws and seed, through `compare` and through `anova(m0, m1)`;
//! - the post-fit calls of [`post_fit_calls`] on the Gaussian LMM and the binomial
//!   GLMM of [`TERM_MODELS`] on `sim_factorial`: `predict` with standard errors
//!   on the new rows [`NEW_SUBJ`] ... [`NEW_X`], one of them a `subj` the fit
//!   did not see, and `simulate` (2 draws, seed 1) of both; `r2`, `icc` and
//!   `pairwise` of `g` (Holm and Tukey) of the Gaussian one. A predict table
//!   has R's columns `fit`, `se.fit`, `lower`, `upper`, a simulate table
//!   `sim_1`, `sim_2`, both with rows named `1`, `2`, ...; a pairwise table has
//!   emmeans' numeric columns, with each row named after its contrast.
//!
//! A cell is `{"call", "row", "col", "row_index", "col_index", "value"}`, the
//! value a number, or the string `"NaN"`, `"Inf"` or `"-Inf"` where the table
//! has no finite number (the smaller model's test cells are NaN). A call the
//! front end refuses writes one cell, column `error`, in place of its table.
//! A per-term table is followed by three text cells, `{"call", "row", "col",
//! "row_index", "col_index", "text"}`: row `heading`, column `text`, the
//! table's heading and notes as the port prints them, one line each, joined
//! by newlines; and row `term_test_auto`, columns `test` and `reason`, the
//! fit's tags of the test `"auto"` gives it.
//! `compare_anova_ports.R` pins the call list and the expected refusals.
//!
//! This front end makes every call through `glmm_bind::call`, the entry point
//! both ports call, with the arguments a port passes, and lays each table out
//! the way both ports do (`glmm-py/glmm/anova.py`, `glmm-r/R/fastglmm-anova.R`):
//! R's column names, one row per model with the smaller model first, rows
//! named after the models as given (`m0`/`m1` from `anova`, `f0`/`f1` from
//! `compare`). Its text cells are the lines glmm-bind returns, so the gate
//! checks that each port prints them unaltered.
//!
//! Run from the repository root:
//! `cargo run --release -p validation --example anova_ports > anova_rust.jsonl`

use std::collections::{BTreeSet, HashMap};

use glmm_bind::{Port, Record, Value};
use serde_json::{json, Value as Json};

/// `validation/`, so the example runs from any working directory.
const VALIDATION_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// The `case`s of the manifest's two-model comparisons.
const PAIR_CASES: &[&str] = &[
    "random_pair",
    "ml_pair",
    "ols_pair",
    "glm_pair",
    "glmm_pair",
    "glmm_vs_glm",
    "gamma_glmm_vs_glm",
];

/// One model whose per-term tables are compared.
struct TermModel {
    label: &'static str,
    data: &'static str,
    factors: &'static [&'static str],
    formula: &'static str,
    family: &'static str,
    link: &'static str,
}

/// The models whose per-term tables are compared, by label.
const TERM_MODELS: &[TermModel] = &[
    TermModel {
        label: "lmer_sim_factorial",
        data: "data/simulated/sim_factorial.csv",
        factors: &["subj", "g", "h"],
        formula: "y ~ g * h + x + (1 + x | subj)",
        family: "gaussian",
        link: "identity",
    },
    TermModel {
        label: "lm_sim_factorial",
        data: "data/simulated/sim_factorial.csv",
        factors: &["subj", "g", "h"],
        formula: "y ~ g * h + x",
        family: "gaussian",
        link: "identity",
    },
    TermModel {
        label: "glmer_sim_factorial_bern",
        data: "data/simulated/sim_factorial_bern.csv",
        factors: &["subj", "g", "h"],
        formula: "y ~ g * h + x + (1 | subj)",
        family: "binomial",
        link: "logit",
    },
    TermModel {
        label: "glm_sim_factorial_bern",
        data: "data/simulated/sim_factorial_bern.csv",
        factors: &["subj", "g", "h"],
        formula: "y ~ g * h + x",
        family: "binomial",
        link: "logit",
    },
    TermModel {
        label: "glmer_sim_binomial_slope1",
        data: "data/simulated/sim_binomial_slope1.csv",
        factors: &["g"],
        formula: "y ~ x + (1 + x | g)",
        family: "binomial",
        link: "logit",
    },
    TermModel {
        label: "glmer_cbpp",
        data: "data/empirical/cbpp.csv",
        factors: &["herd", "period"],
        formula: "cbind(incidence, size - incidence) ~ period + (1 | herd)",
        family: "binomial",
        link: "logit",
    },
    TermModel {
        label: "glmer_poisson_grouseticks",
        data: "data/empirical/grouseticks.csv",
        factors: &["YEAR", "BROOD"],
        formula: "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)",
        family: "poisson",
        link: "log",
    },
    TermModel {
        label: "glmer_nb_sim_nb",
        data: "data/simulated/sim_nb.csv",
        factors: &["cluster", "grp"],
        formula: "y ~ 1 + x + grp + (1 | cluster)",
        family: "negativebinomial",
        link: "log",
    },
    TermModel {
        label: "glmer_gamma_sim_gamma",
        data: "data/simulated/sim_gamma.csv",
        factors: &["cluster", "grp"],
        formula: "y ~ 1 + x + grp + (1 | cluster)",
        family: "gamma",
        link: "log",
    },
];

/// One fit's per-term calls: the model (a label in [`TERM_MODELS`]), the
/// tables' types, and `anova`'s `test`. The fit is made with
/// `anova_test = "LRT"` for `test = "LRT"` and at the default `"auto"`
/// otherwise. A call is named `terms:<label>:<type>`, with `:<test>`
/// appended when `named`.
struct TermCall {
    model: &'static str,
    types: &'static [&'static str],
    test: &'static str,
    named: bool,
}

const TERM_CALLS: &[TermCall] = &[
    TermCall {
        model: "lmer_sim_factorial",
        types: &["III", "II"],
        test: "auto",
        named: false,
    },
    TermCall {
        model: "lm_sim_factorial",
        types: &["III", "II"],
        test: "auto",
        named: false,
    },
    TermCall {
        model: "glmer_sim_factorial_bern",
        types: &["III", "II"],
        test: "auto",
        named: false,
    },
    TermCall {
        model: "glm_sim_factorial_bern",
        types: &["III", "II"],
        test: "auto",
        named: false,
    },
    TermCall {
        model: "glmer_sim_factorial_bern",
        types: &["III", "II"],
        test: "LRT",
        named: true,
    },
    TermCall {
        model: "glm_sim_factorial_bern",
        types: &["III", "II"],
        test: "LRT",
        named: true,
    },
    TermCall {
        model: "glmer_cbpp",
        types: &["III", "II"],
        test: "LRT",
        named: true,
    },
    TermCall {
        model: "glmer_sim_binomial_slope1",
        types: &["III"],
        test: "LRT",
        named: true,
    },
    TermCall {
        model: "glmer_poisson_grouseticks",
        types: &["III"],
        test: "auto",
        named: true,
    },
    TermCall {
        model: "glmer_poisson_grouseticks",
        types: &["III"],
        test: "Wald",
        named: true,
    },
    TermCall {
        model: "glmer_nb_sim_nb",
        types: &["III"],
        test: "auto",
        named: true,
    },
    TermCall {
        model: "glmer_gamma_sim_gamma",
        types: &["III"],
        test: "auto",
        named: true,
    },
];

/// The row of `wls_basic.csv` (0-based) whose `x2` is made missing.
const MISSING_ROW: usize = 4;

/// The columns of one CSV, typed as the grid's engines type them: a column
/// named in `factors`, or one with a cell that is not a number, is a factor
/// with its distinct labels in byte order as levels. `read_data` mirrors
/// `grid/engines/common.rs`'s `read_csv_path` and `build_table` -- change together.
struct Data {
    numeric: HashMap<String, Vec<f64>>,
    factor: HashMap<String, (Vec<String>, Vec<u32>)>,
}

fn read_data(path: &str, factors: &[&str]) -> Data {
    let raw = std::fs::read_to_string(format!("{VALIDATION_DIR}/{path}"))
        .unwrap_or_else(|e| panic!("{path}: {e}"));
    let split = |line: &str| -> Vec<String> {
        line.split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect()
    };
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let header = split(lines.next().expect("a CSV header"));
    let rows: Vec<Vec<String>> = lines.map(split).collect();
    let mut data = Data {
        numeric: HashMap::new(),
        factor: HashMap::new(),
    };
    for (j, name) in header.iter().enumerate() {
        let cells: Vec<&str> = rows.iter().map(|r| r[j].as_str()).collect();
        let parsed: Result<Vec<f64>, _> = cells.iter().map(|c| c.parse::<f64>()).collect();
        match parsed {
            Ok(v) if !factors.contains(&name.as_str()) => {
                data.numeric.insert(name.clone(), v);
            }
            _ => {
                data.factor.insert(name.clone(), factor_of(&cells));
            }
        }
    }
    data
}

/// A factor column: its distinct labels in byte order as levels, and each
/// cell's 0-based code.
fn factor_of(cells: &[&str]) -> (Vec<String>, Vec<u32>) {
    let levels: Vec<String> = cells
        .iter()
        .map(|c| c.to_string())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let codes = cells
        .iter()
        .map(|c| levels.iter().position(|l| l == c).expect("a level") as u32)
        .collect();
    (levels, codes)
}

/// `cbind(s, t - s) ~ ...` as `cbind(s, failures) ~ ...` with the column
/// `failures = t - s` added to `data`: the formula frontend takes two column
/// names inside `cbind()`. Any other formula is returned as it is.
fn aggregated(formula: &str, data: &mut Data) -> String {
    let Some(rest) = formula.strip_prefix("cbind(") else {
        return formula.to_string();
    };
    let (inner, tail) = rest.split_once(')').expect("a closed cbind()");
    let (succ, fail) = inner.split_once(',').expect("cbind() of two arguments");
    let Some((total, minus)) = fail.split_once('-') else {
        return formula.to_string();
    };
    let col = |name: &str| data.numeric[name.trim()].clone();
    let failures = col(total)
        .iter()
        .zip(col(minus))
        .map(|(t, s)| t - s)
        .collect();
    data.numeric.insert("failures".to_string(), failures);
    format!("cbind({}, failures){tail}", succ.trim())
}

/// One manifest member's family, link, `REML` and node count.
struct Member {
    formula: String,
    family: String,
    link: String,
    reml: Option<bool>,
    nagq: u8,
}

fn member(m: &Json, data: &mut Data) -> Member {
    let s = |k: &str| m[k].as_str().unwrap_or_else(|| panic!("member {k}"));
    Member {
        formula: aggregated(s("formula"), data),
        family: s("family").to_string(),
        link: s("link").to_string(),
        reml: m["REML"].as_bool(),
        nagq: m["nAGQ"]
            .as_u64()
            .map_or(1, |k| u8::try_from(k).expect("nAGQ fits a u8")),
    }
}

/// Writes one JSONL row per cell of a table: `rows` in order, `cols` as
/// (name, one value per row).
fn table(call: &str, rows: &[&str], cols: &[(&str, Vec<f64>)]) {
    for (i, row) in rows.iter().enumerate() {
        for (j, (col, values)) in cols.iter().enumerate() {
            let v = values[i];
            let value = if v.is_finite() {
                json!(v)
            } else if v.is_nan() {
                json!("NaN")
            } else if v > 0.0 {
                json!("Inf")
            } else {
                json!("-Inf")
            };
            println!(
                "{}",
                json!({"call": call, "row": row, "col": col, "row_index": i,
                       "col_index": j, "value": value})
            );
        }
    }
}

/// The one cell of a call the front end refused.
fn refused(call: &str, why: &str) {
    eprintln!("{call}: refused: {why}");
    table(call, &[""], &[("error", vec![f64::NAN])]);
}

/// `glmm_bind::call` as the Python port makes it; a refusal as its message.
fn call(name: &str, args: Vec<(&str, Value)>) -> Result<Value, String> {
    let mut rec = Record::new();
    for (k, v) in args {
        rec.push(k, v);
    }
    glmm_bind::call(Port::Python, name, rec).map_err(|e| format!("{} ({})", e.message, e.code))
}

fn text_value(s: &str) -> Value {
    Value::Str(s.to_string())
}

/// A field of a record.
fn get<'a>(v: &'a Value, key: &str) -> &'a Value {
    match v {
        Value::Record(r) => r.get(key).unwrap_or(&Value::Null),
        other => panic!("{key}: not a record: {other:?}"),
    }
}

/// A number field (a count crosses as an integer).
fn num(v: &Value, key: &str) -> f64 {
    match get(v, key) {
        Value::F64(x) => *x,
        Value::I64(n) => *n as f64,
        Value::U64(n) => *n as f64,
        other => panic!("{key}: not a number: {other:?}"),
    }
}

/// A float-vector field.
fn nums(v: &Value, key: &str) -> Vec<f64> {
    match get(v, key) {
        Value::Vec(x) => x.clone(),
        Value::List(x) if x.is_empty() => Vec::new(),
        other => panic!("{key}: not a float vector: {other:?}"),
    }
}

/// A string-vector field.
fn strs(v: &Value, key: &str) -> Vec<String> {
    match get(v, key) {
        Value::StrVec(x) => x.clone(),
        Value::List(x) if x.is_empty() => Vec::new(),
        other => panic!("{key}: not a string vector: {other:?}"),
    }
}

/// A string field.
fn string(v: &Value, key: &str) -> String {
    match get(v, key) {
        Value::Str(s) => s.clone(),
        other => panic!("{key}: not a string: {other:?}"),
    }
}

/// The columns of `data` as a port passes them: `numeric` and `factors`.
fn columns(data: &Data) -> [Value; 2] {
    let mut numeric = Record::new();
    let mut names: Vec<&String> = data.numeric.keys().collect();
    names.sort();
    for name in names {
        numeric.push(name, Value::Vec(data.numeric[name].clone()));
    }
    let mut factors = Record::new();
    let mut names: Vec<&String> = data.factor.keys().collect();
    names.sort();
    for name in names {
        factors.push(name, factor_value(&data.factor[name]));
    }
    [Value::Record(numeric), Value::Record(factors)]
}

fn factor_value((levels, codes): &(Vec<String>, Vec<u32>)) -> Value {
    Value::Factor {
        levels: levels.clone(),
        codes: codes.iter().map(|&c| c as i32).collect(),
    }
}

/// A comparison record's table as both ports lay it out; `labels` names the
/// two sides as given.
fn comparison_table(call: &str, r: &Value, labels: [&str; 2]) {
    let large = num(r, "larger") as usize;
    let order = [1 - large, large];
    let Value::List(all) = get(r, "sides") else {
        panic!("{call}: no sides")
    };
    let sides = order.map(|i| &all[i]);
    let rows = order.map(|i| labels[i]);
    let tested = |x: f64| vec![f64::NAN, x];
    let both = |key: &str| sides.iter().map(|s| num(s, key)).collect::<Vec<f64>>();
    let (statistic, df, p) = (num(r, "statistic"), num(r, "df"), num(r, "p_value"));
    let mut cols: Vec<(&str, Vec<f64>)> = Vec::new();
    if string(r, "test") == "lr" {
        cols.push(("npar", both("npar")));
        if sides.iter().all(|s| !get(s, "aic").is_null()) {
            cols.push(("AIC", both("aic")));
            cols.push(("BIC", both("bic")));
        }
        cols.push(("logLik", both("loglik")));
        cols.push(("Chisq", tested(statistic)));
        // The exact reference is not a chi-squared: no Df.
        let exact = string(r, "reference_kind") == "exact";
        cols.push(("Df", tested(if exact { f64::NAN } else { df })));
        cols.push(("Pr(>Chisq)", tested(p)));
    } else {
        cols.push(("Res.Df", both("res_df")));
        let ols = matches!(get(r, "ols"), Value::Bool(true));
        if ols {
            cols.push(("RSS", both("rss")));
        }
        cols.push(("Df", tested(df)));
        if ols {
            cols.push(("Sum of Sq", tested(num(r, "sum_sq"))));
        }
        cols.push(("F", tested(statistic)));
        cols.push(("Pr(>F)", tested(p)));
    }
    table(call, &rows, &cols);
}

/// The native `fit` of member `m` on `data`, as `glmm.fit` passes it.
fn fit(
    data: &Data,
    m: &Member,
    weights: Option<Vec<f64>>,
    anova_test: &str,
) -> Result<Value, String> {
    let [numeric, factors] = columns(data);
    call(
        "fit",
        vec![
            ("formula", text_value(&m.formula)),
            ("numeric", numeric),
            ("factors", factors),
            ("family", text_value(&m.family)),
            ("link", text_value(&m.link)),
            // `glmm.fit`'s defaults.
            ("wald_se", text_value("hessian")),
            ("satterthwaite", Value::Bool(true)),
            ("condvar", Value::Bool(true)),
            ("nagq", Value::I64(i64::from(m.nagq))),
            ("weights", weights.map_or(Value::Null, Value::Vec)),
            ("reml", m.reml.map_or(Value::Null, Value::Bool)),
            ("anova_test", text_value(anova_test)),
            // The weighted members check the weights as given.
            ("weight_power", Value::F64(1.0)),
            ("row_base", Value::I64(0)),
        ],
    )
}

/// What `anova(m0, m1)` reads of a stored fit: its record and its formula.
fn side(m: &Member, r: Value) -> Value {
    let Value::Record(mut rec) = r else {
        panic!("a fit record")
    };
    rec.set("formula", text_value(&m.formula));
    Value::Record(rec)
}

/// `anova(m0, m1)` on the two members fitted one by one.
fn anova_call(name: &str, data: &Data, m: [&Member; 2], weights: Option<Vec<f64>>) {
    let fits = match (
        fit(data, m[0], weights.clone(), "Wald"),
        fit(data, m[1], weights, "Wald"),
    ) {
        (Ok(a), Ok(b)) => [a, b],
        (Err(e), _) | (_, Err(e)) => return refused(name, &e),
    };
    let [a, b] = fits;
    match call(
        "compare_fits",
        vec![("m0", side(m[0], a)), ("m1", side(m[1], b))],
    ) {
        Ok(r) => comparison_table(name, &r, ["m0", "m1"]),
        Err(e) => refused(name, &e),
    }
}

/// `compare(data, f0, f1)`; with `nobs`, each fit's `nobs` as two more cells
/// (call `<call>:nobs`).
fn compare_call(name: &str, data: &Data, m: [&Member; 2], weights: Option<Vec<f64>>, nobs: bool) {
    let [numeric, factors] = columns(data);
    let r = call(
        "compare",
        vec![
            ("f0", text_value(&m[0].formula)),
            ("f1", text_value(&m[1].formula)),
            ("numeric", numeric),
            ("factors", factors),
            ("family", text_value(&m[0].family)),
            ("link", text_value(&m[0].link)),
            ("nagq", Value::I64(i64::from(m[0].nagq.max(m[1].nagq)))),
            ("weights", weights.map_or(Value::Null, Value::Vec)),
            ("weight_power", Value::F64(1.0)),
            ("row_base", Value::I64(0)),
        ],
    );
    match r {
        Ok(r) => {
            comparison_table(name, &r, ["f0", "f1"]);
            if nobs {
                let n = vec![num(get(&r, "fit0"), "nobs"), num(get(&r, "fit1"), "nobs")];
                table(&format!("{name}:nobs"), &["fit0", "fit1"], &[("nobs", n)]);
            }
        }
        Err(e) => refused(name, &e),
    }
}

fn pairs() {
    let manifest: Json = serde_json::from_str(
        &std::fs::read_to_string(format!("{VALIDATION_DIR}/manifest.json")).expect("manifest"),
    )
    .expect("manifest JSON");
    for e in manifest["inference_references"]
        .as_array()
        .expect("inference_references")
    {
        let case = e["case"].as_str().unwrap_or("");
        if e["kind"] != "anova" || !PAIR_CASES.contains(&case) {
            continue;
        }
        let name = e["name"].as_str().expect("a name");
        let factors: Vec<&str> = e["factors"]
            .as_array()
            .expect("factors")
            .iter()
            .map(|f| f.as_str().expect("a factor name"))
            .collect();
        let mut data = read_data(e["data"].as_str().expect("data"), &factors);
        let members = e["members"].as_array().expect("members");
        let m = [
            member(&members[0], &mut data),
            member(&members[1], &mut data),
        ];
        let weights = members[0]["weights"]
            .as_str()
            .map(|w| data.numeric[w].clone());
        compare_call(
            &format!("compare:{name}"),
            &data,
            [&m[0], &m[1]],
            weights.clone(),
            false,
        );
        anova_call(&format!("anova:{name}"), &data, [&m[0], &m[1]], weights);
    }
}

/// The data and the model of a label in [`TERM_MODELS`].
fn term_model(label: &str) -> (Data, Member) {
    let t = TERM_MODELS
        .iter()
        .find(|t| t.label == label)
        .expect("a model of TERM_MODELS");
    let mut data = read_data(t.data, t.factors);
    let m = Member {
        formula: aggregated(t.formula, &mut data),
        family: t.family.to_string(),
        link: t.link.to_string(),
        reml: None,
        nagq: 1,
    };
    (data, m)
}

fn term_calls() {
    for c in TERM_CALLS {
        let (data, m) = term_model(c.model);
        let anova_test = if c.test == "LRT" { "LRT" } else { "auto" };
        let r = fit(&data, &m, None, anova_test);
        for ss in c.types {
            let name = if c.named {
                format!("terms:{}:{ss}:{}", c.model, c.test)
            } else {
                format!("terms:{}:{ss}", c.model)
            };
            let done = match &r {
                Ok(r) => term_table(&name, r, ss, c.test),
                Err(e) => Err(e.clone()),
            };
            if let Err(e) = done {
                refused(&name, &e);
            }
        }
    }
}

/// `anova(m, type = ss, test = test)` on the fit record `r`, as both ports
/// read it: the formula's refusal (the Wald slot's) first, then the refusal
/// of an explicit `"Wald"` or `"LRT"` on F rows; then the likelihood-ratio
/// table for `"LRT"`, and for `"auto"` where the fit's auto choice is the
/// likelihood-ratio test; the Wald table otherwise. Laid out as both ports
/// lay it out (lmerTest's columns with `Eta2_partial` before `Pr(>F)` on F
/// rows, car's on chi-squared rows, Wald or likelihood-ratio), then the text
/// cells: the heading and notes `term_table_text` gives, and the fit's
/// `term_test_auto` tags.
fn term_table(name: &str, r: &Value, ss: &str, test: &str) -> Result<(), String> {
    let field = if ss == "III" {
        "term_table_type3"
    } else {
        "term_table_type2"
    };
    let auto = get(r, "term_test_auto");
    let words = |table: &Value| {
        call(
            "term_table_text",
            vec![
                ("table", table.clone()),
                ("test", text_value(test)),
                ("df_method", get(r, "df_method").clone()),
                ("converged", get(r, "converged").clone()),
                ("term_test_auto", auto.clone()),
            ],
        )
    };
    let wald = get(r, field);
    if !get(wald, "error").is_null() {
        words(wald)?;
    }
    if test != "auto" && get(wald, "reason") == &text_value("f_test") {
        // Both ports refuse with the crate's text, which no cell carries.
        return Err("an explicit test on F rows".to_string());
    }
    let stored = if test == "LRT" || (test == "auto" && string(auto, "test") == "lrt") {
        match get(r, &format!("{field}_lrt")) {
            Value::Null => return Err("no likelihood-ratio table stored".to_string()),
            t => t,
        }
    } else {
        wald
    };
    let words = words(stored)?;
    let labels = strs(stored, "label");
    let f_rows = match strs(stored, "kind").first() {
        Some(kind) => kind == "F",
        None => string(r, "df_method") != "none",
    };
    let cols = if f_rows {
        vec![
            ("Sum Sq", nums(stored, "sum_sq")),
            ("Mean Sq", nums(stored, "mean_sq")),
            ("NumDF", nums(stored, "num_df")),
            ("DenDF", nums(stored, "den_df")),
            ("F value", nums(stored, "statistic")),
            ("Eta2_partial", nums(stored, "eta2_partial")),
            ("Pr(>F)", nums(stored, "p_value")),
        ]
    } else {
        vec![
            ("Chisq", nums(stored, "statistic")),
            ("Df", nums(stored, "num_df")),
            ("Pr(>Chisq)", nums(stored, "p_value")),
        ]
    };
    let rows: Vec<&str> = labels.iter().map(String::as_str).collect();
    let (n, k) = (rows.len(), cols.len());
    table(name, &rows, &cols);
    let lines = [strs(&words, "heading"), strs(&words, "notes")].concat();
    text(name, "heading", "text", [n, k], &lines.join("\n"));
    text(
        name,
        "term_test_auto",
        "test",
        [n + 1, k + 1],
        &string(auto, "test"),
    );
    text(
        name,
        "term_test_auto",
        "reason",
        [n + 1, k + 2],
        &string(auto, "reason"),
    );
    Ok(())
}

/// One text cell (a heading, a tag) at `[row_index, col_index]`.
fn text(call: &str, row: &str, col: &str, at: [usize; 2], s: &str) {
    println!(
        "{}",
        json!({"call": call, "row": row, "col": col, "row_index": at[0],
               "col_index": at[1], "text": s})
    );
}

/// `compare` on `wls_basic` with `x2` missing at [`MISSING_ROW`]: the row is
/// dropped from every column and the weights, as both ports drop it before
/// they call `compare`.
fn missing_value_call() {
    let mut data = read_data("data/simulated/wls_basic.csv", &[]);
    for v in data.numeric.values_mut() {
        v.remove(MISSING_ROW);
    }
    let weights = Some(data.numeric["w"].clone());
    let m = ["y ~ x1", "y ~ x1 + x2"].map(|f| Member {
        formula: f.to_string(),
        family: "gaussian".to_string(),
        link: "identity".to_string(),
        reml: None,
        nagq: 1,
    });
    compare_call(
        "compare:missing_x2_wls_basic",
        &data,
        [&m[0], &m[1]],
        weights,
        true,
    );
}

/// The exact restricted likelihood-ratio test at the default draws and seed:
/// `Yield ~ 1` against `Yield ~ 1 + (1 | Batch)`, both REML.
fn rlrt_calls() {
    let data = read_data("data/empirical/Dyestuff.csv", &["Batch"]);
    let m = ["Yield ~ 1", "Yield ~ 1 + (1 | Batch)"].map(|f| Member {
        formula: f.to_string(),
        family: "gaussian".to_string(),
        link: "identity".to_string(),
        reml: Some(true),
        nagq: 1,
    });
    compare_call("compare:rlrt_dyestuff", &data, [&m[0], &m[1]], None, false);
    anova_call("anova:rlrt_dyestuff", &data, [&m[0], &m[1]], None);
}

/// The new rows of the `predict` calls, on the columns of `sim_factorial`:
/// `s99` is a `subj` the fit did not see. All labels are lowercase, so byte
/// order and R's locale order agree.
const NEW_SUBJ: [&str; 4] = ["s01", "s02", "s99", "s18"];
const NEW_G: [&str; 4] = ["a", "b", "c", "c"];
const NEW_H: [&str; 4] = ["p", "q", "p", "q"];
const NEW_X: [f64; 4] = [0.5, -1.0, 0.0, 1.25];

/// The post-fit calls, each named `<kind>:<label>` (`pairwise` with
/// `:g:<adjust>` appended): `predict` and `simulate` on both models, `r2`,
/// `icc` and `pairwise` on the Gaussian one, each fitted once.
fn post_fit_calls() {
    for (label, gaussian) in [
        ("lmer_sim_factorial", true),
        ("glmer_sim_factorial_bern", false),
    ] {
        let mut calls = vec!["predict", "simulate"];
        if gaussian {
            calls.extend(["r2", "icc", "holm", "tukey"]);
        }
        let name = |kind: &str| match kind {
            "holm" | "tukey" => format!("pairwise:{label}:g:{kind}"),
            _ => format!("{kind}:{label}"),
        };
        let (data, m) = term_model(label);
        match fit(&data, &m, None, "auto") {
            Ok(r) => {
                for kind in calls {
                    if let Err(e) = post_fit(&name(kind), kind, &data, &m, &r) {
                        refused(&name(kind), &e);
                    }
                }
            }
            Err(e) => calls.iter().for_each(|kind| refused(&name(kind), &e)),
        }
    }
}

/// One post-fit call on the fit record `r`, with the training data and the
/// stored numbers both ports pass back.
fn post_fit(name: &str, kind: &str, data: &Data, m: &Member, r: &Value) -> Result<(), String> {
    let [numeric, factors] = columns(data);
    let model_data = vec![
        ("formula", text_value(&m.formula)),
        ("numeric", numeric),
        ("factors", factors),
        ("family", text_value(&m.family)),
        ("link", text_value(&m.link)),
    ];
    let mut parts = Record::new();
    for key in [
        "beta",
        "vcov",
        "model_vcov",
        "varcorr",
        "ranef",
        "ranef_levels",
        "condvar",
        "condvar_x",
        "condvar_cross",
        "dispersion",
        "weight_power",
    ] {
        parts.push(key, get(r, key).clone());
    }
    let with_parts = |rest: Vec<(&'static str, Value)>| {
        let mut args = model_data.clone();
        args.push(("parts", Value::Record(parts.clone())));
        args.extend(rest);
        args
    };
    let numbered = |n: usize| (1..=n).map(|i| i.to_string()).collect::<Vec<_>>();
    match kind {
        "predict" => {
            let new_numeric = Record::new().with("x", Value::Vec(NEW_X.to_vec()));
            let new_factors = Record::new()
                .with("subj", factor_value(&factor_of(&NEW_SUBJ)))
                .with("g", factor_value(&factor_of(&NEW_G)))
                .with("h", factor_value(&factor_of(&NEW_H)));
            let p = call(
                "predict",
                with_parts(vec![
                    ("new_numeric", Value::Record(new_numeric)),
                    ("new_factors", Value::Record(new_factors)),
                    ("n_rows", Value::I64(NEW_X.len() as i64)),
                    ("scale", text_value("link")),
                    ("allow_new_levels", Value::Bool(true)),
                    ("se", Value::Bool(true)),
                    ("level", Value::F64(0.95)),
                    ("future", Value::Bool(false)),
                    ("row_base", Value::I64(0)),
                ]),
            )?;
            let rows = numbered(NEW_X.len());
            let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
            let cols = [
                ("fit", nums(&p, "fit")),
                ("se.fit", nums(&p, "se")),
                ("lower", nums(&p, "lower")),
                ("upper", nums(&p, "upper")),
            ];
            table(name, &rows, &cols);
        }
        "simulate" => {
            let draws = call(
                "simulate",
                with_parts(vec![
                    ("nsim", Value::I64(2)),
                    ("seed", Value::I64(1)),
                    ("row_base", Value::I64(0)),
                ]),
            )?;
            let Value::Vec(draws) = draws else {
                panic!("{name}: draws {draws:?}")
            };
            let nobs = num(r, "nobs") as usize;
            let (first, second) = draws.split_at(nobs);
            let rows = numbered(nobs);
            let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
            table(
                name,
                &rows,
                &[("sim_1", first.to_vec()), ("sim_2", second.to_vec())],
            );
        }
        "r2" | "icc" => {
            let e = call(
                "effect_sizes",
                with_parts(vec![
                    ("aliased", get(r, "aliased").clone()),
                    ("converged", get(r, "converged").clone()),
                    ("objective", get(r, "objective").clone()),
                    ("objective_nodes", get(r, "objective_nodes").clone()),
                    ("null_refit", Value::Bool(true)),
                ]),
            )?;
            let keys = if kind == "r2" {
                ["marginal", "conditional"]
            } else {
                ["adjusted", "unadjusted"]
            };
            let cols = keys.map(|k| (k, vec![num(&e, k)]));
            table(name, &["fit"], &cols);
        }
        adjust => {
            let mut args = model_data;
            args.extend([
                ("df_method", get(r, "df_method").clone()),
                ("beta", get(r, "beta").clone()),
                ("vcov", get(r, "vcov").clone()),
                ("model_vcov", get(r, "model_vcov").clone()),
                ("robust", get(r, "robust").clone()),
                ("aliased", get(r, "aliased").clone()),
                ("nobs", get(r, "nobs").clone()),
                ("record", get(r, "satterthwaite_record").clone()),
                ("names", get(r, "names").clone()),
                ("factor", text_value("g")),
                ("adjust", text_value(adjust)),
                ("converged", get(r, "converged").clone()),
            ]);
            let pw = call("pairwise", args)?;
            let contrast = strs(&pw, "contrast");
            let rows: Vec<&str> = contrast.iter().map(String::as_str).collect();
            let statistic = if string(r, "df_method") == "none" {
                "z.ratio"
            } else {
                "t.ratio"
            };
            // No ratio columns: the Gaussian model's identity link has none.
            let cols = [
                ("estimate", nums(&pw, "estimate")),
                ("SE", nums(&pw, "se")),
                ("df", nums(&pw, "df")),
                (statistic, nums(&pw, "statistic")),
                ("p.value", nums(&pw, "p_adjusted")),
            ];
            table(name, &rows, &cols);
        }
    }
    Ok(())
}

fn main() {
    pairs();
    term_calls();
    missing_value_call();
    rlrt_calls();
    post_fit_calls();
}
