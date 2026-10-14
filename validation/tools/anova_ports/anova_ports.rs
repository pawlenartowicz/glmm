//! The Rust front end of the anova port gate: one fixed list of model
//! comparisons and per-term tables, each table written to stdout as one JSONL
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
//! - the Type III and Type II per-term tables of the four models in
//!   [`TERM_MODELS`] (`y ~ g * h + x` with and without a random intercept,
//!   gaussian on `sim_factorial` and binomial on `sim_factorial_bern`: the
//!   lmer F layout with Satterthwaite's DenDF, the OLS F layout on the
//!   residual df, and the Wald chi-squared layout of a GLMM and of a GLM);
//! - one `compare` on `wls_basic` with a missing value in `x2`, which only the
//!   larger formula uses, with each fit's `nobs` as two more cells;
//! - one exact restricted likelihood-ratio test on Dyestuff, at the default
//!   draws and seed, through `compare` and through `anova(m0, m1)`.
//!
//! A cell is `{"call", "row", "col", "row_index", "col_index", "value"}`, the
//! value a number, or the string `"NaN"`, `"Inf"` or `"-Inf"` where the table
//! has no finite number (the smaller model's test cells are NaN). A call the
//! front end refuses writes one cell, column `error`, in place of its table.
//! `compare_anova_ports.R` pins the call list and the expected refusals.
//!
//! The Rust side has no table object, so this file lays each table out the
//! way both ports do (`python/glmm/anova.py`, `r/R/fastglmm-anova.R`): R's
//! column names, one row per model with the smaller model first, rows named
//! after the models as given (`m0`/`m1` from `anova`, `f0`/`f1` from
//! `compare`).
//!
//! Run from the repository root:
//! `cargo run --release -p validation --example anova_ports > anova_rust.jsonl`

use std::collections::{BTreeSet, HashMap};

use glmm::formula::anova::{compare, Comparison, ComparisonTest, Model, ModelInfo, Side};
use glmm::inference::{Objective, Reference};
use glmm::orchestrate::{self, FitResult};
use serde_json::{json, Value};

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

/// The models whose Type III and Type II per-term tables are compared.
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
];

/// The row of `wls_basic.csv` (0-based) whose `x2` is made missing.
const MISSING_ROW: usize = 4;

/// The columns of one CSV, typed as the grid's engines type them: a column
/// named in `factors`, or one with a cell that is not a number, is a factor
/// with its distinct labels in byte order as levels.
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
                data.factor.insert(name.clone(), (levels, codes));
            }
        }
    }
    data
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

fn member(m: &Value, data: &mut Data) -> Member {
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

/// A comparison's table as both ports lay it out; `labels` names the two
/// sides as given, `rss` their residual sums of squares (NaN off OLS).
fn comparison_table(call: &str, c: &Comparison, labels: [&str; 2], rss: [f64; 2]) {
    let order = match c.larger {
        Side::First => [1, 0],
        Side::Second => [0, 1],
    };
    let infos = [&c.m0, &c.m1];
    let sides = order.map(|i| infos[i]);
    let rows = order.map(|i| labels[i]);
    let tested = |x: f64| vec![f64::NAN, x];
    let both = |f: fn(&ModelInfo) -> f64| sides.map(f).to_vec();
    let mut cols: Vec<(&str, Vec<f64>)> = Vec::new();
    match &c.test {
        ComparisonTest::Lr(t) => {
            cols.push(("npar", both(|s| s.npar as f64)));
            if sides.iter().all(|s| s.aic.is_some()) {
                cols.push(("AIC", both(|s| s.aic.unwrap_or(f64::NAN))));
                cols.push(("BIC", both(|s| s.bic.unwrap_or(f64::NAN))));
            }
            cols.push(("logLik", both(|s| s.loglik)));
            cols.push(("Chisq", tested(t.statistic)));
            // The exact reference is not a chi-squared: no Df.
            let df = match t.reference {
                Reference::ExactRlrt { .. } => f64::NAN,
                _ => t.df as f64,
            };
            cols.push(("Df", tested(df)));
            cols.push(("Pr(>Chisq)", tested(t.p_value)));
        }
        ComparisonTest::Linear(t) => {
            // The residual df n - p, p the retained columns (npar counts
            // sigma or phi too).
            cols.push(("Res.Df", both(|s| s.nobs as f64 - (s.npar as f64 - 1.0))));
            let ols = sides.iter().all(|s| s.objective == Objective::LeastSquares);
            let rss = order.map(|i| rss[i]);
            if ols {
                cols.push(("RSS", rss.to_vec()));
            }
            cols.push(("Df", tested(t.num_df)));
            if ols {
                cols.push(("Sum of Sq", tested(rss[0] - rss[1])));
            }
            cols.push(("F", tested(t.statistic)));
            cols.push(("Pr(>F)", tested(t.p_value)));
        }
        other => panic!("{call}: no table for the test {other:?}"),
    }
    table(call, &rows, &cols);
}

/// What `anova(m0, m1)` reads of a stored fit, as both ports build it.
fn model_of(m: &Member, r: &FitResult) -> Model {
    Model {
        formula: m.formula.clone(),
        family: orchestrate::family_from_str(&m.family, &m.link).expect("a family"),
        loglik: r.loglik,
        df: r.df,
        nobs: r.nobs,
        reml: r.reml,
        objective: r.objective,
        converged: r.converged,
        varcorr: r.varcorr.clone(),
        re_groups: r.re_groups.clone(),
        y: r.y.clone(),
        weights: r.weights.clone(),
        offset: r.offset.clone(),
        dispersion_held: None,
        design_summary: r.design_summary.clone(),
        rss: r.rss,
        coef_names: r.names.clone(),
        coef: r.beta.clone(),
        vcov: r.vcov.clone(),
        aliased: r.aliased.clone(),
    }
}

fn fit(data: &Data, m: &Member, weights: Option<Vec<f64>>) -> Result<FitResult, String> {
    orchestrate::run_fit(
        &m.formula,
        data.numeric.clone(),
        data.factor.clone(),
        &m.family,
        &m.link,
        "hessian",
        m.nagq,
        None,
        weights,
        None,
        None,
        true,
        m.reml,
    )
}

/// `anova(m0, m1)` on the two members fitted one by one.
fn anova_call(call: &str, data: &Data, m: [&Member; 2], weights: Option<Vec<f64>>) {
    let fits = match (fit(data, m[0], weights.clone()), fit(data, m[1], weights)) {
        (Ok(a), Ok(b)) => [a, b],
        (Err(e), _) | (_, Err(e)) => return refused(call, &e),
    };
    let opts = orchestrate::compare_options(None, None).expect("the default options");
    let (a, b) = (model_of(m[0], &fits[0]), model_of(m[1], &fits[1]));
    match compare(&a, &b, &opts) {
        Ok(c) => comparison_table(call, &c, ["m0", "m1"], [fits[0].rss, fits[1].rss]),
        Err(e) => refused(call, &e.to_string()),
    }
}

/// `compare(data, f0, f1)`; with `nobs`, each fit's `nobs` as two more cells
/// (call `<call>:nobs`).
fn compare_call(call: &str, data: &Data, m: [&Member; 2], weights: Option<Vec<f64>>, nobs: bool) {
    let r = orchestrate::run_compare(
        &m[0].formula,
        &m[1].formula,
        data.numeric.clone(),
        data.factor.clone(),
        &m[0].family,
        &m[0].link,
        m[0].nagq.max(m[1].nagq),
        weights,
        None,
        None,
        None,
    );
    match r {
        Ok(r) => {
            comparison_table(call, &r.comparison, ["f0", "f1"], [r.fit0.rss, r.fit1.rss]);
            if nobs {
                let n = vec![r.fit0.nobs as f64, r.fit1.nobs as f64];
                table(&format!("{call}:nobs"), &["fit0", "fit1"], &[("nobs", n)]);
            }
        }
        Err(e) => refused(call, &e),
    }
}

fn pairs() {
    let manifest: Value = serde_json::from_str(
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

fn term_tables() {
    for t in TERM_MODELS {
        let data = read_data(t.data, t.factors);
        let m = Member {
            formula: t.formula.to_string(),
            family: t.family.to_string(),
            link: t.link.to_string(),
            reml: None,
            nagq: 1,
        };
        let r = fit(&data, &m, None);
        for (ss, which) in [("III", 3), ("II", 2)] {
            let call = format!("terms:{}:{ss}", t.label);
            let stored = match &r {
                Ok(r) if which == 3 => &r.term_table_type3,
                Ok(r) => &r.term_table_type2,
                Err(e) => {
                    refused(&call, e);
                    continue;
                }
            };
            let tc = match stored {
                Ok(t) => orchestrate::term_columns(t),
                Err(e) => {
                    refused(&call, e);
                    continue;
                }
            };
            let rows: Vec<&str> = tc.label.iter().map(String::as_str).collect();
            let cols = if tc.kind.first() == Some(&"F") {
                vec![
                    ("Sum Sq", tc.sum_sq),
                    ("Mean Sq", tc.mean_sq),
                    ("NumDF", tc.num_df),
                    ("DenDF", tc.den_df),
                    ("F value", tc.statistic),
                    ("Pr(>F)", tc.p_value),
                ]
            } else {
                vec![
                    ("Chisq", tc.statistic),
                    ("Df", tc.num_df),
                    ("Pr(>Chisq)", tc.p_value),
                ]
            };
            table(&call, &rows, &cols);
        }
    }
}

/// `compare` on `wls_basic` with `x2` missing at [`MISSING_ROW`]: the caller
/// of `run_compare` drops the row from every column and the weights, as both
/// ports do before they call it.
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

fn main() {
    pairs();
    term_tables();
    missing_value_call();
    rlrt_calls();
}
