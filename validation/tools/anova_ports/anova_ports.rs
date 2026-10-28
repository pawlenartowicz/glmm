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
//! The Rust side has no table object, so this file lays each table out the
//! way both ports do (`python/glmm/anova.py`, `r/R/fastglmm-anova.R`): R's
//! column names, one row per model with the smaller model first, rows named
//! after the models as given (`m0`/`m1` from `anova`, `f0`/`f1` from
//! `compare`); and it writes a per-term table's heading as both ports do,
//! from copies of their texts.
//!
//! Run from the repository root:
//! `cargo run --release -p validation --example anova_ports > anova_rust.jsonl`

use std::collections::{BTreeSet, HashMap};

use glmm::formula::anova::{compare_stored, Comparison, ComparisonTest, Model, ModelInfo, Side};
use glmm::inference::{Objective, Reference};
use glmm::orchestrate::{self, FitResult, StoredParts};
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

// The per-term headings' texts as both ports write them
// (`python/glmm/anova.py`, `r/R/fastglmm-anova.R`), copied so that the gate
// compares each port's heading with this one: change together with both.
const LRT_LINE: &str = "Likelihood-ratio chi-squared tests (each term's model refitted without it)";
const WALD_LINE: &str = "Wald chi-squared tests";
const ETA2_LINE: &str =
    "Partial eta-squared uses the Satterthwaite denominator df, so it is approximate.";
const NOT_CONVERGED: &str = "The fit did not converge, so its estimates are not at a maximum \
     and these tests are not reliable. Change the model until it converges (a simpler random \
     part often helps), then test again.";
/// Why `"auto"` gave a chi-squared table its test, by the fit's
/// `term_test_auto` reason (the ports' `_AUTO_REASONS` and `.AUTO_REASONS`).
const AUTO_REASONS: &[(&str, &str)] = &[
    ("poisson_mixed", "Poisson model with random effects"),
    ("binomial_mixed", "binomial model with random effects"),
    (
        "negative_binomial_mixed",
        "negative binomial model with random effects",
    ),
    ("other_mixed", "Gamma model with random effects"),
    ("no_random_effects", "model without random effects"),
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

fn fit(
    data: &Data,
    m: &Member,
    weights: Option<Vec<f64>>,
    anova_test: &str,
) -> Result<FitResult, String> {
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
        true,
        m.reml,
        anova_test,
    )
}

/// `anova(m0, m1)` on the two members fitted one by one.
fn anova_call(call: &str, data: &Data, m: [&Member; 2], weights: Option<Vec<f64>>) {
    let fits = match (
        fit(data, m[0], weights.clone(), "Wald"),
        fit(data, m[1], weights, "Wald"),
    ) {
        (Ok(a), Ok(b)) => [a, b],
        (Err(e), _) | (_, Err(e)) => return refused(call, &e),
    };
    let opts = orchestrate::compare_options(None, None).expect("the default options");
    let (a, b) = (model_of(m[0], &fits[0]), model_of(m[1], &fits[1]));
    match compare_stored(&a, &b, &opts) {
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
            let call = if c.named {
                format!("terms:{}:{ss}:{}", c.model, c.test)
            } else {
                format!("terms:{}:{ss}", c.model)
            };
            match &r {
                Ok(r) => term_table(&call, r, ss, c.test),
                Err(e) => refused(&call, e),
            }
        }
    }
}

/// `anova(m, type = ss, test = test)` on a stored fit, as both ports read it:
/// the formula's refusal (the Wald slot's) first, then the refusal of an
/// explicit `"Wald"` or `"LRT"` on F rows; then the likelihood-ratio table
/// for `"LRT"`, and for `"auto"` where the fit's auto choice is the
/// likelihood-ratio test; the Wald table otherwise. Laid out as both ports
/// lay it out (lmerTest's columns with `Eta2_partial` before `Pr(>F)` on F
/// rows, car's on chi-squared rows, Wald or likelihood-ratio), then the text
/// cells: the heading and the fit's `term_test_auto` tags.
fn term_table(call: &str, r: &FitResult, ss: &str, test: &str) {
    let (wald, lrt) = if ss == "III" {
        (&r.term_table_type3, &r.term_table_type3_lrt)
    } else {
        (&r.term_table_type2, &r.term_table_type2_lrt)
    };
    let (auto_test, auto_reason) = orchestrate::choice_tags(&r.term_test_auto);
    let wald = match wald {
        Ok(t) => t,
        Err(e) => return refused(call, e),
    };
    if test != "auto" && f_rows(&orchestrate::term_columns(wald), r.df_method) {
        // Both ports refuse with the crate's text, which no cell carries.
        return refused(call, "an explicit test on F rows");
    }
    let stored = if test == "LRT" || (test == "auto" && auto_test == "lrt") {
        match lrt {
            Some(Ok(t)) => t,
            Some(Err(e)) => return refused(call, e),
            None => return refused(call, "no likelihood-ratio table stored"),
        }
    } else {
        wald
    };
    let tc = orchestrate::term_columns(stored);
    let heading = heading(&tc, r.df_method, test, auto_reason, r.converged);
    let rows: Vec<&str> = tc.label.iter().map(String::as_str).collect();
    let cols = if f_rows(&tc, r.df_method) {
        vec![
            ("Sum Sq", tc.sum_sq),
            ("Mean Sq", tc.mean_sq),
            ("NumDF", tc.num_df),
            ("DenDF", tc.den_df),
            ("F value", tc.statistic),
            ("Eta2_partial", tc.eta2_partial),
            ("Pr(>F)", tc.p_value),
        ]
    } else {
        vec![
            ("Chisq", tc.statistic),
            ("Df", tc.num_df),
            ("Pr(>Chisq)", tc.p_value),
        ]
    };
    let (n, k) = (rows.len(), cols.len());
    table(call, &rows, &cols);
    text(call, "heading", "text", [n, k], &heading);
    text(call, "term_test_auto", "test", [n + 1, k + 1], auto_test);
    text(
        call,
        "term_test_auto",
        "reason",
        [n + 1, k + 2],
        auto_reason,
    );
}

/// Whether a per-term table has F rows, as both ports tell: by its rows, or
/// on a table with none by the fit's `df_method`.
fn f_rows(tc: &orchestrate::TermColumns, df_method: &str) -> bool {
    match tc.kind.first() {
        Some(kind) => *kind == "F",
        None => df_method != "none",
    }
}

/// A per-term table's heading and notes, one line each, as both ports print
/// them: the title (an F table names Satterthwaite's method when some row
/// has its df, or on a table with no rows when the fit's `df_method` is
/// Satterthwaite's; a likelihood-ratio table, or any chi-squared table under
/// `"auto"`, takes a second line naming the test and why it ran; an explicit
/// `"Wald"` keeps one line), then the not-converged note, the table's notes,
/// each row's notes, and on a Satterthwaite F table the note on its partial
/// eta-squared.
fn heading(
    tc: &orchestrate::TermColumns,
    df_method: &str,
    test: &str,
    auto_reason: &str,
    converged: bool,
) -> String {
    let ss = tc.ss_type;
    let mut lines = Vec::new();
    let satterthwaite = if tc.label.is_empty() {
        df_method == "satterthwaite"
    } else {
        tc.den_df_kind
            .iter()
            .zip(&tc.den_df)
            .any(|(kind, d)| *kind == "satterthwaite" && d.is_finite())
    };
    if f_rows(tc, df_method) {
        let method = if satterthwaite {
            " with Satterthwaite's method"
        } else {
            ""
        };
        lines.push(format!("Type {ss} Analysis of Variance Table{method}"));
    } else if tc.test == "lrt" || test == "auto" {
        let line = if tc.test == "lrt" {
            let reduced = if ss == "III" {
                " on the reduced design"
            } else {
                ""
            };
            format!("{LRT_LINE}{reduced}")
        } else {
            WALD_LINE.to_string()
        };
        let why = if test == "LRT" {
            "test = \"LRT\"".to_string()
        } else {
            let reason = AUTO_REASONS
                .iter()
                .find(|(tag, _)| *tag == auto_reason)
                .unwrap_or_else(|| panic!("no text for the reason {auto_reason}"));
            format!("test = \"auto\": {}", reason.1)
        };
        lines.push(format!("Type {ss} Analysis of Deviance Table"));
        lines.push(format!("{line}; {why}"));
    } else {
        lines.push(format!(
            "Type {ss} Analysis of Deviance Table (Wald chi-squared tests)"
        ));
    }
    if !converged {
        lines.push(NOT_CONVERGED.to_string());
    }
    lines.extend(tc.notes.iter().cloned());
    for (label, notes) in tc.label.iter().zip(&tc.row_notes) {
        lines.extend(notes.iter().map(|n| format!("{label}: {n}")));
    }
    if f_rows(tc, df_method) && satterthwaite {
        lines.push(ETA2_LINE.to_string());
    }
    lines.join("\n")
}

/// One text cell (a heading, a tag) at `[row_index, col_index]`.
fn text(call: &str, row: &str, col: &str, at: [usize; 2], s: &str) {
    println!(
        "{}",
        json!({"call": call, "row": row, "col": col, "row_index": at[0],
               "col_index": at[1], "text": s})
    );
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

/// One post-fit call on a stored fit, through the orchestrate functions both
/// ports call, with the parts both ports store.
fn post_fit(call: &str, kind: &str, data: &Data, m: &Member, r: &FitResult) -> Result<(), String> {
    let parts = StoredParts {
        beta: r.beta.clone(),
        vcov: r.vcov.clone(),
        varcorr: r.varcorr.clone(),
        ranef: r.ranef.clone(),
        ranef_levels: r.ranef_levels.clone(),
        condvar: r.condvar.clone(),
        condvar_x: r.condvar_x.clone(),
        condvar_cross: r.condvar_cross.clone(),
        dispersion: r.dispersion,
    };
    let (numeric, factor) = (data.numeric.clone(), data.factor.clone());
    let numbered = |n: usize| (1..=n).map(|i| i.to_string()).collect::<Vec<_>>();
    match kind {
        "predict" => {
            let new_numeric = HashMap::from([("x".to_string(), NEW_X.to_vec())]);
            let new_factor = HashMap::from([
                ("subj".to_string(), factor_of(&NEW_SUBJ)),
                ("g".to_string(), factor_of(&NEW_G)),
                ("h".to_string(), factor_of(&NEW_H)),
            ]);
            let p = orchestrate::predict(
                &m.formula,
                numeric,
                factor,
                &m.family,
                &m.link,
                None,
                None,
                &parts,
                Some((new_numeric, new_factor, NEW_X.len())),
                "link",
                false,
                true,
                true,
                0.95,
                false,
            )?;
            let rows = numbered(NEW_X.len());
            let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
            let cols = [
                ("fit", p.fit),
                ("se.fit", p.se),
                ("lower", p.lower),
                ("upper", p.upper),
            ];
            table(call, &rows, &cols);
        }
        "simulate" => {
            let draws = orchestrate::simulate(
                &m.formula, numeric, factor, &m.family, &m.link, None, None, &parts, 2, 1,
            )?;
            let (first, second) = draws.split_at(r.nobs);
            let rows = numbered(r.nobs);
            let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
            table(
                call,
                &rows,
                &[("sim_1", first.to_vec()), ("sim_2", second.to_vec())],
            );
        }
        "r2" | "icc" => {
            let (r2, icc) = orchestrate::effect_sizes(
                &m.formula,
                numeric,
                factor,
                &m.family,
                &m.link,
                None,
                None,
                &parts,
                &r.aliased,
                r.converged,
                orchestrate::objective_parts(r.objective),
                true,
            )?
            .expect("a null refit is allowed");
            let cols = if kind == "r2" {
                [
                    ("marginal", vec![r2.marginal]),
                    ("conditional", vec![r2.conditional]),
                ]
            } else {
                [
                    ("adjusted", vec![icc.adjusted]),
                    ("unadjusted", vec![icc.unadjusted]),
                ]
            };
            table(call, &["fit"], &cols);
        }
        adjust => {
            let pw = orchestrate::pairwise(
                &m.formula,
                numeric,
                factor,
                &m.family,
                &m.link,
                r.df_method,
                &r.beta,
                &r.vcov,
                &r.aliased,
                r.nobs,
                r.satterthwaite_record.clone(),
                "g",
                None,
                None,
                adjust,
            )?;
            let pc = orchestrate::pairwise_columns(&pw, &r.names);
            let rows: Vec<&str> = pc.contrast.iter().map(String::as_str).collect();
            let statistic = if r.df_method == "none" {
                "z.ratio"
            } else {
                "t.ratio"
            };
            // No ratio columns: the Gaussian model's identity link has none.
            let cols = [
                ("estimate", pc.estimate),
                ("SE", pc.se),
                ("df", pc.df),
                (statistic, pc.statistic),
                ("p.value", pc.p_adjusted),
            ];
            table(call, &rows, &cols);
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
