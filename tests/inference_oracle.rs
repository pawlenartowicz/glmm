//! The inference tier's end-to-end gates: glmm's df, p-values and test statistics
//! against the frozen references under `validation/goldens/inference/`.
//!
//! Inference tests fall in three classes:
//!
//! - fit-free arithmetic (a formula evaluated on frozen inputs): hard bands, no
//!   registry, in the default test matrix;
//! - end-to-end against a frozen reference (a fit, then its inference compared
//!   with the reference's): registry-backed, under `oracle-tests`, here for the
//!   public surface;
//! - Monte Carlo (seeded draws, bands from DKW or binomial standard errors), under
//!   `montecarlo-tests`.
//!
//! The bands are `TOL_INFERENCE` in `validation/grid/tol.R`, mirrored in
//! `oracle_support::tol`.
//!
//! The registry rule: an end-to-end quantity past its band passes only under an
//! `oracle-tier` entry of `validation/grid/divergences.json` whose `cell` is the
//! reference's name and whose `quantities` hold `df`, `p_value`, `lrt`, `f` or
//! `confint` ([`QUANTITIES`]). The in-crate gate at lme4's θ̂ compares the same
//! references under its own labels (`df_at_lme4`, `dendf_at_lme4`,
//! `f_at_lme4`), so each driver reads only the entries under its labels.
//! [`inference_tier`] then fails on every such entry on a compared reference that
//! did not fire, because the goldens tier's stale-entry check skips every `cell`
//! that is not one of its goldens.
//!
//! The lmerTest references come on both criteria: the REML ones and the `_ml`
//! ones (`REML = FALSE`), each fitted with `FitOptions::reml` from the entry's
//! `reml`. On an ML reference [`lmertest_end_to_end`] also checks glmm's ML
//! criterion against lme4's frozen `deviance`, hard and two-sided as the goldens
//! tier's deviance gate (`tests/validation_oracle.rs`): `−2·Fit::loglik` minus
//! the reference at most `DEV_EPS`, and at most `DEV_BIG` in absolute value (a
//! larger gap below the reference is a constants bug). It has no registry
//! quantity.
//!
//! The model comparisons and per-term tables (`kind` `anova`): [`anova_pairs`]
//! fits both sides of each pair, and of each `ranova` row, and compares them
//! through [`compare`], the deviance gate first; [`anova_term_rows`] reads
//! car's tables through [`term_tests`]. A per-term statistic, an F or a Wald
//! χ², is the quantity `f`.
//!
//! Where a gate goes: an integration gate on the public surface joins [`GATES`].
//! A gate on a `pub(crate)` item cannot be reached from here, so it lives in an
//! in-crate test file with its own driver, which asserts the same `stale_entries`
//! result empty on its own compared names and fired set.
//!
//! ```sh
//! cargo test --features oracle-tests --test inference_oracle
//! ```
//!
//! `oracle-tests` implies `orchestrate` (and so `formula`), which brings the
//! ports' interval function [`glmm::orchestrate::confint`] that
//! [`lmertest_confint`] gates.
#![cfg(feature = "oracle-tests")]

// Shared with `tests/validation_oracle.rs`, which uses the rest of it; this file
// uses the registry, the inference reader and lowering, `assert_rel`,
// `assert_rel_to` and the inference bands of `tol`.
#[allow(dead_code)]
mod oracle_support;

use glmm::formula::anova::{
    compare, term_tests, CompareOptions, ComparisonTest, Model, SsType, TermTestOptions,
};
use glmm::formula::Lowered;
use glmm::inference::{LinearTest, LrTest, Reference};
use glmm::{BinomialLink, Family, Fit, GammaLink, NegBinomialLink, Note, PoissonLink, WaldSe};
use oracle_support::{assert_rel, assert_rel_to, divergence, inference, inference_lowered, tol};
use serde_json::Value;
use std::collections::BTreeSet;
use std::sync::OnceLock;

/// The registry quantities this driver adjudicates; an entry under any other
/// label belongs to another driver.
const QUANTITIES: &[&str] = &["df", "p_value", "lrt", "f", "confint"];

/// One end-to-end gate: it adds the name of every reference it compared.
type Gate = fn(&mut BTreeSet<String>);

/// The end-to-end gates on the public surface, as (name, gate), run in order by
/// [`inference_tier`].
const GATES: &[(&str, Gate)] = &[
    ("lmertest_end_to_end", lmertest_end_to_end),
    ("lmertest_confint", lmertest_confint),
    ("anova_pairs", anova_pairs),
    ("anova_term_rows", anova_term_rows),
];

fn numbers(v: &Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what}: not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("{what}: {x}")))
        .collect()
}

/// A reference's name, JSON, `fit_cold` fit and the caller column of each of
/// its coefficients.
type RefFit = (String, Value, Fit, Vec<usize>);

/// The per-coefficient lmerTest references of both criteria (X7: kind
/// `lmertest` with `proposal_dependent` false; `reml` true or false), each
/// fitted end to end by `fit_cold` on its lowered formula (the ML ones with
/// `reml = Some(false)`), with its JSON and the caller column of each of the
/// reference's coefficients. Fitted once per process, for every gate that
/// reads them.
fn lmertest_fits() -> &'static [RefFit] {
    static FITS: OnceLock<Vec<RefFit>> = OnceLock::new();
    FITS.get_or_init(fit_lmertest_references)
}

fn fit_lmertest_references() -> Vec<RefFit> {
    let out: Vec<_> = inference::entries(inference::Kind::Lmertest)
        .into_iter()
        .filter(|e| {
            let flag = e.raw["proposal_dependent"].as_bool();
            let reml = e.raw["reml"].as_bool();
            assert!(
                flag.is_some() && reml.is_some(),
                "{}: proposal_dependent or reml is not a bool",
                e.name
            );
            flag == Some(false)
        })
        .map(|e| {
            assert!(!e.is_pending(), "{}: not frozen", e.name);
            let json = inference::load(inference::Kind::Lmertest, &e.name);
            let lo = inference_lowered(&e);
            let fit = glmm::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
            assert_eq!(
                Some(fit.reml),
                e.raw["reml"].as_bool(),
                "{}: the fit's criterion",
                e.name
            );
            let cols = json["coef_names"]
                .as_array()
                .unwrap_or_else(|| panic!("{}: coef_names", e.name))
                .iter()
                .map(|c| {
                    let c = c.as_str().expect("a coefficient name");
                    lo.col_names
                        .iter()
                        .position(|n| n == c)
                        .unwrap_or_else(|| panic!("{}: no design column {c}", e.name))
                })
                .collect();
            (e.name, json, fit, cols)
        })
        .collect();
    let n_ml = out.iter().filter(|(_, json, _, _)| is_ml(json)).count();
    eprintln!(
        "LMERTEST {} per-coefficient references fitted, {n_ml} of them ML",
        out.len()
    );
    assert!(
        n_ml > 0 && n_ml < out.len(),
        "the per-coefficient lmertest references of both criteria"
    );
    out
}

/// An `_ml` reference: lmerTest fitted with `REML = FALSE`.
fn is_ml(json: &Value) -> bool {
    json["reml"].as_bool() == Some(false)
}

/// Gate 1 end to end: glmm's `coef_df` at its own θ̂ against lmerTest's df,
/// per coefficient, within `DF_E2E_REL` (the cross-engine SE band: df moves
/// with θ̂ as the SE does). t and p follow from df, β̂ and SE. Each fit's pass
/// takes the derivative route the fit arms (the gradient route where the
/// per-group gradient is armed, the value route elsewhere), here and in
/// [`lmertest_confint`]; the in-crate gate at lme4's θ̂ runs both routes. On
/// an ML reference, first the ML criterion against lme4's: `−2·Fit::loglik`
/// against the frozen `deviance`, two-sided (see the module doc).
fn lmertest_end_to_end(compared: &mut BTreeSet<String>) {
    for (name, json, fit, cols) in lmertest_fits() {
        if is_ml(json) {
            let dev_ref = json["deviance"].as_f64().expect("deviance");
            let d = -2.0 * fit.loglik - dev_ref;
            eprintln!("LMERTEST {name}: ML deviance gap {d:.3e}");
            assert!(
                d.abs() <= oracle_support::DEV_BIG,
                "{name}: |Δdev|={d:.3e} > DEV_BIG, suspected convention mismatch"
            );
            assert!(
                d <= oracle_support::DEV_EPS,
                "{name}: Δdev={d:.3e} > DEV_EPS, worse optimum than lme4"
            );
        }
        let want = numbers(&json["df"], name);
        let worst = cols
            .iter()
            .enumerate()
            .map(|(i, &j)| ((fit.coef_df[j] - want[i]) / want[i]).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "LMERTEST {name}: df max rel {worst:.2e} (band {:.0e})",
            tol::DF_E2E_REL
        );
        for (i, &j) in cols.iter().enumerate() {
            assert_rel(
                fit.coef_df[j],
                want[i],
                tol::DF_E2E_REL,
                &format!("{name}: df[{i}]"),
            );
        }
        compared.insert(name.clone());
    }
}

/// Gate 1's interval comparison: the ports' Satterthwaite t interval
/// ([`glmm::orchestrate::confint`]: `β̂ ± t_{df}(1 − (1 − level)/2)·SE` on a
/// finite positive df) of each coefficient, on glmm's `coef_df` at its own θ̂,
/// against lmerTest's `contest1D(m, L, confint = TRUE)` bounds at the frozen
/// `level`. Each bound's difference is measured in units of lmerTest's
/// interval width, within `CONFINT_REL` (registry quantity `confint`): a bound
/// may sit near 0, where a difference relative to the bound itself has no
/// fixed scale.
fn lmertest_confint(compared: &mut BTreeSet<String>) {
    for (name, json, fit, cols) in lmertest_fits() {
        let level = json["level"].as_f64().expect("level");
        let (lower, upper) = (numbers(&json["lower"], name), numbers(&json["upper"], name));
        let ci = glmm::orchestrate::confint(&fit.beta, &fit.se, &fit.coef_df, level)
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        let worst = cols
            .iter()
            .enumerate()
            .flat_map(|(i, &j)| {
                let width = upper[i] - lower[i];
                [ci[j][0] - lower[i], ci[j][1] - upper[i]].map(|d| (d / width).abs())
            })
            .fold(0.0, f64::max);
        eprintln!(
            "LMERTEST {name}: confint max |diff|/width {worst:.2e} (band {:.0e})",
            tol::CONFINT_REL
        );
        for (i, &j) in cols.iter().enumerate() {
            let [lo, hi] = ci[j];
            let width = upper[i] - lower[i];
            let ctx = format!("{name}: confint[{i}]");
            assert_rel_to(lo, lower[i], width, tol::CONFINT_REL, &ctx);
            assert_rel_to(hi, upper[i], width, tol::CONFINT_REL, &ctx);
        }
        compared.insert(name.clone());
    }
}

// ── Model comparisons and per-term rows ─────────────────────────────────────

/// The anova cases that are a pair of fits (or rows of pairs) compared by
/// [`compare`]; the other cases are per-term tables or fit-free.
const PAIR_CASES: &[&str] = &[
    "random_pair",
    "ml_pair",
    "ranova",
    "ols_pair",
    "glm_pair",
    "glmm_pair",
    "glmm_vs_glm",
    "gamma_glmm_vs_glm",
];

/// The anova cases whose frozen rows are car's per-term tables, or the one
/// linear hypothesis, read through [`term_tests`].
const TERM_CASES: &[&str] = &["car_glmm", "car_glm", "car_lm", "linear_hypothesis"];

/// The pairs with a model glmm's formula frontend refuses: a random
/// slope whose variable is not a fixed term (`Reaction ~ 1 + (Days |
/// Subject)`, `formula::Error::SlopeVarNotInDesign`, a documented limit).
/// The gate asserts the refusal and compares nothing, so the name is not
/// recorded; once the frontend admits the model, the assertion goes red and
/// the pair joins the gate.
const REFUSED_PAIRS: &[&str] = &["anova_ml_pair_sleepstudy_fixed"];

/// Pair sides whose frozen `logLik` is lme4's `glmer.nb`, as (pair, side,
/// golden): `glmer.nb` maximises another objective than the Laplace
/// approximation glmm and glmmTMB maximise (the goldens tier's registered
/// `sim_nb_glmm` entries), so its `logLik` cannot adjudicate glmm's optimum.
/// The deviance gate reads that side against the frozen glmmTMB fit of the
/// same model and data instead, and the pair's statistic, off by the two
/// objectives' gap, may pass under a registry entry.
const LME4_NB_SIDES: &[(&str, usize, &str)] = &[("anova_glmm_vs_glm_sim_nb", 1, "sim_nb_glmm_tmb")];

/// The Gamma GLM pair's gated reference: the Wald F of this entry, on the
/// same larger model.
const GAMMA_WALD_F: &str = "anova_linear_hypothesis_glm_gamma";

/// The exact test's draws on the pairs inside its envelope (one unweighted
/// `(1 | g)` against OLS, both REML). Only their statistic is gated here
/// (their p is gated in `src/inference/rlrt_oracle_tests.rs`), so any valid
/// count does.
const ENVELOPE_OPTS: CompareOptions = CompareOptions {
    nsim: 1000,
    seed: 20261012,
};

/// One fitted side of a comparison, or the one fit of a per-term table.
struct Side {
    lo: Lowered,
    fit: Fit,
    model: Model,
}

/// A manifest member's family and link.
fn member_family(m: &Value) -> Family {
    match (
        m["family"].as_str().expect("a member family"),
        m["link"].as_str().expect("a member link"),
    ) {
        ("gaussian", "identity") => Family::Gaussian,
        ("binomial", "logit") => Family::Binomial {
            link: BinomialLink::Logit,
        },
        ("poisson", "log") => Family::Poisson {
            link: PoissonLink::Log,
        },
        ("gamma", "log") => Family::Gamma {
            link: GammaLink::Log,
        },
        ("negativebinomial", "log") => Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        (f, l) => panic!("no family {f} with link {l}"),
    }
}

/// glmm's fit of `formula` on `e`'s data with the member's family, weights
/// and `nAGQ`, and `FitOptions::reml = reml`.
fn fit_side(e: &inference::Entry, member: &Value, formula: &str, reml: Option<bool>) -> Side {
    let family = member_family(member);
    let mut lo = oracle_support::lower_on_entry(e, formula, family, member["weights"].as_str())
        .unwrap_or_else(|err| panic!("{} ({formula}): {err}", e.name));
    lo.opts.reml = reml;
    if let Some(k) = member["nAGQ"].as_u64() {
        lo.opts.nagq = u8::try_from(k).expect("nAGQ fits a u8");
    }
    let fit = glmm::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let model = Model::from_fit(&oracle_support::glmm_formula(formula), &lo, &fit);
    Side { lo, fit, model }
}

/// A member fitted as its case needs: a Gaussian member on the criterion of
/// its `REML` (`lm` has none: `None`), every other family by ML (`None`).
fn fit_member(e: &inference::Entry, member: &Value) -> Side {
    let formula = member["formula"].as_str().expect("a member formula");
    let reml = match member_family(member) {
        Family::Gaussian => member["REML"].as_bool(),
        _ => None,
    };
    fit_side(e, member, formula, reml)
}

fn number(v: &Value, what: &str) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("{what}: not a number: {v}"))
}

/// The deviance gate, read before a pair's statistic: per side,
/// `−2·Fit::loglik` minus `−2·logLik` frozen for the same model (never the
/// frozen `deviance`, whose scale differs by fit), at most `DEV_EPS` (glmm's
/// optimum no worse than the oracle's; worse is glmm's bug) and at most
/// `DEV_BIG` in absolute value (a convention mismatch). Returns whether glmm
/// improves on the oracle by more than `DEV_EPS` on either side, when a
/// statistic past its band may pass under a registry entry (the other case is
/// [`LME4_NB_SIDES`]).
fn deviance_gate(ctx: &str, sides: [(&Fit, f64); 2]) -> bool {
    let mut improved = false;
    for (i, (fit, loglik)) in sides.into_iter().enumerate() {
        let d = -2.0 * fit.loglik + 2.0 * loglik;
        eprintln!("ANOVA {ctx}: side {i} deviance gap {d:.3e}");
        assert!(
            d.abs() <= oracle_support::DEV_BIG,
            "{ctx}: side {i} |Δdev|={d:.3e} > DEV_BIG, suspected convention mismatch"
        );
        assert!(
            d <= oracle_support::DEV_EPS,
            "{ctx}: side {i} Δdev={d:.3e} > DEV_EPS, worse optimum than the oracle"
        );
        improved |= d < -oracle_support::DEV_EPS;
    }
    improved
}

/// One quantity of a pair against its frozen value: `|got − want| / scale`
/// at most `band` (`scale` 1 for an absolute band, `|want|` for a relative
/// one). Past the band it passes only under a registry entry (`ctx` is
/// `"<name>: <quantity>"`), and only where the deviance gate found glmm's
/// optimum the better one; otherwise a miss is red.
fn pair_quantity(ctx: &str, got: f64, want: f64, scale: f64, band: f64, improved: bool) {
    let d = (got - want).abs() / scale;
    eprintln!("ANOVA {ctx}: glmm {got:.10e} oracle {want:.10e} diff {d:.3e} (band {band:.0e})");
    if improved {
        assert_rel_to(got, want, scale, band, ctx);
    } else {
        assert!(
            d <= band,
            "{ctx}: glmm={got} oracle={want} ({d:.3e} > {band:.0e}) where glmm's deviance \
             is not better than the oracle's"
        );
    }
}

/// A likelihood-ratio row: the reference `reference` names (`chisq`,
/// `mixture` with `mixture_k`, or `envelope`, the exact test), the statistic
/// within `LRT_ABS`, and the p-value (`p_chisq` or `p_mixture`) within
/// `LR_P_REL`; inside the exact test's envelope the statistic only.
fn lr_row(name: &str, at: &str, test: &LrTest, row: &Value, reference: &str, improved: bool) {
    let ctx = format!("{name}{at}");
    let want_ref = match reference {
        "chisq" => Some(Reference::ChiSquared {
            df: row["df"].as_u64().or(row["Df"].as_u64()).expect("df") as usize,
        }),
        "mixture" => Some(Reference::ChiBarSquared {
            df_low: row["mixture_k"].as_u64().expect("mixture_k") as usize,
        }),
        "envelope" => None,
        other => panic!("{ctx}: no likelihood-ratio reference {other}"),
    };
    match &want_ref {
        Some(r) => assert_eq!(&test.reference, r, "{ctx}: the reference"),
        None => assert!(
            matches!(test.reference, Reference::ExactRlrt { .. }),
            "{ctx}: inside the envelope, the exact test: {:?}",
            test.reference
        ),
    }
    let stat = row
        .get("statistic")
        .or(row.get("LRT"))
        .expect("a frozen statistic");
    let want = number(stat, &ctx);
    let ctx_lrt = format!("{name}: lrt{at}");
    pair_quantity(&ctx_lrt, test.statistic, want, 1.0, tol::LRT_ABS, improved);
    let p = match reference {
        "chisq" => &row["p_chisq"],
        "mixture" => &row["p_mixture"],
        _ => return,
    };
    let want = number(p, &ctx);
    let ctx_p = format!("{name}: p_value{at}");
    pair_quantity(
        &ctx_p,
        test.p_value,
        want,
        want.abs(),
        tol::LR_P_REL,
        improved,
    );
}

/// An F row (`anova.lm`'s, or the Gamma pair's Wald F): the statistic
/// within `WALD_REL` (quantity `f`), the two df exactly, the p-value within
/// `WALD_P_REL`.
fn f_row(name: &str, test: &LinearTest, frozen: [f64; 4], improved: bool) {
    let [f, num_df, den_df, p] = frozen;
    let Reference::F {
        num_df: got_num,
        den_df: got_den,
        ..
    } = test.reference
    else {
        panic!("{name}: not an F reference: {:?}", test.reference)
    };
    assert_eq!((got_num, got_den), (num_df, den_df), "{name}: the F's df");
    let ctx_f = format!("{name}: f");
    pair_quantity(&ctx_f, test.statistic, f, f.abs(), tol::WALD_REL, improved);
    let ctx_p = format!("{name}: p_value");
    pair_quantity(&ctx_p, test.p_value, p, p.abs(), tol::WALD_P_REL, improved);
}

/// The two-fit comparisons, end to end on glmm's own fits: every
/// anova pair and every `ranova` row (its full formula against its
/// reduced one), each side fitted as its case needs, through [`compare`].
/// The deviance gate is read first; then the statistic against the frozen
/// one and the p-value against the one the entry's `reference` names: the
/// OLS pairs `anova.lm`'s F, the Gamma GLM pair the Wald F of
/// [`GAMMA_WALD_F`], the rows inside the exact test's envelope (`reference`
/// `envelope`) the statistic only, their p being gated against RLRsim
/// (`rlrt_dyestuff`, `rlrt_sleepstudy`). A `ranova` row that reduces to no random effects also
/// gates the reduced REML value, the OLS fit made with `reml = Some(true)`,
/// against its `rlrt_entry`'s frozen `logLik(lm, REML = TRUE)` within
/// `REML0_LOGLIK_ABS`, hard.
fn anova_pairs(compared: &mut BTreeSet<String>) {
    for e in inference::entries(inference::Kind::Anova) {
        let case = e.raw["case"].as_str().expect("an anova case");
        if !PAIR_CASES.contains(&case) {
            continue;
        }
        assert!(!e.is_pending(), "{}: not frozen", e.name);
        let name = e.name.as_str();
        let json = inference::load(inference::Kind::Anova, name);
        let members = e.raw["members"].as_array().expect("members");
        let reference = json["reference"].as_str().expect("a reference");
        if REFUSED_PAIRS.contains(&name) {
            let refused = members.iter().any(|m| {
                let formula = m["formula"].as_str().expect("a member formula");
                let lowered = oracle_support::lower_on_entry(
                    &e,
                    formula,
                    member_family(m),
                    m["weights"].as_str(),
                );
                matches!(lowered, Err(glmm::formula::Error::SlopeVarNotInDesign(_)))
            });
            assert!(
                refused,
                "{name}: the frontend now admits both models; gate the pair"
            );
            continue;
        }
        if case == "ranova" {
            let member = &members[0];
            let full_formula = json["full"]["formula"].as_str().expect("full formula");
            let full = fit_side(&e, member, full_formula, Some(true));
            let full_ll = number(&json["full"]["logLik"], name);
            for (i, row) in json["rows"]
                .as_array()
                .expect("ranova rows")
                .iter()
                .enumerate()
            {
                let at = format!("{name} row {i}");
                let reduced_formula = row["reduced_formula"].as_str().expect("reduced formula");
                let reduced = fit_side(&e, member, reduced_formula, Some(true));
                let improved = deviance_gate(
                    &at,
                    [
                        (&reduced.fit, number(&row["logLik"], &at)),
                        (&full.fit, full_ll),
                    ],
                );
                let c = compare(&reduced.model, &full.model, &ENVELOPE_OPTS)
                    .unwrap_or_else(|err| panic!("{at}: {err}"));
                let ComparisonTest::Lr(test) = &c.test else {
                    panic!("{at}: not a likelihood-ratio test")
                };
                let row_ref = if row["envelope"].as_bool() == Some(true) {
                    "envelope"
                } else {
                    reference
                };
                lr_row(name, &format!("[{i}]"), test, row, row_ref, improved);
                if reduced.lo.model.re.is_none() {
                    let rlrt = e.raw["rlrt_entry"]
                        .as_str()
                        .unwrap_or_else(|| panic!("{at}: a reduction to OLS names its rlrt_entry"));
                    let want = number(
                        &inference::load(inference::Kind::Rlrt, rlrt)["loglik_lm_reml"],
                        rlrt,
                    );
                    let d = (reduced.fit.loglik - want).abs();
                    eprintln!("ANOVA {at}: reduced REML loglik against {rlrt}: |d| {d:.3e}");
                    assert!(
                        reduced.fit.reml && d <= tol::REML0_LOGLIK_ABS,
                        "{at}: the reduced REML loglik {} against {rlrt}'s {want} (|d| {d:e})",
                        reduced.fit.loglik
                    );
                }
            }
        } else {
            let frozen = json["members"].as_array().expect("frozen members");
            // The Gamma case freezes glmmTMB's two fits first: its statistic's.
            let sides: Vec<Side> = members[..2].iter().map(|m| fit_member(&e, m)).collect();
            // The Wald F pair has no deviance gate: its reference is no
            // likelihood, and R's `logLik` of a weighted Gamma `glm` is not
            // glmm's likelihood (R multiplies each log-density by its weight
            // and takes φ as the mean deviance; glmm's weights are precision
            // weights, φ at its maximum).
            let mut logliks = [0, 1].map(|i| number(&frozen[i]["logLik"], name));
            let mut other_objective = false;
            for &(pair, side, golden) in LME4_NB_SIDES {
                if pair == name {
                    let g = oracle_support::load_golden(golden);
                    logliks[side] = g.estimates.loglik.expect("the golden's loglik");
                    other_objective = true;
                }
            }
            let improved = reference != "wald_f"
                && (deviance_gate(
                    name,
                    [(&sides[0].fit, logliks[0]), (&sides[1].fit, logliks[1])],
                ) || other_objective);
            let c = compare(&sides[0].model, &sides[1].model, &ENVELOPE_OPTS)
                .unwrap_or_else(|err| panic!("{name}: {err}"));
            match (reference, &c.test) {
                ("f", ComparisonTest::Linear(test)) => {
                    let a = &json["anova_lm"];
                    let res_df = a["res_df"].as_array().expect("res_df");
                    let frozen = [&a["F"], &a["df"], &res_df[1], &a["p"]];
                    f_row(name, test, frozen.map(|v| number(v, name)), improved);
                }
                ("wald_f", ComparisonTest::Linear(test)) => {
                    let w = inference::load(inference::Kind::Anova, GAMMA_WALD_F);
                    let frozen = [&w["F"], &w["num_df"], &w["den_df"], &w["p"]];
                    f_row(
                        name,
                        test,
                        frozen.map(|v| number(v, GAMMA_WALD_F)),
                        improved,
                    );
                }
                (_, ComparisonTest::Lr(test)) => lr_row(name, "", test, &json, reference, improved),
                (r, t) => panic!("{name}: reference {r} got {t:?}"),
            }
        }
        compared.insert(e.name.clone());
    }
}

/// The options of a `term_tests` call asking for `ss_type`.
fn options(ss_type: SsType) -> TermTestOptions {
    let mut opts = TermTestOptions::default();
    opts.ss_type = ss_type;
    opts
}

/// A per-term statistic, relative to car's within `band` (quantity `f`).
fn term_statistic(ctx: &str, got: f64, want: f64, band: f64) {
    eprintln!(
        "ANOVA {ctx}: glmm {got:.10e} oracle {want:.10e} rel {:.3e} (band {band:.0e})",
        (got - want).abs() / want.abs()
    );
    assert_rel_to(got, want, want.abs(), band, ctx);
}

/// The per-term rows, end to end on glmm's own fit through
/// [`term_tests`]: car's Type II table and the Type III table of its
/// `contr.sum` refit (the treatment-coded Type III and car's `(Intercept)`
/// rows are recorded only), and the one linear hypothesis of
/// [`GAMMA_WALD_F`] as its term's Type III row. Per row the statistic
/// (quantity `f`, an F or a Wald χ²) within `WALD_REL`, the GLMM rows within
/// `WALD_GLMM_REL` (the frozen Nelder_Mead fit's floor), the numerator df
/// exactly, an F row's residual df exactly, and the p-value (`p_value`)
/// within `WALD_P_REL`. The GLMM rows first
/// assert the default `wald_se` and no `Note::HessianSeFallback` on the fit:
/// car's frozen vcov is lme4's finite-difference Hessian, and the RX route
/// moves the χ² by 5 to 13%. The registry may excuse a miss here (no
/// deviance gate can: no `logLik` of these models is frozen).
fn anova_term_rows(compared: &mut BTreeSet<String>) {
    for e in inference::entries(inference::Kind::Anova) {
        let case = e.raw["case"].as_str().expect("an anova case");
        if !TERM_CASES.contains(&case) {
            continue;
        }
        assert!(!e.is_pending(), "{}: not frozen", e.name);
        let name = e.name.as_str();
        let json = inference::load(inference::Kind::Anova, name);
        let side = fit_member(&e, &e.raw["members"][0]);
        let glmm_wald = side.lo.model.re.is_some();
        if glmm_wald {
            assert_eq!(
                side.lo.opts.wald_se,
                WaldSe::Hessian,
                "{name}: the Wald SEs must be the Hessian ones (car's vcov is lme4's Hessian)"
            );
            assert!(
                !side
                    .fit
                    .diagnostics
                    .notes
                    .iter()
                    .any(|n| matches!(n, Note::HessianSeFallback)),
                "{name}: the SEs fell back from the Hessian"
            );
        }
        let tables: Vec<(&str, SsType)> = if case == "linear_hypothesis" {
            vec![("", SsType::Type3)]
        } else {
            vec![("type2", SsType::Type2), ("type3_sum", SsType::Type3)]
        };
        for (table, ss_type) in tables {
            let got = term_tests(&side.lo, &side.fit, &options(ss_type))
                .unwrap_or_else(|err| panic!("{name}: {err}"));
            let rows: Vec<Value> = if table.is_empty() {
                // The linear hypothesis as the row of the term owning its columns.
                let tested: Vec<&str> = json["row_names"]
                    .as_array()
                    .expect("row_names")
                    .iter()
                    .map(|s| s.as_str().expect("a column name"))
                    .collect();
                let term = side
                    .lo
                    .terms
                    .iter()
                    .find(|t| {
                        t.columns
                            .clone()
                            .map(|j| side.lo.col_names[j].as_str())
                            .eq(tested.iter().copied())
                    })
                    .unwrap_or_else(|| panic!("{name}: no term owns {tested:?}"));
                vec![serde_json::json!({
                    "term": term.label,
                    "statistic": json["F"],
                    "df": json["num_df"],
                    "res_df": json["den_df"],
                    "p": json["p"],
                })]
            } else {
                assert_eq!(
                    json[table]["gated"].as_bool(),
                    Some(true),
                    "{name}: {table} gated"
                );
                json[table]["rows"]
                    .as_array()
                    .expect("rows")
                    .iter()
                    .filter(|r| r["term"] != "(Intercept)")
                    .cloned()
                    .collect()
            };
            for row in &rows {
                let term = row["term"].as_str().expect("a term");
                let at = format!("{name}: [{table} {term}]");
                let r = got
                    .fixed
                    .iter()
                    .find(|r| r.label == term)
                    .unwrap_or_else(|| panic!("{at}: no row"));
                assert_eq!(r.test.num_df, number(&row["df"], &at), "{at}: num df");
                if let Some(res_df) = row["res_df"].as_f64() {
                    let Reference::F { den_df, .. } = r.test.reference else {
                        panic!("{at}: not an F row: {:?}", r.test.reference)
                    };
                    assert_eq!(den_df, res_df, "{at}: residual df");
                }
                let band = if glmm_wald {
                    tol::WALD_GLMM_REL
                } else {
                    tol::WALD_REL
                };
                let ctx = format!("{name}: f[{table} {term}]");
                term_statistic(&ctx, r.test.statistic, number(&row["statistic"], &at), band);
                let (p, want_p) = (r.test.p_value, number(&row["p"], &at));
                eprintln!(
                    "ANOVA {at}: p glmm {p:.10e} oracle {want_p:.10e} rel {:.3e} (band {:.0e})",
                    (p - want_p).abs() / want_p,
                    tol::WALD_P_REL
                );
                let ctx = format!("{name}: p_value[{table} {term}]");
                assert_rel_to(p, want_p, want_p, tol::WALD_P_REL, &ctx);
            }
            if !table.is_empty() {
                assert_eq!(got.fixed.len(), rows.len(), "{name} {table}: every term");
            }
        }
        compared.insert(e.name.clone());
    }
}

/// The per-term rows of the lmerTest tables through [`term_tests`]: Type III
/// and Type II on the REML and ML `sim_factorial` fits and on
/// `sim_factorial_empty`, F and DenDF at the end-to-end band `DF_E2E_REL`,
/// `Sum Sq` and `Mean Sq` at `SUM_SQ_REL` (F times σ̂²), NumDF exactly, the
/// p-value within `SATT_P_REL`. The in-crate driver of
/// `src/formula/hypotheses_oracle_tests.rs` registry-gates these names (one
/// driver per name), so no registry entry excuses a miss here, and the names
/// are recorded in no `compared`.
#[test]
fn lmertest_term_rows_match_anova() {
    const NAMES: [&str; 3] = [
        "lmertest_anova_sim_factorial",
        "lmertest_anova_sim_factorial_ml",
        "lmertest_anova_sim_factorial_empty",
    ];
    let entries = inference::entries(inference::Kind::Lmertest);
    for name in NAMES {
        let e = entries
            .iter()
            .find(|e| e.name == name)
            .unwrap_or_else(|| panic!("{name}: not in the manifest"));
        let json = inference::load(inference::Kind::Lmertest, name);
        let lo = inference_lowered(e);
        let fit = glmm::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        assert_eq!(Some(fit.reml), json["reml"].as_bool(), "{name}: criterion");
        for (table, ss_type) in [("type3", SsType::Type3), ("type2", SsType::Type2)] {
            let frozen = &json[table];
            let terms = frozen["terms"].as_array().expect("terms");
            let got = term_tests(&lo, &fit, &options(ss_type))
                .unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_eq!(got.fixed.len(), terms.len(), "{name} {table}: rows");
            for (k, r) in got.fixed.iter().enumerate() {
                let at = format!("{name} {table} {}", r.label);
                assert_eq!(r.label, terms[k].as_str().unwrap_or_default(), "{at}");
                let want = |field: &str| number(&frozen[field][k], &at);
                assert_eq!(r.test.num_df, want("NumDF"), "{at}: NumDF");
                let Reference::F { den_df, .. } = r.test.reference else {
                    panic!("{at}: not an F row")
                };
                let checks = [
                    ("F", r.test.statistic, want("F"), tol::DF_E2E_REL),
                    ("DenDF", den_df, want("DenDF"), tol::DF_E2E_REL),
                    (
                        "Sum Sq",
                        r.sum_sq.expect("an F row has Sum Sq"),
                        want("sum_sq"),
                        tol::SUM_SQ_REL,
                    ),
                    (
                        "Mean Sq",
                        r.mean_sq.expect("an F row has Mean Sq"),
                        want("mean_sq"),
                        tol::SUM_SQ_REL,
                    ),
                    ("p", r.test.p_value, want("p_value"), tol::SATT_P_REL),
                ];
                for (what, g, w, band) in checks {
                    let d = (g - w).abs() / w.abs();
                    eprintln!("LMERTEST-TERMS {at}: {what} glmm {g:.10e} oracle {w:.10e} rel {d:.3e} (band {band:.0e})");
                    assert!(
                        d <= band,
                        "{at}: {what} glmm={g} oracle={w} ({d:.3e} > {band:.0e})"
                    );
                }
            }
        }
    }
}

#[test]
fn inference_tier() {
    let mut compared = BTreeSet::new();
    for (_, gate) in GATES {
        gate(&mut compared);
    }
    assert_inference_divergences_all_fired(&compared);
}

/// A NaN comparison never passes. A NaN glmm value is not swallowed by the
/// near-zero exemption of the relative difference (the reference here is under
/// it), and the comparison then fails with no entry to consult.
#[test]
#[should_panic(expected = "glmm=NaN")]
fn nan_against_a_near_zero_reference_fails() {
    oracle_support::assert_rel(f64::NAN, 1e-4, 1e-6, "no_such_reference: p_value");
}

/// A NaN difference on a registered cell and quantity is not documented and
/// fires nothing, so the entry cannot look fresh on a broken comparison.
#[test]
fn nan_under_a_registered_entry_is_not_documented() {
    let reg = divergence::Registry::new(vec![divergence::Entry {
        id: "ref-x-df".to_string(),
        cell: "ref_x".to_string(),
        comparison: vec!["oracle-tier".to_string()],
        quantities: vec!["df".to_string()],
        max_rel: 0.1,
        direction: String::new(),
        summary: String::new(),
    }]);
    assert!(matches!(
        reg.covers("ref_x: df", f64::NAN),
        divergence::Coverage::NotDocumented
    ));
    assert!(reg.fired().is_empty());
    // The same entry does cover a finite difference within its bound.
    assert!(matches!(
        reg.covers("ref_x: df", 0.05),
        divergence::Coverage::Documented
    ));
}

/// The registry's teeth for this tier: an `oracle-tier` entry on a reference a
/// gate compared must have fired in this process, or it is stale and fails; and
/// every entry that fired must be on a compared reference, so a gate that
/// adjudicates a reference but forgets to record it in `compared` fails too.
///
/// Printed, never silent: each fired entry on a compared reference reaches the
/// operator on any failure in this test and under `--nocapture`.
fn assert_inference_divergences_all_fired(compared: &BTreeSet<String>) {
    let reg = divergence::registry();
    let fired = reg.fired();
    for e in reg.scoped() {
        if compared.contains(&e.cell) && fired.contains(&e.id) {
            eprintln!(
                "documented divergence: {} [{}] <= {:.1e}\n  {}\n  direction: {}",
                e.cell,
                e.quantities.join(","),
                e.max_rel,
                e.summary,
                e.direction
            );
        }
    }
    let unrecorded: Vec<&str> = reg
        .scoped()
        .filter(|e| fired.contains(&e.id) && !compared.contains(&e.cell))
        .map(|e| e.id.as_str())
        .collect();
    assert!(
        unrecorded.is_empty(),
        "these entries fired on a reference no gate recorded in `compared`: {unrecorded:?}"
    );
    let mine = |q: &String| QUANTITIES.contains(&q.as_str());
    let scoped: Vec<inference::DivEntry> = inference::registry_entries()
        .into_iter()
        .filter(|e| e.quantities.iter().any(mine))
        .collect();
    // One driver per reference and label: on a compared reference an entry
    // under these labels lists only them. A label may also be another
    // driver's on references this tier does not compare (the per-term rows'
    // `df` and `f`).
    for e in scoped.iter().filter(|e| compared.contains(&e.cell)) {
        assert!(
            e.quantities.iter().all(mine),
            "registry entry {} mixes this driver's quantities {QUANTITIES:?} with others: {:?}",
            e.id,
            e.quantities
        );
    }
    let stale = inference::stale_entries(&scoped, compared, &fired);
    assert!(
        stale.is_empty(),
        "inference divergence registry is out of date: these entries on a compared \
         reference no longer fire and must be deleted: {stale:?}"
    );
}
