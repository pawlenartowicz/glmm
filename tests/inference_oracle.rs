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
//! reference's name and whose `quantities` hold `df`, `p_value`, `lrt`, `f`,
//! `confint`, `r2`, `icc`, `eta2` or `posthoc` ([`QUANTITIES`]). The in-crate gate at lme4's θ̂ compares the same
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
//! χ², is the quantity `f`. [`anova_lrt_rows`] reads the likelihood-ratio
//! term tables (afex, car, and car's Type II as `glmer` refit pairs) stored by
//! [`run_fit`] under `anova_test = "LRT"`; their statistic is the quantity
//! `lrt`.
//!
//! The effect sizes (`kind` `effectsize`): [`effect_sizes`] reads R², the ICC
//! and the partial η² of each term row against performance's and
//! effectsize's.
//!
//! The pairwise comparisons (`kind` `posthoc`): [`posthoc_rows`] reads the
//! rows of [`pairwise`] (quantity `posthoc`) and their intervals (`confint`)
//! against emmeans'.
//!
//! The extended-precision GLM references on ill-conditioned designs (`kind`
//! `conditioning`, with `loop_advanced`) gate one fit:
//! `design_b_logit_halves_back_from_an_overshoot` asserts that design b's
//! unweighted logit fit at `τ_ref` 3.6e-11 converges at the reference β.
//! `conditioning_report` fits them all through the loop tier and prints the
//! accuracy per decade of the reference pivot ratio, asserting only how many
//! fits it ran.
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
    compare, term_tests, ChoiceReason, CompareOptions, ComparisonTest, SsType, TermTable, TermTest,
    TermTestError, TermTestOptions,
};
use glmm::formula::{fit_lowered, pairwise, Adjust, By, Column, Model, Pairwise};
use glmm::inference::{LinearKind, LinearTest, LrTest, Reference};
use glmm::orchestrate::run_fit;
use glmm::{
    BinomialLink, Family, Fit, FitOptions, GammaLink, NegBinomialLink, Note, PoissonLink, WaldSe,
};
use oracle_support::{assert_rel, assert_rel_to, divergence, inference, inference_lowered, tol};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

/// The registry quantities this driver adjudicates; an entry under any other
/// label belongs to another driver.
const QUANTITIES: &[&str] = &[
    "df", "p_value", "lrt", "f", "confint", "r2", "icc", "eta2", "posthoc",
];

/// One end-to-end gate: it adds the name of every reference it compared.
type Gate = fn(&mut BTreeSet<String>);

/// The end-to-end gates on the public surface, as (name, gate), run in order by
/// [`inference_tier`].
const GATES: &[(&str, Gate)] = &[
    ("lmertest_end_to_end", lmertest_end_to_end),
    ("lmertest_confint", lmertest_confint),
    ("anova_pairs", anova_pairs),
    ("anova_term_rows", anova_term_rows),
    ("anova_lrt_rows", anova_lrt_rows),
    ("effect_sizes", effect_sizes),
    ("posthoc_rows", posthoc_rows),
];

fn numbers(v: &Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what}: not an array: {v}"))
        .iter()
        .map(|x| x.as_f64().unwrap_or_else(|| panic!("{what}: {x}")))
        .collect()
}

/// A reference's name, JSON, `glmm::fit` fit and the caller column of each of
/// its coefficients.
type RefFit = (String, Value, Fit, Vec<usize>);

/// The per-coefficient lmerTest references of both criteria (X7: kind
/// `lmertest` with `proposal_dependent` false; `reml` true or false), each
/// fitted end to end by `glmm::fit` on its lowered formula (the ML ones with
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
            let (lo, fit_opts) = inference_lowered(&e);
            let fit = glmm::fit(&lo.data(), lo.spec(), &fit_opts).unwrap();
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

/// One fitted side of a comparison, or the one fit of a per-term table, with
/// the options it was fitted with.
struct Side {
    opts: FitOptions,
    model: Model,
}

/// The family and link of a manifest member, or of a `conditioning` entry.
fn member_family(m: &Value) -> Family {
    match (
        m["family"].as_str().expect("a member family"),
        m["link"].as_str().expect("a member link"),
    ) {
        ("gaussian", "identity") => Family::Gaussian,
        ("binomial", "logit") => Family::Binomial {
            link: BinomialLink::Logit,
        },
        ("binomial", "probit") => Family::Binomial {
            link: BinomialLink::Probit,
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
    let lo = oracle_support::lower_on_entry(e, formula, family, member["weights"].as_str())
        .unwrap_or_else(|err| panic!("{} ({formula}): {err}", e.name));
    let mut fit_opts = FitOptions::default();
    fit_opts.reml = reml;
    if let Some(k) = member["nAGQ"].as_u64() {
        fit_opts.nagq = u8::try_from(k).expect("nAGQ fits a u8");
    }
    let model = fit_lowered(&oracle_support::glmm_formula(formula), lo, &fit_opts).unwrap();
    Side {
        opts: fit_opts,
        model,
    }
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
/// (`rlrt_dyestuff`, `rlrt_sleepstudy`).
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
                        (reduced.model.fit(), number(&row["logLik"], &at)),
                        (full.model.fit(), full_ll),
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
                    [
                        (sides[0].model.fit(), logliks[0]),
                        (sides[1].model.fit(), logliks[1]),
                    ],
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

/// The per-term table of `ss_type`, its χ² rows pinned to the Wald test so
/// they stay Wald whatever the default test gives; a fit whose rows are F
/// tests refuses an explicit test, so its table takes the default, which must
/// be its F table (a χ² fit that refused the Wald test fails here).
fn term_table(model: &Model, ss_type: SsType) -> Result<TermTable, TermTestError> {
    let mut opts = TermTestOptions::default();
    opts.ss_type = ss_type;
    opts.test = TermTest::Wald;
    match term_tests(model, &opts) {
        Err(TermTestError::FTable { .. }) => {
            opts.test = TermTest::Auto;
            let table = term_tests(model, &opts);
            if let Ok(t) = &table {
                assert_eq!(
                    t.choice.test,
                    LinearKind::F,
                    "a χ² fit refused the Wald test"
                );
            }
            table
        }
        other => other,
    }
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
    let mut seen = BTreeSet::new();
    for e in inference::entries(inference::Kind::Anova) {
        let case = e.raw["case"].as_str().expect("an anova case");
        if !TERM_CASES.contains(&case) {
            continue;
        }
        assert!(!e.is_pending(), "{}: not frozen", e.name);
        let name = e.name.as_str();
        let json = inference::load(inference::Kind::Anova, name);
        let side = fit_member(&e, &e.raw["members"][0]);
        let (lo, fit) = (side.model.lowered(), side.model.fit());
        let glmm_wald = lo.model.re.is_some();
        if glmm_wald {
            assert_eq!(
                side.opts.wald_se,
                WaldSe::Hessian,
                "{name}: the Wald SEs must be the Hessian ones (car's vcov is lme4's Hessian)"
            );
            assert!(
                !fit.diagnostics
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
            let got =
                term_table(&side.model, ss_type).unwrap_or_else(|err| panic!("{name}: {err}"));
            let rows: Vec<Value> = if table.is_empty() {
                // The linear hypothesis as the row of the term owning its columns.
                let tested: Vec<&str> = json["row_names"]
                    .as_array()
                    .expect("row_names")
                    .iter()
                    .map(|s| s.as_str().expect("a column name"))
                    .collect();
                let term = lo
                    .terms
                    .iter()
                    .find(|t| {
                        t.columns
                            .clone()
                            .map(|j| lo.col_names[j].as_str())
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
                match (r.test.reference, row["res_df"].as_f64()) {
                    (Reference::F { den_df, .. }, Some(res_df)) => {
                        assert_eq!(den_df, res_df, "{at}: residual df")
                    }
                    (Reference::ChiSquared { .. }, None) => {}
                    (got, want) => panic!("{at}: reference {got:?} against frozen res_df {want:?}"),
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
        seen.insert(case.to_string());
    }
    let want: BTreeSet<String> = TERM_CASES.iter().map(|c| c.to_string()).collect();
    assert_eq!(
        seen, want,
        "anova_term_rows: a term case compared no reference"
    );
}

/// The likelihood-ratio term tables' case: afex `mixed(method = "LRT")` on
/// the GLMMs, car `Anova(test.statistic = "LR")` under `contr.sum` on the
/// GLMs, and car's Type II containment as explicit `glmer` refit pairs
/// (engine `lme4::glmer`).
const LRT_CASE: &str = "lrt_terms";

/// afex's `type = 2` table on `sim_factorial_bern`, which is sequential, not
/// car's Type II: its row `x` compares `g + h + x` with `g + h`, leaving out
/// `g:h`, which car's containment keeps. It is no reference for glmm's Type
/// II rows and is not gated. [`LRT_TYPE2_COMPANION`] is: on `g`, `h` and
/// `g:h` the two rules give the same pair of models, so the companion gates
/// those hypotheses too, and on `x` it gates car's.
const LRT_SEQUENTIAL: &str = "anova_lrt2_glmm_sim_factorial_bern";

/// The Type II likelihood-ratio reference of the GLMM on `sim_factorial_bern`:
/// car's containment as `glmer` refit pairs.
const LRT_TYPE2_COMPANION: &str = "anova_lrt2car_glmm_sim_factorial_bern";

/// Factor columns as [`run_fit`] takes them: `(levels, codes)` by name.
type Factors = HashMap<String, (Vec<String>, Vec<u32>)>;

/// `e`'s data as [`run_fit`] takes it: `oracle_support::entry_data`'s
/// columns split into the numeric and factor maps. R's response
/// `cbind(s, n - s)` becomes the ports' `cbind(s, failures)` on an added
/// column `n − s`, so the formula returned is the one to fit.
fn entry_columns(
    e: &inference::Entry,
    r_formula: &str,
) -> (String, HashMap<String, Vec<f64>>, Factors) {
    let (header, _, columns) = oracle_support::entry_data(e);
    let (mut numeric, mut factors) = (HashMap::new(), Factors::new());
    for (name, col) in columns {
        match col {
            Column::Numeric(v) => {
                numeric.insert(name, v);
            }
            Column::Factor { levels, codes } => {
                factors.insert(name, (levels, codes));
            }
        }
    }
    let Some((succ, total, rhs)) = oracle_support::aggregated_response(r_formula) else {
        return (r_formula.to_string(), numeric, factors);
    };
    assert!(
        r_formula.starts_with(&format!("cbind({succ}, {total} - {succ}) ~ ")),
        "{}: not cbind(s, n - s): {r_formula}",
        e.name
    );
    let failures = numeric[&total]
        .iter()
        .zip(&numeric[&succ])
        .map(|(n, s)| n - s)
        .collect();
    assert!(!header.iter().any(|h| h == "failures"), "{}", e.name);
    numeric.insert("failures".to_string(), failures);
    (format!("cbind({succ}, failures) ~ {rhs}"), numeric, factors)
}

/// The likelihood-ratio term rows, end to end through
/// [`run_fit`] with `anova_test = "LRT"`, the ports' route: each
/// [`LRT_CASE`] entry's one member fitted with Hessian SEs and its `nAGQ`,
/// and the stored likelihood-ratio table of the entry's `ss_type` read
/// against the frozen rows: the same terms in the same order, every row a
/// likelihood-ratio χ² with no note, referred to χ² on the frozen df exactly,
/// the statistic within `LRT_ABS` (quantity `lrt`) and the p-value within
/// `LT_LR_P_REL` (quantity `p_value`). A term whose columns carry a random
/// slope is gated the same way: its refit keeps the slope, and its row is a
/// likelihood-ratio row like any other. The registry may excuse a miss (no
/// deviance gate can: these entries freeze no `logLik`). [`LRT_SEQUENTIAL`]
/// is left out; its model's Type II rows are read from
/// [`LRT_TYPE2_COMPANION`].
fn anova_lrt_rows(compared: &mut BTreeSet<String>) {
    let mut sequential_seen = false;
    for e in inference::entries(inference::Kind::Anova) {
        if e.raw["case"].as_str() != Some(LRT_CASE) {
            continue;
        }
        let name = e.name.as_str();
        if name == LRT_SEQUENTIAL {
            sequential_seen = true;
            continue;
        }
        assert!(!e.is_pending(), "{name}: not frozen");
        let json = inference::load(inference::Kind::Anova, name);
        let members = e.raw["members"].as_array().map(Vec::len);
        assert_eq!(members, Some(1), "{name}: one member");
        let member = &e.raw["members"][0];
        assert!(member["weights"].is_null(), "{name}: weights");
        let text = |key: &str| {
            member[key]
                .as_str()
                .unwrap_or_else(|| panic!("{name}: {key}"))
        };
        let (formula, numeric, factors) = entry_columns(&e, text("formula"));
        let nagq = member["nAGQ"]
            .as_u64()
            .map_or(1, |k| u8::try_from(k).expect("nAGQ fits a u8"));
        let r = run_fit(
            &formula,
            numeric,
            factors,
            text("family"),
            text("link"),
            "hessian",
            nagq,
            None,
            None,
            None,
            None,
            true,
            true,
            None,
            "LRT",
        )
        .unwrap_or_else(|err| panic!("{name} ({formula}): {err}"));
        assert!(r.converged, "{name}: the fit did not converge");
        let stored = match e.raw["ss_type"].as_u64() {
            Some(2) => &r.term_table_type2_lrt,
            Some(3) => &r.term_table_type3_lrt,
            other => panic!("{name}: ss_type {other:?}"),
        };
        let table = stored
            .as_ref()
            .expect("asked for")
            .as_ref()
            .unwrap_or_else(|err| panic!("{name}: {err}"));
        assert_eq!(table.choice.test, LinearKind::LikelihoodRatio, "{name}");
        assert!(
            matches!(table.choice.reason, ChoiceReason::Requested),
            "{name}: {:?}",
            table.choice.reason
        );
        let rows = json["rows"].as_array().expect("frozen rows");
        let labels: Vec<&str> = table.fixed.iter().map(|r| r.label.as_str()).collect();
        let want: Vec<&str> = rows
            .iter()
            .map(|r| r["term"].as_str().expect("a term"))
            .collect();
        assert_eq!(labels, want, "{name}: the rows");
        for (got, row) in table.fixed.iter().zip(rows) {
            let term = &got.label;
            let at = format!("{name}: [{term}]");
            let df = number(&row["df"], &at);
            let t = &got.test;
            assert_eq!(t.kind, LinearKind::LikelihoodRatio, "{at}: {t:?}");
            assert!(t.notes.is_empty(), "{at}: {:?}", t.notes);
            assert_eq!(t.num_df, df, "{at}: df");
            assert_eq!(
                t.reference,
                Reference::ChiSquared { df: df as usize },
                "{at}: the reference"
            );
            let want = number(&row["statistic"], &at);
            let d = (t.statistic - want).abs();
            eprintln!(
                "ANOVA {at}: lrt glmm {:.10e} oracle {want:.10e} diff {d:.3e} (band {:.0e})",
                t.statistic,
                tol::LRT_ABS
            );
            assert_rel_to(
                t.statistic,
                want,
                1.0,
                tol::LRT_ABS,
                &format!("{name}: lrt[{term}]"),
            );
            let want_p = number(&row["p"], &at);
            eprintln!(
                "ANOVA {at}: p glmm {:.10e} oracle {want_p:.10e} rel {:.3e} (band {:.0e})",
                t.p_value,
                (t.p_value - want_p).abs() / want_p,
                tol::LT_LR_P_REL
            );
            assert_rel_to(
                t.p_value,
                want_p,
                want_p,
                tol::LT_LR_P_REL,
                &format!("{name}: p_value[{term}]"),
            );
        }
        compared.insert(e.name.clone());
    }
    assert!(
        sequential_seen,
        "{LRT_SEQUENTIAL}: no longer in the manifest; drop it here"
    );
    assert!(
        compared.contains(LRT_TYPE2_COMPANION),
        "{LRT_TYPE2_COMPANION}: not gated"
    );
}

// ── Effect sizes ────────────────────────────────────────────────────────────

/// The effect sizes (`kind` `effectsize`), end to end on glmm's fit of each
/// entry's one member: `performance::r2_nakagawa`'s marginal and conditional
/// R² (quantity `r2`) and `performance::icc`'s adjusted and unadjusted ICC
/// (`icc`) through [`Fit::r2`] and [`Fit::icc`], within `R2_ICC_REL`, on the
/// mixed models; and `effectsize::F_to_eta2` on each fixed-term row of the
/// reference's Type III table (`eta2`) against [`FixedRow::eta2_partial`] of
/// glmm's, within `ETA2_REL`. The frozen tables have no `(Intercept)` row;
/// on `lm` and `glm` they are car's on the `contr.sum` refit, on the LMMs
/// lmerTest's Satterthwaite tables, whose DenDF carries lmerTest's
/// numerical-derivative noise (222.0000014 on `effectsize_cake`'s `temp`,
/// an integer df), which moves η² far less than the band.
///
/// [`FixedRow::eta2_partial`]: glmm::formula::anova::FixedRow::eta2_partial
fn effect_sizes(compared: &mut BTreeSet<String>) {
    let mut worst: BTreeMap<&str, f64> = BTreeMap::new();
    let mut gap = |ctx: &str, quantity: &'static str, got: f64, want: f64, band: f64| {
        let r = ((got - want) / want).abs();
        eprintln!("EFFECT {ctx}: glmm {got:.10e} oracle {want:.10e} rel {r:.3e} (band {band:.0e})");
        let w = worst.entry(quantity).or_insert(0.0);
        *w = w.max(r);
        assert_rel(got, want, band, ctx);
    };
    let (mut with_r2_icc, mut with_eta2) = (0, 0);
    let mut names = Vec::new();
    for e in inference::entries(inference::Kind::Effectsize) {
        assert!(!e.is_pending(), "{}: not frozen", e.name);
        let name = e.name.as_str();
        let json = inference::load(inference::Kind::Effectsize, name);
        let side = fit_member(&e, &e.raw["members"][0]);
        let (lo, fit) = (side.model.lowered(), side.model.fit());
        if !json["r2"].is_null() || !json["icc"].is_null() {
            let (data, spec) = (lo.data(), lo.spec());
            let r2 = fit
                .r2(&data, spec)
                .unwrap_or_else(|err| panic!("{name}: r2: {err}"));
            let icc = fit
                .icc(&data, spec)
                .unwrap_or_else(|err| panic!("{name}: icc: {err}"));
            for (q, field, got) in [
                ("r2", "marginal", r2.marginal),
                ("r2", "conditional", r2.conditional),
                ("icc", "adjusted", icc.adjusted),
                ("icc", "unadjusted", icc.unadjusted),
            ] {
                let want = number(&json[q][field], &format!("{name}: {q}.{field}"));
                gap(
                    &format!("{name}: {q}[{field}]"),
                    q,
                    got,
                    want,
                    tol::R2_ICC_REL,
                );
            }
            with_r2_icc += 1;
        }
        if let Some(rows) = json["eta2"].as_array() {
            let table = term_table(&side.model, SsType::Type3)
                .unwrap_or_else(|err| panic!("{name}: {err}"));
            assert_eq!(table.fixed.len(), rows.len(), "{name}: every term");
            for row in rows {
                let term = row["term"].as_str().expect("a term");
                let r = table
                    .fixed
                    .iter()
                    .find(|r| r.label == term)
                    .unwrap_or_else(|| panic!("{name}: no row {term}"));
                let want = number(&row["eta2_partial"], &format!("{name}: {term}"));
                let ctx = format!("{name}: eta2[{term}]");
                gap(&ctx, "eta2", r.eta2_partial(), want, tol::ETA2_REL);
            }
            with_eta2 += 1;
        }
        compared.insert(e.name.clone());
        names.push(e.name);
    }
    eprintln!("EFFECT compared {}: {names:?}", names.len());
    eprintln!("EFFECT worst rel per quantity: {worst:?}");
    assert_eq!(
        (with_r2_icc, with_eta2),
        (7, 6),
        "effect_sizes: entries with R² and ICC, entries with η² rows"
    );
}

// ── Pairwise comparisons ────────────────────────────────────────────────────

/// The `posthoc` entries whose Tukey p-values are not compared against
/// emmeans': their df (about 5 on `posthoc_machines`, about 10 on
/// `posthoc_oats`) lie outside the df region where R's `ptukey` is known to
/// agree with the studentized range glmm uses ([`ptukey_df_agrees`]). The
/// crate's own Tukey tests cover those df (the k = 2 identity with the t
/// test, and the adjusted p as the studentized-range tail of each row).
const TUKEY_UNCOMPARED: &[&str] = &["posthoc_machines", "posthoc_oats"];

/// The smallest tail, upper or lower, at which a frozen Tukey p-value is
/// compared. R's `ptukey(lower.tail = FALSE)` is `1 − ptukey(lower)`: it
/// integrates the lower tail and subtracts, so an upper tail far below 1
/// carries the absolute rounding error of the lower one and is cancellation
/// noise (5.97e-10 frozen where a quadrature of the upper tail itself, with
/// no subtraction, gives 1.13e-10, at k = 3 and df 94). `commonstats`'
/// studentized-range doc comment gives its agreement with `ptukey` only
/// where both tails are at least this; a row outside is not compared.
const PTUKEY_TAIL_MIN: f64 = 1e-3;

/// The confidence level of the frozen `confint` rows.
const POSTHOC_LEVEL: f64 = 0.95;

/// The df on which R's `ptukey(q, k, df)` agrees with `commonstats`'
/// studentized range to 1e-5 relative, as that type's doc comment measured
/// (with both tails at least [`PTUKEY_TAIL_MIN`]): df infinite, or
/// 30 ≤ df ≤ 25000.
fn ptukey_df_agrees(df: f64) -> bool {
    df == f64::INFINITY || (30.0..=25000.0).contains(&df)
}

/// Holm's step-down adjustment of `p`: R's `p.adjust(p, "holm")`,
/// `min(1, cummax((m − i + 1)·p₍ᵢ₎))` in the order of `p`.
fn holm(p: &[f64]) -> Vec<f64> {
    let m = p.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p[a].total_cmp(&p[b]));
    let mut out = vec![0.0; m];
    let mut run = 0.0_f64;
    for (rank, &i) in order.iter().enumerate() {
        run = run.max(((m - rank) as f64 * p[i]).min(1.0));
        out[i] = run;
    }
    out
}

/// The levels of the factor `factor` of `e`'s data, in their sorted order,
/// which is glmm's level order.
fn entry_levels(e: &inference::Entry, factor: &str) -> Vec<String> {
    let (_, _, columns) = oracle_support::entry_data(e);
    match columns.into_iter().find(|(n, _)| n == factor) {
        Some((_, Column::Factor { levels, .. })) => levels,
        _ => panic!("{}: {factor} is not a factor column", e.name),
    }
}

/// The one frozen row of `rows` with emmeans' `label`, `by` level and, on an
/// interval row, `adjust`.
fn frozen_row<'a>(
    rows: &'a [Value],
    label: &str,
    by: Option<&str>,
    adjust: Option<&str>,
    ctx: &str,
) -> &'a Value {
    let hits: Vec<&Value> = rows
        .iter()
        .filter(|r| {
            r["contrast"].as_str() == Some(label)
                && r["by"].as_str() == by
                && adjust.is_none_or(|a| r["adjust"].as_str() == Some(a))
        })
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "{ctx}: frozen rows {label} by {by:?} adjust {adjust:?}"
    );
    hits[0]
}

/// The pairwise comparisons (`kind` `posthoc`), end to end on glmm's fit of
/// each entry's one member through [`pairwise`] (quantity `posthoc`),
/// against emmeans' `pairs(emmeans(..))` on lme4's, lm's or glm's fit with
/// `lmer.df = "satterthwaite"`. Per row of the `Adjust::None` result: the
/// estimate within `BETA_REL` of its SE, the SE within `SE_REL`
/// (`WALD_GLMM_REL` on a GLMM), a t row's df within `DF_E2E_REL` and a z
/// row's df infinite (emmeans freezes no df there), the statistic `est/SE`
/// within `BETA_REL + SE band` of `max(1, |t|)` (its error is at most
/// `|Δest|/SE + |t|·|ΔSE|/SE`), and the unadjusted p within `SATT_P_REL` (t)
/// or `WALD_P_REL` (z). Holm is applied to the frozen unadjusted p within
/// each `by` family here and compared with the `Adjust::Holm` result in the
/// same p band. The `Adjust::Tukey` p-values against emmeans' within
/// `TUKEY_P_REL`, except on [`TUKEY_UNCOMPARED`] and on the rows whose
/// frozen p or its complement is below [`PTUKEY_TAIL_MIN`]. On a log or logit link the
/// `response` rows: the ratio within `BETA_REL` of the ratio's SE, and the
/// ratio's SE (`ratio·SE`) within the SE band plus `BETA_REL` times the
/// link-scale SE (the estimate's error enters through `exp`).
///
/// On the entries with `confint_adjust` (quantity `confint`): the intervals
/// [`Pairwise::confint`] at 0.95 of each listed adjustment's result, and of
/// the Holm result against the frozen Bonferroni rows (Holm has no
/// simultaneous interval), each bound within `CONFINT_REL` of emmeans'
/// interval width, on the link scale and, on a logit link, the odds ratio's.
///
/// emmeans labels a comparison of levels that all read as numbers with the
/// factor's name in front ("period1 - period2"), and a response-scale one
/// with " / ", so glmm's `"a - b"` is matched under those rules.
///
/// [`Pairwise::confint`]: glmm::formula::Pairwise::confint
fn posthoc_rows(compared: &mut BTreeSet<String>) {
    let mut worst: BTreeMap<&str, f64> = BTreeMap::new();
    let mut gap = |ctx: &str, what: &'static str, got: f64, want: f64, scale: f64, band: f64| {
        let d = (got - want).abs() / scale;
        eprintln!(
            "POSTHOC {ctx}: glmm {got:.10e} oracle {want:.10e} diff {d:.3e} (band {band:.1e})"
        );
        let w = worst.entry(what).or_insert(0.0);
        *w = w.max(d);
        assert_rel_to(got, want, scale, band, ctx);
    };
    // Rows compared: contrasts, Holm, Tukey, response, link intervals,
    // response intervals; then the Tukey rows left out for their tail.
    let mut counts = [0usize; 7];
    let mut names = Vec::new();
    for e in inference::entries(inference::Kind::Posthoc) {
        assert!(!e.is_pending(), "{}: not frozen", e.name);
        let name = e.name.as_str();
        let json = inference::load(inference::Kind::Posthoc, name);
        let side = fit_member(&e, &e.raw["members"][0]);
        let model = &side.model;
        assert!(model.fit().converged(), "{name}: the fit did not converge");
        let glmm_se = model.lowered().model.re.is_some()
            && !matches!(model.lowered().spec().family, Family::Gaussian);
        let se_band = if glmm_se {
            tol::WALD_GLMM_REL
        } else {
            tol::SE_REL
        };
        let factor = e.raw["factor"].as_str().expect("a factor");
        let by = e.raw["by"].as_str();
        let run = |adjust| {
            pairwise(model, factor, by.map(By::Levels), adjust)
                .unwrap_or_else(|err| panic!("{name} {adjust:?}: {err}"))
        };
        let [none, holm_pw, bonferroni, tukey] = [
            Adjust::None,
            Adjust::Holm,
            Adjust::Bonferroni,
            Adjust::Tukey,
        ]
        .map(run);
        let levels = entry_levels(&e, factor);
        let numeric = levels.iter().all(|l| l.parse::<f64>().is_ok());
        let tag = |l: &str| {
            if numeric {
                format!("{factor}{l}")
            } else {
                l.to_string()
            }
        };
        let emmeans_label = |contrast: &str, sep: &str| -> String {
            let pair = levels
                .iter()
                .enumerate()
                .flat_map(|(i, a)| levels[i + 1..].iter().map(move |b| (a, b)))
                .find(|(a, b)| format!("{a} - {b}") == contrast)
                .unwrap_or_else(|| panic!("{name}: no pair of levels gives {contrast}"));
            format!("{}{sep}{}", tag(pair.0), tag(pair.1))
        };
        let frozen = json["contrasts"].as_array().expect("contrasts");
        assert_eq!(none.rows.len(), frozen.len(), "{name}: every row");
        for pw in [&holm_pw, &bonferroni, &tukey] {
            let same = pw
                .rows
                .iter()
                .zip(&none.rows)
                .all(|(a, b)| a.contrast == b.contrast && a.by_level == b.by_level);
            assert!(
                same && pw.rows.len() == none.rows.len(),
                "{name}: the row order of {:?}",
                pw.adjust
            );
        }
        let tukey_compared = !TUKEY_UNCOMPARED.contains(&name);
        let mut families: BTreeMap<Option<&str>, Vec<usize>> = BTreeMap::new();
        let mut p_frozen = Vec::with_capacity(none.rows.len());
        for (i, row) in none.rows.iter().enumerate() {
            let by_level = row.by_level.as_deref();
            let label = emmeans_label(&row.contrast, " - ");
            let at = format!("{label} by {by_level:?}");
            let want = frozen_row(frozen, &label, by_level, None, name);
            let q = |field: &str| number(&want[field], &format!("{name} {at}: {field}"));
            let ctx = |what: &str| format!("{name}: posthoc[{what} {at}]");
            let want_se = q("SE");
            gap(
                &ctx("estimate"),
                "estimate",
                row.estimate,
                q("estimate"),
                want_se,
                tol::BETA_REL,
            );
            gap(&ctx("SE"), "SE", row.se, want_se, want_se, se_band);
            let z = match want["statistic_name"].as_str() {
                Some("z") => true,
                Some("t") => false,
                other => panic!("{name} {at}: statistic_name {other:?}"),
            };
            if z {
                assert!(want["df"].is_null(), "{name} {at}: a z row with a df");
                assert_eq!(row.df, f64::INFINITY, "{name} {at}: a z row's df");
            } else {
                let want_df = q("df");
                gap(&ctx("df"), "df", row.df, want_df, want_df, tol::DF_E2E_REL);
            }
            let want_t = q("statistic");
            gap(
                &ctx("statistic"),
                "statistic",
                row.statistic,
                want_t,
                want_t.abs().max(1.0),
                tol::BETA_REL + se_band,
            );
            let p_band = if z { tol::WALD_P_REL } else { tol::SATT_P_REL };
            let want_p = q("p_none");
            gap(&ctx("p"), "p", row.p_value, want_p, want_p, p_band);
            families.entry(by_level).or_default().push(i);
            p_frozen.push((want_p, p_band));
            counts[0] += 1;
            if tukey_compared {
                assert!(
                    ptukey_df_agrees(row.df),
                    "{name} {at}: df {} outside ptukey's agreement region",
                    row.df
                );
                let want_p = q("p_tukey");
                let got = tukey.rows[i].p_adjusted;
                if want_p.min(1.0 - want_p) < PTUKEY_TAIL_MIN {
                    eprintln!(
                        "POSTHOC {name}: p_tukey {at} not compared: frozen {want_p:.4e} \
                         (glmm {got:.4e}), a tail below {PTUKEY_TAIL_MIN:.0e}"
                    );
                    counts[6] += 1;
                    continue;
                }
                gap(
                    &ctx("p_tukey"),
                    "p_tukey",
                    got,
                    want_p,
                    want_p,
                    tol::TUKEY_P_REL,
                );
                counts[2] += 1;
            }
        }
        if !tukey_compared {
            assert!(
                none.rows.iter().any(|r| !ptukey_df_agrees(r.df)),
                "{name}: every df is inside ptukey's agreement region; compare its Tukey rows"
            );
        }
        for members in families.values() {
            let p: Vec<f64> = members.iter().map(|&i| p_frozen[i].0).collect();
            for (&i, want) in members.iter().zip(holm(&p)) {
                let row = &holm_pw.rows[i];
                let at = format!("{} by {:?}", row.contrast, row.by_level);
                let ctx = format!("{name}: posthoc[p_holm {at}]");
                gap(&ctx, "p_holm", row.p_adjusted, want, want, p_frozen[i].1);
                counts[1] += 1;
            }
        }
        if let Some(response) = json["response"].as_array() {
            assert!(
                none.ratio.is_some(),
                "{name}: no ratio on a log or logit link"
            );
            assert_eq!(
                none.rows.len(),
                response.len(),
                "{name}: every response row"
            );
            for row in &none.rows {
                let by_level = row.by_level.as_deref();
                let label = emmeans_label(&row.contrast, " / ");
                let at = format!("{label} by {by_level:?}");
                let want = frozen_row(response, &label, by_level, None, name);
                let link = frozen_row(
                    frozen,
                    &emmeans_label(&row.contrast, " - "),
                    by_level,
                    None,
                    name,
                );
                let q =
                    |v: &Value, field: &str| number(&v[field], &format!("{name} {at}: {field}"));
                let want_ratio_se = q(want, "SE");
                gap(
                    &format!("{name}: posthoc[ratio {at}]"),
                    "ratio",
                    row.ratio,
                    q(want, "ratio"),
                    want_ratio_se,
                    tol::BETA_REL,
                );
                gap(
                    &format!("{name}: posthoc[ratio_se {at}]"),
                    "ratio_se",
                    row.ratio_se,
                    want_ratio_se,
                    want_ratio_se,
                    se_band + tol::BETA_REL * q(link, "SE"),
                );
                counts[3] += 1;
            }
        } else {
            assert!(
                none.ratio.is_none(),
                "{name}: a ratio with no frozen response rows"
            );
        }
        if let Some(adjusts) = e.raw["confint_adjust"].as_array() {
            let link = json["confint"].as_array().expect("confint rows");
            let response = json["confint_response"].as_array();
            let mut runs: Vec<(&str, &Pairwise, Adjust)> = adjusts
                .iter()
                .map(|a| match a.as_str() {
                    Some("none") => ("none", &none, Adjust::None),
                    Some("bonferroni") => ("bonferroni", &bonferroni, Adjust::Bonferroni),
                    Some("tukey") => ("tukey", &tukey, Adjust::Tukey),
                    other => panic!("{name}: confint_adjust {other:?}"),
                })
                .collect();
            runs.push(("bonferroni", &holm_pw, Adjust::Bonferroni));
            for (frozen_adjust, pw, want_adjust) in runs {
                let ci = pw
                    .confint(POSTHOC_LEVEL)
                    .unwrap_or_else(|err| panic!("{name} {:?}: {err}", pw.adjust));
                assert_eq!(
                    ci.adjust, want_adjust,
                    "{name} {:?}: the intervals' adjustment",
                    pw.adjust
                );
                for (row, iv) in pw.rows.iter().zip(&ci.rows) {
                    let by_level = row.by_level.as_deref();
                    let mut scales = vec![(" - ", link, [iv.lower, iv.upper], "link")];
                    if pw.ratio.is_some() {
                        let r = response.unwrap_or_else(|| panic!("{name}: confint_response"));
                        scales.push((" / ", r, [iv.ratio_lower, iv.ratio_upper], "ratio"));
                    }
                    for (sep, rows, got, scale) in scales {
                        let label = emmeans_label(&row.contrast, sep);
                        let at = format!("{:?} {scale} {label} by {by_level:?}", pw.adjust);
                        let want = frozen_row(rows, &label, by_level, Some(frozen_adjust), name);
                        let bound = |f: &str| number(&want[f], &format!("{name} {at}: {f}"));
                        let (lower, upper) = (bound("lower"), bound("upper"));
                        let width = upper - lower;
                        for (side, g, w) in [("lower", got[0], lower), ("upper", got[1], upper)] {
                            gap(
                                &format!("{name}: confint[{side} {at}]"),
                                "confint",
                                g,
                                w,
                                width,
                                tol::CONFINT_REL,
                            );
                        }
                        counts[if scale == "link" { 4 } else { 5 }] += 1;
                    }
                }
            }
        }
        compared.insert(e.name.clone());
        names.push(e.name);
    }
    eprintln!("POSTHOC compared {}: {names:?}", names.len());
    eprintln!(
        "POSTHOC rows (contrasts, holm, tukey, response, intervals, ratio intervals compared; \
         tukey rows left out for their tail): {counts:?}"
    );
    eprintln!("POSTHOC worst per quantity (|diff| / scale): {worst:?}");
    assert_eq!(names.len(), 16, "posthoc_rows: entries compared");
    assert_eq!(
        counts,
        [66, 66, 47, 33, 48, 24, 13],
        "posthoc_rows: rows compared (contrasts, holm, tukey, response, intervals, ratio \
         intervals) and Tukey rows left out for their tail"
    );
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
        let (lo, fit_opts) = inference_lowered(e);
        let formula = e.r_formula.as_deref().expect("the entry has a formula");
        let model = fit_lowered(formula, lo, &fit_opts).unwrap();
        let fit = model.fit();
        assert_eq!(Some(fit.reml), json["reml"].as_bool(), "{name}: criterion");
        for (table, ss_type) in [("type3", SsType::Type3), ("type2", SsType::Type2)] {
            let frozen = &json[table];
            let terms = frozen["terms"].as_array().expect("terms");
            let got = term_table(&model, ss_type).unwrap_or_else(|err| panic!("{name}: {err}"));
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

// ── The GLM conditioning band ───────────────────────────────────────────────

/// The GLM conditioning band report and its one gate. Every fit goes through
/// the loop tier with the default standardization: `glmm::fit` runs the alias
/// gate, which drops design a's last column once its raw pivot passes
/// `ALIAS_EPS`, and the loop tier skips that gate.
#[cfg(feature = "loop_advanced")]
mod conditioning {
    use super::{inference, member_family, number, numbers};
    use glmm::formula::Column;
    use glmm::loop_advanced::{build_workspace, fit_on, FitDiagnostics, FitWorkspace};
    use glmm::{
        BinomialLink, Data, Family, FitOptions, GammaLink, GroupIds, ModelSpec, PoissonLink,
        ReStructure,
    };
    use serde_json::{json, Value};
    use std::collections::BTreeMap;

    /// The `conditioning` entries per design and family (each family once
    /// unweighted and once on the prior weights `w`), and their fits in all.
    /// Written out so an emptied or shrunken kind fails instead of reporting
    /// on nothing.
    const ENTRIES: [(&str, &str, usize); 10] = [
        ("a", "logit", 2),
        ("a", "probit", 2),
        ("a", "poisson", 2),
        ("a", "gamma", 2),
        ("b", "logit", 2),
        ("b", "probit", 2),
        ("b", "poisson", 2),
        ("c", "logit", 2),
        ("c", "probit", 2),
        ("c", "poisson", 2),
    ];
    const FITS: usize = 288;

    /// Design d, `y ~ 1 + a + x2` with `x2` an exact copy of `a`: no
    /// reference, one CSV `glm_band_d_<family>.csv` per family.
    const DUPLICATE_FAMILIES: [&str; 4] = ["logit", "probit", "poisson", "gamma"];

    /// Rows of the random-intercept grouping of the cross-route refit:
    /// `g = row mod GROUPS`.
    const GROUPS: usize = 10;

    /// The short family name the report prints: the link on the binomial,
    /// the family otherwise.
    fn short_name(raw: &Value) -> String {
        let family = raw["family"].as_str().expect("a family");
        match family {
            "binomial" => raw["link"].as_str().expect("a link").to_string(),
            f => f.to_string(),
        }
    }

    fn duplicate_family(name: &str) -> Family {
        match name {
            "logit" => Family::Binomial {
                link: BinomialLink::Logit,
            },
            "probit" => Family::Binomial {
                link: BinomialLink::Probit,
            },
            "poisson" => Family::Poisson {
                link: PoissonLink::Log,
            },
            "gamma" => Family::Gamma {
                link: GammaLink::Log,
            },
            other => panic!("no design d family {other}"),
        }
    }

    /// An entry's CSV columns by name; every column of these files is numeric.
    fn numeric_columns(e: &inference::Entry) -> BTreeMap<String, Vec<f64>> {
        let (_, _, columns) = super::oracle_support::entry_data(e);
        columns
            .into_iter()
            .map(|(name, col)| match col {
                Column::Numeric(v) => (name, v),
                Column::Factor { .. } => panic!("{}: column {name} is not numeric", e.name),
            })
            .collect()
    }

    fn column<'c>(cols: &'c BTreeMap<String, Vec<f64>>, name: &str) -> &'c [f64] {
        cols.get(name).unwrap_or_else(|| panic!("no column {name}"))
    }

    /// One fit's design `[1, c1, c2]`, row-major, and its offset, read off
    /// `r_formula`: `y ~ 1 + c1 + c2`, optionally `+ offset(o)`, the one shape
    /// the generator reads.
    fn band_design<'c>(
        r_formula: &str,
        cols: &'c BTreeMap<String, Vec<f64>>,
    ) -> (Vec<f64>, Option<&'c [f64]>) {
        let rhs = r_formula
            .strip_prefix("y ~ 1 + ")
            .unwrap_or_else(|| panic!("{r_formula}: not y ~ 1 + c1 + c2"));
        let (mut terms, mut offset) = (Vec::new(), None);
        for t in rhs.split(" + ") {
            match t.strip_prefix("offset(").and_then(|o| o.strip_suffix(')')) {
                Some(o) => offset = Some(column(cols, o)),
                None => terms.push(column(cols, t)),
            }
        }
        assert_eq!(
            terms.len(),
            2,
            "{r_formula}: two columns beside the intercept"
        );
        let x = (0..terms[0].len())
            .flat_map(|i| [1.0, terms[0][i], terms[1][i]])
            .collect();
        (x, offset)
    }

    /// What one `fit_on` call returned, copied out of the view.
    struct BandFit {
        diag: FitDiagnostics,
        beta: Vec<f64>,
        var_diag: Vec<f64>,
        t_sq: Vec<f64>,
    }

    fn fit_band(ws: &mut FitWorkspace, data: &Data<'_>, opts: &FitOptions) -> BandFit {
        let v = fit_on(ws, data, None, opts);
        BandFit {
            diag: v.diagnostics(),
            beta: v.betas().to_vec(),
            var_diag: v.var_diag().to_vec(),
            t_sq: v.t_sq().to_vec(),
        }
    }

    /// The region a fit lands in.
    fn region(d: &FitDiagnostics) -> &'static str {
        if d.separated {
            "separated"
        } else if !d.converged {
            "not settled"
        } else if d.se_unavailable {
            "SE unavailable"
        } else if d.ill_conditioned {
            "flag"
        } else {
            "accurate"
        }
    }

    /// The largest of `v`, NaN when any is NaN.
    fn worst(v: impl Iterator<Item = f64>) -> f64 {
        v.fold(
            0.0,
            |m: f64, x| if x.is_nan() { f64::NAN } else { m.max(x) },
        )
    }

    /// A converged fit against its reference, over the coefficients: the
    /// largest β̂ relative error, SE relative error and `|z − z_ref|`, and
    /// whether some `|β̂ − β_ref|` exceeds the SE. SE is `√var_diag` against
    /// `√diag(cov_ref)`, z is `β̂/SE` against `β_ref/√diag(cov_ref)`. No
    /// dispersion enters: the GLM view's `var_diag` is the unscaled
    /// `(X'WX)⁻¹`, and on the Gamma log link the Fisher weight is the prior
    /// weight alone. NaN, and no understatement, on a fit that did not
    /// converge.
    struct Errors {
        beta: f64,
        se: f64,
        dz: f64,
        understates: bool,
    }

    fn errors(f: &BandFit, beta_ref: &[f64], cov_ref: &[Vec<f64>]) -> Errors {
        if !f.diag.converged {
            return Errors {
                beta: f64::NAN,
                se: f64::NAN,
                dz: f64::NAN,
                understates: false,
            };
        }
        let se_ref: Vec<f64> = (0..beta_ref.len()).map(|j| cov_ref[j][j].sqrt()).collect();
        let se: Vec<f64> = f.var_diag.iter().map(|v| v.sqrt()).collect();
        let j = 0..beta_ref.len();
        Errors {
            beta: worst(
                j.clone()
                    .map(|j| ((f.beta[j] - beta_ref[j]) / beta_ref[j]).abs()),
            ),
            se: worst(j.clone().map(|j| ((se[j] - se_ref[j]) / se_ref[j]).abs())),
            dz: worst(
                j.clone()
                    .map(|j| (f.beta[j] / se[j] - beta_ref[j] / se_ref[j]).abs()),
            ),
            understates: j
                .into_iter()
                .any(|j| (f.beta[j] - beta_ref[j]).abs() > se[j]),
        }
    }

    /// Counts and worst values over the fits of one decade of `τ_ref` (or of
    /// design c, which has no reference). The maxima skip NaN: they read the
    /// converged fits.
    struct Tally {
        fits: usize,
        converged: usize,
        not_settled: usize,
        separated: usize,
        se_unavailable: usize,
        beta: f64,
        se: f64,
        dz: f64,
        understates: usize,
        tau_gap: f64,
    }

    impl Tally {
        fn new() -> Tally {
            Tally {
                fits: 0,
                converged: 0,
                not_settled: 0,
                separated: 0,
                se_unavailable: 0,
                beta: f64::NAN,
                se: f64::NAN,
                dz: f64::NAN,
                understates: 0,
                tau_gap: f64::NAN,
            }
        }

        fn add(&mut self, d: &FitDiagnostics, err: Option<&Errors>, tau_gap: f64) {
            self.fits += 1;
            self.converged += d.converged as usize;
            self.not_settled += (!d.converged && !d.separated) as usize;
            self.separated += d.separated as usize;
            self.se_unavailable += d.se_unavailable as usize;
            if let Some(e) = err {
                self.beta = self.beta.max(e.beta);
                self.se = self.se.max(e.se);
                self.dz = self.dz.max(e.dz);
                self.understates += e.understates as usize;
            }
            self.tau_gap = self.tau_gap.max(tau_gap);
        }

        fn counts(&self) -> String {
            format!(
                "fits {:>3} converged {:>3} not settled {:>3} separated {:>3} se_unavailable {:>3}",
                self.fits, self.converged, self.not_settled, self.separated, self.se_unavailable
            )
        }
    }

    /// The glmm-side measurement of the GLM conditioning band, printed, per
    /// fit and per decade of `τ_ref`: each `conditioning` reference fitted
    /// through `fit_on` (`se_for` all), design c with its verdict only;
    /// designs a and b refitted with a random intercept on `g = row mod 10`,
    /// the regions of the two routes side by side; design d, an exact
    /// duplicate column, from its CSVs. τ is `diagnostics().pivot`, on the
    /// standardized design at the converged weights; `τ_ref` is on the design
    /// as written. Asserts only how many fits ran, never their accuracy.
    ///
    /// ```sh
    /// cargo test --features oracle-tests,loop_advanced --test inference_oracle conditioning_report -- --nocapture
    /// ```
    #[test]
    fn conditioning_report() {
        let opts = FitOptions::default();
        let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
        let mut n_fits = 0;
        let mut decades: BTreeMap<(String, i32), Tally> = BTreeMap::new();
        let mut verdicts_c = Tally::new();
        let mut routes = Vec::new();
        for e in inference::entries(inference::Kind::Conditioning) {
            assert!(!e.is_pending(), "{}: not frozen", e.name);
            let json = inference::load(inference::Kind::Conditioning, &e.name);
            let design = e.raw["design"].as_str().expect("a design").to_string();
            let family_name = short_name(&e.raw);
            *seen
                .entry((design.clone(), family_name.clone()))
                .or_default() += 1;
            let family = member_family(&e.raw);
            let cols = numeric_columns(&e);
            let y = column(&cols, "y");
            let weights = e.raw["weights"].as_str().map(|w| column(&cols, w));
            let w_label = e.raw["weights"].as_str().unwrap_or("-");
            let n = y.len();
            let mut ws =
                build_workspace(&ModelSpec::fixed(family), &GroupIds::default(), n, 3, &opts);
            let mut ws_mixed = (design != "c").then(|| {
                let g = (0..n).map(|i| (i % GROUPS) as u32).collect();
                build_workspace(
                    &ModelSpec::mixed(family, ReStructure::intercept()),
                    &GroupIds::new(g),
                    n,
                    3,
                    &opts,
                )
            });
            for fit in json["fits"].as_array().expect("fits") {
                n_fits += 1;
                let label = fit["label"].as_str().expect("a label");
                let (x, offset) = band_design(fit["r_formula"].as_str().expect("a formula"), &cols);
                let mut data = Data::new(&x, y, n, 3);
                if let Some(w) = weights {
                    data = data.weights(w);
                }
                if let Some(o) = offset {
                    data = data.offset(o);
                }
                let f = fit_band(&mut ws, &data, &opts);
                let d = &f.diag;
                let at = format!("{design} {family_name:<7} w {w_label} {label:>6}");
                let Some(ws_mixed) = ws_mixed.as_mut() else {
                    assert_eq!(fit["finite_mle"].as_bool(), Some(false), "{at}: design c");
                    eprintln!(
                        "CONDITIONING {at}: converged {:<5} separated {:<5} se_unavailable {:<5}",
                        d.converged, d.separated, d.se_unavailable
                    );
                    verdicts_c.add(d, None, f64::NAN);
                    continue;
                };
                let beta_ref = numbers(&fit["beta_ref"], &at);
                let cov_ref: Vec<Vec<f64>> = fit["cov_ref"]
                    .as_array()
                    .expect("cov_ref")
                    .iter()
                    .map(|r| numbers(r, &at))
                    .collect();
                let tau_ref = number(&fit["tau_ref"], &at);
                let err = errors(&f, &beta_ref, &cov_ref);
                let tau_gap = (d.pivot / tau_ref).log10().abs();
                eprintln!(
                    "CONDITIONING {at}: tau_ref {tau_ref:9.2e} tau {:9.2e} converged {:<5} \
                     separated {:<5} se_unavailable {:<5} beta_rel {:9.2e} se_rel {:9.2e} \
                     dz {:9.2e} se<err {}",
                    d.pivot,
                    d.converged,
                    d.separated,
                    d.se_unavailable,
                    err.beta,
                    err.se,
                    err.dz,
                    err.understates
                );
                let decade = tau_ref.log10().floor() as i32;
                decades
                    .entry((design.clone(), -decade))
                    .or_insert_with(Tally::new)
                    .add(d, Some(&err), tau_gap);
                let mixed = fit_band(ws_mixed, &data, &opts);
                routes.push((at, region(d), region(&mixed.diag)));
            }
        }
        for ((design, neg_decade), t) in &decades {
            eprintln!(
                "CONDITIONING-DECADE {design} tau_ref in [1e{:+03}, 1e{:+03}): {} | max beta_rel \
                 {:9.2e} se_rel {:9.2e} dz {:9.2e} | se<err {} | max |log10(tau/tau_ref)| {:.2}",
                -neg_decade,
                1 - neg_decade,
                t.counts(),
                t.beta,
                t.se,
                t.dz,
                t.understates,
                t.tau_gap
            );
        }
        eprintln!(
            "CONDITIONING-DECADE c (no reference): {}",
            verdicts_c.counts()
        );
        let mut mismatches = Vec::new();
        for (at, glm, glmm) in &routes {
            eprintln!("CONDITIONING-ROUTES {at}: glm {glm:<14} glmm (1 | g) {glmm}");
            if glm != glmm {
                mismatches.push(format!("{at}: glm {glm}, glmm {glmm}"));
            }
        }
        eprintln!(
            "CONDITIONING-ROUTES {} of {} fits land in another region with (1 | g):",
            mismatches.len(),
            routes.len()
        );
        for m in &mismatches {
            eprintln!("  {m}");
        }
        for name in DUPLICATE_FAMILIES {
            let stem = format!("glm_band_d_{name}");
            let e = inference::Entry::from_raw(json!({
                "name": stem,
                "kind": "conditioning",
                "data": format!("data/inference/{stem}.csv"),
                "r_formula": null,
                "generator": "goldens_glm_band.py",
            }))
            .expect("a design d entry");
            let cols = numeric_columns(&e);
            let y = column(&cols, "y");
            let n = y.len();
            let (x, _) = band_design("y ~ 1 + a + x2", &cols);
            let mut ws = build_workspace(
                &ModelSpec::fixed(duplicate_family(name)),
                &GroupIds::default(),
                n,
                3,
                &opts,
            );
            let f = fit_band(&mut ws, &Data::new(&x, y, n, 3), &opts);
            eprintln!(
                "CONDITIONING-DUPLICATE {name:<7}: converged {:<5} se_unavailable {:<5} \
                 t_sq[x2] {:9.2e} pivot {:9.2e} (column {})",
                f.diag.converged, f.diag.se_unavailable, f.t_sq[2], f.diag.pivot, f.diag.pivot_col
            );
        }
        let want: BTreeMap<(String, String), usize> = ENTRIES
            .iter()
            .map(|&(d, f, k)| ((d.to_string(), f.to_string()), k))
            .collect();
        assert_eq!(seen, want, "the conditioning entries per design and family");
        assert_eq!(n_fits, FITS, "the conditioning fits");
    }

    /// Design b's unweighted logit fit at `τ_ref` 3.6e-11 converges at the
    /// reference β. From η = 0 its first Newton step raises the deviance by
    /// 18% (relative), and the steps after it grow until β runs off to 1e299,
    /// which the divergence guard reports as separated. A finite maximum
    /// exists (`finite_mle`), so the fit must halve its way back to it.
    #[test]
    fn design_b_logit_halves_back_from_an_overshoot() {
        let opts = FitOptions::default();
        let e = inference::entries(inference::Kind::Conditioning)
            .into_iter()
            .find(|e| e.name == "conditioning_b_logit")
            .expect("the conditioning_b_logit entry");
        let json = inference::load(inference::Kind::Conditioning, &e.name);
        let fit = json["fits"]
            .as_array()
            .expect("fits")
            .iter()
            .find(|f| f["label"] == "1e-11")
            .expect("the 1e-11 fit");
        assert_eq!(fit["finite_mle"].as_bool(), Some(true));
        let cols = numeric_columns(&e);
        let y = column(&cols, "y");
        let n = y.len();
        let (x, offset) = band_design(fit["r_formula"].as_str().expect("a formula"), &cols);
        let data = Data::new(&x, y, n, 3).offset(offset.expect("design b has an offset"));
        let mut ws = build_workspace(
            &ModelSpec::fixed(member_family(&e.raw)),
            &GroupIds::default(),
            n,
            3,
            &opts,
        );
        let f = fit_band(&mut ws, &data, &opts);
        assert!(
            f.diag.converged && !f.diag.separated,
            "converged {} separated {}",
            f.diag.converged,
            f.diag.separated
        );
        let beta_ref = numbers(&fit["beta_ref"], "beta_ref");
        for (j, (b, r)) in f.beta.iter().zip(&beta_ref).enumerate() {
            let rel = ((b - r) / r).abs();
            assert!(
                rel < 1e-4,
                "beta[{j}] {b} vs reference {r}: relative {rel:.2e}"
            );
        }
    }
}
