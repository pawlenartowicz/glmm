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

use glmm::Fit;
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
