//! Gate 1's per-term rows: lmerTest's Type III and Type II tables (the
//! `lmertest_anova_*` references, `proposal_dependent: true`, on REML fits and,
//! as the `_ml` ones, on ML fits, each fitted under its own `reml`) against the
//! crate's hypothesis matrices through the contest layer, in both parts of
//! gate 1. At lme4's θ̂: the pass at the reference's θ̂ (its twin's) on each
//! route the fit can take (the value route, and the gradient route on a
//! per-group fit), with lme4's β̂, NumDF equal and DenDF and F within gate 1's
//! band there for the route, the spread read from the term's frozen
//! `DenDF_second`. End to end the pass takes whichever route the fit does: `fit_cold`
//! and [`Fit::contrast_f`](crate::Fit::contrast_f), NumDF equal and DenDF and
//! F within [`DF_E2E_REL`].
//!
//! Registry-backed, one driver per part: a row over its band passes only under
//! an `oracle-tier` entry of `validation/grid/divergences.json` on the
//! reference under the row's own quantity (`dendf_at_lme4` and `f_at_lme4` at
//! lme4's θ̂, `df` for DenDF and `f` for F end to end), and an entry under a
//! part's quantities on a reference compared here fails when it did not fire
//! or when it also lists a quantity outside that part.

use faer::Mat;
use serde_json::Value;

use super::hypotheses::{type2_xtx, type3, TermHypothesis};
use crate::inference::contest::{contest_md, ContestInputs};
use crate::inference::test_refs::{self, Kind};
use crate::lmm::satterthwaite::tests::gate1::{
    at_lme4_theta, band, lmertest_entries, lowered, newton_point, numbers, route_pass_at, routes,
    spread, strings, Driver,
};
use crate::lmm::satterthwaite::{DerivRoute, SatterthwaiteState};

/// `df_e2e_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`): gate 1's
/// end-to-end band on a df, at the crate's own θ̂. Change together.
const DF_E2E_REL: f64 = 1e-3;

/// The tables of each reference, as their JSON keys.
const TABLES: [&str; 2] = ["type3", "type2"];

/// The crate's hypothesis matrices of `table` on the `retained` columns.
fn hypotheses(
    table: &str,
    lo: &crate::formula::Lowered,
    retained: &[usize],
) -> Vec<TermHypothesis> {
    match table {
        "type3" => type3(lo, retained),
        _ => type2_xtx(lo, retained),
    }
}

/// `L` restricted to the `kept` columns, the layout of the state's matrices.
fn restrict(l: &Mat<f64>, kept: &[usize]) -> Mat<f64> {
    Mat::from_fn(l.nrows(), kept.len(), |i, j| l[(i, kept[j])])
}

/// lmerTest's inputs from the pass's state: `Jac_eff = [∂C/∂θ_k …, C/τ̂]`.
fn jac_eff(st: &SatterthwaiteState) -> Vec<Mat<f64>> {
    let mut jac = st.jac_theta.clone();
    jac.push(Mat::from_fn(st.c.nrows(), st.c.ncols(), |a, b| {
        st.c[(a, b)] / st.tau
    }));
    jac
}

/// The non-marginal formula `y ~ x + x:g + (1 + x | subj)` end to end, against
/// lmerTest's frozen Type III and Type II rows (`anova_nonmarginal_sim_factorial`):
/// `fit_cold`, the crate's matrices and [`Fit::contrast_f`](crate::Fit::contrast_f),
/// NumDF equal and F and DenDF within [`DF_E2E_REL`], registry-backed under the
/// end-to-end labels `df` and `f` as `per_term_f_matches_anova`'s rows are.
/// Containment read over every variable of a term (not only those R codes 1)
/// puts `x` inside `x:g` and moves F(x) far outside the band.
#[test]
fn nonmarginal_f_matches_anova() {
    let mut end_to_end = Driver::new(&["df", "f"]);
    let e = test_refs::entries(Kind::Anova)
        .into_iter()
        .find(|e| e.name == "anova_nonmarginal_sim_factorial")
        .expect("the non-marginal anova entry is registered");
    let json = test_refs::load(Kind::Anova, &e.name);
    let lo = lowered(&e);
    let fit = crate::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged(), "the fit converges");
    assert!(fit.aliased().iter().all(|&a| !a), "full rank");
    let all: Vec<usize> = (0..lo.p).collect();
    for table in TABLES {
        let rows = json[format!("lmertest_{table}").as_str()]["rows"]
            .as_array()
            .unwrap_or_else(|| panic!("{table}: rows"));
        let hyps = hypotheses(table, &lo, &all);
        assert_eq!(hyps.len(), rows.len(), "{table}: terms");
        for (h, row) in hyps.iter().zip(rows) {
            assert_eq!(Some(h.term.as_str()), row["term"].as_str(), "{table}");
            let flat: Vec<f64> = (0..h.l.nrows())
                .flat_map(|i| (0..lo.p).map(move |j| (i, j)))
                .map(|(i, j)| h.l[(i, j)])
                .collect();
            let got = fit
                .contrast_f(&flat, h.l.nrows())
                .unwrap_or_else(|err| panic!("{table} {}: {err}", h.term));
            assert_eq!(
                Some(got.num_df),
                row["NumDF"].as_f64(),
                "{table} {}: NumDF",
                h.term
            );
            for (quantity, what, g, key) in [
                ("f", "F", got.f, "F value"),
                ("df", "DenDF", got.den_df, "DenDF"),
            ] {
                let w = row[key].as_f64().expect("a number");
                let what = format!("{table} {} {what}", h.term);
                end_to_end.check(&e.name, quantity, &what, (g, w), DF_E2E_REL);
            }
        }
    }
    end_to_end.compared(&e.name);
    end_to_end.finish();
}

#[test]
fn per_term_f_matches_anova() {
    let mut at_lme4 = Driver::new(&["dendf_at_lme4", "f_at_lme4"]);
    let mut per_group = Vec::new();
    let mut end_to_end = Driver::new(&["df", "f"]);
    let entries: Vec<_> = [true, false]
        .into_iter()
        .flat_map(|reml| lmertest_entries(true, reml))
        .collect();
    let n_ml = entries.iter().filter(|e| e.name.ends_with("_ml")).count();
    eprintln!(
        "PERTERM {} per-term references compared, {n_ml} of them ML",
        entries.len()
    );
    for e in entries {
        let name = e.name.as_str();
        let json = test_refs::load(Kind::Lmertest, name);
        let fit_json = match json.get("twin") {
            Some(Value::String(twin)) => test_refs::load(Kind::Lmertest, twin),
            _ => json.clone(),
        };
        let lo = lowered(&e);
        let beta = numbers(&fit_json["beta"], name);

        // At lme4's θ̂, with lme4's β̂ (a function of θ̂ alone).
        let kept = at_lme4_theta(name, &lo, &fit_json, |kept, ws, map, theta| {
            let routes = routes(ws);
            if routes.contains(&DerivRoute::GradientDiff) {
                per_group.push(name.to_string());
            }
            for route in routes {
                let st0 = route_pass_at(ws, theta, map, route);
                let theta1 = newton_point(ws, &st0, theta);
                let st1 = route_pass_at(ws, &theta1, map, route);
                let (jac0, jac1) = (jac_eff(&st0), jac_eff(&st1));
                let at = |st: &SatterthwaiteState, jac: &[Mat<f64>], l: &Mat<f64>| {
                    let inputs = ContestInputs {
                        beta: &beta,
                        vcov_beta: st.c.as_ref(),
                        vcov_varpar: st.a_eff.as_ref(),
                        jac,
                    };
                    contest_md(&inputs, l.as_ref()).unwrap_or_else(|err| panic!("{name}: {err:?}"))
                };
                for table in TABLES {
                    let frozen = &json[table];
                    let terms = strings(&frozen["terms"], name);
                    let num_df = numbers(&frozen["NumDF"], name);
                    let den_df = numbers(&frozen["DenDF"], name);
                    let f = numbers(&frozen["F"], name);
                    let second = frozen["DenDF_second"]
                        .as_array()
                        .unwrap_or_else(|| panic!("{name} {table}: DenDF_second"));
                    let hyps = hypotheses(table, &lo, kept);
                    assert_eq!(hyps.len(), terms.len(), "{name} {table}: terms");
                    for (k, h) in hyps.iter().enumerate() {
                        assert_eq!(h.term, terms[k], "{name} {table}: term {k}");
                        assert!(h.l.nrows() > 0, "{name} {table} {}: no rows", h.term);
                        let l = restrict(&h.l, kept);
                        let (md0, md1) = (at(&st0, &jac0, &l), at(&st1, &jac1, &l));
                        let what = format!("{table} {} at lme4's theta, {route:?} route", h.term);
                        assert_eq!(md0.num_df as f64, num_df[k], "{name} {what}: NumDF");
                        let sp = spread(den_df[k], &numbers(&second[k], name));
                        let nonstat = (md0.den_df - md1.den_df).abs() / den_df[k];
                        let b = band(route, sp, nonstat);
                        eprintln!(
                            "PERTERM {name} {what}: DenDF rel {:.2e} F rel {:.2e} spread {sp:.2e} nonstat {nonstat:.2e} band {b:.2e}",
                            (md0.den_df - den_df[k]).abs() / den_df[k],
                            (md0.f_value - f[k]).abs() / f[k]
                        );
                        at_lme4.check(
                            name,
                            "dendf_at_lme4",
                            &format!("{what} DenDF"),
                            (md0.den_df, den_df[k]),
                            b,
                        );
                        at_lme4.check(
                            name,
                            "f_at_lme4",
                            &format!("{what} F"),
                            (md0.f_value, f[k]),
                            b,
                        );
                    }
                }
            }
            kept.to_vec()
        });

        // End to end, at the crate's own θ̂.
        let fit = crate::fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        assert!(fit.converged(), "{name}: the fit converges");
        assert_eq!(
            Some(fit.reml),
            e.raw["reml"].as_bool(),
            "{name}: the fit's criterion"
        );
        let retained: Vec<usize> = (0..lo.p).filter(|&j| !fit.aliased()[j]).collect();
        assert_eq!(
            retained, kept,
            "{name}: the crate aliases what lme4 dropped"
        );
        for table in TABLES {
            let frozen = &json[table];
            let num_df = numbers(&frozen["NumDF"], name);
            let den_df = numbers(&frozen["DenDF"], name);
            let f = numbers(&frozen["F"], name);
            for (k, h) in hypotheses(table, &lo, &retained).iter().enumerate() {
                let flat: Vec<f64> = (0..h.l.nrows())
                    .flat_map(|i| (0..lo.p).map(move |j| (i, j)))
                    .map(|(i, j)| h.l[(i, j)])
                    .collect();
                let got = fit
                    .contrast_f(&flat, h.l.nrows())
                    .unwrap_or_else(|err| panic!("{name}: {err}"));
                let what = format!("{table} {} end to end", h.term);
                assert_eq!(got.num_df, num_df[k], "{name} {what}: NumDF");
                eprintln!(
                    "PERTERM {name} {what}: DenDF rel {:.2e} F rel {:.2e}",
                    (got.den_df - den_df[k]).abs() / den_df[k],
                    (got.f - f[k]).abs() / f[k]
                );
                end_to_end.check(
                    name,
                    "df",
                    &format!("{what} DenDF"),
                    (got.den_df, den_df[k]),
                    DF_E2E_REL,
                );
                end_to_end.check(name, "f", &format!("{what} F"), (got.f, f[k]), DF_E2E_REL);
            }
        }
        at_lme4.compared(name);
        end_to_end.compared(name);
    }
    at_lme4.finish();
    per_group.sort();
    assert_eq!(
        per_group,
        [
            "lmertest_anova_Machines",
            "lmertest_anova_Machines_ml",
            "lmertest_anova_sim_factorial",
            "lmertest_anova_sim_factorial_empty",
            "lmertest_anova_sim_factorial_ml",
            "lmertest_anova_sleepstudy",
            "lmertest_anova_sleepstudy_ml",
        ],
        "the references whose fit arms the per-group gradient"
    );
    end_to_end.finish();
}
