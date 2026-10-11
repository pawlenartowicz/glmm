//! The term map and the hypothesis matrices against lmerTest's, frozen in the
//! `lmertest_anova_*` references (`proposal_dependent: true` entries of kind
//! `lmertest`).

use serde_json::Value;

use super::hypotheses::{type2_xtx, type3, TermHypothesis};
use super::{lower, Lowered, Table};
use crate::inference::test_refs::{self, Entry, Kind};
use crate::test_support::csv_table;
use crate::Family;

/// Entry-by-entry band on the frozen matrices: lmerTest rounds them with
/// `zapsmall` (7 significant digits of the largest entry), which this port
/// does not reproduce bit for bit.
const L_ABS: f64 = 1e-6;

/// The per-term (anova) entries of kind `lmertest`, selected by their
/// `proposal_dependent` flag.
fn anova_entries() -> Vec<Entry> {
    let out: Vec<Entry> = test_refs::entries(Kind::Lmertest)
        .into_iter()
        .filter(|e| e.raw["proposal_dependent"] == Value::Bool(true))
        .collect();
    assert!(
        out.iter()
            .any(|e| e.name == "lmertest_anova_sim_factorial_empty"),
        "the rank-deficient anova entry is registered"
    );
    out
}

/// The entry's data as a [`Table`] ([`csv_table`]): the manifest `factors`,
/// and every column that does not parse as a number, as factor columns.
pub(crate) fn table_of(e: &Entry) -> Table {
    let data = e.data.as_ref().expect("anova entry has data");
    let path = format!("{}/validation/{data}", env!("CARGO_MANIFEST_DIR"));
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("{path}: {err}"));
    let factors: Vec<&str> = e.factors.iter().map(String::as_str).collect();
    csv_table(&raw, &factors)
}

fn lowered(e: &Entry) -> Lowered {
    let formula = e.r_formula.as_ref().expect("anova entry has a formula");
    lower(formula, &table_of(e), Family::Gaussian).unwrap_or_else(|err| panic!("{}: {err}", e.name))
}

/// The fit fields (`cols_dropped`, ...): the T1a twin's JSON when the entry
/// names one, the entry's own otherwise.
fn fit_fields(json: &Value) -> Value {
    match json.get("twin") {
        Some(Value::String(twin)) => test_refs::load(Kind::Lmertest, twin),
        _ => json.clone(),
    }
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|s| s.as_str().expect("string").to_string())
        .collect()
}

/// lme4's retained columns (`assign_names`) as caller column indices.
fn retained(lo: &Lowered, json: &Value, name: &str) -> Vec<usize> {
    strings(&json["assign_names"])
        .iter()
        .map(|c| {
            lo.col_names
                .iter()
                .position(|n| n == c)
                .unwrap_or_else(|| panic!("{name}: no design column {c}"))
        })
        .collect()
}

/// `got` against the frozen `hypotheses` of one table: the same terms in
/// order, the same row count per term, each entry within [`L_ABS`] on the
/// retained columns and exactly zero on the others.
fn assert_hypotheses(
    name: &str,
    table: &str,
    got: &[TermHypothesis],
    want: &Value,
    retained: &[usize],
    p: usize,
) {
    let want = want.as_array().expect("hypotheses array");
    let labels: Vec<&str> = got.iter().map(|h| h.term.as_str()).collect();
    let want_labels: Vec<&str> = want.iter().map(|h| h["term"].as_str().unwrap()).collect();
    assert_eq!(labels, want_labels, "{name} {table}: terms");
    for (h, w) in got.iter().zip(want) {
        let rows = w["L"].as_array().unwrap();
        let ctx = format!("{name} {table} {}", h.term);
        assert_eq!(h.l.nrows(), rows.len(), "{ctx}: rows");
        assert_eq!(h.l.ncols(), p, "{ctx}: columns");
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_array().unwrap();
            assert_eq!(row.len(), retained.len(), "{ctx}: frozen row width");
            for (j, &c) in retained.iter().enumerate() {
                let (g, w) = (h.l[(i, c)], row[j].as_f64().unwrap());
                assert!(
                    (g - w).abs() <= L_ABS,
                    "{ctx}: L[{i}, {c}] = {g}, lmerTest {w}"
                );
            }
            for c in (0..p).filter(|c| !retained.contains(c)) {
                assert_eq!(h.l[(i, c)], 0.0, "{ctx}: aliased column {c}, row {i}");
            }
        }
    }
}

#[test]
fn lowered_terms_match_r_assign() {
    for e in anova_entries() {
        let json = test_refs::load(Kind::Lmertest, &e.name);
        let lo = lowered(&e);
        let name = e.name.as_str();
        let labels: Vec<&str> = lo.terms.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, strings(&json["term_labels"]), "{name}: term labels");
        // The ranges partition the columns after the intercept, in order.
        let mut next = usize::from(lo.has_intercept);
        for t in &lo.terms {
            assert_eq!(t.columns.start, next, "{name} {}: contiguous", t.label);
            next = t.columns.end;
            for (v, &num) in t.vars.iter().zip(&t.numeric) {
                assert_eq!(num, !e.factors.contains(v), "{name} {v}: kind");
            }
        }
        assert_eq!(next, lo.p, "{name}: the terms cover the design");
        // `assign` covers lme4's retained columns only; the others are the
        // fit's `cols_dropped`.
        let assign: Vec<u64> = json["assign"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_u64().unwrap())
            .collect();
        let kept = retained(&lo, &json, name);
        assert_eq!(assign.len(), kept.len(), "{name}: assign length");
        for (&a, &c) in assign.iter().zip(&kept) {
            if a == 0 {
                assert!(lo.has_intercept && c == 0, "{name}: intercept column {c}");
            } else {
                let t = &lo.terms[a as usize - 1];
                assert!(
                    t.columns.contains(&c),
                    "{name}: column {c} not in {}",
                    t.label
                );
            }
        }
        let mut dropped: Vec<String> = (0..lo.p)
            .filter(|c| !kept.contains(c))
            .map(|c| lo.col_names[c].clone())
            .collect();
        dropped.sort();
        let mut want = strings(&fit_fields(&json)["cols_dropped"]);
        want.sort();
        assert_eq!(dropped, want, "{name}: columns outside assign");
    }
}

#[test]
fn type3_matches_lmertest_hypotheses() {
    for e in anova_entries() {
        let json = test_refs::load(Kind::Lmertest, &e.name);
        let lo = lowered(&e);
        let kept = retained(&lo, &json, &e.name);
        let got = type3(&lo, &kept);
        assert_hypotheses(
            &e.name,
            "type3",
            &got,
            &json["type3"]["hypotheses"],
            &kept,
            lo.p,
        );
    }
}

#[test]
fn type2_xtx_matches_lmertest_hypotheses() {
    for e in anova_entries() {
        let json = test_refs::load(Kind::Lmertest, &e.name);
        let lo = lowered(&e);
        let kept = retained(&lo, &json, &e.name);
        let got = type2_xtx(&lo, &kept);
        assert_hypotheses(
            &e.name,
            "type2",
            &got,
            &json["type2"]["hypotheses"],
            &kept,
            lo.p,
        );
    }
}

/// Non-marginal formulas, where R codes a variable 2 in
/// `attr(terms, "factors")` (`x` inside `x:g`, `g` inside `g:h`): lmerTest
/// counts only the variables coded 1, finds no term containing another, and
/// gives every term its unit rows in both tables. Computed with lmerTest 3.2.1
/// on `sim_factorial`, for each formula `f` below with `(1 | subj)` (the
/// matrices do not depend on the random part):
/// `m <- lmer(f, d); lmerTest:::get_contrasts_type3(m);
/// lmerTest:::get_contrasts_type2_unfolded(m)`. Counting every variable
/// instead puts `x` inside `x:g` and `g` inside `g:h`, which moves Type III
/// F(x) on `y ~ x + x:g + (1 | subj)` from lmerTest's 13.50 to 23.06. The
/// first formula is the frozen `anova_nonmarginal_sim_factorial`'s, whose term
/// labels and coefficient names the lowering must match.
#[test]
fn nonmarginal_terms_contain_nothing() {
    let e = test_refs::entries(Kind::Anova)
        .into_iter()
        .find(|e| e.name == "anova_nonmarginal_sim_factorial")
        .expect("the non-marginal anova entry is registered");
    let json = test_refs::load(Kind::Anova, &e.name);
    let first = e.r_formula.clone().expect("anova entry has a formula");
    // `h`, which the frozen formula does not name, reads as a factor too.
    let data = table_of(&e);
    for formula in [first.as_str(), "y ~ g + g:h + x + (1 | subj)"] {
        let lo = lower(formula, &data, Family::Gaussian)
            .unwrap_or_else(|err| panic!("{formula}: {err}"));
        if formula == first {
            let labels: Vec<&str> = lo.terms.iter().map(|t| t.label.as_str()).collect();
            assert_eq!(labels, strings(&json["term_labels"]), "{formula}: terms");
            assert_eq!(
                lo.col_names,
                strings(&json["coef_names"]),
                "{formula}: columns"
            );
        }
        let all: Vec<usize> = (0..lo.p).collect();
        for (table, got) in [("type3", type3(&lo, &all)), ("type2", type2_xtx(&lo, &all))] {
            assert_eq!(got.len(), lo.terms.len(), "{formula} {table}: terms");
            for (h, term) in got.iter().zip(&lo.terms) {
                assert_eq!(
                    h.l.nrows(),
                    term.columns.len(),
                    "{formula} {table} {}",
                    h.term
                );
                for i in 0..h.l.nrows() {
                    for j in 0..lo.p {
                        let want = f64::from(u8::from(j == term.columns.start + i));
                        assert!(
                            (h.l[(i, j)] - want).abs() <= L_ABS,
                            "{formula} {table} {}: L[{i}, {j}] = {}, lmerTest {want}",
                            h.term,
                            h.l[(i, j)]
                        );
                    }
                }
            }
        }
    }
}

/// `sim_factorial_empty` has no `gc:hq` cell, so lme4 drops that column:
/// lmerTest's matrices live on the six retained columns, and `type3` puts
/// them in the seven-column caller layout with zeros on `gc:hq`.
#[test]
fn type3_rank_deficient_scatters_zeros() {
    let e = anova_entries()
        .into_iter()
        .find(|e| e.name == "lmertest_anova_sim_factorial_empty")
        .unwrap();
    let json = test_refs::load(Kind::Lmertest, &e.name);
    let lo = lowered(&e);
    let kept = retained(&lo, &json, &e.name);
    let aliased = lo.col_names.iter().position(|c| c == "gc:hq").unwrap();
    assert_eq!((lo.p, kept.len()), (7, 6));
    assert!(!kept.contains(&aliased));
    let got = type3(&lo, &kept);
    assert_hypotheses(
        &e.name,
        "type3",
        &got,
        &json["type3"]["hypotheses"],
        &kept,
        lo.p,
    );
    assert!(got
        .iter()
        .all(|h| h.l.col(aliased).iter().all(|&v| v == 0.0)));
}
