//! The term map and the hypothesis matrices against lmerTest's, frozen in the
//! `lmertest_anova_*` references (`proposal_dependent: true` entries of kind
//! `lmertest`), and against car's (the `anova_car_*` entries of kind
//! `anova`); the marginal-term check and the missing-cell note.

use faer::Mat;
use serde_json::Value;

use super::hypotheses::{
    check_marginal, missing_cells_note, type2_vcov, type2_xtx, type3, TermHypothesis,
};
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

/// car's GLM and GLMM entries on `sim_factorial_bern`, whose `g`, `h` and
/// `x` columns are `sim_factorial`'s, so their design is
/// `lmertest_anova_sim_factorial`'s.
const CAR_BERN: [&str; 2] = [
    "anova_car_glm_sim_factorial_bern",
    "anova_car_glmm_sim_factorial_bern",
];

/// Projector band of gate 2: equal row spaces.
const PROJ_ABS: f64 = 1e-10;

fn anova_entry(name: &str) -> Entry {
    test_refs::entries(Kind::Anova)
        .into_iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("{name} is registered"))
}

fn matrix_rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("matrix as row arrays")
        .iter()
        .map(|r| {
            r.as_array()
                .expect("row array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        })
        .collect()
}

/// A frozen `L` (rows over the named columns) in the caller layout of `lo`.
fn frozen_l(rows: &Value, cols: &[String], lo: &Lowered) -> Mat<f64> {
    let rows = matrix_rows(rows);
    let pos: Vec<usize> = cols
        .iter()
        .map(|c| lo.col_names.iter().position(|n| n == c).unwrap())
        .collect();
    let mut l = Mat::<f64>::zeros(rows.len(), lo.p);
    for (i, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), pos.len(), "frozen row width");
        for (&v, &c) in row.iter().zip(&pos) {
            l[(i, c)] = v;
        }
    }
    l
}

/// The orthogonal projector `L'(LL')⁻¹L` on the row space of `L`.
fn projector(l: &Mat<f64>) -> Mat<f64> {
    let mut x = l.clone();
    let gram = l * l.transpose();
    crate::linalg::SeqLlt::new(gram.as_ref())
        .expect("L has full row rank")
        .solve_in_place(x.as_mut());
    l.transpose() * x
}

fn max_abs_diff(a: &Mat<f64>, b: &Mat<f64>) -> f64 {
    let mut d = 0.0_f64;
    for j in 0..a.ncols() {
        for i in 0..a.nrows() {
            d = d.max((a[(i, j)] - b[(i, j)]).abs());
        }
    }
    d
}

/// Gate 2, car's Type II: `type2_vcov` fed the frozen treatment-fit `vcov`
/// (car's metric, not glmm's own fit) spans each term's car `L` (built by
/// `car:::ConjComp` over `car:::relatives`) on the GLM and the GLMM.
#[test]
fn type2_vcov_spans_car_type2_rows() {
    for name in CAR_BERN {
        let e = anova_entry(name);
        let json = test_refs::load(Kind::Anova, name);
        let lo = lowered(&e);
        assert_eq!(
            lo.col_names,
            strings(&json["treatment"]["coef_names"]),
            "{name}: columns"
        );
        let all: Vec<usize> = (0..lo.p).collect();
        let vcov = matrix_rows(&json["treatment"]["vcov"]);
        let got = type2_vcov(&lo, &all, &vcov);
        let want = json["type2"]["rows"].as_array().unwrap();
        let labels: Vec<&str> = got.iter().map(|h| h.term.as_str()).collect();
        let want_labels: Vec<&str> = want.iter().map(|w| w["term"].as_str().unwrap()).collect();
        assert_eq!(labels, want_labels, "{name}: terms");
        for (h, w) in got.iter().zip(want) {
            let l = frozen_l(&w["L"], &strings(&w["col_names"]), &lo);
            assert_eq!(h.l.nrows(), l.nrows(), "{name} {}: rows", h.term);
            let d = max_abs_diff(&projector(&h.l), &projector(&l));
            assert!(d <= PROJ_ABS, "{name} {}: projector off by {d:e}", h.term);
        }
    }
}

/// Relative band of `V`-orthogonality, on `Σ_j |L[i, j]·V[j][c]|`.
const ORTH_REL: f64 = 1e-12;

fn column(lo: &Lowered, name: &str) -> usize {
    lo.col_names
        .iter()
        .position(|n| n == name)
        .unwrap_or_else(|| panic!("no design column {name}"))
}

/// Every row of `h` is `V`-orthogonal to each column named in `cols`
/// (`L·V·e_c = 0` over the `retained` columns), as car's Type II rows are to
/// every relative column.
fn assert_v_orthogonal(
    h: &TermHypothesis,
    vcov: &[Vec<f64>],
    lo: &Lowered,
    retained: &[usize],
    cols: &[&str],
) {
    for name in cols {
        let c = column(lo, name);
        for i in 0..h.l.nrows() {
            let (dot, scale) = retained
                .iter()
                .map(|&j| h.l[(i, j)] * vcov[j][c])
                .fold((0.0, 0.0), |(d, s), t: f64| (d + t, s + t.abs()));
            assert!(
                dot.abs() <= ORTH_REL * scale,
                "{}: row {i}·V·e({name}) = {dot:e} on a scale of {scale:e}",
                h.term
            );
        }
    }
}

/// car's relatives count numeric variables (`car:::relatives` reads every
/// variable `attr(terms, "factors")` marks), lmerTest's containment does
/// not: on `y ~ g * x` (`lmertest_anova_sim_factorial_gx`, `g` a 3-level
/// factor, `x` numeric) `g:x` is a relative of `g` for car and does not
/// contain `g` for lmerTest. Fed that entry's frozen `vcov`, `g`'s rows are
/// `V`-orthogonal to the `gb:x` and `gc:x` columns, and not zero on them.
#[test]
fn type2_vcov_counts_numeric_relatives() {
    let e = anova_entries()
        .into_iter()
        .find(|e| e.name == "lmertest_anova_sim_factorial_gx")
        .expect("the g * x anova entry is registered");
    let json = test_refs::load(Kind::Lmertest, &e.name);
    let lo = lowered(&e);
    assert_eq!(lo.col_names, strings(&json["coef_names"]), "columns");
    let all: Vec<usize> = (0..lo.p).collect();
    let vcov = matrix_rows(&json["vcov_beta"]);
    let got = type2_vcov(&lo, &all, &vcov);
    let g = got.iter().find(|h| h.term == "g").expect("term g");
    assert_eq!(g.l.nrows(), 2);
    let relatives = ["gb:x", "gc:x"];
    assert_v_orthogonal(g, &vcov, &lo, &all, &relatives);
    for name in relatives {
        let c = column(&lo, name);
        assert!(
            (0..g.l.nrows()).any(|i| g.l[(i, c)] != 0.0),
            "g: zero on {name}"
        );
    }
}

/// `type2_vcov` reads only the retained block of the `p × p` `vcov`. On
/// `y ~ g * h * x` over `sim_factorial_empty` (no `gc:hq` cell) the all-zero
/// `gc:hq` and `gc:hq:x` columns are aliased, `gc:hq` before retained ones,
/// and their `vcov` rows and columns are NaN; the retained block is
/// `(XᵀX)⁻¹` there. Every row is finite and zero on the aliased columns, and
/// `g`'s and `h`'s rows are `V`-orthogonal to their retained relative
/// columns.
#[test]
fn type2_vcov_reads_the_retained_block() {
    let e = anova_entries()
        .into_iter()
        .find(|e| e.name == "lmertest_anova_sim_factorial_empty")
        .unwrap();
    let lo = lower(
        "y ~ g * h * x + (1 | subj)",
        &table_of(&e),
        Family::Gaussian,
    )
    .unwrap();
    let (n, p) = (lo.n, lo.p);
    let aliased: Vec<usize> = (0..p)
        .filter(|&c| (0..n).all(|i| lo.x[i * p + c] == 0.0))
        .collect();
    let names: Vec<&str> = aliased.iter().map(|&c| lo.col_names[c].as_str()).collect();
    assert_eq!(names, ["gc:hq", "gc:hq:x"]);
    let kept: Vec<usize> = (0..p).filter(|c| !aliased.contains(c)).collect();
    let pr = kept.len();
    let gram = Mat::from_fn(pr, pr, |a, b| {
        (0..n)
            .map(|i| lo.x[i * p + kept[a]] * lo.x[i * p + kept[b]])
            .sum::<f64>()
    });
    let mut inv = Mat::<f64>::identity(pr, pr);
    crate::linalg::SeqLlt::new(gram.as_ref())
        .expect("full column rank on the kept columns")
        .solve_in_place(inv.as_mut());
    let mut vcov = vec![vec![f64::NAN; p]; p];
    for (a, &ca) in kept.iter().enumerate() {
        for (b, &cb) in kept.iter().enumerate() {
            vcov[ca][cb] = inv[(a, b)];
        }
    }
    let got = type2_vcov(&lo, &kept, &vcov);
    for h in &got {
        for i in 0..h.l.nrows() {
            for j in 0..p {
                let v = h.l[(i, j)];
                if aliased.contains(&j) {
                    assert_eq!(v, 0.0, "{}: aliased column {j}, row {i}", h.term);
                } else {
                    assert!(v.is_finite(), "{}: L[{i}, {j}] = {v}", h.term);
                }
            }
        }
    }
    for (term, relatives) in [
        ("g", &["gb:hq", "gb:x", "gc:x", "gb:hq:x"][..]),
        ("h", &["gb:hq", "hq:x", "gb:hq:x"][..]),
    ] {
        let h = got.iter().find(|h| h.term == term).expect("term");
        assert_v_orthogonal(h, &vcov, &lo, &kept, relatives);
    }
}

/// Gate 2, Type III: on the `sim_factorial_bern` design, `type3` spans
/// lmerTest's Type III rows (`lmertest_anova_sim_factorial`) at
/// [`PROJ_ABS`]: `type3` rounds with `zapsmall` as lmerTest does, so its rows
/// equal lmerTest's rounded ones.
#[test]
fn type3_spans_lmertest_type3_rows_on_bern_design() {
    let lt = test_refs::load(Kind::Lmertest, "lmertest_anova_sim_factorial");
    let cols = strings(&lt["assign_names"]);
    let want = lt["type3"]["hypotheses"].as_array().unwrap();
    for name in CAR_BERN {
        let lo = lowered(&anova_entry(name));
        let all: Vec<usize> = (0..lo.p).collect();
        let got = type3(&lo, &all);
        let labels: Vec<&str> = got.iter().map(|h| h.term.as_str()).collect();
        let want_labels: Vec<&str> = want.iter().map(|w| w["term"].as_str().unwrap()).collect();
        assert_eq!(labels, want_labels, "{name}: terms");
        for (h, w) in got.iter().zip(want) {
            let l = frozen_l(&w["L"], &cols, &lo);
            assert_eq!(h.l.nrows(), l.nrows(), "{name} {}: rows", h.term);
            let d = max_abs_diff(&projector(&h.l), &projector(&l));
            assert!(d <= PROJ_ABS, "{name} {}: projector off by {d:e}", h.term);
        }
    }
}

/// Gate 2, row order: `Lowered::terms` arrives in R's
/// `attr(terms, "term.labels")` order on every per-term `anova` entry (the
/// `lmertest` entries: [`lowered_terms_match_r_assign`]); nothing sorts it.
#[test]
fn lowered_terms_follow_r_term_labels() {
    let mut seen = 0;
    for e in test_refs::entries(Kind::Anova) {
        let json = test_refs::load(Kind::Anova, &e.name);
        let Some(want) = json.get("term_labels") else {
            continue;
        };
        let lo = lowered(&e);
        let labels: Vec<&str> = lo.terms.iter().map(|t| t.label.as_str()).collect();
        assert_eq!(labels, strings(want), "{}: term labels", e.name);
        seen += 1;
    }
    assert!(
        seen >= 3,
        "the car and non-marginal entries carry term labels"
    );
}

/// The per-term tests refuse a formula whose interaction lacks a marginal
/// term, naming it: `anova_nonmarginal_sim_factorial` records lmerTest and
/// car disagreeing on `y ~ x + x:g` (Type III F(x) 11.4243 against car's Type
/// II χ² 20.14) with no condition from either, so the refusal stands.
#[test]
fn marginal_check_names_the_missing_term() {
    let e = anova_entry("anova_nonmarginal_sim_factorial");
    let msg = check_marginal(&lowered(&e)).expect_err("y ~ x + x:g is refused");
    assert!(msg.contains("`g`") && msg.contains("`x:g`"), "{msg}");
    let lo = lower(
        "y ~ g + g:h + x + (1 | subj)",
        &table_of(&e),
        Family::Gaussian,
    )
    .unwrap();
    let msg = check_marginal(&lo).expect_err("y ~ g + g:h + x is refused");
    assert!(msg.contains("`h`") && !msg.contains("`g`"), "{msg}");
    for name in CAR_BERN {
        assert_eq!(
            check_marginal(&lowered(&anova_entry(name))),
            Ok(()),
            "{name}"
        );
    }
}

/// lmerTest's missing-cell message on `sim_factorial_empty` (frozen among
/// the entry's conditions), whitespace aside; none on `sim_factorial`.
#[test]
fn missing_cells_note_on_empty_cell() {
    for e in anova_entries() {
        let json = test_refs::load(Kind::Lmertest, &e.name);
        let frozen = json["conditions"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|c| c["text"].as_str())
            .find(|t| t.starts_with("Missing cells"))
            .map(|t| t.split_whitespace().collect::<Vec<_>>().join(" "));
        let got = missing_cells_note(&lowered(&e));
        assert_eq!(got, frozen, "{}", e.name);
        if e.name == "lmertest_anova_sim_factorial_empty" {
            assert!(got.is_some_and(|n| n.contains("gc:hq")), "{}", e.name);
        }
    }
}
