//! The term map and the hypothesis matrices against lmerTest's, frozen in the
//! `lmertest_anova_*` references (`proposal_dependent: true` entries of kind
//! `lmertest`), and against car's (the `anova_car_*` entries of kind
//! `anova`); the marginal-term check and the missing-cell note.

use faer::Mat;
use serde_json::Value;

use super::hypotheses::{
    check_marginal, missing_cells_note, remap_slopes, type2_lr_sets, type2_vcov, type2_xtx, type3,
    type3_null_basis, type3_unrounded, untouched_columns, ReducedDesignError, TermHypothesis,
};
use super::{lower, Lowered, Table};
use crate::inference::test_refs::{self, Entry, Kind};
use crate::test_support::csv_table;
use crate::{ColumnId, Family};

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
    // The cases whose `term_labels` this reads, by name (not the
    // likelihood-ratio tables: the two cbpp ones have a cbind() response,
    // which `lower` refuses).
    const CASES: [&str; 4] = ["car_glmm", "car_glm", "car_lm", "nonmarginal"];
    let mut seen = 0;
    for e in test_refs::entries(Kind::Anova) {
        let json = test_refs::load(Kind::Anova, &e.name);
        if !CASES.contains(&json["case"].as_str().expect("case")) {
            continue;
        }
        let want = json.get("term_labels").expect("term_labels");
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

/// `‖m‖_F`.
fn frobenius(m: &Mat<f64>) -> f64 {
    m.col_iter()
        .flat_map(|c| c.iter().copied())
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// Gate 1, Type III: on `y ~ g * h + x` (the `sim_factorial_bern` design, all
/// columns retained) each term's `N` is an orthonormal basis of the null
/// space of its unrounded `L` ([`type3_unrounded`]): `‖L·N‖ ≤ 1e-12·‖L‖`,
/// `NᵀN = I`, the columns the rounded `L` ([`type3`]) does not touch lead `N`
/// as unit columns, and `N` has `p − r` columns (`r` = 2, 1, 1, 2 for `g`,
/// `h`, `x`, `g:h`). A row that is the sum of `L`'s rows adds no rank. `x`'s
/// `L` touches `x` alone, so its `N` is the column drop.
#[test]
fn type3_null_basis_spans_the_null_space() {
    let lo = lowered(&anova_entry(CAR_BERN[0]));
    let p = lo.p;
    assert_eq!(p, 7, "columns");
    let all: Vec<usize> = (0..p).collect();
    let hyp = type3(&lo, &all);
    let exact = type3_unrounded(&lo, &all);
    let labels: Vec<&str> = hyp.iter().map(|h| h.term.as_str()).collect();
    assert_eq!(labels, ["g", "h", "x", "g:h"]);
    let stack = |l: &Mat<f64>| {
        let rows = l.nrows();
        Mat::from_fn(rows + 1, p, |i, j| {
            if i < rows {
                l[(i, j)]
            } else {
                (0..rows).map(|k| l[(k, j)]).sum()
            }
        })
    };
    for ((h, e), r) in hyp.iter().zip(&exact).zip([2, 1, 1, 2]) {
        assert_eq!(e.term, h.term);
        let (stacked, stacked_exact) = (stack(&h.l), stack(&e.l));
        for (l, el) in [(&h.l, &e.l), (&stacked, &stacked_exact)] {
            let ctx = format!("{} ({} rows)", h.term, l.nrows());
            let n = type3_null_basis(l.as_ref(), el.as_ref(), &[])
                .unwrap_or_else(|e| panic!("{ctx}: {e:?}"));
            assert_eq!(n.ncols(), p - r, "{ctx}: p − r columns");
            let ln = el * &n;
            assert!(
                frobenius(&ln) <= 1e-12 * frobenius(el),
                "{ctx}: ‖L·N‖ = {:e}, ‖L‖ = {:e}",
                frobenius(&ln),
                frobenius(el)
            );
            let gram = n.transpose() * &n;
            let d = max_abs_diff(&gram, &Mat::identity(p - r, p - r));
            assert!(d <= 1e-12, "{ctx}: NᵀN off the identity by {d:e}");
            let kept = untouched_columns(l.as_ref());
            assert!(kept.contains(&0), "{ctx}: the intercept is untouched");
            for (j, &c) in kept.iter().enumerate() {
                for i in 0..p {
                    let unit = f64::from(u8::from(i == c));
                    assert_eq!(n[(i, j)], unit, "{ctx}: N[{i}, {j}], unit column {c}");
                }
            }
            if h.term == "x" {
                assert_eq!(kept.len(), p - 1, "{ctx}: the column drop");
            }
        }
    }
}

/// Gate 1, Type III against afex's column drop (fit-free): on an unbalanced
/// `y ~ a * b + z` (`a` 2 levels, `b` 3, `z` numeric, cells of 1 to 4 rows),
/// each term's `X·N` spans the same columns as the sum-coded design without
/// the term's columns, the reduced design afex fits under `contr.sum`, at
/// [`PROJ_ABS`] on every term: the null basis comes from the unrounded
/// hypothesis ([`type3_unrounded`]). From `type3`'s, which rounds `L` with
/// `zapsmall` as lmerTest does (`a`'s weights 1/3 become 0.3333333), `a`'s
/// span would move by about 5e-8.
#[test]
fn type3_reduced_design_spans_the_sum_coded_drop() {
    let mut csv = String::from("y,a,b,z\n");
    let mut cells: Vec<(usize, usize, f64)> = Vec::new();
    let mut i = 0usize;
    for (cell, &m) in [2usize, 3, 1, 4, 2, 3].iter().enumerate() {
        let (a, b) = (cell / 3, cell % 3);
        for _ in 0..m {
            i += 1;
            let z = ((i * 7) % 11) as f64 / 3.0;
            let y = ((i * 5) % 13) as f64;
            csv.push_str(&format!(
                "{y},{},{},{z}\n",
                ["p", "q"][a],
                ["u", "v", "w"][b]
            ));
            cells.push((a, b, z));
        }
    }
    let lo = lower(
        "y ~ a * b + z",
        &csv_table(&csv, &["a", "b"]),
        Family::Gaussian,
    )
    .unwrap();
    let (n, p) = (lo.n, lo.p);
    assert_eq!(
        lo.col_names,
        ["(Intercept)", "aq", "bv", "bw", "z", "aq:bv", "aq:bw"]
    );
    let x = Mat::from_fn(n, p, |i, j| lo.x[i * p + j]);
    // contr.sum: the last level codes −1 on every column.
    let sum_a = |a: usize| [1.0, -1.0][a];
    let sum_b = |b: usize| [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]][b];
    let sum_cols = |term: &str, (a, b, z): (usize, usize, f64)| -> Vec<f64> {
        match term {
            "a" => vec![sum_a(a)],
            "b" => sum_b(b).to_vec(),
            "z" => vec![z],
            "a:b" => sum_b(b).iter().map(|v| sum_a(a) * v).collect(),
            other => panic!("term {other}"),
        }
    };
    let all: Vec<usize> = (0..p).collect();
    for (h, e) in type3(&lo, &all).into_iter().zip(type3_unrounded(&lo, &all)) {
        let nb = type3_null_basis(h.l.as_ref(), e.l.as_ref(), &[]).expect("a null basis");
        let xn = &x * &nb;
        let rows: Vec<Vec<f64>> = cells
            .iter()
            .map(|&cell| {
                let mut row = vec![1.0];
                for t in lo.terms.iter().filter(|t| t.label != h.term) {
                    row.extend(sum_cols(&t.label, cell));
                }
                row
            })
            .collect();
        let xs = Mat::from_fn(n, rows[0].len(), |i, j| rows[i][j]);
        assert_eq!(xs.ncols(), xn.ncols(), "{}: widths", h.term);
        let d = max_abs_diff(
            &projector(&xn.transpose().to_owned()),
            &projector(&xs.transpose().to_owned()),
        );
        assert!(d <= PROJ_ABS, "{}: column spaces differ by {d:e}", h.term);
    }
}

/// [`type3_null_basis`] refuses what it cannot check: a hypothesis with a NaN
/// entry (as [`type3`] gives on a design singular on the retained columns),
/// one too ill-conditioned for the bound (`diag(1, 1e-6)`: the eigenvalue
/// 1e-12 falls under the cut and `‖L·e₂‖ = 1e-6`), and an unrounded
/// hypothesis of another rank than the rounded one.
#[test]
fn type3_null_basis_refuses_what_it_cannot_check() {
    let nan = Mat::from_fn(1, 3, |_, j| [1.0, f64::NAN, 0.0][j]);
    let ill = Mat::from_fn(2, 2, |i, j| if i == j { [1.0, 1e-6][i] } else { 0.0 });
    let rank1 = Mat::from_fn(1, 2, |_, j| [1.0, 2.0][j]);
    let rank2 = Mat::from_fn(2, 2, |i, j| [[1.0, 2.0], [1.0, 3.0]][i][j]);
    for (l, exact) in [(&nan, &nan), (&ill, &ill), (&rank1, &rank2)] {
        assert_eq!(
            type3_null_basis(l.as_ref(), exact.as_ref(), &[]),
            Err(ReducedDesignError::NoNullBasis),
            "{exact:?}"
        );
    }
}

/// Gate 1, random slopes, on `y ~ g * h + x` with a slope on `x` (columns
/// `(Intercept)`, `gb`, `gc`, `hq`, `x`, `gb:hq`, `gc:hq`): `x`'s Type III `L`
/// touches the slope column and is refused; `g`'s leaves it a unit column of
/// `N`, and [`remap_slopes`] moves the slope to that column, past `g`'s
/// mixed columns, on the same variable. Type II likewise: `g`'s reduced set
/// drops `gb` and `gc` before `x` and the slope follows `x`; `x`'s reduced set
/// drops the slope column and is refused.
#[test]
fn reduced_designs_keep_or_refuse_slope_columns() {
    let lo = lowered(&anova_entry(CAR_BERN[0]));
    let all: Vec<usize> = (0..lo.p).collect();
    let x = column(&lo, "x");
    assert_eq!(x, 4);
    let slope = [ColumnId::try_from(x).unwrap()];
    let hyp = type3(&lo, &all);
    let exact = type3_unrounded(&lo, &all);
    let at = |term: &str| hyp.iter().position(|h| h.term == term).unwrap();

    assert_eq!(
        type3_null_basis(hyp[at("x")].l.as_ref(), exact[at("x")].l.as_ref(), &[x]),
        Err(ReducedDesignError::SlopeTouched)
    );
    let lg = &hyp[at("g")].l;
    let n = type3_null_basis(lg.as_ref(), exact[at("g")].l.as_ref(), &[x])
        .expect("g's L leaves x alone");
    let kept = untouched_columns(lg.as_ref());
    let j = remap_slopes(&slope, &kept).expect("x is kept")[0] as usize;
    assert!(j < x, "the slope moves left past g's columns: {j}");
    assert_eq!(lo.col_names[kept[j]], "x");
    for i in 0..lo.p {
        assert_eq!(n[(i, j)], f64::from(u8::from(i == x)), "N[{i}, {j}]");
    }

    let sets = type2_lr_sets(&lo, &all);
    let (full, reduced) = &sets[0];
    assert_eq!(remap_slopes(&slope, full), Ok(vec![4]));
    assert_eq!(remap_slopes(&slope, reduced), Ok(vec![2]));
    assert_eq!(lo.col_names[reduced[2]], "x");
    assert_eq!(
        remap_slopes(&slope, &sets[2].1),
        Err(ReducedDesignError::SlopeTouched)
    );
}

/// Gate 1, Type II sets by hand: on `y ~ g * h + x` car's marginality tests
/// `g` and `h` in `g + h + x` (their relative `g:h` left out), and `x` and
/// `g:h` in the full model; an aliased column leaves every set.
#[test]
fn type2_lr_sets_follow_car_marginality() {
    let lo = lowered(&anova_entry(CAR_BERN[0]));
    assert_eq!(
        lo.col_names,
        ["(Intercept)", "gb", "gc", "hq", "x", "gb:hq", "gc:hq"]
    );
    let all: Vec<usize> = (0..lo.p).collect();
    let want: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![0, 1, 2, 3, 4], vec![0, 3, 4]),
        (vec![0, 1, 2, 3, 4], vec![0, 1, 2, 4]),
        (vec![0, 1, 2, 3, 4, 5, 6], vec![0, 1, 2, 3, 5, 6]),
        (vec![0, 1, 2, 3, 4, 5, 6], vec![0, 1, 2, 3, 4]),
    ];
    assert_eq!(type2_lr_sets(&lo, &all), want);
    let retained = [0, 1, 2, 3, 4, 5];
    let want: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![0, 1, 2, 3, 4], vec![0, 3, 4]),
        (vec![0, 1, 2, 3, 4], vec![0, 1, 2, 4]),
        (vec![0, 1, 2, 3, 4, 5], vec![0, 1, 2, 3, 5]),
        (vec![0, 1, 2, 3, 4, 5], vec![0, 1, 2, 3, 4]),
    ];
    assert_eq!(type2_lr_sets(&lo, &retained), want);
}

/// Gate 1, Type II sets follow car's containment, not lmerTest's: on
/// `y ~ x * g` (`x` numeric, `g` a 3-level factor: the `time * group` shape)
/// `x:g` contains `g` under car's rule, which counts every variable, so `g`
/// is tested in `x + g` against `x`. Under lmerTest's numeric-aware rule
/// ([`type2_xtx`]) `x:g` does not contain `g`, which would test `g` in the
/// full model.
#[test]
fn type2_lr_sets_follow_car_containment_on_numeric_time() {
    let e = anova_entries()
        .into_iter()
        .find(|e| e.name == "lmertest_anova_sim_factorial_gx")
        .expect("the g * x anova entry is registered");
    let lo = lower("y ~ x * g + (1 | subj)", &table_of(&e), Family::Gaussian).unwrap();
    assert_eq!(
        lo.col_names,
        ["(Intercept)", "x", "gb", "gc", "x:gb", "x:gc"]
    );
    let labels: Vec<&str> = lo.terms.iter().map(|t| t.label.as_str()).collect();
    assert_eq!(labels, ["x", "g", "x:g"]);
    let all: Vec<usize> = (0..lo.p).collect();
    let want: Vec<(Vec<usize>, Vec<usize>)> = vec![
        (vec![0, 1, 2, 3], vec![0, 2, 3]),
        (vec![0, 1, 2, 3], vec![0, 1]),
        (vec![0, 1, 2, 3, 4, 5], vec![0, 1, 2, 3]),
    ];
    assert_eq!(type2_lr_sets(&lo, &all), want);
}
