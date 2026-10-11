//! Gate 2: the contest layer against lmerTest's own outputs, fed lmerTest's
//! frozen inputs (`β`, `vcov_beta`, `vcov_varpar`, `Jac_list` and the contrast
//! matrices) from the `lmertest` references under
//! `validation/goldens/inference/lmertest/`.

use super::{contest_1d, contest_md, fstat_ddf, Contest1D, ContestInputs};
use crate::inference::test_refs::{entries, load, Kind};
use faer::Mat;
use serde_json::Value;

/// `contest_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`, fit-free):
/// `contest_1d`'s estimate, standard error, df and t, and the `fstat_ddf` case
/// table. Change together.
const CONTEST_REL: f64 = 1e-12;
/// `contest_p_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`, fit-free):
/// `contest_1d`'s p, which carries the t and df errors through the tail. A
/// frozen p of exactly 0 is compared exactly. Change together.
const CONTEST_P_REL: f64 = 8e-12;
/// `contest_md_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`, fit-free):
/// `contest_md`'s F, DenDF, p and `ν_m`. Change together.
const CONTEST_MD_REL: f64 = 1e-10;
/// The smallest relative eigen-gap of `L C Lᵀ` a frozen multi-direction case
/// may have; the band assumes it.
const MIN_REL_GAP: f64 = 1e-4;

fn num(v: &Value, what: &str) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("{what}: not a number: {v}"))
}

fn vector(v: &Value, what: &str) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("{what}: not an array: {v}"))
        .iter()
        .map(|x| num(x, what))
        .collect()
}

/// A matrix written by jsonlite: an array of rows.
fn matrix(v: &Value, what: &str) -> Mat<f64> {
    let rows: Vec<Vec<f64>> = v
        .as_array()
        .unwrap_or_else(|| panic!("{what}: not an array of rows: {v}"))
        .iter()
        .map(|r| vector(r, what))
        .collect();
    let ncols = rows.first().map_or(0, Vec::len);
    assert!(rows.iter().all(|r| r.len() == ncols), "{what}: ragged rows");
    Mat::from_fn(rows.len(), ncols, |i, j| rows[i][j])
}

/// The Satterthwaite inputs of one reference JSON, owned.
struct Inputs {
    beta: Vec<f64>,
    vcov_beta: Mat<f64>,
    vcov_varpar: Mat<f64>,
    jac: Vec<Mat<f64>>,
}

impl Inputs {
    fn read(json: &Value, name: &str) -> Inputs {
        let jac = json["Jac_list"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: no Jac_list"))
            .iter()
            .map(|m| matrix(m, name))
            .collect();
        Inputs {
            beta: vector(&json["beta"], name),
            vcov_beta: matrix(&json["vcov_beta"], name),
            vcov_varpar: matrix(&json["vcov_varpar"], name),
            jac,
        }
    }

    fn view(&self) -> ContestInputs<'_> {
        ContestInputs {
            beta: &self.beta,
            vcov_beta: self.vcov_beta.as_ref(),
            vcov_varpar: self.vcov_varpar.as_ref(),
            jac: &self.jac,
        }
    }
}

/// `|got − want| ≤ band·|want|`; equal values pass. A frozen 0 (a p that
/// underflowed) therefore passes only on an exact 0.
fn assert_rel(got: f64, want: f64, band: f64, what: &str) {
    let err = (got - want).abs();
    assert!(
        got == want || err <= band * want.abs(),
        "{what}: got {got:e}, lmerTest {want:e}, relative error {:e} > {band:e}",
        err / want.abs()
    );
}

/// The per-coefficient lmertest references (manifest
/// `proposal_dependent: false`), read whole with their manifest entry. The
/// per-term references (`proposal_dependent: true`) carry another layout and
/// are read only by name.
fn per_coefficient_references() -> Vec<(String, bool, Value)> {
    let refs: Vec<_> = entries(Kind::Lmertest)
        .into_iter()
        .filter(|e| {
            let flag = e.raw["proposal_dependent"].as_bool();
            assert!(
                flag.is_some(),
                "{}: proposal_dependent is not a bool",
                e.name
            );
            flag == Some(false)
        })
        .map(|e| {
            assert!(!e.is_pending(), "{}: not frozen", e.name);
            let reml = e.raw["reml"]
                .as_bool()
                .unwrap_or_else(|| panic!("{}: reml is not a bool", e.name));
            let json = load(Kind::Lmertest, &e.name);
            (e.name, reml, json)
        })
        .collect();
    assert!(!refs.is_empty(), "no per-coefficient lmertest reference");
    refs
}

/// `json[field]`, which the selected reference must carry.
fn field<'a>(json: &'a Value, field: &str, name: &str) -> &'a Value {
    json.get(field)
        .unwrap_or_else(|| panic!("{name}: missing `{field}`"))
}

#[test]
fn fstat_ddf_case_table() {
    // The case table is frozen once, on sleepstudy.
    let name = "lmertest_sleepstudy";
    let json = load(Kind::Lmertest, name);
    let cases = field(&json, "fstat_ddf_cases", name)
        .as_array()
        .unwrap_or_else(|| panic!("{name}: fstat_ddf_cases is not an array"));
    for case in cases {
        let what = format!("{name} fstat_ddf {}", case["case"]);
        let nu = vector(&case["nu"], &what);
        let got = fstat_ddf(&nu, num(&case["tol"], &what));
        assert_rel(got, num(&case["ddf"], &what), CONTEST_REL, &what);
    }
    assert_eq!(cases.len(), 5, "{name}: the five lmerTest cases");
}

/// `contest_1d`'s five outputs against the frozen fields of `want`.
fn check_1d(got: Contest1D, want: &Value, index: Option<usize>, what: &str) {
    for (name, value, band) in [
        ("estimate", got.estimate, CONTEST_REL),
        ("std_error", got.std_error, CONTEST_REL),
        ("df", got.df, CONTEST_REL),
        ("t_value", got.t_value, CONTEST_REL),
        ("p_value", got.p_value, CONTEST_P_REL),
    ] {
        let what = format!("{what} {name}");
        let frozen = field(want, name, &what);
        let frozen = index.map_or(frozen, |i| &frozen[i]);
        assert_rel(value, num(frozen, &what), band, &what);
    }
}

#[test]
fn contest_1d_matches_lmertest_summary() {
    for (name, reml, json) in per_coefficient_references() {
        let inputs = Inputs::read(&json, &name);
        let p = inputs.beta.len();
        // summary(): contest1D on each unit vector.
        for i in 0..p {
            let mut e = vec![0.0; p];
            e[i] = 1.0;
            let got = contest_1d(&inputs.view(), &e);
            check_1d(got, &json, Some(i), &format!("{name} coefficient {i}"));
        }
        // The seeded l, frozen on every REML reference and on no ML one.
        if reml {
            let case = field(&json, "contest_1d", &name);
            let l = vector(field(case, "l", &name), &name);
            let got = contest_1d(&inputs.view(), &l);
            check_1d(got, case, None, &format!("{name} contest_1d"));
        } else {
            assert!(
                json.get("contest_1d").is_none(),
                "{name}: an ML reference carries contest_1d"
            );
        }
    }
}

/// One frozen `contestMD` case: `L`, F, NumDF, DenDF, p and `ν_m`.
fn check_md_case(inputs: &Inputs, case: &Value, what: &str) {
    check_md(inputs, &matrix(&case["L"], what), case, what);
}

/// [`check_md_case`] with an `L` the caller built in place of the frozen one.
fn check_md(inputs: &Inputs, l: &Mat<f64>, case: &Value, what: &str) {
    let got = contest_md(&inputs.view(), l.as_ref()).unwrap_or_else(|e| panic!("{what}: {e:?}"));
    let num_df = case["NumDF"]
        .as_u64()
        .unwrap_or_else(|| panic!("{what}: NumDF"));
    assert_eq!(got.num_df as u64, num_df, "{what}: NumDF");
    assert_eq!(got.rows, l.nrows(), "{what}: rows");
    // One retained direction has no eigen-gap, and the frozen case no
    // `min_rel_gap`.
    if num_df >= 2 {
        let gap = num(&case["min_rel_gap"], what);
        assert!(gap >= MIN_REL_GAP, "{what}: frozen with gap {gap:e}");
    }
    assert_rel(
        got.f_value,
        num(&case["F"], what),
        CONTEST_MD_REL,
        &format!("{what} F"),
    );
    assert_rel(
        got.den_df,
        num(&case["DenDF"], what),
        CONTEST_MD_REL,
        &format!("{what} DenDF"),
    );
    assert_rel(
        got.p_value,
        num(&case["p_value"], what),
        CONTEST_MD_REL,
        &format!("{what} p"),
    );
    let nu = vector(&case["nu_m"], what);
    assert_eq!(got.nu.len(), nu.len(), "{what}: nu_m length");
    for (m, (&g, &w)) in got.nu.iter().zip(&nu).enumerate() {
        assert_rel(g, w, CONTEST_MD_REL, &format!("{what} nu_m[{m}]"));
    }
}

#[test]
fn contest_md_matches_lmertest() {
    // Frozen on every REML reference with p >= 2 and on no other.
    let mut unfrozen = 0;
    for (name, reml, json) in per_coefficient_references() {
        let inputs = Inputs::read(&json, &name);
        if !reml || inputs.beta.len() < 2 {
            assert!(
                json.get("contest_md").is_none(),
                "{name}: contest_md on an ML or p = 1 reference"
            );
            continue;
        }
        let cases = field(&json, "contest_md", &name)
            .as_array()
            .unwrap_or_else(|| panic!("{name}: contest_md is not an array"));
        assert!(!cases.is_empty(), "{name}: no contest_md case");
        for case in cases {
            let what = format!("{name} contest_md {}", case["label"]);
            let frozen = field(case, "frozen", &what)
                .as_bool()
                .unwrap_or_else(|| panic!("{what}: frozen is not a bool"));
            if !frozen {
                // Only the eigen-gap rule leaves a case unfrozen, with a reason.
                let gap = num(field(case, "min_rel_gap", &what), &what);
                assert!(gap < MIN_REL_GAP, "{what}: unfrozen at gap {gap:e}");
                assert!(
                    field(case, "reason", &what).is_string(),
                    "{what}: reason is not a string"
                );
                unfrozen += 1;
                continue;
            }
            check_md_case(&inputs, case, &what);
        }
    }
    // One today: lmertest_cake's identity L (gap 3.9e-7).
    assert_eq!(unfrozen, 1, "unfrozen contest_md cases");
    // The basis pair of the per-term reference: lmerTest's Type III L of `g`
    // and an invertible mix of its rows, on its twin's inputs.
    let name = "lmertest_anova_sim_factorial";
    let json = load(Kind::Lmertest, name);
    let pair = field(&json, "basis_pair", name);
    let twin = field(&json, "twin", name)
        .as_str()
        .unwrap_or_else(|| panic!("{name}: twin is not a string"));
    let inputs = Inputs::read(&load(Kind::Lmertest, twin), twin);
    for label in ["lmertest", "mixed"] {
        let what = format!("{name} basis_pair {label}");
        check_md_case(&inputs, field(pair, label, &what), &what);
    }
}

/// The basis pair (a Type III `L` and an invertible mix `M·L` of its rows,
/// which leaves F unchanged and moves contestMD's DenDF, since that depends on
/// the basis of `L`) through the crate's own Type III builder
/// (`formula::hypotheses::type3`), fed to the gate above on the twin's
/// frozen inputs: the crate's `L` of the pair's term reproduces lmerTest's
/// DenDF, and its rows mixed by the frozen `M` the mixed basis's own DenDF,
/// with the same F. So the builder gives lmerTest's basis, not only its row
/// space.
#[cfg(feature = "formula")]
#[test]
fn basis_pair_ddf() {
    use crate::formula::hypotheses::type3;
    use crate::formula::hypotheses_tests::table_of;
    let name = "lmertest_anova_sim_factorial";
    let entry = entries(Kind::Lmertest)
        .into_iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("{name} is registered"));
    let json = load(Kind::Lmertest, name);
    let pair = field(&json, "basis_pair", name);
    let term = field(pair, "term", name).as_str().expect("term");
    let twin = field(&json, "twin", name).as_str().expect("twin");
    let inputs = Inputs::read(&load(Kind::Lmertest, twin), twin);
    let formula = entry.r_formula.as_deref().expect("formula");
    let lo = crate::formula::lower(formula, &table_of(&entry), crate::Family::Gaussian)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(
        lo.p,
        inputs.beta.len(),
        "{name}: full rank, every column retained"
    );
    let all: Vec<usize> = (0..lo.p).collect();
    let l = type3(&lo, &all)
        .into_iter()
        .find(|h| h.term == term)
        .unwrap_or_else(|| panic!("{name}: no Type III term {term}"))
        .l;
    let m = matrix(field(pair, "M", name), name);
    assert_eq!((m.nrows(), m.ncols()), (l.nrows(), l.nrows()), "{name}: M");
    let mixed = Mat::from_fn(l.nrows(), l.ncols(), |i, j| {
        (0..l.nrows()).map(|k| m[(i, k)] * l[(k, j)]).sum()
    });
    for (label, l) in [("lmertest", &l), ("mixed", &mixed)] {
        let what = format!("{name} basis_pair {label}, the crate's L");
        check_md(&inputs, l, field(pair, label, &what), &what);
    }
}

#[test]
fn contest_1d_scaled_l_same_df() {
    let json = load(Kind::Lmertest, "lmertest_sleepstudy");
    let inputs = Inputs::read(&json, "lmertest_sleepstudy");
    let l = vector(&json["contest_1d"]["l"], "l");
    let scaled: Vec<f64> = l.iter().map(|x| 1e3 * x).collect();
    let base = contest_1d(&inputs.view(), &l);
    let got = contest_1d(&inputs.view(), &scaled);
    assert_rel(got.df, base.df, CONTEST_REL, "df");
    assert_rel(got.t_value, base.t_value, CONTEST_REL, "t");
    assert_rel(got.p_value, base.p_value, CONTEST_REL, "p");
}
