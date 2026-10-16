#![cfg(feature = "formula")]
//! The two-fit comparison: the formula nesting check, the refusals of
//! `formula::anova::compare`, and the test each pair gets, each checked on
//! two `Model` sides.
//!
//! The nesting tests read formula strings only: no data, no column names, no
//! tolerance. The refusal tests build each side from a small fit and then
//! change the one field a refusal reads, since `compare` reads numbers only.
//! The case tests fit both sides on one data set; a few then change a
//! side's numbers (a variance, a log-likelihood) to reach a rule no fit hits
//! on its own. Reference values frozen from R are not read here.
//!
//! The per-term tests of one fit (`term_tests`) are checked at the end, on
//! the `sim_factorial` designs: the rows, their tests by family, and their
//! agreement with the fit's own `summary()` statistics and with the two-fit
//! comparison.

use std::mem::discriminant;

use glmm::formula::anova::{
    compare, nesting, term_tests, CompareError, CompareOptions, Comparison, ComparisonTest,
    Differs, Model, Nesting, Side, SsType, TermTable, TermTest, TermTestError, TermTestOptions,
};
use glmm::formula::{lower, Column, Lowered, Table};
use glmm::inference::contrast::NoDfReason;
use glmm::inference::{
    DenDf, LinearKind, LinearNote, LinearTest, LrNote, LrTest, Objective, Reference,
};
use glmm::{
    fit_cold, BinomialLink, Family, Fit, GammaLink, InverseGaussianLink, Note, PoissonLink, WaldSe,
};

// ── The formula nesting check ───────────────────────────────────────────────

/// The answer for `f0` against `f1`, after checking that the two formulas
/// swapped give the mirrored answer (the other side is then the larger).
fn nest(f0: &str, f1: &str) -> Nesting {
    let forward = nesting(f0, f1).unwrap_or_else(|e| panic!("{f0} against {f1}: {e}"));
    let mirrored = match forward {
        Nesting::Nested { larger, differs } => Nesting::Nested {
            larger: match larger {
                Side::First => Side::Second,
                Side::Second => Side::First,
            },
            differs,
        },
        Nesting::Same => Nesting::Same,
    };
    assert_eq!(nesting(f1, f0).unwrap(), mirrored, "{f1} against {f0}");
    forward
}

/// The refusal for `f0` against `f1`, after checking that the swapped pair is
/// refused the same way.
fn refused(f0: &str, f1: &str) -> CompareError {
    let forward = nesting(f0, f1).expect_err(&format!("{f0} against {f1}"));
    let back = nesting(f1, f0).expect_err(&format!("{f1} against {f0}"));
    assert_eq!(discriminant(&forward), discriminant(&back));
    forward
}

fn nested(larger: Side, differs: Differs) -> Nesting {
    Nesting::Nested { larger, differs }
}

#[test]
fn a_smaller_fixed_part_is_nested() {
    assert_eq!(
        nest("y ~ x + (1 | g)", "y ~ x + z + (1 | g)"),
        nested(Side::Second, Differs::Fixed)
    );
    assert_eq!(
        nest("y ~ a * b", "y ~ a + b"),
        nested(Side::First, Differs::Fixed)
    );
}

#[test]
fn a_smaller_random_part_is_nested() {
    assert_eq!(
        nest("y ~ x + (1 | g)", "y ~ x + (1 + x | g)"),
        nested(Side::Second, Differs::Random)
    );
    assert_eq!(
        nest("y ~ x + (1 | g) + (1 | h)", "y ~ x + (1 | g)"),
        nested(Side::First, Differs::Random)
    );
    assert_eq!(
        nest("y ~ x", "y ~ x + (1 | g)"),
        nested(Side::Second, Differs::Random)
    );
}

#[test]
fn both_parts_smaller_is_nested() {
    assert_eq!(
        nest("y ~ x + (1 | g)", "y ~ x + z + (1 + x | g)"),
        nested(Side::Second, Differs::Both)
    );
    assert_eq!(
        nest("y ~ 1", "y ~ x + (1 | g)"),
        nested(Side::Second, Differs::Both)
    );
}

#[test]
fn order_does_not_matter() {
    // Terms, variables inside an interaction, groupings, slopes inside a
    // grouping: every reordering is the same formula.
    assert_eq!(nest("y ~ x + z + a:b", "y ~ b:a + z + x"), Nesting::Same);
    assert_eq!(
        nest("y ~ x + (1 | g) + (1 | h)", "y ~ x + (1 | h) + (1 | g)"),
        Nesting::Same
    );
    assert_eq!(
        nest("y ~ x + (1 + x + z | g)", "y ~ x + (1 + z + x | g)"),
        Nesting::Same
    );
    // The same orderings inside a nested pair.
    let want = nested(Side::Second, Differs::Both);
    for (f0, f1) in [
        (
            "y ~ x + a:b + (1 | g)",
            "y ~ a:b + z + x + (1 + x | g) + (1 | h)",
        ),
        (
            "y ~ b:a + x + (1 | g)",
            "y ~ x + z + a:b + (1 | h) + (x | g)",
        ),
        (
            "y ~ x + a:b + (1 | g)",
            "y ~ (1 + x | g) + b:a + x + (1 | h) + z",
        ),
    ] {
        assert_eq!(nest(f0, f1), want, "{f0} against {f1}");
    }
    assert_eq!(
        nest("y ~ x + (1 + x | g)", "y ~ x + (1 + z + x | g)"),
        nested(Side::Second, Differs::Random)
    );
}

#[test]
fn different_responses_refuse() {
    let e = refused("y ~ x", "w ~ x + z");
    assert!(matches!(e, CompareError::DifferentResponse { .. }), "{e:?}");
    assert!(e.to_string().contains("different responses"), "{e}");
}

#[test]
fn fewer_slopes_on_the_larger_side_refuses() {
    let e = refused("y ~ x + (1 + x | g)", "y ~ x + z + (1 | g)");
    assert!(matches!(e, CompareError::NotNested { .. }), "{e:?}");
    // A grouping only the smaller fixed part's model has.
    let e = refused("y ~ x + (1 | h)", "y ~ x + z + (1 | g)");
    assert!(matches!(e, CompareError::NotNested { .. }), "{e:?}");
    // Different slopes, neither set inside the other.
    let e = refused("y ~ x + z + (1 + x | g)", "y ~ x + z + (1 + z | g)");
    assert!(matches!(e, CompareError::NotNested { .. }), "{e:?}");
    // Different fixed terms.
    let e = refused("y ~ x", "y ~ z");
    assert!(matches!(e, CompareError::NotNested { .. }), "{e:?}");
    assert!(e.to_string().contains("nested"), "{e}");
}

#[test]
fn the_fixed_intercept_is_a_term() {
    assert_eq!(
        nest("y ~ 0 + x", "y ~ x"),
        nested(Side::Second, Differs::Fixed)
    );
    // Nested column spaces when `f` is a factor, but the formulas are not.
    let e = refused("y ~ 1", "y ~ 0 + f");
    assert!(matches!(e, CompareError::NotNested { .. }), "{e:?}");
    // Read as both parts differing, as written.
    assert_eq!(
        nest("y ~ 0 + f + (1 | g)", "y ~ f + (1 + x | g)"),
        nested(Side::Second, Differs::Both)
    );
}

#[test]
fn a_nested_grouping_contains_its_inner_grouping() {
    // The Pastes `ranova` row: dropping `(1 | batch)` from `(1 | batch/cask)`
    // leaves `(1 | cask:batch)`, the smaller model.
    assert_eq!(
        nest(
            "strength ~ 1 + (1 | batch/cask)",
            "strength ~ 1 + (1 | cask:batch)"
        ),
        nested(Side::First, Differs::Random)
    );
}

// ── The refusals of `compare` ───────────────────────────────────────────────

/// 30 rows: six groups `g` of five, a numeric `x` and `z`, a two-level `a`
/// and a three-level `b` (every pair of levels present), and a response with
/// a group effect.
fn table() -> Table {
    let n = 30;
    let (mut y, mut x, mut z) = (Vec::new(), Vec::new(), Vec::new());
    let (mut g, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let xi = ((i * 7) % 11) as f64 / 10.0;
        let zi = ((i * 5) % 13) as f64 / 6.0;
        let u = [0.9, -0.6, 0.3, -1.1, 0.7, -0.2][i / 5];
        let e = (((i * 37) % 17) as f64 - 8.0) / 9.0;
        y.push(2.0 + 0.8 * xi - 0.5 * zi + u + e);
        x.push(xi);
        z.push(zi);
        g.push(format!("g{}", i / 5));
        a.push(format!("a{}", i % 2));
        b.push(format!("b{}", i % 3));
    }
    Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("x".into(), Column::Numeric(x)),
            ("z".into(), Column::Numeric(z)),
            ("g".into(), Column::factor_from_labels(&g)),
            ("a".into(), Column::factor_from_labels(&a)),
            ("b".into(), Column::factor_from_labels(&b)),
        ],
        n,
    }
}

/// A Gaussian fit of `formula` with `reml`, as a `Model`.
fn model(formula: &str, reml: Option<bool>) -> Model {
    let mut lo = lower(formula, &table(), Family::Gaussian).unwrap();
    lo.opts.reml = reml;
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged(), "{formula} did not converge");
    Model::from_fit(formula, &lo, &fit)
}

fn opts() -> CompareOptions {
    CompareOptions {
        nsim: 1000,
        seed: 1,
    }
}

/// The statistic, p-value and reference of a comparison's test.
fn outcome(c: &Comparison) -> (u64, u64, Reference) {
    match &c.test {
        ComparisonTest::Lr(t) => (t.statistic.to_bits(), t.p_value.to_bits(), t.reference),
        ComparisonTest::Linear(t) => (t.statistic.to_bits(), t.p_value.to_bits(), t.reference),
        other => panic!("a test this file does not know: {other:?}"),
    }
}

/// `compare(a, b)`, after checking that `compare(b, a)` gives the same answer
/// (the sides mirrored, the same test) or the same refusal.
fn cmp(a: &Model, b: &Model) -> Result<Comparison, CompareError> {
    let forward = compare(a, b, &opts());
    let back = compare(b, a, &opts());
    match (&forward, &back) {
        (Ok(f), Ok(b)) => {
            assert_ne!(f.larger, b.larger, "the larger side follows its model");
            assert_eq!(f.differs, b.differs);
            assert_eq!((&f.m0, &f.m1), (&b.m1, &b.m0), "the sides as given");
            assert_eq!(outcome(f), outcome(b), "the same test either way round");
        }
        (Err(f), Err(b)) => assert_eq!(discriminant(f), discriminant(b), "{f} / {b}"),
        _ => panic!("swapping the sides changed the answer: {forward:?} / {back:?}"),
    }
    forward
}

/// Which side is the larger and which parts differ, as a [`Nesting`].
fn shape(c: Result<Comparison, CompareError>) -> Result<Nesting, CompareError> {
    c.map(|c| Nesting::Nested {
        larger: c.larger,
        differs: c.differs,
    })
}

fn refusal(a: &Model, b: &Model) -> CompareError {
    cmp(a, b).expect_err("refused")
}

#[test]
fn a_side_that_did_not_converge_refuses() {
    let m0 = model("y ~ x + (1 | g)", None);
    let mut m1 = model("y ~ x + z + (1 | g)", Some(false));
    let mut m0ml = model("y ~ x + (1 | g)", Some(false));
    m1.converged = false;
    let e = refusal(&m0ml, &m1);
    assert_eq!(
        e,
        CompareError::NotConverged {
            m0: false,
            m1: true
        }
    );
    assert!(
        e.to_string()
            .starts_with("the second model did not converge"),
        "{e}"
    );
    m0ml.converged = false;
    let e = refusal(&m0ml, &m1);
    assert!(e.to_string().starts_with("neither model converged"), "{e}");
    // Checked first: before the estimator check that this pair also fails.
    let e = refusal(&m0, &m1);
    assert_eq!(
        e,
        CompareError::NotConverged {
            m0: false,
            m1: true
        }
    );
}

#[test]
fn different_row_counts_refuse() {
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ x + z", None);
    m1.nobs -= 1;
    m1.y.pop();
    let e = refusal(&m0, &m1);
    assert_eq!(e, CompareError::DifferentRows { m0: 30, m1: 29 });
    assert_eq!(
        e.to_string(),
        "the first model has 30 rows and the second has 29; a comparison needs both fits \
         on the same rows. Fit both models to the same rows, dropping the rows with a \
         missing value in any variable of either formula first."
    );
}

#[test]
fn a_response_differing_in_one_entry_refuses() {
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ x + z", None);
    m1.y[17] = f64::from_bits(m1.y[17].to_bits() + 1);
    let e = refusal(&m0, &m1);
    assert_eq!(e, CompareError::DifferentY);
    assert_eq!(
        e.to_string(),
        "the two models have different responses y (the same row count, different \
         values); a comparison needs both fits on the same data. Fit both models to the \
         same data."
    );
}

#[test]
fn different_weights_refuse() {
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ x + z", None);
    let mut w = vec![1.0; 30];
    // All-ones weights are no weights.
    m1.weights = Some(w.clone());
    assert!(cmp(&m0, &m1).is_ok());
    w[4] = 2.0;
    m1.weights = Some(w);
    let e = refusal(&m0, &m1);
    assert_eq!(e, CompareError::DifferentWeights);
    assert_eq!(
        e.to_string(),
        "the two models were fitted with different weights; a comparison needs the same \
         weights in both. Fit both models with the same weights."
    );
}

#[test]
fn different_offsets_refuse() {
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ x + z", None);
    let mut o = vec![0.0; 30];
    m1.offset = Some(o.clone());
    assert!(cmp(&m0, &m1).is_ok());
    o[9] = 0.5;
    m1.offset = Some(o);
    let e = refusal(&m0, &m1);
    assert_eq!(e, CompareError::DifferentOffset);
    assert_eq!(
        e.to_string(),
        "the two models were fitted with different offsets; a comparison needs the same \
         offset in both. Fit both models with the same offset."
    );
}

#[test]
fn a_different_family_or_link_refuses() {
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ x + z", None);
    m1.family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let e = refusal(&m0, &m1);
    assert!(matches!(e, CompareError::DifferentFamily { .. }), "{e:?}");
    assert_eq!(
        e.to_string(),
        "the first model was fitted as gaussian with identity link and the second as \
         poisson with log link; a comparison needs the same family and link in both. Fit \
         both models with the same family and link."
    );
}

#[test]
fn a_different_held_dispersion_refuses() {
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ x + z", None);
    m1.dispersion_held = Some(2.0);
    let e = refusal(&m0, &m1);
    assert_eq!(
        e,
        CompareError::DifferentDispersionHeld {
            m0: None,
            m1: Some(2.0)
        }
    );
    assert!(e.to_string().contains("dispersion"), "{e}");
    let mut m0 = m0;
    m0.dispersion_held = Some(1.5);
    let e = refusal(&m0, &m1);
    assert!(
        matches!(e, CompareError::DifferentDispersionHeld { .. }),
        "{e:?}"
    );
    m0.dispersion_held = Some(2.0);
    assert!(cmp(&m0, &m1).is_ok());
}

#[test]
fn formulas_not_nested_refuse() {
    let e = refusal(&model("y ~ x", None), &model("y ~ z", None));
    assert!(matches!(e, CompareError::NotNested { .. }), "{e:?}");
    let e = refusal(
        &model("y ~ x + (1 + x | g)", None),
        &model("y ~ x + z + (1 | g)", None),
    );
    assert!(matches!(e, CompareError::NotNested { .. }), "{e:?}");
}

#[test]
fn one_reml_fit_and_one_ml_fit_refuse() {
    let reml = model("y ~ x + (1 | g)", None);
    let ml = model("y ~ x + z + (1 | g)", Some(false));
    let e = refusal(&reml, &ml);
    assert_eq!(e, CompareError::MixedEstimators { reml: Side::First });
    assert_eq!(
        e.to_string(),
        "the first model was fitted by REML and the second by ML; a comparison needs both \
         fitted the same way. Fit both models by maximum likelihood (ML), or both by REML \
         when their fixed parts are the same."
    );
    let e = refusal(&ml, &reml);
    assert!(
        e.to_string()
            .starts_with("the first model was fitted by ML and the second by REML;"),
        "{e}"
    );
}

#[test]
fn an_ols_fit_with_the_default_against_a_reml_mixed_fit_refuses() {
    let ols = model("y ~ x", None);
    let mixed = model("y ~ x + (1 | g)", None);
    let e = refusal(&ols, &mixed);
    assert_eq!(e, CompareError::MixedEstimators { reml: Side::Second });
    // The OLS fit made with REML passes the estimator check.
    assert!(cmp(&model("y ~ x", Some(true)), &mixed).is_ok());
}

#[test]
fn two_reml_fits_with_different_fixed_parts_refuse() {
    let m0 = model("y ~ x + (1 | g)", None);
    let m1 = model("y ~ x + z + (1 | g)", None);
    let e = refusal(&m0, &m1);
    assert_eq!(e, CompareError::RemlDifferentFixed);
    assert_eq!(
        e.to_string(),
        "the two models are REML fits with different fixed effects; the REML likelihood \
         depends on the fixed effects, so a comparison needs ML fits. Fit both models by \
         maximum likelihood (ML)."
    );
    // Both parts differing is a different fixed part too.
    let m1 = model("y ~ x + z + (1 + x | g)", None);
    assert_eq!(refusal(&m0, &m1), CompareError::RemlDifferentFixed);
    // The same fixed part passes.
    let m1 = model("y ~ x + (1 + x | g)", None);
    assert_eq!(
        shape(cmp(&m0, &m1)),
        Ok(nested(Side::Second, Differs::Random))
    );
}

#[test]
fn two_ols_fits_pass_the_estimator_check_whatever_their_reml() {
    let want = Ok(nested(Side::Second, Differs::Fixed));
    assert_eq!(
        shape(cmp(
            &model("y ~ x", Some(true)),
            &model("y ~ x + z", Some(true))
        )),
        want
    );
    assert_eq!(
        shape(cmp(&model("y ~ x", None), &model("y ~ x + z", Some(true)))),
        want
    );
    assert_eq!(
        shape(cmp(&model("y ~ x", None), &model("y ~ x + z", None))),
        want
    );
}

#[test]
fn the_same_ols_fixed_part_has_nothing_to_test() {
    let e = refusal(&model("y ~ x", None), &model("y ~ x", Some(true)));
    assert_eq!(e, CompareError::SameParts);
}

#[test]
fn the_same_parameter_count_has_nothing_to_test() {
    // With two factors, `a:b` spans the columns of `a * b`.
    let (m0, m1) = (model("y ~ a:b", None), model("y ~ a * b", None));
    assert_eq!(m0.df, m1.df);
    let e = refusal(&m0, &m1);
    assert_eq!(
        e,
        CompareError::NothingToTest {
            m0: m0.df,
            m1: m1.df
        }
    );
    assert_eq!(
        e.to_string(),
        format!(
            "the two models have the same number of parameters ({}), so there is nothing to \
             test: the larger formula adds no parameter to the smaller one. Compare two models \
             that differ by at least one parameter.",
            m0.df
        )
    );
    // With a numeric `x` they are different, nested models.
    assert_eq!(
        shape(cmp(&model("y ~ x:a", None), &model("y ~ x * a", None))),
        Ok(nested(Side::Second, Differs::Fixed))
    );
}

#[test]
fn the_data_checks_come_before_the_nesting_check() {
    // Not nested, and fitted on different data: the data refusal is reported.
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ z", None);
    m1.y[17] = f64::from_bits(m1.y[17].to_bits() + 1);
    assert_eq!(refusal(&m0, &m1), CompareError::DifferentY);
    m1.nobs -= 1;
    m1.y.pop();
    assert_eq!(
        refusal(&m0, &m1),
        CompareError::DifferentRows { m0: 30, m1: 29 }
    );
}

#[test]
fn one_formula_on_both_sides_has_the_same_parts_whatever_the_parameter_counts() {
    // The refusal names the same parts, not the parameter counts: two fits
    // of one formula have equal counts, and a side built by hand can carry
    // another count for the same formula.
    let m0 = model("y ~ x + (1 | g)", Some(false));
    let mut m1 = m0.clone();
    for _ in 0..2 {
        let e = refusal(&m0, &m1);
        assert_eq!(e, CompareError::SameParts);
        assert_eq!(
            e.to_string(),
            "the two models have the same fixed and random parts, so there is nothing to \
             test. Compare two models that differ by at least one parameter."
        );
        m1.df += 1;
    }
}

#[test]
fn formulas_that_differ_only_by_an_offset_refuse_by_the_offset() {
    // `nesting` does not read the offset, so the two formulas have the same
    // parts; the offset check comes first and names what differs.
    let (m0, m1) = (model("y ~ x + offset(z)", None), model("y ~ x", None));
    assert_eq!(nesting(&m0.formula, &m1.formula), Ok(Nesting::Same));
    assert_eq!(refusal(&m0, &m1), CompareError::DifferentOffset);
}

// ── The cases and their references ──────────────────────────────────────────

/// A CSV of the validation corpus as a `Table`: the columns named in
/// `factors` are factors, every other column is numeric.
fn csv_table(csv: &str, factors: &[&str]) -> Table {
    let split = |l: &str| -> Vec<String> {
        l.split(',')
            .map(|c| c.trim().trim_matches('"').to_string())
            .collect()
    };
    let mut lines = csv.lines().filter(|l| !l.trim().is_empty());
    let header = split(lines.next().expect("a CSV header"));
    let rows: Vec<Vec<String>> = lines.map(split).collect();
    let columns = header
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let cells: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
            let column = if factors.contains(&name.as_str()) {
                Column::factor_from_labels(&cells)
            } else {
                Column::Numeric(cells.iter().map(|c| c.parse().unwrap()).collect())
            };
            (name.clone(), column)
        })
        .collect();
    Table {
        columns,
        n: rows.len(),
    }
}

fn sleepstudy() -> Table {
    csv_table(
        include_str!("../validation/data/empirical/sleepstudy.csv"),
        &["Subject"],
    )
}

fn binomial() -> Family {
    Family::Binomial {
        link: BinomialLink::Logit,
    }
}

/// A converged fit of `formula` on `table` with `reml` and `nagq` nodes, and
/// its `Model`.
fn fitted(
    formula: &str,
    table: &Table,
    family: Family,
    reml: Option<bool>,
    nagq: u8,
) -> (Model, Fit) {
    let mut lo = lower(formula, table, family).unwrap();
    lo.opts.reml = reml;
    lo.opts.nagq = nagq;
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged(), "{formula} did not converge");
    (Model::from_fit(formula, &lo, &fit), fit)
}

fn side(formula: &str, table: &Table, family: Family, reml: Option<bool>) -> Model {
    fitted(formula, table, family, reml, 1).0
}

fn lr(c: &Comparison) -> &LrTest {
    match &c.test {
        ComparisonTest::Lr(t) => t,
        other => panic!("not a likelihood-ratio test: {other:?}"),
    }
}

fn linear(c: &Comparison) -> &LinearTest {
    match &c.test {
        ComparisonTest::Linear(t) => t,
        other => panic!("not an F test: {other:?}"),
    }
}

/// `max(0, 2(ℓ₁ − ℓ₀))` of a comparison's two sides.
fn floored(c: &Comparison) -> f64 {
    let (small, large) = match c.larger {
        Side::First => (&c.m1, &c.m0),
        Side::Second => (&c.m0, &c.m1),
    };
    (2.0 * (large.loglik - small.loglik)).max(0.0)
}

const MIXTURE: LrNote = LrNote::BoundaryMixture { approximate: false };

#[test]
fn a_random_slope_dropped_gets_the_mixture_on_one_and_two_df() {
    let d = sleepstudy();
    let m0 = side(
        "Reaction ~ Days + (1 | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let m1 = side(
        "Reaction ~ Days + (Days | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let c = cmp(&m0, &m1).unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 1 });
    assert_eq!(t.df, 2);
    assert_eq!(t.objective, Objective::Reml);
    assert_eq!(t.notes, vec![MIXTURE]);
    assert!(t.statistic > 0.0);
    assert_eq!(t.statistic, floored(&c));
    assert!(t.p_value > 0.0 && t.p_value < 1.0);
    assert_eq!(
        c.note_lines(),
        vec!["Conservative: the true p-value is usually smaller."]
    );
}

#[test]
fn random_parts_more_than_one_dimension_apart_get_chi_squared_ignoring_the_boundary() {
    let d = sleepstudy();
    let ols = side("Reaction ~ Days", &d, Family::Gaussian, Some(true));
    let full = side(
        "Reaction ~ Days + (Days | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let c = cmp(&ols, &full).unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiSquared { df: 3 });
    assert_eq!(t.notes, vec![LrNote::BoundaryIgnored]);
    assert!(c.note_lines()[0].contains("ignores the boundary; conservative"));
    assert_eq!(t.statistic, floored(&c));
}

#[test]
fn a_one_grouping_mixture_beside_another_grouping_is_approximate() {
    let d = csv_table(
        include_str!("../validation/data/empirical/Penicillin.csv"),
        &["plate", "sample"],
    );
    let m0 = side("diameter ~ 1 + (1 | plate)", &d, Family::Gaussian, None);
    let m1 = side(
        "diameter ~ 1 + (1 | plate) + (1 | sample)",
        &d,
        Family::Gaussian,
        None,
    );
    let c = cmp(&m0, &m1).unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(t.notes, vec![LrNote::BoundaryMixture { approximate: true }]);
    let lines = c.note_lines();
    assert_eq!(lines.len(), 2);
    assert!(lines[1].starts_with("Derived for one variance component at zero"));
}

#[test]
fn a_dropped_component_at_zero_gives_t_zero_and_p_one_on_every_reference() {
    // Exact reference: Dyestuff2's batch variance is estimated at zero.
    let d = csv_table(
        include_str!("../validation/data/empirical/Dyestuff2.csv"),
        &["Batch"],
    );
    let ols = side("Yield ~ 1", &d, Family::Gaussian, Some(true));
    let full = side("Yield ~ 1 + (1 | Batch)", &d, Family::Gaussian, None);
    assert_eq!(
        full.varcorr,
        vec![vec![0.0]],
        "Dyestuff2 pins the batch variance"
    );
    let c = cmp(&ols, &full).unwrap();
    let t = lr(&c);
    assert_eq!(t.statistic.to_bits(), 0.0f64.to_bits());
    assert_eq!(t.p_value, 1.0);
    // No draws are made at a zero statistic: the exact test returns `nsim = 0`.
    assert_eq!(
        t.reference,
        Reference::ExactRlrt {
            nsim: 0,
            seed: opts().seed
        }
    );

    // Mixture reference: the same pair given weights, all equal to 1, keeps
    // no design summary.
    let weighted = |f: &str, reml| {
        let mut lo = lower(f, &d, Family::Gaussian).unwrap();
        lo.opts.reml = reml;
        lo.opts.weights = Some(vec![1.0; lo.n]);
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        Model::from_fit(f, &lo, &fit)
    };
    let (ols_w, full_w) = (
        weighted("Yield ~ 1", Some(true)),
        weighted("Yield ~ 1 + (1 | Batch)", None),
    );
    assert_eq!(full_w.design_summary, None);
    let c = cmp(&ols_w, &full_w).unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(t.notes, vec![LrNote::NoDesignSummary, MIXTURE]);
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));

    // Mixture and χ² references, from a side whose dropped variances are set
    // to zero while its log-likelihood stays a round-off above the smaller
    // one's: the rule, not the round-off, decides.
    let s = sleepstudy();
    let int = side(
        "Reaction ~ Days + (1 | Subject)",
        &s,
        Family::Gaussian,
        None,
    );
    let ols = side("Reaction ~ Days", &s, Family::Gaussian, Some(true));
    let mut slope = side(
        "Reaction ~ Days + (Days | Subject)",
        &s,
        Family::Gaussian,
        None,
    );
    let v = slope.varcorr[0].clone();
    // The vech block is (intercept, covariance, slope).
    slope.varcorr = vec![vec![v[0], 0.0, 0.0]];
    slope.loglik = int.loglik + 1e-9;
    let t = lr(&cmp(&int, &slope).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 1 });
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));
    slope.varcorr = vec![vec![0.0, 0.0, 0.0]];
    slope.loglik = ols.loglik + 1e-9;
    let t = lr(&cmp(&ols, &slope).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiSquared { df: 3 });
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));
    // A zero variance the smaller model keeps does not trigger the rule.
    slope.varcorr = vec![vec![0.0, 0.0, v[2]]];
    slope.loglik = int.loglik + 1e-9;
    let t = lr(&cmp(&int, &slope).unwrap()).clone();
    assert!(t.statistic > 0.0 && t.p_value < 1.0, "{t:?}");
}

#[test]
fn a_fit_at_the_correlation_boundary_is_not_a_zero_component() {
    let d = csv_table(
        include_str!("../validation/data/simulated/sim_corr_boundary.csv"),
        &["g"],
    );
    let (full, fit) = fitted("y ~ x + (1 + x | g)", &d, Family::Gaussian, None, 1);
    // The trailing θ diagonal is pinned while the slope variance is not zero.
    assert!(fit.diagnostics.pinned.iter().flatten().any(|&p| p));
    assert!(full.varcorr[0][2] > 0.0, "{:?}", full.varcorr);
    let reduced = side("y ~ x + (1 | g)", &d, Family::Gaussian, None);
    let t = lr(&cmp(&reduced, &full).unwrap()).clone();
    assert!(t.statistic > 0.0, "{t:?}");
}

#[test]
fn a_raw_statistic_below_zero_is_floored_with_a_note_naming_the_larger_fit() {
    let d = sleepstudy();
    let small = side(
        "Reaction ~ Days + (1 | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let mut large = side(
        "Reaction ~ Days + (Days | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    // Beyond round-off (2·dev_eps = 8e-5 deviance units).
    large.loglik = small.loglik - 1e-4;
    let raw = 2.0 * (large.loglik - small.loglik);
    let c = cmp(&small, &large).unwrap();
    let t = lr(&c);
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));
    assert_eq!(t.notes, vec![LrNote::NegativeStatistic { raw }, MIXTURE]);
    let negative = "The larger model did not reach its optimum";
    assert!(c.note_lines()[0].starts_with(negative));
    let back = compare(&large, &small, &opts()).unwrap();
    assert!(back.note_lines()[0].starts_with(negative));
    // Within round-off: the floor, with no note.
    large.loglik = small.loglik - 1e-5;
    let t = lr(&cmp(&small, &large).unwrap()).clone();
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));
    assert_eq!(t.notes, vec![MIXTURE]);
}

/// Six groups with the same ten rows each, so no grouping varies: a
/// binomial and a count response on a numeric `x`.
fn identical_groups() -> Table {
    let (mut x, mut yb, mut yc, mut g) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for group in 0..6 {
        for k in 0..10 {
            x.push(k as f64 / 10.0);
            yb.push([0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0][k]);
            yc.push([0.0, 1.0, 1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0][k]);
            g.push(format!("g{group}"));
        }
    }
    Table {
        columns: vec![
            ("yb".into(), Column::Numeric(yb)),
            ("yc".into(), Column::Numeric(yc)),
            ("x".into(), Column::Numeric(x)),
            ("g".into(), Column::factor_from_labels(&g)),
        ],
        n: 60,
    }
}

#[test]
fn a_binomial_or_poisson_glmm_at_zero_variance_has_the_glm_log_likelihood() {
    let d = identical_groups();
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    for (y, family) in [("yb", binomial()), ("yc", poisson)] {
        let glm = side(&format!("{y} ~ x"), &d, family, None);
        let glmm = side(&format!("{y} ~ x + (1 | g)"), &d, family, None);
        assert_eq!(glmm.varcorr, vec![vec![0.0]], "{y}: θ̂ = 0");
        let gap = (glmm.loglik - glm.loglik).abs();
        assert!(
            gap <= 1e-10 * glm.loglik.abs(),
            "{y}: GLMM {} against GLM {}",
            glmm.loglik,
            glm.loglik
        );
        let t = lr(&cmp(&glm, &glmm).unwrap()).clone();
        assert_eq!((t.statistic, t.p_value), (0.0, 1.0), "{y}");
    }
}

#[test]
fn a_grouping_lowering_drops_does_not_hide_a_dropped_component_at_zero() {
    // `h` has one level, so lowering drops `(1 | h)`: the larger fit keeps
    // one block, `g`'s, at zero, whatever the formula's term count.
    let mut d = identical_groups();
    let only = vec!["only".to_string(); d.n];
    d.columns
        .push(("h".into(), Column::factor_from_labels(&only)));
    let glm = side("yb ~ x", &d, binomial(), None);
    for larger in ["yb ~ x + (1 | g)", "yb ~ x + (1 | g) + (1 | h)"] {
        let glmm = side(larger, &d, binomial(), None);
        assert_eq!(glmm.varcorr, vec![vec![0.0]], "{larger}");
        let t = lr(&cmp(&glm, &glmm).unwrap()).clone();
        assert_eq!(t.statistic.to_bits(), 0.0f64.to_bits(), "{larger}: {t:?}");
        assert_eq!(t.p_value, 1.0, "{larger}");
    }
}

#[test]
fn a_slope_dropped_beside_a_factor_slope_at_zero_gives_t_zero() {
    let mut d = sleepstudy();
    let phase: Vec<String> = (0..d.n)
        .map(|i| ["a", "b", "c"][i % 3].to_string())
        .collect();
    d.columns
        .push(("phase".into(), Column::factor_from_labels(&phase)));
    let small = side(
        "Reaction ~ Days + phase + (phase | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let mut large = side(
        "Reaction ~ Days + phase + (phase + Days | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    // One block of four dimensions, (Intercept), phaseb, phasec, Days, packed
    // by rows of its upper triangle: zero Days' variance and covariances.
    let q = 4;
    assert_eq!(large.varcorr[0].len(), q * (q + 1) / 2);
    for i in 0..q {
        large.varcorr[0][i * q - (i * i - i) / 2 + (q - 1 - i)] = 0.0;
    }
    large.loglik = small.loglik + 1e-9;
    let t = lr(&cmp(&small, &large).unwrap()).clone();
    assert_eq!(t.statistic.to_bits(), 0.0f64.to_bits(), "{t:?}");
    assert_eq!(t.p_value, 1.0);
}

#[test]
fn a_fixed_effect_beside_a_grouping_at_zero_is_still_tested() {
    // The fixed parts differ too, so the larger optimum is not in the smaller
    // model's space and the zero-component rule does not apply.
    let d = identical_groups();
    let glm = side("yb ~ 1", &d, binomial(), None);
    let glmm = side("yb ~ x + (1 | g)", &d, binomial(), None);
    assert_eq!(glmm.varcorr, vec![vec![0.0]]);
    let c = cmp(&glm, &glmm).unwrap();
    assert_eq!(c.differs, Differs::Both);
    let t = lr(&c);
    assert!(t.statistic > 0.0 && t.p_value < 1.0, "{t:?}");
    assert_eq!(t.statistic, floored(&c));
}

#[test]
fn a_failed_fit_gives_nan_even_with_the_dropped_variance_at_zero() {
    // Three pairs whose dropped variances are zero, one per reference: the
    // mixture, χ² and the exact test.
    let s = sleepstudy();
    let int = side(
        "Reaction ~ Days + (1 | Subject)",
        &s,
        Family::Gaussian,
        None,
    );
    let ols = side("Reaction ~ Days", &s, Family::Gaussian, Some(true));
    let mut slope = side(
        "Reaction ~ Days + (Days | Subject)",
        &s,
        Family::Gaussian,
        None,
    );
    let v = slope.varcorr[0][0];
    slope.varcorr = vec![vec![v, 0.0, 0.0]];
    slope.loglik = int.loglik + 1e-9;
    let mut flat = slope.clone();
    flat.varcorr = vec![vec![0.0, 0.0, 0.0]];
    flat.loglik = ols.loglik + 1e-9;
    let d2 = csv_table(
        include_str!("../validation/data/empirical/Dyestuff2.csv"),
        &["Batch"],
    );
    let ols2 = side("Yield ~ 1", &d2, Family::Gaussian, Some(true));
    let full2 = side("Yield ~ 1 + (1 | Batch)", &d2, Family::Gaussian, None);
    assert_eq!(full2.varcorr, vec![vec![0.0]]);
    // The p-quality notes each reference carries on a finite pair; a NaN
    // statistic replaces them with the one note naming its cause.
    let pairs = [
        (&int, &slope, vec![MIXTURE]),
        (&ols, &flat, vec![LrNote::BoundaryIgnored]),
        (&ols2, &full2, vec![]),
    ];
    for (small, large, finite_notes) in pairs {
        let finite = lr(&compare(small, large, &opts()).unwrap()).clone();
        assert_eq!((finite.statistic, finite.p_value), (0.0, 1.0), "{finite:?}");
        assert_eq!(finite.notes, finite_notes, "{}", large.formula);
        for bad in [f64::NEG_INFINITY, f64::NAN] {
            for on_larger in [true, false] {
                let (mut s, mut l) = (small.clone(), large.clone());
                if on_larger {
                    l.loglik = bad;
                } else {
                    s.loglik = bad;
                }
                let c = compare(&s, &l, &opts()).unwrap();
                let t = lr(&c).clone();
                let what = format!("{} at {bad}, larger {on_larger}", large.formula);
                assert!(t.statistic.is_nan(), "{what}: {t:?}");
                assert!(t.p_value.is_nan(), "{what}: {t:?}");
                assert_eq!(t.reference, finite.reference, "{what}");
                assert_eq!(t.notes, vec![LrNote::NonFiniteLoglik], "{what}");
                let side = if on_larger {
                    "the larger model"
                } else {
                    "the smaller model"
                };
                assert_eq!(
                    c.note_lines(),
                    vec![format!(
                        "The log-likelihood of {side} is not finite (a fit that failed or \
                         collapsed), so the statistic and its p-value are NaN."
                    )],
                    "{what}"
                );
            }
        }
    }
}

#[test]
fn a_nan_statistic_keeps_the_notes_on_the_pair() {
    // The notes that state a fact about the pair stay beside the one naming
    // the cause; the notes on the p-value ("Conservative ...") go.
    let d = csv_table(
        include_str!("../validation/data/empirical/Dyestuff2.csv"),
        &["Batch"],
    );
    let weighted = |f: &str, reml| {
        let mut lo = lower(f, &d, Family::Gaussian).unwrap();
        lo.opts.reml = reml;
        lo.opts.weights = Some(vec![1.0; lo.n]);
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        Model::from_fit(f, &lo, &fit)
    };
    let small = weighted("Yield ~ 1", Some(true));
    let mut large = weighted("Yield ~ 1 + (1 | Batch)", None);
    let s = slope1();
    let node = |f: &str, nagq| fitted(f, &s, binomial(), None, nagq).0;
    let (glmm1, mut glmm7) = (node("y ~ 1 + (1 | g)", 1), node("y ~ x + (1 | g)", 7));
    assert_eq!(
        lr(&compare(&small, &large, &opts()).unwrap()).notes,
        vec![LrNote::NoDesignSummary, MIXTURE]
    );
    large.loglik = f64::NAN;
    let c = compare(&small, &large, &opts()).unwrap();
    assert!(lr(&c).statistic.is_nan());
    assert_eq!(
        lr(&c).notes,
        vec![LrNote::NoDesignSummary, LrNote::NonFiniteLoglik]
    );
    let lines = c.note_lines();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(
        lines.iter().all(|l| !l.starts_with("Conservative")),
        "{lines:?}"
    );
    glmm7.loglik = f64::NEG_INFINITY;
    let c = compare(&glmm1, &glmm7, &opts()).unwrap();
    assert!(lr(&c).statistic.is_nan());
    assert_eq!(
        lr(&c).notes,
        vec![LrNote::NodeCounts { m0: 1, m1: 7 }, LrNote::NonFiniteLoglik]
    );
}

#[test]
fn an_ml_pair_inside_the_exact_tests_envelope_gets_the_mixture() {
    // The exact test is a REML test: two ML fits of the same pair, the larger
    // keeping its design summary, get the mixture.
    let d = csv_table(
        include_str!("../validation/data/empirical/Dyestuff.csv"),
        &["Batch"],
    );
    let ols = side("Yield ~ 1", &d, Family::Gaussian, Some(false));
    let full = side("Yield ~ 1 + (1 | Batch)", &d, Family::Gaussian, Some(false));
    assert!(full.design_summary.is_some() && !full.reml && !ols.reml);
    let t = lr(&cmp(&ols, &full).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(t.objective, Objective::MaximumLikelihood);
    assert_eq!(t.notes, vec![MIXTURE]);
}

/// Sleepstudy with a crossed grouping `blk`, the row index mod 6.
fn sleepstudy_blk() -> Table {
    let mut d = sleepstudy();
    let blk: Vec<String> = (0..d.n).map(|i| format!("b{}", i % 6)).collect();
    d.columns
        .push(("blk".into(), Column::factor_from_labels(&blk)));
    d
}

/// The position of grouping `name` among a model's `varcorr` blocks.
fn block(m: &Model, name: &str) -> usize {
    m.re_groups.iter().position(|(n, _)| n == name).unwrap()
}

#[test]
fn one_dropped_variance_at_zero_beside_a_non_zero_one_keeps_the_statistic() {
    // The larger fit puts `blk` at exactly 0.0 but Subject's Days variance
    // well above it: the rule needs every dropped variance at zero, so the
    // real effect stays (T near 42.8, p near 2.7e-9).
    let d = sleepstudy_blk();
    let small = side(
        "Reaction ~ Days + (1 | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let large = side(
        "Reaction ~ Days + (Days | Subject) + (1 | blk)",
        &d,
        Family::Gaussian,
        None,
    );
    assert_eq!(large.varcorr[block(&large, "blk")], vec![0.0]);
    assert!(large.varcorr[block(&large, "Subject")][2] > 1.0);
    let c = cmp(&small, &large).unwrap();
    let t = lr(&c);
    assert_eq!(t.statistic, floored(&c));
    assert!(t.statistic > 40.0 && t.p_value < 1e-8, "{t:?}");
}

#[test]
fn a_term_kept_in_another_grouping_is_still_dropped_from_its_own() {
    // Days is dropped from Subject and kept in blk: only Subject's Days
    // variance decides the rule.
    let d = sleepstudy_blk();
    let small = side(
        "Reaction ~ Days + (1 | Subject) + (Days | blk)",
        &d,
        Family::Gaussian,
        None,
    );
    let large = side(
        "Reaction ~ Days + (Days | Subject) + (Days | blk)",
        &d,
        Family::Gaussian,
        None,
    );
    let (s, b) = (block(&large, "Subject"), block(&large, "blk"));
    for (subject, blk, zero) in [
        (vec![600.0, 0.0, 0.0], vec![1.0, 0.2, 3.0], true),
        (vec![600.0, 10.0, 35.0], vec![1.0, 0.0, 0.0], false),
    ] {
        let mut l = large.clone();
        l.varcorr[s] = subject;
        l.varcorr[b] = blk;
        l.loglik = small.loglik + 1e-9;
        let t = lr(&cmp(&small, &l).unwrap()).clone();
        assert_eq!(t.statistic == 0.0, zero, "{:?}: {t:?}", l.varcorr);
        assert!(t.statistic < 1e-8, "{t:?}");
    }
}

#[test]
fn groupings_are_matched_as_sets_of_variable_names() {
    // `(1 | batch/cask)` holds `batch` and `batch:cask`, the smaller fit's
    // `cask:batch`: dropping `batch` at zero gives T = 0.
    let d = csv_table(
        include_str!("../validation/data/empirical/Pastes.csv"),
        &["batch", "cask", "sample"],
    );
    let small = side(
        "strength ~ 1 + (1 | cask:batch)",
        &d,
        Family::Gaussian,
        None,
    );
    let large = side(
        "strength ~ 1 + (1 | batch/cask)",
        &d,
        Family::Gaussian,
        None,
    );
    let b = block(&large, "batch");
    for (variance, zero) in [(0.0, true), (0.5, false)] {
        let mut l = large.clone();
        l.varcorr[b] = vec![variance];
        l.loglik = small.loglik + 1e-9;
        let t = lr(&cmp(&small, &l).unwrap()).clone();
        assert_eq!(t.statistic == 0.0, zero, "batch at {variance}: {t:?}");
        assert_eq!(t.p_value == 1.0, zero, "batch at {variance}: {t:?}");
    }
}

#[test]
fn a_variance_at_zero_in_the_smaller_fit_makes_the_mixture_approximate() {
    let s = sleepstudy();
    let mut int = side(
        "Reaction ~ Days + (1 | Subject)",
        &s,
        Family::Gaussian,
        None,
    );
    let slope = side(
        "Reaction ~ Days + (Days | Subject)",
        &s,
        Family::Gaussian,
        None,
    );
    int.varcorr = vec![vec![0.0]];
    let t = lr(&cmp(&int, &slope).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 1 });
    assert_eq!(t.notes, vec![LrNote::BoundaryMixture { approximate: true }]);
}

#[test]
fn each_f_test_note_renders_its_numbers() {
    let mut c = cmp(&model("y ~ x", Some(true)), &model("y ~ x + z", Some(true))).unwrap();
    let ComparisonTest::Linear(t) = &mut c.test else {
        panic!("not an F test: {c:?}")
    };
    t.notes = vec![
        LinearNote::AliasedColumn { column: 2 },
        LinearNote::RankDeficientL { rank: 1, rows: 2 },
    ];
    let lines = c.note_lines();
    assert_eq!(lines.len(), 2);
    // Named by the larger fit's coefficient name, never an index.
    assert_eq!(
        lines[0],
        "The coefficient z is aliased, so the hypothesis was not tested."
    );
    assert!(
        lines[1].contains("The 2 tested coefficients span 1"),
        "{}",
        lines[1]
    );
}

#[test]
fn two_gaussian_ml_fits_get_chi_squared_or_the_boundary_rule_with_aic_and_bic() {
    let mut d = sleepstudy();
    let days = match &d.columns.iter().find(|(n, _)| n == "Days").unwrap().1 {
        Column::Numeric(v) => v.iter().map(|x| x * x).collect(),
        _ => unreachable!("Days is numeric"),
    };
    d.columns.push(("Days2".into(), Column::Numeric(days)));
    let ml = |f: &str| side(f, &d, Family::Gaussian, Some(false));
    // The fixed part differs.
    let c = cmp(
        &ml("Reaction ~ Days + (Days | Subject)"),
        &ml("Reaction ~ Days + Days2 + (Days | Subject)"),
    )
    .unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiSquared { df: 1 });
    assert_eq!(t.objective, Objective::MaximumLikelihood);
    assert!(t.notes.is_empty());
    for info in [&c.m0, &c.m1] {
        let k = info.npar as f64;
        assert_eq!(info.aic, Some(-2.0 * info.loglik + 2.0 * k));
        assert_eq!(
            info.bic,
            Some(-2.0 * info.loglik + (info.nobs as f64).ln() * k)
        );
    }
    // Both parts differ, the random parts one dimension apart.
    let c = cmp(
        &ml("Reaction ~ 1 + (1 | Subject)"),
        &ml("Reaction ~ Days + (Days | Subject)"),
    )
    .unwrap();
    let t = lr(&c);
    assert_eq!(t.df, 3);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 2 });
    assert_eq!(t.notes, vec![MIXTURE]);
    // More than one dimension apart.
    let c = cmp(
        &side("Reaction ~ 1", &d, Family::Gaussian, None),
        &ml("Reaction ~ Days + (Days | Subject)"),
    )
    .unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiSquared { df: 4 });
    assert_eq!(t.notes, vec![LrNote::BoundaryIgnored]);
    assert_eq!(t.objective, Objective::MaximumLikelihood);
}

#[test]
fn aic_and_bic_are_left_out_on_reml_fits() {
    let d = sleepstudy();
    let c = cmp(
        &side(
            "Reaction ~ Days + (1 | Subject)",
            &d,
            Family::Gaussian,
            None,
        ),
        &side(
            "Reaction ~ Days + (Days | Subject)",
            &d,
            Family::Gaussian,
            None,
        ),
    )
    .unwrap();
    for info in [&c.m0, &c.m1] {
        assert_eq!((info.aic, info.bic), (None, None));
        assert_eq!(info.objective, Objective::Reml);
    }
}

fn slope1() -> Table {
    csv_table(
        include_str!("../validation/data/simulated/sim_binomial_slope1.csv"),
        &["g"],
    )
}

#[test]
fn glm_and_glmm_pairs_are_labelled_with_the_larger_models_objective() {
    let d = slope1();
    let fit = |f: &str, nagq| fitted(f, &d, binomial(), None, nagq).0;
    // GLMM pair, the same random part.
    let t = lr(&cmp(&fit("y ~ 1 + (1 | g)", 1), &fit("y ~ x + (1 | g)", 1)).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiSquared { df: 1 });
    assert_eq!(t.objective, Objective::Laplace);
    assert!(t.notes.is_empty());
    // A GLM against its GLMM, Laplace or AGQ: one dimension, no node note.
    let glm = fit("y ~ x", 1);
    for (nagq, objective) in [(1, Objective::Laplace), (7, Objective::Agq { nodes: 7 })] {
        let t = lr(&cmp(&glm, &fit("y ~ x + (1 | g)", nagq)).unwrap()).clone();
        assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
        assert_eq!(t.objective, objective);
        assert_eq!(t.notes, vec![MIXTURE]);
    }
    // A GLMM pair one dimension apart.
    let t = lr(&cmp(&fit("y ~ x + (1 | g)", 1), &fit("y ~ x + (1 + x | g)", 1)).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 1 });
    assert_eq!(t.notes, vec![MIXTURE]);
    // More than one dimension apart.
    let t = lr(&cmp(&glm, &fit("y ~ x + (1 + x | g)", 1)).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiSquared { df: 3 });
    assert_eq!(t.notes, vec![LrNote::BoundaryIgnored]);
}

#[test]
fn glm_pairs_without_dispersion_keep_the_likelihood_ratio_test() {
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    for (y, family, d) in [
        ("y", binomial(), slope1()),
        ("yc", poisson, identical_groups()),
    ] {
        let c = cmp(
            &side(&format!("{y} ~ 1"), &d, family, None),
            &side(&format!("{y} ~ x"), &d, family, None),
        )
        .unwrap();
        let t = lr(&c);
        assert_eq!(t.reference, Reference::ChiSquared { df: 1 }, "{y}");
        assert_eq!(t.objective, Objective::MaximumLikelihood);
        assert_eq!(t.statistic, floored(&c));
    }
}

#[test]
fn two_mixed_fits_with_different_node_counts_carry_a_note_naming_both() {
    let d = slope1();
    let fit = |f: &str, nagq| fitted(f, &d, binomial(), None, nagq).0;
    let (small, large) = (fit("y ~ 1 + (1 | g)", 1), fit("y ~ x + (1 | g)", 7));
    let c = compare(&small, &large, &opts()).unwrap();
    let t = lr(&c);
    assert_eq!(t.notes, vec![LrNote::NodeCounts { m0: 1, m1: 7 }]);
    assert_eq!(t.objective, Objective::Agq { nodes: 7 });
    assert!(c.note_lines()[0].contains("node counts, 1 and 7"));
    let c = compare(&large, &small, &opts()).unwrap();
    let t = lr(&c);
    assert_eq!(t.notes, vec![LrNote::NodeCounts { m0: 7, m1: 1 }]);
    assert_eq!(t.objective, Objective::Agq { nodes: 7 });
    // The larger model's objective, whichever node count it has.
    let (small, large) = (fit("y ~ 1 + (1 | g)", 7), fit("y ~ x + (1 | g)", 1));
    let t = lr(&cmp(&small, &large).unwrap()).clone();
    assert_eq!(t.objective, Objective::Laplace);
    assert_eq!(t.notes, vec![LrNote::NodeCounts { m0: 7, m1: 1 }]);
}

/// The F of `L β = 0` on `fit`, `L` picking the columns named in `names`.
fn term_f(fit: &Fit, columns: &[String], names: &[&str]) -> f64 {
    let p = columns.len();
    let mut l = vec![0.0; names.len() * p];
    for (i, name) in names.iter().enumerate() {
        l[i * p + columns.iter().position(|c| c == name).unwrap()] = 1.0;
    }
    fit.contrast_f(&l, names.len()).unwrap().f
}

fn assert_close(got: f64, want: f64, what: &str) {
    assert!(
        (got - want).abs() <= 1e-9 * want.abs(),
        "{what}: {got} against {want}"
    );
}

#[test]
fn two_ols_fits_get_the_exact_f_whatever_their_reml() {
    let d = table();
    let mut answers = Vec::new();
    for (r0, r1) in [(None, None), (Some(true), Some(true)), (None, Some(true))] {
        let c = cmp(&model("y ~ x", r0), &model("y ~ x + z", r1)).unwrap();
        answers.push(outcome(&c));
        let t = linear(&c);
        assert_eq!(t.kind, LinearKind::F);
        assert_eq!(
            t.reference,
            Reference::F {
                num_df: 1.0,
                den_df: 27.0,
                den_df_kind: DenDf::Residual
            }
        );
    }
    assert!(answers.windows(2).all(|w| w[0] == w[1]), "{answers:?}");
    // The F of the term the larger fit adds, on that fit.
    let lo = lower("y ~ x + z", &d, Family::Gaussian).unwrap();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let c = cmp(&model("y ~ x", None), &model("y ~ x + z", None)).unwrap();
    assert_close(
        linear(&c).statistic,
        term_f(&fit, &lo.col_names, &["z"]),
        "y ~ x + z",
    );
    // No column matching: `y ~ 0 + b` spans the intercept.
    let c = cmp(&model("y ~ 0 + b", None), &model("y ~ b + x", None)).unwrap();
    let t = linear(&c);
    assert_eq!(
        (t.num_df, t.reference),
        (
            1.0,
            Reference::F {
                num_df: 1.0,
                den_df: 26.0,
                den_df_kind: DenDf::Residual
            }
        )
    );
    let lo = lower("y ~ b + x", &d, Family::Gaussian).unwrap();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert_close(
        t.statistic,
        term_f(&fit, &lo.col_names, &["x"]),
        "y ~ b + x",
    );
}

#[test]
fn two_dispersion_glms_get_the_wald_f_on_the_residual_df() {
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let igauss = Family::InverseGaussian {
        link: InverseGaussianLink::Log,
    };
    for (family, d) in [
        (
            gamma,
            csv_table(
                include_str!("../validation/data/simulated/glm_gamma.csv"),
                &["grp"],
            ),
        ),
        (
            igauss,
            csv_table(
                include_str!("../validation/data/simulated/sim_igauss.csv"),
                &["grp"],
            ),
        ),
    ] {
        let c = cmp(
            &side("y ~ x", &d, family, None),
            &side("y ~ x + grp", &d, family, None),
        )
        .unwrap();
        let t = linear(&c);
        let lo = lower("y ~ x + grp", &d, family).unwrap();
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let grp: Vec<&str> = lo
            .col_names
            .iter()
            .filter(|n| n.starts_with("grp"))
            .map(String::as_str)
            .collect();
        let r = grp.len() as f64;
        assert_eq!(
            t.reference,
            Reference::F {
                num_df: r,
                den_df: (lo.n - lo.p) as f64,
                den_df_kind: DenDf::Residual
            }
        );
        assert_close(
            t.statistic,
            term_f(&fit, &lo.col_names, &grp),
            "y ~ x + grp",
        );
    }
}

#[test]
fn dispersion_glms_whose_names_do_not_nest_fall_back_to_the_asymptotic_lr_test() {
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let mut d = csv_table(
        include_str!("../validation/data/simulated/glm_gamma.csv"),
        &["grp"],
    );
    // `y ~ 0 + grp` names no intercept, which `y ~ grp + x` has.
    let c = cmp(
        &side("y ~ 0 + grp", &d, gamma, None),
        &side("y ~ grp + x", &d, gamma, None),
    )
    .unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiSquared { df: 1 });
    assert_eq!(t.notes, vec![LrNote::NoWaldF]);
    assert!(c.note_lines()[0].contains("asymptotic"));
    // A copy of `x` aliased in the larger fit only.
    let x = match &d.columns.iter().find(|(n, _)| n == "x").unwrap().1 {
        Column::Numeric(v) => v.clone(),
        _ => unreachable!("x is numeric"),
    };
    d.columns.push(("x2".into(), Column::Numeric(x)));
    let large = side("y ~ x + x2 + grp", &d, gamma, None);
    assert!(large.aliased.contains(&true));
    let t = lr(&cmp(&side("y ~ x2", &d, gamma, None), &large).unwrap()).clone();
    assert_eq!(t.notes, vec![LrNote::NoWaldF]);
}

#[test]
fn dispersion_glms_with_a_held_dispersion_get_the_lr_test() {
    // With φ held there is no residual df to estimate it on: the pair is an
    // ordinary likelihood-ratio pair.
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let d = csv_table(
        include_str!("../validation/data/simulated/glm_gamma.csv"),
        &["grp"],
    );
    let held = |f: &str| {
        let mut lo = lower(f, &d, gamma).unwrap();
        lo.opts.dispersion = Some(1.0);
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        assert!(fit.converged(), "{f} did not converge");
        Model::from_fit(f, &lo, &fit)
    };
    let c = cmp(&held("y ~ x"), &held("y ~ x + grp")).unwrap();
    let t = lr(&c);
    assert_eq!(t.objective, Objective::MaximumLikelihood);
    assert!(t.notes.is_empty());
    assert_eq!(t.statistic, floored(&c));
}

#[test]
fn a_gamma_glmm_against_its_glm_gets_the_mixture() {
    let d = csv_table(
        include_str!("../validation/data/simulated/sim_gamma.csv"),
        &["cluster", "grp"],
    );
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let c = cmp(
        &side("y ~ 1 + x + grp", &d, gamma, None),
        &side("y ~ 1 + x + grp + (1 | cluster)", &d, gamma, None),
    )
    .unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(t.objective, Objective::Laplace);
    assert_eq!(t.notes, vec![MIXTURE]);
    assert!(t.statistic > 0.0);
}

#[test]
fn the_sides_are_kept_as_given_and_the_larger_is_named() {
    let d = sleepstudy();
    let small = side(
        "Reaction ~ Days + (1 | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let large = side(
        "Reaction ~ Days + (Days | Subject)",
        &d,
        Family::Gaussian,
        None,
    );
    let c = compare(&large, &small, &opts()).unwrap();
    assert_eq!(c.m0.formula_label, "Reaction ~ Days + (Days | Subject)");
    assert_eq!(c.m1.formula_label, "Reaction ~ Days + (1 | Subject)");
    assert_eq!(c.larger, Side::First);
    assert_eq!(
        (c.m0.npar, c.m0.nobs, c.m0.loglik),
        (large.df, large.nobs, large.loglik)
    );
}

#[test]
fn the_same_call_twice_gives_the_same_bytes() {
    let d = csv_table(
        include_str!("../validation/data/empirical/Dyestuff.csv"),
        &["Batch"],
    );
    let ols = side("Yield ~ 1", &d, Family::Gaussian, Some(true));
    let full = side("Yield ~ 1 + (1 | Batch)", &d, Family::Gaussian, None);
    let s = slope1();
    let pair = [
        (ols, full),
        (
            fitted("y ~ x", &s, binomial(), None, 1).0,
            fitted("y ~ x + (1 | g)", &s, binomial(), None, 1).0,
        ),
    ];
    for (a, b) in &pair {
        let once = format!("{:?}", compare(a, b, &opts()).unwrap());
        assert_eq!(once, format!("{:?}", compare(a, b, &opts()).unwrap()));
    }
    let c = compare(&pair[0].0, &pair[0].1, &opts()).unwrap();
    assert_eq!(
        lr(&c).reference,
        Reference::ExactRlrt {
            nsim: opts().nsim,
            seed: opts().seed
        }
    );
}

// ── The per-term tests of one fit ───────────────────────────────────────────

const LMM: &str = "y ~ g * h + x + (1 + x | subj)";
const FIXED: &str = "y ~ g * h + x";
const GLMM: &str = "y ~ g * h + x + (1 | subj)";
/// R's `term.labels` of the three formulas above.
const LABELS: [&str; 4] = ["g", "h", "x", "g:h"];

fn sim_factorial() -> Table {
    csv_table(
        include_str!("../validation/data/simulated/sim_factorial.csv"),
        &["subj", "g", "h"],
    )
}

fn sim_factorial_bern() -> Table {
    csv_table(
        include_str!("../validation/data/simulated/sim_factorial_bern.csv"),
        &["subj", "g", "h"],
    )
}

fn glm_gamma() -> Table {
    csv_table(
        include_str!("../validation/data/simulated/glm_gamma.csv"),
        &["grp"],
    )
}

fn sim_gamma() -> Table {
    csv_table(
        include_str!("../validation/data/simulated/sim_gamma.csv"),
        &["cluster", "grp"],
    )
}

fn gamma() -> Family {
    Family::Gamma {
        link: GammaLink::Log,
    }
}

/// A converged fit of `formula` on `table` with `reml`, and its lowering.
fn lowered_fit(formula: &str, table: &Table, family: Family, reml: Option<bool>) -> (Lowered, Fit) {
    let mut lo = lower(formula, table, family).unwrap();
    lo.opts.reml = reml;
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged(), "{formula} did not converge");
    (lo, fit)
}

/// The options of a `term_tests` call asking for `ss_type`.
fn options(ss_type: SsType) -> TermTestOptions {
    let mut opts = TermTestOptions::default();
    opts.ss_type = ss_type;
    opts
}

/// The per-term table `opts` asks for, its χ² rows pinned to the Wald test so
/// they stay Wald whatever the default test gives; a fit whose rows are F
/// tests refuses an explicit test, so its table takes the default, which must
/// be its F table (a χ² fit that refused the Wald test fails here).
fn term_table(
    lo: &Lowered,
    fit: &Fit,
    mut opts: TermTestOptions,
) -> Result<TermTable, TermTestError> {
    opts.test = TermTest::Wald;
    match term_tests(lo, fit, &opts) {
        Err(TermTestError::FTable { .. }) => {
            opts.test = TermTest::Auto;
            let table = term_tests(lo, fit, &opts);
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

fn tables(lo: &Lowered, fit: &Fit) -> [TermTable; 2] {
    [SsType::Type3, SsType::Type2].map(|ss_type| term_table(lo, fit, options(ss_type)).unwrap())
}

fn labels(t: &TermTable) -> Vec<&str> {
    t.fixed.iter().map(|r| r.label.as_str()).collect()
}

/// The fits of the per-term tests, one per kind of row: the Gaussian LMM by
/// REML and by ML, OLS, the binomial GLM and GLMM, a Gamma GLM with φ
/// estimated and with φ held (at 0.5, so a χ² row scaled by the dispersion
/// shows), and a Gamma GLMM (φ̂ not 1). Each with its name, the kind of its
/// rows and, on the F rows, the denominator df's kind.
fn fits() -> Vec<(&'static str, Lowered, Fit, LinearKind, Option<DenDf>)> {
    let (d, b) = (sim_factorial(), sim_factorial_bern());
    let sat = Some(DenDf::Satterthwaite);
    let res = Some(DenDf::Residual);
    let f = LinearKind::F;
    let chi = LinearKind::ChiSquared;
    let fits = [
        ("REML LMM", LMM, &d, Family::Gaussian, None, f, sat),
        ("ML LMM", LMM, &d, Family::Gaussian, Some(false), f, sat),
        ("OLS", FIXED, &d, Family::Gaussian, None, f, res),
        ("GLM", FIXED, &b, binomial(), None, chi, None),
        ("GLMM", GLMM, &b, binomial(), None, chi, None),
    ];
    let mut out: Vec<_> = fits
        .into_iter()
        .map(|(name, formula, table, family, reml, kind, den)| {
            let (lo, fit) = lowered_fit(formula, table, family, reml);
            (name, lo, fit, kind, den)
        })
        .collect();
    let (lo, fit) = lowered_fit("y ~ x + grp", &glm_gamma(), gamma(), None);
    out.push(("Gamma GLM", lo, fit, f, res));
    let mut lo = lower("y ~ x + grp", &glm_gamma(), gamma()).unwrap();
    lo.opts.dispersion = Some(0.5);
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged());
    out.push(("Gamma GLM, φ held", lo, fit, chi, None));
    let (lo, fit) = lowered_fit("y ~ x + grp + (1 | cluster)", &sim_gamma(), gamma(), None);
    assert!((fit.dispersion - 1.0).abs() > 0.1, "{}", fit.dispersion);
    out.push(("Gamma GLMM", lo, fit, chi, None));
    out
}

#[test]
fn each_fit_gets_one_row_per_fixed_term_in_r_order_with_its_familys_test() {
    for (name, lo, fit, kind, den) in fits() {
        if name == "ML LMM" {
            assert!(!fit.reml, "{name}");
        }
        if name == "GLMM" {
            // The GLMM's rows stand on the default Hessian covariance.
            assert_eq!(lo.opts.wald_se, WaldSe::Hessian);
            assert!(
                !fit.diagnostics
                    .notes
                    .iter()
                    .any(|n| matches!(n, Note::HessianSeFallback)),
                "{name}"
            );
        }
        let want: Vec<&str> = if name.starts_with("Gamma GLM") {
            vec!["x", "grp"]
        } else {
            LABELS.to_vec()
        };
        for t in tables(&lo, &fit) {
            let what = format!("{name}, {:?}", t.ss_type);
            // No intercept row and no random-effect row.
            assert_eq!(labels(&t), want, "{what}");
            for row in &t.fixed {
                let test = &row.test;
                assert_eq!(test.kind, kind, "{what} {}", row.label);
                assert!(test.statistic > 0.0, "{what} {}: {test:?}", row.label);
                assert!(
                    test.p_value > 0.0 && test.p_value < 1.0,
                    "{what} {}: {test:?}",
                    row.label
                );
                assert!(test.notes.is_empty(), "{what} {}: {test:?}", row.label);
                match (kind, test.reference) {
                    (
                        LinearKind::F,
                        Reference::F {
                            num_df,
                            den_df,
                            den_df_kind,
                        },
                    ) => {
                        assert_eq!(Some(den_df_kind), den, "{what} {}", row.label);
                        assert_eq!(num_df, test.num_df);
                        if den == Some(DenDf::Residual) {
                            assert_eq!(den_df, (lo.n - lo.p) as f64, "{what} {}", row.label);
                        }
                        let ms = test.statistic * fit.dispersion;
                        assert_eq!(row.mean_sq, Some(ms), "{what} {}", row.label);
                        assert_eq!(row.sum_sq, Some(ms * num_df), "{what} {}", row.label);
                    }
                    (LinearKind::ChiSquared, Reference::ChiSquared { df }) => {
                        assert_eq!(df as f64, test.num_df, "{what} {}", row.label);
                        assert_eq!((row.sum_sq, row.mean_sq), (None, None), "{what}");
                    }
                    other => panic!("{what} {}: {other:?}", row.label),
                }
            }
        }
    }
}

#[test]
fn a_one_df_type_iii_row_of_an_uncontained_term_is_the_summary_statistic_squared() {
    // `x` is in no other term, so its Type III hypothesis is `β_x = 0`: the
    // row is t² on t's df (z² on a Wald χ² row).
    for (name, lo, fit, kind, _) in fits() {
        let j = lo.col_names.iter().position(|c| c == "x").unwrap();
        let t3 = term_table(&lo, &fit, TermTestOptions::default()).unwrap();
        assert_eq!(t3.ss_type, SsType::Type3);
        let row = t3.fixed.iter().find(|r| r.label == "x").unwrap();
        let z = fit.beta[j] / fit.se[j];
        assert_eq!(row.test.num_df, 1.0, "{name}");
        assert_close(row.test.statistic, z * z, name);
        if kind == LinearKind::F {
            let Reference::F { den_df, .. } = row.test.reference else {
                panic!("{name}: {:?}", row.test.reference)
            };
            assert_close(den_df, fit.coef_df[j], name);
        }
    }
}

#[test]
fn the_row_kind_is_read_from_the_fit_not_from_the_options() {
    // A Gamma GLM with φ estimated, its lowering then given a held φ: the
    // rows stay the fit's F rows. And the reverse: a fit with φ held keeps
    // its χ² rows beside options that estimate φ.
    let (mut lo, fit) = lowered_fit("y ~ x + grp", &glm_gamma(), gamma(), None);
    let want = tables(&lo, &fit);
    for t in &want {
        assert!(t.fixed.iter().all(|r| r.test.kind == LinearKind::F));
    }
    lo.opts.dispersion = Some(0.5);
    assert_eq!(tables(&lo, &fit), want);
    let mut lo = lower("y ~ x + grp", &glm_gamma(), gamma()).unwrap();
    lo.opts.dispersion = Some(0.5);
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let want = tables(&lo, &fit);
    for t in &want {
        assert!(t
            .fixed
            .iter()
            .all(|r| r.test.kind == LinearKind::ChiSquared));
    }
    lo.opts.dispersion = None;
    assert_eq!(tables(&lo, &fit), want);
}

#[test]
fn without_the_satterthwaite_pass_the_lmm_rows_have_no_df_and_say_why() {
    let mut lo = lower(LMM, &sim_factorial(), Family::Gaussian).unwrap();
    lo.opts.satterthwaite = false;
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    for t in tables(&lo, &fit) {
        assert_eq!(labels(&t), LABELS);
        for row in &t.fixed {
            let test = &row.test;
            assert_eq!(test.kind, LinearKind::F);
            assert!(test.statistic.is_nan() && test.num_df.is_nan() && test.p_value.is_nan());
            assert!(
                matches!(test.reference, Reference::F { den_df, den_df_kind: DenDf::Satterthwaite, .. } if den_df.is_nan()),
                "{test:?}"
            );
            assert_eq!(
                test.notes,
                vec![LinearNote::NoDf {
                    reason: NoDfReason::Disabled
                }]
            );
        }
    }
}

#[test]
fn an_empty_cell_puts_lmertests_note_on_the_type_iii_table() {
    let empty = csv_table(
        include_str!("../validation/data/simulated/sim_factorial_empty.csv"),
        &["subj", "g", "h"],
    );
    let (lo, fit) = lowered_fit(LMM, &empty, Family::Gaussian, None);
    let [t3, t2] = tables(&lo, &fit);
    assert_eq!(
        t3.notes,
        vec!["Missing cells for: gc:hq. Interpret type III hypotheses with care.".to_string()]
    );
    assert!(t2.notes.is_empty(), "{:?}", t2.notes);
    assert_eq!(labels(&t3), LABELS);
    let (lo, fit) = lowered_fit(LMM, &sim_factorial(), Family::Gaussian, None);
    assert!(tables(&lo, &fit).iter().all(|t| t.notes.is_empty()));
}

#[test]
fn an_interaction_without_its_marginal_term_is_refused() {
    let (lo, fit) = lowered_fit(
        "y ~ x + x:g + (1 + x | subj)",
        &sim_factorial(),
        Family::Gaussian,
        None,
    );
    for ss_type in [SsType::Type3, SsType::Type2] {
        let e = term_table(&lo, &fit, options(ss_type)).unwrap_err();
        let TermTestError::NotMarginal(message) = &e else {
            panic!("{e:?}")
        };
        assert!(message.contains("`g` (a margin of `x:g`)"), "{message}");
        assert_eq!(e.to_string(), *message);
    }
}

#[test]
fn a_term_whose_columns_are_all_aliased_gets_an_untested_row() {
    // `x2` copies `x`, and `k` is constant, aliased on the intercept (which
    // leaves the intercept alone retained). The F path and the χ² path.
    for (table, family) in [
        (sim_factorial(), Family::Gaussian),
        (sim_factorial_bern(), binomial()),
    ] {
        let mut d = table;
        let x = match &d.columns.iter().find(|(n, _)| n == "x").unwrap().1 {
            Column::Numeric(v) => v.clone(),
            _ => unreachable!("x is numeric"),
        };
        d.columns.push(("x2".into(), Column::Numeric(x)));
        d.columns
            .push(("k".into(), Column::Numeric(vec![2.0; d.n])));
        for (formula, label, column) in [("y ~ x + x2", "x2", 2), ("y ~ k", "k", 1)] {
            let (lo, fit) = lowered_fit(formula, &d, family, None);
            assert!(fit.aliased()[column], "{formula}");
            for t in tables(&lo, &fit) {
                let row = t.fixed.iter().find(|r| r.label == label).unwrap();
                let what = format!("{formula}, {family:?}, {:?}", t.ss_type);
                assert!(row.test.statistic.is_nan(), "{what}: {row:?}");
                assert!(row.test.p_value.is_nan(), "{what}: {row:?}");
                assert_eq!(
                    row.test.notes,
                    vec![LinearNote::AliasedColumn { column }],
                    "{what}"
                );
                // The name its row note prints.
                assert_eq!(t.coef_names[column], label, "{what}");
            }
        }
    }
}

#[test]
fn a_covariance_that_is_not_finite_gives_nan_rows() {
    // car's Type II reads the covariance; a NaN one gives NaN rows, not a
    // panic, and so does the Wald χ² of Type III.
    let (lo, mut fit) = lowered_fit(FIXED, &sim_factorial_bern(), binomial(), None);
    fit.vcov = vec![vec![f64::NAN; lo.p]; lo.p];
    for t in tables(&lo, &fit) {
        assert_eq!(labels(&t), LABELS);
        assert!(
            t.fixed.iter().all(|r| r.test.statistic.is_nan()),
            "{:?}",
            t.ss_type
        );
    }
}

#[test]
fn a_pair_differing_by_one_term_gives_that_terms_f_row_on_the_larger_fit() {
    let d = sim_factorial();
    let g = glm_gamma();
    for (small, large, table, family, term) in [
        ("y ~ h + x", "y ~ g + h + x", &d, Family::Gaussian, "g"),
        ("y ~ x", "y ~ x + grp", &g, gamma(), "grp"),
    ] {
        let (lo0, fit0) = lowered_fit(small, table, family, None);
        let (lo1, fit1) = lowered_fit(large, table, family, None);
        let c = cmp(
            &Model::from_fit(small, &lo0, &fit0),
            &Model::from_fit(large, &lo1, &fit1),
        )
        .unwrap();
        let pair = linear(&c);
        for t in tables(&lo1, &fit1) {
            let row = &t.fixed.iter().find(|r| r.label == term).unwrap().test;
            let what = format!("{large}, {:?}", t.ss_type);
            assert_eq!(row.reference, pair.reference, "{what}");
            assert_close(row.statistic, pair.statistic, &what);
            assert_close(row.p_value, pair.p_value, &what);
        }
    }
}

/// The Wald χ² of `β_S = 0` for the columns named in `names`:
/// `β_Sᵀ V_SS⁻¹ β_S`, by Gaussian elimination.
fn wald_of(fit: &Fit, lo: &Lowered, names: &[&str]) -> f64 {
    let s: Vec<usize> = names
        .iter()
        .map(|n| lo.col_names.iter().position(|c| c == n).unwrap())
        .collect();
    let k = s.len();
    let mut a: Vec<Vec<f64>> = s
        .iter()
        .map(|&i| {
            let mut row: Vec<f64> = s.iter().map(|&j| fit.vcov[i][j]).collect();
            row.push(fit.beta[i]);
            row
        })
        .collect();
    for c in 0..k {
        let piv = (c..k)
            .max_by(|&x, &y| a[x][c].abs().total_cmp(&a[y][c].abs()))
            .unwrap();
        a.swap(c, piv);
        let pivot = a[c].clone();
        for (_, row) in a.iter_mut().enumerate().filter(|(r, _)| *r != c) {
            let m = row[c] / pivot[c];
            for (x, p) in row[c..].iter_mut().zip(&pivot[c..]) {
                *x -= m * p;
            }
        }
    }
    (0..k).map(|i| fit.beta[s[i]] * a[i][k] / a[i][i]).sum()
}

#[test]
fn type_ii_on_a_glm_or_glmm_is_cars_covariance_metric() {
    // car's Type II of a term with relatives R is the Wald χ² of the term's
    // and R's columns minus that of R's alone (the covariance metric; the
    // `XᵀX` metric gives another number on these fits).
    let b = sim_factorial_bern();
    let gh = ["gb:hq", "gc:hq"];
    for formula in [FIXED, GLMM] {
        let (lo, fit) = lowered_fit(formula, &b, binomial(), None);
        assert!(gh.iter().all(|n| lo.col_names.iter().any(|c| c == n)));
        let t2 = term_table(&lo, &fit, options(SsType::Type2)).unwrap();
        for (term, cols) in [("g", vec!["gb", "gc"]), ("h", vec!["hq"])] {
            let row = t2.fixed.iter().find(|r| r.label == term).unwrap();
            let with: Vec<&str> = cols.iter().copied().chain(gh).collect();
            let want = wald_of(&fit, &lo, &with) - wald_of(&fit, &lo, &gh);
            assert_close(row.test.statistic, want, &format!("{formula} {term}"));
            assert_eq!(row.test.num_df, cols.len() as f64);
        }
    }
}

#[test]
fn type_ii_on_weighted_ols_is_cars_covariance_metric() {
    // With prior weights the OLS covariance is proportional to (XᵀWX)⁻¹, not
    // (XᵀX)⁻¹, so only the covariance metric gives car's Type II F.
    let d = csv_table(
        include_str!("../validation/data/simulated/glm_gamma.csv"),
        &["grp"],
    );
    let w = match &d.columns.iter().find(|(n, _)| n == "w").unwrap().1 {
        Column::Numeric(v) => v.clone(),
        _ => unreachable!("w is numeric"),
    };
    let mut lo = lower("y ~ x * grp", &d, Family::Gaussian).unwrap();
    lo.opts.weights = Some(w);
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged());
    let rel = ["x:grpb"];
    assert!(lo.col_names.iter().any(|c| c == rel[0]));
    let t2 = term_table(&lo, &fit, options(SsType::Type2)).unwrap();
    for (term, col) in [("x", "x"), ("grp", "grpb")] {
        let row = t2.fixed.iter().find(|r| r.label == term).unwrap();
        let want = wald_of(&fit, &lo, &[col, rel[0]]) - wald_of(&fit, &lo, &rel);
        assert_close(row.test.statistic * row.test.num_df, want, term);
    }
}
