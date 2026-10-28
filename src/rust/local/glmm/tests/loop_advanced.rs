#![cfg(feature = "loop_advanced")]
//! The loop tier's direct access to the hypothesis tests: each re-export of
//! `glmm::loop_advanced`, called on the numbers of a small fit, gives the
//! value the stable entry points (`compare`, `term_tests`) report, bit for bit.

#[cfg(feature = "formula")]
use glmm::formula::anova::{
    compare, term_tests, CompareOptions, ComparisonTest, TermTest, TermTestOptions,
};
#[cfg(feature = "formula")]
use glmm::formula::{Column, Model, Table};
#[cfg(feature = "formula")]
use glmm::inference::Reference;
#[cfg(feature = "formula")]
use glmm::loop_advanced::{lr_p_value_pub, type3_pub, wald_chisq_pub};
#[cfg(feature = "formula")]
use glmm::{Family, FitOptions, PoissonLink};

#[cfg(feature = "formula")]
/// 30 rows: six groups `g` of five, a numeric `x`, a two-level `a` and a
/// three-level `b` (every pair of levels present), a Gaussian response `y`
/// with a group effect and a count response `k`.
fn table() -> Table {
    let n = 30;
    let (mut y, mut k, mut x) = (Vec::new(), Vec::new(), Vec::new());
    let (mut g, mut a, mut b) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let xi = ((i * 7) % 11) as f64 / 10.0;
        let u = [0.9, -0.6, 0.3, -1.1, 0.7, -0.2][i / 5];
        let e = (((i * 37) % 17) as f64 - 8.0) / 9.0;
        y.push(2.0 + 0.8 * xi + u + e);
        k.push(((i * 13) % 7 + i % 2 * 3) as f64);
        x.push(xi);
        g.push(format!("g{}", i / 5));
        a.push(format!("a{}", i % 2));
        b.push(format!("b{}", i % 3));
    }
    Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("k".into(), Column::Numeric(k)),
            ("x".into(), Column::Numeric(x)),
            ("g".into(), Column::factor_from_labels(&g)),
            ("a".into(), Column::factor_from_labels(&a)),
            ("b".into(), Column::factor_from_labels(&b)),
        ],
        n,
    }
}

#[cfg(feature = "formula")]
fn poisson() -> Family {
    Family::Poisson {
        link: PoissonLink::Log,
    }
}

#[cfg(feature = "formula")]
/// The converged fit of `formula` with `reml`.
fn model(formula: &str, family: Family, reml: Option<bool>) -> Model {
    let mut fit_opts = FitOptions::default();
    fit_opts.reml = reml;
    let m = glmm::formula::fit(formula, &table(), family, &fit_opts).unwrap();
    assert!(m.fit().converged(), "{formula} did not converge");
    m
}

#[test]
#[cfg(feature = "formula")]
fn the_lr_p_value_is_the_one_compare_reports() {
    let pairs = [
        ("k ~ x", "k ~ x + a", poisson(), None),
        ("y ~ x", "y ~ x + (1 | g)", Family::Gaussian, Some(false)),
    ];
    let opts = CompareOptions {
        nsim: 1000,
        seed: 1,
    };
    let mut references = Vec::new();
    for (f0, f1, family, reml) in pairs {
        let (m0, m1) = (model(f0, family, reml), model(f1, family, reml));
        let c = compare(&m0, &m1, &opts).unwrap();
        let ComparisonTest::Lr(t) = &c.test else {
            panic!("{f0} against {f1}: not a likelihood-ratio test");
        };
        assert_eq!(
            lr_p_value_pub(t.statistic, &t.reference).to_bits(),
            t.p_value.to_bits(),
            "{f0} against {f1}"
        );
        references.push(t.reference);
    }
    assert_eq!(
        references,
        [
            Reference::ChiSquared { df: 1 },
            Reference::ChiBarSquared { df_low: 0 }
        ]
    );
}

#[test]
#[cfg(feature = "formula")]
fn type3_and_the_wald_chisq_give_the_term_tests_rows() {
    let m = model("k ~ x + a * b", poisson(), None);
    let (lo, fit) = (m.lowered(), m.fit());
    let mut opts = TermTestOptions::default();
    opts.test = TermTest::Wald;
    let table = term_tests(&m, &opts).unwrap();
    let retained: Vec<usize> = (0..lo.p).filter(|&j| !fit.aliased()[j]).collect();
    let hypotheses = type3_pub(lo, &retained);
    assert_eq!(hypotheses.len(), table.fixed.len());
    for (h, row) in hypotheses.iter().zip(&table.fixed) {
        assert_eq!(h.term, row.label);
        let rows = h.l.nrows();
        let l: Vec<f64> = (0..rows)
            .flat_map(|i| (0..lo.p).map(move |j| h.l[(i, j)]))
            .collect();
        let t = wald_chisq_pub(&fit.beta, &fit.vcov, fit.aliased(), &l, rows);
        let bits = |v: f64| v.to_bits();
        assert_eq!(bits(t.statistic), bits(row.test.statistic), "{}", h.term);
        assert_eq!(bits(t.num_df), bits(row.test.num_df), "{}", h.term);
        assert_eq!(bits(t.p_value), bits(row.test.p_value), "{}", h.term);
        assert_eq!(
            (t.kind, t.reference, &t.notes),
            (row.test.kind, row.test.reference, &row.test.notes),
            "{}",
            h.term
        );
    }
}
