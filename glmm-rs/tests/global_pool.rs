#![cfg(all(feature = "formula", target_os = "linux"))]
//! A fit, its likelihood-ratio term table and a vector AGQ fit start no
//! threads while in-fit parallelism is off. faer reads its global parallelism
//! wherever a call does not pass `Par` itself, and on a native build that
//! starts rayon's global pool; in a `parallel` build the AGQ kernels run on
//! rayon whenever they are handed a cluster index. The test is the only one in
//! its binary, so no other test can start the pool first.

use glmm::formula::anova::{term_tests, TermTestOptions};
use glmm::formula::{Column, Table};
use glmm::inference::LinearKind;
use glmm::{BinomialLink, Family, FitOptions, PoissonLink, WaldSe};

/// The process's thread count, from `/proc/self/status`.
fn threads() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix("Threads:"))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

#[test]
fn term_tables_and_vector_agq_fits_start_no_threads() {
    let n = 120;
    let (mut y, mut yb, mut x, mut g, mut h) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let xi = ((i * 7) % 11) as f64 / 10.0;
        y.push(((i * 13) % 5) as f64 + (xi * 3.0).floor());
        yb.push(f64::from(((i * 13) % 5 + i % 8) % 2 == 0));
        x.push(xi);
        g.push(format!("g{}", i % 8));
        h.push(format!("h{}", (i / 3) % 5));
    }
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("yb".into(), Column::Numeric(yb)),
            ("x".into(), Column::Numeric(x)),
            ("g".into(), Column::factor_from_labels(&g)),
            ("h".into(), Column::factor_from_labels(&h)),
        ],
        n,
    };
    let family = Family::Poisson {
        link: PoissonLink::Log,
    };
    let before = threads();
    let model = glmm::formula::fit(
        "y ~ x + (1 | g) + (1 | h)",
        &table,
        family,
        &FitOptions::default(),
    )
    .unwrap();
    let terms = term_tests(&model, &TermTestOptions::default()).unwrap();
    assert_eq!(terms.choice.test, LinearKind::LikelihoodRatio);

    // A random slope makes the AGQ vector (q = 2); `WaldSe::Rx` keeps its SE
    // off the exact-derivative path, which runs on rayon in a `parallel` build.
    let mut opts = FitOptions::default();
    opts.nagq = 3;
    opts.wald_se = WaldSe::Rx;
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    glmm::formula::fit("yb ~ x + (1 + x | g)", &table, binomial, &opts).unwrap();
    assert_eq!(threads(), before);
}
