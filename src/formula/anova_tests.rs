//! The two-fit comparison against its crate-private engines: the exact
//! restricted likelihood-ratio test called directly on the same numbers, its
//! refusal reasons, the likelihood-ratio tails, and the refusal texts; and
//! the per-term table's Type II metric on the Gaussian LMM.

use std::collections::BTreeSet;

use super::anova::{
    compare, linear_note_line, nesting, term_tests, CompareError, CompareOptions, Comparison,
    ComparisonTest, Differs, Model, ModelInfo, Side, SsType, TermTestError, TermTestOptions,
};
use super::hypotheses::{
    check_marginal, missing_cells_note, type2_vcov, type2_xtx, TermHypothesis,
};
use crate::formula::{lower, Column, Table};
use crate::inference::contrast::{ContrastError, NoDfReason};
use crate::inference::evd::EvdError;
use crate::inference::lr::lr_p_value;
use crate::inference::rlrt::{exact_rlrt, RlrtConfig, RlrtRefusal};
use crate::inference::{LinearKind, LinearNote, LinearTest, LrNote, LrTest, Objective, Reference};
use crate::test_support::csv_table;
use crate::{fit_cold, Family, PoissonLink};

const OPTS: CompareOptions = CompareOptions {
    nsim: 2000,
    seed: 20_261_012,
};

/// A converged Gaussian fit of `formula` with `reml` and `weights`, as a
/// `Model`.
fn side(formula: &str, table: &Table, reml: Option<bool>, weights: Option<Vec<f64>>) -> Model {
    let mut lo = lower(formula, table, Family::Gaussian).unwrap();
    lo.opts.reml = reml;
    lo.opts.weights = weights;
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged(), "{formula} did not converge");
    Model::from_fit(formula, &lo, &fit)
}

fn lr(c: &Comparison) -> &LrTest {
    match &c.test {
        ComparisonTest::Lr(t) => t,
        other => panic!("not a likelihood-ratio test: {other:?}"),
    }
}

fn dyestuff() -> (Model, Model) {
    let d = csv_table(
        include_str!("../../validation/data/empirical/Dyestuff.csv"),
        &["Batch"],
    );
    (
        side("Yield ~ 1", &d, Some(true), None),
        side("Yield ~ 1 + (1 | Batch)", &d, None, None),
    )
}

/// The exact test of the larger side's design summary, called directly.
fn direct(ols: &Model, full: &Model, parallel_inner: bool) -> crate::inference::rlrt::RlrtResult {
    let (counts, s, c) = full.design_summary.as_ref().expect("inside the envelope");
    let cfg = RlrtConfig {
        nsim: OPTS.nsim,
        seed: OPTS.seed,
        parallel_inner,
    };
    exact_rlrt(counts, s, c, full.loglik, ols.loglik, false, &cfg).unwrap()
}

/// Six groups of eight rows, a numeric `x1` and a response with a group
/// effect (the design of the kernel's `lmm_intercept` tests).
fn lmm_intercept() -> Table {
    let (clusters, n) = (6, 48);
    let mut st = 13u64;
    let (mut x1, mut y, mut g) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..n {
        let id = i % clusters;
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = ((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0;
        let re = 0.3 * (id as f64 - clusters as f64 / 2.0);
        x1.push(x);
        y.push(0.5 + 0.4 * x + re + 0.1 * ((i % 5) as f64 - 2.0));
        g.push(format!("c{id}"));
    }
    Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("x1".into(), Column::Numeric(x1)),
            ("g".into(), Column::factor_from_labels(&g)),
        ],
        n,
    }
}

#[test]
fn the_envelope_pair_gets_the_exact_test_as_p9_computes_it() {
    let (ols, full) = dyestuff();
    let c = compare(&ols, &full, &OPTS).unwrap();
    let t = lr(&c);
    let r = direct(&ols, &full, false);
    assert_eq!(
        t.reference,
        Reference::ExactRlrt {
            nsim: OPTS.nsim,
            seed: OPTS.seed
        }
    );
    let own = (2.0 * (full.loglik - ols.loglik)).max(0.0);
    assert!(own > 0.0);
    assert_eq!(t.statistic.to_bits(), own.to_bits());
    assert_eq!(t.statistic.to_bits(), r.statistic.to_bits());
    assert_eq!(t.p_value.to_bits(), r.p_value.to_bits());
    assert_eq!(t.df, 1);
    assert!(t.notes.is_empty());
}

#[test]
fn the_exact_reference_is_the_same_bytes_every_call_and_every_split() {
    let (ols, full) = dyestuff();
    let once = format!("{:?}", compare(&ols, &full, &OPTS).unwrap());
    assert_eq!(once, format!("{:?}", compare(&ols, &full, &OPTS).unwrap()));
    // Under `parallel` the direct call splits the draws over rayon.
    let p = lr(&compare(&ols, &full, &OPTS).unwrap()).p_value;
    assert_eq!(p.to_bits(), direct(&ols, &full, true).p_value.to_bits());
}

#[test]
fn a_weighted_larger_fit_has_no_design_summary_and_gets_the_mixture() {
    let d = lmm_intercept();
    let w: Vec<f64> = (0..d.n).map(|i| 1.0 + 0.5 * (i % 3) as f64).collect();
    let ols = side("y ~ x1", &d, Some(true), Some(w.clone()));
    let full = side("y ~ x1 + (1 | g)", &d, None, Some(w));
    assert_eq!(full.design_summary, None);
    let c = compare(&ols, &full, &OPTS).unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(
        t.notes,
        vec![
            LrNote::NoDesignSummary,
            LrNote::BoundaryMixture { approximate: false }
        ]
    );
    assert!(t.statistic > 0.0);
    assert_eq!(
        t.p_value.to_bits(),
        lr_p_value(t.statistic, &t.reference).to_bits()
    );
    assert!(c.note_lines()[0].contains("unweighted (1 | g)"));
}

#[test]
fn groups_the_fixed_part_absorbs_get_the_mixture_with_the_refusal_reason() {
    let d = lmm_intercept();
    let ols = side("y ~ g", &d, Some(true), None);
    let full = side("y ~ g + (1 | g)", &d, None, None);
    assert!(full.design_summary.is_some());
    let c = compare(&ols, &full, &OPTS).unwrap();
    let t = lr(&c);
    let reason = RlrtRefusal::NoGroupVariation.reason();
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(
        t.notes,
        vec![
            LrNote::ExactRlrtRefused { reason },
            LrNote::BoundaryMixture { approximate: false }
        ]
    );
    assert!(c.note_lines()[0].contains(reason));
}

#[test]
fn a_mixture_row_takes_its_references_tail() {
    let d = csv_table(
        include_str!("../../validation/data/empirical/sleepstudy.csv"),
        &["Subject"],
    );
    let c = compare(
        &side("Reaction ~ Days + (1 | Subject)", &d, None, None),
        &side("Reaction ~ Days + (Days | Subject)", &d, None, None),
        &OPTS,
    )
    .unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 1 });
    assert_eq!(
        t.p_value.to_bits(),
        lr_p_value(t.statistic, &t.reference).to_bits()
    );
}

/// One refusal of each kind, with every wording `NotConverged` and
/// `MixedEstimators` choose between. The match has no wildcard arm, so a
/// refusal added later does not compile until it is listed here.
fn every_refusal() -> Vec<CompareError> {
    let formula = nesting("y ~ (x", "y ~ x").expect_err("does not parse");
    let every = vec![
        CompareError::NotConverged { m0: true, m1: true },
        CompareError::NotConverged {
            m0: true,
            m1: false,
        },
        CompareError::NotConverged {
            m0: false,
            m1: true,
        },
        CompareError::DifferentRows { m0: 30, m1: 29 },
        CompareError::DifferentY,
        CompareError::DifferentWeights,
        CompareError::DifferentOffset,
        CompareError::DifferentFamily {
            m0: Family::Gaussian,
            m1: Family::Poisson {
                link: PoissonLink::Log,
            },
        },
        CompareError::DifferentDispersionHeld {
            m0: None,
            m1: Some(2.0),
        },
        formula,
        CompareError::DifferentResponse {
            m0: "y".into(),
            m1: "w".into(),
        },
        CompareError::NotNested {
            m0: "y ~ x".into(),
            m1: "y ~ z".into(),
        },
        CompareError::MixedEstimators { reml: Side::First },
        CompareError::MixedEstimators { reml: Side::Second },
        CompareError::RemlDifferentFixed,
        CompareError::SameParts,
        CompareError::NothingToTest { m0: 3, m1: 3 },
    ];
    let kind = |e: &CompareError| match e {
        CompareError::NotConverged { .. } => 0,
        CompareError::DifferentRows { .. } => 1,
        CompareError::DifferentY => 2,
        CompareError::DifferentWeights => 3,
        CompareError::DifferentOffset => 4,
        CompareError::DifferentFamily { .. } => 5,
        CompareError::DifferentDispersionHeld { .. } => 6,
        CompareError::Formula { .. } => 7,
        CompareError::DifferentResponse { .. } => 8,
        CompareError::NotNested { .. } => 9,
        CompareError::MixedEstimators { .. } => 10,
        CompareError::RemlDifferentFixed => 11,
        CompareError::SameParts => 12,
        CompareError::NothingToTest { .. } => 13,
    };
    let kinds: BTreeSet<usize> = every.iter().map(kind).collect();
    assert_eq!(kinds, (0..14).collect(), "one refusal of each kind");
    every
}

/// The refusal texts are port-neutral: each says what is wrong and what to do
/// in plain words and leaves the call syntax (the comparison function, the
/// REML argument) to the port that shows it.
#[test]
fn no_refusal_text_carries_port_syntax() {
    let syntax = [
        "compare(",
        "REML =",
        "reml=",
        "REML=FALSE",
        "REML=TRUE",
        "reml = False",
        "reml = True",
    ];
    for e in every_refusal() {
        let text = e.to_string();
        for s in syntax {
            assert!(!text.contains(s), "{e:?}: {text}");
        }
    }
}

/// Every reason a contrast or a per-term row has no denominator df. The
/// match has no wildcard arm, so a reason added later does not compile until
/// it is listed here.
fn every_no_df_reason() -> Vec<NoDfReason> {
    let every = vec![
        NoDfReason::NotGaussianMixed,
        NoDfReason::Disabled,
        NoDfReason::StateMissing,
        NoDfReason::Failed,
    ];
    let kind = |r: &NoDfReason| match r {
        NoDfReason::NotGaussianMixed => 0,
        NoDfReason::Disabled => 1,
        NoDfReason::StateMissing => 2,
        NoDfReason::Failed => 3,
    };
    let kinds: BTreeSet<usize> = every.iter().map(kind).collect();
    assert_eq!(kinds, (0..4).collect(), "one reason of each kind");
    every
}

/// Every note text a port renders as the crate gives it: each per-term row
/// note ([`LinearNote`], every no-df reason), each comparison note
/// ([`LrNote`], every exact-test refusal, the non-finite note on a NaN side),
/// the contrast refusal, the per-term refusal and the missing-cell note. The
/// matches have no wildcard arm, so a note added later does not compile until
/// it is listed here.
fn every_note_text() -> Vec<String> {
    let reasons = every_no_df_reason();
    let mut linear = vec![
        LinearNote::AliasedColumn { column: 2 },
        LinearNote::RankDeficientL { rank: 1, rows: 2 },
    ];
    let names = ["(Intercept)", "x", "z"].map(String::from);
    linear.extend(reasons.iter().map(|&reason| LinearNote::NoDf { reason }));
    let linear_kind = |n: &LinearNote| match n {
        LinearNote::AliasedColumn { .. } => 0,
        LinearNote::RankDeficientL { .. } => 1,
        LinearNote::NoDf { .. } => 2,
    };
    let kinds: BTreeSet<usize> = linear.iter().map(linear_kind).collect();
    assert_eq!(kinds, (0..3).collect(), "one row note of each kind");

    let refusals = [
        RlrtRefusal::NoResidualDf,
        RlrtRefusal::NoGroupVariation,
        RlrtRefusal::Eigen(EvdError::NoConvergence),
        RlrtRefusal::MalformedSummary,
    ];
    let refusal_kind = |r: &RlrtRefusal| match r {
        RlrtRefusal::NoResidualDf => 0,
        RlrtRefusal::NoGroupVariation => 1,
        RlrtRefusal::Eigen(_) => 2,
        RlrtRefusal::MalformedSummary => 3,
    };
    let kinds: BTreeSet<usize> = refusals.iter().map(refusal_kind).collect();
    assert_eq!(
        kinds,
        (0..4).collect(),
        "one exact-test refusal of each kind"
    );
    let mut lr_notes = vec![
        LrNote::NegativeStatistic { raw: -1.5e-3 },
        LrNote::BoundaryMixture { approximate: true },
        LrNote::BoundaryIgnored,
        LrNote::NoDesignSummary,
        LrNote::NodeCounts { m0: 1, m1: 5 },
        LrNote::NoWaldF,
        LrNote::NonFiniteLoglik,
    ];
    lr_notes.extend(
        refusals
            .iter()
            .map(|r| LrNote::ExactRlrtRefused { reason: r.reason() }),
    );
    let lr_kind = |n: &LrNote| match n {
        LrNote::NegativeStatistic { .. } => 0,
        LrNote::BoundaryMixture { .. } => 1,
        LrNote::BoundaryIgnored => 2,
        LrNote::NoDesignSummary => 3,
        LrNote::ExactRlrtRefused { .. } => 4,
        LrNote::NodeCounts { .. } => 5,
        LrNote::NoWaldF => 6,
        LrNote::NonFiniteLoglik => 7,
    };
    let kinds: BTreeSet<usize> = lr_notes.iter().map(lr_kind).collect();
    assert_eq!(kinds, (0..8).collect(), "one comparison note of each kind");
    let info = |loglik: f64| ModelInfo {
        formula_label: "y ~ 1 + (1 | g)".into(),
        npar: 3,
        nobs: 30,
        loglik,
        aic: None,
        bic: None,
        objective: Objective::Reml,
        coef_names: names.to_vec(),
    };
    let comparison = Comparison {
        m0: info(f64::NAN),
        m1: info(-40.0),
        larger: Side::Second,
        differs: Differs::Random,
        test: ComparisonTest::Lr(LrTest {
            statistic: f64::NAN,
            df: 1,
            reference: Reference::ChiSquared { df: 1 },
            p_value: f64::NAN,
            objective: Objective::Reml,
            notes: lr_notes,
        }),
    };

    let d = csv_table(
        include_str!("../../validation/data/simulated/sim_factorial_empty.csv"),
        &["subj", "g", "h"],
    );
    let empty = lower("y ~ g * h + (1 | subj)", &d, Family::Gaussian).unwrap();
    let not_marginal = lower("y ~ x + x:g + (1 | subj)", &d, Family::Gaussian).unwrap();

    let contrast: Vec<ContrastError> = reasons.into_iter().map(ContrastError::NoDf).collect();
    let contrast_kind = |e: &ContrastError| match e {
        ContrastError::NoDf(_) => 0,
    };
    let kinds: BTreeSet<usize> = contrast.iter().map(contrast_kind).collect();
    assert_eq!(kinds, (0..1).collect(), "one contrast refusal of each kind");
    let term = [TermTestError::NotMarginal(
        check_marginal(&not_marginal).expect_err("lacks g"),
    )];
    let term_kind = |e: &TermTestError| match e {
        TermTestError::NotMarginal(_) => 0,
    };
    let kinds: BTreeSet<usize> = term.iter().map(term_kind).collect();
    assert_eq!(kinds, (0..1).collect(), "one per-term refusal of each kind");

    let mut texts: Vec<String> = linear
        .into_iter()
        .map(|n| linear_note_line(n, &names))
        .collect();
    texts.extend(comparison.note_lines());
    texts.extend(contrast.iter().map(ContrastError::to_string));
    texts.extend(term.iter().map(TermTestError::to_string));
    texts.push(missing_cells_note(&empty).expect("an empty cell"));
    texts
}

/// The Rust syntax a port-neutral text must not carry: a path (`::`), a
/// field set to a boolean, a field read off a value (`opts.satterthwaite`,
/// a '.' between two lowercase letters or underscores; "e.g." and "i.e."
/// are words), or an identifier in snake_case or CamelCase (`NaN` is a
/// value, not an identifier).
fn rust_syntax(text: &str) -> Option<String> {
    for s in ["::", "= false", "= true", "=false", "=true"] {
        if text.contains(s) {
            return Some(s.to_string());
        }
    }
    let field = |c: char| c.is_ascii_lowercase() || c == '_';
    let dotted = text
        .split(|c: char| !(c.is_alphanumeric() || c == '_' || c == '.'))
        .map(|w| w.trim_matches('.'))
        .find(|&w| {
            let b: Vec<char> = w.chars().collect();
            !["e.g", "i.e"].contains(&w)
                && b.windows(3)
                    .any(|t| t[1] == '.' && field(t[0]) && field(t[2]))
        });
    if let Some(w) = dotted {
        return Some(w.to_string());
    }
    text.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .find(|&w| {
            if w == "NaN" {
                return false;
            }
            let snake = w.trim_matches('_').contains('_');
            let camel = w
                .chars()
                .zip(w.chars().skip(1))
                .any(|(a, b)| a.is_lowercase() && b.is_uppercase());
            snake || camel
        })
        .map(str::to_string)
}

/// The note and refusal texts are port-neutral: each port renders them as
/// given, so none names a Rust path, field or identifier.
#[test]
fn no_note_text_carries_rust_syntax() {
    for field in ["(opts.satterthwaite)", "set fit.dispersion first"] {
        assert!(rust_syntax(field).is_some(), "{field}");
    }
    assert_eq!(rust_syntax("e.g. a value, i.e. one of 1.5 or 2."), None);
    let mut texts = every_note_text();
    texts.extend(every_refusal().iter().map(CompareError::to_string));
    assert!(texts.iter().all(|t| !t.is_empty()));
    for text in texts {
        assert_eq!(rust_syntax(&text), None, "{text}");
    }
}

/// A row note on an aliased column names the coefficient, the same name in
/// every port, never a column index.
#[test]
fn an_aliased_column_note_names_its_coefficient() {
    let mut d = csv_table(
        include_str!("../../validation/data/simulated/sim_factorial.csv"),
        &["subj", "g", "h"],
    );
    let x = match &d.columns.iter().find(|(n, _)| n == "x").unwrap().1 {
        Column::Numeric(v) => v.clone(),
        _ => unreachable!("x is numeric"),
    };
    d.columns.push(("x2".into(), Column::Numeric(x)));
    let lo = lower("y ~ g + x + x2", &d, Family::Gaussian).unwrap();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let t = term_tests(&lo, &fit, &TermTestOptions::default()).unwrap();
    let row = t.fixed.iter().find(|r| r.label == "x2").unwrap();
    let lines: Vec<String> = row
        .test
        .notes
        .iter()
        .map(|n| linear_note_line(*n, &t.coef_names))
        .collect();
    assert_eq!(
        lines,
        ["The coefficient x2 is aliased, so the hypothesis was not tested."]
    );
}

/// A hand-built comparison whose aliased-column note points past the larger
/// side's names gets a note that names no coefficient, not a panic.
#[test]
fn an_aliased_column_note_out_of_range_names_no_coefficient() {
    let info = ModelInfo {
        formula_label: "y ~ x".into(),
        npar: 3,
        nobs: 30,
        loglik: -40.0,
        aic: None,
        bic: None,
        objective: Objective::MaximumLikelihood,
        coef_names: ["(Intercept)", "x"].map(String::from).to_vec(),
    };
    let comparison = Comparison {
        m0: info.clone(),
        m1: info,
        larger: Side::Second,
        differs: Differs::Fixed,
        test: ComparisonTest::Linear(LinearTest {
            statistic: f64::NAN,
            kind: LinearKind::ChiSquared,
            num_df: f64::NAN,
            reference: Reference::ChiSquared { df: 0 },
            p_value: f64::NAN,
            notes: vec![LinearNote::AliasedColumn { column: 5 }],
        }),
    };
    assert_eq!(
        comparison.note_lines(),
        ["A tested coefficient is aliased, so the hypothesis was not tested."]
    );
}

/// The F of `h`'s rows on `fit`, `L` flattened row-major.
fn hypothesis_f(fit: &crate::Fit, h: &TermHypothesis) -> f64 {
    let (rows, p) = (h.l.nrows(), h.l.ncols());
    let l: Vec<f64> = (0..rows)
        .flat_map(|i| (0..p).map(move |j| h.l[(i, j)]))
        .collect();
    fit.contrast_f(&l, rows).unwrap().f
}

#[test]
fn type_ii_on_the_gaussian_lmm_is_lmertests_xtx_metric() {
    // lmerTest builds the LMM's Type II from `XᵀX`; car's covariance metric
    // gives another number for a term with containers (`g`, in `g:h`).
    let d = csv_table(
        include_str!("../../validation/data/simulated/sim_factorial.csv"),
        &["subj", "g", "h"],
    );
    let lo = lower("y ~ g * h + x + (1 + x | subj)", &d, Family::Gaussian).unwrap();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged());
    let retained: Vec<usize> = (0..lo.p).filter(|&j| !fit.aliased()[j]).collect();
    let t2 = term_tests(
        &lo,
        &fit,
        &TermTestOptions {
            ss_type: SsType::Type2,
        },
    )
    .unwrap();
    let xtx = type2_xtx(&lo, &retained);
    let vcov = type2_vcov(&lo, &retained, &fit.vcov);
    for ((row, h), car) in t2.fixed.iter().zip(&xtx).zip(&vcov) {
        assert_eq!(row.label, h.term);
        assert_eq!(row.test.statistic, hypothesis_f(&fit, h), "{}", row.label);
        if row.label == "g" {
            let other = hypothesis_f(&fit, car);
            let gap = (row.test.statistic - other).abs() / other;
            assert!(
                gap > 1e-4,
                "g: {} against car's {other}",
                row.test.statistic
            );
        }
    }
}
