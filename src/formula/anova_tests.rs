//! The two-fit comparison against its crate-private engines: the exact
//! restricted likelihood-ratio test called directly on the same numbers, its
//! refusal reasons, the likelihood-ratio tails, and the refusal texts; the
//! per-term table's Type II metric on the Gaussian LMM; its test choice; and
//! its likelihood-ratio rows against `compare` on their formula pairs.

use std::collections::BTreeSet;

use faer::Mat;

use super::anova::{
    compare, linear_note_line, lrt_no_basis_note, lrt_not_converged_note, lrt_test, nesting,
    refit_on_basis, term_tests, theta_start, ChoiceReason, CompareError, CompareOptions,
    Comparison, ComparisonTest, Differs, Model, ModelInfo, Side, SsType, TermTest, TermTestError,
    TermTestOptions,
};
use super::hypotheses::{
    check_marginal, missing_cells_note, type2_vcov, type2_xtx, type3, type3_null_basis,
    type3_unrounded, untouched_columns, TermHypothesis,
};
use crate::formula::{lower, Column, Lowered, Table};
use crate::inference::contrast::{ContrastError, NoDfReason};
use crate::inference::evd::EvdError;
use crate::inference::lr::lr_p_value;
use crate::inference::rlrt::{exact_rlrt, RlrtConfig, RlrtRefusal};
use crate::inference::{
    DenDf, LinearKind, LinearNote, LinearTest, LrNote, LrTest, Objective, Reference,
};
use crate::test_support::csv_table;
use crate::{fit_cold, BinomialLink, Family, Fit, GammaLink, NegBinomialLink, PoissonLink};

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
        LinearNote::ReducedNotConverged,
        LinearNote::RandomSlopeColumn,
        LinearNote::NegativeStatistic,
    ];
    let names = ["(Intercept)", "x", "z"].map(String::from);
    linear.extend(reasons.iter().map(|&reason| LinearNote::NoDf { reason }));
    let linear_kind = |n: &LinearNote| match n {
        LinearNote::AliasedColumn { .. } => 0,
        LinearNote::RankDeficientL { .. } => 1,
        LinearNote::NoDf { .. } => 2,
        LinearNote::ReducedNotConverged => 3,
        LinearNote::RandomSlopeColumn => 4,
        LinearNote::NegativeStatistic => 5,
    };
    let kinds: BTreeSet<usize> = linear.iter().map(linear_kind).collect();
    assert_eq!(kinds, (0..6).collect(), "one row note of each kind");

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
    let mut term = vec![TermTestError::NotMarginal(
        check_marginal(&not_marginal).expect_err("lacks g"),
    )];
    for asked in [TermTest::Wald, TermTest::Lrt] {
        for den_df in [DenDf::Satterthwaite, DenDf::Residual] {
            term.push(TermTestError::FTable { asked, den_df });
        }
    }
    term.push(TermTestError::RefitNodeCount { fit: 5, refits: 1 });
    let term_kind = |e: &TermTestError| match e {
        TermTestError::NotMarginal(_) => 0,
        TermTestError::FTable { .. } => 1,
        TermTestError::RefitNodeCount { .. } => 2,
    };
    let kinds: BTreeSet<usize> = term.iter().map(term_kind).collect();
    assert_eq!(kinds, (0..3).collect(), "one per-term refusal of each kind");

    let mut texts: Vec<String> = linear
        .into_iter()
        .map(|n| linear_note_line(n, &names))
        .collect();
    texts.extend(comparison.note_lines());
    texts.extend(contrast.iter().map(ContrastError::to_string));
    texts.extend(term.iter().map(TermTestError::to_string));
    texts.push(missing_cells_note(&empty).expect("an empty cell"));
    texts.push(lrt_not_converged_note());
    texts.push(lrt_no_basis_note("g:h"));
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
            test: TermTest::Auto,
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

/// The options of a per-term table of `ss_type` running `test`.
fn term_options(ss_type: SsType, test: TermTest) -> TermTestOptions {
    TermTestOptions { ss_type, test }
}

/// The rows of a per-term table and its notes, as one string (`{:?}` of an
/// `f64` round-trips, so equal strings are equal bits).
fn rows_of(t: &super::anova::TermTable) -> String {
    format!("{:?}", (t.ss_type, &t.fixed, &t.notes, &t.coef_names))
}

/// On fits whose rows are χ² tests, a Poisson model with random effects
/// aside (a binomial GLM and GLMM, a Poisson GLM, a negative binomial GLMM,
/// and a Gamma GLM with φ held, whose family has F rows when φ is
/// estimated, a binomial GLMM with trials and a Gamma GLMM), the default test
/// gives the rows an explicit Wald gives, bit for bit, with the reason of
/// each: the family with random effects, or none. An explicit Wald is
/// accepted, and every row is of the table's test.
#[test]
fn the_default_test_gives_the_wald_rows_bit_for_bit() {
    let bern = csv_table(
        include_str!("../../validation/data/simulated/sim_factorial_bern.csv"),
        &["subj", "g", "h"],
    );
    let gamma_data = csv_table(
        include_str!("../../validation/data/simulated/glm_gamma.csv"),
        &["grp"],
    );
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let counts = csv_table(
        include_str!("../../validation/data/simulated/glm_poisson.csv"),
        &["grp"],
    );
    let nb_data = csv_table(
        include_str!("../../validation/data/simulated/sim_nb.csv"),
        &["cluster", "grp"],
    );
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let nb = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let gamma_clustered = csv_table(
        include_str!("../../validation/data/simulated/sim_gamma.csv"),
        &["cluster", "grp"],
    );
    // cbpp: a binomial with trials, `period` a 3-df term.
    let mut cbpp = csv_table(
        include_str!("../../validation/data/empirical/cbpp.csv"),
        &["herd", "period"],
    );
    let column = |t: &Table, name: &str| match &t.columns.iter().find(|(n, _)| n == name).unwrap().1
    {
        Column::Numeric(v) => v.clone(),
        _ => unreachable!("{name} is numeric"),
    };
    let failures = column(&cbpp, "size")
        .iter()
        .zip(column(&cbpp, "incidence"))
        .map(|(n, k)| n - k)
        .collect();
    cbpp.columns
        .push(("failures".into(), Column::Numeric(failures)));
    let fixed = ChoiceReason::NoRandomEffects;
    let cases = [
        ("y ~ g * h + x", &bern, binomial, None, 4, fixed),
        (
            "y ~ g * h + x + (1 | subj)",
            &bern,
            binomial,
            None,
            4,
            ChoiceReason::BinomialMixed,
        ),
        (
            "cbind(incidence, failures) ~ period + (1 | herd)",
            &cbpp,
            binomial,
            None,
            1,
            ChoiceReason::BinomialMixed,
        ),
        ("y ~ x + grp", &gamma_data, gamma, Some(0.5), 2, fixed),
        ("y ~ x * grp", &counts, poisson, None, 3, fixed),
        (
            "y ~ x + grp + (1 | cluster)",
            &nb_data,
            nb,
            None,
            2,
            ChoiceReason::NegativeBinomialMixed,
        ),
        (
            "y ~ x + grp + (1 | cluster)",
            &gamma_clustered,
            gamma,
            None,
            2,
            ChoiceReason::OtherMixed,
        ),
    ];
    for (formula, table, family, dispersion, terms, reason) in cases {
        let mut lo = lower(formula, table, family).unwrap();
        lo.opts.dispersion = dispersion;
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        assert!(fit.converged(), "{formula}");
        for ss_type in [SsType::Type3, SsType::Type2] {
            let auto = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Auto)).unwrap();
            let wald = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Wald)).unwrap();
            let at = format!("{formula}, {ss_type:?}");
            assert_eq!(auto.choice.test, LinearKind::ChiSquared, "{at}");
            assert_eq!(auto.choice.reason, reason, "{at}");
            assert_eq!(wald.choice.test, LinearKind::ChiSquared, "{at}");
            assert_eq!(wald.choice.reason, ChoiceReason::Requested, "{at}");
            assert_eq!(auto.fixed.len(), terms, "{at}");
            assert!(
                wald.fixed.iter().all(|r| r.test.kind == wald.choice.test),
                "{at}"
            );
            assert_eq!(rows_of(&auto), rows_of(&wald), "{at}");
        }
    }
}

/// The default test gives a Poisson model with random effects the
/// likelihood-ratio rows, those of an explicit likelihood-ratio test bit for
/// bit, whatever its size or shape: 15 clusters with prior weights, two
/// groupings of 10 and 4 levels, a 3-df term, and 118 broods with a 2-df
/// term. The same
/// data without the random effect keeps the Wald rows.
#[test]
fn the_default_test_gives_a_poisson_glmm_the_likelihood_ratio_rows() {
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let weighted = csv_table(
        include_str!("../../validation/data/simulated/glmm_poisson.csv"),
        &["g"],
    );
    let two = csv_table(
        include_str!("../../validation/data/simulated/sim_poisson_nested.csv"),
        &["g1", "g2"],
    );
    let grouse = csv_table(
        include_str!("../../validation/data/empirical/grouseticks.csv"),
        &["YEAR", "BROOD"],
    );
    let w = match &weighted.columns.iter().find(|(n, _)| n == "w").unwrap().1 {
        Column::Numeric(v) => v.clone(),
        _ => unreachable!("w is numeric"),
    };
    let both = [SsType::Type3, SsType::Type2];
    let cases = [
        ("y ~ x + (1 | g)", &weighted, Some(w), &both[..]),
        ("y ~ x + (1 | g1) + (1 | g2)", &two, None, &both[..1]),
        ("y ~ x + g2 + (1 | g1)", &two, None, &both[..1]),
        (
            "TICKS ~ YEAR + cHEIGHT + (1 | BROOD)",
            &grouse,
            None,
            &both[..1],
        ),
    ];
    for (formula, table, weights, types) in cases {
        let mut lo = lower(formula, table, poisson).unwrap();
        lo.opts.weights = weights;
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        assert!(fit.converged(), "{formula}");
        for &ss_type in types {
            let at = format!("{formula}, {ss_type:?}");
            let auto = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Auto)).unwrap();
            let lrt = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Lrt)).unwrap();
            assert_eq!(
                (auto.choice.test, auto.choice.reason),
                (LinearKind::LikelihoodRatio, ChoiceReason::PoissonMixed),
                "{at}"
            );
            assert_eq!(lrt.choice.reason, ChoiceReason::Requested, "{at}");
            assert!(
                auto.fixed
                    .iter()
                    .all(|r| r.test.kind == LinearKind::LikelihoodRatio),
                "{at}"
            );
            assert_eq!(rows_of(&auto), rows_of(&lrt), "{at}");
        }
    }
    let lo = lower("y ~ x", &weighted, poisson).unwrap();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let t = term_tests(&lo, &fit, &TermTestOptions::default()).unwrap();
    assert_eq!(
        (t.choice.test, t.choice.reason),
        (LinearKind::ChiSquared, ChoiceReason::NoRandomEffects)
    );
}

/// On fits whose rows are F tests (the Gaussian LMM, OLS, a Gamma GLM with φ
/// estimated) the default gives the F rows, and an explicit Wald or
/// likelihood-ratio test is refused with an error naming the F test.
#[test]
fn an_explicit_test_on_an_f_table_is_refused() {
    let d = csv_table(
        include_str!("../../validation/data/simulated/sim_factorial.csv"),
        &["subj", "g", "h"],
    );
    let g = csv_table(
        include_str!("../../validation/data/simulated/glm_gamma.csv"),
        &["grp"],
    );
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let cases = [
        (
            "y ~ g * h + x + (1 + x | subj)",
            &d,
            Family::Gaussian,
            DenDf::Satterthwaite,
            "Satterthwaite's",
        ),
        (
            "y ~ g * h + x",
            &d,
            Family::Gaussian,
            DenDf::Residual,
            "the residual",
        ),
        ("y ~ x + grp", &g, gamma, DenDf::Residual, "the residual"),
    ];
    for (formula, table, family, den_df, df_words) in cases {
        let lo = lower(formula, table, family).unwrap();
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        assert!(fit.converged(), "{formula}");
        for ss_type in [SsType::Type3, SsType::Type2] {
            let at = format!("{formula}, {ss_type:?}");
            let t = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Auto)).unwrap();
            assert_eq!(t.choice.test, LinearKind::F, "{at}");
            assert_eq!(t.choice.reason, ChoiceReason::FTest, "{at}");
            assert!(t.fixed.iter().all(|r| r.test.kind == LinearKind::F), "{at}");
            for (asked, test_words) in [
                (TermTest::Wald, "Wald chi-squared"),
                (TermTest::Lrt, "likelihood-ratio"),
            ] {
                let e = term_tests(&lo, &fit, &term_options(ss_type, asked)).unwrap_err();
                assert_eq!(e, TermTestError::FTable { asked, den_df }, "{at}");
                let text = e.to_string();
                let f_test = format!("F tests with {df_words} denominator df");
                assert!(text.contains(&f_test), "{at}: {text}");
                assert!(text.contains(test_words), "{at}: {text}");
            }
        }
    }
}

/// Likelihood-ratio rows refit by the lowering's options, so a fit made at
/// another quadrature node count is refused, under an explicit
/// likelihood-ratio test and under the default on a Poisson model with
/// random effects, either way round; its Wald rows are unaffected.
#[test]
fn likelihood_ratio_rows_refuse_a_fit_made_at_another_node_count() {
    let poisson = Family::Poisson {
        link: PoissonLink::Log,
    };
    let table = csv_table(
        include_str!("../../validation/data/simulated/glmm_poisson.csv"),
        &["g"],
    );
    let mut lo = lower("y ~ x + (1 | g)", &table, poisson).unwrap();
    let agq = crate::FitOptions {
        nagq: 5,
        ..lo.opts.clone()
    };
    let at5 = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &agq);
    let at1 = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(at5.converged() && at1.converged());
    for ss_type in [SsType::Type3, SsType::Type2] {
        for test in [TermTest::Lrt, TermTest::Auto] {
            let e = term_tests(&lo, &at5, &term_options(ss_type, test)).unwrap_err();
            assert_eq!(
                e,
                TermTestError::RefitNodeCount { fit: 5, refits: 1 },
                "{ss_type:?} {test:?}"
            );
        }
        let wald = term_tests(&lo, &at5, &term_options(ss_type, TermTest::Wald)).unwrap();
        assert!(
            wald.fixed
                .iter()
                .all(|r| r.test.kind == LinearKind::ChiSquared),
            "{ss_type:?}"
        );
    }
    lo.opts.nagq = 5;
    let e = term_tests(&lo, &at1, &term_options(SsType::Type3, TermTest::Lrt)).unwrap_err();
    assert_eq!(e, TermTestError::RefitNodeCount { fit: 1, refits: 5 });
}

/// `theta_start` turns each vech-packed covariance block into its lower
/// Cholesky factor, vech-packed column-major: a 3 x 3 block (where column-
/// and row-major packing differ) and a 2 x 2 block with a zero variance,
/// whose column stays zero.
#[test]
fn theta_start_packs_each_block_cholesky_factor_column_major() {
    let case = factorial_case("binomial GLMM", " + (1 | subj)", 1, "");
    let (_, mut fit) = case.fit(&case.formula);
    // L = [[2, 0, 0], [0.5, 1.5, 0], [-1, 0.3, 0.8]], D = L Lᵀ.
    fit.varcorr = vec![vec![4.0, 1.0, -2.0, 2.5, -0.05, 1.73], vec![0.0, 0.0, 4.0]];
    let want = [2.0, 0.5, -1.0, 1.5, 0.3, 0.8, 0.0, 0.0, 2.0];
    let got = theta_start(&fit);
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= 1e-12, "{got:?}");
    }
}

/// The absolute part of the band of a likelihood-ratio row against `compare`
/// on its formula pair (gate 2): `TOL_INFERENCE$lt_compare_abs` in
/// `validation/grid/tol.R`, where the measurement is. Change together.
const LT_COMPARE_ABS: f64 = 2e-11;

/// The relative part of that band: `TOL_INFERENCE$lt_compare_rel`. Change
/// together.
const LT_COMPARE_REL: f64 = 4e-12;

/// A numeric column of `table`.
fn numeric(table: &Table, name: &str) -> Vec<f64> {
    match &table.columns.iter().find(|(n, _)| n == name).unwrap().1 {
        Column::Numeric(v) => v.clone(),
        _ => panic!("{name} is not numeric"),
    }
}

/// A model of gate 2: the full formula, with the options every fit of the
/// case takes (node count, weights, held dispersion), the formula pair each
/// Type II row stands for (term, larger, smaller), and the terms whose Type
/// III reduced design is a column drop, with the formula of that drop.
struct LrtCase {
    name: &'static str,
    table: Table,
    formula: String,
    family: Family,
    nagq: u8,
    weights: Option<&'static str>,
    dispersion: Option<f64>,
    type2: Vec<(&'static str, String, String)>,
    type3: Vec<(&'static str, String)>,
}

impl LrtCase {
    /// The cold fit of `formula` with the case's options, converged.
    fn fit(&self, formula: &str) -> (Lowered, Fit) {
        let mut lo = lower(formula, &self.table, self.family).unwrap();
        lo.opts.nagq = self.nagq;
        lo.opts.dispersion = self.dispersion;
        lo.opts.weights = self.weights.map(|w| numeric(&self.table, w));
        let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        assert!(fit.converged(), "{}: {formula}", self.name);
        (lo, fit)
    }
}

/// `y ~ g * h + x` with the random part `re` on `sim_factorial_bern`: car's
/// Type II pairs (`g` and `h` in `g + h + x`, `x` and `g:h` in the full
/// model) and the Type III column drops (`x`, and `g:h`, whose `L` touches
/// its own columns alone), leaving out `skip` (a term on a slope column).
fn factorial_case(name: &'static str, re: &str, nagq: u8, skip: &str) -> LrtCase {
    let f = |rhs: &str| format!("y ~ {rhs}{re}");
    let type2 = vec![
        ("g", f("g + h + x"), f("h + x")),
        ("h", f("g + h + x"), f("g + x")),
        ("x", f("g * h + x"), f("g * h")),
        ("g:h", f("g * h + x"), f("g + h + x")),
    ];
    let type3 = vec![("x", f("g * h")), ("g:h", f("g + h + x"))];
    LrtCase {
        name,
        table: csv_table(
            include_str!("../../validation/data/simulated/sim_factorial_bern.csv"),
            &["subj", "g", "h"],
        ),
        formula: f("g * h + x"),
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        nagq,
        weights: None,
        dispersion: None,
        type2: type2.into_iter().filter(|(t, ..)| *t != skip).collect(),
        type3: type3.into_iter().filter(|(t, _)| *t != skip).collect(),
    }
}

/// `y ~ x + grp` with the random part `re`: each term tested in the full
/// model, Type II and III alike, each a column drop.
fn main_effects_case(
    name: &'static str,
    table: Table,
    re: &str,
    family: Family,
    weights: Option<&'static str>,
    dispersion: Option<f64>,
) -> LrtCase {
    let f = |rhs: &str| format!("y ~ {rhs}{re}");
    LrtCase {
        name,
        table,
        formula: f("x + grp"),
        family,
        nagq: 1,
        weights,
        dispersion,
        type2: vec![("x", f("x + grp"), f("grp")), ("grp", f("x + grp"), f("x"))],
        type3: vec![("x", f("grp")), ("grp", f("x"))],
    }
}

fn lrt_cases() -> Vec<LrtCase> {
    let clustered = |csv: &str| csv_table(csv, &["cluster", "grp"]);
    let nb = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let gamma = Family::Gamma {
        link: GammaLink::Log,
    };
    let offset = "y ~ x + offset(log_exposure) + (1 | cluster)";
    vec![
        factorial_case("binomial GLM", "", 1, ""),
        factorial_case("binomial GLMM", " + (1 | subj)", 1, ""),
        factorial_case("binomial GLMM, nAGQ 5", " + (1 | subj)", 5, ""),
        factorial_case("binomial GLMM, slope on x", " + (1 + x | subj)", 1, "x"),
        main_effects_case(
            "weighted NB GLM",
            csv_table(
                include_str!("../../validation/data/simulated/glm_nb.csv"),
                &["grp"],
            ),
            "",
            nb,
            Some("w"),
            None,
        ),
        main_effects_case(
            "NB GLMM",
            clustered(include_str!("../../validation/data/simulated/sim_nb.csv")),
            " + (1 | cluster)",
            nb,
            None,
            None,
        ),
        main_effects_case(
            "Gamma GLM, φ held",
            csv_table(
                include_str!("../../validation/data/simulated/glm_gamma.csv"),
                &["grp"],
            ),
            "",
            gamma,
            None,
            Some(0.5),
        ),
        main_effects_case(
            "Gamma GLMM",
            clustered(include_str!(
                "../../validation/data/simulated/sim_gamma.csv"
            )),
            " + (1 | cluster)",
            gamma,
            None,
            None,
        ),
        LrtCase {
            name: "Poisson GLMM with an offset",
            table: csv_table(
                include_str!("../../validation/data/simulated/sim_poisson_offset.csv"),
                &["cluster"],
            ),
            formula: offset.into(),
            family: Family::Poisson {
                link: PoissonLink::Log,
            },
            nagq: 1,
            weights: None,
            dispersion: None,
            type2: vec![(
                "x",
                offset.into(),
                "y ~ 1 + offset(log_exposure) + (1 | cluster)".into(),
            )],
            type3: vec![("x", "y ~ 1 + offset(log_exposure) + (1 | cluster)".into())],
        },
    ]
}

/// Gate 2: every Type II likelihood-ratio row equals `compare` on the two
/// formula fits it stands for, and every Type III row whose reduced design is
/// a column drop equals `compare` on that pair, the statistic within
/// `LT_COMPARE_ABS + LT_COMPARE_REL·|T|` (`compare`'s fits start cold, the
/// rows' refits warm from the full fit), the df and the reference exactly.
/// On a binomial GLM, binomial GLMMs by Laplace, by AGQ and with a random
/// slope (whose column the refits keep and move), a weighted negative
/// binomial GLM and a negative binomial GLMM (each refit re-estimates θ, as
/// `compare`'s fits do), a Gamma GLM with φ held, a Gamma GLMM and a Poisson
/// GLMM with an offset: each refit takes the fit's own options.
#[test]
fn likelihood_ratio_rows_equal_compare_on_their_formula_pairs() {
    for case in lrt_cases() {
        let (lo, fit) = case.fit(&case.formula);
        let full = Model::from_fit(&case.formula, &lo, &fit);
        let side = |formula: &str| {
            if formula == case.formula {
                return full.clone();
            }
            let (lo, fit) = case.fit(formula);
            Model::from_fit(formula, &lo, &fit)
        };
        let all: Vec<usize> = (0..lo.p).collect();
        let hyp = type3(&lo, &all);
        let type3_pairs: Vec<(&str, String, String)> = case
            .type3
            .iter()
            .map(|(term, smaller)| {
                let l = &hyp.iter().find(|h| h.term == *term).unwrap().l;
                let touched = lo.p - untouched_columns(l.as_ref()).len();
                assert_eq!(touched, l.nrows(), "{}: {term}'s X·N is a drop", case.name);
                (*term, case.formula.clone(), smaller.clone())
            })
            .collect();
        for (ss_type, pairs) in [(SsType::Type2, &case.type2), (SsType::Type3, &type3_pairs)] {
            let t = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Lrt)).unwrap();
            assert_eq!(t.choice.test, LinearKind::LikelihoodRatio, "{}", case.name);
            assert_eq!(t.choice.reason, ChoiceReason::Requested, "{}", case.name);
            for (term, larger, smaller) in pairs {
                let at = format!("{}, {ss_type:?} {term}", case.name);
                let row = &t.fixed.iter().find(|r| r.label == *term).unwrap().test;
                let c = compare(&side(larger), &side(smaller), &OPTS).unwrap();
                let want = lr(&c);
                assert_eq!(row.kind, LinearKind::LikelihoodRatio, "{at}");
                assert_eq!(
                    want.reference,
                    Reference::ChiSquared { df: want.df },
                    "{at}"
                );
                assert_eq!(row.reference, want.reference, "{at}");
                assert_eq!(row.num_df, want.df as f64, "{at}");
                assert!(row.notes.is_empty(), "{at}: {:?}", row.notes);
                assert_eq!(
                    row.p_value.to_bits(),
                    lr_p_value(row.statistic, &row.reference).to_bits(),
                    "{at}"
                );
                let gap = (row.statistic - want.statistic).abs();
                let band = LT_COMPARE_ABS + LT_COMPARE_REL * want.statistic.abs();
                eprintln!(
                    "likelihood-ratio gate 2 {at}: row {:.12e} compare {:.12e} gap {gap:.3e} rel {:.3e} \
                     band {band:.3e} bits {:016x}",
                    row.statistic,
                    want.statistic,
                    gap / want.statistic.abs(),
                    row.statistic.to_bits()
                );
                assert!(
                    gap <= band,
                    "{at}: {} against {}",
                    row.statistic,
                    want.statistic
                );
            }
        }
    }
}

/// A term whose test would leave out or mix a random-slope column (`x`, in
/// `(1 + x | subj)`) gets its Wald row, bit for bit, with
/// [`LinearNote::RandomSlopeColumn`], on both types; the table's other rows
/// are likelihood-ratio rows and the table's test stays the likelihood-ratio
/// test.
#[test]
fn a_slope_touching_term_gets_its_wald_row_and_note() {
    let case = factorial_case("slope", " + (1 + x | subj)", 1, "");
    let (lo, fit) = case.fit(&case.formula);
    for ss_type in [SsType::Type3, SsType::Type2] {
        let lrt = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Lrt)).unwrap();
        let wald = term_tests(&lo, &fit, &term_options(ss_type, TermTest::Wald)).unwrap();
        assert_eq!(lrt.choice.test, LinearKind::LikelihoodRatio);
        for (row, w) in lrt.fixed.iter().zip(&wald.fixed) {
            let at = format!("{ss_type:?} {}", row.label);
            if row.label == "x" {
                let mut want = w.test.clone();
                want.notes.push(LinearNote::RandomSlopeColumn);
                assert_eq!(format!("{:?}", row.test), format!("{want:?}"), "{at}");
            } else {
                assert_eq!(row.test.kind, LinearKind::LikelihoodRatio, "{at}");
                assert!(row.test.statistic.is_finite(), "{at}: {:?}", row.test);
                assert!(row.test.notes.is_empty(), "{at}: {:?}", row.test);
            }
        }
    }
}

/// A refit that did not converge, the smaller or (a Type II set) the larger,
/// gives a NaN row with [`LinearNote::ReducedNotConverged`] on the term's df;
/// a fit that did not converge gives every row NaN with no row note, and the
/// table one note that says why.
#[test]
fn a_fit_or_refit_that_did_not_converge_gives_nan_rows() {
    let case = factorial_case("binomial GLM", "", 1, "");
    let (_, larger) = case.fit("y ~ g * h + x");
    let (_, smaller) = case.fit("y ~ g * h");
    let ok = lrt_test(&larger, &smaller, 1);
    assert!(ok.statistic > 0.0 && ok.notes.is_empty(), "{ok:?}");
    let failed = |formula: &str| {
        let (lo, mut fit) = case.fit(formula);
        fit.diagnostics.converged = false;
        (lo, fit)
    };
    let (_, failed_smaller) = failed("y ~ g * h");
    let (lo, failed_larger) = failed("y ~ g * h + x");
    for t in [
        lrt_test(&larger, &failed_smaller, 1),
        lrt_test(&failed_larger, &smaller, 1),
    ] {
        assert!(t.statistic.is_nan() && t.p_value.is_nan(), "{t:?}");
        assert_eq!(t.notes, [LinearNote::ReducedNotConverged]);
        assert_eq!(t.kind, LinearKind::LikelihoodRatio);
        assert_eq!(t.num_df, 1.0);
        assert_eq!(t.reference, Reference::ChiSquared { df: 1 });
    }
    for ss_type in [SsType::Type3, SsType::Type2] {
        let t = term_tests(&lo, &failed_larger, &term_options(ss_type, TermTest::Lrt)).unwrap();
        assert_eq!(t.fixed.len(), 4);
        for row in &t.fixed {
            let test = &row.test;
            assert!(test.statistic.is_nan() && test.p_value.is_nan(), "{test:?}");
            assert_eq!(test.kind, LinearKind::LikelihoodRatio);
            assert!(test.notes.is_empty(), "{test:?}");
        }
        assert_eq!(t.notes, [lrt_not_converged_note()], "{ss_type:?}");
    }
}

/// A refit under the hypothesis that ends above the larger model's fit by
/// more than round-off (here the two fits passed the other way round, so
/// `2(ℓ₁ − ℓ₀)` is −3.5e-3) gives the statistic 0, p-value 1, with
/// [`LinearNote::NegativeStatistic`].
#[test]
fn a_refit_above_the_fit_takes_the_negative_statistic_note() {
    let case = factorial_case("binomial GLM", "", 1, "");
    let (_, larger) = case.fit("y ~ g * h + x");
    let (_, smaller) = case.fit("y ~ g * h");
    let t = lrt_test(&smaller, &larger, 1);
    assert!(2.0 * (smaller.loglik - larger.loglik) < -1e-3, "{t:?}");
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0), "{t:?}");
    assert_eq!(t.notes, [LinearNote::NegativeStatistic]);
    assert_eq!(t.kind, LinearKind::LikelihoodRatio);
}

/// Type II likelihood-ratio rows read the fit's column sets, never its
/// covariance: with every entry of `vcov` NaN (a fit whose Hessian and
/// fallback both failed) the rows are the same, bit for bit.
#[test]
fn type_ii_lrt_rows_do_not_read_the_covariance() {
    let case = factorial_case("binomial GLMM", " + (1 | subj)", 1, "");
    let (lo, fit) = case.fit(&case.formula);
    let opts = term_options(SsType::Type2, TermTest::Lrt);
    let want = term_tests(&lo, &fit, &opts).unwrap();
    let (_, mut blind) = case.fit(&case.formula);
    for row in &mut blind.vcov {
        row.fill(f64::NAN);
    }
    let got = term_tests(&lo, &blind, &opts).unwrap();
    assert!(want.fixed.iter().all(|r| r.test.statistic.is_finite()));
    assert_eq!(format!("{:?}", got.fixed), format!("{:?}", want.fixed));
}

/// The Type III rows whose reduced design mixes columns (`g` and `h`, whose
/// `L` reaches into `g:h`), and the column drops (`x`, `g:h`), against afex's
/// `mixed(type = 3, method = "LRT")` on the same model, frozen in
/// `anova_lrt3_glmm_sim_factorial_bern`: the statistic within the
/// registry's `lrt` band, the df exactly.
#[test]
fn type_iii_lrt_rows_match_afex_on_the_frozen_glmm() {
    let json = crate::inference::test_refs::load(
        crate::inference::test_refs::Kind::Anova,
        "anova_lrt3_glmm_sim_factorial_bern",
    );
    let case = factorial_case("binomial GLMM", " + (1 | subj)", 1, "");
    let (lo, fit) = case.fit(&case.formula);
    let t = term_tests(&lo, &fit, &term_options(SsType::Type3, TermTest::Lrt)).unwrap();
    let rows = json["rows"].as_array().expect("frozen rows");
    assert_eq!(rows.len(), t.fixed.len());
    for (row, want) in t.fixed.iter().zip(rows) {
        assert_eq!(row.label, want["term"].as_str().unwrap());
        assert_eq!(
            row.test.num_df,
            want["df"].as_f64().unwrap(),
            "{}",
            row.label
        );
        let gap = (row.test.statistic - want["statistic"].as_f64().unwrap()).abs();
        assert!(
            gap <= LRT_ABS,
            "{}: {:?} off afex by {gap:e}",
            row.label,
            row.test
        );
    }
}

/// The Type III row `h` of the binomial GLMM on `sim_factorial_bern` (its
/// `L` reaches into `g:h`, so the reduced design is `X·N`, not a column
/// drop) is the refit on the null basis of the unrounded hypothesis
/// ([`type3_unrounded`]), not of [`type3`]'s rounded one: the table's
/// statistic equals the refit built here on the unrounded basis to
/// round-off, and the rounded basis gives a statistic far enough from it
/// (3.1e-8 measured) for the pin to tell the two apart.
#[test]
fn type_iii_lrt_h_row_refits_on_the_unrounded_null_basis() {
    let case = factorial_case("binomial GLMM", " + (1 | subj)", 1, "");
    let (lo, fit) = case.fit(&case.formula);
    assert!(!fit.aliased().iter().any(|&a| a), "a full-rank design");
    let table = term_tests(&lo, &fit, &term_options(SsType::Type3, TermTest::Lrt)).unwrap();
    let h = lo
        .terms
        .iter()
        .position(|t| t.label == "h")
        .expect("a term h");
    assert_eq!(table.fixed[h].label, "h");
    let all: Vec<usize> = (0..lo.p).collect();
    let rounded = type3(&lo, &all).swap_remove(h).l;
    let exact = type3_unrounded(&lo, &all).swap_remove(h).l;
    let statistic = |l: &Mat<f64>| {
        let basis = type3_null_basis(rounded.as_ref(), l.as_ref(), &[]).unwrap();
        let kept = untouched_columns(rounded.as_ref()).len();
        assert!(basis.ncols() > kept, "N mixes columns");
        let reduced = refit_on_basis(&lo, &fit, &all, &basis, &[Vec::new()]);
        lrt_test(&fit, &reduced, lo.p - basis.ncols()).statistic
    };
    let (g_exact, g_rounded) = (statistic(&exact), statistic(&rounded));
    let got = table.fixed[h].test.statistic;
    eprintln!(
        "type III h: table {got:e}, unrounded basis {g_exact:e}, rounded basis {g_rounded:e}"
    );
    assert!(
        (got - g_exact).abs() <= 1e-10,
        "the h row {got} is not the unrounded basis's refit {g_exact} (the rounded basis gives \
         {g_rounded})"
    );
    assert!(
        (g_rounded - g_exact).abs() >= 1e-8,
        "the two bases give {g_rounded} and {g_exact}: too close for this pin to tell apart"
    );
}

/// The band of a likelihood-ratio statistic against a frozen reference,
/// absolute: `TOL_INFERENCE$lrt_abs` in `validation/grid/tol.R` (also
/// copied in `tests/oracle_support/mod.rs`). Change together.
const LRT_ABS: f64 = 4e-4;
