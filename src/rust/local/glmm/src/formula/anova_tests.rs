//! The two-fit comparison against its crate-private engines: the exact
//! restricted likelihood-ratio test called directly on the same numbers, its
//! refusal reasons, the likelihood-ratio tails, and the refusal texts; the
//! per-term table's Type II metric on the Gaussian LMM; its test choice; its
//! rows' partial η²; its likelihood-ratio rows against `compare` on their formula pairs; and the
//! comparison checks and per-term tests that need a side or a fit changed by
//! hand, which no fitted `formula::Model` carries.

use crate::test_support::fit_warm_wo;
use crate::FitOptions;
use std::collections::BTreeSet;
use std::mem::discriminant;

use faer::Mat;

use super::anova::{
    compare_stored, linear_note_line, lrt_no_basis_note, lrt_not_converged_note, lrt_test, nesting,
    refit_on_basis, term_tests_cached, theta_start, ChoiceReason, CompareError, CompareOptions,
    Comparison, ComparisonTest, Differs, Model, ModelInfo, RefitCache, Side, SsType, TermTable,
    TermTest, TermTestError, TermTestOptions,
};
use super::hypotheses::{
    check_marginal, missing_cells_note, type2_vcov, type2_xtx, type3, type3_null_basis,
    type3_unrounded, untouched_columns, TermHypothesis,
};
use crate::formula::{lower, Column, Lowered, Table};
use crate::inference::contrast::{ContrastError, NoDfReason};
use crate::inference::evd::EvdError;
use crate::inference::lr::lr_p_value;
use crate::inference::rlrt::{exact_rlrt, RlrtConfig, RlrtRefusal, RLRT_MAX_LEVELS};
use crate::inference::{
    DenDf, LinearKind, LinearNote, LinearTest, LrNote, LrTest, Objective, Reference,
};
use crate::test_support::csv_table;
use crate::{BinomialLink, Family, Fit, GammaLink, NegBinomialLink, PoissonLink};

const OPTS: CompareOptions = CompareOptions {
    nsim: 2000,
    seed: 20_261_012,
};

/// A converged Gaussian fit of `formula` with `reml` and `weights`, as a
/// `Model`.
fn side(formula: &str, table: &Table, reml: Option<bool>, weights: Option<Vec<f64>>) -> Model {
    let mut lo = lower(formula, table, Family::Gaussian).unwrap();
    let fit_opts = FitOptions {
        reml,
        ..FitOptions::default()
    };
    lo.weights = weights;
    let fit = fit_warm_wo(
        &lo.x,
        &lo.y,
        lo.n,
        lo.p,
        &lo.model,
        &lo.ids,
        None,
        lo.weights.as_deref(),
        lo.offset.as_deref(),
        &fit_opts,
    );
    assert!(fit.converged(), "{formula} did not converge");
    Model::from_fit(formula, &lo, &fit_opts, &fit)
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
fn direct(ols: &Model, full: &Model) -> crate::inference::rlrt::RlrtResult {
    let (counts, s, c) = full.design_summary.as_ref().expect("inside the envelope");
    let cfg = RlrtConfig {
        nsim: OPTS.nsim,
        seed: OPTS.seed,
        parallel_inner: false,
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
fn the_envelope_pair_gets_the_exact_test_as_exact_rlrt_computes_it() {
    let (ols, full) = dyestuff();
    let c = compare_stored(&ols, &full, &OPTS).unwrap();
    let t = lr(&c);
    let r = direct(&ols, &full);
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
fn groups_the_fixed_part_absorbs_get_the_mixture_with_the_refusal_reason() {
    let d = lmm_intercept();
    let ols = side("y ~ g", &d, Some(true), None);
    let full = side("y ~ g + (1 | g)", &d, None, None);
    assert!(full.design_summary.is_some());
    let c = compare_stored(&ols, &full, &OPTS).unwrap();
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

/// `levels` declared levels of `g`, three rows each except the levels in
/// `empty`, which have none; a numeric `x` and a response with a group
/// effect.
fn three_per_level(levels: usize, empty: &[usize]) -> Table {
    let mut st = 29u64;
    let mut unif = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 33) as f64 / (1u64 << 31) as f64) - 1.0
    };
    let (mut x, mut y, mut codes) = (Vec::new(), Vec::new(), Vec::new());
    for a in (0..levels).filter(|a| !empty.contains(a)) {
        let re = 0.6 * unif();
        for _ in 0..3 {
            let xi = unif();
            x.push(xi);
            y.push(0.5 + 0.4 * xi + re + 0.3 * unif());
            codes.push(a as u32);
        }
    }
    let n = y.len();
    let levels = (0..levels).map(|a| format!("g{a:03}")).collect();
    Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("x".into(), Column::Numeric(x)),
            ("g".into(), Column::Factor { levels, codes }),
        ],
        n,
    }
}

const FEW_DRAWS: CompareOptions = CompareOptions {
    nsim: 200,
    seed: OPTS.seed,
};

/// The larger side of `y ~ x` against `y ~ x + (1 | g)`, and the comparison.
fn level_pair(d: &Table) -> (Model, Comparison) {
    let ols = side("y ~ x", d, Some(true), None);
    let full = side("y ~ x + (1 | g)", d, None, None);
    let c = compare_stored(&ols, &full, &FEW_DRAWS).unwrap();
    (full, c)
}

/// The exact test runs up to 100 levels with rows. A declared level with no
/// rows does not count: the frontend drops it, so 101 declared levels with one
/// empty are 100 levels.
#[test]
fn the_exact_test_runs_up_to_100_levels_with_rows() {
    let exact = Reference::ExactRlrt {
        nsim: FEW_DRAWS.nsim,
        seed: FEW_DRAWS.seed,
    };
    for d in [three_per_level(100, &[]), three_per_level(101, &[50])] {
        let (full, c) = level_pair(&d);
        let (counts, _, _) = full.design_summary.as_ref().expect("inside the envelope");
        assert_eq!(counts.len(), 100);
        assert!(counts.iter().all(|&n| n > 0.0));
        let t = lr(&c);
        assert_eq!(t.reference, exact);
        assert!(t.statistic > 0.0);
        assert!(t.notes.is_empty(), "{:?}", t.notes);
    }
}

/// Above 100 levels with rows the pair gets the 50:50 mixture, with the
/// refusal's reason in the note.
#[test]
fn above_100_levels_with_rows_the_mixture_is_used_with_the_reason() {
    let (full, c) = level_pair(&three_per_level(101, &[]));
    assert!(full.design_summary.is_some());
    let t = lr(&c);
    let reason = RlrtRefusal::TooManyLevels.reason();
    assert!(reason.contains(&format!("more than {RLRT_MAX_LEVELS} groups")));
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(
        t.notes,
        vec![
            LrNote::ExactRlrtRefused { reason },
            LrNote::BoundaryMixture { approximate: false }
        ]
    );
    assert!(t.statistic > 0.0);
    assert_eq!(
        t.p_value.to_bits(),
        lr_p_value(t.statistic, &t.reference).to_bits()
    );
    assert_eq!(
        c.note_lines()[0],
        format!(
            "The exact restricted likelihood-ratio test was not computed: {reason}. \
             The 50:50 mixture is used."
        )
    );
}

/// A side whose `re_groups` does not describe its random part (one grouping
/// per variance block, one term per block dimension) is refused by name,
/// never compared on the raw statistic.
#[test]
fn groupings_that_do_not_match_the_variance_blocks_are_refused() {
    let (ols, full) = dyestuff();
    let refusal = |a: &Model, b: &Model| compare_stored(a, b, &OPTS).err();
    let mut cleared = full.clone();
    cleared.re_groups.clear();
    assert_eq!(
        refusal(&ols, &cleared),
        Some(CompareError::ReGroups { side: Side::Second })
    );
    let mut wide = full.clone();
    wide.re_groups[0].1.push("x".into());
    assert_eq!(
        refusal(&ols, &wide),
        Some(CompareError::ReGroups { side: Side::Second })
    );
    let mut stray = ols.clone();
    stray.re_groups = full.re_groups.clone();
    assert_eq!(
        refusal(&stray, &full),
        Some(CompareError::ReGroups { side: Side::First })
    );
    let mut odd = full.clone();
    odd.varcorr[0] = vec![1.0, 0.0];
    assert_eq!(
        refusal(&ols, &odd),
        Some(CompareError::ReGroups { side: Side::Second })
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
        CompareError::ReGroups { side: Side::First },
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
        CompareError::ReGroups { .. } => 14,
    };
    let kinds: BTreeSet<usize> = every.iter().map(kind).collect();
    assert_eq!(kinds, (0..15).collect(), "one refusal of each kind");
    every
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
        LinearNote::NegativeStatistic,
    ];
    let names = ["(Intercept)", "x", "z"].map(String::from);
    linear.extend(reasons.iter().map(|&reason| LinearNote::NoDf { reason }));
    let linear_kind = |n: &LinearNote| match n {
        LinearNote::AliasedColumn { .. } => 0,
        LinearNote::RankDeficientL { .. } => 1,
        LinearNote::NoDf { .. } => 2,
        LinearNote::ReducedNotConverged => 3,
        LinearNote::NegativeStatistic => 4,
    };
    let kinds: BTreeSet<usize> = linear.iter().map(linear_kind).collect();
    assert_eq!(kinds, (0..5).collect(), "one row note of each kind");

    let refusals = [
        RlrtRefusal::NoResidualDf,
        RlrtRefusal::NoGroupVariation,
        RlrtRefusal::Eigen(EvdError::NoConvergence),
        RlrtRefusal::MalformedSummary,
        RlrtRefusal::TooManyLevels,
    ];
    let refusal_kind = |r: &RlrtRefusal| match r {
        RlrtRefusal::NoResidualDf => 0,
        RlrtRefusal::NoGroupVariation => 1,
        RlrtRefusal::Eigen(_) => 2,
        RlrtRefusal::MalformedSummary => 3,
        RlrtRefusal::TooManyLevels => 4,
    };
    let kinds: BTreeSet<usize> = refusals.iter().map(refusal_kind).collect();
    assert_eq!(
        kinds,
        (0..5).collect(),
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
/// given, so none names a Rust path, field or identifier, and none carries a
/// port's call syntax (the comparison function, the REML argument): each
/// says what is wrong and what to do in plain words.
#[test]
fn no_note_text_carries_rust_syntax() {
    for field in ["(opts.satterthwaite)", "set fit.dispersion first"] {
        assert!(rust_syntax(field).is_some(), "{field}");
    }
    assert_eq!(rust_syntax("e.g. a value, i.e. one of 1.5 or 2."), None);
    let port_syntax = [
        "compare(",
        "REML =",
        "reml=",
        "REML=FALSE",
        "REML=TRUE",
        "reml = False",
        "reml = True",
    ];
    let mut texts = every_note_text();
    texts.extend(every_refusal().iter().map(CompareError::to_string));
    assert!(texts.iter().all(|t| !t.is_empty()));
    for text in texts {
        assert_eq!(rust_syntax(&text), None, "{text}");
        for s in port_syntax {
            assert!(!text.contains(s), "{s}: {text}");
        }
    }
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
    let fit = fit_warm_wo(
        &lo.x,
        &lo.y,
        lo.n,
        lo.p,
        &lo.model,
        &lo.ids,
        None,
        lo.weights.as_deref(),
        lo.offset.as_deref(),
        &FitOptions::default(),
    );
    assert!(fit.converged());
    let retained: Vec<usize> = (0..lo.p).filter(|&j| !fit.aliased()[j]).collect();
    let t2 = term_tests_cached(
        &lo,
        &FitOptions::default(),
        &fit,
        &TermTestOptions {
            ss_type: SsType::Type2,
            test: TermTest::Auto,
        },
        &mut RefitCache::default(),
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
        let lo = lower(formula, table, family).unwrap();
        let fit_opts = FitOptions {
            dispersion,
            ..FitOptions::default()
        };
        let fit = fit_warm_wo(
            &lo.x,
            &lo.y,
            lo.n,
            lo.p,
            &lo.model,
            &lo.ids,
            None,
            lo.weights.as_deref(),
            lo.offset.as_deref(),
            &fit_opts,
        );
        assert!(fit.converged(), "{formula}");
        for ss_type in [SsType::Type3, SsType::Type2] {
            let auto = term_tests_cached(
                &lo,
                &fit_opts,
                &fit,
                &term_options(ss_type, TermTest::Auto),
                &mut RefitCache::default(),
            )
            .unwrap();
            let wald = term_tests_cached(
                &lo,
                &fit_opts,
                &fit,
                &term_options(ss_type, TermTest::Wald),
                &mut RefitCache::default(),
            )
            .unwrap();
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
        lo.weights = weights;
        let fit = fit_warm_wo(
            &lo.x,
            &lo.y,
            lo.n,
            lo.p,
            &lo.model,
            &lo.ids,
            None,
            lo.weights.as_deref(),
            lo.offset.as_deref(),
            &FitOptions::default(),
        );
        assert!(fit.converged(), "{formula}");
        for &ss_type in types {
            let at = format!("{formula}, {ss_type:?}");
            let auto = term_tests_cached(
                &lo,
                &FitOptions::default(),
                &fit,
                &term_options(ss_type, TermTest::Auto),
                &mut RefitCache::default(),
            )
            .unwrap();
            let lrt = term_tests_cached(
                &lo,
                &FitOptions::default(),
                &fit,
                &term_options(ss_type, TermTest::Lrt),
                &mut RefitCache::default(),
            )
            .unwrap();
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
    let fit = fit_warm_wo(
        &lo.x,
        &lo.y,
        lo.n,
        lo.p,
        &lo.model,
        &lo.ids,
        None,
        lo.weights.as_deref(),
        lo.offset.as_deref(),
        &FitOptions::default(),
    );
    let t = term_tests_cached(
        &lo,
        &FitOptions::default(),
        &fit,
        &TermTestOptions::default(),
        &mut RefitCache::default(),
    )
    .unwrap();
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
            "a Gaussian mixed model are F tests",
        ),
        (
            "y ~ g * h + x",
            &d,
            Family::Gaussian,
            DenDf::Residual,
            "this fit are F tests on the residual df",
        ),
        (
            "y ~ x + grp",
            &g,
            gamma,
            DenDf::Residual,
            "this fit are F tests on the residual df",
        ),
    ];
    for (formula, table, family, den_df, df_words) in cases {
        let lo = lower(formula, table, family).unwrap();
        let fit = fit_warm_wo(
            &lo.x,
            &lo.y,
            lo.n,
            lo.p,
            &lo.model,
            &lo.ids,
            None,
            lo.weights.as_deref(),
            lo.offset.as_deref(),
            &FitOptions::default(),
        );
        assert!(fit.converged(), "{formula}");
        for ss_type in [SsType::Type3, SsType::Type2] {
            let at = format!("{formula}, {ss_type:?}");
            let t = term_tests_cached(
                &lo,
                &FitOptions::default(),
                &fit,
                &term_options(ss_type, TermTest::Auto),
                &mut RefitCache::default(),
            )
            .unwrap();
            assert_eq!(t.choice.test, LinearKind::F, "{at}");
            assert_eq!(t.choice.reason, ChoiceReason::FTest, "{at}");
            assert!(t.fixed.iter().all(|r| r.test.kind == LinearKind::F), "{at}");
            for (asked, test_words) in [
                (TermTest::Wald, "Wald chi-squared"),
                (TermTest::Lrt, "likelihood-ratio"),
            ] {
                let e = term_tests_cached(
                    &lo,
                    &FitOptions::default(),
                    &fit,
                    &term_options(ss_type, asked),
                    &mut RefitCache::default(),
                )
                .unwrap_err();
                assert_eq!(e, TermTestError::FTable { asked, den_df }, "{at}");
                let text = e.to_string();
                let f_test = format!("per-term tests on {df_words};");
                assert!(text.contains(&f_test), "{at}: {text}");
                assert!(text.contains(test_words), "{at}: {text}");
            }
        }
    }
}

/// Likelihood-ratio rows refit by the options `term_tests_cached` is given, so a
/// fit made at another quadrature node count is refused, under an explicit
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
    let lo = lower("y ~ x + (1 | g)", &table, poisson).unwrap();
    let agq = crate::FitOptions {
        nagq: 5,
        ..FitOptions::default()
    };
    let at5 = fit_warm_wo(
        &lo.x,
        &lo.y,
        lo.n,
        lo.p,
        &lo.model,
        &lo.ids,
        None,
        lo.weights.as_deref(),
        lo.offset.as_deref(),
        &agq,
    );
    let at1 = fit_warm_wo(
        &lo.x,
        &lo.y,
        lo.n,
        lo.p,
        &lo.model,
        &lo.ids,
        None,
        lo.weights.as_deref(),
        lo.offset.as_deref(),
        &FitOptions::default(),
    );
    assert!(at5.converged() && at1.converged());
    for ss_type in [SsType::Type3, SsType::Type2] {
        for test in [TermTest::Lrt, TermTest::Auto] {
            let e = term_tests_cached(
                &lo,
                &FitOptions::default(),
                &at5,
                &term_options(ss_type, test),
                &mut RefitCache::default(),
            )
            .unwrap_err();
            assert_eq!(
                e,
                TermTestError::RefitNodeCount { fit: 5, refits: 1 },
                "{ss_type:?} {test:?}"
            );
        }
        let wald = term_tests_cached(
            &lo,
            &FitOptions::default(),
            &at5,
            &term_options(ss_type, TermTest::Wald),
            &mut RefitCache::default(),
        )
        .unwrap();
        assert!(
            wald.fixed
                .iter()
                .all(|r| r.test.kind == LinearKind::ChiSquared),
            "{ss_type:?}"
        );
    }
    let fit_opts = FitOptions {
        nagq: 5,
        ..FitOptions::default()
    };
    let e = term_tests_cached(
        &lo,
        &fit_opts,
        &at1,
        &term_options(SsType::Type3, TermTest::Lrt),
        &mut RefitCache::default(),
    )
    .unwrap_err();
    assert_eq!(e, TermTestError::RefitNodeCount { fit: 1, refits: 5 });
}

/// Partial η² is `effectsize::F_to_eta2` of the row's own F and df,
/// `F·df1 / (F·df1 + df2)`, bit for bit on every row of an OLS Type III table
/// and of a Gaussian LMM's Satterthwaite table; NaN on every row of a Wald χ²
/// table, of a likelihood-ratio table, and of an F table with no df (the
/// Satterthwaite pass off).
#[test]
fn partial_eta2_follows_f_to_eta2() {
    fn table(
        formula: &str,
        t: &Table,
        family: Family,
        satterthwaite: bool,
        test: TermTest,
    ) -> TermTable {
        let fit_opts = FitOptions {
            satterthwaite,
            ..FitOptions::default()
        };
        let m = crate::formula::fit(formula, t, family, &fit_opts).unwrap();
        assert!(m.fit().converged(), "{formula}");
        super::anova::term_tests(&m, &term_options(SsType::Type3, test)).unwrap()
    }
    let d = csv_table(
        include_str!("../../validation/data/simulated/sim_factorial.csv"),
        &["subj", "g", "h"],
    );
    let bern = csv_table(
        include_str!("../../validation/data/simulated/sim_factorial_bern.csv"),
        &["subj", "g", "h"],
    );
    let binomial = Family::Binomial {
        link: BinomialLink::Logit,
    };
    let (ols, lmm) = ("y ~ g * h + x", "y ~ g * h + x + (1 + x | subj)");
    let gaussian = Family::Gaussian;
    for (formula, den_kind) in [(ols, DenDf::Residual), (lmm, DenDf::Satterthwaite)] {
        let t = table(formula, &d, gaussian, true, TermTest::Auto);
        assert_eq!(t.fixed.len(), 4, "{formula}");
        for row in &t.fixed {
            let Reference::F {
                num_df,
                den_df,
                den_df_kind,
            } = row.test.reference
            else {
                panic!("{formula} {}: {:?}", row.label, row.test)
            };
            assert_eq!(den_df_kind, den_kind, "{formula} {}", row.label);
            assert!(den_df.is_finite(), "{formula} {}", row.label);
            let f_df1 = row.test.statistic * num_df;
            let want = f_df1 / (f_df1 + den_df);
            assert_eq!(
                row.eta2_partial().to_bits(),
                want.to_bits(),
                "{formula} {}",
                row.label
            );
        }
    }
    let untested = [
        (
            table(ols, &bern, binomial, true, TermTest::Wald),
            LinearKind::ChiSquared,
        ),
        (
            table(ols, &bern, binomial, true, TermTest::Lrt),
            LinearKind::LikelihoodRatio,
        ),
        (
            table(lmm, &d, gaussian, false, TermTest::Auto),
            LinearKind::F,
        ),
    ];
    for (t, kind) in untested {
        assert_eq!(t.fixed.len(), 4, "{kind:?}");
        for row in &t.fixed {
            assert_eq!(row.test.kind, kind, "{}", row.label);
            assert!(row.eta2_partial().is_nan(), "{kind:?} {}", row.label);
        }
    }
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
    /// The fit options of every fit of the case: its node count and held
    /// dispersion.
    fn options(&self) -> FitOptions {
        FitOptions {
            nagq: self.nagq,
            dispersion: self.dispersion,
            ..FitOptions::default()
        }
    }

    /// The cold fit of `formula` with the case's options, converged.
    fn fit(&self, formula: &str) -> (Lowered, Fit) {
        let mut lo = lower(formula, &self.table, self.family).unwrap();
        let fit_opts = self.options();
        lo.weights = self.weights.map(|w| numeric(&self.table, w));
        let fit = fit_warm_wo(
            &lo.x,
            &lo.y,
            lo.n,
            lo.p,
            &lo.model,
            &lo.ids,
            None,
            lo.weights.as_deref(),
            lo.offset.as_deref(),
            &fit_opts,
        );
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
        let full = Model::from_fit(&case.formula, &lo, &case.options(), &fit);
        let side = |formula: &str| {
            if formula == case.formula {
                return full.clone();
            }
            let (lo, fit) = case.fit(formula);
            Model::from_fit(formula, &lo, &case.options(), &fit)
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
            let t = term_tests_cached(
                &lo,
                &case.options(),
                &fit,
                &term_options(ss_type, TermTest::Lrt),
                &mut RefitCache::default(),
            )
            .unwrap();
            assert_eq!(t.choice.test, LinearKind::LikelihoodRatio, "{}", case.name);
            assert_eq!(t.choice.reason, ChoiceReason::Requested, "{}", case.name);
            for (term, larger, smaller) in pairs {
                let at = format!("{}, {ss_type:?} {term}", case.name);
                let row = &t.fixed.iter().find(|r| r.label == *term).unwrap().test;
                let c = compare_stored(&side(larger), &side(smaller), &OPTS).unwrap();
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

/// `(x || g)` is nested in `(1 + x | g)`: every term of the smaller random
/// part lies inside one term of the larger. On sleepstudy the two ML fits are
/// one parameter apart, the correlation.
#[test]
fn uncorrelated_is_nested_in_correlated() {
    assert_eq!(
        nesting("y ~ x + (x || g)", "y ~ x + (1 + x | g)"),
        Ok(super::anova::Nesting::Nested {
            larger: Side::Second,
            differs: Differs::Random,
        })
    );
    assert_eq!(
        nesting("y ~ x + (1 | g) + (0 + x | g)", "y ~ x + (x || g)"),
        Ok(super::anova::Nesting::Same)
    );
    assert!(nesting("y ~ x + (0 + x | g)", "y ~ x + (1 | g)").is_err());
    let d = sleepstudy();
    let uncorr = side("Reaction ~ Days + (Days || Subject)", &d, Some(false), None);
    let corr = side("Reaction ~ Days + (Days | Subject)", &d, Some(false), None);
    let c = cmp(&uncorr, &corr).unwrap();
    assert_eq!(c.larger, Side::Second);
    assert_eq!(lr(&c).df, 1);
}

/// The likelihood-ratio row of `h` in `y ~ h + x + (x || subj)` refits the
/// model without `h`, which moves the slope from column 2 to column 1; the
/// refit rewrites the terms with the slopes, so the shape check accepts it and
/// the row is a likelihood-ratio row with a finite statistic and no note.
#[test]
fn lrt_rows_refit_an_uncorrelated_model() {
    let case = factorial_case("uncorrelated", "", 1, "");
    let (lo, fit) = case.fit("y ~ h + x + (x || subj)");
    assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![2]);
    let t = term_tests_cached(
        &lo,
        &case.options(),
        &fit,
        &term_options(SsType::Type2, TermTest::Lrt),
        &mut RefitCache::default(),
    )
    .unwrap();
    let row = t
        .fixed
        .iter()
        .find(|r| r.label == "h")
        .expect("a row for h");
    assert_eq!(row.test.kind, LinearKind::LikelihoodRatio, "{:?}", row.test);
    assert!(row.test.statistic.is_finite(), "{:?}", row.test);
    assert!(row.test.notes.is_empty(), "{:?}", row.test);
}

/// Every row of a likelihood-ratio table is a likelihood-ratio row, with a
/// finite statistic and no note, when a term's columns carry a random slope:
/// on `(1 + x | subj)` the rows of `x` drop the slope column, and on
/// `(1 + g | subj)` the Type III rows of `g` and `h` fit `X·N`, the `N` of
/// `g` mixing `g`'s slope columns (`h`'s leaves them unit columns). Such a
/// slope column moves to a
/// random-effect-only column of the refit, so the reduced model keeps the
/// random slope. Both types.
#[test]
fn a_slope_term_gets_a_likelihood_ratio_row() {
    for re in [" + (1 + x | subj)", " + (1 + g | subj)"] {
        let case = factorial_case("slope", re, 1, "");
        let (lo, fit) = case.fit(&case.formula);
        if re.contains("g |") {
            let all: Vec<usize> = (0..lo.p).collect();
            let g_cols = &lo.terms.iter().find(|t| t.label == "g").unwrap().columns;
            let exact = type3_unrounded(&lo, &all);
            for (h, e) in type3(&lo, &all).iter().zip(&exact) {
                if h.term != "g" && h.term != "h" {
                    continue;
                }
                let kept = untouched_columns(h.l.as_ref());
                let basis = type3_null_basis(h.l.as_ref(), e.l.as_ref()).unwrap();
                assert!(basis.ncols() > kept.len(), "{}: N mixes columns", h.term);
                assert_eq!(
                    g_cols.clone().any(|c| !kept.contains(&c)),
                    h.term == "g",
                    "{}: L touches a slope column",
                    h.term
                );
            }
        }
        for ss_type in [SsType::Type3, SsType::Type2] {
            let lrt = term_tests_cached(
                &lo,
                &case.options(),
                &fit,
                &term_options(ss_type, TermTest::Lrt),
                &mut RefitCache::default(),
            )
            .unwrap();
            assert_eq!(lrt.choice.test, LinearKind::LikelihoodRatio);
            for row in &lrt.fixed {
                let at = format!("{re} {ss_type:?} {}", row.label);
                assert_eq!(row.test.kind, LinearKind::LikelihoodRatio, "{at}");
                assert!(row.test.statistic.is_finite(), "{at}: {:?}", row.test);
                assert!(row.test.notes.is_empty(), "{at}: {:?}", row.test);
            }
        }
    }
}

/// A factor slope term: `compare` of `y ~ 1 + (1 + g | subj)` (the slope's
/// columns random-effect-only) against `y ~ g + (1 + g | subj)`, and the Type
/// III likelihood-ratio row `g` of the larger fit, whose refit moves `g`'s
/// slope columns out of the fixed design, give the same statistic and df.
#[test]
fn compare_and_the_lrt_row_agree_on_a_factor_slope_term() {
    let case = factorial_case("factor slope", "", 1, "");
    let fit = |f: &str| crate::formula::fit(f, &case.table, case.family, &case.options()).unwrap();
    let (smaller, larger) = (fit("y ~ 1 + (1 + g | subj)"), fit("y ~ g + (1 + g | subj)"));
    assert!(smaller.fit().converged() && larger.fit().converged());
    assert_eq!(smaller.lowered().re_p, 2);
    let c = super::anova::compare(&smaller, &larger, &OPTS).unwrap();
    let t = lr(&c);
    let table =
        super::anova::term_tests(&larger, &term_options(SsType::Type3, TermTest::Lrt)).unwrap();
    let row = &table.fixed.iter().find(|r| r.label == "g").unwrap().test;
    assert_eq!(row.kind, LinearKind::LikelihoodRatio);
    assert!(row.notes.is_empty(), "{row:?}");
    assert_eq!(row.num_df, t.df as f64);
    let gap = (row.statistic - t.statistic).abs();
    eprintln!(
        "factor slope: lrt row {:e}, compare {:e}",
        row.statistic, t.statistic
    );
    assert!(gap <= LRT_ABS, "{row:?} off compare {t:?} by {gap:e}");
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
        let t = term_tests_cached(
            &lo,
            &case.options(),
            &failed_larger,
            &term_options(ss_type, TermTest::Lrt),
            &mut RefitCache::default(),
        )
        .unwrap();
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
    let want = term_tests_cached(
        &lo,
        &case.options(),
        &fit,
        &opts,
        &mut RefitCache::default(),
    )
    .unwrap();
    let (_, mut blind) = case.fit(&case.formula);
    for row in &mut blind.vcov {
        row.fill(f64::NAN);
    }
    let got = term_tests_cached(
        &lo,
        &case.options(),
        &blind,
        &opts,
        &mut RefitCache::default(),
    )
    .unwrap();
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
    let t = term_tests_cached(
        &lo,
        &case.options(),
        &fit,
        &term_options(SsType::Type3, TermTest::Lrt),
        &mut RefitCache::default(),
    )
    .unwrap();
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

/// The Type III likelihood-ratio row `x` of `y ~ x + (1 + x | g)`, binomial
/// logit on `sim_binomial_slope1`, whose refit moves the slope on `x` to a
/// random-effect-only column, against afex's `mixed(type = 3, method =
/// "LRT")`, frozen in `anova_lrt3_glmm_sim_binomial_slope1`: the statistic
/// within the registry's `lrt` band, the df exactly.
#[test]
fn type_iii_lrt_rows_with_a_random_slope_match_afex() {
    let json = crate::inference::test_refs::load(
        crate::inference::test_refs::Kind::Anova,
        "anova_lrt3_glmm_sim_binomial_slope1",
    );
    let case = LrtCase {
        name: "binomial slope GLMM",
        table: csv_table(
            include_str!("../../validation/data/simulated/sim_binomial_slope1.csv"),
            &["g"],
        ),
        formula: "y ~ x + (1 + x | g)".into(),
        family: Family::Binomial {
            link: BinomialLink::Logit,
        },
        nagq: 1,
        weights: None,
        dispersion: None,
        type2: Vec::new(),
        type3: Vec::new(),
    };
    let (lo, fit) = case.fit(&case.formula);
    let t = term_tests_cached(
        &lo,
        &case.options(),
        &fit,
        &term_options(SsType::Type3, TermTest::Lrt),
        &mut RefitCache::default(),
    )
    .unwrap();
    let rows = json["rows"].as_array().expect("frozen rows");
    assert_eq!(rows.len(), t.fixed.len());
    for (row, want) in t.fixed.iter().zip(rows) {
        assert_eq!(row.label, want["term"].as_str().unwrap());
        assert_eq!(row.test.kind, LinearKind::LikelihoodRatio, "{}", row.label);
        assert!(row.test.notes.is_empty(), "{}: {:?}", row.label, row.test);
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
    let table = term_tests_cached(
        &lo,
        &case.options(),
        &fit,
        &term_options(SsType::Type3, TermTest::Lrt),
        &mut RefitCache::default(),
    )
    .unwrap();
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
        let basis = type3_null_basis(rounded.as_ref(), l.as_ref()).unwrap();
        let kept = untouched_columns(rounded.as_ref()).len();
        assert!(basis.ncols() > kept, "N mixes columns");
        let reduced = refit_on_basis(
            &lo,
            &case.options(),
            &fit,
            &all,
            &basis,
            &(vec![Vec::new()], vec![]),
        );
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

/// `anova_ml_pair_sleepstudy_fixed`: `Reaction ~ 1 + (Days | Subject)`,
/// whose slope variable is in no fixed term, against `Reaction ~ Days +
/// (Days | Subject)`, both ML. The statistic within the registry's `lrt`
/// band of the one from the frozen logLiks, the df exactly.
#[test]
fn the_sleepstudy_ml_pair_matches_its_frozen_reference() {
    let json = crate::inference::test_refs::load(
        crate::inference::test_refs::Kind::Anova,
        "anova_ml_pair_sleepstudy_fixed",
    );
    let d = sleepstudy();
    let fit_opts = FitOptions {
        reml: Some(false),
        ..FitOptions::default()
    };
    let fit = |f: &str| crate::formula::fit(f, &d, Family::Gaussian, &fit_opts).unwrap();
    let (smaller, larger) = (
        fit("Reaction ~ 1 + (Days | Subject)"),
        fit("Reaction ~ Days + (Days | Subject)"),
    );
    let c = super::anova::compare(&smaller, &larger, &OPTS).unwrap();
    let t = lr(&c);
    let ll = |i: usize| json["members"][i]["logLik"].as_f64().unwrap();
    let want = 2.0 * (ll(1) - ll(0));
    assert_eq!(t.df, 1);
    let gap = (t.statistic - want).abs();
    assert!(gap <= LRT_ABS, "{:?} off lme4 by {gap:e}", t);
}

/// The band of a likelihood-ratio statistic against a frozen reference,
/// absolute: `TOL_INFERENCE$lrt_abs` in `validation/grid/tol.R` (also
/// copied in `tests/oracle_support/mod.rs`). Change together.
const LRT_ABS: f64 = 4e-4;

// The comparison checks and rules below run on sides changed by hand (a
// variance, a log-likelihood, a field a refusal reads) and the per-term tests
// on fits paired with other options or a covariance changed by hand: inputs
// no fitted `formula::Model` carries.

fn opts() -> CompareOptions {
    CompareOptions {
        nsim: 1000,
        seed: 1,
    }
}

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

/// A Gaussian fit of `formula` on [`table`] with `reml`, as a `Model`.
fn model(formula: &str, reml: Option<bool>) -> Model {
    side(formula, &table(), reml, None)
}

/// The statistic, p-value and reference of a comparison's test.
fn outcome(c: &Comparison) -> (u64, u64, Reference) {
    match &c.test {
        ComparisonTest::Lr(t) => (t.statistic.to_bits(), t.p_value.to_bits(), t.reference),
        ComparisonTest::Linear(t) => (t.statistic.to_bits(), t.p_value.to_bits(), t.reference),
    }
}

/// `compare_stored(a, b)`, after checking that `compare_stored(b, a)` gives
/// the same answer (the sides mirrored, the same test) or the same refusal.
fn cmp(a: &Model, b: &Model) -> Result<Comparison, CompareError> {
    let forward = compare_stored(a, b, &opts());
    let back = compare_stored(b, a, &opts());
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

fn refusal(a: &Model, b: &Model) -> CompareError {
    cmp(a, b).expect_err("refused")
}

fn sleepstudy() -> Table {
    csv_table(
        include_str!("../../validation/data/empirical/sleepstudy.csv"),
        &["Subject"],
    )
}

fn binomial() -> Family {
    Family::Binomial {
        link: BinomialLink::Logit,
    }
}

fn gamma() -> Family {
    Family::Gamma {
        link: GammaLink::Log,
    }
}

const MIXTURE: LrNote = LrNote::BoundaryMixture { approximate: false };

fn slope1() -> Table {
    csv_table(
        include_str!("../../validation/data/simulated/sim_binomial_slope1.csv"),
        &["g"],
    )
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

const FIXED: &str = "y ~ g * h + x";
/// R's `term.labels` of [`FIXED`].
const LABELS: [&str; 4] = ["g", "h", "x", "g:h"];

fn sim_factorial_bern() -> Table {
    csv_table(
        include_str!("../../validation/data/simulated/sim_factorial_bern.csv"),
        &["subj", "g", "h"],
    )
}

fn glm_gamma() -> Table {
    csv_table(
        include_str!("../../validation/data/simulated/glm_gamma.csv"),
        &["grp"],
    )
}

/// A converged fit of `formula` on `table` with `reml`, and its lowering.
fn lowered_fit(formula: &str, table: &Table, family: Family, reml: Option<bool>) -> (Lowered, Fit) {
    let lo = lower(formula, table, family).unwrap();
    let fit_opts = FitOptions {
        reml,
        ..FitOptions::default()
    };
    let fit = crate::fit(&lo.data(), lo.spec(), &fit_opts).unwrap();
    assert!(fit.converged(), "{formula} did not converge");
    (lo, fit)
}

/// The per-term table of `ss_type`, its χ² rows pinned to the Wald test so
/// they stay Wald whatever the default test gives; a fit whose rows are F
/// tests refuses an explicit test, so its table takes the default, which must
/// be its F table (a χ² fit that refused the Wald test fails here).
fn term_table(
    lo: &Lowered,
    fit_opts: &FitOptions,
    fit: &Fit,
    ss_type: SsType,
) -> Result<TermTable, TermTestError> {
    let mut opts = term_options(ss_type, TermTest::Wald);
    match term_tests_cached(lo, fit_opts, fit, &opts, &mut RefitCache::default()) {
        Err(TermTestError::FTable { .. }) => {
            opts.test = TermTest::Auto;
            let table = term_tests_cached(lo, fit_opts, fit, &opts, &mut RefitCache::default());
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

/// The Type III and Type II tables of `fit` ([`term_table`]).
fn tables(lo: &Lowered, fit_opts: &FitOptions, fit: &Fit) -> [TermTable; 2] {
    [SsType::Type3, SsType::Type2].map(|ss_type| term_table(lo, fit_opts, fit, ss_type).unwrap())
}

fn labels(t: &TermTable) -> Vec<&str> {
    t.fixed.iter().map(|r| r.label.as_str()).collect()
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
fn the_data_checks_come_before_the_nesting_check() {
    // Not nested, and fitted on different data: the data refusal is reported.
    let m0 = model("y ~ x", None);
    let mut m1 = model("y ~ z", None);
    // The premise: unmodified, the pair is refused as not nested.
    assert!(
        matches!(refusal(&m0, &m1), CompareError::NotNested { .. }),
        "the pair is not nested"
    );
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
    // An OLS pair whose reml settings differ: OLS skips the estimator check,
    // so the same parts are reported.
    assert_eq!(
        refusal(&model("y ~ x", None), &model("y ~ x", Some(true))),
        CompareError::SameParts
    );
}

#[test]
fn a_dropped_component_at_zero_gives_t_zero_and_p_one_on_every_reference() {
    // Exact reference: Dyestuff2's batch variance is estimated at zero.
    let d = csv_table(
        include_str!("../../validation/data/empirical/Dyestuff2.csv"),
        &["Batch"],
    );
    let ols = side("Yield ~ 1", &d, Some(true), None);
    let full = side("Yield ~ 1 + (1 | Batch)", &d, None, None);
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
    let weighted = |f: &str, reml| side(f, &d, reml, Some(vec![1.0; d.n]));
    let (ols_w, full_w) = (
        weighted("Yield ~ 1", Some(true)),
        weighted("Yield ~ 1 + (1 | Batch)", None),
    );
    assert_eq!(full_w.design_summary, None);
    let c = cmp(&ols_w, &full_w).unwrap();
    let t = lr(&c);
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(t.notes, vec![LrNote::NoDesignSummary, MIXTURE]);
    assert!(c.note_lines()[0].contains("unweighted (1 | g)"));
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));

    // Mixture and χ² references, from a side whose dropped variances are set
    // to zero while its log-likelihood stays a round-off above the smaller
    // one's: the rule, not the round-off, decides.
    let s = sleepstudy();
    let int = side("Reaction ~ Days + (1 | Subject)", &s, None, None);
    let ols = side("Reaction ~ Days", &s, Some(true), None);
    let mut slope = side("Reaction ~ Days + (Days | Subject)", &s, None, None);
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
fn a_raw_statistic_below_zero_is_floored_with_a_note_naming_the_larger_fit() {
    let d = sleepstudy();
    let small = side("Reaction ~ Days + (1 | Subject)", &d, None, None);
    let mut large = side("Reaction ~ Days + (Days | Subject)", &d, None, None);
    // Beyond round-off (2·dev_eps = 8e-5 deviance units).
    large.loglik = small.loglik - 1e-4;
    let raw = 2.0 * (large.loglik - small.loglik);
    let c = cmp(&small, &large).unwrap();
    let t = lr(&c);
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));
    assert_eq!(t.notes, vec![LrNote::NegativeStatistic { raw }, MIXTURE]);
    let negative = "The larger model did not reach its optimum";
    assert!(c.note_lines()[0].starts_with(negative));
    let back = compare_stored(&large, &small, &opts()).unwrap();
    assert!(back.note_lines()[0].starts_with(negative));
    // Within round-off: the floor, with no note.
    large.loglik = small.loglik - 1e-5;
    let t = lr(&cmp(&small, &large).unwrap()).clone();
    assert_eq!((t.statistic, t.p_value), (0.0, 1.0));
    assert_eq!(t.notes, vec![MIXTURE]);
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
        None,
        None,
    );
    let mut large = side(
        "Reaction ~ Days + phase + (phase + Days | Subject)",
        &d,
        None,
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
fn a_failed_fit_gives_nan_even_with_the_dropped_variance_at_zero() {
    // Three pairs whose dropped variances are zero, one per reference: the
    // mixture, χ² and the exact test.
    let s = sleepstudy();
    let int = side("Reaction ~ Days + (1 | Subject)", &s, None, None);
    let ols = side("Reaction ~ Days", &s, Some(true), None);
    let mut slope = side("Reaction ~ Days + (Days | Subject)", &s, None, None);
    let v = slope.varcorr[0][0];
    slope.varcorr = vec![vec![v, 0.0, 0.0]];
    slope.loglik = int.loglik + 1e-9;
    let mut flat = slope.clone();
    flat.varcorr = vec![vec![0.0, 0.0, 0.0]];
    flat.loglik = ols.loglik + 1e-9;
    let d2 = csv_table(
        include_str!("../../validation/data/empirical/Dyestuff2.csv"),
        &["Batch"],
    );
    let ols2 = side("Yield ~ 1", &d2, Some(true), None);
    let full2 = side("Yield ~ 1 + (1 | Batch)", &d2, None, None);
    assert_eq!(full2.varcorr, vec![vec![0.0]]);
    // The p-quality notes each reference carries on a finite pair; a NaN
    // statistic replaces them with the one note naming its cause.
    let pairs = [
        (&int, &slope, vec![MIXTURE]),
        (&ols, &flat, vec![LrNote::BoundaryIgnored]),
        (&ols2, &full2, vec![]),
    ];
    for (small, large, finite_notes) in pairs {
        let finite = lr(&compare_stored(small, large, &opts()).unwrap()).clone();
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
                let c = compare_stored(&s, &l, &opts()).unwrap();
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
        include_str!("../../validation/data/empirical/Dyestuff2.csv"),
        &["Batch"],
    );
    let weighted = |f: &str, reml| side(f, &d, reml, Some(vec![1.0; d.n]));
    let small = weighted("Yield ~ 1", Some(true));
    let mut large = weighted("Yield ~ 1 + (1 | Batch)", None);
    let s = slope1();
    let node = |f: &str, nagq| {
        let lo = lower(f, &s, binomial()).unwrap();
        let fit_opts = FitOptions {
            nagq,
            ..FitOptions::default()
        };
        let fit = crate::fit(&lo.data(), lo.spec(), &fit_opts).unwrap();
        assert!(fit.converged(), "{f} did not converge");
        Model::from_fit(f, &lo, &fit_opts, &fit)
    };
    let (glmm1, mut glmm7) = (node("y ~ 1 + (1 | g)", 1), node("y ~ x + (1 | g)", 7));
    assert_eq!(
        lr(&compare_stored(&small, &large, &opts()).unwrap()).notes,
        vec![LrNote::NoDesignSummary, MIXTURE]
    );
    large.loglik = f64::NAN;
    let c = compare_stored(&small, &large, &opts()).unwrap();
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
    let c = compare_stored(&glmm1, &glmm7, &opts()).unwrap();
    assert!(lr(&c).statistic.is_nan());
    assert_eq!(
        lr(&c).notes,
        vec![LrNote::NodeCounts { m0: 1, m1: 7 }, LrNote::NonFiniteLoglik]
    );
}

#[test]
fn a_term_kept_in_another_grouping_is_still_dropped_from_its_own() {
    // Days is dropped from Subject and kept in blk: only Subject's Days
    // variance decides the rule.
    let d = sleepstudy_blk();
    let small = side(
        "Reaction ~ Days + (1 | Subject) + (Days | blk)",
        &d,
        None,
        None,
    );
    let large = side(
        "Reaction ~ Days + (Days | Subject) + (Days | blk)",
        &d,
        None,
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
        include_str!("../../validation/data/empirical/Pastes.csv"),
        &["batch", "cask", "sample"],
    );
    let small = side("strength ~ 1 + (1 | cask:batch)", &d, None, None);
    let large = side("strength ~ 1 + (1 | batch/cask)", &d, None, None);
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
    let mut int = side("Reaction ~ Days + (1 | Subject)", &s, None, None);
    let slope = side("Reaction ~ Days + (Days | Subject)", &s, None, None);
    int.varcorr = vec![vec![0.0]];
    let t = lr(&cmp(&int, &slope).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 1 });
    assert_eq!(t.notes, vec![LrNote::BoundaryMixture { approximate: true }]);
}

#[test]
fn the_row_kind_is_read_from_the_fit_not_from_the_options() {
    // A Gamma GLM with φ estimated, its lowering then given a held φ: the
    // rows stay the fit's F rows. And the reverse: a fit with φ held keeps
    // its χ² rows beside options that estimate φ.
    let (lo, fit) = lowered_fit("y ~ x + grp", &glm_gamma(), gamma(), None);
    let want = tables(&lo, &FitOptions::default(), &fit);
    for t in &want {
        assert!(t.fixed.iter().all(|r| r.test.kind == LinearKind::F));
    }
    let mut fit_opts = FitOptions {
        dispersion: Some(0.5),
        ..FitOptions::default()
    };
    assert_eq!(tables(&lo, &fit_opts, &fit), want);
    let lo = lower("y ~ x + grp", &glm_gamma(), gamma()).unwrap();
    let fit = crate::fit(&lo.data(), lo.spec(), &fit_opts).unwrap();
    let want = tables(&lo, &fit_opts, &fit);
    for t in &want {
        assert!(t
            .fixed
            .iter()
            .all(|r| r.test.kind == LinearKind::ChiSquared));
    }
    fit_opts.dispersion = None;
    assert_eq!(tables(&lo, &fit_opts, &fit), want);
}

#[test]
fn a_covariance_that_is_not_finite_gives_nan_rows() {
    // car's Type II reads the covariance; a NaN one gives NaN rows, not a
    // panic, and so does the Wald χ² of Type III.
    let (lo, mut fit) = lowered_fit(FIXED, &sim_factorial_bern(), binomial(), None);
    fit.vcov = vec![vec![f64::NAN; lo.p]; lo.p];
    for t in tables(&lo, &FitOptions::default(), &fit) {
        assert_eq!(labels(&t), LABELS);
        assert!(
            t.fixed.iter().all(|r| r.test.statistic.is_nan()),
            "{:?}",
            t.ss_type
        );
    }
}

#[test]
fn an_ml_pair_inside_the_exact_tests_envelope_gets_the_mixture() {
    // The exact test is a REML test: two ML fits of the same pair, the larger
    // keeping its design summary, get the mixture.
    let d = csv_table(
        include_str!("../../validation/data/empirical/Dyestuff.csv"),
        &["Batch"],
    );
    let ols = side("Yield ~ 1", &d, Some(false), None);
    let full = side("Yield ~ 1 + (1 | Batch)", &d, Some(false), None);
    assert!(full.design_summary.is_some() && !full.reml && !ols.reml);
    let t = lr(&cmp(&ols, &full).unwrap()).clone();
    assert_eq!(t.reference, Reference::ChiBarSquared { df_low: 0 });
    assert_eq!(t.objective, Objective::MaximumLikelihood);
    assert_eq!(t.notes, vec![MIXTURE]);
}
