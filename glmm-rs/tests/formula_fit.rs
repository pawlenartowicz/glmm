#![cfg(feature = "formula")]
//! End-to-end: drive the validation datasets through `lower` → `glmm::fit` and assert
//! the formula frontend lowers to the same design, the same grouping ids, and
//! therefore the same fit as the hand-built `ModelSpec`/`GroupIds` the in-crate
//! kernel tests use. Closes the loop formula → design → kernel.
//!
//! **Cross-engine agreement for these fits is not this file's claim.**
//! `tests/validation_oracle.rs` refits `sleepstudy_lmm`, `penicillin_lmm`,
//! `pastes_lmm`, `cbpp_agq_k1` and `grouseticks_agq_k1` from each golden's own
//! recorded formula — through this same frontend — and gates them against the
//! frozen lme4 values at `validation/grid/tol.R`'s bands. This file does not repeat
//! that comparison: doing so would assert nothing the oracle tier does not
//! already assert more tightly. What is left is the one claim the oracle tier
//! cannot make, because it only ever fits through the frontend: that the
//! frontend agrees with the hand-built spec.

use glmm::formula::{lower, Column, Table};
#[cfg(feature = "orchestrate")]
use glmm::orchestrate::run_fit;
use glmm::{
    Family, Fit, FitOptions, GroupIds, Grouping, GroupingRelation, ModelSpec, Note, ReStructure,
};
#[cfg(feature = "orchestrate")]
use std::collections::HashMap;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Band for the frontend-equivalence assertions below.
///
/// Mirrors `PIN_REL_ITER` in `src/fit/common_tests.rs` — change together. It is
/// restated rather than imported because `common_tests` is `pub(crate)` and
/// `cfg(test)`, so an integration test cannot reach it, and widening the crate's
/// surface to relocate a constant is the wrong trade.
///
/// In practice every comparison here lands on 0.0: the two routes hand the
/// kernel byte-identical inputs, and the kernel is deterministic. The band is
/// margin against a future frontend that legitimately reorders rows or clusters,
/// which would perturb the last bit without changing the model.
const PIN_REL_ITER: f64 = 1e-7;

/// Parse a committed validation CSV into trimmed, unquoted string fields (mirrors the
/// hand-parse the in-crate kernel tests use — no CSV crate).
fn rows(csv: &str) -> Vec<Vec<String>> {
    csv.lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            l.split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect()
        })
        .collect()
}

fn numeric(rows: &[Vec<String>], col: usize) -> Column {
    Column::Numeric(rows.iter().map(|r| r[col].parse().unwrap()).collect())
}
fn factor(rows: &[Vec<String>], col: usize) -> Column {
    let labels: Vec<String> = rows.iter().map(|r| r[col].clone()).collect();
    Column::factor_from_labels(&labels)
}

/// Column `col`'s values as f64.
fn nums(rows: &[Vec<String>], col: usize) -> Vec<f64> {
    rows.iter().map(|r| r[col].parse().unwrap()).collect()
}

/// Dense 0-based level codes for column `col`, assigned in the lexicographic
/// level order `Column::factor_from_labels` uses — so a hand-built `GroupIds`
/// lines up index-for-index with the one `lower()` produces.
fn codes(rows: &[Vec<String>], col: usize) -> Vec<u32> {
    let labels: Vec<String> = rows.iter().map(|r| r[col].clone()).collect();
    match Column::factor_from_labels(&labels) {
        Column::Factor { codes, .. } => codes,
        Column::Numeric(_) => unreachable!("factor_from_labels returns a Factor"),
    }
}

/// Distinct level count for column `col`.
fn n_levels(rows: &[Vec<String>], col: usize) -> u32 {
    codes(rows, col).iter().max().map_or(0, |m| m + 1)
}

/// Treatment-contrast dummy columns for column `col`: one per non-base level, in
/// level order (level 0 is the base). This is the coding the frontend must
/// reproduce, written out by hand so a contrast bug has something to disagree
/// with.
fn dummies(rows: &[Vec<String>], col: usize) -> Vec<Vec<f64>> {
    let c = codes(rows, col);
    let k = n_levels(rows, col);
    (1..k)
        .map(|lvl| c.iter().map(|&r| f64::from(r == lvl)).collect())
        .collect()
}

/// Row-major n×p design from per-column vectors, intercept first.
fn design(n: usize, cols: &[Vec<f64>]) -> Vec<f64> {
    let p = cols.len() + 1;
    let mut x = vec![0.0; n * p];
    for i in 0..n {
        x[i * p] = 1.0;
        for (j, c) in cols.iter().enumerate() {
            x[i * p + 1 + j] = c[i];
        }
    }
    x
}

/// `|a-b| <= atol + rtol*|b|`.
fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn assert_vec_close(got: &[f64], want: &[f64], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            close(g, w, PIN_REL_ITER, 0.0),
            "{ctx}[{i}]: formula {g} vs hand-built {w}"
        );
    }
}

/// Every fitted quantity the frontend can move: a lowering bug shows up in the
/// design, the ids, or nowhere. Asserting the whole `Fit` rather than β alone is
/// what makes "nowhere" checkable.
fn assert_same_fit(got: &Fit, want: &Fit, ctx: &str) {
    assert_eq!(got.converged(), want.converged(), "{ctx}: converged");
    assert_vec_close(got.beta(), want.beta(), &format!("{ctx}: beta"));
    assert_vec_close(got.se(), want.se(), &format!("{ctx}: se"));
    assert_eq!(
        got.n_groupings(),
        want.n_groupings(),
        "{ctx}: varcorr block count"
    );
    for k in 0..want.n_groupings() {
        assert_vec_close(
            &got.varcorr(k),
            &want.varcorr(k),
            &format!("{ctx}: varcorr[{k}]"),
        );
    }
    assert!(
        close(got.dispersion(), want.dispersion(), PIN_REL_ITER, 0.0),
        "{ctx}: dispersion {} vs hand-built {}",
        got.dispersion(),
        want.dispersion()
    );
    assert!(
        close(got.loglik(), want.loglik(), PIN_REL_ITER, 0.0),
        "{ctx}: loglik {} vs hand-built {}",
        got.loglik(),
        want.loglik()
    );
}

/// The by-hand half of a frontend-equivalence case: the design, response, spec
/// and ids a caller would build without the formula frontend.
struct HandBuilt {
    x: Vec<f64>,
    y: Vec<f64>,
    p: usize,
    model: ModelSpec,
    ids: GroupIds,
    /// Coefficient names the frontend must emit, in column order.
    col_names: &'static [&'static str],
}

/// Lower `formula`, fit it, fit the hand-built spec, and assert the two agree —
/// on the design and ids first (where a frontend bug actually lives) and then on
/// every fitted quantity.
fn assert_frontend_matches_hand_built(
    formula: &str,
    table: &Table,
    family: Family,
    hand: &HandBuilt,
    ctx: &str,
) {
    let lo = lower(formula, table, family).unwrap();
    assert_eq!(lo.col_names, hand.col_names, "{ctx}: coef names");
    assert_eq!(lo.p, hand.p, "{ctx}: design width");
    assert_eq!(lo.x, hand.x, "{ctx}: design matrix");
    assert_eq!(lo.y, hand.y, "{ctx}: response");
    assert_eq!(lo.ids.primary, hand.ids.primary, "{ctx}: primary ids");
    assert_eq!(lo.ids.extra, hand.ids.extra, "{ctx}: extra grouping ids");

    let via_formula = glmm::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
    let by_hand = glmm::fit(
        &glmm::Data::new(&hand.x, &hand.y, hand.y.len(), hand.p).ids(&hand.ids),
        &hand.model,
        &FitOptions::default(),
    )
    .unwrap();
    assert!(
        via_formula.converged(),
        "{ctx}: formula fit did not converge"
    );
    assert!(
        by_hand.converged(),
        "{ctx}: hand-built fit did not converge"
    );
    assert_same_fit(&via_formula, &by_hand, ctx);
}

/// A random-intercept `ReStructure` over `extras` additional groupings.
fn intercept_re(extras: Vec<Grouping>) -> ReStructure {
    extras
        .into_iter()
        .fold(ReStructure::intercept(), ReStructure::extra)
}

// ── LMM ──────────────────────────────────────────────────────────────────────

#[test]
fn sleepstudy_random_slope_matches_hand_built_spec() {
    let data = rows(include_str!(
        "../../validation/data/empirical/sleepstudy.csv"
    ));
    let table = Table {
        columns: vec![
            ("Reaction".into(), numeric(&data, 0)),
            ("Days".into(), numeric(&data, 1)),
            ("Subject".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    let y = nums(&data, 0);
    let days = nums(&data, 1);
    let x = design(data.len(), &[days]);
    // Random slope on Days, design column 1.
    let model = ModelSpec::mixed(Family::Gaussian, ReStructure::intercept().slopes([1]));
    let ids = GroupIds::new(codes(&data, 2));
    let hand = HandBuilt {
        x,
        y,
        p: 2,
        model,
        ids,
        col_names: &["(Intercept)", "Days"],
    };
    assert_frontend_matches_hand_built(
        "Reaction ~ Days + (1 + Days | Subject)",
        &table,
        Family::Gaussian,
        &hand,
        "sleepstudy",
    );
}

/// `(1 + log(x) | g)` must lower to the same design and fit as computing the
/// column by hand and naming it — the transform is a name, not a model.
#[test]
fn random_slope_on_a_transform_matches_precomputed_column() {
    let data = rows(include_str!(
        "../../validation/data/empirical/sleepstudy.csv"
    ));
    let reaction: Vec<f64> = data.iter().map(|r| r[0].parse().unwrap()).collect();
    let days: Vec<f64> = data
        .iter()
        .map(|r| r[1].parse::<f64>().unwrap() + 1.0)
        .collect();
    let n = reaction.len();
    let by_hand = Table {
        columns: vec![
            ("Reaction".into(), Column::Numeric(reaction.clone())),
            (
                "ld".into(),
                Column::Numeric(days.iter().map(|d| d.ln()).collect()),
            ),
            ("Subject".into(), factor(&data, 2)),
        ],
        n,
    };
    let via = Table {
        columns: vec![
            ("Reaction".into(), Column::Numeric(reaction)),
            ("Days1".into(), Column::Numeric(days)),
            ("Subject".into(), factor(&data, 2)),
        ],
        n,
    };
    let lo_hand = lower(
        "Reaction ~ ld + (1 + ld | Subject)",
        &by_hand,
        Family::Gaussian,
    )
    .unwrap();
    let lo_via = lower(
        "Reaction ~ log(Days1) + (1 + log(Days1) | Subject)",
        &via,
        Family::Gaussian,
    )
    .unwrap();
    assert_eq!(lo_via.col_names, vec!["(Intercept)", "log(Days1)"]);
    assert_eq!(lo_via.x, lo_hand.x);
    assert_eq!(lo_via.model, lo_hand.model);
    assert_eq!(lo_via.re_groups[0].terms, vec!["(Intercept)", "log(Days1)"]);
    let a = glmm::fit(&lo_via.data(), lo_via.spec(), &FitOptions::default()).unwrap();
    let b = glmm::fit(&lo_hand.data(), lo_hand.spec(), &FitOptions::default()).unwrap();
    assert_same_fit(&a, &b, "log slope");
}

#[test]
fn penicillin_crossed_matches_hand_built_spec() {
    let data = rows(include_str!(
        "../../validation/data/empirical/Penicillin.csv"
    ));
    let table = Table {
        columns: vec![
            ("diameter".into(), numeric(&data, 0)),
            ("plate".into(), factor(&data, 1)),
            ("sample".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    let y = nums(&data, 0);
    let x = design(data.len(), &[]);
    let model = ModelSpec::mixed(Family::Gaussian, intercept_re(vec![Grouping::crossed()]));
    let ids = GroupIds::new(codes(&data, 1)).extra(codes(&data, 2));
    let hand = HandBuilt {
        x,
        y,
        p: 1,
        model,
        ids,
        col_names: &["(Intercept)"],
    };
    assert_frontend_matches_hand_built(
        "diameter ~ (1|plate) + (1|sample)",
        &table,
        Family::Gaussian,
        &hand,
        "penicillin",
    );
}

#[test]
fn pastes_nested_matches_hand_built_spec() {
    let data = rows(include_str!("../../validation/data/empirical/Pastes.csv"));
    let table = Table {
        columns: vec![
            ("strength".into(), numeric(&data, 0)),
            ("batch".into(), factor(&data, 1)),
            ("cask".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    let y = nums(&data, 0);
    let x = design(data.len(), &[]);
    // `(1|batch/cask)`'s inner grouping is the batch×cask composite, which the
    // fixture already carries as its globally-unique `sample` column (col 3) —
    // so the hand-built ids read that column rather than re-deriving the pairing.
    let model = ModelSpec::mixed(Family::Gaussian, intercept_re(vec![Grouping::nested()]));
    let ids = GroupIds::new(codes(&data, 1)).extra(codes(&data, 3));
    let hand = HandBuilt {
        x,
        y,
        p: 1,
        model,
        ids,
        col_names: &["(Intercept)"],
    };
    assert_frontend_matches_hand_built(
        "strength ~ (1|batch/cask)",
        &table,
        Family::Gaussian,
        &hand,
        "pastes",
    );
}

/// Same Pastes fit as `pastes_nested_matches_hand_built_spec`, but via the FLAT
/// lme4 idiom `(1|batch)+(1|sample)`: `sample` is the globally-unique
/// `batch:cask` column, so it genuinely nests in `batch` and must be classified
/// `NestedWithin` — not `Crossed` — yet produce the identical fit.
///
/// The equality is against the explicit `(1|batch/cask)` lowering rather than
/// against an oracle: "the two idioms name the same model" is a claim about this
/// frontend, and stating it as glmm-vs-glmm makes it fail on exactly the bug it
/// is looking for. lme4 agreement for the underlying fit is the `pastes_lmm`
/// cell in `tests/validation_oracle.rs`.
#[test]
fn pastes_flat_nested_equals_the_explicit_nesting() {
    let data = rows(include_str!("../../validation/data/empirical/Pastes.csv"));
    let flat = Table {
        columns: vec![
            ("strength".into(), numeric(&data, 0)),
            ("batch".into(), factor(&data, 1)),
            ("sample".into(), factor(&data, 3)),
        ],
        n: data.len(),
    };
    let explicit = Table {
        columns: vec![
            ("strength".into(), numeric(&data, 0)),
            ("batch".into(), factor(&data, 1)),
            ("cask".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };

    let lo_flat = lower("strength ~ (1|batch) + (1|sample)", &flat, Family::Gaussian).unwrap();
    let extra = &lo_flat.model.re.as_ref().unwrap().extra_groupings[0];
    assert!(
        matches!(extra.relation, GroupingRelation::NestedWithin),
        "flat (1|batch)+(1|sample) should be detected nested, got {:?}",
        extra.relation
    );

    let lo_exp = lower("strength ~ (1|batch/cask)", &explicit, Family::Gaussian).unwrap();
    let f_flat = glmm::fit(&lo_flat.data(), lo_flat.spec(), &FitOptions::default()).unwrap();
    let f_exp = glmm::fit(&lo_exp.data(), lo_exp.spec(), &FitOptions::default()).unwrap();
    assert!(
        f_flat.converged() && f_exp.converged(),
        "pastes did not converge"
    );
    assert_same_fit(&f_flat, &f_exp, "pastes flat vs explicit");

    // The two idioms disagree on what to CALL the inner grouping — the flat form
    // takes the user's column name, the explicit form composes outer:inner — so
    // the names are checked separately from the fit rather than folded into it.
    assert_eq!(lo_flat.re_groups[0].name, "batch");
    assert_eq!(lo_flat.re_groups[1].name, "sample");
    assert_eq!(lo_exp.re_groups[0].name, "batch");
    assert_eq!(lo_exp.re_groups[1].name, "batch:cask");
}

// ── GLMM ─────────────────────────────────────────────────────────────────────

/// `cbind(incidence, failures) ~ …` must lower to exactly the proportion +
/// trial-count fit the tutorials describe as the workaround.
#[test]
fn cbind_response_equals_proportion_plus_weights() {
    let data = rows(include_str!("../../validation/data/empirical/cbpp.csv"));
    let inc: Vec<f64> = data.iter().map(|r| r[1].parse().unwrap()).collect();
    let size: Vec<f64> = data.iter().map(|r| r[2].parse().unwrap()).collect();
    let fail: Vec<f64> = inc.iter().zip(&size).map(|(i, s)| s - i).collect();
    let prop: Vec<f64> = inc.iter().zip(&size).map(|(i, s)| i / s).collect();
    let n = inc.len();
    let table = Table {
        columns: vec![
            ("inc".into(), Column::Numeric(inc)),
            ("fail".into(), Column::Numeric(fail)),
            ("prop".into(), Column::Numeric(prop)),
            ("period".into(), factor(&data, 3)),
            ("herd".into(), factor(&data, 0)),
        ],
        n,
    };
    let fam = Family::Binomial {
        link: glmm::BinomialLink::Logit,
    };
    let lo_c = lower("cbind(inc, fail) ~ period + (1|herd)", &table, fam).unwrap();
    let mut lo_w = lower("prop ~ period + (1|herd)", &table, fam).unwrap();
    lo_w.weights = Some(size);
    assert_eq!(lo_c.y, lo_w.y);
    assert_eq!(lo_c.weights, lo_w.weights);
    let a = glmm::fit(&lo_c.data(), lo_c.spec(), &FitOptions::default()).unwrap();
    let b = glmm::fit(&lo_w.data(), lo_w.spec(), &FitOptions::default()).unwrap();
    assert_same_fit(&a, &b, "cbind");
}

/// cbpp with its failure count as a column (`cbind()` takes column names
/// only): `incidence`, `failures = size - incidence`, `period`, `herd`.
fn cbpp() -> Table {
    let data = rows(include_str!("../../validation/data/empirical/cbpp.csv"));
    let inc = nums(&data, 1);
    let fail: Vec<f64> = inc.iter().zip(nums(&data, 2)).map(|(i, s)| s - i).collect();
    Table {
        columns: vec![
            ("incidence".into(), Column::Numeric(inc)),
            ("failures".into(), Column::Numeric(fail)),
            ("period".into(), factor(&data, 3)),
            ("herd".into(), factor(&data, 0)),
        ],
        n: data.len(),
    }
}

fn logit() -> Family {
    Family::Binomial {
        link: glmm::BinomialLink::Logit,
    }
}

/// A default-options fit reports the SEs of every coefficient through the
/// formula door.
#[test]
fn default_options_report_every_se_through_the_formula_door() {
    let table = cbpp();
    let f = "cbind(incidence, failures) ~ period + (1 | herd)";
    let model = glmm::formula::fit(f, &table, logit(), &FitOptions::default()).unwrap();
    let se = model.fit().se();
    assert_eq!(se.len(), 4);
    assert!(se.iter().all(|s| s.is_finite()), "{se:?}");
}

/// A nested child id is `parent·W + within`, `W` the widest parent's child
/// count: ids numbered densely over all parents put a child of one parent in
/// another's slots, and are refused.
#[test]
fn nested_ids_outside_the_parent_block_are_refused() {
    // Parent 0 has one child, parent 1 three: W = 3.
    let primary: Vec<u32> = [0u32, 0, 1, 1, 1, 1, 1, 1].to_vec();
    let padded: Vec<u32> = [0u32, 0, 3, 3, 4, 4, 5, 5].to_vec();
    let dense: Vec<u32> = [0u32, 0, 1, 1, 2, 2, 3, 3].to_vec();
    let y = [1.0, 1.2, 2.0, 2.3, 3.1, 2.9, 4.0, 4.4];
    let x = [1.0; 8];
    let spec = ModelSpec::mixed(Family::Gaussian, intercept_re(vec![Grouping::nested()]));
    let fit_with = |child: &[u32]| {
        let ids = GroupIds::new(primary.clone()).extra(child.to_vec());
        glmm::fit(
            &glmm::Data::new(&x, &y, 8, 1).ids(&ids),
            &spec,
            &FitOptions::default(),
        )
    };
    assert!(fit_with(&padded).is_ok());
    match fit_with(&dense) {
        Err(glmm::FitError::Ids(m)) => {
            assert!(m.contains("parent * 3"), "{m}");
            // Row 2 is the first child id in another parent's slots.
            assert_eq!(m.row(), Some(2));
            assert!(m.ends_with("(row 2)"), "{m}");
        }
        other => panic!("expected FitError::Ids, got {:?}", other.err()),
    }
}

/// A level count is `max(id) + 1`, so one huge id would size about 4e9 levels
/// and abort the process; it is refused as a caller error instead.
#[test]
fn a_group_id_not_below_n_is_refused() {
    let y = [1.0, 1.2, 2.0, 2.3];
    let x = [1.0; 4];
    let spec = ModelSpec::mixed(Family::Gaussian, intercept_re(vec![]));
    let ids = GroupIds::new(vec![0, 0, 1, u32::MAX]);
    match glmm::fit(
        &glmm::Data::new(&x, &y, 4, 1).ids(&ids),
        &spec,
        &FitOptions::default(),
    ) {
        Err(glmm::FitError::Ids(m)) => {
            assert!(m.contains("not below the row count 4"), "{m}");
            assert_eq!(m.row(), Some(3));
            assert!(
                m.text_with_row(4).contains("(row 4); number the levels"),
                "{m}"
            );
        }
        other => panic!("expected FitError::Ids, got {:?}", other.err()),
    }
}

/// A binomial GLMM whose response is all 0 or all 1 has no finite maximum: it
/// ends not converged and `Separated`, as a Poisson GLMM on all-zero counts
/// does, never converged at |β₀| ≈ 50.
#[test]
fn constant_binomial_response_is_separated_not_converged() {
    let n = 40;
    let g: Vec<String> = (0..n).map(|i| format!("g{}", i % 8)).collect();
    for value in [0.0, 1.0] {
        let table = Table {
            columns: vec![
                ("y".into(), Column::Numeric(vec![value; n])),
                (
                    "x".into(),
                    Column::Numeric((0..n).map(|i| i as f64 / n as f64).collect()),
                ),
                ("g".into(), Column::factor_from_labels(&g)),
            ],
            n,
        };
        let model =
            glmm::formula::fit("y ~ x + (1 | g)", &table, logit(), &FitOptions::default()).unwrap();
        let fit = model.fit();
        assert!(!fit.converged(), "y = {value}");
        assert!(
            fit.diagnostics()
                .notes
                .iter()
                .any(|n| matches!(n, Note::Separated)),
            "y = {value}: {:?}",
            fit.diagnostics().notes
        );
    }
}

/// The formula door returns the kernel's refusal of an input as
/// `Error::Fit`: quadrature on a GLMM with crossed groupings.
#[test]
fn formula_fit_returns_unsupported_for_nagq_on_a_crossed_glmm() {
    let mut opts = FitOptions::default();
    opts.nagq = 3;
    let got = glmm::formula::fit(
        "cbind(incidence, failures) ~ 1 + (1 | herd) + (1 | period)",
        &cbpp(),
        logit(),
        &opts,
    );
    assert!(
        matches!(
            got,
            Err(glmm::formula::Error::Fit(glmm::FitError::Unsupported(_)))
        ),
        "{:?}",
        got.err()
    );
}

#[test]
fn cbind_rejects_zero_trials_and_non_binomial() {
    let table = Table {
        columns: vec![
            ("s".into(), Column::Numeric(vec![1.0, 0.0, 2.0])),
            ("f".into(), Column::Numeric(vec![1.0, 0.0, 1.0])),
            ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3])),
        ],
        n: 3,
    };
    let fam = Family::Binomial {
        link: glmm::BinomialLink::Logit,
    };
    let err = match lower("cbind(s, f) ~ x", &table, fam) {
        Ok(_) => panic!("expected the zero-trials row to be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("row 1"), "{err}");
    let err = match lower("cbind(s, f) ~ x", &table, Family::Gaussian) {
        Ok(_) => panic!("expected a non-binomial cbind() response to be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("binomial"), "{err}");
    // A negative count passes the positive-sum check (-2 + 10 = 8) and would
    // lower to a proportion outside [0, 1]; it must be its own rejection.
    let table = Table {
        columns: vec![
            ("s".into(), Column::Numeric(vec![1.0, -2.0, 2.0])),
            ("f".into(), Column::Numeric(vec![1.0, 10.0, 1.0])),
            ("x".into(), Column::Numeric(vec![0.1, 0.2, 0.3])),
        ],
        n: 3,
    };
    let err = match lower("cbind(s, f) ~ x", &table, fam) {
        Ok(_) => panic!("expected the negative-count row to be rejected"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("non-negative") && err.contains("row 1"),
        "{err}"
    );
}

// ── Integer grouping columns ─────────────────────────────────────────────────

/// Lower `formula` against both tables and assert they give the same design,
/// the same grouping ids and level labels, and the same fit.
fn assert_same_lowering(formula: &str, a: &Table, b: &Table, family: Family, ctx: &str) {
    let lo_a = lower(formula, a, family).unwrap();
    let lo_b = lower(formula, b, family).unwrap_or_else(|e| panic!("{ctx}: lower failed: {e}"));
    assert_eq!(lo_b.x, lo_a.x, "{ctx}: design matrix");
    assert_eq!(lo_b.ids.primary, lo_a.ids.primary, "{ctx}: primary ids");
    let labels = |lo: &glmm::formula::Lowered| -> Vec<Vec<Option<String>>> {
        lo.re_groups.iter().map(|g| g.slot_labels.clone()).collect()
    };
    assert_eq!(labels(&lo_b), labels(&lo_a), "{ctx}: level labels");
    let fit_a = glmm::fit(&lo_a.data(), lo_a.spec(), &FitOptions::default()).unwrap();
    let fit_b = glmm::fit(&lo_b.data(), lo_b.spec(), &FitOptions::default()).unwrap();
    assert!(fit_b.converged(), "{ctx}: fit did not converge");
    assert_same_fit(&fit_b, &fit_a, ctx);
}

/// `read.csv` and `pd.read_csv` load sleepstudy's `Subject` and cbpp's `herd`
/// as integers, and both ports pass an integer column on as numeric. It must
/// lower to the grouping lme4 builds with `factor()`: levels in ascending
/// numeric order, labelled as integers. cbpp's herds `1..15` are the case where
/// that order differs from the lexicographic one.
#[test]
fn integer_grouping_column_lowers_like_its_factor() {
    let data = rows(include_str!(
        "../../validation/data/empirical/sleepstudy.csv"
    ));
    let sleep = |subject: Column| Table {
        columns: vec![
            ("Reaction".into(), numeric(&data, 0)),
            ("Days".into(), numeric(&data, 1)),
            ("Subject".into(), subject),
        ],
        n: data.len(),
    };
    assert_same_lowering(
        "Reaction ~ Days + (1 + Days | Subject)",
        &sleep(factor(&data, 2)),
        &sleep(numeric(&data, 2)),
        Family::Gaussian,
        "sleepstudy",
    );

    let data = rows(include_str!("../../validation/data/empirical/cbpp.csv"));
    let herd = nums(&data, 0);
    let inc = nums(&data, 1);
    let fail: Vec<f64> = inc.iter().zip(nums(&data, 2)).map(|(i, s)| s - i).collect();
    let cbpp = |herd: Column| Table {
        columns: vec![
            ("inc".into(), Column::Numeric(inc.clone())),
            ("fail".into(), Column::Numeric(fail.clone())),
            ("period".into(), factor(&data, 3)),
            ("herd".into(), herd),
        ],
        n: data.len(),
    };
    let numeric_order = Column::Factor {
        levels: (1..=15).map(|k: u32| k.to_string()).collect(),
        codes: herd.iter().map(|&h| h as u32 - 1).collect(),
    };
    assert_same_lowering(
        "cbind(inc, fail) ~ period + (1|herd)",
        &cbpp(numeric_order),
        &cbpp(Column::Numeric(herd.clone())),
        Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
        "cbpp",
    );
}

/// Column names are Unicode wherever the formula reads them: a non-ASCII
/// grouping factor, transform argument and slope lower to the same inputs as
/// the ASCII spelling of the same model.
#[test]
fn non_ascii_column_names_lower_like_ascii_ones() {
    let data = rows(include_str!(
        "../../validation/data/empirical/sleepstudy.csv"
    ));
    let sleep = |[y, x, g]: [&str; 3]| Table {
        columns: vec![
            (y.into(), numeric(&data, 0)),
            (x.into(), numeric(&data, 1)),
            (g.into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    let ascii = lower(
        "Reaction ~ sqrt(Days) + (1 + sqrt(Days) | Subject)",
        &sleep(["Reaction", "Days", "Subject"]),
        Family::Gaussian,
    )
    .unwrap();
    let polish = lower(
        "czas_reakcji ~ sqrt(dzień) + (1 + sqrt(dzień) | osoba_ż)",
        &sleep(["czas_reakcji", "dzień", "osoba_ż"]),
        Family::Gaussian,
    )
    .unwrap();
    assert_eq!(polish.col_names, ["(Intercept)", "sqrt(dzień)"]);
    assert_eq!(polish.x, ascii.x);
    assert_eq!(polish.y, ascii.y);
    assert_eq!(polish.ids.primary, ascii.ids.primary);
    assert_eq!(polish.re_groups[0].name, "osoba_ż");
    assert_eq!(polish.re_groups[0].terms, ["(Intercept)", "sqrt(dzień)"]);
}

/// A grouping column with a fractional value is not an id column; it is
/// refused rather than cut into one level per distinct value.
#[test]
fn fractional_grouping_column_is_refused() {
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            ("g".into(), Column::Numeric(vec![1.0, 1.5, 2.0, 2.0])),
        ],
        n: 4,
    };
    match lower("y ~ (1|g)", &table, Family::Gaussian) {
        Err(glmm::formula::Error::WrongColumnKind { name, .. }) => assert_eq!(name, "g"),
        Err(e) => panic!("wrong error: {e}"),
        Ok(_) => panic!("a fractional grouping column must be refused"),
    }
}

#[test]
fn grouseticks_poisson_matches_hand_built_spec() {
    let data = rows(include_str!(
        "../../validation/data/empirical/grouseticks.csv"
    ));
    // cols: INDEX,TICKS,BROOD,HEIGHT,YEAR,LOCATION,cHEIGHT
    let table = Table {
        columns: vec![
            ("TICKS".into(), numeric(&data, 1)),
            ("YEAR".into(), factor(&data, 4)),
            ("cHEIGHT".into(), numeric(&data, 6)),
            ("INDEX".into(), factor(&data, 0)),
        ],
        n: data.len(),
    };
    let y = nums(&data, 1);
    // YEAR is a 3-level factor → 2 dummies, then the numeric cHEIGHT, in the
    // order the formula names the terms.
    let mut cols = dummies(&data, 4);
    cols.push(nums(&data, 6));
    let x = design(data.len(), &cols);
    let model = ModelSpec::mixed(
        Family::Poisson {
            link: glmm::PoissonLink::Log,
        },
        intercept_re(vec![]),
    );
    let ids = GroupIds::new(codes(&data, 0));
    let hand = HandBuilt {
        x,
        y,
        p: 4,
        model,
        ids,
        col_names: &["(Intercept)", "YEAR96", "YEAR97", "cHEIGHT"],
    };
    assert_frontend_matches_hand_built(
        "TICKS ~ YEAR + cHEIGHT + (1|INDEX)",
        &table,
        Family::Poisson {
            link: glmm::PoissonLink::Log,
        },
        &hand,
        "grouseticks",
    );
}

// ── `re_groups` (varcorr/tau2 name+term metadata) ────────────────────────────
//
// `lower()` is pure enough not to need a converging fit for this — small
// fabricated tables are enough to exercise `lower_random_effects`'s
// primary/extra bookkeeping.

/// The first-factor promotion (`y ~ 0 + f + ...`) is a FIXED-DESIGN coding
/// convention only. A random slope on the promoted factor must not inherit the
/// promotion: `(f|g)` should resolve to the same treatment-coded slope block
/// (intercept + non-base dummies) that `y ~ f + (f|g)` gets, never the
/// promoted factor's full indicator set (which would make the RE block
/// structurally singular — intercept + 3 dummies for a 3-level factor).
#[test]
fn promoted_factor_re_slope_keeps_treatment_dummies() {
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            (
                "f".into(),
                Column::factor_from_labels(&["a", "b", "c", "a", "b", "c"].map(String::from)),
            ),
            (
                "g".into(),
                Column::factor_from_labels(&["g1", "g1", "g1", "g2", "g2", "g2"].map(String::from)),
            ),
        ],
        n: 6,
    };

    let promoted = lower("y ~ 0 + f + (f|g)", &table, Family::Gaussian).unwrap();
    assert_eq!(promoted.col_names, vec!["fa", "fb", "fc"]);
    assert_eq!(promoted.re_groups[0].terms, vec!["(Intercept)", "fb", "fc"]);

    let plain = lower("y ~ f + (f|g)", &table, Family::Gaussian).unwrap();
    assert_eq!(plain.col_names, vec!["(Intercept)", "fb", "fc"]);
    assert_eq!(plain.re_groups[0].terms, vec!["(Intercept)", "fb", "fc"]);

    // Same slope columns in both designs — resolve each design's `ColumnId`s
    // back through its own `col_names` and check they name the same dummies,
    // rather than assuming identical indices (the intercept column shifts
    // everything after it by one when present).
    let promoted_slopes = &promoted.model.re.as_ref().unwrap().slopes;
    let plain_slopes = &plain.model.re.as_ref().unwrap().slopes;
    let promoted_names: Vec<&str> = promoted_slopes
        .iter()
        .map(|&c| promoted.col_names[c as usize].as_str())
        .collect();
    let plain_names: Vec<&str> = plain_slopes
        .iter()
        .map(|&c| plain.col_names[c as usize].as_str())
        .collect();
    assert_eq!(promoted_names, vec!["fb", "fc"]);
    assert_eq!(plain_names, vec!["fb", "fc"]);
}

/// The `log(x)` row has a finite input and a non-finite transform output.
#[test]
fn a_non_finite_response_or_predictor_is_reported_at_its_row() {
    let cases = [
        ("x", f64::INFINITY, "y ~ x", "x"),
        ("x", f64::NAN, "y ~ x", "x"),
        ("y", f64::NEG_INFINITY, "y ~ x", "y"),
        ("x", 1e200, "y ~ x:z", "x:z"),
        ("x", -1.0, "y ~ log(x)", "log(x)"),
        ("z", f64::INFINITY, "y ~ x + offset(z)", "offset(z)"),
    ];
    for (col, bad, formula, term) in cases {
        let mut columns = vec![
            ("y".to_string(), vec![1.0, 2.0, 3.0, 4.0]),
            ("x".to_string(), vec![0.5, 1.0, 1.5, 2.0]),
            ("z".to_string(), vec![1.0, 1e200, 1.0, 1.0]),
        ];
        columns.iter_mut().find(|(n, _)| n == col).unwrap().1[1] = bad;
        let table = Table {
            columns: columns
                .into_iter()
                .map(|(n, v)| (n, Column::Numeric(v)))
                .collect(),
            n: 4,
        };
        let err = match lower(formula, &table, Family::Gaussian) {
            Ok(_) => panic!("expected {bad} in {col} to be rejected for {formula}"),
            Err(e) => e,
        };
        assert_eq!(
            err.to_string(),
            format!("term '{term}' is not finite at row 1")
        );
        assert_eq!(err.row(), Some(1));
        assert_eq!(
            err.to_string_with_row(2),
            format!("term '{term}' is not finite at row 2")
        );
    }
}

#[test]
fn transform_of_a_factor_column_is_rejected_as_not_numeric() {
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            (
                "f".into(),
                Column::factor_from_labels(&["a", "b", "a", "b"].map(String::from)),
            ),
        ],
        n: 4,
    };
    let err = match lower("y ~ log(f)", &table, Family::Gaussian) {
        Ok(_) => panic!("expected log(f) on a factor column to be rejected"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("is not numeric"), "{err}");
}

#[test]
fn intercept_written_first_becomes_primary_over_slope() {
    // Formula order decides the primary grouping: `(1|g)` written before
    // `(1+x|h)` makes `g` primary even though the parser's slope stage runs
    // before its intercept stage. Lowering keeps the written order, so the
    // slope block sits on the EXTRA grouping here; the fit then moves the slope
    // grouping first (the size rule in `fit/common.rs`).
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            ("x".into(), Column::Numeric(vec![0.5, 1.5, 2.5, 3.5])),
            (
                "g".into(),
                Column::factor_from_labels(&["g1", "g1", "g2", "g2"].map(String::from)),
            ),
            (
                "h".into(),
                Column::factor_from_labels(&["h1", "h2", "h1", "h2"].map(String::from)),
            ),
        ],
        n: 4,
    };
    let lo = glmm::formula::lower("y ~ x + (1|g) + (1+x|h)", &table, Family::Gaussian).unwrap();
    assert_eq!(lo.re_groups[0].name, "g");
    assert_eq!(lo.re_groups[0].terms, vec!["(Intercept)".to_string()]);
    assert_eq!(lo.re_groups[1].name, "h");
    assert_eq!(
        lo.re_groups[1].terms,
        vec!["(Intercept)".to_string(), "x".to_string()]
    );
    let re = lo.model.re.as_ref().unwrap();
    assert!(re.slopes.is_empty(), "primary grouping carries no slopes");
    assert_eq!(re.extra_groupings.len(), 1);
    assert_eq!(
        re.extra_groupings[0].slopes,
        vec![1],
        "x is design column 1"
    );
}

// ── Crossed interaction grouping, `(1|A:B)` ──────────────────────────────────

#[test]
fn interaction_grouping_produces_dense_composite_ids() {
    // recipe × replicate: 4 distinct (recipe,replicate) pairs over 6 rows, with
    // row 4 repeating row 0's pair and row 5 repeating row 3's — crossed, not
    // nested: replicate "1"/"2" are shared labels across recipes, not unique per
    // recipe, so this must NOT collapse to the nested `A/B` id scheme.
    // This id-layout shape's fitted-model oracle lives externally: validation
    // rung 13 (cake, `validation/manifest.json`) exercises the `(1|recipe:replicate)`
    // grouping through a real fit.
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            (
                "recipe".into(),
                Column::factor_from_labels(&["A", "A", "B", "B", "A", "B"].map(String::from)),
            ),
            (
                "replicate".into(),
                Column::factor_from_labels(&["1", "2", "1", "2", "1", "2"].map(String::from)),
            ),
        ],
        n: 6,
    };
    let lo = glmm::formula::lower("y ~ (1|recipe:replicate)", &table, Family::Gaussian).unwrap();
    assert_eq!(lo.re_groups.len(), 1);
    assert_eq!(lo.re_groups[0].name, "recipe:replicate");
    // Levels sorted lexicographically over the joined "recipe:replicate" label
    // ("A:1","A:2","B:1","B:2") — dense codes 0..3, row4 repeats row0's pair
    // (A:1), row5 repeats row3's (B:2).
    assert_eq!(lo.ids.primary, vec![0, 1, 2, 3, 0, 3]);
    assert!(lo.ids.extra.is_empty());
}

// ── Too few rows for a term's random effects ────────────────────────────────
//
// lme4 stops when the rows are at most a term's random effects (`lmer`) or
// below them (`glmer`), checking each term on its own (`check.nobs.vs.nRE`).
// glmm fits these models and notes them.

/// A factor column with labels `s<code>`.
fn coded(codes: &[usize]) -> Column {
    let labels: Vec<String> = codes.iter().map(|c| format!("s{c}")).collect();
    Column::factor_from_labels(&labels)
}

/// `y`, `time` and `id`, with `rows_per_id[i]` rows for subject `i` at times
/// `0, 1, ...`: the data for `(1 + time | id)`, which has `2 * rows_per_id.len()`
/// random effects.
fn slope_table(rows_per_id: &[usize]) -> Table {
    let id: Vec<usize> = rows_per_id
        .iter()
        .enumerate()
        .flat_map(|(i, &k)| std::iter::repeat_n(i, k))
        .collect();
    let time = rows_per_id
        .iter()
        .flat_map(|&k| (0..k).map(|t| t as f64))
        .collect();
    let n = id.len();
    Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric((0..n).map(|i| ((i * 7) % 5) as f64).collect()),
            ),
            ("time".into(), Column::Numeric(time)),
            ("id".into(), coded(&id)),
        ],
        n,
    }
}

/// The lowering's `FewRowsForRandomEffects` notes, as
/// `(terms, grouping, random_effects, rows)`.
fn few_rows_notes(formula: &str, table: &Table, family: Family) -> Vec<(String, String, u32, u32)> {
    lower(formula, table, family)
        .unwrap()
        .notes
        .into_iter()
        .filter_map(|note| match note {
            Note::FewRowsForRandomEffects {
                terms,
                grouping,
                random_effects,
                rows,
            } => Some((terms, grouping, random_effects, rows)),
            _ => None,
        })
        .collect()
}

#[test]
fn a_term_with_more_random_effects_than_rows_is_noted() {
    // 10 subjects, two with one row: 18 rows for 20 random effects, a model
    // lme4's lmer stops on.
    let table = slope_table(&[1, 2, 2, 2, 2, 1, 2, 2, 2, 2]);
    assert_eq!(
        few_rows_notes("y ~ time + (1 + time | id)", &table, Family::Gaussian),
        vec![("1 + time".to_string(), "id".to_string(), 20, 18)]
    );
}

#[test]
fn a_gaussian_model_is_noted_up_to_as_many_rows_as_random_effects() {
    let formula = "y ~ time + (1 + time | id)";
    let twenty = slope_table(&[2; 10]);
    assert_eq!(
        few_rows_notes(formula, &twenty, Family::Gaussian),
        vec![("1 + time".to_string(), "id".to_string(), 20, 20)]
    );
    let mut one_more = [2; 10];
    one_more[0] = 3;
    assert!(few_rows_notes(formula, &slope_table(&one_more), Family::Gaussian).is_empty());
}

#[test]
fn an_observation_level_intercept_on_a_glmm_is_not_noted() {
    // One level per row: as many random effects as rows, which glmer allows.
    let n = 40;
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric((0..n).map(|i| (i % 4) as f64).collect()),
            ),
            (
                "x".into(),
                Column::Numeric((0..n).map(|i| i as f64 / 10.0).collect()),
            ),
            ("obs".into(), coded(&(0..n).collect::<Vec<_>>())),
        ],
        n,
    };
    let poisson = Family::Poisson {
        link: glmm::PoissonLink::Log,
    };
    assert!(few_rows_notes("y ~ x + (1 | obs)", &table, poisson).is_empty());
}

#[test]
fn crossed_terms_are_checked_one_at_a_time() {
    // 15 rows; 9 levels of s and 8 of i, 17 together. lme4 fits this model.
    let n = 15;
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric((0..n).map(|i| ((i * 3) % 7) as f64).collect()),
            ),
            (
                "s".into(),
                coded(&(0..n).map(|i| i % 9).collect::<Vec<_>>()),
            ),
            (
                "i".into(),
                coded(&(0..n).map(|i| i % 8).collect::<Vec<_>>()),
            ),
        ],
        n,
    };
    assert!(few_rows_notes("y ~ 1 + (1 | s) + (1 | i)", &table, Family::Gaussian).is_empty());
}

#[test]
fn a_dropped_single_level_grouping_is_not_checked() {
    // `(1 + f | one)` has 1 level times 5 terms, as many as the rows, but the
    // term is dropped for its single level, so only that note is raised.
    let n = 5;
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric((0..n).map(|i| i as f64).collect()),
            ),
            ("f".into(), coded(&(0..n).collect::<Vec<_>>())),
            ("one".into(), coded(&[0; 5])),
        ],
        n,
    };
    let lo = lower("y ~ f + (1 + f | one)", &table, Family::Gaussian).unwrap();
    assert!(
        matches!(
            lo.notes.as_slice(),
            [Note::SingleLevelGroupingDropped { grouping }] if grouping == "one"
        ),
        "{:?}",
        lo.notes
    );
}

/// The formula door reaches the same identification check as the matrix door:
/// `g/one` with one `one` per `g` gives a nested grouping with the same groups
/// as `g`, and a factor slope constant within `g` with two levels is a slope
/// the data cannot separate from the intercept. A slope that varies within
/// groups gets no note. The same grouping spelled as an interaction with a
/// one-level factor is noted too.
#[test]
fn formula_door_shares_the_identification_check() {
    let n = 96;
    let g: Vec<usize> = (0..n).map(|i| i / 8).collect();
    let table = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(
                    (0..n)
                        .map(|i| {
                            0.3 * ((i * 5) % 8) as f64
                                + 0.5 * ((g[i] * 7) % 5) as f64
                                + 0.2 * ((i * 3) % 7) as f64
                        })
                        .collect(),
                ),
            ),
            (
                "x".into(),
                Column::Numeric((0..n).map(|i| ((i * 5) % 8) as f64).collect()),
            ),
            ("g".into(), coded(&g)),
            ("one".into(), coded(&[0; 96])),
            (
                "cat".into(),
                coded(&g.iter().map(|&l| usize::from(l >= 6)).collect::<Vec<_>>()),
            ),
        ],
        n,
    };
    let notes = |formula: &str| {
        let model =
            glmm::formula::fit(formula, &table, Family::Gaussian, &FitOptions::default()).unwrap();
        let notes = model.fit().diagnostics().notes.clone();
        notes
            .into_iter()
            .filter(|n| {
                matches!(
                    n,
                    Note::SingleLevelGrouping { .. }
                        | Note::RepeatedGrouping { .. }
                        | Note::SlopeConstantWithinLevels { .. }
                        | Note::SeparatedByGrouping { .. }
                )
            })
            .map(|n| format!("{n:?}"))
            .collect::<Vec<_>>()
    };
    let repeated = Note::RepeatedGrouping {
        first: 0,
        second: 1,
    };
    let constant = Note::SlopeConstantWithinLevels {
        grouping: 0,
        slopes: vec![0],
    };
    assert_eq!(notes("y ~ x + (1 | g/one)"), [format!("{repeated:?}")]);
    assert_eq!(
        notes("y ~ x + (1 | g) + (1 | g:one)"),
        [format!("{repeated:?}")]
    );
    assert_eq!(notes("y ~ cat + (1 + cat | g)"), [format!("{constant:?}")]);
    assert!(notes("y ~ x + (1 + x | g)").is_empty());
}

/// `re_groups` must line up index-for-index with `varcorr`, since `summary()`
/// labels block `k` with name `k` — a swap mislabels a variance component, which
/// is a wrong answer, not a cosmetic slip.
///
/// Penicillin is the fixture where a swap is visible: two named crossed
/// groupings whose variances differ ~2.3×. The stddevs are pinned from glmm's
/// own output; the same fit is gated against lme4 by the `penicillin_lmm` cell
/// in `tests/validation_oracle.rs`, which is what makes them valid answers to pin.
#[test]
fn re_groups_align_with_varcorr_blocks() {
    // Pinned from glmm; validated by the `penicillin_lmm` oracle cell.
    // Re-pinned 2026-10-02 with the per-search BOBYQA start radii (LMM blind
    // start 0.5): the search takes another path to the same optimum, plate sd
    // 1.2e-7 and sample sd 1.9e-7 relative, REML criterion 1.5e-12 lower, no
    // converged, singular or pinned flag changed. Moved by round-off when the
    // response got one units step (shift and scale): plate sd 2.5e-7 and
    // sample sd 2.3e-7 relative, REML criterion 2.2e-11 higher, no flag
    // changed.
    const PLATE_SD: f64 = 0.8467043253286997;
    const SAMPLE_SD: f64 = 1.9315583874608409;

    let data = rows(include_str!(
        "../../validation/data/empirical/Penicillin.csv"
    ));
    let table = Table {
        columns: vec![
            ("diameter".into(), numeric(&data, 0)),
            ("plate".into(), factor(&data, 1)),
            ("sample".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    let lo = lower(
        "diameter ~ (1|plate) + (1|sample)",
        &table,
        Family::Gaussian,
    )
    .unwrap();
    let f = glmm::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
    assert!(f.converged());

    // The shim's invariant: one name per varcorr block, same order.
    assert_eq!(
        lo.re_groups.len(),
        f.n_groupings(),
        "re_groups and varcorr must agree in length"
    );
    let want: [(&str, f64); 2] = [("plate", PLATE_SD), ("sample", SAMPLE_SD)];
    for (k, (name, sd)) in want.iter().enumerate() {
        assert_eq!(&lo.re_groups[k].name, name, "block {k} name");
        assert_eq!(
            lo.re_groups[k].terms,
            vec!["(Intercept)".to_string()],
            "block {k} terms"
        );
        // The variance actually sitting in block k must be the one belonging to
        // the grouping named at index k — this is what a swap breaks.
        let got = f.varcorr(k)[0].sqrt();
        assert!(
            close(got, *sd, PIN_REL_ITER, 0.0),
            "block {k} ({name}) sd {got} vs pinned {sd}"
        );
    }
    // Guard the guard: the two stddevs must actually differ, or a swap would
    // slip through the check above.
    assert!(
        (PLATE_SD - SAMPLE_SD).abs() > 0.5,
        "fixture no longer distinguishes a swap"
    );
}

/// A factor's level order is the caller's, and level 0 is the
/// treatment-contrast base. A sorted level order would base
/// `["low","med","high"]` against `"high"`: the same fit quality, a different
/// β, a different question answered, and nothing in the output to reveal it.
#[test]
fn declared_factor_level_order_picks_the_reference_level() {
    let labels = ["low", "high", "med", "low", "high", "med"].map(String::from);
    let y = Column::Numeric(vec![1.0, 3.0, 2.0, 1.1, 3.1, 2.1]);
    let declared = Table {
        columns: vec![
            (
                "y".into(),
                Column::Numeric(vec![1.0, 3.0, 2.0, 1.1, 3.1, 2.1]),
            ),
            (
                "f".into(),
                Column::Factor {
                    levels: vec!["low".into(), "med".into(), "high".into()],
                    codes: vec![0, 2, 1, 0, 2, 1],
                },
            ),
        ],
        n: 6,
    };
    let lo = lower("y ~ f", &declared, Family::Gaussian).unwrap();
    // Base is "low" (level 0), so the emitted dummies are the OTHER two, in the
    // caller's order — not lexicographic ("high" would sort first).
    assert_eq!(lo.col_names, vec!["(Intercept)", "fmed", "fhigh"]);

    // The same labels with no declared order sort lexicographically, base
    // "high" — the default ordering, not an imposition.
    let sorted = Table {
        columns: vec![
            ("y".into(), y),
            ("f".into(), Column::factor_from_labels(&labels)),
        ],
        n: 6,
    };
    let lo2 = lower("y ~ f", &sorted, Family::Gaussian).unwrap();
    assert_eq!(lo2.col_names, vec!["(Intercept)", "flow", "fmed"]);

    // Not just names: the reference level moved, so the intercept moved with it.
    let f1 = glmm::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
    let f2 = glmm::fit(&lo2.data(), lo2.spec(), &FitOptions::default()).unwrap();
    assert!(f1.converged() && f2.converged());
    // Intercept = mean of the base level: "low" ≈ 1.05 vs "high" ≈ 3.05.
    assert!(
        close(f1.beta()[0], 1.05, 1e-6, 1e-6),
        "base low: {}",
        f1.beta()[0]
    );
    assert!(
        close(f2.beta()[0], 3.05, 1e-6, 1e-6),
        "base high: {}",
        f2.beta()[0]
    );
}

/// `y ~ x:z + xz` with `xz = x * z` is rank deficient. R sorts terms by degree,
/// so the design is `(Intercept) xz x:z` and the later column, `x:z`, is the
/// aliased one: lme4 2.0.6 drops it ("dropping 1 column", `col.dropped` 3) and
/// reports `(Intercept) xz`, as `lm` reports `x:z` as NA. The kept coefficients
/// are the fit of `y ~ xz`.
#[test]
fn rank_deficient_interaction_drops_the_column_r_drops() {
    let n = 60;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
    let z: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).cos() + 0.5).collect();
    let xz: Vec<f64> = x.iter().zip(&z).map(|(a, b)| a * b).collect();
    let g: Vec<f64> = (0..n).map(|i| (i / 6) as f64).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.8 * xz[i] + 0.3 * (g[i] - 4.5) + (i as f64 * 1.3).sin())
        .collect();
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("x".into(), Column::Numeric(x)),
            ("z".into(), Column::Numeric(z)),
            ("xz".into(), Column::Numeric(xz)),
            ("g".into(), Column::Numeric(g)),
        ],
        n,
    };
    let lo = lower("y ~ x:z + xz + (1|g)", &table, Family::Gaussian).unwrap();
    assert_eq!(lo.col_names, vec!["(Intercept)", "xz", "x:z"]);
    let fit = glmm::fit(&lo.data(), lo.spec(), &FitOptions::default()).unwrap();
    assert_eq!(fit.aliased(), &[false, false, true]);
    assert!(fit.beta()[2].is_nan());

    let reduced = lower("y ~ xz + (1|g)", &table, Family::Gaussian).unwrap();
    let want = glmm::fit(&reduced.data(), reduced.spec(), &FitOptions::default()).unwrap();
    assert!(fit.converged() && want.converged());
    assert_vec_close(&fit.beta()[..2], want.beta(), "kept coefficients");
}

/// `offset(log(e))` in the formula must produce the same fit as passing the
/// same vector through `Data::offset`.
#[test]
fn offset_term_equals_offset_argument() {
    let data = rows(include_str!(
        "../../validation/data/empirical/grouseticks.csv"
    ));
    let ticks: Vec<f64> = data.iter().map(|r| r[1].parse().unwrap()).collect();
    let height: Vec<f64> = data.iter().map(|r| r[3].parse().unwrap()).collect();
    let n = ticks.len();
    let table = Table {
        columns: vec![
            ("TICKS".into(), Column::Numeric(ticks)),
            ("HEIGHT".into(), Column::Numeric(height.clone())),
            ("BROOD".into(), factor(&data, 2)),
        ],
        n,
    };
    let fam = Family::Poisson {
        link: glmm::PoissonLink::Log,
    };
    let lo_term = lower("TICKS ~ offset(log(HEIGHT)) + (1|BROOD)", &table, fam).unwrap();
    let mut lo_arg = lower("TICKS ~ (1|BROOD)", &table, fam).unwrap();
    lo_arg.offset = Some(height.iter().map(|h| h.ln()).collect());
    assert_eq!(lo_term.col_names, vec!["(Intercept)"]);
    assert_eq!(lo_term.offset, lo_arg.offset);
    let a = glmm::fit(&lo_term.data(), lo_term.spec(), &FitOptions::default()).unwrap();
    let b = glmm::fit(&lo_arg.data(), lo_arg.spec(), &FitOptions::default()).unwrap();
    assert_same_fit(&a, &b, "offset term");
}

// ── FitResult carries the response and the row count ─────────────────────────

/// `y` on the result is the response AFTER lowering — for `cbind(s, f)` the
/// proportion the kernel fitted, not either raw column — and `nobs` is the
/// lowered row count. Both ports print a header from these on fits where
/// `fitted` is empty, so they must not be derived from `fitted`.
///
/// Behind `orchestrate`: `run_fit` and `FitResult` are gated on that
/// off-by-default feature, so this test is too — the plain `cargo test`
/// default-feature build never sees this import.
#[cfg(feature = "orchestrate")]
#[test]
fn run_fit_returns_lowered_response_and_nobs() {
    let mut numeric = HashMap::new();
    numeric.insert("s".to_string(), vec![1.0, 2.0, 0.0, 3.0, 1.0, 2.0]);
    numeric.insert("f".to_string(), vec![3.0, 2.0, 4.0, 1.0, 3.0, 2.0]);
    numeric.insert("x".to_string(), vec![0.1, 0.5, 0.9, 0.2, 0.6, 0.8]);
    let factor = HashMap::new();
    let r = run_fit(
        "cbind(s, f) ~ x",
        numeric,
        factor,
        "binomial",
        "logit",
        "hessian",
        1,
        None,
        None,
        None,
        None,
        true,
        true,
        None,
        "Wald",
        None,
        None,
    )
    .unwrap();
    assert_eq!(r.nobs, 6);
    let expected: Vec<f64> = [1.0, 2.0, 0.0, 3.0, 1.0, 2.0]
        .iter()
        .zip([3.0, 2.0, 4.0, 1.0, 3.0, 2.0])
        .map(|(s, f)| s / (s + f))
        .collect();
    assert_eq!(r.y, expected);
}

/// The `reml` rule of `run_fit`, decided on the lowered model: `Some(true)` on a
/// non-Gaussian family is stripped with `reml_warning` set, and every other
/// value reaches `FitOptions::reml`. "The same fit" here means bitwise.
#[cfg(feature = "orchestrate")]
mod reml_rule {
    use super::*;
    use glmm::orchestrate::FitResult;

    const GLMM: &str = "cbind(incidence, healthy) ~ period + (1 | herd)";
    const LMM: &str = "Reaction ~ Days + (1 | Subject)";
    const OLS: &str = "Reaction ~ Days";

    /// A factor column of `n` rows and a single level.
    fn one_level(n: usize) -> Column {
        Column::factor_from_labels(&vec!["a".to_string(); n])
    }

    /// sleepstudy, plus `one`, a grouping column of a single level.
    fn sleepstudy() -> Table {
        let data = rows(include_str!(
            "../../validation/data/empirical/sleepstudy.csv"
        ));
        let n = data.len();
        Table {
            columns: vec![
                ("Reaction".into(), numeric(&data, 0)),
                ("Days".into(), numeric(&data, 1)),
                ("Subject".into(), factor(&data, 2)),
                ("one".into(), one_level(n)),
            ],
            n,
        }
    }

    /// cbpp in its aggregated form (`healthy = size − incidence`), plus `one`.
    fn cbpp() -> Table {
        let data = rows(include_str!("../../validation/data/empirical/cbpp.csv"));
        let n = data.len();
        let incidence = nums(&data, 1);
        let healthy = nums(&data, 2)
            .iter()
            .zip(&incidence)
            .map(|(s, i)| s - i)
            .collect();
        Table {
            columns: vec![
                ("incidence".into(), Column::Numeric(incidence)),
                ("healthy".into(), Column::Numeric(healthy)),
                ("period".into(), factor(&data, 3)),
                ("herd".into(), factor(&data, 0)),
                ("one".into(), one_level(n)),
            ],
            n,
        }
    }

    /// `run_fit` on `table`'s columns with Hessian SEs, Laplace and the
    /// Satterthwaite pass on.
    fn run(
        formula: &str,
        table: &Table,
        family: &str,
        link: &str,
        reml: Option<bool>,
    ) -> FitResult {
        let (mut numeric, mut factors) = (HashMap::new(), HashMap::new());
        for (name, col) in &table.columns {
            match col {
                Column::Numeric(v) => {
                    numeric.insert(name.clone(), v.clone());
                }
                Column::Factor { levels, codes } => {
                    factors.insert(name.clone(), (levels.clone(), codes.clone()));
                }
            }
        }
        run_fit(
            formula, numeric, factors, family, link, "hessian", 1, None, None, None, None, true,
            true, reml, "Wald", None, None,
        )
        .unwrap_or_else(|e| panic!("{formula}, reml {reml:?}: {e}"))
    }

    fn gaussian(formula: &str, reml: Option<bool>) -> FitResult {
        run(formula, &sleepstudy(), "gaussian", "identity", reml)
    }

    fn binomial(formula: &str, reml: Option<bool>) -> FitResult {
        run(formula, &cbpp(), "binomial", "logit", reml)
    }

    /// `glmm::fit` on the lowered formula with `FitOptions::reml` set.
    fn cold(formula: &str, table: &Table, family: Family, reml: Option<bool>) -> Fit {
        let lo = lower(formula, table, family).unwrap();
        let mut fit_opts = FitOptions::default();
        fit_opts.reml = reml;
        glmm::fit(&lo.data(), lo.spec(), &fit_opts).unwrap()
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// Every `f64` a `FitResult` carries, in a fixed order.
    fn result_f64s(r: &FitResult) -> Vec<f64> {
        let scalars = [
            r.dispersion,
            r.outer_decrement,
            r.deviance,
            r.loglik,
            r.rss,
            r.r_squared,
            r.adj_r_squared,
        ];
        let vectors = [
            &r.beta,
            &r.se,
            &r.tau2,
            &r.stddev_se,
            &r.fitted,
            &r.ranef,
            &r.y,
            &r.coef_df,
            &r.coef_p,
        ];
        scalars
            .iter()
            .chain(vectors.into_iter().flatten())
            .chain(r.vcov.iter().flatten())
            .chain(r.varcorr.iter().flatten())
            .chain(r.ranef_blocks.iter().flat_map(|b| &b.3))
            .chain(r.weights.iter().flatten())
            .chain(r.notes.iter().map(|n| &n.pivot))
            .copied()
            .collect()
    }

    /// `got` is the fit `want` is: every `f64` by bits, every other field
    /// through `Debug`, `reml_warning` aside.
    fn assert_same_result(mut got: FitResult, want: &FitResult, ctx: &str) {
        assert_eq!(
            bits(&result_f64s(&got)),
            bits(&result_f64s(want)),
            "{ctx}: f64 bits"
        );
        got.reml_warning = want.reml_warning;
        assert_eq!(format!("{got:?}"), format!("{want:?}"), "{ctx}");
    }

    fn has_dropped_note(r: &FitResult) -> bool {
        r.notes
            .iter()
            .any(|n| n.kind == "single_level_grouping_dropped")
    }

    #[test]
    fn run_fit_reml_true_ignored_with_warning_on_binomial_glmm() {
        let on = binomial(GLMM, Some(true));
        assert!(on.reml_warning, "Some(true) on a GLMM must warn");
        assert!(!on.reml);
        assert_same_result(on, &binomial(GLMM, None), "GLMM Some(true) vs None");
    }

    #[test]
    fn run_fit_reml_false_silent_on_binomial_glmm() {
        let off = binomial(GLMM, Some(false));
        assert!(!off.reml_warning);
        assert_same_result(off, &binomial(GLMM, None), "GLMM Some(false) vs None");
    }

    #[test]
    fn run_fit_reml_true_on_ols_reports_reml_loglik() {
        let on = gaussian(OLS, Some(true));
        let none = gaussian(OLS, None);
        assert!(on.reml && !none.reml);
        assert!(!on.reml_warning);
        assert_eq!(bits(&on.beta), bits(&none.beta));
        assert_eq!(bits(&on.se), bits(&none.se));
        assert_eq!(on.dispersion.to_bits(), none.dispersion.to_bits());
        let want = cold(OLS, &sleepstudy(), Family::Gaussian, Some(true));
        assert_eq!(on.loglik.to_bits(), want.loglik().to_bits());
        assert_ne!(
            on.loglik, none.loglik,
            "the REML and ML log-likelihoods differ"
        );
    }

    #[test]
    fn run_fit_reml_true_accepted_on_gaussian_mixed() {
        let on = gaussian(LMM, Some(true));
        assert!(on.reml);
        assert!(!on.reml_warning);
        assert_same_result(on, &gaussian(LMM, None), "LMM Some(true) vs None");
    }

    #[test]
    fn run_fit_reml_accepted_when_grouping_dropped() {
        // Gaussian: fitted as OLS, so the OLS row applies.
        let f = "Reaction ~ Days + (1 | one)";
        let (none, on, off) = (
            gaussian(f, None),
            gaussian(f, Some(true)),
            gaussian(f, Some(false)),
        );
        for r in [&none, &on, &off] {
            assert!(has_dropped_note(r), "{:?}", r.notes);
            assert!(!r.reml_warning);
        }
        assert!(on.reml);
        let want = cold(OLS, &sleepstudy(), Family::Gaussian, Some(true));
        assert_eq!(on.loglik.to_bits(), want.loglik().to_bits());
        assert_eq!(bits(&on.beta), bits(&none.beta));
        assert_eq!(on.dispersion.to_bits(), none.dispersion.to_bits());
        assert!(!off.reml);
        assert_same_result(off, &none, "dropped Gaussian Some(false) vs None");

        // Binomial: fitted as a GLM, so the GLM rule applies.
        let g = "cbind(incidence, healthy) ~ period + (1 | one)";
        let none = binomial(g, None);
        let (on, off) = (binomial(g, Some(true)), binomial(g, Some(false)));
        assert!(has_dropped_note(&none) && has_dropped_note(&on) && has_dropped_note(&off));
        assert!(on.reml_warning, "Some(true) on a GLM must warn");
        assert!(!off.reml_warning);
        assert_same_result(on, &none, "dropped binomial Some(true) vs None");
        assert_same_result(off, &none, "dropped binomial Some(false) vs None");
    }

    /// An enum variant as the ports' snake-case string, underscores dropped
    /// (`AtBoundary` and `"at_boundary"` both give `"atboundary"`).
    fn flat(name: &str) -> String {
        name.replace('_', "").to_lowercase()
    }

    #[test]
    fn run_fit_reml_false_on_gaussian_mixed_fits_ml() {
        let r = gaussian(LMM, Some(false));
        let want = cold(LMM, &sleepstudy(), Family::Gaussian, Some(false));
        let d = want.diagnostics();
        assert!(!r.reml && !want.reml());
        assert!(!r.reml_warning);
        assert_eq!(r.converged, d.converged);
        assert_eq!(
            (r.singular, &r.aliased, &r.pinned),
            (d.singular, &d.aliased, &d.pinned)
        );
        assert_eq!(
            (r.n_eval, r.n_grad, r.df),
            (want.n_eval(), d.n_grad, want.df())
        );
        assert_eq!(r.outer_decrement.to_bits(), d.outer_decrement.to_bits());
        assert_eq!(flat(r.boundary), flat(&format!("{:?}", d.boundary)));
        assert_eq!(
            r.outer_stop.map(flat),
            d.outer_stop.map(|s| flat(&format!("{s:?}")))
        );
        assert_eq!(r.notes.len(), d.notes.len());
        assert_eq!(bits(&r.beta), bits(want.beta()));
        assert_eq!(bits(&r.se), bits(want.se()));
        assert_eq!(bits(&r.fitted), bits(want.fitted()));
        let ranef: Vec<f64> = (0..want.n_groupings())
            .flat_map(|g| want.ranef(g).iter().copied())
            .collect();
        assert_eq!(bits(&r.ranef), bits(&ranef));
        assert_eq!(bits(&r.coef_df), bits(want.coef_df()));
        assert_eq!(r.ranef_levels, want.ranef_levels());
        // Row by row, so a missing or extra row fails rather than zipping away.
        let rows = |m: &[Vec<f64>]| m.iter().map(|v| bits(v)).collect::<Vec<_>>();
        let p = want.beta().len();
        let want_rows: Vec<Vec<u64>> = want.vcov().chunks(p).map(bits).collect();
        assert_eq!(rows(&r.vcov), want_rows);
        // The port keeps each grouping's vech; the kernel gives the q × q.
        let vech = |g: usize| {
            let (q, m) = (want.n_re(g), want.varcorr(g));
            (0..q)
                .flat_map(|j| (j..q).map(move |i| i * q + j))
                .map(|k| m[k])
                .collect::<Vec<f64>>()
        };
        let want_varcorr: Vec<Vec<f64>> = (0..want.n_groupings()).map(vech).collect();
        assert_eq!(rows(&r.varcorr), rows(&want_varcorr));
        assert!(!r.vcov.is_empty() && !r.varcorr.is_empty());
        for (g, w) in [
            (r.dispersion, want.dispersion()),
            (r.deviance, want.deviance()),
            (r.loglik, want.loglik()),
        ] {
            assert_eq!(g.to_bits(), w.to_bits());
        }
        // `coef_p` is the t test on each unit vector; `rss` is OLS only.
        let p = want.beta().len();
        let coef_p: Vec<f64> = (0..p)
            .map(|j| {
                let mut e = vec![0.0; p];
                e[j] = 1.0;
                want.contrast_t(&e).map_or(f64::NAN, |t| t.p_value)
            })
            .collect();
        assert_eq!(bits(&r.coef_p), bits(&coef_p));
        assert!(r.rss.is_nan());
        assert_eq!(r.df_method, "satterthwaite");
        assert!(r
            .coef_df
            .iter()
            .all(|&v| v.is_finite() && v <= r.nobs as f64));
        // The ML fit is not the REML one.
        assert_ne!(r.loglik, gaussian(LMM, None).loglik);
    }
}

/// The readers of a fit's weights: a fit with `weight_power = 0.5`
/// reads the weights raised to 0.5 wherever the claimed fit on `w^0.5`
/// reads its own: `rss`, `r_squared`, `adj_r_squared`, the comparison's F
/// (its `rss` from `anova::Model::from_fit`), and the simulated draws, by the
/// Rust door and by the ports' stored numbers.
#[cfg(feature = "orchestrate")]
mod weight_power_readers {
    use super::*;
    use glmm::orchestrate::{run_compare, simulate, FitResult, StoredParts};
    use glmm::{Simulator, WeightPower};

    const HALF: Option<WeightPower> = Some(WeightPower::Fixed(0.5));
    const ONE: Option<WeightPower> = Some(WeightPower::Fixed(1.0));

    /// sleepstudy's numeric columns, Subject's codes and weights from 1 to 4.
    fn data() -> (HashMap<String, Vec<f64>>, Vec<f64>, Vec<f64>) {
        let d = rows(include_str!(
            "../../validation/data/empirical/sleepstudy.csv"
        ));
        let w: Vec<f64> = (0..d.len()).map(|i| 1.0 + (i % 4) as f64).collect();
        let half = w.iter().map(|v| v.sqrt()).collect();
        let numeric = HashMap::from([
            ("Reaction".to_string(), nums(&d, 0)),
            ("Days".to_string(), nums(&d, 1)),
        ]);
        (numeric, w, half)
    }

    fn ols(w: Vec<f64>, power: Option<WeightPower>) -> FitResult {
        let (numeric, ..) = data();
        run_fit(
            "Reaction ~ Days",
            numeric,
            HashMap::new(),
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            Some(w),
            None,
            None,
            true,
            true,
            None,
            "Wald",
            power,
            None,
        )
        .unwrap()
    }

    fn parts(r: &FitResult) -> StoredParts {
        StoredParts {
            beta: r.beta.clone(),
            vcov: r.vcov.clone(),
            model_vcov: r.model_vcov.clone(),
            varcorr: r.varcorr.clone(),
            ranef: r.ranef.clone(),
            ranef_levels: r.ranef_levels.clone(),
            condvar: r.condvar.clone(),
            condvar_x: r.condvar_x.clone(),
            condvar_cross: r.condvar_cross.clone(),
            dispersion: r.dispersion,
            weight_power: r.weight_power,
        }
    }

    #[test]
    fn ols_summaries_comparison_and_draws_read_the_powered_weights() {
        let (numeric, w, half) = data();
        let (powered, claimed) = (ols(w.clone(), HALF), ols(half.clone(), ONE));
        assert_eq!(powered.weight_power, 0.5);
        assert_eq!(powered.weights.as_deref(), Some(&w[..]));
        for (a, b) in [
            (powered.rss, claimed.rss),
            (powered.r_squared, claimed.r_squared),
            (powered.adj_r_squared, claimed.adj_r_squared),
        ] {
            assert_eq!(a.to_bits(), b.to_bits());
        }

        let compare = |w: Vec<f64>, power| {
            run_compare(
                "Reaction ~ 1",
                "Reaction ~ Days",
                numeric.clone(),
                HashMap::new(),
                "gaussian",
                "identity",
                1,
                Some(w),
                None,
                None,
                None,
                power,
                None,
            )
            .unwrap()
            .comparison
        };
        assert_eq!(compare(w.clone(), HALF), compare(half.clone(), ONE));

        let draws = |w: Vec<f64>, r: &FitResult| {
            simulate(
                "Reaction ~ Days",
                numeric.clone(),
                HashMap::new(),
                "gaussian",
                "identity",
                Some(w),
                None,
                &parts(r),
                3,
                7,
            )
            .unwrap()
        };
        let ours = draws(w.clone(), &powered);
        assert_eq!(ours, draws(half.clone(), &claimed));

        // The Rust door on a `Fit` made with the power, on the caller's `w`.
        let table = Table {
            columns: numeric
                .iter()
                .map(|(k, v)| (k.clone(), Column::Numeric(v.clone())))
                .collect(),
            n: w.len(),
        };
        let lo = lower("Reaction ~ Days", &table, Family::Gaussian).unwrap();
        let mut opts = FitOptions::default();
        opts.weight_power = HALF;
        let fit = glmm::fit(&lo.data().weights(&w), lo.spec(), &opts).unwrap();
        let design = lo.design().weights(&w);
        let sim = Simulator::new(&fit, lo.spec(), &design).unwrap();
        assert_eq!(sim.simulate(7, 0), ours[..w.len()]);
    }
}

/// Robust fits: the CR2 and HC2 standard errors and
/// df through the formula door, against clubSandwich, and the default.
mod robust_fits {
    use super::*;
    use glmm::{SeType, WeightPower};
    use serde_json::Value;

    /// A frozen reference under `validation/goldens/inference/cr2/`.
    fn golden(name: &str) -> Value {
        let path = format!(
            "{}/../validation/goldens/inference/cr2/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
    }

    fn numbers(v: &Value) -> Vec<f64> {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    }

    /// `validation/data/inference/<file>.csv` (header `g?,x1,x2,…,w,…,y`)
    /// lowered on `formula`, with `w` as the weights.
    fn lowered(file: &str, formula: &str) -> glmm::formula::Lowered {
        let path = format!(
            "{}/../validation/data/inference/{file}.csv",
            env!("CARGO_MANIFEST_DIR")
        );
        let raw = std::fs::read_to_string(&path).unwrap();
        let header: Vec<String> = raw
            .lines()
            .next()
            .unwrap()
            .split(',')
            .map(str::to_owned)
            .collect();
        let d = rows(&raw);
        let col = |name: &str| header.iter().position(|h| h == name).unwrap();
        let mut columns = vec![
            ("x1".to_string(), numeric(&d, col("x1"))),
            ("x2".to_string(), numeric(&d, col("x2"))),
            ("y".to_string(), numeric(&d, col("y"))),
        ];
        if header.iter().any(|h| h == "g") {
            columns.push(("g".to_string(), factor(&d, col("g"))));
        }
        let table = Table {
            columns,
            n: d.len(),
        };
        let mut lo = lower(formula, &table, Family::Gaussian).unwrap();
        lo.weights = Some(nums(&d, col("w")));
        lo
    }

    fn max_rel(got: &[f64], want: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .map(|(g, w)| (g / w - 1.0).abs())
            .fold(0.0, f64::max)
    }

    /// The eight `cr2_lme_*` cells: (reference, data, formula).
    const LME: [(&str, &str, &str); 8] = [
        (
            "cr2_lme_ri_fixed_s10",
            "cr2_ri_s10",
            "y ~ 1 + x1 + x2 + (1 | g)",
        ),
        (
            "cr2_lme_ri_estimated_s10",
            "cr2_ri_s10",
            "y ~ 1 + x1 + x2 + (1 | g)",
        ),
        (
            "cr2_lme_ri_fixed_s100",
            "cr2_ri_s100",
            "y ~ 1 + x1 + x2 + (1 | g)",
        ),
        (
            "cr2_lme_ri_estimated_s100",
            "cr2_ri_s100",
            "y ~ 1 + x1 + x2 + (1 | g)",
        ),
        ("cr2_lme_gls_fixed_n20", "hc2_n20", "y ~ 1 + x1 + x2"),
        ("cr2_lme_gls_estimated_n20", "hc2_n20", "y ~ 1 + x1 + x2"),
        ("cr2_lme_gls_fixed_n100", "hc2_n100", "y ~ 1 + x1 + x2"),
        ("cr2_lme_gls_estimated_n100", "hc2_n100", "y ~ 1 + x1 + x2"),
    ];

    /// clubSandwich's CR2 on nlme's `lme` (one grouping factor) and HC2
    /// on `gls`, REML, at the reference's power (1, or nlme's δ̂ shared),
    /// SE and df to 1e-7. The variance estimates are printed beside them:
    /// nlme settles τ² to about 3e-7 relative, which bounds what the SEs can
    /// share.
    #[test]
    fn cr2_and_hc2_match_club_sandwich_on_nlme_fits() {
        for (name, data, formula) in LME {
            let g = golden(name);
            let mut opts = FitOptions::default();
            opts.weight_power = Some(WeightPower::Fixed(g["delta"].as_f64().unwrap()));
            opts.se_type = Some(SeType::Robust);
            opts.reml = Some(true);
            let model = glmm::formula::fit_lowered(formula, lowered(data, formula), &opts).unwrap();
            let fit = model.fit();
            assert_eq!(fit.se_type(), SeType::Robust);
            let (se, df) = (
                max_rel(fit.se(), &numbers(&g["se"])),
                max_rel(fit.coef_df(), &numbers(&g["df"])),
            );
            let beta = max_rel(fit.beta(), &numbers(&g["beta"]));
            let s2 = (fit.dispersion() / g["sigma2"].as_f64().unwrap() - 1.0).abs();
            let t2 = g["tau2"]
                .as_f64()
                .map_or(0.0, |t| (fit.varcorr(0)[0] / t - 1.0).abs());
            eprintln!(
                "G5 {name}: se {se:.2e} df {df:.2e} beta {beta:.2e} sigma2 {s2:.2e} tau2 {t2:.2e}"
            );
            assert!(se < 1e-7 && df < 1e-7, "{name}: se {se:.2e} df {df:.2e}");
        }
    }

    fn opts(power: Option<WeightPower>, se: Option<SeType>, reml: Option<bool>) -> FitOptions {
        let mut o = FitOptions::default();
        o.weight_power = power;
        o.se_type = se;
        o.reml = reml;
        o
    }

    fn fit_on(data: &str, formula: &str, o: &FitOptions) -> glmm::formula::Model {
        glmm::formula::fit_lowered(formula, lowered(data, formula), o).unwrap()
    }

    const RI: &str = "y ~ 1 + x1 + x2 + (1 | g)";
    const WLS: &str = "y ~ 1 + x1 + x2";
    const ONE: Option<WeightPower> = Some(WeightPower::Fixed(1.0));
    const ROBUST: Option<SeType> = Some(SeType::Robust);

    /// G4 through the formula door: claimed weights with a named robust SE
    /// are HC2 with the Bell–McCaffrey df, to 1e-10 of clubSandwich on `lm`.
    #[test]
    fn claimed_weights_with_robust_se_are_hc2() {
        for name in ["hc2_n20", "hc2_n100"] {
            let g = golden(name);
            let m = fit_on(name, WLS, &opts(ONE, ROBUST, None));
            let fit = m.fit();
            assert_eq!(fit.se_type(), SeType::Robust);
            assert_eq!(fit.weight_power(), 1.0);
            let (se, df) = (
                max_rel(fit.se(), &numbers(&g["se"])),
                max_rel(fit.coef_df(), &numbers(&g["df"])),
            );
            let p = fit.beta().len();
            let want: Vec<f64> = g["vcov"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(numbers)
                .collect();
            let vcov = fit
                .vcov()
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / want[0].abs().max(want[p + 1].abs()))
                .fold(0.0, f64::max);
            eprintln!("G4 {name}: se {se:.2e} df {df:.2e} vcov {vcov:.2e}");
            assert!(se < 1e-10 && df < 1e-10 && vcov < 1e-10, "{name}");
        }
    }

    /// The default `se_type`: robust on a REML δ fit with one
    /// grouping factor and no random slope, or with none; the model SE on a
    /// δ fit with a random slope, an ML δ fit, claimed weights and no weights.
    #[test]
    fn default_se_type_rows() {
        let rs = "y ~ 1 + x1 + x2 + (1 + x1 | g)";
        let est = Some(WeightPower::Estimate);
        let rows: [(&str, &str, &str, FitOptions, SeType); 6] = [
            (
                "δ fit, one grouping",
                "cr2_ri_s10",
                RI,
                opts(None, None, None),
                SeType::Robust,
            ),
            (
                "δ fit, no grouping",
                "hc2_n20",
                WLS,
                opts(None, None, None),
                SeType::Robust,
            ),
            (
                "δ fit, random slope",
                "cr2_rs_s10",
                rs,
                opts(None, None, None),
                SeType::Model,
            ),
            (
                "ML δ fit",
                "cr2_ri_s10",
                RI,
                opts(est, None, Some(false)),
                SeType::Model,
            ),
            (
                "claimed weights",
                "cr2_ri_s10",
                RI,
                opts(ONE, None, None),
                SeType::Model,
            ),
            (
                "fixed power 0.5",
                "hc2_n20",
                WLS,
                opts(Some(WeightPower::Fixed(0.5)), None, None),
                SeType::Model,
            ),
        ];
        for (what, data, formula, o, want) in rows {
            let m = fit_on(data, formula, &o);
            assert_eq!(m.fit().se_type(), want, "{what}");
        }
        // Without weights.
        let mut lo = lowered("cr2_ri_s10", RI);
        lo.weights = None;
        let m = glmm::formula::fit_lowered(RI, lo, &FitOptions::default()).unwrap();
        assert_eq!(m.fit().se_type(), SeType::Model);
    }

    /// End to end: the default call on WLS data is the δ fit with
    /// HC2; with a grouping factor, CR2. Each equals the fit named at its own
    /// δ̂ with `se_type = Robust`, and nlme's δ̂ is printed beside it.
    #[test]
    fn default_weighted_call_is_the_delta_fit_with_robust_se() {
        for (data, formula, reference) in [
            ("hc2_n20", WLS, "cr2_lme_gls_estimated_n20"),
            ("hc2_n100", WLS, "cr2_lme_gls_estimated_n100"),
            ("cr2_ri_s10", RI, "cr2_lme_ri_estimated_s10"),
        ] {
            let m = fit_on(data, formula, &FitOptions::default());
            let fit = m.fit();
            assert_eq!(fit.se_type(), SeType::Robust, "{data}");
            let delta = fit.weight_power();
            let named = fit_on(
                data,
                formula,
                &opts(Some(WeightPower::Fixed(delta)), ROBUST, Some(true)),
            );
            assert_eq!(fit.se(), named.fit().se(), "{data}");
            assert_eq!(fit.coef_df(), named.fit().coef_df(), "{data}");
            let g = golden(reference);
            let nlme = g["delta"].as_f64().unwrap();
            eprintln!(
                "default {data}: delta {delta:.6} (nlme {nlme:.6}), se gap {:.2e}",
                max_rel(fit.se(), &numbers(&g["se"]))
            );
        }
    }

    /// With `se_for` naming some columns, the robust `se`
    /// and `vcov` are NaN outside them and equal the full fit's on them.
    #[test]
    fn se_for_is_honoured_on_a_robust_fit() {
        let full = fit_on("cr2_ri_s10", RI, &opts(ONE, ROBUST, None));
        let mut o = opts(ONE, ROBUST, None);
        o.se_for = glmm::SeFor::Only(vec![1]);
        let some = fit_on("cr2_ri_s10", RI, &o);
        let (f, s) = (full.fit(), some.fit());
        let p = f.beta().len();
        for i in 0..p {
            assert_eq!(s.se()[i].is_nan(), i != 1, "se {i}");
            assert_eq!(s.coef_df()[i].is_nan(), i != 1, "df {i}");
            for j in 0..p {
                let v = s.vcov()[i * p + j];
                assert_eq!(v.is_nan(), i != 1 || j != 1, "vcov {i},{j}");
            }
        }
        assert_eq!(s.se()[1], f.se()[1]);
        assert_eq!(s.coef_df()[1], f.coef_df()[1]);
    }

    /// An aliased column on a robust fit is NaN in `se`,
    /// `vcov` and `coef_df`, the rest equals the fit without it, a contrast
    /// on it gets `AliasedColumn`, and the robust notes name caller columns.
    #[test]
    fn an_aliased_column_on_a_robust_fit() {
        let path = format!(
            "{}/../validation/data/inference/cr2_ri_s10.csv",
            env!("CARGO_MANIFEST_DIR")
        );
        let d = rows(&std::fs::read_to_string(&path).unwrap());
        let x1 = nums(&d, 1);
        let table = Table {
            columns: vec![
                ("g".to_string(), factor(&d, 0)),
                ("x1".to_string(), Column::Numeric(x1.clone())),
                (
                    "x1b".to_string(),
                    Column::Numeric(x1.iter().map(|v| 2.0 * v).collect()),
                ),
                ("x2".to_string(), numeric(&d, 2)),
                ("y".to_string(), numeric(&d, 6)),
            ],
            n: d.len(),
        };
        let fit_of = |formula: &str| {
            let mut lo = lower(formula, &table, Family::Gaussian).unwrap();
            lo.weights = Some(nums(&d, 4));
            glmm::formula::fit_lowered(formula, lo, &opts(ONE, ROBUST, None)).unwrap()
        };
        let plain = fit_of(RI);
        let aliased = fit_of("y ~ 1 + x1 + x1b + x2 + (1 | g)");
        let (a, b) = (aliased.fit(), plain.fit());
        assert_eq!(a.aliased(), &[false, false, true, false]);
        let keep = [0usize, 1, 3];
        for (r, &o) in keep.iter().enumerate() {
            assert!((a.se()[o] / b.se()[r] - 1.0).abs() < 1e-12, "se {o}");
            assert!(
                (a.coef_df()[o] / b.coef_df()[r] - 1.0).abs() < 1e-12,
                "df {o}"
            );
        }
        assert!(a.se()[2].is_nan() && a.coef_df()[2].is_nan());
        assert!((0..4).all(|j| a.vcov()[2 * 4 + j].is_nan() && a.vcov()[j * 4 + 2].is_nan()));
        let t = a.contrast_t(&[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(
            t.notes,
            vec![glmm::inference::contrast::ContrastNote::AliasedColumn { column: 2 }]
        );
        let small = |fit: &Fit| {
            fit.diagnostics()
                .notes
                .iter()
                .find_map(|n| match n {
                    Note::RobustDfSmall { columns, df } => Some((columns.clone(), df.clone())),
                    _ => None,
                })
                .expect("x2 has a robust df below 4")
        };
        let (cols, df) = small(a);
        let (want_cols, want_df) = small(b);
        assert_eq!(
            cols,
            want_cols
                .iter()
                .map(|&c| keep[c as usize] as u32)
                .collect::<Vec<_>>()
        );
        assert!(cols.contains(&3));
        assert_eq!(df, want_df);
    }

    /// The robust notes: a named robust SE on a random slope, a coefficient
    /// only one cluster informs, and a df below 4.
    #[test]
    fn robust_notes() {
        let kinds = |fit: &Fit| -> Vec<String> {
            fit.diagnostics()
                .notes
                .iter()
                .map(|n| format!("{n:?}"))
                .collect()
        };
        let slope = fit_on(
            "cr2_rs_s10",
            "y ~ 1 + x1 + x2 + (1 + x1 | g)",
            &opts(ONE, ROBUST, None),
        );
        assert!(kinds(slope.fit())
            .iter()
            .any(|k| k == "RobustWithRandomSlope"));
        let model = fit_on(
            "cr2_rs_s10",
            "y ~ 1 + x1 + x2 + (1 + x1 | g)",
            &opts(ONE, None, None),
        );
        assert!(!kinds(model.fit()).iter().any(|k| k.starts_with("Robust")));

        // A column that is non-zero in one cluster only: that cluster alone
        // identifies its coefficient.
        let path = format!(
            "{}/../validation/data/inference/cr2_ri_s10.csv",
            env!("CARGO_MANIFEST_DIR")
        );
        let d = rows(&std::fs::read_to_string(&path).unwrap());
        let first: Vec<f64> = d.iter().map(|r| f64::from(r[0] == "1")).collect();
        let table = Table {
            columns: vec![
                ("g".to_string(), factor(&d, 0)),
                ("x1".to_string(), numeric(&d, 1)),
                ("one".to_string(), Column::Numeric(first)),
                ("y".to_string(), numeric(&d, 6)),
            ],
            n: d.len(),
        };
        let formula = "y ~ 1 + x1 + one + (1 | g)";
        let mut lo = lower(formula, &table, Family::Gaussian).unwrap();
        lo.weights = Some(nums(&d, 4));
        let m = glmm::formula::fit_lowered(formula, lo, &opts(ONE, ROBUST, None)).unwrap();
        let fit = m.fit();
        assert!(fit.se()[2].is_nan() && fit.coef_df()[2].is_nan());
        assert!(fit.se()[1].is_finite());
        assert!(fit
            .diagnostics()
            .notes
            .iter()
            .any(|n| matches!(n, Note::RobustSeUnidentified { columns } if columns == &[2])));
    }

    fn unit(p: usize, j: usize) -> Vec<f64> {
        (0..p).map(|k| f64::from(u8::from(k == j))).collect()
    }

    fn rel(a: f64, b: f64) -> f64 {
        (a / b - 1.0).abs()
    }

    /// On a robust fit `contrast_t` reads the pieces, the Satterthwaite
    /// pass off too. A unit vector gives the coefficient's robust SE and df,
    /// another contrast `√(lᵀVl)` on its own df; one-row `contrast_f` is t²
    /// on that df, and a joint test is the model fit's F with
    /// `ModelBasedJointTest` (none without the pass). x2 has df below 4.
    #[test]
    fn contrasts_of_a_robust_fit() {
        use glmm::inference::contrast::{ContrastError, ContrastNote, NoDfReason};
        let mut o = opts(ONE, ROBUST, None);
        o.satterthwaite = false;
        let m = fit_on("cr2_ri_s10", RI, &o);
        let fit = m.fit();
        let p = fit.beta().len();
        for j in 0..p {
            let t = fit.contrast_t(&unit(p, j)).unwrap();
            assert_eq!((t.se, t.df), (fit.se()[j], fit.coef_df()[j]), "{j}");
            let small = fit.coef_df()[j] < 4.0;
            assert_eq!(
                t.notes,
                if small {
                    vec![ContrastNote::RobustDfSmall]
                } else {
                    vec![]
                }
            );
        }
        assert!(fit.coef_df()[2] < 4.0 && fit.coef_df()[1] >= 4.0);
        let l = [0.0, 1.0, -1.0];
        let t = fit.contrast_t(&l).unwrap();
        let v = fit.vcov();
        let var: f64 = (0..p)
            .flat_map(|i| (0..p).map(move |j| (i, j)))
            .map(|(i, j)| l[i] * l[j] * v[i * p + j])
            .sum();
        assert!(rel(t.se, var.sqrt()) < 1e-14);
        assert!(t.df.is_finite() && t.df > 0.0 && t.df != fit.coef_df()[1]);
        assert_eq!(t.t, t.estimate / t.se);
        let f = fit.contrast_f(&l, 1).unwrap();
        assert_eq!(
            (f.f, f.num_df, f.den_df, f.p_value, &f.notes),
            (t.t * t.t, 1.0, t.df, t.p_value, &t.notes)
        );
        // `lᵀVl` must not overflow at 1e160 or underflow at 1e-160.
        for c in [1e160, 1e-160] {
            let lc: Vec<f64> = l.iter().map(|v| v * c).collect();
            let tc = fit.contrast_t(&lc).unwrap();
            assert!(
                rel(tc.t, t.t) < 1e-12 && rel(tc.se, c * t.se) < 1e-12,
                "{c}: {tc:?}"
            );
            assert!(rel(tc.df, t.df) < 1e-12, "{c}: {tc:?}");
        }

        let two = [unit(p, 1), unit(p, 2)].concat();
        assert_eq!(
            fit.contrast_f(&two, 2),
            Err(ContrastError::NoDf(NoDfReason::Disabled))
        );
        let robust = fit_on("cr2_ri_s10", RI, &opts(ONE, ROBUST, None));
        let model = fit_on("cr2_ri_s10", RI, &opts(ONE, Some(SeType::Model), None));
        let (r, mm) = (
            robust.fit().contrast_f(&two, 2).unwrap(),
            model.fit().contrast_f(&two, 2).unwrap(),
        );
        assert!(rel(r.f, mm.f) < 1e-12 && rel(r.den_df, mm.den_df) < 1e-12);
        assert_eq!(r.num_df, 2.0);
        assert_eq!(r.notes, vec![ContrastNote::ModelBasedJointTest]);
        // An all-zero one-row L takes the model F's rank-0 result.
        let zero = robust.fit().contrast_f(&vec![0.0; p], 1).unwrap();
        assert_eq!(zero.num_df, 0.0);
        assert!(zero.f.is_nan() && zero.den_df.is_nan() && zero.p_value.is_nan());
        assert_eq!(
            zero.notes,
            vec![ContrastNote::RankDeficientL { rank: 0, rows: 1 }]
        );
    }

    /// A contrast one cluster alone identifies gets a NaN SE and df with
    /// `RobustUnidentified`; one that leaves that column out keeps its CR2.
    #[test]
    fn a_contrast_one_cluster_identifies() {
        use glmm::inference::contrast::ContrastNote;
        let path = format!(
            "{}/../validation/data/inference/cr2_ri_s10.csv",
            env!("CARGO_MANIFEST_DIR")
        );
        let d = rows(&std::fs::read_to_string(&path).unwrap());
        let first: Vec<f64> = d.iter().map(|r| f64::from(r[0] == "1")).collect();
        let table = Table {
            columns: vec![
                ("g".to_string(), factor(&d, 0)),
                ("x1".to_string(), numeric(&d, 1)),
                ("one".to_string(), Column::Numeric(first)),
                ("y".to_string(), numeric(&d, 6)),
            ],
            n: d.len(),
        };
        let formula = "y ~ 1 + x1 + one + (1 | g)";
        let mut lo = lower(formula, &table, Family::Gaussian).unwrap();
        lo.weights = Some(nums(&d, 4));
        let m = glmm::formula::fit_lowered(formula, lo, &opts(ONE, ROBUST, None)).unwrap();
        let fit = m.fit();
        for l in [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]] {
            let t = fit.contrast_t(&l).unwrap();
            assert!(t.estimate.is_finite());
            assert!(t.se.is_nan() && t.df.is_nan() && t.p_value.is_nan());
            assert_eq!(t.notes, vec![ContrastNote::RobustUnidentified]);
            let f = fit.contrast_f(&l, 1).unwrap();
            assert!(f.f.is_nan() && f.den_df.is_nan());
            assert_eq!(f.notes, vec![ContrastNote::RobustUnidentified]);
        }
        let t = fit.contrast_t(&[0.0, 1.0, 0.0]).unwrap();
        assert_eq!((t.se, t.df), (fit.se()[1], fit.coef_df()[1]));
    }

    /// For contrasts, with `se_for` naming x1 only, a robust
    /// contrast on x1 equals the full fit's, and one touching another
    /// column is NaN.
    #[test]
    fn a_contrast_outside_se_for_is_nan_on_a_robust_fit() {
        let full = fit_on("cr2_ri_s10", RI, &opts(ONE, ROBUST, None));
        let mut o = opts(ONE, ROBUST, None);
        o.se_for = glmm::SeFor::Only(vec![1]);
        let some = fit_on("cr2_ri_s10", RI, &o);
        let e1 = [0.0, 2.0, 0.0];
        assert_eq!(some.fit().contrast_t(&e1), full.fit().contrast_t(&e1));
        for l in [[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]] {
            let t = some.fit().contrast_t(&l).unwrap();
            assert!(t.estimate.is_finite(), "{l:?}");
            assert!(
                t.se.is_nan() && t.df.is_nan() && t.notes.is_empty(),
                "{l:?}"
            );
        }
    }

    /// For contrasts, on a robust fit with an aliased column,
    /// a contrast that leaves it out equals the fit without the column, in
    /// caller indices.
    #[test]
    fn a_contrast_beside_an_aliased_column_on_a_robust_fit() {
        let path = format!(
            "{}/../validation/data/inference/cr2_ri_s10.csv",
            env!("CARGO_MANIFEST_DIR")
        );
        let d = rows(&std::fs::read_to_string(&path).unwrap());
        let x1 = nums(&d, 1);
        let table = Table {
            columns: vec![
                ("g".to_string(), factor(&d, 0)),
                ("x1".to_string(), Column::Numeric(x1.clone())),
                (
                    "x1b".to_string(),
                    Column::Numeric(x1.iter().map(|v| 2.0 * v).collect()),
                ),
                ("x2".to_string(), numeric(&d, 2)),
                ("y".to_string(), numeric(&d, 6)),
            ],
            n: d.len(),
        };
        let fit_of = |formula: &str| {
            let mut lo = lower(formula, &table, Family::Gaussian).unwrap();
            lo.weights = Some(nums(&d, 4));
            glmm::formula::fit_lowered(formula, lo, &opts(ONE, ROBUST, None)).unwrap()
        };
        let plain = fit_of(RI);
        let aliased = fit_of("y ~ 1 + x1 + x1b + x2 + (1 | g)");
        let a = aliased.fit().contrast_t(&[0.0, 1.0, 0.0, -1.0]).unwrap();
        let b = plain.fit().contrast_t(&[0.0, 1.0, -1.0]).unwrap();
        assert!(rel(a.estimate, b.estimate) < 1e-12);
        assert!(rel(a.se, b.se) < 1e-12 && rel(a.df, b.df) < 1e-12);
        assert_eq!(a.notes, b.notes);
    }
}

/// A table with no rows is refused by the fit entry, not fitted to a NaN fit
/// with its empty grouping dropped as "one level"; also when a fixed effect is
/// a factor, which would otherwise be refused as a factor with one level.
#[test]
fn zero_row_table_is_refused() {
    let empty = || Table {
        columns: ["y", "x", "g"]
            .into_iter()
            .map(|c| (c.into(), Column::Numeric(vec![])))
            .collect(),
        n: 0,
    };
    let numeric = empty();
    let mut with_factor = empty();
    with_factor
        .columns
        .push(("f".into(), Column::factor_from_labels(&[])));
    let cases = [
        ("y ~ x + (1 | g)", &numeric),
        ("y ~ f + x + (1 | g)", &with_factor),
    ];
    for (formula, table) in cases {
        let err = glmm::formula::fit(formula, table, Family::Gaussian, &FitOptions::default())
            .err()
            .expect("zero rows must be refused");
        assert!(
            matches!(err, glmm::formula::Error::Fit(glmm::FitError::Shape(_))),
            "{formula}: {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "Data has no rows (n = 0); a fit needs at least one row",
            "{formula}"
        );
    }
}

/// A repeated column name or a repeated level label inside a factor is
/// refused: `Table` lookups by name, and factor codes by label, would
/// otherwise read the first copy without a word.
#[test]
fn duplicate_column_names_and_factor_levels_are_refused() {
    let fit_err = |columns: Vec<(&str, Column)>| {
        let table = Table {
            columns: columns.into_iter().map(|(n, c)| (n.into(), c)).collect(),
            n: 4,
        };
        glmm::formula::fit("y ~ x", &table, Family::Gaussian, &FitOptions::default())
            .err()
            .expect("must be refused")
            .to_string()
    };
    let num = |v: [f64; 4]| Column::Numeric(v.to_vec());
    let e = fit_err(vec![
        ("y", num([1.0, 2.0, 3.0, 5.0])),
        ("x", num([0.0, 1.0, 2.0, 3.0])),
        ("x", num([3.0, 2.0, 1.0, 0.0])),
    ]);
    assert!(e.contains(r#"duplicate column name "x""#), "{e}");
    let e = fit_err(vec![
        ("y", num([1.0, 2.0, 3.0, 5.0])),
        (
            "x",
            Column::Factor {
                levels: vec!["a".into(), "a".into()],
                codes: vec![0, 1, 0, 1],
            },
        ),
    ]);
    assert!(
        e.contains(r#"factor column "x" has the level "a" more than once"#),
        "{e}"
    );
}
