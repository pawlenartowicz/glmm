#![cfg(feature = "formula")]
//! End-to-end: drive the validation datasets through `lower` → `fit_cold` and assert
//! the formula frontend lowers to the same design, the same grouping ids, and
//! therefore the same fit as the hand-built `ModelSpec`/`GroupIds` the in-crate
//! kernel tests use. Closes the loop formula → design → kernel.
//!
//! **Cross-engine agreement for these fits is not this file's claim.**
//! `tests/validation_oracle.rs` refits `sleepstudy_lmm`, `penicillin_lmm`,
//! `pastes_lmm`, `cbpp_agq_k1` and `grouseticks_agq_k1` from each golden's own
//! recorded formula — through this same frontend — and gates them against the
//! frozen lme4 values at `validation/tol.R`'s bands. This file used to read the same
//! five goldens at bands one to two orders of magnitude looser, which asserted
//! nothing the oracle tier did not already assert more tightly. Those reads are
//! gone. What is left is the one claim the oracle tier cannot make, because it
//! only ever fits through the frontend: that the frontend agrees with the
//! hand-built spec.

use glmm::formula::{lower, Column, Table};
use glmm::{
    fit_cold, Family, Fit, FitOptions, GroupIds, Grouping, GroupingRelation, ModelSpec,
    ReStructure, Sizing,
};

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

/// `target_indices = 0..p`, everything else defaulted — what `lower()` fills
/// `Lowered::opts` with.
fn opts_for(p: usize) -> FitOptions {
    FitOptions {
        target_indices: (0..p as u32).collect(),
        ..FitOptions::default()
    }
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
    assert_vec_close(&got.beta, &want.beta, &format!("{ctx}: beta"));
    assert_vec_close(&got.se, &want.se, &format!("{ctx}: se"));
    assert_vec_close(&got.tau2, &want.tau2, &format!("{ctx}: tau2"));
    assert_eq!(
        got.varcorr.len(),
        want.varcorr.len(),
        "{ctx}: varcorr block count"
    );
    for (k, (g, w)) in got.varcorr.iter().zip(&want.varcorr).enumerate() {
        assert_vec_close(g, w, &format!("{ctx}: varcorr[{k}]"));
    }
    assert!(
        close(got.dispersion, want.dispersion, PIN_REL_ITER, 0.0),
        "{ctx}: dispersion {} vs hand-built {}",
        got.dispersion,
        want.dispersion
    );
    assert!(
        close(got.loglik, want.loglik, PIN_REL_ITER, 0.0),
        "{ctx}: loglik {} vs hand-built {}",
        got.loglik,
        want.loglik
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
    assert_eq!(
        lo.opts.target_indices,
        (0..hand.p as u32).collect::<Vec<_>>(),
        "{ctx}: target indices"
    );

    let via_formula = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let by_hand = fit_cold(
        &hand.x,
        &hand.y,
        hand.y.len(),
        hand.p,
        &hand.model,
        &hand.ids,
        &opts_for(hand.p),
    );
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

/// A random-intercept `ReStructure` over `extras` additional groupings. Cluster
/// counts in `ModelSpec` are placeholders — `Lowered::model`'s own doc says the
/// kernel re-derives real level counts from `ids` — so they are written as the
/// true counts here only to keep the spec readable.
fn intercept_re(primary_clusters: u32, extras: Vec<Grouping>) -> ReStructure {
    ReStructure {
        sizing: Sizing::FixedClusters {
            n_clusters: primary_clusters,
        },
        slopes: vec![],
        extra_groupings: extras,
    }
}

// ── LMM ──────────────────────────────────────────────────────────────────────

#[test]
fn sleepstudy_random_slope_matches_hand_built_spec() {
    let data = rows(include_str!("../validation/data/empirical/sleepstudy.csv"));
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
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(ReStructure {
            sizing: Sizing::FixedClusters {
                n_clusters: n_levels(&data, 2),
            },
            slopes: vec![1], // random slope on Days, design column 1
            extra_groupings: vec![],
        }),
    };
    let ids = GroupIds {
        primary: codes(&data, 2),
        extra: vec![],
    };
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

#[test]
fn penicillin_crossed_matches_hand_built_spec() {
    let data = rows(include_str!("../validation/data/empirical/Penicillin.csv"));
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
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(intercept_re(
            n_levels(&data, 1),
            vec![Grouping {
                relation: GroupingRelation::Crossed {
                    n_clusters: n_levels(&data, 2),
                },
                slopes: vec![],
            }],
        )),
    };
    let ids = GroupIds {
        primary: codes(&data, 1),
        extra: vec![codes(&data, 2)],
    };
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
    let data = rows(include_str!("../validation/data/empirical/Pastes.csv"));
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
    let n_batch = n_levels(&data, 1);
    let n_sample = n_levels(&data, 3);
    let model = ModelSpec {
        family: Family::Gaussian,
        re: Some(intercept_re(
            n_batch,
            vec![Grouping {
                relation: GroupingRelation::NestedWithin {
                    n_per_parent: n_sample / n_batch,
                },
                slopes: vec![],
            }],
        )),
    };
    let ids = GroupIds {
        primary: codes(&data, 1),
        extra: vec![codes(&data, 3)],
    };
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
    let data = rows(include_str!("../validation/data/empirical/Pastes.csv"));
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
        matches!(extra.relation, GroupingRelation::NestedWithin { .. }),
        "flat (1|batch)+(1|sample) should be detected nested, got {:?}",
        extra.relation
    );

    let lo_exp = lower("strength ~ (1|batch/cask)", &explicit, Family::Gaussian).unwrap();
    let f_flat = fit_cold(
        &lo_flat.x,
        &lo_flat.y,
        lo_flat.n,
        lo_flat.p,
        &lo_flat.model,
        &lo_flat.ids,
        &lo_flat.opts,
    );
    let f_exp = fit_cold(
        &lo_exp.x,
        &lo_exp.y,
        lo_exp.n,
        lo_exp.p,
        &lo_exp.model,
        &lo_exp.ids,
        &lo_exp.opts,
    );
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

/// Guard the other direction: `(1|plate)+(1|sample)` on Penicillin is genuinely
/// crossed (every sample spans every plate), so the nesting detection must NOT
/// reclassify it — a false positive would corrupt the padded family-block
/// Cholesky.
#[test]
fn penicillin_stays_crossed() {
    let data = rows(include_str!("../validation/data/empirical/Penicillin.csv"));
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
    let extra = &lo.model.re.as_ref().unwrap().extra_groupings[0];
    assert!(
        matches!(extra.relation, GroupingRelation::Crossed { .. }),
        "crossed (1|plate)+(1|sample) must stay Crossed, got {:?}",
        extra.relation
    );
}

// ── GLMM ─────────────────────────────────────────────────────────────────────

/// cbpp as expanded Bernoulli rows. The aggregated form is the one the oracle
/// froze (and the one `tests/validation_oracle.rs` fits); here the point is only
/// that the frontend and the hand-built spec agree, so the cheaper expansion is
/// fine and matches what the in-crate kernel tests build.
#[test]
fn cbpp_binomial_matches_hand_built_spec() {
    let data = rows(include_str!("../validation/data/empirical/cbpp.csv"));
    // Each aggregated row contributes `size` rows: `incidence` ones then zeros.
    let mut expanded: Vec<Vec<String>> = Vec::new();
    let mut y = Vec::new();
    for r in &data {
        let incidence: usize = r[1].parse().unwrap();
        let size: usize = r[2].parse().unwrap();
        for k in 0..size {
            y.push(f64::from(k < incidence));
            expanded.push(r.clone());
        }
    }
    let n = y.len();
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(y.clone())),
            ("period".into(), factor(&expanded, 3)),
            ("herd".into(), factor(&expanded, 0)),
        ],
        n,
    };
    // period is a 4-level factor → 3 treatment-contrast dummies after the
    // intercept; herd is the grouping.
    let x = design(n, &dummies(&expanded, 3));
    let model = ModelSpec {
        family: Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
        re: Some(intercept_re(n_levels(&expanded, 0), vec![])),
    };
    let ids = GroupIds {
        primary: codes(&expanded, 0),
        extra: vec![],
    };
    let hand = HandBuilt {
        x,
        y,
        p: 4,
        model,
        ids,
        col_names: &["(Intercept)", "period2", "period3", "period4"],
    };
    assert_frontend_matches_hand_built(
        "y ~ period + (1|herd)",
        &table,
        Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
        &hand,
        "cbpp",
    );
}

#[test]
fn grouseticks_poisson_matches_hand_built_spec() {
    let data = rows(include_str!("../validation/data/empirical/grouseticks.csv"));
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
    let model = ModelSpec {
        family: Family::Poisson {
            link: glmm::PoissonLink::Log,
        },
        re: Some(intercept_re(n_levels(&data, 0), vec![])),
    };
    let ids = GroupIds {
        primary: codes(&data, 0),
        extra: vec![],
    };
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

#[test]
fn re_groups_intercept_only() {
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            (
                "g".into(),
                Column::factor_from_labels(&["a", "a", "b", "b"].map(String::from)),
            ),
        ],
        n: 4,
    };
    let lo = glmm::formula::lower("y ~ (1|g)", &table, Family::Gaussian).unwrap();
    assert_eq!(lo.re_groups.len(), 1);
    assert_eq!(lo.re_groups[0].name, "g");
    assert_eq!(lo.re_groups[0].terms, vec!["(Intercept)".to_string()]);
}

#[test]
fn re_groups_intercept_and_slope() {
    let table = Table {
        columns: vec![
            ("Reaction".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            ("Days".into(), Column::Numeric(vec![0.0, 1.0, 0.0, 1.0])),
            (
                "Subject".into(),
                Column::factor_from_labels(&["s1", "s1", "s2", "s2"].map(String::from)),
            ),
        ],
        n: 4,
    };
    let lo = glmm::formula::lower(
        "Reaction ~ Days + (1 + Days | Subject)",
        &table,
        Family::Gaussian,
    )
    .unwrap();
    assert_eq!(lo.re_groups.len(), 1);
    assert_eq!(lo.re_groups[0].name, "Subject");
    assert_eq!(
        lo.re_groups[0].terms,
        vec!["(Intercept)".to_string(), "Days".to_string()]
    );
}

#[test]
fn re_groups_primary_and_extra() {
    // Primary = plate (declared first), extra = sample (declared second) —
    // both intercept-only, crossed. Order must mirror declaration order, which
    // is what `Fit::varcorr`/`Fit::tau2` blocks follow.
    let table = Table {
        columns: vec![
            ("diameter".into(), Column::Numeric(vec![1.0, 2.0, 3.0, 4.0])),
            (
                "plate".into(),
                Column::factor_from_labels(&["p1", "p1", "p2", "p2"].map(String::from)),
            ),
            (
                "sample".into(),
                Column::factor_from_labels(&["s1", "s2", "s1", "s2"].map(String::from)),
            ),
        ],
        n: 4,
    };
    let lo = glmm::formula::lower(
        "diameter ~ (1|plate) + (1|sample)",
        &table,
        Family::Gaussian,
    )
    .unwrap();
    assert_eq!(lo.re_groups.len(), 2);
    assert_eq!(lo.re_groups[0].name, "plate");
    assert_eq!(lo.re_groups[0].terms, vec!["(Intercept)".to_string()]);
    assert_eq!(lo.re_groups[1].name, "sample");
    assert_eq!(lo.re_groups[1].terms, vec!["(Intercept)".to_string()]);
}

#[test]
fn intercept_written_first_becomes_primary_over_slope() {
    // Formula order decides the primary grouping: `(1|g)` written before
    // `(1+x|h)` makes `g` primary even though the parser's slope stage runs
    // before its intercept stage. The slope block then sits on the EXTRA
    // grouping — which is itself a sparse-routing trigger (algorithms-lmm.md),
    // the orientation that re-landed validation rung 24 (`sim_sparse_gamma`).
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

#[test]
fn re_groups_categorical_slope_expands_dummies() {
    // Machine (3-level factor, base "A") as a random slope on Worker — mirrors
    // nlme::Machines' `score ~ 1 + Machine + (1 + Machine | Worker)`. Expect
    // Machine's 2 dummy columns (MachineB, MachineC) both as slope ColumnIds
    // and as ReGroupInfo term names, not the bare "Machine".
    // This lowering shape's fit-level oracle lives externally: validation rung 10
    // (nlme::Machines, `validation/manifest.json`).
    let table = Table {
        columns: vec![
            (
                "score".into(),
                Column::Numeric(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ),
            (
                "Machine".into(),
                Column::factor_from_labels(&["A", "B", "C", "A", "B", "C"].map(String::from)),
            ),
            (
                "Worker".into(),
                Column::factor_from_labels(&["w1", "w1", "w1", "w2", "w2", "w2"].map(String::from)),
            ),
        ],
        n: 6,
    };
    let lo = glmm::formula::lower(
        "score ~ Machine + (1 + Machine | Worker)",
        &table,
        Family::Gaussian,
    )
    .unwrap();

    // Fixed design: "(Intercept)", "MachineB", "MachineC" — MachineB/C at
    // ColumnIds 1, 2 (0 is the intercept).
    assert_eq!(
        lo.col_names,
        vec![
            "(Intercept)".to_string(),
            "MachineB".to_string(),
            "MachineC".to_string()
        ]
    );
    assert_eq!(lo.model.re.as_ref().unwrap().slopes, vec![1, 2]);

    assert_eq!(lo.re_groups.len(), 1);
    assert_eq!(lo.re_groups[0].name, "Worker");
    assert_eq!(
        lo.re_groups[0].terms,
        vec![
            "(Intercept)".to_string(),
            "MachineB".to_string(),
            "MachineC".to_string()
        ]
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
    const PLATE_SD: f64 = 0.846704431738483;
    const SAMPLE_SD: f64 = 1.9315583121903848;

    let data = rows(include_str!("../validation/data/empirical/Penicillin.csv"));
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
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(f.converged());

    // The shim's invariant: one name per varcorr block, same order.
    assert_eq!(
        lo.re_groups.len(),
        f.varcorr.len(),
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
        let got = f.varcorr[k][0].sqrt();
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

/// A factor's level order is the caller's, and level
/// 0 is the treatment-contrast base. Before this, `factor_levels` sorted every
/// factor through a `BTreeSet`, so `["low","med","high"]` silently based against
/// `"high"` — same fit quality, different β, a different question answered, and
/// nothing in the output to reveal it.
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
    // "high" — today's behavior, now a default rather than an imposition.
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
    let f1 = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let f2 = fit_cold(
        &lo2.x, &lo2.y, lo2.n, lo2.p, &lo2.model, &lo2.ids, &lo2.opts,
    );
    assert!(f1.converged() && f2.converged());
    // Intercept = mean of the base level: "low" ≈ 1.05 vs "high" ≈ 3.05.
    assert!(
        close(f1.beta[0], 1.05, 1e-6, 1e-6),
        "base low: {}",
        f1.beta[0]
    );
    assert!(
        close(f2.beta[0], 3.05, 1e-6, 1e-6),
        "base high: {}",
        f2.beta[0]
    );
}
