#![cfg(feature = "formula")]
//! End-to-end: drive the existing parity datasets through `lower` → `fit_cold`
//! and assert the fit matches the frozen lme4 goldens in `parity/goldens/` (d3
//! §6c). Closes the loop formula → design → kernel → oracle. The datasets and
//! reference values are the same ones the in-crate kernel tests use; here the
//! design/spec/ids are built from a formula string instead of by hand.

use glmm::formula::{lower, Column, Table};
use glmm::{fit_cold, Family};
use serde::Deserialize;

// ── Golden schema (a subset of the parity JSON) ──────────────────────────────

#[derive(Deserialize)]
struct Golden {
    coef_names: Vec<String>,
    estimates: Est,
}
#[derive(Deserialize)]
struct Est {
    beta: Vec<f64>,
    #[serde(default)]
    se: Option<Vec<f64>>, // LMM goldens
    #[serde(default)]
    se_hessian: Option<Vec<f64>>, // GLMM goldens (glmm default = WaldSe::Hessian)
    varcomp: Vec<Vc>,
}
#[derive(Deserialize)]
struct Vc {
    stddev: Vec<f64>,
    #[serde(default)]
    corr: Option<Vec<Vec<f64>>>,
    /// lme4's name for this variance block's grouping — the reference
    /// `Lowered::re_groups[k].name` must agree with (see
    /// `re_groups_align_with_varcorr_blocks`).
    #[serde(default)]
    group: String,
    #[serde(default)]
    terms: Vec<String>,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Parse a committed parity CSV into trimmed, unquoted string fields (mirrors the
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

/// `|a-b| <= atol + rtol*|b|`.
fn close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn assert_beta(got: &[f64], want: &[f64], rtol: f64, atol: f64, ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: beta length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            close(g, w, rtol, atol),
            "{ctx}: beta[{i}] = {g} vs lme4 {w}"
        );
    }
}

// ── LMM ──────────────────────────────────────────────────────────────────────

#[test]
fn sleepstudy_random_slope() {
    let data = rows(include_str!("../parity/data_empirical/sleepstudy.csv"));
    let table = Table {
        columns: vec![
            ("Reaction".into(), numeric(&data, 0)),
            ("Days".into(), numeric(&data, 1)),
            ("Subject".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    let g: Golden =
        serde_json::from_str(include_str!("../parity/goldens/sleepstudy_lmm.json")).unwrap();

    let lo = lower(
        "Reaction ~ Days + (1 + Days | Subject)",
        &table,
        Family::Gaussian,
    )
    .unwrap();
    assert_eq!(lo.col_names, g.coef_names);
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(f.converged, "sleepstudy did not converge");

    assert_beta(&f.beta, &g.estimates.beta, 1e-3, 1e-6, "sleepstudy");
    let se = g.estimates.se.unwrap();
    for (i, &w) in se.iter().enumerate() {
        assert!(close(f.se[i], w, 2e-2, 1e-6), "sleepstudy: se[{i}]");
    }
    // 2×2 RE covariance from varcorr[0] (vech col-major lower-tri [D00,D10,D11]).
    let vc = &f.varcorr[0];
    let (sd0, sd1) = (vc[0].sqrt(), vc[2].sqrt());
    let corr = vc[1] / (sd0 * sd1);
    let block = &g.estimates.varcomp[0];
    assert!(close(sd0, block.stddev[0], 2e-2, 1e-6), "sleepstudy: sd0");
    assert!(close(sd1, block.stddev[1], 2e-2, 1e-6), "sleepstudy: sd1");
    let want_corr = block.corr.as_ref().unwrap()[0][1];
    assert!(
        close(corr, want_corr, 0.0, 5e-2),
        "sleepstudy: corr {corr} vs {want_corr}"
    );
}

/// A scalar-intercept crossed/nested LMM: assert the intercept and each grouping's
/// variance component (`tau2` per grouping, declaration order).
fn scalar_lmm(csv: &str, golden: &str, formula: &str, table: Table, ctx: &str) {
    let g: Golden = serde_json::from_str(golden).unwrap();
    let _ = csv;
    let lo = lower(formula, &table, Family::Gaussian).unwrap();
    assert_eq!(lo.col_names, g.coef_names, "{ctx}: coef names");
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(f.converged, "{ctx} did not converge");
    assert_beta(&f.beta, &g.estimates.beta, 1e-3, 1e-6, ctx);
    // tau2[k] is grouping k's variance (declaration order: primary, then extras).
    for (k, block) in g.estimates.varcomp.iter().enumerate() {
        let got_sd = f.tau2[k].sqrt();
        assert!(
            close(got_sd, block.stddev[0], 2e-2, 1e-6),
            "{ctx}: grouping {k} sd {got_sd} vs lme4 {}",
            block.stddev[0]
        );
    }
}

#[test]
fn penicillin_crossed() {
    let data = rows(include_str!("../parity/data_empirical/Penicillin.csv"));
    let table = Table {
        columns: vec![
            ("diameter".into(), numeric(&data, 0)),
            ("plate".into(), factor(&data, 1)),
            ("sample".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    scalar_lmm(
        "",
        include_str!("../parity/goldens/penicillin_lmm.json"),
        "diameter ~ (1|plate) + (1|sample)",
        table,
        "penicillin",
    );
}

#[test]
fn pastes_nested() {
    let data = rows(include_str!("../parity/data_empirical/Pastes.csv"));
    let table = Table {
        columns: vec![
            ("strength".into(), numeric(&data, 0)),
            ("batch".into(), factor(&data, 1)),
            ("cask".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    // Golden varcomp order is [batch, cask:batch] — matches declaration order
    // (primary=batch, extra=nested cask).
    scalar_lmm(
        "",
        include_str!("../parity/goldens/pastes_lmm.json"),
        "strength ~ (1|batch/cask)",
        table,
        "pastes",
    );
}

/// Same Pastes fit as `pastes_nested`, but via the FLAT lme4 idiom
/// `(1|batch)+(1|sample)` (T3): `sample` is the globally-unique `batch:cask`
/// column, so it genuinely nests in `batch` and must be classified
/// `NestedWithin` — not `Crossed` — yet produce the identical fit. Asserts both
/// the relation (T3 fired) and the same lme4 golden as the explicit form.
#[test]
fn pastes_flat_nested() {
    let data = rows(include_str!("../parity/data_empirical/Pastes.csv"));
    let table = Table {
        columns: vec![
            ("strength".into(), numeric(&data, 0)),
            ("batch".into(), factor(&data, 1)),
            ("sample".into(), factor(&data, 3)),
        ],
        n: data.len(),
    };
    let lo = lower(
        "strength ~ (1|batch) + (1|sample)",
        &table,
        Family::Gaussian,
    )
    .unwrap();
    let extra = &lo.model.re.as_ref().unwrap().extra_groupings[0];
    assert!(
        matches!(extra.relation, glmm::GroupingRelation::NestedWithin { .. }),
        "flat (1|batch)+(1|sample) should be detected nested, got {:?}",
        extra.relation
    );
    // Golden varcomp order [batch, cask:batch] == [batch, sample] here.
    let g: Golden =
        serde_json::from_str(include_str!("../parity/goldens/pastes_lmm.json")).unwrap();
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(f.converged, "pastes flat-nested did not converge");
    assert_beta(&f.beta, &g.estimates.beta, 1e-3, 1e-6, "pastes_flat");
    for (k, block) in g.estimates.varcomp.iter().enumerate() {
        let got_sd = f.tau2[k].sqrt();
        assert!(
            close(got_sd, block.stddev[0], 2e-2, 1e-6),
            "pastes_flat: grouping {k} sd {got_sd} vs lme4 {}",
            block.stddev[0]
        );
    }
}

/// Guard the other direction: `(1|plate)+(1|sample)` on Penicillin is genuinely
/// crossed (every sample spans every plate), so T3 must NOT reclassify it as
/// nested — a false positive would corrupt the padded family-block Cholesky.
#[test]
fn penicillin_stays_crossed() {
    let data = rows(include_str!("../parity/data_empirical/Penicillin.csv"));
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
        matches!(extra.relation, glmm::GroupingRelation::Crossed { .. }),
        "crossed (1|plate)+(1|sample) must stay Crossed, got {:?}",
        extra.relation
    );
}

// ── GLMM ─────────────────────────────────────────────────────────────────────

#[test]
fn cbpp_binomial() {
    // Aggregated binomial → Bernoulli 0/1 rows (the kernel is Bernoulli; §8 makes
    // this the caller's prep). Each aggregated row contributes `size` rows.
    let data = rows(include_str!("../parity/data_empirical/cbpp.csv"));
    let (mut y, mut herd, mut period) = (Vec::new(), Vec::new(), Vec::new());
    for r in &data {
        let (herd_l, incidence, size, period_l) = (
            r[0].clone(),
            r[1].parse::<usize>().unwrap(),
            r[2].parse::<usize>().unwrap(),
            r[3].clone(),
        );
        for k in 0..size {
            y.push(if k < incidence { 1.0 } else { 0.0 });
            herd.push(herd_l.clone());
            period.push(period_l.clone());
        }
    }
    let n = y.len();
    let table = Table {
        columns: vec![
            ("y".into(), Column::Numeric(y)),
            ("period".into(), Column::factor_from_labels(&period)),
            ("herd".into(), Column::factor_from_labels(&herd)),
        ],
        n,
    };
    let g: Golden =
        serde_json::from_str(include_str!("../parity/goldens/cbpp_agq_k1.json")).unwrap();

    let lo = lower(
        "y ~ period + (1|herd)",
        &table,
        Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
    )
    .unwrap();
    assert_eq!(lo.col_names, g.coef_names);
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(f.converged, "cbpp did not converge");
    assert_beta(&f.beta, &g.estimates.beta, 1e-2, 1e-3, "cbpp");
    let se = g.estimates.se_hessian.unwrap();
    for (i, &w) in se.iter().enumerate() {
        assert!(
            close(f.se[i], w, 3e-2, 1e-3),
            "cbpp: se[{i}] = {} vs {w}",
            f.se[i]
        );
    }
    let sd = f.tau2[0].sqrt();
    assert!(
        close(sd, g.estimates.varcomp[0].stddev[0], 3e-2, 1e-3),
        "cbpp: herd sd {sd}"
    );
}

#[test]
fn grouseticks_poisson() {
    let data = rows(include_str!("../parity/data_empirical/grouseticks.csv"));
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
    let g: Golden =
        serde_json::from_str(include_str!("../parity/goldens/grouseticks_agq_k1.json")).unwrap();

    let lo = lower(
        "TICKS ~ YEAR + cHEIGHT + (1|INDEX)",
        &table,
        Family::Poisson {
            link: glmm::PoissonLink::Log,
        },
    )
    .unwrap();
    assert_eq!(lo.col_names, g.coef_names);
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(f.converged, "grouseticks did not converge");
    assert_beta(&f.beta, &g.estimates.beta, 1e-2, 1e-3, "grouseticks");
    let se = g.estimates.se_hessian.unwrap();
    for (i, &w) in se.iter().enumerate() {
        assert!(
            close(f.se[i], w, 3e-2, 1e-4),
            "grouseticks: se[{i}] = {} vs {w}",
            f.se[i]
        );
    }
    let sd = f.tau2[0].sqrt();
    assert!(
        close(sd, g.estimates.varcomp[0].stddev[0], 3e-2, 1e-3),
        "grouseticks: INDEX sd {sd}"
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
fn re_groups_categorical_slope_expands_dummies() {
    // Machine (3-level factor, base "A") as a random slope on Worker — mirrors
    // nlme::Machines' `score ~ 1 + Machine + (1 + Machine | Worker)`. Expect
    // Machine's 2 dummy columns (MachineB, MachineC) both as slope ColumnIds
    // and as ReGroupInfo term names, not the bare "Machine".
    // This lowering shape's fit-level oracle lives externally: parity rung 10
    // (nlme::Machines, `parity/manifest.json`).
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
    // This id-layout shape's fitted-model oracle lives externally: parity
    // rung 13 (cake, `parity/manifest.json`) exercises the `(1|recipe:replicate)`
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

/// Plan gate 1 (docs/GLMM/plans/2026-07-15-python-port-bugs-spec.md): `re_groups`
/// must line up index-for-index with `varcorr`, since `summary()` labels block
/// `k` with name `k` — a swap mislabels a variance component, which is a wrong
/// answer, not a cosmetic slip. The Rust shim asserts the lengths agree; this
/// pins the ORDER against the oracle.
///
/// Penicillin is the fixture where a swap is visible: two named crossed
/// groupings whose variances differ ~2.3× (lme4: plate sd 0.847, sample sd
/// 1.932). Pairing them the wrong way round contradicts the golden rather than
/// just renaming a block.
#[test]
fn re_groups_align_with_varcorr_blocks() {
    let data = rows(include_str!("../parity/data_empirical/Penicillin.csv"));
    let table = Table {
        columns: vec![
            ("diameter".into(), numeric(&data, 0)),
            ("plate".into(), factor(&data, 1)),
            ("sample".into(), factor(&data, 2)),
        ],
        n: data.len(),
    };
    let g: Golden =
        serde_json::from_str(include_str!("../parity/goldens/penicillin_lmm.json")).unwrap();
    let lo = lower(
        "diameter ~ (1|plate) + (1|sample)",
        &table,
        Family::Gaussian,
    )
    .unwrap();
    let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(f.converged);

    // The shim's invariant: one name per varcorr block, same order.
    assert_eq!(
        lo.re_groups.len(),
        f.varcorr.len(),
        "re_groups and varcorr must agree in length"
    );
    assert_eq!(
        g.estimates.varcomp.len(),
        2,
        "fixture must have 2 groupings"
    );

    for (k, block) in g.estimates.varcomp.iter().enumerate() {
        assert_eq!(
            lo.re_groups[k].name, block.group,
            "grouping name at block {k}"
        );
        assert_eq!(
            lo.re_groups[k].terms, block.terms,
            "term names at block {k}"
        );
        // The variance actually sitting in block k must be the one lme4 reports
        // for the grouping named at index k — this is what a swap breaks.
        let sd = f.varcorr[k][0].sqrt();
        assert!(
            close(sd, block.stddev[0], 2e-2, 1e-6),
            "block {k} ({}) sd {sd} vs lme4 {}",
            block.group,
            block.stddev[0]
        );
    }
    // Guard the guard: the two stddevs must actually differ, or a swap would
    // slip through the check above.
    let (a, b) = (
        g.estimates.varcomp[0].stddev[0],
        g.estimates.varcomp[1].stddev[0],
    );
    assert!(
        (a - b).abs() > 0.5,
        "fixture no longer distinguishes a swap"
    );
}

/// §6 of the Python-port spec: a factor's level order is the caller's, and level
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
    assert!(f1.converged && f2.converged);
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
