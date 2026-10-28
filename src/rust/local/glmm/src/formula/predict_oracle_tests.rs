//! The frozen `predict` references (`goldens_predict.R`) against
//! [`predict`]: lme4's predictions on each entry's new rows, lme4's and
//! glmmTMB's `se.fit`, and lme4's `ranef(condVar = TRUE)`. Per entry:
//!
//! - **At lme4's parameters**: the new rows through [`lower_new`] and
//!   [`predict_parts`] with lme4's fixed effects (mapped by column name) and
//!   conditional modes (mapped by level label), both modes and both scales,
//!   within the round-off of the row's sum, `(p + q)·ε·Σ|term|`, plus the
//!   storage term below.
//! - **End to end**: [`fit`] then [`predict`], within
//!   `Σ|x_j||Δβ_j| + Σ|z_k||Δb_k|` plus that round-off and the storage term,
//!   `Δ` the crate's parameters minus lme4's on the row; the unseen rows carry
//!   the notes, and each refused row is a [`FitError::Ids`] without
//!   `allow_new_levels`.
//! - On the mixed entries, the standard errors in band of lme4's or glmmTMB's
//!   `se.fit`, and the conditional covariances and SDs in band of lme4's.
//! - On `predict_glmm_sim_factorial_bern`, a refit under [`WaldSe::Rx`]: its
//!   standard errors in band of lme4's `se.fit`, hard.
//!
//! The storage term: the references store 15 significant digits, so each
//! stored value is off its double by up to `h(v)`, half a unit in the 15th
//! significant digit of `v`. The row's sum carries `Σ|x_j|h(β_j) +
//! Σ|z_k|h(b_k)` from lme4's stored parameters, and the stored prediction its
//! own `h(want)`.
//!
//! On the response scale a band's link-scale part is multiplied by `|dμ/dη|`,
//! the larger of the two engines' values; `h(want)` is the rounding of the
//! stored response itself and is not. The first two parts and the
//! [`WaldSe::Rx`] refit are hard; the standard errors and conditional
//! variances of the default fit are registry-backed: a value over its band
//! passes only under an `oracle-tier` entry of
//! `validation/grid/divergences.json` on the reference with quantity `se_fit`
//! or `condvar`, and an entry on a compared reference that does not fire
//! fails as stale.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use super::materialize::lower_new;
use super::{fit, label_ranef, predict, Column, Error, Lowered, Model, Table};
use crate::family::mu_eta;
use crate::inference::rlrt_refs::csv;
use crate::inference::test_refs::{self, Coverage, DivEntry, Entry, Kind};
use crate::lmm::satterthwaite::tests::gate1::{numbers, strings};
use crate::lmm::slope_value;
use crate::orchestrate::family_from_str;
use crate::predict::{predict_parts, reported_columns, FitParts};
use crate::test_support::csv_table;
use crate::{Design, Family, FitError, FitOptions, Note, PredictOptions, Scale, WaldSe};

/// Mirrors `tol::SE_REL` in `tests/oracle_support/mod.rs`; change together.
const SE_REL: f64 = 1e-3;

/// The entry whose `se.fit` is checked against a [`WaldSe::Rx`] refit.
const RX_ENTRY: &str = "predict_glmm_sim_factorial_bern";

/// The registry-backed quantities of this driver.
const QUANTITIES: [&str; 2] = ["se_fit", "condvar"];

/// The rows `rows` of `table`, factor levels kept.
fn rows_of(table: &Table, rows: &[usize]) -> Table {
    let columns = table
        .columns
        .iter()
        .map(|(name, col)| {
            let col = match col {
                Column::Numeric(v) => Column::Numeric(rows.iter().map(|&i| v[i]).collect()),
                Column::Factor { levels, codes } => Column::Factor {
                    levels: levels.clone(),
                    codes: rows.iter().map(|&i| codes[i]).collect(),
                },
            };
            (name.clone(), col)
        })
        .collect();
    Table {
        columns,
        n: rows.len(),
    }
}

/// The crate grouping that lme4's `name` is, and whether its labels are
/// lme4's with the `:` parts reversed: lme4 names a nested inner grouping
/// `cask:batch` and its levels `a:B`, the lowering `batch:cask` and `B:a`.
fn grouping_of(lo: &Lowered, name: &str) -> (usize, bool) {
    let reversed: String = name.rsplit(':').collect::<Vec<_>>().join(":");
    if let Some(g) = lo.re_groups.iter().position(|g| g.name == name) {
        return (g, false);
    }
    let g = lo
        .re_groups
        .iter()
        .position(|g| g.name == reversed)
        .unwrap_or_else(|| panic!("no grouping {name}"));
    (g, name.contains(':'))
}

fn label(lme4: &str, reversed: bool) -> String {
    if reversed {
        lme4.rsplit(':').collect::<Vec<_>>().join(":")
    } else {
        lme4.to_string()
    }
}

/// lme4's fixed effects in the lowering's column order.
fn lme4_beta(json: &Value, lo: &Lowered) -> Vec<f64> {
    let block = json.get("fixef").unwrap_or(&json["coef"]);
    let names = strings(&block["names"], "fixef names");
    let values = numbers(&block["values"], "fixef values");
    assert_eq!(names.len(), lo.p, "lme4 and the lowering have the same p");
    lo.col_names
        .iter()
        .map(|c| {
            let k = names
                .iter()
                .position(|n| n == c)
                .unwrap_or_else(|| panic!("lme4 has no column {c}"));
            values[k]
        })
        .collect()
}

/// lme4's conditional modes in the layout of [`crate::Fit::ranef`]: per
/// grouping, per slot, the grouping's terms; zero on a padded slot.
fn lme4_ranef(json: &Value, lo: &Lowered) -> Vec<f64> {
    let mut blocks: Vec<Vec<f64>> = lo
        .re_groups
        .iter()
        .map(|g| vec![f64::NAN; g.slot_labels.len() * g.terms.len()])
        .collect();
    for r in json["ranef"].as_array().expect("ranef") {
        let (g, reversed) = grouping_of(lo, r["grouping"].as_str().expect("grouping"));
        let info = &lo.re_groups[g];
        let q = info.terms.len();
        let terms = strings(&r["terms"], "ranef terms");
        let levels = strings(&r["levels"], "ranef levels");
        let values = r["values"].as_array().expect("ranef values");
        assert_eq!(terms.len(), q, "{}: terms", info.name);
        for (l, slot) in info.slot_labels.iter().enumerate() {
            let Some(slot) = slot else {
                blocks[g][l * q..(l + 1) * q].fill(0.0);
                continue;
            };
            let k = levels
                .iter()
                .position(|v| &label(v, reversed) == slot)
                .unwrap_or_else(|| panic!("{}: lme4 has no level {slot}", info.name));
            let row = numbers(&values[k], "ranef row");
            for (t, term) in info.terms.iter().enumerate() {
                let c = terms.iter().position(|n| n == term).expect("term");
                blocks[g][l * q + t] = row[c];
            }
        }
    }
    let out: Vec<f64> = blocks.concat();
    assert!(out.iter().all(|v| v.is_finite()), "every slot mapped");
    out
}

/// Row `i`'s random-effect part on `design`: per grouping, the start of the
/// row's level block in `ranef` (`None` for a level the fit did not see) and
/// the row's `z` values of that block.
fn row_effects(
    spec: &crate::ModelSpec,
    design: &Design<'_>,
    ranef_levels: &[usize],
    i: usize,
) -> Vec<(Option<usize>, Vec<f64>)> {
    let (Some(re), Some(ids)) = (spec.re.as_ref(), design.ids) else {
        return Vec::new();
    };
    let xm = faer::MatRef::from_row_major_slice(design.x, design.n, design.p);
    let zm = faer::MatRef::from_row_major_slice(design.re.unwrap_or(&[]), design.n, design.s);
    let mut at = 0;
    reported_columns(re, ids)
        .into_iter()
        .zip(ranef_levels)
        .map(|((ids, cols), &levels)| {
            let q = cols.len();
            let z = cols
                .iter()
                .map(|c| c.map_or(1.0, |c| slope_value(xm, zm, design.p, i, c)))
                .collect();
            let level = ids[i] as usize;
            let start = (level < levels).then_some(at + level * q);
            at += levels * q;
            (start, z)
        })
        .collect()
}

/// Row `i`'s `(Σ|term|, p + q)` at lme4's parameters `theirs` and
/// `Σ|x_j||Δβ_j| + Σ|z_k||Δb_k|`, `Δ` the parameters `ours` minus lme4's,
/// each pair `(β, ranef)`. The terms are `x_jβ_j`, the offset and the row's
/// `z_k b_k`; an empty `ranef` (population mode) reads no random effect.
fn row_bounds(
    spec: &crate::ModelSpec,
    design: &Design<'_>,
    levels: &[usize],
    ours: (&[f64], &[f64]),
    theirs: (&[f64], &[f64]),
    i: usize,
) -> (f64, usize, f64) {
    let x = &design.x[i * design.p..(i + 1) * design.p];
    let mut sum: f64 = x.iter().zip(theirs.0).map(|(v, b)| (v * b).abs()).sum();
    sum += design.offset.map_or(0.0, |o| o[i].abs());
    let mut delta: f64 = (0..design.p)
        .map(|j| (x[j] * (ours.0[j] - theirs.0[j])).abs())
        .sum();
    let mut count = design.p;
    if theirs.1.is_empty() {
        return (sum, count, delta);
    }
    for (start, z) in row_effects(spec, design, levels, i) {
        count += z.len();
        let Some(s) = start else { continue };
        for (r, v) in z.iter().enumerate() {
            sum += (v * theirs.1[s + r]).abs();
            delta += (v * (ours.1[s + r] - theirs.1[s + r])).abs();
        }
    }
    (sum, count, delta)
}

/// Half a unit in the 15th significant digit of `v`: how far a value stored
/// with 15 significant digits (jsonlite `digits = NA`) can sit from its double.
fn half_digit(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        0.5 * 10f64.powi(v.abs().log10().floor() as i32 - 14)
    }
}

/// `gap` within `band`; false when either is NaN.
fn in_band(gap: f64, band: f64) -> bool {
    gap <= band
}

/// The worst `gap / band` per entry and quantity, printed at the end.
#[derive(Default)]
struct Worst(BTreeMap<(String, String), (f64, String)>);

impl Worst {
    fn record(&mut self, name: &str, quantity: &str, ratio: f64, what: String) {
        let slot = self
            .0
            .entry((name.to_string(), quantity.to_string()))
            .or_insert((0.0, String::new()));
        if ratio > slot.0 || ratio.is_nan() {
            *slot = (ratio, what);
        }
    }
}

/// The registry-backed half: `se_fit` and `condvar`.
struct Driver {
    registry: Vec<DivEntry>,
    compared: BTreeSet<String>,
    fired: BTreeSet<String>,
    failures: Vec<String>,
}

impl Driver {
    fn new() -> Driver {
        let registry = test_refs::registry_entries()
            .into_iter()
            .filter(|e| {
                e.quantities
                    .iter()
                    .any(|q| QUANTITIES.contains(&q.as_str()))
            })
            .collect();
        Driver {
            registry,
            compared: BTreeSet::new(),
            fired: BTreeSet::new(),
            failures: Vec::new(),
        }
    }

    /// `got` in band of at least one `(engine, want, band)`; past every band,
    /// an entry on `name` listing `quantity` decides at the smallest relative
    /// difference. A NaN never passes.
    fn check(
        &mut self,
        worst: &mut Worst,
        (name, quantity, what): (&str, &str, &str),
        got: f64,
        refs: &[(&str, f64, f64)],
    ) {
        let ratio = refs
            .iter()
            .map(|&(_, want, band)| (got - want).abs() / band)
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        worst.record(name, quantity, ratio, what.to_string());
        if ratio <= 1.0 {
            return;
        }
        let rel = refs
            .iter()
            .map(|&(_, want, _)| (got - want).abs() / want.abs())
            .fold(f64::INFINITY, |a, b| if b < a { b } else { a });
        let against: Vec<String> = refs
            .iter()
            .map(|(e, want, band)| format!("{e} {want} (band {band:.3e})"))
            .collect();
        let msg = format!(
            "{name} {what}: {got} against {} (rel {rel:.3e})",
            against.join(", ")
        );
        match test_refs::covers(&self.registry, name, quantity, rel) {
            Coverage::Within { id } => {
                eprintln!("documented divergence {id}: {msg}");
                self.fired.insert(id);
            }
            Coverage::NotCovered => self.failures.push(msg),
            Coverage::Exceeds { id, max_rel } => self.failures.push(format!(
                "{msg}; documented divergence `{id}` covers only {max_rel:.1e}"
            )),
        }
    }

    /// Fails on the hard failures `hard` and the over-band values together,
    /// then on a stale entry.
    fn finish(&self, hard: &[String]) {
        let stale = test_refs::stale_entries(&self.registry, &self.compared, &self.fired);
        let all: Vec<&String> = hard.iter().chain(&self.failures).collect();
        assert!(
            all.is_empty(),
            "{} hard failure(s) and {} value(s) out of band:\n{}",
            hard.len(),
            self.failures.len(),
            all.iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
        assert!(
            stale.is_empty(),
            "inference divergence registry is out of date: these entries on a compared \
             reference no longer fire and must be deleted: {stale:?}"
        );
    }
}

/// The entry's family and its fit options: lme4's `REML` on a Gaussian mixed
/// fit, the defaults otherwise (Laplace on a GLMM).
fn family_and_options(e: &Entry) -> (Family, FitOptions) {
    let m = &e.raw["members"][0];
    let family = family_from_str(
        m["family"].as_str().expect("family"),
        m["link"].as_str().expect("link"),
    )
    .unwrap_or_else(|err| panic!("{}: {err}", e.name));
    let opts = FitOptions {
        reml: m["REML"].as_bool().filter(|_| family == Family::Gaussian),
        condvar: true,
        ..FitOptions::default()
    };
    (family, opts)
}

#[test]
fn predict_references() {
    let entries = test_refs::entries(Kind::Predict);
    let mut driver = Driver::new();
    let mut worst = Worst::default();
    let mut hard = Vec::new();
    let eps = f64::EPSILON;
    for e in &entries {
        let name = e.name.as_str();
        let json = test_refs::load(Kind::Predict, name);
        let factors: Vec<&str> = e.factors.iter().map(String::as_str).collect();
        let train = csv_table(&csv(e.data.as_deref().expect("data")), &factors);
        let newdata = e.raw["newdata"].as_str().expect("newdata");
        let new = csv_table(&csv(newdata), &factors);
        let formula = e.r_formula.as_deref().expect("formula");
        let (family, opts) = family_and_options(e);
        let model: Model =
            fit(formula, &train, family, &opts).unwrap_or_else(|err| panic!("{name}: {err}"));
        assert!(model.fit().converged(), "{name}: the fit converges");
        let lo = model.lowered();
        let spec = lo.spec();
        let glmm = model.fit();
        let rows = lower_new(formula, lo, &new).unwrap_or_else(|err| panic!("{name}: {err}"));
        let design = rows.design();
        let mixed = spec.re.is_some();
        let beta_ref = lme4_beta(&json, lo);
        let ranef_ref = if mixed {
            lme4_ranef(&json, lo)
        } else {
            Vec::new()
        };
        let at_lme4 = FitParts {
            beta: &beta_ref,
            ranef: &ranef_ref,
            condvar: &[],
            condvar_x: &[],
            ..FitParts::of(glmm, spec)
        };
        let modes: &[(&str, bool)] = if mixed {
            &[("population", true), ("conditional", false)]
        } else {
            &[("population", true)]
        };
        let levels = &glmm.ranef_levels;
        // lme4's stored parameters each moved by its storage rounding: their
        // row bound against the stored values is the storage term of the sum.
        let beta_stored: Vec<f64> = beta_ref.iter().map(|&v| v + half_digit(v)).collect();
        let ranef_stored: Vec<f64> = ranef_ref.iter().map(|&v| v + half_digit(v)).collect();
        for &(mode, population) in modes {
            for (scale_name, scale) in [("link", Scale::Link), ("response", Scale::Response)] {
                let want = numbers(&json["predict"][mode][scale_name], "predict");
                assert_eq!(want.len(), rows.n, "{name}: rows");
                let popts = PredictOptions {
                    population,
                    scale,
                    ..PredictOptions::default()
                };
                let mine = predict_parts(&at_lme4, spec, &design, &popts)
                    .unwrap_or_else(|err| panic!("{name}: {err}"));
                let theirs =
                    predict(&model, &new, &popts).unwrap_or_else(|err| panic!("{name}: {err}"));
                let eta_opts = PredictOptions {
                    scale: Scale::Link,
                    ..popts.clone()
                };
                let eta_mine = predict_parts(&at_lme4, spec, &design, &eta_opts).unwrap();
                let eta_glmm = predict(&model, &new, &eta_opts).unwrap();
                let (ranef_lme4, ranef_glmm, ranef_bumped): (&[f64], &[f64], &[f64]) = if population
                {
                    (&[], &[], &[])
                } else {
                    (&ranef_ref, &glmm.ranef, &ranef_stored)
                };
                for (i, &want) in want.iter().enumerate() {
                    let (sum, count, delta) = row_bounds(
                        spec,
                        &design,
                        levels,
                        (&glmm.beta, ranef_glmm),
                        (&beta_ref, ranef_lme4),
                        i,
                    );
                    let round = count as f64 * eps * sum;
                    let (_, _, stor) = row_bounds(
                        spec,
                        &design,
                        levels,
                        (&beta_stored, ranef_bumped),
                        (&beta_ref, ranef_lme4),
                        i,
                    );
                    let slope = match scale {
                        Scale::Link => 1.0,
                        _ => mu_eta(family, eta_mine.fit[i])
                            .abs()
                            .max(mu_eta(family, eta_glmm.fit[i]).abs()),
                    };
                    let what = format!("{mode} {scale_name} row {i}");
                    let band = slope * (round + stor) + half_digit(want);
                    let gap = (mine.fit[i] - want).abs();
                    worst.record(name, "at_lme4", gap / band, what.clone());
                    if !in_band(gap, band) {
                        hard.push(format!(
                            "{name} at lme4's parameters {what}: {} against lme4 {} (gap {gap:.3e} > band {band:.3e})",
                            mine.fit[i], want
                        ));
                    }
                    let band = slope * (round + delta + stor) + half_digit(want);
                    let gap = (theirs.fit[i] - want).abs();
                    worst.record(name, "glmm_predict", gap / band, what.clone());
                    if !in_band(gap, band) {
                        hard.push(format!(
                            "{name} end to end {what}: {} against lme4 {} (gap {gap:.3e} > band {band:.3e})",
                            theirs.fit[i], want
                        ));
                    }
                }
                // The rows lme4 predicts with a zero effect carry the notes.
                let noted: BTreeSet<u32> = theirs
                    .notes
                    .iter()
                    .flat_map(|n| match n {
                        Note::NewGroupingLevels { rows, .. } => rows.clone(),
                        other => panic!("{name}: unexpected note {other:?}"),
                    })
                    .collect();
                let unseen: BTreeSet<u32> = numbers(&json["unseen_rows"], "unseen_rows")
                    .iter()
                    .map(|&r| r as u32)
                    .collect();
                if population {
                    assert!(theirs.notes.is_empty(), "{name}: a note in population mode");
                } else {
                    assert_eq!(noted, unseen, "{name} {scale_name}: the noted rows");
                }
            }
        }
        let strict = PredictOptions {
            allow_new_levels: false,
            ..PredictOptions::default()
        };
        let refused = json["default_errors"].as_array().expect("default_errors");
        for r in refused {
            let row = r["row"].as_u64().expect("row") as usize;
            match predict(&model, &rows_of(&new, &[row]), &strict) {
                Err(Error::Fit(FitError::Ids(_))) => {}
                other => panic!(
                    "{name} row {row}: expected Error::Fit(FitError::Ids(..)) as lme4's {}, got {:?}",
                    r["error"],
                    other.map(|p| p.fit)
                ),
            }
        }

        if mixed {
            // lme4's `se.fit` reads the RX covariance of the fixed effects; a
            // refit under `WaldSe::Rx` checks the formula itself against it.
            let rx = (name == RX_ENTRY).then(|| {
                let rx_opts = FitOptions {
                    wald_se: WaldSe::Rx,
                    ..opts.clone()
                };
                fit(formula, &train, family, &rx_opts).unwrap_or_else(|err| panic!("{name}: {err}"))
            });
            for (mode, population) in [("population", true), ("conditional", false)] {
                for (scale_name, scale) in [("link", Scale::Link), ("response", Scale::Response)] {
                    let popts = PredictOptions {
                        population,
                        scale,
                        se: true,
                        ..PredictOptions::default()
                    };
                    let got =
                        predict(&model, &new, &popts).unwrap_or_else(|err| panic!("{name}: {err}"));
                    let lme4 = numbers(&json["se_fit_lme4"][mode][scale_name]["se_fit"], "lme4");
                    let tmb = numbers(
                        &json["se_fit_glmmtmb"][mode][scale_name]["se_fit"],
                        "glmmTMB",
                    );
                    for i in 0..rows.n {
                        driver.check(
                            &mut worst,
                            (name, "se_fit", &format!("{mode} {scale_name} se row {i}")),
                            got.se[i],
                            &[
                                ("lme4", lme4[i], SE_REL * lme4[i].abs()),
                                ("glmmTMB", tmb[i], SE_REL * tmb[i].abs()),
                            ],
                        );
                    }
                    let Some(rx) = &rx else { continue };
                    let got =
                        predict(rx, &new, &popts).unwrap_or_else(|err| panic!("{name}: {err}"));
                    for (i, (&se, &want)) in got.se.iter().zip(&lme4).enumerate() {
                        let what = format!("{mode} {scale_name} se row {i}");
                        let band = SE_REL * want.abs();
                        let gap = (se - want).abs();
                        worst.record(name, "se_fit_rx", gap / band, what.clone());
                        if !in_band(gap, band) {
                            hard.push(format!(
                                "{name} RX refit {what}: {se} against lme4 {want} (gap {gap:.3e} > band {band:.3e})"
                            ));
                        }
                    }
                }
            }
            let blocks = label_ranef(glmm, &lo.re_groups).expect("labelled ranef");
            for c in json["condvar"].as_array().expect("condvar") {
                let (g, reversed) = grouping_of(lo, c["grouping"].as_str().expect("grouping"));
                let block = &blocks[g];
                let q = block.terms.len();
                let v = q * (q + 1) / 2;
                assert_eq!(
                    block.condvar.len(),
                    block.levels.len() * v,
                    "{name}: condvar"
                );
                let terms = strings(&c["terms"], "condvar terms");
                let at: Vec<usize> = block
                    .terms
                    .iter()
                    .map(|t| terms.iter().position(|n| n == t).expect("term"))
                    .collect();
                let levels = strings(&c["levels"], "condvar levels");
                let postvar = &c["postvar"];
                let pv = |r: usize, s: usize, l: usize| {
                    postvar[r][s][l].as_f64().expect("a postVar entry")
                };
                for (l, lme4_label) in levels.iter().enumerate() {
                    let lbl = label(lme4_label, reversed);
                    let k = block
                        .levels
                        .iter()
                        .position(|x| x == &lbl)
                        .unwrap_or_else(|| panic!("{name}: no level {lbl}"));
                    let vech = &block.condvar[k * v..(k + 1) * v];
                    let mut t = 0;
                    for col in 0..q {
                        for row in col..q {
                            let want = pv(at[row], at[col], l);
                            let band =
                                SE_REL * (pv(at[row], at[row], l) * pv(at[col], at[col], l)).sqrt();
                            let what = format!(
                                "{} {lbl} postVar[{}, {}]",
                                block.group, block.terms[row], block.terms[col]
                            );
                            driver.check(
                                &mut worst,
                                (name, "condvar", &what),
                                vech[t],
                                &[("lme4", want, band)],
                            );
                            t += 1;
                        }
                    }
                }
                for sd in c["condsd"].as_array().expect("condsd") {
                    let lbl = label(sd["level"].as_str().expect("level"), reversed);
                    let term = sd["term"].as_str().expect("term");
                    let want = sd["condsd"].as_f64().expect("condsd");
                    let k = block.levels.iter().position(|x| x == &lbl).expect("level");
                    let r = block.terms.iter().position(|x| x == term).expect("term");
                    // The column-major vech offset of the diagonal `(r, r)`.
                    let diag = (0..r).map(|c| q - c).sum::<usize>();
                    let got = block.condvar[k * v + diag].sqrt();
                    driver.check(
                        &mut worst,
                        (
                            name,
                            "condvar",
                            &format!("{} {lbl} condsd {term}", block.group),
                        ),
                        got,
                        &[("lme4", want, SE_REL * want.abs())],
                    );
                }
            }
        }
        driver.compared.insert(name.to_string());
    }
    for ((name, quantity), (ratio, what)) in &worst.0 {
        eprintln!("PREDICT {name} {quantity}: worst gap {ratio:.3e} of the band ({what})");
    }
    assert_eq!(
        driver.compared.len(),
        8,
        "the eight predict references compared"
    );
    driver.finish(&hard);
}
