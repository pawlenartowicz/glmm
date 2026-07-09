//! glmm side of the cross-language parity harness — fits the parity datasets with
//! the local `glmm` crate and writes `results/glmm/<ds>.json` in the common
//! schema (parity/README.md), to be checked against the frozen lme4 / MixedModels
//! references by `compare.R`. Committed alongside the other engines' results.
//!
//! The design matrix, `ModelSpec` and per-row `GroupIds` are built by the `d3`
//! formula frontend (`glmm_formula::lower`, a dev-dependency here) from an R-style
//! formula string + a columnar `Table`, not hand-assembled — the same path SDOC /
//! MCPower will drive. This doubles as the frontend's end-to-end oracle: the
//! numbers it feeds `fit_cold` must reproduce the frozen references.
//!
//! Manifest-driven (mirrors `fit.R`/`fit.jl`'s `fit_one` loop): one `fit_one`
//! function reads each `manifest.json` entry and fits it generically, rather than
//! nine hand-written per-dataset functions. `jl_formula` (guaranteed `cbind`-free,
//! parseable the same way for every rung) is the lowering source, not `r_formula`.
//! The reference lme4 JSON supplies the RE-grouping name order the output is
//! reindexed to (`compare.R` aligns `varcomp` positionally, not by name) — a
//! data-driven lookup at run time, not a re-derivation of lme4's own convention.
//!
//! Run via `run.sh` (ENGINES has "rust") or
//! `cargo run --release --example parity_fit`. Paths are anchored at
//! `CARGO_MANIFEST_DIR` so the cwd does not matter.

use std::time::Instant;

use glmm::{fit_cold, BinomialLink, Family, Fit, FitOptions, PoissonLink, WaldSe};
use glmm_formula::{lower, Column, Lowered, ReGroupInfo, Table};
use serde_json::{json, Value};

const DIR: &str = env!("CARGO_MANIFEST_DIR");
const VERSION: &str = env!("CARGO_PKG_VERSION");
// Timing loop: first (cold) pass discarded, MEDIAN of the rest reported. Median, not
// min, is the user's chosen estimator for this harness. NOT a locked-machine
// benchmark — the timing field is indicative only until the box is stabilized/locked
// (see README "Timing"); the user owns CPU-clock locking, this harness only records.
// 10 runs (was 100): the corpus now holds multi-second GLMM fits where 100 repeats
// cost an hour per engine for no extra precision; each JSON records its own n_runs,
// so files timed under the old convention stay self-describing.
const N_RUNS: usize = 10;

fn main() {
    std::fs::create_dir_all(format!("{DIR}/parity/results/glmm")).expect("mk glmm");
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/parity/manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");
    // PARITY_ONLY=<name>[,<name>...]: fit only the named datasets (mirrors
    // fit.R / fit.jl) — reruns a single rung without repaying the full-corpus
    // timing cost (grouseticks alone is ~200 multi-second fits).
    let only = std::env::var("PARITY_ONLY").unwrap_or_default();
    let want = |ds: &str| only.is_empty() || only.split(',').any(|s| s == ds);
    for spec in manifest["datasets"].as_array().expect("manifest.datasets") {
        let ds = spec["name"].as_str().expect("dataset entry missing name");
        if want(ds) {
            fit_one(spec);
        }
    }
}

/// Fit one manifest entry end-to-end (load → lower → fit(+SE split) → time →
/// reindex varcomp to the reference's grouping order → write). Mirrors `fit.R`'s
/// `fit_one` / `fit.jl`'s `fit_one`.
fn fit_one(spec: &Value) {
    let ds = spec["name"].as_str().expect("manifest entry missing name");
    let rung = spec["rung"].as_u64().expect("manifest entry missing rung");
    let family_str = spec["family"]
        .as_str()
        .expect("manifest entry missing family");
    let gaussian = family_str == "gaussian";
    let factors: Vec<String> = spec["factors"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
        .unwrap_or_default();

    let (header, rows) = read_csv(ds);

    // jl_formula, not r_formula: guaranteed cbind-free for every rung (design doc),
    // so it is already a safe generic lowering source. Strip the literal
    // "@formula(...)" wrapper to get a plain formula string.
    let jl = spec["jl_formula"]
        .as_str()
        .expect("manifest entry missing jl_formula");
    let formula_str = jl
        .strip_prefix("@formula(")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or_else(|| panic!("jl_formula not in @formula(...) shape: {jl}"))
        .to_string();
    // Julia's own crossed-interaction grouping operator is `&` (e.g. cake's
    // `(1 | recipe & replicate)`); this crate's `glmm_formula` parser uses `:`
    // for the same grouping (`(1|A:B)`). `&` occurs nowhere else in the manifest
    // (checked: only cake's RE term), so a global replace is safe and generic.
    let formula_str = formula_str.replace(" & ", ":");
    // Julia's @formula requires an explicit `1` intercept term; this crate's
    // parser (mirroring MCPower's) treats the intercept as always-implicit and
    // has no term for a literal `1`, so strip it — every jl_formula in the
    // manifest writes it as the fixed side's leading term, `"<dep> ~ 1 + ..."`.
    let formula_str = formula_str.replacen(" ~ 1 + ", " ~ ", 1);

    let family = match family_str {
        "gaussian" => Family::Gaussian,
        "binomial" => Family::Binomial {
            link: BinomialLink::Logit,
        },
        "poisson" => Family::Poisson {
            link: PoissonLink::Log,
        },
        other => panic!("unsupported family: {other}"),
    };

    let mut lo = lower_dataset(spec, &header, &rows, &factors, &formula_str, family);
    let timing_batch = spec["timing_batch"].as_u64().unwrap_or(1) as usize;

    // Reference grouping order (compare.R aligns varcomp positionally, not by
    // name) — read off the already-frozen lme4 result rather than re-deriving
    // lme4's own convention.
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/parity/results/lme4/{ds}.json"))
            .expect("read lme4 reference"),
    )
    .expect("parse lme4 reference");
    let ref_order: Vec<String> = reference["estimates"]["varcomp"]
        .as_array()
        .expect("reference estimates.varcomp array")
        .iter()
        .map(|e| {
            e["group"]
                .as_str()
                .expect("reference varcomp group name")
                .to_string()
        })
        .collect();

    let (converged, estimates, timing) = if gaussian {
        let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let t = json!({
            "fit_seconds_median": median_secs(timing_batch, || {
                let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
            }),
            "n_runs": N_RUNS, "warmup_discarded": 1, "fits_per_sample": timing_batch,
        });
        let est = json!({
            "beta": nums(&f.beta),
            "se": nums(&f.se),
            "varcomp": varcomp(&f, &lo.re_groups, &ref_order, false),
        });
        (f.converged, est, t)
    } else {
        // GLMM SE has two genuinely different variants (Laplace) — emit both so
        // compare.R checks like to like: se_hessian (keeps θ–β coupling, glmm
        // default) vs se_rx (conditional on θ̂). β/τ is wald_se-independent.
        let o_r = FitOptions {
            target_indices: lo.opts.target_indices.clone(),
            wald_se: WaldSe::Rx,
            weights: lo.opts.weights.clone(),
            ..FitOptions::default()
        };
        let fh = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let fr = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &o_r);
        // Split timing by SE method — the FD-Hessian is the main time consumer,
        // Rx is one closed-form Schur solve. Same PIRLS fit underlies both.
        let t = json!({
            "fit_seconds_median_rx": median_secs(timing_batch, || {
                let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &o_r);
            }),
            "fit_seconds_median_hessian": median_secs(timing_batch, || {
                let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
            }),
            "n_runs": N_RUNS, "warmup_discarded": 1, "fits_per_sample": timing_batch,
        });
        let est = json!({
            "beta": nums(&fh.beta),
            "se_hessian": nums(&fh.se),
            "se_rx": nums(&fr.se),
            // stddev_se from the Hessian fit's θ block (fh = WaldSe::Hessian).
            "varcomp": varcomp(&fh, &lo.re_groups, &ref_order, true),
        });
        (fh.converged && fr.converged, est, t)
    };

    let res = json!({
        "dataset": ds, "engine": "glmm", "engine_version": format!("{VERSION}-local"),
        "family": family_str,
        "reml": if gaussian { json!(spec["reml"].as_bool().unwrap_or(false)) } else { Value::Null },
        "rung": rung,
        "converged": converged, "singular": false,
        "coef_names": std::mem::take(&mut lo.col_names),
        "estimates": estimates,
        "timing": timing,
    });
    write_result(ds, res);
}

/// Build the lowered fit inputs for one dataset. Handles the one genuinely
/// data-shape-dependent branch in the corpus: an aggregated-binomial rung
/// (manifest `weights`) synthesizes `prop = incidence/weights_col` (mirrors
/// fit.jl:44-46) so jl_formula's `prop ~ ...` response resolves, then either:
/// - passes `Some(weights)` into `FitOptions::weights` when the RE structure
///   routes to the sparse binomial GLMM path (the only path that honors prior
///   weights — `src/fit.rs`'s own boundary assert), one row per aggregate
///   observation (sim_sparse_binomial: 7 crossed extras ⇒ sparse); or
/// - falls back to the historical per-trial Bernoulli expansion when it does not
///   (cbpp: single grouping ⇒ dense NoZ, where weights are not honored) — same
///   argmin as the aggregated-weights objective, so the numbers agree.
/// Solver choice is read off the lowered RE shape (grouping/term counts vs the
/// engine's own envelope caps), not the dataset name, so a future weighted rung
/// routes itself correctly either way.
fn lower_dataset(
    spec: &Value,
    header: &[String],
    rows: &[Vec<String>],
    factors: &[String],
    formula_str: &str,
    family: Family,
) -> Lowered {
    let Some(w_name) = spec["weights"].as_str() else {
        let table = build_table(header, rows, factors);
        return lower(formula_str, &table, family).unwrap_or_else(|e| panic!("lower: {e}"));
    };

    let w_idx = header
        .iter()
        .position(|h| h == w_name)
        .expect("weights column in header");
    let inc_idx = header
        .iter()
        .position(|h| h == "incidence")
        .expect("incidence column in header");
    let sizes: Vec<f64> = rows.iter().map(|r| r[w_idx].parse().unwrap()).collect();
    let incid: Vec<f64> = rows.iter().map(|r| r[inc_idx].parse().unwrap()).collect();
    let prop: Vec<f64> = incid.iter().zip(&sizes).map(|(i, s)| i / s).collect();

    let mut table = build_table(header, rows, factors);
    table.columns.push(("prop".into(), Column::Numeric(prop)));
    let lo_agg = lower(formula_str, &table, family).unwrap_or_else(|e| panic!("lower: {e}"));

    // Mirrors src/fit.rs's classify_design for the binomial (non-Gaussian) case,
    // computed from the lowered RE shape alone: over-envelope OR any
    // slope-carrying extra grouping (q_g ≥ 2) routes Sparse — change together.
    let extras = lo_agg.re_groups.len().saturating_sub(1);
    let q_p = lo_agg.re_groups.first().map_or(0, |g| g.terms.len());
    let over_extra = lo_agg.re_groups[1..]
        .iter()
        .any(|g| g.terms.len() > glmm::consts::MAX_EXTRA_Q);
    let slope_extras = lo_agg.re_groups[1..].iter().any(|g| g.terms.len() > 1);
    let sparse = extras > glmm::consts::MAX_EXTRA_GROUPINGS
        || q_p > glmm::consts::MAX_PRIMARY_Q
        || over_extra
        || slope_extras;

    if sparse {
        let mut lo = lo_agg;
        lo.opts.weights = Some(sizes);
        lo
    } else {
        let erows = expand_bernoulli(rows, inc_idx, w_idx);
        let mut etable = build_table(header, &erows, factors);
        let prop_vals = match &etable
            .columns
            .iter()
            .find(|(n, _)| n == "incidence")
            .unwrap()
            .1
        {
            Column::Numeric(v) => v.clone(),
            Column::Factor { .. } => unreachable!("incidence is numeric"),
        };
        etable
            .columns
            .push(("prop".into(), Column::Numeric(prop_vals)));
        lower(formula_str, &etable, family).unwrap_or_else(|e| panic!("lower: {e}"))
    }
}

/// Expand aggregated (incidence, size) rows into `size` per-trial Bernoulli rows
/// (the kernel is Bernoulli on the dense path). Every other column is duplicated
/// as-is; `incidence` becomes the 0/1 trial outcome.
fn expand_bernoulli(rows: &[Vec<String>], inc_idx: usize, size_idx: usize) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    for r in rows {
        let incidence: u32 = r[inc_idx].parse().unwrap();
        let size: u32 = r[size_idx].parse().unwrap();
        for k in 0..size {
            let mut nr = r.clone();
            nr[inc_idx] = if k < incidence {
                "1".into()
            } else {
                "0".into()
            };
            out.push(nr);
        }
    }
    out
}

/// A `Table` built by column NAME: manifest `factors` become `Column::Factor`,
/// everything else `Column::Numeric` — except a column that fails to parse as
/// `f64` anywhere (e.g. Pastes' `cask`, a categorical helper column the parity
/// corpus carries but no jl_formula references) falls back to `Column::Factor`
/// rather than panicking, since it may be present in the CSV without being
/// referenced by the formula at all.
fn build_table(header: &[String], rows: &[Vec<String>], factors: &[String]) -> Table {
    let n = rows.len();
    let columns = header
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let is_factor = factors.iter().any(|f| f == name)
                || rows.iter().any(|r| r[j].parse::<f64>().is_err());
            let col = if is_factor {
                Column::Factor {
                    labels: rows.iter().map(|r| r[j].clone()).collect(),
                }
            } else {
                Column::Numeric(rows.iter().map(|r| r[j].parse().unwrap()).collect())
            };
            // Mirrors fit.jl's read_dataset: R-origin CSV headers can carry dots
            // (Arabidopsis's `total.fruits`) that jl_formula sanitizes to
            // underscores because Julia's @formula reader can't parse a dot as
            // part of an identifier. Rename here so the Table's column names
            // match jl_formula's (already-underscored) reference.
            (name.replace('.', "_"), col)
        })
        .collect();
    Table { columns, n }
}

/// Variance components in the common schema, one entry per grouping factor, from
/// `Fit::stddev_corr` (arbitrary q). `stddev_se` (GLMM only) is attached from the
/// Hessian fit's θ block, laid out per θ coordinate: cumulative vech length across
/// groupings in DECLARATION order (matches `Fit::stddev_se`'s own layout), gated
/// to scalar (q=1) groupings — the only shape the θ==stddev identity holds for,
/// same as lme4's own gating. Reindexed to `ref_order` (compare.R aligns
/// positionally, not by name) — entries MUST be given in the reference's order.
fn varcomp(f: &Fit, re_groups: &[ReGroupInfo], ref_order: &[String], include_se: bool) -> Value {
    let mut theta_offset = 0usize;
    let natural: Vec<Value> = re_groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let (stddev, corr) = f.stddev_corr(i);
            let q = g.terms.len();
            let mut entry = json!({
                "group": g.name,
                "terms": g.terms,
                "stddev": nums(&stddev),
                "corr": Value::Array(corr.iter().map(|row| nums(row)).collect()),
            });
            if include_se && q == 1 {
                entry["stddev_se"] = nums(&[f.stddev_se[theta_offset]]);
            }
            theta_offset += q * (q + 1) / 2;
            entry
        })
        .collect();
    Value::Array(
        ref_order
            .iter()
            .map(|name| {
                let idx = re_groups
                    .iter()
                    .position(|g| group_names_match(&g.name, name))
                    .unwrap_or_else(|| {
                        panic!("reference group {name:?} not found in fit's re_groups")
                    });
                natural[idx].clone()
            })
            .collect(),
    )
}

/// Compares a `glmm_formula` grouping name against the lme4 reference's grouping
/// name, order-invariant on `:`-joined components: lme4 names a nested inner
/// group `child:parent` (e.g. `"Variety:Block"` for `(1|Block/Variety)`) while
/// `glmm_formula` names the same grouping `parent:child` (`"Block:Variety"`) — a
/// pure display-convention difference, not a different grouping. Comparing the
/// `:`-split component sets (rather than the joined string) matches either order,
/// and is a no-op for non-composite names (single-component groupings compare
/// equal only if identical, as before).
fn group_names_match(a: &str, b: &str) -> bool {
    let mut sa: Vec<&str> = a.split(':').collect();
    let mut sb: Vec<&str> = b.split(':').collect();
    sa.sort_unstable();
    sb.sort_unstable();
    sa == sb
}

// ── shared helpers ────────────────────────────────────────────────────────────

fn read_csv(ds: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let raw = std::fs::read_to_string(format!("{DIR}/parity/data/{ds}.csv")).expect("read csv");
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().unwrap().split(',').map(unquote).collect();
    let rows = lines.map(|l| l.split(',').map(unquote).collect()).collect();
    (header, rows)
}
fn unquote(s: &str) -> String {
    s.trim().trim_matches('"').to_string()
}

/// NaN/Inf → JSON null (serde_json cannot serialize non-finite floats, and a
/// non-converged fit leaves NaN-filled estimates) so an unconverged run still
/// writes valid JSON the comparators read as "missing", not a crash.
fn num(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}
fn nums(xs: &[f64]) -> Value {
    Value::Array(xs.iter().map(|&x| num(x)).collect())
}

/// Median seconds over N_RUNS samples, warm-up (first) discarded. Each sample times
/// `batch` fits so sub-resolution fits stay above the timer floor (mirrors the manifest
/// `timing_batch` the R/Julia oracles read) — the returned median is for `batch` fits;
/// divide by `batch` for the per-fit estimate. GLMM rungs call this once per SE method
/// (Rx vs Hessian) because the FD-Hessian is the dominant cost.
fn median_secs(batch: usize, mut f: impl FnMut()) -> f64 {
    let mut t = Vec::with_capacity(N_RUNS);
    for _ in 0..N_RUNS {
        let t0 = Instant::now();
        for _ in 0..batch {
            f();
        }
        t.push(t0.elapsed().as_secs_f64());
    }
    median(&t[1..])
}
fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}

fn write_result(ds: &str, res: Value) {
    let out = format!("{DIR}/parity/results/glmm/{ds}.json");
    std::fs::write(
        &out,
        format!("{}\n", serde_json::to_string_pretty(&res).unwrap()),
    )
    .expect("write result");
    let conv = res["converged"].as_bool().unwrap_or(false);
    // GLMM rungs store the split rx/hessian medians, not a single median — show
    // the Rx one on the console (mirrors fit.jl's t_disp fallback).
    let t = res["timing"]["fit_seconds_median"]
        .as_f64()
        .or_else(|| res["timing"]["fit_seconds_median_rx"].as_f64())
        .unwrap_or(f64::NAN);
    println!(
        "glmm  {ds:<12}  rung {}  converged={conv}  fit_median={t:.4}s",
        res["rung"]
    );
}
