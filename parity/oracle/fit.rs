//! glmm side of the cross-language parity harness — fits the parity datasets with
//! the local `glmm` crate and writes `results/glmm_{empirical,simulated}/<ds>.json`
//! in the common schema (parity/README.md), to be checked against the frozen lme4 /
//! MixedModels references by `compare.R`. Committed alongside the other engines' results.
//!
//! The design matrix, `ModelSpec` and per-row `GroupIds` are built by the `d3`
//! formula frontend (`glmm::formula::lower`, the `formula` feature) from an R-style
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
//! `cargo run --release -p parity --example parity_fit`. Paths are anchored at
//! `CARGO_MANIFEST_DIR` (the `parity/` crate dir) so the cwd does not matter.

use std::time::Instant;

use glmm::formula::{lower, ReGroupInfo};
use glmm::{
    fit_cold, BinomialLink, Family, Fit, FitOptions, GammaLink, NegBinomialLink, PoissonLink,
    WaldSe,
};
use serde_json::{json, Value};

#[path = "harness_common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Suite directory (manifest + data + results root). `PARITY_SUITE_DIR`
/// overrides at RUN time (a sub-suite's run.sh, e.g. `parity/weights/`, sets
/// it); unset = the main `parity/` dir, byte-identical to the pre-override
/// behavior. Read per call site rather than cached: the compile-time
/// `CARGO_MANIFEST_DIR` anchor (the `parity/` crate dir) stays the fallback.
fn suite_dir() -> String {
    std::env::var("PARITY_SUITE_DIR").unwrap_or_else(|_| DIR.to_string())
}
// Timing loop: first (cold) pass discarded, MEDIAN of the rest reported. Median, not
// min, is the user's chosen estimator for this harness. NOT a locked-machine
// benchmark — the timing field is indicative only until the box is stabilized/locked
// (see README "Timing").
// 10 runs (was 100): the corpus now holds multi-second GLMM fits where 100 repeats
// cost an hour per engine for no extra precision; each JSON records its own n_runs,
// so files timed under the old convention stay self-describing.
//
// TWO timings per fit, both recorded: `fit_seconds_median*` times `fit_cold` ALONE
// (lowering hoisted out) — the solver-isolation number the per-eval / solve-gap
// analyses need; `fit_seconds_median*_full` times `lower + fit_cold`, the
// construction-inclusive span lme4 / MixedModels / the Python port all measure, so
// the cross-engine speedups and the port's `py_gap` compare same-to-same
// (summarize_timing.R reads the `_full` fields for glmm).
const N_RUNS: usize = 10;

fn main() {
    let suite = suite_dir();
    std::fs::create_dir_all(format!("{suite}/results/glmm_empirical")).expect("mk glmm_empirical");
    std::fs::create_dir_all(format!("{suite}/results/glmm_simulated")).expect("mk glmm_simulated");
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{suite}/manifest.json")).expect("read manifest"),
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
    let source = if spec["source"].as_str() == Some("sim") {
        "simulated"
    } else {
        "empirical"
    };

    let suite = suite_dir();
    // `data` field: CSV to read when it differs from the rung name (mirrors
    // fit.R/fit.jl) — a re-linked rung (cbpp_probit) reuses the committed
    // dataset byte-for-byte instead of duplicating it.
    let data_name = spec["data"].as_str().unwrap_or(ds);
    let (header, rows) = read_csv_path(&format!("{suite}/data_{source}/{data_name}.csv"));

    // jl_formula, not r_formula: guaranteed cbind-free for every rung (design doc),
    // so it is already a safe generic lowering source. Strip the literal
    // "@formula(...)" wrapper to get a plain formula string. Rungs WITHOUT a
    // jl_formula (weights suite fixed-only / R-only rungs -- the field's absence
    // is fit.jl's skip signal) fall back to r_formula; an aggregated-binomial
    // r_formula's `cbind(...)` response is rewritten to the `prop` column
    // lower_dataset_generic synthesizes (the same lowering fit.jl's manifest
    // entries spell out by hand).
    let formula_str = match spec["jl_formula"].as_str() {
        Some(jl) => jl
            .strip_prefix("@formula(")
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or_else(|| panic!("jl_formula not in @formula(...) shape: {jl}"))
            .to_string(),
        None => {
            let r = spec["r_formula"]
                .as_str()
                .expect("manifest entry missing both jl_formula and r_formula");
            match r.split_once('~') {
                Some((resp, rhs)) if resp.trim_start().starts_with("cbind(") => {
                    format!("prop ~{rhs}")
                }
                _ => r.to_string(),
            }
        }
    };
    // Julia's own crossed-interaction grouping operator is `&` (e.g. cake's
    // `(1 | recipe & replicate)`); the formula frontend's parser uses `:`
    // for the same grouping (`(1|A:B)`). `&` occurs nowhere else in the manifest
    // (checked: only cake's RE term), so a global replace is safe and generic.
    let formula_str = formula_str.replace(" & ", ":");
    // Julia's @formula requires an explicit `1` intercept term; this crate's
    // parser (mirroring MCPower's) treats the intercept as always-implicit and
    // has no term for a literal `1`, so strip it — every jl_formula in the
    // manifest writes it as the fixed side's leading term, `"<dep> ~ 1 + ..."`.
    let formula_str = formula_str.replacen(" ~ 1 + ", " ~ ", 1);

    // `link` field: non-canonical link override (cbpp_probit). Absent = the
    // family's canonical link, the pre-existing behavior for every other rung.
    let link_str = spec["link"].as_str();
    let family = match family_str {
        "gaussian" => Family::Gaussian,
        "binomial" => Family::Binomial {
            link: match link_str {
                None | Some("logit") => BinomialLink::Logit,
                Some("probit") => BinomialLink::Probit,
                Some(other) => panic!("unsupported binomial link: {other}"),
            },
        },
        "poisson" => Family::Poisson {
            link: PoissonLink::Log,
        },
        "gamma" => Family::Gamma {
            link: match link_str {
                None | Some("log") => GammaLink::Log,
                Some("inverse") => GammaLink::Inverse,
                Some(other) => panic!("unsupported gamma link: {other}"),
            },
        },
        "negbin" => Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        other => panic!("unsupported family: {other}"),
    };

    // `table` is kept (not just `lo`) so the construction-inclusive timing below
    // can re-run `lower(&table)` in a loop — the Rust analogue of what lme4 /
    // MixedModels / the Python port time (formula+data → model → fit), so all
    // four engines compare same-to-same. See the `_full` timing fields.
    let (mut lo, table) =
        lower_dataset_generic(spec, &header, &rows, &factors, &formula_str, family);
    // `weights_col`: plain per-row prior weights read off the named CSV column
    // (every weights-suite rung except the aggregated-binomial ones, which use
    // the `weights` field lower_dataset_generic already routes -- the two are
    // mutually exclusive per rung by design).
    if let Some(wc) = spec["weights_col"].as_str() {
        assert!(
            spec["weights"].is_null(),
            "{ds}: weights_col and weights are mutually exclusive"
        );
        let w_idx = header
            .iter()
            .position(|h| h == wc)
            .unwrap_or_else(|| panic!("{ds}: weights_col {wc:?} not in CSV header"));
        lo.opts.weights = Some(rows.iter().map(|r| r[w_idx].parse().unwrap()).collect());
    }
    // `offset` field: per-row known additive term on the linear-predictor scale
    // (R's `offset=`) -- a named CSV column, mirroring `weights_col` above.
    if let Some(oc) = spec["offset"].as_str() {
        let o_idx = header
            .iter()
            .position(|h| h == oc)
            .unwrap_or_else(|| panic!("{ds}: offset {oc:?} not in CSV header"));
        lo.opts.offset = Some(rows.iter().map(|r| r[o_idx].parse().unwrap()).collect());
    }
    let timing_batch = spec["timing_batch"].as_u64().unwrap_or(1) as usize;

    // Reference grouping order (compare.R aligns varcomp positionally, not by
    // name) — read off the already-frozen lme4 result rather than re-deriving
    // lme4's own convention.
    let reference: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{suite}/results/lme4_{source}/{ds}.json"))
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

    // Unit-weights identity gate (weights suite U1): the `weighted` fast-path
    // split must be invisible at w ≡ 1 -- fit once with the (all-ones) weights
    // and once with `weights: None`, and require β/SE/θ-stddev to agree to
    // 1e-12 relative BEFORE the result JSON is written. Failure is loud at fit
    // time, not a tolerance row in compare.R.
    if spec["gate"].as_str() == Some("unit_identity") {
        let o_u = FitOptions {
            target_indices: lo.opts.target_indices.clone(),
            weights: None,
            ..FitOptions::default()
        };
        let fw = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let fu = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &o_u);
        let check = |what: &str, a: &[f64], b: &[f64]| {
            assert_eq!(a.len(), b.len(), "{ds}: unit-identity {what} length");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let rel = (x - y).abs() / x.abs().max(y.abs()).max(1e-12);
                assert!(
                    rel < 1e-12,
                    "{ds}: unit-identity gate FAILED on {what}[{i}]: w≡1 gives {x}, \
                     weights:None gives {y} (rel {rel:.3e} ≥ 1e-12)"
                );
            }
        };
        check("beta", &fw.beta, &fu.beta);
        check("se", &fw.se, &fu.se);
        for i in 0..lo.re_groups.len() {
            check("stddev", &fw.stddev_corr(i).0, &fu.stddev_corr(i).0);
        }
        println!("glmm  {ds:<12}  unit-identity gate ok (w≡1 == weights:None to 1e-12)");
    }

    // Construction-inclusive timing: median of `lower(&table) + fit_cold`, the
    // span lme4 (`lmer(formula, df)`), MixedModels (`fit(MixedModel, f, df)`) and
    // the Python port (`glmm.fit(data, formula)`) all measure — model matrices
    // are rebuilt from the formula on every call there, so timing `fit_cold`
    // alone (the `_median` fields, retained for the solver-isolation analyses)
    // is the ONLY engine that excludes it. `table` is pre-built (the typed-column
    // DataFrame analogue), so this excludes CSV string-parsing, matching the
    // reference engines' pre-typed `df`. `opts` is captured, not re-derived from
    // the fresh lowering: target_indices are column positions the identical table
    // reproduces, and weights are per-row — both stay valid across re-lowers.
    let time_full = |opts: &FitOptions| -> f64 {
        median_secs(timing_batch, || {
            let l = lower(&formula_str, &table, family)
                .unwrap_or_else(|e| panic!("re-lower {ds}: {e}"));
            let _ = fit_cold(&l.x, &l.y, l.n, l.p, &l.model, &l.ids, opts);
        })
    };

    let fixed_only = lo.re_groups.is_empty();
    let (converged, singular, estimates, timing, n_eval, deviance) = if gaussian {
        let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let t = json!({
            "fit_seconds_median": median_secs(timing_batch, || {
                let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
            }),
            "fit_seconds_median_full": time_full(&lo.opts),
            "n_runs": N_RUNS, "warmup_discarded": 1, "fits_per_sample": timing_batch,
        });
        let est = json!({
            "beta": nums(&f.beta),
            "se": nums(&f.se),
            "loglik": num(f.loglik),
            "df": f.df,
            "varcomp": varcomp(&f, &lo.re_groups, &ref_order, false),
        });
        (f.converged, f.singular, est, t, f.n_eval, f.deviance)
    } else if fixed_only {
        // Fixed-only GLM (weights suite): no θ, so the Rx-vs-Hessian method
        // split is moot — one fit, one SE, emitted as `se_rx` to line up with
        // the single SE fit.R's `glm`/`glm.nb` writes for these rungs.
        let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let t = json!({
            "fit_seconds_median": median_secs(timing_batch, || {
                let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
            }),
            "fit_seconds_median_full": time_full(&lo.opts),
            "n_runs": N_RUNS, "warmup_discarded": 1, "fits_per_sample": timing_batch,
        });
        let est = json!({
            "beta": nums(&f.beta),
            "se_rx": nums(&f.se),
            "loglik": num(f.loglik),
            "df": f.df,
            "varcomp": varcomp(&f, &lo.re_groups, &ref_order, false),
        });
        (f.converged, f.singular, est, t, f.n_eval, f.deviance)
    } else {
        // GLMM SE has two genuinely different variants (Laplace) — emit both so
        // compare.R checks like to like: se_hessian (keeps θ–β coupling, glmm
        // default) vs se_rx (conditional on θ̂). β/τ is wald_se-independent.
        let o_r = FitOptions {
            target_indices: lo.opts.target_indices.clone(),
            wald_se: WaldSe::Rx,
            weights: lo.opts.weights.clone(),
            offset: lo.opts.offset.clone(),
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
            "fit_seconds_median_rx_full": time_full(&o_r),
            "fit_seconds_median_hessian_full": time_full(&lo.opts),
            "n_runs": N_RUNS, "warmup_discarded": 1, "fits_per_sample": timing_batch,
        });
        let est = json!({
            "beta": nums(&fh.beta),
            "se_hessian": nums(&fh.se),
            "se_rx": nums(&fr.se),
            "loglik": num(fh.loglik),
            "df": fh.df,
            // stddev_se from the Hessian fit's θ block (fh = WaldSe::Hessian).
            "varcomp": varcomp(&fh, &lo.re_groups, &ref_order, true),
        });
        (
            fh.converged && fr.converged,
            // singular from the Hessian fit (fh) — same PIRLS fit as fr, so the
            // boundary decision is identical; fh is the one whose θ block the
            // estimates come from.
            fh.singular,
            est,
            t,
            fh.n_eval + fr.n_eval,
            fh.deviance,
        )
    };

    let res = json!({
        "dataset": ds, "engine": "glmm", "engine_version": format!("{VERSION}-local"),
        "family": family_str,
        "reml": if gaussian { json!(spec["reml"].as_bool().unwrap_or(false)) } else { Value::Null },
        "rung": rung,
        "converged": converged, "singular": singular,
        "optimizer": "bobyqa",
        "n_eval": n_eval,
        "deviance": num(deviance),
        "coef_names": std::mem::take(&mut lo.col_names),
        "estimates": estimates,
        "timing": timing,
    });
    write_result(ds, source, res);
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

/// Compares a formula-frontend grouping name against the lme4 reference's grouping
/// name, order-invariant on `:`-joined components: lme4 names a nested inner
/// group `child:parent` (e.g. `"Variety:Block"` for `(1|Block/Variety)`) while
/// the formula frontend names the same grouping `parent:child` (`"Block:Variety"`) — a
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
// read_csv_path/unquote/build_table/num/nums/lower_dataset_generic
// live in harness_common.rs (shared with grid_fit.rs) — change together.

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

fn write_result(ds: &str, source: &str, res: Value) {
    let out = format!("{}/results/glmm_{source}/{ds}.json", suite_dir());
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
