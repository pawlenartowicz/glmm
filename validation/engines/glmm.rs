//! glmm side of the cross-language validation suite — fits the validation datasets with
//! the local `glmm` crate and writes `results/glmm_{empirical,simulated}/<ds>.json`
//! in the common schema (validation/README.md), to be checked against the frozen lme4 /
//! MixedModels references by `compare.R`. Committed alongside the other engines' results.
//!
//! The design matrix, `ModelSpec` and per-row `GroupIds` are built by the `d3`
//! formula frontend (`glmm::formula::lower`, the `formula` feature) from an R-style
//! formula string + a columnar `Table`, not hand-assembled — the same path SDOC /
//! MCPower will drive. This doubles as the frontend's end-to-end oracle: the
//! numbers it feeds `fit_cold` must reproduce the frozen references.
//!
//! Manifest-driven (mirrors `lme4.R`/`mixedmodels.jl`'s `fit_one` loop): one `fit_one`
//! function reads each `manifest.json` entry and fits it generically, rather than
//! nine hand-written per-dataset functions. `jl_formula` (guaranteed `cbind`-free,
//! parseable the same way for every rung) is the lowering source, not `r_formula`.
//! The reference lme4 JSON supplies the RE-grouping name order the output is
//! reindexed to (`compare.R` aligns `varcomp` positionally, not by name) — a
//! data-driven lookup at run time, not a re-derivation of lme4's own convention.
//!
//! Run via `run.sh` (ENGINES has "rust") or
//! `cargo run --release -p validation --example validation_fit`. Paths are anchored at
//! `CARGO_MANIFEST_DIR` (the `validation/` crate dir) so the cwd does not matter.

use std::time::Instant;

use glmm::formula::{lower, ReGroupInfo};
use glmm::{fit_cold, Fit, FitOptions};
use serde_json::{json, Value};

#[path = "common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Suite directory (manifest + data + results root): the `validation/` crate dir.
fn suite_dir() -> String {
    DIR.to_string()
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
// Timing is OPT-IN, and its sample count lives in run.sh rather than here, so the
// five engines no longer carry mirrored N_RUNS constants to keep in step. compare.R
// reads no timing field at all, so the default gate pays only the one fit per SE
// method it already needed for the estimates, instead of N x (Rx, Hessian, and both
// `_full` variants) — ~35 seconds over the corpus rather than ~10 minutes.

/// Sample count for this run, or `None` when timing is off. Read once per dataset,
/// never per loop.
///
/// THE contract, mirrored in lme4.R / mixedmodels.jl / glmm_python.py / glmm_r.R
/// `timing_runs` — five languages that cannot share code, so change together:
/// VALIDATION_TIMINGS unset / "" / "0" means do not time; otherwise it IS the sample
/// count, an integer >= 2, first sample discarded, median of the rest. run.sh
/// validates it, and this panics rather than silently not timing when the engine is
/// run by hand with a malformed value.
fn timing_runs() -> Option<usize> {
    let raw = match std::env::var("VALIDATION_TIMINGS") {
        Ok(v) => v,
        Err(_) => return None,
    };
    let v = raw.trim();
    if v.is_empty() || v == "0" {
        return None;
    }
    match v.parse::<usize>() {
        Ok(n) if n >= 2 => Some(n),
        _ => panic!(
            "VALIDATION_TIMINGS must be 0 or an integer >= 2 (got {v:?}); \
             N=2 keeps 1 sample after the warm-up discard"
        ),
    }
}

/// AGQ timing pass, opt-in and orthogonal to `timing_runs`. `VALIDATION_AGQ=<k>`
/// refits the manifest's `agq`-marked datasets at `nagq = k` into
/// `results/glmm_agq_*`. The separate tree is load-bearing: compare.R globs
/// `results/lme4_{empirical,simulated}/*.json` to discover references, and the
/// curated 6-rung oracle is deliberately not expanded by an AGQ timing pass.
/// Mirrors lme4.R's `agq_k` / glmm_python.py's `agq_nagq` / glmm_r.R's `agq_k` —
/// change together.
fn agq_nagq() -> Option<u8> {
    let raw = std::env::var("VALIDATION_AGQ").ok()?;
    let v = raw.trim();
    if v.is_empty() || v == "0" {
        return None;
    }
    match v.parse::<u8>() {
        Ok(n) if n >= 1 && n % 2 == 1 => Some(n),
        _ => panic!(
            "VALIDATION_AGQ must be an ODD integer >= 1 (got {v:?}); \
             the Gauss-Hermite table is built for orders 1, 3, 5, ..."
        ),
    }
}

/// `glmm_agq` on an AGQ pass, `glmm` otherwise — the results subdirectory stem.
fn out_stem() -> &'static str {
    if agq_nagq().is_some() {
        "glmm_agq"
    } else {
        "glmm"
    }
}

fn main() {
    let suite = suite_dir();
    let stem = out_stem();
    std::fs::create_dir_all(format!("{suite}/results/{stem}_empirical")).expect("mk empirical");
    std::fs::create_dir_all(format!("{suite}/results/{stem}_simulated")).expect("mk simulated");
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{suite}/manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");
    // VALIDATION_ONLY=<name>[,<name>...]: fit only the named datasets (mirrors
    // lme4.R / mixedmodels.jl) — reruns a single rung without repaying the full-corpus
    // timing cost (grouseticks alone is ~200 multi-second fits).
    let only = std::env::var("VALIDATION_ONLY").unwrap_or_default();
    let want = |ds: &str| only.is_empty() || only.split(',').any(|s| s == ds);
    // An AGQ pass covers only the `agq`-marked datasets — the shapes glmm's AGQ
    // gate accepts (binomial/Poisson, one grouping factor, q <= 3).
    let agq = agq_nagq().is_some();
    for spec in manifest["datasets"].as_array().expect("manifest.datasets") {
        let ds = spec["name"].as_str().expect("dataset entry missing name");
        if want(ds) && (!agq || rung_agq(spec).is_some()) {
            fit_one(spec);
        }
    }
}

/// Fit one manifest entry end-to-end (load → lower → fit(+SE split) → time →
/// reindex varcomp to the reference's grouping order → write). Mirrors `lme4.R`'s
/// `fit_one` / `mixedmodels.jl`'s `fit_one`.
fn fit_one(spec: &Value) {
    let ds = spec["name"].as_str().expect("manifest entry missing name");
    let rung = spec["rung"].as_u64().expect("manifest entry missing rung");
    let family_str = spec["family"]
        .as_str()
        .expect("manifest entry missing family");
    let gaussian = family_str == "gaussian";
    let source = if spec["source"].as_str() == Some("sim") {
        "simulated"
    } else {
        "empirical"
    };

    let suite = suite_dir();
    // `table` is kept (not just `lo`) so the construction-inclusive timing below
    // can re-run `lower(&table)` in a loop — the Rust analogue of what lme4 /
    // MixedModels / the Python port time (formula+data → model → fit), so all
    // four engines compare same-to-same. See the `_full` timing fields. Data
    // load, formula/family resolution, lowering, and weights_col/offset are
    // shared with `bit_identity/dump.rs` via `lower_rung` (common.rs).
    let (mut lo, table, family, formula_str) = lower_rung(spec, &suite);
    // AGQ pass: quadrature order from the env, and `parallel_inner` left OFF. This
    // pass once turned it on to time the shipped config, which made the aK row
    // uninterpretable: run.sh pins timed fits to one core, so rayon had nothing to
    // spread across and the flag cost 27-37% on the sub-4ms rungs, while the earlier
    // UNPINNED results it was compared against had all P-cores (sim_binomial_slope2
    // measured 0.212 s unpinned vs 1.569 s pinned, both parallel — a 7.4x swing on
    // the pin alone, which read as the two ports being 5.6x slower than the same
    // kernel). Serial is the only config all five engines can share, and neither
    // port can turn inner parallelism on at all (both wrapper crates take glmm with
    // "orchestrate" only). Inner parallelism is measured in
    // campaigns/speed-grid/agq_par_probe.rs, which is built for it.
    if let Some(k) = agq_nagq() {
        lo.opts.nagq = k;
    }
    let timing_batch = spec["timing_batch"].as_u64().unwrap_or(1) as usize;
    let timings = timing_runs();

    // Reference grouping order (compare.R aligns varcomp positionally, not by
    // name) — read off the already-frozen lme4 result rather than re-deriving
    // lme4's own convention. Some rungs (InstEval, rung 47) are TIMING ONLY and
    // never get an lme4 reference generated (manifest.json's note on that entry) —
    // compare.R discovers references by globbing `results/lme4_*/*.json`, so a
    // rung with no file there is simply invisible to it, not a compare.R failure
    // to route around. Falling back to the fit's own declaration order when the
    // file is absent keeps such rungs fitting (and their varcomp populated) under
    // a plain `./run.sh`, which never runs the `lme4`/`jl` engines that would
    // produce the file.
    let ref_order: Vec<String> =
        match std::fs::read_to_string(format!("{suite}/results/lme4_{source}/{ds}.json")) {
            Ok(raw) => {
                let reference: Value = serde_json::from_str(&raw).expect("parse lme4 reference");
                reference["estimates"]["varcomp"]
                    .as_array()
                    .expect("reference estimates.varcomp array")
                    .iter()
                    .map(|e| {
                        e["group"]
                            .as_str()
                            .expect("reference varcomp group name")
                            .to_string()
                    })
                    .collect()
            }
            Err(_) => lo.re_groups.iter().map(|g| g.name.clone()).collect(),
        };

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
    let time_full = |n_runs: usize, opts: &FitOptions| -> f64 {
        median_secs(n_runs, timing_batch, || {
            let l = lower(&formula_str, &table, family)
                .unwrap_or_else(|e| panic!("re-lower {ds}: {e}"));
            let _ = fit_cold(&l.x, &l.y, l.n, l.p, &l.model, &l.ids, opts);
        })
    };

    let fixed_only = lo.re_groups.is_empty();
    let (converged, singular, estimates, timing, n_eval, deviance) = if gaussian {
        let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let t = if let Some(n_runs) = timings {
            json!({
                "fit_seconds_median": median_secs(n_runs, timing_batch, || {
                    let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
                }),
                "fit_seconds_median_full": time_full(n_runs, &lo.opts),
                "n_runs": n_runs, "warmup_discarded": 1, "fits_per_sample": timing_batch,
            })
        } else {
            Value::Null
        };
        let est = json!({
            "beta": nums(&f.beta),
            "se": nums(&f.se),
            "loglik": num(f.loglik),
            "df": f.df,
            "varcomp": varcomp(&f, &lo.re_groups, &ref_order, false),
        });
        (f.converged(), f.singular(), est, t, f.n_eval, f.deviance)
    } else if fixed_only {
        // Fixed-only GLM (weights suite): no θ, so the Rx-vs-Hessian method
        // split is moot — one fit, one SE, emitted as `se_rx` to line up with
        // the single SE lme4.R's `glm`/`glm.nb` writes for these rungs.
        let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let t = if let Some(n_runs) = timings {
            json!({
                "fit_seconds_median": median_secs(n_runs, timing_batch, || {
                    let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
                }),
                "fit_seconds_median_full": time_full(n_runs, &lo.opts),
                "n_runs": n_runs, "warmup_discarded": 1, "fits_per_sample": timing_batch,
            })
        } else {
            Value::Null
        };
        let est = json!({
            "beta": nums(&f.beta),
            "se_rx": nums(&f.se),
            "loglik": num(f.loglik),
            "df": f.df,
            "varcomp": varcomp(&f, &lo.re_groups, &ref_order, false),
        });
        (f.converged(), f.singular(), est, t, f.n_eval, f.deviance)
    } else {
        // GLMM SE has two genuinely different variants (Laplace) — emit both so
        // compare.R checks like to like: se_hessian (keeps θ–β coupling, glmm
        // default) vs se_rx (conditional on θ̂). β/τ is wald_se-independent.
        let o_r = rx_options(&lo.opts);
        let fh = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let fr = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &o_r);
        // Split timing by SE method — the FD-Hessian is the main time consumer,
        // Rx is one closed-form Schur solve. Same PIRLS fit underlies both.
        let t = if let Some(n_runs) = timings {
            json!({
                "fit_seconds_median_rx": median_secs(n_runs, timing_batch, || {
                    let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &o_r);
                }),
                "fit_seconds_median_hessian": median_secs(n_runs, timing_batch, || {
                    let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
                }),
                "fit_seconds_median_rx_full": time_full(n_runs, &o_r),
                "fit_seconds_median_hessian_full": time_full(n_runs, &lo.opts),
                "n_runs": n_runs, "warmup_discarded": 1, "fits_per_sample": timing_batch,
            })
        } else {
            Value::Null
        };
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
            fh.converged() && fr.converged(),
            // singular from the Hessian fit (fh) — same PIRLS fit as fr, so the
            // boundary decision is identical; fh is the one whose θ block the
            // estimates come from.
            fh.singular(),
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
// live in common.rs (shared with grid_fit.rs) — change together.

/// Median seconds over `n_runs` samples, warm-up (first) discarded. `n_runs` comes
/// from `timing_runs()`, i.e. from run.sh's `--timings[=N]`; callers must already
/// have guarded on it, nothing here checks. Each sample times
/// `batch` fits so sub-resolution fits stay above the timer floor (mirrors the manifest
/// `timing_batch` the R/Julia oracles read) — the returned median is for `batch` fits;
/// divide by `batch` for the per-fit estimate. GLMM rungs call this once per SE method
/// (Rx vs Hessian) because the FD-Hessian is the dominant cost.
fn median_secs(n_runs: usize, batch: usize, mut f: impl FnMut()) -> f64 {
    let mut t = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
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
    let out = format!("{}/results/{}_{source}/{ds}.json", suite_dir(), out_stem());
    std::fs::write(
        &out,
        format!("{}\n", serde_json::to_string_pretty(&res).unwrap()),
    )
    .expect("write result");
    let conv = res["converged"].as_bool().unwrap_or(false);
    // GLMM rungs store the split rx/hessian medians, not a single median — show
    // the Rx one on the console (mirrors mixedmodels.jl's t_disp fallback).
    // None on an untimed run (`timing` is null) — the console line then omits the
    // time rather than printing a NaN, matching the other four engines.
    let t = res["timing"]["fit_seconds_median"]
        .as_f64()
        .or_else(|| res["timing"]["fit_seconds_median_rx"].as_f64());
    let t_disp = match t {
        Some(t) => format!("  fit_median={t:.4}s"),
        None => String::new(),
    };
    println!(
        "glmm  {ds:<12}  rung {}  converged={conv}{t_disp}",
        res["rung"]
    );
}
