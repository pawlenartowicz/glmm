//! glmm side of the optimizer-grid campaign (manifest.json): one JSONL
//! record per cell, appended to $GRID_OUT. Resume-safe: cells already present
//! in the output are skipped, so the watchdog (run.sh) can kill and
//! relaunch at will. Per-cell panics are caught and recorded as engine-fail —
//! grid corners are expected to break engines; a crash is a data point.
use std::io::Write;
use std::time::Instant;

use glmm::fit_cold;
use serde_json::{json, Value};

// path mirrors validation/engines layout — Task 6 renames it
#[path = "../../engines/common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

fn main() {
    let manifest_path = std::env::var("GRID_MANIFEST")
        .unwrap_or_else(|_| format!("{DIR}/campaigns/speed-grid/manifest.json"));
    let out_path = std::env::var("GRID_OUT")
        .unwrap_or_else(|_| format!("{DIR}/campaigns/speed-grid/results/glmm_shipped.jsonl"));
    let tag = std::env::var("GRID_CONFIG_TAG").unwrap_or_default();
    let only = std::env::var("GRID_ONLY").unwrap_or_default();
    // One cell per process (run.sh sets this for the glmm engine only — the
    // two sites change together): a cell's wall time swung 2-10x on sub-100ms
    // cells depending on the allocator/cache state left behind by whatever
    // ran before it in the same process, and on where kill-relaunch
    // boundaries fell between passes. A fresh process per cell gives every
    // cell the same timing context.
    let one_cell = !std::env::var("GRID_ONE_CELL")
        .unwrap_or_default()
        .is_empty();
    std::fs::create_dir_all(std::path::Path::new(&out_path).parent().unwrap()).unwrap();

    // run.sh owns the budget and exports it per launch; the fallback lets
    // this example still run standalone (change together with run.sh's
    // GRID_CELL_BUDGET export).
    let budget: f64 = std::env::var("GRID_CELL_BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(240.0);

    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read grid manifest"))
            .expect("parse grid manifest");
    let done = done_case_ids(&out_path);
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .unwrap();

    for cell in manifest["cells"].as_array().expect("cells") {
        let case_id = cell["case_id"].as_str().unwrap();
        if !only.is_empty() && !only.split(',').any(|s| s == case_id) {
            continue;
        }
        if done.contains(case_id) {
            continue;
        }
        let rec = fit_cell(cell, &tag, budget);
        writeln!(out, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
        out.flush().unwrap(); // line-per-fit flush: the watchdog watches mtime
        if one_cell {
            return;
        }
    }
}

/// θ̂ in the common cross-engine schema, one entry per grouping factor, from
/// `Fit::stddev_corr` (arbitrary q) — same shape as `engines/glmm.rs::varcomp`
/// (minus the reference-order reindex: the diligent run's R side joins by
/// group name, not position) and R's `varcomp_of`/`goldens_agq.R`'s
/// GLMMadaptive branch, so all three engines' diligent output is directly
/// joinable on `{group, terms}`. `include_se` gates `stddev_se` to the mixed
/// non-Gaussian Hessian path (`Fit::stddev_se` is populated only there, and
/// only for scalar (q=1) groupings — see its doc comment); LMM callers pass
/// `false`.
fn varcomp(f: &glmm::Fit, re_groups: &[glmm::formula::ReGroupInfo], include_se: bool) -> Value {
    let mut theta_offset = 0usize;
    Value::Array(
        re_groups
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
            .collect(),
    )
}

fn done_case_ids(path: &str) -> std::collections::HashSet<String> {
    let Ok(s) = std::fs::read_to_string(path) else {
        return Default::default();
    };
    s.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| v["case_id"].as_str().map(str::to_string))
        .collect()
}

/// config_tag: `<site>:<npt formula or shipped>[:<GRID_CONFIG_TAG>]`. The site
/// is derived from the LOWERED model — relations (`Crossed` vs `NestedWithin`,
/// which reflects the frontend's flat-nesting detection outcome) and slope
/// widths from `lo.model.re`, crossed level counts from `lo.ids.extra[g]`
/// (`max+1`; the lowered spec itself carries placeholder counts) — by mirroring
/// src/fit.rs's classify_design: over-envelope, any slope-carrying extra
/// grouping, or Σ crossed levels past MAX_CROSSED_LEVELS routes Sparse
/// (mirrors classify_design — change together).
fn config_tag(lo: &glmm::formula::Lowered, gaussian: bool, user_tag: &str) -> String {
    let sparse = match lo.model.re.as_ref() {
        None => false,
        Some(re) => {
            let crossed_levels: usize = re
                .extra_groupings
                .iter()
                .enumerate()
                .filter(|(_, g)| matches!(g.relation, glmm::GroupingRelation::Crossed { .. }))
                .map(|(g, _)| {
                    lo.ids.extra[g]
                        .iter()
                        .copied()
                        .max()
                        .map_or(1, |m| m as usize + 1)
                })
                .sum();
            re.extra_groupings.len() > glmm::consts::MAX_EXTRA_GROUPINGS
                || 1 + re.slopes.len() > glmm::consts::MAX_PRIMARY_Q
                || re.extra_groupings.iter().any(|g| !g.slopes.is_empty())
                || crossed_levels > glmm::consts::MAX_CROSSED_LEVELS
        }
    };
    let site = match (gaussian, sparse) {
        (true, false) => "lmm-dense",
        (true, true) => "lmm-sparse",
        (false, true) => "glmm-sparse",
        (false, false) => "glmm",
    };
    let npt = std::env::var("LMM_NPT_FORMULA").unwrap_or_else(|_| "shipped".into());
    let two = if std::env::var("LMM_TWO_STAGE").is_ok() {
        ":2stage"
    } else {
        ""
    };
    if user_tag.is_empty() {
        format!("{site}:{npt}{two}")
    } else {
        format!("{site}:{npt}{two}:{user_tag}")
    }
}

/// Fit one cell, timed (`wall_seconds`). `lower_seconds` (CSV→matrix
/// marshalling, `lower_grid_cell`) stays OUTSIDE the timed region — it is the
/// Rust-side analogue of MM's harness already holding a `DataFrame` before
/// `fit_cell` starts.
///
/// Timing protocol: one discarded warm-up fit, then up to two timed fits,
/// reporting the min-wall rep when both run. The warm-up absorbs one-time
/// per-process lazy init — faer's first triangular matmul costs ~4–9 ms and
/// used to land inside the timed solve of whichever cell first took the
/// unbalanced dense-LMM path (the 40× `lmm_int1_g300p5_skew_base` outlier in
/// every pre-fix pass). The min is honest: fits are deterministic (identical
/// eval sequence — `rep_mismatch` flags any drift) and timing noise on a
/// locked machine is one-sided. Mirrors fit.jl's warm-up discard.
///
/// `budget` (seconds, shared across the cell's fits — not per-fit) caps how
/// many fits are attempted: after each fit finishes, the next one starts only
/// if `elapsed + duration_of_the_fit_just_finished <= budget`, predicting the
/// next fit's duration from the one that just finished (fits within a cell
/// are deterministic and near-identical in duration) rather than checking
/// `elapsed < budget` — the plain form would start a fit at e.g. budget-1s
/// and let it run for minutes before the shell watchdog kills the whole
/// cell, discarding a perfectly good partial measurement. `wall_source`
/// records which case applied (`min_of_2`, `single`, `warmup`) and `n_reps`
/// (0, 1 or 2) the number of *timed* fits that contributed. run.sh owns the
/// budget number and passes it via `GRID_CELL_BUDGET` — change together.
fn fit_cell(cell: &Value, user_tag: &str, budget: f64) -> Value {
    let case_id = cell["case_id"].as_str().unwrap().to_string();
    let seed = cell["seed"].as_i64().unwrap_or(0);
    let mut rec = json!({
        "case_id": case_id, "seed": seed, "engine": "glmm",
        "optimizer": "bobyqa",
    });
    // Lower first (own catch_unwind: a lowering panic is engine-fail with no
    // site to report), so config_tag can read the lowered relations/ids.
    let t_lower = Instant::now();
    let mut lowered =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lower_grid_cell(cell, DIR)));
    let lower_seconds = t_lower.elapsed().as_secs_f64();
    // Diligent manifests stamp `nagq` on the 33 AGQ-eligible cells (spec Part 6);
    // absent on every other cell (shipped/eval-efficiency manifests included),
    // where this is a no-op leaving `FitOptions::nagq`'s default (1 = Laplace).
    if let Ok((lo, ..)) = &mut lowered {
        if let Some(k) = cell["nagq"].as_u64() {
            lo.opts.nagq = k as u8;
        }
    }
    let (result, wall_seconds): (Option<glmm::Fit>, f64) = match &lowered {
        Ok((lo, gaussian, _)) => {
            rec["config_tag"] = json!(config_tag(lo, *gaussian, user_tag));
            let one_fit = || -> (Option<glmm::Fit>, f64) {
                let t_fit = Instant::now();
                let f = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts)
                }))
                .ok();
                let wall = t_fit.elapsed().as_secs_f64();
                (f, wall)
            };
            let warm = one_fit();
            if warm.0.is_none() {
                // Warm-up panicked — record it as-is; re-fitting the same
                // cell would panic again.
                rec["n_reps"] = json!(0);
                rec["wall_source"] = json!("warmup");
                warm
            } else if warm.1 + warm.1 > budget {
                // Predictive check: elapsed (= warm.1) + duration of the
                // fit that just finished (= warm.1) would exceed budget.
                rec["n_reps"] = json!(0);
                rec["wall_source"] = json!("warmup");
                warm
            } else {
                let a = one_fit();
                let elapsed = warm.1 + a.1;
                if elapsed + a.1 > budget {
                    rec["n_reps"] = json!(1);
                    rec["wall_source"] = json!("single");
                    a
                } else {
                    let b = one_fit();
                    rec["n_reps"] = json!(2);
                    rec["wall_source"] = json!("min_of_2");
                    if let (Some(fa), Some(fb)) = (&a.0, &b.0) {
                        // Determinism QA — identical work per rep is what makes
                        // min-of-2 honest; flag drift rather than silently
                        // taking the min over disagreeing fits.
                        if fa.deviance.to_bits() != fb.deviance.to_bits() || fa.n_eval != fb.n_eval
                        {
                            rec["rep_mismatch"] = json!(true);
                        }
                    }
                    match (a.0.is_some(), b.0.is_some()) {
                        (true, false) => a,
                        (false, true) => b,
                        _ => {
                            if a.1 <= b.1 {
                                a
                            } else {
                                b
                            }
                        }
                    }
                }
            }
        }
        Err(_) => {
            rec["config_tag"] = json!(if user_tag.is_empty() {
                "lower-fail".to_string()
            } else {
                format!("lower-fail:{user_tag}")
            });
            (None, 0.0)
        }
    };
    match result {
        Some(f) => {
            let max_fun = lowered.as_ref().map(|(_, _, m)| *m).unwrap_or(0);
            let maxeval = !f.converged() && f.n_eval >= max_fun;
            rec["n_eval"] = json!(f.n_eval);
            rec["converged"] = json!(f.converged());
            rec["singular"] = json!(f.singular());
            rec["deviance"] = num(f.deviance);
            rec["beta"] = nums(&f.beta);
            rec["se"] = nums(&f.se);
            // theta hat (diligent-run recording, spec Part 6): reduced to
            // stddev+corr per grouping via `Fit::stddev_corr`, mirroring
            // `engines/glmm.rs::varcomp`/R's `varcomp_of` schema so the three
            // runners join on the same field. Empty for fixed-only fits (no
            // `re_groups`). `include_se` mirrors fit.rs: only the mixed
            // non-Gaussian path has a populated `stddev_se` (scalar groupings).
            rec["varcomp"] = lowered
                .as_ref()
                .ok()
                .map(|(lo, gaussian, _)| varcomp(&f, &lo.re_groups, !gaussian))
                .unwrap_or_else(|| json!([]));
            rec["status"] = json!(if maxeval {
                "maxeval"
            } else if f.converged() {
                "ok"
            } else {
                "engine-fail"
            });
        }
        None => {
            rec["n_eval"] = json!(0);
            rec["converged"] = json!(false);
            rec["singular"] = json!(false);
            rec["deviance"] = Value::Null;
            rec["beta"] = json!([]);
            rec["se"] = json!([]);
            rec["varcomp"] = json!([]);
            rec["status"] = json!("engine-fail");
        }
    }
    rec["lower_seconds"] = num(lower_seconds);
    rec["wall_seconds"] = num(wall_seconds);
    rec
}
