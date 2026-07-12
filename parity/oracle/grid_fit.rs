//! glmm side of the optimizer-grid campaign (manifest_grid.json): one JSONL
//! record per cell, appended to $GRID_OUT. Resume-safe: cells already present
//! in the output are skipped, so the watchdog (run_grid.sh) can kill and
//! relaunch at will. Per-cell panics are caught and recorded as engine-fail —
//! grid corners are expected to break engines; a crash is a data point.
use std::io::Write;
use std::time::Instant;

use glmm::fit_cold;
use serde_json::{json, Value};

#[path = "harness_common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

fn main() {
    let manifest_path = std::env::var("GRID_MANIFEST")
        .unwrap_or_else(|_| format!("{DIR}/parity/manifest_grid.json"));
    let out_path = std::env::var("GRID_OUT")
        .unwrap_or_else(|_| format!("{DIR}/parity/results/grid/glmm_shipped.jsonl"));
    let tag = std::env::var("GRID_CONFIG_TAG").unwrap_or_default();
    let only = std::env::var("GRID_ONLY").unwrap_or_default();
    std::fs::create_dir_all(std::path::Path::new(&out_path).parent().unwrap()).unwrap();

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
        let rec = fit_cell(cell, &tag);
        writeln!(out, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
        out.flush().unwrap(); // line-per-fit flush: the watchdog watches mtime
    }
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
fn config_tag(lo: &glmm_formula::Lowered, gaussian: bool, user_tag: &str) -> String {
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

fn fit_cell(cell: &Value, user_tag: &str) -> Value {
    let case_id = cell["case_id"].as_str().unwrap().to_string();
    let seed = cell["seed"].as_i64().unwrap_or(0);
    let mut rec = json!({
        "case_id": case_id, "seed": seed, "engine": "glmm",
        "optimizer": "bobyqa",
    });
    let t0 = Instant::now();
    // Lower first (own catch_unwind: a lowering panic is engine-fail with no
    // site to report), so config_tag can read the lowered relations/ids.
    let lowered =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lower_grid_cell(cell, DIR)));
    let result: Option<glmm::Fit> = match &lowered {
        Ok((lo, gaussian, _)) => {
            rec["config_tag"] = json!(config_tag(lo, *gaussian, user_tag));
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts)
            }))
            .ok()
        }
        Err(_) => {
            rec["config_tag"] = json!(if user_tag.is_empty() {
                "lower-fail".to_string()
            } else {
                format!("lower-fail:{user_tag}")
            });
            None
        }
    };
    let wall = t0.elapsed().as_secs_f64();
    match result {
        Some(f) => {
            let max_fun = lowered.as_ref().map(|(_, _, m)| *m).unwrap_or(0);
            let maxeval = !f.converged && f.n_eval >= max_fun;
            rec["n_eval"] = json!(f.n_eval);
            rec["converged"] = json!(f.converged);
            rec["singular"] = json!(f.singular);
            rec["deviance"] = num(f.deviance);
            rec["beta"] = nums(&f.beta);
            rec["se"] = nums(&f.se);
            rec["status"] = json!(if maxeval {
                "maxeval"
            } else if f.converged {
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
            rec["status"] = json!("engine-fail");
        }
    }
    rec["wall_seconds"] = json!(wall);
    rec
}
