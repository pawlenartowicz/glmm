//! Scratch probe for the NB θ-search step: fits the three NB GLMM `m3_goldens`
//! rungs (`sim_nb_glmm`, `sim_nb_nested_glmm`, `sim_sparse_nb`) and prints one
//! JSON line each with the evaluation count, the bracket's node bookkeeping,
//! θ̂_NB, the deviance, the marginal log-likelihood and the minimum wall over
//! `NB_PROBE_RUNS` fits (default 5; the first fit is not timed). Nothing is
//! written to disk. No existing driver fits these rungs outside `cargo test`:
//! `run.sh` and the bit-identity dump fit `datasets` only.
//!
//!   cargo run --release -p validation --example nb_theta_probe --features validation/counters
//!
//! Timing is meaningful only on a locked machine (validation/README.md
//! §Running). Delete this file and its Cargo.toml entry once the sparse NB
//! route has lost its bracket too.
use std::time::Instant;

use glmm::fit_cold;
use serde_json::{json, Value};

#[path = "engines/common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");
const RUNGS: [&str; 3] = ["sim_nb_glmm", "sim_nb_nested_glmm", "sim_sparse_nb"];

fn main() {
    let runs: usize = std::env::var("NB_PROBE_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");
    let specs = manifest["m3_goldens"]
        .as_array()
        .expect("manifest.m3_goldens");
    for name in RUNGS {
        let mut spec = specs
            .iter()
            .find(|s| s["name"].as_str() == Some(name))
            .unwrap_or_else(|| panic!("{name} not in m3_goldens"))
            .clone();
        // `m3_goldens` entries carry a bare `data` name; all three live under
        // data/simulated/, which `lower_rung` selects on `source == "sim"`.
        spec["source"] = json!("sim");
        let (lo, _table, _family, _formula) = lower_rung(&spec, DIR);
        // Untimed first fit: the reported numbers, and the warm-up.
        let f = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
        let mut wall_min = f64::INFINITY;
        for _ in 0..runs {
            let t0 = Instant::now();
            let _ = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
            wall_min = wall_min.min(t0.elapsed().as_secs_f64());
        }
        // Mutated only under `counters` (below); plain builds never write into
        // it after construction.
        #[cfg_attr(not(feature = "counters"), allow(unused_mut))]
        let mut rec = json!({
            "rung": name,
            "converged": f.converged(),
            "n_eval": f.n_eval,
            "dispersion": num(f.dispersion),
            "deviance": num(f.deviance),
            "loglik": num(f.loglik),
            "beta": nums(&f.beta),
            "wall_min_seconds": wall_min,
            "wall_runs": runs,
        });
        #[cfg(feature = "counters")]
        {
            rec["nb_nodes"] = json!(f.counters.nb_nodes);
            rec["nb_evals_total"] = json!(f.counters.nb_evals_total);
        }
        println!("{rec}");
    }
}
