//! glmm side of the accuracy grid (`grid/manifest.json`): one JSONL record per
//! cell, appended to `$GRID_OUT`. Resume-safe: cells already present in the
//! output are skipped, so the watchdog can kill and relaunch at will. Per-cell
//! panics are caught and recorded as engine-fail — grid corners are expected to
//! break engines; a crash is a data point.
//!
//! Environment, the same five variables every engine of this grid reads:
//! `GRID_DIR` (the `grid/` directory), `GRID_MANIFEST`, `GRID_OUT`,
//! `GRID_CELLS` (comma-separated ids this worker must fit) and `GRID_TIMED`
//! (`""`/`"0"` = untimed, else the sample count, an integer >= 2). Each has a
//! default so the example can be run by hand.
//!
//! The record builder itself lives in `common.rs`, so the crate's own reference
//! check over the `fast` cells fits them through exactly this code.

use std::io::Write;

use serde_json::Value;

#[path = "common.rs"]
mod harness_common;
use harness_common::*;

const VERSION: &str = env!("CARGO_PKG_VERSION");
/// `grid/` relative to the `validation/` crate dir, matching the engine file's
/// own `../`, so the cwd does not matter when the example is run by hand.
const DEFAULT_GRID_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/grid");

fn main() {
    let grid_dir = std::env::var("GRID_DIR").unwrap_or_else(|_| DEFAULT_GRID_DIR.to_string());
    let manifest_path =
        std::env::var("GRID_MANIFEST").unwrap_or_else(|_| format!("{grid_dir}/manifest.json"));
    let out_path =
        std::env::var("GRID_OUT").unwrap_or_else(|_| format!("{grid_dir}/results.jsonl"));
    let only = std::env::var("GRID_CELLS").unwrap_or_default();
    let timed = timed_samples();

    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).expect("mk output dir");
    }
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read grid manifest"))
            .expect("parse grid manifest");
    let done = done_cells(&out_path);
    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)
        .expect("open grid output");

    let cells = manifest["cells"].as_array().expect("manifest.cells");
    let cells_by_id: std::collections::HashMap<&str, &Value> = cells
        .iter()
        .map(|c| (c["cell"].as_str().expect("cell.cell"), c))
        .collect();
    // Fit order is GRID_CELLS's own order (manifest order when it is empty),
    // not a re-derived one: the runner's watchdog blames a timeout on the
    // first cell of GRID_CELLS still missing from the output, so fitting in
    // any other order fits the wrong cell first and the blame lands on the
    // wrong one (mirrors run.sh's `next_missing` -- change together).
    let ids: Vec<&str> = if only.is_empty() {
        cells
            .iter()
            .map(|c| c["cell"].as_str().expect("cell.cell"))
            .collect()
    } else {
        only.split(',').collect()
    };

    for id in ids {
        if done.contains(id) {
            continue;
        }
        let cell = cells_by_id
            .get(id)
            .unwrap_or_else(|| panic!("unknown cell id not in manifest: {id}"));
        let rec = fit_cell(cell, &grid_dir, timed, VERSION);
        writeln!(out, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
        out.flush().unwrap(); // line-per-fit flush: the watchdog watches mtime
    }
}

/// Sample count for this run, or `None` when timing is off.
///
/// THE contract, mirrored by every engine of this grid — five languages that
/// cannot share code, so change together: `GRID_TIMED` unset / `""` / `"0"`
/// means do not time; otherwise it IS the sample count, an integer >= 2, first
/// sample discarded, median of the rest. This panics rather than silently not
/// timing when the engine is run by hand with a malformed value.
fn timed_samples() -> Option<usize> {
    let raw = std::env::var("GRID_TIMED").ok()?;
    let v = raw.trim();
    if v.is_empty() || v == "0" {
        return None;
    }
    match v.parse::<usize>() {
        Ok(n) if n >= 2 => Some(n),
        _ => panic!(
            "GRID_TIMED must be 0 or an integer >= 2 (got {v:?}); \
             N=2 keeps 1 sample after the warm-up discard"
        ),
    }
}
