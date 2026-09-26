#![cfg(feature = "formula")]
//! Saved-reference check: refit every `fast` cell of the accuracy grid and
//! compare it against the newest committed glmm run under
//! `validation/grid/runs/glmm/`.
//!
//! This is a REGRESSION gate against the crate's own recorded answers, not a
//! cross-engine agreement check. It needs no R, no Julia and no network: the
//! manifest, the run and the data files it reads are all committed. Which is
//! what makes it the accuracy check that can run on every push, while the
//! oracle comparisons stay local.
//!
//! The comparison record is built by `fit_cell` in
//! `validation/grid/engines/common.rs` -- the same function the `grid_glmm`
//! engine that produced the reference calls. Sharing it is the point: a
//! convention that lived in two places (which SE arm is primary, whether
//! `sigma` is a variance or a scale, what `varcomp` holds) would drift, and the
//! drift would read as a regression.
//!
//! WHAT A FAILURE MEANS, in the order the test checks it:
//!   * no committed run with `subset: "full"` -- nothing here covers the cells
//!     this gate reads. A partial run is not a smaller reference, it is not one.
//!   * `engine_version` mismatch -- the crate version moved and nobody recorded
//!     a new reference run. Record one (see the message the test prints).
//!   * a `fast` cell with no record, or a record that did not converge -- the
//!     reference run is not usable as one.
//!   * a quantity outside the band -- either the kernel changed an answer, or
//!     the band does not cover this machine. `validation/grid/tol.R`'s
//!     `ci_ref_rel` comment says how that is settled; it is never settled by
//!     widening the band here.

#[allow(dead_code)]
#[path = "../validation/grid/engines/common.rs"]
mod grid_common;

use grid_common::fit_cell;
use serde_json::Value;

/// The band every quantity here is compared at. Mirrors `TOL$ci_ref_rel` in
/// `validation/grid/tol.R` and `CI_REF_REL` in
/// `r/tests/testthat/helper-pins.R` -- change all three together. `tol.R`
/// carries the provenance: it is the crate's cross-architecture pin band for the
/// iterative fit paths, plus the release-versus-test build profile difference,
/// and the measurement on CI's own runner pool is still pending.
///
/// It is restated rather than imported because nothing in a Rust test can read
/// an R file, and because `src/fit/common_tests.rs` is `pub(crate)` and
/// `cfg(test)`, so an integration test cannot reach `PIN_REL_ITER` either.
const CI_REF_REL: f64 = 1e-7;

/// The grid directory, absolute at compile time so the test does not depend on
/// the working directory. Mirrors how `tests/oracle_support` resolves
/// `validation/goldens/`.
const GRID: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/validation/grid");

fn read_json(path: &str) -> Value {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// The glmm run this gate compares against: the newest run directory under
/// `runs/glmm/` whose `run_meta.json` says `subset: "full"`.
///
/// Two filters, each for its own reason.
///
/// `scratch/` is skipped. That is where a run started without `--keep` lands, it
/// is gitignored, and a local smoke run must never be the reference a push is
/// gated on.
///
/// `subset` must be `"full"`. Only an invocation that fitted the whole manifest
/// covers the cells this gate reads, and `run.sh` is the only writer of that
/// field. The filter is not decoration: a `--fast --keep` run lands right beside
/// the baseline, so without it the newest directory could be a 58-cell run that
/// happens to carry the same cells, or a cell-restricted run that carries almost
/// none.
///
/// Among what is left, newest wins, by the `date` field of `run_meta.json`
/// compared as a string. The dates are ISO-8601 with an offset, so string order
/// is time order for runs recorded in one zone, which is what a committed
/// reference run is.
fn newest_committed_run() -> (String, Value) {
    let root = format!("{GRID}/runs/glmm");
    let mut best: Option<(String, String, Value)> = None;
    for entry in std::fs::read_dir(&root).unwrap_or_else(|e| panic!("read dir {root}: {e}")) {
        let dir = entry.expect("dir entry").path();
        if !dir.is_dir() || dir.file_name().is_some_and(|n| n == "scratch") {
            continue;
        }
        let dir = dir.to_str().expect("utf-8 path").to_string();
        if !std::path::Path::new(&format!("{dir}/run_meta.json")).exists()
            || !std::path::Path::new(&format!("{dir}/results.jsonl")).exists()
        {
            continue;
        }
        let meta = read_json(&format!("{dir}/run_meta.json"));
        if meta["subset"].as_str() != Some("full") {
            continue;
        }
        let date = meta["date"].as_str().expect("run_meta.date").to_string();
        if best.as_ref().is_none_or(|(d, _, _)| date > *d) {
            best = Some((date, dir, meta));
        }
    }
    let (_, dir, meta) = best.unwrap_or_else(|| {
        panic!(
            "no committed glmm run with subset \"full\" under {root} (scratch/ \
             excluded). Only a full run covers the cells this gate reads. Record \
             one with `validation/grid/run.sh glmm --keep --label \"release \
             <version> baseline\"` and commit it."
        )
    });
    (dir, meta)
}

/// Every record of a run, keyed by cell id.
fn read_records(path: &str) -> std::collections::HashMap<String, Value> {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap_or_else(|e| panic!("{path}: {e}"));
            let id = v["cell"].as_str().expect("record.cell").to_string();
            (id, v)
        })
        .collect()
}

/// Relative where the reference has magnitude, absolute where it does not.
///
/// The absolute leg is what makes a correlation coordinate comparable: an
/// off-diagonal that sits at 3e-9 in both runs has no relative question to
/// answer, and `CI_REF_REL` in absolute terms is far below anything a
/// correlation carries as signal.
fn close(got: f64, want: f64) -> bool {
    (got - want).abs() <= CI_REF_REL * (1.0 + want.abs())
}

/// The relative difference reported for a pair, for the worst-gap line the test
/// prints. Absolute below 1 for the same reason `close` is.
fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / (1.0 + want.abs())
}

/// Compare two records' values, which must have the same shape all the way
/// down: same array lengths, same object keys, same nulls, same strings, and
/// every number inside the band. A shape difference is reported, not
/// papered over -- a `varcomp` block that lost a grouping, or a `se_hessian`
/// that turned up on a gaussian cell, is a regression and not a near miss.
/// A key absent from the record and a key present with an explicit `null`
/// compare equal on either side, because the caller already turned a missing
/// key into `Value::Null` before this function sees it: a schema change from
/// omitting a key to writing `null` for it, or the reverse, passes unnoticed.
fn cmp_json(got: &Value, want: &Value, ctx: &str, fails: &mut Vec<String>, worst: &mut f64) {
    match (got, want) {
        (Value::Null, Value::Null) => {}
        (Value::Bool(a), Value::Bool(b)) => {
            if a != b {
                fails.push(format!("{ctx}: {a} vs reference {b}"));
            }
        }
        (Value::String(a), Value::String(b)) => {
            if a != b {
                fails.push(format!("{ctx}: {a:?} vs reference {b:?}"));
            }
        }
        (Value::Number(a), Value::Number(b)) => {
            let (a, b) = (a.as_f64().expect("f64"), b.as_f64().expect("f64"));
            let r = rel(a, b);
            if r > *worst {
                *worst = r;
            }
            if !close(a, b) {
                fails.push(format!("{ctx}: {a} vs reference {b} (rel {r:.2e})"));
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                fails.push(format!(
                    "{ctx}: length {} vs reference {}",
                    a.len(),
                    b.len()
                ));
                return;
            }
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                cmp_json(x, y, &format!("{ctx}[{i}]"), fails, worst);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            let ka: Vec<&String> = a.keys().collect();
            let kb: Vec<&String> = b.keys().collect();
            if ka != kb {
                fails.push(format!("{ctx}: keys {ka:?} vs reference {kb:?}"));
                return;
            }
            for k in ka {
                cmp_json(&a[k], &b[k], &format!("{ctx}.{k}"), fails, worst);
            }
        }
        _ => fails.push(format!("{ctx}: shape changed -- {got} vs reference {want}")),
    }
}

/// The keys compared. Every fitted quantity the record carries, and nothing
/// else.
///
/// `n_eval` is deliberately OUT: the optimizer's evaluation count is a path
/// length, and a different build or CPU can reach the same optimum by a
/// different route, so gating it would report a rounding difference as a
/// regression. `wall_seconds` is null on an untimed run, `message` is prose,
/// and `engine`/`engine_version`/`cell`/`status`/`fits_per_sample` are
/// provenance the test checks separately or not at all. `deviance` is kept even
/// though `set_loglik` derives it from `loglik`: it costs nothing and it is the
/// quantity the grid's own hard gate reads.
const GATED: [&str; 11] = [
    "converged",
    "singular",
    "coef_names",
    "beta",
    "se_rx",
    "se_hessian",
    "varcomp",
    "sigma",
    "nb_theta",
    "loglik",
    "deviance",
];

#[test]
fn fast_cells_match_the_committed_glmm_run() {
    let manifest = read_json(&format!("{GRID}/manifest.json"));
    let (run_dir, meta) = newest_committed_run();
    // Printed so the run being gated against is never a guess. `subset` is
    // "full" by the selection rule, not by luck -- `newest_committed_run` skips
    // anything else -- and printing it is what makes that visible in a log.
    println!(
        "reference run: {run_dir}\n  subset: {}   label: {}   cells: {}",
        meta["subset"], meta["label"], meta["cells"]
    );

    // The reference has to have been fitted by THIS crate version. A version
    // bump is the moment to record a new run; failing here is how that gets
    // remembered.
    assert_eq!(
        meta["engine_version"].as_str(),
        Some(env!("CARGO_PKG_VERSION")),
        "the newest committed glmm run {run_dir} was fitted by glmm {}, but this \
         crate is {}. Record a new full run with `validation/grid/run.sh glmm \
         --keep --label \"release {} baseline\"`, commit it, and do not edit the \
         old one.",
        meta["engine_version"],
        env!("CARGO_PKG_VERSION"),
        env!("CARGO_PKG_VERSION")
    );

    let reference = read_records(&format!("{run_dir}/results.jsonl"));
    let cells: Vec<&Value> = manifest["cells"]
        .as_array()
        .expect("manifest.cells")
        .iter()
        .filter(|c| {
            c["tags"]
                .as_array()
                .is_some_and(|t| t.iter().any(|x| x == "fast"))
        })
        .collect();
    assert!(
        !cells.is_empty(),
        "no cell in {GRID}/manifest.json carries the `fast` tag"
    );

    let mut fails: Vec<String> = Vec::new();
    // Cell ids that produced at least one line in `fails`, tracked separately
    // from `fails` itself: `fails` holds one line per offending QUANTITY, so a
    // cell whose 12-coefficient `beta` all move counts as one bad cell, not 12.
    let mut fail_cells: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut worst = 0.0f64;
    let mut worst_cell = String::new();
    for cell in &cells {
        let id = cell["cell"].as_str().expect("cell.cell");
        let Some(want) = reference.get(id) else {
            fails.push(format!(
                "{id}: no record in the reference run -- it is tagged `fast`, so the \
                 run that was recorded did not cover the subset this gate reads"
            ));
            fail_cells.insert(id);
            continue;
        };
        // A reference record that is not a converged fit is not a reference. It
        // is skipped nowhere: it fails, because the alternative is a green run
        // that compared nothing on this cell.
        if want["status"].as_str() != Some("ok") || want["converged"] != Value::Bool(true) {
            fails.push(format!(
                "{id}: the reference record is status {} / converged {} -- record a \
                 clean run before gating against it",
                want["status"], want["converged"]
            ));
            fail_cells.insert(id);
            continue;
        }
        let got = fit_cell(cell, GRID, None, env!("CARGO_PKG_VERSION"));
        if got["converged"] != Value::Bool(true) {
            fails.push(format!(
                "{id}: this build did not converge -- {}",
                got["message"]
            ));
            fail_cells.insert(id);
            continue;
        }
        let before = worst;
        let fails_before_cell = fails.len();
        for key in GATED {
            cmp_json(
                got.get(key).unwrap_or(&Value::Null),
                want.get(key).unwrap_or(&Value::Null),
                &format!("{id}.{key}"),
                &mut fails,
                &mut worst,
            );
        }
        if fails.len() > fails_before_cell {
            fail_cells.insert(id);
        }
        if worst > before {
            worst_cell = id.to_string();
        }
    }

    // Printed on every run, pass or fail: this is the local half of
    // `ci_ref_rel`'s measurement, and a run that prints it is a run whose band
    // can be re-sized from evidence.
    println!(
        "{} fast cells compared; worst relative gap {worst:.2e} ({worst_cell}), band {CI_REF_REL:.0e}",
        cells.len()
    );
    assert!(
        fails.is_empty(),
        "{} of {} fast cells differ from the committed glmm run {run_dir}:\n{}",
        fail_cells.len(),
        cells.len(),
        fails.join("\n")
    );
}
