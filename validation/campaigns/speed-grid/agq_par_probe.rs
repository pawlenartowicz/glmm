//! Scratch probe for the AGQ parallel-default question: for every AGQ-eligible
//! speed-grid cell, time `parallel_inner` off vs on at several `nagq` orders and
//! print the ratio. One fit per arm, nothing written to disk — the numbers are a
//! sanity check on whether inner rayon is worth anything at `nagq > 1`, not a
//! campaign result.
//!
//! Requires the `parallel` feature on glmm, otherwise both arms run the same
//! serial code (`src/glmm/mod.rs:332` gates `cluster_rows` on the cfg):
//!
//!   cargo run --release -p validation --example agq_par_probe --features glmm/parallel
use std::io::Write;
use std::time::Instant;

use glmm::fit_cold;
use serde_json::Value;

#[path = "../../engines/common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

/// Odd orders only — the GH table is built for `1,3,…,MAX_NAGQ`. `nagq = 1` is
/// omitted: it is Laplace, both arms take the identical serial path there, and
/// the shipped speed-grid pass already has those walls.
const NAGQS: [u8; 3] = [3, 7, 11];

/// Per-cell cap across all its arms. A cell that blows through it gets its
/// remaining orders skipped, reported rather than dropped quietly.
const CELL_BUDGET_SECONDS: f64 = 120.0;

fn main() {
    let manifest_path = format!("{DIR}/campaigns/speed-grid/manifest.json");
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read grid manifest"))
            .expect("parse grid manifest");

    // AGQ fires only on binomial/Poisson with a single grouping factor and
    // q_p <= 3 (the gate in src/glmm/deviance.rs:149); anything else silently
    // falls back to Laplace and would compare nothing.
    let mut cells: Vec<&Value> = manifest["cells"]
        .as_array()
        .expect("cells")
        .iter()
        .filter(|c| {
            let family = c["family"].as_str().unwrap_or("");
            let re_q = c["re_q"].as_array().map(Vec::as_slice).unwrap_or(&[]);
            matches!(family, "binomial" | "poisson")
                && re_q.len() == 1
                && re_q[0].as_u64().unwrap_or(99) <= 3
        })
        .collect();
    // Smallest first: the short cells are where inner parallelism is most
    // likely to lose to its own overhead, so they are the ones worth seeing early.
    cells.sort_by_key(|c| {
        (
            c["n_obs"].as_u64().unwrap_or(0),
            c["re_q"][0].as_u64().unwrap_or(0),
            c["case_id"].as_str().unwrap_or("").to_string(),
        )
    });

    println!("{} AGQ-eligible cells, nagq {:?}\n", cells.len(), NAGQS);
    println!(
        "{:<34} {:>5} {:>2} {:>5} {:>9} {:>9} {:>7} {:>7}  bits",
        "case_id", "n", "q", "nagq", "serial_s", "par_s", "ratio", "n_eval"
    );

    for cell in cells {
        let case_id = cell["case_id"].as_str().unwrap();
        let n_obs = cell["n_obs"].as_u64().unwrap_or(0);
        let q = cell["re_q"][0].as_u64().unwrap_or(0);

        // Lowering is outside every timed region — it is CSV marshalling, not fit work.
        let Ok((mut lo, ..)) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lower_grid_cell(cell, DIR)))
        else {
            println!("{case_id:<34} {n_obs:>5} {q:>2}   lowering panicked");
            continue;
        };

        let mut spent = 0.0;
        for nagq in NAGQS {
            if spent > CELL_BUDGET_SECONDS {
                println!("{case_id:<34} {n_obs:>5} {q:>2} {nagq:>5}   skipped (cell over budget)");
                continue;
            }
            lo.opts.nagq = nagq;

            lo.opts.parallel_inner = false;
            let serial = one_fit(&lo);
            lo.opts.parallel_inner = true;
            let par = one_fit(&lo);
            spent += serial.1 + par.1;

            match (&serial.0, &par.0) {
                (Some(a), Some(b)) => {
                    let bits = if a.deviance.to_bits() == b.deviance.to_bits() {
                        "ok"
                    } else {
                        "DIFFER"
                    };
                    println!(
                        "{case_id:<34} {n_obs:>5} {q:>2} {nagq:>5} {:>9.4} {:>9.4} {:>7.2} {:>7}  {bits}",
                        serial.1,
                        par.1,
                        serial.1 / par.1,
                        a.n_eval,
                    );
                }
                _ => println!("{case_id:<34} {n_obs:>5} {q:>2} {nagq:>5}   fit panicked"),
            }
            std::io::stdout().flush().unwrap();
        }
    }
}

fn one_fit(lo: &glmm::formula::Lowered) -> (Option<glmm::Fit>, f64) {
    let t = Instant::now();
    let f = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts)
    }))
    .ok();
    (f, t.elapsed().as_secs_f64())
}
