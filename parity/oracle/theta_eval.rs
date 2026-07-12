//! Mismatch-adjudication driver (oracle spec 2026-07-11): drives the dev-only
//! `loop_advanced` seam over grid cells. Reads request JSONL from
//! `$THETA_EVAL_IN`, appends one result line per request to `$THETA_EVAL_OUT`
//! (stdout when unset). Requests:
//!
//! - `{"case_id", "action": "eval",  "theta": [...]}` — objective at fixed θ
//!   (glmm vech layout: primary column-major lower triangle, then extras in
//!   declaration order).
//! - `{"case_id", "action": "replay"}` — shipped-config fit through the seam
//!   (blind start, shipped rho/npt/cap); must reproduce the campaign record.
//! - `{"case_id", "action": "sweep", "theta0": [...], "rho_end": f,
//!    "max_fun": n}` — custom-schedule fit from a verbatim θ₀.
//!
//! Any fit action takes optional `"traj_out": path` — per-eval JSONL
//! `{k, f, theta}` trajectory log — and an optional `"label"` echoed back.
use std::io::Write;

use glmm::loop_advanced::{lmm_objective_at, lmm_sweep_fit};
use serde_json::{json, Value};

#[path = "harness_common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

fn main() {
    let in_path = std::env::var("THETA_EVAL_IN").expect("THETA_EVAL_IN request file");
    let manifest_path = std::env::var("GRID_MANIFEST")
        .unwrap_or_else(|_| format!("{DIR}/parity/manifest_grid.json"));
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("read grid manifest"))
            .expect("parse grid manifest");
    let mut out: Box<dyn Write> = match std::env::var("THETA_EVAL_OUT") {
        Ok(p) => Box::new(
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .unwrap(),
        ),
        Err(_) => Box::new(std::io::stdout()),
    };

    for line in std::fs::read_to_string(&in_path).expect("read requests").lines() {
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = serde_json::from_str(line).expect("parse request");
        let case_id = req["case_id"].as_str().expect("case_id");
        let cell = manifest["cells"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["case_id"] == case_id)
            .unwrap_or_else(|| panic!("case {case_id} not in manifest"));
        let (lo, gaussian, _) = lower_grid_cell(cell, DIR);
        assert!(gaussian, "adjudication targets are Gaussian LMM cells");
        let action = req["action"].as_str().expect("action");
        let mut rec = json!({
            "case_id": case_id, "action": action, "label": req["label"],
        });
        match action {
            "eval" => {
                let theta = floats(&req["theta"]);
                let dev = lmm_objective_at(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &theta);
                rec["dev"] = num(dev);
            }
            "replay" | "sweep" => {
                let theta0 = (action == "sweep").then(|| floats(&req["theta0"]));
                let rho_end = req["rho_end"].as_f64().unwrap_or(1e-6);
                let max_fun = req["max_fun"].as_u64().map(|v| v as usize);
                // Per-eval trajectory log, line-buffered via a BufWriter the
                // trace closure borrows.
                let mut traj = req["traj_out"].as_str().map(|p| {
                    std::io::BufWriter::new(std::fs::File::create(p).expect("create traj_out"))
                });
                let mut trace = |k: usize, xs: &[f64], f: f64| {
                    if let Some(w) = traj.as_mut() {
                        writeln!(w, "{}", json!({"k": k, "f": num(f), "theta": nums(xs)}))
                            .unwrap();
                    }
                };
                let outc = lmm_sweep_fit(
                    &lo.x,
                    &lo.y,
                    lo.n,
                    lo.p,
                    &lo.model,
                    &lo.ids,
                    theta0.as_deref(),
                    rho_end,
                    max_fun,
                    Some(&mut trace),
                );
                rec["dev"] = num(outc.deviance);
                rec["theta"] = nums(&outc.theta);
                rec["n_eval"] = json!(outc.n_eval);
                rec["converged"] = json!(outc.converged);
            }
            other => panic!("unknown action {other}"),
        }
        writeln!(out, "{}", serde_json::to_string(&rec).unwrap()).unwrap();
        out.flush().unwrap();
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("theta array")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
