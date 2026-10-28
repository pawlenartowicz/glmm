//! Machine-local "what moved" report: fits all 49 `manifest.json` rungs,
//! appends two more nAGQ=1 twin records (see `NAGQ1_TWIN_RUNGS` below), and
//! writes one JSON array, `{rung, config, deviance, theta, beta, se_hessian,
//! se_rx, n_eval, converged, singular}` per record (manifest rungs first in
//! manifest order, then the twins), to `tools/bit_identity/<config>.json`. Dump
//! before and after a change and diff the two files to see which rungs moved and
//! by how much; this is under a second, the full accuracy grid (`grid/run.sh`)
//! is minutes. It is a report, not a gate. The dumps are not committed
//! (`.gitignore`) and do not reproduce across CPU microarchitectures, so both
//! sides of a diff come from the same machine.
//!
//! `theta` is `Fit::tau2` (`theta[k]^2 * sigma_sq`, `sigma_sq` ≡ 1 off the
//! Gaussian path, declaration order), one value per θ coordinate; `Fit` has no
//! raw-θ accessor. (`Fit::stddev_se` shares that layout but is the SE of the
//! reported standard deviation, not a θ-scale quantity, and is not dumped.)
//! `se_hessian`/`se_rx` follow the cross-engine schema's own
//! convention: a gaussian rung's single SE lands in `se_hessian` (`se_rx`
//! null), a fixed-only GLM's lands in `se_rx` (`se_hessian` null, matching
//! that schema's labeling there), and a GLMM rung gets both, from two
//! separate `glmm::fit` calls (`WaldSe::Hessian` / `WaldSe::Rx`).
//!
//! `nagq` is 1 (Laplace) except on the manifest's `agq`-marked rungs, which
//! fit at the manifest's stated order (`harness_common::rung_agq`), so a move
//! in `src/glmm/agq.rs` shows up here without an opt-in env var. Two of those
//! rungs (26, 44) also get a twin record fit at nAGQ=1 instead (see
//! `NAGQ1_TWIN_RUNGS`).
//!
//! `config` is a required CLI arg, used only to label the JSON `config` field
//! and the output filename; the crate feature set that actually produced the
//! binary comes from how it was built (see the commands below), not from
//! anything this binary reads at runtime.
//!
//! `validation/Cargo.toml` enables `formula` on the `glmm` dependency, so a
//! `-p validation` build always has it. Every run is `--release`, like the
//! builds whose results it reports (debug is also needlessly slow, dominated by
//! the double-fit GLMM/grouseticks rungs). The configs:
//!
//!   default        cargo run --release -p validation --example bit_identity -- default
//!   parallel       cargo run --release -p validation --example bit_identity --features glmm/parallel -- parallel
//!   loop_advanced  not dumped: the feature only adds re-exports, so its dump
//!                  would be the default one.
//!   no-default-features  not reachable from this crate: `tools/common.rs`
//!                  and this file both call `glmm::formula::lower`, so
//!                  dropping `formula` breaks the build the harness needs
//!                  to read the manifest at all.

use glmm::FitOptions;
use serde_json::Value;

#[path = "../common.rs"]
mod harness_common;
use harness_common::*;

const DIR: &str = env!("CARGO_MANIFEST_DIR");

/// Manifest rungs that get a SECOND record fitted at nAGQ=1. Their manifest
/// `agq` order forces `nagq > 1`, so `exact_profile_shape`'s first clause
/// (`src/glmm/mod.rs`) rejects them and they route the joint search; without a
/// twin the dump has no canonical-link (logit / Poisson-log) single-grouping
/// record on the exact-profile route, and no exact-profile record at all with
/// a random slope (rung 26, `q = 2`).
/// Twin `rung` id = 1000 + the manifest rung: two dumps are compared by keying
/// records on `rung`, so a twin needs an id of its own, and 1000+ keeps it
/// unmistakable for a manifest rung.
const NAGQ1_TWIN_RUNGS: [u64; 2] = [26, 44];
const TWIN_RUNG_OFFSET: u64 = 1000;

fn main() {
    let config = std::env::args()
        .nth(1)
        .expect("usage: bit_identity <config-name>");
    let manifest: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{DIR}/manifest.json")).expect("read manifest"),
    )
    .expect("parse manifest");

    let mut out = String::from("[\n");
    let datasets = manifest["datasets"].as_array().expect("manifest.datasets");
    for (i, spec) in datasets.iter().enumerate() {
        if i > 0 {
            out.push_str(",\n");
        }
        let rung = spec["rung"].as_u64().expect("manifest entry missing rung");
        out.push_str(&rung_record(spec, &config, rung, rung_agq(spec)));
    }
    for spec in datasets {
        let rung = spec["rung"].as_u64().expect("manifest entry missing rung");
        if NAGQ1_TWIN_RUNGS.contains(&rung) {
            out.push_str(",\n");
            out.push_str(&rung_record(
                spec,
                &config,
                TWIN_RUNG_OFFSET + rung,
                Some(1),
            ));
        }
    }
    out.push_str("\n]\n");

    std::fs::create_dir_all(format!("{DIR}/tools/bit_identity")).expect("mk bit_identity");
    let path = format!("{DIR}/tools/bit_identity/{config}.json");
    std::fs::write(&path, &out).expect("write dump");
    println!(
        "bit_identity  config={config}  rungs={}  -> {path}",
        datasets.len() + NAGQ1_TWIN_RUNGS.len()
    );
}

/// One record for `rung` (a manifest rung id, or a twin's `TWIN_RUNG_OFFSET`-shifted
/// id), field order fixed by hand (not `serde_json::Map`'s default alphabetical
/// order) to match the spec's `{rung, config, deviance, theta, beta, se_hessian,
/// se_rx, n_eval, converged, singular}`. `nagq` is the caller's choice, not
/// derived from `spec`, so a twin can reuse its parent's `spec` at a different
/// AGQ order.
fn rung_record(spec: &Value, config: &str, rung: u64, nagq: Option<u8>) -> String {
    let family_str = spec["family"]
        .as_str()
        .expect("manifest entry missing family");
    let gaussian = family_str == "gaussian";

    // Data load, formula/family resolution, lowering, and weights_col/offset
    // come from `lower_rung` (common.rs) — one lowering, not a re-derivation.
    let (lo, _table, _family, _formula_str) = lower_rung(spec, DIR);
    let mut fit_opts = FitOptions::default();
    // A no-op on a serial build (`FitOptions::parallel_inner` is only live
    // under the `parallel` Cargo feature; see its doc), so this line is safe
    // to run unconditionally in both configs.
    fit_opts.parallel_inner = config == "parallel";
    // The dump applies the manifest's per-rung AGQ order unconditionally, so a
    // move in `src/glmm/agq.rs` shows up in it.
    if let Some(k) = nagq {
        fit_opts.nagq = k;
    }

    let fixed_only = lo.re_groups.is_empty();
    let (deviance, theta, beta, se_hessian, se_rx, n_eval, converged, singular) =
        if gaussian || fixed_only {
            // One fit: Gaussian reports its SE as Hessian-side, fixed-only as Rx-side.
            let f = glmm::fit(&lo.data(), lo.spec(), &fit_opts).unwrap();
            (
                f.deviance,
                f.tau2.clone(),
                f.beta.clone(),
                gaussian.then(|| f.se.clone()),
                (!gaussian).then(|| f.se.clone()),
                f.n_eval,
                f.converged(),
                f.singular(),
            )
        } else {
            let o_r = rx_options(&fit_opts);
            let fh = glmm::fit(&lo.data(), lo.spec(), &fit_opts).unwrap();
            let fr = glmm::fit(&lo.data(), lo.spec(), &o_r).unwrap();
            (
                fh.deviance,
                fh.tau2.clone(),
                fh.beta.clone(),
                Some(fh.se.clone()),
                Some(fr.se.clone()),
                fh.n_eval + fr.n_eval,
                fh.converged() && fr.converged(),
                fh.singular(),
            )
        };

    let se_hessian_json = se_hessian.map(|v| nums(&v)).unwrap_or(Value::Null);
    let se_rx_json = se_rx.map(|v| nums(&v)).unwrap_or(Value::Null);
    format!(
        "  {{\n    \"rung\": {rung},\n    \"config\": {config_json},\n    \"deviance\": {deviance_json},\n    \"theta\": {theta_json},\n    \"beta\": {beta_json},\n    \"se_hessian\": {se_hessian_json},\n    \"se_rx\": {se_rx_json},\n    \"n_eval\": {n_eval},\n    \"converged\": {converged},\n    \"singular\": {singular}\n  }}",
        config_json = serde_json::to_string(config).unwrap(),
        deviance_json = serde_json::to_string(&num(deviance)).unwrap(),
        theta_json = serde_json::to_string(&nums(&theta)).unwrap(),
        beta_json = serde_json::to_string(&nums(&beta)).unwrap(),
        se_hessian_json = serde_json::to_string(&se_hessian_json).unwrap(),
        se_rx_json = serde_json::to_string(&se_rx_json).unwrap(),
    )
}
