//! Shared helpers for the dev-only examples that lower a manifest entry or a bare
//! CSV + formula pair into a `Lowered`: `bit_identity/dump.rs`, `memory/memory_fit.rs`,
//! and the speed campaign's `fit.rs` and `agq_par_probe.rs`.
//! Included via `#[path = ...]` rather than a library module — every example here is
//! dev-only, so this stays a plain shared source file rather than adding a crate for
//! four consumers. The CSV reader, `build_table`, `num`/`nums`, `rx_options` and
//! `lower_dataset_generic` are the accuracy grid's own, from
//! `grid/engines/common.rs`; this file adds the readers of the per-rung manifest
//! (`validation/manifest.json`) and of the speed campaign's cells.

use glmm::formula::{Lowered, Table};
use glmm::{BinomialLink, Family, GammaLink, NegBinomialLink, PoissonLink};
use serde_json::Value;

// Relative to this file's directory, whichever example includes it. The grid
// engine's own helpers (its record builder, cell loop) are unused here.
#[allow(dead_code)]
#[path = "../grid/engines/common.rs"]
mod grid_common;
#[allow(unused_imports)]
pub use grid_common::{
    build_table, lower_dataset_generic, num, nums, read_csv_path, rx_options, unquote,
};

/// Load a grid cell's CSV → lower. Returns the lowered inputs, whether the
/// family is Gaussian, and the cell's pre-registered eval cap. `data_dir` holds
/// the cell CSVs: the speed campaign's own (`prep.R`'s output,
/// `campaigns/speed-campaign/data/`) or another campaign's generated corpus.
/// The CSV is `<data_dir>/<data>.csv`, where `data` is the cell's optional
/// `data` field (several cells fitting one dataset, e.g. its REML and ML fits,
/// or its Laplace and AGQ fits) and defaults to `case_id`. Used by the grid
/// drivers (`grid_fit`, `agq_par_probe`) — allow(dead_code) because
/// `bit_identity` includes this file without calling it.
#[allow(dead_code)]
pub fn lower_grid_cell(cell: &Value, data_dir: &str) -> (Lowered, bool, usize) {
    let case_id = cell["case_id"].as_str().unwrap();
    let data_name = cell["data"].as_str().unwrap_or(case_id);
    let path = format!("{data_dir}/{data_name}.csv");
    let (header, rows) = read_csv_path(&path);
    let factors: Vec<String> = cell["factors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let jl = cell["jl_formula"].as_str().unwrap();
    let formula = jl
        .strip_prefix("@formula(")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap();
    let family = match cell["family"].as_str().unwrap() {
        "gaussian" => Family::Gaussian,
        "binomial" => Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
        "poisson" => Family::Poisson {
            link: glmm::PoissonLink::Log,
        },
        "negbin" => Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        "gamma" => Family::Gamma {
            link: GammaLink::Log,
        },
        other => panic!("family {other}"),
    };
    // aggregated binomial reuses the tier-0 lowering (weights/prop/expansion);
    // the grid driver does not time lowering, so the Table is dropped here.
    let (lo, _table) = lower_dataset_generic(
        cell,
        &header,
        &rows,
        &factors,
        formula,
        family,
        // The speed campaign's manifests name no response; their aggregated
        // binomial column is `incidence`, as `lower_rung` assumes.
        Some(cell["response"].as_str().unwrap_or("incidence")),
    );
    let max_fun = cell["max_fun"].as_u64().unwrap_or(0) as usize;
    (lo, matches!(family, Family::Gaussian), max_fun)
}

/// Load one manifest rung's CSV, build its formula string and `Family` from the manifest
/// fields, lower it, and apply `weights_col`/`offset` — the block
/// `bit_identity/dump.rs`'s `rung_record` needs before it can call `glmm::fit`.
/// `manifest_dir` is the `validation/` crate dir (`CARGO_MANIFEST_DIR`). Returns the
/// lowered inputs, the pre-typed `Table` (kept for a caller that re-lowers it for a
/// construction-inclusive timing), the resolved `Family`, and the formula string
/// lowering used (a re-lower needs the same string, not just the `Family`). Used by
/// `bit_identity` — allow(dead_code) because `grid_fit`/`memory_fit`/`agq_par_probe`
/// include this file without calling it (they lower grid cells or a bare CSV+formula
/// pair instead of a manifest rung).
#[allow(dead_code)]
pub fn lower_rung(spec: &Value, manifest_dir: &str) -> (Lowered, Table, Family, String) {
    let ds = spec["name"].as_str().expect("manifest entry missing name");
    let family_str = spec["family"]
        .as_str()
        .expect("manifest entry missing family");
    let factors: Vec<String> = spec["factors"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
        .unwrap_or_default();
    let source = if spec["source"].as_str() == Some("sim") {
        "simulated"
    } else {
        "empirical"
    };
    // `data` field: CSV to read when it differs from the rung name (a
    // validation/manifest.json field) — a re-linked rung (cbpp_probit) reuses the
    // committed dataset byte-for-byte instead of duplicating it.
    let data_name = spec["data"].as_str().unwrap_or(ds);
    let (header, rows) = read_csv_path(&format!("{manifest_dir}/data/{source}/{data_name}.csv"));

    // jl_formula, not r_formula: guaranteed cbind-free for every rung (design doc),
    // so it is already a safe generic lowering source. Strip the literal
    // "@formula(...)" wrapper to get a plain formula string. Rungs WITHOUT a
    // jl_formula (weights suite fixed-only / R-only rungs -- the field's absence
    // marks a rung with no Julia reference) fall back to r_formula; an aggregated-binomial
    // r_formula's `cbind(...)` response is rewritten to the `prop` column
    // lower_dataset_generic synthesizes (the same lowering validation/manifest.json's
    // jl_formula entries spell out by hand).
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
    // parser treats the intercept as always-implicit and
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
                Some("cloglog") => BinomialLink::Cloglog,
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

    // The rungs' aggregated-binomial response is always `incidence`; the per-rung
    // manifest carries no `response` field.
    let (mut lo, table) = lower_dataset_generic(
        spec,
        &header,
        &rows,
        &factors,
        &formula_str,
        family,
        Some("incidence"),
    );
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
        lo.weights = Some(rows.iter().map(|r| r[w_idx].parse().unwrap()).collect());
    }
    // `offset` field: per-row known additive term on the linear-predictor scale
    // (R's `offset=`) -- a named CSV column, mirroring `weights_col` above.
    if let Some(oc) = spec["offset"].as_str() {
        let o_idx = header
            .iter()
            .position(|h| h == oc)
            .unwrap_or_else(|| panic!("{ds}: offset {oc:?} not in CSV header"));
        lo.offset = Some(rows.iter().map(|r| r[o_idx].parse().unwrap()).collect());
    }

    (lo, table, family, formula_str)
}

/// Manifest's per-rung AGQ order (`spec["agq"]`), present only on rungs glmm's
/// AGQ gate accepts: a binomial, Poisson, negative-binomial or Gamma GLMM with a
/// single grouping factor and at most 3 random effects per group (`q ≤ 3`). `bit_identity/dump.rs` applies it unconditionally so a move in the
/// AGQ solve path (`src/glmm/agq.rs`) shows up in its report without a
/// separate env var.
#[allow(dead_code)]
pub fn rung_agq(spec: &Value) -> Option<u8> {
    spec["agq"].as_u64().map(|k| k as u8)
}
