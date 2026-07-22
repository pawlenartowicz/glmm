//! Shared helpers between the two glmm validation harnesses: `glmm.rs` (the frozen
//! per-dataset corpus, `validation_fit` example) and `grid_fit.rs` (the
//! optimizer-grid campaign, `grid_fit` example). Included via `#[path = ...]`
//! rather than a library module — both examples are dev-only, so this stays a
//! plain shared source file rather than adding a crate for two consumers.

use glmm::formula::{lower, Column, Lowered, Table};
use glmm::Family;
use serde_json::{json, Value};

/// Read a validation CSV at an explicit path (unquoted header + rows, `,`-split).
/// The dataset-name+source variant (`glmm.rs`'s `read_csv`) delegates here —
/// the grid corpus has no empirical/simulated split, only a flat `grid/` dir
/// keyed by `case_id`, so the path is the only thing that varies.
pub fn read_csv_path(path: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read csv {path}: {e}"));
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().unwrap().split(',').map(unquote).collect();
    let rows = lines.map(|l| l.split(',').map(unquote).collect()).collect();
    (header, rows)
}

pub fn unquote(s: &str) -> String {
    s.trim().trim_matches('"').to_string()
}

/// A `Table` built by column NAME: manifest `factors` become `Column::Factor`,
/// everything else `Column::Numeric` — except a column that fails to parse as
/// `f64` anywhere (e.g. Pastes' `cask`, a categorical helper column the validation
/// corpus carries but no jl_formula references) falls back to `Column::Factor`
/// rather than panicking, since it may be present in the CSV without being
/// referenced by the formula at all.
pub fn build_table(header: &[String], rows: &[Vec<String>], factors: &[String]) -> Table {
    let n = rows.len();
    let columns = header
        .iter()
        .enumerate()
        .map(|(j, name)| {
            let is_factor = factors.iter().any(|f| f == name)
                || rows.iter().any(|r| r[j].parse::<f64>().is_err());
            let col = if is_factor {
                // A CSV column carries no declared level order, so the
                // lexicographic default is the whole of what the oracle can
                // know — and it is what R's own `factor()` did on the reference
                // side when the golden was frozen.
                let labels: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                Column::factor_from_labels(&labels)
            } else {
                Column::Numeric(rows.iter().map(|r| r[j].parse().unwrap()).collect())
            };
            // Mirrors mixedmodels.jl's read_dataset: R-origin CSV headers can carry dots
            // (Arabidopsis's `total.fruits`) that jl_formula sanitizes to
            // underscores because Julia's @formula reader can't parse a dot as
            // part of an identifier. Rename here so the Table's column names
            // match jl_formula's (already-underscored) reference.
            (name.replace('.', "_"), col)
        })
        .collect();
    Table { columns, n }
}

/// NaN/Inf → JSON null (serde_json cannot serialize non-finite floats, and a
/// non-converged fit leaves NaN-filled estimates) so an unconverged run still
/// writes valid JSON the comparators read as "missing", not a crash.
pub fn num(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}
pub fn nums(xs: &[f64]) -> Value {
    Value::Array(xs.iter().map(|&x| num(x)).collect())
}

/// Load a grid cell's CSV → lower. Returns the lowered inputs, whether the
/// family is Gaussian, and the cell's pre-registered eval cap. `manifest_dir`
/// is the `validation/` crate dir (`CARGO_MANIFEST_DIR`, shared by every bin
/// target regardless of which campaign subdir the source file sits in); grid
/// data is campaign-local (`prep.R`'s output), under
/// `campaigns/speed-grid/data/`. Used by the grid drivers (`grid_fit`,
/// `theta_eval`) — allow(dead_code) because `validation_fit` includes this file
/// without calling it.
#[allow(dead_code)]
pub fn lower_grid_cell(cell: &Value, manifest_dir: &str) -> (Lowered, bool, usize) {
    let case_id = cell["case_id"].as_str().unwrap();
    let path = format!("{manifest_dir}/campaigns/speed-grid/data/{case_id}.csv");
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
        .unwrap()
        .replacen(" ~ 1 + ", " ~ ", 1);
    let family = match cell["family"].as_str().unwrap() {
        "gaussian" => Family::Gaussian,
        "binomial" => Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
        "poisson" => Family::Poisson {
            link: glmm::PoissonLink::Log,
        },
        other => panic!("family {other}"),
    };
    // aggregated binomial reuses the tier-0 lowering (weights/prop/expansion);
    // the grid driver does not time lowering, so the Table is dropped here.
    let (lo, _table) = lower_dataset_generic(cell, &header, &rows, &factors, &formula, family);
    let max_fun = cell["max_fun"].as_u64().unwrap_or(0) as usize;
    (lo, matches!(family, Family::Gaussian), max_fun)
}

/// Build the lowered fit inputs for one manifest entry (dataset or grid cell).
/// Handles the one genuinely data-shape-dependent branch in the corpus: an
/// aggregated-binomial rung (manifest `weights`) synthesizes `prop =
/// incidence/weights_col` (mirrors mixedmodels.jl:44-46) so jl_formula's `prop ~ ...`
/// response resolves, then passes `Some(sizes)` into `FitOptions::weights`,
/// one row per aggregate observation — every RE shape now honors prior
/// weights (`src/fit.rs`'s boundary assert only rejects nAGQ>1 with weights),
/// so there is no more dense-vs-sparse split or per-trial Bernoulli expansion
/// to keep the two argmins in agreement.
///
/// Returns the built `Table` alongside the `Lowered`, so a caller that wants to
/// TIME the lowering (glmm.rs's construction-inclusive timing, matching how lme4 /
/// MixedModels / the Python port measure `formula+data -> model -> fit`) can
/// re-run `lower(formula, &table, family)` in a loop without re-parsing the CSV.
/// The Table is the typed-columns analogue of an lme4/Julia DataFrame — built
/// once, then lowered repeatedly.
pub fn lower_dataset_generic(
    spec: &Value,
    header: &[String],
    rows: &[Vec<String>],
    factors: &[String],
    formula_str: &str,
    family: Family,
) -> (Lowered, Table) {
    let Some(w_name) = spec["weights"].as_str() else {
        let table = build_table(header, rows, factors);
        let lo = lower(formula_str, &table, family).unwrap_or_else(|e| panic!("lower: {e}"));
        return (lo, table);
    };

    let w_idx = header
        .iter()
        .position(|h| h == w_name)
        .expect("weights column in header");
    let inc_idx = header
        .iter()
        .position(|h| h == "incidence")
        .expect("incidence column in header");
    let sizes: Vec<f64> = rows.iter().map(|r| r[w_idx].parse().unwrap()).collect();
    let incid: Vec<f64> = rows.iter().map(|r| r[inc_idx].parse().unwrap()).collect();
    let prop: Vec<f64> = incid.iter().zip(&sizes).map(|(i, s)| i / s).collect();

    let mut table = build_table(header, rows, factors);
    table.columns.push(("prop".into(), Column::Numeric(prop)));
    let mut lo = lower(formula_str, &table, family).unwrap_or_else(|e| panic!("lower: {e}"));
    lo.opts.weights = Some(sizes);
    (lo, table)
}
