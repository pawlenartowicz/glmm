//! Shared helpers between the glmm validation harnesses: `glmm.rs` (the frozen
//! per-dataset corpus, `validation_fit` example), `grid_fit.rs` (the
//! optimizer-grid campaign, `grid_fit` example), and `bit_identity/dump.rs` (the
//! bit-identity dump, `bit_identity` example). Included via `#[path = ...]`
//! rather than a library module — every example here is dev-only, so this stays
//! a plain shared source file rather than adding a crate for multiple consumers.

use glmm::formula::{lower, Column, Lowered, Table};
use glmm::{BinomialLink, Family, FitOptions, GammaLink, NegBinomialLink, PoissonLink, WaldSe};
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
/// `f64` anywhere (e.g. Pastes' `sample`, a categorical helper column the validation
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
/// writes valid JSON the comparators read as "missing", not a crash. Used by
/// every driver that emits comparison JSON — allow(dead_code) because
/// `memory_fit` includes this file without calling either (it prints one
/// status line, never a result record).
#[allow(dead_code)]
pub fn num(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}
#[allow(dead_code)]
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

/// Load one manifest rung's CSV, build its formula string and `Family` from
/// the manifest fields, lower it, and apply `weights_col`/`offset` — the block
/// `engines/glmm.rs`'s `fit_one` and `bit_identity/dump.rs`'s `rung_record`
/// both need before they can call `fit_cold`. `manifest_dir` is the
/// `validation/` crate dir (`CARGO_MANIFEST_DIR`), matching `lower_grid_cell`'s
/// convention above. Returns the lowered inputs, the pre-typed `Table` (kept
/// for a caller that re-lowers it, e.g. `glmm.rs`'s construction-inclusive
/// timing), the resolved `Family`, and the formula string lowering used (a
/// re-lower needs the same string, not just the `Family`). Used by
/// `validation_fit` and `bit_identity` — allow(dead_code) because
/// `grid_fit`/`theta_eval`/`memory_fit`/`agq_par_probe` include this file
/// without calling it (they lower grid cells or a bare CSV+formula pair
/// instead of a manifest rung).
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
    // `data` field: CSV to read when it differs from the rung name (mirrors
    // lme4.R/mixedmodels.jl) — a re-linked rung (cbpp_probit) reuses the
    // committed dataset byte-for-byte instead of duplicating it.
    let data_name = spec["data"].as_str().unwrap_or(ds);
    let (header, rows) = read_csv_path(&format!("{manifest_dir}/data/{source}/{data_name}.csv"));

    // jl_formula, not r_formula: guaranteed cbind-free for every rung (design doc),
    // so it is already a safe generic lowering source. Strip the literal
    // "@formula(...)" wrapper to get a plain formula string. Rungs WITHOUT a
    // jl_formula (weights suite fixed-only / R-only rungs -- the field's absence
    // is mixedmodels.jl's skip signal) fall back to r_formula; an aggregated-binomial
    // r_formula's `cbind(...)` response is rewritten to the `prop` column
    // lower_dataset_generic synthesizes (the same lowering mixedmodels.jl's manifest
    // entries spell out by hand).
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
    // parser (mirroring MCPower's) treats the intercept as always-implicit and
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

    let (mut lo, table) =
        lower_dataset_generic(spec, &header, &rows, &factors, &formula_str, family);
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
        lo.opts.weights = Some(rows.iter().map(|r| r[w_idx].parse().unwrap()).collect());
    }
    // `offset` field: per-row known additive term on the linear-predictor scale
    // (R's `offset=`) -- a named CSV column, mirroring `weights_col` above.
    if let Some(oc) = spec["offset"].as_str() {
        let o_idx = header
            .iter()
            .position(|h| h == oc)
            .unwrap_or_else(|| panic!("{ds}: offset {oc:?} not in CSV header"));
        lo.opts.offset = Some(rows.iter().map(|r| r[o_idx].parse().unwrap()).collect());
    }

    (lo, table, family, formula_str)
}

/// The `WaldSe::Rx` twin of a rung's fit options, for the GLMM SE
/// comparison both `engines/glmm.rs` and `bit_identity/dump.rs` emit. Every
/// field that shapes the fit is carried over, NOT defaulted: `nagq` is 1 on a
/// normal run, but on an AGQ pass a defaulted twin would silently run the Rx
/// arm at Laplace while the Hessian arm ran quadrature.
#[allow(dead_code)]
pub fn rx_options(opts: &FitOptions) -> FitOptions {
    FitOptions {
        target_indices: opts.target_indices.clone(),
        wald_se: WaldSe::Rx,
        weights: opts.weights.clone(),
        offset: opts.offset.clone(),
        nagq: opts.nagq,
        parallel_inner: opts.parallel_inner,
        ..FitOptions::default()
    }
}

/// Manifest's per-rung AGQ order (`spec["agq"]`), present only on the
/// binomial/Poisson single-grouping-factor (q ≤ 3) rungs glmm's AGQ gate
/// accepts. `engines/glmm.rs`'s `fit_one` applies this to `lo.opts.nagq` only
/// during an opt-in `VALIDATION_AGQ` timing pass; `bit_identity/dump.rs`
/// applies it unconditionally so the AGQ solve path (`src/glmm/agq.rs`) stays
/// on the bit-identity tripwire without a separate env var.
#[allow(dead_code)]
pub fn rung_agq(spec: &Value) -> Option<u8> {
    spec["agq"].as_u64().map(|k| k as u8)
}
