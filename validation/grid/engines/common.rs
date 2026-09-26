//! Shared helpers for the glmm side of the accuracy grid (`grid/manifest.json`).
//! Included via `#[path = ...]` rather than a library module, by two callers:
//! `glmm.rs`, the grid's dev-only example engine, and `tests/grid_reference.rs`,
//! the crate's saved-reference check over the `fast` cells. Sharing this file is
//! what keeps both callers' record shape identical.
//!
//! `validation/tools/common.rs` is a SEPARATE file that reads the per-rung
//! manifest shape of `validation/manifest.json` and is the `#[path]` target of the
//! examples built on it. Neither file is a drop-in for the other.

use glmm::formula::{lower, Column, Lowered, ReGroupInfo, Table};
use glmm::{
    fit_cold, BinomialLink, Family, Fit, FitOptions, GammaLink, InverseGaussianLink,
    NegBinomialLink, PoissonLink, WaldSe,
};
use serde_json::{json, Value};
use std::time::Instant;

/// Read a grid CSV at an explicit path (unquoted header + rows, `,`-split).
/// Every cell names its own file through the manifest's `data` field, so the
/// path is the only thing that varies between a generated cell and a crate
/// fixture.
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
/// `f64` anywhere (e.g. Pastes' `sample`, a categorical helper column the CSV
/// carries but no formula references) falls back to `Column::Factor` rather
/// than panicking, since it may be present in the CSV without being referenced
/// by the formula at all.
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
                // lexicographic default is the whole of what the harness can
                // know — and it is what R's own `factor()` did on the reference
                // side.
                let labels: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
                Column::factor_from_labels(&labels)
            } else {
                Column::Numeric(rows.iter().map(|r| r[j].parse().unwrap()).collect())
            };
            // R-origin CSV headers can carry dots (Arabidopsis's `total.fruits`)
            // that the manifest's formulas sanitize to underscores, because
            // Julia's @formula reader cannot parse a dot as part of an
            // identifier. Rename here so the Table's column names match the
            // (already-underscored) formula strings.
            (name.replace('.', "_"), col)
        })
        .collect();
    Table { columns, n }
}

/// NaN/Inf → JSON null (serde_json cannot serialize non-finite floats, and a
/// non-converged fit leaves NaN-filled estimates) so an unconverged run still
/// writes valid JSON the comparator reads as "missing", not a crash.
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

/// Load one grid cell's CSV and lower it. `grid_dir` is the `grid/` directory;
/// the cell's `data` field is a path RELATIVE TO IT, so a generated cell
/// (`data/<id>.csv`) and a crate fixture (`../data/simulated/<name>.csv`) are
/// read by the same line and the manifest is the only place that decides which.
/// Returns the lowered inputs and whether the family is Gaussian.
pub fn lower_grid_cell(cell: &Value, grid_dir: &str) -> (Lowered, bool) {
    let path = format!("{grid_dir}/{}", cell["data"].as_str().expect("cell.data"));
    let (header, rows) = read_csv_path(&path);
    let factors: Vec<String> = cell["factors"]
        .as_array()
        .map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect())
        .unwrap_or_default();
    // `glmm_formula`, NOT jl_formula: the manifest generator already emitted the
    // crate's dialect -- the "@formula(...)" wrapper and the leading "1 + "
    // removed, and crucially NO offset(...) term, because the offset arrives
    // through opts.offset below. Lowering jl_formula here instead would apply
    // the offset twice on every offset cell. No `&`-to-`:` rewrite is needed
    // either: the generated cells never use `&`, and the read-in cells had
    // theirs rewritten when `glmm_formula` was built.
    let formula = cell["glmm_formula"].as_str().expect("cell.glmm_formula");
    let family = family_of(cell);
    let (mut lo, _table) = lower_dataset_generic(cell, &header, &rows, &factors, formula, family);
    // Prior (precision) weights. Mutually exclusive with the aggregated-binomial
    // `weights` field lower_dataset_generic already routed, and the manifest
    // generator asserts the exclusion, so this cannot overwrite trial counts.
    if let Some(wc) = cell["weights_col"].as_str() {
        let j = header
            .iter()
            .position(|h| h == wc)
            .expect("weights_col in header");
        lo.opts.weights = Some(rows.iter().map(|r| r[j].parse().unwrap()).collect());
    }
    // The offset, applied HERE and only here for this engine.
    if let Some(oc) = cell["offset_col"].as_str() {
        let j = header
            .iter()
            .position(|h| h == oc)
            .expect("offset_col in header");
        lo.opts.offset = Some(rows.iter().map(|r| r[j].parse().unwrap()).collect());
    }
    if let Some(k) = cell["nagq"].as_u64() {
        lo.opts.nagq = k as u8;
    }
    (lo, matches!(family, Family::Gaussian))
}

/// The grid manifest's `family` + `link` pair -> `glmm::Family`. `inversegaussian`
/// appears only on the two GLM fixture cells, where the kernel's fixed-effects
/// GLM path handles it (its mixed path is not built).
pub fn family_of(cell: &Value) -> Family {
    let fam = cell["family"].as_str().expect("cell.family");
    let link = cell["link"].as_str().expect("cell.link");
    match (fam, link) {
        ("gaussian", "identity") => Family::Gaussian,
        ("binomial", "logit") => Family::Binomial {
            link: BinomialLink::Logit,
        },
        ("binomial", "probit") => Family::Binomial {
            link: BinomialLink::Probit,
        },
        ("binomial", "cloglog") => Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        ("poisson", "log") => Family::Poisson {
            link: PoissonLink::Log,
        },
        ("gamma", "log") => Family::Gamma {
            link: GammaLink::Log,
        },
        ("gamma", "inverse") => Family::Gamma {
            link: GammaLink::Inverse,
        },
        ("negativebinomial", "log") => Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        ("inversegaussian", "log") => Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        },
        ("inversegaussian", "inverse_squared") => Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        },
        other => panic!("unsupported family/link pair: {other:?}"),
    }
}

/// Build the lowered fit inputs for one cell, handling the one genuinely
/// data-shape-dependent branch: an aggregated-binomial cell (manifest `weights`)
/// synthesizes `prop = <response>/<weights column>` so the formula's `prop ~ ...`
/// response resolves, then passes the trial counts into `FitOptions::weights`,
/// one row per aggregate observation -- exactly lme4's `cbind(s, m-s)`
/// objective, whose deviance differs from the expanded-Bernoulli one only by a
/// data-only saturated constant.
pub fn lower_dataset_generic(
    cell: &Value,
    header: &[String],
    rows: &[Vec<String>],
    factors: &[String],
    formula_str: &str,
    family: Family,
) -> (Lowered, Table) {
    let Some(w_name) = cell["weights"].as_str() else {
        let table = build_table(header, rows, factors);
        let lo = lower(formula_str, &table, family).unwrap_or_else(|e| panic!("lower: {e}"));
        return (lo, table);
    };
    let resp = cell["response"].as_str().expect("cell.response");
    let w_idx = header
        .iter()
        .position(|h| h == w_name)
        .expect("weights column in header");
    // Compared with the dot->underscore rename applied to both sides, so the
    // response matches whether the manifest spells it the CSV's way
    // (`total.fruits`) or the formula's way (`total_fruits`).
    let r_name = resp.replace('.', "_");
    let r_idx = header
        .iter()
        .position(|h| h.replace('.', "_") == r_name)
        .expect("response column in header");
    let sizes: Vec<f64> = rows.iter().map(|r| r[w_idx].parse().unwrap()).collect();
    let succ: Vec<f64> = rows.iter().map(|r| r[r_idx].parse().unwrap()).collect();
    let prop: Vec<f64> = succ.iter().zip(&sizes).map(|(i, s)| i / s).collect();
    let mut table = build_table(header, rows, factors);
    table.columns.push(("prop".into(), Column::Numeric(prop)));
    let mut lo = lower(formula_str, &table, family).unwrap_or_else(|e| panic!("lower: {e}"));
    lo.opts.weights = Some(sizes);
    (lo, table)
}

/// The `WaldSe::Rx` twin of a cell's fit options. Every field that shapes the fit
/// is carried over, NOT defaulted: `nagq` especially, because a defaulted twin
/// would silently run the Rx arm at Laplace while the Hessian arm ran
/// quadrature.
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

/// Cell ids already present in `path`, so a killed worker resumes where it
/// stopped. A `kill -9` can truncate the final line -- a line that does not
/// parse is skipped, never fatal.
pub fn done_cells(path: &str) -> std::collections::HashSet<String> {
    let Ok(s) = std::fs::read_to_string(path) else {
        return Default::default();
    };
    s.lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|v| v["cell"].as_str().map(str::to_string))
        .collect()
}

/// The comparison record, pre-seeded with every key at its "missing" value, so
/// a fit that failed early still emits the full field set instead of a short
/// record the comparator would have to guess about. `se_rx` and `se_hessian`
/// are NOT seeded: they are the two keys that are legitimately ABSENT where an
/// engine has none, so the caller inserts them only when it has them.
pub fn base_record(cell_id: &str, engine: &str, engine_version: &str) -> Value {
    json!({
        "cell": cell_id, "engine": engine, "engine_version": engine_version,
        "converged": false, "singular": false, "status": "engine-fail",
        "message": Value::Null, "coef_names": Value::Array(vec![]),
        "beta": Value::Array(vec![]), "varcomp": Value::Array(vec![]),
        "sigma": Value::Null, "nb_theta": Value::Null,
        "loglik": Value::Null, "deviance": Value::Null, "n_eval": Value::Null,
        "wall_seconds": Value::Null, "fits_per_sample": 1
    })
}

/// Set `loglik` AND the `deviance` that must agree with it, in one call, so no
/// engine can set one without the other. `deviance = -2 * loglik`, null when
/// `loglik` is non-finite. This is the only place either field is written.
pub fn set_loglik(rec: &mut Value, loglik: f64) {
    rec["loglik"] = num(loglik);
    rec["deviance"] = num(-2.0 * loglik);
}

/// Fit one grid cell and build its comparison record. Non-gaussian cells are
/// fitted TWICE, once under each Wald-SE method, because the two methods answer
/// different questions: se_hessian keeps the theta-beta coupling, se_rx is
/// conditional on theta-hat, and the two sit ~1-1.5% apart on the same fit, so a
/// comparison that crosses them reads that spread as an engine disagreement.
/// Gaussian cells have one profiled SE and are fitted once, into the `se_rx` slot
/// with `se_hessian` absent.
///
/// `engine_version` is the version string stamped into the record. It is a
/// parameter and not `env!("CARGO_PKG_VERSION")` because this function is
/// compiled into two different crates -- the grid engine example and the crate's
/// own reference check -- and the macro would report a different package in each.
///
/// Panics are caught per cell and recorded as status "engine-fail" with the
/// panic text in `message`: grid corners are expected to break engines, and a
/// crash is a data point, not a reason to lose the other 800 cells.
///
/// NOTHING HERE RESCALES ANY ESTIMATE. `varcomp` is `Fit::stddev_corr` verbatim.
/// The crate's convention is what the truth gate is measuring; a correction
/// applied here would hide it.
///
/// There is no budget parameter: the harness watchdog is the only cell cap, and
/// it works by killing the process from outside.
pub fn fit_cell(cell: &Value, grid_dir: &str, timed: Option<usize>, engine_version: &str) -> Value {
    let cell_id = cell["cell"].as_str().expect("cell.cell");
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut rec = base_record(cell_id, "glmm", engine_version);
        let (lo, gaussian) = lower_grid_cell(cell, grid_dir);
        let family = family_of(cell);
        let fit = |o: &FitOptions| fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, o);

        // The shipped configuration (`WaldSe::Hessian`) is the primary fit: its
        // estimates are the ones reported, and it is the one the timing loop
        // below re-runs. On a gaussian cell `wald_se` is not read at all -- the
        // LMM path has a single profiled SE -- so the same fit serves as the
        // `se_rx` value.
        let f = fit(&lo.opts);
        if gaussian {
            rec["se_rx"] = nums(&f.se);
        } else {
            rec["se_hessian"] = nums(&f.se);
            rec["se_rx"] = nums(&fit(&rx_options(&lo.opts)).se);
        }

        // Lowering stays outside the timed region: the reference engines are
        // already holding a typed data frame when their own timer starts.
        if let Some(n) = timed {
            let mut samples = Vec::with_capacity(n);
            for _ in 0..n {
                let t0 = Instant::now();
                let _ = fit(&lo.opts);
                samples.push(t0.elapsed().as_secs_f64());
            }
            rec["wall_seconds"] = num(median(&samples[1..]));
        }

        rec["coef_names"] = json!(lo.col_names);
        rec["beta"] = nums(&f.beta);
        // `Fit::dispersion` is σ̂² on gaussian and φ̂ on gamma, while the
        // schema's `sigma` is a standard deviation / scale, so the square root
        // is what makes the two families one convention. It is the estimated
        // shape on negative-binomial, which belongs in `nb_theta` and nowhere
        // else, and a fixed 1.0 on the remaining families, which report null.
        rec["sigma"] = match family {
            Family::Gaussian | Family::Gamma { .. } => num(f.dispersion.sqrt()),
            _ => Value::Null,
        };
        rec["nb_theta"] = match family {
            Family::NegativeBinomial { .. } => num(f.dispersion),
            _ => Value::Null,
        };
        set_loglik(&mut rec, f.loglik);
        rec["n_eval"] = json!(f.n_eval);
        rec["converged"] = json!(f.converged());
        rec["singular"] = json!(f.singular());
        // No "maxeval" status: nothing caps evaluations here, so a fit either
        // converges, fails, or is killed by the watchdog from outside.
        rec["status"] = json!(if f.converged() { "ok" } else { "engine-fail" });
        if !f.converged() {
            rec["message"] = json!(not_converged_text(&f));
        }
        // Built last because it is the one field derived from a shape the fit may
        // not have filled in; a panic in it still costs the whole record, which
        // the `catch_unwind` around this block turns into an engine-fail.
        rec["varcomp"] = varcomp(&f, &lo.re_groups);
        rec
    }));
    match built {
        Ok(rec) => rec,
        Err(payload) => {
            let mut rec = base_record(cell_id, "glmm", engine_version);
            rec["message"] = json!(panic_text(payload.as_ref()));
            rec
        }
    }
}

/// The panic message a caught `catch_unwind` payload carries. `panic!` with a
/// formatted message boxes a `String`, a literal message a `&'static str`;
/// anything else is not a message at all.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&'static str>() {
        s.to_string()
    } else {
        "panicked".to_string()
    }
}

/// What the fit says about itself when it did not converge: the boundary state,
/// which variance components were pinned there, and any solver notes. Keeps a
/// non-convergence diagnosable in a run of ~800 cells, where the alternative is
/// a record that only says "engine-fail".
fn not_converged_text(f: &Fit) -> String {
    let d = &f.diagnostics;
    format!(
        "not converged: boundary={:?}, pinned={:?}, notes={:?}",
        d.boundary, d.pinned, d.notes
    )
}

/// Variance components in the grid schema, one entry per grouping factor in
/// declaration order, from `Fit::stddev_corr` (arbitrary q). Empty for a
/// fixed-only cell, which has no `re_groups`.
///
/// Also empty when the fit did not fill `varcorr`: the crate leaves it empty on
/// a non-converged mixed fit, and `Fit::stddev_corr` indexes it directly, so
/// calling it per declared grouping would panic and cost the record every other
/// field.
fn varcomp(f: &Fit, re_groups: &[ReGroupInfo]) -> Value {
    if f.varcorr.len() != re_groups.len() {
        return Value::Array(vec![]);
    }
    Value::Array(
        re_groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let (stddev, corr) = f.stddev_corr(i);
                json!({
                    "group": g.name,
                    "terms": g.terms,
                    "stddev": nums(&stddev),
                    "corr": Value::Array(corr.iter().map(|row| nums(row)).collect()),
                })
            })
            .collect(),
    )
}

fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = v.len() / 2;
    if v.len() % 2 == 1 {
        v[m]
    } else {
        (v[m - 1] + v[m]) / 2.0
    }
}
