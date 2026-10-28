//! extendr binding: the `#[extendr]` surface the `fastglmm` R package calls.
//! All logic lives in `glmm::orchestrate` (the crate's shared
//! port-orchestration module, behind its `orchestrate` feature) — this file is
//! only the FFI shim. Re-exported into the R package's staticlib by
//! `r/src/rust` (`extendr_module! { mod fastglmm; use glmm_r; }`).
//!
//! Mirror crate: `glmm-python` — the two ports bind the kernel at the same
//! level. Absent optionals cross this boundary as empty
//! vectors / `NaN`, decoded below; the R wrapper (`r/R/fastglmm.R`) owns all
//! argument validation, so every `Err` here is a boundary backstop, not a
//! user-facing message.

use std::collections::HashMap;

use glmm::formula::anova::{
    compare_stored, CompareError, Comparison, ComparisonTest, Differs, Model, ModelInfo, Side,
    TermTest, TermTestError,
};
use glmm::inference::{DenDf, Reference};
use glmm::orchestrate;

use extendr_api::error::Result;
use extendr_api::prelude::*;

/// Named list of double vectors -> `HashMap` for `orchestrate::run_fit`.
fn numeric_map(numeric: List) -> Result<HashMap<String, Vec<f64>>> {
    let mut out = HashMap::new();
    for (name, col) in numeric.iter() {
        let values = col.as_real_vector().ok_or_else(|| {
            Error::Other(format!("numeric column {name:?} is not a double vector"))
        })?;
        out.insert(name.to_string(), values);
    }
    Ok(out)
}

/// Two parallel named lists (levels: character vectors, codes: 0-based integer
/// vectors) -> the `(levels, codes)` factor map. Parallel-by-name rather than
/// nested pairs so the R side builds them with two plain `lapply`s.
#[allow(clippy::type_complexity)] // run_fit's own factor-map type — not worth an alias here
fn factor_map(
    factor_levels: List,
    factor_codes: List,
) -> Result<HashMap<String, (Vec<String>, Vec<u32>)>> {
    let mut codes_by_name: HashMap<String, Vec<u32>> = HashMap::new();
    for (name, col) in factor_codes.iter() {
        let codes = col.as_integer_vector().ok_or_else(|| {
            Error::Other(format!("factor codes {name:?} are not an integer vector"))
        })?;
        let codes: Vec<u32> = codes
            .into_iter()
            .map(|c| {
                u32::try_from(c)
                    .map_err(|_| Error::Other(format!("factor column {name:?}: negative code {c}")))
            })
            .collect::<Result<_>>()?;
        codes_by_name.insert(name.to_string(), codes);
    }
    let mut out = HashMap::new();
    for (name, col) in factor_levels.iter() {
        let levels = col.as_string_vector().ok_or_else(|| {
            Error::Other(format!("factor levels {name:?} are not a character vector"))
        })?;
        let codes = codes_by_name.remove(name).ok_or_else(|| {
            Error::Other(format!("factor column {name:?} has levels but no codes"))
        })?;
        out.insert(name.to_string(), (levels, codes));
    }
    if let Some(name) = codes_by_name.keys().next() {
        return Err(Error::Other(format!(
            "factor column {name:?} has codes but no levels"
        )));
    }
    Ok(out)
}

/// Fit `formula` against pre-marshalled columns and return the `FitResult` as
/// a named R list - one entry per field, mirroring `glmm-python/src/lib.rs::
/// fit_dict` (a named container at both ends makes an omitted field a lookup
/// error in R instead of silence; every field of `glmm::Fit` belongs here).
/// ASCII only here: these lines become R code comments in the generated
/// wrapper, and R CMD check flags non-ASCII R sources.
///
/// Optionals use in-band absent values (the raw bridge is not the user API):
/// an empty vector means absent - for `dispersion` too (a scalar `f64` would
/// force NA/NaN through extendr's NA check, which rejects it), and for
/// `reml` (`logical(0)` is `None`, one `TRUE`/`FALSE` is `Some`).
/// `vcov` crosses flattened row-major with `names` carrying p; the R wrapper
/// rebuilds the p x p matrix (it is symmetric, so the order is moot).
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_fit(
    formula: &str,
    numeric: List,
    factor_levels: List,
    factor_codes: List,
    family: &str,
    link: &str,
    wald_se: &str,
    nagq: i32,
    dispersion: &[f64],
    weights: &[f64],
    offset: &[f64],
    start_beta: &[f64],
    start_theta: &[f64],
    start_dispersion: &[f64],
    satterthwaite: bool,
    condvar: bool,
    reml: &[Rbool],
    anova_test: &str,
) -> Result<List> {
    let nagq = u8::try_from(nagq)
        .map_err(|_| Error::Other(format!("nagq {nagq} is out of range for u8")))?;
    let dispersion = dispersion.first().copied();
    let weights = optional(weights);
    let offset = optional(offset);
    let start_dispersion = start_dispersion.first().copied();
    let reml = match reml {
        [] => None,
        [v] if !v.is_na() => Some(v.is_true()),
        _ => return Err(Error::Other("reml must be empty or one TRUE/FALSE".into())),
    };
    let warm_start =
        if start_beta.is_empty() && start_theta.is_empty() && start_dispersion.is_none() {
            None
        } else {
            Some((start_beta.to_vec(), start_theta.to_vec(), start_dispersion))
        };

    let r = orchestrate::run_fit(
        formula,
        numeric_map(numeric)?,
        factor_map(factor_levels, factor_codes)?,
        family,
        link,
        wald_se,
        nagq,
        dispersion,
        weights,
        offset,
        warm_start,
        satterthwaite,
        condvar,
        reml,
        anova_test,
    )
    .map_err(Error::Other)?;
    Ok(fit_list(r))
}

/// The empty-means-absent decoding of an optional vector argument.
fn optional(values: &[f64]) -> Option<Vec<f64>> {
    (!values.is_empty()).then(|| values.to_vec())
}

/// A `FitResult` as the named R list `fastglmm_fit` returns; `fastglmm_compare`
/// returns its two fits the same way.
fn fit_list(r: orchestrate::FitResult) -> List {
    let vcov_flat: Vec<f64> = r.vcov.iter().flatten().copied().collect();
    // What `"auto"` gives on this fit: list(test, reason).
    let (auto_test, auto_reason) = orchestrate::choice_tags(&r.term_test_auto);
    let varcorr = List::from_values(r.varcorr.iter().map(|v| r!(v.clone())));
    let re_group_names: Vec<String> = r.re_groups.iter().map(|(n, _)| n.clone()).collect();
    let re_group_terms = List::from_values(r.re_groups.iter().map(|(_, t)| r!(t.clone())));
    // 1-based like the notes' columns below.
    let re_term_index = List::from_values(
        r.re_term_index
            .iter()
            .map(|t| r!(t.iter().map(|&k| k as i32 + 1).collect::<Vec<i32>>())),
    );
    let agq_warning: Robj = match r.agq_warning {
        Some(msg) => r!(msg),
        None => r!(NULL),
    };
    let reml_warning: Robj = match r.reml_warning {
        Some(msg) => r!(msg),
        None => r!(NULL),
    };
    let weights: Robj = match r.weights {
        Some(w) => r!(w),
        None => r!(NULL),
    };
    let design_summary: Robj = match r.design_summary {
        Some((counts, s, c)) => r!(list!(counts = counts, s = s, c = c)),
        None => r!(NULL),
    };
    let pinned = List::from_values(r.pinned.iter().map(|flags| r!(flags.clone())));
    // Column indices become 1-based HERE, once, so nothing on the R side of
    // this boundary ever handles a 0-based index: `notes[[i]]$columns` indexes
    // `names` directly. The Python port keeps them 0-based for the same reason
    // - change together, and keep the two conventions each idiomatic.
    let notes = List::from_values(r.notes.iter().map(|n| {
        r!(list!(
            kind = n.kind,
            columns = n
                .columns
                .iter()
                .map(|&c| c as i32 + 1)
                .collect::<Vec<i32>>(),
            pivot = n.pivot,
            evals = n.evals as i32,
            final_eval = n.final_eval,
            detail = n.detail.clone()
        ))
    }));
    // Labelled conditional modes, one entry per grouping. `ranef.fastglmm`
    // reshapes each into lme4's named data.frame. The flat `ranef` vector is
    // not sent, because only the kernel knows the block layout.
    // `condvar` is each level's vech, row-major over the levels, empty where
    // the fit formed none; `ranef.fastglmm` builds lme4's `postVar` from it.
    let ranef_blocks = List::from_values(r.ranef_blocks.iter().map(
        |(g, terms, levels, values, condvar)| {
            r!(list!(
                group = g.clone(),
                terms = terms.clone(),
                levels = levels.clone(),
                values = values.clone(),
                condvar = condvar.clone()
            ))
        },
    ));
    // What `predict.fastglmm` and `simulate.fastglmm` hand back to
    // `fastglmm_predict` and `fastglmm_simulate` (`parts_of` reads it), kept
    // private on the fit: the flat modes and their level counts, and the
    // conditional variances in the kernel's layout.
    let condvar_cross: Robj = match &r.condvar_cross {
        Some(c) => r!(list!(
            family_grouping = c.family_grouping as f64,
            family = c.family.clone(),
            tail = c.tail.clone(),
            coupling_at = c
                .coupling_at
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<f64>>(),
            coupling_cols = c
                .coupling_cols
                .iter()
                .map(|&v| v as f64)
                .collect::<Vec<f64>>(),
            coupling = c.coupling.clone()
        )),
        None => r!(NULL),
    };
    let kernel_parts = list!(
        ranef = r.ranef.clone(),
        ranef_levels = r
            .ranef_levels
            .iter()
            .map(|&n| n as f64)
            .collect::<Vec<f64>>(),
        condvar = r.condvar.clone(),
        condvar_x = r.condvar_x.clone(),
        condvar_cross = condvar_cross
    );

    let offset: Robj = match r.offset {
        Some(o) => r!(o),
        None => r!(NULL),
    };
    let (objective, objective_nodes) = orchestrate::objective_parts(r.objective);
    // Its retained columns become 1-based here, as the notes' columns do, and
    // `record_of` turns them back.
    let satterthwaite_record: Robj = match r.satterthwaite_record {
        Some(rec) => r!(list!(
            a_eff = rec.a_eff,
            jac_theta = List::from_values(rec.jac_theta.into_iter().map(|j| r!(j))),
            c = rec.c,
            tau = rec.tau,
            nu = rec.nu,
            retained = rec.retained.map_or(r!(NULL), |kept| {
                r!(kept.iter().map(|&j| j as i32 + 1).collect::<Vec<i32>>())
            })
        )),
        None => r!(NULL),
    };

    list!(
        beta = r.beta,
        se = r.se,
        vcov = vcov_flat,
        varcorr = varcorr,
        stddev_se = r.stddev_se,
        aliased = r.aliased,
        dispersion = r.dispersion,
        converged = r.converged,
        n_eval = r.n_eval as f64,
        n_grad = r.n_grad as f64,
        outer_stop = r.outer_stop,
        outer_decrement = r.outer_decrement,
        deviance = r.deviance,
        singular = r.singular,
        names = r.names,
        re_group_names = re_group_names,
        re_group_terms = re_group_terms,
        re_term_index = re_term_index,
        agq_warning = agq_warning,
        reml_warning = reml_warning,
        loglik = r.loglik,
        df = r.df as f64,
        reml = r.reml,
        fitted = r.fitted,
        ranef_blocks = ranef_blocks,
        kernel_parts = kernel_parts,
        y = r.y,
        weights = weights,
        design_summary = design_summary,
        nobs = r.nobs as f64,
        boundary = r.boundary,
        pinned = pinned,
        notes = notes,
        coef_df = r.coef_df,
        coef_p = r.coef_p,
        df_method = r.df_method,
        satterthwaite_record = satterthwaite_record,
        rss = r.rss,
        r_squared = r.r_squared,
        adj_r_squared = r.adj_r_squared,
        offset = offset,
        objective = objective,
        objective_nodes = objective_nodes as i32,
        term_table_type3 = term_list(&r.term_table_type3),
        term_table_type2 = term_list(&r.term_table_type2),
        term_table_type3_lrt = lrt_list(&r.term_table_type3_lrt),
        term_table_type2_lrt = lrt_list(&r.term_table_type2_lrt),
        term_test_auto = list!(test = auto_test, reason = auto_reason)
    )
}

/// A likelihood-ratio table's slot: `NULL` unless the fit asked for the
/// table, else as `term_list`.
fn lrt_list(slot: &Option<std::result::Result<glmm::formula::anova::TermTable, String>>) -> Robj {
    slot.as_ref().map_or(r!(NULL), |t| term_list(t).into())
}

/// A stored per-term result as a named list: `error` is the refusal's
/// message, or `NULL` beside the table's columns (`orchestrate::TermColumns`,
/// one entry per row; `row_notes` a list of character vectors).
fn term_list(result: &std::result::Result<glmm::formula::anova::TermTable, String>) -> List {
    match result {
        Err(message) => list!(error = message.clone()),
        Ok(table) => {
            let c = orchestrate::term_columns(table);
            list!(
                error = r!(NULL),
                ss_type = c.ss_type,
                test = c.test,
                reason = c.reason,
                label = c.label,
                kind = c.kind,
                statistic = c.statistic,
                num_df = c.num_df,
                den_df = c.den_df,
                den_df_kind = c.den_df_kind,
                p_value = c.p_value,
                sum_sq = c.sum_sq,
                mean_sq = c.mean_sq,
                eta2_partial = c.eta2_partial,
                row_notes = List::from_values(c.row_notes.iter().map(|n| r!(n.clone()))),
                notes = c.notes
            )
        }
    }
}

/// `glmm::orchestrate::confint` for `confint.fastglmm`: the lower and upper
/// bounds per coefficient, the t quantile on a finite df and the normal one
/// elsewhere, so both ports give the same intervals from one rule. `NaN`
/// entries (aliased coefficients, z rows' df) cross as plain doubles. `level`
/// crosses as a vector, so an `NA` reaches the shared check (and its message)
/// instead of extendr's scalar NA rejection.
///
/// @usage NULL
/// @keywords internal
#[extendr]
fn fastglmm_confint(beta: &[f64], se: &[f64], coef_df: &[f64], level: &[f64]) -> Result<List> {
    let level = match level {
        [v] => *v,
        _ => f64::NAN,
    };
    let ci = orchestrate::confint(beta, se, coef_df, level).map_err(Error::Other)?;
    let lower: Vec<f64> = ci.iter().map(|b| b[0]).collect();
    let upper: Vec<f64> = ci.iter().map(|b| b[1]).collect();
    Ok(list!(lower = lower, upper = upper))
}

/// `glmm::orchestrate::same_fixed_part` for `AIC.fastglmm` and
/// `BIC.fastglmm`: TRUE when the two formula strings have the same fixed
/// terms (compared as sets of variable names) and the same fixed intercept.
/// The response, the offset and the random part are not read. A formula that
/// does not parse is an error.
///
/// @usage NULL
/// @keywords internal
#[extendr]
fn fastglmm_same_fixed_part(f0: &str, f1: &str) -> Result<bool> {
    orchestrate::same_fixed_part(f0, f1).map_err(Error::Other)
}

/// `nsim` and `seed` of a comparison, decoded for
/// `orchestrate::compare_options`, which checks the range of `nsim`: an
/// empty vector is `None`. The R wrapper has checked that each is one whole
/// number.
fn nsim_seed(nsim: &[f64], seed: &[f64]) -> Result<(Option<usize>, Option<u64>)> {
    let whole = |v: f64, max: f64| v >= 0.0 && v.fract() == 0.0 && v <= max;
    let nsim = match nsim {
        [] => None,
        [v] if whole(*v, 1e15) => Some(*v as usize),
        [v] => {
            return Err(Error::Other(format!(
                "nsim must be a whole number from 1 to {}, got {v}",
                orchestrate::MAX_NSIM
            )))
        }
        _ => return Err(Error::Other("nsim must be empty or one number".into())),
    };
    let seed = match seed {
        [] => None,
        [v] if whole(*v, 9_007_199_254_740_992.0) => Some(*v as u64),
        _ => {
            return Err(Error::Other(
                "seed must be empty or one whole number from 0 to 2^53".into(),
            ))
        }
    };
    Ok((nsim, seed))
}

/// `glmm::orchestrate::run_compare` for `compare()`: fits the two formulas on
/// the same pre-marshalled columns (the R wrapper has dropped every row with
/// a missing value in a variable of either formula, the weights or the
/// offset) and returns `list(comparison, fit0, fit1, nsim, seed)`: the
/// comparison as `comparison_list` flattens it, each fit as `fastglmm_fit`
/// returns one, and the draws requested and the seed of the exact test.
/// Empty `weights`, `offset`, `nsim` and `seed` are absent.
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_compare(
    f0: &str,
    f1: &str,
    numeric: List,
    factor_levels: List,
    factor_codes: List,
    family: &str,
    link: &str,
    nagq: i32,
    weights: &[f64],
    offset: &[f64],
    nsim: &[f64],
    seed: &[f64],
) -> Result<List> {
    let nagq = u8::try_from(nagq)
        .map_err(|_| Error::Other(format!("nagq {nagq} is out of range for u8")))?;
    let (nsim, seed) = nsim_seed(nsim, seed)?;
    let r = orchestrate::run_compare(
        f0,
        f1,
        numeric_map(numeric)?,
        factor_map(factor_levels, factor_codes)?,
        family,
        link,
        nagq,
        optional(weights),
        optional(offset),
        nsim,
        seed,
    )
    .map_err(Error::Other)?;
    Ok(list!(
        comparison = comparison_list(&r.comparison)?,
        fit0 = fit_list(r.fit0),
        fit1 = fit_list(r.fit1),
        nsim = r.nsim as f64,
        seed = r.seed as f64
    ))
}

/// `glmm::formula::anova::compare_stored` for `anova(m0, m1)`: compares two stored
/// fits as they are, with no refit and no data. `m0` and `m1` are named lists
/// of the fields a fit object stores (the R wrapper's `.compare_side()`
/// builds them). Returns `list(comparison, nsim, seed)` as `fastglmm_compare`
/// does, or, when the pair cannot be compared, `list(refusal, kind)`: the
/// refusal's message and its kind, so the wrapper can point to its own
/// `compare()` where the kind calls for it (`kind` is `"other"` for the
/// refusals that need no pointer).
///
/// @usage NULL
/// @keywords internal
#[extendr]
fn fastglmm_anova_pair(m0: List, m1: List, nsim: &[f64], seed: &[f64]) -> Result<List> {
    let (nsim, seed) = nsim_seed(nsim, seed)?;
    let opts = orchestrate::compare_options(nsim, seed).map_err(Error::Other)?;
    match compare_stored(&model_of(&m0)?, &model_of(&m1)?, &opts) {
        Ok(c) => Ok(list!(
            comparison = comparison_list(&c)?,
            nsim = opts.nsim as f64,
            seed = opts.seed as f64
        )),
        Err(e) => Ok(list!(refusal = e.to_string(), kind = refusal_kind(&e))),
    }
}

/// The refusals whose R message points to `compare()`, by kind.
fn refusal_kind(e: &CompareError) -> &'static str {
    match e {
        CompareError::DifferentRows { .. } => "different_rows",
        CompareError::DifferentY => "different_y",
        CompareError::DifferentWeights => "different_weights",
        CompareError::DifferentOffset => "different_offset",
        CompareError::DifferentFamily { .. } => "different_family",
        CompareError::MixedEstimators { .. } => "mixed_estimators",
        CompareError::RemlDifferentFixed => "reml_different_fixed",
        _ => "other",
    }
}

/// One side of `anova(m0, m1)`: a `Model` from a fit object's stored fields,
/// named as in `Model` (`family` and `link` in the ports' vocabulary,
/// `objective` and `objective_nodes` as `orchestrate::objective_parts` gives
/// them, `re_group_names` and `re_group_terms` for `re_groups`, `vcov`
/// flattened row-major, `design_summary` `NULL` or `list(counts, s, c)`).
/// An empty `weights`, `offset` or `dispersion_held` is absent.
fn model_of(side: &List) -> Result<Model> {
    let fields: HashMap<&str, Robj> = side.iter().collect();
    let bad = |name: &str| Error::Other(format!("fit field {name:?} is missing or malformed"));
    let get = |name: &str| fields.get(name).ok_or_else(|| bad(name));
    let reals = |name: &str| get(name)?.as_real_vector().ok_or_else(|| bad(name));
    let real = |name: &str| -> Result<f64> {
        match reals(name)?.as_slice() {
            [v] => Ok(*v),
            _ => Err(bad(name)),
        }
    };
    let count = |name: &str| -> Result<usize> {
        let v = real(name)?;
        if v >= 0.0 && v.fract() == 0.0 {
            Ok(v as usize)
        } else {
            Err(bad(name))
        }
    };
    let text = |name: &str| {
        get(name)?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| bad(name))
    };
    let flag = |name: &str| get(name)?.as_bool().ok_or_else(|| bad(name));
    let list = |name: &str| List::try_from(get(name)?).map_err(|_| bad(name));
    let flags = |name: &str| -> Result<Vec<bool>> {
        let v = get(name)?.as_logical_vector().ok_or_else(|| bad(name))?;
        Ok(v.iter().map(|b| b.is_true()).collect())
    };

    let family =
        orchestrate::family_from_str(&text("family")?, &text("link")?).map_err(Error::Other)?;
    let nodes = u8::try_from(count("objective_nodes")?).map_err(|_| bad("objective_nodes"))?;
    let objective =
        orchestrate::objective_from_parts(&text("objective")?, nodes).map_err(Error::Other)?;
    let varcorr = list("varcorr")?
        .values()
        .map(|b| b.as_real_vector().ok_or_else(|| bad("varcorr")))
        .collect::<Result<Vec<_>>>()?;
    let names = get("re_group_names")?
        .as_string_vector()
        .ok_or_else(|| bad("re_group_names"))?;
    let terms = list("re_group_terms")?
        .values()
        .map(|t| t.as_string_vector().ok_or_else(|| bad("re_group_terms")))
        .collect::<Result<Vec<_>>>()?;
    let design_summary = match get("design_summary")? {
        d if d.is_null() => None,
        d => {
            let d: HashMap<&str, Robj> = List::try_from(d)
                .map_err(|_| bad("design_summary"))?
                .iter()
                .collect();
            let part = |name: &str| {
                d.get(name)
                    .and_then(|v| v.as_real_vector())
                    .ok_or_else(|| bad("design_summary"))
            };
            Some((part("counts")?, part("s")?, part("c")?))
        }
    };
    let coef = reals("coef")?;
    let p = coef.len();
    let vcov_flat = reals("vcov")?;
    if vcov_flat.len() != p * p {
        return Err(bad("vcov"));
    }
    Ok(Model {
        formula: text("formula")?,
        family,
        loglik: real("loglik")?,
        df: count("df")?,
        nobs: count("nobs")?,
        reml: flag("reml")?,
        objective,
        converged: flag("converged")?,
        varcorr,
        re_groups: names.into_iter().zip(terms).collect(),
        y: reals("y")?,
        weights: optional(&reals("weights")?),
        offset: optional(&reals("offset")?),
        dispersion_held: reals("dispersion_held")?.first().copied(),
        design_summary,
        rss: real("rss")?,
        coef_names: get("coef_names")?
            .as_string_vector()
            .ok_or_else(|| bad("coef_names"))?,
        coef,
        vcov: vcov_flat.chunks(p.max(1)).map(<[f64]>::to_vec).collect(),
        aliased: flags("aliased")?,
    })
}

/// A `Comparison` as a named list for the R wrapper:
/// - `larger`: `"m0"` or `"m1"`, the side whose model is the larger;
/// - `differs`: `"fixed"`, `"random"` or `"both"`, the parts that differ;
/// - `m0`, `m1`: each side's `formula`, `npar`, `nobs`, `loglik`, `aic` and
///   `bic` (`NULL` on a REML fit) and `objective` (`objective_parts`' label);
/// - the test: `test` (`"lr"` or `"linear"`), `statistic`, `df` (the LR
///   test's df or the F's numerator df), `p_value`, and on an LR test the
///   `objective` and `objective_nodes` it is labelled with (`""` and 0 on an
///   F);
/// - the reference: `reference` (`"chisq"`, `"chibar"`, `"exact"` or `"F"`),
///   `ref_df` (the chi-squared df, the mixture's lower df, the F's numerator
///   df), `den_df` (the F's), and on the exact test `draws`, the draws made
///   (0 when none were needed), and `seed`;
/// - `notes`: the test's notes as sentences.
fn comparison_list(c: &Comparison) -> Result<List> {
    let side = |m: &ModelInfo| {
        let opt = |v: Option<f64>| v.map_or(r!(NULL), |v| r!(v));
        list!(
            formula = m.formula_label.clone(),
            npar = m.npar as f64,
            nobs = m.nobs as f64,
            loglik = m.loglik,
            aic = opt(m.aic),
            bic = opt(m.bic),
            objective = orchestrate::objective_parts(m.objective).0
        )
    };
    let (test, statistic, df, p_value, reference, objective) = match &c.test {
        ComparisonTest::Lr(t) => (
            "lr",
            t.statistic,
            t.df as f64,
            t.p_value,
            t.reference,
            orchestrate::objective_parts(t.objective),
        ),
        ComparisonTest::Linear(t) => (
            "linear",
            t.statistic,
            t.num_df,
            t.p_value,
            t.reference,
            ("", 0),
        ),
        _ => {
            return Err(Error::Other(
                "a comparison test this port does not know".into(),
            ))
        }
    };
    let nan = f64::NAN;
    let (kind, ref_df, den_df, draws, seed) = match reference {
        Reference::ChiSquared { df } => ("chisq", df as f64, nan, nan, nan),
        Reference::ChiBarSquared { df_low } => ("chibar", df_low as f64, nan, nan, nan),
        Reference::ExactRlrt { nsim, seed } => ("exact", nan, nan, nsim as f64, seed as f64),
        Reference::F { num_df, den_df, .. } => ("F", num_df, den_df, nan, nan),
        _ => return Err(Error::Other("a reference this port does not know".into())),
    };
    Ok(list!(
        larger = match c.larger {
            Side::First => "m0",
            Side::Second => "m1",
        },
        differs = match c.differs {
            Differs::Fixed => "fixed",
            Differs::Random => "random",
            Differs::Both => "both",
        },
        m0 = side(&c.m0),
        m1 = side(&c.m1),
        test = test,
        statistic = statistic,
        df = df,
        p_value = p_value,
        objective = objective.0,
        objective_nodes = objective.1 as i32,
        reference = kind,
        ref_df = ref_df,
        den_df = den_df,
        draws = draws,
        seed = seed,
        notes = c.note_lines()
    ))
}

/// `glmm::orchestrate::contrast_test` for `contrast_test()`: one Wald test of
/// `L beta = 0` on the fields a fit object stores. `vcov` and `l` cross
/// flattened row-major, `l` with `rows` rows; `dispersion` crosses as a
/// vector, so a NaN dispersion (a fit that did not converge) is not refused
/// by extendr's scalar NA check; `record` is the fit's
/// `satterthwaite_record`, `NULL` when it has none. Returns the one-row
/// table as `term_list` gives it.
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_contrast_test(
    df_method: &str,
    beta: &[f64],
    vcov: &[f64],
    aliased: &[Rbool],
    nobs: f64,
    dispersion: &[f64],
    record: Robj,
    names: Vec<String>,
    l: &[f64],
    rows: i32,
) -> Result<List> {
    let dispersion = match dispersion {
        [v] => *v,
        _ => return Err(Error::Other("dispersion must be one number".into())),
    };
    let (vcov, aliased, record) = stored_test_fields(beta.len(), vcov, aliased, &record)?;
    let table = orchestrate::contrast_test(
        df_method,
        beta,
        &vcov,
        &aliased,
        nobs as usize,
        dispersion,
        record,
        &names,
        l,
        usize::try_from(rows).unwrap_or(0),
    )
    .map_err(Error::Other)?;
    Ok(term_list(&Ok(table)))
}

/// What `stored_test_fields` returns.
type TestFields = (
    Vec<Vec<f64>>,
    Vec<bool>,
    Option<orchestrate::SatterthwaiteRecord>,
);

/// What `fastglmm_contrast_test` and `fastglmm_pairwise` read of a fit
/// besides `beta`, for `p` coefficients: `vcov` (row-major) as its rows,
/// `aliased` as booleans, and `record` (`NULL` when the fit has none) as an
/// `orchestrate::SatterthwaiteRecord`.
fn stored_test_fields(
    p: usize,
    vcov: &[f64],
    aliased: &[Rbool],
    record: &Robj,
) -> Result<TestFields> {
    if vcov.len() != p * p {
        return Err(Error::Other(format!(
            "vcov has {} entries for {p} coefficients",
            vcov.len()
        )));
    }
    let vcov: Vec<Vec<f64>> = vcov.chunks(p.max(1)).map(<[f64]>::to_vec).collect();
    let aliased: Vec<bool> = aliased.iter().map(|a| a.is_true()).collect();
    let record = if record.is_null() {
        None
    } else {
        Some(record_of(&List::try_from(record)?)?)
    };
    Ok((vcov, aliased, record))
}

/// `glmm::orchestrate::pairwise` for `pairwise()`: the training columns as
/// `fastglmm_predict` takes them, the fit's fields as `fastglmm_contrast_test`
/// takes them, then `factor`, `by` (`""` for none), `by_values` (empty for
/// none) and `adjust`. Returns the table as `orchestrate::pairwise_columns`
/// flattens it, a list keyed by its fields; `by` and `ratio_kind` are `NULL`
/// when absent and `levels` crosses as a double.
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_pairwise(
    formula: &str,
    numeric: List,
    factor_levels: List,
    factor_codes: List,
    family: &str,
    link: &str,
    df_method: &str,
    beta: &[f64],
    vcov: &[f64],
    aliased: &[Rbool],
    nobs: f64,
    record: Robj,
    names: Vec<String>,
    factor: &str,
    by: &str,
    by_values: &[f64],
    adjust: &str,
) -> Result<List> {
    let (vcov, aliased, record) = stored_test_fields(beta.len(), vcov, aliased, &record)?;
    let pw = orchestrate::pairwise(
        formula,
        numeric_map(numeric)?,
        factor_map(factor_levels, factor_codes)?,
        family,
        link,
        df_method,
        beta,
        &vcov,
        &aliased,
        nobs as usize,
        record,
        factor,
        (!by.is_empty()).then_some(by),
        optional(by_values),
        adjust,
    )
    .map_err(Error::Other)?;
    let c = orchestrate::pairwise_columns(&pw, &names);
    let or_null = |v: Option<String>| v.map_or_else(|| r!(NULL), |s| r!(s));
    Ok(list!(
        factor = c.factor,
        by = or_null(c.by),
        adjust = c.adjust,
        ratio_kind = or_null(c.ratio_kind.map(str::to_string)),
        levels = c.levels as f64,
        contrast = c.contrast,
        by_level = c.by_level,
        estimate = c.estimate,
        se = c.se,
        df = c.df,
        statistic = c.statistic,
        p_value = c.p_value,
        p_adjusted = c.p_adjusted,
        ratio = c.ratio,
        ratio_se = c.ratio_se,
        row_notes = List::from_values(c.row_notes.iter().map(|n| r!(n.clone())))
    ))
}

/// `glmm::orchestrate::pairwise_confint` for `confint()` of a pairwise
/// table: the columns `fastglmm_pairwise` gave (`estimate`, `se`, `df`, the
/// unadjusted `p_value`, `adjust`, `levels` as a double, `ratio` the
/// `ratio_kind`, `""` for none). `level` crosses as a vector, as in
/// `fastglmm_confint`. Returns `list(adjust, lower, upper, ratio_lower,
/// ratio_upper)`, `adjust` the adjustment the intervals use.
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_pairwise_confint(
    estimate: &[f64],
    se: &[f64],
    df: &[f64],
    p_value: &[f64],
    adjust: &str,
    levels: f64,
    ratio: &str,
    level: &[f64],
) -> Result<List> {
    let level = match level {
        [v] => *v,
        _ => f64::NAN,
    };
    let ci = orchestrate::pairwise_confint(
        estimate,
        se,
        df,
        p_value,
        adjust,
        whole(levels, "levels")?,
        (!ratio.is_empty()).then_some(ratio),
        level,
    )
    .map_err(Error::Other)?;
    let col = |f: fn(&glmm::formula::PairwiseInterval) -> f64| -> Vec<f64> {
        ci.rows.iter().map(f).collect()
    };
    Ok(list!(
        adjust = orchestrate::adjust_tag(ci.adjust),
        lower = col(|r| r.lower),
        upper = col(|r| r.upper),
        ratio_lower = col(|r| r.ratio_lower),
        ratio_upper = col(|r| r.ratio_upper)
    ))
}

/// A fit object's `satterthwaite_record`, as `fit_list` writes it, read back
/// into `orchestrate::SatterthwaiteRecord` (its 1-based retained columns
/// 0-based again).
fn record_of(rec: &List) -> Result<orchestrate::SatterthwaiteRecord> {
    let fields: HashMap<&str, Robj> = rec.iter().collect();
    let bad = |name: &str| {
        Error::Other(format!(
            "satterthwaite_record field {name:?} is missing or malformed"
        ))
    };
    let get = |name: &str| fields.get(name).ok_or_else(|| bad(name));
    let reals = |name: &str| get(name)?.as_real_vector().ok_or_else(|| bad(name));
    let real = |name: &str| -> Result<f64> {
        match reals(name)?.as_slice() {
            [v] => Ok(*v),
            _ => Err(bad(name)),
        }
    };
    let jac_theta = List::try_from(get("jac_theta")?)
        .map_err(|_| bad("jac_theta"))?
        .values()
        .map(|j| j.as_real_vector().ok_or_else(|| bad("jac_theta")))
        .collect::<Result<Vec<_>>>()?;
    let retained = match get("retained")? {
        r if r.is_null() => None,
        r => Some(
            r.as_integer_vector()
                .ok_or_else(|| bad("retained"))?
                .into_iter()
                .map(|j| usize::try_from(j - 1).map_err(|_| bad("retained")))
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    Ok(orchestrate::SatterthwaiteRecord {
        a_eff: reals("a_eff")?,
        jac_theta,
        c: reals("c")?,
        tau: real("tau")?,
        nu: real("nu")?,
        retained,
    })
}

/// A whole number of at least 0 sent as a double, as `usize`.
fn whole(v: f64, what: &str) -> Result<usize> {
    if v >= 0.0 && v.fract() == 0.0 && v <= 9_007_199_254_740_992.0 {
        Ok(v as usize)
    } else {
        Err(Error::Other(format!(
            "{what} must be a whole number, got {v}"
        )))
    }
}

/// The numbers `predict.fastglmm`, `simulate.fastglmm` and `r2()` / `icc()`
/// send (the R wrapper's `.stored_numbers()`), read into `orchestrate::StoredParts`:
/// `beta`, `vcov` flattened row-major, `varcorr`, `dispersion`, and the
/// fit's `kernel_parts` (`fit_list` writes them) with the counts as doubles.
fn parts_of(parts: &List) -> Result<orchestrate::StoredParts> {
    let fields: HashMap<&str, Robj> = parts.iter().collect();
    let bad = |name: &str| Error::Other(format!("stored field {name:?} is missing or malformed"));
    let get = |name: &str| fields.get(name).ok_or_else(|| bad(name));
    let reals = |name: &str| get(name)?.as_real_vector().ok_or_else(|| bad(name));
    let counts = |v: Vec<f64>, name: &str| -> Result<Vec<usize>> {
        v.into_iter().map(|c| whole(c, name)).collect()
    };
    let beta = reals("beta")?;
    let p = beta.len();
    let vcov = reals("vcov")?;
    if vcov.len() != p * p {
        return Err(bad("vcov"));
    }
    let varcorr = List::try_from(get("varcorr")?)
        .map_err(|_| bad("varcorr"))?
        .values()
        .map(|b| b.as_real_vector().ok_or_else(|| bad("varcorr")))
        .collect::<Result<Vec<_>>>()?;
    let dispersion = match reals("dispersion")?.as_slice() {
        [v] => *v,
        _ => return Err(bad("dispersion")),
    };
    let condvar_cross = match get("condvar_cross")? {
        c if c.is_null() => None,
        c => {
            let c: HashMap<&str, Robj> = List::try_from(c)
                .map_err(|_| bad("condvar_cross"))?
                .iter()
                .collect();
            let part = |name: &str| {
                c.get(name)
                    .and_then(|v| v.as_real_vector())
                    .ok_or_else(|| bad("condvar_cross"))
            };
            let u32s = |name: &str| -> Result<Vec<u32>> {
                part(name)?
                    .into_iter()
                    .map(|v| u32::try_from(whole(v, name)?).map_err(|_| bad(name)))
                    .collect()
            };
            Some(orchestrate::CondvarCrossRecord {
                family_grouping: match part("family_grouping")?.as_slice() {
                    [v] => whole(*v, "family_grouping")?,
                    _ => return Err(bad("condvar_cross")),
                },
                family: part("family")?,
                tail: part("tail")?,
                coupling_at: u32s("coupling_at")?,
                coupling_cols: u32s("coupling_cols")?,
                coupling: part("coupling")?,
            })
        }
    };
    Ok(orchestrate::StoredParts {
        beta,
        vcov: vcov.chunks(p.max(1)).map(<[f64]>::to_vec).collect(),
        varcorr,
        ranef: reals("ranef")?,
        ranef_levels: counts(reals("ranef_levels")?, "ranef_levels")?,
        condvar: reals("condvar")?,
        condvar_x: reals("condvar_x")?,
        condvar_cross,
        dispersion,
    })
}

/// `glmm::orchestrate::predict` for `predict.fastglmm`: the training columns
/// as `fastglmm_fit` took them (the R wrapper marshals the fit's model frame
/// again), the stored numbers `parts` (`parts_of`), and the new table's
/// columns, or none when `has_new` is FALSE (the model frame's rows);
/// `n_rows` is the new table's row count. `level` crosses as a vector, as
/// in `fastglmm_confint`. Returns `list(fit, se, lower, upper, future_lower,
/// future_upper, notes)`, a part not asked for empty; each note is
/// `list(kind, detail)` with the rows in `detail` 1-based.
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_predict(
    formula: &str,
    numeric: List,
    factor_levels: List,
    factor_codes: List,
    family: &str,
    link: &str,
    weights: &[f64],
    offset: &[f64],
    parts: List,
    has_new: bool,
    new_numeric: List,
    new_levels: List,
    new_codes: List,
    n_rows: f64,
    scale: &str,
    population: bool,
    allow_new_levels: bool,
    se: bool,
    level: &[f64],
    future: bool,
) -> Result<List> {
    let level = match level {
        [v] => *v,
        _ => f64::NAN,
    };
    let new = if has_new {
        Some((
            numeric_map(new_numeric)?,
            factor_map(new_levels, new_codes)?,
            whole(n_rows, "n_rows")?,
        ))
    } else {
        None
    };
    let r = orchestrate::predict(
        formula,
        numeric_map(numeric)?,
        factor_map(factor_levels, factor_codes)?,
        family,
        link,
        optional(weights),
        optional(offset),
        &parts_of(&parts)?,
        new,
        scale,
        population,
        allow_new_levels,
        se,
        level,
        future,
    )
    .map_err(Error::Other)?;
    // The rows of a note become 1-based here, as the fit's columns do.
    let notes = List::from_values(r.notes.iter().map(|n| {
        let detail = match n.detail.split_once(": ") {
            Some((g, rows)) => {
                let rows: Vec<String> = rows
                    .split(", ")
                    .map(|i| {
                        i.parse::<u64>()
                            .map_or(i.to_string(), |i| (i + 1).to_string())
                    })
                    .collect();
                format!("{g}: {}", rows.join(", "))
            }
            None => n.detail.clone(),
        };
        r!(list!(kind = n.kind, detail = detail))
    }));
    Ok(list!(
        fit = r.fit,
        se = r.se,
        lower = r.lower,
        upper = r.upper,
        future_lower = r.future_lower,
        future_upper = r.future_upper,
        notes = notes
    ))
}

/// `glmm::orchestrate::simulate` for `simulate.fastglmm`: the columns and
/// `parts` as `fastglmm_predict` takes them; `nsim` and `seed` whole numbers
/// sent as doubles (the R wrapper has checked them). The draws draw after
/// draw, `nobs * nsim` doubles.
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_simulate(
    formula: &str,
    numeric: List,
    factor_levels: List,
    factor_codes: List,
    family: &str,
    link: &str,
    weights: &[f64],
    offset: &[f64],
    parts: List,
    nsim: f64,
    seed: f64,
) -> Result<Vec<f64>> {
    orchestrate::simulate(
        formula,
        numeric_map(numeric)?,
        factor_map(factor_levels, factor_codes)?,
        family,
        link,
        optional(weights),
        optional(offset),
        &parts_of(&parts)?,
        whole(nsim, "nsim")?,
        whole(seed, "seed")? as u64,
    )
    .map_err(Error::Other)
}

/// `glmm::orchestrate::effect_sizes` for `r2()`, `icc()` and
/// `summary.fastglmm`: the columns and `parts` as `fastglmm_predict` takes
/// them, and the fit's `aliased`, `converged`, `objective` and
/// `objective_nodes`. Returns `list(marginal, conditional, adjusted,
/// unadjusted)`, or `NULL` where the values need the null refit that
/// `null_refit = FALSE` skips.
///
/// @usage NULL
/// @keywords internal
#[extendr]
#[allow(clippy::too_many_arguments)]
fn fastglmm_effect_sizes(
    formula: &str,
    numeric: List,
    factor_levels: List,
    factor_codes: List,
    family: &str,
    link: &str,
    weights: &[f64],
    offset: &[f64],
    parts: List,
    aliased: &[Rbool],
    converged: bool,
    objective: &str,
    objective_nodes: i32,
    null_refit: bool,
) -> Result<Robj> {
    let aliased: Vec<bool> = aliased.iter().map(|a| a.is_true()).collect();
    let nodes = u8::try_from(objective_nodes)
        .map_err(|_| Error::Other(format!("objective_nodes {objective_nodes} is not 0 to 255")))?;
    let sizes = orchestrate::effect_sizes(
        formula,
        numeric_map(numeric)?,
        factor_map(factor_levels, factor_codes)?,
        family,
        link,
        optional(weights),
        optional(offset),
        &parts_of(&parts)?,
        &aliased,
        converged,
        (objective, nodes),
        null_refit,
    )
    .map_err(Error::Other)?;
    Ok(match sizes {
        Some((r2, icc)) => list!(
            marginal = r2.marginal,
            conditional = r2.conditional,
            adjusted = icc.adjusted,
            unadjusted = icc.unadjusted
        )
        .into(),
        None => r!(NULL),
    })
}

/// The refusal of `anova(m, test = "Wald")` or `test = "LRT"` on a fit
/// whose per-term rows are F tests, in the crate's words, so both ports say
/// the same. `satterthwaite` is TRUE when the F rows' denominator df are
/// Satterthwaite's (a mixed model), FALSE for the residual df.
///
/// @usage NULL
/// @keywords internal
#[extendr]
fn fastglmm_term_test_refusal(test: &str, satterthwaite: bool) -> Result<String> {
    let asked = match test {
        "Wald" => TermTest::Wald,
        "LRT" => TermTest::Lrt,
        other => return Err(Error::Other(format!("no refusal for test {other:?}"))),
    };
    let den_df = if satterthwaite {
        DenDf::Satterthwaite
    } else {
        DenDf::Residual
    };
    Ok(TermTestError::FTable { asked, den_df }.to_string())
}

extendr_module! {
    mod glmm_r;
    fn fastglmm_fit;
    fn fastglmm_confint;
    fn fastglmm_contrast_test;
    fn fastglmm_pairwise;
    fn fastglmm_pairwise_confint;
    fn fastglmm_same_fixed_part;
    fn fastglmm_compare;
    fn fastglmm_anova_pair;
    fn fastglmm_term_test_refusal;
    fn fastglmm_predict;
    fn fastglmm_simulate;
    fn fastglmm_effect_sizes;
}
