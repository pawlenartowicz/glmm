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
/// force NA/NaN through extendr's NA check, which rejects it).
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
) -> Result<List> {
    let nagq = u8::try_from(nagq)
        .map_err(|_| Error::Other(format!("nagq {nagq} is out of range for u8")))?;
    let dispersion = dispersion.first().copied();
    let weights = if weights.is_empty() {
        None
    } else {
        Some(weights.to_vec())
    };
    let offset = if offset.is_empty() {
        None
    } else {
        Some(offset.to_vec())
    };
    let warm_start = if start_beta.is_empty() && start_theta.is_empty() {
        None
    } else {
        Some((start_beta.to_vec(), start_theta.to_vec()))
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
    )
    .map_err(Error::Other)?;

    let vcov_flat: Vec<f64> = r.vcov.iter().flatten().copied().collect();
    let varcorr = List::from_values(r.varcorr.iter().map(|v| r!(v.clone())));
    let re_group_names: Vec<String> = r.re_groups.iter().map(|(n, _)| n.clone()).collect();
    let re_group_terms = List::from_values(r.re_groups.iter().map(|(_, t)| r!(t.clone())));
    let agq_warning: Robj = match r.agq_warning {
        Some(msg) => r!(msg),
        None => r!(NULL),
    };
    let weights: Robj = match r.weights {
        Some(w) => r!(w),
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
            detail = n.detail.clone(),
            ratio = n.ratio
        ))
    }));
    // Labelled conditional modes, one entry per grouping. `ranef.fastglmm`
    // reshapes each into lme4's named data.frame; nothing on the R side slices
    // the flat `ranef` vector, because only the kernel knows the block layout.
    let ranef_blocks =
        List::from_values(r.ranef_blocks.iter().map(|(g, terms, levels, values)| {
            r!(list!(
                group = g.clone(),
                terms = terms.clone(),
                levels = levels.clone(),
                values = values.clone()
            ))
        }));

    Ok(list!(
        beta = r.beta,
        se = r.se,
        vcov = vcov_flat,
        tau2 = r.tau2,
        varcorr = varcorr,
        stddev_se = r.stddev_se,
        aliased = r.aliased,
        dispersion = r.dispersion,
        converged = r.converged,
        n_eval = r.n_eval as f64,
        deviance = r.deviance,
        singular = r.singular,
        names = r.names,
        re_group_names = re_group_names,
        re_group_terms = re_group_terms,
        agq_warning = agq_warning,
        loglik = r.loglik,
        df = r.df as f64,
        reml = r.reml,
        fitted = r.fitted,
        ranef = r.ranef,
        ranef_levels = r
            .ranef_levels
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<f64>>(),
        ranef_blocks = ranef_blocks,
        y = r.y,
        weights = weights,
        nobs = r.nobs as f64,
        boundary = r.boundary,
        pinned = pinned,
        notes = notes
    ))
}

extendr_module! {
    mod glmm_r;
    fn fastglmm_fit;
}
