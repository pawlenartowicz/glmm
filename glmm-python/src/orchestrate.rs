//! Orchestrates one fit: formula + data -> `glmm-formula::lower` -> option
//! overrides -> `glmm::fit_warm`, wrapped in `catch_unwind` so the kernel's
//! `assert!`-based boundary faults become a normal `Err`, never a process
//! abort across the FFI. No `pyo3` types here — plain-Rust, unit-testable.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use glmm::{fit_warm, Family, StartValues, WaldSe};
use glmm_formula::{lower, Column, Table};

use crate::convert::family_from_str;

#[derive(Debug)]
pub struct FitResult {
    pub beta: Vec<f64>,
    pub se: Vec<f64>,
    pub tau2: Vec<f64>,
    pub varcorr: Vec<Vec<f64>>,
    pub stddev_se: Vec<f64>,
    pub aliased: Vec<bool>,
    pub dispersion: f64,
    pub converged: bool,
    /// `true` iff the fit converged onto a variance-component boundary
    /// (mirrors `glmm::Fit::singular` / lme4's `isSingular`) — boundary-fits
    /// follow-up spec Part B step 4 (docs/GLMM/plans/2026-07-14-boundary-
    /// fits-followup-spec.md): the flag was already computed but never
    /// surfaced through this wrapper.
    pub singular: bool,
    pub names: Vec<String>,
    /// §3.5 warn-and-strip message for an ineligible-shape `nagq>1` (the fit
    /// proceeded with Laplace); surfaced by `glmm.fit` as a `UserWarning`.
    pub agq_warning: Option<String>,
}

/// `FitResult` flattened for the PyO3 return — field order matches the
/// unpacking in `glmm/__init__.py::fit`.
pub type FitTuple = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<bool>,
    f64,
    bool,
    bool,
    Vec<String>,
    Option<String>,
);

impl FitResult {
    pub fn into_tuple(self) -> FitTuple {
        (
            self.beta,
            self.se,
            self.tau2,
            self.varcorr,
            self.stddev_se,
            self.aliased,
            self.dispersion,
            self.converged,
            self.singular,
            self.names,
            self.agq_warning,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_fit(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, Vec<String>>,
    family: &str,
    link: &str,
    wald_se: &str,
    nagq: u8,
    dispersion: Option<f64>,
    weights: Option<Vec<f64>>,
    targets: Option<Vec<String>>,
    warm_start: Option<(Vec<f64>, Vec<f64>)>,
) -> Result<FitResult, String> {
    let fam = family_from_str(family, link)?;

    let mut columns: Vec<(String, Column)> = Vec::new();
    let mut lengths: Vec<(String, usize)> = Vec::new();
    for (name, values) in numeric_columns {
        lengths.push((name.clone(), values.len()));
        columns.push((name, Column::Numeric(values)));
    }
    for (name, labels) in factor_columns {
        lengths.push((name.clone(), labels.len()));
        columns.push((name, Column::Factor { labels }));
    }
    // Require every column to agree on row count rather than taking the max:
    // a ragged column would otherwise be silently zero-padded by `lower()`'s
    // materialize step, converging to wrong coefficients with no error.
    // `numeric_columns`/`factor_columns` are HashMaps, so their iteration
    // order (and thus which entry lands first in `lengths`) is not stable
    // across calls -- the error message names every column instead of
    // picking out a single "offending" one, so it stays deterministic.
    let n = match lengths.first() {
        Some((_, first_len)) => {
            let n = *first_len;
            if lengths.iter().any(|(_, len)| *len != n) {
                let mut detail: Vec<String> = lengths
                    .iter()
                    .map(|(name, len)| format!("{name:?}: {len}"))
                    .collect();
                detail.sort();
                return Err(format!(
                    "columns have mismatched lengths ({}); all columns must have the same length",
                    detail.join(", ")
                ));
            }
            n
        }
        None => 0,
    };
    let table = Table { columns, n };

    let mut lowered = catch_unwind(AssertUnwindSafe(|| lower(formula, &table, fam)))
        .map_err(panic_message)?
        .map_err(|e| e.to_string())?;

    lowered.opts.wald_se = match wald_se {
        "hessian" => WaldSe::Hessian,
        "rx" => WaldSe::Rx,
        other => return Err(format!("unsupported wald_se {other:?}")),
    };
    // nagq>1 on an ineligible shape is valid-but-inapplicable — warn-and-strip
    // (the Python layer's §3.5 policy: nothing inapplicable may reach the
    // kernel, whose shape check is an `assert!`, and a Rust panic across the
    // FFI is not an acceptable user error). The shape is only knowable here,
    // after `lower()`, so this is the strip site; `glmm.fit` emits the message
    // as a `UserWarning`. Eligibility mirrors
    // `src/fit/common.rs::assert_model_shape` — change together: nagq>1 needs
    // a mixed binomial/Poisson model with a single grouping factor and
    // q_p = 1 + #slopes ≤ 3 (the temporary cost/oracle cap).
    let mut nagq = nagq;
    let mut agq_warning: Option<String> = None;
    if nagq > 1 {
        let agq_family = matches!(fam, Family::Binomial { .. } | Family::Poisson { .. });
        let eligible = match lowered.model.re.as_ref() {
            Some(re) => {
                let q_p = 1 + re.slopes.len(); // intercept + slopes, as in assert_model_shape
                agq_family && re.extra_groupings.is_empty() && q_p <= 3
            }
            None => false,
        };
        if !eligible {
            agq_warning = Some(format!(
                "nagq={nagq} (adaptive quadrature) applies only to binomial/Poisson \
                 mixed models with a single grouping factor and at most 3 random \
                 effects per group; fitting with Laplace (nagq=1)"
            ));
            nagq = 1;
        }
    }
    lowered.opts.nagq = nagq;
    lowered.opts.dispersion = dispersion;
    lowered.opts.weights = weights;
    if let Some(names) = &targets {
        let mut idx = Vec::with_capacity(names.len());
        for name in names {
            let pos = lowered
                .col_names
                .iter()
                .position(|c| c == name)
                .ok_or_else(|| format!("unknown target column {name:?}"))?;
            idx.push(pos as u32);
        }
        lowered.opts.target_indices = idx;
    }

    let start = warm_start.map(|(beta, theta)| StartValues { beta, theta });

    let fit = catch_unwind(AssertUnwindSafe(|| {
        fit_warm(
            &lowered.x,
            &lowered.y,
            lowered.n,
            lowered.p,
            &lowered.model,
            &lowered.ids,
            start.as_ref(),
            &lowered.opts,
        )
    }))
    .map_err(panic_message)?;

    Ok(FitResult {
        beta: fit.beta,
        se: fit.se,
        tau2: fit.tau2,
        varcorr: fit.varcorr,
        stddev_se: fit.stddev_se,
        aliased: fit.aliased,
        dispersion: fit.dispersion,
        converged: fit.converged,
        singular: fit.singular,
        names: lowered.col_names,
        agq_warning,
    })
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "glmm kernel panicked with a non-string payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn toy_ols() -> (HashMap<String, Vec<f64>>, HashMap<String, Vec<String>>) {
        let y = vec![1.0, 2.0, 2.9, 4.1, 5.0, 6.2, 6.8, 8.1, 9.0, 10.2];
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut numeric = HashMap::new();
        numeric.insert("y".to_string(), y);
        numeric.insert("x".to_string(), x);
        (numeric, HashMap::new())
    }

    #[test]
    fn gaussian_ols_end_to_end() {
        let (numeric, factor) = toy_ols();
        let result = run_fit(
            "y ~ x", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
        )
        .expect("fit should succeed");
        assert_eq!(
            result.names,
            vec!["(Intercept)".to_string(), "x".to_string()]
        );
        assert_eq!(result.beta.len(), 2);
        assert!(result.converged);
        // y ~= 1 + x, slope near 1.0 by construction.
        assert!(
            (result.beta[1] - 1.0).abs() < 0.1,
            "slope = {}",
            result.beta[1]
        );
    }

    #[test]
    fn unknown_column_is_a_clean_error() {
        let (numeric, factor) = toy_ols();
        let err = run_fit(
            "y ~ z", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
        )
        .unwrap_err();
        assert!(err.contains("z"), "{err}");
    }

    #[test]
    fn ineligible_nagq_is_stripped_with_a_warning_not_an_error() {
        // Gaussian LMM with nagq=3: ineligible family — must strip to Laplace
        // and report the §3.5 warning, never surface the kernel's shape panic.
        let (numeric, mut factor) = toy_ols();
        factor.insert(
            "g".to_string(),
            ["a", "b", "c", "d", "e"]
                .iter()
                .cycle()
                .take(10)
                .map(|s| s.to_string())
                .collect(),
        );
        let result = run_fit(
            "y ~ x + (1 | g)",
            numeric,
            factor,
            "gaussian",
            "identity",
            "hessian",
            3,
            None,
            None,
            None,
            None,
        )
        .expect("ineligible nagq must be stripped, not an error");
        let msg = result.agq_warning.as_deref().expect("warning expected");
        assert!(msg.contains("nagq=3"), "{msg}");
    }

    #[test]
    fn unknown_target_is_a_clean_error() {
        let (numeric, factor) = toy_ols();
        let err = run_fit(
            "y ~ x",
            numeric,
            factor,
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            None,
            Some(vec!["nope".to_string()]),
            None,
        )
        .unwrap_err();
        assert!(err.contains("nope"), "{err}");
    }

    #[test]
    fn malformed_formula_panic_becomes_a_clean_error_not_a_process_abort() {
        let (numeric, factor) = toy_ols();
        let err = run_fit(
            "y ~ :", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
        )
        .unwrap_err();
        // lower()'s panic (an unguarded index in glmm-formula's interaction-term
        // handling) must become a clean Err via catch_unwind, not abort the process.
        assert!(!err.is_empty());
    }

    #[test]
    fn mismatched_column_lengths_is_a_clean_error() {
        let mut numeric = std::collections::HashMap::new();
        numeric.insert("y".to_string(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        numeric.insert("x".to_string(), vec![0.0, 1.0]); // ragged: 2 vs 5
        let err = run_fit(
            "y ~ x",
            numeric,
            std::collections::HashMap::new(),
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(err.contains("x"), "{err}");
        assert!(err.contains('5') && err.contains('2'), "{err}");
    }

    #[test]
    fn unused_longer_column_does_not_silently_inflate_n() {
        let (mut numeric, factor) = toy_ols();
        numeric.insert("junk".to_string(), vec![0.0; 1000]); // not referenced by the formula
        let err = run_fit(
            "y ~ x", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
        )
        .unwrap_err();
        assert!(err.contains("junk"), "{err}");
    }
}
