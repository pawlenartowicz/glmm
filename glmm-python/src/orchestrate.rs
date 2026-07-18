//! Orchestrates one fit: formula + data -> `glmm::formula::lower` -> option
//! overrides -> `glmm::fit_warm`, wrapped in `catch_unwind` so the kernel's
//! `assert!`-based boundary faults become a normal `Err`, never a process
//! abort across the FFI. No `pyo3` types here — plain-Rust, unit-testable.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use glmm::formula::{lower, Column, Table};
use glmm::{fit_warm, Family, StartValues, WaldSe};

use crate::convert::family_from_str;

#[derive(Debug)]
pub struct FitResult {
    pub beta: Vec<f64>,
    pub se: Vec<f64>,
    /// Full p×p fixed-effect covariance (`glmm::Fit::vcov`) — the off-diagonals
    /// `se` alone cannot carry, which any hand-built Wald contrast needs.
    pub vcov: Vec<Vec<f64>>,
    pub tau2: Vec<f64>,
    pub varcorr: Vec<Vec<f64>>,
    pub stddev_se: Vec<f64>,
    pub aliased: Vec<bool>,
    pub dispersion: f64,
    pub converged: bool,
    /// Optimizer evaluation count (`glmm::Fit::n_eval`).
    pub n_eval: usize,
    /// Minimized optimizer criterion (`glmm::Fit::deviance`) — carries that
    /// field's not-comparable-across-models caveat; see `glmm/__init__.py`.
    pub deviance: f64,
    /// `true` iff the fit converged onto a variance-component boundary
    /// (mirrors `glmm::Fit::singular` / lme4's `isSingular`) — the flag was
    /// already computed but never surfaced through this wrapper until now.
    pub singular: bool,
    pub names: Vec<String>,
    /// Per-grouping `(name, term_names)` from `glmm::formula::Lowered::re_groups`,
    /// in `varcorr` block order (primary, then each extra in declaration order)
    /// — `ReGroupInfo` flattened for the tuple. Without it `summary()` has no
    /// grouping name to print and falls back to `group 0`.
    pub re_groups: Vec<(String, Vec<String>)>,
    /// §3.5 warn-and-strip message for an ineligible-shape `nagq>1` (the fit
    /// proceeded with Laplace); surfaced by `glmm.fit` as a `UserWarning`.
    pub agq_warning: Option<String>,
    /// Log-likelihood at the fitted parameters (`glmm::Fit::loglik`).
    pub loglik: f64,
    /// Parameters counted for AIC/BIC (`glmm::Fit::df`).
    pub df: usize,
    /// `true` iff `loglik` is a REML criterion, not an ML log-likelihood
    /// (`glmm::Fit::reml`).
    pub reml: bool,
    /// Fitted means per row (`glmm::Fit::fitted`).
    pub fitted: Vec<f64>,
    /// Random-effect conditional modes (`glmm::Fit::ranef`).
    pub ranef: Vec<f64>,
    /// Level count per grouping, for slicing `ranef` (`glmm::Fit::ranef_levels`).
    pub ranef_levels: Vec<usize>,
}

#[allow(clippy::too_many_arguments)]
pub fn run_fit(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: &str,
    link: &str,
    wald_se: &str,
    nagq: u8,
    dispersion: Option<f64>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    warm_start: Option<(Vec<f64>, Vec<f64>)>,
) -> Result<FitResult, String> {
    let fam = family_from_str(family, link)?;

    let mut columns: Vec<(String, Column)> = Vec::new();
    let mut lengths: Vec<(String, usize)> = Vec::new();
    for (name, values) in numeric_columns {
        lengths.push((name.clone(), values.len()));
        columns.push((name, Column::Numeric(values)));
    }
    // Factors arrive pre-coded: `glmm.fit` supplies the level order (a pandas
    // Categorical's `categories`, or the sorted distinct labels of a plain
    // string column), so the caller's reference level survives to here rather
    // than being re-derived by a sort. An out-of-range code would index past
    // `levels` inside `materialize`, so it is rejected at the boundary.
    for (name, (levels, codes)) in factor_columns {
        if let Some(&bad) = codes.iter().find(|&&c| c as usize >= levels.len()) {
            return Err(format!(
                "factor column {name:?}: code {bad} is out of range for {} levels",
                levels.len()
            ));
        }
        lengths.push((name.clone(), codes.len()));
        columns.push((name, Column::Factor { levels, codes }));
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
    lowered.opts.offset = offset;

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

    // `varcorr` and `re_groups` are both emitted in RE declaration order
    // (primary, then each extra), so index i of one names index i of the other.
    // Assert rather than trust it: a silent misalignment relabels a variance
    // component in `summary()`, which reads as a wrong answer, not a cosmetic
    // slip. Non-mixed fits leave both empty, so the check holds there too.
    let re_groups: Vec<(String, Vec<String>)> = lowered
        .re_groups
        .into_iter()
        .map(|g| (g.name, g.terms))
        .collect();
    assert_eq!(
        re_groups.len(),
        fit.varcorr.len(),
        "re_groups and varcorr must agree in length and order"
    );

    Ok(FitResult {
        beta: fit.beta,
        se: fit.se,
        vcov: fit.vcov,
        tau2: fit.tau2,
        varcorr: fit.varcorr,
        stddev_se: fit.stddev_se,
        aliased: fit.aliased,
        dispersion: fit.dispersion,
        converged: fit.converged,
        n_eval: fit.n_eval,
        deviance: fit.deviance,
        singular: fit.singular,
        names: lowered.col_names,
        re_groups,
        agq_warning,
        loglik: fit.loglik,
        df: fit.df,
        reml: fit.reml,
        fitted: fit.fitted,
        ranef: fit.ranef,
        ranef_levels: fit.ranef_levels,
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

    /// A factor column in the `(levels, codes)` form `run_fit` takes, with the
    /// lexicographic level order a plain string column gets from `glmm.fit`.
    fn factor_col(labels: &[&str]) -> (Vec<String>, Vec<u32>) {
        let mut levels: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        levels.sort();
        levels.dedup();
        let codes = labels
            .iter()
            .map(|l| levels.iter().position(|v| v == l).unwrap() as u32)
            .collect();
        (levels, codes)
    }

    #[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
    fn toy_ols() -> (
        HashMap<String, Vec<f64>>,
        HashMap<String, (Vec<String>, Vec<u32>)>,
    ) {
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
        let g: Vec<&str> = ["a", "b", "c", "d", "e"]
            .iter()
            .copied()
            .cycle()
            .take(10)
            .collect();
        factor.insert("g".to_string(), factor_col(&g));
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
    fn malformed_formula_panic_becomes_a_clean_error_not_a_process_abort() {
        let (numeric, factor) = toy_ols();
        let err = run_fit(
            "y ~ :", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
        )
        .unwrap_err();
        // lower()'s panic (an unguarded index in the formula frontend's interaction-term
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
