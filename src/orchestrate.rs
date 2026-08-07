//! String-typed fit orchestration shared by the FFI ports (`glmm-python`,
//! `glmm-r`): formula + data -> [`crate::formula::lower`] -> option overrides
//! -> [`crate::fit_warm`], wrapped in `catch_unwind` so the kernel's
//! `assert!`-based boundary faults become a normal `Err`, never a process
//! abort across an FFI boundary. Plain-Rust types only — no `pyo3`/`extendr`
//! types — so everything here runs under plain `cargo test`, and the two
//! ports flatten the same fit the same way from one definition instead of
//! two mirrored copies.
//!
//! Behind the default-off `orchestrate` cargo feature. Like `loop_advanced`
//! it carries NO semver guarantees: its shape follows the ports' needs.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::formula::{label_ranef, lower, Column, Table};
use crate::{fit_warm, Boundary, Family, Note, StartValues, WaldSe};
use crate::{BinomialLink, GammaLink, NegBinomialLink, PoissonLink};

/// Maps the ports' `family`/`link` strings to [`Family`]. `family` and `link`
/// are already validated against the full (0.1.1-aware) spec table by each
/// port's wrapper layer (`glmm/__init__.py`, `r/R/fastglmm.R`) before this
/// ever runs — the "not yet implemented in the kernel" arms below are the
/// FFI-boundary backstop for the two 0.1.1 additions (`inversegaussian`,
/// `cloglog`) the current `Family`/`BinomialLink` enums have no variant for;
/// every other `Err` arm is unreachable via the ports and exists only for
/// direct callers.
pub fn family_from_str(family: &str, link: &str) -> Result<Family, String> {
    match family {
        "gaussian" => Ok(Family::Gaussian),
        "binomial" => match link {
            "logit" => Ok(Family::Binomial {
                link: BinomialLink::Logit,
            }),
            "probit" => Ok(Family::Binomial {
                link: BinomialLink::Probit,
            }),
            "cloglog" => Err(
                "link 'cloglog' requires GLMM 0.1.1; not yet implemented in the kernel".to_string(),
            ),
            other => Err(format!("unsupported binomial link {other:?}")),
        },
        "poisson" => match link {
            "log" => Ok(Family::Poisson {
                link: PoissonLink::Log,
            }),
            other => Err(format!("unsupported poisson link {other:?}")),
        },
        "gamma" => match link {
            "log" => Ok(Family::Gamma {
                link: GammaLink::Log,
            }),
            "inverse" => Ok(Family::Gamma {
                link: GammaLink::Inverse,
            }),
            other => Err(format!("unsupported gamma link {other:?}")),
        },
        "negativebinomial" => match link {
            "log" => Ok(Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            }),
            other => Err(format!("unsupported negativebinomial link {other:?}")),
        },
        "inversegaussian" => Err(
            "family 'inversegaussian' requires GLMM 0.1.1; not yet implemented \
             in the kernel"
                .to_string(),
        ),
        other => Err(format!("unknown family {other:?}")),
    }
}

/// One [`Note`], flattened for the ports. `kind` is the stable identifier each
/// port keys its warning category (Python) or condition class (R) off — the
/// English message is built there, not here.
///
/// `note_infos` matches [`Note`] exhaustively (in-crate, `#[non_exhaustive]`
/// does not bind), so a new variant fails compilation until it is given a
/// flattening — no note can silently drop. The ports' Python/R layers still
/// keep an unrecognized-`kind` fallback of their own.
#[derive(Debug)]
pub struct NoteInfo {
    /// The stable note identifier (see the struct docs).
    pub kind: &'static str,
    /// Fixed-effect column indices, 0-based into the fitted names (the R shim
    /// adds 1 at its boundary). Carries only the column the kernel detected —
    /// the columns it is entangled with are not identified. Empty for a
    /// variant that names no column.
    pub columns: Vec<u32>,
    /// The scaled pivot behind the note; `NaN` for a variant that carries none.
    pub pivot: f64,
    /// `PirlsExhausted` payload: how many fit-path evals hit the inner-PIRLS
    /// cap (that variant's `evals`). 0 for every other variant.
    pub evals: u32,
    /// `PirlsExhausted` payload: whether the final re-evaluation at the
    /// converged fit itself hit the cap, so the truncated solve's ũ/W̃ feed
    /// the reported estimates (that variant's `final_eval`) — the case the
    /// ports' warnings must distinguish from a rejected trial point. `false`
    /// for every other variant.
    pub final_eval: bool,
    /// Free text a variant needs and the fields above cannot carry (the
    /// grouping and level names of `UnusedGroupingLevels`). Empty otherwise;
    /// the `kind`, not this string, stays the stable identifier.
    pub detail: String,
}

/// One `crate::formula::RanefBlock` flattened for the ports:
/// `(grouping name, term names, level labels, row-major values)`.
pub type RanefBlockTuple = (String, Vec<String>, Vec<String>, Vec<f64>);

/// [`Boundary`] as the string the ports publish. Exhaustive for the same
/// reason `note_infos` is: a new variant must pick its string here.
fn boundary_name(boundary: Boundary) -> &'static str {
    match boundary {
        Boundary::Interior => "interior",
        Boundary::AtBoundary => "at_boundary",
        Boundary::NoOptimum => "no_optimum",
    }
}

fn note_infos(notes: Vec<Note>) -> Vec<NoteInfo> {
    notes
        .into_iter()
        .map(|note| match note {
            Note::IllConditioned { columns, pivot } => NoteInfo {
                kind: "ill_conditioned",
                columns,
                pivot,
                evals: 0,
                final_eval: false,
                detail: String::new(),
            },
            Note::PirlsExhausted { evals, final_eval } => NoteInfo {
                kind: "pirls_exhausted",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals,
                final_eval,
                detail: String::new(),
            },
            Note::UnusedGroupingLevels { grouping, levels } => NoteInfo {
                kind: "unused_grouping_levels",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: format!("{grouping}: {}", levels.join(", ")),
            },
        })
        .collect()
}

/// Everything a port publishes about one fit: [`crate::Fit`] plus the
/// lowering's naming, flattened to plain vectors, tuples, and strings the
/// FFI shims marshal field by field.
#[derive(Debug)]
pub struct FitResult {
    /// Fixed-effect estimates (`crate::Fit::beta`).
    pub beta: Vec<f64>,
    /// Wald standard errors per fixed effect (`crate::Fit::se`).
    pub se: Vec<f64>,
    /// Full p×p fixed-effect covariance (`crate::Fit::vcov`) — the off-diagonals
    /// `se` alone cannot carry, which any hand-built Wald contrast needs.
    pub vcov: Vec<Vec<f64>>,
    /// Variance components (`crate::Fit::tau2`).
    pub tau2: Vec<f64>,
    /// Per-grouping variance/correlation blocks (`crate::Fit::varcorr`).
    pub varcorr: Vec<Vec<f64>>,
    /// Standard errors of the RE standard deviations (`crate::Fit::stddev_se`).
    pub stddev_se: Vec<f64>,
    /// Which fixed-effect columns were dropped as exactly redundant
    /// (`crate::Diagnostics::aliased`).
    pub aliased: Vec<bool>,
    /// Dispersion/scale estimate (`crate::Fit::dispersion`).
    pub dispersion: f64,
    /// Optimizer convergence flag (`crate::Diagnostics::converged`).
    pub converged: bool,
    /// Optimizer evaluation count (`crate::Fit::n_eval`).
    pub n_eval: usize,
    /// Minimized optimizer criterion (`crate::Fit::deviance`) — carries that
    /// field's not-comparable-across-models caveat.
    pub deviance: f64,
    /// `true` iff the fit converged onto a variance-component boundary
    /// (mirrors `crate::Diagnostics::singular` / lme4's `isSingular`).
    pub singular: bool,
    /// Fixed-effect column names from the lowering.
    pub names: Vec<String>,
    /// Per-grouping `(name, term_names)` from `crate::formula::Lowered::re_groups`,
    /// in `varcorr` block order (primary, then each extra in declaration order)
    /// — `ReGroupInfo` flattened for the tuple. Without it a port's `summary()`
    /// has no grouping name to print and falls back to `group 0`.
    pub re_groups: Vec<(String, Vec<String>)>,
    /// Warn-and-strip message for an ineligible-shape `nagq>1` (the fit
    /// proceeded with Laplace); surfaced by each port as a Python
    /// `UserWarning` / R `warning()`.
    pub agq_warning: Option<String>,
    /// Log-likelihood at the fitted parameters (`crate::Fit::loglik`).
    pub loglik: f64,
    /// Parameters counted for AIC/BIC (`crate::Fit::df`).
    pub df: usize,
    /// `true` iff `loglik` is a REML criterion, not an ML log-likelihood
    /// (`crate::Fit::reml`).
    pub reml: bool,
    /// Fitted means per row (`crate::Fit::fitted`).
    pub fitted: Vec<f64>,
    /// Random-effect conditional modes (`crate::Fit::ranef`).
    pub ranef: Vec<f64>,
    /// Level count per grouping, for slicing `ranef` (`crate::Fit::ranef_levels`).
    pub ranef_levels: Vec<usize>,
    /// The same conditional modes, LABELLED — `crate::formula::label_ranef`'s
    /// blocks flattened for the ports: per grouping, `(name, term names, level
    /// labels, row-major values)`. Padded nested slots are already dropped, so
    /// `values.len() == levels.len() * terms.len()`. Empty exactly when `ranef`
    /// is. A port does NOT slice `ranef` itself: which layout a grouping lands
    /// in is a data-dependent routing decision inside the kernel, so only the
    /// crate can label it.
    pub ranef_blocks: Vec<RanefBlockTuple>,
    /// Where the accepted θ sits, from `crate::Diagnostics::boundary`; see
    /// `boundary_name` for the vocabulary.
    pub boundary: &'static str,
    /// Which variance components the optimizer pinned at 0, aligned with the
    /// `varcorr` blocks (`crate::Diagnostics::pinned`). **Empty means nothing
    /// was pinned** — every route with variance components fills this on a
    /// converged fit, and a model with no variance components at all (OLS,
    /// GLM, fixed-effect-only negative binomial) leaves it empty too.
    pub pinned: Vec<Vec<bool>>,
    /// Solver observations with no dedicated field (`crate::Diagnostics::notes`).
    pub notes: Vec<NoteInfo>,
}

/// Fit `formula` against pre-marshalled columns and return the flattened
/// [`FitResult`], with every kernel panic caught and returned as `Err` — the
/// one entry point both FFI ports call.
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
    // Factors arrive pre-coded: the port supplies the level order (e.g. a
    // pandas Categorical's `categories`, or the sorted distinct labels of a
    // plain string column), so the caller's reference level survives to here
    // rather than being re-derived by a sort. An out-of-range code would index
    // past `levels` inside `materialize`, so it is rejected at the boundary.
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
    // nagq>1 on an ineligible shape is valid-but-inapplicable — warn-and-strip:
    // nothing inapplicable may reach the kernel, whose shape check is an
    // `assert!`, and a Rust panic across an FFI boundary is not an acceptable
    // user error. The shape is only knowable here, after `lower()`, so this is
    // the strip site; each port emits the message (Python `UserWarning`, R
    // `warning()`). Eligibility mirrors
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

    // Taken before the fit so the borrow below is clean; these are the
    // LOWERING's observations, decided before any solver ran.
    let lowered_notes = std::mem::take(&mut lowered.notes);

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
    // component in a port's `summary()`, which reads as a wrong answer, not a
    // cosmetic slip. Non-mixed fits leave both empty, so the check holds there
    // too. Labelled BEFORE `re_groups` is flattened — `label_ranef` needs the
    // slot labels, which the tuple form drops. A shape mismatch here is a
    // kernel bug, not a user error, so it surfaces as an `Err` rather than
    // being swallowed.
    let ranef_blocks: Vec<RanefBlockTuple> = label_ranef(&fit, &lowered.re_groups)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|b| (b.group, b.terms, b.levels, b.values))
        .collect();
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

    // Taken out whole before the field-by-field move below: the forwarding
    // accessors borrow all of `fit`, which a partial move rules out.
    let diagnostics = fit.diagnostics;
    Ok(FitResult {
        beta: fit.beta,
        se: fit.se,
        vcov: fit.vcov,
        tau2: fit.tau2,
        varcorr: fit.varcorr,
        stddev_se: fit.stddev_se,
        aliased: diagnostics.aliased,
        dispersion: fit.dispersion,
        converged: diagnostics.converged,
        n_eval: fit.n_eval,
        deviance: fit.deviance,
        singular: diagnostics.singular,
        names: lowered.col_names,
        re_groups,
        agq_warning,
        loglik: fit.loglik,
        df: fit.df,
        reml: fit.reml,
        fitted: fit.fitted,
        ranef: fit.ranef,
        ranef_levels: fit.ranef_levels,
        ranef_blocks,
        boundary: boundary_name(diagnostics.boundary),
        pinned: diagnostics.pinned,
        // The lowering's own observations (unused grouping levels) join the
        // solver's in one channel: the user does not care which layer noticed.
        notes: note_infos(lowered_notes.into_iter().chain(diagnostics.notes).collect()),
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

    #[test]
    fn gaussian_maps() {
        assert_eq!(
            family_from_str("gaussian", "identity"),
            Ok(Family::Gaussian)
        );
    }

    #[test]
    fn binomial_logit_and_probit_map() {
        assert_eq!(
            family_from_str("binomial", "logit"),
            Ok(Family::Binomial {
                link: BinomialLink::Logit
            })
        );
        assert_eq!(
            family_from_str("binomial", "probit"),
            Ok(Family::Binomial {
                link: BinomialLink::Probit
            })
        );
    }

    #[test]
    fn binomial_cloglog_is_a_kernel_gap() {
        let err = family_from_str("binomial", "cloglog").unwrap_err();
        assert!(err.contains("not yet implemented in the kernel"), "{err}");
    }

    #[test]
    fn poisson_maps() {
        assert_eq!(
            family_from_str("poisson", "log"),
            Ok(Family::Poisson {
                link: PoissonLink::Log
            })
        );
    }

    #[test]
    fn gamma_log_and_inverse_map() {
        assert_eq!(
            family_from_str("gamma", "log"),
            Ok(Family::Gamma {
                link: GammaLink::Log
            })
        );
        assert_eq!(
            family_from_str("gamma", "inverse"),
            Ok(Family::Gamma {
                link: GammaLink::Inverse
            })
        );
    }

    #[test]
    fn negativebinomial_maps() {
        assert_eq!(
            family_from_str("negativebinomial", "log"),
            Ok(Family::NegativeBinomial {
                link: NegBinomialLink::Log
            })
        );
    }

    #[test]
    fn inversegaussian_is_a_kernel_gap() {
        let err = family_from_str("inversegaussian", "log").unwrap_err();
        assert!(err.contains("not yet implemented in the kernel"), "{err}");
    }

    #[test]
    fn pirls_exhausted_payload_survives_flattening() {
        // No known dataset reaches `final_eval == true` end-to-end, so the
        // ports' message branch is asserted from constructed notes; this pins
        // the payload they branch on.
        let notes = note_infos(vec![
            Note::PirlsExhausted {
                evals: 3,
                final_eval: false,
            },
            Note::PirlsExhausted {
                evals: 0,
                final_eval: true,
            },
        ]);
        assert_eq!(notes[0].kind, "pirls_exhausted");
        assert_eq!(notes[0].evals, 3);
        assert!(!notes[0].final_eval);
        assert_eq!(notes[1].kind, "pirls_exhausted");
        assert_eq!(notes[1].evals, 0);
        assert!(notes[1].final_eval);
    }

    /// A factor column in the `(levels, codes)` form `run_fit` takes, with the
    /// lexicographic level order a plain string column gets from the ports.
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
        // and report the warn-and-strip message, never surface the kernel's
        // shape panic.
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
