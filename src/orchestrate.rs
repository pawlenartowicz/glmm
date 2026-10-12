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

use crate::fit::theta_width;
use crate::formula::{
    fixed_part_relation, label_ranef, lower, parse, Column, FixedPartRelation, Table,
};
use crate::inference::contrast::{ContrastError, NoDfReason};
use crate::inference::tails::{normal_quantile, t_quantile};
use crate::{fit_warm, Boundary, Family, Fit, Note, OuterStop, StartValues, WaldSe};
use crate::{BinomialLink, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink};

/// Maps the ports' `family`/`link` strings to [`Family`]. `family` and `link`
/// are already validated against the full spec table by each port's wrapper
/// layer (`glmm/__init__.py`, `r/R/fastglmm.R`) before this ever runs — every
/// family and link in that table now maps to a `Family`/`BinomialLink`
/// variant, so every `Err` arm below is unreachable via the ports and exists
/// only for direct callers.
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
            "cloglog" => Ok(Family::Binomial {
                link: BinomialLink::Cloglog,
            }),
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
        "inversegaussian" => match link {
            "log" => Ok(Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            }),
            "inverse_squared" => Ok(Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            }),
            other => Err(format!("unsupported inversegaussian link {other:?}")),
        },
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
    /// cap (that variant's `evals`). Also carries `NbShapeUnsettled`'s
    /// `rounds` and `NonIntegerResponse`'s `rows`, which need the same "small
    /// count" shape and no variant needs more than one at once. 0 for every
    /// other variant.
    pub evals: u32,
    /// `PirlsExhausted` payload: whether the final re-evaluation at the
    /// converged fit itself hit the cap, so the truncated solve's ũ/W̃ feed
    /// the reported estimates (that variant's `final_eval`) — the case the
    /// ports' warnings must distinguish from a rejected trial point. `false`
    /// for every other variant.
    pub final_eval: bool,
    /// Free text a variant needs and the fields above cannot carry (the
    /// grouping and level names of `UnusedGroupingLevels`, the grouping name
    /// of `SingleLevelGroupingDropped`). Empty otherwise; the `kind`, not this
    /// string, stays the stable identifier.
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

/// [`OuterStop`] as the string the ports publish, exhaustive like
/// `boundary_name`.
fn outer_stop_name(stop: OuterStop) -> &'static str {
    match stop {
        OuterStop::Fdelta => "fdelta",
        OuterStop::Pgrad => "pgrad",
        OuterStop::NoiseFloor => "noise_floor",
        OuterStop::LineSearch => "line_search",
        OuterStop::MaxIter => "max_iter",
        OuterStop::GradUnavailable => "grad_unavailable",
        OuterStop::NonFinite => "non_finite",
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
            Note::HessianSeFallback => NoteInfo {
                kind: "hessian_se_fallback",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: String::new(),
            },
            Note::NbShapeUnsettled { rounds } => NoteInfo {
                kind: "nb_shape_unsettled",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: rounds,
                final_eval: false,
                detail: String::new(),
            },
            Note::NonIntegerResponse { rows } => NoteInfo {
                kind: "non_integer_response",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: rows,
                final_eval: false,
                detail: String::new(),
            },
            Note::SingleLevelGroupingDropped { grouping } => NoteInfo {
                kind: "single_level_grouping_dropped",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: grouping,
            },
            Note::ExactProfileFallback => NoteInfo {
                kind: "exact_profile_fallback",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: String::new(),
            },
            Note::GradientSearchFallback => NoteInfo {
                kind: "gradient_search_fallback",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: String::new(),
            },
            Note::SeUnavailable => NoteInfo {
                kind: "se_unavailable",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: String::new(),
            },
            Note::PrecisionLimited => NoteInfo {
                kind: "precision_limited",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: String::new(),
            },
            Note::Separated => NoteInfo {
                kind: "separated",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: String::new(),
            },
            Note::SatterthwaiteCurvature {
                dropped,
                negative,
                converged,
                nonfinite,
            } => NoteInfo {
                kind: "satterthwaite_curvature",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: 0,
                final_eval: false,
                detail: format!(
                    "dropped {dropped}, negative {negative}, converged {converged}, nonfinite {nonfinite}"
                ),
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
    /// Gradient evaluations of the BFGS θ search (`crate::Diagnostics::n_grad`).
    pub n_grad: usize,
    /// Why the BFGS θ search stopped, `None` when none ran
    /// (`crate::Diagnostics::outer_stop`); see `outer_stop_name` for the
    /// vocabulary.
    pub outer_stop: Option<&'static str>,
    /// BFGS decrement at the search's endpoint
    /// (`crate::Diagnostics::outer_decrement`).
    pub outer_decrement: f64,
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
    /// Set when `reml = Some(true)` was given on a non-Gaussian family (GLM or
    /// GLMM), where the argument has no meaning: it was ignored and the model
    /// fitted by ML, as under `None`. Each port raises its own warning, with
    /// its own spelling of the argument, when this is set.
    pub reml_warning: Option<String>,
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
    /// The response as the kernel fitted it (`Lowered::y`): after lowering, so
    /// a `cbind(s, f)` response is the proportion `s/(s+f)` and a formula
    /// transform on the LHS is already applied. Retained so a port can compute
    /// residuals and the summary's scaled-residual quartiles without a second
    /// copy of the data. Coupled sites — change together: `glmm-python`'s
    /// `fit_dict` and `glmm-r`'s returned list flatten every field here.
    pub y: Vec<f64>,
    /// Prior weights the kernel fitted with, `None` when unweighted. For a
    /// `cbind(s, f)` response these are the trial counts `s + f` the lowering
    /// made up, which the caller never passed — the ports' Pearson residuals
    /// need them, and `weights=` alone cannot say.
    pub weights: Option<Vec<f64>>,
    /// Row count the kernel fitted (`Lowered::n`). NOT `fitted.len()`: `fitted`
    /// is empty on a non-converged fit and the ports' headers print on those.
    pub nobs: usize,
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
    /// Degrees of freedom per coefficient (`crate::Fit::coef_df`): the
    /// Satterthwaite df on a Gaussian LMM, `n − p` on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated, NaN where the test is z and on
    /// an OLS or dispersion GLM fit that did not converge (still a t test,
    /// `df_method` `"residual"`).
    pub coef_df: Vec<f64>,
    /// The t test's two-sided p-value per coefficient, from
    /// `crate::Fit::contrast_t` on the unit vector; NaN where there is no df.
    pub coef_p: Vec<f64>,
    /// Where `coef_df` comes from: `"satterthwaite"` (Gaussian LMM),
    /// `"residual"` (OLS and the dispersion GLMs with φ estimated, whatever
    /// `satterthwaite` says), `"none"` (the families with no df),
    /// `"disabled"` (a Gaussian LMM fitted with `satterthwaite = false`) or
    /// `"failed"` (a Gaussian LMM whose Satterthwaite pass failed). The ports
    /// print t (with a df column on `"satterthwaite"`) on the first two and z
    /// on the rest.
    pub df_method: &'static str,
    /// Weighted residual sum of squares `Σwᵢ(yᵢ − μ̂ᵢ)²` of an OLS fit; NaN on
    /// every other fit and on a non-converged one.
    pub rss: f64,
    /// `summary.lm`'s multiple R² of an OLS fit (0 on an intercept-only
    /// model); NaN on every other fit and on a non-converged one.
    pub r_squared: f64,
    /// `summary.lm`'s adjusted R², with the same NaN and 0 cases as
    /// `r_squared`.
    pub adj_r_squared: f64,
}

/// Fit `formula` against pre-marshalled columns and return the flattened
/// [`FitResult`], with every kernel panic caught and returned as `Err` — the
/// one entry point both FFI ports call. `satterthwaite` is
/// `crate::FitOptions::satterthwaite`: `false` skips the Gaussian LMM's
/// Satterthwaite pass (its df are then not computed and it is tested with z).
///
/// `reml` is resolved on the lowered model, after a single-level grouping has
/// been dropped:
/// - a non-Gaussian family (GLM or GLMM): `Some(true)` is ignored, the model is
///   fitted by ML exactly as under `None`, and [`FitResult::reml_warning`] is
///   set; `Some(false)` gives the `None` fit with no warning;
/// - Gaussian with no random part (OLS): `Some(true)` gives the same estimates
///   and reports the REML log-likelihood; `Some(false)` gives the `None` fit;
/// - Gaussian mixed model: `None` and `Some(true)` fit by REML, `Some(false)`
///   by ML.
///
/// Every value but the stripped one reaches `crate::FitOptions::reml`.
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
    warm_start: Option<(Vec<f64>, Vec<f64>, Option<f64>)>,
    satterthwaite: bool,
    reml: Option<bool>,
) -> Result<FitResult, String> {
    let fam = family_from_str(family, link)?;
    // `fit_warm`'s own checks of `dispersion` and the warm start name kernel
    // struct fields (`FitOptions.dispersion`, `StartValues.beta`) a port user
    // never sees, so both are checked here first under the argument's name.
    if let Some(d) = dispersion {
        if !(d.is_finite() && d > 0.0) {
            return Err(format!("dispersion must be a finite number > 0, got {d}"));
        }
    }

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
    // a mixed binomial/Poisson/negative-binomial/Gamma model with a single grouping factor and
    // q_p = 1 + #slopes ≤ 3 (the temporary cost/oracle cap).
    let mut nagq = nagq;
    let mut agq_warning: Option<String> = None;
    if nagq > 1 {
        let agq_family = matches!(
            fam,
            Family::Binomial { .. }
                | Family::Poisson { .. }
                | Family::NegativeBinomial { .. }
                | Family::Gamma { .. }
        );
        let eligible = match lowered.model.re.as_ref() {
            Some(re) => {
                let q_p = 1 + re.slopes.len(); // intercept + slopes, as in assert_model_shape
                agq_family && re.extra_groupings.is_empty() && q_p <= 3
            }
            None => false,
        };
        if !eligible {
            agq_warning = Some(format!(
                "nagq={nagq} (adaptive quadrature) applies only to binomial/Poisson/negative-binomial/Gamma \
                 mixed models with a single grouping factor and at most 3 random \
                 effects per group; fitting with Laplace (nagq=1)"
            ));
            nagq = 1;
        }
    }
    lowered.opts.nagq = nagq;
    // `reml` is resolved here too, on the lowered model, so both ports follow
    // one rule; a single-level grouping the lowering dropped has already made a
    // mixed formula an OLS or GLM fit. A GLM or GLMM has no REML/ML split:
    // `Some(true)` there is stripped with a warning (the fit is ML, the `None`
    // fit), and `Some(false)` is the `None` fit, silently. Every other value
    // goes to `FitOptions::reml`, which resolves it on a Gaussian model.
    let mut reml_warning: Option<String> = None;
    lowered.opts.reml = match reml {
        Some(true) if lowered.model.family != Family::Gaussian => {
            reml_warning = Some(
                "reml=true applies only to Gaussian models; it was ignored and the model \
                 fitted by maximum likelihood"
                    .to_string(),
            );
            None
        }
        other => other,
    };
    lowered.opts.dispersion = dispersion;
    lowered.opts.satterthwaite = satterthwaite;
    // Mirrors `formula::materialize`'s `opts` construction — change together:
    // lowering sets `weights`/`offset` only for a `cbind()` response /
    // `offset()` formula term, so a value here AND the matching argument is
    // the user saying it twice.
    if weights.is_some() && lowered.opts.weights.is_some() {
        return Err(
            "weights given both as a cbind() response and as the weights= argument; use one"
                .to_string(),
        );
    }
    if weights.is_some() {
        lowered.opts.weights = weights;
    }
    if offset.is_some() && lowered.opts.offset.is_some() {
        return Err(
            "offset given both as an offset() formula term and as the offset= argument; use one"
                .to_string(),
        );
    }
    if offset.is_some() {
        lowered.opts.offset = offset;
    }

    // Taken before the fit so the borrow below is clean; these are the
    // LOWERING's observations, decided before any solver ran.
    let lowered_notes = std::mem::take(&mut lowered.notes);

    // Widths as `fit_warm` checks them: exact, or empty to cold-start that part.
    if let Some((beta, theta, disp)) = &warm_start {
        if let Some(d) = disp {
            if !(d.is_finite() && *d > 0.0) {
                return Err(format!(
                    "warm-start dispersion must be a finite number > 0, got {d}"
                ));
            }
        }
        if !beta.is_empty() && beta.len() != lowered.p {
            return Err(format!(
                "warm-start beta has {} entries but the model has {} fixed-effect \
                 coefficients; give {}, or none to cold-start beta",
                beta.len(),
                lowered.p,
                lowered.p
            ));
        }
        let n_theta = theta_width(lowered.model.re.as_ref());
        if !theta.is_empty() && theta.len() != n_theta {
            return Err(format!(
                "warm-start theta has {} entries but the model has {} random-effect \
                 Cholesky entries; give {}, or none to cold-start theta",
                theta.len(),
                n_theta,
                n_theta
            ));
        }
    }
    let start = warm_start.map(|(beta, theta, dispersion)| StartValues {
        beta,
        theta,
        dispersion,
    });

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

    let (coef_p, df_method) = t_tests(&fit, &lowered.opts.target_indices);
    let (rss, r_squared, adj_r_squared) = if lowered.model.re.is_none() && fam == Family::Gaussian {
        ols_r_squared(
            &fit,
            &lowered.y,
            lowered.opts.weights.as_deref(),
            lowered.opts.offset.as_deref(),
            lowered.has_intercept,
        )
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    // `varcorr` and `re_groups` are both emitted in RE declaration order
    // (primary, then each extra), so index i of one names index i of the other.
    // Check rather than trust it: a silent misalignment relabels a variance
    // component in a port's `summary()`, which reads as a wrong answer, not a
    // cosmetic slip. Labelled BEFORE `re_groups` is flattened — `label_ranef`
    // needs the slot labels, which the tuple form drops. A shape mismatch here
    // is a kernel bug, not a user error, so it surfaces as an `Err` rather than
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
    // An EMPTY `varcorr` is not a mismatch — it is the crate's numerical-failure
    // convention, "the fit assembled no variance components". Every failure
    // return uses it: the degenerate LMM endpoint (`fit/lmm.rs`, alongside its
    // NaN-filled `tau2`/`vcov`), a FAILED dense GLMM (`fit/glmm.rs`); a capped
    // dense GLMM now reports its finite endpoint's varcorr, non-empty, under
    // the plateau policy instead. Also the unfittable-random-slope return and
    // the sparse NaN return (`fit/common.rs`). `Fit::has_negligible_component` documents
    // the same reading, and both ports already gate their random-effects block
    // on `varcorr`'s length, so an empty one prints no RE section rather than
    // mislabelling one. Only a NON-empty block list of the wrong length is the
    // kernel bug this check is for. Non-mixed fits leave both empty.
    if !fit.varcorr.is_empty() && fit.varcorr.len() != re_groups.len() {
        return Err(format!(
            "re_groups and varcorr must agree in length and order: \
             {} grouping(s) lowered, {} varcorr block(s) returned",
            re_groups.len(),
            fit.varcorr.len()
        ));
    }

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
        n_grad: diagnostics.n_grad,
        outer_stop: diagnostics.outer_stop.map(outer_stop_name),
        outer_decrement: diagnostics.outer_decrement,
        deviance: fit.deviance,
        singular: diagnostics.singular,
        names: lowered.col_names,
        re_groups,
        agq_warning,
        reml_warning,
        loglik: fit.loglik,
        df: fit.df,
        reml: fit.reml,
        fitted: fit.fitted,
        ranef: fit.ranef,
        ranef_levels: fit.ranef_levels,
        ranef_blocks,
        y: lowered.y,
        weights: lowered.opts.weights,
        nobs: lowered.n,
        boundary: boundary_name(diagnostics.boundary),
        pinned: diagnostics.pinned,
        // The lowering's own observations (unused grouping levels) join the
        // solver's in one channel: the user does not care which layer noticed.
        notes: note_infos(lowered_notes.into_iter().chain(diagnostics.notes).collect()),
        coef_df: fit.coef_df,
        coef_p,
        df_method,
        rss,
        r_squared,
        adj_r_squared,
    })
}

/// The t test of each target coefficient (`Fit::contrast_t` on its unit
/// vector) as `(coef_p, df_method)`; see [`FitResult::df_method`]. The label
/// keys on the fit's df record, not on its notes: a failed pass is the
/// `NoDfReason::Failed` record.
pub(crate) fn t_tests(fit: &Fit, targets: &[u32]) -> (Vec<f64>, &'static str) {
    let p = fit.beta.len();
    let mut coef_p = vec![f64::NAN; p];
    let df_method = match fit.df_state() {
        Ok(st) if st.residual_only() => "residual",
        Ok(_) => "satterthwaite",
        Err(ContrastError::NoDf(reason)) => match reason {
            NoDfReason::NotGaussianMixed => "none",
            NoDfReason::Disabled => "disabled",
            // The not-run record is the loop tier's, which `run_fit` never
            // builds; were it to happen, its df are missing as after a failed
            // pass.
            NoDfReason::Failed | NoDfReason::StateMissing => "failed",
        },
    };
    if matches!(df_method, "residual" | "satterthwaite") {
        let mut e = vec![0.0; p];
        for &j in targets {
            let j = j as usize;
            e[j] = 1.0;
            if let Ok(t) = fit.contrast_t(&e) {
                coef_p[j] = t.p_value;
            }
            e[j] = 0.0;
        }
    }
    (coef_p, df_method)
}

/// `(rss, R², adjusted R²)` of an OLS fit in `summary.lm`'s form: `rss =
/// Σwᵢ(yᵢ − μ̂ᵢ)²`; `mss` from the fitted values less the offset, centred on
/// their weighted mean when the formula has an intercept and uncentred
/// otherwise; `R² = mss/(mss + rss)`, adjusted `1 − (1 − R²)(n − df.int)/(n −
/// p)` with `p` the retained columns. 0 for both R² when `p` is only the
/// intercept (or nothing); NaN for all three on a non-converged fit.
fn ols_r_squared(
    fit: &Fit,
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    has_intercept: bool,
) -> (f64, f64, f64) {
    let n = y.len();
    if !fit.converged() || fit.fitted.len() != n {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let w = |i: usize| weights.map_or(1.0, |w| w[i]);
    let f = |i: usize| fit.fitted[i] - offset.map_or(0.0, |o| o[i]);
    let rss: f64 = (0..n).map(|i| w(i) * (y[i] - fit.fitted[i]).powi(2)).sum();
    let rank = fit.aliased().iter().filter(|a| !**a).count();
    let df_int = usize::from(has_intercept);
    if rank == df_int {
        return (rss, 0.0, 0.0);
    }
    let mss: f64 = if has_intercept {
        let sw: f64 = (0..n).map(w).sum();
        let m = (0..n).map(|i| w(i) * f(i)).sum::<f64>() / sw;
        (0..n).map(|i| w(i) * (f(i) - m).powi(2)).sum()
    } else {
        (0..n).map(|i| w(i) * f(i).powi(2)).sum()
    };
    let r2 = mss / (mss + rss);
    let adj = 1.0 - (1.0 - r2) * ((n - df_int) as f64 / (n - rank) as f64);
    (rss, r2, adj)
}

/// Confidence intervals for the fixed effects, the one rule both ports'
/// `confint` call: per coefficient `β̂ ± q·se`, with `q` the Student t
/// quantile on `coef_df[i]` where that df is finite and positive (the rows
/// `summary()` tests with t: the Satterthwaite df of a Gaussian LMM, `n − p`
/// on OLS and the dispersion GLMs) and the standard normal quantile
/// otherwise (z rows, a pass switched off or failed). The quantiles are taken
/// at `(1 − level)/2` and `1 − (1 − level)/2`, as `confint.lm` takes them.
/// An interval is NaN only where `β̂` or `se` is (an aliased column).
///
/// # Errors
///
/// When `level` is not strictly between 0 and 1, or the three slices differ
/// in length.
pub fn confint(
    beta: &[f64],
    se: &[f64],
    coef_df: &[f64],
    level: f64,
) -> Result<Vec<[f64; 2]>, String> {
    if !(level > 0.0 && level < 1.0) {
        return Err(format!(
            "level must be a number strictly between 0 and 1, got {level}"
        ));
    }
    if se.len() != beta.len() || coef_df.len() != beta.len() {
        return Err(format!(
            "beta, se and coef_df must have the same length, got {}, {} and {}",
            beta.len(),
            se.len(),
            coef_df.len()
        ));
    }
    let a = (1.0 - level) / 2.0;
    Ok((0..beta.len())
        .map(|i| {
            let df = coef_df[i];
            let (lo, hi) = if df.is_finite() && df > 0.0 {
                (t_quantile(a, df), t_quantile(1.0 - a, df))
            } else {
                (normal_quantile(a), normal_quantile(1.0 - a))
            };
            [beta[i] + lo * se[i], beta[i] + hi * se[i]]
        })
        .collect())
}

/// Whether two formula strings have the same fixed part, for the ports'
/// `AIC` and `BIC` on two fits: [`crate::formula::fixed_part_relation`]
/// answers [`FixedPartRelation::Same`] (the same fixed terms, compared as sets
/// of variable names, and the same fixed intercept). The response, the offset
/// and the random part are not read.
///
/// # Errors
///
/// When either formula does not parse; the message quotes it.
pub fn same_fixed_part(f0: &str, f1: &str) -> Result<bool, String> {
    let parsed = |f: &str| parse(f).map_err(|e| format!("formula {f:?} does not parse: {e}"));
    Ok(fixed_part_relation(&parsed(f0)?, &parsed(f1)?) == FixedPartRelation::Same)
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
    use crate::inference::tails::t_two_sided_p;
    use std::collections::HashMap;

    #[test]
    fn family_and_link_strings_map_to_kernel_families() {
        let ok = [
            ("gaussian", "identity", Family::Gaussian),
            (
                "binomial",
                "logit",
                Family::Binomial {
                    link: BinomialLink::Logit,
                },
            ),
            (
                "binomial",
                "probit",
                Family::Binomial {
                    link: BinomialLink::Probit,
                },
            ),
            (
                "binomial",
                "cloglog",
                Family::Binomial {
                    link: BinomialLink::Cloglog,
                },
            ),
            (
                "poisson",
                "log",
                Family::Poisson {
                    link: PoissonLink::Log,
                },
            ),
            (
                "gamma",
                "log",
                Family::Gamma {
                    link: GammaLink::Log,
                },
            ),
            (
                "gamma",
                "inverse",
                Family::Gamma {
                    link: GammaLink::Inverse,
                },
            ),
            (
                "negativebinomial",
                "log",
                Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
            ),
            (
                "inversegaussian",
                "log",
                Family::InverseGaussian {
                    link: InverseGaussianLink::Log,
                },
            ),
            (
                "inversegaussian",
                "inverse_squared",
                Family::InverseGaussian {
                    link: InverseGaussianLink::InverseSquared,
                },
            ),
        ];
        for (family, link, expected) in ok {
            assert_eq!(
                family_from_str(family, link),
                Ok(expected),
                "{family}/{link}"
            );
        }
        let err = family_from_str("inversegaussian", "identity").unwrap_err();
        assert!(err.contains("unsupported inversegaussian link"), "{err}");
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

    #[test]
    fn note_payloads_survive_flattening() {
        let notes = note_infos(vec![
            Note::SingleLevelGroupingDropped {
                grouping: "Subject".to_string(),
            },
            Note::HessianSeFallback,
            Note::NbShapeUnsettled { rounds: 25 },
            Note::ExactProfileFallback,
        ]);
        assert_eq!(notes[0].kind, "single_level_grouping_dropped");
        assert_eq!(notes[0].detail, "Subject");
        assert_eq!(notes[1].kind, "hessian_se_fallback");
        assert_eq!(notes[1].detail, "");
        assert_eq!(notes[2].kind, "nb_shape_unsettled");
        assert_eq!(notes[2].evals, 25);
        assert_eq!(notes[3].kind, "exact_profile_fallback");
        assert_eq!(notes[3].detail, "");
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
            true, None,
        )
        .expect("fit should succeed");
        assert_eq!(
            result.names,
            vec!["(Intercept)".to_string(), "x".to_string()]
        );
        assert_eq!(result.beta.len(), 2);
        assert!(result.converged);
        // Closed-form OLS slope for this fixture: Sxy/Sxx with x̄=4.5, ȳ=5.53
        // gives 83.55/82.5 = 1.0127272727272727; the fit matches it to
        // machine precision (measured gap ~7e-16).
        assert!(
            (result.beta[1] - 1.0127272727272727).abs() < 1e-9,
            "slope = {}",
            result.beta[1]
        );
    }

    #[test]
    fn unknown_column_is_a_clean_error() {
        let (numeric, factor) = toy_ols();
        let err = run_fit(
            "y ~ z", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
            true, None,
        )
        .unwrap_err();
        assert_eq!(err, "unknown column: z");
    }

    #[test]
    fn offset_given_both_ways_is_a_clean_error() {
        let (numeric, factor) = toy_ols();
        let n = numeric["y"].len();
        let err = run_fit(
            "y ~ x + offset(x)",
            numeric,
            factor,
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            None,
            Some(vec![0.0; n]),
            None,
            true,
            None,
        )
        .unwrap_err();
        assert!(err.contains("use one"), "{err}");
    }

    #[test]
    fn weights_given_both_ways_is_a_clean_error() {
        let mut numeric = HashMap::new();
        numeric.insert("s".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
        numeric.insert("f".to_string(), vec![3.0, 2.0, 1.0, 0.5]);
        numeric.insert("x".to_string(), vec![0.0, 1.0, 2.0, 3.0]);
        let n = numeric["s"].len();
        let err = run_fit(
            "cbind(s, f) ~ x",
            numeric,
            HashMap::new(),
            "binomial",
            "logit",
            "hessian",
            1,
            None,
            Some(vec![1.0; n]),
            None,
            None,
            true,
            None,
        )
        .unwrap_err();
        assert!(err.contains("use one"), "{err}");
    }

    #[test]
    fn ineligible_nagq_is_stripped_with_a_warning_not_an_error() {
        // Each shape must strip to Laplace and report the warn-and-strip
        // message, never surface the kernel's shape panic.
        let (numeric, mut factor) = toy_ols();
        let g: Vec<&str> = ["a", "b", "c", "d", "e"]
            .iter()
            .copied()
            .cycle()
            .take(10)
            .collect();
        factor.insert("g".to_string(), factor_col(&g));
        // Gaussian: ineligible family.
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
            true,
            None,
        )
        .expect("ineligible nagq must be stripped, not an error");
        let msg = result.agq_warning.as_deref().expect("warning expected");
        assert!(msg.contains("nagq=3"), "{msg}");

        // Binomial with q_p = 1 + 3 slopes = 4 > 3: over the random-effect cap.
        let n = 40;
        let mut numeric = HashMap::new();
        let mut y = Vec::new();
        for i in 0..n {
            y.push(if (i * 7 + i / 3) % 3 == 0 { 1.0 } else { 0.0 });
        }
        numeric.insert("y".to_string(), y);
        for (k, name) in ["x1", "x2", "x3"].iter().enumerate() {
            let col: Vec<f64> = (0..n)
                .map(|i| (((i * (k + 3) + k) % 11) as f64) / 11.0 - 0.5)
                .collect();
            numeric.insert(name.to_string(), col);
        }
        let mut factor = HashMap::new();
        let labels: Vec<&str> = ["a", "b", "c", "d", "e"]
            .iter()
            .copied()
            .cycle()
            .take(n)
            .collect();
        factor.insert("g".to_string(), factor_col(&labels));
        let result = run_fit(
            "y ~ x1 + x2 + x3 + (1 + x1 + x2 + x3 | g)",
            numeric,
            factor,
            "binomial",
            "logit",
            "hessian",
            3,
            None,
            None,
            None,
            None,
            true,
            None,
        )
        .expect("over-cap nagq must be stripped, not an error");
        let msg = result.agq_warning.as_deref().expect("warning expected");
        assert!(msg.contains("nagq=3"), "{msg}");
    }

    #[test]
    fn kernel_panic_becomes_a_clean_error_not_a_process_abort() {
        // A panic inside the kernel fit (a negative Poisson count) takes the
        // catch_unwind around fit_warm.
        let (mut numeric, factor) = toy_ols();
        numeric.get_mut("y").unwrap()[3] = -1.0;
        let err = run_fit(
            "y ~ x", numeric, factor, "poisson", "log", "hessian", 1, None, None, None, None, true,
            None,
        )
        .unwrap_err();
        assert!(err.contains("Poisson response must be >= 0"), "{err}");
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
            true,
            None,
        )
        .unwrap_err();
        assert!(err.contains("x"), "{err}");
        assert!(err.contains('5') && err.contains('2'), "{err}");

        // A column the formula never references still counts toward the row count.
        let (mut numeric, factor) = toy_ols();
        numeric.insert("junk".to_string(), vec![0.0; 1000]);
        let err = run_fit(
            "y ~ x", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
            true, None,
        )
        .unwrap_err();
        assert!(err.contains("junk"), "{err}");
    }

    /// Warm-start values reach the mixed-model fit: a start far from the
    /// optimum takes a different number of evaluations than a cold start yet
    /// lands on the same estimates. Fixed-only fits ignore `StartValues`, so
    /// this needs a random effect.
    #[test]
    fn mixed_warm_start_reaches_same_optimum_with_different_evals() {
        let (numeric, mut factor) = toy_ols();
        let g: Vec<&str> = ["a", "b", "c", "d", "e"]
            .iter()
            .copied()
            .cycle()
            .take(10)
            .collect();
        factor.insert("g".to_string(), factor_col(&g));
        let cold = run_fit(
            "y ~ x + (1 | g)",
            numeric.clone(),
            factor.clone(),
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            None,
            None,
            None,
            true,
            None,
        )
        .expect("cold mixed fit should converge");
        assert!(cold.converged);

        // θ₀ = 5 sits far from this fit's optimum, so the warm search runs
        // well past the cold one's count (25 evaluations against 11, x86_64).
        // A nearer start can tie the cold count by chance, and a tie says
        // nothing about whether the start reached the fit.
        let warm = run_fit(
            "y ~ x + (1 | g)",
            numeric,
            factor,
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            None,
            None,
            Some((cold.beta.clone(), vec![5.0], None)),
            true,
            None,
        )
        .expect("warm-started mixed fit should converge");
        assert!(warm.converged);
        for (c, w) in cold.beta.iter().zip(warm.beta.iter()) {
            assert!((c - w).abs() < 1e-6, "beta moved: {c} vs {w}");
        }
        assert_ne!(
            warm.n_eval, cold.n_eval,
            "a far-off warm start must change the evaluation count"
        );
    }

    /// A Gaussian random-intercept fixture: 8 groups of 6 rows, `y = 1 + 0.5x +
    /// b_g + e` with deterministic `b_g` and `e`.
    #[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
    fn toy_lmm() -> (
        HashMap<String, Vec<f64>>,
        HashMap<String, (Vec<String>, Vec<u32>)>,
    ) {
        let (mut x, mut y, mut g) = (Vec::new(), Vec::new(), Vec::new());
        let labels = ["a", "b", "c", "d", "e", "f", "g", "h"];
        for (k, label) in labels.iter().enumerate() {
            let b = ((k * 5) % 8) as f64 / 4.0 - 0.9;
            for j in 0..6 {
                let xi = j as f64 - 2.5 + 0.1 * k as f64;
                let e = (((k * 7 + j * 3) % 11) as f64 - 5.0) / 6.0;
                x.push(xi);
                y.push(1.0 + 0.5 * xi + b + e);
                g.push(*label);
            }
        }
        let mut numeric = HashMap::new();
        numeric.insert("y".to_string(), y);
        numeric.insert("x".to_string(), x);
        let mut factor = HashMap::new();
        factor.insert("g".to_string(), factor_col(&g));
        (numeric, factor)
    }

    #[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
    fn fit_with(
        formula: &str,
        (numeric, factor): (
            HashMap<String, Vec<f64>>,
            HashMap<String, (Vec<String>, Vec<u32>)>,
        ),
        family: &str,
        link: &str,
        dispersion: Option<f64>,
        satterthwaite: bool,
    ) -> FitResult {
        run_fit(
            formula,
            numeric,
            factor,
            family,
            link,
            "hessian",
            1,
            dispersion,
            None,
            None,
            None,
            satterthwaite,
            None,
        )
        .expect("fit should succeed")
    }

    fn has_curvature_note(r: &FitResult) -> bool {
        r.notes.iter().any(|n| n.kind == "satterthwaite_curvature")
    }

    /// `coef_p` is the t test on each row's df (`contrast_t`'s df agrees with
    /// `coef_df` to rounding).
    fn assert_t_p(r: &FitResult) {
        for i in 0..r.beta.len() {
            let want = t_two_sided_p(r.beta[i] / r.se[i], r.coef_df[i]);
            assert!(
                (r.coef_p[i] - want).abs() < 1e-12,
                "coef_p[{i}] = {}, want {want}",
                r.coef_p[i]
            );
        }
    }

    #[test]
    fn lmm_carries_satterthwaite_df_and_p_unless_switched_off() {
        let on = fit_with(
            "y ~ x + (1 | g)",
            toy_lmm(),
            "gaussian",
            "identity",
            None,
            true,
        );
        assert!(on.converged);
        assert_eq!(on.df_method, "satterthwaite");
        assert_eq!(on.coef_df.len(), 2);
        assert!(on.coef_df.iter().all(|d| d.is_finite() && *d > 0.0));
        assert_t_p(&on);
        assert!(on.rss.is_nan() && on.r_squared.is_nan() && on.adj_r_squared.is_nan());

        let off = fit_with(
            "y ~ x + (1 | g)",
            toy_lmm(),
            "gaussian",
            "identity",
            None,
            false,
        );
        assert_eq!(off.df_method, "disabled");
        assert!(off.coef_df.iter().all(|d| d.is_nan()));
        assert!(off.coef_p.iter().all(|v| v.is_nan()));
        assert!(!has_curvature_note(&off));
        assert_eq!(off.beta, on.beta);
        assert_eq!(off.se, on.se);
    }

    #[test]
    fn ols_and_dispersion_glm_keep_residual_df_whatever_the_switch() {
        for switch in [true, false] {
            let ols = fit_with("y ~ x", toy_ols(), "gaussian", "identity", None, switch);
            assert_eq!(ols.df_method, "residual");
            assert_eq!(ols.coef_df, vec![8.0, 8.0]);
            assert_t_p(&ols);

            let gamma = fit_with("y ~ x", toy_gamma(), "gamma", "log", None, switch);
            assert_eq!(gamma.df_method, "residual");
            assert_eq!(gamma.coef_df, vec![98.0, 98.0]);
            assert_t_p(&gamma);
        }
        // A held φ keeps z, as `summary.glm(dispersion = v)` does.
        let held = fit_with("y ~ x", toy_gamma(), "gamma", "log", Some(2.5), true);
        assert_eq!(held.df_method, "none");
        assert!(held.coef_df.iter().chain(&held.coef_p).all(|v| v.is_nan()));
        let glmm = fit_with(
            "cbind(incidence, healthy) ~ period + (1 | herd)",
            cbpp_columns(),
            "binomial",
            "logit",
            None,
            true,
        );
        assert_eq!(glmm.df_method, "none");
        assert!(glmm.coef_df.iter().chain(&glmm.coef_p).all(|v| v.is_nan()));
        assert!(glmm.rss.is_nan() && glmm.r_squared.is_nan());
    }

    /// A Gaussian LMM with no finite endpoint (a random slope on an aliased
    /// column) carries the failed record and the note; switched off, neither.
    #[test]
    fn degenerate_lmm_is_failed_with_the_curvature_note() {
        let data = || {
            let (mut numeric, factor) = toy_lmm();
            let x2: Vec<f64> = numeric["x"].iter().map(|v| 2.0 * v).collect();
            numeric.insert("x2".to_string(), x2);
            (numeric, factor)
        };
        let formula = "y ~ x + x2 + (1 + x2 | g)";
        let r = fit_with(formula, data(), "gaussian", "identity", None, true);
        assert!(!r.converged);
        assert_eq!(r.df_method, "failed");
        let note = r
            .notes
            .iter()
            .find(|n| n.kind == "satterthwaite_curvature")
            .expect("the failed pass carries the note");
        assert!(note.detail.ends_with("nonfinite true"), "{}", note.detail);
        let off = fit_with(formula, data(), "gaussian", "identity", None, false);
        assert_eq!(off.df_method, "disabled");
        assert!(!has_curvature_note(&off));
    }

    /// `summary.lm`'s R² on closed forms: simple regression (R² is the squared
    /// correlation of x and y), intercept only (0), no intercept (uncentred).
    /// The weighted and offset fixtures are checked against `summary.lm` itself
    /// in the R port's tests.
    #[test]
    fn ols_r_squared_follows_summary_lm() {
        let (numeric, _) = toy_ols();
        let (x, y) = (numeric["x"].clone(), numeric["y"].clone());
        let n = y.len() as f64;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (mx, my) = (mean(&x), mean(&y));
        let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
        let syy: f64 = y.iter().map(|v| (v - my).powi(2)).sum();
        let sxy: f64 = x.iter().zip(&y).map(|(a, b)| (a - mx) * (b - my)).sum();
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);

        let r = fit_with("y ~ x", toy_ols(), "gaussian", "identity", None, true);
        let r2 = sxy * sxy / (sxx * syy);
        assert!(close(r.r_squared, r2), "{} vs {r2}", r.r_squared);
        let adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - 2.0);
        assert!(close(r.adj_r_squared, adj), "{} vs {adj}", r.adj_r_squared);
        assert!(close(r.rss, syy - sxy * sxy / sxx), "{}", r.rss);

        let one = fit_with("y ~ 1", toy_ols(), "gaussian", "identity", None, true);
        assert_eq!((one.r_squared, one.adj_r_squared), (0.0, 0.0));
        assert!(close(one.rss, syy), "{}", one.rss);

        let origin = fit_with("y ~ 0 + x", toy_ols(), "gaussian", "identity", None, true);
        let (sxx0, syy0) = (
            x.iter().map(|v| v * v).sum::<f64>(),
            y.iter().map(|v| v * v).sum::<f64>(),
        );
        let sxy0: f64 = x.iter().zip(&y).map(|(a, b)| a * b).sum();
        let r2 = sxy0 * sxy0 / (sxx0 * syy0);
        assert!(close(origin.r_squared, r2), "{} vs {r2}", origin.r_squared);
        let adj = 1.0 - (1.0 - r2) * n / (n - 1.0);
        assert!(
            close(origin.adj_r_squared, adj),
            "{} vs {adj}",
            origin.adj_r_squared
        );

        let mut tiny = HashMap::new();
        tiny.insert("y".to_string(), vec![1.0, 3.0]);
        tiny.insert("x".to_string(), vec![0.0, 1.0]);
        let unfit = fit_with(
            "y ~ x",
            (tiny, HashMap::new()),
            "gaussian",
            "identity",
            None,
            true,
        );
        assert!(!unfit.converged);
        assert!(unfit.rss.is_nan() && unfit.r_squared.is_nan() && unfit.adj_r_squared.is_nan());
    }

    #[test]
    fn same_fixed_part_reads_the_fixed_terms_and_intercept_only() {
        assert_eq!(
            same_fixed_part("y ~ a * b + (1 | g)", "y ~ b:a + b + a"),
            Ok(true)
        );
        assert_eq!(same_fixed_part("y ~ x", "y ~ 0 + x"), Ok(false));
        let err = same_fixed_part("y ~ x", "y ~ x - z").unwrap_err();
        assert!(err.contains("\"y ~ x - z\""), "{err}");
    }

    #[test]
    fn confint_takes_t_on_a_finite_df_and_normal_elsewhere() {
        // qt(0.975, 8), qnorm(0.975) and qnorm(0.95) in R.
        let (qt8, qn, qn90) = (2.3060041352041654, 1.9599639845400534, 1.6448536269514715);
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
        let beta = [1.0, 2.0, f64::NAN, 3.0];
        let se = [0.5, 0.5, f64::NAN, 0.25];
        let df = [8.0, f64::NAN, f64::NAN, 0.0];
        let ci = confint(&beta, &se, &df, 0.95).unwrap();
        assert!(close(ci[0][0], 1.0 - qt8 * 0.5) && close(ci[0][1], 1.0 + qt8 * 0.5));
        assert!(close(ci[1][0], 2.0 - qn * 0.5) && close(ci[1][1], 2.0 + qn * 0.5));
        assert!(ci[2][0].is_nan() && ci[2][1].is_nan());
        // A df of 0 has no t distribution: normal, never NaN while se is finite.
        assert!(close(ci[3][0], 3.0 - qn * 0.25) && close(ci[3][1], 3.0 + qn * 0.25));
        let ci90 = confint(&beta, &se, &df, 0.9).unwrap();
        assert!(close(ci90[1][1], 2.0 + qn90 * 0.5));
        for bad in [0.0, 1.0, 1.5, -0.1, f64::NAN] {
            assert!(confint(&beta, &se, &df, bad).is_err(), "level {bad}");
        }
        assert!(confint(&beta, &se[..2], &df, 0.95).is_err());
    }

    /// Deterministic fixed-effects-only Gamma(log) fixture: `y = exp(1 +
    /// 0.5·x) · jitter` on a centered `x`, `jitter` a 7-cycle deterministic
    /// wobble around 1 (same shape as `glm.rs`'s Gamma fixtures) so the
    /// estimated-vs-fixed dispersion paths are both exercised on real data.
    #[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
    fn toy_gamma() -> (
        HashMap<String, Vec<f64>>,
        HashMap<String, (Vec<String>, Vec<u32>)>,
    ) {
        let n = 100;
        let mut x = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let xi = (i as f64) / (n as f64) - 0.5;
            let mu = (1.0 + 0.5 * xi).exp();
            let jitter = 1.0 + 0.05 * (((i % 7) as f64) - 3.0) / 3.0;
            x[i] = xi;
            y[i] = mu * jitter;
        }
        let mut numeric = HashMap::new();
        numeric.insert("y".to_string(), y);
        numeric.insert("x".to_string(), x);
        (numeric, HashMap::new())
    }

    /// `dispersion: Some(v)` must hold φ fixed at exactly `v` instead of
    /// estimating it — the only way `FitOptions::dispersion`
    /// reaches the kernel from a port is through this string/option layer.
    #[test]
    fn gamma_end_to_end_holds_dispersion_fixed() {
        let (numeric, factor) = toy_gamma();
        let result = run_fit(
            "y ~ x",
            numeric,
            factor,
            "gamma",
            "log",
            "hessian",
            1,
            Some(2.5),
            None,
            None,
            None,
            true,
            None,
        )
        .expect("gamma fit should converge");
        assert!(result.converged);
        assert_eq!(
            result.dispersion, 2.5,
            "dispersion: Some(v) must hold phi fixed at v, not estimate it"
        );
    }

    /// The committed `cbpp` herd dataset (`validation/data/empirical/cbpp.csv`)
    /// through `run_fit`'s `cbind()` aggregated-binomial path: `incidence`
    /// successes out of `size` trials per herd/period. `healthy = size −
    /// incidence` is precomputed since `cbind()` only accepts two bare column
    /// names, not an expression.
    #[allow(clippy::type_complexity)] // test fixture: the numeric+factor column maps run_fit takes
    fn cbpp_columns() -> (
        HashMap<String, Vec<f64>>,
        HashMap<String, (Vec<String>, Vec<u32>)>,
    ) {
        let csv = include_str!("../validation/data/empirical/cbpp.csv");
        let mut herd_labels: Vec<String> = Vec::new();
        let mut period_labels: Vec<String> = Vec::new();
        let mut incidence = Vec::new();
        let mut healthy = Vec::new();
        for line in csv.lines().skip(1).filter(|l| !l.trim().is_empty()) {
            let f: Vec<&str> = line.split(',').map(|s| s.trim_matches('"')).collect();
            let inc: f64 = f[1].parse().unwrap();
            let size: f64 = f[2].parse().unwrap();
            herd_labels.push(f[0].to_string());
            incidence.push(inc);
            healthy.push(size - inc);
            period_labels.push(f[3].to_string());
        }
        let herd_refs: Vec<&str> = herd_labels.iter().map(String::as_str).collect();
        let period_refs: Vec<&str> = period_labels.iter().map(String::as_str).collect();
        let mut numeric = HashMap::new();
        numeric.insert("incidence".to_string(), incidence);
        numeric.insert("healthy".to_string(), healthy);
        let mut factor = HashMap::new();
        factor.insert("herd".to_string(), factor_col(&herd_refs));
        factor.insert("period".to_string(), factor_col(&period_refs));
        (numeric, factor)
    }

    /// `wald_se="rx"` is the only route by which the Python/R ports select
    /// `WaldSe::Rx`; a bad string→enum hand-off (or a no-op enum arm) would
    /// leave `se` identical to the `"hessian"` default and pass silently.
    /// Same herd/period random-intercept model `fit_grouped_honors_opts_wald_se`
    /// (`fit/glmm_tests.rs`) checks at the kernel-struct layer — this is the
    /// string-typed orchestration layer above it.
    #[test]
    fn cbpp_glmm_wald_se_rx_differs_from_hessian() {
        let (numeric, factor) = cbpp_columns();
        let hessian = run_fit(
            "cbind(incidence, healthy) ~ period + (1 | herd)",
            numeric.clone(),
            factor.clone(),
            "binomial",
            "logit",
            "hessian",
            1,
            None,
            None,
            None,
            None,
            true,
            None,
        )
        .expect("hessian-SE fit should converge");
        assert!(hessian.converged);
        assert_eq!(hessian.beta.len(), 4);

        let rx = run_fit(
            "cbind(incidence, healthy) ~ period + (1 | herd)",
            numeric,
            factor,
            "binomial",
            "logit",
            "rx",
            1,
            None,
            None,
            None,
            None,
            true,
            None,
        )
        .expect("rx-SE fit should converge");
        assert!(rx.converged);

        for (h, r) in hessian.beta.iter().zip(rx.beta.iter()) {
            assert!(
                (h - r).abs() < 1e-6,
                "wald_se must select an SE denominator, not move the fitted optimum: {h} vs {r}"
            );
        }
        let se_differs = hessian
            .se
            .iter()
            .zip(rx.se.iter())
            .any(|(h, r)| (h - r).abs() > 1e-6);
        assert!(
            se_differs,
            "hessian={:?} rx={:?} must differ, or wald_se=\"rx\" is a silent no-op",
            hessian.se, rx.se
        );
    }
}
