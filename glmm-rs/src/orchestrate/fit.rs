//! Fitting a port's data: `run_fit` fits one formula, `run_compare` fits two
//! on the same rows and compares them, and `flatten` turns a fit into the
//! `FitResult` both ports publish.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::fit::{agq_shape, theta_width};
use crate::formula::anova::{
    auto_choice, compare_stored, linear_note_line, nesting, term_tests_cached, CompareError,
    CompareOptions, Comparison, Differs, FixedRow, Model, Nesting, RefitCache, SsType, TermTable,
    TermTest, TermTestOptions, TestChoice, WeightPowerMismatch,
};
use crate::formula::{label_ranef, lower, parse, Lowered, ReGroupInfo, Table};
use crate::inference::contrast::{ContrastError, NoDfReason};
use crate::inference::tails::t_two_sided_p;
use crate::inference::{DenDf, LinearKind, Objective, Reference};
use crate::{Dispersion, Family, Fit, FitOptions, Note, SeType, StartValues, WeightPower};

use super::design::table_of;
use super::stored::{record_of, CondvarCrossRecord, RobustRecord, SatterthwaiteRecord};
use super::{
    boundary_name, choice_tags, kind_tag, outer_stop_name, panic_message, PortError, StopReason,
};

/// One [`Note`], flattened for the ports. `kind` is the stable identifier each
/// port keys its warning category (Python) or condition class (R) off; the
/// English message, and the `detail` text a stored note carries, are built
/// from these fields by the `glmm-bind` crate, not here.
///
/// `note_infos` matches [`Note`] exhaustively (in-crate, `#[non_exhaustive]`
/// does not bind), so a new variant fails compilation until it is given a
/// flattening — no note can silently drop. The message renderer keeps an
/// unrecognized-`kind` fallback of its own.
#[derive(Debug)]
pub struct NoteInfo {
    /// The stable note identifier (see the struct docs).
    pub kind: &'static str,
    /// Fixed-effect column indices, 0-based into the fitted names (the R shim
    /// adds 1 at its boundary): on `IllConditioned` only the column the
    /// kernel detected (the columns it is entangled with are not identified),
    /// on `RobustSeUnidentified` and `RobustDfSmall` the columns they name.
    /// Empty for a variant that names no column.
    pub columns: Vec<u32>,
    /// The scaled pivot behind the note; `NaN` for a variant that carries none.
    pub pivot: f64,
    /// `PirlsExhausted` payload: how many fit-path evals hit the inner-PIRLS
    /// cap (that variant's `evals`). Also carries `NbShapeUnsettled`'s
    /// `rounds`, `NonIntegerResponse`'s `rows` and `FewRowsForRandomEffects`'s
    /// `random_effects`, which need the same "small count" shape and no
    /// variant needs more than one at once. 0 for every other variant.
    pub evals: u32,
    /// `PirlsExhausted` payload: whether the final re-evaluation at the
    /// converged fit itself hit the cap, so the truncated solve's ũ/W̃ feed
    /// the reported estimates (that variant's `final_eval`) — the case the
    /// ports' warnings must distinguish from a rejected trial point. `false`
    /// for every other variant.
    pub final_eval: bool,
    /// The names a variant carries: the grouping name of
    /// `SingleLevelGroupingDropped`, `SingleLevelGrouping`,
    /// `SeparatedByGrouping`, `OneTrialPerLevel` and `NewGroupingLevels`;
    /// `FewRowsForRandomEffects`' term as `[terms, grouping]` (its row count
    /// is `FitResult::nobs`); the two names of `RepeatedGrouping`; and
    /// `SlopeConstantWithinLevels`' slope labels with its grouping name last.
    /// Empty for every other variant.
    pub names: Vec<String>,
    /// The rows of `NewGroupingLevels` and `RowsNotEstimable`, 0-based into
    /// the rows predicted. Empty for every other variant.
    pub rows: Vec<u32>,
    /// `SatterthwaiteCurvature` payload: the flat directions left out of the
    /// curvature. 0 for every other variant.
    pub dropped: u32,
    /// `SatterthwaiteCurvature` payload: the negative directions. 0 for every
    /// other variant.
    pub negative: u32,
    /// `SatterthwaiteCurvature` payload: whether the variance search
    /// converged. `false` for every other variant.
    pub converged: bool,
    /// `SatterthwaiteCurvature` payload: whether the fit had no finite end
    /// point or a non-finite deviance near it. `false` for every other
    /// variant.
    pub nonfinite: bool,
    /// The fitted power on `WeightPowerAtBound`, `WeightPowerOutsideUnit`
    /// and `WeightPowerImplausible`. `NaN` for every other variant.
    pub delta: f64,
    /// `WeightPowerImplausible` payload: `s^|delta − 1|`. On
    /// `DispersionAtLimit`, the box edge on the internal weight scale
    /// (`edge`), which sets the message's percentage. `NaN` for every other
    /// variant.
    pub factor: f64,
    /// `WeightPowerAtBound` payload: the search bracket's ends. On
    /// `NbShapeAtLimit` and `DispersionAtLimit` the limit the parameter ended
    /// on (for `DispersionAtLimit` on the reported dispersion's scale), in
    /// `lower` when it is the lower limit and in `upper` otherwise, the
    /// other `NaN`. `NaN` for
    /// every other variant.
    pub lower: f64,
    /// See `lower`.
    pub upper: f64,
    /// `WeightPowerSearchFailures` payload: search fits that failed, and
    /// those with a finite log-likelihood that did not converge. 0 for every
    /// other variant.
    pub failed: u32,
    /// See `failed`.
    pub not_converged: u32,
    /// `RobustDfSmall` payload: the df of each of `columns`, in their order.
    /// Empty for every other variant.
    pub df: Vec<f64>,
}

/// A [`NoteInfo`] with no payload, the base every variant fills in.
const NO_PAYLOAD: NoteInfo = NoteInfo {
    kind: "",
    columns: Vec::new(),
    pivot: f64::NAN,
    evals: 0,
    final_eval: false,
    names: Vec::new(),
    rows: Vec::new(),
    dropped: 0,
    negative: 0,
    converged: false,
    nonfinite: false,
    delta: f64::NAN,
    factor: f64::NAN,
    lower: f64::NAN,
    upper: f64::NAN,
    failed: 0,
    not_converged: 0,
    df: Vec::new(),
};

/// One grouping's labelled conditional modes, [`crate::formula::RanefBlock`]
/// as the ports read it.
#[derive(Debug)]
pub struct RanefBlock {
    /// The grouping's name.
    pub group: String,
    /// Its term names, `"(Intercept)"` first.
    pub terms: Vec<String>,
    /// One label per retained level.
    pub levels: Vec<String>,
    /// Row-major `levels.len() × terms.len()` conditional modes.
    pub values: Vec<f64>,
    /// Row-major `levels.len() × q(q+1)/2`, `q = terms.len()`: per level the
    /// vech of its `q × q` conditional covariance block, as
    /// `crate::formula::RanefBlock::condvar` lays it out; empty where
    /// `Fit::condvar(g)` is.
    pub condvar: Vec<f64>,
}

/// The design summary of a Gaussian fit whose only random term is one random
/// intercept ([`FitResult::design_summary`]). With `K` the number of groups
/// and `p` retained fixed-effect columns, `counts` has `K` entries, `s` is the
/// `p × K` matrix `X̃ᵀZ` and `c` the `p × p` matrix `X̃ᵀX̃`, both flat and
/// column-major, on the fit's orthogonalized design `X̃`
/// (`crate::fit::DesignSummary` has the detail).
#[derive(Debug)]
pub struct DesignSummary {
    /// The row count of each group.
    pub counts: Vec<f64>,
    /// `X̃ᵀZ`, `p × K`, column-major.
    pub s: Vec<f64>,
    /// `X̃ᵀX̃`, `p × p`, column-major.
    pub c: Vec<f64>,
}

/// `notes` flattened for the ports. `re_groups` names the groupings a note
/// indexes, in declaration order.
pub(super) fn note_infos(notes: Vec<Note>, re_groups: &[ReGroupInfo]) -> Vec<NoteInfo> {
    // A note's grouping index comes from a fit of the model `re_groups` was
    // lowered with, so it is in range.
    let name = |g: u32| re_groups[g as usize].name.clone();
    notes
        .into_iter()
        .map(|note| match note {
            Note::IllConditioned { columns, pivot } => NoteInfo {
                kind: "ill_conditioned",
                columns,
                pivot,
                ..NO_PAYLOAD
            },
            Note::PirlsExhausted { evals, final_eval } => NoteInfo {
                kind: "pirls_exhausted",
                evals,
                final_eval,
                ..NO_PAYLOAD
            },
            Note::HessianSeFallback => NoteInfo {
                kind: "hessian_se_fallback",
                ..NO_PAYLOAD
            },
            Note::NbShapeUnsettled { rounds } => NoteInfo {
                kind: "nb_shape_unsettled",
                evals: rounds,
                ..NO_PAYLOAD
            },
            Note::NbShapeAtLimit { limit } => {
                let lower_end = limit == crate::consts::NB_THETA_LO;
                NoteInfo {
                    kind: "nb_shape_at_limit",
                    lower: if lower_end { limit } else { f64::NAN },
                    upper: if lower_end { f64::NAN } else { limit },
                    ..NO_PAYLOAD
                }
            }
            Note::NonIntegerResponse { rows } => NoteInfo {
                kind: "non_integer_response",
                evals: rows,
                ..NO_PAYLOAD
            },
            Note::SingleLevelGroupingDropped { grouping } => NoteInfo {
                kind: "single_level_grouping_dropped",
                names: vec![grouping],
                ..NO_PAYLOAD
            },
            Note::FewRowsForRandomEffects {
                terms,
                grouping,
                random_effects,
                rows: _,
            } => NoteInfo {
                kind: "few_rows_for_random_effects",
                evals: random_effects,
                names: vec![terms, grouping],
                ..NO_PAYLOAD
            },
            Note::SingleLevelGrouping { grouping } => NoteInfo {
                kind: "single_level_grouping",
                names: vec![name(grouping)],
                ..NO_PAYLOAD
            },
            Note::RepeatedGrouping { first, second } => NoteInfo {
                kind: "repeated_grouping",
                names: vec![name(first), name(second)],
                ..NO_PAYLOAD
            },
            Note::SlopeConstantWithinLevels { grouping, slopes } => NoteInfo {
                kind: "slope_constant_within_levels",
                // The block has a random intercept, so its slope `k` is
                // term name `1 + k`.
                names: slopes
                    .iter()
                    .map(|&k| re_groups[grouping as usize].terms[1 + k as usize].clone())
                    .chain([name(grouping)])
                    .collect(),
                ..NO_PAYLOAD
            },
            Note::SeparatedByGrouping { grouping } => NoteInfo {
                kind: "separated_by_grouping",
                names: vec![name(grouping)],
                ..NO_PAYLOAD
            },
            Note::OneTrialPerLevel { grouping } => NoteInfo {
                kind: "one_trial_per_level",
                names: vec![name(grouping)],
                ..NO_PAYLOAD
            },
            Note::ExactProfileFallback => NoteInfo {
                kind: "exact_profile_fallback",
                ..NO_PAYLOAD
            },
            Note::GradientSearchFallback => NoteInfo {
                kind: "gradient_search_fallback",
                ..NO_PAYLOAD
            },
            Note::SeUnavailable => NoteInfo {
                kind: "se_unavailable",
                ..NO_PAYLOAD
            },
            Note::PrecisionLimited => NoteInfo {
                kind: "precision_limited",
                ..NO_PAYLOAD
            },
            Note::Separated => NoteInfo {
                kind: "separated",
                ..NO_PAYLOAD
            },
            Note::VarianceAtLimit => NoteInfo {
                kind: "variance_at_limit",
                ..NO_PAYLOAD
            },
            Note::DispersionAtLimit { limit, edge } => {
                let lower_end = edge == crate::consts::GAMMA_PHI_LO;
                NoteInfo {
                    kind: "dispersion_at_limit",
                    factor: edge,
                    lower: if lower_end { limit } else { f64::NAN },
                    upper: if lower_end { f64::NAN } else { limit },
                    ..NO_PAYLOAD
                }
            }
            Note::NewGroupingLevels { grouping, rows } => NoteInfo {
                kind: "new_grouping_levels",
                names: vec![name(grouping)],
                rows,
                ..NO_PAYLOAD
            },
            Note::RowsNotEstimable { rows } => NoteInfo {
                kind: "rows_not_estimable",
                rows,
                ..NO_PAYLOAD
            },
            Note::WeightPowerAtBound {
                delta,
                lower,
                upper,
            } => NoteInfo {
                kind: "weight_power_at_bound",
                delta,
                lower,
                upper,
                ..NO_PAYLOAD
            },
            Note::WeightPowerSearchFailures {
                failed,
                not_converged,
            } => NoteInfo {
                kind: "weight_power_search_failures",
                failed,
                not_converged,
                ..NO_PAYLOAD
            },
            Note::WeightPowerOutsideUnit { delta } => NoteInfo {
                kind: "weight_power_outside_unit",
                delta,
                ..NO_PAYLOAD
            },
            Note::WeightPowerImplausible { delta, factor } => NoteInfo {
                kind: "weight_power_implausible",
                delta,
                factor,
                ..NO_PAYLOAD
            },
            Note::SatterthwaiteCurvature {
                dropped,
                negative,
                converged,
                nonfinite,
            } => NoteInfo {
                kind: "satterthwaite_curvature",
                dropped,
                negative,
                converged,
                nonfinite,
                ..NO_PAYLOAD
            },
            Note::RobustWithRandomSlope => NoteInfo {
                kind: "robust_with_random_slope",
                ..NO_PAYLOAD
            },
            Note::RobustSeUnidentified { columns } => NoteInfo {
                kind: "robust_se_unidentified",
                columns,
                ..NO_PAYLOAD
            },
            Note::RobustDfSmall { columns, df } => NoteInfo {
                kind: "robust_df_small",
                columns,
                df,
                ..NO_PAYLOAD
            },
            Note::PredictSeModelBased => NoteInfo {
                kind: "predict_se_model_based",
                ..NO_PAYLOAD
            },
        })
        .collect()
}

/// Everything a port publishes about one fit: [`crate::Fit`] plus the
/// lowering's naming, flattened to plain vectors, tuples, and strings that
/// `glmm-bind`'s `fit_record` turns into the ports' fit record.
#[derive(Debug)]
pub struct FitResult {
    /// Fixed-effect estimates (`crate::Fit::beta`).
    pub beta: Vec<f64>,
    /// Wald standard errors per fixed effect (`crate::Fit::se`).
    pub se: Vec<f64>,
    /// Full p×p fixed-effect covariance (`crate::Fit::vcov`) — the off-diagonals
    /// `se` alone cannot carry, which any hand-built Wald contrast needs.
    pub vcov: Vec<Vec<f64>>,
    /// Per-element variance components `θ_k²·σ̂²`, copied from the fit and laid
    /// out like `varcorr`: the ports' `tau2`.
    pub tau2: Vec<f64>,
    /// One vech-packed lower-triangular covariance block per grouping,
    /// declaration order (column-major: `(0,0),(1,0),…,(q-1,q-1)`); the same
    /// values [`crate::Fit::varcorr()`] gives as `q × q`.
    pub varcorr: Vec<Vec<f64>>,
    /// Standard errors of the RE standard deviations: one vech per grouping,
    /// laid out like `varcorr`, the SEs at the diagonal slots and NaN off them;
    /// NaN where the fit has no SE (a Gaussian LMM). [`crate::Fit::stddev_se()`]
    /// gives one grouping's diagonal.
    pub stddev_se: Vec<f64>,
    /// Which fixed-effect columns were dropped as exactly redundant
    /// (`crate::Diagnostics::aliased`).
    pub aliased: Vec<bool>,
    /// Dispersion/scale estimate (`crate::Fit::dispersion`).
    pub dispersion: f64,
    /// Optimizer convergence flag (`crate::Diagnostics::converged`).
    pub converged: bool,
    /// Why the fit did not converge, `None` exactly when `converged`
    /// (the kernel's own record, not inferred from the estimates).
    pub stop: Option<StopReason>,
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
    /// Per grouping, in `re_groups` order, the term of each of its term names
    /// (`crate::formula::ReGroupInfo::term_index`): `(x || g)` gives `[0, 1]`.
    /// A port's `summary()` prints a correlation only between two columns of
    /// one term, as lme4 does.
    pub re_term_index: Vec<Vec<usize>>,
    /// Set when an ineligible-shape `nagq>1` was stripped (the fit proceeded
    /// with Laplace). Each port raises its own warning when this is set.
    pub agq_warning: bool,
    /// Set when `reml = Some(true)` was given on a non-Gaussian family (GLM or
    /// GLMM), where the argument has no meaning: it was ignored and the model
    /// fitted by ML, as under `None`. Each port raises its own warning, with
    /// its own spelling of the argument, when this is set.
    pub reml_warning: bool,
    /// Log-likelihood at the fitted parameters (`crate::Fit::loglik`).
    pub loglik: f64,
    /// Parameters counted for AIC/BIC (`crate::Fit::df`).
    pub df: usize,
    /// `true` iff `loglik` is a REML criterion, not an ML log-likelihood
    /// (`crate::Fit::reml`).
    pub reml: bool,
    /// Fitted means per row (`crate::Fit::fitted`).
    pub fitted: Vec<f64>,
    /// Random-effect conditional modes: every grouping's
    /// [`crate::Fit::ranef()`] block concatenated, declaration order.
    pub ranef: Vec<f64>,
    /// Level count per grouping, for slicing `ranef` (`crate::Fit::ranef_levels`).
    pub ranef_levels: Vec<usize>,
    /// Per grouping, in `re_groups` order, the number of its levels the data
    /// has: a nested grouping's observed combinations, not the padded count
    /// of `ranef_levels`. What a port's `summary()` prints; filled on a fit
    /// that failed too.
    pub re_group_levels: Vec<usize>,
    /// The same conditional modes, LABELLED — `crate::formula::label_ranef`'s
    /// blocks flattened for the ports: per grouping, `(name, term names, level
    /// labels, row-major values)`. Padded nested slots are already dropped, so
    /// `values.len() == levels.len() * terms.len()`. Empty on a fit without
    /// random effects or without conditional modes. A port does NOT slice `ranef` itself: which layout a grouping lands
    /// in is a data-dependent routing decision inside the kernel, so only the
    /// crate can label it.
    pub ranef_blocks: Vec<RanefBlock>,
    /// The conditional covariances of the random effects
    /// (every grouping's [`crate::Fit::condvar()`] block concatenated), in the
    /// flat layout of `ranef`, padded nested
    /// slots included; a port stores it, with `condvar_x` and
    /// `condvar_cross`, for [`predict`](super::predict). Coupled sites, as for `y`.
    pub condvar: Vec<f64>,
    /// The rows behind the conditional standard error's fixed-effect part
    /// (the crate-private `Fit::condvar_x`), per level of `condvar` the
    /// row-major `q × p` block.
    pub condvar_x: Vec<f64>,
    /// The conditional covariance across groupings (the crate-private
    /// `Fit::condvar_cross`) as plain numbers, `None` where the fit has
    /// none (one grouping, or no `condvar`).
    pub condvar_cross: Option<CondvarCrossRecord>,
    /// The response as the kernel fitted it (`Lowered::y`): after lowering, so
    /// a `cbind(s, f)` response is the proportion `s/(s+f)` and a formula
    /// transform on the LHS is already applied. Retained so a port can compute
    /// residuals and the summary's scaled-residual quartiles without a second
    /// copy of the data. Coupled sites, change together: `glmm-bind`'s
    /// `fit_record` flattens every field here.
    pub y: Vec<f64>,
    /// Prior weights the kernel fitted with, `None` when unweighted: the
    /// caller's, before [`FitResult::weight_power`]. For a
    /// `cbind(s, f)` response these are the trial counts `s + f` the lowering
    /// made up, which the caller never passed — the ports' Pearson residuals
    /// need them, and `weights=` alone cannot say.
    pub weights: Option<Vec<f64>>,
    /// The power the fit put on `weights` (`crate::Fit::weight_power`): 1 on
    /// the weights as given, NaN without weights. Pearson residuals, `rss`
    /// and the simulated draws read `weights` raised to it.
    pub weight_power: f64,
    /// Whether the fit estimated `weight_power` (δ̂) rather than holding it
    /// fixed.
    pub weight_power_estimated: bool,
    /// The standard error behind `se`, `vcov` and `coef_df`
    /// (`crate::Fit::se_type`).
    pub se_type: SeType,
    /// On a robust fit, the model covariance of every retained column
    /// (the crate-private `Fit::model_vcov`), `p` rows of `p`, NaN on an
    /// aliased column; empty on a model fit, whose `vcov` is it. What a
    /// robust fit's `predict`, Type II hypotheses and robust df read. Coupled
    /// sites, as for `y`.
    pub model_vcov: Vec<Vec<f64>>,
    /// On a robust fit, its CR2 or HC2 pieces; `None` on a model fit. Coupled
    /// sites, as for `y`.
    pub robust: Option<RobustRecord>,
    /// The offset the kernel fitted with, from the `offset=` argument or an
    /// `offset()` term of the formula; `None` without one. Kept so a
    /// comparison of two stored fits can check that their offsets agree.
    /// Coupled sites, as for `y`.
    pub offset: Option<Vec<f64>>,
    /// The design summary of a Gaussian fit whose only random term is one
    /// random intercept, with no prior weights and at least one fixed-effect
    /// column; `None` on every other fit. Coupled sites, as for `y`:
    /// `glmm-bind`'s `fit_record` flattens it.
    pub design_summary: Option<DesignSummary>,
    /// Row count the kernel fitted (`Lowered::n`). NOT `fitted.len()`: `fitted`
    /// is NaN on a non-converged fit and the ports' headers print on those.
    pub nobs: usize,
    /// Where the accepted θ sits, from `crate::Diagnostics::boundary`; see
    /// `boundary_name` for the vocabulary.
    pub boundary: &'static str,
    /// Which variance components the optimizer pinned at 0, aligned with the
    /// `varcorr` blocks (`crate::Diagnostics::pinned`): one row per grouping,
    /// all false when nothing was pinned; empty on a model with no variance
    /// components (OLS, GLM, fixed-effect-only negative binomial).
    pub pinned: Vec<Vec<bool>>,
    /// Solver observations with no dedicated field (`crate::Diagnostics::notes`).
    pub notes: Vec<NoteInfo>,
    /// Degrees of freedom per coefficient (`crate::Fit::coef_df`): the
    /// Satterthwaite df on a Gaussian LMM, `n − p` on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated, the robust df on a robust fit,
    /// NaN where the test is z and on an OLS or dispersion GLM fit that did
    /// not converge (still a t test, `df_method` `"residual"`).
    pub coef_df: Vec<f64>,
    /// The t test's two-sided p-value per coefficient, from
    /// `crate::Fit::contrast_t` on the unit vector (on a robust fit, `β̂/se`
    /// on `coef_df`); NaN where there is no df.
    pub coef_p: Vec<f64>,
    /// Where `coef_df` comes from, and so how the fit tests; see
    /// [`DfMethod`].
    pub df_method: DfMethod,
    /// The Satterthwaite record [`contrast_test`](super::contrast_test) reads, on a fit whose
    /// `df_method` is `"satterthwaite"`, and on a `"pustejovsky_tipton"` fit
    /// whose pass ran, for its model-based joint tests; `None` on every other
    /// fit, whose test reads only `beta`, `vcov` (on a robust fit also
    /// `model_vcov` and `robust`), `aliased`, `nobs` and `dispersion`.
    /// Coupled sites, as for `y`.
    pub satterthwaite_record: Option<SatterthwaiteRecord>,
    /// Weighted residual sum of squares `Σwᵢ(yᵢ − μ̂ᵢ)²` of an OLS fit, `wᵢ`
    /// the working weights (`weights` raised to `weight_power`); NaN on
    /// every other fit and on a non-converged one.
    pub rss: f64,
    /// `summary.lm`'s multiple R² of an OLS fit (0 on an intercept-only
    /// model); NaN on every other fit, on a non-converged one, and when
    /// neither the response nor the fitted values vary beyond round-off.
    pub r_squared: f64,
    /// `summary.lm`'s adjusted R², with the same NaN and 0 cases as
    /// `r_squared`.
    pub adj_r_squared: f64,
    /// What the fit maximised (`crate::Fit::objective`); [`objective_parts`](super::objective_parts)
    /// flattens it to a label and a node count, [`objective_from_parts`](super::objective_from_parts)
    /// rebuilds it from them. Coupled sites, as for `y`.
    pub objective: Objective,
    /// The Type III per-term table
    /// ([`term_tests`](crate::formula::anova::term_tests)), built at fit time so a
    /// port's `anova(m)` reads it and computes nothing; with an empty cell,
    /// the missing-cell note is among its notes. `Err` holds the refusal (a
    /// formula whose interaction lacks a marginal term), which the fit itself
    /// does not raise, or a panic caught while the table was built
    /// ([`TermTableError`]). [`term_columns`]
    /// flattens a table. The F or Wald χ² table: built under
    /// [`TermTest::Auto`], so its [`TermTable::choice`] is the test `"auto"`
    /// gives and why, except on a Poisson model with random effects, whose
    /// `"auto"` is the likelihood-ratio table (`term_table_type3_lrt`), and
    /// whose table here is built under [`TermTest::Wald`]. Coupled sites, as
    /// for `y`.
    pub term_table_type3: Result<TermTable, TermTableError>,
    /// The Type II per-term table, as `term_table_type3`.
    pub term_table_type2: Result<TermTable, TermTableError>,
    /// The Type III likelihood-ratio table
    /// ([`term_tests`](crate::formula::anova::term_tests) under
    /// [`TermTest::Lrt`]): `None` unless the fit stores it, under
    /// [`run_fit`]'s `anova_test` [`TermTest::Lrt`], and under
    /// [`TermTest::Auto`] on a Poisson model with random effects (built under
    /// [`TermTest::Auto`]: the same
    /// rows, with its reason). `Err` as for `term_table_type3`, and also the
    /// refusal of a fit whose rows are F tests. Coupled sites, as for `y`.
    pub term_table_type3_lrt: Option<Result<TermTable, TermTableError>>,
    /// The Type II likelihood-ratio table, as `term_table_type3_lrt`.
    pub term_table_type2_lrt: Option<Result<TermTable, TermTableError>>,
    /// What `"auto"`, the default test of a port's `anova(m)`, gives on this
    /// fit and why: the F or Wald χ² of `term_table_type3` and
    /// `term_table_type2`, or, on a Poisson model with random effects, the
    /// likelihood-ratio χ² of `term_table_type3_lrt` and
    /// `term_table_type2_lrt`. [`choice_tags`] flattens it. Coupled sites, as
    /// for `y`.
    pub term_test_auto: TestChoice,
}

/// Why a stored per-term table ([`FitResult::term_table_type3`] and the
/// other three) is missing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermTableError {
    /// The crate refused the table, with its message.
    Refused(String),
    /// A panic caught while the table was built: a bug, not the model's
    /// fault. The message starts "internal error".
    Internal(String),
}

impl std::fmt::Display for TermTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Refused(m) | Self::Internal(m) => f.write_str(m),
        }
    }
}

/// A per-term table as the plain columns a port stores, one entry per row
/// in each vector, flattened from [`TermTable`] by [`term_columns`].
#[derive(Clone, Debug, PartialEq)]
pub struct TermColumns {
    /// `"III"` or `"II"`.
    pub ss_type: &'static str,
    /// The table's test ([`TermTable::choice`]), named as in `kind`.
    pub test: &'static str,
    /// Why the table runs `test`: `"requested"` (the test was asked for),
    /// `"f_test"` (the F is the only test the fit's table has), or what
    /// `"auto"` found: `"poisson_mixed"` (a Poisson model with random effects:
    /// the likelihood-ratio χ²), and the Wald χ² on `"binomial_mixed"`,
    /// `"negative_binomial_mixed"` and `"other_mixed"` (a model of that
    /// family, or of another family with χ² rows, with random effects) and
    /// `"no_random_effects"` (χ² rows and no random effects).
    pub reason: &'static str,
    /// The term labels, as R writes them (`"g:h"`).
    pub label: Vec<String>,
    /// `"F"` for an F row, `"chisq"` for a Wald χ² row, `"lrt"` for a
    /// likelihood-ratio χ² row.
    pub kind: Vec<&'static str>,
    /// The F statistic, or the χ² (Wald or likelihood-ratio).
    pub statistic: Vec<f64>,
    /// The numerator df (the χ² df on a χ² row); NaN when not tested.
    pub num_df: Vec<f64>,
    /// The denominator df of an F row; NaN on a χ² row.
    pub den_df: Vec<f64>,
    /// Where an F row's denominator df comes from: `"satterthwaite"`,
    /// `"residual"`, or on a one-row test of a robust fit
    /// `"pustejovsky_tipton"` or `"bell_mccaffrey"`; `""` on a χ² row.
    pub den_df_kind: Vec<&'static str>,
    /// The p-value.
    pub p_value: Vec<f64>,
    /// lmerTest's `Sum Sq` on an F row; NaN on a χ² row.
    pub sum_sq: Vec<f64>,
    /// lmerTest's `Mean Sq` on an F row; NaN on a χ² row.
    pub mean_sq: Vec<f64>,
    /// The partial η² of an F row ([`FixedRow::eta2_partial`]); NaN on a χ²
    /// row and on an untested row.
    pub eta2_partial: Vec<f64>,
    /// Each row's notes as plain sentences (an aliased column, a rank below
    /// the row count, no denominator df and why; on a likelihood-ratio
    /// table, a refit that did not converge and a refit above the larger
    /// model by more than round-off (its statistic set to 0)).
    pub row_notes: Vec<Vec<String>>,
    /// The table's notes as plain sentences (the missing-cell note; on a
    /// likelihood-ratio table, a fit that did not converge, with nothing
    /// refitted, and a term whose model under its hypothesis could not be
    /// built).
    pub notes: Vec<String>,
}

/// A [`TermTable`] as [`TermColumns`], for `glmm-bind`'s term-table records.
pub fn term_columns(table: &TermTable) -> TermColumns {
    let rows = &table.fixed;
    let col = |f: fn(&FixedRow) -> f64| rows.iter().map(f).collect();
    let (test, reason) = choice_tags(&table.choice);
    TermColumns {
        ss_type: match table.ss_type {
            SsType::Type3 => "III",
            SsType::Type2 => "II",
        },
        test,
        reason,
        label: rows.iter().map(|r| r.label.clone()).collect(),
        kind: rows.iter().map(|r| kind_tag(r.test.kind)).collect(),
        statistic: col(|r| r.test.statistic),
        num_df: col(|r| r.test.num_df),
        den_df: col(|r| den_df_fields(r.test.reference).0),
        den_df_kind: rows
            .iter()
            .map(|r| den_df_fields(r.test.reference).1)
            .collect(),
        p_value: col(|r| r.test.p_value),
        sum_sq: col(|r| r.sum_sq.unwrap_or(f64::NAN)),
        mean_sq: col(|r| r.mean_sq.unwrap_or(f64::NAN)),
        eta2_partial: col(FixedRow::eta2_partial),
        row_notes: rows
            .iter()
            .map(|r| {
                r.test
                    .notes
                    .iter()
                    .map(|n| linear_note_line(*n, &table.coef_names))
                    .collect()
            })
            .collect(),
        notes: table.notes.clone(),
    }
}

/// A row's denominator df and its source, as [`TermColumns::den_df`] and
/// [`TermColumns::den_df_kind`] name them: NaN and `""` off an F reference.
pub(super) fn den_df_fields(reference: Reference) -> (f64, &'static str) {
    match reference {
        Reference::F {
            den_df,
            den_df_kind,
            ..
        } => (
            den_df,
            match den_df_kind {
                DenDf::Satterthwaite => "satterthwaite",
                DenDf::Residual => "residual",
                DenDf::PustejovskyTipton => "pustejovsky_tipton",
                DenDf::BellMcCaffrey => "bell_mccaffrey",
            },
        ),
        _ => (f64::NAN, ""),
    }
}

/// Fit `formula` against pre-marshalled columns and return the flattened
/// [`FitResult`] — the one entry point both FFI ports call, through
/// `glmm-bind`, which builds `opts` from the ports' arguments. `start` is the
/// warm start, each of its vectors empty to cold-start that part.
///
/// Two options are resolved on the lowered model, after a single-level
/// grouping has been dropped, and stripped with a warning where they do not
/// apply ([`FitResult::agq_warning`], [`FitResult::reml_warning`]): an `nagq`
/// above 1 on a shape adaptive quadrature does not take is fitted with
/// Laplace, and `reml`:
/// - a non-Gaussian family (GLM or GLMM): `Some(true)` is ignored, the model is
///   fitted by ML exactly as under `None`, and [`FitResult::reml_warning`] is
///   set; `Some(false)` gives the `None` fit with no warning;
/// - Gaussian with no random part (OLS): without weights or with a fixed
///   weight power, `Some(true)` gives the same estimates and reports the REML
///   log-likelihood, and `Some(false)` gives the `None` fit. Where the default
///   weight power is estimated, `None` and `Some(true)` estimate it by REML
///   and `Some(false)` takes the weights as given (power 1), so the estimates
///   differ;
/// - Gaussian mixed model: `None` and `Some(true)` fit by REML, `Some(false)`
///   by ML.
///
/// Every other value reaches the fit as given.
///
/// `anova_test` says which per-term tables the fit stores, as the ports'
/// `anova(m, test)` names them. Every fit stores its F or Wald χ² tables
/// ([`FitResult::term_table_type3`], `term_table_type2`). [`TermTest::Lrt`]
/// stores the likelihood-ratio tables as well
/// ([`FitResult::term_table_type3_lrt`], `term_table_type2_lrt`), which refit
/// the model once per distinct reduced design after the fit, and so does
/// [`TermTest::Auto`] where `"auto"` gives them (a Poisson model with random
/// effects, [`FitResult::term_test_auto`]); [`TermTest::Wald`] stores none.
///
/// # Errors
///
/// A held dispersion that is not finite and `> 0`, a warm start of the wrong
/// width or with a value that is not finite, `weights`/`offset` given that the
/// formula also gives, and every lowering and fit error.
#[allow(clippy::too_many_arguments)]
pub fn run_fit(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    opts: FitOptions,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    start: Option<StartValues>,
    anova_test: TermTest,
) -> Result<FitResult, PortError> {
    // The fit entry's own checks of `dispersion` and the warm start name kernel
    // struct fields (`FitOptions.dispersion`, `StartValues.beta`) a port user
    // never sees, so both are checked here first under the argument's name.
    if let Dispersion::Fixed(d) = opts.dispersion {
        if !(d.is_finite() && d > 0.0) {
            return Err(PortError::argument(format!(
                "dispersion must be a finite number > 0, got {d}"
            )));
        }
    }
    let table = table_of(numeric_columns, factor_columns)?;
    let prepared = lower_with(formula, &table, family, opts, weights, offset)?;
    let lowered = &prepared.lowered;

    // Widths as the fit entry checks them: exact, or empty to cold-start that part.
    if let Some(StartValues {
        beta,
        theta,
        dispersion: disp,
    }) = &start
    {
        if let Some(d) = disp {
            if !(d.is_finite() && *d > 0.0) {
                return Err(PortError::argument(format!(
                    "warm-start dispersion must be a finite number > 0, got {d}"
                )));
            }
        }
        if !beta.is_empty() && beta.len() != lowered.p {
            return Err(PortError::argument(format!(
                "warm-start beta has {} entries but the model has {} fixed-effect \
                 coefficients; give {}, or none to cold-start beta",
                beta.len(),
                lowered.p,
                lowered.p
            )));
        }
        let n_theta = theta_width(lowered.model.re.as_ref());
        if !theta.is_empty() && theta.len() != n_theta {
            return Err(PortError::argument(format!(
                "warm-start theta has {} entries but the model has {} random-effect \
                 Cholesky entries; give {}, or none to cold-start theta",
                theta.len(),
                n_theta,
                n_theta
            )));
        }
        for (name, values) in [("beta", beta), ("theta", theta)] {
            if let Some(v) = values.iter().find(|v| !v.is_finite()) {
                return Err(PortError::argument(format!(
                    "warm-start {name} must be finite numbers, got {v}"
                )));
            }
        }
    }

    let fit = fit_lowered(lowered, &prepared.opts, start.as_ref())?;
    Ok(flatten(prepared, fit, anova_test)?)
}

/// The default seed of the exact restricted likelihood-ratio test in a
/// comparison ([`run_compare`], and the ports' `anova(m0, m1)`): the same call
/// gives the same p-value on every run and in both ports, whatever the caller's
/// own random state. Any fixed value serves.
pub const DEFAULT_SEED: u64 = 20_261_008;

/// The default number of draws of the exact test's simulated null, RLRsim's
/// default (Monte Carlo standard error about 2.2e-3 at p = 0.05).
pub const DEFAULT_NSIM: usize = 10_000;

/// The largest `nsim` [`compare_options`] and [`simulate`](super::simulate) accept: 10⁶ draws,
/// a hundred times the default. The null keeps every draw in memory (16 bytes each), and the
/// time grows linearly with the draws and the group count.
pub const MAX_NSIM: usize = 1_000_000;

/// The options of a comparison from the ports' optional arguments: `None`
/// takes [`DEFAULT_NSIM`] and [`DEFAULT_SEED`].
///
/// # Errors
///
/// When `nsim` is 0 or above [`MAX_NSIM`].
pub fn compare_options(nsim: Option<usize>, seed: Option<u64>) -> Result<CompareOptions, String> {
    let nsim = nsim.unwrap_or(DEFAULT_NSIM);
    if !(1..=MAX_NSIM).contains(&nsim) {
        return Err(format!(
            "nsim must be a whole number from 1 to {MAX_NSIM}, got {nsim}"
        ));
    }
    Ok(CompareOptions {
        nsim,
        seed: seed.unwrap_or(DEFAULT_SEED),
    })
}

/// What [`run_compare`] returns: the comparison and the two fits, `fit0` the
/// fit of `f0` and `fit1` that of `f1`, as given.
#[derive(Debug)]
pub struct CompareResult {
    /// The comparison of the two fits.
    pub comparison: Comparison,
    /// The fit of the first formula.
    pub fit0: FitResult,
    /// The fit of the second formula.
    pub fit1: FitResult,
    /// The draws requested of the exact test (the comparison's reference
    /// carries the draws made, 0 when none were needed).
    pub nsim: usize,
    /// The seed of the exact test.
    pub seed: u64,
}

/// `compare(data, f0, f1)` of the ports: fits two nested formulas on the same
/// columns and compares the two fits ([`crate::formula::anova::compare_stored`]).
/// `family`, `opts`, `weights` and `offset` are given once and used for both
/// fits, which are made as [`run_fit`] makes them, with no warm start and
/// [`TermTest::Wald`], so neither stores a likelihood-ratio table. The caller
/// drops the rows with a missing value in any variable of either formula, the
/// weights or the offset before the call.
///
/// The estimator follows from what differs, so `opts.reml` is not read: on a
/// Gaussian model, both fits are REML fits when only the random part differs
/// (the OLS side made with `reml = true`, so its log-likelihood is the REML
/// value) and ML fits when the fixed part differs, alone or with the random
/// part; a GLM or GLMM is fitted by ML. `nsim` and `seed` are the exact test's
/// ([`compare_options`]).
///
/// A `None` `opts.weight_power` resolves as [`crate::FitOptions::weight_power`]
/// does, with compare's criterion for the fit's `reml`: to
/// [`WeightPower::Estimate`] when both models take it (Gaussian, weights that
/// vary, at most one grouping factor, at least 20 rows), so with different
/// fixed parts both sides are ML fits that each estimate their power, and
/// otherwise to the weights as given.
///
/// # Errors
///
/// An `nsim` out of range; formulas that are not nested, or that have the
/// same fixed and random parts and the same `offset()` term
/// ([`CompareError::SameParts`]), and an estimated power with the same fixed
/// parts ([`WeightPowerMismatch::DeltaRandomPartOnly`]), each refused before
/// either fit is made; any
/// error of [`run_fit`] on either fit; and every refusal of the comparison
/// (a fit that did not converge, different offsets, nothing to test).
#[allow(clippy::too_many_arguments)]
pub fn run_compare(
    f0: &str,
    f1: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: Family,
    opts: FitOptions,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    nsim: Option<usize>,
    seed: Option<u64>,
) -> Result<CompareResult, PortError> {
    let compare = compare_options(nsim, seed)?;
    let nested = nesting(f0, f1).map_err(|e| e.to_string())?;
    // The comparison's own refusal of two equal parts, made before either
    // fit. `nesting` does not read the offset, so formulas whose `offset()`
    // terms differ are fitted, and the comparison names the offsets first.
    if nested == Nesting::Same && same_offset_term(f0, f1) {
        return Err(CompareError::SameParts.to_string().into());
    }
    let reml = (family == Family::Gaussian).then_some(matches!(
        nested,
        Nesting::Nested {
            differs: Differs::Random,
            ..
        }
    ));
    let table = table_of(numeric_columns, factor_columns)?;
    let lower = |formula: &str, weights, offset| {
        let opts = FitOptions {
            reml,
            ..opts.clone()
        };
        lower_with(formula, &table, family, opts, weights, offset)
    };
    let mut prepared0 = lower(f0, weights.clone(), offset.clone())?;
    let mut prepared1 = lower(f1, weights, offset)?;
    // `None` resolves as a fit's does, with compare's own criterion for
    // the fit's `reml`, to one power for both sides: the fitted power where
    // both models take it.
    let power = match opts.weight_power {
        Some(v) => v,
        None => {
            let estimated = |p: &Prepared| {
                let opts = FitOptions {
                    reml: None,
                    weight_power: None,
                    ..p.opts.clone()
                };
                crate::fit::resolve_weight_power(p.lowered.spec(), &p.lowered.data(), &opts)
                    .is_ok_and(|v| v == WeightPower::Estimate)
            };
            if estimated(&prepared0) && estimated(&prepared1) {
                WeightPower::Estimate
            } else {
                WeightPower::Fixed(1.0)
            }
        }
    };
    if power == WeightPower::Estimate
        && matches!(
            nested,
            Nesting::Nested {
                differs: Differs::Random,
                ..
            }
        )
    {
        let e = CompareError::WeightPowerMismatch(WeightPowerMismatch::DeltaRandomPartOnly);
        return Err(e.to_string().into());
    }
    let side = |formula: &str, prepared: &mut Prepared| -> Result<_, PortError> {
        prepared.opts.weight_power = Some(power);
        let fit = fit_lowered(&prepared.lowered, &prepared.opts, None)?;
        let model = Model::from_fit(formula, &prepared.lowered, &prepared.opts, &fit);
        Ok((model, fit))
    };
    let (m0, fit0) = side(f0, &mut prepared0)?;
    let (m1, fit1) = side(f1, &mut prepared1)?;
    let comparison = compare_stored(&m0, &m1, &compare).map_err(|e| e.to_string())?;
    Ok(CompareResult {
        comparison,
        fit0: flatten(prepared0, fit0, TermTest::Wald)?,
        fit1: flatten(prepared1, fit1, TermTest::Wald)?,
        nsim: compare.nsim,
        seed: compare.seed,
    })
}

/// Whether two formulas, both of which parse, have the same `offset()` term
/// (or none).
fn same_offset_term(f0: &str, f1: &str) -> bool {
    let offset = |f: &str| parse(f).map(|p| p.offset).ok();
    offset(f0) == offset(f1)
}

/// A lowered model with every fit argument applied ([`lower_with`]): the
/// weights and offset on the lowering, the rest in the options it is fitted
/// with; and the two warn-and-strip flags.
pub(super) struct Prepared {
    pub(super) lowered: Lowered,
    pub(super) opts: FitOptions,
    agq_warning: bool,
    reml_warning: bool,
}

/// `formula` lowered on `table`, with the `weights=`/`offset=` arguments
/// applied to the lowering (an error when the formula gives the same thing)
/// and `opts`' `nagq` and `reml` stripped where the lowered model does not
/// take them ([`run_fit`] states the rule). [`run_fit`] and [`run_compare`]
/// share it, so a model lowers the same way in both.
pub(super) fn lower_with(
    formula: &str,
    table: &Table,
    fam: Family,
    mut opts: FitOptions,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
) -> Result<Prepared, PortError> {
    let mut lowered = lower(formula, table, fam)?;

    // nagq>1 on an ineligible shape is valid-but-inapplicable — warn-and-strip:
    // the fit entry ignores it on a Gaussian or fixed-only model and refuses it
    // on the other ineligible shapes, and a port user gets a warning and a
    // Laplace fit for all of them. The shape is only knowable here, after
    // `lower()`, so this is the strip site; `glmm-bind` words the message
    // (`options::argument_notes`) and each port raises it (Python
    // `UserWarning`, R `warning()`). Eligibility is the fit's own
    // rule, `fit::common::agq_shape`, on a mixed model.
    let mut agq_warning = false;
    if opts.nagq > 1 {
        let eligible = lowered
            .model
            .re
            .as_ref()
            .is_some_and(|re| agq_shape(fam, re).is_ok());
        if !eligible {
            agq_warning = true;
            opts.nagq = 1;
        }
    }
    // `reml` is resolved here too, on the lowered model, so both ports follow
    // one rule; a single-level grouping the lowering dropped has already made a
    // mixed formula an OLS or GLM fit. A GLM or GLMM has no REML/ML split:
    // `Some(true)` there is stripped with a warning (the fit is ML, the `None`
    // fit), and `Some(false)` is the `None` fit, silently. Every other value
    // goes to `FitOptions::reml`, which resolves it on a Gaussian model.
    let mut reml_warning = false;
    if opts.reml == Some(true) && lowered.model.family != Family::Gaussian {
        reml_warning = true;
        opts.reml = None;
    }
    // Mirrors `formula::materialize`'s `weights`/`offset` — change together:
    // lowering sets `weights`/`offset` only for a `cbind()` response /
    // `offset()` formula term, so a value here AND the matching argument is
    // the user saying it twice.
    if weights.is_some() && lowered.weights.is_some() {
        return Err(
            "weights given both as a cbind() response and as the weights= argument; use one"
                .to_string()
                .into(),
        );
    }
    if weights.is_some() {
        lowered.weights = weights;
    }
    if offset.is_some() && lowered.offset.is_some() {
        return Err(
            "offset given both as an offset() formula term and as the offset= argument; use one"
                .to_string()
                .into(),
        );
    }
    if offset.is_some() {
        lowered.offset = offset;
    }
    Ok(Prepared {
        lowered,
        opts,
        agq_warning,
        reml_warning,
    })
}

/// The kernel fit of `lowered` with `opts`, a refused input
/// ([`crate::FitError`]) returned as `Err` with its message.
pub(super) fn fit_lowered(
    lowered: &Lowered,
    opts: &FitOptions,
    start: Option<&StartValues>,
) -> Result<Fit, PortError> {
    let data = lowered.data();
    match start {
        Some(s) => crate::fit_with_start(&data, lowered.spec(), s, opts),
        None => crate::fit(&data, lowered.spec(), opts),
    }
    .map_err(PortError::from)
}

/// `fit` of `prepared` as a [`FitResult`], with its two per-term tables, and
/// the two likelihood-ratio tables when `anova_test` ([`run_fit`]'s) gives
/// them.
pub(super) fn flatten(
    prepared: Prepared,
    fit: Fit,
    anova_test: TermTest,
) -> Result<FitResult, String> {
    let Prepared {
        mut lowered,
        opts: fit_opts,
        agq_warning,
        reml_warning,
    } = prepared;
    // The LOWERING's observations, decided before any solver ran.
    let lowered_notes = std::mem::take(&mut lowered.notes);

    let (coef_p, df_method) = t_tests(&fit, &fit_opts.se_for.columns(lowered.p));
    // Also on a robust fit, whose joint tests are model-based.
    let satterthwaite_record = match fit.df_state() {
        Ok(st) if !st.residual_only() => Some(record_of(st)),
        _ => None,
    };
    let (rss, r_squared, adj_r_squared) =
        if lowered.model.re.is_none() && lowered.model.family == Family::Gaussian {
            ols_r_squared(
                &fit,
                &lowered.y,
                crate::fit::fitted_weights(lowered.weights.as_deref(), fit.weight_power).as_deref(),
                lowered.offset.as_deref(),
                lowered.has_intercept,
            )
        } else {
            (f64::NAN, f64::NAN, f64::NAN)
        };

    // The per-term tables, while `lowered` and `fit` are whole: the F or
    // Wald χ² tables of both types, and both likelihood-ratio tables when
    // `anova_test` gives them, which share their refits. A panic in one is
    // stored in its slot, marked as an internal error so a port never shows a
    // bug as a refusal: the fit itself stands.
    let mut cache = RefitCache::default();
    let mut term_table = |ss_type, test| {
        catch_unwind(AssertUnwindSafe(|| {
            let opts = TermTestOptions { ss_type, test };
            term_tests_cached(&lowered, &fit_opts, &fit, &opts, &mut cache)
        }))
        .map_err(|payload| {
            TermTableError::Internal(format!(
                "internal error while building the per-term table (please report it): {}",
                panic_message(payload)
            ))
        })?
        .map_err(|e| TermTableError::Refused(e.to_string()))
    };
    let term_test_auto = auto_choice(&lowered, &fit);
    let auto_lrt = term_test_auto.test == LinearKind::LikelihoodRatio;
    let wald = if auto_lrt {
        TermTest::Wald
    } else {
        TermTest::Auto
    };
    let lrt = match anova_test {
        TermTest::Lrt => Some(TermTest::Lrt),
        TermTest::Auto if auto_lrt => Some(TermTest::Auto),
        _ => None,
    };
    let term_table_type3 = term_table(SsType::Type3, wald);
    let term_table_type2 = term_table(SsType::Type2, wald);
    let term_table_type3_lrt = lrt.map(|test| term_table(SsType::Type3, test));
    let term_table_type2_lrt = lrt.map(|test| term_table(SsType::Type2, test));

    // `varcorr` and `re_groups` are both emitted in RE declaration order
    // (primary, then each extra), so index i of one names index i of the other.
    // Check rather than trust it: a silent misalignment relabels a variance
    // component in a port's `summary()`, which reads as a wrong answer, not a
    // cosmetic slip. Labelled BEFORE `re_groups` is flattened — `label_ranef`
    // needs the slot labels, which the tuple form drops. A shape mismatch here
    // is a kernel bug, not a user error, so it surfaces as an `Err` rather than
    // being swallowed.
    let ranef_blocks: Vec<RanefBlock> = label_ranef(&fit, &lowered.re_groups)
        .map_err(|e| e.to_string())?
        .into_iter()
        .map(|b| RanefBlock {
            group: b.group,
            terms: b.terms,
            levels: b.levels,
            values: b.values,
            condvar: b.condvar,
        })
        .collect();
    // The lowering's own observations (dropped or thin groupings) join the
    // solver's in one channel: the user does not care which layer noticed.
    // Named before `re_groups` is flattened below.
    let notes = note_infos(
        lowered_notes
            .into_iter()
            .chain(fit.diagnostics.notes.iter().cloned())
            .collect(),
        &lowered.re_groups,
    );
    let re_group_levels = lowered
        .re_groups
        .iter()
        .map(|g| g.slot_labels.iter().flatten().count())
        .collect();
    let (re_groups, re_term_index): (Vec<_>, Vec<_>) = lowered
        .re_groups
        .into_iter()
        .map(|g| ((g.name, g.terms), g.term_index))
        .unzip();
    // A block list of another length than the lowered groupings is a kernel bug.
    if fit.varcorr.len() != re_groups.len() {
        return Err(format!(
            "re_groups and varcorr must agree in length and order: \
             {} grouping(s) lowered, {} varcorr block(s) returned",
            re_groups.len(),
            fit.varcorr.len()
        ));
    }

    // Taken out whole before the field-by-field move below: the forwarding
    // accessors borrow all of `fit`, which a partial move rules out.
    let vcov = fit.vcov_rows();
    let p = fit.beta.len();
    let model_vcov = fit
        .model_vcov
        .chunks_exact(p.max(1))
        .map(<[f64]>::to_vec)
        .collect();
    let diagnostics = fit.diagnostics;
    Ok(FitResult {
        beta: fit.beta,
        se: fit.se,
        vcov,
        tau2: fit.tau2,
        varcorr: fit.varcorr,
        stddev_se: fit.stddev_se,
        aliased: diagnostics.aliased,
        dispersion: fit.dispersion,
        converged: diagnostics.converged,
        stop: diagnostics.stop,
        n_eval: fit.n_eval,
        n_grad: diagnostics.n_grad,
        outer_stop: diagnostics.outer_stop.map(outer_stop_name),
        outer_decrement: diagnostics.outer_decrement,
        deviance: fit.deviance,
        singular: diagnostics.singular,
        names: lowered.col_names,
        re_groups,
        re_term_index,
        re_group_levels,
        agq_warning,
        reml_warning,
        loglik: fit.loglik,
        df: fit.df,
        reml: fit.reml,
        fitted: fit.fitted,
        ranef: fit.ranef,
        ranef_levels: fit.ranef_levels,
        ranef_blocks,
        condvar: fit.condvar,
        condvar_x: fit.condvar_x,
        condvar_cross: fit.condvar_cross.map(CondvarCrossRecord::from),
        y: lowered.y,
        weights: lowered.weights,
        weight_power: fit.weight_power,
        weight_power_estimated: fit.weight_power_estimated,
        se_type: fit.se_type,
        model_vcov,
        robust: fit.robust.map(RobustRecord::from),
        offset: lowered.offset,
        design_summary: fit.design_summary.map(|d| DesignSummary {
            counts: d.counts,
            s: d.s,
            c: d.c,
        }),
        nobs: lowered.n,
        boundary: boundary_name(diagnostics.boundary),
        pinned: diagnostics.pinned,
        notes,
        coef_df: fit.coef_df,
        coef_p,
        df_method,
        satterthwaite_record,
        rss,
        r_squared,
        adj_r_squared,
        objective: fit.objective,
        term_table_type3,
        term_table_type2,
        term_table_type3_lrt,
        term_test_auto,
        term_table_type2_lrt,
    })
}

/// Where a fit's per-coefficient df come from ([`FitResult::df_method`]).
/// The ports carry it as the text [`as_str`](Self::as_str) gives, and print
/// t on the four kinds that [`tests_with_t`](Self::tests_with_t), z on the
/// rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DfMethod {
    /// `"satterthwaite"`: a Gaussian LMM's Satterthwaite pass.
    Satterthwaite,
    /// `"residual"`: `n − p`, on OLS and the dispersion GLMs with φ
    /// estimated, whatever `satterthwaite` says.
    Residual,
    /// `"none"`: the families with no df.
    NoDf,
    /// `"disabled"`: a Gaussian LMM fitted with `satterthwaite = false`.
    Disabled,
    /// `"failed"`: a Gaussian LMM whose Satterthwaite pass failed.
    Failed,
    /// `"pustejovsky_tipton"`: a robust fit with a grouping factor (CR2).
    PustejovskyTipton,
    /// `"bell_mccaffrey"`: a robust fit without one (HC2).
    BellMcCaffrey,
}

impl DfMethod {
    /// Every kind, in declaration order.
    pub const ALL: [Self; 7] = [
        Self::Satterthwaite,
        Self::Residual,
        Self::NoDf,
        Self::Disabled,
        Self::Failed,
        Self::PustejovskyTipton,
        Self::BellMcCaffrey,
    ];

    /// The text the ports store.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Satterthwaite => "satterthwaite",
            Self::Residual => "residual",
            Self::NoDf => "none",
            Self::Disabled => "disabled",
            Self::Failed => "failed",
            Self::PustejovskyTipton => "pustejovsky_tipton",
            Self::BellMcCaffrey => "bell_mccaffrey",
        }
    }

    /// The kind [`as_str`](Self::as_str) spells as `s`.
    ///
    /// # Errors
    ///
    /// A text that is none of the seven.
    pub fn parse(s: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| format!("unknown df_method {s:?}"))
    }

    /// The fit tests with t: the residual, Satterthwaite and the two robust
    /// df. The rest test with z.
    #[must_use]
    pub fn tests_with_t(self) -> bool {
        self.per_row_df() || self == Self::Residual
    }

    /// The df differ per coefficient, so the ports print a df column: the
    /// Satterthwaite and the two robust df.
    #[must_use]
    pub fn per_row_df(self) -> bool {
        self == Self::Satterthwaite || self.is_robust()
    }

    /// A robust fit.
    #[must_use]
    pub fn is_robust(self) -> bool {
        matches!(self, Self::PustejovskyTipton | Self::BellMcCaffrey)
    }

    /// The fit's tests read the residual df `n − p`: OLS and the dispersion
    /// GLMs, and a robust fit without a grouping factor.
    #[must_use]
    pub fn residual_df(self) -> bool {
        matches!(self, Self::Residual | Self::BellMcCaffrey)
    }
}

/// The t test of each target coefficient (`Fit::contrast_t` on its unit
/// vector) as `(coef_p, df_method)`. The kind keys on the fit's df record,
/// not on its notes: a failed pass is the `NoDfReason::Failed` record.
fn t_tests(fit: &Fit, targets: &[u32]) -> (Vec<f64>, DfMethod) {
    let p = fit.beta.len();
    let mut coef_p = vec![f64::NAN; p];
    if fit.se_type == SeType::Robust {
        for &j in targets {
            let j = j as usize;
            coef_p[j] = t_two_sided_p(fit.beta[j] / fit.se[j], fit.coef_df[j]);
        }
        let method = if fit.ranef_levels.is_empty() {
            DfMethod::BellMcCaffrey
        } else {
            DfMethod::PustejovskyTipton
        };
        return (coef_p, method);
    }
    let df_method = match fit.df_state() {
        Ok(st) if st.residual_only() => DfMethod::Residual,
        Ok(_) => DfMethod::Satterthwaite,
        Err(ContrastError::NoDf(reason)) => match reason {
            NoDfReason::NotGaussianMixed => DfMethod::NoDf,
            NoDfReason::Disabled => DfMethod::Disabled,
            // The not-run record is the loop tier's, which `run_fit` never
            // builds; were it to happen, its df are missing as after a failed
            // pass.
            NoDfReason::Failed | NoDfReason::StateMissing => DfMethod::Failed,
        },
    };
    if df_method.tests_with_t() {
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
/// intercept (or nothing); NaN for both R² when neither `mss` nor `rss` is
/// above round-off of the response's sum of squares `Σwᵢ(yᵢ − oᵢ)²`; NaN for
/// all three on a non-converged fit.
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
    // The factor 64 leaves a margin of about 250 over a constant response's
    // round-off; a larger one starts to blank real fits.
    let tiny = (64.0 * f64::EPSILON).powi(2)
        * (0..n)
            .map(|i| w(i) * (y[i] - offset.map_or(0.0, |o| o[i])).powi(2))
            .sum::<f64>();
    if mss <= tiny && rss <= tiny {
        return (rss, f64::NAN, f64::NAN);
    }
    let r2 = mss / (mss + rss);
    let adj = 1.0 - (1.0 - r2) * ((n - df_int) as f64 / (n - rank) as f64);
    (rss, r2, adj)
}
