//! String-typed fit orchestration shared by the FFI ports (`glmm-python`,
//! `glmm-r`): formula + data -> [`crate::formula::lower`] -> option overrides
//! -> [`crate::fit_warm`], wrapped in `catch_unwind` so the kernel's
//! `assert!`-based boundary faults become a normal `Err`, never a process
//! abort across an FFI boundary. Plain-Rust types only — no `pyo3`/`extendr`
//! types — so everything here runs under plain `cargo test`, and the two
//! ports flatten the same fit the same way from one definition instead of
//! two mirrored copies. [`run_fit`] fits one formula; [`run_compare`] fits
//! two on the same rows and compares them; [`run_model_matrix`] returns a
//! formula's fixed-effects design without fitting; [`term_labels`] reads a
//! formula's fixed-term labels; [`contrast_test`] tests `Lβ = 0` on what a
//! port stored of one fit.
//!
//! Behind the default-off `orchestrate` cargo feature. Like `loop_advanced`
//! it carries NO semver guarantees: its shape follows the ports' needs.

use std::collections::HashMap;
use std::panic::{catch_unwind, AssertUnwindSafe};

use faer::Mat;

use crate::fit::theta_width;
use crate::formula::anova::{
    auto_choice, compare, fixed_row, linear_note_line, nesting, term_tests_cached, untested_test,
    ChoiceReason, CompareError, CompareOptions, Comparison, Differs, FixedRow, Model, Nesting,
    RefitCache, SsType, TermTable, TermTest, TermTestOptions, TestChoice,
};
use crate::formula::{
    design_columns, fixed_design, fixed_part_relation, label_ranef, lower, parse, r_terms, Column,
    FixedPartRelation, Lowered, ParsedFormula, Table,
};
use crate::inference::contrast::{contrast_f_of, ContrastError, NoDfReason};
use crate::inference::tails::{normal_quantile, t_quantile};
use crate::inference::wald::wald_chisq;
use crate::inference::{DenDf, LinearKind, LinearNote, LinearTest, Objective, Reference};
use crate::lmm::satterthwaite::SatterthwaiteState;
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
        "gaussian" => match link {
            "identity" => Ok(Family::Gaussian),
            other => Err(format!("unsupported gaussian link {other:?}")),
        },
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
    /// Free text a variant needs and the fields above cannot carry (the
    /// grouping and level names of `UnusedGroupingLevels`, the grouping name
    /// of `SingleLevelGroupingDropped`, and `FewRowsForRandomEffects`'s term
    /// as `"<terms> | <grouping>"`, whose row count the ports read from
    /// `FitResult::nobs`). Empty otherwise; the `kind`, not this string, stays
    /// the stable identifier.
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
        Boundary::UpperLimit => "upper_limit",
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
            Note::FewRowsForRandomEffects {
                terms,
                grouping,
                random_effects,
                rows: _,
            } => NoteInfo {
                kind: "few_rows_for_random_effects",
                columns: Vec::new(),
                pivot: f64::NAN,
                evals: random_effects,
                final_eval: false,
                detail: format!("{terms} | {grouping}"),
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
            Note::VarianceAtLimit => NoteInfo {
                kind: "variance_at_limit",
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
    /// The offset the kernel fitted with, from the `offset=` argument or an
    /// `offset()` term of the formula; `None` without one. Kept so a
    /// comparison of two stored fits can check that their offsets agree.
    /// Coupled sites, as for `y`.
    pub offset: Option<Vec<f64>>,
    /// The design summary `(counts, s, c)` of a Gaussian fit whose only random
    /// term is one random intercept, with no prior weights and at least one
    /// fixed-effect column; `None` on every other fit. With `K` the highest
    /// observed level code plus one (a level below it with no rows counts 0)
    /// and `p` retained fixed-effect columns, `counts` has `K` entries, `s` is
    /// the `p × K` matrix `X̃ᵀZ` and `c` the `p × p` matrix `X̃ᵀX̃`, both flat
    /// and column-major, on the fit's orthogonalized design `X̃`
    /// (`crate::fit::DesignSummary` has the detail). Coupled sites, as for
    /// `y`: `fit_dict` and the R port's returned list flatten it.
    pub design_summary: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
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
    /// The Satterthwaite record [`contrast_test`] reads, on a fit whose
    /// `df_method` is `"satterthwaite"`; `None` on every other fit, whose
    /// test reads only `beta`, `vcov`, `aliased`, `nobs` and `dispersion`.
    /// Coupled sites, as for `y`.
    pub satterthwaite_record: Option<SatterthwaiteRecord>,
    /// Weighted residual sum of squares `Σwᵢ(yᵢ − μ̂ᵢ)²` of an OLS fit; NaN on
    /// every other fit and on a non-converged one.
    pub rss: f64,
    /// `summary.lm`'s multiple R² of an OLS fit (0 on an intercept-only
    /// model); NaN on every other fit and on a non-converged one.
    pub r_squared: f64,
    /// `summary.lm`'s adjusted R², with the same NaN and 0 cases as
    /// `r_squared`.
    pub adj_r_squared: f64,
    /// What the fit maximised (`crate::Fit::objective`); [`objective_parts`]
    /// flattens it to a label and a node count, [`objective_from_parts`]
    /// rebuilds it from them. Coupled sites, as for `y`.
    pub objective: Objective,
    /// The Type III per-term table
    /// ([`term_tests`](crate::formula::anova::term_tests)), built at fit time so a
    /// port's `anova(m)` reads it and computes nothing; with an empty cell,
    /// the missing-cell note is among its notes. `Err` holds the refusal's
    /// message (a formula whose interaction lacks a marginal term), which the
    /// fit itself does not raise, or, after a panic while the table was
    /// built, a message that starts "internal error". [`term_columns`]
    /// flattens a table. The F or Wald χ² table: built under
    /// [`TermTest::Auto`], so its [`TermTable::choice`] is the test `"auto"`
    /// gives and why, except on a Poisson model with random effects, whose
    /// `"auto"` is the likelihood-ratio table (`term_table_type3_lrt`), and
    /// whose table here is built under [`TermTest::Wald`]. Coupled sites, as
    /// for `y`.
    pub term_table_type3: Result<TermTable, String>,
    /// The Type II per-term table, as `term_table_type3`.
    pub term_table_type2: Result<TermTable, String>,
    /// The Type III likelihood-ratio table
    /// ([`term_tests`](crate::formula::anova::term_tests) under
    /// [`TermTest::Lrt`]): `None` unless the fit stores it, under
    /// [`run_fit`]'s `anova_test = "LRT"`, and under `"auto"` on a Poisson
    /// model with random effects (built under [`TermTest::Auto`]: the same
    /// rows, with its reason). `Err` as for `term_table_type3`, and also the
    /// refusal of a fit whose rows are F tests. Coupled sites, as for `y`.
    pub term_table_type3_lrt: Option<Result<TermTable, String>>,
    /// The Type II likelihood-ratio table, as `term_table_type3_lrt`.
    pub term_table_type2_lrt: Option<Result<TermTable, String>>,
    /// What `"auto"`, the default test of a port's `anova(m)`, gives on this
    /// fit and why: the F or Wald χ² of `term_table_type3` and
    /// `term_table_type2`, or, on a Poisson model with random effects, the
    /// likelihood-ratio χ² of `term_table_type3_lrt` and
    /// `term_table_type2_lrt`. [`choice_tags`] flattens it. Coupled sites, as
    /// for `y`.
    pub term_test_auto: TestChoice,
}

/// A Gaussian LMM's Satterthwaite record as plain numbers
/// ([`FitResult::satterthwaite_record`]): a port stores it on its fit and
/// passes it back to [`contrast_test`]. The `p_r × p_r` matrices are over the
/// retained columns, `p_r` of them, and every matrix is row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct SatterthwaiteRecord {
    /// `A_eff = 2·blockdiag(H_p⁺, τ̂²/ν)`, `(n_free + 1) × (n_free + 1)`, with
    /// `n_free` the free θ coordinates.
    pub a_eff: Vec<f64>,
    /// `∂C/∂θ̃_k`, one `p_r × p_r` matrix per free θ coordinate.
    pub jac_theta: Vec<Vec<f64>>,
    /// `C = Var(β̂)`, `p_r × p_r`.
    pub c: Vec<f64>,
    /// τ̂ = σ̂².
    pub tau: f64,
    /// The divisor of σ̂²: `N − p_r` on a REML fit, `N` on an ML fit.
    pub nu: f64,
    /// The retained columns, 0-based and ascending, on a rank-deficient fit;
    /// `None` when every column was retained.
    pub retained: Option<Vec<usize>>,
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
    /// Where an F row's denominator df comes from: `"satterthwaite"` or
    /// `"residual"`; `""` on a χ² row.
    pub den_df_kind: Vec<&'static str>,
    /// The p-value.
    pub p_value: Vec<f64>,
    /// lmerTest's `Sum Sq` on an F row; NaN on a χ² row.
    pub sum_sq: Vec<f64>,
    /// lmerTest's `Mean Sq` on an F row; NaN on a χ² row.
    pub mean_sq: Vec<f64>,
    /// Each row's notes as plain sentences (an aliased column, a rank below
    /// the row count, no denominator df and why; on a likelihood-ratio
    /// table, a refit that did not converge, a refit above the larger model
    /// by more than round-off (its statistic set to 0), and a random-slope
    /// term kept on its Wald row).
    pub row_notes: Vec<Vec<String>>,
    /// The table's notes as plain sentences (the missing-cell note; on a
    /// likelihood-ratio table, a fit that did not converge, with nothing
    /// refitted, and a term whose model under its hypothesis could not be
    /// built).
    pub notes: Vec<String>,
}

/// A row's or a table's test as [`TermColumns::kind`] names it.
fn kind_tag(kind: LinearKind) -> &'static str {
    match kind {
        LinearKind::F => "F",
        LinearKind::ChiSquared => "chisq",
        LinearKind::LikelihoodRatio => "lrt",
    }
}

/// A [`TestChoice`] as the two tags both FFI ports store: the test, named as
/// in [`TermColumns::kind`], and the reason, as in [`TermColumns::reason`].
pub fn choice_tags(choice: &TestChoice) -> (&'static str, &'static str) {
    let reason = match choice.reason {
        ChoiceReason::Requested => "requested",
        ChoiceReason::FTest => "f_test",
        ChoiceReason::PoissonMixed => "poisson_mixed",
        ChoiceReason::BinomialMixed => "binomial_mixed",
        ChoiceReason::NegativeBinomialMixed => "negative_binomial_mixed",
        ChoiceReason::OtherMixed => "other_mixed",
        ChoiceReason::NoRandomEffects => "no_random_effects",
    };
    (kind_tag(choice.test), reason)
}

/// A [`TermTable`] as [`TermColumns`], for both FFI ports.
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
fn den_df_fields(reference: Reference) -> (f64, &'static str) {
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
            },
        ),
        _ => (f64::NAN, ""),
    }
}

/// An [`Objective`] as `(label, node count)` for both FFI ports: the label is
/// one of `"least_squares"`, `"maximum_likelihood"`, `"reml"`, `"laplace"`
/// and `"agq"`; the node count is 1 on `"laplace"`, the quadrature's on
/// `"agq"` and 0 elsewhere.
pub fn objective_parts(objective: Objective) -> (&'static str, u8) {
    match objective {
        Objective::LeastSquares => ("least_squares", 0),
        Objective::MaximumLikelihood => ("maximum_likelihood", 0),
        Objective::Reml => ("reml", 0),
        Objective::Laplace => ("laplace", 1),
        Objective::Agq { nodes } => ("agq", nodes),
    }
}

/// The inverse of [`objective_parts`], for a port that rebuilds a fit's
/// [`Objective`] from the two stored fields.
///
/// # Errors
///
/// When `(label, nodes)` is not a pair [`objective_parts`] gives: an unknown
/// label, a node count other than 0 on `"least_squares"`,
/// `"maximum_likelihood"` and `"reml"` or other than 1 on `"laplace"`, or
/// fewer than 2 nodes on `"agq"`.
pub fn objective_from_parts(label: &str, nodes: u8) -> Result<Objective, String> {
    match (label, nodes) {
        ("least_squares", 0) => Ok(Objective::LeastSquares),
        ("maximum_likelihood", 0) => Ok(Objective::MaximumLikelihood),
        ("reml", 0) => Ok(Objective::Reml),
        ("laplace", 1) => Ok(Objective::Laplace),
        ("agq", 2..) => Ok(Objective::Agq { nodes }),
        _ => Err(format!(
            "no fit stores the objective {label:?} with {nodes} quadrature {}",
            if nodes == 1 { "node" } else { "nodes" }
        )),
    }
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
///
/// `anova_test` says which per-term tables the fit stores, with the values
/// of the ports' `anova(m, test)`. Every fit stores its F or Wald χ² tables
/// ([`FitResult::term_table_type3`], `term_table_type2`). `"LRT"` stores the
/// likelihood-ratio tables as well ([`FitResult::term_table_type3_lrt`],
/// `term_table_type2_lrt`), which refit the model once per distinct reduced
/// design after the fit, and so does `"auto"` where `"auto"` gives them (a
/// Poisson model with random effects, [`FitResult::term_test_auto`]);
/// `"Wald"` stores none. Any other value is an error, raised before the fit.
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
    anova_test: &str,
) -> Result<FitResult, String> {
    let anova_test = match anova_test {
        "auto" => TermTest::Auto,
        "Wald" => TermTest::Wald,
        "LRT" => TermTest::Lrt,
        other => {
            return Err(format!(
                r#"anova_test must be "auto", "Wald" or "LRT", got {other:?}"#
            ))
        }
    };
    let fam = family_from_str(family, link)?;
    // `fit_warm`'s own checks of `dispersion` and the warm start name kernel
    // struct fields (`FitOptions.dispersion`, `StartValues.beta`) a port user
    // never sees, so both are checked here first under the argument's name.
    if let Some(d) = dispersion {
        if !(d.is_finite() && d > 0.0) {
            return Err(format!("dispersion must be a finite number > 0, got {d}"));
        }
    }
    let table = table_of(numeric_columns, factor_columns)?;
    let prepared = lower_with(
        formula,
        &table,
        fam,
        wald_se,
        nagq,
        dispersion,
        weights,
        offset,
        satterthwaite,
        reml,
    )?;
    let lowered = &prepared.lowered;

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
        for (name, values) in [("beta", beta), ("theta", theta)] {
            if let Some(v) = values.iter().find(|v| !v.is_finite()) {
                return Err(format!("warm-start {name} must be finite numbers, got {v}"));
            }
        }
    }
    let start = warm_start.map(|(beta, theta, dispersion)| StartValues {
        beta,
        theta,
        dispersion,
    });

    let fit = fit_lowered(lowered, start.as_ref())?;
    flatten(prepared, fit, anova_test)
}

/// The default seed of the exact restricted likelihood-ratio test in a
/// comparison ([`run_compare`], and the ports' `anova(m0, m1)`): the same call
/// gives the same p-value on every run and in both ports, whatever the caller's
/// own random state. The value is a date, 2026-10-08, the day the default was
/// decided; any fixed value serves.
pub const DEFAULT_SEED: u64 = 20_261_008;

/// The default number of draws of the exact test's simulated null, RLRsim's
/// default (Monte Carlo standard error about 2.2e-3 at p = 0.05).
pub const DEFAULT_NSIM: usize = 10_000;

/// The largest `nsim` [`compare_options`] accepts: 10⁶ draws, a hundred times
/// the default. The null keeps every draw in memory (16 bytes each), and the
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
/// columns and compares the two fits ([`crate::formula::anova::compare`]).
/// `family`, `link`, `nagq`, `weights` and `offset` are given once and used
/// for both fits, which are made as [`run_fit`] makes them with
/// `wald_se = "hessian"`, `satterthwaite = true`, no warm start and
/// `anova_test = "Wald"`, so neither stores a likelihood-ratio table. The
/// caller drops the rows with a missing value in any variable of either
/// formula, the weights or the offset before the call.
///
/// The estimator follows from what differs, so there is no `reml` argument:
/// on a Gaussian model, both fits are REML fits when only the random part
/// differs (the OLS side made with `reml = true`, so its log-likelihood is
/// the REML value) and ML fits when the fixed part differs, alone or with
/// the random part; a GLM or GLMM is fitted by ML. `nsim` and `seed` are the
/// exact test's ([`compare_options`]).
///
/// # Errors
///
/// An `nsim` out of range; formulas that are not nested, or that have the
/// same fixed and random parts and the same `offset()` term
/// ([`CompareError::SameParts`]), refused before either fit is made; any
/// error of [`run_fit`] on either fit; and every refusal of the comparison
/// (a fit that did not converge, different offsets, nothing to test).
#[allow(clippy::too_many_arguments)]
pub fn run_compare(
    f0: &str,
    f1: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    family: &str,
    link: &str,
    nagq: u8,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    nsim: Option<usize>,
    seed: Option<u64>,
) -> Result<CompareResult, String> {
    let opts = compare_options(nsim, seed)?;
    let fam = family_from_str(family, link)?;
    let nested = nesting(f0, f1).map_err(|e| e.to_string())?;
    // The comparison's own refusal of two equal parts, made before either
    // fit. `nesting` does not read the offset, so formulas whose `offset()`
    // terms differ are fitted, and the comparison names the offsets first.
    if nested == Nesting::Same && same_offset_term(f0, f1) {
        return Err(CompareError::SameParts.to_string());
    }
    let reml = (fam == Family::Gaussian).then_some(matches!(
        nested,
        Nesting::Nested {
            differs: Differs::Random,
            ..
        }
    ));
    let table = table_of(numeric_columns, factor_columns)?;
    let side = |formula: &str, weights, offset| -> Result<_, String> {
        let prepared = lower_with(
            formula, &table, fam, "hessian", nagq, None, weights, offset, true, reml,
        )?;
        let fit = fit_lowered(&prepared.lowered, None)?;
        let model = Model::from_fit(formula, &prepared.lowered, &fit);
        Ok((model, prepared, fit))
    };
    let (m0, prepared0, fit0) = side(f0, weights.clone(), offset.clone())?;
    let (m1, prepared1, fit1) = side(f1, weights, offset)?;
    let comparison = compare(&m0, &m1, &opts).map_err(|e| e.to_string())?;
    Ok(CompareResult {
        comparison,
        fit0: flatten(prepared0, fit0, TermTest::Wald)?,
        fit1: flatten(prepared1, fit1, TermTest::Wald)?,
        nsim: opts.nsim,
        seed: opts.seed,
    })
}

/// Whether two formulas, both of which parse, have the same `offset()` term
/// (or none).
fn same_offset_term(f0: &str, f1: &str) -> bool {
    let offset = |f: &str| parse(f).map(|p| p.offset).ok();
    offset(f0) == offset(f1)
}

/// `formula` parsed for a fixed-effects design alone ([`model_matrix_columns`],
/// [`run_model_matrix`]), with whether it has no response. Refused: a formula
/// with no `~` (the parser would take its default response name for it), a
/// random-effect term (a fit drops the rows missing a grouping value, which
/// can remove a factor level and change the design), and an `offset()` term
/// (it adds no design column).
///
/// Both checks on `~` read the string, because the parsed formula cannot tell
/// `~ x` from `x`, nor `y = x` from `y ~ x`.
fn design_formula(formula: &str) -> Result<(ParsedFormula, bool), String> {
    if !formula.contains('~') {
        return Err(format!(
            "a design formula needs ~, as in \"~ x + f\" or \"y ~ x + f\"; got {formula:?}"
        ));
    }
    let one_sided = formula.trim_start().starts_with('~');
    let ast = parse(formula).map_err(|e| e.to_string())?;
    if !ast.random_effects.is_empty() {
        return Err(
            "a design formula takes no random-effect terms; give the fixed part only".to_string(),
        );
    }
    if ast.offset.is_some() {
        return Err(
            "a design formula takes no offset() term: an offset adds no design column".to_string(),
        );
    }
    Ok((ast, one_sided))
}

/// What [`run_model_matrix`] reads from a dataset whose column names are
/// `available`: `(used, missing, numeric_only)`, as the Python port's
/// `formula_columns` gives them for a fit. A response (`y ~ x`) is read, so
/// it must exist, though it is not part of the design; a one-sided formula
/// (`~ x`) reads none.
///
/// # Errors
///
/// The refusals of a design formula (no `~`, a random-effect term, an
/// `offset()` term) and a formula that does not parse.
#[allow(clippy::type_complexity)]
pub fn model_matrix_columns(
    formula: &str,
    available: &[String],
) -> Result<(Vec<String>, Vec<String>, Vec<String>), String> {
    let (ast, one_sided) = design_formula(formula)?;
    Ok(design_columns(&ast, available, one_sided))
}

/// The fixed-effects design of `formula` on pre-marshalled columns, as
/// `(x, n, p, col_names)`: the row-major `n × p` design and the coefficient
/// names [`run_fit`] gives the same formula, every column kept (an aliased one
/// too). `n_rows` is the caller's row count, read only when the formula
/// reads no column (`~ 1`); otherwise the columns give it. The caller drops
/// no row: a missing value is refused before the call.
///
/// # Errors
///
/// The refusals of [`model_matrix_columns`], the column errors of
/// [`run_fit`] (mismatched lengths, a bad factor code), and the design errors
/// of [`crate::formula::lower`] (an unknown column, a single-level factor, an
/// empty design, a non-finite value). A panic is returned as `Err`.
pub fn run_model_matrix(
    formula: &str,
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
    n_rows: usize,
) -> Result<(Vec<f64>, usize, usize, Vec<String>), String> {
    let (ast, _) = design_formula(formula)?;
    let mut table = table_of(numeric_columns, factor_columns)?;
    // `table_of` takes `n` from the columns, 0 when there are none.
    if table.columns.is_empty() {
        table.n = n_rows;
    }
    let design = catch_unwind(AssertUnwindSafe(|| fixed_design(&ast, &table)))
        .map_err(panic_message)?
        .map_err(|e| e.to_string())?;
    Ok((design.x, table.n, design.p, design.col_names))
}

/// The ports' columns as a [`Table`], every column the same length.
fn table_of(
    numeric_columns: HashMap<String, Vec<f64>>,
    factor_columns: HashMap<String, (Vec<String>, Vec<u32>)>,
) -> Result<Table, String> {
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
    Ok(Table { columns, n })
}

/// A lowered model with every fit argument applied ([`lower_with`]), and the
/// two warn-and-strip messages.
struct Prepared {
    lowered: Lowered,
    agq_warning: Option<String>,
    reml_warning: Option<String>,
}

/// `formula` lowered on `table`, with the fit arguments applied to its
/// options: `wald_se`, `nagq` (warn-and-strip), `reml` (resolved as
/// [`run_fit`] states), `dispersion`, `satterthwaite`, and the
/// `weights=`/`offset=` arguments (an error when the formula gives the same
/// thing). [`run_fit`] and [`run_compare`] share it, so a model lowers the
/// same way in both.
#[allow(clippy::too_many_arguments)]
fn lower_with(
    formula: &str,
    table: &Table,
    fam: Family,
    wald_se: &str,
    nagq: u8,
    dispersion: Option<f64>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    satterthwaite: bool,
    reml: Option<bool>,
) -> Result<Prepared, String> {
    let mut lowered = catch_unwind(AssertUnwindSafe(|| lower(formula, table, fam)))
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
    Ok(Prepared {
        lowered,
        agq_warning,
        reml_warning,
    })
}

/// The kernel fit of `lowered`, a panic caught and returned as `Err`.
fn fit_lowered(lowered: &Lowered, start: Option<&StartValues>) -> Result<Fit, String> {
    catch_unwind(AssertUnwindSafe(|| {
        fit_warm(
            &lowered.x,
            &lowered.y,
            lowered.n,
            lowered.p,
            &lowered.model,
            &lowered.ids,
            start,
            &lowered.opts,
        )
    }))
    .map_err(panic_message)
}

/// `fit` of `prepared` as a [`FitResult`], with its two per-term tables, and
/// the two likelihood-ratio tables when `anova_test` ([`run_fit`]'s) gives
/// them.
fn flatten(prepared: Prepared, fit: Fit, anova_test: TermTest) -> Result<FitResult, String> {
    let Prepared {
        mut lowered,
        agq_warning,
        reml_warning,
    } = prepared;
    // The LOWERING's observations, decided before any solver ran.
    let lowered_notes = std::mem::take(&mut lowered.notes);

    let (coef_p, df_method) = t_tests(&fit, &lowered.opts.target_indices);
    let satterthwaite_record = match (&fit.satterthwaite, df_method) {
        (Some(st), "satterthwaite") => Some(record_of(st)),
        _ => None,
    };
    let (rss, r_squared, adj_r_squared) =
        if lowered.model.re.is_none() && lowered.model.family == Family::Gaussian {
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

    // The per-term tables, while `lowered` and `fit` are whole: the F or
    // Wald χ² tables of both types, and both likelihood-ratio tables when
    // `anova_test` gives them, which share their refits. A panic in one is
    // stored in its slot, marked as an internal error so a port never shows a
    // bug as a refusal: the fit itself stands.
    let mut cache = RefitCache::default();
    let mut term_table = |ss_type, test| {
        catch_unwind(AssertUnwindSafe(|| {
            let opts = TermTestOptions { ss_type, test };
            term_tests_cached(&lowered, &fit, &opts, &mut cache)
        }))
        .map_err(|payload| {
            format!(
                "internal error while building the per-term table (please report it): {}",
                panic_message(payload)
            )
        })?
        .map_err(|e| e.to_string())
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
        offset: lowered.opts.offset,
        design_summary: fit.design_summary.map(|d| (d.counts, d.s, d.c)),
        nobs: lowered.n,
        boundary: boundary_name(diagnostics.boundary),
        pinned: diagnostics.pinned,
        // The lowering's own observations (unused grouping levels) join the
        // solver's in one channel: the user does not care which layer noticed.
        notes: note_infos(lowered_notes.into_iter().chain(diagnostics.notes).collect()),
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

/// The t test of each target coefficient (`Fit::contrast_t` on its unit
/// vector) as `(coef_p, df_method)`; see [`FitResult::df_method`]. The label
/// keys on the fit's df record, not on its notes: a failed pass is the
/// `NoDfReason::Failed` record.
fn t_tests(fit: &Fit, targets: &[u32]) -> (Vec<f64>, &'static str) {
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
/// `AIC` and `BIC` on two fits: `crate::formula::fixed_part_relation`
/// answers `FixedPartRelation::Same` (the same fixed terms, compared as sets
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

/// The fixed-term labels of `formula` and whether it has an intercept, read
/// from its parse alone, with no data: R's `attr(terms(f), "term.labels")`.
/// Each term's variables are joined by `:` in the order they first appear in
/// the fixed part, as the lowering labels a term (`FixedTerm::label`), and the
/// terms are sorted by degree, in the written order within a degree.
/// Random-effect terms and `offset()` are not among them.
///
/// # Errors
///
/// A formula that does not parse, with the parser's message.
pub fn term_labels(formula: &str) -> Result<(Vec<String>, bool), String> {
    let ast = parse(formula).map_err(|e| e.to_string())?;
    let labels = r_terms(&ast).iter().map(|vars| vars.join(":")).collect();
    Ok((labels, ast.has_intercept))
}

/// A row-major copy of `m`.
fn row_major(m: &Mat<f64>) -> Vec<f64> {
    (0..m.nrows())
        .flat_map(|i| (0..m.ncols()).map(move |j| m[(i, j)]))
        .collect()
}

/// A Satterthwaite pass's state as the record a port stores.
fn record_of(st: &SatterthwaiteState) -> SatterthwaiteRecord {
    SatterthwaiteRecord {
        a_eff: row_major(&st.a_eff),
        jac_theta: st.jac_theta.iter().map(row_major).collect(),
        c: row_major(&st.c),
        tau: st.tau,
        nu: st.nu,
        retained: st.retained.clone(),
    }
}

/// The inverse of [`record_of`] on a fit with `p` coefficients, refusing a
/// record whose sizes do not fit together or with `p`.
fn record_state(r: SatterthwaiteRecord, p: usize) -> Result<SatterthwaiteState, String> {
    let malformed = || "the fit's satterthwaite_record is malformed: refit the model".to_string();
    let k = (r.a_eff.len() as f64).sqrt().round() as usize;
    if k == 0 || k * k != r.a_eff.len() {
        return Err(malformed());
    }
    let pr = r.retained.as_ref().map_or(p, Vec::len);
    let square = |v: &[f64]| v.len() == pr * pr;
    if !square(&r.c) || r.jac_theta.len() != k - 1 || !r.jac_theta.iter().all(|j| square(j)) {
        return Err(malformed());
    }
    if let Some(kept) = &r.retained {
        if kept.windows(2).any(|w| w[0] >= w[1]) || kept.last().is_some_and(|&j| j >= p) {
            return Err(malformed());
        }
    }
    let mat = |v: &[f64], n: usize| Mat::from_fn(n, n, |i, j| v[i * n + j]);
    Ok(SatterthwaiteState::from_record(
        mat(&r.a_eff, k),
        r.jac_theta.iter().map(|j| mat(j, pr)).collect(),
        mat(&r.c, pr),
        r.tau,
        r.nu,
        r.retained,
    ))
}

/// One Wald test of `Lβ = 0` on the fields a port stored of one fit, with no
/// refit: `l` is the `rows × p` matrix `L`, row-major over all `p`
/// coefficients, aliased ones included. The test follows `df_method`
/// ([`FitResult::df_method`]):
///
/// - `"satterthwaite"` (a Gaussian LMM) and `"residual"` (OLS, a Gamma or
///   inverse Gaussian GLM with φ estimated): the F of [`Fit::contrast_f`] on
///   the fit's df record, rebuilt from `record` on the first and, on the
///   second, as the fit builds it: `C` is `vcov` on the columns not
///   `aliased`, `ν = nobs` less those columns;
/// - `"disabled"` and `"failed"` (a Gaussian LMM with no pass): the NaN F row
///   with the no-df note and its reason, as the per-term table gives it;
/// - `"none"`: the Wald χ² of the per-term table's Wald rows.
///
/// The result is a one-row [`TermTable`] labelled `contrast`, with `Sum Sq`
/// and `Mean Sq` from `dispersion` on an F row, and `names`, the fit's
/// coefficient names, as its `coef_names`. A contrast has no SS type, so
/// `ss_type` (Type III) and `choice.reason` (requested) are placeholders,
/// which the ports drop.
///
/// # Errors
///
/// `L` with no row, with a length other than `rows·p` or with a value that is
/// not finite; an unknown `df_method`; a `"satterthwaite"` fit with no
/// `record`, or a malformed one. A panic is returned as `Err`.
#[allow(clippy::too_many_arguments)]
pub fn contrast_test(
    df_method: &str,
    beta: &[f64],
    vcov: &[Vec<f64>],
    aliased: &[bool],
    nobs: usize,
    dispersion: f64,
    record: Option<SatterthwaiteRecord>,
    names: &[String],
    l: &[f64],
    rows: usize,
) -> Result<TermTable, String> {
    let p = beta.len();
    if rows == 0 {
        return Err("L must have at least one row".to_string());
    }
    if l.len() != rows * p {
        return Err(format!(
            "L must have {p} columns, one per coefficient, aliased ones included; got {} \
             values in {rows} row(s)",
            l.len()
        ));
    }
    if let Some(v) = l.iter().find(|v| !v.is_finite()) {
        return Err(format!("L must be finite numbers, got {v}"));
    }
    let test = catch_unwind(AssertUnwindSafe(|| -> Result<LinearTest, String> {
        Ok(match df_method {
            "satterthwaite" => {
                let record = record.ok_or_else(|| {
                    "a fit whose df_method is \"satterthwaite\" needs its satterthwaite_record"
                        .to_string()
                })?;
                let st = record_state(record, p)?;
                LinearTest::from_contrast_f(contrast_f_of(beta, &st, l, rows), DenDf::Satterthwaite)
            }
            "residual" => {
                let kept: Vec<usize> = (0..p).filter(|&j| !aliased[j]).collect();
                let c = Mat::from_fn(kept.len(), kept.len(), |a, b| vcov[kept[a]][kept[b]]);
                let mut st = SatterthwaiteState::residual(c, nobs as f64 - kept.len() as f64);
                if kept.len() < p {
                    st.retained = Some(kept);
                }
                LinearTest::from_contrast_f(contrast_f_of(beta, &st, l, rows), DenDf::Residual)
            }
            "disabled" => untested_test(
                true,
                DenDf::Satterthwaite,
                vec![LinearNote::NoDf {
                    reason: NoDfReason::Disabled,
                }],
            ),
            "failed" => untested_test(
                true,
                DenDf::Satterthwaite,
                vec![LinearNote::NoDf {
                    reason: NoDfReason::Failed,
                }],
            ),
            "none" => wald_chisq(beta, vcov, aliased, l, rows),
            other => return Err(format!("unknown df_method {other:?}")),
        })
    }))
    .map_err(panic_message)??;
    Ok(TermTable {
        ss_type: SsType::Type3,
        choice: TestChoice {
            test: test.kind,
            reason: ChoiceReason::Requested,
        },
        fixed: vec![fixed_row("contrast".to_string(), test, dispersion)],
        notes: Vec::new(),
        coef_names: names.to_vec(),
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
    use crate::formula::anova::{term_tests, ComparisonTest, Side};
    use crate::inference::tails::t_two_sided_p;
    use crate::inference::LrNote;
    use crate::test_support::csv_table;
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
        let err = family_from_str("gaussian", "log").unwrap_err();
        assert!(err.contains("unsupported gaussian link"), "{err}");
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
            // No known dataset reaches `final_eval == true` end to end, so the
            // ports' message branch is pinned from constructed notes.
            Note::PirlsExhausted {
                evals: 3,
                final_eval: false,
            },
            Note::PirlsExhausted {
                evals: 0,
                final_eval: true,
            },
        ]);
        assert_eq!(notes[0].kind, "single_level_grouping_dropped");
        assert_eq!(notes[0].detail, "Subject");
        assert_eq!(notes[1].kind, "hessian_se_fallback");
        assert_eq!(notes[1].detail, "");
        assert_eq!(notes[2].kind, "nb_shape_unsettled");
        assert_eq!(notes[2].evals, 25);
        assert_eq!(notes[3].kind, "exact_profile_fallback");
        assert_eq!(notes[3].detail, "");
        assert_eq!(notes[4].kind, "pirls_exhausted");
        assert_eq!((notes[4].evals, notes[4].final_eval), (3, false));
        assert_eq!(notes[5].kind, "pirls_exhausted");
        assert_eq!((notes[5].evals, notes[5].final_eval), (0, true));
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
    fn unknown_column_is_a_clean_error() {
        let (numeric, factor) = toy_ols();
        let err = run_fit(
            "y ~ z", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
            true, None, "Wald",
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
            "Wald",
        )
        .unwrap_err();
        assert_eq!(
            err,
            "offset given both as an offset() formula term and as the offset= argument; use one"
        );
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
            "Wald",
        )
        .unwrap_err();
        assert_eq!(
            err,
            "weights given both as a cbind() response and as the weights= argument; use one"
        );
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
            "Wald",
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
            "Wald",
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
            None, "Wald",
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
            "Wald",
        )
        .unwrap_err();
        assert!(err.contains("x"), "{err}");
        assert!(err.contains('5') && err.contains('2'), "{err}");

        // A column the formula never references still counts toward the row count.
        let (mut numeric, factor) = toy_ols();
        numeric.insert("junk".to_string(), vec![0.0; 1000]);
        let err = run_fit(
            "y ~ x", numeric, factor, "gaussian", "identity", "hessian", 1, None, None, None, None,
            true, None, "Wald",
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
            "Wald",
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
            "Wald",
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
            "Wald",
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

    /// `run_fit` hands a `(1 | g)` fit's design summary through unchanged, and
    /// none for a fit with no random effect or with prior weights.
    #[test]
    fn run_fit_carries_the_design_summary_inside_the_envelope() {
        let formula = "y ~ x + (1 | g)";
        let r = fit_with(formula, toy_lmm(), "gaussian", "identity", None, true);
        let (numeric, factor) = toy_lmm();
        let mut columns: Vec<(String, Column)> = numeric
            .into_iter()
            .map(|(name, v)| (name, Column::Numeric(v)))
            .collect();
        columns.extend(
            factor
                .into_iter()
                .map(|(name, (levels, codes))| (name, Column::Factor { levels, codes })),
        );
        let table = Table { columns, n: 48 };
        let lo = lower(formula, &table, Family::Gaussian).expect("lowers");
        let fit = fit_warm(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, None, &lo.opts);
        let want = fit.design_summary.expect("a (1 | g) fit keeps it");
        let (counts, s, c) = r.design_summary.expect("run_fit carries it");
        assert_eq!(counts, want.counts);
        assert_eq!(s, want.s);
        assert_eq!(c, want.c);
        assert_eq!((counts.len(), s.len(), c.len()), (8, 2 * 8, 2 * 2));

        let ols = fit_with("y ~ x", toy_lmm(), "gaussian", "identity", None, true);
        assert!(ols.design_summary.is_none());
        let (numeric, factor) = toy_lmm();
        let weighted = run_fit(
            formula,
            numeric,
            factor,
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            Some((0..48).map(|i| 0.75 + 0.1 * (i % 6) as f64).collect()),
            None,
            None,
            true,
            None,
            "Wald",
        )
        .expect("fit should succeed");
        assert!(weighted.design_summary.is_none());
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
        assert!(held.converged);
        assert_eq!(
            held.dispersion, 2.5,
            "dispersion: Some(v) holds phi at v, not estimated"
        );
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
        assert_eq!(r.names, vec!["(Intercept)".to_string(), "x".to_string()]);
        assert!(r.converged);
        assert!(close(r.beta[1], sxy / sxx), "slope {}", r.beta[1]);
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
        // The response and the offset are not read.
        assert_eq!(same_fixed_part("y ~ x + offset(z)", "w ~ x"), Ok(true));
        assert_eq!(same_fixed_part("y ~ x", "y ~ 0 + x"), Ok(false));
        let err = same_fixed_part("y ~ x", "y ~ x - z").unwrap_err();
        assert!(err.contains("\"y ~ x - z\""), "{err}");
        let err = same_fixed_part("y ~ x + ", "y ~ x").unwrap_err();
        assert!(err.contains("ends with '+'"), "{err}");
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
    /// Same herd/period random-intercept model `fit_glmm_cbpp_hessian_rx_agree_on_fitted`
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
            "Wald",
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
            "Wald",
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

    // ── run_compare, the stored per-term tables, the offset ───────────────

    type Columns = (
        HashMap<String, Vec<f64>>,
        HashMap<String, (Vec<String>, Vec<u32>)>,
    );

    /// A committed simulated dataset (`validation/data/simulated/<name>.csv`)
    /// as the column maps `run_fit` takes, `subj`, `g` and `h` as factors.
    fn sim(name: &str) -> Columns {
        let path = format!(
            "{}/validation/data/simulated/{name}.csv",
            env!("CARGO_MANIFEST_DIR")
        );
        let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut numeric = HashMap::new();
        let mut factor = HashMap::new();
        for (col, values) in csv_table(&raw, &["subj", "g", "h"]).columns {
            match values {
                Column::Numeric(v) => {
                    numeric.insert(col, v);
                }
                Column::Factor { levels, codes } => {
                    factor.insert(col, (levels, codes));
                }
            }
        }
        (numeric, factor)
    }

    /// The same columns as a `Table`, for the direct kernel path.
    fn table((numeric, factor): &Columns) -> Table {
        table_of(numeric.clone(), factor.clone()).expect("a table")
    }

    /// `formula` lowered and fitted straight through the kernel, with the
    /// options `run_fit` and `run_compare` set (`wald_se = "hessian"`,
    /// `satterthwaite = true`): the reference the stored results must equal.
    fn direct(
        formula: &str,
        data: &Columns,
        family: Family,
        nagq: u8,
        reml: Option<bool>,
        offset: Option<Vec<f64>>,
    ) -> (Lowered, Fit) {
        let mut lo = lower(formula, &table(data), family).expect("lowers");
        lo.opts.wald_se = WaldSe::Hessian;
        lo.opts.nagq = nagq;
        lo.opts.reml = reml;
        lo.opts.satterthwaite = true;
        if offset.is_some() {
            lo.opts.offset = offset;
        }
        let fit = fit_warm(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, None, &lo.opts);
        (lo, fit)
    }

    fn run(
        formula: &str,
        (numeric, factor): Columns,
        family: &str,
        link: &str,
        nagq: u8,
        offset: Option<Vec<f64>>,
        reml: Option<bool>,
    ) -> FitResult {
        run_fit(
            formula, numeric, factor, family, link, "hessian", nagq, None, None, offset, None,
            true, reml, "Wald",
        )
        .unwrap_or_else(|e| panic!("{formula}: {e}"))
    }

    /// Every `FitResult` field read off the kernel's `Fit`, as one string
    /// (`{:?}` of an `f64` round-trips, so equal strings are equal bits).
    fn kernel_fields(r: &FitResult) -> String {
        format!(
            "{:?}",
            (
                (&r.beta, &r.se, &r.vcov, &r.tau2, &r.varcorr, &r.stddev_se),
                (&r.aliased, r.dispersion, r.converged, r.n_eval, r.deviance),
                (r.loglik, r.df, r.reml, &r.fitted, &r.ranef, &r.ranef_levels),
                (&r.coef_df, &r.design_summary, r.nobs),
            )
        )
    }

    fn fit_fields(lo: &Lowered, fit: &Fit) -> String {
        let ds = fit
            .design_summary
            .as_ref()
            .map(|d| (d.counts.clone(), d.s.clone(), d.c.clone()));
        format!(
            "{:?}",
            (
                (
                    &fit.beta,
                    &fit.se,
                    &fit.vcov,
                    &fit.tau2,
                    &fit.varcorr,
                    &fit.stddev_se
                ),
                (
                    fit.aliased(),
                    fit.dispersion,
                    fit.converged(),
                    fit.n_eval,
                    fit.deviance
                ),
                (
                    fit.loglik,
                    fit.df,
                    fit.reml,
                    &fit.fitted,
                    &fit.ranef,
                    &fit.ranef_levels
                ),
                (&fit.coef_df, &ds, lo.n),
            )
        )
    }

    fn bits(v: &Option<Vec<f64>>) -> Option<Vec<u64>> {
        v.as_ref().map(|v| v.iter().map(|x| x.to_bits()).collect())
    }

    /// `run_fit` hands on the kernel's fit bit for bit, with its objective
    /// (an AGQ fit's node count included) and the offset it fitted with,
    /// from `offset=` or from `offset()` in the formula.
    #[test]
    fn run_fit_keeps_the_kernel_fit_bitwise_with_its_objective_and_offset() {
        let (mut numeric, factor) = toy_ols();
        let off: Vec<f64> = (0..10).map(|i| 0.1 * i as f64).collect();
        numeric.insert("o".to_string(), off.clone());
        let ols = (numeric, factor);
        let gaussian = Family::Gaussian;
        let binomial = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let cbpp = "cbind(incidence, healthy) ~ period + (1 | herd)";
        let lmm = "y ~ g * h + x + (1 + x | subj)";
        let cases = [
            (
                lmm,
                sim("sim_factorial"),
                "gaussian",
                gaussian,
                1,
                None,
                Objective::Reml,
            ),
            (
                cbpp,
                cbpp_columns(),
                "binomial",
                binomial,
                5,
                None,
                Objective::Agq { nodes: 5 },
            ),
            (
                "y ~ x",
                ols.clone(),
                "gaussian",
                gaussian,
                1,
                Some(off.clone()),
                Objective::LeastSquares,
            ),
            (
                "y ~ x + offset(o)",
                ols,
                "gaussian",
                gaussian,
                1,
                None,
                Objective::LeastSquares,
            ),
        ];
        for (formula, data, family, fam, nagq, offset, objective) in cases {
            let link = if family == "binomial" {
                "logit"
            } else {
                "identity"
            };
            let want = (offset.is_some() || formula.contains("offset(")).then(|| off.clone());
            let r = run(
                formula,
                data.clone(),
                family,
                link,
                nagq,
                offset.clone(),
                None,
            );
            let (lo, fit) = direct(formula, &data, fam, nagq, None, offset);
            assert_eq!(kernel_fields(&r), fit_fields(&lo, &fit), "{formula}");
            assert_eq!(r.objective, fit.objective, "{formula}");
            assert_eq!(r.objective, objective, "{formula}");
            assert_eq!(bits(&r.offset), bits(&lo.opts.offset), "{formula}");
            assert_eq!(bits(&r.offset), bits(&want), "{formula}");
        }
    }

    fn table_string(t: &Result<TermTable, String>) -> String {
        format!("{t:?}")
    }

    /// The stored tables are `term_tests` on the fit's own `Lowered` and
    /// `Fit`, both types, bit for bit: an LMM, its OLS twin, a GLMM; the
    /// refusal of a formula without its marginal term is stored for both
    /// types; an empty cell puts the missing-cell note on the Type III table.
    #[test]
    fn run_fit_stores_both_per_term_tables_as_term_tests_gives_them() {
        let gaussian = Family::Gaussian;
        let binomial = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let cases = [
            ("y ~ g * h + x + (1 + x | subj)", "sim_factorial", gaussian),
            ("y ~ g * h + x", "sim_factorial", gaussian),
            ("y ~ g * h + x + (1 | subj)", "sim_factorial_bern", binomial),
            ("y ~ x + x:g", "sim_factorial", gaussian),
            (
                "y ~ g * h + x + (1 + x | subj)",
                "sim_factorial_empty",
                gaussian,
            ),
        ];
        for (formula, data, fam) in cases {
            let (family, link) = if fam == gaussian {
                ("gaussian", "identity")
            } else {
                ("binomial", "logit")
            };
            let r = run(formula, sim(data), family, link, 1, None, None);
            let (lo, fit) = direct(formula, &sim(data), fam, 1, None, None);
            for (stored, ss_type) in [
                (&r.term_table_type3, SsType::Type3),
                (&r.term_table_type2, SsType::Type2),
            ] {
                let opts = TermTestOptions {
                    ss_type,
                    test: TermTest::Auto,
                };
                let want = term_tests(&lo, &fit, &opts).map_err(|e| e.to_string());
                assert_eq!(
                    table_string(stored),
                    table_string(&want),
                    "{formula} on {data}"
                );
            }
        }
        // What the cases above hold: a refusal for both types, and the note.
        let nonmarginal = run(
            "y ~ x + x:g",
            sim("sim_factorial"),
            "gaussian",
            "identity",
            1,
            None,
            None,
        );
        assert!(nonmarginal.term_table_type3.is_err() && nonmarginal.term_table_type2.is_err());
        let empty = run(
            "y ~ g * h + x + (1 + x | subj)",
            sim("sim_factorial_empty"),
            "gaussian",
            "identity",
            1,
            None,
            None,
        );
        let t3 = empty.term_table_type3.as_ref().expect("a table");
        assert!(
            t3.notes.iter().any(|n| n.contains("Missing cells")),
            "{:?}",
            t3.notes
        );
        let lmm = run(
            "y ~ g * h + x + (1 + x | subj)",
            sim("sim_factorial"),
            "gaussian",
            "identity",
            1,
            None,
            None,
        );
        let cols = term_columns(lmm.term_table_type3.as_ref().expect("a table"));
        assert_eq!(cols.label, ["g", "h", "x", "g:h"]);
        assert!(cols.den_df_kind.iter().all(|k| *k == "satterthwaite"));
        assert_eq!(choice_tags(&lmm.term_test_auto), ("F", "f_test"));
    }

    /// `run_fit` as [`run`] makes it at `nagq = 1`, with `anova_test` given.
    fn run_asking(
        formula: &str,
        (numeric, factor): Columns,
        family: &str,
        link: &str,
        anova_test: &str,
    ) -> FitResult {
        run_fit(
            formula, numeric, factor, family, link, "hessian", 1, None, None, None, None, true,
            None, anova_test,
        )
        .unwrap_or_else(|e| panic!("{formula}: {e}"))
    }

    /// With `anova_test = "LRT"` the stored likelihood-ratio tables are
    /// `term_tests` under `TermTest::Lrt` on the fit's own `Lowered` and
    /// `Fit`, both types, bit for bit, although the two tables share their
    /// refits: a GLMM (two of its four Type III refits are Type II designs),
    /// its GLM twin, and a Gaussian LMM, whose two slots hold the refusal of
    /// a fit with F rows. `term_columns` names the table's test and reason.
    #[test]
    fn run_fit_stores_the_likelihood_ratio_tables_as_term_tests_gives_them() {
        let binomial = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let glmm = "y ~ g * h + x + (1 | subj)";
        let cases = [
            (glmm, "sim_factorial_bern", binomial),
            ("y ~ g * h + x", "sim_factorial_bern", binomial),
            (
                "y ~ g * h + x + (1 + x | subj)",
                "sim_factorial",
                Family::Gaussian,
            ),
        ];
        for (formula, data, fam) in cases {
            let (family, link) = if fam == Family::Gaussian {
                ("gaussian", "identity")
            } else {
                ("binomial", "logit")
            };
            let r = run_asking(formula, sim(data), family, link, "LRT");
            let (lo, fit) = direct(formula, &sim(data), fam, 1, None, None);
            for (stored, ss_type) in [
                (&r.term_table_type3_lrt, SsType::Type3),
                (&r.term_table_type2_lrt, SsType::Type2),
            ] {
                let stored = stored.as_ref().expect("asked for");
                assert_eq!(stored.is_ok(), fam != Family::Gaussian, "{formula}");
                let opts = TermTestOptions {
                    ss_type,
                    test: TermTest::Lrt,
                };
                let want = term_tests(&lo, &fit, &opts).map_err(|e| e.to_string());
                assert_eq!(
                    table_string(stored),
                    table_string(&want),
                    "{formula} on {data}"
                );
            }
        }
        let r = run_asking(glmm, sim("sim_factorial_bern"), "binomial", "logit", "LRT");
        let t3 = r.term_table_type3_lrt.as_ref().expect("asked for");
        let cols = term_columns(t3.as_ref().expect("a table"));
        assert_eq!((cols.test, cols.reason), ("lrt", "requested"));
        assert!(cols.kind.iter().all(|k| *k == "lrt"), "{:?}", cols.kind);
    }

    /// A fit other than a Poisson GLMM (here a binomial GLMM and GLM, whose
    /// `"auto"` is the Wald χ²) stores no likelihood-ratio table unless it
    /// asks with `anova_test = "LRT"`, and asking moves nothing else: the fit
    /// and its two Wald tables are the same bits under `"auto"`, `"Wald"` and
    /// `"LRT"`. Any other value is refused.
    #[test]
    fn no_fit_stores_a_likelihood_ratio_table_unless_asked() {
        let glm = run_asking(
            "y ~ g * h + x",
            sim("sim_factorial_bern"),
            "binomial",
            "logit",
            "auto",
        );
        assert_eq!(
            choice_tags(&glm.term_test_auto),
            ("chisq", "no_random_effects")
        );
        assert!(glm.term_table_type3_lrt.is_none() && glm.term_table_type2_lrt.is_none());
        let formula = "y ~ g * h + x + (1 | subj)";
        let data = || sim("sim_factorial_bern");
        let asked = run_asking(formula, data(), "binomial", "logit", "LRT");
        assert!(asked.term_table_type3_lrt.is_some() && asked.term_table_type2_lrt.is_some());
        assert_eq!(
            choice_tags(&asked.term_test_auto),
            ("chisq", "binomial_mixed")
        );
        for anova_test in ["auto", "Wald"] {
            let r = run_asking(formula, data(), "binomial", "logit", anova_test);
            assert_eq!(
                choice_tags(&r.term_test_auto),
                ("chisq", "binomial_mixed"),
                "{anova_test}"
            );
            assert!(
                r.term_table_type3_lrt.is_none() && r.term_table_type2_lrt.is_none(),
                "{anova_test}"
            );
            assert_eq!(kernel_fields(&r), kernel_fields(&asked), "{anova_test}");
            for (t, u) in [
                (&r.term_table_type3, &asked.term_table_type3),
                (&r.term_table_type2, &asked.term_table_type2),
            ] {
                assert_eq!(table_string(t), table_string(u), "{anova_test}");
            }
        }
        let (numeric, factor) = data();
        let err = run_fit(
            formula, numeric, factor, "binomial", "logit", "hessian", 1, None, None, None, None,
            true, None, "lrt",
        )
        .unwrap_err();
        assert_eq!(
            err,
            r#"anova_test must be "auto", "Wald" or "LRT", got "lrt""#
        );
    }

    /// A Poisson GLMM's `"auto"` is the likelihood-ratio χ², so the fit
    /// stores the likelihood-ratio tables under `"auto"` (built under
    /// `TermTest::Auto`) as under `"LRT"`, with the same rows; `"Wald"`
    /// stores none. Its two other slots hold the Wald tables under every
    /// value, and the fit is the same bits under all three.
    #[test]
    fn auto_stores_the_likelihood_ratio_tables_of_a_poisson_glmm() {
        let formula = "y ~ x + (1 | g)";
        let poisson = Family::Poisson {
            link: PoissonLink::Log,
        };
        let (lo, fit) = direct(formula, &sim("glmm_poisson"), poisson, 1, None, None);
        let want = |ss_type, test| {
            let opts = TermTestOptions { ss_type, test };
            table_string(&term_tests(&lo, &fit, &opts).map_err(|e| e.to_string()))
        };
        let values = ["auto", "Wald", "LRT"];
        let fits = values.map(|t| run_asking(formula, sim("glmm_poisson"), "poisson", "log", t));
        for (r, anova_test) in fits.iter().zip(values) {
            assert_eq!(
                choice_tags(&r.term_test_auto),
                ("lrt", "poisson_mixed"),
                "{anova_test}"
            );
            assert_eq!(kernel_fields(r), kernel_fields(&fits[0]), "{anova_test}");
            for (wald, lrt, ss_type) in [
                (&r.term_table_type3, &r.term_table_type3_lrt, SsType::Type3),
                (&r.term_table_type2, &r.term_table_type2_lrt, SsType::Type2),
            ] {
                assert_eq!(
                    table_string(wald),
                    want(ss_type, TermTest::Wald),
                    "{anova_test}"
                );
                let asked = match anova_test {
                    "auto" => TermTest::Auto,
                    "LRT" => TermTest::Lrt,
                    _ => {
                        assert!(lrt.is_none(), "{anova_test}");
                        continue;
                    }
                };
                let lrt = lrt.as_ref().expect("stored");
                assert_eq!(table_string(lrt), want(ss_type, asked), "{anova_test}");
            }
        }
        let cols = |r: &FitResult| {
            term_columns(r.term_table_type3_lrt.as_ref().unwrap().as_ref().unwrap())
        };
        let (auto, asked) = (cols(&fits[0]), cols(&fits[2]));
        assert_eq!((auto.test, auto.reason), ("lrt", "poisson_mixed"));
        assert_eq!((asked.test, asked.reason), ("lrt", "requested"));
        let rows =
            |c: &TermColumns| format!("{:?}", (&c.label, &c.statistic, &c.p_value, &c.notes));
        assert_eq!(rows(&auto), rows(&asked));
    }

    /// `compare()`'s two fits are made as `"Wald"` makes them: on a Poisson
    /// GLMM pair, whose `"auto"` is still the likelihood-ratio test, neither
    /// stores a likelihood-ratio table.
    #[test]
    fn run_compare_makes_its_fits_as_wald() {
        let (numeric, factor) = sim("glmm_poisson");
        let r = run_compare(
            "y ~ 1 + (1 | g)",
            "y ~ x + (1 | g)",
            numeric,
            factor,
            "poisson",
            "log",
            1,
            None,
            None,
            None,
            None,
        )
        .expect("compares");
        for f in [&r.fit0, &r.fit1] {
            assert_eq!(choice_tags(&f.term_test_auto), ("lrt", "poisson_mixed"));
            assert!(f.term_table_type3_lrt.is_none() && f.term_table_type2_lrt.is_none());
        }
    }

    /// The four estimator cases of `run_compare`: each pair fitted the way
    /// its case needs, and the comparison equal, bit for bit, to the two-fit
    /// `compare` on the same two fits made directly.
    #[test]
    fn run_compare_picks_the_estimator_from_what_differs() {
        let gaussian = Family::Gaussian;
        let binomial = Family::Binomial {
            link: BinomialLink::Logit,
        };
        // (f0, f1, data, family, the fits' `reml` argument, `Fit::reml`)
        let cases = [
            (
                "y ~ g * h + x",
                "y ~ g * h + x + (1 | subj)",
                "sim_factorial",
                gaussian,
                Some(true),
                true,
            ),
            (
                "y ~ g * h + x + (1 | subj)",
                "y ~ g + h + x + (1 | subj)",
                "sim_factorial",
                gaussian,
                Some(false),
                false,
            ),
            (
                "y ~ g + h + x + (1 | subj)",
                "y ~ g * h + x + (1 + x | subj)",
                "sim_factorial",
                gaussian,
                Some(false),
                false,
            ),
            (
                "y ~ g * h + x + (1 | subj)",
                "y ~ g * h + x",
                "sim_factorial_bern",
                binomial,
                None,
                false,
            ),
        ];
        for (f0, f1, data, fam, reml, fit_reml) in cases {
            let (family, link) = if fam == gaussian {
                ("gaussian", "identity")
            } else {
                ("binomial", "logit")
            };
            let (numeric, factor) = sim(data);
            let got = run_compare(
                f0, f1, numeric, factor, family, link, 1, None, None, None, None,
            )
            .unwrap_or_else(|e| panic!("{f0} vs {f1}: {e}"));
            for fit in [&got.fit0, &got.fit1] {
                assert_eq!(fit.reml, fit_reml, "{f0} vs {f1}");
                assert!(fit.reml_warning.is_none());
                // satterthwaite = true: a Gaussian mixed side's table has its F rows.
                if fam == gaussian && !fit.varcorr.is_empty() {
                    let cols = term_columns(fit.term_table_type3.as_ref().expect("a table"));
                    assert!(cols.den_df_kind.iter().all(|k| *k == "satterthwaite"));
                }
            }
            let (lo0, fit0) = direct(f0, &sim(data), fam, 1, reml, None);
            let (lo1, fit1) = direct(f1, &sim(data), fam, 1, reml, None);
            let want = compare(
                &Model::from_fit(f0, &lo0, &fit0),
                &Model::from_fit(f1, &lo1, &fit1),
                &compare_options(None, None).unwrap(),
            )
            .expect("comparable");
            assert_eq!(
                format!("{:?}", got.comparison),
                format!("{want:?}"),
                "{f0} vs {f1}"
            );
        }
    }

    /// The formulas swapped give the same test; the fits follow the
    /// formulas as given.
    #[test]
    fn run_compare_takes_the_formulas_in_either_order() {
        let (f0, f1) = (
            "y ~ g + h + x + (1 | subj)",
            "y ~ g * h + x + (1 + x | subj)",
        );
        let go = |a: &str, b: &str| {
            let (numeric, factor) = sim("sim_factorial");
            run_compare(
                a, b, numeric, factor, "gaussian", "identity", 1, None, None, None, None,
            )
            .expect("compares")
        };
        let (ab, ba) = (go(f0, f1), go(f1, f0));
        assert_eq!(
            format!("{:?}", ab.comparison.test),
            format!("{:?}", ba.comparison.test)
        );
        assert_eq!(ab.comparison.larger, Side::Second);
        assert_eq!(ba.comparison.larger, Side::First);
        assert_eq!(ba.comparison.m0.formula_label, f1);
        assert_eq!(kernel_fields(&ab.fit0), kernel_fields(&ba.fit1));
        assert_eq!(kernel_fields(&ab.fit1), kernel_fields(&ba.fit0));
    }

    /// Formulas that are not nested refuse before either fit (an unknown
    /// column would be the fit's error); a fit that does not converge
    /// refuses the comparison.
    #[test]
    fn run_compare_refuses_unnested_formulas_and_an_unconverged_fit() {
        let (numeric, factor) = sim("sim_factorial");
        let (f0, f1) = ("y ~ g + nosuch", "y ~ h");
        let err = run_compare(
            f0, f1, numeric, factor, "gaussian", "identity", 1, None, None, None, None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            CompareError::NotNested {
                m0: f0.to_string(),
                m1: f1.to_string()
            }
            .to_string()
        );

        let (mut numeric, factor) = toy_lmm();
        let x2: Vec<f64> = numeric["x"].iter().map(|v| 2.0 * v).collect();
        numeric.insert("x2".to_string(), x2);
        let err = run_compare(
            "y ~ x + x2 + (1 | g)",
            "y ~ x + x2 + (1 + x2 | g)",
            numeric,
            factor,
            "gaussian",
            "identity",
            1,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(
            err,
            CompareError::NotConverged {
                m0: false,
                m1: true
            }
            .to_string()
        );
    }

    /// A `cbind()` response sets the weights without tripping the
    /// double specification, and `nagq` is stripped per model as `run_fit`
    /// strips it: `Agq{5}` against `Laplace`, with the node-count note.
    #[test]
    fn run_compare_strips_nagq_per_model_and_reads_a_cbind_response() {
        let (numeric, mut factor) = cbpp_columns();
        let blk: Vec<String> = factor["herd"]
            .1
            .iter()
            .map(|c| format!("b{}", c % 3))
            .collect();
        let blk: Vec<&str> = blk.iter().map(String::as_str).collect();
        factor.insert("blk".to_string(), factor_col(&blk));
        let f0 = "cbind(incidence, healthy) ~ period + (1 | herd)";
        let f1 = "cbind(incidence, healthy) ~ period + (1 | herd) + (1 | blk)";
        let got = run_compare(
            f0, f1, numeric, factor, "binomial", "logit", 5, None, None, None, None,
        )
        .expect("compares");
        assert_eq!(got.fit0.objective, Objective::Agq { nodes: 5 });
        assert!(got.fit0.agq_warning.is_none());
        assert_eq!(got.fit1.objective, Objective::Laplace);
        assert!(got.fit1.agq_warning.as_deref().unwrap().contains("nagq=5"));
        assert!(got.fit0.weights.is_some());
        let ComparisonTest::Lr(t) = &got.comparison.test else {
            panic!("an LR test")
        };
        assert!(
            t.notes.contains(&LrNote::NodeCounts { m0: 5, m1: 1 }),
            "{:?}",
            t.notes
        );
    }

    /// `seed = None` takes `DEFAULT_SEED`: the same p-value as passing it.
    /// The result carries `nsim` and the seed; `nsim` is bounded.
    #[test]
    fn run_compare_defaults_the_seed_to_the_constant() {
        let (f0, f1) = ("y ~ g * h + x", "y ~ g * h + x + (1 | subj)");
        let go = |seed: Option<u64>| {
            let (numeric, factor) = sim("sim_factorial");
            run_compare(
                f0,
                f1,
                numeric,
                factor,
                "gaussian",
                "identity",
                1,
                None,
                None,
                Some(999),
                seed,
            )
            .expect("compares")
        };
        let p = |r: &CompareResult| match &r.comparison.test {
            ComparisonTest::Lr(t) => (t.p_value, t.reference),
            other => panic!("{other:?}"),
        };
        let (default, given) = (go(None), go(Some(DEFAULT_SEED)));
        assert_eq!(p(&default).0.to_bits(), p(&given).0.to_bits());
        assert_eq!((default.nsim, default.seed), (999, DEFAULT_SEED));
        assert_eq!(
            p(&default).1,
            Reference::ExactRlrt {
                nsim: 999,
                seed: DEFAULT_SEED
            }
        );
        for bad in [0, MAX_NSIM + 1] {
            let (numeric, factor) = sim("sim_factorial");
            let err = run_compare(
                f0,
                f1,
                numeric,
                factor,
                "gaussian",
                "identity",
                1,
                None,
                None,
                Some(bad),
                None,
            )
            .unwrap_err();
            assert!(err.starts_with("nsim must be"), "{err}");
        }
    }

    /// `x` and `y` are the same `f64`, a NaN equal to any NaN.
    fn same(x: f64, y: f64) -> bool {
        x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
    }

    /// `term_columns` puts each `FixedRow` field in its column, row by row,
    /// on an F table with Satterthwaite df, an F table with the residual df,
    /// a Wald χ² table, and a table with row notes and a table note.
    #[test]
    fn term_columns_flattens_each_row_into_its_columns() {
        let gaussian = Family::Gaussian;
        let binomial = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let lmm = "y ~ g * h + x + (1 + x | subj)";
        // `x2` is `x` doubled: its column is aliased, its row untested.
        let (mut numeric, factor) = sim("sim_factorial");
        let x2: Vec<f64> = numeric["x"].iter().map(|v| 2.0 * v).collect();
        numeric.insert("x2".to_string(), x2);
        let aliased = (numeric, factor);
        // (formula, data, family, ss type, each row's kind and den_df_kind)
        let cases = [
            (
                lmm,
                "sim_factorial",
                gaussian,
                SsType::Type3,
                "F",
                "satterthwaite",
            ),
            (
                "y ~ g * h + x",
                "sim_factorial",
                gaussian,
                SsType::Type2,
                "F",
                "residual",
            ),
            (
                "y ~ g * h + x + (1 | subj)",
                "sim_factorial_bern",
                binomial,
                SsType::Type3,
                "chisq",
                "",
            ),
            (
                lmm,
                "sim_factorial_empty",
                gaussian,
                SsType::Type3,
                "F",
                "satterthwaite",
            ),
            (
                "y ~ g * h + x + x2",
                "",
                gaussian,
                SsType::Type3,
                "F",
                "residual",
            ),
        ];
        let (mut multi_df_f_rows, mut row_notes, mut table_notes) = (0, 0, 0);
        for (formula, data, fam, ss_type, kind, den_df_kind) in cases {
            let columns = if data.is_empty() {
                aliased.clone()
            } else {
                sim(data)
            };
            let (lo, fit) = direct(formula, &columns, fam, 1, None, None);
            let opts = TermTestOptions {
                ss_type,
                test: TermTest::Auto,
            };
            let t = term_tests(&lo, &fit, &opts).expect("a table");
            let c = term_columns(&t);
            let reason = if kind == "F" {
                "f_test"
            } else {
                "binomial_mixed"
            };
            assert_eq!((c.test, c.reason), (kind, reason), "{formula} on {data}");
            let n = t.fixed.len();
            assert!(n > 0, "{formula} on {data}");
            let ss = if ss_type == SsType::Type3 {
                "III"
            } else {
                "II"
            };
            assert_eq!(c.ss_type, ss);
            let lens = [
                c.label.len(),
                c.kind.len(),
                c.statistic.len(),
                c.num_df.len(),
                c.den_df.len(),
                c.den_df_kind.len(),
                c.p_value.len(),
                c.sum_sq.len(),
                c.mean_sq.len(),
                c.row_notes.len(),
            ];
            assert_eq!(lens, [n; 10], "{formula} on {data}");
            for (i, row) in t.fixed.iter().enumerate() {
                let at = format!("{formula} on {data}, row {i}");
                assert_eq!(c.label[i], row.label, "{at}");
                assert_eq!(c.kind[i], kind, "{at}");
                assert_eq!(c.den_df_kind[i], den_df_kind, "{at}");
                assert!(same(c.statistic[i], row.test.statistic), "{at}");
                assert!(same(c.num_df[i], row.test.num_df), "{at}");
                assert!(same(c.p_value[i], row.test.p_value), "{at}");
                match row.test.reference {
                    Reference::F { den_df, .. } => {
                        assert_eq!(kind, "F", "{at}");
                        assert!(same(c.den_df[i], den_df), "{at}");
                    }
                    _ => {
                        assert_eq!(kind, "chisq", "{at}");
                        assert!(c.den_df[i].is_nan(), "{at}");
                    }
                }
                match (row.sum_sq, row.mean_sq) {
                    (Some(ss), Some(ms)) => {
                        assert!(same(c.sum_sq[i], ss) && same(c.mean_sq[i], ms), "{at}");
                        multi_df_f_rows += usize::from(row.test.num_df > 1.0 && ss != ms);
                    }
                    (None, None) => {
                        assert!(c.sum_sq[i].is_nan() && c.mean_sq[i].is_nan(), "{at}");
                    }
                    other => panic!("{at}: {other:?}"),
                }
                let lines: Vec<String> = row
                    .test
                    .notes
                    .iter()
                    .map(|note| linear_note_line(*note, &t.coef_names))
                    .collect();
                assert_eq!(c.row_notes[i], lines, "{at}");
                row_notes += lines.len();
            }
            assert_eq!(c.notes, t.notes, "{formula} on {data}");
            table_notes += t.notes.len();
        }
        // What the cases hold: an F row on more than one df (its Sum Sq is
        // not its Mean Sq), a row note and a table note.
        assert!(
            multi_df_f_rows > 0 && row_notes > 0 && table_notes > 0,
            "{multi_df_f_rows} multi-df F rows, {row_notes} row notes, {table_notes} table notes"
        );
    }

    /// `objective_parts` on every objective, and `objective_from_parts`
    /// back; a pair `objective_parts` never gives is refused.
    #[test]
    fn objective_parts_round_trip_on_every_objective() {
        let cases = [
            (Objective::LeastSquares, ("least_squares", 0)),
            (Objective::MaximumLikelihood, ("maximum_likelihood", 0)),
            (Objective::Reml, ("reml", 0)),
            (Objective::Laplace, ("laplace", 1)),
            (Objective::Agq { nodes: 5 }, ("agq", 5)),
        ];
        for (objective, (label, nodes)) in cases {
            assert_eq!(objective_parts(objective), (label, nodes));
            assert_eq!(objective_from_parts(label, nodes), Ok(objective));
        }
        assert_eq!(
            objective_from_parts("reml", 1),
            Err("no fit stores the objective \"reml\" with 1 quadrature node".to_string())
        );
        for (label, nodes) in [
            ("least_squares", 1),
            ("reml", 1),
            ("laplace", 0),
            ("laplace", 2),
            ("agq", 0),
            ("agq", 1),
            ("ml", 0),
        ] {
            assert!(
                objective_from_parts(label, nodes).is_err(),
                "{label}, {nodes}"
            );
        }
    }

    /// `nsim` from 1 to `MAX_NSIM` is taken; 0 and `MAX_NSIM + 1` are refused.
    #[test]
    fn compare_options_takes_nsim_from_1_to_max_nsim() {
        assert_eq!(
            compare_options(None, None),
            Ok(CompareOptions {
                nsim: DEFAULT_NSIM,
                seed: DEFAULT_SEED
            })
        );
        for nsim in [1, MAX_NSIM] {
            assert_eq!(
                compare_options(Some(nsim), Some(7)),
                Ok(CompareOptions { nsim, seed: 7 })
            );
        }
        for nsim in [0, MAX_NSIM + 1] {
            assert_eq!(
                compare_options(Some(nsim), None),
                Err(format!(
                    "nsim must be a whole number from 1 to {MAX_NSIM}, got {nsim}"
                ))
            );
        }
    }

    /// Two formulas with the same fixed and random parts are refused before
    /// either fit (an unknown column would be the fit's error).
    #[test]
    fn run_compare_refuses_equal_parts_before_any_fit() {
        let (numeric, factor) = sim("sim_factorial");
        let err = run_compare(
            "y ~ g + nosuch + (1 | subj)",
            "y ~ nosuch + g + (1 | subj)",
            numeric,
            factor,
            "gaussian",
            "identity",
            1,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err, CompareError::SameParts.to_string());
    }

    /// Two formulas that differ only by an `offset()` term have the same
    /// parts, but the refusal names the offsets, as `compare` does.
    #[test]
    fn run_compare_names_the_offset_when_only_the_offset_term_differs() {
        let (numeric, factor) = sim("sim_factorial");
        let err = run_compare(
            "y ~ g + offset(x) + (1 | subj)",
            "y ~ g + (1 | subj)",
            numeric,
            factor,
            "gaussian",
            "identity",
            1,
            None,
            None,
            None,
            None,
        )
        .unwrap_err();
        assert_eq!(err, CompareError::DifferentOffset.to_string());
    }

    /// A panic while a per-term table is built, a likelihood-ratio table
    /// included, is stored as that table's internal error, never as a
    /// refusal, and the fit stands.
    #[test]
    fn a_panic_in_the_term_tables_is_stored_as_an_internal_error() {
        let data = table(&sim("sim_factorial_bern"));
        let prepared = lower_with(
            "y ~ g + x",
            &data,
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            "hessian",
            1,
            None,
            None,
            None,
            true,
            None,
        )
        .expect("lowers");
        let mut fit = fit_lowered(&prepared.lowered, None).expect("fits");
        // `term_tests` reads `aliased` at each column, past its end now.
        fit.diagnostics.aliased.clear();
        let r = flatten(prepared, fit, TermTest::Lrt).expect("the fit stands");
        let lrt = [&r.term_table_type3_lrt, &r.term_table_type2_lrt]
            .map(|t| t.as_ref().expect("asked for"));
        for t in [&r.term_table_type3, &r.term_table_type2]
            .into_iter()
            .chain(lrt)
        {
            let err = t.as_ref().unwrap_err();
            assert!(
                err.starts_with(
                    "internal error while building the per-term table (please report it): "
                ),
                "{err}"
            );
        }
    }

    #[test]
    fn term_labels_expand_groups_and_drop_random_terms_and_the_intercept() {
        let cases: [(&str, &[&str], bool); 7] = [
            (
                "y ~ age + sex*_ssd_text + (1 | id)",
                &["age", "sex", "_ssd_text", "sex:_ssd_text"],
                true,
            ),
            ("y ~ a*(b + c)", &["a", "b", "c", "a:b", "a:c"], true),
            ("y ~ 0 + a + b", &["a", "b"], false),
            ("y ~ a + b - 1", &["a", "b"], false),
            ("y ~ x + offset(o) + (1 + x | g) + (1 | h)", &["x"], true),
            // Variables are ordered by first appearance, whatever the
            // interaction's own order; the response is optional.
            ("~ a + b + b:a", &["a", "b", "a:b"], true),
            (
                "y ~ c:b:a + x + a*b",
                &["x", "a", "b", "b:a", "c:b:a"],
                true,
            ),
        ];
        for (formula, labels, intercept) in cases {
            let (got, has_intercept) = term_labels(formula).expect(formula);
            assert_eq!(got, labels, "{formula}");
            assert_eq!(has_intercept, intercept, "{formula}");
        }
        assert!(term_labels("y ~ x + (1 | g").is_err());
    }

    /// `contrast_test` on the fields `run_fit` stores gives the kernel's own
    /// test of the same `L` bit for bit: `Fit::contrast_f` on a Gaussian LMM,
    /// on OLS (rank-deficient design included) and on a Gamma GLM,
    /// `wald_chisq` on a binomial GLMM. The stored record carries everything the test reads.
    #[test]
    fn contrast_test_on_the_stored_fields_is_the_kernel_test_bitwise() {
        let binomial = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let (mut numeric, factor) = sim("sim_factorial");
        let x2: Vec<f64> = numeric["x"].iter().map(|v| 2.0 * v).collect();
        numeric.insert("x2".to_string(), x2);
        let with_x2 = (numeric, factor);
        let cases = [
            (
                "y ~ g + h + x + (1 + x | subj)",
                sim("sim_factorial"),
                "gaussian",
                "identity",
                Family::Gaussian,
                "satterthwaite",
            ),
            (
                "y ~ g + h + x",
                sim("sim_factorial"),
                "gaussian",
                "identity",
                Family::Gaussian,
                "residual",
            ),
            (
                "y ~ g + x + x2 + h",
                with_x2,
                "gaussian",
                "identity",
                Family::Gaussian,
                "residual",
            ),
            (
                "y ~ x + grp",
                sim("glm_gamma"),
                "gamma",
                "log",
                Family::Gamma {
                    link: GammaLink::Log,
                },
                "residual",
            ),
            (
                "y ~ g + h + x + (1 | subj)",
                sim("sim_factorial_bern"),
                "binomial",
                "logit",
                binomial,
                "none",
            ),
        ];
        let mut tested = 0;
        for (formula, data, family, link, fam, df_method) in cases {
            let r = run(formula, data.clone(), family, link, 1, None, None);
            let (_, fit) = direct(formula, &data, fam, 1, None, None);
            assert_eq!(r.df_method, df_method, "{formula}");
            assert_eq!(
                r.satterthwaite_record.is_some(),
                df_method == "satterthwaite"
            );
            let p = r.beta.len();
            let unit = |j: usize| {
                (0..p)
                    .map(|k| f64::from(u8::from(k == j)))
                    .collect::<Vec<_>>()
            };
            let mixed: Vec<f64> = (0..p).map(|k| 0.5 * k as f64 - 1.0).collect();
            // Two rows, the same two plus their sum (rank 2 of 3), one row
            // touching every column (the aliased `x2` on the third case).
            let ls: Vec<Vec<f64>> = vec![
                [unit(1), unit(2)].concat(),
                [
                    unit(1),
                    mixed.clone(),
                    unit(1).iter().zip(&mixed).map(|(a, b)| a + b).collect(),
                ]
                .concat(),
                mixed.clone(),
            ];
            for l in ls {
                let rows = l.len() / p;
                let table = contrast_test(
                    r.df_method,
                    &r.beta,
                    &r.vcov,
                    &r.aliased,
                    r.nobs,
                    r.dispersion,
                    r.satterthwaite_record.clone(),
                    &r.names,
                    &l,
                    rows,
                )
                .expect("a test");
                let got = term_columns(&table);
                let want = match df_method {
                    "none" => wald_chisq(&fit.beta, &fit.vcov, fit.aliased(), &l, rows),
                    _ => LinearTest::from_contrast_f(
                        fit.contrast_f(&l, rows).expect("a df record"),
                        if df_method == "satterthwaite" {
                            DenDf::Satterthwaite
                        } else {
                            DenDf::Residual
                        },
                    ),
                };
                let (den_df, den_df_kind) = den_df_fields(want.reference);
                assert_eq!(
                    [
                        got.statistic[0],
                        got.num_df[0],
                        got.den_df[0],
                        got.p_value[0]
                    ]
                    .map(f64::to_bits),
                    [want.statistic, want.num_df, den_df, want.p_value].map(f64::to_bits),
                    "{formula}: {l:?}"
                );
                assert_eq!(
                    (got.kind[0], got.den_df_kind[0]),
                    (kind_tag(want.kind), den_df_kind)
                );
                let notes: Vec<String> = want
                    .notes
                    .iter()
                    .map(|n| linear_note_line(*n, &r.names))
                    .collect();
                assert_eq!(got.row_notes[0], notes, "{formula}: {l:?}");
                tested += 1;
            }
        }
        assert_eq!(tested, 15);
    }

    #[test]
    fn contrast_test_refuses_a_missing_or_bad_record() {
        let r = run(
            "y ~ g + h + x + (1 | subj)",
            sim("sim_factorial"),
            "gaussian",
            "identity",
            1,
            None,
            None,
        );
        let e = vec![1.0; r.beta.len()];
        let test = |df_method: &str, record: Option<SatterthwaiteRecord>| {
            contrast_test(
                df_method,
                &r.beta,
                &r.vcov,
                &r.aliased,
                r.nobs,
                r.dispersion,
                record,
                &r.names,
                &e,
                1,
            )
        };
        let rec = r.satterthwaite_record.clone();
        assert!(test("satterthwaite", rec.clone()).is_ok());
        let err = |res: Result<TermTable, String>| res.unwrap_err();
        assert!(err(test("satterthwaite", None)).contains("needs its satterthwaite_record"));
        let mut short = rec.expect("a record");
        short.c.pop();
        assert!(err(test("satterthwaite", Some(short))).contains("is malformed"));
        assert!(err(test("other", None)).contains("unknown df_method"));
        // A Gaussian LMM with no pass: the NaN F row with its reason.
        let row = term_columns(&test("failed", None).expect("a row"));
        assert_eq!((row.kind[0], row.den_df_kind[0]), ("F", "satterthwaite"));
        assert!([
            row.statistic[0],
            row.num_df[0],
            row.den_df[0],
            row.p_value[0],
            row.sum_sq[0]
        ]
        .iter()
        .all(|v| v.is_nan()));
        assert_eq!(
            row.row_notes[0],
            [linear_note_line(
                LinearNote::NoDf {
                    reason: NoDfReason::Failed
                },
                &r.names
            )]
        );
    }
}
