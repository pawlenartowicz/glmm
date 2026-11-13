//! Friendly stable `fit` entry point for the `glmm` crate.
//!
//! Owns all scratch; dispatches on `(family, re.is_some())`; returns `Fit`.
//! This is the additive stable public surface —
//! it never touches any kernel; only marshals data and scratch into kernel
//! calls, then copies results out. The full route table — every path from
//! [`fit`]/[`fit_with_start`] to its solver — is in `documentation/algorithms.md`;
//! no route there is feature-gated. This file only tracks its own module
//! split.
//!
//! # Calling convention
//!
//! `x` is a design matrix in **row-major f64** layout: element `(i, j)` is at
//! `x[i * p + j]`. `y` is the response vector of length `n`. Both arrive in a
//! [`Data`]. The kernels use faer column-major f64 internally; conversion is
//! done here.
//!
//! Cluster ids and extra-grouping ids come from the caller's [`GroupIds`]; every
//! random-effect level count is derived from them.
//!
//! # Module layout
//!
//! `fit_checked` dispatches on `(family, re.is_some())` into one estimator
//! module per family — `ols`/`lmm` (Gaussian), `glm` (fixed-effects
//! binomial/Poisson/Gamma/NB), `glmm` (mixed non-Gaussian, plus GLMM-NB) —
//! each of which marshals into the matching numerical kernel
//! (`src/ols.rs`/`src/lmm/mod.rs`/`src/glm.rs`/`src/glmm/`). `common` holds
//! helpers shared by 2+ of those modules.

use crate::consts::{MAX_EXTRA_GROUPINGS, MAX_EXTRA_Q, MAX_PRIMARY_Q};
use crate::inference::cr2::Cr2Pieces;
use crate::inference::Objective;
use std::borrow::Cow;

use crate::{Data, Design, Family, GroupIds, GroupingRelation, ModelSpec, StartValues, WaldSe};

mod common;
mod core;
mod delta;
pub use delta::WEIGHT_POWER_SIMULATED;
mod glm;
mod glmm;
mod lmm;
mod ols;
pub(crate) use ols::ShiftBuf;
mod robust;

use common::fit_rank_deficient;
// `pub(crate)` so `Fit::r2` and `Fit::icc` check their data as the fit entry
// points do.
pub(crate) use common::validate;
// `pub(crate)` so `crate::predict` checks new rows as the fit checks its data.
pub(crate) use common::validate_design;
// `pub(crate)` so the calibration tier decides a design's aliasing with the
// alias gate's own test.
pub(crate) use common::detect_aliased;
// `pub(crate)` so `orchestrate::predict` finds the rows a port's stored
// rank-deficient fit cannot estimate.
#[cfg(feature = "orchestrate")]
pub(crate) use common::null_basis;
// `pub(crate)` so `orchestrate::run_fit` can check a port's warm-start θ width
// before the fit, the way `validate` checks it.
#[cfg(any(test, feature = "orchestrate"))]
pub(crate) use common::theta_width;
// `pub(crate)` so `orchestrate::run_compare` resolves one weight power for
// both of its fits.
#[cfg(feature = "orchestrate")]
pub(crate) use common::resolve_weight_power;
// The spec with its level counts, which the RE workspace builders in
// `crate::lmm` and `crate::glmm` read.
pub(crate) use common::SizedSpec;
// The random-effect-only column count, which the LMM accumulator sizes its
// slope buffers by.
pub(crate) use common::re_only_count;
// One block's covariance off θ̃, which the LMM profile reads its estimates
// through, as `Fit::varcorr` is assembled.
pub(crate) use common::varcorr_block;
// The flat-nesting check and its padded layout, which the formula front end
// runs on the groupings as written.
#[cfg(feature = "formula")]
pub(crate) use common::{detect_flat_nesting, nested_padded_ids};
// The sizer, for the tests outside `fit` that build kernel workspaces from
// data-derived ids.
#[cfg(test)]
pub(crate) use common::{sized_as_declared, spec_sized_from_ids};
// The workspace door, for the tests outside `fit` that need a `Fit` only the
// loop tier can produce (a prestandardized design); `loop_advanced` exports
// it below.
#[cfg(all(test, feature = "formula", not(feature = "loop_advanced")))]
pub(crate) use core::{build_workspace, fit_on};

/// Result of `fit`. Fixed-effect estimates cover all p predictors; SE has
/// the ranges below. Non-target SE slots are NaN.
///
/// Every per-grouping and per-θ field has one entry per grouping, or θ
/// coordinate, on every fit, and `fitted` one entry per row; a value the fit
/// has no estimate for is NaN. Only `condvar`, `condvar_x` and `condvar_cross`
/// can be empty (or `None`), as their docs say.
///
/// Construct one only through [`fit`]/[`fit_with_start`]; its fields are
/// private, so read it through its accessors. Everything the fit reports
/// about ITSELF, as opposed to about the data, lives on [`Fit::diagnostics`],
/// which is `#[non_exhaustive]`: a seventh diagnostic is then an additive
/// change instead of a major version.
#[non_exhaustive]
pub struct Fit {
    /// Storage behind [`Fit::beta()`].
    pub(crate) beta: Vec<f64>,
    /// Storage behind [`Fit::se()`].
    pub(crate) se: Vec<f64>,
    /// Storage behind [`Fit::vcov()`]: `p × p`, row-major.
    pub(crate) vcov: Vec<f64>,
    /// Per-element Cholesky-scaled values `theta[k]^2 * sigma_sq`. These equal
    /// the random-effect variance components only for diagonal/scalar RE
    /// components (q=1 / scalar-extra — the currently reachable case); slope
    /// (q≥2) models are not yet validated through this field. Empty on a fit
    /// without random effects, NaN on a fit with no finite endpoint.
    /// Laid out like `varcorr`: a block without a random intercept reports
    /// its slopes only.
    pub(crate) tau2: Vec<f64>,
    /// Storage behind [`Fit::dispersion()`].
    pub(crate) dispersion: f64,
    /// Storage behind [`Fit::diagnostics()`].
    pub(crate) diagnostics: Diagnostics,
    /// Storage behind [`Fit::varcorr()`]: one **vech-packed lower-triangular**
    /// block per grouping, declaration order. Vech order is column-major
    /// lower-triangular (matching the θ vech convention): for a `q×q` block,
    /// `(0,0),(1,0),…,(q-1,0),(1,1),…,(q-1,q-1)`. Empty on a fit without
    /// random effects; all NaN on a fit with no finite endpoint.
    pub(crate) varcorr: Vec<Vec<f64>>,
    /// Storage behind [`Fit::stddev_se`]: laid out like `tau2`, one vech per grouping, the SEs
    /// at the diagonal slots and NaN off them and wherever no SE was formed.
    pub(crate) stddev_se: Vec<f64>,
    /// Storage behind [`Fit::n_eval()`].
    pub(crate) n_eval: usize,
    /// Storage behind [`Fit::counters()`].
    #[cfg(feature = "counters")]
    pub(crate) counters: crate::counters::EvalCounters,
    /// Storage behind [`Fit::deviance()`].
    pub(crate) deviance: f64,
    /// Storage behind [`Fit::loglik()`].
    pub(crate) loglik: f64,
    /// Storage behind [`Fit::df()`].
    pub(crate) df: usize,
    /// Storage behind [`Fit::reml()`].
    pub(crate) reml: bool,
    /// Storage behind [`Fit::weight_power()`].
    pub(crate) weight_power: f64,
    /// Whether the fit estimated [`Fit::weight_power()`] ([`WeightPower::Estimate`]).
    pub(crate) weight_power_estimated: bool,
    /// Storage behind [`Fit::se_type()`].
    pub(crate) se_type: SeType,
    /// On a robust fit, the model covariance `σ̂²(XᵀV⁻¹X)⁻¹` of every retained
    /// column whatever [`FitOptions::se_for`] says, `p × p` row-major, NaN on
    /// an aliased column's row and column; empty on a model fit, whose `vcov`
    /// is it. What `predict` and the Type II hypotheses read on a robust fit.
    pub(crate) model_vcov: Vec<f64>,
    /// The CR2 or HC2 pieces of a robust fit; `None` on a model fit. Their
    /// `t` is in the caller's coordinates, `p` rows (zero on an aliased
    /// column); the rest is in the whitened coordinates of the retained
    /// columns.
    pub(crate) robust: Option<Cr2Pieces>,
    /// Storage behind [`Fit::nobs()`].
    pub(crate) nobs: usize,
    /// Storage behind [`Fit::objective()`].
    pub(crate) objective: Objective,
    /// Storage behind [`Fit::fitted()`].
    pub(crate) fitted: Vec<f64>,
    /// Storage behind [`Fit::ranef()`]: the groupings' blocks concatenated in declaration
    /// order; all NaN on a fit that did not converge.
    pub(crate) ranef: Vec<f64>,
    /// Storage behind [`Fit::ranef_levels()`]: one level count per grouping on
    /// every fit with random effects.
    pub(crate) ranef_levels: Vec<usize>,
    /// Storage behind [`Fit::condvar()`]: the groupings' blocks concatenated in declaration
    /// order. Empty when [`FitOptions::condvar`] is off, on a layout that forms
    /// none, and on a fit with no conditional modes (one that did not converge).
    pub(crate) condvar: Vec<f64>,
    /// Per level `l`, in the level order of [`Fit::condvar`], the row-major
    /// `q × p` block `K_l` of `Λ̂(Λ̂ᵀZᵀWZΛ̂ + I)⁻¹Λ̂ᵀZᵀWX` on the caller's columns, the
    /// same solves as `condvar` (`q` without a hidden intercept, `p` the
    /// caller's width; an aliased column's entries are NaN). With `V` the
    /// fixed-effect covariance, `Cov(b̂_l, β̂) = −K_l·V`. Empty exactly when
    /// `condvar` is.
    pub(crate) condvar_x: Vec<f64>,
    /// The conditional covariance between the random effects of different
    /// groupings, which `condvar`'s per-level blocks leave out, on its scale;
    /// see [`CondvarCross`]. `None` on a fit with one grouping and whenever
    /// `condvar` is empty.
    pub(crate) condvar_cross: Option<CondvarCross>,
    /// Storage behind [`Fit::coef_df()`].
    pub(crate) coef_df: Vec<f64>,
    /// What the Satterthwaite pass (or the residual df) left for
    /// [`Fit::contrast_t`] and [`Fit::contrast_f`]: on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated, the residual-only record (the
    /// full `C` and `ν = n − p`); on a Gaussian LMM (REML or ML), the pass's
    /// state, its switched-off record, the failed record of a fit with no
    /// finite endpoint, or the not-run record of a `Fit` the loop tier's
    /// `FitView::into_fit` built; `None` on every other family.
    pub(crate) satterthwaite: Option<crate::lmm::satterthwaite::SatterthwaiteState>,
    /// The design summary of a Gaussian fit whose only random term is one
    /// random intercept `(1 | g)`, which has no prior weights (an offset is
    /// allowed) and at least one fixed-effect column: `Some` on every such
    /// fit, whatever its criterion (REML or ML) and whether or not it
    /// converged, since the arrays depend on the design alone; `None` on
    /// every other fit, a `(1 | g)` fit with no fixed effects included.
    /// Copied out of the dense LMM accumulator where the `Fit` is assembled,
    /// so no fit computes anything extra for it. The arrays and their flat
    /// layout are on [`DesignSummary`].
    pub(crate) design_summary: Option<DesignSummary>,
    /// Orthonormal basis of the null space of the fixed design `X` the fit
    /// was given, `k × p` row-major, one row per aliased column; empty on a
    /// full-rank fit. [`crate::predict`] reads it to find rows that are not
    /// estimable.
    pub(crate) null_basis: Vec<f64>,
}

/// The aggregates a `(1 | g)` fit's dense accumulator holds, as three flat
/// arrays. They are on the design the fit ran on: after aliased columns are
/// dropped, so `p` here is the retained column count, and in the fit's
/// orthogonalized basis `X̃ = X·B·U⁻¹` (`B` the column standardization, `U`
/// the fixed-effect factor), which spans the column space of the caller's `X`.
/// `K` is the highest primary level id with rows plus one, and a level below
/// it with no rows holds zeros.
///
/// - `counts`, length `K`: the row count of each level.
/// - `s`, length `p·K`, the `p × K` matrix `X̃ᵀZ` column-major:
///   `s[a·p + j] = Σ X̃ᵢⱼ` over the rows `i` of level `a`.
/// - `c`, length `p·p`, the `p × p` matrix `X̃ᵀX̃` column-major, both triangles
///   filled (it is symmetric): `c[j·p + k] = c[k·p + j] = Σᵢ X̃ᵢⱼ·X̃ᵢₖ`.
///
/// With `P₀` the residual projection of `X`, `ZᵀP₀Z = diag(counts) − sᵀc⁻¹s`.
// Read by the two-fit comparison (`formula::anova::Model::from_fit`), so a
// build without `formula` (or tests) holds the arrays unread.
#[cfg_attr(not(any(test, feature = "formula")), allow(dead_code))]
pub(crate) struct DesignSummary {
    pub(crate) counts: Vec<f64>,
    pub(crate) s: Vec<f64>,
    pub(crate) c: Vec<f64>,
}

/// The conditional covariance between the random effects of different
/// groupings (all of `Λ̂(Λ̂ᵀZᵀWZΛ̂ + I)⁻¹Λ̂ᵀ` that [`Fit::condvar`] does not
/// hold), as flat arrays on `condvar`'s scale and reported columns.
///
/// One grouping is the family grouping: each of its levels is a family,
/// together with its children when the model has a nested grouping (`m`
/// child slots per family, `m = 0` without one). Every other grouping is a
/// crossed, intercept-only one, and their levels form the tail: the
/// groupings in declaration order with the nested one skipped, except that
/// the declared primary stands at the family grouping's place when the two
/// differ, unless it is the nested grouping; each grouping's levels in order.
///
/// - `family_grouping`: the declaration index of the family grouping; 0
///   unless the fit made a crossed extra its primary.
/// - `nested_slots`: when the declared primary is the nested grouping (the
///   fit moved a slope grouping first and nested the primary in it), per
///   declared level `l` of it the child slot `f·m + j` it holds; empty
///   otherwise, and then a nested level's id is its slot.
/// - `family`, per family `f`, the vech (column-major lower triangle, as
///   `varcorr`) of the `w × w` covariance of its `w = q + m` coefficients: the
///   level's `q` reported coefficients, then the nested grouping's slots
///   `f·m .. (f+1)·m`.
/// - `tail`, the vech of the `t × t` covariance of the tail levels.
/// - `coupling_at` (one entry per family plus one), `coupling_cols` and
///   `coupling`: family `f`'s entries `e` in `coupling_at[f]..coupling_at[f+1]`
///   name a tail level `coupling_cols[e]` and hold `w` values
///   `H_f[e] = coupling[e·w..(e+1)·w]`, one per family coefficient. Tail levels
///   no entry names have `H_f[e] = 0`. With `T` the `tail` matrix,
///   `Cov(coefficient a of f, tail level t) = −Σ_e H_f[e][a]·T[cols_e, t]`, and
///   between coefficients of two different families `f ≠ g`
///   `Σ_e Σ_e' H_f[e][a]·T[cols_e, cols_e']·H_g[e'][a']`.
///
/// Memory: `Σ_f w(w+1)/2 + t(t+1)/2` values plus `w` per entry, the entries
/// at most one per family and tail level its training rows share, and `t` at
/// most `MAX_CROSSED_LEVELS`.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CondvarCross {
    pub(crate) family_grouping: usize,
    pub(crate) family: Vec<f64>,
    pub(crate) tail: Vec<f64>,
    pub(crate) coupling_at: Vec<u32>,
    pub(crate) coupling_cols: Vec<u32>,
    pub(crate) coupling: Vec<f64>,
    pub(crate) nested_slots: Vec<u32>,
}

/// What the kernels' factor walks return, in kernel coordinates: the
/// per-level blocks behind [`Fit::condvar`], the rows behind
/// [`Fit::condvar_x`], and the pieces `common::assemble_condvar_cross` maps to
/// [`Fit::condvar_cross`].
pub(crate) type CondvarBlocks = (Vec<f64>, Vec<f64>, Option<CondvarCross>);

/// Everything a fit reports about itself, reached as `fit.diagnostics()`.
///
/// **Coverage is not uniform across routes**, and the per-field docs say where
/// each one is real. The short version: `converged` and `aliased` are filled
/// everywhere; `pinned` is real on every route that has variance components to
/// pin, while `boundary` distinguishes all three states only on the LMM and
/// GLMM routes; `notes` can only ever be raised by OLS, GLM and LMM — the GLMM
/// forms no factor to measure a pivot on. An absent note is therefore "not
/// detected", never "checked and clean".
///
/// `#[non_exhaustive]`: match with a trailing `..`. That is the whole point of
/// collecting these here — the six channels below arrived one at a time, each
/// arrival breaking `Fit`, and a seventh now costs nobody a major version.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Diagnostics {
    /// Whether the optimizer reached its convergence criterion. `false`
    /// usually means `se()`, `vcov()`, `dispersion()` and `varcorr(g)` return
    /// the NaN-fill described on each of those accessors. The exception is an LMM or GLMM fit that hit
    /// `MaxFunReached`: `se()`, `vcov()`, `dispersion()` and `varcorr(g)` all
    /// report finite values at that budget-exhausted endpoint. `stddev_se`,
    /// `fitted` and `ranef` stay gated on `converged` on both routes; `df`
    /// follows that same gate only on the GLMM route (an LMM `MaxFunReached`
    /// endpoint reports its usual nonzero `df` instead, see [`Fit::df`]).
    pub converged: bool,
    /// `true` iff the fit converged onto the θ boundary (≥ 1 diagonal variance
    /// component pinned at 0 — `boundary == AtBoundary`, OR a converged
    /// diagonal stddev negligible next to its fit's largest — see
    /// `Fit::has_negligible_component`). This is glmm's own rule: lme4's
    /// `isSingular` compares its `tol` with the relative covariance factor,
    /// which is on another scale. `false` for OLS/GLM and for an LMM or GLMM
    /// `MaxFunReached` cap-out — a capped endpoint is reported as a point,
    /// not accepted onto the boundary, so the post-hoc negligible-component
    /// check does not run there even though `varcorr` is finite (both routes
    /// gate that check on `converged` for exactly this reason), and it never
    /// sets this flag even when its diagonals are near zero.
    ///
    /// Not a pure restatement of `boundary`: the negligible-component check is
    /// a reporting rule applied to the assembled `varcorr` after the fact, so
    /// `singular` can be `true` with `boundary == Interior`.
    pub singular: bool,
    /// Rank-deficiency mask, length `p`: `true` for a fixed-effect
    /// column dropped because it is aliased (linearly dependent) on an earlier
    /// column, mirroring lme4's `NA`-coefficient behavior. The corresponding
    /// `beta`/`se` slots are `NaN` and `converged` stays `true` (the reduced
    /// model fits). All-`false` when the design is full-rank.
    ///
    /// Decided by the alias gate that runs BEFORE dispatch, not by any fitting
    /// route — a column merely hard to identify is fitted and flagged with a
    /// [`Note::IllConditioned`], not aliased.
    pub aliased: Vec<bool>,
    /// Where in the θ parameter space the accepted point sits.
    pub boundary: Boundary,
    /// Which variance components were pinned to the boundary: per grouping, per
    /// dimension, ALIGNED WITH THE `varcorr` BLOCKS, so `pinned[g][i]` pairs
    /// with `stddev_corr(g).0[i]`. That alignment is the point — it is the
    /// layout both wrapper packages already iterate.
    ///
    /// A pinned θ diagonal marks a rank-deficient Σ̂, not necessarily a zero
    /// variance: component `i` has zero variance *conditional on the components
    /// before it in its block*. It is a zero variance component only when the
    /// rest of its Λ row is zero too. At a correlation boundary (`|ρ| = 1`) the
    /// trailing diagonal pins while that component's own variance (row `i` of
    /// `ΛΛ′`) stays positive, so read variances from `varcorr`, never from this
    /// flag.
    ///
    /// The flag means that because the pin is followed by a re-factorization
    /// of each pinned RE block with the pin inside it
    /// (`crate::lmm::canonicalize_pinned_blocks`): the reported Λ is the
    /// canonical lower Cholesky of Σ̂, so a pinned diagonal's whole column is
    /// exactly zero and the variance that used to sit below it has been folded
    /// into a trailing diagonal, and a trailing pivot the fold leaves at or
    /// below the pin threshold pins in the same pass. The pin pass's fold is
    /// not Σ-preserving: Σ̂ loses each pinned column's conditional variance (at
    /// most `PIN_THETA²`) and its conditional covariances with the later
    /// components. On a converged Laplace or LMM fit a diagonal is flagged
    /// exactly when it is 0, except where the GLMM incumbent check below undoes
    /// a pin. Without the fold, a zero θ diagonal on a `q ≥ 2` block could sit
    /// above a live off-diagonal and flag a direction in which Σ̂ still varies.
    ///
    /// **The GLMM incumbent check.** A pin that raises the GLMM objective past
    /// its slack is undone. At the search's endpoint θ goes back to that
    /// endpoint, unfolded, so a diagonal in (0, `PIN_THETA`] stays at its small
    /// value and is not flagged. After a trough polish, the undone point is
    /// folded with Σ kept, and only the diagonals that fold leaves exactly 0
    /// are flagged.
    ///
    /// **Exception: AGQ (`nagq > 1`).** The flags come from the same canonical
    /// form, but the fit reports the search's own Λ, because the vector AGQ
    /// deviance depends on Λ and not only on Σ. So a pinned diagonal of the
    /// fitted θ̂ can sit above a live entry, and an exact-zero diagonal of the
    /// fitted θ̂ can be unflagged. The flags still describe Σ̂ up to the
    /// pinned pivots (at most `PIN_THETA²` each).
    ///
    /// **Centered slopes.** When a slope of a block is centered, the flags are
    /// the exact zeros of the caller's θ, the lower factor of `N·Λ̃Λ̃ᵀ·Nᵀ`,
    /// which can sit on another diagonal than the kernel's.
    ///
    /// **Exception: the sparse route.** `canonicalize_pinned_blocks` skips a
    /// block with `q` past `MAX_PRIMARY_Q` (`crate::lmm::canonicalize_pinned_blocks`),
    /// which only the sparse route ever builds, so on that route a pinned
    /// diagonal can sit above a live off-diagonal after all.
    ///
    /// **One row per grouping, aligned with `varcorr`, on every fit with
    /// random effects; all `false` when nothing was pinned.** Each view mapper
    /// reshapes the fitted mask through `common::materialize_diagnostics`, so
    /// there is no route that knows a component pinned but declines to say
    /// which. Empty on a fit without random effects (OLS, GLM, and the
    /// fixed-effect-only negative binomial). The one structural exception: a
    /// design past the 64 components the internal mask holds — outside the
    /// crate's currently validated envelope — cannot be represented here at
    /// all.
    pub pinned: Vec<Vec<bool>>,
    /// Solver observations that are not one of the fixed channels above. Empty
    /// on a clean fit, and empty allocates nothing.
    pub notes: Vec<Note>,
    /// Gradient evaluations the BFGS outer search made, on the LMM and GLMM
    /// routes that search with it, Laplace and AGQ (listed on
    /// [`Note::GradientSearchFallback`]).
    /// [`Fit::n_eval`] does not count them, because a gradient reuses its
    /// point's value evaluation. 0 on every fit that ran no gradient search.
    /// This field and the two below describe the BFGS search even when BOBYQA
    /// finished the fit after it.
    pub n_grad: usize,
    /// Why the BFGS θ search stopped; `None` on every fit that ran no gradient
    /// search.
    pub outer_stop: Option<OuterStop>,
    /// The BFGS decrement `½·gᵀHg` at the search's endpoint (deviance units),
    /// the quasi-Newton estimate of the decrease left. NaN on every fit that
    /// ran no gradient search.
    pub outer_decrement: f64,
}

/// Where the accepted θ sits in its parameter space.
///
/// Every θ-carrying route distinguishes all four: the LMM over either kernel
/// and the GLMM over every layout report the state the search reached, cap-out
/// (`NoOptimum`) included. OLS and GLM have no θ and always report `Interior`,
/// and so does a `Fit` that failed before any search ran (the degenerate `n`/`p`
/// guards), which says so through `converged: false`.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    /// Every variance component settled strictly inside its range.
    Interior,
    /// ≥ 1 diagonal variance component accepted at 0.
    AtBoundary,
    /// The optimizer capped out at a finite endpoint — reported as a point, not
    /// accepted onto the boundary. `deviance` still carries that endpoint.
    NoOptimum,
    /// ≥ 1 θ coordinate ended on the upper limit of its search box
    /// (`crate::lmm::THETA_HI`, 1000 on the internal scale), on a fit that
    /// would otherwise read `Interior`: the search stopped on that wall, not at
    /// an optimum. Always comes with [`Note::VarianceAtLimit`], which an
    /// `AtBoundary` or `NoOptimum` fit also carries when it reaches the limit.
    /// `converged`, `singular` and the estimates are the search's own.
    ///
    /// Cold path only: the loop tier's `FitDiagnostics` carrier has no code for
    /// it, so a loop-tier read of such a fit says interior.
    UpperLimit,
}

/// A solver observation about the fit that has no dedicated field. The enum
/// variant, not any English sentence, is the stable identifier a caller filters
/// on; the wrappers turn each variant into their own warning category (Python)
/// or condition class (R).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum Note {
    /// Fitted, but these columns are not separately identified: the design is
    /// computable and the estimates are honest, while the standard errors are
    /// large because the data genuinely cannot separate the columns. Not a
    /// failure and not a dropped column — contrast `aliased`, which is the
    /// genuinely-redundant case.
    ///
    /// `columns` carries the single column the pivot search named. Its
    /// entangled partners are NOT currently identified: the measurement finds
    /// the worst-conditioned column, not the set it is confounded with, and
    /// inventing a partner rule would be guessing. `pivot` is the measured
    /// scale-invariant ratio at that column, so a caller can rank severity
    /// across fits instead of treating the flag as binary.
    IllConditioned {
        /// Design-matrix column indices, into the `x` the caller passed.
        columns: Vec<u32>,
        /// The measured scale-invariant pivot ratio — smaller is worse.
        pivot: f64,
    },
    /// A GLMM PIRLS inner solve ran the full `PIRLS_MAX_ITERS` (200) cap without
    /// satisfying its convergence band — never a failure surfaced any other way
    /// (that is the `(NaN, NaN, NaN, false)` halving-exhaustion/Cholesky-failure
    /// case, which the fit already rejects or NaN-fills through the usual
    /// channels). The cap stays 200. A note where `final_eval` is `false` costs
    /// nothing observable — that inner solve was a rejected trial point, not
    /// what the fit reports; a note where `final_eval` is `true` means the
    /// reported numbers rest on a PIRLS solve that never settled inside the cap.
    /// FD-Hessian SE evals are excluded — only fit-path evals count.
    PirlsExhausted {
        /// How many fit-path BOBYQA objective evals hit the cap over the course
        /// of this fit (0 if `final_eval` alone is what fired).
        evals: u32,
        /// Whether the FINAL re-evaluation at the reported γ̂ itself hit the
        /// cap — the case that matters, since that solve's ũ/W̃ feed the
        /// reported estimates directly rather than being a rejected trial
        /// point. That re-evaluation runs on a budget-exhausted outer exit
        /// too, not only on a converged one.
        final_eval: bool,
    },
    /// The negative-binomial GLM's alternation between β̂ (IRLS at fixed θ) and θ̂
    /// (profile maximisation at fixed μ̂) ran its full `NB_MAX_OUTER` rounds
    /// without meeting its relative-change tolerance on θ. The fit is reported at
    /// the last θ, with β̂ and the SEs refit there, so the numbers are one
    /// model's; that θ may still be short of the profile optimum. `converged`
    /// keeps the last inner IRLS fit's flag. Mixed negative-binomial models
    /// search ln θ inside their outer BOBYQA instead and never raise this.
    NbShapeUnsettled {
        /// Alternation rounds run (the cap, `NB_MAX_OUTER`, in production).
        rounds: u32,
    },
    /// The negative-binomial shape θ̂ ended on a limit of its search range
    /// [0.001, 10000]: within 1e-6 of it on the log scale. The shape and the
    /// estimates computed at it are at that limit, not at an optimum. At the
    /// upper limit the data show no overdispersion the shape can measure (a
    /// near-Poisson fit); at the lower one they are more overdispersed than
    /// the range allows. `converged` and every estimate are the search's own.
    /// Raised by the NB GLM on a converged fit and by the NB GLMM on a fit
    /// with a finite endpoint.
    NbShapeAtLimit {
        /// The limit θ̂ ended on, exactly 0.001 or 10000.
        limit: f64,
    },
    /// `FitOptions::wald_se == WaldSe::Hessian` was requested, but the
    /// finite-difference joint Hessian came back non-positive-definite (or a
    /// perturbed deviance evaluation was non-finite), so the SE pass fell back
    /// to the RX/Schur route instead. `Fit::se`/`vcov` are still filled (from
    /// the fallback), but `Fit::stddev_se` is NaN — no θ-block SE comes out of
    /// a Schur inverse. Never fires under `WaldSe::Rx`, which never attempts
    /// the joint Hessian.
    HessianSeFallback,
    /// The GLMM's `ExactProfile` route (the exact β-profile outer search,
    /// `src/glmm/mod.rs`) ended not converged, so the fit reran once on the
    /// `PqlThenJoint` route instead, from the same start. The reported
    /// estimates come from that rerun when it converged; when it also failed
    /// to converge, they come from whichever of the two attempts reached the
    /// lower deviance — `converged` stays `false` either way, and `n_eval`
    /// counts both attempts. Never raised for a shape that does not route
    /// `ExactProfile` (`Joint` and `PqlThenJoint` have no fallback of their
    /// own).
    ExactProfileFallback,
    /// The gradient search (BFGS: LMMs whose random slopes are all on one
    /// grouping; binomial, Poisson and Gamma Laplace GLMMs with at least two θ
    /// on the blocked or structured layout; binomial logit and probit and
    /// Poisson log AGQ GLMMs with one scalar random intercept, no other
    /// grouping and under 50 rows per cluster, cold or from a warm θ of at
    /// least 0.05) did not converge, so BOBYQA, the
    /// derivative-free search every other model uses, continued from the best
    /// point it had reached, with its own full evaluation budget. The reported
    /// estimates and `converged` are that search's. `n_eval` counts both.
    GradientSearchFallback,
    /// A GLMM search reached its endpoint, but no fixed-effect covariance could
    /// be formed there: the β information (the Rx Schur complement, and under
    /// `WaldSe::Hessian` the joint Hessian before it) is not positive definite
    /// to rounding. The estimates, the variance components, the deviance and
    /// `converged` are the search's own; `Fit::se`, `Fit::vcov` and
    /// `Fit::stddev_se` are NaN. Reached on a near-duplicate fixed-effect
    /// column the alias gate keeps, and on fits whose working weights span
    /// many decades.
    ///
    /// On a GLM: the fit converged, but the QR of `√W·X` at its last step has
    /// a zero or non-finite diagonal at a column that is not zero under the
    /// weights, so that column cannot be separated from the ones before it.
    /// [`Note::IllConditioned`] comes with it and names the first such column
    /// (`pivot` 0); `Fit::beta` is NaN at every such column, since the fit held
    /// them at their start values. The other estimates, the deviance, `loglik`,
    /// `fitted` and `converged` stand; `Fit::se` and `Fit::vcov` are NaN.
    SeUnavailable,
    /// The response is not within `1e-3` of an integer count — the raw value
    /// for Poisson/negative-binomial, the derived successes `y·wᵢ` for
    /// binomial. The fit still runs (the deviance and score are continuous in
    /// `y`), but the reported numbers describe a distribution that assumes
    /// integer counts. The `1e-3` tolerance matches R's
    /// `binomial()$initialize` non-integer-count check; R's Poisson/NB
    /// warning is unrelated (it comes from `dpois`/`dnbinom` inside
    /// `poisson()$aic`/`negative.binomial()$aic`, at a much tighter
    /// `1e-7·max(1,|y|)`, so R warns on e.g. `y = 2.0005` where glmm stays
    /// silent). Never raised for Gaussian, Gamma or inverse-Gaussian, whose
    /// response is continuous.
    NonIntegerResponse {
        /// Rows whose response (successes, for binomial) misses the nearest
        /// integer by more than the tolerance.
        rows: u32,
    },
    /// A random-effect term's grouping factor has only one observed level, so
    /// no between-group variance is identifiable from it — lme4 errors here
    /// ("grouping factors must have > 1 sampled level"); this crate instead
    /// drops the term and fits the rest of the model. When every random-effect
    /// term is dropped this way, the model fits with `re: None`, the same as a
    /// formula that declared no random effects at all.
    ///
    /// Raised by the formula frontend, not by a solver: the level count is a
    /// property of the lowering, not of any fitted model. Reached through
    /// [`crate::formula::Lowered::notes`].
    SingleLevelGroupingDropped {
        /// The grouping factor, as the formula spells it.
        grouping: String,
    },
    /// A random-effect term has at least as many random effects as the fit
    /// has rows (more than the rows, on a family other than Gaussian), so its
    /// variance parameters probably cannot be estimated from these data. The
    /// count is the term's observed levels times its columns (the intercept,
    /// one per numeric slope, one per non-base level of a factor slope):
    /// `(1 + x | g)` on 10 levels has 20. lme4 stops on such a model by default
    /// (`check.nobs.vs.nRE`: `lmer` when the rows are at most the count,
    /// `glmer` when they are below it, so an observation-level `(1 | obs)`
    /// passes on a GLMM). Each term is checked against all rows on its own;
    /// the counts are not summed across terms. The fit runs as usual.
    ///
    /// Raised by the formula frontend, not by a solver, for the same reason
    /// [`Note::SingleLevelGroupingDropped`] is. A term dropped by
    /// [`Note::SingleLevelGroupingDropped`] is not checked. Reached through
    /// [`crate::formula::Lowered::notes`].
    FewRowsForRandomEffects {
        /// The term's left-hand side with its intercept written out, `"1"` or
        /// `"1 + x"`.
        terms: String,
        /// The grouping factor, as the formula spells it.
        grouping: String,
        /// The term's random effects: observed levels times columns.
        random_effects: u32,
        /// The rows of the fit.
        rows: u32,
    },
    /// Every row of a grouping has the same level, so the data hold no
    /// between-group variance for it. Groupings are numbered in declaration
    /// order, as on [`Note::NewGroupingLevels`]. The fit runs as usual and no
    /// estimate changes. Reachable from the matrix door only: the formula
    /// frontend drops such a grouping first and raises
    /// [`Note::SingleLevelGroupingDropped`].
    SingleLevelGrouping {
        /// The grouping's declaration index: 0 the primary grouping, `g + 1`
        /// the extra grouping `g`.
        grouping: u32,
    },
    /// Two groupings put the rows into exactly the same groups (the same
    /// partition of the rows, whatever the ids are called) and their blocks
    /// share a random-effect column: both have a random intercept, or one
    /// slope column is in both. The data cannot tell the two variances apart,
    /// so how the fit splits the variance between them is arbitrary. The fit
    /// runs as usual and no estimate changes.
    RepeatedGrouping {
        /// The earlier grouping's declaration index, as on
        /// [`Note::NewGroupingLevels`].
        first: u32,
        /// The later grouping's declaration index; `first < second`.
        second: u32,
    },
    /// A grouping's block has a random intercept, and some of its slope
    /// columns take one value within every level of the grouping. With `k`
    /// such columns, the level effects identify the covariance of the
    /// intercept and those slopes only through as many equations as there
    /// are distinct per-level vectors of the products of their values within
    /// each [`crate::ReTerm`] (the intercept's value is 1); the note is
    /// raised when that count is below the number of covariance parameters
    /// among them: `(k + 1)(k + 2)/2` when they share one term, fewer when
    /// terms split them. So a slope of ±1 per level in a term of its own
    /// gets the note. Values are compared with `==`. Then
    /// their variances and correlations cannot be estimated from these data. The fit runs as usual and no estimate
    /// changes. A block without a random intercept gets no note: there a
    /// slope constant within levels is a random intercept with a per-level
    /// scale, which is identified.
    SlopeConstantWithinLevels {
        /// The grouping's declaration index, as on [`Note::NewGroupingLevels`].
        grouping: u32,
        /// Positions in the grouping's slope list, ascending: position `k` is
        /// entry `1 + k` of [`Fit::stddev_corr`] for this grouping.
        slopes: Vec<u32>,
    },
    /// A binomial response is 0 or 1 on every row and takes one value within
    /// every level of a grouping with a random intercept, and some level holds
    /// more than one trial (two or more rows, or a row whose weight, its trial
    /// count, is above 1). The likelihood then keeps rising as that grouping's
    /// variance grows, so the variance has no finite estimate, and the fixed
    /// effects depend on the reported one. Checked on every level, not most.
    /// A grouping of one-trial levels gets [`Note::OneTrialPerLevel`]
    /// instead. A response with one value on every row
    /// gets no note either: that is the constant-response case. The fit runs
    /// as usual and no estimate changes.
    SeparatedByGrouping {
        /// The grouping's declaration index, as on [`Note::NewGroupingLevels`].
        grouping: u32,
    },
    /// A binomial grouping whose every level holds at most one trial: the sum
    /// of the level's prior weights (its trials) is at most 1, as with one 0/1
    /// row per level. The variance of such a grouping only moves each row's
    /// marginal success probability, which the fixed effects can match
    /// (exactly under probit, where it is `Φ(η/√(1 + σ²))`), so the variance
    /// trades against the fixed effects and is not identified. Checked on
    /// every grouping, with or without a random intercept. The fit runs as
    /// usual and no estimate changes.
    OneTrialPerLevel {
        /// The grouping's declaration index, as on [`Note::NewGroupingLevels`].
        grouping: u32,
    },
    /// A GLMM's deviance is too large for `f64` to resolve its optimum: its
    /// estimated rounding error at the endpoint exceeds 1e-2 deviance units,
    /// which moves θ̂ by about a tenth of its standard error, and the search
    /// sees a noise several times that. β̂, θ̂ and the deviance may sit off the
    /// optimum; `converged` and every estimate are the search's own. Two
    /// sources add up in the estimate: the deviance itself, about 1e13 or more
    /// (a model that misses a large effect at counts in the billions), and,
    /// with more than one grouping factor, `log|A|`, whose rounding grows with
    /// the extra factors' `θ²·Σwᵢ` (crossed or nested intercepts at counts in
    /// the trillions). A model that captures the missing effect removes the
    /// first source. On a Gaussian mixed model the estimate is
    /// `(k + p + 1)·ε·y′Wy/σ̂²` (`k` random-effect columns, `y′Wy` on the
    /// working weights): the residual sum of squares is a difference of terms
    /// of the size of `y′Wy`, known to about that times ε. Under REML it also
    /// counts each fixed-effect pivot of X′V⁻¹X, `(k + p + 1)·ε·C_jj/L_jj²`
    /// (`C_jj` the column's weighted sum of squares, `L_jj²` its pivot): the
    /// pivot is a difference of two large numbers when a fixed-effect column
    /// lies near the span of a random-effect column with a large variance (an
    /// intercept beside a random intercept). The first term passes 1e-2 when
    /// the residuals are tiny against the response: a response far from zero
    /// next to its residual SD (about 1e5 times it at n = 1000), or weights
    /// spanning about 1e11 or more. There θ̂ stops where the search first
    /// meets the exact-fit floor, so the log-likelihood can sit far below its
    /// maximum, and a fitted weight power can stop far from the best one.
    PrecisionLimited,
    /// The Satterthwaite pass of a Gaussian LMM ([`Fit::coef_df`]) met a
    /// curvature it cannot take at face value. Raised when the Hessian of the
    /// profiled deviance over the free variance parameters had directions the
    /// pseudo-inverse left out (flat to `1e-8` of its largest eigenvalue, or
    /// negative), as lmerTest warns; when the df were computed at an endpoint
    /// that did not converge (`MaxFunReached`); and when the pass failed: a
    /// deviance or covariance on the pass's stencil was not finite, the
    /// eigendecomposition of that Hessian failed, or the fit has no finite
    /// endpoint at all. A failed pass fails for the whole fit: every `coef_df`
    /// is NaN and the contrasts return
    /// [`crate::inference::contrast::NoDfReason::Failed`]. Not raised on a fit
    /// with no fixed-effect coefficients, which has no df to compute.
    SatterthwaiteCurvature {
        /// Directions of the Hessian dropped as flat.
        dropped: u32,
        /// Directions of the Hessian below `−1e-8` of its largest eigenvalue,
        /// not inverted.
        negative: u32,
        /// Whether the endpoint the df were computed at converged.
        converged: bool,
        /// The pass failed: a deviance or covariance on the stencil was not
        /// finite, the Hessian's eigendecomposition failed, or the fit has no
        /// finite endpoint.
        nonfinite: bool,
    },
    /// The fixed-effects IRLS stopped on separation: the deviance settled
    /// while the last step still moved some linear predictor by a large
    /// amount, so at least one coefficient runs off to ±∞ (binomial, Poisson,
    /// negative binomial). `converged` is `false` and `beta` holds the last
    /// step. On a GLM fit that is the fit itself; on a GLMM fit it is the GLM
    /// the search starts from, so the search never ran: the random-effect
    /// outputs, the SEs and the deviance are NaN. A binomial GLMM whose
    /// response is all 0 or all 1 carries it too, without any IRLS
    /// iteration, with `beta` NaN.
    Separated,
    /// A random-effect parameter ended on the upper limit of the θ search box
    /// (`crate::lmm::THETA_HI`, 1000 on the search's internal scale, which on
    /// a Gamma inverse-link GLMM is θ in the units of the rescaled response
    /// the search ran on, not the reported θ): the estimates sit
    /// on that limit, not at an optimum. On a Gaussian model this usually means
    /// the residual variance is estimated at or near zero (θ is a standard
    /// deviation relative to the residual one); on other families the random
    /// effect has no finite estimate. Raised on any fit with a finite endpoint
    /// (sets [`Boundary::UpperLimit`] when the boundary would otherwise read
    /// `Interior`); `converged` and `singular` are not changed.
    VarianceAtLimit,
    /// A Gamma GLMM's free dispersion ended on an edge of its search box
    /// `[GAMMA_PHI_LO, GAMMA_PHI_HI]`. On a weighted fit the box applies to φ
    /// divided by a power of two near the geometric mean of the weights. The
    /// fit is refused: `converged` is false, the boundary is `NoOptimum` and
    /// every estimate is NaN. A held dispersion is not searched and never gets
    /// it.
    DispersionAtLimit {
        /// The edge φ̂ ended on, on the scale of the dispersion the fit
        /// reports: `edge` times the power of two the weights were divided
        /// by, so `edge` itself on an unweighted fit.
        limit: f64,
        /// The edge of the box φ̂ ended on, exactly `GAMMA_PHI_LO` or
        /// `GAMMA_PHI_HI`. A typical row's coefficient of variation there is
        /// about `√edge`.
        edge: f64,
    },
    /// A prediction on new rows ([`crate::predict`]) met grouping levels the
    /// fit has no conditional mode for: an id at or past the grouping's
    /// [`Fit::ranef_levels`] count. Those rows take a zero random effect for
    /// this grouping, and nothing from it in their standard error, while the
    /// row's other groupings keep theirs, as lme4's `allow.new.levels = TRUE`
    /// does. One note per grouping with such rows, in declaration order.
    /// Raised only when [`crate::PredictOptions::allow_new_levels`] is set (the
    /// default); without it the same rows are a [`FitError::Ids`]. Never raised
    /// by a fit.
    NewGroupingLevels {
        /// The grouping's declaration index: 0 the primary grouping, `g + 1`
        /// the extra grouping `g`.
        grouping: u32,
        /// The 0-based rows of the new design with an unseen level, ascending.
        rows: Vec<u32>,
    },
    /// A prediction on new rows ([`crate::predict`]) met rows outside the row
    /// space of a rank-deficient fit's design: `‖N·x‖ > 1e-6·‖x‖`, `N` an
    /// orthonormal basis of the null space of the training design, R's
    /// `predict.lm` rule. Their prediction, standard error, interval and
    /// future bounds are NaN. Only in [`crate::Prediction::notes`].
    RowsNotEstimable {
        /// The 0-based rows of the new design outside the row space,
        /// ascending.
        rows: Vec<u32>,
    },
    /// The fitted weight power ([`WeightPower::Estimate`]) is not a maximum:
    /// the log-likelihood at the bracket's end nearer to it is at least its
    /// own (to 1e-9 relative), so the profile rises, or goes flat to rounding,
    /// toward that end. With at most `p` rows off the common weight the
    /// profile goes flat and can have no finite maximum, and on very narrow
    /// weights the maximum can lie past the bracket.
    WeightPowerAtBound {
        /// The power the fit used, [`Fit::weight_power`].
        delta: f64,
        /// The bracket's lower end, `−min(max(2, min(30, ln(1e28)/ln s)),
        /// ln(1e250)/maxᵢ|ln wᵢ|)` with `s` the largest weight over the
        /// smallest.
        lower: f64,
        /// The bracket's upper end, `−lower`.
        upper: f64,
    },
    /// Fits inside the weight-power search failed or did not converge. A
    /// failed one (its working weights not finite and `> 0`, or a log-likelihood
    /// that is not finite) scores `−1e300·(1 + |δ|)`, which steers the search
    /// toward `δ = 0`; one that did not converge scores its log-likelihood. The
    /// fit at the power found is the fit reported, with its own `converged`.
    WeightPowerSearchFailures {
        /// Search fits that failed.
        failed: u32,
        /// Search fits with a finite log-likelihood that did not converge.
        not_converged: u32,
    },
    /// The fitted weight power ([`WeightPower::Estimate`]) is outside [0, 1]
    /// (0: the weights carry no variance information; 1: the weights as
    /// claimed). With few rows or a weak fit this happens by chance and is not
    /// by itself a problem. Not raised alongside
    /// [`Note::WeightPowerImplausible`] or [`Note::WeightPowerAtBound`].
    WeightPowerOutsideUnit {
        /// The power the fit used, [`Fit::weight_power`].
        delta: f64,
    },
    /// The fitted weight power ([`WeightPower::Estimate`]) is outside
    /// [`WEIGHT_POWER_SIMULATED`] and the fitted variances differ from the
    /// claimed weights by more than 10 times across the data:
    /// `s^|δ − 1| > 10`, `s` the largest weight over the smallest.
    WeightPowerImplausible {
        /// The power the fit used, [`Fit::weight_power`].
        delta: f64,
        /// `s^|δ − 1|`; `inf` past the double range.
        factor: f64,
    },
    /// [`SeType::Robust`] on a fit with a random slope. The robust standard
    /// error's arithmetic is checked there, but in simulations with 10 groups
    /// its Type I error ran at .056 to .066, with cells above 7.5%; the model
    /// standard error ([`SeType::Model`]) is the default on such a fit.
    RobustWithRandomSlope,
    /// Robust standard errors left NaN, with their df and their rows and
    /// columns of [`Fit::vcov`], because one cluster alone identifies the
    /// coefficient: no other cluster carries information on it, so that
    /// cluster has no residual for it and the standard error would follow the
    /// weights, not the data. More groups are needed;
    /// [`SeType::Model`] assumes the groups share one variance.
    RobustSeUnidentified {
        /// Design-matrix column indices, into the `x` the caller passed.
        columns: Vec<u32>,
    },
    /// Robust df ([`Fit::coef_df`]) below 4, where the robust test's Type I
    /// error has no assurance (Pustejovsky and Tipton 2018, section 3.1).
    /// When such a predictor is constant within groups, [`SeType::Model`] is
    /// the better route.
    RobustDfSmall {
        /// Design-matrix column indices, into the `x` the caller passed.
        columns: Vec<u32>,
        /// Their df, in the order of `columns`.
        df: Vec<f64>,
    },
    /// In [`crate::Prediction::notes`], when the standard error is asked for
    /// on a fit with [`SeType::Robust`]: the prediction's standard error and
    /// interval are model-based, not robust, since its formula holds only
    /// with the model's covariance.
    PredictSeModelBased,
}

impl Diagnostics {
    /// The diagnostics a `Fit` built without a solver carrier can honestly fill:
    /// `boundary` back-derived from `singular`, nothing pinned, no notes, no
    /// aliased column. Its caller is [`Fit::empty`], which stands in for a
    /// search that never ran and passes `converged: false`. Every fitting route
    /// goes through `common::materialize_diagnostics` instead, which has a
    /// carrier to read.
    ///
    /// `pinned` is empty here; `Fit::empty` sets it from its `varcorr` through
    /// `common::pinned_flags`, which places one row per block.
    pub(crate) fn from_flags(converged: bool, singular: bool, p: usize) -> Self {
        Diagnostics {
            converged,
            singular,
            aliased: vec![false; p],
            boundary: if singular {
                Boundary::AtBoundary
            } else {
                Boundary::Interior
            },
            pinned: vec![],
            notes: vec![],
            n_grad: 0,
            outer_stop: None,
            outer_decrement: f64::NAN,
        }
    }
}

/// `q` from a vech length: `len == q(q+1)/2` inverted by the quadratic formula.
/// Single source for the varcorr-block width — [`Fit::stddev_corr`] and the
/// `pinned` reshape in `common::materialize_diagnostics` must agree on it, and
/// that agreement is what makes `pinned[g][i]` pair with `stddev_corr(g).0[i]`.
pub(crate) fn vech_q(len: usize) -> usize {
    (((1 + 8 * len) as f64).sqrt() as usize - 1) / 2
}

/// [`Fit::stddev_corr`] on one vech-packed block, whose length must be
/// `q(q+1)/2`. Column `c`'s entries start at vech offset `c*q - c*(c-1)/2`
/// (the running cursor `varcorr_block` builds: for `r >= c`,
/// `idx(r,c) = c*q - c*(c-1)/2 + (r-c)`); change together. `pub` only through
/// the unstable `orchestrate` re-export, for the ports' stored fits.
pub fn vech_stddev_corr(vech: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let q = vech_q(vech.len());
    // `c*(c-1)/2` written as `(c*c - c)/2` to avoid a `c - 1` underflow at
    // c=0 in usize arithmetic (c*c >= c holds for all c >= 0).
    let idx = |r: usize, c: usize| -> usize { c * q - (c * c - c) / 2 + (r - c) };
    let stddev: Vec<f64> = (0..q).map(|i| vech[idx(i, i)].sqrt()).collect();
    let mut corr = vec![vec![0.0; q]; q];
    #[allow(clippy::needless_range_loop)]
    for i in 0..q {
        corr[i][i] = 1.0;
    }
    for c in 0..q {
        for r in (c + 1)..q {
            let rho = vech[idx(r, c)] / (stddev[r] * stddev[c]);
            corr[r][c] = rho;
            corr[c][r] = rho;
        }
    }
    (stddev, corr)
}

/// Relative tolerance for [`Fit::has_negligible_component`]: a converged
/// diagonal stddev at or below this fraction of its fit's largest is reported
/// singular even when the optimizer's own stopping point (governed by
/// `PIN_THETA`, an absolute 1e-4 threshold on the internal θ scale — left
/// untouched, this is a reporting-only check) landed just short of an exact
/// pin. Measured on `pois_cross4_g3000p20_bal_nearzero`: glmm converges to
/// stddev 2.5e-4 against a ~0.44-scale sibling component (ratio ~5.7e-4)
/// while lme4's own optimizer pins the same component to exactly 0 and flags
/// `isSingular`; 1e-3 catches this and any tighter true pin with margin.
const SINGULAR_REL_TOL: f64 = 1e-3;

impl Fit {
    /// The fit with no estimates, for a search that never ran: every estimate
    /// NaN at its full length (`fitted` length `n`), `ranef_levels` filled,
    /// `pinned` all false, `converged` false. `groupings` is `None` on a
    /// fixed-only fit, which leaves every per-grouping field empty.
    fn empty(n: usize, p: usize, groupings: Option<&crate::lmm::LmmGroupings>) -> Fit {
        let varcorr = groupings.map_or_else(Vec::new, common::nan_varcorr);
        let theta = groupings.map_or_else(Vec::new, common::nan_theta);
        let mut diagnostics = Diagnostics::from_flags(false, false, p);
        diagnostics.pinned = common::pinned_flags(0, &varcorr);
        Fit {
            beta: vec![f64::NAN; p],
            se: vec![f64::NAN; p],
            vcov: common::nan_vcov(p),
            tau2: theta.clone(),
            dispersion: f64::NAN,
            diagnostics,
            varcorr,
            stddev_se: theta,
            n_eval: 0,
            #[cfg(feature = "counters")]
            counters: crate::counters::EvalCounters::new(),
            deviance: f64::NAN,
            loglik: f64::NAN,
            df: 0,
            reml: false,
            weight_power: f64::NAN,
            weight_power_estimated: false,
            se_type: SeType::Model,
            model_vcov: Vec::new(),
            robust: None,
            nobs: n,
            objective: Objective::MaximumLikelihood,
            fitted: vec![f64::NAN; n],
            ranef: groupings.map_or_else(Vec::new, common::nan_ranef),
            ranef_levels: groupings.map_or_else(Vec::new, common::ranef_level_counts),
            condvar: vec![],
            condvar_x: vec![],
            condvar_cross: None,
            coef_df: vec![f64::NAN; p],
            satterthwaite: None,
            design_summary: None,
            null_basis: Vec::new(),
        }
    }

    /// Whether the optimizer reached its convergence criterion — the most-read
    /// field in the API, forwarded so that moving it behind `diagnostics` costs
    /// its callers a `()` and nothing else. See [`Diagnostics::converged`].
    pub fn converged(&self) -> bool {
        self.diagnostics.converged
    }

    /// Whether the fit is singular, by glmm's own rule. Forwarder — see
    /// [`Diagnostics::singular`].
    pub fn singular(&self) -> bool {
        self.diagnostics.singular
    }

    /// The length-`p` rank-deficiency mask. Forwarder — see
    /// [`Diagnostics::aliased`].
    pub fn aliased(&self) -> &[bool] {
        &self.diagnostics.aliased
    }

    /// Fixed-effect estimates, length p.
    pub fn beta(&self) -> &[f64] {
        &self.beta
    }

    /// Standard errors: `se[j] = sqrt(Var(β̂_j))` for target predictors,
    /// NaN for non-targets. Length p. Model-based, or on a fit with
    /// [`Fit::se_type`] [`SeType::Robust`] the robust (CR2 or HC2) ones, NaN
    /// where one cluster alone identifies the coefficient
    /// ([`Note::RobustSeUnidentified`]).
    pub fn se(&self) -> &[f64] {
        &self.se
    }

    /// Fixed-effect covariance `Cov(β̂)` of the fit's [`Fit::se_type`]: a full
    /// symmetric **p×p** matrix,
    /// row-major with both triangles filled (entry `(i, j)` at `vcov()[i * p + j]`,
    /// `p = beta().len()`), NOT vech-packed: unlike the random-effect
    /// covariances, which come one variable-sized block per grouping
    /// ([`Fit::varcorr()`]), this is a single square block and every consumer
    /// (R's `vcov()`/`confint`, `multcomp::glht`, any hand-built Wald contrast)
    /// indexes it directly. `se` is its diagonal:
    /// `se[j] == vcov()[j * p + j].sqrt()` wherever both are finite.
    ///
    /// Finite exactly where `se` is — the off-diagonal entry `(i, j)` is finite
    /// iff `se[i]` and `se[j]` both are. So a [`SeFor::Only`]
    /// subset leaves everything outside the target block NaN (there is no
    /// covariance to report for a coefficient whose variance was never
    /// computed), and a non-converged fit is all-NaN — except an LMM or GLMM
    /// fit that hit `MaxFunReached`: that budget-exhausted exit reports its
    /// finite covariance at the capped endpoint here instead. A GLM or GLMM fit
    /// under [`Note::SeUnavailable`] is all-NaN here and in `se` whatever its
    /// `converged`.
    ///
    /// Sources, by path: OLS/GLM/LMM invert the same Cholesky factor `se`'s
    /// forward solve already walks; GLMM `WaldSe::Hessian` takes the β block of
    /// the joint (θ,β) Hessian covariance (exact on every shape
    /// `derivative::supports_shape` accepts, finite-difference elsewhere),
    /// and `WaldSe::Rx` the p×p Schur
    /// inverse — both already formed in full and previously discarded down to a
    /// diagonal. Unlike `stddev_se`, this is populated on the Hessian's RX
    /// fallback too (that fallback inverts a full p×p covariance; only a
    /// double failure, where the Hessian AND the fallback both fail, leaves it
    /// NaN, under [`Note::SeUnavailable`]).
    pub fn vcov(&self) -> &[f64] {
        &self.vcov
    }

    /// [`Fit::vcov()`] as `p` rows of `p`, for the consumers that take a
    /// `&[Vec<f64>]`; each calls it once per test, predict call or port fit,
    /// not per draw.
    pub(crate) fn vcov_rows(&self) -> Vec<Vec<f64>> {
        let p = self.beta.len();
        debug_assert_eq!(self.vcov.len(), p * p, "vcov is not p × p");
        if p == 0 {
            return Vec::new();
        }
        self.vcov.chunks_exact(p).map(<[f64]>::to_vec).collect()
    }

    /// The model covariance as `p` rows of `p`: [`Fit::vcov_rows`] on a model
    /// fit, the private `model_vcov` on a robust one that has it, which
    /// covers every retained column whatever [`FitOptions::se_for`] says.
    pub(crate) fn model_vcov_rows(&self) -> Vec<Vec<f64>> {
        let p = self.beta.len();
        if self.model_vcov.is_empty() || p == 0 {
            return self.vcov_rows();
        }
        self.model_vcov
            .chunks_exact(p)
            .map(<[f64]>::to_vec)
            .collect()
    }

    /// The number of random-effect groupings the fit reports, declaration
    /// order (primary, then each extra): the valid `g` of every per-grouping
    /// accessor is below it. 0 on a fixed-effects fit (OLS, GLM) only.
    pub fn n_groupings(&self) -> usize {
        self.varcorr.len()
    }

    /// The number of random effects `q` of grouping `g` as reported: the
    /// intercept (when the block has one) and the slopes. A block without a
    /// random intercept (`(0 + x | g)`) counts its slopes only; the hidden
    /// intercept column the kernel keeps there is not counted.
    ///
    /// # Panics
    ///
    /// If `g >= n_groupings()`.
    pub fn n_re(&self, g: usize) -> usize {
        vech_q(self.varcorr[g].len())
    }

    /// Grouping `g`'s random-effect covariance `D̂ = σ̂²·Λ̂Λ̂'`, the full `q × q`
    /// matrix row-major with both triangles filled (entry `(r, c)` at
    /// `varcorr(g)[r * q + c]`, `q = n_re(g)`), the layout of [`Fit::vcov()`].
    /// Groupings are in declaration order (primary, then each extra), `g` below
    /// [`Fit::n_groupings()`]; lme4's `VarCorr(m)[[g]]`. The standard deviations
    /// and correlations of the same block are [`Fit::stddev_corr`]. σ̂² is the
    /// residual scale for an LMM and ≡ 1 on every GLMM family: a Gamma GLMM carries φ in its
    /// PIRLS weights, so θ̂ is already in the response's absolute units, as
    /// glmmTMB's `VarCorr` reports it (lme4's reports `σ·θ̂` instead). Validated against lme4
    /// `VarCorr` (`validation/goldens/sleepstudy_lmm.json`) and glmmTMB's on Gamma
    /// (`validation/goldens/sim_gamma_glmm_tmb.json`). OLS/GLM fits have no
    /// groupings (no random effects). All NaN on a fit with no finite
    /// endpoint.
    ///
    /// A block without a random intercept (`(0 + x | g)`,
    /// [`crate::ReTerm::no_intercept`]) reports its slopes only: the kernel
    /// keeps an intercept column there at zero variance, and that column is
    /// dropped from this block and from [`Fit::stddev_se`] and
    /// [`Fit::ranef()`]. The covariance between two terms of one grouping
    /// (`(x || g)`) is exactly `0.0`.
    ///
    /// # Panics
    ///
    /// If `g >= n_groupings()`.
    pub fn varcorr(&self, g: usize) -> Vec<f64> {
        let vech = &self.varcorr[g];
        let q = vech_q(vech.len());
        let mut d = vec![0.0; q * q];
        // The vech walks each column's lower triangle in turn, `(c, c)` first.
        let mut k = 0;
        for c in 0..q {
            for r in c..q {
                d[r * q + c] = vech[k];
                d[c * q + r] = vech[k];
                k += 1;
            }
        }
        d
    }

    /// Grouping `g`'s SEs of its random-effect standard deviations, length
    /// `q = n_re(g)`, aligned with `stddev_corr(g).0`: entry `k` is the SE of
    /// row k's stddev `√D̂_kk` (the stddev [`Fit::stddev_corr`] reports).
    /// Populated ONLY on a converged GLMM `WaldSe::Hessian` fit, from the θ block
    /// of the joint (θ,β) Hessian covariance that `joint_hessian_cov` inverts and
    /// otherwise discards (exact Hessian on every shape
    /// `derivative::supports_shape` accepts, finite-difference elsewhere).
    /// NaN under `WaldSe::Rx`, on the Hessian RX fallback, and
    /// on a Gaussian LMM (no Hessian machinery). The joint Hessian is taken in the
    /// fit's internal coordinates and the SE is the delta method's on them
    /// (the stddev's square is a polynomial in the internal θ). A stddev of
    /// exactly 0 has no derivative and reports NaN, except where it reads a
    /// single θ coordinate (a SCALAR grouping, or the intercept of a block
    /// whose slopes are not centered): there the SE is that coordinate's,
    /// as at every other value.
    ///
    /// Scale caveat for dispersion families (Gamma): this SE stays on the
    /// **θ scale** — lme4's θ-Hessian convention — while `varcorr` carries
    /// the σ̂² factor. Deliberately NOT multiplied by σ̂: the joint Hessian does
    /// not carry cov(σ̂, θ̂), so a σ̂-rescaled SE would be a delta-method value
    /// that matches no oracle. For the φ≡1 families the two scales coincide and
    /// there is no split. (The validation suite skips `sd_se` gating for
    /// dispersion families accordingly.)
    ///
    /// # Panics
    ///
    /// If `g >= n_groupings()`.
    pub fn stddev_se(&self, g: usize) -> Vec<f64> {
        let q = self.n_re(g);
        let start: usize = self.varcorr[..g].iter().map(Vec::len).sum();
        // Diagonal slot `(k, k)` of a column-major vech: `k*q - k*(k-1)/2`.
        (0..q)
            .map(|k| self.stddev_se[start + k * q - (k * k - k) / 2])
            .collect()
    }

    /// Random-effect conditional modes `b̂ = Λ̂û` on the natural (link) scale —
    /// lme4's `ranef()` values, for grouping `g` (declaration order: primary,
    /// then each extra — the order of [`Fit::varcorr()`]): a `levels × q`
    /// block, row-major, level `l`'s `q = n_re(g)` values (intercept, then
    /// slopes in declaration order; slopes only on a block without a random
    /// intercept, see [`Fit::varcorr()`]) at `[l·q .. (l+1)·q]`, with `levels =
    /// ranef_levels()[g]`. `η̂ = Xβ̂ + Zb̂` reproduces the linear predictor
    /// behind [`Fit::fitted()`]. For a nested grouping the block spans
    /// `n_parents·n_per_parent` slots (child id = parent·n_per_parent +
    /// within), padded with zero modes for parents with fewer observed
    /// children: the padded slots are part of the block. All NaN on a fit that
    /// did not converge.
    ///
    /// This is the numbers only. To get them LABELLED — which level each row
    /// belongs to, with the padded slots dropped — fit through the formula
    /// door and use [`crate::formula::Model::ranef`], which reads the block
    /// layout in the one place it is interpreted. Do not re-derive the slicing
    /// in a consumer: which layout a grouping lands in is a data-dependent
    /// speed decision, so a consumer that inferred it would be wrong on a
    /// dataset nobody tested.
    ///
    /// The Gaussian LMM paths never form these during the fit (the profiled
    /// criterion, REML or ML, does not need them) and recover them once at θ̂ by
    /// back-substitution afterwards, which is why they are here without the
    /// per-evaluation cost of forming them.
    ///
    /// # Panics
    ///
    /// If `g >= n_groupings()`.
    pub fn ranef(&self, g: usize) -> &[f64] {
        let q = self.n_re(g);
        let start: usize = (0..g).map(|h| self.ranef_levels[h] * self.n_re(h)).sum();
        &self.ranef[start..start + self.ranef_levels[g] * q]
    }

    /// Level count per grouping for slicing [`Fit::ranef()`], declaration order
    /// (primary, then each extra): `ranef(g).len() == ranef_levels()[g] * n_re(g)`.
    /// Filled on every fit with random effects, including one that did not
    /// converge; empty on a fit without random effects.
    pub fn ranef_levels(&self) -> &[usize] {
        &self.ranef_levels
    }

    /// Conditional covariance of each level's random effects given the data,
    /// at the fitted parameters, for grouping `g`: lme4's
    /// `ranef(condVar = TRUE)` (`postVar`), on the scale of [`Fit::ranef()`].
    /// The layout of `ranef(g)` with each level's `q` values replaced by the
    /// vech of its `q × q` block (column-major lower triangle:
    /// `(0,0),(1,0),…,(q-1,0),(1,1),…,(q-1,q-1)`), so the block holds
    /// `ranef_levels()[g]·q(q+1)/2` values, padded nested slots included; the
    /// conditional SDs are the square roots of the diagonal entries. A block
    /// without a random intercept
    /// reports its slopes only, as `ranef` does, and a padded nested slot
    /// holds the prior covariance (it has no rows).
    ///
    /// On a Gaussian LMM the block is `σ̂²·Λ̂(Λ̂ᵀZᵀWZΛ̂ + I)⁻¹Λ̂ᵀ` restricted to
    /// the level (`W` the prior weights); on a GLMM the same with `σ̂² = 1`
    /// and `W` the observed curvature `−∂²ℓᵢ/∂ηᵢ²` at the conditional modes
    /// (the Laplace approximation; on a Gamma GLMM it carries `1/φ̂`). Both
    /// hold β̂ fixed, as lme4's do.
    ///
    /// Formed once when the `Fit` is assembled, from the factors at the
    /// optimum, and only when [`FitOptions::condvar`] asks for it. Empty on a
    /// fit that did not ask, on a fit that did not converge (its `ranef` is
    /// NaN), and on the layouts where it is not computed: a Gaussian LMM on the sparse route (random slopes on two or
    /// more groupings, a lone slope grouping written after others that leave
    /// more than `MAX_CROSSED_LEVELS` crossed levels when it is moved first or
    /// whose slope block is wider than `MAX_PRIMARY_Q`,
    /// over the dense envelope or more than `MAX_CROSSED_LEVELS` crossed levels) and a GLMM on the packed layout,
    /// which would need a selected inverse of the whole system.
    ///
    /// # Panics
    ///
    /// If `g >= n_groupings()`.
    pub fn condvar(&self, g: usize) -> &[f64] {
        let width = |h: usize| {
            let q = self.n_re(h);
            q * (q + 1) / 2
        };
        let w = width(g);
        if self.condvar.is_empty() {
            return &[];
        }
        let start: usize = (0..g).map(|h| self.ranef_levels[h] * width(h)).sum();
        &self.condvar[start..start + self.ranef_levels[g] * w]
    }

    /// Degrees of freedom of each coefficient `β̂_j`, length p: lmerTest's
    /// Satterthwaite df of the contrast `e_j` on a Gaussian LMM, and the
    /// residual df `n − p` (`p` the retained columns) on OLS and on a Gamma or
    /// inverse Gaussian GLM with φ estimated, whatever
    /// [`FitOptions::satterthwaite`] says. Filled on the target columns
    /// ([`FitOptions::se_for`]) of a Gaussian mixed fit whose pass ran
    /// and of a converged OLS or dispersion GLM fit; NaN on non-target and
    /// aliased columns (as `se`), on an OLS or dispersion GLM fit that did not
    /// converge (as `se`), on every other family (binomial, Poisson, negative
    /// binomial, every GLMM, a Gamma or inverse Gaussian GLM with φ held), when
    /// the pass was switched off, and when it failed
    /// ([`Note::SatterthwaiteCurvature`] with `nonfinite` set). On a robust fit
    /// ([`SeType::Robust`]) the Pustejovsky–Tipton df of the CR2 standard
    /// error, or the Bell–McCaffrey df of HC2 without a grouping factor,
    /// whatever [`FitOptions::satterthwaite`] says, NaN where `se` is. Not
    /// `df`, which is the AIC parameter count.
    pub fn coef_df(&self) -> &[f64] {
        &self.coef_df
    }

    /// Fitted means μ̂ per row (length `n`): the conditional means through the
    /// inverse link — `g⁻¹(Xβ̂ + Zb̂)` for mixed fits (lme4 `fitted()`),
    /// `g⁻¹(Xβ̂)` for fixed-only. NaN on a fit that did not converge. Always on the
    /// original `y` scale, including on the Gaussian LMM paths — see
    /// `fit::common::lmm_fitted` for how the offset is restored there.
    pub fn fitted(&self) -> &[f64] {
        &self.fitted
    }

    /// The actual log-likelihood at the fitted parameters — `deviance` with its
    /// dropped data-only constants restored, on the `logLik()` scale (R/lme4):
    ///
    /// - **OLS/GLM** — the standard closed forms (R `logLik.lm`/`logLik.glm`,
    ///   `MASS::glm.nb`), including under prior weights, EXCEPT Gamma and
    ///   inverse-Gaussian: both report the precision log-likelihood (row `i`
    ///   has variance `φ·V(μᵢ)/wᵢ`) rather than R's `logLik.glm`/`Gamma()$aic`
    ///   case-weight one (`−½·(D/φ̂ + gamma_dispersion_term)` at the maximised
    ///   φ̂ for Gamma, `family::inv_gaussian_aic`'s precision form for
    ///   inverse-Gaussian). The two agree only at unit weights; a weighted
    ///   Gamma GLM's `loglik` is the maximised precision value, not R's
    ///   `logLik(fg)` (measured: −47.27 vs −118.21 on
    ///   `fit_glm_gamma_weighted_matches_r`'s data). On a fit that reproduces
    ///   the data exactly the likelihood rises without bound as φ → 0, so a
    ///   deviance that rounds to zero gives `+∞`. An OLS fit made with
    ///   [`FitOptions::reml`] `= Some(true)` reports the REML log-likelihood
    ///   instead, R's `logLik(lm, REML = TRUE)`: `−½(d + (n−p)·(1 + ln 2π))`
    ///   with `d = ln|XᵀWX| + (n−p)·ln(RSS_w/(n−p)) − Σ ln wᵢ` on the caller's
    ///   X (`RSS_w` the weighted residual sum of squares). `d` is the REML
    ///   `deviance` an LMM with the same fixed part reports when its variance
    ///   is pinned at zero, so the two are on one scale (`deviance` itself
    ///   stays NaN on OLS).
    /// - **LMM**: the fit's criterion ([`Fit::reml`]). Under REML (the
    ///   default) the **REML criterion** `−REMLcrit/2`,
    ///   `−½(deviance + (n−p)·(1 + ln 2π))`, matching `lme4::logLik` on a
    ///   `REML=TRUE` fit. REML criteria are comparable only between models with
    ///   IDENTICAL fixed effects: an AIC/LRT across different fixed parts is
    ///   meaningless, so check [`Fit::reml`] before comparing. Under ML
    ///   ([`FitOptions::reml`] `= Some(false)`) the ML log-likelihood
    ///   `−½(deviance + n·(1 + ln 2π))`, matching `lme4::logLik` on a
    ///   `REML=FALSE` fit.
    /// - **GLMM** — the marginal Laplace/AGQ log-likelihood:
    ///   `−½·deviance + saturated_loglik` (binomial/Poisson, see
    ///   `family::saturated_loglik`), `−½·deviance` alone for Gamma and NB
    ///   (Gamma: `deviance` carries the whole precision log-density through
    ///   `family::gamma_dispersion_term` at the ML φ̂ (`Fit::dispersion` on a
    ///   GLMM, unlike the GLM's Pearson moment); NB: `deviance` already
    ///   carries the θ̂-dependent saturated term the outer θ search adds — see
    ///   `fit::common::glmm_loglik`). Matches `lme4::logLik` on the same fit on
    ///   the canonical binomial and Poisson links, including the
    ///   aggregated-binomial `cbind(s, m−s)` form under `weights=`. On
    ///   non-canonical links, NB and Gamma it matches glmmTMB's instead: lme4
    ///   builds its Laplace log-determinant from the expected (Fisher) weight
    ///   there, and on Gamma fits at its `pwrss/n` plug-in scale, so its
    ///   logLik is a different function of the parameters.
    ///
    /// NaN wherever `deviance`'s failure modes apply (non-converged/degenerate
    /// fits); finite on an LMM or GLMM `MaxFunReached` endpoint, like
    /// `deviance`.
    /// `AIC = 2·df − 2·loglik`, `BIC = df·ln(n) − 2·loglik` with [`Fit::df`] —
    /// on a GLMM budget-exhausted exit `df` is 0 (see [`Fit::df`]) and this
    /// identity does not apply there, even though `loglik` itself is finite.
    pub fn loglik(&self) -> f64 {
        self.loglik
    }

    /// Minimized optimizer criterion at the accepted point. LMM: the profiled
    /// deviance of the fit's criterion ([`Fit::reml`]) as computed by
    /// `reml_deviance`, on the caller's design. A REML fit's equals lme4's
    /// REMLcrit minus the data-independent constant ν·(1 + ln 2π), ν = n − p
    /// (validated against the frozen lme4 sleepstudy reference); an ML fit's
    /// equals lme4's ML `deviance()` minus the same constant with ν = n. GLMM: the
    /// marginal Laplace deviance `d(y,ũ) + ‖ũ‖² + log|A|` (see
    /// `GlmmFit::deviance`), which differs from −2·logLik by a data-only
    /// saturated constant on binomial/Poisson fits and equals −2·logLik
    /// exactly on Gamma and NB fits (see `fit::common::glmm_loglik`). NaN for
    /// OLS/GLM and on optimizer/numerical failure (GLMM non-convergence
    /// surfaces as +∞ internally; mapped to NaN here). An LMM or GLMM fit
    /// that hits `MaxFunReached` still reports the finite endpoint deviance
    /// here with `converged == false` — the plateau policy: a
    /// `MaxFunReached` cap-out reports its finite endpoint with
    /// `converged == false` rather than NaN-filling.
    pub fn deviance(&self) -> f64 {
        self.deviance
    }

    /// Parameters counted for AIC/BIC: retained fixed effects (`p` minus
    /// aliased columns, matching lme4's NA-coefficient handling) + the RE θ
    /// entries the fit estimates (not those a grouping's terms hold at zero,
    /// as in `(x || g)`) + 1 if the family estimates a dispersion/scale (Gaussian
    /// σ², Gamma φ unless held fixed via [`FitOptions::dispersion`], NB θ),
    /// and 1 more for an estimated [`Fit::weight_power`].
    /// 0 on degenerate NaN-fill paths, and also 0 on a GLMM fit that hits
    /// `MaxFunReached` (gated on `converged`, unlike `loglik`/`deviance`) — the
    /// AIC/BIC identity in [`Fit::loglik`] does not apply at that endpoint. An
    /// LMM `MaxFunReached` endpoint reports its usual nonzero `df` instead.
    pub fn df(&self) -> usize {
        self.df
    }

    /// The number of rows the fit was given, `n`, on every route and whatever
    /// the outcome: a non-converged or degenerate fit reports it too (where
    /// `fitted` is NaN), and the rank-deficient salvage reports its reduced
    /// fit's, which is the same `n`.
    pub fn nobs(&self) -> usize {
        self.nobs
    }

    /// Estimated dispersion: `φ` for Gamma (the Pearson moment on a GLM,
    /// `summary(glm)`'s; the maximum-likelihood estimate on a GLMM) and
    /// inverse-Gaussian (the Pearson moment — GLM only, no mixed route), the
    /// estimated shape `θ` for negative-binomial, the residual variance `σ̂²`
    /// for Gaussian — `RSS/(n−p)` for OLS (raw-row df, matching R
    /// `summary.lm`'s `sigma²`; the same `sigma_sq` that scales `se`/`vcov`)
    /// and the criterion's `pwrss/ν` for LMM, `pwrss/(n−p)` under REML and
    /// `pwrss/n` under ML (matching lme4 `sigma()²` on the same fit; oracle:
    /// `validation/goldens/sleepstudy_lmm.json` `sigma`, asserted in
    /// `fit_sleepstudy_slope_varcorr_matches_lme4`) — and `1.0` for
    /// binomial/Poisson (where dispersion is fixed by the family, not
    /// estimated). NaN on a fit reporting `converged == false`, except an LMM
    /// or GLMM fit that hit `MaxFunReached`, which reports it at the capped
    /// endpoint, as `vcov` does. This includes the fixed families' `1.0` and a
    /// φ held fixed through [`FitOptions::dispersion`].
    pub fn dispersion(&self) -> f64 {
        self.dispersion
    }

    /// `true` iff `loglik` is a REML criterion rather than an ML
    /// log-likelihood: a Gaussian mixed fit under REML (the default; a fit
    /// made with [`FitOptions::reml`] `= Some(false)` is by ML and reports
    /// `false`), read off the criterion the fit minimized, and an OLS fit made
    /// with [`FitOptions::reml`] `= Some(true)`.
    /// Model comparisons (AIC/LRT) across fits with different fixed effects are
    /// invalid when this is set — mirror of lme4's REML-fit `anova` warning.
    pub fn reml(&self) -> bool {
        self.reml
    }

    /// The power `v` the fit put on the caller's weights
    /// ([`FitOptions::weight_power`]): row `i` has variance `σ²·wᵢ^(−v)`, so
    /// the fit is the weighted fit on `wᵢ^v`. The estimate δ̂ on a fit that
    /// estimated it, 1 on a fit with the weights as given, NaN on a fit
    /// without weights.
    pub fn weight_power(&self) -> f64 {
        self.weight_power
    }

    /// The standard error behind [`Fit::se`], [`Fit::vcov`] and
    /// [`Fit::coef_df`]: [`FitOptions::se_type`] as the fit resolved it.
    pub fn se_type(&self) -> SeType {
        self.se_type
    }

    /// The objective the fit maximised, the label a likelihood-ratio
    /// comparison carries: [`Objective::LeastSquares`] on OLS (also on an OLS
    /// fit made with [`FitOptions::reml`] `= Some(true)`, whose `loglik` is
    /// the REML value), [`Objective::MaximumLikelihood`] on a GLM,
    /// [`Objective::Reml`] or [`Objective::MaximumLikelihood`] on a Gaussian
    /// mixed fit from its criterion ([`Fit::reml`]), and on a GLMM
    /// [`Objective::Laplace`] at [`FitOptions::nagq`] `= 1` or
    /// [`Objective::Agq`] with the node count `nagq` above it. Set on every
    /// fit, a non-converged or degenerate one included. Whether `loglik` is a
    /// REML criterion is [`Fit::reml`]'s answer; this field adds the label and
    /// the node count.
    pub fn objective(&self) -> Objective {
        self.objective
    }

    /// Objective value evaluations of the outer search over θ (LMM) or the
    /// joint `[θ|β]` (GLMM): BOBYQA's and BFGS's together, every stage and
    /// fallback included. A BFGS gradient reuses its point's value evaluation
    /// and is not counted here (see [`Diagnostics::n_grad`]). 0 where no outer
    /// search runs (OLS/GLM closed-form or IRLS paths). Deterministic and
    /// clock-independent; the optimizer-grid campaign's primary metric.
    pub fn n_eval(&self) -> usize {
        self.n_eval
    }

    /// Everything the fit reports about itself: convergence, singularity, the
    /// aliased-column mask, the θ boundary state, which variance components
    /// were pinned there, and any [`Note`]s the solver raised. Single storage
    /// location — [`Fit::converged`], [`Fit::singular`] and [`Fit::aliased`]
    /// are one-hop forwarders onto it, not copies.
    pub fn diagnostics(&self) -> &Diagnostics {
        &self.diagnostics
    }

    /// Dev-only optimizer evaluation counters — the stage split, the shrink
    /// phase, the PIRLS-iteration histogram, the AGQ node cost and the searches
    /// run after a pin. Present only under the off-by-default `counters`
    /// feature, which is NOT
    /// semver-covered; zeros on every route that runs no outer search (OLS,
    /// GLM).
    #[cfg(feature = "counters")]
    pub fn counters(&self) -> &crate::counters::EvalCounters {
        &self.counters
    }

    /// Reduce grouping `group_idx`'s `q×q` covariance block ([`Fit::varcorr()`])
    /// into per-dimension standard deviations and a full symmetric `q×q`
    /// correlation matrix, mirroring lme4's `VarCorr` stddev/corr split. It
    /// reads the stored vech (column-major lower-triangular): `q` is recovered
    /// from `vech.len() = q(q+1)/2` via the quadratic formula.
    ///
    /// # Panics
    ///
    /// If `group_idx >= n_groupings()`.
    pub fn stddev_corr(&self, group_idx: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let vech = &self.varcorr[group_idx];
        debug_assert_eq!(
            vech_q(vech.len()) * (vech_q(vech.len()) + 1) / 2,
            vech.len(),
            "varcorr[{group_idx}] is not a valid vech"
        );
        vech_stddev_corr(vech)
    }

    /// Post-hoc singular check beyond the optimizer's exact-pin decision
    /// ([`Diagnostics::boundary`], unchanged by this — see
    /// [`SINGULAR_REL_TOL`]).
    /// Changes no returned estimate. `false` when `varcorr` is empty (a fit
    /// without random effects).
    ///
    /// The comparison runs on the fit's internal standard deviations
    /// `kernel_sd` (`LmmGroupings::kernel_stddevs`): the row norms of the
    /// kernel's Λ, one block per grouping, primary first. σ cancels in the
    /// ratio, so they stand in for `σ·‖row‖`. The reported standard deviations
    /// are not commensurate with each other: an intercept's is in response units
    /// and a slope's is in response per covariate unit, so their raw ratio moves
    /// when the covariate is re-expressed and the verdict would follow it. In
    /// the kernel's coordinates every component is "how much variation this
    /// term contributes" on the standardized design, and the check means what
    /// it says. On a predictor whose mean is far from zero the intercept's is
    /// therefore taken at the mean, not at x = 0.
    pub(crate) fn has_negligible_component(&self, kernel_sd: &[Vec<f64>]) -> bool {
        if self.varcorr.is_empty() {
            return false;
        }
        let max_sd = kernel_sd.iter().flatten().cloned().fold(0.0_f64, f64::max);
        if max_sd <= 0.0 {
            return false;
        }
        kernel_sd
            .iter()
            .flatten()
            .any(|&s| s <= SINGULAR_REL_TOL * max_sd)
    }
}

/// Options for `fit`. Carries the safe, defaulted method knobs: unlike
/// a warm start these do not silently move the estimate based on a caller's guess.
///
/// `FitOptions` is `#[non_exhaustive]`: outside this crate a struct literal,
/// including `..` update syntax, does not compile.
///
/// ```compile_fail
/// let o = glmm::FitOptions { se_for: glmm::SeFor::Only(vec![0]), ..Default::default() };
/// ```
///
/// Start from [`FitOptions::default()`] (or a clone of existing options) and
/// assign fields instead:
///
/// ```rust
/// let mut o = glmm::FitOptions::default();
/// o.se_for = glmm::SeFor::Only(vec![0]);
/// ```
// Non-exhaustive so a new knob is an additive change, not a break.
#[non_exhaustive]
#[derive(Clone)]
pub struct FitOptions {
    /// Which coefficients get a standard error. Default [`SeFor::All`].
    pub se_for: SeFor,
    /// Wald-SE denominator (relocated from `ModelSpec`). Default `Hessian`.
    pub wald_se: WaldSe,
    /// Adaptive Gauss–Hermite node count (relocated from `ModelSpec`). Default 1
    /// (= Laplace). Must be odd and in `1..=MAX_NAGQ`; `>1` is honored on a
    /// binomial/Poisson/negative-binomial/Gamma GLMM with a single grouping factor and `q_p ≤ 3` random
    /// effects per group (scalar intercept → `agq::agq_deviance`, vector RE →
    /// `agq::agq_deviance_vec`, a `k^q_p` product grid). `q_p ≥ 4` is refused
    /// with [`FitError::Unsupported`] (a temporary cost/oracle boundary), as
    /// are the other ineligible non-Gaussian mixed shapes. `nagq > 1` is
    /// ignored on Gaussian models and on models without random effects: the
    /// fit runs at `nagq = 1` with no note attached, and the ports warn.
    pub nagq: u8,
    /// Gamma/inverse-Gaussian dispersion directive. Default
    /// [`Dispersion::Default`]: estimate φ (the Pearson moment,
    /// `Σwᵢrᵢ²/(n−p)`, on a GLM; maximum likelihood on a GLMM);
    /// [`Dispersion::Fixed`]`(v)` holds φ at `v`. Every other family ignores
    /// it: binomial and Poisson have φ ≡ 1, and the Gaussian residual
    /// variance σ² and the negative-binomial shape θ are always estimated. A
    /// `Fixed` value that is not finite and `> 0` is refused on every family
    /// ([`FitError::OptionValue`]).
    pub dispersion: Dispersion,
    /// **Experimental.** Opt into the parallel inner-fit kernels (cluster-outer
    /// AGQ, per-`(i,j)` FD-Hessian grid) — live ONLY when built with the
    /// `parallel` Cargo feature on a non-`wasm32` target. The exclusion is
    /// compile-time, not a runtime check: serial and `wasm32` builds run the
    /// original sequential kernels byte-for-byte regardless of this flag, and
    /// never pull `rayon` in.
    /// (Cluster-outer AGQ is deliberately NOT offered serially: its per-cluster
    /// overhead regresses many-tiny-cluster shapes — measured +12–16% on
    /// observation-level REs — so it exists only as the rayon substrate.)
    ///
    /// Default `false`: parallel results are bit-identical to serial (tested),
    /// but the kernels are new and their perf envelope isn't characterized yet,
    /// so a fit uses them only on explicit opt-in. Batch callers driving many
    /// fits at once (e.g. MCPower's `loop_advanced` hot loop) should keep this
    /// `false` even after opting builds into `parallel` — that caller already
    /// owns the outer parallelism, and stacking a second parallel axis inside
    /// every fit adds per-split overhead (and, for AGQ, a real per-fit CSR
    /// preprocessing cost) for no benefit once the outer loop saturates the
    /// cores.
    ///
    /// Wired to the AGQ cluster-outer restructuring (`agq::agq_deviance`'s
    /// `cluster_rows` path): `false` runs scalar AGQ's original node-outer loop
    /// with no per-fit `ClusterRowIndex` build (vector AGQ is cluster-outer on
    /// every build, and a serial build gives it the index once per fit). Also
    /// gates the FD-Hessian grids (dense and sparse): `false` keeps the serial
    /// cell-by-cell loop. See the design doc above
    /// for the full rationale, including why nesting this under an outer `rayon`
    /// batch loop is safe (one shared work-stealing pool) in a way naive OS-thread
    /// or BLAS-thread nesting is not.
    pub parallel_inner: bool,
    /// Run the Satterthwaite pass on a Gaussian LMM fit ([`Fit::coef_df`],
    /// [`Fit::contrast_t`], [`Fit::contrast_f`]): one finite-difference pass
    /// at θ̂, over the profiled deviance (or over its analytic gradient where
    /// the fit's per-group gradient is armed), after the fit and without a
    /// refit.
    /// Default `true`. `false` skips it: `coef_df` stays NaN and the contrasts
    /// return [`crate::inference::contrast::NoDfReason::Disabled`]. No other
    /// family reads it, and no other field of the fit depends on it.
    pub satterthwaite: bool,
    /// Form the conditional variances of the random effects ([`Fit::condvar`])
    /// on a mixed model, which a conditional standard error or future bound
    /// from [`crate::predict`] reads. Default `false`: they cost a set of
    /// solves after the fit (on a GLMM, a re-solve of the mode first) that
    /// grows with the level count, up to as much as the fit itself on a
    /// crossed model with many levels, and a caller fitting in a loop rarely
    /// reads them. With `false`, `condvar` is empty and such a prediction
    /// returns [`FitError::Unsupported`]; no other field of the fit depends on
    /// it.
    pub condvar: bool,
    /// The estimator of a Gaussian mixed model: restricted maximum likelihood
    /// (REML) or maximum likelihood (ML). Default `None`, which resolves to
    /// the usual estimator for the model:
    ///
    /// | Model | `None` | `Some(true)` | `Some(false)` |
    /// |---|---|---|---|
    /// | Gaussian, random part | REML | REML | ML |
    /// | Gaussian, no random part (OLS) | ML log-likelihood | REML log-likelihood | ML log-likelihood |
    /// | any other family | ML | ML | ML |
    ///
    /// On OLS without weights, or with a fixed [`FitOptions::weight_power`],
    /// the value moves only the reported log-likelihood (see [`Fit::loglik`])
    /// and [`Fit::reml`]: β̂, the SEs and `σ̂² = RSS/(n − p)` are the same under
    /// all three. Where the default weight power is estimated (a Gaussian fit
    /// with weights), `None` and `Some(true)` estimate it by REML, with the
    /// REML log-likelihood, and `Some(false)` takes the weights as given
    /// (power 1), so the estimates differ; [`WeightPower::Estimate`] under
    /// `Some(false)` estimates the power by ML. It is read when the `Fit` is assembled,
    /// from the options handed to `FitView::into_fit`, and is not frozen on an
    /// OLS `FitWorkspace`, since the solve does not depend on it. On a GLM or
    /// GLMM (Laplace or AGQ) the fit is by maximum likelihood whatever the
    /// value, so `Some(true)` is ignored there, as [`FitOptions::dispersion`]
    /// is on the families with no dispersion. `Some(false)` off the Gaussian
    /// mixed model, and `Some(true)` on a GLM or GLMM, give a `Fit` bitwise
    /// identical to the `None` fit, with no note attached: the ports warn
    /// about an ignored value, the kernel does not. "Random part" means the
    /// model actually fitted: a grouping dropped for having one level leaves
    /// an OLS or GLM fit, and its row applies.
    pub reml: Option<bool>,
    /// The power on the caller's weights ([`Data::weights`]). Default `None`,
    /// which resolves to [`WeightPower::Estimate`] on a Gaussian fit with
    /// weights, by REML (`reml` `None` or `Some(true)`), with at most one
    /// grouping factor, at least 20 rows and weights not all equal (largest
    /// over smallest above `1 + 1e-12`), and to [`WeightPower::Fixed`]`(1.0)`,
    /// the weights as given, everywhere else.
    /// [`WeightPower::Estimate`] fits `δ` in `Var(εᵢ) = σ²·wᵢ^(−δ)` by
    /// golden section on the fit's log-likelihood (REML, or ML with
    /// `reml = Some(false)`) over `±max(2, min(30, ln(1e28)/ln s))`, `s` the
    /// largest weight over the smallest, narrowed to keep every `wᵢ^δ` within
    /// 1e±250, to width 1e-4, then fits at δ̂ with
    /// these options and `reml` `Some(true)` or `Some(false)`; δ̂ counts in
    /// [`Fit::df`], and the model standard error treats it as known. Named, it
    /// is refused on a model outside that shape ([`FitError::Unsupported`]) and
    /// without weights or with weights all equal ([`FitError::Weights`]), and
    /// it runs under ML when `reml = Some(false)`.
    /// [`WeightPower::Fixed`]`(v)` fits with working weights `wᵢ^v` and the
    /// model standard error: 0 ignores the weights, and a `v` other than 0
    /// and 1 has no simulation evidence behind it. Without weights every
    /// `wᵢ = 1`, so the value changes nothing (the ports warn). Refused: a
    /// `v` that is not finite ([`FitError::OptionValue`]), a `wᵢ^v` that is
    /// not finite and `> 0` ([`FitError::Weights`], naming the row), and
    /// `v ≠ 1` with weights on a family other than Gaussian, whose weights are
    /// the family's prior weights ([`FitError::Unsupported`]).
    /// [`Fit::weight_power`] reports the value used.
    pub weight_power: Option<WeightPower>,
    /// The standard error the fit reports. Default `None`, which resolves to
    /// [`SeType::Robust`] on a REML fit with [`WeightPower::Estimate`] (named
    /// or resolved) and either one grouping factor and no random slope (CR2)
    /// or no grouping factor (HC2), and to [`SeType::Model`] everywhere else.
    /// Named, [`SeType::Robust`] works on any Gaussian fit with at most one
    /// grouping factor, by REML when it has one or estimates the weight
    /// power, and raises [`Note::RobustWithRandomSlope`] on a random slope;
    /// it is refused on any other ([`FitError::Unsupported`]).
    /// [`Fit::se_type`] reports the value used.
    pub se_type: Option<SeType>,
}

/// The power on the caller's weights ([`FitOptions::weight_power`]): row `i`
/// has variance `σ²·wᵢ^(−v)`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightPower {
    /// Estimate the power from the data.
    Estimate,
    /// Fit with working weights `wᵢ^v`: 1 is the weights as given, 0 ignores
    /// them.
    Fixed(f64),
}

/// The standard error a fit reports ([`FitOptions::se_type`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeType {
    /// The model-based standard error.
    Model,
    /// The cluster-robust standard error: CR2 with the Pustejovsky–Tipton df
    /// on the fit's grouping factor, or HC2 with the Bell–McCaffrey df (one
    /// cluster per row) without one.
    Robust,
}

/// Which fixed-effect coefficients a fit computes standard errors for
/// ([`FitOptions::se_for`]). A coefficient without one has a NaN `se` slot and
/// a NaN row and column in `vcov`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum SeFor {
    /// Every coefficient.
    #[default]
    All,
    /// Only these columns of `x` (0-based). `Only(vec![])` computes none:
    /// every `se` and every `vcov` cell is NaN. On OLS, a GLM and an LMM it
    /// skips the per-coefficient variance solves; a GLMM still forms its
    /// covariance at the endpoint (under [`WaldSe::Hessian`], the joint
    /// Hessian).
    Only(Vec<u32>),
}

impl SeFor {
    /// The column indices this selects on a `p`-column design.
    pub(crate) fn columns(&self, p: usize) -> Cow<'_, [u32]> {
        match self {
            SeFor::All => Cow::Owned((0..p as u32).collect()),
            SeFor::Only(cols) => Cow::Borrowed(cols),
        }
    }
}

/// The dispersion φ a fit uses ([`FitOptions::dispersion`]).
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq)]
pub enum Dispersion {
    /// Estimate φ on Gamma and inverse-Gaussian (the Pearson moment on a GLM,
    /// maximum likelihood on a GLMM). Binomial and Poisson have φ ≡ 1; the
    /// Gaussian σ² and the negative-binomial θ are estimated whatever the
    /// value.
    #[default]
    Default,
    /// Hold φ at this value (finite, > 0) on Gamma and inverse-Gaussian; on
    /// inverse-Gaussian within `[1/MAX_DATA_ABS, MAX_DATA_ABS]`
    /// ([`MAX_DATA_ABS`](crate::limits::MAX_DATA_ABS)); on Gamma `v/s` within
    /// [`GAMMA_PHI_LO`](crate::limits::GAMMA_PHI_LO),
    /// [`GAMMA_PHI_HI`](crate::limits::GAMMA_PHI_HI), with `s` a power of two
    /// near the geometric mean of the weights (1 without weights). Ignored by
    /// every other family: binomial and Poisson (φ ≡ 1), Gaussian (σ²
    /// estimated) and negative binomial (θ estimated).
    Fixed(f64),
}

impl Dispersion {
    /// The held value, `None` when φ is estimated.
    pub(crate) fn held(&self) -> Option<f64> {
        match *self {
            Dispersion::Default => None,
            Dispersion::Fixed(v) => Some(v),
        }
    }
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            se_for: SeFor::All,
            wald_se: WaldSe::Hessian,
            nagq: 1,
            dispersion: Dispersion::Default,
            parallel_inner: false,
            satterthwaite: true,
            condvar: false,
            reml: None,
            weight_power: None,
            se_type: None,
        }
    }
}

/// The message of a [`FitError`], and the row it names when it names one.
///
/// It reads as the text: it derefs to `str`, compares equal to a `&str` and
/// displays as the text, so `m.contains("weights")` and `m == "..."` work on
/// it. [`row`](Self::row) is the 0-based row of the data the check failed at,
/// and [`text_with_row`](Self::text_with_row) gives the text naming another
/// number for that row (a port that counts rows from 1, or that dropped rows
/// before the fit), so no caller parses the text.
///
/// Two messages are equal when their texts are: the row is a reading of the
/// text, so a message built from a `String` equals the refusal with that text.
#[derive(Clone)]
pub struct ErrorMessage {
    text: String,
    row: Option<RowAt>,
}

/// Where a row sits in an [`ErrorMessage`]: the 0-based row and the byte
/// range of its digits in the text.
#[derive(Clone, Copy)]
struct RowAt {
    row: usize,
    start: usize,
    end: usize,
}

impl ErrorMessage {
    /// The 0-based row of the data the check failed at, `None` when the
    /// message names no row.
    pub fn row(&self) -> Option<usize> {
        self.row.map(|at| at.row)
    }

    /// The text with `shown` in place of the row's number; the text as it is
    /// when the message names no row. `text_with_row(row)` is the text.
    pub fn text_with_row(&self, shown: usize) -> String {
        match self.row {
            Some(at) => splice_row(&self.text, (at.start, at.end), shown),
            None => self.text.clone(),
        }
    }

    /// `before`, the row, then `after`: the one way a row enters a message.
    pub(crate) fn at_row(mut before: String, row: usize, after: &str) -> ErrorMessage {
        use std::fmt::Write;
        let start = before.len();
        write!(before, "{row}").expect("writing to a String cannot fail");
        let end = before.len();
        before.push_str(after);
        ErrorMessage {
            text: before,
            row: Some(RowAt { row, start, end }),
        }
    }

    /// The byte range of the row's digits in the text.
    #[cfg(feature = "orchestrate")]
    pub(crate) fn digits(&self) -> Option<(usize, usize)> {
        self.row.map(|at| (at.start, at.end))
    }
}

/// `text` with `shown` in place of its bytes `start..end`.
pub(crate) fn splice_row(text: &str, (start, end): (usize, usize), shown: usize) -> String {
    format!("{}{shown}{}", &text[..start], &text[end..])
}

impl std::ops::Deref for ErrorMessage {
    type Target = str;
    fn deref(&self) -> &str {
        &self.text
    }
}

impl From<String> for ErrorMessage {
    fn from(text: String) -> Self {
        ErrorMessage { text, row: None }
    }
}

impl From<&str> for ErrorMessage {
    fn from(text: &str) -> Self {
        text.to_string().into()
    }
}

impl From<ErrorMessage> for String {
    fn from(m: ErrorMessage) -> Self {
        m.text
    }
}

impl PartialEq for ErrorMessage {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text
    }
}

impl Eq for ErrorMessage {}

impl PartialEq<str> for ErrorMessage {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

impl PartialEq<&str> for ErrorMessage {
    fn eq(&self, other: &&str) -> bool {
        self.text == *other
    }
}

impl PartialEq<String> for ErrorMessage {
    fn eq(&self, other: &String) -> bool {
        self.text == *other
    }
}

impl PartialEq<ErrorMessage> for str {
    fn eq(&self, other: &ErrorMessage) -> bool {
        *self == other.text
    }
}

impl PartialEq<ErrorMessage> for &str {
    fn eq(&self, other: &ErrorMessage) -> bool {
        **self == other.text
    }
}

impl PartialEq<ErrorMessage> for String {
    fn eq(&self, other: &ErrorMessage) -> bool {
        *self == other.text
    }
}

/// The text as a string literal, as the `String` payload it replaced printed.
impl std::fmt::Debug for ErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.text, f)
    }
}

impl std::fmt::Display for ErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

/// A caller input the fit refuses, returned by [`fit`] and [`fit_with_start`]
/// before any fitting (and a fit at a fitted weight power that fails, see
/// [`FitError::Weights`]). Each variant carries an [`ErrorMessage`] naming the
/// field, and the row or the row and column where the check failed (rows
/// 0-based, [`FitError::row`]); its [`Display`](std::fmt::Display) is that
/// message alone.
///
/// Any other numerical failure is not a `FitError`: it is reported inside
/// the [`Fit`] (`converged`, the notes, NaN estimates).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FitError {
    /// The data do not have the shape `n` and `p` say (`x` not `n*p` long, `y`
    /// not `n` long, a non-finite `x`) or no rows, or a random slope names a
    /// column past `p`.
    Shape(ErrorMessage),
    /// A response outside the family's support: a non-finite `y`, a Poisson
    /// or negative-binomial `y < 0`, a binomial `y` outside `[0, 1]`, a Gamma
    /// or inverse-Gaussian `y <= 0`.
    ResponseOutOfSupport(ErrorMessage),
    /// Prior weights of the wrong length, or not finite and `> 0`, also once
    /// raised to [`FitOptions::weight_power`]; a [`WeightPower::Estimate`]
    /// without weights or with weights all equal, or whose fit at the power
    /// found has a log-likelihood or β̂ that is not finite where a fit in the
    /// search had a finite log-likelihood.
    Weights(ErrorMessage),
    /// An offset of the wrong length, or not finite.
    Offset(ErrorMessage),
    /// Grouping ids that do not match the model: missing on a model with
    /// random effects, given on a model without them, or of the wrong length
    /// or count.
    Ids(ErrorMessage),
    /// A warm start of the wrong width, with a non-finite entry, or with a
    /// dispersion that is not finite and `> 0`.
    Start(ErrorMessage),
    /// A [`FitOptions`] value no model can use: an even or zero `nagq`, a
    /// held `dispersion` that is not finite and `> 0`, a [`SeFor::Only`]
    /// column past `p`, a fixed `weight_power` that is not finite.
    OptionValue(ErrorMessage),
    /// A model the crate does not fit: a mixed inverse-Gaussian model,
    /// `nagq > 1` on a non-Gaussian mixed model outside the quadrature shapes
    /// (one grouping factor, at most 3 random effects per group), a second
    /// `NestedWithin` grouping, or a [`FitOptions::weight_power`] or
    /// [`FitOptions::se_type`] the model does not take (each message says
    /// what to use instead).
    Unsupported(ErrorMessage),
    /// A value past a fixed capacity of the kernels: `nagq` above
    /// [`MAX_NAGQ`](crate::limits::MAX_NAGQ).
    Envelope(ErrorMessage),
}

impl FitError {
    /// The message, whatever the variant.
    pub fn message(&self) -> &ErrorMessage {
        let (FitError::Shape(m)
        | FitError::ResponseOutOfSupport(m)
        | FitError::Weights(m)
        | FitError::Offset(m)
        | FitError::Ids(m)
        | FitError::Start(m)
        | FitError::OptionValue(m)
        | FitError::Unsupported(m)
        | FitError::Envelope(m)) = self;
        m
    }

    /// The 0-based row of the data the check failed at, `None` when the
    /// error names no row ([`ErrorMessage::row`]).
    pub fn row(&self) -> Option<usize> {
        self.message().row()
    }
}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for FitError {}

/// Fit `spec` to `data` — the default real-data entry, with a cold start.
/// Random-effect level ids are supplied per row through [`Data::ids`]
/// ([`GroupIds`]); every level count is derived from them, so `ModelSpec`
/// stays structure-only. Exactly [`fit_with_start`] with an empty
/// [`StartValues`].
///
/// Dispatches on `(family, re.is_some())` (`Gaussian`→OLS/LMM,
/// `Binomial`→GLM/GLMM, Poisson/Gamma/NB likewise; `re: None`→fixed-only,
/// `Some`→mixed). Binomial counts are fit as expanded 0/1 Bernoulli rows (the
/// kernel is Bernoulli) — except when `y` is an aggregated success PROPORTION
/// and [`Data::weights`] carries the trial count, which fits directly
/// without the expansion. That aggregated form is supported on every
/// (family, RE, solver) path, including AGQ (`nagq > 1`) on its binomial/Poisson/NB/Gamma
/// shapes (see [`Data::weights`]'s support matrix).
///
/// # Errors
///
/// A [`FitError`] when the data, the model or the options are inputs no fit
/// can take (see its variants), checked before anything is fitted. Numerical
/// failures are not errors: they signal through `fit.converged() == false`
/// with NaN-filled estimates, including the degenerate-data cases. A
/// rank-deficient design has its aliased columns dropped and the reduced
/// model fit (`Fit::aliased` flags them, their β/se are NaN). A dropped
/// column that is also a random slope keeps its random slope.
///
/// # Examples
///
/// Fixed-only OLS (`re: None`): `y = 2x` exactly, so the fitted slope is 2.
///
/// ```rust
/// use glmm::{Data, Family, FitOptions, ModelSpec};
///
/// let x = vec![1.0, 0.0, 1.0, 1.0, 1.0, 2.0]; // n=3, p=2, row-major [intercept, x]
/// let y = vec![0.0, 2.0, 4.0];
/// let model = ModelSpec::fixed(Family::Gaussian);
/// let fit = glmm::fit(&Data::new(&x, &y, 3, 2), &model, &FitOptions::default())?;
/// assert!(fit.converged());
/// assert!((fit.beta()[1] - 2.0).abs() < 1e-9);
/// # Ok::<(), glmm::FitError>(())
/// ```
///
/// A mistake in the inputs comes back as an error, not a panic:
///
/// ```rust
/// use glmm::{Data, Family, FitError, FitOptions, ModelSpec, PoissonLink};
///
/// let x = vec![1.0; 3];
/// let y = vec![1.0, -1.0, 2.0];
/// let model = ModelSpec::fixed(Family::Poisson { link: PoissonLink::Log });
/// let Err(err) = glmm::fit(&Data::new(&x, &y, 3, 1), &model, &FitOptions::default()) else {
///     panic!("a negative Poisson count is refused");
/// };
/// assert!(matches!(err, FitError::ResponseOutOfSupport(_)));
/// assert_eq!(err.to_string(), "Poisson response must be >= 0 (row 1)");
/// ```
pub fn fit(data: &Data<'_>, spec: &ModelSpec, opts: &FitOptions) -> Result<Fit, FitError> {
    let resolved = validate(data, spec, None, opts)?;
    fit_resolved(data, spec, None, opts, resolved)
}

/// [`fit`] from a warm start. One fit, one answer — the start only shortens
/// the path. The start threads `beta`+`theta` into the LMM/GLMM kernels, and
/// `dispersion` into the negative-binomial shape search (GLM and GLMM) and the
/// Gamma GLMM's `ln φ` coordinate; the other fixed-only (OLS/GLM) fits ignore
/// it (see [`crate::StartValues`]). Either vector may be left empty to
/// cold-start that component alone, and an all-empty start is [`fit`].
///
/// # Errors
///
/// As [`fit`], plus [`FitError::Start`]: `start.beta.len()` outside `{0, p}`,
/// `start.theta.len()` outside `{0, n_theta}` (the model's RE θ width), a
/// non-finite entry in `start.beta` or `start.theta`, or a `start.dispersion`
/// that is not finite and `> 0`.
///
/// # Examples
///
/// An intercept-only Gaussian LMM (3 clusters, `n_theta = 1`) warm-started off a
/// perturbed θ:
///
/// ```rust
/// use glmm::{Data, Family, FitOptions, GroupIds, ModelSpec, Re, StartValues};
///
/// let x = vec![1.0; 6]; // n=6, p=1 (intercept-only design)
/// let y = vec![1.0, 2.0, 1.5, 1.2, 2.1, 1.4];
/// let model = ModelSpec::mixed(Family::Gaussian, Re::intercept());
/// let ids = GroupIds::new(vec![0, 1, 2, 0, 1, 2]);
/// let start = StartValues::new(vec![0.0], vec![1.0]);
/// let data = Data::new(&x, &y, 6, 1).ids(&ids);
/// let fit = glmm::fit_with_start(&data, &model, &start, &FitOptions::default())?;
/// assert_eq!(fit.beta().len(), 1);
/// # Ok::<(), glmm::FitError>(())
/// ```
pub fn fit_with_start(
    data: &Data<'_>,
    spec: &ModelSpec,
    start: &StartValues,
    opts: &FitOptions,
) -> Result<Fit, FitError> {
    let resolved = validate(data, spec, Some(start), opts)?;
    fit_resolved(data, spec, Some(start), opts, resolved)
}

/// The fit of [`fit`] and [`fit_with_start`] at the weight power and the
/// standard error `opts` resolved to.
fn fit_resolved(
    data: &Data<'_>,
    spec: &ModelSpec,
    start: Option<&StartValues>,
    opts: &FitOptions,
    (power, se): (WeightPower, SeType),
) -> Result<Fit, FitError> {
    let opts = &FitOptions {
        se_type: Some(se),
        ..opts.clone()
    };
    let mut fit = match power {
        WeightPower::Estimate => {
            delta::fit_delta(data, spec, opts, start, opts.reml != Some(false))?
        }
        WeightPower::Fixed(_) => {
            let mut fit = fit_checked(data, spec, start, opts);
            fit.weight_power = power_used(data.design.weights, opts);
            fit
        }
    };
    // Non-integer counts: still fittable (the deviance and score are
    // continuous in y), but the reported numbers describe a distribution that
    // assumes integer counts, so R's `glm` warns instead of refusing —
    // matched here rather than refused. Binomial's count is the derived
    // successes `y·wᵢ`, not `y` itself. This and the identification notes
    // read the data alone, so they are computed once per user fit, on the
    // caller's design, and the aliased re-entry and the δ search skip them.
    let rows = count_non_integer_response(data.y, spec.family, data.design.weights);
    if rows > 0 {
        fit.diagnostics
            .notes
            .push(Note::NonIntegerResponse { rows });
    }
    (fit.diagnostics.notes).extend(common::identification_notes(&data.design, data.y, spec));
    #[cfg(debug_assertions)]
    check_shape(&fit);
    Ok(fit)
}

/// The layout rule of [`Fit`], asserted on every fit a build with debug
/// assertions makes. Each grouping's `q` is read off its `varcorr` block, so
/// a block without a random intercept counts its slopes only. A converged
/// fit's log-likelihood may be infinite (an exact OLS fit) but never NaN.
#[cfg(debug_assertions)]
fn check_shape(fit: &Fit) {
    assert_eq!(
        fit.varcorr.len(),
        fit.ranef_levels.len(),
        "ranef_levels: one level count per varcorr block"
    );
    let qs: Vec<usize> = fit.varcorr.iter().map(|b| vech_q(b.len())).collect();
    for (g, (block, &q)) in fit.varcorr.iter().zip(&qs).enumerate() {
        assert_eq!(
            block.len(),
            q * (q + 1) / 2,
            "varcorr[{g}]: not a vech length"
        );
    }
    let n_theta: usize = qs.iter().map(|q| q * (q + 1) / 2).sum();
    let levels_q = || qs.iter().zip(&fit.ranef_levels);
    assert_eq!(fit.tau2.len(), n_theta, "tau2: one entry per θ coordinate");
    assert_eq!(
        fit.stddev_se.len(),
        n_theta,
        "stddev_se: one entry per θ coordinate"
    );
    let n_ranef: usize = levels_q().map(|(q, l)| l * q).sum();
    assert_eq!(fit.ranef.len(), n_ranef, "ranef: levels·q per grouping");
    let pinned = &fit.diagnostics.pinned;
    assert_eq!(pinned.len(), qs.len(), "pinned: one row per grouping");
    for (g, (row, &q)) in pinned.iter().zip(&qs).enumerate() {
        assert_eq!(row.len(), q, "pinned[{g}]: one flag per standard deviation");
    }
    let p = fit.beta.len();
    assert_eq!(
        fit.diagnostics.aliased.len(),
        p,
        "diagnostics.aliased: one flag per column"
    );
    assert_eq!(fit.fitted.len(), fit.nobs, "fitted: one entry per row");
    let n_aliased = fit.diagnostics.aliased.iter().filter(|&&a| a).count();
    assert_eq!(
        fit.null_basis.len(),
        n_aliased * p,
        "null_basis: one p-wide row per aliased column"
    );
    if fit.condvar.is_empty() {
        assert!(fit.condvar_x.is_empty(), "condvar_x: empty when condvar is");
        assert!(
            fit.condvar_cross.is_none(),
            "condvar_cross: None when condvar is empty"
        );
    } else {
        let n_condvar: usize = levels_q().map(|(q, l)| l * q * (q + 1) / 2).sum();
        assert_eq!(fit.condvar.len(), n_condvar, "condvar: one vech per level");
        let n_condvar_x: usize = levels_q().map(|(q, l)| l * q * p).sum();
        assert_eq!(
            fit.condvar_x.len(),
            n_condvar_x,
            "condvar_x: one q × p block per level"
        );
    }
    assert!(
        !(fit.diagnostics.converged && fit.loglik.is_nan()),
        "loglik: NaN on a converged fit"
    );
}

/// The fit of [`fit`] and [`fit_with_start`] on inputs [`validate`] accepted,
/// without the data notes ([`Note::NonIntegerResponse`] and the
/// identification notes) and with [`Fit::weight_power`] left at the
/// mapper's default; `fit_resolved` adds both. `pub(crate)` for the callers
/// that refit a model they already know is valid (the rank-deficiency
/// salvage, the per-term likelihood-ratio refits, the weight-power probes),
/// which skip the check.
pub(crate) fn fit_checked(
    data: &Data<'_>,
    spec: &ModelSpec,
    start: Option<&StartValues>,
    opts: &FitOptions,
) -> Fit {
    let Design {
        x,
        n,
        p,
        ids,
        weights,
        ..
    } = data.design;
    let y = data.y;
    // Rank-deficiency salvage: drop fixed-effect columns aliased on earlier
    // columns and fit the reduced model (lme4 behavior — NA coefficient, still
    // converges). Path-agnostic: preprocesses the fixed design X before the
    // solver dispatch, so it serves OLS/GLM/LMM/GLMM alike. Runs after the
    // input check (full p still valid) and short-circuits into a reduced re-entry.
    // Tests the UNWEIGHTED X'X at `ALIAS_EPS`, i.e. only genuine redundancy —
    // columns indistinguishable in f64, where there is no separate coefficient
    // to estimate. This is the only place a column is ever dropped. A design
    // that is merely ill-conditioned has a unique, computable answer, and the
    // estimator routes fit it in full and record how badly conditioned it was.
    if n > 0 && p > 0 {
        let aliased = detect_aliased(x, n, p);
        if aliased.iter().any(|&a| a) {
            let mut fit = fit_rank_deficient(data, spec, start, opts, &aliased);
            fit.null_basis = common::null_basis(x, n, p, &aliased);
            return fit;
        }
    }
    // A fixed power other than 1 fits on the working weights `w^v`. Applied
    // past the alias branch, so the reduced re-entry applies it once.
    let powered = match (weights, opts.weight_power) {
        (Some(w), Some(WeightPower::Fixed(v))) if v != 1.0 => Some(working_weights(w, v)),
        _ => None,
    };
    // Dispatch over the unified fit core: size the spec from the ids, classify
    // and allocate the per-shape workspace, solve on it, map the lean view to
    // the full `Fit`. The workspace owns the ids, so the data handed to
    // `fit_on` carries none.
    let no_ids = GroupIds::default();
    let mut ws = core::build_workspace(spec, ids.unwrap_or(&no_ids), n, p, opts);
    // The stable door checks a β start for separation; the loop tier trusts the caller's.
    ws.check_start_separation();
    let rows = Data {
        design: Design {
            ids: None,
            weights: powered.as_deref().or(weights),
            ..data.design
        },
        y,
    };
    let view = core::fit_on(&mut ws, &rows, start, opts);
    let mut fit = view.into_fit(&rows, spec, opts);
    // After `into_fit`, which is the last read of the kernel's factor for the
    // fit's own outputs: the pass leaves that factor at its last stencil
    // point (on the shifted response), and nothing reads `ws` after it.
    core::attach_satterthwaite(&mut ws, &mut fit, opts);
    // After the pass, which writes the model df into `coef_df`.
    robust::apply(&mut fit, &opts.se_for.columns(p), opts.se_type, spec);
    fit
}

/// [`Fit::weight_power`] of a fit with `opts` on `weights`: NaN without
/// weights, else the fixed power (1 for `None`, which resolves to it).
pub(crate) fn power_used(weights: Option<&[f64]>, opts: &FitOptions) -> f64 {
    match (weights, opts.weight_power) {
        (None, _) => f64::NAN,
        (Some(_), Some(WeightPower::Fixed(v))) => v,
        (Some(_), _) => 1.0,
    }
}

/// The working weights `wᵢ^v` of a fit with weight power `v`.
pub(crate) fn working_weights(w: &[f64], v: f64) -> Vec<f64> {
    w.iter().map(|wi| wi.powf(v)).collect()
}

/// The weights a fit with [`Fit::weight_power`] `power` fitted with: the
/// caller's `weights` as given at power 1, else raised to it.
pub(crate) fn fitted_weights(weights: Option<&[f64]>, power: f64) -> Option<Cow<'_, [f64]>> {
    weights.map(|w| {
        if power == 1.0 {
            Cow::Borrowed(w)
        } else {
            Cow::Owned(working_weights(w, power))
        }
    })
}

/// Rows whose response is not within [`NON_INTEGER_TOL`] of an integer —
/// the raw count for Poisson/negative-binomial, or the derived successes
/// `y·wᵢ` for binomial (unit weights ⇒ Bernoulli, always integer). `0` for
/// Gaussian, Gamma and inverse-Gaussian, whose response is continuous.
/// [`NON_INTEGER_TOL`] matches R's `binomial()$initialize` tolerance; R's
/// Poisson/NB non-integer warning uses a different, tighter check (see
/// [`Note::NonIntegerResponse`]).
fn count_non_integer_response(y: &[f64], family: Family, weights: Option<&[f64]>) -> u32 {
    const NON_INTEGER_TOL: f64 = 1e-3;
    let off = |v: f64| (v - v.round()).abs() > NON_INTEGER_TOL;
    match family {
        Family::Poisson { .. } | Family::NegativeBinomial { .. } => {
            y.iter().filter(|&&v| off(v)).count() as u32
        }
        Family::Binomial { .. } => y
            .iter()
            .enumerate()
            .filter(|&(i, &v)| off(v * weights.map_or(1.0, |w| w[i])))
            .count() as u32,
        Family::Gaussian | Family::Gamma { .. } | Family::InverseGaussian { .. } => 0,
    }
}
/// Which LMM/GLMM solver a design routes to. `NoZ` is the dense
/// no-Z fast path with bounded stack scratch (the `MAX_*` envelope), kept
/// byte-identical. `Sparse` is the heap sparse-Z path that lifts the caps. The
/// boundary is currently hard-coded at the cap edge; pulling in-envelope-but-large
/// designs into `Sparse` for speed is future, measured work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Solver {
    NoZ,
    Sparse,
}

/// Route a design to `NoZ` (in-envelope) or `Sparse` (over-envelope). The
/// RE-envelope caps — extra-grouping count, primary width `q_p`, per-extra
/// width `q_g` — are a scratch ceiling, not a model ceiling: over the caps
/// routes to `Sparse` instead of failing. Fixed-only and nAGQ legality are
/// handled by `check_model_shape` (which runs first and refuses the shapes
/// no kernel fits).
///
/// On top of the structural caps, slope-carrying extra groupings always route
/// `Sparse`, and so does a total `Crossed` level count past
/// `MAX_CROSSED_LEVELS` — see the comments at the clauses.
///
/// The level-count clause reads the crossed counts of `sized`
/// ([`SizedSpec::extra_counts`]).
pub(crate) fn classify_design(sized: &SizedSpec, _nagq: u8) -> Solver {
    let Some(re) = sized.spec.re.as_ref() else {
        return Solver::NoZ; // no random effects ⇒ nothing to make sparse
    };
    let q_p = 1 + re.slopes.len();
    let over = re.extra_groupings.len() > MAX_EXTRA_GROUPINGS
        || q_p > MAX_PRIMARY_Q
        || re
            .extra_groupings
            .iter()
            .any(|g| 1 + g.slopes.len() > MAX_EXTRA_Q);
    // Extra-grouping random slopes always route Sparse. Gaussian: measured on a
    // locked-clock benchmark run — Sparse wins 4–13× across q_g ∈ {2,3,4},
    // n_extra ∈ {2,4,6}; intercept-only extras stay NoZ, 2–1500× the other
    // way. Non-Gaussian: the dense NoZ GLMM
    // blocked and structured kernels build intercept-only extras exclusively
    // (`build_packed_m` carries a debug_assert), so Sparse — whose PIRLS applies
    // full q_g×q_g Λ-blocks per extra level — is the only implemented route.
    let slope_extras = re.extra_groupings.iter().any(|g| !g.slopes.is_empty());
    // Many-level crossed extras route Sparse: every crossed level lands in the
    // dense tail (`reml_deviance`'s `t_dim`; the dense GLMM path's Schur
    // complement likewise), which is cubic in the TOTAL crossed level count —
    // hence the sum over factors, not a per-factor cap. See MAX_CROSSED_LEVELS
    // for the measured blowup and the placeholder threshold's rationale.
    let crossed_levels: usize = re
        .extra_groupings
        .iter()
        .zip(&sized.extra_counts)
        .map(|(g, &levels)| match g.relation {
            GroupingRelation::Crossed => levels,
            GroupingRelation::NestedWithin => 0,
        })
        .sum();
    if over || slope_extras || crossed_levels > crate::consts::MAX_CROSSED_LEVELS {
        Solver::Sparse
    } else {
        Solver::NoZ
    }
}

// ---------------------------------------------------------------------------
// Re-exports for consumers outside `fit` (mirrors `src/glmm/mod.rs`'s
// mod+re-export convention): `sparse.rs` reaches these as `crate::fit::X`,
// and the `loop_advanced` cargo feature's `crate::loop_advanced` re-exports the
// loop tier from here.
// ---------------------------------------------------------------------------

// The one diagnostics carrier. Always `pub` (it is `FitView::diagnostics`'s
// return type), re-exported only for the loop tier — the stable path reads it
// through `Fit`, never directly.
#[cfg(feature = "loop_advanced")]
pub use common::FitDiagnostics;
pub use common::OuterStop;
pub(crate) use glm::{nb_count_groups, nb_profile_loglik, nb_saturated_loglik_grouped};
// Test-only since the GLMM NB coordinate stopped seeding from it (it seeds from
// `fit_glm_nb`'s θ̂ instead): `fit_glm_nb_capped` calls it inside `glm.rs`, so the
// only cross-module readers left are the tests that pin it.
#[cfg(test)]
pub(crate) use glm::nb_theta_moment_seed;
// The GLMM β cold start. `fit/glmm.rs` calls it directly; the re-export serves
// the tests that seed a hand-built workspace the same way a fit does.
#[cfg(test)]
pub(crate) use glmm::glm_warm_start_beta;
// The packed-row cross-check entry point: `src/sparse/tests.rs` fits an
// in-envelope design both ways through it.
#[cfg(test)]
pub(crate) use glmm::fit_glmm_packed;
// The test-only GLMM build and the shipped solve, for `src/sparse/fd_margin.rs`.
#[cfg(all(test, feature = "formula"))]
pub(crate) use glmm::{fit_glmm_build, run_glmm_on};
#[cfg(test)]
pub(crate) use lmm::{fit_mle_noz_pub, fit_mle_sparse_pub};

/// [`fit`]'s fit of a full-rank mixed design from a cold start, returned with
/// the workspace it ran on, so the caller can read what the fit left there
/// (`FitWorkspace::profile_kernel`): the profile intervals search on that
/// workspace. `data` carries the ids and passed [`validate`]; no
/// rank-deficiency salvage and no Satterthwaite pass run here.
pub(crate) fn fit_cold_keep_workspace(
    data: &Data<'_>,
    model: &ModelSpec,
    opts: &FitOptions,
) -> (Fit, core::FitWorkspace) {
    let Design { n, p, ids, .. } = data.design;
    let ids = ids.expect("a mixed design carries its ids");
    let mut ws = core::build_workspace(model, ids, n, p, opts);
    let rows = Data {
        design: Design {
            ids: None,
            ..data.design
        },
        y: data.y,
    };
    let fit = core::fit_on(&mut ws, &rows, None, opts).into_fit(&rows, model, opts);
    (fit, ws)
}

// Unified fit-core surface for the loop tier.
#[cfg(feature = "loop_advanced")]
pub use core::{
    curvature_hint, fit_on, fit_on_hinted, satterthwaite, satterthwaite_with_contrasts, FitView,
    FitWorkspace,
};

/// The loop tier's [`core::build_workspace`]: the weights as given and the
/// model standard error, so `weight_power` and `se_type` must be `None` or
/// those values.
///
/// # Panics
///
/// As [`core::build_workspace`], and on a named [`WeightPower::Estimate`], a
/// [`WeightPower::Fixed`]`(v)` with `v ≠ 1` or [`SeType::Robust`].
#[cfg(feature = "loop_advanced")]
pub fn build_workspace(
    spec: &ModelSpec,
    ids: &GroupIds,
    n_max: usize,
    p: usize,
    opts: &FitOptions,
) -> core::FitWorkspace {
    let claimed = matches!(opts.weight_power, None | Some(WeightPower::Fixed(1.0)));
    let model_se = matches!(opts.se_type, None | Some(SeType::Model));
    if !(claimed && model_se) {
        panic!(
            "the loop tier fits the weights as given with the model standard error; \
             weight_power {:?} and se_type {:?} need glmm::fit",
            opts.weight_power, opts.se_type
        );
    }
    core::build_workspace(spec, ids, n_max, p, opts)
}

// `pub(crate)` only so `src/sparse/tests.rs` can reach the shared Tier 1 pin
// bands; the module is `cfg(test)` either way.
#[cfg(test)]
pub(crate) mod common_tests;
#[cfg(test)]
mod delta_tests;
#[cfg(test)]
mod glm_tests;
#[cfg(test)]
mod glmm_tests;
#[cfg(test)]
mod lmm_tests;
#[cfg(test)]
mod ols_tests;
#[cfg(test)]
mod standardize_tests;
