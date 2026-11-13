//! `glmm` — standalone f64 GLMM fit kernels (OLS → GLM → LMM → GLMM).
//!
//! The full dispatch — every route from [`fit`]/[`fit_with_start`] down to its
//! solver — is traced in `documentation/algorithms.md`.
//!
//! The public surface, by tier:
//! - The matrix door, covered by semver: [`fit`]/[`fit_with_start`] on a
//!   [`Data`] and [`predict()`] on a [`Design`], with its types
//!   ([`PredictOptions`], [`Prediction`], [`Scale`]), [`ModelSpec`] and the
//!   spec types, [`GroupIds`], [`FitOptions`] and its enums ([`SeFor`], [`WaldSe`],
//!   [`Dispersion`], [`WeightPower`], [`SeType`]), [`StartValues`],
//!   [`Fit`] and its methods (the accessors and the inference methods
//!   `contrast_t`, `contrast_f`, `r2`, `icc`, `profile_intervals`),
//!   [`Diagnostics`], [`Note`], [`Boundary`], [`OuterStop`], [`FitError`],
//!   [`ErrorMessage`] and [`limits`]. A `limits` constant may grow in a minor
//!   release; a name does not disappear.
//! - The formula door ([`formula`], cargo feature on by default), covered by
//!   semver: `formula::{lower, fit, fit_lowered, predict, Table, Column,
//!   Lowered, Model}`, `formula::anova::{compare, term_tests, nesting}`,
//!   `formula::pairwise`, their types and the error types.
//!   `formula::fit("y ~ x + (1|g)", &table, family, &opts)` fits a formula
//!   string against a data table, and `formula::lower` builds the kernel's
//!   inputs from them instead of by hand.
//! - The [`inference`] types the two doors return, and [`Simulator`], covered
//!   by semver.
//! - `loop_advanced` (cargo feature, off by default): the unstable
//!   scratch-explicit hot path (loop tier) for warm-start callers like
//!   MCPower, exposing the kernels and their workspaces. NO semver guarantees;
//!   its items are named for a capability, never for a caller.
//! - `orchestrate` (cargo feature, off by default): the string-typed fit
//!   orchestration the FFI ports (`glmm-python`, `glmm-r`) share, and the
//!   formula internals they use (`formula::parse`, `formula::materialize`,
//!   `formula::label_ranef`, `formula::anova::compare_stored` and the like).
//!   NO semver guarantees.
//! - `parallel` (cargo feature, off by default, **experimental**, outside
//!   semver): enables in-fit parallelism (AGQ cluster loop, FD-Hessian grid)
//!   via rayon's global pool; a no-op on wasm32; gated at runtime by
//!   `FitOptions::parallel_inner` (also off by default: both the feature and
//!   the knob are explicit opt-ins). Its results are bit-identical to serial.
//! - `counters` (cargo feature, off by default, dev-only, outside semver):
//!   adds `glmm::EvalCounters` on `Fit`: the stage-1/stage-2 evaluation
//!   split, evaluations after the last incumbent improvement, a
//!   PIRLS-iterations-per-evaluation histogram, AGQ evaluations x nodes, and
//!   the searches run after a pin (the trough polish).
//! - `alloc-tests` (cargo feature, off by default, dev-only, outside semver):
//!   installs dhat as the test binary's global allocator for the
//!   bounded-allocation tests.
//! - `oracle-tests` (cargo feature, off by default, dev-only, outside semver):
//!   the cross-engine tier that checks the crate against the frozen lme4 /
//!   MixedModels / GLMMadaptive goldens.
//! - `montecarlo-tests` (cargo feature, off by default, dev-only, outside
//!   semver): the Monte Carlo tier, level studies checked against seeded
//!   bands.
//!
//! Everything else is private.

// Every fit route (OLS, GLM, LMM, GLMM) is wired into the stable `fit` module
// regardless of this feature — see `documentation/algorithms.md` for the full
// dispatch. What goes dead with the feature OFF is mainly the narrower
// `loop_advanced` surface: the `FitView` hot-loop accessors and the
// gradient/Hessian kernels in `src/glmm/derivative.rs`,
// plus a few other loop-only helpers scattered through `src/lmm/mod.rs`,
// `src/glmm/assembled.rs`, and `src/ols.rs`. So suppress dead_code only in
// that build. The `loop_advanced` build uses all of it, so genuinely dead
// code is still caught there (it's the superset).
#![cfg_attr(not(feature = "loop_advanced"), allow(dead_code))]
// Every public statistical item must state its convention and cite its oracle.
#![warn(missing_docs)]

mod consts;
mod dual;
pub mod inference;
mod linalg;
mod scalar;

/// The kernels' capacity constants and the input ranges. A limit may grow in
/// a minor release; a name does not disappear.
///
/// - `MAX_PRIMARY_Q`, `MAX_EXTRA_GROUPINGS`, `MAX_EXTRA_Q` and
///   `MAX_CROSSED_LEVELS` bound the dense (no-Z) solver: the primary
///   grouping's random-effect width, the number of extra groupings, the width
///   of each extra grouping, and the total level count over crossed extra
///   groupings. A design past any of them is not refused: it routes to the
///   sparse solver and still fits. `MAX_THETA` is the θ length those four
///   bounds imply on the dense path.
/// - `MAX_NAGQ` is a hard limit: the largest adaptive Gauss–Hermite order a
///   fit accepts. A larger `FitOptions::nagq` is refused with
///   [`FitError::Envelope`].
/// - Input ranges. `MAX_DATA_ABS` bounds |y|, every |x| and random-effect-only
///   column entry, and |offset| on a link other than log; weights and an
///   inverse-Gaussian held dispersion lie in `[1/MAX_DATA_ABS, MAX_DATA_ABS]`.
///   `MAX_LOG_OFFSET_ABS` bounds |offset| on a log link, and `MAX_TRIALS`
///   (2⁵³) a binomial weight. `GAMMA_PHI_LO`, `GAMMA_PHI_HI` are the box of the
///   Gamma dispersion search and `NB_THETA_LO`, `NB_THETA_HI` that of the
///   negative-binomial shape; a negative-binomial shape seed outside its box
///   is refused, and so is a Gamma held dispersion outside the Gamma box or
///   an inverse-Gaussian one outside `[1/MAX_DATA_ABS, MAX_DATA_ABS]`. [`fit`], [`fit_with_start`] and [`predict()`] refuse an input
///   outside its range before fitting, with the value and the constant's name
///   in the message.
pub mod limits {
    pub use crate::consts::{
        GAMMA_PHI_HI, GAMMA_PHI_LO, MAX_CROSSED_LEVELS, MAX_DATA_ABS, MAX_EXTRA_GROUPINGS,
        MAX_EXTRA_Q, MAX_LOG_OFFSET_ABS, MAX_NAGQ, MAX_PRIMARY_Q, MAX_THETA, MAX_TRIALS,
        NB_THETA_HI, NB_THETA_LO,
    };
}

// R-style formula frontend (`formula` feature, on by default). Off for the
// formula-free hot path, which then links no `regex`. Module docs live in
// src/formula/mod.rs — keep them there: a `///` fragment here would resolve the
// module's intra-doc links in the crate root's scope, breaking them.
#[cfg(feature = "formula")]
pub mod formula;

mod counters;
mod data;
mod family;
mod fit;
mod glm;
mod glmm;
mod ids;
mod lmm;
mod ols;
mod optim;
mod predict;
mod simd_transcendental;
mod simulate;
mod sparse;
mod spec;
mod standardize;
mod start;
#[cfg(feature = "counters")]
pub use counters::{EvalCounters, Stage};
pub use data::{Data, Design};
pub use fit::{
    fit, fit_with_start, Boundary, Diagnostics, Dispersion, ErrorMessage, Fit, FitError,
    FitOptions, Note, OuterStop, SeFor, SeType, WeightPower, WEIGHT_POWER_SIMULATED,
};
pub use ids::GroupIds;
pub use inference::{Icc, R2};
pub use predict::{predict, PredictOptions, Prediction, Scale};
pub use simulate::Simulator;
pub use spec::*;

#[cfg(feature = "loop_advanced")]
pub mod loop_advanced;

// String-typed fit orchestration shared by the FFI ports (`orchestrate`
// feature, off by default). Like `loop_advanced`: real surface, no semver
// promise — its shape follows the ports' needs.
#[cfg(feature = "orchestrate")]
pub mod orchestrate;

pub use start::StartValues;

#[cfg(test)]
mod test_support;

// dhat alloc-profiling for the `#[ignore]` warm-path bounded-alloc tests.
// Gated behind the `alloc-tests` feature (not plain cfg(test)): dhat's
// allocator takes a global lock on every allocation, which serializes the
// otherwise-parallel test suite. The library never ships a custom allocator.
#[cfg(all(test, feature = "alloc-tests"))]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
