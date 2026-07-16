//! `glmm` — standalone f64 GLMM fit kernels (OLS → GLM → LMM → GLMM).
//!
//! Two public surfaces:
//! - [`fit_cold`]/[`fit_warm`] + [`ModelSpec`] + [`GroupIds`]: the stable,
//!   semver-covered friendly API.
//! - `loop_advanced` (cargo feature, off by default): the unstable scratch-explicit
//!   hot-path surface (loop tier) that warm-start consumers like MCPower bind to,
//!   exposing the kernels and [`StartValues`]. NO semver guarantees.
//! - [`formula`] (cargo feature, on by default): the R-style formula frontend —
//!   `formula::lower("y ~ x + (1|g)", &table, family)` builds the kernel's inputs
//!   from a formula string and a data table instead of by hand.
//!
//! `parallel` (cargo feature, off by default, **experimental**): enables in-fit
//! parallelism (AGQ cluster loop, FD-Hessian grid) via rayon's global pool; a
//! no-op on wasm32; gated at runtime by `FitOptions::parallel_inner` (also off
//! by default — both the feature and the knob are explicit opt-ins).

// The scratch-explicit kernels (glm/glmm/lme + most of lmm) exist to serve the
// `loop_advanced` feature; the stable `fit` wires only a subset (OLS + LMM). With
// the feature OFF they are intentionally unreachable, not stale — so suppress
// dead_code only in that build. The `loop_advanced` build uses all of it, so
// genuinely dead code is still caught there (it's the superset).
#![cfg_attr(not(feature = "loop_advanced"), allow(dead_code))]
// Every public statistical item must state its convention and cite its oracle.
#![warn(missing_docs)]

pub mod consts;
pub mod linalg;
pub mod simd_transcendental;

// R-style formula frontend (`formula` feature, on by default). Off for the
// formula-free hot path, which then links no `regex`. Module docs live in
// src/formula/mod.rs — keep them there: a `///` fragment here would resolve the
// module's intra-doc links in the crate root's scope, breaking them.
#[cfg(feature = "formula")]
pub mod formula;

mod family;
mod fit;
mod glm;
mod glmm;
mod ids;
mod lme;
mod lmm;
mod ols;
mod sparse;
mod spec;
mod start;
pub use fit::{fit_cold, fit_warm, Fit, FitOptions};
pub use ids::GroupIds;
pub use spec::*;

/// Tiny float guard — magnitudes below this are treated as zero (rank /
/// division-by-zero sentinel): comparing `β̂²/var_diag` against thresholds with
/// NaN propagates as "fail" downstream.
pub(crate) const FLOAT_NEAR_ZERO: f64 = 1e-30;

#[cfg(feature = "loop_advanced")]
pub mod loop_advanced;

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
