//! `glmm` — standalone f64 GLMM fit kernels (OLS → GLM → LMM → GLMM).
//!
//! Two public surfaces (see the carve design spec §5):
//! - [`fit`] + [`ModelSpec`]: the stable, semver-covered friendly API.
//! - `mcpower` (cargo feature, off by default): the unstable scratch-explicit
//!   hot-path surface MCPower's simulation layer binds to. NO semver guarantees.

// The scratch-explicit kernels (glm/glmm/lme + most of lmm) exist to serve the
// `mcpower` feature; the stable `fit` wires only a subset (OLS + LMM). With the
// feature OFF they are intentionally unreachable, not stale — so suppress
// dead_code only in that build. The `mcpower` build uses all of it, so genuinely
// dead code is still caught there (it's the superset).
#![cfg_attr(not(feature = "mcpower"), allow(dead_code))]

pub mod consts;
pub mod linalg;
pub mod simd_transcendental;

mod fit;
mod glm;
mod glmm;
mod lme;
mod lmm;
mod ols;
mod spec;
pub use fit::{fit, Fit, FitOptions};
pub use spec::*;

/// Tiny float guard — magnitudes below this are treated as zero (rank /
/// division-by-zero sentinel). Mirrors v1's FLOAT_NEAR_ZERO; comparing
/// `β̂²/var_diag` against thresholds with NaN propagates as "fail" downstream.
/// Duplicated in engine-core (still used there by data_gen/posthoc) — a 1-line
/// numeric sentinel is fewer moving parts than a cross-crate `pub` edge.
pub(crate) const FLOAT_NEAR_ZERO: f64 = 1e-30;

#[cfg(feature = "mcpower")]
pub mod mcpower;

#[cfg(test)]
mod test_support;

// dhat alloc-profiling for the `#[ignore]` warm-path bounded-alloc tests.
// cfg(test) only — the externalizable library never ships a custom allocator.
#[cfg(test)]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;
