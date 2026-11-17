//! UNSTABLE scratch-explicit hot-path surface (the loop tier) for warm-start
//! consumers like MCPower. Gated by the `loop_advanced` cargo feature (off by
//! default). NO semver guarantees: it may change in ANY release.
//! [`crate::StartValues`] (crate root, semver-covered) seeds these kernels too.

// The unified fit core — the build-once/fit-many surface for the loop tier, and
// the same dispatch body `fit`/`fit_with_start` run, so a model or routing change
// lands in every caller identically. Prefer it over the individual kernels below,
// which do not classify the design for you.
pub use crate::fit::{
    build_workspace, curvature_hint, fit_on, fit_on_hinted, satterthwaite,
    satterthwaite_with_contrasts, FitDiagnostics, FitView, FitWorkspace, OuterStop,
};
pub use crate::glm::{glm_contrast_t_sq, glm_irls_fit, GlmFitView, GlmScratch, MAX_IRLS_ITERS};
pub use crate::glmm::GlmmFit;
/// The blind θ start: diagonal vech entries at this value, off-diagonals at 0.
/// A caller that builds a full θ for [`crate::StartValues`] by hand starts its
/// variance components here to match the cold start.
pub use crate::lmm::THETA0;
pub use crate::lmm::{LmmFit, LmmGroupings, LmmSuffStats};
pub use crate::ols::{
    fit_suff_stats_t_sq, ols_contrast_t_sq, OlsFitView, OlsScratch, OlsSuffStats, PANEL_ROWS,
};
pub use crate::optim::CurvatureHint;
