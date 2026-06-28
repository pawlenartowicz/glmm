//! UNSTABLE scratch-explicit surface for MCPower's simulation hot path.
//! Gated by the `mcpower` cargo feature (off by default). NO semver guarantees —
//! may change in ANY release (carve spec §5).

pub use crate::glm::{glm_irls_fit, sigmoid_stable, GlmFitView, GlmScratch, MAX_IRLS_ITERS};
pub use crate::glmm::{build_z, fit_glmm, GlmmFit, GlmmWorkspace};
pub use crate::lme::{lme_fit, LmeFitView, LmeScratch, LmeSuffStats};
pub use crate::lmm::{
    cluster_theta_truth, fit_lmm, primary_lambda, LmmFit, LmmSuffStats, LmmWorkspace,
};
pub use crate::ols::{
    fit_suff_stats_t_sq, ols_contrast_t_sq, OlsFitView, OlsScratch, OlsSuffStats, PANEL_ROWS,
};
