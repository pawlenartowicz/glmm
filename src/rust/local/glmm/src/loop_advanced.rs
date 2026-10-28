//! UNSTABLE scratch-explicit hot-path surface (the loop tier) for warm-start
//! consumers like MCPower. Gated by the `loop_advanced` cargo feature (off by
//! default). NO semver guarantees — may change in ANY release. The warm-start
//! primitive [`crate::StartValues`] is re-exported `pub` only behind this feature.

// The unified fit core — the build-once/fit-many surface for the loop tier, and
// the same dispatch body `fit`/`fit_with_start` run, so a model or routing change
// lands in every caller identically. Prefer it over the individual kernels below,
// which do not classify the design for you.
pub use crate::fit::{
    build_workspace, curvature_hint, fit_on, fit_on_hinted, satterthwaite, FitDiagnostics, FitView,
    FitWorkspace, OuterStop,
};
pub use crate::glm::{glm_irls_fit, sigmoid_stable, GlmFitView, GlmScratch, MAX_IRLS_ITERS};
pub use crate::glmm::GlmmFit;
pub use crate::lmm::{primary_lambda, LmmFit, LmmGroupings, LmmSuffStats};
pub use crate::ols::{
    fit_suff_stats_t_sq, ols_contrast_t_sq, OlsFitView, OlsScratch, OlsSuffStats, PANEL_ROWS,
};
pub use crate::optim::CurvatureHint;
// Direct access to the hypothesis tests, for a caller that decides per draw
// which tests it pays for: the likelihood-ratio p-value of a statistic, the
// Wald χ² of `Lβ = 0` on a fit's estimates and covariance, and, with
// `formula`, lmerTest's Type III hypothesis matrices of a design, built once
// per design. Like the rest of this module, no semver promise.
#[cfg(feature = "formula")]
pub use crate::formula::hypotheses::{type3_pub, TermHypothesis};
pub use crate::inference::lr::lr_p_value_pub;
pub use crate::inference::wald::wald_chisq_pub;
