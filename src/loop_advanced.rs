//! UNSTABLE scratch-explicit hot-path surface (the loop tier) for warm-start
//! consumers like MCPower. Gated by the `loop_advanced` cargo feature (off by
//! default). NO semver guarantees — may change in ANY release. The warm-start
//! primitive [`crate::StartValues`] is re-exported `pub` only behind this feature.

// The unified fit core — the build-once/fit-many surface for the loop tier, and
// the same dispatch body `fit_cold`/`fit_warm` run, so a model or routing change
// lands in every caller identically. Prefer it over the individual kernels below,
// which do not classify the design for you.
pub use crate::fit::{
    build_lmm_seam_ws, build_workspace, fit_on, lmm_objective_at, lmm_sweep_fit, lmm_sweep_fit_on,
    FitDiagnostics, FitView, FitWorkspace, LmmSeamWs, LmmSweepOutcome,
};
// The RE-level-count normalizer `build_workspace` expects its `sized` spec to have
// gone through — exposed so loop-tier consumers size specs the validated way
// instead of reimplementing the crossed/nested count derivation. It also decides
// the grouping order the kernels see, which `Perm` records: the ids it returns
// are what `fit_on` must be handed, and the `Perm` is what maps a caller's own θ
// (into `StartValues`, out of `LmmSweepOutcome`) between the two orders.
pub use crate::fit::{spec_sized_from_ids_pub, Perm};
pub use crate::glm::{glm_irls_fit, sigmoid_stable, GlmFitView, GlmScratch, MAX_IRLS_ITERS};
pub use crate::glmm::GlmmFit;
pub use crate::lmm::{primary_lambda, LmmFit, LmmGroupings, LmmSuffStats};
pub use crate::ols::{
    fit_suff_stats_t_sq, ols_contrast_t_sq, OlsFitView, OlsScratch, OlsSuffStats, PANEL_ROWS,
};
