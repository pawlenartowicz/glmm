//! `StartValues` — the warm-start primitive.
//!
//! Raw optimizer state, *not* high-level variances: the dominant use is threading
//! the previous fit's state into the next refit (loop tier / future inference
//! layer), so it mirrors what the kernels already hold (`ws.params`/`theta`, the β
//! estimate). The stable `fit_warm` takes this as an optional warm start; `fit_cold`
//! (≡ `fit_warm(.., None, ..)`) omits it and cold-starts the kernels with
//! `theta_start = None`. Shared internal primitive: `pub(crate)`
//! always, and now unconditionally `pub` (a stable input to `fit_warm`).
//!
//! Carries only `beta`/`theta`: Gamma φ is profiled (`family::gamma_aic`, no
//! warm-startable state) and the GLMM neg-binomial θ search is a global bracket
//! (`fit::golden_max_ln_theta`), so neither warm-starts anything reachable here.

/// Raw optimizer warm-start state for one model fit.
///
/// - `beta`: fixed-effect start, length `p`.
/// - `theta`: RE Cholesky parameters (the kernel's `θ`, column-major vech),
///   length `n_theta` for the model's RE structure. Empty for fixed-only
///   (OLS/GLM) models.
///
/// Either field may be left EMPTY to cold-start that component on its own, which
/// is what a caller supplying only one of the two (lme4's `start = list(theta =
/// …)` shape) needs: β then seeds from the no-RE GLM fit and θ from the blind
/// `THETA0` shape, exactly as `start = None` would. The two cold seeds are
/// computed inside the kernels, so a caller cannot reproduce them itself.
#[derive(Debug, Clone, PartialEq)]
pub struct StartValues {
    /// Warm-start β, length `p`; empty cold-starts β.
    pub beta: Vec<f64>,
    /// Warm-start θ (RE Cholesky vech), length `n_theta`; empty cold-starts θ
    /// (and is the only valid value for fixed-only models).
    pub theta: Vec<f64>,
}
