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
/// - `beta`: fixed-effect start, length `p` (cold = all-zero).
/// - `theta`: RE Cholesky parameters (the kernel's `θ`, column-major vech),
///   length `n_theta` for the model's RE structure (cold = the kernel's flat
///   `THETA0` blind start). Empty for fixed-only (OLS/GLM) models.
#[derive(Debug, Clone, PartialEq)]
pub struct StartValues {
    pub beta: Vec<f64>,
    pub theta: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn start_values_round_trips_fields() {
        let s = StartValues {
            beta: vec![0.1, -0.2],
            theta: vec![0.7],
        };
        assert_eq!(s.beta.len(), 2);
        assert_eq!(s.theta, vec![0.7]);
    }
}
