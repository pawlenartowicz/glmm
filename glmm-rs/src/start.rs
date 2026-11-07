//! `StartValues` — the warm-start primitive.
//!
//! Raw optimizer state, *not* high-level variances: the dominant use is threading
//! the previous fit's state into the next refit (loop tier / future inference
//! layer), so it mirrors what the kernels already hold (`ws.params`/`theta`, the β
//! estimate). The stable `fit_with_start` takes this as its warm start; `fit`
//! omits it and cold-starts the kernels with `theta_start = None`. A stable
//! input, so `pub`.
//!
//! Carries `beta`, `theta` and an optional `dispersion`: the negative-binomial
//! shape θ_NB and the Gamma φ, the outer search's trailing coordinate on a
//! GLMM (and the NB GLM's shape search). Left `None`, θ_NB cold-starts from
//! the no-RE GLM-NB's own θ̂ (`fit::fit_glm_nb`) and `ln φ` from the no-RE
//! deviance dispersion (`glmm::fit_glmm`).

/// Raw optimizer warm-start state for one model fit.
///
/// - `beta`: fixed-effect start, length `p`.
/// - `theta`: RE Cholesky parameters (the kernel's `θ`, column-major vech),
///   length `n_theta` for the model's RE structure. Empty for fixed-only
///   (OLS/GLM) models.
///
/// - `dispersion`: the family's estimated dispersion in the units
///   [`crate::Fit::dispersion`] reports it, so a previous fit's value carries
///   over as is: the shape θ for negative-binomial, φ for Gamma. `None`
///   cold-starts it. Read by the negative-binomial GLM and GLMM and the Gamma
///   GLMM; ignored by every other family and route (and by a Gamma fit whose
///   φ is held through [`crate::FitOptions::dispersion`]).
///
/// Either vector may be left EMPTY to cold-start that component on its own, which
/// is what a caller supplying only one of the two (lme4's `start = list(theta =
/// …)` shape) needs: β then seeds from the no-RE GLM fit and θ from the blind
/// `THETA0` shape, exactly as `start = None` would. The two cold seeds are
/// computed inside the kernels, so a caller cannot reproduce them itself.
/// `StartValues::default()` is the all-cold start; [`StartValues::new`] sets
/// β and θ, and the dispersion is assigned through its field.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StartValues {
    /// Warm-start β, length `p`; empty cold-starts β.
    pub beta: Vec<f64>,
    /// Warm-start θ (RE Cholesky vech), length `n_theta`; empty cold-starts θ
    /// (and is the only valid value for fixed-only models).
    pub theta: Vec<f64>,
    /// Warm-start dispersion (NB shape θ, Gamma φ), finite and `> 0`; `None`
    /// cold-starts it.
    pub dispersion: Option<f64>,
}

impl StartValues {
    /// A start from β and θ, with the dispersion cold-started (`None`).
    pub fn new(beta: Vec<f64>, theta: Vec<f64>) -> StartValues {
        StartValues {
            beta,
            theta,
            dispersion: None,
        }
    }
}
