//! Family/link IRLS math primitives for the **new** M3 outcome families
//! (Poisson, Gamma, negative-binomial, and the binomial probit link).
//!
//! Single source of the four McCullagh–Nelder (1989) GLM quantities per
//! `(family, link)` — inverse link, link derivative, variance function, and
//! per-observation deviance — plus the assembled IRLS weight / working residual.
//! `match`-dispatched, no `dyn`.
//!
//! The canonical-logit hot path is **not** routed through here: it stays in the
//! fused-SIMD kernel (`pw_and_log1pexp_sum`) for byte-identity. `Binomial{Logit}`
//! is implemented for completeness/tests only. The new families take the general
//! Fisher-scoring branch, which therefore cannot perturb anything that works
//! today.
//!
//! Convention: weights/residuals are the working-response IRLS form of MN89
//! (`z = η + (y−μ)·dη/dμ`, `W = (dμ/dη)²/(φ·V(μ))`), with `φ` folded as 1 here —
//! Gamma/NB dispersion scales the SE post-fit / via the deviance, not the weight.
//! Weights are returned raw; the IRLS caller applies the
//! `glm::WEIGHT_CLAMP` floor.

use crate::spec::{BinomialLink, Family, GammaLink};

/// `exp(η)` stays finite up to η≈709; clamp short of it so log-link μ never
/// overflows to `inf` mid-IRLS.
const ETA_MAX: f64 = 700.0;
/// Floor for log/inverse-link μ so `V(μ)` and the working residual never divide
/// by zero (the IRLS weight floor `glm::WEIGHT_CLAMP` is the downstream guard).
const MU_FLOOR: f64 = 1e-10;
/// Binomial μ is kept in `(PROB_EPS, 1−PROB_EPS)` so probit deviance/weights stay
/// finite at the saturated ends.
const PROB_EPS: f64 = 1e-12;
/// `1/√(2π)` — the standard-normal pdf normalizer (probit `dμ/dη`).
const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Clamp η to each link's safe domain: `±ETA_MAX` for log links (overflow guard)
/// and `η>0` for the Gamma inverse link (`μ=1/η>0`). Identity links pass through.
pub(crate) fn clamp_eta(family: Family, eta: f64) -> f64 {
    match family {
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => eta.clamp(MU_FLOOR, ETA_MAX),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. } => eta.clamp(-ETA_MAX, ETA_MAX),
        Family::Binomial { .. } | Family::Gaussian => eta,
    }
}

/// Clamp μ to each family's valid domain: `≥ MU_FLOOR` for Poisson/Gamma/NB
/// (positive mean), `(PROB_EPS, 1−PROB_EPS)` for binomial. Gaussian passes through.
pub(crate) fn clamp_mu(family: Family, mu: f64) -> f64 {
    match family {
        Family::Binomial { .. } => mu.clamp(PROB_EPS, 1.0 - PROB_EPS),
        Family::Poisson { .. } | Family::Gamma { .. } | Family::NegativeBinomial { .. } => {
            mu.max(MU_FLOOR)
        }
        Family::Gaussian => mu,
    }
}

/// Inverse link `g⁻¹(η) → μ`, with the link's domain clamps applied so μ is
/// always valid for [`variance`]/[`dev_resid`].
pub(crate) fn link_inv(family: Family, eta: f64) -> f64 {
    let eta = clamp_eta(family, eta);
    let mu = match family {
        Family::Gaussian => eta,
        Family::Binomial {
            link: BinomialLink::Logit,
        } => crate::glm::sigmoid_stable(eta),
        Family::Binomial {
            link: BinomialLink::Probit,
        } => crate::simd_transcendental::phi_hp(eta),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. } => eta.exp(),
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => 1.0 / eta,
    };
    clamp_mu(family, mu)
}

/// Link derivative `dμ/dη` at η. Used by the general Fisher-scoring weight and
/// working residual for the non-canonical links.
pub(crate) fn mu_eta(family: Family, eta: f64) -> f64 {
    let eta = clamp_eta(family, eta);
    match family {
        Family::Gaussian => 1.0,
        Family::Binomial {
            link: BinomialLink::Logit,
        } => {
            let mu = crate::glm::sigmoid_stable(eta);
            mu * (1.0 - mu)
        }
        Family::Binomial {
            link: BinomialLink::Probit,
        } => FRAC_1_SQRT_2PI * (-0.5 * eta * eta).exp(),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. } => eta.exp(),
        // μ=1/η ⇒ dμ/dη = −1/η² = −μ².
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => {
            let mu = 1.0 / eta;
            -mu * mu
        }
    }
}

/// Family variance function `V(μ)`. `nb_theta` is the NB dispersion θ̂ the fit's
/// outer loop fixes for this evaluation — read only by the NB arm; every other
/// family ignores it (pass `f64::NAN`).
pub(crate) fn variance(family: Family, nb_theta: f64, mu: f64) -> f64 {
    match family {
        Family::Gaussian => 1.0,
        Family::Binomial { .. } => mu * (1.0 - mu),
        Family::Poisson { .. } => mu,
        Family::Gamma { .. } => mu * mu,
        Family::NegativeBinomial { .. } => mu + mu * mu / nb_theta,
    }
}

/// Per-observation deviance contribution `dᵢ ≥ 0` (`Σ dᵢ` is the GLM deviance,
/// −2·log-likelihood up to the saturated constant). Zero at `y=μ`.
pub(crate) fn dev_resid(family: Family, nb_theta: f64, y: f64, mu: f64) -> f64 {
    match family {
        Family::Gaussian => {
            let r = y - mu;
            r * r
        }
        // 2[ y ln(y/μ) + (1−y) ln((1−y)/(1−μ)) ], with the 0·ln0→0 limits.
        Family::Binomial { .. } => {
            let a = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
            let b = if y < 1.0 {
                (1.0 - y) * ((1.0 - y) / (1.0 - mu)).ln()
            } else {
                0.0
            };
            2.0 * (a + b)
        }
        // 2[ y ln(y/μ) − (y−μ) ], y·ln(y/μ)→0 at y=0.
        Family::Poisson { .. } => {
            let t = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
            2.0 * (t - (y - mu))
        }
        // 2[ −ln(y/μ) + (y−μ)/μ ]; same form for log and inverse links.
        Family::Gamma { .. } => 2.0 * (-(y / mu).ln() + (y - mu) / mu),
        // 2[ y ln(y/μ) − (y+θ) ln((y+θ)/(μ+θ)) ], y·ln(y/μ)→0 at y=0; θ = nb_theta.
        Family::NegativeBinomial { .. } => {
            let t = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
            2.0 * (t - (y + nb_theta) * ((y + nb_theta) / (mu + nb_theta)).ln())
        }
    }
}

/// lme4's `Gamma()$aic` — the Gamma family's contribution to the Laplace deviance,
/// substituted for the bare deviance `D` in the Gamma-GLMM objective. The
/// dispersion is **profiled** inside this term as `disp = D/Σwₖ` (`Σwₖ = n` when
/// `prior_w` is `None` — unit prior weights), not carried as a free parameter, so
/// ```text
///   aic = −2·Σᵢ wᵢ·log dgamma(yᵢ; shape = 1/disp, scale = μᵢ·disp) + 2
///   log dgamma(y; a, s) = (a−1)·ln y − y/s − a·ln s − lnΓ(a)
/// ```
/// matching R's weighted `Gamma()$aic`: `disp = dev/Σwᵢ`, each log-density term
/// scaled by its row's `wᵢ`. This is the **only** place the dispersion enters
/// glmer's Gamma fit (its PIRLS weights and the `‖u‖²` penalty are unit-scale —
/// no `1/φ` weighting, no φ-ridge; confirmed against lme4 `src/glmFamily.cpp`).
/// Swapping `D → aic` in the objective is what makes the kernel's β̂/τ̂ and
/// FD-Hessian SE pick up the dispersion coupling. Needs `lnΓ` (one call, on the
/// scalar shape `1/disp`) — no digamma, since the dispersion is profiled rather
/// than ML-solved. Validated against the `fit::tests::fit_glmm_gamma_sim_matches_lme4`
/// golden (unweighted) and `fit_glmm_gamma_weighted_matches_lme4` (weighted).
pub(crate) fn gamma_aic(y: &[f64], mu: &[f64], dev: f64, n: usize, prior_w: Option<&[f64]>) -> f64 {
    let sum_w = prior_w.map_or(n as f64, |w| w[..n].iter().sum());
    let disp = dev / sum_w;
    let a = 1.0 / disp; // shape = 1/disp
    let ln_gamma_a = crate::simd_transcendental::ln_gamma(a);
    let mut s = 0.0;
    for (i, (&yi, &mui)) in y.iter().zip(mu).take(n).enumerate() {
        let scale = mui * disp; // sᵢ = μᵢ·disp
        s += prior_w.map_or(1.0, |w| w[i])
            * ((a - 1.0) * yi.ln() - yi / scale - a * scale.ln() - ln_gamma_a);
    }
    -2.0 * s + 2.0
}

/// lme4's `sigma(merMod)²` for a GLMM with a free scale: `σ̂² = pwrss/n =
/// (Σᵢ wᵢ·rᵢ² + ‖û‖²)/n` with Pearson residuals `rᵢ = (yᵢ−μᵢ)/√V(μᵢ)` (for Gamma,
/// `V(μ)=μ²` ⇒ `rᵢ=(yᵢ−μᵢ)/μᵢ`) and `wᵢ` the row's prior weight (`prior_w = None`
/// ⇒ unit weights). Fixed-scale families (binomial/Poisson/NB — their
/// overdispersion lives in θ, not σ²) return 1. This is the factor lme4's
/// `vcov(use.hessian = FALSE)` puts on the RX/Schur vcov and the one its VarCorr
/// stddevs carry — distinct from the Pearson/(n−p) `dispersion` moment reported
/// separately. `mu`/`u` are the CONVERGED conditional means/modes (post pinned-γ̂
/// re-eval); Gamma Rx-SE gating: `fit::tests::fit_glmm_gamma_sim_matches_lme4`.
/// The denominator stays the RAW `y.len()` (not `Σwᵢ`) under weighting — verified
/// against the `fit_glmm_gamma_weighted_matches_lme4` golden, which matches lme4's
/// `pwrss/n` with `n` the raw row count even when `weights=` is supplied.
pub(crate) fn glmm_sigma_sq(
    family: Family,
    y: &[f64],
    mu: &[f64],
    u: &[f64],
    prior_w: Option<&[f64]>,
) -> f64 {
    match family {
        Family::Gamma { .. } => {
            let mut wrss = 0.0;
            for (i, (&yi, &mui)) in y.iter().zip(mu).enumerate() {
                let r = (yi - mui) / mui;
                wrss += prior_w.map_or(1.0, |w| w[i]) * r * r;
            }
            let usq: f64 = u.iter().map(|&v| v * v).sum();
            (wrss + usq) / y.len() as f64
        }
        _ => 1.0,
    }
}

/// Pearson-moment dispersion `φ̂ = Σᵢ wᵢrᵢ²/(n−p)`, `rᵢ = (yᵢ−μᵢ)/√V(μᵢ)`, raw
/// `n−p` degrees of freedom (not `Σwᵢ−p`) — matches R's `summary.glm`'s
/// (weighted) Pearson dispersion. `prior_w = None` ⇒ unit weights.
pub(crate) fn pearson_dispersion(
    y: &[f64],
    mu: &[f64],
    family: Family,
    nb_theta: f64,
    n: usize,
    p: usize,
    prior_w: Option<&[f64]>,
) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        let r = (y[i] - mu[i]) / variance(family, nb_theta, mu[i]).sqrt();
        let pw = prior_w.map_or(1.0, |w| w[i]);
        s += pw * r * r;
    }
    s / (n - p) as f64
}

/// Canonical-link test: logit (binomial) and log (Poisson) are the links whose
/// IRLS weight collapses to the simplified Newton form (`irls_weight_and_resid`)
/// and whose PIRLS exit overshoots to machine precision at the standard
/// tolerance (`glmm::pirls_tol`) — both keyed off this same set.
pub(crate) fn is_canonical(family: Family) -> bool {
    matches!(
        family,
        Family::Binomial {
            link: BinomialLink::Logit
        } | Family::Poisson { .. }
    )
}

/// IRLS triple `(μ, W, working_residual)` at the current η. For **canonical**
/// links (logit, Poisson-log) the simplified form `W=V(μ)`, `r=(y−μ)/V(μ)`; for
/// **non-canonical** links (probit, Gamma-log/inverse, NB-log) the general
/// Fisher-scoring form `W=(dμ/dη)²/V(μ)`, `r=(y−μ)·dη/dμ`. `φ` folded as 1. The
/// working response the caller forms is `z = η + r`; weights are raw (caller
/// floors with `glm::WEIGHT_CLAMP`).
pub(crate) fn irls_weight_and_resid(
    family: Family,
    nb_theta: f64,
    y: f64,
    eta: f64,
) -> (f64, f64, f64) {
    let mu = link_inv(family, eta);
    let v = variance(family, nb_theta, mu);
    if is_canonical(family) {
        // dμ/dη = V(μ) here, so the general form collapses to this shortcut.
        (mu, v, (y - mu) / v)
    } else {
        let dm = mu_eta(family, eta);
        (mu, dm * dm / v, (y - mu) / dm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinomialLink, Family, GammaLink, NegBinomialLink, PoissonLink};

    #[test]
    fn poisson_log_canonical_quantities() {
        let f = Family::Poisson {
            link: PoissonLink::Log,
        };
        let eta = 0.5_f64;
        let mu = link_inv(f, eta);
        assert!((mu - eta.exp()).abs() < 1e-12); // g⁻¹ = exp
        assert!((variance(f, f64::NAN, mu) - mu).abs() < 1e-12); // V(μ)=μ
                                                                 // canonical: w = V(μ) = μ; working resid = (y−μ)/μ
        let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 3.0, eta);
        assert!((m - mu).abs() < 1e-12 && (w - mu).abs() < 1e-12);
        assert!((r - (3.0 - mu) / mu).abs() < 1e-12);
    }

    #[test]
    fn gamma_log_noncanonical_weight() {
        let f = Family::Gamma {
            link: GammaLink::Log,
        };
        let eta = 0.2_f64;
        let mu = eta.exp();
        // log link on Gamma is non-canonical: dμ/dη=μ, V=μ² → w=μ²/μ²=1
        let (_m, w, r) = irls_weight_and_resid(f, f64::NAN, 1.0, eta);
        assert!((w - 1.0).abs() < 1e-12, "w={w}");
        assert!((r - (1.0 - mu) / mu).abs() < 1e-12); // (y−μ)·dη/dμ = (y−μ)/μ
    }

    #[test]
    fn gamma_inverse_residual_sign() {
        let f = Family::Gamma {
            link: GammaLink::Inverse,
        };
        let eta = 0.5_f64; // η>0 required; μ=1/η=2
        let mu = 1.0 / eta;
        // dμ/dη=−μ², V=μ² → w=(μ²)²/μ²=μ²; resid=(y−μ)·dη/dμ=−(y−μ)/μ²
        let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 3.0, eta);
        assert!((m - mu).abs() < 1e-12 && (w - mu * mu).abs() < 1e-12);
        assert!((r - (-(3.0 - mu) / (mu * mu))).abs() < 1e-12, "r={r}");
    }

    #[test]
    fn poisson_deviance_resid_zero_at_fit() {
        let f = Family::Poisson {
            link: PoissonLink::Log,
        };
        // d_i = 2[ y log(y/μ) − (y−μ) ]; at y=μ → 0
        assert!(dev_resid(f, f64::NAN, 4.0, 4.0).abs() < 1e-10);
        assert!(dev_resid(f, f64::NAN, 4.0, 2.0) > 0.0);
        // y=0, μ=1: t=0 (0·ln0→0 limit), so d = 2[0 − (0−1)] = 2.0 exactly.
        assert!((dev_resid(f, f64::NAN, 0.0, 1.0) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn nb_variance_uses_theta() {
        let f = Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        };
        // V(μ) = μ + μ²/θ, with θ̂ threaded explicitly.
        let mu = 3.0;
        assert!((variance(f, 2.0, mu) - (mu + mu * mu / 2.0)).abs() < 1e-12);
    }

    #[test]
    fn probit_mu_eta_is_normal_pdf() {
        let f = Family::Binomial {
            link: BinomialLink::Probit,
        };
        // g⁻¹=Φ (via the high-precision `phi_hp`, ~1e-15), dμ/dη=φ(η) exact pdf.
        assert!((link_inv(f, 0.0) - 0.5).abs() < 1e-13);
        assert!((mu_eta(f, 0.0) - (1.0 / (2.0 * std::f64::consts::PI).sqrt())).abs() < 1e-12);
    }
}
