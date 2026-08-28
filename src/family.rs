//! Family/link IRLS math primitives for the **new** M3 outcome families
//! (Poisson, Gamma, negative-binomial, and the binomial probit link).
//!
//! Single source of the four McCullagh–Nelder (1989) GLM quantities per
//! `(family, link)` — inverse link, link derivative, variance function, and
//! per-observation deviance — plus the assembled IRLS weight / working residual.
//! `match`-dispatched, no `dyn`.
//!
//! These are the **scalar statement** of the math, and the reference the
//! vectorized arms of `simd_transcendental::family_pass` are written against;
//! the fit path itself goes through that batched kernel rather than calling
//! these per row. Where an arm's μ differs from the function here it is from the
//! owned SIMD `exp` standing in for libm's: a couple of ULP on the log-link arms,
//! and up to the 5 ULP `erfc_blend_accuracy_and_head_tail_identity` pins on the
//! probit arm, whose blend composes two owned `exp`s through a product. The
//! unweighted-Bernoulli-logit route is the exception: `family_pass` hands it
//! straight to `pw_and_log1pexp_sum`, so it computes none of these quantities.
//!
//! Convention: weights/residuals are the working-response IRLS form of MN89
//! (`z = η + (y−μ)·dη/dμ`, `W = (dμ/dη)²/(φ·V(μ))`), with `φ` folded as 1 here —
//! Gamma/NB dispersion scales the SE post-fit / via the deviance, not the weight.
//! Weights are returned raw; the IRLS caller applies the
//! `glm::WEIGHT_CLAMP` floor.

use crate::spec::{BinomialLink, Family, GammaLink, InverseGaussianLink};

/// `exp(η)` stays finite up to η≈709; clamp short of it so log-link μ never
/// overflows to `inf` mid-IRLS.
pub(crate) const ETA_MAX: f64 = 700.0;
/// Floor for log/inverse-link μ so `V(μ)` and the working residual never divide
/// by zero (the IRLS weight floor `glm::WEIGHT_CLAMP` is the downstream guard).
pub(crate) const MU_FLOOR: f64 = 1e-10;
/// Binomial μ is kept in `(PROB_EPS, 1−PROB_EPS)` so probit deviance/weights stay
/// finite at the saturated ends.
pub(crate) const PROB_EPS: f64 = 1e-12;
/// `1/√(2π)` — the standard-normal pdf normalizer (probit `dμ/dη`).
pub(crate) const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Clamp η to each link's safe domain: `±ETA_MAX` for log links (Poisson,
/// Gamma-log, Negative-Binomial, Inverse-Gaussian-log — overflow guard only);
/// `[MU_FLOOR, ETA_MAX]` for the Gamma inverse link and the Inverse-Gaussian
/// `InverseSquared` link (both need `η>0`, `μ=1/η` and `μ=η^(−1/2)`
/// respectively); `[−ETA_MAX, ln ETA_MAX]` for binomial cloglog (`link_inv`
/// evaluates `exp(exp(η))`, which overflows above `η = ln(ETA_MAX)`). Logit,
/// probit, and Gaussian bound their own range internally or need none, so η
/// passes through unclamped.
pub(crate) fn clamp_eta(family: Family, eta: f64) -> f64 {
    match family {
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        }
        | Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => eta.clamp(MU_FLOOR, ETA_MAX),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => eta.clamp(-ETA_MAX, ETA_MAX),
        // Logit (`sigmoid_stable`) and probit (`phi_hp`) each bound their own
        // range internally, so η passes through. Cloglog does not: `link_inv`
        // evaluates exp(exp(η)), which overflows above η = ln(ETA_MAX) ≈ 6.55.
        Family::Binomial {
            link: BinomialLink::Logit | BinomialLink::Probit,
        }
        | Family::Gaussian => eta,
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => eta.clamp(-ETA_MAX, ETA_MAX.ln()),
    }
}

/// True iff η lies outside the link's OPEN domain — the Gamma inverse link
/// (μ = 1/η needs η > 0) and the inverse-Gaussian `InverseSquared` link
/// (μ = η^(−1/2) needs η > 0) are the two with one; every other family/link's
/// η domain is all of ℝ, where [`clamp_eta`]'s bounds are overflow guards, not
/// domain edges, so this is constant-false there. PIRLS treats a trial iterate that
/// violates this as a failed step and halves toward the last accepted feasible
/// iterate (R `glm.fit`'s `valideta` step-halving), because letting
/// [`clamp_eta`]'s boundary projection stand would let the solve converge ON
/// the boundary: at η = MU_FLOOR the working weight is μ² ≈ 1e20, the pinned
/// row dominates the WLS solve, and PIRLS reports a spuriously converged
/// boundary answer (measured on the Gamma-inverse `sim_gamma` cell: a ~98-unit
/// deviance cliff in the θ surface, one clamped row carrying all of it).
pub(crate) fn eta_infeasible(family: Family, eta: f64) -> bool {
    matches!(
        family,
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } | Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        }
    ) && eta <= 0.0
}

/// Clamp μ to each family's valid domain: `≥ MU_FLOOR` for Poisson/Gamma/NB
/// (positive mean), `(PROB_EPS, 1−PROB_EPS)` for binomial. Gaussian passes through.
pub(crate) fn clamp_mu(family: Family, mu: f64) -> f64 {
    match family {
        Family::Binomial { .. } => mu.clamp(PROB_EPS, 1.0 - PROB_EPS),
        Family::Poisson { .. }
        | Family::Gamma { .. }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian { .. } => mu.max(MU_FLOOR),
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
        // μ = 1 − exp(−exp η). `−expm1(−t)` rather than `1 − exp(−t)` so small μ
        // keeps its relative precision (McCullagh–Nelder 1989 §4.3.1).
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => -((-eta.exp()).exp_m1()),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => eta.exp(),
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => 1.0 / eta,
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => 1.0 / eta.sqrt(),
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
        // dμ/dη = exp(η − exp η). Bounded above by e⁻¹ at η=0 and underflowing at
        // both tails; the `glm::WEIGHT_CLAMP` floor is the downstream guard, as
        // for probit.
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => (eta - eta.exp()).exp(),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => eta.exp(),
        // μ=1/η ⇒ dμ/dη = −1/η² = −μ².
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => {
            let mu = 1.0 / eta;
            -mu * mu
        }
        // μ=η^(−1/2) ⇒ dμ/dη = −½·η^(−3/2) = −μ³/2.
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => {
            let mu = 1.0 / eta.sqrt();
            -0.5 * mu * mu * mu
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
        Family::InverseGaussian { .. } => mu * mu * mu,
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
        // 2[ y ln(y/μ) − (y−μ) ], y·ln(y/μ)→0 at y=0. Written y·(ln y − ln μ), not
        // y·ln(y/μ): for subnormal y and μ ≥ 2 the quotient y/μ underflows to exactly
        // 0.0 before the log, punching a −inf hole in the objective.
        Family::Poisson { .. } => {
            let t = if y > 0.0 { y * (y.ln() - mu.ln()) } else { 0.0 };
            2.0 * (t - (y - mu))
        }
        // 2[ −ln(y/μ) + (y−μ)/μ ]; same form for log and inverse links.
        Family::Gamma { .. } => 2.0 * (-(y / mu).ln() + (y - mu) / mu),
        // 2[ y ln(y/μ) − (y+θ) ln((y+θ)/(μ+θ)) ], y·ln(y/μ)→0 at y=0; θ = nb_theta.
        // Same subtraction form as Poisson — see the subnormal-underflow note above.
        Family::NegativeBinomial { .. } => {
            let t = if y > 0.0 { y * (y.ln() - mu.ln()) } else { 0.0 };
            2.0 * (t - (y + nb_theta) * ((y + nb_theta) / (mu + nb_theta)).ln())
        }
        // dᵢ = (yᵢ−μᵢ)²/(μᵢ²·yᵢ) (McCullagh–Nelder 1989 §2.2.4; R
        // inverse.gaussian()$dev.resids). Requires y>0.
        Family::InverseGaussian { .. } => {
            let r = y - mu;
            r * r / (mu * mu * y)
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

/// R's `inverse.gaussian()$aic` — the family's `−2·logLik + 2` with the
/// dispersion **profiled** as `disp = D/Σwᵢ` rather than carried as a free
/// parameter, the same convention `gamma_aic` follows:
/// ```text
///   logLik = −½ Σᵢ wᵢ·[ (yᵢ−μᵢ)²/(μᵢ²·yᵢ·φ) + ln(2π·φ·yᵢ³) ]
/// ```
/// Substituting `Σᵢ wᵢ(yᵢ−μᵢ)²/(μᵢ²yᵢ) = D` and `φ = D/Σwᵢ` collapses the first
/// sum to `Σwᵢ`, leaving
/// ```text
///   aic = Σwᵢ·(ln(2π·disp) + 1) + 3·Σᵢ wᵢ·ln yᵢ + 2
/// ```
/// which is R's expression verbatim (R `src/library/stats/R/family.R`,
/// `inverse.gaussian()$aic`). `μ` enters only through `dev`, so it is not a
/// parameter here. Requires `y > 0`, the family's own domain.
pub(crate) fn inv_gaussian_aic(y: &[f64], dev: f64, n: usize, prior_w: Option<&[f64]>) -> f64 {
    let sum_w = prior_w.map_or(n as f64, |w| w[..n].iter().sum());
    let disp = dev / sum_w;
    let mut ln_y = 0.0;
    for (i, &yi) in y.iter().take(n).enumerate() {
        ln_y += prior_w.map_or(1.0, |w| w[i]) * yi.ln();
    }
    sum_w * ((2.0 * std::f64::consts::PI * disp).ln() + 1.0) + 3.0 * ln_y + 2.0
}

/// Saturated log-likelihood `Σᵢ log f(yᵢ; μ=yᵢ)` — the data-only constant the
/// deviance convention drops: `−2·logLik = Σᵢ wᵢ·dᵢ − 2·saturated_loglik`, so
/// `logLik = −½·deviance + saturated_loglik` wherever the reported deviance is
/// the (weighted) `dev_resid` sum. Per family:
///
/// - **Binomial** — the aggregated form: row `i` is `mᵢ = wᵢ` trials with
///   `sᵢ = wᵢ·yᵢ` successes (unit weights ⇒ Bernoulli, where every term is 0),
///   so the saturated density at `μ=y` is `ln C(mᵢ,sᵢ) + sᵢ·ln yᵢ +
///   (mᵢ−sᵢ)·ln(1−yᵢ)` with the binomial coefficient via `lnΓ` (continuous in
///   `wᵢ`; R's `dbinom` rounds — identical on the integer trial counts the
///   aggregated convention carries).
/// - **Poisson** — `wᵢ·(yᵢ·ln yᵢ − yᵢ − lnΓ(yᵢ+1))` (0 at `yᵢ=0`).
/// - **NegativeBinomial** — `nb_profile_loglik(y, y, θ, w) − Σᵢ wᵢ·lnΓ(yᵢ+1)`
///   (the θ-dependent saturated normalizer plus the count term that profile
///   deliberately omits).
/// - **Gaussian** — 0 (its `dev_resid` is the bare RSS; the Gaussian paths
///   build their log-likelihood directly and never call this).
/// - **Gamma** — NaN on purpose: the Gamma objective substitutes `gamma_aic`
///   (already `−2·Σwᵢ·log f + 2`), so its logLik is `−½(deviance − 2)` with no
///   saturated term; a caller reaching this arm is a bug, surfaced as NaN.
/// - **InverseGaussian** — NaN for the same reason as Gamma: the objective
///   substitutes `inv_gaussian_aic` (D1), which already carries the profiled
///   dispersion, so there is no free-standing saturated constant to restore.
pub(crate) fn saturated_loglik(
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: Option<&[f64]>,
) -> f64 {
    let lgamma = crate::simd_transcendental::ln_gamma;
    match family {
        Family::Gaussian => 0.0,
        Family::Gamma { .. } | Family::InverseGaussian { .. } => f64::NAN,
        Family::Binomial { .. } => {
            let mut s = 0.0;
            for (i, &yi) in y.iter().enumerate() {
                let m = prior_w.map_or(1.0, |w| w[i]);
                let succ = m * yi;
                s += lgamma(m + 1.0) - lgamma(succ + 1.0) - lgamma(m - succ + 1.0);
                if yi > 0.0 {
                    s += succ * yi.ln();
                }
                if yi < 1.0 {
                    s += (m - succ) * (1.0 - yi).ln();
                }
            }
            s
        }
        Family::Poisson { .. } => {
            let mut s = 0.0;
            for (i, &yi) in y.iter().enumerate() {
                let t = if yi > 0.0 { yi * yi.ln() } else { 0.0 };
                s += prior_w.map_or(1.0, |w| w[i]) * (t - yi - lgamma(yi + 1.0));
            }
            s
        }
        Family::NegativeBinomial { .. } => {
            let profile = crate::fit::nb_profile_loglik(y, y, nb_theta, prior_w);
            let counts: f64 = y
                .iter()
                .enumerate()
                .map(|(i, &yi)| prior_w.map_or(1.0, |w| w[i]) * lgamma(yi + 1.0))
                .sum();
            profile - counts
        }
    }
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
    use crate::{
        BinomialLink, Family, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink,
    };

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

    /// The identity `Fit.loglik` relies on: `−½·Σwᵢdᵢ + saturated_loglik`
    /// must equal the exact `Σwᵢ·log f(yᵢ; μᵢ)` at ANY μ (not just the fitted
    /// one), per family — checked against directly-written log-densities.
    #[test]
    fn saturated_loglik_restores_exact_densities() {
        let lg = crate::simd_transcendental::ln_gamma;
        let y = [0.0, 1.0, 3.0, 7.0];
        let mu = [0.5, 1.2, 2.5, 6.0];
        let w = [1.0, 2.0, 1.0, 3.0];

        let fp = Family::Poisson {
            link: PoissonLink::Log,
        };
        let dev: f64 = (0..4)
            .map(|i| w[i] * dev_resid(fp, f64::NAN, y[i], mu[i]))
            .sum();
        let direct: f64 = (0..4)
            .map(|i| w[i] * (y[i] * mu[i].ln() - mu[i] - lg(y[i] + 1.0)))
            .sum();
        let restored = -0.5 * dev + saturated_loglik(fp, f64::NAN, &y, Some(&w));
        assert!(
            (restored - direct).abs() < 1e-10,
            "poisson {restored} vs {direct}"
        );

        let th = 1.7;
        let fnb = Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        };
        let dev: f64 = (0..4).map(|i| w[i] * dev_resid(fnb, th, y[i], mu[i])).sum();
        let direct: f64 = (0..4)
            .map(|i| {
                w[i] * (lg(y[i] + th) - lg(th) - lg(y[i] + 1.0)
                    + th * (th / (th + mu[i])).ln()
                    + y[i] * (mu[i] / (th + mu[i])).ln())
            })
            .sum();
        let restored = -0.5 * dev + saturated_loglik(fnb, th, &y, Some(&w));
        assert!(
            (restored - direct).abs() < 1e-10,
            "nb {restored} vs {direct}"
        );

        // Aggregated binomial: y is a success PROPORTION, w the trial count m.
        let fb = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let yb = [0.0, 0.5, 2.0 / 3.0, 1.0];
        let m = [2.0, 4.0, 3.0, 5.0];
        let mub = [0.3, 0.55, 0.6, 0.8];
        let dev: f64 = (0..4)
            .map(|i| m[i] * dev_resid(fb, f64::NAN, yb[i], mub[i]))
            .sum();
        let direct: f64 = (0..4)
            .map(|i| {
                let s = m[i] * yb[i];
                lg(m[i] + 1.0) - lg(s + 1.0) - lg(m[i] - s + 1.0)
                    + s * mub[i].ln()
                    + (m[i] - s) * (1.0 - mub[i]).ln()
            })
            .sum();
        let restored = -0.5 * dev + saturated_loglik(fb, f64::NAN, &yb, Some(&m));
        assert!(
            (restored - direct).abs() < 1e-10,
            "binomial {restored} vs {direct}"
        );
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

    #[test]
    fn cloglog_link_and_derivative() {
        let f = Family::Binomial {
            link: BinomialLink::Cloglog,
        };
        // μ = 1 − exp(−exp η); at η=0 that is 1 − e⁻¹.
        let mu0 = link_inv(f, 0.0);
        assert!((mu0 - (1.0 - (-1.0f64).exp())).abs() < 1e-15, "μ(0)={mu0}");
        // dμ/dη = exp(η − exp η); max at η=0, value e⁻¹.
        let d0 = mu_eta(f, 0.0);
        assert!((d0 - (-1.0f64).exp()).abs() < 1e-15, "dμ/dη(0)={d0}");
        // Asymmetric: μ approaches 1 much faster than 0.
        assert!(link_inv(f, 2.0) > 0.999);
        assert!(link_inv(f, -2.0) < 0.13);
        // General Fisher weight (dμ/dη)²/V(μ), V = μ(1−μ).
        let eta = 0.4_f64;
        let mu = link_inv(f, eta);
        let dm = mu_eta(f, eta);
        let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 1.0, eta);
        assert!((m - mu).abs() < 1e-15);
        assert!((w - dm * dm / (mu * (1.0 - mu))).abs() < 1e-12, "w={w}");
        assert!((r - (1.0 - mu) / dm).abs() < 1e-12, "r={r}");
    }

    #[test]
    fn cloglog_eta_is_clamped_above_at_ln_eta_max() {
        let f = Family::Binomial {
            link: BinomialLink::Cloglog,
        };
        // η above ln(ETA_MAX) would overflow exp(exp(η)) — clamped, so μ stays
        // finite and inside the binomial μ-domain.
        let mu = link_inv(f, 1e6);
        assert!(mu.is_finite() && mu <= 1.0 - PROB_EPS, "μ={mu}");
        assert!(mu_eta(f, 1e6).is_finite());
        // Logit and probit are untouched by the clamp split.
        let logit = Family::Binomial {
            link: BinomialLink::Logit,
        };
        assert_eq!(clamp_eta(logit, 1e6), 1e6);
    }

    #[test]
    fn inverse_gaussian_inverse_squared_quantities() {
        let f = Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        };
        let eta = 0.25_f64; // η>0 required; μ = η^(−1/2) = 2
        let mu = link_inv(f, eta);
        assert!((mu - 2.0).abs() < 1e-12, "μ={mu}");
        // V(μ) = μ³
        assert!((variance(f, f64::NAN, mu) - 8.0).abs() < 1e-12);
        // dμ/dη = −½·η^(−3/2) = −μ³/2
        let dm = mu_eta(f, eta);
        assert!((dm - (-4.0)).abs() < 1e-12, "dμ/dη={dm}");
        // General branch: w = (dμ/dη)²/V = μ³/4; resid = (y−μ)/(dμ/dη)
        let (m, w, r) = irls_weight_and_resid(f, f64::NAN, 3.0, eta);
        assert!((m - mu).abs() < 1e-12 && (w - 2.0).abs() < 1e-12, "w={w}");
        assert!((r - (3.0 - 2.0) / -4.0).abs() < 1e-12, "r={r}");
        // η ≤ 0 is outside the OPEN domain, like Gamma-inverse.
        assert!(eta_infeasible(f, 0.0));
        assert!(eta_infeasible(f, -1.0));
        assert!(!eta_infeasible(f, 1e-3));
    }

    #[test]
    fn inverse_gaussian_log_quantities() {
        let f = Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        };
        let eta = 0.7_f64;
        let mu = eta.exp();
        assert!((link_inv(f, eta) - mu).abs() < 1e-12);
        // dμ/dη = μ, V = μ³ → w = μ²/μ³ = 1/μ; resid = (y−μ)/μ
        let (_m, w, r) = irls_weight_and_resid(f, f64::NAN, 4.0, eta);
        assert!((w - 1.0 / mu).abs() < 1e-12, "w={w}");
        assert!((r - (4.0 - mu) / mu).abs() < 1e-12, "r={r}");
        assert!(!eta_infeasible(f, -50.0)); // log link's η domain is all of ℝ
    }

    #[test]
    fn inverse_gaussian_deviance_resid() {
        let f = Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        };
        // dᵢ = (y−μ)²/(μ² y); zero at y=μ, positive otherwise.
        assert!(dev_resid(f, f64::NAN, 2.0, 2.0).abs() < 1e-14);
        let d = dev_resid(f, f64::NAN, 4.0, 2.0);
        assert!(
            (d - (4.0 - 2.0f64).powi(2) / (4.0 * 4.0)).abs() < 1e-14,
            "d={d}"
        );
        assert!(d > 0.0);
    }

    #[test]
    fn inverse_gaussian_saturated_loglik_is_nan() {
        // Like Gamma: the objective substitutes `inv_gaussian_aic`, which already
        // carries the profiled dispersion, so there is no saturated constant to
        // restore. A caller reaching this arm is a bug, surfaced as NaN.
        let f = Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        };
        assert!(saturated_loglik(f, f64::NAN, &[1.0, 2.0], None).is_nan());
    }

    #[test]
    fn inv_gaussian_aic_matches_r_formula() {
        // R: aic = sum(wt)*(log(dev/sum(wt)*2*pi)+1) + 3*sum(log(y)*wt) + 2
        let y = [1.5_f64, 2.0, 0.75, 3.25];
        let n = y.len();
        let dev = 0.42_f64;
        let got = inv_gaussian_aic(&y, dev, n, None);
        let disp = dev / n as f64;
        let want = n as f64 * ((2.0 * std::f64::consts::PI * disp).ln() + 1.0)
            + 3.0 * y.iter().map(|v| v.ln()).sum::<f64>()
            + 2.0;
        assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        // Prior weights enter as Σw in place of n and weight each log y.
        let w = [1.0_f64, 2.0, 0.5, 1.5];
        let sw: f64 = w.iter().sum();
        let gotw = inv_gaussian_aic(&y, dev, n, Some(&w));
        let dispw = dev / sw;
        let wantw = sw * ((2.0 * std::f64::consts::PI * dispw).ln() + 1.0)
            + 3.0 * y.iter().zip(w).map(|(v, wi)| wi * v.ln()).sum::<f64>()
            + 2.0;
        assert!((gotw - wantw).abs() < 1e-12);
    }
}
