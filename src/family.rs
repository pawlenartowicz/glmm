//! Family/link IRLS math primitives for the Poisson, Gamma, negative-binomial,
//! and binomial-probit outcome families.
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

use crate::scalar::Scalar;
use crate::spec::{BinomialLink, Family, GammaLink, InverseGaussianLink};

/// `exp(η)` stays finite up to η≈709; the log links and cloglog clamp η short of
/// it so `e^η` never overflows to `inf` mid-IRLS.
pub(crate) const ETA_MAX: f64 = 700.0;
/// Floor for a data value used as a starting or null mean on a positive-mean
/// family ([`clamp_mu`]), so a zero response cannot seed `ln 0` or `1/0`. The
/// fitted μ itself carries no floor: every row quantity stays finite without
/// one (see [`in_tail`]).
pub(crate) const MU_FLOOR: f64 = 1e-10;
/// Binomial μ bound: probit and cloglog store μ inside `[PROB_EPS, 1−PROB_EPS]`
/// so `V(μ) = μ(1−μ)` is never 0 in a μ-form formula, and on every binomial link
/// a row with μ at or past it is a tail row ([`in_tail`]) whose deviance,
/// score and weights are computed from η instead.
pub(crate) const PROB_EPS: f64 = 1e-12;
/// Log-link tail: past `|η| = 230` (`μ` outside `[1.3e-100, 7.7e99]`) the
/// μ-form variance of the inverse-Gaussian (`μ³`), Gamma (`μ²`) and NB
/// (`μ + μ²/θ`) links is near its overflow or underflow, so the NB,
/// Gamma-log and IG-log rows there switch to the ratio forms of [`in_tail`].
pub(crate) const LOG_TAIL_ETA: f64 = 230.0;
/// η bounds of the Gamma inverse and inverse-Gaussian `InverseSquared` links:
/// both need `η > 0`, and inside `[1e-75, 1e75]` their μ-form weights stay
/// finite (Gamma-inverse `(dμ/dη)² = μ⁴`, μ = 1/η; IG `μ⁶/4`, μ = η^(−1/2)).
pub(crate) const INV_ETA_MIN: f64 = 1e-75;
/// Upper twin of [`INV_ETA_MIN`].
pub(crate) const INV_ETA_MAX: f64 = 1e75;
/// `1/√(2π)` — the standard-normal pdf normalizer (probit `dμ/dη`).
pub(crate) const FRAC_1_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// Clamp η to each link's safe domain: `±ETA_MAX` for the log links (Poisson,
/// Gamma-log, Negative-Binomial, Inverse-Gaussian-log) and binomial cloglog
/// (`e^η` stays finite); `[INV_ETA_MIN, INV_ETA_MAX]` for the Gamma inverse
/// link and the Inverse-Gaussian `InverseSquared` link (both need `η>0`,
/// `μ=1/η` and `μ=η^(−1/2)` respectively). Logit, probit, and Gaussian need
/// none, so η passes through unclamped. Past a bound the row's deviance stops
/// moving, so a PIRLS trial there is refused where the row's data pull it back
/// ([`eta_infeasible`]).
pub(crate) fn clamp_eta<T: Scalar>(family: Family, eta: T) -> T {
    let (lo, hi) = clamp_eta_bounds(family);
    match family {
        Family::Binomial {
            link: BinomialLink::Logit | BinomialLink::Probit,
        }
        | Family::Gaussian => eta,
        _ => eta.clamp_f64(lo, hi),
    }
}

/// The `(lo, hi)` bounds [`clamp_eta`] holds η inside, per family. Also read
/// by a caller asking "did the clamp bind on this row?" after the fact:
/// `glmm::assembled::eta_clamped_rows`.
pub(crate) fn clamp_eta_bounds(family: Family) -> (f64, f64) {
    match family {
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        }
        | Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => (INV_ETA_MIN, INV_ETA_MAX),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        }
        | Family::Binomial {
            link: BinomialLink::Cloglog,
        } => (-ETA_MAX, ETA_MAX),
        Family::Binomial {
            link: BinomialLink::Logit | BinomialLink::Probit,
        }
        | Family::Gaussian => (f64::NEG_INFINITY, f64::INFINITY),
    }
}

/// True iff a raw η lies past [`clamp_eta`]'s bounds where the row's data pull
/// it back. Past a bound the row's η, and so its deviance, stops moving, so
/// the objective is flat there while the true one keeps rising. PIRLS treats
/// a trial iterate with such a row as a failed step and halves toward the last
/// accepted iterate (R `glm.fit`'s `valideta` step-halving), and the exit
/// refresh refuses such a point, because letting [`clamp_eta`]'s projection
/// stand would let the solve converge ON the bound. Per bound:
///
/// - the Gamma inverse and inverse-Gaussian `InverseSquared` links, either
///   bound, whatever y: the lower one holds their open domain's edge η ≤ 0,
///   and past either the deviance grows without limit. On these links a
///   pinned row also dominates the WLS solve (measured on the Gamma-inverse
///   `sim_gamma` cell with a row pinned at η = 1e-10: a ~98-unit deviance
///   cliff in the θ surface, one clamped row carrying all of it);
/// - the log links, the upper bound whatever y, and the lower bound when
///   y > 0; a y = 0 row there is at its deviance's own limit (μ → 0), where
///   the true objective is flat too — the all-zero cluster of a large RE
///   variance runs there;
/// - cloglog, the lower bound when y > 0 and the upper bound when y < 1, by
///   the same reading.
///
/// Constant-false on logit, probit and Gaussian, which have no bounds, and
/// false on a NaN η. GLM IRLS reads its own guards instead
/// (`glm::ETA_DIVERGENCE_CAP` and the non-finite checks).
pub(crate) fn eta_infeasible<T: Scalar>(family: Family, y: f64, eta: T) -> bool {
    let (lo, hi) = clamp_eta_bounds(family);
    let e = eta.value();
    match family {
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => (e < lo && y > 0.0) || (e > hi && y < 1.0),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => (e < lo && y > 0.0) || e > hi,
        _ => e < lo || e > hi,
    }
}

/// Floor a data value used as a starting or null mean into its family's μ
/// domain: `≥ MU_FLOOR` for Poisson/Gamma/NB/inverse-Gaussian (positive mean),
/// `[PROB_EPS, 1−PROB_EPS]` for binomial. Gaussian passes through. The fit's
/// own μ never goes through this: [`link_inv`] bounds only the probit and
/// cloglog μ, at `PROB_EPS`.
pub(crate) fn clamp_mu<T: Scalar>(family: Family, mu: T) -> T {
    match family {
        Family::Binomial { .. } => mu.clamp_f64(PROB_EPS, 1.0 - PROB_EPS),
        Family::Poisson { .. }
        | Family::Gamma { .. }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian { .. } => mu.max_f64(MU_FLOOR),
        Family::Gaussian => mu,
    }
}

/// Inverse link `g⁻¹(η) → μ` after [`clamp_eta`]. Probit and cloglog store μ
/// inside `[PROB_EPS, 1−PROB_EPS]` so `V(μ)` is never 0 in a μ-form formula; a
/// row on that bound is a tail row ([`in_tail`]) whose other quantities come
/// from η. Every other link returns the exact `g⁻¹`: `e^η` on `|η| ≤ ETA_MAX`
/// is never 0 or `inf`, the inverse links' η bounds keep μ inside
/// `[1e-75, 1e75]`, and logit's `σ(η)` enters no μ-form division on a tail row.
pub(crate) fn link_inv<T: Scalar>(family: Family, eta: T) -> T {
    let eta = clamp_eta(family, eta);
    match family {
        Family::Gaussian => eta,
        Family::Binomial {
            link: BinomialLink::Logit,
        } => eta.sigmoid(),
        Family::Binomial {
            link: BinomialLink::Probit,
        } => eta.probit_cdf().clamp_f64(PROB_EPS, 1.0 - PROB_EPS),
        // μ = 1 − exp(−exp η). `−expm1(−t)` rather than `1 − exp(−t)` so small μ
        // keeps its relative precision (McCullagh–Nelder 1989 §4.3.1).
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => (-((-Scalar::exp(eta)).exp_m1())).clamp_f64(PROB_EPS, 1.0 - PROB_EPS),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => Scalar::exp(eta),
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => T::ONE / eta,
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => T::ONE / eta.sqrt(),
    }
}

/// Link derivative `dμ/dη` at η. Used by the general Fisher-scoring weight and
/// working residual for the non-canonical links.
pub(crate) fn mu_eta<T: Scalar>(family: Family, eta: T) -> T {
    let eta = clamp_eta(family, eta);
    match family {
        Family::Gaussian => T::ONE,
        Family::Binomial {
            link: BinomialLink::Logit,
        } => {
            let mu = eta.sigmoid();
            mu * (T::ONE - mu)
        }
        Family::Binomial {
            link: BinomialLink::Probit,
        } => T::from_f64(FRAC_1_SQRT_2PI) * (T::from_f64(-0.5) * eta * eta).exp(),
        // dμ/dη = exp(η − exp η). Bounded above by e⁻¹ at η=0 and underflowing at
        // both tails; the `glm::WEIGHT_CLAMP` floor is the downstream guard, as
        // for probit.
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => (eta - Scalar::exp(eta)).exp(),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => Scalar::exp(eta),
        // μ=1/η ⇒ dμ/dη = −1/η² = −μ².
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => {
            let mu = T::ONE / eta;
            -mu * mu
        }
        // μ=η^(−1/2) ⇒ dμ/dη = −½·η^(−3/2) = −μ³/2.
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => {
            let mu = T::ONE / eta.sqrt();
            T::from_f64(-0.5) * mu * mu * mu
        }
    }
}

/// Second derivative `d²μ/dη²` at η. Every arm is `mu_eta(family, eta) · g(η)`,
/// the same `w·g(η, μ)` shape [`weight_eta_deriv`] has, with `μ'` and `g` both
/// read at the unbounded μ, never at the `PROB_EPS`-bounded value [`link_inv`]
/// stores for probit and cloglog. Applies [`clamp_eta`] at the top exactly as
/// [`mu_eta`] does, so the two agree row for row. Per link, with
/// `σ = η.sigmoid()` and `φ = mu_eta` the probit pdf:
///
/// - Gaussian identity: `μ'' = 0`.
/// - Binomial logit: `μ' = σ(1−σ)`, `g = 1 − 2σ`, `μ'' = σ(1−σ)(1−2σ)`.
/// - Binomial probit: `μ' = φ`, `g = −η`, `μ'' = −η·φ`.
/// - Binomial cloglog: `μ' = e^{η−e^η}`, `g = 1 − e^η`, `μ'' = e^{η−e^η}(1−e^η)`.
/// - Poisson-log, Gamma-log, NB-log, InverseGaussian-log: `μ' = μ = e^η`,
///   `g = 1`, `μ'' = e^η`.
/// - Gamma inverse: `μ_r = 1/η`, `μ' = −μ_r²`, `g = −2μ_r`, `μ'' = 2μ_r³ = 2/η³`.
/// - InverseGaussian inverse-squared: `μ_r = η^(−1/2)`, `μ' = −μ_r³/2`,
///   `g = −1.5μ_r²`, `μ'' = 0.75·μ_r⁵ = (3/4)·η^(−5/2)`.
pub(crate) fn mu_eta_eta<T: Scalar>(family: Family, eta: T) -> T {
    let eta = clamp_eta(family, eta);
    let dm = mu_eta(family, eta);
    match family {
        Family::Gaussian => T::ZERO,
        Family::Binomial {
            link: BinomialLink::Logit,
        } => {
            let mu = eta.sigmoid();
            dm * (T::ONE - T::from_f64(2.0) * mu)
        }
        Family::Binomial {
            link: BinomialLink::Probit,
        } => -eta * dm,
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => dm * (T::ONE - Scalar::exp(eta)),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => dm,
        Family::Gamma {
            link: GammaLink::Inverse,
            ..
        } => {
            let mu_r = T::ONE / eta;
            dm * T::from_f64(-2.0) * mu_r
        }
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => {
            let mu_r = T::ONE / eta.sqrt();
            dm * T::from_f64(-1.5) * mu_r * mu_r
        }
    }
}

/// Family variance function `V(μ)`. `nb_theta` is the NB dispersion θ̂ the fit's
/// outer loop fixes for this evaluation — read only by the NB arm; every other
/// family ignores it (pass `f64::NAN`).
pub(crate) fn variance<T: Scalar>(family: Family, nb_theta: f64, mu: T) -> T {
    match family {
        Family::Gaussian => T::ONE,
        Family::Binomial { .. } => mu * (T::ONE - mu),
        Family::Poisson { .. } => mu,
        Family::Gamma { .. } => mu * mu,
        Family::NegativeBinomial { .. } => mu + mu * mu / T::from_f64(nb_theta),
        Family::InverseGaussian { .. } => mu * mu * mu,
    }
}

/// Per-observation deviance contribution `dᵢ ≥ 0` (`Σ dᵢ` is the GLM deviance,
/// −2·log-likelihood up to the saturated constant). Zero at `y=μ`.
pub(crate) fn dev_resid<T: Scalar>(family: Family, nb_theta: f64, y: f64, mu: T) -> T {
    match family {
        Family::Gaussian => {
            let r = T::from_f64(y) - mu;
            r * r
        }
        // 2[ y ln(y/μ) + (1−y) ln((1−y)/(1−μ)) ], with the 0·ln0→0 limits.
        Family::Binomial { .. } => {
            let a = if y > 0.0 {
                T::from_f64(y) * (T::from_f64(y) / mu).ln()
            } else {
                T::ZERO
            };
            let b = if y < 1.0 {
                T::from_f64(1.0 - y) * (T::from_f64(1.0 - y) / (T::ONE - mu)).ln()
            } else {
                T::ZERO
            };
            T::from_f64(2.0) * (a + b)
        }
        // 2[ y ln(y/μ) − (y−μ) ], y·ln(y/μ)→0 at y=0. Written y·(ln y − ln μ), not
        // y·ln(y/μ): for subnormal y and μ ≥ 2 the quotient y/μ underflows to exactly
        // 0.0 before the log, punching a −inf hole in the objective.
        Family::Poisson { .. } => {
            let t = if y > 0.0 {
                T::from_f64(y) * (T::from_f64(y.ln()) - mu.ln())
            } else {
                T::ZERO
            };
            T::from_f64(2.0) * (t - (T::from_f64(y) - mu))
        }
        // 2[ −ln(y/μ) + (y−μ)/μ ]; same form for log and inverse links.
        Family::Gamma { .. } => {
            T::from_f64(2.0) * (-(T::from_f64(y) / mu).ln() + (T::from_f64(y) - mu) / mu)
        }
        // 2[ y ln(y/μ) − (y+θ) ln((y+θ)/(μ+θ)) ], y·ln(y/μ)→0 at y=0; θ = nb_theta.
        // Same subtraction form as Poisson — see the subnormal-underflow note above.
        Family::NegativeBinomial { .. } => {
            let t = if y > 0.0 {
                T::from_f64(y) * (T::from_f64(y.ln()) - mu.ln())
            } else {
                T::ZERO
            };
            T::from_f64(2.0)
                * (t - (T::from_f64(y + nb_theta))
                    * (T::from_f64(y + nb_theta) / (mu + T::from_f64(nb_theta))).ln())
        }
        // dᵢ = (yᵢ−μᵢ)²/(μᵢ²·yᵢ) (McCullagh–Nelder 1989 §2.2.4; R
        // inverse.gaussian()$dev.resids). Requires y>0.
        Family::InverseGaussian { .. } => {
            let r = T::from_f64(y) - mu;
            r * r / (mu * mu * T::from_f64(y))
        }
    }
}

/// Whether a row's deviance, score and weights come from the η forms below
/// instead of the μ-form formulas above. A row is in the tail where the μ
/// form loses it:
///
/// - **Binomial**, μ at or past `PROB_EPS` from 0 or 1: `1 − μ` is no longer
///   representable near 1, and probit's `φ(η)` and `Φ(η)` underflow together
///   near 0. `μ` here is the stored value, so probit and cloglog (bounded by
///   [`link_inv`]) enter at the bound and logit (unbounded) past it.
/// - **NB, Gamma-log, inverse-Gaussian-log**, `|η| > LOG_TAIL_ETA`: `V(μ)`
///   nears overflow or underflow, and with it the μ-form score and weight.
///
/// Poisson, Gaussian and the inverse links have no tail: their μ forms stay
/// finite on the whole clamped η range. The tail forms are the same functions
/// of η as the μ forms, written to stay finite, so the objective keeps growing
/// on a tail row and the score and weights stay its exact derivatives; the
/// two meet at the switch to round-off (to the μ form's own precision on the
/// binomial upper side, where `1 − μ` near `PROB_EPS` carries ~4 digits).
/// Branches on values only.
pub(crate) fn in_tail(family: Family, eta: f64, mu: f64) -> bool {
    match family {
        Family::Binomial { .. } => mu <= PROB_EPS || mu >= 1.0 - PROB_EPS,
        Family::NegativeBinomial { .. }
        | Family::Gamma {
            link: GammaLink::Log,
        }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => eta.abs() > LOG_TAIL_ETA,
        _ => false,
    }
}

/// `(ln Φ(x), λ, λ', λ'')` with `λ = φ/Φ` the inverse Mills ratio, for
/// `|x| ≥ 4√2`, the range a probit tail row reaches (`|η| ≥ 7.03`).
///
/// Small side, `x = −t < 0`: Cody's third `erfc` region
/// (`simd_transcendental::erfc_region3_r`, `s = 2/t²`) gives
/// `erfc(t/√2) = e^{−t²/2}(1 − c)/(√π·t/√2)` with `c = √π·s·R(s)`, so
/// `ln Φ(x) = −t²/2 + ln((1 − c)/(√(2π)·t))` and `λ = t/(1 − c)`, finite for
/// any `t`. Its derivatives are taken off that same expression rather than
/// the Mills-ratio equation `λ' = −λ(x + λ)`, which cancels to `O(1/t²)` and
/// loses the approximant's last digits by a factor near `t⁴`: with
/// `ds/dx = 2s/t`, `1 + λ' = N/D`, `N = √π·s(R + 2sR') + c²`,
/// `D = (1 − c)²`, and `λ'' = (2s/t)(N_s·D − N·D_s)/D²`, no term of which
/// cancels.
///
/// Large side, `x > 0`: `p = Φ(−x) ≤ 8e-9` from the small side,
/// `ln Φ(x) = ln(1 − p) = −p(1 + p/2)` to below round-off,
/// `λ = φ(x)/(1 − p)`, and `λ'`, `λ''` from the Mills-ratio equation, which
/// does not cancel there.
fn probit_side<T: Scalar>(x: T) -> (T, T, T, T) {
    use crate::simd_transcendental::{erfc_region3_r, SQRT_PI};
    if x.value() < 0.0 {
        let t = -x;
        let s = T::from_f64(2.0) / (t * t);
        let (r, r1, r2) = erfc_region3_r(s);
        let sp = T::from_f64(SQRT_PI);
        let two = T::from_f64(2.0);
        let c = sp * s * r;
        let c1 = sp * (r + s * r1);
        let omc = T::ONE - c;
        let lam = t / omc;
        let ln_phi = T::from_f64(-0.5) * t * t + (omc * T::from_f64(FRAC_1_SQRT_2PI) / t).ln();
        let nn = sp * s * (r + two * s * r1) + c * c;
        let nn1 = sp * (r + T::from_f64(5.0) * s * r1 + two * s * s * r2) + two * c * c1;
        let dd = omc * omc;
        let dd1 = -(two * omc * c1);
        let d1 = nn / dd - T::ONE;
        let d2 = two * s / t * (nn1 * dd - nn * dd1) / (dd * dd);
        (ln_phi, lam, d1, d2)
    } else {
        let p = probit_side(-x).0.exp();
        let lam = T::from_f64(FRAC_1_SQRT_2PI) * (T::from_f64(-0.5) * x * x).exp() / (T::ONE - p);
        let d1 = -(lam * (x + lam));
        let d2 = -(d1 * (x + lam)) - lam * (T::ONE + d1);
        (-(p * (T::ONE + T::from_f64(0.5) * p)), lam, d1, d2)
    }
}

/// The per-row quantities of a tail row ([`in_tail`]), per unit prior
/// weight: `score = −½·∂d/∂η`, the Fisher weight `w = μ'²/V`, the observed
/// weight `w_obs = ½·∂²d/∂η²`, and their η-derivatives `w_eta`, `w_obs_eta`.
/// `w_eta` is carried as `w · (ln w)'` so a weight that has underflowed to 0
/// never divides.
struct TailRow<T> {
    score: T,
    w: T,
    w_log_eta: T,
    w_obs: T,
    w_obs_eta: T,
}

/// [`TailRow`] at `(eta, mu)`, `eta` already [`clamp_eta`]'d, `mu` the stored
/// μ (only the log links read it; it is `e^η` there, never bounded).
///
/// Binomial, with `ℓ = y·ln μ + (1−y)·ln(1−μ)`: let `a = (ln μ)'` and
/// `b = −(ln(1−μ))'` with derivatives `a', a'', b', b''`. Then score
/// `y·a − (1−y)·b`, `w = a·b`, `(ln w)' = a'/a + b'/b`, `w_obs = −y·a' + (1−y)·b'`,
/// `w_obs' = −y·a'' + (1−y)·b''`. Per link:
///
/// - logit: `a = σ(−η)`, `b = σ(η)`, `a' = −ab`, `b' = ab`, `a'' = −ab(a−b)`,
///   `b'' = ab(a−b)`.
/// - probit: `a = λ(η)`, `b = λ(−η)` ([`probit_side`]), `a' = λ'(η)`,
///   `b' = −λ'(−η)`, `a'' = λ''(η)`, `b'' = λ''(−η)`.
/// - cloglog, `t = e^η`, `μ = −expm1(−t)`: `a = t·e^{−t}/μ`, `c = t/μ`,
///   `a' = a(1−c)`, `a'' = a'(1−c) − ac(1−a)`, `b = b' = b'' = t`.
///
/// Log links, `μ = e^η`, `z = y/μ`:
///
/// - NB, `A = θ/(θ+μ)`, `B = μ/(θ+μ)`: score `(y−μ)A`, `w = μA`,
///   `(ln w)' = A`, `w_obs = (y+θ)AB`, `w_obs' = (y+θ)AB(A−B)`.
/// - Gamma-log: score `z − 1`, `w = 1`, `(ln w)' = 0`, `w_obs = z`,
///   `w_obs' = −z`.
/// - IG-log: score `(z − 1)/μ`, `w = 1/μ`, `(ln w)' = −1`, `w_obs = (2z − 1)/μ`,
///   `w_obs' = (1 − 4z)/μ`.
fn tail_row<T: Scalar>(family: Family, nb_theta: f64, y: f64, eta: T, mu: T) -> TailRow<T> {
    let yt = T::from_f64(y);
    let binom = |a: T, a1: T, a2: T, b: T, b1: T, b2: T, w_log_eta: T| {
        let yc = T::from_f64(1.0 - y);
        TailRow {
            score: yt * a - yc * b,
            w: a * b,
            w_log_eta,
            w_obs: -(yt * a1) + yc * b1,
            w_obs_eta: -(yt * a2) + yc * b2,
        }
    };
    match family {
        Family::Binomial {
            link: BinomialLink::Logit,
        } => {
            let (a, b) = ((-eta).sigmoid(), eta.sigmoid());
            let ab = a * b;
            let d = a - b;
            binom(a, -ab, -(ab * d), b, ab, ab * d, d)
        }
        Family::Binomial {
            link: BinomialLink::Probit,
        } => {
            let (_, a, a1, a2) = probit_side(eta);
            let (_, b, bm1, b2) = probit_side(-eta);
            binom(a, a1, a2, b, -bm1, b2, b - a - T::from_f64(2.0) * eta)
        }
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => {
            let t = Scalar::exp(eta);
            let m = -((-t).exp_m1());
            let a = t * (-t).exp() / m;
            let c = t / m;
            let a1 = a * (T::ONE - c);
            let a2 = a1 * (T::ONE - c) - a * c * (T::ONE - a);
            binom(a, a1, a2, t, t, t, T::from_f64(2.0) - c)
        }
        Family::NegativeBinomial { .. } => {
            let th = T::from_f64(nb_theta);
            let den = th + mu;
            let (aa, bb) = (th / den, mu / den);
            let c = T::from_f64(y + nb_theta) * aa * bb;
            TailRow {
                score: (yt - mu) * aa,
                w: mu * aa,
                w_log_eta: aa,
                w_obs: c,
                w_obs_eta: c * (aa - bb),
            }
        }
        Family::Gamma {
            link: GammaLink::Log,
        } => {
            let z = yt / mu;
            TailRow {
                score: z - T::ONE,
                w: T::ONE,
                w_log_eta: T::ZERO,
                w_obs: z,
                w_obs_eta: -z,
            }
        }
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => {
            let z = yt / mu;
            let inv = T::ONE / mu;
            TailRow {
                score: (z - T::ONE) * inv,
                w: inv,
                w_log_eta: -T::ONE,
                w_obs: (T::from_f64(2.0) * z - T::ONE) * inv,
                w_obs_eta: (T::ONE - T::from_f64(4.0) * z) * inv,
            }
        }
        _ => unreachable!("in_tail admits only the binomial and NB/Gamma/IG log links"),
    }
}

/// The unit deviance of a tail row, the same function of η as [`dev_resid`]
/// at `μ = g⁻¹(η)`: binomial `2[y(ln y − ln μ) + (1−y)(ln(1−y) − ln(1−μ))]` on
/// `ln μ`, `ln(1−μ)` taken from η (logit `−log1pexp(∓η)`, probit
/// [`probit_side`], cloglog `ln(−expm1(−t))` and `−t`); Gamma-log
/// `2[(ln μ − ln y) + y/μ − 1]`, finite where `y/μ` alone would meet
/// `−ln(y/μ)` as `inf − inf`; IG-log `(y/μ − 1)²/y`, finite where `μ²` would
/// overflow; NB's μ form is already finite there.
fn tail_dev<T: Scalar>(family: Family, nb_theta: f64, y: f64, eta: T, mu: T) -> T {
    let two = T::from_f64(2.0);
    let (ln_mu, ln_q) = match family {
        Family::Binomial {
            link: BinomialLink::Logit,
        } => (-((-eta).log1pexp()), -(eta.log1pexp())),
        Family::Binomial {
            link: BinomialLink::Probit,
        } => (probit_side(eta).0, probit_side(-eta).0),
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => {
            let t = Scalar::exp(eta);
            ((-((-t).exp_m1())).ln(), -t)
        }
        Family::NegativeBinomial { .. } => return dev_resid(family, nb_theta, y, mu),
        Family::Gamma {
            link: GammaLink::Log,
        } => {
            let z = T::from_f64(y) / mu;
            return two * ((mu.ln() - T::from_f64(y.ln())) + (z - T::ONE));
        }
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => {
            let r = T::from_f64(y) / mu - T::ONE;
            return r * r / T::from_f64(y);
        }
        _ => unreachable!("in_tail admits only the binomial and NB/Gamma/IG log links"),
    };
    let mut d = T::ZERO;
    if y > 0.0 {
        d += T::from_f64(y) * (T::from_f64(y.ln()) - ln_mu);
    }
    if y < 1.0 {
        d += T::from_f64(1.0 - y) * (T::from_f64((1.0 - y).ln()) - ln_q);
    }
    two * d
}

/// Unit deviance of a row at its `(η, μ)`: [`dev_resid`] off the tail, the η
/// form ([`in_tail`]) on it. `eta` is the [`clamp_eta`]'d η and `mu` the
/// stored μ the same pass produced. Every deviance fold over a fit's own rows
/// goes through this rather than [`dev_resid`], which stays the μ-form
/// statement for a μ not tied to an η (the null deviance at `ȳ`).
pub(crate) fn dev_resid_at<T: Scalar>(family: Family, nb_theta: f64, y: f64, eta: T, mu: T) -> T {
    if in_tail(family, eta.value(), mu.value()) {
        tail_dev(family, nb_theta, y, eta, mu)
    } else {
        dev_resid(family, nb_theta, y, mu)
    }
}

/// [`dev_resid_at`] from a raw η: [`clamp_eta`], then [`link_inv`]. The AGQ
/// node sums, which build η and never store μ, read the deviance through this.
pub(crate) fn dev_resid_eta<T: Scalar>(family: Family, nb_theta: f64, y: f64, eta: T) -> T {
    let e = clamp_eta(family, eta);
    dev_resid_at(family, nb_theta, y, e, link_inv(family, e))
}

/// The row score `ρ = prior_w·(dμ/dη)(y − μ)/V(μ) = −½·∂(prior_w·d)/∂η` at the
/// row's `(η, μ)`, the right-hand side PIRLS's mode equation `u = M'ρ` sums.
/// Off the tail it is exactly that μ-form expression, left to right; on a
/// tail row ([`in_tail`]) the η form of [`TailRow`], still the derivative of
/// [`dev_resid_at`].
pub(crate) fn row_score<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: f64,
    prior_w: f64,
    eta: T,
    mu: T,
) -> T {
    if in_tail(family, eta.value(), mu.value()) {
        return T::from_f64(prior_w) * tail_row(family, nb_theta, y, eta, mu).score;
    }
    let dmu = mu_eta(family, eta);
    let v = variance(family, nb_theta, mu);
    T::from_f64(prior_w) * dmu * (T::from_f64(y) - mu) / v
}

/// The unit Fisher weight and working residual `(w, r)` of a tail row
/// ([`in_tail`]), `r = score/max(w, WEIGHT_CLAMP)`: past e^{−t}'s underflow
/// (cloglog η > ln 745) or Φ's (probit |η| > 38) w is exactly 0, and dividing
/// by the same floor the callers put on the weight keeps z = η + r finite.
/// The batched family pass fills its tail rows through this after its μ-form
/// arm, and [`irls_weight_and_resid`] returns it on the tail.
pub(crate) fn tail_weight_and_resid<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: f64,
    eta: T,
    mu: T,
) -> (T, T) {
    let t = tail_row(family, nb_theta, y, eta, mu);
    (t.w, t.score / t.w.max_f64(crate::glm::WEIGHT_CLAMP))
}

/// The precision-weight normaliser `s`: the power of two closest to the
/// geometric mean of `w`, `1.0` for unit weights. The Gaussian LMM route
/// (`fit::lmm::accumulate_lmm_rows`) fits on `ŵ = w/s` too, reporting `σ̂² · s`.
/// The Gamma precision-weight
/// path runs on `ŵᵢ = wᵢ/s` internally: row shape `aᵢ = wᵢ/φ = ŵᵢ/(φ/s)`, so
/// on `ŵ` the internal dispersion coordinate is `φ/s`, unchanged by rescaling
/// every `wᵢ` by one constant `c` — exactly so when `c` is a power of two:
/// `log₂(c·wᵢ) = k + log₂ wᵢ` for `c = 2^k`, so the mean of `log₂ w` shifts
/// by exactly `k` and rounds to `k` more, giving `s(c·w) = c·s(w)` and `ŵ`
/// bit-exact under the rescale — unless the unscaled mean already sits within
/// round-off of a half-integer, where rounding can tip the other way and the
/// chosen power differs by one. That changes nothing in the fit's math (`ŵ`
/// and `φ/s` still satisfy `aᵢ = wᵢ/φ = ŵᵢ/(φ/s)`); it only shifts which
/// internal scale the solver runs on. The raw-scale φ̂ is `s·φ̂_int`.
///
/// A plain arithmetic mean is not robust to a single outlier weight — it can
/// push the internal `φ_int = φ/s` outside the fixed dispersion box on data
/// that fits fine at the raw scale — and can overflow `Σw` at very large
/// weights. Averaging `log₂ wᵢ` instead of `wᵢ` resists both.
pub(crate) fn weight_scale(w: Option<&[f64]>, n: usize) -> f64 {
    match w {
        None => 1.0,
        Some(w) => {
            let mean_log2: f64 = w[..n].iter().map(|&wi| wi.log2()).sum::<f64>() / n as f64;
            2.0_f64.powi(mean_log2.round() as i32)
        }
    }
}

/// The dispersion-only part of the Gamma Laplace/GLM objective, as a function of
/// `ψ = ln φ` (row shape `aᵢ = ŵᵢ/φ`, `ŵᵢ = 1` unweighted). With the Gamma unit
/// deviance `dᵢ = 2[(yᵢ−μᵢ)/μᵢ − ln(yᵢ/μᵢ)]`, the log-density rearranges to
/// ```text
///   −2·log f(yᵢ; aᵢ, μᵢ) = aᵢ·dᵢ + 2aᵢ − 2aᵢ·ln aᵢ + 2·lnΓ(aᵢ) + 2·ln yᵢ
/// ```
/// so `−2·Σᵢ log f(yᵢ; aᵢ, μᵢ) = D/φ + Σᵢ(2aᵢ − 2aᵢ·ln aᵢ + 2·lnΓ(aᵢ)) + 2·Σᵢ ln yᵢ`,
/// `D = Σᵢ ŵᵢ·dᵢ`. PIRLS on the prior weights `ŵᵢ/φ` returns the first term
/// itself; this is the rest, which depends on φ and the data alone. Rows have
/// precision-weighted variance `φ·V(μᵢ)/wᵢ` (McCullagh & Nelder 1989 §2.2), so
/// the `ln yᵢ` term carries no weight — `sum_ln_y = Σᵢ ln yᵢ`. `weights =
/// None` is its own arithmetic path with every `aᵢ = a` (rather than routing
/// through `Some` on an all-ones slice), so unweighted callers get exact,
/// reproducible arithmetic independent of any weighted call's rounding.
pub(crate) fn gamma_dispersion_term(
    ln_phi: f64,
    weights: Option<&[f64]>,
    n: usize,
    sum_ln_y: f64,
) -> f64 {
    let a = (-ln_phi).exp();
    let row = |ai: f64| {
        if ai < 20.0 {
            return 2.0 * ai - 2.0 * ai * ai.ln() + 2.0 * crate::simd_transcendental::ln_gamma(ai);
        }
        // Above a = 20 the three terms, each of order a·ln a, cancel to a
        // value of order ln a: measured against 60-digit values, the direct
        // form is 1e-10 relative off at a = 1e6, 2e-6 at 1e10, and keeps no
        // correct digit at 1e16 (a near-exact fit's φ̂). So the row comes from
        // Stirling's series instead, 2a − 2a·ln a + 2·lnΓ(a) = ln 2π − ln a +
        // 2·R(a), R(a) = 1/(12a) − 1/(360a³) + 1/(1260a⁵) − 1/(1680a⁷)
        // (Abramowitz & Stegun 6.1.41), truncated where the next term, 1/(1188a⁹),
        // is 1.6e-15 at a = 20. Same switch point as `ln_minus_digamma`.
        let r = 1.0 / ai;
        let r2 = r * r;
        let stirling = r * (1.0 / 12.0 + r2 * (-1.0 / 360.0 + r2 * (1.0 / 1260.0 - r2 / 1680.0)));
        (2.0 * std::f64::consts::PI).ln() - ai.ln() + 2.0 * stirling
    };
    let term = match weights {
        None => n as f64 * row(a),
        Some(w) => w[..n].iter().map(|&wi| row(wi * a)).sum(),
    };
    term + 2.0 * sum_ln_y
}

/// `∂/∂ψ` of [`gamma_dispersion_term`], `ψ = ln φ`: `2·Σᵢ aᵢ·(ln aᵢ − ψ₀(aᵢ))`,
/// ψ₀ the digamma function, `aᵢ = wᵢ·a`. Setting `∂/∂ψ (D/φ + term) = 0` at unit
/// weights gives `ln a − ψ₀(a) = D/(2n)`, the Gamma maximum-likelihood shape
/// equation.
#[cfg(test)]
pub(crate) fn gamma_dispersion_term_d1(ln_phi: f64, weights: Option<&[f64]>, n: usize) -> f64 {
    let a = (-ln_phi).exp();
    let term = match weights {
        None => n as f64 * a * (a.ln() - crate::dual::digamma(a)),
        Some(w) => w[..n]
            .iter()
            .map(|&wi| {
                let ai = wi * a;
                ai * ln_minus_digamma(ai).0
            })
            .sum(),
    };
    2.0 * term
}

/// `∂²/∂ψ²` of [`gamma_dispersion_term`]:
/// `−2·Σᵢ aᵢ·(ln aᵢ − ψ₀(aᵢ) + 1 − aᵢ·ψ₁(aᵢ))`, ψ₁ the trigamma function,
/// `aᵢ = wᵢ·a`. The `ln φ` diagonal of the mixed-Gamma joint Hessian is this
/// plus `D/φ` (`glmm::se::joint_hessian_cov`). The weighted arm reads
/// `(g, dg) = ln_minus_digamma(aᵢ)` and uses `1 − aᵢ·ψ₁(aᵢ) = aᵢ·dg`, the same
/// cancelling difference the ML solver below needs at large `aᵢ`.
pub(crate) fn gamma_dispersion_term_d2(ln_phi: f64, weights: Option<&[f64]>, n: usize) -> f64 {
    let a = (-ln_phi).exp();
    let term = match weights {
        None => {
            n as f64 * a * (a.ln() - crate::dual::digamma(a) + 1.0 - a * crate::dual::trigamma(a))
        }
        Some(w) => w[..n]
            .iter()
            .map(|&wi| {
                let ai = wi * a;
                let (g, dg) = ln_minus_digamma(ai);
                ai * (g + ai * dg)
            })
            .sum(),
    };
    -2.0 * term
}

/// The maximum-likelihood Gamma dispersion at fixed means: the root `φ = 1/a`
/// (unweighted) or the per-row `φ = 1/a` with `aᵢ = weightsᵢ·a` solving
/// `Σᵢ weightsᵢ·(ln aᵢ − ψ₀(aᵢ)) = dev/2`, the stationary point in `ln φ` of
/// `dev/φ` plus [`gamma_dispersion_term`] evaluated at those same `weights`,
/// and `MASS::gamma.shape`'s equation at unit weights. A caller passing the
/// precision-normalised `ŵ = w/s` (see [`weight_scale`]) must hand in
/// `dev = D/s`: the raw-scale equation `Σᵢ wᵢ·(ln aᵢ − ψ₀(aᵢ)) = D/2` becomes,
/// on `w = s·ŵ`, `s·Σᵢ ŵᵢ·(ln aᵢ − ψ₀(aᵢ)) = D/2`, i.e.
/// `Σᵢ ŵᵢ·(ln aᵢ − ψ₀(aᵢ)) = D/(2s)`, so the equation this function actually
/// solves (`Σweightsᵢ·gᵢ = dev/2`) needs `dev = D/s` to land on that same
/// target. The returned root is `φ̂` on that internal scale — the caller
/// rescales by `s` to recover the true φ̂. The left
/// side falls strictly from ∞ to 0 in `a`, so the root is unique; Newton in
/// `ln a` from MASS's start `(6 + 2d)/(d·(6 + d))`, `d = D/n`, per-row terms
/// through `ln_minus_digamma` as the weighted `_d2` does — at large per-row
/// `aᵢ` the plain `ln aᵢ − ψ₀(aᵢ)` difference loses enough digits that Newton
/// never meets the stopping rule. A zero deviance (an exact fit) gives φ = 0,
/// and so does a negative one: `D ≥ 0` exactly, so a finite negative value is
/// the round-off of an exact fit.
pub(crate) fn gamma_ml_dispersion(dev: f64, weights: Option<&[f64]>, n: usize) -> f64 {
    let c = dev / (2.0 * n as f64);
    if !c.is_finite() {
        return f64::NAN;
    }
    if c <= 0.0 {
        return 0.0;
    }
    let d = 2.0 * c;
    let mut t = ((6.0 + 2.0 * d) / (d * (6.0 + d))).ln();
    match weights {
        None => {
            for _ in 0..100 {
                let a = t.exp();
                let (g, dg_da) = ln_minus_digamma(a);
                // d/dt = a·d/da, negative everywhere.
                let step = (g - c) / (a * dg_da);
                t -= step;
                if step.abs() <= 4.0 * f64::EPSILON * t.abs().max(1.0) {
                    break;
                }
            }
        }
        Some(w) => {
            let w = &w[..n];
            let target = dev / 2.0;
            for _ in 0..100 {
                let a = t.exp();
                let mut f = -target;
                let mut df = 0.0;
                for &wi in w {
                    let ai = wi * a;
                    let (g, dg) = ln_minus_digamma(ai);
                    f += wi * g;
                    df += wi * dg * ai; // d/dt, negative everywhere.
                }
                let step = f / df;
                t -= step;
                if step.abs() <= 4.0 * f64::EPSILON * t.abs().max(1.0) {
                    break;
                }
            }
        }
    }
    (-t).exp()
}

/// `(ln a − ψ₀(a), 1/a − ψ₁(a))`. Above `a = 20` the two differences cancel to
/// a small fraction of `ln a`, so they come from the asymptotic series
/// (Abramowitz & Stegun 6.3.18, 6.4.12) instead, truncated where the next term
/// is about 2e-16 relative at `a = 20` and smaller above.
fn ln_minus_digamma(a: f64) -> (f64, f64) {
    if a < 20.0 {
        return (
            a.ln() - crate::dual::digamma(a),
            1.0 / a - crate::dual::trigamma(a),
        );
    }
    let r = 1.0 / a;
    let r2 = r * r;
    let g = r
        * (0.5
            + r * (1.0 / 12.0
                + r2 * (-1.0 / 120.0 + r2 * (1.0 / 252.0 + r2 * (-1.0 / 240.0 + r2 / 132.0)))));
    let dg = -r2
        * (0.5
            + r * (1.0 / 6.0
                + r2 * (-1.0 / 30.0 + r2 * (1.0 / 42.0 + r2 * (-1.0 / 30.0 + r2 * 5.0 / 66.0)))));
    (g, dg)
}

/// The inverse-Gaussian family's `−2·logLik + 2`, precision weights (row `i`
/// has variance `φ·V(μᵢ)/wᵢ`):
/// ```text
///   logLik = −½ Σᵢ [ wᵢ·(yᵢ−μᵢ)²/(μᵢ²·yᵢ·φ) + ln(2π·φ·yᵢ³) − ln wᵢ ]
/// ```
/// which is, since `Σᵢ wᵢ(yᵢ−μᵢ)²/(μᵢ²yᵢ) = D`,
/// ```text
///   aic = D/φ + n·ln(2π·φ) + 3·Σᵢ ln yᵢ − Σᵢ ln wᵢ + 2
/// ```
/// `held_phi = None` profiles φ at its ML value `D/n` — [`gamma_ml_dispersion`]'s
/// role for the Gamma family — which collapses `D/φ` to exactly `n`, leaving
/// `aic = n·(ln(2π·disp) + 1) + …`; `held_phi = Some(v)` fixes φ at `v` and
/// keeps `D/φ` literal, since a held φ carries no such identity with `D`. `μ`
/// enters only through `dev`, so it is not a parameter here. At unit weights
/// and `held_phi = None` this is R's `inverse.gaussian()$aic` verbatim (R
/// `src/library/stats/R/family.R`), and the `−Σ ln wᵢ` term is absent. Requires
/// `y > 0`, the family's own domain.
pub(crate) fn inv_gaussian_aic<T: Scalar>(
    y: &[f64],
    dev: T,
    n: usize,
    prior_w: Option<&[f64]>,
    held_phi: Option<T>,
) -> T {
    let nf = T::from_f64(n as f64);
    // `dev_over_phi` is exactly `nf` (not a division) on the profiled path,
    // matching the `D/φ = n` identity that holds only at the profiled `φ̂ = D/n`.
    let (ln_two_pi_phi, dev_over_phi) = match held_phi {
        None => {
            let disp = dev / nf;
            ((T::from_f64(2.0 * std::f64::consts::PI) * disp).ln(), nf)
        }
        Some(phi) => (
            (T::from_f64(2.0 * std::f64::consts::PI) * phi).ln(),
            dev / phi,
        ),
    };
    let mut ln_y = 0.0;
    for &yi in y.iter().take(n) {
        ln_y += yi.ln();
    }
    let ln_w: f64 = prior_w.map_or(0.0, |w| w[..n].iter().map(|wi| wi.ln()).sum());
    nf * ln_two_pi_phi + dev_over_phi + T::from_f64(3.0 * ln_y) - T::from_f64(ln_w)
        + T::from_f64(2.0)
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
/// - **Gamma** — NaN on purpose: both Gamma paths carry the whole log-density
///   through [`gamma_dispersion_term`] at their φ̂, so there is no saturated
///   term to restore; a caller reaching this arm is a bug, surfaced as NaN.
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
/// floors with `glm::WEIGHT_CLAMP`). A tail row ([`in_tail`]) returns
/// [`tail_weight_and_resid`] beside its μ.
pub(crate) fn irls_weight_and_resid<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: f64,
    eta: T,
) -> (T, T, T) {
    let mu = link_inv(family, eta);
    let e = clamp_eta(family, eta);
    if in_tail(family, e.value(), mu.value()) {
        let (w, r) = tail_weight_and_resid(family, nb_theta, y, e, mu);
        return (mu, w, r);
    }
    let v = variance(family, nb_theta, mu);
    if is_canonical(family) {
        // dμ/dη = V(μ) here, so the general form collapses to this shortcut.
        (mu, v, (T::from_f64(y) - mu) / v)
    } else {
        let dm = mu_eta(family, eta);
        (mu, dm * dm / v, (T::from_f64(y) - mu) / dm)
    }
}

/// Observed (Newton) IRLS weight `½·d²devᵢ/dηᵢ²` at η, from the Fisher weight
/// `w = (dμ/dη)²/V` the caller already holds: `w_obs = w − (y−μ)·dr/dη` with
/// `r(η) = (dμ/dη)/V(μ)` the score factor (`−½·d devᵢ/dη = r·(y−μ)`). `dr/dη`
/// is 0 exactly where `r` is constant — the canonical links, Gamma-inverse
/// (`r ≡ −1`) and inverse-Gaussian inverse-squared (`r ≡ −½`) — so the
/// observed and Fisher weights coincide there. Per link, `μ' = dμ/dη`:
/// probit `r = φ/V`, `dr/dη = −φ(ηV + φ(1−2μ))/V²`; cloglog `r = eᵑ/μ`,
/// `dr/dη = r(1 − μ'/μ)`; Gamma-log `r = 1/μ`, `dr/dη = −1/μ`; NB-log
/// `r = θ/(θ+μ)`, `dr/dη = −θμ/(θ+μ)²`; inverse-Gaussian-log `r = 1/μ²`,
/// `dr/dη = −2/μ²`. Checked against a central difference of `r` per family
/// in `observed_weight_matches_fd_of_score_factor`. `eta`/`mu` are the pass's
/// already-clamped values; `w` carries the prior weight, so the correction is
/// scaled by `prior_w` too. Read by `glmm::pirls::observed_weights_in_place`
/// (the Newton step weight of the three `f64` PIRLS loops and the exit
/// refresh's `log|A|`), by the dual derivative kernels' observed twin
/// (`pirls::DualStep::observed`), and by the assembled SE engine. On a tail
/// row ([`in_tail`]) of a link whose exact curvature differs from Fisher it is
/// the η form `prior_w·w_obs` of [`TailRow`] instead, and `w` is not read.
pub(crate) fn observed_weight<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: f64,
    prior_w: f64,
    eta: T,
    mu: T,
    w: T,
) -> T {
    if exact_curvature_differs(family) && in_tail(family, eta.value(), mu.value()) {
        return T::from_f64(prior_w) * tail_row(family, nb_theta, y, eta, mu).w_obs;
    }
    let dr = match family {
        Family::Gaussian
        | Family::Poisson { .. }
        | Family::Binomial {
            link: BinomialLink::Logit,
        }
        | Family::Gamma {
            link: GammaLink::Inverse,
        }
        | Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => return w,
        Family::Binomial {
            link: BinomialLink::Probit,
        } => {
            let phi = mu_eta(family, eta);
            let v = mu * (T::ONE - mu);
            -phi * (eta * v + phi * (T::ONE - T::from_f64(2.0) * mu)) / (v * v)
        }
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => {
            let r = Scalar::exp(eta) / mu;
            r * (T::ONE - mu_eta(family, eta) / mu)
        }
        Family::Gamma {
            link: GammaLink::Log,
        } => -(T::ONE / mu),
        Family::NegativeBinomial { .. } => {
            let th = T::from_f64(nb_theta);
            let d = th + mu;
            -th * mu / (d * d)
        }
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => T::from_f64(-2.0) / (mu * mu),
    };
    w - T::from_f64(prior_w) * (T::from_f64(y) - mu) * dr
}

/// Whether the exact curvature of a row's log-likelihood in η, `W_obs`
/// ([`observed_weight`]), differs from the Fisher working weight: every
/// non-canonical link except the ones whose link is canonical up to sign
/// (Gamma/inverse, inverse-Gaussian/inverse-squared) and the Gaussian identity.
/// On these links the Laplace objective's `log|A|` is taken off
/// `A_obs = M'W_obs M + I` (`glmm::pirls::evaluate_at_mode`), not the Fisher
/// `A`, and the `f64` PIRLS loops step with `W_obs` (the Newton step).
pub(crate) fn exact_curvature_differs(family: Family) -> bool {
    !is_canonical(family)
        && !matches!(
            family,
            Family::Gaussian
                | Family::Gamma {
                    link: GammaLink::Inverse,
                }
                | Family::InverseGaussian {
                    link: InverseGaussianLink::InverseSquared,
                }
        )
}

/// `dW_obs/dη` for [`observed_weight`], the third η-derivative of the row's
/// log-likelihood (negated). With the score `ρ = prior_w·(y−μ)·g(η)`,
/// `g = μ'/V(μ)`, `W_obs = −dρ/dη = prior_w·[μ'·g − (y−μ)·g']` and
/// ```text
///   dW_obs/dη = prior_w·[μ''·g + 2μ'·g' − (y−μ)·g'']
/// ```
/// with `g'` the same per-link factor `observed_weight` uses and `g''`:
///
/// - probit (`g' = N/V²`, `N = −φ(ηV + φ(1−2μ))`, `V' = (1−2μ)φ`):
///   `g'' = N'/V² − 2N·V'/V³`, `N' = ηφ(ηV + φ(1−2μ)) − φ(V − 2φ²)`;
/// - cloglog (`g = r = eᵑ/μ`, `g' = r·s`, `s = 1 − μ'/μ`):
///   `g'' = r·s² − r·(μ''μ − μ'²)/μ²`;
/// - Gamma/log (`g = 1/μ`): `g'' = 1/μ`, so `dW_obs/dη = −W_obs`;
/// - NB/log (`g = θ/(θ+μ)`): `g'' = −θμ(θ−μ)/(θ+μ)³`;
/// - inverse-Gaussian/log (`g = μ⁻²`): `g'' = 4μ⁻²`.
///
/// On a link where the observed and Fisher weights coincide
/// (`!exact_curvature_differs`) it is the Fisher `dw/dη`, handed in as
/// `w_eta` ([`weight_eta_deriv`]). `eta`/`mu` are the pass's already-clamped
/// values, as for `observed_weight`, and a tail row ([`in_tail`]) takes the
/// η form `prior_w·w_obs'` of [`TailRow`]. Held against the `Dual<1>` lane of
/// `observed_weight` and against a central difference in
/// `observed_weight_eta_deriv_matches_*`.
pub(crate) fn observed_weight_eta_deriv<T: Scalar>(
    family: Family,
    nb_theta: f64,
    y: f64,
    prior_w: f64,
    eta: T,
    mu: T,
    w_eta: T,
) -> T {
    if !exact_curvature_differs(family) {
        return w_eta;
    }
    if in_tail(family, eta.value(), mu.value()) {
        return T::from_f64(prior_w) * tail_row(family, nb_theta, y, eta, mu).w_obs_eta;
    }
    let two = T::from_f64(2.0);
    let d1 = mu_eta(family, eta);
    let d2 = mu_eta_eta(family, eta);
    let v = variance(family, nb_theta, mu);
    let g = d1 / v;
    let (g1, g2) = match family {
        Family::Binomial {
            link: BinomialLink::Probit,
        } => {
            let k = eta * v + d1 * (T::ONE - two * mu);
            let nn = -d1 * k;
            let dv = (T::ONE - two * mu) * d1;
            let dn = eta * d1 * k - d1 * (v - two * d1 * d1);
            (nn / (v * v), dn / (v * v) - two * nn * dv / (v * v * v))
        }
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => {
            let r = Scalar::exp(eta) / mu;
            let sfac = T::ONE - d1 / mu;
            (
                r * sfac,
                r * sfac * sfac - r * (d2 * mu - d1 * d1) / (mu * mu),
            )
        }
        Family::Gamma {
            link: GammaLink::Log,
        } => (-(T::ONE / mu), T::ONE / mu),
        Family::NegativeBinomial { .. } => {
            let th = T::from_f64(nb_theta);
            let d = th + mu;
            (-th * mu / (d * d), -th * mu * (th - mu) / (d * d * d))
        }
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => (T::from_f64(-2.0) / (mu * mu), T::from_f64(4.0) / (mu * mu)),
        _ => unreachable!("exact_curvature_differs admits only the links above"),
    };
    T::from_f64(prior_w) * (d2 * g + two * d1 * g1 - (T::from_f64(y) - mu) * g2)
}

/// `dw/dη` of the (prior-weighted) IRLS working weight `w = (dμ/dη)²/V(μ)`
/// (general form) or `w = V(μ)` (canonical). Every arm reduces to `w·g(η,μ)`,
/// so a caller that passes an already prior-weighted `w` gets a
/// prior-weighted derivative back with no separate `prior_w` argument — unlike
/// [`observed_weight`], whose `w − prior_w·(y−μ)·dr` shape needs `prior_w`
/// explicitly. `eta`/`mu` are the pass's already-clamped values, as for
/// [`observed_weight`]: the probit arm reads `η` and the cloglog arm `exp(η)`
/// directly, so an unclamped caller disagrees with the `Dual<1>` lines this
/// function is held equal to. Per link, with `μ' = dμ/dη`, `φ = μ'` on probit:
///
/// - Gaussian identity, Gamma log: `w ≡ 1`, `dw/dη = 0`.
/// - Poisson log (canonical, `w=μ`): `dw/dη = μ' = μ`.
/// - Binomial logit (canonical, `w=μ(1−μ)`): `dw/dη = (1−2μ)μ' = w(1−2μ)`.
/// - Binomial probit (`w=φ²/V`, `V=μ(1−μ)`): `φ'=−ηφ`, `V'=(1−2μ)φ`, so
///   `dw/dη = 2φφ'/V − φ²V'/V² = −w[2η + φ(1−2μ)/V]`.
/// - Binomial cloglog (`w=e^{2η}(1−μ)/μ`): with `t=e^η`, `s=1−μ`,
///   `d ln w/dη = 2 − t − ts/μ`, and `μ+s=1` collapses `t+ts/μ` to `t/μ`, so
///   `dw/dη = w(2 − t/μ) = w(2 − e^η/μ)`.
/// - Gamma inverse (`μ'=−μ²`, `V=μ²`, `w=μ'²/V=μ²`): `dw/dη = 2μμ' = −2μ³ =
///   −2wμ`. **Negative**, unlike the log arm above.
/// - NegBinomial log (`w=μθ/(θ+μ)`): `dw/dη = θ²μ'/(θ+μ)² = w·θ/(θ+μ)` (using
///   `μ'=μ` on the log link).
/// - InvGaussian log (`w=μ'²/V=1/μ`): `dw/dη = −μ'/μ² = −1/μ = −w`.
/// - InvGaussian inverse-squared (`μ'=−μ³/2`, `V=μ³`, `w=μ'²/V=μ³/4`):
///   `dw/dη = (3μ²/4)μ' = −3μ⁵/8 = −1.5wμ²`.
///
/// The exact β-profile's `c_β` (`pirls::row_weight_eta_deriv`) calls this
/// on every non-tail row of a link whose observed and Fisher weights
/// coincide, and reads the `Dual<1>` derivative of `irls_weight_and_resid` on
/// that link's tail rows. Off the tail a test
/// (`weight_eta_deriv_matches_dual1_of_irls_weight`) holds the two forms
/// equal.
///
/// A tail row ([`in_tail`]) of a link whose exact curvature differs from
/// Fisher reads `w·(ln w)'` off [`TailRow`] (probit `(ln w)' = λ(−η) − λ(η) −
/// 2η`, cloglog `2 − t/μ` at the unbounded `μ = −expm1(−t)`), since the stored
/// probit/cloglog μ sits on its `PROB_EPS` bound there. Logit keeps
/// `w(1−2μ)`: its μ is never bounded.
pub(crate) fn weight_eta_deriv<T: Scalar>(family: Family, nb_theta: f64, eta: T, mu: T, w: T) -> T {
    if exact_curvature_differs(family) && in_tail(family, eta.value(), mu.value()) {
        // `(ln w)'` reads neither y nor the prior weight.
        return w * tail_row(family, nb_theta, 0.0, eta, mu).w_log_eta;
    }
    match family {
        Family::Gaussian
        | Family::Gamma {
            link: GammaLink::Log,
        } => T::ZERO,
        Family::Poisson { .. } => w,
        Family::Binomial {
            link: BinomialLink::Logit,
        } => w * (T::ONE - T::from_f64(2.0) * mu),
        Family::Binomial {
            link: BinomialLink::Probit,
        } => {
            let v = mu * (T::ONE - mu);
            let phi = mu_eta(family, eta);
            -w * (T::from_f64(2.0) * eta + phi * (T::ONE - T::from_f64(2.0) * mu) / v)
        }
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => w * (T::from_f64(2.0) - Scalar::exp(eta) / mu),
        Family::Gamma {
            link: GammaLink::Inverse,
        } => T::from_f64(-2.0) * w * mu,
        Family::NegativeBinomial { .. } => {
            let th = T::from_f64(nb_theta);
            w * th / (th + mu)
        }
        Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => -w,
        Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        } => T::from_f64(-1.5) * w * mu * mu,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BinomialLink, Family, GammaLink, InverseGaussianLink, NegBinomialLink, PoissonLink,
    };

    /// Past cloglog's e^{−t} underflow (η > ln 745) and probit's Φ underflow
    /// (|η| > 38) the tail weight is exactly 0; the working residual must still
    /// be finite, or z = η + r poisons X'Wz.
    #[test]
    fn tail_working_residual_is_finite_where_the_weight_underflows() {
        let cloglog = Family::Binomial {
            link: BinomialLink::Cloglog,
        };
        let probit = Family::Binomial {
            link: BinomialLink::Probit,
        };
        for (family, eta) in [
            (cloglog, 7.0),
            (cloglog, 15.0),
            (probit, 40.0),
            (probit, -40.0),
        ] {
            for y in [0.0, 1.0] {
                let (_, _, r) = irls_weight_and_resid(family, f64::NAN, y, eta);
                assert!(r.is_finite(), "{family:?} η = {eta} y = {y}: r = {r}");
            }
        }
    }

    /// `observed_weight`'s `dr/dη` per link against a central difference of
    /// `r(η) = (dμ/dη)/V(μ(η))`, read back through `w_obs = w − (y−μ)·dr/dη`
    /// at `y − μ = 1` (so `dr/dη = w − w_obs`). Step 1e-5, band 1e-7: the
    /// truncation error is O(h²) ≈ 1e-10 on these smooth links, the
    /// cancellation error ≈ 1e-11, so the band is ~1000× the expected error.
    #[test]
    fn observed_weight_matches_fd_of_score_factor() {
        let fams = [
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            Family::Binomial {
                link: BinomialLink::Cloglog,
            },
            Family::Gamma {
                link: GammaLink::Log,
            },
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
            // Constant-`r` links: the derivative is exactly 0.
            Family::Gamma {
                link: GammaLink::Inverse,
            },
            Family::InverseGaussian {
                link: InverseGaussianLink::InverseSquared,
            },
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            Family::Poisson {
                link: PoissonLink::Log,
            },
        ];
        let nb_theta = 2.5;
        let r = |f: Family, eta: f64| mu_eta(f, eta) / variance(f, nb_theta, link_inv(f, eta));
        for f in fams {
            // Positive η only: Gamma-inverse and IG-inverse-squared need η > 0.
            for eta in [0.3_f64, 0.9, 1.7] {
                let h = 1e-5;
                let fd = (r(f, eta + h) - r(f, eta - h)) / (2.0 * h);
                let mu = link_inv(f, eta);
                let w = 1.0;
                let got = w - observed_weight(f, nb_theta, mu + 1.0, 1.0, eta, mu, w);
                assert!(
                    (got - fd).abs() < 1e-7,
                    "{f:?} eta={eta}: dr/deta {got} vs fd {fd}"
                );
            }
        }
    }

    /// The ten `(family, link)` cells `weight_eta_deriv` and D4's table cover,
    /// paired with the η domain each needs (Gamma-inverse and IG-inverse-squared
    /// need `η > 0`).
    fn weight_eta_deriv_cells() -> Vec<(Family, &'static [f64])> {
        const GEN: &[f64] = &[-1.3, 0.2, 1.7];
        // 0.5 rather than 0.3: at h = 1e-5 the central difference's O(h²) truncation
        // term tracks the third derivative of w(η), which is steep enough near
        // η = 0.3 on the Gamma-inverse link (w = η⁻²) to exceed the 1e-7 band;
        // IG-inverse-squared (w = η^(-3/2)/4) is milder and rides along.
        const POS: &[f64] = &[0.5, 0.9, 1.7];
        vec![
            (Family::Gaussian, GEN),
            (
                Family::Poisson {
                    link: PoissonLink::Log,
                },
                GEN,
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Logit,
                },
                GEN,
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Probit,
                },
                GEN,
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Cloglog,
                },
                GEN,
            ),
            (
                Family::Gamma {
                    link: GammaLink::Log,
                },
                GEN,
            ),
            (
                Family::Gamma {
                    link: GammaLink::Inverse,
                },
                POS,
            ),
            (
                Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
                GEN,
            ),
            (
                Family::InverseGaussian {
                    link: InverseGaussianLink::Log,
                },
                GEN,
            ),
            (
                Family::InverseGaussian {
                    link: InverseGaussianLink::InverseSquared,
                },
                POS,
            ),
        ]
    }

    /// `weight_eta_deriv` against the `Dual<1>` derivative of the Fisher weight
    /// `irls_weight_and_resid` returns. The two must agree to round-off: the
    /// exact β-profile's pass A calls the closed form off the tail and the
    /// `Dual<1>` line on it, and the border's curvature differentiates the
    /// closed form again.
    #[test]
    fn weight_eta_deriv_matches_dual1_of_irls_weight() {
        use crate::dual::Dual;
        let nb_theta = 2.5;
        for (f, etas) in weight_eta_deriv_cells() {
            for &eta in etas {
                // The 2.5 row is implied by the 1.0 row while every arm is
                // `w·g(η, μ)`; it guards a future arm that is not.
                for &prior_w in &[1.0_f64, 2.5_f64] {
                    let e = Dual::<1> { v: eta, d: [1.0] };
                    let (mu_d, w_d, _) = irls_weight_and_resid(f, nb_theta, 1.0, e);
                    let want = prior_w * w_d.d[0];
                    let got = weight_eta_deriv(f, nb_theta, eta, mu_d.v, prior_w * w_d.v);
                    assert!(
                        (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                        "{f:?} eta={eta} prior_w={prior_w}: got {got} want {want}"
                    );
                }
            }
        }
    }

    /// `mu_eta_eta` against the `Dual<1>` derivative of `mu_eta` at the same η,
    /// for every family/link, over a grid that includes each link's
    /// [`clamp_eta`] bounds and ordinary interior points. Band fixed from the
    /// first run: worst observed relative error 2.1e-16, round-off on both
    /// sides.
    #[test]
    fn mu_eta_eta_matches_dual1_of_mu_eta() {
        use crate::dual::Dual;
        let cells: Vec<(Family, Vec<f64>)> = vec![
            (Family::Gaussian, vec![-5.0, 0.0, 3.0]),
            (
                Family::Binomial {
                    link: BinomialLink::Logit,
                },
                vec![-27.7, -3.0, 0.0, 1.7, 27.7],
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Probit,
                },
                vec![-7.03, -1.0, 0.0, 1.0, 7.03],
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Cloglog,
                },
                vec![-ETA_MAX, -27.6, 0.0, 3.32, ETA_MAX.ln()],
            ),
            (
                Family::Poisson {
                    link: PoissonLink::Log,
                },
                vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
            ),
            (
                Family::Gamma {
                    link: GammaLink::Log,
                },
                vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
            ),
            (
                Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
                vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
            ),
            (
                Family::InverseGaussian {
                    link: InverseGaussianLink::Log,
                },
                vec![-ETA_MAX, -23.03, 0.0, 5.0, ETA_MAX],
            ),
            (
                Family::Gamma {
                    link: GammaLink::Inverse,
                },
                vec![MU_FLOOR, 0.5, 5.0, ETA_MAX],
            ),
            (
                Family::InverseGaussian {
                    link: InverseGaussianLink::InverseSquared,
                },
                vec![MU_FLOOR, 0.5, 5.0, ETA_MAX],
            ),
        ];
        for (f, etas) in cells {
            for eta in etas {
                let e = Dual::<1> { v: eta, d: [1.0] };
                let dual_lane = mu_eta(f, e).d[0];
                let closed = mu_eta_eta(f, eta);
                let band = 1e-12 * closed.abs().max(1.0);
                assert!(
                    (dual_lane - closed).abs() <= band,
                    "{f:?} eta={eta}: dual {dual_lane} vs closed {closed}"
                );
            }
        }
    }

    /// Every family with a tail ([`in_tail`]), at tail points on each
    /// reachable side: the tail row must really be in the tail, its unit
    /// deviance finite, and every per-row quantity the exact derivative the
    /// kernels treat it as — held against the `Dual<1>` lane of the quantity
    /// one order down, through the same entry points the kernels call.
    /// `row_score = −½·∂dev/∂η`, `observed_weight = −∂row_score/∂η`,
    /// `weight_eta_deriv = ∂w/∂η` of the `irls_weight_and_resid` weight, and
    /// `observed_weight_eta_deriv = ∂observed_weight/∂η`. The stored μ each
    /// quantity is handed is the one [`link_inv`] builds, so probit and
    /// cloglog see their `PROB_EPS`-bounded μ, as in a fit. Band fixed from
    /// the first run with an order of magnitude of headroom.
    #[test]
    fn tail_forms_match_dual1() {
        use crate::dual::Dual;
        let nb_theta = 2.5;
        let probit = Family::Binomial {
            link: BinomialLink::Probit,
        };
        let cloglog = Family::Binomial {
            link: BinomialLink::Cloglog,
        };
        let logit = Family::Binomial {
            link: BinomialLink::Logit,
        };
        let nb = Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        };
        let gamma = Family::Gamma {
            link: GammaLink::Log,
        };
        let ig = Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        };
        // (family, y values, tail η points)
        let cells: &[(Family, &[f64], &[f64])] = &[
            (
                probit,
                &[0.0, 0.3, 1.0],
                &[-7.1, -12.0, -40.0, -300.0, 7.1, 12.0, 40.0],
            ),
            (
                cloglog,
                &[0.0, 0.3, 1.0],
                &[-27.7, -60.0, -600.0, 3.4, 6.0, 30.0, 650.0],
            ),
            (
                logit,
                &[0.0, 0.3, 1.0],
                &[-27.7, -60.0, -600.0, 27.7, 60.0, 600.0],
            ),
            (nb, &[0.0, 3.0], &[-231.0, -600.0, 231.0, 600.0]),
            (gamma, &[0.5, 3.0], &[-231.0, -600.0, 231.0, 600.0]),
            (ig, &[0.5, 3.0], &[-231.0, 231.0, 600.0]),
        ];
        let close = |got: f64, want: f64| (got - want).abs() <= 1e-11 * want.abs().max(1.0);
        for &(f, ys, etas) in cells {
            for &eta in etas {
                let mu = link_inv(f, eta);
                assert!(in_tail(f, eta, mu), "{f:?} eta={eta}: not a tail point");
                let e = Dual::<1> { v: eta, d: [1.0] };
                let mu_d = link_inv(f, e);
                for &y in ys {
                    let dev = dev_resid_at(f, nb_theta, y, e, mu_d);
                    assert!(dev.v.is_finite(), "{f:?} eta={eta} y={y}: dev {}", dev.v);
                    let score = row_score(f, nb_theta, y, 1.0, e, mu_d);
                    assert!(
                        close(score.v, -0.5 * dev.d[0]),
                        "{f:?} eta={eta} y={y}: score {} vs −½·dev' {}",
                        score.v,
                        -0.5 * dev.d[0]
                    );
                    let (_, w_d, _) = irls_weight_and_resid(f, nb_theta, y, e);
                    let w_eta = weight_eta_deriv(f, nb_theta, eta, mu, w_d.v);
                    assert!(
                        close(w_eta, w_d.d[0]),
                        "{f:?} eta={eta} y={y}: w' {w_eta} vs dual {}",
                        w_d.d[0]
                    );
                    let wo_d = observed_weight(f, nb_theta, y, 1.0, e, mu_d, w_d);
                    assert!(
                        close(wo_d.v, -score.d[0]),
                        "{f:?} eta={eta} y={y}: w_obs {} vs −score' {}",
                        wo_d.v,
                        -score.d[0]
                    );
                    let wo_eta = observed_weight_eta_deriv(f, nb_theta, y, 1.0, eta, mu, w_eta);
                    assert!(
                        close(wo_eta, wo_d.d[0]),
                        "{f:?} eta={eta} y={y}: w_obs' {wo_eta} vs dual {}",
                        wo_d.d[0]
                    );
                }
            }
        }
    }

    /// The deviance keeps growing into the tail: on each family with a μ bound
    /// or a tail, walking η further out against the data (a y > 0 row toward
    /// μ → 0, a y < 1 or y < μ row toward the upper side) strictly raises
    /// `dev_resid_at`, finite at every step, with no constant stretch past
    /// `MU_FLOOR` or `PROB_EPS`. And at the switch from the μ form to the tail
    /// form the two agree to round-off on the side where μ is exactly
    /// representable.
    #[test]
    fn tail_deviance_grows_and_meets_the_mu_form() {
        let nb_theta = 2.5;
        let rising = |f: Family, y: f64, etas: &[f64]| {
            let d: Vec<f64> = etas
                .iter()
                .map(|&e| {
                    let e = clamp_eta(f, e);
                    dev_resid_at(f, nb_theta, y, e, link_inv(f, e))
                })
                .collect();
            for k in 1..d.len() {
                assert!(
                    d[k].is_finite() && d[k] > d[k - 1],
                    "{f:?} y={y}: deviance {:?} at η {:?} is not rising",
                    d,
                    etas
                );
            }
        };
        for link in [
            BinomialLink::Logit,
            BinomialLink::Probit,
            BinomialLink::Cloglog,
        ] {
            let f = Family::Binomial { link };
            rising(f, 1.0, &[-5.0, -10.0, -30.0, -60.0, -100.0, -600.0]);
            rising(f, 0.0, &[3.0, 5.0, 10.0, 30.0, 100.0, 600.0]);
        }
        for f in [
            Family::Poisson {
                link: PoissonLink::Log,
            },
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            Family::Gamma {
                link: GammaLink::Log,
            },
        ] {
            rising(
                f,
                2.0,
                &[-10.0, -23.1, -30.0, -100.0, -229.0, -231.0, -500.0],
            );
        }
        // IG's deviance `(y/μ − 1)²/y` passes f64's range near η = −354, where
        // the true value does too.
        rising(
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
            2.0,
            &[-10.0, -23.1, -30.0, -100.0, -229.0, -231.0, -350.0],
        );
        // Upper side of the log links: NB and Gamma keep rising; IG's
        // deviance tends to its true bound 1/y as μ → ∞.
        for f in [
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            Family::Gamma {
                link: GammaLink::Log,
            },
        ] {
            rising(f, 2.0, &[10.0, 100.0, 229.0, 231.0, 500.0, 699.0]);
        }
        // Continuity at the switch, on the side where μ is exact.
        let meet = |f: Family, y: f64, eta_in: f64, eta_out: f64| {
            let (mi, mo) = (link_inv(f, eta_in), link_inv(f, eta_out));
            assert!(!in_tail(f, eta_in, mi) && in_tail(f, eta_out, mo));
            let (di, dout) = (
                dev_resid_at(f, nb_theta, y, eta_in, mi),
                dev_resid_at(f, nb_theta, y, eta_out, mo),
            );
            // Both points sit 1e-9 apart in η; the deviance slope there is
            // O(|η|), so the gap is the slope's share plus round-off.
            assert!(
                (di - dout).abs() <= 1e-6 * di.abs().max(1.0),
                "{f:?} y={y}: μ form {di} vs tail form {dout} across the switch"
            );
        };
        let lo_probit = -7.034_483_825_117_6;
        meet(
            Family::Binomial {
                link: BinomialLink::Probit,
            },
            1.0,
            lo_probit + 1e-9,
            lo_probit - 1e-9,
        );
        let lo_logit = (PROB_EPS / (1.0 - PROB_EPS)).ln();
        meet(
            Family::Binomial {
                link: BinomialLink::Logit,
            },
            1.0,
            lo_logit + 1e-9,
            lo_logit - 1e-9,
        );
        for f in [
            Family::NegativeBinomial {
                link: NegBinomialLink::Log,
            },
            Family::Gamma {
                link: GammaLink::Log,
            },
            Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            },
        ] {
            meet(f, 2.0, -LOG_TAIL_ETA + 1e-9, -LOG_TAIL_ETA - 1e-9);
            meet(f, 2.0, LOG_TAIL_ETA - 1e-9, LOG_TAIL_ETA + 1e-9);
        }
    }

    /// `weight_eta_deriv` against a central difference of the same
    /// `irls_weight_and_resid` weight, at the step and band
    /// `observed_weight_matches_fd_of_score_factor` uses. Independent of the
    /// gate above: both could pass together wrong only if `Dual<1>`'s chain
    /// rule itself were wrong, which this catches.
    #[test]
    fn weight_eta_deriv_matches_fd_of_irls_weight() {
        let nb_theta = 2.5;
        let h = 1e-5;
        for (f, etas) in weight_eta_deriv_cells() {
            let w_at = |e: f64| irls_weight_and_resid(f, nb_theta, 1.0, e).1;
            for &eta in etas {
                for &prior_w in &[1.0_f64, 2.5_f64] {
                    let fd = prior_w * (w_at(eta + h) - w_at(eta - h)) / (2.0 * h);
                    let mu = link_inv(f, eta);
                    let w = prior_w * w_at(eta);
                    let got = weight_eta_deriv(f, nb_theta, eta, mu, w);
                    assert!(
                        (got - fd).abs() < 1e-7,
                        "{f:?} eta={eta} prior_w={prior_w}: got {got} vs fd {fd}"
                    );
                }
            }
        }
    }

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
    fn cloglog_eta_is_clamped_at_eta_max() {
        let f = Family::Binomial {
            link: BinomialLink::Cloglog,
        };
        assert_eq!(clamp_eta(f, 1e6), ETA_MAX);
        assert_eq!(clamp_eta_bounds(f), (-ETA_MAX, ETA_MAX));
        // At the bound `e^η` is finite, μ sits on its `PROB_EPS` bound, and
        // the tail deviance `2e^η` of a y = 0 row is finite.
        let mu = link_inv(f, 1e6);
        assert_eq!(mu, 1.0 - PROB_EPS);
        assert!(mu_eta(f, 1e6).is_finite());
        let d = dev_resid_at(f, f64::NAN, 0.0, ETA_MAX, mu);
        assert!(d.is_finite() && d > 1e300, "d={d}");
        // Logit and probit take no η clamp.
        for link in [BinomialLink::Logit, BinomialLink::Probit] {
            assert_eq!(clamp_eta(Family::Binomial { link }, 1e6), 1e6);
        }
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
        assert!(eta_infeasible(f, 1.0, 0.0));
        assert!(eta_infeasible(f, 1.0, -1.0));
        assert!(!eta_infeasible(f, 1.0, 1e-3));
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
        // The log link evaluates η on ±ETA_MAX; past it the row's deviance
        // would stop moving, so a trial there is refused.
        assert!(!eta_infeasible(f, 1.0, -50.0));
        assert!(eta_infeasible(f, 1.0, -ETA_MAX - 1.0));
        assert!(eta_infeasible(f, 0.0, ETA_MAX + 1.0));
        assert!(!eta_infeasible(f, 1.0, f64::NAN));
        // A y = 0 row past the lower bound is at its deviance's own limit.
        assert!(!eta_infeasible(f, 0.0, -ETA_MAX - 1.0));
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

    /// Links and interior η points for the observed-weight derivative checks
    /// (positive η for the inverse links, whose domain is η > 0). Covers every
    /// link, so a link wrongly classified by `exact_curvature_differs` in
    /// either direction is caught: wrongly true reaches the `unreachable!`,
    /// wrongly false returns the Fisher `dw/dη` and misses the `Dual<1>` lane.
    fn observed_weight_cells() -> Vec<(Family, Vec<f64>)> {
        vec![
            (Family::Gaussian, vec![-1.0, 0.3]),
            (
                Family::InverseGaussian {
                    link: InverseGaussianLink::InverseSquared,
                },
                vec![0.4, 1.5],
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Probit,
                },
                vec![-1.3, 0.0, 0.4, 2.1],
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Cloglog,
                },
                vec![-2.0, -0.3, 0.5, 1.2],
            ),
            (
                Family::Gamma {
                    link: GammaLink::Log,
                },
                vec![-1.0, 0.2, 2.3],
            ),
            (
                Family::NegativeBinomial {
                    link: NegBinomialLink::Log,
                },
                vec![-1.0, 0.2, 2.3],
            ),
            (
                Family::InverseGaussian {
                    link: InverseGaussianLink::Log,
                },
                vec![-0.5, 0.2, 1.1],
            ),
            (
                Family::Binomial {
                    link: BinomialLink::Logit,
                },
                vec![-1.0, 0.3],
            ),
            (
                Family::Poisson {
                    link: PoissonLink::Log,
                },
                vec![-1.0, 0.3],
            ),
            (
                Family::Gamma {
                    link: GammaLink::Inverse,
                },
                vec![0.4, 1.5],
            ),
        ]
    }

    /// `observed_weight_eta_deriv` (`dW_obs/dη`, the third η-derivative of the
    /// row's log-likelihood) against the `Dual<1>` lane of `observed_weight`
    /// itself, with the Fisher weight and μ carried as duals of the same η.
    /// Band 1e-11 relative: both sides are closed forms at `f64`.
    #[test]
    fn observed_weight_eta_deriv_matches_dual1_of_observed_weight() {
        use crate::dual::Dual;
        let nb_theta = 2.5;
        for (f, etas) in observed_weight_cells() {
            for &eta in &etas {
                for &(y, prior_w) in &[(0.3_f64, 1.0_f64), (1.0, 2.5), (4.0, 0.7)] {
                    // Binomial rows need y in [0, 1].
                    let y = if matches!(f, Family::Binomial { .. }) {
                        y.min(1.0) * 0.8
                    } else {
                        y
                    };
                    let e = Dual::<1> { v: eta, d: [1.0] };
                    let (mu_d, w_raw, _) = irls_weight_and_resid(f, nb_theta, y, e);
                    let w_d = Dual::<1>::from_f64(prior_w) * w_raw;
                    let want = observed_weight(f, nb_theta, y, prior_w, e, mu_d, w_d).d[0];
                    let w_eta = weight_eta_deriv(f, nb_theta, eta, mu_d.v, w_d.v);
                    let got =
                        observed_weight_eta_deriv(f, nb_theta, y, prior_w, eta, mu_d.v, w_eta);
                    assert!(
                        (got - want).abs() <= 1e-11 * want.abs().max(1.0),
                        "{f:?} eta={eta} y={y} pw={prior_w}: got {got} want {want}"
                    );
                }
            }
        }
    }

    /// The same derivative against a central difference of the `f64`
    /// `observed_weight` in η. Step 1e-5, band 1e-6 relative (O(h²) truncation
    /// on smooth links, far below the band).
    #[test]
    fn observed_weight_eta_deriv_matches_fd_of_observed_weight() {
        let nb_theta = 2.5;
        let wobs = |f: Family, y: f64, pw: f64, eta: f64| {
            let (mu, w_raw, _) = irls_weight_and_resid(f, nb_theta, y, eta);
            observed_weight(f, nb_theta, y, pw, eta, mu, pw * w_raw)
        };
        for (f, etas) in observed_weight_cells() {
            for &eta in &etas {
                let y = if matches!(f, Family::Binomial { .. }) {
                    0.6
                } else {
                    1.7
                };
                let pw = 1.3;
                let h = 1e-5;
                let fd = (wobs(f, y, pw, eta + h) - wobs(f, y, pw, eta - h)) / (2.0 * h);
                let (mu, w_raw, _) = irls_weight_and_resid(f, nb_theta, y, eta);
                let w_eta = weight_eta_deriv(f, nb_theta, eta, mu, pw * w_raw);
                let got = observed_weight_eta_deriv(f, nb_theta, y, pw, eta, mu, w_eta);
                assert!(
                    (got - fd).abs() <= 1e-6 * fd.abs().max(1.0),
                    "{f:?} eta={eta}: got {got} fd {fd}"
                );
            }
        }
    }

    /// `gamma_dispersion_term(ln φ) + D/φ` is the whole `−2·Σᵢ log f(yᵢ; shape
    /// aᵢ, mean μᵢ)`, `aᵢ = wᵢ/φ`, normalising terms included, at every φ:
    /// PIRLS on the prior weights `wᵢ/φ` returns `D/φ`, and this term is the
    /// rest. The reference is the Gamma log-density written out term by term,
    /// `aᵢ·ln aᵢ − aᵢ·ln μᵢ + (aᵢ−1)·ln yᵢ − aᵢ·yᵢ/μᵢ − lnΓ(aᵢ)`, precision
    /// weights (row `i` has variance `φ·V(μᵢ)/wᵢ`).
    #[test]
    fn gamma_dispersion_term_completes_the_gamma_log_density() {
        let y = [0.7_f64, 1.4, 2.2, 3.1, 0.2];
        let mu = [0.9_f64, 1.2, 2.6, 2.8, 0.5];
        let w = [1.0_f64, 2.0, 0.5, 1.5, 1.0];
        let n = y.len();
        let family = Family::Gamma {
            link: GammaLink::Log,
        };
        let sum_ln_y: f64 = y.iter().map(|v| v.ln()).sum();
        let dev: f64 = (0..n)
            .map(|i| w[i] * dev_resid(family, f64::NAN, y[i], mu[i]))
            .sum();
        for &phi in &[0.05_f64, 0.3, 1.0, 2.5] {
            let want = -2.0
                * (0..n)
                    .map(|i| {
                        let a = w[i] / phi;
                        let lg = crate::simd_transcendental::ln_gamma(a);
                        a * a.ln() - a * mu[i].ln() + (a - 1.0) * y[i].ln() - a * y[i] / mu[i] - lg
                    })
                    .sum::<f64>();
            let got = gamma_dispersion_term(phi.ln(), Some(&w), n, sum_ln_y) + dev / phi;
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "φ = {phi}: {got} vs {want}"
            );
        }
        // Unweighted (`weights = None`) must equal every row sharing `w = 1`.
        let ones = [1.0_f64; 5];
        for &phi in &[0.05_f64, 0.3, 1.0, 2.5] {
            let via_none = gamma_dispersion_term(phi.ln(), None, n, sum_ln_y);
            let via_unit_weights = gamma_dispersion_term(phi.ln(), Some(&ones), n, sum_ln_y);
            assert!(
                (via_none - via_unit_weights).abs() <= 1e-12 * via_unit_weights.abs().max(1.0),
                "φ = {phi}: None {via_none} vs unit weights {via_unit_weights}"
            );
        }
    }

    /// `gamma_dispersion_term` per row, `2a − 2a·ln a + 2·lnΓ(a)` at `a = 1/φ`,
    /// against 60-digit values (mpmath `loggamma`) from the switch to Stirling's
    /// series at a = 20 out to the a ≈ 1e16 of a near-exact fit. Band 1e-14
    /// relative; the direct form misses it from about a = 50 (1e-10 off at 1e6,
    /// no correct digit at 1e16).
    #[test]
    fn gamma_dispersion_term_row_is_accurate_at_small_phi() {
        let refs = [
            (20.0_f64, -1.149_522_567_760_651_7),
            (50.0, -2.070_812_650_124_833_8),
            (1e3, -5.069_711_545_911_68),
            (1e6, -11.977_633_324_888_262),
            (1e10, -21.187_973_863_514_445),
            (1e16, -35.003_484_421_495_385),
            (1e30, -67.239_675_723_412_02),
        ];
        for (a, want) in refs {
            let got = gamma_dispersion_term(-a.ln(), None, 1, 0.0);
            assert!(
                (got - want).abs() <= 1e-14 * want.abs(),
                "a = {a:e}: {got} vs {want}"
            );
        }
    }

    /// The closed-form first and second `ln φ` derivatives of
    /// `gamma_dispersion_term` against central differences of the term itself,
    /// unweighted and with row-varying weights. Bands: 1e-6 and 1e-4 relative,
    /// the O(h²) truncation of a step of 1e-4.
    #[test]
    fn gamma_dispersion_term_derivatives_match_central_differences() {
        let n = 4;
        let sum_ln_y = 1.3_f64;
        let w = [0.5_f64, 1.0, 2.5, 4.0];
        let h = 1e-4;
        for weights in [None, Some(&w[..])] {
            let g = |t: f64| gamma_dispersion_term(t, weights, n, sum_ln_y);
            for &psi in &[-3.0_f64, -1.0, 0.0, 0.9] {
                let d1 = (g(psi + h) - g(psi - h)) / (2.0 * h);
                let d2 = (g(psi + h) - 2.0 * g(psi) + g(psi - h)) / (h * h);
                let got1 = gamma_dispersion_term_d1(psi, weights, n);
                let got2 = gamma_dispersion_term_d2(psi, weights, n);
                assert!(
                    (got1 - d1).abs() <= 1e-6 * d1.abs().max(1.0),
                    "weighted={} ψ = {psi}: d1 {got1} vs {d1}",
                    weights.is_some()
                );
                assert!(
                    (got2 - d2).abs() <= 1e-4 * d2.abs().max(1.0),
                    "weighted={} ψ = {psi}: d2 {got2} vs {d2}",
                    weights.is_some()
                );
            }
        }
    }

    /// `gamma_ml_dispersion` solves the Gamma ML shape equation
    /// `ln a − ψ(a) = D/(2n)`, `a = 1/φ`, from the near-Poisson end (large a)
    /// to the heavy-tailed one (small a), and a zero deviance gives φ = 0.
    #[test]
    fn gamma_ml_dispersion_solves_the_shape_equation() {
        let n = 288;
        for &c in &[1e-8_f64, 1e-4, 0.05, 0.4418, 1.0, 5.0, 40.0] {
            let phi = gamma_ml_dispersion(2.0 * c * n as f64, None, n);
            let a = 1.0 / phi;
            // The residual is itself computed with the cancelling difference,
            // so its own rounding, a few ulp of ln a, is allowed on top.
            let resid = a.ln() - crate::dual::digamma(a) - c;
            let tol = 1e-12 * c + 16.0 * f64::EPSILON * a.ln().abs().max(1.0);
            assert!(
                phi > 0.0 && resid.abs() <= tol,
                "c = {c}: φ = {phi}, residual {resid}"
            );
            // Near the Poisson end the root is φ ≈ 2c (a ≈ 1/(2c)).
            if c < 1e-3 {
                assert!(
                    (phi / (2.0 * c) - 1.0).abs() < 2.0 * c,
                    "c = {c}: φ = {phi}"
                );
            }
        }
        assert_eq!(gamma_ml_dispersion(0.0, None, n), 0.0);
        // A deviance that rounds below zero on data the mean model reproduces
        // exactly is still an exact fit, not a NaN.
        assert_eq!(gamma_ml_dispersion(-1.4e-15, None, n), 0.0);
        assert!(gamma_ml_dispersion(f64::NAN, None, n).is_nan());
        // MASS::gamma.shape on sim_gamma's log-link GLM: dev 254.45332674125353
        // over 288 rows gives 1/alpha = 0.78574357084515345.
        let phi = gamma_ml_dispersion(254.45332674125353, None, 288);
        assert!((phi - 0.7857435708451534).abs() < 1e-14, "φ = {phi}");
    }

    /// The weighted arm solves the per-row equation `Σᵢ wᵢ·(ln aᵢ − ψ₀(aᵢ)) =
    /// D/2`, `aᵢ = wᵢ/φ`: check the residual of that equation directly at the
    /// returned root, on weights that vary row to row (not all-equal, so the
    /// unweighted closed form does not apply), and that `Σŵ = n` unit weights
    /// reproduce the unweighted root.
    #[test]
    fn gamma_ml_dispersion_weighted_solves_the_per_row_equation() {
        let w = [0.4_f64, 0.8, 1.3, 2.5, 3.0];
        let n = w.len();
        for &dev in &[0.5_f64, 3.0, 12.0, 40.0] {
            let phi = gamma_ml_dispersion(dev, Some(&w), n);
            assert!(phi > 0.0 && phi.is_finite(), "dev = {dev}: φ = {phi}");
            let resid: f64 = w
                .iter()
                .map(|&wi| {
                    let ai = wi / phi;
                    wi * (ai.ln() - crate::dual::digamma(ai))
                })
                .sum::<f64>()
                - dev / 2.0;
            assert!(
                resid.abs() <= 1e-9 * dev.max(1.0),
                "dev = {dev}: φ = {phi}, residual {resid}"
            );
        }
        let ones = [1.0_f64; 5];
        let via_ones = gamma_ml_dispersion(12.0, Some(&ones), 5);
        let via_none = gamma_ml_dispersion(12.0, None, 5);
        assert!(
            (via_ones - via_none).abs() <= 1e-12 * via_none.abs().max(1.0),
            "unit weights {via_ones} vs None {via_none}"
        );
    }

    /// Unweighted matches R's `inverse.gaussian()$aic` formula exactly; the
    /// weighted half is the precision closed form (row `i` has variance
    /// `φ·V(μᵢ)/wᵢ`, so `φ̂ = D/n`, `aic = n·(ln(2πφ̂)+1) + 3Σln yᵢ − Σln wᵢ +
    /// 2`), a different formula from R's `Σwᵢ`/`Σwᵢln yᵢ` case-weight one.
    #[test]
    fn inv_gaussian_aic_matches_precision_formula() {
        // R: aic = n*(log(dev/n*2*pi)+1) + 3*sum(log(y)) + 2
        let y = [1.5_f64, 2.0, 0.75, 3.25];
        let n = y.len();
        let dev = 0.42_f64;
        let got = inv_gaussian_aic(&y, dev, n, None, None);
        let disp = dev / n as f64;
        let want = n as f64 * ((2.0 * std::f64::consts::PI * disp).ln() + 1.0)
            + 3.0 * y.iter().map(|v| v.ln()).sum::<f64>()
            + 2.0;
        assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        // Precision weights: φ̂ = D/n (n, not Σw), and an extra −Σln wᵢ term
        // (`−2ℓ = D/φ + n·ln(2πφ) + 3·Σln yᵢ − Σln wᵢ`).
        let w = [1.0_f64, 2.0, 0.5, 1.5];
        let gotw = inv_gaussian_aic(&y, dev, n, Some(&w), None);
        let wantw = n as f64 * ((2.0 * std::f64::consts::PI * disp).ln() + 1.0)
            + 3.0 * y.iter().map(|v| v.ln()).sum::<f64>()
            - w.iter().map(|wi| wi.ln()).sum::<f64>()
            + 2.0;
        assert!((gotw - wantw).abs() < 1e-12, "got {gotw} want {wantw}");
        // Unit weights must reproduce the unweighted value exactly (ln 1 = 0).
        let ones = [1.0_f64; 4];
        let via_ones = inv_gaussian_aic(&y, dev, n, Some(&ones), None);
        assert!((via_ones - got).abs() < 1e-12);
    }

    /// A held φ (`held_phi = Some(v)`) does NOT collapse `D/φ` to `n` — that
    /// identity only holds at the profiled `φ̂ = D/n`. At an arbitrary `v` the
    /// formula keeps `D/v` literal: `−2ℓ = D/v + n·ln(2πv) + 3Σln yᵢ − Σln wᵢ`.
    /// Cross-checked against R's closed form (`sum(statmod::dinvgauss(y, mu,
    /// dispersion = v))` reduces to the same expression at fixed `v`, since
    /// `dinvgauss`'s `1/dispersion` scaling is this family's `1/φ`).
    #[test]
    fn inv_gaussian_aic_matches_precision_formula_at_held_phi() {
        let y = [1.5_f64, 2.0, 0.75, 3.25];
        let n = y.len();
        let dev = 0.42_f64;
        let v = 0.3_f64; // deliberately NOT dev/n (0.105), so D/φ != n.
        let got = inv_gaussian_aic(&y, dev, n, None, Some(v));
        let want = dev / v
            + n as f64 * (2.0 * std::f64::consts::PI * v).ln()
            + 3.0 * y.iter().map(|v| v.ln()).sum::<f64>()
            + 2.0;
        assert!((got - want).abs() < 1e-12, "got {got} want {want}");
        // Precision weights carry the same −Σln wᵢ term as the profiled case.
        let w = [1.0_f64, 2.0, 0.5, 1.5];
        let gotw = inv_gaussian_aic(&y, dev, n, Some(&w), Some(v));
        let wantw = want - w.iter().map(|wi| wi.ln()).sum::<f64>();
        assert!((gotw - wantw).abs() < 1e-12, "got {gotw} want {wantw}");
        // A held φ EQUAL to the profiled value must reproduce the profiled AIC.
        let disp = dev / n as f64;
        let via_held = inv_gaussian_aic(&y, dev, n, None, Some(disp));
        let via_profiled = inv_gaussian_aic(&y, dev, n, None, None);
        assert!(
            (via_held - via_profiled).abs() < 1e-12,
            "held-at-profiled {via_held} vs profiled {via_profiled}"
        );
    }
}
