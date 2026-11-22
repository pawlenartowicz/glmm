//! Family/link IRLS math primitives for all six families: Gaussian, binomial,
//! Poisson, Gamma, negative binomial and inverse Gaussian.
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
/// it so `e^η` never overflows to `inf` mid-IRLS. Bounded by the owned SIMD
/// `exp`'s domain, `simd_transcendental`'s `[EXP_ARG_FLOOR, EXP_ARG_CEIL]`,
/// which the family kernel's clamped η feeds; a compile-time assert there
/// holds it inside.
pub(crate) const ETA_MAX: f64 = 700.0;
/// Floor for a data value used as a null mean on a count family
/// ([`clamp_mu`]), so an all-zero response cannot seed `ln 0`. The fitted μ
/// itself carries no floor: every row quantity stays finite without one (see
/// [`in_tail`]).
pub(crate) const MU_FLOOR: f64 = 1e-10;
/// Binomial μ bound: probit and cloglog store μ inside `[PROB_EPS, 1−PROB_EPS]`
/// so `V(μ) = μ(1−μ)` is never 0 in a μ-form formula, and on every binomial link
/// a row with μ at or past it is a tail row ([`in_tail`]) whose deviance,
/// score and weights are computed from η instead.
pub(crate) const PROB_EPS: f64 = 1e-12;
/// Binomial upper-side tail switch: a row with `1 − μ` at or below this is a
/// tail row ([`in_tail`]). The μ forms take `1 − μ` by subtraction from the
/// stored μ, which keeps only μ's absolute precision, `ε/(1 − μ)` relative:
/// 1e-12 at this switch, but 1e-4 near `PROB_EPS`, where on a y = 0 row it
/// goes straight into the score `−μ'/(1 − μ)` and moves the fit's maximum.
/// The switch is not at μ = ½, where the subtraction is still exact, because
/// the probit η form needs an `erfc` per row: over the whole upper half of
/// the rows that costs a probit GLMM fit 1.9 to 2.6 times its instructions.
pub(crate) const UPPER_TAIL_Q: f64 = 1e-4;
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

/// True when `eta` has no mean in the family's range: η NaN or ≤ 0 on the
/// Gamma inverse and inverse-Gaussian `InverseSquared` links, whose mean
/// `1/η` or `η^(−1/2)` must be positive. [`clamp_eta`] would map such η to
/// a mean near `1/INV_ETA_MIN`, so a caller that reports a mean asks here
/// first.
pub(crate) fn mean_out_of_range(family: Family, eta: f64) -> bool {
    is_reciprocal_link(family) && (eta.is_nan() || eta <= 0.0)
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
/// a trial iterate with such a row as a failed step (R `glm.fit`'s `valideta`
/// step-halving) and pulls it back toward the last accepted iterate, in one
/// evaluation, to the largest power-of-two fraction of the step that is
/// inside the bounds, and the exit refresh refuses such a point, because
/// letting [`clamp_eta`]'s projection stand would let the solve converge ON
/// the bound. Per bound:
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
/// false on a NaN η. GLM IRLS halves such a step too, and ends the fit
/// non-converged when its seed is infeasible or halving cannot recover.
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

/// Floor a data value used as a null mean into its family's μ domain:
/// `≥ MU_FLOOR` for Poisson/NB, whose mean of y can be 0, and
/// `[PROB_EPS, 1−PROB_EPS]` for binomial. Gamma and inverse-Gaussian y are
/// positive, so their mean passes through, as does Gaussian: a floor there
/// would put a units limit on the null deviance. The fit's own μ never goes
/// through this: [`link_inv`] bounds only the probit and cloglog μ, at
/// `PROB_EPS`.
pub(crate) fn clamp_mu<T: Scalar>(family: Family, mu: T) -> T {
    match family {
        Family::Binomial { .. } => mu.clamp_f64(PROB_EPS, 1.0 - PROB_EPS),
        Family::Poisson { .. } | Family::NegativeBinomial { .. } => mu.max_f64(MU_FLOOR),
        Family::Gamma { .. } | Family::InverseGaussian { .. } | Family::Gaussian => mu,
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

/// `ln(a/b)` for `a, b > 0`, given `d = a − b`. A unit deviance multiplies
/// this log by the row's scale (`y` on Poisson and NB, the prior weight on
/// binomial and Gamma) and subtracts a term of nearly the same size, so near
/// `a = b` the log's absolute error is what survives. From a quotient or a
/// difference of two logs that error is `ε`-sized, `ε·y` in the deviance:
/// from counts (or trial counts) near 1e9 it outgrows the PIRLS exit band and
/// the fit stops converging. `ln_1p(d/b)` keeps the log's relative precision
/// instead, because the callers form `d` from `y − μ`, which carries no
/// cancellation error. Below `a = b/2` it takes the difference of the two
/// logs, as precise there, which stays finite for a subnormal `a`, where
/// `d/b` would round to −1 (and a quotient `a/b` to 0).
fn ln_ratio<T: Scalar>(a: f64, b: T, d: T) -> T {
    if a >= 0.5 * b.value() {
        (d / b).ln_1p()
    } else {
        T::from_f64(a.ln()) - b.ln()
    }
}

/// Per-observation deviance contribution `dᵢ ≥ 0` (`Σ dᵢ` is the GLM deviance,
/// −2·log-likelihood up to the saturated constant). Zero at `y=μ`. Every log
/// of a ratio goes through [`ln_ratio`].
pub(crate) fn dev_resid<T: Scalar>(family: Family, nb_theta: f64, y: f64, mu: T) -> T {
    let y_minus_mu = T::from_f64(y) - mu;
    match family {
        Family::Gaussian => {
            let r = T::from_f64(y) - mu;
            r * r
        }
        // 2[ y ln(y/μ) + (1−y) ln((1−y)/(1−μ)) ], with the 0·ln0→0 limits.
        Family::Binomial { .. } => {
            let a = if y > 0.0 {
                T::from_f64(y) * ln_ratio(y, mu, y_minus_mu)
            } else {
                T::ZERO
            };
            let b = if y < 1.0 {
                T::from_f64(1.0 - y) * ln_ratio(1.0 - y, T::ONE - mu, -y_minus_mu)
            } else {
                T::ZERO
            };
            T::from_f64(2.0) * (a + b)
        }
        // 2[ y ln(y/μ) − (y−μ) ], y·ln(y/μ)→0 at y=0.
        Family::Poisson { .. } => {
            let t = if y > 0.0 {
                T::from_f64(y) * ln_ratio(y, mu, y_minus_mu)
            } else {
                T::ZERO
            };
            T::from_f64(2.0) * (t - y_minus_mu)
        }
        // 2[ −ln(y/μ) + (y−μ)/μ ]; same form for log and inverse links.
        Family::Gamma { .. } => T::from_f64(2.0) * (-ln_ratio(y, mu, y_minus_mu) + y_minus_mu / mu),
        // 2[ y ln(y/μ) − (y+θ) ln((y+θ)/(μ+θ)) ], θ = nb_theta, rearranged to
        // 2[ y ln(a/b) − θ ln((y+θ)/(μ+θ)) ], a = y/(y+θ), b = μ/(μ+θ),
        // y·ln(a/b)→0 at y=0. The two forms are equal, but in the first both
        // terms are of order |y−μ| and cancel to a deviance of order θ when
        // μ ≫ θ: from counts near 1e10 that leaves about 2e-7 of round-off
        // per row, 6e-6 on 200 rows, past the PIRLS exit band. In the second
        // neither term is larger than in the first, and when μ ≫ θ both are
        // of order θ. The first log's `a − b` is θ(y−μ)/((y+θ)(μ+θ)) and the
        // second's is `y − μ`, so `ln_ratio` gets both without cancellation.
        // Mirrored by the NB arm of `simd_transcendental::family_pass` —
        // change together.
        Family::NegativeBinomial { .. } => {
            let th = T::from_f64(nb_theta);
            let mu_th = mu + th;
            let t = if y > 0.0 {
                let y_th = y + nb_theta;
                let a_minus_b = th * y_minus_mu / (mu_th * T::from_f64(y_th));
                T::from_f64(y) * ln_ratio(y / y_th, mu / mu_th, a_minus_b)
            } else {
                T::ZERO
            };
            T::from_f64(2.0) * (t - th * ln_ratio(y + nb_theta, mu_th, y_minus_mu))
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
/// - **Binomial**, μ at or past `PROB_EPS` from 0, where probit's `φ(η)` and
///   `Φ(η)` underflow together, or `1 − μ` at or below `UPPER_TAIL_Q`, where
///   the μ forms' `1 − μ` has lost digits. `μ` here is the stored value, so
///   probit and cloglog (bounded by [`link_inv`]) enter the lower side at the
///   bound and logit (unbounded) past it.
/// - **NB, Gamma-log, inverse-Gaussian-log**, `|η| > LOG_TAIL_ETA`: `V(μ)`
///   nears overflow or underflow, and with it the μ-form score and weight.
///
/// Poisson, Gaussian and the inverse links have no tail: their μ forms stay
/// finite on the whole clamped η range. The tail forms are the same functions
/// of η as the μ forms, written to stay finite, so the objective keeps growing
/// on a tail row and the score and weights stay its exact derivatives; the
/// two meet at the switch to round-off (to the μ form's own 1e-12 on the
/// binomial upper side). Branches on values only.
pub(crate) fn in_tail(family: Family, eta: f64, mu: f64) -> bool {
    match family {
        Family::Binomial { .. } => mu <= PROB_EPS || mu >= 1.0 - UPPER_TAIL_Q,
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

/// `(ln Φ(x), λ, λ', λ'')` with `λ = φ/Φ` the inverse Mills ratio, at any `x`.
///
/// Far small side, `x = −t < −4√2`: Cody's third `erfc` region
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
/// Elsewhere Φ comes from `erfc` ([`Scalar::probit_cdf`]): directly on the
/// small side, where it keeps its relative precision, and on the large side
/// `x ≥ 0` as `1 − p` with `p = Φ(−x)`, so that `ln Φ(x) = ln_1p(−p)` keeps
/// it too. Then `λ = φ(x)/Φ(x)`, and `λ'`, `λ''` from the Mills-ratio
/// equation, whose cancellation grows toward `x = −4√2`: there `λ'` is off
/// by up to 1e-13 and `λ''` by up to 1e-10 relative (against 50-digit
/// values), and on `x ≥ 0` neither cancels.
pub(crate) fn probit_side<T: Scalar>(x: T) -> (T, T, T, T) {
    use crate::simd_transcendental::{erfc_region3_r, SQRT_PI};
    if x.value() < -4.0 * std::f64::consts::SQRT_2 {
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
        return (ln_phi, lam, d1, d2);
    }
    let (ln_phi, cdf) = if x.value() < 0.0 {
        let c = x.probit_cdf();
        (c.ln(), c)
    } else {
        let p = (-x).probit_cdf();
        ((-p).ln_1p(), T::ONE - p)
    };
    let lam = T::from_f64(FRAC_1_SQRT_2PI) * (T::from_f64(-0.5) * x * x).exp() / cdf;
    let d1 = -(lam * (x + lam));
    let d2 = -(d1 * (x + lam)) - lam * (T::ONE + d1);
    (ln_phi, lam, d1, d2)
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
    let dmu = mu_eta_at(family, eta, mu);
    let v = variance(family, nb_theta, mu);
    T::from_f64(prior_w) * dmu * (T::from_f64(y) - mu) / v
}

/// Rows per tile of [`ProbitRows`]' `φ(η)`.
pub(crate) const PROBIT_PHI_TILE: usize = 64;

const PROBIT: Family = Family::Binomial {
    link: BinomialLink::Probit,
};

/// The `f64` probit row functions ([`row_score`], [`observed_weight`],
/// [`observed_weight_eta_deriv`]) over one pass's rows, with `φ(η) = dμ/dη`
/// filled [`PROBIT_PHI_TILE`] rows at a time through the SIMD `exp`
/// (`simd_transcendental::exp_fill`) instead of one libm `exp` per row and per
/// function: on a probit Laplace fit a per-row libm `exp` is the largest scalar
/// cost of the PIRLS iteration after the deviance fold.
///
/// Each method is its generic counterpart's off-tail expression, left to right
/// and term for term, with the tile's `φ` in place of `mu_eta`: the value
/// differs from it only by the SIMD `exp`'s last bits (`simd_exp_reduced`,
/// the kernel the family pass's own arms use). A tail row ([`in_tail`]) calls
/// the generic function itself, so its η form is unchanged, and reads no `φ`.
/// `φ`'s argument `−½·η²` is `mu_eta`'s `(−½·η)·η` (probit takes no η clamp);
/// off the tail `−7.04 < η < 3.72`, inside `exp_fill`'s `[−700, 700]`.
///
/// A row reads its own tile entry, refilled when the row leaves the tile, so
/// the callers keep their row loops and row order (and so their sums'
/// association); a pass that walks the rows in order fills each tile once, and
/// any other order is still exact, only slower. The tile is the only state,
/// on the stack: no per-row buffer.
pub(crate) struct ProbitRows<'a> {
    eta: &'a [f64],
    prob: &'a [f64],
    phi: [f64; PROBIT_PHI_TILE],
    start: usize,
    len: usize,
}

impl<'a> ProbitRows<'a> {
    /// `Some` on a probit fit whose η and μ are plain `f64` (the PIRLS loops,
    /// `T = f64`); `None` on every other family and on a dual scalar, whose
    /// rows keep the generic functions.
    pub(crate) fn new<T: Scalar>(family: Family, eta: &'a [T], prob: &'a [T]) -> Option<Self> {
        if !matches!(
            family,
            Family::Binomial {
                link: BinomialLink::Probit
            }
        ) {
            return None;
        }
        Some(Self {
            eta: T::as_f64s(eta)?,
            prob: T::as_f64s(prob)?,
            phi: [0.0; PROBIT_PHI_TILE],
            start: 0,
            len: 0,
        })
    }

    #[inline]
    fn phi(&mut self, i: usize) -> f64 {
        if i.wrapping_sub(self.start) >= self.len {
            self.fill(i);
        }
        self.phi[i - self.start]
    }

    #[inline(never)]
    fn fill(&mut self, i: usize) {
        let len = (self.eta.len() - i).min(PROBIT_PHI_TILE);
        let tile = &mut self.phi[..len];
        for (o, &e) in tile.iter_mut().zip(&self.eta[i..i + len]) {
            *o = -0.5 * e * e;
        }
        crate::simd_transcendental::exp_fill(tile);
        for o in tile.iter_mut() {
            *o *= FRAC_1_SQRT_2PI;
        }
        self.start = i;
        self.len = len;
    }

    /// [`row_score`] at row `i`.
    #[inline]
    pub(crate) fn score(&mut self, i: usize, y: f64, prior_w: f64) -> f64 {
        let (eta, mu) = (self.eta[i], self.prob[i]);
        if in_tail(PROBIT, eta, mu) {
            return row_score(PROBIT, f64::NAN, y, prior_w, eta, mu);
        }
        let v = mu * (1.0 - mu);
        prior_w * self.phi(i) * (y - mu) / v
    }

    /// [`observed_weight`] at row `i`, from its Fisher weight `w`.
    #[inline]
    pub(crate) fn observed_weight(&mut self, i: usize, y: f64, prior_w: f64, w: f64) -> f64 {
        let (eta, mu) = (self.eta[i], self.prob[i]);
        if in_tail(PROBIT, eta, mu) {
            return observed_weight(PROBIT, f64::NAN, y, prior_w, eta, mu, w);
        }
        let dr = probit_dr(eta, mu, self.phi(i));
        w - prior_w * (y - mu) * dr
    }

    /// [`observed_weight_eta_deriv`] at row `i` (`μ'' = −η·φ`, `mu_eta_eta`'s
    /// probit arm).
    #[inline]
    pub(crate) fn observed_weight_eta_deriv(&mut self, i: usize, y: f64, prior_w: f64) -> f64 {
        let (eta, mu) = (self.eta[i], self.prob[i]);
        if in_tail(PROBIT, eta, mu) {
            return observed_weight_eta_deriv(PROBIT, f64::NAN, y, prior_w, eta, mu, 0.0);
        }
        let two = 2.0;
        let d1 = self.phi(i);
        let d2 = -eta * d1;
        let v = mu * (1.0 - mu);
        let g = d1 / v;
        let (g1, g2) = probit_g_derivs(eta, mu, v, d1);
        prior_w * (d2 * g + two * d1 * g1 - (y - mu) * g2)
    }
}

/// Probit `dr/dη` of [`observed_weight`], `−φ(ηV + φ(1−2μ))/V²`, with
/// `φ = dμ/dη` handed in: the generic arm passes `mu_eta`'s, [`ProbitRows`]
/// its tile's.
#[inline]
fn probit_dr<T: Scalar>(eta: T, mu: T, phi: T) -> T {
    let v = mu * (T::ONE - mu);
    -phi * (eta * v + phi * (T::ONE - T::from_f64(2.0) * mu)) / (v * v)
}

/// Probit `(g', g'')` of [`observed_weight_eta_deriv`] at `V = μ(1−μ)`, with
/// `d1 = φ = dμ/dη` handed in: the generic arm passes `mu_eta`'s,
/// [`ProbitRows`] its tile's.
#[inline]
fn probit_g_derivs<T: Scalar>(eta: T, mu: T, v: T, d1: T) -> (T, T) {
    let two = T::from_f64(2.0);
    let k = eta * v + d1 * (T::ONE - two * mu);
    let nn = -d1 * k;
    let dv = (T::ONE - two * mu) * d1;
    let dn = eta * d1 * k - d1 * (v - two * d1 * d1);
    (nn / (v * v), dn / (v * v) - two * nn * dv / (v * v * v))
}

/// Whether `μ = e^η` is the link inverse: then `dμ/dη = d²μ/dη² = μ`.
#[inline]
pub(crate) fn is_log_link(family: Family) -> bool {
    matches!(
        family,
        Family::Poisson { .. }
            | Family::NegativeBinomial { .. }
            | Family::Gamma {
                link: GammaLink::Log,
            }
            | Family::InverseGaussian {
                link: InverseGaussianLink::Log,
            }
    )
}

/// Whether the link is one of the two reciprocal links, Gamma inverse
/// (`η = 1/μ`) and inverse-Gaussian `InverseSquared` (`η = 1/μ²`): both need
/// `η > 0` and clamp η to `[INV_ETA_MIN, INV_ETA_MAX]`.
#[inline]
pub(crate) fn is_reciprocal_link(family: Family) -> bool {
    matches!(
        family,
        Family::Gamma {
            link: GammaLink::Inverse,
        } | Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        }
    )
}

/// [`mu_eta`] at a row whose stored μ belongs to η: on a log link that is μ
/// itself (`dμ/dη = e^η = μ`), read rather than recomputed through a second
/// `exp`; every other link goes to [`mu_eta`]. The stored μ comes from the
/// family pass's own `exp` (the SIMD kernel on the `f64` path), so the value
/// differs from `mu_eta`'s libm `exp` in the last bits at most.
#[inline]
pub(crate) fn mu_eta_at<T: Scalar>(family: Family, eta: T, mu: T) -> T {
    if is_log_link(family) {
        mu
    } else {
        mu_eta(family, eta)
    }
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
/// geometric mean of `w`, `1.0` for unit weights. Every route of a family
/// with a dispersion fits on `ŵ = w/s` (`standardize::Units`) and reports the
/// dispersion times `s`; the GLM IRLS floor's case scale and the held Gamma
/// dispersion check in `fit::common::validate` read it too. On Gamma the
/// row shape is `aᵢ = wᵢ/φ = ŵᵢ/(φ/s)`, so
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

/// Whether the deviance must take its weighted form: prior weights are given,
/// or a logit binomial response has a value outside {0, 1}, where the fused
/// Bernoulli deviance does not hold.
pub(crate) fn weighted_deviance_needed(family: Family, y: &[f64], prior_w: Option<&[f64]>) -> bool {
    prior_w.is_some()
        || (matches!(
            family,
            Family::Binomial {
                link: BinomialLink::Logit
            }
        ) && y.iter().any(|&v| v != 0.0 && v != 1.0))
}

/// `2a − 2a·ln a + 2·lnΓ(a)`, that is `2·(lnΓ(a) − a·ln a + a)`, for `a > 0`,
/// to absolute round-off at every `a`. The Gamma dispersion term
/// ([`gamma_dispersion_term`]), the Poisson and binomial rows of
/// [`saturated_loglik`] and the NB saturated log-likelihood
/// (`fit::glm::nb_saturated_loglik_grouped`) are sums of it.
pub(crate) fn ln_gamma_excess2(a: f64) -> f64 {
    if a < 20.0 {
        return 2.0 * a - 2.0 * a * a.ln() + 2.0 * crate::simd_transcendental::ln_gamma(a);
    }
    // Above a = 20 the three terms, each of order a·ln a, cancel to a
    // value of order ln a: measured against 60-digit values, the direct
    // form is 1e-10 relative off at a = 1e6, 2e-6 at 1e10, and keeps no
    // correct digit at 1e16 (a near-exact fit's φ̂). So the value comes from
    // Stirling's series instead, 2a − 2a·ln a + 2·lnΓ(a) = ln 2π − ln a +
    // 2·R(a), R(a) = 1/(12a) − 1/(360a³) + 1/(1260a⁵) − 1/(1680a⁷)
    // (Abramowitz & Stegun 6.1.41), truncated where the next term, 1/(1188a⁹),
    // is 1.6e-15 at a = 20. Same switch point as `ln_minus_digamma`.
    let r = 1.0 / a;
    let r2 = r * r;
    let stirling = r * (1.0 / 12.0 + r2 * (-1.0 / 360.0 + r2 * (1.0 / 1260.0 - r2 / 1680.0)));
    (2.0 * std::f64::consts::PI).ln() - a.ln() + 2.0 * stirling
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
    let term = match weights {
        None => n as f64 * ln_gamma_excess2(a),
        Some(w) => w[..n].iter().map(|&wi| ln_gamma_excess2(wi * a)).sum(),
    };
    term + 2.0 * sum_ln_y
}

/// `∂/∂ψ` of [`gamma_dispersion_term`], `ψ = ln φ`: `2·Σᵢ aᵢ·(ln aᵢ − ψ₀(aᵢ))`,
/// ψ₀ the digamma function, `aᵢ = wᵢ·a`. Setting `∂/∂ψ (D/φ + term) = 0` at unit
/// weights gives `ln a − ψ₀(a) = D/(2n)`, the Gamma maximum-likelihood shape
/// equation. Both arms take `ln a − ψ₀(a)` from `ln_minus_digamma`: the direct
/// difference cancels at large `a`, that is at small φ.
pub(crate) fn gamma_dispersion_term_d1(ln_phi: f64, weights: Option<&[f64]>, n: usize) -> f64 {
    let a = (-ln_phi).exp();
    let term = match weights {
        None => n as f64 * a * ln_minus_digamma(a).0,
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
/// plus `D/φ` (`glmm::se::joint_hessian_cov`). Both arms read
/// `(g, dg) = ln_minus_digamma(aᵢ)` and use `1 − aᵢ·ψ₁(aᵢ) = aᵢ·dg`, the same
/// cancelling difference the ML solver below needs at large `aᵢ`.
pub(crate) fn gamma_dispersion_term_d2(ln_phi: f64, weights: Option<&[f64]>, n: usize) -> f64 {
    let a = (-ln_phi).exp();
    let term = match weights {
        None => {
            let (g, dg) = ln_minus_digamma(a);
            n as f64 * a * (g + a * dg)
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
/// through `ln_minus_digamma` as `_d2` does — at large per-row
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
/// is about 2e-16 relative at `a = 20` and smaller above. Separate from
/// `dual::digamma`'s series on purpose (switch point 20, not 14): this one
/// needs the form free of cancellation.
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
/// and `held_phi = None` this is R's `inverse.gaussian()$aic` verbatim, and the
/// `−Σ ln wᵢ` term is absent. Requires
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
///   `sᵢ = wᵢ·yᵢ` successes (unit weights and 0/1 y ⇒ Bernoulli, where every
///   term is 0, the condition [`weighted_deviance_needed`] decides; a
///   fractional y with unit weights keeps its nonzero term),
///   so the saturated density at `μ=y` is `ln C(mᵢ,sᵢ) + sᵢ·ln yᵢ +
///   (mᵢ−sᵢ)·ln(1−yᵢ)` with the binomial coefficient via `lnΓ` (continuous in
///   `wᵢ`; R's `dbinom` rounds — identical on the integer trial counts the
///   aggregated convention carries).
/// - **Poisson** — `wᵢ·(yᵢ·ln yᵢ − yᵢ − lnΓ(yᵢ+1))` (0 at `yᵢ=0`).
/// - **NegativeBinomial** — `nb_profile_loglik(y, y, θ, w)`.
/// - **Gaussian** — 0 (its `dev_resid` is the bare RSS; the Gaussian paths
///   build their log-likelihood directly and never call this).
/// - **Gamma** — NaN on purpose: both Gamma paths carry the whole log-density
///   through [`gamma_dispersion_term`] at their φ̂, so there is no saturated
///   term to restore; a caller reaching this arm is a bug, surfaced as NaN.
/// - **InverseGaussian** — NaN for the same reason as Gamma: the objective
///   substitutes `inv_gaussian_aic`, which already carries the profiled
///   dispersion, so there is no free-standing saturated constant to restore.
pub(crate) fn saturated_loglik(
    family: Family,
    nb_theta: f64,
    y: &[f64],
    prior_w: Option<&[f64]>,
) -> f64 {
    // Both rows are written through `E = ln_gamma_excess2`, with
    // `lnΓ(a+1) = ½·E(a) − a + a·ln a + ln a`: the `lnΓ` terms and the
    // `a·ln a`-sized parts then cancel in closed form, so every term stays of
    // order `ln a`. Formed directly, the terms are of order `y·ln y` and the
    // row keeps only that sum's ulp (1e-5 absolute at `y = 1e10`).
    let e = ln_gamma_excess2;
    match family {
        Family::Gaussian => 0.0,
        Family::Gamma { .. } | Family::InverseGaussian { .. } => f64::NAN,
        Family::Binomial { .. } => {
            // `s = m·y` successes and `f = m − s` failures; the row is 0 when
            // either is 0, and otherwise `½·(E(m) − E(s) − E(f)) + ln m − ln s
            // − ln f` (the `y = s/m` logs cancel the `a·ln a` parts exactly).
            let mut s = 0.0;
            for (i, &yi) in y.iter().enumerate() {
                let m = prior_w.map_or(1.0, |w| w[i]);
                let succ = m * yi;
                let fail = m - succ;
                if succ > 0.0 && fail > 0.0 {
                    s += 0.5 * (e(m) - e(succ) - e(fail)) + m.ln() - succ.ln() - fail.ln();
                }
            }
            s
        }
        Family::Poisson { .. } => {
            let mut s = 0.0;
            for (i, &yi) in y.iter().enumerate() {
                if yi > 0.0 {
                    s += prior_w.map_or(1.0, |w| w[i]) * (-0.5 * e(yi) - yi.ln());
                }
            }
            s
        }
        Family::NegativeBinomial { .. } => crate::fit::nb_profile_loglik(y, y, nb_theta, prior_w),
    }
}

/// Pearson-moment dispersion `φ̂ = Σᵢ wᵢrᵢ²/(n−p)`, `rᵢ = (yᵢ−μᵢ)/√V(μᵢ)`, raw
/// `n−p` degrees of freedom (not `Σwᵢ−p`) — matches R's `summary.glm`'s
/// (weighted) Pearson dispersion. `prior_w = None` ⇒ unit weights. Gamma and
/// inverse Gaussian only, the families that estimate φ.
pub(crate) fn pearson_dispersion(
    y: &[f64],
    mu: &[f64],
    family: Family,
    n: usize,
    p: usize,
    prior_w: Option<&[f64]>,
) -> f64 {
    let mut s = 0.0;
    for i in 0..n {
        // √V without forming V: on the log links μ² (Gamma) and μ³
        // (inverse-Gaussian) leave the f64 range at |ln μ| ≈ 354 and ≈ 236,
        // where √V would round to 0 or inf and φ̂ to inf or 0.
        let r = match family {
            Family::Gamma { .. } => (y[i] - mu[i]) / mu[i],
            Family::InverseGaussian { .. } => (y[i] - mu[i]) / mu[i] / mu[i].sqrt(),
            _ => unreachable!("no Pearson φ̂ on {family:?}"),
        };
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
        } => probit_dr(eta, mu, mu_eta(family, eta)),
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
/// `observed_weight` in `observed_weight_eta_deriv_matches_dual1_of_observed_weight`.
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
    // Log links: `μ' = μ'' = μ` (see `mu_eta_at`).
    let (d1, d2) = if is_log_link(family) {
        (mu, mu)
    } else {
        (mu_eta(family, eta), mu_eta_eta(family, eta))
    };
    let v = variance(family, nb_theta, mu);
    let g = d1 / v;
    let (g1, g2) = match family {
        Family::Binomial {
            link: BinomialLink::Probit,
        } => probit_g_derivs(eta, mu, v, d1),
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
/// probit/cloglog μ sits on its `PROB_EPS` bound there or carries a `1 − μ`
/// short of digits. Logit keeps `w(1−2μ)`, which forms no `1 − μ`.
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
#[path = "family_tests.rs"]
mod tests;
