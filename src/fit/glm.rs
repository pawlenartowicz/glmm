//! GLM (fixed-effects binomial/Poisson/Gamma/negative-binomial, `re: None`)
//! dispatch — estimator dispatch, the numerical kernel lives in `src/glm.rs`.
//! Owns the reusable IRLS scratch (`GlmScratchBuf`), maps `GlmFitView` to
//! `Fit`, and drives the NB alternating outer-θ loop (`MASS::glm.nb` style).

use faer::Mat;

use crate::glm::{glm_irls_fit_mapped, GlmFitView, GlmScratch};
use crate::inference::Objective;
use crate::standardize::ColumnScaling;
use crate::{Family, NegBinomialLink};

use super::common::{fill_se_compact, nan_vcov, residual_df, vcov_from_chol, FitDiagnostics};
use super::{Diagnostics, Fit, FitOptions};

impl GlmFitView<'_> {
    /// This route's [`FitDiagnostics`]. No θ, so no boundary or pin state. The
    /// pivot is measured on the CONVERGED `X'WX`, and it shares OLS's floor
    /// because the two were calibrated together — the IRLS weights are what
    /// makes it worth reporting: they come out of the fit itself, so nothing
    /// upstream can predict the conditioning of the matrix actually solved.
    pub(crate) fn diagnostics(&self) -> FitDiagnostics {
        FitDiagnostics {
            pivot: self.pivot,
            pivot_col: self.pivot_col,
            ill_conditioned: self.pivot < crate::ols::PIVOT_MIN,
            separated: self.separated,
            se_unavailable: self.se_unavailable,
            ..FitDiagnostics::fixed_only(self.converged)
        }
    }
}

/// Builds the standard non-converged NaN `Fit` used to seed
/// [`fit_glm_nb_capped`]'s θ↔β alternation before the first inner IRLS fit
/// runs. If `max_outer` were ever `0` this is what the loop returns unmodified,
/// `dispersion: NaN` included — the loop only overwrites `dispersion` with the
/// alternation's θ when the final fit converged; in practice `max_outer >= 1`
/// always, so the first iteration replaces it before it's read.
fn fit_unsupported_family(n: usize, p: usize) -> Fit {
    Fit {
        beta: vec![f64::NAN; p],
        se: vec![f64::NAN; p],
        vcov: nan_vcov(p),
        tau2: vec![],
        dispersion: f64::NAN,
        diagnostics: Diagnostics::from_flags(false, false, p),
        varcorr: vec![],
        stddev_se: vec![],
        n_eval: 0,
        #[cfg(feature = "counters")]
        counters: crate::counters::EvalCounters::new(),
        deviance: f64::NAN,
        loglik: f64::NAN,
        df: 0,
        reml: false,
        nobs: n,
        objective: Objective::MaximumLikelihood,
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
        condvar: vec![],
        condvar_x: vec![],
        condvar_cross: None,
        coef_df: vec![f64::NAN; p],
        satterthwaite: None,
        design_summary: None,
    }
}
// ---------------------------------------------------------------------------
// GLM dispatch (Binomial{Logit}, re: None)
// ---------------------------------------------------------------------------

/// Owned IRLS scratch for [`fit_glm`], sized off n/p/t exactly as [`GlmScratch`]
/// documents. Hoistable across repeated fixed-θ GLM fits (the NB outer-θ
/// alternation in [`fit_glm_nb_capped`]): [`crate::glm::glm_irls_fit`] re-seeds or NaN-fills
/// every populated slot at entry, so a reused buffer is bit-identical to a fresh
/// allocation — except `mu` (`irls_p`) on the two pre-loop early returns (n ≤ p
/// short-circuit, Binomial all-0/all-1 short-circuit), where `irls_p` is left
/// untouched and `mu` carries the previous fit's means instead of zeros;
/// harmless since `mu`'s own contract gates it on `converged`.
pub(crate) struct GlmScratchBuf {
    irls_eta: Vec<f64>,
    irls_p: Vec<f64>,
    irls_w: Vec<f64>,
    irls_z: Vec<f64>,
    irls_betas: Vec<f64>,
    irls_betas_new: Vec<f64>,
    irls_var_diag: Vec<f64>,
    irls_t_sq: Vec<f64>,
    irls_u_scratch: Vec<f64>,
    irls_xtwx: Mat<f64>,
    irls_xtwz: Vec<f64>,
    irls_l: Mat<f64>,
    irls_wx: Vec<f64>,
}

impl GlmScratchBuf {
    pub(crate) fn new(n: usize, p: usize, t: usize) -> Self {
        let (n1, p1, t1) = (n.max(1), p.max(1), t.max(1));
        GlmScratchBuf {
            irls_eta: vec![0.0f64; n1],
            irls_p: vec![0.0f64; n1],
            irls_w: vec![0.0f64; n1],
            irls_z: vec![0.0f64; n1],
            irls_betas: vec![0.0f64; p1],
            irls_betas_new: vec![0.0f64; p1],
            irls_var_diag: vec![0.0f64; t1],
            irls_t_sq: vec![0.0f64; t1],
            irls_u_scratch: vec![0.0f64; p1],
            irls_xtwx: Mat::<f64>::zeros(p1, p1),
            irls_xtwz: vec![0.0f64; p1],
            irls_l: Mat::<f64>::zeros(p1, p1),
            irls_wx: vec![0.0f64; n1 * p1], // column-major W∘X, needs ≥ n·p
        }
    }

    fn as_scratch(&mut self) -> GlmScratch<'_> {
        GlmScratch {
            irls_eta: &mut self.irls_eta,
            irls_p: &mut self.irls_p,
            irls_w: &mut self.irls_w,
            irls_z: &mut self.irls_z,
            irls_betas: &mut self.irls_betas,
            irls_betas_new: &mut self.irls_betas_new,
            irls_var_diag: &mut self.irls_var_diag,
            irls_t_sq: &mut self.irls_t_sq,
            irls_u_scratch: &mut self.irls_u_scratch,
            irls_xtwx: self.irls_xtwx.as_mut(),
            irls_xtwz: &mut self.irls_xtwz,
            irls_l: self.irls_l.as_mut(),
            irls_wx: &mut self.irls_wx,
        }
    }
}

/// GLM dispatch adapter. Owns the `irls_*` scratch inline (the analog of
/// `OlsWorkspace::new`'s `OlsScratch` allocation; on the simulation path these live in
/// `SimWorkspace`), converts the row-major input to a column-major faer `Mat`,
/// runs the `family`-selected IRLS kernel cold-started at β=0, and maps the view
/// to `Fit`. No random effects ⇒ `tau2` empty. Binomial/Poisson keep
/// `dispersion = 1.0`; Gamma/NB recover their dispersion in their own arms.
/// Test-only baseline since the stable path dispatches through the unified core
/// ([`super::core::fit_on`]) over `fit_glm_prebuilt`/`glm_view_to_fit`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_glm(
    family: Family,
    nb_theta: f64,
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    opts: &FitOptions,
) -> Fit {
    let mut buf = GlmScratchBuf::new(n, p, opts.se_for.columns(p).len());
    let x_mat = super::common::to_col_major(x, n, p);
    let view = fit_glm_prebuilt(
        family,
        nb_theta,
        x_mat.as_ref().subrows(0, n),
        y,
        weights,
        offset,
        &opts.se_for.columns(p),
        &mut buf,
        ColumnScaling::IDENTITY,
        None,
    );
    glm_view_to_fit(
        &view,
        view.betas.to_vec(),
        y,
        weights,
        family,
        nb_theta,
        n,
        p,
        opts,
        ColumnScaling::IDENTITY,
    )
}

/// [`fit_glm`]'s solve half over a prebuilt column-major `x_mat` and reusable
/// IRLS scratch, returning the borrowed [`GlmFitView`]. Hoisted so the NB
/// outer-θ loop ([`fit_glm_nb_capped`]) allocates the scratch ONCE, re-running
/// IRLS per fixed-θ fit; the unified fit core drives it per draw. [`glm_view_to_fit`] maps the returned view to the full `Fit`.
///
/// `map` is the column map `x_mat` was fitted through: the view's `betas` stay
/// on `x_mat`'s design, its `var_diag`/`t_sq` are the caller's
/// (`crate::glm::glm_irls_fit_mapped`). `beta_start` is `glm_irls_fit`'s: `None`
/// is the cold start, `Some(β)` (on `x_mat`'s design) starts from β.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_glm_prebuilt<'a>(
    family: Family,
    nb_theta: f64,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    targets: &[u32],
    buf: &'a mut GlmScratchBuf,
    map: &ColumnScaling,
    beta_start: Option<&[f64]>,
) -> GlmFitView<'a> {
    glm_irls_fit_mapped(
        family,
        nb_theta,
        x_mat,
        y,
        targets,
        beta_start,
        weights,
        offset,
        buf.as_scratch(),
        map,
    )
}

/// Maps a [`GlmFitView`] to the full stable `Fit`: SE from `var_diag`, vcov from
/// the cached Cholesky factor, Gamma dispersion + √φ SE scaling, fitted means μ̂,
/// and the family log-likelihood. Needs raw `y` (Gamma dispersion + saturated
/// loglik read it) and `nb_theta` (the NB shape θ̂; only NB/Gamma read it, pass
/// `f64::NAN` otherwise). `beta` is the caller's β̂, `B·β̃` of the view's, with
/// `map` the column map the view was fitted through.
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_view_to_fit(
    view: &GlmFitView<'_>,
    mut beta: Vec<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    family: Family,
    nb_theta: f64,
    n: usize,
    p: usize,
    opts: &FitOptions,
    map: &ColumnScaling,
) -> Fit {
    // --- map GlmFitView → Fit ---
    // view.var_diag is target-compact [0..t] (like OLS).
    let diag = view.diagnostics();
    let converged = diag.converged;
    // A broken-down column (`GlmFitView::se_unavailable`) sits at its start
    // value: not an estimate, so NaN, as an aliased column's is. The other
    // estimates, the deviance and the fitted means stand; no covariance does.
    if view.se_unavailable {
        beta[view.pivot_col as usize] = f64::NAN;
    }
    // The coefficients the fit estimates: the model df, AIC and the Pearson
    // divisor count them as R's `glm` counts its `rank`, so this fit and one
    // with the column dropped beforehand agree.
    let p_rank = p - usize::from(view.se_unavailable);
    let has_vcov = converged && !view.se_unavailable;
    // Weighted Σwᵢdᵢ at the accepted iterate (NaN unless converged).
    let irls_deviance = view.deviance;
    let mut se = vec![f64::NAN; p];
    fill_se_compact(view.var_diag, &opts.se_for.columns(p), &mut se);
    // Unscaled (X'WX)⁻¹; Gamma's φ multiplies it afterwards, exactly as √φ scales
    // `se`. `view.l` is stale-or-zero unless `converged` without
    // `se_unavailable` (its documented contract).
    let mut vcov = if has_vcov {
        vcov_from_chol(view.l, p, &opts.se_for.columns(p), 1.0, map)
    } else {
        nan_vcov(p)
    };

    // Dispersion. Binomial/Poisson hold φ≡1 (the kernel's `(XᵀWX)⁻¹` is the full
    // covariance). Gamma and inverse-Gaussian recover φ post-fit — the mean
    // model β is φ-independent, so φ stays out of the IRLS — and scale the SE
    // by √φ (the kernel folded φ=1, so Var(β̂)=φ·(XᵀWX)⁻¹). `dispersion: Some(v)`
    // holds φ=v fixed. `None` is the Pearson moment `φ̂=Σ wᵢrᵢ²/(n−p)`,
    // `rᵢ=(yᵢ−μ̂ᵢ)/√V(μ̂ᵢ)`, raw-row df, raw weights — exactly `summary(glm(...,
    // weights=w))$dispersion` on both families (precision weights: row `i` has
    // variance `φ·V(μᵢ)/wᵢ`, so Pearson's `wᵢrᵢ²` is already on that scale).
    // Gamma's maximum-likelihood φ̂ is a DIFFERENT estimator, computed below for
    // `loglik` alone — see the Gamma arm.
    let dispersion = if !converged {
        f64::NAN
    } else {
        match family {
            Family::Gamma { .. } | Family::InverseGaussian { .. } => {
                let phi = match opts.dispersion {
                    Some(v) => v,
                    None => crate::family::pearson_dispersion(
                        y, view.mu, family, nb_theta, n, p_rank, weights,
                    ),
                };
                let sqrt_phi = phi.sqrt();
                for v in se.iter_mut() {
                    if v.is_finite() {
                        *v *= sqrt_phi;
                    }
                }
                phi
            }
            _ => 1.0,
        }
    };

    // Var(β̂) = φ·(X'WX)⁻¹ — the same φ the SE loop applied as √φ. A no-op for
    // the φ≡1 families, where `dispersion` is 1.0.
    for row in vcov.iter_mut() {
        for v in row.iter_mut() {
            *v *= dispersion;
        }
    }

    // Fitted means μ̂ (the kernel's converged IRLS means) and the family
    // log-likelihood on R's `logLik.glm`/`MASS::glm.nb` scale.
    // Binomial/Poisson/NB: the IRLS deviance is the weighted `dev_resid` sum,
    // so restoring the saturated constant gives the exact log-likelihood.
    // Gamma: the log-likelihood at the MAXIMISED φ̂ — a different estimator
    // from `dispersion` above (the Pearson moment) — `−½·(D/φ +
    // gamma_dispersion_term)`, `D` raw-weighted. The ML search and
    // `gamma_dispersion_term` itself run on `ŵ = w/s`: the seed and solver
    // start are not scale-free on raw `w` (raw-weight Newton from MASS's
    // start can overshoot to NaN at large `w`; on `ŵ` it does not), so φ̂ is
    // `s` times the internal root and `gamma_dispersion_term` is read at
    // `ln(φ̂/s)`. A held φ (`opts.dispersion
    // = Some(v)`) is used as-is for `D/φ`, with `Gₚ` still read on `ŵ` at
    // `ln(v/s)` — the same internal coordinate the mixed path's held-φ arm
    // uses (`fit::glmm::run_glmm_on`). Inverse-Gaussian: `opts.dispersion`
    // passes straight through to `inv_gaussian_aic`'s own `held_phi` arm —
    // `None` profiles φ at `D/n`, `Some(v)` holds it at `v`.
    let (fitted, loglik) = if converged {
        let mu = view.mu.to_vec();
        let ll = match family {
            Family::Gamma { .. } => {
                let s = crate::family::weight_scale(weights, n);
                let hat_w: Option<Vec<f64>> =
                    weights.map(|w| w[..n].iter().map(|&wi| wi / s).collect());
                let sum_ln_y: f64 = y[..n].iter().map(|v| v.ln()).sum();
                let phi = match opts.dispersion {
                    Some(v) => v,
                    None => {
                        // The raw-scale ML equation `Σwᵢgᵢ = D/2` becomes, on
                        // `w = s·ŵ`, `Σᵢ ŵᵢ·(ln aᵢ − ψ₀(aᵢ)) = D/(2s)`, so the
                        // deviance handed to the per-row solver is `D/s`, not
                        // raw `D` (see `gamma_ml_dispersion`'s own doc comment).
                        let dev_int = irls_deviance / s;
                        s * crate::family::gamma_ml_dispersion(dev_int, hat_w.as_deref(), n)
                    }
                };
                if phi == 0.0 {
                    // An exact fit: the likelihood rises without bound as φ → 0.
                    f64::INFINITY
                } else {
                    let psi_int = (phi / s).ln();
                    -0.5 * (irls_deviance / phi
                        + crate::family::gamma_dispersion_term(
                            psi_int,
                            hat_w.as_deref(),
                            n,
                            sum_ln_y,
                        ))
                }
            }
            Family::InverseGaussian { .. } => {
                -0.5 * (crate::family::inv_gaussian_aic(
                    y,
                    irls_deviance,
                    n,
                    weights,
                    opts.dispersion,
                ) - 2.0)
            }
            _ => {
                -0.5 * irls_deviance + crate::family::saturated_loglik(family, nb_theta, y, weights)
            }
        };
        (mu, ll)
    } else {
        (vec![], f64::NAN)
    };

    // Gamma and inverse Gaussian with the Pearson φ̂ take t on its divisor
    // `n − p` (`summary.glm`'s `df.residual`), whatever `opts.satterthwaite`
    // says; a held φ, and the families with φ ≡ 1 or an NB shape, keep z.
    let (coef_df, satterthwaite) = match family {
        Family::Gamma { .. } | Family::InverseGaussian { .. } if opts.dispersion.is_none() => {
            let (df, state) = residual_df(
                view.l,
                n,
                p,
                dispersion,
                map,
                &opts.se_for.columns(p),
                has_vcov,
            );
            (df, Some(state))
        }
        _ => (vec![f64::NAN; p], None),
    };

    Fit {
        beta,
        se,
        vcov,
        tau2: vec![],
        dispersion,
        // `singular` is constant false here for the same reason it is on OLS —
        // no θ, so `boundary_hit` stays 0. See `ols.rs`'s note at this field.
        diagnostics: super::common::materialize_diagnostics(&diag, p, &[]),
        varcorr: vec![],
        stddev_se: vec![],
        n_eval: 0,
        #[cfg(feature = "counters")]
        counters: crate::counters::EvalCounters::new(),
        deviance: f64::NAN,
        loglik,
        df: if converged {
            super::common::model_df(family, p_rank, 0, opts.dispersion.is_some())
        } else {
            0
        },
        reml: false,
        nobs: n,
        objective: Objective::MaximumLikelihood,
        fitted,
        ranef: vec![],
        ranef_levels: vec![],
        condvar: vec![],
        condvar_x: vec![],
        condvar_cross: None,
        coef_df,
        satterthwaite,
        design_summary: None,
    }
}
// ---------------------------------------------------------------------------
// Negative-binomial GLM — alternating outer-θ loop (MASS::glm.nb style)
// ---------------------------------------------------------------------------

/// NB θ outer-loop caps. Convergence on `|Δθ|/θ < NB_THETA_TOL`; `NB_MAX_OUTER`
/// alternations between the β fit (fixed θ) and the 1-D θ optimisation.
pub(super) const NB_MAX_OUTER: usize = 25;
const NB_THETA_TOL: f64 = 1e-6;
/// θ search bracket (on ln θ): `θ ∈ [1e-3, 1e4]`. NB variance is `μ + μ²/θ`,
/// so large θ (near `1e4`) drives `μ²/θ` toward zero — near-Poisson, low
/// overdispersion; small θ (near `1e-3`) is the highly overdispersed end,
/// implausibly so beyond this bound for the validation datasets.
pub(crate) const NB_THETA_LO: f64 = 1e-3;
pub(crate) const NB_THETA_HI: f64 = 1e4;

/// NB saturated log-likelihood of one count, `log f(y; μ = y, θ)`:
/// `lnΓ(y+θ) − lnΓ(θ) − lnΓ(y+1) + θ·ln(θ/(θ+y)) + y·ln(y/(θ+y))`, 0 at
/// `y = 0` (and at a NaN `y`, which only the loop tier lets through).
/// `excess_theta` is [`crate::family::ln_gamma_excess2`]`(θ)`, shared by every
/// row. With `E = ln_gamma_excess2` the row is exactly
/// `½·(E(θ+y) − E(θ) − E(y)) − ln y`: the `lnΓ` terms and the `y·ln y`-sized
/// parts cancel inside each `E` in closed form, so every term stays of order
/// `ln y` and the row keeps absolute round-off at any count. Formed directly,
/// the row's terms are of order `y·ln y` (2.2e11 at `y = 1e10`), and the
/// NB GLMM objective, which carries `−2·Σ` of the rows, then resolves nothing
/// finer than that sum's ulp (1.6e-2 on 200 such rows).
fn nb_saturated_row(y: f64, theta: f64, excess_theta: f64) -> f64 {
    if y > 0.0 {
        let e = crate::family::ln_gamma_excess2;
        0.5 * (e(theta + y) - excess_theta - e(y)) - y.ln()
    } else {
        0.0
    }
}

/// NB log-likelihood of `y` at means `μ` and shape θ,
/// `Σ wᵢ·log f(yᵢ; μᵢ, θ)`: each row is its saturated log-likelihood
/// ([`nb_saturated_row`]) less half its unit deviance
/// (`family::dev_resid`), both free of cancellation at large counts.
/// `MASS::theta.ml` maximises the same function of θ at fixed μ̂ (it drops
/// the θ-free `−lnΓ(yᵢ+1)`; this keeps it, so `nb_profile_loglik(y, y, θ, w)`
/// is the NB saturated log-likelihood, the term the NB GLMM marginal-θ
/// objective adds back to the Laplace deviance).
///
/// `weights: Some(w)` multiplies each row's contribution by the prior weight
/// `wᵢ` (matches `MASS::glm.nb(weights=)`'s profile — `theta.ml` weights the
/// per-row deviance terms the same way `glm.fit` weights IRLS); `None` is unit
/// weights.
pub(crate) fn nb_profile_loglik(y: &[f64], mu: &[f64], theta: f64, weights: Option<&[f64]>) -> f64 {
    let family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    let excess_theta = crate::family::ln_gamma_excess2(theta);
    let mut ll = 0.0;
    for (i, (&yi, &mi)) in y.iter().zip(mu.iter()).enumerate() {
        let s = nb_saturated_row(yi, theta, excess_theta)
            - 0.5 * crate::family::dev_resid(family, theta, yi, mi);
        ll += weights.map_or(1.0, |w| w[i]) * s;
    }
    ll
}

/// The rows of `y` grouped by count for [`nb_saturated_loglik_grouped`]:
/// `(yᵥ, Σ wᵢ over the rows with yᵢ = yᵥ)`, ascending in `yᵥ`, written into
/// `out` (cleared first; its capacity is reused across fits). `y` is fixed
/// for a fit, so this runs once per fit.
pub(crate) fn nb_count_groups(y: &[f64], weights: Option<&[f64]>, out: &mut Vec<(f64, f64)>) {
    out.clear();
    for (i, &yi) in y.iter().enumerate() {
        let wi = weights.map_or(1.0, |w| w[i]);
        out.push((yi, wi));
    }
    out.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut k = 0;
    for j in 0..out.len() {
        if k > 0 && out[k - 1].0.to_bits() == out[j].0.to_bits() {
            out[k - 1].1 += out[j].1;
        } else {
            out[k] = out[j];
            k += 1;
        }
    }
    out.truncate(k);
}

/// `nb_profile_loglik(y, y, θ, w)` (the NB saturated log-likelihood) off
/// the count groups of [`nb_count_groups`]: each distinct count's row term
/// is formed once and weighted by its group's total weight. A group's term is
/// that function's row term bit for bit (its unit deviance at `μ = y` is 0);
/// only the outer sum over rows is reassociated (into one product per count).
pub(crate) fn nb_saturated_loglik_grouped(groups: &[(f64, f64)], theta: f64) -> f64 {
    let excess_theta = crate::family::ln_gamma_excess2(theta);
    let mut ll = 0.0;
    for &(yi, wsum) in groups {
        ll += wsum * nb_saturated_row(yi, theta, excess_theta);
    }
    ll
}

/// Maximise `g(ln θ)` over `ln θ ∈ [ln NB_THETA_LO, ln NB_THETA_HI]` by
/// golden-section (the NB likelihood is far more symmetric in ln θ than in θ).
/// Returns `θ̂ = exp(argmax)`. The GLM conditional θ profile
/// ([`optimize_nb_theta`]) is its only caller — the GLMM searches `ln θ` as a
/// coordinate of its outer BOBYQA instead; `g` is the log-likelihood to maximise
/// as a function of `ln θ`.
///
/// Stopping width `1e-7` on `ln θ`, so the returned `θ̂` is the maximum of `g`
/// to within `5e-8` relative. That is finer than the outer alternation's own
/// stop (`NB_THETA_TOL = 1e-6` on `|Δθ|/θ` between rounds), so the alternation,
/// not this bracket, sets how far `θ̂` ends from the joint maximum over `(β, θ)`.
/// A width near or above `NB_THETA_TOL` leaves `θ̂` up to half the width off
/// that maximum with nothing in the loop to see it. A narrower width does not
/// help: `θ̂` is then already limited by `NB_THETA_TOL` (on 400 simulated NB
/// GLMs checked against the joint maximum computed at 50 digits, the worst `θ̂`
/// is 3.4e-7 relative off at this width and 3.6e-7 at `1e-8`), and each further
/// decade costs about five more evaluations of `g` per round. `g` is the
/// closed-form profile at fixed `μ̂`, smooth in `ln θ` with no inner fit: in a
/// 128-draw sweep of 1-ULP changes to `x` and `y` on the NB GLM fixtures, `θ̂`
/// moved by at most 5e-7 relative and never jumped to another branch.
pub(crate) fn golden_max_ln_theta(mut g: impl FnMut(f64) -> f64) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9; // 1/golden ratio
    let (mut a, mut b) = (NB_THETA_LO.ln(), NB_THETA_HI.ln());
    let mut c = b - (b - a) * INV_PHI;
    let mut d = a + (b - a) * INV_PHI;
    let (mut fc, mut fd) = (g(c), g(d));
    for _ in 0..200 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - (b - a) * INV_PHI;
            fc = g(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + (b - a) * INV_PHI;
            fd = g(d);
        }
        if (b - a).abs() < 1e-7 {
            break;
        }
    }
    (0.5 * (a + b)).exp()
}

/// Maximise [`nb_profile_loglik`] over θ at fixed μ̂, weighted by `weights`
/// (`Data::weights`, forwarded from the caller). Returns θ̂.
fn optimize_nb_theta(y: &[f64], mu: &[f64], weights: Option<&[f64]>) -> f64 {
    golden_max_ln_theta(|t| nb_profile_loglik(y, mu, t.exp(), weights))
}

/// Cold-start seed for the NB dispersion: the method-of-moments
/// `θ₀ = ȳ²/max(s²−ȳ, ε)`, clamped into `[NB_THETA_LO, NB_THETA_HI]`. Unweighted on
/// purpose (a seed only). The GLM outer loop's own start; the GLMM `ln θ_NB`
/// coordinate does NOT use it directly — it seeds from `fit_glm_nb`'s θ̂, which
/// reaches this only as that fit's own start. On a GLMM the random effects
/// inflate `s²`, so this seed sits one to two orders of magnitude below θ̂,
/// which is why the GLMM route does not take it.
///
/// With an offset `oᵢ` the rows have different means, and `s²` of the raw y
/// would charge the spread of the exposures `e^{oᵢ}` to overdispersion (θ₀ at
/// the box floor when they differ by decades). So the moments are taken
/// around the null-model means `μ̂ᵢ = r·eᵢ`, `eᵢ = e^{oᵢ}` (1 without an
/// offset), `r = Σy/Σe`: from `E(y−μ)² = μ + μ²/θ`,
/// `θ₀ = mean(μ̂²)/max(Σ(yᵢ−μ̂ᵢ)²/(n−1) − mean(μ̂), ε)`, which is the formula
/// above when every `oᵢ` is equal.
pub(crate) fn nb_theta_moment_seed(y: &[f64], n: usize, offset: Option<&[f64]>) -> f64 {
    let df = (n.max(2) - 1) as f64;
    let e = |i: usize| offset.map_or(1.0, |o| o[i].exp());
    let rate = y[..n].iter().sum::<f64>() / (0..n).map(e).sum::<f64>();
    let (mut m1, mut m2, mut ss) = (0.0, 0.0, 0.0);
    for (i, &yi) in y[..n].iter().enumerate() {
        let mu = rate * e(i);
        m1 += mu;
        m2 += mu * mu;
        ss += (yi - mu) * (yi - mu);
    }
    let theta = (m2 / n as f64) / (ss / df - m1 / n as f64).max(1e-6);
    theta.clamp(NB_THETA_LO, NB_THETA_HI)
}

/// Negative-binomial GLM via the alternating outer-θ loop (`MASS::glm.nb`):
/// (1) fit the GLM at fixed θ; (2) 1-D maximise the NB profile log-likelihood
/// over θ holding β̂/μ̂; (3) repeat to convergence. `theta_seed = Some(v)` seeds
/// step 1; `None` cold-starts from a method-of-moments estimate
/// `θ₀ = ȳ²/max(s²−ȳ, ε)` (seed only, left unweighted — the outer loop's
/// weighted θ optimisation below washes out the seed's influence). The β SE
/// conditions on θ̂ (lme4/MASS convention; θ-uncertainty is out of scope), so
/// `dispersion = θ̂` and the SE comes straight from the final fixed-θ fit (NB
/// has `φ≡1`; overdispersion lives in `V=μ+μ²/θ`). Returns `(Fit, θ)`: the
/// `Fit`'s own `dispersion` is NaN unless the fit converged, but the second
/// element is always the last θ the alternation stood on, for callers that
/// need a seed regardless of convergence (`fit_glmm_nb`'s outer-BOBYQA seed).
///
/// `x` is the column-major design the kernel fits (`n` rows) and `map` the
/// column map it was standardized through (`β = B·β̃`); the returned `Fit` is
/// the caller's.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_glm_nb(
    x: faer::MatRef<'_, f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    theta_seed: Option<f64>,
    opts: &FitOptions,
    map: &ColumnScaling,
) -> (Fit, f64) {
    fit_glm_nb_capped(
        x,
        y,
        weights,
        offset,
        n,
        p,
        theta_seed,
        opts,
        NB_MAX_OUTER,
        map,
    )
}

/// [`fit_glm_nb`]'s loop body with the outer-iteration cap as a parameter —
/// a seam so the cap-exhaustion semantics are testable without engineering a
/// dataset that legitimately burns all `NB_MAX_OUTER` alternations. Production
/// code only ever calls it through [`fit_glm_nb`] (cap = `NB_MAX_OUTER`).
///
/// Cap-exhaustion semantics (pinned by `fit_glm_nb_outer_cap_semantics`): the
/// exit is REPORTED, not silent, and consistent — a cap exit whose last inner
/// IRLS converged is refit at the final θ (β̂/SE recomputed there, not left at
/// a stale pre-update θ), `converged` reflects that refit, and a
/// [`crate::Note::NbShapeUnsettled`] carrying the round count is pushed onto
/// the returned `Fit`'s diagnostics. The returned `Fit::dispersion` carries
/// the newer θ only when the (possibly refit) fit converged (NaN otherwise);
/// the second return value carries the newer θ unconditionally.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_glm_nb_capped(
    x: faer::MatRef<'_, f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    offset: Option<&[f64]>,
    n: usize,
    p: usize,
    theta_seed: Option<f64>,
    opts: &FitOptions,
    max_outer: usize,
    map: &ColumnScaling,
) -> (Fit, f64) {
    let mut theta = theta_seed.unwrap_or_else(|| nb_theta_moment_seed(y, n, offset));

    let family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    // θ-invariant state, built once and reused across the θ↔β alternation: the
    // IRLS scratch and the μ̂ buffer.
    let targets = opts.se_for.columns(p);
    let mut buf = GlmScratchBuf::new(n, p, targets.len());
    let mut mu = vec![0.0f64; n];

    // The caller's β̂ of a view, `B·β̃`.
    let caller_beta = |view: &GlmFitView<'_>| {
        let mut beta = view.betas.to_vec();
        map.beta_to_caller(&mut beta);
        beta
    };
    // Every fit after the first starts from the last fit's β, as `MASS::glm.nb`
    // passes `etastart`. From a cold start each round walks the same path and
    // stops wherever its stop rule first fires. At large counts that is the
    // round-off stop, up to a Fisher step short of the maximum, at a point that
    // jumps between two nearby θ; θ̂ read off those μ̂ then alternates between
    // two values (5e-5 apart on a 20-row fit at μ ≈ 1.6e5) and never settles.
    // From the last β the rounds continue one iteration instead.
    let mut beta_start: Vec<f64> = Vec::with_capacity(p);
    let mut fit_result = fit_unsupported_family(n, p);
    let mut settled = false;
    let mut rounds: u32 = 0;
    for _ in 0..max_outer {
        rounds += 1;
        // θ is fixed for this β fit and threaded explicitly (the spec is θ-free).
        let start = (!beta_start.is_empty()).then_some(&beta_start[..]);
        let view = fit_glm_prebuilt(
            family, theta, x, y, weights, offset, &targets, &mut buf, map, start,
        );
        fit_result = glm_view_to_fit(
            &view,
            caller_beta(&view),
            y,
            weights,
            family,
            theta,
            n,
            p,
            opts,
            map,
        );
        if !fit_result.converged() {
            break;
        }
        beta_start.clear();
        beta_start.extend_from_slice(view.betas);
        // μ̂ = exp(o + Xβ̂) for the θ optimisation. `x` is the design the
        // kernel fitted and `view.betas` its β̃, so `x·β̃ = x_caller·β` with no
        // map (`fit_result.beta` is the caller's β, for the caller's design).
        for (i, mi) in mu.iter_mut().enumerate() {
            let mut eta: f64 = (0..p).map(|j| x[(i, j)] * view.betas[j]).sum();
            if let Some(o) = offset {
                eta += o[i];
            }
            *mi = crate::family::link_inv(family, eta);
        }
        let new_theta = optimize_nb_theta(y, &mu, weights);
        let converged = (new_theta - theta).abs() / theta < NB_THETA_TOL;
        theta = new_theta;
        if converged {
            settled = true;
            // Final β/SE at the converged θ for consistency.
            let view = fit_glm_prebuilt(
                family,
                theta,
                x,
                y,
                weights,
                offset,
                &targets,
                &mut buf,
                map,
                Some(&beta_start),
            );
            fit_result = glm_view_to_fit(
                &view,
                caller_beta(&view),
                y,
                weights,
                family,
                theta,
                n,
                p,
                opts,
                map,
            );
            break;
        }
    }
    if !settled && rounds == max_outer as u32 && fit_result.converged() {
        // Cap exit: refit β̂/SE at the θ this fit reports, so the returned Fit is
        // one model, and say that θ never settled (`Note::NbShapeUnsettled`).
        let view = fit_glm_prebuilt(
            family,
            theta,
            x,
            y,
            weights,
            offset,
            &targets,
            &mut buf,
            map,
            Some(&beta_start),
        );
        fit_result = glm_view_to_fit(
            &view,
            caller_beta(&view),
            y,
            weights,
            family,
            theta,
            n,
            p,
            opts,
            map,
        );
        if fit_result.converged() {
            fit_result
                .diagnostics
                .notes
                .push(crate::Note::NbShapeUnsettled { rounds });
        }
    }
    if fit_result.converged() {
        fit_result.dispersion = theta;
    }
    (fit_result, theta)
}
