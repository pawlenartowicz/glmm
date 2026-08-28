//! GLM (fixed-effects binomial/Poisson/Gamma/negative-binomial, `re: None`)
//! dispatch — estimator dispatch, the numerical kernel lives in `src/glm.rs`.
//! Owns the reusable IRLS scratch (`GlmScratchBuf`), maps `GlmFitView` to
//! `Fit`, and drives the NB alternating outer-θ loop (`MASS::glm.nb` style).

use faer::Mat;

use crate::glm::{glm_irls_fit, GlmFitView, GlmScratch};
use crate::{Family, NegBinomialLink};

use super::common::{fill_se_compact, nan_vcov, to_col_major, vcov_from_chol, FitDiagnostics};
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
            ..FitDiagnostics::fixed_only(self.converged)
        }
    }
}

/// Builds the standard non-converged NaN `Fit` used to seed
/// [`fit_glm_nb_capped`]'s θ↔β alternation before the first inner IRLS fit
/// runs. If `max_outer` were ever `0` this is what the loop would return
/// unmodified; in practice `max_outer >= 1` always, so the first iteration
/// overwrites it before it's read.
fn fit_unsupported_family(p: usize) -> Fit {
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
        deviance: f64::NAN,
        loglik: f64::NAN,
        df: 0,
        reml: false,
        fitted: vec![],
        ranef: vec![],
        ranef_levels: vec![],
    }
}
// ---------------------------------------------------------------------------
// GLM dispatch (Binomial{Logit}, re: None)
// ---------------------------------------------------------------------------

/// Owned IRLS scratch for [`fit_glm`], sized off n/p/t exactly as [`GlmScratch`]
/// documents. Hoistable across repeated fixed-θ GLM fits (the NB outer-θ
/// alternation in [`fit_glm_nb_capped`]): [`glm_irls_fit`] re-seeds or NaN-fills
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
/// `fit_ols`'s `OlsScratch` allocation; on the simulation path these live in
/// `SimWorkspace`), converts the row-major input to a column-major faer `Mat`,
/// runs the `family`-selected IRLS kernel cold-started at β=0, and maps the view
/// to `Fit`. No random effects ⇒ `tau2` empty. Binomial/Poisson keep
/// `dispersion = 1.0`; Gamma/NB recover their dispersion in their own arms.
/// Test-only baseline since the stable path dispatches through the unified core
/// ([`super::core::fit_on`]) over `fit_glm_prebuilt`/`glm_view_to_fit`.
#[cfg(test)]
pub(super) fn fit_glm(
    family: Family,
    nb_theta: f64,
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    opts: &FitOptions,
) -> Fit {
    let mut buf = GlmScratchBuf::new(n, p, opts.target_indices.len());
    let x_mat = to_col_major(x, n, p);
    let view = fit_glm_prebuilt(
        family,
        nb_theta,
        x_mat.as_ref().subrows(0, n),
        y,
        opts,
        &mut buf,
    );
    glm_view_to_fit(&view, y, family, nb_theta, n, p, opts)
}

/// [`fit_glm`]'s solve half over a prebuilt column-major `x_mat` and reusable
/// IRLS scratch, returning the borrowed [`GlmFitView`]. Hoisted so the NB
/// outer-θ loop ([`fit_glm_nb_capped`]) converts `x` and allocates the scratch
/// ONCE, re-running IRLS per fixed-θ fit; the unified fit core drives it per
/// draw. [`glm_view_to_fit`] maps the returned view to the full `Fit`.
pub(crate) fn fit_glm_prebuilt<'a>(
    family: Family,
    nb_theta: f64,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    opts: &FitOptions,
    buf: &'a mut GlmScratchBuf,
) -> GlmFitView<'a> {
    // None = β=0 cold start (no spec-derived truth on the standalone path).
    glm_irls_fit(
        family,
        nb_theta,
        x_mat,
        y,
        &opts.target_indices,
        None,
        opts.weights.as_deref(),
        opts.offset.as_deref(),
        buf.as_scratch(),
    )
}

/// Maps a [`GlmFitView`] to the full stable `Fit`: SE from `var_diag`, vcov from
/// the cached Cholesky factor, Gamma dispersion + √φ SE scaling, fitted means μ̂,
/// and the family log-likelihood. Needs raw `y` (Gamma dispersion + saturated
/// loglik read it) and `nb_theta` (the NB shape θ̂; only NB/Gamma read it, pass
/// `f64::NAN` otherwise).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glm_view_to_fit(
    view: &GlmFitView<'_>,
    y: &[f64],
    family: Family,
    nb_theta: f64,
    n: usize,
    p: usize,
    opts: &FitOptions,
) -> Fit {
    // --- map GlmFitView → Fit ---
    // view.betas is full [0..p]; view.var_diag is target-compact [0..t] (like OLS).
    let beta = view.betas.to_vec();
    let diag = view.diagnostics();
    let converged = diag.converged;
    // Weighted Σwᵢdᵢ at the accepted iterate (NaN unless converged).
    let irls_deviance = view.deviance;
    let mut se = vec![f64::NAN; p];
    fill_se_compact(view.var_diag, &opts.target_indices, &mut se);
    // Unscaled (X'WX)⁻¹; Gamma's φ multiplies it afterwards, exactly as √φ scales
    // `se`. `view.l` is stale-or-zero unless `converged` (its documented contract).
    let mut vcov = if converged {
        vcov_from_chol(view.l, p, &opts.target_indices, 1.0)
    } else {
        nan_vcov(p)
    };

    // Dispersion. Binomial/Poisson hold φ≡1 (the kernel's `(XᵀWX)⁻¹` is the full
    // covariance). Gamma and inverse-Gaussian recover φ post-fit — the mean
    // model β is φ-independent, so φ stays out of the IRLS — and scale the SE
    // by √φ (the kernel folded φ=1, so Var(β̂)=φ·(XᵀWX)⁻¹). `dispersion: Some(v)`
    // holds φ=v fixed; `None` estimates the Pearson moment `φ̂=Σ wᵢrᵢ²/(n−p)`,
    // `rᵢ=(yᵢ−μ̂ᵢ)/√V(μ̂ᵢ)`, raw-row df — exactly
    // `summary(glm(family=Gamma/inverse.gaussian, weights=w))$dispersion`.
    let dispersion = match family {
        Family::Gamma { .. } | Family::InverseGaussian { .. } if converged => {
            let phi = match opts.dispersion {
                Some(v) => v,
                None => crate::family::pearson_dispersion(
                    y,
                    view.mu,
                    family,
                    nb_theta,
                    n,
                    p,
                    opts.weights.as_deref(),
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
    // Gamma: R's `Gamma()$aic` convention — dispersion profiled as dev/Σwᵢ
    // inside `gamma_aic` (NOT the Pearson `dispersion` above, matching R, which
    // also mixes the two conventions between `logLik` and `summary`).
    let (fitted, loglik) = if converged {
        let mu = view.mu.to_vec();
        let ll = match family {
            Family::Gamma { .. } => {
                -0.5 * (crate::family::gamma_aic(y, &mu, irls_deviance, n, opts.weights.as_deref())
                    - 2.0)
            }
            Family::InverseGaussian { .. } => {
                -0.5 * (crate::family::inv_gaussian_aic(
                    y,
                    irls_deviance,
                    n,
                    opts.weights.as_deref(),
                ) - 2.0)
            }
            _ => {
                -0.5 * irls_deviance
                    + crate::family::saturated_loglik(family, nb_theta, y, opts.weights.as_deref())
            }
        };
        (mu, ll)
    } else {
        (vec![], f64::NAN)
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
        deviance: f64::NAN,
        loglik,
        df: if converged {
            super::common::model_df(family, p, 0, opts.dispersion.is_some())
        } else {
            0
        },
        reml: false,
        fitted,
        ranef: vec![],
        ranef_levels: vec![],
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
pub(super) const NB_THETA_LO: f64 = 1e-3;
pub(super) const NB_THETA_HI: f64 = 1e4;

/// NB profile log-likelihood in θ at fixed μ̂, up to the θ-independent `−ln(yᵢ!)`:
/// `Σ[ lnΓ(yᵢ+θ) − lnΓ(θ) + θ·ln(θ/(θ+μ̂ᵢ)) + yᵢ·ln(μ̂ᵢ/(θ+μ̂ᵢ)) ]`. Counts are
/// integers, so `lnΓ(y+θ)−lnΓ(θ) = Σ_{k=0}^{y−1} ln(θ+k)` exactly — no lgamma,
/// and identical to `MASS::theta.ml`'s objective. (`Σ_{k}` is `O(Σy)`; fine at
/// validation scale.)
///
/// The `μᵢ==0` guard makes this finite when evaluated at the **saturated** mean
/// `μ=y` (so `nb_profile_loglik(y, y, θ)` = the NB saturated log-likelihood, the
/// term the NB GLMM marginal-θ objective adds back to the Laplace deviance): at a
/// zero count `μ=y=0`, both mean terms vanish in the limit, but `0·ln(0)` would
/// otherwise produce a `NaN`. For the GLM caller `μ̂>0` always, so the guard never
/// fires there.
///
/// `weights: Some(w)` multiplies each row's contribution by the prior weight
/// `wᵢ` (matches `MASS::glm.nb(weights=)`'s profile — `theta.ml` weights the
/// per-row deviance terms the same way `glm.fit` weights IRLS); `None` is unit
/// weights.
pub(crate) fn nb_profile_loglik(y: &[f64], mu: &[f64], theta: f64, weights: Option<&[f64]>) -> f64 {
    let mut ll = 0.0;
    for (i, (&yi, &mi)) in y.iter().zip(mu.iter()).enumerate() {
        let mut s = 0.0;
        for k in 0..(yi.round() as u64) {
            s += (theta + k as f64).ln();
        }
        if mi > 0.0 {
            s += theta * (theta / (theta + mi)).ln() + yi * (mi / (theta + mi)).ln();
        }
        ll += weights.map_or(1.0, |w| w[i]) * s;
    }
    ll
}

/// Maximise `g(ln θ)` over `ln θ ∈ [ln NB_THETA_LO, ln NB_THETA_HI]` by
/// golden-section (the NB likelihood is far more symmetric in ln θ than in θ).
/// Returns `θ̂ = exp(argmax)`. Shared by the GLM conditional θ profile
/// ([`optimize_nb_theta`]) and the GLMM marginal-θ objective in [`fit_glmm_nb`];
/// `g` is the log-likelihood to maximise as a function of `ln θ`.
///
/// Stopping width `1e-4` on `ln θ` (2026-08-06, was `1e-8`): for the GLMM route,
/// `g` is a full inner GLMM refit at fixed θ, and that refit's own inner
/// two-stage BOBYQA is converged only to `GLMM_RHO_END = 3e-6` (`src/lmm/mod.rs`) —
/// no evaluation of `g` resolves θ differences finer than that on its own. Below
/// that radius the inner refit exhibits a knife-edge: two nearby θ values can
/// land the refit in different local optima of its own objective, with the same
/// `g(ln θ)` shape otherwise unchanged, so a search still iterating there is not
/// resolving curvature — it is picking a side of that knife-edge by whatever
/// rounding happens to be present in the inputs. Measured on the dense NB GLMM
/// fixture: the old `1e-8` width let a 1-ULP input perturbation flip the
/// reported β by ~9.5e-5 relative through exactly this mechanism (branch flip at
/// golden-section iteration 40 of 45, interval width 4.35e-8, ~69x tighter than
/// the noise floor); at `1e-4` the same perturbation pair converges to a
/// bit-identical β. `1e-4` was chosen as the loosest of {1e-8, 1e-7, 1e-6, 1e-5,
/// 1e-4, 1e-3} that both removes the flip and keeps every cross-engine NB golden
/// (`cargo test --features oracle-tests -- goldens_agree_with_the_references`)
/// inside `validation/tol.R`'s bands (`1e-3` also held; `1e-4` is one decade
/// tighter for margin).
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
        if (b - a).abs() < 1e-4 {
            break;
        }
    }
    (0.5 * (a + b)).exp()
}

/// Maximise [`nb_profile_loglik`] over θ at fixed μ̂, weighted by `weights`
/// (`FitOptions::weights`, forwarded from the caller). Returns θ̂.
fn optimize_nb_theta(y: &[f64], mu: &[f64], weights: Option<&[f64]>) -> f64 {
    golden_max_ln_theta(|t| nb_profile_loglik(y, mu, t.exp(), weights))
}

/// Negative-binomial GLM via the alternating outer-θ loop (`MASS::glm.nb`):
/// (1) fit the GLM at fixed θ; (2) 1-D maximise the NB profile log-likelihood
/// over θ holding β̂/μ̂; (3) repeat to convergence. `theta_seed = Some(v)` seeds
/// step 1; `None` cold-starts from a method-of-moments estimate
/// `θ₀ = ȳ²/max(s²−ȳ, ε)` (seed only, left unweighted — the outer loop's
/// weighted θ optimisation below washes out the seed's influence). The β SE
/// conditions on θ̂ (lme4/MASS convention; θ-uncertainty is out of scope), so
/// `dispersion = θ̂` and the SE comes straight from the final fixed-θ fit (NB
/// has `φ≡1`; overdispersion lives in `V=μ+μ²/θ`).
pub(super) fn fit_glm_nb(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    theta_seed: Option<f64>,
    opts: &FitOptions,
) -> Fit {
    fit_glm_nb_capped(x, y, n, p, theta_seed, opts, NB_MAX_OUTER)
}

/// [`fit_glm_nb`]'s loop body with the outer-iteration cap as a parameter —
/// a seam so the cap-exhaustion semantics are testable without engineering a
/// dataset that legitimately burns all `NB_MAX_OUTER` alternations. Production
/// code only ever calls it through [`fit_glm_nb`] (cap = `NB_MAX_OUTER`).
///
/// Cap-exhaustion semantics (pinned by `fit_glm_nb_outer_cap_semantics`): the
/// exit is SILENT — `converged` reflects only the last inner IRLS fit, β/se
/// stay at the stale pre-update θ, and `dispersion` carries the newer θ.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_glm_nb_capped(
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    theta_seed: Option<f64>,
    opts: &FitOptions,
    max_outer: usize,
) -> Fit {
    let mut theta = theta_seed.unwrap_or_else(|| {
        let ybar = y.iter().sum::<f64>() / n as f64;
        let var = y.iter().map(|&yi| (yi - ybar).powi(2)).sum::<f64>() / (n.max(2) - 1) as f64;
        (ybar * ybar / (var - ybar).max(1e-6)).clamp(NB_THETA_LO, NB_THETA_HI)
    });

    let family = Family::NegativeBinomial {
        link: NegBinomialLink::Log,
    };
    // θ-invariant state, built once and reused across the θ↔β alternation: the
    // column-major design, the IRLS scratch, and the μ̂ buffer.
    let x_mat = to_col_major(x, n, p);
    let x_ref = x_mat.as_ref().subrows(0, n);
    let mut buf = GlmScratchBuf::new(n, p, opts.target_indices.len());
    let mut mu = vec![0.0f64; n];

    let mut fit_result = fit_unsupported_family(p);
    for _ in 0..max_outer {
        // θ is fixed for this β fit and threaded explicitly (the spec is θ-free).
        let view = fit_glm_prebuilt(family, theta, x_ref, y, opts, &mut buf);
        fit_result = glm_view_to_fit(&view, y, family, theta, n, p, opts);
        if !fit_result.converged() {
            break;
        }
        // μ̂ = exp(o + Xβ̂) for the θ optimisation.
        for (i, mi) in mu.iter_mut().enumerate() {
            let mut eta: f64 = (0..p).map(|j| x[i * p + j] * fit_result.beta[j]).sum();
            if let Some(o) = &opts.offset {
                eta += o[i];
            }
            *mi = crate::family::link_inv(family, eta);
        }
        let new_theta = optimize_nb_theta(y, &mu, opts.weights.as_deref());
        let converged = (new_theta - theta).abs() / theta < NB_THETA_TOL;
        theta = new_theta;
        if converged {
            // Final β/SE at the converged θ for consistency.
            let view = fit_glm_prebuilt(family, theta, x_ref, y, opts, &mut buf);
            fit_result = glm_view_to_fit(&view, y, family, theta, n, p, opts);
            break;
        }
    }
    fit_result.dispersion = theta;
    fit_result
}
