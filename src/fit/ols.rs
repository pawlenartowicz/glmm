//! OLS (`Family::Gaussian`, `re: None`) dispatch — estimator dispatch, the
//! numerical kernel lives in `src/ols.rs`. Owns the `OlsScratch`/`OlsSuffStats`
//! allocation, applies √wᵢ row weighting, and maps `OlsFitView` to `Fit`.

use faer::Mat;

use crate::ols::{OlsFitView, OlsScratch, OlsSuffStats, PANEL_ROWS};

use super::common::{fill_se_compact, nan_vcov, vcov_from_chol, FitDiagnostics};
use super::{Fit, FitOptions};

impl OlsFitView<'_> {
    /// This route's [`FitDiagnostics`]. No θ, so no boundary or pin state; the
    /// recorded pivot is measured on the **weighted** Gram, which is the whole
    /// reason it is worth reporting — the pre-dispatch alias gate tests the raw
    /// `x` and a design full-rank unweighted can be near-singular once weighted.
    pub(crate) fn diagnostics(&self) -> FitDiagnostics {
        FitDiagnostics {
            pivot: self.pivot,
            pivot_col: self.pivot_col,
            ill_conditioned: self.pivot < crate::ols::PIVOT_MIN,
            ..FitDiagnostics::fixed_only(self.converged)
        }
    }
}

// ---------------------------------------------------------------------------
// OLS dispatch
// ---------------------------------------------------------------------------

/// Owned OLS scratch for [`fit_ols`], sized off n/p/t. Hoistable across repeated
/// fixed-shape fits (the loop tier): [`fit_ols_prebuilt`] re-seeds every
/// populated slot at entry, so a reused buffer is near-identical to a fresh
/// allocation (mirrors [`super::glm`]'s `GlmScratchBuf` contract). Unlike the
/// OLS *kernel*, which takes no weights, the √wᵢ row scaling + identity offset
/// shift happen here — the scaled copies live in `scaled_x`/`scaled_y` (sized to
/// `n_max`) so a weighted loop refit reuses them instead of reallocating.
pub(crate) struct OlsWorkspace {
    fit_betas: Vec<f64>,
    fit_var_diag: Vec<f64>,
    fit_t_sq: Vec<f64>,
    fit_u_scratch: Vec<f64>,
    fit_factor: Mat<f64>,
    fit_rhs: Mat<f64>,
    suff_xtx: Mat<f64>,
    suff_xty: Vec<f64>,
    suff_xtx_work: Mat<f64>,
    panel_x: Vec<f64>,
    panel_y: Vec<f64>,
    // √wᵢ-scaled design / offset-shifted-and-scaled response, n_max-sized.
    // `scaled_x` is read only on the weighted branch (`:91-99` below); gated
    // 0×0 when the build is unweighted (`has_weights` is frozen at build, so
    // the workspace knows at construction whether the buffer can ever be
    // read). 0×0 rather than `.max(1)`, deliberately: a read on a route that
    // is supposed to have none must panic on the bounds check, not silently
    // return a zero.
    scaled_x: Mat<f64>,
    scaled_y: Vec<f64>,
}

impl OlsWorkspace {
    pub(crate) fn new(n_max: usize, p: usize, t: usize, has_weights: bool) -> Self {
        let n1 = n_max.max(1);
        let p1 = p.max(1); // guard zero-column degenerate call
        let t1 = t.max(1);
        Self {
            fit_betas: vec![0.0f64; p1],
            fit_var_diag: vec![0.0f64; t1],
            fit_t_sq: vec![0.0f64; t1],
            fit_u_scratch: vec![0.0f64; p1],
            fit_factor: Mat::<f64>::zeros(p1, p1),
            fit_rhs: Mat::<f64>::zeros(p1, 1),
            suff_xtx: Mat::<f64>::zeros(p1, p1),
            suff_xty: vec![0.0f64; p1],
            suff_xtx_work: Mat::<f64>::zeros(p1, p1),
            // panel buffers: PANEL_ROWS * p1 is always sufficient (see PANEL_ROWS).
            panel_x: vec![0.0f64; PANEL_ROWS * p1],
            panel_y: vec![0.0f64; PANEL_ROWS],
            scaled_x: if has_weights {
                Mat::<f64>::zeros(n1, p1)
            } else {
                Mat::<f64>::zeros(0, 0)
            },
            scaled_y: vec![0.0f64; n1],
        }
    }
}

/// [`fit_ols`]'s accumulate + solve half over a prebuilt column-major (raw,
/// unscaled) `x_mat` and reusable scratch. Applies WLS by √wᵢ row scaling and
/// the identity-link offset y-shift internally (the OLS kernel takes neither),
/// then accumulates X'WX / X'Wy / y'Wy and runs the sufficient-statistics solve.
/// `sum_y`/`sst` are NOT weight-consistent under this scaling but are never
/// mapped into `Fit` (see [`ols_view_to_fit`]).
pub(crate) fn fit_ols_prebuilt<'a>(
    ws: &'a mut OlsWorkspace,
    x_mat: faer::MatRef<'_, f64>,
    y: &[f64],
    n: usize,
    p: usize,
    opts: &FitOptions,
) -> OlsFitView<'a> {
    // Reset accumulators — a reused ws must not carry the prior fit's sums.
    ws.suff_xtx.fill(0.0);
    ws.suff_xty.iter_mut().for_each(|z| *z = 0.0);
    let mut suff_yty = 0.0f64;
    let mut suff_sum_y = 0.0f64;
    let mut suff_n_rows = 0usize;

    // Offset applied to y BEFORE √wᵢ scaling so weighting composes; x is scaled
    // only when weighted. Effective x/y point at ws.scaled_* (weighted/offset) or
    // straight at the caller's buffers (unweighted, no offset → bit-identical to
    // passing x_mat/y directly).
    let weighted = opts.weights.is_some();
    let offset = opts.offset.is_some();
    let x_eff = if weighted {
        let w = opts.weights.as_ref().unwrap();
        for i in 0..n {
            let s = w[i].sqrt();
            for j in 0..p {
                ws.scaled_x[(i, j)] = s * x_mat[(i, j)];
            }
        }
        ws.scaled_x.as_ref().subrows(0, n)
    } else {
        x_mat
    };
    let y_eff: &[f64] = if weighted || offset {
        for i in 0..n {
            let yi = y[i] - opts.offset.as_ref().map_or(0.0, |o| o[i]);
            ws.scaled_y[i] = if weighted {
                yi * opts.weights.as_ref().unwrap()[i].sqrt()
            } else {
                yi
            };
        }
        &ws.scaled_y[..n]
    } else {
        y
    };

    {
        let mut suff = OlsSuffStats {
            xtx: ws.suff_xtx.as_mut(),
            xty: &mut ws.suff_xty,
            yty: &mut suff_yty,
            sum_y: &mut suff_sum_y,
            n_rows: &mut suff_n_rows,
            panel_x: &mut ws.panel_x,
            panel_y: &mut ws.panel_y,
        };
        if n > 0 && p > 0 {
            suff.add_rows(x_eff, y_eff);
        }
    }

    let scratch = OlsScratch {
        fit_betas: &mut ws.fit_betas,
        fit_var_diag: &mut ws.fit_var_diag,
        fit_t_sq: &mut ws.fit_t_sq,
        fit_u_scratch: &mut ws.fit_u_scratch,
        fit_factor: ws.fit_factor.as_mut(),
        fit_rhs: ws.fit_rhs.as_mut(),
    };
    crate::ols::fit_suff_stats_t_sq(
        ws.suff_xtx.as_ref(),
        &ws.suff_xty,
        suff_yty,
        suff_sum_y,
        suff_n_rows,
        &opts.target_indices,
        ws.suff_xtx_work.as_mut(),
        scratch,
    )
}

/// OLS dispatch adapter. Builds a throwaway [`OlsWorkspace`], converts the
/// row-major input to a column-major faer `Mat`, runs the sufficient-statistics
/// solve, and maps the view to `Fit`. Test-only baseline since the stable path
/// dispatches through the unified core ([`super::core::fit_on`]) over
/// `fit_ols_prebuilt`/`ols_view_to_fit`.
#[cfg(test)]
pub(super) fn fit_ols(x: &[f64], y: &[f64], n: usize, p: usize, opts: &FitOptions) -> Fit {
    let mut ws = OlsWorkspace::new(n, p, opts.target_indices.len(), opts.weights.is_some());
    let x_mat = super::common::to_col_major(x, n, p);
    let view = fit_ols_prebuilt(&mut ws, x_mat.as_ref().subrows(0, n), y, n, p, opts);
    ols_view_to_fit(&view, x, y, n, p, opts)
}

/// Maps an [`OlsFitView`] to the full stable `Fit`. Needs the raw `x`/`y`/`n`
/// because `fitted = o + Xβ̂` (raw rows) and the weighted ML Gaussian loglik are
/// recomputed from the original-scale data, not the √wᵢ-scaled accumulators.
pub(crate) fn ols_view_to_fit(
    view: &OlsFitView<'_>,
    x: &[f64],
    y: &[f64],
    n: usize,
    p: usize,
    opts: &FitOptions,
) -> Fit {
    // view.betas is compact [0..p]; view.var_diag is compact [0..t] at target rank
    // (OLS/GLM are target-compact; LME/LMM are predictor-indexed)
    let beta = view.betas.to_vec();
    let diag = view.diagnostics();
    let converged = diag.converged;
    let mut se = vec![f64::NAN; p];
    fill_se_compact(view.var_diag, &opts.target_indices, &mut se);
    // `view.factor` is stale-or-zero unless `converged` (its documented
    // contract) — a non-converged fit reports an all-NaN vcov, as `se` does.
    let vcov = if converged {
        vcov_from_chol(view.factor, p, &opts.target_indices, view.sigma_sq)
    } else {
        nan_vcov(p)
    };

    // Fitted means Xβ̂ (raw rows — the √wᵢ scaling above is a solver device,
    // not part of the mean) and the ML Gaussian log-likelihood off the weighted
    // RSS, R `logLik.lm`: ½(Σlog wᵢ − n(ln 2π + 1 − ln n + ln Σwᵢrᵢ²)).
    let (fitted, loglik) = if converged && n > 0 {
        let fitted: Vec<f64> = (0..n)
            .map(|i| {
                let o = opts.offset.as_ref().map_or(0.0, |o| o[i]);
                o + (0..p).map(|j| x[i * p + j] * beta[j]).sum::<f64>()
            })
            .collect();
        let rss: f64 = (0..n)
            .map(|i| {
                let r = y[i] - fitted[i];
                opts.weights.as_ref().map_or(1.0, |w| w[i]) * r * r
            })
            .sum();
        let sum_log_w = opts
            .weights
            .as_ref()
            .map_or(0.0, |w| w.iter().map(|v| v.ln()).sum());
        let nf = n as f64;
        let ll =
            0.5 * (sum_log_w - nf * ((2.0 * std::f64::consts::PI).ln() + 1.0 - nf.ln() + rss.ln()));
        (fitted, ll)
    } else {
        (vec![], f64::NAN)
    };

    Fit {
        beta,
        se,
        vcov,
        tau2: vec![],
        // σ̂² = RSS/(n−p); NaN when not converged (nonconverged_view's fill).
        dispersion: view.sigma_sq,
        // `singular` comes out of `boundary_hit == 1`, and this route has no θ
        // to pin: the carrier documents `boundary_hit` as meaningless here and
        // fixes it at 0, so `singular` is constant false on OLS. That is the
        // right report — "no variance component collapsed" is trivially true
        // where there are none — but it means nothing on this route can ever
        // make it true, and a test comparing against it is comparing constants.
        // Mirrors `glm.rs` — change together.
        diagnostics: super::common::materialize_diagnostics(&diag, p, &[]),
        varcorr: vec![],
        stddev_se: vec![],
        n_eval: 0,
        deviance: f64::NAN,
        loglik,
        // p fixed effects + σ² (R logLik.lm's df).
        df: if converged { p + 1 } else { 0 },
        reml: false,
        fitted,
        ranef: vec![],
        ranef_levels: vec![],
    }
}
