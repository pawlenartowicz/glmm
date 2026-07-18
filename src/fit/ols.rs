//! OLS (`Family::Gaussian`, `re: None`) dispatch — estimator dispatch, the
//! numerical kernel lives in `src/ols.rs`. Owns the `OlsScratch`/`OlsSuffStats`
//! allocation, applies √wᵢ row weighting, and maps `OlsFitView` to `Fit`.

use faer::Mat;

use crate::ols::{OlsScratch, OlsSuffStats, PANEL_ROWS};

use super::common::{fill_se_compact, nan_vcov, vcov_from_chol};
use super::{Fit, FitOptions};

// ---------------------------------------------------------------------------
// OLS dispatch
// ---------------------------------------------------------------------------

pub(super) fn fit_ols(x: &[f64], y: &[f64], n: usize, p: usize, opts: &FitOptions) -> Fit {
    let t = opts.target_indices.len();

    // --- scratch allocation (mirrors SimWorkspace field sizes in workspace.rs) ---
    let p1 = p.max(1); // guard zero-column degenerate call
    let mut fit_betas = vec![0.0f64; p1];
    let mut fit_var_diag = vec![0.0f64; t.max(1)];
    let mut fit_t_sq = vec![0.0f64; t.max(1)];
    let mut fit_u_scratch = vec![0.0f64; p1];
    let mut fit_factor = Mat::<f64>::zeros(p1, p1);
    let mut fit_rhs = Mat::<f64>::zeros(p1, 1);
    let mut suff_xtx = Mat::<f64>::zeros(p1, p1);
    let mut suff_xty = vec![0.0f64; p1];
    let mut suff_yty = 0.0f64;
    let mut suff_sum_y = 0.0f64;
    let mut suff_n_rows = 0usize;
    let mut suff_xtx_work = Mat::<f64>::zeros(p1, p1);
    // panel buffers: PANEL_ROWS * p1 is always sufficient (see PANEL_ROWS comment)
    let mut panel_x = vec![0.0f64; PANEL_ROWS * p1];
    let mut panel_y = vec![0.0f64; PANEL_ROWS];

    // --- convert row-major f64 input to column-major f64 faer matrix ---
    // WLS by √wᵢ row scaling: the unweighted accumulator then yields
    // X'WX / X'Wy / y'Wy, so RSS = y'Wy − β̂'X'Wy is the weighted RSS and
    // σ̂² = RSS/(n−p) matches R lm(weights=) (raw-row df). `sum_y`/`sst`
    // are NOT weight-consistent under this scaling but are never mapped
    // into `Fit` — leave them unwired.
    let sqrt_w: Option<Vec<f64>> = opts
        .weights
        .as_ref()
        .map(|w| w.iter().map(|v| v.sqrt()).collect());
    let mut x_mat = Mat::<f64>::zeros(n.max(1), p1);
    for i in 0..n {
        let s = sqrt_w.as_ref().map_or(1.0, |sw| sw[i]);
        for j in 0..p {
            x_mat[(i, j)] = s * x[i * p + j];
        }
    }
    // Identity-link offset is an exact y-shift: fit on y − o, report means as
    // o + Xβ̂ below. Applied BEFORE the √wᵢ scaling so weighting composes.
    let y_shifted: Vec<f64>;
    let y_base: &[f64] = match &opts.offset {
        Some(o) => {
            y_shifted = y.iter().zip(o).map(|(&yi, &oi)| yi - oi).collect();
            &y_shifted
        }
        None => y,
    };
    let y_scaled: Vec<f64>;
    let y_eff: &[f64] = match &sqrt_w {
        Some(sw) => {
            y_scaled = y_base.iter().zip(sw).map(|(&yi, &s)| yi * s).collect();
            &y_scaled
        }
        None => y_base,
    };

    {
        let mut suff = OlsSuffStats {
            xtx: suff_xtx.as_mut(),
            xty: &mut suff_xty,
            yty: &mut suff_yty,
            sum_y: &mut suff_sum_y,
            n_rows: &mut suff_n_rows,
            panel_x: &mut panel_x,
            panel_y: &mut panel_y,
        };
        if n > 0 && p > 0 {
            suff.add_rows(x_mat.as_ref().subrows(0, n), y_eff);
        }
    }

    let view = {
        let scratch = OlsScratch {
            fit_betas: &mut fit_betas,
            fit_var_diag: &mut fit_var_diag,
            fit_t_sq: &mut fit_t_sq,
            fit_u_scratch: &mut fit_u_scratch,
            fit_factor: fit_factor.as_mut(),
            fit_rhs: fit_rhs.as_mut(),
        };
        crate::ols::fit_suff_stats_t_sq(
            suff_xtx.as_ref(),
            &suff_xty,
            suff_yty,
            suff_sum_y,
            suff_n_rows,
            &opts.target_indices,
            1e-12,
            suff_xtx_work.as_mut(),
            scratch,
        )
    };

    // --- map OlsFitView → Fit ---
    // view.betas is compact [0..p]; view.var_diag is compact [0..t] at target rank
    // (OLS/GLM are target-compact; LME/LMM are predictor-indexed)
    let beta = view.betas.to_vec();
    let converged = view.converged;
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
        converged,
        varcorr: vec![],
        stddev_se: vec![],
        aliased: vec![false; p],
        n_eval: 0,
        deviance: f64::NAN,
        singular: false,
        loglik,
        // p fixed effects + σ² (R logLik.lm's df).
        df: if converged { p + 1 } else { 0 },
        reml: false,
        fitted,
        ranef: vec![],
        ranef_levels: vec![],
    }
}
