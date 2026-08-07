//! UNSTABLE scratch-explicit hot-path surface (the loop tier) for warm-start
//! consumers like MCPower. Gated by the `loop_advanced` cargo feature (off by
//! default). NO semver guarantees — may change in ANY release. The warm-start
//! primitive [`crate::StartValues`] is re-exported `pub` only behind this feature.

// The unified fit core — the build-once/fit-many surface for the loop tier, and
// the same dispatch body `fit_cold`/`fit_warm` run, so a model or routing change
// lands in every caller identically. Prefer it over the individual kernels below,
// which do not classify the design for you.
pub use crate::fit::{
    build_lmm_seam_ws, build_workspace, fit_on, lmm_objective_at, lmm_sweep_fit, lmm_sweep_fit_on,
    FitDiagnostics, FitView, FitWorkspace, LmmSeamWs, LmmSweepOutcome,
};
// The RE-level-count normalizer `build_workspace` expects its `sized` spec to have
// gone through — exposed so loop-tier consumers size specs the validated way
// instead of reimplementing the crossed/nested count derivation.
pub use crate::fit::spec_sized_from_ids_pub;
pub use crate::glm::{glm_irls_fit, sigmoid_stable, GlmFitView, GlmScratch, MAX_IRLS_ITERS};
pub use crate::glmm::GlmmFit;
pub use crate::lme::{lme_fit, LmeFitView, LmeScratch, LmeSuffStats};
pub use crate::lmm::{primary_lambda, LmmFit, LmmGroupings, LmmSuffStats};
pub use crate::ols::{
    fit_suff_stats_t_sq, ols_contrast_t_sq, OlsFitView, OlsScratch, OlsSuffStats, PANEL_ROWS,
};

#[cfg(test)]
mod tests {
    use crate::lmm::{fit_lmm, LmmWorkspace};
    use crate::start::StartValues;
    use crate::{Family, ModelSpec, ReStructure, Sizing};
    use faer::Mat;

    /// Deterministic pseudo-data (NR LCG), uniform in (−1, 1) — mirrors lmm's tests.
    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((*state >> 11) as f64) / ((1u64 << 53) as f64)) * 2.0 - 1.0
    }

    /// n=48, p=3, 6 clusters; intercept-only Gaussian LMM (mirror of lmm's
    /// `hand_dataset`): `y = 0.5 + 0.4·x1 − 0.2·x2 + u_c + 0.8·e`.
    fn hand_dataset() -> (Mat<f64>, Vec<f64>, Vec<u32>) {
        let n = 48usize;
        let n_clusters = 6usize;
        let mut st = 42u64;
        let u_c: Vec<f64> = (0..n_clusters).map(|_| 0.6 * lcg(&mut st)).collect();
        let mut x = Mat::<f64>::zeros(n, 3);
        let mut y = vec![0.0f64; n];
        let mut ids = vec![0u32; n];
        for i in 0..n {
            let c = i % n_clusters;
            ids[i] = c as u32;
            let x1 = lcg(&mut st);
            let x2 = lcg(&mut st);
            x[(i, 0)] = 1.0;
            x[(i, 1)] = x1;
            x[(i, 2)] = x2;
            y[i] = 0.5 + 0.4 * x1 - 0.2 * x2 + u_c[c] + 0.8 * lcg(&mut st);
        }
        (x, y, ids)
    }

    fn intercept_lmm_spec() -> ModelSpec {
        ModelSpec {
            family: Family::Gaussian,
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 6 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        }
    }

    /// `StartValues.theta` threads into the loop kernel `fit_lmm`, and the LMM MLE is
    /// start-independent: a cold fit (`None` → the kernel's THETA0 blind start) and a
    /// warm fit from a perturbed `StartValues.theta` reach the same β̂ up to optimizer
    /// tolerance. Proves the loop-tier warm-start path — exposure
    /// plus thread-through, since the kernel already honors `theta_start`.
    #[test]
    fn start_values_theta_threads_into_loop_kernel() {
        let (x, y, pid) = hand_dataset();
        let (n, p) = (x.nrows(), 3);
        let model = intercept_lmm_spec();

        let mut ws_cold = LmmWorkspace::for_cluster_spec(p, &model, n, &[]);
        ws_cold.suff.reset();
        ws_cold.suff.add_rows_multi(x.as_ref(), &y, &pid, &[], None);
        let cold = fit_lmm(&mut ws_cold, &[1, 2], None);

        // n_theta == 1 here (one intercept variance component); warm-start it well off
        // THETA0 (1.0) to make the thread-through meaningful.
        let warm_start = StartValues {
            beta: vec![0.0; p],
            theta: vec![5.0],
        };
        let mut ws_warm = LmmWorkspace::for_cluster_spec(p, &model, n, &[]);
        ws_warm.suff.reset();
        ws_warm.suff.add_rows_multi(x.as_ref(), &y, &pid, &[], None);
        let warm = fit_lmm(&mut ws_warm, &[1, 2], Some(&warm_start.theta));

        assert!(
            cold.converged && warm.converged,
            "both starts must converge"
        );
        for j in [1usize, 2] {
            let (a, b) = (ws_cold.fit.betas[j], ws_warm.fit.betas[j]);
            let d = (a - b).abs();
            assert!(
                d <= 1e-7 || d <= 1e-6 * a.abs().max(b.abs()),
                "loop MLE must be start-independent: β[{j}] cold {a} vs warm {b}"
            );
        }
    }
}
