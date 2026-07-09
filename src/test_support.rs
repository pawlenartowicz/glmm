//! Test-only scaffolding shared across the fit-module `#[cfg(test)]` blocks.
//!
//! `TestWs` is a buffer-bag mirroring the field names/reset semantics of the
//! kernel's own workspace, for the subset the fit-module tests borrow, so each test's
//! field-borrowing helper body (`suff_stats`, `glm_scratch`,
//! `build_lme_scratch`, `shipped_workspace`) ports with minimal change. It carries no
//! design-gen / RNG / critval machinery — the tests only use it as pre-sized
//! scratch. The alloc lines + the two reset methods mirror the kernel workspace's
//! `new` / its resets; the cluster count is passed directly.

use crate::ols::PANEL_ROWS;
use faer::Mat;

pub(crate) struct TestWs {
    // OLS suff-stats
    pub suff_xtx: Mat<f64>,
    pub suff_xty: Vec<f64>,
    pub suff_yty: f64,
    pub suff_sum_y: f64,
    pub suff_n_rows: usize,
    pub suff_xtx_work: Mat<f64>,
    pub panel_x: Vec<f64>,
    pub panel_y: Vec<f64>,

    // OLS fit scratch
    pub fit_betas: Vec<f64>,
    pub fit_var_diag: Vec<f64>,
    pub fit_t_sq: Vec<f64>,
    pub fit_u_scratch: Vec<f64>,
    pub fit_factor: Mat<f64>,
    pub fit_rhs: Mat<f64>,

    // IRLS scratch
    pub irls_eta: Vec<f64>,
    pub irls_p: Vec<f64>,
    pub irls_w: Vec<f64>,
    pub irls_z: Vec<f64>,
    pub irls_betas: Vec<f64>,
    pub irls_betas_new: Vec<f64>,
    pub irls_var_diag: Vec<f64>,
    pub irls_t_sq: Vec<f64>,
    pub irls_u_scratch: Vec<f64>,
    pub irls_xtwx: Mat<f64>,
    pub irls_xtwz: Vec<f64>,
    pub irls_l: Mat<f64>,
    pub irls_wx: Vec<f64>,

    // LME scratch
    pub lme_xtx: Mat<f64>,
    pub lme_xty: Vec<f64>,
    pub lme_yty: f64,
    pub lme_sum_xc: Mat<f64>,
    pub lme_sum_yc: Vec<f64>,
    pub lme_cluster_sizes: Vec<u32>,
    pub lme_n_clusters_seen: u32,
    pub lme_xtvix: Mat<f64>,
    pub lme_xtviy: Vec<f64>,
    pub lme_xtvix_factor: Mat<f64>,
    pub lme_v_diag_inv: Vec<f64>,
    pub lme_betas: Vec<f64>,
    pub lme_var_diag: Vec<f64>,
    pub lme_t_sq: Vec<f64>,
    pub lme_u_scratch: Vec<f64>,
    pub lme_brent_log_a: f64,
    pub lme_brent_log_b: f64,
    pub lme_brent_log_c: f64,
    pub lme_brent_fa: f64,
    pub lme_brent_fb: f64,
    pub lme_brent_fc: f64,
    pub lme_joint_sigma_t_chol: Mat<f64>,
    pub lme_joint_rhs: Vec<f64>,
    pub lme_joint_k_inv: Mat<f64>,
}

impl TestWs {
    /// Allocate scratch sized for `max_n` rows, `n_predictors` columns, and
    /// `max_n_clusters` clusters. Alloc lines copied verbatim from
    /// `SimWorkspace::new`, with the cluster count passed directly (instead of
    /// being derived from a `ClusterSpec`).
    pub(crate) fn new(max_n: usize, n_predictors: usize, max_n_clusters: usize) -> Self {
        Self {
            fit_betas: vec![0.0; n_predictors],
            fit_var_diag: vec![0.0; n_predictors],
            fit_t_sq: vec![0.0; n_predictors],
            fit_u_scratch: vec![0.0; n_predictors],
            fit_factor: Mat::<f64>::zeros(n_predictors, n_predictors),
            fit_rhs: Mat::<f64>::zeros(max_n.max(n_predictors), 1),

            suff_xtx: Mat::<f64>::zeros(n_predictors, n_predictors),
            suff_xty: vec![0.0; n_predictors],
            suff_yty: 0.0,
            suff_sum_y: 0.0,
            suff_n_rows: 0,
            suff_xtx_work: Mat::<f64>::zeros(n_predictors, n_predictors),
            panel_x: vec![0.0f64; PANEL_ROWS * n_predictors],
            panel_y: vec![0.0f64; PANEL_ROWS],

            irls_eta: vec![0.0; max_n],
            irls_p: vec![0.0; max_n],
            irls_w: vec![0.0; max_n],
            irls_z: vec![0.0; max_n],
            irls_betas: vec![0.0; n_predictors],
            irls_betas_new: vec![0.0; n_predictors],
            irls_var_diag: vec![0.0; n_predictors],
            irls_t_sq: vec![0.0; n_predictors],
            irls_u_scratch: vec![0.0; n_predictors],
            irls_xtwx: Mat::<f64>::zeros(n_predictors, n_predictors),
            irls_xtwz: vec![0.0; n_predictors],
            irls_l: Mat::<f64>::zeros(n_predictors, n_predictors),
            irls_wx: vec![0.0; max_n * n_predictors],

            lme_xtx: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_xty: vec![0.0; n_predictors],
            lme_yty: 0.0,
            lme_sum_xc: Mat::<f64>::zeros(n_predictors, max_n_clusters.max(1)),
            lme_sum_yc: vec![0.0; max_n_clusters.max(1)],
            lme_cluster_sizes: vec![0u32; max_n_clusters.max(1)],
            lme_n_clusters_seen: 0,
            lme_xtvix: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_xtviy: vec![0.0; n_predictors],
            lme_xtvix_factor: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_v_diag_inv: vec![0.0; max_n_clusters.max(1)],
            lme_betas: vec![0.0; n_predictors],
            lme_var_diag: vec![0.0; n_predictors],
            lme_t_sq: vec![0.0; n_predictors],
            lme_u_scratch: vec![0.0; n_predictors],
            lme_brent_log_a: 0.0,
            lme_brent_log_b: 0.0,
            lme_brent_log_c: 0.0,
            lme_brent_fa: 0.0,
            lme_brent_fb: 0.0,
            lme_brent_fc: 0.0,
            lme_joint_sigma_t_chol: Mat::<f64>::zeros(n_predictors, n_predictors),
            lme_joint_rhs: vec![0.0; n_predictors],
            lme_joint_k_inv: {
                let mut m = Mat::<f64>::zeros(n_predictors, n_predictors);
                for i in 0..n_predictors {
                    m[(i, i)] = 1.0;
                }
                m
            },
        }
    }

    /// Reset the OLS sufficient-statistics accumulator to "no rows seen".
    /// Reuses storage — zeros only the lower triangle of `suff_xtx` (the upper
    /// triangle is never read or written by `add_rows_suff` / Cholesky).
    pub(crate) fn reset_suff_stats(&mut self) {
        let p = self.suff_xty.len();
        for j in 0..p {
            for i in j..p {
                self.suff_xtx[(i, j)] = 0.0;
            }
            self.suff_xty[j] = 0.0;
        }
        self.suff_yty = 0.0;
        self.suff_sum_y = 0.0;
        self.suff_n_rows = 0;
    }

    /// Reset the LME sufficient-statistics accumulator to "no rows seen".
    pub(crate) fn reset_lme_suff_stats(&mut self) {
        let p = self.lme_xty.len();
        let k = self.lme_sum_yc.len();
        for j in 0..p {
            for i in j..p {
                self.lme_xtx[(i, j)] = 0.0;
            }
            self.lme_xty[j] = 0.0;
            for c in 0..k {
                self.lme_sum_xc[(j, c)] = 0.0;
            }
        }
        for v in self.lme_sum_yc.iter_mut() {
            *v = 0.0;
        }
        for v in self.lme_cluster_sizes.iter_mut() {
            *v = 0;
        }
        self.lme_yty = 0.0;
        self.lme_n_clusters_seen = 0;

        for j in 0..p {
            for i in 0..p {
                self.lme_joint_k_inv[(i, j)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
}

/// Intercept-only `ModelSpec` — mirrors `engine_contract::ClusterSpec::intercept_only`.
pub(crate) fn intercept_only_spec(sizing: crate::Sizing, _tau: f64) -> crate::ModelSpec {
    crate::ModelSpec {
        family: crate::Family::Gaussian,
        re: Some(crate::ReStructure {
            sizing,
            slopes: vec![],
            extra_groupings: vec![],
        }),
    }
}

/// Levels one grouping contributes to an atom block. Verbatim mirror of
/// `engine_contract::GroupingRelation::block_levels` — used only by the
/// test-only DGP layout helpers below (glmm's `ModelSpec` deliberately omits
/// the sim-layer layout methods; the lmm test datasets need them).
pub(crate) fn block_levels(rel: &crate::GroupingRelation) -> usize {
    match rel {
        crate::GroupingRelation::Crossed { n_clusters } => (*n_clusters).max(1) as usize,
        crate::GroupingRelation::NestedWithin { n_per_parent } => (*n_per_parent).max(1) as usize,
    }
}

/// Grid atom for the full grouping structure. Verbatim mirror of
/// `engine_contract::ClusterSpec::atom` (test-only DGP layout).
pub(crate) fn model_atom(spec: &crate::ModelSpec) -> usize {
    let re = spec.re.as_ref().expect("model_atom requires re: Some");
    re.extra_groupings
        .iter()
        .fold(re.sizing.atom(), |a, g| a * block_levels(&g.relation))
}

/// Level of extra grouping `g` that row `i` belongs to. Verbatim mirror of
/// `engine_contract::ClusterSpec::extra_level_of_row` (test-only DGP layout).
pub(crate) fn extra_level_of_row(spec: &crate::ModelSpec, g: usize, i: usize) -> usize {
    let re = spec
        .re
        .as_ref()
        .expect("extra_level_of_row requires re: Some");
    let rel = &re.extra_groupings[g].relation;
    match &re.sizing {
        crate::Sizing::FixedClusters { n_clusters } => {
            let s = (*n_clusters).max(1) as usize;
            let mut stride = s;
            for h in &re.extra_groupings[..g] {
                stride *= block_levels(&h.relation);
            }
            let within = (i / stride) % block_levels(rel);
            match rel {
                crate::GroupingRelation::Crossed { .. } => within,
                crate::GroupingRelation::NestedWithin { n_per_parent } => {
                    (i % s) * (*n_per_parent).max(1) as usize + within
                }
            }
        }
        crate::Sizing::FixedSize { cluster_size } => {
            let cs = (*cluster_size).max(1) as usize;
            let np = block_levels(rel);
            (i / cs) * np + (i % cs) % np
        }
    }
}

/// Build an `LmeScratch` from a `TestWs` whose `lme_*` suff-stats are already
/// populated. Shared by `lme.rs` and `lmm.rs` tests; keep in sync with the
/// `lme::LmeScratch` field list.
pub(crate) fn build_lme_scratch<'w>(
    ws: &'w mut TestWs,
    n_rows: u32,
    n_clusters: u32,
) -> crate::lme::LmeScratch<'w> {
    use faer::reborrow::IntoConst;
    crate::lme::LmeScratch {
        xtx: ws.lme_xtx.as_ref(),
        xty: &ws.lme_xty,
        yty: ws.lme_yty,
        ols_scratch: crate::ols::OlsScratch {
            fit_betas: &mut ws.fit_betas,
            fit_var_diag: &mut ws.fit_var_diag,
            fit_t_sq: &mut ws.fit_t_sq,
            fit_u_scratch: &mut ws.fit_u_scratch,
            fit_factor: ws.fit_factor.as_mut(),
            fit_rhs: ws.fit_rhs.as_mut(),
        },
        sum_xc: ws.lme_sum_xc.as_mut().into_const(),
        sum_yc: &ws.lme_sum_yc,
        cluster_sizes: &ws.lme_cluster_sizes,
        n_clusters,
        n_rows,
        xtvix: ws.lme_xtvix.as_mut(),
        xtviy: &mut ws.lme_xtviy,
        xtvix_factor: ws.lme_xtvix_factor.as_mut(),
        v_diag_inv: &mut ws.lme_v_diag_inv,
        betas: &mut ws.lme_betas,
        var_diag: &mut ws.lme_var_diag,
        t_sq: &mut ws.lme_t_sq,
        u_scratch: &mut ws.lme_u_scratch,
        brent_log_a: &mut ws.lme_brent_log_a,
        brent_log_b: &mut ws.lme_brent_log_b,
        brent_log_c: &mut ws.lme_brent_log_c,
        brent_fa: &mut ws.lme_brent_fa,
        brent_fb: &mut ws.lme_brent_fb,
        brent_fc: &mut ws.lme_brent_fc,
        joint_sigma_t_chol: ws.lme_joint_sigma_t_chol.as_mut(),
        joint_rhs: &mut ws.lme_joint_rhs,
        joint_k_inv: ws.lme_joint_k_inv.as_mut(),
        sigma_sq: 0.0,
    }
}
