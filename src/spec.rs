//! `glmm`'s owned model-spec input vocabulary. Minimal copy of the subset of
//! MCPower's `engine_contract` cluster types that the fit kernels actually read
//! MCPower converts its `ClusterSpec` into this via a
//! conversion fn on its side — `glmm` never sees `engine_contract`.

/// Predictor column index (mirrors `engine_contract::ColumnId`'s underlying type).
pub type ColumnId = u32;

/// Cluster sizing regime. Mirror of `engine_contract::ClusterSizing`.
#[derive(Debug, Clone, PartialEq)]
pub enum Sizing {
    FixedClusters { n_clusters: u32 },
    FixedSize { cluster_size: u32 },
}

impl Sizing {
    /// Smallest legal increment in total N (the grid atom).
    pub fn atom(&self) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => (*cluster_size).max(1) as usize,
        }
    }
    pub fn n_clusters_at(&self, n: usize) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => n / (*cluster_size).max(1) as usize,
        }
    }
    pub fn cluster_of_row(&self, i: usize) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => i % (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => i / (*cluster_size).max(1) as usize,
        }
    }
}

/// One random slope. Mirror of `engine_contract::SlopeTerm`.
#[derive(Debug, Clone, PartialEq)]
pub struct SlopeTerm {
    pub column: ColumnId,
    pub variance: f64,
    pub corr_with_intercept: f64,
    pub corr_with: Vec<f64>,
}

/// An extra grouping factor. Mirror of `engine_contract::GroupingSpec`.
#[derive(Debug, Clone, PartialEq)]
pub struct Grouping {
    pub relation: GroupingRelation,
    pub tau_squared: f64,
    pub slopes: Vec<SlopeTerm>,
}

impl Grouping {
    /// The `q_g×q_g` RE *correlation* matrix `R_g` over `[intercept, slope_0, …]`
    /// (`q_g = 1 + slopes.len()`). Identical recipe to the primary's
    /// [`ModelSpec::re_correlation_matrix`]; `cluster_theta_truth` reads it to form
    /// each extra factor's relative-covariance factor `Λ_g = chol(D_g)`.
    pub fn re_correlation_matrix(&self) -> (usize, Vec<f64>) {
        re_correlation_from_slopes(&self.slopes)
    }
}

/// Build the `q×q` RE correlation matrix over `[intercept, slope_0, …]` from a
/// slope list (`q = 1 + slopes.len()`), row-major: diagonal 1;
/// `R[0][k+1]=R[k+1][0]=slopes[k].corr_with_intercept`;
/// `R[i+1][k+1]=R[k+1][i+1]=slopes[k].corr_with[i]` for `i < k`. Shared by the
/// primary factor ([`ModelSpec::re_correlation_matrix`]) and each extra
/// [`Grouping::re_correlation_matrix`] — one source, no drift.
fn re_correlation_from_slopes(slopes: &[SlopeTerm]) -> (usize, Vec<f64>) {
    let q = 1 + slopes.len();
    let mut r = vec![0.0; q * q];
    for d in 0..q {
        r[d * q + d] = 1.0;
    }
    for (k, s) in slopes.iter().enumerate() {
        r[k + 1] = s.corr_with_intercept; // R[0][k+1]
        r[(k + 1) * q] = s.corr_with_intercept; // R[k+1][0]
        for (i, &cik) in s.corr_with.iter().enumerate() {
            r[(i + 1) * q + (k + 1)] = cik;
            r[(k + 1) * q + (i + 1)] = cik;
        }
    }
    (q, r)
}

#[derive(Debug, Clone, PartialEq)]
pub enum GroupingRelation {
    Crossed { n_clusters: u32 },
    NestedWithin { n_per_parent: u32 },
}

/// Estimator class. Mirror of `engine_contract::EstimatorSpec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Estimator {
    Ols,
    Glm,
    Mle,
}

/// GLMM fixed-effect Wald-SE denominator. Mirror of `engine_contract::WaldSe`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WaldSe {
    #[default]
    Hessian,
    Rx,
}

/// The kernels' model-spec input. Exactly the fields the fit side reads.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    pub sizing: Sizing,
    pub tau_squared: f64,
    pub slopes: Vec<SlopeTerm>,
    pub extra_groupings: Vec<Grouping>,
    pub estimator: Estimator,
    pub wald_se: WaldSe,
}

impl ModelSpec {
    /// The `q_p×q_p` RE *correlation* matrix `R` over `[intercept, slope_0, …]`,
    /// row-major (`q_p = 1 + slopes.len()`). Diagonal 1; `R[0][k+1]=R[k+1][0]=
    /// slopes[k].corr_with_intercept`; `R[i+1][k+1]=R[k+1][i+1]=slopes[k].corr_with[i]`
    /// for `i < k`. Multiply by `diag(τ)` on both sides for `D`. Verbatim mirror of
    /// `engine_contract::ClusterSpec::re_correlation_matrix` — `cluster_theta_truth`
    /// reads it after the caller converts its `ClusterSpec` to `&ModelSpec`.
    pub fn re_correlation_matrix(&self) -> (usize, Vec<f64>) {
        re_correlation_from_slopes(&self.slopes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_spec_constructs_and_reports_q() {
        let spec = ModelSpec {
            sizing: Sizing::FixedClusters { n_clusters: 30 },
            tau_squared: 0.5,
            slopes: vec![SlopeTerm {
                column: 1,
                variance: 0.2,
                corr_with_intercept: 0.1,
                corr_with: vec![],
            }],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 12 },
                tau_squared: 0.3,
                slopes: vec![],
            }],
            estimator: Estimator::Mle,
            wald_se: WaldSe::Hessian,
        };
        // primary RE width q_p = 1 (intercept) + #slopes
        assert_eq!(1 + spec.slopes.len(), 2);
        assert_eq!(spec.sizing.atom(), 30);
    }
}
