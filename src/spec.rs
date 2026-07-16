//! `glmm`'s model-spec input vocabulary: the structural subset of cluster and
//! grouping types the fit kernels read (family, RE topology, sizing). Pure
//! structure — magnitudes are fitted, not spec-carried (see [`ModelSpec`]).

/// Predictor column index — indexes into the `p`-wide design matrix `x`.
pub type ColumnId = u32;

/// Cluster sizing regime: a fixed cluster count, or a fixed per-cluster size.
#[derive(Debug, Clone, PartialEq)]
pub enum Sizing {
    /// Fixed number of clusters; per-cluster size grows with total N.
    FixedClusters {
        /// Cluster count, held constant as N scales.
        n_clusters: u32,
    },
    /// Fixed per-cluster size; cluster count grows with total N.
    FixedSize {
        /// Rows per cluster, held constant as N scales.
        cluster_size: u32,
    },
}

impl Sizing {
    /// Smallest legal increment in total N (the grid atom).
    pub fn atom(&self) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => (*cluster_size).max(1) as usize,
        }
    }
    /// Number of clusters when total row count is `n`: the fixed `n_clusters`
    /// itself under `FixedClusters`, or `n / cluster_size` (floor) under
    /// `FixedSize`.
    pub fn n_clusters_at(&self, n: usize) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => n / (*cluster_size).max(1) as usize,
        }
    }
    /// Cluster index owning row `i` (0-based row index into `x`/`y`). Under
    /// `FixedClusters`, rows are dealt round-robin (`i % n_clusters`); under
    /// `FixedSize`, rows are contiguous per cluster (`i / cluster_size`).
    pub fn cluster_of_row(&self, i: usize) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => i % (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => i / (*cluster_size).max(1) as usize,
        }
    }
}

/// An extra (crossed or nested) grouping factor beyond the primary grouping.
/// `slopes` are the random-slope design columns; the RE correlation structure is
/// always full (parametrized by the fitted θ). Structure-only, like [`ModelSpec`]:
/// magnitudes and warm starts are not carried here.
#[derive(Debug, Clone, PartialEq)]
pub struct Grouping {
    /// How this grouping's clusters relate to the primary grouping's rows.
    pub relation: GroupingRelation,
    /// Random-slope design columns for this grouping (0-based indices into `x`);
    /// empty means random-intercept only.
    pub slopes: Vec<ColumnId>,
}

/// How an extra grouping's clusters map onto rows, relative to the primary
/// grouping.
#[derive(Debug, Clone, PartialEq)]
pub enum GroupingRelation {
    /// Independent (crossed) with the primary grouping: `n_clusters` clusters,
    /// each potentially touching any primary cluster (e.g. items crossed with
    /// subjects).
    Crossed {
        /// Cluster count for this crossed factor.
        n_clusters: u32,
    },
    /// Nested within the primary grouping: each primary cluster contains
    /// `n_per_parent` clusters of this factor, uniquely owned by that parent.
    NestedWithin {
        /// Number of this factor's clusters per parent (primary) cluster.
        n_per_parent: u32,
    },
}

/// Outcome distribution + link. Selects the fit kernel together with
/// [`ModelSpec::re`] (`re.is_some()` ⇒ mixed): `Gaussian` → OLS / LMM,
/// `Binomial{Logit}` → GLM / GLMM. M2 wires only the variants whose kernel
/// exists — `Poisson`/probit/etc. are added when their kernels land (M3), so no
/// kernel-less variant is reachable through [`crate::fit_cold`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// Normal response, identity link → OLS (`re: None`) or LMM (`re: Some`).
    Gaussian,
    /// Bernoulli/binomial response → logistic/probit GLM (`re: None`) or GLMM
    /// (`re: Some`). Counts are fit as expanded 0/1 rows (the kernel is
    /// Bernoulli; see [`crate::fit_cold`]). Variance `V(μ)=μ(1−μ)`,
    /// dispersion `φ≡1`.
    Binomial {
        /// Link function — [`BinomialLink::Logit`] (canonical) or `Probit`.
        link: BinomialLink,
    },
    /// Poisson count response → log-link GLM/GLMM. Variance `V(μ)=μ`, dispersion
    /// `φ≡1`. Deviance residual `dᵢ=2[yᵢ·ln(yᵢ/μᵢ)−(yᵢ−μᵢ)]`. Validated against R
    /// `glm(family=poisson)` / `lme4::glmer(family=poisson)` (parity goldens
    /// `grouseticks_glm`, `grouseticks`).
    Poisson {
        /// Link function — log only (canonical).
        link: PoissonLink,
    },
    /// Gamma response (`y>0`) → GLM/GLMM. Variance `V(μ)=μ²`. Dispersion `φ` is
    /// estimated post-fit as the Pearson moment estimator `φ̂=Σ rᵢ²/(n−p)`
    /// (`rᵢ=(yᵢ−μ̂ᵢ)/μ̂ᵢ`) and scales the SE by `√φ̂`. Validated against R
    /// `glm(family=Gamma(link))` / `lme4::glmer(family=Gamma)` (parity goldens
    /// `sim_gamma_*`).
    Gamma {
        /// Link function — [`GammaLink::Log`] (safe default) or `Inverse`. The
        /// dispersion directive (estimate vs hold-fixed φ) lives in
        /// [`crate::FitOptions`], not here.
        link: GammaLink,
    },
    /// Negative-binomial count response → log-link GLM/GLMM. Variance
    /// `V(μ)=μ+μ²/θ`. The shape `θ` is estimated by an alternating outer loop
    /// (`MASS::glm.nb`/`lme4::glmer.nb` style) and reported in `Fit.dispersion`;
    /// the β SE conditions on `θ̂`. θ̂ is not spec-carried (structure-only, see
    /// [`ModelSpec`]): the MLE is start-independent, so a spec-supplied warm-start
    /// could only seed the optimizer without changing the converged θ̂. The fit
    /// threads θ̂ explicitly through the numeric stack instead. Validated against R
    /// `MASS::glm.nb` / `lme4::glmer.nb` (parity goldens `sim_nb_*`).
    NegativeBinomial {
        /// Link function — log only (`log(μ/(μ+θ))` canonical link not offered).
        link: NegBinomialLink,
    },
}

/// Binomial link function. `Logit` is canonical (the fused-SIMD kernel);
/// `Probit` (added M3) uses the general Fisher-scoring branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinomialLink {
    /// Canonical logit link `g(μ) = ln(μ/(1−μ))`.
    Logit,
    /// Probit link `g(μ) = Φ⁻¹(μ)` (inverse standard-normal CDF). Non-canonical:
    /// `μ=Φ(η)`, `dμ/dη=φ(η)`. Validated against R `binomial(link="probit")`.
    Probit,
}

/// Poisson link function. M3 ships the canonical log link only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonLink {
    /// Canonical log link `g(μ) = ln(μ)`, `μ=exp(η)`.
    Log,
}

/// Gamma link function. `Log` is the safe default; `Inverse` is the classic
/// Gamma link but can drive `μ≤0` mid-IRLS (domain-clamped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GammaLink {
    /// Log link `g(μ) = ln(μ)`, `μ=exp(η)`. Non-canonical for Gamma but stable.
    Log,
    /// Inverse link `g(μ) = 1/μ`, `μ=1/η`. This is the *negative* of the Gamma
    /// natural parameter `θ=−1/μ`, so it is non-canonical here and uses the
    /// general Fisher-scoring branch (the canonical shortcut would mis-sign the
    /// working residual). Requires `μ>0` → `η>0`.
    Inverse,
}

/// Negative-binomial link function. M3 ships the log link only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegBinomialLink {
    /// Log link `g(μ) = ln(μ)`, `μ=exp(η)`. Non-canonical (the NB canonical link
    /// `log(μ/(μ+θ))` is not offered) → general Fisher-scoring branch.
    Log,
}

/// GLMM fixed-effect Wald-SE denominator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WaldSe {
    /// FD-Hessian of the joint (θ, β) Laplace deviance — the lme4
    /// `use.hessian = TRUE`-matching default.
    #[default]
    Hessian,
    /// Direct inverse of the expected-information Schur complement (assumes
    /// β–θ orthogonality); anticonservative for the GLMM.
    Rx,
}

/// Random-effect structure of a mixed model — present iff the model has random
/// effects. Carried in [`ModelSpec::re`] as `Option`, so `None` is a fixed-only
/// model (OLS/GLM) and `Some` is mixed (LMM/GLMM): an OLS-with-grouping state is
/// unrepresentable. Holds exactly the RE fields the LMM/GLMM kernels read.
#[derive(Debug, Clone, PartialEq)]
pub struct ReStructure {
    /// Primary grouping's cluster-count regime.
    pub sizing: Sizing,
    /// Random-slope design columns for the primary grouping (0-based indices
    /// into `x`); empty means random-intercept only.
    pub slopes: Vec<ColumnId>,
    /// Additional crossed/nested grouping factors beyond the primary grouping.
    pub extra_groupings: Vec<Grouping>,
}

/// The kernels' model-spec input. `family` selects the outcome kernel; `re`
/// selects fixed-only vs mixed (`None` → OLS/GLM, `Some` → LMM/GLMM).
///
/// Structure-only: `ModelSpec` and its fields (`Family`, [`ReStructure`],
/// [`Grouping`]) carry topology and column indices, never fitted magnitudes or
/// warm-start state. Fitted variances/covariances live in the returned `Fit`;
/// method knobs (`wald_se`, `nagq`, the Gamma φ directive) live in
/// [`crate::FitOptions`]; any warm start is the caller-supplied
/// [`crate::StartValues`]. This split is why the same spec serves both a cold and
/// a warm fit unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Outcome distribution + link, selecting the fit kernel.
    pub family: Family,
    /// Random-effect structure; `None` for a fixed-only model (OLS/GLM),
    /// `Some` for mixed (LMM/GLMM).
    pub re: Option<ReStructure>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_spec_constructs_and_reports_q() {
        let re = ReStructure {
            sizing: Sizing::FixedClusters { n_clusters: 30 },
            slopes: vec![1],
            extra_groupings: vec![Grouping {
                relation: GroupingRelation::Crossed { n_clusters: 12 },
                slopes: vec![],
            }],
        };
        let spec = ModelSpec {
            family: Family::Gaussian,
            re: Some(re),
        };
        let re = spec.re.as_ref().unwrap();
        assert_eq!(re.sizing.atom(), 30);
    }

    #[test]
    fn m3_families_construct_and_are_copy() {
        fn assert_copy<T: Copy>(_: T) {}
        let f = Family::Gamma {
            link: GammaLink::Log,
        };
        assert_copy(f); // Family stays Copy (structure-only, see ModelSpec)
        let _ = Family::Poisson {
            link: PoissonLink::Log,
        };
        let _ = Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        };
        let _ = Family::Binomial {
            link: BinomialLink::Probit,
        };
    }

    #[test]
    #[should_panic(expected = "nagq")]
    fn nagq_even_rejected() {
        let model = ModelSpec {
            family: Family::Binomial {
                link: BinomialLink::Logit,
            },
            re: Some(ReStructure {
                sizing: Sizing::FixedClusters { n_clusters: 4 },
                slopes: vec![],
                extra_groupings: vec![],
            }),
        };
        // nagq=4 (even) is now a FitOptions value, passed straight to the checker.
        crate::fit::assert_model_shape_pub(&model, 2, 4);
    }
}
