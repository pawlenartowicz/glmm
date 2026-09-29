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
    /// Number of clusters when total row count is `n`: the fixed `n_clusters`
    /// itself under `FixedClusters`, or `n / cluster_size` rounded UP under
    /// `FixedSize`. Rounding up matches the positional row layout (row `i` in
    /// cluster `i / cluster_size`): off-grid `n` leaves a partial trailing cluster
    /// whose rows still carry a real id, and that id must be in range.
    pub fn n_clusters_at(&self, n: usize) -> usize {
        match self {
            Sizing::FixedClusters { n_clusters } => (*n_clusters).max(1) as usize,
            Sizing::FixedSize { cluster_size } => n.div_ceil((*cluster_size).max(1) as usize),
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
/// `Binomial{Logit}` → GLM / GLMM. Every variant here has a wired kernel, so no
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
    /// `glm(family=poisson)` / `lme4::glmer(family=poisson)` (validation goldens
    /// `grouseticks_glm`, `grouseticks`).
    Poisson {
        /// Link function — log only (canonical).
        link: PoissonLink,
    },
    /// Gamma response (`y>0`) → GLM/GLMM. Variance `V(μ)=μ²`. Dispersion `φ` is
    /// estimated by maximum likelihood — post-fit on a GLM (`MASS::gamma.shape`'s
    /// equation), as a coordinate of the Laplace objective on a GLMM — and scales
    /// the SE by `√φ̂`. Validated against R `glm(family=Gamma(link))` at the ML φ̂
    /// and glmmTMB (validation goldens `sim_gamma_*_ml` and `*_tmb`).
    Gamma {
        /// Link function — [`GammaLink::Log`] (safe default) or `Inverse`. The
        /// dispersion directive (estimate vs hold-fixed φ) lives in
        /// [`crate::FitOptions`], not here.
        link: GammaLink,
    },
    /// Negative-binomial count response → log-link GLM/GLMM. Variance
    /// `V(μ)=μ+μ²/θ`. On a GLM the shape `θ` is estimated by an alternating
    /// outer loop (`MASS::glm.nb`/`lme4::glmer.nb` style), reported in
    /// `Fit.dispersion`, and the β SE conditions on `θ̂` (θ-uncertainty out of
    /// scope). On a GLMM, `ln θ_NB` is instead a coordinate of the outer
    /// search, and the β SE carries its uncertainty through an appended
    /// Hessian row (see `crate::glmm::joint_hessian_cov`). θ̂ is not
    /// spec-carried (structure-only, see [`ModelSpec`]): the MLE is
    /// start-independent, so a spec-supplied warm-start could only seed the
    /// optimizer without changing the converged θ̂. The fit threads θ̂
    /// explicitly through the numeric stack instead. Validated against R
    /// `MASS::glm.nb` / `lme4::glmer.nb` (validation goldens `sim_nb_*`).
    NegativeBinomial {
        /// Link function — log only (`log(μ/(μ+θ))` canonical link not offered).
        link: NegBinomialLink,
    },
    /// Inverse-Gaussian response (`y>0`) → GLM. Variance `V(μ)=μ³`. Dispersion
    /// `φ` is estimated post-fit as the Pearson moment estimator
    /// `φ̂=Σ rᵢ²/(n−p)` (`rᵢ=(yᵢ−μ̂ᵢ)/√(μ̂ᵢ³)`), `summary(glm)`'s, and scales the
    /// SE by `√φ̂`. **Mixed models are not wired**:
    /// `fit` faults at the model-shape gate for `re: Some(..)`, because the
    /// profiled `inverse.gaussian()$aic` objective term the GLMM needs is not
    /// built. Validated against R `glm(family=inverse.gaussian(link))`
    /// (validation goldens `sim_igauss_glm`, `sim_igauss_inv_sq_glm`).
    InverseGaussian {
        /// Link function — [`InverseGaussianLink::Log`] (safe default) or
        /// `InverseSquared`.
        link: InverseGaussianLink,
    },
}

/// Binomial link function. `Logit` is canonical (the fused-SIMD kernel);
/// `Probit` and `Cloglog` both use the general Fisher-scoring branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinomialLink {
    /// Canonical logit link `g(μ) = ln(μ/(1−μ))`.
    Logit,
    /// Probit link `g(μ) = Φ⁻¹(μ)` (inverse standard-normal CDF). Non-canonical:
    /// `μ=Φ(η)`, `dμ/dη=φ(η)`. Validated against R `binomial(link="probit")`.
    Probit,
    /// Complementary log-log link `g(μ) = ln(−ln(1−μ))`. Non-canonical:
    /// `μ = 1−exp(−exp(η))`, `dμ/dη = exp(η−exp(η))`, so it uses the general
    /// Fisher-scoring branch. Asymmetric — μ approaches 1 much faster than 0,
    /// which is why η carries an upper clamp the other two links do not need.
    /// Validated against R `binomial(link="cloglog")` (validation goldens
    /// `sim_cloglog_glm`, `sim_cloglog_glmm`).
    Cloglog,
}

/// Poisson link function. Only the canonical log link is offered.
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

/// Negative-binomial link function. Only the log link is offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegBinomialLink {
    /// Log link `g(μ) = ln(μ)`, `μ=exp(η)`. Non-canonical (the NB canonical link
    /// `log(μ/(μ+θ))` is not offered) → general Fisher-scoring branch.
    Log,
}

/// Inverse-Gaussian link function. `Log` is the safe default; `InverseSquared`
/// is R's `inverse.gaussian()` default and can drive `μ≤0` mid-IRLS
/// (domain-clamped).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InverseGaussianLink {
    /// `g(μ) = 1/μ²`, `μ = η^(−1/2)` (R `inverse.gaussian(link="1/mu^2")`).
    /// This is the family's canonical link only up to sign and scale (the
    /// natural parameter is `θ = −1/(2μ²)`), so `dμ/dη ≠ V(μ)` and the
    /// canonical IRLS shortcut would mis-sign and mis-scale the working
    /// residual — it takes the general Fisher-scoring branch, exactly as
    /// [`GammaLink::Inverse`] does. Requires `μ>0` → `η>0`.
    InverseSquared,
    /// Log link `g(μ) = ln μ`, `μ = exp(η)`. Non-canonical but stable.
    Log,
}

/// GLMM fixed-effect Wald-SE denominator.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WaldSe {
    /// Hessian of the joint (θ, β) Laplace deviance — the lme4
    /// `use.hessian = TRUE`-matching default. Computed exactly wherever an
    /// exact pass takes the shape: the assembled adjoint pass, which includes
    /// the packed-row layout, and the hyper-dual pass on the shapes
    /// `derivative::supports_shape` accepts (the blocked path and the
    /// structured-extras shapes within the measured tail bound). A shape both
    /// passes decline falls back to a finite-difference Hessian.
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
