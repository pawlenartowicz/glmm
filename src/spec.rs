//! `glmm`'s model-spec input vocabulary: the structural subset of cluster and
//! grouping types the fit kernels read (family, RE topology). Pure structure:
//! level counts come from the [`crate::GroupIds`] of each fit, and magnitudes
//! are fitted, not spec-carried (see [`ModelSpec`]).

/// Predictor column index — indexes into the `p`-wide design matrix `x`.
pub type ColumnId = u32;

/// One random-effect term on a grouping: the random intercept or not, and a
/// list of random-slope columns, with an unstructured covariance among the
/// term's own columns. Two terms on one grouping are independent: the
/// covariances between their columns are zero. Built by [`ReTerm::intercept`]
/// or [`ReTerm::no_intercept`], then [`ReTerm::slopes`]; a grouping takes its
/// terms through [`ReStructure::terms`] or [`Grouping::terms`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ReTerm {
    /// Whether the term holds the grouping's random intercept.
    pub intercept: bool,
    /// The term's random-slope columns (0-based indices into `x`).
    pub slopes: Vec<ColumnId>,
}

impl ReTerm {
    /// A term holding the random intercept, no slopes yet.
    pub fn intercept() -> ReTerm {
        ReTerm {
            intercept: true,
            slopes: Vec::new(),
        }
    }

    /// A term without the random intercept; give it at least one slope.
    pub fn no_intercept() -> ReTerm {
        ReTerm {
            intercept: false,
            slopes: Vec::new(),
        }
    }

    /// Sets the term's random-slope columns (0-based indices into `x`).
    pub fn slopes(mut self, cols: impl IntoIterator<Item = ColumnId>) -> ReTerm {
        self.slopes = cols.into_iter().collect();
        self
    }
}

/// The terms and the block's `slopes` for the `terms` builders: the intercept
/// term moved first (a stable move, the others keep their order), and
/// `slopes` the terms' columns concatenated in that order.
fn ordered_terms(terms: impl IntoIterator<Item = ReTerm>) -> (Vec<ReTerm>, Vec<ColumnId>) {
    let mut terms: Vec<ReTerm> = terms.into_iter().collect();
    terms.sort_by_key(|t| !t.intercept);
    let slopes = terms
        .iter()
        .flat_map(|t| t.slopes.iter().copied())
        .collect();
    (terms, slopes)
}

/// One term per column of a block whose current terms are `terms` (empty:
/// one term over the intercept and `slopes`): the intercept term, if the block
/// has a random intercept, then one term per slope in `slopes` order.
fn uncorrelated_terms(terms: &[ReTerm], slopes: &[ColumnId]) -> Vec<ReTerm> {
    let intercept = terms.is_empty() || terms.iter().any(|t| t.intercept);
    intercept
        .then(ReTerm::intercept)
        .into_iter()
        .chain(slopes.iter().map(|&c| ReTerm::no_intercept().slopes([c])))
        .collect()
}

/// Rewrites the terms' columns from `slopes`, which has their total length:
/// each term takes the next `term.slopes.len()` entries, in order. The step
/// that rewrites a block's `slopes` (a reduced design) calls it, so `terms`
/// keeps agreeing with `slopes`. A no-op on empty `terms`.
pub(crate) fn set_term_slopes(terms: &mut [ReTerm], slopes: &[ColumnId]) {
    let mut rest = slopes;
    for t in terms {
        let (head, tail) = rest.split_at(t.slopes.len());
        t.slopes.copy_from_slice(head);
        rest = tail;
    }
}

/// An extra (crossed or nested) grouping factor beyond the primary grouping.
/// `slopes` are the random-slope design columns; with no `terms` the RE
/// correlation structure is full (parametrized by the fitted θ). Structure-only,
/// like [`ModelSpec`]: magnitudes and warm starts are not carried here. Built by
/// [`Grouping::crossed`] or [`Grouping::nested`], then [`Grouping::slopes`] or
/// [`Grouping::terms`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Grouping {
    /// How this grouping's clusters relate to the primary grouping's rows.
    pub relation: GroupingRelation,
    /// Random-slope design columns for this grouping (0-based indices into `x`);
    /// empty means random-intercept only.
    pub slopes: Vec<ColumnId>,
    /// This grouping's random-effect terms, as on [`ReStructure::terms`].
    pub terms: Vec<ReTerm>,
}

impl Grouping {
    /// A crossed extra grouping, intercept only.
    pub fn crossed() -> Grouping {
        Grouping {
            relation: GroupingRelation::Crossed,
            slopes: Vec::new(),
            terms: Vec::new(),
        }
    }

    /// An extra grouping nested within the primary one, intercept only.
    pub fn nested() -> Grouping {
        Grouping {
            relation: GroupingRelation::NestedWithin,
            slopes: Vec::new(),
            terms: Vec::new(),
        }
    }

    /// Sets this grouping's random-slope columns (0-based indices into `x`).
    pub fn slopes(mut self, cols: impl IntoIterator<Item = ColumnId>) -> Grouping {
        self.slopes = cols.into_iter().collect();
        self
    }

    /// Sets this grouping's terms, as [`ReStructure::terms`] does.
    pub fn terms(mut self, terms: impl IntoIterator<Item = ReTerm>) -> Grouping {
        (self.terms, self.slopes) = ordered_terms(terms);
        self
    }

    /// One term per column, as [`ReStructure::uncorrelated`] does.
    pub fn uncorrelated(mut self) -> Grouping {
        self.terms = uncorrelated_terms(&self.terms, &self.slopes);
        self
    }
}

/// How an extra grouping's clusters map onto rows, relative to the primary
/// grouping. The level counts come from the fit's [`crate::GroupIds`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupingRelation {
    /// Independent (crossed) with the primary grouping: each level may touch
    /// any primary cluster (e.g. items crossed with subjects).
    Crossed,
    /// Nested within the primary grouping: each level belongs to exactly one
    /// primary cluster. Its ids are global child levels, dense over all
    /// parents.
    NestedWithin,
}

/// Outcome distribution + link. Selects the fit kernel together with
/// [`ModelSpec::re`] (`re.is_some()` ⇒ mixed): `Gaussian` → OLS / LMM,
/// `Binomial{Logit}` → GLM / GLMM. Every variant here has a wired kernel, so no
/// kernel-less variant is reachable through [`crate::fit`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// Normal response, identity link → OLS (`re: None`) or LMM (`re: Some`).
    Gaussian,
    /// Bernoulli/binomial response → logistic/probit GLM (`re: None`) or GLMM
    /// (`re: Some`). Counts are fit as expanded 0/1 rows (the kernel is
    /// Bernoulli; see [`crate::fit`]). Variance `V(μ)=μ(1−μ)`,
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
    /// Gamma response (`y>0`) → GLM/GLMM. Variance `V(μ)=μ²`. On a GLM the
    /// reported dispersion `φ̂` is the Pearson moment (`summary(glm)$dispersion`),
    /// and it is the one that scales the SE by `√φ̂`; the maximum-likelihood φ̂
    /// (`MASS::gamma.shape`'s equation) enters the log-likelihood only. On a GLMM
    /// `φ` is a coordinate of the Laplace objective, estimated by maximum
    /// likelihood, reported as `φ̂`, and carried into the SE through the PIRLS
    /// weights `wᵢ/φ̂`. Validated against R `glm(family=Gamma(link))` at the ML φ̂
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
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonLink {
    /// Canonical log link `g(μ) = ln(μ)`, `μ=exp(η)`.
    Log,
}

/// Gamma link function. `Log` is the safe default; `Inverse` is the classic
/// Gamma link but can drive `μ≤0` mid-IRLS (domain-clamped).
#[non_exhaustive]
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
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NegBinomialLink {
    /// Log link `g(μ) = ln(μ)`, `μ=exp(η)`. Non-canonical (the NB canonical link
    /// `log(μ/(μ+θ))` is not offered) → general Fisher-scoring branch.
    Log,
}

/// Inverse-Gaussian link function. `Log` is the safe default; `InverseSquared`
/// is R's `inverse.gaussian()` default and can drive `μ≤0` mid-IRLS
/// (domain-clamped).
#[non_exhaustive]
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
/// Built by [`ReStructure::intercept`], then [`ReStructure::slopes`] or
/// [`ReStructure::terms`], and [`ReStructure::extra`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ReStructure {
    /// Random-slope design columns for the primary grouping (0-based indices
    /// into `x`); empty means random-intercept only.
    pub slopes: Vec<ColumnId>,
    /// The primary grouping's random-effect terms. Empty means one
    /// unstructured term over the intercept and `slopes`. When set, `slopes`
    /// must be the terms' columns concatenated in term order, at most one
    /// term holds the intercept and it comes first, no term is empty, and no
    /// column is in two terms; [`crate::fit`] refuses anything else with
    /// [`crate::FitError::Shape`]. With no intercept term the grouping has no
    /// random intercept.
    pub terms: Vec<ReTerm>,
    /// Additional crossed/nested grouping factors beyond the primary grouping.
    pub extra_groupings: Vec<Grouping>,
}

impl ReStructure {
    /// A random intercept on the primary grouping: no slopes, no extra
    /// groupings.
    pub fn intercept() -> ReStructure {
        ReStructure {
            slopes: Vec::new(),
            terms: Vec::new(),
            extra_groupings: Vec::new(),
        }
    }

    /// Sets the primary grouping's random-slope columns (0-based indices into
    /// `x`).
    pub fn slopes(mut self, cols: impl IntoIterator<Item = ColumnId>) -> ReStructure {
        self.slopes = cols.into_iter().collect();
        self
    }

    /// Sets the primary grouping's terms, and `slopes` to their columns. The
    /// intercept term, if any, is moved first; the others keep their order.
    /// `Re::intercept().terms([ReTerm::intercept().slopes([1]),
    /// ReTerm::no_intercept().slopes([2])])` is lme4's
    /// `(1 + x1 | g) + (0 + x2 | g)`.
    pub fn terms(mut self, terms: impl IntoIterator<Item = ReTerm>) -> ReStructure {
        (self.terms, self.slopes) = ordered_terms(terms);
        self
    }

    /// One term per column: the random intercept (when the grouping has one)
    /// and each slope get independent variances, with zero covariances.
    /// `Re::intercept().slopes([1]).uncorrelated()` is lme4's `(x || g)`.
    pub fn uncorrelated(mut self) -> ReStructure {
        self.terms = uncorrelated_terms(&self.terms, &self.slopes);
        self
    }

    /// Appends one extra grouping, after those already declared.
    pub fn extra(mut self, g: Grouping) -> ReStructure {
        self.extra_groupings.push(g);
        self
    }
}

/// Short name for [`ReStructure`].
pub type Re = ReStructure;

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
///
/// # Examples
///
/// A logistic GLMM with a random intercept and slope on column 1 for the
/// primary grouping, plus a crossed random intercept, and a fixed-only OLS:
///
/// ```rust
/// use glmm::{BinomialLink, Family, GroupIds, Grouping, GroupingRelation, ModelSpec, Re};
///
/// let (primary, items) = (vec![0, 0, 1, 1], vec![0, 1, 0, 1]);
/// let spec = ModelSpec::mixed(
///     Family::Binomial { link: BinomialLink::Logit },
///     Re::intercept().slopes([1]).extra(Grouping::crossed()),
/// );
/// let ids = GroupIds::new(primary).extra(items);
/// let ols = ModelSpec::fixed(Family::Gaussian);
///
/// let re = spec.re.as_ref().unwrap();
/// assert_eq!(re.slopes, vec![1]);
/// assert_eq!(re.extra_groupings.len(), 1);
/// assert_eq!(re.extra_groupings[0].relation, GroupingRelation::Crossed);
/// assert_eq!(ids.extra.len(), 1);
/// assert!(ols.re.is_none());
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSpec {
    /// Outcome distribution + link, selecting the fit kernel.
    pub family: Family,
    /// Random-effect structure; `None` for a fixed-only model (OLS/GLM),
    /// `Some` for mixed (LMM/GLMM).
    pub re: Option<ReStructure>,
}

impl ModelSpec {
    /// A fixed-only model (OLS or GLM).
    pub fn fixed(family: Family) -> ModelSpec {
        ModelSpec { family, re: None }
    }

    /// A mixed model (LMM or GLMM).
    pub fn mixed(family: Family, re: ReStructure) -> ModelSpec {
        ModelSpec {
            family,
            re: Some(re),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Data, FitError, FitOptions, GroupIds};

    /// A spec whose terms do not describe its blocks returns
    /// `FitError::Shape` naming the grouping and the reason, on the primary
    /// and on an extra. Each row breaks one rule, so the reason names the
    /// guard that refused it.
    #[test]
    fn bad_terms_are_a_shape_error() {
        let n = 12;
        let x: Vec<f64> = (0..n)
            .flat_map(|i| [1.0, (i % 5) as f64, (i % 3) as f64])
            .collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 7) % 11) as f64).collect();
        let ids = GroupIds::new((0..n as u32).map(|i| i % 4).collect())
            .extra((0..n as u32).map(|i| i % 3).collect());
        let data = Data::new(&x, &y, n, 3).ids(&ids);
        let shape_error = |re: Re, grouping: &str, reason: &str| {
            let spec = ModelSpec::mixed(Family::Gaussian, re);
            match crate::fit(&data, &spec, &FitOptions::default()) {
                Err(FitError::Shape(m)) => {
                    assert!(m.contains(grouping) && m.contains(reason), "{m}")
                }
                Err(e) => panic!("{spec:?}: {e:?}"),
                Ok(_) => panic!("{spec:?}: fitted"),
            }
        };
        let crossed = || Re::intercept().extra(Grouping::crossed());
        let not_in_order = "not the terms' columns in term order";
        let empty = "neither the intercept nor a slope";

        // `slopes` changed after the terms were set.
        shape_error(
            Re::intercept()
                .terms([ReTerm::intercept().slopes([1])])
                .slopes([2]),
            "primary grouping",
            not_in_order,
        );
        // A slope column in two terms.
        shape_error(
            Re::intercept().terms([
                ReTerm::intercept().slopes([1]),
                ReTerm::no_intercept().slopes([1]),
            ]),
            "primary grouping",
            "slope column 1 is in two terms",
        );
        // The intercept in two terms.
        shape_error(
            Re::intercept().terms([ReTerm::intercept(), ReTerm::intercept().slopes([1])]),
            "primary grouping",
            "must be in term 0",
        );
        // An empty term.
        shape_error(
            Re::intercept().terms([ReTerm::intercept(), ReTerm::no_intercept()]),
            "primary grouping",
            empty,
        );
        // The intercept term not first, which only a field write can do.
        // `slopes` is swapped too, so it still lists the terms' columns in
        // term order and only the intercept guard can refuse the spec.
        let mut re = Re::intercept().terms([
            ReTerm::intercept().slopes([1]),
            ReTerm::no_intercept().slopes([2]),
        ]);
        re.terms.swap(0, 1);
        re.slopes.swap(0, 1);
        shape_error(re, "primary grouping", "must be in term 0");
        // The same checks reach an extra grouping.
        shape_error(
            crossed().extra(Grouping::crossed().terms([ReTerm::no_intercept()])),
            "extra grouping 1",
            empty,
        );
        let mut re = crossed();
        re.extra_groupings[0] = Grouping::crossed()
            .terms([ReTerm::no_intercept().slopes([1])])
            .slopes([1, 2]);
        shape_error(re, "extra grouping 0", not_in_order);
    }
}
