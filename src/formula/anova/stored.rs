//! The plain-numbers side of a comparison: [`Model`], what a port's stored
//! fit object keeps, and [`compare_stored`], the comparison of two of them
//! with no refit. Public with the `orchestrate` feature, which the ports build
//! with; crate-private otherwise, where [`compare`](super::compare) reaches
//! them through a fitted [`formula::Model`](crate::formula::Model).

use super::{
    case, nesting, same_or_default, CompareError, CompareOptions, Comparison, Differs, ModelInfo,
    Nesting, Side,
};
use crate::fit::vech_q;
use crate::formula::Lowered;
use crate::inference::Objective;
use crate::{Family, Fit, FitOptions};

/// What a comparison reads of one fitted model: plain numbers, as a port's fit
/// object keeps them, so two stored fits are compared with no refit.
///
/// [`compare`](super::compare) reads one off each fitted
/// [`formula::Model`](crate::formula::Model); a
/// port builds one from the fields it stored and calls [`compare_stored`].
#[derive(Clone, Debug)]
pub struct Model {
    /// The formula the model was fitted with.
    pub formula: String,
    /// The family and its link.
    pub family: Family,
    /// [`Fit::loglik`].
    pub loglik: f64,
    /// [`Fit::df`], the parameter count.
    pub df: usize,
    /// [`Fit::nobs`].
    pub nobs: usize,
    /// [`Fit::reml`]: whether `loglik` is a REML criterion.
    pub reml: bool,
    /// [`Fit::objective`].
    pub objective: Objective,
    /// [`Fit::converged`].
    pub converged: bool,
    /// The vech-packed blocks behind [`Fit::varcorr()`].
    pub varcorr: Vec<Vec<f64>>,
    /// Each `varcorr` block's grouping name and term names, in the same
    /// order ([`Lowered::re_groups`]); empty when there is no random part.
    /// The zero-component rule reads it to match blocks and terms by name.
    /// [`compare_stored`] refuses a side whose `re_groups` does not list one
    /// grouping per `varcorr` block, each with one term per block dimension
    /// ([`CompareError::ReGroups`]), so fill it from the fit's `re_groups`.
    pub re_groups: Vec<(String, Vec<String>)>,
    /// The response as fitted ([`Lowered::y`]).
    pub y: Vec<f64>,
    /// The prior weights, `None` when unweighted.
    pub weights: Option<Vec<f64>>,
    /// The offset, `None` when there is none.
    pub offset: Option<Vec<f64>>,
    /// The value of [`crate::FitOptions::dispersion`]: `Some(v)` for
    /// [`crate::Dispersion::Fixed`]`(v)`, `None` for
    /// [`crate::Dispersion::Default`]. It is stored as given on every family,
    /// also on one that ignored it (Gaussian, binomial, Poisson, negative
    /// binomial), and [`compare_stored`] refuses two sides that differ in it.
    pub dispersion_held: Option<f64>,
    /// The design summary `(counts, s, c)` of a Gaussian fit whose only random
    /// term is one unweighted `(1 | g)`, flat and column-major; `None` on
    /// every other fit.
    pub design_summary: Option<(Vec<f64>, Vec<f64>, Vec<f64>)>,
    /// The weighted residual sum of squares `Σwᵢ(yᵢ − μ̂ᵢ)²` of a converged
    /// OLS fit; NaN on every other fit.
    pub rss: f64,
    /// The coefficient names, one per fixed-effect column.
    pub coef_names: Vec<String>,
    /// The coefficients β̂ ([`Fit::beta`]).
    pub coef: Vec<f64>,
    /// The coefficients' covariance ([`Fit::vcov`]).
    pub vcov: Vec<Vec<f64>>,
    /// The aliased-column mask ([`Fit::aliased`]).
    pub aliased: Vec<bool>,
}

impl Model {
    /// The model of `fit`, fitted on `lowered` from `formula` with
    /// `fit_opts` (the weights and the offset are read from `lowered`, the
    /// held dispersion from `fit_opts`).
    pub(crate) fn from_fit(
        formula: &str,
        lowered: &Lowered,
        fit_opts: &FitOptions,
        fit: &Fit,
    ) -> Model {
        let ols = lowered.model.re.is_none() && lowered.model.family == Family::Gaussian;
        let rss = if ols && fit.converged() && fit.fitted.len() == lowered.y.len() {
            let w = |i: usize| lowered.weights.as_ref().map_or(1.0, |w| w[i]);
            (0..lowered.y.len())
                .map(|i| w(i) * (lowered.y[i] - fit.fitted[i]).powi(2))
                .sum()
        } else {
            f64::NAN
        };
        Model {
            formula: formula.to_string(),
            family: lowered.model.family,
            loglik: fit.loglik,
            df: fit.df,
            nobs: fit.nobs,
            reml: fit.reml,
            objective: fit.objective,
            converged: fit.converged(),
            varcorr: fit.varcorr.clone(),
            re_groups: lowered
                .re_groups
                .iter()
                .map(|g| (g.name.clone(), g.terms.clone()))
                .collect(),
            y: lowered.y.clone(),
            weights: lowered.weights.clone(),
            offset: lowered.offset.clone(),
            dispersion_held: fit_opts.dispersion.held(),
            design_summary: fit
                .design_summary
                .as_ref()
                .map(|d| (d.counts.clone(), d.s.clone(), d.c.clone())),
            rss,
            coef_names: lowered.col_names.clone(),
            coef: fit.beta.clone(),
            vcov: fit.vcov_rows(),
            aliased: fit.aliased().to_vec(),
        }
    }

    /// An OLS fit: Gaussian with no random part, whatever its `reml`.
    pub(super) fn is_ols(&self) -> bool {
        self.objective == Objective::LeastSquares
    }

    /// A Gamma or inverse Gaussian GLM whose dispersion is estimated. With φ
    /// held, the pair is an ordinary likelihood-ratio pair.
    pub(super) fn is_dispersion_glm(&self) -> bool {
        matches!(
            self.family,
            Family::Gamma { .. } | Family::InverseGaussian { .. }
        ) && self.objective == Objective::MaximumLikelihood
            && self.dispersion_held.is_none()
    }

    /// The number of random-effect dimensions: the width of every `varcorr`
    /// block, summed.
    pub(super) fn random_dims(&self) -> usize {
        self.varcorr.iter().map(|b| vech_q(b.len())).sum()
    }
}

/// Compares two stored models, given in either order: runs the checks, then
/// the test of the pair's case (see the [`anova`](super) module header for
/// both). The exact
/// restricted likelihood-ratio test draws `opts.nsim` times from `opts.seed`,
/// serially, so one call gives the same p-value in every build.
///
/// # Errors
/// The first check that fails, as a [`CompareError`].
pub fn compare_stored(
    a: &Model,
    b: &Model,
    opts: &CompareOptions,
) -> Result<Comparison, CompareError> {
    // 1. Both converged.
    if !(a.converged && b.converged) {
        return Err(CompareError::NotConverged {
            m0: !a.converged,
            m1: !b.converged,
        });
    }
    // Each side's groupings describe its variance blocks. Checked after
    // convergence: a fit that did not converge may have no blocks (an empty
    // `varcorr`) beside its groupings.
    for (side, m) in [(Side::First, a), (Side::Second, b)] {
        let described = m.re_groups.len() == m.varcorr.len()
            && m.re_groups
                .iter()
                .zip(&m.varcorr)
                .all(|((_, terms), block)| {
                    let t = terms.len();
                    t * (t + 1) / 2 == block.len()
                });
        if !described {
            return Err(CompareError::ReGroups { side });
        }
    }
    // 2. The same data.
    if a.nobs != b.nobs {
        return Err(CompareError::DifferentRows {
            m0: a.nobs,
            m1: b.nobs,
        });
    }
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    if bits(&a.y) != bits(&b.y) {
        return Err(CompareError::DifferentY);
    }
    if !same_or_default(&a.weights, &b.weights, a.nobs, 1.0) {
        return Err(CompareError::DifferentWeights);
    }
    if !same_or_default(&a.offset, &b.offset, a.nobs, 0.0) {
        return Err(CompareError::DifferentOffset);
    }
    if a.family != b.family {
        return Err(CompareError::DifferentFamily {
            m0: a.family,
            m1: b.family,
        });
    }
    if a.dispersion_held.map(f64::to_bits) != b.dispersion_held.map(f64::to_bits) {
        return Err(CompareError::DifferentDispersionHeld {
            m0: a.dispersion_held,
            m1: b.dispersion_held,
        });
    }
    // 3. Nested, and which way.
    let nested = nesting(&a.formula, &b.formula)?;
    // 4. The estimator. Two OLS fits pass whatever their `reml`.
    if !(a.is_ols() && b.is_ols()) {
        match (a.reml, b.reml) {
            (true, false) => {
                return Err(CompareError::MixedEstimators { reml: Side::First });
            }
            (false, true) => {
                return Err(CompareError::MixedEstimators { reml: Side::Second });
            }
            (true, true)
                if matches!(
                    nested,
                    Nesting::Nested {
                        differs: Differs::Fixed | Differs::Both,
                        ..
                    }
                ) =>
            {
                return Err(CompareError::RemlDifferentFixed);
            }
            _ => {}
        }
    }
    // 5. Something to test.
    let Nesting::Nested { larger, differs } = nested else {
        return Err(CompareError::SameParts);
    };
    if a.df == b.df {
        return Err(CompareError::NothingToTest { m0: a.df, m1: b.df });
    }
    Ok(Comparison {
        m0: ModelInfo::of(a),
        m1: ModelInfo::of(b),
        larger,
        differs,
        test: case(a, b, larger, differs, opts),
    })
}
