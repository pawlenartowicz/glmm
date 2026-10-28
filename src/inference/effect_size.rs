//! Marginal and conditional R² and the intraclass correlation of a mixed
//! model: [`Fit::r2`] and [`Fit::icc`], from one split of the variance into
//! a fixed, a random and a residual part, as `performance::r2_nakagawa` and
//! `performance::icc` compute it (performance 0.18.2 over insight 1.5.4's
//! `compute_variances`).

use std::f64::consts::PI;

use faer::MatRef;

use super::Objective;
use crate::lmm::slope_value;
use crate::spec::set_term_slopes;
use crate::{
    BinomialLink, ColumnId, Data, Design, Family, Fit, FitError, FitOptions, GammaLink, ModelSpec,
    ReTerm, SeFor,
};

/// Marginal and conditional R² of a mixed model ([`Fit::r2`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct R2 {
    /// The share of the fixed effects: `σ²_f / (σ²_f + σ²_r + σ²_e)`.
    pub marginal: f64,
    /// The share of the fixed and the random effects:
    /// `(σ²_f + σ²_r) / (σ²_f + σ²_r + σ²_e)`.
    pub conditional: f64,
}

/// The intraclass correlation of a mixed model, over all its groupings
/// ([`Fit::icc`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Icc {
    /// `σ²_r / (σ²_r + σ²_e)`.
    pub adjusted: f64,
    /// `σ²_r / (σ²_f + σ²_r + σ²_e)`.
    pub unadjusted: f64,
}

/// The three variances R² and the ICC are ratios of, on the link scale.
pub(crate) struct VarianceParts {
    /// `σ²_f`.
    pub(crate) fixed: f64,
    /// `σ²_r`, summed over the groupings.
    pub(crate) random: f64,
    /// `σ²_e`.
    pub(crate) residual: f64,
}

impl VarianceParts {
    pub(crate) const NAN: VarianceParts = VarianceParts {
        fixed: f64::NAN,
        random: f64::NAN,
        residual: f64::NAN,
    };

    pub(crate) fn r2(&self) -> R2 {
        let total = self.fixed + self.random + self.residual;
        R2 {
            marginal: self.fixed / total,
            conditional: (self.fixed + self.random) / total,
        }
    }

    pub(crate) fn icc(&self) -> Icc {
        let total = self.fixed + self.random + self.residual;
        Icc {
            adjusted: self.random / (self.random + self.residual),
            unadjusted: self.random / total,
        }
    }
}

impl Fit {
    /// Marginal and conditional R² of a mixed model (Nakagawa & Schielzeth
    /// 2013; Nakagawa, Johnson & Schielzeth 2017), as
    /// `performance::r2_nakagawa` computes them. `data` and `spec` are the
    /// ones the fit was made with.
    ///
    /// The fixed variance is `var(Xβ̂)` over the rows (denominator `n − 1`,
    /// aliased columns left out, no offset). The random variance is, per
    /// grouping, the mean over the rows of `zᵢᵀ D̂ zᵢ` (Johnson 2014), with
    /// `D̂` the grouping's [`Fit::varcorr`] block and `zᵢ` the row's values of
    /// its columns, summed over the groupings. A grouping with one level per
    /// row (an observation-level random effect) is left out of it: on a GLMM
    /// its variance joins the residual, on a Gaussian model it is dropped, as
    /// insight does. The residual variance is `σ̂²` on a Gaussian model and
    /// the distribution-specific variance on a GLMM,
    /// by the lognormal approximation only: binomial `π²/(3w̄)` (logit),
    /// `1/w̄` (probit), `π²/(6w̄)` (cloglog), `w̄` the mean prior weight (on
    /// trial counts these are performance's values for a `cbind()` response;
    /// a `glmer` fitted with `weights =` gets other values there);
    /// Poisson `ln(1 + 1/μ)`; negative binomial `ln(1 + 1/μ + 1/θ̂)`; Gamma
    /// with the log link `ln(1 + φ̂)`. On Poisson and negative binomial fits
    /// `μ = exp(β̂₀ + ½ Σ D̂₀₀)` comes from a refit of the null model (the
    /// intercept, the same random terms, weights, offset and ids; the sum over
    /// the groupings with a random intercept that are not observation-level),
    /// with the fit's node count and no standard errors.
    ///
    /// Zero random variance gives a conditional R² equal to the marginal one,
    /// singular or not. On a fit that did not converge, or whose null refit
    /// did not, both fields are NaN.
    ///
    /// # Errors
    ///
    /// [`FitError::Unsupported`] on a model without random effects and on a
    /// family and link outside the list above; [`FitError::Shape`] when
    /// `data` has another `n` or `p` than the fit, or `spec` another set of
    /// random-effect blocks; any error of [`crate::fit`] on `data` and
    /// `spec`, and of the null refit.
    pub fn r2(&self, data: &Data<'_>, spec: &ModelSpec) -> Result<R2, FitError> {
        Ok(self.variance_parts(data, spec)?.r2())
    }

    /// The adjusted and the unadjusted intraclass correlation of a mixed
    /// model, over all its groupings (Nakagawa, Johnson & Schielzeth 2017),
    /// as `performance::icc` computes them, from the variances of
    /// [`Fit::r2`] (Johnson 2014 for random slopes; the lognormal
    /// approximation of the distribution-specific variance on a GLMM).
    ///
    /// Zero random variance gives 0, singular or not. On a fit that did not
    /// converge, or whose null refit did not, both fields are NaN.
    ///
    /// # Errors
    ///
    /// As [`Fit::r2`].
    pub fn icc(&self, data: &Data<'_>, spec: &ModelSpec) -> Result<Icc, FitError> {
        Ok(self.variance_parts(data, spec)?.icc())
    }

    /// [`variance_parts`] of this fit, made on `data` and `spec`.
    pub(crate) fn variance_parts(
        &self,
        data: &Data<'_>,
        spec: &ModelSpec,
    ) -> Result<VarianceParts, FitError> {
        variance_parts(
            self.nobs,
            &self.beta,
            &self.varcorr,
            self.aliased(),
            self.converged(),
            self.dispersion,
            self.objective,
            data,
            spec,
        )
    }
}

/// Each grouping's random-effect block, primary first: whether its reported
/// block has the intercept column, and its slope indices.
fn blocks(spec: &ModelSpec) -> Vec<(bool, &[ColumnId])> {
    let has_intercept = |terms: &[ReTerm]| terms.is_empty() || terms.iter().any(|t| t.intercept);
    spec.re
        .iter()
        .flat_map(|re| {
            std::iter::once((has_intercept(&re.terms), re.slopes.as_slice())).chain(
                re.extra_groupings
                    .iter()
                    .map(|g| (has_intercept(&g.terms), g.slopes.as_slice())),
            )
        })
        .collect()
}

/// The fixed, random and residual variances of a fit made on `data` and
/// `spec`, that [`Fit::r2`] and [`Fit::icc`] report the ratios of. The other
/// arguments are the fit's fields of the same names, so a port can pass the
/// ones it stored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn variance_parts(
    nobs: usize,
    beta: &[f64],
    varcorr: &[Vec<f64>],
    aliased: &[bool],
    converged: bool,
    dispersion: f64,
    objective: Objective,
    data: &Data<'_>,
    spec: &ModelSpec,
) -> Result<VarianceParts, FitError> {
    if spec.re.is_none() {
        return Err(FitError::Unsupported(
            "R2 and the ICC need a model with random effects".to_string(),
        ));
    }
    let supported = match spec.family {
        Family::Gaussian
        | Family::Binomial { .. }
        | Family::Poisson { .. }
        | Family::NegativeBinomial { .. } => true,
        Family::Gamma { link } => link == GammaLink::Log,
        Family::InverseGaussian { .. } => false,
    };
    if !supported {
        return Err(FitError::Unsupported(format!(
            "R2 and the ICC have no distribution-specific variance for {:?}",
            spec.family
        )));
    }
    let Design {
        x,
        n,
        p,
        re,
        s,
        weights,
        ..
    } = data.design;
    if n != nobs || p != beta.len() {
        return Err(FitError::Shape(format!(
            "the data have n = {n}, p = {p}; the fit was made on n = {}, p = {}",
            nobs,
            beta.len()
        )));
    }
    let nagq = match objective {
        Objective::Agq { nodes } => nodes,
        _ => 1,
    };
    let opts = FitOptions {
        nagq,
        se_for: SeFor::Only(vec![]),
        ..FitOptions::default()
    };
    crate::fit::validate(data, spec, None, &opts)?;
    if !converged {
        return Ok(VarianceParts::NAN);
    }
    let blocks = blocks(spec);
    let width_ok = |(&(intercept, slopes), vech): (&(bool, &[ColumnId]), &Vec<f64>)| {
        let q = usize::from(intercept) + slopes.len();
        vech.len() == q * (q + 1) / 2
    };
    if blocks.len() != varcorr.len() || !blocks.iter().zip(varcorr).all(width_ok) {
        return Err(FitError::Shape(
            "the spec's random-effect blocks are not the fit's".to_string(),
        ));
    }

    let x_mat = MatRef::from_row_major_slice(x, n, p);
    let z_mat = MatRef::from_row_major_slice(re.unwrap_or(&[]), n, s);
    let eta: Vec<f64> = (0..n)
        .map(|i| {
            (0..p)
                .filter(|&j| !aliased[j])
                .map(|j| x_mat[(i, j)] * beta[j])
                .sum()
        })
        .collect();
    let eta_mean = eta.iter().sum::<f64>() / n as f64;
    let fixed = eta.iter().map(|e| (e - eta_mean).powi(2)).sum::<f64>() / (n - 1) as f64;

    // A grouping with one level per row is an observation-level random
    // effect: insight counts its variance as residual (dispersion) on a GLMM
    // and leaves it out on an LMM, whose σ̂² already holds that variation.
    let ids = data
        .design
        .ids
        .expect("validate requires ids on a mixed model");
    let observation_level: Vec<bool> = std::iter::once(&ids.primary)
        .chain(&ids.extra)
        .map(|g| {
            let mut levels = g.clone();
            levels.sort_unstable();
            levels.dedup();
            levels.len() == n
        })
        .collect();

    // Johnson's (2014) row mean of zᵢᵀ D zᵢ, walking each vech block in its
    // column-major order: an off-diagonal entry counts twice.
    let mut random = 0.0;
    let mut obs_variance = 0.0;
    let mut z = Vec::new();
    for ((&(intercept, slopes), vech), &obs) in blocks.iter().zip(varcorr).zip(&observation_level) {
        let mut sum = 0.0;
        for i in 0..n {
            z.clear();
            if intercept {
                z.push(1.0);
            }
            z.extend(
                slopes
                    .iter()
                    .map(|&c| slope_value(x_mat, z_mat, p, i, c as usize)),
            );
            let mut k = 0;
            for c in 0..z.len() {
                for r in c..z.len() {
                    let times = if r == c { 1.0 } else { 2.0 };
                    sum += times * vech[k] * z[r] * z[c];
                    k += 1;
                }
            }
        }
        if obs {
            obs_variance += sum / n as f64;
        } else {
            random += sum / n as f64;
        }
    }

    let w_bar = weights.map_or(1.0, |w| w.iter().sum::<f64>() / n as f64);
    let null_mu = || -> Result<f64, FitError> {
        // The slopes of the null model read every slope column as a
        // random-effect-only column, since its design is the intercept alone.
        let mut sources: Vec<usize> = Vec::new();
        let mut moved = |cols: &[ColumnId]| -> Vec<ColumnId> {
            cols.iter()
                .map(|&c| {
                    let c = c as usize;
                    let at = sources.iter().position(|&s| s == c).unwrap_or_else(|| {
                        sources.push(c);
                        sources.len() - 1
                    });
                    (1 + at) as ColumnId
                })
                .collect()
        };
        let mut null_spec = spec.clone();
        let null_re = null_spec.re.as_mut().expect("a mixed model");
        null_re.slopes = moved(&null_re.slopes);
        set_term_slopes(&mut null_re.terms, &null_re.slopes);
        for g in &mut null_re.extra_groupings {
            g.slopes = moved(&g.slopes);
            set_term_slopes(&mut g.terms, &g.slopes);
        }
        let null_z: Vec<f64> = (0..n)
            .flat_map(|i| {
                sources
                    .iter()
                    .map(move |&c| slope_value(x_mat, z_mat, p, i, c))
            })
            .collect();
        let ones = vec![1.0; n];
        let mut null_data = Data {
            design: Design {
                x: &ones,
                p: 1,
                re: None,
                s: 0,
                ..data.design
            },
            ..*data
        };
        if !sources.is_empty() {
            null_data = null_data.re_columns(&null_z, sources.len());
        }
        let null = crate::fit(&null_data, &null_spec, &opts)?;
        if !null.converged() {
            return Ok(f64::NAN);
        }
        // insight sums the intercept variances only: the null design has no
        // slope column to weight the others by.
        let d00: f64 = blocks
            .iter()
            .zip(&null.varcorr)
            .zip(&observation_level)
            .filter(|(((intercept, _), _), &obs)| *intercept && !obs)
            .map(|((_, vech), _)| vech[0])
            .sum();
        Ok((null.beta[0] + 0.5 * d00).exp())
    };
    // Poisson and negative binomial are the families that refit the null
    // model; `orchestrate::effect_sizes` names the same two - change together.
    let distribution = match spec.family {
        Family::Gaussian => dispersion,
        Family::Binomial { link } => match link {
            BinomialLink::Logit => PI * PI / (3.0 * w_bar),
            BinomialLink::Probit => 1.0 / w_bar,
            BinomialLink::Cloglog => PI * PI / (6.0 * w_bar),
        },
        Family::Poisson { .. } => (1.0 / null_mu()?).ln_1p(),
        Family::NegativeBinomial { .. } => (1.0 / null_mu()? + 1.0 / dispersion).ln_1p(),
        Family::Gamma { .. } => dispersion.ln_1p(),
        Family::InverseGaussian { .. } => unreachable!("refused above"),
    };
    let residual = match spec.family {
        Family::Gaussian => distribution,
        _ => distribution + obs_variance,
    };
    Ok(VarianceParts {
        fixed,
        random,
        residual,
    })
}

#[cfg(all(test, feature = "formula"))]
#[path = "effect_size_tests.rs"]
mod effect_size_tests;
