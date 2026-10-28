//! Seeded draws of a response from a fitted model on a design
//! ([`Simulator`]), lme4's `simulate()`: new random effects from the fitted
//! covariance, then the family's draw around the conditional mean.

use commonstats::dist::{
    Bernoulli, Binomial, DiscreteSampler, Gamma, InverseGaussian, NegBinomial, Normal, Poisson,
    Sampler,
};
use commonstats::rng::{CommonStatsRng, STREAM_TAG_SIMULATE};
use faer::MatRef;

use crate::family::{clamp_eta, link_inv, PROB_EPS};
use crate::fit::validate_design;
use crate::lmm::re_map::{semidef_lower_chol, PivotFloor};
use crate::lmm::slope_value;
use crate::predict::{predict_parts, reported_columns, FitParts, PredictOptions};
use crate::{
    BinomialLink, Design, Family, Fit, FitError, GammaLink, InverseGaussianLink, ModelSpec,
};

/// `ids` renumbered `0..k` in id order, and `k`. A level's effect is drawn
/// only when a row carries it, so the draw count is at most `n` whatever the
/// id values: sizing by `max(id) + 1` would let one huge id ask for billions
/// of draws and abort the process.
fn dense_ids(ids: &[u32]) -> (Vec<u32>, usize) {
    let mut levels = ids.to_vec();
    levels.sort_unstable();
    levels.dedup();
    let dense = ids
        .iter()
        .map(|id| levels.binary_search(id).expect("an id of `ids`") as u32)
        .collect();
    (dense, levels.len())
}

/// Draws of the response of a fitted model on a design, lme4's
/// `μ = g⁻¹(η)` of a draw: [`link_inv`], with `libm`'s `exp`, `expm1` and
/// `erfc` in place of `f64::exp`, which follows the platform's libm. So a
/// draw has the same last bit on every target. Mirrors `link_inv`, the clamp
/// included — change together.
fn draw_mean(family: Family, eta: f64) -> f64 {
    let eta = clamp_eta(family, eta);
    match family {
        Family::Binomial {
            link: BinomialLink::Logit,
        } => {
            if eta >= 0.0 {
                1.0 / (1.0 + libm::exp(-eta))
            } else {
                let z = libm::exp(eta);
                z / (1.0 + z)
            }
        }
        Family::Binomial {
            link: BinomialLink::Probit,
        } => (0.5 * libm::erfc(-eta * std::f64::consts::FRAC_1_SQRT_2))
            .clamp(PROB_EPS, 1.0 - PROB_EPS),
        Family::Binomial {
            link: BinomialLink::Cloglog,
        } => (-libm::expm1(-libm::exp(eta))).clamp(PROB_EPS, 1.0 - PROB_EPS),
        Family::Poisson { .. }
        | Family::Gamma {
            link: GammaLink::Log,
            ..
        }
        | Family::NegativeBinomial { .. }
        | Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        } => libm::exp(eta),
        // Identity, 1/η and 1/√η: IEEE operations, the same everywhere.
        _ => link_inv(family, eta),
    }
}

/// `simulate(fit)` with its default `use.u = FALSE`: each draw takes new
/// random effects from the fitted covariance ([`Fit::varcorr`]) for every
/// level the design's ids name, forms `η = Xβ̂ + o + Zb`, and draws each row
/// from the family at `μ = g⁻¹(η)` with the fitted dispersion. On an LMM this
/// is the per-group effects plus iid errors; no `Z` matrix is formed.
///
/// A draw is a pure function of `(seed, draw)` and the design: draw `d` reads
/// only its own Philox stream, `commonstats`'
/// `CommonStatsRng::new_tagged(seed, d, STREAM_TAG_SIMULATE)`, so the same
/// pair gives the same response on every target and thread count. The stream
/// is read in a fixed order: per grouping in declaration order, per level
/// `0..=max id` of the design, `q` standard normals `z` and the effects
/// `b = Lz`, `L` the lower Cholesky factor of the grouping's covariance block
/// (a column whose pivot vanishes at round-off is zero, so a variance at the
/// boundary draws zero); then one family draw per row, in row order.
///
/// The weights are read as R's family `simulate` functions read them:
/// binomial trial counts (a draw is `Binomial(w, μ)/w`, a proportion, and a
/// Bernoulli 0/1 without weights); precision weights on the Gaussian
/// (`N(μ, φ̂/w)`), the Gamma (shape `w/φ̂`) and the inverse Gaussian (shape
/// `w/φ̂`); ignored by the Poisson and the negative binomial (size
/// [`Fit::dispersion`]), as `poisson()$simulate` and MASS's are. A
/// coefficient that is NaN (an aliased column, a GLM column the fit could not
/// separate) contributes zero, as in [`crate::predict`]. A row whose mean is
/// outside its family's range (a negative mean through an identity or inverse
/// link) draws NaN, as R's samplers do.
///
/// For a `formula::Lowered` model `m`, `Simulator::new(fit, m.spec(),
/// &m.design())` draws from the rows and trial counts the fit saw.
#[derive(Debug, Clone)]
pub struct Simulator<'a> {
    family: Family,
    design: Design<'a>,
    /// `Xβ̂ + o` per row, a NaN coefficient left out.
    eta_fixed: Vec<f64>,
    groupings: Vec<SimGrouping>,
    /// [`Fit::dispersion`]: `σ̂²`, `φ̂`, the negative-binomial size, or 1.
    dispersion: f64,
}

/// One grouping of a [`Simulator`].
#[derive(Debug, Clone)]
struct SimGrouping {
    /// The design's ids renumbered `0..levels` in id order ([`dense_ids`]).
    ids: Vec<u32>,
    /// The reported columns, as [`Fit::varcorr`] lays them out.
    cols: Vec<Option<usize>>,
    /// The number of distinct ids in the design.
    levels: usize,
    /// Row-major `q × q` lower factor of the grouping's `varcorr` block.
    l: Vec<f64>,
}

impl<'a> Simulator<'a> {
    /// The simulator of `fit`, fitted with `spec`, on the rows `design`: the
    /// training design, or new rows with the fit's `p` columns in the same
    /// order and coding (and the random-effect-only columns, ids, weights and
    /// offset the model reads). Levels the fit did not see draw their effects
    /// like the seen ones.
    ///
    /// # Errors
    ///
    /// - The input checks of [`crate::fit`] on `design` without a response
    ///   ([`FitError::Shape`], [`FitError::Ids`], [`FitError::Weights`],
    ///   [`FitError::Offset`], [`FitError::Unsupported`]); [`FitError::Shape`]
    ///   also when `design` has a `p` other than the fit's or `spec` does not
    ///   match the fit's random effects.
    /// - [`FitError::Weights`]: a binomial weight that is not a whole number
    ///   of trials.
    /// - [`FitError::Unsupported`]: a fit whose dispersion is NaN (one that
    ///   did not converge).
    pub fn new(
        fit: &Fit,
        spec: &ModelSpec,
        design: &Design<'a>,
    ) -> Result<Simulator<'a>, FitError> {
        Self::from_parts(&FitParts::of(fit, spec), spec, design)
    }

    /// Draw `draw` of the response at `seed`, one value per row of the design.
    pub fn simulate(&self, seed: u64, draw: u64) -> Vec<f64> {
        let mut y = vec![0.0; self.design.n];
        self.fill(seed, draw, &mut y);
        y
    }

    /// [`Simulator::simulate`] into the caller's buffer `y`, bit-equal to it.
    ///
    /// # Errors
    ///
    /// [`FitError::Shape`] when `y` does not have one entry per row of the
    /// design.
    pub fn simulate_into(&self, seed: u64, draw: u64, y: &mut [f64]) -> Result<(), FitError> {
        if y.len() != self.design.n {
            return Err(FitError::Shape(format!(
                "the buffer has {} entries, the design {} rows",
                y.len(),
                self.design.n
            )));
        }
        self.fill(seed, draw, y);
        Ok(())
    }

    /// [`Simulator::new`] on a fit's parts. The ports call it with the
    /// numbers they stored.
    pub(crate) fn from_parts(
        parts: &FitParts<'_>,
        spec: &ModelSpec,
        design: &Design<'a>,
    ) -> Result<Simulator<'a>, FitError> {
        if parts.dispersion.is_nan() {
            return Err(FitError::Unsupported(
                "the fit has no dispersion (it did not converge), so there is no \
                 fitted distribution to draw from"
                    .to_string(),
            ));
        }
        validate_design(design, spec, 1, true)?;
        if let (Family::Binomial { .. }, Some(w)) = (parts.family, design.weights) {
            if let Some(i) = w.iter().position(|v| v.fract() != 0.0) {
                return Err(FitError::Weights(format!(
                    "a binomial simulation needs whole trial counts as weights \
                     (row {i} has {})",
                    w[i]
                )));
            }
        }
        let population = PredictOptions {
            population: true,
            ..PredictOptions::default()
        };
        let eta_fixed = predict_parts(parts, spec, design, &population)?.fit;
        let mut groupings = Vec::new();
        if let (Some(re), Some(ids)) = (spec.re.as_ref(), design.ids) {
            let reported = reported_columns(re, ids);
            let mismatch = |what: &str| {
                Err(FitError::Shape(format!(
                    "the model does not match the fit: {what}"
                )))
            };
            if parts.varcorr.len() != reported.len() {
                return mismatch("the number of groupings");
            }
            for (g, ((ids, cols), vech)) in reported.into_iter().zip(parts.varcorr).enumerate() {
                let q = cols.len();
                if vech.len() != q * (q + 1) / 2 {
                    return mismatch(&format!("grouping {g} has another number of terms"));
                }
                // The column-major vech into the row-major lower triangle.
                let mut sig = vec![0.0; q * q];
                let mut t = 0;
                for c in 0..q {
                    for r in c..q {
                        sig[r * q + c] = vech[t];
                        t += 1;
                    }
                }
                let mut l = vec![0.0; q * q];
                semidef_lower_chol(&sig, q, &mut l, PivotFloor::Trace);
                let (ids, levels) = dense_ids(ids);
                groupings.push(SimGrouping {
                    ids,
                    cols,
                    levels,
                    l,
                });
            }
        }
        Ok(Simulator {
            family: parts.family,
            design: *design,
            eta_fixed,
            groupings,
            dispersion: parts.dispersion,
        })
    }

    /// Draw `draw` at `seed` into `y`, `y.len() == n`.
    fn fill(&self, seed: u64, draw: u64, y: &mut [f64]) {
        let mut rng = CommonStatsRng::new_tagged(seed, draw, STREAM_TAG_SIMULATE);
        let normal = Normal::new(0.0, 1.0).expect("the standard normal");
        let Design {
            x,
            n,
            p,
            re,
            s,
            weights,
            ..
        } = self.design;
        let xm = MatRef::from_row_major_slice(x, n, p);
        let zm = MatRef::from_row_major_slice(re.unwrap_or(&[]), n, s);
        y.copy_from_slice(&self.eta_fixed);
        let (mut b, mut z) = (Vec::new(), Vec::new());
        for g in &self.groupings {
            let q = g.cols.len();
            b.clear();
            z.resize(q, 0.0);
            for _ in 0..g.levels {
                for zk in z.iter_mut() {
                    *zk = normal.sample(&mut rng);
                }
                for r in 0..q {
                    b.push((0..=r).map(|k| g.l[r * q + k] * z[k]).sum::<f64>());
                }
            }
            for (i, eta) in y.iter_mut().enumerate() {
                let level = &b[g.ids[i] as usize * q..][..q];
                *eta += g
                    .cols
                    .iter()
                    .zip(level)
                    .map(|(c, bv)| bv * c.map_or(1.0, |c| slope_value(xm, zm, p, i, c)))
                    .sum::<f64>();
            }
        }
        for (i, v) in y.iter_mut().enumerate() {
            let mu = draw_mean(self.family, *v);
            *v = self.draw_row(mu, weights.map(|w| w[i]), &normal, &mut rng);
        }
    }

    /// One family draw at mean `mu` and prior weight `w`; NaN where the
    /// family's distribution is not defined at `mu`.
    fn draw_row(&self, mu: f64, w: Option<f64>, normal: &Normal, rng: &mut CommonStatsRng) -> f64 {
        let phi = self.dispersion;
        let wv = w.unwrap_or(1.0);
        let drawn = match self.family {
            // `Normal::new(μ, sd).sample` computes `μ + sd·Φ⁻¹(u)`, bit for bit.
            Family::Gaussian => Ok(mu + (phi / wv).sqrt() * normal.sample(rng)),
            Family::Binomial { .. } => match w {
                None => Bernoulli::new(mu).map(|d| d.sample(rng) as f64),
                Some(w) => Binomial::new(w as i64, mu).map(|d| d.sample(rng) as f64 / w),
            },
            Family::Poisson { .. } => Poisson::new(mu).map(|d| d.sample(rng) as f64),
            Family::NegativeBinomial { .. } => {
                NegBinomial::from_mean_size(mu, phi).map(|d| d.sample(rng) as f64)
            }
            Family::Gamma { .. } => {
                let shape = wv / phi;
                Gamma::new(shape, shape / mu).map(|d| d.sample(rng))
            }
            Family::InverseGaussian { .. } => {
                InverseGaussian::new(mu, wv / phi).map(|d| d.sample(rng))
            }
        };
        drawn.unwrap_or(f64::NAN)
    }
}

#[cfg(all(test, feature = "formula"))]
#[path = "simulate_tests.rs"]
mod simulate_tests;
