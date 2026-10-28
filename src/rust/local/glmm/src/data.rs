//! `Data` and `Design` — the per-fit data a model is fitted to, borrowed from
//! the caller.
//!
//! A [`Design`] is everything about the rows except the response: the design
//! matrix, the random-effect-only columns, the grouping ids, the prior weights
//! and the offset. A [`Data`] is a
//! `Design` and its response `y`. The fit entry points ([`crate::fit`],
//! [`crate::fit_with_start`]) take a `Data` and check it against the model
//! before fitting.

use crate::GroupIds;

/// The rows of a model without the response: the row-major `n × p` design
/// `x`, and optionally the random-effect-only columns, the grouping ids, the
/// prior weights and the offset.
///
/// Built with [`Design::new`] and the builders below, which borrow every
/// slice: nothing is copied.
#[derive(Debug, Clone, Copy)]
pub struct Design<'a> {
    /// Row-major `n × p`: element `(i, j)` is `x[i * p + j]`.
    pub(crate) x: &'a [f64],
    pub(crate) n: usize,
    pub(crate) p: usize,
    /// Row-major `n × s` random-effect-only columns ([`Design::re_columns`]).
    pub(crate) re: Option<&'a [f64]>,
    pub(crate) s: usize,
    pub(crate) ids: Option<&'a GroupIds>,
    pub(crate) weights: Option<&'a [f64]>,
    pub(crate) offset: Option<&'a [f64]>,
}

impl<'a> Design<'a> {
    /// The design `x`, row-major `n × p` (element `(i, j)` is `x[i * p + j]`),
    /// with no ids, unit weights and no offset.
    pub fn new(x: &'a [f64], n: usize, p: usize) -> Design<'a> {
        Design {
            x,
            n,
            p,
            re: None,
            s: 0,
            ids: None,
            weights: None,
            offset: None,
        }
    }

    /// Columns that a random slope reads and that are not fixed-effect
    /// columns, row-major `n × s` (element `(i, j)` is `z[i * s + j]`), finite:
    /// lme4's `y ~ 1 + (x | g)`, where `x` is in no fixed term. A slope index
    /// `c` names design column `c` of `x` when `c < p` and column `c − p` of
    /// `z` when `p ≤ c < p + s`, and every column of `z` must be read by at
    /// least one slope. They carry no coefficient: `beta`, `se` and `vcov`
    /// stay `p` wide.
    pub fn re_columns(mut self, z: &'a [f64], s: usize) -> Design<'a> {
        self.re = Some(z);
        self.s = s;
        self
    }

    /// The per-row level ids of the model's groupings. A model with random
    /// effects needs them; a model without random effects must not have them
    /// (empty ids, `GroupIds::default()`, count as none).
    pub fn ids(mut self, ids: &'a GroupIds) -> Design<'a> {
        self.ids = Some(ids);
        self
    }

    /// Per-row prior weights, length `n`, finite and `> 0`. See
    /// [`Data::weights`] for what they mean on each family.
    pub fn weights(mut self, w: &'a [f64]) -> Design<'a> {
        self.weights = Some(w);
        self
    }

    /// Per-row offset on the linear-predictor scale, length `n`, finite. See
    /// [`Data::offset`].
    pub fn offset(mut self, o: &'a [f64]) -> Design<'a> {
        self.offset = Some(o);
        self
    }
}

/// The data a model is fitted to: a [`Design`] and the response `y`, length
/// `n`.
///
/// ```rust
/// use glmm::{Data, GroupIds};
///
/// let x = vec![1.0; 4]; // n = 4, p = 1: an intercept column
/// let y = vec![0.5, 1.5, 1.0, 2.0];
/// let ids = GroupIds::new(vec![0, 0, 1, 1]);
/// let w = vec![1.0, 2.0, 1.0, 2.0];
/// let data = Data::new(&x, &y, 4, 1).ids(&ids).weights(&w);
/// # let _ = data;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Data<'a> {
    pub(crate) design: Design<'a>,
    pub(crate) y: &'a [f64],
}

impl<'a> Data<'a> {
    /// The design `x`, row-major `n × p`, and the response `y`, length `n`,
    /// with no ids, unit weights and no offset.
    pub fn new(x: &'a [f64], y: &'a [f64], n: usize, p: usize) -> Data<'a> {
        Data {
            design: Design::new(x, n, p),
            y,
        }
    }

    /// The random-effect-only columns, row-major `n × s`
    /// ([`Design::re_columns`]).
    pub fn re_columns(mut self, z: &'a [f64], s: usize) -> Data<'a> {
        self.design = self.design.re_columns(z, s);
        self
    }

    /// The per-row level ids of the model's groupings ([`Design::ids`]).
    pub fn ids(mut self, ids: &'a GroupIds) -> Data<'a> {
        self.design = self.design.ids(ids);
        self
    }

    /// Per-row prior weights `wᵢ`, length `n`, finite and `> 0` — lme4's
    /// `weights=`. Without them every row has unit weight.
    /// For an aggregated binomial, `y` is the success PROPORTION and `wᵢ` the
    /// trial count: this is exactly lme4's `cbind(s, m−s)` objective, whose
    /// deviance differs from the expanded-Bernoulli one only by a data-only
    /// saturated constant (same argmin — same β/SE/varcomp).
    ///
    /// On every family with an estimated dispersion (Gaussian, Gamma,
    /// inverse-Gaussian) `wᵢ` is a PRECISION weight: row `i` has variance
    /// `φ·V(μᵢ)/wᵢ` (`lm`/`glm`'s and lme4's convention, McCullagh & Nelder
    /// 1989 §2.2), not glmmTMB's `wᵢ` multiplying the log-density. Rescaling
    /// every weight by one constant `c` changes nothing about the fitted mean
    /// model or the SEs, and moves `dispersion` by exactly `c` — the fit
    /// depends on the RELATIVE weights only (the GLM IRLS's own stopping
    /// tolerance, `glm::DEVIANCE_TOL`, is a relative rule on the deviance, so
    /// it is scale-free too for `|deviance| ≫ 0.1`). Moment estimates (σ̂²,
    /// inverse-Gaussian and Gamma-GLM's Pearson φ̂) divide by the raw row count
    /// `n−p`, never `Σwᵢ−p`. Binomial weights are trial counts (no dispersion
    /// to divide); Poisson's fit is identical either way (the two conventions
    /// differ by a weight-only constant); NB's weights keep multiplying the
    /// log-likelihood (`MASS::glm.nb`'s convention — NB's variance `μ+μ²/θ`
    /// has no `φ` to divide).
    ///
    /// Support matrix: every (family, RE structure, solver) combination,
    /// including AGQ (`nagq > 1`) on the binomial/Poisson/NB/Gamma shapes it covers — the
    /// per-row `dev_resid` sums in `glmm/agq.rs` carry `wᵢ`, and PIRLS folds the
    /// weights into the conditional mode/curvature (aggregated binomial with small
    /// clusters, e.g. `glmer(cbind(s,m−s) ~ …, nAGQ=k)`, is the canonical case).
    ///
    /// - Gaussian fixed-only: WLS via √wᵢ row pre-scaling, `σ̂² = Σwᵢrᵢ²/(n−p)`;
    ///   matches R `lm(weights=)` (`fit_ols_weighted_matches_r_lm`).
    /// - GLM fixed-only, any family (Binomial/Poisson/Gamma/NB): weighted IRLS —
    ///   `wᵢ` multiplies the working weight and deviance; both Gamma's and an
    ///   inverse-Gaussian's Pearson φ̂ are `Σwᵢrᵢ²/(n−p)`, `summary(glm)`'s value;
    ///   Gamma's separate maximum-likelihood φ̂ (used only for `loglik`) takes
    ///   `ŵᵢ = wᵢ/s` in the shape equation (precision weights: this Newton
    ///   search is not scale-free on raw `w`); the null
    ///   deviance uses the weighted mean. Matches R
    ///   `glm(weights=)` (`fit_glm_gamma_weighted_matches_r`,
    ///   `fit_glm_binomial_weighted_aggregated_matches_r`,
    ///   `glm_weighted_deviance_null_golden_value`) and `MASS::glm.nb(weights=)`
    ///   (`fit_glm_nb_weighted_matches_mass`, weighted θ profile).
    /// - Sparse (`Solver::Sparse`) non-Gaussian GLMM, every family: `wᵢ` weights
    ///   the sparse PIRLS working weight/deviance/score identically to the
    ///   dense mixed path below; Gamma's dispersion term
    ///   (`family::gamma_dispersion_term`) takes `ŵᵢ = wᵢ/s` (`s` the
    ///   weight-scale normaliser, `family::weight_scale`) in place of a row
    ///   count, its PIRLS runs on `ŵᵢ/φ`, and NB's marginal-θ profile
    ///   (`nb_profile_loglik`) takes the weights raw. Validated against
    ///   `fit_sparse_gamma_glmm_weighted_matches_glmmtmb` (glmmTMB), the
    ///   `sparse_weighted_binomial_*` expanded-vs-aggregated tests, and the
    ///   `sparse_weighted_{poisson,nb}_matches_replicated` weighted-vs-
    ///   replicated-row equivalence tests.
    /// - Dense (`Solver::NoZ`) mixed GLMM, every non-Gaussian family
    ///   (Binomial/Poisson/Gamma/NB): `wᵢ` weights the working weight,
    ///   deviance, and β-gradient score identically to the sparse binomial path
    ///   above; Gamma's dispersion term (`family::gamma_dispersion_term`) takes
    ///   `ŵᵢ = wᵢ/s` in place of a row count. Validated against
    ///   `fit_glmm_cbpp_aggregated_matches_lme4`,
    ///   `fit_glmm_poisson_weighted_matches_lme4`, and
    ///   `fit_glmm_gamma_weighted_matches_glmmtmb` (lme4 `glmer` for the
    ///   first two, glmmTMB with a dispersion offset for the Gamma one; all dense).
    /// - Dense (`Solver::NoZ`) mixed Gaussian LMM (weighted REML or ML): every
    ///   row of `[X y Z]` is conceptually √wᵢ-scaled before hitting the unit-
    ///   weight suff-stats/deviance kernel (`LmmSuffStats::add_rows_multi`),
    ///   which folds `wᵢ` straight into the Gram accumulators (`c`, `s`,
    ///   `counts`: the last is `Σ z²·wᵢ`, not a row count; `ν`, `n − p`
    ///   under REML and `n` under ML, still comes from the raw row count).
    ///   `σ̂² = pwrss_w/ν` (raw ν) falls out of the weighted Grams
    ///   automatically; `tau2 = θ²·σ̂²` and SEs inherit.
    ///   The reported deviance carries an extra `−Σlog wᵢ` (the weighted
    ///   Gaussian log-density's `+½Σlog wᵢ` per row, on the −2ℓ scale) atop the
    ///   usual stripped-constant convention, on both criteria, matching lme4's
    ///   weighted `lmer(weights=)` REMLcrit (ML: `deviance()`) up to that same
    ///   stripped constant.
    ///   Validated against `fit_lmm_weighted_matches_lme4` (lme4 `lmer`, dense,
    ///   REML), `fit_lmm_weight_scale_invariant_dense_and_sparse` (both
    ///   criteria), and
    ///   `fit_lmm_weighted_boundary_matches_wls`.
    /// - Sparse (`Solver::Sparse`) Gaussian LMM: identical √wᵢ scaling threaded
    ///   through the sparse-Z accumulator instead — `for_each_z_entry`'s
    ///   emitted z values and every raw x/y read in `SparseLmmWorkspace::new`'s
    ///   pass 2 each carry one `√wᵢ` factor, so every Gram product (`ztxy`,
    ///   `cxy`, the packed `pk_*` raw-Gram streams) ends up carrying exactly
    ///   `wᵢ`. Deviance/ν stay raw (`n − p` or `n`); the `−Σlog wᵢ` constant is added
    ///   in `lmm_view_to_fit`, the one site both kernels reach. Validated against
    ///   `fit_sparse_lmm_weighted_matches_lme4` (lme4 `lmer`, sparse) and
    ///   `fit_lmm_weight_scale_invariant_dense_and_sparse`.
    pub fn weights(mut self, w: &'a [f64]) -> Data<'a> {
        self.design = self.design.weights(w);
        self
    }

    /// Per-row additive offset `oᵢ` on the linear-predictor scale, length `n`,
    /// finite — R's `offset=`: `η = o + Xβ (+ Zb)`, every family, every solver
    /// path. A fixed known contribution, NOT a parameter: it adds no column to
    /// `X` and never appears in `beta`/`se`/`vcov`/`df`. The canonical use is a
    /// Poisson exposure, `offset = ln(exposure)`. Gaussian identity-link paths
    /// (OLS/LMM) implement it as the exact `y − o` shift before the
    /// sufficient-statistics accumulation; `Fit::fitted` still reports means
    /// on the original `y` scale (μ̂ = o + Xβ̂ + Zb̂ through the link). Oracle:
    /// `glm(offset=)` / `glmer(offset=)`.
    pub fn offset(mut self, o: &'a [f64]) -> Data<'a> {
        self.design = self.design.offset(o);
        self
    }
}
