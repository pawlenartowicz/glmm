//! Predictions of a fitted model on new rows ([`predict`]): the linear
//! predictor at the fitted fixed effects and, on a mixed model, the
//! conditional modes of the levels the fit saw, on the link or the response
//! scale, with standard errors, intervals and, on a GLMM, the quantiles of a
//! future observation.

use std::f64::consts::{PI, SQRT_2};

use commonstats::dist::{Binomial, ContinuousCdf, DiscreteCdf, Gamma, NegBinomial, Poisson};
use faer::MatRef;

use crate::consts::{GH_NODES, GH_OFFSETS, GH_WEIGHTS};
use crate::family::{link_inv, mu_eta};
use crate::fit::validate_design;
use crate::fit::CondvarCross;
use crate::inference::tails::normal_quantile;
use crate::lmm::slope_value;
use crate::{
    Design, Family, Fit, FitError, GroupIds, GroupingRelation, ModelSpec, Note, ReStructure,
};

/// The scale [`predict`] reports on, lme4's `type`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Scale {
    /// The linear predictor `η` (lme4's `type = "link"`), the default.
    #[default]
    Link,
    /// The mean `μ = g⁻¹(η)` (lme4's `type = "response"`).
    Response,
}

/// What [`predict`] computes. `#[non_exhaustive]`: start from
/// `PredictOptions::default()` and set the fields you need.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PredictOptions {
    /// Fixed effects only, lme4's `re.form = NA`, when set; otherwise
    /// (the default) conditional on every grouping's conditional modes,
    /// lme4's `re.form = NULL`. No effect on a model without random effects.
    pub population: bool,
    /// The scale of [`Prediction::fit`], [`Prediction::se`] and the interval.
    /// Default [`Scale::Link`].
    pub scale: Scale,
    /// Whether a grouping level the fit has no conditional mode for is
    /// accepted (the default): it contributes a zero random effect, and the
    /// prediction carries a [`Note::NewGroupingLevels`]. When unset such a
    /// level is a [`FitError::Ids`], as lme4's default
    /// `allow.new.levels = FALSE` errors.
    pub allow_new_levels: bool,
    /// Whether to report the standard error and the interval at `level`.
    /// Default off.
    pub se: bool,
    /// The coverage of the interval and of the future-observation bounds, in
    /// `(0, 1)`. Default 0.95.
    pub level: f64,
    /// Whether to report the bounds of a future observation, on a
    /// non-Gaussian mixed model only. Default off.
    pub future: bool,
}

impl Default for PredictOptions {
    fn default() -> Self {
        PredictOptions {
            population: false,
            scale: Scale::Link,
            allow_new_levels: true,
            se: false,
            level: 0.95,
            future: false,
        }
    }
}

/// The result of [`predict`], one entry per new row in each requested part;
/// a part that was not requested is empty.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Prediction {
    /// The prediction on [`PredictOptions::scale`].
    pub fit: Vec<f64>,
    /// Its standard error on the same scale, when [`PredictOptions::se`] is
    /// set; on the response scale by the delta method, `|dμ/dη|` times the
    /// link-scale one.
    pub se: Vec<f64>,
    /// The lower end of the interval at [`PredictOptions::level`], when `se` is
    /// set: `η̂ − z·se` on the link scale, `z` the standard normal quantile of
    /// `(1 + level)/2`, and on the response scale the inverse link of the
    /// link-scale interval, its ends ordered.
    pub lower: Vec<f64>,
    /// The upper end of the same interval.
    pub upper: Vec<f64>,
    /// The `(1 − level)/2` quantile of a future observation on the response
    /// scale, whatever [`PredictOptions::scale`] is, when
    /// [`PredictOptions::future`] is set; a binomial bound is a proportion of
    /// the row's trials.
    pub future_lower: Vec<f64>,
    /// The `(1 + level)/2` quantile of a future observation.
    pub future_upper: Vec<f64>,
    /// One [`Note::NewGroupingLevels`] per grouping with unseen levels, in
    /// declaration order; empty in population mode.
    pub notes: Vec<Note>,
}

/// Predict a fitted model on new rows `new`, lme4's `predict(newdata = )`.
///
/// `spec` is the model `fit` was fitted with, and `new` has the fit's `p`
/// design columns in the same order and coding, its random-effect-only
/// columns ([`Design::re_columns`]) when the model has some, and the offset
/// when the model had one; its weights are read only by the
/// future-observation bounds, as binomial trial counts or Gamma precision
/// weights (default 1). The prediction is
/// `η = Xβ̂ + o` in population mode and `η = Xβ̂ + o + Zb̂` otherwise, `b̂` the
/// conditional modes ([`Fit::ranef`]), through the inverse link on the
/// response scale. In conditional mode the rows need [`Design::ids`]: an id at
/// or past [`Fit::ranef_levels`] for its grouping is a level the fit did not
/// see, and gives that grouping a zero random effect (see
/// [`PredictOptions::allow_new_levels`]). A nested grouping's id is the global
/// child id, as in [`crate::GroupIds`].
///
/// A coefficient that is NaN contributes zero: an aliased column's, and a
/// GLM column the fit could not separate ([`Note::SeUnavailable`]). There
/// [`Fit::fitted`] keeps that column at its start value instead, so on the
/// training rows the two differ when that start was not zero, as after
/// [`crate::fit_with_start`].
///
/// The link-scale standard error is lme4 2.0.6's `predict(se.fit = TRUE)`
/// form, from the joint covariance of `(b̂, β̂)` at the fitted variance
/// parameters: `se² = zᵀCz + (x − Kᵀz)ᵀV(x − Kᵀz)`, with `C` the
/// conditional covariance of the row's random effects (the
/// [`Fit::condvar`] blocks of its levels and the covariance between its levels
/// in different groupings), `K` the matching rows of
/// `Λ̂(Λ̂ᵀZᵀWZΛ̂ + I)⁻¹Λ̂ᵀZᵀWX` (so `Cov(b̂_l, β̂) = −K_l·V`), and `V` =
/// [`Fit::vcov`] over the columns with a coefficient; in population mode
/// `se² = xᵀVx`. It differs from lme4's on a GLMM, where `V` is the fit's
/// [`crate::WaldSe`] covariance and lme4's the RX one. A fit whose `vcov` is NaN (not converged,
/// [`Note::SeUnavailable`]) gives a NaN standard error. An unseen level adds
/// nothing to the variance, as in lme4. Neither form carries the uncertainty
/// of the variance parameters, which glmmTMB's `se.fit` does.
///
/// The future-observation bounds are the quantiles of the family's
/// distribution mixed over `η ~ N(η̂, se²)` (the row's link-scale standard
/// error in the chosen mode) by 21-node Gauss-Hermite quadrature, with the
/// fitted dispersion: the negative-binomial size [`Fit::dispersion`], the Gamma
/// shape `w/φ̂`. On a discrete family a bound is the smallest count whose
/// mixture CDF reaches the probability.
///
/// # Errors
///
/// - [`FitError::OptionValue`]: `level` outside `(0, 1)`; `future` on a model
///   that is not a non-Gaussian mixed model.
/// - The input checks of [`crate::fit`] on `new` without a response
///   ([`FitError::Shape`], [`FitError::Ids`], [`FitError::Weights`],
///   [`FitError::Offset`], [`FitError::Unsupported`]), the ids required in
///   conditional mode only; [`FitError::Shape`] also when `new` has a `p`
///   other than the fit's or `spec` does not match the fit's random effects.
/// - [`FitError::Ids`]: an unseen level with `allow_new_levels` unset.
/// - [`FitError::Unsupported`]: conditional mode on a mixed fit with no
///   conditional modes (one that did not converge); a conditional standard
///   error or future bound where [`Fit::condvar`] is empty.
/// - [`FitError::Weights`]: binomial future bounds with a weight that is not
///   a whole number of trials.
pub fn predict(
    fit: &Fit,
    spec: &ModelSpec,
    new: &Design<'_>,
    opts: &PredictOptions,
) -> Result<Prediction, FitError> {
    predict_parts(&FitParts::of(fit, spec), spec, new, opts)
}

/// The numbers of a fit that [`predict`] and the simulator read, borrowed:
/// from a [`Fit`] by [`FitParts::of`], or from the numbers a port stored.
pub(crate) struct FitParts<'a> {
    pub(crate) family: Family,
    pub(crate) beta: &'a [f64],
    pub(crate) vcov: &'a [Vec<f64>],
    pub(crate) varcorr: &'a [Vec<f64>],
    pub(crate) ranef: &'a [f64],
    pub(crate) ranef_levels: &'a [usize],
    pub(crate) condvar: &'a [f64],
    pub(crate) condvar_x: &'a [f64],
    pub(crate) condvar_cross: Option<&'a CondvarCross>,
    pub(crate) dispersion: f64,
}

impl<'a> FitParts<'a> {
    /// The parts of `fit`, fitted with `spec`.
    pub(crate) fn of(fit: &'a Fit, spec: &ModelSpec) -> FitParts<'a> {
        FitParts {
            family: spec.family,
            beta: &fit.beta,
            vcov: &fit.vcov,
            varcorr: &fit.varcorr,
            ranef: &fit.ranef,
            ranef_levels: &fit.ranef_levels,
            condvar: &fit.condvar,
            condvar_x: &fit.condvar_x,
            condvar_cross: fit.condvar_cross.as_ref(),
            dispersion: fit.dispersion,
        }
    }
}

/// One grouping as [`Fit::ranef`] lays it out.
struct Block<'a> {
    /// The new rows' level ids.
    ids: &'a [u32],
    /// The reported columns in order: `None` the intercept, `Some(c)` slope
    /// index `c`.
    cols: Vec<Option<usize>>,
    levels: usize,
    /// Starts of the grouping's block in `ranef`, `condvar` and `condvar_x`.
    ranef: usize,
    condvar: usize,
    condvar_x: usize,
}

/// Each grouping of `re` in declaration order: the rows' level ids and the
/// columns [`Fit::ranef`] and [`Fit::varcorr`] report for it, `None` the
/// intercept and `Some(c)` slope index `c`. A block without an intercept term
/// reports its slopes only.
pub(crate) fn reported_columns<'a>(
    re: &ReStructure,
    ids: &'a GroupIds,
) -> Vec<(&'a [u32], Vec<Option<usize>>)> {
    std::iter::once((&re.slopes, &re.terms, &ids.primary[..]))
        .chain(
            re.extra_groupings
                .iter()
                .zip(&ids.extra)
                .map(|(g, e)| (&g.slopes, &g.terms, &e[..])),
        )
        .map(|(slopes, terms, ids)| {
            let intercept = terms.is_empty() || terms.iter().any(|t| t.intercept);
            let cols = intercept
                .then_some(None)
                .into_iter()
                .chain(slopes.iter().map(|&c| Some(c as usize)))
                .collect();
            (ids, cols)
        })
        .collect()
}

/// The groupings of `spec` against the fit's layout, declaration order.
fn blocks<'a>(
    parts: &FitParts<'_>,
    spec: &ModelSpec,
    new: &Design<'a>,
) -> Result<Vec<Block<'a>>, FitError> {
    let (Some(re), Some(ids)) = (spec.re.as_ref(), new.ids) else {
        return Ok(Vec::new());
    };
    let p = parts.beta.len();
    let mismatch = |what: &str| {
        Err(FitError::Shape(format!(
            "the model does not match the fit: {what}"
        )))
    };
    if parts.ranef_levels.len() != 1 + re.extra_groupings.len()
        || parts.varcorr.len() != parts.ranef_levels.len()
    {
        return mismatch("the number of groupings");
    }
    let (mut at_b, mut at_c, mut at_x) = (0, 0, 0);
    let mut out = Vec::new();
    for (g, (ids, cols)) in reported_columns(re, ids).into_iter().enumerate() {
        let (q, levels) = (cols.len(), parts.ranef_levels[g]);
        if parts.varcorr[g].len() != q * (q + 1) / 2 {
            return mismatch(&format!("grouping {g} has another number of terms"));
        }
        out.push(Block {
            ids,
            cols,
            levels,
            ranef: at_b,
            condvar: at_c,
            condvar_x: at_x,
        });
        at_b += levels * q;
        at_c += levels * q * (q + 1) / 2;
        at_x += levels * q * p;
    }
    if at_b != parts.ranef.len()
        || !(parts.condvar.is_empty() || parts.condvar.len() == at_c)
        || parts.condvar_x.len() != if parts.condvar.is_empty() { 0 } else { at_x }
    {
        return mismatch("the random-effect layout");
    }
    Ok(out)
}

/// [`predict`] on a fit's parts. The ports call it with the numbers they
/// stored.
pub(crate) fn predict_parts(
    parts: &FitParts<'_>,
    spec: &ModelSpec,
    new: &Design<'_>,
    opts: &PredictOptions,
) -> Result<Prediction, FitError> {
    let mixed = spec.re.is_some();
    let conditional = mixed && !opts.population;
    if !(opts.level > 0.0 && opts.level < 1.0) {
        return Err(FitError::OptionValue(format!(
            "PredictOptions.level must be in (0, 1), got {}",
            opts.level
        )));
    }
    if opts.future && !(mixed && spec.family != Family::Gaussian) {
        return Err(FitError::OptionValue(
            "PredictOptions.future applies only to a non-Gaussian mixed model".to_string(),
        ));
    }
    validate_design(new, spec, 1, conditional)?;
    let &Design {
        x,
        n,
        p,
        re,
        s,
        weights,
        offset,
        ..
    } = new;
    if p != parts.beta.len() {
        return Err(FitError::Shape(format!(
            "the design has p={p} columns, the fit {}",
            parts.beta.len()
        )));
    }
    let need_se = opts.se || opts.future;
    let blocks = if conditional {
        if parts.ranef.is_empty() {
            return Err(FitError::Unsupported(
                "the fit has no conditional modes (it did not converge); \
                 predict in population mode"
                    .to_string(),
            ));
        }
        blocks(parts, spec, new)?
    } else {
        Vec::new()
    };
    if conditional && need_se && parts.condvar.is_empty() {
        return Err(FitError::Unsupported(
            "this fit has no conditional variances (it was fitted with condvar off, or \
             its layout does not form them), so a conditional standard error is not \
             available; refit with condvar on, or predict in population mode"
                .to_string(),
        ));
    }
    let cross = match (
        conditional && need_se,
        parts.condvar_cross,
        spec.re.as_ref(),
    ) {
        (true, Some(c), Some(re)) => Some(Cross::new(c, re, &blocks)?),
        _ => None,
    };
    let family = parts.family;
    if opts.future && matches!(family, Family::Binomial { .. }) {
        if let Some(i) = weights.and_then(|w| w.iter().position(|v| v.fract() != 0.0)) {
            return Err(FitError::Weights(format!(
                "binomial future bounds need whole trial counts as weights (row {i})"
            )));
        }
    }
    let mut unseen: Vec<Vec<u32>> = vec![Vec::new(); blocks.len()];
    for i in 0..n {
        for (g, b) in blocks.iter().enumerate() {
            let id = b.ids[i] as usize;
            if id < b.levels {
                continue;
            }
            if !opts.allow_new_levels {
                return Err(FitError::Ids(format!(
                    "grouping {g} has a level the fit did not see at row {i} (id {id}, \
                     the fit has {} levels); set allow_new_levels to predict it with a \
                     zero random effect",
                    b.levels
                )));
            }
            unseen[g].push(i as u32);
        }
    }

    let xm = MatRef::from_row_major_slice(x, n, p);
    let zm = MatRef::from_row_major_slice(re.unwrap_or(&[]), n, s);
    let zq = normal_quantile(0.5 + opts.level / 2.0);
    let response = opts.scale == Scale::Response;
    let mut out = Prediction {
        fit: Vec::with_capacity(n),
        se: Vec::new(),
        lower: Vec::new(),
        upper: Vec::new(),
        future_lower: Vec::new(),
        future_upper: Vec::new(),
        notes: Vec::new(),
    };
    // Per row: `d = x − Σ K_lᵀz_l`, the row's `z` over its seen levels, and
    // each seen level as (grouping, level, start in `z`).
    let mut d = vec![0.0f64; p];
    let mut z = Vec::new();
    let mut touched = Vec::new();
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        let xb: f64 = row
            .iter()
            .zip(parts.beta)
            .filter(|(_, b)| !b.is_nan())
            .map(|(v, b)| v * b)
            .sum();
        let mut eta = xb + offset.map_or(0.0, |o| o[i]);
        d.copy_from_slice(row);
        let mut var_b = 0.0f64;
        z.clear();
        touched.clear();
        for (g, b) in blocks.iter().enumerate() {
            let level = b.ids[i] as usize;
            if level >= b.levels {
                continue;
            }
            let q = b.cols.len();
            let at = z.len();
            touched.push((g, level, at));
            z.extend(
                b.cols
                    .iter()
                    .map(|c| c.map_or(1.0, |c| slope_value(xm, zm, p, i, c))),
            );
            let z = &z[at..];
            let modes = &parts.ranef[b.ranef + level * q..b.ranef + (level + 1) * q];
            eta += modes.iter().zip(z).map(|(m, v)| m * v).sum::<f64>();
            if !need_se {
                continue;
            }
            // `condvar` holds the level's block as a column-major vech.
            let v = q * (q + 1) / 2;
            let c = &parts.condvar[b.condvar + level * v..b.condvar + (level + 1) * v];
            let mut t = 0;
            for col in 0..q {
                for r in col..q {
                    let both = if r == col { 1.0 } else { 2.0 };
                    var_b += both * c[t] * z[r] * z[col];
                    t += 1;
                }
            }
            let k =
                &parts.condvar_x[b.condvar_x + level * q * p..b.condvar_x + (level + 1) * q * p];
            for (r, &zr) in z.iter().enumerate() {
                for (dj, kj) in d.iter_mut().zip(&k[r * p..(r + 1) * p]) {
                    *dj -= kj * zr;
                }
            }
        }
        if let Some(cross) = &cross {
            var_b += cross.variance(&blocks, &touched, &z);
        }
        let se_link = need_se.then(|| {
            let mut quad = 0.0f64;
            for (j, &dj) in d.iter().enumerate() {
                if parts.beta[j].is_nan() {
                    continue;
                }
                for (k, &dk) in d.iter().enumerate() {
                    if !parts.beta[k].is_nan() {
                        quad += dj * parts.vcov[j][k] * dk;
                    }
                }
            }
            // Round-off can take a variance near zero below it; NaN stays NaN.
            let var = var_b + quad;
            if var < 0.0 {
                0.0
            } else {
                var.sqrt()
            }
        });
        if opts.future {
            let se_link = se_link.expect("formed whenever future is set");
            let w = weights.map_or(1.0, |w| w[i]);
            let (lo, hi) = future_bounds(family, eta, se_link, w, parts.dispersion, opts.level);
            out.future_lower.push(lo);
            out.future_upper.push(hi);
        }
        if let (true, Some(se_link)) = (opts.se, se_link) {
            let (a, b) = (eta - zq * se_link, eta + zq * se_link);
            if response {
                let (a, b) = (link_inv(family, a), link_inv(family, b));
                // A decreasing inverse link (Gamma's inverse) swaps the ends.
                let (a, b) = if a <= b { (a, b) } else { (b, a) };
                out.se.push(mu_eta(family, eta).abs() * se_link);
                out.lower.push(a);
                out.upper.push(b);
            } else {
                out.se.push(se_link);
                out.lower.push(a);
                out.upper.push(b);
            }
        }
        out.fit
            .push(if response { link_inv(family, eta) } else { eta });
    }
    out.notes = unseen
        .into_iter()
        .enumerate()
        .filter(|(_, rows)| !rows.is_empty())
        .map(|(g, rows)| Note::NewGroupingLevels {
            grouping: g as u32,
            rows,
        })
        .collect();
    Ok(out)
}

/// Where a grouping's levels sit in a [`CondvarCross`].
#[derive(Clone, Copy)]
enum Place {
    Family,
    Nested,
    /// The index of the grouping's level 0 among the tail levels.
    Tail(usize),
}

/// A fit's [`CondvarCross`] with each grouping's place in it.
struct Cross<'a> {
    c: &'a CondvarCross,
    places: Vec<Place>,
    /// The family grouping's reported coefficients, child slots per family,
    /// coefficients per family and tail levels.
    q: usize,
    m: usize,
    w: usize,
    t: usize,
}

impl<'a> Cross<'a> {
    /// Places `blocks`' groupings in `c`, checking its layout against them.
    fn new(c: &'a CondvarCross, re: &ReStructure, blocks: &[Block<'_>]) -> Result<Self, FitError> {
        let mismatch = || {
            Err(FitError::Shape(
                "the model does not match the fit: the conditional covariance across groupings"
                    .to_string(),
            ))
        };
        let fg = c.family_grouping;
        if fg >= blocks.len() {
            return mismatch();
        }
        // The kernel's slot `j` holds grouping `j`, with 0 and `fg` exchanged.
        let mut places = vec![Place::Family; blocks.len()];
        let (mut t, mut nested) = (0, None);
        for j in 1..blocks.len() {
            let g = if j == fg { 0 } else { j };
            if blocks[g].cols.len() != 1 {
                return mismatch();
            }
            if g > 0 && re.extra_groupings[g - 1].relation == GroupingRelation::NestedWithin {
                places[g] = Place::Nested;
                nested = Some(g);
            } else {
                places[g] = Place::Tail(t);
                t += blocks[g].levels;
            }
        }
        let families = blocks[fg].levels;
        let m = nested.map_or(0, |g| blocks[g].levels / families.max(1));
        let q = blocks[fg].cols.len();
        let w = q + m;
        if (nested.is_some() && m == 0)
            || c.family.len() != families * w * (w + 1) / 2
            || c.tail.len() != t * (t + 1) / 2
            || c.coupling_at.len() != families + 1
            || c.coupling_at.last().map(|&e| e as usize) != Some(c.coupling_cols.len())
            || c.coupling.len() != c.coupling_cols.len() * w
            || c.coupling_cols.iter().any(|&b| b as usize >= t)
        {
            return mismatch();
        }
        Ok(Cross {
            c,
            places,
            q,
            m,
            w,
            t,
        })
    }

    /// The part of a row's `zᵀCz` between its levels in different groupings:
    /// `touched` holds each seen level as (grouping, level, start of its
    /// values in `z`). The products are the ones [`CondvarCross`] documents.
    fn variance(&self, blocks: &[Block<'_>], touched: &[(usize, usize, usize)], z: &[f64]) -> f64 {
        let c = self.c;
        let tail = |a: usize, b: usize| c.tail[vech_at(self.t, a.max(b), a.min(b))];
        // The row's family coefficients as (family, first local index,
        // values), at most a level and a child; its tail levels as (index,
        // value).
        let mut fams: Vec<(usize, usize, &[f64])> = Vec::new();
        let mut tails: Vec<(usize, f64)> = Vec::new();
        for &(g, level, at) in touched {
            let zl = &z[at..at + blocks[g].cols.len()];
            match self.places[g] {
                Place::Family => fams.push((level, 0, zl)),
                Place::Nested => fams.push((level / self.m, self.q + level % self.m, zl)),
                Place::Tail(s) => tails.push((s + level, zl[0])),
            }
        }
        // Per family part, `Σ_a z_a·H_f[e][a]` on each of the family's entries.
        let us: Vec<Vec<(usize, f64)>> = fams
            .iter()
            .map(|&(f, l0, zl)| {
                (c.coupling_at[f] as usize..c.coupling_at[f + 1] as usize)
                    .map(|e| {
                        let h = &c.coupling[e * self.w + l0..e * self.w + l0 + zl.len()];
                        let u = h.iter().zip(zl).map(|(h, z)| h * z).sum();
                        (c.coupling_cols[e] as usize, u)
                    })
                    .collect()
            })
            .collect();
        let vw = self.w * (self.w + 1) / 2;
        let mut v = 0.0f64;
        for (i, &(fi, li, zi)) in fams.iter().enumerate() {
            for (j, &(fj, lj, zj)) in fams.iter().enumerate().skip(i + 1) {
                let cov: f64 = if fi == fj {
                    let blk = &c.family[fi * vw..(fi + 1) * vw];
                    let mut s = 0.0;
                    for (a, za) in zi.iter().enumerate() {
                        for (b, zb) in zj.iter().enumerate() {
                            let (r, k) = (li + a, lj + b);
                            s += za * zb * blk[vech_at(self.w, r.max(k), r.min(k))];
                        }
                    }
                    s
                } else {
                    us[i]
                        .iter()
                        .map(|&(a, ua)| {
                            ua * us[j].iter().map(|&(b, ub)| tail(a, b) * ub).sum::<f64>()
                        })
                        .sum()
                };
                v += 2.0 * cov;
            }
        }
        for u in &us {
            for &(t, zt) in &tails {
                v -= 2.0 * zt * u.iter().map(|&(b, ub)| ub * tail(b, t)).sum::<f64>();
            }
        }
        for (i, &(s, zs)) in tails.iter().enumerate() {
            for &(t, zt) in &tails[i + 1..] {
                v += 2.0 * zs * zt * tail(s, t);
            }
        }
        v
    }
}

/// The index of entry `(r, c)`, `r ≥ c`, in the column-major lower-triangle
/// vech of an `n × n` matrix.
fn vech_at(n: usize, r: usize, c: usize) -> usize {
    c * (2 * n - c + 1) / 2 + r - c
}

/// The Gauss-Hermite order of the future-observation mixture.
const FUTURE_NODES: usize = 21;

/// The `(1 − level)/2` and `(1 + level)/2` quantiles of `family`'s
/// distribution mixed over `η ~ N(eta, se²)`: the nodes `eta + √2·se·x_k`
/// with weights `w_k/√π` of the physicists' Gauss-Hermite rule. `w` is the
/// row's prior weight (binomial trials, the Gamma precision), `phi` the fit's
/// dispersion (the negative-binomial size, the Gamma `φ`). NaN on a
/// non-finite input or a family the bounds do not cover.
fn future_bounds(family: Family, eta: f64, se: f64, w: f64, phi: f64, level: f64) -> (f64, f64) {
    let nan = (f64::NAN, f64::NAN);
    if !(eta.is_finite() && se.is_finite()) {
        return nan;
    }
    // Order `2k + 1` is block `k` of the table.
    let k = FUTURE_NODES / 2;
    let block = GH_OFFSETS[k]..GH_OFFSETS[k + 1];
    let nodes: Vec<(f64, f64)> = GH_NODES[block.clone()]
        .iter()
        .zip(&GH_WEIGHTS[block])
        .map(|(&x, &wk)| (link_inv(family, eta + SQRT_2 * se * x), wk / PI.sqrt()))
        .collect();
    let mu_max = nodes.iter().fold(0.0f64, |m, &(mu, _)| m.max(mu));
    let probs = [(1.0 - level) / 2.0, (1.0 + level) / 2.0];
    let bounds = |q: &dyn Fn(f64) -> f64| (q(probs[0]), q(probs[1]));
    match family {
        Family::Binomial { .. } => {
            let trials = w as i64;
            let Ok(d) = nodes
                .iter()
                .map(|&(mu, wk)| Binomial::new(trials, mu).map(|d| (d, wk)))
                .collect::<Result<Vec<_>, _>>()
            else {
                return nan;
            };
            let cdf = |k: i64| d.iter().map(|(d, wk)| wk * d.cdf(k)).sum::<f64>();
            bounds(&|p| discrete_quantile(&cdf, p, trials, trials) as f64 / w)
        }
        Family::Poisson { .. } => {
            let Ok(d) = nodes
                .iter()
                .map(|&(mu, wk)| Poisson::new(mu).map(|d| (d, wk)))
                .collect::<Result<Vec<_>, _>>()
            else {
                return nan;
            };
            let cdf = |k: i64| d.iter().map(|(d, wk)| wk * d.cdf(k)).sum::<f64>();
            bounds(&|p| discrete_quantile(&cdf, p, mu_max.ceil() as i64, i64::MAX) as f64)
        }
        Family::NegativeBinomial { .. } => {
            let Ok(d) = nodes
                .iter()
                .map(|&(mu, wk)| NegBinomial::from_mean_size(mu, phi).map(|d| (d, wk)))
                .collect::<Result<Vec<_>, _>>()
            else {
                return nan;
            };
            let cdf = |k: i64| d.iter().map(|(d, wk)| wk * d.cdf(k)).sum::<f64>();
            bounds(&|p| discrete_quantile(&cdf, p, mu_max.ceil() as i64, i64::MAX) as f64)
        }
        Family::Gamma { .. } => {
            let shape = w / phi;
            let Ok(d) = nodes
                .iter()
                .map(|&(mu, wk)| Gamma::new(shape, shape / mu).map(|d| (d, wk)))
                .collect::<Result<Vec<_>, _>>()
            else {
                return nan;
            };
            let cdf = |y: f64| d.iter().map(|(d, wk)| wk * d.cdf(y)).sum::<f64>();
            bounds(&|p| continuous_quantile(&cdf, p, mu_max))
        }
        Family::Gaussian | Family::InverseGaussian { .. } => nan,
    }
}

/// The smallest `k ≥ 0` with `cdf(k) ≥ p`, at most `max`: an upper bracket
/// doubled from `start`, then bisection on the integers.
fn discrete_quantile(cdf: &dyn Fn(i64) -> f64, p: f64, start: i64, max: i64) -> i64 {
    if cdf(0) >= p {
        return 0;
    }
    let mut hi = start.clamp(1, max);
    while hi < max && cdf(hi) < p {
        hi = hi.saturating_mul(2).saturating_add(1).min(max);
    }
    // cdf(lo) < p ≤ cdf(hi), or hi == max.
    let mut lo = 0i64;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

/// The `y > 0` with `cdf(y) = p` for a CDF on the positive half-line: an
/// upper bracket doubled from `start`, then bisection to 1e-12 relative.
fn continuous_quantile(cdf: &dyn Fn(f64) -> f64, p: f64, start: f64) -> f64 {
    let mut hi = if start > 0.0 { start } else { 1.0 };
    while cdf(hi) < p {
        hi *= 2.0;
        if !hi.is_finite() {
            return f64::NAN;
        }
    }
    let mut lo = 0.0f64;
    while hi - lo > 1e-12 * hi {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) >= p {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(all(test, feature = "formula"))]
#[path = "predict_tests.rs"]
pub(crate) mod predict_tests;
