//! Profile-likelihood confidence intervals of a Gaussian linear mixed model,
//! as lme4's `confint(method = "profile")` computes them: for each
//! random-effect SD and correlation, the residual SD σ and each coefficient,
//! the parameter is held at points on each side of its estimate, the others
//! are re-optimized, and a bound is where the signed root of the deviance
//! change, `ζ(v) = sign(v − v̂)·√(dev(v) − dev̂)`, reaches `±z`,
//! `z = Φ⁻¹((1 + level)/2)`.
//!
//! Every interval is read off the ML deviance: a REML fit is refitted by ML
//! first, as lme4's profile does, and the intervals report the ML estimates.
//! The variance parameters are on lme4's scale (SDs and correlations); a
//! coefficient is held through the Schur complement of the fixed-effect
//! block, so it needs no refit with its column dropped. Both objectives are
//! evaluated on the LMM kernel the ML fit leaves behind.
//!
//! A bound is a root of `ζ(v) = ±z`, found by Brent's method once the side is
//! bracketed, not read off a spline of a profile grid. An SD lives in
//! `[0, ∞)`, a correlation in `[−1, 1]`, σ in `(0, ∞)` and a coefficient in
//! ℝ. A side whose estimate sits at its limit (an SD at 0, a correlation at
//! ±1) has the limit as its bound, as does a side whose finite limit is
//! reached with `|ζ| ≤ z`; a one-sided interval keeps the two-sided cutoff,
//! as lme4 does. A correlation next to an SD estimated at 0 is not
//! identified, and its row is NaN.

use std::fmt;

use super::tails::normal_quantile;
use crate::fit::{detect_aliased, fit_cold_keep_workspace, validate};
use crate::lmm::profile::{ProfileKernel, VarCoord};
use crate::{ColumnId, Data, Design, Family, Fit, FitError, FitOptions, ModelSpec, ReTerm};

/// Expansion steps per side; a side that has not bracketed the root after
/// them is undetermined.
const MAX_EXPAND: usize = 30;
/// Held minimizations the root search runs per side.
const MAX_ROOT_EVALS: usize = 60;
/// How close `|ζ|` comes to `z` at a bound; also how far `|ζ|` may fall from
/// one expansion step to the next before the side is undetermined.
const ZETA_TOL: f64 = 1e-6;
/// How far below the fit's deviance (kernel deviance units) a held minimum
/// may come before it is a better optimum; smaller drops read as `ζ = 0`.
const BETTER_OPTIMUM_TOL: f64 = 1e-6;

/// One parameter of a Gaussian linear mixed model, as
/// [`Fit::profile_intervals`] names it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileParam {
    /// The SD of column `term` of grouping `group` (`varcorr` block order).
    Sd {
        /// The grouping: 0 the primary, then each extra in declaration order.
        group: usize,
        /// The column of the grouping's `varcorr` block.
        term: usize,
    },
    /// The correlation of columns `row > col` of one term of grouping `group`.
    Cor {
        /// The grouping, as on [`ProfileParam::Sd`].
        group: usize,
        /// The later column of the block.
        row: usize,
        /// The earlier column of the block.
        col: usize,
    },
    /// The residual SD σ.
    Sigma,
    /// Fixed-effect coefficient `j`, a caller column.
    Beta(usize),
}

/// The profile-likelihood interval of one parameter.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProfileInterval {
    /// The parameter.
    pub param: ProfileParam,
    /// Its ML estimate, in the caller's units; NaN for a correlation next to
    /// an SD estimated at 0, which no covariance identifies.
    pub estimate: f64,
    /// The lower bound: NaN when undetermined, the limit itself (0 on an SD,
    /// −1 on a correlation) when the bound is the limit.
    pub lower: f64,
    /// The upper bound: NaN when undetermined, 1 on a correlation whose
    /// bound is the limit.
    pub upper: f64,
}

/// The profile-likelihood intervals of a fit, from
/// [`Fit::profile_intervals`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileIntervals {
    /// The confidence level, in `(0, 1)`.
    pub level: f64,
    /// −2 × the ML log-likelihood at the ML fit the profiles start from.
    pub deviance: f64,
    /// One row per parameter profiled, in glmm's order: per grouping
    /// (primary first) its SDs and correlations column-major, each SD
    /// followed by the correlations of its column with the later columns of
    /// its term; then σ, then the coefficients. lme4 2.0's `confint` lists a
    /// block's SDs before its correlations, so a comparison with lme4
    /// matches rows by name.
    pub rows: Vec<ProfileInterval>,
}

/// Why [`Fit::profile_intervals`] returned no intervals.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum ProfileError {
    /// The request was refused before any work.
    Fit(FitError),
    /// The ML fit the profiles start from did not converge.
    NotConverged,
    /// Holding `param` found a deviance `drop` (kernel deviance units) below
    /// the ML fit's, so the fit is not at its optimum and no bound measured
    /// from it holds.
    BetterOptimum {
        /// The parameter whose held search found it.
        param: ProfileParam,
        /// How far below the fit's deviance it is.
        drop: f64,
    },
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::Fit(e) => write!(f, "{e}"),
            ProfileError::NotConverged => {
                f.write_str("the maximum-likelihood fit the profiles start from did not converge")
            }
            ProfileError::BetterOptimum { param, drop } => write!(
                f,
                "profiling {param:?} found a deviance {drop} below the maximum-likelihood fit's: the fit is not at its optimum"
            ),
        }
    }
}

impl std::error::Error for ProfileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProfileError::Fit(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FitError> for ProfileError {
    fn from(e: FitError) -> Self {
        ProfileError::Fit(e)
    }
}

impl Fit {
    /// Profile-likelihood confidence intervals at `level` of a Gaussian
    /// linear mixed model, lme4's `confint(method = "profile")`: one row per
    /// random-effect SD and correlation, σ and coefficient (`which: None`),
    /// or per parameter in `which`, in glmm's order either way
    /// ([`ProfileIntervals::rows`]). `data` and `spec` are the fit's own.
    ///
    /// The profiles are of the ML deviance: the model is refitted by ML from
    /// a cold start (on an ML fit, the same fit again), and the rows report
    /// the ML estimates. A bound is where `|ζ|` reaches the normal quantile
    /// `Φ⁻¹((1 + level)/2)`, found by Brent's method to `|ζ| − z` within
    /// `1e-6`. An SD estimated at 0 has 0 as its lower bound, and a
    /// correlation whose profile stays below the quantile up to ±1 has ±1;
    /// a correlation next to an SD estimated at 0, and a side that never
    /// reaches the quantile or whose held search does not converge, give
    /// NaN.
    ///
    /// # Errors
    ///
    /// [`ProfileError::Fit`] wrapping: [`FitError::OptionValue`] when
    /// `level` is not strictly between 0 and 1 or `which` names a parameter
    /// the model does not have; [`FitError::Unsupported`] on a model without
    /// random effects, a family other than Gaussian, or a design with an
    /// aliased column; [`FitError::Shape`] when `data` has another `n` or `p`
    /// than the fit, or `spec` another set of random-effect blocks; any error
    /// of [`crate::fit`] on `data` and `spec`. [`ProfileError::NotConverged`]
    /// when the ML fit does not converge, and
    /// [`ProfileError::BetterOptimum`] when a held search finds a deviance
    /// below it.
    pub fn profile_intervals(
        &self,
        data: &Data<'_>,
        spec: &ModelSpec,
        level: f64,
        which: Option<&[ProfileParam]>,
    ) -> Result<ProfileIntervals, ProfileError> {
        let Design { n, p, .. } = data.design;
        if n != self.nobs || p != self.beta.len() {
            return Err(FitError::Shape(
                format!(
                    "the data have n = {n}, p = {p}; the fit was made on n = {}, p = {}",
                    self.nobs,
                    self.beta.len()
                )
                .into(),
            )
            .into());
        }
        if self.converged() && spec.re.is_some() {
            let blocks = block_terms(spec);
            let same = blocks.len() == self.varcorr.len()
                && blocks
                    .iter()
                    .zip(&self.varcorr)
                    .all(|(t, vech)| vech.len() == t.len() * (t.len() + 1) / 2);
            if !same {
                return Err(FitError::Shape(
                    "the spec's random-effect blocks are not the fit's".into(),
                )
                .into());
            }
        }
        profile_on(data, spec, level, which)
    }
}

/// Per grouping, primary first: the term of each column of its reported
/// block (the intercept, if any, then the slopes).
fn block_terms(spec: &ModelSpec) -> Vec<Vec<usize>> {
    let one = |terms: &[ReTerm], slopes: &[ColumnId]| -> Vec<usize> {
        if terms.is_empty() {
            return vec![0; 1 + slopes.len()];
        }
        terms
            .iter()
            .enumerate()
            .flat_map(|(t, term)| {
                std::iter::repeat_n(t, usize::from(term.intercept) + term.slopes.len())
            })
            .collect()
    };
    spec.re
        .iter()
        .flat_map(|re| {
            std::iter::once(one(&re.terms, &re.slopes))
                .chain(re.extra_groupings.iter().map(|g| one(&g.terms, &g.slopes)))
        })
        .collect()
}

/// Every parameter of a Gaussian mixed `spec` with `p` columns, in the order
/// of [`ProfileIntervals::rows`].
pub(crate) fn profile_params(spec: &ModelSpec, p: usize) -> Vec<ProfileParam> {
    let mut out = Vec::new();
    for (group, terms) in block_terms(spec).iter().enumerate() {
        let q = terms.len();
        for col in 0..q {
            for row in (col..q).filter(|&r| terms[r] == terms[col]) {
                out.push(if row == col {
                    ProfileParam::Sd { group, term: row }
                } else {
                    ProfileParam::Cor { group, row, col }
                });
            }
        }
    }
    out.push(ProfileParam::Sigma);
    out.extend((0..p).map(ProfileParam::Beta));
    out
}

/// The parameter of each of the kernel's coordinates. `hidden[b]` marks a
/// block without an intercept, whose idle column 0 the kernel keeps and the
/// reported block drops.
pub(super) fn coord_params(coords: &[VarCoord], hidden: &[bool]) -> Vec<ProfileParam> {
    coords
        .iter()
        .map(|&c| match c {
            VarCoord::Sd { block, row, .. } => ProfileParam::Sd {
                group: block,
                term: row - usize::from(hidden[block]),
            },
            VarCoord::Cor {
                block, row, col, ..
            } => {
                let h = usize::from(hidden[block]);
                ProfileParam::Cor {
                    group: block,
                    row: row - h,
                    col: col - h,
                }
            }
            VarCoord::Sigma => ProfileParam::Sigma,
        })
        .collect()
}

/// The ML fit, its fit-dependent refusals and the profiles;
/// [`Fit::profile_intervals`] and the ports both call it. Refuses as
/// [`Fit::profile_intervals`] does, except for the checks against a fit.
pub(crate) fn profile_on(
    data: &Data<'_>,
    spec: &ModelSpec,
    level: f64,
    which: Option<&[ProfileParam]>,
) -> Result<ProfileIntervals, ProfileError> {
    if !(level > 0.0 && level < 1.0) {
        return Err(FitError::OptionValue(
            format!("level must be a number strictly between 0 and 1, got {level}").into(),
        )
        .into());
    }
    if spec.re.is_none() || !matches!(spec.family, Family::Gaussian) {
        return Err(FitError::Unsupported(
            "profile intervals need a Gaussian model with random effects".into(),
        )
        .into());
    }
    let opts = FitOptions {
        reml: Some(false),
        satterthwaite: false,
        condvar: false,
        ..FitOptions::default()
    };
    validate(data, spec, None, &opts)?;
    let Design { x, n, p, ids, .. } = data.design;
    let params = profile_params(spec, p);
    if let Some(bad) = which.into_iter().flatten().find(|w| !params.contains(w)) {
        return Err(
            FitError::OptionValue(format!("the model has no parameter {bad:?}").into()).into(),
        );
    }
    if p > 0 && detect_aliased(x, n, p).contains(&true) {
        return Err(FitError::Unsupported(
            "profile intervals need a fixed-effect design without aliased columns".into(),
        )
        .into());
    }

    let (fit, mut fws) = fit_cold_keep_workspace(data, spec, &opts);
    if !fit.converged() {
        return Err(ProfileError::NotConverged);
    }
    let ids = ids.expect("validate requires ids on a mixed model");
    let level_counts: Vec<f64> = std::iter::once(&ids.primary)
        .chain(&ids.extra)
        .map(|g| {
            let mut l = g.clone();
            l.sort_unstable();
            l.dedup();
            l.len() as f64
        })
        .collect();
    let (ws, map) = fws
        .lmm_parts()
        .expect("a Gaussian mixed fit runs on the LMM workspace");
    let n_rows = ws.kernel.n_rows() as f64;
    let hidden = ws.kernel.groupings().hidden.clone();
    let mut k = ProfileKernel::new(ws, map);
    let coords = k.coords().to_vec();
    let var_params = coord_params(&coords, &hidden);
    assert!(
        params.starts_with(&var_params) && params.len() == var_params.len() + p,
        "the kernel's coordinates label their rows {var_params:?}, the spec's parameters are \
         {params:?}"
    );

    let z = normal_quantile(0.5 * (1.0 + level));
    let est = k.estimate();
    let base = k.kernel_deviance_at_hat();
    let sigma_hat = est[est.len() - 1];
    let wanted = |q: &ProfileParam| which.is_none_or(|w| w.contains(q));
    let sd_at = |b: usize, r: usize| {
        coords
            .iter()
            .zip(&est)
            .find_map(|(c, &e)| match *c {
                VarCoord::Sd { block, row, .. } if block == b && row == r => Some(e),
                _ => None,
            })
            .expect("each column of a term has its SD")
    };

    let mut rows = Vec::new();
    for (i, (&coord, &param)) in coords.iter().zip(&var_params).enumerate() {
        if !wanted(&param) {
            continue;
        }
        let e = est[i];
        let mut side = |toward: Toward<'_>| var_bound(&mut k, i, param, &est, base, z, toward);
        let (estimate, lower, upper) = match coord {
            VarCoord::Sd { block, .. } => {
                let lower = side(Toward::Limit(0.0))?;
                let upper = if e > 0.0 {
                    let h = z / (2.0 * level_counts[block]).sqrt();
                    side(Toward::Expand(&|t| (e.ln() + h * t).exp()))?
                } else {
                    side(Toward::Expand(&|t| sigma_hat * t / 8.0))?
                };
                (e, lower, upper)
            }
            VarCoord::Cor {
                block, row, col, ..
            } => {
                if sd_at(block, row) == 0.0 || sd_at(block, col) == 0.0 {
                    (f64::NAN, f64::NAN, f64::NAN)
                } else {
                    (e, side(Toward::Limit(-1.0))?, side(Toward::Limit(1.0))?)
                }
            }
            VarCoord::Sigma => {
                let h = z / (2.0 * n_rows).sqrt();
                let lower = side(Toward::Expand(&|t| (e.ln() - h * t).exp()))?;
                let upper = side(Toward::Expand(&|t| (e.ln() + h * t).exp()))?;
                (e, lower, upper)
            }
        };
        rows.push(ProfileInterval {
            param,
            estimate,
            lower,
            upper,
        });
    }

    let theta_hat = k.theta_hat().to_vec();
    for j in 0..p {
        let param = ProfileParam::Beta(j);
        if !wanted(&param) {
            continue;
        }
        let (b_hat, se) = (fit.beta[j], fit.se[j]);
        let mut side = |s: f64| {
            let zeta = path(z, base, param, theta_hat.clone(), |b, start| {
                k.hold_beta(j, b, start)
            });
            bound(z, b_hat, Toward::Expand(&|t| b_hat + s * z * se * t), zeta)
        };
        let lower = side(-1.0)?;
        let upper = side(1.0)?;
        rows.push(ProfileInterval {
            param,
            estimate: b_hat,
            lower,
            upper,
        });
    }

    Ok(ProfileIntervals {
        level,
        deviance: -2.0 * fit.loglik,
        rows,
    })
}

/// Where a side's bound lies.
pub(super) enum Toward<'s> {
    /// Up to a finite limit (0, −1 or 1).
    Limit(f64),
    /// With no finite limit: the expansion point at `t = 2^k`, `k = 0, 1, …`.
    Expand(&'s dyn Fn(f64) -> f64),
}

/// One side of variance coordinate `i`.
fn var_bound(
    k: &mut ProfileKernel<'_>,
    i: usize,
    param: ProfileParam,
    est: &[f64],
    base: f64,
    z: f64,
    toward: Toward<'_>,
) -> Result<f64, ProfileError> {
    let zeta = path(z, base, param, est.to_vec(), |v, start| {
        k.hold_var(i, v, start)
    });
    bound(z, est[i], toward, zeta)
}

/// `|ζ|` at each point of one side, off `hold(v, start)`, the held minimum
/// at `v` from `start` (overwritten with the argmin). Each held search
/// starts from the argmin of the farthest point found so far with
/// `|ζ| < z`, the first from `start`: the path walks out from the estimate,
/// and an argmin past the bound (a limit, an overshoot of the root search)
/// is never a start, since one on a variance's boundary, an SD at 0, can
/// hold the search there.
pub(super) fn path(
    z: f64,
    base: f64,
    param: ProfileParam,
    start: Vec<f64>,
    mut hold: impl FnMut(f64, &mut [f64]) -> (f64, bool),
) -> impl FnMut(f64) -> Result<Option<f64>, ProfileError> {
    let mut trial = start.clone();
    let mut inner = start;
    move |v| {
        trial.copy_from_slice(&inner);
        let (d, ok) = hold(v, &mut trial);
        let r = zeta_of(d, ok, base, param)?;
        if r.is_some_and(|r| r < z) {
            inner.copy_from_slice(&trial);
        }
        Ok(r)
    }
}

/// `|ζ|` off a held minimum `d`: `None` when the search did not converge,
/// [`ProfileError::BetterOptimum`] when `d` is below the base by more than
/// [`BETTER_OPTIMUM_TOL`].
fn zeta_of(
    d: f64,
    converged: bool,
    base: f64,
    param: ProfileParam,
) -> Result<Option<f64>, ProfileError> {
    if d < base - BETTER_OPTIMUM_TOL {
        return Err(ProfileError::BetterOptimum {
            param,
            drop: base - d,
        });
    }
    Ok(converged.then(|| (d - base).max(0.0).sqrt()))
}

/// One side's bound from the estimate `est`; `zeta(v)` is `|ζ|` at `v`
/// (`None` when its held search failed, which makes the side NaN). Toward a
/// limit: the limit when `est` is on it or `|ζ(limit)| ≤ z`, else the root
/// between them. Otherwise the expansion points are tried in turn until one
/// has `|ζ| ≥ z`, and the root lies between it and the point before; a side
/// that does not bracket in [`MAX_EXPAND`] steps, or whose `|ζ|` falls by
/// more than [`ZETA_TOL`] from one step to the next, is NaN.
pub(super) fn bound(
    z: f64,
    est: f64,
    toward: Toward<'_>,
    mut zeta: impl FnMut(f64) -> Result<Option<f64>, ProfileError>,
) -> Result<f64, ProfileError> {
    match toward {
        Toward::Limit(lim) => {
            if est == lim {
                return Ok(lim);
            }
            let Some(r) = zeta(lim)? else {
                return Ok(f64::NAN);
            };
            if r <= z {
                return Ok(lim);
            }
            root(&mut zeta, z, est, -z, lim, r - z)
        }
        Toward::Expand(step) => {
            let (mut a, mut ra, mut t) = (est, 0.0, 1.0);
            for _ in 0..MAX_EXPAND {
                let b = step(t);
                let Some(rb) = zeta(b)? else {
                    return Ok(f64::NAN);
                };
                if rb < ra - ZETA_TOL {
                    return Ok(f64::NAN);
                }
                if rb >= z {
                    return root(&mut zeta, z, a, ra - z, b, rb - z);
                }
                (a, ra, t) = (b, rb, 2.0 * t);
            }
            Ok(f64::NAN)
        }
    }
}

/// Brent's method on `g(v) = |ζ(v)| − z` over the bracket `[a, b]`, given
/// `g(a) = ga < 0 ≤ g(b) = gb`: the first point with `|g| ≤ ZETA_TOL`. NaN
/// when a held search fails, the bracket shrinks to rounding first, or
/// [`MAX_ROOT_EVALS`] evaluations do not reach it.
fn root(
    zeta: &mut impl FnMut(f64) -> Result<Option<f64>, ProfileError>,
    z: f64,
    mut a: f64,
    mut ga: f64,
    mut b: f64,
    mut gb: f64,
) -> Result<f64, ProfileError> {
    let (mut c, mut gc) = (b, gb);
    let (mut d, mut e) = (b - a, b - a);
    let mut evals = 0;
    loop {
        if (gb > 0.0 && gc > 0.0) || (gb < 0.0 && gc < 0.0) {
            (c, gc) = (a, ga);
            d = b - a;
            e = d;
        }
        if gc.abs() < gb.abs() {
            (a, ga) = (b, gb);
            (b, gb) = (c, gc);
            (c, gc) = (a, ga);
        }
        if gb.abs() <= ZETA_TOL {
            return Ok(b);
        }
        let tol = 2.0 * f64::EPSILON * b.abs().max(c.abs());
        let m = 0.5 * (c - b);
        if m.abs() <= tol || evals == MAX_ROOT_EVALS {
            return Ok(f64::NAN);
        }
        if e.abs() >= tol && ga.abs() > gb.abs() {
            // Inverse quadratic interpolation, or the secant when two of the
            // three points coincide.
            let s = gb / ga;
            let (mut p, mut q) = if a == c {
                (2.0 * m * s, 1.0 - s)
            } else {
                let (qa, r) = (ga / gc, gb / gc);
                (
                    s * (2.0 * m * qa * (qa - r) - (b - a) * (r - 1.0)),
                    (qa - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * m * q - (tol * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = m;
                e = m;
            }
        } else {
            d = m;
            e = m;
        }
        (a, ga) = (b, gb);
        b += if d.abs() > tol { d } else { tol.copysign(m) };
        let Some(r) = zeta(b)? else {
            return Ok(f64::NAN);
        };
        gb = r - z;
        evals += 1;
    }
}
