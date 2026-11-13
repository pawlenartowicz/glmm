//! A robust fit ([`SeType::Robust`]): the CR2 or HC2 pieces at the final fit,
//! built in the kernel's coordinates and mapped to the caller's once,
//! then the standard errors, df and notes they give.

use faer::{Mat, MatRef};

use super::common::has_random_slope;
use super::{Fit, Note, SeType};
use crate::inference::cr2::{cr2_pieces, ClusterBlock, Cr2Pieces, DF_SMALL};
use crate::inference::evd::sym_eigen;
use crate::standardize::ColumnScaling;
use crate::ModelSpec;

/// A Gaussian fit's rows and estimates, in the caller's units, as its
/// assembly holds them.
pub(crate) struct Rows<'a> {
    /// Row-major `n × p`.
    pub(crate) x: &'a [f64],
    pub(crate) y: &'a [f64],
    pub(crate) offset: Option<&'a [f64]>,
    /// The caller's prior weights, paired with `sigma2`.
    pub(crate) weights: Option<&'a [f64]>,
    pub(crate) beta: &'a [f64],
    /// The residual variance at weight 1.
    pub(crate) sigma2: f64,
}

/// The primary grouping of a mixed fit: each row's level, and its
/// random-effect row `z` (row-major `n × q`) with covariance `d` (`q × q`,
/// row-major), so Φ_k = diag(σ²/w) + Z_k D Z_kᵀ.
pub(crate) struct Groups<'a> {
    pub(crate) ids: &'a [u32],
    pub(crate) z: Vec<f64>,
    pub(crate) d: Vec<f64>,
}

/// The pieces of `rows` clustered by `groups` (one row per cluster without),
/// at the fit's factor `factor` of X̃ᵀV⁻¹X̃ and its σ̂², `sigma2`, in the
/// kernel's coordinates X̃ = X·B, B the column map `map`; their `t` mapped to
/// the caller's. `factor` is lower triangular, its leading `p × p` block read.
/// `None` where [`cr2_pieces`] has none (an exact fit, an overflow).
pub(crate) fn pieces(
    rows: &Rows<'_>,
    groups: Option<&Groups<'_>>,
    factor: MatRef<'_, f64>,
    sigma2: f64,
    map: &ColumnScaling,
) -> Option<Cr2Pieces> {
    let p = rows.beta.len();
    let n = rows.y.len();
    // Rows in cluster order: a counting sort by level.
    let (order, bounds) = match groups {
        Some(g) => {
            let levels = g.ids.iter().max().map_or(0, |&l| l as usize + 1);
            let mut start = vec![0usize; levels + 1];
            for &l in g.ids {
                start[l as usize + 1] += 1;
            }
            for l in 0..levels {
                start[l + 1] += start[l];
            }
            let mut next = start.clone();
            let mut order = vec![0usize; n];
            for (i, &l) in g.ids.iter().enumerate() {
                order[next[l as usize]] = i;
                next[l as usize] += 1;
            }
            start.dedup();
            (order, start)
        }
        None => ((0..n).collect(), (0..=n).collect()),
    };
    let mut xt = Mat::from_fn(n, p, |i, j| rows.x[order[i] * p + j]);
    for j in 0..p {
        map.standardize_column(j, xt.col_as_slice_mut(j));
    }
    let r: Vec<f64> = order
        .iter()
        .map(|&i| {
            let xb: f64 = (0..p).map(|j| rows.x[i * p + j] * rows.beta[j]).sum();
            rows.y[i] - rows.offset.map_or(0.0, |o| o[i]) - xb
        })
        .collect();
    let lambda: Vec<f64> = order
        .iter()
        .map(|&i| rows.sigma2 / rows.weights.map_or(1.0, |w| w[i]))
        .collect();
    // U = Z C with D = C Cᵀ; a direction of D at or below zero adds nothing.
    let u = match groups {
        Some(g) => {
            let q = (g.d.len() as f64).sqrt() as usize;
            let d = MatRef::from_row_major_slice(&g.d, q, q);
            let eig = sym_eigen(d, true).ok()?;
            let v = eig.vectors.expect("vectors asked for");
            let c = Mat::from_fn(q, q, |a, b| v[(a, b)] * eig.values[b].max(0.0).sqrt());
            Mat::from_fn(n, q, |i, b| {
                (0..q).map(|a| g.z[order[i] * q + a] * c[(a, b)]).sum()
            })
        }
        None => Mat::zeros(n, 0),
    };
    let blocks: Vec<ClusterBlock<'_>> = bounds
        .windows(2)
        .map(|w| {
            let (s, len) = (w[0], w[1] - w[0]);
            ClusterBlock {
                x: xt.as_ref().subrows(s, len),
                r: &r[w[0]..w[1]],
                lambda: &lambda[w[0]..w[1]],
                u: u.as_ref().subrows(s, len),
            }
        })
        .collect();
    // R = Lᵀ/σ̂, so R M Rᵀ = I for M = σ̂²(L Lᵀ)⁻¹, the fit's model covariance.
    let sd = sigma2.sqrt();
    let r = Mat::from_fn(p, p, |i, j| if i <= j { factor[(j, i)] / sd } else { 0.0 });
    let mut pc = cr2_pieces(&blocks, r.as_ref())?;
    for k in 0..pc.t.ncols() {
        map.beta_to_caller(pc.t.col_as_slice_mut(k));
    }
    Some(pc)
}

/// On a fit with [`SeType::Robust`] in `se_type`: the standard errors,
/// `vcov` and `coef_df` of `fit.robust` on the `targets` columns (NaN on the
/// rest, and where one cluster alone identifies a coefficient), and the
/// robust notes. A fit without pieces (no endpoint, an exact fit) has NaN
/// SEs, `vcov` and df.
pub(crate) fn apply(fit: &mut Fit, targets: &[u32], se_type: Option<SeType>, spec: &ModelSpec) {
    if se_type != Some(SeType::Robust) {
        return;
    }
    fit.se_type = SeType::Robust;
    if has_random_slope(spec) {
        fit.diagnostics.notes.push(Note::RobustWithRandomSlope);
    }
    let p = fit.beta.len();
    fit.coef_df = vec![f64::NAN; p];
    fit.se = vec![f64::NAN; p];
    fit.vcov = vec![f64::NAN; p * p];
    let Some(pc) = &fit.robust else {
        return;
    };
    let v = pc.vcov();
    let mut unidentified = Vec::new();
    let mut small = (Vec::new(), Vec::new());
    let mut kept = Vec::with_capacity(targets.len());
    for &t in targets {
        let j = t as usize;
        let mut e = vec![0.0; p];
        e[j] = 1.0;
        if pc.identified_by_one(&e) {
            unidentified.push(t);
            continue;
        }
        let df = pc.df(&e);
        fit.coef_df[j] = df;
        if df < DF_SMALL {
            small.0.push(t);
            small.1.push(df);
        }
        kept.push(j);
    }
    for &i in &kept {
        fit.se[i] = v[(i, i)].sqrt();
        for &j in &kept {
            fit.vcov[i * p + j] = v[(i, j)];
        }
    }
    if !unidentified.is_empty() {
        fit.diagnostics.notes.push(Note::RobustSeUnidentified {
            columns: unidentified,
        });
    }
    if !small.0.is_empty() {
        fit.diagnostics.notes.push(Note::RobustDfSmall {
            columns: small.0,
            df: small.1,
        });
    }
}

#[cfg(test)]
#[path = "robust_tests.rs"]
mod tests;
