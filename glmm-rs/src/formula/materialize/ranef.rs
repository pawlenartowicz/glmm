use super::ReGroupInfo;
use crate::formula::error::Error;

/// One grouping's random-effect conditional modes, labelled — the shape every
/// consumer renders, and the ONLY place the kernel's RE block layout is
/// interpreted. Rows are levels, columns are terms; padded nested slots are
/// dropped, so `levels.len()` is the row count and `values.len() = levels.len()
/// · terms.len()`.
pub struct RanefBlock {
    /// Grouping factor name — [`ReGroupInfo::name`].
    pub group: String,
    /// Column names, `"(Intercept)"` first — [`ReGroupInfo::terms`].
    pub terms: Vec<String>,
    /// Row labels, one per retained level, in kernel block order.
    pub levels: Vec<String>,
    /// Row-major `levels.len() × terms.len()` conditional modes.
    pub values: Vec<f64>,
    /// Row-major `levels.len() × q(q+1)/2`, `q = terms.len()`: each level's
    /// conditional covariance, the vech of its block in
    /// [`crate::Fit::condvar()`] (column-major lower triangle, the stored
    /// `varcorr` order). Empty where `Fit::condvar(g)` is.
    pub condvar: Vec<f64>,
}

/// Label a fit's random-effect conditional modes: the [`crate::Fit::ranef()`]
/// blocks zipped against the lowering's slot labels, padded slots dropped, one
/// [`RanefBlock`] per grouping in declaration order.
///
/// This is the whole of the RE block-layout knowledge, written once. Every
/// consumer — the Python and R packages, a Rust caller — goes through here
/// rather than re-deriving the slicing, because a nested grouping's padding
/// depends on the data and cannot be inferred from the formula.
///
/// A fit with no conditional modes (fixed-only, where `Fit::ranef` is empty,
/// or non-converged, where it is NaN) yields an empty vec — not an error.
///
/// # Errors
/// [`Error::RanefShapeMismatch`] when `re_groups` does not describe this fit:
/// a different grouping count, a slot-label count that disagrees with
/// [`crate::Fit::ranef_levels()`], or a total that disagrees with the stored modes' length
/// (or with a non-empty `condvar`'s length).
/// Partial results are never returned — a mislabelled mode reads as a wrong
/// answer, not a cosmetic slip.
pub fn label_ranef(fit: &crate::Fit, re_groups: &[ReGroupInfo]) -> Result<Vec<RanefBlock>, Error> {
    if fit.ranef.is_empty() || fit.ranef.iter().any(|v| v.is_nan()) {
        return Ok(Vec::new());
    }
    let mismatch = |what: &str| Error::RanefShapeMismatch(what.to_string());
    if fit.ranef_levels.len() != re_groups.len() {
        return Err(mismatch(&format!(
            "fit has {} grouping(s), the formula lowered {}",
            fit.ranef_levels.len(),
            re_groups.len()
        )));
    }
    let total: usize = fit
        .ranef_levels
        .iter()
        .zip(re_groups)
        .map(|(&l, g)| l * g.terms.len())
        .sum();
    if total != fit.ranef.len() {
        return Err(mismatch(&format!(
            "ranef holds {} value(s), the lowered blocks span {total}",
            fit.ranef.len()
        )));
    }
    let vech = |q: usize| q * (q + 1) / 2;
    let total_cv: usize = fit
        .ranef_levels
        .iter()
        .zip(re_groups)
        .map(|(&l, g)| l * vech(g.terms.len()))
        .sum();
    if !fit.condvar.is_empty() && total_cv != fit.condvar.len() {
        return Err(mismatch(&format!(
            "condvar holds {} value(s), the lowered blocks span {total_cv}",
            fit.condvar.len()
        )));
    }
    let mut out = Vec::with_capacity(re_groups.len());
    let mut base = 0usize;
    let mut base_cv = 0usize;
    for (g, info) in re_groups.iter().enumerate() {
        let n_levels = fit.ranef_levels[g];
        let q = info.terms.len();
        if info.slot_labels.len() != n_levels {
            return Err(mismatch(&format!(
                "grouping {:?} has {} slot label(s) for {n_levels} level(s)",
                info.name,
                info.slot_labels.len()
            )));
        }
        let mut levels = Vec::new();
        let mut values = Vec::new();
        let mut condvar = Vec::new();
        let v = vech(q);
        for (l, label) in info.slot_labels.iter().enumerate() {
            // A padded nested slot is not a level: no row is assigned to it and
            // its mode is zero by construction, so it is dropped rather than
            // reported (lme4 has no counterpart to it either).
            let Some(label) = label else { continue };
            levels.push(label.clone());
            values.extend_from_slice(&fit.ranef[base + l * q..base + (l + 1) * q]);
            if !fit.condvar.is_empty() {
                condvar.extend_from_slice(&fit.condvar[base_cv + l * v..base_cv + (l + 1) * v]);
            }
        }
        out.push(RanefBlock {
            group: info.name.clone(),
            terms: info.terms.clone(),
            levels,
            values,
            condvar,
        });
        base += n_levels * q;
        base_cv += n_levels * v;
    }
    Ok(out)
}
