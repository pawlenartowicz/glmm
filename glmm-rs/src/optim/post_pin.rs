//! The probe a fit path runs at a pinned endpoint. It takes the objective as
//! a closure, so the path keeps its own evaluation (on `fit_glmm`, the
//! β-fixed Laplace or AGQ deviance) and its own evaluation count.

/// The stationarity probe at a pinned corner: whether the free θ coordinates
/// sit at a minimum of the nested model the pin leaves. Every coordinate
/// `j < lower.len()` that `skip` does not rule out is probed `step` either
/// way from `theta` (on its inner side only when the other side leaves
/// `[lower[j], upper[j]]`), and the gain the two probes predict along it is
/// the larger of the drop from `f0` to the lower probe and the parabola's
/// `g²/(2c)` through the three points. Returns the largest gain over the
/// coordinates and the evaluations run, and writes into `curv[j]` the
/// curvature `c = (f₊ + f₋ − 2·f0)/step²` the three points give along
/// coordinate `j`, NaN where the coordinate is skipped or a side is not
/// evaluated or not finite. `theta` and `scratch` have the same length, which
/// may run past `lower.len()` (coordinates the probe holds); `scratch` is
/// overwritten and every evaluation reads it. `curv` is at least
/// `lower.len()` long.
#[allow(clippy::too_many_arguments)]
pub(crate) fn stationarity_gain(
    theta: &[f64],
    scratch: &mut [f64],
    skip: &dyn Fn(usize) -> bool,
    lower: &[f64],
    upper: &[f64],
    f0: f64,
    step: f64,
    curv: &mut [f64],
    eval: &mut impl FnMut(&[f64]) -> f64,
) -> (f64, u32) {
    let mut gain = 0.0f64;
    let mut evals = 0u32;
    scratch.copy_from_slice(theta);
    for j in 0..lower.len() {
        curv[j] = f64::NAN;
        if skip(j) {
            continue;
        }
        let mut f = [f64::NAN; 2];
        for (side, sgn) in [1.0f64, -1.0].into_iter().enumerate() {
            let t = theta[j] + sgn * step;
            if t < lower[j] || t > upper[j] {
                continue;
            }
            scratch[j] = t;
            f[side] = eval(scratch);
            evals += 1;
        }
        scratch[j] = theta[j];
        // `f64::min` passes over a NaN side.
        let lower_probe = f[0].min(f[1]);
        if lower_probe.is_finite() {
            gain = gain.max(f0 - lower_probe);
        }
        if f[0].is_finite() && f[1].is_finite() {
            let h = step;
            let g = (f[0] - f[1]) / (2.0 * h);
            let c = (f[0] + f[1] - 2.0 * f0) / (h * h);
            curv[j] = c;
            if c > 0.0 {
                gain = gain.max(g * g / (2.0 * c));
            }
        }
    }
    (gain, evals)
}
