//! The fitted weight power ([`WeightPower::Estimate`]): δ in
//! `Var(εᵢ) = σ²·wᵢ^(−δ)` by golden section on the fit's log-likelihood, then
//! the fit at δ̂.

use super::common::{first_bad_power, weight_spread};
use super::glm::golden_max;
use super::{fit_checked, Fit, FitError, FitOptions, Note, SeFor, SeType, WeightPower};
use crate::{Data, ModelSpec, StartValues};

/// The search's stop width on δ, far below the noise in δ̂.
const DELTA_TOL: f64 = 1e-4;

/// The range of the weight power the simulations behind its estimate covered.
pub const WEIGHT_POWER_SIMULATED: (f64, f64) = (-1.0, 2.0);

/// The search bracket `±min(max(2, min(30, ln(1e28)/ln s)), ln(1e250)/maxᵢ|ln wᵢ|)`,
/// `s` the largest weight over the smallest: inner fits fail erratically once
/// the working weights spread far past 1e28, and the bracket shrinks below ±2
/// (where the claimed weights stay in reach) only to keep every `wᵢ^δ` within
/// 1e±250, so neither it nor XᵀWX overflows.
pub(crate) fn delta_bracket(w: &[f64]) -> (f64, f64) {
    let r = (1e28_f64.ln() / weight_spread(w).ln()).clamp(2.0, 30.0);
    let largest_log = w.iter().fold(0.0_f64, |a, v| a.max(v.ln().abs()));
    let r = r.min(1e250_f64.ln() / largest_log);
    (-r, r)
}

/// The options of a search probe: `opts` by REML (`reml`) or ML, with no
/// SEs, df or conditional variances, which the criterion does not need.
pub(crate) fn probe_options(opts: &FitOptions, reml: bool) -> FitOptions {
    FitOptions {
        se_for: SeFor::Only(vec![]),
        satterthwaite: false,
        condvar: false,
        se_type: Some(SeType::Model),
        reml: Some(reml),
        ..opts.clone()
    }
}

/// The profile at `delta`: the caller-scale [`Fit::loglik`] of the fit with
/// `probe` on `w^delta` (NaN when `w^delta` is not finite and `> 0`), and
/// whether that fit converged.
pub(crate) fn profile(
    data: &Data<'_>,
    spec: &ModelSpec,
    probe: &FitOptions,
    start: Option<&StartValues>,
    delta: f64,
) -> (f64, bool) {
    let w = data.design.weights.unwrap_or(&[]);
    if first_bad_power(w, delta).is_some() {
        return (f64::NAN, false);
    }
    let opts = FitOptions {
        weight_power: Some(WeightPower::Fixed(delta)),
        ..probe.clone()
    };
    let f = fit_checked(data, spec, start, &opts);
    (f.loglik, f.converged())
}

/// The fit at the power that maximizes [`profile`] under REML (`reml`) or ML
/// over [`delta_bracket`], plus one probe at the bracket's nearer end for
/// [`Note::WeightPowerAtBound`]. A failed probe scores `−1e300·(1 + |δ|)`, a slope
/// toward δ = 0 (the search keeps the right part on NaN and on a tie, so a
/// NaN would steer it silently). The fit at δ̂ runs with `opts`; `start` seeds
/// it and every probe. A fit at δ̂ whose log-likelihood or β̂ is not finite,
/// after a probe whose log-likelihood was, is a [`FitError::Weights`] (spec
/// 2-dcr §3.1).
pub(crate) fn fit_delta(
    data: &Data<'_>,
    spec: &ModelSpec,
    opts: &FitOptions,
    start: Option<&StartValues>,
    reml: bool,
) -> Result<Fit, FitError> {
    let w = data
        .design
        .weights
        .expect("a fitted weight power has weights");
    let (lower, upper) = delta_bracket(w);
    let probe = probe_options(opts, reml);
    let (mut failed, mut not_converged) = (0u32, 0u32);
    // The best finite probe: δ = 0 always has finite weights.
    let mut best = (f64::NEG_INFINITY, 0.0);
    let delta = golden_max(
        |d| match profile(data, spec, &probe, start, d) {
            (ll, converged) if ll.is_finite() => {
                not_converged += u32::from(!converged);
                if ll > best.0 {
                    best = (ll, d);
                }
                ll
            }
            _ => {
                failed += 1;
                -1e300 * (1.0 + d.abs())
            }
        },
        lower,
        upper,
        DELTA_TOL,
    );
    // A maximum on the edge of the powers whose `w^δ` is finite can leave the
    // bracket's midpoint just past it.
    let delta = if first_bad_power(w, delta).is_some() {
        best.1
    } else {
        delta
    };
    let at = FitOptions {
        weight_power: Some(WeightPower::Fixed(delta)),
        reml: Some(reml),
        ..opts.clone()
    };
    let mut fit = fit_checked(data, spec, start, &at);
    // It can fail where its probe did not (it computes what the probes skip).
    // Where no probe had a finite log-likelihood the model fails at every
    // power, as a fixed one would: that fit stands, with the failures note.
    if refit_failed(&fit) && best.0.is_finite() {
        return Err(FitError::Weights(
            format!(
                "the fit at the fitted weight power {delta} failed (its log-likelihood or \
                 coefficients are not finite); pass weight_power = 1, or another fixed power"
            )
            .into(),
        ));
    }
    // A profile that rises toward the nearer end, or goes flat there to
    // rounding, has no maximum inside the bracket, wherever the search stopped.
    let end = if delta > 0.0 { upper } else { lower };
    let at_bound =
        profile(data, spec, &probe, start, end).0 >= fit.loglik - 1e-9 * fit.loglik.abs().max(1.0);
    fit.weight_power_estimated = true;
    if fit.df > 0 {
        fit.df += 1;
    }
    let notes = &mut fit.diagnostics.notes;
    if failed + not_converged > 0 {
        notes.push(Note::WeightPowerSearchFailures {
            failed,
            not_converged,
        });
    }
    if at_bound {
        notes.push(Note::WeightPowerAtBound {
            delta,
            lower,
            upper,
        });
    }
    if let Some(note) = implausibility_note(delta, w, at_bound) {
        notes.push(note);
    }
    Ok(fit)
}

/// The note on how far δ̂ sits from the weights as claimed, if any:
/// [`Note::WeightPowerImplausible`] when δ̂ is outside
/// [`WEIGHT_POWER_SIMULATED`] and `s^|δ̂ − 1| > 10` (`s` the largest weight over
/// the smallest), else [`Note::WeightPowerOutsideUnit`] when δ̂ is outside
/// [0, 1] and the fit has no [`Note::WeightPowerAtBound`].
pub(crate) fn implausibility_note(delta: f64, w: &[f64], at_bound: bool) -> Option<Note> {
    let (lo, hi) = WEIGHT_POWER_SIMULATED;
    // In logs: s^|δ̂ − 1| overflows on wide weights.
    let log_factor = (delta - 1.0).abs() * weight_spread(w).ln();
    if !(lo..=hi).contains(&delta) && log_factor > 10f64.ln() {
        Some(Note::WeightPowerImplausible {
            delta,
            factor: log_factor.exp(),
        })
    } else if !at_bound && !(0.0..=1.0).contains(&delta) {
        Some(Note::WeightPowerOutsideUnit { delta })
    } else {
        None
    }
}

/// A refit at δ̂ whose β̂ or log-likelihood is not finite (an aliased
/// column's NaN aside).
fn refit_failed(fit: &Fit) -> bool {
    !fit.loglik.is_finite()
        || fit
            .beta
            .iter()
            .zip(fit.aliased())
            .any(|(b, &a)| !a && !b.is_finite())
}
