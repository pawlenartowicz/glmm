//! The lme4-shaped summary record both ports print (`summary`), and the
//! Pearson residuals its scaled-residual block reads (`pearson_residuals`).
//! Each port renders the record; the port's name in the title and the
//! footer of a fit with Satterthwaite switched off, which spells the port's
//! own argument, stay with the port.

use std::f64::consts::SQRT_2;

use glmm::orchestrate;
use glmm::WEIGHT_POWER_SIMULATED;

use crate::coerce;
use crate::error::BindError;
use crate::inputs::Inputs;
use crate::records::{int, opt_text, read_re_groups, text};
use crate::value::{Record, Value};

/// The footer of a Gaussian mixed model's t tests.
const SATTERTHWAITE_FOOTNOTE: &str =
    "t tests use Satterthwaite degrees of freedom (finite-difference θ-Hessian).";

/// `x` to four significant digits, trailing zeros dropped.
fn sig4(x: f64) -> String {
    if x == 0.0 || !x.is_finite() {
        return format!("{x}");
    }
    let decimals = (3 - x.abs().log10().floor() as i32).clamp(0, 10) as usize;
    let s = format!("{x:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        s
    }
}

/// The footer of a robust fit's t tests, by its df method: CR2 clustered by
/// the grouping factor, or HC2.
fn robust_footnote(df_method: &str, group: Option<&str>) -> Option<String> {
    match df_method {
        "pustejovsky_tipton" => Some(format!(
            "Standard errors: CR2, clustered by {}; t tests use Pustejovsky-Tipton degrees of \
             freedom.",
            group.unwrap_or("the grouping factor")
        )),
        "bell_mccaffrey" => {
            Some("Standard errors: HC2; t tests use Bell-McCaffrey degrees of freedom.".to_owned())
        }
        _ => None,
    }
}

/// The weight power's lines: the power when estimated or other than 1; when
/// estimated, whether it left the simulated range, and, under a model SE,
/// that the SE treats it as known (and, on an ML fit, that the simulations
/// estimated it by REML).
fn weight_power_lines(power: f64, estimated: bool, robust: bool, reml: bool) -> Vec<String> {
    if !estimated {
        return if power.is_finite() && power != 1.0 {
            vec![format!("Weight power: {}", sig4(power))]
        } else {
            Vec::new()
        };
    }
    let mut lines = vec![format!("Weight power: {} (estimated)", sig4(power))];
    let (lo, hi) = WEIGHT_POWER_SIMULATED;
    if !(lo..=hi).contains(&power) {
        lines.push(format!(
            "The estimated weight power is outside [{lo}, {hi}], the range the simulations \
             behind it covered."
        ));
    }
    if !robust {
        lines.push(if reml {
            "Standard errors treat the estimated weight power as known.".to_owned()
        } else {
            "Standard errors treat the estimated weight power as known; the simulations behind \
             it estimated it by REML."
                .to_owned()
        });
    }
    lines
}

/// sqrt(V(mu)) of `family`, taken without forming V: mu^2 or mu^3 leaves the
/// double range at extreme means where sqrt(V) does not (so not the engine's
/// `family::variance`). `theta` is the negative-binomial shape a fit reports
/// as `dispersion`.
fn sqrt_variance(family: &str) -> Result<fn(f64, f64) -> f64, BindError> {
    let f: fn(f64, f64) -> f64 = match family {
        "gaussian" => |_, _| 1.0,
        "binomial" => |mu, _| (mu * (1.0 - mu)).sqrt(),
        "poisson" => |mu, _| mu.sqrt(),
        "gamma" => |mu, _| mu,
        "negativebinomial" => |mu, theta| mu.sqrt() * (1.0 + mu / theta).sqrt(),
        "inversegaussian" => |mu, _| mu * mu.sqrt(),
        other => {
            return Err(BindError::invalid_argument(format!(
                "family {other:?} is not a glmm family"
            )))
        }
    };
    Ok(f)
}

/// `(y - mu) / sqrt(phi * V(mu) / w^v)`, `v` the fit's weight power.
fn pearson(
    family: &str,
    y: &[f64],
    mu: &[f64],
    weights: Option<&[f64]>,
    dispersion: f64,
    weight_power: f64,
) -> Result<Vec<f64>, BindError> {
    let sd = sqrt_variance(family)?;
    if y.len() != mu.len() || weights.is_some_and(|w| w.len() != mu.len()) {
        return Err(BindError::invalid_argument(format!(
            "y, fitted and weights must have the same length, got {}, {} and {}",
            y.len(),
            mu.len(),
            weights.map_or(mu.len(), <[f64]>::len)
        )));
    }
    // Only these families' dispersion is a scale phi on V(mu); negbin's is
    // theta, already inside V.
    let phi = if matches!(family, "gaussian" | "gamma" | "inversegaussian") {
        dispersion
    } else {
        1.0
    };
    Ok(y.iter()
        .zip(mu)
        .enumerate()
        .map(|(i, (&y, &mu))| {
            let w = weights.map_or(1.0, |w| {
                if weight_power == 1.0 {
                    w[i]
                } else {
                    w[i].powf(weight_power)
                }
            });
            (y - mu) / sd(mu, dispersion) * (w / phi).sqrt()
        })
        .collect())
}

/// `pearson_residuals`: per row `(y - fitted) / sqrt(phi * V(fitted) / w^v)`,
/// `v` the fit's `weight_power` (1 when absent).
pub(crate) fn pearson_residuals(inp: &mut Inputs) -> Result<Value, BindError> {
    let family = inp.string("family")?;
    let y = inp.vec("y")?;
    let fitted = inp.vec("fitted")?;
    let weights = inp.opt_vec("weights")?;
    let dispersion = inp.f64("dispersion")?;
    let power = inp.opt_f64("weight_power")?.unwrap_or(1.0);
    pearson(&family, &y, &fitted, weights.as_deref(), dispersion, power).map(Value::Vec)
}

/// `deviance_residual_sum`: `Σ w^v · d(y, fitted)`, `d` the family's unit
/// deviance and `v` the fit's `weight_power` (1 when absent), as the Pearson
/// residuals read the weights. `dispersion` is read only on negative
/// binomial, as its shape theta.
pub(crate) fn deviance_residual_sum(inp: &mut Inputs) -> Result<Value, BindError> {
    let family = inp.string("family")?;
    let y = inp.vec("y")?;
    let fitted = inp.vec("fitted")?;
    let weights = inp.opt_vec("weights")?;
    let theta = inp.f64("dispersion")?;
    let power = inp.opt_f64("weight_power")?.unwrap_or(1.0);
    let weights = weights.map(|w| {
        if power == 1.0 {
            w
        } else {
            w.iter().map(|w| w.powf(power)).collect()
        }
    });
    orchestrate::unit_deviance_sum(&family, theta, &y, &fitted, weights.as_deref())
        .map(Value::F64)
        .map_err(BindError::invalid_argument)
}

/// Min, quartiles and max as R's `quantile()` (type 7) gives them; all NaN
/// when a value is NaN, as numpy's `quantile` gives them.
fn five_numbers(mut r: Vec<f64>) -> Vec<f64> {
    if r.iter().any(|v| v.is_nan()) {
        return vec![f64::NAN; 5];
    }
    r.sort_by(f64::total_cmp);
    let last = (r.len() - 1) as f64;
    [0.0, 0.25, 0.5, 0.75, 1.0]
        .iter()
        .map(|q| {
            let index = last * q;
            let lo = index.floor();
            let (a, b) = (r[lo as usize], r[index.ceil() as usize]);
            let h = index - lo;
            if h > 0.0 && a != b {
                (1.0 - h) * a + h * b
            } else {
                a
            }
        })
        .collect()
}

/// A vech-packed (column-major lower-triangular) covariance block as its
/// standard deviations and its correlation matrix (column-major), after the
/// check the engine's `vech_stddev_corr` leaves to its caller.
fn stddev_corr(vech: &[f64], group: usize) -> Result<(Vec<f64>, Vec<f64>), BindError> {
    let m = vech.len();
    let q = ((1 + 8 * m).isqrt() - 1) / 2;
    if q * (q + 1) / 2 != m {
        return Err(BindError::invalid_argument(format!(
            "fit.varcorr[{group}] is not a valid vech (length {m})"
        )));
    }
    let (sd, corr) = orchestrate::vech_stddev_corr(vech);
    // Symmetric, so its rows laid end to end are its columns.
    Ok((sd, corr.concat()))
}

fn method_line(family: &str, mixed: bool, reml: bool, nagq: usize) -> String {
    let gaussian = family == "gaussian";
    match (mixed, gaussian) {
        (false, true) => "Linear model fit by least squares".to_owned(),
        (false, false) => "Generalized linear model fit by IRLS".to_owned(),
        (true, true) if reml => "Linear mixed model fit by REML".to_owned(),
        (true, true) => "Linear mixed model fit by maximum likelihood".to_owned(),
        (true, false) if nagq > 1 => format!(
            "Generalized linear mixed model fit by maximum likelihood \
             (Adaptive Gauss-Hermite Quadrature, nAGQ = {nagq})"
        ),
        (true, false) => {
            "Generalized linear mixed model fit by maximum likelihood (Laplace Approximation)"
                .to_owned()
        }
    }
}

/// Gamma phi is the Pearson moment on a GLM and the maximum-likelihood value
/// on a GLMM; inverse-Gaussian phi is always the Pearson moment; a held
/// `dispersion=` reports as fixed.
fn dispersion_label(family: &str, held: bool, mixed: bool) -> Option<&'static str> {
    match family {
        "negativebinomial" => Some("Shape (theta)"),
        "gamma" | "inversegaussian" if held => Some("Dispersion (phi, fixed)"),
        "gamma" if mixed => Some("Dispersion (phi, ML)"),
        "gamma" | "inversegaussian" => Some("Dispersion (phi, Pearson)"),
        _ => None,
    }
}

/// NaN on an aliased row. A NaN already there is kept bit for bit, so R's
/// `NA` comes back `NA`.
fn mask(v: Vec<f64>, aliased: &[bool]) -> Vec<f64> {
    v.into_iter()
        .zip(aliased)
        .map(|(x, &a)| if a && !x.is_nan() { f64::NAN } else { x })
        .collect()
}

fn opt_f64(x: Option<f64>) -> Value {
    x.map_or(Value::Null, Value::F64)
}

/// `summary`: the record of every block lme4's `print.summary.merMod`
/// prints, in its order, and the weight power's lines. Fields a fit stored
/// by an earlier version may lack (`df_method`, `coef_df`, `coef_p`,
/// `dispersion_held`, `rss`, `r_squared`, `adj_r_squared`, `warnings`,
/// `weight_power`, `weight_power_estimated`, the groupings) are optional:
/// such a fit summarizes as the table its version printed, at power 1.
pub(crate) fn summary(inp: &mut Inputs) -> Result<Value, BindError> {
    let need_refit = inp.bool("need_refit")?;
    let sizes = match inp.opt_record("effect_sizes")? {
        Some(mut e) => [
            e.opt_f64("marginal")?,
            e.opt_f64("conditional")?,
            e.opt_f64("adjusted")?,
        ],
        None => [None; 3],
    };
    let mut fit = inp.record("fit")?;
    let family = fit.string("family")?;
    let reml = fit.bool("reml")?;
    let nobs = fit.usize("nobs")?;
    let df = fit.usize("df")?;
    let loglik = fit.f64("loglik")?;
    let names = fit.strs("names")?;
    let p = names.len();
    let beta = fit.vec("beta")?;
    let se = fit.vec("se")?;
    let aliased = fit.bools("aliased")?;
    if beta.len() != p || se.len() != p || aliased.len() != p {
        return Err(BindError::invalid_argument(format!(
            "fit.beta, fit.se and fit.aliased must have one entry per name ({p}), got {}, {} \
             and {}",
            beta.len(),
            se.len(),
            aliased.len()
        )));
    }
    let vcov = fit.rows("vcov", p)?;
    let dispersion = fit.f64("dispersion")?;
    let varcorr = fit.opt_vecs("varcorr")?.unwrap_or_default();
    // A mixed fit that did not converge has no `varcorr` but keeps its groupings.
    let re_groups = read_re_groups(&mut fit)?;
    let mixed = !re_groups.is_empty();
    let df_method = fit
        .opt_string("df_method")?
        .unwrap_or_else(|| "none".to_owned());
    let per_row = |fit: &mut Inputs, name: &str| -> Result<Vec<f64>, BindError> {
        let v = fit.vec(name)?;
        if v.len() == p {
            Ok(v)
        } else {
            Err(BindError::invalid_argument(format!(
                "fit.{name} has {} entries, expected {p}",
                v.len()
            )))
        }
    };

    // t (with the fit's p) where the fit has a df, Wald z on the normal
    // elsewhere; a df column only for the Satterthwaite and robust df, which
    // differ per row.
    let per_row_df = matches!(
        df_method.as_str(),
        "satterthwaite" | "pustejovsky_tipton" | "bell_mccaffrey"
    );
    let t_test = per_row_df || df_method == "residual";
    let statistic: Vec<f64> = beta.iter().zip(&se).map(|(b, s)| b / s).collect();
    let p_value = if t_test {
        per_row(&mut fit, "coef_p")?
    } else {
        statistic
            .iter()
            .map(|z| libm::erfc(z.abs() / SQRT_2))
            .collect()
    };
    let coef_df = if per_row_df {
        Value::Vec(mask(per_row(&mut fit, "coef_df")?, &aliased))
    } else {
        Value::Null
    };
    let n_aliased = aliased.iter().filter(|&&a| a).count();
    // n - p, the residual df of OLS and the dispersion GLMs: a non-converged
    // fit's coef_df is NaN, and its footer still prints n - p. A robust OLS
    // fit (Bell-McCaffrey) keeps it: its σ̂ is rss / (n - p) as on the model
    // SE, also at an estimated weight power, which `df` counts instead.
    let residual_df = if matches!(df_method.as_str(), "residual" | "bell_mccaffrey") {
        Value::I64(nobs as i64 - (p - n_aliased) as i64)
    } else {
        Value::Null
    };

    let criterion = if reml {
        Record::new().with("REML criterion at convergence", Value::F64(-2.0 * loglik))
    } else {
        let bic = if nobs == 0 {
            f64::NAN
        } else {
            -2.0 * loglik + df as f64 * (nobs as f64).ln()
        };
        Record::new()
            .with("AIC", Value::F64(-2.0 * loglik + 2.0 * df as f64))
            .with("BIC", Value::F64(bic))
            .with("logLik", Value::F64(loglik))
            .with("deviance", Value::F64(-2.0 * loglik))
            // `df` is 0 where it is not counted (a fit without estimates).
            .with(
                "df.resid",
                if df == 0 {
                    Value::F64(f64::NAN)
                } else {
                    Value::I64(nobs as i64 - df as i64)
                },
            )
    };

    let y = fit.vec("y")?;
    let fitted = fit.vec("fitted")?;
    let weights = fit.opt_vec("weights")?;
    let power = fit.opt_f64("weight_power")?.unwrap_or(1.0);
    let estimated = fit.opt_bool("weight_power_estimated")?.unwrap_or(false);
    let scaled = if !fitted.is_empty() && fitted.len() == y.len() {
        Value::Vec(five_numbers(pearson(
            &family,
            &y,
            &fitted,
            weights.as_deref(),
            dispersion,
            power,
        )?))
    } else {
        Value::Null
    };

    let term_index = match fit.value("re_term_index") {
        Value::Null => Vec::new(),
        Value::List(items) => items
            .into_iter()
            .map(coerce::to_usizes)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|w| BindError::invalid_argument(format!("fit.re_term_index {w}")))?,
        other => vec![coerce::to_usizes(other)
            .map_err(|w| BindError::invalid_argument(format!("fit.re_term_index {w}")))?],
    };
    let mut random_effects = Vec::with_capacity(varcorr.len());
    for (g, vech) in varcorr.iter().enumerate() {
        let (sd, corr) = stddev_corr(vech, g)?;
        let (group, terms) = re_groups.get(g).cloned().unwrap_or_default();
        let q = sd.len();
        random_effects.push(Value::Record(
            Record::new()
                .with("group", Value::Str(group))
                .with("terms", Value::StrVec(terms))
                .with(
                    "term_index",
                    Value::Index(term_index.get(g).cloned().unwrap_or_default()),
                )
                .with("variance", Value::Vec(sd.iter().map(|s| s * s).collect()))
                .with("stddev", Value::Vec(sd))
                .with(
                    "corr",
                    Value::Mat {
                        nrow: q,
                        ncol: q,
                        data: corr,
                    },
                ),
        ));
    }
    let levels = fit.opt_usizes("ranef_levels")?.unwrap_or_default();
    let groups = if levels.len() == re_groups.len() {
        Record::new()
            .with(
                "names",
                Value::StrVec(re_groups.iter().map(|(n, _)| n.clone()).collect()),
            )
            .with(
                "levels",
                Value::IntVec(levels.iter().map(|&n| n as i64).collect()),
            )
    } else {
        Record::new()
            .with("names", Value::StrVec(Vec::new()))
            .with("levels", Value::IntVec(Vec::new()))
    };

    let corr_fixed = if p >= 2 {
        let d: Vec<f64> = (0..p).map(|i| vcov[i][i].sqrt()).collect();
        Value::Mat {
            nrow: p,
            ncol: p,
            data: (0..p * p)
                .map(|k| {
                    let (i, j) = (k % p, k / p);
                    vcov[i][j] / (d[i] * d[j])
                })
                .collect(),
        }
    } else {
        Value::Null
    };

    let nagq = fit.usize("nagq")?;
    let held = fit.has("dispersion_held");
    let nan_or = |fit: &mut Inputs, name: &str| fit.opt_f64(name).map(|x| x.unwrap_or(f64::NAN));
    let rss = nan_or(&mut fit, "rss")?;
    let r_squared = nan_or(&mut fit, "r_squared")?;
    let adj_r_squared = nan_or(&mut fit, "adj_r_squared")?;
    let robust_note = robust_footnote(&df_method, re_groups.first().map(|g| g.0.as_str()));
    let power_lines = weight_power_lines(power, estimated, robust_note.is_some(), reml);
    let warnings = match fit.value("warnings") {
        Value::Null => Value::List(Vec::new()),
        w => w,
    };
    Ok(Value::Record(
        Record::new()
            .with(
                "method",
                Value::Str(method_line(&family, mixed, reml, nagq)),
            )
            .with("formula", Value::Str(fit.string("formula")?))
            .with("family", text(&family))
            .with("link", Value::Str(fit.string("link")?))
            .with("reml", Value::Bool(reml))
            .with("criterion", Value::Record(criterion))
            .with("scaled_residuals", scaled)
            .with("random_effects", Value::List(random_effects))
            .with(
                "residual_variance",
                opt_f64((mixed && family == "gaussian").then_some(dispersion)),
            )
            .with("nobs", int(nobs))
            .with("groups", Value::Record(groups))
            .with("coef_names", Value::StrVec(names))
            .with("estimate", Value::Vec(mask(beta, &aliased)))
            .with("std_error", Value::Vec(mask(se, &aliased)))
            .with("coef_df", coef_df)
            .with("statistic", Value::Vec(mask(statistic, &aliased)))
            .with("test", text(if t_test { "t" } else { "z" }))
            .with("p_value", Value::Vec(mask(p_value, &aliased)))
            .with("residual_df", residual_df)
            .with("rss", Value::F64(rss))
            .with("r_squared", Value::F64(r_squared))
            .with("adj_r_squared", Value::F64(adj_r_squared))
            .with("r2_marginal", opt_f64(sizes[0]))
            .with("r2_conditional", opt_f64(sizes[1]))
            .with("icc_adjusted", opt_f64(sizes[2]))
            .with("r2_need_refit", Value::Bool(need_refit))
            .with("aliased", Value::BoolVec(aliased))
            .with("corr_fixed", corr_fixed)
            .with(
                "dispersion_label",
                opt_text(dispersion_label(&family, held, mixed)),
            )
            .with("dispersion", Value::F64(dispersion))
            .with("n_eval", int(fit.usize("n_eval")?))
            .with("converged", Value::Bool(fit.bool("converged")?))
            .with("singular", Value::Bool(fit.bool("singular")?))
            .with("n_aliased", int(n_aliased))
            .with(
                "footnote",
                robust_note.map_or_else(
                    || opt_text((df_method == "satterthwaite").then_some(SATTERTHWAITE_FOOTNOTE)),
                    Value::Str,
                ),
            )
            .with("weight_power_lines", Value::StrVec(power_lines))
            .with("warnings", warnings),
    ))
}
