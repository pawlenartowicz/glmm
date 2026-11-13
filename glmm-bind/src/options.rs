//! The options of the registered functions, parsed and checked once for both
//! ports. Each refusal names the argument as the calling port spells it and
//! ends in the value given.

use glmm::limits::MAX_NAGQ;
use glmm::orchestrate::{FitResult, MAX_NSIM};
use glmm::WeightPower;

use crate::coerce::{self, describe};
use crate::error::BindError;
use crate::inputs::Inputs;
use crate::notes::{fmt_g, warning_record};
use crate::port::{spell, Port};
use crate::value::{Record, Value};

/// `(name, default link, links)`, the links in the order a refusal lists
/// them.
const FAMILIES: &[(&str, &str, &[&str])] = &[
    ("gaussian", "identity", &["identity"]),
    ("binomial", "logit", &["logit", "probit", "cloglog"]),
    ("poisson", "log", &["log"]),
    ("gamma", "log", &["log", "inverse"]),
    ("negativebinomial", "log", &["log"]),
    ("inversegaussian", "log", &["log", "inverse_squared"]),
];

/// The warm start as `orchestrate::run_fit` takes it: `beta` and `theta`
/// (each empty to cold-start that part) and the dispersion seed.
pub(crate) type WarmStart = (Vec<f64>, Vec<f64>, Option<f64>);

/// `fit`'s options after parsing, which the fit record echoes.
pub(crate) struct FitArgs {
    pub family: &'static str,
    pub link: &'static str,
    /// The nodes asked for.
    pub nagq: u8,
    /// The dispersion held fixed, after the families that do not read it
    /// dropped it.
    pub dispersion: Option<f64>,
    pub warm_start: Option<WarmStart>,
    /// The weight power asked for, which a fit without weights does not use.
    pub weight_power: Option<WeightPower>,
    /// The `argument_ignored` notes of the options dropped before the fit.
    pub notes: Vec<Value>,
}

fn refuse(message: String) -> BindError {
    BindError::invalid_argument(message)
}

fn ignored(message: String) -> Value {
    warning_record("argument_ignored", message)
}

/// `v` as a refusal ends in it, a null value (an option the port passed as
/// `None` / `NULL`) and a bool in the port's spelling.
pub(crate) fn shown(port: Port, v: &Value) -> String {
    match coerce::to_bool(v) {
        Ok(b) => spell(port, if b { "true" } else { "false" }).to_owned(),
        Err(_) if v.is_null() => spell(port, "null").to_owned(),
        Err(_) => describe(v),
    }
}

/// The family and link, case-insensitive, R's `negative.binomial` and
/// `inverse.gaussian` taken for the crate's names, the link the family's
/// default when absent. `formulas` are the call's formulas: inverse Gaussian
/// has no random-effect fit.
pub(crate) fn family_link(
    inp: &mut Inputs,
    formulas: &[&str],
) -> Result<(&'static str, &'static str), BindError> {
    let v = inp.value("family");
    let given = coerce::to_string(v.clone()).map_err(|_| {
        refuse(format!(
            "family must be a string, got {}",
            shown(inp.port(), &v)
        ))
    })?;
    let lower = given.to_lowercase();
    let name = match lower.as_str() {
        "negative.binomial" => "negativebinomial",
        "inverse.gaussian" => "inversegaussian",
        other => other,
    };
    let &(name, default, links) = FAMILIES.iter().find(|f| f.0 == name).ok_or_else(|| {
        let names: Vec<&str> = FAMILIES.iter().map(|f| f.0).collect();
        refuse(format!(
            "unknown family '{given}'; expected one of {}",
            names.join(", ")
        ))
    })?;
    let link = match inp.opt_string("link")? {
        None => default,
        Some(l) => links.iter().copied().find(|&k| k == l).ok_or_else(|| {
            refuse(format!(
                "family '{name}' does not support link '{l}'; expected one of {}",
                links.join(", ")
            ))
        })?,
    };
    if name == "inversegaussian" && formulas.iter().any(|f| f.contains('|')) {
        return Err(refuse(
            "family 'inversegaussian' is GLM-only: random-effect terms are not supported".into(),
        ));
    }
    Ok((name, link))
}

/// `nagq`: odd, in `1..=MAX_NAGQ`.
pub(crate) fn nagq(inp: &mut Inputs) -> Result<u8, BindError> {
    let v = inp.value("nagq");
    coerce::to_usize(&v)
        .ok()
        .and_then(|n| u8::try_from(n).ok())
        .filter(|n| (1..=MAX_NAGQ).contains(n) && n % 2 == 1)
        .ok_or_else(|| {
            refuse(format!(
                "{} must be an odd integer in 1..={MAX_NAGQ}, got {}",
                spell(inp.port(), "nagq"),
                shown(inp.port(), &v)
            ))
        })
}

/// A required bool, `shown_as` the argument as the message names it.
pub(crate) fn flag(inp: &mut Inputs, name: &str, shown_as: &str) -> Result<bool, BindError> {
    let v = inp.value(name);
    coerce::to_bool(&v).map_err(|_| {
        let port = inp.port();
        refuse(format!(
            "{shown_as} must be {} or {}, got {}",
            spell(port, "true"),
            spell(port, "false"),
            shown(port, &v)
        ))
    })
}

/// `reml`: absent, or a bool.
pub(crate) fn reml(inp: &mut Inputs) -> Result<Option<bool>, BindError> {
    let v = inp.value("reml");
    if v.is_null() {
        return Ok(None);
    }
    coerce::to_bool(&v).map(Some).map_err(|_| {
        let port = inp.port();
        refuse(format!(
            "{} must be {}, {} or {}, got {}",
            spell(port, "reml"),
            spell(port, "null"),
            spell(port, "true"),
            spell(port, "false"),
            shown(port, &v)
        ))
    })
}

/// `wald_se`: `"hessian"` or `"rx"`.
pub(crate) fn wald_se(inp: &mut Inputs) -> Result<String, BindError> {
    let v = inp.value("wald_se");
    coerce::to_string(v.clone())
        .ok()
        .filter(|s| s == "hessian" || s == "rx")
        .ok_or_else(|| {
            refuse(format!(
                "{} must be 'hessian' or 'rx', got {}",
                spell(inp.port(), "wald_se"),
                shown(inp.port(), &v)
            ))
        })
}

fn is_estimate(v: &Value) -> bool {
    coerce::to_string(v.clone()).is_ok_and(|s| s == "estimate")
}

/// `dispersion`, `init_theta` and `warm_start` of `fit`, for `family` and a
/// formula with random effects or not (`mixed`), with an `argument_ignored`
/// note for each one dropped where the family does not read it: a
/// dispersion off binomial, Poisson, Gamma and inverse Gaussian, and off a
/// mixed binomial or Poisson model; `init_theta` off negative binomial; a
/// warm-start key other than `beta`, `theta` and `dispersion`; a warm-start
/// dispersion off negative binomial and a mixed Gamma model, the only fits
/// that seed from it (`StartValues::dispersion`). A dropped value's form is not read. `"estimate"` on Gamma and inverse
/// Gaussian is their default, so nothing is held. `init_theta` fills the
/// warm start's dispersion. Only the form is checked here;
/// `orchestrate::run_fit` checks the values.
#[allow(clippy::type_complexity)]
pub(crate) fn dispersion_and_start(
    inp: &mut Inputs,
    family: &str,
    mixed: bool,
) -> Result<(Option<f64>, Option<WarmStart>, Vec<Value>), BindError> {
    let port = inp.port();
    let mut notes = Vec::new();
    let quasi = matches!(family, "binomial" | "poisson");
    let mut asked = inp.value("dispersion");
    if !asked.is_null() && !(quasi || matches!(family, "gamma" | "inversegaussian")) {
        notes.push(ignored(format!(
            "dispersion= has no effect for family '{family}'."
        )));
        asked = Value::Null;
    }
    // `Some(None)` is "estimate".
    let mut dispersion = match &asked {
        Value::Null => None,
        v if is_estimate(v) => Some(None),
        v => Some(Some(coerce::to_f64(v).map_err(|_| {
            refuse(format!(
                "dispersion must be {}, 'estimate', or a number, got {}",
                spell(port, "null"),
                shown(port, v)
            ))
        })?)),
    };
    if quasi && mixed && dispersion.is_some() {
        notes.push(ignored(
            "Quasi-likelihood dispersion= is not supported yet for binomial or Poisson models. \
             The default dispersion of 1 was used."
                .into(),
        ));
        dispersion = None;
    }
    let mut init_theta = inp.value("init_theta");
    if !init_theta.is_null() && family != "negativebinomial" {
        notes.push(ignored(format!(
            "{}= is not used for family '{family}'.",
            spell(port, "init_theta")
        )));
        init_theta = Value::Null;
    }
    let warm = match inp.value("warm_start") {
        Value::Null => None,
        Value::Record(r) => Some(r),
        // R's `list()` carries no names.
        Value::List(x) if x.is_empty() => Some(Record::new()),
        other => {
            return Err(refuse(format!(
                "{} must be {} 'beta', 'theta' and/or 'dispersion' (theta is the random-effect \
                 Cholesky vector, not the negative-binomial shape - that is {} or dispersion), \
                 got {}",
                spell(port, "warm_start"),
                spell(port, "warm_start_record"),
                spell(port, "init_theta"),
                shown(port, &other)
            )))
        }
    };
    if let Some(w) = &warm {
        let unknown: Vec<&str> = w
            .keys()
            .filter(|k| !matches!(*k, "beta" | "theta" | "dispersion"))
            .collect();
        if !unknown.is_empty() {
            notes.push(ignored(format!(
                "{} accepts only 'beta', 'theta' and 'dispersion'; these entries were ignored: \
                 {}.",
                spell(port, "warm_start"),
                unknown.join(", ")
            )));
        }
    }
    if quasi && dispersion.is_some() {
        return Err(refuse(format!(
            "quasi-likelihood dispersion on family '{family}' is not yet implemented in the kernel"
        )));
    }
    let Some(mut warm) = warm.or_else(|| (!init_theta.is_null()).then(Record::new)) else {
        return Ok((dispersion.flatten(), None, notes));
    };
    let mut entry = |key: &str| -> Result<Vec<f64>, BindError> {
        match warm.take(key) {
            Value::Null => Ok(Vec::new()),
            v => coerce::to_vec(v.clone()).map_err(|_| {
                refuse(format!(
                    "{} entry '{key}' must be a numeric vector, got {}",
                    spell(port, "warm_start"),
                    shown(port, &v)
                ))
            }),
        }
    };
    let beta = entry("beta")?;
    let theta = entry("theta")?;
    let mut seed = warm.take("dispersion");
    if !seed.is_null() && !(family == "negativebinomial" || (family == "gamma" && mixed)) {
        notes.push(ignored(format!(
            "{} is not used for family '{family}'.",
            spell(port, "warm_start_dispersion")
        )));
        seed = Value::Null;
    }
    if !init_theta.is_null() {
        if !seed.is_null() {
            return Err(refuse(format!(
                "give the negative-binomial shape seed once: {}= or {}, not both",
                spell(port, "init_theta"),
                spell(port, "warm_start_dispersion")
            )));
        }
        seed = init_theta;
    }
    let seed = match seed {
        Value::Null => None,
        v => Some(coerce::to_f64(&v).map_err(|_| {
            refuse(format!(
                "{} (or {}) must be one number, got {}",
                spell(port, "warm_start_dispersion"),
                spell(port, "init_theta"),
                shown(port, &v)
            ))
        })?),
    };
    Ok((dispersion.flatten(), Some((beta, theta, seed)), notes))
}

/// The fit record's `argument_notes`: the notes of `args`, then those of
/// the options the fit found it did not use, a weight power without
/// weights, `nagq` where nothing is approximated or adaptive quadrature does
/// not apply (`agq_fallback`), and `reml` off a Gaussian model.
pub(crate) fn argument_notes(port: Port, args: &FitArgs, r: &FitResult) -> Vec<Value> {
    let mut notes = args.notes.clone();
    if let Some(WeightPower::Fixed(v)) = args.weight_power {
        if v != 1.0 && r.weight_power.is_nan() {
            notes.push(ignored(format!(
                "weight_power = {} has no effect without weights.",
                fmt_g(v, 7)
            )));
        }
    }
    if r.agq_warning {
        let nagq = format!("{}={}", spell(port, "nagq"), args.nagq);
        let reason = if args.family == "gaussian" {
            Some("a Gaussian model")
        } else if r.re_groups.is_empty() {
            Some("a model without random effects")
        } else {
            None
        };
        notes.push(match reason {
            Some(reason) => ignored(format!(
                "{nagq} has no effect for {reason}, because nothing is approximated."
            )),
            None => warning_record(
                "agq_fallback",
                format!(
                    "{nagq} works only for binomial, Poisson, negative-binomial or Gamma models \
                     whose random effects are in one grouping factor, with at most 3 random \
                     effects in it. This model was fitted without adaptive quadrature."
                ),
            ),
        });
    }
    if r.reml_warning {
        notes.push(ignored(format!(
            "{} applies only to Gaussian models; it was ignored and the model was fitted by \
             maximum likelihood.",
            spell(port, "reml_true")
        )));
    }
    notes
}

/// `nsim`: a whole number in `1..=MAX_NSIM`.
pub(crate) fn nsim(inp: &mut Inputs) -> Result<usize, BindError> {
    let v = inp.value("nsim");
    coerce::to_usize(&v)
        .ok()
        .filter(|n| (1..=MAX_NSIM).contains(n))
        .ok_or_else(|| {
            refuse(format!(
                "nsim must be a whole number from 1 to {MAX_NSIM}, got {}",
                shown(inp.port(), &v)
            ))
        })
}

/// `nsim`: absent, or as [`nsim`].
pub(crate) fn opt_nsim(inp: &mut Inputs) -> Result<Option<usize>, BindError> {
    if inp.has("nsim") {
        nsim(inp).map(Some)
    } else {
        Ok(None)
    }
}

/// 2^53, the largest whole number an R double carries exactly; R has no
/// 64-bit integer, so its seeds stop here.
const R_SEED_MAX: u64 = 1 << 53;

/// `seed`: a whole number in the port's range.
pub(crate) fn seed(inp: &mut Inputs) -> Result<u64, BindError> {
    let v = inp.value("seed");
    let port = inp.port();
    coerce::to_seed(&v)
        .ok()
        .filter(|&s| port != Port::R || s <= R_SEED_MAX)
        .ok_or_else(|| {
            refuse(format!(
                "seed must be {}, got {}",
                spell(port, "seed_range"),
                shown(port, &v)
            ))
        })
}

/// `seed`: absent, or as [`seed`].
pub(crate) fn opt_seed(inp: &mut Inputs) -> Result<Option<u64>, BindError> {
    if inp.has("seed") {
        seed(inp).map(Some)
    } else {
        Ok(None)
    }
}

/// A string option, one of `allowed`, `shown_as` the argument as the
/// message names it. A string given is quoted as it was typed.
pub(crate) fn choice(
    inp: &mut Inputs,
    name: &str,
    shown_as: &str,
    allowed: &[&str],
) -> Result<String, BindError> {
    let v = inp.value(name);
    match coerce::to_string(v.clone()) {
        Ok(s) if allowed.contains(&s.as_str()) => Ok(s),
        given => {
            let given = given.map_or_else(|_| shown(inp.port(), &v), |s| format!("{s:?}"));
            let quoted: Vec<String> = allowed.iter().map(|a| format!("{a:?}")).collect();
            let (last, rest) = quoted.split_last().expect("a choice has options");
            Err(refuse(format!(
                "{shown_as} must be {} or {last}, got {given}",
                rest.join(", ")
            )))
        }
    }
}

/// `re_form`: absent for the random effects of every grouping, `"NA"` for
/// the fixed effects only; true for the latter.
pub(crate) fn population(inp: &mut Inputs) -> Result<bool, BindError> {
    match inp.value("re_form") {
        Value::Null => Ok(false),
        v if coerce::to_string(v.clone()).is_ok_and(|s| s == "NA") => Ok(true),
        v => {
            let port = inp.port();
            Err(refuse(format!(
                "{} must be {} (the random effects of every grouping) or {} (the fixed effects \
                 only), got {}",
                spell(port, "re_form"),
                spell(port, "null"),
                spell(port, "na"),
                shown(port, &v)
            )))
        }
    }
}
