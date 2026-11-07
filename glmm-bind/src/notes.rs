//! The text of every warning a fit raises: each kind's tier and title, the
//! message of each kernel note, and the one severe warning of a fit that did
//! not converge. Both ports raise these as given; each keeps only its own
//! warning class per kind and the warnings that spell its own arguments.
//! `documentation/warnings.md` documents the same table and text — change
//! together.

use glmm::orchestrate::NoteInfo;

use crate::error::BindError;
use crate::inputs::Inputs;
use crate::records::text;
use crate::value::{Record, Value};

/// `(kind, tier, title)`, one fixed pair per kind.
const KINDS: &[(&str, &str, &str)] = &[
    ("search_limit", "severe", "Search stopped at its step limit"),
    ("fit_failed", "severe", "Fit failed"),
    ("glm_diverged", "severe", "Fit diverged"),
    (
        "design_unsolvable",
        "severe",
        "Predictors could not be separated",
    ),
    ("constant_response", "severe", "Response does not vary"),
    ("too_few_rows", "severe", "Too few rows"),
    ("no_coefficients", "severe", "Nothing to estimate"),
    (
        "pirls_exhausted",
        "caution",
        "Last fitting step did not finish",
    ),
    (
        "nb_shape_unsettled",
        "caution",
        "Shape search did not settle",
    ),
    ("singular", "caution", "Singular fit"),
    ("ill_conditioned", "caution", "Nearly collinear columns"),
    (
        "hessian_se_fallback",
        "caution",
        "Simpler standard errors used",
    ),
    ("se_unavailable", "caution", "Standard errors not computed"),
    (
        "exact_profile_fallback",
        "caution",
        "Search retried with a different method",
    ),
    (
        "gradient_search_fallback",
        "caution",
        "Search finished by another method",
    ),
    (
        "precision_limited",
        "caution",
        "Numbers too large for full precision",
    ),
    ("agq_fallback", "caution", "Adaptive quadrature not used"),
    ("non_integer_response", "caution", "Non-integer response"),
    (
        "rows_dropped_na",
        "caution",
        "Rows dropped for missing values",
    ),
    (
        "satterthwaite_curvature",
        "caution",
        "Satterthwaite df uncertain",
    ),
    (
        "few_rows_for_random_effects",
        "caution",
        "Too few rows for the random effects",
    ),
    (
        "variance_at_limit",
        "caution",
        "Variance at its search limit",
    ),
    (
        "weight_power_at_bound",
        "caution",
        "Weight power at its search limit",
    ),
    (
        "weight_power_search_failures",
        "caution",
        "Weight power search had failed fits",
    ),
    (
        "weight_power_implausible",
        "caution",
        "Implausible weight power",
    ),
    (
        "robust_with_random_slope",
        "caution",
        "Robust standard errors with a random slope",
    ),
    (
        "robust_se_unidentified",
        "caution",
        "Robust standard error not identified",
    ),
    ("robust_df_small", "caution", "Robust df below 4"),
    ("argument_ignored", "note", "Argument ignored"),
    (
        "single_level_grouping_dropped",
        "note",
        "Random effect dropped (single level)",
    ),
    (
        "weight_power_outside_unit",
        "note",
        "Weight power outside [0, 1]",
    ),
];

/// The tier and title of a kind this table has no entry for.
const UNKNOWN: (&str, &str) = ("caution", "Unrecognized solver message");

/// `kind`'s `(tier, title)`.
pub(crate) fn tier_title(kind: &str) -> (&'static str, &'static str) {
    KINDS
        .iter()
        .find(|k| k.0 == kind)
        .map_or(UNKNOWN, |&(_, tier, title)| (tier, title))
}

/// One warning as the ports store it: `tier`, `kind`, `title`, `message`.
pub(crate) fn warning_record(kind: &str, message: String) -> Value {
    let (tier, title) = tier_title(kind);
    Value::Record(
        Record::new()
            .with("tier", text(tier))
            .with("kind", text(kind))
            .with("title", text(title))
            .with("message", Value::Str(message)),
    )
}

/// `warning_kind`: the `tier` and `title` of `kind`, for a warning the port
/// words itself.
pub(crate) fn warning_kind(inp: &mut Inputs) -> Result<Value, BindError> {
    let (tier, title) = tier_title(&inp.string("kind")?);
    Ok(Value::Record(
        Record::new()
            .with("tier", text(tier))
            .with("title", text(title)),
    ))
}

/// C printf's `%.{prec}g` (Python's `{:.{prec}g}`): `prec` significant
/// digits, exponent form below 1e-4 or from 10^prec on, trailing zeros
/// dropped; `nan`, `inf` and `-inf` as Python spells them.
pub(crate) fn fmt_g(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    let p = prec.max(1);
    // The decimal exponent after rounding to `p` digits, as %e reports it.
    let e = format!("{x:.*e}", p - 1);
    let (mantissa, exp) = e.split_once('e').unwrap_or((&e, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let trim = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s.to_owned()
        }
    };
    if (-4..p as i32).contains(&exp) {
        let decimals = (p as i32 - 1 - exp) as usize;
        trim(&format!("{x:.decimals$}"))
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", trim(mantissa), exp.abs())
    }
}

fn count(n: usize, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else {
        format!("{n} {word}s")
    }
}

/// What a note's message reads besides the note: the coefficient names, the
/// fit's convergence and row count, and how a column index past `names`
/// prints (`row_base` 0 from Python, 1 from R).
pub(crate) struct FitText<'a> {
    pub names: &'a [String],
    pub converged: bool,
    pub nobs: usize,
    pub row_base: usize,
}

impl FitText<'_> {
    /// `x` as [`fmt_g`] prints it, except that the R port (`row_base` 1)
    /// gets R's `NaN`, `Inf` and `-Inf`.
    fn num(&self, x: f64, prec: usize) -> String {
        match x {
            _ if self.row_base == 1 && x.is_nan() => "NaN".to_owned(),
            _ if self.row_base == 1 && x.is_infinite() => {
                if x > 0.0 { "Inf" } else { "-Inf" }.to_owned()
            }
            _ => fmt_g(x, prec),
        }
    }

    /// The names of 0-based `columns`, joined by ", ".
    fn columns(&self, columns: &[u32]) -> String {
        columns
            .iter()
            .map(|&i| {
                let i = i as usize;
                self.names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("column {}", i + self.row_base))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The message of each of a fit's notes, `None` for a note that raises
/// nothing. A fit with both `se_unavailable` and `ill_conditioned`, whose
/// named estimates are NaN (the GLM's breakdown), prints one warning: the
/// `se_unavailable` text naming the `ill_conditioned` note's columns. A GLMM
/// keeps that estimate, so it prints the two notes as they are.
pub(crate) fn note_messages(notes: &[NoteInfo], beta: &[f64], cx: &FitText) -> Vec<Option<String>> {
    let ill = notes.iter().find(|n| n.kind == "ill_conditioned");
    let paired = ill.is_some_and(|ill| {
        notes.iter().any(|n| n.kind == "se_unavailable")
            && ill
                .columns
                .iter()
                .all(|&i| beta.get(i as usize).is_some_and(|b| b.is_nan()))
    });
    notes
        .iter()
        .map(|n| match (paired, n.kind, ill) {
            (true, "ill_conditioned", _) => None,
            (true, "se_unavailable", Some(ill)) => note_message(n, &ill.columns, cx),
            _ => note_message(n, &n.columns, cx),
        })
        .collect()
}

/// One note's message, `columns` standing for its own; `None` to raise
/// nothing. The `kind`, not this text, is the stable identifier.
fn note_message(n: &NoteInfo, columns: &[u32], cx: &FitText) -> Option<String> {
    let s = match n.kind {
        // Only on a non-converged fit, whose one warning (`failure`) names it.
        "separated" => return None,
        "ill_conditioned" => format!(
            "{} is almost a combination of other columns in the model, so its standard error \
             is large. Its estimate is still correct, but imprecise. The other columns involved \
             are not named. Consider dropping or combining predictors that carry the same \
             information.",
            cx.columns(columns)
        ),
        // Raised only when the final re-evaluation of a converged fit hit the
        // cap. A rejected trial point changes no reported number and stays in
        // the notes; on a non-converged fit the cap-out is one sentence of the
        // failure warning instead.
        "pirls_exhausted" if n.final_eval && cx.converged => {
            "The final step that computes the reported results ran out of iterations. The \
             estimates and their standard errors may be less accurate than usual. Try \
             simplifying the random effects."
                .to_owned()
        }
        "pirls_exhausted" => return None,
        "nb_shape_unsettled" => format!(
            "The search for the negative binomial shape parameter stopped at its limit of {} \
             rounds before it settled. The coefficients and standard errors are computed at the \
             last value it reached, which may not be the best one.",
            n.evals
        ),
        "single_level_grouping_dropped" => format!(
            "Grouping factor '{}' has only one level, so no variance between groups can be \
             estimated from it. Its random effect was dropped; the rest of the model was fitted \
             without it.",
            n.detail
        ),
        // `detail` is the term as "<terms> | <grouping>" and `evals` its
        // random-effect count; the rows are the fit's own.
        "few_rows_for_random_effects" => format!(
            "The term ({}) has {} random effects for {}, so its variance parameters probably \
             cannot be estimated from these data; lme4 stops on such a model by default. The \
             estimates are shown; read the variance components with care, or simplify the \
             random-effects structure.",
            n.detail,
            n.evals,
            count(cx.nobs, "row")
        ),
        "variance_at_limit" => "A random-effect parameter reached the upper limit of the search \
             (1000 on glmm's internal scale). On a Gaussian model this usually means the residual \
             variance is estimated at or near zero; on other families the random-effect variance \
             has no finite estimate. The estimates are at that limit, not at an optimum. Check \
             for a response that the predictors and groups explain exactly."
            .to_owned(),
        "hessian_se_fallback" => "The usual standard errors could not be computed, so a simpler \
             method was used. Its standard errors tend to be too small, so p-values and \
             confidence intervals may look more precise than they are. Standard errors for the \
             random-effect standard deviations are not available."
            .to_owned(),
        "se_unavailable" if !columns.is_empty() => format!(
            "Standard errors could not be computed: {} cannot be separated from the other \
             columns in the model. Its estimate is not reported. The other estimates and the \
             deviance can be used; standard errors, p-values and confidence intervals are not \
             reported. Check for predictors that are copies or near-copies of each other.",
            cx.columns(columns)
        ),
        "se_unavailable" => "Standard errors could not be computed at the fit's end point: the \
             fixed effects carry too little separate information. The estimates are usable; \
             standard errors, p-values and confidence intervals are not reported. Check for \
             predictors that are copies or near-copies of each other."
            .to_owned(),
        "gradient_search_fallback" => "The fast search for this model did not settle on an \
             answer, so the fit finished with the standard search from the best point it had \
             reached. The reported estimates come from that search."
            .to_owned(),
        "exact_profile_fallback" => "The default search method for this model did not settle on \
             an answer, so the fit tried a different search method from the same starting point. \
             The reported estimates come from whichever method reached the better answer."
            .to_owned(),
        "precision_limited" => "The numbers in this fit are so large that rounding in the \
             computer's arithmetic hides small changes in its deviance. The random-effect \
             estimates, and the fixed effects that depend on them, may be off by a tenth of a \
             standard error or more. This happens with counts in the billions or more, or when a \
             model misses a large effect in such data; on a Gaussian model, with a response far \
             from zero next to its residual spread (subtract a constant from it), or with \
             weights spanning many orders of magnitude."
            .to_owned(),
        // `evals` carries the affected row count.
        "non_integer_response" => format!(
            "The response is not a whole number in {}. This family expects integer counts \
             (binomial: the number of successes, y * weights). The fit still ran, but check \
             whether this is the right family for the data.",
            count(n.evals as usize, "row")
        ),
        "satterthwaite_curvature" => satterthwaite_message(&n.detail),
        "weight_power_at_bound" => format!(
            "The fitted weight power, {}, is not a maximum: the log-likelihood rises, or stays \
             flat, toward the end of the search range [{}, {}]. This happens when at most as \
             many rows as there are coefficients differ from the common weight, when the weights \
             vary little, or when they are so large or small (beyond about 1e8 or 1e-8) that the \
             range stops a larger power from overflowing. Weights that large or small can be \
             divided by their mean and the model refitted: the weight power does not depend on \
             their scale. Otherwise consider weight_power = 1 (the weights as given) or \
             weight_power = 0 (the weights ignored).",
            cx.num(n.delta, 4),
            cx.num(n.lower, 4),
            cx.num(n.upper, 4)
        ),
        "weight_power_search_failures" => format!(
            "In the search for the weight power, {} failed and {} did not converge. A failed \
             fit counts as the worst value there, which steers the search toward 0 (the weights \
             ignored). The reported fit is the one at the power found; check it, or use \
             weight_power = 1.",
            count(n.failed as usize, "fit"),
            count(n.not_converged as usize, "fit")
        ),
        "weight_power_outside_unit" => format!(
            "The fitted weight power, {}, is outside [0, 1] (0: the weights carry no information \
             on the variance; 1: the weights as given). With few rows or a weak fit (low R\u{b2}) \
             this happens by chance and is not by itself a problem.",
            cx.num(n.delta, 4)
        ),
        "weight_power_implausible" => format!(
            "The fitted weight power, {}, is outside [-1, 2], the range the simulations behind \
             it covered, and the fitted variances differ from the weights by up to {} times \
             across the data. Check the weights, or fix the power with weight_power = 1 (the \
             weights as given).",
            cx.num(n.delta, 4),
            cx.num(n.factor, 4)
        ),
        "robust_with_random_slope" => "Robust standard errors were asked for on a model with a \
             random slope. Their arithmetic is checked there, but in simulations with 10 groups \
             their Type I error ran at .056 to .066; the model-based standard errors (se_type = \
             \"model\") are the default on this model."
            .to_owned(),
        "robust_se_unidentified" => format!(
            "{}: one group alone carries the information on this coefficient, so its robust \
             standard error and df are not computed (NaN). That group has no residual left to \
             measure it with. More groups are needed; se_type = \"model\" assumes the groups \
             share one variance.",
            cx.columns(columns)
        ),
        "robust_df_small" => {
            let cols = columns
                .iter()
                .zip(&n.df)
                .map(|(&c, &d)| format!("{} ({})", cx.columns(&[c]), cx.num(d, 3)))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "The robust degrees of freedom of {cols} are below 4, where the robust test \
                 gives no assurance on its error rate. If such a predictor is constant within \
                 groups, se_type = \"model\" is the better route."
            )
        }
        other => format!(
            "The solver reported something ('{other}') that this installed version does not \
             recognize. Please report it at https://github.com/pawlenartowicz/glmm/issues."
        ),
    };
    Some(s)
}

/// The `satterthwaite_curvature` message from the kernel's detail string,
/// "dropped <n>, negative <n>, converged <bool>, nonfinite <bool>".
fn satterthwaite_message(detail: &str) -> String {
    let field = |name: &str| {
        detail
            .split(", ")
            .find_map(|part| part.strip_prefix(name)?.strip_prefix(' '))
            .unwrap_or("")
    };
    if field("nonfinite") == "true" {
        return "The Satterthwaite degrees of freedom could not be computed: the fit has no \
                finite end point, or the deviance near it was not finite. Every coefficient is \
                tested with z, and its confidence interval uses the normal distribution."
            .to_owned();
    }
    let dropped: u64 = field("dropped").parse().unwrap_or(0);
    let negative: u64 = field("negative").parse().unwrap_or(0);
    let mut parts = Vec::new();
    if dropped > 0 {
        parts.push(format!(
            "the curvature of the deviance in the variance parameters has {}, left out",
            count(dropped as usize, "flat direction")
        ));
    }
    if negative > 0 {
        parts.push(format!(
            "it has {}, so the fit may not be at a minimum",
            count(negative as usize, "negative direction")
        ));
    }
    if field("converged") != "true" {
        parts.push(
            "the variance search stopped at its step limit, so they are computed at its last \
             point"
                .to_owned(),
        );
    }
    format!(
        "The Satterthwaite degrees of freedom are approximate: {}. The t tests and confidence \
         intervals use them as computed.",
        parts.join("; ")
    )
}

const INNER_STEPS: &str = "Some of its inner steps ran out of iterations.";

// glm_diverged messages by family: separation only means something for a
// binomial response, and the Gamma and inverse-Gaussian fits skip the
// linear-predictor check (glmm-rs/src/glm.rs), so theirs cannot be called a
// divergence to an extreme.
const DIVERGED_BINOMIAL: &str = "The fit did not converge. This usually means a predictor, or a \
    combination of predictors, predicts the outcome perfectly (separation), so some fitted \
    probabilities go to 0 or 1. The coefficients are from the last step; standard errors are not \
    reported. Check the data for separation.";
const DIVERGED_COUNTS: &str = "The fit did not converge. This usually means that some category \
    of a predictor, or some combination of predictors, has only zero counts, so some fitted \
    counts go to 0. The coefficients are from the last step; standard errors are not reported. \
    Check for categories whose counts are all zero.";
const DIVERGED_CONTINUOUS: &str = "The fit did not settle on an answer: the fitting steps \
    stopped before converging. The coefficients are from the last step; standard errors are not \
    reported. Check predictors with extreme values; with a link other than log, the log link is \
    usually more stable.";

// A response far outside the reciprocal links' range: the kernel holds the
// Gamma 1/mu and inverse-Gaussian 1/mu^2 linear predictor inside
// [1e-75, 1e75] (`INV_ETA_MIN` / `INV_ETA_MAX` in glmm-rs/src/family.rs —
// change together), and a fit whose means must sit outside it cannot converge.
const ETA_RANGE: &str = "Some response values are so large or so small that the linear \
    predictor of this link (1/mu for gamma, 1/mu^2 for inverse Gaussian) leaves the range the \
    solver supports, 1e-75 to 1e75. Rescale the response, for example divide it by a power of \
    ten, or use the log link.";

/// Whether some positive response value puts the reciprocal link's
/// eta = 1/y (gamma) or 1/y^2 (inverse Gaussian) outside [1e-75, 1e75].
fn eta_out_of_range(family: &str, link: &str, y: &[f64]) -> bool {
    let power = match (family, link) {
        ("gamma", "inverse") => 1,
        ("inversegaussian", "inverse_squared") => 2,
        _ => return false,
    };
    y.iter().filter(|&&v| v > 0.0).any(|&v| {
        let eta = 1.0 / v.powi(power);
        !(1e-75..=1e75).contains(&eta)
    })
}

/// What [`failure`] reads off a fit that did not converge.
pub(crate) struct Failed<'a> {
    pub family: &'a str,
    pub link: &'a str,
    /// Whether the fitted model kept a random effect.
    pub mixed: bool,
    pub notes: &'a [NoteInfo],
    pub beta: &'a [f64],
    pub aliased: &'a [bool],
    pub deviance: f64,
    pub y: &'a [f64],
}

/// The one severe warning of a fit that did not converge, as `(kind,
/// message)`. The kernel does not say which stopping rule fired, so the cause
/// is read off what the fit reports, most specific first: no coefficient, too
/// few rows (a one-row response is trivially constant, so this comes first),
/// a constant response, then the model. A model with random effects whose
/// fixed-effects-only GLM stopped on separation (the `separated` note) never
/// ran its search; its coefficients are that GLM's last step, so it gets the
/// GLM's message. Otherwise, with random effects, a finite deviance means the
/// kernel reached an end point (the budget stop, reported at its best point),
/// and finite estimated coefficients (aliased slots are NaN by contract)
/// confirm it; anything else failed. The deviance is what decides a model
/// with no fixed effects. A GLMM inner cap-out during the search is one extra
/// sentence rather than a second warning. On the reciprocal links a response
/// outside their range names that cause instead of the general advice.
pub(crate) fn failure(f: &Failed) -> (&'static str, String) {
    let estimated = || {
        f.beta
            .iter()
            .zip(f.aliased)
            .filter(|(_, &a)| !a)
            .map(|(&b, _)| b)
    };
    let n_est = f.aliased.iter().filter(|&&a| !a).count();
    if !f.mixed && n_est == 0 {
        return (
            "no_coefficients",
            "The model has no coefficients and no random effects, so there is nothing to \
             estimate. Add an intercept or a predictor."
                .to_owned(),
        );
    }
    if f.y.len() <= n_est {
        return (
            "too_few_rows",
            format!(
                "The model has {} to estimate but only {}, so no estimates were computed. Use \
                 more rows or fewer predictors.",
                count(n_est, "coefficient"),
                count(f.y.len(), "row")
            ),
        );
    }
    if let Some(&y0) = f.y.first() {
        if f.y.iter().all(|&v| v == y0) {
            return (
                "constant_response",
                format!(
                    "Every value of the response is {}, so there is nothing to estimate. Check \
                     the response column.",
                    fmt_g(y0, 6)
                ),
            );
        }
    }
    let has = |kind: &str| f.notes.iter().any(|n| n.kind == kind);
    let eta_range = eta_out_of_range(f.family, f.link, f.y);
    if !f.mixed || has("separated") {
        // A GLM whose first solve failed took no step: no coefficient at all.
        if f.family == "gaussian" || (!f.mixed && estimated().all(f64::is_nan)) {
            return (
                "design_unsolvable",
                "The predictors could not be separated numerically, so no estimates were \
                 computed. Check for predictors that are copies or near-copies of each other."
                    .to_owned(),
            );
        }
        let message = match f.family {
            "binomial" => DIVERGED_BINOMIAL.to_owned(),
            "poisson" | "negativebinomial" => DIVERGED_COUNTS.to_owned(),
            _ if eta_range => format!(
                "The fit did not converge. The coefficients are from the last step; standard \
                 errors are not reported. {ETA_RANGE}"
            ),
            _ => DIVERGED_CONTINUOUS.to_owned(),
        };
        return ("glm_diverged", message);
    }
    let inner = if has("pirls_exhausted") {
        format!(" {INNER_STEPS}")
    } else {
        String::new()
    };
    let advice = "Try a simpler random-effects structure or check for nearly collinear predictors.";
    if f.deviance.is_finite() && estimated().all(f64::is_finite) {
        return (
            "search_limit",
            format!(
                "The search for the variance estimates reached its step limit before it \
                 settled.{inner} The estimates shown are the best point found; they are often \
                 close, but this is not checked. Do not use them until the fit converges. {advice}"
            ),
        );
    }
    let cause = if eta_range { ETA_RANGE } else { advice };
    (
        "fit_failed",
        format!("The fitting algorithm failed and returned no estimates.{inner} {cause}"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_g_matches_python() {
        // Expected values are Python's format(x, ".{p}g").
        let cases: &[(f64, usize, &str)] = &[
            (0.0, 4, "0"),
            (-0.0, 6, "-0"),
            (1.0, 4, "1"),
            (0.5, 4, "0.5"),
            (1234.5, 4, "1234"),
            (12345.0, 4, "1.234e+04"),
            (0.00012345, 4, "0.0001234"),
            (0.000012345, 4, "1.234e-05"),
            (1e-5, 3, "1e-05"),
            (123456.0, 6, "123456"),
            (1234567.0, 6, "1.23457e+06"),
            (2.5, 1, "2"),
            (0.125, 2, "0.12"),
            (-1.23456e-7, 3, "-1.23e-07"),
            (1e100, 4, "1e+100"),
            (9.9999, 4, "10"),
            (99999.5, 4, "1e+05"),
            (f64::NAN, 4, "nan"),
            (f64::INFINITY, 4, "inf"),
            (f64::NEG_INFINITY, 4, "-inf"),
        ];
        for &(x, p, want) in cases {
            assert_eq!(fmt_g(x, p), want, "{x:e} at {p}");
        }
    }

    fn note(kind: &'static str) -> NoteInfo {
        NoteInfo {
            kind,
            columns: Vec::new(),
            pivot: f64::NAN,
            evals: 0,
            final_eval: false,
            detail: String::new(),
            delta: f64::NAN,
            factor: f64::NAN,
            lower: f64::NAN,
            upper: f64::NAN,
            failed: 0,
            not_converged: 0,
            df: Vec::new(),
        }
    }

    const NO_NAMES: FitText = FitText {
        names: &[],
        converged: true,
        nobs: 0,
        row_base: 0,
    };

    fn message(n: &NoteInfo, cx: &FitText) -> Option<String> {
        note_messages(std::slice::from_ref(n), &[], cx).remove(0)
    }

    #[test]
    fn r_spells_a_non_finite_number_as_r_does() {
        let n = NoteInfo {
            delta: f64::NAN,
            ..note("weight_power_outside_unit")
        };
        let r = FitText {
            row_base: 1,
            ..NO_NAMES
        };
        let (py, r) = (message(&n, &NO_NAMES).unwrap(), message(&n, &r).unwrap());
        assert!(py.starts_with("The fitted weight power, nan, "), "{py}");
        assert!(r.starts_with("The fitted weight power, NaN, "), "{r}");
    }

    #[test]
    fn constructed_notes_print_their_text() {
        let cases = [
            (
                NoteInfo {
                    detail: "onegroup".into(),
                    ..note("single_level_grouping_dropped")
                },
                "Grouping factor 'onegroup' has only one level, so no variance between groups \
                 can be estimated from it. Its random effect was dropped; the rest of the model \
                 was fitted without it.",
            ),
            (
                note("hessian_se_fallback"),
                "The usual standard errors could not be computed, so a simpler method was used. \
                 Its standard errors tend to be too small, so p-values and confidence intervals \
                 may look more precise than they are. Standard errors for the random-effect \
                 standard deviations are not available.",
            ),
            (
                note("se_unavailable"),
                "Standard errors could not be computed at the fit's end point: the fixed effects \
                 carry too little separate information. The estimates are usable; standard \
                 errors, p-values and confidence intervals are not reported. Check for \
                 predictors that are copies or near-copies of each other.",
            ),
            (
                note("precision_limited"),
                "The numbers in this fit are so large that rounding in the computer's arithmetic \
                 hides small changes in its deviance. The random-effect estimates, and the fixed \
                 effects that depend on them, may be off by a tenth of a standard error or more. \
                 This happens with counts in the billions or more, or when a model misses a \
                 large effect in such data; on a Gaussian model, with a response far from zero \
                 next to its residual spread (subtract a constant from it), or with weights \
                 spanning many orders of magnitude.",
            ),
            (
                note("gradient_search_fallback"),
                "The fast search for this model did not settle on an answer, so the fit finished \
                 with the standard search from the best point it had reached. The reported \
                 estimates come from that search.",
            ),
            (
                note("exact_profile_fallback"),
                "The default search method for this model did not settle on an answer, so the \
                 fit tried a different search method from the same starting point. The reported \
                 estimates come from whichever method reached the better answer.",
            ),
            (
                NoteInfo {
                    final_eval: true,
                    ..note("pirls_exhausted")
                },
                "The final step that computes the reported results ran out of iterations. The \
                 estimates and their standard errors may be less accurate than usual. Try \
                 simplifying the random effects.",
            ),
            (
                NoteInfo {
                    evals: 25,
                    ..note("nb_shape_unsettled")
                },
                "The search for the negative binomial shape parameter stopped at its limit of 25 \
                 rounds before it settled. The coefficients and standard errors are computed at \
                 the last value it reached, which may not be the best one.",
            ),
            (
                NoteInfo {
                    detail: "dropped 0, negative 0, converged false, nonfinite true".into(),
                    ..note("satterthwaite_curvature")
                },
                "The Satterthwaite degrees of freedom could not be computed: the fit has no \
                 finite end point, or the deviance near it was not finite. Every coefficient is \
                 tested with z, and its confidence interval uses the normal distribution.",
            ),
            (
                NoteInfo {
                    detail: "dropped 1, negative 2, converged false, nonfinite false".into(),
                    ..note("satterthwaite_curvature")
                },
                "The Satterthwaite degrees of freedom are approximate: the curvature of the \
                 deviance in the variance parameters has 1 flat direction, left out; it has 2 \
                 negative directions, so the fit may not be at a minimum; the variance search \
                 stopped at its step limit, so they are computed at its last point. The t tests \
                 and confidence intervals use them as computed.",
            ),
            // Left to fits in the weight-power search that do not converge.
            (
                NoteInfo {
                    not_converged: 2,
                    ..note("weight_power_search_failures")
                },
                "In the search for the weight power, 0 fits failed and 2 fits did not converge. \
                 A failed fit counts as the worst value there, which steers the search toward 0 \
                 (the weights ignored). The reported fit is the one at the power found; check \
                 it, or use weight_power = 1.",
            ),
            (
                note("from_the_future"),
                "The solver reported something ('from_the_future') that this installed version \
                 does not recognize. Please report it at \
                 https://github.com/pawlenartowicz/glmm/issues.",
            ),
        ];
        for (n, want) in &cases {
            assert_eq!(message(n, &NO_NAMES).as_deref(), Some(*want), "{}", n.kind);
        }
        assert_eq!(tier_title("from_the_future"), UNKNOWN);
    }

    #[test]
    fn se_unavailable_with_ill_conditioned_prints_one_warning() {
        let names = ["(Intercept)".to_owned(), "x".to_owned(), "x2".to_owned()];
        let cx = FitText {
            names: &names,
            ..NO_NAMES
        };
        let ill = NoteInfo {
            columns: vec![2],
            ..note("ill_conditioned")
        };
        let pair = [ill, note("se_unavailable")];
        let nan_x2 = [1.0, 2.0, f64::NAN];
        assert_eq!(
            note_messages(&pair, &nan_x2, &cx),
            [
                None,
                Some(
                    "Standard errors could not be computed: x2 cannot be separated from the \
                     other columns in the model. Its estimate is not reported. The other \
                     estimates and the deviance can be used; standard errors, p-values and \
                     confidence intervals are not reported. Check for predictors that are \
                     copies or near-copies of each other."
                        .to_owned()
                )
            ]
        );
        // A GLMM keeps the estimate: both notes print as they are.
        let kept = note_messages(&pair, &[1.0, 2.0, 3.0], &cx);
        assert!(kept[0].as_deref().unwrap().starts_with("x2 is almost"));
        assert!(kept[1]
            .as_deref()
            .unwrap()
            .starts_with("Standard errors could not be computed at the fit's end point:"));
    }

    #[test]
    fn pirls_exhausted_prints_only_on_a_converged_final_eval() {
        let trial = NoteInfo {
            evals: 3,
            ..note("pirls_exhausted")
        };
        let last = NoteInfo {
            final_eval: true,
            ..note("pirls_exhausted")
        };
        let failed = FitText {
            converged: false,
            ..NO_NAMES
        };
        assert_eq!(message(&trial, &NO_NAMES), None);
        assert_eq!(message(&trial, &failed), None);
        assert_eq!(message(&last, &failed), None);
        assert!(message(&last, &NO_NAMES).is_some());
    }

    #[test]
    fn failure_picks_one_kind_in_the_documented_order() {
        // The 20-row response of the ports' warning tests: 0.3x + noise.
        let xs = (-10..=10).filter(|&v| v != 0).map(f64::from);
        let yy: Vec<f64> = xs
            .enumerate()
            .map(|(i, x)| 0.3 * x + (((i * 7) % 5) as f64 - 2.0) * 0.4)
            .collect();
        let pirls = [NoteInfo {
            evals: 3,
            ..note("pirls_exhausted")
        }];
        let separated = [note("separated")];
        let (fin, nan, none) = ([1.5, -2.0], [f64::NAN; 2], [false; 2]);
        let advice =
            "Try a simpler random-effects structure or check for nearly collinear predictors.";
        let inner = " Some of its inner steps ran out of iterations.";
        let search = |inner: &str| {
            format!(
                "The search for the variance estimates reached its step limit before it \
                 settled.{inner} The estimates shown are the best point found; they are often \
                 close, but this is not checked. Do not use them until the fit converges. {advice}"
            )
        };
        let failed = |inner: &str| {
            format!("The fitting algorithm failed and returned no estimates.{inner} {advice}")
        };
        let unsolvable = "The predictors could not be separated numerically, so no estimates \
            were computed. Check for predictors that are copies or near-copies of each other.";
        let too_few = |p: &str, n: &str| {
            format!(
                "The model has {p} to estimate but only {n}, so no estimates were computed. Use \
                 more rows or fewer predictors."
            )
        };
        let constant = |v: &str| {
            format!(
                "Every value of the response is {v}, so there is nothing to estimate. Check the \
                 response column."
            )
        };
        let base = Failed {
            family: "gaussian",
            link: "identity",
            mixed: false,
            notes: &[],
            beta: &fin,
            aliased: &none,
            deviance: f64::NAN,
            y: &yy,
        };
        let one_aliased = [false, true];
        let first_aliased = [true, false];
        let cases: Vec<(Failed, &str, String)> = vec![
            (
                Failed { beta: &nan, ..base },
                "design_unsolvable",
                unsolvable.to_owned(),
            ),
            (
                Failed {
                    family: "binomial",
                    ..base
                },
                "glm_diverged",
                DIVERGED_BINOMIAL.to_owned(),
            ),
            (
                Failed {
                    family: "poisson",
                    ..base
                },
                "glm_diverged",
                DIVERGED_COUNTS.to_owned(),
            ),
            // No inner-steps sentence without random effects.
            (
                Failed {
                    family: "negativebinomial",
                    notes: &pirls,
                    ..base
                },
                "glm_diverged",
                DIVERGED_COUNTS.to_owned(),
            ),
            (
                Failed {
                    family: "gamma",
                    ..base
                },
                "glm_diverged",
                DIVERGED_CONTINUOUS.to_owned(),
            ),
            (
                Failed {
                    family: "gamma",
                    beta: &nan,
                    ..base
                },
                "design_unsolvable",
                unsolvable.to_owned(),
            ),
            (
                Failed {
                    family: "inversegaussian",
                    ..base
                },
                "glm_diverged",
                DIVERGED_CONTINUOUS.to_owned(),
            ),
            (
                Failed {
                    mixed: true,
                    deviance: 12.5,
                    ..base
                },
                "search_limit",
                search(""),
            ),
            (
                Failed {
                    family: "binomial",
                    mixed: true,
                    notes: &pirls,
                    deviance: 12.5,
                    ..base
                },
                "search_limit",
                search(inner),
            ),
            (
                Failed {
                    family: "poisson",
                    mixed: true,
                    beta: &nan,
                    ..base
                },
                "fit_failed",
                failed(""),
            ),
            (
                Failed {
                    family: "poisson",
                    mixed: true,
                    notes: &pirls,
                    beta: &nan,
                    ..base
                },
                "fit_failed",
                failed(inner),
            ),
            // Finite coefficients but no end point: failed, not a search limit.
            (
                Failed {
                    mixed: true,
                    ..base
                },
                "fit_failed",
                failed(""),
            ),
            // ... unless the fixed-effects-only GLM stopped on separation.
            (
                Failed {
                    family: "binomial",
                    mixed: true,
                    notes: &separated,
                    ..base
                },
                "glm_diverged",
                DIVERGED_BINOMIAL.to_owned(),
            ),
            (
                Failed {
                    family: "poisson",
                    mixed: true,
                    notes: &separated,
                    ..base
                },
                "glm_diverged",
                DIVERGED_COUNTS.to_owned(),
            ),
            // No fixed effects at all (y ~ 0 + (1 | g)): the deviance decides.
            (
                Failed {
                    mixed: true,
                    beta: &[],
                    aliased: &[],
                    ..base
                },
                "fit_failed",
                failed(""),
            ),
            (
                Failed {
                    mixed: true,
                    beta: &[],
                    aliased: &[],
                    deviance: 12.5,
                    ..base
                },
                "search_limit",
                search(""),
            ),
            // An aliased NaN slot is not a failed coefficient.
            (
                Failed {
                    mixed: true,
                    beta: &[1.0, f64::NAN],
                    aliased: &one_aliased,
                    deviance: 12.5,
                    ..base
                },
                "search_limit",
                search(""),
            ),
            // A constant response wins over every model-based kind.
            (
                Failed {
                    family: "binomial",
                    beta: &nan,
                    y: &[0.0; 20],
                    ..base
                },
                "constant_response",
                constant("0"),
            ),
            (
                Failed {
                    family: "poisson",
                    mixed: true,
                    notes: &pirls,
                    beta: &nan,
                    y: &[3.0; 20],
                    ..base
                },
                "constant_response",
                constant("3"),
            ),
            (
                Failed {
                    beta: &[],
                    aliased: &[],
                    ..base
                },
                "no_coefficients",
                "The model has no coefficients and no random effects, so there is nothing to \
                 estimate. Add an intercept or a predictor."
                    .to_owned(),
            ),
            (
                Failed {
                    beta: &nan,
                    y: &[1.0, 2.0],
                    ..base
                },
                "too_few_rows",
                too_few("2 coefficients", "2 rows"),
            ),
            (
                Failed {
                    family: "binomial",
                    mixed: true,
                    beta: &nan,
                    y: &[1.0],
                    ..base
                },
                "too_few_rows",
                too_few("2 coefficients", "1 row"),
            ),
            (
                Failed {
                    beta: &[f64::NAN, 1.0],
                    aliased: &first_aliased,
                    y: &[4.0],
                    ..base
                },
                "too_few_rows",
                too_few("1 coefficient", "1 row"),
            ),
        ];
        assert_eq!(cases.len(), 23);
        for (f, kind, message) in &cases {
            assert_eq!(failure(f), (*kind, message.clone()), "{kind}");
            assert_eq!(tier_title(kind).0, "severe");
        }
    }

    #[test]
    fn kind_table_matches_warnings_md() {
        // The rows of the "## Warnings" table: `kind`, tier, title.
        let md = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../documentation/warnings.md"
        ))
        .unwrap();
        let mut rows: Vec<(String, String, String)> = md
            .lines()
            .skip_while(|l| !l.starts_with("## Warnings"))
            .skip(1)
            .take_while(|l| !l.starts_with("## "))
            .filter(|l| l.starts_with("| ") && !l.starts_with("| Kind"))
            .map(|l| {
                let c: Vec<&str> = l.split('|').map(str::trim).collect();
                (
                    c[1].trim_matches('`').to_owned(),
                    c[2].to_lowercase(),
                    c[3].to_owned(),
                )
            })
            .collect();
        let mut want: Vec<(String, String, String)> = KINDS
            .iter()
            .chain([&("any other kind", UNKNOWN.0, UNKNOWN.1)])
            .map(|&(k, t, ti)| (k.to_owned(), t.to_owned(), ti.to_owned()))
            .collect();
        // A kind with more than one message has a row per message.
        rows.sort();
        rows.dedup();
        want.sort();
        assert_eq!(rows, want);
    }
}
