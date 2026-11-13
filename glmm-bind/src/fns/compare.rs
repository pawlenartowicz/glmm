use glmm::formula::anova::{compare_stored, CompareError, WeightPowerMismatch};
use glmm::orchestrate;

use crate::error::BindError;
use crate::fns::{check_response, check_rows, Rows};
use crate::inputs::Inputs;
use crate::options::{self, FitArgs};
use crate::port::{spell, Port};
use crate::records::{comparison_record, fit_record, read_model};
use crate::value::Value;

/// `orchestrate::run_compare`, the options read first: the comparison
/// record plus `fit0` and `fit1`.
pub(crate) fn compare(inp: &mut Inputs) -> Result<Value, BindError> {
    let f0 = inp.string("f0")?;
    let f1 = inp.string("f1")?;
    check_response(&f0)?;
    check_response(&f1)?;
    let (family, link) = options::family_link(inp, &[&f0, &f1])?;
    let nagq = options::nagq(inp)?;
    let nsim = options::opt_nsim(inp)?;
    let seed = options::opt_seed(inp)?;
    let weight_power = inp.opt_weight_power("weight_power")?;
    let se_type = inp.opt_se_type("se_type")?;
    let numeric = inp.numeric("numeric")?;
    let factors = inp.factors("factors")?;
    let weights = inp.opt_vec("weights")?;
    let offset = inp.opt_vec("offset")?;
    let rows = Rows::read(inp)?;
    check_rows(&numeric, &factors)?;
    let r = orchestrate::run_compare(
        &f0,
        &f1,
        numeric,
        factors,
        family,
        link,
        nagq,
        weights,
        offset,
        nsim,
        seed,
        weight_power,
        se_type,
    )
    .map_err(|e| rows.error(e))?;
    let args = FitArgs {
        family,
        link,
        nagq,
        dispersion: None,
        warm_start: None,
        weight_power: None,
        notes: Vec::new(),
    };
    let rec = comparison_record(&r.comparison, [r.fit0.rss, r.fit1.rss], r.nsim, r.seed)?
        .with("fit0", fit_record(r.fit0, &args, inp.port(), rows.base()))
        .with("fit1", fit_record(r.fit1, &args, inp.port(), rows.base()));
    Ok(Value::Record(rec))
}

/// The refusals [`refusal`] points to the port's `compare` for, by kind;
/// `estimated` when the sides estimated their weight power, whose ML refit
/// needs the power named.
fn refusal_kind(e: &CompareError, estimated: bool) -> &'static str {
    match e {
        CompareError::DifferentRows { .. } => "different_rows",
        CompareError::DifferentY => "different_y",
        CompareError::DifferentWeights => "different_weights",
        CompareError::DifferentOffset => "different_offset",
        CompareError::DifferentFamily { .. } => "different_family",
        CompareError::MixedEstimators { .. } => "mixed_estimators",
        CompareError::RemlDifferentFixed if estimated => "reml_different_fixed_estimated",
        CompareError::RemlDifferentFixed => "reml_different_fixed",
        CompareError::WeightPowerMismatch(
            WeightPowerMismatch::EstimatedVsFixed | WeightPowerMismatch::DifferentFixed { .. },
        ) => "weight_power_mismatch",
        _ => "other",
    }
}

/// The crate's refusal, which names no call, followed by the pointer to
/// the port's `compare` (and to refitting by ML) on the refusals that
/// `compare` avoids.
fn refusal(port: Port, e: &CompareError, estimated: bool) -> String {
    let compare = spell(port, "compare_call");
    let pointer = match refusal_kind(e, estimated) {
        "different_rows" => format!("Or use {compare}, which drops those rows once for both fits."),
        "different_y" => format!("Or use {compare}, which fits both to the same data."),
        "different_weights" => format!("Or use {compare}, which takes the weights once."),
        "different_offset" => format!("Or use {compare}, which takes the offset once."),
        "different_family" => format!("Or use {compare}, which takes the family and link once."),
        "mixed_estimators" => format!("Or use {compare}, which picks the estimator itself."),
        "reml_different_fixed" => format!(
            "Pass {} to both fits, or use {compare}, which fits both by ML.",
            spell(port, "reml_false")
        ),
        "reml_different_fixed_estimated" => format!(
            "Pass {} and {} to both fits, or use {compare}, which fits both by ML.",
            spell(port, "reml_false"),
            spell(port, "weight_power_estimate")
        ),
        "weight_power_mismatch" => format!("Or use {compare}, which takes weight_power once."),
        _ => return e.to_string(),
    };
    format!("{e} {pointer}")
}

/// `compare_stored` on two stored fits, no refit. A refusal is a
/// `compare_error`.
pub(crate) fn compare_fits(inp: &mut Inputs) -> Result<Value, BindError> {
    let opts = orchestrate::compare_options(options::opt_nsim(inp)?, options::opt_seed(inp)?)?;
    let a = read_model(inp.record("m0")?)?;
    let b = read_model(inp.record("m1")?)?;
    let c = compare_stored(&a, &b, &opts).map_err(|e| {
        BindError::new(
            "compare_error",
            refusal(inp.port(), &e, a.weight_power_estimated),
        )
    })?;
    Ok(Value::Record(comparison_record(
        &c,
        [a.rss, b.rss],
        opts.nsim,
        opts.seed,
    )?))
}
