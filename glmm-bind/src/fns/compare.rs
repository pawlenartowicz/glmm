use glmm::formula::anova::{compare_stored, CompareError, WeightPowerMismatch};
use glmm::orchestrate::{self, MAX_NSIM};

use crate::coerce;
use crate::error::BindError;
use crate::fns::Rows;
use crate::inputs::Inputs;
use crate::records::{comparison_record, fit_record, read_model};
use crate::value::Value;

/// `orchestrate::run_compare`: the comparison record plus `fit0` and `fit1`.
pub(crate) fn compare(inp: &mut Inputs) -> Result<Value, BindError> {
    let f0 = inp.string("f0")?;
    let f1 = inp.string("f1")?;
    let numeric = inp.numeric("numeric")?;
    let factors = inp.factors("factors")?;
    let rows = Rows::read(inp)?;
    let family = inp.string("family")?;
    let link = inp.string("link")?;
    let r = orchestrate::run_compare(
        &f0,
        &f1,
        numeric,
        factors,
        &family,
        &link,
        inp.u8("nagq")?,
        inp.opt_vec("weights")?,
        inp.opt_vec("offset")?,
        read_nsim(inp)?,
        inp.opt_seed("seed")?,
        inp.opt_weight_power("weight_power")?,
        inp.opt_se_type("se_type")?,
    )
    .map_err(|e| rows.error(e))?;
    let rec = comparison_record(&r.comparison, [r.fit0.rss, r.fit1.rss], r.nsim, r.seed)?
        .with("fit0", fit_record(r.fit0, &family, &link, rows.base()))
        .with("fit1", fit_record(r.fit1, &family, &link, rows.base()));
    Ok(Value::Record(rec))
}

/// `nsim`: a whole number out of range, negative or past 2^53 included, gets
/// `compare_options`' message (R sends doubles, which a count refuses first).
fn read_nsim(inp: &mut Inputs) -> Result<Option<usize>, BindError> {
    let v = inp.value("nsim");
    if v.is_null() {
        return Ok(None);
    }
    if let Ok(x) = coerce::to_f64(&v) {
        if x.fract() == 0.0 && !(1.0..=MAX_NSIM as f64).contains(&x) {
            return Err(BindError::invalid_argument(format!(
                "nsim must be a whole number from 1 to {MAX_NSIM}, got {x}"
            )));
        }
    }
    coerce::to_usize(&v)
        .map(Some)
        .map_err(|w| BindError::invalid_argument(format!("nsim {w}")))
}

/// The refusals a port's message points to its own `compare` for, by kind;
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

/// `compare_stored` on two stored fits, no refit. A refusal is a
/// `compare_error` whose `kind` says which.
pub(crate) fn compare_fits(inp: &mut Inputs) -> Result<Value, BindError> {
    let opts = orchestrate::compare_options(read_nsim(inp)?, inp.opt_seed("seed")?)?;
    let a = read_model(inp.record("m0")?)?;
    let b = read_model(inp.record("m1")?)?;
    let c = compare_stored(&a, &b, &opts).map_err(|e| BindError {
        kind: Some(refusal_kind(&e, a.weight_power_estimated)),
        ..BindError::new("compare_error", e.to_string())
    })?;
    Ok(Value::Record(comparison_record(
        &c,
        [a.rss, b.rss],
        opts.nsim,
        opts.seed,
    )?))
}
