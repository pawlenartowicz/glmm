use glmm::orchestrate;

use crate::error::BindError;
use crate::fns::{check_response, check_rows, Rows};
use crate::inputs::Inputs;
use crate::options::{self, FitArgs};
use crate::records::fit_record;
use crate::value::Value;

/// `orchestrate::run_fit`, the options read first. `warm_start` is a record
/// of `beta`, `theta` (each empty or absent to cold-start that part) and
/// `dispersion`.
pub(crate) fn fit(inp: &mut Inputs) -> Result<Value, BindError> {
    let formula = inp.string("formula")?;
    check_response(&formula)?;
    let (family, link) = options::family_link(inp, &[&formula])?;
    let wald_se = options::wald_se(inp)?;
    let nagq = options::nagq(inp)?;
    let (dispersion, warm_start, notes) =
        options::dispersion_and_start(inp, family, formula.contains('|'))?;
    let satterthwaite = options::flag(inp, "satterthwaite", "satterthwaite")?;
    let condvar = options::flag(inp, "condvar", "condvar")?;
    let reml = options::reml(inp)?;
    let anova_test = options::choice(inp, "anova_test", "anova_test", &["auto", "Wald", "LRT"])?;
    let weight_power = inp.opt_weight_power("weight_power")?;
    let se_type = inp.opt_se_type("se_type")?;
    let args = FitArgs {
        family,
        link,
        nagq,
        dispersion,
        warm_start,
        weight_power,
        notes,
    };
    let numeric = inp.numeric("numeric")?;
    let factors = inp.factors("factors")?;
    let weights = inp.opt_vec("weights")?;
    let offset = inp.opt_vec("offset")?;
    let rows = Rows::read(inp)?;
    check_rows(&numeric, &factors)?;
    let r = orchestrate::run_fit(
        &formula,
        numeric,
        factors,
        family,
        link,
        &wald_se,
        nagq,
        dispersion,
        weights,
        offset,
        args.warm_start.clone(),
        satterthwaite,
        condvar,
        reml,
        &anova_test,
        weight_power,
        se_type,
    )
    .map_err(|e| rows.error(e))?;
    Ok(fit_record(r, &args, inp.port(), rows.base()))
}
