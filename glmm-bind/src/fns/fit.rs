use glmm::orchestrate;

use crate::error::BindError;
use crate::fns::Rows;
use crate::inputs::Inputs;
use crate::records::fit_record;
use crate::value::Value;

/// `orchestrate::run_fit`. `warm_start` is a record of `beta`, `theta` (each
/// empty or absent to cold-start that part) and `dispersion`.
pub(crate) fn fit(inp: &mut Inputs) -> Result<Value, BindError> {
    let formula = inp.string("formula")?;
    let numeric = inp.numeric("numeric")?;
    let factors = inp.factors("factors")?;
    let warm_start = match inp.opt_record("warm_start")? {
        None => None,
        Some(mut w) => Some((
            w.opt_vec("beta")?.unwrap_or_default(),
            w.opt_vec("theta")?.unwrap_or_default(),
            w.opt_f64("dispersion")?,
        )),
    };
    let rows = Rows::read(inp)?;
    let family = inp.string("family")?;
    let link = inp.string("link")?;
    let r = orchestrate::run_fit(
        &formula,
        numeric,
        factors,
        &family,
        &link,
        &inp.string("wald_se")?,
        inp.u8("nagq")?,
        inp.opt_f64("dispersion")?,
        inp.opt_vec("weights")?,
        inp.opt_vec("offset")?,
        warm_start,
        inp.bool("satterthwaite")?,
        inp.bool("condvar")?,
        inp.opt_bool("reml")?,
        &inp.string("anova_test")?,
        inp.opt_weight_power("weight_power")?,
        inp.opt_se_type("se_type")?,
    )
    .map_err(|e| rows.error(e))?;
    Ok(fit_record(r, &family, &link, rows.base()))
}
