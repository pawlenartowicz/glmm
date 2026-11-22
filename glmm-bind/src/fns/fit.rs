use glmm::formula::anova::TermTest;
use glmm::orchestrate;

use crate::error::BindError;
use crate::fns::{check_response, check_rows, Rows};
use crate::inputs::Inputs;
use crate::options;
use crate::records::fit_record;
use crate::value::Value;

/// `orchestrate::run_fit`, the options read first. `warm_start` is a record
/// of `beta`, `theta` (each empty or absent to cold-start that part) and
/// `dispersion`.
pub(crate) fn fit(inp: &mut Inputs) -> Result<Value, BindError> {
    let formula = inp.string("formula")?;
    check_response(&formula)?;
    let (family, link) = options::family_link(inp, &[&formula])?;
    let args = options::fit_options(inp, family, link, &[&formula])?;
    let anova_test = options::choice(
        inp,
        "anova_test",
        "anova_test",
        &[
            ("auto", TermTest::Auto),
            ("Wald", TermTest::Wald),
            ("LRT", TermTest::Lrt),
        ],
    )?;
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
        orchestrate::family_from_str(family, link)?,
        args.options.clone(),
        weights,
        offset,
        args.warm_start.clone(),
        anova_test,
    )
    .map_err(|e| rows.error(e))?;
    Ok(fit_record(r, &args, inp.port(), rows.base()))
}
