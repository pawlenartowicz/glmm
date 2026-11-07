use glmm::formula::anova::{TermTest, TermTestError};
use glmm::formula::PairwiseInterval;
use glmm::inference::DenDf;
use glmm::orchestrate;

use crate::error::BindError;
use crate::fns::ModelData;
use crate::inputs::Inputs;
use glmm::inference::LinearNote;

use crate::records::{
    pairwise_record, read_model_vcov, read_parts, read_robust, read_satterthwaite,
    term_table_record, text,
};
use crate::value::{Record, Value};

/// `orchestrate::confint`: per coefficient `lower` and `upper`.
pub(crate) fn confint(inp: &mut Inputs) -> Result<Value, BindError> {
    let ci = orchestrate::confint(
        &inp.vec("beta")?,
        &inp.vec("se")?,
        &inp.vec("coef_df")?,
        inp.f64("level")?,
    )?;
    Ok(Value::Record(
        Record::new()
            .with("lower", Value::Vec(ci.iter().map(|b| b[0]).collect()))
            .with("upper", Value::Vec(ci.iter().map(|b| b[1]).collect())),
    ))
}

/// What `contrast_test` and `pairwise` read of a stored fit besides `beta`.
struct StoredTest {
    vcov: Vec<Vec<f64>>,
    model_vcov: Vec<Vec<f64>>,
    aliased: Vec<bool>,
    nobs: usize,
    record: Option<orchestrate::SatterthwaiteRecord>,
    robust: Option<orchestrate::RobustRecord>,
}

fn stored_test(inp: &mut Inputs, p: usize) -> Result<StoredTest, BindError> {
    Ok(StoredTest {
        vcov: inp.rows("vcov", p)?,
        model_vcov: read_model_vcov(inp, p)?,
        aliased: inp.bools("aliased")?,
        nobs: inp.usize("nobs")?,
        record: inp
            .opt_record("record")?
            .map(read_satterthwaite)
            .transpose()?,
        robust: inp.opt_record("robust")?.map(read_robust).transpose()?,
    })
}

/// The robust notes of a stored fit's tests, which the ports raise as
/// warnings: each `kind` once, its `message` the sentence of its first row,
/// after the labels of the rows that carry it when `labelled`. Each row is
/// its label, its notes and their sentences, in the same order.
fn robust_warnings<'a>(
    rows: impl Iterator<Item = (&'a str, &'a [LinearNote], &'a [String])>,
    labelled: bool,
) -> Value {
    let mut out: Vec<(&str, &str, Vec<String>)> = Vec::new();
    for (label, notes, lines) in rows {
        for (&note, line) in notes.iter().zip(lines) {
            let kind = match note {
                LinearNote::RobustUnidentified => "robust_se_unidentified",
                LinearNote::RobustDfSmall => "robust_df_small",
                LinearNote::ModelBasedJointTest => "model_based_joint_test",
                _ => continue,
            };
            match out.iter_mut().find(|(k, ..)| *k == kind) {
                Some((.., labels)) => labels.push(label.to_string()),
                None => out.push((kind, line, vec![label.to_string()])),
            }
        }
    }
    Value::List(
        out.into_iter()
            .map(|(kind, line, labels)| {
                let message = if labelled {
                    format!("{}: {line}", labels.join(", "))
                } else {
                    line.to_string()
                };
                Value::Record(
                    Record::new()
                        .with("kind", text(kind))
                        .with("message", Value::Str(message)),
                )
            })
            .collect(),
    )
}

/// `orchestrate::contrast_test`, `l` the row-major `rows x p` matrix: the
/// one-row table as a stored per-term result.
pub(crate) fn contrast_test(inp: &mut Inputs) -> Result<Value, BindError> {
    let df_method = inp.string("df_method")?;
    let beta = inp.vec("beta")?;
    let s = stored_test(inp, beta.len())?;
    let names = inp.strs("names")?;
    let table = orchestrate::contrast_test(
        &df_method,
        &beta,
        &s.vcov,
        &s.model_vcov,
        &s.aliased,
        s.nobs,
        inp.f64("dispersion")?,
        s.record,
        s.robust,
        &names,
        &inp.vec("l")?,
        inp.usize("rows")?,
    )?;
    let c = orchestrate::term_columns(&table);
    let warnings = robust_warnings(
        table
            .fixed
            .iter()
            .zip(&c.row_notes)
            .map(|(r, lines)| (r.label.as_str(), &r.test.notes[..], &lines[..])),
        false,
    );
    Ok(Value::Record(
        term_table_record(c).with("warnings", warnings),
    ))
}

/// `orchestrate::pairwise` on the stored fit, then its table.
pub(crate) fn pairwise(inp: &mut Inputs) -> Result<Value, BindError> {
    let d = ModelData::read(inp)?;
    let df_method = inp.string("df_method")?;
    let beta = inp.vec("beta")?;
    let s = stored_test(inp, beta.len())?;
    let names = inp.strs("names")?;
    let by = inp.opt_string("by")?;
    let pw = orchestrate::pairwise(
        &d.formula,
        d.numeric,
        d.factors,
        &d.family,
        &d.link,
        &df_method,
        &beta,
        &s.vcov,
        &s.model_vcov,
        &s.aliased,
        s.nobs,
        s.record,
        s.robust,
        &inp.string("factor")?,
        by.as_deref(),
        inp.opt_vec("by_values")?,
        &inp.string("adjust")?,
    )?;
    let c = orchestrate::pairwise_columns(&pw, &names);
    let labels: Vec<String> = c
        .contrast
        .iter()
        .zip(&c.by_level)
        .map(|(contrast, level)| match &c.by {
            Some(by) => format!("{contrast}, {by} = {level}"),
            None => contrast.clone(),
        })
        .collect();
    let warnings = robust_warnings(
        labels
            .iter()
            .zip(&pw.rows)
            .zip(&c.row_notes)
            .map(|((l, r), lines)| (l.as_str(), &r.notes[..], &lines[..])),
        true,
    );
    Ok(Value::Record(pairwise_record(c).with("warnings", warnings)))
}

/// `orchestrate::pairwise_confint` on the columns `pairwise` gave.
pub(crate) fn pairwise_confint(inp: &mut Inputs) -> Result<Value, BindError> {
    let ratio = inp.opt_string("ratio")?;
    let ci = orchestrate::pairwise_confint(
        &inp.vec("estimate")?,
        &inp.vec("se")?,
        &inp.vec("df")?,
        &inp.vec("p_value")?,
        &inp.string("adjust")?,
        inp.usize("levels")?,
        ratio.as_deref(),
        inp.f64("level")?,
    )?;
    let col = |f: fn(&PairwiseInterval) -> f64| Value::Vec(ci.rows.iter().map(f).collect());
    Ok(Value::Record(
        Record::new()
            .with("adjust", text(orchestrate::adjust_tag(ci.adjust)))
            .with("lower", col(|r| r.lower))
            .with("upper", col(|r| r.upper))
            .with("ratio_lower", col(|r| r.ratio_lower))
            .with("ratio_upper", col(|r| r.ratio_upper)),
    ))
}

/// The refusal of a per-term `test` of `"Wald"` or `"LRT"` on a fit whose
/// rows are F tests, `den_df_kind` the F rows' df source.
pub(crate) fn term_test_refusal(inp: &mut Inputs) -> Result<Value, BindError> {
    let asked = match inp.string("test")?.as_str() {
        "Wald" => TermTest::Wald,
        "LRT" => TermTest::Lrt,
        other => {
            return Err(BindError::invalid_argument(format!(
                "no refusal for test {other:?}"
            )))
        }
    };
    let den_df = match inp.string("den_df_kind")?.as_str() {
        "satterthwaite" => DenDf::Satterthwaite,
        "residual" => DenDf::Residual,
        other => {
            return Err(BindError::invalid_argument(format!(
                "unknown df source {other:?}"
            )))
        }
    };
    Ok(Value::Str(
        TermTestError::FTable { asked, den_df }.to_string(),
    ))
}

/// `orchestrate::effect_sizes`: `marginal`, `conditional`, `adjusted` and
/// `unadjusted`, or `Null` where they need the null refit `null_refit` skips.
pub(crate) fn effect_sizes(inp: &mut Inputs) -> Result<Value, BindError> {
    let d = ModelData::read(inp)?;
    let parts = read_parts(inp.record("parts")?)?;
    let objective = inp.string("objective")?;
    let sizes = orchestrate::effect_sizes(
        &d.formula,
        d.numeric,
        d.factors,
        &d.family,
        &d.link,
        d.weights,
        d.offset,
        &parts,
        &inp.bools("aliased")?,
        inp.bool("converged")?,
        (&objective, inp.u8("objective_nodes")?),
        inp.bool("null_refit")?,
    )?;
    Ok(sizes.map_or(Value::Null, |(r2, icc)| {
        Value::Record(
            Record::new()
                .with("marginal", Value::F64(r2.marginal))
                .with("conditional", Value::F64(r2.conditional))
                .with("adjusted", Value::F64(icc.adjusted))
                .with("unadjusted", Value::F64(icc.unadjusted)),
        )
    }))
}

/// `orchestrate::profile_intervals`: `names`, `estimate`, `lower`, `upper`
/// and `level`.
pub(crate) fn profile_intervals(inp: &mut Inputs) -> Result<Value, BindError> {
    let d = ModelData::read(inp)?;
    let r = orchestrate::profile_intervals(
        &d.formula,
        d.numeric,
        d.factors,
        &d.family,
        &d.link,
        d.weights,
        d.offset,
        inp.bool("reml")?,
        // A fit stored before the weight power used the weights as given.
        inp.opt_f64("weight_power")?.unwrap_or(1.0),
        inp.opt_bool("weight_power_estimated")?.unwrap_or(false),
        inp.f64("level")?,
        inp.opt_strs("parm")?,
    )?;
    Ok(Value::Record(
        Record::new()
            .with("names", Value::StrVec(r.names))
            .with("estimate", Value::Vec(r.estimate))
            .with("lower", Value::Vec(r.lower))
            .with("upper", Value::Vec(r.upper))
            .with("level", Value::F64(r.level)),
    ))
}
