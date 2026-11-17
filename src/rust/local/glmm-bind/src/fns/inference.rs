use glmm::formula::anova::{TermTest, TermTestError};
use glmm::formula::PairwiseInterval;
use glmm::inference::DenDf;
use glmm::orchestrate;

use crate::coerce;
use crate::error::BindError;
use crate::fns::ModelData;
use crate::inputs::Inputs;
use crate::notes::fmt_g;
use crate::options;
use crate::port::spell;
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
        &inp.string("df_method")?,
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

/// `factor`, `by` and `at` of `pairwise`: one variable name, an optional
/// one, and the values of a numeric `by` to compare at.
type PairwiseOptions = (String, Option<String>, Option<Vec<f64>>);

fn pairwise_options(inp: &mut Inputs) -> Result<PairwiseOptions, BindError> {
    let v = inp.value("factor");
    let factor = coerce::to_string(v.clone()).map_err(|_| {
        BindError::invalid_argument(format!(
            "factor must be one variable name, got {}",
            options::shown(inp.port(), &v)
        ))
    })?;
    let by = match inp.value("by") {
        Value::Null => None,
        v => Some(coerce::to_string(v.clone()).map_err(|_| {
            BindError::invalid_argument(format!(
                "by must be {} or one variable name, got {}",
                spell(inp.port(), "null"),
                options::shown(inp.port(), &v)
            ))
        })?),
    };
    let at = match inp.value("at") {
        Value::Null => None,
        _ if by.is_none() => {
            return Err(BindError::invalid_argument(
                "at gives the values of a numeric by, so it needs by",
            ))
        }
        v => {
            let bad = |got: String| {
                BindError::invalid_argument(format!(
                    "at must be one or more finite numbers, got {got}"
                ))
            };
            let x = coerce::to_vec(v.clone())
                .ok()
                .filter(|x| !x.is_empty())
                .ok_or_else(|| bad(options::shown(inp.port(), &v)))?;
            if let Some(&a) = x.iter().find(|a| !a.is_finite()) {
                return Err(bad(coerce::describe(&Value::F64(a))));
            }
            Some(x)
        }
    };
    Ok((factor, by, at))
}

/// The first note of a pairwise table of a fit that did not converge.
const TESTS_NOT_CONVERGED: &str = "The fit did not converge, so its estimates are not at a \
     maximum and these tests are not reliable. Change the model until it converges (a simpler \
     random part often helps), then test again.";

/// The first note of the intervals of such a table.
const INTERVALS_NOT_CONVERGED: &str = "The fit did not converge, so its estimates are not at a \
     maximum and these intervals are not reliable. Change the model until it converges (a \
     simpler random part often helps), then compute them again.";

/// Holm adjusts p-values step by step, so it has no simultaneous interval.
const HOLM_INTERVALS: &str = "Holm has no simultaneous interval; Bonferroni intervals shown.";

/// The rows of each `by` group of a pairwise table over `levels` levels:
/// `p` cut into runs of `levels·(levels − 1)/2`, the pairs `(i, j)`, `i < j`,
/// in row order within each run.
fn by_groups(p: &[f64], levels: usize) -> std::slice::Chunks<'_, f64> {
    p.chunks((levels * levels.saturating_sub(1) / 2).max(1))
}

/// Each `by` group's family size under `adjust`, the distinct sizes in
/// ascending order. Holm and Bonferroni count the tested rows (a p-value
/// that is not NaN); Tukey counts the levels in at least one tested row, as
/// the crate's `tukey_k` does.
fn family_sizes(p: &[f64], levels: usize, tukey: bool) -> Vec<usize> {
    let pairs: Vec<(usize, usize)> = (0..levels)
        .flat_map(|i| (i + 1..levels).map(move |j| (i, j)))
        .collect();
    let mut sizes: Vec<usize> = by_groups(p, levels)
        .map(|group| {
            let tested = group.iter().zip(&pairs).filter(|(v, _)| !v.is_nan());
            if tukey {
                let mut seen = vec![false; levels];
                for (_, &(i, j)) in tested {
                    seen[i] = true;
                    seen[j] = true;
                }
                seen.iter().filter(|&&s| s).count()
            } else {
                tested.count()
            }
        })
        .collect();
    sizes.sort_unstable();
    sizes.dedup();
    sizes
}

/// The note naming `adjust` and its family size, emmeans' words; `None` for
/// `"none"`.
fn adjust_line(p: &[f64], levels: usize, adjust: &str, intervals: bool) -> Option<String> {
    let lead = if intervals {
        "Conf-level adjustment"
    } else {
        "P value adjustment"
    };
    match adjust {
        "none" => None,
        "tukey" => Some(match family_sizes(p, levels, true)[..] {
            [k] => format!("{lead}: tukey method for comparing a family of {k} estimates"),
            _ => format!("{lead}: tukey method for varying family sizes"),
        }),
        _ => {
            let sizes: Vec<String> = family_sizes(p, levels, false)
                .iter()
                .map(ToString::to_string)
                .collect();
            let what = if intervals { "estimates" } else { "tests" };
            Some(format!(
                "{lead}: {adjust} method for {} {what}",
                sizes.join(" or ")
            ))
        }
    }
}

/// Each row's notes, as `"a - b: note"`, or `"a - b, h = h1: note"` within a
/// `by` level.
fn row_notes(
    contrast: &[String],
    by: Option<&str>,
    by_level: &[String],
    notes: &[Vec<String>],
) -> Vec<String> {
    let mut out = Vec::new();
    for (i, (c, row)) in contrast.iter().zip(notes).enumerate() {
        let label = match by {
            Some(by) => format!("{c}, {by} = {}", by_level[i]),
            None => c.clone(),
        };
        out.extend(row.iter().map(|n| format!("{label}: {n}")));
    }
    out
}

/// `orchestrate::pairwise` on the stored fit, then its table, with its
/// `converged` and its `notes`: the not-converged note when the fit did not
/// converge, the adjustment, then each row's notes.
pub(crate) fn pairwise(inp: &mut Inputs) -> Result<Value, BindError> {
    let (factor, by, at) = pairwise_options(inp)?;
    let converged = inp.bool("converged")?;
    let d = ModelData::read(inp)?;
    let df_method = inp.string("df_method")?;
    let beta = inp.vec("beta")?;
    let s = stored_test(inp, beta.len())?;
    let names = inp.strs("names")?;
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
        &factor,
        by.as_deref(),
        at,
        &options::choice(
            inp,
            "adjust",
            "adjust",
            &["holm", "bonferroni", "tukey", "none"],
        )?,
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
    let notes: Vec<String> = (!converged)
        .then(|| TESTS_NOT_CONVERGED.to_owned())
        .into_iter()
        .chain(adjust_line(&c.p_value, c.levels, c.adjust, false))
        .chain(row_notes(
            &c.contrast,
            c.by.as_deref(),
            &c.by_level,
            &c.row_notes,
        ))
        .collect();
    Ok(Value::Record(
        pairwise_record(c)
            .with("warnings", warnings)
            .with("converged", Value::Bool(converged))
            .with("notes", Value::StrVec(notes)),
    ))
}

/// `orchestrate::pairwise_confint` on `table`, the record `pairwise` gave,
/// at `level`, with the `notes` of the intervals: the not-converged note,
/// the level, the adjustment (Bonferroni's for a Holm table), then each
/// row's notes.
pub(crate) fn pairwise_confint(inp: &mut Inputs) -> Result<Value, BindError> {
    let level = inp.f64("level")?;
    let mut t = inp.record("table")?;
    let p_value = t.vec("p_value")?;
    let adjust = t.string("adjust")?;
    let levels = t.usize("levels")?;
    let ratio = t.opt_string("ratio_kind")?;
    let ci = orchestrate::pairwise_confint(
        &t.vec("estimate")?,
        &t.vec("se")?,
        &t.vec("df")?,
        &p_value,
        &adjust,
        levels,
        ratio.as_deref(),
        level,
    )?;
    let by = t.opt_string("by")?;
    let notes_of_rows = match t.value("row_notes") {
        Value::List(rows) => rows
            .into_iter()
            .map(coerce::to_strs)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|w| BindError::invalid_argument(format!("table.row_notes {w}")))?,
        other => {
            return Err(BindError::invalid_argument(format!(
                "table.row_notes must be a list, got {}",
                coerce::describe(&other)
            )))
        }
    };
    let interval_adjust = orchestrate::adjust_tag(ci.adjust);
    let notes: Vec<String> = (!t.bool("converged")?)
        .then(|| INTERVALS_NOT_CONVERGED.to_owned())
        .into_iter()
        .chain([format!("Confidence level used: {}", fmt_g(level, 6))])
        .chain(if adjust == "holm" {
            Some(HOLM_INTERVALS.to_owned())
        } else {
            adjust_line(&p_value, levels, interval_adjust, true)
        })
        .chain(row_notes(
            &t.strs("contrast")?,
            by.as_deref(),
            &t.strs("by_level")?,
            &notes_of_rows,
        ))
        .collect();
    let col = |f: fn(&PairwiseInterval) -> f64| Value::Vec(ci.rows.iter().map(f).collect());
    Ok(Value::Record(
        Record::new()
            .with("adjust", text(orchestrate::adjust_tag(ci.adjust)))
            .with("lower", col(|r| r.lower))
            .with("upper", col(|r| r.upper))
            .with("ratio_lower", col(|r| r.ratio_lower))
            .with("ratio_upper", col(|r| r.ratio_upper))
            .with("notes", Value::StrVec(notes)),
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
