//! The engine's result structs as [`Record`]s, and the records a port passes
//! back (a stored fit's Satterthwaite record, its numbers, one side of a
//! comparison) as the engine's structs. Field names are those both ports
//! read today; every index field is a [`Value::Index`] (D7).

use glmm::formula::anova::{
    Comparison, ComparisonTest, Differs, Model, ModelInfo, Side, TermTable,
};
use glmm::inference::Reference;
use glmm::orchestrate::{
    self, CondvarCrossRecord, FitResult, NoteInfo, PairwiseColumns, RobustRecord,
    SatterthwaiteRecord, StoredParts, TermColumns,
};
use glmm::SeType;

use crate::error::BindError;
use crate::inputs::Inputs;
use crate::notes::{self, Failed, FitText};
use crate::value::{Record, Value};

pub(crate) fn text(s: &str) -> Value {
    Value::Str(s.to_owned())
}

pub(crate) fn opt_text(s: Option<&str>) -> Value {
    s.map_or(Value::Null, text)
}

pub(crate) fn int(n: usize) -> Value {
    Value::I64(i64::try_from(n).unwrap_or(i64::MAX))
}

pub(crate) fn ints(v: impl IntoIterator<Item = u32>) -> Value {
    Value::IntVec(v.into_iter().map(i64::from).collect())
}

pub(crate) fn opt_vec(v: Option<Vec<f64>>) -> Value {
    v.map_or(Value::Null, Value::Vec)
}

pub(crate) fn vecs(v: Vec<Vec<f64>>) -> Value {
    Value::List(v.into_iter().map(Value::Vec).collect())
}

fn str_lists(v: Vec<Vec<String>>) -> Value {
    Value::List(v.into_iter().map(Value::StrVec).collect())
}

fn statics(v: Vec<&'static str>) -> Value {
    Value::StrVec(v.into_iter().map(str::to_owned).collect())
}

/// A note with its `tier`, `title` and `message` (`Null` on a note that
/// raises nothing).
fn note_record(n: NoteInfo, message: Option<String>) -> Value {
    let (tier, title) = notes::tier_title(n.kind);
    Value::Record(
        Record::new()
            .with("kind", text(n.kind))
            .with("tier", text(tier))
            .with("title", text(title))
            .with("message", message.map_or(Value::Null, Value::Str))
            .with(
                "columns",
                Value::Index(n.columns.into_iter().map(|c| c as usize).collect()),
            )
            .with("pivot", Value::F64(n.pivot))
            .with("evals", Value::I64(i64::from(n.evals)))
            .with("final_eval", Value::Bool(n.final_eval))
            .with("detail", Value::Str(n.detail))
            .with("delta", Value::F64(n.delta))
            .with("factor", Value::F64(n.factor))
            .with("lower", Value::F64(n.lower))
            .with("upper", Value::F64(n.upper))
            .with("failed", Value::I64(i64::from(n.failed)))
            .with("not_converged", Value::I64(i64::from(n.not_converged)))
            .with("df", Value::Vec(n.df)),
    )
}

fn cross_record(c: &CondvarCrossRecord) -> Value {
    Value::Record(
        Record::new()
            .with("family_grouping", int(c.family_grouping))
            .with("family", Value::Vec(c.family.clone()))
            .with("tail", Value::Vec(c.tail.clone()))
            .with("coupling_at", ints(c.coupling_at.iter().copied()))
            .with("coupling_cols", ints(c.coupling_cols.iter().copied()))
            .with("coupling", Value::Vec(c.coupling.clone())),
    )
}

fn satterthwaite_record(rec: SatterthwaiteRecord) -> Value {
    Value::Record(
        Record::new()
            .with("a_eff", Value::Vec(rec.a_eff))
            .with("jac_theta", vecs(rec.jac_theta))
            .with("c", Value::Vec(rec.c))
            .with("tau", Value::F64(rec.tau))
            .with("nu", Value::F64(rec.nu))
            .with("retained", rec.retained.map_or(Value::Null, Value::Index)),
    )
}

/// A robust fit's pieces ([`RobustRecord`]), the cluster of each null
/// direction an index.
fn robust_record(rec: RobustRecord) -> Value {
    Value::Record(
        Record::new()
            .with("er", Value::Vec(rec.er))
            .with("f", vecs(rec.f))
            .with("epe", vecs(rec.epe))
            .with("row_e", Value::Vec(rec.row_e))
            .with("row_x", Value::Vec(rec.row_x))
            .with("row_phi", Value::Vec(rec.row_phi))
            .with("null_cluster", Value::Index(rec.null_cluster))
            .with("null", vecs(rec.null)),
    )
}

/// A [`FitResult`] as the fit record, the warning text of its notes and of
/// its non-convergence (`failure`, `Null` on a converged fit) rendered for
/// the `family` and `link` the call fitted and the port's `row_base`.
pub(crate) fn fit_record(r: FitResult, family: &str, link: &str, row_base: usize) -> Value {
    let messages = notes::note_messages(
        &r.notes,
        &r.beta,
        &FitText {
            names: &r.names,
            converged: r.converged,
            nobs: r.nobs,
            row_base,
        },
    );
    let failure = if r.converged {
        Value::Null
    } else {
        let (kind, message) = notes::failure(&Failed {
            family,
            link,
            mixed: !r.re_groups.is_empty(),
            notes: &r.notes,
            beta: &r.beta,
            aliased: &r.aliased,
            deviance: r.deviance,
            y: &r.y,
        });
        notes::warning_record(kind, message)
    };
    let (objective, nodes) = orchestrate::objective_parts(r.objective);
    let (auto_test, auto_reason) = orchestrate::choice_tags(&r.term_test_auto);
    let cross = r.condvar_cross.as_ref().map_or(Value::Null, cross_record);
    let ranef_levels = Value::IntVec(
        r.ranef_levels
            .iter()
            .map(|&n| i64::try_from(n).unwrap_or(i64::MAX))
            .collect(),
    );
    let (re_group_names, re_group_terms): (Vec<String>, Vec<Vec<String>>) =
        r.re_groups.into_iter().unzip();
    let design_summary = r.design_summary.map_or(Value::Null, |(counts, s, c)| {
        Value::Record(
            Record::new()
                .with("counts", Value::Vec(counts))
                .with("s", Value::Vec(s))
                .with("c", Value::Vec(c)),
        )
    });
    let ranef_blocks = r
        .ranef_blocks
        .into_iter()
        .map(|(group, terms, levels, values, condvar)| {
            Value::Record(
                Record::new()
                    .with("group", Value::Str(group))
                    .with("terms", Value::StrVec(terms))
                    .with("levels", Value::StrVec(levels))
                    .with("values", Value::Vec(values))
                    .with("condvar", Value::Vec(condvar)),
            )
        })
        .collect();
    let lrt = |t: &Option<Result<TermTable, String>>| t.as_ref().map_or(Value::Null, term_record);
    Value::Record(
        Record::new()
            .with("beta", Value::Vec(r.beta))
            .with("se", Value::Vec(r.se))
            .with("vcov", vecs(r.vcov))
            .with("tau2", Value::Vec(r.tau2))
            .with("varcorr", vecs(r.varcorr))
            .with("stddev_se", Value::Vec(r.stddev_se))
            .with("aliased", Value::BoolVec(r.aliased))
            .with("dispersion", Value::F64(r.dispersion))
            .with("converged", Value::Bool(r.converged))
            .with("n_eval", int(r.n_eval))
            .with("n_grad", int(r.n_grad))
            .with("outer_stop", opt_text(r.outer_stop))
            .with("outer_decrement", Value::F64(r.outer_decrement))
            .with("deviance", Value::F64(r.deviance))
            .with("singular", Value::Bool(r.singular))
            .with("names", Value::StrVec(r.names))
            .with("re_group_names", Value::StrVec(re_group_names))
            .with("re_group_terms", str_lists(re_group_terms))
            .with(
                "re_term_index",
                Value::List(r.re_term_index.into_iter().map(Value::Index).collect()),
            )
            .with("agq_warning", opt_text(r.agq_warning.as_deref()))
            .with("reml_warning", opt_text(r.reml_warning.as_deref()))
            .with("loglik", Value::F64(r.loglik))
            .with("df", int(r.df))
            .with("reml", Value::Bool(r.reml))
            .with("fitted", Value::Vec(r.fitted))
            .with("ranef", Value::Vec(r.ranef))
            .with("ranef_levels", ranef_levels)
            .with("y", Value::Vec(r.y))
            .with("weights", opt_vec(r.weights))
            .with("weight_power", Value::F64(r.weight_power))
            .with(
                "weight_power_estimated",
                Value::Bool(r.weight_power_estimated),
            )
            .with("se_type", text(se_type_name(r.se_type)))
            .with("model_vcov", vecs(r.model_vcov))
            .with("robust", r.robust.map_or(Value::Null, robust_record))
            .with("design_summary", design_summary)
            .with("nobs", int(r.nobs))
            .with("ranef_blocks", Value::List(ranef_blocks))
            .with("condvar", Value::Vec(r.condvar))
            .with("condvar_x", Value::Vec(r.condvar_x))
            .with("condvar_cross", cross)
            .with("boundary", text(r.boundary))
            .with(
                "pinned",
                Value::List(r.pinned.into_iter().map(Value::BoolVec).collect()),
            )
            .with(
                "notes",
                Value::List(
                    r.notes
                        .into_iter()
                        .zip(messages)
                        .map(|(n, m)| note_record(n, m))
                        .collect(),
                ),
            )
            .with("failure", failure)
            .with("coef_df", Value::Vec(r.coef_df))
            .with("coef_p", Value::Vec(r.coef_p))
            .with("df_method", text(r.df_method))
            .with(
                "satterthwaite_record",
                r.satterthwaite_record
                    .map_or(Value::Null, satterthwaite_record),
            )
            .with("rss", Value::F64(r.rss))
            .with("r_squared", Value::F64(r.r_squared))
            .with("adj_r_squared", Value::F64(r.adj_r_squared))
            .with("offset", opt_vec(r.offset))
            .with("objective", text(objective))
            .with("objective_nodes", Value::I64(i64::from(nodes)))
            .with("term_table_type3", term_record(&r.term_table_type3))
            .with("term_table_type2", term_record(&r.term_table_type2))
            .with("term_table_type3_lrt", lrt(&r.term_table_type3_lrt))
            .with("term_table_type2_lrt", lrt(&r.term_table_type2_lrt))
            .with(
                "term_test_auto",
                Value::Record(
                    Record::new()
                        .with("test", text(auto_test))
                        .with("reason", text(auto_reason)),
                ),
            ),
    )
}

/// The ports' spelling of a standard-error type.
fn se_type_name(t: SeType) -> &'static str {
    match t {
        SeType::Robust => "robust",
        _ => "model",
    }
}

/// A stored per-term result: `error` the refusal's message, or `Null` beside
/// the table's columns.
pub(crate) fn term_record(result: &Result<TermTable, String>) -> Value {
    Value::Record(match result {
        Err(message) => Record::new().with("error", text(message)),
        Ok(table) => term_table_record(orchestrate::term_columns(table)),
    })
}

/// A per-term table's columns (`orchestrate::TermColumns`, one entry per
/// row), with `error` `Null`.
pub(crate) fn term_table_record(c: TermColumns) -> Record {
    Record::new()
        .with("error", Value::Null)
        .with("ss_type", text(c.ss_type))
        .with("test", text(c.test))
        .with("reason", text(c.reason))
        .with("label", Value::StrVec(c.label))
        .with("kind", statics(c.kind))
        .with("statistic", Value::Vec(c.statistic))
        .with("num_df", Value::Vec(c.num_df))
        .with("den_df", Value::Vec(c.den_df))
        .with("den_df_kind", statics(c.den_df_kind))
        .with("p_value", Value::Vec(c.p_value))
        .with("sum_sq", Value::Vec(c.sum_sq))
        .with("mean_sq", Value::Vec(c.mean_sq))
        .with("eta2_partial", Value::Vec(c.eta2_partial))
        .with("row_notes", str_lists(c.row_notes))
        .with("notes", Value::StrVec(c.notes))
}

/// `pairwise`'s table, a record keyed by its fields.
pub(crate) fn pairwise_record(c: PairwiseColumns) -> Record {
    Record::new()
        .with("factor", Value::Str(c.factor))
        .with("by", c.by.map_or(Value::Null, Value::Str))
        .with("adjust", text(c.adjust))
        .with("ratio_kind", opt_text(c.ratio_kind))
        .with("levels", int(c.levels))
        .with("contrast", Value::StrVec(c.contrast))
        .with("by_level", Value::StrVec(c.by_level))
        .with("estimate", Value::Vec(c.estimate))
        .with("se", Value::Vec(c.se))
        .with("df", Value::Vec(c.df))
        .with("statistic", Value::Vec(c.statistic))
        .with("p_value", Value::Vec(c.p_value))
        .with("p_adjusted", Value::Vec(c.p_adjusted))
        .with("ratio", Value::Vec(c.ratio))
        .with("ratio_se", Value::Vec(c.ratio_se))
        .with("row_notes", str_lists(c.row_notes))
}

fn side_record(m: &ModelInfo, rss: f64) -> Value {
    Value::Record(
        Record::new()
            .with("formula", text(&m.formula_label))
            .with("npar", int(m.npar))
            .with("nobs", int(m.nobs))
            .with("loglik", Value::F64(m.loglik))
            .with("aic", m.aic.map_or(Value::Null, Value::F64))
            .with("bic", m.bic.map_or(Value::Null, Value::F64))
            .with("rss", Value::F64(rss))
            .with(
                "objective",
                text(orchestrate::objective_parts(m.objective).0),
            ),
    )
}

/// A [`Comparison`]: `sides` holds `m0` and `m1` as given (each with its
/// residual sum of squares `rss`, NaN off OLS), `larger` the index of the
/// larger one, the test (`"lr"` or `"f"`) with `objective` and
/// `objective_nodes` (`""` and 0 on `"f"`), its `reference` record and notes,
/// and the `nsim` and `seed` the call used.
pub(crate) fn comparison_record(
    c: &Comparison,
    rss: [f64; 2],
    nsim: usize,
    seed: u64,
) -> Result<Record, BindError> {
    let (test, statistic, df, p_value, reference, (objective, nodes)) = match &c.test {
        ComparisonTest::Lr(t) => (
            "lr",
            t.statistic,
            t.df as f64,
            t.p_value,
            t.reference,
            orchestrate::objective_parts(t.objective),
        ),
        ComparisonTest::Linear(t) => ("f", t.statistic, t.num_df, t.p_value, t.reference, ("", 0)),
        other => {
            return Err(BindError::internal(format!(
                "no rendering for the comparison test {other:?}"
            )))
        }
    };
    let reference = match reference {
        Reference::ChiSquared { df } => Record::new()
            .with("kind", text("chisq"))
            .with("df", int(df)),
        Reference::ChiBarSquared { df_low } => Record::new()
            .with("kind", text("chibar"))
            .with("df", int(df_low)),
        Reference::ExactRlrt { nsim, seed } => Record::new()
            .with("kind", text("exact"))
            .with("nsim", int(nsim))
            .with("seed", Value::U64(seed)),
        Reference::F { num_df, den_df, .. } => Record::new()
            .with("kind", text("f"))
            .with("df", Value::F64(num_df))
            .with("den_df", Value::F64(den_df)),
        other => {
            return Err(BindError::internal(format!(
                "no rendering for the reference {other:?}"
            )))
        }
    };
    Ok(Record::new()
        .with(
            "sides",
            Value::List(vec![side_record(&c.m0, rss[0]), side_record(&c.m1, rss[1])]),
        )
        .with(
            "larger",
            Value::I64(match c.larger {
                Side::First => 0,
                Side::Second => 1,
            }),
        )
        .with(
            "differs",
            text(match c.differs {
                Differs::Fixed => "fixed",
                Differs::Random => "random",
                Differs::Both => "both",
            }),
        )
        .with("objective", text(objective))
        .with("objective_nodes", Value::I64(i64::from(nodes)))
        .with("test", text(test))
        .with("statistic", Value::F64(statistic))
        .with("df", Value::F64(df))
        .with("p_value", Value::F64(p_value))
        .with("reference", Value::Record(reference))
        .with("notes", Value::StrVec(c.note_lines()))
        .with("nsim", int(nsim))
        .with("seed", Value::U64(seed)))
}

/// A stored fit's `satterthwaite_record`, as [`fit_record`] writes it.
pub(crate) fn read_satterthwaite(mut r: Inputs) -> Result<SatterthwaiteRecord, BindError> {
    Ok(SatterthwaiteRecord {
        a_eff: r.vec("a_eff")?,
        jac_theta: r.vecs("jac_theta")?,
        c: r.vec("c")?,
        tau: r.f64("tau")?,
        nu: r.f64("nu")?,
        retained: r.opt_usizes("retained")?,
    })
}

/// A stored fit's `robust`, as [`fit_record`] writes it.
pub(crate) fn read_robust(mut r: Inputs) -> Result<RobustRecord, BindError> {
    Ok(RobustRecord {
        er: r.vec("er")?,
        f: r.vecs("f")?,
        epe: r.vecs("epe")?,
        row_e: r.vec("row_e")?,
        row_x: r.vec("row_x")?,
        row_phi: r.vec("row_phi")?,
        null_cluster: r.usizes("null_cluster")?,
        null: r.vecs("null")?,
    })
}

/// A stored fit's `model_vcov`, `p` rows of `p`; empty when absent (a model
/// fit, or one saved before it).
pub(crate) fn read_model_vcov(r: &mut Inputs, p: usize) -> Result<Vec<Vec<f64>>, BindError> {
    if r.has("model_vcov") {
        r.rows("model_vcov", p)
    } else {
        Ok(Vec::new())
    }
}

/// The numbers predict, simulate and the effect sizes read back: `beta`,
/// `vcov`, `model_vcov` (empty when absent), `varcorr`, `dispersion`,
/// `weight_power` (1 when absent, as on a fit saved before it) and the fields
/// of `kernel_parts`.
pub(crate) fn read_parts(mut r: Inputs) -> Result<StoredParts, BindError> {
    let beta = r.vec("beta")?;
    let vcov = r.rows("vcov", beta.len())?;
    let model_vcov = read_model_vcov(&mut r, beta.len())?;
    let condvar_cross = match r.opt_record("condvar_cross")? {
        None => None,
        Some(mut c) => {
            let u32s = |c: &mut Inputs, name: &str| -> Result<Vec<u32>, BindError> {
                c.usizes(name)?
                    .into_iter()
                    .map(|v| {
                        u32::try_from(v).map_err(|_| {
                            BindError::invalid_argument(format!(
                                "condvar_cross.{name} is too large"
                            ))
                        })
                    })
                    .collect()
            };
            Some(CondvarCrossRecord {
                family_grouping: c.usize("family_grouping")?,
                family: c.vec("family")?,
                tail: c.vec("tail")?,
                coupling_at: u32s(&mut c, "coupling_at")?,
                coupling_cols: u32s(&mut c, "coupling_cols")?,
                coupling: c.vec("coupling")?,
            })
        }
    };
    Ok(StoredParts {
        beta,
        vcov,
        model_vcov,
        varcorr: r.vecs("varcorr")?,
        ranef: r.vec("ranef")?,
        ranef_levels: r.usizes("ranef_levels")?,
        condvar: r.vec("condvar")?,
        condvar_x: r.vec("condvar_x")?,
        condvar_cross,
        dispersion: r.f64("dispersion")?,
        weight_power: r.opt_f64("weight_power")?.unwrap_or(1.0),
    })
}

/// A stored fit's groupings as `(name, terms)`: from `re_groups` (pairs of
/// name and terms, Python) or, when absent, from `re_group_names` and
/// `re_group_terms` (R).
pub(crate) fn read_re_groups(m: &mut Inputs) -> Result<Vec<(String, Vec<String>)>, BindError> {
    let bad = |what: &str| BindError::invalid_argument(format!("fit field {what:?} is malformed"));
    Ok(if m.has("re_groups") {
        let Value::List(groups) = m.value("re_groups") else {
            return Err(bad("re_groups"));
        };
        groups
            .into_iter()
            .map(|g| match g {
                Value::List(pair) if pair.len() == 2 => {
                    let mut pair = pair.into_iter();
                    match (pair.next(), pair.next()) {
                        (Some(Value::Str(name)), Some(terms)) => Ok((
                            name,
                            crate::coerce::to_strs(terms).map_err(|_| bad("re_groups"))?,
                        )),
                        _ => Err(bad("re_groups")),
                    }
                }
                _ => Err(bad("re_groups")),
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let names = m.opt_strs("re_group_names")?.unwrap_or_default();
        let terms = match m.value("re_group_terms") {
            Value::List(t) => t
                .into_iter()
                .map(crate::coerce::to_strs)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|_| bad("re_group_terms"))?,
            Value::Null => Vec::new(),
            _ => return Err(bad("re_group_terms")),
        };
        if names.len() != terms.len() {
            return Err(bad("re_group_terms"));
        }
        names.into_iter().zip(terms).collect()
    })
}

/// One side of a comparison from a stored fit's fields. The groupings are
/// read by [`read_re_groups`]; `design_summary` is `Null`, a record of
/// `counts`, `s` and `c`, or a list of the three.
pub(crate) fn read_model(mut m: Inputs) -> Result<Model, BindError> {
    let bad = |what: &str| BindError::invalid_argument(format!("fit field {what:?} is malformed"));
    let family = orchestrate::family_from_str(&m.string("family")?, &m.string("link")?)?;
    let nodes = m.u8("objective_nodes")?;
    let objective = orchestrate::objective_from_parts(&m.string("objective")?, nodes)?;
    let re_groups = read_re_groups(&mut m)?;
    let design_summary = match m.value("design_summary") {
        Value::Null => None,
        Value::Record(r) => {
            let mut d = Inputs::new(r);
            Some((d.vec("counts")?, d.vec("s")?, d.vec("c")?))
        }
        Value::List(parts) if parts.len() == 3 => {
            let mut parts = parts.into_iter().map(crate::coerce::to_vec);
            let mut next = || {
                parts
                    .next()
                    .unwrap_or(Ok(Vec::new()))
                    .map_err(|_| bad("design_summary"))
            };
            Some((next()?, next()?, next()?))
        }
        _ => return Err(bad("design_summary")),
    };
    let coef = m.vec("coef")?;
    let vcov = m.rows("vcov", coef.len())?;
    Ok(Model {
        formula: m.string("formula")?,
        family,
        loglik: m.f64("loglik")?,
        df: m.usize("df")?,
        nobs: m.usize("nobs")?,
        reml: m.bool("reml")?,
        objective,
        converged: m.bool("converged")?,
        varcorr: m.vecs("varcorr")?,
        re_groups,
        y: m.vec("y")?,
        weights: m.opt_vec("weights")?,
        // A fit stored before the weight power used the weights as given.
        weight_power: m.opt_f64("weight_power")?.unwrap_or(1.0),
        weight_power_estimated: m.opt_bool("weight_power_estimated")?.unwrap_or(false),
        offset: m.opt_vec("offset")?,
        dispersion_held: m.opt_f64("dispersion_held")?,
        design_summary,
        rss: m.f64("rss")?,
        coef_names: m.strs("coef_names")?,
        coef,
        vcov,
        aliased: m.bools("aliased")?,
    })
}
