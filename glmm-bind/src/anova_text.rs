//! The texts of the anova tables both ports print: a per-term table's
//! heading, notes and per-row references, read at `anova(m)` time from the
//! stored table ([`term_table_text`]), and a comparison's heading, reference
//! and notes, which the comparison record carries
//! ([`crate::records::comparison_record`]). The ports lay out the columns
//! and print these lines as they get them.

use glmm::formula::anova::{Comparison, ComparisonTest, Differs};
use glmm::inference::{Objective, Reference};

use crate::coerce;
use crate::error::BindError;
use crate::inputs::Inputs;
use crate::summary::sig4;
use crate::value::{Record, Value};

/// The first note of a per-term or pairwise table of a fit that did not
/// converge.
pub(crate) const NOT_CONVERGED: &str =
    "The fit did not converge, so its estimates are not at a maximum \
     and these tests are not reliable. Change the model until it converges (a simpler random \
     part often helps), then test again.";

/// The second heading line of a likelihood-ratio per-term table.
const LRT_LINE: &str = "Likelihood-ratio chi-squared tests (each term's model refitted without it)";

/// The second heading line of a Wald chi-squared table under `test = "auto"`.
const WALD_LINE: &str = "Wald chi-squared tests";

/// The last note of a Satterthwaite F table with the `Eta2_partial` column.
const ETA2_LINE: &str =
    "Partial eta-squared uses the Satterthwaite denominator df, so it is approximate.";

/// The exact restricted likelihood-ratio test's reference.
const RLRT: &str = "exact RLRT null distribution (Crainiceanu & Ruppert 2004)";

/// `compare`'s note on a Gaussian pair it fitted by ML.
const ML_NOTE: &str = "Both models were fitted by ML because their fixed parts differ.";

/// The heading's name of each denominator-df source, in heading order.
const DF_METHODS: &[(&str, &str)] = &[
    ("pustejovsky_tipton", "Pustejovsky-Tipton's"),
    ("bell_mccaffrey", "Bell-McCaffrey's"),
    ("satterthwaite", "Satterthwaite's"),
];

/// Why `"auto"` gave a chi-squared table its test, by the fit's
/// `term_test_auto` reason (`orchestrate::choice_tags`).
const AUTO_REASONS: &[(&str, &str)] = &[
    ("poisson_mixed", "Poisson model with random effects"),
    ("binomial_mixed", "binomial model with random effects"),
    (
        "negative_binomial_mixed",
        "negative binomial model with random effects",
    ),
    ("other_mixed", "Gamma model with random effects"),
    ("no_random_effects", "model without random effects"),
];

/// A df in a reference, never in scientific notation: a whole df as an
/// integer at any size, the nearest whole number from 10000 up, else four
/// significant digits; NaN as `NA`, an infinite df as `Inf` or `-Inf`.
/// Mirrors `fmt_df` in `glmm-py/glmm/summary.py`, which prints the df cells
/// beside these references: change together.
fn fmt_df(x: f64) -> String {
    if x.is_nan() {
        "NA".to_owned()
    } else if x.is_infinite() {
        if x > 0.0 { "Inf" } else { "-Inf" }.to_owned()
    } else if x == x.round() || x.abs() >= 1e4 {
        format!("{x:.0}")
    } else {
        sig4(x)
    }
}

/// `F(a, b)`, or `""` when either df is NaN (an aliased term, a row with no
/// denominator df).
fn f_reference(num_df: f64, den_df: f64) -> String {
    if num_df.is_nan() || den_df.is_nan() {
        String::new()
    } else {
        format!("F({}, {})", fmt_df(num_df), fmt_df(den_df))
    }
}

/// `χ²(k)`, or `""` when the df is NaN.
fn chisq_reference(df: f64) -> String {
    if df.is_nan() {
        String::new()
    } else {
        format!("χ²({})", fmt_df(df))
    }
}

/// A comparison's heading line, by its test and the likelihood it compares.
pub(crate) fn comparison_heading(test: &ComparisonTest) -> String {
    match test {
        ComparisonTest::Lr(t) => match t.objective {
            Objective::Reml => {
                "Model comparison: likelihood ratio test on the REML criterion".to_owned()
            }
            Objective::Laplace => "Model comparison: likelihood ratio test (Laplace)".to_owned(),
            Objective::Agq { nodes } => {
                format!("Model comparison: likelihood ratio test (AGQ {nodes})")
            }
            _ => "Model comparison: likelihood ratio test (ML)".to_owned(),
        },
        _ => "Model comparison: F test".to_owned(),
    }
}

/// The distribution a comparison's p-value is read from, in words.
pub(crate) fn comparison_reference(r: &Reference) -> Result<String, BindError> {
    Ok(match *r {
        Reference::ChiSquared { df } => chisq_reference(df as f64),
        Reference::ChiBarSquared { df_low } => format!(
            "50:50 mixture of {} and {}",
            chisq_reference(df_low as f64),
            chisq_reference(df_low as f64 + 1.0)
        ),
        Reference::ExactRlrt { nsim: 0, .. } => {
            format!("{RLRT}; no draws needed: the statistic is 0, so p = 1")
        }
        Reference::ExactRlrt { nsim, .. } => format!("{RLRT}, {nsim} draws"),
        Reference::F { num_df, den_df, .. } => f_reference(num_df, den_df),
        other => {
            return Err(BindError::internal(format!(
                "no rendering for the reference {other:?}"
            )))
        }
    })
}

/// A comparison's notes: the crate's, then, from `compare` on a Gaussian
/// pair (`ml_note`), why both were fitted by ML.
pub(crate) fn comparison_notes(c: &Comparison, ml_note: bool) -> Vec<String> {
    let mut notes = c.note_lines();
    if ml_note && matches!(c.test, ComparisonTest::Lr(_)) && c.differs != Differs::Random {
        notes.push(ML_NOTE.to_owned());
    }
    notes
}

/// The code of a stored per-term table's `error`: its `error_code`, or on a
/// table stored by glmm 0.9.3, which has none, `internal` when the message
/// says so and `fit_error` otherwise.
fn error_code(code: Option<String>, message: &str) -> Result<&'static str, BindError> {
    match code.as_deref() {
        Some("internal") => Ok("internal"),
        Some("fit_error") => Ok("fit_error"),
        Some(other) => Err(BindError::invalid_argument(format!(
            "table.error_code must be \"internal\" or \"fit_error\", got {other:?}"
        ))),
        None if message.starts_with("internal error") => Ok("internal"),
        None => Ok("fit_error"),
    }
}

/// `term_table_text`: the heading lines, notes and per-row `references` of a
/// stored per-term table read for `anova(m, test = test)`, or of
/// `contrast_test`'s row, whose `ss_type` is null, so its heading names no
/// type. A stored refusal raises its error under its `error_code`.
pub(crate) fn term_table_text(inp: &mut Inputs) -> Result<Value, BindError> {
    let mut t = inp.record("table")?;
    if let Some(message) = t.opt_string("error")? {
        let code = error_code(t.opt_string("error_code")?, &message)?;
        return Err(BindError::new(code, message));
    }
    let test = inp.string("test")?;
    let df_method = inp.opt_string("df_method")?;
    let converged = inp.bool("converged")?;
    let ss = t.opt_string("ss_type")?;
    let prefix = ss.as_ref().map_or(String::new(), |s| format!("Type {s} "));
    let labels = t.opt_strs("label")?.unwrap_or_default();
    let kind = t.opt_strs("kind")?.unwrap_or_default();
    let num_df = t.opt_vec("num_df")?.unwrap_or_default();
    let den_df = t.opt_vec("den_df")?.unwrap_or_default();
    let den_df_kind = t.opt_strs("den_df_kind")?.unwrap_or_default();
    // The heading names the df method of the rows that have df: on a fit with
    // the pass switched off or failed, no row does. A table with no rows
    // takes its kind from the fit's `df_method`.
    let (f_rows, kinds): (bool, Vec<String>) = match kind.first() {
        Some(k) => (
            k == "F",
            den_df_kind
                .into_iter()
                .zip(&den_df)
                .filter(|(_, d)| d.is_finite())
                .map(|(k, _)| k)
                .collect(),
        ),
        None => {
            let m = df_method.unwrap_or_else(|| "none".to_owned());
            (m != "none", vec![m])
        }
    };
    let satterthwaite = kinds.iter().any(|k| k == "satterthwaite");
    let lrt_table = t.opt_string("test")?.as_deref() == Some("lrt");
    let heading = if f_rows {
        let names: Vec<&str> = DF_METHODS
            .iter()
            .filter(|(k, _)| kinds.iter().any(|x| x == k))
            .map(|(_, name)| *name)
            .collect();
        let method = match names[..] {
            [] => String::new(),
            [one] => format!(" with {one} method"),
            _ => format!(" with {} methods", names.join(" and ")),
        };
        vec![format!("{prefix}Analysis of Variance Table{method}")]
    } else if lrt_table || test == "auto" {
        // The second line names the test and why it ran; an explicit "Wald"
        // keeps the one-line heading.
        let line = if lrt_table {
            let reduced = if ss.as_deref() == Some("III") {
                " on the reduced design"
            } else {
                ""
            };
            format!("{LRT_LINE}{reduced}")
        } else {
            WALD_LINE.to_owned()
        };
        let why = if test == "LRT" {
            r#"test = "LRT""#.to_owned()
        } else {
            let mut auto = inp.record("term_test_auto")?;
            let reason = auto.string("reason")?;
            let words = AUTO_REASONS
                .iter()
                .find(|(tag, _)| *tag == reason)
                .ok_or_else(|| {
                    BindError::invalid_argument(format!(
                        "term_test_auto.reason {reason:?} is not a reason \"auto\" gives"
                    ))
                })?
                .1;
            format!(r#"test = "auto": {words}"#)
        };
        vec![
            format!("{prefix}Analysis of Deviance Table"),
            format!("{line}; {why}"),
        ]
    } else {
        vec![format!(
            "{prefix}Analysis of Deviance Table (Wald chi-squared tests)"
        )]
    };
    let row_notes = match t.value("row_notes") {
        Value::Null => Vec::new(),
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
    // A fit stored before the Eta2_partial column, and a contrast, have none.
    let eta2 = t.has("eta2_partial");
    let notes: Vec<String> = (!converged)
        .then(|| NOT_CONVERGED.to_owned())
        .into_iter()
        .chain(t.opt_strs("notes")?.unwrap_or_default())
        .chain(
            labels
                .iter()
                .zip(&row_notes)
                .flat_map(|(label, notes)| notes.iter().map(move |n| format!("{label}: {n}"))),
        )
        .chain((f_rows && eta2 && satterthwaite).then(|| ETA2_LINE.to_owned()))
        .collect();
    // Each row's reference, from its own df.
    let references: Vec<String> = if f_rows {
        num_df
            .iter()
            .zip(&den_df)
            .map(|(&a, &b)| f_reference(a, b))
            .collect()
    } else {
        num_df.iter().map(|&k| chisq_reference(k)).collect()
    };
    Ok(Value::Record(
        Record::new()
            .with("heading", Value::StrVec(heading))
            .with("notes", Value::StrVec(notes))
            .with("references", Value::StrVec(references)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::Port;
    use crate::records::text;

    fn lines(r: &Record, key: &str) -> Vec<String> {
        match r.get(key) {
            Some(Value::StrVec(v)) => v.clone(),
            other => panic!("{key}: {other:?}"),
        }
    }

    fn render(table: Record, test: &str, extra: Record) -> Result<Record, BindError> {
        let mut args = Record::new()
            .with("table", Value::Record(table))
            .with("test", text(test));
        for (k, v) in extra.into_fields() {
            args.push(&k, v);
        }
        if args.get("converged").is_none() {
            args.push("converged", Value::Bool(true));
        }
        match term_table_text(&mut Inputs::new(args, Port::Python))? {
            Value::Record(r) => Ok(r),
            other => panic!("{other:?}"),
        }
    }

    fn f_table(den_df_kind: &[&str], den_df: &[f64]) -> Record {
        let n = den_df.len();
        Record::new()
            .with("error", Value::Null)
            .with("ss_type", text("III"))
            .with("test", text("f"))
            .with(
                "label",
                Value::StrVec((0..n).map(|i| format!("t{i}")).collect()),
            )
            .with("kind", Value::StrVec(vec!["F".to_owned(); n]))
            .with("num_df", Value::Vec(vec![1.0; n]))
            .with("den_df", Value::Vec(den_df.to_vec()))
            .with(
                "den_df_kind",
                Value::StrVec(den_df_kind.iter().map(|s| (*s).to_owned()).collect()),
            )
            .with("eta2_partial", Value::Vec(vec![0.1; n]))
            .with("row_notes", Value::List(vec![Value::StrVec(vec![]); n]))
            .with("notes", Value::StrVec(vec![]))
    }

    fn chisq_table(ss: &str, test: &str) -> Record {
        Record::new()
            .with("error", Value::Null)
            .with("ss_type", text(ss))
            .with("test", text(test))
            .with("label", Value::StrVec(vec!["x".to_owned()]))
            .with("kind", Value::StrVec(vec!["chisq".to_owned()]))
            .with("num_df", Value::Vec(vec![2.0]))
            .with("den_df", Value::Vec(vec![f64::NAN]))
            .with("den_df_kind", Value::StrVec(vec!["none".to_owned()]))
            .with(
                "row_notes",
                Value::List(vec![Value::StrVec(vec!["a row note".to_owned()])]),
            )
            .with("notes", Value::StrVec(vec!["a table note".to_owned()]))
    }

    fn auto(reason: &str) -> Record {
        Record::new().with(
            "term_test_auto",
            Value::Record(Record::new().with("reason", text(reason))),
        )
    }

    #[test]
    fn fmt_df_matches_both_ports() {
        let cases: &[(f64, &str)] = &[
            (1.0, "1"),
            (29996.0, "29996"),
            (107.4123, "107.4"),
            (12345.6, "12346"),
            (2.5, "2.5"),
            (0.5, "0.5"),
            (9.99995, "10"),
            (999.95, "1000"),
            (5.00004, "5"),
            (10007.0, "10007"),
            (12.3456, "12.35"),
            (0.99996, "1"),
            (f64::NAN, "NA"),
            (f64::INFINITY, "Inf"),
            (f64::NEG_INFINITY, "-Inf"),
        ];
        for &(x, want) in cases {
            assert_eq!(fmt_df(x), want, "{x}");
        }
    }

    #[test]
    fn the_f_heading_names_each_df_method_of_its_rows() {
        let cases: &[(&[&str], &[f64], &str)] = &[
            (
                &["satterthwaite", "satterthwaite"],
                &[10.5, 3.0],
                "Type III Analysis of Variance Table with Satterthwaite's method",
            ),
            (
                &["residual"],
                &[30.0],
                "Type III Analysis of Variance Table",
            ),
            (
                &["pustejovsky_tipton", "satterthwaite"],
                &[8.5, 187.5],
                "Type III Analysis of Variance Table with Pustejovsky-Tipton's and \
                 Satterthwaite's methods",
            ),
            (
                &["bell_mccaffrey"],
                &[12.0],
                "Type III Analysis of Variance Table with Bell-McCaffrey's method",
            ),
            // A row with no df names no method.
            (
                &["satterthwaite"],
                &[f64::NAN],
                "Type III Analysis of Variance Table",
            ),
        ];
        for &(kinds, df, want) in cases {
            let r = render(f_table(kinds, df), "auto", Record::new()).unwrap();
            assert_eq!(lines(&r, "heading"), [want]);
            let eta2 = lines(&r, "notes").contains(&ETA2_LINE.to_owned());
            assert_eq!(eta2, kinds.contains(&"satterthwaite") && df[0].is_finite());
        }
        let r = render(f_table(&["satterthwaite"], &[107.4]), "auto", Record::new()).unwrap();
        assert_eq!(lines(&r, "references"), ["F(1, 107.4)"]);
    }

    #[test]
    fn a_table_with_no_rows_takes_its_kind_from_df_method() {
        let empty = || f_table(&[], &[]);
        let sat = Record::new().with("df_method", text("satterthwaite"));
        let r = render(empty(), "auto", sat).unwrap();
        assert_eq!(
            lines(&r, "heading"),
            ["Type III Analysis of Variance Table with Satterthwaite's method"]
        );
        assert_eq!(lines(&r, "notes"), [ETA2_LINE]);
        let none = Record::new().with("df_method", text("none"));
        let r = render(empty(), "Wald", none).unwrap();
        assert_eq!(
            lines(&r, "heading"),
            ["Type III Analysis of Deviance Table (Wald chi-squared tests)"]
        );
    }

    #[test]
    fn the_chi_squared_heading_names_the_test_and_why_auto_chose_it() {
        for (reason, words) in AUTO_REASONS {
            let r = render(chisq_table("II", "wald"), "auto", auto(reason)).unwrap();
            assert_eq!(
                lines(&r, "heading"),
                [
                    "Type II Analysis of Deviance Table".to_owned(),
                    format!(r#"Wald chi-squared tests; test = "auto": {words}"#),
                ]
            );
        }
        let r = render(chisq_table("III", "lrt"), "LRT", Record::new()).unwrap();
        assert_eq!(
            lines(&r, "heading"),
            [
                "Type III Analysis of Deviance Table",
                "Likelihood-ratio chi-squared tests (each term's model refitted without it) on \
                 the reduced design; test = \"LRT\"",
            ]
        );
        let r = render(chisq_table("II", "lrt"), "auto", auto("poisson_mixed")).unwrap();
        assert_eq!(
            lines(&r, "heading")[1],
            "Likelihood-ratio chi-squared tests (each term's model refitted without it); \
             test = \"auto\": Poisson model with random effects"
        );
        let r = render(chisq_table("III", "wald"), "Wald", Record::new()).unwrap();
        assert_eq!(
            lines(&r, "heading"),
            ["Type III Analysis of Deviance Table (Wald chi-squared tests)"]
        );
        assert_eq!(lines(&r, "notes"), ["a table note", "x: a row note"]);
        assert_eq!(lines(&r, "references"), ["χ²(2)"]);
    }

    #[test]
    fn a_contrast_names_no_type_and_a_stalled_fit_says_so_first() {
        let mut t = chisq_table("III", "wald");
        t.set("ss_type", Value::Null);
        let stalled = Record::new().with("converged", Value::Bool(false));
        let r = render(t, "Wald", stalled).unwrap();
        assert_eq!(
            lines(&r, "heading"),
            ["Analysis of Deviance Table (Wald chi-squared tests)"]
        );
        assert_eq!(lines(&r, "notes")[0], NOT_CONVERGED);
    }

    #[test]
    fn a_stored_refusal_raises_under_its_code() {
        let stored = |code: Option<&str>, message: &str| {
            let mut t = Record::new().with("error", text(message));
            if let Some(c) = code {
                t.push("error_code", text(c));
            }
            render(t, "auto", Record::new()).unwrap_err()
        };
        let e = stored(Some("internal"), "internal error while building the table");
        assert_eq!(
            (e.code, e.message.as_str()),
            ("internal", "internal error while building the table")
        );
        assert_eq!(stored(Some("fit_error"), "a refusal").code, "fit_error");
    }

    #[test]
    fn a_table_stored_by_0_9_3_reads_its_error_code_from_the_message() {
        let stored = |message: &str| {
            let t = Record::new().with("error", text(message));
            render(t, "auto", Record::new()).unwrap_err().code
        };
        assert_eq!(
            stored("internal error while building the per-term table: x"),
            "internal"
        );
        assert_eq!(stored("the formula lacks a marginal term"), "fit_error");
    }

    #[test]
    fn the_comparison_references_in_words() {
        let words = |r| comparison_reference(&r).unwrap();
        assert_eq!(words(Reference::ChiSquared { df: 2 }), "χ²(2)");
        assert_eq!(
            words(Reference::ChiBarSquared { df_low: 1 }),
            "50:50 mixture of χ²(1) and χ²(2)"
        );
        assert_eq!(
            words(Reference::ExactRlrt {
                nsim: 10_000,
                seed: 1
            }),
            "exact RLRT null distribution (Crainiceanu & Ruppert 2004), 10000 draws"
        );
        assert_eq!(
            words(Reference::ExactRlrt { nsim: 0, seed: 1 }),
            "exact RLRT null distribution (Crainiceanu & Ruppert 2004); no draws needed: the \
             statistic is 0, so p = 1"
        );
    }
}
