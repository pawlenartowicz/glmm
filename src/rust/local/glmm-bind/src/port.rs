//! The calling port, and the argument names and literals that differ between
//! the ports, so one message serves both.

/// The port a call comes from; each native crate passes its own as a
/// constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Port {
    /// The Python package `glmm`.
    Python,
    /// The R package `fastglmm`.
    R,
}

/// `(key, python, r)`.
const SPELLINGS: &[(&str, &str, &str)] = &[
    ("nagq", "nagq", "nAGQ"),
    ("reml", "reml", "REML"),
    ("reml_true", "reml=True", "REML = TRUE"),
    ("reml_false", "reml=False", "REML = FALSE"),
    ("wald_se", "wald_se", "wald.se"),
    ("warm_start", "warm_start", "start"),
    (
        "warm_start_dispersion",
        r#"warm_start["dispersion"]"#,
        "start$dispersion",
    ),
    (
        "warm_start_record",
        "a dict with keys",
        "a list with elements",
    ),
    ("init_theta", "init_theta", "init.theta"),
    ("re_form", "re_form", "re.form"),
    ("se_fit", "se_fit", "se.fit"),
    ("allow_new_levels", "allow_new_levels", "allow.new.levels"),
    (
        "allow_new_levels_true",
        "allow_new_levels=True",
        "allow.new.levels = TRUE",
    ),
    ("true", "True", "TRUE"),
    ("false", "False", "FALSE"),
    ("null", "None", "NULL"),
    ("na", r#""NA""#, "NA"),
    (
        "seed_range",
        "None or a whole number from 0 to 2**64 - 1",
        "NULL or one whole number from 0 to 2^53",
    ),
    (
        "compare_call",
        "glmm.compare(data, f0, f1)",
        "compare(data, f0, f1)",
    ),
    (
        "weight_power_estimate",
        r#"weight_power="estimate""#,
        r#"weight_power = "estimate""#,
    ),
    ("suffix", "[glmm]", "[fastglmm]"),
];

/// `key` as `port` spells it.
///
/// # Panics
///
/// On a key missing from the table, a bug in bind that `call` reports as an
/// `internal` error.
pub(crate) fn spell(port: Port, key: &str) -> &'static str {
    let &(_, python, r) = SPELLINGS
        .iter()
        .find(|(k, _, _)| *k == key)
        .unwrap_or_else(|| panic!("no spelling for {key:?}"));
    match port {
        Port::Python => python,
        Port::R => r,
    }
}

#[cfg(test)]
mod tests {
    use super::SPELLINGS;

    #[test]
    fn every_spelling_has_both_ports() {
        for (i, (key, python, r)) in SPELLINGS.iter().enumerate() {
            assert!(!python.is_empty() && !r.is_empty(), "{key}");
            assert!(
                SPELLINGS[..i].iter().all(|(k, _, _)| k != key),
                "{key} twice"
            );
        }
    }
}
