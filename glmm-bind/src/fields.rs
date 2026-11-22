//! The argument and record field names a binding converts by name rather
//! than by value type. The names are those of the registry's parameters and
//! of the records the calls return; the test
//! `every_binding_field_name_is_an_argument_or_a_record_field` checks each
//! one, so a rename there fails it.

/// Arguments whose map holds data columns by name (both bindings): their
/// entries are named `column "x"` in an error, so a column's name is never
/// read as a field's.
pub const COLUMN_MAPS: &[&str] = &["numeric", "factors", "new_numeric", "new_factors"];

/// Python: the argument sent as int64 bytes; every other bytes value is
/// float64.
pub const PY_INT64_ARG: &str = "row_index";

/// Python: the fit record's per-row fields, which cross back as bytes.
pub const PY_FIT_BYTES: &[&str] = &[
    "fitted",
    "ranef",
    "ranef_levels",
    "y",
    "weights",
    "offset",
    "condvar",
    "condvar_x",
];

/// Python: the fields of a fit's `robust` record that cross back as bytes,
/// the ones that grow with the rows (HC2 has one cluster per row).
pub const PY_ROBUST_BYTES: &[&str] = &["er", "row_e", "row_x", "row_phi"];

/// Python: each call's result fields that cross back as bytes.
pub const PY_BYTES: &[(&str, &[&str])] = &[
    ("fit", PY_FIT_BYTES),
    (
        "predict",
        &[
            "fit",
            "se",
            "lower",
            "upper",
            "future_lower",
            "future_upper",
        ],
    ),
    ("model_matrix", &["x"]),
    ("plain_column", &["values", "codes"]),
];

/// Python: the records nested in a result whose fields cross back as bytes,
/// by field name: `compare`'s `fit0` and `fit1` (fit records) and a fit
/// record's `robust`.
pub const PY_NESTED_BYTES: &[(&str, &[&str])] = &[
    ("fit0", PY_FIT_BYTES),
    ("fit1", PY_FIT_BYTES),
    ("robust", PY_ROBUST_BYTES),
];

/// R: the fit record's fields that hold positions (in its
/// `satterthwaite_record` and `robust` records too), counted from 1 in R and
/// from 0 here. Only these are read −1 when R passes them back.
pub const R_INDEX_FIELDS: &[&str] = &["retained", "re_term_index", "null_cluster"];

/// R: the counts of a fit record (its own and its notes') that leave as
/// integers; every other count leaves as a double, as the fit objects saved
/// by earlier versions hold them.
pub const R_INT_FIELDS: &[&str] = &["objective_nodes", "evals"];
