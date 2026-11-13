// The pins of the reference engines, `validation/grid/versions.json`, shared by
// `tests/golden_provenance.rs` and `tests/inference_references.rs` through
// `#[path]`. Depends only on `std` and `serde_json`, so both build without
// default features.

use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// Pin key (`lme4`, `R`, `mpmath`, ...) to the pinned version as
/// `versions.json` spells it.
pub type Pins = BTreeMap<String, String>;

/// `v` with every `-` mapped to `.`, as `validation/grid/engines/versions.R`'s
/// `norm_version` does: CRAN spells a version `2.0-6`, `packageVersion()` writes
/// `2.0.6`, and the two are one version.
pub fn norm_version(v: &str) -> String {
    v.replace('-', ".")
}

/// The pins of `validation/grid/versions.json`. Keys starting with `_` are notes,
/// not pins, and drop out.
pub fn load_pins() -> Pins {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../validation/grid/versions.json");
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let json: Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    json.as_object()
        .unwrap_or_else(|| panic!("{}: not a JSON object", path.display()))
        .iter()
        .filter(|(k, _)| !k.starts_with('_'))
        .map(|(k, v)| {
            let v = v
                .as_str()
                .unwrap_or_else(|| panic!("{}: pin {k} is not a string", path.display()));
            (k.clone(), v.to_string())
        })
        .collect()
}

/// A synthetic pin map for the rule tests, in the CRAN spelling where CRAN
/// uses a dash.
pub fn pins() -> Pins {
    [
        ("lme4", "2.0-6"),
        ("R", "4.5.3"),
        ("mpmath", "1.4.1"),
        ("GLMMadaptive", "0.9-7"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}
