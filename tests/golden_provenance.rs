//! Provenance of the frozen goldens: every reference under `validation/goldens/`
//! that records its engine as `engine` / `engine_version` (the `m3_goldens`
//! entries at the top level, the weights-tier `datasets` rows under `weights/`)
//! carries both stamps, and the version matches the engine's pin in
//! `validation/grid/versions.json`.
//!
//! The pin key is the engine's text before `::` (`lme4::glmer` reads `lme4`,
//! `stats::glm` reads `R` with the stamp's `R-` prefix dropped); versions
//! compare with `-` read as `.`, so the CRAN spelling `2.0-6` and the
//! `packageVersion()` spelling `2.0.6` are one version.
//!
//! A weights-tier result's `engine` is also checked against the fitter its row
//! implies ([`weights_engine`]), so a fixed-only row fitted by `stats` or `MASS`
//! cannot carry an lme4 stamp.
//!
//! A failure means one of five things, and the message says which:
//! - a JSON file directly under `goldens/` or `goldens/weights/` has no
//!   manifest entry: register it, or its stamps are never checked;
//! - the engine has no pin: pin it in `versions.json`;
//! - the golden was written under another version: regenerate it under the pin,
//!   or, when it stays frozen under the old one, declare that in its manifest
//!   entry as `"frozen_under": "<the stamp as recorded>"` and say why in the
//!   comment that governs the entry;
//! - a `frozen_under` no longer matches: drop it when the stamp is on the pin,
//!   correct it when it names another version;
//! - a weights-tier result names another fitter than its row's: fix the
//!   generator's stamp and regenerate the row.
//!
//! Never edit a golden to make its stamp pass. The inference references
//! (`goldens/inference/`) carry their versions in `provenance` and are checked
//! by `tests/inference_references.rs`. No feature gate: this reads JSON only and
//! runs in CI.

#[path = "oracle_support/pins.rs"]
mod pin_support;

use pin_support::{load_pins, norm_version, Pins};
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The `versions.json` key of an engine stamp: the text before `::`, with
/// `stats` read as `R`.
fn pin_key(engine: &str) -> &str {
    match engine.split_once("::").map_or(engine, |(prefix, _)| prefix) {
        "stats" => "R",
        prefix => prefix,
    }
}

/// The `engine` stamp `tools/goldens_weights.R` writes for a weights-tier row:
/// `lme4` on a mixed row (a random term in `r_formula`), else the fixed-only
/// fitter of its family. Mirrors that script's `fit_call` and `fit_one_fixed`;
/// change together.
fn weights_engine(row: &Value) -> &'static str {
    let mixed = row["r_formula"].as_str().is_some_and(|f| f.contains('|'));
    match (mixed, row["family"].as_str()) {
        (true, _) => "lme4",
        (false, Some("gaussian")) => "stats::lm",
        (false, Some("negbin")) => "MASS::glm.nb",
        (false, _) => "stats::glm",
    }
}

/// The findings on one golden's `engine` / `engine_version` stamps. `frozen_under`
/// is its manifest entry's declaration, the stamp exactly as recorded, under
/// which an off-pin golden passes. Each finding starts with `name`.
fn stamp_findings(
    name: &str,
    golden: &Value,
    frozen_under: Option<&str>,
    pins: &Pins,
) -> Vec<String> {
    let engine = golden.get("engine").and_then(Value::as_str);
    let stamp = golden.get("engine_version").and_then(Value::as_str);
    let (engine, stamp) = match (engine, stamp) {
        (Some(e), Some(s)) => (e, s),
        (e, s) => {
            let missing: Vec<&str> = [("engine", e), ("engine_version", s)]
                .into_iter()
                .filter(|(_, v)| v.is_none())
                .map(|(k, _)| k)
                .collect();
            return vec![format!(
                "{name}: no string `{}` stamp",
                missing.join("` or `")
            )];
        }
    };
    let key = pin_key(engine);
    let Some(pin) = pins.get(key) else {
        return vec![format!(
            "{name}: engine {engine} has no pin `{key}` in validation/grid/versions.json; pin it there"
        )];
    };
    let version = if key == "R" {
        stamp.strip_prefix("R-").unwrap_or(stamp)
    } else {
        stamp
    };
    let on_pin = norm_version(version) == norm_version(pin);
    match (on_pin, frozen_under) {
        (true, None) => vec![],
        (true, Some(f)) => vec![format!(
            "{name}: frozen_under {f} is stale: the {engine} stamp {stamp} matches the {key} pin {pin}; drop the field"
        )],
        (false, Some(f)) if f == stamp => vec![],
        (false, Some(f)) => vec![format!(
            "{name}: frozen_under {f} differs from the {engine} stamp {stamp} (the {key} pin is {pin})"
        )],
        (false, None) => vec![format!(
            "{name}: the {engine} stamp {stamp} does not match the {key} pin {pin} in \
             validation/grid/versions.json; regenerate it under the pin, or declare \
             \"frozen_under\": \"{stamp}\" in its manifest entry"
        )],
    }
}

/// Reads and parses `path`, or says why not as a finding on `name`.
fn read_golden(name: &str, path: &Path) -> Result<Value, String> {
    let raw =
        std::fs::read_to_string(path).map_err(|e| format!("{name}: {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("{name}: {}: {e}", path.display()))
}

#[test]
fn every_frozen_golden_matches_its_pin() {
    let pins = load_pins();
    let suite = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation");
    let manifest =
        read_golden("manifest", &suite.join("manifest.json")).unwrap_or_else(|e| panic!("{e}"));
    let block = |b: &str| -> &Vec<Value> {
        manifest[b]
            .as_array()
            .unwrap_or_else(|| panic!("manifest.json has a `{b}` array"))
    };
    // A pending entry (`pending_reference`, a non-empty reason) has no golden yet.
    let m3: Vec<&Value> = block("m3_goldens")
        .iter()
        .filter(|s| {
            !s["pending_reference"]
                .as_str()
                .is_some_and(|r| !r.is_empty())
        })
        .collect();
    let weights: Vec<&Value> = block("datasets")
        .iter()
        .filter(|s| s["tier"].as_str() == Some("weights"))
        .collect();
    let names = |entries: &[Value], tier: Option<&str>| -> Vec<String> {
        entries
            .iter()
            .filter(|s| tier.is_none_or(|t| s["tier"].as_str() == Some(t)))
            .filter_map(|s| s["name"].as_str().map(str::to_owned))
            .collect()
    };
    // Every name of each directory's block, pending entries included.
    let registered = [
        names(block("m3_goldens"), None),
        names(block("datasets"), Some("weights")),
    ];
    let mut findings = Vec::new();
    let mut checked = [0usize; 2];
    let mut on_disk = [0usize; 2];
    let mut declared = 0;
    for (i, (entries, dir)) in [
        (&m3, suite.join("goldens")),
        (&weights, suite.join("goldens/weights")),
    ]
    .into_iter()
    .enumerate()
    {
        let listing = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
            .map(|entry| entry.expect("a readable directory entry").path());
        for path in listing.filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "json")) {
            on_disk[i] += 1;
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            if !registered[i].iter().any(|n| n == stem) {
                findings.push(format!(
                    "{}: no manifest entry, so its stamps are never checked; register it",
                    path.display()
                ));
            }
        }
        for s in entries {
            let name = s["name"].as_str().expect("manifest entry has a name");
            let frozen_under = s["frozen_under"].as_str();
            declared += usize::from(frozen_under.is_some());
            match read_golden(name, &dir.join(format!("{name}.json"))) {
                Ok(golden) => {
                    checked[i] += 1;
                    findings.extend(stamp_findings(name, &golden, frozen_under, &pins));
                    if i == 1 {
                        let want = weights_engine(s);
                        let got = golden["engine"].as_str().unwrap_or("");
                        if got != want {
                            findings.push(format!(
                                "{name}: engine {got}, but its row is fitted by {want}"
                            ));
                        }
                    }
                }
                Err(e) => findings.push(e),
            }
        }
    }
    let [m3_checked, w_checked] = checked;
    println!(
        "checked {m3_checked} goldens, {w_checked} weights-tier results; {declared} declared frozen_under"
    );
    let [m3_disk, w_disk] = on_disk;
    assert!(
        m3_checked == m3.len()
            && w_checked == weights.len()
            && !m3.is_empty()
            && !weights.is_empty(),
        "checked {m3_checked} of {} goldens ({m3_disk} JSON files on disk) and {w_checked} of {} \
         weights-tier results ({w_disk} on disk; each count must be > 0):\n{}",
        m3.len(),
        weights.len(),
        findings.join("\n")
    );
    assert!(
        findings.is_empty(),
        "{} golden stamps out of provenance:\n{}",
        findings.len(),
        findings.join("\n")
    );
}

/// Each rule on synthetic stamps against [`pin_support::pins`].
mod rules {
    use super::pin_support::pins;
    use super::{pin_key, stamp_findings};
    use serde_json::json;

    #[test]
    fn pin_key_takes_the_prefix() {
        assert_eq!(pin_key("lme4::glmer.nb"), "lme4");
        assert_eq!(pin_key("lme4"), "lme4");
        assert_eq!(pin_key("glmmTMB::glmmTMB"), "glmmTMB");
        assert_eq!(pin_key("GLMMadaptive"), "GLMMadaptive");
        assert_eq!(pin_key("MASS::glm.nb"), "MASS");
        assert_eq!(pin_key("stats::glm+ML-dispersion"), "R");
    }

    #[test]
    fn stamp_on_pin_passes_in_either_spelling() {
        for v in ["2.0.6", "2.0-6"] {
            let g = json!({"engine": "lme4::lmer", "engine_version": v});
            assert!(stamp_findings("g", &g, None, &pins()).is_empty());
        }
        let g = json!({"engine": "GLMMadaptive", "engine_version": "0.9.7"});
        assert!(stamp_findings("g", &g, None, &pins()).is_empty());
    }

    #[test]
    fn stamp_off_pin_fails_unless_declared() {
        let g = json!({"engine": "lme4::glmer", "engine_version": "1.1.38"});
        let f = stamp_findings("g", &g, None, &pins());
        assert_eq!(f.len(), 1);
        assert!(f[0].contains("1.1.38") && f[0].contains("2.0-6"));
        assert!(stamp_findings("g", &g, Some("1.1.38"), &pins()).is_empty());
        assert_eq!(stamp_findings("g", &g, Some("1.1.37"), &pins()).len(), 1);
    }

    #[test]
    fn frozen_under_on_a_pinned_stamp_is_stale() {
        let g = json!({"engine": "lme4::lmer", "engine_version": "2.0.6"});
        let f = stamp_findings("g", &g, Some("2.0.6"), &pins());
        assert_eq!(f.len(), 1);
        assert!(f[0].contains("stale"));
    }

    #[test]
    fn missing_or_unknown_stamp_fails() {
        assert_eq!(
            stamp_findings("g", &json!({"engine": "lme4::lmer"}), None, &pins()).len(),
            1
        );
        assert_eq!(
            stamp_findings("g", &json!({"engine_version": "2.0.6"}), None, &pins()).len(),
            1
        );
        let g = json!({"engine": "nlme::lme", "engine_version": "3.1"});
        assert_eq!(stamp_findings("g", &g, None, &pins()).len(), 1);
    }

    #[test]
    fn stats_engine_reads_the_r_pin() {
        let ok = json!({"engine": "stats::glm", "engine_version": "R-4.5.3"});
        assert!(stamp_findings("g", &ok, None, &pins()).is_empty());
        let old = json!({"engine": "stats::glm", "engine_version": "R-4.5.2"});
        assert_eq!(stamp_findings("g", &old, None, &pins()).len(), 1);
    }
}
