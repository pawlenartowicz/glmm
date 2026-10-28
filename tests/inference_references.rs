//! Registration of the inference references: the manifest's
//! `inference_references` block against the JSONs under
//! `validation/goldens/inference/<kind>/`.
//!
//! A frozen reference that no manifest entry names cannot be regenerated, and an
//! entry whose JSON is missing gates nothing; both fail here. So does an
//! `oracle-tier` entry of `validation/grid/divergences.json` whose `cell` is
//! neither a golden nor an inference reference: no driver compares that cell,
//! so none can find the entry stale. Every JSON carries the provenance keys its
//! generator writes: the R keys on every kind but `conditioning`, whose
//! generator is a Python script with keys of its own. The rules are pure
//! functions over the manifest entries, the registry and the files on disk,
//! with synthetic cases beside them, so each rule is shown to fire. No feature
//! gate, so this runs in CI on the default and `--no-default-features` builds.
//! It reads the reader by `#[path]`, not through `mod oracle_support`, which
//! imports `glmm::formula`.

#[allow(dead_code)]
#[path = "oracle_support/inference.rs"]
mod inference_support;

use inference_support::{DivEntry, Entry, Kind};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

/// One kind directory: file stem to parsed JSON. A missing directory is empty.
type KindDir = BTreeMap<String, Value>;

/// The `provenance` keys `validation/tools/inference_common.R::provenance` writes,
/// on every kind but `conditioning`.
const PROVENANCE_KEYS: [&str; 7] = [
    "r_version",
    "packages",
    "rng_kind",
    "blas",
    "lapack",
    "threads",
    "generator",
];

/// The `provenance` keys of a `conditioning` reference, which a Python generator
/// writes. Mirrors `validation/tools/goldens_glm_band.py::provenance`; change
/// together.
const PY_PROVENANCE_KEYS: [&str; 4] = ["python_version", "packages", "mp_dps", "generator"];

/// The rules across kinds, over the raw block: each entry parses (so its `kind`
/// is one of the seven), names are unique and none is in `reserved` (the
/// `datasets` and `m3_goldens` names), `data` names a file or is `null` only on
/// a fit-free entry, `generator` is a bare file name naming a file under
/// `tools/`, and each `factors` name is a column of the `data` CSV. `exists`
/// takes a path relative to `validation/`, `header` gives the column names of
/// such a CSV. Returns the entries that parse.
fn manifest_rules(
    raw: &[Value],
    reserved: &BTreeSet<String>,
    exists: &dyn Fn(&str) -> bool,
    header: &dyn Fn(&str) -> Vec<String>,
) -> (Vec<Entry>, Vec<String>) {
    let mut entries = Vec::new();
    let mut found = Vec::new();
    let mut seen = BTreeSet::new();
    for r in raw {
        let e = match Entry::from_raw(r.clone()) {
            Ok(e) => e,
            Err(err) => {
                found.push(format!("manifest entry: {err}"));
                continue;
            }
        };
        let name = &e.name;
        if !seen.insert(name.clone()) {
            found.push(format!("{name}: duplicate name in inference_references"));
        }
        if reserved.contains(name) {
            found.push(format!("{name}: also a `datasets` or `m3_goldens` name"));
        }
        if let Some(data) = e.data.as_deref().filter(|p| !is_plain_relative(p)) {
            found.push(format!(
                "{name}: data {data} is not a plain relative path (no root, no `..`)"
            ));
        }
        if !is_bare_name(&e.generator) {
            found.push(format!(
                "{name}: generator {} is not a bare file name (no directory)",
                e.generator
            ));
        }
        match (&e.data, &e.r_formula) {
            (Some(data), _) if !exists(data) => found.push(format!(
                "{name}: data {data} names no file under validation/"
            )),
            (None, Some(_)) => found.push(format!(
                "{name}: `data` is null but `r_formula` is set; only a fit-free entry has no data"
            )),
            _ => {}
        }
        if !exists(&format!("tools/{}", e.generator)) {
            found.push(format!(
                "{name}: generator {} names no file under validation/tools/",
                e.generator
            ));
        }
        let columns = match &e.data {
            Some(data) if exists(data) => header(data),
            _ => Vec::new(),
        };
        for f in e.factors.iter().filter(|f| !columns.contains(f)) {
            found.push(format!(
                "{name}: factor {f} is not a column of its data CSV"
            ));
        }
        entries.push(e);
    }
    (entries, found)
}

/// Every component is a plain name: no root, no `.`, no `..`, so the path stays
/// inside the directory it is joined to.
fn is_plain_relative(path: &str) -> bool {
    Path::new(path)
        .components()
        .all(|c| matches!(c, Component::Normal(_)))
}

/// Exactly one component, a plain name: a file directly inside the directory it
/// is joined to.
fn is_bare_name(path: &str) -> bool {
    let mut components = Path::new(path).components();
    matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(_)), None)
    )
}

/// The rules of one kind: its entries against the JSONs of its directory.
fn kind_rules(kind: Kind, entries: &[Entry], jsons: &KindDir) -> Vec<String> {
    let k = kind.as_str();
    let mut found = Vec::new();
    for stem in jsons.keys() {
        if !entries.iter().any(|e| &e.name == stem) {
            found.push(format!(
                "{k}/{stem}.json has no manifest entry of that kind and name"
            ));
        }
    }
    for e in entries {
        let name = &e.name;
        let frozen = jsons.contains_key(name);
        match e.pending_reference.as_deref() {
            None if !frozen => found.push(format!(
                "{name}: not pending, but {k}/{name}.json does not exist"
            )),
            Some("") => found.push(format!(
                "{name}: pending_reference is empty, an exclusion with no reason"
            )),
            Some(_) if frozen => found.push(format!(
                "{name}: pending_reference is set but {k}/{name}.json exists; drop the field"
            )),
            _ => {}
        }
    }
    let keys: &[&str] = match kind {
        Kind::Conditioning => &PY_PROVENANCE_KEYS,
        _ => &PROVENANCE_KEYS,
    };
    for (stem, json) in jsons {
        match json.get("provenance").and_then(Value::as_object) {
            None => found.push(format!("{k}/{stem}.json: no `provenance` object")),
            Some(p) => {
                for key in keys.iter().filter(|key| !p.contains_key(**key)) {
                    found.push(format!("{k}/{stem}.json: provenance lacks `{key}`"));
                }
            }
        }
        if json.get("name").and_then(Value::as_str) != Some(stem) {
            found.push(format!(
                "{k}/{stem}.json: `name` is {}, not its file stem",
                json.get("name").unwrap_or(&Value::Null)
            ));
        }
    }
    found
}

/// Every violation, across kinds and per kind. `dirs` is keyed by the kind's
/// manifest string; a kind with no key has an empty directory.
fn violations(
    raw: &[Value],
    dirs: &BTreeMap<&str, KindDir>,
    reserved: &BTreeSet<String>,
    exists: &dyn Fn(&str) -> bool,
    header: &dyn Fn(&str) -> Vec<String>,
) -> Vec<String> {
    let (entries, mut found) = manifest_rules(raw, reserved, exists, header);
    let empty = KindDir::new();
    for kind in Kind::ALL {
        let of_kind: Vec<Entry> = entries.iter().filter(|e| e.kind == kind).cloned().collect();
        found.extend(kind_rules(
            kind,
            &of_kind,
            dirs.get(kind.as_str()).unwrap_or(&empty),
        ));
    }
    found
}

/// Every `oracle-tier` entry's `cell` is a golden of the goldens tier or an
/// inference reference name. A driver judges an entry stale only on a cell it
/// compared, so an entry on any other cell (a typo, a dataset name for a golden
/// name, a removed reference) is never reported.
fn registry_rules(
    scoped: &[DivEntry],
    goldens: &BTreeSet<String>,
    references: &BTreeSet<String>,
) -> Vec<String> {
    scoped
        .iter()
        .filter(|e| !goldens.contains(&e.cell) && !references.contains(&e.cell))
        .map(|e| {
            format!(
                "{}: oracle-tier cell {} is neither a golden nor an inference reference name",
                e.id, e.cell
            )
        })
        .collect()
}

/// The JSONs of `kind` on disk.
fn read_kind_dir(kind: Kind) -> KindDir {
    let dir = kind.dir();
    let listing = match std::fs::read_dir(&dir) {
        Ok(listing) => listing,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return KindDir::new(),
        Err(e) => panic!("{}: {e}", dir.display()),
    };
    let mut jsons = KindDir::new();
    for item in listing {
        let path = item.expect("dir entry").path();
        if path.extension().is_some_and(|e| e == "json") {
            let stem = path
                .file_stem()
                .expect("stem")
                .to_string_lossy()
                .into_owned();
            let json = inference_support::load(kind, &stem);
            jsons.insert(stem, json);
        }
    }
    jsons
}

#[test]
fn inference_references_are_registered() {
    let manifest = inference_support::manifest();
    let raw = manifest["inference_references"]
        .as_array()
        .expect("manifest.json has an `inference_references` array");
    let mut reserved = BTreeSet::new();
    // The goldens tier's corpus (`tests/validation_oracle.rs::corpus`): every
    // `m3_goldens` entry and the weights-tier `datasets` rows. A pending
    // `m3_goldens` entry counts too: its entries are judged once it is frozen.
    let mut goldens = BTreeSet::new();
    for block in ["datasets", "m3_goldens"] {
        let specs = manifest[block]
            .as_array()
            .unwrap_or_else(|| panic!("manifest.json has a `{block}` array"));
        for s in specs {
            let name = s["name"].as_str().expect("named").to_string();
            if block == "m3_goldens" || s["tier"].as_str() == Some("weights") {
                goldens.insert(name.clone());
            }
            reserved.insert(name);
        }
    }
    let dirs: BTreeMap<&str, KindDir> = Kind::ALL
        .into_iter()
        .map(|k| (k.as_str(), read_kind_dir(k)))
        .collect();
    let suite = Path::new(env!("CARGO_MANIFEST_DIR")).join("validation");
    let header = |p: &str| -> Vec<String> {
        let path = suite.join(p);
        let raw =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        raw.lines()
            .next()
            .unwrap_or("")
            .split(',')
            .map(|s| s.trim().trim_matches('"').to_string())
            .collect()
    };
    let mut found = violations(raw, &dirs, &reserved, &|p| suite.join(p).is_file(), &header);
    let references: BTreeSet<String> = raw
        .iter()
        .filter_map(|r| r["name"].as_str().map(str::to_string))
        .collect();
    found.extend(registry_rules(
        &inference_support::registry_entries(),
        &goldens,
        &references,
    ));
    assert!(
        found.is_empty(),
        "inference references out of registration:\n{}",
        found.join("\n")
    );
}

#[test]
fn oracle_tier_entry_on_neither_a_golden_nor_a_reference_is_rejected() {
    let entry = |id: &str, cell: &str| DivEntry {
        id: id.to_string(),
        cell: cell.to_string(),
        comparison: vec!["oracle-tier".to_string()],
        quantities: vec!["beta".to_string()],
        max_rel: 0.01,
    };
    let found = registry_rules(
        &[
            entry("on-golden", "cbpp_probit_glmm"),
            entry("on-reference", "ref_a"),
            entry("on-data", "cbpp"),
        ],
        &BTreeSet::from(["cbpp_probit_glmm".to_string()]),
        &BTreeSet::from(["ref_a".to_string()]),
    );
    assert_eq!(
        found,
        ["on-data: oracle-tier cell cbpp is neither a golden nor an inference reference name"]
    );
}

/// Each rule on synthetic input: a registered, frozen `lmertest` reference
/// that passes, perturbed once per test.
mod rules {
    use super::{violations, KindDir, PROVENANCE_KEYS, PY_PROVENANCE_KEYS};
    use serde_json::{json, Value};
    use std::collections::{BTreeMap, BTreeSet};

    struct World {
        raw: Vec<Value>,
        dirs: BTreeMap<&'static str, KindDir>,
        reserved: BTreeSet<String>,
        files: BTreeSet<String>,
        headers: BTreeMap<String, Vec<String>>,
    }

    fn reference(name: &str) -> Value {
        json!({
            "name": name,
            "provenance": {
                "r_version": "4.5.1",
                "packages": { "lmerTest": "3.1-3" },
                "rng_kind": ["Mersenne-Twister", "Inversion", "Rejection"],
                "blas": "libflexiblas.so.3",
                "lapack": "libflexiblas.so.3",
                "threads": { "OMP_NUM_THREADS": "1" },
                "generator": "goldens_lmertest.R"
            }
        })
    }

    impl World {
        fn valid() -> World {
            let entry = json!({
                "name": "ref_a",
                "kind": "lmertest",
                "data": "data/simulated/sim_factorial.csv",
                "r_formula": "y ~ a * b + (1 | g)",
                "generator": "goldens_lmertest.R"
            });
            World {
                raw: vec![entry],
                dirs: BTreeMap::from([(
                    "lmertest",
                    KindDir::from([("ref_a".to_string(), reference("ref_a"))]),
                )]),
                reserved: ["Dyestuff", "sleepstudy_lmm"].map(String::from).into(),
                files: [
                    "data/simulated/sim_factorial.csv",
                    "tools/goldens_lmertest.R",
                ]
                .map(String::from)
                .into(),
                headers: BTreeMap::from([(
                    "data/simulated/sim_factorial.csv".to_string(),
                    ["subj", "g", "h", "x", "y"].map(String::from).into(),
                )]),
            }
        }

        fn json(&mut self, kind: &'static str, stem: &str) -> &mut Value {
            self.dirs
                .get_mut(kind)
                .and_then(|d| d.get_mut(stem))
                .expect("json present")
        }

        fn violations(&self) -> Vec<String> {
            violations(
                &self.raw,
                &self.dirs,
                &self.reserved,
                &|p| self.files.contains(p),
                &|p| self.headers.get(p).cloned().unwrap_or_default(),
            )
        }

        fn assert_flags(&self, needle: &str) {
            let found = self.violations();
            assert!(
                found.iter().any(|m| m.contains(needle)),
                "expected a violation containing {needle:?}, got {found:?}"
            );
        }
    }

    #[test]
    fn orphan_json_is_rejected() {
        let mut w = World::valid();
        w.dirs.insert(
            "rlrt",
            KindDir::from([("ref_b".to_string(), reference("ref_b"))]),
        );
        w.assert_flags("rlrt/ref_b.json has no manifest entry");
    }

    #[test]
    fn non_pending_entry_without_json_is_rejected() {
        let mut w = World::valid();
        w.dirs.clear();
        w.assert_flags("ref_a: not pending, but lmertest/ref_a.json does not exist");
    }

    #[test]
    fn pending_entry_with_json_is_rejected() {
        let mut w = World::valid();
        w.raw[0]["pending_reference"] = json!("awaiting the R run");
        w.assert_flags("ref_a: pending_reference is set but lmertest/ref_a.json exists");
    }

    #[test]
    fn pending_entry_with_empty_reason_is_rejected() {
        let mut w = World::valid();
        w.dirs.clear();
        w.raw[0]["pending_reference"] = json!("");
        w.assert_flags("ref_a: pending_reference is empty");
    }

    #[test]
    fn duplicate_name_across_kinds_is_rejected() {
        let mut w = World::valid();
        let mut twin = w.raw[0].clone();
        twin["kind"] = json!("rlrt");
        w.raw.push(twin);
        w.dirs.insert(
            "rlrt",
            KindDir::from([("ref_a".to_string(), reference("ref_a"))]),
        );
        w.assert_flags("ref_a: duplicate name");
    }

    #[test]
    fn name_of_a_dataset_or_golden_is_rejected() {
        for name in ["Dyestuff", "sleepstudy_lmm"] {
            let mut w = World::valid();
            w.raw[0]["name"] = json!(name);
            w.dirs.insert(
                "lmertest",
                KindDir::from([(name.to_string(), reference(name))]),
            );
            w.assert_flags(&format!("{name}: also a `datasets` or `m3_goldens` name"));
        }
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let mut w = World::valid();
        w.raw[0]["kind"] = json!("lmerTest");
        w.assert_flags("unknown kind \"lmerTest\"");
    }

    #[test]
    fn json_without_provenance_or_one_of_its_keys_is_rejected() {
        let mut w = World::valid();
        w.json("lmertest", "ref_a")
            .as_object_mut()
            .unwrap()
            .remove("provenance");
        w.assert_flags("lmertest/ref_a.json: no `provenance` object");
        for key in PROVENANCE_KEYS {
            let mut w = World::valid();
            w.json("lmertest", "ref_a")["provenance"]
                .as_object_mut()
                .unwrap()
                .remove(key);
            w.assert_flags(&format!("lmertest/ref_a.json: provenance lacks `{key}`"));
        }
    }

    #[test]
    fn conditioning_reference_needs_the_python_provenance_keys() {
        let conditioning = |provenance: Value| {
            let mut w = World::valid();
            w.raw[0]["kind"] = json!("conditioning");
            let mut json = reference("ref_a");
            json["provenance"] = provenance;
            w.dirs =
                BTreeMap::from([("conditioning", KindDir::from([("ref_a".to_string(), json)]))]);
            w
        };
        let python = json!({
            "python_version": "3.14.2",
            "packages": { "mpmath": "1.4.1" },
            "mp_dps": 50,
            "generator": "goldens_glm_band.py"
        });
        assert_eq!(
            conditioning(python.clone()).violations(),
            Vec::<String>::new()
        );
        for key in PY_PROVENANCE_KEYS {
            let mut provenance = python.clone();
            provenance.as_object_mut().unwrap().remove(key);
            conditioning(provenance).assert_flags(&format!(
                "conditioning/ref_a.json: provenance lacks `{key}`"
            ));
        }
        let r_keys = reference("ref_a")["provenance"].clone();
        conditioning(r_keys).assert_flags("conditioning/ref_a.json: provenance lacks `mp_dps`");
    }

    #[test]
    fn json_name_other_than_its_stem_is_rejected() {
        let mut w = World::valid();
        w.json("lmertest", "ref_a")["name"] = json!("ref_x");
        w.assert_flags("lmertest/ref_a.json: `name` is");
    }

    #[test]
    fn data_naming_no_file_is_rejected() {
        let mut w = World::valid();
        w.raw[0]["data"] = json!("data/simulated/missing.csv");
        w.assert_flags("ref_a: data data/simulated/missing.csv names no file");
    }

    #[test]
    fn null_data_with_a_formula_is_rejected() {
        let mut w = World::valid();
        w.raw[0]["data"] = Value::Null;
        w.assert_flags("ref_a: `data` is null but `r_formula` is set");
    }

    #[test]
    fn fit_free_entry_passes() {
        let mut w = World::valid();
        w.raw[0]["data"] = Value::Null;
        w.raw[0]["r_formula"] = Value::Null;
        assert_eq!(w.violations(), Vec::<String>::new());
    }

    #[test]
    fn null_pending_flag_is_not_pending() {
        let mut w = World::valid();
        w.raw[0]["pending_reference"] = Value::Null;
        assert_eq!(w.violations(), Vec::<String>::new());
    }

    #[test]
    fn generator_naming_no_file_is_rejected() {
        let mut w = World::valid();
        w.raw[0]["generator"] = json!("goldens_missing.R");
        w.assert_flags("ref_a: generator goldens_missing.R names no file");
    }

    #[test]
    fn factor_that_is_no_column_of_the_csv_is_rejected() {
        let mut w = World::valid();
        w.raw[0]["factors"] = json!(["g", "k"]);
        w.assert_flags("ref_a: factor k is not a column of its data CSV");
        let found = w.violations();
        assert_eq!(found.len(), 1, "{found:?}");
    }

    #[test]
    fn factors_that_are_not_an_array_are_rejected() {
        let mut w = World::valid();
        w.raw[0]["factors"] = json!("g");
        w.assert_flags("manifest entry: `factors` is not an array of strings");
    }

    #[test]
    fn data_path_leaving_its_directory_is_rejected() {
        // Each path is also put on disk, so only the path rule can fire.
        for path in ["../GLMM/x.csv", "/home/x.csv", "data/../x.csv"] {
            let mut w = World::valid();
            w.raw[0]["data"] = json!(path);
            w.files.insert(path.to_string());
            w.assert_flags(&format!("ref_a: data {path} is not a plain relative path"));
        }
    }

    #[test]
    fn generator_that_is_not_a_bare_file_name_is_rejected() {
        // Each script is also put on disk under tools/, so only the bare-name
        // rule can fire; the second is a plain relative path.
        for path in ["../grid/engines/lme4.R", "prep/gen_inference_data.R"] {
            let mut w = World::valid();
            w.raw[0]["generator"] = json!(path);
            w.files.insert(format!("tools/{path}"));
            w.assert_flags(&format!("ref_a: generator {path} is not a bare file name"));
        }
    }
}
