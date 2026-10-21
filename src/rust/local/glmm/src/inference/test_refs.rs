// Reader for the frozen inference references: the `inference_references` block
// of `validation/manifest.json`, the JSONs under
// `validation/goldens/inference/<kind>/`, and the `oracle-tier` entries of
// `validation/grid/divergences.json`.
//
// One file serves two kinds of test. In-crate unit tests reach it as
// `crate::inference::test_refs`; integration tests `include!` it verbatim
// through `tests/oracle_support/inference.rs`. So it uses no `crate::` or
// `glmm::` path, carries no inner attribute, and depends only on `std`, `serde`
// and `serde_json`. Files are read at run time from `CARGO_MANIFEST_DIR`, so the
// `wasm32-wasip1` lib run (with the directory mapped) reads them too.

use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Which inference reference an entry is: its manifest `kind` string and its
/// directory under `validation/goldens/inference/`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// lmerTest Satterthwaite tests (`goldens_lmertest.R`).
    Lmertest,
    /// Exact restricted likelihood ratio tests (`goldens_rlrt.R`).
    Rlrt,
    /// ANOVA tables and model comparisons (`goldens_anova.R`).
    Anova,
}

impl Kind {
    /// Every kind, in manifest order.
    pub const ALL: [Kind; 3] = [Kind::Lmertest, Kind::Rlrt, Kind::Anova];

    /// The manifest `kind` string, which is also the directory name.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Lmertest => "lmertest",
            Kind::Rlrt => "rlrt",
            Kind::Anova => "anova",
        }
    }

    /// The kind named by a manifest `kind` string, `None` for any other string.
    fn parse(s: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.as_str() == s)
    }

    /// `CARGO_MANIFEST_DIR/validation/goldens/inference/<kind>`. Git keeps no
    /// empty directory, so it may be absent: a missing directory holds no
    /// reference.
    pub fn dir(self) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("validation/goldens/inference")
            .join(self.as_str())
    }
}

/// One entry of the manifest's `inference_references` block.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The reference name, also its JSON's file stem.
    pub name: String,
    /// The kind, which picks the directory.
    pub kind: Kind,
    /// The CSV path relative to `validation/`, `None` on a fit-free entry.
    pub data: Option<String>,
    /// The R formula, `None` on a fit-free entry.
    pub r_formula: Option<String>,
    /// The generator script, a bare file name (no directory) of a file under
    /// `validation/tools/`.
    pub generator: String,
    /// The `data` columns read as factors (the optional `factors`); empty when
    /// absent.
    pub factors: Vec<String>,
    /// Why the JSON does not exist yet; set while the reference is not frozen.
    pub pending_reference: Option<String>,
    /// The whole entry, for the kind-specific fields its owning step reads.
    pub raw: Value,
}

impl Entry {
    /// Parses one manifest entry. `name`, `kind` and `generator` are strings;
    /// `data` and `r_formula` are present, each a string or `null`;
    /// `factors` is absent or an array of strings; `pending_reference` is
    /// absent, `null` or a string. Any other shape is an error naming the field.
    pub fn from_raw(raw: Value) -> Result<Entry, String> {
        let string = |field: &str| match raw.get(field) {
            Some(Value::String(s)) => Ok(s.clone()),
            _ => Err(format!("`{field}` is not a string in {raw}")),
        };
        let nullable = |field: &str| match raw.get(field) {
            Some(Value::String(s)) => Ok(Some(s.clone())),
            Some(Value::Null) => Ok(None),
            _ => Err(format!("`{field}` is neither a string nor null in {raw}")),
        };
        let kind_str = string("kind")?;
        let kind = Kind::parse(&kind_str).ok_or_else(|| {
            format!("unknown kind \"{kind_str}\" (one of lmertest, rlrt, anova) in {raw}")
        })?;
        let factors = match raw.get("factors") {
            None => Vec::new(),
            Some(Value::Array(names)) => names
                .iter()
                .map(|f| f.as_str().map(str::to_string))
                .collect::<Option<Vec<String>>>()
                .ok_or_else(|| format!("`factors` is not an array of strings in {raw}"))?,
            Some(_) => return Err(format!("`factors` is not an array of strings in {raw}")),
        };
        let pending_reference = match raw.get("pending_reference") {
            None => None,
            Some(_) => nullable("pending_reference")?,
        };
        Ok(Entry {
            name: string("name")?,
            kind,
            data: nullable("data")?,
            r_formula: nullable("r_formula")?,
            generator: string("generator")?,
            factors,
            pending_reference,
            raw,
        })
    }

    /// The entry carries `pending_reference`: registered, not frozen.
    pub fn is_pending(&self) -> bool {
        self.pending_reference.is_some()
    }
}

/// The parsed `validation/manifest.json`.
pub fn manifest() -> Value {
    read_json(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/manifest.json"))
}

fn read_json(path: &std::path::Path) -> Value {
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The entries of `kind`, pending ones included. Panics on an entry that does
/// not parse.
pub fn entries(kind: Kind) -> Vec<Entry> {
    manifest()["inference_references"]
        .as_array()
        .expect("manifest.json has an `inference_references` array")
        .iter()
        .map(|raw| Entry::from_raw(raw.clone()).unwrap_or_else(|e| panic!("manifest.json: {e}")))
        .filter(|e| e.kind == kind)
        .collect()
}

/// The frozen reference `<name>.json` of `kind`. Panics when it is missing or
/// does not parse.
pub fn load(kind: Kind, name: &str) -> Value {
    read_json(&kind.dir().join(format!("{name}.json")))
}

/// The registry scope of the in-crate oracle tiers.
const SCOPE: &str = "oracle-tier";

/// One entry of `validation/grid/divergences.json`, the fields the inference
/// tier reads.
#[derive(Clone, Debug, Deserialize)]
pub struct DivEntry {
    /// Stable key.
    pub id: String,
    /// The reference name the entry is about.
    pub cell: String,
    /// The scopes the entry applies to.
    pub comparison: Vec<String>,
    /// The quantities it covers, each the label one driver adjudicates:
    /// `df`, `p_value`, `lrt`, `f` and `confint` end to end; `df_at_lme4`,
    /// `dendf_at_lme4` and `f_at_lme4` for gate 1 at lme4's θ̂.
    pub quantities: Vec<String>,
    /// The largest relative difference it covers.
    pub max_rel: f64,
}

/// The `oracle-tier` entries of `validation/grid/divergences.json`, in file
/// order.
pub fn registry_entries() -> Vec<DivEntry> {
    #[derive(Deserialize)]
    struct File {
        entries: Vec<DivEntry>,
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("validation/grid/divergences.json");
    let file: File = serde_json::from_value(read_json(&path))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    file.entries
        .into_iter()
        .filter(DivEntry::in_scope)
        .collect()
}

impl DivEntry {
    fn in_scope(&self) -> bool {
        self.comparison.iter().any(|c| c == SCOPE)
    }
}

/// How the registry adjudicates one over-band comparison.
#[derive(Clone, Debug, PartialEq)]
pub enum Coverage {
    /// No `oracle-tier` entry covers this cell and quantity: the caller fails.
    NotCovered,
    /// Entry `id` covers it at this magnitude: the caller passes and records
    /// `id` as fired.
    Within {
        /// The covering entry.
        id: String,
    },
    /// Entry `id` covers it, but the difference outgrew `max_rel`: the caller
    /// fails.
    Exceeds {
        /// The covering entry.
        id: String,
        /// Its bound.
        max_rel: f64,
    },
}

/// The first `oracle-tier` entry on `cell` listing `quantity` decides:
/// `Within` at `observed_rel <= max_rel`, `Exceeds` otherwise, so a NaN
/// difference gives `Exceeds` and never passes. The match rule of the goldens
/// tier's `divergence::Registry::covers`, which agrees on NaN (it returns
/// `NotDocumented` and fires nothing), without its fired set: the caller keeps
/// the ids of its `Within` hits and passes them to [`stale_entries`].
pub fn covers(entries: &[DivEntry], cell: &str, quantity: &str, observed_rel: f64) -> Coverage {
    let Some(e) = entries
        .iter()
        .find(|e| e.in_scope() && e.cell == cell && e.quantities.iter().any(|q| q == quantity))
    else {
        return Coverage::NotCovered;
    };
    if observed_rel <= e.max_rel {
        Coverage::Within { id: e.id.clone() }
    } else {
        Coverage::Exceeds {
            id: e.id.clone(),
            max_rel: e.max_rel,
        }
    }
}

/// The ids of the `oracle-tier` entries that are stale for one driver: the
/// entry's `cell` is among the reference names the driver `compared`, and its
/// id is not in `fired`. An entry on a reference the driver did not compare is
/// not its to judge.
pub fn stale_entries(
    scoped: &[DivEntry],
    compared: &BTreeSet<String>,
    fired: &BTreeSet<String>,
) -> Vec<String> {
    scoped
        .iter()
        .filter(|e| e.in_scope() && compared.contains(&e.cell) && !fired.contains(&e.id))
        .map(|e| e.id.clone())
        .collect()
}
