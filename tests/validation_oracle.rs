//! Tier 2 — the cross-engine tier.
//!
//! Asserts the crate against the frozen lme4 / MASS / GLMMadaptive / glmmTMB references
//! at `validation/grid/tol.R`'s agreement bands: 61 goldens in `validation/goldens/` plus
//! the weights tier (rungs 29-43) in `validation/results/lme4_simulated/`. Reads frozen
//! JSON, so it needs neither R nor Julia at runtime — but it refits all 76, so
//! it is off by default:
//!
//! ```sh
//! cargo test --features oracle-tests
//! ```
//!
//! A failure here means glmm and the reference no longer agree by more than the
//! calibration allowed. There are exactly three honest endings: a bug in glmm
//! (fix it), a documented expected divergence (add the entry), or
//! a reference-vs-reference gap. Widening a band
//! to make red go green is none of them.
//!
//! This is NOT `validation/grid/run.sh`, which refits the external oracles to check
//! what they say today. Different claims, both kept.
#![cfg(all(feature = "oracle-tests", feature = "formula"))]

mod oracle_support;

use glmm::WaldSe;
use oracle_support::{
    align_coefs, assert_abs, assert_coefs, assert_rel, load_golden, refit, refit_with, tol, Golden,
};
use serde_json::Value;

#[test]
fn aligned_dev_default_is_minus_two_loglik() {
    let g = load_golden("pastes_lmm");
    let ll = g.estimates.loglik.expect("gaussian golden has loglik");
    assert_eq!(oracle_support::dev_align::aligned_dev(&g), Some(-2.0 * ll));
}

/// The exactly-six GLMMadaptive vector-RE-AGQ goldens that carry no `loglik` —
/// the deviance gate's only loud exclusions (`DEV-NA` in
/// `goldens_agree_with_the_references`). Pinned by name so a golden losing its
/// `loglik` (or one of these six regaining it) changes this test, not a corpus
/// count that could silently drift.
#[test]
fn dev_align_none_matches_the_six_vector_agq_goldens() {
    let expect_none = [
        "sim_binomial_slope1_agq_k7",
        "sim_binomial_slope1_agq_k11",
        "sim_binomial_slope2_agq_k7",
        "sim_binomial_slope2_agq_k11",
        "sim_poisson_slope1_agq_k7",
        "sim_poisson_slope1_agq_k11",
    ];
    for (g, _) in corpus() {
        let is_none = oracle_support::dev_align::aligned_dev(&g).is_none();
        assert_eq!(
            is_none,
            expect_none.contains(&g.name.as_str()),
            "{}: aligned_dev None-ness disagrees with the pinned set of six",
            g.name
        );
    }
}

/// The goldens whose reference maximises a different objective from glmm's
/// (`objective_differs`), pinned by name as the six above are, and each with a
/// twin in the corpus — glmmTMB (`_tmb`) for the lme4 GLMMs, the ML-dispersion
/// GLM (`_ml`) for the `stats::glm` Gamma GLMs — since the twin is what gates
/// glmm's deviance on that data and model.
#[test]
fn objective_differs_matches_the_pinned_goldens() {
    let expect = [
        ("cbpp_probit_glmm", "_tmb"),
        ("sim_cloglog_glmm", "_tmb"),
        ("sim_gamma_glmm", "_tmb"),
        ("sim_gamma_inv_glmm", "_tmb"),
        ("sim_sparse_gamma", "_tmb"),
        ("sim_nb_glmm", "_tmb"),
        ("sim_nb_nested_glmm", "_tmb"),
        ("sim_sparse_nb", "_tmb"),
        ("sim_gamma_glm", "_ml"),
        ("sim_gamma_inv_glm", "_ml"),
        ("sim_scale_gamma_inv_glm", "_ml"),
        // Weights tier: no twin. Its precision loglik is gated against R
        // directly by `fit::glm_tests::fit_glm_gamma_weighted_matches_r`.
        ("glm_gamma", ""),
    ];
    let names: std::collections::BTreeSet<String> =
        corpus().into_iter().map(|(g, _)| g.name).collect();
    for (g, _) in corpus() {
        assert_eq!(
            objective_differs(&g).is_some(),
            expect.iter().any(|(n, _)| *n == g.name),
            "{}: objective_differs disagrees with the pinned set",
            g.name
        );
    }
    for (n, suffix) in expect.into_iter().filter(|(_, s)| !s.is_empty()) {
        let twin = format!("{n}{suffix}");
        assert!(names.contains(&twin), "{n}: no twin `{twin}` in the corpus");
        let t = load_golden(&twin);
        let engine = match suffix {
            "_tmb" => "glmmTMB::glmmTMB",
            _ => "stats::glm+ML-dispersion",
        };
        assert_eq!(t.engine, engine, "{twin}: twin is not a {engine} golden");
        assert!(
            objective_differs(&t).is_none(),
            "{twin}: the twin must stay in the deviance gate"
        );
    }
}

/// Why a golden's deviance cannot be compared with glmm's, when it cannot.
///
/// lme4's `glmer` builds the Laplace log-determinant from the expected (Fisher)
/// weight, and on a non-canonical link that is not the curvature of the
/// integrand at the mode, so its logLik is a different function of the
/// parameters; on Gamma it also plugs in its `pwrss/n` scale instead of
/// maximising over φ. glmm maximises the Laplace approximation itself, and gates
/// its deviance against the glmmTMB twin of each such golden. The parameters
/// are still compared, and their differences are registered divergences.
///
/// `stats::glm`'s Gamma logLik plugs in `D/Σw` for φ; glmm's is maximised at
/// the ML φ̂, and is gated against the `_ml` twin (same β, ML φ̂ as `dispersion`;
/// its `se` is a different quantity from glmm's Pearson-scaled SE and stays
/// unasserted).
fn objective_differs(g: &Golden) -> Option<&'static str> {
    if g.kind == "glm" && g.family == "gamma" && g.weights_suite {
        return Some(
            "R's logLik.glm multiplies each row's log-density by its weight and \
             plugs in D/Σw for φ; glmm maximises the precision-weighted \
             likelihood (row i's shape wᵢ/φ), a different function of the \
             parameters — gated against R directly by \
             fit_glm_gamma_weighted_matches_r",
        );
    }
    if g.kind == "glm" && g.family == "gamma" && g.engine == "stats::glm" {
        return Some(
            "stats::glm: Gamma logLik at the D/Σw plug-in, not the ML φ̂ — deviance \
             gated on the `_ml` twin",
        );
    }
    if g.kind != "glmm" || !g.engine.starts_with("lme4::") {
        return None;
    }
    let link = g.link.as_str().unwrap_or("");
    match (g.family.as_str(), link) {
        ("gamma", _) => Some(
            "lme4: Fisher-weight Laplace with the pwrss/n plug-in Gamma scale — deviance \
             gated on the `_tmb` twin",
        ),
        ("negbin", _) | ("binomial", "probit" | "cloglog") => Some(
            "lme4: Fisher-weight Laplace on a non-canonical link — deviance gated on the \
             `_tmb` twin",
        ),
        _ => None,
    }
}

/// The whole cross-engine corpus: the `m3_goldens` tree plus the prior-weights
/// suite. Two frozen reference trees, one set of assertions.
fn corpus() -> Vec<(Golden, Vec<String>)> {
    let mut all = m3_corpus();
    let weights = weights_corpus();
    assert_eq!(weights.len(), 15, "weights tier lost a rung");
    all.extend(weights);
    all
}

fn factors_of(spec: &Value) -> Vec<String> {
    spec["factors"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|f| f.as_str().expect("factor name").to_string())
                .collect()
        })
        .unwrap_or_default()
}

/// A manifest `m3_goldens` entry whose reference JSON has not been generated yet:
/// `pending_reference` carries the reason as a non-empty string.
///
/// Registering a spec and freezing its reference are two separate acts, and the
/// spec has to land FIRST — `validation/tools/goldens_agq.R` reads the manifest to know
/// what to fit, so there is no ordering in which the JSON can exist before the
/// entry does. Without this flag that gap makes [`m3_corpus`] panic on a missing
/// file and takes the whole 56-golden tier down with it, which is the opposite of
/// what a not-yet-frozen addition should cost. Flagged entries are skipped here
/// and reported by `pending_references_are_absent_and_reasoned`, which also stops
/// the flag outliving the JSON's arrival: once the reference exists, the field
/// must go, and removing it is what puts the golden under Tier 2.
fn is_pending(spec: &Value) -> bool {
    spec["pending_reference"]
        .as_str()
        .is_some_and(|r| !r.is_empty())
}

/// The `validation/goldens/` tree, with the manifest's factor list. The manifest is
/// the registry: a golden that no script can regenerate is not an
/// oracle, so anything in `validation/goldens/` must appear in `m3_goldens`, and
/// `all_goldens_are_registered` holds that line. Entries still awaiting their
/// reference are excluded ([`is_pending`]).
fn m3_corpus() -> Vec<(Golden, Vec<String>)> {
    let manifest: Value =
        serde_json::from_str(include_str!("../validation/manifest.json")).expect("manifest parses");
    let specs = manifest["m3_goldens"]
        .as_array()
        .expect("manifest has m3_goldens");
    specs
        .iter()
        .filter(|s| !is_pending(s))
        .map(|s| {
            let name = s["name"].as_str().expect("spec has a name");
            let path = format!(
                "{}/validation/goldens/{name}.json",
                env!("CARGO_MANIFEST_DIR")
            );
            let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let mut golden: Golden =
                serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{name}: {e}"));
            golden.source = path;
            (golden, factors_of(s))
        })
        .collect()
}

/// The weights tier (rungs 29-43 of `validation/manifest.json`), read where it
/// was generated rather than copied into `validation/goldens/`.
///
/// These are promoted into `m3_goldens` coverage: asserted from
/// `cargo test` instead of only by `grid/compare.R` on a machine with R. Reading them
/// in place achieves that without a second copy of 15 frozen JSONs. The `goldens/`
/// and `results/` trees would risk drifting onto different `tolPwrss`
/// settings if it held a second, separately-maintained copy of these values;
/// reading `results/` in place keeps them on one footing. A third overlapping
/// tree would rebuild exactly that hazard.
///
/// Its result JSONs use the harness's own field names (`dataset`, no `kind`, no
/// `r_formula` — the model lives in the manifest, not the result). The three
/// keys are injected here, in one visible place, so `Golden` stays strict for
/// both trees.
fn weights_corpus() -> Vec<(Golden, Vec<String>)> {
    let dir = format!("{}/validation", env!("CARGO_MANIFEST_DIR"));
    let manifest: Value =
        serde_json::from_str(include_str!("../validation/manifest.json")).expect("manifest parses");
    manifest["datasets"]
        .as_array()
        .expect("manifest has datasets")
        .iter()
        .filter(|s| s["tier"].as_str() == Some("weights"))
        .map(|s| {
            let name = s["name"].as_str().expect("spec has a name");
            let path = format!("{dir}/results/lme4_simulated/{name}.json");
            let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let mut v: Value = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{name}: {e}"));
            let r_formula = s["r_formula"].clone();
            let obj = v.as_object_mut().expect("result is an object");
            obj.insert("name".into(), s["name"].clone());
            obj.insert("data".into(), s["name"].clone());
            obj.insert("kind".into(), kind_of(s).into());
            obj.insert("r_formula".into(), r_formula);
            let mut golden: Golden =
                serde_json::from_value(v).unwrap_or_else(|e| panic!("{name}: {e}"));
            golden.source = path;
            golden.csv = Some(format!("{dir}/data/simulated/{name}.csv"));
            golden.weights_col = s["weights_col"].as_str().map(str::to_string);
            golden.weights_suite = true;
            (golden, factors_of(s))
        })
        .collect()
}

/// The `kind` the weights manifest does not record, on the weights-tier references' own
/// branch structure: Gaussian goes to `lm`/`lmer` (which report `sigma` and a
/// plain `se`), anything else to `glm`/`glmer`, split by whether the formula has
/// a random-effect term.
fn kind_of(spec: &Value) -> &'static str {
    let has_re = spec["r_formula"]
        .as_str()
        .expect("spec has r_formula")
        .contains('|');
    match (spec["family"].as_str().expect("spec has a family"), has_re) {
        ("gaussian", _) => "lmm",
        (_, true) => "glmm",
        (_, false) => "glm",
    }
}

/// The shape a golden's `estimates` block takes, which fixes both the field set
/// the reader must cover and the tolerance family the asserts use.
#[derive(PartialEq)]
enum Shape {
    Lmm,
    Glm,
    Glmm,
    /// GLMMadaptive vector-RE AGQ: no `loglik` (deliberately — the two engines
    /// drop different constants), wider bands.
    VectorAgq,
    /// Fixed-only non-Gaussian from the weights tier (rungs 29-43). Same fit as
    /// [`Shape::Glm`], different frozen field set: the weights-tier references branch on
    /// family alone and files every non-Gaussian SE under `se_rx`, so these rungs
    /// carry `se_rx` where the `m3_goldens` GLMs carry `se`, plus an empty
    /// `varcomp`.
    WeightedGlm,
}

fn shape_of(g: &Golden) -> Shape {
    match (g.kind.as_str(), g.engine.as_str()) {
        (_, "GLMMadaptive") => Shape::VectorAgq,
        ("lmm", _) => Shape::Lmm,
        ("glm", _) if g.weights_suite => Shape::WeightedGlm,
        ("glm", _) => Shape::Glm,
        ("glmm", _) => Shape::Glmm,
        (k, _) => panic!("{}: unknown kind {k}", g.name),
    }
}

/// Fields the Tier 2 asserts below actually read and check, per shape. Adding a
/// field to a golden without adding it here — or here without asserting it —
/// fails `golden_fields_are_all_asserted`.
fn asserted_fields(shape: &Shape, g: &Golden) -> Vec<&'static str> {
    let mut f = match shape {
        Shape::Lmm => vec!["beta", "se", "sigma", "loglik", "varcomp"],
        Shape::Glm => vec!["beta", "se", "loglik"],
        Shape::WeightedGlm => vec!["beta", "se_rx", "loglik", "varcomp"],
        // glmmTMB reports no Rx (Schur-complement) standard error.
        Shape::Glmm if g.engine.starts_with("glmmTMB::") => {
            vec!["beta", "se_hessian", "loglik", "varcomp"]
        }
        Shape::Glmm => vec!["beta", "se_hessian", "se_rx", "loglik", "varcomp"],
        Shape::VectorAgq => vec!["beta", "se_hessian", "varcomp"],
    };
    // Conditional on presence, not on family: the two reference trees freeze
    // different amounts for the same family — `goldens_agq.R` writes `theta`
    // and `dispersion`, the weights-tier references carry neither.
    // Listing them unconditionally would fail the "asserted but absent" check on
    // the weights negbin and gamma rungs. The direction that catches the defect
    // this tier exists for — a field the golden carries that nothing reads — is
    // unaffected.
    if g.estimates.theta.is_some() {
        f.push("theta");
    }
    if g.estimates.dispersion.is_some() {
        f.push("dispersion");
    }
    f.retain(|k| !excused(g, k));
    f
}

/// `unasserted_fields` names `field` for `g`.
fn excused(g: &Golden, field: &str) -> bool {
    unasserted_fields(g).iter().any(|(f, _)| *f == field)
}

/// Golden fields Tier 2 deliberately does not assert, each with the reason.
///
/// This list exists so that "not asserted" is a written decision rather than a
/// struct silently dropping a field — the defect this tier was built to fix. An
/// entry here is a claim under review, not a way to quiet a failure.
fn unasserted_fields(g: &Golden) -> Vec<(&'static str, &'static str)> {
    let mut out = unasserted_by_convention(g);
    if objective_differs(g).is_some() {
        out.push((
            "loglik",
            "the reference's logLik is a different function of the parameters \
             (objective_differs), so it is not compared; the deviance is gated on \
             the twin instead, and the exclusion prints as DEV-OBJ.",
        ));
    }
    out
}

/// The per-convention half of [`unasserted_fields`].
fn unasserted_by_convention(g: &Golden) -> Vec<(&'static str, &'static str)> {
    if g.family == "gamma" && g.kind == "glmm" && g.engine.starts_with("lme4::") {
        return vec![
            (
                "sigma",
                "lme4's sigma() on a Gamma glmer is its internal pwrss/n scale \
                 (0.57258 as a variance on sim_gamma_glmm), the plug-in value lme4 \
                 fits at rather than an estimate of φ. glmm estimates φ by maximum \
                 likelihood and has no pwrss/n counterpart; its φ̂ is asserted as \
                 `dispersion` against the glmmTMB twin (`<name>_tmb`), whose \
                 sigma()² is the same ML quantity.",
            ),
            (
                "dispersion",
                "goldens_agq.R's Pearson moment estimator at lme4's fit. glmm's \
                 Fit::dispersion on a Gamma GLMM is the ML φ̂ of the objective it \
                 maximises, a different estimator, so the two are not comparable \
                 at any band; the ML value is asserted against the glmmTMB twin.",
            ),
        ];
    }
    if g.family == "gamma" && g.kind == "glm" && g.engine == "stats::glm+ML-dispersion" {
        return vec![
            (
                "dispersion",
                "the ML φ̂ (MASS::gamma.shape's equation). glmm's Gamma GLM reports \
                 the Pearson moment, asserted against the base `stats::glm` twin.",
            ),
            (
                "se",
                "SEs scaled by the ML φ̂. glmm's are scaled by the Pearson moment, \
                 and asserted against the base `stats::glm` twin.",
            ),
        ];
    }
    Vec::new()
}

// ── Structural gates ─────────────────────────────────────────────────────────

/// Every golden has a manifest entry, and therefore a generator. Three
/// goldens (sleepstudy, penicillin, pastes) sat unregistered and unregenerable
/// until 2026-07-21; this is what stops a fourth appearing.
#[test]
fn all_goldens_are_registered() {
    let registered: std::collections::BTreeSet<String> =
        m3_corpus().into_iter().map(|(g, _)| g.name).collect();
    let dir = format!("{}/validation/goldens", env!("CARGO_MANIFEST_DIR"));
    let mut on_disk = Vec::new();
    for entry in std::fs::read_dir(&dir).expect("goldens dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_some_and(|e| e == "json") {
            on_disk.push(
                path.file_stem()
                    .expect("stem")
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    on_disk.sort();
    let orphans: Vec<&String> = on_disk
        .iter()
        .filter(|n| !registered.contains(*n))
        .collect();
    assert!(
        orphans.is_empty(),
        "goldens with no manifest entry (unregenerable, so not oracles): {orphans:?}"
    );
    assert_eq!(
        on_disk.len(),
        registered.len(),
        "manifest registers goldens that are not on disk"
    );
}

/// The other half of [`is_pending`]: a `pending_reference` must be a real reason,
/// and it must describe a golden that genuinely is not on disk yet.
///
/// The failure this exists to catch is the flag going stale. A spec whose
/// reference HAS been generated but which still carries the field is silently
/// excluded from every assertion in this file — a frozen oracle that costs an R
/// run and gates nothing, the same defect `golden_fields_are_all_asserted` was
/// built for one level down. So the arrival of the JSON turns this test red until
/// the field is removed.
#[test]
fn pending_references_are_absent_and_reasoned() {
    let manifest: Value =
        serde_json::from_str(include_str!("../validation/manifest.json")).expect("manifest parses");
    let pending = pending_specs(
        manifest["m3_goldens"]
            .as_array()
            .expect("manifest has m3_goldens"),
    );
    // Printed, not asserted to be empty: entries here are a legitimate transient
    // state (spec landed, freeze pending). Visible so a `cargo test` run says out
    // loud which goldens the tier is NOT covering.
    if !pending.is_empty() {
        println!("m3_goldens awaiting a reference, excluded from Tier 2: {pending:?}");
    }
}

/// Names of the flagged specs, asserting both halves of the flag's contract on
/// the way past: the reason is non-empty, and the golden really is still absent.
///
/// Split out of the test above so the asserts can be driven by synthetic specs
/// (`flag_semantics` below). With no flagged entry in the real manifest — the
/// normal state — every line here is unreachable from the manifest alone, so the
/// flag's semantics would otherwise be untested code.
fn pending_specs(specs: &[Value]) -> Vec<String> {
    let mut pending = Vec::new();
    for s in specs {
        let name = s["name"].as_str().expect("spec has a name");
        let Some(reason) = s["pending_reference"].as_str() else {
            continue;
        };
        assert!(
            !reason.is_empty(),
            "{name}: pending_reference is present but empty — an exclusion with no reason"
        );
        let path = format!(
            "{}/validation/goldens/{name}.json",
            env!("CARGO_MANIFEST_DIR")
        );
        assert!(
            !std::path::Path::new(&path).exists(),
            "{name}: pending_reference is set but {path} exists — drop the field, \
             or the golden is frozen and gated by nothing"
        );
        pending.push(name.to_string());
    }
    pending
}

/// The flag's own semantics, on synthetic specs rather than the manifest — the
/// manifest carries no flagged entry, and once the references of a release are
/// frozen it never does, so a gate reading the real specs asserts nothing.
mod flag_semantics {
    use super::{is_pending, pending_specs};
    use serde_json::{json, Value};

    /// A spec named after a golden that IS on disk, so the stale-flag assert can
    /// reach its failing branch. `sleepstudy_lmm` rather than an invented name:
    /// an invented one could never collide, which is the case being tested.
    fn frozen_golden_spec(reason: Value) -> Value {
        json!({ "name": "sleepstudy_lmm", "pending_reference": reason })
    }

    #[test]
    fn absent_empty_and_non_empty_reasons() {
        assert!(!is_pending(&json!({ "name": "x" })));
        assert!(!is_pending(
            &json!({ "name": "x", "pending_reference": "" })
        ));
        assert!(is_pending(
            &json!({ "name": "x", "pending_reference": "awaiting the R run" })
        ));
    }

    #[test]
    #[should_panic(expected = "an exclusion with no reason")]
    fn empty_reason_is_rejected() {
        pending_specs(&[json!({ "name": "not_a_golden", "pending_reference": "" })]);
    }

    #[test]
    #[should_panic(expected = "drop the field")]
    fn flag_outliving_the_golden_is_rejected() {
        pending_specs(&[frozen_golden_spec(json!("awaiting the R run"))]);
    }

    #[test]
    fn an_unfrozen_flagged_spec_is_reported() {
        let pending = pending_specs(&[
            json!({ "name": "plain_spec" }),
            json!({ "name": "not_a_golden", "pending_reference": "awaiting the R run" }),
        ]);
        assert_eq!(pending, vec!["not_a_golden".to_string()]);
    }
}

/// The defect this test guards against: a reader struct silently dropping a
/// golden field, so the value is frozen, costs an R run, and is never checked.
/// It walks every key under `estimates` in the golden JSON and requires each
/// one to be either asserted (`asserted_fields`) or explicitly excused
/// (`unasserted_fields`).
///
/// Deliberately not `deny_unknown_fields`: every golden carries top-level
/// metadata no assertion reads, and refusing unknown fields is a different
/// claim from "the reader covers what the golden carries".
#[test]
fn golden_fields_are_all_asserted() {
    let mut missing = Vec::new();
    for (g, _) in corpus() {
        let raw = std::fs::read_to_string(&g.source).expect("golden");
        let v: Value = serde_json::from_str(&raw).expect("golden parses");
        let shape = shape_of(&g);
        let covered = asserted_fields(&shape, &g);
        let excused = unasserted_fields(&g);
        for key in v["estimates"].as_object().expect("estimates object").keys() {
            let k = key.as_str();
            if !covered.contains(&k) && !excused.iter().any(|(f, _)| *f == k) {
                missing.push(format!("{}.{key}", g.name));
            }
        }
        for (field, reason) in &excused {
            assert!(
                v["estimates"].get(field).is_some(),
                "{}: `{field}` is excused but the golden has no such field",
                g.name
            );
            assert!(
                !reason.is_empty(),
                "{}: `{field}` is excused with no reason",
                g.name
            );
        }
        // The other direction: a field this table claims is asserted must
        // actually be present, or the claim is decoration.
        for key in &covered {
            assert!(
                v["estimates"].get(key).is_some(),
                "{}: asserted_fields lists `{key}`, but the golden has no such field",
                g.name
            );
        }
    }
    assert!(
        missing.is_empty(),
        "golden fields that no Tier 2 assertion reads: {missing:#?}"
    );
}

// ── The cross-engine assertions ──────────────────────────────────────────────

/// Refit every golden from its own recorded `r_formula` and compare every field
/// it carries, at `grid/tol.R`'s bands.
///
/// One test over the corpus rather than 56 hand-written ones: the goldens differ
/// in data and model, not in what agreement means, and a per-golden test would
/// be 61 copies of the same twelve asserts. A failure names the golden and the
/// field, so it localises the same way.
#[test]
fn goldens_agree_with_the_references() {
    // 46 `m3_goldens` + 15 prior-weights rungs. Pinned because a loader that
    // silently returns fewer — a renamed results directory, a manifest key that
    // moved — would leave this test green while asserting nothing, which is the
    // failure mode the whole tier is built against.
    //
    // 53 -> 55 on 2026-07-30: the two `sim_binomial_bigsd_agq_k{7,11}` references
    // were frozen, so their specs lost `pending_reference` and rejoined
    // `m3_corpus()`. 55 -> 56 on 2026-08-01: `sim_dynrange_lmm` joined: this
    // design now fits instead of NaN-filling on conditioning grounds, which is
    // what made a reference possible at all. 56 -> 61 on 2026-08-06: the five
    // `sim_scale_*_glm` goldens joined, pinning that the GLM divergence guard's
    // switch from bounding |β| to bounding |η| gives the same accept/reject
    // decision `stats::glm` does on the three-unit-system, separated, and
    // Gamma-inverse-large-η design classes. 61 -> 62 on 2026-08-26:
    // `sim_cloglog_glm` joined (the cloglog GLM arm). 62 -> 63 on 2026-08-26:
    // `sim_cloglog_glmm` joined (the cloglog GLMM arm — no kernel change, PIRLS
    // reaches the link through `family_pass`). 63 -> 65 on 2026-08-26:
    // `sim_igauss_glm` and `sim_igauss_inv_sq_glm` joined (the two inverse-Gaussian
    // GLM link cells, log and 1/μ², on the new `sim_igauss` fixture — GLM-only,
    // no glmm cell, since the family faults at the model-shape gate with random
    // effects). 65 -> 73 on 2026-09-24: the eight glmmTMB `<name>_tmb` twins of
    // the lme4 GLMM goldens whose objective is not glmm's (`objective_differs`)
    // joined, as the exact-Laplace references those fits are gated against. The
    // 58 is what `all_goldens_are_registered` proves against the goldens
    // directory; the 15 is asserted inside `corpus()` itself. 73 -> 76 on
    // 2026-09-24: the three `_ml` twins of the `stats::glm` Gamma GLM goldens
    // joined when glmm's Gamma GLM moved to the ML dispersion, and the 58 became
    // 61.
    assert_eq!(corpus().len(), 76, "the cross-engine corpus changed size");
    for (g, factors) in corpus() {
        let shape = shape_of(&g);
        let factor_refs: Vec<&str> = factors.iter().map(String::as_str).collect();
        let (f, cols, groups) = refit(&g, &factor_refs);
        let name = &g.name;
        let align = align_coefs(&cols, &g.coef_names, f.aliased(), name);

        assert_eq!(
            f.converged(),
            g.converged,
            "{name}: convergence flag disagrees with the oracle"
        );
        assert_eq!(
            f.singular(),
            g.singular,
            "{name}: singularity flag disagrees with the oracle"
        );
        if !g.converged {
            continue;
        }

        let (beta_band, se_band, sd_band) = match shape {
            Shape::VectorAgq => (
                tol::AGQ_BETA_REL,
                tol::AGQ_SE_HESSIAN_REL,
                tol::AGQ_STDDEV_REL,
            ),
            _ => (tol::BETA_REL, tol::SE_REL, tol::STDDEV_REL),
        };

        assert_coefs(
            &f.beta,
            f.aliased(),
            &align,
            &g.estimates.beta,
            beta_band,
            &format!("{name}: beta"),
        );

        // SE — `se` on LMM/GLM, `se_hessian` on GLMM (glmm's default WaldSe
        // matches lme4's `use.hessian=TRUE`), and `se_rx` where the golden
        // froze the Schur-complement method too.
        if let Some(se) = g.estimates.se.as_ref().filter(|_| !excused(&g, "se")) {
            assert_coefs(
                &f.se,
                f.aliased(),
                &align,
                se,
                se_band,
                &format!("{name}: se"),
            );
        }
        if let Some(se) = &g.estimates.se_hessian {
            let band = if shape == Shape::VectorAgq {
                tol::AGQ_SE_HESSIAN_REL
            } else {
                tol::SE_HESSIAN_REL
            };
            assert_coefs(
                &f.se,
                f.aliased(),
                &align,
                se,
                band,
                &format!("{name}: se_hessian"),
            );
        }

        // `se_rx` is lme4's other SE method (Schur complement conditional on
        // θ̂). It needs its own fit under the matching glmm setting — comparing
        // it against the Hessian SE would be comparing two estimators.
        if let Some(se_rx) = g.estimates.se_rx.as_ref().filter(|_| !excused(&g, "se_rx")) {
            let (fx, _, _) = refit_with(&g, &factor_refs, WaldSe::Rx);
            assert_coefs(
                &fx.se,
                fx.aliased(),
                &align,
                se_rx,
                tol::SE_REL,
                &format!("{name}: se_rx"),
            );
        }

        // σ̂ — the LMM residual scale, where `Fit::dispersion` is σ̂². On a Gamma
        // GLMM the golden's `sigma` is a different quantity entirely; see
        // `unasserted_fields`.
        if let (Some(sigma), Shape::Lmm) = (g.estimates.sigma, &shape) {
            assert_rel(
                f.dispersion.sqrt(),
                sigma,
                tol::STDDEV_REL,
                &format!("{name}: sigma"),
            );
        }
        if let Some(theta) = g.estimates.theta {
            assert_rel(
                f.dispersion,
                theta,
                tol::BETA_REL,
                &format!("{name}: theta"),
            );
        }
        if let Some(phi) = g
            .estimates
            .dispersion
            .filter(|_| !excused(&g, "dispersion"))
        {
            assert_rel(
                f.dispersion,
                phi,
                tol::BETA_REL,
                &format!("{name}: dispersion"),
            );
        }
        // Deviance gate: dev = -2*loglik on each side. lme4's nAGQ>1
        // logLik is on a different scale than nAGQ=1 — it drops the
        // saturated-model logLik that nAGQ=1 restores (measured on cbpp: raw
        // Δdev 84.04) — and `aligned_dev` adds that deficit back before the
        // comparison, so what survives on both sides of the nAGQ break is a
        // fraction of a unit rather than tens of units (cbpp_agq_k7: corrected
        // dev 183.9667 vs the nAGQ=1 reference 184.0526, Δdev 0.086). That
        // residual clears the gate because the gate is one-sided — glmm landing
        // BELOW the reference is free, and DEV_BIG is the two-sided guard the
        // 0.086 is measured against. It is 430x DEV_EPS, which only bounds how
        // much WORSE than the reference glmm may be. A golden with no `loglik` at
        // all (the six GLMMadaptive vector-RE-AGQ rungs, `Shape::VectorAgq`,
        // which never report one) has no deviance to gate — printed loudly
        // rather than silently skipped, and `dev_align_none_matches_the_six_
        // vector_agq_goldens` below pins that this set of six can never grow
        // quietly. A golden whose engine maximises a different objective
        // (`objective_differs`) is the other loud exclusion, pinned the same way
        // by `objective_differs_matches_the_pinned_goldens`.
        match (
            objective_differs(&g),
            oracle_support::dev_align::aligned_dev(&g),
        ) {
            (Some(why), _) => {
                eprintln!("DEV-OBJ {name}: {why}")
            }
            (None, None) => {
                eprintln!("DEV-NA {name}: golden carries no loglik — parameter sanity only")
            }
            (None, Some(dev_ref)) => {
                let dev_g = -2.0 * f.loglik;
                let d = dev_g - dev_ref;
                assert!(
                    d.abs() <= oracle_support::DEV_BIG,
                    "{name}: |Δdev|={d:.3e} > DEV_BIG — suspected convention mismatch"
                );
                assert!(
                    d <= oracle_support::DEV_EPS,
                    "{name}: Δdev={d:.3e} > DEV_EPS — worse optimum than reference"
                );
            }
        }

        // Variance components, paired by group NAME — glmm emits declaration
        // order, lme4 descending level count.
        if let Some(varcomp) = &g.estimates.varcomp {
            assert_eq!(groups.len(), varcomp.len(), "{name}: grouping count");
            for (k, gname) in groups.iter().enumerate() {
                let block = g.estimates.block(&oracle_name(gname));
                let (sds, corr) = f.stddev_corr(k);
                assert_eq!(sds.len(), block.stddev.len(), "{name}/{gname}: block width");
                assert_eq!(
                    block.terms.len(),
                    sds.len(),
                    "{name}/{gname}: the oracle names {} RE terms for a {}-wide block",
                    block.terms.len(),
                    sds.len()
                );
                for (t, (&got, &want)) in sds.iter().zip(&block.stddev).enumerate() {
                    assert_rel(got, want, sd_band, &format!("{name}: {gname} stddev[{t}]"));
                }
                if let Some(want_corr) = &block.corr {
                    for a in 0..sds.len() {
                        for b in (a + 1)..sds.len() {
                            assert_abs(
                                corr[a][b],
                                want_corr[a][b],
                                tol::AGQ_CORR_ABS,
                                &format!("{name}: {gname} corr[{a}][{b}]"),
                            );
                        }
                    }
                }
            }
        }
    }
    assert_documented_divergences_all_fired();
}

/// The registry's teeth. This tier reports a documented divergence instead of
/// failing on it (`validation/grid/divergences.json`), which only stays honest if an
/// entry cannot outlive the divergence it describes: an entry whose fixture the
/// corpus above actually refit and which never matched is a standing exemption
/// for whatever drifts there next, so it fails here.
///
/// Printed, never silent. The print reaches the
/// operator on any failure in this test and under `--nocapture`; the assertion
/// below is what makes the match a checked fact rather than a quiet pass.
fn assert_documented_divergences_all_fired() {
    use oracle_support::divergence;
    let reg = divergence::registry();
    let fired = reg.fired();
    // Keyed on the golden's `name`, not its `data`: `Registry::covers` parses the
    // fixture out of the `"{name}: {quantity}"` assertion context, so an oracle-tier
    // entry's `cell` field holds a golden name. (`validation/grid/compare.R` reads that
    // same field as a grid cell id — an entry scoped to both consumers only works
    // where the two names coincide.)
    let in_corpus: std::collections::BTreeSet<String> =
        corpus().into_iter().map(|(g, _)| g.name).collect();

    let mut expected = std::collections::BTreeSet::new();
    for e in reg.scoped() {
        if !in_corpus.contains(&e.cell) {
            continue; // this tier has no golden for it; grid/compare.R owns that entry
        }
        expected.insert(e.id.clone());
        if fired.contains(&e.id) {
            eprintln!(
                "documented divergence: {} [{}] <= {:.1e}\n  {}\n  direction: {}",
                e.cell,
                e.quantities.join(","),
                e.max_rel,
                e.summary,
                e.direction
            );
        }
    }
    assert_eq!(
        fired, expected,
        "documented-divergence registry is out of date: entries scoped to this \
         tier that no longer fire must be deleted, and a divergence that starts \
         firing must be written up first"
    );
}

/// glmm names a nested grouping outer:inner (`batch:cask`); lme4 names it
/// inner:outer (`cask:batch`). Cosmetic, but it decides which golden block a
/// name-keyed read finds, so it is translated in exactly one place.
fn oracle_name(glmm_name: &str) -> String {
    match glmm_name.split_once(':') {
        Some((outer, inner)) => format!("{inner}:{outer}"),
        None => glmm_name.to_string(),
    }
}
