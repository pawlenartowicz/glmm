//! Tier 2 — the cross-engine tier.
//!
//! Asserts the crate against the frozen lme4 / MASS / GLMMadaptive references at
//! `validation/tol.R`'s agreement bands: 50 goldens in `validation/goldens/` plus the
//! weights tier (rungs 29-43) in `validation/results/lme4_simulated/`. Reads frozen
//! JSON, so it needs neither R nor Julia at runtime — but it refits all 65, so
//! it is off by default:
//!
//! ```sh
//! cargo test --features oracle-tests
//! ```
//!
//! A failure here means glmm and the reference no longer agree by more than the
//! calibration allowed. There are exactly three honest endings: a bug in glmm
//! (fix it), a documented expected divergence (add the entry, under review), or
//! a reference-vs-reference gap. Widening a band
//! to make red go green is none of them.
//!
//! This is NOT `validation/run.sh`, which refits R and Julia to check the frozen
//! values still reflect what the references say today. Different claims, both
//! kept; `run.sh --rust-tier2` runs this then that.
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

#[test]
fn aligned_dev_none_when_golden_lacks_loglik() {
    let g = load_golden("sim_binomial_slope1_agq_k7");
    assert_eq!(oracle_support::dev_align::aligned_dev(&g), None);
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
/// spec has to land FIRST — `engines/goldens_agq.R` reads the manifest to know
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
/// `cargo test` instead of only by `compare.R` on a machine with R. Reading them
/// in place achieves that without a second copy of 15 frozen JSONs. The `goldens/`
/// and `results/` trees held
/// overlapping values that had silently drifted onto different `tolPwrss`
/// settings, and the fix was to put them back on the same footing. A third
/// overlapping tree would rebuild exactly that hazard.
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

/// The `kind` the weights manifest does not record, on `engines/lme4.R`'s own
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
    /// [`Shape::Glm`], different frozen field set: `engines/lme4.R` branches on
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
        Shape::Glmm => vec!["beta", "se_hessian", "se_rx", "loglik", "varcomp"],
        Shape::VectorAgq => vec!["beta", "se_hessian", "varcomp"],
    };
    // Conditional on presence, not on family: the two reference trees freeze
    // different amounts for the same family — `goldens_agq.R` writes `theta`
    // and `dispersion`, `engines/lme4.R`'s weights-tier branch writes neither.
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
    f
}

/// Golden fields Tier 2 deliberately does not assert, each with the reason.
///
/// This list exists so that "not asserted" is a written decision rather than a
/// struct silently dropping a field — the defect this tier was built to fix. An
/// entry here is a claim under review, not a way to quiet a failure.
fn unasserted_fields(g: &Golden) -> Vec<(&'static str, &'static str)> {
    if g.family == "gamma" && g.kind == "glmm" {
        return vec![(
            "sigma",
            "lme4's sigma() on a Gamma glmer is its internal pwrss/n scale \
             (0.57258 as a variance on sim_gamma_glmm), which matches neither the \
             Pearson moment estimator nor deviance/df.residual. goldens_agq.R \
             freezes both it and the Pearson `dispersion` so the in-crate test can \
             pick; the crate picked Pearson and reports it as Fit::dispersion, \
             which `dispersion` above asserts. glmm has no reported counterpart \
             for this second scale. Reviewed and adopted as a deliberate \
             divergence 2026-07-21 — see the Gamma GLMM dispersion entry under \
             'differences that change the answer' in the crate's lme4 comparison \
             notes, which records why the second scale is not exposed on `Fit`. \
             Unasserted is not unverified: glmm computes the same pwrss/n \
             (`family::glmm_sigma_sq`) and this tier gates it through two derived \
             fields it does assert — `se_rx`, which carries σ̂² as its scale factor \
             for Gamma, and `varcomp`, which is θ̂²·σ̂² on lme4's VarCorr \
             convention. Reading `sigma` here would be a third check on the same \
             number.",
        )];
    }
    Vec::new()
}

/// Goldens Tier 2 cannot gate yet, each with the open decision it waits on.
///
/// Not a way to quiet a failure: an entry means the gap is understood, recorded
/// elsewhere, and blocked on a decision that is not this tier's to make. The
/// count is asserted below so entries cannot accumulate quietly.
fn known_open(_g: &Golden) -> Option<&'static str> {
    // Empty since 2026-07-21: the last entry (sim_sparse_gamma) closed when the
    // formula frontend switched to lowering random effects in formula order,
    // which routes that model sparse — see `//gamma_rungs` in validation/manifest.json.
    None
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

/// The defect this tier was built to fix: a reader struct silently dropping a
/// golden field, so the value is frozen, costs an R run, and is never checked.
/// `Est` used to drop `sigma`, `se_rx`, `loglik`, `dispersion` and `theta`.
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
/// it carries, at `tol.R`'s bands.
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
    // `m3_corpus()`. 55 -> 56 on 2026-08-01: `sim_dynrange_lmm` joined, a design
    // the crate used to NaN-fill on conditioning grounds and now fits — which is
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
    // effects). The 50 is what `all_goldens_are_registered` proves against the
    // goldens directory; the 15 is asserted inside `corpus()` itself.
    assert_eq!(corpus().len(), 65, "the cross-engine corpus changed size");
    let mut open = Vec::new();
    for (g, factors) in corpus() {
        if let Some(reason) = known_open(&g) {
            assert!(!reason.is_empty());
            open.push(g.name.clone());
            continue;
        }
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
        if let Some(se) = &g.estimates.se {
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
        if let Some(se_rx) = &g.estimates.se_rx {
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
        if let Some(phi) = g.estimates.dispersion {
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
        // quietly.
        match oracle_support::dev_align::aligned_dev(&g) {
            None => eprintln!("DEV-NA {name}: golden carries no loglik — parameter sanity only"),
            Some(dev_ref) => {
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
    assert_open_set_unchanged(&open);
    assert_documented_divergences_all_fired();
}

/// The registry's teeth. This tier reports a documented divergence instead of
/// failing on it (`validation/divergences.json`), which only stays honest if an
/// entry cannot outlive the divergence it describes: an entry whose dataset the
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
    // dataset out of the `"{name}: {quantity}"` assertion context, so an oracle-tier
    // entry's `dataset` is a golden name. (`validation/compare.R` reads the same field
    // as a manifest dataset name — an entry scoped to both tiers only works where the
    // two coincide.)
    let in_corpus: std::collections::BTreeSet<String> =
        corpus().into_iter().map(|(g, _)| g.name).collect();

    let mut expected = std::collections::BTreeSet::new();
    for e in reg.scoped() {
        if !in_corpus.contains(&e.dataset) {
            continue; // this tier has no golden for it; compare.R owns that entry
        }
        expected.insert(e.id.clone());
        if fired.contains(&e.id) {
            eprintln!(
                "documented divergence: {} rung {} [{}] <= {:.1e}\n  {}\n  direction: {}\n  see: {}",
                e.dataset,
                e.rung,
                e.quantities.join(","),
                e.max_rel,
                e.summary,
                e.direction,
                e.review
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

/// A count, not a list, so adding a `known_open` entry is a deliberate act that
/// has to be made here too — the failure mode this whole tier exists to stop is
/// a case quietly leaving coverage.
fn assert_open_set_unchanged(open: &[String]) {
    assert_eq!(
        open,
        Vec::<String>::new(),
        "the set of goldens Tier 2 cannot gate has changed"
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

/// Calibration measurement for `KKT_INTERIOR_MAX` (`src/test_support.rs`) —
/// prints, for every GLMM rung the tier loads plus the committed
/// `tests/fixtures/glmm_hessian_vcov.json` fixture: name, boundary, deviance
/// and `kkt_grad_norm`. It asserts nothing; the constant is pinned BY HAND
/// from its output (ceil-to-one-significant-figure of ten times the worst
/// finite value — the margin convention `validation/tol.R` uses for its own
/// measured bands). NaN rows are the shapes with no exact gradient
/// (structured extras, dense fallback, sparse) and are not part of the
/// calibration. Run:
///
/// ```sh
/// cargo test --features oracle-tests kkt_calibration -- --ignored --nocapture
/// ```
#[test]
#[ignore]
fn kkt_calibration_measurement() {
    for (g, factors) in corpus() {
        if !matches!(shape_of(&g), Shape::Glmm | Shape::VectorAgq) {
            continue;
        }
        let factor_refs: Vec<&str> = factors.iter().map(String::as_str).collect();
        let (f, _cols, _groups) = refit(&g, &factor_refs);
        println!(
            "{}\tboundary={:?}\tdeviance={}\tkkt={:e}",
            g.name, f.diagnostics.boundary, f.deviance, f.diagnostics.kkt_grad_norm
        );
    }
    // The committed n=96 / 12-cluster `y ~ x1 + (1|grp)` binomial fixture,
    // through the same public `fit_cold` route.
    let s = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/glmm_hessian_vcov.json"
    ))
    .expect("read hessian fixture");
    let v: Value = serde_json::from_str(&s).expect("parse hessian fixture");
    let n = v["n"].as_u64().unwrap() as usize;
    let x_rows: Vec<Vec<f64>> = v["x"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r.as_array()
                .unwrap()
                .iter()
                .map(|e| e.as_f64().unwrap())
                .collect()
        })
        .collect();
    let p = x_rows[0].len();
    let x: Vec<f64> = x_rows.iter().flat_map(|r| r.iter().copied()).collect();
    let y: Vec<f64> = v["y"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    let ids: Vec<u32> = v["cluster_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_u64().unwrap() as u32)
        .collect();
    let n_clusters = ids.iter().max().unwrap() + 1;
    let model = glmm::ModelSpec {
        family: glmm::Family::Binomial {
            link: glmm::BinomialLink::Logit,
        },
        re: Some(glmm::ReStructure {
            sizing: glmm::Sizing::FixedClusters { n_clusters },
            slopes: vec![],
            extra_groupings: vec![],
        }),
    };
    let opts = glmm::FitOptions {
        target_indices: (0..p as u32).collect(),
        ..glmm::FitOptions::default()
    };
    let f = glmm::fit_cold(
        &x,
        &y,
        n,
        p,
        &model,
        &glmm::GroupIds {
            primary: ids,
            extra: vec![],
        },
        &opts,
    );
    println!(
        "glmm_hessian_vcov_fixture\tboundary={:?}\tdeviance={}\tkkt={:e}",
        f.diagnostics.boundary, f.deviance, f.diagnostics.kkt_grad_norm
    );
}
