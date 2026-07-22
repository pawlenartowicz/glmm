//! Tier 2 — the cross-engine tier (RULE 6).
//!
//! Asserts the crate against the frozen lme4 / MASS / GLMMadaptive references at
//! `validation/tol.R`'s agreement bands: 38 goldens in `validation/goldens/` plus the
//! weights tier (rungs 29-43) in `validation/results/lme4_simulated/`. Reads frozen
//! JSON, so it needs neither R nor Julia at runtime — but it refits all 53, so
//! it is off by default:
//!
//! ```sh
//! cargo test --features oracle-tests
//! ```
//!
//! A failure here means glmm and the reference no longer agree by more than the
//! calibration allowed. There are exactly three honest endings: a bug in glmm
//! (fix it), a documented expected divergence (add the entry, under review), or
//! a reference-vs-reference gap (`reference_disagreements.md`). Widening a band
//! to make red go green is none of them.
//!
//! This is NOT `validation/run.sh`, which refits R and Julia to check the frozen
//! values still reflect what the references say today. Different claims, both
//! kept; `run.sh --rust-tier2` runs this then that.
#![cfg(all(feature = "oracle-tests", feature = "formula"))]

mod oracle_support;

use glmm::WaldSe;
use oracle_support::{
    align_coefs, assert_abs, assert_coefs, assert_rel, refit, refit_with, tol, Golden,
};
use serde_json::Value;

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

/// The `validation/goldens/` tree, with the manifest's factor list. The manifest is
/// the registry — RULE 0: a golden that no script can regenerate is not an
/// oracle, so anything in `validation/goldens/` must appear in `m3_goldens`, and
/// `all_goldens_are_registered` holds that line.
fn m3_corpus() -> Vec<(Golden, Vec<String>)> {
    let manifest: Value =
        serde_json::from_str(include_str!("../validation/manifest.json")).expect("manifest parses");
    let specs = manifest["m3_goldens"]
        .as_array()
        .expect("manifest has m3_goldens");
    specs
        .iter()
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
/// D3 asked for these to be "promoted into `m3_goldens`", meaning: asserted from
/// `cargo test` instead of only by `compare.R` on a machine with R. Reading them
/// in place achieves that without a second copy of 15 frozen JSONs — and §15.1
/// is the case for not making one. The `goldens/` and `results/` trees held
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

/// RULE 0: every golden has a manifest entry, and therefore a generator. Three
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
/// One test over the corpus rather than 35 hand-written ones: the goldens differ
/// in data and model, not in what agreement means, and a per-golden test would
/// be 35 copies of the same twelve asserts. A failure names the golden and the
/// field, so it localises the same way.
#[test]
fn goldens_agree_with_the_references() {
    // 35 `m3_goldens` + 15 prior-weights rungs. Pinned because a loader that
    // silently returns fewer — a renamed results directory, a manifest key that
    // moved — would leave this test green while asserting nothing, which is the
    // failure mode the whole tier is built against.
    assert_eq!(corpus().len(), 53, "the cross-engine corpus changed size");
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
        let align = align_coefs(&cols, &g.coef_names, &f.aliased, name);

        assert_eq!(
            f.converged, g.converged,
            "{name}: convergence flag disagrees with the oracle"
        );
        assert_eq!(
            f.singular, g.singular,
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
            &f.aliased,
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
                &f.aliased,
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
                &f.aliased,
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
                &fx.aliased,
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
        if let Some(ll) = g.estimates.loglik {
            if shape != Shape::VectorAgq && g.nagq > 1 {
                // EXPECTED DIVERGENCE — lme4's logLik() is not on the same scale
                // at nAGQ>1 as at nAGQ=1, and glmm's is. Measured on cbpp at
                // tolPwrss=1e-13, where the DEVIANCE barely moves (73.4715 at
                // nAGQ=1, 73.3730 at 7, 73.3731 at 11 — the expected small AGQ
                // improvement) while lme4's logLik jumps -92.0263 → -50.0050 →
                // -50.0050. The constant it restores at nAGQ>1 is not the one it
                // restores at nAGQ=1, and it is not the aggregated-binomial
                // normalising term either (Σ ln C(nᵢ,sᵢ) = 185.48, the jump is
                // 42.02). grouseticks shows the same break, -957.3996 → -492.3276.
                // glmm reports -91.9834 at nAGQ=7 — consistent with its own
                // Laplace value, which is the behaviour a user can compare across
                // nAGQ. Asserted as a divergence so that lme4 fixing this fails
                // here rather than passing silently.
                assert!(
                    (f.loglik - ll).abs() > 1.0,
                    "{name}: lme4's nAGQ>1 logLik scale break has disappeared \
                     (glmm={}, oracle={ll}) — re-check the divergence and gate \
                     this at LOGLIK_ABS_GLMM instead",
                    f.loglik
                );
            } else {
                let band = if g.kind == "lmm" {
                    tol::LOGLIK_ABS_LMM
                } else {
                    tol::LOGLIK_ABS_GLMM
                };
                assert_abs(f.loglik, ll, band, &format!("{name}: loglik"));
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
