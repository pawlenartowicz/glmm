//! Tier 2 support: golden schema, `validation/tol.R` mirror, and the generic driver
//! that refits a golden from its own recorded `r_formula`.
//!
//! Tier 2 is the cross-engine tier — it checks the crate against the frozen
//! lme4 / MASS / GLMMadaptive results in `validation/goldens/`. The bands here are
//! AGREEMENT bands (measured worst gap plus margin), not correctness
//! thresholds; a difference beyond one means two implementations drifted apart by
//! more than the calibration allowed, which is a flag to investigate. Tier 1's
//! tight pins live beside the code in `src/**/*_tests.rs`.
//!
//! This is a REFERENCE CHECK, not a pass/fail gate: a difference past its band
//! passes when `validation/divergences.json` documents it, and fails otherwise —
//! see the [`divergence`] module for the registry and the rules that keep it from
//! decaying into a blanket exemption.

use glmm::formula::{lower, Column, Table};
use glmm::{
    fit_cold, BinomialLink, Family, Fit, GammaLink, InverseGaussianLink, NegBinomialLink,
    PoissonLink, WaldSe,
};
use serde::Deserialize;
use serde_json::Value;

// ── Tolerances — mirrors `validation/tol.R`, change together ─────────────────────
//
// `tol.R` is the single source of truth; these are its numbers, and the
// reciprocal note lives there. Each band's calibration argument is recorded in
// `tol.R` next to the value — not repeated here, so the two cannot drift into
// disagreeing rationales. Tier 1 bands are per dispatch path, live beside their
// pins, and are deliberately NOT here: they are not agreement bands.
pub mod tol {
    pub const BETA_REL: f64 = 1e-3;
    pub const STDDEV_REL: f64 = 1e-3;
    pub const SE_REL: f64 = 1e-3;
    /// The retired 3e-2 band is gone: it existed only while the frozen oracle
    /// carried lme4's lagged-`ldL2` `tolPwrss` artifact, which was fixed and the
    /// band tightened to 1e-3 on 2026-07-04. `tests/formula_fit.rs` never
    /// followed; Tier 2 does.
    pub const SE_HESSIAN_REL: f64 = 1e-3;
    // `tol.R`'s stddev_se_rel = 3e-3 is deliberately NOT mirrored: no golden in
    // `validation/goldens/` carries a `stddev_se` field. lme4's SE-of-RE-stddev is
    // frozen only in the curated `validation/results/` tree, which `compare.R` gates
    // and this tier does not read. Mirroring an unused constant would imply a
    // claim Tier 2 does not make.

    // Vector-RE AGQ rungs vs GLMMadaptive.
    pub const AGQ_BETA_REL: f64 = 3e-3;
    pub const AGQ_STDDEV_REL: f64 = 4e-3;
    /// ABSOLUTE — correlations near zero break a relative band. Reused for the
    /// non-AGQ LMM correlation blocks too: `tol.R` has no separate non-AGQ
    /// correlation constant, the estimate-grid `analyze.R` already reuses this one, and
    /// only three goldens carry a real off-diagonal — too thin a base to
    /// calibrate a second constant on.
    pub const AGQ_CORR_ABS: f64 = 4e-3;
    pub const AGQ_SE_HESSIAN_REL: f64 = 2e-2;
}

// ── Golden schema ────────────────────────────────────────────────────────────
//
// One struct covering all eight shapes in the corpus; absent fields land as
// `None`. `deny_unknown_fields` is deliberately NOT used — every golden carries
// top-level metadata no assertion reads, and rejecting unknown fields is a
// different claim from "the reader covers what the golden carries". That claim
// is made by `golden_fields_are_all_asserted` instead.

#[derive(Deserialize)]
pub struct Golden {
    pub name: String,
    pub engine: String,
    pub kind: String,
    pub data: String,
    /// Absolute path of the JSON this was read from, so the field-coverage test
    /// can re-open it. Set by the loader, never deserialized.
    #[serde(skip)]
    pub source: String,
    /// Absolute path of the dataset CSV. `None` means the `sim_`-prefix directory
    /// convention `csv_for` implements; the weights tier's names don't carry that
    /// prefix, so its loader sets this explicitly rather than teaching `csv_for` a
    /// second convention.
    #[serde(skip)]
    pub csv: Option<String>,
    /// Column holding per-row prior weights. Comes from the manifest entry's
    /// `weights_col`, not from the frozen result — the reference records what it
    /// estimated, not how the harness fed it.
    #[serde(skip)]
    pub weights_col: Option<String>,
    /// From the weights tier (rungs 29-43), whose `lme4.R` writes a different field
    /// set for fixed-only non-Gaussian fits than `goldens_agq.R` does. Only
    /// `shape_of` reads it.
    #[serde(skip)]
    pub weights_suite: bool,
    pub family: String,
    /// `jsonlite` writes R's `NULL` link (the Gaussian LMM rungs, where `lmer`
    /// takes no family object) as `{}`, not as a string — so this stays a raw
    /// `Value` and `family_of` narrows it.
    #[serde(default)]
    pub link: Value,
    #[serde(default = "one")]
    pub nagq: u8,
    pub r_formula: String,
    pub converged: bool,
    pub singular: bool,
    pub coef_names: Vec<String>,
    pub estimates: Est,
}

fn one() -> u8 {
    1
}

#[derive(Deserialize)]
pub struct Est {
    /// `None` at a coefficient the reference DROPPED as aliased — R writes `NA`,
    /// which `jsonlite` emits as `null`. glmm instead reports NaN and flags
    /// `aliased[j]`, a deliberate divergence, so these slots are asserted as
    /// divergences rather than parsed away.
    pub beta: Vec<Option<f64>>,
    /// LMM / plain-GLM standard errors.
    #[serde(default)]
    pub se: Option<Vec<Option<f64>>>,
    /// GLMM SE keeping the θ–β coupling (glmer `use.hessian=TRUE`) — the method
    /// glmm's default `WaldSe::Hessian` matches.
    #[serde(default)]
    pub se_hessian: Option<Vec<Option<f64>>>,
    /// GLMM SE conditional on θ̂ (Schur complement, glmer `use.hessian=FALSE`).
    #[serde(default)]
    pub se_rx: Option<Vec<Option<f64>>>,
    #[serde(default)]
    pub sigma: Option<f64>,
    #[serde(default)]
    pub loglik: Option<f64>,
    #[serde(default)]
    pub dispersion: Option<f64>,
    #[serde(default)]
    pub theta: Option<f64>,
    #[serde(default)]
    pub varcomp: Option<Vec<Vc>>,
}

#[derive(Deserialize)]
pub struct Vc {
    pub group: String,
    pub terms: Vec<String>,
    pub stddev: Vec<f64>,
    #[serde(default)]
    pub corr: Option<Vec<Vec<f64>>>,
}

impl Est {
    /// The golden's variance block for grouping `name`.
    ///
    /// Never index `varcomp` positionally across more than one block: glmm emits
    /// declaration order, lme4's `VarCorr` descending level count, and
    /// `pastes_lmm` is the fixture where the two disagree.
    pub fn block(&self, name: &str) -> &Vc {
        self.varcomp
            .as_ref()
            .expect("golden has no varcomp")
            .iter()
            .find(|b| b.group == name)
            .unwrap_or_else(|| panic!("golden has no varcomp block named {name}"))
    }
}

/// Read and parse a single golden by name, setting `source` the same way
/// `m3_corpus`/`weights_corpus` do. For tests that need one named golden rather
/// than the whole corpus (e.g. `dev_align`'s self-checks) — the multi-golden
/// readers stay in `validation_oracle.rs` since they also carry manifest/factor
/// logic this single-file read has no use for.
pub fn load_golden(name: &str) -> Golden {
    let path = format!(
        "{}/validation/goldens/{name}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut golden: Golden = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{name}: {e}"));
    golden.source = path;
    golden
}

// ── Comparison helpers ───────────────────────────────────────────────────────

/// Coordinates both engines put at zero, below which a relative difference has
/// no answer to give: glmm pins to a hard 0.0 while an oracle stops on its own
/// residue, and `|x-y| / max(|x|,|y|)` reads exactly 1.0 however small that
/// residue gets. Mirrors `TOL$near_zero_abs` in `validation/tol.R`, which
/// carries the measurement it was sized from — change together.
const NEAR_ZERO_ABS: f64 = 1e-3;

/// Relative difference against the larger magnitude, matching `tol.R::rel_max`
/// (`|x-y| / max(|x|,|y|,1e-12)`, exempting coordinates under `NEAR_ZERO_ABS`)
/// so a Rust failure and an R failure mean the same thing on the same pair.
fn rel(got: f64, want: f64) -> f64 {
    let scale = got.abs().max(want.abs()).max(1e-12);
    if scale <= NEAR_ZERO_ABS {
        return 0.0;
    }
    (got - want).abs() / scale
}

/// The one place an over-band cross-engine comparison is adjudicated.
///
/// Implements the reference-check rule stated in the module header — not
/// restated here, so the two cannot drift. What is specific to this function:
/// a documented match is recorded so the corpus driver can print it and prove
/// the registry is not stale, and everything else keeps its teeth — no entry, or
/// a difference past the entry's own recorded `max_rel`, still panics here.
///
/// `ctx` is `"<dataset>: <quantity>"` (with an optional `[i]` index suffix),
/// which is the key `divergences.json` is written against.
fn adjudicate(observed: f64, band: f64, ctx: &str, detail: &dyn Fn() -> String) {
    if observed <= band {
        return;
    }
    match divergence::registry().covers(ctx, observed) {
        divergence::Coverage::Documented => {}
        divergence::Coverage::Exceeded { id, max_rel } => panic!(
            "{}\n  documented divergence `{id}` covers only {max_rel:.1e} — this one grew",
            detail()
        ),
        divergence::Coverage::NotDocumented => panic!("{}", detail()),
    }
}

pub fn assert_rel(got: f64, want: f64, band: f64, ctx: &str) {
    let r = rel(got, want);
    adjudicate(r, band, ctx, &|| {
        format!("{ctx}: glmm={got} oracle={want} (rel {r:.3e} > {band:.0e})")
    });
}

pub fn assert_abs(got: f64, want: f64, band: f64, ctx: &str) {
    let d = (got - want).abs();
    adjudicate(d, band, ctx, &|| {
        format!("{ctx}: glmm={got} oracle={want} (abs {d:.3e} > {band:.0e})")
    });
}

// ── documented-divergence registry ───────────────────────────────────────────

/// Reader for `validation/divergences.json`, the registry this tier and
/// `validation/compare.R` share. The reference-check rule it serves is stated in
/// the module header.
///
/// The registry is deliberately hostile to rot — [`Registry::fired`]
/// records every match so the corpus driver can assert that each entry scoped to
/// this tier actually fired, which is what stops a fixed divergence from leaving
/// a standing exemption behind.
pub mod divergence {
    use serde::Deserialize;
    use std::collections::BTreeSet;
    use std::sync::{Mutex, OnceLock};

    #[derive(Deserialize)]
    pub struct Entry {
        pub id: String,
        pub dataset: String,
        pub rung: u32,
        pub comparison: Vec<String>,
        pub quantities: Vec<String>,
        pub max_rel: f64,
        pub direction: String,
        pub summary: String,
        pub review: String,
    }

    #[derive(Deserialize)]
    struct File {
        entries: Vec<Entry>,
    }

    pub enum Coverage {
        /// An entry covers this comparison at this magnitude.
        Documented,
        /// An entry covers the comparison, but the difference outgrew it.
        Exceeded { id: String, max_rel: f64 },
        /// Nothing covers it — the caller must fail.
        NotDocumented,
    }

    pub struct Registry {
        entries: Vec<Entry>,
        fired: Mutex<BTreeSet<String>>,
    }

    /// This tier's name in an entry's `comparison` list.
    const SCOPE: &str = "oracle-tier";

    impl Registry {
        /// Adjudicate one over-band comparison. `ctx` is `"<dataset>: <quantity>"`,
        /// optionally with an `[i]` coefficient-index suffix, as every assertion in
        /// this module formats it.
        pub fn covers(&self, ctx: &str, observed: f64) -> Coverage {
            let Some((dataset, rest)) = ctx.split_once(": ") else {
                return Coverage::NotDocumented;
            };
            // `"beta[0]"` and `"beta"` are the same quantity to the registry: an
            // entry names the quantity, never which coordinate of it moved.
            let quantity = rest.split('[').next().unwrap_or(rest).trim();
            let Some(e) = self.entries.iter().find(|e| {
                e.dataset == dataset
                    && e.quantities.iter().any(|q| q == quantity)
                    && e.comparison.iter().any(|c| c == SCOPE)
            }) else {
                return Coverage::NotDocumented;
            };
            if observed > e.max_rel {
                return Coverage::Exceeded {
                    id: e.id.clone(),
                    max_rel: e.max_rel,
                };
            }
            self.fired
                .lock()
                .expect("divergence registry mutex")
                .insert(e.id.clone());
            Coverage::Documented
        }

        /// Entry ids matched so far.
        pub fn fired(&self) -> BTreeSet<String> {
            self.fired
                .lock()
                .expect("divergence registry mutex")
                .clone()
        }

        /// Entries scoped to this tier, in file order.
        pub fn scoped(&self) -> impl Iterator<Item = &Entry> {
            self.entries
                .iter()
                .filter(|e| e.comparison.iter().any(|c| c == SCOPE))
        }
    }

    pub fn registry() -> &'static Registry {
        static REG: OnceLock<Registry> = OnceLock::new();
        REG.get_or_init(|| {
            let path = format!("{}/validation/divergences.json", env!("CARGO_MANIFEST_DIR"));
            let raw = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let f: File = serde_json::from_str(&raw).unwrap_or_else(|e| panic!("{path}: {e}"));
            Registry {
                entries: f.entries,
                fired: Mutex::new(BTreeSet::new()),
            }
        })
    }
}

// ── deviance-convention alignment ───────────────────────────────────────────

// mirrors validation/tol.R dev_eps / dev_big — change together.
// Pinned 2026-08-24 from the corpus Δdev floor measurement.
pub const DEV_EPS: f64 = 2e-4;
pub const DEV_BIG: f64 = 0.5;

// mirrors validation/dev_align.R — change together
pub mod dev_align {
    //! Per-family × per-method deviance convention alignment, mirroring
    //! `validation/dev_align.R` on the Rust side (checklist comment at both
    //! sites — change together). Shared convention: `dev = -2 * loglik` as each
    //! engine reports it, corrected only for the one documented case below.

    use super::{col_index, csv_for, parse_formula, split_line, Golden};

    /// `ln Γ(x)` for `x > 0`. Same Lanczos g=7 series and coefficients as
    /// `src/simd_transcendental.rs::ln_gamma`, duplicated here because that one
    /// is `pub(crate)` and out of reach from this external test crate — this
    /// harness needs one scalar call, not the SIMD kernel it backs.
    #[allow(clippy::excessive_precision)]
    fn ln_gamma(x: f64) -> f64 {
        const C: [f64; 9] = [
            0.999_999_999_999_809_93,
            676.520_368_121_885_1,
            -1_259.139_216_722_402_8,
            771.323_428_777_653_13,
            -176.615_029_162_140_59,
            12.507_343_278_686_905,
            -0.138_571_095_265_720_12,
            9.984_369_578_019_571_6e-6,
            1.505_632_735_149_311_6e-7,
        ];
        const G: f64 = 7.0;
        const LN_SQRT_2PI: f64 = 0.918_938_533_204_672_74; // ½·ln(2π)
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        LN_SQRT_2PI + (x + 0.5) * t.ln() - t + a.ln()
    }

    /// `ln C(n, k)` via `lnΓ`, continuous in `n` the same way `dev_align.R`'s
    /// `lchoose` is (R's `lchoose` is also `lnΓ`-based, not a factorial table).
    fn lchoose(n: f64, k: f64) -> f64 {
        ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0)
    }

    /// Golden `engine` strings are qualified R calls (`"lme4::glmer"`,
    /// `"lme4::glmer.nb"`, `"lme4::lmer"`), not the bare `"lme4"` the interface
    /// doc names — matched by prefix, same as `dev_align.R::is_lme4`, so the
    /// nAGQ>1 correction actually fires rather than never matching in silence.
    fn is_lme4(engine: &str) -> bool {
        engine.starts_with("lme4")
    }

    /// Closed form of the saturated-model loglik (binomial/poisson), ported
    /// (sign and form) via `dev_align.R::saturated_loglik_deficit` — the
    /// verified closed-form saturated-model logLik correction for lme4
    /// nAGQ>1, verified 2026-08-24. This is the value `aligned_dev` adds
    /// directly to lme4's reported nAGQ>1 loglik. Sign verified against the
    /// frozen `cbpp_agq_k1`/`cbpp_agq_k7` goldens (closes an 84-unit raw
    /// deviance gap to <1 unit).
    fn saturated_loglik_deficit(g: &Golden) -> f64 {
        let spec = parse_formula(&g.r_formula);
        let raw =
            std::fs::read_to_string(csv_for(g)).unwrap_or_else(|e| panic!("{}: {e}", csv_for(g)));
        let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
        let header = split_line(lines.next().expect("CSV has a header"));
        let rows: Vec<Vec<String>> = lines.map(split_line).collect();

        match g.family.as_str() {
            "binomial" => {
                // Aggregated `cbind(s, n-s) ~ ...` carries n per row; the bare
                // `y ~ ...` Bernoulli form is n=1 everywhere, which zeroes every
                // term of the sum below (a Bernoulli fit is always saturated) —
                // not a special case, just this formula evaluated at n=1.
                let (yi, ni) = match &spec.aggregated {
                    Some((succ, total)) => {
                        (col_index(&header, succ), Some(col_index(&header, total)))
                    }
                    None => (col_index(&header, &spec.response), None),
                };
                let mut s = 0.0;
                for r in &rows {
                    let y: f64 = r[yi].parse().expect("successes parse");
                    let n: f64 = match ni {
                        Some(ti) => r[ti].parse().expect("total parse"),
                        None => 1.0,
                    };
                    let p = y / n;
                    s += lchoose(n, y);
                    if y > 0.0 {
                        s += y * p.ln();
                    }
                    if y < n {
                        s += (n - y) * (1.0 - p).ln();
                    }
                }
                s
            }
            "poisson" => {
                let ri = col_index(&header, &spec.response);
                let mut s = 0.0;
                for r in &rows {
                    let y: f64 = r[ri].parse().expect("response parse");
                    let t = if y > 0.0 { y * y.ln() } else { 0.0 };
                    s += t - y - ln_gamma(y + 1.0);
                }
                s
            }
            f => panic!("no verified saturated correction for family {f}"),
        }
    }

    /// Deviance on the shared `-2*loglik` convention, aligned across engines.
    /// Same three branches as `dev_align.R::aligned_dev`: default `-2*loglik`;
    /// lme4 `nagq > 1` adds the saturated-model logLik deficit; no reference
    /// loglik at all -> `None`, which the caller must exclude loudly,
    /// never pass hollow.
    pub fn aligned_dev(g: &Golden) -> Option<f64> {
        let ll = g.estimates.loglik?;
        let ll = if is_lme4(&g.engine) && g.nagq > 1 {
            ll + saturated_loglik_deficit(g)
        } else {
            ll
        };
        Some(-2.0 * ll)
    }
}

/// Compare a per-coefficient vector against the reference, asserting the
/// rank-deficiency divergence wherever the reference dropped a column.
///
/// lme4 and `stats::glm` silently drop an aliased column and report `NA`; glmm
/// keeps the slot, reports NaN and sets `aliased[j]` — a deliberate divergence.
/// Asserting both directions means the divergence has to still be there — if a reference ever
/// starts reporting the coefficient, or glmm stops flagging it, this fails
/// instead of silently passing.
pub fn assert_coefs(
    got: &[f64],
    aliased: &[bool],
    align: &[usize],
    want: &[Option<f64>],
    band: f64,
    ctx: &str,
) {
    assert_eq!(align.len(), want.len(), "{ctx}: coefficient count");
    for (o, (&j, w)) in align.iter().zip(want).enumerate() {
        match w {
            Some(w) => {
                assert!(
                    !aliased[j],
                    "{ctx}[{o}]: glmm flags this column aliased, the reference reports {w}"
                );
                assert_rel(got[j], *w, band, &format!("{ctx}[{o}]"));
            }
            None => assert!(
                aliased[j],
                "{ctx}[{o}]: the reference dropped this column as aliased, glmm reports {}",
                got[j]
            ),
        }
    }
}

/// Map each oracle coefficient onto its glmm design column by NAME, and assert
/// that every column the oracle has no entry for is one glmm flagged aliased.
///
/// The two engines express rank deficiency in two different shapes, and both
/// appear in the corpus: `stats::glm` keeps the coefficient name and writes `NA`
/// (`sim_collinear_glm`), while `lme4::lmer` drops the name from `fixef`
/// altogether (`sim_collinear_lmm` — 3 names against glmm's 4). glmm keeps every
/// column and flags it instead. Aligning by name covers both without either
/// engine's convention leaking into the comparison.
pub fn align_coefs(
    glmm_names: &[String],
    oracle_names: &[String],
    aliased: &[bool],
    ctx: &str,
) -> Vec<usize> {
    let align: Vec<usize> = oracle_names
        .iter()
        .map(|n| {
            glmm_names
                .iter()
                .position(|m| m == n)
                .unwrap_or_else(|| panic!("{ctx}: oracle names column `{n}`, glmm does not"))
        })
        .collect();
    for (j, name) in glmm_names.iter().enumerate() {
        assert!(
            align.contains(&j) || aliased[j],
            "{ctx}: the oracle omits column `{name}` but glmm does not flag it aliased"
        );
    }
    align
}

// ── The generic driver ───────────────────────────────────────────────────────

/// Which directory a dataset lives in — the `sim_` prefix convention the R side
/// uses (`goldens_agq.R::data_dir_of_name`). A golden carrying an explicit
/// `csv` overrides it.
fn csv_for(g: &Golden) -> String {
    if let Some(path) = &g.csv {
        return path.clone();
    }
    let dir = if g.data.starts_with("sim_") {
        "simulated"
    } else {
        "empirical"
    };
    format!(
        "{}/validation/data/{dir}/{}.csv",
        env!("CARGO_MANIFEST_DIR"),
        g.data
    )
}

/// Split a CSV line on commas, trimming whitespace and surrounding quotes.
/// Mirrors the hand-parse the in-crate kernel tests use — no CSV crate, and no
/// quoted-comma fields anywhere in the validation corpus.
fn split_line(line: &str) -> Vec<String> {
    line.split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect()
}

/// Every identifier-like token in `s`, so a column can be tested for whether the
/// formula actually references it.
fn mentions(s: &str, name: &str) -> bool {
    let bytes = s.as_bytes();
    let mut from = 0;
    while let Some(hit) = s[from..].find(name) {
        let a = from + hit;
        let b = a + name.len();
        let left_ok = a == 0 || !is_ident(bytes[a - 1]);
        let right_ok = b == bytes.len() || !is_ident(bytes[b]);
        if left_ok && right_ok {
            return true;
        }
        from = a + 1;
    }
    false
}

fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'.' || b == b'_'
}

/// A golden's `r_formula` in the form the crate's formula frontend accepts,
/// plus the aggregated-binomial response columns when the oracle used lme4's
/// `cbind(successes, failures)` idiom.
struct Spec {
    /// `(successes_column, total_column)` for `cbind(s, n - s) ~ …`.
    aggregated: Option<(String, String)>,
    /// Response column name — synthesised as `y` for the aggregated form.
    response: String,
    /// Right-hand side, verbatim from the oracle.
    rhs: String,
}

/// `cbind(incidence, size - incidence) ~ rhs` → successes `incidence`, total
/// `size`. lme4 writes the second argument as failures; the total is the first
/// identifier in it.
fn parse_formula(r_formula: &str) -> Spec {
    let (lhs, rhs) = r_formula
        .split_once('~')
        .expect("golden r_formula has no `~`");
    // The oracle specs write the intercept explicitly (`y ~ 1 + x`); the crate's
    // formula frontend has no `1` token and takes the intercept as implied. Same
    // model either way — every golden in the corpus includes the intercept, and
    // `refit` asserts the lowered column names against the oracle's own
    // `coef_names`, so a dropped or added intercept fails there rather than
    // quietly changing the model.
    let lhs = lhs.trim();
    let rhs = rhs.trim();
    let rhs = rhs.strip_prefix("1 +").unwrap_or(rhs).trim().to_string();

    let Some(args) = lhs.strip_prefix("cbind(").and_then(|s| s.strip_suffix(')')) else {
        return Spec {
            aggregated: None,
            response: lhs.to_string(),
            rhs,
        };
    };
    let (succ, fail) = args.split_once(',').expect("cbind needs two arguments");
    let total = fail
        .trim()
        .split(|c: char| !c.is_ascii() || !is_ident(c as u8))
        .find(|t| !t.is_empty())
        .expect("cbind failure term names no column");
    Spec {
        aggregated: Some((succ.trim().to_string(), total.to_string())),
        response: "y".to_string(),
        rhs,
    }
}

/// Refit a golden from its own recorded `r_formula` and dataset.
///
/// The oracle records the model it fitted, so Tier 2 drives the crate from that
/// string rather than a hand-built design — a hand-built design can silently
/// answer a different question than the golden froze, which is exactly the class
/// of defect this tier exists to catch.
pub fn refit(g: &Golden, factors: &[&str]) -> (Fit, Vec<String>, Vec<String>) {
    refit_with(g, factors, WaldSe::Hessian)
}

/// As [`refit`], under a chosen Wald-SE method. The GLMM goldens freeze both of
/// lme4's methods (`se_hessian` = `use.hessian=TRUE`, `se_rx` = the Schur
/// complement conditional on θ̂), and each must be compared against the matching
/// glmm setting — crossing them compares two different estimators and would
/// have to be excused with a loose band.
pub fn refit_with(
    g: &Golden,
    factors: &[&str],
    wald_se: WaldSe,
) -> (Fit, Vec<String>, Vec<String>) {
    let spec = parse_formula(&g.r_formula);
    let raw = std::fs::read_to_string(csv_for(g)).unwrap_or_else(|e| panic!("{}: {e}", csv_for(g)));
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());
    let header = split_line(lines.next().expect("CSV has a header"));
    let rows: Vec<Vec<String>> = lines.map(split_line).collect();

    // Aggregated binomial: fit the proportion with the trial count as a prior
    // weight, which is the objective `glm`/`glmer` actually minimise for
    // `cbind(s, n - s)`. Expanding to Bernoulli rows would find the same MLE but
    // a different logLik — the aggregated form keeps the `ln C(nᵢ, sᵢ)` terms —
    // so the golden's `loglik` is only comparable against this form.
    let (y_agg, weights) = match &spec.aggregated {
        None => (None, None),
        Some((succ, total)) => {
            let si = col_index(&header, succ);
            let ti = col_index(&header, total);
            let mut y = Vec::with_capacity(rows.len());
            let mut w = Vec::with_capacity(rows.len());
            for r in &rows {
                let s: f64 = r[si].parse().expect("successes parse");
                let n: f64 = r[ti].parse().expect("total parse");
                y.push(s / n);
                w.push(n);
            }
            (Some(y), Some(w))
        }
    };
    // Plain per-row prior weights (the weights tier, rungs 29-43). The manifest
    // makes this mutually exclusive with the aggregated-binomial trial count,
    // which occupies the same `FitOptions::weights` slot.
    let weights = match &g.weights_col {
        None => weights,
        Some(col) => {
            assert!(
                weights.is_none(),
                "{}: weights_col and an aggregated response both claim the weight slot",
                g.name
            );
            let wi = col_index(&header, col);
            Some(
                rows.iter()
                    .map(|r| r[wi].parse().expect("weight parse"))
                    .collect(),
            )
        }
    };

    let mut columns: Vec<(String, Column)> = Vec::new();
    if let Some(y) = y_agg {
        columns.push((spec.response.clone(), Column::Numeric(y)));
    }
    for (j, name) in header.iter().enumerate() {
        // Only columns the formula names: the validation CSVs carry extra columns
        // (ids, labels) that are not numeric and are not part of any model.
        if !mentions(&spec.rhs, name) && *name != spec.response {
            continue;
        }
        if columns.iter().any(|(n, _)| n == name) {
            continue;
        }
        let cells: Vec<String> = rows.iter().map(|r| r[j].clone()).collect();
        let col = if factors.contains(&name.as_str()) {
            Column::factor_from_labels(&cells)
        } else {
            Column::Numeric(
                cells
                    .iter()
                    .map(|c| c.parse().expect("numeric parse"))
                    .collect(),
            )
        };
        columns.push((name.clone(), col));
    }

    let table = Table {
        n: rows.len(),
        columns,
    };
    let formula = format!("{} ~ {}", spec.response, spec.rhs);
    // The design is NOT asserted equal to `coef_names` here: where the fit is
    // rank deficient the oracle's list is the shorter one (see `align_coefs`).
    // The caller aligns by name, which subsumes the equality check — an
    // unexpected extra or missing column fails there.
    let lo = lower(&formula, &table, family_of(g)).unwrap_or_else(|e| panic!("{formula}: {e:?}"));

    let opts = glmm::FitOptions {
        nagq: g.nagq,
        wald_se,
        weights,
        ..lo.opts
    };
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &opts);
    let group_names = lo.re_groups.iter().map(|r| r.name.clone()).collect();
    (fit, lo.col_names, group_names)
}

fn col_index(header: &[String], name: &str) -> usize {
    header
        .iter()
        .position(|h| h == name)
        .unwrap_or_else(|| panic!("CSV has no column {name}"))
}

fn family_of(g: &Golden) -> Family {
    // A golden with no explicit `link` used the harness default, which is the
    // canonical link per family EXCEPT for Gamma: `engines/lme4.R` and
    // `goldens_agq.R` both pass `Gamma(link = "log")`, not R's own
    // `Gamma()` default of inverse. Getting this wrong would fit a different
    // model than the one frozen, so it is spelled out rather than left to
    // `unwrap_or("identity")`.
    let link = g.link.as_str().unwrap_or_else(|| match g.family.as_str() {
        "gaussian" => "identity",
        "binomial" => "logit",
        "poisson" | "gamma" | "negbin" => "log",
        f => panic!("golden {}: no default link known for family {f}", g.name),
    });
    match (g.family.as_str(), link) {
        ("gaussian", _) => Family::Gaussian,
        ("binomial", "logit") => Family::Binomial {
            link: BinomialLink::Logit,
        },
        ("binomial", "probit") => Family::Binomial {
            link: BinomialLink::Probit,
        },
        ("binomial", "cloglog") => Family::Binomial {
            link: BinomialLink::Cloglog,
        },
        ("poisson", "log") => Family::Poisson {
            link: PoissonLink::Log,
        },
        ("gamma", "log") => Family::Gamma {
            link: GammaLink::Log,
        },
        ("gamma", "inverse") => Family::Gamma {
            link: GammaLink::Inverse,
        },
        ("inversegaussian", "log") => Family::InverseGaussian {
            link: InverseGaussianLink::Log,
        },
        ("inversegaussian", "inverse_squared") => Family::InverseGaussian {
            link: InverseGaussianLink::InverseSquared,
        },
        ("negbin", "log") => Family::NegativeBinomial {
            link: NegBinomialLink::Log,
        },
        (f, l) => panic!("golden {}: unsupported family/link {f}/{l}", g.name),
    }
}
