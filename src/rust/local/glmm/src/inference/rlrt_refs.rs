//! The frozen exact-RLRT references, typed, for the in-crate RLRT gates.
//!
//! A thin layer over [`super::test_refs`], which keeps the manifest parsing,
//! the JSON paths and the divergence registry: this module adds only the typed
//! fields of `validation/goldens/inference/rlrt/<name>.json` (as
//! `validation/tools/goldens_rlrt.R` writes them), the corpus CSV read by
//! `CARGO_MANIFEST_DIR`, the lowering of each entry's `r_formula`, and the
//! gates' [`bands`].

use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::Value;

use super::test_refs::{self, Entry, Kind};

#[cfg(feature = "formula")]
use crate::{
    formula::{lower, Column},
    test_support::csv_table,
    Family, GroupIds, ModelSpec,
};

/// One frozen reference. Every entry carries `n`, `p` and the two REML
/// log-likelihoods; the RLRsim and F-test blocks are present only where the
/// generator writes them.
#[derive(Debug, Deserialize)]
pub(crate) struct RlrtRef {
    /// The reference name, the JSON's file stem.
    pub name: String,
    /// Rows of the data.
    pub n: usize,
    /// Columns of the fixed-effects design.
    pub p: usize,
    /// lme4's REML log-likelihood of the `(1 | g)` model.
    pub loglik_reml: f64,
    /// `logLik(lm, REML = TRUE)` of the fixed-only model (weighted on the
    /// entry with a `weights_col`).
    pub loglik_lm_reml: f64,
    /// RLRsim's block, keyed on `K`: on every entry but the weighted
    /// reduced-REML one, which RLRsim does not handle.
    #[serde(skip)]
    pub rlrsim: Option<Rlrsim>,
    /// The balanced intercept-only F test, keyed on `F_obs`.
    #[serde(skip)]
    pub f_test: Option<FTest>,
}

/// RLRsim's design, observed test and null at the entry's seed.
#[derive(Debug, Deserialize)]
pub(crate) struct Rlrsim {
    /// Levels of the grouping factor.
    #[serde(rename = "K")]
    pub k: usize,
    /// The normalized eigenvalues, all `K`, zeros kept.
    pub mu: Vec<f64>,
    /// The largest eigenvalue before normalization.
    pub max_mu: f64,
    /// RLRsim's λ grid, in units of `1 / max_mu`.
    pub grid: Vec<f64>,
    /// The `set.seed` of the null.
    pub seed: u64,
    /// Draws of the null.
    pub nsim: usize,
    /// RLRsim's observed statistic.
    pub statistic: f64,
    /// RLRsim's p-value on `statistic` at `nsim`.
    pub p_value: f64,
    /// Null draws exactly zero.
    pub null_n_zero: usize,
    /// The positive null draws, sorted ascending.
    pub null_pos: Vec<f64>,
    /// The first draws' inputs and outputs, in draw order; only on the
    /// entries whose manifest `paired_draws` is positive.
    pub paired: Option<Paired>,
}

/// RLRsim's per-draw inputs and outputs, one element per draw.
#[derive(Debug, Deserialize)]
pub(crate) struct Paired {
    /// The tail χ² draw (`n − p − k` degrees of freedom).
    #[serde(rename = "ChiK")]
    pub chi_k: Vec<f64>,
    /// The `k` one-degree χ² draws.
    #[serde(rename = "Chi1")]
    pub chi1: Vec<Vec<f64>>,
    /// The statistic at the first decrease along the grid.
    pub res: Vec<f64>,
    /// Its 1-based grid index.
    pub lambdaind: Vec<usize>,
}

/// The F test of a balanced intercept-only design.
#[derive(Debug, Deserialize)]
pub(crate) struct FTest {
    /// The ANOVA F statistic.
    #[serde(rename = "F_obs")]
    pub f_obs: f64,
    /// `pf(F_obs, K − 1, n − K, lower = FALSE)`.
    #[serde(rename = "F_p")]
    pub p: f64,
    /// `pf(1, K − 1, n − K)`, the null's mass at zero.
    #[serde(rename = "F_atom")]
    pub atom: f64,
}

/// The manifest entries of kind `rlrt`.
pub(crate) fn entries() -> Vec<Entry> {
    test_refs::entries(Kind::Rlrt)
}

/// The frozen reference `name`, typed. Panics when a field is missing or
/// mistyped.
pub(crate) fn load(name: &str) -> RlrtRef {
    let json = test_refs::load(Kind::Rlrt, name);
    let mut r: RlrtRef = typed(&json, name);
    r.rlrsim = json.get("K").map(|_| typed(&json, name));
    r.f_test = json.get("F_obs").map(|_| typed(&json, name));
    r
}

fn typed<T: DeserializeOwned>(json: &Value, name: &str) -> T {
    T::deserialize(json).unwrap_or_else(|e| panic!("rlrt/{name}.json: {e}"))
}

/// The text of a corpus CSV, `path` relative to `validation/` (an entry's
/// `data`).
pub(crate) fn csv(path: &str) -> String {
    let full = format!("{}/validation/{path}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("{full}: {e}"))
}

/// An entry's model on its data: `(x, y, n, p, model, ids, weights)`.
#[cfg(feature = "formula")]
pub(crate) type Design = (
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
    ModelSpec,
    GroupIds,
    Option<Vec<f64>>,
);

/// Entry `name`'s Gaussian `r_formula` lowered on its CSV: the columns in
/// `factors` read as factors (and any column that is not numeric), `x`
/// row-major, the `(1 | g)` model and its group ids, and the prior weights of
/// the entry's `weights_col` (`None` without one).
#[cfg(feature = "formula")]
pub(crate) fn design(name: &str) -> Design {
    let entry = entries()
        .into_iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("{name}: no rlrt manifest entry"));
    let data = entry.data.as_deref().expect("an rlrt entry has data");
    let formula = entry
        .r_formula
        .as_deref()
        .expect("an rlrt entry has a formula");
    let factors: Vec<&str> = entry.factors.iter().map(String::as_str).collect();
    let table = csv_table(&csv(data), &factors);
    let weights = entry.raw.get("weights_col").map(|w| {
        let col = w.as_str().expect("`weights_col` is a string");
        match table.columns.iter().find(|(c, _)| c == col) {
            Some((_, Column::Numeric(v))) => v.clone(),
            _ => panic!("{name}: no numeric column {col}"),
        }
    });
    let lo = lower(formula, &table, Family::Gaussian).unwrap_or_else(|e| panic!("{name}: {e}"));
    (lo.x, lo.y, lo.n, lo.p, lo.model, lo.ids, weights)
}

/// The gates' bands. Each gate adds its constant here with the measurement
/// that set it, and its row in the inference section of
/// `validation/grid/tol.R` in the same change: mirrors `validation/grid/tol.R`,
/// change together. Fit-free arithmetic is hard, an end-to-end comparison
/// against R is registry-backed, and a Monte Carlo band comes from fixed seeds
/// and its standard error.
pub(crate) mod bands {
    /// `rlrt_mu_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`, hard): the
    /// normalized nonzero design eigenvalues against RLRsim's, per value, in
    /// units of the largest (`inference::rlrt_oracle_tests`). Set a priori;
    /// measured (2026-10-12, pawel) worst 8.99e-15 on `rlrt_large_k`.
    /// Change together.
    pub(crate) const RLRT_MU_REL: f64 = 1e-10;
    /// `rlrt_paired_rel` in `validation/grid/tol.R` (`TOL_INFERENCE`, hard):
    /// with [`RLRT_PAIRED_ABS`], the first-decrease maximum of the kernel's
    /// profile on RLRsim's frozen per-draw inputs against RLRsim's `res`
    /// (`inference::rlrt_oracle_tests::paired_profile_matches_rlrsim`). Set
    /// from the R replay's 4.3e-14 and the 15 significant digits the JSON
    /// keeps; measured (2026-10-12, pawel) over the 2000 paired draws: worst
    /// absolute difference 5.91e-14, worst use of the band 5.0%
    /// (`rlrt_cake`, draw 206), every index equal. Change together.
    pub(crate) const RLRT_PAIRED_REL: f64 = 1e-12;
    /// `rlrt_paired_abs` in `validation/grid/tol.R` (`TOL_INFERENCE`, hard):
    /// the absolute part of the band of [`RLRT_PAIRED_REL`], for draws whose
    /// `res` is at or near zero. Change together.
    pub(crate) const RLRT_PAIRED_ABS: f64 = 1e-13;
    /// `rlrt_ks_coef` in `validation/grid/tol.R` (`TOL_INFERENCE`, Monte
    /// Carlo): the two-sample KS distance between the null at `B` draws and
    /// RLRsim's frozen null at `B_ref`, within this times
    /// `√((B + B_ref)/(B·B_ref))`, the asymptotic critical value at
    /// α = 0.001 (`inference::rlrt_oracle_tests::ks_null_matches_rlrsim`);
    /// 0.0087 at `B = B_ref = 10⁵`. Measured (2026-10-12, pawel) over the six
    /// RLRsim references: worst 4.90e-3, 56% of the band (`rlrt_cake`).
    /// Change together.
    pub(crate) const RLRT_KS_COEF: f64 = 1.95;
    /// `rlrt_zero_mass_se` in `validation/grid/tol.R` (`TOL_INFERENCE`, Monte
    /// Carlo): the null's mass at zero against RLRsim's frozen null's, within
    /// this many pooled two-proportion standard errors (same test). Measured
    /// (2026-10-12, pawel) over the six RLRsim references: worst 1.77
    /// (`rlrt_cake`). Change together.
    pub(crate) const RLRT_ZERO_MASS_SE: f64 = 4.0;
    /// `rlrt_stat_abs` in `validation/grid/tol.R` (`TOL_INFERENCE`,
    /// registry-backed, quantity `rlrt`): the observed statistic from the two
    /// fits against RLRsim's, absolute
    /// (`inference::rlrt_oracle_tests::rlrt_observed_matches_rlrsim`). The
    /// statistic is `2(ℓ_R,1 − ℓ_R,0)`, so twice `loglik_abs_lmm` (2e-6 on the
    /// LMM REML log-likelihood against lme4). Measured (2026-10-12, pawel) over
    /// the six RLRsim references: worst 8.19e-12 (`rlrt_sleepstudy`);
    /// `rlrt_dyestuff2`'s exact 0 against RLRsim's 5.68e-14. Change together.
    pub(crate) const RLRT_STAT_ABS: f64 = 4e-6;
    /// `rlrt_dkw_alpha` in `validation/grid/tol.R` (`TOL_INFERENCE`, Monte
    /// Carlo): gate 1's fit-free part, the simulated null of four balanced
    /// intercept-only designs at `B = 10⁵` against the one-way ANOVA F test,
    /// its ECDF at the F quantiles and its mass at zero within the DKW band
    /// `√(ln(2/α)/(2B))` at this α, 6.165e-3
    /// (`inference::rlrt_tests::balanced_null_matches_anova_f`). Measured
    /// (2026-10-12, pawel): worst 1.76e-3, 29% of the band (the mass at zero
    /// of `(K, J) = (10, 3)`), every quantile gap at or below 1.48e-3.
    /// Change together.
    pub(crate) const RLRT_DKW_ALPHA: f64 = 1e-3;
    /// `rlrt_p_se` in `validation/grid/tol.R` (`TOL_INFERENCE`): the
    /// simulated p-value within this many binomial standard errors plus
    /// `1/(B + 1)`, the p-value's own floor. Against RLRsim's frozen p at
    /// `B = B_ref = 10⁵` (registry-backed, quantity `p_value`), the two
    /// independent nulls' standard error `√(2p(1 − p)/B)`
    /// (`inference::rlrt_oracle_tests::rlrt_observed_matches_rlrsim`); against
    /// Dyestuff's exact F-test p, `√(p(1 − p)/B)`
    /// (`inference::rlrt_tests::rlrt_dyestuff_p_matches_anova_f`). Measured
    /// (2026-10-12, pawel): against RLRsim, worst 44% of the band
    /// (`rlrt_few_unbal`), exactly `1/(B + 1)` where RLRsim's p is 0, and
    /// `rlrt_dyestuff2` registered; against the F test, 9%. Change together.
    pub(crate) const RLRT_P_SE: f64 = 4.0;
    /// `rlrt_level_se` in `validation/grid/tol.R` (`TOL_INFERENCE`, Monte
    /// Carlo): gate 7, the level study's rejection rate at each α within this
    /// many binomial standard errors `√(α(1 − α)/R_kept)` of α, `R_kept` the
    /// datasets kept (`R` less those whose `(1 | g)` fit did not converge or
    /// whose statistic is NaN, at most 1%); ±0.0146 at α = 0.05 with all
    /// `R = 2000` kept
    /// (`inference::rlrt_montecarlo_tests::level_rejection_rates`). Set a
    /// priori. Change together.
    pub(crate) const RLRT_LEVEL_SE: f64 = 3.0;
    /// `reml0_loglik_abs` in `validation/grid/tol.R` (`TOL_INFERENCE`, hard):
    /// the REML log-likelihood of the OLS fit made with `reml = Some(true)`,
    /// the value at `θ = 0`, against R's `logLik(lm, REML = TRUE)`, absolute,
    /// on every rlrt reference, the weighted one included
    /// (`inference::rlrt_oracle_tests::ols_reml_loglik_matches_lm_reml`). Both
    /// are one closed form, so the band is set a priori. Measured (2026-10-12,
    /// pawel) over the seven references: worst 9.09e-13 (`rlrt_large_k`).
    /// Change together.
    pub(crate) const REML0_LOGLIK_ABS: f64 = 1e-9;
}

#[cfg(all(feature = "oracle-tests", feature = "formula"))]
#[test]
fn every_rlrt_reference_loads() {
    let entries = entries();
    assert!(!entries.is_empty(), "the manifest registers rlrt entries");
    for e in &entries {
        let name = &e.name;
        assert!(!e.is_pending(), "{name}: still pending");
        let r = load(name);
        assert_eq!(&r.name, name, "{name}: the JSON's own name");
        let (_, _, n, p, _, ids, _) = design(name);
        assert_eq!(
            (n, p),
            (r.n, r.p),
            "{name}: (n, p) lowered against the JSON"
        );
        let positive = |field: &str| e.raw[field].as_u64().is_some_and(|v| v > 0);
        assert_eq!(r.rlrsim.is_some(), positive("nsim"), "{name}: RLRsim block");
        // Any `F_` key means the F block, so a renamed `F_obs` cannot read as
        // an absent block.
        let json = test_refs::load(Kind::Rlrt, name);
        let f_keys = json
            .as_object()
            .is_some_and(|o| o.keys().any(|k| k.starts_with("F_")));
        assert_eq!(r.f_test.is_some(), f_keys, "{name}: F block");
        if let Some(sim) = &r.rlrsim {
            let levels = ids
                .primary
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len();
            assert_eq!(sim.k, levels, "{name}: K against the lowered levels");
            assert_eq!(sim.mu.len(), sim.k, "{name}: mu has K values");
            assert_eq!(
                sim.paired.is_some(),
                positive("paired_draws"),
                "{name}: paired block"
            );
            if let Some(pd) = &sim.paired {
                let draws = e.raw["paired_draws"].as_u64().unwrap() as usize;
                for (field, len) in [
                    ("ChiK", pd.chi_k.len()),
                    ("Chi1", pd.chi1.len()),
                    ("res", pd.res.len()),
                    ("lambdaind", pd.lambdaind.len()),
                ] {
                    assert_eq!(len, draws, "{name}: {field} has paired_draws rows");
                }
                assert!(
                    pd.chi1.iter().all(|row| row.len() == sim.k),
                    "{name}: every Chi1 row has K values"
                );
            }
        }
    }
}
