//! Gate 7 of the exact RLRT, its level at nominal α (Monte Carlo tier,
//! `montecarlo-tests`): on three designs under the null, the rejection rate of
//! the simulated null's p-value against α, with the rates of the χ²₁ and the
//! 50:50 mixture of χ²₀ and χ²₁ printed beside it as a record of their
//! conservatism, not gated.
//!
//! The test is pivotal in `β` and `σ²` under the null, so every dataset is
//! `y = ε`, `ε` iid `N(0, 1)`, on the design's `X` and groups.

use super::rlrt::{rlrt_design, rlrt_null, rlrt_statistic, RlrtConfig};
use super::rlrt_refs::{bands, design};
use super::tails::chisq_upper_p;
use crate::{fit_cold, Family, FitOptions, GroupIds, ModelSpec};
use commonstats::dist::{Normal, Sampler};
use commonstats::rng::{CommonStatsRng, STREAM_TAG_SIMULATE};
use std::time::Instant;

/// Datasets per design.
const R: usize = 2000;

/// Draws of each design's null.
const B: usize = 100_000;

/// The nominal levels.
const ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];

/// Per design: the rlrt reference whose `X` and groups it takes, the seed of
/// its datasets and the seed of its null.
const DESIGNS: [(&str, u64, u64); 3] = [
    ("rlrt_dyestuff", 20261051, 20261041),
    ("rlrt_few_unbal", 20261052, 20261042),
    ("rlrt_cake", 20261053, 20261043),
];

/// Gate 7. Per design (Dyestuff's, `K = 6` of `J = 5` with `X = 1`; the
/// unbalanced `K = 4` simulated design; cake's, `K = 45`): one null of
/// `B = 10⁵` draws from the design summary of the first dataset's `(1 | g)`
/// fit (the summary does not depend on `y`), then per dataset `r < R` the
/// `(1 | g)` fit and the OLS fit made with `reml = Some(true)`, both with
/// `fit_cold` and `satterthwaite: false`, the statistic from their two
/// log-likelihoods and the pinned flag, and its p-value from the null. A
/// dataset whose `(1 | g)` fit does not converge, or whose statistic is NaN,
/// is counted and left out; more than 1% of `R` fails the test. The rejection
/// rate at each α within [`bands::RLRT_LEVEL_SE`] binomial standard errors
/// over the datasets kept.
#[test]
fn level_rejection_rates() {
    let start = Instant::now();
    let full_opts = FitOptions {
        satterthwaite: false,
        ..FitOptions::default()
    };
    let ols_opts = FitOptions {
        satterthwaite: false,
        reml: Some(true),
        ..FitOptions::default()
    };
    let ols = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let normal = Normal::new(0.0, 1.0).expect("N(0, 1)");
    let mut misses = vec![];
    for (name, data_seed, null_seed) in DESIGNS {
        let t0 = Instant::now();
        let (x, _, n, p, model, ids, weights) = design(name);
        assert!(weights.is_none(), "{name}: unweighted");
        let response = |r: usize| -> Vec<f64> {
            let mut rng = CommonStatsRng::new_tagged(data_seed, r as u64, STREAM_TAG_SIMULATE);
            (0..n).map(|_| normal.sample(&mut rng)).collect()
        };

        let first = fit_cold(&x, &response(0), n, p, &model, &ids, &full_opts);
        let ds = first
            .design_summary
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: no design summary"));
        let d = rlrt_design(&ds.counts, &ds.s, &ds.c)
            .unwrap_or_else(|e| panic!("{name}: {}", e.reason()));
        let cfg = RlrtConfig {
            nsim: B,
            seed: null_seed,
            parallel_inner: true,
        };
        let null = rlrt_null(&d, &cfg);
        let null_s = t0.elapsed().as_secs_f64();

        // Rows: the simulated null, χ²₁, the 50:50 mixture; columns: `ALPHAS`.
        let mut rejected = [[0usize; 3]; 3];
        let mut failed = 0;
        for r in 0..R {
            let y = response(r);
            let full = fit_cold(&x, &y, n, p, &model, &ids, &full_opts);
            let reduced = fit_cold(&x, &y, n, p, &ols, &GroupIds::default(), &ols_opts);
            assert!(full.reml && reduced.reml, "{name}, dataset {r}: REML");
            let pinned = full.diagnostics.pinned.iter().flatten().any(|&b| b);
            let t = rlrt_statistic(full.loglik, reduced.loglik, pinned);
            if !full.converged() || t.is_nan() {
                failed += 1;
                continue;
            }
            let chi1 = chisq_upper_p(t, 1.0);
            let mixture = if t > 0.0 { 0.5 * chi1 } else { 1.0 };
            for (row, p_value) in rejected.iter_mut().zip([null.p_value(t), chi1, mixture]) {
                for (count, &alpha) in row.iter_mut().zip(&ALPHAS) {
                    *count += usize::from(p_value <= alpha);
                }
            }
        }
        assert!(
            100 * failed <= R,
            "{name}: {failed} of {R} datasets failed, more than 1%"
        );

        let kept = R - failed;
        let rate = |count: usize| count as f64 / kept as f64;
        eprintln!(
            "{name}: n = {n}, p = {p}, K = {}, k = {}, null mass at zero {:.4}, \
             {failed} of {R} datasets failed",
            ds.counts.len(),
            d.mu.len(),
            1.0 - null.draws_pos.len() as f64 / B as f64,
        );
        for (j, &alpha) in ALPHAS.iter().enumerate() {
            let band = bands::RLRT_LEVEL_SE * (alpha * (1.0 - alpha) / kept as f64).sqrt();
            let exact = rate(rejected[0][j]);
            let gap = (exact - alpha).abs();
            eprintln!(
                "{name}: alpha {alpha:.2}: exact {exact:.4} (|d| {gap:.4}, band {band:.4}, \
                 {:.0}%), chi2_1 {:.4}, mixture {:.4}",
                100.0 * gap / band,
                rate(rejected[1][j]),
                rate(rejected[2][j]),
            );
            if gap > band {
                misses.push(format!("{name}, alpha {alpha}: {exact} outside ±{band}"));
            }
        }
        eprintln!(
            "{name}: wall {:.1} s (design and null {null_s:.1} s)",
            t0.elapsed().as_secs_f64()
        );
    }
    eprintln!("level study: wall {:.1} s", start.elapsed().as_secs_f64());
    assert!(misses.is_empty(), "rates outside their bands: {misses:#?}");
}
