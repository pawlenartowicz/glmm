//! The exact RLRT against RLRsim's frozen references
//! (`validation/goldens/inference/rlrt/`, read through
//! [`super::rlrt_refs`]): the design eigenvalues, the grid kernel, the
//! simulated null, and the observed statistic and p-value of `exact_rlrt`
//! with the REML log-likelihood at `θ = 0` (against R's `lm`).

use super::rlrt::{
    exact_rlrt, max_on_grid, profile, rlrt_design, rlrt_null, GridTables, RlrtConfig, LAMBDA_GRID,
    MU_ZERO_REL,
};
use super::rlrt_refs::{bands, design, entries, load};
use super::test_refs::{covers, registry_entries, stale_entries, Coverage, DivEntry};
use crate::test_support::{fit_cold, fit_warm_wo};
use crate::{Family, Fit, FitOptions, GroupIds, ModelSpec};
use std::collections::BTreeSet;
use std::time::Instant;

/// Per reference with an RLRsim block: the lowered `(1 | g)` model fitted
/// with `fit_cold`, `rlrt_design` on the fit's design summary, and its
/// normalized nonzero `μ` against RLRsim's frozen `mu` within
/// [`bands::RLRT_MU_REL`] (both are in units of `μ_max`), with as many frozen
/// values below `MU_ZERO_REL` as `rlrt_design` dropped. Hard: no registry
/// entry excuses a miss.
#[test]
fn rlrt_mu_matches_rlrsim() {
    let mut compared = 0;
    let mut overall = (0.0f64, String::new());
    for e in entries() {
        let name = &e.name;
        let r = load(name);
        let Some(sim) = &r.rlrsim else { continue };
        let (x, y, n, p, model, ids, weights) = design(name);
        assert!(weights.is_none(), "{name}: RLRsim has no weights");
        let fit = fit_cold(&x, &y, n, p, &model, &ids, &FitOptions::default());
        let ds = fit
            .design_summary
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: no design summary"));
        let d = rlrt_design(&ds.counts, &ds.s, &ds.c)
            .unwrap_or_else(|e| panic!("{name}: {}", e.reason()));
        assert_eq!(ds.counts.len(), sim.k, "{name}: K");
        assert_eq!(d.n_minus_p, r.n - r.p, "{name}: n - p");
        let mut frozen = sim.mu.clone();
        frozen.sort_by(|a, b| b.total_cmp(a));
        let k = frozen.iter().filter(|&&m| m >= MU_ZERO_REL).count();
        assert_eq!(d.mu.len(), k, "{name}: nonzero count");
        let worst =
            d.mu.iter()
                .zip(&frozen)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);
        eprintln!("{name}: K = {}, k = {k}, max |dmu| = {worst:.3e}", sim.k);
        assert!(
            worst <= bands::RLRT_MU_REL,
            "{name}: max |dmu| {worst:e} > {:e}",
            bands::RLRT_MU_REL
        );
        if worst >= overall.0 {
            overall = (worst, name.clone());
        }
        compared += 1;
    }
    assert!(compared > 0, "no RLRsim reference compared");
    eprintln!(
        "rlrt_mu: worst {:.3e} on {} over {compared} references",
        overall.0, overall.1
    );
}

/// Gate 2, hard. Per reference with a paired block, RLRsim's own inputs fed
/// to [`profile`]: its `mu` (all `K`, zeros kept, in its order), its grid,
/// and per draw its `Chi1` and `ChiK`, with `n − p` from the JSON. RLRsim's
/// rule applied to the profile (walk while `f_i` is at least the running
/// maximum, stop at the first decrease) must give its `res` within
/// [`bands::RLRT_PAIRED_REL`] relative plus [`bands::RLRT_PAIRED_ABS`], and
/// its 1-based `lambdaind`. The global maximum, from [`max_on_grid`] on the
/// same inputs, is printed wherever it differs from the first decrease, not
/// failed. Also reports [`LAMBDA_GRID`] against each frozen grid (the two
/// come from different `exp` and `seq` arithmetic, and the JSON keeps 15
/// significant digits).
#[test]
fn paired_profile_matches_rlrsim() {
    let (mut compared, mut draws, mut differ) = (0, 0, 0);
    let (mut worst_abs, mut worst_rel, mut worst_use) = (0.0f64, 0.0f64, (0.0f64, String::new()));
    let mut grid_worst = 0.0f64;
    for e in entries() {
        let name = &e.name;
        let r = load(name);
        let Some(sim) = &r.rlrsim else { continue };
        assert_eq!(sim.grid.len(), LAMBDA_GRID.len(), "{name}: grid length");
        let grid_rel = sim
            .grid
            .iter()
            .zip(LAMBDA_GRID.iter())
            .map(|(a, b)| (a - b).abs() / b.abs().max(f64::MIN_POSITIVE))
            .fold(0.0f64, f64::max);
        eprintln!("{name}: max relative |LAMBDA_GRID - grid| = {grid_rel:.3e}");
        grid_worst = grid_worst.max(grid_rel);
        let Some(pd) = &sim.paired else { continue };
        let n_minus_p = r.n - r.p;
        let tables = GridTables::new(&sim.mu, &sim.grid);
        for (d, (chi1, &chi_k)) in pd.chi1.iter().zip(&pd.chi_k).enumerate() {
            let f = profile(&sim.mu, &sim.grid, chi1, chi_k, n_minus_p);
            let (mut first, mut at) = (f[0], 0);
            for (i, &v) in f.iter().enumerate().skip(1) {
                if v >= first {
                    (first, at) = (v, i);
                } else {
                    break;
                }
            }
            let (res, ind) = (pd.res[d], pd.lambdaind[d]);
            let diff = (first - res).abs();
            let band = bands::RLRT_PAIRED_REL * res.abs() + bands::RLRT_PAIRED_ABS;
            assert!(
                diff <= band && at + 1 == ind,
                "{name} draw {}: first decrease {first} at {} against res {res} at {ind}",
                d + 1,
                at + 1
            );
            worst_abs = worst_abs.max(diff);
            if res != 0.0 {
                worst_rel = worst_rel.max(diff / res.abs());
            }
            if diff / band >= worst_use.0 {
                worst_use = (diff / band, format!("{name} draw {}", d + 1));
            }
            let (t, g) = max_on_grid(&tables, chi1, chi_k, n_minus_p);
            assert_eq!(t.to_bits(), f[g].to_bits(), "{name} draw {}", d + 1);
            if g != at {
                differ += 1;
                eprintln!(
                    "{name} draw {}: global maximum {t} at {} above the first decrease {first} at {}",
                    d + 1,
                    g + 1,
                    at + 1
                );
            }
            draws += 1;
        }
        compared += 1;
    }
    assert!(compared > 0, "no paired block compared");
    eprintln!(
        "paired: {draws} draws over {compared} references; worst |d| {worst_abs:.3e}, \
         worst relative {worst_rel:.3e}, worst share of the band {:.3e} ({}); \
         first decrease differs from the global maximum on {differ} draws; \
         worst relative grid difference {grid_worst:.3e}",
        worst_use.0, worst_use.1
    );
}

/// The two-sample Kolmogorov-Smirnov distance between two nulls, each given
/// as its count of exact zeros, its positive draws sorted ascending and its
/// size: the largest gap between the two ECDFs over every value either takes.
fn ks_distance(a: (usize, &[f64], usize), b: (usize, &[f64], usize)) -> f64 {
    let ((za, pa, na), (zb, pb, nb)) = (a, b);
    let ecdf = |z: usize, i: usize, n: usize| (z + i) as f64 / n as f64;
    let (mut i, mut j) = (0, 0);
    let mut d = (ecdf(za, 0, na) - ecdf(zb, 0, nb)).abs();
    while i < pa.len() || j < pb.len() {
        let x = match (pa.get(i), pb.get(j)) {
            (Some(&u), Some(&v)) => u.min(v),
            (Some(&u), None) => u,
            (None, Some(&v)) => v,
            (None, None) => unreachable!(),
        };
        while pa.get(i).is_some_and(|&u| u <= x) {
            i += 1;
        }
        while pb.get(j).is_some_and(|&v| v <= x) {
            j += 1;
        }
        d = d.max((ecdf(za, i, na) - ecdf(zb, j, nb)).abs());
    }
    d
}

/// Gate 4, a Monte Carlo band. Per reference with an RLRsim block: the
/// `(1 | g)` fit's design summary through `rlrt_design`, its null at
/// `B = 10⁵` and seed `20261021 +` the entry's index among the manifest's
/// `rlrt` entries, against RLRsim's frozen null. The two-sample KS distance
/// within `bands::RLRT_KS_COEF·√((B + B_ref)/(B·B_ref))` (α = 0.001; the
/// atom at zero makes the KS conservative), and the mass at zero within
/// `bands::RLRT_ZERO_MASS_SE` pooled two-proportion standard errors. Prints
/// the wall time of each case (unlocked, indicative).
#[test]
fn ks_null_matches_rlrsim() {
    const B: usize = 100_000;
    let (mut compared, mut worst_ks, mut worst_z) =
        (0, (0.0f64, String::new()), (0.0f64, String::new()));
    for (index, e) in entries().into_iter().enumerate() {
        let name = &e.name;
        let r = load(name);
        let Some(sim) = &r.rlrsim else { continue };
        let start = Instant::now();
        let (x, y, n, p, model, ids, _) = design(name);
        let fit = fit_cold(&x, &y, n, p, &model, &ids, &FitOptions::default());
        let ds = fit
            .design_summary
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: no design summary"));
        let d = rlrt_design(&ds.counts, &ds.s, &ds.c)
            .unwrap_or_else(|e| panic!("{name}: {}", e.reason()));
        let fitted = start.elapsed();
        let cfg = RlrtConfig {
            nsim: B,
            seed: 20261021 + index as u64,
            parallel_inner: true,
        };
        let null = rlrt_null(&d, &cfg);
        let total = start.elapsed();

        let b_ref = sim.nsim;
        assert_eq!(
            sim.null_n_zero + sim.null_pos.len(),
            b_ref,
            "{name}: frozen null size"
        );
        let ks = ks_distance(
            (null.n_zero, &null.draws_pos, B),
            (sim.null_n_zero, &sim.null_pos, b_ref),
        );
        let (bf, rf) = (B as f64, b_ref as f64);
        let ks_band = bands::RLRT_KS_COEF * ((bf + rf) / (bf * rf)).sqrt();
        let (p1, p2) = (null.n_zero as f64 / bf, sim.null_n_zero as f64 / rf);
        let pooled = (null.n_zero + sim.null_n_zero) as f64 / (bf + rf);
        let se = (pooled * (1.0 - pooled) * (1.0 / bf + 1.0 / rf)).sqrt();
        let z = (p1 - p2).abs() / se;
        eprintln!(
            "{name}: k = {}, seed {}, KS {ks:.4e} (band {ks_band:.4e}), mass at zero {p1:.5} \
             against {p2:.5} ({z:.2} SE), edge hits {}, fit {:.2} s, total {:.2} s",
            d.mu.len(),
            cfg.seed,
            null.edge_hits,
            fitted.as_secs_f64(),
            total.as_secs_f64()
        );
        assert!(ks <= ks_band, "{name}: KS {ks:e} > {ks_band:e}");
        assert!(
            z <= bands::RLRT_ZERO_MASS_SE,
            "{name}: mass at zero {p1} against {p2}, {z} standard errors"
        );
        if ks / ks_band >= worst_ks.0 {
            worst_ks = (ks / ks_band, name.clone());
        }
        if z >= worst_z.0 {
            worst_z = (z, name.clone());
        }
        compared += 1;
    }
    assert!(compared > 0, "no RLRsim null compared");
    eprintln!(
        "ks: {compared} references; worst KS share of the band {:.3} ({}), \
         worst mass-at-zero gap {:.2} SE ({})",
        worst_ks.0, worst_ks.1, worst_z.0, worst_z.1
    );
}

/// The OLS fit of an entry's `x` and `y` made with `reml = Some(true)`, with
/// the entry's prior weights: the REML log-likelihood at `θ = 0`.
fn ols_reml(x: &[f64], y: &[f64], n: usize, p: usize, weights: Option<Vec<f64>>) -> Fit {
    let model = ModelSpec {
        family: Family::Gaussian,
        re: None,
    };
    let opts = FitOptions {
        reml: Some(true),
        ..FitOptions::default()
    };
    fit_warm_wo(
        x,
        y,
        n,
        p,
        &model,
        &GroupIds::default(),
        None,
        weights.as_deref(),
        None,
        &opts,
    )
}

/// Gates 5 and 6, registry-backed, one driver over the quantity labels
/// `rlrt` (the observed statistic) and `p_value`. Per reference with an
/// RLRsim block, the lowered `(1 | g)` model and the OLS model made with
/// `reml = Some(true)`, both fitted with `fit_cold`, through `exact_rlrt` at
/// `B = 10⁵` and gate 4's seed (`20261021 +` the entry's index among the
/// manifest's `rlrt` entries):
///
/// - the statistic against RLRsim's within [`bands::RLRT_STAT_ABS`];
/// - the p-value against RLRsim's frozen p within
///   `bands::RLRT_P_SE·√(2p(1 − p)/B) + 1/(B + 1)`, `p` RLRsim's. Where that p
///   is 0 the band is `1/(B + 1)`, so the p-value must be exactly
///   `1/(B + 1)`: the statistic above every draw.
///
/// A difference over its band goes to the `oracle-tier` registry entry on
/// the reference and label, at `observed_rel = |Δ| / |RLRsim's value|`
/// (`+Inf` on a zero frozen p, which no entry covers): `Within` passes and
/// fires the entry, `Exceeds` and `NotCovered` fail. On `rlrt_dyestuff2`,
/// whose fit is pinned, the statistic is exactly 0 and the p-value exactly 1,
/// hard. Last, every entry under the two labels on a compared reference must
/// have fired.
#[test]
fn rlrt_observed_matches_rlrsim() {
    const B: usize = 100_000;
    const LABELS: [&str; 2] = ["rlrt", "p_value"];
    let scoped: Vec<DivEntry> = registry_entries()
        .into_iter()
        .filter(|e| e.quantities.iter().any(|q| LABELS.contains(&q.as_str())))
        .collect();
    let (mut compared, mut fired) = (BTreeSet::new(), BTreeSet::new());
    let mut adjudicate = |name: &str, quantity: &str, what: String, rel: f64| match covers(
        &scoped, name, quantity, rel,
    ) {
        Coverage::Within { id } => {
            eprintln!("documented divergence {id}: {name} {what} (rel {rel:.4e})");
            fired.insert(id);
        }
        Coverage::NotCovered => panic!("{name} {what} (rel {rel:.4e}): over its band"),
        Coverage::Exceeds { id, max_rel } => panic!(
            "{name} {what} (rel {rel:.4e}): documented divergence `{id}` covers only {max_rel:.1e}"
        ),
    };
    let bf = B as f64;
    let (mut worst_t, mut worst_p) = ((0.0f64, String::new()), (0.0f64, String::new()));
    // Where RLRsim's p is 0 the band is `1/(B + 1)`, which glmm's p meets
    // exactly, so those references are listed apart from the worst use.
    let mut exact_floor = Vec::new();
    for (index, e) in entries().into_iter().enumerate() {
        let name = &e.name;
        let r = load(name);
        let Some(sim) = &r.rlrsim else { continue };
        let (x, y, n, p, model, ids, weights) = design(name);
        assert!(weights.is_none(), "{name}: RLRsim has no weights");
        let full = fit_cold(&x, &y, n, p, &model, &ids, &FitOptions::default());
        assert!(
            full.converged() && full.reml,
            "{name}: a converged REML fit"
        );
        let reduced = ols_reml(&x, &y, n, p, None);
        assert!(reduced.reml, "{name}: the OLS fit is REML");
        let pinned = full.diagnostics.pinned.iter().flatten().any(|&b| b);
        let ds = full
            .design_summary
            .as_ref()
            .unwrap_or_else(|| panic!("{name}: no design summary"));
        let cfg = RlrtConfig {
            nsim: B,
            seed: 20261021 + index as u64,
            parallel_inner: true,
        };
        let res = exact_rlrt(
            &ds.counts,
            &ds.s,
            &ds.c,
            full.loglik,
            reduced.loglik,
            pinned,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("{name}: {}", e.reason()));

        let dt = (res.statistic - sim.statistic).abs();
        let p_ref = sim.p_value;
        let p_band =
            bands::RLRT_P_SE * (2.0 * p_ref * (1.0 - p_ref) / bf).sqrt() + 1.0 / (bf + 1.0);
        let dp = (res.p_value - p_ref).abs();
        eprintln!(
            "{name}: pinned {pinned}, statistic {:.10e} against RLRsim {:.10e} (|d| {dt:.3e}); \
             p {:.6e} against RLRsim {p_ref:.6e} (|d| {dp:.3e}, band {p_band:.3e}, {:.1}%), \
             nsim {}, seed {}, mass at zero {:.5}, k {}, edge hits {}",
            res.statistic,
            sim.statistic,
            res.p_value,
            100.0 * dp / p_band,
            res.nsim,
            res.seed,
            res.mass_at_zero,
            res.k_eigen,
            res.edge_hits
        );
        if name == "rlrt_dyestuff2" {
            assert!(pinned, "{name}: the fit pins its component");
            assert!(
                res.statistic == 0.0 && res.p_value == 1.0,
                "{name}: statistic {} and p {}, not exactly 0 and 1",
                res.statistic,
                res.p_value
            );
        }
        // A NaN difference goes to the registry, which never passes it.
        if dt.is_nan() || dt > bands::RLRT_STAT_ABS {
            let what = format!(
                "statistic {} against RLRsim {}",
                res.statistic, sim.statistic
            );
            adjudicate(name, "rlrt", what, dt / sim.statistic.abs());
        } else if dt >= worst_t.0 {
            worst_t = (dt, name.clone());
        }
        if dp.is_nan() || dp > p_band {
            let what = format!("p {} against RLRsim {p_ref}", res.p_value);
            adjudicate(name, "p_value", what, dp / p_ref);
        } else if p_ref == 0.0 {
            exact_floor.push(name.clone());
        } else if dp / p_band >= worst_p.0 {
            worst_p = (dp / p_band, name.clone());
        }
        compared.insert(name.clone());
    }
    assert!(!compared.is_empty(), "no RLRsim reference compared");
    eprintln!(
        "observed: {} references; worst |statistic - RLRsim| within its band {:.3e} ({}), \
         worst p within its band {:.1}% of the band ({}) over the references with a \
         nonzero RLRsim p, p = 1/(B + 1) against RLRsim's 0 on {exact_floor:?}; \
         fired {fired:?}",
        compared.len(),
        worst_t.0,
        worst_t.1,
        100.0 * worst_p.0,
        worst_p.1
    );
    let stale = stale_entries(&scoped, &compared, &fired);
    assert!(
        stale.is_empty(),
        "inference divergence registry is out of date: these entries on a compared \
         reference no longer fire and must be deleted: {stale:?}"
    );
}

/// Gate 5's reduced log-likelihood and gate 11's R part, hard. Per `rlrt`
/// reference, the weighted `rlrt_reml0_weighted` included: `Fit::loglik` of
/// the OLS fit made with `reml = Some(true)` (with the entry's weights)
/// against R's frozen `logLik(lm, REML = TRUE)` within
/// [`bands::REML0_LOGLIK_ABS`].
#[test]
fn ols_reml_loglik_matches_lm_reml() {
    let (mut compared, mut worst) = (0, (0.0f64, String::new()));
    for e in entries() {
        let name = &e.name;
        let r = load(name);
        let (x, y, n, p, _, _, weights) = design(name);
        let weighted = weights.is_some();
        let fit = ols_reml(&x, &y, n, p, weights);
        assert!(fit.reml, "{name}: the OLS fit is REML");
        let d = (fit.loglik - r.loglik_lm_reml).abs();
        eprintln!(
            "{name}: weighted {weighted}, loglik {:.12} against R {:.12} (|d| {d:.3e})",
            fit.loglik, r.loglik_lm_reml
        );
        assert!(
            d <= bands::REML0_LOGLIK_ABS,
            "{name}: loglik {} against R's {} (|d| {d:e})",
            fit.loglik,
            r.loglik_lm_reml
        );
        if d >= worst.0 {
            worst = (d, name.clone());
        }
        compared += 1;
    }
    assert!(compared > 0, "no rlrt reference compared");
    eprintln!(
        "reml0: {compared} references; worst |d| {:.3e} ({})",
        worst.0, worst.1
    );
}
