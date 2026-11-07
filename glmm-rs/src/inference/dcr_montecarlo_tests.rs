//! Spec 2-dcr's Monte Carlo gates (`montecarlo-tests`) on the weighted cells
//! of [`DCR_TYPE1_CELLS`] and [`DCR_LRT_CELLS`]: every dataset drawn under the
//! null of `x` ([`DcrDesign::null_y`]) and fitted through the ports' default
//! calls, [`run_fit`] and [`run_compare`], with no `weight_power` and no
//! `se_type`.
//!
//! - G11 (`dcr_type1_of_x`): the fit `y ~ x` (`+ (1 | g)`, or `+ (1 + x | g)`
//!   on the `S` cells) and its p-value of `x` (`coef_p`, on the SE and df the
//!   defaults resolve to, D22). Its rejection rate at α = .05 within
//!   `.05 ± (MC_ASYM_REL·.05 + MC_LEVEL_K·SE)` (±.034 at R = 2000) and at most
//!   .075, per cell.
//! - G13 (`dcr_lrt_null`): `compare(y ~ 1, y ~ x)` (with `(1 | g)` on the
//!   grouped cells), whose default fits δ by ML in both models (D16), and its
//!   likelihood-ratio p-value. The rate at α = .05 at most .09 per cell, and
//!   the mean of the 10 cells' rates at most .07: the test is somewhat
//!   liberal where the noise is not a power of the weights (the `floor_mid`
//!   cells), and the comparison says so ([`crate::inference::LrNote::EstimatedWeightPower`]).
//!
//! `MC_R` sets the datasets per cell (default `mc_r`, 2000); at any other `R`
//! the rates are printed and nothing is gated. `MC_THREADS` splits each
//! cell's datasets across threads (default the available parallelism; the
//! counts do not depend on it). `MC_DCR_CELLS` (comma-separated ids) measures
//! and gates only the cells named; G13's mean is gated only on all 10.
//! `MC_DCR_LRT_ARM` (`shared` or `claimed`, see [`lrt_null`]) measures G13 on
//! another route and gates nothing. A
//! dataset whose fit errs, does not converge or gives a p-value that is not
//! finite is failed and left out; a cell that loses more than
//! [`bands::MC_FAIL_MAX`] of its datasets fails.

use super::mc_calibration_tests::env_count;
use super::mc_data::{bands, DcrDesign, DCR_LRT_CELLS, DCR_TYPE1_CELLS};
use crate::formula::anova::ComparisonTest;
use crate::orchestrate::{run_compare, run_fit, FitResult};
use crate::{SeType, WeightPower};
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

const ALPHA: f64 = 0.05;

/// G11's per-cell ceiling, Bradley's liberal limit at α = .05.
const CEILING: f64 = 0.075;

/// G13's per-cell ceiling and ceiling on the mean of its 10 cells.
const LRT_CEILING: f64 = 0.09;
const LRT_MEAN_CEILING: f64 = 0.07;

/// One dataset: whether `x` was rejected and the arm the defaults resolved
/// to, or why it failed.
type Outcome = Result<(bool, String), String>;

/// One cell measured.
struct Measured {
    id: &'static str,
    r_count: u64,
    kept: usize,
    rejected: usize,
    /// The failed datasets, with the first one's index and reason.
    failed: usize,
    first_failure: Option<(u64, String)>,
    /// Datasets per arm.
    arms: BTreeMap<String, usize>,
    wall: f64,
}

impl Measured {
    fn rate(&self) -> f64 {
        self.rejected as f64 / self.kept as f64
    }

    fn failed_fraction(&self) -> f64 {
        self.failed as f64 / self.r_count as f64
    }
}

/// The fitted random part: none on a WLS cell, a random slope on `x` on the
/// `S` cells only (the `E` cells' slope is the one their fit leaves out).
fn random_part(d: &DcrDesign) -> &'static str {
    match (d.groups, d.id.starts_with("S/")) {
        (0, _) => "",
        (_, true) => " + (1 + x | g)",
        _ => " + (1 | g)",
    }
}

type Numeric = HashMap<String, Vec<f64>>;
type Factors = HashMap<String, (Vec<String>, Vec<u32>)>;

/// The columns of a dataset: `y`, `x`, and the factor `g` on a grouped cell.
fn columns(d: &DcrDesign, y: Vec<f64>) -> (Numeric, Factors) {
    let numeric = HashMap::from([("y".to_string(), y), ("x".to_string(), d.x.clone())]);
    let mut factors = HashMap::new();
    if d.groups > 0 {
        let levels = (1..=d.groups).map(|k| k.to_string()).collect();
        factors.insert("g".to_string(), (levels, d.g.clone()));
    }
    (numeric, factors)
}

/// The power, criterion and df method a fit ran with.
fn arm(f: &FitResult) -> String {
    let power = if f.weight_power_estimated {
        "delta".to_string()
    } else {
        format!("w^{}", f.weight_power)
    };
    let criterion = if f.reml { "REML" } else { "ML" };
    format!("{power}/{criterion}/{}", f.df_method)
}

/// G11 on dataset `r` of `d`.
fn type1_of_x(d: &DcrDesign, r: u64) -> Outcome {
    let (numeric, factors) = columns(d, d.null_y(r));
    let formula = format!("y ~ x{}", random_part(d));
    let f = run_fit(
        &formula,
        numeric,
        factors,
        "gaussian",
        "identity",
        "hessian",
        1,
        None,
        Some(d.w.clone()),
        None,
        None,
        true,
        false,
        None,
        "auto",
        None,
        None,
    )
    .map_err(|e| e.to_string())?;
    let j = f.names.iter().position(|n| n == "x").expect("a column x");
    let p = f.coef_p[j];
    if !f.converged || !p.is_finite() {
        return Err(format!("converged = {}, p = {p}", f.converged));
    }
    Ok((p < ALPHA, arm(&f)))
}

/// G13 on dataset `r` of `d`. With `MC_DCR_LRT_ARM=shared` (an experiment,
/// not the product's route): δ̂ by REML on the larger model, then both sides
/// compared at that fixed power, so a WLS pair takes the F test;
/// `MC_DCR_LRT_ARM=claimed` compares at the claimed weights (power 1).
fn lrt_null(d: &DcrDesign, r: u64) -> Outcome {
    let (numeric, factors) = columns(d, d.null_y(r));
    let re = random_part(d);
    let arm_env = std::env::var("MC_DCR_LRT_ARM");
    let power = if arm_env.as_deref() == Ok("claimed") {
        Some(WeightPower::Fixed(1.0))
    } else if arm_env.as_deref() == Ok("shared") {
        let f = run_fit(
            &format!("y ~ x{re}"),
            numeric.clone(),
            factors.clone(),
            "gaussian",
            "identity",
            "hessian",
            1,
            None,
            Some(d.w.clone()),
            None,
            None,
            false,
            false,
            None,
            "auto",
            None,
            Some(SeType::Model),
        )
        .map_err(|e| e.to_string())?;
        Some(WeightPower::Fixed(f.weight_power))
    } else {
        None
    };
    let c = run_compare(
        &format!("y ~ 1{re}"),
        &format!("y ~ x{re}"),
        numeric,
        factors,
        "gaussian",
        "identity",
        1,
        Some(d.w.clone()),
        None,
        None,
        None,
        power,
        None,
    )
    .map_err(|e| e.to_string())?;
    let p_value = match &c.comparison.test {
        ComparisonTest::Lr(t) => t.p_value,
        ComparisonTest::Linear(t) if power.is_some() => t.p_value,
        _ => panic!(
            "{}: dataset {r} is not compared by a likelihood-ratio test",
            d.id
        ),
    };
    if !p_value.is_finite() {
        return Err(format!("p = {p_value}"));
    }
    let label = match power {
        Some(_) => match c.comparison.test {
            ComparisonTest::Lr(_) => "fixed power, LR test".to_string(),
            _ => "fixed power, F test".to_string(),
        },
        None => format!("{} vs {}", arm(&c.fit0), arm(&c.fit1)),
    };
    Ok((p_value < ALPHA, label))
}

/// Datasets `0..r_count` of cell `id` on `threads` threads, each taking the
/// next dataset not yet taken; counted in dataset order.
fn measure(
    id: &'static str,
    r_count: u64,
    threads: u64,
    one: fn(&DcrDesign, u64) -> Outcome,
) -> Measured {
    let d = DcrDesign::load(id);
    let start = Instant::now();
    let next = AtomicU64::new(0);
    let mut done: Vec<(u64, Outcome)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let r = next.fetch_add(1, Ordering::Relaxed);
                        if r >= r_count {
                            break out;
                        }
                        out.push((r, one(&d, r)));
                    }
                })
            })
            .collect();
        workers
            .into_iter()
            .flat_map(|w| w.join().expect("a worker thread"))
            .collect()
    });
    done.sort_by_key(|(r, _)| *r);
    let mut m = Measured {
        id,
        r_count,
        kept: 0,
        rejected: 0,
        failed: 0,
        first_failure: None,
        arms: BTreeMap::new(),
        wall: start.elapsed().as_secs_f64(),
    };
    for (r, outcome) in done {
        match outcome {
            Ok((reject, arm)) => {
                m.kept += 1;
                m.rejected += usize::from(reject);
                *m.arms.entry(arm).or_default() += 1;
            }
            Err(why) => {
                m.failed += 1;
                m.first_failure.get_or_insert((r, why));
            }
        }
    }
    m
}

/// The cells of `cells` named in `MC_DCR_CELLS`, or all of them; and whether
/// that is all of them.
fn selected(cells: &'static [&'static str]) -> (Vec<&'static str>, bool) {
    let Ok(list) = std::env::var("MC_DCR_CELLS") else {
        return (cells.to_vec(), true);
    };
    let names: Vec<&str> = list.split(',').map(str::trim).collect();
    for n in &names {
        assert!(
            DCR_TYPE1_CELLS.contains(n) || DCR_LRT_CELLS.contains(n),
            "MC_DCR_CELLS: no cell {n}"
        );
    }
    let chosen: Vec<_> = cells
        .iter()
        .filter(|c| names.contains(c))
        .copied()
        .collect();
    let all = chosen.len() == cells.len();
    (chosen, all)
}

/// Measures the selected cells of `cells`, printing one line per cell, and
/// returns them with whether the run is gated (`R = mc_r`) and whole.
fn run(
    gate: &str,
    cells: &'static [&'static str],
    one: fn(&DcrDesign, u64) -> Outcome,
) -> (Vec<Measured>, bool, bool) {
    let r_count = env_count("MC_R").unwrap_or(bands::MC_R as u64);
    let threads = env_count("MC_THREADS")
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get() as u64));
    let (chosen, all) = selected(cells);
    let gated = r_count == bands::MC_R as u64;
    println!(
        "DCR {gate}: {} of {} cells, R={r_count}, threads={threads}{}",
        chosen.len(),
        cells.len(),
        if gated {
            ""
        } else {
            "; not gated (the bands are built for R = mc_r = 2000)"
        }
    );
    let measured: Vec<Measured> = chosen
        .into_iter()
        .map(|id| {
            let m = measure(id, r_count, threads, one);
            let arms: Vec<String> = m.arms.iter().map(|(a, k)| format!("{a} x{k}")).collect();
            println!(
                "DCR {gate} {id} rate={:.4} kept={} failed={} wall={:.1}s ms_per_dataset_core={:.1} arm={}",
                m.rate(),
                m.kept,
                m.failed,
                m.wall,
                1000.0 * m.wall * threads as f64 / r_count as f64,
                arms.join("; ")
            );
            if let Some((r, why)) = &m.first_failure {
                println!("DCR {gate} {id} first failed dataset {r}: {why}");
            }
            m
        })
        .collect();
    (measured, gated, all)
}

/// The checks every gated cell shares: the failed fraction and the ceiling.
fn cell_misses(gate: &str, m: &Measured, ceiling: f64) -> Vec<String> {
    let mut misses = Vec::new();
    if m.failed_fraction() > bands::MC_FAIL_MAX {
        misses.push(format!(
            "{gate} {}: {} of {} datasets failed, above mc_fail_max {}",
            m.id,
            m.failed,
            m.r_count,
            bands::MC_FAIL_MAX
        ));
    }
    if m.rate() > ceiling {
        misses.push(format!(
            "{gate} {}: rate {:.4} above {ceiling}",
            m.id,
            m.rate()
        ));
    }
    misses
}

/// G11, the Type I of `x` of the default weighted fit on the 16 cells of
/// spec 2-dcr §10.6.
#[test]
fn dcr_type1_of_x() {
    let (measured, gated, _) = run("G11", &DCR_TYPE1_CELLS, type1_of_x);
    if !gated {
        return;
    }
    let mut misses = Vec::new();
    for m in &measured {
        misses.extend(cell_misses("G11", m, CEILING));
        let se = (ALPHA * (1.0 - ALPHA) / m.kept as f64).sqrt();
        let half = bands::MC_ASYM_REL * ALPHA + bands::MC_LEVEL_K * se;
        if (m.rate() - ALPHA).abs() > half {
            misses.push(format!(
                "G11 {}: rate {:.4} outside {ALPHA} ± {half:.4}",
                m.id,
                m.rate()
            ));
        }
    }
    assert!(misses.is_empty(), "{}", misses.join("\n"));
}

/// G13, the null Type I of `compare()`'s likelihood-ratio test with δ fitted
/// by ML in both models, on the 10 cells of spec 2-dcr §10.7.
#[test]
fn dcr_lrt_null() {
    let (measured, gated, all) = run("G13", &DCR_LRT_CELLS, lrt_null);
    let mean = measured.iter().map(Measured::rate).sum::<f64>() / measured.len() as f64;
    println!("DCR G13 mean rate={mean:.4} ceiling={LRT_MEAN_CEILING}");
    if !gated || std::env::var_os("MC_DCR_LRT_ARM").is_some() {
        return;
    }
    let mut misses: Vec<String> = measured
        .iter()
        .flat_map(|m| cell_misses("G13", m, LRT_CEILING))
        .collect();
    if all && mean > LRT_MEAN_CEILING {
        misses.push(format!("G13: mean rate {mean:.4} above {LRT_MEAN_CEILING}"));
    }
    if !all {
        println!("DCR G13: SUBSET (MC_DCR_CELLS), the mean is not gated");
    }
    assert!(misses.is_empty(), "{}", misses.join("\n"));
}
