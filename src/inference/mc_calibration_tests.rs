//! The Monte Carlo calibration tier's driver and gates (`montecarlo-tests`):
//! per cell of [`CELLS`] and per dataset, every quantity of the tier against
//! the generating truth, accumulated per cell, printed, and written as one
//! JSON line per cell when `MC_TIER_OUT` names a file; then, in
//! `calibration_gates`, the frozen gated set ([`bands::GATED`]) asserted on
//! those counts, every other (cell, quantity) reported.
//!
//! Every quantity comes through the surfaces a user reaches: fits made with
//! `fit_cold` from `formula::lower` with the lowering's own options (`nagq`
//! and `reml` set), `formula::anova::{compare, term_tests}`,
//! `Fit::{contrast_t, contrast_f, coef_df}`, and the tails and quantiles the
//! ports' `summary()` and `confint` use.
//!
//! Aliasing by design. Per dataset the full fixed design (`1`, `time`,
//! `group`, `time:group`) is tested for aliased columns from its lowered `x`
//! with the test `fit_cold` applies before any fit: `group` constant, or
//! `time:group` aliased because one group's rows share one time. Such a
//! dataset is left out of every quantity, `f3`'s included, since every fit of
//! the dataset uses that design or a part of it and a dropped column moves the
//! other coefficients' targets: it is counted as aliased, never as failed, and
//! no fit is made. A level of `f3` absent keeps its zero column aliased and the
//! `f3` row is tested at the df it carries (1 instead of 2); with every `f3`
//! column aliased (a 0-df row) the `f3` quantities count as aliased. A
//! quantity fails when a fit it needs did not converge, or when its value is
//! not finite; `R_kept` is the datasets left.

use super::mc_data::bands::{Gate, Side};
use super::mc_data::{bands, generate, Cell, Response, Truth, CELLS};
use super::tails::{chisq_upper_p, normal_quantile, t_quantile};
use crate::formula::anova::{
    compare, term_tests, CompareOptions, ComparisonTest, Model, SsType, TermTest, TermTestOptions,
};
use crate::formula::{lower, Column, Lowered, Table};
use crate::inference::{LinearKind, Reference};
use crate::{fit_cold, BinomialLink, Family, Fit, PoissonLink};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// The nominal levels of the tests.
const ALPHAS: [f64; 3] = [0.01, 0.05, 0.10];

/// The nominal levels of the intervals.
const LEVELS: [f64; 3] = [0.90, 0.95, 0.99];

/// The options of every comparison. No pair of the tier is an exact
/// restricted likelihood-ratio case (a Gaussian mixed fit against OLS), so no
/// draw is made.
const COMPARE: CompareOptions = CompareOptions {
    nsim: 10_000,
    seed: 20_261_100,
};

/// The full fixed part.
const FULL: &str = "time + group + time:group";
/// `time:group` dropped: the LRT of `time:group`, and the larger model of
/// `group`'s Type II LRT.
const DROP_TG: &str = "time + group";
/// The `group` column dropped: `group`'s Type III LRT.
const DROP_GROUP: &str = "time + time:group";
/// The smaller model of the 2-df LRT and of `group`'s Type II LRT.
const TIME: &str = "time";
/// The full fixed part with the null factor `f3`.
const WITH_F3: &str = "time + group + time:group + f3";
/// Per coefficient `β_j`, the fixed part of the fit that holds it at its
/// truth through the offset `off_j`.
const OFFSET: [&str; 4] = [
    "0 + time + group + time:group + offset(off0)",
    "group + time:group + offset(off1)",
    "time + time:group + offset(off2)",
    "time + group + offset(off3)",
];

const WALD_CI: [&str; 4] = ["wald_ci_b0", "wald_ci_b1", "wald_ci_b2", "wald_ci_b3"];
const PROFILE_CI: [&str; 4] = [
    "profile_ci_b0",
    "profile_ci_b1",
    "profile_ci_b2",
    "profile_ci_b3",
];
const GLM_WALD_CI: [&str; 4] = [
    "glm_wald_ci_b0",
    "glm_wald_ci_b1",
    "glm_wald_ci_b2",
    "glm_wald_ci_b3",
];
const SATT_CI: [&str; 4] = ["satt_ci_b0", "satt_ci_b1", "satt_ci_b2", "satt_ci_b3"];

/// The `f3` quantities.
const F3: [&str; 3] = ["wald_f3", "lrt_f3", "satt_f_f3"];

/// The nulls both Wald and the LRT measure on a GLMM cell, for the paired
/// decision counts of the pre-declared comparison: the pair, its Wald
/// quantity, its LRT quantity.
const PAIRS: [(&str, &str, &str); 5] = [
    ("tg", "wald_tg", "lrt_tg"),
    ("tg00", "wald_tg00", "lrt_tg00"),
    ("group3", "wald3_group", "lrt3_group"),
    ("group2", "wald2_group", "lrt2_group"),
    ("f3", "wald_f3", "lrt_f3"),
];

/// The order of a pair's four counts at one α.
const PAIR_COLUMNS: [&str; 4] = ["both", "wald_only", "lrt_only", "neither"];

/// The tests of a cell, by the names [`bands::GATED`] gates them under.
fn tests_of(cell: &Cell) -> Vec<&'static str> {
    let mix = if cell.slope {
        "mix_slope_power"
    } else {
        "mix_slope"
    };
    if cell.response == Response::Gauss {
        return vec![
            "satt_t_tg",
            "satt_f_group",
            "satt_f_tg00",
            "satt_f_f3",
            "satt_f_joint",
            "ml_lrt_2df",
            "satt_t_tg_power",
            mix,
        ];
    }
    let mut tests = vec![
        "wald_tg",
        "lrt_tg",
        "wald3_group",
        "lrt3_group",
        "wald2_group",
        "lrt2_group",
        "wald_tg00",
        "lrt_tg00",
        "wald_f3",
        "lrt_f3",
        "lrt_2df",
        "wald_tg_power",
        "lrt_2df_power",
        mix,
    ];
    if !cell.slope {
        tests.extend([
            "mix_intercept",
            "agq_wald_tg",
            "agq_lrt_tg",
            "glm_wald_tg",
            "glm_lrt_tg",
        ]);
    }
    tests
}

/// The intervals of a cell.
fn covers_of(cell: &Cell) -> Vec<&'static str> {
    if cell.response == Response::Gauss {
        return SATT_CI.to_vec();
    }
    let mut covers = [WALD_CI, PROFILE_CI].concat();
    if !cell.slope {
        covers.extend(GLM_WALD_CI);
    }
    covers
}

/// The family of a response level.
fn family(response: Response) -> Family {
    match response {
        Response::Bern54 | Response::Bern22 | Response::Bern10 => Family::Binomial {
            link: BinomialLink::Logit,
        },
        Response::Pois165 | Response::Pois034 => Family::Poisson {
            link: PoissonLink::Log,
        },
        Response::Gauss => Family::Gaussian,
    }
}

/// The random part of a cell (`slope`), or of the other design (`!slope`).
fn random_part(slope: bool) -> &'static str {
    if slope {
        "(1 + time | id)"
    } else {
        "(1 | id)"
    }
}

/// `y ~ fixed + random`, or `y ~ fixed` with no random part.
fn formula(y: &str, fixed: &str, random: &str) -> String {
    if random.is_empty() {
        format!("{y} ~ {fixed}")
    } else {
        format!("{y} ~ {fixed} + {random}")
    }
}

/// The design's expected fraction of datasets aliased by design (an upper
/// bound at `K = 20`), from the generator's design rule.
fn expected_aliased(cell: &Cell) -> f64 {
    match (cell.k, cell.m) {
        (10, 3) => 0.026,
        (10, 9) => 0.011,
        (10, _) => 0.009,
        (20, _) => 0.0004,
        _ => 0.0,
    }
}

fn lowered(formula: &str, table: &Table, family: Family) -> Lowered {
    lower(formula, table, family).unwrap_or_else(|e| panic!("{formula}: {e}"))
}

/// The design columns `fit_cold` drops as aliased: its own alias gate
/// (`fit::detect_aliased`) on the lowered `x`.
fn aliased_columns(lo: &Lowered) -> Vec<bool> {
    crate::fit::detect_aliased(&lo.x, lo.n, lo.p)
}

/// Whether the full fixed design of `table` has an aliased column.
fn design_aliased(table: &Table, family: Family, random: &str) -> bool {
    let lo = lowered(&formula("y_full", FULL, random), table, family);
    aliased_columns(&lo).contains(&true)
}

/// How many `f3` columns of the full design with `f3` are aliased. Mostly an
/// absent level (1 column, a 1-df row; 2 columns, a 0-df row); rarely an
/// `f3` level set that equals `group`, where one column of that design drops
/// (an `f3` column, or `time:group`, which the lowering orders after `f3`).
fn f3_aliased_columns(table: &Table, family: Family, random: &str) -> usize {
    let lo = lowered(&formula("y_00", WITH_F3, random), table, family);
    let mask = aliased_columns(&lo);
    let term = lo
        .terms
        .iter()
        .find(|t| t.label == "f3")
        .expect("the f3 term");
    term.columns.clone().filter(|&j| mask[j]).count()
}

/// A converged fit with its lowering and formula.
struct Fitted {
    formula: String,
    lo: Lowered,
    fit: Fit,
}

/// One fit of `formula` on `table`: the lowering's options with `nagq` and
/// `reml` set, timed under `kind`; `None` when it did not converge.
fn fit_one(
    acc: &mut CellAcc,
    kind: &'static str,
    table: &Table,
    formula: String,
    family: Family,
    nagq: u8,
    reml: Option<bool>,
) -> Option<Fitted> {
    let mut lo = lowered(&formula, table, family);
    lo.opts.nagq = nagq;
    lo.opts.reml = reml;
    let start = Instant::now();
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    let wall = start.elapsed().as_secs_f64();
    let converged = fit.converged();
    let e = acc.fits.entry(kind).or_default();
    e.0 += 1;
    e.1 += u64::from(!converged);
    e.2 += wall;
    converged.then_some(Fitted { formula, lo, fit })
}

/// The design column named `name`.
fn col(lo: &Lowered, name: &str) -> usize {
    lo.col_names
        .iter()
        .position(|c| c == name)
        .unwrap_or_else(|| panic!("no column {name}"))
}

/// The Wald z test of one column, `p = chisq_upper_p(z², 1)` (the Type III
/// Wald χ² row of a 1-df term).
fn wald_p(f: &Option<Fitted>, name: &str) -> Option<f64> {
    let f = f.as_ref()?;
    let j = col(&f.lo, name);
    let z = f.fit.beta[j] / f.fit.se[j];
    z.is_finite().then(|| chisq_upper_p(z * z, 1.0))
}

/// The p-value of a term's row of `term_tests` (a Wald χ² on a GLMM, an F on
/// Satterthwaite's df on a Gaussian LMM), at the df the row carries.
fn term_p(f: &Option<Fitted>, ss_type: SsType, label: &str) -> Option<f64> {
    let f = f.as_ref()?;
    // The test is named, never left to the default, so a size rule behind
    // `Auto` cannot turn a measured Wald row into a likelihood-ratio row.
    // The Gaussian LMM's Satterthwaite F rows are its only test, which
    // `term_tests` runs under `Auto` alone.
    let test = if f.lo.model.family == Family::Gaussian {
        TermTest::Auto
    } else {
        TermTest::Wald
    };
    let opts = TermTestOptions { ss_type, test };
    let table = term_tests(&f.lo, &f.fit, &opts).ok()?;
    // The rows must be the test measured: the F on the Gaussian LMM, the Wald
    // χ² elsewhere. Any other table loses the quantity.
    let kind = match test {
        TermTest::Auto => LinearKind::F,
        _ => LinearKind::ChiSquared,
    };
    if table.choice.test != kind {
        return None;
    }
    let row = table.fixed.iter().find(|r| r.label == label)?;
    Some(row.test.p_value)
}

/// The likelihood-ratio test of two fits through `compare`.
fn lrt_p(small: &Option<Fitted>, large: &Option<Fitted>) -> Option<f64> {
    let (a, b) = (small.as_ref()?, large.as_ref()?);
    let ma = Model::from_fit(&a.formula, &a.lo, &a.fit);
    let mb = Model::from_fit(&b.formula, &b.lo, &b.fit);
    match compare(&ma, &mb, &COMPARE).ok()?.test {
        ComparisonTest::Lr(t) => Some(t.p_value),
        ComparisonTest::Linear(t) => Some(t.p_value),
    }
}

/// The Satterthwaite t test of one column (`contrast_t` on the unit vector).
fn satt_t_p(f: &Option<Fitted>, name: &str) -> Option<f64> {
    let f = f.as_ref()?;
    let mut l = vec![0.0; f.lo.p];
    l[col(&f.lo, name)] = 1.0;
    f.fit.contrast_t(&l).ok().map(|c| c.p_value)
}

/// The Satterthwaite F of `group` and `time:group` jointly (`contrast_f`
/// with two rows).
fn satt_joint_p(f: &Option<Fitted>) -> Option<f64> {
    let f = f.as_ref()?;
    let p = f.lo.p;
    let mut l = vec![0.0; 2 * p];
    l[col(&f.lo, "group")] = 1.0;
    l[p + col(&f.lo, "time:group")] = 1.0;
    f.fit.contrast_f(&l, 2).ok().map(|c| c.p_value)
}

/// How `β̂_j ± q·SE` covers `truth` at each of [`LEVELS`], `q` the normal
/// quantile, or with `t_df` the t quantile on `coef_df[j]`, as `confint`
/// builds it: the margin `q·SE − |β̂_j − truth|`, which covers when `≥ 0`.
fn interval_covers(f: &Option<Fitted>, j: usize, truth: f64, t_df: bool) -> Option<[f64; 3]> {
    let f = f.as_ref()?;
    let (b, se) = (f.fit.beta[j], f.fit.se[j]);
    let mut out = [0.0; 3];
    for (o, level) in out.iter_mut().zip(LEVELS) {
        let p = 0.5 + level / 2.0;
        let q = if t_df {
            t_quantile(p, f.fit.coef_df[j])
        } else {
            normal_quantile(p)
        };
        let margin = q * se - (b - truth).abs();
        if !margin.is_finite() {
            return None;
        }
        *o = margin;
    }
    Some(out)
}

/// How the profile-likelihood interval of `β_j` covers its truth at each of
/// [`LEVELS`], by duality: it covers exactly when the LRT of
/// `β_j = β_j,true`, from the full fit's and the offset fit's
/// log-likelihoods, does not reject at `1 − level` (the interval is
/// `{b : 2Δℓ(b) ≤ χ²₁,level}`). The margin is `p − (1 − level)`, which
/// covers when `≥ 0` (a p-value exactly at α, where the test rejects, has
/// probability zero). A negative statistic (the full fit short of its
/// optimum) is floored at 0.
fn profile_covers(loglik_full: f64, loglik_offset: f64) -> Option<[f64; 3]> {
    let raw = 2.0 * (loglik_full - loglik_offset);
    if !raw.is_finite() {
        return None;
    }
    let p = chisq_upper_p(raw.max(0.0), 1.0);
    Some(LEVELS.map(|level| p - (1.0 - level)))
}

/// The column `name` of `table`.
fn numeric<'a>(table: &'a Table, name: &str) -> &'a [f64] {
    match table.columns.iter().find(|(n, _)| n == name) {
        Some((_, Column::Numeric(v))) => v,
        _ => panic!("no numeric column {name}"),
    }
}

/// One dataset's measurements, before they are counted.
#[derive(Default)]
struct Measured {
    /// The full fixed design is aliased: nothing is measured.
    aliased: bool,
    /// Every `f3` column is aliased: the `f3` quantities are not measured.
    f3_aliased: bool,
    /// Per test, its p-value; `None` when a fit it needs did not converge or
    /// `compare` or `term_tests` refused.
    p: BTreeMap<&'static str, Option<f64>>,
    /// Per interval, its margin at each level (it covers the truth when the
    /// margin is `≥ 0`); `None` when a fit it needs did not converge or a
    /// bound is not finite.
    covers: BTreeMap<&'static str, Option<[f64; 3]>>,
    /// The intervals not measured on this design (the `β1` profile on a
    /// random-slope cell, whose slope variable must stay in the design).
    skipped: Vec<&'static str>,
}

/// Measures `table`, a dataset of `cell`; the data measures, the estimates
/// and the fits' walls go to `acc`.
fn measure(cell: &Cell, table: &Table, acc: &mut CellAcc) -> Measured {
    let fam = family(cell.response);
    let random = random_part(cell.slope);
    let (n, k) = (table.n as f64, cell.k as f64);
    acc.add_measure("n_obs", n);
    acc.add_measure("cluster_size", n / k);
    if cell.response != Response::Gauss {
        let events: f64 = numeric(table, "y_0").iter().sum();
        let minority = match fam {
            Family::Binomial { .. } => events.min(n - events),
            _ => events,
        };
        acc.add_measure("events", events);
        acc.add_measure("events_per_cluster", events / k);
        acc.add_measure("prevalence", events / n);
        acc.add_measure("minority", minority);
        acc.add_measure("minority_per_cluster", minority / k);
        if !cell.slope {
            // The GLM rows' own response, `y_0f`, without random effects.
            let events: f64 = numeric(table, "y_0f").iter().sum();
            let minority = match fam {
                Family::Binomial { .. } => events.min(n - events),
                _ => events,
            };
            acc.add_measure("glm_events", events);
            acc.add_measure("glm_events_per_cluster", events / k);
            acc.add_measure("glm_prevalence", events / n);
            acc.add_measure("glm_minority", minority);
            acc.add_measure("glm_minority_per_cluster", minority / k);
        }
    }
    // The share of datasets whose full design with `f3` drops one column,
    // mostly an absent `f3` level (a 1-df `f3` row).
    let f3_dropped = f3_aliased_columns(table, fam, random);
    acc.add_measure("f3_df1", if f3_dropped == 1 { 1.0 } else { 0.0 });
    let mut m = Measured::default();
    if design_aliased(table, fam, random) {
        m.aliased = true;
        return m;
    }
    m.f3_aliased = f3_dropped >= 2;
    if cell.response == Response::Gauss {
        measure_lmm(cell, table, acc, &mut m);
    } else {
        measure_glmm(cell, table, acc, &mut m);
    }
    m
}

/// The GLMM cell's quantities (Laplace; the AGQ arm, the boundary test and
/// the GLM rows on the `(1 | id)` cells).
fn measure_glmm(cell: &Cell, table: &Table, acc: &mut CellAcc, m: &mut Measured) {
    let fam = family(cell.response);
    let truth = Truth::of(cell);
    let (re, alt) = (random_part(cell.slope), random_part(!cell.slope));
    let fit = |acc: &mut CellAcc, kind, y: &str, fixed: &str, random: &str, nagq| {
        fit_one(acc, kind, table, formula(y, fixed, random), fam, nagq, None)
    };
    let full = fit(acc, "glmm_full", "y_full", FULL, re, 1);
    let time = fit(acc, "glmm_time", "y_full", TIME, re, 1);
    let full_0 = fit(acc, "glmm_full", "y_0", FULL, re, 1);
    let drop_tg_0 = fit(acc, "glmm_drop", "y_0", DROP_TG, re, 1);
    let full_00 = fit(acc, "glmm_full", "y_00", FULL, re, 1);
    let drop_group_00 = fit(acc, "glmm_drop", "y_00", DROP_GROUP, re, 1);
    let drop_tg_00 = fit(acc, "glmm_drop", "y_00", DROP_TG, re, 1);
    let time_00 = fit(acc, "glmm_time", "y_00", TIME, re, 1);
    let alt_full = fit(acc, "glmm_alt_random", "y_full", FULL, alt, 1);
    m.p.insert("wald_tg", wald_p(&full_0, "time:group"));
    m.p.insert("lrt_tg", lrt_p(&drop_tg_0, &full_0));
    m.p.insert("wald3_group", wald_p(&full_00, "group"));
    m.p.insert("lrt3_group", lrt_p(&drop_group_00, &full_00));
    m.p.insert("wald2_group", term_p(&full_00, SsType::Type2, "group"));
    m.p.insert("lrt2_group", lrt_p(&time_00, &drop_tg_00));
    m.p.insert("wald_tg00", wald_p(&full_00, "time:group"));
    m.p.insert("lrt_tg00", lrt_p(&drop_tg_00, &full_00));
    m.p.insert("lrt_2df", lrt_p(&time_00, &full_00));
    m.p.insert("wald_tg_power", wald_p(&full, "time:group"));
    m.p.insert("lrt_2df_power", lrt_p(&time, &full));
    if cell.slope {
        m.p.insert("mix_slope_power", lrt_p(&alt_full, &full));
    } else {
        m.p.insert("mix_slope", lrt_p(&full, &alt_full));
    }
    if !m.f3_aliased {
        let with_f3 = fit(acc, "glmm_f3", "y_00", WITH_F3, re, 1);
        m.p.insert("wald_f3", term_p(&with_f3, SsType::Type3, "f3"));
        m.p.insert("lrt_f3", lrt_p(&full_00, &with_f3));
    }
    for (j, (wald, profile)) in WALD_CI.into_iter().zip(PROFILE_CI).enumerate() {
        m.covers
            .insert(wald, interval_covers(&full, j, truth.beta[j], false));
        if cell.slope && j == 1 {
            m.skipped.push(profile);
            continue;
        }
        let held = fit(acc, "glmm_offset", "y_full", OFFSET[j], re, 1);
        let covers = match (&full, &held) {
            (Some(a), Some(b)) => profile_covers(a.fit.loglik, b.fit.loglik),
            _ => None,
        };
        m.covers.insert(profile, covers);
    }
    estimates(acc, &full, &truth, cell);
    if cell.slope {
        return;
    }
    let glmm_b0 = fit(acc, "glmm_b0", "y_b0", FULL, re, 1);
    let glm_b0 = fit(acc, "glm_b0", "y_b0", FULL, "", 1);
    m.p.insert("mix_intercept", lrt_p(&glm_b0, &glmm_b0));
    let agq_full = fit(acc, "agq11_full", "y_0", FULL, re, 11);
    let agq_drop = fit(acc, "agq11_drop", "y_0", DROP_TG, re, 11);
    m.p.insert("agq_wald_tg", wald_p(&agq_full, "time:group"));
    m.p.insert("agq_lrt_tg", lrt_p(&agq_drop, &agq_full));
    let glm_full = fit(acc, "glm_full", "y_0f", FULL, "", 1);
    let glm_drop = fit(acc, "glm_drop", "y_0f", DROP_TG, "", 1);
    m.p.insert("glm_wald_tg", wald_p(&glm_full, "time:group"));
    m.p.insert("glm_lrt_tg", lrt_p(&glm_drop, &glm_full));
    // `y_0f`'s truth has `β3 = 0`.
    let truth_0f = [truth.beta[0], truth.beta[1], truth.beta[2], 0.0];
    for (j, name) in GLM_WALD_CI.into_iter().enumerate() {
        m.covers
            .insert(name, interval_covers(&glm_full, j, truth_0f[j], false));
    }
}

/// The Gaussian LMM cell's quantities (REML unless stated, Satterthwaite on).
/// The Type II rows equal the Type III ones on this design, so they are
/// measured once.
fn measure_lmm(cell: &Cell, table: &Table, acc: &mut CellAcc, m: &mut Measured) {
    let truth = Truth::of(cell);
    let (re, alt) = (random_part(cell.slope), random_part(!cell.slope));
    let fit = |acc: &mut CellAcc, kind, y: &str, fixed: &str, random: &str, reml| {
        fit_one(
            acc,
            kind,
            table,
            formula(y, fixed, random),
            Family::Gaussian,
            1,
            reml,
        )
    };
    let full = fit(acc, "lmm_reml_full", "y_full", FULL, re, None);
    let full_0 = fit(acc, "lmm_reml_full", "y_0", FULL, re, None);
    let full_00 = fit(acc, "lmm_reml_full", "y_00", FULL, re, None);
    let ml_full = fit(acc, "lmm_ml_full", "y_00", FULL, re, Some(false));
    let ml_time = fit(acc, "lmm_ml_time", "y_00", TIME, re, Some(false));
    let alt_full = fit(acc, "lmm_reml_alt_random", "y_full", FULL, alt, None);
    m.p.insert("satt_t_tg", satt_t_p(&full_0, "time:group"));
    m.p.insert("satt_t_tg_power", satt_t_p(&full, "time:group"));
    m.p.insert("satt_f_group", term_p(&full_00, SsType::Type3, "group"));
    m.p.insert("satt_f_tg00", term_p(&full_00, SsType::Type3, "time:group"));
    m.p.insert("satt_f_joint", satt_joint_p(&full_00));
    m.p.insert("ml_lrt_2df", lrt_p(&ml_time, &ml_full));
    if cell.slope {
        m.p.insert("mix_slope_power", lrt_p(&alt_full, &full));
    } else {
        m.p.insert("mix_slope", lrt_p(&full, &alt_full));
    }
    if !m.f3_aliased {
        let with_f3 = fit(acc, "lmm_reml_f3", "y_00", WITH_F3, re, None);
        m.p.insert("satt_f_f3", term_p(&with_f3, SsType::Type3, "f3"));
    }
    for (j, name) in SATT_CI.into_iter().enumerate() {
        m.covers
            .insert(name, interval_covers(&full, j, truth.beta[j], true));
    }
    estimates(acc, &full, &truth, cell);
}

/// The errors of the full `y_full` fit's estimates against the truth: `β`,
/// `τ0`, on random-slope cells `τ1` and `ρ`, on Gaussian cells `σ`.
fn estimates(acc: &mut CellAcc, f: &Option<Fitted>, truth: &Truth, cell: &Cell) {
    let Some(f) = f else { return };
    for (name, (b, t)) in ["b0", "b1", "b2", "b3"]
        .into_iter()
        .zip(f.fit.beta.iter().zip(truth.beta))
    {
        acc.add_estimate(name, b - t);
    }
    let block = &f.fit.varcorr[0];
    let tau0 = block[0].sqrt();
    acc.add_estimate("tau0", tau0 - truth.tau0);
    if cell.slope && block.len() == 3 {
        let tau1 = block[2].sqrt();
        acc.add_estimate("tau1", tau1 - truth.tau1);
        acc.add_estimate("rho", block[1] / (tau0 * tau1) - truth.rho);
    }
    if cell.response == Response::Gauss {
        acc.add_estimate("sigma", f.fit.dispersion.sqrt() - truth.sigma);
    }
}

/// One rate's counts over a cell's datasets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    /// Datasets aliased by design.
    aliased: u64,
    /// Datasets that lost the quantity: a fit did not converge, or the value
    /// is not finite.
    failed: u64,
    /// Datasets on which the quantity is not measured by design.
    skipped: u64,
    /// The datasets kept, `R_kept`.
    used: u64,
    /// Per α the rejections (a test), or per level the intervals covering
    /// the truth (an interval), among the datasets kept.
    hits: [u64; 3],
}

impl Counts {
    fn add(&mut self, o: &Counts) {
        self.aliased += o.aliased;
        self.failed += o.failed;
        self.skipped += o.skipped;
        self.used += o.used;
        for (h, x) in self.hits.iter_mut().zip(o.hits) {
            *h += x;
        }
    }

    fn rates(&self) -> [f64; 3] {
        self.hits.map(|h| h as f64 / self.used as f64)
    }
}

/// A cell's accumulators. Datasets are counted one by one and merged in
/// dataset order, so every count and sum is the same at any thread count.
#[derive(Default)]
struct CellAcc {
    datasets: u64,
    /// Datasets whose full fixed design is aliased.
    aliased: u64,
    tests: BTreeMap<&'static str, Counts>,
    covers: BTreeMap<&'static str, Counts>,
    /// Per pair of [`PAIRS`] and α, the datasets in each of [`PAIR_COLUMNS`],
    /// over the datasets where both p-values are finite.
    pairs: BTreeMap<&'static str, [[u64; 4]; 3]>,
    /// Per estimate: the count, `Σ(θ̂ − θ)` and `Σ(θ̂ − θ)²`.
    estimates: BTreeMap<&'static str, (u64, f64, f64)>,
    /// Per data measure, its sum over the datasets.
    measures: BTreeMap<&'static str, f64>,
    /// Per fit kind: the fits, those not converged, and their seconds.
    fits: BTreeMap<&'static str, (u64, u64, f64)>,
    /// The moments of `d = 1{Wald rejects} − 1{LRT rejects}` over the 15
    /// (null, α) of [`PAIRS`] and [`ALPHAS`], null-major, for the Monte Carlo
    /// SE of a sum of paired differences.
    d_moments: Moments,
    /// The same moments of the 30 indicators (the 15 Wald ones, then the 15
    /// LRT ones), for the SE of any linear combination of the two rates (a
    /// loss weighting Wald and LRT errors differently needs them).
    wl_moments: Moments,
}

/// Cross moments of an integer vector whose entries may be missing:
/// `n[i][j]` datasets with entries `i` and `j` both present, `sum[i][j]` the
/// sum of entry `i` over those datasets, `cross[i][j]` the sum of the
/// products. Means, variances and covariances on each joint subset follow
/// exactly.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Moments {
    n: Vec<Vec<u64>>,
    sum: Vec<Vec<i64>>,
    cross: Vec<Vec<i64>>,
}

impl Moments {
    fn sized(&mut self, k: usize) {
        if self.n.is_empty() {
            self.n = vec![vec![0; k]; k];
            self.sum = vec![vec![0; k]; k];
            self.cross = vec![vec![0; k]; k];
        }
    }

    fn add(&mut self, x: &[Option<i64>]) {
        self.sized(x.len());
        for (i, xi) in x.iter().enumerate() {
            let Some(xi) = xi else { continue };
            for (j, xj) in x.iter().enumerate() {
                let Some(xj) = xj else { continue };
                self.n[i][j] += 1;
                self.sum[i][j] += xi;
                self.cross[i][j] += xi * xj;
            }
        }
    }

    fn merge(&mut self, o: &Moments) {
        if o.n.is_empty() {
            return;
        }
        self.sized(o.n.len());
        for i in 0..o.n.len() {
            for j in 0..o.n.len() {
                self.n[i][j] += o.n[i][j];
                self.sum[i][j] += o.sum[i][j];
                self.cross[i][j] += o.cross[i][j];
            }
        }
    }

    fn json(&self) -> Value {
        json!({"n": self.n, "sum": self.sum, "cross": self.cross})
    }
}

impl CellAcc {
    fn add_measure(&mut self, name: &'static str, v: f64) {
        *self.measures.entry(name).or_default() += v;
    }

    fn add_estimate(&mut self, name: &'static str, err: f64) {
        if err.is_finite() {
            let e = self.estimates.entry(name).or_default();
            e.0 += 1;
            e.1 += err;
            e.2 += err * err;
        }
    }

    /// Counts one dataset's measurements.
    fn record(&mut self, cell: &Cell, m: &Measured) {
        self.datasets += 1;
        self.aliased += u64::from(m.aliased);
        for q in tests_of(cell) {
            let c = self.tests.entry(q).or_default();
            if m.aliased || (m.f3_aliased && F3.contains(&q)) {
                c.aliased += 1;
                continue;
            }
            match m.p.get(q).unwrap_or_else(|| panic!("{q} not measured")) {
                Some(p) if p.is_finite() => {
                    c.used += 1;
                    for (h, alpha) in c.hits.iter_mut().zip(ALPHAS) {
                        *h += u64::from(*p <= alpha);
                    }
                }
                _ => c.failed += 1,
            }
        }
        for q in covers_of(cell) {
            let c = self.covers.entry(q).or_default();
            if m.aliased {
                c.aliased += 1;
            } else if m.skipped.contains(&q) {
                c.skipped += 1;
            } else {
                match m
                    .covers
                    .get(q)
                    .unwrap_or_else(|| panic!("{q} not measured"))
                {
                    Some(margins) => {
                        c.used += 1;
                        for (h, x) in c.hits.iter_mut().zip(margins) {
                            *h += u64::from(*x >= 0.0);
                        }
                    }
                    None => c.failed += 1,
                }
            }
        }
        if cell.response == Response::Gauss {
            return;
        }
        // Per null and α (null-major), the Wald and LRT rejection indicators,
        // present when both p-values of the null are finite.
        let mut w = [None; 15];
        let mut l = [None; 15];
        for (i, (name, wald, lrt)) in PAIRS.into_iter().enumerate() {
            let counts = self.pairs.entry(name).or_default();
            let finite = |q: &str| {
                m.p.get(q)
                    .copied()
                    .flatten()
                    .filter(|p: &f64| p.is_finite())
            };
            if let (Some(pw), Some(pl)) = (finite(wald), finite(lrt)) {
                for (a, alpha) in ALPHAS.into_iter().enumerate() {
                    w[3 * i + a] = Some(i64::from(pw <= alpha));
                    l[3 * i + a] = Some(i64::from(pl <= alpha));
                }
                for (row, alpha) in counts.iter_mut().zip(ALPHAS) {
                    let column = match (pw <= alpha, pl <= alpha) {
                        (true, true) => 0,
                        (true, false) => 1,
                        (false, true) => 2,
                        (false, false) => 3,
                    };
                    row[column] += 1;
                }
            }
        }
        let d: Vec<Option<i64>> = w
            .iter()
            .zip(&l)
            .map(|(w, l)| Some(w.as_ref()? - l.as_ref()?))
            .collect();
        self.d_moments.add(&d);
        let wl: Vec<Option<i64>> = w.iter().chain(&l).copied().collect();
        self.wl_moments.add(&wl);
    }

    fn merge(&mut self, o: &CellAcc) {
        self.datasets += o.datasets;
        self.aliased += o.aliased;
        for (k, v) in &o.tests {
            self.tests.entry(k).or_default().add(v);
        }
        for (k, v) in &o.covers {
            self.covers.entry(k).or_default().add(v);
        }
        for (k, v) in &o.pairs {
            let e = self.pairs.entry(k).or_default();
            for (row, x) in e.iter_mut().zip(v) {
                for (a, b) in row.iter_mut().zip(x) {
                    *a += b;
                }
            }
        }
        for (k, v) in &o.estimates {
            let e = self.estimates.entry(k).or_default();
            e.0 += v.0;
            e.1 += v.1;
            e.2 += v.2;
        }
        for (k, v) in &o.measures {
            *self.measures.entry(k).or_default() += v;
        }
        for (k, v) in &o.fits {
            let e = self.fits.entry(k).or_default();
            e.0 += v.0;
            e.1 += v.1;
            e.2 += v.2;
        }
        self.d_moments.merge(&o.d_moments);
        self.wl_moments.merge(&o.wl_moments);
    }
}

/// Dataset `r` of `cell`, measured and counted.
fn run_dataset(cell: &Cell, r: u64) -> CellAcc {
    let mut acc = CellAcc::default();
    let table = generate(cell, r);
    let m = measure(cell, &table, &mut acc);
    acc.record(cell, &m);
    acc
}

/// Datasets `0..r_count` of `cell` on `threads` threads, each taking the
/// next dataset not yet taken (so a slower core takes fewer); merged in
/// dataset order, so the result does not depend on which thread ran what.
/// Returns the accumulators and the cell's wall in seconds.
fn run_cell(cell: &Cell, r_count: u64, threads: u64) -> (CellAcc, f64) {
    assert!(threads > 0, "at least one thread");
    let start = Instant::now();
    let next = AtomicU64::new(0);
    let mut done: Vec<(u64, CellAcc)> = std::thread::scope(|s| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let r = next.fetch_add(1, Ordering::Relaxed);
                        if r >= r_count {
                            break out;
                        }
                        out.push((r, run_dataset(cell, r)));
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
    let mut acc = CellAcc::default();
    for (_, d) in &done {
        acc.merge(d);
    }
    (acc, start.elapsed().as_secs_f64())
}

/// The cell's JSON object without its walls: the cell's axes, the data
/// measures (means over the datasets), and per quantity its counts, rates
/// or coverages, the paired decisions, the estimates' bias and RMSE, and the
/// fits per kind.
fn cell_json(cell: &Cell, acc: &CellAcc, r_count: u64) -> Map<String, Value> {
    let mut tests = Map::new();
    for q in tests_of(cell) {
        let c = &acc.tests[q];
        tests.insert(
            q.into(),
            json!({
                "aliased": c.aliased,
                "failed": c.failed,
                "used": c.used,
                "rejections": c.hits,
                "rates": c.rates(),
            }),
        );
    }
    let mut coverages = Map::new();
    for q in covers_of(cell) {
        let c = &acc.covers[q];
        coverages.insert(
            q.into(),
            json!({
                "aliased": c.aliased,
                "failed": c.failed,
                "skipped": c.skipped,
                "used": c.used,
                "covered": c.hits,
                "coverages": c.rates(),
            }),
        );
    }
    let datasets = acc.datasets as f64;
    let measures: Map<String, Value> = acc
        .measures
        .iter()
        .map(|(k, v)| ((*k).into(), json!(v / datasets)))
        .collect();
    let pairs: Map<String, Value> = acc
        .pairs
        .iter()
        .map(|(k, v)| ((*k).into(), json!(v)))
        .collect();
    let estimates: Map<String, Value> = acc
        .estimates
        .iter()
        .map(|(k, &(n, s, ss))| {
            let n_f = n as f64;
            (
                (*k).into(),
                json!({"n": n, "bias": s / n_f, "rmse": (ss / n_f).sqrt()}),
            )
        })
        .collect();
    let fits: Map<String, Value> = acc
        .fits
        .iter()
        .map(|(k, &(n, nonconv, _))| ((*k).into(), json!({"fits": n, "nonconv": nonconv})))
        .collect();
    let truth = Truth::of(cell);
    let wl_labels = [pair_labels("wald_"), pair_labels("lrt_")].concat();
    let Value::Object(map) = json!({
        "cell": cell.name(),
        "k": cell.k,
        "m": cell.m,
        "response": cell.response.tag(),
        "beta0": truth.beta[0],
        "random": if cell.slope { "rdis" } else { "rdi" },
        "r": r_count,
        "datasets": acc.datasets,
        "aliased": acc.aliased,
        "aliased_fraction": acc.aliased as f64 / datasets,
        "aliased_expected": expected_aliased(cell),
        "alphas": ALPHAS,
        "levels": LEVELS,
        "measures": measures,
        "tests": tests,
        "coverages": coverages,
        "pair_columns": PAIR_COLUMNS,
        "pairs": pairs,
        "pair_moments": {
            "d_labels": pair_labels(""),
            "d": acc.d_moments.json(),
            "wl_labels": wl_labels,
            "wl": acc.wl_moments.json(),
        },
        "estimates": estimates,
        "fits": fits,
    }) else {
        unreachable!("an object literal")
    };
    map
}

/// The 15 (null, α) labels of the paired moments, null-major, each prefixed.
fn pair_labels(prefix: &str) -> Vec<String> {
    PAIRS
        .iter()
        .flat_map(|(name, _, _)| ALPHAS.map(|a| format!("{prefix}{name}@{a}")))
        .collect()
}

/// The cell's JSON line: [`cell_json`] with the walls (the cell's seconds,
/// and per fit kind the seconds and the mean milliseconds per fit).
fn json_line(cell: &Cell, acc: &CellAcc, r_count: u64, wall: f64) -> String {
    let mut map = cell_json(cell, acc, r_count);
    let seconds: Map<String, Value> = acc
        .fits
        .iter()
        .map(|(k, &(_, _, s))| ((*k).into(), json!(s)))
        .collect();
    let mean_ms: Map<String, Value> = acc
        .fits
        .iter()
        .map(|(k, &(n, _, s))| ((*k).into(), json!(1000.0 * s / n as f64)))
        .collect();
    map.insert(
        "walls".into(),
        json!({"cell_s": wall, "fit_s": seconds, "fit_mean_ms": mean_ms}),
    );
    serde_json::to_string(&Value::Object(map)).expect("a JSON line")
}

/// `(q̂ − q0)/√(q0(1 − q0)/R_kept)` at each of the three nominal values.
fn standardized(c: &Counts, nominal: [f64; 3]) -> [f64; 3] {
    let rates = c.rates();
    let n = c.used as f64;
    [0, 1, 2].map(|i| (rates[i] - nominal[i]) / (nominal[i] * (1.0 - nominal[i]) / n).sqrt())
}

fn three(v: [f64; 3], digits: usize) -> String {
    v.map(|x| format!("{x:.digits$}")).join(" ")
}

/// The printed report of one cell.
fn print_cell(cell: &Cell, acc: &CellAcc, r_count: u64, wall: f64) {
    let name = cell.name();
    let datasets = acc.datasets as f64;
    let mean = |k: &str| acc.measures.get(k).map_or(f64::NAN, |v| v / datasets);
    println!(
        "CELL {name} K={} M={} response={} R={r_count} aliased={} ({:.2}%, design {:.2}%) \
         wall={wall:.2}s mean_N={:.1} mean_cluster={:.2} events/cluster={:.3} minority/cluster={:.3} \
         f3_1df={:.2}%",
        cell.k,
        cell.m,
        cell.response.tag(),
        acc.aliased,
        100.0 * acc.aliased as f64 / datasets,
        100.0 * expected_aliased(cell),
        mean("n_obs"),
        mean("cluster_size"),
        mean("events_per_cluster"),
        mean("minority_per_cluster"),
        100.0 * mean("f3_df1"),
    );
    for q in tests_of(cell) {
        let c = &acc.tests[q];
        println!(
            "TEST {name} {q} R_kept={} aliased={} failed={} rate(.01 .05 .10)=[{}] z=[{}]",
            c.used,
            c.aliased,
            c.failed,
            three(c.rates(), 4),
            three(standardized(c, ALPHAS), 2),
        );
    }
    for q in covers_of(cell) {
        let c = &acc.covers[q];
        println!(
            "COVER {name} {q} R_kept={} aliased={} failed={} skipped={} cover(.90 .95 .99)=[{}] z=[{}]",
            c.used,
            c.aliased,
            c.failed,
            c.skipped,
            three(c.rates(), 4),
            three(standardized(c, LEVELS), 2),
        );
    }
    for (k, v) in &acc.pairs {
        let at = |i: usize| v[i].map(|x| x.to_string()).join("/");
        println!(
            "PAIR {name} {k} both/wald_only/lrt_only/neither .01=[{}] .05=[{}] .10=[{}]",
            at(0),
            at(1),
            at(2),
        );
    }
    for (k, &(n, s, ss)) in &acc.estimates {
        let n_f = n as f64;
        println!(
            "EST {name} {k} n={n} bias={:.4} rmse={:.4}",
            s / n_f,
            (ss / n_f).sqrt()
        );
    }
    for (k, &(n, nonconv, s)) in &acc.fits {
        println!(
            "FIT {name} {k} fits={n} nonconv={nonconv} mean_ms={:.3}",
            1000.0 * s / n as f64
        );
    }
}

/// A count from the environment variable `name`, when set.
fn env_count(name: &str) -> Option<u64> {
    let v = std::env::var(name).ok()?;
    Some(
        v.trim()
            .parse()
            .unwrap_or_else(|_| panic!("{name}={v}: not a count")),
    )
}

/// The cells named in `MC_CELLS` (comma-separated), in canonical order, or
/// every cell.
fn selected_cells() -> Vec<Cell> {
    let Ok(list) = std::env::var("MC_CELLS") else {
        return CELLS.to_vec();
    };
    let names: Vec<&str> = list.split(',').map(str::trim).collect();
    for n in &names {
        assert!(
            CELLS.iter().any(|c| c.name() == *n),
            "MC_CELLS: no cell {n}"
        );
    }
    CELLS
        .iter()
        .filter(|c| names.contains(&c.name().as_str()))
        .copied()
        .collect()
}

/// Every cell (or those named in `MC_CELLS`) at `r_count` datasets on
/// `MC_THREADS` threads (default the available parallelism), printed, and
/// written to `MC_TIER_OUT` when set, one JSON line per cell.
fn measure_cells(r_count: u64) -> Vec<(Cell, CellAcc)> {
    let threads = env_count("MC_THREADS")
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get() as u64));
    let cells = selected_cells();
    let mut out = std::env::var("MC_TIER_OUT").ok().map(|path| {
        std::fs::File::create(&path).unwrap_or_else(|e| panic!("MC_TIER_OUT={path}: {e}"))
    });
    println!(
        "MC calibration: {} cells, R={r_count}, threads={threads}; \
         z = (q̂ − q0)/√(q0(1 − q0)/R_kept), a coverage's against its level",
        cells.len()
    );
    let start = Instant::now();
    let mut measured = Vec::with_capacity(cells.len());
    for cell in cells {
        let (acc, wall) = run_cell(&cell, r_count, threads);
        print_cell(&cell, &acc, r_count, wall);
        if let Some(f) = out.as_mut() {
            writeln!(f, "{}", json_line(&cell, &acc, r_count, wall)).expect("MC_TIER_OUT written");
            f.flush().expect("MC_TIER_OUT flushed");
        }
        measured.push((cell, acc));
    }
    println!(
        "MC calibration done: {} cells in {:.1}s",
        measured.len(),
        start.elapsed().as_secs_f64()
    );
    measured
}

/// The gates of `cell` on its counts. Per gated quantity ([`bands::gates`]):
/// the aliased fraction at most [`bands::MC_ALIASED_MAX`]; the failed
/// fraction of the datasets not aliased at most [`bands::MC_FAIL_MAX`]; and
/// each gate's band at each α (a rejection rate) or level (a coverage, read on
/// its miss rate), or at the gate's only α. Returns the band checks made and
/// one message per miss, naming the cell, the quantity, the rate, the band and
/// the standardized error.
fn gate_cell(cell: &Cell, acc: &CellAcc) -> (usize, Vec<String>) {
    let name = cell.name();
    let (mut checks, mut misses) = (0, Vec::new());
    let tests = tests_of(cell)
        .into_iter()
        .map(|q| (q, &acc.tests[q], false));
    let covers = covers_of(cell)
        .into_iter()
        .map(|q| (q, &acc.covers[q], true));
    for (q, c, cover) in tests.chain(covers) {
        let gates: Vec<&Gate> = bands::gates(&name, q).collect();
        if gates.is_empty() {
            continue;
        }
        let datasets = c.aliased + c.failed + c.skipped + c.used;
        let aliased = c.aliased as f64 / datasets as f64;
        if aliased > bands::MC_ALIASED_MAX {
            misses.push(format!(
                "{name} {q}: aliased on {} of {datasets} datasets ({:.2}%), \
                 above mc_aliased_max {:.0}%",
                c.aliased,
                100.0 * aliased,
                100.0 * bands::MC_ALIASED_MAX
            ));
        }
        // Every dataset aliased leaves a NaN here, which the guard above has
        // already reported.
        let at_risk = c.failed + c.used;
        let failed = c.failed as f64 / at_risk as f64;
        if failed > bands::MC_FAIL_MAX {
            misses.push(format!(
                "{name} {q}: failed on {} of {at_risk} datasets not aliased ({:.2}%), \
                 above mc_fail_max {:.0}%",
                c.failed,
                100.0 * failed,
                100.0 * bands::MC_FAIL_MAX
            ));
        }
        let (k, nominal) = if cover {
            (bands::MC_COVER_K, LEVELS)
        } else {
            (bands::MC_LEVEL_K, ALPHAS)
        };
        for g in gates {
            for (&nom, &rate) in nominal.iter().zip(&c.rates()) {
                if g.alpha.is_some_and(|a| a != nom) {
                    continue;
                }
                checks += 1;
                // A coverage is read on its miss rate against `1 − level`.
                let (q0, q_hat) = if cover {
                    (1.0 - nom, 1.0 - rate)
                } else {
                    (nom, rate)
                };
                let se = (q0 * (1.0 - q0) / c.used as f64).sqrt();
                let half = g.delta * q0 + k * se;
                let err = q_hat - q0;
                // A NaN rate (no dataset kept) is outside every band.
                let (inside, band) = match g.side {
                    Side::Two => (
                        err.abs() <= half,
                        format!("[{:.4}, {:.4}]", q0 - half, q0 + half),
                    ),
                    Side::Upper => (err <= half, format!("at most {:.4}", q0 + half)),
                    Side::Lower => (-err <= half, format!("at least {:.4}", q0 - half)),
                };
                if !inside {
                    let (at, what) = if cover {
                        ("level", "miss rate")
                    } else {
                        ("alpha", "rate")
                    };
                    misses.push(format!(
                        "{name} {q} ({}): {at} {nom}: {what} {q_hat:.4}, band {band}, \
                         z = {:.2}, R_kept = {}",
                        g.family,
                        err / se,
                        c.used
                    ));
                }
            }
        }
    }
    (checks, misses)
}

/// The measurement: [`measure_cells`] at `R` from `MC_R` (default `mc_r`),
/// with no gate.
#[test]
#[ignore = "the measurement run: run by name with --ignored"]
fn calibration_measurement() {
    measure_cells(env_count("MC_R").unwrap_or(bands::MC_R as u64));
}

/// The gates of the frozen set: [`measure_cells`] at `mc_r`, the `R` the
/// bands are built for (`MC_R` set to another value is refused), then
/// [`gate_cell`] on every cell measured, the misses collected and asserted
/// once. With `MC_CELLS` set, only the cells named are measured and gated,
/// and a line says the run is a subset, not the release gate; a subset that
/// gates nothing fails, and the full run must make all 4416 checks.
#[test]
fn calibration_gates() {
    let r_count = bands::MC_R as u64;
    if let Some(r) = env_count("MC_R") {
        assert_eq!(
            r, r_count,
            "MC_R={r}: calibration_gates measures at mc_r = {r_count}, the R its bands are \
             built for; unset MC_R, or run calibration_measurement for another R"
        );
    }
    let measured = measure_cells(r_count);
    let (mut checks, mut gated, mut misses) = (0, 0, Vec::new());
    for (cell, acc) in &measured {
        let (n, m) = gate_cell(cell, acc);
        checks += n;
        gated += usize::from(n > 0);
        misses.extend(m);
    }
    println!(
        "MC gates: {gated} gated cells of the {} measured ({} in the grid), \
         {checks} band checks, {} misses",
        measured.len(),
        CELLS.len(),
        misses.len()
    );
    if measured.len() < CELLS.len() {
        println!(
            "MC gates: SUBSET, {} of the {} cells (MC_CELLS), not the release gate; \
             unset MC_CELLS to gate the tier",
            measured.len(),
            CELLS.len()
        );
    }
    assert!(
        checks > 0,
        "MC gates: none of the {} cells measured carries a gated quantity, so nothing \
         was gated (MC_CELLS names reported cells only)",
        measured.len()
    );
    if measured.len() == CELLS.len() {
        assert_eq!(
            checks, 4416,
            "MC gates: the full run made {checks} band checks, not the frozen set's 4416"
        );
    }
    assert!(
        misses.is_empty(),
        "{} gated checks missed:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

fn cell(name: &str) -> Cell {
    *CELLS
        .iter()
        .find(|c| c.name() == name)
        .unwrap_or_else(|| panic!("no cell {name}"))
}

/// The number of `f3` levels present in `table`.
fn f3_levels(table: &Table) -> usize {
    match table.columns.iter().find(|(n, _)| n == "f3") {
        Some((_, Column::Factor { codes, .. })) => {
            let mut seen = [false; 3];
            for &c in codes {
                seen[c as usize] = true;
            }
            seen.iter().filter(|&&s| s).count()
        }
        _ => panic!("no f3"),
    }
}

/// The first dataset of `cell` that `keep` accepts.
fn find(cell: &Cell, keep: impl Fn(&Table) -> bool) -> Table {
    (0..10_000)
        .map(|r| generate(cell, r))
        .find(keep)
        .unwrap_or_else(|| panic!("{}: no such dataset", cell.name()))
}

/// A dataset of `cell` with a full-rank design and every `f3` level present.
fn clean(cell: &Cell) -> Table {
    let random = random_part(cell.slope);
    let fam = family(cell.response);
    find(cell, |t| {
        f3_levels(t) == 3 && !design_aliased(t, fam, random)
    })
}

/// The Wald z of one column gives the Type III Wald χ² row of `term_tests`.
#[test]
fn wald_p_is_the_type3_wald_row() {
    let c = cell("k10_m9_bern54_rdi");
    let table = clean(&c);
    let mut acc = CellAcc::default();
    let f = fit_one(
        &mut acc,
        "glmm_full",
        &table,
        formula("y_0", FULL, "(1 | id)"),
        family(c.response),
        1,
        None,
    );
    assert!(f.is_some(), "converged");
    for name in ["group", "time:group"] {
        let (z, row) = (
            wald_p(&f, name).unwrap(),
            term_p(&f, SsType::Type3, name).unwrap(),
        );
        assert!((z - row).abs() <= 1e-10 * row, "{name}: {z} against {row}");
    }
}

/// The profile duality decides as the explicit check of the statistic
/// against the χ²₁ quantile `z²` at each level.
#[test]
fn profile_duality_matches_the_quantile_check() {
    for i in 0..1200 {
        let stat = f64::from(i) * 0.01;
        let margins = profile_covers(stat / 2.0, 0.0).unwrap();
        for (margin, level) in margins.into_iter().zip(LEVELS) {
            let q = normal_quantile(0.5 + level / 2.0);
            assert_eq!(margin >= 0.0, stat <= q * q, "stat {stat} level {level}");
        }
    }
    let floored = profile_covers(-12.0, -10.0).unwrap();
    assert_eq!(floored.map(|m| m >= 0.0), [true; 3]);
    assert_eq!(profile_covers(f64::NAN, -10.0), None);
}

/// A dataset whose full design is aliased (as `fit_cold` decides it) is
/// counted as aliased in every quantity, and fitted no further.
#[test]
fn aliased_designs_are_left_out_of_every_quantity() {
    let c = cell("k10_m3_gauss_rdi");
    let table = find(&c, |t| design_aliased(t, Family::Gaussian, "(1 | id)"));
    let lo = lowered(
        &formula("y_full", FULL, "(1 | id)"),
        &table,
        Family::Gaussian,
    );
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged());
    assert_eq!(fit.aliased(), aliased_columns(&lo), "fit_cold's own test");
    let mut acc = CellAcc::default();
    let m = measure(&c, &table, &mut acc);
    acc.record(&c, &m);
    assert_eq!(acc.aliased, 1);
    assert!(acc.fits.is_empty());
    for (q, k) in acc.tests.iter().chain(&acc.covers) {
        assert_eq!((k.aliased, k.failed, k.used), (1, 0, 0), "{q}");
    }
}

/// An `f3` level absent leaves a 1-df `f3` row that is tested; every `f3`
/// column aliased leaves the `f3` quantities aliased and the rest kept.
#[test]
fn f3_rows_carry_their_df() {
    let c = cell("k10_m3_gauss_rdi");
    let random = "(1 | id)";
    let table = find(&c, |t| {
        f3_levels(t) == 2 && !design_aliased(t, Family::Gaussian, random)
    });
    let mut acc = CellAcc::default();
    let m = measure(&c, &table, &mut acc);
    assert!(!m.f3_aliased);
    assert_eq!(acc.measures["f3_df1"], 1.0);
    assert!(m.p["satt_f_f3"].is_some_and(f64::is_finite));
    let f = fit_one(
        &mut acc,
        "lmm_reml_f3",
        &table,
        formula("y_00", WITH_F3, random),
        Family::Gaussian,
        1,
        None,
    )
    .expect("converged");
    let opts = TermTestOptions {
        ss_type: SsType::Type3,
        test: TermTest::Auto,
    };
    let rows = term_tests(&f.lo, &f.fit, &opts).unwrap();
    let row = rows.fixed.iter().find(|r| r.label == "f3").unwrap();
    assert_eq!(row.test.num_df, 1.0);

    let mut one_level = clean(&c);
    for (name, column) in &mut one_level.columns {
        if name.as_str() != "f3" {
            continue;
        }
        if let Column::Factor { codes, .. } = column {
            codes.iter_mut().for_each(|code| *code = 0);
        }
    }
    let mut acc = CellAcc::default();
    let m = measure(&c, &one_level, &mut acc);
    acc.record(&c, &m);
    assert!(m.f3_aliased);
    for (q, k) in &acc.tests {
        let aliased = u64::from(F3.contains(q));
        assert_eq!(k.aliased, aliased, "{q}");
        assert_eq!(k.aliased + k.failed + k.used, 1, "{q}");
    }
}

/// The counts and sums of a cell do not depend on the thread count: its JSON
/// line, walls excluded, is the same at 1 and at 3 threads.
#[test]
fn counts_do_not_depend_on_threads() {
    let c = cell("k10_m3_bern54_rdi");
    let line = |threads| {
        let (acc, _) = run_cell(&c, 12, threads);
        assert_eq!(acc.datasets, 12);
        serde_json::to_string(&Value::Object(cell_json(&c, &acc, 12))).unwrap()
    };
    assert_eq!(line(1), line(3));
}

/// A fit as the hand rebuild makes it: the formula (`RE` and `ALT` stand for
/// the cell's random part and the other one), `nagq` and `reml`.
#[derive(Clone)]
struct Spec(String, u8, Option<bool>);

/// A quantity rebuilt by hand from glmm's public pieces, independently of the
/// driver's helpers and constants.
enum Hand {
    /// The Wald z of a column: `chisq_upper_p(z², 1)`.
    Z(Spec, &'static str),
    /// A `term_tests` row: the table's type, its named test, the rows' kind.
    Row(Spec, SsType, TermTest, LinearKind, &'static str),
    /// `compare` of a smaller and a larger fit, with the reference it must use.
    Lr(Spec, Spec, Reference),
    /// `contrast_t` on a column's unit vector.
    T(Spec, &'static str),
    /// `contrast_f` of `group` and `time:group` jointly.
    Joint(Spec),
    /// The interval of `β_j` against a truth, with the t quantile on
    /// `coef_df` when true: margins `q·SE − |β̂ − truth|`.
    Ci(Spec, usize, bool, f64),
    /// The profile interval by duality: the full fit and the fit holding
    /// `β_j` through its offset; margins `p − (1 − level)`.
    Profile(Spec, Spec),
}

/// The hand table of a cell: every quantity of the cell but the skipped
/// ones, each with its rebuild.
fn hand_table(cell: &Cell) -> Vec<(&'static str, Hand)> {
    let re = random_part(cell.slope);
    let alt = random_part(!cell.slope);
    let sp = |f: &str, nagq: u8, reml: Option<bool>| {
        Spec(f.replace("ALT", alt).replace("RE", re), nagq, reml)
    };
    let la = |f: &str| sp(f, 1, None);
    let chi = |df| Reference::ChiSquared { df };
    let bar = |df_low| Reference::ChiBarSquared { df_low };
    let beta = Truth::of(cell).beta;
    let mut t: Vec<(&'static str, Hand)> = Vec::new();
    if cell.response == Response::Gauss {
        let full = |y: &str| la(&format!("{y} ~ time + group + time:group + RE"));
        let ml = |f: &str| sp(f, 1, Some(false));
        let f3 = la("y_00 ~ time + group + time:group + f3 + RE");
        let f_row =
            |f: Spec, label| Hand::Row(f, SsType::Type3, TermTest::Auto, LinearKind::F, label);
        t.push(("satt_t_tg", Hand::T(full("y_0"), "time:group")));
        t.push(("satt_t_tg_power", Hand::T(full("y_full"), "time:group")));
        t.push(("satt_f_group", f_row(full("y_00"), "group")));
        t.push(("satt_f_tg00", f_row(full("y_00"), "time:group")));
        t.push(("satt_f_f3", f_row(f3, "f3")));
        t.push(("satt_f_joint", Hand::Joint(full("y_00"))));
        t.push((
            "ml_lrt_2df",
            Hand::Lr(
                ml("y_00 ~ time + RE"),
                ml("y_00 ~ time + group + time:group + RE"),
                chi(2),
            ),
        ));
        let alt_full = la("y_full ~ time + group + time:group + ALT");
        t.push(if cell.slope {
            (
                "mix_slope_power",
                Hand::Lr(alt_full, full("y_full"), bar(1)),
            )
        } else {
            ("mix_slope", Hand::Lr(full("y_full"), alt_full, bar(1)))
        });
        for (j, name) in SATT_CI.into_iter().enumerate() {
            t.push((name, Hand::Ci(full("y_full"), j, true, beta[j])));
        }
        return t;
    }
    let full = |y: &str| la(&format!("{y} ~ time + group + time:group + RE"));
    let wald_row =
        |f: Spec, ss, label| Hand::Row(f, ss, TermTest::Wald, LinearKind::ChiSquared, label);
    t.push(("wald_tg", Hand::Z(full("y_0"), "time:group")));
    t.push((
        "lrt_tg",
        Hand::Lr(la("y_0 ~ time + group + RE"), full("y_0"), chi(1)),
    ));
    t.push(("wald3_group", Hand::Z(full("y_00"), "group")));
    t.push((
        "lrt3_group",
        Hand::Lr(la("y_00 ~ time + time:group + RE"), full("y_00"), chi(1)),
    ));
    t.push((
        "wald2_group",
        wald_row(full("y_00"), SsType::Type2, "group"),
    ));
    t.push((
        "lrt2_group",
        Hand::Lr(
            la("y_00 ~ time + RE"),
            la("y_00 ~ time + group + RE"),
            chi(1),
        ),
    ));
    t.push(("wald_tg00", Hand::Z(full("y_00"), "time:group")));
    t.push((
        "lrt_tg00",
        Hand::Lr(la("y_00 ~ time + group + RE"), full("y_00"), chi(1)),
    ));
    let f3 = la("y_00 ~ time + group + time:group + f3 + RE");
    t.push(("wald_f3", wald_row(f3.clone(), SsType::Type3, "f3")));
    t.push(("lrt_f3", Hand::Lr(full("y_00"), f3, chi(2))));
    t.push((
        "lrt_2df",
        Hand::Lr(la("y_00 ~ time + RE"), full("y_00"), chi(2)),
    ));
    t.push(("wald_tg_power", Hand::Z(full("y_full"), "time:group")));
    t.push((
        "lrt_2df_power",
        Hand::Lr(la("y_full ~ time + RE"), full("y_full"), chi(2)),
    ));
    let alt_full = la("y_full ~ time + group + time:group + ALT");
    t.push(if cell.slope {
        (
            "mix_slope_power",
            Hand::Lr(alt_full, full("y_full"), bar(1)),
        )
    } else {
        ("mix_slope", Hand::Lr(full("y_full"), alt_full, bar(1)))
    });
    let offsets = [
        "y_full ~ 0 + time + group + time:group + offset(off0) + RE",
        "y_full ~ group + time:group + offset(off1) + RE",
        "y_full ~ time + time:group + offset(off2) + RE",
        "y_full ~ time + group + offset(off3) + RE",
    ];
    for j in 0..4 {
        t.push((WALD_CI[j], Hand::Ci(full("y_full"), j, false, beta[j])));
        if !(cell.slope && j == 1) {
            t.push((PROFILE_CI[j], Hand::Profile(full("y_full"), la(offsets[j]))));
        }
    }
    if cell.slope {
        return t;
    }
    t.push((
        "mix_intercept",
        Hand::Lr(la("y_b0 ~ time + group + time:group"), full("y_b0"), bar(0)),
    ));
    let agq = |f: &str| sp(f, 11, None);
    let agq_full = agq("y_0 ~ time + group + time:group + RE");
    t.push(("agq_wald_tg", Hand::Z(agq_full.clone(), "time:group")));
    t.push((
        "agq_lrt_tg",
        Hand::Lr(agq("y_0 ~ time + group + RE"), agq_full, chi(1)),
    ));
    let glm = la("y_0f ~ time + group + time:group");
    t.push(("glm_wald_tg", Hand::Z(glm.clone(), "time:group")));
    t.push((
        "glm_lrt_tg",
        Hand::Lr(la("y_0f ~ time + group"), glm.clone(), chi(1)),
    ));
    let beta_0f = [beta[0], beta[1], beta[2], 0.0];
    for j in 0..4 {
        t.push((GLM_WALD_CI[j], Hand::Ci(glm.clone(), j, false, beta_0f[j])));
    }
    t
}

/// A converged fit of `spec`, made here.
fn hand_fit(table: &Table, family: Family, spec: &Spec) -> (Lowered, Fit) {
    let mut lo = lower(&spec.0, table, family).unwrap();
    lo.opts.nagq = spec.1;
    lo.opts.reml = spec.2;
    let fit = fit_cold(&lo.x, &lo.y, lo.n, lo.p, &lo.model, &lo.ids, &lo.opts);
    assert!(fit.converged(), "{}", spec.0);
    (lo, fit)
}

/// The rebuilt value: a p-value, or an interval's three margins.
enum Rebuilt {
    P(f64),
    Margins([f64; 3]),
}

fn rebuild(table: &Table, family: Family, hand: &Hand) -> Rebuilt {
    let column = |lo: &Lowered, name: &str| lo.col_names.iter().position(|c| c == name).unwrap();
    match hand {
        Hand::Z(spec, name) => {
            let (lo, fit) = hand_fit(table, family, spec);
            let j = column(&lo, name);
            let z = fit.beta[j] / fit.se[j];
            Rebuilt::P(chisq_upper_p(z * z, 1.0))
        }
        Hand::Row(spec, ss_type, test, kind, label) => {
            let (lo, fit) = hand_fit(table, family, spec);
            let opts = TermTestOptions {
                ss_type: *ss_type,
                test: *test,
            };
            let rows = term_tests(&lo, &fit, &opts).unwrap();
            assert_eq!(rows.choice.test, *kind, "{}", spec.0);
            let row = rows.fixed.iter().find(|r| r.label == *label).unwrap();
            Rebuilt::P(row.test.p_value)
        }
        Hand::Lr(small, large, reference) => {
            let model = |spec: &Spec| {
                let (lo, fit) = hand_fit(table, family, spec);
                Model::from_fit(&spec.0, &lo, &fit)
            };
            let comparison = compare(&model(small), &model(large), &COMPARE).unwrap();
            let ComparisonTest::Lr(test) = comparison.test else {
                panic!("{}: not a likelihood-ratio test", large.0)
            };
            assert_eq!(
                test.reference, *reference,
                "{} against {}",
                small.0, large.0
            );
            Rebuilt::P(test.p_value)
        }
        Hand::T(spec, name) => {
            let (lo, fit) = hand_fit(table, family, spec);
            let mut l = vec![0.0; lo.p];
            l[column(&lo, name)] = 1.0;
            Rebuilt::P(fit.contrast_t(&l).unwrap().p_value)
        }
        Hand::Joint(spec) => {
            let (lo, fit) = hand_fit(table, family, spec);
            let mut l = vec![0.0; 2 * lo.p];
            l[column(&lo, "group")] = 1.0;
            l[lo.p + column(&lo, "time:group")] = 1.0;
            Rebuilt::P(fit.contrast_f(&l, 2).unwrap().p_value)
        }
        Hand::Ci(spec, j, t_df, truth) => {
            let (_, fit) = hand_fit(table, family, spec);
            Rebuilt::Margins(LEVELS.map(|level| {
                let q = if *t_df {
                    t_quantile(0.5 + level / 2.0, fit.coef_df[*j])
                } else {
                    normal_quantile(0.5 + level / 2.0)
                };
                q * fit.se[*j] - (fit.beta[*j] - truth).abs()
            }))
        }
        Hand::Profile(full, held) => {
            let (_, a) = hand_fit(table, family, full);
            let (_, b) = hand_fit(table, family, held);
            let p = chisq_upper_p((2.0 * (a.loglik - b.loglik)).max(0.0), 1.0);
            Rebuilt::Margins(LEVELS.map(|level| p - (1.0 - level)))
        }
    }
}

/// On one clean dataset of a well-behaved cell: every quantity is measured
/// and kept (none failed), and each equals its hand rebuild bit for bit.
fn check_against_hand(name: &str) {
    let c = cell(name);
    let table = clean(&c);
    let mut acc = CellAcc::default();
    let m = measure(&c, &table, &mut acc);
    acc.record(&c, &m);
    for (q, k) in acc.tests.iter().chain(&acc.covers) {
        let kept = u64::from(!m.skipped.contains(q));
        assert_eq!(
            (k.used, k.failed, k.aliased, k.skipped),
            (kept, 0, 0, 1 - kept),
            "{name} {q}"
        );
    }
    let pairs = if c.response == Response::Gauss {
        0
    } else {
        PAIRS.len()
    };
    assert_eq!(acc.pairs.len(), pairs, "{name}: Wald/LRT pairs");
    let hand = hand_table(&c);
    let mut listed: Vec<&str> = hand
        .iter()
        .map(|(q, _)| *q)
        .chain(m.skipped.iter().copied())
        .collect();
    listed.sort_unstable();
    let mut all = [tests_of(&c), covers_of(&c)].concat();
    all.sort_unstable();
    assert_eq!(listed, all, "{name}: the hand table covers every quantity");
    let fam = family(c.response);
    for (q, h) in &hand {
        match rebuild(&table, fam, h) {
            Rebuilt::P(p) => assert_eq!(
                m.p[q].map(f64::to_bits),
                Some(p.to_bits()),
                "{name} {q}: {:?} against {p}",
                m.p[q]
            ),
            Rebuilt::Margins(v) => assert_eq!(
                m.covers[q].map(|x| x.map(f64::to_bits)),
                Some(v.map(f64::to_bits)),
                "{name} {q}: {:?} against {v:?}",
                m.covers[q]
            ),
        }
    }
}

#[test]
fn glmm_intercept_quantities_match_their_hand_rebuild() {
    check_against_hand("k50_m9_bern54_rdi");
}

#[test]
fn glmm_slope_quantities_match_their_hand_rebuild() {
    check_against_hand("k50_m9_bern54_rdis");
}

#[test]
fn lmm_intercept_quantities_match_their_hand_rebuild() {
    check_against_hand("k30_m9_gauss_rdi");
}

#[test]
fn lmm_slope_quantities_match_their_hand_rebuild() {
    check_against_hand("k30_m9_gauss_rdis");
}

/// The paired moments on three hand-made datasets: `tg` decided on all
/// three, `f3` lost on the second (a failed Wald row), `tg00` on the third
/// (a non-finite LRT); every other p-value 0.5.
#[test]
fn paired_moments_match_a_hand_count() {
    let c = cell("k10_m9_bern54_rdi");
    let base = |set: &[(&'static str, Option<f64>)]| {
        let mut m = Measured::default();
        for q in tests_of(&c) {
            m.p.insert(q, Some(0.5));
        }
        for q in covers_of(&c) {
            m.covers.insert(q, Some([1.0; 3]));
        }
        for (q, p) in set {
            m.p.insert(q, *p);
        }
        m
    };
    let datasets = [
        base(&[("wald_tg", Some(0.03)), ("lrt_tg", Some(0.07))]),
        base(&[
            ("wald_tg", Some(0.2)),
            ("lrt_tg", Some(0.004)),
            ("wald_f3", None),
        ]),
        base(&[("lrt_tg00", Some(f64::NAN))]),
    ];
    let mut acc = CellAcc::default();
    for m in &datasets {
        acc.record(&c, m);
    }
    // Index 3·null + α: tg is null 0, tg00 null 1, f3 null 4.
    let (tg01, tg05, tg00_01, f3_05) = (0, 1, 3, 13);
    let d = &acc.d_moments;
    // d(tg, 0.05) = 1, -1, 0; d(tg, 0.01) = 0, -1, 0; f3 present on 1 and 3.
    assert_eq!(
        (d.n[tg05][tg05], d.sum[tg05][tg05], d.cross[tg05][tg05]),
        (3, 0, 2)
    );
    assert_eq!(
        (d.n[tg01][tg05], d.sum[tg01][tg05], d.cross[tg01][tg05]),
        (3, -1, 1)
    );
    assert_eq!(
        (d.n[tg05][f3_05], d.sum[tg05][f3_05], d.cross[tg05][f3_05]),
        (2, 1, 0)
    );
    assert_eq!(d.n[tg00_01][tg00_01], 2);
    assert_eq!(d.n[f3_05][tg00_01], 1);
    let wl = &acc.wl_moments;
    let (w, l) = (|i: usize| i, |i: usize| 15 + i);
    // Wald at 0.05 rejects on the first dataset only, the LRT on the second.
    assert_eq!(
        (
            wl.n[w(tg05)][l(tg05)],
            wl.sum[w(tg05)][l(tg05)],
            wl.cross[w(tg05)][l(tg05)]
        ),
        (3, 1, 0)
    );
    // The d moments follow from the indicator moments: d = w − l.
    for i in 0..15 {
        for j in 0..15 {
            let x = |a: usize, b: usize| wl.cross[a][b];
            let cross = x(w(i), w(j)) - x(w(i), l(j)) - x(l(i), w(j)) + x(l(i), l(j));
            assert_eq!(d.cross[i][j], cross, "cross {i} {j}");
            assert_eq!(
                d.sum[i][j],
                wl.sum[w(i)][w(j)] - wl.sum[l(i)][l(j)],
                "sum {i} {j}"
            );
            assert_eq!(d.n[i][j], wl.n[w(i)][w(j)], "n {i} {j}");
        }
    }
    // Merged datasets give the same moments as one pass.
    let mut merged = CellAcc::default();
    for m in &datasets {
        let mut one = CellAcc::default();
        one.record(&c, m);
        merged.merge(&one);
    }
    assert_eq!(merged.d_moments, acc.d_moments);
    assert_eq!(merged.wl_moments, acc.wl_moments);
}

/// A cell's counts with every quantity kept on 2000 datasets at its nominal
/// rate: rejections at α, coverage at the level.
fn nominal_counts(cell: &Cell) -> CellAcc {
    let mut acc = CellAcc {
        datasets: 2000,
        ..CellAcc::default()
    };
    let kept = |hits| Counts {
        used: 2000,
        hits,
        ..Counts::default()
    };
    for q in tests_of(cell) {
        acc.tests.insert(q, kept([20, 100, 200]));
    }
    for q in covers_of(cell) {
        acc.covers.insert(q, kept([1800, 1900, 1980]));
    }
    acc
}

/// A band miss names the cell, the quantity and its family, the α, the rate,
/// the band, the standardized error and `R_kept`.
#[test]
fn a_band_miss_names_cell_quantity_rate_band_and_z() {
    let c = cell("k10_m3_gauss_rdi");
    let mut acc = nominal_counts(&c);
    acc.tests.get_mut("satt_t_tg").expect("satt_t_tg").hits[1] = 144;
    assert_eq!(
        gate_cell(&c, &acc).1,
        [
            "k10_m3_gauss_rdi satt_t_tg (satt_upper): alpha 0.05: rate 0.0720, \
             band at most 0.0719, z = 4.51, R_kept = 2000"
        ]
    );
}

/// At nominal counts the gates make the frozen set's 4416 checks and pass.
/// Each side of a band, the failure cap and the aliased guard pass a count at
/// their bound and miss one dataset past it; a reported quantity has no gate.
#[test]
fn gates_miss_just_past_their_bounds() {
    let mut total = 0;
    for c in &CELLS {
        let (checks, misses) = gate_cell(c, &nominal_counts(c));
        assert!(misses.is_empty(), "{misses:?}");
        total += checks;
    }
    assert_eq!(total, 4416);
    let c = cell("k10_m3_gauss_rdi");
    // satt_upper and satt_upper_f3 (9 quantities) and mix_slope, three α each.
    assert_eq!(gate_cell(&c, &nominal_counts(&c)).0, 30);
    type Edit = fn(&mut Counts);
    let cases: [(&str, &str, Edit, usize); 17] = [
        // Upper, δ = 0: at most 0.05 + 4.5·SE = 0.0719 at α = 0.05.
        ("k10_m3_gauss_rdi", "satt_t_tg", |c| c.hits[1] = 143, 0),
        ("k10_m3_gauss_rdi", "satt_t_tg", |c| c.hits[1] = 144, 1),
        ("k10_m3_gauss_rdi", "satt_t_tg", |c| c.hits = [0; 3], 0),
        // Two-sided, δ = 0: at least 0.0281 at α = 0.05; a coverage's miss
        // rate at most 0.0719 at level 0.95.
        ("k20_m3_gauss_rdi", "satt_f_group", |c| c.hits[1] = 57, 0),
        ("k20_m3_gauss_rdi", "satt_f_group", |c| c.hits[1] = 56, 1),
        ("k20_m3_gauss_rdi", "satt_ci_b0", |c| c.hits[1] = 1857, 0),
        ("k20_m3_gauss_rdi", "satt_ci_b0", |c| c.hits[1] = 1856, 1),
        // Two-sided, δ = 0.25: at least 0.10 − 0.025 − 4.5·SE = 0.0448 at
        // α = 0.10.
        ("k50_m9_bern54_rdi", "wald2_group", |c| c.hits[2] = 90, 0),
        ("k50_m9_bern54_rdi", "wald2_group", |c| c.hits[2] = 89, 1),
        // The lower bound, at α = 0.10 only (0.0698); the upper side passes.
        ("k50_m3_gauss_rdi", "mix_slope", |c| c.hits = [0, 0, 140], 0),
        ("k50_m3_gauss_rdi", "mix_slope", |c| c.hits = [0, 0, 139], 1),
        // The failure cap, 1% of the datasets not aliased.
        (
            "k50_m9_bern54_rdi",
            "wald_tg",
            |c| (c.failed, c.used) = (20, 1980),
            0,
        ),
        (
            "k50_m9_bern54_rdi",
            "wald_tg",
            |c| (c.failed, c.used) = (21, 1979),
            1,
        ),
        // The aliased guard, 10% of the datasets.
        (
            "k50_m9_bern54_rdi",
            "lrt_tg",
            |c| (c.aliased, c.used) = (200, 1800),
            0,
        ),
        (
            "k50_m9_bern54_rdi",
            "lrt_tg",
            |c| (c.aliased, c.used) = (201, 1799),
            1,
        ),
        // Reported: the GLMM f3 row, and the Satterthwaite f3 row moved after
        // the measurement.
        ("k50_m9_bern54_rdi", "wald_f3", |c| c.hits = [2000; 3], 0),
        ("k10_m3_gauss_rdis", "satt_f_f3", |c| c.hits = [2000; 3], 0),
    ];
    for (name, q, edit, want) in cases {
        let c = cell(name);
        let mut acc = nominal_counts(&c);
        match acc.tests.get_mut(q) {
            Some(counts) => edit(counts),
            None => edit(acc.covers.get_mut(q).expect(q)),
        }
        let (_, misses) = gate_cell(&c, &acc);
        assert_eq!(misses.len(), want, "{name} {q}: {misses:?}");
    }
}
