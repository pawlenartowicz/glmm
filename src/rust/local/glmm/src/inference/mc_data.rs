//! The design of the Monte Carlo calibration tier: the grid of cells, the
//! generating truth, the per-cell seeds and the generator of each cell's
//! datasets. The tier's only reference is this truth, so the pins in
//! `mc_data_tests` are its frozen inputs.
//!
//! The longitudinal design ports the simulation design of Li, X. (2026),
//! *A Comparison of R Packages for Estimating Generalized Linear Mixed
//! Models*, arXiv:2606.15933: subjects observed at 1 to `M` distinct times of
//! `(0, 1, ..., M-1)/(M-1)`, a subject-level binary `group` with probability
//! 0.4, the linear predictor `β0 + β1·time + β2·group + β3·time·group + b0 +
//! b1·time` with `β = (β0, 0.3, -0.2, 0.1)`, `τ0 = 0.8` and on the slope
//! designs `τ1 = 0.4`, `ρ = 0.2`, and the nested responses `y_full`, `y_0`
//! (`β3 = 0`) and `y_00` (`β2 = β3 = 0`). Only the design is taken: no code or
//! data of that work is used, and the seeds are our own.
//!
//! One dataset is one Philox stream,
//! `CommonStatsRng::new_tagged(cell_seed(name), r, MC_STREAM_TAG)`, so the same
//! `(cell, r)` gives the same table on every platform, wasm32 included: every
//! sampler is `commonstats`' and the inverse links use `libm::exp`. Per
//! subject, in order: the visit count `1 + bounded(M)`; the visits by a partial
//! Fisher-Yates shuffle of `0..M` with `bounded`, then sorted; `group` from
//! `Bernoulli(0.4)`; `f3` from `bounded(3)`; `z1` (and `z2` on the slope
//! designs) from `Normal(0, 1)`, with `b0 = τ0·z1` and
//! `b1 = τ1·(ρ·z1 + √(1−ρ²)·z2)`; then per visit the five responses in column
//! order, one draw each. `f3` enters no linear predictor.

use crate::formula::{Column, Table};
use commonstats::dist::{Bernoulli, DiscreteSampler, Normal, Poisson, Sampler};
use commonstats::rng::CommonStatsRng;

/// The stream tag of the tier's datasets; the bytes spell `MCCALIB1`, unlike
/// every `commonstats` `STREAM_TAG_*`.
pub(crate) const MC_STREAM_TAG: u64 = u64::from_be_bytes(*b"MCCALIB1");

/// The response levels of the grid: Bernoulli at prevalence about 0.54, 0.22
/// and 0.10, Poisson at mean count about 1.65 and 0.34, and Gaussian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Response {
    Bern54,
    Bern22,
    Bern10,
    Pois165,
    Pois034,
    Gauss,
}

impl Response {
    /// The levels in canonical order.
    const ALL: [Response; 6] = [
        Response::Bern54,
        Response::Bern22,
        Response::Bern10,
        Response::Pois165,
        Response::Pois034,
        Response::Gauss,
    ];

    /// The level's part of a cell's name.
    pub(crate) fn tag(self) -> &'static str {
        match self {
            Response::Bern54 => "bern54",
            Response::Bern22 => "bern22",
            Response::Bern10 => "bern10",
            Response::Pois165 => "pois165",
            Response::Pois034 => "pois034",
            Response::Gauss => "gauss",
        }
    }

    /// The intercept `β0`, which sets the prevalence or the mean count.
    pub(crate) fn beta0(self) -> f64 {
        match self {
            Response::Bern54 | Response::Pois165 | Response::Gauss => 0.1,
            Response::Bern22 | Response::Pois034 => -1.5,
            Response::Bern10 => -2.5,
        }
    }
}

/// The cluster counts `K`.
const KS: [usize; 6] = [10, 20, 30, 50, 100, 200];

/// The maximum visits `M`.
const MS: [usize; 3] = [3, 9, 17];

/// One cell of the grid.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cell {
    /// Clusters (subjects), `K`.
    pub(crate) k: usize,
    /// Maximum visits per subject, `M`.
    pub(crate) m: usize,
    /// The response level.
    pub(crate) response: Response,
    /// `(1 + time | id)` when true, `(1 | id)` when false.
    pub(crate) slope: bool,
}

impl Cell {
    /// `k<K>_m<M>_<response>_<rdi|rdis>`; the cell's seed is keyed on it.
    pub(crate) fn name(&self) -> String {
        let part = if self.slope { "rdis" } else { "rdi" };
        format!("k{}_m{}_{}_{part}", self.k, self.m, self.response.tag())
    }
}

/// The 216 cells in canonical order: `(1 | id)` before `(1 + time | id)`, then
/// `K`, `M` and the response level. The order is the report's only; a cell's
/// data depend on its name alone.
pub(crate) const CELLS: [Cell; 216] = {
    let mut cells = [Cell {
        k: 0,
        m: 0,
        response: Response::Gauss,
        slope: false,
    }; 216];
    let mut i = 0;
    while i < 216 {
        let j = i % 108;
        cells[i] = Cell {
            k: KS[j / 18],
            m: MS[j / 6 % 3],
            response: Response::ALL[j % 6],
            slope: i >= 108,
        };
        i += 1;
    }
    cells
};

/// A cell's seed: `20261100 + fnv1a64(name)`, wrapping, so a cell added later
/// moves no other cell's data.
pub(crate) fn cell_seed(name: &str) -> u64 {
    20_261_100u64.wrapping_add(fnv1a64(name))
}

/// FNV-1a, 64-bit.
fn fnv1a64(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The generating truth of a cell.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Truth {
    /// `(β0, β1, β2, β3)` of `y_full`: intercept, `time`, `group`,
    /// `time:group`.
    pub(crate) beta: [f64; 4],
    /// The random intercept's SD.
    pub(crate) tau0: f64,
    /// The random slope's SD (zero on `(1 | id)` cells).
    pub(crate) tau1: f64,
    /// The intercept-slope correlation (zero on `(1 | id)` cells).
    pub(crate) rho: f64,
    /// The Gaussian residual SD (unused by the other responses).
    pub(crate) sigma: f64,
}

impl Truth {
    /// The truth of `cell`.
    pub(crate) fn of(cell: &Cell) -> Truth {
        let (tau1, rho) = if cell.slope { (0.4, 0.2) } else { (0.0, 0.0) };
        Truth {
            beta: [cell.response.beta0(), 0.3, -0.2, 0.1],
            tau0: 0.8,
            tau1,
            rho,
            sigma: 1.0,
        }
    }
}

/// Dataset `r` of `cell`: the columns `y_full`, `y_0`, `y_00`, `y_b0`, `y_0f`,
/// `time`, `group`, `f3`, `id`, and `off0` to `off3` (the truth's `β_j·x_j`).
pub(crate) fn generate(cell: &Cell, r: u64) -> Table {
    generate_with(cell, r, &Truth::of(cell)).0
}

/// [`generate`] from a given truth, with the same draws; also returns each
/// subject's `(b0, b1)` (`b1 = 0` on `(1 | id)` cells, where no slope is
/// drawn).
pub(crate) fn generate_with(cell: &Cell, r: u64, truth: &Truth) -> (Table, Vec<[f64; 2]>) {
    let mut rng = CommonStatsRng::new_tagged(cell_seed(&cell.name()), r, MC_STREAM_TAG);
    let normal = Normal::new(0.0, 1.0).expect("the standard normal");
    let group_draw = Bernoulli::new(0.4).expect("a probability");
    let (m, beta) = (cell.m, truth.beta);
    // Per response, in column order: its coefficients and whether it carries
    // the random effects.
    let responses: [([f64; 4], bool); 5] = [
        (beta, true),
        ([beta[0], beta[1], beta[2], 0.0], true),
        ([beta[0], beta[1], 0.0, 0.0], true),
        (beta, false),
        ([beta[0], beta[1], beta[2], 0.0], false),
    ];
    let rot = (1.0 - truth.rho * truth.rho).sqrt();
    let (mut id, mut f3) = (Vec::new(), Vec::new());
    let (mut time, mut group) = (Vec::new(), Vec::new());
    let mut ys: [Vec<f64>; 5] = Default::default();
    let mut offsets: [Vec<f64>; 4] = Default::default();
    let mut b = Vec::with_capacity(cell.k);
    for s in 0..cell.k {
        let visits = 1 + rng.bounded(m as u32) as usize;
        let mut slots: Vec<usize> = (0..m).collect();
        for j in 0..visits {
            let pick = j + rng.bounded((m - j) as u32) as usize;
            slots.swap(j, pick);
        }
        slots.truncate(visits);
        slots.sort_unstable();
        let g = group_draw.sample(&mut rng) as f64;
        let level = rng.bounded(3);
        let z1 = normal.sample(&mut rng);
        let b1 = if cell.slope {
            let z2 = normal.sample(&mut rng);
            truth.tau1 * (truth.rho * z1 + rot * z2)
        } else {
            0.0
        };
        let b0 = truth.tau0 * z1;
        b.push([b0, b1]);
        for &slot in &slots {
            let t = slot as f64 / (m - 1) as f64;
            id.push(s as u32);
            f3.push(level);
            time.push(t);
            group.push(g);
            let terms = [beta[0], beta[1] * t, beta[2] * g, beta[3] * t * g];
            for (o, term) in offsets.iter_mut().zip(terms) {
                o.push(term);
            }
            for ((c, random), y) in responses.iter().zip(ys.iter_mut()) {
                let xb = c[0] + c[1] * t + c[2] * g + c[3] * t * g;
                let eta = if *random { xb + b0 + b1 * t } else { xb };
                y.push(draw(cell.response, eta, truth.sigma, &normal, &mut rng));
            }
        }
    }
    let n = id.len();
    let [y_full, y_0, y_00, y_b0, y_0f] = ys;
    let mut columns = vec![
        ("y_full".to_string(), Column::Numeric(y_full)),
        ("y_0".to_string(), Column::Numeric(y_0)),
        ("y_00".to_string(), Column::Numeric(y_00)),
        ("y_b0".to_string(), Column::Numeric(y_b0)),
        ("y_0f".to_string(), Column::Numeric(y_0f)),
        ("time".to_string(), Column::Numeric(time)),
        ("group".to_string(), Column::Numeric(group)),
        (
            "f3".to_string(),
            Column::Factor {
                levels: vec!["a".into(), "b".into(), "c".into()],
                codes: f3,
            },
        ),
        (
            "id".to_string(),
            Column::Factor {
                levels: (0..cell.k).map(|s| format!("s{s:03}")).collect(),
                codes: id,
            },
        ),
    ];
    for (j, o) in offsets.into_iter().enumerate() {
        columns.push((format!("off{j}"), Column::Numeric(o)));
    }
    (Table { columns, n }, b)
}

/// One draw of a response with linear predictor `eta`.
fn draw(
    response: Response,
    eta: f64,
    sigma: f64,
    normal: &Normal,
    rng: &mut CommonStatsRng,
) -> f64 {
    match response {
        Response::Bern54 | Response::Bern22 | Response::Bern10 => {
            let p = 1.0 / (1.0 + libm::exp(-eta));
            Bernoulli::new(p).expect("a probability").sample(rng) as f64
        }
        Response::Pois165 | Response::Pois034 => Poisson::new(libm::exp(eta))
            .expect("a positive mean")
            .sample(rng) as f64,
        Response::Gauss => eta + sigma * normal.sample(rng),
    }
}

/// The calibration tier's bands and its frozen gated set: rejection rates
/// under a true null and interval coverage at the truth, per cell of
/// [`super::CELLS`]. Every value is fixed before the measurement, from theory
/// and an independent pilot run (`R = 1000` per cell, its own seeds and
/// stream tag, so no pilot dataset is a measurement dataset). A rate `q̂` over
/// `R_kept` datasets passes against its nominal `q0` (α for a rejection rate;
/// for a coverage, the miss rate `1 − q̂` against `1 − level`) when
/// `|q̂ − q0| ≤ δ·q0 + k·√(q0(1 − q0)/R_kept)`, at α = 0.01, 0.05, 0.10 and
/// level 0.90, 0.95, 0.99; a one-sided gate uses one side. The run is
/// deterministic, so a noise failure can appear only at the measurement,
/// where it is investigated (a bug first), never re-rolled with another seed.
#[cfg_attr(not(feature = "montecarlo-tests"), allow(dead_code))]
pub(crate) mod bands {
    /// `mc_r`: datasets per cell. With every dataset kept the binomial SE is
    /// 0.0049 at α = 0.05 and at coverage 0.95 (0.0022 at 0.01, 0.0067 at
    /// 0.10). `R = 5000` would narrow the noise band at α = 0.05 to ±0.014
    /// for 2.5 times the run.
    pub(crate) const MC_R: usize = 2000;
    /// `mc_level_k`: the noise multiple `k` on a rejection rate, so the noise
    /// band is ±0.022 at α = 0.05 and `R = 2000` (±0.010 at 0.01, ±0.030 at
    /// 0.10). From exact binomial tails a check calibrated exactly fails with
    /// probability 2.3e-5, 1.3e-5 and 8.3e-6 at the three α (two-sided,
    /// δ = 0; the normal value is 6.8e-6), so about 0.065 expected noise
    /// failures over the 4416 gated checks of [`GATED`] (4446 when frozen,
    /// before two Deviations) if all had δ = 0, and 0.014 with the declared δ
    /// (the Satterthwaite rows keep δ = 0 and carry nearly all of it); at
    /// `k = 4` about 0.55, at `k = 3` about 11 (δ = 0). At δ = 0 the lower
    /// bound at α = 0.01 is 0.000, so a conservative error is caught only at
    /// 0.05 and 0.10.
    pub(crate) const MC_LEVEL_K: f64 = 4.5;
    /// `mc_cover_k`: the same multiple on a coverage's miss rate, so the same
    /// widths around coverage 0.99, 0.95 and 0.90.
    pub(crate) const MC_COVER_K: f64 = 4.5;
    /// `mc_asym_rel`: the margin δ of the methods calibrated only in the limit
    /// (GLMM Wald and LRT, the two mixtures, the Gaussian ML LRT), 0.0025,
    /// 0.0125 and 0.025 at α = 0.01, 0.05 and 0.10 before noise. Without it
    /// the band tests exact calibration, and a finite-sample error of fixed
    /// size grows in SE units as `R` grows (the pilot's GLMM gated cells had 4
    /// Wald cells at 4.2 to 5.1 SE and 2 LRT cells at 4.1 to 4.4 SE at
    /// `R = 1000`), so the gate would be red at the measurement or green only
    /// by re-selecting cells on the data it gates. The Satterthwaite rows,
    /// built to be near nominal at small samples, use δ = 0.
    pub(crate) const MC_ASYM_REL: f64 = 0.25;
    /// `mc_fail_max`: a gated (cell, quantity) fails when more than this
    /// fraction of its datasets lose it to a fit that did not converge or to a
    /// value that is not finite on a converged fit. A dataset whose
    /// fixed-effects design is aliased, decided from the lowered design by
    /// `fit_cold`'s own aliasing test and never from a NaN row or from the
    /// fit's aliased mask (`group` constant, or `time:group` aliased because
    /// one group's rows share one time), is left out of the cap and of
    /// `R_kept` for every quantity of that fit, and counted and printed apart:
    /// the dropped column moves the other coefficients' targets, and since the
    /// design is drawn independently of every response, leaving it out changes
    /// no null distribution and no coverage target. Expected (arithmetic, from
    /// simulated designs): 2.6%, 1.1% and 0.9% of datasets at `K = 10` and
    /// `M = 3, 9, 17`, 0.04% or less at `K = 20`, none at `K ≥ 30`. A level of
    /// `f3` absent (5.2% of datasets at `K = 10`, 0.09% at `K = 20`) is not
    /// such a case: its zero column is kept as aliased and the `f3` row has
    /// 1 df. A reported cell prints the failed fraction beside its rate.
    /// Pilot: no Gaussian LMM fit failed to converge (its losses were aliased
    /// designs); GLMM losses (aliased designs and failed fits together) at
    /// most 0.7% on the GLMM gated cells, up to 19% on the `K = 10` mixture
    /// candidates, which are reported. At `R = 2000` a quantity whose fits
    /// fail on 0.7% of datasets exceeds the cap with probability 0.05, on 0.9%
    /// with 0.27.
    pub(crate) const MC_FAIL_MAX: f64 = 0.01;
    /// `mc_aliased_max`: a gated (cell, quantity) also fails when more than
    /// this fraction of its datasets are aliased by design, about four times
    /// the largest expected design rate (2.6% at `K = 10`, `M = 3`), so a
    /// generator change that inflates aliasing, and so shrinks `R_kept` and
    /// widens the band, fails instead of passing on fewer datasets.
    pub(crate) const MC_ALIASED_MAX: f64 = 0.10;

    /// The side of the band a gate asserts.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) enum Side {
        /// `|q̂ − q0| ≤ δ·q0 + k·SE`.
        Two,
        /// Not anti-conservative: `q̂ − q0 ≤ δ·q0 + k·SE`.
        Upper,
        /// Not conservative: `q0 − q̂ ≤ δ·q0 + k·SE`.
        Lower,
    }

    /// One family of the gated set: each of its quantities is gated on each
    /// of its cells.
    #[derive(Debug)]
    pub(crate) struct Gate {
        /// The family's name.
        pub(crate) family: &'static str,
        /// The side of the band.
        pub(crate) side: Side,
        /// δ: [`MC_ASYM_REL`] or 0.
        pub(crate) delta: f64,
        /// The only α gated, when not all three.
        pub(crate) alpha: Option<f64>,
        /// The quantities, by the names the calibration tier's driver records
        /// them under.
        pub(crate) quantities: &'static [&'static str],
        /// The cells, by [`super::Cell::name`].
        pub(crate) cells: &'static [&'static str],
    }

    /// The Satterthwaite quantities but the `f3` F row, which has its own
    /// families.
    const SATT: [&str; 8] = [
        "satt_t_tg",
        "satt_f_group",
        "satt_f_tg00",
        "satt_f_joint",
        "satt_ci_b0",
        "satt_ci_b1",
        "satt_ci_b2",
        "satt_ci_b3",
    ];

    /// The GLMM Wald and LRT quantities. The `f3` rows (`wald_f3`, `lrt_f3`)
    /// are reported, as the unpiloted GLM rows are: they were not piloted, and
    /// an out-of-sample run (datasets `r ≥ 100000`) lost 0.75% to 1.5% of the
    /// `f3` fits to separation on gated sparse binary cells (an `f3` level
    /// with no events), near or above [`MC_FAIL_MAX`]. The Gaussian `f3` row
    /// (`satt_f_f3`) follows its family's cells.
    const GLMM: [&str; 13] = [
        "wald_tg",
        "wald3_group",
        "wald2_group",
        "wald_tg00",
        "wald_ci_b0",
        "wald_ci_b1",
        "wald_ci_b2",
        "wald_ci_b3",
        "lrt_tg",
        "lrt3_group",
        "lrt2_group",
        "lrt_tg00",
        "lrt_2df",
    ];

    /// The gated set, frozen before the measurement and never re-selected on
    /// it. Every (cell, quantity) not listed is reported: power, bias and RMSE
    /// everywhere, the AGQ arm, the profile-interval coverage, the GLM rows,
    /// every cell outside these regimes, the failed-dataset fractions. Built
    /// from theory, then screened against the pilot: each candidate (cell,
    /// quantity) was checked against its declared band (its δ, `k = 4.5`, the
    /// pilot's `R_kept`) and against [`MC_FAIL_MAX`] (failed fits only), and a
    /// candidate the pilot fails is reported. The band moved none; the cap
    /// moved 21 mixture candidates, named under their families. Two Deviations
    /// approved after the measurement (2026-10-13) moved the `f3` F row on the
    /// random-slope Gaussian cells with `K ≤ 30` (Satterthwaite's F on a
    /// 3-level between-cluster factor is anti-conservative there, as
    /// lmerTest's is on the same data) and the boundary test on
    /// `k20_m9_bern22_rdi` (21 of 2000 datasets separated) to reported; the
    /// set holds 4416 checks (0.0143 expected noise failures with the declared
    /// δ).
    pub(crate) const GATED: [Gate; 9] = [
        // Satterthwaite t and F rows and t intervals on the Gaussian cells
        // with K ≥ 20: two-sided, δ = 0. The t and F references with
        // Satterthwaite's df are built to be near nominal at small samples
        // (Kuznetsova, Brockhoff & Christensen 2017; Luke 2017). Pilot: no
        // check beyond 4 SE, the largest errors conservative and at K = 10.
        // At K ≥ 20 this gate cannot tell Satterthwaite's t from z on these
        // designs (a normal reference gives 0.066 at α = 0.05 for df 18,
        // inside the band); the df themselves are checked against lmerTest in
        // the oracle tier.
        Gate {
            family: "satt_two",
            side: Side::Two,
            delta: 0.0,
            alpha: None,
            quantities: &SATT,
            cells: &[
                "k20_m3_gauss_rdi",
                "k20_m9_gauss_rdi",
                "k20_m17_gauss_rdi",
                "k30_m3_gauss_rdi",
                "k30_m9_gauss_rdi",
                "k30_m17_gauss_rdi",
                "k50_m3_gauss_rdi",
                "k50_m9_gauss_rdi",
                "k50_m17_gauss_rdi",
                "k100_m3_gauss_rdi",
                "k100_m9_gauss_rdi",
                "k100_m17_gauss_rdi",
                "k200_m3_gauss_rdi",
                "k200_m9_gauss_rdi",
                "k200_m17_gauss_rdi",
                "k20_m3_gauss_rdis",
                "k20_m9_gauss_rdis",
                "k20_m17_gauss_rdis",
                "k30_m3_gauss_rdis",
                "k30_m9_gauss_rdis",
                "k30_m17_gauss_rdis",
                "k50_m3_gauss_rdis",
                "k50_m9_gauss_rdis",
                "k50_m17_gauss_rdis",
                "k100_m3_gauss_rdis",
                "k100_m9_gauss_rdis",
                "k100_m17_gauss_rdis",
                "k200_m3_gauss_rdis",
                "k200_m9_gauss_rdis",
                "k200_m17_gauss_rdis",
            ],
        },
        // The Satterthwaite F of the null factor `f3` (2 df, between
        // clusters) on the satt_two cells, two-sided, δ = 0, less the
        // random-slope cells with K ≤ 30, which are reported (a Deviation
        // approved 2026-10-13, after the measurement): on `(1 + time | id)` at
        // K ≤ 30 the row is anti-conservative (0.059 to 0.098 at α = 0.05,
        // R = 2000; five cells outside the band), and lmerTest's Satterthwaite
        // F gives the same rates on the same datasets, so the error is the
        // method's at few clusters (Kenward-Roger: conservative at K = 10,
        // slightly above nominal at K = 30). The whole regime is reported
        // (K = 10, 20, 30, every M), not only the cells that missed.
        Gate {
            family: "satt_two_f3",
            side: Side::Two,
            delta: 0.0,
            alpha: None,
            quantities: &["satt_f_f3"],
            cells: &[
                "k20_m3_gauss_rdi",
                "k20_m9_gauss_rdi",
                "k20_m17_gauss_rdi",
                "k30_m3_gauss_rdi",
                "k30_m9_gauss_rdi",
                "k30_m17_gauss_rdi",
                "k50_m3_gauss_rdi",
                "k50_m9_gauss_rdi",
                "k50_m17_gauss_rdi",
                "k100_m3_gauss_rdi",
                "k100_m9_gauss_rdi",
                "k100_m17_gauss_rdi",
                "k200_m3_gauss_rdi",
                "k200_m9_gauss_rdi",
                "k200_m17_gauss_rdi",
                "k50_m3_gauss_rdis",
                "k50_m9_gauss_rdis",
                "k50_m17_gauss_rdis",
                "k100_m3_gauss_rdis",
                "k100_m9_gauss_rdis",
                "k100_m17_gauss_rdis",
                "k200_m3_gauss_rdis",
                "k200_m9_gauss_rdis",
                "k200_m17_gauss_rdis",
            ],
        },
        // satt_two's quantities at K = 10, one-sided: not anti-conservative (a
        // rate at most α plus the band, a coverage at least the level minus
        // it). The pilot's largest Satterthwaite errors are conservative and
        // here (coverage 0.936 at 0.90 with random slopes; the Type III
        // time:group row 0.066 at α = 0.10). This side sees t against z: a
        // normal reference gives 0.086 at df 8 against the upper bound 0.072
        // at α = 0.05.
        Gate {
            family: "satt_upper",
            side: Side::Upper,
            delta: 0.0,
            alpha: None,
            quantities: &SATT,
            cells: &[
                "k10_m3_gauss_rdi",
                "k10_m9_gauss_rdi",
                "k10_m17_gauss_rdi",
                "k10_m3_gauss_rdis",
                "k10_m9_gauss_rdis",
                "k10_m17_gauss_rdis",
            ],
        },
        // The `f3` F row at K = 10, one-sided, on the `(1 | id)` cells only:
        // the random-slope cells are reported with satt_two_f3's regime.
        Gate {
            family: "satt_upper_f3",
            side: Side::Upper,
            delta: 0.0,
            alpha: None,
            quantities: &["satt_f_f3"],
            cells: &["k10_m3_gauss_rdi", "k10_m9_gauss_rdi", "k10_m17_gauss_rdi"],
        },
        // GLMM Wald (z rows, the Type II Wald chi-square, z intervals) and
        // GLMM LRT (per term, 2 df), Laplace: two-sided with δ on the 78
        // cells with K ≥ 50 and at least 0.49 minority events per cluster
        // (Bernoulli min(Σy, N − Σy)/K, Poisson Σy/K, the pilot's means). Both
        // are asymptotic in the number of clusters (Bolker et al. 2009;
        // Pinheiro & Bates 2000, 2.4.2) and rest on the Laplace
        // approximation, whose error is of the order of one over the
        // information per cluster and does not shrink as K grows with
        // clusters held small (Breslow & Lin 1995; Joe 2008): hence K with an
        // information floor. The floor leaves out the 12 Bernoulli cells with
        // K ≥ 50, M = 3 and prevalence 0.22 or 0.10. Pilot, declared band:
        // closest k50_m17_pois165_rdi wald_ci_b0, miss rate 0.026 at level
        // 0.99, 0.96 of its band (5.1 SE: at R = 2000 the same error would
        // fail); every lost fraction at or below 0.7%.
        Gate {
            family: "glmm",
            side: Side::Two,
            delta: MC_ASYM_REL,
            alpha: None,
            quantities: &GLMM,
            cells: &[
                "k50_m3_bern54_rdi",
                "k50_m3_pois165_rdi",
                "k50_m3_pois034_rdi",
                "k50_m9_bern54_rdi",
                "k50_m9_bern22_rdi",
                "k50_m9_bern10_rdi",
                "k50_m9_pois165_rdi",
                "k50_m9_pois034_rdi",
                "k50_m17_bern54_rdi",
                "k50_m17_bern22_rdi",
                "k50_m17_bern10_rdi",
                "k50_m17_pois165_rdi",
                "k50_m17_pois034_rdi",
                "k100_m3_bern54_rdi",
                "k100_m3_pois165_rdi",
                "k100_m3_pois034_rdi",
                "k100_m9_bern54_rdi",
                "k100_m9_bern22_rdi",
                "k100_m9_bern10_rdi",
                "k100_m9_pois165_rdi",
                "k100_m9_pois034_rdi",
                "k100_m17_bern54_rdi",
                "k100_m17_bern22_rdi",
                "k100_m17_bern10_rdi",
                "k100_m17_pois165_rdi",
                "k100_m17_pois034_rdi",
                "k200_m3_bern54_rdi",
                "k200_m3_pois165_rdi",
                "k200_m3_pois034_rdi",
                "k200_m9_bern54_rdi",
                "k200_m9_bern22_rdi",
                "k200_m9_bern10_rdi",
                "k200_m9_pois165_rdi",
                "k200_m9_pois034_rdi",
                "k200_m17_bern54_rdi",
                "k200_m17_bern22_rdi",
                "k200_m17_bern10_rdi",
                "k200_m17_pois165_rdi",
                "k200_m17_pois034_rdi",
                "k50_m3_bern54_rdis",
                "k50_m3_pois165_rdis",
                "k50_m3_pois034_rdis",
                "k50_m9_bern54_rdis",
                "k50_m9_bern22_rdis",
                "k50_m9_bern10_rdis",
                "k50_m9_pois165_rdis",
                "k50_m9_pois034_rdis",
                "k50_m17_bern54_rdis",
                "k50_m17_bern22_rdis",
                "k50_m17_bern10_rdis",
                "k50_m17_pois165_rdis",
                "k50_m17_pois034_rdis",
                "k100_m3_bern54_rdis",
                "k100_m3_pois165_rdis",
                "k100_m3_pois034_rdis",
                "k100_m9_bern54_rdis",
                "k100_m9_bern22_rdis",
                "k100_m9_bern10_rdis",
                "k100_m9_pois165_rdis",
                "k100_m9_pois034_rdis",
                "k100_m17_bern54_rdis",
                "k100_m17_bern22_rdis",
                "k100_m17_bern10_rdis",
                "k100_m17_pois165_rdis",
                "k100_m17_pois034_rdis",
                "k200_m3_bern54_rdis",
                "k200_m3_pois165_rdis",
                "k200_m3_pois034_rdis",
                "k200_m9_bern54_rdis",
                "k200_m9_bern22_rdis",
                "k200_m9_bern10_rdis",
                "k200_m9_pois165_rdis",
                "k200_m9_pois034_rdis",
                "k200_m17_bern54_rdis",
                "k200_m17_bern22_rdis",
                "k200_m17_bern10_rdis",
                "k200_m17_pois165_rdis",
                "k200_m17_pois034_rdis",
            ],
        },
        // Gaussian ML LRT for the fixed part (2 df): two-sided with δ at
        // K ≥ 30. Anti-conservative at few clusters (Pinheiro & Bates 2000,
        // 2.4.2); pilot: 0.074 to 0.099 at α = 0.05 at K = 10, up to 0.081 at
        // K = 20 with slopes, every cell beyond 4 SE at K ≤ 20.
        Gate {
            family: "ml_lrt",
            side: Side::Two,
            delta: MC_ASYM_REL,
            alpha: None,
            quantities: &["ml_lrt_2df"],
            cells: &[
                "k30_m3_gauss_rdi",
                "k30_m9_gauss_rdi",
                "k30_m17_gauss_rdi",
                "k50_m3_gauss_rdi",
                "k50_m9_gauss_rdi",
                "k50_m17_gauss_rdi",
                "k100_m3_gauss_rdi",
                "k100_m9_gauss_rdi",
                "k100_m17_gauss_rdi",
                "k200_m3_gauss_rdi",
                "k200_m9_gauss_rdi",
                "k200_m17_gauss_rdi",
                "k30_m3_gauss_rdis",
                "k30_m9_gauss_rdis",
                "k30_m17_gauss_rdis",
                "k50_m3_gauss_rdis",
                "k50_m9_gauss_rdis",
                "k50_m17_gauss_rdis",
                "k100_m3_gauss_rdis",
                "k100_m9_gauss_rdis",
                "k100_m17_gauss_rdis",
                "k200_m3_gauss_rdis",
                "k200_m9_gauss_rdis",
                "k200_m17_gauss_rdis",
            ],
        },
        // The random-slope variance test, 50:50 mixture of chi-square 1 and 2
        // (labelled conservative): one-sided upper with δ (not
        // anti-conservative; Self & Liang 1987, Stram & Lee 1994) on the rdi
        // cells, Gaussian all, Poisson except mean 0.34 at M = 3, Bernoulli at
        // M ≥ 9 except prevalence 0.10 at M = 9. Reported on the cells of
        // least information, where the pilot's rates rise with K (prevalence
        // 0.10, M = 9: 0.011 at K = 10 to 0.074 at K = 200 at α = 0.05;
        // Laplace on sparse binary data with two observations per cluster).
        // Reported by MC_FAIL_MAX, with the pilot's lost fraction:
        // k10_m3_pois165_rdi 8.0%, k10_m9_bern54_rdi 4.7%, k10_m9_bern22_rdi
        // 13.2%, k10_m9_pois165_rdi 1.1%, k10_m9_pois034_rdi 10.1%,
        // k10_m17_bern54_rdi 2.1%, k10_m17_bern22_rdi 3.6%, k10_m17_bern10_rdi
        // 18.0%, k10_m17_pois034_rdi 2.4%, k20_m9_bern22_rdi 1.5%,
        // k20_m9_pois034_rdi 1.1%, k20_m17_bern10_rdi 2.3%.
        Gate {
            family: "mix_slope",
            side: Side::Upper,
            delta: MC_ASYM_REL,
            alpha: None,
            quantities: &["mix_slope"],
            cells: &[
                "k10_m3_gauss_rdi",
                "k10_m9_gauss_rdi",
                "k10_m17_pois165_rdi",
                "k10_m17_gauss_rdi",
                "k20_m3_pois165_rdi",
                "k20_m3_gauss_rdi",
                "k20_m9_bern54_rdi",
                "k20_m9_pois165_rdi",
                "k20_m9_gauss_rdi",
                "k20_m17_bern54_rdi",
                "k20_m17_bern22_rdi",
                "k20_m17_pois165_rdi",
                "k20_m17_pois034_rdi",
                "k20_m17_gauss_rdi",
                "k30_m3_pois165_rdi",
                "k30_m3_gauss_rdi",
                "k30_m9_bern54_rdi",
                "k30_m9_bern22_rdi",
                "k30_m9_pois165_rdi",
                "k30_m9_pois034_rdi",
                "k30_m9_gauss_rdi",
                "k30_m17_bern54_rdi",
                "k30_m17_bern22_rdi",
                "k30_m17_bern10_rdi",
                "k30_m17_pois165_rdi",
                "k30_m17_pois034_rdi",
                "k30_m17_gauss_rdi",
                "k50_m3_pois165_rdi",
                "k50_m3_gauss_rdi",
                "k50_m9_bern54_rdi",
                "k50_m9_bern22_rdi",
                "k50_m9_pois165_rdi",
                "k50_m9_pois034_rdi",
                "k50_m9_gauss_rdi",
                "k50_m17_bern54_rdi",
                "k50_m17_bern22_rdi",
                "k50_m17_bern10_rdi",
                "k50_m17_pois165_rdi",
                "k50_m17_pois034_rdi",
                "k50_m17_gauss_rdi",
                "k100_m3_pois165_rdi",
                "k100_m3_gauss_rdi",
                "k100_m9_bern54_rdi",
                "k100_m9_bern22_rdi",
                "k100_m9_pois165_rdi",
                "k100_m9_pois034_rdi",
                "k100_m9_gauss_rdi",
                "k100_m17_bern54_rdi",
                "k100_m17_bern22_rdi",
                "k100_m17_bern10_rdi",
                "k100_m17_pois165_rdi",
                "k100_m17_pois034_rdi",
                "k100_m17_gauss_rdi",
                "k200_m3_pois165_rdi",
                "k200_m3_gauss_rdi",
                "k200_m9_bern54_rdi",
                "k200_m9_bern22_rdi",
                "k200_m9_pois165_rdi",
                "k200_m9_pois034_rdi",
                "k200_m9_gauss_rdi",
                "k200_m17_bern54_rdi",
                "k200_m17_bern22_rdi",
                "k200_m17_bern10_rdi",
                "k200_m17_pois165_rdi",
                "k200_m17_pois034_rdi",
                "k200_m17_gauss_rdi",
            ],
        },
        // The random-intercept variance test, GLMM against GLM, 50:50 mixture
        // of chi-square 0 and 1: one-sided upper with δ on the same rule's
        // GLMM cells (pilot, mean 0.34 at M = 3: 0.003 at K = 10 to 0.070 at
        // K = 200). Reported by MC_FAIL_MAX, with the pilot's lost fraction:
        // k10_m3_pois165_rdi 4.9%, k10_m9_bern54_rdi 3.9%, k10_m9_bern22_rdi
        // 13.2%, k10_m9_pois034_rdi 11.5%, k10_m17_bern54_rdi 1.5%,
        // k10_m17_bern22_rdi 3.6%, k10_m17_bern10_rdi 19.2%,
        // k10_m17_pois034_rdi 2.2%, k20_m17_bern10_rdi 1.8%. Reported after
        // the measurement (a Deviation approved 2026-10-13):
        // k20_m9_bern22_rdi, whose boundary-test fits failed on 21 of 2000
        // datasets (1.05%, above MC_FAIL_MAX), every one separated (no
        // maximum-likelihood estimate); its rate passes.
        Gate {
            family: "mix_intercept",
            side: Side::Upper,
            delta: MC_ASYM_REL,
            alpha: None,
            quantities: &["mix_intercept"],
            cells: &[
                "k10_m9_pois165_rdi",
                "k10_m17_pois165_rdi",
                "k20_m3_pois165_rdi",
                "k20_m9_bern54_rdi",
                "k20_m9_pois165_rdi",
                "k20_m9_pois034_rdi",
                "k20_m17_bern54_rdi",
                "k20_m17_bern22_rdi",
                "k20_m17_pois165_rdi",
                "k20_m17_pois034_rdi",
                "k30_m3_pois165_rdi",
                "k30_m9_bern54_rdi",
                "k30_m9_bern22_rdi",
                "k30_m9_pois165_rdi",
                "k30_m9_pois034_rdi",
                "k30_m17_bern54_rdi",
                "k30_m17_bern22_rdi",
                "k30_m17_bern10_rdi",
                "k30_m17_pois165_rdi",
                "k30_m17_pois034_rdi",
                "k50_m3_pois165_rdi",
                "k50_m9_bern54_rdi",
                "k50_m9_bern22_rdi",
                "k50_m9_pois165_rdi",
                "k50_m9_pois034_rdi",
                "k50_m17_bern54_rdi",
                "k50_m17_bern22_rdi",
                "k50_m17_bern10_rdi",
                "k50_m17_pois165_rdi",
                "k50_m17_pois034_rdi",
                "k100_m3_pois165_rdi",
                "k100_m9_bern54_rdi",
                "k100_m9_bern22_rdi",
                "k100_m9_pois165_rdi",
                "k100_m9_pois034_rdi",
                "k100_m17_bern54_rdi",
                "k100_m17_bern22_rdi",
                "k100_m17_bern10_rdi",
                "k100_m17_pois165_rdi",
                "k100_m17_pois034_rdi",
                "k200_m3_pois165_rdi",
                "k200_m9_bern54_rdi",
                "k200_m9_bern22_rdi",
                "k200_m9_pois165_rdi",
                "k200_m9_pois034_rdi",
                "k200_m17_bern54_rdi",
                "k200_m17_bern22_rdi",
                "k200_m17_bern10_rdi",
                "k200_m17_pois165_rdi",
                "k200_m17_pois034_rdi",
            ],
        },
        // A lower bound, δ = 0, at α = 0.10 only, on the Gaussian random-slope
        // mixture with K ≥ 50, where the 50:50 mixture is asymptotically exact
        // (Stram & Lee 1994) and the pilot near nominal (0.079 to 0.107 at
        // α = 0.10): a chi-square 2 reference in place of the mixture (about
        // 0.066) fails it (bound 0.070). At α = 0.05 the same bound would not
        // catch that (about 0.032 against 0.028) and would carry most of this
        // side's false-miss risk, so it is not applied there. The GLMM
        // mixtures get no lower bound: they are mostly conservative in the
        // pilot even at high information.
        Gate {
            family: "mix_slope_lower",
            side: Side::Lower,
            delta: 0.0,
            alpha: Some(0.10),
            quantities: &["mix_slope"],
            cells: &[
                "k50_m3_gauss_rdi",
                "k50_m9_gauss_rdi",
                "k50_m17_gauss_rdi",
                "k100_m3_gauss_rdi",
                "k100_m9_gauss_rdi",
                "k100_m17_gauss_rdi",
                "k200_m3_gauss_rdi",
                "k200_m9_gauss_rdi",
                "k200_m17_gauss_rdi",
            ],
        },
    ];

    /// The gates of `quantity` on `cell` (by [`super::Cell::name`]), from
    /// [`GATED`]: none for a reported (cell, quantity); two for `mix_slope` on
    /// the Gaussian `(1 | id)` cells with `K ≥ 50` (the upper side with δ, and
    /// the lower bound at α = 0.10). Read by the calibration tier's gates.
    #[cfg(feature = "montecarlo-tests")]
    pub(crate) fn gates<'a>(
        cell: &'a str,
        quantity: &'a str,
    ) -> impl Iterator<Item = &'static Gate> + 'a {
        let gated: &'static [Gate] = &GATED;
        gated
            .iter()
            .filter(move |g| g.cells.contains(&cell) && g.quantities.contains(&quantity))
    }
}
